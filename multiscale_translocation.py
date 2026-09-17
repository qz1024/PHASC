#!/usr/bin/env python3
"""Multi-resolution intra-chromosomal translocation detection for mcool files.

Each Hi-C resolution is one biological observation scale. Candidates detected at
several resolutions are merged in genomic coordinates and reported as one consensus
event, avoiding duplicate scoring of the same biological event. Multiple PLM gamma
values remain available as an optional sensitivity analysis, but gamma is fixed to
1.0 by default and does not increase cross-resolution confidence.

Multi-chromosome mcool files are processed chromosome by chromosome.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

import cooler
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkit as nk
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm


CANDIDATE_COLUMNS = [
    "scale_id",
    "resolution",
    "gamma",
    "label",
    "start_bin",
    "end_bin",
    "start_bp",
    "end_bp",
    "length_bp",
    "main_start_bp",
    "main_end_bp",
    "distance_bp",
    "distance_percent",
    "severity",
]


def run_length_segments(labels: np.ndarray) -> pd.DataFrame:
    """Convert bin-level labels into half-open, contiguous segments."""
    if labels.size == 0:
        return pd.DataFrame(columns=["label", "start_bin", "end_bin", "length_bins"])

    changes = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.r_[0, changes]
    ends = np.r_[changes, labels.size]
    return pd.DataFrame(
        {
            "label": labels[starts],
            "start_bin": starts,
            "end_bin": ends,
            "length_bins": ends - starts,
        }
    )


def bridge_short_interruptions(
    labels: np.ndarray, max_interruption_bins: int, max_passes: int = 10
) -> np.ndarray:
    """Fill short A-B-A interruptions to reduce single-bin community noise."""
    result = np.asarray(labels, dtype=np.int64).copy()
    if max_interruption_bins <= 0:
        return result

    for _ in range(max_passes):
        segments = run_length_segments(result)
        changed = False
        for i in range(1, len(segments) - 1):
            row = segments.iloc[i]
            left = segments.iloc[i - 1]
            right = segments.iloc[i + 1]
            if (
                row["length_bins"] <= max_interruption_bins
                and left["label"] == right["label"]
                and row["label"] != left["label"]
            ):
                result[int(row["start_bin"]) : int(row["end_bin"])] = int(left["label"])
                changed = True
        if not changed:
            break
    return result


def detect_scale_candidates(
    labels: np.ndarray,
    resolution: int,
    gamma: float,
    chrom_size_bp: int,
    min_segment_bp: int,
    bridge_bp: int,
    base_penalty: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find secondary genomic occurrences of the same graph community."""
    min_bins = max(2, math.ceil(min_segment_bp / resolution))
    bridge_bins = max(0, math.floor(bridge_bp / resolution))
    clean_labels = bridge_short_interruptions(labels, bridge_bins)
    segments = run_length_segments(clean_labels)
    segments["is_filtered"] = segments["length_bins"] < min_bins
    segments["is_candidate"] = False
    segments["main_segment_index"] = pd.NA

    valid = segments.loc[~segments["is_filtered"]].copy()
    # Once short noisy pieces are removed, adjacent occurrences of the same label
    # belong to one block and must not be mistaken for a rearrangement.
    valid["block_id"] = valid["label"].ne(valid["label"].shift()).cumsum()
    for _, group in valid.groupby("label", sort=False):
        if group["block_id"].nunique() < 2:
            continue
        block_lengths = group.groupby("block_id")["length_bins"].sum()
        main_block = block_lengths.idxmax()
        main_rows = group.loc[group["block_id"] == main_block]
        main_index = main_rows["length_bins"].idxmax()
        secondary_indices = group.index[group["block_id"] != main_block]
        segments.loc[secondary_indices, "is_candidate"] = True
        segments.loc[secondary_indices, "main_segment_index"] = main_index

    scale_id = f"r{resolution}_g{gamma:g}"
    records = []
    for index, row in segments.loc[segments["is_candidate"]].iterrows():
        main = segments.loc[int(row["main_segment_index"])]
        start_bp = int(row["start_bin"]) * resolution
        end_bp = min(int(row["end_bin"]) * resolution, chrom_size_bp)
        main_start_bp = int(main["start_bin"]) * resolution
        main_end_bp = min(int(main["end_bin"]) * resolution, chrom_size_bp)
        center = (start_bp + end_bp) / 2.0
        main_center = (main_start_bp + main_end_bp) / 2.0
        distance_bp = abs(center - main_center)
        distance_percent = 100.0 * distance_bp / chrom_size_bp
        records.append(
            {
                "scale_id": scale_id,
                "resolution": resolution,
                "gamma": gamma,
                "label": int(row["label"]),
                "start_bin": int(row["start_bin"]),
                "end_bin": int(row["end_bin"]),
                "start_bp": start_bp,
                "end_bp": end_bp,
                "length_bp": end_bp - start_bp,
                "main_start_bp": main_start_bp,
                "main_end_bp": main_end_bp,
                "distance_bp": distance_bp,
                "distance_percent": distance_percent,
                "severity": base_penalty + distance_percent,
                "segment_index": int(index),
            }
        )

    candidates = pd.DataFrame(records)
    if candidates.empty:
        candidates = pd.DataFrame(columns=CANDIDATE_COLUMNS + ["segment_index"])
    return candidates, segments


def _balanced_sparse_matrix(clr: cooler.Cooler, chrom: str, balance: bool):
    try:
        return clr.matrix(balance=balance, sparse=True).fetch(chrom), balance
    except (ValueError, KeyError) as exc:
        if not balance:
            raise
        print(f"  warning: balanced matrix unavailable ({exc}); using raw counts")
        return clr.matrix(balance=False, sparse=True).fetch(chrom), False


def community_labels_for_scale(
    mcool_path: Path,
    chrom: str,
    resolution: int,
    gamma: float,
    edge_quantile: float,
    balance: bool,
    seed: int,
) -> tuple[np.ndarray, int, int]:
    """Build a sparse graph from strong non-zero contacts and run PLM."""
    clr = cooler.Cooler(f"{mcool_path}::/resolutions/{resolution}")
    matrix, _ = _balanced_sparse_matrix(clr, chrom, balance)
    coo = matrix.tocoo()
    valid = (
        (coo.row < coo.col)
        & np.isfinite(coo.data)
        & (coo.data > 0)
    )
    rows = coo.row[valid]
    cols = coo.col[valid]
    weights = np.asarray(coo.data[valid], dtype=float)
    if weights.size == 0:
        raise RuntimeError(f"no positive contacts at resolution {resolution}")

    threshold = float(np.quantile(weights, edge_quantile))
    keep = weights >= threshold
    graph = nk.graph.Graph(matrix.shape[0], weighted=True, directed=False)
    for row, col, weight in zip(rows[keep], cols[keep], weights[keep]):
        graph.addEdge(int(row), int(col), float(weight))

    if graph.numberOfEdges() == 0:
        raise RuntimeError(f"contact threshold removed every edge at resolution {resolution}")

    nk.setSeed(seed, False)
    algorithm = nk.community.PLM(graph, refine=True, gamma=gamma)
    algorithm.run()
    labels = np.asarray(algorithm.getPartition().getVector(), dtype=np.int64)
    return labels, graph.numberOfNodes(), graph.numberOfEdges()


def discover_resolutions(mcool_path: Path) -> list[int]:
    groups = cooler.fileops.list_coolers(str(mcool_path))
    resolutions = []
    for group in groups:
        match = re.fullmatch(r"/?resolutions/(\d+)", group)
        if match:
            resolutions.append(int(match.group(1)))
    if not resolutions:
        raise ValueError(f"no /resolutions/<number> groups found in {mcool_path}")
    return sorted(set(resolutions))


def evenly_spaced(values: Sequence[int], count: int) -> list[int]:
    if len(values) <= count:
        return list(values)
    indices = np.linspace(0, len(values) - 1, count).round().astype(int)
    return [values[i] for i in sorted(set(indices))]


def safe_path_component(text: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text))
    return safe or "chromosome"


def discover_chromosomes(mcool_path: Path, resolution: int) -> list[tuple[str, int]]:
    probe = cooler.Cooler(f"{mcool_path}::/resolutions/{resolution}")
    return [
        (chrom, int(probe.chromsizes.loc[chrom]))
        for chrom in probe.chromnames
    ]


def select_resolutions_for_chromosome(
    chrom_size_bp: int,
    available: Sequence[int],
    requested: Sequence[int] | None,
    max_scales: int,
    max_bins: int,
) -> list[int]:
    if requested:
        missing = sorted(set(requested) - set(available))
        if missing:
            raise ValueError(f"requested resolutions absent from mcool: {missing}")
        chosen = sorted(set(requested))
    else:
        affordable = [r for r in available if math.ceil(chrom_size_bp / r) <= max_bins]
        if not affordable:
            affordable = [available[-1]]
        chosen = evenly_spaced(affordable, max_scales)
    return chosen


def interval_similarity(
    a_start: int, a_end: int, b_start: int, b_end: int
) -> float:
    overlap = max(0, min(a_end, b_end) - max(a_start, b_start))
    shorter = min(a_end - a_start, b_end - b_start)
    return overlap / shorter if shorter > 0 else 0.0


def build_consensus(
    candidates: pd.DataFrame,
    total_resolutions: int,
    min_scale_support: int,
    min_resolution_support: int,
    overlap_fraction: float,
) -> pd.DataFrame:
    """Merge overlapping candidates using connected components in interval space."""
    columns = [
        "event_id", "start_bp", "end_bp", "length_bp", "support_scales",
        "support_resolutions", "support_gammas", "support_fraction",
        "partner_start_bp", "partner_end_bp",
        "distance_bp", "distance_percent", "severity", "event_score", "passed",
    ]
    if candidates.empty:
        return pd.DataFrame(columns=columns)

    count = len(candidates)
    parent = list(range(count))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    ordered = candidates.sort_values(["start_bp", "end_bp"]).reset_index(drop=True)
    for i in range(count):
        for j in range(i + 1, count):
            if int(ordered.iloc[j]["start_bp"]) >= int(ordered.iloc[i]["end_bp"]):
                break
            if ordered.iloc[i]["scale_id"] == ordered.iloc[j]["scale_id"]:
                continue
            source_similarity = interval_similarity(
                int(ordered.iloc[i]["start_bp"]),
                int(ordered.iloc[i]["end_bp"]),
                int(ordered.iloc[j]["start_bp"]),
                int(ordered.iloc[j]["end_bp"]),
            )
            partner_similarity = interval_similarity(
                int(ordered.iloc[i]["main_start_bp"]),
                int(ordered.iloc[i]["main_end_bp"]),
                int(ordered.iloc[j]["main_start_bp"]),
                int(ordered.iloc[j]["main_end_bp"]),
            )
            if source_similarity >= overlap_fraction and partner_similarity >= overlap_fraction:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(count):
        groups.setdefault(find(i), []).append(i)

    summaries = []
    for indices in groups.values():
        group = ordered.iloc[indices]
        support_scales = group["scale_id"].nunique()
        support_resolutions = group["resolution"].nunique()
        confidence = support_resolutions / total_resolutions
        start_bp = int(np.median(group["start_bp"]))
        end_bp = int(np.median(group["end_bp"]))
        severity = float(np.median(group["severity"]))
        summaries.append(
            {
                "start_bp": start_bp,
                "end_bp": end_bp,
                "length_bp": max(0, end_bp - start_bp),
                "support_scales": support_scales,
                "support_resolutions": ",".join(map(str, sorted(group["resolution"].unique()))),
                "support_gammas": ",".join(f"{x:g}" for x in sorted(group["gamma"].unique())),
                "support_fraction": confidence,
                "partner_start_bp": int(np.median(group["main_start_bp"])),
                "partner_end_bp": int(np.median(group["main_end_bp"])),
                "distance_bp": float(np.median(group["distance_bp"])),
                "distance_percent": float(np.median(group["distance_percent"])),
                "severity": severity,
                "event_score": severity * confidence,
                "passed": (
                    support_scales >= min_scale_support
                    and support_resolutions >= min_resolution_support
                ),
            }
        )

    result = pd.DataFrame(summaries).sort_values(["start_bp", "end_bp"]).reset_index(drop=True)
    result.insert(0, "event_id", [f"TRA_{i:03d}" for i in range(1, len(result) + 1)])
    return result[columns]


def _dense_matrix(clr: cooler.Cooler, chrom: str, balance: bool) -> np.ndarray:
    try:
        matrix = clr.matrix(balance=balance).fetch(chrom)
    except (ValueError, KeyError) as exc:
        if not balance:
            raise
        print(f"  warning: balanced plot unavailable ({exc}); using raw counts")
        matrix = clr.matrix(balance=False).fetch(chrom)
    return np.asarray(matrix, dtype=float)


def plot_consensus_report(
    mcool_path: Path,
    chrom: str,
    chrom_size_bp: int,
    available_resolutions: Sequence[int],
    consensus: pd.DataFrame,
    community_segments: pd.DataFrame,
    community_resolution: int,
    normalized_score: float,
    balance: bool,
    max_plot_bins: int,
    output_path: Path,
) -> None:
    plot_options = [r for r in available_resolutions if math.ceil(chrom_size_bp / r) <= max_plot_bins]
    plot_resolution = min(plot_options) if plot_options else max(available_resolutions)
    clr = cooler.Cooler(f"{mcool_path}::/resolutions/{plot_resolution}")
    matrix = _dense_matrix(clr, chrom, balance)
    positive = matrix[np.isfinite(matrix) & (matrix > 0)]
    chrom_mb = chrom_size_bp / 1_000_000.0
    passed = consensus.loc[consensus["passed"].astype(bool)].copy()

    fig, axes = plt.subplots(1, 3, figsize=(22, 7), gridspec_kw={"width_ratios": [1.2, 2.2, 1.2]})

    ax = axes[0]
    for _, segment in community_segments.iterrows():
        start_mb = int(segment["start_bin"]) * community_resolution / 1e6
        end_mb = min(int(segment["end_bin"]) * community_resolution, chrom_size_bp) / 1e6
        if bool(segment["is_filtered"]):
            color, linewidth, alpha = "lightgray", 1.5, 0.45
        elif bool(segment["is_candidate"]):
            color, linewidth, alpha = "red", 4.0, 0.95
        else:
            color, linewidth, alpha = "royalblue", 3.0, 0.85
        ax.plot(
            [start_mb, end_mb],
            [int(segment["label"]), int(segment["label"])],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            solid_capstyle="butt",
        )
    ax.set_xlim(0, chrom_mb)
    ax.set_xlabel("Genomic position (Mb)")
    ax.set_ylabel("Community cluster")
    ax.set_title(
        f"Community clustering ({community_resolution / 1000:g} kb)\n"
        "Red=candidate, Blue=main, Gray=filtered"
    )
    ax.grid(axis="x", color="0.9", linewidth=0.6)

    ax = axes[1]
    if positive.size:
        vmin = max(float(np.quantile(positive, 0.05)), np.finfo(float).tiny)
        vmax = max(float(np.quantile(positive, 0.995)), vmin * 1.01)
        shown = np.ma.masked_invalid(np.ma.masked_less_equal(matrix, 0))
        ax.imshow(
            shown,
            cmap="Reds",
            norm=LogNorm(vmin=vmin, vmax=vmax),
            extent=[0, chrom_mb, chrom_mb, 0],
            interpolation="nearest",
        )
    else:
        ax.imshow(matrix, cmap="Reds", extent=[0, chrom_mb, chrom_mb, 0])
    for _, event in passed.iterrows():
        start = event.start_bp / 1_000_000.0
        width = event.length_bp / 1_000_000.0
        ax.add_patch(patches.Rectangle((start, start), width, width, fill=False, edgecolor="cyan", linewidth=1.5))
    ax.set_xlabel("Position (Mb)")
    ax.set_ylabel("Position (Mb)")
    ax.set_title(f"{chrom} heatmap ({plot_resolution / 1000:g} kb)\nScore: {normalized_score:.2f} / 100 Mb")

    ax = axes[2]
    if passed.empty:
        ax.text(0.5, 0.5, "No event passed support threshold", transform=ax.transAxes, ha="center")
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        y = np.arange(len(passed))
        bars = ax.barh(y, passed["event_score"], color="firebrick", alpha=0.85)
        ax.set_yticks(y, passed["event_id"])
        ax.invert_yaxis()
        ax.set_xlabel("Confidence-weighted score")
        ax.set_title("Consensus event scores")
        for bar, support in zip(bars, passed["support_scales"]):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f"  n={support}", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _centered_region(center_bp: float, half_width_bp: int, chrom_size_bp: int) -> tuple[int, int]:
    start = max(0, int(center_bp - half_width_bp))
    end = min(chrom_size_bp, int(center_bp + half_width_bp))
    if end <= start:
        end = min(chrom_size_bp, start + 1)
    return start, end


def _fetch_event_blocks(
    clr: cooler.Cooler,
    source_region: str,
    partner_region: str,
    balance: bool,
) -> tuple[list[np.ndarray], bool]:
    pairs = [
        (source_region, source_region),
        (source_region, partner_region),
        (partner_region, source_region),
        (partner_region, partner_region),
    ]

    def fetch(use_balance: bool) -> list[np.ndarray]:
        selector = clr.matrix(balance=use_balance)
        return [np.asarray(selector.fetch(first, second), dtype=float) for first, second in pairs]

    try:
        return fetch(balance), balance
    except (ValueError, KeyError) as exc:
        if not balance:
            raise
        print(f"  warning: balanced local matrices unavailable ({exc}); using raw counts")
        return fetch(False), False


def plot_local_event_reports(
    mcool_path: Path,
    chrom: str,
    chrom_size_bp: int,
    resolutions: Sequence[int],
    consensus: pd.DataFrame,
    balance: bool,
    flank_bp: int,
    max_local_bins: int,
    output_dir: Path,
) -> list[Path]:
    """Create one discontinuous 2x2 local Hi-C map for every passed event."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for _, event in consensus.loc[consensus["passed"].astype(bool)].iterrows():
        source_center = (float(event["start_bp"]) + float(event["end_bp"])) / 2.0
        partner_center = (
            float(event["partner_start_bp"]) + float(event["partner_end_bp"])
        ) / 2.0
        half_width = max(
            flank_bp,
            int(float(event["length_bp"]) / 2.0) + flank_bp,
        )
        source_start, source_end = _centered_region(source_center, half_width, chrom_size_bp)
        partner_start, partner_end = _centered_region(partner_center, half_width, chrom_size_bp)
        max_region_bp = max(source_end - source_start, partner_end - partner_start)
        eligible = [r for r in resolutions if math.ceil(max_region_bp / r) <= max_local_bins]
        local_resolution = min(eligible) if eligible else max(resolutions)

        source_region = f"{chrom}:{source_start}-{source_end}"
        partner_region = f"{chrom}:{partner_start}-{partner_end}"
        clr = cooler.Cooler(f"{mcool_path}::/resolutions/{local_resolution}")
        blocks, _ = _fetch_event_blocks(clr, source_region, partner_region, balance)
        positives = [block[np.isfinite(block) & (block > 0)] for block in blocks]
        positives = [values for values in positives if values.size]
        norm = None
        if positives:
            values = np.concatenate(positives)
            vmin = max(float(np.quantile(values, 0.05)), np.finfo(float).tiny)
            vmax = max(float(np.quantile(values, 0.995)), vmin * 1.01)
            norm = LogNorm(vmin=vmin, vmax=vmax)

        source_bounds = (source_start / 1e6, source_end / 1e6)
        partner_bounds = (partner_start / 1e6, partner_end / 1e6)
        source_event = (
            float(event["start_bp"]) / 1e6,
            float(event["end_bp"]) / 1e6,
        )
        partner_event = (
            float(event["partner_start_bp"]) / 1e6,
            float(event["partner_end_bp"]) / 1e6,
        )
        extents = [
            [*source_bounds, source_bounds[1], source_bounds[0]],
            [*partner_bounds, source_bounds[1], source_bounds[0]],
            [*source_bounds, partner_bounds[1], partner_bounds[0]],
            [*partner_bounds, partner_bounds[1], partner_bounds[0]],
        ]
        rectangles = [
            (source_event[0], source_event[0], source_event[1] - source_event[0], source_event[1] - source_event[0]),
            (partner_event[0], source_event[0], partner_event[1] - partner_event[0], source_event[1] - source_event[0]),
            (source_event[0], partner_event[0], source_event[1] - source_event[0], partner_event[1] - partner_event[0]),
            (partner_event[0], partner_event[0], partner_event[1] - partner_event[0], partner_event[1] - partner_event[0]),
        ]
        titles = ["Candidate x Candidate", "Candidate x Partner", "Partner x Candidate", "Partner x Partner"]

        fig, axes = plt.subplots(2, 2, figsize=(11, 10))
        image = None
        for ax, block, extent, rectangle, title in zip(
            axes.flat, blocks, extents, rectangles, titles
        ):
            shown = np.ma.masked_invalid(np.ma.masked_less_equal(block, 0)) if norm else block
            image = ax.imshow(
                shown,
                cmap="Reds",
                norm=norm,
                extent=extent,
                interpolation="nearest",
                aspect="auto",
            )
            ax.add_patch(
                patches.Rectangle(
                    (rectangle[0], rectangle[1]),
                    rectangle[2],
                    rectangle[3],
                    fill=False,
                    edgecolor="cyan",
                    linewidth=1.5,
                )
            )
            ax.set_title(title)
            ax.set_xlabel("Position (Mb)")
            ax.set_ylabel("Position (Mb)")

        if norm is not None and image is not None:
            fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.75, label="Hi-C contact")
        event_id = str(event["event_id"])
        resolution_support = len(str(event["support_resolutions"]).split(","))
        fig.suptitle(
            f"{event_id} | {chrom} | {local_resolution / 1000:g} kb | "
            f"resolution support={resolution_support}",
            fontsize=13,
        )
        fig.subplots_adjust(top=0.91, right=0.88, hspace=0.28, wspace=0.28)
        output_path = output_dir / f"{event_id}_local_hic.png"
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        written.append(output_path)
    return written


def parse_number_list(text: str | None, cast) -> list | None:
    if text is None or text.lower() == "auto":
        return None
    values = [cast(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("list cannot be empty")
    return values


def analyze_scale(
    mcool_path: Path,
    chrom: str,
    chrom_size_bp: int,
    resolution: int,
    gamma: float,
    edge_quantile: float,
    balance: bool,
    seed: int,
    min_segment_bp: int,
    bridge_bp: int,
    base_penalty: float,
    networkit_threads: int,
) -> tuple[int, float, int, int, pd.DataFrame, pd.DataFrame]:
    """Run one independent resolution/gamma analysis in a worker process."""
    if hasattr(nk, "setNumberOfThreads"):
        nk.setNumberOfThreads(max(1, networkit_threads))
    labels, nodes, edges = community_labels_for_scale(
        mcool_path,
        chrom,
        resolution,
        gamma,
        edge_quantile,
        balance,
        seed,
    )
    candidates, segments = detect_scale_candidates(
        labels,
        resolution,
        gamma,
        chrom_size_bp,
        min_segment_bp,
        bridge_bp,
        base_penalty,
    )
    return resolution, gamma, nodes, edges, candidates, segments


def run_one_chromosome(
    args: argparse.Namespace,
    mcool_path: Path,
    available: Sequence[int],
    requested: Sequence[int] | None,
    gammas: Sequence[float],
    chrom: str,
    chrom_size_bp: int,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    resolutions = select_resolutions_for_chromosome(
        chrom_size_bp, available, requested, args.max_resolutions, args.max_bins
    )
    min_segment_bp = int(args.min_segment_kb * 1000)
    bridge_bp = int(args.bridge_kb * 1000)
    total_runs = len(resolutions) * len(gammas)

    print(f"\nchromosome: {chrom} ({chrom_size_bp / 1e6:.2f} Mb)")
    print(f"resolutions: {resolutions}")
    worker_count = min(args.workers, total_runs)
    threads_per_worker = max(1, (os.cpu_count() or worker_count) // worker_count)
    print(
        f"gammas: {list(gammas)}; analysis runs: {total_runs}; "
        f"parallel workers: {worker_count}"
    )
    if len(resolutions) < args.max_resolutions and requested is None:
        print(
            f"warning: only {len(resolutions)} eligible resolutions are available; "
            f"cannot construct {args.max_resolutions} resolution scales"
        )

    frames = []
    segment_tables: dict[tuple[int, float], pd.DataFrame] = {}
    successful_resolutions = set()
    successful_runs = 0
    tasks = [
        (run_number, resolution, gamma)
        for run_number, (resolution, gamma) in enumerate(
            ((resolution, gamma) for resolution in resolutions for gamma in gammas),
            start=1,
        )
    ]
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=mp.get_context("spawn"),
    ) as executor:
        future_to_scale = {
            executor.submit(
                analyze_scale,
                mcool_path,
                chrom,
                chrom_size_bp,
                resolution,
                gamma,
                args.edge_quantile,
                not args.no_balance,
                args.seed + run_number,
                min_segment_bp,
                bridge_bp,
                args.base_penalty,
                threads_per_worker,
            ): (resolution, gamma)
            for run_number, resolution, gamma in tasks
        }
        for completed, future in enumerate(as_completed(future_to_scale), start=1):
            resolution, gamma = future_to_scale[future]
            try:
                resolution, gamma, nodes, edges, candidates, segments = future.result()
            except (RuntimeError, ValueError, KeyError, OSError) as exc:
                print(
                    f"[{completed}/{total_runs}] {chrom} resolution={resolution}, "
                    f"gamma={gamma:g} skipped: {exc}"
                )
                continue
            print(
                f"[{completed}/{total_runs}] {chrom} resolution={resolution}, gamma={gamma:g}; "
                f"graph={nodes} nodes/{edges} edges; candidates={len(candidates)}"
            )
            frames.append(candidates)
            segment_tables[(resolution, gamma)] = segments
            successful_runs += 1
            successful_resolutions.add(resolution)

    if successful_runs == 0:
        raise RuntimeError(f"all analysis scales failed for {chrom}")

    community_key = min(segment_tables, key=lambda key: (key[0], abs(key[1] - 1.0)))
    community_resolution, _ = community_key
    community_segments = segment_tables[community_key]

    all_candidates = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=CANDIDATE_COLUMNS)
    if not all_candidates.empty:
        all_candidates.insert(0, "chrom", chrom)
        all_candidates = all_candidates.sort_values(
            ["chrom", "start_bp", "end_bp", "resolution", "gamma"]
        ).reset_index(drop=True)
    else:
        all_candidates.insert(0, "chrom", pd.Series(dtype=str))

    consensus = build_consensus(
        all_candidates.drop(columns=["chrom"], errors="ignore"),
        len(successful_resolutions),
        args.min_scale_support,
        min(args.min_resolution_support, len(successful_resolutions)),
        args.overlap_fraction,
    )
    consensus.insert(0, "chrom", chrom)

    passed = consensus.loc[consensus["passed"].astype(bool)]
    normalized_score = 0.0
    if chrom_size_bp > 0 and not passed.empty:
        normalized_score = passed["event_score"].sum() / (chrom_size_bp / 1e6) * 100.0

    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = output_dir / "per_scale_candidates.csv"
    consensus_path = output_dir / "consensus_translocations.csv"
    report_path = output_dir / f"multiscale_report_{safe_path_component(chrom)}.png"
    local_report_dir = output_dir / "local_hic_events"
    all_candidates.to_csv(candidates_path, index=False, float_format="%.6g")
    consensus.to_csv(consensus_path, index=False, float_format="%.6g")
    plot_consensus_report(
        mcool_path,
        chrom,
        chrom_size_bp,
        available,
        consensus,
        community_segments,
        community_resolution,
        normalized_score,
        not args.no_balance,
        args.max_plot_bins,
        report_path,
    )
    local_reports = plot_local_event_reports(
        mcool_path,
        chrom,
        chrom_size_bp,
        resolutions,
        consensus,
        not args.no_balance,
        int(args.local_flank_kb * 1000),
        args.max_local_bins,
        local_report_dir,
    )

    print(f"consensus events: {len(consensus)}; passed: {len(passed)}")
    print(f"local Hi-C reports: {len(local_reports)}")
    print(f"normalized instability score: {normalized_score:.2f} / 100 Mb")
    print(f"results: {output_dir}")
    return all_candidates, consensus, normalized_score


def run_pipeline(args: argparse.Namespace) -> tuple[pd.DataFrame, float]:
    mcool_path = Path(args.mcool).expanduser().resolve()
    if not mcool_path.is_file():
        raise FileNotFoundError(mcool_path)

    available = discover_resolutions(mcool_path)
    requested = parse_number_list(args.resolutions, int)
    gammas = parse_number_list(args.gammas, float) or [1.0]
    chrom_records = discover_chromosomes(mcool_path, available[-1])
    if args.chromosomes:
        wanted = set(parse_number_list(args.chromosomes, str) or [])
        missing = sorted(wanted - {chrom for chrom, _ in chrom_records})
        if missing:
            raise ValueError(f"requested chromosomes absent from mcool: {missing}")
        chrom_records = [
            (chrom, size) for chrom, size in chrom_records if chrom in wanted
        ]
    if not chrom_records:
        raise RuntimeError("no chromosomes selected")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    multi_chrom = len(chrom_records) > 1
    print(f"chromosomes selected: {', '.join(chrom for chrom, _ in chrom_records)}")

    candidate_frames = []
    consensus_frames = []
    score_rows = []
    for chrom, chrom_size_bp in chrom_records:
        chrom_output_dir = (
            output_dir / safe_path_component(chrom)
            if multi_chrom
            else output_dir
        )
        try:
            candidates, consensus, normalized_score = run_one_chromosome(
                args,
                mcool_path,
                available,
                requested,
                gammas,
                chrom,
                chrom_size_bp,
                chrom_output_dir,
            )
        except RuntimeError as exc:
            print(f"[SKIP] {chrom}: {exc}")
            continue
        candidate_frames.append(candidates)
        consensus_frames.append(consensus)
        score_rows.append(
            {
                "chrom": chrom,
                "chrom_size_bp": chrom_size_bp,
                "consensus_events": len(consensus),
                "passed_events": int(consensus["passed"].astype(bool).sum()) if "passed" in consensus else 0,
                "normalized_score_per_100mb": normalized_score,
                "output_dir": str(chrom_output_dir),
            }
        )

    if not consensus_frames:
        raise RuntimeError("all selected chromosomes failed")

    all_candidates = pd.concat(candidate_frames, ignore_index=True)
    all_consensus = pd.concat(consensus_frames, ignore_index=True)
    all_consensus["event_id"] = [
        f"TRA_{i:05d}" for i in range(1, len(all_consensus) + 1)
    ]
    all_candidates.to_csv(
        output_dir / "all_chromosomes_per_scale_candidates.csv",
        index=False,
        float_format="%.6g",
    )
    all_consensus.to_csv(
        output_dir / "all_chromosomes_consensus_translocations.csv",
        index=False,
        float_format="%.6g",
    )
    score_table = pd.DataFrame(score_rows)
    score_table.to_csv(
        output_dir / "chromosome_scores.csv",
        index=False,
        float_format="%.6g",
    )

    total_size_mb = sum(row["chrom_size_bp"] for row in score_rows) / 1e6
    passed_mask = (
        all_consensus["passed"].astype(bool)
        if not all_consensus.empty
        else pd.Series([], dtype=bool)
    )
    total_passed_score = all_consensus.loc[passed_mask, "event_score"].sum()
    genome_score = (
        float(total_passed_score) / total_size_mb * 100.0
        if total_size_mb > 0
        else 0.0
    )
    print("\nAll selected chromosomes finished.")
    print(f"passed events: {int(passed_mask.sum())}")
    print(f"genome-wide normalized instability score: {genome_score:.2f} / 100 Mb")
    print(f"combined results: {output_dir}")
    return all_consensus, genome_score


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mcool", help="one- or multi-chromosome .mcool input")
    parser.add_argument("--output-dir", default="multiscale_translocation_results")
    parser.add_argument(
        "--chromosomes",
        default=None,
        help="comma-separated chromosome names to analyze; default analyzes all chromosomes",
    )
    parser.add_argument("--resolutions", default="auto", help="comma-separated resolutions, or auto")
    parser.add_argument(
        "--gammas",
        default="1.0",
        help="PLM gamma values; keep 1.0 for resolution-only multi-scale analysis",
    )
    parser.add_argument("--max-resolutions", type=int, default=4, help="maximum auto-selected resolutions")
    parser.add_argument("--workers", type=int, default=4, help="parallel scale processes")
    parser.add_argument("--max-bins", type=int, default=15000, help="skip finer auto resolutions above this bin count")
    parser.add_argument("--max-plot-bins", type=int, default=4000)
    parser.add_argument("--max-local-bins", type=int, default=1200)
    parser.add_argument(
        "--local-flank-kb",
        type=float,
        default=2000.0,
        help="flanking sequence shown around each event region",
    )
    parser.add_argument("--min-segment-kb", type=float, default=2000.0)
    parser.add_argument("--bridge-kb", type=float, default=500.0, help="bridge short A-B-A label interruptions")
    parser.add_argument("--edge-quantile", type=float, default=0.90, help="quantile among positive contacts")
    parser.add_argument("--min-scale-support", type=int, default=2)
    parser.add_argument(
        "--min-resolution-support",
        type=int,
        default=2,
        help="minimum distinct resolutions supporting an event",
    )
    parser.add_argument("--overlap-fraction", type=float, default=0.50, help="minimum overlap / shorter interval")
    parser.add_argument("--base-penalty", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-balance", action="store_true", help="use raw contact counts")
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not 0 < args.edge_quantile < 1:
        parser.error("--edge-quantile must be between 0 and 1")
    if not 0 < args.overlap_fraction <= 1:
        parser.error("--overlap-fraction must be in (0, 1]")
    if args.min_scale_support < 1 or args.min_resolution_support < 1:
        parser.error("support thresholds must be at least 1")
    if (
        args.max_resolutions < 1
        or args.workers < 1
        or args.max_bins < 1
        or args.max_plot_bins < 1
        or args.max_local_bins < 1
    ):
        parser.error("resolution and bin limits must be positive")
    if args.min_segment_kb <= 0 or args.bridge_kb < 0 or args.local_flank_kb < 0:
        parser.error("segment sizes must be valid")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)
    run_pipeline(args)


if __name__ == "__main__":
    main()

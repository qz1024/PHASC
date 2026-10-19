#!/usr/bin/env python3
"""Multi-resolution intra-chromosomal translocation detection (INV-aligned workflow).

CLI / outputs:
  --outdir, --jobs, --min-support, --no-balance
  ALL_TRA.tsv                 final calls
  HiC/local, HiC/ALL_TRA_*.png

Chromosomes run in parallel via --jobs (no manual sbatch array split).

Community clustering alone is NOT a TRA call. Final confirmation requires
candidate×partner Hi-C mean to be > --min-enrichment-fold times the mean
contact between the candidate and its flanking (surrounding) communities.
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import re
import shutil
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
from matplotlib.colors import LogNorm, PowerNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator

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
]

FINAL_COLUMNS = [
    "cluster_id",
    "chrom",
    "start",
    "end",
    "size",
    "partner_start",
    "partner_end",
    "partner_size",
    "scale_support",
    "call_support",
    "resolutions",
    "distance_bp",
    "distance_percent",
    "cross_mean",
    "neighbor_mean",
    "enrichment_fold",
    "repeat_flag",
]

# ponytail: include 20 kb so local TRA plots / mapping align with INV finest
PLOT_RESOLUTIONS = (20_000, 50_000, 100_000, 200_000, 500_000)


def run_length_segments(labels: np.ndarray) -> pd.DataFrame:
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


def island_tra_pairs(
    labels: np.ndarray,
    bins: list[tuple[int, int]],
    means: np.ndarray,
    min_fold: float = 1.5,
) -> list[tuple[int, int]]:
    """Type-2 TRA: a short island community whose best contact is a non-flank block.

    Geometry: segment i sits between two other labels and is no longer than either
    flank (the relocated piece, not the partner). Contact to some non-adjacent
    block j must be ≥ min_fold × the stronger flank contact.
    Returns (island_index, partner_index) into bins.
    """
    n = len(bins)
    pairs: list[tuple[int, int]] = []
    if n < 3 or means.size == 0:
        return pairs
    for i in range(1, n - 1):
        lab = int(labels[bins[i][0]])
        if int(labels[bins[i - 1][0]]) == lab or int(labels[bins[i + 1][0]]) == lab:
            continue
        length = bins[i][1] - bins[i][0]
        left_len = bins[i - 1][1] - bins[i - 1][0]
        right_len = bins[i + 1][1] - bins[i + 1][0]
        if length > min(left_len, right_len):
            continue
        flank = max(float(means[i, i - 1]), float(means[i, i + 1]))
        if not np.isfinite(flank) or flank < 0:
            flank = 0.0
        best_j, best_v = -1, -1.0
        for j in range(n):
            if abs(j - i) <= 1:
                continue
            v = float(means[i, j])
            if v > best_v:
                best_v, best_j = v, j
        if best_j >= 0 and best_v >= min_fold * max(flank, 1e-12):
            pairs.append((i, best_j))
    return pairs


def detect_scale_candidates(
    labels: np.ndarray,
    resolution: int,
    gamma: float,
    chrom_size_bp: int,
    min_segment_bp: int,
    bridge_bp: int,
    csr=None,
    island_fold: float = 1.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """TRA candidates of two community forms.

    Type 1 — same graph community appears in ≥2 genomic blocks (a continuous
    region's piece clustered into another community).
    Type 2 — a short island community whose strongest contact is a non-flank
    block (standalone TRA community, e.g. chr6B 39.80–40.00).
    When two same-label runs are separated only by type-2 islands, the flanks
    are not called as type 1 (the island is the TRA, not the split flanks).
    """
    min_bins = max(2, math.ceil(min_segment_bp / resolution))
    bridge_bins = max(0, math.floor(bridge_bp / resolution))
    clean_labels = bridge_short_interruptions(labels, bridge_bins)
    segments = run_length_segments(clean_labels)
    segments["is_filtered"] = segments["length_bins"] < min_bins
    segments["is_candidate"] = False
    segments["main_segment_index"] = pd.NA

    bins = [
        (int(seg.start_bin), int(seg.end_bin))
        for seg in segments.itertuples(index=False)
    ]
    island_idx: set[int] = set()
    partner_of: dict[int, int] = {}
    if csr is not None and len(bins) >= 3:
        means = community_contact_means(csr, bins)
        for i, j in island_tra_pairs(clean_labels, bins, means, island_fold):
            island_idx.add(i)
            partner_of[i] = j

    valid = segments.loc[~segments["is_filtered"]].copy()
    valid["block_id"] = valid["label"].ne(valid["label"].shift()).cumsum()
    for _, group in valid.groupby("label", sort=False):
        if group["block_id"].nunique() < 2:
            continue
        block_lengths = group.groupby("block_id")["length_bins"].sum()
        main_block = block_lengths.idxmax()
        main_rows = group.loc[group["block_id"] == main_block]
        main_index = int(main_rows["length_bins"].idxmax())
        for sec_index in group.index[group["block_id"] != main_block]:
            lo, hi = sorted((int(sec_index), main_index))
            gap = range(lo + 1, hi)
            # flanks split only by type-2 islands → not type 1
            if gap and all(g in island_idx for g in gap):
                continue
            # gap-edge micro copies with zero partner contact are not TRA
            if csr is not None:
                a0, a1 = bins[int(sec_index)]
                b0, b1 = bins[main_index]
                if _block_mean_csr(csr, a0, a1, b0, b1) <= 0:
                    continue
            segments.loc[sec_index, "is_candidate"] = True
            segments.loc[sec_index, "main_segment_index"] = main_index

    # type 2: island → contact partner (may overwrite type-1 partner for the island)
    for i, j in partner_of.items():
        if bool(segments.loc[i, "is_filtered"]):
            continue
        segments.loc[i, "is_candidate"] = True
        segments.loc[i, "main_segment_index"] = j

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
                "distance_percent": 100.0 * distance_bp / chrom_size_bp,
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


def distance_positive_medians(dist: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Median positive contact at each genomic distance, in bins."""
    order = np.argsort(dist, kind="mergesort")
    dist_s = dist[order]
    weight_s = weight[order]
    change = np.flatnonzero(np.diff(dist_s)) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, dist_s.size]
    med = np.zeros(int(dist_s[-1]) + 1, dtype=float)
    for start, end in zip(starts, ends):
        if end - start < 5:
            continue
        med[int(dist_s[start])] = float(np.median(weight_s[start:end]))
    return med


def _pair_corr(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Row-wise Pearson correlation. Zero when a profile has no variance."""
    if left.shape[1] < 3:
        return np.zeros(left.shape[0], dtype=float)
    a = left - left.mean(axis=1, keepdims=True)
    b = right - right.mean(axis=1, keepdims=True)
    denom = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    num = (a * b).sum(axis=1)
    out = np.zeros(num.shape, dtype=float)
    ok = denom > 0
    out[ok] = num[ok] / denom[ok]
    out[~np.isfinite(out)] = 0.0
    return out


def tra_edge_weights(
    rows: np.ndarray,
    cols: np.ndarray,
    raw: np.ndarray,
    n_bins: int,
    radius: int,
    direction_lambda: float,
    long_fold: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Edge weight from the signal between the two bins.

    s = raw / median(positive contacts at that distance)
    D, C from the ±radius signal profiles of the two endpoints, as in inversion
    weight = s * (1 + λ D C)

    Local edges (distance <= radius) use this weight whenever the signal is
    positive, so a weak boundary can split the chain. A farther edge is added
    only when weight > long_fold, i.e. the signal between those two bins is
    several times the background at that distance.
    """
    dist = cols - rows
    med = distance_positive_medians(dist, raw)
    bg = np.zeros(raw.shape, dtype=float)
    known = dist < med.size
    bg[known] = med[dist[known]]
    contact = np.divide(raw, bg, out=np.zeros_like(raw), where=bg > 0)
    band = np.zeros((n_bins, radius * 2), dtype=float)
    local = (dist >= 1) & (dist <= radius) & (contact > 0)
    if np.any(local):
        dloc = dist[local]
        band[rows[local], radius + dloc - 1] = contact[local]
        band[cols[local], radius - dloc] = contact[local]
    left = band[rows]
    right = band[cols]
    forward = _pair_corr(left[:, :-1], right[:, 1:])
    reverse = _pair_corr(left[:, 1:], right[:, :-1])
    score = forward - reverse
    confidence = np.abs(score) / (np.abs(forward) + np.abs(reverse) + 1e-8)
    weight = contact * (1.0 + direction_lambda * score * confidence)
    keep = ((dist <= radius) & (contact > 0) & (weight > 0)) | (
        (dist > radius) & (weight > long_fold)
    )
    return rows[keep], cols[keep], weight[keep]


def community_labels_for_scale(
    mcool_path: Path,
    chrom: str,
    resolution: int,
    gamma: float,
    edge_quantile: float,
    balance: bool,
    seed: int,
    radius: int = 8,
    direction_lambda: float = 0.5,
    peak_fraction: float = 1.0,
    community_signal_quantile: float = 0.05,
) -> tuple[np.ndarray, int, int, object]:
    # ponytail: radius/direction_lambda kept in signature for CLI compat; unused
    del radius, direction_lambda
    clr = cooler.Cooler(f"{mcool_path}::/resolutions/{resolution}")
    matrix, _ = _balanced_sparse_matrix(clr, chrom, balance)
    coo = matrix.tocoo()
    valid = (coo.row < coo.col) & np.isfinite(coo.data) & (coo.data > 0)
    rows = np.asarray(coo.row[valid], dtype=np.int64)
    cols = np.asarray(coo.col[valid], dtype=np.int64)
    weights = np.asarray(coo.data[valid], dtype=float)
    if weights.size == 0:
        raise RuntimeError(f"no positive contacts at resolution {resolution}")
    # original TRA graph: keep top (1 - edge_quantile) raw contacts as edges
    threshold = float(np.quantile(weights, edge_quantile))
    keep = weights >= threshold
    rows, cols, weights = rows[keep], cols[keep], weights[keep]
    graph = nk.graph.Graph(int(matrix.shape[0]), weighted=True, directed=False)
    for row, col, weight in zip(rows, cols, weights):
        graph.addEdge(int(row), int(col), float(weight))
    if graph.numberOfEdges() == 0:
        raise RuntimeError(
            f"contact threshold removed every edge at resolution {resolution}"
        )
    nk.setSeed(seed, False)
    algorithm = nk.community.PLM(graph, refine=True, gamma=gamma)
    algorithm.run()
    labels = np.asarray(algorithm.getPartition().getVector(), dtype=np.int64)
    labels, n_split = resplit_large_blocks(
        labels, rows, cols, weights, resolution, gamma, seed
    )
    if n_split:
        print(
            f"[SPLIT] {chrom} r={resolution}: second Louvain on {n_split} large blocks",
            flush=True,
        )
    # ponytail: 2 Mb sandwich ceiling; true TRA with large intervening sequence stays
    max_sandwich_bins = max(2, math.ceil(2_000_000 / resolution))
    csr = matrix.tocsr()
    labels, absorbed = absorb_weak_communities(
        labels,
        csr,
        peak_fraction,
        max_sandwich_bins=max_sandwich_bins,
        community_signal_quantile=community_signal_quantile,
    )
    if absorbed:
        print(
            f"[WEAK] {chrom} r={resolution}: merged {absorbed} weak/micro/sandwich "
            f"communities into a neighbor",
            flush=True,
        )
    return labels, graph.numberOfNodes(), graph.numberOfEdges(), csr


def resplit_large_blocks(
    labels: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    weights: np.ndarray,
    resolution: int,
    gamma: float,
    seed: int,
    min_bp: int = 5_000_000,
) -> tuple[np.ndarray, int]:
    """Run Louvain again inside each community longer than min_bp.

    Uses the same kept edges, restricted to both ends inside the block.
    ponytail: one extra pass only; blocks shorter than 5 Mb or 20 bins stay.
    """
    out = np.asarray(labels, dtype=np.int64).copy()
    if out.size == 0:
        return out, 0
    next_id = int(out.max()) + 1
    n_split = 0
    for seg in run_length_segments(out).itertuples(index=False):
        b0, b1 = int(seg.start_bin), int(seg.end_bin)
        if (b1 - b0) < 20 or (b1 - b0) * resolution < min_bp:
            continue
        inside = (rows >= b0) & (rows < b1) & (cols >= b0) & (cols < b1)
        if int(inside.sum()) < 2:
            continue
        graph = nk.graph.Graph(b1 - b0, weighted=True, directed=False)
        for row, col, weight in zip(rows[inside], cols[inside], weights[inside]):
            graph.addEdge(int(row - b0), int(col - b0), float(weight))
        if graph.numberOfEdges() == 0:
            continue
        nk.setSeed(seed + b0, False)
        algorithm = nk.community.PLM(graph, refine=True, gamma=gamma)
        algorithm.run()
        local = np.asarray(algorithm.getPartition().getVector(), dtype=np.int64)
        if np.unique(local).size < 2:
            continue
        mapping = {int(loc): next_id + i for i, loc in enumerate(np.unique(local))}
        next_id += len(mapping)
        out[b0:b1] = np.array([mapping[int(v)] for v in local], dtype=np.int64)
        n_split += 1
    return out, n_split


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
    return [(chrom, int(probe.chromsizes.loc[chrom])) for chrom in probe.chromnames]


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
        return sorted(set(requested))
    affordable = [r for r in available if math.ceil(chrom_size_bp / r) <= max_bins]
    if not affordable:
        affordable = [available[-1]]
    return evenly_spaced(affordable, max_scales)


def interval_similarity(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    overlap = max(0, min(a_end, b_end) - max(a_start, b_start))
    shorter = min(a_end - a_start, b_end - b_start)
    return overlap / shorter if shorter > 0 else 0.0


def build_consensus(
    candidates: pd.DataFrame,
    min_support: int,
    overlap_fraction: float,
) -> pd.DataFrame:
    """Merge overlapping candidates; keep events with scale_support >= min_support."""
    if candidates.empty:
        return pd.DataFrame(
            columns=[
                "start_bp",
                "end_bp",
                "length_bp",
                "support_scales",
                "support_resolutions",
                "partner_start_bp",
                "partner_end_bp",
                "distance_bp",
                "distance_percent",
            ]
        )

    count = len(candidates)
    parent = list(range(count))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

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
        support_scales = int(group["scale_id"].nunique())
        support_resolutions = int(group["resolution"].nunique())
        if support_scales < min_support or support_resolutions < min_support:
            continue
        start_bp = int(np.median(group["start_bp"]))
        end_bp = int(np.median(group["end_bp"]))
        summaries.append(
            {
                "start_bp": start_bp,
                "end_bp": end_bp,
                "length_bp": max(0, end_bp - start_bp),
                "support_scales": support_scales,
                "support_resolutions": ",".join(
                    map(str, sorted(group["resolution"].unique()))
                ),
                "partner_start_bp": int(np.median(group["main_start_bp"])),
                "partner_end_bp": int(np.median(group["main_end_bp"])),
                "distance_bp": float(np.median(group["distance_bp"])),
                "distance_percent": float(np.median(group["distance_percent"])),
            }
        )
    if not summaries:
        return pd.DataFrame(
            columns=[
                "start_bp",
                "end_bp",
                "length_bp",
                "support_scales",
                "support_resolutions",
                "partner_start_bp",
                "partner_end_bp",
                "distance_bp",
                "distance_percent",
            ]
        )
    return pd.DataFrame(summaries).sort_values(["start_bp", "end_bp"]).reset_index(drop=True)


def block_mean_contact(
    clr: cooler.Cooler, chrom: str, a0: int, a1: int, b0: int, b1: int
) -> float:
    """Mean raw contact in rectangle [a0,a1) × [b0,b1)."""
    a0, a1 = max(0, int(a0)), max(int(a0) + 1, int(a1))
    b0, b1 = max(0, int(b0)), max(int(b0) + 1, int(b1))
    mat = np.asarray(
        clr.matrix(balance=False).fetch((chrom, a0, a1), (chrom, b0, b1)),
        dtype=float,
    )
    mat = mat[np.isfinite(mat)]
    if mat.size == 0:
        return 0.0
    return float(np.mean(np.maximum(mat, 0.0)))


def _interval_overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def flanking_neighbor_intervals(
    segments: pd.DataFrame,
    resolution: int,
    chrom_size_bp: int,
    cand_start: int,
    cand_end: int,
    partner_start: int,
    partner_end: int,
) -> list[tuple[int, int]]:
    """Left/right community intervals around candidate, excluding partner."""
    blocks = []
    for _, seg in segments.iterrows():
        if bool(seg["is_filtered"]):
            continue
        s = int(seg["start_bin"]) * resolution
        e = min(int(seg["end_bin"]) * resolution, chrom_size_bp)
        if e <= s:
            continue
        # skip self / partner
        if _interval_overlap(s, e, cand_start, cand_end) > 0.5 * min(e - s, cand_end - cand_start):
            continue
        if _interval_overlap(s, e, partner_start, partner_end) > 0.5 * min(
            e - s, max(1, partner_end - partner_start)
        ):
            continue
        blocks.append((s, e))
    if not blocks:
        return []
    blocks.sort()
    left = None
    right = None
    for s, e in blocks:
        if e <= cand_start:
            left = (s, e)
        elif s >= cand_end and right is None:
            right = (s, e)
            break
    out = []
    if left is not None:
        out.append(left)
    if right is not None:
        out.append(right)
    # ponytail: if adjacency empty (telomere), fall back to nearest other block
    if not out:
        cand_mid = 0.5 * (cand_start + cand_end)
        blocks.sort(key=lambda se: abs(0.5 * (se[0] + se[1]) - cand_mid))
        out = blocks[:2]
    return out


def choose_contact_resolution(
    available: Sequence[int],
    chrom_size_bp: int,
    event_resolutions: str,
    max_bins: int = 8000,
) -> int:
    want = []
    for part in str(event_resolutions).split(","):
        part = part.strip()
        if part.isdigit():
            r = int(part)
            if r in available:
                want.append(r)
    pool = sorted(want) or sorted(
        r for r in available if math.ceil(chrom_size_bp / r) <= max_bins
    )
    if not pool:
        return max(available)
    # prefer finest supported scale that still fits
    for r in pool:
        if math.ceil(chrom_size_bp / r) <= max_bins:
            return r
    return pool[-1]


def score_event_vs_neighbors(
    clr: cooler.Cooler,
    chrom: str,
    chrom_size_bp: int,
    segments: pd.DataFrame,
    community_resolution: int,
    cand_start: int,
    cand_end: int,
    partner_start: int,
    partner_end: int,
) -> dict:
    """Compare candidate×partner contact to candidate×flanking communities."""
    cross = block_mean_contact(
        clr, chrom, cand_start, cand_end, partner_start, partner_end
    )
    neighbors = flanking_neighbor_intervals(
        segments,
        community_resolution,
        chrom_size_bp,
        cand_start,
        cand_end,
        partner_start,
        partner_end,
    )
    neigh_vals = [
        block_mean_contact(clr, chrom, cand_start, cand_end, s, e) for s, e in neighbors
    ]
    neighbor_mean = float(np.mean(neigh_vals)) if neigh_vals else 0.0
    if neighbor_mean > 0:
        fold = cross / neighbor_mean
    else:
        fold = float("inf") if cross > 0 else 0.0
    return {
        "cross_mean": cross,
        "neighbor_mean": neighbor_mean,
        "enrichment_fold": fold,
        "n_neighbor_blocks": len(neighbors),
    }


def repeat_mask_from_means(means: np.ndarray, peak_fraction: float) -> tuple[np.ndarray, set[tuple[int, int]]]:
    """Communities whose total contact to all other communities is still weak.

    Strength of community i is sum_{j≠i} means[i,j] (block-mean contacts).
    Baseline is the 75th percentile of each community's contact sum to its
    genomic neighbor(s), so the threshold is also a sum and depleted single
    edges do not set the bar. Returns (is_repeat, flank_pairs).
    ponytail: O(n^2) block means, n = community count; upgrade path is a binned
    summary matrix if n gets into the thousands.
    """
    n = int(means.shape[0])
    mask = np.zeros(n, dtype=bool)
    if n < 3:
        return mask, set()
    neighbor_sum = np.zeros(n, dtype=float)
    for i in range(n):
        if i > 0:
            neighbor_sum[i] += float(means[i, i - 1])
        if i + 1 < n:
            neighbor_sum[i] += float(means[i, i + 1])
    neighbor_sum = neighbor_sum[np.isfinite(neighbor_sum)]
    if neighbor_sum.size == 0:
        return mask, set()
    baseline = float(np.percentile(neighbor_sum, 75))
    if baseline <= 0:
        return mask, set()
    off = np.array(means, dtype=float, copy=True)
    np.fill_diagonal(off, 0.0)
    total = np.nansum(off, axis=1)
    mask = np.isfinite(total) & (total < peak_fraction * baseline)
    pairs: set[tuple[int, int]] = set()
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i + 1
        while j < n and mask[j]:
            j += 1
        left, right = i - 1, j
        if left >= 0 and right < n:
            pairs.add((left, right))
        i = j
    return mask, pairs


def _block_mean_csr(csr, a0: int, a1: int, b0: int, b1: int) -> float:
    area = (a1 - a0) * (b1 - b0)
    if area <= 0:
        return 0.0
    total = float(csr[a0:a1, b0:b1].sum())
    if not np.isfinite(total):
        return 0.0
    return total / area


def absorb_weak_runs(
    labels: np.ndarray,
    means: np.ndarray,
    bins: list[tuple[int, int]],
    mask: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Paint a run of absorbable communities into one flanking anchor.

    The whole run goes to the left or right non-absorbable neighbor, chosen by
    length-weighted contact of the run to each flank. Per-community choices
    would leave a string of micro-communities half-merged into each other.
    """
    out = np.array(labels, copy=True)
    absorbed = 0
    i = 0
    n = int(mask.size)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i + 1
        while j < n and mask[j]:
            j += 1
        left = i - 1 if i > 0 else None
        right = j if j < n else None
        if left is None and right is None:
            i = j
            continue
        if left is not None and right is not None:
            score_l = 0.0
            score_r = 0.0
            for k in range(i, j):
                length = bins[k][1] - bins[k][0]
                score_l += float(means[k, left]) * length
                score_r += float(means[k, right]) * length
            prefer = left if score_l >= score_r else right
        elif left is not None:
            prefer = left
        else:
            prefer = right
        new_label = int(labels[bins[prefer][0]])
        for k in range(i, j):
            b0, b1 = bins[k]
            out[b0:b1] = new_label
            absorbed += 1
        i = j
    return out, absorbed


def chrom_diag_threshold(csr, quantile: float) -> float:
    """Chromosome-wide diagonal contact quantile (positive bins only)."""
    diag = np.asarray(csr.diagonal(), dtype=float)
    pos = diag[np.isfinite(diag) & (diag > 0)]
    if pos.size == 0:
        return 0.0
    return float(np.quantile(pos, quantile))


def community_diag_means(csr, bins: list[tuple[int, int]]) -> np.ndarray:
    """Mean diagonal (self) contact inside each community."""
    diag = np.asarray(csr.diagonal(), dtype=float)
    out = np.zeros(len(bins), dtype=float)
    for i, (a0, a1) in enumerate(bins):
        chunk = diag[a0:a1]
        ok = chunk[np.isfinite(chunk)]
        out[i] = float(ok.mean()) if ok.size else 0.0
    return out


def sandwich_absorbable_mask(
    labels: np.ndarray,
    bins: list[tuple[int, int]],
    max_gap_bins: int,
    max_micro_bins: int = 2,
    is_micro: np.ndarray | None = None,
) -> np.ndarray:
    """Mark a short gap that splits two runs of the same label.

    Requires each flanking run to be at least as long as the gap, so a chain of
    small same-label fragments cannot zipper-merge a whole chromosome.
    Non-micro secondary copies of a label are kept (type-1 TRA); micro copies
    stay absorbable unless a later island-contact check protects them.
    """
    n = len(bins)
    mask = np.zeros(n, dtype=bool)
    if n < 3 or max_gap_bins <= 0:
        return mask
    seg_labels = np.array([int(labels[b0]) for b0, _ in bins], dtype=np.int64)
    lengths = np.array([b1 - b0 for b0, b1 in bins], dtype=np.int64)
    if is_micro is None:
        is_micro = lengths <= max_micro_bins
    left = 0
    while left < n:
        right = left + 1
        while right < n and seg_labels[right] == seg_labels[left]:
            right += 1
        flank_left = int(lengths[left:right].sum())
        k = right
        while k < n and seg_labels[k] != seg_labels[left]:
            k += 1
        if k < n and right < k:
            gap_bins = int(bins[k][0] - bins[right - 1][1])
            k_end = k + 1
            while k_end < n and seg_labels[k_end] == seg_labels[k]:
                k_end += 1
            flank_right = int(lengths[k:k_end].sum())
            if (
                0 < gap_bins <= max_gap_bins
                and flank_left >= gap_bins
                and flank_right >= gap_bins
            ):
                mask[right:k] = True
        left = right
    # non-micro secondary label copies are type-1 TRA pieces — do not sandwich them
    counts: dict[int, int] = {}
    for b0, _ in bins:
        lab = int(labels[b0])
        counts[lab] = counts.get(lab, 0) + 1
    for i, (b0, _) in enumerate(bins):
        if counts[int(labels[b0])] > 1 and not bool(is_micro[i]):
            mask[i] = False
    return mask


def absorb_weak_communities(
    labels: np.ndarray,
    csr,
    peak_fraction: float,
    max_micro_bins: int = 2,
    max_sandwich_bins: int = 40,
    community_signal_quantile: float = 0.05,
) -> tuple[np.ndarray, int]:
    """Merge weak / micro / same-label-sandwich runs into a flanking anchor.

    Absorbable communities are:
      (1) weak by the repeat rule — sum of mean contacts to all other
          communities still below --repeat-peak-fraction × P75 of each
          community's sum of contacts to its genomic neighbor(s); or
      (2) micro — at most max_micro_bins long, or mean diagonal contact
          below the chromosome-wide diagonal quantile; or
      (3) sandwich — short gap between two longer runs of the same label
          (step-2 form of repeat_flank).

    Single pass each — iterating weak|micro zippers the chromosome into one
    community because every merge creates new short/weak edges.
    """
    current = np.asarray(labels, dtype=np.int64).copy()
    total = 0
    segments = run_length_segments(current)
    bins = [
        (int(seg.start_bin), int(seg.end_bin))
        for seg in segments.itertuples(index=False)
        if int(seg.end_bin) > int(seg.start_bin)
    ]
    if len(bins) < 3:
        return current, 0
    means = community_contact_means(csr, bins)
    weak, _ = repeat_mask_from_means(means, peak_fraction)
    sig_thresh = chrom_diag_threshold(csr, community_signal_quantile)
    diag_means = community_diag_means(csr, bins)
    micro = np.array(
        [
            (b1 - b0) <= max_micro_bins or diag_means[i] < sig_thresh
            for i, (b0, b1) in enumerate(bins)
        ],
        dtype=bool,
    )
    absorbable = weak | micro
    # keep non-micro secondary copies (type 1). Micro junk near gaps often
    # shares a distant label by chance — still absorb unless type-2 contact.
    label_counts: dict[int, int] = {}
    for b0, _ in bins:
        lab = int(current[b0])
        label_counts[lab] = label_counts.get(lab, 0) + 1
    for i, (b0, _) in enumerate(bins):
        if label_counts[int(current[b0])] > 1 and not micro[i]:
            absorbable[i] = False
    # type-2 keeps a real island; a below-quantile diagonal is still micro
    for i, _ in island_tra_pairs(current, bins, means):
        if not micro[i]:
            absorbable[i] = False
    if bool(absorbable.any()):
        current, n_abs = absorb_weak_runs(current, means, bins, absorbable)
        total += n_abs
    # sandwich on the post-micro partition; flanks must be ≥ gap (see mask)
    segments = run_length_segments(current)
    bins = [
        (int(seg.start_bin), int(seg.end_bin))
        for seg in segments.itertuples(index=False)
        if int(seg.end_bin) > int(seg.start_bin)
    ]
    if len(bins) >= 3 and max_sandwich_bins > 0:
        means = community_contact_means(csr, bins)
        diag_means = community_diag_means(csr, bins)
        is_micro = np.array(
            [
                (b1 - b0) <= max_micro_bins or diag_means[i] < sig_thresh
                for i, (b0, b1) in enumerate(bins)
            ],
            dtype=bool,
        )
        sandwich = sandwich_absorbable_mask(
            current,
            bins,
            max_sandwich_bins,
            max_micro_bins=max_micro_bins,
            is_micro=is_micro,
        )
        for i, _ in island_tra_pairs(current, bins, means):
            if not is_micro[i]:
                sandwich[i] = False
        if bool(sandwich.any()):
            current, n_abs = absorb_weak_runs(current, means, bins, sandwich)
            total += n_abs
    return current, total


def community_contact_means(csr, bins: list[tuple[int, int]]) -> np.ndarray:
    n = len(bins)
    means = np.zeros((n, n), dtype=float)
    for i, (a0, a1) in enumerate(bins):
        for j in range(i, n):
            b0, b1 = bins[j]
            val = _block_mean_csr(csr, a0, a1, b0, b1)
            means[i, j] = means[j, i] = val
    return means


def _best_segment(bounds: list[tuple[int, int]], start: int, end: int) -> int | None:
    best_i = None
    best_ov = 0
    for i, (s, e) in enumerate(bounds):
        ov = _interval_overlap(start, end, s, e)
        if ov > best_ov:
            best_i, best_ov = i, ov
    if best_i is None or best_ov <= 0:
        return None
    return best_i


def flag_repeat_translocations(
    mcool_path: Path,
    chrom: str,
    chrom_size_bp: int,
    scored: pd.DataFrame,
    segments: pd.DataFrame,
    community_resolution: int,
    balance: bool,
    peak_fraction: float,
) -> pd.DataFrame:
    """Mark repeat communities and calls that are just the two sides of one.

    repeat: the community's contact with every other community is weak.
    repeat_flank: the call is the pair of communities on either side of a repeat,
    which is one region split by the repeat, not a translocation.
    """
    out = scored.copy()
    out["repeat_flag"] = "ok"
    valid = segments.loc[~segments["is_filtered"]].sort_values("start_bin")
    bins: list[tuple[int, int]] = []
    bounds: list[tuple[int, int]] = []
    for _, seg in valid.iterrows():
        b0 = int(seg["start_bin"])
        b1 = int(seg["end_bin"])
        if b1 <= b0:
            continue
        bins.append((b0, b1))
        bounds.append((b0 * community_resolution, min(b1 * community_resolution, chrom_size_bp)))
    if len(bins) < 3 or out.empty:
        return out
    clr = cooler.Cooler(f"{mcool_path}::/resolutions/{community_resolution}")
    csr, _ = _balanced_sparse_matrix(clr, chrom, balance)
    csr = csr.tocsr()
    mask, flank_pairs = repeat_mask_from_means(
        community_contact_means(csr, bins), peak_fraction
    )
    n_repeat = int(mask.sum())
    n_flank = 0
    flags = []
    for _, row in out.iterrows():
        cand = _best_segment(bounds, int(row["start_bp"]), int(row["end_bp"]))
        partner = _best_segment(bounds, int(row["partner_start_bp"]), int(row["partner_end_bp"]))
        flag = "ok"
        if cand is not None and mask[cand]:
            flag = "repeat"
        elif partner is not None and mask[partner]:
            flag = "repeat"
        elif (
            cand is not None
            and partner is not None
            and (min(cand, partner), max(cand, partner)) in flank_pairs
        ):
            flag = "repeat_flank"
            n_flank += 1
        flags.append(flag)
    out["repeat_flag"] = flags
    print(
        f"[REPEAT] {chrom}: communities={len(bins)} repeats={n_repeat} "
        f"flank_pairs={len(flank_pairs)} "
        f"calls repeat/flank/ok="
        f"{flags.count('repeat')}/{flags.count('repeat_flank')}/{flags.count('ok')}",
        flush=True,
    )
    return out


def filter_consensus_by_enrichment(
    mcool_path: Path,
    chrom: str,
    chrom_size_bp: int,
    consensus: pd.DataFrame,
    segment_tables: dict[tuple[int, float], pd.DataFrame],
    available: Sequence[int],
    min_enrichment_fold: float,
    balance: bool,
    repeat_peak_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score all Louvain consensus; keep only enrichment-confirmed TRA.

    Returns (scored_all, filtered_pass).
    """
    if consensus.empty or not segment_tables:
        empty = consensus.copy()
        for col in ("cross_mean", "neighbor_mean", "enrichment_fold"):
            if col not in empty.columns:
                empty[col] = pd.Series(dtype=float)
        empty["repeat_flag"] = pd.Series(dtype=str)
        return empty, empty.iloc[0:0].copy()

    community_key = min(segment_tables, key=lambda key: (key[0], abs(key[1] - 1.0)))
    community_resolution, _ = community_key
    segments = segment_tables[community_key]

    scored_rows = []
    for _, row in consensus.iterrows():
        contact_res = choose_contact_resolution(
            available, chrom_size_bp, str(row["support_resolutions"])
        )
        clr = cooler.Cooler(f"{mcool_path}::/resolutions/{contact_res}")
        metrics = score_event_vs_neighbors(
            clr,
            chrom,
            chrom_size_bp,
            segments,
            community_resolution,
            int(row["start_bp"]),
            int(row["end_bp"]),
            int(row["partner_start_bp"]),
            int(row["partner_end_bp"]),
        )
        scored = row.to_dict()
        scored.update(metrics)
        scored_rows.append(scored)
        print(
            f"[ENRICH] {chrom} {int(row['start_bp'])}-{int(row['end_bp'])}: "
            f"cross={metrics['cross_mean']:.4g} neigh={metrics['neighbor_mean']:.4g} "
            f"fold={metrics['enrichment_fold']:.3g} "
            f"{'PASS' if metrics['enrichment_fold'] > min_enrichment_fold else 'DROP'}",
            flush=True,
        )

    scored_all = pd.DataFrame(scored_rows)
    scored_all = flag_repeat_translocations(
        mcool_path,
        chrom,
        chrom_size_bp,
        scored_all,
        segments,
        community_resolution,
        balance,
        repeat_peak_fraction,
    )
    # strict >: "强过周围社区多少倍" — equal to threshold stays DROP
    passed = scored_all.loc[
        (scored_all["enrichment_fold"] > min_enrichment_fold)
        & (scored_all["repeat_flag"] == "ok")
    ].reset_index(drop=True)
    return scored_all, passed


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
    networkit_threads: int,
    radius: int = 8,
    direction_lambda: float = 0.5,
    peak_fraction: float = 1.0,
    community_signal_quantile: float = 0.05,
) -> tuple[int, float, int, int, pd.DataFrame, pd.DataFrame]:
    if hasattr(nk, "setNumberOfThreads"):
        nk.setNumberOfThreads(max(1, networkit_threads))
    labels, nodes, edges, csr = community_labels_for_scale(
        mcool_path,
        chrom,
        resolution,
        gamma,
        edge_quantile,
        balance,
        seed,
        radius,
        direction_lambda,
        peak_fraction,
        community_signal_quantile,
    )
    candidates, segments = detect_scale_candidates(
        labels,
        resolution,
        gamma,
        chrom_size_bp,
        min_segment_bp,
        bridge_bp,
        csr=csr,
    )
    return resolution, gamma, nodes, edges, candidates, segments


def _dense_matrix(clr: cooler.Cooler, chrom: str, balance: bool) -> np.ndarray:
    try:
        matrix = clr.matrix(balance=balance).fetch(chrom)
    except (ValueError, KeyError) as exc:
        if not balance:
            raise
        print(f"  warning: balanced plot unavailable ({exc}); using raw counts")
        matrix = clr.matrix(balance=False).fetch(chrom)
    return np.asarray(matrix, dtype=float)


def plot_community_report(
    mcool_path: Path,
    chrom: str,
    chrom_size_bp: int,
    available_resolutions: Sequence[int],
    consensus: pd.DataFrame,
    community_segments: pd.DataFrame,
    community_resolution: int,
    balance: bool,
    max_plot_bins: int,
    output_path: Path,
) -> None:
    """Fixed per-chrom community figure: clustering track + whole-chrom Hi-C.

    Score panel removed (INV-aligned; no severity ranking).
    """
    plot_options = [
        r
        for r in available_resolutions
        if math.ceil(chrom_size_bp / r) <= max_plot_bins
    ]
    plot_resolution = min(plot_options) if plot_options else max(available_resolutions)
    clr = cooler.Cooler(f"{mcool_path}::/resolutions/{plot_resolution}")
    matrix = _dense_matrix(clr, chrom, balance)
    positive = matrix[np.isfinite(matrix) & (matrix > 0)]
    chrom_mb = chrom_size_bp / 1e6

    fig, axes = plt.subplots(
        1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1.2, 2.2]}
    )

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
        lab = int(segment["label"])
        ax.plot(
            [start_mb, end_mb],
            [lab, lab],
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
    ax.grid(True, axis="x", alpha=0.3)

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
            aspect="auto",
        )
    else:
        ax.imshow(matrix, cmap="Reds", extent=[0, chrom_mb, chrom_mb, 0])

    for _, event in consensus.iterrows():
        start = float(event["start_bp"]) / 1e6
        width = float(event["length_bp"]) / 1e6
        ax.add_patch(
            patches.Rectangle(
                (start, start),
                width,
                width,
                fill=False,
                edgecolor="cyan",
                linewidth=1.5,
            )
        )
        if "partner_start_bp" in event.index:
            p0 = float(event["partner_start_bp"]) / 1e6
            p1 = float(event["partner_end_bp"]) / 1e6
            pw = max(0.0, p1 - p0)
            ax.add_patch(
                patches.Rectangle(
                    (p0, p0),
                    pw,
                    pw,
                    fill=False,
                    edgecolor="#00bfa5",
                    linewidth=1.2,
                )
            )
    ax.set_xlabel("Position (Mb)")
    ax.set_ylabel("Position (Mb)")
    ax.set_title(
        f"{chrom} heatmap ({plot_resolution / 1000:g} kb)\n"
        "cyan=candidate  teal=partner"
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] community report -> {output_path}", flush=True)


def parse_number_list(text: str | None, cast) -> list | None:
    if text is None or str(text).lower() == "auto":
        return None
    values = [cast(item.strip()) for item in str(text).split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("list cannot be empty")
    return values


def run_one_chromosome(
    mcool_path: Path,
    available: Sequence[int],
    requested: Sequence[int] | None,
    gammas: Sequence[float],
    chrom: str,
    chrom_size_bp: int,
    max_resolutions: int,
    max_bins: int,
    workers: int,
    edge_quantile: float,
    balance: bool,
    seed: int,
    min_segment_bp: int,
    bridge_bp: int,
    min_support: int,
    overlap_fraction: float,
    community_outdir: Path | None = None,
    max_plot_bins: int = 4000,
    min_enrichment_fold: float = 1.5,
    repeat_peak_fraction: float = 1.0,
    radius: int = 8,
    direction_lambda: float = 0.5,
    community_signal_quantile: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    resolutions = select_resolutions_for_chromosome(
        chrom_size_bp, available, requested, max_resolutions, max_bins
    )
    total_runs = len(resolutions) * len(gammas)
    worker_count = min(max(1, workers), total_runs)
    threads_per_worker = max(1, (os.cpu_count() or worker_count) // max(worker_count, 1))
    print(
        f"[INFO] {chrom}: resolutions={resolutions} workers={worker_count}",
        flush=True,
    )

    frames = []
    segment_tables: dict[tuple[int, float], pd.DataFrame] = {}
    successful_resolutions: set[int] = set()
    tasks = [
        (run_number, resolution, gamma)
        for run_number, (resolution, gamma) in enumerate(
            ((r, g) for r in resolutions for g in gammas), start=1
        )
    ]

    def _consume(completed, resolution, gamma, result_or_exc):
        if isinstance(result_or_exc, Exception):
            print(
                f"[{completed}/{total_runs}] {chrom} r={resolution} skipped: {result_or_exc}",
                flush=True,
            )
            return
        resolution, gamma, nodes, edges, candidates, segments = result_or_exc
        print(
            f"[{completed}/{total_runs}] {chrom} r={resolution}: "
            f"graph={nodes}/{edges} candidates={len(candidates)}",
            flush=True,
        )
        frames.append(candidates)
        segment_tables[(resolution, gamma)] = segments
        successful_resolutions.add(resolution)

    # serial scales when worker_count==1 (avoids nested ProcessPool under --jobs)
    if worker_count == 1:
        for completed, (run_number, resolution, gamma) in enumerate(tasks, start=1):
            try:
                result = analyze_scale(
                    mcool_path,
                    chrom,
                    chrom_size_bp,
                    resolution,
                    gamma,
                    edge_quantile,
                    balance,
                    seed + run_number,
                    min_segment_bp,
                    bridge_bp,
                    threads_per_worker,
                    radius,
                    direction_lambda,
                    repeat_peak_fraction,
                    community_signal_quantile,
                )
                _consume(completed, resolution, gamma, result)
            except (RuntimeError, ValueError, KeyError, OSError) as exc:
                _consume(completed, resolution, gamma, exc)
    else:
        with ProcessPoolExecutor(
            max_workers=worker_count, mp_context=mp.get_context("spawn")
        ) as executor:
            future_to_scale = {
                executor.submit(
                    analyze_scale,
                    mcool_path,
                    chrom,
                    chrom_size_bp,
                    resolution,
                    gamma,
                    edge_quantile,
                    balance,
                    seed + run_number,
                    min_segment_bp,
                    bridge_bp,
                    threads_per_worker,
                    radius,
                    direction_lambda,
                    repeat_peak_fraction,
                    community_signal_quantile,
                ): (resolution, gamma)
                for run_number, resolution, gamma in tasks
            }
            for completed, future in enumerate(as_completed(future_to_scale), start=1):
                resolution, gamma = future_to_scale[future]
                try:
                    _consume(completed, resolution, gamma, future.result())
                except (RuntimeError, ValueError, KeyError, OSError) as exc:
                    _consume(completed, resolution, gamma, exc)

    if not frames:
        raise RuntimeError(f"all analysis scales failed for {chrom}")

    nonempty = [f for f in frames if not f.empty]
    if nonempty:
        all_candidates = pd.concat(nonempty, ignore_index=True)
    else:
        all_candidates = pd.DataFrame(columns=["chrom"] + CANDIDATE_COLUMNS)
    if "chrom" not in all_candidates.columns:
        all_candidates.insert(0, "chrom", chrom)
    consensus = build_consensus(
        all_candidates.drop(columns=["chrom"], errors="ignore"),
        min_support,
        overlap_fraction,
    )
    if not consensus.empty:
        consensus.insert(0, "chrom", chrom)
    else:
        consensus = pd.DataFrame(
            columns=["chrom"]
            + [
                "start_bp",
                "end_bp",
                "length_bp",
                "support_scales",
                "support_resolutions",
                "partner_start_bp",
                "partner_end_bp",
                "distance_bp",
                "distance_percent",
            ]
        )
    print(
        f"[INFO] {chrom}: candidates={len(all_candidates)} "
        f"consensus={len(consensus)} (min-support={min_support})",
        flush=True,
    )

    # community report shows Louvain consensus BEFORE contact enrichment
    if community_outdir is not None and segment_tables:
        community_key = min(
            segment_tables, key=lambda key: (key[0], abs(key[1] - 1.0))
        )
        community_resolution, _ = community_key
        out_png = (
            Path(community_outdir)
            / f"multiscale_report_{safe_path_component(chrom)}.png"
        )
        try:
            plot_community_report(
                mcool_path,
                chrom,
                chrom_size_bp,
                available,
                consensus,
                segment_tables[community_key],
                community_resolution,
                balance,
                max_plot_bins,
                out_png,
            )
        except (RuntimeError, ValueError, KeyError, OSError, MemoryError) as exc:
            print(f"[WARN] {chrom}: community report failed: {exc}", flush=True)

    # final TRA judgment: require candidate×partner >> flanking communities
    scored_raw, filtered = filter_consensus_by_enrichment(
        mcool_path,
        chrom,
        chrom_size_bp,
        consensus,
        segment_tables,
        available,
        min_enrichment_fold,
        balance,
        repeat_peak_fraction,
    )
    print(
        f"[INFO] {chrom}: enrichment {len(filtered)}/{len(scored_raw)} kept "
        f"(fold>{min_enrichment_fold:g})",
        flush=True,
    )
    return all_candidates, scored_raw, filtered


def _chrom_worker(
    payload: dict,
) -> tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, str | None]:
    """Top-level worker for chromosome-level ProcessPool."""
    chrom = payload["chrom"]
    try:
        candidates, scored_raw, filtered = run_one_chromosome(
            Path(payload["mcool_path"]),
            payload["available"],
            payload["requested"],
            payload["gammas"],
            chrom,
            payload["chrom_size_bp"],
            payload["max_resolutions"],
            payload["max_bins"],
            payload["workers"],
            payload["edge_quantile"],
            payload["balance"],
            payload["seed"],
            payload["min_segment_bp"],
            payload["bridge_bp"],
            payload["min_support"],
            payload["overlap_fraction"],
            community_outdir=(
                Path(payload["community_outdir"])
                if payload.get("community_outdir")
                else None
            ),
            max_plot_bins=payload["max_plot_bins"],
            min_enrichment_fold=payload["min_enrichment_fold"],
            repeat_peak_fraction=payload["repeat_peak_fraction"],
            radius=payload["radius"],
            direction_lambda=payload["direction_lambda"],
            community_signal_quantile=payload["community_signal_quantile"],
        )
        return chrom, candidates, scored_raw, filtered, None
    except RuntimeError as exc:
        empty = pd.DataFrame()
        return chrom, empty, empty, empty, str(exc)


def to_final_table(consensus: pd.DataFrame) -> pd.DataFrame:
    if consensus.empty:
        return pd.DataFrame(columns=FINAL_COLUMNS)
    rows = []
    for i, row in enumerate(consensus.itertuples(index=False), start=1):
        p0, p1 = int(row.partner_start_bp), int(row.partner_end_bp)
        rows.append(
            {
                "cluster_id": i,
                "chrom": str(row.chrom),
                "start": int(row.start_bp),
                "end": int(row.end_bp),
                "size": int(row.length_bp),
                "partner_start": p0,
                "partner_end": p1,
                "partner_size": max(0, p1 - p0),
                "scale_support": int(row.support_scales),
                "call_support": int(row.support_scales),
                "resolutions": str(row.support_resolutions),
                "distance_bp": float(row.distance_bp),
                "distance_percent": float(row.distance_percent),
                "cross_mean": float(getattr(row, "cross_mean", float("nan"))),
                "neighbor_mean": float(getattr(row, "neighbor_mean", float("nan"))),
                "enrichment_fold": float(getattr(row, "enrichment_fold", float("nan"))),
                "repeat_flag": str(getattr(row, "repeat_flag", "ok")),
            }
        )
    return pd.DataFrame(rows)[FINAL_COLUMNS]


# ---- Hi-C images (INV-style) ----


def heatmap_limits(matrix, upper=99.5, ignore_diag_bins=0):
    work = np.asarray(matrix, dtype=float)
    if ignore_diag_bins > 0 and work.ndim == 2 and work.shape[0] == work.shape[1]:
        mask = np.ones(work.shape, dtype=bool)
        for k in range(-ignore_diag_bins, ignore_diag_bins + 1):
            idx = np.arange(max(0, -k), min(work.shape[0], work.shape[0] - k))
            mask[idx, idx + k] = False
        positive = work[mask & np.isfinite(work) & (work > 0)]
    else:
        positive = work[np.isfinite(work) & (work > 0)]
    if not len(positive):
        return 0.0, 1.0
    upper_v = float(np.percentile(positive, upper))
    return 0.0, upper_v if upper_v > 0 else 1.0


def fetch_log1p(clr, region):
    mat = clr.matrix(balance=False).fetch(region)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    return np.log1p(np.maximum(mat, 0.0))


def draw_local(ax, matrix):
    n = int(min(matrix.shape[:2]))
    band = max(3, n // 12)
    vmin, vmax = heatmap_limits(matrix, 99.5, ignore_diag_bins=band)
    vmax = max(vmax * 1.5, 1e-6)
    return ax.imshow(
        matrix,
        cmap="Reds",
        origin="upper",
        norm=PowerNorm(gamma=1.6, vmin=vmin, vmax=vmax, clip=True),
        interpolation="bilinear",
        extent=(0, n, n, 0),
    )


def style_axes(ax, origin_bp, resolution, ylabel=True):
    def _fmt(v, _):
        mb = (origin_bp + v * resolution) / 1e6
        return f"{mb:.1f}" if mb < 100 else f"{mb:.0f}"

    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    tick = FuncFormatter(_fmt)
    ax.xaxis.set_major_formatter(tick)
    ax.yaxis.set_major_formatter(tick)
    ax.set_xlabel("Mb", fontsize=9)
    ax.set_ylabel("Mb" if ylabel else "", fontsize=9)
    ax.tick_params(labelsize=7)


def mark_box(ax, start_bin, end_bin, color="#00e5ff", lw=1.4):
    width = max(1.0, end_bin - start_bin)
    ax.add_patch(
        Rectangle(
            (start_bin, start_bin), width, width, fill=False, edgecolor=color, linewidth=lw
        )
    )


def mark_offdiag(ax, a0, a1, b0, b1, color="#00e5ff", lw=1.2):
    ax.add_patch(
        Rectangle(
            (b0, a0),
            max(1.0, b1 - b0),
            max(1.0, a1 - a0),
            fill=False,
            edgecolor=color,
            linewidth=lw,
            linestyle="--",
        )
    )
    ax.add_patch(
        Rectangle(
            (a0, b0),
            max(1.0, a1 - a0),
            max(1.0, b1 - b0),
            fill=False,
            edgecolor=color,
            linewidth=lw,
            linestyle="--",
        )
    )


def tra_plot_window(start, end, partner_start, partner_end, chrom_size, flank=2_000_000):
    left = min(int(start), int(partner_start))
    right = max(int(end), int(partner_end))
    ws = max(0, left - int(flank))
    we = min(int(chrom_size), right + int(flank))
    if we <= ws:
        we = min(int(chrom_size), ws + max(right - left, 1))
    return ws, we


def choose_local_plot_resolutions(available, region_bp, want=PLOT_RESOLUTIONS, max_bins=1200):
    ordered = [r for r in want if r in available] or sorted(available)
    kept = [r for r in ordered if 8 <= int(np.ceil(region_bp / r)) <= max_bins]
    if kept:
        return kept
    return [min(ordered, key=lambda r: abs(int(np.ceil(region_bp / r)) - 400))]


def collage_local(local_dir: Path, out_path: Path, pattern="TRA*.png", columns=3, pad=12):
    from PIL import Image

    paths = sorted(local_dir.glob(pattern))
    if not paths:
        return
    images = [Image.open(p).convert("RGB") for p in paths]
    n = len(images)
    columns = max(1, min(int(columns), n))
    rows = int(np.ceil(n / columns))
    cell_w = max(im.width for im in images)
    cell_h = max(im.height for im in images)
    canvas = Image.new(
        "RGB",
        (columns * cell_w + pad * (columns + 1), rows * cell_h + pad * (rows + 1)),
        (255, 255, 255),
    )
    for i, im in enumerate(images):
        r, c = divmod(i, columns)
        x = pad + c * (cell_w + pad) + (cell_w - im.width) // 2
        y = pad + r * (cell_h + pad) + (cell_h - im.height) // 2
        canvas.paste(im, (x, y))
        im.close()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, dpi=(150, 150))
    print(f"[INFO] local collage -> {out_path}", flush=True)


def plot_local_tras(mcool_path: Path, calls: pd.DataFrame, available: list[int], image_dir: Path):
    local_dir = image_dir / "local"
    local_dir.mkdir(parents=True, exist_ok=True)
    for old in local_dir.glob("*.png"):
        old.unlink()
    want_pool = [r for r in PLOT_RESOLUTIONS if r in available] or available
    coolers = {r: cooler.Cooler(f"{mcool_path}::/resolutions/{r}") for r in want_pool}
    base = coolers[want_pool[0]]

    for _, row in calls.iterrows():
        chrom = str(row["chrom"])
        start, end = int(row["start"]), int(row["end"])
        p0, p1 = int(row["partner_start"]), int(row["partner_end"])
        chrom_size = int(base.chromsizes[chrom])
        ws, we = tra_plot_window(start, end, p0, p1, chrom_size)
        panels = choose_local_plot_resolutions(want_pool, we - ws)
        matrices = [fetch_log1p(coolers[r], (chrom, ws, we)) for r in panels]
        n = len(panels)
        fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 4.4), squeeze=False)
        for col, (resolution, matrix) in enumerate(zip(panels, matrices)):
            ax = axes[0, col]
            draw_local(ax, matrix)
            a0, a1 = (start - ws) / resolution, (end - ws) / resolution
            b0, b1 = (p0 - ws) / resolution, (p1 - ws) / resolution
            mark_box(ax, a0, a1)
            mark_box(ax, b0, b1, color="#00bfa5", lw=1.2)
            mark_offdiag(ax, a0, a1, b0, b1)
            ax.set_title(f"{resolution / 1e3:.0f} kb", fontsize=12, pad=4, weight="bold")
            style_axes(ax, ws, resolution, ylabel=(col == 0))
            if col > 0:
                ax.tick_params(axis="y", labelleft=False)
        cid = int(row["cluster_id"])
        fig.suptitle(
            f"TRA{cid:04d} | {chrom}  {start/1e6:.2f}–{end/1e6:.2f} Mb  "
            f"partner {p0/1e6:.2f}–{p1/1e6:.2f} Mb\n"
            f"support: {int(row['scale_support'])} scales ({row['resolutions']})",
            fontsize=11,
            y=0.98,
        )
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.90])
        fig.subplots_adjust(wspace=0.22, top=0.86)
        out = local_dir / f"TRA{cid:04d}_{chrom}_{start}_{end}.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] {out.name}", flush=True)
    collage_local(local_dir, image_dir / "ALL_TRA_local.png")


def plot_overview(mcool_path: Path, calls: pd.DataFrame, available: list[int], image_dir: Path):
    chroms = list(dict.fromkeys(calls["chrom"].astype(str)))
    if not chroms:
        return
    resolution = 500_000 if 500_000 in available else max(available)
    clr = cooler.Cooler(f"{mcool_path}::/resolutions/{resolution}")
    cols = min(3, len(chroms))
    rows = int(np.ceil(len(chroms) / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(5.2 * cols, 4.8 * rows), squeeze=False, constrained_layout=True
    )
    for ax, chrom in zip(axes.flat, chroms):
        mat = fetch_log1p(clr, chrom)
        n = mat.shape[0]
        if n > 800:
            step = int(np.ceil(n / 800))
            mat = mat[::step, ::step]
            eff = resolution * step
        else:
            eff = resolution
        draw_local(ax, mat)
        for _, row in calls[calls["chrom"].astype(str) == chrom].iterrows():
            a0, a1 = int(row["start"]) / eff, int(row["end"]) / eff
            b0, b1 = int(row["partner_start"]) / eff, int(row["partner_end"]) / eff
            mark_box(ax, a0, a1, lw=1.0)
            mark_box(ax, b0, b1, color="#00bfa5", lw=1.0)
            mark_offdiag(ax, a0, a1, b0, b1, lw=0.8)
        ax.set_title(f"{chrom} | {resolution/1e3:.0f} kb", fontsize=11)
        style_axes(ax, 0, eff)
    for ax in axes.flat[len(chroms) :]:
        ax.axis("off")
    fig.suptitle(
        "Multi-scale supported translocations — Hi-C overview\n"
        "cyan=candidate  teal=partner  dashed=contact block",
        fontsize=13,
    )
    out = image_dir / "ALL_TRA_overview.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] overview -> {out}", flush=True)


def render_hic_images(
    mcool_path: Path,
    result: pd.DataFrame,
    available: list[int],
    outdir: Path,
    subdir: str = "HiC",
):
    image_dir = Path(outdir) / subdir
    image_dir.mkdir(parents=True, exist_ok=True)
    if result.empty:
        print(f"[INFO] no TRA calls; skip {subdir}", flush=True)
        return image_dir
    plot_overview(mcool_path, result, available, image_dir)
    plot_local_tras(mcool_path, result, available, image_dir)
    return image_dir


def run_pipeline(args: argparse.Namespace) -> pd.DataFrame:
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
        chrom_records = [(c, s) for c, s in chrom_records if c in wanted]
    if not chrom_records:
        raise RuntimeError("no chromosomes selected")

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    # drop leftovers from earlier layouts (raw calls, pre-filter images, community plots)
    for stale in (
        "HiC_images",
        "HiC_images_passed",
        "community",
        "HiC",
        "ALL_TRA_community_raw.tsv",
        "ALL_TRA_per_scale_candidates.tsv",
        "ALL_TRA_multiscale.tsv",
        "ALL_TRA.tsv",
    ):
        path = outdir / stale
        if path.is_dir():
            shutil.rmtree(path)
        elif path.is_file():
            path.unlink()
    balance = bool(args.balance)
    # avoid nested oversubscription: chrom-parallel → serial scales
    scale_workers = 1 if args.jobs > 1 else max(1, args.workers)
    print(
        f"[INFO] chroms={len(chrom_records)} jobs={args.jobs} "
        f"scale_workers={scale_workers} min-support={args.min_support} "
        f"min-enrichment-fold={args.min_enrichment_fold} balance={balance}",
        flush=True,
    )

    if args.plot_only:
        result = pd.read_csv(args.plot_only, sep="\t")
        need = {"cluster_id", "chrom", "start", "end", "partner_start", "partner_end", "scale_support", "resolutions"}
        missing = need - set(result.columns)
        if missing:
            raise SystemExit(f"{args.plot_only} missing: {sorted(missing)}")
        image_dir = render_hic_images(mcool_path, result, available, outdir)
        print(f"[DONE] Hi-C -> {image_dir}", flush=True)
        return result

    payloads = [
        {
            "mcool_path": str(mcool_path),
            "available": list(available),
            "requested": requested,
            "gammas": list(gammas),
            "chrom": chrom,
            "chrom_size_bp": chrom_size_bp,
            "max_resolutions": args.max_resolutions,
            "max_bins": args.max_bins,
            "workers": scale_workers,
            "edge_quantile": args.edge_quantile,
            "balance": balance,
            "seed": args.seed,
            "min_segment_bp": int(args.min_segment_kb * 1000),
            "bridge_bp": int(args.bridge_kb * 1000),
            "min_support": args.min_support,
            "overlap_fraction": args.overlap_fraction,
            "community_outdir": None,
            "max_plot_bins": args.max_plot_bins,
            "min_enrichment_fold": args.min_enrichment_fold,
            "repeat_peak_fraction": args.repeat_peak_fraction,
            "radius": args.radius,
            "direction_lambda": args.direction_lambda,
            "community_signal_quantile": args.community_signal_quantile,
        }
        for chrom, chrom_size_bp in chrom_records
    ]

    candidate_frames = []
    raw_frames = []
    filtered_frames = []
    chrom_jobs = min(args.jobs, len(payloads))
    with ProcessPoolExecutor(
        max_workers=chrom_jobs, mp_context=mp.get_context("spawn")
    ) as executor:
        futures = [executor.submit(_chrom_worker, payload) for payload in payloads]
        for future in as_completed(futures):
            chrom, candidates, scored_raw, filtered, err = future.result()
            if err:
                print(f"[SKIP] {chrom}: {err}", flush=True)
                continue
            if not candidates.empty:
                candidate_frames.append(candidates)
            if not scored_raw.empty:
                raw_frames.append(scored_raw)
            if not filtered.empty:
                filtered_frames.append(filtered)

    n_raw = 0
    if raw_frames:
        n_raw = sum(len(frame) for frame in raw_frames)

    # final TRA after neighbor-enrichment filter
    if filtered_frames:
        all_filt = pd.concat(filtered_frames, ignore_index=True)
        all_filt = all_filt.sort_values(["chrom", "start_bp", "end_bp"]).reset_index(
            drop=True
        )
        result = to_final_table(all_filt)
    else:
        result = pd.DataFrame(columns=FINAL_COLUMNS)
        print(
            f"[WARN] no TRA passed enrichment fold>{args.min_enrichment_fold:g}",
            flush=True,
        )

    result_path = outdir / "ALL_TRA.tsv"
    result.to_csv(result_path, sep="\t", index=False, float_format="%.6g")
    print(
        f"[INFO] final {len(result)}/{n_raw} events -> {result_path}",
        flush=True,
    )

    image_dir = render_hic_images(mcool_path, result, available, outdir)
    print(f"[DONE] results -> {outdir}", flush=True)
    print(f"[DONE] Hi-C -> {image_dir}", flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mcool", help="input .mcool file")
    parser.add_argument(
        "--balance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use ICE-balanced contacts (default: --no-balance, raw counts)",
    )
    parser.add_argument("--outdir", default="TRA_results")
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="parallel chromosome workers (INV-style)",
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=1,
        help="minimum distinct resolutions for community-stage consensus (default: 1)",
    )
    parser.add_argument(
        "--min-enrichment-fold",
        type=float,
        default=1.5,
        help=(
            "final TRA gate: candidate×partner mean contact must be > this "
            "fold vs candidate×flanking communities (default: 1.5)"
        ),
    )
    parser.add_argument(
        "--chromosomes",
        default=None,
        help="comma-separated chromosome names; default = all",
    )
    parser.add_argument("--resolutions", default="auto")
    parser.add_argument("--gammas", default="1.0")
    parser.add_argument("--max-resolutions", type=int, default=4)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="parallel scale processes when --jobs 1; ignored if jobs>1",
    )
    parser.add_argument("--max-bins", type=int, default=15000)
    parser.add_argument(
        "--max-plot-bins",
        type=int,
        default=4000,
        help="max bins for community-report chrom heatmap",
    )
    parser.add_argument(
        "--repeat-peak-fraction",
        type=float,
        default=1.0,
        help=(
            "a community is a repeat when the sum of its mean contacts to all "
            "other communities is still below this fraction of the 75th "
            "percentile of per-community neighbor-contact sums (default: 1)"
        ),
    )
    parser.add_argument(
        "--community-signal-quantile",
        type=float,
        default=0.05,
        help=(
            "treat a community as micro when its mean diagonal contact is "
            "below this quantile of the chromosome-wide positive diagonal "
            "(default: 0.05)"
        ),
    )
    parser.add_argument("--min-segment-kb", type=float, default=2000.0)
    parser.add_argument("--bridge-kb", type=float, default=500.0)
    parser.add_argument(
        "--edge-quantile",
        type=float,
        default=0.90,
        help=(
            "keep raw contacts at or above this quantile as graph edges "
            "(0.90 = top 10%; default: 0.90)"
        ),
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=8,
        help="unused (legacy INV edge formula); kept for CLI compatibility",
    )
    parser.add_argument(
        "--direction-lambda",
        type=float,
        default=0.5,
        help="unused (legacy INV edge formula); kept for CLI compatibility",
    )
    parser.add_argument("--overlap-fraction", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--plot-only",
        default=None,
        metavar="TSV",
        help="skip detection; replot Hi-C from ALL_TRA.tsv",
    )
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not 0 < args.edge_quantile < 1:
        parser.error("--edge-quantile must be between 0 and 1")
    if not 0 < args.overlap_fraction <= 1:
        parser.error("--overlap-fraction must be in (0, 1]")
    if args.min_support < 1:
        parser.error("--min-support must be >= 1")
    if args.min_enrichment_fold <= 0:
        parser.error("--min-enrichment-fold must be > 0")
    if args.repeat_peak_fraction <= 0:
        parser.error("--repeat-peak-fraction must be > 0")
    if not 0 <= args.community_signal_quantile < 1:
        parser.error("--community-signal-quantile must be in [0, 1)")
    if args.radius < 1:
        parser.error("--radius must be >= 1")
    if args.direction_lambda < 0:
        parser.error("--direction-lambda must be >= 0")
    if (
        args.jobs < 1
        or args.workers < 1
        or args.max_resolutions < 1
        or args.max_bins < 1
        or args.max_plot_bins < 1
    ):
        parser.error("jobs/workers/resolution limits must be positive")
    if args.min_segment_kb <= 0 or args.bridge_kb < 0:
        parser.error("segment sizes must be valid")
    if args.plot_only and not Path(args.plot_only).expanduser().is_file():
        parser.error(f"file does not exist: {args.plot_only}")


def _selfcheck() -> None:
    # ponytail: window + consensus + enrichment fold gate
    ws, we = tra_plot_window(100, 200, 50, 80, 1000, flank=10)
    assert (ws, we) == (40, 210), (ws, we)
    empty = build_consensus(pd.DataFrame(columns=CANDIDATE_COLUMNS), min_support=2, overlap_fraction=0.5)
    assert empty.empty
    assert _interval_overlap(0, 10, 5, 15) == 5
    # fold = cross/neighbor
    assert (0.9 / 0.3) > 1.5
    assert (0.45 / 0.3) == 1.5  # equal must DROP under strict >
    assert (0.44 / 0.3) < 1.5
    # sum-of-external-means vs P75(neighbor sums): only the quiet island is weak
    quiet = np.array(
        [
            [10.0, 0.05, 5.0, 6.0],
            [0.05, 0.1, 0.05, 0.05],
            [5.0, 0.05, 10.0, 8.0],
            [6.0, 0.05, 8.0, 10.0],
        ]
    )
    mask, pairs = repeat_mask_from_means(quiet, 1.0)
    assert mask.tolist() == [False, True, False, False], mask
    assert pairs == {(0, 2)}, pairs
    strong = np.array(
        [
            [10.0, 9.0, 8.0, 2.0],
            [9.0, 10.0, 8.0, 7.0],
            [8.0, 8.0, 10.0, 9.0],
            [2.0, 7.0, 9.0, 10.0],
        ]
    )
    mask2, pairs2 = repeat_mask_from_means(strong, 1.0)
    assert not mask2.any(), mask2
    assert pairs2 == set()
    # s = raw/median; D=0 on empty profiles, so weight>1 keeps only the enriched pair
    erows = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5], dtype=np.int64)
    ecols = np.array([1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15], dtype=np.int64)
    eraw = np.array([2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 3], dtype=float)
    kept_r, kept_c, kept_w = tra_edge_weights(erows, ecols, eraw, 16, radius=2, direction_lambda=0.5)
    assert int(((kept_c - kept_r) == 1).sum()) == 5
    assert int(((kept_c - kept_r) > 2).sum()) == 0
    assert np.all(kept_w > 0)
    weak_labels = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    weak_bins = [(0, 2), (2, 4), (4, 6), (6, 8)]
    weak_means = np.array(
        [
            [5.0, 0.05, 6.0, 5.0],
            [0.05, 0.1, 0.05, 0.05],
            [6.0, 0.05, 5.0, 8.0],
            [5.0, 0.05, 8.0, 5.0],
        ]
    )
    weak_mask, _ = repeat_mask_from_means(weak_means, 1.0)
    assert weak_mask.tolist() == [False, True, False, False], weak_mask
    merged, n_abs = absorb_weak_runs(weak_labels, weak_means, weak_bins, weak_mask)
    assert n_abs == 1
    assert merged.tolist() == [0, 0, 0, 0, 2, 2, 3, 3], merged
    # micro-run: 1-bin islands with a high single link are not weak by sum, but
    # the micro rule still folds them into the absorbable run.
    micro_labels = np.array([0, 0, 0, 1, 2, 3, 4, 0, 0, 0])
    micro_bins = [(0, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 10)]
    micro_means = np.array(
        [
            [5.0, 0.05, 0.05, 0.05, 0.05, 4.0],
            [0.05, 0.1, 1.0, 0.0, 0.0, 0.05],
            [0.05, 1.0, 0.1, 0.0, 0.0, 0.05],
            [0.05, 0.0, 0.0, 0.1, 0.02, 0.05],
            [0.05, 0.0, 0.0, 0.02, 0.1, 0.05],
            [4.0, 0.05, 0.05, 0.05, 0.05, 5.0],
        ]
    )
    micro_weak, _ = repeat_mask_from_means(micro_means, 1.0)
    micro_mask = micro_weak | np.array([False, True, True, True, True, False])
    merged_m, n_m = absorb_weak_runs(micro_labels, micro_means, micro_bins, micro_mask)
    assert n_m == 4
    assert merged_m.tolist() == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], merged_m
    # sandwich: same label split by a non-weak mid-sized gap → absorb in step 2
    sand_labels = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0])
    sand_bins = [(0, 3), (3, 6), (6, 9)]
    sand_mask = sandwich_absorbable_mask(sand_labels, sand_bins, max_gap_bins=4)
    assert sand_mask.tolist() == [False, True, False], sand_mask
    sand_means = np.array(
        [
            [5.0, 0.5, 4.0],
            [0.5, 5.0, 0.5],
            [4.0, 0.5, 5.0],
        ]
    )
    merged_s, n_s = absorb_weak_runs(sand_labels, sand_means, sand_bins, sand_mask)
    assert n_s == 1
    assert merged_s.tolist() == [0, 0, 0, 0, 0, 0, 0, 0, 0], merged_s
    # gap larger than ceiling stays
    assert not sandwich_absorbable_mask(sand_labels, sand_bins, max_gap_bins=2).any()
    # micro secondary copy stays sandwich-absorbable at the mask level; only a
    # type-2 island contact (or non-micro secondary) keeps it in absorb.
    tra_labels = np.array([0, 0, 2, 0, 0, 0, 2, 2, 2])
    tra_bins = [(0, 2), (2, 3), (3, 6), (6, 9)]
    tra_mask = sandwich_absorbable_mask(tra_labels, tra_bins, max_gap_bins=4)
    assert tra_mask[1], tra_mask
    # non-micro secondary copy is not sandwich-absorbable
    tra2_bins = [(0, 2), (2, 5), (5, 8), (8, 12)]
    tra2_labels = np.array([0, 0, 2, 2, 2, 0, 0, 0, 2, 2, 2, 2])
    tra2_mask = sandwich_absorbable_mask(tra2_labels, tra2_bins, max_gap_bins=4)
    assert not tra2_mask[1], tra2_mask
    # two cliques joined by a 5 Mb-scale block of bins get split by the second Louvain
    split_labels = np.zeros(40, dtype=np.int64)
    split_rows = np.array([0, 1, 2, 20, 21, 22], dtype=np.int64)
    split_cols = np.array([1, 2, 0, 21, 22, 20], dtype=np.int64)
    split_w = np.ones(6, dtype=float)
    split_out, n_split = resplit_large_blocks(
        split_labels, split_rows, split_cols, split_w, resolution=200_000, gamma=1.0, seed=1, min_bp=1
    )
    assert n_split == 1
    assert split_out[0] == split_out[1] == split_out[2]
    assert split_out[20] == split_out[21] == split_out[22]
    assert split_out[0] != split_out[20]
    # same-label flanks with a short mid gap → sandwich absorb into label 0
    import scipy.sparse as sp
    hole = np.zeros((10, 10), dtype=float)
    hole[0:3, 0:3] = 5
    hole[6:10, 6:10] = 5
    hole[0:3, 6:10] = hole[6:10, 0:3] = 4
    hole[3:6, 3:6] = 0.1
    hole_labels = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0])
    merged_h, n_h = absorb_weak_communities(
        hole_labels, sp.csr_matrix(hole), 1.0, max_micro_bins=1, max_sandwich_bins=4
    )
    assert n_h >= 1
    assert set(merged_h.tolist()) == {0}, merged_h
    # type-2 island: short mid community contacts a far block harder than flanks
    # bins: 0=flank(3), 1=island(1), 2=flank(3), 3=partner(3)
    isl_labels = np.array([0, 0, 0, 1, 2, 2, 2, 3, 3, 3])
    isl_bins = [(0, 3), (3, 4), (4, 7), (7, 10)]
    isl_means = np.array(
        [
            [5.0, 0.4, 0.2, 0.1],
            [0.4, 5.0, 0.3, 1.2],
            [0.2, 0.3, 5.0, 0.2],
            [0.1, 1.2, 0.2, 5.0],
        ]
    )
    pairs = island_tra_pairs(isl_labels, isl_bins, isl_means, min_fold=1.5)
    assert pairs == [(1, 3)], pairs
    # unique-label island must survive sandwich absorb
    isl_mat = np.zeros((10, 10), dtype=float)
    isl_mat[0:3, 0:3] = 5
    isl_mat[4:7, 4:7] = 5
    isl_mat[7:10, 7:10] = 5
    isl_mat[0:3, 4:7] = isl_mat[4:7, 0:3] = 0.4
    isl_mat[3:4, 0:3] = isl_mat[0:3, 3:4] = 0.4
    isl_mat[3:4, 4:7] = isl_mat[4:7, 3:4] = 0.3
    isl_mat[3:4, 7:10] = isl_mat[7:10, 3:4] = 1.2
    isl_mat[3, 3] = 5
    merged_i, _ = absorb_weak_communities(
        isl_labels, sp.csr_matrix(isl_mat), 1.0, max_micro_bins=0, max_sandwich_bins=4
    )
    assert int(merged_i[3]) == 1, merged_i
    # low diagonal-mean community is micro vs chrom-wide diagonal quantile
    low = np.zeros((12, 12), dtype=float)
    low[0:4, 0:4] = 20
    low[8:12, 8:12] = 20
    low[0:4, 8:12] = low[8:12, 0:4] = 3
    low[4:8, 4:8] = 0.5  # diag mean 0.5 << flank diag 20
    low_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
    merged_low, n_low = absorb_weak_communities(
        low_labels,
        sp.csr_matrix(low),
        1.0,
        max_micro_bins=1,
        max_sandwich_bins=0,
        community_signal_quantile=0.5,
    )
    assert n_low >= 1
    assert set(merged_low[4:8].tolist()) == {0}, merged_low


def main() -> None:
    _selfcheck()
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)
    run_pipeline(args)


if __name__ == "__main__":
    main()

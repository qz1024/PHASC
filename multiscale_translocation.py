#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multi-resolution x multi-Louvain TRA detection (standalone).

One script, no other project imports. Flow:

  for Hi-C resolution in {50,100,200,500} kb:
      for Louvain/PLM gamma in {0.5, 0.75, 1.0, 1.25, 1.5, 2.0}:
          cluster on top-10% contacts -> candidates -> clear-insert ILP
  merge across (resolution, gamma)
      - reciprocal overlap or smaller-contained-in-larger
      - if still overlapping, keep higher scale_support (then size)

Example:
  python tra_multiscale.py \
    --mcool sample.mcool --chrom chr3B \
    --outdir TRA_out --skip-plot

Deps: cooler, networkit, numpy, pandas, pulp, matplotlib (+ GLPK/CBC).
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import random
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cooler
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkit as nk
import numpy as np
import pandas as pd
import pulp
from matplotlib.colors import LogNorm

# =============================================================================
# Community clustering
# =============================================================================

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


def detect_scale_candidates(
    labels: np.ndarray,
    resolution: int,
    gamma: float,
    chrom_size_bp: int,
    min_segment_bp: int,
    bridge_bp: int,
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


def _plm_seed(base_seed: int, resolution: int = 0, gamma: float = 1.0, salt: int = 0) -> int:
    """Deterministic per-scale seed (avoid cross-talk between scales)."""
    return int(base_seed) + int(resolution) + int(round(float(gamma) * 1000)) + int(salt)


def _graph_from_coo_edges(
    n: int,
    rows: np.ndarray,
    cols: np.ndarray,
    weights: np.ndarray,
) -> "nk.graph.Graph":
    """Build an undirected weighted graph with stable edge insertion order."""
    graph = nk.graph.Graph(int(n), weighted=True, directed=False)
    if rows.size == 0:
        return graph
    order = np.lexsort((cols, rows))
    for i in order:
        graph.addEdge(int(rows[i]), int(cols[i]), float(weights[i]))
    return graph


def _run_plm(
    graph: "nk.graph.Graph",
    gamma: float,
    seed: int,
    par: str = "balanced",
) -> np.ndarray:
    """Louvain/PLM community detection (default ``par='balanced'`` = parallel)."""
    nk.setSeed(int(seed), True)
    algorithm = nk.community.PLM(
        graph, refine=True, gamma=float(gamma), par=str(par)
    )
    algorithm.run()
    return np.asarray(algorithm.getPartition().getVector(), dtype=np.int64)


def _balanced_sparse_matrix(clr: cooler.Cooler, chrom: str, balance: bool):
    try:
        return clr.matrix(balance=balance, sparse=True).fetch(chrom), balance
    except (ValueError, KeyError) as exc:
        if not balance:
            raise
        print(f"  warning: balanced matrix unavailable ({exc}); using raw counts")
        return clr.matrix(balance=False, sparse=True).fetch(chrom), False


def build_contact_graph(
    mcool_path: Path,
    chrom: str,
    resolution: int,
    edge_quantile: float,
    balance: bool,
) -> tuple["nk.graph.Graph", int, int]:
    """Build the thresholded Hi-C graph once per resolution (reuse across trials)."""
    clr = cooler.Cooler(f"{mcool_path}::/resolutions/{resolution}")
    matrix, _ = _balanced_sparse_matrix(clr, chrom, balance)
    coo = matrix.tocoo()
    valid = (coo.row < coo.col) & np.isfinite(coo.data) & (coo.data > 0)
    rows, cols = coo.row[valid], coo.col[valid]
    weights = np.asarray(coo.data[valid], dtype=float)
    if weights.size == 0:
        raise RuntimeError(f"no positive contacts at resolution {resolution}")
    threshold = float(np.quantile(weights, edge_quantile))
    keep = weights >= threshold
    graph = _graph_from_coo_edges(
        matrix.shape[0], rows[keep], cols[keep], weights[keep]
    )
    if graph.numberOfEdges() == 0:
        raise RuntimeError(f"contact threshold removed every edge at resolution {resolution}")
    return graph, graph.numberOfNodes(), graph.numberOfEdges()


def community_labels_for_scale(
    mcool_path: Path,
    chrom: str,
    resolution: int,
    gamma: float,
    edge_quantile: float,
    balance: bool,
    seed: int,
    plm_par: str = "balanced",
    seed_salt: int = 0,
    graph: "nk.graph.Graph | None" = None,
) -> tuple[np.ndarray, int, int]:
    if graph is None:
        graph, nodes, edges = build_contact_graph(
            mcool_path, chrom, resolution, edge_quantile, balance
        )
    else:
        nodes, edges = graph.numberOfNodes(), graph.numberOfEdges()
    scale_seed = _plm_seed(
        seed, resolution=resolution, gamma=gamma, salt=int(seed_salt)
    )
    labels = _run_plm(graph, gamma=gamma, seed=scale_seed, par=plm_par)
    return labels, nodes, edges


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


def block_mean_contact(
    clr: cooler.Cooler, chrom: str, a0: int, a1: int, b0: int, b1: int
) -> float:
    """Mean raw contact in rectangle [a0,a1) x [b0,b1)."""
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
    """Compare candidatexpartner contact to candidatexflanking communities."""
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
    plm_par: str = "balanced",
    seed_salt: int = 0,
    graph: "nk.graph.Graph | None" = None,
) -> tuple[int, float, int, int, pd.DataFrame, pd.DataFrame]:
    n_threads = 1 if str(plm_par) == "none" else max(1, networkit_threads)
    if hasattr(nk, "setNumberOfThreads"):
        nk.setNumberOfThreads(n_threads)
    labels, nodes, edges = community_labels_for_scale(
        mcool_path,
        chrom,
        resolution,
        gamma,
        edge_quantile,
        balance,
        seed,
        plm_par=plm_par,
        seed_salt=seed_salt,
        graph=graph,
    )
    candidates, segments = detect_scale_candidates(
        labels, resolution, gamma, chrom_size_bp, min_segment_bp, bridge_bp
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

# =============================================================================
# Clear-insert restore + multi-resolution merge
# =============================================================================

# project root (.../10.sim_inv_tra_compare) for default mcool/truth paths only
ROOT = Path(__file__).resolve().parents[1]
RESOLUTIONS_DEFAULT = (50_000, 100_000, 200_000, 500_000)
GAMMAS_DEFAULT = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)

# ---------------------------------------------------------------------------
# Clear-insert ILP restore
# ---------------------------------------------------------------------------

def _default_ilp_solver():
    """Pick an available MIP solver (GLPK / CBC / HiGHS)."""
    candidates = []
    for name in ("GLPK_CMD", "COIN_CMD", "HiGHS_CMD"):
        if hasattr(pulp, name):
            candidates.append(getattr(pulp, name))
    last_exc: Exception | None = None
    for factory in candidates:
        try:
            return factory(msg=False)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            continue
    if last_exc is not None:
        raise RuntimeError(f"no PuLP MIP solver available: {last_exc}") from last_exc
    raise RuntimeError("no PuLP MIP solver available (install glpk or coincbc)")


def _overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def _gap_of(j0: int, j1: int, p0: int, p1: int) -> tuple[int, int] | None:
    """Open genomic gap between a call arm and its partner, if separable."""
    if p1 <= j0:
        return p1, j0
    if j1 <= p0:
        return j1, p0
    return None


def _subtract_blocks(
    left: int, right: int, blocks: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Subtract merged blocks from [left, right); return residual intervals."""
    if right <= left:
        return []
    segs = [(left, right)]
    for b0, b1 in sorted(blocks):
        if b1 <= b0:
            continue
        nxt: list[tuple[int, int]] = []
        for s0, s1 in segs:
            if b1 <= s0 or b0 >= s1:
                nxt.append((s0, s1))
                continue
            if s0 < b0:
                nxt.append((s0, min(b0, s1)))
            if b1 < s1:
                nxt.append((max(b1, s0), s1))
        segs = [(a, b) for a, b in nxt if b > a]
    return segs


def _merge_interval_list(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    xs = sorted(intervals)
    out = [xs[0]]
    for a, b in xs[1:]:
        la, lb = out[-1]
        if a <= lb:
            out[-1] = (la, max(lb, b))
        else:
            out.append((a, b))
    return out


def clearly_fills_gap(
    gl: int,
    gr: int,
    i0: int,
    i1: int,
    max_residual_frac: float = 0.25,
    min_cover_frac: float = 0.5,
) -> bool:
    """True if [i0,i1] clearly plugs [gl,gr] so flanks can join."""
    if gr <= gl or i1 <= i0:
        return False
    ov = _overlap(gl, gr, i0, i1)
    gap = gr - gl
    if ov < min_cover_frac * gap:
        return False
    # residual holes inside the gap on either side of the insert
    res_left = max(0, min(max(i0, gl), gr) - gl) if i0 > gl else 0
    res_right = max(0, gr - max(min(i1, gr), gl)) if i1 < gr else 0
    # clamp: if insert starts inside gap
    if i0 >= gl:
        res_left = i0 - gl
    else:
        res_left = 0
    if i1 <= gr:
        res_right = gr - i1
    else:
        res_right = 0
    spill = max(0, gl - i0) + max(0, i1 - gr)
    if spill > 0.25 * gap:
        return False
    return (res_left + res_right) <= max_residual_frac * gap


def load_calls(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    rename = {
        "start_bp": "start",
        "end_bp": "end",
        "length_bp": "size",
        "main_start_bp": "partner_start",
        "main_end_bp": "partner_end",
        "partner_start_bp": "partner_start",
        "partner_end_bp": "partner_end",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    need = ["chrom", "start", "end", "partner_start", "partner_end"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"missing columns {missing} in {path}")
    if "cluster_id" not in df.columns:
        df.insert(0, "cluster_id", range(1, len(df) + 1))
    if "enrichment_fold" not in df.columns:
        df["enrichment_fold"] = 1.0
    if "scale_support" not in df.columns:
        df["scale_support"] = 1
    if "size" not in df.columns:
        df["size"] = df["end"] - df["start"]
    if "partner_size" not in df.columns:
        df["partner_size"] = df["partner_end"] - df["partner_start"]
    df = df.sort_values(["chrom", "start", "end"]).reset_index(drop=True)
    df["idx"] = df.index
    return df


def gain(row: pd.Series) -> float:
    fold = float(row["enrichment_fold"])
    support = float(row["scale_support"])
    size_term = math.log10(max(int(row["size"]), 1)) * 0.15
    partner_term = math.log10(max(int(row["partner_size"]), 1)) * 0.1
    if fold >= 1.0:
        enrich_term = math.log2(1.0 + fold) * 3.0
    else:
        enrich_term = math.log2(1.0 + max(fold, 1e-6)) * 0.5 - 2.0
    return enrich_term + support + size_term + partner_term


def build_conflicts(df: pd.DataFrame, min_overlap_frac: float = 0.3) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    rows = df.to_dict("records")
    for a in range(len(rows)):
        for b in range(a + 1, len(rows)):
            ra, rb = rows[a], rows[b]
            if ra["chrom"] != rb["chrom"]:
                continue
            ov = _overlap(int(ra["start"]), int(ra["end"]), int(rb["start"]), int(rb["end"]))
            if ov <= 0:
                continue
            min_len = min(int(ra["end"]) - int(ra["start"]), int(rb["end"]) - int(rb["start"]))
            if min_len > 0 and ov / min_len >= min_overlap_frac:
                edges.append((a, b))
    return edges


def pair_gap_info(row: dict) -> tuple[int, int, int, int, int] | None:
    """Return (gap_l, gap_r, left_size, right_size, gap_size) for a call/partner pair."""
    j0, j1 = int(row["start"]), int(row["end"])
    p0, p1 = int(row["partner_start"]), int(row["partner_end"])
    gap = _gap_of(j0, j1, p0, p1)
    if gap is None:
        return None
    gl, gr = gap
    if gr <= gl:
        return None
    if j1 <= p0:
        left_size, right_size = j1 - j0, p1 - p0
    else:
        left_size, right_size = p1 - p0, j1 - j0
    return gl, gr, left_size, right_size, gr - gl


def is_near_arm_pair(row: dict, min_gap_bp: int = 500_000) -> bool:
    """Call and partner are adjacent with a small middle gap -> outermost flank pair.

    Either arm of such a pair is an outermost flank after the gap is filled,
    not a foreign insert. Never accept these as true TRA.
    """
    info = pair_gap_info(row)
    if info is None:
        return False
    _gl, _gr, left_size, right_size, gap_size = info
    return (
        gap_size >= min_gap_bp
        and gap_size < left_size
        and gap_size < right_size
        and left_size > 0
        and right_size > 0
    )


def has_both_flanks(row: dict, min_gap_bp: int = 500_000, min_flank_bp: int = 500_000) -> bool:
    info = pair_gap_info(row)
    if info is None:
        return False
    _gl, _gr, left_size, right_size, gap_size = info
    return gap_size >= min_gap_bp and left_size >= min_flank_bp and right_size >= min_flank_bp


def same_label_community_gaps(
    df: pd.DataFrame,
    min_gap_bp: int = 500_000,
    min_flank_bp: int = 500_000,
) -> list[tuple[list[int], list[int], int, int, int, int, str, int]]:
    """Gaps between consecutive pieces of the same community label.

    Pieces = every candidate arm with that label PLUS its partner/main block
    (partner is the largest same-label segment and is often not a separate row).

    Returns
      (left_touching_idxs, right_touching_idxs, gap_l, gap_r,
       left_size, right_size, chrom, label)
    where *_touching_idxs are candidate row indices that overlap the flank
    piece (may be empty if the flank is only a partner/main).
    """
    if "label" not in df.columns:
        return []
    rows = df.to_dict("records")

    # (chrom, label) -> list of (start, end, source_idx_or_-1)
    pieces: dict[tuple[str, int], list[tuple[int, int, int]]] = {}
    for idx, r in enumerate(rows):
        try:
            lab = int(r["label"])
        except (TypeError, ValueError):
            continue
        if lab < 0:
            continue
        chrom = str(r["chrom"])
        key = (chrom, lab)
        pieces.setdefault(key, []).append(
            (int(r["start"]), int(r["end"]), idx)
        )
        # partner/main is same community; include even if not a candidate row
        p0, p1 = int(r["partner_start"]), int(r["partner_end"])
        if p1 > p0:
            pieces[key].append((p0, p1, -1))

    gaps: list[tuple[list[int], list[int], int, int, int, int, str, int]] = []
    for (chrom, lab), segs in pieces.items():
        # merge overlapping/abutting intervals; keep member idxs
        segs = sorted(segs, key=lambda t: (t[0], t[1]))
        merged: list[tuple[int, int, list[int]]] = []
        for s, e, src in segs:
            if not merged or s > merged[-1][1]:
                merged.append((s, e, [] if src < 0 else [src]))
            else:
                ps, pe, members = merged[-1]
                members = list(members)
                if src >= 0 and src not in members:
                    members.append(src)
                merged[-1] = (ps, max(pe, e), members)

        for a, b in zip(merged, merged[1:]):
            la0, la1, left_idxs = a
            rb0, rb1, right_idxs = b
            gl, gr = la1, rb0
            gap_size = gr - gl
            left_size = la1 - la0
            right_size = rb1 - rb0
            if (
                gap_size >= min_gap_bp
                and left_size >= min_flank_bp
                and right_size >= min_flank_bp
            ):
                gaps.append(
                    (
                        list(left_idxs),
                        list(right_idxs),
                        gl,
                        gr,
                        left_size,
                        right_size,
                        chrom,
                        lab,
                    )
                )
    return gaps


def build_clear_insert_relations(
    df: pd.DataFrame,
    merged_blocks: dict[str, list[tuple[int, int]]] | None = None,
    min_gap_bp: int = 500_000,
    min_flank_bp: int = 500_000,
    max_residual_frac: float = 0.25,
    min_cover_frac: float = 0.5,
    seed: int = 42,
) -> tuple[list[tuple[int, int, int, int]], list[dict]]:
    """Return (fill triples, mutual_records).

    i clearly fills a residual gap between two consecutive *same-label*
    community pieces. Mutual A<->B (A B C D -> A C B D: both views valid):
    **report both** inserters (do not drop either). Evaluation treats either
    (or both) as correct for the same TRA event.
    """
    merged_blocks = merged_blocks or {}
    fills: list[tuple[int, int, int, int]] = []
    # parallel meta: flank sizes of the seam each edge plugs
    fill_meta: list[tuple[int, int, int, int, int, int]] = []
    rows = df.to_dict("records")
    community_gaps = same_label_community_gaps(
        df, min_gap_bp=min_gap_bp, min_flank_bp=min_flank_bp
    )

    for left_idxs, right_idxs, gl, gr, ls, rs, chrom, flank_lab in community_gaps:
        residuals = _subtract_blocks(gl, gr, merged_blocks.get(chrom, []))
        residuals = [(a, b) for a, b in residuals if (b - a) >= min_gap_bp]
        if len(residuals) != 1:
            continue
        res_l, res_r = residuals[0]
        flank_set = set(left_idxs) | set(right_idxs)
        for i, ri in enumerate(rows):
            if i in flank_set or str(ri["chrom"]) != chrom:
                continue
            try:
                i_lab = int(ri["label"])
            except (TypeError, ValueError):
                i_lab = -1
            if i_lab >= 0 and i_lab == flank_lab:
                continue
            if is_near_arm_pair(ri, min_gap_bp=min_gap_bp):
                continue
            i0, i1 = int(ri["start"]), int(ri["end"])
            skip = False
            for j in flank_set:
                j0, j1 = int(rows[j]["start"]), int(rows[j]["end"])
                if _overlap(i0, i1, j0, j1) > 0.5 * min(
                    max(1, i1 - i0), max(1, j1 - j0)
                ):
                    skip = True
                    break
            if skip:
                continue
            if clearly_fills_gap(
                res_l,
                res_r,
                i0,
                i1,
                max_residual_frac=max_residual_frac,
                min_cover_frac=min_cover_frac,
            ):
                targets = list(flank_set)
                if not targets:
                    for k, rk in enumerate(rows):
                        try:
                            if (
                                str(rk["chrom"]) == chrom
                                and int(rk["label"]) == flank_lab
                                and k != i
                            ):
                                targets.append(k)
                        except (TypeError, ValueError):
                            continue
                if not targets:
                    continue
                for j in targets:
                    fills.append((i, j, res_l, res_r))
                    fill_meta.append((i, j, res_l, res_r, ls, rs))

    # Mutual A<->B: both views valid (A B C D -> A C B D). Keep BOTH.
    pair_set = {(i, j) for i, j, _a, _b in fills}
    mutual_pairs: list[tuple[int, int]] = []
    seen_pair: set[tuple[int, int]] = set()
    for i, j in pair_set:
        if (j, i) in pair_set:
            a, b = (i, j) if i < j else (j, i)
            if (a, b) not in seen_pair:
                seen_pair.add((a, b))
                mutual_pairs.append((a, b))

    def _best_flank_score(inserter: int, other: int) -> tuple[int, int, int, int, int]:
        best: tuple[int, int, int, int, int, int] | None = None
        for i, j, rl, rr, ls, rs in fill_meta:
            if i != inserter:
                continue
            score = max(ls, rs)
            boost = 1 if j == other else 0
            cand = (boost, score, ls, rs, rl, rr)
            if best is None or cand > best:
                best = cand
        if best is None:
            return 0, 0, 0, 0, 0
        _boost, score, ls, rs, gl, gr = best
        return score, ls, rs, gl, gr

    # seed retained for API compat; no longer used to drop one side
    _ = int(seed)
    mutual_records: list[dict] = []
    for a, b in mutual_pairs:
        sa, lsa, rsa, gla, gra = _best_flank_score(a, b)
        sb, lsb, rsb, glb, grb = _best_flank_score(b, a)
        for idx, score, ls, rs, gl, gr, other in (
            (a, sa, lsa, rsa, gla, gra, b),
            (b, sb, lsb, rsb, glb, grb, a),
        ):
            r = rows[idx]
            o = rows[other]
            mutual_records.append(
                {
                    "chrom": str(r["chrom"]),
                    "start": int(r["start"]),
                    "end": int(r["end"]),
                    "partner_start": int(r["partner_start"]),
                    "partner_end": int(r["partner_end"]),
                    "cluster_id": int(r.get("cluster_id", idx + 1)),
                    "label": int(r["label"]) if "label" in r else -1,
                    "enrichment_fold": float(r.get("enrichment_fold", 1.0)),
                    "size": int(r["end"]) - int(r["start"]),
                    "flank_left_bp": int(ls),
                    "flank_right_bp": int(rs),
                    "flank_score": int(score),
                    "gap_start": int(gl),
                    "gap_end": int(gr),
                    "other_start": int(o["start"]),
                    "other_end": int(o["end"]),
                    "other_cluster_id": int(o.get("cluster_id", other + 1)),
                    "mutual_decision": "keep",
                }
            )

    # Do not filter fills: both mutual inserters stay eligible for ILP accept.
    return fills, mutual_records


def _fill_quality(
    res_l: int, res_r: int, i0: int, i1: int
) -> tuple[float, float]:
    """Return (cover_frac, residual_frac) of insert [i0,i1] vs residual gap."""
    gap = res_r - res_l
    if gap <= 0 or i1 <= i0:
        return 0.0, 1.0
    ov = _overlap(res_l, res_r, i0, i1)
    cover = ov / gap
    res_left = max(0, i0 - res_l) if i0 > res_l else 0
    res_right = max(0, res_r - i1) if i1 < res_r else 0
    return cover, (res_left + res_right) / gap


def solve_round_ilp(
    df: pd.DataFrame,
    fill_edges: list[tuple[int, int, int, int]],
    conf_edges: list[tuple[int, int]],
    beta_scale: float = 0.90,
    gamma_scale: float = 0.05,
    accept_cost: float | None = None,
    min_true_fold: float = 1.0,
) -> pd.DataFrame:
    """One round: only clear-insert indices may take x=1; filled flanks take z=1.

    Eligibility is geometry-only (tight plug). enrichment_fold / min_true_fold
    are ignored here so weak large domains cannot ride a strong flank fold.
    """
    del min_true_fold  # kept for call-site compat; not used at clear-insert
    n = len(df)
    # Objective also ignores fold: use neutral fold so size/support dominate.
    g = []
    for i in range(n):
        pseudo = df.iloc[i].copy()
        pseudo["enrichment_fold"] = 1.0
        g.append(gain(pseudo))
    if accept_cost is None:
        accept_cost = gain(
            pd.Series(
                {
                    "enrichment_fold": 1.0,
                    "scale_support": 2,
                    "size": 1_000_000,
                    "partner_size": 5_000_000,
                }
            )
        )
    beta = [beta_scale * max(g[j], 0.5) for j in range(n)]
    gamma = [gamma_scale * max(g[j], 0.5) for j in range(n)]

    pair_edges = [(i, j) for i, j, _a, _b in fill_edges]
    clear_inserters = sorted({i for i, _j in pair_edges})
    explainer_of: dict[int, list[int]] = {j: [] for j in range(n)}
    best_fill_q: dict[int, tuple[float, float]] = {
        i: (0.0, 1.0) for i in range(n)
    }
    for i, j, res_l, res_r in fill_edges:
        explainer_of[j].append(i)
        i0, i1 = int(df.iloc[i]["start"]), int(df.iloc[i]["end"])
        cover, resid = _fill_quality(res_l, res_r, i0, i1)
        prev = best_fill_q[i]
        # prefer higher cover, then lower residual
        if cover > prev[0] or (cover == prev[0] and resid < prev[1]):
            best_fill_q[i] = (cover, resid)

    # Tight geometric plugs only (no fold gate).
    perfect_fillers = {
        i
        for i in clear_inserters
        if best_fill_q[i][0] >= 0.95 and best_fill_q[i][1] <= 0.05
    }

    eligible = set()
    for i in perfect_fillers:
        if not is_near_arm_pair(df.iloc[i].to_dict()):
            eligible.add(i)

    m = pulp.LpProblem("tra_clear_insert_round", pulp.LpMaximize)
    idxs = list(range(n))
    x = m.add_variable_dicts("x", idxs, lowBound=0, upBound=1, cat="Binary")
    z = m.add_variable_dicts("z", idxs, lowBound=0, upBound=1, cat="Binary")
    u = m.add_variable_dicts("u", idxs, lowBound=0, upBound=1, cat="Binary")

    m += (
        pulp.lpSum((g[i] - accept_cost) * x[i] for i in idxs)
        + pulp.lpSum(beta[j] * z[j] for j in idxs)
        + pulp.lpSum(-gamma[j] * u[j] for j in idxs)
    )

    for i in idxs:
        m += x[i] + z[i] + u[i] == 1
        if i not in eligible:
            m += x[i] == 0

    for a, b in conf_edges:
        m += x[a] + x[b] <= 1

    # inserter vs the pair it fills: cannot both be true
    for i, j in pair_edges:
        m += x[i] + x[j] <= 1

    # filled flank j may be shadow only if some clear inserter of j is true;
    # accepting an inserter forces its filled flanks into shadow
    for j, expl in explainer_of.items():
        if expl:
            m += z[j] <= pulp.lpSum(x[i] for i in expl)
            for i in expl:
                m += z[j] >= x[i]
        else:
            m += z[j] == 0

    stats = m.solve(_default_ilp_solver())
    status_name = getattr(stats.status, "name", str(stats.status))
    if "Optimal" not in status_name and not getattr(stats, "has_solution", False):
        raise RuntimeError(f"ILP failed: {stats}")

    out = df.copy()
    out["gain"] = g
    out["accept_net"] = [g[i] - accept_cost for i in idxs]
    out["role"] = [
        "true" if x[i].value() > 0.5 else "shadow" if z[i].value() > 0.5 else "reject"
        for i in idxs
    ]
    out["x"] = [int(x[i].value() > 0.5) for i in idxs]
    out["z"] = [int(z[i].value() > 0.5) for i in idxs]
    out["u"] = [int(u[i].value() > 0.5) for i in idxs]
    out["explained_by"] = ""
    out["explain_kind"] = ""
    for i, j in pair_edges:
        if out.at[j, "z"] == 1 and out.at[i, "x"] == 1:
            prev = out.at[j, "explained_by"]
            tag = str(int(out.at[i, "cluster_id"]))
            out.at[j, "explained_by"] = tag if not prev else f"{prev},{tag}"
            out.at[j, "explain_kind"] = "clear_insert"
    return out


def apply_round_merges(
    round_out: pd.DataFrame,
    fill_edges: list[tuple[int, int, int, int]],
    merged_blocks: dict[str, list[tuple[int, int]]],
) -> tuple[pd.DataFrame, list[dict], dict[str, list[tuple[int, int]]]]:
    """Remove accepted inserts + filled flanks; extend merged blocks as wholes."""
    true_idx = set(round_out.index[round_out["x"] == 1].tolist())
    shadow_idx = set(round_out.index[round_out["z"] == 1].tolist())
    filled_by: dict[int, list[int]] = {i: [] for i in true_idx}
    for i, j, _a, _b in fill_edges:
        if i in true_idx:
            filled_by[i].append(j)
            # force filled flank to shadow for bookkeeping
            if round_out.at[j, "x"] != 1:
                round_out.at[j, "z"] = 1
                round_out.at[j, "u"] = 0
                round_out.at[j, "role"] = "shadow"
                if not round_out.at[j, "explain_kind"]:
                    round_out.at[j, "explain_kind"] = "filled_flank"
                prev = str(round_out.at[j, "explained_by"] or "")
                tag = str(int(round_out.at[i, "cluster_id"]))
                if tag not in prev.split(","):
                    round_out.at[j, "explained_by"] = tag if not prev else f"{prev},{tag}"
                shadow_idx.add(j)

    units: list[dict] = []
    new_blocks = {c: list(v) for c, v in merged_blocks.items()}

    for i in sorted(true_idx):
        ri = round_out.loc[i]
        chrom = str(ri["chrom"])
        # Whole = insert + the same-label flank intervals it plugged between.
        # Do NOT expand through distant partners (partner is no longer the
        # other side of the clear-insert seam).
        spans = [(int(ri["start"]), int(ri["end"]))]
        resolved: list[int] = []
        for j in sorted(set(filled_by.get(i, []))):
            rj = round_out.loc[j]
            resolved.append(int(rj["cluster_id"]))
            spans.append((int(rj["start"]), int(rj["end"])))

        u0 = min(s for s, _e in spans)
        u1 = max(e for _s, e in spans)
        # Occupied wholes: union only when intervals overlap or abut
        new_blocks[chrom] = _merge_interval_list(
            new_blocks.get(chrom, []) + [(u0, u1)]
        )
        units.append(
            {
                "insert_cluster": int(ri["cluster_id"]),
                "explained_clusters": resolved,
                "unit_start": u0,
                "unit_end": u1,
                "chrom": chrom,
            }
        )

    drop = set(true_idx) | set(shadow_idx)
    keep = round_out.drop(index=list(drop)).reset_index(drop=True)
    if len(keep):
        keep["idx"] = keep.index
    return keep, units, new_blocks


def resolve_persistent_mutuals(
    mutual_records: list[dict],
    true_tables: dict,
    min_rec: float = 0.5,
) -> tuple[pd.DataFrame, int]:
    """Final pass: ensure *both* sides of mutual-insert pairs are reported.

    A B C D -> A C B D: B and C are mutually insertable. Recover any side that
    never became ``true`` at any scale. Never demote the other side.
    Returns (recovered_rows, n_demote) with n_demote always 0.
    """
    empty = pd.DataFrame()
    if not mutual_records:
        return empty, 0

    ivs: list[tuple[str, int, int]] = []
    for rec in mutual_records:
        chrom = str(rec["chrom"])
        ivs.append((chrom, int(rec["start"]), int(rec["end"])))
        ivs.append((chrom, int(rec["other_start"]), int(rec["other_end"])))
    uniq: list[tuple[str, int, int]] = []
    seen_iv: set[tuple[str, int, int]] = set()
    for iv in ivs:
        if iv not in seen_iv:
            seen_iv.add(iv)
            uniq.append(iv)

    parent = list(range(len(uniq)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            if uniq[i][0] != uniq[j][0]:
                continue
            if _reciprocal(uniq[i][1], uniq[i][2], uniq[j][1], uniq[j][2]) >= min_rec:
                union(i, j)

    members: dict[int, list[int]] = {}
    for i in range(len(uniq)):
        members.setdefault(find(i), []).append(i)
    rep_iv: dict[int, tuple[str, int, int]] = {}
    for root, idxs in members.items():
        best = max(idxs, key=lambda i: uniq[i][2] - uniq[i][1])
        rep_iv[root] = uniq[best]

    def root_of(chrom: str, s: int, e: int) -> int | None:
        for i, iv in enumerate(uniq):
            if iv[0] != chrom:
                continue
            if _reciprocal(s, e, iv[1], iv[2]) >= min_rec:
                return find(i)
        return None

    best_rec: dict[int, dict] = {}
    edges: set[tuple[int, int]] = set()
    for rec in mutual_records:
        chrom = str(rec["chrom"])
        ra = root_of(chrom, int(rec["start"]), int(rec["end"]))
        rb = root_of(chrom, int(rec["other_start"]), int(rec["other_end"]))
        if ra is None or rb is None or ra == rb:
            continue
        prev = best_rec.get(ra)
        if prev is None or int(rec["flank_score"]) > int(prev["flank_score"]):
            best_rec[ra] = dict(rec)
        e = (ra, rb) if ra < rb else (rb, ra)
        edges.add(e)

    def covered(chrom: str, s: int, e: int) -> bool:
        for df in true_tables.values():
            if df is None or len(df) == 0 or "role" not in df.columns:
                continue
            for _, r in df.iterrows():
                if str(r.get("role")) != "true":
                    continue
                if str(r["chrom"]) != chrom:
                    continue
                if _reciprocal(int(r["start"]), int(r["end"]), s, e) >= min_rec:
                    return True
        return False

    recovered_rows: list[dict] = []
    roots_needed: set[int] = set()
    for ra, rb in edges:
        roots_needed.add(ra)
        roots_needed.add(rb)

    for root in sorted(roots_needed):
        rec = best_rec.get(root)
        if rec is None:
            # fall back to representative interval only (no partner) — skip
            continue
        chrom = str(rec["chrom"])
        s, e = int(rec["start"]), int(rec["end"])
        if covered(chrom, s, e):
            continue
        recovered_rows.append(
            {
                "cluster_id": 10_000 + len(recovered_rows) + 1,
                "chrom": chrom,
                "start": s,
                "end": e,
                "partner_start": int(rec["partner_start"]),
                "partner_end": int(rec["partner_end"]),
                "enrichment_fold": float(rec.get("enrichment_fold", 1.0)),
                "scale_support": 1,
                "size": e - s,
                "partner_size": int(rec["partner_end"]) - int(rec["partner_start"]),
                "label": int(rec.get("label", -1)),
                "role": "true",
                "resolution": rec.get("resolution", -1),
                "gamma": rec.get("gamma", float("nan")),
                "recover_source": "mutual_both_sides",
                "flank_score": int(rec["flank_score"]),
            }
        )

    recovered = pd.DataFrame(recovered_rows) if recovered_rows else empty
    return recovered, 0


def annotate_mutual_insert_on_merged(
    merged: pd.DataFrame,
    mutual_records: list[dict] | None,
    min_rec: float = 0.5,
) -> pd.DataFrame:
    """Mark mutual-insert pairs on final TRA calls.

    Adds columns:
      mutual_insert      1 if call is one side of a mutual-insert pair
      mutual_pair_id     shared id for the two sides (0 = not mutual)
      mutual_eval_keep   1 = keep for eval; 0 = drop the duplicate side
                         (prefer higher enrichment_fold, then larger size)
      mutual_partner_start / mutual_partner_end  the other side's interval
    Both sides stay in ALL_TRA; eval should drop rows with mutual_eval_keep==0.
    """
    out = merged.copy()
    n = len(out)
    out["mutual_insert"] = 0
    out["mutual_pair_id"] = 0
    out["mutual_eval_keep"] = 1
    out["mutual_partner_start"] = -1
    out["mutual_partner_end"] = -1
    if n == 0 or not mutual_records:
        return out

    def _hit(s: int, e: int, chrom: str) -> int | None:
        best_i, best_r = None, -1.0
        for i in range(n):
            if str(out.at[i, "chrom"]) != chrom:
                continue
            r = _reciprocal(s, e, int(out.at[i, "start"]), int(out.at[i, "end"]))
            if r >= min_rec and r > best_r:
                best_i, best_r = i, r
        return best_i

    # unique unordered pairs of call indices
    pair_set: set[tuple[int, int]] = set()
    pair_coords: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for rec in mutual_records:
        chrom = str(rec["chrom"])
        ia = _hit(int(rec["start"]), int(rec["end"]), chrom)
        ib = _hit(int(rec["other_start"]), int(rec["other_end"]), chrom)
        if ia is None or ib is None or ia == ib:
            continue
        key = (ia, ib) if ia < ib else (ib, ia)
        pair_set.add(key)
        pair_coords[key] = (
            int(out.at[key[0], "start"]),
            int(out.at[key[0], "end"]),
            int(out.at[key[1], "start"]),
            int(out.at[key[1], "end"]),
        )

    for pid, (ia, ib) in enumerate(sorted(pair_set), start=1):
        sa, ea = int(out.at[ia, "start"]), int(out.at[ia, "end"])
        sb, eb = int(out.at[ib, "start"]), int(out.at[ib, "end"])
        # Prefer larger arm (usually the donor), then enrichment / support.
        score_a = (
            ea - sa,
            float(out.at[ia, "enrichment_fold"]),
            int(out.at[ia, "scale_support"]) if "scale_support" in out.columns else 0,
        )
        score_b = (
            eb - sb,
            float(out.at[ib, "enrichment_fold"]),
            int(out.at[ib, "scale_support"]) if "scale_support" in out.columns else 0,
        )
        keep_a = score_a >= score_b
        for i, other_s, other_e, keep in (
            (ia, sb, eb, keep_a),
            (ib, sa, ea, not keep_a),
        ):
            out.at[i, "mutual_insert"] = 1
            out.at[i, "mutual_pair_id"] = pid
            out.at[i, "mutual_eval_keep"] = 1 if keep else 0
            out.at[i, "mutual_partner_start"] = other_s
            out.at[i, "mutual_partner_end"] = other_e
    return out


def recover_mutual_by_flank_score(
    mutual_records: list[dict],
    existing_trues: list[pd.DataFrame],
    min_rec: float = 0.5,
    max_recover_bp: int = 25_000_000,
) -> pd.DataFrame:
    true_tables = {i: df for i, df in enumerate(existing_trues)}
    recovered, _ = resolve_persistent_mutuals(
        mutual_records, true_tables, min_rec=min_rec
    )
    return recovered


def suppress_mutual_losers_in_true_tables(
    true_tables: dict,
    mutual_records: list[dict],
    min_rec: float = 0.5,
) -> int:
    """No-op: mutual-insert pairs are both kept (never demoted)."""
    return 0


def iterate_clear_insert(
    df: pd.DataFrame,
    max_rounds: int = 20,
    beta_scale: float = 0.90,
    gamma_scale: float = 0.05,
    accept_cost: float | None = None,
    min_true_fold: float = 1.0,
    min_gap_bp: int = 500_000,
    min_flank_bp: int = 500_000,
    max_residual_frac: float = 0.25,
    min_cover_frac: float = 0.5,
    seed: int = 42,
) -> tuple[pd.DataFrame, list[dict], list[dict], list[dict]]:
    """Run clear-insert ILP rounds with merged wholes until no new accepts.

    Returns (roles_df, history, units, mutual_records).
    """
    active = df.copy().reset_index(drop=True)
    active["idx"] = active.index

    final_roles: dict[int, dict] = {}
    all_units: list[dict] = []
    history: list[dict] = []
    merged_blocks: dict[str, list[tuple[int, int]]] = {}
    all_mutual: list[dict] = []

    for rnd in range(1, max_rounds + 1):
        if active.empty:
            break
        fill_edges, mutual_recs = build_clear_insert_relations(
            active,
            merged_blocks=merged_blocks,
            min_gap_bp=min_gap_bp,
            min_flank_bp=min_flank_bp,
            max_residual_frac=max_residual_frac,
            min_cover_frac=min_cover_frac,
            seed=int(seed) + rnd,
        )
        if mutual_recs:
            for rec in mutual_recs:
                rec = dict(rec)
                rec["round"] = rnd
                all_mutual.append(rec)
            n_pairs = sum(1 for r in mutual_recs if r.get("mutual_decision") == "keep") // 2
            print(
                f"[round {rnd}] mutual={n_pairs} pairs (both sides kept) "
                f"(randomly keep one side)",
                flush=True,
            )
        conf = build_conflicts(active)
        clear_inserters = sorted({i for i, _j, _a, _b in fill_edges})
        block_n = sum(len(v) for v in merged_blocks.values())
        print(
            f"[round {rnd}] active={len(active)} fill_edges={len(fill_edges)} "
            f"merged_blocks={block_n} "
            f"clear_inserters={[int(active.at[i, 'cluster_id']) for i in clear_inserters]}"
        )

        round_out: pd.DataFrame | None = None
        n_true = 0
        n_shadow = 0

        if fill_edges:
            round_out = solve_round_ilp(
                active,
                fill_edges,
                conf,
                beta_scale=beta_scale,
                gamma_scale=gamma_scale,
                accept_cost=accept_cost,
                min_true_fold=min_true_fold,
            )
            n_true = int(round_out["x"].sum())
            n_shadow = int(round_out["z"].sum())
            print(
                f"[round {rnd}] accepted="
                f"{[int(c) for c in round_out.loc[round_out.x == 1, 'cluster_id']]} "
                f"shadow={[int(c) for c in round_out.loc[round_out.z == 1, 'cluster_id']]}"
            )

        if not fill_edges or n_true == 0:
            break

        assert round_out is not None
        history.append(
            {
                "round": rnd,
                "n_active": len(active),
                "n_fill_edges": len(fill_edges),
                "n_true": n_true,
                "n_shadow": n_shadow,
                "true_clusters": [
                    int(c) for c in round_out.loc[round_out.x == 1, "cluster_id"]
                ],
                "shadow_clusters": [
                    int(c) for c in round_out.loc[round_out.z == 1, "cluster_id"]
                ],
                "merged_blocks": block_n,
            }
        )

        for _, row in round_out.iterrows():
            cid = int(row["cluster_id"])
            if row["x"] == 1:
                final_roles[cid] = {**row.to_dict(), "role": "true", "round": rnd}
            elif row["z"] == 1:
                final_roles[cid] = {**row.to_dict(), "role": "shadow", "round": rnd}

        active, units, merged_blocks = apply_round_merges(
            round_out, fill_edges, merged_blocks
        )
        for u in units:
            u["round"] = rnd
        all_units.extend(units)
        print(
            f"[round {rnd}] merged_units={len(units)} "
            f"blocks={ {c: v for c, v in merged_blocks.items()} } "
            f"remaining_active={len(active)}"
        )

    out = df.copy()
    out["gain"] = [gain(df.iloc[i]) for i in range(len(df))]
    out["role"] = "reject"
    out["round"] = 0
    out["explained_by"] = ""
    out["explain_kind"] = ""
    out["x"] = 0
    out["z"] = 0
    out["u"] = 1
    for cid, info in final_roles.items():
        mask = out["cluster_id"] == cid
        out.loc[mask, "role"] = info["role"]
        out.loc[mask, "round"] = info["round"]
        out.loc[mask, "explained_by"] = info.get("explained_by", "")
        out.loc[mask, "explain_kind"] = info.get("explain_kind", "")
        if info["role"] == "true":
            out.loc[mask, "x"] = 1
            out.loc[mask, "z"] = 0
            out.loc[mask, "u"] = 0
        elif info["role"] == "shadow":
            out.loc[mask, "x"] = 0
            out.loc[mask, "z"] = 1
            out.loc[mask, "u"] = 0
    return out, history, all_units, all_mutual

# ---------------------------------------------------------------------------
# Per-resolution orchestration + merge
# ---------------------------------------------------------------------------

def _reciprocal(a0: int, a1: int, b0: int, b1: int) -> float:
    o = _overlap(a0, a1, b0, b1)
    if o <= 0:
        return 0.0
    return min(o / max(1, a1 - a0), o / max(1, b1 - b0))


def candidates_to_calls(
    cands: pd.DataFrame,
    chrom: str,
    resolution: int,
) -> pd.DataFrame:
    """Convert analyze_scale / per-scale candidate rows into optimizer calls."""
    if cands is None or cands.empty:
        return pd.DataFrame()
    g = cands.copy()
    if "chrom" not in g.columns:
        g["chrom"] = chrom
    out = pd.DataFrame(
        {
            "cluster_id": np.arange(1, len(g) + 1, dtype=int),
            "chrom": g["chrom"].astype(str).values,
            "start": g["start_bp"].astype(int).values,
            "end": g["end_bp"].astype(int).values,
            "partner_start": g["main_start_bp"].astype(int).values,
            "partner_end": g["main_end_bp"].astype(int).values,
            "scale_support": 1,
            "resolutions": str(int(resolution)),
            "label": g["label"].values if "label" in g.columns else -1,
            "scale_id": g["scale_id"].astype(str).values
            if "scale_id" in g.columns
            else f"r{resolution}",
            "gamma": g["gamma"].values if "gamma" in g.columns else float("nan"),
        }
    )
    out["size"] = out["end"] - out["start"]
    out["partner_size"] = out["partner_end"] - out["partner_start"]
    out["enrichment_fold"] = 1.0
    out = out.sort_values(["chrom", "start", "end"]).reset_index(drop=True)
    out["cluster_id"] = np.arange(1, len(out) + 1, dtype=int)
    out["idx"] = out.index
    return out


def per_scale_to_calls(ps: pd.DataFrame, resolution: int) -> pd.DataFrame:
    """Convert one resolution's rows from ALL_TRA_per_scale_candidates.tsv."""
    g = ps[ps["resolution"] == resolution].copy()
    if g.empty:
        return g
    chrom = str(g["chrom"].iloc[0]) if "chrom" in g.columns else "chr"
    return candidates_to_calls(g, chrom, resolution)


def score_enrichment_for_calls(
    calls: pd.DataFrame,
    mcool: Path,
    chrom: str,
    chrom_size: int,
    resolution: int,
    segments: pd.DataFrame,
) -> pd.DataFrame:
    """Fill enrichment_fold using candidatexpartner vs flanking communities."""
    clr = cooler.Cooler(f"{mcool}::/resolutions/{resolution}")
    folds = []
    crosses = []
    neighs = []
    for _, row in calls.iterrows():
        metrics = score_event_vs_neighbors(
            clr,
            chrom,
            chrom_size,
            segments,
            resolution,
            int(row["start"]),
            int(row["end"]),
            int(row["partner_start"]),
            int(row["partner_end"]),
        )
        folds.append(float(metrics["enrichment_fold"]))
        crosses.append(float(metrics["cross_mean"]))
        neighs.append(float(metrics["neighbor_mean"]))
    out = calls.copy()
    out["enrichment_fold"] = folds
    out["cross_mean"] = crosses
    out["neighbor_mean"] = neighs
    # guard inf
    out["enrichment_fold"] = out["enrichment_fold"].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)
    return out


def _interval_merge_score(a0: int, a1: int, b0: int, b1: int) -> float:
    """Reciprocal overlap, or containment of the smaller in the larger."""
    rec = _reciprocal(a0, a1, b0, b1)
    ov = _overlap(a0, a1, b0, b1)
    if ov <= 0:
        return 0.0
    smaller = min(max(1, a1 - a0), max(1, b1 - b0))
    return max(rec, ov / smaller)


def merge_true_across_resolutions(
    true_tables: dict,
    min_rec: float = 0.5,
) -> pd.DataFrame:
    """Cluster true calls across scales by overlap / containment.

    Keys may be resolution (int) or (resolution, gamma) tuples when multiple
    Louvain gammas are run per resolution.

    Two calls are linked if reciprocal overlap >= min_rec *or* the smaller
    is mostly contained in the larger (overlap / min_len >= min_rec). Among
    overlapping merged groups, keep the one with higher scale_support
    (then larger size).
    """
    rows = []
    def _sort_key(kv):
        k = kv[0]
        if not isinstance(k, tuple):
            return (int(k), float("nan"), 0)
        if len(k) >= 3:
            return (int(k[0]), float(k[1]), int(k[2]))
        return (int(k[0]), float(k[1]), 0)

    for key, df in sorted(true_tables.items(), key=_sort_key):
        if isinstance(key, tuple):
            res, gamma = int(key[0]), float(key[1])
        else:
            res, gamma = int(key), float("nan")
        tr = df[df["role"] == "true"].copy()
        for _, r in tr.iterrows():
            g = float(r["gamma"]) if "gamma" in tr.columns and pd.notna(r.get("gamma")) else gamma
            rows.append(
                {
                    "resolution": int(res),
                    "gamma": g,
                    "local_cluster_id": int(r["cluster_id"]),
                    "chrom": str(r["chrom"]),
                    "start": int(r["start"]),
                    "end": int(r["end"]),
                    "partner_start": int(r["partner_start"]),
                    "partner_end": int(r["partner_end"]),
                    "enrichment_fold": float(r["enrichment_fold"]),
                    "gain": float(r.get("gain", 0.0) or 0.0),
                    "round": int(r.get("round", 0) or 0),
                }
            )
    if not rows:
        return pd.DataFrame()

    items = pd.DataFrame(rows).reset_index(drop=True)
    n = len(items)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if items.at[i, "chrom"] != items.at[j, "chrom"]:
                continue
            score = _interval_merge_score(
                int(items.at[i, "start"]),
                int(items.at[i, "end"]),
                int(items.at[j, "start"]),
                int(items.at[j, "end"]),
            )
            if score >= min_rec:
                union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    merged = []
    for idxs in sorted(clusters.values(), key=lambda ix: items.loc[ix[0], "start"]):
        sub = items.loc[idxs].copy()
        sub["size"] = sub["end"] - sub["start"]
        res_list = sorted(set(int(x) for x in sub["resolution"]))
        scale_keys = sorted(
            {
                (int(r.resolution), float(r.gamma) if pd.notna(r.gamma) else float("nan"))
                for _, r in sub.iterrows()
            }
        )
        # representative coords: largest member (covers contained fragments),
        # then enrichment / gain as tie-break
        rep = sub.sort_values(
            ["size", "enrichment_fold", "gain"], ascending=False
        ).iloc[0]
        gammas = sorted(
            {float(x) for x in sub["gamma"] if pd.notna(x)},
            key=lambda v: v,
        )
        merged.append(
            {
                "cluster_id": len(merged) + 1,
                "chrom": rep["chrom"],
                "start": int(rep["start"]),
                "end": int(rep["end"]),
                "partner_start": int(rep["partner_start"]),
                "partner_end": int(rep["partner_end"]),
                "enrichment_fold": float(sub["enrichment_fold"].max()),
                "scale_support": len(scale_keys),
                "resolutions": ",".join(str(r) for r in res_list),
                "gammas": ",".join(f"{g:g}" for g in gammas),
                "member_resolutions": ";".join(
                    (
                        f"{int(r.resolution)}_g{float(r.gamma):g}:c{int(r.local_cluster_id)}"
                        if pd.notna(r.gamma)
                        else f"{int(r.resolution)}:c{int(r.local_cluster_id)}"
                    )
                    for _, r in sub.sort_values(["resolution", "gamma"]).iterrows()
                ),
                "n_members": len(sub),
                "start_min": int(sub["start"].min()),
                "end_max": int(sub["end"].max()),
            }
        )
    if not merged:
        return pd.DataFrame()

    out = pd.DataFrame(merged).sort_values(["chrom", "start", "end"]).reset_index(
        drop=True
    )
    # Final pass: if two merged calls still overlap, keep higher scale_support
    # (then larger size, then enrichment).
    keep = [True] * len(out)
    for i in range(len(out)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(out)):
            if not keep[j]:
                continue
            if out.at[i, "chrom"] != out.at[j, "chrom"]:
                continue
            score = _interval_merge_score(
                int(out.at[i, "start"]),
                int(out.at[i, "end"]),
                int(out.at[j, "start"]),
                int(out.at[j, "end"]),
            )
            if score < min_rec:
                continue
            si = (
                int(out.at[i, "scale_support"]),
                int(out.at[i, "end"]) - int(out.at[i, "start"]),
                float(out.at[i, "enrichment_fold"]),
            )
            sj = (
                int(out.at[j, "scale_support"]),
                int(out.at[j, "end"]) - int(out.at[j, "start"]),
                float(out.at[j, "enrichment_fold"]),
            )
            if sj > si:
                keep[i] = False
                break
            keep[j] = False
    out = out.loc[keep].reset_index(drop=True)
    out["cluster_id"] = np.arange(1, len(out) + 1, dtype=int)
    return out


def plot_resolution_community(
    mcool: Path,
    chrom: str,
    chrom_size: int,
    resolution: int,
    segments: pd.DataFrame,
    true_calls: pd.DataFrame,
    truth: pd.DataFrame,
    out_path: Path,
    plot_res: int = 500_000,
) -> None:
    """Community track + chrom Hi-C with this resolution's optimized trues."""
    tra = truth[truth["svtype"] == "TRA"].copy()

    def annotate(df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, c in df.iterrows():
            best_name, best_rec = None, 0.0
            for _, t in tra.iterrows():
                rec = _reciprocal(int(c.start), int(c.end), int(t.start), int(t.end))
                if rec > best_rec:
                    best_rec = rec
                    best_name = t["name"]
            rows.append(
                {
                    **c.to_dict(),
                    "truth": best_name,
                    "rec_ov": best_rec,
                    "tp_fp": "TP" if best_rec >= 0.5 else "FP",
                }
            )
        return pd.DataFrame(rows)

    ann = annotate(true_calls) if len(true_calls) else true_calls
    chrom_mb = chrom_size / 1e6
    available = discover_resolutions(mcool)
    use_plot = plot_res if plot_res in available else max(available)
    clr_plot = cooler.Cooler(f"{mcool}::/resolutions/{use_plot}")
    matrix = _dense_matrix(clr_plot, chrom, False)
    positive = matrix[np.isfinite(matrix) & (matrix > 0)]

    fig, axes = plt.subplots(
        1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1.2, 2.2]}
    )
    ax = axes[0]
    for _, segment in segments.iterrows():
        start_mb = int(segment["start_bin"]) * resolution / 1e6
        end_mb = min(int(segment["end_bin"]) * resolution, chrom_size) / 1e6
        if bool(segment.get("is_filtered", False)):
            color, lw, alpha = "lightgray", 1.5, 0.45
        elif bool(segment.get("is_candidate", False)):
            color, lw, alpha = "red", 4.0, 0.95
        else:
            color, lw, alpha = "royalblue", 3.0, 0.85
        lab = int(segment["label"])
        ax.plot(
            [start_mb, end_mb],
            [lab, lab],
            color=color,
            linewidth=lw,
            alpha=alpha,
            solid_capstyle="butt",
            zorder=1,
        )

    ymax = float(segments["label"].max()) if len(segments) else 1.0
    if len(ann):
        for _, ev in ann.iterrows():
            s = ev["start"] / 1e6
            e = ev["end"] / 1e6
            face = "#2ca02c" if ev["tp_fp"] == "TP" else "#ff7f0e"
            ax.axvspan(s, e, color=face, alpha=0.22, zorder=0)
            mid = 0.5 * (s + e)
            ax.annotate(
                f"c{int(ev['cluster_id'])}",
                xy=(mid, ymax * 0.98),
                xytext=(mid, ymax * 1.06),
                ha="center",
                va="bottom",
                fontsize=7,
                color=face,
                fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=face, lw=0.8),
                clip_on=False,
            )

    ax.set_xlim(0, chrom_mb)
    ax.set_ylim(-2, ymax * 1.12)
    ax.set_xlabel("Genomic position (Mb)")
    ax.set_ylabel("Community cluster")
    n_true = len(ann)
    n_tp = int((ann["tp_fp"] == "TP").sum()) if len(ann) else 0
    ax.set_title(
        f"Community ({resolution/1000:g} kb) + per-res optimize true "
        f"(n={n_true}, TP={n_tp})\n"
        "Red=candidate, Blue=main, Gray=filtered; green=TP, orange=FP"
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
        )
    else:
        ax.imshow(matrix, cmap="Reds", extent=[0, chrom_mb, chrom_mb, 0])
    if len(ann):
        for _, ev in ann.iterrows():
            s = ev["start"] / 1e6
            e = ev["end"] / 1e6
            face = "#2ca02c" if ev["tp_fp"] == "TP" else "#ff7f0e"
            ax.add_patch(
                plt.Rectangle(
                    (s, s),
                    max(0.05, e - s),
                    max(0.05, e - s),
                    fill=False,
                    edgecolor=face,
                    lw=1.6,
                )
            )
            p0, p1 = ev["partner_start"] / 1e6, ev["partner_end"] / 1e6
            ax.add_patch(
                plt.Rectangle(
                    (p0, p0),
                    max(0.05, p1 - p0),
                    max(0.05, p1 - p0),
                    fill=False,
                    edgecolor=face,
                    lw=1.0,
                    linestyle="--",
                )
            )
            ax.plot([s, p0], [e, p1], color=face, lw=0.7, alpha=0.7)
            ax.plot([e, p1], [s, p0], color=face, lw=0.7, alpha=0.7)
    ax.set_xlim(0, chrom_mb)
    ax.set_ylim(chrom_mb, 0)
    ax.set_xlabel("Mb")
    ax.set_ylabel("Mb")
    ax.set_title(f"{chrom} heatmap ({use_plot/1000:g} kb)\nsolid=true arm, dashed=partner")
    fig.suptitle(
        f"{chrom} per-resolution restore @ {resolution/1000:g} kb",
        fontsize=13,
        y=0.98,
    )
    fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] wrote {out_path}", flush=True)


def _calls_abut(a0: int, a1: int, b0: int, b1: int, slack: int = 500_000) -> bool:
    """True if intervals touch or nearly touch (reciprocal TRA neighbors)."""
    if a1 <= b0:
        return (b0 - a1) <= slack
    if b1 <= a0:
        return (a0 - b1) <= slack
    return _overlap(a0, a1, b0, b1) > 0


def evaluate_against_truth(calls: pd.DataFrame, truth: pd.DataFrame, thr: float = 0.5):
    """Score calls vs truth TRA intervals.

    A B C D -> A C B D: calling B *or* the abutting C are both correct for
    that event. Match by reciprocal overlap, or by abutting a truth TRA
    (reciprocal insert view). Recall = fraction of truth events hit either way.
    """
    tra = truth[truth["svtype"] == "TRA"]
    pred = calls.copy().reset_index(drop=True)
    matched = set()
    details: list[tuple] = []
    tp = fp = 0

    for _, r in pred.iterrows():
        best_name, best_rec, how = None, 0.0, None
        p0, p1 = int(r["start"]), int(r["end"])
        for _, t in tra.iterrows():
            t0, t1 = int(t["start"]), int(t["end"])
            rec = _reciprocal(p0, p1, t0, t1)
            if rec >= thr and (how != "overlap" or rec > best_rec):
                best_name, best_rec, how = t["name"], rec, "overlap"
            elif how is None and _calls_abut(p0, p1, t0, t1):
                best_name, best_rec, how = t["name"], rec, "recip"
        if how is not None:
            tp += 1
            matched.add(best_name)
            tag = "TP" if how == "overlap" else "TP_recip"
        else:
            fp += 1
            tag = "FP"
        details.append(
            (int(r.get("cluster_id", -1)), best_name, round(best_rec, 3), tag)
        )

    fn = [t["name"] for _, t in tra.iterrows() if t["name"] not in matched]
    prec = tp / max(1, tp + fp)
    recall = len(matched) / max(1, len(tra))
    f1 = 0.0 if prec + recall == 0 else 2 * prec * recall / (prec + recall)
    return {
        "tp": tp,
        "fp": fp,
        "fn": len(fn),
        "precision": prec,
        "recall": recall,
        "f1": f1,
        "matched": sorted(matched),
        "fn_names": fn,
        "details": details,
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--per-scale",
        type=Path,
        default=Path(__file__).resolve().parent / "ALL_TRA_per_scale_candidates.tsv",
    )
    ap.add_argument(
        "--mcool",
        type=Path,
        required=True,
        help="input .mcool file",
    )
    ap.add_argument(
        "--truth",
        type=Path,
        default=None,
        help=(
            "optional truth BED (chrom start end name size svtype status); "
            "when set, print precision/recall against TRA rows"
        ),
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=Path("TRA_results"),
        help="output directory (default: TRA_results)",
    )
    ap.add_argument(
        "--resolutions",
        type=str,
        default=",".join(str(r) for r in RESOLUTIONS_DEFAULT),
    )
    ap.add_argument("--chrom", type=str, default="chr3B")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="single Louvain gamma (ignored if --gammas is set)")
    ap.add_argument(
        "--gammas",
        type=str,
        default=",".join(f"{g:g}" for g in GAMMAS_DEFAULT),
        help="comma-separated Louvain/PLM gammas "
             f"(default: {','.join(f'{g:g}' for g in GAMMAS_DEFAULT)})",
    )
    ap.add_argument(
        "--edge-quantile",
        type=float,
        default=0.9,
        help="keep Hi-C edges at/above this weight quantile "
             "(default 0.9 = top 10%% strongest contacts)",
    )
    ap.add_argument("--min-segment-bp", type=int, default=500_000)
    ap.add_argument("--bridge-bp", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-true-fold", type=float, default=1.0)
    ap.add_argument("--merge-min-rec", type=float, default=0.5)
    ap.add_argument(
        "--from-tsv",
        action="store_true",
        help="use --per-scale TSV instead of regenerating via analyze_scale",
    )
    ap.add_argument("--skip-plot", action="store_true")
    ap.add_argument("--skip-enrichment", action="store_true",
                    help="do not re-score enrichment (use fold=1)")
    ap.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="parallel (resolutionxgamma) workers (0=auto: cpu/2, 1=serial)",
    )
    ap.add_argument(
        "--threads",
        type=int,
        default=0,
        help=(
            "NetworKit PLM threads per scale (default: 0 = auto cpu/2; "
            "set 1 for fully serial PLM)"
        ),
    )
    return ap


def _run_one_res_gamma(payload: dict) -> dict:
    """Worker for one (resolution, gamma): cluster → enrich → clear-insert.

    Built for ProcessPoolExecutor (spawn). Rebuilds the Hi-C graph inside the
    worker (NetworKit graphs are not reliably shareable across processes).
    """
    res = int(payload["resolution"])
    gamma = float(payload["gamma"])
    mcool = Path(payload["mcool"])
    chrom = str(payload["chrom"])
    chrom_size = int(payload["chrom_size"])
    outdir = Path(payload["outdir"])
    n_threads = int(payload["n_threads"])
    edge_quantile = float(payload["edge_quantile"])
    seed = int(payload["seed"])
    min_segment_bp = int(payload["min_segment_bp"])
    bridge_bp = int(payload["bridge_bp"])
    min_true_fold = float(payload["min_true_fold"])
    skip_enrichment = bool(payload["skip_enrichment"])
    from_tsv = bool(payload["from_tsv"])
    truth_path = payload.get("truth_path")

    log_lines: list[str] = []

    def log(msg: str) -> None:
        log_lines.append(msg)

    log(f"\n======== resolution {res}  gamma={gamma:g} ========")
    cands = pd.DataFrame()
    segments = pd.DataFrame()
    nodes = edges = 0

    if from_tsv:
        ps = pd.read_csv(payload["per_scale"], sep="\t")
        calls = per_scale_to_calls(ps, res)
    else:
        log(
            f"[INFO] analyze_scale @ {res} g={gamma:g} "
            f"(edge_quantile={edge_quantile}, plm=par=balanced, "
            f"threads={n_threads}) ..."
        )
        _, _, nodes, edges, cands, segments = analyze_scale(
            mcool,
            chrom,
            chrom_size,
            res,
            gamma,
            edge_quantile,
            False,
            seed,
            min_segment_bp,
            bridge_bp,
            n_threads,
            plm_par="balanced",
            seed_salt=0,
            graph=None,
        )
        if "chrom" not in cands.columns:
            cands = cands.copy()
            cands["chrom"] = chrom
        calls = candidates_to_calls(cands, chrom, res)

    log(
        f"[INFO] graph={nodes}/{edges} segments={len(segments)} "
        f"candidates={len(calls)}"
    )
    if calls.empty:
        return {
            "resolution": res,
            "gamma": gamma,
            "empty": True,
            "log": "\n".join(log_lines),
        }

    if not skip_enrichment:
        calls = score_enrichment_for_calls(
            calls, mcool, chrom, chrom_size, res, segments
        )
        log(
            f"[INFO] enrichment fold median="
            f"{calls['enrichment_fold'].median():.3f} "
            f"max={calls['enrichment_fold'].max():.3f}"
        )

    tag = f"r{res}_g{gamma:g}"
    res_dir = outdir / tag
    res_dir.mkdir(parents=True, exist_ok=True)
    calls.to_csv(res_dir / "calls_input.tsv", sep="\t", index=False)
    if not from_tsv and len(cands):
        cands.to_csv(res_dir / "scale_candidates.tsv", sep="\t", index=False)

    out, history, units, mutual_recs = iterate_clear_insert(
        calls,
        min_true_fold=min_true_fold,
        seed=seed,
    )
    out = out.copy()
    out["gamma"] = gamma
    out.to_csv(res_dir / "iterative_restore_calls.tsv", sep="\t", index=False)
    pd.DataFrame(history).to_csv(res_dir / "round_history.tsv", sep="\t", index=False)
    if units:
        pd.DataFrame(units).to_csv(res_dir / "merged_units.tsv", sep="\t", index=False)

    mutual_out: list[dict] = []
    if mutual_recs:
        for rec in mutual_recs:
            rec = dict(rec)
            rec["resolution"] = res
            rec["gamma"] = gamma
            mutual_out.append(rec)
        pd.DataFrame(mutual_recs).assign(resolution=res, gamma=gamma).to_csv(
            res_dir / "mutual_dropped.tsv", sep="\t", index=False
        )

    role_counts = out["role"].value_counts().to_dict()
    tr = out[out["role"] == "true"].copy()
    if truth_path:
        truth = pd.read_csv(
            truth_path,
            sep="\t",
            header=None,
            names=["chrom", "start", "end", "name", "size", "svtype", "status"],
        )
        ev = evaluate_against_truth(tr, truth) if len(tr) else {
            "tp": 0,
            "fp": 0,
            "fn": int((truth.svtype == "TRA").sum()),
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "matched": [],
            "fn_names": [],
            "details": [],
        }
        log(
            f"[RES {res} g={gamma:g}] roles={role_counts}  "
            f"P={ev['precision']:.3f} R={ev['recall']:.3f} "
            f"F1={ev['f1']:.3f} TP={ev['tp']} FP={ev['fp']}"
        )
        for cid, name, rec, tag_ev in ev.get("details") or []:
            log(f"  {tag_ev} c{cid} -> {name}:{rec}")
    else:
        ev = {
            "tp": "",
            "fp": "",
            "precision": "",
            "recall": "",
            "f1": "",
            "matched": [],
        }
        log(
            f"[RES {res} g={gamma:g}] roles={role_counts}  "
            f"n_true={int(role_counts.get('true', 0))}"
        )

    summary = {
        "resolution": res,
        "gamma": gamma,
        "n_candidates": len(calls),
        "n_true": int(role_counts.get("true", 0)),
        "n_shadow": int(role_counts.get("shadow", 0)),
        "n_reject": int(role_counts.get("reject", 0)),
        "tp": ev["tp"],
        "fp": ev["fp"],
        "precision": ev["precision"],
        "recall": ev["recall"],
        "f1": ev["f1"],
        "matched": ",".join(ev["matched"]),
    }
    return {
        "resolution": res,
        "gamma": gamma,
        "empty": False,
        "out": out,
        "cands": cands if not from_tsv else pd.DataFrame(),
        "mutual_recs": mutual_out,
        "summary": summary,
        "segments": segments,
        "log": "\n".join(log_lines),
    }


def run_pipeline(args: argparse.Namespace) -> pd.DataFrame:
    """Run multi-(resolution x gamma) TRA detect -> clear-insert -> merge.

    Returns the merged true-call table (may be empty).
    """
    resolutions = [int(x) for x in args.resolutions.split(",") if x.strip()]
    if str(args.gammas).strip():
        gammas = [float(x) for x in str(args.gammas).split(",") if x.strip()]
    else:
        gammas = [float(args.gamma)]
    args.outdir.mkdir(parents=True, exist_ok=True)
    community_dir = args.outdir / "community"
    community_dir.mkdir(parents=True, exist_ok=True)

    if args.truth is not None:
        truth = pd.read_csv(
            args.truth,
            sep="\t",
            header=None,
            names=["chrom", "start", "end", "name", "size", "svtype", "status"],
        )
    else:
        truth = pd.DataFrame(
            columns=["chrom", "start", "end", "name", "size", "svtype", "status"]
        )

    available = discover_resolutions(args.mcool)
    use_resolutions = [r for r in resolutions if r in available]
    for r in resolutions:
        if r not in available:
            print(f"[WARN] resolution {r} not in mcool; skip", flush=True)
    if not use_resolutions:
        raise SystemExit("[ERROR] no requested resolutions present in mcool")

    clr0 = cooler.Cooler(f"{args.mcool}::/resolutions/{use_resolutions[0]}")
    chrom_size = int(clr0.chromsizes.loc[args.chrom])

    n_tasks = len(use_resolutions) * len(gammas)
    if int(getattr(args, "jobs", 0) or 0) > 0:
        n_jobs = int(args.jobs)
    else:
        n_jobs = max(1, (os.cpu_count() or 4) // 2)
    n_jobs = max(1, min(n_jobs, n_tasks))

    if int(getattr(args, "threads", 0) or 0) > 0:
        n_threads = int(args.threads)
    else:
        # avoid oversubscription when many (res,gamma) workers run at once
        n_threads = max(1, (os.cpu_count() or 4) // max(1, n_jobs))

    true_tables: dict = {}
    summary_rows = []
    all_cands_frames: list[pd.DataFrame] = []
    all_mutual_records: list[dict] = []

    print(
        f"[INFO] scales: resolutions={use_resolutions} gammas={gammas} "
        f"edge_quantile={args.edge_quantile} jobs={n_jobs} "
        f"threads/worker={n_threads} seed={args.seed}",
        flush=True,
    )

    payloads = []
    for res in use_resolutions:
        for gamma in gammas:
            payloads.append(
                {
                    "resolution": res,
                    "gamma": gamma,
                    "mcool": str(args.mcool),
                    "chrom": args.chrom,
                    "chrom_size": chrom_size,
                    "outdir": str(args.outdir),
                    "n_threads": n_threads,
                    "edge_quantile": float(args.edge_quantile),
                    "seed": int(args.seed),
                    "min_segment_bp": int(args.min_segment_bp),
                    "bridge_bp": int(args.bridge_bp),
                    "min_true_fold": float(args.min_true_fold),
                    "skip_enrichment": bool(args.skip_enrichment),
                    "from_tsv": bool(args.from_tsv),
                    "per_scale": str(args.per_scale) if args.from_tsv else "",
                    "truth_path": str(args.truth) if args.truth is not None else None,
                }
            )

    results: list[dict] = []
    if n_jobs == 1:
        for payload in payloads:
            results.append(_run_one_res_gamma(payload))
    else:
        print(
            f"[INFO] parallel (resolution x gamma): {n_tasks} tasks, "
            f"{n_jobs} workers",
            flush=True,
        )
        with ProcessPoolExecutor(
            max_workers=n_jobs, mp_context=mp.get_context("spawn")
        ) as ex:
            futs = {ex.submit(_run_one_res_gamma, p): p for p in payloads}
            for fut in as_completed(futs):
                results.append(fut.result())

    # stable order for logs / merge
    results.sort(key=lambda r: (int(r["resolution"]), float(r["gamma"])))
    for result in results:
        print(result.get("log", ""), flush=True)
        if result.get("empty"):
            continue
        res = int(result["resolution"])
        gamma = float(result["gamma"])
        out = result["out"]
        true_tables[(res, gamma)] = out
        summary_rows.append(result["summary"])
        if result.get("cands") is not None and len(result["cands"]):
            all_cands_frames.append(result["cands"])
        if result.get("mutual_recs"):
            all_mutual_records.extend(result["mutual_recs"])
        if not args.skip_plot:
            tr = out[out["role"] == "true"].copy()
            tag = f"r{res}_g{gamma:g}"
            plot_resolution_community(
                args.mcool,
                args.chrom,
                chrom_size,
                res,
                result.get("segments", pd.DataFrame()),
                tr,
                truth,
                community_dir / f"community_opt_{args.chrom}_{tag}.png",
            )

    if all_mutual_records:
        pd.DataFrame(all_mutual_records).to_csv(
            args.outdir / "mutual_pairs_all.tsv", sep="\t", index=False
        )
        # also keep legacy filename for downstream scripts
        pd.DataFrame(all_mutual_records).to_csv(
            args.outdir / "mutual_dropped_all.tsv", sep="\t", index=False
        )
        # Recover any mutual-insert side that never became role=true
        recovered, _ = resolve_persistent_mutuals(
            all_mutual_records, true_tables, min_rec=args.merge_min_rec
        )
        if len(recovered):
            print(
                f"[INFO] mutual recover both sides: {len(recovered)} calls",
                flush=True,
            )
            # numeric key so merge_true_across_resolutions sort stays valid
            true_tables[(-1, 0.0)] = recovered

    # merge across resolutions
    merged = merge_true_across_resolutions(
        true_tables,
        min_rec=args.merge_min_rec,
    )
    if len(merged):
        merged = annotate_mutual_insert_on_merged(
            merged, all_mutual_records, min_rec=args.merge_min_rec
        )
        n_mutual = int((merged["mutual_insert"] == 1).sum()) if "mutual_insert" in merged.columns else 0
        n_drop = int((merged["mutual_eval_keep"] == 0).sum()) if "mutual_eval_keep" in merged.columns else 0
        if n_mutual:
            print(
                f"[INFO] mutual-insert marked: {n_mutual} calls "
                f"({n_drop} secondary for eval drop)",
                flush=True,
            )
    merged_path = args.outdir / "merged_true_calls.tsv"
    if len(merged):
        if "size" not in merged.columns and {"start", "end"} <= set(merged.columns):
            merged = merged.copy()
            merged["size"] = merged["end"].astype(int) - merged["start"].astype(int)
        merged.to_csv(merged_path, sep="\t", index=False)
        # PHASC-style alias for pipeline / downstream tools
        merged.to_csv(args.outdir / "ALL_TRA.tsv", sep="\t", index=False)
    else:
        pd.DataFrame().to_csv(merged_path, sep="\t", index=False)
        pd.DataFrame().to_csv(args.outdir / "ALL_TRA.tsv", sep="\t", index=False)
    print(f"\n[INFO] merged true calls: {len(merged)} -> {merged_path}", flush=True)
    print(f"[INFO] ALL_TRA.tsv -> {args.outdir / 'ALL_TRA.tsv'}", flush=True)

    if len(merged) and args.truth is not None:
        mev = evaluate_against_truth(merged, truth)
        print(
            f"[MERGED] P={mev['precision']:.3f} R={mev['recall']:.3f} F1={mev['f1']:.3f} "
            f"TP={mev['tp']} FP={mev['fp']} FN={mev['fn']}",
            flush=True,
        )
        print(f"  matched: {mev['matched']}", flush=True)
        print(f"  FN: {mev['fn_names']}", flush=True)
        for cid, name, rec, tag in mev["details"]:
            row = merged.loc[merged.cluster_id == cid].iloc[0]
            gstr = row["gammas"] if "gammas" in merged.columns else ""
            print(
                f"  {tag} M{cid} scales={row['resolutions']} "
                f"gammas={gstr} supp={int(row['scale_support'])} "
                f"-> {name}:{rec}",
                flush=True,
            )
        summary_rows.append(
            {
                "resolution": "MERGED",
                "gamma": ",".join(f"{g:g}" for g in gammas),
                "n_candidates": sum(
                    int(s["n_candidates"])
                    for s in summary_rows
                    if s["resolution"] != "MERGED"
                ),
                "n_true": len(merged),
                "n_shadow": "",
                "n_reject": "",
                "tp": mev["tp"],
                "fp": mev["fp"],
                "precision": mev["precision"],
                "recall": mev["recall"],
                "f1": mev["f1"],
                "matched": ",".join(mev["matched"]),
            }
        )
    elif len(merged):
        summary_rows.append(
            {
                "resolution": "MERGED",
                "gamma": ",".join(f"{g:g}" for g in gammas),
                "n_candidates": sum(
                    int(s["n_candidates"])
                    for s in summary_rows
                    if s["resolution"] != "MERGED"
                ),
                "n_true": len(merged),
                "n_shadow": "",
                "n_reject": "",
                "tp": "",
                "fp": "",
                "precision": "",
                "recall": "",
                "f1": "",
                "matched": "",
            }
        )

    if all_cands_frames:
        pd.concat(all_cands_frames, ignore_index=True).to_csv(
            args.outdir / "ALL_TRA_per_scale_candidates.tsv",
            sep="\t",
            index=False,
        )

    pd.DataFrame(summary_rows).to_csv(
        args.outdir / "per_resolution_summary.tsv", sep="\t", index=False
    )
    print(f"[INFO] summary -> {args.outdir / 'per_resolution_summary.tsv'}", flush=True)
    return merged if len(merged) else pd.DataFrame()


def main(argv: list[str] | None = None) -> pd.DataFrame:
    args = build_parser().parse_args(argv)
    print(
        "[INFO] tra_multiscale\n"
        f"  mcool          = {args.mcool}\n"
        f"  chrom          = {args.chrom}\n"
        f"  resolutions    = {args.resolutions}\n"
        f"  gammas         = {args.gammas}\n"
        f"  edge_quantile  = {args.edge_quantile}  "
        f"(top {(1 - float(args.edge_quantile)) * 100:.0f}%)\n"
        f"  outdir         = {args.outdir}",
        flush=True,
    )
    return run_pipeline(args)


if __name__ == "__main__":
    main()

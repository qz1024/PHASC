#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multi-resolution × multi-Louvain *inter*-chromosomal TRA detection.

Mirrors ``multiscale_intra_translocation.py``, adapted for chromA↔chromB:

  for Hi-C resolution in {50,100,200,500} kb:
      for Louvain/PLM gamma in {0.5, 0.75, 1.0, 1.25, 1.5, 2.0}:
          joint graph (A intra + B intra + A×B inter, top-10% edges)
          → communities spanning both chroms → tip×tip O/E vs chrom-pair mean
  merge across (resolution, gamma) on both arms
  enrichment gate (default fold > 1.5)
  reciprocal optimize-merge (abutting corner calls → mid×mid exchange)
      → ALL_INTER_TRA.tsv  (pre-optimize: ALL_INTER_TRA_pre_optimize.tsv)

Intra clear-insert ILP is skipped (gap-fill is intra-only); inter uses
geometry optimize-merge instead.

Example:
  python multiscale_inter_translocation.py \\
    --mcool sample.tra.mapq1.mcool \\
    --chrom-a chr1A --chrom-b chr1B \\
    --outdir INTER_TRA_out --skip-plot

Deps: cooler, networkit, numpy, pandas, matplotlib.
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import cooler
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkit as nk
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

# =============================================================================
# Schema / defaults
# =============================================================================

RESOLUTIONS_DEFAULT = (50_000, 100_000, 200_000, 500_000)
GAMMAS_DEFAULT = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)

ALL_INTER_TRA_COLUMNS = [
    "cluster_id",
    "chrom_a",
    "start_a",
    "end_a",
    "chrom_b",
    "start_b",
    "end_b",
    "enrichment_fold",
    "enrichment_fold_block",
    "enrichment_fold_flank",
    "scale_support",
    "resolutions",
    "gammas",
    "member_resolutions",
    "n_members",
    "size_a",
    "size_b",
    "merged_reciprocal",
    "n_source_calls",
    "source_cluster_ids",
]


def empty_inter_tra_table() -> pd.DataFrame:
    return pd.DataFrame(columns=ALL_INTER_TRA_COLUMNS)


CANDIDATE_COLUMNS = [
    "scale_id",
    "resolution",
    "gamma",
    "label",
    "chrom_a",
    "start_a_bin",
    "end_a_bin",
    "start_a",
    "end_a",
    "chrom_b",
    "start_b_bin",
    "end_b_bin",
    "start_b",
    "end_b",
    "length_a_bp",
    "length_b_bp",
]

# =============================================================================
# Segments / communities
# =============================================================================


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


def _largest_block_per_label(
    labels: np.ndarray,
    resolution: int,
    chrom_size_bp: int,
    min_bins: int,
    bridge_bins: int,
) -> dict[int, tuple[int, int, int, int]]:
    """label → (start_bin, end_bin, start_bp, end_bp) for the longest block."""
    clean = bridge_short_interruptions(labels, bridge_bins)
    segs = run_length_segments(clean)
    out: dict[int, tuple[int, int, int, int]] = {}
    for _, row in segs.iterrows():
        if int(row["length_bins"]) < min_bins:
            continue
        lab = int(row["label"])
        sb, eb = int(row["start_bin"]), int(row["end_bin"])
        sbp = sb * resolution
        ebp = min(eb * resolution, chrom_size_bp)
        length = ebp - sbp
        if lab not in out or length > (out[lab][3] - out[lab][2]):
            out[lab] = (sb, eb, sbp, ebp)
    return out


def detect_inter_candidates(
    labels_joint: np.ndarray,
    n_a: int,
    resolution: int,
    gamma: float,
    chrom_a: str,
    chrom_b: str,
    size_a: int,
    size_b: int,
    min_segment_bp: int,
    bridge_bp: int,
) -> pd.DataFrame:
    """Shared community labels with a substantial block on *both* chromosomes."""
    min_bins = max(2, math.ceil(min_segment_bp / resolution))
    bridge_bins = max(0, math.floor(bridge_bp / resolution))
    labels_a = labels_joint[:n_a]
    labels_b = labels_joint[n_a:]
    blocks_a = _largest_block_per_label(
        labels_a, resolution, size_a, min_bins, bridge_bins
    )
    blocks_b = _largest_block_per_label(
        labels_b, resolution, size_b, min_bins, bridge_bins
    )
    shared = sorted(set(blocks_a) & set(blocks_b))
    scale_id = f"r{resolution}_g{gamma:g}"
    records = []
    for lab in shared:
        sa0, sa1, a0, a1 = blocks_a[lab]
        sb0, sb1, b0, b1 = blocks_b[lab]
        records.append(
            {
                "scale_id": scale_id,
                "resolution": resolution,
                "gamma": gamma,
                "label": lab,
                "chrom_a": chrom_a,
                "start_a_bin": sa0,
                "end_a_bin": sa1,
                "start_a": a0,
                "end_a": a1,
                "chrom_b": chrom_b,
                "start_b_bin": sb0,
                "end_b_bin": sb1,
                "start_b": b0,
                "end_b": b1,
                "length_a_bp": a1 - a0,
                "length_b_bp": b1 - b0,
            }
        )
    if not records:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)
    return pd.DataFrame(records)


# =============================================================================
# Graph / PLM
# =============================================================================


def _plm_seed(base_seed: int, resolution: int = 0, gamma: float = 1.0, salt: int = 0) -> int:
    return int(base_seed) + int(resolution) + int(round(float(gamma) * 1000)) + int(salt)


def _graph_from_coo_edges(
    n: int,
    rows: np.ndarray,
    cols: np.ndarray,
    weights: np.ndarray,
) -> "nk.graph.Graph":
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
    nk.setSeed(int(seed), True)
    algorithm = nk.community.PLM(
        graph, refine=True, gamma=float(gamma), par=str(par)
    )
    algorithm.run()
    return np.asarray(algorithm.getPartition().getVector(), dtype=np.int64)


def _fetch_sparse(clr: cooler.Cooler, region1, region2=None, balance: bool = False):
    try:
        if region2 is None:
            return clr.matrix(balance=balance, sparse=True).fetch(region1), balance
        return clr.matrix(balance=balance, sparse=True).fetch(region1, region2), balance
    except (ValueError, KeyError) as exc:
        if not balance:
            raise
        print(f"  warning: balanced matrix unavailable ({exc}); using raw counts")
        if region2 is None:
            return clr.matrix(balance=False, sparse=True).fetch(region1), False
        return clr.matrix(balance=False, sparse=True).fetch(region1, region2), False


def _coo_upper_edges(coo) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = (coo.row < coo.col) & np.isfinite(coo.data) & (coo.data > 0)
    return (
        np.asarray(coo.row[valid], dtype=np.int64),
        np.asarray(coo.col[valid], dtype=np.int64),
        np.asarray(coo.data[valid], dtype=float),
    )


def build_joint_contact_graph(
    mcool_path: Path,
    chrom_a: str,
    chrom_b: str,
    resolution: int,
    edge_quantile: float,
    balance: bool,
) -> tuple["nk.graph.Graph", int, int, int]:
    """Joint graph: A-intra + B-intra + A↔B inter, thresholded together.

    Returns (graph, n_a, n_nodes, n_edges). Nodes [0,n_a) = A, [n_a,n) = B.
    """
    clr = cooler.Cooler(f"{mcool_path}::/resolutions/{resolution}")
    mat_aa, _ = _fetch_sparse(clr, chrom_a, balance=balance)
    mat_bb, _ = _fetch_sparse(clr, chrom_b, balance=balance)
    mat_ab, _ = _fetch_sparse(clr, chrom_a, chrom_b, balance=balance)
    n_a = int(mat_aa.shape[0])
    n_b = int(mat_bb.shape[0])
    n = n_a + n_b

    ra, ca, wa = _coo_upper_edges(mat_aa.tocoo())
    rb, cb, wb = _coo_upper_edges(mat_bb.tocoo())
    ab = mat_ab.tocoo()
    valid = np.isfinite(ab.data) & (ab.data > 0)
    # undirected: store row < col in joint index space
    row_ab = ab.row[valid].astype(np.int64)  # in A
    col_ab = ab.col[valid].astype(np.int64) + n_a  # in B offset
    w_ab = np.asarray(ab.data[valid], dtype=float)

    rows = np.concatenate([ra, rb + n_a, row_ab])
    cols = np.concatenate([ca, cb + n_a, col_ab])
    weights = np.concatenate([wa, wb, w_ab])
    if weights.size == 0:
        raise RuntimeError(
            f"no positive contacts for {chrom_a}/{chrom_b} at resolution {resolution}"
        )
    threshold = float(np.quantile(weights, edge_quantile))
    keep = weights >= threshold
    graph = _graph_from_coo_edges(n, rows[keep], cols[keep], weights[keep])
    if graph.numberOfEdges() == 0:
        raise RuntimeError(
            f"contact threshold removed every edge for {chrom_a}/{chrom_b} "
            f"at resolution {resolution}"
        )
    return graph, n_a, graph.numberOfNodes(), graph.numberOfEdges()


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


# =============================================================================
# Enrichment (inter-chrom O/E; no distance decay)
# =============================================================================


def block_mean_inter(
    clr: cooler.Cooler,
    chrom_a: str,
    a0: int,
    a1: int,
    chrom_b: str,
    b0: int,
    b1: int,
) -> float:
    """Mean raw contact in rectangle chrom_a[a0,a1) × chrom_b[b0,b1)."""
    a0, a1 = max(0, int(a0)), max(int(a0) + 1, int(a1))
    b0, b1 = max(0, int(b0)), max(int(b0) + 1, int(b1))
    mat = np.asarray(
        clr.matrix(balance=False).fetch((chrom_a, a0, a1), (chrom_b, b0, b1)),
        dtype=float,
    )
    mat = mat[np.isfinite(mat)]
    if mat.size == 0:
        return 0.0
    return float(np.mean(np.maximum(mat, 0.0)))


def chrom_pair_mean_contact(
    clr: cooler.Cooler, chrom_a: str, chrom_b: str
) -> float:
    """Genome-wide mean raw inter-chrom contact for the pair (E for O/E)."""
    mat = np.asarray(clr.matrix(balance=False).fetch(chrom_a, chrom_b), dtype=float)
    mat = mat[np.isfinite(mat)]
    if mat.size == 0:
        return 0.0
    return float(np.mean(np.maximum(mat, 0.0)))


def _interval_tips(start: int, end: int, tip_bp: int) -> list[tuple[int, int]]:
    s, e = int(start), int(end)
    if e <= s:
        e = s + 1
    length = e - s
    w = max(1, int(tip_bp))
    if length <= w * 2:
        return [(s, e)]
    return [(s, s + w), (e - w, e)]


def score_inter_event(
    clr: cooler.Cooler,
    chrom_a: str,
    chrom_b: str,
    size_a: int,
    size_b: int,
    a0: int,
    a1: int,
    b0: int,
    b1: int,
    pair_mean: float,
    tip_window_bp: int = 2_000_000,
    resolution: int = 50_000,
) -> dict:
    """Primary fold = max tip×tip O/E; block + flank kept for audit.

    Expected = chrom-pair mean contact (no genomic-distance decay across chroms).
    Flank = mean of A-block × (left/right of B-block clipped to chrom B).
    """
    cross_block = block_mean_inter(clr, chrom_a, a0, a1, chrom_b, b0, b1)
    expected = float(pair_mean)
    if expected > 0:
        fold_block = cross_block / expected
    else:
        fold_block = float("inf") if cross_block > 0 else 0.0

    tip_bp = max(int(resolution), int(tip_window_bp))
    tips_a = _interval_tips(a0, a1, tip_bp)
    tips_b = _interval_tips(b0, b1, tip_bp)
    best = {
        "fold": 0.0,
        "cross": 0.0,
        "cand_tip": tips_a[0],
        "partner_tip": tips_b[0],
    }
    for ca0, ca1 in tips_a:
        for pb0, pb1 in tips_b:
            cross = block_mean_inter(clr, chrom_a, ca0, ca1, chrom_b, pb0, pb1)
            fold = (cross / expected) if expected > 0 else (float("inf") if cross > 0 else 0.0)
            if fold > best["fold"]:
                best = {
                    "fold": fold,
                    "cross": cross,
                    "cand_tip": (ca0, ca1),
                    "partner_tip": (pb0, pb1),
                }

    # flank on B: windows of same length as B-block immediately left/right
    blen = max(1, int(b1) - int(b0))
    flanks = []
    left1 = max(0, int(b0) - blen)
    left0 = left1
    left1 = int(b0)
    if left1 > left0:
        flanks.append((left0, left1))
    right0 = int(b1)
    right1 = min(int(size_b), int(b1) + blen)
    if right1 > right0:
        flanks.append((right0, right1))
    flank_vals = [
        block_mean_inter(clr, chrom_a, a0, a1, chrom_b, s, e) for s, e in flanks
    ]
    neighbor_mean = float(np.mean(flank_vals)) if flank_vals else 0.0
    if neighbor_mean > 0:
        fold_flank = cross_block / neighbor_mean
    else:
        fold_flank = float("inf") if cross_block > 0 else 0.0

    ct0, ct1 = best["cand_tip"]
    pt0, pt1 = best["partner_tip"]
    return {
        "enrichment_fold": float(best["fold"]),
        "enrichment_fold_block": float(fold_block),
        "enrichment_fold_flank": float(fold_flank),
        "cross_mean": float(best["cross"]),
        "cross_mean_block": float(cross_block),
        "expected_mean": float(expected),
        "neighbor_mean": float(neighbor_mean),
        "tip_window_bp": float(tip_bp),
        "tip_a_start": float(ct0),
        "tip_a_end": float(ct1),
        "tip_b_start": float(pt0),
        "tip_b_end": float(pt1),
    }


def score_enrichment_for_calls(
    calls: pd.DataFrame,
    mcool: Path,
    chrom_a: str,
    chrom_b: str,
    size_a: int,
    size_b: int,
    resolution: int,
    tip_window_bp: int = 2_000_000,
) -> pd.DataFrame:
    clr = cooler.Cooler(f"{mcool}::/resolutions/{resolution}")
    pair_mean = chrom_pair_mean_contact(clr, chrom_a, chrom_b)
    rows_metrics = []
    for _, row in calls.iterrows():
        rows_metrics.append(
            score_inter_event(
                clr,
                chrom_a,
                chrom_b,
                size_a,
                size_b,
                int(row["start_a"]),
                int(row["end_a"]),
                int(row["start_b"]),
                int(row["end_b"]),
                pair_mean=pair_mean,
                tip_window_bp=tip_window_bp,
                resolution=resolution,
            )
        )
    out = calls.copy()
    keys = [
        "enrichment_fold",
        "enrichment_fold_block",
        "enrichment_fold_flank",
        "cross_mean",
        "cross_mean_block",
        "expected_mean",
        "neighbor_mean",
        "tip_window_bp",
        "tip_a_start",
        "tip_a_end",
        "tip_b_start",
        "tip_b_end",
    ]
    for key in keys:
        out[key] = [float(m[key]) for m in rows_metrics]
    for col in ("enrichment_fold", "enrichment_fold_block", "enrichment_fold_flank"):
        out[col] = out[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


# =============================================================================
# Analyze one scale / merge / gate
# =============================================================================


def analyze_scale(
    mcool_path: Path,
    chrom_a: str,
    chrom_b: str,
    size_a: int,
    size_b: int,
    resolution: int,
    gamma: float,
    edge_quantile: float,
    balance: bool,
    seed: int,
    min_segment_bp: int,
    bridge_bp: int,
    networkit_threads: int,
    plm_par: str = "balanced",
) -> tuple[int, float, int, int, pd.DataFrame]:
    n_threads = 1 if str(plm_par) == "none" else max(1, networkit_threads)
    if hasattr(nk, "setNumberOfThreads"):
        nk.setNumberOfThreads(n_threads)
    graph, n_a, nodes, edges = build_joint_contact_graph(
        mcool_path, chrom_a, chrom_b, resolution, edge_quantile, balance
    )
    labels = _run_plm(
        graph,
        gamma=gamma,
        seed=_plm_seed(seed, resolution=resolution, gamma=gamma),
        par=plm_par,
    )
    cands = detect_inter_candidates(
        labels,
        n_a,
        resolution,
        gamma,
        chrom_a,
        chrom_b,
        size_a,
        size_b,
        min_segment_bp,
        bridge_bp,
    )
    return resolution, gamma, nodes, edges, cands


def _overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def _reciprocal(a0: int, a1: int, b0: int, b1: int) -> float:
    o = _overlap(a0, a1, b0, b1)
    if o <= 0:
        return 0.0
    return min(o / max(1, a1 - a0), o / max(1, b1 - b0))


def _interval_merge_score(a0: int, a1: int, b0: int, b1: int) -> float:
    rec = _reciprocal(a0, a1, b0, b1)
    ov = _overlap(a0, a1, b0, b1)
    if ov <= 0:
        return 0.0
    smaller = min(max(1, a1 - a0), max(1, b1 - b0))
    return max(rec, ov / smaller)


def _pair_merge_score(row_i, row_j, min_rec: float) -> bool:
    """True if both arms overlap enough (orientation-aware via chrom match)."""
    if str(row_i["chrom_a"]) != str(row_j["chrom_a"]) or str(row_i["chrom_b"]) != str(
        row_j["chrom_b"]
    ):
        return False
    sa = _interval_merge_score(
        int(row_i["start_a"]),
        int(row_i["end_a"]),
        int(row_j["start_a"]),
        int(row_j["end_a"]),
    )
    sb = _interval_merge_score(
        int(row_i["start_b"]),
        int(row_i["end_b"]),
        int(row_j["start_b"]),
        int(row_j["end_b"]),
    )
    return sa >= min_rec and sb >= min_rec


def merge_true_across_resolutions(
    true_tables: dict,
    min_rec: float = 0.5,
) -> pd.DataFrame:
    rows = []
    for key, df in sorted(
        true_tables.items(),
        key=lambda kv: (
            int(kv[0][0]) if isinstance(kv[0], tuple) else int(kv[0]),
            float(kv[0][1]) if isinstance(kv[0], tuple) else float("nan"),
        ),
    ):
        if isinstance(key, tuple):
            res, gamma = int(key[0]), float(key[1])
        else:
            res, gamma = int(key), float("nan")
        if df is None or df.empty:
            continue
        for _, r in df.iterrows():
            g = (
                float(r["gamma"])
                if "gamma" in df.columns and pd.notna(r.get("gamma"))
                else gamma
            )
            rows.append(
                {
                    "resolution": int(res),
                    "gamma": g,
                    "local_cluster_id": int(r.get("cluster_id", 0) or 0),
                    "chrom_a": str(r["chrom_a"]),
                    "start_a": int(r["start_a"]),
                    "end_a": int(r["end_a"]),
                    "chrom_b": str(r["chrom_b"]),
                    "start_b": int(r["start_b"]),
                    "end_b": int(r["end_b"]),
                    "enrichment_fold": float(r["enrichment_fold"]),
                    "enrichment_fold_block": float(
                        r["enrichment_fold_block"]
                        if "enrichment_fold_block" in r.index
                        and pd.notna(r.get("enrichment_fold_block"))
                        else r["enrichment_fold"]
                    ),
                    "enrichment_fold_flank": float(
                        r["enrichment_fold_flank"]
                        if "enrichment_fold_flank" in r.index
                        and pd.notna(r.get("enrichment_fold_flank"))
                        else r["enrichment_fold"]
                    ),
                }
            )
    if not rows:
        return empty_inter_tra_table()

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
            if _pair_merge_score(items.loc[i], items.loc[j], min_rec):
                union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    merged = []
    for idxs in sorted(clusters.values(), key=lambda ix: (items.loc[ix[0], "chrom_a"], items.loc[ix[0], "start_a"])):
        sub = items.loc[idxs].copy()
        sub["size_a"] = sub["end_a"] - sub["start_a"]
        sub["size_b"] = sub["end_b"] - sub["start_b"]
        res_list = sorted(set(int(x) for x in sub["resolution"]))
        scale_keys = sorted(
            {
                (int(r.resolution), float(r.gamma) if pd.notna(r.gamma) else float("nan"))
                for _, r in sub.iterrows()
            }
        )
        rep = sub.sort_values(
            ["size_a", "size_b", "enrichment_fold"], ascending=False
        ).iloc[0]
        gammas = sorted({float(x) for x in sub["gamma"] if pd.notna(x)})
        merged.append(
            {
                "cluster_id": len(merged) + 1,
                "chrom_a": rep["chrom_a"],
                "start_a": int(rep["start_a"]),
                "end_a": int(rep["end_a"]),
                "chrom_b": rep["chrom_b"],
                "start_b": int(rep["start_b"]),
                "end_b": int(rep["end_b"]),
                "enrichment_fold": float(sub["enrichment_fold"].max()),
                "enrichment_fold_block": float(sub["enrichment_fold_block"].max()),
                "enrichment_fold_flank": float(sub["enrichment_fold_flank"].max()),
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
                "size_a": int(rep["end_a"]) - int(rep["start_a"]),
                "size_b": int(rep["end_b"]) - int(rep["start_b"]),
            }
        )
    if not merged:
        return empty_inter_tra_table()

    out = pd.DataFrame(merged).sort_values(
        ["chrom_a", "start_a", "chrom_b", "start_b"]
    ).reset_index(drop=True)

    # drop overlapping duplicates, keep higher scale_support
    keep = [True] * len(out)
    for i in range(len(out)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(out)):
            if not keep[j]:
                continue
            if not _pair_merge_score(out.loc[i], out.loc[j], min_rec):
                continue
            si = (
                int(out.at[i, "scale_support"]),
                int(out.at[i, "size_a"]) + int(out.at[i, "size_b"]),
                float(out.at[i, "enrichment_fold"]),
            )
            sj = (
                int(out.at[j, "scale_support"]),
                int(out.at[j, "size_a"]) + int(out.at[j, "size_b"]),
                float(out.at[j, "enrichment_fold"]),
            )
            if sj > si:
                keep[i] = False
                break
            keep[j] = False
    out = out.loc[keep].reset_index(drop=True)
    out["cluster_id"] = np.arange(1, len(out) + 1, dtype=int)
    return out


def apply_enrichment_gate(
    merged: pd.DataFrame,
    min_enrichment_fold: float = 1.5,
    outdir: Path | None = None,
) -> pd.DataFrame:
    if merged is None or not isinstance(merged, pd.DataFrame):
        merged = empty_inter_tra_table()
    out = merged.copy()
    if outdir is not None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        out.to_csv(outdir / "ALL_INTER_TRA_pre_enrichment.tsv", sep="\t", index=False)
    if len(out) == 0:
        if outdir is not None:
            empty_inter_tra_table().to_csv(
                outdir / "ALL_INTER_TRA_enrichment_dropped.tsv", sep="\t", index=False
            )
        return empty_inter_tra_table()

    min_fold = float(min_enrichment_fold)
    if min_fold <= 0:
        print("[INFO] enrichment gate disabled (min-enrichment-fold<=0)", flush=True)
        return out
    if "enrichment_fold" not in out.columns:
        print("[WARN] enrichment_fold missing; keeping all", flush=True)
        return out

    folds = pd.to_numeric(out["enrichment_fold"], errors="coerce").fillna(0.0)
    keep = folds > min_fold
    dropped = out.loc[~keep].copy()
    kept = out.loc[keep].reset_index(drop=True)
    if len(kept):
        kept = kept.copy()
        kept["cluster_id"] = np.arange(1, len(kept) + 1, dtype=int)
    print(
        f"[INFO] enrichment gate O/E fold>{min_fold:g}: "
        f"{len(out)} -> {len(kept)} kept ({int((~keep).sum())} dropped)",
        flush=True,
    )
    if outdir is not None and len(dropped):
        dropped.to_csv(
            outdir / "ALL_INTER_TRA_enrichment_dropped.tsv", sep="\t", index=False
        )
    return kept


# =============================================================================
# Reciprocal optimize-merge (inter analog of intra clear-insert / mutual-insert)
# =============================================================================


def _calls_abut(a0: int, a1: int, b0: int, b1: int, slack: int = 500_000) -> bool:
    """True if intervals touch, overlap, or nearly touch."""
    if a1 <= b0:
        return (b0 - a1) <= slack
    if b1 <= a0:
        return (a0 - b1) <= slack
    return _overlap(a0, a1, b0, b1) > 0


def collapse_overlapping_intervals(
    intervals: list[tuple[int, int]], min_rec: float = 0.5
) -> list[tuple[int, int]]:
    """Cluster high-overlap intervals; abutting neighbors stay separate pieces."""
    xs = [(int(a), int(b)) for a, b in intervals if b > a]
    if not xs:
        return []
    n = len(xs)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        for j in range(i + 1, n):
            if _reciprocal(xs[i][0], xs[i][1], xs[j][0], xs[j][1]) >= min_rec:
                union(i, j)

    groups: dict[int, list[tuple[int, int]]] = {}
    for i, iv in enumerate(xs):
        groups.setdefault(find(i), []).append(iv)
    segs = [
        (min(a for a, _ in members), max(b for _, b in members))
        for members in groups.values()
    ]
    return sorted(segs, key=lambda s: s[0])


def _connected_call_components(
    calls: pd.DataFrame, slack: int = 500_000
) -> list[list[int]]:
    n = len(calls)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        for j in range(i + 1, n):
            if str(calls.at[i, "chrom_a"]) != str(calls.at[j, "chrom_a"]):
                continue
            if str(calls.at[i, "chrom_b"]) != str(calls.at[j, "chrom_b"]):
                continue
            if _calls_abut(
                int(calls.at[i, "start_a"]),
                int(calls.at[i, "end_a"]),
                int(calls.at[j, "start_a"]),
                int(calls.at[j, "end_a"]),
                slack,
            ) or _calls_abut(
                int(calls.at[i, "start_b"]),
                int(calls.at[i, "end_b"]),
                int(calls.at[j, "start_b"]),
                int(calls.at[j, "end_b"]),
                slack,
            ):
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def pick_interior_segment(segs: list[tuple[int, int]]) -> tuple[int, int]:
    """Left/right flanks + mid exchange → keep mid."""
    if not segs:
        raise ValueError("empty segments")
    if len(segs) == 1:
        return segs[0]
    if len(segs) == 2:
        return min(segs, key=lambda s: s[1] - s[0])
    interior = segs[1:-1]
    return (interior[0][0], interior[-1][1])


def _reconstruct_reciprocal_event(sub: pd.DataFrame) -> dict:
    segs_a = collapse_overlapping_intervals(
        [(int(r.start_a), int(r.end_a)) for _, r in sub.iterrows()]
    )
    segs_b = collapse_overlapping_intervals(
        [(int(r.start_b), int(r.end_b)) for _, r in sub.iterrows()]
    )
    a0, a1 = pick_interior_segment(segs_a)
    b0, b1 = pick_interior_segment(segs_b)
    member_res = ""
    if "member_resolutions" in sub.columns:
        member_res = ";".join(
            str(x) for x in sub["member_resolutions"].astype(str).tolist() if x
        )
    return {
        "chrom_a": str(sub.iloc[0]["chrom_a"]),
        "start_a": a0,
        "end_a": a1,
        "chrom_b": str(sub.iloc[0]["chrom_b"]),
        "start_b": b0,
        "end_b": b1,
        "size_a": a1 - a0,
        "size_b": b1 - b0,
        "n_members": int(sub["n_members"].sum())
        if "n_members" in sub.columns
        else len(sub),
        "n_source_calls": len(sub),
        "source_cluster_ids": ",".join(str(int(x)) for x in sub["cluster_id"]),
        "scale_support": int(sub["scale_support"].max()),
        "resolutions": ",".join(
            sorted(
                {
                    p
                    for s in sub["resolutions"].astype(str)
                    for p in s.split(",")
                    if p
                },
                key=lambda x: int(x),
            )
        ),
        "gammas": ",".join(
            sorted(
                {p for s in sub["gammas"].astype(str) for p in s.split(",") if p},
                key=lambda x: float(x),
            )
        ),
        "member_resolutions": member_res,
        "merged_reciprocal": 1,
        "enrichment_fold": float(sub["enrichment_fold"].max()),
        "enrichment_fold_block": float(sub["enrichment_fold_block"].max())
        if "enrichment_fold_block" in sub.columns
        else float(sub["enrichment_fold"].max()),
        "enrichment_fold_flank": float(sub["enrichment_fold_flank"].max())
        if "enrichment_fold_flank" in sub.columns
        else float(sub["enrichment_fold"].max()),
    }


def optimize_merge_inter_calls(
    calls: pd.DataFrame,
    mcool: Path | None = None,
    abut_bp: int = 500_000,
    tip_window_bp: int = 2_000_000,
    score_resolution: int = 50_000,
    rescore: bool = True,
) -> pd.DataFrame:
    """Merge abutting reciprocal corner calls into mid×mid exchange events.

    ponytail: geometry merge only (no ILP). Ceiling = conflicting multi-events
    on the same arm; upgrade = pulp like intra clear-insert if that appears.
    """
    if calls is None or not isinstance(calls, pd.DataFrame) or len(calls) == 0:
        return empty_inter_tra_table()

    rows: list[dict] = []
    for idxs in _connected_call_components(calls.reset_index(drop=True), slack=abut_bp):
        sub = calls.reset_index(drop=True).loc[idxs].copy()
        if len(sub) == 1:
            r = sub.iloc[0]
            rows.append(
                {
                    "chrom_a": str(r["chrom_a"]),
                    "start_a": int(r["start_a"]),
                    "end_a": int(r["end_a"]),
                    "chrom_b": str(r["chrom_b"]),
                    "start_b": int(r["start_b"]),
                    "end_b": int(r["end_b"]),
                    "size_a": int(r["end_a"]) - int(r["start_a"]),
                    "size_b": int(r["end_b"]) - int(r["start_b"]),
                    "n_members": int(r["n_members"])
                    if "n_members" in sub.columns
                    else 1,
                    "n_source_calls": 1,
                    "source_cluster_ids": str(int(r["cluster_id"])),
                    "scale_support": int(r["scale_support"]),
                    "resolutions": str(r["resolutions"]),
                    "gammas": str(r["gammas"]),
                    "member_resolutions": str(r["member_resolutions"])
                    if "member_resolutions" in sub.columns
                    else "",
                    "merged_reciprocal": 0,
                    "enrichment_fold": float(r["enrichment_fold"]),
                    "enrichment_fold_block": float(
                        r["enrichment_fold_block"]
                        if "enrichment_fold_block" in sub.columns
                        else r["enrichment_fold"]
                    ),
                    "enrichment_fold_flank": float(
                        r["enrichment_fold_flank"]
                        if "enrichment_fold_flank" in sub.columns
                        else r["enrichment_fold"]
                    ),
                }
            )
        else:
            rows.append(_reconstruct_reciprocal_event(sub))

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return empty_inter_tra_table()

    if rescore and mcool is not None:
        available = set(discover_resolutions(Path(mcool)))
        use_res = (
            score_resolution
            if score_resolution in available
            else min(available, key=lambda r: abs(r - score_resolution))
        )
        for (ca, cb), idx in out.groupby(["chrom_a", "chrom_b"], sort=False).groups.items():
            clr = cooler.Cooler(f"{mcool}::/resolutions/{use_res}")
            size_a = int(clr.chromsizes[str(ca)])
            size_b = int(clr.chromsizes[str(cb)])
            pair_mean = chrom_pair_mean_contact(clr, str(ca), str(cb))
            for i in idx:
                m = score_inter_event(
                    clr,
                    str(ca),
                    str(cb),
                    size_a,
                    size_b,
                    int(out.at[i, "start_a"]),
                    int(out.at[i, "end_a"]),
                    int(out.at[i, "start_b"]),
                    int(out.at[i, "end_b"]),
                    pair_mean=pair_mean,
                    tip_window_bp=tip_window_bp,
                    resolution=use_res,
                )
                for k in (
                    "enrichment_fold",
                    "enrichment_fold_block",
                    "enrichment_fold_flank",
                ):
                    val = float(m[k])
                    if not np.isfinite(val):
                        val = 0.0
                    out.at[i, k] = val

    out = out.sort_values(
        ["chrom_a", "start_a", "chrom_b", "start_b"]
    ).reset_index(drop=True)
    out.insert(0, "cluster_id", np.arange(1, len(out) + 1, dtype=int))
    ordered = [c for c in ALL_INTER_TRA_COLUMNS if c in out.columns]
    extra = [c for c in out.columns if c not in ordered]
    return out.loc[:, ordered + extra]


def apply_optimize_merge(
    merged: pd.DataFrame,
    mcool: Path,
    outdir: Path | None = None,
    abut_bp: int = 500_000,
    tip_window_bp: int = 2_000_000,
) -> pd.DataFrame:
    """Write pre-optimize table, run optimize-merge, return optimized calls."""
    if merged is None or not isinstance(merged, pd.DataFrame):
        merged = empty_inter_tra_table()
    if outdir is not None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        merged.to_csv(outdir / "ALL_INTER_TRA_pre_optimize.tsv", sep="\t", index=False)
    if len(merged) == 0:
        return empty_inter_tra_table()
    n_before = len(merged)
    opt = optimize_merge_inter_calls(
        merged,
        mcool=Path(mcool),
        abut_bp=abut_bp,
        tip_window_bp=tip_window_bp,
        rescore=True,
    )
    n_merged = int(opt["merged_reciprocal"].sum()) if len(opt) and "merged_reciprocal" in opt.columns else 0
    print(
        f"[INFO] reciprocal optimize-merge: {n_before} -> {len(opt)} "
        f"({n_merged} reciprocal event(s) collapsed)",
        flush=True,
    )
    return opt


# =============================================================================
# Plot (optional)
# =============================================================================


def plot_inter_pair(
    mcool: Path,
    chrom_a: str,
    chrom_b: str,
    size_a: int,
    size_b: int,
    calls: pd.DataFrame,
    out_path: Path,
    plot_res: int = 500_000,
) -> None:
    available = discover_resolutions(mcool)
    use_plot = plot_res if plot_res in available else max(available)
    clr = cooler.Cooler(f"{mcool}::/resolutions/{use_plot}")
    mat = np.asarray(clr.matrix(balance=False).fetch(chrom_a, chrom_b), dtype=float)
    mat = np.nan_to_num(mat, nan=0.0)
    positive = mat[mat > 0]
    vmax = float(np.percentile(positive, 99)) if positive.size else 1.0
    vmin = max(float(np.percentile(positive, 5)), 1e-6) if positive.size else 1e-6

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(
        mat,
        origin="lower",
        aspect="auto",
        norm=LogNorm(vmin=vmin, vmax=max(vmax, vmin * 10)),
        extent=[0, size_b / 1e6, 0, size_a / 1e6],
        interpolation="nearest",
    )
    for _, r in calls.iterrows():
        ax.add_patch(
            plt.Rectangle(
                (int(r["start_b"]) / 1e6, int(r["start_a"]) / 1e6),
                (int(r["end_b"]) - int(r["start_b"])) / 1e6,
                (int(r["end_a"]) - int(r["start_a"])) / 1e6,
                fill=False,
                edgecolor="cyan",
                linewidth=1.5,
            )
        )
    ax.set_xlabel(f"{chrom_b} (Mb)")
    ax.set_ylabel(f"{chrom_a} (Mb)")
    ax.set_title(f"Inter-TRA {chrom_a} × {chrom_b}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_joint_hic_annotated(
    mcool: Path,
    chrom_a: str,
    chrom_b: str,
    size_a: int,
    size_b: int,
    calls: pd.DataFrame,
    out_path: Path,
    plot_res: int = 500_000,
) -> None:
    """Concatenated A+B Hi-C with TRA boxes on AB/BA quadrants."""
    available = discover_resolutions(mcool)
    use_plot = plot_res if plot_res in available else max(available)
    clr = cooler.Cooler(f"{mcool}::/resolutions/{use_plot}")
    aa = np.nan_to_num(clr.matrix(balance=False).fetch(chrom_a), nan=0.0)
    bb = np.nan_to_num(clr.matrix(balance=False).fetch(chrom_b), nan=0.0)
    ab = np.nan_to_num(clr.matrix(balance=False).fetch(chrom_a, chrom_b), nan=0.0)
    mat = np.block([[aa, ab], [ab.T, bb]])
    pos = mat[mat > 0]
    vmax = float(np.percentile(pos, 99)) if pos.size else 1.0
    vmin = max(float(np.percentile(pos, 5)), 1e-6) if pos.size else 1e-6
    xmax = (size_a + size_b) / 1e6

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(
        mat,
        origin="lower",
        aspect="equal",
        norm=LogNorm(vmin=vmin, vmax=max(vmax, vmin * 10)),
        extent=[0, xmax, 0, xmax],
        interpolation="nearest",
        cmap="YlOrRd",
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="contacts")
    ax.axvline(size_a / 1e6, color="white", lw=1.2, ls="--")
    ax.axhline(size_a / 1e6, color="white", lw=1.2, ls="--")
    ax.set_xticks([size_a / 2e6, size_a / 1e6 + size_b / 2e6])
    ax.set_xticklabels([chrom_a, chrom_b])
    ax.set_yticks([size_a / 2e6, size_a / 1e6 + size_b / 2e6])
    ax.set_yticklabels([chrom_a, chrom_b])
    for _, r in calls.iterrows():
        a0, a1 = int(r["start_a"]) / 1e6, int(r["end_a"]) / 1e6
        b0, b1 = int(r["start_b"]) / 1e6, int(r["end_b"]) / 1e6
        ax.add_patch(
            plt.Rectangle(
                (size_a / 1e6 + b0, a0),
                b1 - b0,
                a1 - a0,
                fill=False,
                edgecolor="cyan",
                lw=2.0,
            )
        )
        ax.add_patch(
            plt.Rectangle(
                (a0, size_a / 1e6 + b0),
                a1 - a0,
                b1 - b0,
                fill=False,
                edgecolor="cyan",
                lw=2.0,
            )
        )
    ax.set_title(f"Inter-TRA Hi-C {chrom_a}+{chrom_b} ({use_plot // 1000} kb)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_joint_community(
    mcool: Path,
    chrom_a: str,
    chrom_b: str,
    size_a: int,
    size_b: int,
    calls: pd.DataFrame,
    out_path: Path,
    resolution: int = 200_000,
    gamma: float = 1.0,
    edge_quantile: float = 0.9,
    seed: int = 42,
    min_segment_bp: int = 500_000,
    bridge_bp: int = 500_000,
) -> None:
    """One joint community track (A+B concatenated) for the PLM partition."""
    available = discover_resolutions(mcool)
    if resolution not in available:
        resolution = min(available, key=lambda r: abs(r - resolution))
    graph, n_a, nodes, edges = build_joint_contact_graph(
        mcool, chrom_a, chrom_b, resolution, edge_quantile, balance=False
    )
    labels = _run_plm(
        graph,
        gamma=gamma,
        seed=_plm_seed(seed, resolution=resolution, gamma=gamma),
    )
    segs = run_length_segments(labels)
    cands = detect_inter_candidates(
        labels,
        n_a,
        resolution,
        gamma,
        chrom_a,
        chrom_b,
        size_a,
        size_b,
        min_segment_bp,
        bridge_bp,
    )
    shared = set(int(x) for x in cands["label"].tolist()) if len(cands) else set()
    total = size_a + size_b
    ymax = float(segs["label"].max()) if len(segs) else 1.0

    fig, ax = plt.subplots(figsize=(14, 4.5))
    for _, s in segs.iterrows():
        lab = int(s["label"])
        b0, b1 = int(s["start_bin"]), int(s["end_bin"])
        if b1 <= n_a:
            x0, x1 = b0 * resolution, min(b1 * resolution, size_a)
        elif b0 >= n_a:
            x0 = size_a + (b0 - n_a) * resolution
            x1 = size_a + min((b1 - n_a) * resolution, size_b)
        else:
            x0, x1 = b0 * resolution, size_a
            ax.plot(
                [x0 / 1e6, x1 / 1e6],
                [lab, lab],
                color="red" if lab in shared else "royalblue",
                lw=4.0 if lab in shared else 2.0,
                solid_capstyle="butt",
            )
            x0 = size_a
            x1 = size_a + min((b1 - n_a) * resolution, size_b)
        ax.plot(
            [x0 / 1e6, x1 / 1e6],
            [lab, lab],
            color="red" if lab in shared else "royalblue",
            lw=4.0 if lab in shared else 2.0,
            alpha=0.95 if lab in shared else 0.75,
            solid_capstyle="butt",
        )
    for _, r in calls.iterrows():
        ax.axvspan(
            int(r["start_a"]) / 1e6,
            int(r["end_a"]) / 1e6,
            color="#00ff66",
            alpha=0.15,
            zorder=0,
        )
        ax.axvspan(
            (size_a + int(r["start_b"])) / 1e6,
            (size_a + int(r["end_b"])) / 1e6,
            color="#00ff66",
            alpha=0.15,
            zorder=0,
        )
    ax.axvline(size_a / 1e6, color="k", ls="--", lw=1.2)
    ax.set_xlim(0, total / 1e6)
    ax.set_ylim(-2, ymax * 1.08)
    ax.set_xticks([size_a / 2e6, size_a / 1e6 + size_b / 2e6])
    ax.set_xticklabels([chrom_a, chrom_b])
    ax.set_ylabel("community id")
    ax.set_title(
        f"Joint PLM communities {chrom_a}+{chrom_b}  "
        f"(r={resolution // 1000} kb, γ={gamma:g}, nodes={nodes}, edges={edges})\n"
        f"red=shared TRA communities {sorted(shared)}; green=final TRA calls"
    )
    ax.grid(True, axis="x", alpha=0.25)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] community plot -> {out_path}", flush=True)


# =============================================================================
# Worker / pipeline
# =============================================================================


def _run_one_res_gamma(payload: dict) -> dict:
    res = int(payload["resolution"])
    gamma = float(payload["gamma"])
    mcool = Path(payload["mcool"])
    chrom_a = str(payload["chrom_a"])
    chrom_b = str(payload["chrom_b"])
    size_a = int(payload["size_a"])
    size_b = int(payload["size_b"])
    outdir = Path(payload["outdir"])
    n_threads = int(payload["n_threads"])
    tip_window_bp = int(payload.get("tip_window_bp", 2_000_000))
    min_true_fold = float(payload["min_true_fold"])
    skip_enrichment = bool(payload["skip_enrichment"])

    log_lines: list[str] = []

    def log(msg: str) -> None:
        log_lines.append(msg)

    log(f"\n======== {chrom_a}×{chrom_b}  r{res}  g={gamma:g} ========")
    try:
        _, _, nodes, edges, cands = analyze_scale(
            mcool,
            chrom_a,
            chrom_b,
            size_a,
            size_b,
            res,
            gamma,
            float(payload["edge_quantile"]),
            False,
            int(payload["seed"]),
            int(payload["min_segment_bp"]),
            int(payload["bridge_bp"]),
            n_threads,
            plm_par="balanced",
        )
    except RuntimeError as exc:
        log(f"[WARN] skip scale: {exc}")
        return {
            "resolution": res,
            "gamma": gamma,
            "chrom_a": chrom_a,
            "chrom_b": chrom_b,
            "empty": True,
            "log": "\n".join(log_lines),
        }

    log(f"[INFO] graph={nodes}/{edges} shared-community candidates={len(cands)}")
    if cands.empty:
        return {
            "resolution": res,
            "gamma": gamma,
            "chrom_a": chrom_a,
            "chrom_b": chrom_b,
            "empty": True,
            "log": "\n".join(log_lines),
        }

    calls = cands.copy()
    calls["cluster_id"] = np.arange(1, len(calls) + 1, dtype=int)
    if not skip_enrichment:
        calls = score_enrichment_for_calls(
            calls,
            mcool,
            chrom_a,
            chrom_b,
            size_a,
            size_b,
            res,
            tip_window_bp=tip_window_bp,
        )
        log(
            f"[INFO] tip O/E fold median={calls['enrichment_fold'].median():.3f} "
            f"max={calls['enrichment_fold'].max():.3f}"
        )
    else:
        calls["enrichment_fold"] = 1.0
        calls["enrichment_fold_block"] = 1.0
        calls["enrichment_fold_flank"] = 1.0

    # per-scale soft filter (like min_true_fold before merge)
    true = calls[calls["enrichment_fold"] >= min_true_fold].copy()
    true["role"] = "true"
    true["gamma"] = gamma

    if bool(payload.get("write_scale_dirs", False)):
        tag = f"{chrom_a}__{chrom_b}_r{res}_g{gamma:g}"
        res_dir = outdir / tag
        res_dir.mkdir(parents=True, exist_ok=True)
        calls.to_csv(res_dir / "calls_input.tsv", sep="\t", index=False)
        cands.to_csv(res_dir / "scale_candidates.tsv", sep="\t", index=False)
        true.to_csv(res_dir / "true_calls.tsv", sep="\t", index=False)

    log(f"[RES {res} g={gamma:g}] n_cand={len(calls)} n_true={len(true)}")
    return {
        "resolution": res,
        "gamma": gamma,
        "chrom_a": chrom_a,
        "chrom_b": chrom_b,
        "empty": False,
        "out": true,
        "cands": cands,
        "summary": {
            "chrom_a": chrom_a,
            "chrom_b": chrom_b,
            "resolution": res,
            "gamma": gamma,
            "n_candidates": len(calls),
            "n_true": len(true),
        },
        "log": "\n".join(log_lines),
    }


def _parse_chrom_pairs(
    args: argparse.Namespace, chromsizes: pd.Series
) -> list[tuple[str, str]]:
    names = list(chromsizes.index.astype(str))
    if args.chrom_a and args.chrom_b:
        for c in (args.chrom_a, args.chrom_b):
            if c not in chromsizes.index:
                raise SystemExit(f"[ERROR] chromosome {c} not in mcool")
        if args.chrom_a == args.chrom_b:
            raise SystemExit("[ERROR] --chrom-a and --chrom-b must differ")
        return [(args.chrom_a, args.chrom_b)]
    if args.chromosomes:
        wanted = [x.strip() for x in str(args.chromosomes).split(",") if x.strip()]
        for c in wanted:
            if c not in chromsizes.index:
                raise SystemExit(f"[ERROR] chromosome {c} not in mcool")
        return [(a, b) for a, b in combinations(wanted, 2)]
    # default: all unordered pairs (can be heavy)
    return list(combinations(names, 2))


def run_pipeline(args: argparse.Namespace) -> pd.DataFrame:
    resolutions = [int(x) for x in args.resolutions.split(",") if x.strip()]
    if str(args.gammas).strip():
        gammas = [float(x) for x in str(args.gammas).split(",") if x.strip()]
    else:
        gammas = [float(args.gamma)]
    args.outdir.mkdir(parents=True, exist_ok=True)
    plot_dir = args.outdir / "plots"
    if not args.skip_plot:
        plot_dir.mkdir(parents=True, exist_ok=True)

    available = discover_resolutions(args.mcool)
    use_resolutions = [r for r in resolutions if r in available]
    for r in resolutions:
        if r not in available:
            print(f"[WARN] resolution {r} not in mcool; skip", flush=True)
    if not use_resolutions:
        raise SystemExit("[ERROR] no requested resolutions present in mcool")

    clr0 = cooler.Cooler(f"{args.mcool}::/resolutions/{use_resolutions[0]}")
    chromsizes = clr0.chromsizes
    pairs = _parse_chrom_pairs(args, chromsizes)
    if not pairs:
        raise SystemExit("[ERROR] need at least two chromosomes")

    n_tasks = len(pairs) * len(use_resolutions) * len(gammas)
    if int(getattr(args, "jobs", 0) or 0) > 0:
        n_jobs = int(args.jobs)
    else:
        n_jobs = max(1, (os.cpu_count() or 4) // 2)
    n_jobs = max(1, min(n_jobs, n_tasks))
    if int(getattr(args, "threads", 0) or 0) > 0:
        n_threads = int(args.threads)
    else:
        n_threads = max(1, (os.cpu_count() or 4) // max(1, n_jobs))

    print(
        f"[INFO] pairs={len(pairs)} resolutions={use_resolutions} gammas={gammas} "
        f"jobs={n_jobs} threads/worker={n_threads}",
        flush=True,
    )

    payloads = []
    for chrom_a, chrom_b in pairs:
        size_a = int(chromsizes.loc[chrom_a])
        size_b = int(chromsizes.loc[chrom_b])
        for res in use_resolutions:
            for gamma in gammas:
                payloads.append(
                    {
                        "resolution": res,
                        "gamma": gamma,
                        "mcool": str(args.mcool),
                        "chrom_a": chrom_a,
                        "chrom_b": chrom_b,
                        "size_a": size_a,
                        "size_b": size_b,
                        "outdir": str(args.outdir),
                        "n_threads": n_threads,
                        "edge_quantile": float(args.edge_quantile),
                        "seed": int(args.seed),
                        "min_segment_bp": int(args.min_segment_bp),
                        "bridge_bp": int(args.bridge_bp),
                        "min_true_fold": float(args.min_true_fold),
                        "skip_enrichment": bool(args.skip_enrichment),
                        "tip_window_bp": int(args.tip_window_bp),
                        "write_scale_dirs": bool(args.write_scale_dirs),
                    }
                )

    results: list[dict] = []
    if n_jobs == 1:
        for payload in payloads:
            results.append(_run_one_res_gamma(payload))
    else:
        print(
            f"[INFO] parallel: {n_tasks} tasks, {n_jobs} workers",
            flush=True,
        )
        with ProcessPoolExecutor(
            max_workers=n_jobs, mp_context=mp.get_context("spawn")
        ) as ex:
            futs = {ex.submit(_run_one_res_gamma, p): p for p in payloads}
            for fut in as_completed(futs):
                results.append(fut.result())

    results.sort(
        key=lambda r: (
            str(r.get("chrom_a", "")),
            str(r.get("chrom_b", "")),
            int(r["resolution"]),
            float(r["gamma"]),
        )
    )

    true_tables: dict = {}
    summary_rows = []
    all_cands: list[pd.DataFrame] = []
    for result in results:
        print(result.get("log", ""), flush=True)
        if result.get("empty"):
            continue
        key = (
            str(result["chrom_a"]),
            str(result["chrom_b"]),
            int(result["resolution"]),
            float(result["gamma"]),
        )
        # merge_true expects (res, gamma) keys; keep pair in the dataframe
        true_tables[(key[2], key[3], key[0], key[1])] = result["out"]
        summary_rows.append(result["summary"])
        if result.get("cands") is not None and len(result["cands"]):
            all_cands.append(result["cands"])

    compact = bool(getattr(args, "compact", True))
    aux_outdir = None if compact else args.outdir

    # Merge per chrom-pair, then concat.
    merged_frames: list[pd.DataFrame] = []
    pair_keys = sorted({(str(k[2]), str(k[3])) for k in true_tables})
    for ca, cb in pair_keys:
        sub = {
            (res, gamma): df
            for (res, gamma, a, b), df in true_tables.items()
            if a == ca and b == cb
        }
        m = merge_true_across_resolutions(sub, min_rec=args.merge_min_rec)
        if len(m):
            merged_frames.append(m)

    if merged_frames:
        merged = pd.concat(merged_frames, ignore_index=True)
        merged["cluster_id"] = np.arange(1, len(merged) + 1, dtype=int)
        merged = apply_enrichment_gate(
            merged,
            min_enrichment_fold=float(args.min_enrichment_fold),
            outdir=aux_outdir,
        )
    else:
        merged = empty_inter_tra_table()

    if not args.skip_optimize_merge and len(merged):
        merged = apply_optimize_merge(
            merged,
            mcool=args.mcool,
            outdir=aux_outdir,
            abut_bp=int(args.optimize_abut_bp),
            tip_window_bp=int(args.tip_window_bp),
        )
    elif len(merged) and not compact:
        merged.to_csv(
            args.outdir / "ALL_INTER_TRA_pre_optimize.tsv", sep="\t", index=False
        )
        print("[INFO] reciprocal optimize-merge skipped", flush=True)

    ordered = [c for c in ALL_INTER_TRA_COLUMNS if c in merged.columns]
    extra = [c for c in merged.columns if c not in ordered]
    if ordered or extra:
        merged = merged.loc[:, ordered + extra]

    tra_path = args.outdir / "ALL_INTER_TRA.tsv"
    merged.to_csv(tra_path, sep="\t", index=False)
    print(f"\n[INFO] ALL_INTER_TRA.tsv -> {tra_path}  (n={len(merged)})", flush=True)

    if not compact:
        merged.to_csv(args.outdir / "merged_true_calls.tsv", sep="\t", index=False)
        if all_cands:
            pd.concat(all_cands, ignore_index=True).to_csv(
                args.outdir / "ALL_INTER_TRA_per_scale_candidates.tsv",
                sep="\t",
                index=False,
            )
        if summary_rows:
            pd.DataFrame(summary_rows).to_csv(
                args.outdir / "per_resolution_summary.tsv", sep="\t", index=False
            )

    # final plots only: community + annotated Hi-C
    if not args.skip_plot and len(merged):
        plot_dir.mkdir(parents=True, exist_ok=True)
        for (ca, cb), sub in merged.groupby(["chrom_a", "chrom_b"], sort=False):
            ca, cb = str(ca), str(cb)
            size_a = int(chromsizes.loc[ca])
            size_b = int(chromsizes.loc[cb])
            plot_joint_community(
                args.mcool,
                ca,
                cb,
                size_a,
                size_b,
                sub,
                plot_dir / f"community_{ca}_{cb}.png",
                resolution=200_000,
                gamma=1.0,
                edge_quantile=float(args.edge_quantile),
                seed=int(args.seed),
                min_segment_bp=int(args.min_segment_bp),
                bridge_bp=int(args.bridge_bp),
            )
            plot_joint_hic_annotated(
                args.mcool,
                ca,
                cb,
                size_a,
                size_b,
                sub,
                plot_dir / f"hic_{ca}_{cb}_annotated.png",
            )
            print(
                f"[INFO] Hi-C annotated -> {plot_dir / f'hic_{ca}_{cb}_annotated.png'}",
                flush=True,
            )
    return merged


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mcool", type=Path, required=True, help="input .mcool file")
    ap.add_argument(
        "--outdir",
        type=Path,
        default=Path("INTER_TRA_results"),
        help="output directory (default: INTER_TRA_results)",
    )
    ap.add_argument(
        "--chrom-a",
        type=str,
        default=None,
        help="first chromosome (pair mode with --chrom-b)",
    )
    ap.add_argument(
        "--chrom-b",
        type=str,
        default=None,
        help="second chromosome (pair mode with --chrom-a)",
    )
    ap.add_argument(
        "--chromosomes",
        type=str,
        default=None,
        help="comma-separated chroms: run all unordered pairs "
        "(overrides nothing if --chrom-a/--chrom-b set)",
    )
    ap.add_argument(
        "--resolutions",
        type=str,
        default=",".join(str(r) for r in RESOLUTIONS_DEFAULT),
    )
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument(
        "--gammas",
        type=str,
        default=",".join(f"{g:g}" for g in GAMMAS_DEFAULT),
        help=f"Louvain/PLM gammas (default: {','.join(f'{g:g}' for g in GAMMAS_DEFAULT)})",
    )
    ap.add_argument(
        "--edge-quantile",
        type=float,
        default=0.9,
        help="keep edges at/above this weight quantile (default 0.9 = top 10%%)",
    )
    ap.add_argument("--min-segment-bp", type=int, default=500_000)
    ap.add_argument("--bridge-bp", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--min-true-fold",
        type=float,
        default=1.0,
        help="per-scale soft fold before merge (default: 1.0)",
    )
    ap.add_argument(
        "--min-enrichment-fold",
        type=float,
        default=1.5,
        help="final gate: keep enrichment_fold > this (default: 1.5; <=0 disables)",
    )
    ap.add_argument(
        "--tip-window-bp",
        type=int,
        default=2_000_000,
        help="breakpoint tip window for primary O/E (default: 2 Mb)",
    )
    ap.add_argument("--merge-min-rec", type=float, default=0.5)
    ap.add_argument("--skip-plot", action="store_true")
    ap.add_argument(
        "--compact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="compact outputs: ALL_INTER_TRA.tsv + community/Hi-C plots only "
        "(default: true; use --no-compact for intermediates)",
    )
    ap.add_argument(
        "--write-scale-dirs",
        action="store_true",
        help="write per-(resolution,gamma) folders (off by default)",
    )
    ap.add_argument(
        "--skip-enrichment",
        action="store_true",
        help="do not re-score enrichment (use fold=1)",
    )
    ap.add_argument(
        "--skip-optimize-merge",
        action="store_true",
        help="skip reciprocal corner-call optimize-merge after enrichment gate",
    )
    ap.add_argument(
        "--optimize-abut-bp",
        type=int,
        default=500_000,
        help="abut slack for reciprocal optimize-merge (default: 500 kb)",
    )
    ap.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="parallel workers (0=auto cpu/2, 1=serial)",
    )
    ap.add_argument(
        "--threads",
        type=int,
        default=0,
        help="NetworKit PLM threads per scale (0=auto)",
    )
    return ap


def main(argv: list[str] | None = None) -> pd.DataFrame:
    args = build_parser().parse_args(argv)
    if (args.chrom_a is None) ^ (args.chrom_b is None):
        raise SystemExit("[ERROR] set both --chrom-a and --chrom-b, or neither")
    if args.chrom_a is None and args.chromosomes is None:
        print(
            "[WARN] no --chrom-a/--chrom-b or --chromosomes; "
            "will run ALL unordered chrom pairs in the mcool",
            flush=True,
        )
    print(
        "[INFO] inter_tra_multiscale\n"
        f"  mcool               = {args.mcool}\n"
        f"  chrom-a/b           = {args.chrom_a}/{args.chrom_b}\n"
        f"  chromosomes         = {args.chromosomes}\n"
        f"  resolutions         = {args.resolutions}\n"
        f"  gammas              = {args.gammas}\n"
        f"  edge_quantile       = {args.edge_quantile}\n"
        f"  min-enrichment-fold = {args.min_enrichment_fold}\n"
        f"  tip-window-bp       = {args.tip_window_bp}\n"
        f"  optimize-merge      = {not args.skip_optimize_merge}\n"
        f"  optimize-abut-bp    = {args.optimize_abut_bp}\n"
        f"  compact             = {args.compact}\n"
        f"  write-scale-dirs    = {args.write_scale_dirs}\n"
        f"  outdir              = {args.outdir}",
        flush=True,
    )
    return run_pipeline(args)


# =============================================================================
# Self-check (ponytail: smallest assert that fails if shared-label logic breaks)
# =============================================================================


def _self_check() -> None:
    # labels: chromA bins 0..5 community 7 in [1,3); chromB bins 0..4 community 7 in [2,4)
    labels_a = np.array([0, 7, 7, 1, 1, 2], dtype=np.int64)
    labels_b = np.array([3, 3, 7, 7, 4], dtype=np.int64)
    labels = np.concatenate([labels_a, labels_b])
    cands = detect_inter_candidates(
        labels,
        n_a=len(labels_a),
        resolution=100_000,
        gamma=1.0,
        chrom_a="chrA",
        chrom_b="chrB",
        size_a=600_000,
        size_b=500_000,
        min_segment_bp=100_000,
        bridge_bp=0,
    )
    assert len(cands) == 1, cands
    assert int(cands.iloc[0]["label"]) == 7
    assert int(cands.iloc[0]["start_a"]) == 100_000
    assert int(cands.iloc[0]["end_a"]) == 300_000
    assert int(cands.iloc[0]["start_b"]) == 200_000
    assert int(cands.iloc[0]["end_b"]) == 400_000
    tips = _interval_tips(0, 5_000_000, 2_000_000)
    assert tips == [(0, 2_000_000), (3_000_000, 5_000_000)]
    assert _pair_merge_score(
        {"chrom_a": "A", "start_a": 0, "end_a": 100, "chrom_b": "B", "start_b": 0, "end_b": 100},
        {"chrom_a": "A", "start_a": 10, "end_a": 90, "chrom_b": "B", "start_b": 20, "end_b": 80},
        0.5,
    )
    # reciprocal corner calls → one mid×mid exchange
    corner = pd.DataFrame(
        [
            {
                "cluster_id": 1,
                "chrom_a": "A",
                "start_a": 0,
                "end_a": 28_000_000,
                "chrom_b": "B",
                "start_b": 797_000_000,
                "end_b": 835_500_000,
                "enrichment_fold": 100.0,
                "enrichment_fold_block": 20.0,
                "enrichment_fold_flank": 10.0,
                "scale_support": 10,
                "resolutions": "50000,100000",
                "gammas": "1",
                "member_resolutions": "",
                "n_members": 10,
                "size_a": 28_000_000,
                "size_b": 38_500_000,
            },
            {
                "cluster_id": 2,
                "chrom_a": "A",
                "start_a": 27_800_000,
                "end_a": 66_200_000,
                "chrom_b": "B",
                "start_b": 835_600_000,
                "end_b": 873_000_000,
                "enrichment_fold": 80.0,
                "enrichment_fold_block": 10.0,
                "enrichment_fold_flank": 10.0,
                "scale_support": 8,
                "resolutions": "50000,100000",
                "gammas": "1",
                "member_resolutions": "",
                "n_members": 8,
                "size_a": 38_400_000,
                "size_b": 37_400_000,
            },
            {
                "cluster_id": 3,
                "chrom_a": "A",
                "start_a": 28_000_000,
                "end_a": 66_500_000,
                "chrom_b": "B",
                "start_b": 667_500_000,
                "end_b": 797_500_000,
                "enrichment_fold": 120.0,
                "enrichment_fold_block": 15.0,
                "enrichment_fold_flank": 7.0,
                "scale_support": 7,
                "resolutions": "50000,100000",
                "gammas": "1",
                "member_resolutions": "",
                "n_members": 7,
                "size_a": 38_500_000,
                "size_b": 130_000_000,
            },
            {
                "cluster_id": 4,
                "chrom_a": "A",
                "start_a": 66_200_000,
                "end_a": 759_000_000,
                "chrom_b": "B",
                "start_b": 797_200_000,
                "end_b": 819_600_000,
                "enrichment_fold": 90.0,
                "enrichment_fold_block": 7.0,
                "enrichment_fold_flank": 10.0,
                "scale_support": 9,
                "resolutions": "50000,100000",
                "gammas": "1",
                "member_resolutions": "",
                "n_members": 9,
                "size_a": 692_800_000,
                "size_b": 22_400_000,
            },
        ]
    )
    opt = optimize_merge_inter_calls(corner, mcool=None, rescore=False)
    assert len(opt) == 1, opt
    assert int(opt.iloc[0]["merged_reciprocal"]) == 1
    assert int(opt.iloc[0]["start_a"]) == 27_800_000
    assert int(opt.iloc[0]["end_a"]) == 66_500_000
    assert int(opt.iloc[0]["start_b"]) == 797_000_000
    assert int(opt.iloc[0]["end_b"]) == 835_500_000
    print("[self-check] ok")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--self-check":
        _self_check()
        raise SystemExit(0)
    main()
    raise SystemExit(0)

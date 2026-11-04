#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASC structural-consistency index (INV + intra-TRA only).

Design (aligned with PHASC detectors)
-------------------------------------
Inversion
  - Resolutions: 50 / 100 / 200 / 500 kb
  - Graph: PHASC inversion construction (distance-normalize → band graph →
    direction-guided edge weights → Louvain)
  - Primary Louvain (resolution=1.0), then secondary Louvain (2.0) on large
    blocks (≥ 3 Mb by default)
  - Exhaustively score every leaf community with flip_improvement
  - Penalty = sum of max(0, Δflip) over all evaluated communities

Intra-chromosomal translocation
  - Same resolutions; Louvain/PLM gammas = TRA defaults
    {0.5, 0.75, 1.0, 1.25, 1.5, 2.0}
  - Graph: TRA-style top edge-quantile contacts
  - Penalty = sum of community genomic-dispersion over chrom × res × gamma

Score (genome-size normalized)
  - Inversion: length-weighted flip penalty density
      D_inv = Σ (clip(max(0,Δflip)/2) · length_bp) / (n_resolutions · genome_bp)
  - Translocation: chromosome-length-weighted mean dispersion, then mean
    over (resolution × gamma)
  - Both D ∈ [0, 1] and are comparable across genome sizes; then
      D' = clip(scale · D, 0, 1)   (penalty-scale amplifies sensitivity)
      QV = 100 × (1 − clip(w_inv·D'_inv + w_tra·D'_tra, 0, 1))
    INV + TRA penalties are added (combined damage), not averaged.
  - Component scores 100×(1−D') are INV-only / TRA-only views for audit.
  - Higher QV ⇒ higher structural quality
  - Default --penalty-scale 25,12; use 1 for unscaled D.

Dependencies: cooler, numpy, scipy, pandas, networkx;
optional networkit (preferred for TRA clustering, matches detector).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import networkx as nx

LOG = logging.getLogger("PHASC-SCI")
EPS = 1e-10

RESOLUTIONS_DEFAULT = (50_000, 100_000, 200_000, 500_000)
TRA_GAMMAS_DEFAULT = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
INV_LOUVAIN_DEFAULT = 1.0
INV_SECONDARY_LOUVAIN_DEFAULT = 2.0
INV_SECONDARY_MIN_BP_DEFAULT = 3_000_000
INV_FLANK_FRACTIONS_DEFAULT = (0.2, 0.5, 1.0)
INV_RADIUS_DEFAULT = 8
INV_DIRECTION_LAMBDA_DEFAULT = 2.0
TRA_EDGE_QUANTILE_DEFAULT = 0.9


def parse_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def finite_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def list_resolutions(mcool_path: str) -> List[int]:
    import cooler
    groups = cooler.fileops.list_coolers(mcool_path)
    out = []
    for g in groups:
        s = str(g).strip("/").split("/")
        if len(s) == 2 and s[0] == "resolutions" and s[1].isdigit():
            out.append(int(s[1]))
    return sorted(set(out))


def build_uri(path: str, resolution: Optional[int]) -> str:
    if "::" in path:
        return path
    if path.endswith(".mcool"):
        if resolution is None:
            raise ValueError("A resolution is required for .mcool")
        return f"{path}::/resolutions/{int(resolution)}"
    return path


def load_cooler(uri: str, balance: bool):
    try:
        import cooler
    except ImportError as exc:
        raise RuntimeError("Missing dependency 'cooler'. Install with: pip install cooler") from exc
    clr = cooler.Cooler(uri)
    use_balance = bool(balance)
    if use_balance:
        try:
            first_chrom = str(clr.chroms()[:]["name"].iloc[0])
            _ = clr.matrix(balance=True, sparse=True).fetch(first_chrom)
        except Exception as exc:
            LOG.warning("ICE/KR balancing unavailable (%s); falling back to raw counts", exc)
            use_balance = False
    return clr, use_balance


def read_matrix(clr, chrom: str, balance: bool) -> np.ndarray:
    try:
        x = clr.matrix(balance=balance, sparse=False).fetch(chrom)
    except Exception:
        if balance:
            LOG.warning("Could not fetch balanced matrix for %s; using raw counts", chrom)
            x = clr.matrix(balance=False, sparse=False).fetch(chrom)
        else:
            raise
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.maximum((x + x.T) * 0.5, 0.0)
    return x


def get_chrom_names(clr) -> List[str]:
    ch = clr.chroms()[:]
    return [str(x) for x in ch["name"].tolist()]


def genome_size_bp(clr, chroms: Optional[Sequence[str]] = None) -> int:
    """Total assembled length in bp (sum of selected chromosome sizes)."""
    if chroms is None:
        chroms = get_chrom_names(clr)
    return int(sum(int(clr.chromsizes[c]) for c in chroms))


# ---------------------------------------------------------------------------
# Inversion primitives (same construction as multiscale_inversion.py)
# ---------------------------------------------------------------------------

def _inv_distance_normalize(matrix: np.ndarray, min_positive: int = 5) -> np.ndarray:
    m = np.asarray(matrix, dtype=float)
    n = m.shape[0]
    out = np.zeros_like(m, dtype=float)
    for d in range(n):
        diag = np.diag(m, k=d)
        vals = diag[np.isfinite(diag) & (diag > 0)]
        if vals.size < min_positive:
            continue
        bg = float(np.median(vals))
        if bg <= 0 or not np.isfinite(bg):
            continue
        idx = np.arange(n - d)
        out[idx, idx + d] = m[idx, idx + d] / bg
        if d > 0:
            out[idx + d, idx] = m[idx + d, idx] / bg
    return out


def _inv_get_profile(matrix: np.ndarray, i: int, radius: int) -> np.ndarray:
    n = matrix.shape[0]
    values = []
    for j in range(i - radius, i + radius + 1):
        if j == i:
            continue
        values.append(matrix[i, j] if 0 <= j < n else 0.0)
    return np.asarray(values, dtype=float)


def _inv_direction_information(matrix: np.ndarray, radius: int) -> Tuple[np.ndarray, np.ndarray]:
    n = matrix.shape[0]
    profiles = [_inv_get_profile(matrix, i, radius) for i in range(n)]
    direction = np.zeros(max(0, n - 1), dtype=float)
    confidence = np.zeros(max(0, n - 1), dtype=float)
    for i in range(n - 1):
        p1, p2 = profiles[i], profiles[i + 1]
        forward = reverse = 0.0
        if len(p1) >= 4 and len(p2) >= 4:
            with np.errstate(divide="ignore", invalid="ignore"):
                forward = finite_float(np.corrcoef(p1[:-1], p2[1:])[0, 1], 0.0)
                reverse = finite_float(np.corrcoef(p1[1:], p2[:-1])[0, 1], 0.0)
        d = forward - reverse
        direction[i] = d
        confidence[i] = abs(d) / (abs(forward) + abs(reverse) + EPS)
    return direction, confidence


def _inv_build_contact_graph(matrix: np.ndarray, radius: int, gap_mask: Optional[np.ndarray] = None) -> nx.Graph:
    n = matrix.shape[0]
    g = nx.Graph()
    g.add_nodes_from(range(n))
    gap = None
    if gap_mask is not None:
        gap = np.asarray(gap_mask, dtype=bool)
        if gap.shape[0] != n:
            gap = None
    for i in range(n):
        if gap is not None and gap[i]:
            continue
        for j in range(i + 1, min(n, i + radius + 1)):
            if gap is not None and gap[j]:
                continue
            w = matrix[i, j]
            if w > 0:
                g.add_edge(i, j, contact_weight=float(w))
    return g


def _inv_fuse_direction(graph: nx.Graph, direction: np.ndarray, confidence: np.ndarray, lam: float) -> nx.Graph:
    for u, v, data in graph.edges(data=True):
        if abs(u - v) == 1:
            idx = min(u, v)
            data["weight"] = data["contact_weight"] * (1.0 + lam * direction[idx] * confidence[idx])
        else:
            data["weight"] = data["contact_weight"]
    return graph


def _inv_run_louvain(graph: nx.Graph, resolution: float, seed: int) -> np.ndarray:
    n = graph.number_of_nodes()
    if n == 0:
        return np.zeros(0, dtype=int)
    deg_sum = sum(d for _, d in graph.degree(weight="weight"))
    if graph.number_of_edges() == 0 or deg_sum == 0:
        return np.arange(n, dtype=int)
    communities = nx.community.louvain_communities(
        graph, weight="weight", resolution=float(resolution), seed=int(seed)
    )
    labels = np.zeros(n, dtype=int)
    for cid, community in enumerate(communities):
        for node in community:
            labels[int(node)] = cid
    return labels


def _inv_gap_mask(hic: np.ndarray, diag_frac: float = 0.35) -> np.ndarray:
    diag = np.asarray(hic.diagonal(), dtype=float)
    positive = diag[diag > 0]
    if positive.size == 0:
        return np.zeros(diag.size, dtype=bool)
    thr = float(diag_frac) * float(np.median(positive))
    return diag < max(thr, 1.0)


def _inv_paint_gap_labels(labels: np.ndarray, gap_mask: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=int).copy()
    gap = np.asarray(gap_mask, dtype=bool)
    if gap.shape[0] != labels.shape[0] or not gap.any():
        return labels
    n = int(labels.size)
    for i in np.where(gap)[0]:
        left = right = None
        for j in range(i - 1, -1, -1):
            if not gap[j]:
                left = int(labels[j])
                break
        for j in range(i + 1, n):
            if not gap[j]:
                right = int(labels[j])
                break
        if left is not None and right is not None and left == right:
            labels[i] = left
        elif left is not None:
            labels[i] = left
        elif right is not None:
            labels[i] = right
    return labels


def _inv_louvain_labels(
    norm: np.ndarray,
    radius: int,
    gap_mask: np.ndarray,
    louvain_resolution: float,
    direction_lambda: float,
    seed: int,
) -> np.ndarray:
    norm = np.asarray(norm, dtype=float)
    gap = np.asarray(gap_mask, dtype=bool)
    n = norm.shape[0]
    if gap.shape[0] != n or not gap.any():
        g = _inv_build_contact_graph(norm, radius, None)
        direction, confidence = _inv_direction_information(norm, radius)
        g = _inv_fuse_direction(g, direction, confidence, direction_lambda)
        return _inv_run_louvain(g, louvain_resolution, seed)

    keep = np.where(~gap)[0]
    if keep.size < 4:
        g = _inv_build_contact_graph(norm, radius, gap)
        direction, confidence = _inv_direction_information(norm, radius)
        g = _inv_fuse_direction(g, direction, confidence, direction_lambda)
        return _inv_paint_gap_labels(_inv_run_louvain(g, louvain_resolution, seed), gap)

    sub = norm[np.ix_(keep, keep)]
    g = _inv_build_contact_graph(sub, radius, None)
    direction, confidence = _inv_direction_information(sub, radius)
    g = _inv_fuse_direction(g, direction, confidence, direction_lambda)
    sub_labels = _inv_run_louvain(g, louvain_resolution, seed)
    labels = np.full(n, -1, dtype=int)
    labels[keep] = sub_labels
    return _inv_paint_gap_labels(labels, gap)


def _inv_labels_to_blocks(labels: np.ndarray, origin_bp: int, resolution: int) -> List[List[int]]:
    labels = np.asarray(labels)
    blocks = []
    i = 0
    n = int(labels.size)
    while i < n:
        s0 = i
        cur = int(labels[i])
        i += 1
        while i < n and int(labels[i]) == cur:
            i += 1
        blocks.append([int(origin_bp + s0 * resolution), int(origin_bp + i * resolution), cur])
    return blocks


def _inv_absorb_short_blocks(blocks: List[List[int]], min_size: int) -> List[List[int]]:
    blocks = [list(b) for b in blocks]
    while len(blocks) > 1:
        short = [i for i, b in enumerate(blocks) if b[1] - b[0] < min_size]
        if not short:
            break
        i = short[0]
        if i == 0:
            blocks[1][0] = blocks[0][0]
            del blocks[0]
        elif i == len(blocks) - 1:
            blocks[-2][1] = blocks[-1][1]
            del blocks[-1]
        else:
            left_len = blocks[i - 1][1] - blocks[i - 1][0]
            right_len = blocks[i + 1][1] - blocks[i + 1][0]
            if left_len <= right_len:
                blocks[i - 1][1] = blocks[i][1]
            else:
                blocks[i + 1][0] = blocks[i][0]
            del blocks[i]
    return blocks


def _inv_secondary_split(
    hic: np.ndarray,
    start: int,
    end: int,
    resolution: int,
    radius: int,
    louvain_resolution: float,
    min_size: int,
    direction_lambda: float,
    seed: int,
    gap_mask: Optional[np.ndarray] = None,
) -> List[List[int]]:
    start, end = int(start), int(end)
    i0, i1 = start // resolution, end // resolution
    if i1 - i0 < 4:
        return [[start, end, 0]]
    sub = np.nan_to_num(np.asarray(hic[i0:i1, i0:i1], dtype=float), nan=0.0)
    if gap_mask is None:
        local_gap = _inv_gap_mask(sub)
    else:
        local_gap = np.asarray(gap_mask[i0:i1], dtype=bool)
        if local_gap.shape[0] != sub.shape[0]:
            local_gap = _inv_gap_mask(sub)
    norm = _inv_distance_normalize(sub)
    labels = _inv_louvain_labels(norm, radius, local_gap, louvain_resolution, direction_lambda, seed)
    blocks = _inv_labels_to_blocks(labels, start, resolution)
    return _inv_absorb_short_blocks(blocks, min_size)


def _robust_spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return np.nan
    x, y = x[ok], y[ok]
    if np.nanstd(x) <= EPS or np.nanstd(y) <= EPS:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho = spearmanr(x, y).statistic
    return finite_float(rho)


def _rect_distance_decay_rho(mat: np.ndarray) -> float:
    m = np.nan_to_num(np.asarray(mat, dtype=float), nan=0.0)
    if m.ndim != 2 or min(m.shape) < 2:
        return np.nan
    nr, nc = m.shape
    rr = np.arange(nr, dtype=float)[:, None]
    cc = np.arange(nc, dtype=float)[None, :]
    dist = (nr - 1.0 - rr) + cc
    return _robust_spearman(m.ravel(), -dist.ravel())


def _flip_interval(local: np.ndarray, s0: int, s1: int) -> np.ndarray:
    out = np.asarray(local, dtype=float).copy()
    out[s0:s1, :] = out[s0:s1, :][::-1, :]
    out[:, s0:s1] = out[:, s0:s1][:, ::-1]
    return out


def _junction_continuity(local: np.ndarray, left_n: int, target_n: int, right_n: int) -> float:
    scores = []
    if left_n >= 2:
        rho = _rect_distance_decay_rho(local[:left_n, left_n:left_n + target_n])
        if np.isfinite(rho):
            scores.append(rho)
    if right_n >= 2:
        rho = _rect_distance_decay_rho(local[left_n:left_n + target_n, left_n + target_n:])
        if np.isfinite(rho):
            scores.append(rho)
    return float(np.mean(scores)) if scores else np.nan


def flip_improvement_for_block(
    hic: np.ndarray,
    chrom_size_bp: int,
    start_bp: int,
    end_bp: int,
    resolution: int,
    flank_fraction: float,
) -> float:
    """Δflip for one community block with genomic flanks = fraction × |S|."""
    res = int(resolution)
    s0 = max(0, int(start_bp))
    s1 = min(int(chrom_size_bp), int(end_bp))
    target_bins = (s1 - s0) // res
    if target_bins < 2:
        return np.nan
    flank_bp = max(2 * res, int(round((s1 - s0) * float(flank_fraction))))
    left0 = max(0, s0 - flank_bp)
    right1 = min(int(chrom_size_bp), s1 + flank_bp)
    left_n = (s0 - left0) // res
    right_n = (right1 - s1) // res
    if left_n < 2 and right_n < 2:
        return np.nan
    # Align to bin grid used by hic (chromosome-local dense matrix).
    r0 = left0 // res
    r1 = int(math.ceil(right1 / res))
    r1 = min(r1, hic.shape[0])
    if r1 - r0 < 4:
        return np.nan
    local = np.nan_to_num(np.asarray(hic[r0:r1, r0:r1], dtype=float), nan=0.0)
    target_local0 = (s0 // res) - r0
    target_local1 = target_local0 + target_bins
    if target_local1 > local.shape[0] or target_local0 < 0:
        return np.nan
    # Recompute left/right sizes from the local window.
    left_n = target_local0
    right_n = local.shape[0] - target_local1
    if left_n < 2 and right_n < 2:
        return np.nan
    before = _junction_continuity(local, left_n, target_bins, right_n)
    flipped = _flip_interval(local, target_local0, target_local1)
    after = _junction_continuity(flipped, left_n, target_bins, right_n)
    if not (np.isfinite(before) and np.isfinite(after)):
        return np.nan
    return float(after - before)


def collect_inv_leaf_communities(
    hic: np.ndarray,
    resolution: int,
    radius: int,
    louvain_resolution: float,
    secondary_louvain_resolution: float,
    secondary_min_bp: int,
    direction_lambda: float,
    seed: int,
    min_community_bp: int,
) -> List[List[int]]:
    gap = _inv_gap_mask(hic)
    norm = _inv_distance_normalize(hic)
    labels = _inv_louvain_labels(
        norm, radius, gap, louvain_resolution, direction_lambda, seed
    )
    primary = _inv_labels_to_blocks(labels, 0, resolution)
    leaves: List[List[int]] = []
    for start, end, cid in primary:
        size = end - start
        if (
            secondary_louvain_resolution > 0
            and secondary_min_bp > 0
            and size >= secondary_min_bp
        ):
            subs = _inv_secondary_split(
                hic, start, end, resolution, radius,
                secondary_louvain_resolution, min_community_bp,
                direction_lambda, seed, gap_mask=gap,
            )
            leaves.extend(subs)
        else:
            leaves.append([start, end, cid])
    # Drop tiny communities that cannot support a flip test.
    return [b for b in leaves if (b[1] - b[0]) >= max(min_community_bp, 2 * resolution)]


def evaluate_inversion(
    mcool: str,
    resolutions: Sequence[int],
    balance: bool,
    radius: int,
    louvain_resolution: float,
    secondary_louvain_resolution: float,
    secondary_min_bp: int,
    direction_lambda: float,
    flank_fractions: Sequence[float],
    min_community_bp: int,
    seed: int,
) -> Tuple[float, float, float, int, pd.DataFrame]:
    """Return (D_inv, raw_sum_positive_flip, flip_per_mb, genome_bp, table).

    Genome-size normalization
    -------------------------
    D_inv = Σ_res Σ_community [bounded · length_bp]
            / (n_resolutions · genome_bp)

    so a fixed absolute error burden does not look better on a smaller genome.
    Unscored sequence contributes 0 (no flip evidence).
    """
    records = []
    genome_bp = 0
    n_res_used = 0
    weighted_penalty_sum = 0.0
    raw_sum = 0.0

    for res in resolutions:
        uri = build_uri(mcool, res)
        clr, use_balance = load_cooler(uri, balance)
        chroms = get_chrom_names(clr)
        g_bp = genome_size_bp(clr, chroms)
        if genome_bp <= 0:
            genome_bp = g_bp
        n_res_used += 1
        LOG.info(
            "INV: resolution=%d, chroms=%d, genome=%.3f Mb, balance=%s",
            res, len(chroms), g_bp / 1e6, use_balance,
        )
        for chrom in chroms:
            hic = read_matrix(clr, chrom, use_balance)
            chrom_size = int(clr.chromsizes[chrom])
            leaves = collect_inv_leaf_communities(
                hic, int(clr.binsize), radius, louvain_resolution,
                secondary_louvain_resolution, secondary_min_bp,
                direction_lambda, seed, min_community_bp,
            )
            for start, end, cid in leaves:
                best_delta = np.nan
                best_ff = np.nan
                for ff in flank_fractions:
                    delta = flip_improvement_for_block(
                        hic, chrom_size, start, end, int(clr.binsize), ff
                    )
                    if not np.isfinite(delta):
                        continue
                    if not np.isfinite(best_delta) or delta > best_delta:
                        best_delta = float(delta)
                        best_ff = float(ff)
                if not np.isfinite(best_delta):
                    continue
                length_bp = int(end - start)
                pos = float(max(0.0, best_delta))
                bounded = float(np.clip(pos / 2.0, 0.0, 1.0))
                weighted_penalty_sum += bounded * float(length_bp)
                raw_sum += pos
                records.append({
                    "metric": "inversion_flip_improvement",
                    "chrom": chrom,
                    "resolution": int(clr.binsize),
                    "community_id": int(cid),
                    "start": int(start),
                    "end": int(end),
                    "length_bp": length_bp,
                    "chrom_length_bp": chrom_size,
                    "best_flank_fraction": best_ff,
                    "delta_flip": best_delta,
                    "positive_delta": pos,
                    "bounded_penalty": bounded,
                    "length_weighted_penalty": bounded * float(length_bp),
                })
            del hic
        LOG.info(
            "INV: finished resolution=%d; communities_scored=%d",
            res, sum(1 for r in records if r["resolution"] == int(clr.binsize)),
        )

    df = pd.DataFrame(records)
    denom = float(max(n_res_used, 1) * max(genome_bp, 1))
    d_inv = float(np.clip(weighted_penalty_sum / denom, 0.0, 1.0))
    genome_mb = max(genome_bp / 1e6, EPS)
    # Per-resolution-averaged flip density (audit metric).
    flip_per_mb = float(raw_sum / (max(n_res_used, 1) * genome_mb))
    return d_inv, raw_sum, flip_per_mb, int(genome_bp), df


# ---------------------------------------------------------------------------
# Intra-TRA primitives (TRA-style graph + dispersion)
# ---------------------------------------------------------------------------

def _tra_graph_networkx(
    matrix: np.ndarray,
    edge_quantile: float,
    max_edges: int,
) -> nx.Graph:
    n = matrix.shape[0]
    g = nx.Graph()
    g.add_nodes_from(range(n))
    rows, cols = np.triu_indices(n, k=1)
    weights = np.asarray(matrix, dtype=float)[rows, cols]
    mask = np.isfinite(weights) & (weights > 0)
    rows, cols, weights = rows[mask], cols[mask], weights[mask]
    if weights.size == 0:
        return g
    threshold = float(np.quantile(weights, edge_quantile))
    keep = weights >= threshold
    rows, cols, weights = rows[keep], cols[keep], weights[keep]
    if max_edges > 0 and weights.size > max_edges:
        take = np.argpartition(weights, -max_edges)[-max_edges:]
        rows, cols, weights = rows[take], cols[take], weights[take]
    for i, j, w in zip(rows, cols, weights):
        g.add_edge(int(i), int(j), weight=float(w))
    return g


def _tra_louvain_labels(
    matrix: np.ndarray,
    gamma: float,
    edge_quantile: float,
    max_edges: int,
    seed: int,
) -> np.ndarray:
    """Prefer NetworKit PLM (detector default); fall back to NetworkX Louvain."""
    n = matrix.shape[0]
    try:
        import networkit as nk
        rows, cols = np.triu_indices(n, k=1)
        weights = np.asarray(matrix, dtype=float)[rows, cols]
        mask = np.isfinite(weights) & (weights > 0)
        rows, cols, weights = rows[mask], cols[mask], weights[mask]
        if weights.size == 0:
            return np.arange(n, dtype=np.int64)
        threshold = float(np.quantile(weights, edge_quantile))
        keep = weights >= threshold
        rows, cols, weights = rows[keep], cols[keep], weights[keep]
        if max_edges > 0 and weights.size > max_edges:
            take = np.argpartition(weights, -max_edges)[-max_edges:]
            rows, cols, weights = rows[take], cols[take], weights[take]
        graph = nk.graph.Graph(int(n), weighted=True, directed=False)
        order = np.lexsort((cols, rows))
        for k in order:
            graph.addEdge(int(rows[k]), int(cols[k]), float(weights[k]))
        if graph.numberOfEdges() == 0:
            return np.arange(n, dtype=np.int64)
        nk.setSeed(int(seed), True)
        algo = nk.community.PLM(graph, refine=True, gamma=float(gamma), par="balanced")
        algo.run()
        return np.asarray(algo.getPartition().getVector(), dtype=np.int64)
    except Exception as exc:
        LOG.debug("NetworKit PLM unavailable (%s); using NetworkX Louvain", exc)
        g = _tra_graph_networkx(matrix, edge_quantile, max_edges)
        if g.number_of_edges() == 0:
            return np.arange(n, dtype=np.int64)
        communities = nx.community.louvain_communities(
            g, weight="weight", resolution=float(gamma), seed=int(seed)
        )
        labels = np.full(n, -1, dtype=np.int64)
        for cid, community in enumerate(communities):
            for node in community:
                labels[int(node)] = cid
        # Fill isolates.
        next_id = int(labels.max()) + 1 if labels.max() >= 0 else 0
        for i in range(n):
            if labels[i] < 0:
                labels[i] = next_id
                next_id += 1
        return labels


def intra_community_dispersion(
    labels: np.ndarray,
    valid: np.ndarray,
    min_nodes: int = 3,
) -> Tuple[float, int, int]:
    """Length-weighted fragmentation × genomic-distance dispersion ∈ [0, 1]."""
    idx = np.flatnonzero(valid & (labels >= 0))
    if idx.size == 0:
        return np.nan, 0, 0
    labs = labels[idx]
    vals, weights = [], []
    kept = 0
    for cid in np.unique(labs):
        members = idx[labs == cid]
        if members.size < min_nodes:
            continue
        cuts = np.where(np.diff(members) > 1)[0] + 1
        chunks = [x for x in np.split(members, cuts) if x.size > 0]
        if len(chunks) < 2:
            continue
        sizes = np.asarray([x.size for x in chunks], dtype=float)
        p = sizes / sizes.sum()
        frag = float(1.0 - np.sum(p * p))
        centers = np.asarray([(x[0] + x[-1]) * 0.5 for x in chunks], dtype=float)
        n = max(1, int(valid.size) - 1)
        dist = 0.0
        for a in range(len(chunks)):
            for b in range(a + 1, len(chunks)):
                dist += 2.0 * p[a] * p[b] * abs(centers[a] - centers[b]) / n
        dispersion = float(np.clip(frag * dist, 0.0, 1.0))
        vals.append(dispersion)
        weights.append(float(members.size))
        kept += 1
    if not vals:
        return 0.0, 0, int(idx.size)
    return float(np.average(vals, weights=weights)), kept, int(idx.size)


def evaluate_translocation(
    mcool: str,
    resolutions: Sequence[int],
    gammas: Sequence[float],
    balance: bool,
    edge_quantile: float,
    max_edges: int,
    min_community_bins: int,
    seed: int,
) -> Tuple[float, float, float, int, pd.DataFrame]:
    """Return (D_tra, raw_sum_dispersion, disp_per_mb, genome_bp, table).

    Genome-size normalization
    -------------------------
    For each (resolution, gamma):
      d_rg = Σ_chrom (dispersion_c · chrom_length_c) / genome_bp
    D_tra = mean over (resolution × gamma) of d_rg

    Chromosome length weighting prevents short chroms from dominating and
    keeps the index comparable across assemblies of different sizes.
    """
    records = []
    scale_vals = []
    raw_sum = 0.0
    genome_bp = 0

    for res in resolutions:
        uri = build_uri(mcool, res)
        clr, use_balance = load_cooler(uri, balance)
        chroms = get_chrom_names(clr)
        g_bp = genome_size_bp(clr, chroms)
        if genome_bp <= 0:
            genome_bp = g_bp
        chrom_lengths = {c: int(clr.chromsizes[c]) for c in chroms}
        LOG.info(
            "TRA: resolution=%d, chroms=%d, genome=%.3f Mb, gammas=%s, balance=%s",
            res, len(chroms), g_bp / 1e6, list(gammas), use_balance,
        )
        # Cache matrices per chrom within this resolution.
        matrices = {}
        valids = {}
        for chrom in chroms:
            matrices[chrom] = read_matrix(clr, chrom, use_balance)
            valids[chrom] = np.asarray(matrices[chrom].sum(axis=1)).ravel() > 0

        for gamma in gammas:
            weighted = 0.0
            length_sum = 0.0
            for chrom in chroms:
                scale_seed = int(seed) + int(clr.binsize) + int(round(float(gamma) * 1000))
                labels = _tra_louvain_labels(
                    matrices[chrom], gamma, edge_quantile, max_edges, scale_seed
                )
                disp, n_comm, n_valid = intra_community_dispersion(
                    labels, valids[chrom], min_nodes=min_community_bins
                )
                if not np.isfinite(disp):
                    continue
                disp = float(np.clip(disp, 0.0, 1.0))
                clen = float(chrom_lengths[chrom])
                weighted += disp * clen
                length_sum += clen
                raw_sum += disp
                records.append({
                    "metric": "intra_community_dispersion",
                    "chrom": chrom,
                    "resolution": int(clr.binsize),
                    "gamma": float(gamma),
                    "value": disp,
                    "chrom_length_bp": int(clen),
                    "length_weighted_value": disp * clen,
                    "n_communities_used": int(n_comm),
                    "n_valid_bins": int(n_valid),
                })
            if length_sum > 0:
                scale_vals.append(weighted / length_sum)
        del matrices, valids

    df = pd.DataFrame(records)
    d_tra = float(np.clip(np.mean(scale_vals), 0.0, 1.0)) if scale_vals else 0.0
    genome_mb = max(genome_bp / 1e6, EPS)
    n_scales = max(len(resolutions) * len(gammas), 1)
    # Audit: raw sum divided by genome Mb and number of scales.
    disp_per_mb = float(raw_sum / (n_scales * genome_mb))
    return d_tra, raw_sum, disp_per_mb, int(genome_bp), df


def _parse_penalty_scales(text: str) -> Tuple[float, float]:
    """Parse one shared scale or inv,tra pair. Values must be > 0."""
    vals = parse_floats(text)
    if len(vals) == 1:
        s = float(vals[0])
        if s <= 0:
            raise ValueError("--penalty-scale must be > 0")
        return s, s
    if len(vals) == 2:
        s_inv, s_tra = float(vals[0]), float(vals[1])
        if s_inv <= 0 or s_tra <= 0:
            raise ValueError("--penalty-scale values must be > 0")
        return s_inv, s_tra
    raise ValueError("--penalty-scale expects 1 value or inv,tra pair")


def summarize_score(
    d_inv: float,
    d_tra: float,
    weights: Sequence[float],
    penalty_scales: Sequence[float] = (1.0, 1.0),
) -> Tuple[float, Dict[str, float]]:
    """Combine genome-normalized D into a 0–100 QV.

    INV and TRA contribute additively to the penalty (combined structural
    damage), not as an average of standalone component scores:

        QV = 100 × (1 − clip(w_inv·D'_inv + w_tra·D'_tra, 0, 1))

    Component scores 100×(1−D') remain available as INV-only / TRA-only views.
    """
    vals = [d_inv, d_tra]
    names = ["inversion", "translocation"]
    w = np.asarray(weights, dtype=float)
    if w.size != 2 or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("--weights must contain two nonnegative values summing to > 0 (inv,tra)")
    scales = np.asarray(penalty_scales, dtype=float)
    if scales.size != 2 or np.any(scales <= 0):
        raise ValueError("penalty_scales must be two positive values (inv,tra)")
    comp = {}
    penalty = 0.0
    any_term = False
    for name, value, weight, scale in zip(names, vals, w, scales):
        if np.isfinite(value):
            raw = float(np.clip(value, 0.0, 1.0))
            scaled = float(np.clip(raw * float(scale), 0.0, 1.0))
            comp[name + "_score"] = 100.0 * (1.0 - scaled)
            comp[name + "_D_raw"] = raw
            comp[name + "_D_scaled"] = scaled
            comp[name + "_penalty_contrib"] = float(weight) * scaled
            penalty += float(weight) * scaled
            any_term = True
        else:
            comp[name + "_score"] = np.nan
            comp[name + "_D_raw"] = np.nan
            comp[name + "_D_scaled"] = np.nan
            comp[name + "_penalty_contrib"] = np.nan
    if not any_term:
        return np.nan, comp
    penalty = float(np.clip(penalty, 0.0, 1.0))
    comp["combined_penalty"] = penalty
    return 100.0 * (1.0 - penalty), comp


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "PHASC structural-consistency index from inversion flip-improvement "
            "and intra-TRA community dispersion (no inter-chromosomal TRA)."
        )
    )
    parser.add_argument("--mcool", required=True, help=".mcool or .cool contact map")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument(
        "--resolutions",
        default=",".join(str(r) for r in RESOLUTIONS_DEFAULT),
        help="Comma-separated resolutions (default: 50000,100000,200000,500000)",
    )
    parser.add_argument(
        "--gammas",
        default=",".join(f"{g:g}" for g in TRA_GAMMAS_DEFAULT),
        help="TRA Louvain/PLM gammas (default: 0.5,0.75,1,1.25,1.5,2)",
    )
    parser.add_argument("--no-balance", action="store_true",
                        help="Use raw counts instead of Cooler balancing weights")
    parser.add_argument("--inv-radius", type=int, default=INV_RADIUS_DEFAULT)
    parser.add_argument("--inv-louvain-resolution", type=float, default=INV_LOUVAIN_DEFAULT)
    parser.add_argument("--inv-secondary-louvain-resolution", type=float,
                        default=INV_SECONDARY_LOUVAIN_DEFAULT)
    parser.add_argument("--inv-secondary-min-bp", type=int,
                        default=INV_SECONDARY_MIN_BP_DEFAULT)
    parser.add_argument("--inv-direction-lambda", type=float,
                        default=INV_DIRECTION_LAMBDA_DEFAULT)
    parser.add_argument(
        "--inv-flank-fractions",
        default=",".join(str(x) for x in INV_FLANK_FRACTIONS_DEFAULT),
        help="Flank sizes as fractions of community length for flip tests",
    )
    parser.add_argument("--inv-min-community-bp", type=int, default=200_000,
                        help="Minimum community length (bp) to score with flip_improvement")
    parser.add_argument("--edge-quantile", type=float, default=TRA_EDGE_QUANTILE_DEFAULT,
                        help="TRA graph: keep edges at/above this quantile (default 0.9)")
    parser.add_argument("--max-edges", type=int, default=200000,
                        help="Cap edges per TRA graph (0 = no cap)")
    parser.add_argument("--min-community-bins", type=int, default=3,
                        help="Minimum TRA community size for dispersion")
    parser.add_argument("--weights", default="1,1",
                        help="Weights for inversion,translocation (default equal)")
    parser.add_argument(
        "--penalty-scale",
        default="25,12",
        help=(
            "Amplify genome-normalized D before scoring: one value (shared) or "
            "inv,tra pair. score uses clip(scale·D). Default 25,12 (INV needs "
            "a larger boost because genome averaging makes D_inv tiny); use 1 "
            "for legacy unscaled scoring."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-inversion", action="store_true")
    parser.add_argument("--skip-translocation", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    if not 0.0 < args.edge_quantile < 1.0:
        parser.error("--edge-quantile must be in (0,1)")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        available = list_resolutions(args.mcool) if args.mcool.endswith(".mcool") else []
        requested = parse_ints(args.resolutions)
        selected = [r for r in requested if not available or r in available]
        if not selected:
            raise ValueError(
                f"None of the requested resolutions {requested} are present in {args.mcool}. "
                f"Available: {available or 'n/a (cool)'}"
            )
        selected = sorted(set(selected))
        gammas = parse_floats(args.gammas)
        flank_fractions = parse_floats(args.inv_flank_fractions)
        balance = not args.no_balance

        d_inv = np.nan
        inv_sum = 0.0
        inv_per_mb = np.nan
        genome_bp = 0
        inv_df = pd.DataFrame()
        if not args.skip_inversion:
            d_inv, inv_sum, inv_per_mb, genome_bp, inv_df = evaluate_inversion(
                args.mcool, selected, balance,
                args.inv_radius, args.inv_louvain_resolution,
                args.inv_secondary_louvain_resolution, args.inv_secondary_min_bp,
                args.inv_direction_lambda, flank_fractions,
                args.inv_min_community_bp, args.seed,
            )

        d_tra = np.nan
        tra_sum = 0.0
        tra_per_mb = np.nan
        tra_df = pd.DataFrame()
        if not args.skip_translocation:
            d_tra, tra_sum, tra_per_mb, genome_bp_tra, tra_df = evaluate_translocation(
                args.mcool, selected, gammas, balance,
                args.edge_quantile, args.max_edges,
                args.min_community_bins, args.seed,
            )
            if genome_bp <= 0:
                genome_bp = genome_bp_tra

        weights = parse_floats(args.weights)
        # If one arm skipped, reweight the remaining arm.
        if args.skip_inversion and not args.skip_translocation:
            weights = [0.0, 1.0]
        elif args.skip_translocation and not args.skip_inversion:
            weights = [1.0, 0.0]
        penalty_scales = list(_parse_penalty_scales(args.penalty_scale))
        total_score, component_scores = summarize_score(
            d_inv, d_tra, weights, penalty_scales=penalty_scales
        )

        summary = {
            "method": "PHASC-SCI-inv-intraTRA",
            "input_mcool": str(Path(args.mcool).resolve()),
            "genome_bp": int(genome_bp),
            "genome_mb": float(genome_bp) / 1e6 if genome_bp else None,
            "resolutions_used": selected,
            "tra_gammas": gammas,
            "inv_louvain_resolution": args.inv_louvain_resolution,
            "inv_secondary_louvain_resolution": args.inv_secondary_louvain_resolution,
            "inv_secondary_min_bp": args.inv_secondary_min_bp,
            "raw_inversion_positive_flip_sum": inv_sum,
            "raw_translocation_dispersion_sum": tra_sum,
            "inversion_positive_flip_per_mb": inv_per_mb,
            "translocation_dispersion_sum_per_mb": tra_per_mb,
            "genome_normalized_D_inversion": d_inv,
            "genome_normalized_D_translocation": d_tra,
            "penalty_scale_inversion_translocation": penalty_scales,
            "D_inversion_scaled": component_scores.get("inversion_D_scaled"),
            "D_translocation_scaled": component_scores.get("translocation_D_scaled"),
            "component_scores_0_100": {
                k: v for k, v in component_scores.items() if k.endswith("_score")
            },
            "weights_inversion_translocation": weights,
            "total_score_0_100": total_score,
            "interpretation": (
                "Higher score = higher structural quality. "
                "D_inv = Σ(bounded·length)/(n_res·genome_bp); "
                "D_tra = mean over scales of length-weighted chrom dispersion; "
                "both genome-size normalized. "
                "score = 100×(1 − weighted D)."
            ),
        }

        if not inv_df.empty:
            inv_df.to_csv(outdir / "inversion_community_flip.tsv", sep="\t", index=False)
        if not tra_df.empty:
            tra_df.to_csv(outdir / "translocation_community_dispersion.tsv", sep="\t", index=False)

        def _json_safe(obj):
            if isinstance(obj, dict):
                return {str(k): _json_safe(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_json_safe(v) for v in obj]
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating, float)):
                return float(obj) if np.isfinite(obj) else None
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            return obj

        with open(outdir / "summary.json", "w", encoding="utf-8") as fh:
            json.dump(_json_safe(summary), fh, ensure_ascii=False, indent=2, allow_nan=False)

        pd.DataFrame([{
            "HiC_SCI": total_score,
            "inversion_score": component_scores.get("inversion_score", np.nan),
            "translocation_score": component_scores.get("translocation_score", np.nan),
            "D_inversion": d_inv,
            "D_translocation": d_tra,
            "D_inversion_scaled": component_scores.get("inversion_D_scaled", np.nan),
            "D_translocation_scaled": component_scores.get("translocation_D_scaled", np.nan),
            "penalty_scale_inv": penalty_scales[0],
            "penalty_scale_tra": penalty_scales[1],
            "genome_bp": genome_bp,
            "genome_mb": float(genome_bp) / 1e6 if genome_bp else np.nan,
            "sum_positive_flip": inv_sum,
            "sum_dispersion": tra_sum,
            "flip_per_mb": inv_per_mb,
            "dispersion_sum_per_mb": tra_per_mb,
        }]).to_csv(outdir / "summary.tsv", sep="\t", index=False)

        LOG.info(
            "Finished. HiC-SCI = %s (INV=%s, TRA=%s)",
            f"{total_score:.3f}" if np.isfinite(total_score) else "NA",
            f"{component_scores.get('inversion_score', float('nan')):.3f}"
            if np.isfinite(component_scores.get("inversion_score", np.nan)) else "NA",
            f"{component_scores.get('translocation_score', float('nan')):.3f}"
            if np.isfinite(component_scores.get("translocation_score", np.nan)) else "NA",
        )
        LOG.info("Results: %s", outdir.resolve())
        return 0
    except Exception as exc:
        LOG.exception("Assessment failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

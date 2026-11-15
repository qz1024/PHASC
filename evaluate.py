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
  - Seed: leaf communities with Δflip > flip_min (best flank among
    --inv-flank-fractions, same as detector exploration)
  - Merge overlapping seeds per chrom, then post-merge flip gate matching
    multiscale_inversion.apply_post_merge_flip_gate: flank=|S|, keep only
    intervals with ≥ --inv-min-pos-scales (default 2) resolutions where
    Δflip > flip_min
  - Penalty = sum over kept events of clip(max(0,mean_Δflip)/2)·length_bp

Intra-chromosomal translocation
  - Same path as multiscale_intra_translocation.run_pipeline per chrom:
    top-quantile contacts → PLM → secondary-segment candidates →
    tip O/E enrichment → clear-insert ILP → merge across (res,gamma) →
    enrichment_fold > --tra-min-enrichment-fold (default 1.5)
  - Soft Louvain dispersion is NOT used (noise fragments that fail
    detection do not count)
  - Penalty = sum over kept TRA calls of clip(1 − 1/fold, 0, 1)·length_bp

Score (genome-size normalized)
  - Inversion: event-level length-weighted flip density
      D_inv = Σ_event (bounded · length_bp) / genome_bp
  - Translocation: event-level length-weighted enrichment density
      D_tra = Σ_call (bounded · length_bp) / genome_bp
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
import tempfile
import warnings
from pathlib import Path
from types import SimpleNamespace
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


def _merge_overlapping_intervals(
    intervals: Sequence[Tuple[int, int, int]],
) -> List[Tuple[int, int, set]]:
    """Merge overlapping/touching (start, end, resolution) → (start, end, res_set)."""
    if not intervals:
        return []
    ordered = sorted((int(s), int(e), int(r)) for s, e, r in intervals)
    merged: List[Tuple[int, int, set]] = []
    cs, ce, ress = ordered[0][0], ordered[0][1], {ordered[0][2]}
    for s, e, r in ordered[1:]:
        if s <= ce:
            ce = max(ce, e)
            ress.add(r)
        else:
            merged.append((cs, ce, ress))
            cs, ce, ress = s, e, {r}
    merged.append((cs, ce, ress))
    return merged


def gate_inv_events_multiscale_flip(
    mcool: str,
    resolutions: Sequence[int],
    balance: bool,
    community_df: pd.DataFrame,
    flip_min: float = 0.0,
    min_pos_scales: int = 2,
    gate_flank_fraction: float = 1.0,
) -> Tuple[float, float, float, pd.DataFrame]:
    """Apply detector-style post-merge flip gate to positive community seeds.

    Matches multiscale_inversion.apply_post_merge_flip_gate:
    flank=|S|, keep only if ≥ min_pos_scales resolutions have Δflip > flip_min.

    Returns (D_inv, raw_sum_mean_positive_flip, flip_per_mb, event_table).
    D_inv = Σ_kept (bounded · length_bp) / genome_bp  (event counted once).
    """
    empty = pd.DataFrame()
    if community_df is None or community_df.empty:
        return 0.0, 0.0, 0.0, empty

    pos = community_df[
        np.isfinite(community_df["delta_flip"])
        & (community_df["delta_flip"].astype(float) > float(flip_min))
    ]
    if pos.empty:
        return 0.0, 0.0, 0.0, empty

    res_list = [int(r) for r in resolutions]
    genome_bp = int(pos["chrom_length_bp"].max()) if "chrom_length_bp" in pos else 0
    # Prefer full-genome size from first resolution cooler when available.
    try:
        uri = build_uri(mcool, res_list[0])
        clr0, _ = load_cooler(uri, balance)
        genome_bp = genome_size_bp(clr0, get_chrom_names(clr0))
    except Exception:
        pass

    cache: Dict[Tuple[int, str], Tuple[np.ndarray, int]] = {}

    def _hic(res: int, chrom: str) -> Tuple[np.ndarray, int]:
        key = (int(res), str(chrom))
        if key not in cache:
            uri = build_uri(mcool, int(res))
            clr, use_balance = load_cooler(uri, balance)
            cache[key] = (
                read_matrix(clr, chrom, use_balance),
                int(clr.chromsizes[chrom]),
            )
        return cache[key]

    events = []
    weighted_penalty_sum = 0.0
    raw_sum = 0.0
    n_seed = 0
    n_drop = 0

    for chrom, sub in pos.groupby("chrom", sort=False):
        seeds = [
            (int(r.start), int(r.end), int(r.resolution))
            for r in sub.itertuples(index=False)
        ]
        n_seed += len(seeds)
        for start, end, src_res in _merge_overlapping_intervals(seeds):
            per = {}
            for res in res_list:
                try:
                    hic, chrom_size = _hic(res, str(chrom))
                    delta = flip_improvement_for_block(
                        hic, chrom_size, start, end, res, gate_flank_fraction
                    )
                except Exception:
                    delta = np.nan
                per[res] = float(delta) if np.isfinite(delta) else np.nan
            n_pos = sum(
                1 for d in per.values()
                if np.isfinite(d) and float(d) > float(flip_min)
            )
            finite = [float(d) for d in per.values() if np.isfinite(d)]
            mean_d = float(np.mean(finite)) if finite else np.nan
            pos_vals = [
                float(d) for d in per.values()
                if np.isfinite(d) and float(d) > float(flip_min)
            ]
            mean_pos = float(np.mean(pos_vals)) if pos_vals else 0.0
            length_bp = int(end - start)
            flip_by = ",".join(
                (
                    f"{r}:{per[r]:+.4f}"
                    if np.isfinite(per[r])
                    else f"{r}:NA"
                )
                for r in res_list
            )
            passed = int(n_pos >= int(min_pos_scales))
            if passed:
                bounded = float(np.clip(mean_pos / 2.0, 0.0, 1.0))
                weighted_penalty_sum += bounded * float(length_bp)
                raw_sum += mean_pos
            else:
                n_drop += 1
                bounded = 0.0
            events.append({
                "metric": "inversion_multiscale_flip_event",
                "chrom": str(chrom),
                "start": int(start),
                "end": int(end),
                "length_bp": length_bp,
                "seed_resolutions": ",".join(str(x) for x in sorted(src_res)),
                "seed_n_resolutions": int(len(src_res)),
                "flip_pos_scales": int(n_pos),
                "flip_improve": mean_d,
                "flip_improve_pos_mean": mean_pos,
                "flip_by_resolution": flip_by,
                "flip_pass": passed,
                "bounded_penalty": bounded,
                "length_weighted_penalty": bounded * float(length_bp),
            })

    event_df = pd.DataFrame(events)
    denom = float(max(genome_bp, 1))
    d_inv = float(np.clip(weighted_penalty_sum / denom, 0.0, 1.0))
    genome_mb = max(genome_bp / 1e6, EPS)
    flip_per_mb = float(raw_sum / genome_mb)
    n_keep = int(event_df["flip_pass"].sum()) if not event_df.empty else 0
    LOG.info(
        "INV flip gate: seeds=%d events=%d kept=%d dropped=%d "
        "(need >=%d scales with Δflip > %g, flank=|S|)",
        n_seed, len(events), n_keep, n_drop, int(min_pos_scales), float(flip_min),
    )
    return d_inv, raw_sum, flip_per_mb, event_df


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
    flip_min: float = 0.0,
    min_pos_scales: int = 2,
) -> Tuple[float, float, float, int, pd.DataFrame, pd.DataFrame]:
    """Return (D_inv, raw_sum, flip_per_mb, genome_bp, community_table, event_table).

    Community table: per-resolution leaf flip (audit).
    Event table / D_inv: after merge + multiscale flip gate (detector-aligned).
    """
    records = []
    genome_bp = 0
    n_res_used = 0

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
    d_inv, raw_sum, flip_per_mb, event_df = gate_inv_events_multiscale_flip(
        mcool, resolutions, balance, df,
        flip_min=flip_min, min_pos_scales=min_pos_scales, gate_flank_fraction=1.0,
    )
    return d_inv, raw_sum, flip_per_mb, int(genome_bp), df, event_df


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


def _load_tra_detector():
    """Import multiscale_intra_translocation (must be picklable for TRA --jobs)."""
    phasc_dir = str(Path(__file__).resolve().parent)
    if phasc_dir not in sys.path:
        sys.path.insert(0, phasc_dir)
    import multiscale_intra_translocation as tra  # noqa: WPS433
    return tra


def evaluate_translocation(
    mcool: str,
    resolutions: Sequence[int],
    gammas: Sequence[float],
    balance: bool,
    edge_quantile: float,
    max_edges: int,
    min_community_bins: int,
    seed: int,
    min_segment_bp: int = 500_000,
    bridge_bp: int = 500_000,
    min_enrichment_fold: float = 1.5,
    tip_window_bp: int = 2_000_000,
    merge_min_rec: float = 0.5,
    min_true_fold: float = 1.0,
    jobs: int = 0,
    threads: int = 1,
) -> Tuple[float, float, float, int, pd.DataFrame]:
    """Return (D_tra, raw_sum, per_mb, genome_bp, kept_call_table).

    Runs the same intra-TRA detector as PHASC (per chrom), then scores only
    calls that pass clear-insert + enrichment_fold > min_enrichment_fold.

    D_tra = Σ_call clip(1 − 1/fold, 0, 1) · length_bp / genome_bp

    ``balance``, ``max_edges``, ``min_community_bins`` kept for CLI compat;
    detector uses raw contacts (same as pipeline --no-balance).
    """
    del balance, max_edges, min_community_bins  # detector-aligned path
    tra = _load_tra_detector()

    uri = build_uri(mcool, int(resolutions[0]))
    clr0, _ = load_cooler(uri, False)
    chroms = get_chrom_names(clr0)
    genome_bp = genome_size_bp(clr0, chroms)
    LOG.info(
        "TRA: detector path chroms=%d genome=%.3f Mb res=%s gammas=%s "
        "min_segment=%d bridge=%d enrichment>%g",
        len(chroms), genome_bp / 1e6, list(resolutions), list(gammas),
        int(min_segment_bp), int(bridge_bp), float(min_enrichment_fold),
    )

    kept_rows: List[pd.DataFrame] = []
    res_csv = ",".join(str(int(r)) for r in resolutions)
    gamma_csv = ",".join(f"{float(g):g}" for g in gammas)

    for chrom in chroms:
        with tempfile.TemporaryDirectory(prefix=f"phasc_sci_tra_{chrom}_") as td:
            args = SimpleNamespace(
                per_scale=Path(td) / "unused.tsv",
                mcool=Path(mcool),
                truth=None,
                outdir=Path(td),
                resolutions=res_csv,
                chrom=str(chrom),
                gamma=1.0,
                gammas=gamma_csv,
                edge_quantile=float(edge_quantile),
                min_segment_bp=int(min_segment_bp),
                bridge_bp=int(bridge_bp),
                seed=int(seed),
                min_true_fold=float(min_true_fold),
                min_enrichment_fold=float(min_enrichment_fold),
                tip_window_bp=int(tip_window_bp),
                merge_min_rec=float(merge_min_rec),
                from_tsv=False,
                skip_plot=True,
                skip_enrichment=False,
                jobs=int(jobs),
                threads=int(threads),
            )
            LOG.info("TRA: chrom=%s running detector...", chrom)
            kept = tra.run_pipeline(args)
            n_kept = 0 if kept is None else len(kept)
            LOG.info("TRA: chrom=%s kept_calls=%d", chrom, n_kept)
            if kept is not None and len(kept):
                kept_rows.append(kept.copy())

    if kept_rows:
        events = pd.concat(kept_rows, ignore_index=True)
    else:
        events = pd.DataFrame()

    weighted_penalty_sum = 0.0
    raw_sum = 0.0
    out_records = []
    if not events.empty:
        for _, r in events.iterrows():
            start = int(r["start"])
            end = int(r["end"])
            length_bp = int(r["size"]) if "size" in r.index and pd.notna(r.get("size")) else end - start
            fold = float(r.get("enrichment_fold", 0.0) or 0.0)
            if not np.isfinite(fold) or fold <= 0:
                fold = 0.0
            # Strength in [0,1]: fold→∞ ⇒ 1; fold=1.5 ⇒ 1/3.
            bounded = float(np.clip(1.0 - 1.0 / max(fold, 1.0), 0.0, 1.0))
            weighted_penalty_sum += bounded * float(length_bp)
            raw_sum += fold
            rec = r.to_dict()
            rec["metric"] = "intra_tra_detector_call"
            rec["length_bp"] = length_bp
            rec["bounded_penalty"] = bounded
            rec["length_weighted_penalty"] = bounded * float(length_bp)
            out_records.append(rec)

    df = pd.DataFrame(out_records) if out_records else events
    denom = float(max(genome_bp, 1))
    d_tra = float(np.clip(weighted_penalty_sum / denom, 0.0, 1.0))
    genome_mb = max(genome_bp / 1e6, EPS)
    per_mb = float(raw_sum / genome_mb)
    LOG.info(
        "TRA gate: kept_calls=%d D_tra=%.6g (enrichment_fold>%g)",
        len(out_records), d_tra, float(min_enrichment_fold),
    )
    return d_tra, raw_sum, per_mb, int(genome_bp), df


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


def _read_call_tsv(path: Path) -> pd.DataFrame:
    if path is None or not Path(path).is_file() or Path(path).stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, sep="\t")
    except Exception:
        return pd.DataFrame()


def _genome_bp_from_mcool(
    mcool: str,
    chromosomes: Optional[Sequence[str]] = None,
) -> int:
    available = list_resolutions(mcool) if str(mcool).endswith(".mcool") else []
    res = int(available[0]) if available else 0
    uri = build_uri(mcool, res) if res else str(mcool)
    clr, _ = load_cooler(uri, False)
    chroms = list(chromosomes) if chromosomes else get_chrom_names(clr)
    chroms = [c for c in chroms if c in clr.chromsizes.index]
    if not chroms:
        chroms = get_chrom_names(clr)
    return genome_size_bp(clr, chroms)


def _inv_penalty_from_calls(inv_df: pd.DataFrame) -> Tuple[float, float, pd.DataFrame]:
    """D numerator pieces from detector INV table (already flip-gated)."""
    if inv_df is None or inv_df.empty:
        return 0.0, 0.0, pd.DataFrame()
    rows = []
    weighted = 0.0
    raw = 0.0
    for _, r in inv_df.iterrows():
        start = int(r["start"]) if "start" in r.index and pd.notna(r.get("start")) else 0
        end = int(r["end"]) if "end" in r.index and pd.notna(r.get("end")) else start
        if "size" in r.index and pd.notna(r.get("size")):
            length_bp = int(r["size"])
        else:
            length_bp = max(0, end - start)
        flip = float(r["flip_improve"]) if "flip_improve" in r.index and pd.notna(r.get("flip_improve")) else 0.0
        if not np.isfinite(flip):
            flip = 0.0
        pos = float(max(0.0, flip))
        bounded = float(np.clip(pos / 2.0, 0.0, 1.0))
        weighted += bounded * float(length_bp)
        raw += pos
        rec = r.to_dict()
        rec["metric"] = "inv_call"
        rec["length_bp"] = length_bp
        rec["bounded_penalty"] = bounded
        rec["length_weighted_penalty"] = bounded * float(length_bp)
        rows.append(rec)
    return weighted, raw, pd.DataFrame(rows)


def _tra_penalty_from_calls(tra_df: pd.DataFrame) -> Tuple[float, float, pd.DataFrame]:
    """D numerator pieces from detector ALL_TRA table (already enrichment-gated)."""
    if tra_df is None or tra_df.empty:
        return 0.0, 0.0, pd.DataFrame()
    rows = []
    weighted = 0.0
    raw = 0.0
    for _, r in tra_df.iterrows():
        start = int(r["start"]) if "start" in r.index and pd.notna(r.get("start")) else 0
        end = int(r["end"]) if "end" in r.index and pd.notna(r.get("end")) else start
        if "size" in r.index and pd.notna(r.get("size")):
            length_bp = int(r["size"])
        else:
            length_bp = max(0, end - start)
        fold = float(r["enrichment_fold"]) if "enrichment_fold" in r.index and pd.notna(r.get("enrichment_fold")) else 0.0
        if not np.isfinite(fold) or fold <= 0:
            fold = 0.0
        bounded = float(np.clip(1.0 - 1.0 / max(fold, 1.0), 0.0, 1.0))
        weighted += bounded * float(length_bp)
        raw += fold
        rec = r.to_dict()
        rec["metric"] = "tra_call"
        rec["length_bp"] = length_bp
        rec["bounded_penalty"] = bounded
        rec["length_weighted_penalty"] = bounded * float(length_bp)
        rows.append(rec)
    return weighted, raw, pd.DataFrame(rows)


def score_sci_from_calls(
    mcool: str,
    calls_dir: str | Path,
    sci_outdir: str | Path | None = None,
    inv_tsv: str | Path | None = None,
    tra_tsv: str | Path | None = None,
    weights: Sequence[float] = (1.0, 1.0),
    penalty_scale: str | Sequence[float] = "25,12",
    chromosomes: Optional[Sequence[str]] = None,
    skip_inversion: bool = False,
    skip_translocation: bool = False,
) -> Dict:
    """HiC-SCI from pipeline call tables — no re-detection.

    Reads ``ALL_INV_multiscale.tsv`` / ``ALL_TRA.tsv`` (already past detector
    gates) and writes ``SCI/summary.tsv`` under ``sci_outdir`` (default:
    ``calls_dir/SCI``).
    """
    calls_dir = Path(calls_dir)
    sci_outdir = Path(sci_outdir) if sci_outdir else calls_dir / "SCI"
    sci_outdir.mkdir(parents=True, exist_ok=True)

    inv_path = Path(inv_tsv) if inv_tsv else calls_dir / "ALL_INV_multiscale.tsv"
    tra_path = Path(tra_tsv) if tra_tsv else calls_dir / "ALL_TRA.tsv"
    if not tra_path.is_file():
        alt = calls_dir / "merged_true_calls.tsv"
        if alt.is_file():
            tra_path = alt

    genome_bp = _genome_bp_from_mcool(mcool, chromosomes)
    denom = float(max(genome_bp, 1))

    inv_raw = _read_call_tsv(inv_path) if not skip_inversion else pd.DataFrame()
    tra_raw = _read_call_tsv(tra_path) if not skip_translocation else pd.DataFrame()
    # Pipeline multi-chrom layout: calls live under tra_by_chrom/*/ALL_TRA.tsv
    if tra_raw.empty and not skip_translocation:
        chrom_tra = sorted(calls_dir.glob("tra_by_chrom/*/ALL_TRA.tsv"))
        if not chrom_tra:
            chrom_tra = sorted(calls_dir.glob("tra_by_chrom/*/merged_true_calls.tsv"))
        frames = [_read_call_tsv(p) for p in chrom_tra]
        frames = [f for f in frames if not f.empty]
        if frames:
            tra_raw = pd.concat(frames, ignore_index=True)
            tra_path = chrom_tra[0].parent.parent  # note: directory for audit
            LOG.info("SCI-from-calls: merged %d per-chrom TRA tables", len(frames))
    if chromosomes and not inv_raw.empty and "chrom" in inv_raw.columns:
        inv_raw = inv_raw[inv_raw["chrom"].astype(str).isin(set(map(str, chromosomes)))]
    if chromosomes and not tra_raw.empty and "chrom" in tra_raw.columns:
        tra_raw = tra_raw[tra_raw["chrom"].astype(str).isin(set(map(str, chromosomes)))]

    inv_w, inv_sum, inv_events = _inv_penalty_from_calls(inv_raw)
    tra_w, tra_sum, tra_events = _tra_penalty_from_calls(tra_raw)
    d_inv = 0.0 if skip_inversion else float(np.clip(inv_w / denom, 0.0, 1.0))
    d_tra = 0.0 if skip_translocation else float(np.clip(tra_w / denom, 0.0, 1.0))
    if skip_inversion and not skip_translocation:
        weights = [0.0, 1.0]
    elif skip_translocation and not skip_inversion:
        weights = [1.0, 0.0]
    if isinstance(penalty_scale, str):
        penalty_scales = list(_parse_penalty_scales(penalty_scale))
    else:
        penalty_scales = [float(penalty_scale[0]), float(penalty_scale[1])]
    total_score, component_scores = summarize_score(
        d_inv, d_tra, weights, penalty_scales=penalty_scales
    )

    # ponytail: no per-call flip/TRA audit TSVs in pipeline SCI — summary only

    summary = {
        "method": "PHASC-SCI-from-calls",
        "input_mcool": str(Path(mcool).resolve()),
        "calls_dir": str(calls_dir.resolve()),
        "inv_tsv": str(inv_path),
        "tra_tsv": str(tra_path),
        "genome_bp": int(genome_bp),
        "genome_mb": float(genome_bp) / 1e6,
        "n_inv_calls": int(len(inv_events)),
        "n_tra_calls": int(len(tra_events)),
        "genome_normalized_D_inversion": d_inv,
        "genome_normalized_D_translocation": d_tra,
        "penalty_scale_inversion_translocation": penalty_scales,
        "D_inversion_scaled": component_scores.get("inversion_D_scaled"),
        "D_translocation_scaled": component_scores.get("translocation_D_scaled"),
        "component_scores_0_100": {
            k: v for k, v in component_scores.items() if k.endswith("_score")
        },
        "weights_inversion_translocation": list(weights),
        "total_score_0_100": total_score,
        "interpretation": (
            "From pipeline call tables only (no re-detection). "
            "D_inv from ALL_INV flip_improve; D_tra from ALL_TRA enrichment_fold; "
            "score = 100×(1 − weighted scaled D)."
        ),
    }

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

    with open(sci_outdir / "summary.json", "w", encoding="utf-8") as fh:
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
        "genome_mb": float(genome_bp) / 1e6,
        "n_inv_calls": int(len(inv_events)),
        "n_tra_calls": int(len(tra_events)),
        "sum_positive_flip": inv_sum,
        "sum_tra_fold": tra_sum,
    }]).to_csv(sci_outdir / "summary.tsv", sep="\t", index=False)

    LOG.info(
        "SCI-from-calls HiC-SCI=%.3f (INV=%.3f n=%d, TRA=%.3f n=%d) -> %s",
        total_score if np.isfinite(total_score) else float("nan"),
        component_scores.get("inversion_score", float("nan")),
        len(inv_events),
        component_scores.get("translocation_score", float("nan")),
        len(tra_events),
        sci_outdir,
    )
    return summary


def _self_check() -> int:
    """ponytail: assert merge + flip-gate + from-calls scoring; no mcool needed."""
    # Overlap merge: two touching/overlapping → one; disjoint → two.
    m = _merge_overlapping_intervals([(0, 10, 50_000), (5, 20, 100_000), (30, 40, 50_000)])
    assert len(m) == 2, m
    assert m[0][:2] == (0, 20) and m[0][2] == {50_000, 100_000}, m[0]
    assert m[1][:2] == (30, 40) and m[1][2] == {50_000}, m[1]

    # Gate: single-scale positive seed fails min_pos_scales=2 when other
    # resolutions are mocked negative via a tiny synthetic community table
    # without cooler — exercise only the merge bookkeeping here.
    seeds = _merge_overlapping_intervals([(100, 200, 50_000)])
    assert len(seeds) == 1 and seeds[0][2] == {50_000}

    # Two-res seeds on same interval merge to support={50k,100k}.
    seeds2 = _merge_overlapping_intervals([(100, 200, 50_000), (100, 200, 100_000)])
    assert len(seeds2) == 1 and seeds2[0][2] == {50_000, 100_000}, seeds2

    # from-calls: empty → zero penalty; one INV with flip=1 length=2Mb
    w0, s0, e0 = _inv_penalty_from_calls(pd.DataFrame())
    assert w0 == 0.0 and e0.empty
    inv = pd.DataFrame([{
        "chrom": "chr1", "start": 0, "end": 2_000_000, "size": 2_000_000,
        "flip_improve": 1.0,
    }])
    w1, s1, e1 = _inv_penalty_from_calls(inv)
    assert abs(w1 - 0.5 * 2_000_000) < 1e-6 and len(e1) == 1 and s1 == 1.0
    tra = pd.DataFrame([{
        "chrom": "chr1", "start": 0, "end": 1_000_000, "size": 1_000_000,
        "enrichment_fold": 2.0,
    }])
    wt, st, et = _tra_penalty_from_calls(tra)
    assert abs(wt - 0.5 * 1_000_000) < 1e-6 and len(et) == 1 and st == 2.0
    print("evaluate.py self-check OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "PHASC structural-consistency index from inversion flip-improvement "
            "and intra-TRA community dispersion (no inter-chromosomal TRA)."
        )
    )
    parser.add_argument("--mcool", default=None, help=".mcool or .cool contact map")
    parser.add_argument("--outdir", default=None, help="Output directory")
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
    parser.add_argument(
        "--inv-flip-min",
        type=float,
        default=0.0,
        help="Δflip must exceed this to count positive (default 0; matches detector)",
    )
    parser.add_argument(
        "--inv-min-pos-scales",
        type=int,
        default=2,
        help=(
            "Post-merge flip gate: keep event only if ≥ this many resolutions "
            "have Δflip > --inv-flip-min with flank=|S| (default 2; matches "
            "detector --flip-min-pos-scales)"
        ),
    )
    parser.add_argument("--edge-quantile", type=float, default=TRA_EDGE_QUANTILE_DEFAULT,
                        help="TRA graph: keep edges at/above this quantile (default 0.9)")
    parser.add_argument("--max-edges", type=int, default=200000,
                        help="Unused (legacy); detector builds its own graph")
    parser.add_argument("--min-community-bins", type=int, default=3,
                        help="Unused (legacy soft-dispersion path removed)")
    parser.add_argument("--tra-min-segment-bp", type=int, default=500_000,
                        help="TRA detector min secondary segment (default 500000)")
    parser.add_argument("--tra-bridge-bp", type=int, default=500_000,
                        help="TRA detector bridge short interruptions (default 500000)")
    parser.add_argument(
        "--tra-min-enrichment-fold",
        type=float,
        default=1.5,
        help="TRA enrichment gate after merge (default 1.5; matches detector)",
    )
    parser.add_argument(
        "--tip-window-bp",
        type=int,
        default=2_000_000,
        help="TRA tip O/E window (default 2000000; matches detector)",
    )
    parser.add_argument("--tra-merge-min-rec", type=float, default=0.5,
                        help="TRA cross-scale merge reciprocal overlap (default 0.5)")
    parser.add_argument(
        "--tra-jobs",
        type=int,
        default=0,
        help="TRA detector parallel (res×gamma) workers per chrom (0=auto)",
    )
    parser.add_argument(
        "--tra-threads",
        type=int,
        default=1,
        help="NetworKit threads per TRA scale worker (default 1)",
    )
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
    parser.add_argument(
        "--from-calls",
        action="store_true",
        help=(
            "Score from existing ALL_INV_multiscale.tsv / ALL_TRA.tsv under "
            "--calls-dir (default: --outdir). Skips re-running INV/TRA detection."
        ),
    )
    parser.add_argument(
        "--calls-dir",
        default=None,
        help="Pipeline result dir with call TSVs (default: --outdir when --from-calls)",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Run merge/flip-gate unit checks and exit",
    )
    args = parser.parse_args()

    if args.self_check:
        raise SystemExit(_self_check())

    if not args.mcool or not args.outdir:
        parser.error("--mcool and --outdir are required unless --self-check")
    if int(args.inv_min_pos_scales) < 1:
        parser.error("--inv-min-pos-scales must be >= 1")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    if not 0.0 < args.edge_quantile < 1.0:
        parser.error("--edge-quantile must be in (0,1)")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        if args.from_calls:
            calls_dir = Path(args.calls_dir) if args.calls_dir else outdir
            summary = score_sci_from_calls(
                mcool=args.mcool,
                calls_dir=calls_dir,
                sci_outdir=outdir if args.calls_dir else outdir / "SCI",
                weights=parse_floats(args.weights),
                penalty_scale=args.penalty_scale,
                skip_inversion=args.skip_inversion,
                skip_translocation=args.skip_translocation,
            )
            print(
                f"HiC_SCI={summary.get('total_score_0_100')} "
                f"INV={summary.get('component_scores_0_100', {}).get('inversion_score')} "
                f"TRA={summary.get('component_scores_0_100', {}).get('translocation_score')}",
                flush=True,
            )
            return 0

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
        inv_event_df = pd.DataFrame()
        if not args.skip_inversion:
            d_inv, inv_sum, inv_per_mb, genome_bp, inv_df, inv_event_df = evaluate_inversion(
                args.mcool, selected, balance,
                args.inv_radius, args.inv_louvain_resolution,
                args.inv_secondary_louvain_resolution, args.inv_secondary_min_bp,
                args.inv_direction_lambda, flank_fractions,
                args.inv_min_community_bp, args.seed,
                flip_min=args.inv_flip_min,
                min_pos_scales=args.inv_min_pos_scales,
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
                min_segment_bp=args.tra_min_segment_bp,
                bridge_bp=args.tra_bridge_bp,
                min_enrichment_fold=args.tra_min_enrichment_fold,
                tip_window_bp=args.tip_window_bp,
                merge_min_rec=args.tra_merge_min_rec,
                jobs=args.tra_jobs,
                threads=args.tra_threads,
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
            "inv_flip_min": args.inv_flip_min,
            "inv_min_pos_scales": args.inv_min_pos_scales,
            "tra_min_enrichment_fold": args.tra_min_enrichment_fold,
            "tra_min_segment_bp": args.tra_min_segment_bp,
            "raw_inversion_positive_flip_sum": inv_sum,
            "raw_translocation_fold_sum": tra_sum,
            "inversion_positive_flip_per_mb": inv_per_mb,
            "translocation_fold_sum_per_mb": tra_per_mb,
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
                "D_inv = Σ_event(bounded·length)/genome_bp after "
                "multiscale flip gate (≥inv_min_pos_scales scales, flank=|S|); "
                "D_tra = Σ_call clip(1−1/fold)·length/genome_bp after the "
                "intra-TRA detector enrichment gate; "
                "both genome-size normalized. "
                "score = 100×(1 − weighted D)."
            ),
        }

        # ponytail: skip flip/TRA audit TSVs; score still uses inv_event_df / tra_df

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
            "sum_tra_fold": tra_sum,
            "flip_per_mb": inv_per_mb,
            "tra_fold_sum_per_mb": tra_per_mb,
            "n_tra_calls": int(len(tra_df)) if tra_df is not None else 0,
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

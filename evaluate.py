#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASC-SCI: an independent Hi-C structural-consistency assessment prototype.

INPUT
-----
A .mcool/.cool contact map built from Hi-C pairs for which BOTH mates passed
MAPQ >= 30. The script cannot verify read-level MAPQ after the pairs have been
aggregated into a contact matrix; it trusts the provenance of the input file.

DESIGN
------
1) Intra-chromosomal translocation-like consistency:
   PHASC-style weighted contact graph + Louvain communities; quantify how
   fragmented and spatially dispersed each community is along its chromosome.
2) Inter-chromosomal consistency:
   genome-wide graph combining local cis edges and normalized trans edges;
   quantify how widely each community is distributed across chromosomes.
3) Inversion-orientation consistency:
   exhaustive fixed-window scan (not candidate calls) using the PHASC
   junction-continuity / flip-improvement idea. Every eligible interval of
   each configured size is evaluated; no inversion list or coordinates are
   emitted and no PHASC detection output is read.

IMPORTANT
---------
The emitted 0-100 value is a PROTOTYPE INDEX, not a validated probability or
an established standard. Natural chromosome territories, compartments, TADs,
repeats and cell-population effects can influence these features. Calibrate it
against controlled simulations and independent assemblies before using it
as a paper-ready quality score.

Dependencies: cooler, numpy, scipy, pandas, networkx
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import spearmanr

LOG = logging.getLogger("PHASC-SCI")
EPS = 1e-10


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


def robust_spearman(x: np.ndarray, y: np.ndarray) -> float:
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


def distance_normalize_phasc(matrix: np.ndarray, min_positive: int = 5) -> np.ndarray:
    """PHASC-style diagonal normalization: divide each diagonal by its
    positive-contact median. Zeros remain zeros; they are not discarded."""
    m = np.asarray(matrix, dtype=float)
    n = m.shape[0]
    out = np.zeros_like(m, dtype=np.float32)
    for d in range(n):
        diag = np.diag(m, k=d)
        vals = diag[np.isfinite(diag) & (diag > 0)]
        if vals.size < min_positive:
            continue
        bg = float(np.median(vals))
        if bg <= 0 or not np.isfinite(bg):
            continue
        idx = np.arange(n - d)
        out[idx, idx + d] = (m[idx, idx + d] / bg).astype(np.float32)
        if d > 0:
            out[idx + d, idx] = out[idx, idx + d]
    return out


def get_profile(matrix: np.ndarray, i: int, radius: int) -> np.ndarray:
    n = matrix.shape[0]
    values = []
    for j in range(i - radius, i + radius + 1):
        if j == i:
            continue
        values.append(matrix[i, j] if 0 <= j < n else 0.0)
    return np.asarray(values, dtype=float)


def calculate_direction_information(matrix: np.ndarray, radius: int) -> Tuple[np.ndarray, np.ndarray]:
    """Same forward-vs-reverse profile-correlation concept used by PHASC."""
    n = matrix.shape[0]
    profiles = [get_profile(matrix, i, radius) for i in range(n)]
    direction = np.zeros(max(0, n - 1), dtype=float)
    confidence = np.zeros(max(0, n - 1), dtype=float)
    for i in range(n - 1):
        p1, p2 = profiles[i], profiles[i + 1]
        forward = 0.0
        reverse = 0.0
        if len(p1) >= 4 and len(p2) >= 4:
            forward = finite_float(np.corrcoef(p1[:-1], p2[1:])[0, 1], 0.0)
            reverse = finite_float(np.corrcoef(p1[1:], p2[:-1])[0, 1], 0.0)
        d = forward - reverse
        direction[i] = d
        confidence[i] = abs(d) / (abs(forward) + abs(reverse) + EPS)
    return direction, confidence


def valid_bins_for_chrom(clr, chrom: str, balance: bool, matrix: np.ndarray) -> np.ndarray:
    """Mask bins with invalid balancing weights when available; otherwise
    exclude fully empty rows. This is a data-availability mask, not an error call."""
    try:
        b = clr.bins().fetch(chrom)
        if balance and "weight" in b.columns:
            w = np.asarray(b["weight"], dtype=float)
            valid = np.isfinite(w) & (w != 0)
            if valid.size == matrix.shape[0] and valid.any():
                return valid
    except Exception:
        pass
    row_sum = np.asarray(np.nan_to_num(matrix, nan=0.0).sum(axis=1)).ravel()
    positive = row_sum[row_sum > 0]
    if positive.size == 0:
        return np.zeros(matrix.shape[0], dtype=bool)
    # Only completely empty rows are masked here. A more aggressive threshold
    # can discard true low-contact biology; report evaluable fraction instead.
    return row_sum > 0


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
            # Force an actual fetch: some Cooler versions defer validation of
            # the balancing column until matrix data are materialized.
            _ = clr.matrix(balance=True, sparse=True).fetch(first_chrom)
        except Exception as exc:
            LOG.warning("ICE/KR balancing unavailable (%s); falling back to raw counts", exc)
            use_balance = False
    return clr, use_balance


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
    # Symmetrize small numeric differences and enforce nonnegative contacts.
    x = np.maximum((x + x.T) * 0.5, 0.0)
    return x


def graph_from_dense(
    matrix: np.ndarray,
    valid: np.ndarray,
    edge_quantile: float,
    max_edges: int,
    direction_guided: bool = False,
    radius: int = 8,
    direction_lambda: float = 0.5,
    seed: int = 1817,
) -> nx.Graph:
    """Build a sparse undirected graph from the upper triangle.

    For intra-TRA, all positive upper-triangle edges are considered (PHASC
    translocation-style). For inversion-style graph summaries, direction can
    modify adjacent-bin edge weights as in PHASC's direction-guided Louvain.
    """
    n = matrix.shape[0]
    g = nx.Graph()
    nodes = np.flatnonzero(valid)
    g.add_nodes_from(int(i) for i in nodes)
    if n < 2 or nodes.size < 2:
        return g

    m = np.asarray(matrix, dtype=np.float32)
    if direction_guided:
        direction, confidence = calculate_direction_information(m, radius)
    else:
        direction = confidence = None

    rows, cols = np.triu_indices(n, k=1)
    weights = m[rows, cols]
    mask = (weights > 0) & valid[rows] & valid[cols] & np.isfinite(weights)
    rows, cols, weights = rows[mask], cols[mask], weights[mask]
    if weights.size == 0:
        return g

    if direction_guided:
        adj = (cols - rows) == 1
        idx = rows[adj]
        factor = 1.0 + direction_lambda * direction[idx] * confidence[idx]
        weights[adj] *= np.maximum(factor, 0.05)

    threshold = float(np.quantile(weights, edge_quantile))
    keep = weights >= threshold
    rows, cols, weights = rows[keep], cols[keep], weights[keep]

    if max_edges > 0 and weights.size > max_edges:
        take = np.argpartition(weights, -max_edges)[-max_edges:]
        rows, cols, weights = rows[take], cols[take], weights[take]

    order = np.lexsort((cols, rows))
    for k in order:
        g.add_edge(int(rows[k]), int(cols[k]), weight=float(weights[k]))
    return g


def louvain_labels(graph: nx.Graph, n_nodes: int, gamma: float, seed: int) -> np.ndarray:
    labels = np.full(n_nodes, -1, dtype=np.int64)
    if graph.number_of_nodes() == 0:
        return labels
    if graph.number_of_edges() == 0 or sum(d for _, d in graph.degree(weight="weight")) <= 0:
        for i, node in enumerate(graph.nodes()):
            labels[int(node)] = i
        return labels
    communities = nx.community.louvain_communities(
        graph, weight="weight", resolution=float(gamma), seed=int(seed)
    )
    for cid, community in enumerate(communities):
        for node in community:
            labels[int(node)] = cid
    return labels


def run_segments(indices: np.ndarray, labels: np.ndarray) -> List[Tuple[int, int, int]]:
    """Return (start_index, end_index_exclusive, community_id) runs."""
    out = []
    if indices.size == 0:
        return out
    start = 0
    for p in range(1, indices.size + 1):
        if p == indices.size or indices[p] != indices[p - 1] + 1 or labels[p] != labels[p - 1]:
            out.append((int(indices[start]), int(indices[p - 1]) + 1, int(labels[start])))
            start = p
    return out


def intra_community_dispersion(labels: np.ndarray, valid: np.ndarray, min_nodes: int = 3) -> Tuple[float, int, int]:
    """Length-weighted fragmentation × genomic-distance dispersion.

    This is a continuous descriptor; it does not label or output translocation
    events. Community members are mapped back to their original bin positions.
    """
    idx = np.flatnonzero(valid & (labels >= 0))
    if idx.size == 0:
        return np.nan, 0, 0
    labs = labels[idx]
    communities = np.unique(labs)
    vals, weights = [], []
    kept = 0
    for cid in communities:
        members = idx[labs == cid]
        if members.size < min_nodes:
            continue
        # Consecutive bins with the same community label form one genomic run.
        cuts = np.where(np.diff(members) > 1)[0] + 1
        chunks = np.split(members, cuts)
        chunks = [x for x in chunks if x.size > 0]
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


def inter_community_dispersion(labels: np.ndarray, chrom_ids: np.ndarray, valid: np.ndarray, min_nodes: int = 3) -> Tuple[float, int, int]:
    """Community chromosome mixing multiplied by within-chromosome block coherence.

    A community scattered as many isolated bins across chromosomes is downweighted;
    a community represented by coherent contiguous blocks on multiple chromosomes
    receives a larger value. This is still a contact-pattern descriptor, not a TRA call.
    """
    idx = np.flatnonzero(valid & (labels >= 0))
    if idx.size == 0:
        return np.nan, 0, 0
    labs = labels[idx]
    vals, weights = [], []
    kept = 0
    n_chrom = max(2, len(np.unique(chrom_ids[valid])))
    denom = 1.0 - 1.0 / n_chrom
    for cid in np.unique(labs):
        members = idx[labs == cid]
        if members.size < min_nodes:
            continue
        unique_chroms = np.unique(chrom_ids[members])
        counts = []
        coherences = []
        for chrom_id in unique_chroms:
            pos = np.sort(members[chrom_ids[members] == chrom_id])
            counts.append(float(pos.size))
            cuts = np.where(np.diff(pos) > 1)[0] + 1
            runs = np.split(pos, cuts)
            longest = max((len(run) for run in runs), default=0)
            coherences.append(float(longest / max(1, pos.size)))
        counts = np.asarray(counts, dtype=float)
        p = counts / counts.sum()
        d = float(1.0 - np.sum(p * p))
        mix = d / denom if denom > 0 else 0.0
        block_coherence = float(np.average(coherences, weights=counts))
        vals.append(mix * block_coherence)
        weights.append(float(members.size))
        kept += 1
    if not vals:
        return 0.0, 0, int(idx.size)
    return float(np.average(vals, weights=weights)), kept, int(idx.size)


def rect_distance_decay_rho(mat: np.ndarray) -> float:
    """PHASC-style Spearman(contact, -distance_to_junction) for a flank×target
    rectangle. Higher values indicate brighter contacts close to the junction."""
    m = np.nan_to_num(np.asarray(mat, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if m.ndim != 2 or min(m.shape) < 2:
        return np.nan
    nr, nc = m.shape
    rr = np.arange(nr, dtype=float)[:, None]
    cc = np.arange(nc, dtype=float)[None, :]
    dist = (nr - 1.0 - rr) + cc
    return robust_spearman(m.ravel(), -dist.ravel())


def flip_interval(local: np.ndarray, s0: int, s1: int) -> np.ndarray:
    out = np.asarray(local, dtype=float).copy()
    out[s0:s1, :] = out[s0:s1, :][::-1, :]
    out[:, s0:s1] = out[:, s0:s1][:, ::-1]
    return out


def junction_continuity(local: np.ndarray, left_n: int, target_n: int, right_n: int) -> float:
    scores = []
    if left_n >= 2:
        left_box = local[:left_n, left_n:left_n + target_n]
        rho = rect_distance_decay_rho(left_box)
        if np.isfinite(rho):
            scores.append(rho)
    if right_n >= 2:
        right_box = local[left_n:left_n + target_n, left_n + target_n:]
        # PHASC defines the right rectangle as target × right flank.
        rho = rect_distance_decay_rho(right_box)
        if np.isfinite(rho):
            scores.append(rho)
    return float(np.mean(scores)) if scores else np.nan


def phasc_calculate_direction(mat: np.ndarray, axis: str, smooth: int = 5,
                               low_signal_frac: float = 0.30) -> float:
    """PHASC-style GCOR trend: filter low-signal axes, aggregate rows/columns,
    smooth the 1-D profile, then calculate Spearman(profile position, signal)."""
    a = np.nan_to_num(np.asarray(mat, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if a.ndim != 2 or a.size == 0:
        return np.nan
    if axis == "column":
        means = np.mean(a, axis=0)
        positive = means[means > 0]
        if positive.size:
            keep = means >= (float(low_signal_frac) * float(np.median(positive)))
            a = a[:, keep]
        signal = np.mean(a, axis=0) if a.shape[1] else np.asarray([])
    elif axis == "row":
        means = np.mean(a, axis=1)
        positive = means[means > 0]
        if positive.size:
            keep = means >= (float(low_signal_frac) * float(np.median(positive)))
            a = a[keep, :]
        signal = np.mean(a, axis=1) if a.shape[0] else np.asarray([])
    else:
        raise ValueError("axis must be 'row' or 'column'")
    if signal.size < 3:
        return np.nan
    if smooth > 1 and signal.size > smooth:
        from scipy.ndimage import uniform_filter1d
        signal = uniform_filter1d(signal.astype(float), size=int(smooth))
    return robust_spearman(np.arange(signal.size, dtype=float), signal)


def rectangle_diag_antidiag_ratio(mat: np.ndarray, band: int = 2) -> float:
    """Scaled diagonal / anti-diagonal mean ratio for a rectangular flank×target
    contact block, following the PHASC corner-enrichment concept."""
    a = np.nan_to_num(np.asarray(mat, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if a.ndim != 2 or min(a.shape) < 2:
        return np.nan
    nr, nc = a.shape
    u = np.linspace(0.0, 1.0, nr)[:, None]
    v = np.linspace(0.0, 1.0, nc)[None, :]
    tol = max(float(band) / max(1, nr - 1), float(band) / max(1, nc - 1))
    diag = np.abs(u - v) <= tol
    anti = np.abs(u + v - 1.0) <= tol
    if not diag.any() or not anti.any():
        return np.nan
    d = float(np.mean(a[diag]))
    ad = float(np.mean(a[anti]))
    if ad <= 0:
        return np.inf if d > 0 else np.nan
    return d / ad


def evaluate_flip_windows(
    clr,
    chroms: Sequence[str],
    resolution: int,
    balance: bool,
    sizes_mb: Sequence[float],
    step_fraction: float,
    flank_fractions: Sequence[float],
    min_occupancy: float,
) -> Tuple[float, pd.DataFrame]:
    """Exhaustive, fixed-grid interval scan. It returns summaries only; it does
    not write interval coordinates or call inversions.

    Uses PHASC-inspired GCOR, diagonal/anti-diagonal corner enrichment, and
    local matrix reversal / junction-continuity improvement.
    """
    matrix_selector = clr.matrix(balance=balance, sparse=False)
    records = []
    for chrom in chroms:
        try:
            n_bins = int(clr.extent(chrom)[1] - clr.extent(chrom)[0])
        except Exception:
            # robust fallback from chromosome length / bin size
            chrom_df = clr.chroms()[:]
            length = int(chrom_df.loc[chrom_df["name"].astype(str) == str(chrom), "length"].iloc[0])
            n_bins = int(math.ceil(length / resolution))
        for size_mb in sizes_mb:
            target_bins = max(4, int(round(size_mb * 1_000_000 / resolution)))
            if target_bins >= n_bins:
                continue
            step = max(1, int(round(target_bins * step_fraction)))
            starts = list(range(0, n_bins - target_bins + 1, step))
            if starts and starts[-1] != n_bins - target_bins:
                starts.append(n_bins - target_bins)
            if not starts:
                continue
            scale_deltas = []
            scale_gcor = []
            scale_corner = []
            scale_flip = []
            scale_evidence = []
            for start in starts:
                end = start + target_bins
                for ff in flank_fractions:
                    flank_bins = max(2, int(round(target_bins * ff)))
                    left0 = max(0, start - flank_bins)
                    right1 = min(n_bins, end + flank_bins)
                    left_n = start - left0
                    right_n = right1 - end
                    if left_n < 2 and right_n < 2:
                        continue
                    # Cooler fetch uses genomic coordinates, not bin indices.
                    bp0 = left0 * resolution
                    bp1 = min(right1 * resolution, int(clr.chromsizes[chrom]))
                    try:
                        local = np.asarray(
                            matrix_selector.fetch((chrom, bp0, bp1)), dtype=float
                        )
                    except Exception:
                        continue
                    local = np.nan_to_num(local, nan=0.0, posinf=0.0, neginf=0.0)
                    # Require enough nonzero cells in both flank×target boxes.
                    target_local0 = left_n
                    target_local1 = left_n + target_bins
                    left_box = local[:left_n, target_local0:target_local1] if left_n >= 2 else None
                    right_box = local[target_local0:target_local1, target_local1:] if right_n >= 2 else None
                    boxes = [b for b in (left_box, right_box) if b is not None]
                    if len(boxes) != 2:
                        continue
                    occupancy = float(np.mean([np.mean(b > 0) for b in boxes]))
                    if occupancy < min_occupancy:
                        continue

                    # GCOR trends: left flank × target uses column trend;
                    # target × right flank uses row trend (PHASC convention).
                    left_rho = phasc_calculate_direction(left_box, "column", smooth=5)
                    right_rho = phasc_calculate_direction(right_box, "row", smooth=5)
                    if not (np.isfinite(left_rho) and np.isfinite(right_rho)):
                        continue

                    # Corner signal: inversion-like patterns enrich the scaled
                    # diagonal relative to anti-diagonal in both flank boxes.
                    corner_left = rectangle_diag_antidiag_ratio(left_box, band=2)
                    corner_right = rectangle_diag_antidiag_ratio(right_box, band=2)
                    corner_values = [x for x in (corner_left, corner_right) if np.isfinite(x)]
                    corner_ratio = float(np.mean(corner_values)) if corner_values else np.nan

                    before = junction_continuity(local, left_n, target_bins, right_n)
                    flipped = flip_interval(local, target_local0, target_local1)
                    after = junction_continuity(flipped, left_n, target_bins, right_n)
                    if not (np.isfinite(before) and np.isfinite(after)):
                        continue
                    delta = float(after - before)
                    # Convert the three continuous PHASC-inspired signals to
                    # bounded descriptive components. These are not event calls.
                    gcor_component = float(np.clip((max(0.0, left_rho) + max(0.0, -right_rho)) / 2.0, 0.0, 1.0))
                    corner_component = (float(np.clip(np.log2(max(corner_ratio, 1.0)) / 2.0, 0.0, 1.0))
                                        if np.isfinite(corner_ratio) else np.nan)
                    flip_component = float(np.clip(max(0.0, delta) / 2.0, 0.0, 1.0))
                    comps = [gcor_component, flip_component]
                    if np.isfinite(corner_component):
                        comps.append(corner_component)
                    evidence = float(np.mean(comps))

                    scale_deltas.append(delta)
                    scale_gcor.append(gcor_component)
                    if np.isfinite(corner_component):
                        scale_corner.append(corner_component)
                    scale_flip.append(flip_component)
                    scale_evidence.append(evidence)
            if scale_deltas:
                records.append({
                    "chrom": str(chrom),
                    "window_mb": float(size_mb),
                    "n_window_flank_evaluations": int(len(scale_deltas)),
                    "mean_delta_flip": float(np.mean(scale_deltas)),
                    "median_delta_flip": float(np.median(scale_deltas)),
                    "positive_delta_fraction": float(np.mean(np.asarray(scale_deltas) > 0)),
                    "mean_gcor_orientation_component": float(np.mean(scale_gcor)),
                    "mean_corner_component": float(np.mean(scale_corner)) if scale_corner else np.nan,
                    "mean_flip_component": float(np.mean(scale_flip)),
                    "mean_inversion_evidence": float(np.mean(scale_evidence)),
                })
                LOG.info("INV orientation scan: %s %.2f Mb; windows=%d; mean Δflip=%.4f",
                         chrom, size_mb, len(scale_deltas), float(np.mean(scale_deltas)))

    if not records:
        return np.nan, pd.DataFrame(columns=["chrom", "window_mb", "n_window_flank_evaluations",
                                             "mean_delta_flip", "median_delta_flip",
                                             "positive_delta_fraction", "mean_gcor_orientation_component",
                                             "mean_corner_component", "mean_flip_component",
                                             "mean_inversion_evidence"])
    df = pd.DataFrame(records)
    # Equal weighting across chromosome × window-size groups prevents long
    # chromosomes from dominating solely because they contain more scan windows.
    per_group = df.groupby(["chrom", "window_mb"], as_index=False)["mean_inversion_evidence"].mean()
    d_inv = float(np.clip(per_group["mean_inversion_evidence"].mean(), 0.0, 1.0))
    return d_inv, df


def get_chrom_names(clr) -> List[str]:
    ch = clr.chroms()[:]
    return [str(x) for x in ch["name"].tolist()]


def get_chrom_bin_counts(clr, chroms: Sequence[str]) -> Dict[str, int]:
    bins = clr.bins()[:]
    c = bins["chrom"].astype(str).value_counts().to_dict()
    return {chrom: int(c.get(chrom, 0)) for chrom in chroms}


def build_intra_metric_for_scale(
    clr,
    chroms: Sequence[str],
    resolution: int,
    balance: bool,
    edge_quantile: float,
    max_edges: int,
    gammas: Sequence[float],
    min_community_bins: int,
    seed: int,
) -> List[dict]:
    rows = []
    for chrom in chroms:
        matrix = read_matrix(clr, chrom, balance)
        valid = valid_bins_for_chrom(clr, chrom, balance, matrix)
        oe = distance_normalize_phasc(matrix)
        graph = graph_from_dense(
            oe, valid, edge_quantile=edge_quantile, max_edges=max_edges,
            direction_guided=False, seed=seed
        )
        for gamma in gammas:
            labels = louvain_labels(graph, matrix.shape[0], gamma, seed)
            disp, n_comm, n_valid = intra_community_dispersion(
                labels, valid, min_nodes=min_community_bins
            )
            rows.append({
                "metric": "intra_community_dispersion",
                "resolution": int(resolution),
                "gamma": float(gamma),
                "chrom": chrom,
                "value": disp,
                "n_communities_used": n_comm,
                "n_valid_bins": n_valid,
            })
        del matrix, oe, graph
    return rows


def sparse_block_edges(mat, offset_i: int, offset_j: int, q: float,
                        max_edges: int, weight_scale: float = 1.0,
                        min_enrichment: float = 1.5) -> List[Tuple[int, int, float]]:
    """Extract top trans edges from one chromosome-pair sparse/dense block.
    Normalize each chromosome pair by its own positive median before thresholding."""
    if hasattr(mat, "tocoo"):
        coo = mat.tocoo()
        r, c, v = coo.row, coo.col, np.asarray(coo.data, dtype=float)
    else:
        a = np.asarray(mat, dtype=float)
        r, c = np.where(a > 0)
        v = a[r, c]
    keep = np.isfinite(v) & (v > 0)
    r, c, v = r[keep], c[keep], v[keep]
    if v.size == 0:
        return []
    med = float(np.median(v))
    if med <= 0:
        return []
    norm = v / med
    threshold = max(float(np.quantile(norm, q)), float(min_enrichment))
    take = norm >= threshold
    r, c, norm = r[take], c[take], norm[take]
    if max_edges > 0 and norm.size > max_edges:
        keep_idx = np.argpartition(norm, -max_edges)[-max_edges:]
        r, c, norm = r[keep_idx], c[keep_idx], norm[keep_idx]
    weights = np.log1p(norm) * weight_scale
    return [(int(offset_i + x), int(offset_j + y), float(w))
            for x, y, w in zip(r, c, weights) if w > 0 and np.isfinite(w)]


def evaluate_inter_communities(
    clr,
    chroms: Sequence[str],
    balance: bool,
    resolution: int,
    edge_quantile: float,
    max_edges: int,
    radius: int,
    gamma: float,
    min_community_bins: int,
    trans_quantile: float,
    trans_weight_scale: float,
    trans_min_enrichment: float,
    seed: int,
) -> Tuple[float, dict]:
    """Whole-genome community analysis combining local cis edges and
    chromosome-pair-normalized trans edges. No translocation calls are made."""
    counts = get_chrom_bin_counts(clr, chroms)
    offsets: Dict[str, int] = {}
    cursor = 0
    for chrom in chroms:
        offsets[chrom] = cursor
        cursor += counts[chrom]
    n_nodes = cursor
    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))

    # Local cis edges preserve chromosome-local context.
    cis_edges = []
    for chrom in chroms:
        m = read_matrix(clr, chrom, balance)
        valid = valid_bins_for_chrom(clr, chrom, balance, m)
        oe = distance_normalize_phasc(m)
        n = oe.shape[0]
        for d in range(1, min(radius, n - 1) + 1):
            vals = np.diag(oe, k=d)
            ok = np.isfinite(vals) & (vals > 0)
            if not ok.any():
                continue
            threshold = float(np.quantile(vals[ok], edge_quantile))
            idx = np.flatnonzero(ok & (vals >= threshold) & valid[:-d] & valid[d:])
            for i in idx:
                cis_edges.append((offsets[chrom] + int(i), offsets[chrom] + int(i + d), float(oe[i, i+d])))
        del m, oe

    if max_edges > 0 and len(cis_edges) > max_edges:
        cis_edges = sorted(cis_edges, key=lambda z: z[2], reverse=True)[:max_edges]
    graph.add_weighted_edges_from(cis_edges)

    # Trans edges: read one chromosome-pair block at a time to limit memory.
    selector = clr.matrix(balance=balance, sparse=True)
    per_pair_cap = max(100, int(max_edges / max(1, len(chroms) * (len(chroms) - 1) / 2))) if max_edges > 0 else 5000
    trans_edges = []
    for i, c1 in enumerate(chroms):
        for c2 in chroms[i + 1:]:
            try:
                block = selector.fetch(c1, c2)
            except Exception:
                # Some cooler versions expect a tuple of regions.
                try:
                    block = selector.fetch((c1, 0, clr.chroms()[:].set_index("name").loc[c1, "length"]),
                                           (c2, 0, clr.chroms()[:].set_index("name").loc[c2, "length"]))
                except Exception as exc:
                    LOG.warning("Skip trans block %s-%s: %s", c1, c2, exc)
                    continue
            edges = sparse_block_edges(
                block, offsets[c1], offsets[c2], trans_quantile,
                per_pair_cap, trans_weight_scale, trans_min_enrichment
            )
            trans_edges.extend(edges)
    if max_edges > 0 and len(trans_edges) > max_edges:
        trans_edges = sorted(trans_edges, key=lambda z: z[2], reverse=True)[:max_edges]
    graph.add_weighted_edges_from(trans_edges)

    labels = louvain_labels(graph, n_nodes, gamma, seed)
    chrom_ids = np.empty(n_nodes, dtype=np.int32)
    valid_global = np.ones(n_nodes, dtype=bool)
    for ci, chrom in enumerate(chroms):
        a, b = offsets[chrom], offsets[chrom] + counts[chrom]
        chrom_ids[a:b] = ci
        # Mask invalid balancing bins from evaluation; node remains isolated.
        try:
            bin_df = clr.bins().fetch(chrom)
            if balance and "weight" in bin_df.columns:
                w = np.asarray(bin_df["weight"], dtype=float)
                valid_global[a:b] = np.isfinite(w) & (w != 0)
        except Exception:
            pass
    disp, n_comm, n_valid = inter_community_dispersion(
        labels, chrom_ids, valid_global, min_nodes=min_community_bins
    )
    info = {
        "metric": "inter_community_chromosome_dispersion",
        "resolution": int(resolution),
        "gamma": float(gamma),
        "value": disp,
        "n_communities_used": int(n_comm),
        "n_valid_bins": int(n_valid),
        "n_nodes": int(n_nodes),
        "n_cis_edges": int(len(cis_edges)),
        "n_trans_edges": int(len(trans_edges)),
    }
    return disp, info


def summarize_score(d_inter: float, d_intra: float, d_inv: float,
                    weights: Sequence[float]) -> Tuple[float, Dict[str, float]]:
    vals = [d_inter, d_intra, d_inv]
    names = ["interchromosomal", "intrachromosomal", "inversion"]
    w = np.asarray(weights, dtype=float)
    if w.size != 3 or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("--weights must contain three nonnegative values summing to > 0")
    w = w / w.sum()
    comp = {}
    terms = []
    ww = []
    for name, value, weight in zip(names, vals, w):
        if np.isfinite(value):
            value = float(np.clip(value, 0.0, 1.0))
            comp[name + "_score"] = 100.0 * (1.0 - value)
            terms.append(value)
            ww.append(float(weight))
        else:
            comp[name + "_score"] = np.nan
    if not terms:
        return np.nan, comp
    ww = np.asarray(ww, dtype=float)
    ww /= ww.sum()
    penalty = float(np.sum(np.asarray(terms) * ww))
    return 100.0 * (1.0 - penalty), comp


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Independent PHASC-style Hi-C structural-consistency assessment (prototype)."
    )
    parser.add_argument("--mcool", required=True, help=".mcool or .cool built from MAPQ30-filtered Hi-C pairs")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--resolutions", default="250000,500000",
                        help="Comma-separated resolutions for intra-community analysis (default: 250000,500000)")
    parser.add_argument("--inter-resolution", type=int, default=250000,
                        help="Resolution for whole-genome inter-community analysis")
    parser.add_argument("--inv-resolution", type=int, default=50000,
                        help="Resolution for exhaustive inversion-orientation scan")
    parser.add_argument("--no-balance", action="store_true", help="Use raw counts instead of Cooler balancing weights")
    parser.add_argument("--edge-quantile", type=float, default=0.90,
                        help="Keep edges at/above this quantile (0.90 = top 10%%)")
    parser.add_argument("--trans-quantile", type=float, default=0.995,
                        help="Within each chromosome pair, keep trans edges at/above this quantile")
    parser.add_argument("--trans-weight-scale", type=float, default=0.5,
                        help="Scale trans edge weights relative to local cis edges")
    parser.add_argument("--trans-min-enrichment", type=float, default=1.5,
                        help="Minimum trans contact / chromosome-pair median to create a graph edge")
    parser.add_argument("--radius", type=int, default=8, help="Local cis graph radius in bins")
    parser.add_argument("--gammas", default="0.5,1.0,1.5", help="Louvain resolution parameters")
    parser.add_argument("--inter-gamma", type=float, default=1.0, help="Louvain gamma for whole-genome graph")
    parser.add_argument("--min-community-bins", type=int, default=3,
                        help="Minimum community size included in dispersion summaries")
    parser.add_argument("--max-edges", type=int, default=100000,
                        help="Maximum edges per local graph / total inter graph (0 = no cap)")
    parser.add_argument("--inv-sizes-mb", default="1,2,5,10",
                        help="Fixed target window sizes for inversion orientation scan, in Mb")
    parser.add_argument("--inv-step-fraction", type=float, default=0.5,
                        help="Scan step as a fraction of target window size")
    parser.add_argument("--flank-fractions", default="0.2,0.5,1.0",
                        help="Left/right flank sizes as fractions of target window size")
    parser.add_argument("--min-occupancy", type=float, default=0.05,
                        help="Minimum nonzero fraction in flank-target rectangles for inversion scan")
    parser.add_argument("--weights", default="1,1,1",
                        help="Weights for inter,intra,inversion components (default equal)")
    parser.add_argument("--seed", type=int, default=1817)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s"
    )
    if not 0.0 < args.edge_quantile < 1.0 or not 0.0 < args.trans_quantile < 1.0:
        parser.error("edge quantiles must be between 0 and 1")
    if not 0.0 < args.inv_step_fraction <= 1.0:
        parser.error("--inv-step-fraction must be in (0,1]")
    if not 0.0 <= args.min_occupancy <= 1.0:
        parser.error("--min-occupancy must be in [0,1]")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        available = list_resolutions(args.mcool) if args.mcool.endswith(".mcool") else []
        requested = parse_ints(args.resolutions)
        selected = [r for r in requested if not available or r in available]
        if not selected:
            if available:
                selected = [min(available, key=lambda x: abs(x - 250000))]
            else:
                selected = [None]
        selected = sorted(set(selected))

        inter_res = args.inter_resolution
        if available and inter_res not in available:
            inter_res = min(available, key=lambda x: abs(x - inter_res))
            LOG.warning("Requested inter resolution unavailable; using %d", inter_res)
        inv_res = args.inv_resolution
        if available and inv_res not in available:
            inv_res = min(available, key=lambda x: abs(x - inv_res))
            LOG.warning("Requested inversion resolution unavailable; using %d", inv_res)

        all_scale_rows = []
        intra_balance_flags = []
        # Intra-chromosomal community dispersion, multi-resolution x multi-gamma.
        for res in selected:
            uri = build_uri(args.mcool, res)
            clr, use_balance = load_cooler(uri, not args.no_balance)
            chroms = get_chrom_names(clr)
            intra_balance_flags.append(bool(use_balance))
            gammas = parse_floats(args.gammas)
            LOG.info("Intra-community analysis: resolution=%s, chromosomes=%d, balance=%s",
                     res, len(chroms), use_balance)
            all_scale_rows.extend(build_intra_metric_for_scale(
                clr, chroms, int(clr.binsize), use_balance,
                args.edge_quantile, args.max_edges, gammas,
                args.min_community_bins, args.seed
            ))

        intra_df = pd.DataFrame(all_scale_rows)
        intra_values = intra_df["value"].dropna().to_numpy(dtype=float) if not intra_df.empty else np.array([])
        d_intra = float(np.mean(intra_values)) if intra_values.size else np.nan

        # Interchromosomal community dispersion at one selected resolution.
        inter_uri = build_uri(args.mcool, inter_res)
        clr_inter, inter_balance = load_cooler(inter_uri, not args.no_balance)
        chroms_inter = get_chrom_names(clr_inter)
        d_inter, inter_info = evaluate_inter_communities(
            clr_inter, chroms_inter, inter_balance, int(clr_inter.binsize),
            args.edge_quantile, args.max_edges, args.radius, args.inter_gamma,
            args.min_community_bins, args.trans_quantile,
            args.trans_weight_scale, args.trans_min_enrichment, args.seed
        )

        # Inversion direction score: scan all fixed windows; no candidate/event output.
        inv_uri = build_uri(args.mcool, inv_res)
        clr_inv, inv_balance = load_cooler(inv_uri, not args.no_balance)
        chroms_inv = get_chrom_names(clr_inv)
        d_inv, inv_df = evaluate_flip_windows(
            clr_inv, chroms_inv, int(clr_inv.binsize), inv_balance,
            parse_floats(args.inv_sizes_mb), args.inv_step_fraction,
            parse_floats(args.flank_fractions), args.min_occupancy
        )

        weights = parse_floats(args.weights)
        total_score, component_scores = summarize_score(d_inter, d_intra, d_inv, weights)
        summary = {
            "method": "PHASC-SCI-prototype",
            "input_mcool": str(Path(args.mcool).resolve()),
            "mapq30_provenance_required": True,
            "mapq30_verified_by_script": False,
            "balance_used_intra": bool(all(intra_balance_flags)) if intra_balance_flags else False,
            "balance_used_inter": bool(inter_balance),
            "balance_used_inversion": bool(inv_balance),
            "intra_resolutions_used": selected,
            "inter_resolution_used": int(clr_inter.binsize),
            "inversion_resolution_used": int(clr_inv.binsize),
            "raw_inter_community_dispersion": d_inter,
            "raw_intra_community_dispersion": d_intra,
            "raw_inversion_flip_penalty": d_inv,
            "component_scores_0_100": component_scores,
            "weights_inter_intra_inversion": weights,
            "provisional_total_score_0_100": total_score,
            "interpretation": "Exploratory, uncalibrated index. Calibrate against simulated assemblies and independent validation before scientific interpretation.",
            "no_detection_outputs_used": True,
            "no_candidate_coordinates_emitted": True,
        }

        if not intra_df.empty:
            intra_df.to_csv(outdir / "intra_community_metrics.tsv", sep="\t", index=False)
        pd.DataFrame([inter_info]).to_csv(outdir / "inter_community_metrics.tsv", sep="\t", index=False)
        if not inv_df.empty:
            inv_df.to_csv(outdir / "inversion_orientation_summary.tsv", sep="\t", index=False)
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
            "HiC_SCI_provisional": total_score,
            "inter_score": component_scores.get("interchromosomal_score", np.nan),
            "intra_score": component_scores.get("intrachromosomal_score", np.nan),
            "inversion_score": component_scores.get("inversion_score", np.nan),
            "D_inter": d_inter,
            "D_intra": d_intra,
            "D_inversion": d_inv,
            "calibration_status": "UNCalibrated_prototype",
        }]).to_csv(outdir / "summary.tsv", sep="\t", index=False)

        LOG.info("Finished. Provisional HiC-SCI = %s", f"{total_score:.3f}" if np.isfinite(total_score) else "NA")
        LOG.info("Results: %s", outdir.resolve())
        return 0
    except Exception as exc:
        LOG.exception("Assessment failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

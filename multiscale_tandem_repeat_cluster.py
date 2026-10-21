#!/usr/bin/env python3
"""Cluster-based tandem-repeat detection from one-chromosome mcool.

Replaces unstable sliding-window calls with:
  1) score every bin from a fixed local square context
  2) 1D connected-component clustering of TR-positive bins
  3) multi-resolution consensus (intersection core when scales agree)

Reuses TR score / classify / plot helpers from multiscale_tandem_repeat.py.
"""

from __future__ import annotations

import argparse
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

import cooler
import numpy as np
import pandas as pd

import multiscale_tandem_repeat as mtr


def score_bins_on_chromosome(
    mcool_path: Path,
    chrom: str,
    chrom_size: int,
    resolution: int,
    context_bins: int,
    balance: bool,
    parallel_min: float,
    lattice_min: float,
    fill_min: float,
    occ_min: float,
    mode: str,
    antiparallel_min: float = 0.80,
    chunk_bins: int = 400,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Assign TR scores to each bin center; return per-bin arrays + class tags."""
    clr = cooler.Cooler(f"{mcool_path}::/resolutions/{resolution}")
    n_bins = int(math.ceil(chrom_size / resolution))
    win = max(6, int(context_bins))
    half = win // 2

    pd_s = np.full(n_bins, np.nan)
    apd_s = np.full(n_bins, np.nan)
    lat_s = np.full(n_bins, np.nan)
    fill_s = np.full(n_bins, np.nan)
    occ_s = np.full(n_bins, np.nan)
    tags = [""] * n_bins
    is_hit = np.zeros(n_bins, dtype=bool)

    if n_bins < win:
        return pd_s, apd_s, lat_s, fill_s, occ_s, tags

    step = max(1, chunk_bins - win + 1)
    for chunk0 in range(0, n_bins - win + 1, step):
        chunk1 = min(n_bins, chunk0 + chunk_bins)
        if chunk1 - chunk0 < win:
            break
        start_bp = chunk0 * resolution
        end_bp = min(chunk1 * resolution, chrom_size)
        mat = mtr.fetch_matrix(clr, chrom, start_bp, end_bp, balance)
        m = mat.shape[0]
        if m < win:
            continue
        for local0 in range(0, m - win + 1):
            sub = mat[local0 : local0 + win, local0 : local0 + win]
            if sub.shape[0] != win:
                continue
            scores = mtr.score_window(sub)
            center = chunk0 + local0 + half
            if center >= n_bins:
                continue
            for key, arr in (
                ("parallel_diag", pd_s),
                ("antiparallel_diag", apd_s),
                ("lattice_peak", lat_s),
                ("offdiag_fill", fill_s),
                ("occupancy", occ_s),
            ):
                v = scores[key]
                if not np.isfinite(v):
                    continue
                old = arr[center]
                if not np.isfinite(old) or v > old:
                    arr[center] = v
            ok, tag = mtr.classify_tr(
                scores,
                parallel_min,
                lattice_min,
                fill_min,
                occ_min,
                mode=mode,
                antiparallel_min=antiparallel_min,
            )
            if ok:
                is_hit[center] = True
                if not tags[center]:
                    tags[center] = tag
                else:
                    parts = set(tags[center].split("+")) | set(tag.split("+"))
                    parts.discard("")
                    tags[center] = "+".join(t for t in mtr.TAG_ORDER if t in parts)

    for i in range(n_bins):
        if not is_hit[i]:
            tags[i] = ""
    return pd_s, apd_s, lat_s, fill_s, occ_s, tags


def cluster_hit_bins(
    tags: list[str],
    resolution: int,
    chrom_size: int,
    max_gap_bins: int,
    min_cluster_bins: int,
    pd_s: np.ndarray,
    apd_s: np.ndarray,
    lat_s: np.ndarray,
    fill_s: np.ndarray,
    occ_s: np.ndarray,
) -> pd.DataFrame:
    """1D connected components on TR-positive bins (allow small gaps)."""
    n = len(tags)
    hit = np.array([bool(t) for t in tags], dtype=bool)
    rows = []
    i = 0
    while i < n:
        if not hit[i]:
            i += 1
            continue
        j = i
        last = i
        while j < n:
            if hit[j]:
                last = j
                j += 1
                continue
            k = j
            while k < n and not hit[k] and (k - last - 1) < max_gap_bins:
                k += 1
            if k < n and hit[k] and (k - last - 1) <= max_gap_bins:
                j = k
                continue
            break
        start_bin, end_bin = i, last + 1
        width = end_bin - start_bin
        if width >= min_cluster_bins:
            sl = slice(start_bin, end_bin)
            parts: set[str] = set()
            for t in tags[start_bin:end_bin]:
                if t:
                    parts.update(t.split("+"))
            parts.discard("")
            start_bp = start_bin * resolution
            end_bp = min(end_bin * resolution, chrom_size)
            rows.append(
                {
                    "start_bin": start_bin,
                    "end_bin": end_bin,
                    "start_bp": start_bp,
                    "end_bp": end_bp,
                    "length_bp": end_bp - start_bp,
                    "n_hit_bins": int(hit[sl].sum()),
                    "tr_class": "+".join(t for t in mtr.TAG_ORDER if t in parts),
                    "parallel_diag": float(np.nanmax(pd_s[sl])),
                    "antiparallel_diag": float(np.nanmax(apd_s[sl])),
                    "lattice_peak": float(np.nanmax(lat_s[sl])),
                    "offdiag_fill": float(np.nanmax(fill_s[sl])),
                    "occupancy": float(np.nanmax(occ_s[sl])),
                }
            )
        i = end_bin
    return pd.DataFrame(rows)


def scan_chromosome_cluster(
    mcool_path: Path,
    chrom: str,
    chrom_size: int,
    resolution: int,
    context_bins: int,
    balance: bool,
    parallel_min: float,
    lattice_min: float,
    fill_min: float,
    occ_min: float,
    mode: str,
    max_gap_bins: int,
    min_cluster_bins: int,
    antiparallel_min: float = 0.80,
) -> pd.DataFrame:
    pd_s, apd_s, lat_s, fill_s, occ_s, tags = score_bins_on_chromosome(
        mcool_path,
        chrom,
        chrom_size,
        resolution,
        context_bins,
        balance,
        parallel_min,
        lattice_min,
        fill_min,
        occ_min,
        mode,
        antiparallel_min=antiparallel_min,
    )
    cl = cluster_hit_bins(
        tags,
        resolution,
        chrom_size,
        max_gap_bins,
        min_cluster_bins,
        pd_s,
        apd_s,
        lat_s,
        fill_s,
        occ_s,
    )
    if cl.empty:
        return cl
    cl.insert(0, "chrom", chrom)
    cl.insert(1, "resolution", resolution)
    cl.insert(2, "scale_id", f"r{resolution}")
    return cl


def build_consensus_cluster(
    candidates: pd.DataFrame,
    min_support: int,
    overlap_fraction: float,
) -> pd.DataFrame:
    """Multi-res consensus; union overlapping clusters across resolutions."""
    cols = [
        "chrom",
        "start_bp",
        "end_bp",
        "length_bp",
        "tr_class",
        "support_scales",
        "support_resolutions",
        "parallel_diag",
        "antiparallel_diag",
        "lattice_peak",
        "offdiag_fill",
        "occupancy",
    ]
    if candidates.empty:
        return pd.DataFrame(columns=cols)

    summaries = []
    for chrom, sub in candidates.groupby("chrom", sort=False):
        ordered = sub.sort_values(["start_bp", "end_bp"]).reset_index(drop=True)
        n = len(ordered)
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
                if int(ordered.iloc[j]["start_bp"]) >= int(ordered.iloc[i]["end_bp"]):
                    break
                if ordered.iloc[i]["scale_id"] == ordered.iloc[j]["scale_id"]:
                    continue
                if (
                    mtr.interval_similarity(
                        int(ordered.iloc[i]["start_bp"]),
                        int(ordered.iloc[i]["end_bp"]),
                        int(ordered.iloc[j]["start_bp"]),
                        int(ordered.iloc[j]["end_bp"]),
                    )
                    >= overlap_fraction
                ):
                    union(i, j)

        groups: dict[int, list[int]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)

        for indices in groups.values():
            group = ordered.iloc[indices]
            support_resolutions = int(group["resolution"].nunique())
            if support_resolutions < min_support:
                continue
            starts = group["start_bp"].astype(int)
            ends = group["end_bp"].astype(int)
            start_bp, end_bp = int(starts.min()), int(ends.max())
            if end_bp <= start_bp:
                fine = group.loc[group["resolution"].astype(int).idxmin()]
                start_bp = int(fine["start_bp"])
                end_bp = int(fine["end_bp"])
            tags: set[str] = set()
            for t in group["tr_class"].astype(str):
                tags.update(t.split("+"))
            tags.discard("")
            anti_col = (
                group["antiparallel_diag"]
                if "antiparallel_diag" in group.columns
                else pd.Series([np.nan] * len(group))
            )
            summaries.append(
                {
                    "chrom": chrom,
                    "start_bp": start_bp,
                    "end_bp": end_bp,
                    "length_bp": max(0, end_bp - start_bp),
                    "tr_class": "+".join(t for t in mtr.TAG_ORDER if t in tags),
                    "support_scales": int(group["scale_id"].nunique()),
                    "support_resolutions": ",".join(
                        map(str, sorted(group["resolution"].unique()))
                    ),
                    "parallel_diag": float(np.nanmax(group["parallel_diag"])),
                    "antiparallel_diag": float(np.nanmax(anti_col)),
                    "lattice_peak": float(np.nanmax(group["lattice_peak"])),
                    "offdiag_fill": float(np.nanmax(group["offdiag_fill"])),
                    "occupancy": float(np.nanmax(group["occupancy"])),
                }
            )
    if not summaries:
        return pd.DataFrame(columns=cols)
    return (
        pd.DataFrame(summaries)
        .sort_values(["chrom", "start_bp", "end_bp"])
        .reset_index(drop=True)
    )


def _selfcheck() -> None:
    # contiguous hits with a 1-bin gap should merge; isolated hits below min drop
    tags = [""] * 30
    for i in range(5, 10):
        tags[i] = "stripe"
    tags[11] = "stripe"  # gap of 1 after 5..9 ends at 10 empty, 11 hit
    tags[20] = "cloud"  # isolated
    pd_s = np.zeros(30)
    apd_s = np.zeros(30)
    lat_s = np.zeros(30)
    fill_s = np.zeros(30)
    occ_s = np.ones(30)
    cl = cluster_hit_bins(
        tags,
        50_000,
        30 * 50_000,
        max_gap_bins=1,
        min_cluster_bins=3,
        pd_s=pd_s,
        apd_s=apd_s,
        lat_s=lat_s,
        fill_s=fill_s,
        occ_s=occ_s,
    )
    assert len(cl) == 1, cl
    assert int(cl.iloc[0]["start_bin"]) == 5 and int(cl.iloc[0]["end_bin"]) == 12
    print("[selfcheck] cluster ok", flush=True)
    mtr._selfcheck()


def _worker(args_tuple):
    return scan_chromosome_cluster(*args_tuple)


def parse_args():
    p = argparse.ArgumentParser(
        description="Cluster-based TR detection (bin scoring + 1D connected components)"
    )
    p.add_argument("mcool", nargs="?", type=Path)
    p.add_argument("--outdir", type=Path, default=Path("TR_results_ms_cluster"))
    p.add_argument("--chromosomes", default="")
    p.add_argument("--resolutions", default="auto")
    p.add_argument(
        "--context-bins",
        type=int,
        default=20,
        help="local square size (bins) used to score each center bin",
    )
    p.add_argument(
        "--max-gap-bins",
        type=int,
        default=2,
        help="allow this many non-hit bins inside a cluster",
    )
    p.add_argument(
        "--min-cluster-bins",
        type=int,
        default=5,
        help="drop clusters shorter than this many bins",
    )
    p.add_argument("--parallel-diag-min", type=float, default=mtr.DEFAULT_PARALLEL_MIN)
    p.add_argument(
        "--antiparallel-diag-min",
        type=float,
        default=0.80,
        help="reverse-homology anti-diag gate (default 0.80; much higher than parallel)",
    )
    p.add_argument("--lattice-peak-min", type=float, default=mtr.DEFAULT_LATTICE_MIN)
    p.add_argument("--offdiag-fill-min", type=float, default=mtr.DEFAULT_FILL_MIN)
    p.add_argument("--min-occupancy", type=float, default=mtr.DEFAULT_OCC_MIN)
    p.add_argument(
        "--mode",
        choices=("stripe", "strict", "any"),
        default="strict",
    )
    p.add_argument("--min-support", type=int, default=1)
    p.add_argument("--overlap-fraction", type=float, default=0.5)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--no-balance", action="store_true")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--selfcheck", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.selfcheck:
        _selfcheck()
        return 0
    if args.mcool is None:
        print("[ERROR] mcool path required (or pass --selfcheck)", file=sys.stderr)
        return 1
    mcool = args.mcool.resolve()
    if not mcool.exists():
        print(f"[ERROR] missing mcool: {mcool}", file=sys.stderr)
        return 1

    available = mtr.discover_resolutions(mcool)
    if not available:
        print("[ERROR] no resolutions in mcool", file=sys.stderr)
        return 1
    if args.resolutions.strip().lower() == "auto":
        resolutions = [r for r in available if r <= 200_000] or available[:3]
    else:
        resolutions = [int(x) for x in args.resolutions.split(",") if x.strip()]
        missing = sorted(set(resolutions) - set(available))
        if missing:
            print(f"[ERROR] resolutions not in mcool: {missing}", file=sys.stderr)
            return 1

    chroms = mtr.discover_chromosomes(mcool, resolutions[0])
    if args.chromosomes.strip():
        want = {c.strip() for c in args.chromosomes.split(",") if c.strip()}
        chroms = [(c, s) for c, s in chroms if c in want]
    if not chroms:
        print("[ERROR] no chromosomes selected", file=sys.stderr)
        return 1

    balance = not args.no_balance
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    hic_dir = outdir / "HiC"
    hic_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[INFO] method=cluster mcool={mcool} resolutions={resolutions} "
        f"chroms={len(chroms)} context={args.context_bins} "
        f"max_gap={args.max_gap_bins} min_cluster={args.min_cluster_bins} "
        f"mode={args.mode} balance={balance}",
        flush=True,
    )

    tasks = []
    for chrom, chrom_size in chroms:
        for resolution in resolutions:
            tasks.append(
                (
                    mcool,
                    chrom,
                    chrom_size,
                    resolution,
                    args.context_bins,
                    balance,
                    args.parallel_diag_min,
                    args.lattice_peak_min,
                    args.offdiag_fill_min,
                    args.min_occupancy,
                    args.mode,
                    args.max_gap_bins,
                    args.min_cluster_bins,
                    args.antiparallel_diag_min,
                )
            )

    tables = []
    jobs = max(1, int(args.jobs))
    if jobs == 1:
        for t in tasks:
            df = scan_chromosome_cluster(*t)
            print(f"[SCAN] {t[1]} r{t[3]}: clusters={len(df)}", flush=True)
            if not df.empty:
                tables.append(df)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futs = {pool.submit(_worker, t): t for t in tasks}
            for fut in as_completed(futs):
                t = futs[fut]
                df = fut.result()
                print(f"[SCAN] {t[1]} r{t[3]}: clusters={len(df)}", flush=True)
                if not df.empty:
                    tables.append(df)

    per_scale = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    per_scale.to_csv(outdir / "TR_clusters_per_scale.tsv", sep="\t", index=False)
    consensus = build_consensus_cluster(
        per_scale, args.min_support, args.overlap_fraction
    )
    consensus.to_csv(outdir / "ALL_TR.tsv", sep="\t", index=False)
    print(f"[DONE] consensus events={len(consensus)} -> {outdir / 'ALL_TR.tsv'}", flush=True)

    if not args.no_plot and not consensus.empty:
        chrom_size_map = dict(chroms)
        for chrom, sub in consensus.groupby("chrom", sort=False):
            mtr.plot_chrom_overview(
                mcool,
                chrom,
                chrom_size_map[chrom],
                available,
                sub,
                balance,
                hic_dir / f"{chrom}_TR_overview.png",
            )
            for _, row in sub.iterrows():
                tag = f"{chrom}_{int(row['start_bp'])}_{int(row['end_bp'])}"
                mtr.plot_local(
                    mcool,
                    chrom,
                    int(row["start_bp"]),
                    int(row["end_bp"]),
                    chrom_size_map[chrom],
                    available,
                    balance,
                    hic_dir / "local" / f"{tag}.png",
                )
        print(f"[PLOT] wrote under {hic_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import gzip
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

for e in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS"
]:
    os.environ.setdefault(e,"1")


import argparse
import cooler
import numpy as np
import pandas as pd

from multiprocessing import Pool

import networkx as nx

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator



############################################################
# Arguments
############################################################

def parse_args():

    parser=argparse.ArgumentParser(
        description=
        "Direction guided Louvain block GCOR inversion detector"
    )


    parser.add_argument(
        "--mcool_file",
        required=True
    )


    parser.add_argument(
        "--balance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use ICE-balanced contacts (default: --no-balance, raw counts)"
    )


    # optional
    # 如果不给，则自动生成
    parser.add_argument(
        "--block_file",
        default=None,
        help="existing Louvain block file"
    )


    parser.add_argument(
        "--lift_blocks_dir",
        default=None,
        help=(
            "directory of finest-scale {chrom}.blocks.tsv; when set, "
            "skip Louvain and project those communities onto --res"
        )
    )


    parser.add_argument(
        "--res",
        type=int,
        default=50000
    )


    parser.add_argument(
        "--min_block",
        type=int,
        default=500000,
        help=(
            "merge communities shorter than this many bp into an adjacent "
            "community (default: 500000)"
        )
    )


    #############################
    # Louvain parameters
    #############################

    parser.add_argument(
        "--radius",
        type=int,
        default=8,
        help="graph connection radius"
    )


    parser.add_argument(
        "--direction_lambda",
        type=float,
        default=0.5
    )


    parser.add_argument(
        "--louvain_resolution",
        type=float,
        default=1.0
    )


    parser.add_argument(
        "--secondary_louvain_resolution",
        type=float,
        default=2.0,
        help=(
            "re-cluster large failed blocks at this Louvain resolution to "
            "separate nested inversions (default: 2.0; <=0 disables)"
        )
    )


    parser.add_argument(
        "--secondary_split_min_size",
        type=int,
        default=3_000_000,
        help=(
            "only secondary-split failed INV blocks at least this many bp "
            "(default: 3000000; <=0 disables)"
        )
    )


    #############################
    # GCOR
    #############################

    parser.add_argument(
        "--smooth",
        type=int,
        default=5
    )


    parser.add_argument(
        "--rho_cutoff",
        type=float,
        default=0.3
    )


    parser.add_argument(
        "--min_neighbor_blocks",
        type=int,
        default=2,
        help=(
            "deprecated: ignored by detection; kept for CLI compatibility "
            "(default: 2)"
        )
    )


    parser.add_argument(
        "--max_neighbor_blocks",
        type=int,
        default=8,
        help=(
            "deprecated: ignored by detection; kept for CLI compatibility "
            "(default: 8)"
        )
    )


    parser.add_argument(
        "--neighbor_scale_fractions",
        type=float,
        nargs="+",
        default=[0.2, 0.5, 1.0],
        help=(
            "flank lengths as fractions of the target block size; "
            "each scale is tested (default: 0.2 0.5 1.0)"
        )
    )


    parser.add_argument(
        "--neighbor_scale_min_support",
        type=int,
        default=2,
        help=(
            "minimum scale votes required on EACH flank (left and right) "
            "when at least this many scales are valid (default: 2)"
        )
    )


    parser.add_argument(
        "--corner_flank_fraction",
        type=float,
        default=1.0,
        help=(
            "flank length for S-flank corner boxes as a fraction of target "
            "size S (default: 1.0). Boxes are left-flank x S and S x right-flank."
        )
    )


    parser.add_argument(
        "--corner_band_bins",
        type=int,
        default=2,
        help=(
            "half-width in bins around the scaled diagonal / anti-diagonal "
            "inside each S-flank box (default: 2)"
        )
    )


    parser.add_argument(
        "--corner_enrichment_min",
        type=float,
        default=0.8,
        help=(
            "minimum per-side diag/anti on a GCOR flank x S matrix; "
            "either left or right >= this keeps an INV (default: 0.8)"
        )
    )


    parser.add_argument(
        "--corner_enrichment_max",
        type=float,
        default=5.0,
        help=(
            "maximum per-side diag/anti on a GCOR flank x S matrix; "
            "rejects hyper-bright local X / tick artifacts "
            "(default: 5.0; <=0 disables upper cap)"
        )
    )


    parser.add_argument(
        "--corner_diag_bg_min",
        type=float,
        default=0.0,
        help=(
            "optional minimum diag/background on the passing flank x S side "
            "(default: 0 disables; kept for diagnostics only)"
        )
    )


    parser.add_argument(
        "--lattice_peak_max",
        type=float,
        default=0.30,
        help=(
            "max allowed fraction of strong secondary diagonals inside S; "
            "rejects checkerboard/repeat lattices (default: 0.30; <=0 disables)"
        )
    )


    parser.add_argument(
        "--parallel_diag_max",
        type=float,
        default=0.30,
        help=(
            "max allowed (best off-diag band mean)/(main-diag) inside S; "
            "rejects parallel-stripe tandem/satellite (default: 0.30; <=0 disables)"
        )
    )


    parser.add_argument(
        "--offdiag_fill_max",
        type=float,
        default=0.25,
        help=(
            "max allowed mean(|i-j|>=3)/mean(1<=|i-j|<=2) inside S; rejects "
            "blocky/grid repeat clouds (default: 0.25; <=0 disables)"
        )
    )


    parser.add_argument(
        "--s_qc_resolution",
        type=int,
        default=50_000,
        help=(
            "fetch S at this resolution (or finer detection res) for lattice/"
            "parallel/fill QC so coarse bins still see stripes (default: 50000)"
        )
    )


    parser.add_argument(
        "--min_s_occupancy",
        type=float,
        default=0.60,
        help=(
            "minimum fraction of nonzero cells in S x S; rejects low-signal / "
            "gappy intervals (default: 0.60; <=0 disables)"
        )
    )


    parser.add_argument(
        "--max_s_gap_frac",
        type=float,
        default=0.25,
        help=(
            "max fraction of S x S cells covered by sparse (white-band) rows "
            "or cols; rejects blocky gap lattices (default: 0.25; <0 disables)"
        )
    )


    parser.add_argument(
        "--oneside_min_size",
        type=int,
        default=8_000_000,
        help=(
            "size gate for pure single-flank GCOR: |S| >= this may pass on "
            "one flank alone; smaller S may still pass oneside only if the "
            "weak flank keeps the correct sign (Lρ>0 / Rρ<0). "
            "default: 8000000; <=0 = always allow pure oneside"
        )
    )


    parser.add_argument(
        "--gap_oneside_min_frac",
        type=float,
        default=0.10,
        help=(
            "if only one flank passes GCOR and S gap fraction >= this, mask "
            "GAP bins and re-test the failing flank (default: 0.10; <=0 disables)"
        )
    )


    parser.add_argument(
        "--gap_oneside_corner_min",
        type=float,
        default=1.20,
        help=(
            "unused (kept for CLI compatibility); gap rescue no longer grants "
            "oneside INV without the other flank recovering"
        )
    )


    parser.add_argument(
        "--gcor_max_gap_bp",
        type=int,
        default=2_000_000,
        help=(
            "max genomic gap between target and a flank block still treated as "
            "abutting for GCOR (default: 2000000)"
        )
    )


    parser.add_argument(
        "--gap_cluster_diag_frac",
        type=float,
        default=0.35,
        help=(
            "before Louvain, drop bins whose diagonal is below this fraction of "
            "the chromosome median (default: 0.35) so non-GAP bins on either "
            "side of a hole become neighbors and can cluster together"
        )
    )


    parser.add_argument(
        "--outdir",
        default="INV_results"
    )


    parser.add_argument(
        "--nprocess",
        type=int,
        default=2
    )


    parser.add_argument(
        "--multiscale_inputs",
        nargs="+",
        default=None,
        help=(
            "merge existing inversion results instead of running detection; "
            "each item is RES=FILE_OR_DIRECTORY, for example "
            "50000=INV_50k 100000=INV_100k"
    )
    )


    parser.add_argument(
        "--merge_outdir",
        default=None,
        help="output directory for multiscale merged results"
    )


    parser.add_argument(
        "--merge_overlap",
        type=float,
        default=0.50,
        help=(
            "minimum overlap/intersection-over-shorter-length for merging; "
            "partial overlaps also require compatible features (default: 0.5)"
        )
    )


    parser.add_argument(
        "--merge_gap",
        type=int,
        default=0,
        help=(
            "fixed gap allowed for feature-supported adjacent calls; default "
            "0 means use only the adaptive gap ratio"
        )
    )


    parser.add_argument(
        "--merge_gap_ratio",
        type=float,
        default=0.25,
        help=(
            "adaptive allowed gap = max(merge_gap, ratio * shorter call); "
            "default: 0.25"
        )
    )


    parser.add_argument(
        "--merge_containment",
        type=float,
        default=0.80,
        help=(
            "if one call contains another, minimum contained fraction for "
            "merging (default: 0.8)"
        )
    )


    parser.add_argument(
        "--feature_distance",
        type=float,
        default=1.5,
        help="maximum standardized feature distance for adjacent calls"
    )


    parser.add_argument(
        "--merge_neighbor_cutoff",
        type=float,
        default=0.3,
        help=(
            "rho cutoff used after merging to test the combined interval "
            "against its outside neighbors (default: 0.3)"
        )
    )


    parser.add_argument(
        "--merge_neighbor_support",
        type=int,
        default=1,
        help=(
            "minimum number of distinct resolutions with an abnormal outside "
            "neighbor signal required for merging (default: 1)"
        )
    )


    parser.add_argument(
        "--merge_flank_ratio",
        type=float,
        default=0.5,
        help=(
            "neighbor flank size as a fraction of the merged interval; "
            "default: 0.5"
        )
    )


    parser.add_argument(
        "--min_scale_support",
        type=int,
        default=1,
        help="minimum number of distinct resolutions supporting a merged call"
    )


    parser.add_argument(
        "--feature_max_bins",
        type=int,
        default=300,
        help="maximum bins used for per-interval feature extraction"
    )


    parser.add_argument(
        "--within_scale_samples",
        type=int,
        default=20,
        help=(
            "number of random internal regions used to validate a proposed "
            "within-resolution merge (default: 20)"
        )
    )


    parser.add_argument(
        "--within_scale_sample_ratio",
        type=float,
        default=0.20,
        help=(
            "length of each sampled internal region relative to the proposed "
            "merged interval (default: 0.20)"
        )
    )


    parser.add_argument(
        "--within_scale_vote_threshold",
        type=int,
        default=4,
        help=(
            "merge only when the number of sampled internal regions voting "
            "as inversions is >= this value (default: 4); high votes mean "
            "the proposed span still looks inverted as a whole"
        )
    )


    parser.add_argument(
        "--within_scale_min_valid",
        type=int,
        default=5,
        help=(
            "minimum number of successfully evaluated random samples needed "
            "to accept a merge (default: 5)"
        )
    )


    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="random seed for reproducible internal sampling"
    )


    args=parser.parse_args()


    if args.min_neighbor_blocks<2:

        parser.error(
            "--min_neighbor_blocks must be >= 2"
        )


    if args.min_block<=0:

        parser.error(
            "--min_block must be > 0"
        )


    if (
        args.max_neighbor_blocks
        <
        args.min_neighbor_blocks
    ):

        parser.error(
            "--max_neighbor_blocks must be >= --min_neighbor_blocks"
        )


    if not args.neighbor_scale_fractions:

        parser.error("--neighbor_scale_fractions must not be empty")


    if any(f <= 0 for f in args.neighbor_scale_fractions):

        parser.error("--neighbor_scale_fractions values must be > 0")


    if args.neighbor_scale_min_support < 1:

        parser.error("--neighbor_scale_min_support must be >= 1")


    if args.corner_flank_fraction <= 0:

        parser.error("--corner_flank_fraction must be > 0")


    if args.corner_band_bins < 0:

        parser.error("--corner_band_bins must be >= 0")


    if args.corner_enrichment_min < 0:

        parser.error("--corner_enrichment_min must be >= 0")


    if args.corner_enrichment_max < 0:

        parser.error("--corner_enrichment_max must be >= 0")


    if (
        args.corner_enrichment_max > 0
        and args.corner_enrichment_max < args.corner_enrichment_min
    ):

        parser.error(
            "--corner_enrichment_max must be >= --corner_enrichment_min "
            "(or <=0 to disable)"
        )


    if args.corner_diag_bg_min < 0:

        parser.error("--corner_diag_bg_min must be >= 0")


    if args.min_s_occupancy > 1:

        parser.error("--min_s_occupancy must be <= 1")


    if args.max_s_gap_frac > 1:

        parser.error("--max_s_gap_frac must be <= 1")


    if not 0<args.merge_overlap<=1:

        parser.error("--merge_overlap must be in (0, 1]")


    if not 0<args.merge_containment<=1:

        parser.error("--merge_containment must be in (0, 1]")


    if args.merge_gap<0:

        parser.error("--merge_gap must be >= 0")


    if args.merge_gap_ratio<0:

        parser.error("--merge_gap_ratio must be >= 0")


    if args.merge_neighbor_cutoff<=0:

        parser.error("--merge_neighbor_cutoff must be > 0")


    if args.merge_neighbor_support<1:

        parser.error("--merge_neighbor_support must be >= 1")


    if args.merge_flank_ratio<=0:

        parser.error("--merge_flank_ratio must be > 0")


    if args.min_scale_support<1:

        parser.error("--min_scale_support must be >= 1")


    if args.within_scale_samples<1:

        parser.error("--within_scale_samples must be >= 1")


    if not 0<args.within_scale_sample_ratio<1:

        parser.error("--within_scale_sample_ratio must be in (0, 1)")


    if not 0<=args.within_scale_vote_threshold<=args.within_scale_samples:

        parser.error(
            "--within_scale_vote_threshold must be between 0 and "
            "--within_scale_samples"
        )


    if not 1<=args.within_scale_min_valid<=args.within_scale_samples:

        parser.error(
            "--within_scale_min_valid must be between 1 and "
            "--within_scale_samples"
        )


    return args





############################################################
# Distance normalization
############################################################


def distance_normalize(M):

    M=np.asarray(
        M,
        dtype=float
    )


    n=M.shape[0]


    out=np.zeros_like(M)



    for d in range(n):


        diag=np.diag(
            M,
            k=d
        )


        vals=diag[
            np.isfinite(diag)
            &
            (diag>0)
        ]


        if len(vals)<5:
            continue


        bg=np.median(vals)


        if bg<=0:
            continue



        idx=np.arange(
            n-d
        )


        out[
            idx,
            idx+d
        ]=M[
            idx,
            idx+d
        ]/bg


        if d>0:

            out[
                idx+d,
                idx
            ]=M[
                idx+d,
                idx
            ]/bg



    return out





############################################################
# Direction information
############################################################


def safe_corr(x,y):


    if len(x)<3:
        return 0


    if np.std(x)==0 or np.std(y)==0:
        return 0


    # Sparse Hi-C profiles can have zero variance.  The original result is
    # retained, but NumPy's expected divide/invalid warnings are suppressed;
    # downstream code already converts a non-finite correlation to zero.
    with np.errstate(
        divide="ignore",
        invalid="ignore"
    ):

        return np.corrcoef(
            x,
            y
        )[0,1]




def get_profile(
        M,
        i,
        radius
):


    n=M.shape[0]


    values=[]


    for j in range(
        i-radius,
        i+radius+1
    ):

        if j==i:
            continue


        if 0<=j<n:
            values.append(
                M[i,j]
            )

        else:
            values.append(
                0
            )


    return np.array(values)





def calculate_direction_information(
        M,
        radius
):


    n=M.shape[0]


    profiles=[
        get_profile(
            M,
            i,
            radius
        )
        for i in range(n)
    ]



    direction=np.zeros(
        n-1
    )



    confidence=np.zeros(
        n-1
    )



    for i in range(n-1):


        p1=profiles[i]
        p2=profiles[i+1]


        forward=safe_corr(
            p1[:-1],
            p2[1:]
        )


        reverse=safe_corr(
            p1[1:],
            p2[:-1]
        )



        d=forward-reverse


        direction[i]=d


        confidence[i]=abs(d)/(
                abs(forward)
            +
            abs(reverse)
            +
            1e-8
        )



    return direction,confidence





############################################################
# Build graph
############################################################


def build_contact_graph(
        M,
        radius,
        gap_mask=None
):
    """
    Contact band graph. If gap_mask is set, GAP nodes are left isolated;
    prefer louvain_labels_gap_compressed so arms across a hole can still meet.
    """


    n=M.shape[0]


    G=nx.Graph()


    G.add_nodes_from(
        range(n)
    )

    gap=None
    if gap_mask is not None:
        gap=np.asarray(gap_mask,dtype=bool)
        if gap.shape[0]!=n:
            gap=None


    for i in range(n):
        if gap is not None and gap[i]:
            continue

        for j in range(
            i+1,
            min(
                n,
                i+radius+1
            )
        ):
            if gap is not None and gap[j]:
                continue


            w=M[i,j]


            if w>0:

                G.add_edge(
                    i,
                    j,
                    contact_weight=float(w)
                )


    return G


def paint_gap_bin_labels(labels, gap_mask):
    """
    Fill GAP bins with flanking non-GAP labels. If both sides share a label
    (arms clustered together after compression), the hole is painted shut.
    """
    labels=np.asarray(labels,dtype=int).copy()
    gap=np.asarray(gap_mask,dtype=bool)
    if gap.shape[0]!=labels.shape[0] or not gap.any():
        return labels
    n=int(labels.size)
    for i in np.where(gap)[0]:
        left=None
        for j in range(i-1,-1,-1):
            if not gap[j]:
                left=int(labels[j])
                break
        right=None
        for j in range(i+1,n):
            if not gap[j]:
                right=int(labels[j])
                break
        if left is not None and right is not None and left==right:
            labels[i]=left
        elif left is not None:
            labels[i]=left
        elif right is not None:
            labels[i]=right
    return labels


def louvain_labels_gap_compressed(
    norm, radius, gap_mask, louvain_resolution=1.0, direction_lambda=0.5
):
    """
    Mask GAP by dropping those bins from the graph topology: non-GAP bins on
    either side of a hole become neighbors under --radius, so they can join
    one Louvain community. Labels are mapped back and GAP bins painted.
    """
    norm=np.asarray(norm,dtype=float)
    gap=np.asarray(gap_mask,dtype=bool)
    n=norm.shape[0]
    if gap.shape[0]!=n or not gap.any():
        G=build_contact_graph(norm, radius, None)
        direction,confidence=calculate_direction_information(norm, radius)
        G=fuse_direction(G, direction, confidence, direction_lambda)
        return run_louvain(G, louvain_resolution)

    keep=np.where(~gap)[0]
    if keep.size<4:
        G=build_contact_graph(norm, radius, gap)
        direction,confidence=calculate_direction_information(norm, radius)
        G=fuse_direction(G, direction, confidence, direction_lambda)
        return paint_gap_bin_labels(run_louvain(G, louvain_resolution), gap)

    sub=norm[np.ix_(keep, keep)]
    G=build_contact_graph(sub, radius, None)
    direction,confidence=calculate_direction_information(sub, radius)
    G=fuse_direction(G, direction, confidence, direction_lambda)
    sub_labels=run_louvain(G, louvain_resolution)
    labels=np.full(n,-1,dtype=int)
    labels[keep]=sub_labels
    return paint_gap_bin_labels(labels, gap)





def fuse_direction(
    G,
    direction,
    confidence,
        lam=0.5
):


    for u,v,data in G.edges(data=True):


        if abs(u-v)==1:


            idx=min(u,v)


            D=direction[idx]


            C=confidence[idx]


            data["weight"]=(
                data["contact_weight"]
                *
                (
                    1+
                    lam*D*C
                )
            )


        else:


            data["weight"]=(
                data["contact_weight"]
            )


    return G





############################################################
# Louvain
############################################################


def run_louvain(
        G,
        resolution=1.0
):


    n=G.number_of_nodes()
    if n==0:
        return np.zeros(0,dtype=int)
    # empty / zero-weight graphs make networkx modularity divide by deg_sum**2
    deg_sum=sum(d for _,d in G.degree(weight="weight"))
    if G.number_of_edges()==0 or deg_sum==0:
        return np.arange(n,dtype=int)

    communities=nx.community.louvain_communities(
        G,
        weight="weight",
        resolution=resolution,
        seed=42
    )


    labels=np.zeros(
        G.number_of_nodes(),
        dtype=int
    )


    for cid,c in enumerate(communities):

        for node in c:

            labels[node]=cid


    return labels


def hic_gap_bin_mask(hic, diag_frac=0.15, abs_floor=1.0):
    """True for bins that look like coverage gaps (very low self-contact)."""
    hic=np.asarray(hic,dtype=float)
    diag=np.asarray(hic.diagonal(),dtype=float)
    positive=diag[diag>0]
    med=float(np.median(positive)) if positive.size else 1.0
    thr=max(float(abs_floor), float(diag_frac)*med)
    return diag<thr


def mask_gap_bins_in_matrix(M, gap_mask):
    """Zero rows/cols of GAP bins so they do not enter Louvain weights."""
    M=np.asarray(M,dtype=float).copy()
    gap=np.asarray(gap_mask,dtype=bool)
    if gap.shape[0]!=M.shape[0]:
        return M
    if not gap.any():
        return M
    M[gap,:]=0
    M[:,gap]=0
    return M


def labels_to_blocks(labels, origin_bp, resolution, gap_mask=None):
    """
    Contiguous same-label runs → [start,end,cid].
    GAP bins are omitted (split communities across white bands).
    """
    labels=np.asarray(labels)
    n=int(labels.size)
    resolution=int(resolution)
    origin_bp=int(origin_bp)
    gap=None
    if gap_mask is not None:
        gap=np.asarray(gap_mask,dtype=bool)
        if gap.shape[0]!=n:
            gap=None
    blocks=[]
    i=0
    while i<n:
        if gap is not None and gap[i]:
            i+=1
            continue
        s0=i
        cur=int(labels[i])
        i+=1
        while (
            i<n
            and not (gap is not None and gap[i])
            and int(labels[i])==cur
        ):
            i+=1
        blocks.append([
            origin_bp+s0*resolution,
            origin_bp+i*resolution,
            cur,
        ])
    return blocks


def absorb_short_blocks(blocks, min_size):
    """
    Merge communities shorter than min_size into a touching neighbor.
    Never expand across a genomic hole (GAP-split blocks stay split).
    """
    blocks=[list(b) for b in blocks]
    while len(blocks)>1:
        short=[
            i for i,b in enumerate(blocks)
            if b[1]-b[0]<min_size
        ]
        if not short:
            break
        i=min(short, key=lambda j: blocks[j][1]-blocks[j][0])
        left_touch=i>0 and blocks[i-1][1]==blocks[i][0]
        right_touch=i+1<len(blocks) and blocks[i][1]==blocks[i+1][0]
        if i==0:
            if right_touch:
                blocks[1][0]=blocks[0][0]
            del blocks[0]
        elif i==len(blocks)-1:
            if left_touch:
                blocks[i-1][1]=blocks[i][1]
            del blocks[i]
        else:
            left_size=blocks[i-1][1]-blocks[i-1][0]
            right_size=blocks[i+1][1]-blocks[i+1][0]
            if left_touch and (not right_touch or left_size>=right_size):
                blocks[i-1][1]=blocks[i][1]
                del blocks[i]
            elif right_touch:
                blocks[i+1][0]=blocks[i][0]
                del blocks[i]
            else:
                del blocks[i]
    merged=[]
    for block in blocks:
        if merged and merged[-1][2]==block[2] and merged[-1][1]==block[0]:
            merged[-1][1]=block[1]
        else:
            merged.append(block)
    return merged


def _local_with_span(local, start, end):
    """Rebuild a block list with one target spanning [start, end)."""
    start=int(start)
    end=int(end)
    left=[list(b) for b in local if int(b[1])<=start]
    right=[]
    for b in local:
        b0,b1=int(b[0]),int(b[1])
        if b0>=end:
            right.append(list(b))
        elif b1>end and b0<end:
            right.append([end, b1, b[2]])
    if not left and local and start>int(local[0][0]):
        left=[[int(local[0][0]), start, local[0][2]]]
    if not right and local and end<int(local[-1][1]):
        right=[[end, int(local[-1][1]), local[-1][2]]]
    trial=left+[[start, end, 0]]+right
    return trial, len(left)


def _grow_edge_for_index(trial, idx, edge_oneside):
    if idx<=0 or idx>=len(trial)-1:
        return edge_oneside or ("left" if idx==len(trial)-1 else "right")
    return None


def _span_side_score(rec, edge_oneside):
    """
    Quality of an INV span for grow selection.
    Both-flank: weaker |ρ| only when BOTH votes pass (avoid oneside creep).
    Tip oneside: the available arm's |ρ|.
    """
    if not rec or rec.get("INV")!=1:
        return -1.0
    L=float(rec["left_rho"])
    R=float(rec["right_rho"])
    if edge_oneside=="left":
        return L
    if edge_oneside=="right":
        return -R
    if int(rec.get("left_vote",0))!=1 or int(rec.get("right_vote",0))!=1:
        return -1.0
    return float(min(L, -R))


def grow_secondary_inv_rec(
    hic, local, index, rec, args, edge_oneside=None, sub_hi=None
):
    """
    Extend a secondary INV into the next/prev secondary sub (partial OK).

    Not the farthest INV=1 end (that overshoots into the next TAD via weak
    oneside). Scan candidates and keep the farthest span whose both-flank
    weaker |ρ| stays within 70% of the peak (chr1B: ~716.2–716.35, not 716.55).
    """
    if not rec or rec.get("INV")!=1:
        return rec
    res=int(args.res)
    start0=int(rec["start"])
    end0=int(rec["end"])
    best=dict(rec)

    def _test(start, end):
        trial,idx=_local_with_span(local, start, end)
        edge=_grow_edge_for_index(trial, idx, edge_oneside)
        return detect_block_INV(hic, trial, idx, args, edge_oneside=edge), edge

    # Right grow
    if index+1<len(local):
        right_limit=int(local[index+1][1])
    else:
        right_limit=end0
    if sub_hi is not None and index+1<sub_hi:
        right_limit=max(right_limit, int(local[min(index+1, len(local)-1)][1]))
    right_limit=min(right_limit, end0+3_000_000)
    if right_limit>end0+res:
        scored=[]
        peak=-1.0
        for end in range(end0, right_limit+res, res):
            r,edge=_test(start0, end)
            s=_span_side_score(r, edge)
            if s<0:
                continue
            scored.append((end, r, s))
            if s>peak:
                peak=s
        if scored and peak>0:
            thr=0.70*peak
            # farthest end still near the peak both-flank strength
            keep=[t for t in scored if t[2]>=thr]
            if keep:
                best=dict(max(keep, key=lambda t: t[0])[1])

    # Left grow (same rule)
    start=int(best["start"])
    end=int(best["end"])
    if index>0:
        left_limit=int(local[index-1][0])
    else:
        left_limit=start
    left_limit=max(left_limit, end-3_000_000)
    if left_limit<start-res:
        scored=[]
        peak=-1.0
        for st in range(start, left_limit-res, -res):
            r,edge=_test(st, end)
            s=_span_side_score(r, edge)
            if s<0:
                continue
            scored.append((st, r, s))
            if s>peak:
                peak=s
        if scored and peak>0:
            thr=0.70*peak
            keep=[t for t in scored if t[2]>=thr]
            if keep:
                best=dict(min(keep, key=lambda t: t[0])[1])
    return best


def _grow_secondary_inv_selfcheck():
    # ponytail: partial grow into the next tile must beat whole-next merge.
    local=[
        [0, 500_000, 0],
        [500_000, 1_000_000, 1],
        [1_000_000, 2_000_000, 2],
        [2_000_000, 3_000_000, 3],
    ]
    trial,idx=_local_with_span(local, 500_000, 1_500_000)
    assert idx==1 and trial[idx]==[500_000, 1_500_000, 0], trial
    assert trial[idx+1][0]==1_500_000 and trial[idx+1][1]==2_000_000


def secondary_louvain_split(
    hic, start, end, resolution, radius, louvain_resolution, min_size,
    direction_lambda=0.5, gap_mask=None, gap_cluster_diag_frac=0.35
):
    """
    Re-Louvain one genomic interval so nested communities can separate.
    Returns sub-blocks covering [start, end] (GAP runs omitted).
    """
    start=int(start)
    end=int(end)
    resolution=int(resolution)
    i0=start//resolution
    i1=end//resolution
    if i1-i0<4:
        return [[start, end, 0]]
    sub=np.asarray(hic[i0:i1, i0:i1], dtype=float)
    sub=np.nan_to_num(sub, nan=0.0)
    if gap_mask is None:
        local_gap=hic_gap_bin_mask(sub, diag_frac=float(gap_cluster_diag_frac))
    else:
        local_gap=np.asarray(gap_mask[i0:i1],dtype=bool)
        if local_gap.shape[0]!=sub.shape[0]:
            local_gap=hic_gap_bin_mask(sub, diag_frac=float(gap_cluster_diag_frac))
    norm=distance_normalize(sub)
    labels=louvain_labels_gap_compressed(
        norm, radius, local_gap, louvain_resolution, direction_lambda
    )
    # Painted labels: emit contiguous runs (GAP holes closed when arms match).
    blocks=labels_to_blocks(labels, start, resolution, gap_mask=None)
    return absorb_short_blocks(blocks, min_size)


############################################################
# Generate block file
############################################################


def generate_block_file(
    labels,
    chrom,
    resolution,
        outfile,
    gap_mask=None
):


    spans=labels_to_blocks(labels, 0, resolution, gap_mask)
    blocks=[
        [chrom, int(s), int(e), int(cid)]
        for s, e, cid in spans
    ]



    pd.DataFrame(
        blocks
    ).to_csv(
        outfile,
        sep="\t",
        header=False,
        index=False
    )


    return outfile


def write_block_table(blocks, chrom, outfile):
    """Write [start,end,cluster] blocks as chr/start/end/cluster TSV."""
    rows=[[chrom, int(s), int(e), int(c)] for s, e, c in blocks]
    pd.DataFrame(rows).to_csv(outfile, sep="\t", header=False, index=False)
    return outfile


def lift_blocks_to_resolution(blocks, target_res):
    """
    Project fine-scale communities onto a coarser bin grid.

    Edges snap to target_res; if snapping would overlap the previous block,
    the start is pushed forward so partitions stay a single ordered cover.
    No Louvain re-clustering.
    """
    target_res=int(target_res)
    if target_res<=0:
        raise ValueError("target_res must be > 0")
    out=[]
    for s, e, cid in blocks:
        s2=(int(s)//target_res)*target_res
        e2=((int(e)+target_res-1)//target_res)*target_res
        if e2<=s2:
            e2=s2+target_res
        if out and s2<out[-1][1]:
            s2=out[-1][1]
            if e2<=s2:
                continue
        out.append([s2, e2, int(cid)])
    return out


def _lift_blocks_selfcheck():
    # ponytail: one assert that fails if snap/overlap logic regresses
    blocks=[[0, 150_000, 1], [150_000, 400_000, 2], [400_000, 900_000, 3]]
    out=lift_blocks_to_resolution(blocks, 200_000)
    assert out==[[0, 200_000, 1], [200_000, 400_000, 2], [400_000, 1_000_000, 3]], out


def _edge_oneside_guard_selfcheck():
    # ponytail: tip oneside must accept only the true list tip; a synthetic
    # terminal INV (left-flank orientation only) must pass INV=1.
    class _A:
        min_block=1
        neighbor_scale_fractions=[1.0]
        neighbor_scale_min_support=1
        gcor_max_gap_bp=0
        res=50_000
        smooth=1
        rho_cutoff=0.3
        corner_band_bins=2
        corner_enrichment_min=0.8
        corner_enrichment_max=5.0
        corner_diag_bg_min=0.0
        lattice_peak_max=0
        parallel_diag_max=0
        offdiag_fill_max=0
        min_s_occupancy=0
        max_s_gap_frac=-1
        oneside_min_size=0
        gap_oneside_min_frac=0
        gap_mask=None
    args=_A()
    blocks=[[0, 500_000, 0], [500_000, 1_000_000, 1], [1_000_000, 1_500_000, 2]]
    hic=np.eye(30, dtype=float)
    assert detect_block_INV(hic, blocks, 2, args, edge_oneside="right") is None
    assert detect_block_INV(hic, blocks, 0, args, edge_oneside="left") is None
    assert detect_block_INV(hic, blocks, 1, args, edge_oneside="left") is None

    # Two tiles: left flank + chrom-end tip. Paint Left×S so column means
    # rise along S (INV left-flank sign) and put mass on the scaled diagonal.
    tip_blocks=[[0, 500_000, 0], [500_000, 1_000_000, 1]]
    n=20
    tip_hic=np.zeros((n, n), dtype=float)
    for i in range(n):
        tip_hic[i, i]=10.0
    for r in range(0, 10):
        for c in range(10, 20):
            # far end of S contacts left flank → diag band in Left×S
            tip_hic[r, c]=1.0+0.5*(c-10)+0.05*r
    rec=detect_block_INV(
        tip_hic, tip_blocks, 1, args, edge_oneside="left"
    )
    assert rec is not None and rec.get("INV")==1, rec
    assert rec.get("left_vote")==1 and rec.get("detection_side")=="left"


############################################################
# Read Louvain blocks
############################################################

def load_blocks(
    block_file,
    chrom,
        min_size
):

    """
    block format:

    chr start end cluster

    merge adjacent same-cluster intervals, then absorb every community shorter
    than min_size into an adjacent community
    """


    df=pd.read_csv(
        block_file,
        sep="\t",
        header=None,
        names=[
            "chr",
            "start",
            "end",
            "cluster"
        ]
    )


    df=df[
        df["chr"]==chrom
    ]


    if df.empty:
        return []


    df=df.sort_values(
        "start"
    )


    blocks=[]


    current=None


    for _,r in df.iterrows():


        if current is None:


            current=[
                int(r.start),
                int(r.end),
                int(r.cluster)
            ]


        elif (
            r.cluster==current[2]
            and
            r.start==current[1]
        ):


            current[1]=int(r.end)



        else:


            blocks.append(
                current
            )


            current=[
                int(r.start),
                int(r.end),
                int(r.cluster)
            ]



    blocks.append(
        current
    )


    # Repeatedly absorb the shortest undersized community.  End communities
    # have one possible destination.  An internal one is merged into the
    # longer adjacent community (left on ties), which avoids an arbitrary
    # coordinate gap while perturbing the established partition as little as
    # possible.
    while len(blocks)>1:

        short_indices=[
            i
            for i,block in enumerate(blocks)
            if block[1]-block[0]<min_size
        ]


        if not short_indices:
            break


        i=min(
            short_indices,
            key=lambda j: blocks[j][1]-blocks[j][0]
        )


        if i==0:

            blocks[1][0]=blocks[0][0]
            del blocks[0]


        elif i==len(blocks)-1:

            blocks[i-1][1]=blocks[i][1]
            del blocks[i]


        else:

            left_size=blocks[i-1][1]-blocks[i-1][0]
            right_size=blocks[i+1][1]-blocks[i+1][0]


            if left_size>=right_size:

                blocks[i-1][1]=blocks[i][1]
                del blocks[i]


            else:

                blocks[i+1][0]=blocks[i][0]
                del blocks[i]


    # Absorbing A-B-A into the left A can make two A blocks adjacent again.
    # Collapse such blocks so they remain one community.
    merged=[]


    for block in blocks:

        if (
            merged
            and merged[-1][2]==block[2]
            and merged[-1][1]==block[0]
        ):

            merged[-1][1]=block[1]


        else:

            merged.append(block)


    blocks=merged


    return blocks





############################################################
# Extract block matrix / gap handling for GCOR
############################################################


def _trim_block_gap_edges(block,gap_mask,resolution):
    """Shrink block ends that are pure GAP bins; keep cid."""
    resolution=int(resolution)
    i0=int(block[0])//resolution
    i1=int(block[1])//resolution
    if i1<=i0:
        return [int(block[0]),int(block[1]),int(block[2])]
    region=np.asarray(gap_mask[i0:i1],dtype=bool)
    if not region.size or not region.any():
        return [int(block[0]),int(block[1]),int(block[2])]
    keep=np.where(~region)[0]
    if keep.size==0:
        return None
    return [
        (i0+int(keep[0]))*resolution,
        (i0+int(keep[-1])+1)*resolution,
        int(block[2]),
    ]


def merge_blocks_across_gaps(
    blocks,gap_mask,resolution,max_gap_bp=2_000_000,gap_frac=0.80
):
    """
    After Louvain: trim GAP edges, drop pure-gap blocks, and bridge neighbors
    separated only by a short gap so GCOR sees one S spanning both arms.
    """
    if not blocks:
        return blocks
    resolution=int(resolution)
    max_gap_bins=max(1,int(max_gap_bp)//resolution)
    kept=[]
    for block in blocks:
        trimmed=_trim_block_gap_edges(block,gap_mask,resolution)
        if trimmed is None:
            continue
        i0=trimmed[0]//resolution
        i1=trimmed[1]//resolution
        region=gap_mask[i0:i1] if i1>i0 else np.array([],dtype=bool)
        if region.size and float(region.mean())>=gap_frac:
            continue
        kept.append(trimmed)
    if not kept:
        return [list(b) for b in blocks]
    out=[kept[0]]
    for block in kept[1:]:
        prev=out[-1]
        sep=block[0]-prev[1]
        if sep<0:
            out.append(block)
            continue
        if sep==0:
            # Abutting tiles stay separate unless same community id.
            if prev[2]==block[2]:
                prev[1]=block[1]
            else:
                out.append(block)
            continue
        if sep>max_gap_bp:
            out.append(block)
            continue
        i0=prev[1]//resolution
        i1=block[0]//resolution
        mid=gap_mask[i0:i1] if i1>i0 else np.array([],dtype=bool)
        if mid.size and mid.size<=max_gap_bins and float(mid.mean())>=gap_frac:
            prev[1]=block[1]
        else:
            out.append(block)
    return out


def _blocks_abut_or_gap_only(end_a,start_b,gap_mask,resolution,max_gap_bp):
    """True if intervals touch, or the open interval between them is only gap."""
    end_a=int(end_a); start_b=int(start_b)
    if end_a==start_b:
        return True
    if start_b<end_a:
        return False
    if gap_mask is None or start_b-end_a>int(max_gap_bp):
        return False
    i0=end_a//int(resolution)
    i1=start_b//int(resolution)
    mid=np.asarray(gap_mask[i0:i1],dtype=bool)
    return bool(mid.size) and float(mid.mean())>=0.80


def get_block_matrix(
    hic,
    block1,
    block2,
    resolution,
    gap_mask=None
):


    r0=block1[0]//resolution
    r1=block1[1]//resolution


    c0=block2[0]//resolution
    c1=block2[1]//resolution



    mat=hic[
        r0:r1,
        c0:c1
    ]


    mat=np.nan_to_num(
        mat,
        nan=0
    )

    # Optional GAP drop (off by default in callers): white-band bins only.
    # Must filter axes independently so stacked flanks keep matching dims.
    if gap_mask is not None and mat.size:
        row_gap=np.asarray(gap_mask[r0:r1],dtype=bool)
        col_gap=np.asarray(gap_mask[c0:c1],dtype=bool)
        if row_gap.size==mat.shape[0]:
            mat=mat[~row_gap,:]
        if col_gap.size==mat.shape[1]:
            mat=mat[:,~col_gap]
        if mat.shape[0]<2 or mat.shape[1]<2:
            return np.zeros((0,0),dtype=float)

    return mat





############################################################
# GCOR directional trend
############################################################


def flank_blocks_by_length(blocks, index, side, length_bp):
    """Accumulate contiguous neighbor blocks until span >= length_bp."""

    length_bp = max(int(length_bp), 1)
    collected = []

    if side == "left":
        i = index - 1
        while i >= 0:
            block = blocks[i]
            if collected and collected[0][0] != block[1]:
                break
            collected.insert(0, block)
            span = collected[-1][1] - collected[0][0]
            if span >= length_bp:
                break
            i -= 1
    elif side == "right":
        i = index + 1
        while i < len(blocks):
            block = blocks[i]
            if collected and collected[-1][1] != block[0]:
                break
            collected.append(block)
            span = collected[-1][1] - collected[0][0]
            if span >= length_bp:
                break
            i += 1
    else:
        raise ValueError("side must be 'left' or 'right'")

    return collected




def calculate_direction(
    mat,
    axis,
        smooth=5
):


    if mat.size==0:

        return 0,1,None



    if axis=="column":

        signal=np.mean(
            mat,
            axis=0
        )


    elif axis=="row":

        signal=np.mean(
            mat,
            axis=1
        )


    else:

        raise ValueError(
            "axis error"
        )



    signal=np.nan_to_num(
        signal
    )


    if len(signal)>smooth:


        from scipy.ndimage import uniform_filter1d


        signal=uniform_filter1d(
            signal,
            size=smooth
        )



    x=np.arange(
        len(signal)
    )


    if len(signal)<3:

        return 0,1,signal



    from scipy.stats import spearmanr


    rho,p=spearmanr(
        x,
        signal
    )


    if np.isnan(rho):

        rho=0



    return rho,p,signal



def _scaled_band_mask(n_row, n_col, anti=False, band=2):
    """Boolean mask for diagonal or anti-diagonal band in a rectangle."""
    if n_row <= 0 or n_col <= 0:
        return np.zeros((max(n_row, 0), max(n_col, 0)), dtype=bool)
    rr = np.arange(n_row, dtype=float)[:, None]
    cc = np.arange(n_col, dtype=float)[None, :]
    # Map both axes to [0, 1] so non-square boxes share a common diagonal.
    if n_row > 1:
        rr = rr / (n_row - 1)
    else:
        rr = rr * 0.0
    if n_col > 1:
        cc = cc / (n_col - 1)
    else:
        cc = cc * 0.0
    target = (1.0 - cc) if anti else cc
    # Convert band bins to fractional tolerance.
    tol = (band + 0.5) / max(n_row, n_col, 1)
    return np.abs(rr - target) <= tol


def rectangle_diag_antidiag_means(mat, band=2):
    """Mean contact on scaled diagonal and anti-diagonal bands."""
    mat = np.asarray(mat, dtype=float)
    mat = np.nan_to_num(mat, nan=0.0)
    if mat.ndim != 2 or mat.size == 0:
        return float("nan"), float("nan")
    n_row, n_col = mat.shape
    dmask = _scaled_band_mask(n_row, n_col, anti=False, band=band)
    amask = _scaled_band_mask(n_row, n_col, anti=True, band=band)
    dvals = mat[dmask]
    avals = mat[amask]
    diag = float(np.mean(dvals)) if dvals.size else float("nan")
    anti = float(np.mean(avals)) if avals.size else float("nan")
    return diag, anti


def rectangle_offband_mean(mat, band=2):
    """Mean contact outside the scaled diag and anti-diag bands (background)."""
    mat = np.asarray(mat, dtype=float)
    mat = np.nan_to_num(mat, nan=0.0)
    if mat.ndim != 2 or mat.size == 0:
        return float("nan")
    n_row, n_col = mat.shape
    dmask = _scaled_band_mask(n_row, n_col, anti=False, band=band)
    amask = _scaled_band_mask(n_row, n_col, anti=True, band=band)
    other = ~(dmask | amask)
    if not other.any():
        return float("nan")
    return float(np.mean(mat[other]))


def s_matrix_lattice_peak_fraction(mat, near_diag=2, peak_ratio=2.0):
    """
    Fraction of off-main-diagonal offsets whose mean exceeds peak_ratio
    times the median offset mean. High for repeat/checkerboard lattices.
    """
    mat = np.asarray(mat, dtype=float)
    mat = np.nan_to_num(mat, nan=0.0)
    n = mat.shape[0]
    if n < 6 or mat.shape[0] != mat.shape[1]:
        return float("nan")
    vals = []
    for k in range(-(n - 1), n):
        if abs(k) <= near_diag:
            continue
        vals.append(float(np.mean(np.diag(mat, k))))
    vals = np.asarray(vals, dtype=float)
    if vals.size == 0 or not np.isfinite(vals).any():
        return float("nan")
    med = float(np.median(vals))
    if med <= 0:
        mean = float(np.mean(vals))
        if mean <= 0:
            return 0.0
        med = mean
    return float(np.mean(vals > peak_ratio * med))



def s_matrix_parallel_diag_ratio(mat, near_diag=2, min_band_bins=3):
    """
    Strongest off-main-diagonal band mean / main-diagonal mean.

    Tandem repeats / satellite arrays draw a bright stripe parallel to the
    main diagonal; true INV bowties are corner-local and keep this ratio low.
    """
    mat = np.asarray(mat, dtype=float)
    mat = np.nan_to_num(mat, nan=0.0)
    n = mat.shape[0]
    if n < 6 or mat.shape[0] != mat.shape[1]:
        return float("nan")
    main = float(np.mean(np.diag(mat, 0)))
    if main <= 0:
        return float("nan")
    best = 0.0
    min_bins = max(int(min_band_bins), max(3, n // 6))
    for k in range(int(near_diag) + 1, n - 1):
        d = np.diag(mat, k)
        if d.size < min_bins:
            continue
        best = max(best, float(np.mean(d)))
    return float(best / main)


def s_matrix_offdiag_fill_ratio(mat, near_lo=1, near_hi=2, far_lo=3):
    """
    mean(|i-j| >= far_lo) / mean(near_lo <= |i-j| <= near_hi).

    Blocky/grid repeat clouds keep high contact far from the diagonal;
    normal Hi-C (and true INV bowties) decay, so this ratio stays low.
    """
    mat = np.asarray(mat, dtype=float)
    mat = np.nan_to_num(mat, nan=0.0)
    n = mat.shape[0]
    if n < 6 or mat.shape[0] != mat.shape[1]:
        return float("nan")
    i, j = np.indices(mat.shape)
    d = np.abs(i - j)
    near = (d >= int(near_lo)) & (d <= int(near_hi))
    far = d >= int(far_lo)
    if not near.any() or not far.any():
        return float("nan")
    near_mean = float(mat[near].mean())
    if near_mean <= 0:
        return float("nan")
    return float(mat[far].mean() / near_mean)


def s_matrix_for_qc(hic, target, args):
    """
    S×S used for lattice / parallel-diag / fill QC.
    Prefer ≤50 kb so coarse detection bins still see repeat stripes.
    """
    start = int(target[0])
    end = int(target[1])
    res = int(args.res)
    qc_cap = int(getattr(args, "s_qc_resolution", 50_000))
    qc_res = min(res, qc_cap) if qc_cap > 0 else res
    mcool = getattr(args, "mcool_file", None)
    chrom = getattr(args, "qc_chrom", None)
    if (
        mcool
        and chrom
        and qc_res < res
        and (end - start) // qc_res >= 6
    ):
        try:
            clr = cooler.Cooler(f"{mcool}::resolutions/{qc_res}")
            mat = clr.matrix(
                balance=bool(getattr(args, "balance", False))
            ).fetch((chrom, start, end))
            return np.nan_to_num(np.asarray(mat, dtype=float), nan=0.0)
        except Exception:
            pass
    return get_block_matrix(hic, target, target, res, None)


def s_matrix_occupancy(mat):
    """Fraction of nonzero cells in S x S (low = sparse / empty signal)."""
    mat = np.asarray(mat, dtype=float)
    mat = np.nan_to_num(mat, nan=0.0)
    if mat.size == 0:
        return float("nan")
    return float(np.mean(mat > 0))


def s_matrix_gap_frac(mat, bright_factor=0.35, floor=0.08):
    """
    Fraction of S x S cells that sit on a sparse row or column.

    Bright cell = contact >= median(nonzero). A row/col is a white band if
    its bright occupancy is < bright_factor * median row/col bright occupancy
    (with an absolute floor). High for gappy checkerboards with empty strips.
    """
    mat = np.asarray(mat, dtype=float)
    mat = np.nan_to_num(mat, nan=0.0)
    if mat.ndim != 2 or min(mat.shape) < 4:
        return float("nan")
    nz = mat[mat > 0]
    if nz.size == 0:
        return 1.0
    bright = mat >= float(np.median(nz))
    row_b = bright.mean(axis=1)
    col_b = bright.mean(axis=0)
    row_thr = max(float(floor), float(np.median(row_b)) * float(bright_factor))
    col_thr = max(float(floor), float(np.median(col_b)) * float(bright_factor))
    gap_row = row_b < row_thr
    gap_col = col_b < col_thr
    return float(np.mean(gap_row[:, None] | gap_col[None, :]))


def s_flank_corner_enrichment(hic, target, flank_bp, resolution, band=2):
    """
    Corner enrichment from the two S-flank boxes in the Hi-C map:

      top box    : left_flank x S
      right box  : S x right_flank

    Within each box, compare diagonal vs anti-diagonal band means.

    Boxes use genomic order:
      left box  rows = left flank (distal -> breakpoint)
                cols = S (start -> end)
      right box rows = S (start -> end)
                cols = right flank (breakpoint -> distal)

    Without inversion, strong contacts sit at the adjacent corner
    (breakpoint x near-end of S), which falls on the anti-diagonal band.
    With inversion, the far end of S contacts the flank, enriching the
    diagonal band.

    Returns
    -------
    enrichment : float
        mean(diag) / mean(anti) over both boxes
    details : dict
    """
    s, e = int(target[0]), int(target[1])
    flank_bp = max(int(flank_bp), int(resolution))
    n = hic.shape[0]
    chrom_end = int(n * resolution)

    left0 = max(0, s - flank_bp)
    right1 = min(chrom_end, e + flank_bp)
    details = {
        "left0": left0,
        "right1": right1,
        "left_diag": float("nan"),
        "left_anti": float("nan"),
        "right_diag": float("nan"),
        "right_anti": float("nan"),
    }

    if left0 >= s or e >= right1:
        return float("nan"), details

    def to_bins(a, b):
        i0 = max(0, int(a) // resolution)
        i1 = min(n, max(i0 + 1, (int(b) + resolution - 1) // resolution))
        return i0, i1

    li0, li1 = to_bins(left0, s)
    si0, si1 = to_bins(s, e)
    ri0, ri1 = to_bins(e, right1)
    if min(li1 - li0, si1 - si0, ri1 - ri0) < 2:
        return float("nan"), details

    mat = np.asarray(hic, dtype=float)
    left_box = np.nan_to_num(mat[li0:li1, si0:si1], nan=0.0)   # left_flank x S
    right_box = np.nan_to_num(mat[si0:si1, ri0:ri1], nan=0.0)  # S x right_flank

    ld, la = rectangle_diag_antidiag_means(left_box, band=band)
    rd, ra = rectangle_diag_antidiag_means(right_box, band=band)
    details.update(
        left_diag=ld, left_anti=la, right_diag=rd, right_anti=ra
    )

    diags = [v for v in (ld, rd) if np.isfinite(v)]
    antis = [v for v in (la, ra) if np.isfinite(v)]
    if not diags or not antis:
        return float("nan"), details

    diag_mean = float(np.mean(diags))
    anti_mean = float(np.mean(antis))
    # INV elevates diagonal; normal adjacency elevates anti-diagonal.
    if anti_mean <= 0:
        score = float("inf") if diag_mean > 0 else 0.0
    else:
        score = diag_mean / anti_mean
    return float(score), details


def gcor_oneside_accept(
        oneside_gcor,
        left_vote,
        right_vote,
        mean_left,
        mean_right,
        target_size,
        oneside_min_size,
):
    """
    Oneside GCOR gate.

    Long (|S| >= oneside_min_size, or oneside_min_size<=0): one flank over
    cutoff is enough. Short: one flank over cutoff, and the weak flank must
    still have the INV-correct sign (Lρ>0 / Rρ<0) even if |ρ| fails cutoff.
    """
    if int(oneside_gcor)!=1:
        return 0
    if int(oneside_min_size)<=0 or int(target_size)>=int(oneside_min_size):
        return 1
    # short: require correct sign on the weak side
    if int(left_vote)==1 and int(right_vote)==0:
        return int(float(mean_right)<0)
    if int(right_vote)==1 and int(left_vote)==0:
        return int(float(mean_left)>0)
    return 0


def detect_block_INV(
    hic,
    blocks,
    index,
        args,
    edge_oneside=None,
):

    """
    Test flanks at multiple fractions of the target block length.

    GCOR pass: both flanks over |ρ| cutoff; or one flank over cutoff with
    size-dependent oneside rules (see gcor_oneside_accept). Gap in S can
    also rescue the failing flank after masking GAP bins.

    edge_oneside:
        None — need both neighbors (default)
        "left" — target is list end; left flank only (chrom 3' / terminal)
        "right" — target is list start; right flank only (chrom 5')
    """


    n_blocks=len(blocks)
    if edge_oneside=="left":
        if index!=n_blocks-1 or index<1:
            return None
    elif edge_oneside=="right":
        if index!=0 or n_blocks<2:
            return None
    elif index<=0 or index>=n_blocks-1:
        return None


    need_left=edge_oneside!="right"
    need_right=edge_oneside!="left"


    target=blocks[index]


    # load_blocks normally absorbs short communities.  Keep this guard for a
    # direct caller that bypasses loading and supplies an undersized target.
    if target[1]-target[0]<args.min_block:
        return None


    target_size=target[1]-target[0]
    fractions=list(args.neighbor_scale_fractions)
    min_support=int(args.neighbor_scale_min_support)
    max_gap_bp=int(getattr(args,"gcor_max_gap_bp",2_000_000))

    scale_records=[]
    left_rhos=[]
    right_rhos=[]
    left_ps=[]
    right_ps=[]
    left_votes=[]
    right_votes=[]
    left_starts=[]
    left_ends=[]
    right_starts=[]
    right_ends=[]


    for frac in fractions:
        flank_bp=max(1, int(round(target_size*float(frac))))
        left_blocks=(
            flank_blocks_by_length(blocks, index, "left", flank_bp)
            if need_left else []
        )
        right_blocks=(
            flank_blocks_by_length(blocks, index, "right", flank_bp)
            if need_right else []
        )


        if (need_left and not left_blocks) or (need_right and not right_blocks):
            scale_records.append({
                "fraction":float(frac),
                "valid":0,
                "left_vote":0,
                "right_vote":0,
            })
            continue


        # Neighbors must abut target, or be separated only by a short gap.
        gap_mask=getattr(args,"gap_mask",None)
        max_gap_bp=int(getattr(args,"gcor_max_gap_bp",2_000_000))
        abut_ok=True
        if need_left:
            abut_ok=abut_ok and _blocks_abut_or_gap_only(
                left_blocks[-1][1],target[0],gap_mask,args.res,max_gap_bp
            )
        if need_right:
            abut_ok=abut_ok and _blocks_abut_or_gap_only(
                target[1],right_blocks[0][0],gap_mask,args.res,max_gap_bp
            )
        if not abut_ok:
            scale_records.append({
                "fraction":float(frac),
                "valid":0,
                "left_vote":0,
                "right_vote":0,
            })
            continue


        left_span=(
            left_blocks[-1][1]-left_blocks[0][0] if need_left else flank_bp
        )
        right_span=(
            right_blocks[-1][1]-right_blocks[0][0] if need_right else flank_bp
        )


        # Chromosome ends may not reach the requested flank; treat as invalid
        # so they do not dilute majority voting.
        if (
            (need_left and left_span<flank_bp)
            or (need_right and right_span<flank_bp)
        ):
            scale_records.append({
                "fraction":float(frac),
                "valid":0,
                "left_vote":0,
                "right_vote":0,
            })
            continue


        # Drop GAP bins inside Left×S / S×Right so white bands do not
        # flatten Spearman orientation (same mask as Louvain clustering).
        left_matrix=None
        right_matrix=None
        if need_left:
            left_parts=[
                get_block_matrix(hic, block, target, args.res, gap_mask)
                for block in left_blocks
            ]
            left_parts=[p for p in left_parts if p.size]
            if not left_parts:
                scale_records.append({
                    "fraction":float(frac),
                    "valid":0,
                    "left_vote":0,
                    "right_vote":0,
                })
                continue
            left_matrix=np.vstack(left_parts)
        if need_right:
            right_parts=[
                get_block_matrix(hic, target, block, args.res, gap_mask)
                for block in right_blocks
            ]
            right_parts=[p for p in right_parts if p.size]
            if not right_parts:
                scale_records.append({
                    "fraction":float(frac),
                    "valid":0,
                    "left_vote":0,
                    "right_vote":0,
                })
                continue
            right_matrix=np.hstack(right_parts)


        if need_left:
            left_rho,left_p,_=calculate_direction(
                left_matrix, "column", args.smooth
            )
            left_vote=int(left_rho>args.rho_cutoff)
        else:
            left_rho,left_p,left_vote=0.0,1.0,0
        if need_right:
            right_rho,right_p,_=calculate_direction(
                right_matrix, "row", args.smooth
            )
            right_vote=int(right_rho< -args.rho_cutoff)
        else:
            right_rho,right_p,right_vote=0.0,1.0,0

        # Same GCOR matrices: per-side diag/anti; either side may pass.
        band=int(args.corner_band_bins)

        def _side_enr(diag, anti):
            if np.isfinite(diag) and np.isfinite(anti) and float(anti)>0:
                return float(diag)/float(anti)
            if np.isfinite(diag) and float(diag)>0 and (
                not np.isfinite(anti) or float(anti)==0
            ):
                return float("inf")
            return float("nan")

        def _diag_bg(diag, bg):
            if np.isfinite(diag) and np.isfinite(bg) and float(bg)>0:
                return float(diag)/float(bg)
            if np.isfinite(diag) and float(diag)>0 and (
                not np.isfinite(bg) or float(bg)==0
            ):
                return float("inf")
            return float("nan")

        if need_left:
            left_diag,left_anti=rectangle_diag_antidiag_means(
                left_matrix, band
            )
            left_bg=rectangle_offband_mean(left_matrix, band)
            left_enr=_side_enr(left_diag, left_anti)
            left_diag_bg=_diag_bg(left_diag, left_bg)
        else:
            left_diag=left_anti=left_enr=left_diag_bg=float("nan")
        if need_right:
            right_diag,right_anti=rectangle_diag_antidiag_means(
                right_matrix, band
            )
            right_bg=rectangle_offband_mean(right_matrix, band)
            right_enr=_side_enr(right_diag, right_anti)
            right_diag_bg=_diag_bg(right_diag, right_bg)
        else:
            right_diag=right_anti=right_enr=right_diag_bg=float("nan")
        side_vals=[
            v for v in (left_enr, right_enr)
            if np.isfinite(v) or v==float("inf")
        ]
        scale_corner=float(max(side_vals)) if side_vals else float("nan")


        scale_records.append({
            "fraction":float(frac),
            "valid":1,
            "left_vote":left_vote,
            "right_vote":right_vote,
            "left_rho":float(left_rho),
            "right_rho":float(right_rho),
            "left_p":float(left_p),
            "right_p":float(right_p),
            "left_diag":float(left_diag) if np.isfinite(left_diag) else None,
            "left_anti":float(left_anti) if np.isfinite(left_anti) else None,
            "right_diag":float(right_diag) if np.isfinite(right_diag) else None,
            "right_anti":float(right_anti) if np.isfinite(right_anti) else None,
            "left_corner":float(left_enr) if np.isfinite(left_enr) or left_enr==float("inf") else None,
            "right_corner":float(right_enr) if np.isfinite(right_enr) or right_enr==float("inf") else None,
            "left_diag_bg":float(left_diag_bg) if np.isfinite(left_diag_bg) or left_diag_bg==float("inf") else None,
            "right_diag_bg":float(right_diag_bg) if np.isfinite(right_diag_bg) or right_diag_bg==float("inf") else None,
            "corner_enrichment":scale_corner,
            "left_start":left_blocks[0][0] if need_left else target[0],
            "left_end":left_blocks[-1][1] if need_left else target[0],
            "right_start":right_blocks[0][0] if need_right else target[1],
            "right_end":right_blocks[-1][1] if need_right else target[1],
            "left_blocks":len(left_blocks) if need_left else 0,
            "right_blocks":len(right_blocks) if need_right else 0,
        })


        if need_left:
            left_rhos.append(float(left_rho))
            left_ps.append(float(left_p))
            left_votes.append(left_vote)
            left_starts.append(left_blocks[0][0])
            left_ends.append(left_blocks[-1][1])
        if need_right:
            right_rhos.append(float(right_rho))
            right_ps.append(float(right_p))
            right_votes.append(right_vote)
            right_starts.append(right_blocks[0][0])
            right_ends.append(right_blocks[-1][1])


    valid_scales=sum(1 for r in scale_records if r["valid"]==1)
    left_support=sum(left_votes)
    right_support=sum(right_votes)


    if valid_scales==0:
        return None


    # Both flanks must pass (left rho>0 and right rho<0 majorities).
    if valid_scales>=min_support:
        left_vote_final=int(left_support>=min_support)
        right_vote_final=int(right_support>=min_support)
    else:
        left_vote_final=int(left_support>=1)
        right_vote_final=int(right_support>=1)
    both_gcor=int(left_vote_final==1 and right_vote_final==1)
    oneside_gcor=int(left_vote_final==1 or right_vote_final==1)
    detection_side=(
        "both" if both_gcor else
        ("left" if left_vote_final else ("right" if right_vote_final else "none"))
    )


    mean_left=float(np.mean(left_rhos)) if left_rhos else 0.0
    mean_right=float(np.mean(right_rhos)) if right_rhos else 0.0
    mean_left_p=float(np.mean(left_ps)) if left_ps else 1.0
    mean_right_p=float(np.mean(right_ps)) if right_ps else 1.0


    # Widest valid flank for reporting genomic context.
    left_start=min(left_starts) if left_starts else target[0]
    left_end=max(left_ends) if left_ends else target[0]
    right_start=min(right_starts) if right_starts else target[1]
    right_end=max(right_ends) if right_ends else target[1]


    # Aggregate per-side diag/anti; pass if either side meets corner ratio
    # and stays under the upper cap (hyper-bright X/tick FPs).
    # Optional diag/background (default off) kept as a diagnostic gate.
    thr=float(args.corner_enrichment_min)
    thr_max=float(getattr(args, "corner_enrichment_max", 5.0))
    bg_thr=float(getattr(args, "corner_diag_bg_min", 0.0))

    def _finite_mean(vals):
        vals=[float(v) for v in vals if v is not None and np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    left_corner=_finite_mean(
        r.get("left_corner") for r in scale_records if r.get("valid")==1
    )
    right_corner=_finite_mean(
        r.get("right_corner") for r in scale_records if r.get("valid")==1
    )
    left_diag_bg=_finite_mean(
        r.get("left_diag_bg") for r in scale_records if r.get("valid")==1
    )
    right_diag_bg=_finite_mean(
        r.get("right_diag_bg") for r in scale_records if r.get("valid")==1
    )

    def _side_corner_ok(corner, diag_bg):
        if not (np.isfinite(corner) and corner>=thr):
            return False
        if thr_max>0 and corner>thr_max:
            return False
        if bg_thr>0:
            return np.isfinite(diag_bg) and diag_bg>=bg_thr
        return True

    corner_pass=int(
        _side_corner_ok(left_corner, left_diag_bg)
        or _side_corner_ok(right_corner, right_diag_bg)
    )
    side_scores=[
        v for v in (left_corner, right_corner) if np.isfinite(v)
    ]
    corner_score=float(max(side_scores)) if side_scores else float("nan")
    # Hyper-bright on either side / any flank fraction → reject.
    # Mean aggregation can dilute a loud fraction below thr_max; OR-of-min
    # can also pass via the quieter side alone.
    peak_corners=[]
    for r in scale_records:
        if r.get("valid")!=1:
            continue
        for key in ("left_corner","right_corner"):
            v=r.get(key)
            if v is None:
                continue
            if np.isfinite(v) or v==float("inf"):
                peak_corners.append(float(v))
    peak_corner=(
        float(max(peak_corners)) if peak_corners else float("nan")
    )
    if (
        thr_max>0
        and np.isfinite(peak_corner)
        and float(peak_corner)>float(thr_max)
    ):
        corner_pass=0

    # S-internal filters on raw S (do not drop GAP here).
    # Lattice / parallel / fill use ≤50kb when available (coarse res hides stripes).
    s_mat=s_matrix_for_qc(hic,target,args)
    s_raw=s_mat
    lattice_max=float(getattr(args, "lattice_peak_max", 0.30))
    # Tip oneside INVs are often one solid domain (chr3D 0-7 Mb); the
    # lattice gate is tuned for internal checkerboards and is slightly
    # harsh on those tip blocks (0.312 vs 0.30).
    if edge_oneside is not None and lattice_max>0:
        lattice_max=max(lattice_max, 0.35)
    lattice_peak=s_matrix_lattice_peak_fraction(s_mat)
    lattice_pass=1
    if lattice_max>0 and np.isfinite(lattice_peak):
        lattice_pass=int(lattice_peak<=lattice_max)

    parallel_max=float(getattr(args, "parallel_diag_max", 0.30))
    parallel_diag=s_matrix_parallel_diag_ratio(s_mat)
    parallel_pass=1
    if parallel_max>0 and np.isfinite(parallel_diag):
        parallel_pass=int(parallel_diag<=parallel_max)

    fill_max=float(getattr(args, "offdiag_fill_max", 0.25))
    offdiag_fill=s_matrix_offdiag_fill_ratio(s_mat)
    fill_pass=1
    if fill_max>0 and np.isfinite(offdiag_fill):
        fill_pass=int(offdiag_fill<=fill_max)

    occ_min=float(getattr(args, "min_s_occupancy", 0.60))
    gap_max=float(getattr(args, "max_s_gap_frac", 0.25))
    s_occupancy=s_matrix_occupancy(s_mat)
    s_gap_frac=s_matrix_gap_frac(s_raw)
    # Coverage-gap bins inside S (diag mask); complements contact white-band frac.
    s_gap_bin_frac=float("nan")
    gm=getattr(args,"gap_mask",None)
    if gm is not None:
        gi0=int(target[0])//int(args.res)
        gi1=int(target[1])//int(args.res)
        greg=np.asarray(gm[gi0:gi1],dtype=bool)
        if greg.size:
            s_gap_bin_frac=float(greg.mean())
    signal_pass=1
    if occ_min>0 and np.isfinite(s_occupancy):
        signal_pass=int(signal_pass and s_occupancy>=occ_min)
    if gap_max>=0 and np.isfinite(s_gap_frac):
        signal_pass=int(signal_pass and s_gap_frac<=gap_max)

    # Gap rescue (not oneside INV): one flank ok + GAP in S → mask GAP,
    # re-test only the failing flank; accept iff that flank then passes too.
    # Skip on chrom-end oneside: the missing flank cannot be rescued.
    gap_min=float(getattr(args, "gap_oneside_min_frac", 0.10))
    gap_score=s_gap_frac if np.isfinite(s_gap_frac) else float("nan")
    if np.isfinite(s_gap_bin_frac):
        gap_score=(
            max(gap_score,s_gap_bin_frac)
            if np.isfinite(gap_score) else s_gap_bin_frac
        )
    gap_rescue=0
    gap_oneside=0  # legacy column; stays 0 (no oneside pass-through)
    if (
        edge_oneside is None
        and both_gcor==0
        and oneside_gcor==1
        and gap_min>0
        and np.isfinite(gap_score)
        and gap_score>=gap_min
        and gm is not None
    ):
        fail_left=left_vote_final==0
        rescue_votes=[]
        rescue_rhos=[]
        rescue_ps=[]
        for frac in fractions:
            flank_bp=max(1,int(round(target_size*float(frac))))
            left_blocks=flank_blocks_by_length(blocks,index,"left",flank_bp)
            right_blocks=flank_blocks_by_length(blocks,index,"right",flank_bp)
            if not left_blocks or not right_blocks:
                continue
            if not (
                _blocks_abut_or_gap_only(
                    left_blocks[-1][1],target[0],gm,args.res,max_gap_bp
                )
                and _blocks_abut_or_gap_only(
                    target[1],right_blocks[0][0],gm,args.res,max_gap_bp
                )
            ):
                continue
            left_span=left_blocks[-1][1]-left_blocks[0][0]
            right_span=right_blocks[-1][1]-right_blocks[0][0]
            if left_span<flank_bp or right_span<flank_bp:
                continue
            if fail_left:
                parts=[
                    get_block_matrix(hic,block,target,args.res,gm)
                    for block in left_blocks
                ]
                parts=[p for p in parts if p.size]
                if not parts:
                    continue
                mat=np.vstack(parts)
                rho,p,_=calculate_direction(mat,"column",args.smooth)
                rescue_votes.append(int(rho>args.rho_cutoff))
            else:
                parts=[
                    get_block_matrix(hic,target,block,args.res,gm)
                    for block in right_blocks
                ]
                parts=[p for p in parts if p.size]
                if not parts:
                    continue
                mat=np.hstack(parts)
                rho,p,_=calculate_direction(mat,"row",args.smooth)
                rescue_votes.append(int(rho<-args.rho_cutoff))
            rescue_rhos.append(float(rho))
            rescue_ps.append(float(p))
        if rescue_votes:
            n_rescue=len(rescue_votes)
            if n_rescue>=min_support:
                rescued=int(sum(rescue_votes)>=min_support)
            else:
                rescued=int(sum(rescue_votes)>=1)
            if rescued:
                gap_rescue=1
                if fail_left:
                    left_vote_final=1
                    mean_left=float(np.mean(rescue_rhos))
                    mean_left_p=float(np.mean(rescue_ps))
                else:
                    right_vote_final=1
                    mean_right=float(np.mean(rescue_rhos))
                    mean_right_p=float(np.mean(rescue_ps))
                both_gcor=1
                detection_side="both"

    # Both flanks always OK. Oneside: long = pure oneside; short = oneside
    # only if the weak flank still has the INV-correct sign.
    # Chrom-end edge_oneside: no opposite flank → pure oneside on the arm.
    oneside_min_size=int(getattr(args,"oneside_min_size",8_000_000))
    if edge_oneside=="left":
        oneside_ok=int(left_vote_final==1)
        gcor_inv=oneside_ok
        detection_side="left" if oneside_ok else "none"
    elif edge_oneside=="right":
        oneside_ok=int(right_vote_final==1)
        gcor_inv=oneside_ok
        detection_side="right" if oneside_ok else "none"
    else:
        oneside_ok=gcor_oneside_accept(
            oneside_gcor=oneside_gcor,
            left_vote=left_vote_final,
            right_vote=right_vote_final,
            mean_left=mean_left,
            mean_right=mean_right,
            target_size=target_size,
            oneside_min_size=oneside_min_size,
        )
        gcor_inv=int(both_gcor==1 or oneside_ok==1)
    inv=int(
        gcor_inv==1 and corner_pass==1 and lattice_pass==1
        and parallel_pass==1 and fill_pass==1 and signal_pass==1
    )

    def _avg_field(name):
        vals=[
            r[name] for r in scale_records
            if r.get("valid")==1 and r.get(name) is not None
        ]
        return float(np.mean(vals)) if vals else "NA"


    return {
        "start":target[0],
        "end":target[1],
        "size":target_size,
        "left_start":left_start,
        "left_end":left_end,
        "right_start":right_start,
        "right_end":right_end,
        "left_rho":mean_left,
        "left_p":mean_left_p,
        "right_rho":mean_right,
        "right_p":mean_right_p,
        "left_vote":left_vote_final,
        "right_vote":right_vote_final,
        "INV":inv,
        "gcor_inv":gcor_inv,
        "corner_enrichment":corner_score if np.isfinite(corner_score) else "NA",
        "corner_left":left_corner if np.isfinite(left_corner) else "NA",
        "corner_right":right_corner if np.isfinite(right_corner) else "NA",
        "corner_left_diag_bg":left_diag_bg if np.isfinite(left_diag_bg) else "NA",
        "corner_right_diag_bg":right_diag_bg if np.isfinite(right_diag_bg) else "NA",
        "corner_pass":corner_pass,
        "lattice_peak":lattice_peak if np.isfinite(lattice_peak) else "NA",
        "lattice_pass":lattice_pass,
        "parallel_diag":parallel_diag if np.isfinite(parallel_diag) else "NA",
        "parallel_pass":parallel_pass,
        "offdiag_fill":offdiag_fill if np.isfinite(offdiag_fill) else "NA",
        "fill_pass":fill_pass,
        "s_occupancy":s_occupancy if np.isfinite(s_occupancy) else "NA",
        "s_gap_frac":s_gap_frac if np.isfinite(s_gap_frac) else "NA",
        "s_gap_bin_frac":s_gap_bin_frac if np.isfinite(s_gap_bin_frac) else "NA",
        "signal_pass":signal_pass,
        "gap_oneside":gap_oneside,
        "gap_rescue":gap_rescue,
        "oneside_ok":oneside_ok,
        "corner_left_diag":_avg_field("left_diag"),
        "corner_left_anti":_avg_field("left_anti"),
        "corner_right_diag":_avg_field("right_diag"),
        "corner_right_anti":_avg_field("right_anti"),
        "detection_side":detection_side,
        "neighbor_blocks":",".join(
            str(r.get("left_blocks",0)+r.get("right_blocks",0))
            for r in scale_records if r["valid"]==1
        ),
        "left_blocks":",".join(
            str(r.get("left_blocks",0))
            for r in scale_records if r["valid"]==1
        ),
        "right_blocks":",".join(
            str(r.get("right_blocks",0))
            for r in scale_records if r["valid"]==1
        ),
        "detection_strength":max(mean_left, -mean_right),
        "neighbor_scale_fractions":",".join(
            str(float(f)) for f in fractions
        ),
        "scale_left_votes":",".join(
            str(r["left_vote"]) if r["valid"]==1 else "NA"
            for r in scale_records
        ),
        "scale_right_votes":",".join(
            str(r["right_vote"]) if r["valid"]==1 else "NA"
            for r in scale_records
        ),
        "left_support":left_support,
        "right_support":right_support,
        "valid_scales":valid_scales,
    }




############################################################
# chromosome processing
############################################################


def process_one_chrom(
    chrom,
        args
):


    print(
        "[INFO]",
        chrom
    )
    args.qc_chrom=chrom



    clr=cooler.Cooler(
        f"{args.mcool_file}::resolutions/{args.res}"
    )



    hic=clr.matrix(
        balance=bool(args.balance)
    ).fetch(
        chrom
    )


    hic=np.nan_to_num(
        hic,
        nan=0
    )



    ##################################################
    # blocks: Louvain (finest) or lift from finest
    ##################################################


    lift_dir=getattr(args, "lift_blocks_dir", None)
    block_file=os.path.join(args.outdir, chrom+".blocks.tsv")


    if lift_dir:


        print(
            "[INFO] lifting Louvain blocks from",
            lift_dir
        )


        src=os.path.join(lift_dir, chrom+".blocks.tsv")
        if not os.path.isfile(src):
            print("[WARN] missing fine blocks:", src)
            return


        fine=load_blocks(src, chrom, args.min_block)
        blocks=lift_blocks_to_resolution(fine, args.res)
        write_block_table(blocks, chrom, block_file)


    elif args.block_file is None:


        print(
            "[INFO] generating Louvain blocks"
        )


        gap_frac=float(getattr(args,"gap_cluster_diag_frac",0.35))
        gap_for_cluster=hic_gap_bin_mask(hic, diag_frac=gap_frac)
        print(
            "[INFO]",
            chrom,
            "GAP compressed for Louvain:",
            int(gap_for_cluster.sum()),
            f"bins (diag_frac={gap_frac}); arms across holes can join",
            flush=True
        )


        norm=distance_normalize(
            hic
        )
        labels=louvain_labels_gap_compressed(
            norm,
            args.radius,
            gap_for_cluster,
            args.louvain_resolution,
            args.direction_lambda,
        )


        # Full painted labels → contiguous blocks (no permanent holes).
        generate_block_file(
            labels,
            chrom,
            args.res,
            block_file,
            gap_mask=None
        )


        blocks=load_blocks(
            block_file,
            chrom,
            args.min_block
        )


    else:


        block_file=args.block_file
        blocks=load_blocks(
        block_file,
        chrom,
            args.min_block
    )

    # Same GAP mask as Louvain: abut-across-gap flanks + gap-rescue retest.
    # Do NOT permanently bridge-merge Louvain tiles (that swallowed true INVs).
    gap_frac=float(getattr(args,"gap_cluster_diag_frac",0.35))
    gap_mask=hic_gap_bin_mask(hic, diag_frac=gap_frac)
    args.gap_mask=gap_mask
    args.gcor_max_gap_bp=int(getattr(args,"gcor_max_gap_bp",2_000_000))


    print(
        chrom,
        "blocks:",
        len(blocks),
    )



    results=[]
    split_min=int(getattr(args, "secondary_split_min_size", 3_000_000))
    sec_lr=float(getattr(args, "secondary_louvain_resolution", 2.0))
    max_gap_bp=args.gcor_max_gap_bp


    def _append_inv(rec, secondary=0, gap_bridge=0):
        if not rec or rec.get("INV")!=1:
            return
        rec=dict(rec)
        rec["chrom"]=chrom
        rec["secondary_split"]=int(secondary)
        rec["gap_bridge"]=int(gap_bridge)
        results.append(rec)


    def _secondary_split_and_test(parent_index, edge_oneside=None):
        """
        Re-Louvain a parent tile and keep sub-INVs.

        edge_oneside is set for chrom first/last parents that the main loop
        never targets: internal subs use both flanks; the sub that remains
        on the chrom end uses pure oneside on the available arm.

        Always oneside-test the intact tip parent first (chr3D 0-7 Mb is one
        Louvain community; split alone only recovered a nested 5-6 Mb core).
        """
        if (
            split_min<=0
            or sec_lr<=0
            or parent_index<0
            or parent_index>=len(blocks)
        ):
            return
        parent=blocks[parent_index]
        parent_inv=False
        if edge_oneside and parent[1]-parent[0]>=args.min_block:
            r_parent=detect_block_INV(
                hic, blocks, parent_index, args,
                edge_oneside=edge_oneside,
            )
            if r_parent and r_parent.get("INV")==1:
                _append_inv(r_parent, secondary=0, gap_bridge=0)
                parent_inv=True
        if parent[1]-parent[0]<split_min:
            return
        # Intact tip already called: skip nested split (avoids short subset).
        if parent_inv:
            return
        subs=secondary_louvain_split(
            hic,
            parent[0],
            parent[1],
            args.res,
            args.radius,
            sec_lr,
            args.min_block,
            direction_lambda=args.direction_lambda,
            gap_mask=gap_mask,
            gap_cluster_diag_frac=gap_frac,
        )
        if len(subs)<=1:
            return
        local=blocks[:parent_index]+subs+blocks[parent_index+1:]
        sub_hi=parent_index+len(subs)

        def _append_grown(j, edge):
            r0=detect_block_INV(
                hic, local, j, args, edge_oneside=edge
            )
            if not r0 or r0.get("INV")!=1:
                return
            r1=grow_secondary_inv_rec(
                hic, local, j, r0, args,
                edge_oneside=edge, sub_hi=sub_hi,
            )
            _append_inv(r1, secondary=1, gap_bridge=0)

        for j in range(parent_index, sub_hi):
            if j<=0:
                # First sub of chrom-start parent: right-flank oneside.
                if edge_oneside=="right":
                    _append_grown(j, "right")
                continue
            if j>=len(local)-1:
                # Last sub of chrom-end parent: left-flank oneside.
                if edge_oneside=="left":
                    _append_grown(j, "left")
                continue
            _append_grown(j, None)


    for i in range(
        1,
        len(blocks)-1
    ):


        r=detect_block_INV(
            hic,
            blocks,
            i,
            args
        )


        if not r:
            continue


        r["chrom"]=chrom
        r["secondary_split"]=0
        r["gap_bridge"]=0
        results.append(r)


        # Large failed parent: re-Louvain inside and keep sub-INVs.
        if (
            r.get("INV")==0
            and (blocks[i][1]-blocks[i][0])>=split_min
        ):
            _secondary_split_and_test(i, edge_oneside=None)

        # Failed tile: also try same-S with the next tile if only GAP between.
        if (
            r.get("INV")==0
            and i+1<len(blocks)-1
            and _blocks_abut_or_gap_only(
                blocks[i][1],blocks[i+1][0],gap_mask,args.res,max_gap_bp
            )
            and blocks[i+1][0]>blocks[i][1]
        ):
            bridged=[
                blocks[i][0],
                blocks[i+1][1],
                blocks[i][2],
            ]
            local=blocks[:i]+[bridged]+blocks[i+2:]
            r3=detect_block_INV(hic, local, i, args)
            _append_inv(r3, secondary=0, gap_bridge=1)


    # Chrom ends are skipped by the main loop (no opposite flank). Still
    # secondary-split large tip communities and oneside-test tip subs.
    if len(blocks)>=2:
        _secondary_split_and_test(0, edge_oneside="right")
        _secondary_split_and_test(len(blocks)-1, edge_oneside="left")



    if results:


        out=os.path.join(
            args.outdir,
            chrom+".INV.tsv"
        )


        pd.DataFrame(
            results
        ).to_csv(
            out,
            sep="\t",
            index=False
        )



    print(
        "[DONE]",
        chrom
    )







############################################################
# parallel
############################################################


def run_parallel(args):


    os.makedirs(
        args.outdir,
        exist_ok=True
    )



    clr=cooler.Cooler(
        f"{args.mcool_file}::resolutions/{args.res}"
    )


    chroms=list(
        clr.chromnames
    )



    with Pool(
        args.nprocess
    ) as pool:


        pool.starmap(
            process_one_chrom,
            [
                (
                    c,
                    args
                )
                for c in chroms
            ]
        )







############################################################
# merge result
############################################################


def merge_results(args):
    files=[]
    for f in os.listdir(args.outdir):
        if f.endswith(".INV.tsv"):
            files.append(os.path.join(args.outdir,f))
    if not files:
        print("No inversion")
        return
    dfs=[pd.read_csv(f,sep="\t") for f in files]
    df=pd.concat(dfs,ignore_index=True)
    df.to_csv(
        os.path.join(args.outdir,"ALL_INV.tsv"),
        sep="\t",
        index=False
    )
    print("Merged:",len(df))



############################################################
# Multi-resolution interval fusion
############################################################


def parse_multiscale_inputs(specs):

    """Read RES=FILE/DIR specifications and attach resolution to each call."""

    frames=[]


    for spec in specs:

        if "=" not in spec:
            raise ValueError(
                "multiscale input must have the form RES=FILE_OR_DIRECTORY: "
                +spec
            )


        res_text,path=spec.split("=",1)
        resolution=int(res_text)


        if os.path.isdir(path):
            paths=sorted(
                os.path.join(path,name)
                for name in os.listdir(path)
                if name.endswith(".INV.tsv")
                and name!="ALL_INV.tsv"
            )
        else:
            paths=sorted(glob.glob(path))


        for filename in paths:

            if not os.path.isfile(filename):
                continue


            frame=pd.read_csv(filename,sep="\t")


            if frame.empty:
                continue


            required={"chrom","start","end"}
            missing=required-set(frame.columns)


            if missing:
                raise ValueError(
                    filename+" is missing columns: "+", ".join(sorted(missing))
                )


            frame=frame.copy()


            if "INV" in frame.columns:
                frame=frame[
                    pd.to_numeric(frame["INV"],errors="coerce")==1
                ].copy()


            if frame.empty:
                continue


            frame["resolution"]=resolution
            frame["source_file"]=filename
            frame["source_row"]=np.arange(len(frame))
            frames.append(frame)


    if not frames:
        return pd.DataFrame(
            columns=["chrom","start","end","resolution"]
        )


    calls=pd.concat(frames,ignore_index=True)
    calls["start"]=calls["start"].astype(int)
    calls["end"]=calls["end"].astype(int)
    calls=calls[calls["end"]>calls["start"]].copy()
    calls["call_size"]=calls["end"]-calls["start"]
    return calls.reset_index(drop=True)



def aggregate_matrix(M, max_bins):

    """Downsample a square contact matrix by block means for stable features."""

    M=np.asarray(M,dtype=float)
    n=M.shape[0]


    if n<=max_bins:
        return M


    edges=np.linspace(0,n,max_bins+1,dtype=int)
    out=np.zeros((max_bins,max_bins),dtype=float)


    for i in range(max_bins):
        for j in range(max_bins):
            tile=M[edges[i]:edges[i+1],edges[j]:edges[j+1]]
            out[i,j]=np.nanmean(tile) if tile.size else 0


    return np.nan_to_num(out,nan=0)



def interval_feature(row, cooler_by_resolution, args):

    """Build graph/trend features for one candidate interval at one scale."""

    clr=cooler_by_resolution[int(row.resolution)]
    chrom=str(row.chrom)
    start=int(row.start)
    end=int(row.end)


    try:
        mat=clr.matrix(balance=bool(getattr(args,"balance",False))).fetch((chrom,start,end))
    except Exception:
        return np.zeros(12,dtype=float)


    mat=np.nan_to_num(mat,nan=0)
    mat=aggregate_matrix(mat,args.feature_max_bins)


    norm=distance_normalize(mat)


    if len(norm)>=2:
        # Zero-variance rows are common in sparse Hi-C intervals.  corrcoef
        # returns NaN for them; the following nan_to_num intentionally maps
        # those entries to zero, so suppress only the expected warning.
        with np.errstate(divide="ignore",invalid="ignore"):
            correlation=np.corrcoef(norm)
        correlation=np.nan_to_num(correlation,nan=0,posinf=0,neginf=0)
        row_rho,_,_=calculate_direction(correlation,"row",args.smooth)
        col_rho,_,_=calculate_direction(correlation,"column",args.smooth)
        graph=build_contact_graph(norm,args.radius)
        direction,confidence=calculate_direction_information(norm,args.radius)
        graph=fuse_direction(
            graph,
            direction,
            confidence,
            args.direction_lambda
        )
        labels=run_louvain(graph,args.louvain_resolution)
        n_communities=len(np.unique(labels)) if len(labels) else 0
    else:
        row_rho=0
        col_rho=0
        direction=np.array([],dtype=float)
        confidence=np.array([],dtype=float)
        n_communities=len(norm)


    # Boundary trends retain the inversion orientation that is lost in a
    # symmetric interval matrix.  Use one interval length as context so a
    # coarse-scale call can represent a complete large event.
    chrom_length=int(clr.chromsizes[chrom])
    interval_size=end-start
    left_start=max(0,start-interval_size)
    right_end=min(chrom_length,end+interval_size)
    left_boundary_rho=0
    right_boundary_rho=0


    try:
        if left_start<start:
            left_mat=clr.matrix(balance=bool(getattr(args,"balance",False))).fetch(
                (chrom,left_start,start),
                (chrom,start,end)
            )
            left_boundary_rho,_,_=calculate_direction(
                np.nan_to_num(left_mat,nan=0),
                "column",
                args.smooth
            )


        if end<right_end:
            right_mat=clr.matrix(balance=bool(getattr(args,"balance",False))).fetch(
                (chrom,start,end),
                (chrom,end,right_end)
            )
            right_boundary_rho,_,_=calculate_direction(
                np.nan_to_num(right_mat,nan=0),
                "row",
                args.smooth
            )
    except Exception:
        left_boundary_rho=0
        right_boundary_rho=0


    left_rho=float(row.get("left_rho",0))
    right_rho=float(row.get("right_rho",0))
    target_signal=max(left_rho,-right_rho)


    return np.array([
        float(row_rho),
        float(col_rho),
        float(left_boundary_rho),
        float(right_boundary_rho),
        left_rho,
        right_rho,
        float(np.mean(direction)) if len(direction) else 0,
        float(np.std(direction)) if len(direction) else 0,
        float(np.mean(confidence)) if len(confidence) else 0,
        float(n_communities),
        float(len(mat)),
        float(target_signal)
    ],dtype=float)



def feature_distance(a,b):

    """Finite Euclidean distance after robust per-feature scaling."""

    a=np.asarray(a,dtype=float)
    b=np.asarray(b,dtype=float)
    return float(np.sqrt(np.mean((a-b)**2)))



def trend_compatible(a,b):

    """Require non-conflicting row/column trend directions when both are clear."""

    for k in (0,1,2,3):
        if abs(a[k])>=0.2 and abs(b[k])>=0.2 and np.sign(a[k])!=np.sign(b[k]):
            return False


    return True



def outside_neighbor_signal(
        chrom,
        start,
        end,
        resolution,
        cooler_by_resolution,
        args
):

    """Measure direction at both outside boundaries of a merged interval."""

    clr=cooler_by_resolution[int(resolution)]
    chrom_length=int(clr.chromsizes[chrom])
    interval_size=max(1,int(end-start))
    flank=max(
        int(resolution),
        int(interval_size*args.merge_flank_ratio)
    )
    left_start=max(0,int(start)-flank)
    right_end=min(chrom_length,int(end)+flank)
    left_rho=0.0
    right_rho=0.0


    try:
        if left_start<int(start):
            left_mat=clr.matrix(balance=bool(getattr(args,"balance",False))).fetch(
                (chrom,left_start,int(start)),
                (chrom,int(start),int(end))
            )
            left_rho,_,_=calculate_direction(
                np.nan_to_num(left_mat,nan=0),
                "column",
                args.smooth
            )


        if int(end)<right_end:
            right_mat=clr.matrix(balance=bool(getattr(args,"balance",False))).fetch(
                (chrom,int(start),int(end)),
                (chrom,int(end),right_end)
            )
            right_rho,_,_=calculate_direction(
                np.nan_to_num(right_mat,nan=0),
                "row",
                args.smooth
            )
    except Exception:
        return 0.0,0.0


    return float(left_rho),float(right_rho)



def merge_neighbor_check(
        a,
        b,
        calls,
        cooler_by_resolution,
        args
):

    """
    Re-evaluate the proposed combined interval against outside neighbors.

    This is deliberately separate from pairwise overlap: two adjacent calls
    are merged only if the combined interval still has an abnormal boundary
    trend at one or both outside sides.  Normal outside trends indicate two
    independent neighboring inversions and block the merge.
    """

    chrom=str(a.chrom)
    combined_start=min(int(a.start),int(b.start))
    combined_end=max(int(a.end),int(b.end))
    scales=sorted(set([int(a.resolution),int(b.resolution)]))
    left_support=0
    right_support=0


    for resolution in scales:
        left_rho,right_rho=outside_neighbor_signal(
            chrom,
            combined_start,
            combined_end,
            resolution,
            cooler_by_resolution,
            args
        )


        if left_rho>args.merge_neighbor_cutoff:
            left_support+=1


        if right_rho< -args.merge_neighbor_cutoff:
            right_support+=1


    abnormal_support=max(left_support,right_support)
    return abnormal_support>=args.merge_neighbor_support



def internal_sampling_vote(
        chrom,
        start,
        end,
    resolution,
        cooler_by_resolution,
        args,
        rng
):

    """
    Vote with random internal subregions against the merged outside flanks.

    A vote means that an internal subregion itself still behaves like an
    inversion relative to an outside neighbor.  Many such votes support
    treating the proposed span as one continuous inversion and accepting
    the merge.
    """

    clr=cooler_by_resolution[int(resolution)]
    chrom_length=int(clr.chromsizes[chrom])
    start=int(start)
    end=int(end)
    interval_size=end-start


    if interval_size<3*resolution:
        return 0,0,0


    sample_length=max(
        3*resolution,
        int(interval_size*args.within_scale_sample_ratio)
    )
    sample_length=min(sample_length,interval_size-resolution)
    sample_bins=max(3,int(np.ceil(sample_length/resolution)))
    sample_length=sample_bins*resolution
    available_bins=max(0,(interval_size-sample_length)//resolution)


    if available_bins<1:
        return 0,0,0


    flank=max(
        resolution,
        int(interval_size*args.merge_flank_ratio)
    )
    left_start=max(0,start-flank)
    right_end=min(chrom_length,end+flank)


    if left_start>=start and end>=right_end:
        return 0,0,0


    sample_count=min(
        args.within_scale_samples,
        available_bins+1
    )
    offsets=rng.choice(
        available_bins+1,
        size=sample_count,
        replace=False
    )
    votes=0
    valid=0


    for offset in offsets:
        sample_start=start+int(offset)*resolution
        sample_end=min(end,sample_start+sample_length)
        left_rho=0.0
        right_rho=0.0


        try:
            if left_start<start:
                left_mat=clr.matrix(balance=bool(getattr(args,"balance",False))).fetch(
                    (chrom,left_start,start),
                    (chrom,sample_start,sample_end)
                )
                left_rho,_,_=calculate_direction(
                    np.nan_to_num(left_mat,nan=0),
                    "column",
                    args.smooth
                )


            if end<right_end:
                right_mat=clr.matrix(balance=bool(getattr(args,"balance",False))).fetch(
                    (chrom,sample_start,sample_end),
                    (chrom,end,right_end)
                )
                right_rho,_,_=calculate_direction(
                    np.nan_to_num(right_mat,nan=0),
                    "row",
                    args.smooth
                )
        except Exception:
            continue


        valid+=1


        if (
            left_rho>args.merge_neighbor_cutoff
            or right_rho< -args.merge_neighbor_cutoff
        ):
            votes+=1


    return votes,valid,sample_count



def within_scale_pair_can_merge(
        left,
        right,
        cooler_by_resolution,
        args,
        rng
):

    """Validate one adjacent, same-resolution merge proposal."""

    resolution=int(left.resolution)
    chrom=str(left.chrom)
    start=min(int(left.start),int(right.start))
    end=max(int(left.end),int(right.end))


    left_rho,right_rho=outside_neighbor_signal(
        chrom,
        start,
        end,
        resolution,
        cooler_by_resolution,
        args
    )
    outside_abnormal=(
        left_rho>args.merge_neighbor_cutoff
        or right_rho< -args.merge_neighbor_cutoff
    )


    if not outside_abnormal:
        return False,0,0,0,left_rho,right_rho


    votes,valid,attempted=internal_sampling_vote(
        chrom,
        start,
        end,
        resolution,
        cooler_by_resolution,
        args,
        rng
    )


    accepted=(
        attempted>0
        and valid>=args.within_scale_min_valid
        and votes>=args.within_scale_vote_threshold
    )
    return accepted,votes,valid,attempted,left_rho,right_rho



def merge_call_rows(
        left,
        right,
        votes,
        valid,
        attempted,
        left_rho,
        right_rho,
        args
):

    """Create a synthetic call representing an accepted within-scale merge."""

    merged=left.copy()
    merged["start"]=min(int(left.start),int(right.start))
    merged["end"]=max(int(left.end),int(right.end))
    merged["call_size"]=int(merged["end"]-merged["start"])
    merged["left_rho"]=left_rho
    merged["right_rho"]=right_rho
    merged["left_vote"]=int(left_rho>args.merge_neighbor_cutoff)
    merged["right_vote"]=int(right_rho< -args.merge_neighbor_cutoff)
    merged["INV"]=1
    merged["detection_side"]=(
        "both" if (
            left_rho>args.merge_neighbor_cutoff
            and right_rho< -args.merge_neighbor_cutoff
        )
        else "left" if left_rho>args.merge_neighbor_cutoff
        else "right"
    )
    merged["within_scale_merged_calls"]=(
        int(left.get("within_scale_merged_calls",1))
        +int(right.get("within_scale_merged_calls",1))
    )
    merged["within_scale_sample_votes"]=int(votes)
    merged["within_scale_valid_samples"]=int(valid)
    merged["within_scale_attempted_samples"]=int(attempted)
    merged["source_file"]=(
        str(left.get("source_file",""))
        +";"
        +str(right.get("source_file",""))
    )
    return merged



def merge_within_each_scale(calls,cooler_by_resolution,args):

    """Iteratively merge adjacent calls independently at each resolution."""

    rng=np.random.default_rng(args.random_seed)
    output=[]
    audit=[]


    for (chrom,resolution),group in calls.groupby(
        ["chrom","resolution"],
        sort=False
    ):
        rows=[row.copy() for _,row in group.sort_values("start").iterrows()]


        for row in rows:
            row["within_scale_merged_calls"]=int(
                row.get("within_scale_merged_calls",1)
            )
            row["within_scale_sample_votes"]=int(
                row.get("within_scale_sample_votes",0)
            )
            row["within_scale_valid_samples"]=int(
                row.get("within_scale_valid_samples",0)
            )
            row["within_scale_attempted_samples"]=int(
                row.get("within_scale_attempted_samples",0)
            )


        changed=True


        while changed and len(rows)>1:
            changed=False
            next_rows=[]
            i=0


            while i<len(rows):
                if i==len(rows)-1:
                    next_rows.append(rows[i])
                    i+=1
                    continue


                left=rows[i]
                right=rows[i+1]
                gap=max(0,int(right.start)-int(left.end))
                # ponytail: fixed 100Mb gap ceiling (was max of the two call lengths)
                max_gap=100_000_000


                if gap>max_gap:
                    next_rows.append(left)
                    i+=1
                    continue


                accepted,votes,valid,attempted,left_rho,right_rho=(
                    within_scale_pair_can_merge(
                        left,
                        right,
                        cooler_by_resolution,
                        args,
                        rng
                    )
                )


                audit.append({
                    "chrom":chrom,
                    "resolution":int(resolution),
                    "left_start":int(left.start),
                    "left_end":int(left.end),
                    "right_start":int(right.start),
                    "right_end":int(right.end),
                    "gap":int(gap),
                    "max_inversion_length":int(max_gap),
                    "merged_start":min(int(left.start),int(right.start)),
                    "merged_end":max(int(left.end),int(right.end)),
                    "merged_left_rho":float(left_rho),
                    "merged_right_rho":float(right_rho),
                    "sample_votes":int(votes),
                    "valid_samples":int(valid),
                    "attempted_samples":int(attempted),
                    "vote_threshold":int(
                        args.within_scale_vote_threshold
                    ),
                    "accepted":int(accepted)
                })


                if accepted:
                    next_rows.append(
                        merge_call_rows(
                            left,
                            right,
                            votes,
                            valid,
                            attempted,
                            left_rho,
                            right_rho,
                            args
                        )
                    )
                    i+=2
                    changed=True
                else:
                    next_rows.append(left)
                    i+=1


            rows=sorted(next_rows,key=lambda row:int(row.start))


        output.extend(rows)


    if not output:
        return calls.iloc[0:0].copy(),pd.DataFrame(audit)


    return (
        pd.DataFrame(output).reset_index(drop=True),
        pd.DataFrame(audit)
    )



def component_neighbor_check(
        nodes,
        calls,
        cooler_by_resolution,
        args
):

    """Check outside trends for the envelope of an entire merged component."""

    if len(nodes)<=1:
        return True


    group=calls.loc[list(nodes)]
    chrom=str(group["chrom"].iloc[0])
    start=int(group["start"].min())
    end=int(group["end"].max())
    left_support=0
    right_support=0


    for resolution in sorted(group["resolution"].unique()):
        left_rho,right_rho=outside_neighbor_signal(
            chrom,
            start,
            end,
            int(resolution),
            cooler_by_resolution,
            args
        )


        if left_rho>args.merge_neighbor_cutoff:
            left_support+=1


        if right_rho< -args.merge_neighbor_cutoff:
            right_support+=1


    return max(left_support,right_support)>=args.merge_neighbor_support



def refine_components(
        components,
        graph,
    calls,
        cooler_by_resolution,
        args
):

    """Split transitive chains whose final envelope has normal flanks."""

    refined=[]


    for component in components:
        pending=[set(component)]


        while pending:
            nodes=pending.pop()


            if len(nodes)<=1 or (
                component_has_coarse_bridge(nodes,calls)
                and
                component_neighbor_check(
                    nodes,
                    calls,
                    cooler_by_resolution,
                    args
                )
            ):
                refined.append(nodes)
                continue


            internal_edges=[
                (u,v,data)
                for u,v,data in graph.edges(nodes,data=True)
                if u in nodes and v in nodes
            ]


            if not internal_edges:
                refined.append(nodes)
                continue


            # Remove the weakest link first.  This breaks A-B-C chains at the
            # least-supported boundary instead of discarding the whole event.
            weakest=min(
                internal_edges,
                key=lambda edge: (
                    edge[2].get("overlap",0),
                    -edge[2].get("feature_distance",np.inf)
                )
            )
            graph.remove_edge(weakest[0],weakest[1])
            split=list(nx.connected_components(graph.subgraph(nodes)))


            if len(split)==1:
                refined.append(nodes)
            else:
                pending.extend(split)


    return refined



def overlap_fraction(a_start,a_end,b_start,b_end):

    overlap=max(0,min(a_end,b_end)-max(a_start,b_start))
    shorter=max(1,min(a_end-a_start,b_end-b_start))
    return overlap/shorter


def dedup_overlapping_inv_calls(calls, min_ios=0.5):
    """
    Final same-chrom collapse: merge calls with IoS >= min_ios into one span.
    Keeps the best scale_support / event_score row and unions coordinates.
    """
    if calls is None or len(calls)==0:
        return calls
    df=calls.sort_values(["chrom","start","end"]).reset_index(drop=True)
    used=set()
    rows=[]
    for i in range(len(df)):
        if i in used:
            continue
        chrom=str(df.at[i,"chrom"])
        members=[i]
        u0=int(df.at[i,"start"]); u1=int(df.at[i,"end"])
        for j in range(i+1,len(df)):
            if j in used:
                continue
            if str(df.at[j,"chrom"])!=chrom:
                break
            s=int(df.at[j,"start"]); e=int(df.at[j,"end"])
            if s>u1:
                break
            if overlap_fraction(u0,u1,s,e)>=float(min_ios):
                members.append(j)
                u0=min(u0,s); u1=max(u1,e)
        used.update(members)
        group=df.loc[members].copy()
        # prefer more scale support, then higher event/max signal, then larger
        for col in ("scale_support","event_score","max_signal","size"):
            if col in group.columns:
                group["_rank"]=pd.to_numeric(group[col],errors="coerce").fillna(0)
                group=group.sort_values("_rank",ascending=False)
                break
        best=group.iloc[0].copy()
        best["start"]=u0
        best["end"]=u1
        best["size"]=u1-u0
        if "full_start" in best.index:
            best["full_start"]=u0
            best["full_end"]=u1
            best["full_size"]=u1-u0
        if "core_start" in best.index and "core_end" in best.index:
            best["core_start"]=int(group["core_start"].min())
            best["core_end"]=int(group["core_end"].max())
            best["core_size"]=int(best["core_end"]-best["core_start"])
        if "resolutions" in group.columns:
            res_parts=[]
            for r in group["resolutions"].astype(str):
                res_parts.extend([x for x in r.split(",") if x and x!="nan"])
            # unique preserve order by numeric
            seen=set(); ordered=[]
            for x in sorted(res_parts, key=lambda z: int(float(z))):
                if x not in seen:
                    seen.add(x); ordered.append(x)
            best["resolutions"]=",".join(ordered)
            best["scale_support"]=len(ordered)
            best["call_support"]=len(ordered)
        if "arm_expanded" in group.columns:
            best["arm_expanded"]=int(pd.to_numeric(
                group["arm_expanded"],errors="coerce"
            ).fillna(0).max())
        rows.append(best)
    out=pd.DataFrame(rows).reset_index(drop=True)
    if "cluster_id" in out.columns:
        out["cluster_id"]=np.arange(1,len(out)+1)
    return out


def containment_fraction(a_start,a_end,b_start,b_end):

    """Fraction of the shorter interval covered by the longer interval."""

    overlap=max(0,min(a_end,b_end)-max(a_start,b_start))
    shorter=max(1,min(a_end-a_start,b_end-b_start))
    return overlap/shorter



def has_coarse_bridge(i,j,calls):

    """Return True if a coarser-scale call spans both candidate intervals."""

    a=calls.loc[i]
    b=calls.loc[j]
    union_start=min(int(a.start),int(b.start))
    union_end=max(int(a.end),int(b.end))
    max_resolution=max(int(a.resolution),int(b.resolution))


    for k,row in calls.iterrows():

        if k==i or k==j or str(row.chrom)!=str(a.chrom):
            continue


        if int(row.resolution)<=max_resolution:
            continue


        if int(row.start)<=union_start and int(row.end)>=union_end:
            return True


    return False



def component_has_coarse_bridge(nodes,calls):

    """Require a coarse-scale envelope for a multi-call component."""

    if len(nodes)<=1:
        return True


    group=calls.loc[list(nodes)]


    # Same-resolution calls are also valid merge candidates.  They are not
    # required to have a coarser bridge; the component-level outside-neighbor
    # trend check remains the safeguard against merging two independent SVs.
    if group["resolution"].nunique()==1:
        return True


    union_start=int(group["start"].min())
    union_end=int(group["end"].max())
    min_resolution=int(group["resolution"].min())


    # At least one call inside the component must represent its full span at a
    # resolution coarser than the finest contributing scale.
    for _,row in group.iterrows():

        if int(row.resolution)<=min_resolution:
            continue


        if int(row.start)<=union_start and int(row.end)>=union_end:
            return True


    return False



def merge_multiscale_results(args):

    calls=parse_multiscale_inputs(args.multiscale_inputs)


    if calls.empty:
        print("No inversion calls supplied for multiscale merge")
        return


    outdir=args.merge_outdir or args.outdir
    os.makedirs(outdir,exist_ok=True)


    resolutions=sorted(calls["resolution"].unique())
    cooler_by_resolution={
        int(res):cooler.Cooler(
            f"{args.mcool_file}::resolutions/{int(res)}"
        )
        for res in resolutions
    }


    print("[INFO] within-resolution iterative merge and sampling vote")
    calls,within_scale_audit=merge_within_each_scale(
        calls,
        cooler_by_resolution,
        args
    )


    within_scale_audit.to_csv(
        os.path.join(outdir,"WITHIN_SCALE_MERGE_AUDIT.tsv"),
        sep="\t",
        index=False
    )


    calls.to_csv(
        os.path.join(outdir,"WITHIN_SCALE_MERGED_CALLS.tsv"),
        sep="\t",
        index=False
    )


    if calls.empty:
        print("No calls remain after within-resolution merge")
        return


    print("[INFO] extracting interval graph/trend features")
    features=np.vstack([
        interval_feature(row,cooler_by_resolution,args)
        for _,row in calls.iterrows()
    ])


    feature_names=[
        "corr_row_rho",
        "corr_column_rho",
        "boundary_left_rho",
        "boundary_right_rho",
        "source_left_rho",
        "source_right_rho",
        "direction_mean",
        "direction_sd",
        "confidence_mean",
        "community_count",
        "matrix_bins",
        "source_signal"
    ]


    for k,name in enumerate(feature_names):
        calls[name]=features[:,k]


    # Robust scaling keeps graph/community counts from dominating correlations.
    median=np.nanmedian(features,axis=0)
    q1=np.nanpercentile(features,25,axis=0)
    q3=np.nanpercentile(features,75,axis=0)
    scale=np.where((q3-q1)>1e-8,q3-q1,1.0)
    scaled=np.nan_to_num((features-median)/scale,nan=0)


    G=nx.Graph()
    G.add_nodes_from(range(len(calls)))


    # Build a candidate graph and repeatedly merge components.  After each
    # round the component envelope becomes the new interval, allowing several
    # internal fragments of one large SV to grow into its full span.
    order=calls.sort_values(["chrom","start"]).index.tolist()


    for pos,i in enumerate(order):

        a=calls.loc[i]


        for j in order[pos+1:]:

            b=calls.loc[j]


            if str(b.chrom)!=str(a.chrom):
                break


            # Same-resolution decisions have already been made by the
            # iterative sampling-vote stage.  The graph stage is strictly for
            # fusing calls across resolutions.
            if int(a.resolution)==int(b.resolution):
                continue


            length_a=int(a.call_size)
            length_b=int(b.call_size)
            max_inversion_length=max(length_a,length_b)
            allowed_gap=max(
                args.merge_gap,
                int(args.merge_gap_ratio*max_inversion_length),
                max_inversion_length
            )


            if int(b.start)>int(a.end)+allowed_gap:
                continue


            overlap=overlap_fraction(
                int(a.start),int(a.end),int(b.start),int(b.end)
            )
            gap=max(0,max(int(a.start),int(b.start))-min(int(a.end),int(b.end)))
            distance=feature_distance(scaled[i],scaled[j])


            contained=containment_fraction(
                int(a.start),int(a.end),int(b.start),int(b.end)
            )
            coarse_bridge=has_coarse_bridge(i,j,calls)


            # Same-resolution calls are allowed to merge.  For calls from
            # different resolutions, a coarse bridge is required whenever the
            # relation is a partial overlap or a distance-based adjacency.
            # This prevents two neighboring real inversions from being
            # connected by a chain of broad fine-scale calls.
            coordinate_match=(
                (
                    contained>=args.merge_containment
                    or overlap>=args.merge_overlap
                    or (gap>0 and gap<=max_inversion_length)
                )
            )


            # Cross-resolution: high overlap/containment is enough; do not
            # demand a coarser bridge (that left near-duplicate 50k/100k calls).
            if int(a.resolution)!=int(b.resolution) and not (
                contained>=args.merge_containment
                or overlap>=args.merge_overlap
                or coarse_bridge
            ):
                coordinate_match=False
            feature_match=(
                distance<=args.feature_distance
                and trend_compatible(features[i],features[j])
            )
            neighbor_match=merge_neighbor_check(
                a,
                b,
                calls,
                cooler_by_resolution,
                args
            )


            # Crucial safeguard: overlap alone is not sufficient.  The
            # combined interval must also show an abnormal trend against an
            # outside flank; normal flank trends mean two neighboring SVs.
            if coordinate_match and feature_match and neighbor_match:
                G.add_edge(
                    i,
                    j,
                    feature_distance=distance,
                    outside_neighbor_abnormal=1,
                    overlap=overlap
                )


    # The graph is already transitive through connected components.  This is
    # the iterative growth step: A-B and B-C support one A..C event even when A
    # and C do not directly overlap.
    components=list(nx.connected_components(G))
    components=refine_components(
        components,
        G,
        calls,
        cooler_by_resolution,
        args
    )
    merged=[]
    calls["merge_cluster_id"]=pd.Series(
        [pd.NA]*len(calls),
        dtype="Int64"
    )


    for cid,nodes in enumerate(components,1):

        group=calls.loc[sorted(nodes)].copy()
        support=int(group["resolution"].nunique())


        if support<args.min_scale_support:
            continue


        calls.loc[list(nodes),"merge_cluster_id"]=cid


        strengths=[]
        left_support=int(
            sum(float(r.get("left_rho",0))>args.rho_cutoff
                for _,r in group.iterrows())
        )
        right_support=int(
            sum(float(r.get("right_rho",0))< -args.rho_cutoff
                for _,r in group.iterrows())
        )


        for _,r in group.iterrows():
            strengths.append(
                max(
                    abs(float(r.get("left_rho",0))),
                    abs(float(r.get("right_rho",0)))
                )
            )


        representative=group.iloc[int(np.argmax(strengths))]
        starts=group["start"].astype(int).to_numpy()
        ends=group["end"].astype(int).to_numpy()
        full_start=int(starts.min())
        full_end=int(ends.max())
        core_start=int(np.ceil(np.percentile(starts,75)))
        core_end=int(np.floor(np.percentile(ends,25)))


        if core_end<=core_start:
            core_start=int(max(starts))
            core_end=int(min(ends))


        if core_end<=core_start:
            core_start=int(representative["start"])
            core_end=int(representative["end"])


        event_score=float(
            max(
                max(
                    float(r.get("left_rho",0)),
                    -float(r.get("right_rho",0))
                )
                for _,r in group.iterrows()
            )
        )


        merged.append({
            "cluster_id":cid,
            "chrom":str(group["chrom"].iloc[0]),
            "core_start":core_start,
            "core_end":core_end,
            "core_size":int(core_end-core_start),
            "full_start":full_start,
            "full_end":full_end,
            "full_size":int(full_end-full_start),
            "start":full_start,
            "end":full_end,
            "size":int(full_end-full_start),
            "scale_support":support,
            "call_support":len(group),
            "resolutions":",".join(
                str(x) for x in sorted(group["resolution"].unique())
            ),
            "representative_resolution":int(representative["resolution"]),
            "representative_source":representative["source_file"],
            "max_signal":float(max(strengths)),
            "event_score":event_score,
            "left_scale_support":left_support,
            "right_scale_support":right_support,
            "mean_row_rho":float(np.mean(features[list(nodes),0])),
            "mean_column_rho":float(np.mean(features[list(nodes),1])),
            "mean_boundary_left_rho":float(
                np.mean(features[list(nodes),2])
            ),
            "mean_boundary_right_rho":float(
                np.mean(features[list(nodes),3])
            ),
            "mean_source_left_rho":float(
                np.mean(features[list(nodes),4])
            ),
            "mean_source_right_rho":float(
                np.mean(features[list(nodes),5])
            ),
            "community_count_mean":float(
                np.mean(features[list(nodes),9])
            )
        })


    result=pd.DataFrame(merged)


    if not result.empty:
        result=result.sort_values(["chrom","full_start","full_end"])


    result.to_csv(
        os.path.join(outdir,"ALL_INV_multiscale.tsv"),
        sep="\t",
        index=False
    )


    calls.to_csv(
        os.path.join(outdir,"ALL_INV_multiscale_input_calls.tsv"),
        sep="\t",
        index=False
    )


    print("Multiscale merged:",len(result))







############################################################
# main
############################################################


N_SCALES=4
BASE_TARGET_RESOLUTION=50_000


def available_resolutions(mcool_file):
    prefix="/resolutions/"
    found=[]
    for uri in cooler.fileops.list_coolers(mcool_file):
        if uri.startswith(prefix):
            try:
                found.append(int(uri[len(prefix):]))
            except ValueError:
                pass
    return sorted(set(found))


def resolve_plot_mcool(mcool_file):
    """Prefer a sibling full-genome mcool when detection used a .chr.mcool."""
    path=Path(mcool_file).expanduser().resolve()
    detection_finest=(available_resolutions(str(path)) or [None])[0]
    stem=path.name
    if stem.endswith(".chr.mcool"):
        full_name=stem.replace(".chr.mcool",".mcool")
    else:
        full_name=stem
    candidates=[
        path.with_name(full_name),
        path.parent.parent/"01.xy6_genome"/full_name,
        path.parent/"01.xy6_genome"/full_name,
    ]
    best=str(path)
    best_finest=detection_finest
    for candidate in candidates:
        if not candidate.is_file():
            continue
        resolutions=available_resolutions(str(candidate))
        if not resolutions:
            continue
        if best_finest is None or resolutions[0]<best_finest:
            best,best_finest=str(candidate.resolve()),resolutions[0]
    return best


def select_four_resolutions(available):
    """Select near 50 kb, prefer larger scales, then backfill smaller ones."""
    available=sorted(set(int(value) for value in available))
    if not available:
        return []
    base=min(
        available,
        # If two scales are equally close to 50 kb (for example 40k and
        # 60k), prefer the larger scale.
        key=lambda value:(abs(value-BASE_TARGET_RESOLUTION),-value)
    )
    selected=[base]

    # Prefer the next larger resolutions after the scale nearest 50 kb.
    for value in available:
        if value>base and len(selected)<N_SCALES:
            selected.append(value)

    # If the file has fewer than three larger resolutions, fill the remaining
    # slots with the closest smaller resolutions. Run the final set from small
    # to large.
    if len(selected)<N_SCALES:
        smaller=sorted(
            (value for value in available if value<base),
            reverse=True
        )
        selected.extend(smaller[:N_SCALES-len(selected)])

    return sorted(selected)


def parse_workflow_args():
    parser=argparse.ArgumentParser(
        description=(
            "Automatic four-scale workflow around the unchanged original "
            "Louvain/GCOR inversion algorithm"
        )
    )
    parser.add_argument("mcool_file",help="input .mcool file")
    parser.add_argument(
        "--balance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use ICE-balanced contacts (default: --no-balance, raw counts)"
    )
    parser.add_argument("--min-support",type=int,default=2)
    parser.add_argument("--min-community-size",type=int,default=500_000)
    parser.add_argument("--max-merge-gap",type=int,default=2_000_000)
    parser.add_argument("--max-inversion-size",type=int,default=100_000_000)
    parser.add_argument("--merge-overlap",type=float,default=0.50)
    parser.add_argument(
        "--hic-alignments",
        default=None,
        help="optional BAM/CRAM, .pairs(.gz), or BEDPE"
    )
    parser.add_argument(
        "--fasta",
        default=None,
        help="optional FASTA; enables GAP and four-scale hicFindTADs refinement"
    )
    parser.add_argument("--jobs",type=int,default=min(4,os.cpu_count() or 1))
    parser.add_argument("--outdir",default="INV_results")
    args=parser.parse_args()

    if not 1<=args.min_support<=N_SCALES:
        parser.error("--min-support must be between 1 and 4")
    if args.min_community_size<=0:
        parser.error("--min-community-size must be > 0")
    if args.max_merge_gap<0:
        parser.error("--max-merge-gap must be >= 0")
    if args.max_inversion_size<=0:
        parser.error("--max-inversion-size must be > 0")
    if not 0<args.merge_overlap<=1:
        parser.error("--merge-overlap must be in (0, 1]")
    if args.jobs<1:
        parser.error("--jobs must be >= 1")
    for name in ("hic_alignments","fasta"):
        value=getattr(args,name)
        if value and not Path(value).expanduser().is_file():
            parser.error("file does not exist: "+value)
    return args


def run_internal_command(command,label):
    print("[START]",label,flush=True)
    environment=os.environ.copy()
    environment["INV_ORIGINAL_INTERNAL"]="1"
    subprocess.run(command,check=True,env=environment)
    print("[DONE]",label,flush=True)


def original_detection_command(
    script,args,resolution,scale_dir,workers,lift_blocks_dir=None
):
    command=[
        sys.executable,str(script),
        "--mcool_file",args.mcool_file,
        "--res",str(resolution),
        "--min_block",str(args.min_community_size),
        "--outdir",str(scale_dir),
        "--nprocess",str(workers),
        "--corner_enrichment_min","0.8",
        "--corner_enrichment_max","5.0",
        "--corner_diag_bg_min","0",
        "--lattice_peak_max","0.30",
        "--parallel_diag_max","0.30",
        "--offdiag_fill_max","0.25",
        "--s_qc_resolution","50000",
        "--min_s_occupancy","0.60",
        "--max_s_gap_frac","0.25",
        "--gap_oneside_min_frac","0.10",
        "--gap_oneside_corner_min","1.20",
        "--oneside_min_size","8000000",
        "--gcor_max_gap_bp","2000000",
        "--gap_cluster_diag_frac","0.35",
        "--secondary_louvain_resolution","2.0",
        "--secondary_split_min_size","3000000",
        ("--balance" if getattr(args,"balance",False) else "--no-balance"),
    ]
    if lift_blocks_dir is not None:
        command.extend(["--lift_blocks_dir", str(lift_blocks_dir)])
    return command


def original_merge_command(script,args,resolutions,scale_dirs,merge_dir):
    command=[
        sys.executable,str(script),
        "--mcool_file",args.mcool_file,
        "--outdir",str(merge_dir),
        "--merge_outdir",str(merge_dir),
        "--merge_overlap",str(args.merge_overlap),
        "--merge_gap",str(args.max_merge_gap),
        "--min_scale_support",str(args.min_support),
        ("--balance" if getattr(args,"balance",False) else "--no-balance"),
        "--multiscale_inputs"
    ]
    command.extend(
        f"{resolution}={directory}"
        for resolution,directory in zip(resolutions,scale_dirs)
    )
    return command


def _open_text(filename):
    if str(filename).endswith(".gz"):
        return gzip.open(filename,"rt")
    return open(filename,encoding="utf-8")


def scan_fasta_gaps(fasta,min_gap=10):
    gaps={}
    chrom=None
    position=0
    gap_start=None
    with _open_text(fasta) as handle:
        for raw_line in handle:
            if raw_line.startswith(">"):
                if chrom is not None and gap_start is not None and position-gap_start>=min_gap:
                    gaps.setdefault(chrom,set()).update((gap_start,position))
                chrom=raw_line[1:].split()[0]
                position=0
                gap_start=None
                continue
            for base in raw_line.strip().upper():
                if base=="N" and gap_start is None:
                    gap_start=position
                elif base!="N" and gap_start is not None:
                    if position-gap_start>=min_gap:
                        gaps.setdefault(chrom,set()).update((gap_start,position))
                    gap_start=None
                position+=1
    if chrom is not None and gap_start is not None and position-gap_start>=min_gap:
        gaps.setdefault(chrom,set()).update((gap_start,position))
    return {chrom:sorted(points) for chrom,points in gaps.items()}


def run_one_hicfindtads(task):
    mcool_file,resolution,tad_dir,processors=task
    prefix=Path(tad_dir)/f"tads_{resolution}"
    subprocess.run([
        "hicFindTADs",
        "-m",f"{mcool_file}::/resolutions/{resolution}",
        "--outPrefix",str(prefix),
        "--numberOfProcessors",str(processors),
        "--minBoundaryDistance",str(resolution),
        "--correctForMultipleTesting","fdr",
        "--thresholdComparisons","0.05"
    ],check=True)
    return resolution,str(prefix)


def detect_tad_boundaries(mcool_file,resolutions,jobs):
    if shutil.which("hicFindTADs") is None:
        raise SystemExit("--fasta requires hicFindTADs from HiCExplorer")
    by_scale={}
    per_run_threads=max(1,jobs//len(resolutions))
    with tempfile.TemporaryDirectory(prefix="inv_tads_") as tad_dir:
        with ThreadPoolExecutor(max_workers=min(len(resolutions),jobs)) as executor:
            outputs=list(executor.map(
                run_one_hicfindtads,
                [(mcool_file,res,tad_dir,per_run_threads) for res in resolutions]
            ))
        for resolution,prefix in outputs:
            beds=[Path(prefix+"_boundaries.bed"),Path(prefix+"_domains.bed")]
            bed=next((path for path in beds if path.is_file()),None)
            if bed is None:
                continue
            points={}
            with open(bed,encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip() or line.startswith(("#","track","browser")):
                        continue
                    fields=line.rstrip().split("\t")
                    if len(fields)<3:
                        continue
                    try:
                        chrom,start,end=fields[0],int(fields[1]),int(fields[2])
                    except ValueError:
                        continue
                    points.setdefault(chrom,[]).append((start+end)//2)
            by_scale[resolution]=points

    consensus={}
    tolerance=resolutions[0]
    chromosomes=set().union(*(set(points) for points in by_scale.values())) if by_scale else set()
    for chrom in chromosomes:
        observations=sorted(
            (point,resolution)
            for resolution,points in by_scale.items()
            for point in points.get(chrom,[])
        )
        groups=[]
        for point,resolution in observations:
            if not groups or point-groups[-1][-1][0]>tolerance:
                groups.append([])
            groups[-1].append((point,resolution))
        consensus[chrom]=sorted(
            int(np.median([point for point,_ in group]))
            for group in groups
            if len({resolution for _,resolution in group})>=2
        )
    return consensus


def load_hic_breakpoints(filename,calls,tolerance):
    """Prefer inversion split reads; use ordinary Hi-C pairs as fallback."""
    windows=[
        (str(row.chrom),int(row.start),int(row.end))
        for _,row in calls.iterrows()
    ]
    support=[
        {"pair_left":[],"pair_right":[],"split_left":[],"split_right":[]}
        for _ in windows
    ]

    def consider(chrom_a,point_a,chrom_b,point_b,evidence="pair"):
        if chrom_a!=chrom_b:
            return
        for i,(chrom,left,right) in enumerate(windows):
            if chrom!=chrom_a:
                continue
            if abs(point_a-left)<=tolerance and abs(point_b-right)<=tolerance:
                support[i][evidence+"_left"].append(point_a)
                support[i][evidence+"_right"].append(point_b)
            elif abs(point_b-left)<=tolerance and abs(point_a-right)<=tolerance:
                support[i][evidence+"_left"].append(point_b)
                support[i][evidence+"_right"].append(point_a)

    suffix=str(filename).lower()
    if suffix.endswith((".bam",".cram")):
        try:
            import pysam
        except ImportError as error:
            raise SystemExit("BAM/CRAM refinement requires pysam") from error
        mode="rc" if suffix.endswith(".cram") else "rb"
        seen_split_reads=set()
        with pysam.AlignmentFile(filename,mode) as alignments:
            for read in alignments.fetch(until_eof=True):
                if (read.is_unmapped or read.is_secondary or read.is_duplicate
                        or read.mapping_quality<20):
                    continue
                chrom=alignments.get_reference_name(read.reference_id)
                if (not read.is_supplementary and read.has_tag("SA")
                        and read.query_name not in seen_split_reads):
                    left_clip=(
                        read.cigartuples[0][1]
                        if read.cigartuples and read.cigartuples[0][0] in (4,5)
                        else 0
                    )
                    right_clip=(
                        read.cigartuples[-1][1]
                        if read.cigartuples and read.cigartuples[-1][0] in (4,5)
                        else 0
                    )
                    primary_left_clip=left_clip>=right_clip
                    primary_point=(
                        int(read.reference_start)
                        if primary_left_clip else int(read.reference_end)
                    )
                    primary_strand="-" if read.is_reverse else "+"
                    for entry in read.get_tag("SA").rstrip(";").split(";"):
                        fields=entry.split(",")
                        if len(fields)<6:
                            continue
                        sa_chrom,sa_pos,sa_strand,sa_cigar,sa_mapq,_=fields[:6]
                        try:
                            if int(sa_mapq)<20 or sa_strand==primary_strand:
                                continue
                            cigar_parts=re.findall(r"(\d+)([MIDNSHP=X])",sa_cigar)
                            if not cigar_parts:
                                continue
                            ref_span=sum(
                                int(length) for length,operation in cigar_parts
                                if operation in "MDN=X"
                            )
                            sa_start=int(sa_pos)-1
                            sa_left_clip_length=(
                                int(cigar_parts[0][0]) if cigar_parts[0][1] in "SH" else 0
                            )
                            sa_right_clip_length=(
                                int(cigar_parts[-1][0]) if cigar_parts[-1][1] in "SH" else 0
                            )
                            if max(left_clip,right_clip)<10 or max(
                                sa_left_clip_length,sa_right_clip_length
                            )<10:
                                continue
                            sa_left_clip=sa_left_clip_length>=sa_right_clip_length
                            sa_point=sa_start if sa_left_clip else sa_start+ref_span
                        except ValueError:
                            continue
                        consider(chrom,primary_point,sa_chrom,sa_point,"split")
                        seen_split_reads.add(read.query_name)
                        break
                if (not read.is_supplementary and not read.mate_is_unmapped
                        and read.is_read1 and read.reference_id==read.next_reference_id
                        and read.mapping_quality>=30):
                    consider(
                        chrom,int(read.reference_start),chrom,
                        int(read.next_reference_start),"pair"
        )
    else:
        with _open_text(filename) as handle:
            for line in handle:
                if not line.strip() or line.startswith("#"):
                    continue
                fields=line.rstrip().split("\t")
                try:
                    if len(fields)>=6 and fields[1].isdigit():
                        consider(fields[0],int(fields[2]),fields[3],int(fields[5]))
                    elif len(fields)>=5:
                        consider(fields[1],int(fields[2]),fields[3],int(fields[4]))
                except ValueError:
                    continue

    def peak(points,minimum):
        if len(points)<minimum:
            return None,len(points)
        bin_size=max(100,min(1_000,tolerance//20))
        counts=Counter(point//bin_size for point in points)
        best_bin,count=counts.most_common(1)[0]
        if count<minimum:
            return None,count
        selected=[point for point in points if point//bin_size==best_bin]
        return int(np.median(selected)),count

    evidence=[]
    for item in support:
        split_left,split_left_n=peak(item["split_left"],2)
        split_right,split_right_n=peak(item["split_right"],2)
        pair_left,pair_left_n=peak(item["pair_left"],3)
        pair_right,pair_right_n=peak(item["pair_right"],3)
        evidence.append({
            "left":split_left if split_left is not None else pair_left,
            "right":split_right if split_right is not None else pair_right,
            "left_source":"hic_split" if split_left is not None else "hic_pairs",
            "right_source":"hic_split" if split_right is not None else "hic_pairs",
            "left_support":split_left_n if split_left is not None else pair_left_n,
            "right_support":split_right_n if split_right is not None else pair_right_n,
            "left_split_support":split_left_n,
            "right_split_support":split_right_n
        })
    return evidence


def refine_final_breakpoints(result,gap_hints,tad_hints,hic_evidence,tolerance):
    if result.empty:
        return result
    result=result.copy()
    result["original_start"]=result["start"].astype(int)
    result["original_end"]=result["end"].astype(int)
    result["left_breakpoint_source"]="multiscale"
    result["right_breakpoint_source"]="multiscale"
    result["left_hic_pair_support"]=0
    result["right_hic_pair_support"]=0
    result["left_hic_split_support"]=0
    result["right_hic_split_support"]=0

    def nearest(points,coordinate):
        if not points:
            return None
        value=min(points,key=lambda point:abs(point-coordinate))
        return value if abs(value-coordinate)<=tolerance else None

    for ordinal,(index,row) in enumerate(result.iterrows()):
        chrom=str(row.chrom)
        old_left,old_right=int(row.start),int(row.end)
        hic_left=hic_right=None
        left_source=right_source="hic_pairs"
        if hic_evidence:
            evidence=hic_evidence[ordinal]
            hic_left,hic_right=evidence["left"],evidence["right"]
            left_source,right_source=evidence["left_source"],evidence["right_source"]
            result.at[index,"left_hic_pair_support"]=evidence["left_support"]
            result.at[index,"right_hic_pair_support"]=evidence["right_support"]
            result.at[index,"left_hic_split_support"]=evidence["left_split_support"]
            result.at[index,"right_hic_split_support"]=evidence["right_split_support"]
        for side,coordinate,hic_point,hic_source in (
            ("left",old_left,hic_left,left_source),
            ("right",old_right,hic_right,right_source)
        ):
            gap=nearest(gap_hints.get(chrom,[]),coordinate)
            primary=[(gap,"gap"),(hic_point,hic_source)]
            primary=[item for item in primary if item[0] is not None]
            if primary:
                point,source=min(primary,key=lambda item:abs(item[0]-coordinate))
            else:
                point=nearest(tad_hints.get(chrom,[]),coordinate)
                source="tad" if point is not None else "multiscale"
            if point is not None:
                result.at[index,"start" if side=="left" else "end"]=point
                result.at[index,side+"_breakpoint_source"]=source
        if int(result.at[index,"end"])>int(result.at[index,"start"]):
            result.at[index,"size"]=(
                int(result.at[index,"end"])-int(result.at[index,"start"])
            )
    return result


def _arm_contact_profile(mat,seed_i,seed_j,lo,hi):
    """Mean raw contact from seed interval to each bin in [lo,hi)."""
    if hi<=lo or seed_j<=seed_i:
        return np.zeros(max(0,hi-lo),dtype=float)
    return mat[seed_i:seed_j,lo:hi].mean(axis=0)


def expand_inv_span_raw(
    mat,start_bin,end_bin,resolution,
    max_gap_bp=2_000_000,max_grow_bp=None
):
    """
    Grow an INV across short low-diag gaps when a distal arm still contacts
    the seed (Juicebox butterfly truncated by coverage gaps).
    """
    n=mat.shape[0]
    size_bins=end_bin-start_bin
    if size_bins<2 or n<4:
        return start_bin,end_bin,False
    max_grow_bins=int(
        (max_grow_bp if max_grow_bp is not None
         else min(8_000_000,max(size_bins*resolution*4,3_000_000)))
        //resolution
    )
    max_gap_bins=max(1,int(max_gap_bp//resolution))
    diag=np.asarray(mat.diagonal(),dtype=float)
    seed_diag=float(np.median(diag[start_bin:end_bin])) or 1.0

    def grow_side(seed_i,seed_j,direction):
        if direction=="right":
            lo,hi=seed_j,min(n,seed_j+max_grow_bins)
            if hi<=lo:
                return seed_j
            profile=_arm_contact_profile(mat,seed_i,seed_j,lo,hi)
            # background: contacts from seed to bins beyond near-diagonal band
            bg=profile.copy()
            near=min(len(bg),max(3,size_bins))
            if near<len(bg):
                thr=float(np.percentile(bg[near:],75)) if len(bg)>near else 0.0
            else:
                thr=float(np.percentile(bg,75)) if len(bg) else 0.0
            thr=max(thr,0.5)
            # allow a short gap then take the first elevated run
            idx=0
            while idx<len(profile) and idx<=max_gap_bins:
                if profile[idx]>=thr and diag[lo+idx]>=0.25*seed_diag:
                    break
                idx+=1
            else:
                return seed_j
            run_end=idx+1
            while (
                run_end<len(profile)
                and profile[run_end]>=thr*0.7
                and diag[lo+run_end]>=0.2*seed_diag
            ):
                run_end+=1
            return lo+run_end
        # left
        lo,hi=max(0,seed_i-max_grow_bins),seed_i
        if hi<=lo:
            return seed_i
        profile=_arm_contact_profile(mat,seed_i,seed_j,lo,hi)
        near=min(len(profile),max(3,size_bins))
        if len(profile)>near:
            thr=float(np.percentile(profile[:-near],75))
        else:
            thr=float(np.percentile(profile,75)) if len(profile) else 0.0
        thr=max(thr,0.5)
        # scan from seed outward (right edge of profile toward left)
        idx=len(profile)-1
        limit=max(0,len(profile)-1-max_gap_bins)
        while idx>=limit:
            if profile[idx]>=thr and diag[lo+idx]>=0.25*seed_diag:
                break
            idx-=1
        else:
            # continuous high-diag growth without distal arm peak
            cur=seed_i
            while cur>lo:
                if diag[cur-1]<0.35*seed_diag:
                    break
                # keep growing while new edge still links to seed right third
                w=max(2,min(10,size_bins//3))
                link=float(mat[cur-1:cur,seed_j-w:seed_j].mean())
                seed_link=float(mat[seed_i:seed_i+w,seed_j-w:seed_j].mean())
                if link<max(0.4*seed_link,0.3):
                    break
                cur-=1
            return cur
        run_start=idx
        while (
            run_start>0
            and profile[run_start-1]>=thr*0.7
            and diag[lo+run_start-1]>=0.2*seed_diag
        ):
            run_start-=1
        return lo+run_start

    new_end=grow_side(start_bin,end_bin,"right")
    new_start=grow_side(start_bin,new_end,"left")
    # second pass with updated span (arms can reinforce each other)
    new_end=grow_side(new_start,new_end,"right")
    new_start=grow_side(new_start,new_end,"left")
    changed=new_start<start_bin or new_end>end_bin
    return new_start,new_end,changed


def _arms_score(mat,i,j):
    """Mean contact between first and last thirds of [i,j)."""
    L=j-i
    if L<6:
        return 0.0
    t=max(2,L//3)
    return float(mat[i:i+t,j-t:j].mean())


def recheck_expanded_span(mat,seed_i,seed_j,new_i,new_j):
    """
    Accept expansion if opposite-arm contact stays enriched vs near-diagonal
    background inside the new span (raw). Synthetic-block GCOR is not used:
    gap-spanning true INVs fail occupancy/gap gates by construction.
    """
    if new_j-new_i<=seed_j-seed_i:
        return False
    arms=_arms_score(mat,new_i,new_j)
    L=new_j-new_i
    t=max(2,L//3)
    # near-diagonal control: first third x middle third
    mid_a=new_i+t
    mid_b=min(new_j,mid_a+t)
    near=float(mat[new_i:new_i+t,mid_a:mid_b].mean()) if mid_b>mid_a else 0.0
    seed_arms=_arms_score(mat,seed_i,seed_j)
    # keep if arms beat near-diag background and do not collapse vs seed
    return (
        arms>=max(1.25*near,0.5)
        and arms>=0.5*max(seed_arms,arms*0.1)
    )


def expand_inversion_calls(mcool_file,calls,resolution,balance=False):
    """
    Post-merge: expand truncated INV arms across short gaps, then arm-enrichment
    recheck. Always expands on raw matrix (no ICE/KR).
    """
    del balance
    if calls is None or calls.empty:
        return calls
    result=calls.copy()
    plot_mcool=resolve_plot_mcool(mcool_file)
    clr=cooler.Cooler(f"{plot_mcool}::resolutions/{int(resolution)}")
    coolers_chr={}
    n_expanded=0
    n_reverted=0
    for index,row in result.iterrows():
        chrom=str(row.chrom)
        start,end=int(row.start),int(row.end)
        if end<=start:
            continue
        if chrom not in coolers_chr:
            mat=clr.matrix(balance=False).fetch(chrom)
            coolers_chr[chrom]=np.nan_to_num(mat,nan=0.0)
        hic=coolers_chr[chrom]
        chrom_size=int(clr.chromsizes[chrom])
        si=start//resolution
        ei=end//resolution
        ni,nj,changed=expand_inv_span_raw(hic,si,ei,resolution)
        if not changed:
            continue
        new_start=max(0,ni*resolution)
        new_end=min(chrom_size,nj*resolution)
        if new_end-new_start<=end-start:
            continue
        if recheck_expanded_span(hic,si,ei,ni,nj):
            result.at[index,"start"]=new_start
            result.at[index,"end"]=new_end
            result.at[index,"size"]=int(new_end-new_start)
            if "full_start" in result.columns:
                result.at[index,"full_start"]=new_start
                result.at[index,"full_end"]=new_end
                result.at[index,"full_size"]=int(new_end-new_start)
            result.at[index,"arm_expanded"]=1
            n_expanded+=1
        else:
            n_reverted+=1
    if "arm_expanded" not in result.columns:
        result["arm_expanded"]=0
    else:
        result["arm_expanded"]=result["arm_expanded"].fillna(0).astype(int)
    print(
        f"[INFO] arm-expand: kept {n_expanded}, recheck-reject {n_reverted}",
        flush=True
    )
    return result


def _expand_inv_selfcheck():
    # synthetic butterfly: left arm bins 10-20 link to right arm 40-50
    n=60
    mat=np.eye(n)*10.0
    mat[10:20,40:50]=8.0
    mat[40:50,10:20]=8.0
    for i in range(n):
        for j in range(n):
            if abs(i-j)<=2:
                mat[i,j]=max(mat[i,j],6.0)
    for i in range(20,40):
        mat[i,i]=0.5
    si,ei=12,18
    ni,nj,changed=expand_inv_span_raw(
        mat,si,ei,resolution=1,max_gap_bp=30,max_grow_bp=50
    )
    assert changed and ni<=12 and nj>=40, (ni,nj,changed)
    assert recheck_expanded_span(mat,si,ei,ni,nj)


def choose_plot_resolution(resolutions,region_size,max_bins=1_500):
    for resolution in resolutions:
        if int(np.ceil(region_size/resolution))<=max_bins:
            return resolution
    return resolutions[-1]


def choose_local_plot_resolution(resolutions,region_size,target_bins=220):
    """Pick a scale near Juicebox density (~200 bins), prefer finer when close."""
    resolutions=sorted(int(value) for value in resolutions)
    if not resolutions:
        raise ValueError("no plot resolutions")
    def score(resolution):
        bins=max(1,int(np.ceil(region_size/resolution)))
        return (abs(bins-target_bins),resolution)
    return min(resolutions,key=score)


def fetch_plot_matrix(clr,region,balance=False):
    # plots stay raw unless caller explicitly asks; no ICE/KR by default
    matrix=clr.matrix(balance=bool(balance)).fetch(region)
    matrix=np.nan_to_num(matrix,nan=0.0,posinf=0.0,neginf=0.0)
    return np.log1p(np.maximum(matrix,0.0))


def heatmap_limits(matrix,upper_percentile=98.0,ignore_diag_bins=0):
    work=np.asarray(matrix,dtype=float)
    if ignore_diag_bins>0 and work.ndim==2 and work.shape[0]==work.shape[1]:
        mask=np.ones(work.shape,dtype=bool)
        for k in range(-ignore_diag_bins,ignore_diag_bins+1):
            idx=np.arange(max(0,-k),min(work.shape[0],work.shape[0]-k))
            mask[idx,idx+k]=False
        positive=work[mask & np.isfinite(work) & (work>0)]
    else:
        positive=work[np.isfinite(work)&(work>0)]
    if not len(positive):
        positive=work[np.isfinite(work)&(work>0)]
    if not len(positive):
        return 0.0,1.0
    upper=float(np.percentile(positive,upper_percentile))
    return 0.0,upper if upper>0 else 1.0


def draw_hic(ax,matrix,local=False):
    # Juicebox-like Observed (raw). gamma>1 keeps mid contacts pale so domains stay visible.
    if local:
        n=int(min(matrix.shape[:2]))
        band=max(3,n//12)
        vmin,vmax=heatmap_limits(matrix,99.5,ignore_diag_bins=band)
        vmax=max(vmax*1.5,1e-6)
        gamma=1.6
        interpolation="bilinear"
    else:
        vmin,vmax=heatmap_limits(matrix,98.0)
        gamma=0.55
        interpolation="nearest"
    return ax.imshow(
        matrix,cmap="Reds",origin="upper",
        norm=PowerNorm(gamma=gamma,vmin=vmin,vmax=vmax,clip=True),
        interpolation=interpolation
    )


def mark_inversion(ax,start_bin,end_bin,label=None):
    width=max(1.0,end_bin-start_bin)
    ax.add_patch(Rectangle(
        (start_bin,start_bin),width,width,fill=False,
        edgecolor="#00e5ff",linewidth=1.0
    ))
    if label:
        ax.text(
            start_bin,max(0,start_bin-3),label,color="#007f8b",
            fontsize=6,ha="left",va="bottom",weight="bold",
            linespacing=1.1,
            bbox={
                "boxstyle":"round,pad=0.15",
                "facecolor":"white",
                "edgecolor":"none",
                "alpha":0.75
            }
        )


def plot_overview(mcool_file,calls,resolution,image_dir,balance=False):
    chromosomes=list(dict.fromkeys(calls.chrom.astype(str)))
    if not chromosomes:
        return
    columns=min(3,len(chromosomes))
    rows=int(np.ceil(len(chromosomes)/columns))
    figure,axes=plt.subplots(
        rows,columns,figsize=(5.2*columns,4.8*rows),
        squeeze=False,constrained_layout=True
    )
    clr=cooler.Cooler(f"{mcool_file}::resolutions/{resolution}")
    for ax,chrom in zip(axes.flat,chromosomes):
        matrix=fetch_plot_matrix(clr,chrom,balance=balance)
        draw_hic(ax,matrix)
        chrom_calls=calls[calls.chrom.astype(str)==chrom]
        for _,call in chrom_calls.iterrows():
            mark_inversion(
                ax,int(call.start)/resolution,int(call.end)/resolution,
                f"INV{int(call.cluster_id)}\n"
                f"{chrom}:{int(call.start):,}-{int(call.end):,}"
            )
        ax.set_title(f"{chrom} | {resolution:,} bp")
        coordinate=FuncFormatter(
            lambda value,_:f"{int(value*resolution):,}"
        )
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.xaxis.set_major_formatter(coordinate)
        ax.yaxis.set_major_formatter(coordinate)
        ax.set_xlabel("Genomic coordinate (bp)")
        ax.set_ylabel("Genomic coordinate (bp)")
        ax.tick_params(axis="x",labelsize=8,labelrotation=30)
        ax.tick_params(axis="y",labelsize=8)
    for ax in axes.flat[len(chromosomes):]:
        ax.axis("off")
    figure.suptitle(
        "Multi-scale supported inversions — Hi-C overview\n"
        "cyan boxes: reported inversions",fontsize=14
    )
    figure.savefig(image_dir/"ALL_INV_overview.png",dpi=180,bbox_inches="tight")
    plt.close(figure)


def plot_local_inversions(mcool_file,calls,resolutions,image_dir,balance=False):
    del balance  # local Hi-C figures are always raw (no ICE/KR)
    local_dir=image_dir/"local"
    local_dir.mkdir(parents=True,exist_ok=True)
    plot_mcool=resolve_plot_mcool(mcool_file)
    plot_resolutions=available_resolutions(plot_mcool) or list(resolutions)
    if plot_mcool!=str(Path(mcool_file).resolve()):
        print(f"[INFO] local plots from {plot_mcool} (raw)",flush=True)
    coolers={}
    for _,call in calls.iterrows():
        chrom=str(call.chrom)
        start,end=int(call.start),int(call.end)
        event_size=max(1,end-start)
        base_clr=coolers.setdefault(
            plot_resolutions[0],
            cooler.Cooler(f"{plot_mcool}::resolutions/{plot_resolutions[0]}")
        )
        chrom_size=int(base_clr.chromsizes[chrom])
        # flank = 2x event (was 1x); Juicebox-style context around the call
        flank=2*event_size
        window_start=max(0,start-flank)
        window_end=min(chrom_size,end+flank)
        resolution=choose_local_plot_resolution(
            plot_resolutions,window_end-window_start
        )
        clr=coolers.setdefault(
            resolution,
            cooler.Cooler(f"{plot_mcool}::resolutions/{resolution}")
        )
        matrix=fetch_plot_matrix(
            clr,(chrom,window_start,window_end),balance=False
        )
        figure,ax=plt.subplots(figsize=(9.2,7.6),constrained_layout=True)
        image=draw_hic(ax,matrix,local=True)
        mark_inversion(
            ax,(start-window_start)/resolution,(end-window_start)/resolution
        )
        scale_text=", ".join(
            f"{int(value):,}" for value in str(call.resolutions).split(",")
        )
        ax.set_title(
            f"INV{int(call.cluster_id)} | {chrom}\n"
            f"{start:,}–{end:,} bp\n"
            f"support: {int(call.scale_support)} scales ({scale_text}) | "
            f"plot: {resolution:,} bp raw (Juicebox-style)",
            fontsize=12,linespacing=1.25,pad=12
        )
        coordinate=FuncFormatter(
            lambda value,_:f"{int(window_start+value*resolution):,}"
        )
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.xaxis.set_major_formatter(coordinate)
        ax.yaxis.set_major_formatter(coordinate)
        ax.set_xlabel("Genomic coordinate (bp)",fontsize=11)
        ax.set_ylabel("Genomic coordinate (bp)",fontsize=11)
        ax.tick_params(axis="x",labelsize=8,labelrotation=30)
        ax.tick_params(axis="y",labelsize=8)
        colorbar=figure.colorbar(image,ax=ax,shrink=0.78,pad=0.035)
        colorbar.set_ticks(np.linspace(0,image.norm.vmax,6))
        colorbar.ax.yaxis.set_major_formatter(
            FuncFormatter(lambda value,_:f"{value:.3f}")
        )
        colorbar.set_label("log1p raw contact",fontsize=10,labelpad=8)
        colorbar.ax.tick_params(labelsize=9)
        filename=f"INV{int(call.cluster_id):04d}_{chrom}_{start}_{end}.png"
        figure.savefig(local_dir/filename,dpi=200,bbox_inches="tight")
        plt.close(figure)


def workflow_main():
    args=parse_workflow_args()
    args.mcool_file=str(Path(args.mcool_file).expanduser().resolve())
    if not Path(args.mcool_file).is_file():
        raise SystemExit("input does not exist: "+args.mcool_file)
    available=available_resolutions(args.mcool_file)
    resolutions=select_four_resolutions(available)
    if len(resolutions)!=N_SCALES:
        raise SystemExit(
            "Cannot select four resolutions around/above 50 kb.\n"
            "Available: "+", ".join(map(str,available))
        )
    print(
        "[INFO] selected resolutions: "+
        ", ".join(f"{resolution:,}" for resolution in resolutions),
        flush=True
    )
    outdir=Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True,exist_ok=True)
    script=Path(__file__).resolve()

    with tempfile.TemporaryDirectory(prefix="original_inv_") as temporary:
        temporary=Path(temporary)
        scale_dirs=[temporary/f"scale_{resolution}" for resolution in resolutions]
        for directory in scale_dirs:
            directory.mkdir(parents=True)
        # Louvain once at the finest resolution; coarser scales only
        # re-evaluate GCOR on projected communities.
        _lift_blocks_selfcheck()
        _edge_oneside_guard_selfcheck()
        _grow_secondary_inv_selfcheck()
        fine_res=resolutions[0]
        fine_dir=scale_dirs[0]
        print(
            "[INFO] Louvain once at finest resolution "
            f"{fine_res:,}; coarser scales lift those blocks",
            flush=True
        )
        run_internal_command(
            original_detection_command(
                script,args,fine_res,fine_dir,args.jobs
            ),
            f"original scale {fine_res:,} (Louvain)"
        )
        coarse=list(zip(resolutions[1:], scale_dirs[1:]))
        if coarse:
            scale_workers=max(1, args.jobs//len(coarse))
            with ThreadPoolExecutor(
                max_workers=min(len(coarse), args.jobs)
            ) as executor:
                futures={
                    executor.submit(
                        run_internal_command,
                        original_detection_command(
                            script,args,resolution,directory,
                            scale_workers,lift_blocks_dir=fine_dir
                        ),
                        f"lifted scale {resolution:,}"
                    ):resolution
                    for resolution,directory in coarse
                }
                for future in as_completed(futures):
                    future.result()

        merge_dir=temporary/"original_merge"
        merge_dir.mkdir()
        run_internal_command(
            original_merge_command(
                script,args,resolutions,scale_dirs,merge_dir
            ),
            "original multiscale merge"
        )
        result_file=merge_dir/"ALL_INV_multiscale.tsv"
        if result_file.is_file() and result_file.stat().st_size:
            try:
                result=pd.read_csv(result_file,sep="\t")
            except pd.errors.EmptyDataError:
                result=pd.DataFrame()
        else:
            result=pd.DataFrame()

    if not result.empty:
        result=result[
            pd.to_numeric(result["size"],errors="coerce")
            <=args.max_inversion_size
        ].copy().reset_index(drop=True)

    if not result.empty:
        _expand_inv_selfcheck()
        print(
            "[INFO] expanding truncated INV arms across short gaps (raw)",
            flush=True
        )
        result=expand_inversion_calls(
            args.mcool_file,result,resolutions[0],balance=False
        )
        result=result[
            pd.to_numeric(result["size"],errors="coerce")
            <=args.max_inversion_size
        ].copy().reset_index(drop=True)
        # Collapse near-duplicate spans left by cross-scale / arm-expand.
        n_before=len(result)
        result=dedup_overlapping_inv_calls(
            result, min_ios=float(getattr(args,"merge_overlap",0.5))
        )
        if len(result)!=n_before:
            print(
                f"[INFO] dedup overlapping INV: {n_before} -> {len(result)}",
                flush=True
            )

    if not result.empty and (args.hic_alignments or args.fasta):
        gap_hints={}
        tad_hints={}
        hic_evidence=None
        if args.fasta:
            print("[INFO] scanning FASTA gaps",flush=True)
            gap_hints=scan_fasta_gaps(args.fasta)
            print("[INFO] running four-scale hicFindTADs",flush=True)
            tad_hints=detect_tad_boundaries(
                args.mcool_file,resolutions,args.jobs
            )
        if args.hic_alignments:
            print("[INFO] finding Hi-C split/pair breakpoints",flush=True)
            hic_evidence=load_hic_breakpoints(
                args.hic_alignments,result,tolerance=resolutions[0]
            )
        result=refine_final_breakpoints(
            result,gap_hints,tad_hints,hic_evidence,resolutions[0]
        )

    result_path=outdir/"ALL_INV_multiscale.tsv"
    if result.empty:
        pd.DataFrame(columns=[
            "cluster_id","chrom","start","end","size","scale_support",
            "call_support","resolutions"
        ]).to_csv(result_path,sep="\t",index=False)
    else:
        result.to_csv(result_path,sep="\t",index=False)

    if not result.empty:
        image_dir=outdir/"HiC_images"
        image_dir.mkdir(parents=True,exist_ok=True)
        reference=cooler.Cooler(
            f"{args.mcool_file}::resolutions/{resolutions[-1]}"
        )
        largest=max(
            int(reference.chromsizes[str(chrom)])
            for chrom in result.chrom.astype(str).unique()
        )
        overview_resolution=choose_plot_resolution(available,largest)
        plot_overview(
            args.mcool_file,result,overview_resolution,image_dir,
            balance=bool(args.balance)
        )
        plot_local_inversions(
            args.mcool_file,result,resolutions,image_dir,
            balance=bool(args.balance)
        )
    print(
        f"[DONE] {len(result)} supported inversions -> {result_path}",
        flush=True
    )


def original_internal_main():
    args=parse_args()
    if args.multiscale_inputs:
        merge_multiscale_results(args)
    else:
        run_parallel(args)
        merge_results(args)


if __name__=="__main__":
    if os.environ.get("INV_ORIGINAL_INTERNAL")=="1":
        original_internal_main()
    else:
        workflow_main()
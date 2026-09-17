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


    # optional
    # 如果不给，则自动生成
    parser.add_argument(
        "--block_file",
        default=None,
        help="existing Louvain block file"
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
            "minimum total number of neighboring blocks used around a "
            "candidate (default: 2)"
        )
    )


    parser.add_argument(
        "--max_neighbor_blocks",
        type=int,
        default=8,
        help=(
            "maximum total number of neighboring blocks tried around a "
            "candidate; contexts are enlarged one block at a time "
            "(default: 8)"
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
        default=2,
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
        radius
):


    n=M.shape[0]


    G=nx.Graph()


    G.add_nodes_from(
        range(n)
    )


    for i in range(n):

        for j in range(
            i+1,
            min(
                n,
                i+radius+1
            )
        ):


            w=M[i,j]


            if w>0:

                G.add_edge(
                    i,
                    j,
                    contact_weight=float(w)
                )


    return G





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





############################################################
# Generate block file
############################################################


def generate_block_file(
        labels,
        chrom,
        resolution,
        outfile
):


    blocks=[]


    start=0

    current=labels[0]



    for i,c in enumerate(
        labels[1:],
        1
    ):


        if c!=current:


            blocks.append(
                [
                    chrom,
                    start*resolution,
                    i*resolution,
                    current
                ]
            )


            start=i
            current=c



    blocks.append(
        [
            chrom,
            start*resolution,
            len(labels)*resolution,
            current
        ]
    )



    pd.DataFrame(
        blocks
    ).to_csv(
        outfile,
        sep="\t",
        header=False,
        index=False
    )


    return outfile


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
# Extract block matrix
############################################################


def get_block_matrix(
        hic,
        block1,
        block2,
        resolution
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


    return mat





############################################################
# GCOR directional trend
############################################################


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






############################################################
# Detect inversion
############################################################


def detect_block_INV(
        hic,
        blocks,
        index,
        args
):

    """Progressively enlarge context; either side may support an inversion."""


    if index<=0 or index>=len(blocks)-1:
        return None


    target=blocks[index]


    # load_blocks normally absorbs short communities.  Keep this guard for a
    # direct caller that bypasses loading and supplies an undersized target.
    if target[1]-target[0]<args.min_block:
        return None


    min_neighbors=max(2, int(args.min_neighbor_blocks))
    max_neighbors=max(min_neighbors, int(args.max_neighbor_blocks))
    max_neighbors=min(max_neighbors, len(blocks)-1)
    best_negative=None


    for total_neighbors in range(min_neighbors, max_neighbors+1):

        level_candidates=[]


        # Keep at least one block on each side.  For a total of three this
        # checks both 1-left/2-right and 2-left/1-right.
        for n_left in range(1, total_neighbors):

            n_right=total_neighbors-n_left


            if index-n_left<0 or index+n_right>=len(blocks):
                continue


            left_blocks=blocks[index-n_left:index]
            right_blocks=blocks[index+1:index+1+n_right]


            # The merged blocks should remain a gap-free genomic partition.
            # Reject a malformed external block file rather than silently
            # treating distant regions as direct neighbors.
            context=left_blocks+[target]+right_blocks


            if any(
                context[i][1]!=context[i+1][0]
                for i in range(len(context)-1)
            ):
                continue


            left_matrix=np.vstack([
                get_block_matrix(hic, block, target, args.res)
                for block in left_blocks
            ])


            right_matrix=np.hstack([
                get_block_matrix(hic, target, block, args.res)
                for block in right_blocks
            ])


            left_rho,left_p,_=calculate_direction(
                left_matrix,
                "column",
                args.smooth
            )


            right_rho,right_p,_=calculate_direction(
                right_matrix,
                "row",
                args.smooth
            )


            left_vote=left_rho>args.rho_cutoff
            right_vote=right_rho< -args.rho_cutoff
            inv=left_vote or right_vote


            if left_vote and right_vote:
                detection_side="both"
            elif left_vote:
                detection_side="left"
            elif right_vote:
                detection_side="right"
            else:
                detection_side="none"


            strength=max(left_rho, -right_rho)


            level_candidates.append({
                "start":target[0],
                "end":target[1],
                "size":target[1]-target[0],
                "left_start":left_blocks[0][0],
                "left_end":left_blocks[-1][1],
                "right_start":right_blocks[0][0],
                "right_end":right_blocks[-1][1],
                "left_rho":left_rho,
                "left_p":left_p,
                "right_rho":right_rho,
                "right_p":right_p,
                "left_vote":int(left_vote),
                "right_vote":int(right_vote),
                "INV":int(inv),
                "detection_side":detection_side,
                "neighbor_blocks":total_neighbors,
                "left_blocks":n_left,
                "right_blocks":n_right,
                "detection_strength":strength
            })


        if not level_candidates:
            continue


        # Use the smallest successful context and, within it, the strongest
        # allocation.  This prevents a very broad window from replacing a
        # clean local signal.
        positives=[
            candidate
            for candidate in level_candidates
            if candidate["INV"]==1
        ]


        if positives:
            return max(
                positives,
                key=lambda candidate: candidate["detection_strength"]
            )


        level_best=max(
            level_candidates,
            key=lambda candidate: candidate["detection_strength"]
        )


        if (
            best_negative is None
            or level_best["detection_strength"]
               >best_negative["detection_strength"]
        ):
            best_negative=level_best


    return best_negative




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



    clr=cooler.Cooler(
        f"{args.mcool_file}::resolutions/{args.res}"
    )



    hic=clr.matrix(
        balance=True
    ).fetch(
        chrom
    )


    hic=np.nan_to_num(
        hic,
        nan=0
    )



    ##################################################
    # generate block automatically
    ##################################################


    if args.block_file is None:


        print(
            "[INFO] generating Louvain blocks"
        )


        norm=distance_normalize(
            hic
        )


        G=build_contact_graph(
            norm,
            args.radius
        )


        direction,confidence=calculate_direction_information(
            norm,
            args.radius
        )



        G=fuse_direction(
            G,
            direction,
            confidence,
            args.direction_lambda
        )


        labels=run_louvain(
            G,
            args.louvain_resolution
        )



        block_file=os.path.join(
            args.outdir,
            chrom+".blocks.tsv"
        )



        generate_block_file(
            labels,
            chrom,
            args.res,
            block_file
        )


    else:

        block_file=args.block_file



    blocks=load_blocks(
        block_file,
        chrom,
        args.min_block
    )


    print(
        chrom,
        "blocks:",
        len(blocks)
    )



    results=[]


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


        if r:

            r["chrom"]=chrom

            results.append(
                r
            )



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


    for f in os.listdir(
        args.outdir
    ):


        if f.endswith(
            ".INV.tsv"
        ):

            files.append(
                os.path.join(
                    args.outdir,
                    f
                )
            )



    if not files:

        print(
            "No inversion"
        )

        return



    dfs=[]


    for f in files:

        dfs.append(
            pd.read_csv(
                f,
                sep="\t"
            )
        )



    df=pd.concat(
        dfs,
        ignore_index=True
    )


    df.to_csv(
        os.path.join(
            args.outdir,
            "ALL_INV.tsv"
        ),
        sep="\t",
        index=False
    )


    print(
        "Merged:",
        len(df)
    )



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
        mat=clr.matrix(balance=True).fetch((chrom,start,end))
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
            left_mat=clr.matrix(balance=True).fetch(
                (chrom,left_start,start),
                (chrom,start,end)
            )
            left_boundary_rho,_,_=calculate_direction(
                np.nan_to_num(left_mat,nan=0),
                "column",
                args.smooth
            )


        if end<right_end:
            right_mat=clr.matrix(balance=True).fetch(
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
            left_mat=clr.matrix(balance=True).fetch(
                (chrom,left_start,int(start)),
                (chrom,int(start),int(end))
            )
            left_rho,_,_=calculate_direction(
                np.nan_to_num(left_mat,nan=0),
                "column",
                args.smooth
            )


        if int(end)<right_end:
            right_mat=clr.matrix(balance=True).fetch(
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
                left_mat=clr.matrix(balance=True).fetch(
                    (chrom,left_start,start),
                    (chrom,sample_start,sample_end)
                )
                left_rho,_,_=calculate_direction(
                    np.nan_to_num(left_mat,nan=0),
                    "column",
                    args.smooth
                )


            if end<right_end:
                right_mat=clr.matrix(balance=True).fetch(
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


            if int(a.resolution)!=int(b.resolution) and not (
                contained>=args.merge_containment
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


def original_detection_command(script,args,resolution,scale_dir,workers):
    return [
        sys.executable,str(script),
        "--mcool_file",args.mcool_file,
        "--res",str(resolution),
        "--min_block",str(args.min_community_size),
        "--outdir",str(scale_dir),
        "--nprocess",str(workers)
    ]


def original_merge_command(script,args,resolutions,scale_dirs,merge_dir):
    command=[
        sys.executable,str(script),
        "--mcool_file",args.mcool_file,
        "--outdir",str(merge_dir),
        "--merge_outdir",str(merge_dir),
        "--merge_overlap",str(args.merge_overlap),
        "--merge_gap",str(args.max_merge_gap),
        "--min_scale_support",str(args.min_support),
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


def choose_plot_resolution(resolutions,region_size,max_bins=1_500):
    for resolution in resolutions:
        if int(np.ceil(region_size/resolution))<=max_bins:
            return resolution
    return resolutions[-1]


def fetch_plot_matrix(clr,region):
    matrix=clr.matrix(balance=True).fetch(region)
    matrix=np.nan_to_num(matrix,nan=0.0,posinf=0.0,neginf=0.0)
    return np.log1p(np.maximum(matrix,0.0))


def heatmap_limits(matrix,upper_percentile=98.0):
    positive=matrix[np.isfinite(matrix)&(matrix>0)]
    if not len(positive):
        return 0.0,1.0
    upper=float(np.percentile(positive,upper_percentile))
    return 0.0,upper if upper>0 else 1.0


def draw_hic(ax,matrix,local=False):
    percentile=95.0 if local else 98.0
    gamma=0.38 if local else 0.55
    vmin,vmax=heatmap_limits(matrix,percentile)
    return ax.imshow(
        matrix,cmap="Reds",origin="upper",
        norm=PowerNorm(gamma=gamma,vmin=vmin,vmax=vmax,clip=True),
        interpolation="nearest"
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


def plot_overview(mcool_file,calls,resolution,image_dir):
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
        matrix=fetch_plot_matrix(clr,chrom)
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


def plot_local_inversions(mcool_file,calls,resolutions,image_dir):
    local_dir=image_dir/"local"
    local_dir.mkdir(parents=True,exist_ok=True)
    coolers={}
    for _,call in calls.iterrows():
        chrom=str(call.chrom)
        start,end=int(call.start),int(call.end)
        event_size=max(1,end-start)
        base_clr=coolers.setdefault(
            resolutions[0],
            cooler.Cooler(f"{mcool_file}::resolutions/{resolutions[0]}")
        )
        chrom_size=int(base_clr.chromsizes[chrom])
        window_start=max(0,start-event_size)
        window_end=min(chrom_size,end+event_size)
        resolution=choose_plot_resolution(
            resolutions,window_end-window_start
        )
        clr=coolers.setdefault(
            resolution,
            cooler.Cooler(f"{mcool_file}::resolutions/{resolution}")
        )
        matrix=fetch_plot_matrix(clr,(chrom,window_start,window_end))
        figure,ax=plt.subplots(figsize=(8.6,7.2),constrained_layout=True)
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
            f"plot: {resolution:,} bp",
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
        colorbar.set_label("log1p balanced contact",fontsize=10,labelpad=8)
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
        scale_workers=max(1,args.jobs//len(resolutions))
        with ThreadPoolExecutor(max_workers=min(len(resolutions),args.jobs)) as executor:
            futures={
                executor.submit(
                    run_internal_command,
                    original_detection_command(
                        script,args,resolution,directory,scale_workers
                    ),
                    f"original scale {resolution:,}"
                ):resolution
                for resolution,directory in zip(resolutions,scale_dirs)
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
            args.mcool_file,result,overview_resolution,image_dir
        )
        plot_local_inversions(
            args.mcool_file,result,resolutions,image_dir
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

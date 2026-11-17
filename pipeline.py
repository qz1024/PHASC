#!/usr/bin/env python3
"""Run inversion and intra-chromosomal translocation detection in parallel.

Both detectors write to private work directories first (TRA would otherwise
wipe INV's HiC_images/), then final artifacts are promoted into --outdir.

TRA runs one detector process per chromosome. Concurrent chromosome workers
are controlled by --chrom-jobs (default: auto). Each worker still uses
--jobs for resolution×gamma parallelism inside that chromosome.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


def _fmt_duration(sec: float) -> str:
    """Human-readable wall time, e.g. '14m 49s (889 s)'."""
    total = max(0, int(round(float(sec))))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        pretty = f"{h}h {m}m {s}s"
    elif m:
        pretty = f"{m}m {s}s"
    else:
        pretty = f"{s}s"
    return f"{pretty} ({total} s)"


assert _fmt_duration(889) == "14m 49s (889 s)"
assert _fmt_duration(3661) == "1h 1m 1s (3661 s)"


def _fmt_score(val) -> str:
    if val is None:
        return "NA"
    try:
        return f"{float(val):.4f}"
    except (TypeError, ValueError):
        return "NA"


def _stream_output(pipe, prefix: str) -> None:
    for line in iter(pipe.readline, ""):
        sys.stdout.write(f"[{prefix}] {line}")
        sys.stdout.flush()
    pipe.close()


def _run_labeled(cmd: list[str], label: str) -> None:
    print(f"[START] {label}", flush=True)
    print(f"[CMD] {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    reader = threading.Thread(
        target=_stream_output, args=(proc.stdout, label), daemon=True
    )
    reader.start()
    rc = proc.wait()
    reader.join()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)
    print(f"[DONE] {label}", flush=True)


def _promote(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def _promote_inv_plots(work: Path, outdir: Path) -> list[str]:
    """Keep only flipped INV panels, renamed HiC_INV_Plot (drop HiC_images)."""
    src = work / "HiC_images_flipped"
    if not src.exists():
        return []
    _promote(src, outdir / "HiC_INV_Plot")
    return ["HiC_INV_Plot"]


def _collect_tra_plots(chrom_works: dict[str, Path], outdir: Path) -> list[str]:
    """Gather TRA plots into HiC_TRA_Plot/: <chrom>/community/ + HiC_TRA_Local/."""
    dest_root = outdir / "HiC_TRA_Plot"
    if dest_root.exists():
        shutil.rmtree(dest_root)
    local_root = dest_root / "HiC_TRA_Local"
    n = 0
    for chrom, work in chrom_works.items():
        src = work / "community"
        if src.exists():
            dest = dest_root / chrom / "community"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dest)
            n += 1
        # event plots: flat under HiC_TRA_Plot/HiC_TRA_Local/ (not under chrom/)
        src_local = work / "HiC_TRA_Local"
        if src_local.exists():
            local_root.mkdir(parents=True, exist_ok=True)
            for png in sorted(src_local.glob("*.png")):
                shutil.copy2(png, local_root / png.name)
                n += 1
    if n == 0:
        return []
    return ["HiC_TRA_Plot"]


def _merge_tra_tables(
    chrom_works: dict[str, Path],
    out_tsv: Path,
    tra_mcool: Path | None = None,
) -> Path | None:
    """Merge per-chrom ALL_TRA; shorter reciprocal arm + internal-signal gate."""
    paths = []
    for work in chrom_works.values():
        for name in ("ALL_TRA.tsv", "merged_true_calls.tsv"):
            p = work / name
            if p.is_file():
                paths.append(p)
                break
    if not paths:
        return None
    if not _merge_tsv_files(paths, out_tsv):
        return None
    try:
        import pandas as pd

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from multiscale_intra_translocation import (
            filter_tra_weak_internal,
            keep_shorter_reciprocal_tra,
        )

        df = pd.read_csv(out_tsv, sep="\t")
        if len(df):
            df2 = keep_shorter_reciprocal_tra(df)
            if tra_mcool is not None:
                df2 = filter_tra_weak_internal(df2, tra_mcool)
            df2.to_csv(out_tsv, sep="\t", index=False)
    except Exception as exc:  # noqa: BLE001 — detection already succeeded
        print(f"[WARN] TRA post-filters skipped: {exc}", flush=True)
    return out_tsv


def _read_tsv(path: Path):
    import pandas as pd

    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, sep="\t")
    except pd.errors.EmptyDataError:
        # headerless / whitespace-only TRA stubs
        return pd.DataFrame()


def _merge_tsv_files(paths: list[Path], out: Path) -> bool:
    import pandas as pd

    frames = []
    for path in paths:
        df = _read_tsv(path)
        if not df.empty:
            frames.append(df)
    if not frames:
        return False
    merged = pd.concat(frames, ignore_index=True, sort=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, sep="\t", index=False)
    return True


def _merge_sv_tables(
    outdir: Path,
    *,
    inv_tsv: Path | None,
    tra_tsv: Path | None,
) -> Path | None:
    """Write a combined ALL_SV.tsv with sv_type=INV/TRA and shared locus columns."""
    import pandas as pd

    frames: list[pd.DataFrame] = []

    inv = _read_tsv(inv_tsv) if inv_tsv else pd.DataFrame()
    if not inv.empty:
        start = inv["full_start"] if "full_start" in inv.columns else inv.get("start")
        end = inv["full_end"] if "full_end" in inv.columns else inv.get("end")
        if "core_size" in inv.columns:
            size = inv["core_size"]
        elif "size" in inv.columns:
            size = inv["size"]
        else:
            size = end - start
        row = pd.DataFrame(
            {
                "sv_type": "INV",
                "chrom": inv["chrom"] if "chrom" in inv.columns else pd.NA,
                "start": start,
                "end": end,
                "size": size,
                "call_support": (
                    inv["call_support"]
                    if "call_support" in inv.columns
                    else inv["scale_support"]
                    if "scale_support" in inv.columns
                    else pd.NA
                ),
            }
        )
        for col in (
            "scale_support",
            "resolutions",
            "event_score",
            "left_rho",
            "right_rho",
            "flip_improve",
            "flip_pos_scales",
            "cluster_id",
        ):
            if col in inv.columns:
                row[col] = inv[col].values
        frames.append(row)

    tra = _read_tsv(tra_tsv) if tra_tsv else pd.DataFrame()
    if not tra.empty:
        if "size" not in tra.columns and {"start", "end"} <= set(tra.columns):
            tra = tra.copy()
            tra["size"] = tra["end"].astype(int) - tra["start"].astype(int)
        support = (
            tra["call_support"]
            if "call_support" in tra.columns
            else tra["scale_support"]
            if "scale_support" in tra.columns
            else pd.NA
        )
        row = pd.DataFrame(
            {
                "sv_type": "TRA",
                "chrom": tra["chrom"] if "chrom" in tra.columns else pd.NA,
                "start": tra["start"] if "start" in tra.columns else pd.NA,
                "end": tra["end"] if "end" in tra.columns else pd.NA,
                "size": tra["size"] if "size" in tra.columns else pd.NA,
                "call_support": support,
            }
        )
        for col in (
            "partner_start",
            "partner_end",
            "partner_size",
            "scale_support",
            "resolutions",
            "gammas",
            "distance_bp",
            "enrichment_fold",
            "cluster_id",
            "mutual_insert",
            "mutual_pair_id",
            "mutual_partner_start",
            "mutual_partner_end",
        ):
            if col in tra.columns:
                row[col] = tra[col].values
        frames.append(row)

    if not frames:
        return None

    merged = pd.concat(frames, ignore_index=True, sort=False)
    sort_cols = [c for c in ("sv_type", "chrom", "start", "end") if c in merged.columns]
    if sort_cols:
        merged = merged.sort_values(sort_cols).reset_index(drop=True)

    out = outdir / "ALL_SV.tsv"
    merged.to_csv(out, sep="\t", index=False)
    return out


def _write_final_report(
    outdir: Path,
    *,
    start_epoch: float,
    wall_sec: float,
    inv_mcool: Path,
    tra_mcool: Path,
    summary: dict | None,
    n_inv: int,
    n_tra: int,
    jobs: int,
    chrom_jobs: int,
    chromosomes: list[str],
) -> Path:
    """Merged runtime + scores -> phasc_final_report.txt (replaces report/runtime)."""
    scores = (summary or {}).get("component_scores_0_100") or {}
    inv_score = scores.get("inversion_score")
    tra_score = scores.get("translocation_score")
    total = (summary or {}).get("total_score_0_100")
    start_dt = datetime.fromtimestamp(start_epoch).astimezone()
    end_dt = datetime.fromtimestamp(start_epoch + wall_sec).astimezone()
    n_chrom = len(chromosomes)
    chrom_preview = ", ".join(chromosomes)
    if len(chrom_preview) > 200:
        chrom_preview = ", ".join(chromosomes[:12]) + f", ... ({n_chrom} total)"

    lines = [
        "PHASC final report",
        "==================",
        "",
        "[Runtime]",
        f"start:     {start_dt.isoformat(timespec='seconds')}",
        f"end:       {end_dt.isoformat(timespec='seconds')}",
        f"wall_time: {_fmt_duration(wall_sec)}",
        "note:      INV and per-chromosome TRA run in parallel; wall_time is total elapsed.",
        "",
        "[Inputs]",
        f"inv_mcool:    {inv_mcool}",
        f"tra_mcool:    {tra_mcool}",
        f"chromosomes:  {n_chrom} ({chrom_preview})",
        f"jobs:         {jobs}  (INV / per-chrom inner workers)",
        f"chrom_jobs:   {chrom_jobs}  (TRA chromosomes in parallel)",
        "",
        "[Results]",
        f"n_inv:                 {n_inv}",
        f"n_tra:                 {n_tra}",
        f"inversion_score:       {_fmt_score(inv_score)}",
        f"translocation_score:   {_fmt_score(tra_score)}",
        f"HiC_SCI:               {_fmt_score(total)}",
        "",
        "[Outputs]",
        "ALL_SV.tsv",
        "HiC_INV_Plot/",
        "HiC_TRA_Plot/<chrom>/community/   # whole-chromosome community plots",
        "HiC_TRA_Plot/HiC_TRA_Local/       # detected TRA: community + Hi-C on one figure",
        "phasc_final_report.txt",
    ]
    path = outdir / "phasc_final_report.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # drop legacy split reports if present
    for legacy in ("report.txt", "runtime.txt"):
        p = outdir / legacy
        if p.is_file():
            p.unlink(missing_ok=True)
    return path


def _chroms_from_mcool(mcool: Path) -> list[str]:
    import cooler

    store = cooler.fileops.list_coolers(str(mcool))
    if not store:
        raise SystemExit(f"[ERROR] no cooler resolutions in {mcool}")
    uri = store[0] if store[0].startswith(str(mcool)) else f"{mcool}::{store[0]}"
    chroms = [str(c) for c in cooler.Cooler(uri).chromnames]
    if not chroms:
        raise SystemExit(f"[ERROR] no chromosomes in {mcool}")
    return chroms


def _resolve_chromosomes(args: argparse.Namespace, tra_mcool: Path) -> list[str]:
    available = _chroms_from_mcool(tra_mcool)
    if args.chromosomes:
        wanted = [c.strip() for c in str(args.chromosomes).split(",") if c.strip()]
        if not wanted:
            raise SystemExit("[ERROR] --chromosomes is empty")
        missing = [c for c in wanted if c not in available]
        if missing:
            raise SystemExit(
                f"[ERROR] chromosomes not in mcool: {', '.join(missing)}"
            )
        return wanted
    # default: all chromosomes in the TRA mcool
    return available


def _build_tra_cmd(
    *,
    tra_script: Path,
    tra_mcool: Path,
    outdir: Path,
    chrom: str,
    jobs: int,
    args: argparse.Namespace,
) -> list[str]:
    cmd = [
        sys.executable,
        str(tra_script),
        "--mcool",
        str(tra_mcool),
        "--outdir",
        str(outdir),
        "--chrom",
        chrom,
        "--jobs",
        str(jobs),
        "--min-enrichment-fold",
        str(args.min_enrichment_fold),
        "--tip-window-bp",
        str(args.tip_window_bp),
    ]
    if args.tra_skip_plot:
        cmd.append("--skip-plot")
    if args.truth:
        cmd.extend(["--truth", str(Path(args.truth).expanduser().resolve())])
    if args.tra_resolutions:
        cmd.extend(["--resolutions", args.tra_resolutions])
    if args.tra_gammas:
        cmd.extend(["--gammas", args.tra_gammas])
    # when many chromosomes run at once, auto threads (0) oversubscribe; clamp
    tra_threads = int(args.tra_threads)
    if tra_threads == 0 and int(args.chrom_jobs_effective) > 1:
        tra_threads = 1
    cmd.extend(["--threads", str(tra_threads)])
    return cmd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run PHASC inversion and intra-chromosomal translocation "
            "detection in parallel and write both results into the same "
            "--outdir. Prefer separate preprocess mcools (INV MAPQ / TRA MAPQ); "
            "a single positional mcool is still accepted for both. "
            "TRA defaults to all chromosomes in the mcool, run concurrently."
        )
    )
    parser.add_argument(
        "mcool",
        nargs="?",
        default=None,
        help="shared .mcool used for both INV and TRA when --inv-mcool/--tra-mcool omitted",
    )
    parser.add_argument(
        "--inv-mcool",
        default=None,
        help="inversion .mcool (e.g. preprocess *.inv.mapq0.mcool)",
    )
    parser.add_argument(
        "--tra-mcool",
        default=None,
        help="intra-TRA .mcool (e.g. preprocess *.tra.mapq1.mcool)",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        default="PHASC_results",
        help="shared output directory for INV + TRA (default: PHASC_results)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help=(
            "worker processes for INV and for each TRA chromosome's "
            "resolution×gamma pool (default: min(4, CPUs)). When multiple "
            "chromosomes run, each chromosome gets max(1, --jobs // --chrom-jobs)."
        ),
    )
    parser.add_argument(
        "--chrom-jobs",
        type=int,
        default=21,
        help=(
            "concurrent TRA chromosome workers (default: 21). "
            "Set 0 for auto min(n_chroms, max(1, CPUs//2)); 1 for serial."
        ),
    )
    parser.add_argument(
        "--balance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use ICE-balanced contacts for inversion (default: --no-balance)",
    )
    parser.add_argument(
        "--inv-min-support",
        type=int,
        default=1,
        help="inversion --min-support (default: 1)",
    )
    parser.add_argument(
        "--fasta",
        default=None,
        help="optional FASTA for inversion refinement",
    )
    parser.add_argument(
        "--hic-alignments",
        default=None,
        help="optional BAM/CRAM/.pairs/BEDPE for inversion refinement",
    )
    parser.add_argument(
        "--chromosomes",
        default=None,
        help=(
            "comma-separated chromosomes for INV and intra-TRA "
            "(default: all chromosomes in the TRA mcool). "
            "Example: chr1A,chr1B,chr3B"
        ),
    )
    parser.add_argument(
        "--truth",
        default=None,
        help="optional TRA truth BED passed to intra-translocation --truth",
    )
    parser.add_argument(
        "--tra-skip-plot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="skip TRA community/HiC plots (default: plots on; --tra-skip-plot to disable)",
    )
    parser.add_argument(
        "--tra-resolutions",
        default=None,
        help="optional TRA --resolutions (comma-separated bp)",
    )
    parser.add_argument(
        "--tra-gammas",
        default=None,
        help="optional TRA --gammas (comma-separated)",
    )
    parser.add_argument(
        "--min-enrichment-fold",
        type=float,
        default=1.5,
        help=(
            "TRA final enrichment gate after clear-insert merge "
            "(enrichment_fold > threshold; default: 1.5; <=0 disables)"
        ),
    )
    parser.add_argument(
        "--tip-window-bp",
        type=int,
        default=2_000_000,
        help=(
            "TRA breakpoint tip window for O/E enrichment_fold "
            "(default: 2000000)"
        ),
    )
    parser.add_argument(
        "--tra-threads",
        type=int,
        default=0,
        help=(
            "NetworKit threads for TRA PLM (default: 0 = auto; forced to 1 "
            "when chrom-jobs > 1 to avoid oversubscription)"
        ),
    )
    parser.add_argument(
        "--keep-work",
        action="store_true",
        help="keep temporary work dirs under outdir/.pipeline_work",
    )
    parser.add_argument(
        "--evaluate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "after INV/TRA, score HiC-SCI into phasc_final_report.txt "
            "(default: --evaluate; no SCI/ folder)"
        ),
    )
    parser.add_argument(
        "--evaluate-penalty-scale",
        default="5000,500",
        help="evaluate --penalty-scale inv,tra (default: 5000,500)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.jobs < 1:
        parser.error("--jobs must be >= 1")
    if args.chrom_jobs < 0:
        parser.error("--chrom-jobs must be >= 0")
    if args.inv_min_support < 1:
        parser.error("--inv-min-support must be >= 1")

    shared = Path(args.mcool).expanduser().resolve() if args.mcool else None
    inv_mcool_arg = args.inv_mcool or args.mcool
    tra_mcool_arg = args.tra_mcool or args.mcool
    if not inv_mcool_arg or not tra_mcool_arg:
        parser.error(
            "provide a positional mcool, or both --inv-mcool and --tra-mcool"
        )

    inv_mcool = Path(inv_mcool_arg).expanduser().resolve()
    tra_mcool = Path(tra_mcool_arg).expanduser().resolve()
    for label, path in (("INV", inv_mcool), ("TRA", tra_mcool)):
        if not path.is_file():
            raise SystemExit(f"[ERROR] {label} mcool not found: {path}")

    for name in ("fasta", "hic_alignments", "truth"):
        value = getattr(args, name)
        if value and not Path(value).expanduser().is_file():
            raise SystemExit(f"[ERROR] file not found: {value}")

    base_dir = Path(__file__).resolve().parent
    inv_script = base_dir / "multiscale_inversion.py"
    tra_script = base_dir / "multiscale_intra_translocation.py"
    for script in (inv_script, tra_script):
        if not script.is_file():
            raise SystemExit(f"[ERROR] missing script: {script}")

    outdir = Path(args.outdir).expanduser().resolve()
    work_root = outdir / ".pipeline_work"
    inv_work = work_root / "inversion"
    tra_root = work_root / "intra_translocation"
    if work_root.exists():
        shutil.rmtree(work_root)
    inv_work.mkdir(parents=True)
    tra_root.mkdir(parents=True)
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    chroms = _resolve_chromosomes(args, tra_mcool)
    n_chroms = len(chroms)
    cpu = os.cpu_count() or 4
    if args.chrom_jobs == 0:
        chrom_jobs = min(n_chroms, max(1, cpu // 2))
    else:
        chrom_jobs = max(1, min(args.chrom_jobs, n_chroms))
    args.chrom_jobs_effective = chrom_jobs
    jobs_per_chrom = max(1, int(args.jobs) // chrom_jobs) if chrom_jobs > 1 else int(args.jobs)

    balance_flag = "--balance" if args.balance else "--no-balance"

    inv_cmd = [
        sys.executable,
        str(inv_script),
        str(inv_mcool),
        balance_flag,
        "--outdir",
        str(inv_work),
        "--jobs",
        str(args.jobs),
        "--min-support",
        str(args.inv_min_support),
    ]
    if args.chromosomes:
        inv_cmd.extend(["--chromosomes", str(args.chromosomes)])
    if args.fasta:
        inv_cmd.extend(["--fasta", str(Path(args.fasta).expanduser().resolve())])
    if args.hic_alignments:
        inv_cmd.extend(
            [
                "--hic-alignments",
                str(Path(args.hic_alignments).expanduser().resolve()),
            ]
        )

    chrom_works: dict[str, Path] = {}
    tra_cmds: dict[str, list[str]] = {}
    for chrom in chroms:
        safe = chrom.replace("/", "_")
        chrom_work = tra_root / safe
        chrom_work.mkdir(parents=True, exist_ok=True)
        chrom_works[chrom] = chrom_work
        tra_cmds[chrom] = _build_tra_cmd(
            tra_script=tra_script,
            tra_mcool=tra_mcool,
            outdir=chrom_work,
            chrom=chrom,
            jobs=jobs_per_chrom,
            args=args,
        )

    print(
        f"[INFO] pipeline outdir={outdir} jobs={args.jobs} "
        f"balance={bool(args.balance)}",
        flush=True,
    )
    print(
        f"[INFO] INV+TRA chromosomes ({n_chroms}): {', '.join(chroms)}",
        flush=True,
    )
    print(
        f"[INFO] TRA chrom-jobs={chrom_jobs} jobs/chrom={jobs_per_chrom}",
        flush=True,
    )
    print(f"[INFO] INV mcool={inv_mcool}", flush=True)
    print(f"[INFO] TRA mcool={tra_mcool}", flush=True)
    if shared is not None and inv_mcool == tra_mcool == shared:
        print("[INFO] using shared mcool for both detectors", flush=True)
    print(
        "[INFO] running inversion + per-chromosome intra-TRA "
        f"(max {chrom_jobs} chrom workers)",
        flush=True,
    )

    # INV + up to chrom_jobs TRA workers at once
    max_workers = 1 + chrom_jobs
    errors: list[tuple[str, BaseException]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_labeled, inv_cmd, "INV"): "inversion",
        }
        for chrom, cmd in tra_cmds.items():
            futures[
                executor.submit(_run_labeled, cmd, f"TRA:{chrom}")
            ] = f"intra-translocation:{chrom}"

        for future in as_completed(futures):
            label = futures[future]
            try:
                future.result()
            except BaseException as exc:  # noqa: BLE001 - surface both failures
                errors.append((label, exc))

    if errors:
        for label, exc in errors:
            print(f"[ERROR] {label} failed: {exc}", flush=True)
        if not args.keep_work:
            print(
                f"[INFO] retaining work dir for debugging: {work_root}",
                flush=True,
            )
        return 1

    # Final layout: ALL_SV.tsv + phasc_final_report.txt + HiC_INV_Plot/ + HiC_TRA_Plot/
    promoted: list[str] = []
    inv_tsv = inv_work / "ALL_INV_multiscale.tsv"
    if not inv_tsv.is_file():
        inv_tsv = None
    tra_tsv = work_root / "ALL_TRA.merged.tsv"
    if not _merge_tra_tables(chrom_works, tra_tsv, tra_mcool=tra_mcool):
        tra_tsv = None

    combined = _merge_sv_tables(outdir, inv_tsv=inv_tsv, tra_tsv=tra_tsv)
    if combined is not None:
        promoted.append(combined.name)
        print(f"[DONE] ALL_SV.tsv -> {combined}", flush=True)

    promoted.extend(_promote_inv_plots(inv_work, outdir))
    promoted.extend(_collect_tra_plots(chrom_works, outdir))

    summary = None
    n_inv = int(len(_read_tsv(inv_tsv))) if inv_tsv else 0
    n_tra = int(len(_read_tsv(tra_tsv))) if tra_tsv else 0
    if args.evaluate:
        try:
            sys.path.insert(0, str(base_dir))
            from evaluate import score_sci_from_calls

            # Score into a throwaway dir; phasc_final_report.txt is user-facing.
            sci_tmp = work_root / "_sci_tmp"
            chrom_list = chroms if args.chromosomes else None
            summary = score_sci_from_calls(
                mcool=str(inv_mcool),
                calls_dir=outdir,
                sci_outdir=sci_tmp,
                inv_tsv=inv_tsv,
                tra_tsv=tra_tsv,
                penalty_scale=args.evaluate_penalty_scale,
                chromosomes=chrom_list,
            )
            print(
                f"[DONE] HiC-SCI={summary.get('total_score_0_100')} "
                f"(INV={summary.get('component_scores_0_100', {}).get('inversion_score')}, "
                f"TRA={summary.get('component_scores_0_100', {}).get('translocation_score')})",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 — detection already succeeded
            print(f"[WARN] evaluate scoring skipped: {exc}", flush=True)

    report = _write_final_report(
        outdir,
        start_epoch=t0,
        wall_sec=time.time() - t0,
        inv_mcool=inv_mcool,
        tra_mcool=tra_mcool,
        summary=summary,
        n_inv=n_inv,
        n_tra=n_tra,
        jobs=int(args.jobs),
        chrom_jobs=int(chrom_jobs),
        chromosomes=list(chroms),
    )
    promoted.append(report.name)
    print(f"[DONE] report -> {report}", flush=True)

    # Drop leftover audit tables if any prior run left them in outdir.
    for junk in (
        "ALL_INV_multiscale.tsv",
        "ALL_TRA.tsv",
        "ALL_TRA_pre_enrichment.tsv",
        "ALL_TRA_enrichment_dropped.tsv",
        "ALL_TRA_per_scale_candidates.tsv",
        "merged_true_calls.tsv",
        "mutual_dropped_all.tsv",
        "mutual_pairs_all.tsv",
        "per_resolution_summary.tsv",
        "HiC_images",
        "HiC_images_flipped",
        "SCI",
        "tra_by_chrom",
        "runtime.txt",
        "report.txt",
    ):
        p = outdir / junk
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.is_file():
            p.unlink(missing_ok=True)

    if not args.keep_work:
        shutil.rmtree(work_root, ignore_errors=True)

    print(f"[DONE] pipeline results -> {outdir}", flush=True)
    if promoted:
        print("[DONE] kept: " + ", ".join(promoted), flush=True)
    else:
        print("[WARN] no result artifacts found", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

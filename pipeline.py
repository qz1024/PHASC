#!/usr/bin/env python3
"""Run inversion and translocation detection in parallel into one outdir.

Both detectors write to private work directories first (TRA would otherwise
wipe INV's HiC_images/), then final artifacts are promoted into --outdir.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


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


def _promote_inversion(work: Path, outdir: Path) -> list[str]:
    promoted: list[str] = []
    for name in ("ALL_INV_multiscale.tsv", "HiC_images", "HiC_images_flipped"):
        src = work / name
        if src.exists():
            _promote(src, outdir / name)
            promoted.append(name)
    return promoted


def _promote_translocation(work: Path, outdir: Path) -> list[str]:
    promoted: list[str] = []
    names = (
        "ALL_TRA.tsv",
        "merged_true_calls.tsv",
        "ALL_TRA_per_scale_candidates.tsv",
        "per_resolution_summary.tsv",
        "mutual_dropped_all.tsv",
        "mutual_pairs_all.tsv",
        "community",
        "HiC",
    )
    for name in names:
        src = work / name
        if src.exists():
            _promote(src, outdir / name)
            promoted.append(name)
    return promoted


def _read_tsv(path: Path):
    import pandas as pd

    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def _merge_sv_tables(outdir: Path) -> Path | None:
    """Write a combined ALL_SV.tsv with sv_type=INV/TRA and shared locus columns."""
    import pandas as pd

    frames: list[pd.DataFrame] = []

    inv = _read_tsv(outdir / "ALL_INV_multiscale.tsv")
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
                "source_file": "ALL_INV_multiscale.tsv",
            }
        )
        for col in (
            "core_start",
            "core_end",
            "representative_resolution",
            "max_signal",
            "left_scale_support",
            "right_scale_support",
            "scale_support",
            "resolutions",
            "event_score",
        ):
            if col in inv.columns:
                row[col] = inv[col].values
        frames.append(row)

    tra = _read_tsv(outdir / "ALL_TRA.tsv")
    if tra.empty:
        tra = _read_tsv(outdir / "merged_true_calls.tsv")
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
                "source_file": "ALL_TRA.tsv",
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
            "mutual_eval_keep",
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


def _first_chrom_from_mcool(mcool: Path) -> str:
    import cooler

    store = cooler.fileops.list_coolers(str(mcool))
    if not store:
        raise SystemExit(f"[ERROR] no cooler resolutions in {mcool}")
    uri = store[0] if store[0].startswith(str(mcool)) else f"{mcool}::{store[0]}"
    chroms = list(cooler.Cooler(uri).chromnames)
    if not chroms:
        raise SystemExit(f"[ERROR] no chromosomes in {mcool}")
    return str(chroms[0])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run PHASC inversion and translocation detection in parallel "
            "and write both results into the same --outdir"
        )
    )
    parser.add_argument("mcool", help="input .mcool file")
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
        help="passed to both detectors as --jobs (default: min(4, CPUs))",
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
            "chromosome for translocation (--chrom); if omitted, use the "
            "first chromosome in the mcool. Only the first name is used."
        ),
    )
    parser.add_argument(
        "--truth",
        default=None,
        help="optional TRA truth BED passed to translocation --truth",
    )
    parser.add_argument(
        "--tra-skip-plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="skip TRA community plots (default: --tra-skip-plot)",
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
        "--tra-threads",
        type=int,
        default=0,
        help=(
            "NetworKit threads for TRA PLM (default: 0 = auto cpu/2; "
            "passed as translocation --threads)"
        ),
    )
    parser.add_argument(
        "--keep-work",
        action="store_true",
        help="keep temporary work dirs under outdir/.pipeline_work",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.jobs < 1:
        parser.error("--jobs must be >= 1")
    if args.inv_min_support < 1:
        parser.error("--inv-min-support must be >= 1")

    mcool = Path(args.mcool).expanduser().resolve()
    if not mcool.is_file():
        raise SystemExit(f"[ERROR] mcool not found: {mcool}")

    for name in ("fasta", "hic_alignments", "truth"):
        value = getattr(args, name)
        if value and not Path(value).expanduser().is_file():
            raise SystemExit(f"[ERROR] file not found: {value}")

    base_dir = Path(__file__).resolve().parent
    inv_script = base_dir / "multiscale_inversion.py"
    tra_script = base_dir / "multiscale_translocation.py"
    for script in (inv_script, tra_script):
        if not script.is_file():
            raise SystemExit(f"[ERROR] missing script: {script}")

    outdir = Path(args.outdir).expanduser().resolve()
    work_root = outdir / ".pipeline_work"
    inv_work = work_root / "inversion"
    tra_work = work_root / "translocation"
    if work_root.exists():
        shutil.rmtree(work_root)
    inv_work.mkdir(parents=True)
    tra_work.mkdir(parents=True)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.chromosomes:
        chrom = args.chromosomes.split(",")[0].strip()
    else:
        chrom = _first_chrom_from_mcool(mcool)

    balance_flag = "--balance" if args.balance else "--no-balance"

    inv_cmd = [
        sys.executable,
        str(inv_script),
        str(mcool),
        balance_flag,
        "--outdir",
        str(inv_work),
        "--jobs",
        str(args.jobs),
        "--min-support",
        str(args.inv_min_support),
    ]
    if args.fasta:
        inv_cmd.extend(["--fasta", str(Path(args.fasta).expanduser().resolve())])
    if args.hic_alignments:
        inv_cmd.extend(
            [
                "--hic-alignments",
                str(Path(args.hic_alignments).expanduser().resolve()),
            ]
        )

    tra_cmd = [
        sys.executable,
        str(tra_script),
        "--mcool",
        str(mcool),
        "--outdir",
        str(tra_work),
        "--chrom",
        chrom,
        "--jobs",
        str(args.jobs),
    ]
    if args.tra_skip_plot:
        tra_cmd.append("--skip-plot")
    if args.truth:
        tra_cmd.extend(["--truth", str(Path(args.truth).expanduser().resolve())])
    if args.tra_resolutions:
        tra_cmd.extend(["--resolutions", args.tra_resolutions])
    if args.tra_gammas:
        tra_cmd.extend(["--gammas", args.tra_gammas])
    tra_cmd.extend(["--threads", str(args.tra_threads)])

    print(
        f"[INFO] pipeline outdir={outdir} jobs={args.jobs} "
        f"balance={bool(args.balance)} chrom={chrom}",
        flush=True,
    )
    print("[INFO] running inversion + translocation in parallel", flush=True)

    errors: list[tuple[str, BaseException]] = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(_run_labeled, inv_cmd, "INV"): "inversion",
            executor.submit(_run_labeled, tra_cmd, "TRA"): "translocation",
        }
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

    promoted = []
    promoted.extend(_promote_inversion(inv_work, outdir))
    promoted.extend(_promote_translocation(tra_work, outdir))

    # Ensure ALL_TRA.tsv exists even if only merged_true_calls was written
    all_tra = outdir / "ALL_TRA.tsv"
    merged_true = outdir / "merged_true_calls.tsv"
    if not all_tra.exists() and merged_true.exists():
        shutil.copy2(merged_true, all_tra)
        promoted.append("ALL_TRA.tsv")

    combined = _merge_sv_tables(outdir)
    if combined is not None:
        promoted.append(combined.name)
        print(f"[DONE] combined SV table -> {combined}", flush=True)

    if not args.keep_work:
        shutil.rmtree(work_root, ignore_errors=True)

    print(f"[DONE] pipeline results -> {outdir}", flush=True)
    if promoted:
        print("[DONE] promoted: " + ", ".join(promoted), flush=True)
    else:
        print("[WARN] no result artifacts found to promote", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

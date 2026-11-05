#!/usr/bin/env python3
"""Preprocess Hi-C reads into genome-wide mcool files.

The workflow is:
1. Create a chrom.sizes file from the reference FASTA.
2. Map paired Hi-C reads once with chromap at MAPQ 0 and write a .pairs file.
3. Filter that pairs file to the INV / TRA MAPQ thresholds.
4. Convert each filtered .pairs into a genome-wide .mcool with cooler.

Defaults: INV keeps MAPQ >= 0 (all pairs); TRA keeps MAPQ >= 1.
Filtering requires chromap pairs that include mapq1/mapq2 columns.
"""

from __future__ import annotations

import argparse
import gzip
import os
import shlex
import shutil
import subprocess
from pathlib import Path


DEFAULT_RESOLUTIONS = "10000,25000,50000,100000,250000,500000,1000000"
# chromap mapping always keeps everything; per-detector thresholds are applied
# afterwards by filtering the MAPQ-0 pairs file.
CHROMAP_MAPQ = 0


def parse_resolution_list(value: str) -> list[int]:
    resolutions = []
    for item in value.replace(" ", "").split(","):
        if not item:
            continue
        try:
            resolution = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid resolution: {item!r}"
            ) from exc
        if resolution <= 0:
            raise argparse.ArgumentTypeError("resolutions must be positive")
        resolutions.append(resolution)
    if not resolutions:
        raise argparse.ArgumentTypeError("at least one resolution is required")
    return sorted(set(resolutions))


def open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return path.open("rt", encoding="utf-8")


def open_text_write(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "wt")
    return path.open("wt", encoding="utf-8")


def fasta_to_chrom_sizes(fasta: Path, chrom_sizes: Path) -> list[tuple[str, int]]:
    records: list[tuple[str, int]] = []
    current_name = None
    current_size = 0

    with open_text(fasta) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_name is not None:
                    records.append((current_name, current_size))
                current_name = line[1:].split()[0]
                current_size = 0
            else:
                current_size += len(line)

    if current_name is not None:
        records.append((current_name, current_size))
    if not records:
        raise SystemExit(f"no FASTA records found in {fasta}")

    chrom_sizes.parent.mkdir(parents=True, exist_ok=True)
    with chrom_sizes.open("w", encoding="utf-8") as handle:
        for name, size in records:
            handle.write(f"{name}\t{size}\n")
    return records


def require_command(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(
            f"required command not found: {name}\n"
            f"Please install {name} and make sure it is available in PATH."
        )


def run_command(command: list[str], label: str, dry_run: bool = False) -> None:
    print(f"[START] {label}", flush=True)
    print("  " + shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)
    print(f"[DONE] {label}", flush=True)


def cooler_cload_pairs_command(
    chrom_sizes: Path,
    resolution: int,
    pairs: Path,
    cool: Path,
    zero_based: bool,
) -> list[str]:
    command = [
        "cooler",
        "cload",
        "pairs",
        "-c1",
        "2",
        "-p1",
        "3",
        "-c2",
        "4",
        "-p2",
        "5",
    ]
    if zero_based:
        command.append("--zero-based")
    command.extend([
        f"{chrom_sizes}:{resolution}",
        str(pairs),
        str(cool),
    ])
    return command


def balance_command(cool: Path, threads: int) -> list[str]:
    return ["cooler", "balance", "--nproc", str(threads), str(cool)]


def zoomify_command(
    cool: Path,
    mcool: Path,
    resolutions: list[int],
    threads: int,
    balance: bool,
) -> list[str]:
    command = [
        "cooler",
        "zoomify",
        "-n",
        str(threads),
        "-r",
        ",".join(str(value) for value in resolutions),
    ]
    if balance:
        command.append("--balance")
    command.extend(["-o", str(mcool), str(cool)])
    return command


def chromap_map_command(
    reference: Path,
    chromap_index: Path,
    read1: str,
    read2: str,
    pairs_path: Path,
    threads: int,
    chromap_extra: str,
) -> list[str]:
    command = [
        "chromap",
        "--preset",
        "hic",
        "-x",
        str(chromap_index),
        "-r",
        str(reference),
        "-1",
        str(read1),
        "-2",
        str(read2),
        "-o",
        str(pairs_path),
        "-t",
        str(threads),
        "-q",
        str(CHROMAP_MAPQ),
    ]
    if chromap_extra:
        command.extend(shlex.split(chromap_extra))
    return command


def parse_pairs_columns(pairs_path: Path) -> list[str] | None:
    """Return column names from a pairs `#columns:` header, if present."""
    with open_text(pairs_path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if not line.startswith("#"):
                break
            if line.lower().startswith("#columns:"):
                return line.split(":", 1)[1].strip().split()
    return None


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.resolve() == src.resolve():
        return
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def filter_pairs_by_mapq(
    src_pairs: Path,
    dst_pairs: Path,
    min_mapq: int,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Keep contacts with mapq1 >= min_mapq and mapq2 >= min_mapq.

    Returns (n_kept, n_total_body_rows).
    """
    if min_mapq <= CHROMAP_MAPQ:
        print(
            f"[INFO] MAPQ>={min_mapq}: reuse full pairs {src_pairs.name}",
            flush=True,
        )
        if not dry_run:
            link_or_copy(src_pairs, dst_pairs)
        return (-1, -1)

    print(
        f"[START] filter pairs MAPQ>={min_mapq}: {src_pairs} -> {dst_pairs}",
        flush=True,
    )
    if dry_run:
        print(
            f"  (dry-run) keep rows with mapq1>={min_mapq} and mapq2>={min_mapq}",
            flush=True,
        )
        print(f"[DONE] filter pairs MAPQ>={min_mapq}", flush=True)
        return (-1, -1)

    columns = parse_pairs_columns(src_pairs)
    if not columns:
        raise SystemExit(
            f"[ERROR] {src_pairs} has no #columns header; cannot filter by MAPQ"
        )
    lower = {name.lower(): idx for idx, name in enumerate(columns)}
    if "mapq1" not in lower or "mapq2" not in lower:
        raise SystemExit(
            f"[ERROR] {src_pairs} has no mapq1/mapq2 columns "
            f"(found: {', '.join(columns)}).\n"
            "Chromap must write MAPQ into the pairs file so TRA/INV thresholds "
            "can be filtered from a single MAPQ-0 mapping.\n"
            "Use a chromap build that emits "
            "'#columns: ... pair_type mapq1 mapq2' "
            "(upstream commit: Add the mapq1,mapq2 fields to the pairs format output)."
        )
    idx1 = lower["mapq1"]
    idx2 = lower["mapq2"]

    n_total = 0
    n_kept = 0
    with open_text(src_pairs) as hin, open_text_write(dst_pairs) as hout:
        for raw_line in hin:
            if raw_line.startswith("#") or not raw_line.strip():
                hout.write(raw_line)
                continue
            n_total += 1
            fields = raw_line.rstrip("\n").split("\t")
            try:
                mq1 = int(fields[idx1])
                mq2 = int(fields[idx2])
            except (IndexError, ValueError) as exc:
                raise SystemExit(
                    f"[ERROR] bad pairs row while filtering {src_pairs}: {raw_line!r}"
                ) from exc
            if mq1 >= min_mapq and mq2 >= min_mapq:
                hout.write(raw_line)
                n_kept += 1

    print(
        f"[DONE] filter pairs MAPQ>={min_mapq}: kept {n_kept}/{n_total}",
        flush=True,
    )
    return (n_kept, n_total)


def build_mcool_from_pairs(
    *,
    label: str,
    chrom_sizes: Path,
    pairs_path: Path,
    cool_path: Path,
    mcool_path: Path,
    base_resolution: int,
    resolutions: list[int],
    threads: int,
    zero_based: bool,
    no_balance: bool,
    dry_run: bool,
) -> None:
    run_command(
        cooler_cload_pairs_command(
            chrom_sizes, base_resolution, pairs_path, cool_path, zero_based
        ),
        f"create {label} cool",
        dry_run,
    )
    if not no_balance:
        run_command(
            balance_command(cool_path, threads),
            f"balance {label} cool",
            dry_run,
        )
    run_command(
        zoomify_command(
            cool_path, mcool_path, resolutions, threads, not no_balance
        ),
        f"create {label} mcool",
        dry_run,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Map Hi-C reads once with chromap (-q 0), filter pairs by INV/TRA "
            "MAPQ, and generate two mcool inputs for PHASC."
        )
    )
    parser.add_argument("-r", "--reference", required=True, help="reference genome FASTA")
    parser.add_argument("-1", "--read1", help="Hi-C read 1 FASTQ/FASTQ.GZ")
    parser.add_argument("-2", "--read2", help="Hi-C read 2 FASTQ/FASTQ.GZ")
    parser.add_argument("-o", "--outdir", default="PHASC_preprocess", help="output directory")
    parser.add_argument("-p", "--prefix", default="hic", help="output prefix")
    parser.add_argument("-t", "--threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument(
        "--resolutions",
        type=parse_resolution_list,
        default=parse_resolution_list(DEFAULT_RESOLUTIONS),
        help=f"comma-separated mcool resolutions (default: {DEFAULT_RESOLUTIONS})",
    )
    parser.add_argument(
        "--base-resolution",
        type=int,
        default=None,
        help="base .cool resolution; default is the smallest requested resolution",
    )
    parser.add_argument(
        "--chromap-index",
        default=None,
        help="chromap index path; created automatically if absent",
    )
    parser.add_argument(
        "--pairs",
        default=None,
        help=(
            "chromap MAPQ-0 .pairs path (output, or input with --skip-mapping). "
            "INV/TRA pairs are filtered from this file."
        ),
    )
    parser.add_argument(
        "--skip-mapping",
        action="store_true",
        help="reuse --pairs and skip chromap mapping",
    )
    parser.add_argument(
        "--inv-mapq",
        type=int,
        default=0,
        help="keep pairs with mapq1/mapq2 >= this for inversion mcool (default: 0)",
    )
    parser.add_argument(
        "--tra-mapq",
        type=int,
        default=1,
        help="keep pairs with mapq1/mapq2 >= this for translocation mcool (default: 1)",
    )
    parser.add_argument(
        "--chromap-extra",
        default="",
        help="extra options passed to chromap, quoted as one string",
    )
    parser.add_argument(
        "--zero-based",
        action="store_true",
        help="pass --zero-based to cooler cload pairs if pair positions are 0-based",
    )
    parser.add_argument(
        "--no-balance",
        action="store_true",
        help="do not run cooler balance or balanced zoomify",
    )
    parser.add_argument("--dry-run", action="store_true", help="print commands without running them")
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    reference = Path(args.reference).expanduser()
    if not reference.is_file():
        parser.error(f"reference FASTA does not exist: {reference}")
    if args.threads < 1:
        parser.error("--threads must be >= 1")
    if args.base_resolution is not None and args.base_resolution <= 0:
        parser.error("--base-resolution must be positive")
    for name in ("inv_mapq", "tra_mapq"):
        value = getattr(args, name)
        if value < 0 or value > 60:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 60]")

    if args.skip_mapping:
        if not args.pairs:
            parser.error("--skip-mapping requires --pairs")
        if not Path(args.pairs).expanduser().is_file():
            parser.error(f"pairs file does not exist: {args.pairs}")
    else:
        if not args.read1 or not args.read2:
            parser.error("mapping requires both --read1 and --read2")
        for reads in (args.read1, args.read2):
            for item in str(reads).split(","):
                if item and not Path(item).expanduser().is_file():
                    parser.error(f"read file does not exist: {item}")


def run_preprocess(args: argparse.Namespace) -> None:
    if not args.dry_run:
        require_command("cooler")
    if not args.skip_mapping and not args.dry_run:
        require_command("chromap")

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix
    reference = Path(args.reference).expanduser().resolve()

    resolutions = sorted(set(args.resolutions))
    base_resolution = args.base_resolution or min(resolutions)
    if base_resolution not in resolutions:
        resolutions = sorted([base_resolution, *resolutions])

    chrom_sizes = outdir / f"{prefix}.chrom.sizes"
    fasta_to_chrom_sizes(reference, chrom_sizes)
    chromap_index = (
        Path(args.chromap_index).expanduser().resolve()
        if args.chromap_index
        else outdir / f"{prefix}.chromap.index"
    )

    inv_mapq = int(args.inv_mapq)
    tra_mapq = int(args.tra_mapq)
    inv_tag = f"inv.mapq{inv_mapq}"
    tra_tag = f"tra.mapq{tra_mapq}"

    base_pairs = (
        Path(args.pairs).expanduser().resolve()
        if args.pairs
        else outdir / f"{prefix}.mapq{CHROMAP_MAPQ}.pairs"
    )
    inv_pairs = outdir / f"{prefix}.{inv_tag}.pairs"
    tra_pairs = outdir / f"{prefix}.{tra_tag}.pairs"

    if not args.skip_mapping:
        if not chromap_index.exists():
            run_command(
                [
                    "chromap",
                    "-i",
                    "-r",
                    str(reference),
                    "-o",
                    str(chromap_index),
                ],
                "build chromap index",
                args.dry_run,
            )
        else:
            print(f"[SKIP] chromap index exists: {chromap_index}", flush=True)

        run_command(
            chromap_map_command(
                reference,
                chromap_index,
                args.read1,
                args.read2,
                base_pairs,
                args.threads,
                args.chromap_extra,
            ),
            f"map Hi-C reads with chromap (MAPQ>={CHROMAP_MAPQ})",
            args.dry_run,
        )
    else:
        print(f"[SKIP] using existing pairs: {base_pairs}", flush=True)

    # Filter MAPQ-0 pairs down to INV / TRA thresholds (no second chromap run).
    filter_pairs_by_mapq(base_pairs, inv_pairs, inv_mapq, args.dry_run)
    filter_pairs_by_mapq(base_pairs, tra_pairs, tra_mapq, args.dry_run)

    targets = (
        ("INV", inv_tag, inv_pairs, inv_mapq),
        ("TRA", tra_tag, tra_pairs, tra_mapq),
    )
    built: dict[str, tuple[Path, Path, Path]] = {}
    artifacts_by_mapq: dict[int, tuple[Path, Path, Path]] = {}

    for role, tag, pairs_path, mapq in targets:
        cool_path = outdir / f"{prefix}.{tag}.{base_resolution}.cool"
        mcool_path = outdir / f"{prefix}.{tag}.mcool"

        if mapq in artifacts_by_mapq:
            src_pairs, src_cool, src_mcool = artifacts_by_mapq[mapq]
            print(
                f"[INFO] {role} reuses cool/mcool for MAPQ>={mapq} "
                f"(from {src_mcool.name})",
                flush=True,
            )
            if not args.dry_run:
                link_or_copy(src_cool, cool_path)
                link_or_copy(src_mcool, mcool_path)
            pairs_path = src_pairs
        else:
            build_mcool_from_pairs(
                label=f"{role} MAPQ>={mapq}",
                chrom_sizes=chrom_sizes,
                pairs_path=pairs_path,
                cool_path=cool_path,
                mcool_path=mcool_path,
                base_resolution=base_resolution,
                resolutions=resolutions,
                threads=args.threads,
                zero_based=args.zero_based,
                no_balance=args.no_balance,
                dry_run=args.dry_run,
            )
            artifacts_by_mapq[mapq] = (pairs_path, cool_path, mcool_path)

        built[role] = (pairs_path, cool_path, mcool_path)

    # Compatibility alias: prefix.mcool -> INV mcool
    alias_mcool = outdir / f"{prefix}.mcool"
    inv_mcool = built["INV"][2]
    if not args.dry_run:
        if alias_mcool.exists() or alias_mcool.is_symlink():
            alias_mcool.unlink()
        try:
            os.symlink(inv_mcool.name, alias_mcool)
        except OSError:
            shutil.copy2(inv_mcool, alias_mcool)

    manifest_rows = [
        ["type", "mapq", "pairs", "cool", "mcool"],
        [
            "base",
            str(CHROMAP_MAPQ),
            str(base_pairs),
            "",
            "",
        ],
        [
            "INV",
            str(inv_mapq),
            str(built["INV"][0]),
            str(built["INV"][1]),
            str(built["INV"][2]),
        ],
        [
            "TRA",
            str(tra_mapq),
            str(built["TRA"][0]),
            str(built["TRA"][1]),
            str(built["TRA"][2]),
        ],
    ]

    manifest = outdir / f"{prefix}.preprocess_manifest.tsv"
    if not args.dry_run:
        with manifest.open("w", encoding="utf-8") as handle:
            for row in manifest_rows:
                handle.write("\t".join(row) + "\n")

    print("\nPreprocess finished.", flush=True)
    print(f"Base pairs (chromap -q {CHROMAP_MAPQ}): {base_pairs}", flush=True)
    print(f"INV mcool (MAPQ>={inv_mapq}): {built['INV'][2]}", flush=True)
    print(f"TRA mcool (MAPQ>={tra_mapq}): {built['TRA'][2]}", flush=True)
    print(f"Alias mcool (INV): {alias_mcool}", flush=True)
    print(f"Manifest: {manifest}", flush=True)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)
    try:
        run_preprocess(args)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc


if __name__ == "__main__":
    main()

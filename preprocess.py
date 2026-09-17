#!/usr/bin/env python3
"""Preprocess Hi-C reads into a genome-wide mcool file.

The workflow is:
1. Create a chrom.sizes file from the reference FASTA.
2. Map paired Hi-C reads with chromap and write a .pairs file.
3. Convert .pairs to a genome-wide .mcool with cooler.
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Map Hi-C reads with chromap and generate mcool inputs for PHASC."
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
        help="existing or output .pairs path; with --skip-mapping this file is reused",
    )
    parser.add_argument(
        "--skip-mapping",
        action="store_true",
        help="reuse --pairs and skip chromap mapping",
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
    chromap_index = Path(args.chromap_index).expanduser().resolve() if args.chromap_index else outdir / f"{prefix}.chromap.index"
    pairs_path = Path(args.pairs).expanduser().resolve() if args.pairs else outdir / f"{prefix}.pairs"

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

        command = [
            "chromap",
            "--preset",
            "hic",
            "-x",
            str(chromap_index),
            "-r",
            str(reference),
            "-1",
            str(args.read1),
            "-2",
            str(args.read2),
            "-o",
            str(pairs_path),
            "-t",
            str(args.threads),
        ]
        if args.chromap_extra:
            command.extend(shlex.split(args.chromap_extra))
        run_command(command, "map Hi-C reads with chromap", args.dry_run)
    else:
        print(f"[SKIP] using existing pairs: {pairs_path}", flush=True)

    full_cool = outdir / f"{prefix}.{base_resolution}.cool"
    full_mcool = outdir / f"{prefix}.mcool"
    run_command(
        cooler_cload_pairs_command(
            chrom_sizes, base_resolution, pairs_path, full_cool, args.zero_based
        ),
        "create genome-wide cool",
        args.dry_run,
    )
    if not args.no_balance:
        run_command(balance_command(full_cool, args.threads), "balance genome-wide cool", args.dry_run)
    run_command(
        zoomify_command(
            full_cool, full_mcool, resolutions, args.threads, not args.no_balance
        ),
        "create genome-wide mcool",
        args.dry_run,
    )

    manifest_rows = [
        ["type", "pairs", "cool", "mcool"],
        ["genome", str(pairs_path), str(full_cool), str(full_mcool)],
    ]

    manifest = outdir / f"{prefix}.preprocess_manifest.tsv"
    if not args.dry_run:
        with manifest.open("w", encoding="utf-8") as handle:
            for row in manifest_rows:
                handle.write("\t".join(row) + "\n")

    print("\nPreprocess finished.", flush=True)
    print(f"Genome-wide mcool: {full_mcool}", flush=True)
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

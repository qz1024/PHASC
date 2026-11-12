#!/usr/bin/env python3
"""Correct scaffold mounting errors inferred from PHASC INV / TRA calls.

Scaffolding mistakes (wrong contig orientation / order) often look like
inversions or intra-chromosomal translocations in Hi-C. This script:

  1. splits each chromosome by runs of N (gaps) into contigs
  2. for each INV/TRA interval, classifies overlapping contig cover:
       < --ignore-cover-below (default 10%): tip → drop
       in [ignore, --cover-frac) (default 10–90%): ambiguous → reject call
       >= --cover-frac (default 90%): keep → correct these contigs
  3. if kept contigs remain, applies a contig-level correction:
       INV: reverse order of spanned pieces + reverse-complement contig seqs
       TRA: move candidate contig block next to the partner block

Inputs: genome FASTA + ALL_INV_multiscale.tsv / ALL_TRA.tsv (or ALL_SV.tsv).
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

COMP = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")
N_RUN = re.compile(r"[Nn]+")


def revcomp(s: str) -> str:
    return s.translate(COMP)[::-1]


def overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


# -----------------------------------------------------------------------------
# FASTA / contig split
# -----------------------------------------------------------------------------

@dataclass
class Piece:
    kind: str  # "contig" | "gap"
    cid: int | None
    seq: str
    # original coords (0-based half-open), only meaningful before edits
    start: int = 0
    end: int = 0


@dataclass
class ContigView:
    cid: int
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def read_fasta(path: Path) -> dict[str, str]:
    seqs: dict[str, str] = {}
    name: str | None = None
    chunks: list[str] = []
    with path.open() as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    seqs[name] = "".join(chunks)
                name = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    if name is not None:
        seqs[name] = "".join(chunks)
    if not seqs:
        raise SystemExit(f"[ERROR] no sequences in {path}")
    return seqs


def write_fasta(path: Path, seqs: dict[str, str], width: int = 60) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for name, seq in seqs.items():
            f.write(f">{name}\n")
            for i in range(0, len(seq), width):
                f.write(seq[i : i + width] + "\n")


def split_by_n(seq: str, min_gap_n: int = 1) -> tuple[list[Piece], list[ContigView]]:
    """Split sequence into contig/gap pieces on N-runs of length >= min_gap_n."""
    pieces: list[Piece] = []
    contigs: list[ContigView] = []
    cid = 0
    pos = 0
    for m in N_RUN.finditer(seq):
        a, b = m.start(), m.end()
        if b - a < min_gap_n:
            continue
        if a > pos:
            cseq = seq[pos:a]
            pieces.append(Piece("contig", cid, cseq, pos, a))
            contigs.append(ContigView(cid, pos, a))
            cid += 1
        pieces.append(Piece("gap", None, seq[a:b], a, b))
        pos = b
    if pos < len(seq):
        cseq = seq[pos:]
        pieces.append(Piece("contig", cid, cseq, pos, len(seq)))
        contigs.append(ContigView(cid, pos, len(seq)))
        cid += 1
    if not pieces:
        pieces.append(Piece("contig", 0, seq, 0, len(seq)))
        contigs.append(ContigView(0, 0, len(seq)))
    return pieces, contigs


def pieces_to_seq(pieces: list[Piece]) -> str:
    return "".join(p.seq for p in pieces)


# -----------------------------------------------------------------------------
# Coverage gate
# -----------------------------------------------------------------------------

@dataclass
class CoverResult:
    ok: bool
    contig_ids: list[int] = field(default_factory=list)
    fracs: list[float] = field(default_factory=list)
    reason: str = ""


def cover_contigs(
    contigs: list[ContigView],
    start: int,
    end: int,
    cover_frac: float,
    min_overlap_bp: int = 1,
    ignore_cover_below: float = 0.1,
) -> CoverResult:
    """Three-band cover filter for one INV/TRA interval.

    For every contig overlapping the call:
      • cover < ignore_cover_below (default 10%): tip → drop
      • ignore_cover_below ≤ cover < cover_frac (default 10–90%):
        ambiguous → reject the whole INV/TRA
      • cover ≥ cover_frac (default ≥90%): keep → correct these contigs

    Only kept (≥ cover_frac) contig ids are returned on success.
    """
    if end <= start:
        return CoverResult(False, reason="empty_interval")

    tips: list[tuple[int, float]] = []
    mids: list[tuple[int, float]] = []
    fulls: list[tuple[int, float]] = []
    for c in contigs:
        ov = overlap(c.start, c.end, start, end)
        if ov < min_overlap_bp:
            continue
        frac = ov / c.length
        if frac < ignore_cover_below:
            tips.append((c.cid, frac))
        elif frac < cover_frac:
            mids.append((c.cid, frac))
        else:
            fulls.append((c.cid, frac))

    if mids:
        detail = ",".join(f"{cid}:{frac:.3f}" for cid, frac in mids)
        # include mid+full ids so the decision table shows what was considered
        kept = mids + fulls
        return CoverResult(
            False,
            contig_ids=[cid for cid, _ in kept],
            fracs=[frac for _, frac in kept],
            reason=f"ambiguous_cover[{detail}]",
        )

    if not fulls:
        if tips:
            detail = ",".join(f"{cid}:{frac:.3f}" for cid, frac in tips)
            return CoverResult(False, reason=f"tips_only[{detail}]")
        return CoverResult(False, reason="no_contig_overlap")

    reason = "ok"
    if tips:
        detail = ",".join(f"{cid}:{frac:.3f}" for cid, frac in tips)
        reason = f"ok_ignore_tips[{detail}]"
    return CoverResult(
        True,
        contig_ids=[cid for cid, _ in fulls],
        fracs=[frac for _, frac in fulls],
        reason=reason,
    )


# -----------------------------------------------------------------------------
# Piece edits (stable contig ids)
# -----------------------------------------------------------------------------

def _span_indices(pieces: list[Piece], contig_ids: set[int]) -> tuple[int, int] | None:
    """Inclusive piece index range covering contig_ids (and internal gaps)."""
    idxs = [i for i, p in enumerate(pieces) if p.kind == "contig" and p.cid in contig_ids]
    if not idxs:
        return None
    lo, hi = min(idxs), max(idxs)
    # claimed contigs must form a contiguous block in current order
    present = {
        p.cid
        for p in pieces[lo : hi + 1]
        if p.kind == "contig" and p.cid is not None
    }
    if present != contig_ids:
        return None
    return lo, hi


def flip_inv(pieces: list[Piece], contig_ids: list[int]) -> bool:
    span = _span_indices(pieces, set(contig_ids))
    if span is None:
        return False
    lo, hi = span
    block = pieces[lo : hi + 1]
    new_block: list[Piece] = []
    for p in reversed(block):
        if p.kind == "contig":
            new_block.append(Piece("contig", p.cid, revcomp(p.seq)))
        else:
            new_block.append(Piece("gap", None, p.seq))
    pieces[lo : hi + 1] = new_block
    return True


def _merge_adjacent_gaps(pieces: list[Piece]) -> None:
    i = 0
    while i < len(pieces) - 1:
        if pieces[i].kind == "gap" and pieces[i + 1].kind == "gap":
            pieces[i] = Piece("gap", None, pieces[i].seq + pieces[i + 1].seq)
            del pieces[i + 1]
        else:
            i += 1


def move_tra(
    pieces: list[Piece],
    cand_ids: list[int],
    partner_ids: list[int],
) -> bool:
    """Cut candidate contig block; insert immediately before partner block.

    Takes the trailing gap after the candidate (if any) with the move so the
    hole and the destination both keep spacing; merges any adjacent gaps.
    """
    cand_set, part_set = set(cand_ids), set(partner_ids)
    if cand_set & part_set:
        return False
    cspan = _span_indices(pieces, cand_set)
    pspan = _span_indices(pieces, part_set)
    if cspan is None or pspan is None:
        return False
    clo, chi = cspan
    plo, _phi = pspan

    # prefer trailing gap so source keeps its leading spacer
    if chi + 1 < len(pieces) and pieces[chi + 1].kind == "gap":
        chi += 1
    elif clo > 0 and pieces[clo - 1].kind == "gap":
        clo -= 1

    block = pieces[clo : chi + 1]
    del pieces[clo : chi + 1]
    if clo < plo:
        plo -= len(block)

    insert_at = plo
    pieces[insert_at:insert_at] = block
    _merge_adjacent_gaps(pieces)
    return True


# -----------------------------------------------------------------------------
# Call loading / planning
# -----------------------------------------------------------------------------

def _norm_inv(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "start" not in out.columns and "full_start" in out.columns:
        out["start"] = out["full_start"]
    if "end" not in out.columns and "full_end" in out.columns:
        out["end"] = out["full_end"]
    need = {"chrom", "start", "end"}
    missing = need - set(out.columns)
    if missing:
        raise SystemExit(f"[ERROR] INV table missing columns: {sorted(missing)}")
    out["sv_type"] = "INV"
    if "cluster_id" not in out.columns:
        out["cluster_id"] = range(1, len(out) + 1)
    if "scale_support" not in out.columns:
        out["scale_support"] = 1
    if "size" not in out.columns:
        out["size"] = out["end"].astype(int) - out["start"].astype(int)
    return out


def _norm_tra(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    need = {"chrom", "start", "end", "partner_start", "partner_end"}
    missing = need - set(out.columns)
    if missing:
        raise SystemExit(f"[ERROR] TRA table missing columns: {sorted(missing)}")
    out["sv_type"] = "TRA"
    if "cluster_id" not in out.columns:
        out["cluster_id"] = range(1, len(out) + 1)
    if "scale_support" not in out.columns:
        out["scale_support"] = 1
    if "size" not in out.columns:
        out["size"] = out["end"].astype(int) - out["start"].astype(int)
    return out


def load_events(
    inv_tsv: Path | None,
    tra_tsv: Path | None,
    sv_tsv: Path | None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if sv_tsv is not None:
        sv = pd.read_csv(sv_tsv, sep="\t")
        if "sv_type" not in sv.columns:
            raise SystemExit(f"[ERROR] {sv_tsv} needs sv_type column")
        inv = _norm_inv(sv.loc[sv["sv_type"].astype(str).str.upper().eq("INV")].copy())
        tra = sv.loc[sv["sv_type"].astype(str).str.upper().eq("TRA")].copy()
        if not tra.empty:
            tra = _norm_tra(tra)
        frames.extend([x for x in (inv, tra) if not x.empty])
    if inv_tsv is not None:
        frames.append(_norm_inv(pd.read_csv(inv_tsv, sep="\t")))
    if tra_tsv is not None:
        frames.append(_norm_tra(pd.read_csv(tra_tsv, sep="\t")))
    if not frames:
        raise SystemExit("[ERROR] provide --inv, --tra, and/or --sv")
    return pd.concat(frames, ignore_index=True, sort=False)


@dataclass
class Plan:
    chrom: str
    sv_type: str
    cluster_id: object
    start: int
    end: int
    partner_start: int | None
    partner_end: int | None
    contig_ids: list[int]
    partner_contig_ids: list[int]
    cover_fracs: list[float]
    scale_support: float
    size: int
    status: str
    reason: str


def plan_event(
    row: pd.Series,
    contigs: list[ContigView],
    cover_frac: float,
    min_overlap_bp: int,
    ignore_cover_below: float = 0.1,
) -> Plan:
    chrom = str(row["chrom"])
    sv = str(row["sv_type"]).upper()
    start, end = int(row["start"]), int(row["end"])
    support = float(row.get("scale_support", 1) or 1)
    size = int(row.get("size", end - start))
    cid = row.get("cluster_id", ".")

    cov = cover_contigs(
        contigs, start, end, cover_frac, min_overlap_bp, ignore_cover_below
    )
    partner_ids: list[int] = []
    p0 = p1 = None

    if sv == "TRA":
        p0, p1 = int(row["partner_start"]), int(row["partner_end"])
        pcov = cover_contigs(
            contigs, p0, p1, cover_frac, min_overlap_bp, ignore_cover_below
        )
        if not cov.ok:
            return Plan(
                chrom, sv, cid, start, end, p0, p1, cov.contig_ids, [],
                cov.fracs, support, size, "skip", f"cand_{cov.reason}",
            )
        if not pcov.ok:
            return Plan(
                chrom, sv, cid, start, end, p0, p1, cov.contig_ids, pcov.contig_ids,
                cov.fracs, support, size, "skip", f"partner_{pcov.reason}",
            )
        if set(cov.contig_ids) & set(pcov.contig_ids):
            return Plan(
                chrom, sv, cid, start, end, p0, p1, cov.contig_ids, pcov.contig_ids,
                cov.fracs, support, size, "skip", "cand_partner_share_contig",
            )
        partner_ids = pcov.contig_ids
        # keep tip-drop notes from both sides when present
        reason = cov.reason if cov.reason != "ok" else pcov.reason
        if cov.reason.startswith("ok_ignore_tips") and pcov.reason.startswith(
            "ok_ignore_tips"
        ):
            reason = f"cand_{cov.reason};partner_{pcov.reason}"
        return Plan(
            chrom, sv, cid, start, end, p0, p1, cov.contig_ids, partner_ids,
            cov.fracs, support, size, "correct", reason,
        )

    # INV
    if not cov.ok:
        return Plan(
            chrom, sv, cid, start, end, None, None, cov.contig_ids, [],
            cov.fracs, support, size, "skip", cov.reason,
        )
    return Plan(
        chrom, sv, cid, start, end, None, None, cov.contig_ids, [],
        cov.fracs, support, size, "correct", cov.reason,
    )


def resolve_conflicts(plans: list[Plan]) -> list[Plan]:
    """Drop lower-priority plans that share contigs with a kept correct plan."""
    correct = [p for p in plans if p.status == "correct"]
    skipped = [p for p in plans if p.status != "correct"]
    # prefer TRA over INV when equal support? prefer higher support, then larger size
    correct.sort(key=lambda p: (p.scale_support, p.size), reverse=True)
    kept: list[Plan] = []
    used: set[tuple[str, int]] = set()
    for p in correct:
        keys = {(p.chrom, c) for c in p.contig_ids + p.partner_contig_ids}
        if keys & used:
            skipped.append(
                Plan(
                    p.chrom, p.sv_type, p.cluster_id, p.start, p.end,
                    p.partner_start, p.partner_end, p.contig_ids, p.partner_contig_ids,
                    p.cover_fracs, p.scale_support, p.size, "skip", "conflict_shared_contig",
                )
            )
            continue
        used |= keys
        kept.append(p)
    # INV before TRA within a chrom so length-preserving flips happen first
    kept.sort(key=lambda p: (p.chrom, 0 if p.sv_type == "INV" else 1, p.start))
    return kept + skipped


def apply_plans(
    pieces_by_chrom: dict[str, list[Piece]],
    plans: list[Plan],
) -> list[Plan]:
    out: list[Plan] = []
    for p in plans:
        if p.status != "correct":
            out.append(p)
            continue
        pieces = pieces_by_chrom[p.chrom]
        ok = False
        if p.sv_type == "INV":
            ok = flip_inv(pieces, p.contig_ids)
            reason = "applied_inv" if ok else "flip_span_broken"
        else:
            ok = move_tra(pieces, p.contig_ids, p.partner_contig_ids)
            reason = "applied_tra" if ok else "move_span_broken"
        # keep tip-drop note from planning (ok_ignore_tips[...])
        if ok and "ok_ignore_tips" in p.reason:
            reason = f"{reason};{p.reason}"
        out.append(
            Plan(
                p.chrom, p.sv_type, p.cluster_id, p.start, p.end,
                p.partner_start, p.partner_end, p.contig_ids, p.partner_contig_ids,
                p.cover_fracs, p.scale_support, p.size,
                "corrected" if ok else "skip",
                reason,
            )
        )
    return out


def plans_to_frame(plans: list[Plan]) -> pd.DataFrame:
    rows = []
    for p in plans:
        rows.append(
            {
                "chrom": p.chrom,
                "sv_type": p.sv_type,
                "cluster_id": p.cluster_id,
                "start": p.start,
                "end": p.end,
                "partner_start": p.partner_start if p.partner_start is not None else "",
                "partner_end": p.partner_end if p.partner_end is not None else "",
                "contig_ids": ",".join(str(x) for x in p.contig_ids),
                "partner_contig_ids": ",".join(str(x) for x in p.partner_contig_ids),
                "cover_fracs": ",".join(f"{x:.4f}" for x in p.cover_fracs),
                "n_contigs": len(p.contig_ids),
                "scale_support": p.scale_support,
                "size": p.size,
                "status": p.status,
                "reason": p.reason,
            }
        )
    return pd.DataFrame(rows)


def write_contig_bed(path: Path, chrom: str, contigs: list[ContigView]) -> None:
    with path.open("a") as f:
        for c in contigs:
            f.write(f"{chrom}\t{c.start}\t{c.end}\tctg{c.cid}\t{c.length}\n")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Correct contig mounting errors using PHASC INV/TRA calls: "
            "split FASTA by N-gaps; drop tip overlaps < --ignore-cover-below; "
            "reject the call if any overlap is in [ignore, cover); "
            "flip/relocate contigs covered >= --cover-frac."
        )
    )
    p.add_argument("-f", "--fasta", default=None, help="input genome FASTA")
    p.add_argument(
        "-o",
        "--outdir",
        default="CORRECT_results",
        help="output directory (default: CORRECT_results)",
    )
    p.add_argument("--inv", default=None, help="ALL_INV_multiscale.tsv")
    p.add_argument("--tra", default=None, help="ALL_TRA.tsv")
    p.add_argument("--sv", default=None, help="ALL_SV.tsv (alternative to --inv/--tra)")
    p.add_argument(
        "--cover-frac",
        type=float,
        default=0.9,
        help=(
            "contigs covered >= this fraction are corrected (default: 0.9). "
            "Any overlapping contig in [--ignore-cover-below, --cover-frac) "
            "rejects the whole INV/TRA."
        ),
    )
    p.add_argument(
        "--min-overlap-bp",
        type=int,
        default=1,
        help="ignore contig overlaps shorter than this (default: 1)",
    )
    p.add_argument(
        "--ignore-cover-below",
        type=float,
        default=0.1,
        help=(
            "drop tip overlaps below this fraction (default: 0.1). "
            "Overlaps in [this, --cover-frac) discard the event."
        ),
    )
    p.add_argument(
        "--min-gap-n",
        type=int,
        default=1,
        help="min N-run length that splits contigs (default: 1)",
    )
    p.add_argument(
        "--chromosomes",
        default=None,
        help="comma-separated chroms to process (default: all in events)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="only write decision table; do not write corrected FASTA",
    )
    p.add_argument(
        "--self-check",
        action="store_true",
        help="run built-in synthetic assert and exit",
    )
    return p


def self_check() -> None:
    # INV: reverse contig order + revcomp (asymmetric so flip is observable)
    seq = "AAAA" + "NN" + "ACGT" + "NN" + "TTGC" + "NN" + "GGGG"
    pieces, contigs = split_by_n(seq, min_gap_n=1)
    assert [c.length for c in contigs] == [4, 4, 4, 4]
    cov = cover_contigs(contigs, contigs[1].start, contigs[2].end, 0.9)
    assert cov.ok and cov.contig_ids == [1, 2]
    assert flip_inv(pieces, [1, 2])
    # ACGT NN TTGC -> rev(TTGC)=GCAA NN rev(ACGT)=ACGT
    assert pieces_to_seq(pieces) == "AAAA" + "NN" + "GCAA" + "NN" + "ACGT" + "NN" + "GGGG"

    # mid-band cover (10–90%) rejects the whole call
    bad = cover_contigs(
        contigs, contigs[1].start, contigs[1].start + 2, 0.9, ignore_cover_below=0.0
    )
    assert not bad.ok and "ambiguous_cover" in bad.reason

    # tip <10% dropped; >=90% kept → correct
    seq_tip = "AAAA" + "NN" + "ACGT" + "NN" + ("C" * 20)
    _, tip_contigs = split_by_n(seq_tip, 1)
    tip = cover_contigs(
        tip_contigs,
        tip_contigs[1].start,
        tip_contigs[2].start + 1,  # full contig1 + 1bp into contig2 (1/20=0.05)
        0.9,
        ignore_cover_below=0.1,
    )
    assert tip.ok and tip.contig_ids == [1], tip
    assert "ok_ignore_tips" in tip.reason, tip

    # mid-band present → discard even if another contig is fully covered
    seq_mid = "AAAA" + "NN" + ("A" * 10) + "NN" + ("C" * 10)
    _, mid_contigs = split_by_n(seq_mid, 1)
    # cover contig1 fully (10/10) + 5bp of contig2 (5/10=0.5 mid)
    mid = cover_contigs(
        mid_contigs,
        mid_contigs[1].start,
        mid_contigs[2].start + 5,
        0.9,
        ignore_cover_below=0.1,
    )
    assert not mid.ok and "ambiguous_cover" in mid.reason, mid

    # TRA: move candidate contig 1 before partner contig 3 (keep gap spacing)
    seq3 = "AAAA" + "NN" + "CCCC" + "NN" + "GGGG" + "NN" + "TTTT"
    pieces3, _ = split_by_n(seq3, 1)
    assert move_tra(pieces3, [1], [3])
    assert pieces_to_seq(pieces3) == "AAAA" + "NN" + "GGGG" + "NN" + "CCCC" + "NN" + "TTTT"
    print("[self-check] ok")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_check:
        self_check()
        return 0

    if not args.fasta:
        raise SystemExit("[ERROR] -f/--fasta is required")
    if not (0 < args.cover_frac <= 1):
        raise SystemExit("[ERROR] --cover-frac must be in (0, 1]")
    if not (0 <= args.ignore_cover_below < 1):
        raise SystemExit("[ERROR] --ignore-cover-below must be in [0, 1)")
    if args.min_gap_n < 1:
        raise SystemExit("[ERROR] --min-gap-n must be >= 1")
    if args.min_overlap_bp < 1:
        raise SystemExit("[ERROR] --min-overlap-bp must be >= 1")
    if args.inv is None and args.tra is None and args.sv is None:
        raise SystemExit("[ERROR] provide at least one of --inv / --tra / --sv")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    events = load_events(
        Path(args.inv) if args.inv else None,
        Path(args.tra) if args.tra else None,
        Path(args.sv) if args.sv else None,
    )
    chrom_filter = None
    if args.chromosomes:
        chrom_filter = {c.strip() for c in args.chromosomes.split(",") if c.strip()}
        events = events.loc[events["chrom"].astype(str).isin(chrom_filter)].copy()

    print(f"[INFO] loading FASTA {args.fasta}", flush=True)
    seqs = read_fasta(Path(args.fasta))

    chroms = sorted(set(events["chrom"].astype(str)) & set(seqs))
    if chrom_filter:
        chroms = [c for c in chroms if c in chrom_filter]
    missing = sorted(set(events["chrom"].astype(str)) - set(seqs))
    if missing:
        print(f"[WARN] chroms in calls missing from FASTA: {missing}", flush=True)

    pieces_by_chrom: dict[str, list[Piece]] = {}
    contigs_by_chrom: dict[str, list[ContigView]] = {}
    bed_path = outdir / "contigs_from_N.bed"
    if bed_path.exists():
        bed_path.unlink()

    for chrom in chroms:
        pieces, contigs = split_by_n(seqs[chrom], min_gap_n=args.min_gap_n)
        pieces_by_chrom[chrom] = pieces
        contigs_by_chrom[chrom] = contigs
        write_contig_bed(bed_path, chrom, contigs)
        print(
            f"[INFO] {chrom}: {len(contigs)} contigs, "
            f"{sum(1 for p in pieces if p.kind == 'gap')} gaps",
            flush=True,
        )

    plans: list[Plan] = []
    for _, row in events.iterrows():
        chrom = str(row["chrom"])
        if chrom not in contigs_by_chrom:
            plans.append(
                Plan(
                    chrom, str(row.get("sv_type", "?")).upper(),
                    row.get("cluster_id", "."),
                    int(row["start"]), int(row["end"]),
                    None, None, [], [], [], float(row.get("scale_support", 1) or 1),
                    int(row.get("size", 0) or 0), "skip", "chrom_not_in_fasta",
                )
            )
            continue
        plans.append(
            plan_event(
                row,
                contigs_by_chrom[chrom],
                args.cover_frac,
                args.min_overlap_bp,
                args.ignore_cover_below,
            )
        )

    plans = resolve_conflicts(plans)
    if not args.dry_run:
        plans = apply_plans(pieces_by_chrom, plans)

    report = plans_to_frame(plans)
    report_path = outdir / "correction_decisions.tsv"
    report.to_csv(report_path, sep="\t", index=False)

    n_corr = int((report["status"] == "corrected").sum()) if not args.dry_run else int(
        (report["status"] == "correct").sum()
    )
    n_skip = int((report["status"] == "skip").sum())
    print(
        f"[INFO] decisions: correctable/corrected={n_corr} skip={n_skip} "
        f"-> {report_path}",
        flush=True,
    )

    if args.dry_run:
        print("[INFO] dry-run: skipped writing FASTA", flush=True)
        return 0

    # rebuild sequences; chroms without events keep original
    out_seqs = dict(seqs)
    for chrom, pieces in pieces_by_chrom.items():
        out_seqs[chrom] = pieces_to_seq(pieces)

    out_fa = outdir / "corrected.fasta"
    write_fasta(out_fa, out_seqs)
    print(f"[INFO] wrote {out_fa}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

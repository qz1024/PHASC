# PHASC

**Pangenome-guided Hi-C Assembly Scaffolding and Correction**

PHASC detects structural issues in genome assemblies from Hi-C contact maps. It maps paired Hi-C reads to a reference, builds a multi-resolution `.mcool` contact map, then runs multi-scale detectors for inversions and intra-/inter-chromosomal translocations.

## Requirements


| Component                                                          | Role                                              |
| ------------------------------------------------------------------ | ------------------------------------------------- |
| Python ≥ 3.9                                                       | Runtime                                           |
| [cooler](https://github.com/open2c/cooler)                         | Read/write `.cool` / `.mcool`; CLI for preprocess |
| NumPy, pandas, SciPy, Matplotlib                                   | Numerical analysis and plotting                   |
| NetworkX                                                           | Louvain communities (inversion)                   |
| NetworKit                                                          | Parallel Louvain / PLM (translocation)            |
| [chromap](https://github.com/haowenz/chromap)                      | Hi-C read mapping in preprocess                   |
| [pysam](https://github.com/pysam-developers/pysam)                 | Optional BAM/CRAM refinement for inversion        |
| [HiCExplorer](https://hicexplorer.readthedocs.io/) (`hicFindTADs`) | Optional TAD refinement when `--fasta` is set     |




## Installation

Create and activate the Conda environment from `environment.yml`:

```bash
conda env create -f phasc_environment.yml
conda activate phasc

# optional: use `phasc` from any directory
chmod +x phasc
export PATH="$PWD:$PATH" 

# verify
./phasc --help
which chromap cooler
```

## Quick start

```bash
# 1. Map Hi-C once (-q 0), filter pairs, build INV + TRA mcools
./phasc preprocess \
  -r genome.fa \
  -1 hic_R1.fq.gz \
  -2 hic_R2.fq.gz \
  -o preprocess_out \
  -p sample \
  -t 16

# 2. Detect inversions
./phasc inversion preprocess_out/sample.inv.mapq0.mcool --outdir INV_results

# 3. Detect intra-chromosomal translocations
./phasc intra-translocation --mcool preprocess_out/sample.tra.mapq1.mcool --outdir TRA_results --chrom chr3B

# 4. Or run INV + TRA together on the two MAPQ mcools
./phasc pipeline \
  --inv-mcool preprocess_out/sample.inv.mapq0.mcool \
  --tra-mcool preprocess_out/sample.tra.mapq1.mcool \
  -o PHASC_results --jobs 4 --chromosomes chr3B
```



## Commands

```bash
./phasc --help
./phasc <command> --help
```


| Command         | Script                        | Description                                             |
| --------------- | ----------------------------- | ------------------------------------------------------- |
| `preprocess`    | `preprocess.py`               | Map Hi-C with chromap → INV/TRA `.pairs` → two `.mcool` |
| `inversion`     | `multiscale_inversion.py`     | Multi-scale inversion detection (Louvain / GCOR)        |
| `intra-translocation` | `multiscale_intra_translocation.py` | Multi-scale intra-chromosomal translocation detection |
| `inter-translocation` | `multiscale_inter_translocation.py` | Multi-scale inter-chromosomal translocation detection |
| `pipeline`            | `pipeline.py`                       | Run INV + TRA in parallel into one shared outdir      |
| `correct`       | `correct_scaffold.py`         | Correct contig mounting errors from INV/TRA vs N-gaps   |




### `preprocess`

Maps paired Hi-C reads **once** with chromap (`-q 0`), then filters that `.pairs` file to INV / TRA MAPQ thresholds and builds two multi-resolution `.mcool` files.

Defaults: `--inv-mapq 0` (keep all), `--tra-mapq 1` (filter `mapq1/mapq2 >= 1`).

Requires a chromap pairs header that includes `mapq1` and `mapq2` columns, e.g.
`#columns: readID chrom1 pos1 chrom2 pos2 strand1 strand2 pair_type mapq1 mapq2`.

```bash
./phasc preprocess \
  -r genome.fa \
  -1 hic_R1.fq.gz \
  -2 hic_R2.fq.gz \
  -o preprocess_out \
  -p sample \
  -t 16
# optional: --inv-mapq 0 --tra-mapq 1
```

Useful options:

- `--inv-mapq 0` — keep pairs with mapq1/mapq2 ≥ this for inversion (default: 0)
- `--tra-mapq 1` — keep pairs with mapq1/mapq2 ≥ this for translocation (default: 1)
- `--resolutions 10000,25000,50000,100000,250000,500000,1000000` — mcool resolutions (default shown)
- `--skip-mapping --pairs existing.mapq0.pairs` — reuse a MAPQ-0 pairs file and only filter + cload
- `--chromap-index PATH` — reuse or set the chromap index path
- `--no-balance` — skip cooler balance / balanced zoomify
- `--dry-run` — print commands without running them

Main outputs:

- `preprocess_out/sample.mapq0.pairs` — chromap output (MAPQ ≥ 0)
- `preprocess_out/sample.inv.mapq0.pairs` / `sample.tra.mapq1.pairs` — filtered pairs
- `preprocess_out/sample.inv.mapq0.mcool` — INV contact map
- `preprocess_out/sample.tra.mapq1.mcool` — TRA contact map
- `preprocess_out/sample.mcool` — symlink/alias to the INV mcool (compatibility)
- `preprocess_out/sample.preprocess_manifest.tsv` — output manifest
- `preprocess_out/sample.chrom.sizes` — chromosome sizes from the FASTA



### `inversion`

Runs a four-scale Louvain / GCOR inversion workflow on an `.mcool` file.

```bash
./phasc inversion preprocess_out/sample.mcool --outdir INV_results
```

Optional refinement:

```bash
./phasc inversion preprocess_out/sample.mcool \
  --outdir INV_results \
  --fasta genome.fa \
  --hic-alignments alignments.bam \
  --jobs 4
```

- `--fasta` enables gap scanning and four-scale `hicFindTADs` refinement (requires HiCExplorer).
- `--hic-alignments` accepts BAM/CRAM, `.pairs(.gz)`, or BEDPE (BAM/CRAM needs pysam).



### `intra-translocation`

Detects intra-chromosomal translocation candidates across multiple Hi-C resolutions × Louvain gammas, then merges overlapping calls.

**Default flow:** cluster → score enrichment → clear-insert ILP → merge scales → **enrichment gate (`fold > 1.5`)** → `ALL_TRA.tsv`.

```bash
./phasc intra-translocation \
  --mcool preprocess_out/sample.mcool \
  --outdir TRA_results \
  --chrom chr3B \
  --skip-plot
```

Useful options:

- `--chrom chr3B` — chromosome to analyze
- `--resolutions 50000,100000,200000,500000` — Hi-C resolutions
- `--gammas 0.5,0.75,1.0,1.25,1.5,2.0` — Louvain/PLM gammas
- `--jobs 4` — parallel (resolution × gamma) workers
- `--min-enrichment-fold 1.5` — final gate after merge (`enrichment_fold >` threshold; **default 1.5**; `<=0` disables). `enrichment_fold` is **breakpoint tip-window** distance-corrected O/E (max over cand-tip×partner-tip); `--tip-window-bp` default 2 Mb. `enrichment_fold_block` / `enrichment_fold_flank` keep full-block O/E and old flank ratio for audit.
- `--truth truth.bed` — optional evaluation against known TRA intervals
- `--skip-plot` — skip community figure rendering

Outputs:

- `ALL_TRA.tsv` — final calls **after** enrichment gate
- `ALL_TRA_pre_enrichment.tsv` — merged calls before the gate
- `ALL_TRA_enrichment_dropped.tsv` — rows removed by the gate


### `inter-translocation`

Detects **inter**-chromosomal translocation candidates: joint Louvain on chromA+chromB (intra + inter edges), shared communities → tip×tip O/E vs chrom-pair mean → merge scales → enrichment gate (`fold > 1.5`) → `ALL_INTER_TRA.tsv`.

```bash
./phasc inter-translocation \
  --mcool preprocess_out/sample.tra.mapq1.mcool \
  --outdir INTER_TRA_results \
  --chrom-a chr1A --chrom-b chr1B \
  --skip-plot
```

Or all unordered pairs among a chrom list:

```bash
./phasc inter-translocation \
  --mcool preprocess_out/sample.tra.mapq1.mcool \
  --outdir INTER_TRA_results \
  --chromosomes chr1A,chr1B,chr2A \
  --skip-plot
```

Useful options match intra TRA (`--resolutions`, `--gammas`, `--jobs`, `--min-enrichment-fold`, `--tip-window-bp`). Outputs: `ALL_INTER_TRA.tsv`, `ALL_INTER_TRA_pre_enrichment.tsv`, `ALL_INTER_TRA_enrichment_dropped.tsv`.


### `pipeline`

Runs inversion and **intra**-chromosomal translocation in parallel, then writes both result tables (and a combined `ALL_SV.tsv`) into one shared `--outdir`. Prefer the two MAPQ-specific mcools from preprocess; a single shared mcool still works.

```bash
./phasc pipeline \
  --inv-mcool preprocess_out/sample.inv.mapq0.mcool \
  --tra-mcool preprocess_out/sample.tra.mapq1.mcool \
  -o PHASC_results \
  --jobs 4 \
  --chromosomes chr3B

# legacy: one mcool for both detectors
./phasc pipeline preprocess_out/sample.mcool -o PHASC_results --jobs 4 --chromosomes chr3B
```

Useful options:

- `--inv-mcool` / `--tra-mcool` — separate contact maps (recommended after preprocess)
- `--chromosomes chr3B` — chromosome passed to intra-translocation (first name only)
- `--inv-min-support 2` — inversion multiscale support threshold
- `--min-enrichment-fold 1.5` — forwarded to intra-translocation final gate (default 1.5)
- `--truth truth.bed` — optional TRA truth BED for evaluation prints
- `--tra-skip-plot` / `--no-tra-skip-plot` — control TRA community plots
- `--keep-work` — retain `outdir/.pipeline_work` for debugging

Main outputs:

- `ALL_INV_multiscale.tsv` — inversion calls
- `ALL_TRA.tsv` / `merged_true_calls.tsv` — translocation calls (enrichment-gated)
- `ALL_SV.tsv` — combined INV + TRA table
- `HiC_images/` — inversion Hi-C panels


### `correct`

After INV/TRA detection, correct **contig mounting** mistakes (wrong orientation/order across scaffolding gaps). Splits the FASTA on N-runs, drops tip overlaps covered below `--ignore-cover-below` (default 0.1), then only corrects a call when it covers ≥ `--cover-frac` of **every remaining** overlapping contig.

```bash
./phasc correct \
  -f genome.fa \
  --inv PHASC_results/ALL_INV_multiscale.tsv \
  --tra PHASC_results/ALL_TRA.tsv \
  -o CORRECT_results \
  --cover-frac 0.9 \
  --ignore-cover-below 0.1
```

- **INV**: reverse the spanned contig/gap block and reverse-complement contig sequences
- **TRA**: move the candidate contig block to immediately before the partner contig block
- `--dry-run` — write `correction_decisions.tsv` only
- `--min-gap-n` — minimum N-run length that splits contigs (default: 1)
- `--ignore-cover-below` — drop contigs covered below this fraction before the cover-frac check (default: 0.1)

Outputs:

- `corrected.fasta` — corrected genome
- `correction_decisions.tsv` — per-call correct/skip with reasons
- `contigs_from_N.bed` — N-split contigs


## License

This project is licensed under the [GNU Affero General Public License v3.0](LICENSE) (AGPL-3.0).
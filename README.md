# PHASC

**Pangenome-guided Hi-C Assembly Scaffolding and Correction**

PHASC detects structural issues in genome assemblies from Hi-C contact maps. It maps paired Hi-C reads to a reference, builds a multi-resolution `.mcool` contact map, then runs multi-scale detectors for inversions and intra-chromosomal translocations.

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
# 1. Map Hi-C reads and build a genome-wide mcool
./phasc preprocess \
  -r genome.fa \
  -1 hic_R1.fq.gz \
  -2 hic_R2.fq.gz \
  -o preprocess_out \
  -p sample \
  -t 16

# 2. Detect inversions
./phasc inversion preprocess_out/sample.mcool --outdir INV_results

# 3. Detect intra-chromosomal translocations
./phasc translocation --mcool preprocess_out/sample.mcool --outdir TRA_results --chrom chr3B

# 4. Or run INV + TRA together
./phasc pipeline preprocess_out/sample.mcool -o PHASC_results --jobs 4 --chromosomes chr3B
```



## Commands

```bash
./phasc --help
./phasc <command> --help
```


| Command         | Script                        | Description                                             |
| --------------- | ----------------------------- | ------------------------------------------------------- |
| `preprocess`    | `preprocess.py`               | Map Hi-C with chromap → `.pairs` → genome-wide `.mcool` |
| `inversion`     | `multiscale_inversion.py`     | Multi-scale inversion detection (Louvain / GCOR)        |
| `translocation` | `multiscale_translocation.py` | Multi-scale intra-chromosomal translocation detection   |
| `pipeline`      | `pipeline.py`                 | Run INV + TRA in parallel into one shared outdir        |




### `preprocess`

Maps paired Hi-C reads with chromap and converts the resulting `.pairs` file into a multi-resolution `.mcool` for downstream detection.

```bash
./phasc preprocess \
  -r genome.fa \
  -1 hic_R1.fq.gz \
  -2 hic_R2.fq.gz \
  -o preprocess_out \
  -p sample \
  -t 16
```

Useful options:

- `--resolutions 10000,25000,50000,100000,250000,500000,1000000` — mcool resolutions (default shown)
- `--skip-mapping --pairs existing.pairs` — reuse an existing pairs file
- `--chromap-index PATH` — reuse or set the chromap index path
- `--no-balance` — skip cooler balance / balanced zoomify
- `--dry-run` — print commands without running them

Main outputs:

- `preprocess_out/sample.mcool` — genome-wide multi-resolution contact map
- `preprocess_out/sample.preprocess_manifest.tsv` — output manifest
- `preprocess_out/sample.pairs` — chromap pairs (unless reused)
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



### `translocation`

Detects intra-chromosomal translocation candidates across multiple Hi-C resolutions × Louvain gammas, then merges overlapping calls into consensus events.

```bash
./phasc translocation \
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
- `--truth truth.bed` — optional evaluation against known TRA intervals
- `--skip-plot` — skip community figure rendering


### `pipeline`

Runs inversion and translocation in parallel, then writes both result tables (and a combined `ALL_SV.tsv`) into one shared `--outdir`.

```bash
./phasc pipeline preprocess_out/sample.mcool \
  -o PHASC_results \
  --jobs 4 \
  --chromosomes chr3B
```

Useful options:

- `--chromosomes chr3B` — chromosome passed to translocation (first name only)
- `--inv-min-support 2` — inversion multiscale support threshold
- `--truth truth.bed` — optional TRA truth BED for evaluation prints
- `--tra-skip-plot` / `--no-tra-skip-plot` — control TRA community plots
- `--keep-work` — retain `outdir/.pipeline_work` for debugging

Main outputs:

- `ALL_INV_multiscale.tsv` — inversion calls
- `ALL_TRA.tsv` / `merged_true_calls.tsv` — translocation calls
- `ALL_SV.tsv` — combined INV + TRA table
- `HiC_images/` — inversion Hi-C panels


## License

This project is licensed under the [GNU Affero General Public License v3.0](LICENSE) (AGPL-3.0).
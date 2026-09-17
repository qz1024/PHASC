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
./phasc translocation preprocess_out/sample.mcool --output-dir TRA_results
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

Detects intra-chromosomal translocation candidates across multiple Hi-C resolutions, then merges overlapping calls into consensus events.

```bash
./phasc translocation preprocess_out/sample.mcool --output-dir TRA_results
```

Useful options:

- `--chromosomes chr1,chr2` — restrict analysis to named chromosomes
- `--resolutions auto` — or a comma-separated list such as `50000,100000,250000`
- `--workers 4` — parallel scale processes
- `--min-resolution-support 2` — minimum distinct resolutions supporting an event
- `--no-balance` — use raw contact counts instead of balanced weights


## License

This project is licensed under the [GNU Affero General Public License v3.0](LICENSE) (AGPL-3.0).
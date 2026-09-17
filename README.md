# PHASC

Pangenome-guided Hi-C Assembly Scaffolding and Correction

## Command line

```bash
./phasc --help
```

## Preprocess Hi-C reads

The `preprocess` subcommand maps paired Hi-C reads with `chromap` and converts
the resulting `.pairs` file to a genome-wide `.mcool` for downstream PHASC
detection.

```bash
./phasc preprocess \
  -r genome.fa \
  -1 hic_R1.fq.gz \
  -2 hic_R2.fq.gz \
  -o preprocess_out \
  -p sample \
  -t 16
```

Main outputs:

- `preprocess_out/sample.mcool`: genome-wide multi-resolution contact map.
- `preprocess_out/sample.preprocess_manifest.tsv`: output manifest.

Then run the existing detectors:

```bash
./phasc inversion preprocess_out/sample.mcool --outdir INV_results
./phasc translocation preprocess_out/sample.mcool --output-dir TRA_results
```

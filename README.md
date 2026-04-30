# GPN-Star Entropy Fine-Mapping Benchmark

This repository benchmarks whether GPN-Star evolutionary-constraint entropy can
prioritize variants in GWAS fine-mapped loci. The first pass is intentionally
lightweight: it treats FinnGen T2D SuSiE/FINEMAP posterior inclusion
probabilities (PIPs) and credible-set assignments as SOTA reference labels, then
asks whether entropy-only rankings recover high-PIP variants.

Entropy alone is not a statistical fine-mapping method. It has no trait
association signal and no LD model, so a weak standalone result would not rule
out using entropy as a functional prior inside SuSiE, FINEMAP, or a
PolyFun-style workflow.

## Data Layout

- `entropy/entropy_chr*.parquet`: local GPN-Star entropy files, one per
  chromosome.
- `data/`: local FinnGen download cache, ignored by git.
- `results/`: benchmark outputs, ignored by git.

Expected entropy parquet columns:

- `chrom`
- `pos`
- `ref`
- `entropy_calibrated`

FinnGen R12 summary statistics are resolved from the public manifest under
`summary_stats/release/`. Fine-mapping SNP files are downloaded from
`finemap/full/susie/` and `finemap/full/finemap/`. The default endpoint is
`T2D`; use `E4_DM2` if a chosen FinnGen release does not expose the core T2D
fine-mapping files.

## Setup

```bash
uv sync
```

If package resolution is unavailable, the dependency declarations are already in
`pyproject.toml`; rerun `uv sync` once PyPI/network access is restored.

## Commands

Inspect entropy coverage and schema:

```bash
uv run gpn-finemap inspect-entropy --entropy-dir entropy --verbose
```

Download/cache FinnGen files:

```bash
uv run gpn-finemap download-finngen --release 12 --endpoint T2D --cache-dir data --verbose
```

Run the benchmark using downloaded or cached files:

```bash
uv run gpn-finemap run \
  --release 12 \
  --endpoint T2D \
  --entropy-dir entropy \
  --cache-dir data \
  --output-dir results/t2d_entropy \
  --verbose
```

If FinnGen fine-mapping files are not found by the built-in public URL
candidates, pass explicit inputs:

```bash
uv run gpn-finemap run \
  --entropy-dir entropy \
  --summary-path data/finngen_R12/T2D/finngen_R12_T2D.gz \
  --susie-snp-path data/finngen_R12/T2D/finngen_R12_T2D.SUSIE.snp.bgz \
  --finemap-snp-path data/finngen_R12/T2D/finngen_R12_T2D.FINEMAP.snp.bgz \
  --output-dir results/t2d_entropy \
  --verbose
```

By default, lower `entropy_calibrated` is treated as more constrained. To test
the opposite direction:

```bash
uv run gpn-finemap run --constrained-direction high
```

All CLI commands support `--verbose`/`-v` to print progress logs for downloads,
parquet scans, chromosome joins, metric computation, and output writing.

Prepare entropy-derived priors for SuSiE and FINEMAP from a completed benchmark
run:

```bash
uv run gpn-finemap prepare-priors \
  --annotated-variants results/t2d_entropy/annotated_finemap_variants.parquet \
  --output-dir results/t2d_entropy_priors \
  --prior-method softmax \
  --temperature 1.0 \
  --finemap-expected-causal-per-region 1.0 \
  --verbose
```

The prior command writes per-region SuSiE `prior_weights` files and FINEMAP
`.z`-style files with an added `prob` column for `--prior-snps`. It also writes
template run files showing where to plug in locus-specific LD matrices/master
files.

## Metrics

The benchmark reports metrics per fine-mapped region and then averages them by
method and PIP threshold:

- Spearman correlation between entropy score and PIP.
- AUROC/AUPRC for predicting variants above PIP thresholds `0.1`, `0.5`, and
  `0.9`.
- Top-k and top-percentile precision/recall for entropy ranking.
- The same ranking metrics for GWAS p-value rank when available.

These are pseudo-label metrics against published fine-mapping outputs, not
causal-truth metrics.

## Outputs

Each benchmark run writes:

- `annotated_finemap_variants.parquet`: FinnGen SNP-level rows annotated with
  entropy.
- `region_metrics.tsv` and `.parquet`: per-region benchmark metrics.
- `top_rank_metrics.tsv` and `.parquet`: top-k/top-percentile precision and
  recall.
- `global_summary.tsv` and `.parquet`: aggregate method comparison.
- `global_auprc.png`: quick visual comparison of entropy and GWAS p-value
  rankings.
- `report.md`: concise interpretation and run metadata.

Prior preparation writes:

- `entropy_priors.parquet` and `.tsv`: all annotated variants plus
  `susie_prior_weight`, `finemap_prior_probability`, and `SNPVAR`.
- `susie/*.prior_weights.tsv`: two-column files for passing to
  `susie_rss(..., prior_weights = ...)`.
- `finemap/*.prior.z`: FINEMAP-style `.z` files with an entropy-derived `prob`
  prior column for `finemap --prior-snps`.
- `prior_manifest.tsv`: mapping from region to generated prior files.

## Future Extension

The next benchmark should run matched uniform-prior and entropy-prior
SuSiE/FINEMAP jobs with the same summary statistics, loci, and LD matrices, then
compare PIP calibration, credible-set size, and recovery of FinnGen lead/high-PIP
variants.

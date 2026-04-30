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

FinnGen files can be downloaded from public release manifests where available,
or passed explicitly as local paths/URLs. The default endpoint is `T2D`; use
`E4_DM2` if a chosen FinnGen release does not expose the core T2D fine-mapping
files.

## Setup

```bash
uv sync
```

If package resolution is unavailable, the dependency declarations are already in
`pyproject.toml`; rerun `uv sync` once PyPI/network access is restored.

## Commands

Inspect entropy coverage and schema:

```bash
uv run gpn-finemap inspect-entropy --entropy-dir entropy
```

Download/cache FinnGen files:

```bash
uv run gpn-finemap download-finngen --release 12 --endpoint T2D --cache-dir data
```

Run the benchmark using downloaded or cached files:

```bash
uv run gpn-finemap run \
  --release 12 \
  --endpoint T2D \
  --entropy-dir entropy \
  --cache-dir data \
  --output-dir results/t2d_entropy
```

If FinnGen fine-mapping files are not found by the built-in public URL
candidates, pass explicit inputs:

```bash
uv run gpn-finemap run \
  --entropy-dir entropy \
  --summary-path data/T2D.gz \
  --susie-snp-path data/T2D.SUSIE.snp.bgz \
  --finemap-snp-path data/T2D.FINEMAP.snp.bgz \
  --output-dir results/t2d_entropy
```

By default, lower `entropy_calibrated` is treated as more constrained. To test
the opposite direction:

```bash
uv run gpn-finemap run --constrained-direction high
```

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

## Future Extension

If entropy-only ranking shows signal, the next benchmark should make entropy a
prior rather than a standalone ranker. A practical path is to convert entropy
into functional annotations, estimate prior causal probabilities genome-wide,
and compare uniform-prior SuSiE/FINEMAP against entropy-informed
SuSiE/FINEMAP/PolyFun-style runs on matched FinnGen loci.

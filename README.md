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
  --entropy-dir entropy \
  --output-dir results/t2d_entropy_priors \
  --prior-method surprise \
  --surprise-gamma 0.25 \
  --prior-weight-max 20 \
  --finemap-expected-causal-per-region 1.0 \
  --verbose
```

The recommended `surprise` prior uses the genome-wide entropy distribution, not
a within-peak softmax. For low-entropy-is-conserved runs, it computes
`u = P0(H <= Hj)`, `s = -log10(u)`, and a capped enrichment
`w = exp(gamma * s)` before normalizing weights within each fine-mapping region.
This keeps absolute conservation information, so a conserved SNP in a globally
extreme tail receives a stronger prior than a merely local outlier. The prior
command writes per-region SuSiE `prior_weights` files and FINEMAP `.z`-style
files with an added `prob` column for `--prior-snps`. It also writes template run
files showing where to plug in locus-specific LD matrices/master files.

Run matched uniform-prior and entropy-prior SuSiE/FINEMAP jobs end-to-end:

```bash
uv run gpn-finemap run-fine-mapping \
  --annotated-variants results/t2d_entropy/annotated_finemap_variants.parquet \
  --entropy-dir entropy \
  --output-dir results/t2d_entropy_finemap \
  --prior-method surprise \
  --surprise-gamma 0.25 \
  --prior-weight-max 20 \
  --ld-bcor-dir data/finngen_ld \
  --ldstore-exe /path/to/ldstore \
  --rscript-exe Rscript \
  --finemap-exe /path/to/finemap \
  --download-ld-bcor \
  --max-regions 2 \
  --verbose
```

Requirements for the end-to-end runner:

- `LDstore v1.1` to extract LD tables from FinnGen/SISu `FG_LD_chr*.bcor`
  files. `LDstore v2.0` has a different CLI (`--bcor-to-text`,
  `--bcor-file`, `--ld-file`, `--range`) and will reject the v1.1
  `--bcor` flag used by this runner.
- `Rscript` with the R package `susieR` installed.
- `FINEMAP v1.4` or compatible executable.
- Enough disk space for FinnGen public LD BCOR files. They are several GB per
  chromosome; omit `--download-ld-bcor` if you staged them yourself.

FinnGen notes that the public SISu LD estimates are not preferred for precise
fine-mapping because in-sample LD is better. Use this runner for a reproducible
public benchmark, and swap in better matched LD through `--ld-matrix-dir` when
available.

SuSiE can reject loci with `The estimated prior variance is unreasonably large`
when the summary-statistic z-scores and LD matrix are inconsistent. Prefer
matched LD via `--ld-matrix-dir`; for exploratory public-LD runs, either skip
SuSiE with `--run-susie false` or intentionally bypass this SuSiE safety check
with `--no-susie-check-prior`.

If you already have per-region FINEMAP/SuSiE LD matrices in the same variant
order as the generated region inputs, use `--ld-matrix-dir` instead of
`--ld-bcor-dir`. For a smoke test only, `--allow-identity-ld --run-susie false
--run-finemap false` prepares inputs without external tools.

Ask whether high-PIP SNPs are enriched for high predicted conservation:

```bash
uv run gpn-finemap conservation-enrichment \
  --annotated-variants results/t2d_entropy/annotated_finemap_variants.parquet \
  --output-dir results/t2d_conservation_enrichment \
  --pip-thresholds 0.1,0.5,0.9 \
  --conservation-quantiles 0.9,0.95 \
  --n-permutations 10000 \
  --verbose
```

This mirrors the ChromBPNet-style question but swaps accessibility for
conservation: within each fine-mapped region, it tests whether high-PIP SNPs
overlap the most conserved SNPs more often than expected after randomly drawing
the same number of SNPs from that region.

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
  `entropy_surprise`, `entropy_prior_enrichment`, `susie_prior_weight`,
  `finemap_prior_probability`, and `SNPVAR`.
- `susie/*.prior_weights.tsv`: two-column files for passing to
  `susie_rss(..., prior_weights = ...)`.
- `finemap/*.prior.z`: FINEMAP-style `.z` files with an entropy-derived `prob`
  prior column for `finemap --prior-snps`.
- `prior_manifest.tsv`: mapping from region to generated prior files.

End-to-end fine-mapping writes:

- `fine_mapping_manifest.tsv`: region-level manifest linking LD, z, and output
  files.
- `regions/<region>/entropy/` and `regions/<region>/uniform/`: matched input
  files for entropy-prior and uniform-prior runs.
- `regions/<region>/*.ld`: LD matrix used for both methods.
- `regions/<region>/susie_entropy.pip.tsv` and
  `regions/<region>/susie_uniform.pip.tsv`: SuSiE PIPs when SuSiE is run.
- `regions/<region>/finemap_entropy/` and `regions/<region>/finemap_uniform/`:
  FINEMAP master/output files when FINEMAP is run.

Conservation enrichment writes:

- `region_enrichment.tsv`: per-region observed vs expected overlap between
  high-PIP SNPs and top-conservation SNPs.
- `global_enrichment.tsv`: aggregate fold enrichment and empirical p-values.
- `null_overlaps.tsv`: permutation null overlap counts.
- `conservation_enrichment.png`: global fold-enrichment plot.
- `overlap_enrichment_by_max_pip.png`: ChromBPNet-style curve plot with max PIP
  on the x-axis and overlap enrichment on a log y-axis.
- `overlap_enrichment_curves.tsv`: data behind the curve plot.
- `conservation_enrichment.md`: compact report for the analysis question.

## Future Extension

The next prior model should replace the unmatched genome-wide background with a
matched background when the necessary annotations are available, for example MAF
bins, variant class, distance to TSS, CpG context, or mappability. The benchmark
should then compare PIP calibration, credible-set size, and recovery of FinnGen
lead/high-PIP variants across uniform and conservation-prior runs.

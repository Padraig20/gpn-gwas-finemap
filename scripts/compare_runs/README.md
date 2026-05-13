# Comparing fine-mapping runs across (ancestry × prior)

`compare_pip.py` consumes the four aggregated FINEMAP tables produced by
`polyfun-gpn aggregate` and answers two questions in one shot:

1. **Cross-ancestry agreement.** Does a variant with high PIP in EUR
   also get a high PIP in EAS? (per prior, paired by variant.)
2. **Entropy prior effect.** Do variants pick up higher PIP when fine-mapped
   with the entropy prior than with the uniform prior? (per ancestry, paired
   by variant.)

## Inputs

`polyfun-gpn aggregate` writes one TSV per (ancestry, prior) pair to
`<output_dir>/results/finemap.demo.<prior>.tsv`. The columns are whatever
PolyFun's `finemapper.py` emitted plus a `locus_id` column the aggregator
prepends. We use `SNP, CHR, BP, A1, A2, PIP, locus_id`; everything else is
ignored.

With the bundled YAMLs (`output/EUR`, `output/EAS`) the four defaults are:

```
output/EUR/results/finemap.demo.none.tsv      --eur-none
output/EUR/results/finemap.demo.entropy.tsv   --eur-entropy
output/EAS/results/finemap.demo.none.tsv      --eas-none
output/EAS/results/finemap.demo.entropy.tsv   --eas-entropy
```

so the common case is a no-flag invocation:

```bash
uv run python scripts/compare_runs/compare_pip.py
```

Override individually when you have non-default output directories:

```bash
uv run python scripts/compare_runs/compare_pip.py \
    --eur-none    output/EUR_noprior/results/finemap.demo.none.tsv \
    --eur-entropy output/EUR_entropy/results/finemap.demo.entropy.tsv \
    --eas-none    output/EAS_noprior/results/finemap.demo.none.tsv \
    --eas-entropy output/EAS_entropy/results/finemap.demo.entropy.tsv \
    --output-dir  output/compare_v2
```

## Variant matching

Variants are joined on `(CHR, BP, A1, A2)` — the same compound key PolyFun
uses internally. Two refinements:

* Within ancestry (`entropy vs uniform`) we **don't** allow allele
  swaps: both runs share the same harmonised GWAS sumstats, so any
  flipped pair would be a real disagreement worth seeing.
* Across ancestries (`EUR vs EAS`) we **do** try the swapped key
  `(A2, A1)` for any variant the same-orientation join missed, so the
  comparison isn't penalised by allele-ordering differences between the
  two GWAS harmonisations.

If multiple loci own the same variant (rare at locus-window boundaries),
the row with the higher PIP wins — that's the locus that "claims" the
SNP.

## Outputs

Everything lands under `--output-dir` (default `output/compare/`):

```
output/compare/
├── plots/
│   ├── threshold_sweep_by_ancestry.png   # like the reference figure: # variants vs PIP threshold
│   ├── threshold_sweep_by_prior.png      # same shape, but EUR vs EAS per prior
│   ├── replication_rate.png              # P(PIP_B > T | PIP_A > T) over T, per prior
│   ├── scatter_entropy_vs_uniform.png    # paired: PIP_uniform vs PIP_entropy, per ancestry
│   ├── scatter_eur_vs_eas.png            # paired: PIP_EUR vs PIP_EAS, per prior
│   └── delta_hist_entropy_vs_uniform.png # histogram of PIP_entropy - PIP_uniform per ancestry
└── summary.tsv
```

### `summary.tsv` schema

`summary.tsv` has one row per comparison plus one row per individual run
(marginal counts). Columns:

| column                  | applies to | meaning |
| ----------------------- | ---------- | ------- |
| `comparison_type`       | all        | `marginal`, `paired_within_ancestry`, `paired_cross_ancestry` |
| `label`                 | all        | e.g. `EUR/entropy`, `EUR: entropy vs uniform`, `none prior: EUR vs EAS` |
| `n_variants`            | marginal   | rows in that single run's TSV |
| `n_paired`              | paired     | inner-join size after allele matching |
| `n_pip_gt_0.5/0.9/0.95` | marginal   | sample size at standard PIP cutoffs |
| `mean_pip`, `median_pip`| marginal   | marginal PIP distribution |
| `spearman_rho`/`_p`     | paired     | rank-correlation between the two PIPs |
| `pearson_r`/`_p`        | paired     | linear correlation |
| `wilcoxon_p_two_sided`  | paired     | paired sign-rank test, drops exact ties |
| `mean_delta_b_minus_a`, `median_delta_b_minus_a` | paired | sign tells direction (see below) |
| `n_b_higher`, `n_a_higher` | paired | # variants where B > A vs A > B |
| `n_both_pip_gt_0.5/0.9` | paired     | replication counts at standard cutoffs |

The `A → B` orientation per row:

* `EUR: entropy vs uniform` → A = uniform (X-axis on the scatter), B = entropy.
  Positive `median_delta` ⇒ entropy *raises* PIPs.
* `EAS: entropy vs uniform` → same orientation.
* `none prior: EUR vs EAS`  → A = EUR (X-axis), B = EAS.
  Positive `median_delta` ⇒ EAS PIPs are higher on average; near 0 ⇒
  cross-ancestry-symmetric.
* `entropy prior: EUR vs EAS` → same.

## How to read the plots against your two hypotheses

**H1: high-PIP variants in EUR are also high-PIP in EAS.**

* The "EUR vs EAS" scatter (per prior) is the headline plot — points
  hugging the y=x diagonal mean the two ancestries fine-map to the same
  causal SNPs.
* The `replication_rate.png` shows the same thing as a curve: of
  variants with PIP > T in one ancestry, what fraction also have PIP > T
  in the other? A flat-near-1 curve says "ancestry-robust"; a curve that
  drops to zero past T ≈ 0.5 says "the ancestries disagree on which SNP
  is causal".
* Headline numbers in `summary.tsv`:
  `spearman_rho` for `none/entropy prior: EUR vs EAS` (the closer to 1
  the better) and `n_both_pip_gt_0.9` (how many variants are jointly
  high-confidence).

**H2: variants get higher PIP with the entropy prior.**

* `threshold_sweep_by_ancestry.png` is the direct analogue of the
  reference figure you shared, one panel per ancestry. If the
  "With entropy prior" line is consistently above "Uniform prior" at
  high PIP cutoffs, the entropy prior is concentrating mass on causal
  candidates.
* `scatter_entropy_vs_uniform.png` (paired): more points above the y=x
  diagonal than below ⇒ entropy raised PIPs net.
* `delta_hist_entropy_vs_uniform.png` (paired): the histogram should be
  shifted right of 0; the printed median is the headline number.
* Headline numbers: `median_delta_b_minus_a` and `wilcoxon_p_two_sided`
  for the two `entropy vs uniform` rows. A significant Wilcoxon plus a
  positive median delta = entropy is doing real work, in the right
  direction.

`n_b_higher` vs `n_a_higher` is a useful complement to the Wilcoxon p:
a large `n_b_higher / (n_a_higher + n_b_higher)` ratio means most paired
variants moved in the predicted direction, even when the per-variant
shift is small. Conversely, a near-50/50 split with a tiny p-value means
the entropy prior moves PIPs slightly but consistently — interesting
but unlikely to flip downstream conclusions.

## When the plots look wrong

* **Empty scatters / `n_paired = 0`.** The two runs share no variants
  under the join. Almost always a build mismatch in the underlying
  sumstats / LD reference (hg19 vs hg38) -- the cross-ancestry case
  tries the swapped allele key, so a 0 join after that points at
  coordinate-system drift rather than strand flips.
* **Replication curve crashes at high T.** Often just sample size: at
  T = 0.99 there might only be 2-3 variants over the threshold in one
  ancestry, making the conditional probability noisy. Look at
  `summary.tsv` for `n_pip_gt_0.95` to see if it's a real disagreement
  or just sparse counts.
* **Wilcoxon p is highly significant but median delta ≈ 0.** With
  thousands of paired variants the Wilcoxon will reject the null on
  arbitrarily small shifts. Read the median and `n_b_higher` together;
  if the median is ~0 and `n_b_higher ≈ n_a_higher`, the entropy prior
  is moving PIPs symmetrically and the significance is just sample
  size.

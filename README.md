# polyfun-gpn

PolyFun + FINEMAP fine-mapping for GWAS, with per-SNP causal priors derived
from genome-wide entropy ("surprise vs. background"). Inspired by the Borzoi-
informed fine-mapping paper, but using a per-position **conservation/entropy**
score instead of Borzoi-derived predictions.

The pipeline keeps the canonical PolyFun + FINEMAP stack intact and only
swaps the source of the per-SNP `SNPVAR` column: instead of S-LDSC, we
compute `-log f_bg(entropy)` against the genome-wide background.

## Approach

1. **Background distribution** `f_bg(e)`: random sample of ~10 M genome-wide
   entropy values (proportional to chromosome length) binned into a 500-bin
   histogram on `[0, 3]`. Cached to `data/background/entropy_bg.npz`.
2. **Per-SNP prior**: `SNPVAR_i = exp(tau * w_i)` with
   `w_i = -log(f_bg(e_i) + epsilon)`. Variants without an entropy lookup
   fall back to the locus-level median; tagged with `prior_source` for
   auditability.
3. **Fine-mapping**: PolyFun's `finemapper.py` is invoked per locus with
   `--method finemap` and on-demand UKB LD blocks. Same flow scales to all
   genome-wide-significant loci via PolyFun's `create_finemapper_jobs.py`
   wrapper (`run-all`).

A `--prior {entropy,uniform,none}` flag swaps between the entropy-driven
prior, a uniform baseline (still functionally informed), and unpriored
fine-mapping for direct comparison.

## Project layout

```
polyfun-gpn/
├── configs/
│   ├── default.yaml          # paths + tau + finemap params
│   └── loci_demo.tsv         # five well-known T2D loci (hg19)
├── data/
│   ├── entropy/              # provided: entropy_chr*.parquet (hg38)
│   ├── gwas/                 # provided: T2D EUR sumstats (hg19)
│   ├── reference/            # filled by `setup`: liftover chain files
│   ├── background/           # filled by `build-bg`: entropy_bg.npz
│   └── ld_cache/             # FINEMAP --cache-dir; subdivided when using --gwas-id
├── external/
│   ├── polyfun/              # cloned by `setup`
│   └── bin/finemap           # FINEMAP v1.4.1 binary, fetched by `setup`
├── output/                   # default layout; see "Multi-GWAS" for output/{id}/
│   ├── loci/{prior}/{locus}/ # per-locus sumstats + FINEMAP outputs
│   └── results/              # aggregated results
├── src/polyfun_gpn/          # the package
├── scripts/setup_environment.sh
└── tests/
```

## Quickstart

```bash
uv sync

uv run polyfun-gpn setup
uv run polyfun-gpn harmonize
uv run polyfun-gpn build-bg

uv run polyfun-gpn run --loci configs/loci_demo.tsv --prior entropy
uv run polyfun-gpn aggregate --prior entropy

uv run polyfun-gpn run --loci configs/loci_demo.tsv --prior none
uv run polyfun-gpn aggregate --prior none

uv run polyfun-gpn run --loci configs/loci_demo.tsv --prior uniform
uv run polyfun-gpn aggregate --prior uniform
```

Each `run` writes per-locus FINEMAP output under
`output/loci/{prior}/{locus_id}/finemap.gz` plus the PolyFun-formatted input
`sumstats.tsv` and an audit table `snpvar_audit.tsv` showing the entropy
value and `prior_source` per SNP. `aggregate` collects them into a single
`output/results/finemap.demo.{prior}.tsv`.

## Scaling to all genome-wide-significant loci

```bash
uv run polyfun-gpn run-all --prior entropy --pvalue 5e-8 --jobs 4
uv run polyfun-gpn aggregate --prior entropy
```

`run-all` calls PolyFun's `create_finemapper_jobs.py`, which auto-partitions
the genome into 3 Mb regions, applies `--pvalue-cutoff`, and emits a
per-region command list. We add the FINEMAP binary path and LD cache to
each command and run them with bounded concurrency (`--jobs`). Outputs land
under `output/genome_wide/{prior}/`. Restrict to one chromosome with
`--chrom`.

## Multi-GWAS / multi-ancestry

Each GWAS (e.g. by ancestry) uses its **own harmonised parquet** and **own
output tree** so runs never collide. The DIAMANTE T2D meta-analyses for the
five major ancestries are wired up out-of-the-box:

| Slug | YAML | Raw file | LD recommendation |
| ---- | ---- | -------- | ----------------- |
| `EUR` | `configs/datasets/EUR.yaml` | `data/gwas/EUR_Metal_LDSC-CORR_Neff.v2.txt` | UKB EUR NPZ (default) |
| `AFA` | `configs/datasets/AFA.yaml` | `data/gwas/AFA_Metal_LDSC-CORR_Neff.v2.txt` | switch to `plink` + 1000G AFR |
| `EAS` | `configs/datasets/EAS.yaml` | `data/gwas/EAS_Metal_LDSC-CORR_Neff.v2.txt` | switch to `plink` + 1000G EAS |
| `HIS` | `configs/datasets/HIS.yaml` | `data/gwas/HIS_Metal_LDSC-CORR_Neff.v2.txt` | switch to `plink` + 1000G AMR |
| `SAS` | `configs/datasets/SAS.yaml` | `data/gwas/SAS_Metal_LDSC-CORR_Neff.v2.txt` | switch to `plink` + 1000G SAS |

Discover them at any time with:

```bash
uv run polyfun-gpn list-datasets
```

**Option A — wrapper script** (uses the per-slug YAML):

```bash
scripts/run_dataset.sh EUR entropy        # harmonize + run + aggregate
scripts/run_dataset.sh AFA none

# Sweep every slug × {entropy, none}:
scripts/run_all_datasets.sh
scripts/compare_priors_all_datasets.sh    # entropy vs none per slug
```

**Option B — explicit CLI** (works without a YAML):

```bash
uv run polyfun-gpn harmonize --gwas-id EAS --gwas-raw data/gwas/EAS_Metal_LDSC-CORR_Neff.v2.txt
uv run polyfun-gpn run       --gwas-id EAS --loci configs/loci_demo.tsv --prior entropy
uv run polyfun-gpn aggregate --gwas-id EAS --prior entropy
```

With `--gwas-id SLUG` (where `SLUG` is not `default`):

- Harmonised output: `data/gwas/{SLUG}/sumstats.hg19.parquet`
- Pipeline outputs: `output/{SLUG}/` (`loci/...`, `results/finemap.demo.*.tsv`)
- FINEMAP LD disk cache: `data/ld_cache/{SLUG}/`

**Option C — YAML per study** — same paths driven from `gwas_dataset.id` +
`gwas_dataset.auto_paths: true`. Each ancestry YAML in `configs/datasets/` is
exactly that. Use `configs/datasets/_plink_template.yaml` as the starting
point for a brand-new study.

Entropy background and liftover chains stay **shared** across ancestries
(`data/background/` and `data/reference/` are not slug-scoped).

### LD panel (configurable)

FINEMAP needs an LD matrix. Two modes (`finemap.ld_mode`):

1. **`precomputed_npz` (default)** — PolyFun downloads Broad-style matrices.
   Use `finemap.ld_npz_url_prefix` as the HTTPS base ending in `UKBB_LD/`
   (or any mirror exposing the **same `{chr}_{start}_{end}` block names**
   PolyFun concatenates onto that prefix for `run`, and optionally per-row URLs
   in `finemap.ld_regions_file` for genome-wide `run-all`). The legacy YAML
   key **`ukb_ld_url_prefix`** is still accepted as an alias for the same URL.

   The stock prefix is UK Biobank white-British (≈EUR). For another population
   you only need NPZ tiling that matches GWAS **`builds.ld`** coordinate system;
   if you host such a mirror, point `ld_npz_url_prefix` (or each row of
   `ld_regions_file`) at it.

2. **`plink`** — LD is computed from a **Plink genotype triplet** on disk
   (ancestry-matched reference, e.g. 1000 Genomes). Set
   `finemap.ld_plink_prefix` to the path **without** `.bed` (PolyFun
   `--geno`). See [configs/datasets/_plink_template.yaml](configs/datasets/_plink_template.yaml).
   The non-EUR ancestry YAMLs ship with a commented-out `plink` block — flip
   to it once you have a 1000 Genomes (or other ancestry-matched) panel.

**CLI mirrors** (for `run` / `run-all`): `--ld-mode`, `--ld-npz-prefix`,
`--ld-plink`, `--ld-regions-file`.

Compare prior modes for a given slug:

```bash
uv run python scripts/compare_finemap_priormodes.py --gwas-id AFR
# or
uv run python scripts/compare_finemap_priormodes.py --results-dir output/AFR/results
```

## Configuration

Defaults live in `configs/default.yaml`. Notable knobs:

- `gwas_dataset.id` / `gwas_dataset.auto_paths` — label and optional YAML-driven
  `output/{id}` + `data/gwas/{id}/` layout (see Multi-GWAS above).
- `prior.tau` — temperature on the surprise score (default 1.0).
- `prior.epsilon` — density floor (avoids `log(0)`).
- `background.n_samples` — random sample size for the background histogram
  (default 10 M; lower it for quick experimentation).
- `finemap.max_num_causal` — passed to FINEMAP (default 5).
- `finemap.max_concurrent_jobs` — parallelism for `run` and `run-all`.
- `finemap.ld_mode` — `precomputed_npz` or `plink`.
- `finemap.ld_npz_url_prefix` — base URL for NPZ `--ld`; legacy alias
  `ukb_ld_url_prefix`.
- `finemap.ld_plink_prefix` — Plink stem (no `.bed`) when using `plink` mode.
- `finemap.ld_regions_file` — optional regions TSV/GZ for `run-all` tiling
  (CHR, START, END, URL_PREFIX); defaults to PolyFun’s bundled `ukb_regions`.
- `paths.ld_cache` — cached LD matrices; auto-scoped under `data/ld_cache/{id}`
  when `--gwas-id` is set.
- `builds` — declares hg/build of GWAS, entropy, and LD inputs. When the
  GWAS and entropy builds differ, positions are lifted over via the chain
  files downloaded by `setup`.

Override the config path with `--config path/to/config.yaml` on any
subcommand.

## Tests

```bash
uv run pytest -q
```

Covers prior math (surprise + median fallback), GWAS parser (Z, MAF, SNP
key, allele filters), locus selection (window merging, UKB block snapping),
and chromosome normalization. Liftover and FINEMAP are exercised in
integration runs once `setup` has fetched chain files and the FINEMAP
binary.

## Caveats

- **LD reference mismatch.** Match your LD source to ancestry and GWAS
  **`builds.ld`**. NPZ presets from Broad are EUR-like UK Biobank; for other
  ancestries configure `plink` LD or supply your own `ld_regions_file`/URL
  prefix — see **LD panel** above. Using EUR LD for non-European GWAS is a
  known approximation (sample-size mismatch with meta-analysis sumstats is
  separate but also worth noting).
- **Reference build.** UKB LD and the GWAS sumstats here are hg19. The
  entropy parquets are hg38 (per GPN-MSA convention). We lift over per-SNP
  positions hg19 → hg38 only for the variants we need, so we never
  rewrite the entropy parquets. If the entropy build is actually hg19, set
  `builds.entropy: hg19` in the config to skip liftover.
- **rsIDs.** The provided GWAS file has no rsIDs, so we synthesize a stable
  `SNP` key as `chr{N}:{BP}:{nonEffect}:{effect}`. PolyFun matches on
  `(CHR, BP, A1, A2)` so this is fine.
- **First-time `run-all`.** UKB LD blocks are downloaded on demand at
  ~1 GB each. A genome-wide T2D run touches a substantial fraction of the
  2,763 published blocks; reserve disk before launching.

## References

- Weissbrod et al. (2020), *PolyFun: functionally-informed fine-mapping*.
  https://www.nature.com/articles/s41588-020-00735-5
- PolyFun GitHub: https://github.com/omerwe/polyfun
- FINEMAP: http://www.christianbenner.com/
- Borzoi-informed fine mapping (the inspiration): the paper's idea applied
  here to a different per-position score (genome-wide entropy /
  conservation).

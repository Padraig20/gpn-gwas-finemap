# polyfun-gpn

PolyFun + FINEMAP fine-mapping for GWAS, with an optional per-SNP causal prior
derived from genome-wide entropy ("surprise vs. background"). Inspired by the
Borzoi-informed fine-mapping paper, but using a per-position
**conservation/entropy** score instead of Borzoi-derived predictions.

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
   `--method finemap` and on-demand UKB LD blocks.

## Prior modes (`--prior`)

| Mode | Source of `SNPVAR` | What FINEMAP sees |
| ---- | ------------------ | ----------------- |
| `none` | (column omitted) | `--non-funct`: uniform causal prior baked into FINEMAP |
| `entropy` | `exp(τ · −log f_bg(e))` from genome-wide entropy | per-SNP prior derived from conservation |

For `entropy`, variants without an entropy lookup fall back to the locus
median (tagged `prior_source = "median_fallback"` in `snpvar_audit.tsv`).

## Project layout

```
polyfun-gpn/
├── configs/
│   ├── default-EUR.yaml      # full config for the EUR T2D meta-analysis
│   ├── default-EAS.yaml      # full config for the EAS T2D meta-analysis
│   └── loci_demo.tsv         # five well-known T2D loci (hg19)
├── data/
│   ├── entropy/              # provided: entropy_chr*.parquet
│   ├── gwas/                 # provided: GWAS sumstats
│   ├── reference/            # filled by `setup`: liftover chain files
│   ├── background/           # filled by `build-bg`: entropy_bg.npz
│   └── ld_cache/             # FINEMAP --cache-dir
├── external/
│   ├── polyfun/              # cloned by `setup`
│   └── bin/finemap           # FINEMAP v1.4.1 binary, fetched by `setup`
├── output/                   # default; override with --output-dir
│   ├── loci/{prior}/{locus}/ # per-locus sumstats + FINEMAP outputs
│   └── results/              # aggregated results
├── src/polyfun_gpn/          # the package
├── scripts/setup_environment.sh
└── tests/
```

## Quickstart

```bash
uv sync

# one-time bootstrap (PolyFun, FINEMAP binary, chain files)
uv run polyfun-gpn setup -c configs/default-EUR.yaml

# preprocessing (idempotent; safe to re-run)
uv run polyfun-gpn harmonize -c configs/default-EUR.yaml
uv run polyfun-gpn build-bg  -c configs/default-EUR.yaml

# fine-map: prior mode comes from the YAML (default: entropy)
uv run polyfun-gpn run       -c configs/default-EUR.yaml
uv run polyfun-gpn aggregate -c configs/default-EUR.yaml

# or override the prior on the CLI to compare against no-prior FINEMAP
uv run polyfun-gpn run       -c configs/default-EUR.yaml --prior none
uv run polyfun-gpn aggregate -c configs/default-EUR.yaml --prior none
```

Each `run` writes per-locus FINEMAP output under
`{output_dir}/loci/{prior}/{locus_id}/finemap.gz` plus the PolyFun-formatted
input `sumstats.tsv` and (for `entropy`) an audit table `snpvar_audit.tsv`
showing the entropy value and `prior_source` per SNP. `aggregate` collects
them into `{output_dir}/results/finemap.demo.{prior}.tsv`.

## Configurable inputs

Everything you need to run the pipeline is in the YAML config; every CLI
flag has a YAML counterpart, and CLI flags (when passed) override YAML.

| Knob | CLI flag | YAML key |
| ---- | -------- | -------- |
| Raw GWAS TSV | `--gwas-raw PATH` | `paths.gwas_raw` |
| Output directory | `--output-dir PATH` (`-o`) | `paths.output_dir` |
| Loci TSV | `--loci PATH` | `paths.loci` |
| Prior mode | `--prior {none,entropy}` | `prior.mode` |
| LD source mode | `--ld-mode {precomputed_npz,plink}` | `finemap.ld_mode` |
| LD NPZ URL prefix | `--ld-npz-prefix URL` | `finemap.ld_npz_url_prefix` |
| LD Plink prefix | `--ld-plink PATH` | `finemap.ld_plink_prefix` |

The remaining YAML sections (`paths.*` for caches/externals, `builds`,
`background`, `prior.tau` / `prior.epsilon`, `finemap.*`) can also be edited
freely; they don't have dedicated CLI flags because they rarely change per
run.

**Example — fully YAML-driven (no CLI overrides):**

```bash
uv run polyfun-gpn run -c configs/default-EUR.yaml
```

`configs/default-EUR.yaml` already pins `paths.gwas_raw`, `paths.output_dir`,
`paths.loci`, `prior.mode`, and the LD source for that study.

**Example — override one knob at a time from the CLI:**

```bash
uv run polyfun-gpn run \
    -c configs/default-EUR.yaml \
    --prior none \
    --output-dir output/EUR_noprior
```

**Example — Plink LD reference (ancestry-matched panel):**

```bash
uv run polyfun-gpn run \
    -c configs/default-EAS.yaml \
    --ld-mode plink \
    --ld-plink data/ld_panels/1000G_EAS
```

For a brand-new study, copy one of `configs/default-EUR.yaml` /
`configs/default-EAS.yaml`, edit the paths in place, and pass
`-c configs/your.yaml` on every subcommand.

## LD panel

FINEMAP needs an LD matrix. Two modes (`finemap.ld_mode`):

1. **`precomputed_npz` (default)** — PolyFun downloads Broad-style matrices.
   Use `finemap.ld_npz_url_prefix` (or `--ld-npz-prefix`) as the HTTPS base
   ending in `UKBB_LD/` (or any mirror exposing the same `{chr}_{start}_{end}`
   block names). The legacy YAML key `ukb_ld_url_prefix` is still accepted as
   an alias.

   The stock prefix is UK Biobank white-British (≈EUR). For another
   population you only need an NPZ tiling that matches your GWAS
   `builds.ld` coordinate system.

2. **`plink`** — LD is computed from a Plink genotype triplet on disk
   (ancestry-matched reference, e.g. 1000 Genomes). Set
   `finemap.ld_plink_prefix` (or `--ld-plink`) to the path **without** `.bed`.

## Configuration

Per-study YAMLs live in `configs/` (one per GWAS, e.g.
`default-EUR.yaml`, `default-EAS.yaml`). Notable knobs:

- `prior.tau` — temperature on the surprise score (default 1.0).
- `prior.epsilon` — density floor (avoids `log(0)`).
- `background.n_samples` — random sample size for the background histogram
  (default 10 M; lower it for quick experimentation).
- `finemap.max_num_causal` — passed to FINEMAP (default 5).
- `finemap.max_concurrent_jobs` — parallelism for `run`.
- `finemap.ld_mode` — `precomputed_npz` or `plink`.
- `finemap.ld_npz_url_prefix` — base URL for NPZ `--ld`; legacy alias
  `ukb_ld_url_prefix`.
- `finemap.ld_plink_prefix` — Plink stem (no `.bed`) when using `plink` mode.
- `paths.ld_cache` — cached LD matrices.
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
chromosome normalization, and config/CLI overrides. Liftover and FINEMAP are
exercised in integration runs once `setup` has fetched chain files and the
FINEMAP binary.

## Caveats

- **LD reference mismatch.** Match your LD source to ancestry and GWAS
  `builds.ld`. NPZ presets from Broad are EUR-like UK Biobank; for other
  ancestries configure `plink` LD or supply your own NPZ URL prefix.
- **Reference build.** UKB LD and the bundled GWAS sumstats here are hg19.
  The entropy parquets are hg38 (per GPN-MSA convention). We lift over
  per-SNP positions hg19 → hg38 only for the variants we need, so we never
  rewrite the entropy parquets. If the entropy build is actually hg19, set
  `builds.entropy: hg19` in the config to skip liftover.
- **rsIDs.** The provided GWAS file has no rsIDs, so we synthesize a stable
  `SNP` key as `chr{N}:{BP}:{nonEffect}:{effect}`. PolyFun matches on
  `(CHR, BP, A1, A2)` so this is fine.

## References

- Weissbrod et al. (2020), *PolyFun: functionally-informed fine-mapping*.
  https://www.nature.com/articles/s41588-020-00735-5
- PolyFun GitHub: https://github.com/omerwe/polyfun
- FINEMAP: http://www.christianbenner.com/
- Borzoi-informed fine mapping (the inspiration): the paper's idea applied
  here to a different per-position score (genome-wide entropy /
  conservation).

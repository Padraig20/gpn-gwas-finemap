# LD panels for FINEMAP

Goal: produce an ancestry-matched 1000G PLINK reference for each ancestry,
then point `polyfun-gpn run --ld-mode plink` at it. PolyFun's `finemapper.py`
gets `--geno <prefix>` and computes per-locus LD on the fly, so we don't
need to pre-build NPZ matrices ourselves.

## Inputs (on the cluster)

All paths live under `/links/groups/cai/REF/1KGP/phase3/` unless noted.

| What | EUR | EAS |
| ---- | --- | --- |
| Combined PLINK fileset | `allchr.EUR.biallelicsnps.{bed,bim,fam}` | `allchr.EAS.biallelicsnps.{bed,bim,fam}` |
| Per-chrom VCFs (GRCh38) | `phase3.chr*.GRCh38.GT.crossmap.vcf*` (both ancestries) ||
| Pedigree | `integrated_call_samples_v3.20200731.ALL.ped` (both) ||
| Related to exclude | `20140625_related_individuals.txt` (both) ||
| Freq (S-LDSC) | `/local1/hdata/scores/gpnstar/frq/1000G.{EUR,EAS}.hg38.*.frq` ||
| Weights / baseline (S-LDSC) | `/local1/hdata/scores/gpnstar/{weights,baseline}{,eas}/` ||

The `frq`, `weights*`, and `baseline*` directories are for PolyFun's
S-LDSC pipeline. **This project doesn't use them** — we substitute
entropy-derived priors via `polyfun-gpn run --prior entropy`. So you only
need the PLINK files and the related-individuals list for LD.

> **Build mismatch — read this first.**
> The 1000G files above are on GRCh38 (the freq files are named
> `...hg38.*.frq` and the VCFs are `*.GRCh38.GT.crossmap.vcf`). The
> bundled GWAS sumstats and the default `builds.ld: hg19` in
> `configs/default-{EUR,EAS}.yaml` are hg19, and the loci TSVs we
> generate carry `lead_pos_hg19`. If you fine-map without harmonising
> coordinates, PolyFun joins LD ⨯ sumstats on `(CHR, BP, A1, A2)` and the
> intersection will be empty. Two ways out:
>
> 1. **Lift 1000G hg38 → hg19 once** (recommended; keeps the YAML default
>    `builds.ld: hg19`). E.g. run CrossMap on the VCFs against
>    `data/reference/hg38ToHg19.over.chain.gz`, then re-PLINK; or remap
>    the `.bim` if you trust the position-level liftover for biallelic
>    SNVs only.
> 2. **Run the whole pipeline in hg38**: set `builds.ld: hg38` and also
>    lift over your loci TSV / harmonised sumstats to hg38 first.
>
> The scripts here are build-agnostic — they just consume whatever PLINK
> files you feed them. Make sure the coordinates of your loci TSV match
> the build of the PLINK reference.

## PLINK `--remove` format

PLINK 1.9 expects two whitespace-separated columns (`FID  IID`). The
1000G related file has one Sample ID per data line:

```
Sample	Population	Gender	Reason for exclusion
HG00124	GBR	female	Second Order:HG00119
HG00501	CHS	female	Sibling:HG00524
...
```

and the `.fam` files use `FID == IID == Sample`:

```
HG00096	HG00096	0	0	0	-9
HG00097	HG00097	0	0	0	-9
...
```

So we just emit `Sample Sample` pairs (`prepare_unrelated_reference.sh`
does this for you with one line of awk).

## Step 1 — build the unrelated reference (once per ancestry)

```bash
bash scripts/compute_ld/prepare_unrelated_reference.sh EUR
bash scripts/compute_ld/prepare_unrelated_reference.sh EAS
```

Output (default `data/ld_panels/`):

```
data/ld_panels/
├── 1000G.EUR.unrelated.{bed,bim,fam}
└── 1000G.EAS.unrelated.{bed,bim,fam}
```

Override paths via env vars when the cluster layout changes:

```bash
REFDIR=/somewhere/else \
RELATED=/somewhere/else/related.txt \
OUT_DIR=/scratch/$USER/ld_panels \
PLINK_BIN=plink1.9 \
    bash scripts/compute_ld/prepare_unrelated_reference.sh EUR --overwrite
```

## Step 2 (optional) — per-locus PLINK files

PolyFun `--geno` handles per-locus windowing internally, so this step is
**not required** to run fine-mapping. It's useful when:

* you want to inspect per-locus genotypes / SNP counts for QC,
* you want a smaller PLINK per locus to ship around a cluster, or
* you want a per-locus "SNP list" — that's just column 2 of the per-locus
  `.bim` file.

Feed it the shared-loci TSV produced by
`scripts/find_shared_loci.py` (or any TSV with the same schema as
`configs/loci_demo.tsv`):

```bash
bash scripts/compute_ld/extract_loci_plink.sh EUR configs/loci_shared_EUR_EAS.tsv
bash scripts/compute_ld/extract_loci_plink.sh EAS configs/loci_shared_EUR_EAS.tsv

# Parallel (GNU parallel must be on PATH):
JOBS=8 bash scripts/compute_ld/extract_loci_plink.sh EUR configs/loci_shared_EUR_EAS.tsv
```

Output:

```
data/ld_panels/loci/EUR/<locus_id>.{bed,bim,fam,plink.log}
data/ld_panels/loci/EAS/<locus_id>.{bed,bim,fam,plink.log}
```

## Step 3 — wire it into `polyfun-gpn`

Edit your YAML (or pass on the CLI) so FINEMAP uses the new prefix:

```yaml
# configs/default-EAS.yaml
finemap:
  ld_mode: plink
  ld_plink_prefix: data/ld_panels/1000G.EAS.unrelated
```

Equivalent one-liner:

```bash
uv run polyfun-gpn run \
    -c configs/default-EAS.yaml \
    --loci configs/loci_shared_EUR_EAS.tsv \
    --ld-mode plink \
    --ld-plink data/ld_panels/1000G.EAS.unrelated
```

Cross-ancestry runs are just two invocations — same loci TSV, different
`--ld-plink` and `-c` per ancestry:

```bash
uv run polyfun-gpn run -c configs/default-EUR.yaml \
    --loci configs/loci_shared_EUR_EAS.tsv \
    --ld-mode plink --ld-plink data/ld_panels/1000G.EUR.unrelated \
    --output-dir output/EUR_shared

uv run polyfun-gpn run -c configs/default-EAS.yaml \
    --loci configs/loci_shared_EUR_EAS.tsv \
    --ld-mode plink --ld-plink data/ld_panels/1000G.EAS.unrelated \
    --output-dir output/EAS_shared
```

## Troubleshooting

* **`plink` not found.** Set `PLINK_BIN=/path/to/plink` (or `plink1.9`).
  Both scripts respect that env var.
* **`File not found: allchr.<ANC>.biallelicsnps.bed`.** Confirm the
  cluster path; if your fileset is per-chromosome instead of `allchr.*`,
  PLINK-merge them once with `plink --merge-list ...` or override
  `REFDIR`/`BFILE`.
* **PolyFun finds 0 overlapping SNPs.** Almost always a build mismatch
  (hg19 ⟷ hg38). See the warning at the top.
* **Empty per-locus PLINK.** Check `data/ld_panels/loci/<ANC>/<id>.plink.log` —
  a locus on chrY/chrM (or an X locus when the reference is autosomes-only)
  will yield 0 variants. The extract script logs and skips these instead of
  aborting the batch.

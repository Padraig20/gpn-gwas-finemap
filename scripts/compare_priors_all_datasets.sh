#!/usr/bin/env bash
# Run scripts/compare_finemap_priormodes.py for every dataset in
# configs/datasets/datasets.tsv. Skips slugs that are missing aggregated TSVs.
#
# Output:
#   stdout: full report per slug
#   compare_priors_<slug>.csv : per-locus summary (one file per slug)
set -uo pipefail

cd "$(dirname "$0")/.."

declare -a SLUGS
mapfile -t SLUGS < <(awk 'NR>1 && $1 !~ /^#/ {print $1}' configs/datasets/datasets.tsv)

for slug in "${SLUGS[@]}"; do
  res_dir="output/${slug}/results"
  if [[ ! -f "${res_dir}/finemap.demo.entropy.tsv" || ! -f "${res_dir}/finemap.demo.none.tsv" ]]; then
    echo "[compare-all] SKIP ${slug} (missing entropy/none TSVs in ${res_dir})"
    continue
  fi
  echo "============================================================"
  echo "  ${slug}"
  echo "============================================================"
  uv run python scripts/compare_finemap_priormodes.py \
    --gwas-id "${slug}" \
    --per-locus-csv "compare_priors_${slug}.csv" || \
    echo "[compare-all] FAIL ${slug}"
done

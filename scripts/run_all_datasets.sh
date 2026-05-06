#!/usr/bin/env bash
# Run end-to-end demo across every dataset listed in configs/datasets/datasets.tsv.
#
# Usage:
#   scripts/run_all_datasets.sh [PRIOR1 PRIOR2 ...]
#
# Defaults to: entropy none
# Each (slug, prior) pair is processed sequentially; per-call failures are
# logged but do not stop the loop.
set -uo pipefail

cd "$(dirname "$0")/.."

PRIORS=("$@")
if [[ ${#PRIORS[@]} -eq 0 ]]; then
  PRIORS=(entropy none)
fi

mapfile -t SLUGS < <(awk 'NR>1 && $1 !~ /^#/ {print $1}' configs/datasets/datasets.tsv)

echo "[run_all_datasets] datasets=${SLUGS[*]}"
echo "[run_all_datasets] priors=${PRIORS[*]}"

declare -a FAILED=()
for slug in "${SLUGS[@]}"; do
  for prior in "${PRIORS[@]}"; do
    echo "------------------------------------------------------------"
    echo "  ${slug} / ${prior}"
    echo "------------------------------------------------------------"
    if ! scripts/run_dataset.sh "${slug}" "${prior}"; then
      echo "[run_all_datasets] FAIL slug=${slug} prior=${prior}" >&2
      FAILED+=("${slug}/${prior}")
    fi
  done
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "[run_all_datasets] failed combos: ${FAILED[*]}" >&2
  exit 1
fi
echo "[run_all_datasets] all combinations completed."

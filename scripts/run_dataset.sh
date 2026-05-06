#!/usr/bin/env bash
# Run end-to-end demo for one ancestry slug + one prior mode.
#
# Usage:
#   scripts/run_dataset.sh SLUG PRIOR [extra polyfun-gpn args...]
#
# Example:
#   scripts/run_dataset.sh EUR entropy
#   scripts/run_dataset.sh AFA none
#   scripts/run_dataset.sh EAS entropy --loci configs/loci_demo.tsv
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 SLUG PRIOR [extra args]" >&2
  echo "  SLUG  : one of EUR, AFA, EAS, HIS, SAS (or any with configs/datasets/<SLUG>.yaml)" >&2
  echo "  PRIOR : entropy | none | uniform" >&2
  exit 2
fi

SLUG="$1"
PRIOR="$2"
shift 2

cd "$(dirname "$0")/.."
CFG="configs/datasets/${SLUG}.yaml"

if [[ ! -f "${CFG}" ]]; then
  echo "Missing ${CFG}; expected per-slug YAML under configs/datasets/." >&2
  exit 3
fi

echo "[run_dataset] slug=${SLUG} prior=${PRIOR} config=${CFG}"

uv run polyfun-gpn harmonize -c "${CFG}" --gwas-id "${SLUG}" "$@"
uv run polyfun-gpn run -c "${CFG}" --gwas-id "${SLUG}" --prior "${PRIOR}" "$@"
uv run polyfun-gpn aggregate -c "${CFG}" --gwas-id "${SLUG}" --prior "${PRIOR}"

echo "[run_dataset] DONE  slug=${SLUG} prior=${PRIOR}"
echo "[run_dataset] aggregated TSV -> output/${SLUG}/results/finemap.demo.${PRIOR}.tsv"

#!/usr/bin/env bash
# Build an "unrelated" 1000G PLINK reference for one ancestry, suitable for
# PolyFun / FINEMAP via `polyfun-gpn run --ld-mode plink`.
#
# What it does:
#   1. Convert the 1000G phase-3 related-individuals list into the
#      ``FID  IID`` 2-column form that ``plink --remove`` expects.
#   2. Run ``plink --bfile allchr.${ANC}.biallelicsnps --remove ... --make-bed``
#      to drop those individuals from the all-chromosome PLINK fileset.
#
# Output (default ``data/ld_panels/``):
#   1000G.${ANC}.unrelated.{bed,bim,fam}
#
# Usage:
#   bash scripts/compute_ld/prepare_unrelated_reference.sh EUR
#   bash scripts/compute_ld/prepare_unrelated_reference.sh EAS --overwrite
#
# Overridable via env vars (defaults match the cluster paths in README.md):
#   REFDIR     base dir holding allchr.${ANC}.biallelicsnps.{bed,bim,fam}
#   RELATED    20140625_related_individuals.txt
#   OUT_DIR    where to write the unrelated PLINK output (default data/ld_panels)
#   PLINK_BIN  plink executable (default: plink)

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: prepare_unrelated_reference.sh ANCESTRY [--overwrite]

ANCESTRY  EUR | EAS | (anything that matches allchr.${ANC}.biallelicsnps.*)

Env overrides: REFDIR, RELATED, OUT_DIR, PLINK_BIN.
USAGE
}

if [[ $# -lt 1 ]]; then
    usage; exit 2
fi

ANC="$1"; shift
OVERWRITE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --overwrite|-f) OVERWRITE=1 ;;
        -h|--help)      usage; exit 0 ;;
        *)              echo "[ld] unknown arg: $1" >&2; usage; exit 2 ;;
    esac
    shift
done

REFDIR="${REFDIR:-/links/groups/cai/REF/1KGP/phase3}"
RELATED="${RELATED:-/links/groups/cai/REF/1KGP/20140625_related_individuals.txt}"
OUT_DIR="${OUT_DIR:-data/ld_panels}"
PLINK_BIN="${PLINK_BIN:-plink}"

BFILE="${REFDIR}/allchr.${ANC}.biallelicsnps"
OUT_PREFIX="${OUT_DIR}/1000G.${ANC}.unrelated"

# Fail fast: all three PLINK files must exist, and the related list too.
for ext in bed bim fam; do
    if [[ ! -e "${BFILE}.${ext}" ]]; then
        echo "[ld] ERROR: missing input ${BFILE}.${ext}" >&2
        exit 1
    fi
done
if [[ ! -e "${RELATED}" ]]; then
    echo "[ld] ERROR: missing related-individuals file: ${RELATED}" >&2
    exit 1
fi

if [[ -e "${OUT_PREFIX}.bed" && "${OVERWRITE}" -eq 0 ]]; then
    echo "[ld] ${OUT_PREFIX}.bed already exists; pass --overwrite to recompute."
    exit 0
fi

mkdir -p "${OUT_DIR}"
TMPDIR_LOCAL="$(mktemp -d -t polyfun-gpn-ld-XXXXXX)"
trap 'rm -rf "${TMPDIR_LOCAL}"' EXIT

REMOVE_TXT="${TMPDIR_LOCAL}/related.remove.fid_iid.txt"
# 1000G related file: header line + "Sample Population Gender Reason..."
# .fam files have FID == IID == Sample, so emit "Sample Sample" pairs.
awk 'NR > 1 && $1 != "" { print $1, $1 }' "${RELATED}" > "${REMOVE_TXT}"
N_REMOVE=$(wc -l < "${REMOVE_TXT}" | tr -d ' ')
echo "[ld] ${ANC}: dropping ${N_REMOVE} related individuals"

echo "[ld] ${ANC}: ${PLINK_BIN} --bfile ${BFILE} -> ${OUT_PREFIX}"
"${PLINK_BIN}" \
    --bfile "${BFILE}" \
    --remove "${REMOVE_TXT}" \
    --make-bed \
    --out "${OUT_PREFIX}"

# Sanity check: how many samples / variants ended up in the output.
N_SAMPLES=$(wc -l < "${OUT_PREFIX}.fam" | tr -d ' ')
N_VARIANTS=$(wc -l < "${OUT_PREFIX}.bim" | tr -d ' ')
echo "[ld] ${ANC}: wrote ${OUT_PREFIX}.{bed,bim,fam}  (${N_SAMPLES} samples, ${N_VARIANTS} variants)"

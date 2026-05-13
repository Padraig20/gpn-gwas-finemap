#!/usr/bin/env bash
# Extract a per-locus PLINK fileset from an ancestry-specific 1000G
# reference, using a loci TSV (same schema as configs/loci_demo.tsv:
# locus_id  chrom  start  end  gene  lead_rsid  lead_pos_hg19).
#
# OPTIONAL step: PolyFun's --geno does its own per-locus windowing, so for
# `polyfun-gpn run --ld-mode plink` you can usually just point at the whole
# 1000G.${ANC}.unrelated prefix from prepare_unrelated_reference.sh. Use
# this script when you want:
#   * per-locus genotype dumps for diagnostics / QC
#   * a smaller PLINK per locus for distributed fine-mapping
#   * the .bim file as a "SNP list per locus"
#
# Usage:
#   bash scripts/compute_ld/extract_loci_plink.sh EUR configs/loci_shared_EUR_EAS.tsv
#   bash scripts/compute_ld/extract_loci_plink.sh EAS configs/loci_shared_EUR_EAS.tsv
#
# Env overrides:
#   REF_PREFIX   default: data/ld_panels/1000G.${ANC}.unrelated
#   OUT_DIR      default: data/ld_panels/loci/${ANC}
#   PLINK_BIN    default: plink
#   JOBS         GNU-parallel concurrency (default 1 = sequential)

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: extract_loci_plink.sh ANCESTRY LOCI_TSV

ANCESTRY  EUR | EAS | ...
LOCI_TSV  TSV with header locus_id\tchrom\tstart\tend\t...

Env overrides: REF_PREFIX, OUT_DIR, PLINK_BIN, JOBS.
USAGE
}

if [[ $# -lt 2 ]]; then
    usage; exit 2
fi

ANC="$1"
LOCI_TSV="$2"

REF_PREFIX="${REF_PREFIX:-data/ld_panels/1000G.${ANC}.unrelated}"
OUT_DIR="${OUT_DIR:-data/ld_panels/loci/${ANC}}"
PLINK_BIN="${PLINK_BIN:-plink}"
JOBS="${JOBS:-1}"

for ext in bed bim fam; do
    if [[ ! -e "${REF_PREFIX}.${ext}" ]]; then
        echo "[ld] ERROR: ${REF_PREFIX}.${ext} not found." >&2
        echo "      Run scripts/compute_ld/prepare_unrelated_reference.sh ${ANC} first." >&2
        exit 1
    fi
done
if [[ ! -e "${LOCI_TSV}" ]]; then
    echo "[ld] ERROR: loci TSV not found: ${LOCI_TSV}" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"

# Single-locus extract that we can either run inline or hand to GNU parallel.
# The function is exported and re-invoked from the parallel branch via
# ``bash -c``. Keep it self-contained.
_extract_one() {
    local locus_id="$1" chrom="$2" start="$3" end="$4"
    local out_prefix="${OUT_DIR}/${locus_id}"
    if [[ -e "${out_prefix}.bed" ]]; then
        echo "[ld] ${locus_id}: ${out_prefix}.bed exists, skipping."
        return 0
    fi
    echo "[ld] ${locus_id}: chr${chrom}:${start}-${end}"
    if ! "${PLINK_BIN}" \
            --bfile "${REF_PREFIX}" \
            --chr "${chrom}" \
            --from-bp "${start}" \
            --to-bp "${end}" \
            --make-bed \
            --out "${out_prefix}" \
            > "${out_prefix}.plink.log" 2>&1; then
        # Empty regions (e.g. chrY when the reference is autosomes) make
        # plink exit non-zero. Don't kill the whole batch -- surface it.
        echo "[ld] ${locus_id}: plink failed (see ${out_prefix}.plink.log)" >&2
        rm -f "${out_prefix}.bed" "${out_prefix}.bim" "${out_prefix}.fam"
        return 0
    fi
}
export -f _extract_one
export OUT_DIR REF_PREFIX PLINK_BIN

# Skip the header. Read the first 4 cols of the loci TSV (rest discarded).
LOCI_ROWS="$(tail -n +2 "${LOCI_TSV}" | awk -F'\t' 'NF >= 4 { print $1"\t"$2"\t"$3"\t"$4 }')"

if [[ -z "${LOCI_ROWS}" ]]; then
    echo "[ld] ERROR: ${LOCI_TSV} has no data rows." >&2
    exit 1
fi

N_LOCI=$(echo "${LOCI_ROWS}" | wc -l | tr -d ' ')
echo "[ld] ${ANC}: extracting ${N_LOCI} loci -> ${OUT_DIR}/   (JOBS=${JOBS})"

if [[ "${JOBS}" -gt 1 ]] && command -v parallel >/dev/null 2>&1; then
    echo "${LOCI_ROWS}" \
        | parallel --colsep '\t' -j "${JOBS}" \
            _extract_one {1} {2} {3} {4}
else
    while IFS=$'\t' read -r locus_id chrom start end; do
        _extract_one "${locus_id}" "${chrom}" "${start}" "${end}"
    done <<< "${LOCI_ROWS}"
fi

# Summary count: how many loci actually produced a .bed.
N_DONE=$(find "${OUT_DIR}" -maxdepth 1 -name '*.bed' | wc -l | tr -d ' ')
echo "[ld] ${ANC}: ${N_DONE}/${N_LOCI} loci have ${OUT_DIR}/<id>.{bed,bim,fam}"
echo "[ld]    tip: the .bim file is your per-locus SNP list (col 2 = variant id)."

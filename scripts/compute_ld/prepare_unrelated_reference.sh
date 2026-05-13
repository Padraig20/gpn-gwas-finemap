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
# PolyFun keys its LD cache on `os.path.basename(plink_prefix) + .chr.bp1.bp2.{npz,gz}`,
# so any stale cache from a previous (dirty) panel build will be reloaded
# verbatim by `find_cached_ld_file` next run and re-trigger PolyFun's
# duplicate check. We invalidate the matching cache files alongside the
# panel here. Default matches `paths.ld_cache` in configs/default-*.yaml.
LD_CACHE_DIR="${LD_CACHE_DIR:-data/ld_cache}"
PLINK_BIN="${PLINK_BIN:-plink}"

BFILE="${REFDIR}/allchr.${ANC}.biallelicsnps"
OUT_PREFIX="${OUT_DIR}/1000G.${ANC}.unrelated"
OUT_BASENAME="$(basename "${OUT_PREFIX}")"

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

# --- duplicate-variant removal -----------------------------------------------
# PolyFun's `set_snpid_index` builds `snpid = chrom.bp.A1.A2` with alleles
# sorted alphabetically, then refuses to proceed if any snpid repeats. The
# 1000G phase-3 PLINK files contain mirror-encoded rows (the same biallelic
# SNP listed once as e.g. A/C and once as C/A) that collapse to the same
# snpid and trip that check.
#
# We used to delegate this to ``plink --list-duplicate-vars``, but the
# modifier syntax (``ids-only suppress-first``) isn't accepted by every
# PLINK 1.9 build -- some exit code 8 silently. Detecting in awk is faster
# (one .bim pass, no genotype I/O), uses *exactly* PolyFun's equivalence
# class (chrom + bp + sorted allele pair), and has no PLINK-version
# coupling. We emit variant IDs (column 2 of the .bim) of every 2nd+
# occurrence and `--exclude` them in the make-bed below; the first
# occurrence stays in the output.
DUPVAR="${TMPDIR_LOCAL}/${ANC}.dupvars.txt"
echo "[ld] ${ANC}: scanning ${BFILE}.bim for duplicate variants (awk)"
awk 'BEGIN { OFS="\t" } {
    a=$5; b=$6;
    if (a > b) { t=a; a=b; b=t }
    key=$1"."$4"."a"."b;
    if (key in seen) {
        print $2
    } else {
        seen[key] = 1
    }
}' "${BFILE}.bim" > "${DUPVAR}"
N_DUP=$(wc -l < "${DUPVAR}" | tr -d ' ')
echo "[ld] ${ANC}: ${N_DUP} duplicate variant IDs flagged for exclusion"

# --exclude on an empty list errors in some plink builds; only pass the
# flag when we actually have something to drop.
PLINK_EXTRA_ARGS=()
if [[ "${N_DUP}" -gt 0 ]]; then
    PLINK_EXTRA_ARGS=(--exclude "${DUPVAR}")
fi

echo "[ld] ${ANC}: ${PLINK_BIN} --bfile ${BFILE} -> ${OUT_PREFIX}"
# Stream PLINK's stdout/stderr through so any failure is visible on the
# terminal (and ends up in cluster job logs) instead of disappearing into
# a temp file the EXIT trap then deletes.
if ! "${PLINK_BIN}" \
        --bfile "${BFILE}" \
        --remove "${REMOVE_TXT}" \
        "${PLINK_EXTRA_ARGS[@]}" \
        --make-bed \
        --out "${OUT_PREFIX}"; then
    echo "[ld] ${ANC}: ERROR -- PLINK make-bed failed (see output above)." >&2
    echo "[ld] ${ANC}:   PLINK_BIN=${PLINK_BIN} ; ${PLINK_BIN} --version  may help diagnose." >&2
    exit 1
fi

# Invalidate any stale LD-cache files PolyFun may have written from an
# earlier (dirty) panel build. PolyFun's `find_cached_ld_file` matches on
# `{basename}.{chr}.{bp1}.{bp2}.{npz,gz}` -- it reads the .gz SNP list
# verbatim and re-runs set_snpid_index on it, so an old .gz with the
# mirror-encoded duplicates would re-trigger the same ValueError even
# after we rebuilt the panel. The .npz alone isn't enough; the .gz is
# what carries the duplicates. Wipe matching {npz,gz,bcor} pairs.
if [[ -d "${LD_CACHE_DIR}" ]]; then
    CACHE_HITS=( "${LD_CACHE_DIR}/${OUT_BASENAME}".*.{npz,gz,bcor} )
    # The glob can stay literal when nothing matches; filter to real files.
    REMOVED=0
    for f in "${CACHE_HITS[@]}"; do
        if [[ -e "$f" ]]; then
            rm -f "$f"
            REMOVED=$((REMOVED + 1))
        fi
    done
    if [[ "${REMOVED}" -gt 0 ]]; then
        echo "[ld] ${ANC}: cleared ${REMOVED} stale LD-cache files under ${LD_CACHE_DIR}/"
    fi
fi

# Sanity check: how many samples / variants ended up in the output, and
# (cheap) confirm no normalised duplicates survive. The awk computes the
# PolyFun-style snpid by sorting the two alleles, then prints how many
# rows share a snpid -- should be 0 after the --exclude pass.
N_SAMPLES=$(wc -l < "${OUT_PREFIX}.fam" | tr -d ' ')
N_VARIANTS=$(wc -l < "${OUT_PREFIX}.bim" | tr -d ' ')
N_RESIDUAL_DUPS=$(
    awk 'BEGIN{OFS="."} {
        a=$5; b=$6;
        if (a > b) { t=a; a=b; b=t }
        key=$1"."$4"."a"."b;
        c[key]++
    } END {
        n=0; for (k in c) if (c[k] > 1) n++;
        print n
    }' "${OUT_PREFIX}.bim"
)
echo "[ld] ${ANC}: wrote ${OUT_PREFIX}.{bed,bim,fam}  (${N_SAMPLES} samples, ${N_VARIANTS} variants)"
if [[ "${N_RESIDUAL_DUPS}" -ne 0 ]]; then
    echo "[ld] ${ANC}: WARNING -- ${N_RESIDUAL_DUPS} residual normalised-allele duplicates in ${OUT_PREFIX}.bim" >&2
    echo "[ld] ${ANC}:   PolyFun will reject this panel. Inspect the .bim for unusual entries." >&2
fi

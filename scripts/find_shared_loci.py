#!/usr/bin/env python3
"""Find GWAS peaks shared across multiple ancestries.

Given two or more GWAS summary-stat files, this script:
  1. Pulls genome-wide-significant SNPs from each input (default P < 5e-8).
  2. Greedily clusters them into peaks per GWAS: top sig SNP becomes the
     lead, gets a +/- ``--window-bp`` window, all sig SNPs inside that
     window are absorbed, repeat (mirrors ``loci.select.select_lead_loci``).
  3. Keeps peaks present in *every* input GWAS: for each peak in the first
     GWAS we require an overlapping peak in each of the others.
  4. Picks the consensus lead = SNP with the largest summed -log10(P)
     across GWAS, restricted to the intersection of overlapping windows.
  5. Emits a loci TSV with the same schema as ``configs/loci_demo.tsv``:

        locus_id  chrom  start  end  gene  lead_rsid  lead_pos_hg19

     so the output can be passed straight to fine-mapping:

        uv run polyfun-gpn run -c configs/default-EUR.yaml \\
            --loci <output.tsv>

Inputs are auto-detected:
  * ``.parquet``/``.pq``: harmonised PolyFun-style sumstats (``CHR``, ``BP``,
    ``P``, ``SNP``) -- e.g. the output of ``polyfun-gpn harmonize``.
  * anything else: DIAMANTE-style raw TSV with columns ``Chromsome``,
    ``Position``, ``Pval`` (and ``EffectAllele``/``NonEffectAllele`` if you
    want a synthetic SNP key).

Coordinate system: whatever the inputs use. All inputs MUST share the same
build (this script does not lift over). If you mix builds, harmonise first.

Example:
    uv run python scripts/find_shared_loci.py \\
        data/gwas/EUR/sumstats.hg19.parquet \\
        data/gwas/EAS/sumstats.hg19.parquet \\
        --output configs/loci_shared_EUR_EAS.tsv
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import polars as pl
import typer


GENOME_WIDE_SIGNIFICANCE = 5e-8
DEFAULT_WINDOW_BP = 1_500_000

app = typer.Typer(
    name="find-shared-loci",
    help="Identify GWAS peaks present in every input ancestry.",
    add_completion=False,
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# Input reading
# ---------------------------------------------------------------------------


# Column synonyms accepted in raw TSVs. PolyFun's harmonised parquet always
# uses CHR/BP/P/SNP, so the parquet branch below doesn't need this map.
_CHR_ALIASES = ("CHR", "chr", "chrom", "Chromsome", "Chromosome", "CHROM")
_BP_ALIASES = ("BP", "bp", "pos", "POS", "Position", "position")
_P_ALIASES = ("P", "p", "Pval", "PVAL", "pvalue", "P_VALUE", "PVALUE")
_A1_ALIASES = ("A1", "a1", "EffectAllele", "effect_allele")
_A2_ALIASES = ("A2", "a2", "NonEffectAllele", "non_effect_allele", "OtherAllele")


def _pick(cols: list[str], aliases: tuple[str, ...]) -> Optional[str]:
    for a in aliases:
        if a in cols:
            return a
    return None


def _read_significant(path: Path, pvalue_cutoff: float) -> pl.DataFrame:
    """Return columns ``CHR`` (str), ``BP`` (int), ``P`` (f64), ``SNP`` (str)
    for SNPs at ``P < pvalue_cutoff``.

    Auto-detects PolyFun parquet vs DIAMANTE-style raw TSV.
    """
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        lf = pl.scan_parquet(path)
        cols = lf.collect_schema().names()
        chr_col = _pick(cols, _CHR_ALIASES) or "CHR"
        bp_col = _pick(cols, _BP_ALIASES) or "BP"
        p_col = _pick(cols, _P_ALIASES) or "P"
        snp_col = "SNP" if "SNP" in cols else None
        a1_col = _pick(cols, _A1_ALIASES)
        a2_col = _pick(cols, _A2_ALIASES)
    else:
        lf = pl.scan_csv(
            path,
            separator="\t",
            has_header=True,
            null_values=["NA", "", "."],
            infer_schema_length=10_000,
        )
        cols = lf.collect_schema().names()
        chr_col = _pick(cols, _CHR_ALIASES)
        bp_col = _pick(cols, _BP_ALIASES)
        p_col = _pick(cols, _P_ALIASES)
        snp_col = "SNP" if "SNP" in cols else None
        a1_col = _pick(cols, _A1_ALIASES)
        a2_col = _pick(cols, _A2_ALIASES)

    if chr_col is None or bp_col is None or p_col is None:
        raise typer.BadParameter(
            f"{path}: could not find chromosome/position/p-value columns. "
            f"Have: {cols}"
        )

    lf = lf.with_columns(
        pl.col(chr_col)
        .cast(pl.Utf8)
        .str.replace(r"^chr", "", literal=False)
        .alias("CHR"),
        pl.col(bp_col).cast(pl.Int64).alias("BP"),
        pl.col(p_col).cast(pl.Float64).alias("P"),
    ).filter(pl.col("P") < pvalue_cutoff)

    # Build a SNP key. Order of preference:
    #   1) explicit SNP column when present.
    #   2) synthesize chr{CHR}:{BP}:{A2}:{A1} -- same key as harmonize_gwas().
    #   3) fall back to chr{CHR}:{BP} when no alleles are available.
    if snp_col is not None:
        lf = lf.with_columns(pl.col(snp_col).cast(pl.Utf8).alias("SNP"))
    elif a1_col is not None and a2_col is not None:
        lf = lf.with_columns(
            (
                pl.lit("chr")
                + pl.col("CHR")
                + pl.lit(":")
                + pl.col("BP").cast(pl.Utf8)
                + pl.lit(":")
                + pl.col(a2_col).cast(pl.Utf8).str.to_uppercase()
                + pl.lit(":")
                + pl.col(a1_col).cast(pl.Utf8).str.to_uppercase()
            ).alias("SNP")
        )
    else:
        lf = lf.with_columns(
            (
                pl.lit("chr")
                + pl.col("CHR")
                + pl.lit(":")
                + pl.col("BP").cast(pl.Utf8)
            ).alias("SNP")
        )

    return lf.select(["CHR", "BP", "P", "SNP"]).collect()


# ---------------------------------------------------------------------------
# Peak detection (one GWAS)
# ---------------------------------------------------------------------------


def _peaks_from_sig(df_sig: pl.DataFrame, *, window_bp: int) -> list[dict]:
    """Greedy merge of sig SNPs into peaks: top P -> +/- window, repeat.

    Mirrors :func:`polyfun_gpn.loci.select.select_lead_loci` but stays in
    plain dicts so we can keep multiple GWAS in lockstep without a Locus
    type.
    """
    if df_sig.is_empty():
        return []
    df = df_sig.sort("P")
    used: dict[str, list[tuple[int, int]]] = {}
    peaks: list[dict] = []
    for row in df.iter_rows(named=True):
        chrom = str(row["CHR"])
        bp = int(row["BP"])
        intervals = used.setdefault(chrom, [])
        if any(s <= bp <= e for s, e in intervals):
            continue
        start = max(1, bp - window_bp)
        end = bp + window_bp
        intervals.append((start, end))
        peaks.append(
            {
                "chrom": chrom,
                "lead_bp": bp,
                "lead_snp": row.get("SNP") or "",
                "lead_p": float(row["P"]),
                "start": start,
                "end": end,
            }
        )
    return peaks


# ---------------------------------------------------------------------------
# Cross-GWAS overlap
# ---------------------------------------------------------------------------


def _overlapping(peak: dict, others: list[dict]) -> list[dict]:
    return [
        p
        for p in others
        if p["chrom"] == peak["chrom"]
        and not (p["end"] < peak["start"] or p["start"] > peak["end"])
    ]


def _consensus_lead(
    sig_dfs: list[pl.DataFrame],
    chrom: str,
    start: int,
    end: int,
) -> tuple[str, int, float]:
    """Pick the lead = SNP with the largest summed -log10(P) across GWAS.

    Restricted to SNPs with the same ``(CHR, BP)`` that show up significant
    in *every* GWAS (`n_gwas` tie-breaker first). Falls back to the most
    significant SNP overall if no single (CHR, BP) is shared.
    """
    chrom = str(chrom)
    parts = []
    for i, df in enumerate(sig_dfs):
        sub = df.filter(
            (pl.col("CHR") == chrom)
            & (pl.col("BP") >= start)
            & (pl.col("BP") <= end)
        ).with_columns(pl.lit(i, dtype=pl.Int64).alias("gwas_idx"))
        if sub.height:
            parts.append(sub)
    if not parts:
        return ("", 0, 1.0)
    cat = pl.concat(parts)
    agg = (
        cat.group_by(["CHR", "BP"])
        .agg(
            pl.col("P").min().alias("min_p"),
            pl.col("P").log10().neg().sum().alias("score"),
            pl.col("SNP").first().alias("SNP"),
            pl.col("gwas_idx").n_unique().alias("n_gwas"),
        )
        .sort(
            ["n_gwas", "score", "min_p"],
            descending=[True, True, False],
        )
    )
    top = agg.row(0, named=True)
    return (str(top["SNP"]), int(top["BP"]), float(top["min_p"]))


def find_shared_loci(
    sig_dfs: list[pl.DataFrame],
    *,
    window_bp: int,
) -> list[dict]:
    """Anchor on the first GWAS; keep peaks with an overlapping peak in
    every other GWAS. Output window is +/- ``window_bp`` around the
    consensus lead (so it matches the per-GWAS peak size and the
    fine-mapping ``finemap.locus_window_bp``).
    """
    if len(sig_dfs) < 2:
        raise ValueError("Need at least two GWAS inputs to find shared peaks.")

    peaks_per_gwas = [_peaks_from_sig(d, window_bp=window_bp) for d in sig_dfs]
    anchor = peaks_per_gwas[0]
    others = peaks_per_gwas[1:]

    shared: list[dict] = []
    for ap in anchor:
        matched_windows = [ap]
        ok = True
        for opeaks in others:
            ovs = _overlapping(ap, opeaks)
            if not ovs:
                ok = False
                break
            matched_windows.extend(ovs)
        if not ok:
            continue
        # Intersection of all matched windows = region where every GWAS
        # has signal. Used to *select* the consensus lead; the output
        # window is then re-centred around that lead.
        inter_start = max(w["start"] for w in matched_windows)
        inter_end = min(w["end"] for w in matched_windows)
        snp, bp, min_p = _consensus_lead(
            sig_dfs, ap["chrom"], inter_start, inter_end
        )
        if bp == 0:
            snp, bp, min_p = ap["lead_snp"], ap["lead_bp"], ap["lead_p"]
        start = max(1, bp - window_bp)
        end = bp + window_bp
        shared.append(
            {
                "chrom": ap["chrom"],
                "start": start,
                "end": end,
                "lead_bp": bp,
                "lead_snp": snp,
                "lead_p": min_p,
            }
        )
    return shared


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _chrom_sort_key(chrom: str) -> int:
    chrom = chrom.upper()
    if chrom.isdigit():
        return int(chrom)
    return {"X": 23, "Y": 24, "MT": 25, "M": 25}.get(chrom, 100)


def _dedupe(shared: list[dict]) -> list[dict]:
    """Drop duplicate output loci.

    Two distinct anchor peaks can produce the same consensus lead because
    ``_peaks_from_sig`` only forbids new leads inside earlier *windows*,
    not overlapping windows themselves. After we re-centre on the consensus
    lead the duplicates collapse to one (chrom, lead_bp); keep the strongest.
    """
    best: dict[tuple[str, int], dict] = {}
    for s in shared:
        key = (str(s["chrom"]), int(s["lead_bp"]))
        prev = best.get(key)
        if prev is None or float(s.get("lead_p", 1.0)) < float(prev.get("lead_p", 1.0)):
            best[key] = s
    return list(best.values())


def write_loci_tsv(shared: list[dict], output: Path) -> None:
    """Emit a TSV matching ``configs/loci_demo.tsv``.

    Columns: ``locus_id  chrom  start  end  gene  lead_rsid  lead_pos_hg19``.
    ``gene`` is left blank: GWAS sumstats alone don't carry gene annotation.
    """
    rows = []
    for s in sorted(
        _dedupe(shared),
        key=lambda r: (_chrom_sort_key(str(r["chrom"])), int(r["start"])),
    ):
        chrom = str(s["chrom"])
        rows.append(
            {
                "locus_id": f"chr{chrom}_{int(s['start'])}_{int(s['end'])}",
                "chrom": chrom,
                "start": int(s["start"]),
                "end": int(s["end"]),
                "gene": "",
                "lead_rsid": str(s.get("lead_snp") or ""),
                "lead_pos_hg19": int(s["lead_bp"]),
            }
        )
    schema = {
        "locus_id": pl.Utf8,
        "chrom": pl.Utf8,
        "start": pl.Int64,
        "end": pl.Int64,
        "gene": pl.Utf8,
        "lead_rsid": pl.Utf8,
        "lead_pos_hg19": pl.Int64,
    }
    df = pl.DataFrame(rows, schema=schema)
    output.parent.mkdir(parents=True, exist_ok=True)
    # quote_style="never" so the empty ``gene`` cells are real empty fields,
    # not the literal string ``""`` (loci_demo.tsv uses plain empty cells).
    df.write_csv(output, separator="\t", include_header=True, quote_style="never")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    gwas: List[Path] = typer.Argument(
        ...,
        help="Two or more GWAS summary-stat files (parquet or raw TSV).",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output loci TSV path (loci_demo.tsv schema).",
    ),
    pvalue: float = typer.Option(
        GENOME_WIDE_SIGNIFICANCE,
        "--pvalue",
        help="Genome-wide significance threshold for peak detection.",
    ),
    window_bp: int = typer.Option(
        DEFAULT_WINDOW_BP,
        "--window-bp",
        help="Half-width (bp) of the +/- window around each lead SNP.",
    ),
) -> None:
    """Find peaks present in every input GWAS and emit a loci TSV."""
    if len(gwas) < 2:
        raise typer.BadParameter("Need at least two GWAS inputs.")

    typer.echo(
        f"[shared-loci] inputs={len(gwas)}  P<{pvalue:.0e}  "
        f"window=+/-{window_bp:,}bp"
    )
    sig_dfs: list[pl.DataFrame] = []
    for g in gwas:
        df = _read_significant(g, pvalue)
        typer.echo(f"[shared-loci]   {g.name}: {df.height:,} significant SNPs")
        sig_dfs.append(df)

    shared = find_shared_loci(sig_dfs, window_bp=window_bp)
    typer.echo(
        f"[shared-loci] shared peaks (present in all {len(gwas)} GWAS): "
        f"{len(shared)}"
    )

    write_loci_tsv(shared, output)
    typer.echo(f"[shared-loci] wrote -> {output}")


if __name__ == "__main__":
    app()

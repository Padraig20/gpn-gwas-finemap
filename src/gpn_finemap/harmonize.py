"""Coordinate harmonization between FinnGen variants and entropy scores."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from gpn_finemap.entropy import scan_entropy_chrom

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HarmonizationDiagnostics:
    """Join diagnostics for entropy annotation coverage."""

    rows: int
    matched_rows: int
    unmatched_rows: int
    match_rate: float
    ref_mismatch_candidates: int | None = None


def join_entropy_scores(
    variants: pl.LazyFrame,
    entropy_dir: Path,
    chroms: list[str] | None = None,
) -> pl.DataFrame:
    """Annotate variant rows with GPN-Star entropy by ``chrom``, ``pos``, ``ref``.

    The entropy parquet files are chromosome-specific and large, so this joins
    one chromosome at a time instead of scanning the full 17 GB dataset at once.
    """

    if chroms is None:
        logger.info("Collecting chromosomes present in variant fine-mapping data")
        chroms = (
            variants.select(pl.col("chrom").cast(pl.Utf8).unique().sort())
            .collect()
            .get_column("chrom")
            .to_list()
        )
    logger.info("Joining entropy scores for chromosomes: %s", ", ".join(map(str, chroms)))

    annotated: list[pl.DataFrame] = []
    for chrom in chroms:
        logger.info("Joining chromosome %s to entropy scores", chrom)
        variants_chr = variants.filter(pl.col("chrom") == chrom)
        entropy_chr = scan_entropy_chrom(entropy_dir, chrom)
        joined = variants_chr.join(entropy_chr, on=["chrom", "pos", "ref"], how="left").collect()
        logger.info(
            "Chromosome %s joined rows=%d entropy_matches=%d",
            chrom,
            joined.height,
            joined.filter(pl.col("entropy_calibrated").is_not_null()).height,
        )
        if joined.height:
            annotated.append(joined)

    if not annotated:
        logger.warning("No rows were annotated with entropy scores")
        return variants.collect().with_columns(pl.lit(None, dtype=pl.Float64).alias("entropy_calibrated"))
    result = pl.concat(annotated, how="vertical_relaxed")
    logger.info("Finished entropy join with %d total rows", result.height)
    return result


def harmonization_diagnostics(frame: pl.DataFrame) -> HarmonizationDiagnostics:
    """Summarize entropy annotation coverage after harmonization."""

    rows = frame.height
    matched = frame.filter(pl.col("entropy_calibrated").is_not_null()).height
    unmatched = rows - matched
    logger.info("Entropy harmonization match rate: %d/%d rows (%.2f%%)", matched, rows, matched / rows * 100 if rows else 0)
    return HarmonizationDiagnostics(
        rows=rows,
        matched_rows=matched,
        unmatched_rows=unmatched,
        match_rate=matched / rows if rows else 0.0,
        ref_mismatch_candidates=_count_ref_mismatch_candidates(frame),
    )


def _count_ref_mismatch_candidates(frame: pl.DataFrame) -> int | None:
    """Count positions with entropy missing but another row at the same site matched."""

    required = {"chrom", "pos", "entropy_calibrated"}
    if not required.issubset(set(frame.columns)):
        return None

    matched_sites = (
        frame.filter(pl.col("entropy_calibrated").is_not_null())
        .select("chrom", "pos")
        .unique()
        .with_columns(pl.lit(True).alias("site_has_entropy_match"))
    )
    if matched_sites.is_empty():
        return 0

    missing = frame.filter(pl.col("entropy_calibrated").is_null()).join(
        matched_sites, on=["chrom", "pos"], how="left"
    )
    return missing.filter(pl.col("site_has_entropy_match") == True).height


def add_entropy_rank_score(frame: pl.DataFrame, constrained_direction: str = "low") -> pl.DataFrame:
    """Add a score where larger values mean stronger predicted constraint."""

    if constrained_direction not in {"low", "high"}:
        raise ValueError("constrained_direction must be 'low' or 'high'")

    expression = pl.col("entropy_calibrated")
    if constrained_direction == "low":
        expression = -expression
    return frame.with_columns(expression.alias("entropy_rank_score"))

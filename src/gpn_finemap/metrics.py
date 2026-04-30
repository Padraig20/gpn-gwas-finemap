"""Benchmark metrics for entropy-only variant prioritization."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable

import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

DEFAULT_PIP_THRESHOLDS = (0.1, 0.5, 0.9)
DEFAULT_TOP_K = (1, 5, 10, 50)
DEFAULT_TOP_FRACTIONS = (0.01, 0.05, 0.1)
logger = logging.getLogger(__name__)


def compute_benchmark_tables(
    frame: pl.DataFrame,
    pip_thresholds: Iterable[float] = DEFAULT_PIP_THRESHOLDS,
    top_k: Iterable[int] = DEFAULT_TOP_K,
    top_fractions: Iterable[float] = DEFAULT_TOP_FRACTIONS,
) -> dict[str, pl.DataFrame]:
    """Compute per-region entropy benchmark summaries."""

    logger.info("Computing benchmark metrics for %d annotated rows", frame.height)
    prepared = _prepare_frame(frame)
    summary_rows: list[dict[str, object]] = []
    top_rows: list[dict[str, object]] = []

    for key, group in prepared.partition_by(["method", "region"], as_dict=True, maintain_order=True).items():
        method, region = key if isinstance(key, tuple) else (None, None)
        logger.info("Computing metrics for %s %s (%d rows)", method, region, group.height)
        clean = group.filter(pl.col("entropy_rank_score").is_not_null() & pl.col("pip").is_not_null())
        if clean.is_empty():
            logger.info("Skipping %s %s because no rows have both entropy and PIP", method, region)
            continue

        entropy = clean.get_column("entropy_rank_score").to_numpy()
        pip = clean.get_column("pip").to_numpy()
        gwas_score = clean.get_column("gwas_rank_score").to_numpy()

        base_row = {
            "method": method,
            "region": region,
            "n_variants": group.height,
            "n_with_entropy": clean.height,
            "max_pip": float(np.nanmax(pip)),
            "mean_pip": float(np.nanmean(pip)),
            "entropy_spearman_pip": _spearman(entropy, pip),
            "gwas_spearman_pip": _spearman(gwas_score, pip),
        }
        for threshold in pip_thresholds:
            labels = pip >= threshold
            threshold_row = base_row | {
                "pip_threshold": threshold,
                "n_positive": int(labels.sum()),
                "auroc_entropy": _safe_auroc(labels, entropy),
                "auprc_entropy": _safe_auprc(labels, entropy),
                "auroc_gwas": _safe_auroc(labels, gwas_score),
                "auprc_gwas": _safe_auprc(labels, gwas_score),
            }
            summary_rows.append(threshold_row)

            for row in _top_rank_rows(clean, threshold, top_k, top_fractions):
                top_rows.append({"method": method, "region": region, "pip_threshold": threshold} | row)

    logger.info("Computed %d region metric rows and %d top-rank metric rows", len(summary_rows), len(top_rows))
    return {
        "region_metrics": pl.DataFrame(summary_rows) if summary_rows else pl.DataFrame(),
        "top_rank_metrics": pl.DataFrame(top_rows) if top_rows else pl.DataFrame(),
    }


def summarize_global(region_metrics: pl.DataFrame) -> pl.DataFrame:
    """Average per-region metrics by method and PIP threshold."""

    if region_metrics.is_empty():
        logger.warning("No region metrics available for global summary")
        return region_metrics

    metric_columns = [
        "entropy_spearman_pip",
        "gwas_spearman_pip",
        "auroc_entropy",
        "auprc_entropy",
        "auroc_gwas",
        "auprc_gwas",
    ]
    logger.info("Summarizing global metrics from %d region rows", region_metrics.height)
    return (
        region_metrics.group_by(["method", "pip_threshold"])
        .agg(
            pl.len().alias("n_regions"),
            pl.col("n_variants").sum().alias("n_variants"),
            pl.col("n_positive").sum().alias("n_positive"),
            *(pl.col(column).mean().alias(f"mean_{column}") for column in metric_columns),
            *(pl.col(column).median().alias(f"median_{column}") for column in metric_columns),
        )
        .sort(["method", "pip_threshold"])
    )


def _prepare_frame(frame: pl.DataFrame) -> pl.DataFrame:
    p_column = "p" if "p" in frame.columns else "pval" if "pval" in frame.columns else None
    if p_column is None:
        return frame.with_columns(pl.lit(None, dtype=pl.Float64).alias("gwas_rank_score"))

    return frame.with_columns(
        pl.when(pl.col(p_column).is_not_null() & (pl.col(p_column) > 0))
        .then(-pl.col(p_column).log10())
        .otherwise(None)
        .alias("gwas_rank_score")
    )


def _top_rank_rows(
    group: pl.DataFrame,
    pip_threshold: float,
    top_k: Iterable[int],
    top_fractions: Iterable[float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    positives_total = group.filter(pl.col("pip") >= pip_threshold).height

    for score_name, score_column in [
        ("entropy", "entropy_rank_score"),
        ("gwas", "gwas_rank_score"),
    ]:
        ranked = group.filter(pl.col(score_column).is_not_null()).sort(score_column, descending=True)
        if ranked.is_empty():
            continue
        for k in top_k:
            rows.append(_top_summary(ranked.head(k), score_name, f"top_{k}", pip_threshold, positives_total))
        for fraction in top_fractions:
            k = max(1, math.ceil(ranked.height * fraction))
            rows.append(
                _top_summary(ranked.head(k), score_name, f"top_{fraction:.0%}", pip_threshold, positives_total)
            )
    return rows


def _top_summary(
    selected: pl.DataFrame,
    score_name: str,
    top_rule: str,
    pip_threshold: float,
    positives_total: int,
) -> dict[str, object]:
    positives_selected = selected.filter(pl.col("pip") >= pip_threshold).height
    return {
        "score": score_name,
        "top_rule": top_rule,
        "n_selected": selected.height,
        "mean_pip": selected.get_column("pip").mean(),
        "max_pip": selected.get_column("pip").max(),
        "positives_selected": positives_selected,
        "recall": positives_selected / positives_total if positives_total else None,
        "precision": positives_selected / selected.height if selected.height else None,
    }


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2 or np.unique(x[mask]).size < 2 or np.unique(y[mask]).size < 2:
        return None
    value = spearmanr(x[mask], y[mask]).statistic
    return float(value) if np.isfinite(value) else None


def _safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    mask = np.isfinite(scores)
    labels = labels[mask]
    scores = scores[mask]
    if labels.size < 2 or np.unique(labels).size < 2:
        return None
    return float(roc_auc_score(labels, scores))


def _safe_auprc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    mask = np.isfinite(scores)
    labels = labels[mask]
    scores = scores[mask]
    if labels.size < 2 or np.unique(labels).size < 2:
        return None
    return float(average_precision_score(labels, scores))

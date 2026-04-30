"""Conservation enrichment among statistically fine-mapped variants."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


def run_conservation_enrichment(
    annotated_variants: Path,
    output_dir: Path,
    *,
    pip_thresholds: list[float],
    conservation_quantiles: list[float],
    constrained_direction: str = "low",
    n_permutations: int = 10_000,
    seed: int = 13,
    chrombpnet_style_plot: bool = True,
) -> dict[str, pl.DataFrame]:
    """Test whether high-PIP SNPs are enriched for high predicted conservation."""

    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pl.read_parquet(annotated_variants)
    tables = compute_conservation_enrichment(
        frame,
        pip_thresholds=pip_thresholds,
        conservation_quantiles=conservation_quantiles,
        constrained_direction=constrained_direction,
        n_permutations=n_permutations,
        seed=seed,
    )
    for name, table in tables.items():
        if not table.is_empty():
            table.write_csv(output_dir / f"{name}.tsv", separator="\t")
            table.write_parquet(output_dir / f"{name}.parquet")
    write_conservation_plot(tables["global_enrichment"], output_dir / "conservation_enrichment.png")
    if chrombpnet_style_plot:
        curve_table = compute_overlap_enrichment_curves(
            frame,
            conservation_quantiles=conservation_quantiles,
            constrained_direction=constrained_direction,
        )
        if not curve_table.is_empty():
            curve_table.write_csv(output_dir / "overlap_enrichment_curves.tsv", separator="\t")
            curve_table.write_parquet(output_dir / "overlap_enrichment_curves.parquet")
            write_overlap_enrichment_curve_plot(curve_table, output_dir / "overlap_enrichment_by_max_pip.png")
    write_conservation_report(tables, output_dir / "conservation_enrichment.md")
    return tables


def compute_conservation_enrichment(
    frame: pl.DataFrame,
    *,
    pip_thresholds: list[float],
    conservation_quantiles: list[float],
    constrained_direction: str = "low",
    n_permutations: int = 10_000,
    seed: int = 13,
) -> dict[str, pl.DataFrame]:
    """Compute per-region and global conservation enrichment tables."""

    if constrained_direction not in {"low", "high"}:
        raise ValueError("constrained_direction must be 'low' or 'high'")
    if "pip" not in frame.columns:
        raise ValueError("annotated variants must contain a 'pip' column")
    if "entropy_calibrated" not in frame.columns:
        raise ValueError("annotated variants must contain 'entropy_calibrated'")

    score_expr = pl.col("entropy_calibrated")
    if constrained_direction == "low":
        score_expr = -score_expr
    prepared = frame.with_columns(score_expr.alias("conservation_score"))
    group_columns = ["method", "region"] if "method" in prepared.columns else ["region"]

    rng = np.random.default_rng(seed)
    region_rows: list[dict[str, object]] = []
    permutation_rows: list[dict[str, object]] = []

    for key, group in prepared.partition_by(group_columns, as_dict=True, maintain_order=True).items():
        method, region = key if isinstance(key, tuple) else ("ALL", key)
        clean = group.filter(pl.col("pip").is_not_null() & pl.col("conservation_score").is_not_null())
        if clean.height < 2:
            continue

        pip = clean.get_column("pip").to_numpy().astype(float)
        conservation = clean.get_column("conservation_score").to_numpy().astype(float)
        for pip_threshold in pip_thresholds:
            high_pip = pip >= pip_threshold
            n_high_pip = int(high_pip.sum())
            if n_high_pip == 0:
                continue

            for quantile in conservation_quantiles:
                threshold = float(np.quantile(conservation, quantile))
                high_conservation = conservation >= threshold
                n_high_conservation = int(high_conservation.sum())
                if n_high_conservation == 0:
                    continue

                observed = int(np.logical_and(high_pip, high_conservation).sum())
                expected = n_high_pip * (n_high_conservation / clean.height)
                fold = observed / expected if expected > 0 else None
                null = _permutation_overlap_counts(
                    high_conservation=high_conservation,
                    n_selected=n_high_pip,
                    n_permutations=n_permutations,
                    rng=rng,
                )
                p_enrichment = (int((null >= observed).sum()) + 1) / (n_permutations + 1)
                p_depletion = (int((null <= observed).sum()) + 1) / (n_permutations + 1)
                permutation_rows.extend(
                    {
                        "method": method,
                        "region": region,
                        "pip_threshold": pip_threshold,
                        "conservation_quantile": quantile,
                        "permutation": idx,
                        "null_overlap": int(value),
                    }
                    for idx, value in enumerate(null)
                )
                region_rows.append(
                    {
                        "method": method,
                        "region": region,
                        "pip_threshold": pip_threshold,
                        "conservation_quantile": quantile,
                        "n_variants": clean.height,
                        "n_high_pip": n_high_pip,
                        "n_high_conservation": n_high_conservation,
                        "conservation_threshold": threshold,
                        "observed_overlap": observed,
                        "expected_overlap": expected,
                        "fold_enrichment": fold,
                        "empirical_p_enrichment": p_enrichment,
                        "empirical_p_depletion": p_depletion,
                        "mean_conservation_high_pip": float(conservation[high_pip].mean()),
                        "mean_conservation_background": float(conservation.mean()),
                    }
                )

    region_table = pl.DataFrame(region_rows) if region_rows else pl.DataFrame()
    permutation_table = pl.DataFrame(permutation_rows) if permutation_rows else pl.DataFrame()
    return {
        "region_enrichment": region_table,
        "global_enrichment": summarize_conservation_enrichment(region_table, permutation_table),
        "null_overlaps": permutation_table,
    }


def summarize_conservation_enrichment(
    region_table: pl.DataFrame,
    permutation_table: pl.DataFrame,
) -> pl.DataFrame:
    """Aggregate overlap enrichments across regions."""

    if region_table.is_empty():
        return pl.DataFrame()

    observed = (
        region_table.group_by(["method", "pip_threshold", "conservation_quantile"])
        .agg(
            pl.len().alias("n_regions"),
            pl.col("n_variants").sum().alias("n_variants"),
            pl.col("n_high_pip").sum().alias("n_high_pip"),
            pl.col("n_high_conservation").sum().alias("n_high_conservation"),
            pl.col("observed_overlap").sum().alias("observed_overlap"),
            pl.col("expected_overlap").sum().alias("expected_overlap"),
            pl.col("fold_enrichment").mean().alias("mean_region_fold_enrichment"),
            pl.col("fold_enrichment").median().alias("median_region_fold_enrichment"),
            pl.col("mean_conservation_high_pip").mean().alias("mean_conservation_high_pip"),
            pl.col("mean_conservation_background").mean().alias("mean_conservation_background"),
        )
        .with_columns((pl.col("observed_overlap") / pl.col("expected_overlap")).alias("global_fold_enrichment"))
    )
    if permutation_table.is_empty():
        return observed

    null = (
        permutation_table.group_by(["method", "pip_threshold", "conservation_quantile", "permutation"])
        .agg(pl.col("null_overlap").sum().alias("global_null_overlap"))
    )
    null_summary = (
        observed.join(null, on=["method", "pip_threshold", "conservation_quantile"], how="left")
        .with_columns(
            (pl.col("global_null_overlap") >= pl.col("observed_overlap")).alias("null_ge_observed"),
            (pl.col("global_null_overlap") <= pl.col("observed_overlap")).alias("null_le_observed"),
        )
        .group_by(["method", "pip_threshold", "conservation_quantile"])
        .agg(
            ((pl.col("null_ge_observed").sum() + 1) / (pl.len() + 1)).alias("global_empirical_p_enrichment"),
            ((pl.col("null_le_observed").sum() + 1) / (pl.len() + 1)).alias("global_empirical_p_depletion"),
        )
    )
    return observed.join(null_summary, on=["method", "pip_threshold", "conservation_quantile"], how="left")


def compute_overlap_enrichment_curves(
    frame: pl.DataFrame,
    *,
    conservation_quantiles: list[float],
    constrained_direction: str = "low",
    pip_grid: np.ndarray | None = None,
) -> pl.DataFrame:
    """Compute ChromBPNet-style overlap enrichment curves.

    For each max-PIP cutoff x, the statistical set is SNPs with PIP >= x.
    Conservation sets are global top-quantile SNPs, so their labels can report
    total genome/locus-wide counts similarly to ChromBPNet overlap plots.
    """

    if pip_grid is None:
        pip_grid = np.concatenate(
            [
                np.linspace(0.001, 0.05, 50),
                np.linspace(0.055, 0.5, 90),
                np.linspace(0.51, 1.0, 50),
            ]
        )
    if constrained_direction not in {"low", "high"}:
        raise ValueError("constrained_direction must be 'low' or 'high'")

    score_expr = pl.col("entropy_calibrated")
    if constrained_direction == "low":
        score_expr = -score_expr
    prepared = frame.with_columns(score_expr.alias("conservation_score"))
    if "method" not in prepared.columns:
        prepared = prepared.with_columns(pl.lit("ALL").alias("method"))

    rows: list[dict[str, object]] = []
    for method, method_frame in prepared.partition_by("method", as_dict=True, maintain_order=True).items():
        method_name = method[0] if isinstance(method, tuple) else str(method)
        clean = method_frame.filter(pl.col("pip").is_not_null() & pl.col("conservation_score").is_not_null())
        if clean.height < 2:
            continue
        pip = clean.get_column("pip").to_numpy().astype(float)
        conservation = clean.get_column("conservation_score").to_numpy().astype(float)
        n_total = len(pip)
        for quantile in conservation_quantiles:
            threshold = float(np.quantile(conservation, quantile))
            high_conservation = conservation >= threshold
            n_high_conservation = int(high_conservation.sum())
            if n_high_conservation == 0:
                continue
            background_rate = n_high_conservation / n_total
            for pip_cutoff in pip_grid:
                high_pip = pip >= pip_cutoff
                n_high_pip = int(high_pip.sum())
                if n_high_pip == 0:
                    continue
                observed = int(np.logical_and(high_pip, high_conservation).sum())
                expected = n_high_pip * background_rate
                rows.append(
                    {
                        "method": method_name,
                        "max_pip": float(pip_cutoff),
                        "conservation_quantile": quantile,
                        "conservation_top_fraction": 1 - quantile,
                        "n_total": n_total,
                        "n_high_pip": n_high_pip,
                        "n_high_conservation": n_high_conservation,
                        "observed_overlap": observed,
                        "expected_overlap": expected,
                        "overlap_enrichment": observed / expected if expected > 0 else None,
                    }
                )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def write_conservation_plot(global_table: pl.DataFrame, path: Path) -> None:
    """Plot global fold enrichment by threshold."""

    if global_table.is_empty():
        return

    import matplotlib.pyplot as plt

    plot_frame = global_table.sort(["method", "pip_threshold", "conservation_quantile"])
    labels = [
        f"{row['method']} PIP>={row['pip_threshold']} top {1 - row['conservation_quantile']:.0%}"
        for row in plot_frame.iter_rows(named=True)
    ]
    values = plot_frame.get_column("global_fold_enrichment").to_list()
    x_positions = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.75), 4))
    ax.axhline(1.0, color="black", linewidth=1)
    ax.bar(x_positions, values)
    ax.set_ylabel("Observed / expected overlap")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_overlap_enrichment_curve_plot(curve_table: pl.DataFrame, path: Path) -> None:
    """Write a ChromBPNet-style overlap enrichment curve plot."""

    if curve_table.is_empty():
        return

    import matplotlib.pyplot as plt

    methods = curve_table.get_column("method").unique().to_list()
    n_panels = len(methods)
    fig, axes = plt.subplots(1, n_panels, figsize=(max(7, 5 * n_panels), 5), squeeze=False)
    colors = plt.cm.magma(np.linspace(0.15, 0.85, curve_table.get_column("conservation_quantile").n_unique()))

    for ax, method in zip(axes[0], methods, strict=False):
        method_frame = curve_table.filter(pl.col("method") == method)
        quantiles = sorted(method_frame.get_column("conservation_quantile").unique().to_list(), reverse=True)
        for color, quantile in zip(colors, quantiles, strict=False):
            subset = method_frame.filter(pl.col("conservation_quantile") == quantile).sort("max_pip")
            count = int(subset.get_column("n_high_conservation")[0])
            label = f"{1 - quantile:g} ({_format_count(count)})"
            ax.plot(
                subset.get_column("max_pip").to_numpy(),
                subset.get_column("overlap_enrichment").to_numpy(),
                color=color,
                linewidth=2,
                label=label,
            )
        ax.axhline(1.0, color="lightgray", linewidth=1)
        ax.set_yscale("log")
        ax.set_xlabel("max PIP")
        ax.set_ylabel("Overlap enrichment")
        ax.set_title(str(method))
        ax.set_xlim(0, 1)
        ax.legend(title="Top conservation", frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_conservation_report(tables: dict[str, pl.DataFrame], path: Path) -> None:
    """Write a compact markdown report."""

    global_table = tables["global_enrichment"]
    lines = [
        "# Conservation Enrichment",
        "",
        "Question: among SNPs with high statistical probability of being causal, "
        "do we see more large predicted conservation than expected by chance?",
        "",
        "Chance is estimated by region-preserving random draws with the same number "
        "of high-PIP SNPs per region.",
    ]
    if not global_table.is_empty():
        lines.extend(["", "## Global Results", ""])
        lines.extend(_markdown_table(global_table))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _permutation_overlap_counts(
    high_conservation: np.ndarray,
    n_selected: int,
    n_permutations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(high_conservation)
    counts = np.empty(n_permutations, dtype=int)
    for idx in range(n_permutations):
        selected = rng.choice(n, size=n_selected, replace=False)
        counts[idx] = int(high_conservation[selected].sum())
    return counts


def _markdown_table(frame: pl.DataFrame, max_rows: int = 30) -> list[str]:
    display = frame.head(max_rows)
    columns = display.columns
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in display.iter_rows(named=True):
        rows.append("| " + " | ".join(_format_cell(row[column]) for column in columns) + " |")
    if frame.height > max_rows:
        rows.append(f"\nShowing first {max_rows} of {frame.height} rows.")
    return rows


def _format_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if value is None:
        return ""
    return str(value)


def _format_count(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{round(value / 1_000):.0f}K"
    return str(value)

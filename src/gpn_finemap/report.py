"""Output writers for benchmark tables and a concise markdown report."""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from gpn_finemap.harmonize import HarmonizationDiagnostics

logger = logging.getLogger(__name__)


def write_benchmark_outputs(
    output_dir: Path,
    annotated_variants: pl.DataFrame,
    tables: dict[str, pl.DataFrame],
    global_summary: pl.DataFrame,
    diagnostics: HarmonizationDiagnostics,
    endpoint: str,
    release: int,
    constrained_direction: str,
) -> None:
    """Write parquet/TSV outputs, plots, and a markdown summary."""

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Writing annotated variants to %s", output_dir / "annotated_finemap_variants.parquet")
    annotated_variants.write_parquet(output_dir / "annotated_finemap_variants.parquet")

    for name, table in tables.items():
        if table.is_empty():
            logger.info("Skipping empty output table: %s", name)
            continue
        logger.info("Writing %s table with %d rows", name, table.height)
        table.write_csv(output_dir / f"{name}.tsv", separator="\t")
        table.write_parquet(output_dir / f"{name}.parquet")

    if not global_summary.is_empty():
        logger.info("Writing global summary with %d rows", global_summary.height)
        global_summary.write_csv(output_dir / "global_summary.tsv", separator="\t")
        global_summary.write_parquet(output_dir / "global_summary.parquet")
        write_metric_plot(global_summary, output_dir / "global_auprc.png")

    logger.info("Writing markdown report to %s", output_dir / "report.md")
    (output_dir / "report.md").write_text(
        render_report(
            endpoint=endpoint,
            release=release,
            constrained_direction=constrained_direction,
            diagnostics=diagnostics,
            global_summary=global_summary,
        )
    )


def render_report(
    endpoint: str,
    release: int,
    constrained_direction: str,
    diagnostics: HarmonizationDiagnostics,
    global_summary: pl.DataFrame,
) -> str:
    """Render a compact markdown report."""

    lines = [
        "# GPN-Star Entropy Fine-Mapping Benchmark",
        "",
        f"- FinnGen release: R{release}",
        f"- Endpoint: `{endpoint}`",
        f"- Entropy direction treated as constrained: `{constrained_direction}`",
        f"- Annotated fine-mapping rows: {diagnostics.rows:,}",
        f"- Rows with entropy match: {diagnostics.matched_rows:,} ({diagnostics.match_rate:.2%})",
        f"- Rows without entropy match: {diagnostics.unmatched_rows:,}",
    ]
    if diagnostics.ref_mismatch_candidates is not None:
        lines.append(f"- Possible reference-allele mismatch rows: {diagnostics.ref_mismatch_candidates:,}")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This benchmark treats FinnGen SuSiE/FINEMAP posterior inclusion probabilities "
            "and credible-set assignments as reference labels. They are useful SOTA "
            "comparators, but they are not ground-truth causal labels.",
            "",
            "Entropy-only ranking is a prior-like signal. It does not model trait association "
            "or LD, so a weak standalone result does not rule out value as a functional prior "
            "inside SuSiE, FINEMAP, or PolyFun-style fine-mapping.",
        ]
    )

    if not global_summary.is_empty():
        lines.extend(["", "## Global Summary", ""])
        lines.extend(_markdown_table(global_summary))

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `annotated_finemap_variants.parquet`: SNP-level FinnGen rows with entropy scores.",
            "- `region_metrics.tsv`: per-region correlation, AUROC, and AUPRC metrics.",
            "- `top_rank_metrics.tsv`: top-k and top-percentile precision/recall metrics.",
            "- `global_summary.tsv`: averaged per-region metrics by method and PIP threshold.",
            "- `global_auprc.png`: quick visual comparison of entropy and GWAS ranking.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_metric_plot(global_summary: pl.DataFrame, path: Path) -> None:
    """Write a small AUPRC comparison plot."""

    import matplotlib.pyplot as plt

    logger.info("Writing metric plot to %s", path)
    plot_frame = global_summary.select(
        "method",
        "pip_threshold",
        "mean_auprc_entropy",
        "mean_auprc_gwas",
    ).sort(["method", "pip_threshold"])

    labels = [
        f"{row['method']} PIP>={row['pip_threshold']}"
        for row in plot_frame.iter_rows(named=True)
    ]
    entropy_values = plot_frame.get_column("mean_auprc_entropy").to_list()
    gwas_values = plot_frame.get_column("mean_auprc_gwas").to_list()
    x_positions = range(len(labels))

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.8), 4))
    width = 0.4
    ax.bar([x - width / 2 for x in x_positions], entropy_values, width=width, label="Entropy")
    ax.bar([x + width / 2 for x in x_positions], gwas_values, width=width, label="GWAS p-value")
    ax.set_ylabel("Mean per-region AUPRC")
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _markdown_table(frame: pl.DataFrame, max_rows: int = 30) -> list[str]:
    display = frame.head(max_rows)
    columns = display.columns
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in display.iter_rows(named=True):
        values = [_format_cell(row[column]) for column in columns]
        rows.append("| " + " | ".join(values) + " |")
    if frame.height > max_rows:
        rows.append(f"\nShowing first {max_rows} of {frame.height} rows.")
    return rows


def _format_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if value is None:
        return ""
    return str(value)

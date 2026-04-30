"""Command line interface for the GPN-Star entropy fine-mapping benchmark."""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl
import typer

from gpn_finemap.entropy import inspect_entropy_files, validate_entropy_files
from gpn_finemap.finngen import (
    DEFAULT_RELEASE,
    download_endpoint_files,
    scan_finemap_snps,
    scan_summary_stats,
)
from gpn_finemap.harmonize import (
    add_entropy_rank_score,
    harmonization_diagnostics,
    join_entropy_scores,
)
from gpn_finemap.log import configure_logging
from gpn_finemap.metrics import compute_benchmark_tables, summarize_global
from gpn_finemap.report import write_benchmark_outputs

app = typer.Typer(help="Benchmark GPN-Star entropy against FinnGen fine-mapping outputs.")
logger = logging.getLogger(__name__)


@app.command("inspect-entropy")
def inspect_entropy(
    entropy_dir: Path = typer.Option(Path("entropy"), help="Directory with entropy_chr*.parquet files."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show progress logging."),
) -> None:
    """Inspect entropy parquet coverage and schema."""

    configure_logging(verbose)
    logger.info("Inspecting entropy inputs")
    infos = inspect_entropy_files(entropy_dir)
    rows = [
        {
            "chrom": info.chrom,
            "path": str(info.path),
            "rows": info.rows,
            "size_mb": round(info.size_bytes / 1_000_000, 2),
            "columns": ",".join(info.columns),
        }
        for info in infos
    ]
    typer.echo(pl.DataFrame(rows))

    problems = validate_entropy_files(entropy_dir)
    if problems:
        typer.echo("\nValidation problems:")
        for problem in problems:
            typer.echo(f"- {problem}")
        raise typer.Exit(1)


@app.command("download-finngen")
def download_finngen(
    cache_dir: Path = typer.Option(Path("data"), help="Cache directory for downloaded FinnGen files."),
    release: int = typer.Option(DEFAULT_RELEASE, help="FinnGen public release number."),
    endpoint: str = typer.Option("T2D", help="FinnGen endpoint, e.g. T2D or E4_DM2."),
    summary_url: str | None = typer.Option(None, help="Explicit summary-statistics URL override."),
    susie_snp_url: str | None = typer.Option(None, help="Explicit SuSiE SNP-level URL override."),
    finemap_snp_url: str | None = typer.Option(None, help="Explicit FINEMAP SNP-level URL override."),
    overwrite: bool = typer.Option(False, help="Redownload files that already exist."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show progress logging."),
) -> None:
    """Download/cache FinnGen files for an endpoint."""

    configure_logging(verbose)
    logger.info("Downloading FinnGen inputs")
    paths = download_endpoint_files(
        cache_dir=cache_dir,
        release=release,
        endpoint=endpoint,
        summary_url=summary_url,
        susie_snp_url=susie_snp_url,
        finemap_snp_url=finemap_snp_url,
        overwrite=overwrite,
    )
    typer.echo(paths)


@app.command("run")
def run_benchmark(
    entropy_dir: Path = typer.Option(Path("entropy"), help="Directory with entropy_chr*.parquet files."),
    cache_dir: Path = typer.Option(Path("data"), help="Cache directory for FinnGen files."),
    output_dir: Path = typer.Option(Path("results"), help="Output directory for benchmark results."),
    release: int = typer.Option(DEFAULT_RELEASE, help="FinnGen public release number."),
    endpoint: str = typer.Option("T2D", help="FinnGen endpoint, e.g. T2D or E4_DM2."),
    summary_path: Path | None = typer.Option(None, help="Local FinnGen summary-statistics file."),
    susie_snp_path: Path | None = typer.Option(None, help="Local FinnGen SuSiE SNP-level file."),
    finemap_snp_path: Path | None = typer.Option(None, help="Local FinnGen FINEMAP SNP-level file."),
    summary_url: str | None = typer.Option(None, help="Explicit summary-statistics URL override."),
    susie_snp_url: str | None = typer.Option(None, help="Explicit SuSiE SNP-level URL override."),
    finemap_snp_url: str | None = typer.Option(None, help="Explicit FINEMAP SNP-level URL override."),
    constrained_direction: str = typer.Option(
        "low",
        help="Whether lower or higher entropy_calibrated means stronger constraint: low|high.",
    ),
    overwrite_downloads: bool = typer.Option(False, help="Redownload FinnGen files that already exist."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show progress logging."),
) -> None:
    """Run the entropy-only benchmark against FinnGen SuSiE/FINEMAP PIPs."""

    configure_logging(verbose)
    logger.info("Starting benchmark run for FinnGen R%s endpoint %s", release, endpoint)
    problems = validate_entropy_files(entropy_dir)
    if problems:
        typer.echo("Entropy validation failed:")
        for problem in problems:
            typer.echo(f"- {problem}")
        raise typer.Exit(1)

    paths = _resolve_finngen_paths(
        cache_dir=cache_dir,
        release=release,
        endpoint=endpoint,
        summary_path=summary_path,
        susie_snp_path=susie_snp_path,
        finemap_snp_path=finemap_snp_path,
        summary_url=summary_url,
        susie_snp_url=susie_snp_url,
        finemap_snp_url=finemap_snp_url,
        overwrite=overwrite_downloads,
    )
    logger.info("Resolved benchmark input paths: %s", paths)
    fine_map = _scan_available_finemapping(paths["susie"], paths["finemap"])
    if fine_map is None:
        raise typer.BadParameter(
            "No fine-mapping SNP file is available. Provide --susie-snp-path/--finemap-snp-path "
            "or URL overrides if the public candidate URLs fail."
        )

    if paths["summary"] is not None:
        logger.info("Joining FinnGen summary-statistics p-values as baseline")
        fine_map = _join_summary_baseline(fine_map, paths["summary"])

    logger.info("Annotating fine-mapping variants with entropy")
    annotated = join_entropy_scores(fine_map, entropy_dir)
    annotated = add_entropy_rank_score(annotated, constrained_direction=constrained_direction)
    diagnostics = harmonization_diagnostics(annotated)

    logger.info("Computing benchmark tables")
    tables = compute_benchmark_tables(annotated)
    global_summary = summarize_global(tables["region_metrics"])
    logger.info("Writing benchmark outputs")
    write_benchmark_outputs(
        output_dir=output_dir,
        annotated_variants=annotated,
        tables=tables,
        global_summary=global_summary,
        diagnostics=diagnostics,
        endpoint=endpoint,
        release=release,
        constrained_direction=constrained_direction,
    )
    logger.info("Benchmark run complete")
    typer.echo(f"Wrote benchmark outputs to {output_dir}")


def _resolve_finngen_paths(
    cache_dir: Path,
    release: int,
    endpoint: str,
    summary_path: Path | None,
    susie_snp_path: Path | None,
    finemap_snp_path: Path | None,
    summary_url: str | None,
    susie_snp_url: str | None,
    finemap_snp_url: str | None,
    overwrite: bool,
) -> dict[str, Path | None]:
    if summary_path or susie_snp_path or finemap_snp_path:
        logger.info("Using local FinnGen input paths")
        return {"summary": summary_path, "susie": susie_snp_path, "finemap": finemap_snp_path}

    downloaded = download_endpoint_files(
        cache_dir=cache_dir,
        release=release,
        endpoint=endpoint,
        summary_url=summary_url,
        susie_snp_url=susie_snp_url,
        finemap_snp_url=finemap_snp_url,
        overwrite=overwrite,
    )
    return {
        "summary": downloaded.summary_stats,
        "susie": downloaded.susie_snp,
        "finemap": downloaded.finemap_snp,
    }


def _scan_available_finemapping(
    susie_path: Path | None,
    finemap_path: Path | None,
) -> pl.LazyFrame | None:
    frames: list[pl.LazyFrame] = []
    if susie_path is not None:
        logger.info("Including SuSiE SNP-level file: %s", susie_path)
        frames.append(scan_finemap_snps(susie_path, "SUSIE"))
    if finemap_path is not None:
        logger.info("Including FINEMAP SNP-level file: %s", finemap_path)
        frames.append(scan_finemap_snps(finemap_path, "FINEMAP"))
    if not frames:
        return None
    return pl.concat(frames, how="diagonal_relaxed")


def _join_summary_baseline(fine_map: pl.LazyFrame, summary_path: Path) -> pl.LazyFrame:
    logger.info("Scanning summary baseline file: %s", summary_path)
    summary = scan_summary_stats(summary_path).select("chrom", "pos", "ref", "alt", "pval")
    return fine_map.join(summary, on=["chrom", "pos", "ref", "alt"], how="left")

"""Typer CLI entry point for polyfun-gpn.

Subcommands:
    setup         clone PolyFun + download FINEMAP binary + chain files
    harmonize     convert raw GWAS TSV to a Polars-friendly hg19 parquet
    build-bg      sample genome-wide entropy and cache background density
    prepare-loci  build per-locus PolyFun sumstats (with SNPVAR)
    run           run FINEMAP for a list of loci
    run-all       fan out genome-wide loci via PolyFun's create_finemapper_jobs
    aggregate     collect per-locus FINEMAP outputs into one results table
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

import typer

from .config import load_config


app = typer.Typer(
    name="polyfun-gpn",
    help="PolyFun + FINEMAP fine-mapping with entropy-derived per-SNP priors.",
    no_args_is_help=True,
    add_completion=False,
)


class PriorMode(str, Enum):
    entropy = "entropy"
    uniform = "uniform"
    none = "none"


def _config_option() -> Path:
    return typer.Option(
        None,
        "--config",
        "-c",
        help="Path to YAML config (defaults to configs/default.yaml).",
    )


@app.command()
def setup(
    config: Optional[Path] = _config_option(),
    skip_polyfun: bool = typer.Option(False, help="Skip cloning PolyFun."),
    skip_finemap: bool = typer.Option(False, help="Skip downloading FINEMAP."),
    skip_chain: bool = typer.Option(False, help="Skip downloading chain files."),
) -> None:
    """Clone PolyFun, download FINEMAP v1.4.1, download liftover chain."""
    from .external.setup import run_setup

    cfg = load_config(config)
    run_setup(
        cfg,
        skip_polyfun=skip_polyfun,
        skip_finemap=skip_finemap,
        skip_chain=skip_chain,
    )


@app.command()
def harmonize(
    config: Optional[Path] = _config_option(),
    overwrite: bool = typer.Option(False, help="Recompute even if output exists."),
) -> None:
    """Stream GWAS TSV and emit a single Parquet with PolyFun columns."""
    from .gwas.io import harmonize_gwas

    cfg = load_config(config)
    harmonize_gwas(cfg, overwrite=overwrite)


@app.command("build-bg")
def build_bg(
    config: Optional[Path] = _config_option(),
    n_samples: Optional[int] = typer.Option(
        None, help="Override config n_samples."
    ),
    overwrite: bool = typer.Option(False, help="Recompute even if cache exists."),
) -> None:
    """Sample genome-wide entropy values and build background density cache."""
    from .entropy.background import build_background

    cfg = load_config(config)
    if n_samples is not None:
        cfg.background.n_samples = n_samples
    build_background(cfg, overwrite=overwrite)


@app.command("prepare-loci")
def prepare_loci(
    config: Optional[Path] = _config_option(),
    loci: Path = typer.Option(
        Path("configs/loci_demo.tsv"),
        help="TSV with columns: locus_id chrom start end ...",
    ),
    prior: PriorMode = typer.Option(PriorMode.entropy, help="Prior mode."),
) -> None:
    """For each locus, write per-locus PolyFun sumstats (with SNPVAR)."""
    from .pipeline.prepare_locus import prepare_loci as _prepare

    cfg = load_config(config)
    _prepare(cfg, loci_path=loci, prior_mode=prior.value)


@app.command()
def run(
    config: Optional[Path] = _config_option(),
    loci: Path = typer.Option(
        Path("configs/loci_demo.tsv"),
        help="TSV of loci to fine-map.",
    ),
    prior: PriorMode = typer.Option(PriorMode.entropy, help="Prior mode."),
    skip_prepare: bool = typer.Option(
        False, help="Assume per-locus sumstats already exist."
    ),
) -> None:
    """End-to-end run on the listed loci: prepare + FINEMAP."""
    from .pipeline.prepare_locus import prepare_loci as _prepare
    from .pipeline.run_finemap import run_loci

    cfg = load_config(config)
    if not skip_prepare:
        _prepare(cfg, loci_path=loci, prior_mode=prior.value)
    run_loci(cfg, loci_path=loci, prior_mode=prior.value)


@app.command("run-all")
def run_all(
    config: Optional[Path] = _config_option(),
    prior: PriorMode = typer.Option(PriorMode.entropy, help="Prior mode."),
    chrom: Optional[str] = typer.Option(None, help="Restrict to one chromosome."),
    pvalue: Optional[float] = typer.Option(
        None, help="Override genome-wide p-value cutoff."
    ),
    jobs: Optional[int] = typer.Option(
        None, help="Concurrent FINEMAP jobs (defaults to config)."
    ),
) -> None:
    """Genome-wide fine-mapping via PolyFun's create_finemapper_jobs.py."""
    from .pipeline.run_finemap import run_all as _run_all

    cfg = load_config(config)
    if pvalue is not None:
        cfg.finemap.pvalue_cutoff = pvalue
    if jobs is not None:
        cfg.finemap.max_concurrent_jobs = jobs
    _run_all(cfg, prior_mode=prior.value, chrom=chrom)


@app.command()
def aggregate(
    config: Optional[Path] = _config_option(),
    prior: PriorMode = typer.Option(PriorMode.entropy, help="Prior mode."),
    chrom: Optional[str] = typer.Option(None, help="Restrict to one chromosome."),
) -> None:
    """Aggregate per-locus FINEMAP outputs into one TSV."""
    from .pipeline.aggregate import aggregate_results

    cfg = load_config(config)
    aggregate_results(cfg, prior_mode=prior.value, chrom=chrom)


if __name__ == "__main__":
    app()

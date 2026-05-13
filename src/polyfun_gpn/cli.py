"""Typer CLI entry point for polyfun-gpn.

Subcommands:
    setup       clone PolyFun + download FINEMAP binary + chain files
    harmonize   convert raw GWAS TSV to a Polars-friendly hg19 parquet
    build-bg    sample genome-wide entropy and cache background density
    run         end-to-end FINEMAP run on a list of loci (with or without
                entropy priors) — auto-runs prepare + finemap
    aggregate   collect per-locus FINEMAP outputs into one results table

Every CLI flag has a YAML counterpart so the full pipeline can be driven
from one ``-c configs/your.yaml``. CLI flags, when passed, override YAML.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

import typer

from .config import load_config, load_pipeline_config


app = typer.Typer(
    name="polyfun-gpn",
    help="PolyFun + FINEMAP fine-mapping with optional entropy-derived per-SNP priors.",
    no_args_is_help=True,
    add_completion=False,
)


class PriorMode(str, Enum):
    """Supported prior modes.

    * ``none`` — uniform causal prior (PolyFun ``--non-funct``).
    * ``entropy`` — global surprise: ``SNPVAR = exp(tau * -log f_bg(e))``.
    * ``entropy_raw`` — local raw entropy, negated: ``SNPVAR = exp(-tau * e)``.
    """

    none = "none"
    entropy = "entropy"
    entropy_raw = "entropy_raw"


def _config_option() -> Path:
    return typer.Option(
        None,
        "--config",
        "-c",
        help="Path to YAML config (e.g. configs/default-EUR.yaml).",
    )


def _gwas_raw_option() -> Optional[Path]:
    return typer.Option(
        None,
        "--gwas-raw",
        help="Path to raw GWAS TSV (overrides paths.gwas_raw in YAML).",
    )


def _output_dir_option() -> Optional[Path]:
    return typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory (overrides paths.output_dir in YAML).",
    )


def _loci_option() -> Optional[Path]:
    return typer.Option(
        None,
        "--loci",
        help="TSV of loci to fine-map (overrides paths.loci in YAML).",
    )


def _prior_option() -> Optional[PriorMode]:
    return typer.Option(
        None,
        "--prior",
        help=(
            "Prior mode (overrides prior.mode in YAML): "
            "'none', 'entropy' (global surprise), or 'entropy_raw' "
            "(negated raw per-locus entropy)."
        ),
    )


def _ld_mode_option() -> Optional[str]:
    return typer.Option(
        None,
        "--ld-mode",
        help="precomputed_npz (download LD matrices) or plink (LD from --ld-plink genotypes).",
    )


def _ld_npz_prefix_option() -> Optional[str]:
    return typer.Option(
        None,
        "--ld-npz-prefix",
        help="Base URL for precomputed LD NPZ matrices (UKB-style); YAML: finemap.ld_npz_url_prefix.",
    )


def _ld_plink_option() -> Optional[Path]:
    return typer.Option(
        None,
        "--ld-plink",
        help="Plink stem without .bed when --ld-mode plink.",
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
    gwas_raw: Optional[Path] = _gwas_raw_option(),
    output_dir: Optional[Path] = _output_dir_option(),
    overwrite: bool = typer.Option(False, help="Recompute even if output exists."),
) -> None:
    """Stream GWAS TSV and emit a single Parquet with PolyFun columns."""
    from .gwas.io import harmonize_gwas

    cfg = load_pipeline_config(config, gwas_raw=gwas_raw, output_dir=output_dir)
    harmonize_gwas(cfg, overwrite=overwrite)


@app.command("build-bg")
def build_bg(
    config: Optional[Path] = _config_option(),
    n_samples: Optional[int] = typer.Option(
        None, help="Override background.n_samples in YAML."
    ),
    overwrite: bool = typer.Option(False, help="Recompute even if cache exists."),
) -> None:
    """Sample genome-wide entropy values and build background density cache."""
    from .entropy.background import build_background

    cfg = load_config(config)
    if n_samples is not None:
        cfg.background.n_samples = n_samples
    build_background(cfg, overwrite=overwrite)


@app.command()
def run(
    config: Optional[Path] = _config_option(),
    gwas_raw: Optional[Path] = _gwas_raw_option(),
    output_dir: Optional[Path] = _output_dir_option(),
    loci: Optional[Path] = _loci_option(),
    prior: Optional[PriorMode] = _prior_option(),
    ld_mode: Optional[str] = _ld_mode_option(),
    ld_npz_url_prefix: Optional[str] = _ld_npz_prefix_option(),
    ld_plink_prefix: Optional[Path] = _ld_plink_option(),
    skip_prepare: bool = typer.Option(
        False, help="Assume per-locus sumstats already exist."
    ),
) -> None:
    """End-to-end FINEMAP run on the listed loci: prepare + FINEMAP."""
    from .pipeline.prepare_locus import prepare_loci as _prepare
    from .pipeline.run_finemap import run_loci

    cfg = load_pipeline_config(
        config,
        gwas_raw=gwas_raw,
        output_dir=output_dir,
        loci=loci,
        prior_mode=prior.value if prior is not None else None,
        ld_mode=ld_mode,
        ld_npz_url_prefix=ld_npz_url_prefix,
        ld_plink_prefix=ld_plink_prefix,
    )
    loci_path = cfg.paths.absolute("loci")
    if not skip_prepare:
        _prepare(cfg, loci_path=loci_path, prior_mode=cfg.prior.mode)
    run_loci(cfg, loci_path=loci_path, prior_mode=cfg.prior.mode)


@app.command()
def aggregate(
    config: Optional[Path] = _config_option(),
    output_dir: Optional[Path] = _output_dir_option(),
    prior: Optional[PriorMode] = _prior_option(),
) -> None:
    """Aggregate per-locus FINEMAP outputs into one TSV."""
    from .pipeline.aggregate import aggregate_results

    cfg = load_pipeline_config(
        config,
        output_dir=output_dir,
        prior_mode=prior.value if prior is not None else None,
    )
    aggregate_results(cfg, prior_mode=cfg.prior.mode)


if __name__ == "__main__":
    app()

"""CLI overrides for GWAS path, output dir, loci file, and prior mode."""

from __future__ import annotations

from pathlib import Path

import pytest

from polyfun_gpn.config import Config, apply_cli_overrides


def test_cli_gwas_raw_override() -> None:
    cfg = Config()
    apply_cli_overrides(cfg, gwas_raw=Path("/tmp/raw.tsv"))
    assert cfg.paths.gwas_raw == Path("/tmp/raw.tsv")


def test_cli_output_dir_override() -> None:
    cfg = Config()
    apply_cli_overrides(cfg, output_dir=Path("/tmp/runs/foo"))
    assert cfg.paths.output_dir == Path("/tmp/runs/foo")


def test_cli_loci_override() -> None:
    cfg = Config()
    apply_cli_overrides(cfg, loci=Path("/tmp/loci.tsv"))
    assert cfg.paths.loci == Path("/tmp/loci.tsv")


def test_cli_prior_mode_override() -> None:
    cfg = Config()
    apply_cli_overrides(cfg, prior_mode="none")
    assert cfg.prior.mode == "none"
    apply_cli_overrides(cfg, prior_mode="entropy")
    assert cfg.prior.mode == "entropy"
    apply_cli_overrides(cfg, prior_mode="entropy_raw")
    assert cfg.prior.mode == "entropy_raw"
    apply_cli_overrides(cfg, prior_mode="ENTROPY_RAW")
    assert cfg.prior.mode == "entropy_raw"


def test_cli_invalid_prior_mode_raises() -> None:
    cfg = Config()
    with pytest.raises(ValueError):
        apply_cli_overrides(cfg, prior_mode="sldsc")


def test_cli_no_override_keeps_yaml_defaults() -> None:
    cfg = Config()
    original_raw = cfg.paths.gwas_raw
    original_out = cfg.paths.output_dir
    original_loci = cfg.paths.loci
    original_prior = cfg.prior.mode
    apply_cli_overrides(cfg)
    assert cfg.paths.gwas_raw == original_raw
    assert cfg.paths.output_dir == original_out
    assert cfg.paths.loci == original_loci
    assert cfg.prior.mode == original_prior

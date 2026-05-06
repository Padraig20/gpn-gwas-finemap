"""GWAS dataset path layout (CLI + YAML auto_paths)."""

from __future__ import annotations

from pathlib import Path

from polyfun_gpn.config import (
    Config,
    apply_gwas_auto_paths_from_yaml,
    apply_gwas_cli_overrides,
)


def test_cli_gwas_id_sets_convention_paths() -> None:
    cfg = Config()
    apply_gwas_cli_overrides(cfg, gwas_id="AFR", gwas_raw=None)
    assert cfg.gwas_dataset.id == "AFR"
    assert cfg.paths.output_dir == Path("output/AFR")
    assert cfg.paths.gwas_harmonised == Path("data/gwas/AFR/sumstats.hg19.parquet")
    assert cfg.paths.ld_cache == Path("data/ld_cache/AFR")


def test_cli_gwas_raw_only() -> None:
    cfg = Config()
    apply_gwas_cli_overrides(cfg, gwas_id=None, gwas_raw=Path("/tmp/raw.tsv"))
    assert cfg.paths.gwas_raw == Path("/tmp/raw.tsv")
    assert cfg.paths.output_dir == Path("output")


def test_cli_gwas_id_default_does_not_renest() -> None:
    cfg = Config()
    apply_gwas_cli_overrides(cfg, gwas_id="default", gwas_raw=None)
    assert cfg.gwas_dataset.id == "default"
    assert cfg.paths.output_dir == Path("output")


def test_yaml_auto_paths() -> None:
    cfg = Config()
    cfg.gwas_dataset.id = "EAS"
    cfg.gwas_dataset.auto_paths = True
    cfg.paths.gwas_raw = Path("data/gwas/EAS/meta.txt")
    apply_gwas_auto_paths_from_yaml(cfg)
    assert cfg.paths.output_dir == Path("output/EAS")
    assert cfg.paths.gwas_harmonised == Path("data/gwas/EAS/sumstats.hg19.parquet")
    assert cfg.paths.ld_cache == Path("data/ld_cache/EAS")
    assert cfg.paths.gwas_raw == Path("data/gwas/EAS/meta.txt")


def test_load_pipeline_config_cli_overrides_yaml_id() -> None:
    cfg = Config()
    cfg.gwas_dataset.id = "YAML"
    cfg.paths.output_dir = Path("output/yaml_custom")
    apply_gwas_cli_overrides(cfg, gwas_id="CLI", gwas_raw=None)
    assert cfg.gwas_dataset.id == "CLI"
    assert cfg.paths.output_dir == Path("output/CLI")

"""LD configuration (NPZ URLs, Plink genotypes, YAML aliases)."""

from __future__ import annotations

from pathlib import Path

import pytest

from polyfun_gpn.config import (
    Config,
    load_config,
    resolve_plink_prefix,
    validate_finemap_ld,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_legacy_ukb_ld_url_yaml_alias() -> None:
    cfg = Config.model_validate(
        {"finemap": {"ukb_ld_url_prefix": "https://example.org/ld/"}}
    )
    assert cfg.finemap.ld_npz_url_prefix == "https://example.org/ld/"
    assert cfg.finemap.ld_mode == "precomputed_npz"


def test_default_yaml_loads_finemap_optional_paths() -> None:
    cfg = load_config(PROJECT_ROOT / "configs/default.yaml")
    assert cfg.finemap.ld_plink_prefix is None
    assert cfg.finemap.ld_regions_file is None


def test_resolve_plink_prefix_and_validate(tmp_path: Path) -> None:
    proj = tmp_path / "proj"
    proj.mkdir()
    stem = proj / "ref_subset"
    for ext in (".bed", ".bim", ".fam"):
        (Path(str(stem) + ext)).touch()
    cfg = Config()
    cfg.paths.project_root = proj
    cfg.finemap.ld_mode = "plink"
    cfg.finemap.ld_plink_prefix = Path("ref_subset")
    assert resolve_plink_prefix(cfg).name == "ref_subset"
    validate_finemap_ld(cfg)


def test_validate_plink_fails_when_triplet_missing(tmp_path: Path) -> None:
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "x.bed").touch()
    cfg = Config()
    cfg.paths.project_root = proj
    cfg.finemap.ld_mode = "plink"
    cfg.finemap.ld_plink_prefix = Path("x")
    with pytest.raises(FileNotFoundError):
        validate_finemap_ld(cfg)


def test_validate_regions_file_missing() -> None:
    cfg = Config()
    cfg.finemap.ld_mode = "precomputed_npz"
    cfg.finemap.ld_regions_file = Path("/nonexistent/ukb_regions.tsv.gz")
    with pytest.raises(FileNotFoundError):
        validate_finemap_ld(cfg)


def test_validate_regions_optional_ok_when_file_exists(tmp_path: Path) -> None:
    f = tmp_path / "regions.tsv"
    f.write_text("CHR\tSTART\tEND\tURL_PREFIX\n")
    cfg = Config()
    cfg.paths.project_root = tmp_path
    cfg.finemap.ld_mode = "precomputed_npz"
    cfg.finemap.ld_regions_file = Path("regions.tsv")
    validate_finemap_ld(cfg)

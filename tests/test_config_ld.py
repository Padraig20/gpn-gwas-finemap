"""LD configuration (NPZ URLs, Plink genotypes, YAML aliases, CLI overrides)."""

from __future__ import annotations

from pathlib import Path

import pytest

from polyfun_gpn.config import (
    Config,
    apply_cli_overrides,
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
    """The shipped EUR YAML drives an ancestry-matched plink LD panel
    (see scripts/compute_ld/) and the shared-loci TSV produced by
    scripts/find_shared_loci.py. We deliberately don't pin ``prior.mode``
    here -- toggling between 'none' and 'entropy' is a routine A/B and
    shouldn't break the suite."""
    cfg = load_config(PROJECT_ROOT / "configs/default-EUR.yaml")
    assert cfg.finemap.ld_mode == "plink"
    assert cfg.finemap.ld_plink_prefix == Path("data/ld_panels/1000G.EUR.unrelated")
    assert cfg.prior.mode in {"none", "entropy"}
    assert cfg.paths.loci == Path("configs/loci_shared_EUR_EAS.tsv")


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


def test_cli_ld_overrides() -> None:
    cfg = Config()
    apply_cli_overrides(
        cfg,
        ld_mode="plink",
        ld_plink_prefix=Path("/tmp/refs/eur_chr1"),
    )
    assert cfg.finemap.ld_mode == "plink"
    assert cfg.finemap.ld_plink_prefix == Path("/tmp/refs/eur_chr1")


def test_cli_ld_npz_prefix_override() -> None:
    cfg = Config()
    apply_cli_overrides(cfg, ld_npz_url_prefix="https://my-mirror/ld/")
    assert cfg.finemap.ld_npz_url_prefix == "https://my-mirror/ld/"


def test_cli_invalid_ld_mode_raises() -> None:
    cfg = Config()
    with pytest.raises(ValueError):
        apply_cli_overrides(cfg, ld_mode="bogus")

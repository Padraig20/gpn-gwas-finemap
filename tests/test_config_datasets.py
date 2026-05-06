"""Per-ancestry dataset YAMLs in configs/datasets/ load and route paths correctly."""

from __future__ import annotations

from pathlib import Path

import pytest

from polyfun_gpn.config import PROJECT_ROOT, load_pipeline_config


REGISTRY = PROJECT_ROOT / "configs" / "datasets" / "datasets.tsv"


def _registry_rows() -> list[tuple[str, Path, Path]]:
    rows: list[tuple[str, Path, Path]] = []
    text = REGISTRY.read_text().splitlines()
    for line in text[1:]:
        if not line.strip() or line.startswith("#"):
            continue
        cols = line.split("\t")
        slug = cols[0]
        yml = PROJECT_ROOT / cols[1]
        raw = PROJECT_ROOT / cols[2]
        rows.append((slug, yml, raw))
    return rows


def test_registry_has_known_slugs() -> None:
    slugs = {r[0] for r in _registry_rows()}
    assert {"EUR", "AFA", "EAS", "HIS", "SAS"}.issubset(slugs)


@pytest.mark.parametrize("slug, yaml_path, raw_path", _registry_rows())
def test_dataset_yaml_resolves_paths(slug: str, yaml_path: Path, raw_path: Path) -> None:
    assert yaml_path.exists(), f"missing {yaml_path}"
    cfg = load_pipeline_config(yaml_path)

    assert cfg.gwas_dataset.id == slug
    assert cfg.gwas_dataset.auto_paths is True
    assert cfg.paths.output_dir == Path("output") / slug
    assert cfg.paths.gwas_harmonised == Path("data/gwas") / slug / "sumstats.hg19.parquet"
    assert cfg.paths.ld_cache == Path("data/ld_cache") / slug
    assert cfg.paths.gwas_raw == raw_path.relative_to(PROJECT_ROOT)
    assert cfg.builds.gwas == "hg19"
    assert cfg.finemap.ld_mode in ("precomputed_npz", "plink")

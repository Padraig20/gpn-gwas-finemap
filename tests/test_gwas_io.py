"""Unit tests for the GWAS harmonizer."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from polyfun_gpn.config import Config
from polyfun_gpn.gwas.io import harmonize_gwas


def _write_fixture(tmp: Path) -> Path:
    raw = tmp / "gwas_fixture.tsv"
    raw.write_text(
        "Chromsome\tPosition\tEffectAllele\tNonEffectAllele\tBeta\tSE\tEAF\tPval\tNcases\tNcontrols\tNeff\n"
        "1\t100\tA\tC\t0.1\t0.05\t0.2\t1.0e-3\t1000\t9000\t1800.0\n"
        "1\t200\tG\tT\t-0.2\t0.1\t0.7\t5.0e-2\t1000\t9000\t1800.0\n"
        "X\t300\tA\tT\t0.05\t0.05\t0.1\t1.0e-1\t1000\t9000\t1800.0\n"
        "MT\t400\tA\tT\t0.05\t0.05\t0.1\t1.0e-1\t1000\t9000\t1800.0\n"
        "1\t500\tA\tNNN\t0.05\t0.05\t0.1\t1.0e-1\t1000\t9000\t1800.0\n"
    )
    return raw


def test_harmonize_basic(tmp_path: Path) -> None:
    raw = _write_fixture(tmp_path)
    out = tmp_path / "sumstats.parquet"

    cfg = Config()
    cfg.paths.gwas_raw = raw
    cfg.paths.gwas_harmonised = out

    harmonize_gwas(cfg, overwrite=True)
    df = pl.read_parquet(out)

    # MT and the multi-character allele row should be dropped; X retained.
    assert sorted(df["CHR"].unique().to_list()) == ["1", "X"]
    assert df.height == 3

    row = df.filter(pl.col("BP") == 100).row(0, named=True)
    assert row["A1"] == "A"
    assert row["A2"] == "C"
    assert row["Z"] == 0.1 / 0.05
    assert row["N"] == 1800
    assert row["MAF"] == 0.2
    assert row["P"] == 1.0e-3
    assert row["SNP"] == "chr1:100:C:A"


def test_harmonize_skips_when_output_exists(tmp_path: Path) -> None:
    raw = _write_fixture(tmp_path)
    out = tmp_path / "sumstats.parquet"
    cfg = Config()
    cfg.paths.gwas_raw = raw
    cfg.paths.gwas_harmonised = out

    harmonize_gwas(cfg, overwrite=True)
    mtime_first = out.stat().st_mtime_ns
    harmonize_gwas(cfg, overwrite=False)
    assert out.stat().st_mtime_ns == mtime_first

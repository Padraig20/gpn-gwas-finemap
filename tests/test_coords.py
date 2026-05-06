"""Unit tests for chromosome normalization (liftover is exercised in
integration runs once chain files have been downloaded by `setup`)."""

from __future__ import annotations

from polyfun_gpn.coords import normalize_chrom


def test_normalize_chrom_strips_prefix() -> None:
    assert normalize_chrom("chr10") == "10"
    assert normalize_chrom("CHR10") == "10"
    assert normalize_chrom("10") == "10"


def test_normalize_chrom_canonicalises_x_y() -> None:
    assert normalize_chrom("x") == "X"
    assert normalize_chrom("Y") == "Y"
    assert normalize_chrom("chrX") == "X"
    assert normalize_chrom("23") == "X"
    assert normalize_chrom("24") == "Y"


def test_normalize_chrom_handles_none() -> None:
    assert normalize_chrom(None) == ""
    assert normalize_chrom("") == ""

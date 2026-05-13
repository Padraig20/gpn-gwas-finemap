"""Tests for scripts/find_shared_loci.py.

Drives the script's pure functions directly with synthetic Polars frames so
we can exercise the algorithm without spinning up a real GWAS file.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import polars as pl
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "find_shared_loci.py"


@pytest.fixture(scope="module")
def mod():
    """Import scripts/find_shared_loci.py as a module."""
    spec = importlib.util.spec_from_file_location("find_shared_loci", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules["find_shared_loci"] = m
    spec.loader.exec_module(m)
    return m


def _sig_df(rows: list[tuple[str, int, float, str]]) -> pl.DataFrame:
    """Build a (CHR, BP, P, SNP) frame in the schema the script expects."""
    return pl.DataFrame(
        {
            "CHR": [r[0] for r in rows],
            "BP": [r[1] for r in rows],
            "P": [r[2] for r in rows],
            "SNP": [r[3] for r in rows],
        },
        schema={
            "CHR": pl.Utf8,
            "BP": pl.Int64,
            "P": pl.Float64,
            "SNP": pl.Utf8,
        },
    )


def test_peaks_greedy_merges_within_window(mod) -> None:
    df = _sig_df(
        [
            ("1", 10_000_000, 1e-12, "a"),
            ("1", 10_500_000, 1e-9, "b"),  # within 1 Mb of "a"
            ("2", 50_000_000, 1e-15, "c"),
            ("2", 80_000_000, 1e-9, "d"),  # well outside 1 Mb of "c"
        ]
    )
    peaks = mod._peaks_from_sig(df, window_bp=1_000_000)
    # chr1: "a" leads; "b" absorbed -> 1 peak. chr2: 2 distinct peaks.
    chr1 = [p for p in peaks if p["chrom"] == "1"]
    chr2 = [p for p in peaks if p["chrom"] == "2"]
    assert len(chr1) == 1
    assert chr1[0]["lead_snp"] == "a"
    assert {p["lead_bp"] for p in chr2} == {50_000_000, 80_000_000}


def test_find_shared_keeps_only_peaks_present_in_every_gwas(mod) -> None:
    # GWAS A: peaks at chr1:10M, chr2:50M, chr3:30M.
    a = _sig_df(
        [
            ("1", 10_000_000, 1e-12, "a1"),
            ("2", 50_000_000, 1e-12, "a2"),
            ("3", 30_000_000, 1e-12, "a3"),
        ]
    )
    # GWAS B: peaks at chr1:10.5M (overlaps A on chr1), chr2:51M (overlaps A on chr2).
    # No chr3 signal -> chr3 should be dropped.
    b = _sig_df(
        [
            ("1", 10_500_000, 1e-11, "b1"),
            ("2", 51_000_000, 1e-13, "b2"),
        ]
    )
    # GWAS C: peaks at chr1:10.1M, chr2:50.5M; no chr3.
    c = _sig_df(
        [
            ("1", 10_100_000, 1e-10, "c1"),
            ("2", 50_500_000, 1e-14, "c2"),
        ]
    )
    shared = mod.find_shared_loci([a, b, c], window_bp=1_500_000)
    chroms = sorted(s["chrom"] for s in shared)
    assert chroms == ["1", "2"]
    # Lead = best summed -log10(P); for chr2 that's b2 (1e-13) or c2 (1e-14).
    chr2 = next(s for s in shared if s["chrom"] == "2")
    assert chr2["lead_bp"] in {50_000_000, 50_500_000, 51_000_000}


def test_shared_window_is_centered_on_consensus_lead(mod) -> None:
    a = _sig_df([("5", 100_000_000, 1e-12, "a")])
    b = _sig_df([("5", 100_500_000, 1e-30, "b")])  # much stronger
    shared = mod.find_shared_loci([a, b], window_bp=1_000_000)
    assert len(shared) == 1
    s = shared[0]
    # Output window is +/- W around the consensus lead (stronger one in B).
    assert s["lead_bp"] == 100_500_000
    assert s["start"] == 100_500_000 - 1_000_000
    assert s["end"] == 100_500_000 + 1_000_000


def test_write_loci_tsv_matches_demo_schema(mod, tmp_path: Path) -> None:
    shared = [
        {
            "chrom": "10",
            "start": 112_000_001,
            "end": 115_000_001,
            "lead_bp": 114_758_349,
            "lead_snp": "chr10:114758349:C:T",
            "lead_p": 1e-30,
        }
    ]
    out = tmp_path / "loci.tsv"
    mod.write_loci_tsv(shared, out)
    df = pl.read_csv(out, separator="\t")
    # Same columns, same order as configs/loci_demo.tsv.
    assert df.columns == [
        "locus_id",
        "chrom",
        "start",
        "end",
        "gene",
        "lead_rsid",
        "lead_pos_hg19",
    ]
    row = df.row(0, named=True)
    # polars read_csv may infer "10" as Int64; loci.demo.load_loci does
    # str(row["chrom"]) so both representations are valid here.
    assert str(row["chrom"]) == "10"
    assert row["start"] == 112_000_001
    assert row["lead_pos_hg19"] == 114_758_349
    assert row["lead_rsid"] == "chr10:114758349:C:T"


def test_read_significant_round_trips_through_parquet(mod, tmp_path: Path) -> None:
    # PolyFun-style harmonised parquet: CHR/BP/P/SNP -> what the script expects.
    src = pl.DataFrame(
        {
            "SNP": ["x", "y", "z"],
            "CHR": ["1", "1", "2"],
            "BP": [100, 200, 300],
            "P": [1e-12, 5e-1, 1e-9],
        }
    )
    pq = tmp_path / "ss.parquet"
    src.write_parquet(pq)
    got = mod._read_significant(pq, pvalue_cutoff=5e-8)
    # The middle SNP fails the threshold; the other two pass.
    assert sorted(got["SNP"].to_list()) == ["x", "z"]


def test_write_loci_tsv_dedupes_identical_leads(mod, tmp_path: Path) -> None:
    """Two anchor peaks can collapse to the same consensus lead; the output
    TSV must not contain two identical loci rows."""
    shared = [
        {
            "chrom": "1",
            "start": 100,
            "end": 200,
            "lead_bp": 150,
            "lead_snp": "chr1:150:G:A",
            "lead_p": 1e-12,
        },
        {  # exact duplicate
            "chrom": "1",
            "start": 100,
            "end": 200,
            "lead_bp": 150,
            "lead_snp": "chr1:150:G:A",
            "lead_p": 1e-12,
        },
    ]
    out = tmp_path / "loci.tsv"
    mod.write_loci_tsv(shared, out)
    df = pl.read_csv(out, separator="\t")
    assert df.height == 1


def test_write_loci_tsv_leaves_gene_column_truly_empty(mod, tmp_path: Path) -> None:
    """loci_demo.tsv uses plain empty cells for missing values; ensure we
    don't emit a literal ``""`` string in the gene column."""
    shared = [
        {
            "chrom": "1",
            "start": 100,
            "end": 200,
            "lead_bp": 150,
            "lead_snp": "chr1:150:G:A",
            "lead_p": 1e-12,
        }
    ]
    out = tmp_path / "loci.tsv"
    mod.write_loci_tsv(shared, out)
    raw = out.read_text().splitlines()
    # Header + 1 data row, tab-separated, gene cell empty (no quoting).
    assert raw[1].split("\t")[4] == ""


def test_read_significant_raw_tsv_with_diamante_columns(mod, tmp_path: Path) -> None:
    """Raw DIAMANTE-style TSV: Chromsome/Position/Pval columns must be picked
    up, and a stable SNP key synthesised from alleles."""
    raw = pl.DataFrame(
        {
            "Chromsome": ["1", "1"],
            "Position": [100, 200],
            "EffectAllele": ["T", "G"],
            "NonEffectAllele": ["C", "A"],
            "Pval": [1e-12, 0.5],
        }
    )
    path = tmp_path / "raw.tsv"
    raw.write_csv(path, separator="\t")
    got = mod._read_significant(path, pvalue_cutoff=5e-8)
    assert got.height == 1
    snp = got["SNP"][0]
    assert snp == "chr1:100:C:T"

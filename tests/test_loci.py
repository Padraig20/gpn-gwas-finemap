"""Unit tests for locus selection and UKB block snapping."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from polyfun_gpn.loci.demo import Locus, load_loci
from polyfun_gpn.loci.select import (
    UKB_BLOCK_SIZE,
    UKB_BLOCK_STEP,
    UKBBlock,
    select_lead_loci,
    snap_to_ukb_blocks,
)


def test_load_loci_demo_tsv() -> None:
    loci = load_loci(Path("configs/loci_demo.tsv"))
    ids = [l.locus_id for l in loci]
    assert {"TCF7L2", "KCNQ1", "CDKAL1", "FTO", "SLC30A8"}.issubset(set(ids))
    tcf = next(l for l in loci if l.locus_id == "TCF7L2")
    assert tcf.chrom == "10"
    assert tcf.lead_pos == 114758349


def test_snap_to_ukb_blocks_centers_block_on_locus() -> None:
    locus = Locus(
        locus_id="x",
        chrom="10",
        start=112_000_001,
        end=115_000_001,
    )
    ((_l, block),) = snap_to_ukb_blocks([locus])
    assert block.chrom == "10"
    # Block size and step come from PolyFun's documented partitioning.
    assert block.end - block.start == UKB_BLOCK_SIZE
    assert (block.start - 1) % UKB_BLOCK_STEP == 0
    # The chosen block should fully contain the locus midpoint.
    mid = (locus.start + locus.end) // 2
    assert block.start <= mid <= block.end


def test_select_lead_loci_merges_by_window(tmp_path: Path) -> None:
    df = pl.DataFrame(
        {
            "SNP": ["a", "b", "c", "d"],
            "CHR": ["1", "1", "2", "2"],
            "BP": [10_000_000, 10_500_000, 50_000_000, 80_000_000],
            "P": [1e-12, 1e-9, 1e-15, 1e-9],
            "A1": ["A"] * 4,
            "A2": ["C"] * 4,
            "Z": [10.0, 6.0, 12.0, 6.0],
            "N": [1000] * 4,
            "MAF": [0.2] * 4,
        }
    )
    parquet = tmp_path / "ss.parquet"
    df.write_parquet(parquet)

    loci = select_lead_loci(parquet, pvalue_cutoff=5e-8, window_bp=1_000_000)
    # On chr1: SNP "a" leads; "b" within 1 Mb is absorbed -> one locus.
    # On chr2: two distinct loci.
    assert len(loci) == 3
    chr1_loci = [l for l in loci if l.chrom == "1"]
    assert len(chr1_loci) == 1
    chr2_loci = [l for l in loci if l.chrom == "2"]
    assert {l.lead_pos for l in chr2_loci} == {50_000_000, 80_000_000}


def test_ukb_block_url_suffix_format() -> None:
    block = UKBBlock(chrom="10", start=112_000_001, end=115_000_001)
    assert block.url_suffix == "chr10_112000001_115000001"

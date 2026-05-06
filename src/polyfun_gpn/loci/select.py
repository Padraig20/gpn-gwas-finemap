"""Locus selection utilities.

Two pieces of functionality:

1. :func:`select_lead_loci` — find genome-wide-significant lead SNPs in the
   harmonised sumstats and merge their windows into non-overlapping loci.
2. :func:`snap_to_ukb_blocks` — for each locus, return the UKB pre-computed
   LD blocks that cover it. The PolyFun UKB LD blocks are 3 Mb regions
   starting at every 1 Mb step (1, 1_000_001, 2_000_001, ...). For a given
   ``[start, end]`` locus we pick the *single* block whose midpoint is
   closest, since FINEMAP needs one LD matrix per fine-mapping run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import polars as pl

from .demo import Locus


UKB_BLOCK_SIZE = 3_000_000
UKB_BLOCK_STEP = 1_000_000


@dataclass(frozen=True)
class UKBBlock:
    chrom: str
    start: int  # 1-based inclusive
    end: int  # 1-based inclusive

    @property
    def url_suffix(self) -> str:
        return f"chr{self.chrom}_{self.start}_{self.end}"


def select_lead_loci(
    sumstats_path: Path,
    *,
    pvalue_cutoff: float = 5e-8,
    window_bp: int = 1_500_000,
) -> list[Locus]:
    """Return a list of merged loci around genome-wide-significant lead SNPs.

    Greedy 1D merge: sort significant SNPs by p-value ascending, take the
    top SNP as a lead, attach a +/- ``window_bp`` window, mark all SNPs
    inside it as covered, and repeat. The result is a deduplicated list of
    non-overlapping windows on the same chromosome.
    """
    df = (
        pl.scan_parquet(sumstats_path)
        .filter(pl.col("P") < pvalue_cutoff)
        .select(["CHR", "BP", "P", "SNP"])
        .sort("P")
        .collect()
    )
    if df.is_empty():
        return []

    loci: list[Locus] = []
    used: dict[str, list[tuple[int, int]]] = {}
    for row in df.iter_rows(named=True):
        chrom = str(row["CHR"])
        bp = int(row["BP"])
        intervals = used.setdefault(chrom, [])
        if any(s <= bp <= e for s, e in intervals):
            continue
        start = max(1, bp - window_bp)
        end = bp + window_bp
        intervals.append((start, end))
        loci.append(
            Locus(
                locus_id=f"chr{chrom}_{start}_{end}",
                chrom=chrom,
                start=start,
                end=end,
                lead_rsid=row.get("SNP"),
                lead_pos=bp,
            )
        )
    loci.sort(key=lambda l: (int(l.chrom) if l.chrom.isdigit() else 100, l.start))
    return loci


def snap_to_ukb_blocks(loci: Iterable[Locus]) -> list[tuple[Locus, UKBBlock]]:
    """For each locus pick the single UKB 3 Mb block centred on it.

    PolyFun publishes blocks at starts ``1, 1_000_001, 2_000_001, ...``. We
    pick the block whose midpoint is closest to the locus midpoint.
    """
    out: list[tuple[Locus, UKBBlock]] = []
    for locus in loci:
        mid = (locus.start + locus.end) // 2
        block_idx = max(0, (mid - UKB_BLOCK_SIZE // 2 - 1) // UKB_BLOCK_STEP)
        candidates = [
            (i, 1 + i * UKB_BLOCK_STEP)
            for i in (block_idx - 1, block_idx, block_idx + 1)
            if i >= 0
        ]
        best = min(
            candidates,
            key=lambda kv: abs((kv[1] + UKB_BLOCK_SIZE // 2) - mid),
        )
        block_start = best[1]
        block_end = block_start + UKB_BLOCK_SIZE
        out.append(
            (
                locus,
                UKBBlock(chrom=locus.chrom, start=block_start, end=block_end),
            )
        )
    return out

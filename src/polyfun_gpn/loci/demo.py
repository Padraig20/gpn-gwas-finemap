"""Hand-curated demo T2D loci (hg19).

These regions are big enough to span at least one UKB 3 Mb LD block. The
``Locus.from_tsv`` constructor accepts the same TSV schema written in
``configs/loci_demo.tsv`` so users can edit / extend that file freely.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl


@dataclass(frozen=True)
class Locus:
    locus_id: str
    chrom: str
    start: int  # 1-based, inclusive
    end: int  # 1-based, inclusive
    gene: str | None = None
    lead_rsid: str | None = None
    lead_pos: int | None = None

    @property
    def length_bp(self) -> int:
        return self.end - self.start + 1


def load_loci(path: Path) -> list[Locus]:
    df = pl.read_csv(path, separator="\t", has_header=True)
    required = {"locus_id", "chrom", "start", "end"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"loci TSV is missing required columns: {missing}")
    out: list[Locus] = []
    for row in df.iter_rows(named=True):
        out.append(
            Locus(
                locus_id=str(row["locus_id"]),
                chrom=str(row["chrom"]).removeprefix("chr"),
                start=int(row["start"]),
                end=int(row["end"]),
                gene=row.get("gene"),
                lead_rsid=row.get("lead_rsid"),
                lead_pos=int(row["lead_pos_hg19"])
                if row.get("lead_pos_hg19") is not None
                else None,
            )
        )
    return out

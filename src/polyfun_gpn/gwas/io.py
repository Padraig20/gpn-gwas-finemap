"""Stream the raw GWAS TSV and emit a Polars-friendly parquet.

The DIAMANTE-style file we ship in ``data/gwas/`` has 19.3M rows and the
header columns ``Chromsome  Position  EffectAllele  NonEffectAllele  Beta  SE
EAF  Pval  Ncases  Ncontrols  Neff`` (note the spelling). It has no rsIDs, so
we synthesize an unambiguous ``SNP`` key from coordinates + alleles.

Output columns (PolyFun-compatible):
    SNP  CHR  BP  A1  A2  Z  N  MAF  P
where ``A1`` is the effect allele (matches the sign of ``Z``).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from ..config import Config
from ..coords import CANONICAL_CHROMS


def harmonize_gwas(cfg: Config, *, overwrite: bool = False) -> Path:
    src = cfg.paths.absolute("gwas_raw")
    dst = cfg.paths.absolute("gwas_harmonised")

    if dst.exists() and not overwrite:
        print(f"[harmonize] {dst} already exists; skipping (use --overwrite).")
        return dst

    if not src.exists():
        raise FileNotFoundError(f"Raw GWAS TSV not found at {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[harmonize] Streaming {src.name} -> {dst.name}")

    lf = (
        pl.scan_csv(
            src,
            separator="\t",
            has_header=True,
            schema_overrides={
                "Chromsome": pl.Utf8,
                "Position": pl.Int64,
                "EffectAllele": pl.Utf8,
                "NonEffectAllele": pl.Utf8,
                "Beta": pl.Float64,
                "SE": pl.Float64,
                "EAF": pl.Float64,
                "Pval": pl.Float64,
                "Ncases": pl.Float64,
                "Ncontrols": pl.Float64,
                "Neff": pl.Float64,
            },
            null_values=["NA", "", "."],
        )
        .with_columns(
            pl.col("Chromsome")
            .str.replace(r"^chr", "", literal=False)
            .alias("CHR"),
            pl.col("Position").alias("BP"),
            pl.col("EffectAllele").str.to_uppercase().alias("A1"),
            pl.col("NonEffectAllele").str.to_uppercase().alias("A2"),
            (pl.col("Beta") / pl.col("SE")).alias("Z"),
            pl.col("Neff").round(0).cast(pl.Int64).alias("N"),
            pl.when(pl.col("EAF") <= 0.5)
            .then(pl.col("EAF"))
            .otherwise(1.0 - pl.col("EAF"))
            .alias("MAF"),
            pl.col("Pval").alias("P"),
        )
        .filter(
            pl.col("CHR").is_in(list(CANONICAL_CHROMS))
            & pl.col("BP").is_not_null()
            & pl.col("Z").is_finite()
            & pl.col("A1").str.len_chars().eq(1)
            & pl.col("A2").str.len_chars().eq(1)
        )
        .with_columns(
            (
                pl.lit("chr")
                + pl.col("CHR")
                + pl.lit(":")
                + pl.col("BP").cast(pl.Utf8)
                + pl.lit(":")
                + pl.col("A2")
                + pl.lit(":")
                + pl.col("A1")
            ).alias("SNP")
        )
        .select(["SNP", "CHR", "BP", "A1", "A2", "Z", "N", "MAF", "P"])
        .sort(["CHR", "BP"])
    )

    lf.sink_parquet(dst, compression="zstd")
    n = pl.scan_parquet(dst).select(pl.len()).collect().item()
    print(f"[harmonize] Wrote {n:,} rows -> {dst}")
    return dst

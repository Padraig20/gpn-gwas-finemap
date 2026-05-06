"""Build per-locus PolyFun sumstats with SNPVAR priors.

For each locus we:
  1. Slice the harmonised genome-wide sumstats parquet to the locus window
     (hg19 coordinates).
  2. (Optionally) lift each SNP position to the entropy build (hg38) and
     look up its entropy.
  3. Compute SNPVAR per ``prior_mode``.
  4. Write ``output/loci/{prior}/{locus_id}/sumstats.gz`` in the format
     PolyFun's ``finemapper.py`` expects: tab-separated with PolyFun
     columns (``SNPVAR`` only for ``entropy`` / ``uniform``).

Output layout:
    output/loci/{prior}/{locus_id}/
        sumstats.gz        # PolyFun input
        snpvar_audit.tsv   # full per-SNP table including prior_source

We keep the audit file unzipped/uncompressed so it can be inspected easily
even if FINEMAP fails on the locus.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl

from ..config import Config
from ..coords import lift_positions, load_liftover, normalize_chrom
from ..entropy.background import load_background
from ..entropy.lookup import lookup_entropy
from ..entropy.priors import attach_priors
from ..loci.demo import Locus, load_loci
from ..loci.select import UKBBlock, snap_to_ukb_blocks


POLYFUN_COLUMNS_BASE = ["SNP", "CHR", "BP", "A1", "A2", "Z", "N", "P", "MAF"]


def _slice_sumstats(sumstats_path: Path, locus: Locus) -> pl.DataFrame:
    chrom = normalize_chrom(locus.chrom)
    return (
        pl.scan_parquet(sumstats_path)
        .filter(
            (pl.col("CHR") == chrom)
            & (pl.col("BP") >= locus.start)
            & (pl.col("BP") <= locus.end)
        )
        .collect()
    )


def _resolve_entropy_for_locus(cfg: Config, df: pl.DataFrame) -> np.ndarray:
    """Return per-SNP entropy values (NaN if missing)."""
    if df.is_empty():
        return np.empty(0, dtype=np.float32)
    chroms = df["CHR"].to_list()
    positions = df["BP"].to_numpy()
    if cfg.builds.gwas != cfg.builds.entropy:
        lo = load_liftover(
            cfg.paths.absolute("reference_dir"), cfg.builds.gwas, cfg.builds.entropy
        )
        new_chrom, new_pos = lift_positions(lo, chroms, positions)
        return lookup_entropy(
            cfg.paths.absolute("entropy_dir"),
            list(new_chrom),
            new_pos,
        )
    return lookup_entropy(cfg.paths.absolute("entropy_dir"), chroms, positions)


def prepare_locus(
    cfg: Config,
    locus: Locus,
    block: UKBBlock,
    *,
    prior_mode: str,
) -> dict:
    """Prepare one locus and return a small descriptor (paths + counts)."""
    sumstats_src = cfg.paths.absolute("gwas_harmonised")
    df = _slice_sumstats(sumstats_src, locus)
    n_total = df.height

    if prior_mode == "entropy" and n_total > 0:
        density, edges = load_background(cfg)
        entropy_values = _resolve_entropy_for_locus(cfg, df)
        df = attach_priors(df, entropy_values, density, edges, cfg.prior, prior_mode="entropy")
        n_with_entropy = int(np.isfinite(entropy_values).sum())
    elif prior_mode == "uniform" and n_total > 0:
        df = attach_priors(
            df,
            np.full(n_total, np.nan, dtype=np.float32),
            np.empty(0),
            np.empty(0),
            cfg.prior,
            prior_mode="uniform",
        )
        n_with_entropy = 0
    else:
        n_with_entropy = 0

    out_dir = (
        cfg.paths.absolute("output_dir")
        / "loci"
        / prior_mode
        / locus.locus_id
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    sumstats_out = out_dir / "sumstats.tsv"
    audit_out = out_dir / "snpvar_audit.tsv"

    # PolyFun expects: SNP CHR BP A1 A2 Z [N] [SNPVAR]. Keep MAF and P for our
    # own auditing but exclude them from the file we hand to PolyFun (they are
    # not part of the documented PolyFun input schema).
    polyfun_cols = ["SNP", "CHR", "BP", "A1", "A2", "Z", "N"]
    if "SNPVAR" in df.columns:
        polyfun_cols.append("SNPVAR")
    df.select(polyfun_cols).write_csv(
        sumstats_out, separator="\t", include_header=True
    )
    df.write_csv(audit_out, separator="\t", include_header=True)

    descriptor = {
        "locus_id": locus.locus_id,
        "chrom": locus.chrom,
        "start": locus.start,
        "end": locus.end,
        "lead_rsid": locus.lead_rsid,
        "ld_block": block.url_suffix,
        "n_variants": n_total,
        "n_with_entropy": n_with_entropy,
        "prior_mode": prior_mode,
        "sumstats": str(sumstats_out),
    }
    with (out_dir / "locus.json").open("w") as fh:
        json.dump({"locus": asdict(locus), **descriptor}, fh, indent=2)
    return descriptor


def prepare_loci(
    cfg: Config, *, loci_path: Path, prior_mode: str
) -> list[dict]:
    """Prepare a TSV-listed batch of loci."""
    loci = load_loci(loci_path)
    pairs = snap_to_ukb_blocks(loci)
    out: list[dict] = []
    for locus, block in pairs:
        print(
            f"[prepare] {locus.locus_id} ({locus.chrom}:{locus.start}-{locus.end}) "
            f"-> LD block {block.url_suffix}"
        )
        out.append(prepare_locus(cfg, locus, block, prior_mode=prior_mode))
        print(
            f"[prepare]   {out[-1]['n_variants']:,} variants, "
            f"{out[-1]['n_with_entropy']:,} with entropy"
        )
    return out

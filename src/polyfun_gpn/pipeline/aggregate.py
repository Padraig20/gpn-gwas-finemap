"""Aggregate per-locus FINEMAP outputs into a single results table.

Reads each ``output_dir/loci/{prior}/{locus_id}/finemap.gz`` that exists,
prepends a ``locus_id`` column, and concatenates them. Output goes to
``output_dir/results/finemap.demo.{prior}.tsv``.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from ..config import Config


def aggregate_results(cfg: Config, *, prior_mode: str) -> Path:
    """Concatenate per-locus FINEMAP outputs for the given ``prior_mode``."""
    base = cfg.paths.absolute("output_dir") / "loci" / prior_mode
    if not base.exists():
        raise FileNotFoundError(
            f"No per-locus outputs found at {base}. Run `polyfun-gpn run` first."
        )
    rows: list[pl.DataFrame] = []
    for locus_dir in sorted(base.iterdir()):
        finemap_out = locus_dir / "finemap.gz"
        if not finemap_out.exists():
            print(f"[aggregate] skip {locus_dir.name}: no finemap.gz")
            continue
        df = pl.read_csv(finemap_out, separator="\t").with_columns(
            pl.lit(locus_dir.name).alias("locus_id")
        )
        rows.append(df)
    if not rows:
        raise RuntimeError("No FINEMAP outputs to aggregate.")
    combined = pl.concat(rows, how="diagonal_relaxed")
    out_path = (
        cfg.paths.absolute("output_dir")
        / "results"
        / f"finemap.demo.{prior_mode}.tsv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.sort(["locus_id", "PIP"], descending=[False, True]).write_csv(
        out_path, separator="\t", include_header=True
    )
    print(f"[aggregate] wrote {len(combined):,} rows -> {out_path}")
    return out_path

"""Aggregate per-locus FINEMAP outputs into a single results table.

Two modes:

  - **demo loci**: read each ``output/loci/{prior}/{locus_id}/finemap.gz``
    that exists, prepend a ``locus_id`` column, and concatenate. Cheap and
    self-contained; we do this without shelling out to PolyFun.

  - **genome-wide**: defer to PolyFun's ``aggregate_finemapper_results.py``
    which knows how to deduplicate SNPs that appear in multiple overlapping
    3 Mb regions (it keeps the result from the region in which the SNP was
    most central).
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import polars as pl

from ..config import Config


def _polyfun_env(cfg: Config) -> dict[str, str]:
    polyfun_dir = cfg.paths.absolute("polyfun_dir")
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{polyfun_dir}{os.pathsep}{env.get('PYTHONPATH', '')}".strip(os.pathsep)
    )
    return env


def _aggregate_demo_loci(cfg: Config, prior_mode: str) -> Path:
    base = cfg.paths.absolute("output_dir") / "loci" / prior_mode
    if not base.exists():
        raise FileNotFoundError(
            f"No demo-locus outputs found at {base}. Run `polyfun-gpn run` first."
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
        cfg.paths.absolute("output_dir") / "results" / f"finemap.demo.{prior_mode}.tsv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.sort(["locus_id", "PIP"], descending=[False, True]).write_csv(
        out_path, separator="\t", include_header=True
    )
    print(f"[aggregate] wrote {len(combined):,} rows -> {out_path}")
    return out_path


def _aggregate_genome_wide(cfg: Config, prior_mode: str, chrom: str | None) -> Path:
    polyfun_dir = cfg.paths.absolute("polyfun_dir")
    agg_script = polyfun_dir / "aggregate_finemapper_results.py"
    if not agg_script.exists():
        raise RuntimeError(f"{agg_script} missing; run `polyfun-gpn setup`.")

    gw_dir = cfg.paths.absolute("output_dir") / "genome_wide" / prior_mode
    out_prefix = gw_dir / "fm"
    sumstats = gw_dir / f"sumstats.{prior_mode}.tsv"
    if not sumstats.exists():
        raise FileNotFoundError(
            f"Genome-wide sumstats {sumstats} missing; run `run-all` first."
        )

    out_path = (
        cfg.paths.absolute("output_dir")
        / "results"
        / f"finemap.gw.{prior_mode}.tsv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(agg_script),
        "--out-prefix", str(out_prefix),
        "--sumstats", str(sumstats),
        "--out", str(out_path),
        "--pvalue-cutoff", str(cfg.finemap.pvalue_cutoff),
    ]
    if chrom is not None:
        cmd += ["--chr", str(chrom)]

    print("$ " + " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, env=_polyfun_env(cfg), check=True)
    return out_path


def aggregate_results(cfg: Config, *, prior_mode: str, chrom: str | None) -> Path:
    """Pick demo-vs-genome-wide aggregation based on what was actually run."""
    gw_dir = cfg.paths.absolute("output_dir") / "genome_wide" / prior_mode
    if gw_dir.exists() and any(gw_dir.glob("fm.*.gz")):
        return _aggregate_genome_wide(cfg, prior_mode, chrom)
    return _aggregate_demo_loci(cfg, prior_mode)

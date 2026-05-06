#!/usr/bin/env python3
"""Compare two aggregated PolyFun/FINEMAP outputs (e.g. entropy vs no prior).

Typical inputs after `polyfun-gpn aggregate`:
    output/results/finemap.demo.entropy.tsv
    output/results/finemap.demo.none.tsv

Or copies at the repo root with the same columns.

Joins on ``locus_id`` + ``SNP`` and reports PIP correlation, top-SNP agreement,
and largest PIP shifts.

Usage:
    uv run python scripts/compare_finemap_priormodes.py \\
        --entropy finemap.demo.entropy.tsv \\
        --none finemap.demo.none.tsv

    uv run python scripts/compare_finemap_priormodes.py \\
        -e output/results/finemap.demo.entropy.tsv \\
        -n output/results/finemap.demo.none.tsv \\
        --per-locus-csv compare_pips_by_locus.csv

    # After ``polyfun-gpn --gwas-id AFR run`` / ``aggregate``:
    uv run python scripts/compare_finemap_priormodes.py --gwas-id AFR
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats


COLS_JOIN = ("locus_id", "SNP")
COL_PIP = "PIP"
COL_CS = "CREDIBLE_SET"


def _read(path: Path) -> pl.DataFrame:
    if not path.exists():
        sys.exit(f"File not found: {path}")
    return pl.read_csv(path, separator="\t", infer_schema_length=10_000)


def _top_snp_by_pip(df: pl.DataFrame) -> pl.DataFrame:
    """One row per locus: SNP with max PIP (tie-break: higher BP)."""
    return (
        df.sort(["locus_id", COL_PIP, "BP"], descending=[False, True, True])
        .group_by("locus_id", maintain_order=True)
        .agg(
            pl.first("SNP").alias("top_snp"),
            pl.first(COL_PIP).alias("top_pip"),
            pl.first(COL_CS).alias("top_credible_set"),
        )
    )


def _safe_corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Pearson r, p, Spearman rho, p."""
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan"), float("nan"), float("nan"), float("nan")
    x_, y_ = x[ok], y[ok]
    if np.unique(x_).size == 1 or np.unique(y_).size == 1:
        return float("nan"), float("nan"), float("nan"), float("nan")
    pr, pp = stats.pearsonr(x_, y_)
    sr, sp = stats.spearmanr(x_, y_)
    return float(pr), float(pp), float(sr), float(sp)


def _default_results_dir(gwas_id: str | None) -> Path:
    if gwas_id and gwas_id.strip() and gwas_id.strip() != "default":
        return Path("output") / gwas_id.strip() / "results"
    return Path("output/results")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--gwas-id",
        type=str,
        default=None,
        help=(
            "If set (not 'default'), default -e/-n become "
            "output/{id}/results/finemap.demo.{entropy,none}.tsv"
        ),
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory containing finemap.demo.*.tsv (overrides --gwas-id defaults)",
    )
    p.add_argument(
        "-e",
        "--entropy",
        type=Path,
        default=None,
        help="TSV from entropy (or any) prior mode",
    )
    p.add_argument(
        "-n",
        "--none",
        dest="none_path",
        type=Path,
        default=None,
        help='TSV from unpriored fine-mapping (--prior none)',
    )
    p.add_argument(
        "--per-locus-csv",
        type=Path,
        default=None,
        help="Optional: write per-locus summary table CSV",
    )
    p.add_argument(
        "--big-shift-threshold",
        type=float,
        default=0.05,
        help="Highlight |PIP_entropy − PIP_none| ≥ this (default 0.05)",
    )
    args = p.parse_args()

    res_dir = args.results_dir or _default_results_dir(args.gwas_id)
    entropy_path = args.entropy or (res_dir / "finemap.demo.entropy.tsv")
    none_path = args.none_path or (res_dir / "finemap.demo.none.tsv")

    print(f"[compare] entropy TSV: {entropy_path.resolve()}")
    print(f"[compare] none TSV:    {none_path.resolve()}")

    ent = _read(entropy_path)
    non = _read(none_path)

    for df, tag in [(ent, "--entropy"), (non, "--none")]:
        missing = [c for c in (*COLS_JOIN, COL_PIP) if c not in df.columns]
        if missing:
            sys.exit(f"{tag}: missing columns {missing}; have {df.columns}")

    e_cols = [*COLS_JOIN, "CHR", "BP", "A1", "A2", COL_PIP, COL_CS]
    if "SNPVAR" in ent.columns:
        e_cols.append("SNPVAR")
    e = ent.select(e_cols).rename({COL_PIP: "PIP_e", COL_CS: "CS_e"})
    n = non.select([*COLS_JOIN, COL_PIP, COL_CS]).rename({COL_PIP: "PIP_n", COL_CS: "CS_n"})

    merged = e.join(n, on=list(COLS_JOIN), how="inner")
    only_e = e.join(n.select(list(COLS_JOIN)), on=list(COLS_JOIN), how="anti").height
    only_n = n.join(e.select(list(COLS_JOIN)), on=list(COLS_JOIN), how="anti").height

    merged = merged.with_columns(
        (pl.col("PIP_e") - pl.col("PIP_n")).alias("delta_pip"),
        (pl.col("PIP_e") - pl.col("PIP_n")).abs().alias("abs_delta_pip"),
    )

    print("=== Rows ===")
    print(f"Entropy file rows:           {ent.height:,}")
    print(f"None file rows:              {non.height:,}")
    print(f"Matched (locus_id + SNP):    {merged.height:,}")
    if only_e or only_n:
        print(f"SNPs only in entropy file:  {only_e:,}")
        print(f"SNPs only in none file:     {only_n:,}")

    x = merged["PIP_e"].to_numpy()
    y = merged["PIP_n"].to_numpy()
    pr, pp, sr, sp = _safe_corr(x, y)
    print("\n=== Global PIP agreement (all matched SNPs) ===")
    print(f"Pearson r:   {pr:.4f}  (p={pp:.2e})" if np.isfinite(pr) else "Pearson r:   n/a")
    print(f"Spearman ρ:  {sr:.4f}  (p={sp:.2e})" if np.isfinite(sr) else "Spearman ρ:  n/a")
    print(f"Mean |ΔPIP|: {merged['abs_delta_pip'].mean():.6f}")
    print(f"Max |ΔPIP|:  {merged['abs_delta_pip'].max():.6f}")

    # Per-locus
    locus_stats = []
    for loc in merged["locus_id"].unique().sort():
        sub = merged.filter(pl.col("locus_id") == loc)
        xe = sub["PIP_e"].to_numpy()
        yn = sub["PIP_n"].to_numpy()
        pr_l, pp_l, sr_l, sp_l = _safe_corr(xe, yn)
        locus_stats.append(
            {
                "locus_id": loc,
                "n_snps": sub.height,
                "pearson_r": pr_l,
                "spearman_rho": sr_l,
                "mean_abs_delta_pip": float(sub["abs_delta_pip"].mean()),
            }
        )

    locus_df = pl.DataFrame(locus_stats)
    print("\n=== Per-locus PIP correlation ===")
    with pl.Config(tbl_rows=20, tbl_width_chars=120):
        print(locus_df)

    # Top SNP per locus
    top_e = _top_snp_by_pip(ent.rename({COL_PIP: "PIP"})).rename(
        {
            "top_snp": "top_snp_e",
            "top_pip": "top_pip_e",
            "top_credible_set": "top_cs_e",
        }
    )
    top_n = _top_snp_by_pip(non.rename({COL_PIP: "PIP"})).rename(
        {
            "top_snp": "top_snp_n",
            "top_pip": "top_pip_n",
            "top_credible_set": "top_cs_n",
        }
    )
    top_cmp = top_e.join(top_n, on="locus_id", how="inner").with_columns(
        (pl.col("top_snp_e") == pl.col("top_snp_n")).alias("same_top_snp")
    )
    print("\n=== Top SNP by PIP (per locus) ===")
    print(
        top_cmp.select(
            "locus_id",
            "same_top_snp",
            "top_snp_e",
            "top_pip_e",
            "top_snp_n",
            "top_pip_n",
        )
    )
    n_same = int(top_cmp["same_top_snp"].sum())
    print(f"\nLoci with identical argmax-PIP SNP: {n_same} / {top_cmp.height}")

    thr = args.big_shift_threshold
    big = merged.filter(pl.col("abs_delta_pip") >= thr).sort(
        "abs_delta_pip", descending=True
    )
    print(f"\n=== Largest |ΔPIP| (≥ {thr}) — top 25 ===")
    print(
        big.head(25).select(
            "locus_id",
            "SNP",
            "BP",
            "PIP_e",
            "PIP_n",
            "delta_pip",
            "CS_e",
            "CS_n",
        )
    )

    if args.per_locus_csv is not None:
        out = locus_df.join(
            top_cmp.select(
                "locus_id",
                "same_top_snp",
                "top_snp_e",
                "top_pip_e",
                "top_snp_n",
                "top_pip_n",
            ),
            on="locus_id",
            how="left",
        )
        out.write_csv(args.per_locus_csv)
        print(f"\nWrote per-locus table -> {args.per_locus_csv}")


if __name__ == "__main__":
    main()

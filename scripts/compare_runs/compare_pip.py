#!/usr/bin/env python3
"""Compare PIPs across fine-mapping runs (ancestry x prior).

Reads the four aggregated FINEMAP tables that ``polyfun-gpn aggregate``
writes -- one per (ancestry, prior) pair -- and produces:

* PIP-threshold sweep plots (the "# variants with PIP > T" curve in the
  reference figure), in two flavours:
    - per ancestry, entropy vs uniform
    - per prior, EUR vs EAS
* Paired PIP scatter plots:
    - per ancestry, entropy vs uniform (test "entropy raises PIP")
    - per prior, EUR vs EAS (test "shared peaks fine-map to the same SNP")
* A cross-ancestry replication-rate curve: for each PIP threshold T,
  the fraction of variants with PIP > T in ancestry A that also have
  PIP > T in ancestry B. Symmetric (mean of A->B and B->A).
* A delta histogram (PIP_entropy - PIP_uniform) per ancestry.
* A summary statistics TSV with:
    - Spearman / Pearson correlation per pairwise comparison
    - Paired Wilcoxon signed-rank p-value (entropy vs uniform)
    - Mean / median PIP delta
    - # variants over PIP thresholds 0.5 / 0.9 in each run
    - # variants over thresholds in BOTH ancestries (replication count)

The four inputs default to the paths produced by the bundled YAMLs:

    output/EUR/results/finemap.demo.none.tsv     (--eur-none)
    output/EUR/results/finemap.demo.entropy.tsv  (--eur-entropy)
    output/EAS/results/finemap.demo.none.tsv     (--eas-none)
    output/EAS/results/finemap.demo.entropy.tsv  (--eas-entropy)

Variants are joined on (CHR, BP, A1, A2) -- the same compound key PolyFun
uses to merge sumstats and LD. For cross-ancestry joins we also try the
strand-swapped key (A2, A1) so the analysis isn't penalised by allele
ordering differences between the two GWAS.

Usage:
    uv run python scripts/compare_runs/compare_pip.py \\
        --output-dir output/compare
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import typer
from scipy import stats


app = typer.Typer(
    name="compare-pip",
    help="Compare PIPs across (ancestry, prior) fine-mapping runs.",
    add_completion=False,
    no_args_is_help=False,
)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Run:
    """One (ancestry, prior) result table -- the schema written by
    ``polyfun-gpn aggregate``."""

    ancestry: str          # "EUR" | "EAS"
    prior: str             # "none" | "entropy"
    path: Path
    df: pl.DataFrame       # SNP CHR BP A1 A2 PIP ... locus_id

    @property
    def label(self) -> str:
        # "EUR/entropy" is short and unambiguous in plot legends / TSVs.
        return f"{self.ancestry}/{self.prior}"


JOIN_KEYS = ("CHR", "BP", "A1", "A2")


def _load_run(path: Path, ancestry: str, prior: str) -> Run:
    if not path.exists():
        raise typer.BadParameter(f"{ancestry}/{prior}: input not found at {path}")
    df = pl.read_csv(path, separator="\t").select(
        # We don't actually need BETA_*/DISTANCE_FROM_CENTER for the
        # comparison; keep only the join keys + PIP + locus_id (helpful
        # for downstream debugging).
        [c for c in ("locus_id", "SNP", "CHR", "BP", "A1", "A2", "PIP") if c in pl.read_csv(path, separator="\t", n_rows=0).columns]
    ).with_columns(
        pl.col("CHR").cast(pl.Utf8).str.replace(r"^chr", "", literal=False),
        pl.col("BP").cast(pl.Int64),
        pl.col("A1").cast(pl.Utf8).str.to_uppercase(),
        pl.col("A2").cast(pl.Utf8).str.to_uppercase(),
        pl.col("PIP").cast(pl.Float64),
    )
    # PolyFun emits one row per variant per locus; if a variant spans
    # multiple locus windows (rare with 1.5 Mb windows but it happens at
    # boundaries) keep the row with the highest PIP -- that's the locus
    # that "owns" the variant.
    df = df.sort("PIP", descending=True).unique(subset=list(JOIN_KEYS), keep="first")
    return Run(ancestry=ancestry, prior=prior, path=path, df=df)


# ---------------------------------------------------------------------------
# Joins
# ---------------------------------------------------------------------------


def _paired_pip(a: Run, b: Run, *, allow_allele_swap: bool) -> pl.DataFrame:
    """Inner-join two runs on (CHR, BP, A1, A2), returning a frame with
    columns ``CHR BP A1 A2 PIP_a PIP_b`` and one row per shared variant.

    For cross-ancestry comparisons we also try the strand-swapped key
    (A2, A1) and merge those matches in. For paired entropy-vs-uniform
    comparisons (same ancestry) the GWAS sumstats are identical so a
    plain (A1, A2) join is enough.
    """
    left = a.df.select(["CHR", "BP", "A1", "A2", "PIP"]).rename({"PIP": "PIP_a"})
    right = b.df.select(["CHR", "BP", "A1", "A2", "PIP"]).rename({"PIP": "PIP_b"})
    direct = left.join(right, on=list(JOIN_KEYS), how="inner")
    if not allow_allele_swap:
        return direct
    # Variants the same-orientation join missed: try (A1, A2) <-> (A2, A1).
    matched_keys = direct.select(JOIN_KEYS)
    unmatched_left = left.join(matched_keys, on=list(JOIN_KEYS), how="anti")
    swapped = right.rename({"A1": "A2_swap", "A2": "A1_swap"}).rename(
        {"A1_swap": "A1", "A2_swap": "A2"}
    )
    swap_join = unmatched_left.join(swapped, on=list(JOIN_KEYS), how="inner")
    if swap_join.height == 0:
        return direct
    return pl.concat([direct, swap_join], how="vertical_relaxed")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


PIP_THRESHOLDS = np.arange(0.50, 1.001, 0.05)
PIP_FINE_THRESHOLDS = np.linspace(0.0, 1.0, 51)

ENTROPY_COLOR = "#1f77b4"   # matplotlib blue, matches the reference plot
UNIFORM_COLOR = "#ff7f0e"   # matplotlib orange, matches the reference plot
EUR_COLOR = "#2ca02c"
EAS_COLOR = "#d62728"


def _count_above(pip: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    # vectorised "# variants with PIP > T" for the threshold sweep plot.
    return (pip[None, :] > thresholds[:, None]).sum(axis=1)


def plot_threshold_sweep_by_ancestry(
    runs: dict[tuple[str, str], Run], out_dir: Path
) -> Path:
    """One panel per ancestry. In each panel, two curves (entropy, uniform).
    Y-axis = # variants with PIP > threshold. Reproduces the reference
    plot you sent, but for both ancestries side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, anc in zip(axes, ("EUR", "EAS")):
        for prior, color in (("entropy", ENTROPY_COLOR), ("none", UNIFORM_COLOR)):
            pip = runs[(anc, prior)].df["PIP"].to_numpy()
            counts = _count_above(pip, PIP_THRESHOLDS)
            ax.plot(
                PIP_THRESHOLDS,
                counts,
                marker="o",
                color=color,
                label=("With entropy prior" if prior == "entropy" else "Uniform prior"),
            )
        ax.set_title(f"FINEMAP — {anc}")
        ax.set_xlabel("PIP threshold")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
    axes[0].set_ylabel("# variants with PIP > threshold")
    out = out_dir / "threshold_sweep_by_ancestry.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_threshold_sweep_by_prior(
    runs: dict[tuple[str, str], Run], out_dir: Path
) -> Path:
    """One panel per prior. In each panel, two curves (EUR, EAS). Lets you
    eyeball whether the two ancestries produce similar PIP-count profiles
    (cross-ancestry agreement at the population level, not paired)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, prior in zip(axes, ("none", "entropy")):
        for anc, color in (("EUR", EUR_COLOR), ("EAS", EAS_COLOR)):
            pip = runs[(anc, prior)].df["PIP"].to_numpy()
            counts = _count_above(pip, PIP_THRESHOLDS)
            ax.plot(PIP_THRESHOLDS, counts, marker="o", color=color, label=anc)
        ax.set_title(f"FINEMAP — {prior} prior")
        ax.set_xlabel("PIP threshold")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
    axes[0].set_ylabel("# variants with PIP > threshold")
    out = out_dir / "threshold_sweep_by_prior.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_replication_rate(
    runs: dict[tuple[str, str], Run], out_dir: Path
) -> Path:
    """For each prior, plot the cross-ancestry replication rate:
    P(PIP_B > T | PIP_A > T), averaged over the two directions
    A=EUR,B=EAS and A=EAS,B=EUR (symmetric)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, prior in zip(axes, ("none", "entropy")):
        paired = _paired_pip(
            runs[("EUR", prior)], runs[("EAS", prior)], allow_allele_swap=True
        )
        if paired.is_empty():
            ax.text(0.5, 0.5, "no shared variants", ha="center", va="center",
                    transform=ax.transAxes)
        else:
            eur = paired["PIP_a"].to_numpy()
            eas = paired["PIP_b"].to_numpy()
            rates = []
            for t in PIP_FINE_THRESHOLDS:
                # # variants above T in EUR; of those, what fraction is also above T in EAS?
                a_above = eur > t
                b_above = eas > t
                n_a = a_above.sum()
                n_b = b_above.sum()
                p_a_to_b = (a_above & b_above).sum() / n_a if n_a > 0 else np.nan
                p_b_to_a = (a_above & b_above).sum() / n_b if n_b > 0 else np.nan
                # Symmetric reading: average of the two conditional probs.
                vals = [v for v in (p_a_to_b, p_b_to_a) if not np.isnan(v)]
                rates.append(np.mean(vals) if vals else np.nan)
            ax.plot(PIP_FINE_THRESHOLDS, rates, marker=".", color="#444")
        ax.set_title(f"EUR <-> EAS replication ({prior} prior)")
        ax.set_xlabel("PIP threshold (in both ancestries)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Replication rate (symmetric)")
    out = out_dir / "replication_rate.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_paired_scatter_entropy_vs_uniform(
    runs: dict[tuple[str, str], Run], out_dir: Path
) -> Path:
    """For each ancestry, paired PIP scatter (uniform on X, entropy on Y).
    Reference y=x. Points above the diagonal had their PIP raised by the
    entropy prior; points below had it lowered."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, anc in zip(axes, ("EUR", "EAS")):
        paired = _paired_pip(
            runs[(anc, "none")], runs[(anc, "entropy")], allow_allele_swap=False
        )
        ax.plot([0, 1], [0, 1], color="grey", lw=1, ls="--")
        if paired.is_empty():
            ax.text(0.5, 0.5, "no shared variants", ha="center", va="center",
                    transform=ax.transAxes)
        else:
            ax.scatter(
                paired["PIP_a"].to_numpy(),
                paired["PIP_b"].to_numpy(),
                s=6, alpha=0.35, color=ENTROPY_COLOR,
            )
        ax.set_title(f"{anc}: uniform vs entropy  (n={paired.height})")
        ax.set_xlabel("PIP (uniform prior)")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("PIP (entropy prior)")
    out = out_dir / "scatter_entropy_vs_uniform.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_scatter_eur_vs_eas(
    runs: dict[tuple[str, str], Run], out_dir: Path
) -> Path:
    """For each prior, scatter PIP_EUR (X) vs PIP_EAS (Y) for paired variants.
    Reference y=x. Cluster on the diagonal = strong cross-ancestry agreement."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, prior in zip(axes, ("none", "entropy")):
        paired = _paired_pip(
            runs[("EUR", prior)], runs[("EAS", prior)], allow_allele_swap=True
        )
        ax.plot([0, 1], [0, 1], color="grey", lw=1, ls="--")
        if paired.is_empty():
            ax.text(0.5, 0.5, "no shared variants", ha="center", va="center",
                    transform=ax.transAxes)
        else:
            ax.scatter(
                paired["PIP_a"].to_numpy(),
                paired["PIP_b"].to_numpy(),
                s=6, alpha=0.35, color=EUR_COLOR if prior == "none" else ENTROPY_COLOR,
            )
        ax.set_title(f"{prior} prior: EUR vs EAS  (n={paired.height})")
        ax.set_xlabel("PIP (EUR)")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("PIP (EAS)")
    out = out_dir / "scatter_eur_vs_eas.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_delta_hist_entropy_vs_uniform(
    runs: dict[tuple[str, str], Run], out_dir: Path
) -> Path:
    """Histogram of PIP_entropy - PIP_uniform per ancestry.
    A right-shifted distribution = entropy raises PIPs on average."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, anc in zip(axes, ("EUR", "EAS")):
        paired = _paired_pip(
            runs[(anc, "none")], runs[(anc, "entropy")], allow_allele_swap=False
        )
        ax.axvline(0, color="grey", lw=1, ls="--")
        if not paired.is_empty():
            delta = paired["PIP_b"].to_numpy() - paired["PIP_a"].to_numpy()
            ax.hist(delta, bins=50, color=ENTROPY_COLOR, alpha=0.85)
            ax.axvline(np.median(delta), color="black", lw=1,
                       label=f"median = {np.median(delta):+.3f}")
            ax.legend(loc="upper right")
        ax.set_title(f"{anc}: PIP(entropy) - PIP(uniform)")
        ax.set_xlabel("PIP delta")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("# variants")
    out = out_dir / "delta_hist_entropy_vs_uniform.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def _per_run_stats(runs: dict[tuple[str, str], Run]) -> list[dict]:
    """Marginal stats per (ancestry, prior) -- counts at standard PIP thresholds."""
    rows = []
    for (anc, prior), run in runs.items():
        pip = run.df["PIP"].to_numpy()
        rows.append({
            "comparison_type": "marginal",
            "label": run.label,
            "n_variants": int(len(pip)),
            "n_pip_gt_0.5": int((pip > 0.5).sum()),
            "n_pip_gt_0.9": int((pip > 0.9).sum()),
            "n_pip_gt_0.95": int((pip > 0.95).sum()),
            "mean_pip": float(np.mean(pip)),
            "median_pip": float(np.median(pip)),
        })
    return rows


def _paired_stats(
    label: str,
    paired: pl.DataFrame,
    *,
    is_within_ancestry: bool,
) -> dict:
    """Stats for one paired comparison. ``PIP_a`` is the X-axis run,
    ``PIP_b`` is the Y-axis run (entropy if within-ancestry, EAS if
    cross-ancestry). Sign of the Wilcoxon test follows that orientation
    so a negative ``median_delta`` means "entropy lowered PIP".
    """
    n = paired.height
    out: dict = {
        "comparison_type": "paired_within_ancestry" if is_within_ancestry else "paired_cross_ancestry",
        "label": label,
        "n_paired": n,
    }
    if n < 3:
        out.update({
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "wilcoxon_p_two_sided": np.nan,
            "mean_delta_b_minus_a": np.nan,
            "median_delta_b_minus_a": np.nan,
            "n_b_higher": 0,
            "n_a_higher": 0,
            "n_both_pip_gt_0.5": 0,
            "n_both_pip_gt_0.9": 0,
        })
        return out
    a = paired["PIP_a"].to_numpy()
    b = paired["PIP_b"].to_numpy()
    delta = b - a
    spearman = stats.spearmanr(a, b)
    pearson = stats.pearsonr(a, b)
    # Wilcoxon: drop exact ties so scipy doesn't warn and "auto"-deweight them.
    nonzero = delta != 0
    if nonzero.sum() >= 3:
        wilcoxon_p = float(stats.wilcoxon(delta[nonzero]).pvalue)
    else:
        wilcoxon_p = np.nan
    out.update({
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "wilcoxon_p_two_sided": wilcoxon_p,
        "mean_delta_b_minus_a": float(np.mean(delta)),
        "median_delta_b_minus_a": float(np.median(delta)),
        "n_b_higher": int((delta > 0).sum()),
        "n_a_higher": int((delta < 0).sum()),
        "n_both_pip_gt_0.5": int(((a > 0.5) & (b > 0.5)).sum()),
        "n_both_pip_gt_0.9": int(((a > 0.9) & (b > 0.9)).sum()),
    })
    return out


def build_summary(runs: dict[tuple[str, str], Run]) -> pl.DataFrame:
    rows: list[dict] = _per_run_stats(runs)

    for anc in ("EUR", "EAS"):
        paired = _paired_pip(
            runs[(anc, "none")], runs[(anc, "entropy")], allow_allele_swap=False
        )
        rows.append(_paired_stats(
            f"{anc}: entropy vs uniform", paired, is_within_ancestry=True
        ))

    for prior in ("none", "entropy"):
        paired = _paired_pip(
            runs[("EUR", prior)], runs[("EAS", prior)], allow_allele_swap=True
        )
        rows.append(_paired_stats(
            f"{prior} prior: EUR vs EAS", paired, is_within_ancestry=False
        ))

    # Order columns: pick a stable union so the TSV stays readable.
    cols = [
        "comparison_type", "label",
        "n_variants", "n_paired",
        "n_pip_gt_0.5", "n_pip_gt_0.9", "n_pip_gt_0.95",
        "mean_pip", "median_pip",
        "spearman_rho", "spearman_p",
        "pearson_r", "pearson_p",
        "wilcoxon_p_two_sided",
        "mean_delta_b_minus_a", "median_delta_b_minus_a",
        "n_b_higher", "n_a_higher",
        "n_both_pip_gt_0.5", "n_both_pip_gt_0.9",
    ]
    normalised = []
    for r in rows:
        normalised.append({c: r.get(c, None) for c in cols})
    return pl.DataFrame(normalised, strict=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _default(path: str) -> Path:
    return Path(path)


@app.command()
def main(
    eur_none: Path = typer.Option(
        _default("output/EUR/results/finemap.demo.none.tsv"),
        "--eur-none",
        help="EUR uniform-prior aggregated FINEMAP TSV.",
    ),
    eur_entropy: Path = typer.Option(
        _default("output/EUR/results/finemap.demo.entropy.tsv"),
        "--eur-entropy",
        help="EUR entropy-prior aggregated FINEMAP TSV.",
    ),
    eas_none: Path = typer.Option(
        _default("output/EAS/results/finemap.demo.none.tsv"),
        "--eas-none",
        help="EAS uniform-prior aggregated FINEMAP TSV.",
    ),
    eas_entropy: Path = typer.Option(
        _default("output/EAS/results/finemap.demo.entropy.tsv"),
        "--eas-entropy",
        help="EAS entropy-prior aggregated FINEMAP TSV.",
    ),
    output_dir: Path = typer.Option(
        _default("output/compare"),
        "--output-dir",
        "-o",
        help="Where to write plots and summary.tsv.",
    ),
) -> None:
    """Build comparison plots + summary stats across four fine-mapping runs."""
    typer.echo("[compare] loading runs:")
    runs = {
        ("EUR", "none"): _load_run(eur_none, "EUR", "none"),
        ("EUR", "entropy"): _load_run(eur_entropy, "EUR", "entropy"),
        ("EAS", "none"): _load_run(eas_none, "EAS", "none"),
        ("EAS", "entropy"): _load_run(eas_entropy, "EAS", "entropy"),
    }
    for r in runs.values():
        typer.echo(f"[compare]   {r.label:<15s}  n={r.df.height:>8,}  ({r.path})")

    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    typer.echo("[compare] writing plots:")
    for p in (
        plot_threshold_sweep_by_ancestry(runs, plots_dir),
        plot_threshold_sweep_by_prior(runs, plots_dir),
        plot_replication_rate(runs, plots_dir),
        plot_paired_scatter_entropy_vs_uniform(runs, plots_dir),
        plot_scatter_eur_vs_eas(runs, plots_dir),
        plot_delta_hist_entropy_vs_uniform(runs, plots_dir),
    ):
        typer.echo(f"[compare]   {p}")

    summary = build_summary(runs)
    summary_path = output_dir / "summary.tsv"
    summary.write_csv(summary_path, separator="\t", include_header=True)
    typer.echo(f"[compare] summary -> {summary_path}")
    # Echo the paired-comparison rows so the headline numbers are immediately
    # visible without opening the TSV. Marginal rows are easy to glance over.
    typer.echo("[compare] paired comparisons:")
    paired_rows = summary.filter(pl.col("comparison_type") != "marginal")
    for row in paired_rows.iter_rows(named=True):
        typer.echo(
            f"[compare]   {row['label']:<35s}  n={row['n_paired']:>7}  "
            f"spearman_rho={row['spearman_rho']:+.3f}  "
            f"wilcoxon_p={row['wilcoxon_p_two_sided']:.2e}  "
            f"median_delta={row['median_delta_b_minus_a']:+.4f}"
        )


if __name__ == "__main__":
    app()

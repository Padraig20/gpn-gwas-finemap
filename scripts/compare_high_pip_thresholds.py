"""Compare high-PIP counts for entropy-prior and uniform fine-mapping runs."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count variants above increasing PIP thresholds for prior vs no-prior runs.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/t2d_entropy_finemap"),
        help="Directory produced by `gpn-finemap run-fine-mapping`.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("results/t2d_entropy_finemap/high_pip_threshold_comparison"),
        help="Output prefix for .tsv and .png files.",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.05,
        help="PIP threshold spacing from --min-threshold to 1, inclusive.",
    )
    parser.add_argument(
        "--min-threshold",
        type=float,
        default=0.5,
        help="Smallest PIP threshold to include.",
    )
    args = parser.parse_args()

    thresholds = make_thresholds(args.threshold_step, args.min_threshold)
    records = collect_counts(args.results_dir, thresholds)
    if not records:
        raise SystemExit(f"No completed SuSiE/FINEMAP PIP outputs found under {args.results_dir}")

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_summary(records, args.output_prefix.with_suffix(".tsv"))
    plot_counts(records, args.output_prefix.with_suffix(".png"))
    print(f"Wrote {args.output_prefix.with_suffix('.tsv')}")
    print(f"Wrote {args.output_prefix.with_suffix('.png')}")


def make_thresholds(step: float, min_threshold: float) -> list[float]:
    if step <= 0 or step > 1:
        raise ValueError("--threshold-step must be in (0, 1]")
    if min_threshold < 0 or min_threshold > 1:
        raise ValueError("--min-threshold must be in [0, 1]")
    n_steps = int(round((1 - min_threshold) / step))
    thresholds = [round(min_threshold + i * step, 10) for i in range(n_steps + 1)]
    if thresholds[-1] != 1.0:
        thresholds.append(1.0)
    return thresholds


def collect_counts(results_dir: Path, thresholds: Iterable[float]) -> list[dict[str, str | float | int]]:
    regions_dir = results_dir / "regions"
    if not regions_dir.exists():
        raise FileNotFoundError(f"Missing regions directory: {regions_dir}")

    series = {
        ("SuSiE", "with_prior"): lambda region: read_susie_pips(region / "susie_entropy.pip.tsv"),
        ("SuSiE", "without_prior"): lambda region: read_susie_pips(region / "susie_uniform.pip.tsv"),
        ("FINEMAP", "with_prior"): lambda region: read_finemap_pips(region / "finemap_entropy" / "region.snp"),
        ("FINEMAP", "without_prior"): lambda region: read_finemap_pips(region / "finemap_uniform" / "region.snp"),
    }

    all_pips: dict[tuple[str, str], list[float]] = {key: [] for key in series}
    region_counts: dict[tuple[str, str], int] = {key: 0 for key in series}
    for region in sorted(path for path in regions_dir.iterdir() if path.is_dir()):
        for key, reader in series.items():
            path_pips = reader(region)
            if path_pips:
                all_pips[key].extend(path_pips)
                region_counts[key] += 1

    rows: list[dict[str, str | float | int]] = []
    for threshold in thresholds:
        for (engine, prior_status), pips in all_pips.items():
            rows.append(
                {
                    "engine": engine,
                    "prior_status": prior_status,
                    "threshold": threshold,
                    "n_regions": region_counts[(engine, prior_status)],
                    "n_variants": len(pips),
                    "n_high_pip": sum(pip >= threshold for pip in pips),
                }
            )
    return rows


def read_susie_pips(path: Path) -> list[float]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return [float(row["pip"]) for row in csv.DictReader(handle, delimiter="\t") if row.get("pip")]


def read_finemap_pips(path: Path) -> list[float]:
    if not path.exists():
        return []
    pips: list[float] = []
    with path.open(newline="") as handle:
        header = handle.readline().split()
        if "prob" not in header:
            return []
        prob_idx = header.index("prob")
        for line in handle:
            parts = line.split()
            if len(parts) > prob_idx:
                pips.append(float(parts[prob_idx]))
    return pips


def write_summary(records: list[dict[str, str | float | int]], output_path: Path) -> None:
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["engine", "prior_status", "threshold", "n_regions", "n_variants", "n_high_pip"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(records)


def plot_counts(records: list[dict[str, str | float | int]], output_path: Path) -> None:
    engines = ["SuSiE", "FINEMAP"]
    labels = {"with_prior": "With entropy prior", "without_prior": "Uniform prior"}
    colors = {"with_prior": "tab:blue", "without_prior": "tab:orange"}

    fig, axes = plt.subplots(1, len(engines), figsize=(11, 4), sharey=False)
    if len(engines) == 1:
        axes = [axes]

    for ax, engine in zip(axes, engines, strict=True):
        for prior_status in ("with_prior", "without_prior"):
            rows = [
                row
                for row in records
                if row["engine"] == engine and row["prior_status"] == prior_status
            ]
            rows.sort(key=lambda row: float(row["threshold"]))
            ax.plot(
                [float(row["threshold"]) for row in rows],
                [int(row["n_high_pip"]) for row in rows],
                marker="o",
                linewidth=1.8,
                markersize=3,
                label=labels[prior_status],
                color=colors[prior_status],
            )
        ax.set_title(engine)
        ax.set_xlabel("PIP threshold")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Variants with PIP >= threshold")
    axes[-1].legend(frameon=False)
    fig.suptitle("High-PIP Counts With vs Without Entropy Priors")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()

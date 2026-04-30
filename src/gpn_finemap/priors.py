"""Convert entropy annotations into SuSiE/FINEMAP prior inputs."""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

PRIOR_METHODS = ("softmax", "rank", "minmax")
MISSING_POLICIES = ("median", "uniform", "least_constrained")


def add_entropy_prior_columns(
    frame: pl.DataFrame,
    *,
    constrained_direction: str = "low",
    prior_method: str = "softmax",
    temperature: float = 1.0,
    prior_floor: float = 1e-6,
    missing_policy: str = "median",
    finemap_expected_causal_per_region: float = 1.0,
) -> pl.DataFrame:
    """Add SuSiE and FINEMAP prior columns derived from entropy.

    SuSiE expects ``prior_weights`` to sum to one within each effect. We write
    region-normalized weights. FINEMAP's SNP prior is represented as per-SNP
    causal probabilities, so the same normalized weights are scaled to the
    requested expected number of causal variants per region.
    """

    _validate_options(constrained_direction, prior_method, temperature, prior_floor, missing_policy)
    if finemap_expected_causal_per_region <= 0:
        raise ValueError("finemap_expected_causal_per_region must be > 0")

    id_frame = _with_variant_id(frame)
    if "region" not in id_frame.columns:
        raise ValueError("Input must contain a 'region' column")

    logger.info("Adding entropy-derived prior columns for %d rows", id_frame.height)
    chunks: list[pl.DataFrame] = []
    group_columns = ["method", "region"] if "method" in id_frame.columns else ["region"]
    for key, group in id_frame.partition_by(group_columns, as_dict=True, maintain_order=True).items():
        logger.info("Computing priors for %s (%d variants)", key, group.height)
        chunks.append(
            _add_group_priors(
                group,
                constrained_direction=constrained_direction,
                prior_method=prior_method,
                temperature=temperature,
                prior_floor=prior_floor,
                missing_policy=missing_policy,
                finemap_expected_causal_per_region=finemap_expected_causal_per_region,
            )
        )

    result = pl.concat(chunks, how="vertical_relaxed") if chunks else id_frame
    logger.info("Added prior columns to %d rows", result.height)
    return result


def write_prior_outputs(
    priors: pl.DataFrame,
    output_dir: Path,
    *,
    include_templates: bool = True,
) -> None:
    """Write combined and per-region prior files for SuSiE and FINEMAP."""

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Writing combined prior tables to %s", output_dir)
    priors.write_parquet(output_dir / "entropy_priors.parquet")
    priors.write_csv(output_dir / "entropy_priors.tsv", separator="\t")

    susie_dir = output_dir / "susie"
    finemap_dir = output_dir / "finemap"
    susie_dir.mkdir(parents=True, exist_ok=True)
    finemap_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    group_columns = ["method", "region"] if "method" in priors.columns else ["region"]
    for key, group in priors.partition_by(group_columns, as_dict=True, maintain_order=True).items():
        method, region = key if isinstance(key, tuple) else ("ALL", key)
        stem = _safe_stem(f"{method}_{region}")
        susie_path = susie_dir / f"{stem}.prior_weights.tsv"
        finemap_path = finemap_dir / f"{stem}.prior.z"

        _write_susie_prior(group, susie_path)
        _write_finemap_prior_z(group, finemap_path)
        manifest_rows.append(
            {
                "method": method,
                "region": region,
                "n_variants": group.height,
                "susie_prior_path": str(susie_path),
                "finemap_prior_z_path": str(finemap_path),
            }
        )

    manifest = pl.DataFrame(manifest_rows)
    manifest.write_csv(output_dir / "prior_manifest.tsv", separator="\t")
    if include_templates:
        _write_susie_template(output_dir / "run_susie_with_entropy_priors.R")
        _write_finemap_template(output_dir / "finemap_entropy_prior.commands.txt", manifest)


def _add_group_priors(
    group: pl.DataFrame,
    *,
    constrained_direction: str,
    prior_method: str,
    temperature: float,
    prior_floor: float,
    missing_policy: str,
    finemap_expected_causal_per_region: float,
) -> pl.DataFrame:
    entropy = group.get_column("entropy_calibrated").to_numpy().astype(float)
    scores = _entropy_to_scores(entropy, constrained_direction, missing_policy)
    raw = _scores_to_positive_weights(scores, prior_method, temperature)
    raw = raw + prior_floor
    raw_sum = float(raw.sum())
    if not math.isfinite(raw_sum) or raw_sum <= 0:
        raw = np.ones_like(raw)
        raw_sum = float(raw.sum())

    susie_weights = raw / raw_sum
    finemap_prob = np.minimum(susie_weights * finemap_expected_causal_per_region, 1.0 - 1e-12)
    return group.with_columns(
        pl.Series("entropy_prior_score", scores),
        pl.Series("susie_prior_weight", susie_weights),
        pl.Series("finemap_prior_probability", finemap_prob),
        pl.Series("SNPVAR", finemap_prob),
    )


def _entropy_to_scores(entropy: np.ndarray, constrained_direction: str, missing_policy: str) -> np.ndarray:
    finite = np.isfinite(entropy)
    if not finite.any():
        return np.zeros_like(entropy, dtype=float)

    filled = entropy.copy()
    if missing_policy == "median":
        fill_value = float(np.nanmedian(entropy[finite]))
    elif missing_policy == "uniform":
        fill_value = float(np.nanmean(entropy[finite]))
    else:
        fill_value = float(np.nanmax(entropy[finite]) if constrained_direction == "low" else np.nanmin(entropy[finite]))
    filled[~finite] = fill_value

    return -filled if constrained_direction == "low" else filled


def _scores_to_positive_weights(scores: np.ndarray, prior_method: str, temperature: float) -> np.ndarray:
    if prior_method == "softmax":
        scaled = scores / temperature
        scaled = scaled - np.nanmax(scaled)
        return np.exp(scaled)

    order = scores.argsort(kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    if prior_method == "rank":
        return ranks

    score_min = float(np.nanmin(scores))
    score_max = float(np.nanmax(scores))
    if score_max == score_min:
        return np.ones_like(scores, dtype=float)
    return (scores - score_min) / (score_max - score_min)


def _with_variant_id(frame: pl.DataFrame) -> pl.DataFrame:
    if "variant_id" in frame.columns:
        return frame
    if "v" in frame.columns:
        return frame.with_columns(pl.col("v").cast(pl.Utf8).alias("variant_id"))
    if "rsid" in frame.columns:
        return frame.with_columns(pl.col("rsid").cast(pl.Utf8).alias("variant_id"))
    required = {"chrom", "pos", "ref", "alt"}
    if required.issubset(set(frame.columns)):
        return frame.with_columns(
            pl.concat_str(["chrom", "pos", "ref", "alt"], separator=":").alias("variant_id")
        )
    raise ValueError("Input must contain one of: variant_id, v, rsid, or chrom/pos/ref/alt")


def _write_susie_prior(group: pl.DataFrame, path: Path) -> None:
    group.select("variant_id", "susie_prior_weight").write_csv(path, separator="\t")


def _write_finemap_prior_z(group: pl.DataFrame, path: Path) -> None:
    id_column = "rsid" if "rsid" in group.columns else "variant_id"
    columns = [
        pl.col(id_column).cast(pl.Utf8).alias("rsid"),
        pl.col("chrom").alias("chromosome"),
        pl.col("pos").alias("position"),
        pl.col("ref").alias("allele1"),
        pl.col("alt").alias("allele2"),
    ]
    for optional in ("maf", "beta", "se"):
        if optional in group.columns:
            columns.append(pl.col(optional))
    columns.append(pl.col("finemap_prior_probability").alias("prob"))
    group.select(columns).write_csv(path, separator=" ")


def _write_susie_template(path: Path) -> None:
    path.write_text(
        """# Template: use entropy priors with susieR.
# Fill in z_scores and R from your locus-specific summary stats and LD matrix.
library(susieR)

prior <- read.delim("susie/REGION.prior_weights.tsv")
variant_ids <- prior$variant_id
prior_weights <- prior$susie_prior_weight

fit <- susie_rss(
  z = z_scores[variant_ids],
  R = ld_matrix[variant_ids, variant_ids],
  prior_weights = prior_weights,
  L = 10
)
""",
        encoding="utf-8",
    )


def _write_finemap_template(path: Path, manifest: pl.DataFrame) -> None:
    lines = [
        "# Template FINEMAP commands. Fill in each region .master/.ld path.",
        "# Use the generated .prior.z files as the FINEMAP .z input.",
        "# They include a prob column with entropy-derived prior causal probabilities.",
    ]
    for row in manifest.iter_rows(named=True):
        lines.append(
            f"# {row['region']}: set z={row['finemap_prior_z_path']} in REGION.master, then run:"
            " finemap --sss --in-files REGION.master --prior-snps"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _safe_stem(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_")[:180] or "region"


def _validate_options(
    constrained_direction: str,
    prior_method: str,
    temperature: float,
    prior_floor: float,
    missing_policy: str,
) -> None:
    if constrained_direction not in {"low", "high"}:
        raise ValueError("constrained_direction must be 'low' or 'high'")
    if prior_method not in PRIOR_METHODS:
        raise ValueError(f"prior_method must be one of {', '.join(PRIOR_METHODS)}")
    if missing_policy not in MISSING_POLICIES:
        raise ValueError(f"missing_policy must be one of {', '.join(MISSING_POLICIES)}")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if prior_floor < 0:
        raise ValueError("prior_floor must be >= 0")

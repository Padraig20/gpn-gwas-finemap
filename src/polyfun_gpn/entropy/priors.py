"""Map per-SNP entropy values to PolyFun ``SNPVAR`` priors.

Two functional prior modes are supported (plus the trivial ``none`` mode that
omits the SNPVAR column and lets PolyFun run non-functionally):

* ``entropy`` — *global surprise* against the genome-wide background::

      w_i      = -log( f_bg(e_i) + epsilon )
      snpvar_i = exp( tau * w_i )

  Variants are scored relative to how rare their entropy value is genome-wide.

* ``entropy_raw`` — *raw per-locus entropy*, negated so low entropy gets a
  higher prior::

      snpvar_i = exp( -tau * e_i )

  No background distribution is consulted; this is a "local" prior that uses
  only the entropy values themselves. ``epsilon`` is unused in this mode.

PolyFun normalizes ``SNPVAR`` within a locus internally, so we only need
positive values that are proportional to the desired per-SNP heritability.

For any variant without a usable entropy lookup we fall back to the *median*
SNPVAR of the variants in the same locus that *did* get one, and tag the row
with ``prior_source = "median_fallback"``. Hits get ``prior_source = "entropy"``
regardless of which mode produced them (the audit column tracks the
provenance of the per-SNP value, not the prior mode).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ..config import PriorParams
from .background import density_lookup


def surprise(
    entropy_values: np.ndarray,
    density: np.ndarray,
    edges: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    f = density_lookup(entropy_values, density, edges)
    return -np.log(f + epsilon)


def entropy_snpvar(
    entropy_values: np.ndarray,
    density: np.ndarray,
    edges: np.ndarray,
    params: PriorParams,
) -> np.ndarray:
    """Compute proportional SNPVAR weights from raw entropy values.

    NaN entropies pass through (caller fills with locus median below).
    """
    out = np.full_like(entropy_values, np.nan, dtype=np.float64)
    finite = np.isfinite(entropy_values)
    if not finite.any():
        return out
    w = surprise(
        entropy_values[finite].astype(np.float64), density, edges, params.epsilon
    )
    out[finite] = np.exp(params.tau * w)
    return out


def entropy_raw_snpvar(
    entropy_values: np.ndarray,
    params: PriorParams,
) -> np.ndarray:
    """SNPVAR weights from the raw per-SNP entropy, negated.

    ``snpvar_i = exp(-tau * e_i)`` — monotone-decreasing in entropy, so low
    entropy positions (more conserved / less variable) receive a higher prior.
    No background distribution is needed; this is a purely local transform.
    NaN entropies pass through (caller fills with locus median below).
    """
    out = np.full_like(entropy_values, np.nan, dtype=np.float64)
    finite = np.isfinite(entropy_values)
    if not finite.any():
        return out
    e = entropy_values[finite].astype(np.float64)
    out[finite] = np.exp(-params.tau * e)
    return out


def _attach_with_median_fallback(
    df: pl.DataFrame, raw: np.ndarray
) -> pl.DataFrame:
    """Common tail: split into hits/misses, fill misses with locus median.

    ``raw`` is the per-SNP SNPVAR (NaN where the entropy lookup missed).
    """
    finite_mask = np.isfinite(raw) & (raw > 0)
    if not finite_mask.any():
        return df.with_columns(
            pl.lit(1.0).alias("SNPVAR"),
            pl.lit("median_fallback").alias("prior_source"),
        )
    median = float(np.median(raw[finite_mask]))
    snpvar = np.where(finite_mask, raw, median)
    source = np.where(finite_mask, "entropy", "median_fallback")
    return df.with_columns(
        pl.Series("SNPVAR", snpvar, dtype=pl.Float64),
        pl.Series("prior_source", source, dtype=pl.Utf8),
    )


def attach_priors(
    df: pl.DataFrame,
    entropy_values: np.ndarray,
    density: np.ndarray,
    edges: np.ndarray,
    params: PriorParams,
    *,
    prior_mode: str,
) -> pl.DataFrame:
    """Add ``SNPVAR`` and ``prior_source`` columns based on ``prior_mode``.

    Modes:
      - ``none``:        no SNPVAR column (PolyFun runs non-functionally).
      - ``entropy``:     surprise-against-background weights with locus-median
                         fallback for variants missing an entropy lookup.
      - ``entropy_raw``: ``exp(-tau * e)`` from the raw per-SNP entropy, with
                         the same locus-median fallback for missing values.
                         ``density`` / ``edges`` are unused in this mode.

    ``entropy_values`` must be aligned row-for-row with ``df``.
    """
    if prior_mode == "none":
        return df
    if prior_mode not in ("entropy", "entropy_raw"):
        raise ValueError(f"Unknown prior_mode: {prior_mode!r}")

    n = df.height
    if entropy_values.size != n:
        raise ValueError(
            f"entropy_values length {entropy_values.size} != df height {n}"
        )

    if prior_mode == "entropy":
        raw = entropy_snpvar(entropy_values, density, edges, params)
    else:
        raw = entropy_raw_snpvar(entropy_values, params)
    return _attach_with_median_fallback(df, raw)

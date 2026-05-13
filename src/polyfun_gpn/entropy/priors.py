"""Map per-SNP entropy values to PolyFun ``SNPVAR`` priors.

For ``--prior entropy`` we compute:

    w_i      = -log( f_bg(e_i) + epsilon )      # surprise vs background
    snpvar_i = exp( tau * w_i )

For ``--prior none`` no SNPVAR column is produced and PolyFun runs in its
non-functional mode (uniform causal prior baked into FINEMAP).

PolyFun normalizes ``SNPVAR`` within a locus internally, so we only need
positive values that are proportional to the desired per-SNP heritability.

For any variant without a usable entropy lookup we fall back to the *median*
SNPVAR of the variants in the same locus that *did* get one, and tag the row
with ``prior_source = "median_fallback"``. Hits get ``prior_source = "entropy"``.
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
      - ``none``:    no SNPVAR column (PolyFun runs non-functionally).
      - ``entropy``: surprise-against-background weights with locus-median
                     fallback for variants missing an entropy lookup.

    ``entropy_values`` must be aligned row-for-row with ``df``.
    """
    if prior_mode == "none":
        return df
    if prior_mode != "entropy":
        raise ValueError(f"Unknown prior_mode: {prior_mode!r}")

    n = df.height
    if entropy_values.size != n:
        raise ValueError(
            f"entropy_values length {entropy_values.size} != df height {n}"
        )

    raw = entropy_snpvar(entropy_values, density, edges, params)
    finite_mask = np.isfinite(raw) & (raw > 0)
    if not finite_mask.any():
        # All-missing → uniform 1.0 with source ``median_fallback`` so PolyFun
        # is still invoked in functionally-informed mode for the locus.
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

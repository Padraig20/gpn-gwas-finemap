"""Look up per-position entropy for a set of variants.

The entropy parquets are *huge* (~16 GB total), so we never load them in
full. The strategy is:

1. Group input variants by chromosome.
2. For each chrom, ``scan_parquet`` the entropy file and inner-join on
   ``pos`` (Polars streams the join). The output keeps the original input
   ordering via a ``__row_idx__`` column.
3. Concatenate per-chrom results, fill missing with NaN.

If ``cfg.builds.entropy != cfg.builds.gwas``, callers should pass
already-lifted coordinates (see ``coords.lift_positions``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl

from ..coords import normalize_chrom


def lookup_entropy(
    entropy_dir: Path,
    chroms: Iterable[str],
    positions: Iterable[int],
) -> np.ndarray:
    """Return a float32 array of entropy values aligned with the inputs.

    Missing variants (no parquet for chrom, or position not in parquet) get
    NaN so callers can apply a fallback.
    """
    chroms = [normalize_chrom(c) for c in chroms]
    positions = np.asarray(positions, dtype=np.int64)
    n = positions.size
    out = np.full(n, np.nan, dtype=np.float32)
    if n == 0:
        return out

    df = pl.DataFrame(
        {
            "__row_idx__": np.arange(n, dtype=np.int64),
            "chrom": chroms,
            "pos": positions,
        }
    )

    for chrom, sub in df.group_by("chrom"):
        chrom_label = chrom[0] if isinstance(chrom, tuple) else chrom
        if not chrom_label:
            continue
        path = entropy_dir / f"entropy_chr{chrom_label}.parquet"
        if not path.exists():
            continue
        joined = (
            pl.scan_parquet(path)
            .select(["pos", "entropy_calibrated"])
            .join(sub.lazy().select(["__row_idx__", "pos"]), on="pos", how="inner")
            .collect(engine="streaming")
        )
        idx = joined["__row_idx__"].to_numpy()
        vals = joined["entropy_calibrated"].to_numpy().astype(np.float32, copy=False)
        out[idx] = vals
    return out

"""Coordinate utilities: chrom normalization and hg19<->hg38 liftover.

The entropy parquets store ``chrom`` as ``"1"..."22","X","Y"`` (no ``chr``
prefix); GWAS chromosomes come in as integers. UKB LD URLs use ``chr{N}_...``.
We normalize everywhere to a canonical no-prefix string so it's the only
representation that flows through Polars joins.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from pyliftover import LiftOver


CANONICAL_CHROMS: tuple[str, ...] = tuple(str(i) for i in range(1, 23)) + ("X", "Y")


def normalize_chrom(value: object) -> str:
    """Canonicalize a chromosome label to ``"1"..."22","X","Y"``."""
    if value is None:
        return ""
    s = str(value).strip()
    if s.lower().startswith("chr"):
        s = s[3:]
    if s in {"23", "x"}:
        s = "X"
    if s in {"24", "y"}:
        s = "Y"
    return s.upper() if s in {"x", "y"} else s


def _chain_filename(src: str, dst: str) -> str:
    src = src.lower()
    dst = dst.lower()
    return f"{src}To{dst[0].upper()}{dst[1:]}.over.chain.gz"


def load_liftover(reference_dir: Path, src: str, dst: str) -> LiftOver:
    """Construct a ``pyliftover.LiftOver`` from a local chain file."""
    chain = reference_dir / _chain_filename(src, dst)
    if not chain.exists():
        raise FileNotFoundError(
            f"Missing chain file at {chain}; run `polyfun-gpn setup` first."
        )
    return LiftOver(str(chain))


def lift_positions(
    lo: LiftOver,
    chroms: Sequence[str],
    positions: Iterable[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized liftover of (chrom, pos) pairs.

    Returns two parallel numpy arrays of the same length as the input:
        ``new_chrom`` (object dtype, "" for failed entries) and
        ``new_pos`` (int64, ``-1`` for failed entries).

    pyliftover expects chroms with the ``chr`` prefix; we add/strip it here.
    Coordinates are 0-based half-open in pyliftover; for SNV-style 1-based
    positions we convert in/out so callers stay in 1-based space.
    """
    positions = np.asarray(positions, dtype=np.int64)
    new_chrom = np.empty(len(positions), dtype=object)
    new_pos = np.full(len(positions), -1, dtype=np.int64)
    for i, (c, p) in enumerate(zip(chroms, positions)):
        c_norm = normalize_chrom(c)
        if not c_norm:
            new_chrom[i] = ""
            continue
        result = lo.convert_coordinate(f"chr{c_norm}", int(p) - 1)
        if not result:
            new_chrom[i] = ""
            continue
        out_chrom, out_pos, _strand, _score = result[0]
        new_chrom[i] = normalize_chrom(out_chrom)
        new_pos[i] = int(out_pos) + 1
    return new_chrom, new_pos

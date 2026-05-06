"""Build a genome-wide entropy background distribution.

We don't try to load all ~3B per-position values into memory. Instead we draw
``n_samples`` random rows in proportion to each chromosome's row count, using
parquet row-group metadata to bound how much we materialize at once.

The cache (``data/background/entropy_bg.npz``) stores a normalized histogram
density over a fixed support ``[entropy_min, entropy_max]`` plus the bin
edges. Per-SNP density lookup is constant-time via ``np.digitize``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from ..config import Config
from ..coords import CANONICAL_CHROMS


def _per_chrom_row_counts(entropy_dir: Path) -> dict[str, tuple[int, Path]]:
    counts: dict[str, tuple[int, Path]] = {}
    for chrom in CANONICAL_CHROMS:
        p = entropy_dir / f"entropy_chr{chrom}.parquet"
        if not p.exists():
            continue
        n = pq.ParquetFile(p).metadata.num_rows
        counts[chrom] = (n, p)
    if not counts:
        raise FileNotFoundError(
            f"No entropy parquets found under {entropy_dir}. "
            "Expected files like entropy_chr1.parquet."
        )
    return counts


def _allocate_samples_per_chrom(
    counts: dict[str, tuple[int, Path]], n_samples: int, rng: np.random.Generator
) -> dict[str, int]:
    total = sum(n for n, _ in counts.values())
    target = min(n_samples, total)
    weights = np.array([n for n, _ in counts.values()], dtype=np.float64)
    weights /= weights.sum()
    raw = weights * target
    floor = np.floor(raw).astype(np.int64)
    remainder = target - int(floor.sum())
    frac = raw - floor
    if remainder > 0:
        order = rng.permutation(np.argsort(-frac))[:remainder]
        floor[order] += 1
    return {chrom: int(floor[i]) for i, chrom in enumerate(counts.keys())}


def _sample_one_chromosome(
    path: Path, n_take: int, rng: np.random.Generator
) -> np.ndarray:
    if n_take <= 0:
        return np.empty(0, dtype=np.float32)
    pf = pq.ParquetFile(path)
    n_total = pf.metadata.num_rows
    if n_take >= n_total:
        return (
            pl.read_parquet(path, columns=["entropy_calibrated"])
            .to_series()
            .to_numpy()
            .astype(np.float32, copy=False)
        )
    rg_sizes = np.array(
        [pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups)],
        dtype=np.int64,
    )
    rg_starts = np.concatenate([[0], np.cumsum(rg_sizes)[:-1]])
    sampled_idx = np.sort(rng.choice(n_total, size=n_take, replace=False))
    rg_assign = np.searchsorted(rg_starts + rg_sizes - 1, sampled_idx)
    out: list[np.ndarray] = []
    for rg in np.unique(rg_assign):
        mask = rg_assign == rg
        local = sampled_idx[mask] - rg_starts[rg]
        rg_table = pf.read_row_group(rg, columns=["entropy_calibrated"])
        col = rg_table.column("entropy_calibrated").to_numpy()
        out.append(col[local].astype(np.float32, copy=False))
    return np.concatenate(out) if out else np.empty(0, dtype=np.float32)


def build_background(cfg: Config, *, overwrite: bool = False) -> Path:
    cache = cfg.paths.absolute("background")
    if cache.exists() and not overwrite:
        print(f"[build-bg] {cache} already exists; skipping (use --overwrite).")
        return cache

    entropy_dir = cfg.paths.absolute("entropy_dir")
    counts = _per_chrom_row_counts(entropy_dir)
    rng = np.random.default_rng(cfg.background.seed)
    alloc = _allocate_samples_per_chrom(counts, cfg.background.n_samples, rng)

    chunks: list[np.ndarray] = []
    total_target = sum(alloc.values())
    print(
        f"[build-bg] Sampling {total_target:,} positions over "
        f"{len(counts)} chromosomes."
    )
    for chrom, (n_chrom, path) in counts.items():
        n_take = alloc[chrom]
        print(
            f"[build-bg]   chr{chrom}: {n_take:,} of {n_chrom:,} "
            f"({100.0 * n_take / n_chrom:.3f}%)"
        )
        chunks.append(_sample_one_chromosome(path, n_take, rng))
    samples = np.concatenate(chunks)
    samples = samples[np.isfinite(samples)]

    n_bins = cfg.background.n_bins
    edges = np.linspace(cfg.background.entropy_min, cfg.background.entropy_max, n_bins + 1)
    hist, _ = np.histogram(samples, bins=edges, density=True)

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache,
        density=hist.astype(np.float64),
        edges=edges.astype(np.float64),
        n_samples=np.int64(samples.size),
        seed=np.int64(cfg.background.seed),
    )
    print(
        f"[build-bg] Cached density (n={samples.size:,}, bins={n_bins}, "
        f"support=[{edges[0]:.2f},{edges[-1]:.2f}]) -> {cache}"
    )
    return cache


def load_background(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(density, edges)`` as float64 numpy arrays."""
    cache = cfg.paths.absolute("background")
    if not cache.exists():
        raise FileNotFoundError(
            f"Background cache missing at {cache}; run `polyfun-gpn build-bg`."
        )
    data = np.load(cache)
    return data["density"], data["edges"]


def density_lookup(
    values: np.ndarray, density: np.ndarray, edges: np.ndarray
) -> np.ndarray:
    """Constant-time per-value density lookup via bin index."""
    idx = np.clip(np.digitize(values, edges) - 1, 0, density.size - 1)
    return density[idx]

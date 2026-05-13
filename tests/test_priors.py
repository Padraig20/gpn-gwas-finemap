"""Unit tests for surprise math and SNPVAR construction."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from polyfun_gpn.config import PriorParams
from polyfun_gpn.entropy.priors import attach_priors, entropy_snpvar, surprise


def _fake_density() -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, 3.0, 31)
    centers = 0.5 * (edges[:-1] + edges[1:])
    density = np.exp(-((centers - 1.5) ** 2) / 0.5)
    density /= density.sum() * (edges[1] - edges[0])
    return density, edges


def test_surprise_monotone_in_distance_from_peak() -> None:
    density, edges = _fake_density()
    near_peak = surprise(np.array([1.5]), density, edges, epsilon=1e-9)
    far_tail = surprise(np.array([0.1]), density, edges, epsilon=1e-9)
    assert far_tail > near_peak


def test_entropy_snpvar_propagates_nans() -> None:
    density, edges = _fake_density()
    vals = np.array([1.5, np.nan, 0.5], dtype=np.float32)
    out = entropy_snpvar(vals, density, edges, PriorParams())
    assert np.isfinite(out[0])
    assert np.isnan(out[1])
    assert np.isfinite(out[2])
    assert out[2] > out[0]


def test_attach_priors_none_mode_omits_snpvar() -> None:
    df = pl.DataFrame({"BP": [1, 2, 3]})
    out = attach_priors(
        df,
        np.array([np.nan] * 3),
        np.empty(0),
        np.empty(0),
        PriorParams(),
        prior_mode="none",
    )
    assert "SNPVAR" not in out.columns


def test_attach_priors_entropy_mode_uses_median_fallback() -> None:
    density, edges = _fake_density()
    df = pl.DataFrame({"BP": [1, 2, 3, 4]})
    vals = np.array([1.5, np.nan, 0.5, np.nan], dtype=np.float32)
    out = attach_priors(df, vals, density, edges, PriorParams(), prior_mode="entropy")
    snpvar = out["SNPVAR"].to_numpy()
    sources = out["prior_source"].to_list()
    assert sources == ["entropy", "median_fallback", "entropy", "median_fallback"]
    finite = snpvar[np.array([0, 2])]
    assert np.allclose(snpvar[1], np.median(finite))
    assert np.allclose(snpvar[3], np.median(finite))


def test_attach_priors_entropy_all_missing_falls_back_to_uniform() -> None:
    density, edges = _fake_density()
    df = pl.DataFrame({"BP": [1, 2, 3]})
    vals = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    out = attach_priors(df, vals, density, edges, PriorParams(), prior_mode="entropy")
    assert (out["SNPVAR"].to_numpy() == 1.0).all()
    assert out["prior_source"].to_list() == ["median_fallback"] * 3


def test_attach_priors_unknown_mode_raises() -> None:
    df = pl.DataFrame({"BP": [1]})
    with pytest.raises(ValueError):
        attach_priors(
            df,
            np.array([1.5], dtype=np.float32),
            np.empty(0),
            np.empty(0),
            PriorParams(),
            prior_mode="something_else",
        )


def test_attach_priors_entropy_length_mismatch_raises() -> None:
    df = pl.DataFrame({"BP": [1, 2, 3]})
    with pytest.raises(ValueError):
        attach_priors(
            df,
            np.array([1.5, 0.5], dtype=np.float32),
            np.empty(0),
            np.empty(0),
            PriorParams(),
            prior_mode="entropy",
        )

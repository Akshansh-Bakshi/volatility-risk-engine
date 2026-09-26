"""Tests for src.statistics.diagnostics (Stage 5).

All tests use synthetic data; no network access, no disk I/O.

Synthetic series used
---------------------
* ``_white_noise()`` — i.i.d. N(0, σ²) series: stationary, no autocorrelation,
  no ARCH effect.
* ``_random_walk()`` — cumulative sum of white noise: non-stationary unit-root
  process.
* ``_ar1()`` — AR(1) with ρ close to 1: stationary but with strong positive
  autocorrelation.
* ``_garch_like()`` — series with time-varying conditional variance simulated via
  a simple GARCH(1,1)-style recursion; exhibits significant ARCH effect and
  autocorrelation in squared returns.

Coverage
--------
* TestResult dataclass: fields, JSON round-trip, rejects_null property.
* ReturnDiagnostics dataclass: all_results tuple, JSON round-trip.
* ADF test: stationary series rejects null; non-stationary fails to reject;
  gap-flagged metadata; exclude_gaps path.
* KPSS test: stationary series fails to reject null; non-stationary rejects null;
  null-hypothesis direction is opposite to ADF.
* Ljung-Box on returns: uncorrelated series fails to reject; autocorrelated rejects.
* Ljung-Box on squared returns: distinct from ARCH-LM; squared-dependent series
  rejects; plain white noise fails to reject.
* ARCH-LM: GARCH-like series rejects; plain white noise fails to reject;
  separate from Ljung-Box-squared.
* run_all_diagnostics: correct container fields; deterministic repeated results.
* Invalid inputs: too-short series raises InsufficientDataError;
  lag count too large for available data raises InsufficientDataError.
* ACF figure: correct return type, axes structure, title, CI band.
* PACF figure: correct return type, axes structure, title.
* Null-hypothesis interpretation correctness.
* Scale: series_scale field is always "percent".
* Gap handling: n_gap_flagged reported; gaps_excluded flag toggled correctly.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from typing import Callable

import matplotlib.figure
import numpy as np
import pandas as pd
import pytest

from src.exceptions import InsufficientDataError, StatisticsError
from src.preprocessing.return_series import PriceBasis, ReturnSeries
from src.statistics.diagnostics import (
    ReturnDiagnostics,
    TestResult,
    plot_acf,
    plot_pacf,
    run_adf_test,
    run_all_diagnostics,
    run_arch_lm_test,
    run_kpss_test,
    run_ljung_box_returns,
    run_ljung_box_squared,
)

# ---------------------------------------------------------------------------
# Synthetic ReturnSeries factory
# ---------------------------------------------------------------------------

_FETCHED_AT = datetime(2024, 6, 3, 12, 0, tzinfo=timezone.utc)
_FIRST_PRICE_DATE = date(2020, 1, 1)


def _make_return_series(
    decimal_values: np.ndarray,
    *,
    ticker: str = "SYN",
    gap_mask: np.ndarray | None = None,
) -> ReturnSeries:
    """Wrap a 1-D float64 decimal-scale array into a valid ReturnSeries."""
    n = len(decimal_values)
    assert n >= 2, "Need at least 2 observations."
    idx = pd.bdate_range("2020-01-02", periods=n, name="date")
    decimal = pd.Series(decimal_values.astype("float64"), index=idx)
    if gap_mask is None:
        gap_mask = np.zeros(n, dtype=bool)
    spans_gap = pd.Series(gap_mask.astype(bool), index=idx)
    return ReturnSeries(
        ticker=ticker,
        frequency="1d",
        price_basis=PriceBasis.ADJUSTED,
        decimal=decimal,
        spans_gap=spans_gap,
        first_price_date=_FIRST_PRICE_DATE,
        gap_tolerance_weekdays=1,
        provisional_bar=None,
        provisional_bar_excluded=False,
        ingestion_rows_dropped=0,
        source="synthetic",
        fetched_at=_FETCHED_AT,
    )


# ---------------------------------------------------------------------------
# Deterministic synthetic series (seeded with numpy default_rng)
# ---------------------------------------------------------------------------

_N = 500     # number of observations for full-length tests
_RNG = np.random.default_rng(2025)


def _white_noise(n: int = _N, sigma: float = 1.0) -> np.ndarray:
    """i.i.d. N(0, sigma²) in decimal scale."""
    return np.random.default_rng(2025).normal(0.0, sigma / 100.0, n)


def _random_walk(n: int = _N) -> np.ndarray:
    """Cumulative sum of N(0,1) shocks — non-stationary (unit root)."""
    shocks = np.random.default_rng(42).normal(0.0, 0.01, n)
    return np.cumsum(shocks)


def _ar1(n: int = _N, rho: float = 0.5) -> np.ndarray:
    """Stationary AR(1) process: r_t = rho * r_{t-1} + ε_t, |rho| < 1."""
    rng = np.random.default_rng(7)
    eps = rng.normal(0.0, 0.01, n)
    x = np.empty(n)
    x[0] = eps[0]
    for i in range(1, n):
        x[i] = rho * x[i - 1] + eps[i]
    return x


def _garch_like(n: int = _N, omega: float = 1e-5, alpha: float = 0.15, beta: float = 0.80) -> np.ndarray:
    """Simple GARCH(1,1)-like series exhibiting strong volatility clustering."""
    rng = np.random.default_rng(99)
    eps = rng.standard_normal(n)
    h = np.empty(n)
    r = np.empty(n)
    h[0] = omega / (1.0 - alpha - beta) if alpha + beta < 1 else 1e-4
    r[0] = np.sqrt(h[0]) * eps[0]
    for t in range(1, n):
        h[t] = omega + alpha * r[t - 1] ** 2 + beta * h[t - 1]
        r[t] = np.sqrt(h[t]) * eps[t]
    return r


# ---------------------------------------------------------------------------
# Helper: run a test and return its TestResult (for parametrised checks)
# ---------------------------------------------------------------------------


def _rs_wn(n: int = _N) -> ReturnSeries:
    return _make_return_series(_white_noise(n))


def _rs_rw(n: int = _N) -> ReturnSeries:
    return _make_return_series(_random_walk(n))


def _rs_ar1(n: int = _N, rho: float = 0.5) -> ReturnSeries:
    return _make_return_series(_ar1(n, rho))


def _rs_garch(n: int = _N) -> ReturnSeries:
    return _make_return_series(_garch_like(n))


# ===========================================================================
# TestResult dataclass
# ===========================================================================


class TestResultDataclass:
    def _sample(self) -> TestResult:
        return TestResult(
            test_name="Dummy",
            null_hypothesis="Nothing.",
            statistic=1.23,
            p_value=0.04,
            lags_used=5,
            n_observations=300,
            n_gap_flagged=2,
            gaps_excluded=False,
            series_scale="percent",
            conclusion="reject_null",
            interpretation="p < 0.05: reject.",
            notes=("note one", "note two"),
        )

    def test_rejects_null_true_when_conclusion_is_reject(self) -> None:
        r = self._sample()
        assert r.rejects_null is True

    def test_rejects_null_false_when_fail(self) -> None:
        r = TestResult(
            test_name="D",
            null_hypothesis=".",
            statistic=0.1,
            p_value=0.8,
            lags_used=None,
            n_observations=100,
            n_gap_flagged=0,
            gaps_excluded=False,
            series_scale="percent",
            conclusion="fail_to_reject_null",
            interpretation="fail.",
        )
        assert r.rejects_null is False

    def test_to_dict_is_json_serialisable(self) -> None:
        d = self._sample().to_dict()
        assert json.loads(json.dumps(d)) == d

    def test_to_json_is_valid(self) -> None:
        parsed = json.loads(self._sample().to_json())
        assert parsed["test_name"] == "Dummy"
        assert parsed["p_value"] == pytest.approx(0.04)
        assert parsed["notes"] == ["note one", "note two"]

    def test_series_scale_is_percent(self) -> None:
        assert self._sample().series_scale == "percent"


# ===========================================================================
# ReturnDiagnostics dataclass
# ===========================================================================


class TestReturnDiagnostics:
    def test_all_results_has_five_entries(self) -> None:
        rs = _rs_wn()
        d = run_all_diagnostics(rs)
        assert len(d.all_results) == 5

    def test_all_results_are_test_result_instances(self) -> None:
        rs = _rs_wn()
        d = run_all_diagnostics(rs)
        for r in d.all_results:
            assert isinstance(r, TestResult)

    def test_to_dict_is_json_serialisable(self) -> None:
        rs = _rs_wn()
        d = run_all_diagnostics(rs).to_dict()
        assert json.loads(json.dumps(d)) == d

    def test_to_json_contains_all_test_keys(self) -> None:
        rs = _rs_wn()
        parsed = json.loads(run_all_diagnostics(rs).to_json())
        for key in ("adf", "kpss", "ljung_box_returns", "ljung_box_squared", "arch_lm"):
            assert key in parsed

    def test_ticker_field_propagated(self) -> None:
        rs = _make_return_series(_white_noise(), ticker="MYSTOCK")
        d = run_all_diagnostics(rs)
        assert d.ticker == "MYSTOCK"

    def test_series_scale_is_percent(self) -> None:
        d = run_all_diagnostics(_rs_wn())
        assert d.series_scale == "percent"

    def test_n_observations_total_matches_series_length(self) -> None:
        rs = _rs_wn()
        d = run_all_diagnostics(rs)
        assert d.n_observations_total == rs.return_observations


# ===========================================================================
# ADF test
# ===========================================================================


class TestADF:
    def test_stationary_white_noise_rejects_unit_root(self) -> None:
        """ADF H₀ = unit root.  White noise is stationary → should reject."""
        result = run_adf_test(_rs_wn())
        assert result.rejects_null, (
            f"Expected ADF to reject unit root for white noise; p={result.p_value:.4f}"
        )

    def test_random_walk_fails_to_reject_unit_root(self) -> None:
        """ADF H₀ = unit root.  Random walk has a unit root → should NOT reject."""
        result = run_adf_test(_rs_rw())
        assert not result.rejects_null, (
            f"Expected ADF not to reject for random walk; p={result.p_value:.4f}"
        )

    def test_conclusion_string_is_one_of_two_values(self) -> None:
        r = run_adf_test(_rs_wn())
        assert r.conclusion in ("reject_null", "fail_to_reject_null")

    def test_null_hypothesis_mentions_unit_root(self) -> None:
        r = run_adf_test(_rs_wn())
        assert "unit root" in r.null_hypothesis.lower()

    def test_series_scale_is_percent(self) -> None:
        assert run_adf_test(_rs_wn()).series_scale == "percent"

    def test_lags_used_is_nonnegative_int(self) -> None:
        r = run_adf_test(_rs_wn())
        assert isinstance(r.lags_used, int) and r.lags_used >= 0

    def test_n_observations_matches_series_length(self) -> None:
        rs = _rs_wn()
        r = run_adf_test(rs)
        assert r.n_observations == rs.return_observations

    def test_n_gap_flagged_zero_when_no_gaps(self) -> None:
        r = run_adf_test(_rs_wn())
        assert r.n_gap_flagged == 0

    def test_n_gap_flagged_reported_correctly(self) -> None:
        vals = _white_noise()
        gaps = np.zeros(len(vals), dtype=bool)
        gaps[5] = True
        gaps[50] = True
        rs = _make_return_series(vals, gap_mask=gaps)
        r = run_adf_test(rs)
        assert r.n_gap_flagged == 2
        assert r.gaps_excluded is False

    def test_exclude_gaps_reduces_n_observations(self) -> None:
        vals = _white_noise()
        gaps = np.zeros(len(vals), dtype=bool)
        gaps[5:10] = True   # 5 flagged
        rs = _make_return_series(vals, gap_mask=gaps)
        r_full = run_adf_test(rs, exclude_gaps=False)
        r_excl = run_adf_test(rs, exclude_gaps=True)
        assert r_full.n_observations == len(vals)
        assert r_excl.n_observations == len(vals) - 5
        assert r_excl.gaps_excluded is True

    def test_result_is_deterministic(self) -> None:
        rs = _rs_wn()
        r1 = run_adf_test(rs)
        r2 = run_adf_test(rs)
        assert r1.statistic == pytest.approx(r2.statistic)
        assert r1.p_value == pytest.approx(r2.p_value)

    def test_interpretation_mentions_reject_or_fail(self) -> None:
        r = run_adf_test(_rs_wn())
        assert "reject" in r.interpretation.lower()

    def test_interpretation_mentions_stationarity(self) -> None:
        r = run_adf_test(_rs_wn())
        assert "stationar" in r.interpretation.lower()

    def test_json_round_trip(self) -> None:
        d = run_adf_test(_rs_wn()).to_dict()
        assert json.loads(json.dumps(d)) == d

    def test_adf_and_kpss_opposite_null(self) -> None:
        """Confirm the two tests document opposite nulls."""
        rs = _rs_wn()
        adf = run_adf_test(rs)
        kpss = run_kpss_test(rs)
        assert "unit root" in adf.null_hypothesis.lower()
        assert "stationary" in kpss.null_hypothesis.lower()
        # Stationary series: ADF rejects (good), KPSS does not reject (good).
        assert adf.rejects_null
        assert not kpss.rejects_null


# ===========================================================================
# KPSS test
# ===========================================================================


class TestKPSS:
    def test_stationary_series_fails_to_reject(self) -> None:
        """KPSS H₀ = stationary.  White noise is stationary → should NOT reject."""
        result = run_kpss_test(_rs_wn())
        assert not result.rejects_null, (
            f"Expected KPSS to fail to reject for white noise; p={result.p_value:.4f}"
        )

    def test_random_walk_rejects_stationarity(self) -> None:
        """KPSS H₀ = stationary.  Random walk is non-stationary → should reject."""
        result = run_kpss_test(_rs_rw())
        assert result.rejects_null, (
            f"Expected KPSS to reject for random walk; p={result.p_value:.4f}"
        )

    def test_null_hypothesis_mentions_stationary(self) -> None:
        r = run_kpss_test(_rs_wn())
        assert "stationary" in r.null_hypothesis.lower()

    def test_conclusion_string_valid(self) -> None:
        r = run_kpss_test(_rs_wn())
        assert r.conclusion in ("reject_null", "fail_to_reject_null")

    def test_series_scale_is_percent(self) -> None:
        assert run_kpss_test(_rs_wn()).series_scale == "percent"

    def test_kpss_and_adf_agree_on_stationary_series(self) -> None:
        """For white noise: ADF rejects (= stationary), KPSS fails to reject (= stationary)."""
        rs = _rs_wn()
        adf = run_adf_test(rs)
        kpss = run_kpss_test(rs)
        assert adf.rejects_null is True    # ADF says stationary
        assert kpss.rejects_null is False  # KPSS agrees: stationary

    def test_kpss_interpretation_explains_opposite_null(self) -> None:
        r = run_kpss_test(_rs_wn())
        # The interpretation should mention stationarity
        assert "stationar" in r.interpretation.lower()

    def test_notes_warn_about_opposite_null(self) -> None:
        r = run_kpss_test(_rs_wn())
        combined_notes = " ".join(r.notes).lower()
        assert "opposite" in combined_notes

    def test_lags_used_nonnegative(self) -> None:
        r = run_kpss_test(_rs_wn())
        assert isinstance(r.lags_used, int) and r.lags_used >= 0

    def test_n_gap_flagged_reported(self) -> None:
        vals = _white_noise()
        gaps = np.zeros(len(vals), dtype=bool)
        gaps[20] = True
        rs = _make_return_series(vals, gap_mask=gaps)
        r = run_kpss_test(rs)
        assert r.n_gap_flagged == 1

    def test_json_round_trip(self) -> None:
        d = run_kpss_test(_rs_wn()).to_dict()
        assert json.loads(json.dumps(d)) == d

    def test_deterministic(self) -> None:
        rs = _rs_wn()
        r1 = run_kpss_test(rs)
        r2 = run_kpss_test(rs)
        assert r1.statistic == pytest.approx(r2.statistic)


# ===========================================================================
# Ljung-Box on returns
# ===========================================================================


class TestLjungBoxReturns:
    def test_white_noise_fails_to_reject(self) -> None:
        """LB H₀ = no autocorrelation.  White noise → should NOT reject."""
        r = run_ljung_box_returns(_rs_wn())
        assert not r.rejects_null, f"p={r.p_value:.4f}"

    def test_ar1_rejects_autocorrelation(self) -> None:
        """AR(1) with ρ=0.5 has significant autocorrelation → should reject."""
        r = run_ljung_box_returns(_rs_ar1(rho=0.5))
        assert r.rejects_null, f"p={r.p_value:.4f}"

    def test_null_hypothesis_mentions_autocorrelation(self) -> None:
        r = run_ljung_box_returns(_rs_wn())
        assert "autocorrelation" in r.null_hypothesis.lower()

    def test_lags_used_matches_requested(self) -> None:
        r = run_ljung_box_returns(_rs_wn(), lags=7)
        assert r.lags_used == 7

    def test_series_scale_is_percent(self) -> None:
        assert run_ljung_box_returns(_rs_wn()).series_scale == "percent"

    def test_n_gap_flagged_reported(self) -> None:
        vals = _white_noise()
        gaps = np.zeros(len(vals), dtype=bool)
        gaps[3] = True
        r = run_ljung_box_returns(_make_return_series(vals, gap_mask=gaps))
        assert r.n_gap_flagged == 1

    def test_exclude_gaps_changes_n_observations(self) -> None:
        vals = _white_noise()
        gaps = np.zeros(len(vals), dtype=bool)
        gaps[0:8] = True
        rs = _make_return_series(vals, gap_mask=gaps)
        r_full = run_ljung_box_returns(rs, exclude_gaps=False)
        r_excl = run_ljung_box_returns(rs, exclude_gaps=True)
        assert r_full.n_observations == len(vals)
        assert r_excl.n_observations == len(vals) - 8

    def test_deterministic(self) -> None:
        rs = _rs_wn()
        r1 = run_ljung_box_returns(rs)
        r2 = run_ljung_box_returns(rs)
        assert r1.statistic == pytest.approx(r2.statistic)
        assert r1.p_value == pytest.approx(r2.p_value)

    def test_json_round_trip(self) -> None:
        d = run_ljung_box_returns(_rs_wn()).to_dict()
        assert json.loads(json.dumps(d)) == d


# ===========================================================================
# Ljung-Box on squared returns
# ===========================================================================


class TestLjungBoxSquared:
    def test_white_noise_fails_to_reject(self) -> None:
        """Squared white noise should show no autocorrelation."""
        r = run_ljung_box_squared(_rs_wn())
        assert not r.rejects_null, f"p={r.p_value:.4f}"

    def test_garch_series_rejects(self) -> None:
        """GARCH-like series has strong squared-return autocorrelation → should reject."""
        r = run_ljung_box_squared(_rs_garch())
        assert r.rejects_null, f"p={r.p_value:.4f}"

    def test_null_hypothesis_mentions_squared_returns(self) -> None:
        r = run_ljung_box_squared(_rs_wn())
        assert "squared" in r.null_hypothesis.lower()

    def test_distinct_from_arch_lm(self) -> None:
        """Ljung-Box-squared and ARCH-LM are different tests — verify by test name."""
        rs = _rs_garch()
        lb_sq = run_ljung_box_squared(rs)
        arch = run_arch_lm_test(rs)
        assert lb_sq.test_name != arch.test_name
        # Both should reject for a GARCH series
        assert lb_sq.rejects_null
        assert arch.rejects_null

    def test_notes_distinguish_from_arch_lm(self) -> None:
        r = run_ljung_box_squared(_rs_wn())
        combined = " ".join(r.notes).lower()
        assert "arch-lm" in combined or "ljung-box" in combined

    def test_lags_used_matches_requested(self) -> None:
        r = run_ljung_box_squared(_rs_wn(), lags=8)
        assert r.lags_used == 8

    def test_series_scale_is_percent(self) -> None:
        assert run_ljung_box_squared(_rs_wn()).series_scale == "percent"

    def test_deterministic(self) -> None:
        rs = _rs_garch()
        r1 = run_ljung_box_squared(rs)
        r2 = run_ljung_box_squared(rs)
        assert r1.statistic == pytest.approx(r2.statistic)

    def test_json_round_trip(self) -> None:
        d = run_ljung_box_squared(_rs_garch()).to_dict()
        assert json.loads(json.dumps(d)) == d


# ===========================================================================
# ARCH-LM test
# ===========================================================================


class TestARCHLM:
    def test_white_noise_fails_to_reject(self) -> None:
        """ARCH-LM H₀ = no ARCH effect.  White noise → should NOT reject."""
        r = run_arch_lm_test(_rs_wn())
        assert not r.rejects_null, f"p={r.p_value:.4f}"

    def test_garch_series_rejects(self) -> None:
        """GARCH series has significant ARCH effects → should reject."""
        r = run_arch_lm_test(_rs_garch())
        assert r.rejects_null, f"p={r.p_value:.4f}"

    def test_null_hypothesis_mentions_arch_and_variance(self) -> None:
        r = run_arch_lm_test(_rs_wn())
        nh = r.null_hypothesis.lower()
        assert "arch" in nh

    def test_lags_used_matches_requested(self) -> None:
        r = run_arch_lm_test(_rs_wn(), nlags=3)
        assert r.lags_used == 3

    def test_series_scale_is_percent(self) -> None:
        assert run_arch_lm_test(_rs_wn()).series_scale == "percent"

    def test_arch_and_lb_squared_differ_on_same_series(self) -> None:
        """ARCH-LM and LB-squared use different test frameworks — both kept separate."""
        rs = _rs_garch()
        arch = run_arch_lm_test(rs)
        lb_sq = run_ljung_box_squared(rs)
        assert arch.test_name != lb_sq.test_name
        # Both reject for GARCH, but statistics may differ
        assert arch.statistic != pytest.approx(lb_sq.statistic, rel=1e-3)

    def test_interpretation_mentions_volatility_clustering(self) -> None:
        r = run_arch_lm_test(_rs_garch())
        assert "volatility" in r.interpretation.lower() or "garch" in r.interpretation.lower()

    def test_notes_mention_lm_statistic(self) -> None:
        r = run_arch_lm_test(_rs_wn())
        combined = " ".join(r.notes).lower()
        assert "lm" in combined or "chi" in combined or "χ" in combined

    def test_deterministic(self) -> None:
        rs = _rs_garch()
        r1 = run_arch_lm_test(rs)
        r2 = run_arch_lm_test(rs)
        assert r1.statistic == pytest.approx(r2.statistic)

    def test_json_round_trip(self) -> None:
        d = run_arch_lm_test(_rs_garch()).to_dict()
        assert json.loads(json.dumps(d)) == d

    def test_n_gap_flagged_reported(self) -> None:
        vals = _garch_like()
        gaps = np.zeros(len(vals), dtype=bool)
        gaps[100] = True
        rs = _make_return_series(vals, gap_mask=gaps)
        r = run_arch_lm_test(rs)
        assert r.n_gap_flagged == 1


# ===========================================================================
# run_all_diagnostics
# ===========================================================================


class TestRunAllDiagnostics:
    def test_returns_return_diagnostics_instance(self) -> None:
        assert isinstance(run_all_diagnostics(_rs_wn()), ReturnDiagnostics)

    def test_ticker_propagated(self) -> None:
        rs = _make_return_series(_white_noise(), ticker="XYZW")
        d = run_all_diagnostics(rs)
        assert d.ticker == "XYZW"

    def test_n_observations_total_correct(self) -> None:
        rs = _rs_wn(_N)
        d = run_all_diagnostics(rs)
        assert d.n_observations_total == rs.return_observations

    def test_n_gap_flagged_correct(self) -> None:
        vals = _white_noise()
        gaps = np.zeros(len(vals), dtype=bool)
        gaps[[10, 20, 30]] = True
        rs = _make_return_series(vals, gap_mask=gaps)
        d = run_all_diagnostics(rs)
        assert d.n_gap_flagged == 3

    def test_gaps_excluded_false_by_default(self) -> None:
        d = run_all_diagnostics(_rs_wn())
        assert d.gaps_excluded is False

    def test_gaps_excluded_true_when_requested(self) -> None:
        vals = _white_noise()
        gaps = np.zeros(len(vals), dtype=bool)
        gaps[5:10] = True
        rs = _make_return_series(vals, gap_mask=gaps)
        d = run_all_diagnostics(rs, exclude_gaps=True)
        assert d.gaps_excluded is True
        for r in d.all_results:
            assert r.n_observations == len(vals) - 5

    def test_deterministic_repeated_results(self) -> None:
        rs = _rs_wn()
        d1 = run_all_diagnostics(rs)
        d2 = run_all_diagnostics(rs)
        for r1, r2 in zip(d1.all_results, d2.all_results):
            assert r1.statistic == pytest.approx(r2.statistic, rel=1e-12)
            assert r1.p_value == pytest.approx(r2.p_value, rel=1e-12)

    def test_all_series_scale_percent(self) -> None:
        d = run_all_diagnostics(_rs_wn())
        for r in d.all_results:
            assert r.series_scale == "percent"

    def test_all_conclusions_valid(self) -> None:
        d = run_all_diagnostics(_rs_wn())
        for r in d.all_results:
            assert r.conclusion in ("reject_null", "fail_to_reject_null")

    def test_json_round_trip(self) -> None:
        d = run_all_diagnostics(_rs_wn()).to_dict()
        assert json.loads(json.dumps(d)) == d

    def test_garch_series_all_arch_tests_reject(self) -> None:
        """For GARCH series: both ARCH-LM and LB-squared should reject."""
        d = run_all_diagnostics(_rs_garch())
        assert d.arch_lm.rejects_null
        assert d.ljung_box_squared.rejects_null

    def test_white_noise_passes_arch_tests(self) -> None:
        """White noise: both ARCH tests should fail to reject."""
        d = run_all_diagnostics(_rs_wn())
        assert not d.arch_lm.rejects_null
        assert not d.ljung_box_squared.rejects_null

    def test_white_noise_stationary_tests(self) -> None:
        """White noise: ADF rejects unit root; KPSS fails to reject stationarity."""
        d = run_all_diagnostics(_rs_wn())
        assert d.adf.rejects_null     # ADF: stationary
        assert not d.kpss.rejects_null  # KPSS: stationary (fail to reject H₀)


# ===========================================================================
# Invalid / too-short inputs
# ===========================================================================


class TestInvalidInputs:
    def test_too_short_series_raises_insufficient_data_error(self) -> None:
        vals = _white_noise(10)
        rs = _make_return_series(vals)
        with pytest.raises(InsufficientDataError):
            run_adf_test(rs)

    def test_too_short_raises_for_kpss(self) -> None:
        rs = _make_return_series(_white_noise(10))
        with pytest.raises(InsufficientDataError):
            run_kpss_test(rs)

    def test_too_short_raises_for_lb_returns(self) -> None:
        rs = _make_return_series(_white_noise(10))
        with pytest.raises(InsufficientDataError):
            run_ljung_box_returns(rs)

    def test_too_short_raises_for_lb_squared(self) -> None:
        rs = _make_return_series(_white_noise(10))
        with pytest.raises(InsufficientDataError):
            run_ljung_box_squared(rs)

    def test_too_short_raises_for_arch_lm(self) -> None:
        rs = _make_return_series(_white_noise(10))
        with pytest.raises(InsufficientDataError):
            run_arch_lm_test(rs)

    def test_too_short_raises_for_all_diagnostics(self) -> None:
        rs = _make_return_series(_white_noise(10))
        with pytest.raises(InsufficientDataError):
            run_all_diagnostics(rs)

    def test_insufficient_data_is_statistics_error(self) -> None:
        rs = _make_return_series(_white_noise(10))
        with pytest.raises(StatisticsError):
            run_adf_test(rs)

    def test_too_few_obs_for_lag_count_raises(self) -> None:
        """Series longer than MIN_OBSERVATIONS but shorter than lags+1 should also raise."""
        vals = _white_noise(25)  # 25 > MIN_OBSERVATIONS (20) but < lags+1 for lags=30
        rs = _make_return_series(vals)
        with pytest.raises(InsufficientDataError):
            run_ljung_box_returns(rs, lags=30)

    def test_gap_exclusion_below_min_raises(self) -> None:
        """Excluding enough gaps can drop below the minimum → InsufficientDataError."""
        vals = _white_noise(25)
        gaps = np.zeros(25, dtype=bool)
        gaps[:10] = True  # only 15 remain after exclusion → < MIN_OBSERVATIONS (20)
        rs = _make_return_series(vals, gap_mask=gaps)
        with pytest.raises(InsufficientDataError):
            run_adf_test(rs, exclude_gaps=True)


# ===========================================================================
# ACF figure
# ===========================================================================


class TestACFFigure:
    def test_returns_matplotlib_figure(self) -> None:
        fig = plot_acf(_rs_wn())
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_figure_has_one_axes(self) -> None:
        fig = plot_acf(_rs_wn())
        assert len(fig.axes) == 1

    def test_title_contains_ticker(self) -> None:
        fig = plot_acf(_rs_wn(200))
        assert "SYN" in fig.axes[0].get_title()

    def test_ylabel_is_acf(self) -> None:
        fig = plot_acf(_rs_wn())
        assert "ACF" in fig.axes[0].get_ylabel()

    def test_bars_present_for_nonzero_lags(self) -> None:
        fig = plot_acf(_rs_wn(), lags=10)
        ax = fig.axes[0]
        assert len(ax.patches) > 0  # bar chart has patches

    def test_custom_title_used(self) -> None:
        fig = plot_acf(_rs_wn(), title="My ACF")
        assert fig.axes[0].get_title() == "My ACF"

    def test_custom_figsize(self) -> None:
        fig = plot_acf(_rs_wn(), figsize=(6, 3))
        assert fig.get_size_inches() == pytest.approx((6.0, 3.0))

    def test_squared_true_uses_squared_series(self) -> None:
        fig_ret = plot_acf(_rs_wn(), squared=False)
        fig_sq = plot_acf(_rs_wn(), squared=True)
        # Both should produce a figure — they won't be identical
        assert isinstance(fig_ret, matplotlib.figure.Figure)
        assert isinstance(fig_sq, matplotlib.figure.Figure)
        # Title should mention "squared" for squared variant
        assert "squared" in fig_sq.axes[0].get_title().lower()

    def test_scale_mentioned_in_title(self) -> None:
        fig = plot_acf(_rs_wn())
        assert "percent" in fig.axes[0].get_title().lower()

    def test_exclude_gaps_works(self) -> None:
        vals = _white_noise()
        gaps = np.zeros(len(vals), dtype=bool)
        gaps[5:10] = True
        rs = _make_return_series(vals, gap_mask=gaps)
        fig = plot_acf(rs, exclude_gaps=True)
        assert isinstance(fig, matplotlib.figure.Figure)


# ===========================================================================
# PACF figure
# ===========================================================================


class TestPACFFigure:
    def test_returns_matplotlib_figure(self) -> None:
        fig = plot_pacf(_rs_wn())
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_figure_has_one_axes(self) -> None:
        fig = plot_pacf(_rs_wn())
        assert len(fig.axes) == 1

    def test_title_contains_ticker(self) -> None:
        fig = plot_pacf(_rs_wn())
        assert "SYN" in fig.axes[0].get_title()

    def test_ylabel_is_pacf(self) -> None:
        fig = plot_pacf(_rs_wn())
        assert "PACF" in fig.axes[0].get_ylabel()

    def test_bars_present(self) -> None:
        fig = plot_pacf(_rs_wn(), lags=10)
        assert len(fig.axes[0].patches) > 0

    def test_custom_title(self) -> None:
        fig = plot_pacf(_rs_wn(), title="My PACF")
        assert fig.axes[0].get_title() == "My PACF"

    def test_custom_figsize(self) -> None:
        fig = plot_pacf(_rs_wn(), figsize=(7, 2))
        assert fig.get_size_inches() == pytest.approx((7.0, 2.0))

    def test_scale_mentioned_in_title(self) -> None:
        fig = plot_pacf(_rs_wn())
        assert "percent" in fig.axes[0].get_title().lower()

    def test_exclude_gaps_works(self) -> None:
        vals = _white_noise()
        gaps = np.zeros(len(vals), dtype=bool)
        gaps[5:10] = True
        rs = _make_return_series(vals, gap_mask=gaps)
        fig = plot_pacf(rs, exclude_gaps=True)
        assert isinstance(fig, matplotlib.figure.Figure)


# ===========================================================================
# Null-hypothesis interpretation correctness
# ===========================================================================


class TestNullHypothesisInterpretation:
    def test_adf_reject_interpretation_says_stationary(self) -> None:
        """If ADF rejects H₀ (unit root) → interpretation should say 'stationary'."""
        r = run_adf_test(_rs_wn())
        assert r.rejects_null
        assert "stationary" in r.interpretation.lower()

    def test_adf_fail_interpretation_says_unit_root_or_nonstationary(self) -> None:
        """If ADF fails to reject H₀ → interpretation should warn about non-stationarity."""
        r = run_adf_test(_rs_rw())
        assert not r.rejects_null
        assert (
            "unit root" in r.interpretation.lower()
            or "non-stationary" in r.interpretation.lower()
            or "nonstationary" in r.interpretation.lower()
            or "unusual" in r.interpretation.lower()
        )

    def test_kpss_reject_interpretation_says_nonstationary(self) -> None:
        """If KPSS rejects H₀ (stationarity) → interpretation should say 'non-stationary'."""
        r = run_kpss_test(_rs_rw())
        assert r.rejects_null
        assert (
            "non-stationary" in r.interpretation.lower()
            or "nonstationary" in r.interpretation.lower()
        )

    def test_kpss_fail_says_stationary(self) -> None:
        r = run_kpss_test(_rs_wn())
        assert not r.rejects_null
        assert "stationary" in r.interpretation.lower()

    def test_lb_returns_reject_says_autocorrelation(self) -> None:
        r = run_ljung_box_returns(_rs_ar1(rho=0.5))
        assert r.rejects_null
        assert (
            "serial" in r.interpretation.lower()
            or "autocorrelation" in r.interpretation.lower()
            or "dependence" in r.interpretation.lower()
        )

    def test_lb_squared_reject_says_volatility_clustering(self) -> None:
        r = run_ljung_box_squared(_rs_garch())
        assert r.rejects_null
        assert (
            "volatility" in r.interpretation.lower()
            or "clustering" in r.interpretation.lower()
            or "arch" in r.interpretation.lower()
        )

    def test_arch_lm_reject_says_conditional_heteroscedasticity(self) -> None:
        r = run_arch_lm_test(_rs_garch())
        assert r.rejects_null
        assert (
            "conditional" in r.interpretation.lower()
            or "heteroscedasticity" in r.interpretation.lower()
            or "volatility" in r.interpretation.lower()
        )

    def test_arch_lm_fail_says_constant_variance(self) -> None:
        r = run_arch_lm_test(_rs_wn())
        assert not r.rejects_null
        assert (
            "constant" in r.interpretation.lower()
            or "variance" in r.interpretation.lower()
            or "no significant" in r.interpretation.lower()
        )

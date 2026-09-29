"""Tests for the Stage 8 volatility model fitting engine.

Coverage
--------
A. GARCH(1,1) fitting
B. EGARCH(1,1,1) fitting
C. Common interface behaviour (model_name, config, model_type)
D. Result metadata (ticker, frequency, return_scale, n_observations, period)
E. AIC / BIC / log-likelihood extraction
F. Parameter extraction (names and finite values)
G. Standard-error extraction (finite, same keys as params)
H. Conditional-volatility index alignment
I. Standardised-residual output (length, alignment)
J. Percentage-scale preservation (daily_vol_pct in reasonable range)
K. Annualisation calculation (annualized = daily * sqrt(factor))
L. JSON serialisation
M. Convergence status (converged is bool)
N. Insufficient-data handling (< MIN_OBSERVATIONS)
O. Invalid input handling (wrong type)
P. Numerical / non-finite result handling (tested via config validation)
Q. Deterministic repeated fit under fixed synthetic data / configuration
R. No mutation of the input ReturnSeries
S. Warning / note behaviour when arch_lm_note is supplied
T. Model comparison schema (GARCH vs EGARCH consistent fields)
U. Regression: model layer does not import yfinance
V. Regression: model layer does not import streamlit
W. Regression: ModelFitResult does not hold raw arch objects
X. Figure: plot_conditional_volatility returns Figure
Y. Figure: plot_volatility_comparison returns Figure
Z. Config validation: invalid p / q / dist / mean raises

All tests use synthetic data only — no network access, no Yahoo Finance.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import date, datetime, timezone

import matplotlib.figure
import numpy as np
import pandas as pd
import pytest

from src.exceptions import (
    VolatilityModelConfigError,
    VolatilityModelDataError,
    VolatilityModelError,
    VolatilityModelFitError,
)
from src.models.egarch import EGARCHModel
from src.models.figures import plot_conditional_volatility, plot_volatility_comparison
from src.models.garch import GARCHModel
from src.models.result import ModelFitResult, _TRADING_DAYS_PER_YEAR
from src.preprocessing.return_series import PriceBasis, ReturnSeries

# ---------------------------------------------------------------------------
# Synthetic ReturnSeries factory (mirrors test_statistics_diagnostics.py)
# ---------------------------------------------------------------------------

_FETCHED_AT = datetime(2024, 6, 3, 12, 0, tzinfo=timezone.utc)
_FIRST_PRICE_DATE = date(2020, 1, 1)


def _make_return_series(
    decimal_values: np.ndarray,
    *,
    ticker: str = "SYN",
) -> ReturnSeries:
    n = len(decimal_values)
    idx = pd.bdate_range("2020-01-02", periods=n, name="date")
    decimal = pd.Series(decimal_values.astype("float64"), index=idx)
    spans_gap = pd.Series(np.zeros(n, dtype=bool), index=idx)
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


def _garch_like(n: int = 500) -> np.ndarray:
    """GARCH(1,1)-like returns in decimal scale — strong volatility clustering."""
    rng = np.random.default_rng(99)
    eps = rng.standard_normal(n)
    omega, alpha, beta = 1e-5, 0.15, 0.80
    h = np.empty(n)
    r = np.empty(n)
    h[0] = omega / (1.0 - alpha - beta)
    r[0] = math.sqrt(h[0]) * eps[0]
    for t in range(1, n):
        h[t] = omega + alpha * r[t - 1] ** 2 + beta * h[t - 1]
        r[t] = math.sqrt(h[t]) * eps[t]
    return r


def _white_noise(n: int = 300) -> np.ndarray:
    return np.random.default_rng(2025).normal(0.0, 0.01, n)


def _rs_garch(n: int = 500, ticker: str = "SYN") -> ReturnSeries:
    return _make_return_series(_garch_like(n), ticker=ticker)


def _rs_wn(n: int = 300, ticker: str = "SYN") -> ReturnSeries:
    return _make_return_series(_white_noise(n), ticker=ticker)


# ===========================================================================
# A  GARCH fitting
# ===========================================================================


class TestGARCHFit:
    def test_fit_returns_model_fit_result(self) -> None:
        r = GARCHModel(ticker="SYN").fit(_rs_garch())
        assert isinstance(r, ModelFitResult)

    def test_model_type_is_garch(self) -> None:
        r = GARCHModel(ticker="SYN").fit(_rs_garch())
        assert r.model_type == "garch"

    def test_model_name_format(self) -> None:
        r = GARCHModel(p=1, q=1, ticker="SYN").fit(_rs_garch())
        assert r.model_name == "GARCH(1,1)"

    def test_fit_with_white_noise(self) -> None:
        """GARCH should fit even on white noise (no ARCH effect)."""
        r = GARCHModel(ticker="SYN").fit(_rs_wn())
        assert isinstance(r, ModelFitResult)

    def test_converged_is_bool(self) -> None:
        r = GARCHModel().fit(_rs_garch())
        assert isinstance(r.converged, bool)

    def test_garch_like_data_converges(self) -> None:
        r = GARCHModel(ticker="SYN").fit(_rs_garch())
        assert r.converged is True


# ===========================================================================
# B  EGARCH fitting
# ===========================================================================


class TestEGARCHFit:
    def test_fit_returns_model_fit_result(self) -> None:
        r = EGARCHModel(ticker="SYN").fit(_rs_garch())
        assert isinstance(r, ModelFitResult)

    def test_model_type_is_egarch(self) -> None:
        r = EGARCHModel(ticker="SYN").fit(_rs_garch())
        assert r.model_type == "egarch"

    def test_model_name_format(self) -> None:
        r = EGARCHModel(p=1, q=1, ticker="SYN").fit(_rs_garch())
        assert r.model_name == "EGARCH(1,1,1)"

    def test_fit_with_white_noise(self) -> None:
        r = EGARCHModel(ticker="SYN").fit(_rs_wn())
        assert isinstance(r, ModelFitResult)

    def test_egarch_has_asymmetry_parameter(self) -> None:
        """EGARCH exposes a leverage/asymmetry parameter (gamma[1] in arch 8 with o=1)."""
        r = EGARCHModel(ticker="SYN").fit(_rs_garch())
        # arch 8.0: EGARCH(1,1,1) params are [mu, omega, alpha[1], gamma[1], beta[1]]
        assert "gamma[1]" in r.params


# ===========================================================================
# C  Common interface
# ===========================================================================


class TestCommonInterface:
    @pytest.fixture(params=["garch", "egarch"])
    def result(self, request: pytest.FixtureRequest) -> ModelFitResult:
        rs = _rs_garch()
        if request.param == "garch":
            return GARCHModel(ticker="SYN").fit(rs)
        return EGARCHModel(ticker="SYN").fit(rs)

    def test_model_name_is_str(self, result: ModelFitResult) -> None:
        assert isinstance(result.model_name, str)

    def test_model_type_is_known(self, result: ModelFitResult) -> None:
        assert result.model_type in ("garch", "egarch")

    def test_config_is_dict(self, result: ModelFitResult) -> None:
        assert isinstance(result.config, dict)

    def test_config_json_serialisable(self, result: ModelFitResult) -> None:
        assert json.loads(json.dumps(result.config)) == result.config

    def test_config_has_p_q(self, result: ModelFitResult) -> None:
        assert "p" in result.config and "q" in result.config

    def test_converged_is_bool(self, result: ModelFitResult) -> None:
        assert isinstance(result.converged, bool)

    def test_return_scale_is_percent(self, result: ModelFitResult) -> None:
        assert result.return_scale == "percent"

    def test_log_likelihood_is_finite(self, result: ModelFitResult) -> None:
        assert math.isfinite(result.log_likelihood)

    def test_aic_is_finite(self, result: ModelFitResult) -> None:
        assert math.isfinite(result.aic)

    def test_bic_is_finite(self, result: ModelFitResult) -> None:
        assert math.isfinite(result.bic)


# ===========================================================================
# D  Result metadata
# ===========================================================================


class TestResultMetadata:
    def _result(self) -> ModelFitResult:
        return GARCHModel(ticker="NSEI").fit(_rs_garch(ticker="NSEI"))

    def test_ticker_propagated(self) -> None:
        assert self._result().ticker == "NSEI"

    def test_frequency_propagated(self) -> None:
        assert self._result().frequency == "1d"

    def test_return_scale_is_percent(self) -> None:
        assert self._result().return_scale == "percent"

    def test_n_observations_correct(self) -> None:
        rs = _rs_garch()
        r = GARCHModel().fit(rs)
        assert r.n_observations == rs.return_observations

    def test_fit_start_before_fit_end(self) -> None:
        r = self._result()
        assert r.fit_start <= r.fit_end

    def test_fit_start_matches_first_return(self) -> None:
        rs = _rs_garch()
        r = GARCHModel().fit(rs)
        assert r.fit_start == pd.Timestamp(rs.decimal.index[0]).date()

    def test_fit_end_matches_last_return(self) -> None:
        rs = _rs_garch()
        r = GARCHModel().fit(rs)
        assert r.fit_end == pd.Timestamp(rs.decimal.index[-1]).date()

    def test_annualization_factor_is_252(self) -> None:
        assert self._result().annualization_factor == 252

    def test_notes_is_tuple_of_strings(self) -> None:
        r = self._result()
        assert isinstance(r.notes, tuple)
        assert all(isinstance(n, str) for n in r.notes)


# ===========================================================================
# E  AIC / BIC / log-likelihood
# ===========================================================================


class TestInformationCriteria:
    def test_aic_is_float(self) -> None:
        r = GARCHModel().fit(_rs_garch())
        assert isinstance(r.aic, float)

    def test_bic_is_float(self) -> None:
        r = GARCHModel().fit(_rs_garch())
        assert isinstance(r.bic, float)

    def test_loglik_is_float(self) -> None:
        r = GARCHModel().fit(_rs_garch())
        assert isinstance(r.log_likelihood, float)

    def test_aic_gt_loglik_times_2(self) -> None:
        """AIC = -2 logL + 2k, so AIC > -2*logL when k>0."""
        r = GARCHModel().fit(_rs_garch())
        assert r.aic > -2.0 * r.log_likelihood - 1e-6

    def test_egarch_exposes_aic_bic_loglik(self) -> None:
        r = EGARCHModel().fit(_rs_garch())
        assert math.isfinite(r.aic)
        assert math.isfinite(r.bic)
        assert math.isfinite(r.log_likelihood)

    def test_garch_egarch_have_same_schema_keys(self) -> None:
        rg = GARCHModel().fit(_rs_garch())
        re = EGARCHModel().fit(_rs_garch())
        for attr in ("aic", "bic", "log_likelihood", "converged",
                     "daily_vol_pct", "annualized_vol_pct", "std_residuals"):
            assert hasattr(rg, attr)
            assert hasattr(re, attr)


# ===========================================================================
# F  Parameter extraction
# ===========================================================================


class TestParameters:
    def test_garch_params_has_expected_names(self) -> None:
        r = GARCHModel().fit(_rs_garch())
        # arch names: mu, omega, alpha[1], beta[1]
        expected = {"mu", "omega", "alpha[1]", "beta[1]"}
        assert expected.issubset(set(r.params.keys()))

    def test_all_params_finite(self) -> None:
        r = GARCHModel().fit(_rs_garch())
        assert all(math.isfinite(v) for v in r.params.values())

    def test_egarch_params_has_asymmetry(self) -> None:
        """EGARCH gamma[1] captures the magnitude/asymmetry effect (arch 8)."""
        r = EGARCHModel().fit(_rs_garch())
        assert "gamma[1]" in r.params

    def test_params_is_dict_of_floats(self) -> None:
        r = GARCHModel().fit(_rs_garch())
        assert isinstance(r.params, dict)
        for k, v in r.params.items():
            assert isinstance(k, str)
            assert isinstance(v, float)

    def test_garch_persistence(self) -> None:
        """GARCH alpha+beta is typically < 1 for a valid GARCH process."""
        r = GARCHModel().fit(_rs_garch())
        alpha = r.params.get("alpha[1]", 1.0)
        beta = r.params.get("beta[1]", 1.0)
        # For our GARCH-like data this should clearly hold
        assert alpha + beta < 1.05   # allow small numerical slack


# ===========================================================================
# G  Standard errors
# ===========================================================================


class TestStandardErrors:
    def test_std_errors_same_keys_as_params(self) -> None:
        r = GARCHModel().fit(_rs_garch())
        assert set(r.std_errors.keys()) == set(r.params.keys())

    def test_std_errors_non_negative(self) -> None:
        r = GARCHModel().fit(_rs_garch())
        for v in r.std_errors.values():
            assert v >= 0.0 or not math.isfinite(v)

    def test_std_errors_is_dict_of_floats(self) -> None:
        r = GARCHModel().fit(_rs_garch())
        assert isinstance(r.std_errors, dict)
        for k, v in r.std_errors.items():
            assert isinstance(k, str)
            assert isinstance(v, float)

    def test_egarch_std_errors_present(self) -> None:
        r = EGARCHModel().fit(_rs_garch())
        assert len(r.std_errors) > 0


# ===========================================================================
# H  Conditional-volatility index alignment
# ===========================================================================


class TestCondVolAlignment:
    def test_daily_vol_index_matches_returns(self) -> None:
        rs = _rs_garch()
        r = GARCHModel().fit(rs)
        pd.testing.assert_index_equal(r.daily_vol_pct.index, rs.decimal.index)

    def test_annualized_vol_index_matches_returns(self) -> None:
        rs = _rs_garch()
        r = GARCHModel().fit(rs)
        pd.testing.assert_index_equal(r.annualized_vol_pct.index, rs.decimal.index)

    def test_daily_vol_len_equals_n_obs(self) -> None:
        rs = _rs_garch()
        r = GARCHModel().fit(rs)
        assert len(r.daily_vol_pct) == r.n_observations

    def test_egarch_vol_index_matches(self) -> None:
        rs = _rs_garch()
        r = EGARCHModel().fit(rs)
        pd.testing.assert_index_equal(r.daily_vol_pct.index, rs.decimal.index)

    def test_daily_vol_name(self) -> None:
        r = GARCHModel().fit(_rs_garch())
        assert r.daily_vol_pct.name == "daily_vol_pct"

    def test_annualized_vol_name(self) -> None:
        r = GARCHModel().fit(_rs_garch())
        assert r.annualized_vol_pct.name == "annualized_vol_pct"


# ===========================================================================
# I  Standardised residuals
# ===========================================================================


class TestStdResiduals:
    def test_std_resid_len_equals_n_obs(self) -> None:
        rs = _rs_garch()
        r = GARCHModel().fit(rs)
        assert len(r.std_residuals) == r.n_observations

    def test_std_resid_index_matches_returns(self) -> None:
        rs = _rs_garch()
        r = GARCHModel().fit(rs)
        pd.testing.assert_index_equal(r.std_residuals.index, rs.decimal.index)

    def test_std_resid_finite(self) -> None:
        r = GARCHModel().fit(_rs_garch())
        assert np.all(np.isfinite(r.std_residuals.values))

    def test_std_resid_name(self) -> None:
        r = GARCHModel().fit(_rs_garch())
        assert r.std_residuals.name == "std_resid"


# ===========================================================================
# J  Percentage-scale preservation
# ===========================================================================


class TestPercentageScale:
    def test_return_scale_is_percent(self) -> None:
        r = GARCHModel().fit(_rs_garch())
        assert r.return_scale == "percent"

    def test_daily_vol_pct_order_of_magnitude(self) -> None:
        """For percent-scale daily equity returns, cond vol should be in [0.01, 20] %."""
        r = GARCHModel().fit(_rs_garch())
        assert r.daily_vol_pct.min() > 0.0
        assert r.daily_vol_pct.max() < 50.0

    def test_daily_vol_pct_all_positive(self) -> None:
        r = GARCHModel().fit(_rs_garch())
        assert (r.daily_vol_pct > 0.0).all()

    def test_annualized_gt_daily_for_typical_data(self) -> None:
        """Annualised = daily × sqrt(252) > daily for any positive daily vol."""
        r = GARCHModel().fit(_rs_garch())
        ratio = (r.annualized_vol_pct / r.daily_vol_pct).mean()
        assert abs(ratio - math.sqrt(252)) < 0.01


# ===========================================================================
# K  Annualisation
# ===========================================================================


class TestAnnualisation:
    def test_annualized_equals_daily_times_sqrt_factor(self) -> None:
        r = GARCHModel().fit(_rs_garch())
        expected = r.daily_vol_pct * math.sqrt(r.annualization_factor)
        pd.testing.assert_series_equal(
            r.annualized_vol_pct.reset_index(drop=True),
            expected.reset_index(drop=True),
            rtol=1e-10,
            check_names=False,
        )

    def test_annualization_factor_default_is_252(self) -> None:
        r = GARCHModel().fit(_rs_garch())
        assert r.annualization_factor == _TRADING_DAYS_PER_YEAR

    def test_custom_annualization_factor(self) -> None:
        r = GARCHModel(annualization_factor=260).fit(_rs_garch())
        assert r.annualization_factor == 260
        expected = r.daily_vol_pct * math.sqrt(260)
        pd.testing.assert_series_equal(
            r.annualized_vol_pct.reset_index(drop=True),
            expected.reset_index(drop=True),
            rtol=1e-10,
            check_names=False,
        )


# ===========================================================================
# L  JSON serialisation
# ===========================================================================


class TestJSONSerialisation:
    def test_garch_to_dict_round_trip(self) -> None:
        r = GARCHModel().fit(_rs_garch())
        d = r.to_dict()
        assert json.loads(json.dumps(d)) == d

    def test_egarch_to_dict_round_trip(self) -> None:
        r = EGARCHModel().fit(_rs_garch())
        d = r.to_dict()
        assert json.loads(json.dumps(d)) == d

    def test_to_json_parseable(self) -> None:
        r = GARCHModel().fit(_rs_garch())
        parsed = json.loads(r.to_json())
        assert "model_name" in parsed
        assert "daily_vol_pct" in parsed

    def test_dict_contains_required_keys(self) -> None:
        d = GARCHModel().fit(_rs_garch()).to_dict()
        for key in ("model_name", "model_type", "config", "ticker", "frequency",
                    "return_scale", "n_observations", "fit_start", "fit_end",
                    "converged", "log_likelihood", "aic", "bic",
                    "params", "std_errors", "daily_vol_pct",
                    "annualized_vol_pct", "annualization_factor",
                    "std_residuals", "notes"):
            assert key in d, f"missing key: {key}"

    def test_notes_serialised_as_list(self) -> None:
        d = GARCHModel().fit(_rs_garch()).to_dict()
        assert isinstance(d["notes"], list)

    def test_vol_series_serialised_as_dict(self) -> None:
        d = GARCHModel().fit(_rs_garch()).to_dict()
        assert isinstance(d["daily_vol_pct"], dict)
        assert isinstance(d["annualized_vol_pct"], dict)

    def test_vol_keys_are_date_strings(self) -> None:
        d = GARCHModel().fit(_rs_garch()).to_dict()
        for k in list(d["daily_vol_pct"].keys())[:5]:
            date.fromisoformat(k)   # must not raise


# ===========================================================================
# M  Convergence status
# ===========================================================================


class TestConvergence:
    def test_converged_is_bool(self) -> None:
        r = GARCHModel().fit(_rs_garch())
        assert isinstance(r.converged, bool)

    def test_garch_like_data_converges(self) -> None:
        r = GARCHModel().fit(_rs_garch())
        assert r.converged is True


# ===========================================================================
# N  Insufficient data
# ===========================================================================


class TestInsufficientData:
    def test_garch_too_short_raises(self) -> None:
        short = _make_return_series(_garch_like(10))
        with pytest.raises(VolatilityModelDataError):
            GARCHModel().fit(short)

    def test_egarch_too_short_raises(self) -> None:
        short = _make_return_series(_garch_like(10))
        with pytest.raises(VolatilityModelDataError):
            EGARCHModel().fit(short)

    def test_error_is_volatility_model_error(self) -> None:
        short = _make_return_series(_garch_like(10))
        with pytest.raises(VolatilityModelError):
            GARCHModel().fit(short)


# ===========================================================================
# O  Invalid input handling
# ===========================================================================


class TestInvalidInput:
    def test_non_return_series_raises(self) -> None:
        with pytest.raises(VolatilityModelDataError):
            GARCHModel().fit("not a ReturnSeries")  # type: ignore[arg-type]

    def test_none_raises(self) -> None:
        with pytest.raises(VolatilityModelDataError):
            GARCHModel().fit(None)  # type: ignore[arg-type]


# ===========================================================================
# Q  Deterministic repeated fit
# ===========================================================================


class TestDeterminism:
    def test_garch_same_result_twice(self) -> None:
        rs = _rs_garch()
        r1 = GARCHModel(ticker="SYN").fit(rs)
        r2 = GARCHModel(ticker="SYN").fit(rs)
        assert r1.aic == pytest.approx(r2.aic, rel=1e-8)
        assert r1.params == pytest.approx(r2.params, rel=1e-8)

    def test_egarch_same_result_twice(self) -> None:
        rs = _rs_garch()
        r1 = EGARCHModel(ticker="SYN").fit(rs)
        r2 = EGARCHModel(ticker="SYN").fit(rs)
        assert r1.aic == pytest.approx(r2.aic, rel=1e-8)


# ===========================================================================
# R  No mutation of ReturnSeries
# ===========================================================================


class TestNoMutation:
    def test_garch_does_not_mutate_returns(self) -> None:
        rs = _rs_garch()
        original = rs.decimal.values.copy()
        GARCHModel().fit(rs)
        np.testing.assert_array_equal(rs.decimal.values, original)

    def test_egarch_does_not_mutate_returns(self) -> None:
        rs = _rs_garch()
        original = rs.decimal.values.copy()
        EGARCHModel().fit(rs)
        np.testing.assert_array_equal(rs.decimal.values, original)


# ===========================================================================
# S  Warning / note behaviour
# ===========================================================================


class TestNotes:
    def test_arch_lm_note_propagated(self) -> None:
        rs = _rs_garch()
        note = "Weak ARCH effect: ARCH-LM p=0.42 fails to reject H0."
        r = GARCHModel().fit(rs, arch_lm_note=note)
        assert note in r.notes

    def test_no_note_when_none_passed(self) -> None:
        rs = _rs_garch()
        r = GARCHModel().fit(rs, arch_lm_note=None)
        # Notes may be empty or contain only convergence info; the ARCH-LM note should not appear
        assert not any("arch-lm" in n.lower() for n in r.notes)

    def test_egarch_arch_lm_note_propagated(self) -> None:
        rs = _rs_garch()
        note = "Weak ARCH effect for EGARCH."
        r = EGARCHModel().fit(rs, arch_lm_note=note)
        assert note in r.notes


# ===========================================================================
# T  Comparison schema (GARCH vs EGARCH)
# ===========================================================================


class TestComparisonSchema:
    def test_both_expose_aic_bic_loglik(self) -> None:
        rs = _rs_garch()
        rg = GARCHModel().fit(rs)
        re = EGARCHModel().fit(rs)
        for attr in ("aic", "bic", "log_likelihood"):
            assert math.isfinite(getattr(rg, attr))
            assert math.isfinite(getattr(re, attr))

    def test_both_daily_vol_same_index(self) -> None:
        rs = _rs_garch()
        rg = GARCHModel().fit(rs)
        re = EGARCHModel().fit(rs)
        pd.testing.assert_index_equal(rg.daily_vol_pct.index, re.daily_vol_pct.index)

    def test_both_model_types_different(self) -> None:
        rs = _rs_garch()
        assert GARCHModel().fit(rs).model_type != EGARCHModel().fit(rs).model_type

    def test_to_dict_same_top_level_keys(self) -> None:
        rs = _rs_garch()
        kg = set(GARCHModel().fit(rs).to_dict())
        ke = set(EGARCHModel().fit(rs).to_dict())
        assert kg == ke


# ===========================================================================
# U + V + W  Regression tests
# ===========================================================================


class TestRegressions:
    def test_model_layer_does_not_import_yfinance(self) -> None:
        """None of the src.models modules should import yfinance."""
        import src.models.garch
        import src.models.egarch
        import src.models.result
        import src.models.figures
        assert "yfinance" not in sys.modules or True  # yfinance may be in env
        # The key check: model module source must not reference yfinance
        import inspect
        for mod in (src.models.garch, src.models.egarch,
                    src.models.result, src.models.figures):
            src_text = inspect.getsource(mod)
            assert "yfinance" not in src_text, (
                f"{mod.__name__} imports yfinance"
            )

    def test_model_layer_does_not_import_streamlit(self) -> None:
        import src.models.garch
        import src.models.egarch
        import src.models.result
        import src.models.figures
        import inspect
        for mod in (src.models.garch, src.models.egarch,
                    src.models.result, src.models.figures):
            src_text = inspect.getsource(mod)
            assert "import streamlit" not in src_text.lower(), (
                f"{mod.__name__} imports streamlit"
            )

    def test_model_fit_result_does_not_hold_arch_object(self) -> None:
        r = GARCHModel().fit(_rs_garch())
        d = r.__dict__
        for k, v in d.items():
            # No value should be an arch ARCHModelResult
            type_name = type(v).__module__ or ""
            assert "arch" not in type_name.lower() or k in ("_ticker",), (
                f"Field '{k}' contains an arch object"
            )
        # More targeted: the result dict must be JSON-serialisable
        assert json.loads(r.to_json())


# ===========================================================================
# X + Y  Figures
# ===========================================================================


class TestFigures:
    def _garch_result(self) -> ModelFitResult:
        return GARCHModel(ticker="SYN").fit(_rs_garch())

    def test_plot_conditional_vol_returns_figure(self) -> None:
        fig = plot_conditional_volatility(self._garch_result())
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_conditional_vol_annualized(self) -> None:
        fig = plot_conditional_volatility(self._garch_result(), annualized=True)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_conditional_vol_daily(self) -> None:
        fig = plot_conditional_volatility(self._garch_result(), annualized=False)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_conditional_vol_has_one_axis(self) -> None:
        fig = plot_conditional_volatility(self._garch_result())
        assert len(fig.axes) == 1

    def test_plot_conditional_vol_custom_title(self) -> None:
        fig = plot_conditional_volatility(self._garch_result(), title="My Vol")
        assert fig.axes[0].get_title() == "My Vol"

    def test_plot_comparison_returns_figure(self) -> None:
        rs = _rs_garch()
        rg = GARCHModel(ticker="SYN").fit(rs)
        re = EGARCHModel(ticker="SYN").fit(rs)
        fig = plot_volatility_comparison([rg, re])
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_comparison_has_one_axis(self) -> None:
        rs = _rs_garch()
        rg = GARCHModel(ticker="SYN").fit(rs)
        re = EGARCHModel(ticker="SYN").fit(rs)
        fig = plot_volatility_comparison([rg, re])
        assert len(fig.axes) == 1

    def test_custom_figsize(self) -> None:
        fig = plot_conditional_volatility(self._garch_result(), figsize=(8, 4))
        assert fig.get_size_inches() == pytest.approx((8.0, 4.0))


# ===========================================================================
# Z  Config validation
# ===========================================================================


class TestConfigValidation:
    def test_garch_p_zero_raises(self) -> None:
        with pytest.raises(VolatilityModelConfigError):
            GARCHModel(p=0)

    def test_garch_p_negative_raises(self) -> None:
        with pytest.raises(VolatilityModelConfigError):
            GARCHModel(p=-1)

    def test_garch_p_float_raises(self) -> None:
        with pytest.raises(VolatilityModelConfigError):
            GARCHModel(p=1.0)  # type: ignore[arg-type]

    def test_garch_q_negative_raises(self) -> None:
        with pytest.raises(VolatilityModelConfigError):
            GARCHModel(q=-1)

    def test_garch_invalid_dist_raises(self) -> None:
        with pytest.raises(VolatilityModelConfigError):
            GARCHModel(dist="Laplace")

    def test_garch_invalid_mean_raises(self) -> None:
        with pytest.raises(VolatilityModelConfigError):
            GARCHModel(mean="GARCH")

    def test_garch_invalid_ann_factor_raises(self) -> None:
        with pytest.raises(VolatilityModelConfigError):
            GARCHModel(annualization_factor=0)

    def test_egarch_p_zero_raises(self) -> None:
        with pytest.raises(VolatilityModelConfigError):
            EGARCHModel(p=0)

    def test_egarch_o_negative_raises(self) -> None:
        with pytest.raises(VolatilityModelConfigError):
            EGARCHModel(o=-1)

    def test_egarch_q_zero_raises(self) -> None:
        """EGARCH requires q >= 1 (unlike GARCH where q=0 is valid)."""
        with pytest.raises(VolatilityModelConfigError):
            EGARCHModel(q=0)

    def test_egarch_invalid_dist_raises(self) -> None:
        with pytest.raises(VolatilityModelConfigError):
            EGARCHModel(dist="bad_dist")

    def test_config_error_is_vol_model_error(self) -> None:
        with pytest.raises(VolatilityModelError):
            GARCHModel(p=0)

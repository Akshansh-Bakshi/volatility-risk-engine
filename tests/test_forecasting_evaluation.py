"""Tests for the Stage 7 forecast evaluation layer.

Coverage
--------
A. RMSE correctness (hand-calculated)
B. MAE correctness (hand-calculated)
C. MAPE correctness (hand-calculated)
D. Zero-value handling in MAPE
E. All-zero actuals → MAPE is None
F. Length mismatch raises EvaluationError
G. Index misalignment raises EvaluationError
H. Empty series raises EvaluationError
I. Non-finite values raise EvaluationError
J. evaluate_forecast produces EvaluationResult
K. EvaluationResult is JSON-serialisable
L. ComparisonResult collects per-model results
M. ComparisonResult.ranked_by ordering
N. compare_models empty list raises
O. Deterministic repeated evaluation
P. plot_forecast_vs_actual figure structure
Q. evaluate_forecast with real fitted model
R. compare_models with multiple real models
S. EvaluationResult metadata correctness
T. Notes populated when zero actuals encountered

All tests are deterministic offline tests.
No Yahoo Finance, no network access.
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime, timezone

import matplotlib.figure
import numpy as np
import pandas as pd
import pytest

from src.exceptions import EvaluationError, ForecastingError
from src.forecasting.arima import ARIMAForecaster
from src.forecasting.evaluation import (
    ComparisonResult,
    EvaluationResult,
    _compute_metrics,
    compare_models,
    evaluate_forecast,
)
from src.forecasting.figures import plot_forecast_vs_actual
from src.forecasting.holtwinters import HoltWintersForecaster
from src.forecasting.result import ForecastResult
from src.forecasting.sarima import SARIMAForecaster

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FETCHED_AT = datetime(2024, 1, 1, tzinfo=timezone.utc)
_BDAYS = pd.bdate_range("2020-01-02", periods=200, name="date")


def _make_price_series(n: int = 200, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    vals = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    vals = vals - vals.min() + 50.0
    return pd.Series(vals, index=pd.bdate_range("2020-01-02", periods=n, name="date"),
                     name="close")


def _make_actual_predicted(
    actual_vals: list[float],
    pred_vals: list[float],
    start: str = "2020-10-01",
) -> tuple[pd.Series, pd.Series]:
    """Build aligned actual / predicted Series on consecutive business days."""
    idx = pd.bdate_range(start, periods=len(actual_vals), name="date")
    actual = pd.Series(actual_vals, index=idx, dtype=float)
    pred = pd.Series(pred_vals, index=idx, dtype=float)
    return actual, pred


def _make_forecast_result(
    pred_vals: list[float],
    start: str = "2020-10-01",
    model_name: str = "ARIMA(1,1,0)",
    model_type: str = "arima",
) -> ForecastResult:
    """Build a ForecastResult with the given predicted values."""
    idx = pd.bdate_range(start, periods=len(pred_vals), name="date")
    forecast_values = {str(pd.Timestamp(t).date()): v for t, v in zip(idx, pred_vals)}
    return ForecastResult(
        model_name=model_name,
        model_type=model_type,
        config={"p": 1, "d": 1, "q": 0},
        ticker="SYN",
        series_description="adjusted close price",
        frequency="1d",
        train_start=date(2020, 1, 2),
        train_end=date(2020, 9, 30),
        n_train=200,
        forecast_start=pd.Timestamp(idx[0]).date(),
        forecast_end=pd.Timestamp(idx[-1]).date(),
        n_forecast=len(pred_vals),
        forecast_values=forecast_values,
        fitted=True,
    )


def _fitted_arima(n_train: int = 160) -> tuple[ARIMAForecaster, pd.Series]:
    train = _make_price_series(n_train)
    m = ARIMAForecaster(order=(1, 1, 0), ticker="SYN")
    m.fit(train)
    return m, train


# ===========================================================================
# A  RMSE correctness
# ===========================================================================


class TestRMSE:
    def test_perfect_forecast_gives_zero_rmse(self) -> None:
        vals = [10.0, 20.0, 30.0, 40.0]
        actual, pred = _make_actual_predicted(vals, vals)
        fr = _make_forecast_result(vals)
        ev = evaluate_forecast(actual, fr)
        assert ev.rmse == pytest.approx(0.0, abs=1e-12)

    def test_known_rmse_hand_calculation(self) -> None:
        # errors = [1, -2, 3, -4]  → MSE = (1+4+9+16)/4 = 30/4 = 7.5
        # RMSE = sqrt(7.5)
        actual = [10.0, 10.0, 10.0, 10.0]
        pred   = [11.0,  8.0, 13.0,  6.0]
        expected_rmse = math.sqrt(7.5)
        actual_s, _ = _make_actual_predicted(actual, pred)
        fr = _make_forecast_result(pred)
        ev = evaluate_forecast(actual_s, fr)
        assert ev.rmse == pytest.approx(expected_rmse, rel=1e-10)

    def test_rmse_non_negative(self) -> None:
        a = [5.0, 6.0, 7.0]
        p = [4.0, 7.0, 8.0]
        actual, _ = _make_actual_predicted(a, p)
        ev = evaluate_forecast(actual, _make_forecast_result(p))
        assert ev.rmse >= 0.0

    def test_rmse_symmetric_errors(self) -> None:
        # +1 and −1 errors → RMSE = 1.0
        a = [10.0, 10.0]
        p = [11.0,  9.0]
        actual, _ = _make_actual_predicted(a, p)
        ev = evaluate_forecast(actual, _make_forecast_result(p))
        assert ev.rmse == pytest.approx(1.0, rel=1e-10)


# ===========================================================================
# B  MAE correctness
# ===========================================================================


class TestMAE:
    def test_perfect_forecast_gives_zero_mae(self) -> None:
        vals = [5.0, 10.0, 15.0]
        actual, _ = _make_actual_predicted(vals, vals)
        ev = evaluate_forecast(actual, _make_forecast_result(vals))
        assert ev.mae == pytest.approx(0.0, abs=1e-12)

    def test_known_mae_hand_calculation(self) -> None:
        # |errors| = [1, 2, 3]  → MAE = 6/3 = 2.0
        actual = [10.0, 10.0, 10.0]
        pred   = [11.0,  8.0, 13.0]
        actual_s, _ = _make_actual_predicted(actual, pred)
        ev = evaluate_forecast(actual_s, _make_forecast_result(pred))
        assert ev.mae == pytest.approx(2.0, rel=1e-10)

    def test_mae_non_negative(self) -> None:
        a = [3.0, 3.0]
        p = [4.0, 2.0]
        actual, _ = _make_actual_predicted(a, p)
        ev = evaluate_forecast(actual, _make_forecast_result(p))
        assert ev.mae >= 0.0

    def test_mae_le_rmse(self) -> None:
        """MAE ≤ RMSE always holds (by Jensen's inequality)."""
        a = [10.0, 20.0, 15.0, 25.0]
        p = [12.0, 18.0, 14.0, 27.0]
        actual, _ = _make_actual_predicted(a, p)
        ev = evaluate_forecast(actual, _make_forecast_result(p))
        assert ev.mae <= ev.rmse + 1e-12


# ===========================================================================
# C  MAPE correctness
# ===========================================================================


class TestMAPE:
    def test_perfect_forecast_gives_zero_mape(self) -> None:
        vals = [10.0, 20.0, 30.0]
        actual, _ = _make_actual_predicted(vals, vals)
        ev = evaluate_forecast(actual, _make_forecast_result(vals))
        assert ev.mape == pytest.approx(0.0, abs=1e-12)

    def test_known_mape_hand_calculation(self) -> None:
        # actual=[100, 200]  pred=[110, 180]
        # |errors|/|actual| = [10/100, 20/200] = [0.10, 0.10]
        # MAPE = 100 * mean([0.10, 0.10]) = 10.0 %
        actual = [100.0, 200.0]
        pred   = [110.0, 180.0]
        actual_s, _ = _make_actual_predicted(actual, pred)
        ev = evaluate_forecast(actual_s, _make_forecast_result(pred))
        assert ev.mape == pytest.approx(10.0, rel=1e-10)

    def test_mape_is_percentage_not_fraction(self) -> None:
        # 10% error → MAPE should be 10.0, not 0.10
        actual = [100.0]
        pred   = [110.0]
        actual_s, _ = _make_actual_predicted(actual, pred)
        ev = evaluate_forecast(actual_s, _make_forecast_result(pred))
        assert ev.mape == pytest.approx(10.0, rel=1e-10)

    def test_mape_non_negative(self) -> None:
        a = [50.0, 60.0]
        p = [48.0, 63.0]
        actual, _ = _make_actual_predicted(a, p)
        ev = evaluate_forecast(actual, _make_forecast_result(p))
        assert ev.mape is not None
        assert ev.mape >= 0.0


# ===========================================================================
# D  Zero-value handling
# ===========================================================================


class TestZeroActuals:
    def test_single_zero_excluded_from_mape(self) -> None:
        # actual=[0, 100, 200], pred=[1, 110, 180]
        # zero at index 0 → excluded; MAPE from [100,200] vs [110,180]
        # |errors|/|actual| = [10/100, 20/200] = [0.10, 0.10] → MAPE=10%
        actual = [0.0, 100.0, 200.0]
        pred   = [1.0, 110.0, 180.0]
        actual_s, _ = _make_actual_predicted(actual, pred)
        ev = evaluate_forecast(actual_s, _make_forecast_result(pred))
        assert ev.n_zero_actuals == 1
        assert ev.mape == pytest.approx(10.0, rel=1e-10)
        assert any("zero" in n.lower() for n in ev.notes)

    def test_all_zero_actuals_mape_is_none(self) -> None:
        actual = [0.0, 0.0, 0.0]
        pred   = [1.0, 2.0, 3.0]
        actual_s, _ = _make_actual_predicted(actual, pred)
        ev = evaluate_forecast(actual_s, _make_forecast_result(pred))
        assert ev.mape is None
        assert ev.n_zero_actuals == 3
        assert any("undefined" in n.lower() or "zero" in n.lower() for n in ev.notes)

    def test_rmse_mae_valid_even_with_zero_actuals(self) -> None:
        actual = [0.0, 0.0]
        pred   = [1.0, 2.0]
        actual_s, _ = _make_actual_predicted(actual, pred)
        ev = evaluate_forecast(actual_s, _make_forecast_result(pred))
        assert math.isfinite(ev.rmse)
        assert math.isfinite(ev.mae)

    def test_no_inf_nan_in_metrics_with_zero_actuals(self) -> None:
        actual = [0.0, 10.0]
        pred   = [1.0, 11.0]
        actual_s, _ = _make_actual_predicted(actual, pred)
        ev = evaluate_forecast(actual_s, _make_forecast_result(pred))
        assert math.isfinite(ev.rmse)
        assert math.isfinite(ev.mae)
        # mape computed over non-zero only: |1|/10 * 100 = 10%
        assert ev.mape == pytest.approx(10.0, rel=1e-10)


# ===========================================================================
# F + G + H + I  Input validation
# ===========================================================================


class TestInputValidation:
    def test_length_mismatch_raises(self) -> None:
        actual = pd.Series([1.0, 2.0, 3.0],
                           index=pd.bdate_range("2020-01-02", periods=3))
        fr = _make_forecast_result([1.0, 2.0])   # length 2 vs 3
        with pytest.raises(EvaluationError, match="length"):
            evaluate_forecast(actual, fr)

    def test_empty_actual_raises(self) -> None:
        actual = pd.Series([], index=pd.bdate_range("2020-01-02", periods=0),
                           dtype=float)
        fr = ForecastResult(
            model_name="ARIMA(1,1,0)",
            model_type="arima",
            config={},
            ticker="SYN",
            series_description="adjusted close price",
            frequency="1d",
            train_start=date(2020, 1, 2),
            train_end=date(2020, 9, 30),
            n_train=0,
            forecast_start=date(2020, 1, 2),
            forecast_end=date(2020, 1, 2),
            n_forecast=0,
            forecast_values={},
            fitted=True,
        )
        with pytest.raises(EvaluationError):
            evaluate_forecast(actual, fr)

    def test_index_mismatch_raises(self) -> None:
        # actual on Jan dates, forecast on Feb dates
        actual = pd.Series([10.0, 20.0],
                           index=pd.bdate_range("2020-01-02", periods=2))
        fr = _make_forecast_result([10.0, 20.0], start="2020-02-03")
        with pytest.raises(EvaluationError):
            evaluate_forecast(actual, fr)

    def test_nan_in_actual_raises(self) -> None:
        idx = pd.bdate_range("2020-10-01", periods=2, name="date")
        actual = pd.Series([float("nan"), 10.0], index=idx)
        fr = _make_forecast_result([10.0, 10.0], start="2020-10-01")
        with pytest.raises(EvaluationError, match="non-finite"):
            evaluate_forecast(actual, fr)

    def test_inf_in_predicted_raises(self) -> None:
        idx = pd.bdate_range("2020-10-01", periods=2, name="date")
        actual = pd.Series([10.0, 20.0], index=idx)
        # Inject inf via a ForecastResult with matching index
        fr = _make_forecast_result([float("inf"), 20.0], start="2020-10-01")
        with pytest.raises(EvaluationError, match="non-finite"):
            evaluate_forecast(actual, fr)

    def test_evaluation_error_is_forecasting_error(self) -> None:
        actual = pd.Series([1.0], index=pd.bdate_range("2020-01-02", periods=1))
        fr = _make_forecast_result([1.0, 2.0])
        with pytest.raises(ForecastingError):
            evaluate_forecast(actual, fr)


# ===========================================================================
# J + K  EvaluationResult object
# ===========================================================================


class TestEvaluationResult:
    def _sample(self) -> EvaluationResult:
        a = [100.0, 110.0, 105.0]
        p = [102.0, 108.0, 107.0]
        actual_s, _ = _make_actual_predicted(a, p)
        return evaluate_forecast(actual_s, _make_forecast_result(p))

    def test_model_name_propagated(self) -> None:
        assert self._sample().model_name == "ARIMA(1,1,0)"

    def test_model_type_propagated(self) -> None:
        assert self._sample().model_type == "arima"

    def test_ticker_propagated(self) -> None:
        assert self._sample().ticker == "SYN"

    def test_series_description_propagated(self) -> None:
        assert "close" in self._sample().series_description.lower()

    def test_frequency_propagated(self) -> None:
        assert self._sample().frequency == "1d"

    def test_n_observations_correct(self) -> None:
        assert self._sample().n_observations == 3

    def test_eval_start_before_eval_end(self) -> None:
        r = self._sample()
        assert r.eval_start <= r.eval_end

    def test_rmse_is_float(self) -> None:
        assert isinstance(self._sample().rmse, float)

    def test_mae_is_float(self) -> None:
        assert isinstance(self._sample().mae, float)

    def test_mape_is_float_or_none(self) -> None:
        mape = self._sample().mape
        assert mape is None or isinstance(mape, float)

    def test_to_dict_json_serialisable(self) -> None:
        d = self._sample().to_dict()
        assert json.loads(json.dumps(d)) == d

    def test_to_json_parseable(self) -> None:
        r = self._sample()
        parsed = json.loads(r.to_json())
        assert "model_name" in parsed
        assert "rmse" in parsed

    def test_notes_serialisable(self) -> None:
        d = self._sample().to_dict()
        assert isinstance(d["notes"], list)


# ===========================================================================
# L + M + N  ComparisonResult
# ===========================================================================


class TestComparisonResult:
    def _comparison(self) -> ComparisonResult:
        a  = [100.0, 110.0, 105.0, 112.0]
        p1 = [102.0, 108.0, 107.0, 110.0]
        p2 = [101.0, 111.0, 104.0, 113.0]
        actual_s, _ = _make_actual_predicted(a, p1)
        fr1 = _make_forecast_result(p1, model_name="ARIMA(1,1,0)", model_type="arima")
        fr2 = _make_forecast_result(p2, model_name="HoltWinters(trend=add)", model_type="holtwinters")
        return compare_models(actual_s, [fr1, fr2])

    def test_results_count(self) -> None:
        assert len(self._comparison().results) == 2

    def test_ticker_propagated(self) -> None:
        assert self._comparison().ticker == "SYN"

    def test_n_observations_correct(self) -> None:
        assert self._comparison().n_observations == 4

    def test_to_dict_json_serialisable(self) -> None:
        d = self._comparison().to_dict()
        assert json.loads(json.dumps(d)) == d

    def test_to_json_parseable(self) -> None:
        parsed = json.loads(self._comparison().to_json())
        assert "results" in parsed
        assert len(parsed["results"]) == 2

    def test_ranked_by_rmse_ascending(self) -> None:
        ranked = self._comparison().ranked_by("rmse")
        rmses = [r.rmse for r in ranked]
        assert rmses == sorted(rmses)

    def test_ranked_by_mae_ascending(self) -> None:
        ranked = self._comparison().ranked_by("mae")
        maes = [r.mae for r in ranked]
        assert maes == sorted(maes)

    def test_ranked_by_descending(self) -> None:
        ranked = self._comparison().ranked_by("rmse", ascending=False)
        rmses = [r.rmse for r in ranked]
        assert rmses == sorted(rmses, reverse=True)

    def test_ranked_by_invalid_metric_raises(self) -> None:
        with pytest.raises(ValueError):
            self._comparison().ranked_by("bad_metric")

    def test_empty_list_raises(self) -> None:
        actual = pd.Series([1.0], index=pd.bdate_range("2020-01-02", periods=1))
        with pytest.raises(EvaluationError):
            compare_models(actual, [])

    def test_no_winner_declared(self) -> None:
        """ranked_by returns a list — the caller decides what to do with it."""
        ranked = self._comparison().ranked_by("rmse")
        assert isinstance(ranked, list)
        # The ComparisonResult itself stores results but makes no declaration
        cmp = self._comparison()
        # No attribute named "best" or "winner"
        assert not hasattr(cmp, "best")
        assert not hasattr(cmp, "winner")

    def test_each_result_is_evaluation_result(self) -> None:
        for r in self._comparison().results:
            assert isinstance(r, EvaluationResult)


# ===========================================================================
# O  Deterministic repeated evaluation
# ===========================================================================


class TestDeterminism:
    def test_same_inputs_same_results(self) -> None:
        a = [100.0, 110.0, 120.0, 130.0]
        p = [101.0, 109.0, 121.0, 129.0]
        actual_s, _ = _make_actual_predicted(a, p)
        fr = _make_forecast_result(p)
        ev1 = evaluate_forecast(actual_s, fr)
        ev2 = evaluate_forecast(actual_s, fr)
        assert ev1.rmse == pytest.approx(ev2.rmse, rel=1e-15)
        assert ev1.mae == pytest.approx(ev2.mae, rel=1e-15)
        assert ev1.mape == pytest.approx(ev2.mape, rel=1e-15)  # type: ignore[arg-type]

    def test_rmse_from_compute_metrics_matches_evaluate(self) -> None:
        actual = np.array([100.0, 110.0, 120.0])
        pred   = np.array([102.0, 108.0, 122.0])
        rmse, mae, mape, n_zero, _ = _compute_metrics(actual, pred)
        # Hand-check: errors=[-2,2,-2], MSE=(4+4+4)/3=4, RMSE=2
        assert rmse == pytest.approx(2.0, rel=1e-10)
        assert mae  == pytest.approx(2.0, rel=1e-10)


# ===========================================================================
# P  Forecast figure
# ===========================================================================


class TestPlotForecastVsActual:
    def _setup(self) -> tuple[pd.Series, ForecastResult]:
        a = [100.0, 102.0, 101.0, 103.0, 102.0]
        p = [101.0, 101.5, 102.0, 102.5, 103.0]
        actual_s, _ = _make_actual_predicted(a, p)
        fr = _make_forecast_result(p)
        return actual_s, fr

    def test_returns_matplotlib_figure(self) -> None:
        actual, fr = self._setup()
        fig = plot_forecast_vs_actual(actual, fr)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_figure_has_two_panels(self) -> None:
        actual, fr = self._setup()
        fig = plot_forecast_vs_actual(actual, fr)
        assert len(fig.axes) == 2

    def test_title_contains_model_name(self) -> None:
        actual, fr = self._setup()
        fig = plot_forecast_vs_actual(actual, fr)
        assert "ARIMA" in fig.axes[0].get_title()

    def test_custom_title(self) -> None:
        actual, fr = self._setup()
        fig = plot_forecast_vs_actual(actual, fr, title="My custom title")
        assert fig.axes[0].get_title() == "My custom title"

    def test_custom_figsize(self) -> None:
        actual, fr = self._setup()
        fig = plot_forecast_vs_actual(actual, fr, figsize=(10, 8))
        assert fig.get_size_inches() == pytest.approx((10.0, 8.0))

    def test_metrics_in_title_when_show_metrics_true(self) -> None:
        actual, fr = self._setup()
        fig = plot_forecast_vs_actual(actual, fr, show_metrics=True)
        title = fig.axes[0].get_title()
        assert "RMSE" in title or "MAE" in title

    def test_no_metrics_when_show_metrics_false(self) -> None:
        actual, fr = self._setup()
        fig = plot_forecast_vs_actual(actual, fr, show_metrics=False, title=None)
        title = fig.axes[0].get_title()
        assert "RMSE" not in title


# ===========================================================================
# Q + R  Integration with real fitted models
# ===========================================================================


class TestIntegrationWithRealModels:
    def test_arima_evaluate_forecast(self) -> None:
        train = _make_price_series(160)
        test  = _make_price_series(20, seed=99)   # different seed for "actual"
        # Align test index to follow train
        test.index = pd.bdate_range(
            train.index[-1] + pd.offsets.BDay(1), periods=20, name="date"
        )
        model = ARIMAForecaster(order=(1, 1, 0), ticker="SYN")
        model.fit(train)
        fr = model.forecast(20)
        # Align actual to forecast index
        actual = pd.Series(
            test.values, index=fr.forecast_series().index, name="close"
        )
        ev = evaluate_forecast(actual, fr)
        assert isinstance(ev, EvaluationResult)
        assert ev.n_observations == 20
        assert math.isfinite(ev.rmse)
        assert math.isfinite(ev.mae)

    def test_compare_three_models(self) -> None:
        train = _make_price_series(160)
        fr_arima = ARIMAForecaster(order=(1, 1, 0), ticker="SYN").fit(train).forecast(10)
        fr_sarima = SARIMAForecaster(
            order=(1, 1, 0), seasonal_order=(0, 0, 0, 0), ticker="SYN"
        ).fit(train).forecast(10)
        fr_hw = HoltWintersForecaster(trend="add", ticker="SYN").fit(train).forecast(10)

        # Build actual on the same index as the ARIMA forecast
        fc_index = fr_arima.forecast_series().index
        actual = pd.Series(
            _make_price_series(10, seed=42).values, index=fc_index, name="close"
        )
        comparison = compare_models(actual, [fr_arima, fr_sarima, fr_hw])
        assert len(comparison.results) == 3
        ranked = comparison.ranked_by("rmse")
        assert len(ranked) == 3
        # ranked is ordered by RMSE ascending
        rmses = [r.rmse for r in ranked]
        assert rmses == sorted(rmses)

    def test_comparison_json_round_trip(self) -> None:
        train = _make_price_series(160)
        fr = ARIMAForecaster(order=(1, 1, 0), ticker="SYN").fit(train).forecast(10)
        fc_index = fr.forecast_series().index
        actual = pd.Series(
            _make_price_series(10, seed=42).values, index=fc_index, name="close"
        )
        cmp = compare_models(actual, [fr])
        d = cmp.to_dict()
        assert json.loads(json.dumps(d)) == d


# ===========================================================================
# S + T  Metadata and notes
# ===========================================================================


class TestMetadataAndNotes:
    def test_eval_start_matches_first_actual(self) -> None:
        a = [10.0, 20.0]
        p = [11.0, 19.0]
        actual_s, _ = _make_actual_predicted(a, p, start="2021-03-01")
        ev = evaluate_forecast(actual_s, _make_forecast_result(p, start="2021-03-01"))
        assert ev.eval_start == pd.Timestamp("2021-03-01").date()

    def test_eval_end_matches_last_actual(self) -> None:
        idx = pd.bdate_range("2021-03-01", periods=3)
        a = [10.0, 20.0, 30.0]
        p = [11.0, 19.0, 31.0]
        actual_s = pd.Series(a, index=idx)
        fr = _make_forecast_result(p, start="2021-03-01")
        ev = evaluate_forecast(actual_s, fr)
        assert ev.eval_end == pd.Timestamp(idx[-1]).date()

    def test_zero_actual_note_in_notes(self) -> None:
        a = [0.0, 100.0]
        p = [1.0, 102.0]
        actual_s, _ = _make_actual_predicted(a, p)
        ev = evaluate_forecast(actual_s, _make_forecast_result(p))
        note_text = " ".join(ev.notes).lower()
        assert "zero" in note_text

    def test_no_notes_when_no_issues(self) -> None:
        a = [100.0, 200.0]
        p = [101.0, 199.0]
        actual_s, _ = _make_actual_predicted(a, p)
        ev = evaluate_forecast(actual_s, _make_forecast_result(p))
        # No zero actuals → no notes needed
        assert ev.n_zero_actuals == 0

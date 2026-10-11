"""Tests for VaR calculations and leakage-safe forecast alignment."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src.preprocessing.return_series import PriceBasis, ReturnSeries
from src.risk.var import build_var_forecasts, historical_simulation_var, normal_var


def _returns(n: int = 253) -> ReturnSeries:
    index = pd.bdate_range("2020-01-02", periods=n, name="date")
    decimal = pd.Series(np.linspace(-0.01, 0.01, n), index=index, dtype="float64")
    gaps = pd.Series(np.zeros(n, dtype=bool), index=index)
    gaps.iloc[10] = True
    return ReturnSeries(
        ticker="SYN",
        frequency="1d",
        price_basis=PriceBasis.ADJUSTED,
        decimal=decimal,
        spans_gap=gaps,
        first_price_date=date(2020, 1, 1),
        gap_tolerance_weekdays=1,
        provisional_bar=None,
        provisional_bar_excluded=False,
        ingestion_rows_dropped=0,
        source="synthetic-test-data",
        fetched_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


def _stage9_forecasts(returns: ReturnSeries, positions: range) -> pd.DataFrame:
    index = returns.decimal.index[list(positions)]
    return pd.DataFrame({
        "garch_conditional_mean_pct": np.zeros(len(index)),
        "garch_variance_pct2": np.full(len(index), 1.0),
        "egarch_conditional_mean_pct": np.full(len(index), 0.1),
        "egarch_variance_pct2": np.full(len(index), 1.5),
    }, index=index)


class TestNormalVaR:
    @pytest.mark.parametrize("confidence", [0.95, 0.99])
    def test_formula_and_positive_loss_sign(self, confidence: float) -> None:
        mean, variance = 0.1, 4.0
        expected = -(mean + norm.ppf(1.0 - confidence) * np.sqrt(variance))
        value = normal_var(mean, variance, confidence)
        assert value == pytest.approx(expected)
        assert value > 0.0

    def test_rejects_nonpositive_variance(self) -> None:
        with pytest.raises(ValueError, match="positive and finite"):
            normal_var(0.0, 0.0, 0.95)


class TestHistoricalSimulationVaR:
    def test_quantile_method_and_sign(self) -> None:
        history = np.array([-3.0, -2.0, -1.0, 0.0, 1.0])
        expected = -np.quantile(history, 0.05, method="linear")
        assert historical_simulation_var(history, 0.95) == pytest.approx(expected)
        assert historical_simulation_var(history, 0.95) > 0.0

    def test_rejects_empty_or_nonfinite_history(self) -> None:
        with pytest.raises(ValueError, match="non-empty and finite"):
            historical_simulation_var([], 0.95)
        with pytest.raises(ValueError, match="non-empty and finite"):
            historical_simulation_var([0.0, np.nan], 0.95)


class TestBuildVaRForecasts:
    def test_units_alignment_gap_policy_and_breach_logic(self) -> None:
        returns = _returns()
        original = returns.decimal.copy(deep=True)
        source = _stage9_forecasts(returns, range(250, 253))
        result = build_var_forecasts(returns, source, historical_window=250)

        pd.testing.assert_index_equal(result.index, source.index)
        pd.testing.assert_series_equal(returns.decimal, original)
        assert result.iloc[0]["realised_return_pct"] == pytest.approx(returns.percent.iloc[250])
        assert result.iloc[0]["realised_loss_pct"] == pytest.approx(-returns.percent.iloc[250])
        assert result.iloc[0]["garch_forecast_std_pct"] == pytest.approx(1.0)
        assert result.iloc[0]["egarch_forecast_std_pct"] == pytest.approx(np.sqrt(1.5))
        assert result.iloc[0]["historical_window_n"] == 250
        assert result.iloc[0]["historical_window_start"] == returns.decimal.index[0]
        assert result.iloc[0]["historical_window_end"] == returns.decimal.index[249]
        assert result.iloc[0]["historical_window_gap_flagged_n"] == 1
        for prefix in ("garch", "egarch", "historical"):
            for level in (95, 99):
                expected_breach = result.iloc[0]["realised_loss_pct"] > result.iloc[0][f"{prefix}_var{level}_pct"]
                assert result.iloc[0][f"{prefix}_breach{level}"] is expected_breach
                assert np.isfinite(result.iloc[0][f"{prefix}_var{level}_pct"])
                assert result.iloc[0][f"{prefix}_var{level}_pct"] > 0

        expected_hs95 = -np.quantile(returns.percent.iloc[:250], 0.05, method="linear")
        assert result.iloc[0]["historical_var95_pct"] == pytest.approx(expected_hs95)

    def test_current_and_future_returns_do_not_change_first_var_thresholds(self) -> None:
        returns = _returns()
        source = _stage9_forecasts(returns, range(250, 253))
        original = build_var_forecasts(returns, source, historical_window=250)
        changed_values = returns.decimal.copy()
        changed_values.iloc[250:] = changed_values.iloc[250:] + 0.05
        changed_returns = replace(returns, decimal=changed_values)
        changed = build_var_forecasts(changed_returns, source, historical_window=250)
        for column in (
            "historical_var95_pct", "historical_var99_pct",
            "garch_var95_pct", "garch_var99_pct",
            "egarch_var95_pct", "egarch_var99_pct",
        ):
            assert original.iloc[0][column] == pytest.approx(changed.iloc[0][column])
        assert original.iloc[0]["realised_return_pct"] != changed.iloc[0]["realised_return_pct"]

    def test_missing_forecast_date_is_rejected(self) -> None:
        returns = _returns()
        source = _stage9_forecasts(returns, range(250, 252))
        source.index = source.index + pd.Timedelta(days=1)
        with pytest.raises(ValueError, match="align in order"):
            build_var_forecasts(returns, source, historical_window=250)

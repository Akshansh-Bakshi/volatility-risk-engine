"""Tests for the Stage 6 classical forecasting layer.

Coverage
--------
A. train/test chronological split
B. no leakage invariant
C. ARIMA fitting and forecast
D. ARIMA forecast length
E. SARIMA fitting and forecast
F. SARIMA forecast length
G. Holt-Winters fitting and forecast
H. forecast index correctness
I. ForecastResult metadata
J. invalid configuration
K. too-short series
L. deterministic repeated execution
M. JSON serialisation
N. model interface consistency (is_fitted, model_name, model_type)
O. no mutation of input series
P. forecast figure (plot_forecast)
Q. split info dict auditability

All tests are deterministic offline tests.
No Yahoo Finance, no network access.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone

import matplotlib.figure
import numpy as np
import pandas as pd
import pytest

from src.data.market_data import PRICE_COLUMN, MarketData, MarketDataRequest
from src.exceptions import (
    ForecastingDataError,
    ForecastingError,
    InvalidModelConfigError,
    ModelNotFittedError,
    VolatilityRiskEngineError,
)
from src.forecasting.arima import ARIMAForecaster
from src.forecasting.figures import plot_forecast
from src.forecasting.holtwinters import HoltWintersForecaster
from src.forecasting.result import ForecastResult
from src.forecasting.sarima import SARIMAForecaster
from src.forecasting.split import TimeSeriesSplit, make_split

# ---------------------------------------------------------------------------
# Synthetic data factories
# ---------------------------------------------------------------------------

_FETCHED_AT = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)


def _make_price_series(n: int = 300, start: str = "2020-01-02", seed: int = 0) -> pd.Series:
    """Deterministic random-walk price series on business days."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n, name="date")
    prices = 100.0 + np.cumsum(rng.normal(0.0, 0.5, n))
    # ensure all prices positive
    prices = prices - prices.min() + 50.0
    return pd.Series(prices, index=idx, name=PRICE_COLUMN)


def _make_market_data(n: int = 300) -> MarketData:
    """Minimal valid MarketData wrapping _make_price_series."""
    price = _make_price_series(n)
    df = price.to_frame()
    for col in ("open", "high", "low", "volume"):
        df[col] = price
    df = df[["open", "high", "low", "close", "volume"]]
    return MarketData(
        request=MarketDataRequest(ticker="SYN", start=date(2020, 1, 2)),
        prices=df,
        source="synthetic",
        fetched_at=_FETCHED_AT,
    )


def _train_series(n: int = 200) -> pd.Series:
    return _make_price_series(n)


# ===========================================================================
# A + B  TimeSeriesSplit / make_split
# ===========================================================================


class TestTimeSeriesSplit:
    def test_split_lengths_correct(self) -> None:
        md = _make_market_data(300)
        split = make_split(md, test_horizon=60)
        assert split.n_train == 240
        assert split.n_test == 60

    def test_no_leakage(self) -> None:
        """train end must be strictly before test start — no overlap."""
        split = make_split(_make_market_data(300), test_horizon=60)
        assert split.no_leakage()
        assert split.train_end < split.test_start

    def test_chronological_order_train(self) -> None:
        split = make_split(_make_market_data(200), test_horizon=40)
        assert split.train_series.index.is_monotonic_increasing

    def test_chronological_order_test(self) -> None:
        split = make_split(_make_market_data(200), test_horizon=40)
        assert split.test_series.index.is_monotonic_increasing

    def test_test_is_last_observations(self) -> None:
        md = _make_market_data(200)
        split = make_split(md, test_horizon=40)
        original = md.price
        pd.testing.assert_series_equal(
            split.test_series.reset_index(drop=True),
            original.iloc[-40:].reset_index(drop=True),
        )

    def test_train_is_first_observations(self) -> None:
        md = _make_market_data(200)
        split = make_split(md, test_horizon=40)
        original = md.price
        pd.testing.assert_series_equal(
            split.train_series.reset_index(drop=True),
            original.iloc[:160].reset_index(drop=True),
        )

    def test_train_plus_test_equals_total(self) -> None:
        md = _make_market_data(250)
        split = make_split(md, test_horizon=50)
        assert split.n_train + split.n_test == md.observations

    def test_ticker_propagated(self) -> None:
        split = make_split(_make_market_data(), test_horizon=30)
        assert split.ticker == "SYN"

    def test_info_dict_is_json_serialisable(self) -> None:
        split = make_split(_make_market_data(), test_horizon=30)
        d = split.info()
        assert json.loads(json.dumps(d)) == d

    def test_info_has_required_keys(self) -> None:
        split = make_split(_make_market_data(), test_horizon=30)
        info = split.info()
        for key in ("ticker", "train_start", "train_end", "n_train",
                    "test_start", "test_end", "n_test", "total_observations",
                    "split_method"):
            assert key in info

    def test_split_method_says_chronological(self) -> None:
        split = make_split(_make_market_data(), test_horizon=30)
        assert "chronological" in split.info()["split_method"].lower()

    def test_info_json_parseable(self) -> None:
        split = make_split(_make_market_data(), test_horizon=30)
        parsed = json.loads(split.info_json())
        assert parsed["n_test"] == 30

    def test_too_short_series_raises(self) -> None:
        md = _make_market_data(35)
        with pytest.raises(ForecastingDataError):
            make_split(md, test_horizon=10)   # 25 train < MIN_TRAIN=30

    def test_invalid_test_horizon_raises(self) -> None:
        md = _make_market_data(200)
        with pytest.raises(ForecastingDataError):
            make_split(md, test_horizon=0)

    def test_float_horizon_raises(self) -> None:
        md = _make_market_data(200)
        with pytest.raises(ForecastingDataError):
            make_split(md, test_horizon=10.5)  # type: ignore[arg-type]

    def test_frequency_propagated(self) -> None:
        split = make_split(_make_market_data(), test_horizon=30)
        assert split.frequency == "1d"

    def test_no_mutation_of_source(self) -> None:
        md = _make_market_data(200)
        original_values = md.price.values.copy()
        make_split(md, test_horizon=40)
        np.testing.assert_array_equal(md.price.values, original_values)


# ===========================================================================
# C + D  ARIMAForecaster
# ===========================================================================


class TestARIMAForecaster:
    def _fitted(self, n: int = 200, order: tuple = (1, 1, 0)) -> ARIMAForecaster:
        m = ARIMAForecaster(order=order, ticker="SYN")
        m.fit(_train_series(n))
        return m

    def test_is_fitted_false_before_fit(self) -> None:
        assert not ARIMAForecaster().is_fitted

    def test_is_fitted_true_after_fit(self) -> None:
        assert self._fitted().is_fitted

    def test_model_name_format(self) -> None:
        m = ARIMAForecaster(order=(2, 1, 1))
        assert m.model_name == "ARIMA(2,1,1)"

    def test_model_type_in_result(self) -> None:
        r = self._fitted().forecast(10)
        assert r.model_type == "arima"

    def test_forecast_length_correct(self) -> None:
        r = self._fitted().forecast(20)
        assert r.n_forecast == 20
        assert len(r.forecast_values) == 20

    def test_forecast_series_length(self) -> None:
        r = self._fitted().forecast(15)
        assert len(r.forecast_series()) == 15

    def test_forecast_starts_after_train_end(self) -> None:
        m = self._fitted()
        r = m.forecast(10)
        assert r.forecast_start > r.train_end

    def test_aic_is_float(self) -> None:
        r = self._fitted().forecast(10)
        assert isinstance(r.aic, float)

    def test_bic_is_float(self) -> None:
        r = self._fitted().forecast(10)
        assert isinstance(r.bic, float)

    def test_fitted_flag_true(self) -> None:
        assert self._fitted().forecast(10).fitted is True

    def test_ticker_in_result(self) -> None:
        m = ARIMAForecaster(ticker="MYSTOCK")
        m.fit(_train_series(200))
        assert m.forecast(5).ticker == "MYSTOCK"

    def test_series_description_in_result(self) -> None:
        r = self._fitted().forecast(5)
        assert "close" in r.series_description.lower()

    def test_config_has_pqd(self) -> None:
        m = ARIMAForecaster(order=(2, 1, 1))
        m.fit(_train_series(200))
        cfg = m.forecast(5).config
        assert cfg == {"p": 2, "d": 1, "q": 1}

    def test_n_train_correct(self) -> None:
        train = _train_series(180)
        m = ARIMAForecaster()
        m.fit(train)
        r = m.forecast(10)
        assert r.n_train == 180

    def test_deterministic_repeated_execution(self) -> None:
        train = _train_series(200)
        m1 = ARIMAForecaster(order=(1, 1, 0), ticker="SYN")
        m1.fit(train)
        r1 = m1.forecast(10)
        m2 = ARIMAForecaster(order=(1, 1, 0), ticker="SYN")
        m2.fit(train)
        r2 = m2.forecast(10)
        assert r1.forecast_values == pytest.approx(r2.forecast_values, rel=1e-10)

    def test_json_round_trip(self) -> None:
        r = self._fitted().forecast(10)
        d = r.to_dict()
        assert json.loads(json.dumps(d)) == d

    def test_no_mutation_of_input(self) -> None:
        train = _train_series(200)
        original = train.values.copy()
        m = ARIMAForecaster()
        m.fit(train)
        m.forecast(10)
        np.testing.assert_array_equal(train.values, original)

    def test_forecast_before_fit_raises(self) -> None:
        with pytest.raises(ModelNotFittedError):
            ARIMAForecaster().forecast(10)

    def test_invalid_order_negative_raises(self) -> None:
        with pytest.raises(InvalidModelConfigError):
            ARIMAForecaster(order=(-1, 1, 0))

    def test_invalid_order_float_raises(self) -> None:
        with pytest.raises(InvalidModelConfigError):
            ARIMAForecaster(order=(1.0, 1, 0))  # type: ignore[arg-type]

    def test_invalid_order_wrong_length_raises(self) -> None:
        with pytest.raises(InvalidModelConfigError):
            ARIMAForecaster(order=(1, 1))  # type: ignore[arg-type]

    def test_too_short_series_raises(self) -> None:
        m = ARIMAForecaster()
        with pytest.raises(ForecastingDataError):
            m.fit(_train_series(5))

    def test_zero_horizon_raises(self) -> None:
        m = self._fitted()
        with pytest.raises(ForecastingDataError):
            m.forecast(0)

    def test_float_horizon_raises(self) -> None:
        m = self._fitted()
        with pytest.raises(ForecastingDataError):
            m.forecast(5.5)  # type: ignore[arg-type]

    def test_model_not_fitted_is_forecasting_error(self) -> None:
        with pytest.raises(ForecastingError):
            ARIMAForecaster().forecast(5)

    def test_invalid_config_is_forecasting_error(self) -> None:
        with pytest.raises(ForecastingError):
            ARIMAForecaster(order=(-1, 0, 0))

    def test_forecast_values_are_floats(self) -> None:
        r = self._fitted().forecast(5)
        for v in r.forecast_values.values():
            assert isinstance(v, float)

    def test_forecast_dates_are_iso_strings(self) -> None:
        r = self._fitted().forecast(5)
        for k in r.forecast_values:
            # should be parseable as a date
            date.fromisoformat(k)


# ===========================================================================
# E + F  SARIMAForecaster
# ===========================================================================


class TestSARIMAForecaster:
    def _fitted(self, seasonal_order: tuple = (0, 0, 0, 0)) -> SARIMAForecaster:
        m = SARIMAForecaster(
            order=(1, 1, 0),
            seasonal_order=seasonal_order,
            ticker="SYN",
        )
        m.fit(_train_series(200))
        return m

    def test_is_fitted_after_fit(self) -> None:
        assert self._fitted().is_fitted

    def test_model_name_no_seasonal(self) -> None:
        m = SARIMAForecaster(order=(1, 1, 0), seasonal_order=(0, 0, 0, 0))
        assert "SARIMA" in m.model_name

    def test_model_name_with_seasonal(self) -> None:
        m = SARIMAForecaster(order=(1, 1, 0), seasonal_order=(1, 0, 0, 5))
        assert "(1,0,0,5)" in m.model_name

    def test_model_type_is_sarima(self) -> None:
        r = self._fitted().forecast(10)
        assert r.model_type == "sarima"

    def test_forecast_length_correct(self) -> None:
        r = self._fitted().forecast(20)
        assert r.n_forecast == 20
        assert len(r.forecast_values) == 20

    def test_forecast_starts_after_train_end(self) -> None:
        r = self._fitted().forecast(10)
        assert r.forecast_start > r.train_end

    def test_config_has_all_keys(self) -> None:
        r = SARIMAForecaster(
            order=(1, 1, 0), seasonal_order=(1, 0, 0, 5), ticker="SYN"
        ).fit(_train_series(200)).forecast(5)
        for k in ("p", "d", "q", "P", "D", "Q", "s"):
            assert k in r.config

    def test_aic_bic_available(self) -> None:
        r = self._fitted().forecast(5)
        assert isinstance(r.aic, float)
        assert isinstance(r.bic, float)

    def test_json_round_trip(self) -> None:
        r = self._fitted().forecast(10)
        d = r.to_dict()
        assert json.loads(json.dumps(d)) == d

    def test_no_mutation_of_input(self) -> None:
        train = _train_series(200)
        original = train.values.copy()
        SARIMAForecaster(order=(1, 1, 0), seasonal_order=(0, 0, 0, 0)).fit(train).forecast(5)
        np.testing.assert_array_equal(train.values, original)

    def test_invalid_seasonal_period_zero_raises(self) -> None:
        with pytest.raises(InvalidModelConfigError):
            SARIMAForecaster(order=(1, 1, 0), seasonal_order=(1, 0, 0, 0))

    def test_invalid_seasonal_period_one_raises(self) -> None:
        with pytest.raises(InvalidModelConfigError):
            SARIMAForecaster(order=(1, 1, 0), seasonal_order=(1, 0, 0, 1))

    def test_invalid_seasonal_negative_raises(self) -> None:
        with pytest.raises(InvalidModelConfigError):
            SARIMAForecaster(order=(1, 1, 0), seasonal_order=(-1, 0, 0, 5))

    def test_invalid_order_raises(self) -> None:
        with pytest.raises(InvalidModelConfigError):
            SARIMAForecaster(order=(-1, 1, 0), seasonal_order=(0, 0, 0, 0))

    def test_forecast_before_fit_raises(self) -> None:
        with pytest.raises(ModelNotFittedError):
            SARIMAForecaster().forecast(5)

    def test_too_short_series_raises(self) -> None:
        m = SARIMAForecaster()
        with pytest.raises(ForecastingDataError):
            m.fit(_train_series(5))

    def test_deterministic(self) -> None:
        train = _train_series(200)
        r1 = SARIMAForecaster(order=(1, 1, 0), seasonal_order=(0, 0, 0, 0)).fit(train).forecast(10)
        r2 = SARIMAForecaster(order=(1, 1, 0), seasonal_order=(0, 0, 0, 0)).fit(train).forecast(10)
        assert r1.forecast_values == pytest.approx(r2.forecast_values, rel=1e-10)


# ===========================================================================
# G  HoltWintersForecaster
# ===========================================================================


class TestHoltWintersForecaster:
    def _fitted(self, trend: str | None = "add", seasonal: str | None = None) -> HoltWintersForecaster:
        m = HoltWintersForecaster(trend=trend, seasonal=seasonal, ticker="SYN")  # type: ignore[arg-type]
        m.fit(_train_series(200))
        return m

    def test_is_fitted_after_fit(self) -> None:
        assert self._fitted().is_fitted

    def test_model_name_contains_trend_type(self) -> None:
        assert "trend=add" in self._fitted(trend="add").model_name

    def test_model_name_no_trend(self) -> None:
        assert "trend=none" in self._fitted(trend=None).model_name

    def test_model_type_is_holtwinters(self) -> None:
        r = self._fitted().forecast(10)
        assert r.model_type == "holtwinters"

    def test_forecast_length_correct(self) -> None:
        r = self._fitted().forecast(20)
        assert r.n_forecast == 20
        assert len(r.forecast_values) == 20

    def test_forecast_starts_after_train_end(self) -> None:
        r = self._fitted().forecast(10)
        assert r.forecast_start > r.train_end

    def test_config_has_expected_keys(self) -> None:
        r = self._fitted().forecast(5)
        cfg = r.config
        assert "trend" in cfg
        assert "seasonal" in cfg
        assert "damped_trend" in cfg

    def test_aic_bic_available(self) -> None:
        r = self._fitted().forecast(5)
        assert isinstance(r.aic, float)
        assert isinstance(r.bic, float)

    def test_json_round_trip(self) -> None:
        r = self._fitted().forecast(10)
        d = r.to_dict()
        assert json.loads(json.dumps(d)) == d

    def test_no_mutation_of_input(self) -> None:
        train = _train_series(200)
        original = train.values.copy()
        HoltWintersForecaster(trend="add", seasonal=None).fit(train).forecast(5)
        np.testing.assert_array_equal(train.values, original)

    def test_damped_trend_requires_trend_raises(self) -> None:
        with pytest.raises(InvalidModelConfigError):
            HoltWintersForecaster(trend=None, damped_trend=True)

    def test_invalid_trend_string_raises(self) -> None:
        with pytest.raises(InvalidModelConfigError):
            HoltWintersForecaster(trend="bad")  # type: ignore[arg-type]

    def test_invalid_seasonal_string_raises(self) -> None:
        with pytest.raises(InvalidModelConfigError):
            HoltWintersForecaster(trend="add", seasonal="xyz")  # type: ignore[arg-type]

    def test_seasonal_without_period_raises(self) -> None:
        with pytest.raises(InvalidModelConfigError):
            HoltWintersForecaster(trend="add", seasonal="add", seasonal_periods=None)

    def test_seasonal_period_one_raises(self) -> None:
        with pytest.raises(InvalidModelConfigError):
            HoltWintersForecaster(trend="add", seasonal="add", seasonal_periods=1)

    def test_forecast_before_fit_raises(self) -> None:
        with pytest.raises(ModelNotFittedError):
            HoltWintersForecaster().forecast(5)

    def test_too_short_series_raises(self) -> None:
        m = HoltWintersForecaster(trend="add", seasonal=None)
        with pytest.raises(ForecastingDataError):
            m.fit(_train_series(5))

    def test_seasonal_needs_two_cycles(self) -> None:
        """Model with s=5 needs at least 10 training observations."""
        m = HoltWintersForecaster(trend="add", seasonal="add", seasonal_periods=5)
        # 8 obs < 2*5 = 10
        short = _train_series(8)
        # Already fails min-obs check (8 < 20) — either error is ForecastingDataError
        with pytest.raises(ForecastingDataError):
            m.fit(short)

    def test_deterministic(self) -> None:
        train = _train_series(200)
        r1 = HoltWintersForecaster(trend="add").fit(train).forecast(10)
        r2 = HoltWintersForecaster(trend="add").fit(train).forecast(10)
        assert r1.forecast_values == pytest.approx(r2.forecast_values, rel=1e-8)

    def test_model_with_damped_trend(self) -> None:
        m = HoltWintersForecaster(trend="add", damped_trend=True, ticker="SYN")
        m.fit(_train_series(200))
        r = m.forecast(10)
        assert "damped" in r.model_name.lower()


# ===========================================================================
# H  Forecast index correctness
# ===========================================================================


class TestForecastIndex:
    def test_arima_forecast_values_keys_are_date_strings(self) -> None:
        r = ARIMAForecaster(order=(1, 1, 0)).fit(_train_series(200)).forecast(5)
        for k in r.forecast_values.keys():
            date.fromisoformat(k)   # must not raise

    def test_sarima_forecast_values_keys_are_date_strings(self) -> None:
        r = SARIMAForecaster().fit(_train_series(200)).forecast(5)
        for k in r.forecast_values.keys():
            date.fromisoformat(k)

    def test_hw_forecast_values_keys_are_date_strings(self) -> None:
        r = HoltWintersForecaster(trend="add").fit(_train_series(200)).forecast(5)
        for k in r.forecast_values.keys():
            date.fromisoformat(k)

    def test_forecast_series_has_datetime_index(self) -> None:
        r = ARIMAForecaster().fit(_train_series(200)).forecast(5)
        assert isinstance(r.forecast_series().index, pd.DatetimeIndex)


# ===========================================================================
# I  ForecastResult metadata
# ===========================================================================


class TestForecastResult:
    def _sample(self) -> ForecastResult:
        return ARIMAForecaster(order=(1, 1, 0), ticker="SYN").fit(_train_series(200)).forecast(10)

    def test_model_name_is_string(self) -> None:
        assert isinstance(self._sample().model_name, str)

    def test_model_type_is_string(self) -> None:
        assert isinstance(self._sample().model_type, str)

    def test_ticker_correct(self) -> None:
        assert self._sample().ticker == "SYN"

    def test_series_description_present(self) -> None:
        assert self._sample().series_description

    def test_frequency_is_1d(self) -> None:
        assert self._sample().frequency == "1d"

    def test_train_start_before_train_end(self) -> None:
        r = self._sample()
        assert r.train_start <= r.train_end

    def test_forecast_start_before_forecast_end(self) -> None:
        r = self._sample()
        assert r.forecast_start <= r.forecast_end

    def test_n_forecast_matches_values(self) -> None:
        r = self._sample()
        assert r.n_forecast == len(r.forecast_values)

    def test_to_dict_json_serialisable(self) -> None:
        d = self._sample().to_dict()
        assert json.loads(json.dumps(d)) == d

    def test_to_json_parseable(self) -> None:
        parsed = json.loads(self._sample().to_json())
        assert "model_name" in parsed

    def test_notes_is_serialisable(self) -> None:
        r = self._sample()
        d = r.to_dict()
        assert isinstance(d["notes"], list)


# ===========================================================================
# N  Interface consistency across all three models
# ===========================================================================


class TestModelInterfaceConsistency:
    """All three model families must expose the same contract."""

    @pytest.fixture(params=["arima", "sarima", "hw"])
    def fitted_model(self, request: pytest.FixtureRequest):  # type: ignore[type-arg]
        train = _train_series(200)
        if request.param == "arima":
            m = ARIMAForecaster(order=(1, 1, 0), ticker="SYN")
        elif request.param == "sarima":
            m = SARIMAForecaster(order=(1, 1, 0), seasonal_order=(0, 0, 0, 0), ticker="SYN")
        else:
            m = HoltWintersForecaster(trend="add", ticker="SYN")
        m.fit(train)
        return m

    def test_is_fitted_true(self, fitted_model) -> None:  # type: ignore[no-untyped-def]
        assert fitted_model.is_fitted

    def test_model_name_is_string(self, fitted_model) -> None:  # type: ignore[no-untyped-def]
        assert isinstance(fitted_model.model_name, str)

    def test_forecast_returns_forecast_result(self, fitted_model) -> None:  # type: ignore[no-untyped-def]
        r = fitted_model.forecast(5)
        assert isinstance(r, ForecastResult)

    def test_forecast_result_is_fitted(self, fitted_model) -> None:  # type: ignore[no-untyped-def]
        assert fitted_model.forecast(5).fitted is True

    def test_forecast_horizon_correct(self, fitted_model) -> None:  # type: ignore[no-untyped-def]
        r = fitted_model.forecast(7)
        assert r.n_forecast == 7

    def test_json_serialisable(self, fitted_model) -> None:  # type: ignore[no-untyped-def]
        d = fitted_model.forecast(5).to_dict()
        assert json.loads(json.dumps(d)) == d

    def test_model_type_is_known_string(self, fitted_model) -> None:  # type: ignore[no-untyped-def]
        r = fitted_model.forecast(5)
        assert r.model_type in ("arima", "sarima", "holtwinters")


# ===========================================================================
# P  Forecast figure
# ===========================================================================


class TestPlotForecast:
    def _result(self) -> ForecastResult:
        return ARIMAForecaster(order=(1, 1, 0), ticker="SYN").fit(_train_series(200)).forecast(20)

    def test_returns_matplotlib_figure(self) -> None:
        fig = plot_forecast(self._result())
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_figure_has_one_axes(self) -> None:
        assert len(plot_forecast(self._result()).axes) == 1

    def test_title_contains_model_name(self) -> None:
        r = self._result()
        fig = plot_forecast(r)
        assert "ARIMA" in fig.axes[0].get_title()

    def test_title_contains_ticker(self) -> None:
        r = self._result()
        fig = plot_forecast(r)
        assert "SYN" in fig.axes[0].get_title()

    def test_custom_title(self) -> None:
        fig = plot_forecast(self._result(), title="My Title")
        assert fig.axes[0].get_title() == "My Title"

    def test_custom_figsize(self) -> None:
        fig = plot_forecast(self._result(), figsize=(8, 4))
        assert fig.get_size_inches() == pytest.approx((8.0, 4.0))

    def test_with_actual_test_series(self) -> None:
        r = self._result()
        actual = _train_series(20)
        fig = plot_forecast(r, actual_test=actual)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_with_train_series(self) -> None:
        r = self._result()
        train = _train_series(200)
        fig = plot_forecast(r, train_series=train)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_with_all_series(self) -> None:
        r = self._result()
        fig = plot_forecast(
            r,
            actual_test=_train_series(20),
            train_series=_train_series(200),
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_hw_result_also_plots(self) -> None:
        r = HoltWintersForecaster(trend="add", ticker="SYN").fit(_train_series(200)).forecast(10)
        fig = plot_forecast(r)
        assert isinstance(fig, matplotlib.figure.Figure)

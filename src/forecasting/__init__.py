"""Classical time-series forecasting benchmark layer.

This package provides classical forecasting models (ARIMA, SARIMA, Holt-Winters)
as required by the university Phase 2 curriculum, and a chronological
train/test splitting utility.

Architecture
------------
The dependency order is strictly::

    data → preprocessing → statistics → forecasting

This package:
- consumes validated :class:`~src.data.market_data.MarketData` (specifically
  the ``close`` price series — see "Target series" below).
- never imports ``yfinance`` directly.
- contains no Streamlit code.
- does not fit GARCH, EGARCH or any volatility model.

Target series
-------------
Classical forecasting models in this package operate on the **adjusted
close-price level** (``MarketData.price``), *not* on log returns.

Rationale: ARIMA, SARIMA and Holt-Winters are designed for the level of an
observable time series.  Log-return volatility modelling (GARCH/EGARCH) is
a separate concern addressed in a later stage.  Mixing these targets would
confuse the two distinct analytical goals.

Stage 6 — Classical Forecasting Benchmarks
-------------------------------------------
Public API::

    from src.forecasting.split import TimeSeriesSplit, make_split
    from src.forecasting.result import ForecastResult
    from src.forecasting.arima import ARIMAForecaster
    from src.forecasting.sarima import SARIMAForecaster
    from src.forecasting.holtwinters import HoltWintersForecaster
    from src.forecasting.figures import plot_forecast
"""

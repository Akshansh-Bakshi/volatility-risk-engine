"""Volatility model fitting and one-step forecasting engine (Stages 8–9).

This package provides GARCH(1,1) and EGARCH(1,1) wrappers around the
``arch`` library.  It consumes :class:`~src.preprocessing.return_series.ReturnSeries`
(percent-scale log returns) and produces frozen, JSON-serialisable
:class:`~src.models.result.ModelFitResult` objects.

Architecture position
---------------------
This package sits directly above the statistics layer::

    data → preprocessing → statistics → [forecasting] → models → risk → backtesting

It knows nothing about ``yfinance``, Streamlit or raw provider objects.

Return-scale convention
-----------------------
Models are fitted on **percent-scale** log returns (``ReturnSeries.percent``).
The ``arch`` library is tuned for data whose standard deviation is of
order 1 (not 0.01), so decimal-scale returns produce numerically equivalent
but less inspectable parameter values.

The model wrappers provide in-sample fit summaries and genuine one-step-ahead
conditional variance forecasts. :mod:`src.models.walk_forward` adds the
expanding-window experiment orchestration and evaluation.

Public API (Stage 8)
--------------------
::

    from src.models.result import ModelFitResult
    from src.models.result import VarianceForecast
    from src.models.garch import GARCHModel
    from src.models.egarch import EGARCHModel
    from src.models.walk_forward import run_expanding_volatility_forecast
    from src.models.figures import plot_conditional_volatility, plot_volatility_comparison
"""

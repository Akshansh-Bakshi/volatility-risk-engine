"""Volatility Analytics & Market Risk Engine.

The package is organised as a one-directional pipeline (see the README for the
architecture): ``data`` -> ``preprocessing`` -> ``statistics`` -> ``models`` ->
``forecasting`` -> ``risk`` -> ``backtesting``.  Cross-cutting concerns
(``config``, ``logging_config``, ``exceptions``, ``utils``) sit underneath all
of them and never depend on a pipeline stage.
"""

__version__ = "0.1.0"

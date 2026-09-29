"""Project-wide exception hierarchy.

Every error this package raises deliberately derives from
:class:`VolatilityRiskEngineError`.  Callers, including the UI layer, can
therefore handle expected failures without masking unrelated programming
errors.  Domain-specific subclasses are added next to the code that raises them.
"""

from __future__ import annotations


class VolatilityRiskEngineError(Exception):
    """Base class for all errors raised deliberately by this package."""


class ConfigurationError(VolatilityRiskEngineError):
    """Raised when configuration values or logging setup are invalid."""


class DataError(VolatilityRiskEngineError):
    """Base class for failures while obtaining or validating market data."""


class DataFetchError(DataError):
    """The data provider could not be reached or failed while serving the request."""


class InvalidTickerError(DataError):
    """The provider does not recognise the requested symbol."""


class EmptyDataError(DataError):
    """The provider returned no rows, or no row carrying a usable price."""


class InsufficientHistoryError(DataError):
    """Fewer usable observations are available than the configured minimum."""


class DataValidationError(DataError):
    """A request, provider output or cached entry violates the market data contract."""


class PreprocessingError(VolatilityRiskEngineError):
    """Base class for failures while turning validated market data into model inputs."""


class InvalidReturnSeriesError(PreprocessingError):
    """Returns cannot be computed safely, or a return series violates its contract.

    Raised when prices are individually valid but produce non-finite log returns
    (for example an overflowing price ratio), and when a
    :class:`~src.preprocessing.return_series.ReturnSeries` is constructed with
    non-finite values, misaligned gap flags or inconsistent metadata.
    """


class StatisticsError(VolatilityRiskEngineError):
    """Base class for failures in the statistical diagnostics layer."""


class InvalidProfileInputError(StatisticsError):
    """The inputs supplied to the EDA layer are invalid or inconsistent."""


class InsufficientDataError(StatisticsError):
    """The series is too short to run a requested statistical test."""


class ForecastingError(VolatilityRiskEngineError):
    """Base class for failures in the classical forecasting layer."""


class InvalidModelConfigError(ForecastingError):
    """A model order, parameter or configuration value is invalid."""


class ModelNotFittedError(ForecastingError):
    """A forecast was requested before the model was fitted."""


class ForecastingDataError(ForecastingError):
    """The input series supplied to the forecasting layer is invalid or too short."""


class EvaluationError(ForecastingError):
    """Raised when forecast evaluation cannot proceed due to bad inputs.

    Examples: actual/predicted length mismatch, index misalignment,
    empty series, or non-finite values that prevent metric computation.
    """


class VolatilityModelError(VolatilityRiskEngineError):
    """Base class for failures in the volatility model fitting layer."""


class VolatilityModelConfigError(VolatilityModelError):
    """Invalid model configuration or parameter specification."""


class VolatilityModelFitError(VolatilityModelError):
    """The model optimizer failed to converge or produced non-finite outputs."""


class VolatilityModelDataError(VolatilityModelError):
    """The return series supplied to the volatility model is invalid or too short."""

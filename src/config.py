"""Central, validated configuration.

Defaults live in this module as constants and frozen dataclasses.  Every
default can be overridden through ``VRE_*`` environment variables (see
``.env.example``), which keeps the code free of hardcoded deployment details
and free of secrets.

Design notes:

* Configuration objects are immutable and validate themselves on construction,
  so an invalid :class:`Settings` instance can never exist.
* ``end_date=None`` means "up to the latest available observation".  It is
  resolved when data is requested, never at import time, so behaviour does not
  silently depend on the day the process started.
* :func:`load_settings` is pure with respect to the mapping it is given, which
  makes it trivial to test; :func:`get_settings` is the cached, process-wide
  entry point used by the application.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.exceptions import ConfigurationError

ENV_PREFIX = "VRE_"

VALID_LOG_LEVELS: tuple[str, ...] = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_TICKER = "^GSPC"
DEFAULT_START_DATE = date(2005, 1, 1)
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_ADJUST_PRICES = True
DEFAULT_MIN_OBSERVATIONS = 250
DEFAULT_USE_CACHE = True
DEFAULT_CACHE_MAX_AGE_HOURS = 6.0
DEFAULT_INCLUDE_PROVISIONAL_BAR = False
DEFAULT_MAX_GAP_WEEKDAYS = 1
DEFAULT_MIN_RETURNS = 250
DEFAULT_CONFIDENCE_LEVELS: tuple[float, ...] = (0.95, 0.99)
DEFAULT_LOG_LEVEL = "INFO"

_DATE_FORMAT = "%Y-%m-%d"


def _require_bool(section: str, name: str, value: object) -> None:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{section}.{name} must be a boolean; got {value!r}.")


def _require_int(section: str, name: str, value: object, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigurationError(
            f"{section}.{name} must be an integer >= {minimum}; got {value!r}."
        )


@dataclass(frozen=True)
class DataConfig:
    """Defaults for market data requests and for how the data layer treats the result.

    Attributes:
        ticker: Data-provider symbol, normalised to upper case.
        start_date: First calendar date requested (inclusive).
        end_date: Last calendar date requested (inclusive), or ``None`` for
            "latest available", resolved each time data is fetched.
        data_dir: Root directory for generated datasets (git-ignored by default).
            Relative paths are resolved against the working directory.
        adjust_prices: ``True`` requests prices adjusted for splits and dividends
            (total-return basis); ``False`` requests split-adjusted prices only.
        min_observations: Minimum number of usable price observations required
            for a download to be accepted.
        use_cache: Whether downloads are cached on disk under ``data_dir``.
        cache_max_age_hours: How long a cached "latest" download counts as current.
    """

    ticker: str = DEFAULT_TICKER
    start_date: date = DEFAULT_START_DATE
    end_date: date | None = None
    data_dir: Path = DEFAULT_DATA_DIR
    adjust_prices: bool = DEFAULT_ADJUST_PRICES
    min_observations: int = DEFAULT_MIN_OBSERVATIONS
    use_cache: bool = DEFAULT_USE_CACHE
    cache_max_age_hours: float = DEFAULT_CACHE_MAX_AGE_HOURS

    def __post_init__(self) -> None:
        ticker = self.ticker.strip().upper() if isinstance(self.ticker, str) else ""
        if not ticker:
            raise ConfigurationError("data.ticker must be a non-empty string.")
        object.__setattr__(self, "ticker", ticker)

        if self.end_date is not None and self.start_date >= self.end_date:
            raise ConfigurationError(
                "data.start_date must be strictly earlier than data.end_date "
                f"(got {self.start_date.isoformat()} and {self.end_date.isoformat()})."
            )

        object.__setattr__(self, "data_dir", Path(self.data_dir).expanduser())
        _require_bool("data", "adjust_prices", self.adjust_prices)
        _require_bool("data", "use_cache", self.use_cache)
        _require_int("data", "min_observations", self.min_observations, minimum=2)
        max_age = self.cache_max_age_hours
        if (
            isinstance(max_age, bool)
            or not isinstance(max_age, (int, float))
            or not math.isfinite(max_age)
            or max_age <= 0
        ):
            raise ConfigurationError(
                f"data.cache_max_age_hours must be a positive, finite number; got {max_age!r}."
            )
        object.__setattr__(self, "cache_max_age_hours", float(max_age))


@dataclass(frozen=True)
class PreprocessingConfig:
    """How validated prices become a return series.

    Attributes:
        include_provisional_bar: Whether a latest bar that may still be forming
            (see :mod:`src.preprocessing.returns`) is kept.  ``False`` excludes it,
            so only finalised observations reach the modelling layers.
        max_gap_weekdays: Number of consecutive weekdays without an observation
            that a return may span before it is flagged as spanning a gap.  ``1``
            tolerates a single closure (an ordinary exchange holiday, which is
            indistinguishable from one missing session without a trading calendar).
        min_returns: Minimum number of returns required to build a return series.
    """

    include_provisional_bar: bool = DEFAULT_INCLUDE_PROVISIONAL_BAR
    max_gap_weekdays: int = DEFAULT_MAX_GAP_WEEKDAYS
    min_returns: int = DEFAULT_MIN_RETURNS

    def __post_init__(self) -> None:
        _require_bool("preprocessing", "include_provisional_bar", self.include_provisional_bar)
        _require_int("preprocessing", "max_gap_weekdays", self.max_gap_weekdays, minimum=0)
        _require_int("preprocessing", "min_returns", self.min_returns, minimum=2)


@dataclass(frozen=True)
class RiskConfig:
    """Defaults for risk measures.

    Attributes:
        confidence_levels: VaR confidence levels as probabilities strictly
            between 0 and 1 (``0.99`` means 99%).  Stored sorted and unique.
    """

    confidence_levels: tuple[float, ...] = DEFAULT_CONFIDENCE_LEVELS

    def __post_init__(self) -> None:
        try:
            levels = tuple(sorted({float(level) for level in self.confidence_levels}))
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"risk.confidence_levels must be numeric; got {self.confidence_levels!r}."
            ) from exc
        if not levels:
            raise ConfigurationError("risk.confidence_levels must not be empty.")
        invalid = [level for level in levels if not 0.0 < level < 1.0]
        if invalid:
            raise ConfigurationError(
                "risk.confidence_levels must be probabilities strictly between 0 and 1 "
                f"(e.g. 0.99 for 99%); got invalid value(s): {invalid}."
            )
        object.__setattr__(self, "confidence_levels", levels)


@dataclass(frozen=True)
class LoggingConfig:
    """Logging behaviour.

    Attributes:
        level: Standard library level name, normalised to upper case.
        log_file: Optional path of a rotating log file; ``None`` logs to stderr only.
    """

    level: str = DEFAULT_LOG_LEVEL
    log_file: Path | None = None

    def __post_init__(self) -> None:
        level = self.level.strip().upper() if isinstance(self.level, str) else ""
        if level not in VALID_LOG_LEVELS:
            raise ConfigurationError(
                f"logging.level must be one of {', '.join(VALID_LOG_LEVELS)}; got {self.level!r}."
            )
        object.__setattr__(self, "level", level)
        if self.log_file is not None:
            object.__setattr__(self, "log_file", Path(self.log_file))


@dataclass(frozen=True)
class Settings:
    """Root configuration object grouping all configuration sections."""

    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation (e.g. for display or run metadata)."""
        return {section: _to_jsonable(values) for section, values in asdict(self).items()}


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _read_env(env: Mapping[str, str], name: str) -> str | None:
    """Return the stripped value of ``VRE_<name>``; blank counts as unset."""
    raw = env.get(f"{ENV_PREFIX}{name}")
    if raw is None:
        return None
    return raw.strip() or None


def _parse_date(name: str, raw: str) -> date:
    try:
        return datetime.strptime(raw, _DATE_FORMAT).date()
    except ValueError as exc:
        raise ConfigurationError(
            f"{ENV_PREFIX}{name} must be an ISO date (YYYY-MM-DD); got {raw!r}."
        ) from exc


def _parse_confidence_levels(name: str, raw: str) -> tuple[float, ...]:
    try:
        return tuple(float(part) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise ConfigurationError(
            f"{ENV_PREFIX}{name} must be a comma-separated list of probabilities "
            f"such as '0.95,0.99'; got {raw!r}."
        ) from exc


_TRUE_WORDS = frozenset({"1", "true", "yes", "on"})
_FALSE_WORDS = frozenset({"0", "false", "no", "off"})


def _parse_bool(name: str, raw: str) -> bool:
    word = raw.lower()
    if word in _TRUE_WORDS:
        return True
    if word in _FALSE_WORDS:
        return False
    raise ConfigurationError(
        f"{ENV_PREFIX}{name} must be a boolean (true/false, yes/no, on/off, 1/0); got {raw!r}."
    )


def _parse_number(name: str, raw: str, convert: Callable[[str], Any], kind: str) -> Any:
    try:
        return convert(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{ENV_PREFIX}{name} must be {kind}; got {raw!r}.") from exc


def _load_data_config(env: Mapping[str, str]) -> DataConfig:
    kwargs: dict[str, Any] = {}
    if (ticker := _read_env(env, "DEFAULT_TICKER")) is not None:
        kwargs["ticker"] = ticker
    if (start := _read_env(env, "DEFAULT_START_DATE")) is not None:
        kwargs["start_date"] = _parse_date("DEFAULT_START_DATE", start)
    if (end := _read_env(env, "DEFAULT_END_DATE")) is not None:
        kwargs["end_date"] = _parse_date("DEFAULT_END_DATE", end)
    if (data_dir := _read_env(env, "DATA_DIR")) is not None:
        kwargs["data_dir"] = Path(data_dir)
    if (adjust := _read_env(env, "ADJUST_PRICES")) is not None:
        kwargs["adjust_prices"] = _parse_bool("ADJUST_PRICES", adjust)
    if (minimum := _read_env(env, "MIN_OBSERVATIONS")) is not None:
        kwargs["min_observations"] = _parse_number(
            "MIN_OBSERVATIONS", minimum, int, "an integer"
        )
    if (use_cache := _read_env(env, "USE_CACHE")) is not None:
        kwargs["use_cache"] = _parse_bool("USE_CACHE", use_cache)
    if (max_age := _read_env(env, "CACHE_MAX_AGE_HOURS")) is not None:
        kwargs["cache_max_age_hours"] = _parse_number(
            "CACHE_MAX_AGE_HOURS", max_age, float, "a number of hours"
        )
    return DataConfig(**kwargs)


def _load_preprocessing_config(env: Mapping[str, str]) -> PreprocessingConfig:
    kwargs: dict[str, Any] = {}
    if (include := _read_env(env, "INCLUDE_PROVISIONAL_BAR")) is not None:
        kwargs["include_provisional_bar"] = _parse_bool("INCLUDE_PROVISIONAL_BAR", include)
    if (gap := _read_env(env, "MAX_GAP_WEEKDAYS")) is not None:
        kwargs["max_gap_weekdays"] = _parse_number("MAX_GAP_WEEKDAYS", gap, int, "an integer")
    if (minimum := _read_env(env, "MIN_RETURNS")) is not None:
        kwargs["min_returns"] = _parse_number("MIN_RETURNS", minimum, int, "an integer")
    return PreprocessingConfig(**kwargs)


def _load_risk_config(env: Mapping[str, str]) -> RiskConfig:
    kwargs: dict[str, Any] = {}
    if (levels := _read_env(env, "CONFIDENCE_LEVELS")) is not None:
        kwargs["confidence_levels"] = _parse_confidence_levels("CONFIDENCE_LEVELS", levels)
    return RiskConfig(**kwargs)


def _load_logging_config(env: Mapping[str, str]) -> LoggingConfig:
    kwargs: dict[str, Any] = {}
    if (level := _read_env(env, "LOG_LEVEL")) is not None:
        kwargs["level"] = level
    if (log_file := _read_env(env, "LOG_FILE")) is not None:
        kwargs["log_file"] = Path(log_file)
    return LoggingConfig(**kwargs)


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Build :class:`Settings` from defaults overridden by ``VRE_*`` variables.

    Args:
        env: Mapping to read overrides from.  Defaults to ``os.environ``.

    Raises:
        ConfigurationError: If any override cannot be parsed or fails validation.
    """
    source = os.environ if env is None else env
    return Settings(
        data=_load_data_config(source),
        preprocessing=_load_preprocessing_config(source),
        risk=_load_risk_config(source),
        logging=_load_logging_config(source),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, loading them from the environment once.

    Call ``get_settings.cache_clear()`` after changing the environment in a
    running process (mainly useful in tests).
    """
    return load_settings()

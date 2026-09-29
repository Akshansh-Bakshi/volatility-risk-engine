"""Tests for src.config: defaults, environment overrides and validation."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import FrozenInstanceError
from datetime import date
from pathlib import Path

import pytest

from src.config import (
    PROJECT_ROOT,
    DataConfig,
    LoggingConfig,
    PreprocessingConfig,
    RiskConfig,
    Settings,
    get_settings,
    load_settings,
)
from src.exceptions import ConfigurationError, VolatilityRiskEngineError


def test_defaults_are_sensible_and_environment_independent() -> None:
    settings = load_settings({})

    assert settings.data.ticker == "^NSEI"
    assert settings.data.start_date == date(2005, 1, 1)
    assert settings.data.end_date is None
    assert settings.risk.confidence_levels == (0.95, 0.99)
    assert settings.logging.level == "INFO"
    assert settings.logging.log_file is None


def test_environment_overrides_every_setting() -> None:
    settings = load_settings(
        {
            "VRE_DEFAULT_TICKER": "spy",
            "VRE_DEFAULT_START_DATE": "2010-03-01",
            "VRE_DEFAULT_END_DATE": "2020-12-31",
            "VRE_CONFIDENCE_LEVELS": "0.975, 0.99",
            "VRE_LOG_LEVEL": "debug",
            "VRE_LOG_FILE": "logs/engine.log",
        }
    )

    assert settings.data.ticker == "SPY"
    assert settings.data.start_date == date(2010, 3, 1)
    assert settings.data.end_date == date(2020, 12, 31)
    assert settings.risk.confidence_levels == (0.975, 0.99)
    assert settings.logging.level == "DEBUG"
    assert settings.logging.log_file == Path("logs/engine.log")


def test_blank_environment_values_fall_back_to_defaults() -> None:
    settings = load_settings(
        {
            "VRE_DEFAULT_TICKER": "  ",
            "VRE_DEFAULT_END_DATE": "",
            "VRE_CONFIDENCE_LEVELS": "",
            "VRE_LOG_LEVEL": "",
            "VRE_LOG_FILE": "",
        }
    )

    assert settings == load_settings({})


def test_confidence_levels_are_sorted_and_deduplicated() -> None:
    settings = load_settings({"VRE_CONFIDENCE_LEVELS": "0.99,0.95,0.99"})

    assert settings.risk.confidence_levels == (0.95, 0.99)


@pytest.mark.parametrize(
    "env",
    [
        {"VRE_DEFAULT_START_DATE": "01/02/2020"},
        {"VRE_DEFAULT_START_DATE": "20200102"},
        {"VRE_DEFAULT_END_DATE": "not-a-date"},
        {"VRE_DEFAULT_START_DATE": "2020-01-01", "VRE_DEFAULT_END_DATE": "2020-01-01"},
        {"VRE_DEFAULT_START_DATE": "2020-06-01", "VRE_DEFAULT_END_DATE": "2019-06-01"},
        {"VRE_CONFIDENCE_LEVELS": "0.95,abc"},
        {"VRE_CONFIDENCE_LEVELS": "95"},
        {"VRE_CONFIDENCE_LEVELS": "1.0"},
        {"VRE_CONFIDENCE_LEVELS": "0"},
        {"VRE_CONFIDENCE_LEVELS": "-0.5"},
        {"VRE_CONFIDENCE_LEVELS": "nan"},
        {"VRE_LOG_LEVEL": "VERBOSE"},
    ],
)
def test_invalid_environment_values_raise_configuration_error(env: dict[str, str]) -> None:
    with pytest.raises(ConfigurationError):
        load_settings(env)


def test_configuration_error_belongs_to_project_hierarchy() -> None:
    assert issubclass(ConfigurationError, VolatilityRiskEngineError)


@pytest.mark.parametrize(
    "build",
    [
        lambda: DataConfig(ticker=""),
        lambda: DataConfig(ticker="   "),
        lambda: RiskConfig(confidence_levels=()),
        lambda: RiskConfig(confidence_levels=("high",)),  # type: ignore[arg-type]
        lambda: LoggingConfig(level="LOUD"),
    ],
)
def test_direct_construction_is_validated_too(build: Callable[[], object]) -> None:
    with pytest.raises(ConfigurationError):
        build()


def test_settings_are_immutable() -> None:
    settings = Settings()

    with pytest.raises(FrozenInstanceError):
        settings.data = DataConfig(ticker="SPY")  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        settings.risk.confidence_levels = (0.9,)  # type: ignore[misc]


def test_get_settings_reads_environment_once_and_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VRE_DEFAULT_TICKER", "qqq")
    first = get_settings()
    monkeypatch.setenv("VRE_DEFAULT_TICKER", "iwm")

    assert first.data.ticker == "QQQ"
    assert get_settings() is first

    get_settings.cache_clear()
    assert get_settings().data.ticker == "IWM"


def test_to_dict_is_json_serialisable_and_faithful() -> None:
    settings = load_settings(
        {"VRE_DEFAULT_END_DATE": "2024-01-31", "VRE_LOG_FILE": "logs/engine.log"}
    )

    payload = json.loads(json.dumps(settings.to_dict()))

    assert payload == {
        "data": {
            "ticker": "^NSEI",
            "start_date": "2005-01-01",
            "end_date": "2024-01-31",
            "data_dir": str(PROJECT_ROOT / "data"),
            "adjust_prices": True,
            "min_observations": 250,
            "use_cache": True,
            "cache_max_age_hours": 6.0,
        },
        "preprocessing": {
            "include_provisional_bar": False,
            "max_gap_weekdays": 1,
            "min_returns": 250,
        },
        "risk": {"confidence_levels": [0.95, 0.99]},
        "logging": {"level": "INFO", "log_file": str(Path("logs/engine.log"))},
    }


# --- data-layer settings ---------------------------------------------------------------


def test_data_layer_defaults() -> None:
    data = load_settings({}).data

    assert data.data_dir == PROJECT_ROOT / "data"
    assert data.adjust_prices is True
    assert data.min_observations == 250
    assert data.use_cache is True
    assert data.cache_max_age_hours == 6.0


def test_data_layer_environment_overrides() -> None:
    data = load_settings(
        {
            "VRE_DATA_DIR": "/tmp/vre-data",
            "VRE_ADJUST_PRICES": "false",
            "VRE_MIN_OBSERVATIONS": "500",
            "VRE_USE_CACHE": "no",
            "VRE_CACHE_MAX_AGE_HOURS": "0.5",
        }
    ).data

    assert data.data_dir == Path("/tmp/vre-data")
    assert data.adjust_prices is False
    assert data.min_observations == 500
    assert data.use_cache is False
    assert data.cache_max_age_hours == 0.5


def test_blank_data_layer_variables_fall_back_to_defaults() -> None:
    blanks = {
        f"VRE_{name}": ""
        for name in (
            "DATA_DIR",
            "ADJUST_PRICES",
            "MIN_OBSERVATIONS",
            "USE_CACHE",
            "CACHE_MAX_AGE_HOURS",
        )
    }

    assert load_settings(blanks) == load_settings({})


@pytest.mark.parametrize("word", ["1", "true", "TRUE", "Yes", "on"])
def test_boolean_true_spellings(word: str) -> None:
    assert load_settings({"VRE_USE_CACHE": word}).data.use_cache is True


@pytest.mark.parametrize("word", ["0", "false", "False", "NO", "off"])
def test_boolean_false_spellings(word: str) -> None:
    assert load_settings({"VRE_ADJUST_PRICES": word}).data.adjust_prices is False


@pytest.mark.parametrize(
    "env",
    [
        {"VRE_ADJUST_PRICES": "maybe"},
        {"VRE_USE_CACHE": "2"},
        {"VRE_MIN_OBSERVATIONS": "many"},
        {"VRE_MIN_OBSERVATIONS": "2.5"},
        {"VRE_MIN_OBSERVATIONS": "1"},
        {"VRE_MIN_OBSERVATIONS": "-10"},
        {"VRE_CACHE_MAX_AGE_HOURS": "soon"},
        {"VRE_CACHE_MAX_AGE_HOURS": "0"},
        {"VRE_CACHE_MAX_AGE_HOURS": "-1"},
        {"VRE_CACHE_MAX_AGE_HOURS": "nan"},
        {"VRE_CACHE_MAX_AGE_HOURS": "inf"},
    ],
)
def test_invalid_data_layer_environment_values_are_rejected(env: dict[str, str]) -> None:
    with pytest.raises(ConfigurationError):
        load_settings(env)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"adjust_prices": "yes"},
        {"use_cache": 1},
        {"min_observations": True},
        {"min_observations": 1},
        {"min_observations": 10.0},
        {"cache_max_age_hours": True},
        {"cache_max_age_hours": 0},
        {"cache_max_age_hours": "6"},
    ],
)
def test_direct_data_config_construction_validates_data_layer_fields(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ConfigurationError):
        DataConfig(**kwargs)  # type: ignore[arg-type]


def test_data_dir_is_normalised_to_a_path_with_user_expansion() -> None:
    config = DataConfig(data_dir="~/vre-data")  # type: ignore[arg-type]

    assert isinstance(config.data_dir, Path)
    assert config.data_dir == Path("~/vre-data").expanduser()
    assert "~" not in str(config.data_dir)


def test_integer_cache_age_is_stored_as_float() -> None:
    assert DataConfig(cache_max_age_hours=12).cache_max_age_hours == 12.0
    assert isinstance(DataConfig(cache_max_age_hours=12).cache_max_age_hours, float)


# --- preprocessing settings ------------------------------------------------------------------


def test_preprocessing_defaults_exclude_the_provisional_bar_and_tolerate_one_closure() -> None:
    config = load_settings({}).preprocessing

    assert config.include_provisional_bar is False
    assert config.max_gap_weekdays == 1
    assert config.min_returns == 250


def test_preprocessing_environment_overrides() -> None:
    config = load_settings(
        {
            "VRE_INCLUDE_PROVISIONAL_BAR": "true",
            "VRE_MAX_GAP_WEEKDAYS": "0",
            "VRE_MIN_RETURNS": "500",
        }
    ).preprocessing

    assert config == PreprocessingConfig(
        include_provisional_bar=True, max_gap_weekdays=0, min_returns=500
    )


def test_blank_preprocessing_variables_fall_back_to_defaults() -> None:
    blanks = {
        f"VRE_{name}": "" for name in ("INCLUDE_PROVISIONAL_BAR", "MAX_GAP_WEEKDAYS", "MIN_RETURNS")
    }

    assert load_settings(blanks) == load_settings({})


@pytest.mark.parametrize(
    "env",
    [
        {"VRE_INCLUDE_PROVISIONAL_BAR": "sometimes"},
        {"VRE_MAX_GAP_WEEKDAYS": "-1"},
        {"VRE_MAX_GAP_WEEKDAYS": "one"},
        {"VRE_MAX_GAP_WEEKDAYS": "1.5"},
        {"VRE_MIN_RETURNS": "1"},
        {"VRE_MIN_RETURNS": "0"},
        {"VRE_MIN_RETURNS": "lots"},
    ],
)
def test_invalid_preprocessing_environment_values_are_rejected(env: dict[str, str]) -> None:
    with pytest.raises(ConfigurationError):
        load_settings(env)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"include_provisional_bar": 1},
        {"include_provisional_bar": "false"},
        {"max_gap_weekdays": True},
        {"max_gap_weekdays": -1},
        {"max_gap_weekdays": 1.0},
        {"min_returns": True},
        {"min_returns": 1},
        {"min_returns": 2.0},
    ],
)
def test_direct_preprocessing_config_construction_is_validated(kwargs: dict[str, object]) -> None:
    with pytest.raises(ConfigurationError):
        PreprocessingConfig(**kwargs)  # type: ignore[arg-type]


def test_validation_messages_name_the_section_and_field() -> None:
    with pytest.raises(ConfigurationError, match=r"preprocessing\.min_returns.*>= 2"):
        PreprocessingConfig(min_returns=1)
    with pytest.raises(ConfigurationError, match=r"data\.min_observations.*>= 2"):
        DataConfig(min_observations=1)

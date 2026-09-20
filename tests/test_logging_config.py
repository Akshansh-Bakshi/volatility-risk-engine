"""Tests for src.logging_config."""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path

import pytest

from src.config import LoggingConfig, get_settings
from src.exceptions import ConfigurationError
from src.logging_config import PACKAGE_LOGGER_NAME, configure_logging


def _managed_handlers(logger: logging.Logger) -> list[logging.Handler]:
    return [h for h in logger.handlers if getattr(h, "_vre_managed", False)]


def test_configure_logging_initialises_package_logger() -> None:
    logger = configure_logging(LoggingConfig(level="WARNING"))

    assert logger.name == PACKAGE_LOGGER_NAME == "src"
    assert logger.level == logging.WARNING
    assert logger.propagate is False
    assert len(_managed_handlers(logger)) == 1


def test_configure_logging_is_idempotent() -> None:
    for _ in range(3):
        logger = configure_logging(LoggingConfig(level="INFO"))

    assert len(_managed_handlers(logger)) == 1


def test_configure_logging_defaults_come_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VRE_LOG_LEVEL", "ERROR")
    get_settings.cache_clear()

    logger = configure_logging()

    assert logger.level == logging.ERROR


def test_module_loggers_inherit_level_and_write_to_file(tmp_path: Path) -> None:
    log_file = tmp_path / "nested" / "engine.log"
    configure_logging(LoggingConfig(level="INFO", log_file=log_file))

    child = logging.getLogger("src.data.some_module")
    child.debug("hidden detail")
    child.info("fetched %d rows", 42)
    try:
        raise ValueError("boom")
    except ValueError:
        child.exception("pipeline step failed")

    content = log_file.read_text(encoding="utf-8")
    assert "hidden detail" not in content
    assert "INFO" in content and "src.data.some_module" in content
    assert "fetched 42 rows" in content
    assert "pipeline step failed" in content and "ValueError: boom" in content
    assert re.search(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z \|", content, re.MULTILINE)


def test_reconfiguring_without_a_file_removes_the_file_handler(tmp_path: Path) -> None:
    configure_logging(LoggingConfig(log_file=tmp_path / "engine.log"))
    logger = configure_logging(LoggingConfig())

    handler_types = {type(h).__name__ for h in _managed_handlers(logger)}
    assert handler_types == {"StreamHandler"}


def test_unwritable_log_file_raises_and_preserves_existing_handlers(tmp_path: Path) -> None:
    logger = configure_logging(LoggingConfig(level="INFO"))
    before = _managed_handlers(logger)
    blocker = tmp_path / "not_a_directory"
    blocker.write_text("occupied", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="Cannot open log file"):
        configure_logging(LoggingConfig(log_file=blocker / "engine.log"))

    assert _managed_handlers(logger) == before


@pytest.mark.skipif(not hasattr(time, "tzset"), reason="time.tzset is unavailable on this platform")
def test_timestamps_are_utc_regardless_of_local_timezone() -> None:
    logger = configure_logging(LoggingConfig())
    formatter = _managed_handlers(logger)[0].formatter
    assert formatter is not None
    record = logging.LogRecord("src.x", logging.INFO, __file__, 1, "msg", None, None)
    record.created = 0.0  # 1970-01-01T00:00:00Z

    original_tz = os.environ.get("TZ")
    os.environ["TZ"] = "IST-5:30"  # POSIX form (UTC+05:30); needs no tz database
    time.tzset()
    try:
        assert formatter.format(record).startswith("1970-01-01T00:00:00Z |")
    finally:
        if original_tz is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = original_tz
        time.tzset()

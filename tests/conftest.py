"""Shared fixtures that keep every test isolated from ambient state."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator

import pytest

from src.config import ENV_PREFIX, get_settings
from src.logging_config import PACKAGE_LOGGER_NAME


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Hide any VRE_* variables from the developer's shell and reset the settings cache."""
    for name in [name for name in os.environ if name.startswith(ENV_PREFIX)]:
        monkeypatch.delenv(name)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class _RecordCollector(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def log_records() -> list[logging.LogRecord]:
    """Collect every record emitted under the package logger during a test."""
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    collector = _RecordCollector()
    logger.addHandler(collector)
    logger.setLevel(logging.DEBUG)
    return collector.records  # the autouse fixture below removes the handler again


@pytest.fixture(autouse=True)
def _restore_package_logger() -> Iterator[None]:
    """Undo any handler, level or propagation changes made to the package logger."""
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    original_handlers = list(logger.handlers)
    original_level = logger.level
    original_propagate = logger.propagate
    yield
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        if handler not in original_handlers:
            handler.close()
    for handler in original_handlers:
        logger.addHandler(handler)
    logger.setLevel(original_level)
    logger.propagate = original_propagate

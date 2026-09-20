"""Reusable logging configuration for the whole package.

All modules obtain their logger with ``logging.getLogger(__name__)``, which
places them under the package logger configured here.  Handlers are attached
to that package logger only, never to the root logger, so the application does
not interfere with (or duplicate output from) logging set up by Streamlit,
pytest or third-party libraries.

:func:`configure_logging` is idempotent.  Streamlit re-executes the entry
script on every user interaction, so calling it repeatedly must not stack
duplicate handlers.
"""

from __future__ import annotations

import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.config import LoggingConfig, get_settings
from src.exceptions import ConfigurationError

PACKAGE_LOGGER_NAME = __name__.split(".")[0]

# Timestamps are UTC and include module, function and line so that a failure in
# a long data/model pipeline can be traced back to its origin from the log alone.
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

_MAX_LOG_FILE_BYTES = 5 * 1024 * 1024
_LOG_FILE_BACKUP_COUNT = 3
_MANAGED_HANDLER_ATTR = "_vre_managed"


def _build_file_handler(path: Path) -> RotatingFileHandler:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        return RotatingFileHandler(
            path,
            maxBytes=_MAX_LOG_FILE_BYTES,
            backupCount=_LOG_FILE_BACKUP_COUNT,
            encoding="utf-8",
        )
    except OSError as exc:
        raise ConfigurationError(f"Cannot open log file {str(path)!r}: {exc}") from exc


def _detach_managed_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        if getattr(handler, _MANAGED_HANDLER_ATTR, False):
            logger.removeHandler(handler)
            handler.close()


def configure_logging(config: LoggingConfig | None = None) -> logging.Logger:
    """Configure and return the package logger.

    Args:
        config: Logging settings.  Defaults to ``get_settings().logging``.

    Returns:
        The package logger; module loggers below it inherit its level and handlers.

    Raises:
        ConfigurationError: If the requested log file cannot be opened.  Any
            previously configured handlers are left untouched in that case.
    """
    config = config if config is not None else get_settings().logging
    formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
    formatter.converter = time.gmtime  # UTC timestamps: reproducible and comparable

    # Build every new handler before touching the live logger so a failure
    # (e.g. an unwritable log path) cannot leave logging half-configured.
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if config.log_file is not None:
        handlers.append(_build_file_handler(config.log_file))

    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    _detach_managed_handlers(logger)
    for handler in handlers:
        handler.setFormatter(formatter)
        setattr(handler, _MANAGED_HANDLER_ATTR, True)
        logger.addHandler(handler)

    logger.setLevel(getattr(logging, config.level))
    logger.propagate = False
    logger.debug(
        "Logging configured (level=%s, log_file=%s).", config.level, config.log_file or "none"
    )
    return logger

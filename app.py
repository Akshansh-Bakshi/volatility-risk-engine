"""Streamlit entry point for the Volatility Analytics & Market Risk Engine.

This is currently an application shell: it wires configuration and logging and
renders the active defaults.  Analytics pages are added in later stages.

Run with::

    streamlit run app.py
"""

from __future__ import annotations

import logging

import streamlit as st

from src import __version__
from src.config import get_settings
from src.exceptions import ConfigurationError
from src.logging_config import PACKAGE_LOGGER_NAME, configure_logging

APP_TITLE = "Volatility Analytics & Market Risk Engine"

logger = logging.getLogger(f"{PACKAGE_LOGGER_NAME}.app")


def main() -> None:
    """Render the application shell."""
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    try:
        settings = get_settings()
        configure_logging(settings.logging)
    except ConfigurationError as exc:
        logger.error("Invalid configuration: %s", exc)
        st.error(f"Invalid configuration: {exc}")
        st.stop()

    st.caption(f"Version {__version__} · foundation stage")
    st.info(
        "The application shell is running. Market data ingestion, volatility models, "
        "Value-at-Risk and backtesting are added in later development stages."
    )
    with st.expander("Active default configuration"):
        st.json(settings.to_dict())


if __name__ == "__main__":
    main()

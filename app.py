"""Streamlit synopsis dashboard for the Volatility Analytics & Market Risk Engine."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import streamlit as st

from src import __version__
from src.config import get_settings
from src.data.factory import create_market_data_loader
from src.data.loader import MarketDataLoader
from src.data.market_data import MarketData, MarketDataRequest
from src.exceptions import ConfigurationError, DataError, PreprocessingError
from src.logging_config import PACKAGE_LOGGER_NAME, configure_logging
from src.preprocessing.returns import build_return_series
from src.preprocessing.return_series import ReturnSeries
from src.statistics.descriptive import ReturnStats, compute_return_stats
from src.statistics.figures import (
    plot_data_quality,
    plot_price_series,
    plot_return_histogram,
    plot_return_series,
)
from src.statistics.profile import DatasetProfile, build_dataset_profile

APP_TITLE = "Volatility Analytics & Market Risk Engine"
DEFAULT_SYNOPSIS_TICKER = "^NSEI"
MIN_SYNOPSIS_OBSERVATIONS = 500

logger = logging.getLogger(f"{PACKAGE_LOGGER_NAME}.app")


@dataclass(frozen=True)
class SynopsisData:
    """Existing Stage 4 EDA objects assembled for the synopsis UI."""

    market_data: MarketData
    returns: ReturnSeries
    profile: DatasetProfile
    stats: ReturnStats


def load_synopsis_data(
    loader: MarketDataLoader,
    ticker: str,
    start: date,
    end: date,
) -> SynopsisData:
    """Load and profile one requested dataset through existing data and EDA layers."""
    request = MarketDataRequest(ticker=ticker, start=start, end=end)
    market_data = loader.load(request)
    returns = build_return_series(market_data)
    profile = build_dataset_profile(market_data, returns)
    stats = compute_return_stats(returns)
    return SynopsisData(market_data, returns, profile, stats)


@st.cache_resource(show_spinner=False)
def _get_market_data_loader() -> MarketDataLoader:
    """Reuse one configured loader so its established local cache remains effective."""
    return create_market_data_loader()


def _render_synopsis(data: SynopsisData) -> None:
    market_data, returns, profile, stats = (
        data.market_data,
        data.returns,
        data.profile,
        data.stats,
    )

    st.subheader("Dataset at a glance")
    meets_requirement = profile.price_observations >= MIN_SYNOPSIS_OBSERVATIONS
    requirement_label = "Meets ≥500 requirement" if meets_requirement else "Below 500 observations"
    metric_cols = st.columns(4)
    metric_cols[0].metric("Price observations", f"{profile.price_observations:,}")
    metric_cols[1].metric("Daily returns", f"{profile.return_observations:,}")
    metric_cols[2].metric("Actual date range", f"{profile.first_price_date} → {profile.last_price_date}")
    metric_cols[3].metric("Synopsis sample", requirement_label)

    st.caption(
        f"Fetched {profile.fetched_at:%Y-%m-%d %H:%M UTC} · "
        f"Actual range {profile.first_price_date} to {profile.last_price_date} · "
        f"{profile.price_observations:,} price observations · "
        f"{'served from cache' if profile.from_cache else 'fresh provider response'}"
    )
    if meets_requirement:
        st.success("The selected dataset satisfies the academic minimum of 500 observations.")
    else:
        st.warning("The selected range has fewer than 500 observations; widen the date range for the synopsis.")

    st.markdown("### Dataset description")
    metadata = {
        "Source": profile.source,
        "Ticker": profile.ticker,
        "Frequency": profile.frequency,
        "Start date (actual)": profile.first_price_date.isoformat(),
        "End date (actual)": profile.last_price_date.isoformat(),
        "Observation count": profile.price_observations,
        "Variables": ", ".join(profile.quality.columns_present),
        "Price basis": profile.price_basis.replace("_", " "),
    }
    st.dataframe(
        [{"Attribute": key, "Value": value} for key, value in metadata.items()],
        hide_index=True,
        use_container_width=True,
    )

    provisional = profile.quality.provisional_bar
    if provisional:
        status = "may be provisional"
        if profile.quality.provisional_bar_excluded:
            status += " and was excluded from returns"
        else:
            status += " and is included in returns"
        st.caption(f"Latest bar: {provisional} {status} under the configured preprocessing policy.")
    else:
        st.caption("Latest bar: no potentially provisional bar was identified.")

    st.markdown("### Exploratory data analysis")
    st.pyplot(plot_price_series(market_data, figsize=(12, 3.4)), use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.pyplot(plot_return_series(returns, figsize=(8, 3.4)), use_container_width=True)
    with right:
        st.pyplot(plot_return_histogram(returns, stats, figsize=(8, 3.4)), use_container_width=True)

    st.markdown("### Data quality and missing data")
    q = profile.quality
    quality_cols = st.columns(4)
    quality_cols[0].metric("Rows removed (missing close)", q.missing_close_rows)
    quality_cols[1].metric("Duplicate timestamps", q.duplicate_timestamps)
    quality_cols[2].metric("Gap-flagged returns", q.gap_flagged_returns)
    quality_cols[3].metric("Missing close values remaining", 0)
    st.caption(
        "Missing close rows are removed during data validation and counted here; "
        "prices are not imputed. Gap flags use the existing weekday tolerance "
        f"({q.gap_tolerance_weekdays} weekday(s)). {q.quality_summary}"
    )
    st.pyplot(plot_data_quality(market_data, returns, figsize=(9, 3.5)), use_container_width=True)

    st.markdown("### Descriptive statistics")
    st.caption("Daily log-return statistics; return values are displayed in percent.")
    st.dataframe(stats.summary_frame(scale="percent"), use_container_width=True)

    st.markdown("### Initial observations")
    st.write(
        f"The selected series contains {stats.count:,} daily log returns. "
        f"The sample mean is {stats.mean_pct:.4f}% per observation and the sample "
        f"standard deviation is {stats.std_pct:.4f}%. Observed returns range from "
        f"{stats.minimum_pct:.2f}% to {stats.maximum_pct:.2f}%; sample skewness is "
        f"{stats.skewness:.3f} and excess kurtosis is {stats.excess_kurtosis:.3f}."
    )

    st.markdown("### Proposed methodology")
    st.markdown(
        "1. Finalise the dataset definition, document provenance, and inspect missing sessions and quality flags.\n"
        "2. Explore stationarity and dependence in returns and squared returns using statistical diagnostics.\n"
        "3. Compare appropriate time-series mean and volatility model families on chronological train/test splits.\n"
        "4. Evaluate forecast errors and residual diagnostics, then assess risk estimates with backtesting.\n"
        "5. Report assumptions, limitations, and reproducible results."
    )
    st.caption("This synopsis page presents descriptive EDA only; subsequent methodology is proposed, not executed here.")


def main() -> None:
    """Render the existing application with its synopsis dashboard as the landing page."""
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    try:
        settings = get_settings()
        configure_logging(settings.logging)
    except ConfigurationError as exc:
        logger.error("Invalid configuration: %s", exc)
        st.error(f"Invalid configuration: {exc}")
        st.stop()

    st.caption(f"Version {__version__} · Synopsis dashboard")
    st.write(
        "A descriptive study of historical market data for the Time Series Analysis synopsis. "
        "Choose a dataset below; no forecasting or risk model is run on this page."
    )

    today = date.today()
    with st.form("synopsis_dataset_form"):
        controls = st.columns([1, 1, 2])
        ticker = controls[0].text_input("Ticker", value=DEFAULT_SYNOPSIS_TICKER).strip()
        selected_range = controls[1].date_input(
            "Date range",
            value=(date(2005, 1, 1), today),
            min_value=date(1900, 1, 1),
            max_value=today,
        )
        controls[2].markdown(
            "**Default example:** NIFTY 50 (`^NSEI`) · daily adjusted prices · "
            "500 observations minimum for the academic sample."
        )
        submitted = st.form_submit_button("Load dataset", type="primary")

    if submitted:
        if not ticker:
            st.error("Enter a ticker symbol.")
            return
        if not isinstance(selected_range, (tuple, list)) or len(selected_range) != 2:
            st.error("Select both a start date and an end date.")
            return
        start, end = selected_range
        if start >= end:
            st.error("The start date must be earlier than the end date.")
            return
        try:
            with st.spinner(f"Loading {ticker.upper()} market data…"):
                dataset = load_synopsis_data(_get_market_data_loader(), ticker, start, end)
            _render_synopsis(dataset)
        except (DataError, PreprocessingError) as exc:
            logger.warning("Synopsis data load failed for %s: %s", ticker, exc)
            st.error(f"Could not load this dataset: {exc}")
    else:
        st.info("Choose a ticker and date range, then select **Load dataset** to build the synopsis EDA.")

    with st.expander("Active default configuration"):
        st.json(settings.to_dict())


if __name__ == "__main__":
    main()

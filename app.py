"""Streamlit entry point for the Volatility Analytics & Market Risk Engine."""

from __future__ import annotations

import html
import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import get_settings
from src.data.factory import create_market_data_loader
from src.data.loader import MarketDataLoader
from src.data.market_data import MarketData, MarketDataRequest
from src.exceptions import ConfigurationError, DataError, PreprocessingError
from src.logging_config import PACKAGE_LOGGER_NAME, configure_logging
from src.preprocessing.returns import build_return_series
from src.preprocessing.return_series import ReturnSeries
from src.risk.figures import plot_var_forecasts
from src.statistics.dashboard import calculate_dashboard_metrics, insight_cards
from src.statistics.descriptive import ReturnStats, compute_return_stats
from src.statistics.profile import DatasetProfile, build_dataset_profile

APP_TITLE = "Volatility Analytics"
DEFAULT_SYNOPSIS_TICKER = "^NSEI"
MIN_SYNOPSIS_OBSERVATIONS = 500
VAR_RESULTS_DIR = Path(__file__).resolve().parent / "results" / "var_backtesting" / "nifty50"
VAR_RESULTS_STALE_AFTER = timedelta(days=30)
ASSET_PRESETS: dict[str, dict[str, str]] = {
    "India": {
        "NIFTY 50 (^NSEI)": "^NSEI",
        "SENSEX (^BSESN)": "^BSESN",
        "RELIANCE (RELIANCE.NS)": "RELIANCE.NS",
        "TCS (TCS.NS)": "TCS.NS",
        "HDFC Bank (HDFCBANK.NS)": "HDFCBANK.NS",
    },
    "United States": {
        "S&P 500 (^GSPC)": "^GSPC",
        "Apple (AAPL)": "AAPL",
        "Microsoft (MSFT)": "MSFT",
        "NVIDIA (NVDA)": "NVDA",
        "Tesla (TSLA)": "TSLA",
    },
}

logger = logging.getLogger(f"{PACKAGE_LOGGER_NAME}.app")


@dataclass(frozen=True)
class SynopsisData:
    """Existing Stage 4 EDA objects assembled for the dashboard."""

    market_data: MarketData
    returns: ReturnSeries
    profile: DatasetProfile
    stats: ReturnStats


@dataclass(frozen=True)
class VaRDashboardArtifacts:
    """Validated saved Stage 10 results for presentation in the dashboard."""

    forecasts: pd.DataFrame | None
    summary: pd.DataFrame | None
    metadata: dict[str, Any] | None
    artifacts_saved_at: datetime | None
    status: str
    message: str

    @property
    def available(self) -> bool:
        return self.forecasts is not None and self.summary is not None and self.metadata is not None


VAR_FORECAST_COLUMNS = (
    "realised_loss_pct",
    "garch_var95_pct", "egarch_var95_pct", "historical_var95_pct",
    "garch_var99_pct", "egarch_var99_pct", "historical_var99_pct",
    "garch_breach95", "egarch_breach95", "historical_breach95",
    "garch_breach99", "egarch_breach99", "historical_breach99",
)
VAR_SUMMARY_COLUMNS = (
    "method", "confidence", "breach_count", "sample_size", "observed_breach_rate",
    "expected_breach_rate", "kupiec_statistic", "kupiec_p_value",
    "christoffersen_independence_statistic", "christoffersen_independence_p_value",
    "conditional_coverage_statistic", "conditional_coverage_p_value",
)
VAR_METHODS = {
    "GARCH Normal", "EGARCH Normal", "250-return historical simulation"
}


def load_var_dashboard_artifacts(
    results_dir: Path | None = None,
    *,
    now: datetime | None = None,
) -> VaRDashboardArtifacts:
    """Read and validate saved Stage 10 files without fitting or scoring models."""
    directory = VAR_RESULTS_DIR if results_dir is None else Path(results_dir)
    forecast_path = directory / "daily_var_forecasts.csv"
    summary_path = directory / "backtest_summary.csv"
    metadata_path = directory / "metadata.json"
    missing = [path.name for path in (forecast_path, summary_path, metadata_path) if not path.is_file()]
    if missing:
        return VaRDashboardArtifacts(
            None, None, None, None, "missing",
            "Saved Stage 10 results are unavailable (missing " + ", ".join(missing) + ").",
        )

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        forecasts = pd.read_csv(forecast_path, parse_dates=["date"], index_col="date")
        summary = pd.read_csv(summary_path)
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a JSON object")
        missing_forecast = sorted(set(VAR_FORECAST_COLUMNS) - set(forecasts.columns))
        missing_summary = sorted(set(VAR_SUMMARY_COLUMNS) - set(summary.columns))
        if missing_forecast or missing_summary:
            details = []
            if missing_forecast:
                details.append("daily forecast columns: " + ", ".join(missing_forecast))
            if missing_summary:
                details.append("summary columns: " + ", ".join(missing_summary))
            raise ValueError("incomplete results (" + "; ".join(details) + ")")

        if (
            not isinstance(forecasts.index, pd.DatetimeIndex)
            or forecasts.empty
            or forecasts.index.hasnans
            or forecasts.index.has_duplicates
            or not forecasts.index.is_monotonic_increasing
        ):
            raise ValueError("forecast dates must be present, unique, and chronological")
        expected_n = int(metadata["evaluation"]["observations"])
        if expected_n != 252 or len(forecasts) != expected_n:
            raise ValueError("the Stage 10 evaluation must contain 252 forecast rows")
        if forecasts.index[0].date().isoformat() != metadata["evaluation"]["start"]:
            raise ValueError("forecast dates do not match the recorded evaluation start")
        if forecasts.index[-1].date().isoformat() != metadata["evaluation"]["end"]:
            raise ValueError("forecast dates do not match the recorded evaluation end")
        if metadata["data"]["ticker"] != "^NSEI":
            raise ValueError("saved risk artifacts are not for the expected NIFTY 50 ticker")

        numeric_forecast_columns = [column for column in VAR_FORECAST_COLUMNS if "breach" not in column]
        for column in numeric_forecast_columns:
            forecasts[column] = pd.to_numeric(forecasts[column], errors="coerce")
            if not np.isfinite(forecasts[column].to_numpy(dtype="float64")).all():
                raise ValueError(f"forecast column {column!r} contains incomplete values")
        for column in (column for column in VAR_FORECAST_COLUMNS if "breach" in column):
            normalized = forecasts[column].astype(str).str.strip().str.lower()
            if not normalized.isin({"true", "false", "1", "0"}).all():
                raise ValueError(f"breach column {column!r} contains invalid values")
            forecasts[column] = normalized.isin({"true", "1"})

        summary_confidence = pd.to_numeric(summary["confidence"], errors="coerce").round(2)
        expected_pairs = {
            (method, confidence)
            for method in VAR_METHODS
            for confidence in (0.95, 0.99)
        }
        actual_pairs = set(zip(summary["method"], summary_confidence, strict=True))
        if len(summary) != 6 or actual_pairs != expected_pairs:
            raise ValueError("backtest summary must contain 95% and 99% results for all three methods")
        if set(pd.to_numeric(summary["sample_size"], errors="coerce")) != {252}:
            raise ValueError("summary sample sizes do not match the 252-observation evaluation")
        for column in VAR_SUMMARY_COLUMNS[1:]:
            summary[column] = pd.to_numeric(summary[column], errors="coerce")
            if not np.isfinite(summary[column].to_numpy(dtype="float64")).all():
                raise ValueError(f"summary column {column!r} contains incomplete values")

        retrieval_time = pd.Timestamp(metadata["data"]["data_fetched_at_utc"]).to_pydatetime()
        if retrieval_time.tzinfo is None:
            raise ValueError("data retrieval timestamp must include a timezone")
        saved_at = datetime.fromtimestamp(metadata_path.stat().st_mtime, tz=timezone.utc)
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        if (current - retrieval_time).total_seconds() > VAR_RESULTS_STALE_AFTER.total_seconds() or (
            current - saved_at
        ).total_seconds() > VAR_RESULTS_STALE_AFTER.total_seconds():
            return VaRDashboardArtifacts(
                forecasts, summary, metadata, saved_at, "stale",
                "These saved risk results are older than 30 days; rerun the Stage 10 experiment to refresh them.",
            )
        return VaRDashboardArtifacts(
            forecasts, summary, metadata, saved_at, "ready", "Saved Stage 10 results loaded."
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, pd.errors.ParserError) as exc:
        return VaRDashboardArtifacts(
            None, None, None, None, "incomplete",
            f"Saved Stage 10 results could not be displayed because they are incomplete or invalid ({exc}).",
        )


def resolve_ticker(selection: str, *, custom_ticker: str = "") -> str:
    """Resolve a preset label or custom ticker into the exact provider symbol."""
    for group in ASSET_PRESETS.values():
        if selection in group:
            return group[selection]
    if selection == "Custom ticker":
        return custom_ticker.strip().upper()
    return selection.strip().upper()


def load_synopsis_data(
    loader: MarketDataLoader,
    ticker: str,
    start: date,
    end: date,
) -> SynopsisData:
    """Load and profile a request through the existing data and EDA layers."""
    request = MarketDataRequest(ticker=ticker, start=start, end=end)
    market_data = loader.load(request)
    returns = build_return_series(market_data)
    profile = build_dataset_profile(market_data, returns)
    stats = compute_return_stats(returns)
    return SynopsisData(market_data, returns, profile, stats)


@st.cache_resource(show_spinner=False)
def _get_market_data_loader() -> MarketDataLoader:
    """Keep the configured loader alive so its established local cache is reused."""
    return create_market_data_loader()


def _format_metric(value: float | None, suffix: str = "", decimals: int = 2) -> str:
    return "N/A" if value is None or not pd.notna(value) else f"{value:,.{decimals}f}{suffix}"


def _price_unit(ticker: str) -> str:
    if ticker in {"^NSEI", "^BSESN", "^GSPC"}:
        return "index points"
    if ticker.endswith(".NS"):
        return "₹"
    if ticker in {"AAPL", "MSFT", "NVDA", "TSLA"}:
        return "$"
    return ""


def _apply_visual_style() -> None:
    st.markdown(
        """<style>
        .stApp { background: #0b1220; color: #e6edf7; }
        [data-testid="stHeader"] { background: rgba(11,18,32,.94); }
        [data-testid="stSidebar"] { background: #101a2b; }
        .block-container { padding-top: 1.1rem; padding-bottom: 1rem; max-width: 1680px; }
        section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div {
            width: 16rem !important; min-width: 16rem !important; max-width: 16rem !important;
        }
        h1, h2, h3 { color: #f3f7ff; letter-spacing: -0.02em; }
        div[data-testid="stMetric"] { background: #111d30; border: 1px solid #24344d;
            padding: .7rem .8rem; border-radius: 12px; height: 112px; box-sizing: border-box;
            display: flex; flex-direction: column; justify-content: space-between; overflow: hidden; }
        div[data-testid="stMetricLabel"] { color: #a9bad1; min-height: 1.6rem; font-size: .82rem; }
        div[data-testid="stMetricValue"] { color: #f4f8ff; font-size: clamp(1.15rem, 1.4vw, 1.55rem); white-space: nowrap; }
        div[role="radiogroup"] { gap: .6rem; }
        .asset-symbol { color: #a8bad4; font-size: .95rem; }
        .status-dot { color: #52d6a1; font-size: 1.05rem; }
        .product-header { display:flex; align-items:center; justify-content:space-between; gap:1rem; margin:.2rem 0 .25rem; }
        .product-title { color:#f3f7ff; font-size:2rem; font-weight:700; line-height:1.15; letter-spacing:-.035em; }
        .product-subtitle { color:#a9bad1; font-size:1.15rem; margin-top:.25rem; }
        .data-status { flex:0 0 auto; border:1px solid #245743; background:#10291f; color:#86e3b6;
            border-radius:999px; padding:.45rem .8rem; font-size:.8rem; font-weight:650; }
        .metadata-strip { display:flex; flex-wrap:wrap; align-items:center; gap:.45rem .9rem; color:#b6c5d9;
            border-top:1px solid #24344d; border-bottom:1px solid #24344d; padding:.55rem .1rem;
            margin:.55rem 0 .8rem; font-size:.84rem; }
        .metadata-strip strong { color:#f3f7ff; }
        .health-line { border-radius:10px; padding:.65rem .8rem; min-height:2.7rem; line-height:1.4;
            background:#111d30; border:1px solid #24344d; color:#c2d0e2; }
        .health-good { color:#78dba9; font-weight:700; margin-right:.45rem; }
        .health-clear { color:#78dba9; font-weight:700; margin-right:.45rem; }
        .health-attention { color:#f0c277; font-weight:700; margin-right:.45rem; }
        .insight-card { height:136px; box-sizing:border-box; overflow:hidden; display:flex; flex-direction:column;
            justify-content:space-between; padding:13px 14px; border:1px solid #2a3a54; border-radius:12px;
            background:linear-gradient(145deg,#142238,#101a2b); }
        .insight-label { color:#9eb1cb; font-size:.69rem; letter-spacing:.075em; font-weight:700;
            line-height:1.25; min-height:1.2rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
        .insight-value { color:#f4f8ff; font-size:clamp(1.4rem,1.75vw,1.9rem); font-weight:700;
            line-height:1.15; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
        .insight-note { color:#a9bad1; font-size:.78rem; line-height:1.25; min-height:1rem; }
        </style>""",
        unsafe_allow_html=True,
    )


def _price_figure(data: SynopsisData) -> go.Figure:
    frame = data.market_data.prices
    unit = _price_unit(data.profile.ticker)
    fig = go.Figure(
        go.Scatter(
            x=frame.index,
            y=frame["close"],
            mode="lines",
            name="Close",
            line={"color": "#43a5ff", "width": 2},
            hovertemplate=(
                "%{x|%d %b %Y}<br>Close: %{y:,.2f} " + unit + "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=f"{data.profile.ticker} · Historical price",
        xaxis_title=None,
        yaxis_title=f"Price ({unit or 'units'})",
        height=340,
        margin={"l": 18, "r": 18, "t": 48, "b": 18},
        hovermode="x unified",
        template="plotly_dark",
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220",
        font={"color": "#dbe6f5"},
        xaxis={"rangeslider": {"visible": False}, "showgrid": False},
        yaxis={"gridcolor": "#25334a", "fixedrange": False},
        hoverlabel={"bgcolor": "#16243a", "font": {"color": "#f3f7ff", "size": 12}},
    )
    return fig


def _returns_figure(data: SynopsisData) -> go.Figure:
    values = data.returns.percent
    fig = go.Figure(
        go.Scatter(
            x=values.index,
            y=values,
            mode="lines",
            name="Daily log return",
            line={"color": "#60d5b0", "width": 1},
            hovertemplate="%{x|%d %b %Y}<br>Return: %{y:.3f}%<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_color="#8798ad", line_width=1)
    return _chart_layout(fig, "Daily log returns", "Log return (%)", height=275)


def _rolling_volatility_figure(data: SynopsisData) -> go.Figure:
    metrics = calculate_dashboard_metrics(data.market_data, data.returns)
    fig = go.Figure(
        go.Scatter(
            x=metrics.rolling_volatility_series_pct.index,
            y=metrics.rolling_volatility_series_pct,
            mode="lines",
            name="20-session volatility",
            line={"color": "#f0b85a", "width": 1.7},
            hovertemplate="%{x|%d %b %Y}<br>Annualised volatility: %{y:.2f}%<extra></extra>",
        )
    )
    return _chart_layout(fig, "Rolling volatility · 20 sessions", "Annualised volatility (%)", height=275)


def _drawdown_figure(data: SynopsisData) -> go.Figure:
    metrics = calculate_dashboard_metrics(data.market_data, data.returns)
    fig = go.Figure(
        go.Scatter(
            x=metrics.drawdown_series_pct.index,
            y=metrics.drawdown_series_pct,
            mode="lines",
            name="Drawdown",
            fill="tozeroy",
            line={"color": "#ef6b73", "width": 1.4},
            fillcolor="rgba(239,107,115,.16)",
            hovertemplate="%{x|%d %b %Y}<br>Drawdown: %{y:.2f}%<extra></extra>",
        )
    )
    return _chart_layout(fig, "Historical drawdown", "Peak-to-trough drawdown (%)", height=250)


def _distribution_figure(data: SynopsisData) -> go.Figure:
    values = data.returns.percent.to_numpy()
    fig = go.Figure(
        go.Histogram(
            x=values,
            nbinsx=60,
            histnorm="probability density",
            name="Empirical distribution",
            marker_color="#9278f6",
            opacity=0.78,
            hovertemplate="Return: %{x:.3f}%<br>Density: %{y:.4f}<extra></extra>",
        )
    )
    fig.add_vline(x=data.stats.mean_pct, line_dash="dash", line_color="#f0b85a", annotation_text="Mean")
    return _chart_layout(fig, "Daily return distribution", "Daily log return (%)", height=300)


def _chart_layout(fig: go.Figure, title: str, y_title: str, *, height: int) -> go.Figure:
    fig.update_layout(
        title=title,
        xaxis_title=None,
        yaxis_title=y_title,
        height=height,
        margin={"l": 18, "r": 14, "t": 46, "b": 18},
        hovermode="x unified",
        template="plotly_dark",
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220",
        font={"color": "#dbe6f5"},
        xaxis={"showgrid": False},
        yaxis={"gridcolor": "#25334a"},
        showlegend=False,
        hoverlabel={"bgcolor": "#16243a", "font": {"color": "#f3f7ff", "size": 12}},
    )
    return fig


def _render_risk_backtesting(selected_ticker: str) -> None:
    """Present saved NIFTY 50 VaR artifacts; this view never runs model fits."""
    artifacts = load_var_dashboard_artifacts()
    if not artifacts.available:
        if artifacts.status == "missing":
            st.info(artifacts.message)
        else:
            st.warning(artifacts.message)
        st.caption("Risk forecasts and tests are generated by the separate Stage 10 experiment.")
        return

    assert artifacts.forecasts is not None
    assert artifacts.summary is not None
    assert artifacts.metadata is not None
    metadata = artifacts.metadata
    evaluation = metadata["evaluation"]
    fetched_at = pd.Timestamp(metadata["data"]["data_fetched_at_utc"])
    st.markdown("#### Saved NIFTY 50 risk evaluation")
    if artifacts.status == "stale":
        st.warning(artifacts.message)
    st.caption(
        f"Data retrieved {fetched_at:%d %b %Y, %H:%M UTC} · "
        f"evaluation {evaluation['start']} to {evaluation['end']} "
        f"({evaluation['observations']} observations) · "
        f"evaluation artifacts saved {artifacts.artifacts_saved_at:%d %b %Y, %H:%M UTC}"
    )
    if selected_ticker != "^NSEI":
        st.info(
            "The risk results below are for NIFTY 50 (^NSEI); the selected exploratory asset is "
            f"{html.escape(selected_ticker)}."
        )

    st.markdown(
        "One-day VaR is shown as a positive loss threshold. A breach occurs when the realised loss "
        "is greater than that threshold. The lines compare realised losses with 95% and 99% VaR; "
        "crosses mark breach dates."
    )
    figure = plot_var_forecasts(
        artifacts.forecasts,
        title=f"NIFTY 50 realised losses and VaR · {evaluation['start']} to {evaluation['end']}",
    )
    st.pyplot(figure, clear_figure=True, use_container_width=True)

    st.markdown("#### Backtest comparison")
    display = artifacts.summary.loc[:, [
        "method", "confidence", "breach_count", "observed_breach_rate", "expected_breach_rate",
        "kupiec_statistic", "kupiec_p_value",
        "christoffersen_independence_statistic", "christoffersen_independence_p_value",
        "conditional_coverage_statistic", "conditional_coverage_p_value",
    ]].rename(columns={
        "method": "Method",
        "confidence": "Confidence",
        "breach_count": "Breaches",
        "observed_breach_rate": "Observed rate",
        "expected_breach_rate": "Expected rate",
        "kupiec_statistic": "Kupiec LR",
        "kupiec_p_value": "Kupiec p",
        "christoffersen_independence_statistic": "Independence LR",
        "christoffersen_independence_p_value": "Independence p",
        "conditional_coverage_statistic": "Conditional coverage LR",
        "conditional_coverage_p_value": "Conditional coverage p",
    }).copy()
    display["Confidence"] = display["Confidence"].map(lambda value: f"{value:.0%}")
    display["Observed rate"] = display["Observed rate"].map(lambda value: f"{value:.2%}")
    display["Expected rate"] = display["Expected rate"].map(lambda value: f"{value:.2%}")
    for column in ("Kupiec LR", "Independence LR", "Conditional coverage LR"):
        display[column] = display[column].map(lambda value: f"{value:.3f}")
    for column in ("Kupiec p", "Independence p", "Conditional coverage p"):
        display[column] = display[column].map(lambda value: f"{value:.4f}")
    st.dataframe(display, hide_index=True, use_container_width=True)
    st.caption(
        "Kupiec tests unconditional breach frequency; Christoffersen independence tests breach clustering; "
        "conditional coverage tests both. At α = 0.05, p < 0.05 rejects the stated null. "
        "Failing to reject a null does not prove that a VaR method is valid."
    )
    st.warning(
        "These results describe one historical 252-observation evaluation period and do not guarantee future risk performance. "
        "At 99% confidence only about 2.52 breaches are expected in 252 observations, so backtests have limited statistical power."
    )


def _render_dashboard(data: SynopsisData) -> None:
    profile, stats = data.profile, data.stats
    metrics = calculate_dashboard_metrics(data.market_data, data.returns)
    ticker = profile.ticker
    unit = _price_unit(ticker)
    ticker_text = html.escape(ticker)
    source_text = html.escape(profile.source.replace("_", " ").title())
    cache_status = "cached" if profile.from_cache else "updated"

    st.markdown(
        f"""<div class="product-header">
          <div><div class="product-title">Volatility Analytics</div>
          <div class="product-subtitle">Market Risk Engine</div></div>
          <div class="data-status"><span class="status-dot">●</span> {source_text} data · {cache_status}</div>
        </div>
        <div class="metadata-strip">
          <strong>{ticker_text}</strong><span>Frequency {html.escape(profile.frequency)}</span>
          <span>{profile.first_price_date:%d %b %Y} – {profile.last_price_date:%d %b %Y}</span>
          <span>{profile.price_observations:,} observations</span>
          <span>Fetched {profile.fetched_at:%d %b %Y · %H:%M UTC}</span>
        </div>""",
        unsafe_allow_html=True,
    )

    if profile.price_observations < MIN_SYNOPSIS_OBSERVATIONS:
        st.warning(
            f"Short sample: {profile.price_observations} price observations "
            "(500 recommended for this synopsis)."
        )

    st.markdown("#### Market snapshot")
    kpis = st.columns(5, gap="small")
    price_suffix = f" {unit}" if unit else ""
    kpis[0].metric("Latest price", _format_metric(metrics.latest_price, price_suffix, 2))
    kpis[1].metric(
        "Last completed daily return",
        _format_metric(metrics.latest_completed_return_pct, "%", 3),
        help=(
            f"Observation date: {metrics.latest_completed_return_date}"
            if metrics.latest_completed_return_date
            else "No completed return is available."
        ),
    )
    kpis[2].metric("20D volatility · ann.", _format_metric(metrics.rolling_volatility_20_pct, "%", 2))
    kpis[3].metric("20D return · historical", _format_metric(metrics.trailing_20_session_return_pct, "%", 2))
    kpis[4].metric("Maximum drawdown", _format_metric(metrics.maximum_drawdown_pct, "%", 2))

    st.plotly_chart(
        _price_figure(data),
        use_container_width=True,
        config={"displaylogo": False, "scrollZoom": True},
    )

    behavior_tab, distribution_tab, profile_tab, risk_tab = st.tabs(
        ["Market behaviour", "Returns & distribution", "Profile & data health", "Risk & Backtesting"]
    )
    with behavior_tab:
        left, right = st.columns(2, gap="small")
        with left:
            st.plotly_chart(_returns_figure(data), use_container_width=True, config={"displaylogo": False})
        with right:
            st.plotly_chart(
                _rolling_volatility_figure(data),
                use_container_width=True,
                config={"displaylogo": False},
            )
        st.plotly_chart(_drawdown_figure(data), use_container_width=True, config={"displaylogo": False})

    with distribution_tab:
        st.plotly_chart(_distribution_figure(data), use_container_width=True, config={"displaylogo": False})
        stats_cols = st.columns(4, gap="small")
        stats_cols[0].metric("Count", f"{stats.count:,}")
        stats_cols[1].metric("Mean / median", f"{stats.mean_pct:.3f}% / {stats.median_pct:.3f}%")
        stats_cols[2].metric("Std. deviation", f"{stats.std_pct:.3f}%")
        stats_cols[3].metric("Skew / excess kurtosis", f"{stats.skewness:.2f} / {stats.excess_kurtosis:.2f}")
        st.caption(
            f"Observed range: {stats.minimum_pct:.2f}% to {stats.maximum_pct:.2f}% · Daily log returns"
        )

    with profile_tab:
        q = profile.quality
        st.markdown("#### Data health")
        health = st.columns(6, gap="small")
        health[0].metric("Price observations", f"{profile.price_observations:,}")
        health[1].metric("Return observations", f"{profile.return_observations:,}")
        health[2].metric("Missing closes removed", q.missing_close_rows)
        health[3].metric("Duplicate timestamps", q.duplicate_timestamps)
        health[4].metric("Gap-flagged returns", q.gap_flagged_returns)
        health[5].metric("Missing closes remaining", 0)

        good, attention = st.columns(2, gap="small")
        with good:
            good.markdown(
                f"<div class='health-line'><span class='health-good'>GOOD</span>"
                f"{q.duplicate_timestamps} duplicate timestamps · 0 missing closes remaining</div>",
                unsafe_allow_html=True,
            )
        provisional_note = (
            f"Latest bar {q.provisional_bar} may be provisional "
            f"({'excluded from returns' if q.provisional_bar_excluded else 'included in returns'})"
            if q.provisional_bar
            else "No latest bar flagged as provisional"
        )
        has_attention = bool(q.gap_flagged_returns or q.provisional_bar)
        attention_label = "ATTENTION" if has_attention else "CLEAR"
        attention_class = "health-attention" if has_attention else "health-clear"
        with attention:
            attention.markdown(
                f"<div class='health-line'><span class='{attention_class}'>{attention_label}</span>"
                f"{q.gap_flagged_returns} gap-flagged returns · {html.escape(provisional_note)}</div>",
                unsafe_allow_html=True,
            )
        st.caption("Gap-spanning returns are retained and flagged for review; they are not treated as missing observations.")

    with risk_tab:
        _render_risk_backtesting(ticker)

    st.markdown("#### What the data shows")
    findings = insight_cards(stats, metrics)
    finding_cols = st.columns(len(findings), gap="small")
    for column, finding in zip(finding_cols, findings):
        column.markdown(
            f"<div class='insight-card'><div class='insight-label'>{html.escape(finding.label)}</div>"
            f"<div class='insight-value'>{html.escape(finding.value)}</div>"
            f"<div class='insight-note'>{html.escape(finding.note)}</div></div>",
            unsafe_allow_html=True,
        )


def main() -> None:
    """Render the market analytics dashboard."""
    st.set_page_config(page_title=f"{APP_TITLE} · Market Risk Engine", layout="wide")
    _apply_visual_style()
    try:
        settings = get_settings()
        configure_logging(settings.logging)
    except ConfigurationError as exc:
        logger.error("Invalid configuration: %s", exc)
        st.error("Application configuration is invalid.")
        st.stop()

    configured_group = next(
        (name for name, assets in ASSET_PRESETS.items() if settings.data.ticker in assets.values()),
        "Custom",
    )
    market_options = ["India", "United States", "Custom"]
    with st.sidebar.form("market_controls"):
        st.markdown("## Market controls")
        group = st.selectbox("Market", market_options, index=market_options.index(configured_group))
        if group == "Custom":
            selection = "Custom ticker"
            custom_ticker = st.text_input("Ticker symbol", value=settings.data.ticker)
        else:
            choices = list(ASSET_PRESETS[group])
            symbols = list(ASSET_PRESETS[group].values())
            default_ticker = settings.data.ticker
            if default_ticker not in symbols:
                default_ticker = DEFAULT_SYNOPSIS_TICKER if group == "India" else symbols[0]
            default_index = symbols.index(default_ticker)
            selection = st.selectbox("Asset", choices, index=default_index)
            custom_ticker = ""
        ticker = resolve_ticker(selection, custom_ticker=custom_ticker)
        date_range = st.date_input(
            "Historical period",
            value=(settings.data.start_date, date.today()),
            min_value=date(1900, 1, 1),
            max_value=date.today(),
        )
        submitted = st.form_submit_button("Load market data", type="primary", use_container_width=True)

    if submitted:
        if not ticker:
            st.sidebar.error("Enter a ticker symbol.")
        elif not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
            st.sidebar.error("Select both a start date and an end date.")
        elif date_range[0] >= date_range[1]:
            st.sidebar.error("The start date must be earlier than the end date.")
        else:
            try:
                with st.spinner(f"Loading {ticker}…"):
                    st.session_state["market_dashboard_data"] = load_synopsis_data(
                        _get_market_data_loader(), ticker, date_range[0], date_range[1]
                    )
            except (DataError, PreprocessingError) as exc:
                logger.warning("Dashboard data load failed for %s: %s", ticker, exc)
                st.error(f"Unable to load {ticker}. Check the symbol, date range, or data connection.")

    data = st.session_state.get("market_dashboard_data")
    if data is None:
        st.title("Volatility Analytics")
        st.markdown("### Market Risk Engine")
        st.info("Select an asset and date range in Market controls to view its historical profile.")
    else:
        _render_dashboard(data)


if __name__ == "__main__":
    main()

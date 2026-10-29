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
from src.backtesting.var_tests import christoffersen_conditional_coverage, christoffersen_independence, kupiec_pof
from src.data.factory import create_market_data_loader
from src.data.loader import MarketDataLoader
from src.data.market_data import MarketData, MarketDataRequest
from src.evaluation.unified import classify_volatility_regime, fz0_var_es_score, summarize_by_regime
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
RISK_MONITORING_RESULTS_DIR = Path(__file__).resolve().parent / "results" / "risk_monitoring" / "nifty50"
STRESS_TESTING_RESULTS_DIR = Path(__file__).resolve().parent / "results" / "stress_testing" / "nifty50"
ADVANCED_FORECAST_RESULTS_DIR = Path(__file__).resolve().parent / "results" / "advanced_forecasting" / "nifty50"
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


@dataclass(frozen=True)
class ForecastDashboardArtifacts:
    """Validated saved Stage 12A forecast artifacts for dashboard display."""

    forecasts: pd.DataFrame | None
    metrics: pd.DataFrame | None
    metadata: dict[str, Any] | None
    status: str
    message: str

    @property
    def available(self) -> bool:
        return self.forecasts is not None and self.metrics is not None and self.metadata is not None


@dataclass(frozen=True)
class Stage13DashboardArtifacts:
    """Validated Stage 13 rolling-risk and stress artifacts for display."""

    monitoring: pd.DataFrame | None
    summary: dict[str, Any] | None
    risk_metadata: dict[str, Any] | None
    scenarios: pd.DataFrame | None
    stress_metadata: dict[str, Any] | None
    status: str
    message: str

    @property
    def available(self) -> bool:
        return all(value is not None for value in (self.monitoring, self.summary, self.risk_metadata, self.scenarios, self.stress_metadata))


def _metadata_is_stale(metadata: dict[str, Any], *, now: datetime, saved_at: datetime) -> bool:
    """Treat either old source data or an old artifact generation as stale."""
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    generated_value = metadata.get("created_at_utc")
    generated = datetime.fromisoformat(generated_value) if generated_value else saved_at
    retrieved = datetime.fromisoformat(
        metadata.get("data", {}).get("retrieved_at_utc")
        or metadata.get("data", {}).get("data_fetched_at_utc")
        or metadata.get("data_fetched_at_utc")
    )
    if generated.tzinfo is None or retrieved.tzinfo is None:
        raise ValueError("artifact timestamps must include a timezone")
    limit = VAR_RESULTS_STALE_AFTER
    return any(now.astimezone(timezone.utc) - stamp.astimezone(timezone.utc) > limit for stamp in (generated, retrieved, saved_at))


def load_stage13_dashboard_artifacts(
    risk_directory: Path | None = None,
    stress_directory: Path | None = None,
    *,
    now: datetime | None = None,
) -> Stage13DashboardArtifacts:
    """Read saved Stage 13 files and verify their schemas and recorded date ranges."""
    risk_directory = RISK_MONITORING_RESULTS_DIR if risk_directory is None else risk_directory
    stress_directory = STRESS_TESTING_RESULTS_DIR if stress_directory is None else stress_directory
    risk_paths = (risk_directory / "daily_risk_monitoring.csv", risk_directory / "risk_summary.json", risk_directory / "metadata.json")
    stress_paths = (stress_directory / "scenario_results.csv", stress_directory / "metadata.json")
    if any(not path.is_file() for path in (*risk_paths, *stress_paths)):
        return Stage13DashboardArtifacts(None, None, None, None, None, "missing", "Stage 13 monitoring and stress results are not saved yet.")
    try:
        monitoring = pd.read_csv(risk_paths[0], parse_dates=["date"], index_col="date")
        summary = json.loads(risk_paths[1].read_text(encoding="utf-8"))
        risk_metadata = json.loads(risk_paths[2].read_text(encoding="utf-8"))
        scenarios = pd.read_csv(stress_paths[0])
        stress_metadata = json.loads(stress_paths[1].read_text(encoding="utf-8"))
        levels = [int(round(float(level) * 100)) for level in risk_metadata["risk_configuration"]["confidence_levels"]]
        volatility_column = f"realised_volatility_{int(risk_metadata['risk_configuration']['realised_volatility_window'])}d_annualised_pct"
        risk_columns = {"log_return_pct", "realised_loss_pct", volatility_column, "drawdown_pct"}
        for level in levels:
            risk_columns.update({f"historical_var_{level}_pct", f"historical_es_{level}_pct", f"historical_breach_{level}"})
        if not isinstance(monitoring.index, pd.DatetimeIndex) or monitoring.empty or monitoring.index.hasnans or monitoring.index.has_duplicates or not monitoring.index.is_monotonic_increasing:
            raise ValueError("risk-monitoring dates must be present, unique, and chronological")
        if not risk_columns.issubset(monitoring.columns):
            raise ValueError("risk-monitoring columns do not match the Stage 13 schema")
        numeric = monitoring.select_dtypes(include=["number"]).to_numpy(dtype="float64")
        if np.isinf(numeric).any():
            raise ValueError("risk-monitoring series contain infinite values")
        if not isinstance(summary, dict) or not isinstance(risk_metadata, dict) or risk_metadata.get("artifact_schema_version") != 1 or risk_metadata.get("experiment") != "Stage 13A trailing historical risk monitoring":
            raise ValueError("risk-monitoring metadata is invalid")
        evaluation = summary
        risk_configuration = risk_metadata["risk_configuration"]
        if not isinstance(risk_configuration, dict) or not isinstance(risk_metadata["data"], dict):
            raise ValueError("risk-monitoring metadata is incomplete")
        if not isinstance(summary.get("latest"), dict) or not isinstance(summary.get("breaches"), dict):
            raise ValueError("risk summary is incomplete")
        if not np.isfinite(float(summary["maximum_drawdown_pct"])):
            raise ValueError("risk summary contains a non-finite drawdown")
        for level in levels:
            if str(level) not in summary["breaches"]:
                raise ValueError("risk summary is missing a confidence level")
            for key in ("all_count", "all_rate", "recent_count", "recent_rate", "recent_observations"):
                if key not in summary["breaches"][str(level)]:
                    raise ValueError("risk summary breach statistics are incomplete")
        if monitoring.index[0].date().isoformat() != evaluation["start"] or monitoring.index[-1].date().isoformat() != evaluation["end"]:
            raise ValueError("risk-monitoring dates do not match summary metadata")
        stress_columns = {"scenario_name", "scenario_type", "scenario_start", "scenario_end", "assumptions", "starting_value", "loss_amount", "loss_pct", "stressed_value", "asset_simple_returns", "asset_contributions"}
        if scenarios.empty or not stress_columns.issubset(scenarios.columns):
            raise ValueError("stress scenario columns do not match the Stage 13 schema")
        stress_numeric = scenarios[["starting_value", "loss_amount", "loss_pct", "stressed_value"]].to_numpy(dtype="float64")
        if not np.isfinite(stress_numeric).all():
            raise ValueError("stress scenarios contain incomplete or non-finite values")
        if not isinstance(stress_metadata, dict) or stress_metadata.get("artifact_schema_version", 1) != 1 or stress_metadata.get("experiment") != "Stage 13B historical and hypothetical portfolio stress scenarios":
            raise ValueError("stress metadata is invalid")
        portfolio = stress_metadata.get("portfolio")
        if not isinstance(portfolio, dict) or not {"name", "currency", "starting_value", "weights", "scope_note"}.issubset(portfolio):
            raise ValueError("stress portfolio metadata is incomplete")
        current = now or datetime.now(timezone.utc)
        saved_at = min(datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc) for path in (*risk_paths, *stress_paths))
        stale = _metadata_is_stale(risk_metadata, now=current, saved_at=saved_at) or _metadata_is_stale(stress_metadata, now=current, saved_at=saved_at)
        status = "stale" if stale else "ready"
        message = "Saved Stage 13 results are older than 30 days; rerun the experiment to refresh them." if stale else "Saved Stage 13 results loaded."
        return Stage13DashboardArtifacts(monitoring, summary, risk_metadata, scenarios, stress_metadata, status, message)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError, json.JSONDecodeError, pd.errors.ParserError) as exc:
        logger.warning("Saved Stage 13 artifacts are invalid: %s", exc)
        return Stage13DashboardArtifacts(None, None, None, None, None, "incomplete", "Saved Stage 13 artifacts are incomplete or invalid.")


def _render_stage13_risk_monitoring() -> None:
    """Present saved rolling-risk and stress outputs without fitting models."""
    artifacts = load_stage13_dashboard_artifacts()
    if not artifacts.available:
        st.caption(artifacts.message)
        return
    if artifacts.status == "stale":
        st.warning(artifacts.message)
    assert artifacts.monitoring is not None and artifacts.summary is not None
    assert artifacts.risk_metadata is not None and artifacts.scenarios is not None
    assert artifacts.stress_metadata is not None
    frame, summary, risk_meta = artifacts.monitoring, artifacts.summary, artifacts.risk_metadata
    risk_config = risk_meta["risk_configuration"]
    volatility_name = f"realised_volatility_{risk_config['realised_volatility_window']}d_annualised_pct"
    evaluation = f"{summary['start']} to {summary['end']}"
    st.markdown("#### Rolling risk monitoring")
    st.caption(
        f"{risk_meta['data']['ticker']} · {evaluation} · source {risk_meta['data']['source']} · "
        f"{risk_meta['data']['return_metadata']['return_definition']}"
    )
    latest_vol = frame[volatility_name].dropna()
    latest_drawdown = frame["drawdown_pct"].dropna()
    risk_levels = [int(round(float(level) * 100)) for level in risk_config["confidence_levels"]]
    metric_columns = st.columns(2 + 2 * len(risk_levels), gap="small")
    metric_columns[0].metric(f"{risk_config['realised_volatility_window']}D realised vol · ann.", f"{latest_vol.iloc[-1]:.2f}%" if len(latest_vol) else "N/A")
    metric_columns[1].metric("Maximum drawdown", f"{summary['maximum_drawdown_pct']:.2f}%")
    for i, level in enumerate(risk_levels):
        metric_columns[2 + 2 * i].metric(f"Historical VaR {level}%", f"{summary['latest'].get(f'historical_var_{level}_pct', float('nan')):.2f}%")
        metric_columns[3 + 2 * i].metric(f"Expected Shortfall {level}%", f"{summary['latest'].get(f'historical_es_{level}_pct', float('nan')):.2f}%")
    from plotly.subplots import make_subplots

    chart = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12, subplot_titles=("Trailing realised volatility (annualised)", "Prior-window historical VaR and Expected Shortfall"))
    chart.add_trace(go.Scatter(x=frame.index, y=frame[volatility_name], name="Realised volatility", line={"color": "#40c4a0"}), row=1, col=1)
    palette = {95: ("#f2b84b", "#ff7188"), 99: ("#8b9eff", "#d783ff")}
    for level in risk_levels:
        var_color, es_color = palette.get(level, ("#f2b84b", "#ff7188"))
        chart.add_trace(go.Scatter(x=frame.index, y=frame[f"historical_var_{level}_pct"], name=f"VaR {level}%", line={"color": var_color}), row=2, col=1)
        chart.add_trace(go.Scatter(x=frame.index, y=frame[f"historical_es_{level}_pct"], name=f"ES {level}%", line={"color": es_color, "dash": "dot"}), row=2, col=1)
    chart.update_yaxes(title_text="Annualised volatility (%)", row=1, col=1)
    chart.update_yaxes(title_text="Loss threshold (% log return)", row=2, col=1)
    chart.update_layout(height=560, margin={"l": 10, "r": 10, "t": 45, "b": 10}, paper_bgcolor="#0b1220", plot_bgcolor="#0b1220", font={"color": "#dbe6f5"}, legend={"orientation": "h"})
    st.plotly_chart(chart, use_container_width=True, config={"displaylogo": False})

    breach_rows = []
    for level in risk_levels:
        stats = summary["breaches"][str(level)]
        breach_series = frame[f"historical_breach_{level}"].dropna().astype(bool).to_numpy()
        tail_probability = 1.0 - level / 100.0
        coverage_test = kupiec_pof(breach_series, tail_probability)
        independence_test = christoffersen_independence(breach_series, tail_probability)
        conditional_test = christoffersen_conditional_coverage(breach_series, tail_probability)
        breach_rows.append({
            "Confidence": f"{level}%",
            "All-period breaches": stats["all_count"],
            "All-period rate": stats["all_rate"],
            "Expected rate": tail_probability,
            "Kupiec p": coverage_test.p_value,
            "Independence p": independence_test.p_value,
            "Conditional coverage p": conditional_test.p_value,
            "Recent breaches": stats["recent_count"],
            "Recent rate": stats["recent_rate"],
            "Recent valid observations": stats["recent_observations"],
        })
    st.dataframe(pd.DataFrame(breach_rows), hide_index=True, use_container_width=True)
    st.caption(
        "Realised volatility describes trailing returns through each date. Historical VaR and Expected Shortfall use only prior returns; "
        "loss is the negative log return, VaR is its empirical confidence quantile, and ES is the average loss at or beyond VaR. "
        "Drawdown uses adjusted close versus its running peak. All-period tests use the Stage 13 rolling historical VaR breaches; Kupiec tests unconditional coverage, Christoffersen tests independence, and conditional coverage combines both. These asymptotic tests have low power for rare breaches and are descriptive for this one sample."
    )

    st.markdown("#### Stress scenarios")
    stress = artifacts.scenarios
    portfolio = artifacts.stress_metadata["portfolio"]
    st.caption(
        f"{portfolio['name']} · starting value {portfolio['currency']} {portfolio['starting_value']:,.0f} · "
        f"weights {portfolio['weights']} · {portfolio['scope_note']}"
    )
    view = stress.loc[:, ["scenario_name", "scenario_type", "scenario_start", "scenario_end", "assumptions", "starting_value", "loss_amount", "loss_pct", "stressed_value", "asset_simple_returns", "asset_contributions"]].rename(columns={
        "scenario_name": "Scenario", "scenario_type": "Type", "scenario_start": "Start", "scenario_end": "End",
        "assumptions": "Assumptions", "starting_value": "Starting value", "loss_amount": "Loss", "loss_pct": "Loss (%)",
        "stressed_value": "Stressed value", "asset_simple_returns": "Asset simple returns", "asset_contributions": "Asset loss contributions",
    })
    st.dataframe(view, hide_index=True, use_container_width=True)
    stress_chart = go.Figure(go.Bar(x=stress["scenario_name"], y=stress["loss_amount"], marker_color=["#ff7188" if kind == "historical" else "#f2b84b" for kind in stress["scenario_type"]]))
    stress_chart.add_hline(y=0, line_color="#9eb1cb", line_width=1)
    stress_chart.update_layout(title="Scenario loss in portfolio currency (negative values indicate gains)", height=360, margin={"l": 10, "r": 10, "t": 55, "b": 100}, paper_bgcolor="#0b1220", plot_bgcolor="#0b1220", font={"color": "#dbe6f5"}, xaxis={"tickangle": -35}, yaxis_title=f"Loss ({portfolio['currency']})")
    st.plotly_chart(stress_chart, use_container_width=True, config={"displaylogo": False})
    st.caption("Historical scenarios are observed one- and five-session return windows. Hypothetical declines are deterministic what-if shocks with no probability claim. Results assume fully invested long-only buy-and-hold weights and exclude fees, cash flows, leverage, and rebalancing.")


def load_forecast_dashboard_artifacts(
    directory: Path | None = None,
    *,
    now: datetime | None = None,
) -> ForecastDashboardArtifacts:
    """Load and validate saved model forecasts without fitting any models."""
    directory = ADVANCED_FORECAST_RESULTS_DIR if directory is None else Path(directory)
    paths = (directory / "daily_forecasts.csv", directory / "evaluation_metrics.csv", directory / "metadata.json")
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        return ForecastDashboardArtifacts(None, None, None, "missing", "Stage 12A forecasts are not available yet.")
    try:
        forecasts = pd.read_csv(paths[0], parse_dates=["date"], index_col="date")
        metrics = pd.read_csv(paths[1])
        metadata = json.loads(paths[2].read_text(encoding="utf-8"))
        required_forecasts = {"actual", "XGBoost recursive", "ARIMA(1,1,0)", "SARIMA(1,1,0)(1,0,0,5)", "HoltWinters(trend=add)"}
        required_metrics = {"model_name", "rmse", "mae", "mape"}
        if not required_forecasts.issubset(forecasts.columns) or not required_metrics.issubset(metrics.columns):
            raise ValueError("forecast or metric columns do not match the Stage 12A schema")
        if (
            not isinstance(forecasts.index, pd.DatetimeIndex)
            or forecasts.empty or forecasts.index.hasnans
            or forecasts.index.has_duplicates or not forecasts.index.is_monotonic_increasing
        ):
            raise ValueError("forecast dates must be present, unique, and chronological")
        if not np.isfinite(forecasts.to_numpy(dtype="float64")).all():
            raise ValueError("forecast values contain missing or non-finite values")
        if not required_metrics.issubset(set(metrics.columns)) or metrics.empty:
            raise ValueError("evaluation metrics are empty or incomplete")
        if not isinstance(metadata, dict) or metadata.get("artifact_schema_version") != 1 or metadata.get("experiment") != "Stage 12A recursive XGBoost adjusted-close forecasting":
            raise ValueError("metadata does not identify the Stage 12A experiment")
        if not np.isfinite(metrics[["rmse", "mae", "mape"]].to_numpy(dtype="float64")).all():
            raise ValueError("evaluation metrics contain missing or non-finite values")
        expected_models = {"XGBoost recursive", "ARIMA(1,1,0)", "SARIMA(1,1,0)(1,0,0,5)", "HoltWinters(trend=add)"}
        if set(metrics["model_name"]) != expected_models:
            raise ValueError("evaluation metrics do not contain the expected model set")
        if len(forecasts) != int(metadata["evaluation"]["observations"]):
            raise ValueError("forecast row count does not match metadata")
        if forecasts.index[0].date().isoformat() != metadata["evaluation"]["start"] or forecasts.index[-1].date().isoformat() != metadata["evaluation"]["end"]:
            raise ValueError("forecast dates do not match metadata")
        current = now or datetime.now(timezone.utc)
        saved_at = datetime.fromtimestamp(paths[2].stat().st_mtime, tz=timezone.utc)
        stale = _metadata_is_stale(metadata, now=current, saved_at=saved_at)
        status = "stale" if stale else "ready"
        message = "Saved Stage 12A results are older than 30 days; rerun the experiment to refresh them." if stale else "Saved Stage 12A forecast results loaded."
        return ForecastDashboardArtifacts(forecasts, metrics, metadata, status, message)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError, json.JSONDecodeError, pd.errors.ParserError) as exc:
        logger.warning("Saved Stage 12A forecast artifacts are invalid: %s", exc)
        return ForecastDashboardArtifacts(None, None, None, "incomplete", "Saved Stage 12A artifacts are incomplete or invalid.")


def _render_forecasting() -> None:
    """Present the saved Stage 12A experiment; rerenders never train models."""
    artifacts = load_forecast_dashboard_artifacts()
    if not artifacts.available:
        st.info(artifacts.message) if artifacts.status == "missing" else st.warning(artifacts.message)
        return
    if artifacts.status == "stale":
        st.warning(artifacts.message)
    assert artifacts.forecasts is not None and artifacts.metrics is not None and artifacts.metadata is not None
    forecasts, metrics, metadata = artifacts.forecasts, artifacts.metrics, artifacts.metadata
    evaluation = metadata["evaluation"]
    st.markdown("#### Saved NIFTY 50 adjusted-close forecast comparison")
    st.caption(
        f"{evaluation['start']} to {evaluation['end']} · {evaluation['observations']} sessions · "
        f"XGBoost {metadata['models']['xgboost']['version']} · recursive test forecast"
    )
    chart = go.Figure()
    chart.add_trace(go.Scatter(x=forecasts.index, y=forecasts["actual"], name="Actual", line={"color": "#dbe6f5"}))
    for name, color in (("XGBoost recursive", "#40c4a0"), ("ARIMA(1,1,0)", "#f2b84b"), ("SARIMA(1,1,0)(1,0,0,5)", "#8b9eff"), ("HoltWinters(trend=add)", "#ff7188")):
        chart.add_trace(go.Scatter(x=forecasts.index, y=forecasts[name], name=name, line={"color": color}))
    chart.update_layout(height=380, margin={"l": 10, "r": 10, "t": 20, "b": 10}, paper_bgcolor="#0b1220", plot_bgcolor="#0b1220", font={"color": "#dbe6f5"}, yaxis_title="Adjusted close price")
    st.plotly_chart(chart, use_container_width=True, config={"displaylogo": False})
    display = metrics.loc[:, ["model_name", "rmse", "mae", "mape"]].rename(columns={"model_name": "Model", "rmse": "RMSE", "mae": "MAE", "mape": "MAPE (%)"})
    st.dataframe(display, hide_index=True, use_container_width=True)
    st.caption("RMSE and MAE use price units. MAPE is secondary and can mislead when targets are near zero. Scores describe this one held-out window only.")

    volatility_dir = Path(__file__).resolve().parent / "results" / "volatility_forecasting" / "nifty50"
    volatility_paths = (volatility_dir / "daily_forecasts.csv", volatility_dir / "summary_metrics.csv", volatility_dir / "metadata.json")
    if any(not path.is_file() for path in volatility_paths):
        st.caption("Saved Stage 9 volatility diagnostics are not available yet.")
    try:
        volatility = pd.read_csv(volatility_dir / "daily_forecasts.csv", parse_dates=["date"], index_col="date")
        volatility_metrics = pd.read_csv(volatility_dir / "summary_metrics.csv")
        volatility_metadata = json.loads((volatility_dir / "metadata.json").read_text(encoding="utf-8"))
        if not isinstance(volatility_metadata, dict) or volatility_metadata.get("artifact_schema_version") != 1 or volatility_metadata.get("experiment") != "Stage 9 expanding-window one-step-ahead volatility forecasting":
            raise ValueError("metadata does not identify the supported Stage 9 schema")
        if not isinstance(volatility.index, pd.DatetimeIndex) or volatility.index.empty or volatility.index.has_duplicates or not volatility.index.is_monotonic_increasing:
            raise ValueError("forecast dates are missing, duplicated, or out of order")
        if len(volatility) != int(volatility_metadata["evaluation"]["observations"]) or volatility.index[0].date().isoformat() != volatility_metadata["evaluation"]["start"] or volatility.index[-1].date().isoformat() != volatility_metadata["evaluation"]["end"]:
            raise ValueError("forecast dates do not match Stage 9 metadata")
        volatility_columns = {"realised_return_pct", "realised_squared_return_pct2", "garch_variance_pct2", "egarch_variance_pct2", "rolling_variance_pct2"}
        metric_columns = {"model", "observations", "qlike_mean_log_h_plus_rv_over_h", "mae_pct2", "rmse_pct2"}
        if not volatility_columns.issubset(volatility.columns) or volatility_metrics.empty or not metric_columns.issubset(volatility_metrics.columns):
            raise ValueError("forecast or summary columns do not match the Stage 9 schema")
        if not np.isfinite(volatility[list(volatility_columns)].to_numpy(dtype="float64")).all():
            raise ValueError("volatility forecast values are incomplete or non-finite")
        if not np.isfinite(volatility_metrics[["observations", "qlike_mean_log_h_plus_rv_over_h", "mae_pct2", "rmse_pct2"]].to_numpy(dtype="float64")).all():
            raise ValueError("volatility summary metrics are incomplete or non-finite")
        if not volatility.empty and {"realised_squared_return_pct2", "garch_variance_pct2", "egarch_variance_pct2", "rolling_variance_pct2"}.issubset(volatility.columns):
            st.markdown("#### Volatility forecast diagnostics")
            if _metadata_is_stale(volatility_metadata, now=datetime.now(timezone.utc), saved_at=datetime.fromtimestamp((volatility_dir / "metadata.json").stat().st_mtime, tz=timezone.utc)):
                st.warning("Saved Stage 9 volatility results are older than 30 days; rerun the experiment to refresh them.")
            st.caption("Variance forecasts versus squared daily log return (a noisy one-session realised-variance proxy); variance values and errors use percent². Separate one-step-ahead evaluation window.")
            st.dataframe(volatility_metrics.loc[:, ["model", "observations", "qlike_mean_log_h_plus_rv_over_h", "mae_pct2", "rmse_pct2"]].rename(columns={
                "model": "Model", "observations": "N", "qlike_mean_log_h_plus_rv_over_h": "Mean QLIKE (lower)", "mae_pct2": "Variance MAE (percent²)", "rmse_pct2": "Variance RMSE (percent²)"
            }), hide_index=True, use_container_width=True)
            prior_vol = volatility["realised_return_pct"].rolling(20, min_periods=20).std(ddof=1).shift(1) * np.sqrt(252)
            regimes = classify_volatility_regime(prior_vol, lookback=252, minimum_history=60)
            regime_values = volatility.assign(
                garch_squared_error=(volatility["garch_variance_pct2"] - volatility["realised_squared_return_pct2"]) ** 2,
                egarch_squared_error=(volatility["egarch_variance_pct2"] - volatility["realised_squared_return_pct2"]) ** 2,
            )
            regime_summary = summarize_by_regime(regime_values, regimes, metrics={
                "GARCH mean squared error (percent⁴)": "garch_squared_error",
                "EGARCH mean squared error (percent⁴)": "egarch_squared_error",
            })
            if not regime_summary.empty:
                st.caption("Regimes use prior-only 20-session annualised realised volatility versus its expanding 252-session historical median; early dates without 60 prior observations are omitted.")
                st.dataframe(regime_summary.rename(columns={"regime": "Prior-volatility regime", "observations": "N"}), hide_index=True, use_container_width=True)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError, json.JSONDecodeError, pd.errors.ParserError) as exc:
        logger.warning("Saved volatility evaluation artifacts unavailable: %s", exc)
        st.warning("Saved Stage 9 volatility diagnostics are incomplete or invalid; rerun the experiment to refresh them.")


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
        if metadata.get("artifact_schema_version") != 1 or metadata.get("experiment") != "Stage 10 one-day-ahead VaR forecasts and statistical backtesting":
            raise ValueError("metadata does not identify the supported Stage 10 artifact schema")
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
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError, json.JSONDecodeError, pd.errors.ParserError) as exc:
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
        _render_stage13_risk_monitoring()
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
    stage13 = load_stage13_dashboard_artifacts()
    if stage13.available and stage13.monitoring is not None and stage13.risk_metadata is not None:
        risk_frame = stage13.monitoring
        risk_config = stage13.risk_metadata["risk_configuration"]
        st.markdown("#### Historical VaR and Expected Shortfall joint score")
        es_rows = []
        for confidence in risk_config["confidence_levels"]:
            level = int(round(float(confidence) * 100))
            valid = risk_frame[["realised_loss_pct", f"historical_var_{level}_pct", f"historical_es_{level}_pct"]].dropna()
            if valid.empty:
                continue
            scores = fz0_var_es_score(valid["realised_loss_pct"], valid[f"historical_var_{level}_pct"], valid[f"historical_es_{level}_pct"], 1.0 - float(confidence))
            es_rows.append({"Confidence": f"{level}%", "N": len(valid), "FZ0 joint score (lower)": float(np.mean(scores)), "Evaluation": f"{valid.index[0].date()} to {valid.index[-1].date()}"})
        if es_rows:
            st.dataframe(pd.DataFrame(es_rows), hide_index=True, use_container_width=True)
            st.caption("FZ0 is a joint scoring rule for positive loss VaR and ES forecasts. Scores assess the Stage 13 rolling historical method and are not directly comparable with price or variance forecast errors.")
    _render_stage13_risk_monitoring()


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

    behavior_tab, distribution_tab, profile_tab, forecast_tab, risk_tab = st.tabs(
        ["Market behaviour", "Returns & distribution", "Profile & data health", "Forecasting", "Risk & Backtesting"]
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

    with forecast_tab:
        _render_forecasting()

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

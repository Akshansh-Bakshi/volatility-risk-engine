"""Smoke tests that the Streamlit entry point renders."""

from __future__ import annotations

import json
from pathlib import Path
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import app
from app import _price_unit, load_forecast_dashboard_artifacts, load_synopsis_data, load_var_dashboard_artifacts, resolve_ticker
from tests.data_fakes import make_market_data, make_prices

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
APP_TITLE = "Volatility Analytics"


def _run_app() -> AppTest:
    return AppTest.from_file(str(APP_PATH), default_timeout=30).run()


def test_app_renders_without_errors() -> None:
    at = _run_app()

    assert not at.exception
    assert [title.value for title in at.title] == [APP_TITLE]
    assert not at.error


def test_app_does_not_expose_configuration_or_internal_details() -> None:
    at = _run_app()

    assert not at.exception
    assert not at.json
    visible = " ".join(
        [element.value for group in (at.title, at.header, at.subheader, at.markdown, at.caption, at.info)
         for element in group]
    )
    assert "Active default configuration" not in visible
    assert "Proposed methodology" not in visible
    assert "C:\\Users\\" not in visible
    assert "Volatility Analytics" in visible


def test_app_reports_invalid_configuration_instead_of_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VRE_LOG_LEVEL", "VERBOSE")

    at = _run_app()

    assert not at.exception
    assert len(at.error) == 1
    assert "configuration" in at.error[0].value.lower()
    assert not at.json


def test_synopsis_orchestration_uses_loader_and_existing_eda() -> None:
    class StubLoader:
        def __init__(self) -> None:
            self.request = None

        def load(self, request):
            self.request = request
            return make_market_data(
                make_prices(periods=520), ticker=request.ticker, source="test_provider"
            )

    loader = StubLoader()
    result = load_synopsis_data(loader, "^NSEI", date(2022, 1, 1), date(2024, 1, 1))

    assert loader.request.ticker == "^NSEI"
    assert loader.request.start == date(2022, 1, 1)
    assert loader.request.end == date(2024, 1, 1)
    assert result.profile.source == "test_provider"
    assert result.profile.price_observations == 520
    assert result.profile.return_observations == 518
    assert result.profile.quality.provisional_bar is not None
    assert result.profile.quality.provisional_bar_excluded
    assert result.stats.count == result.returns.return_observations


def test_asset_presets_and_custom_ticker_resolution() -> None:
    assert resolve_ticker("NIFTY 50 (^NSEI)") == "^NSEI"
    assert resolve_ticker("Apple (AAPL)") == "AAPL"
    assert resolve_ticker("Custom ticker", custom_ticker="  msft ") == "MSFT"


def test_asset_price_units_are_correct_for_indices_and_equities() -> None:
    assert _price_unit("^NSEI") == "index points"
    assert _price_unit("RELIANCE.NS") == "₹"
    assert _price_unit("AAPL") == "$"


def _write_var_artifacts(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    index = pd.bdate_range("2025-09-30", periods=252, name="date")
    forecasts = pd.DataFrame({"realised_loss_pct": [0.5] * 252}, index=index)
    for prefix, threshold95, threshold99 in (
        ("garch", 1.0, 1.8),
        ("egarch", 1.1, 1.9),
        ("historical", 1.2, 2.0),
    ):
        forecasts[f"{prefix}_var95_pct"] = threshold95
        forecasts[f"{prefix}_var99_pct"] = threshold99
        forecasts[f"{prefix}_breach95"] = False
        forecasts[f"{prefix}_breach99"] = False
    forecasts.to_csv(directory / "daily_var_forecasts.csv", index_label="date")

    methods = ["GARCH Normal", "EGARCH Normal", "250-return historical simulation"]
    summary = pd.DataFrame([
        {
            "method": method,
            "confidence": confidence,
            "breach_count": 5,
            "sample_size": 252,
            "observed_breach_rate": 5 / 252,
            "expected_breach_rate": 1 - confidence,
            "kupiec_statistic": 0.1,
            "kupiec_p_value": 0.75,
            "christoffersen_independence_statistic": 0.2,
            "christoffersen_independence_p_value": 0.65,
            "conditional_coverage_statistic": 0.3,
            "conditional_coverage_p_value": 0.85,
        }
        for confidence in (0.95, 0.99)
        for method in methods
    ])
    summary.to_csv(directory / "backtest_summary.csv", index=False)

    metadata = {
        "artifact_schema_version": 1,
        "experiment": "Stage 10 one-day-ahead VaR forecasts and statistical backtesting",
        "data": {
            "ticker": "^NSEI",
            "data_fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "evaluation": {
            "start": index[0].date().isoformat(),
            "end": index[-1].date().isoformat(),
            "observations": 252,
        },
    }
    (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_var_artifact_loader_reads_complete_stage10_results(tmp_path: Path) -> None:
    _write_var_artifacts(tmp_path)

    result = load_var_dashboard_artifacts(tmp_path)

    assert result.available
    assert result.status == "ready"
    assert len(result.forecasts) == 252
    assert len(result.summary) == 6
    assert result.metadata["data"]["ticker"] == "^NSEI"
    assert result.artifacts_saved_at is not None


def test_var_artifact_loader_handles_missing_and_incomplete_files(tmp_path: Path) -> None:
    missing = load_var_dashboard_artifacts(tmp_path)
    assert not missing.available
    assert missing.status == "missing"
    assert "unavailable" in missing.message

    _write_var_artifacts(tmp_path)
    forecasts = pd.read_csv(tmp_path / "daily_var_forecasts.csv").drop(columns="garch_var95_pct")
    forecasts.to_csv(tmp_path / "daily_var_forecasts.csv", index=False)
    incomplete = load_var_dashboard_artifacts(tmp_path)
    assert not incomplete.available
    assert incomplete.status == "incomplete"
    assert "incomplete or invalid" in incomplete.message


def test_var_artifact_loader_marks_old_retrieval_as_stale(tmp_path: Path) -> None:
    _write_var_artifacts(tmp_path)
    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["data"]["data_fetched_at_utc"] = "2025-01-01T00:00:00+00:00"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    result = load_var_dashboard_artifacts(tmp_path)

    assert result.available
    assert result.status == "stale"
    assert "older than 30 days" in result.message


def test_var_artifact_loader_rejects_incompatible_schema(tmp_path: Path) -> None:
    _write_var_artifacts(tmp_path)
    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["artifact_schema_version"] = 99
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    result = load_var_dashboard_artifacts(tmp_path)

    assert not result.available
    assert result.status == "incomplete"


def _write_forecast_artifacts(directory: Path, *, created_at: datetime | None = None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    index = pd.bdate_range("2025-01-02", periods=3, name="date")
    pd.DataFrame({
        "actual": [100.0, 101.0, 102.0],
        "XGBoost recursive": [100.1, 100.8, 102.2],
        "ARIMA(1,1,0)": [100.0, 101.1, 101.9],
        "SARIMA(1,1,0)(1,0,0,5)": [100.2, 100.9, 102.1],
        "HoltWinters(trend=add)": [99.9, 101.0, 102.0],
    }, index=index).to_csv(directory / "daily_forecasts.csv", index_label="date")
    pd.DataFrame([
        {"model_name": name, "rmse": 0.2, "mae": 0.1, "mape": 0.1}
        for name in ("XGBoost recursive", "ARIMA(1,1,0)", "SARIMA(1,1,0)(1,0,0,5)", "HoltWinters(trend=add)")
    ]).to_csv(directory / "evaluation_metrics.csv", index=False)
    created = created_at or datetime.now(timezone.utc)
    metadata = {
        "artifact_schema_version": 1,
        "experiment": "Stage 12A recursive XGBoost adjusted-close forecasting",
        "created_at_utc": created.isoformat(),
        "data": {"retrieved_at_utc": created.isoformat()},
        "evaluation": {"start": index[0].date().isoformat(), "end": index[-1].date().isoformat(), "observations": 3},
        "models": {"xgboost": {"version": "test"}},
    }
    (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_forecast_artifact_loader_handles_fresh_stale_and_bad_schema(tmp_path: Path) -> None:
    fresh_dir = tmp_path / "fresh"
    current = datetime.now(timezone.utc)
    _write_forecast_artifacts(fresh_dir, created_at=current)
    fresh = load_forecast_dashboard_artifacts(fresh_dir, now=current + timedelta(minutes=1))
    assert fresh.available and fresh.status == "ready"

    stale_dir = tmp_path / "stale"
    old = current - timedelta(days=45)
    _write_forecast_artifacts(stale_dir, created_at=old)
    stale = load_forecast_dashboard_artifacts(stale_dir, now=current)
    assert stale.available and stale.status == "stale"

    metadata_path = fresh_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["artifact_schema_version"] = 99
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    invalid = load_forecast_dashboard_artifacts(fresh_dir, now=current)
    assert not invalid.available and invalid.status == "incomplete"


def test_forecast_dashboard_rerender_reads_artifacts_without_training(tmp_path: Path) -> None:
    _write_forecast_artifacts(tmp_path, created_at=datetime.now(timezone.utc))
    script = (
        "import app\n"
        "from pathlib import Path\n"
        "from src.forecasting.xgboost import XGBoostPriceForecaster\n"
        f"app.ADVANCED_FORECAST_RESULTS_DIR = Path({str(tmp_path)!r})\n"
        "original_fit = XGBoostPriceForecaster.fit\n"
        "try:\n"
        "    XGBoostPriceForecaster.fit = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('unexpected fit'))\n"
        "    app._render_forecasting()\n"
        "finally:\n"
        "    XGBoostPriceForecaster.fit = original_fit\n"
    )
    rendered = AppTest.from_string(script, default_timeout=30).run()

    assert not rendered.exception
    assert len(rendered.dataframe) >= 1
    visible = " ".join(item.value for group in (rendered.markdown, rendered.caption) for item in group)
    assert "Saved NIFTY 50 adjusted-close forecast comparison" in visible


def test_risk_tab_renders_saved_metrics_and_scope_note(tmp_path: Path) -> None:
    _write_var_artifacts(tmp_path)
    script = (
        "import app\n"
        "from pathlib import Path\n"
        f"app.VAR_RESULTS_DIR = Path({str(tmp_path)!r})\n"
        f"app.RISK_MONITORING_RESULTS_DIR = Path({str(tmp_path / 'monitoring')!r})\n"
        f"app.STRESS_TESTING_RESULTS_DIR = Path({str(tmp_path / 'stress')!r})\n"
        "app._render_risk_backtesting('AAPL')\n"
    )
    at = AppTest.from_string(script, default_timeout=30).run()

    assert not at.exception
    assert len(at.dataframe) == 1
    table = at.dataframe[0].value
    assert "Kupiec LR" in table.columns and "Kupiec p" in table.columns
    assert "Independence LR" in table.columns and "Independence p" in table.columns
    assert "Conditional coverage LR" in table.columns
    assert "Conditional coverage p" in table.columns
    assert table.loc[0, "Method"] == "GARCH Normal"
    assert table.loc[0, "Breaches"] == 5
    visible = " ".join(
        element.value
        for group in (at.markdown, at.caption, at.info, at.warning, at.subheader)
        for element in group
    )
    assert "Saved NIFTY 50 risk evaluation" in visible
    assert "AAPL" in visible and "NIFTY 50 (^NSEI)" in visible
    assert "252-observation evaluation period" in visible
    assert "limited statistical power" in visible


def test_risk_tab_renders_gracefully_when_artifacts_are_missing(tmp_path: Path) -> None:
    script = (
        "import app\n"
        "from pathlib import Path\n"
        f"app.VAR_RESULTS_DIR = Path({str(tmp_path)!r})\n"
        f"app.RISK_MONITORING_RESULTS_DIR = Path({str(tmp_path / 'monitoring')!r})\n"
        f"app.STRESS_TESTING_RESULTS_DIR = Path({str(tmp_path / 'stress')!r})\n"
        "app._render_risk_backtesting('^NSEI')\n"
    )
    at = AppTest.from_string(script, default_timeout=30).run()

    assert not at.exception
    assert len(at.info) == 1
    assert "Saved Stage 10 results are unavailable" in at.info[0].value
    assert len(at.dataframe) == 0

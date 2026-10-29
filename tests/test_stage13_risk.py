"""Stage 13 rolling-risk, stress scenario, persistence, and dashboard checks."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import app
from scripts.run_stage13_risk_monitoring import run_stage13_experiment
from src.config import PreprocessingConfig
from src.exceptions import DataFetchError
from src.exceptions import DataFetchError
from src.preprocessing.returns import build_return_series
from src.risk.monitoring import RiskMonitoringConfig, calculate_risk_monitoring
from src.risk.stress import Portfolio, historical_scenarios, hypothetical_scenarios
from tests.data_fakes import make_market_data, prices_from_closes
import scripts.run_stage13_risk_monitoring as stage13
import scripts.run_stage13_risk_monitoring as stage13


def _market(closes: list[float]):
    dates = pd.bdate_range("2023-01-02", periods=len(closes), name="date")
    return make_market_data(prices_from_closes(closes, [x.date().isoformat() for x in dates]))


def _returns(closes: list[float]):
    data = _market(closes)
    return data, build_return_series(data, PreprocessingConfig(min_returns=2))


def _monitor_config() -> RiskMonitoringConfig:
    return RiskMonitoringConfig(
        realised_volatility_window=3,
        historical_var_window=4,
        breach_summary_window=2,
        confidence_levels=(0.95, 0.99),
    )


def test_rolling_volatility_prior_var_es_and_drawdown_alignment() -> None:
    closes = [100, 101, 99, 98, 100, 95, 97, 90, 92, 94]
    market, returns = _returns(closes)
    frame, summary = calculate_risk_monitoring(returns, market.price, _monitor_config())
    pct = returns.percent
    t = 4
    expected_vol = pct.iloc[t - 2:t + 1].std(ddof=1) * np.sqrt(252)
    losses = -pct.iloc[t - 4:t].to_numpy()
    expected_var = np.quantile(losses, 0.95, method="linear")
    expected_es = losses[losses >= expected_var].mean()
    assert frame.index.equals(pct.index)
    assert frame.iloc[t]["realised_volatility_3d_annualised_pct"] == pytest.approx(expected_vol)
    assert frame.iloc[t]["historical_var_95_pct"] == pytest.approx(expected_var)
    assert frame.iloc[t]["historical_es_95_pct"] == pytest.approx(expected_es)
    assert frame.iloc[0]["historical_var_95_pct"] is pd.NA or pd.isna(frame.iloc[0]["historical_var_95_pct"])
    assert frame["drawdown_pct"].min() == pytest.approx((90 / 101 - 1) * 100)
    assert summary["maximum_drawdown_pct"] == pytest.approx(frame["drawdown_pct"].min())
    assert summary["windows"]["annualisation_factor"] == 252


def test_historical_var_at_date_does_not_use_that_date_or_future_returns() -> None:
    closes = [100, 101, 99, 98, 100, 95, 97, 90, 92, 94]
    market, returns = _returns(closes)
    base, _ = calculate_risk_monitoring(returns, market.price, _monitor_config())
    changed_closes = closes.copy()
    changed_closes[5] *= 0.5
    changed_market, changed_returns = _returns(changed_closes)
    changed, _ = calculate_risk_monitoring(changed_returns, changed_market.price, _monitor_config())
    # Date 5's return changes, but its forecast only consumes the four earlier returns.
    assert changed.iloc[4]["historical_var_95_pct"] == pytest.approx(base.iloc[4]["historical_var_95_pct"])
    assert changed.iloc[5]["realised_loss_pct"] != pytest.approx(base.iloc[5]["realised_loss_pct"])


def test_breach_counts_rates_and_short_history_are_explicit() -> None:
    market, returns = _returns([100, 101, 99, 98, 100, 95, 97, 90, 92, 94])
    frame, summary = calculate_risk_monitoring(returns, market.price, _monitor_config())
    for level in (95, 99):
        valid = frame[f"historical_breach_{level}"].dropna().astype(bool)
        stats = summary["breaches"][str(level)]
        assert stats["all_count"] == int(valid.sum())
        assert stats["all_rate"] == pytest.approx(float(valid.mean()))
        assert "rolling_breach_rate_" + str(level) in frame

    short_config = RiskMonitoringConfig(realised_volatility_window=3, historical_var_window=100, breach_summary_window=3)
    short, short_summary = calculate_risk_monitoring(returns, market.price, short_config)
    assert short["historical_var_95_pct"].isna().all()
    assert short_summary["breaches"]["95"]["all_observations"] == 0
    assert short_summary["breaches"]["95"]["all_rate"] is None


def test_monitoring_rejects_missing_prices_and_invalid_windows() -> None:
    market, returns = _returns([100, 101, 99, 98, 100, 95])
    bad = market.price.copy()
    bad.iloc[2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        calculate_risk_monitoring(returns, bad, _monitor_config())
    with pytest.raises(ValueError, match="positive integer"):
        RiskMonitoringConfig(realised_volatility_window=0)
    with pytest.raises(ValueError, match="at least 2"):
        RiskMonitoringConfig(realised_volatility_window=1)


def test_portfolio_weight_validation_and_hypothetical_losses() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        Portfolio("bad", 100_000, {"A": 0.6, "B": 0.3})
    with pytest.raises(ValueError, match="non-negative"):
        Portfolio("bad", 100_000, {"A": 1.1, "B": -0.1})
    portfolio = Portfolio("Two asset test", 100_000, {"A": 0.6, "B": 0.4})
    rows = hypothetical_scenarios(portfolio, shocks=(-0.05, -0.10, -0.20))
    assert rows["loss_amount"].tolist() == pytest.approx([5_000, 10_000, 20_000])
    assert rows.iloc[0]["stressed_value"] == pytest.approx(95_000)
    assert json.loads(rows.iloc[0]["asset_contributions"]) == {"A": 3_000.0, "B": 2_000.0}
    single = hypothetical_scenarios(portfolio, shocks=(-0.1,), scope="single_asset", asset="A")
    assert single.iloc[0]["loss_amount"] == pytest.approx(6_000)


def test_historical_scenarios_compound_log_returns_and_keep_gap_flags() -> None:
    dates = pd.bdate_range("2024-01-01", periods=4, name="date")
    returns = pd.DataFrame({"A": [-0.1, -0.1, 0.02, 0.01], "B": [0.0, -0.05, 0.01, 0.0]}, index=dates)
    flags = pd.Series([False, True, False, False], index=dates, dtype=bool)
    portfolio = Portfolio("Mixed", 10_000, {"A": 0.5, "B": 0.5})
    row = historical_scenarios(portfolio, returns, horizon=2, worst_n=1, gap_flags=flags).iloc[0]
    expected_asset_a = np.exp(-0.2) - 1
    expected_asset_b = np.exp(-0.05) - 1
    expected_loss = -10_000 * (0.5 * expected_asset_a + 0.5 * expected_asset_b)
    assert row["loss_amount"] == pytest.approx(expected_loss)
    assert row["horizon_sessions"] == 2
    assert row["gap_flagged_observations"] == 1
    assert len(json.loads(row["asset_contributions"])) == 2


def test_historical_stress_rejects_missing_assets_data_and_short_history() -> None:
    portfolio = Portfolio("One", 1000, {"A": 1.0})
    dates = pd.bdate_range("2024-01-01", periods=2)
    with pytest.raises(ValueError, match="at least 3"):
        historical_scenarios(portfolio, pd.DataFrame({"A": [0.01, 0.02]}, index=dates), horizon=3)
    with pytest.raises(ValueError, match="missing portfolio asset"):
        historical_scenarios(portfolio, pd.DataFrame({"B": [0.01, 0.02]}, index=dates))
    with pytest.raises(ValueError, match="missing or non-finite"):
        historical_scenarios(portfolio, pd.DataFrame({"A": [0.01, np.nan]}, index=dates))
    with pytest.raises(ValueError, match="between -100% and 0%"):
        hypothetical_scenarios(portfolio, shocks=(0.05,))


def test_stage13_persistence_schema_and_dashboard_compatibility(tmp_path) -> None:
    closes = (100 + np.arange(320) * 0.1 + np.sin(np.arange(320) / 4)).tolist()
    market = _market(closes)
    risk_dir, stress_dir = tmp_path / "risk", tmp_path / "stress"
    frame, risk_meta, scenarios, stress_meta = run_stage13_experiment(
        market,
        risk_dir,
        stress_dir,
        risk_config=RiskMonitoringConfig(realised_volatility_window=20, historical_var_window=50, breach_summary_window=20),
        historical_worst_n=2,
    )
    assert len(pd.read_csv(risk_dir / "daily_risk_monitoring.csv")) == len(frame)
    assert json.loads((risk_dir / "metadata.json").read_text(encoding="utf-8"))["data"]["cache_entry_used"] is False
    assert json.loads((stress_dir / "metadata.json").read_text(encoding="utf-8"))["portfolio"]["weights"] == {"TEST": 1.0}
    assert len(pd.read_csv(stress_dir / "scenario_results.csv")) == len(scenarios) == 7
    assert risk_meta["risk_configuration"]["historical_var_es_window"] == 50
    assert "single TEST asset holding" in stress_meta["portfolio"]["scope_note"]
    assert risk_meta["artifact_schema_version"] == stress_meta["artifact_schema_version"] == 1
    assert app.load_stage13_dashboard_artifacts(
        risk_dir, stress_dir, now=datetime(2024, 6, 4, tzinfo=timezone.utc)
    ).status == "ready"
    stale = app.load_stage13_dashboard_artifacts(
        risk_dir, stress_dir, now=datetime(2024, 7, 14, tzinfo=timezone.utc)
    )
    assert stale.available and stale.status == "stale"

    script = (
        "import app\n"
        "from pathlib import Path\n"
        f"app.RISK_MONITORING_RESULTS_DIR = Path({str(risk_dir)!r})\n"
        f"app.STRESS_TESTING_RESULTS_DIR = Path({str(stress_dir)!r})\n"
        "app._render_stage13_risk_monitoring()\n"
    )
    rendered = AppTest.from_string(script, default_timeout=30).run()
    assert not rendered.exception
    assert len(rendered.dataframe) == 2
    visible = " ".join(item.value for group in (rendered.markdown, rendered.caption) for item in group)
    assert "Rolling risk monitoring" in visible
    assert "Stress scenarios" in visible


def test_runner_falls_back_to_identity_matched_validated_cache(monkeypatch, tmp_path) -> None:
    market = _market([100 + i * 0.1 for i in range(300)])

    class BrokenLoader:
        def load(self, request):
            raise DataFetchError("Yahoo lookup failed")

    class LocalCache:
        def __init__(self, directory):
            self.directory = directory

        def read(self, request):
            return market

    monkeypatch.setattr(stage13, "create_market_data_loader", lambda settings: BrokenLoader())
    monkeypatch.setattr(stage13, "MarketDataCache", LocalCache)
    settings = SimpleNamespace(data=SimpleNamespace(use_cache=True, data_dir=tmp_path, min_observations=250))
    result, used_cache = stage13.load_market_data_with_cache_fallback(settings, market.request)
    assert used_cache
    assert result.from_cache
    assert result.fallback_reason == "DataFetchError: Yahoo lookup failed"
    assert result.prices.equals(market.prices)


def test_runner_preserves_provider_error_when_no_cache_exists(monkeypatch, tmp_path) -> None:
    class BrokenLoader:
        def load(self, request):
            raise DataFetchError("Yahoo DNS lookup failed")

    class EmptyCache:
        def __init__(self, directory):
            pass

        def read(self, request):
            return None

    monkeypatch.setattr(stage13, "create_market_data_loader", lambda settings: BrokenLoader())
    monkeypatch.setattr(stage13, "MarketDataCache", EmptyCache)
    settings = SimpleNamespace(data=SimpleNamespace(use_cache=True, data_dir=tmp_path, min_observations=250))
    with pytest.raises(DataFetchError, match="DNS lookup failed"):
        stage13.load_market_data_with_cache_fallback(settings, _market([100 + i * 0.1 for i in range(300)]).request)


def test_stage13_dashboard_loader_handles_missing_and_invalid_artifacts(tmp_path) -> None:
    assert app.load_stage13_dashboard_artifacts(tmp_path / "risk", tmp_path / "stress").status == "missing"
    risk_dir, stress_dir = tmp_path / "risk", tmp_path / "stress"
    risk_dir.mkdir()
    stress_dir.mkdir()
    (risk_dir / "daily_risk_monitoring.csv").write_text("date,log_return_pct\n2024-01-01,0\n", encoding="utf-8")
    (risk_dir / "risk_summary.json").write_text("{}", encoding="utf-8")
    (risk_dir / "metadata.json").write_text("{}", encoding="utf-8")
    (stress_dir / "scenario_results.csv").write_text("scenario_name\nexample\n", encoding="utf-8")
    (stress_dir / "metadata.json").write_text("{}", encoding="utf-8")
    assert app.load_stage13_dashboard_artifacts(risk_dir, stress_dir).status == "incomplete"

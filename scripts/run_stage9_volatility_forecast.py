"""Run and save the Stage 9 NIFTY 50 volatility forecast experiment.

Run from the repository root with::

    python -m scripts.run_stage9_volatility_forecast
"""

from __future__ import annotations

import json
import time
from datetime import date, datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd

from src.config import get_settings
from src.data.factory import create_market_data_loader
from src.data.cache import CACHE_SUBDIRECTORY
from src.data.lineage import build_data_lineage
from src.data.market_data import MarketDataRequest
from src.models.figures import plot_variance_forecasts
from src.models.walk_forward import run_expanding_volatility_forecast
from src.preprocessing.returns import build_return_series
from src.utils.artifacts import write_artifact_bundles


def main() -> None:
    started = time.perf_counter()
    settings = get_settings()
    request = MarketDataRequest(
        ticker="^NSEI",
        start=date(2005, 1, 1),
        end=None,
        adjust_prices=True,
    )
    market_data = create_market_data_loader(settings).load(request)
    returns = build_return_series(market_data, settings.preprocessing)
    walk_forward = run_expanding_volatility_forecast(
        returns,
        evaluation_size=252,
        rolling_window=20,
    )

    output_dir = Path("results/volatility_forecasting/nifty50")

    summary = walk_forward.metrics.copy()
    summary["ticker"] = returns.ticker
    summary["data_fetched_at_utc"] = returns.fetched_at.isoformat()
    summary["evaluation_start"] = walk_forward.evaluation_start
    summary["evaluation_end"] = walk_forward.evaluation_end
    summary["training_start_first_origin"] = returns.first_return_date.isoformat()
    summary["training_end_first_origin"] = pd.Timestamp(
        walk_forward.forecasts["garch_fit_end"].iloc[0]
    ).date().isoformat()
    summary["training_observations_first_origin"] = int(
        walk_forward.forecasts["garch_n_train"].iloc[0]
    )
    summary["evaluation_observations"] = walk_forward.evaluation_observations
    summary["garch_config"] = json.dumps(walk_forward.garch_config, sort_keys=True)
    summary["egarch_config"] = json.dumps(walk_forward.egarch_config, sort_keys=True)

    gap_flags = int(walk_forward.forecasts["gap_flagged"].sum())
    metadata = {
        "artifact_schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "Stage 9 expanding-window one-step-ahead volatility forecasting",
        "data_source": returns.source,
        "data_lineage": build_data_lineage(
            market_data,
            cache_directory=settings.data.data_dir / CACHE_SUBDIRECTORY,
            provisional_handling={
                "date": returns.provisional_bar.isoformat() if returns.provisional_bar else None,
                "excluded": returns.provisional_bar_excluded,
                "include_provisional_bar_setting": settings.preprocessing.include_provisional_bar,
            },
            ingestion_configuration={
                "cache_enabled": settings.data.use_cache,
                "cache_max_age_hours": settings.data.cache_max_age_hours,
                "provider_timeout_seconds": settings.data.provider_timeout_seconds,
                "provider_retries": settings.data.provider_retries,
                "minimum_observations": settings.data.min_observations,
            },
        ),
        "ticker": returns.ticker,
        "requested_start": request.start.isoformat(),
        "requested_end": "latest available",
        "data_fetched_at_utc": returns.fetched_at.isoformat(),
        "actual_price_start": market_data.first_date.isoformat(),
        "actual_price_end": market_data.last_date.isoformat(),
        "price_observations": market_data.observations,
        "ingestion_rows_dropped": market_data.rows_dropped,
        "preprocessing": {
            **returns.metadata(),
            "include_provisional_bar": settings.preprocessing.include_provisional_bar,
            "max_gap_weekdays": settings.preprocessing.max_gap_weekdays,
            "provisional_bar_excluded": returns.provisional_bar_excluded,
            "gap_spanning_returns_retained": True,
            "gap_flagged_evaluation_returns": gap_flags,
        },
        "evaluation": {
            "start": walk_forward.evaluation_start,
            "end": walk_forward.evaluation_end,
            "observations": walk_forward.evaluation_observations,
            "target": "one-session realised variance proxy: squared daily log return",
            "forecast_horizon_sessions": 1,
            "forecast_origin": "prior close; all fitting uses returns strictly before target date",
            "first_origin_training_start": returns.first_return_date.isoformat(),
            "first_origin_training_end": pd.Timestamp(
                walk_forward.forecasts["garch_fit_end"].iloc[0]
            ).date().isoformat(),
            "first_origin_training_observations": int(
                walk_forward.forecasts["garch_n_train"].iloc[0]
            ),
            "design": "expanding window; fit through prior return, forecast next return once",
            "realised_variance_proxy": "squared daily log return",
            "return_scale": "percent",
            "variance_scale": "percent^2",
            "qlike_formula": "mean(log(h_t) + RV_t / h_t)",
            "mae_formula": "mean(abs(h_t - RV_t))",
            "rmse_formula": "sqrt(mean((h_t - RV_t)^2))",
            "rolling_benchmark": {
                "window": 20,
                "alignment": "sample variance (ddof=1) of the 20 returns before the forecast date",
            },
            "realised_squared_return_zero_count": int(
                (walk_forward.forecasts["realised_squared_return_pct2"] == 0).sum()
            ),
            "limitations": [
                "A squared daily return is a noisy single-observation proxy for latent conditional variance.",
                "This is one historical evaluation window and does not establish universal model superiority.",
            ],
        },
        "models": {
            "garch": {
                "configuration": walk_forward.garch_config,
                "convergence_failures": int((~walk_forward.forecasts["garch_converged"]).sum()),
                "notes": sorted(set(filter(None, walk_forward.forecasts["garch_notes"]))),
            },
            "egarch": {
                "configuration": walk_forward.egarch_config,
                "convergence_failures": int((~walk_forward.forecasts["egarch_converged"]).sum()),
                "notes": sorted(set(filter(None, walk_forward.forecasts["egarch_notes"]))),
            },
        },
        "duration_seconds": time.perf_counter() - started,
    }
    fig = plot_variance_forecasts(
        walk_forward.forecasts,
        title=(
            f"NIFTY 50 one-step variance forecasts "
            f"({walk_forward.evaluation_start} to {walk_forward.evaluation_end})"
        ),
    )
    write_artifact_bundles({output_dir: {
        "daily_forecasts.csv": lambda path: walk_forward.forecasts.to_csv(path, index_label="date"),
        "summary_metrics.csv": lambda path: summary.to_csv(path, index=False),
        "metadata.json": lambda path: path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        ),
        "variance_forecast_comparison.png": lambda path: fig.savefig(path, dpi=160, bbox_inches="tight"),
    }})
    fig.clear()
    print(json.dumps({
        "output_dir": str(output_dir),
        "data_fetched_at_utc": returns.fetched_at.isoformat(),
        "evaluation_start": walk_forward.evaluation_start,
        "evaluation_end": walk_forward.evaluation_end,
        "evaluation_observations": walk_forward.evaluation_observations,
        "gap_flagged_evaluation_returns": gap_flags,
        "metrics": walk_forward.metrics.to_dict(orient="records"),
        "garch_convergence_failures": metadata["models"]["garch"]["convergence_failures"],
        "egarch_convergence_failures": metadata["models"]["egarch"]["convergence_failures"],
        "duration_seconds": metadata["duration_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()

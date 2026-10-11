"""Run the Stage 10 NIFTY 50 one-day VaR and backtesting experiment.

Run from the repository root with::

    python -m scripts.run_stage10_var_backtesting

The evaluation dates are read from the saved Stage 9 forecast artifact. Forecasts
are then recomputed with the canonical loader and expanding-window model pipeline;
the CSV supplies dates only, never forecast values or realised returns.
"""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd

from src.backtesting.var_evaluation import evaluate_var_methods
from src.config import get_settings
from src.data.factory import create_market_data_loader
from src.data.market_data import MarketDataRequest
from src.models.walk_forward import run_expanding_volatility_forecast
from src.preprocessing.returns import build_return_series
from src.risk.figures import plot_var_breach_counts, plot_var_forecasts
from src.risk.var import build_var_forecasts


CONFIDENCE_LEVELS = (0.95, 0.99)
HISTORICAL_WINDOW = 250
SIGNIFICANCE_LEVEL = 0.05
STAGE9_DATES = Path("results/volatility_forecasting/nifty50/daily_forecasts.csv")
OUTPUT_DIR = Path("results/var_backtesting/nifty50")


def main() -> None:
    started = time.perf_counter()
    if not STAGE9_DATES.exists():
        raise FileNotFoundError(
            f"Stage 9 date artifact not found at {STAGE9_DATES}; run the Stage 9 experiment first."
        )
    stage9_dates = pd.read_csv(STAGE9_DATES, usecols=["date"], parse_dates=["date"])["date"]
    evaluation_dates = pd.DatetimeIndex(stage9_dates, name="date")
    if (
        len(evaluation_dates) != 252
        or evaluation_dates.has_duplicates
        or not evaluation_dates.is_monotonic_increasing
    ):
        raise ValueError(
            "Stage 9 artifact must contain exactly 252 unique, chronologically ordered dates; "
            f"found {len(evaluation_dates)}."
        )

    settings = get_settings()
    request = MarketDataRequest(
        ticker="^NSEI", start=date(2005, 1, 1), end=None, adjust_prices=True
    )
    market_data = create_market_data_loader(settings).load(request)
    returns = build_return_series(market_data, settings.preprocessing)
    walk_forward = run_expanding_volatility_forecast(
        returns,
        evaluation_dates=evaluation_dates,
        rolling_window=20,
    )
    var_forecasts = build_var_forecasts(
        returns,
        walk_forward.forecasts,
        historical_window=HISTORICAL_WINDOW,
        confidence_levels=CONFIDENCE_LEVELS,
        quantile_method="linear",
    )
    summary = evaluate_var_methods(
        var_forecasts,
        confidence_levels=CONFIDENCE_LEVELS,
        significance_level=SIGNIFICANCE_LEVEL,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    var_forecasts.to_csv(OUTPUT_DIR / "daily_var_forecasts.csv", index_label="date")
    summary.to_csv(OUTPUT_DIR / "backtest_summary.csv", index=False)

    first = walk_forward.forecasts.iloc[0]
    warnings = {
        "garch": sorted(set(filter(None, walk_forward.forecasts["garch_notes"]))),
        "egarch": sorted(set(filter(None, walk_forward.forecasts["egarch_notes"]))),
    }
    metadata = {
        "experiment": "Stage 10 one-day-ahead VaR forecasts and statistical backtesting",
        "data": {
            "source": returns.source,
            "ticker": returns.ticker,
            "requested_start": request.start.isoformat(),
            "requested_end": "latest available",
            "data_fetched_at_utc": returns.fetched_at.isoformat(),
            "actual_price_start": market_data.first_date.isoformat(),
            "actual_price_end": market_data.last_date.isoformat(),
            "price_observations": market_data.observations,
            "ingestion_rows_dropped": market_data.rows_dropped,
            "return_observations": returns.return_observations,
            "return_start": returns.first_return_date.isoformat(),
            "return_end": returns.last_return_date.isoformat(),
            "preprocessing": {
                **returns.metadata(),
                "include_provisional_bar": settings.preprocessing.include_provisional_bar,
                "max_gap_weekdays": settings.preprocessing.max_gap_weekdays,
                "gap_spanning_returns_retained": True,
                "evaluation_gap_flagged_observations": int(var_forecasts["gap_flagged"].sum()),
                "evaluation_observations_with_provisional_bar": int(
                    returns.last_return_is_provisional and returns.last_return_date in var_forecasts.index.date
                ),
            },
        },
        "evaluation": {
            "source_of_dates": str(STAGE9_DATES),
            "start": var_forecasts.index[0].date().isoformat(),
            "end": var_forecasts.index[-1].date().isoformat(),
            "observations": len(var_forecasts),
            "design": "Expanding-window walk-forward; each model refit uses returns strictly before the forecast date.",
            "first_origin_training_start": returns.first_return_date.isoformat(),
            "first_origin_training_end": pd.Timestamp(first["garch_fit_end"]).date().isoformat(),
            "first_origin_training_observations": int(first["garch_n_train"]),
            "last_origin_training_end": pd.Timestamp(walk_forward.forecasts["garch_fit_end"].iloc[-1]).date().isoformat(),
            "model_fit_and_selection_data": "Prior returns only. No parameter selection or tuning on evaluation returns.",
            "historical_simulation_window": HISTORICAL_WINDOW,
            "historical_simulation_alignment": "Previous 250 valid returns; excludes the target return.",
            "historical_simulation_quantile": "numpy linear quantile at probability 1-confidence",
            "confidence_levels": list(CONFIDENCE_LEVELS),
            "significance_level": SIGNIFICANCE_LEVEL,
            "return_scale": "percent log-return points",
            "variance_scale": "percent^2",
            "loss_convention": "Realised loss = -realised return; VaR thresholds are positive losses.",
            "parametric_var_formula": "VaR_c = -(mu_hat + Phi^-1(1-c) * sqrt(h_hat)); Normal innovations.",
            "breach_rule": "realised loss > VaR threshold",
            "backtests": {
                "kupiec_pof": "LR_uc = -2 log(L(p0)/L(phat)); chi-square(1); H0 unconditional breach probability = 1-confidence.",
                "christoffersen_independence": "LR_ind = -2 log(L(iid)/L(first-order Markov)); chi-square(1); H0 breach indicators are first-order independent.",
                "christoffersen_conditional_coverage": "LR_cc = LR_uc + LR_ind; chi-square(2); joint H0 of correct unconditional coverage and independence.",
            },
            "limitations": [
                "Squared daily returns are a noisy one-observation proxy for latent conditional variance.",
                "A single 252-session evaluation window does not establish universal superiority of any method.",
                "At 99% confidence only 2.52 breaches are expected in 252 observations, so coverage tests have low power and discrete outcomes.",
            ],
        },
        "methods": {
            "garch_normal": {"configuration": walk_forward.garch_config, "distribution": "Normal", "mean": "forecast mean from fitted model"},
            "egarch_normal": {"configuration": walk_forward.egarch_config, "distribution": "Normal", "mean": "forecast mean from fitted model"},
            "historical_simulation": {"window": HISTORICAL_WINDOW, "quantile_method": "linear", "refit_frequency": "each observation; prior window only"},
        },
        "model_status": {
            "garch_convergence_failures": int((~walk_forward.forecasts["garch_converged"]).sum()),
            "egarch_convergence_failures": int((~walk_forward.forecasts["egarch_converged"]).sum()),
            "garch_warnings_or_notes": warnings["garch"],
            "egarch_warnings_or_notes": warnings["egarch"],
            "excluded_evaluation_observations": 0,
        },
        "duration_seconds": time.perf_counter() - started,
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    fig = plot_var_forecasts(
        var_forecasts,
        title=f"NIFTY 50 one-day VaR ({metadata['evaluation']['start']} to {metadata['evaluation']['end']})",
    )
    fig.savefig(OUTPUT_DIR / "var_forecast_comparison.png", dpi=160, bbox_inches="tight")
    fig.clear()
    fig = plot_var_breach_counts(summary)
    fig.savefig(OUTPUT_DIR / "var_breach_counts.png", dpi=160, bbox_inches="tight")
    fig.clear()

    print(json.dumps({
        "output_dir": str(OUTPUT_DIR),
        "data_fetched_at_utc": returns.fetched_at.isoformat(),
        "evaluation_start": metadata["evaluation"]["start"],
        "evaluation_end": metadata["evaluation"]["end"],
        "evaluation_observations": len(var_forecasts),
        "gap_flagged_evaluation_observations": int(var_forecasts["gap_flagged"].sum()),
        "convergence_failures": metadata["model_status"],
        "summary": summary.to_dict(orient="records"),
        "duration_seconds": metadata["duration_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()

"""Evaluate aligned VaR breach series with coverage and independence tests."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from src.backtesting.var_tests import (
    christoffersen_conditional_coverage,
    christoffersen_independence,
    kupiec_pof,
)


def evaluate_var_methods(
    forecasts: pd.DataFrame,
    *,
    confidence_levels: Iterable[float] = (0.95, 0.99),
    significance_level: float = 0.05,
) -> pd.DataFrame:
    """Return one summary row for each method and confidence level."""
    if forecasts.empty or not forecasts.index.is_unique or not forecasts.index.is_monotonic_increasing:
        raise ValueError("forecasts must be non-empty and chronologically ordered.")
    methods = {
        "GARCH Normal": "garch",
        "EGARCH Normal": "egarch",
        "250-return historical simulation": "historical",
    }
    rows: list[dict[str, object]] = []
    for confidence in confidence_levels:
        confidence = float(confidence)
        label = int(round(confidence * 100))
        expected = 1.0 - confidence
        for method, prefix in methods.items():
            column = f"{prefix}_breach{label}"
            if column not in forecasts:
                raise ValueError(f"Forecast frame is missing breach column {column!r}.")
            breaches = forecasts[column].to_numpy(dtype=bool)
            unconditional = kupiec_pof(
                breaches, expected, significance_level=significance_level
            )
            independence = christoffersen_independence(
                breaches, expected, significance_level=significance_level
            )
            conditional = christoffersen_conditional_coverage(
                breaches, expected, significance_level=significance_level
            )
            rows.append({
                "method": method,
                "confidence": confidence,
                "expected_breach_rate": expected,
                "breach_count": unconditional.breach_count,
                "sample_size": unconditional.sample_size,
                "observed_breach_rate": unconditional.observed_breach_rate,
                "kupiec_statistic": unconditional.statistic,
                "kupiec_df": unconditional.degrees_of_freedom,
                "kupiec_p_value": unconditional.p_value,
                "kupiec_null": unconditional.null_hypothesis,
                "kupiec_interpretation": unconditional.interpretation,
                "christoffersen_independence_statistic": independence.statistic,
                "christoffersen_independence_df": independence.degrees_of_freedom,
                "christoffersen_independence_p_value": independence.p_value,
                "christoffersen_independence_null": independence.null_hypothesis,
                "christoffersen_independence_interpretation": independence.interpretation,
                "transition_counts": str(independence.transition_counts),
                "independence_notes": " | ".join(independence.notes),
                "conditional_coverage_statistic": conditional.statistic,
                "conditional_coverage_df": conditional.degrees_of_freedom,
                "conditional_coverage_p_value": conditional.p_value,
                "conditional_coverage_null": conditional.null_hypothesis,
                "conditional_coverage_interpretation": conditional.interpretation,
                "conditional_coverage_notes": " | ".join(conditional.notes),
            })
    return pd.DataFrame(rows)

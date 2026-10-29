from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.unified import classify_volatility_regime, fz0_var_es_score, summarize_by_regime


def test_fz0_score_matches_documented_loss_formula() -> None:
    score = fz0_var_es_score([1.0, 4.0], [2.0, 2.0], [3.0, 3.0], 0.1)
    expected = np.array([2 / 3 + np.log(3) - 1, 2 / (0.1 * 3) + 2 / 3 + np.log(3) - 1])
    np.testing.assert_allclose(score, expected)


@pytest.mark.parametrize("values", [([], [], []), ([1], [1, 2], [2]), ([1], [1], [0])])
def test_fz0_rejects_empty_misaligned_or_nonpositive_es(values) -> None:
    with pytest.raises(ValueError):
        fz0_var_es_score(*values, 0.05)


def test_regime_threshold_uses_only_prior_observations() -> None:
    index = pd.date_range("2024-01-01", periods=5)
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0], index=index)
    classified = classify_volatility_regime(values, lookback=3, minimum_history=2)
    assert classified.iloc[0] is pd.NA or pd.isna(classified.iloc[0])
    assert classified.iloc[2] == "high"
    assert classified.iloc[3] == "high"
    assert classified.iloc[4] == "high"


def test_regime_summary_counts_only_classified_aligned_rows() -> None:
    index = pd.date_range("2024-01-01", periods=3)
    metrics = pd.DataFrame({"loss": [1.0, 3.0, 5.0]}, index=index)
    regime = pd.Series(["low", "high", pd.NA], index=index, dtype="string")
    result = summarize_by_regime(metrics, regime, metrics={"mean_loss": "loss"})
    assert result.to_dict(orient="records") == [
        {"regime": "high", "observations": 1, "mean_loss": 3.0},
        {"regime": "low", "observations": 1, "mean_loss": 1.0},
    ]

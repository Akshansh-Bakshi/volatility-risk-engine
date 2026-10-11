"""Tests for Kupiec and Christoffersen VaR backtests."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2

from src.backtesting.var_evaluation import evaluate_var_methods
from src.backtesting.var_tests import (
    christoffersen_conditional_coverage,
    christoffersen_independence,
    kupiec_pof,
)


class TestKupiecPOF:
    def test_known_250_observation_example(self) -> None:
        breaches = np.zeros(250, dtype=bool)
        breaches[:15] = True
        expected_rate = 0.05
        count = int(breaches.sum())
        n = len(breaches)
        observed_rate = count / n
        ll_null = (n - count) * math.log(expected_rate * 0 + 1 - expected_rate) + count * math.log(expected_rate)
        ll_alt = (n - count) * math.log(1 - observed_rate) + count * math.log(observed_rate)
        expected_statistic = 2 * (ll_alt - ll_null)

        result = kupiec_pof(breaches, expected_rate)
        assert result.statistic == pytest.approx(expected_statistic)
        assert result.p_value == pytest.approx(chi2.sf(expected_statistic, df=1))
        assert result.degrees_of_freedom == 1
        assert result.breach_count == 15
        assert result.sample_size == 250
        assert result.observed_breach_rate == pytest.approx(0.06)
        assert result.expected_breach_rate == 0.05
        assert "unconditional breach probability" in result.null_hypothesis

    @pytest.mark.parametrize("breaches", [[False] * 20, [True] * 20])
    def test_zero_or_all_breaches_are_finite(self, breaches: list[bool]) -> None:
        result = kupiec_pof(breaches, 0.01)
        assert math.isfinite(result.statistic)
        assert math.isfinite(result.p_value)
        assert 0.0 <= result.p_value <= 1.0


class TestChristoffersenIndependence:
    def test_known_transition_counts_and_likelihood_ratio(self) -> None:
        breaches = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0], dtype=bool)
        expected_counts = {"n00": 5, "n01": 1, "n10": 1, "n11": 3}
        pi = 4 / 10
        ll_null = 6 * math.log(1 - pi) + 4 * math.log(pi)
        p0, p1 = 1 / 6, 3 / 4
        ll_alt = 5 * math.log(1 - p0) + math.log(p0) + math.log(1 - p1) + 3 * math.log(p1)
        expected_statistic = 2 * (ll_alt - ll_null)

        result = christoffersen_independence(breaches, 0.05)
        assert result.transition_counts == expected_counts
        assert result.statistic == pytest.approx(expected_statistic)
        assert result.p_value == pytest.approx(chi2.sf(expected_statistic, df=1))
        assert result.degrees_of_freedom == 1
        assert "first-order independent" in result.null_hypothesis

    @pytest.mark.parametrize("breaches", [[False] * 20, [True] * 20, [False]])
    def test_degenerate_sequences_use_documented_finite_policy(self, breaches: list[bool]) -> None:
        result = christoffersen_independence(breaches, 0.05)
        assert result.statistic == 0.0
        assert result.p_value == 1.0
        assert math.isfinite(result.statistic)
        assert math.isfinite(result.p_value)
        assert result.notes


class TestConditionalCoverage:
    def test_conditional_coverage_combines_lr_statistics(self) -> None:
        breaches = [False, False, True, False, False, True, True, False, False, False]
        uc = kupiec_pof(breaches, 0.05)
        ind = christoffersen_independence(breaches, 0.05)
        result = christoffersen_conditional_coverage(breaches, 0.05)
        expected_statistic = uc.statistic + ind.statistic
        assert result.statistic == pytest.approx(expected_statistic)
        assert result.p_value == pytest.approx(chi2.sf(expected_statistic, df=2))
        assert result.degrees_of_freedom == 2
        assert result.interpretation
        assert "Reject the null" in result.interpretation

    def test_fail_to_reject_interpretation_is_not_overstated(self) -> None:
        breaches = np.zeros(250, dtype=bool)
        breaches[5:245:20] = True
        result = christoffersen_conditional_coverage(breaches, 0.05)
        assert "Fail to reject" in result.interpretation
        assert "not proof" in result.interpretation

    def test_result_reports_all_breach_metadata(self) -> None:
        result = kupiec_pof([False, False, True, False], 0.05)
        data = result.to_dict()
        for key in (
            "statistic", "degrees_of_freedom", "p_value", "breach_count",
            "sample_size", "observed_breach_rate", "expected_breach_rate",
            "null_hypothesis", "interpretation",
        ):
            assert key in data


def test_method_evaluation_uses_shared_sample_and_confidence_levels() -> None:
    index = pd.date_range("2025-01-01", periods=5, name="date")
    forecasts = pd.DataFrame(index=index)
    for prefix in ("garch", "egarch", "historical"):
        forecasts[f"{prefix}_breach95"] = [False, False, True, False, False]
        forecasts[f"{prefix}_breach99"] = [False] * 5
    summary = evaluate_var_methods(forecasts)
    assert len(summary) == 6
    assert set(summary["sample_size"]) == {5}
    assert set(summary["confidence"]) == {0.95, 0.99}
    assert set(summary["method"]) == {
        "GARCH Normal", "EGARCH Normal", "250-return historical simulation"
    }

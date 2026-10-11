"""Unconditional, independence and conditional-coverage VaR backtests."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import chi2


@dataclass(frozen=True)
class VaRBacktestResult:
    """Serializable result for one VaR coverage hypothesis test."""

    test_name: str
    statistic: float
    degrees_of_freedom: int
    p_value: float
    breach_count: int
    sample_size: int
    observed_breach_rate: float
    expected_breach_rate: float
    null_hypothesis: str
    interpretation: str
    transition_counts: dict[str, int] = field(default_factory=dict)
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_name": self.test_name,
            "statistic": self.statistic,
            "degrees_of_freedom": self.degrees_of_freedom,
            "p_value": self.p_value,
            "breach_count": self.breach_count,
            "sample_size": self.sample_size,
            "observed_breach_rate": self.observed_breach_rate,
            "expected_breach_rate": self.expected_breach_rate,
            "null_hypothesis": self.null_hypothesis,
            "interpretation": self.interpretation,
            "transition_counts": self.transition_counts,
            "notes": list(self.notes),
        }


def kupiec_pof(
    breaches: Iterable[bool | int],
    expected_rate: float,
    *,
    significance_level: float = 0.05,
) -> VaRBacktestResult:
    """Kupiec proportion-of-failures LR test (chi-square, one degree of freedom)."""
    values = _validated_breaches(breaches)
    _validate_probabilities(expected_rate, significance_level)
    sample_size = int(values.size)
    count = int(values.sum())
    observed_rate = count / sample_size

    log_null = _bernoulli_log_likelihood(count, sample_size - count, expected_rate)
    log_alternative = _bernoulli_log_likelihood(count, sample_size - count, observed_rate)
    statistic = max(0.0, 2.0 * (log_alternative - log_null))
    p_value = float(chi2.sf(statistic, df=1))
    return VaRBacktestResult(
        test_name="Kupiec POF",
        statistic=statistic,
        degrees_of_freedom=1,
        p_value=p_value,
        breach_count=count,
        sample_size=sample_size,
        observed_breach_rate=observed_rate,
        expected_breach_rate=expected_rate,
        null_hypothesis="The unconditional breach probability equals the expected tail probability.",
        interpretation=_interpret(p_value, significance_level),
    )


def christoffersen_independence(
    breaches: Iterable[bool | int],
    expected_rate: float,
    *,
    significance_level: float = 0.05,
) -> VaRBacktestResult:
    """Christoffersen first-order Markov independence LR test (chi-square, df=1)."""
    values = _validated_breaches(breaches)
    _validate_probabilities(expected_rate, significance_level)
    count = int(values.sum())
    sample_size = int(values.size)
    observed_rate = count / sample_size

    if sample_size < 2:
        transitions = {"n00": 0, "n01": 0, "n10": 0, "n11": 0}
        return _degenerate_independence_result(
            count, sample_size, observed_rate, expected_rate, transitions,
            "Fewer than two observations provide no transitions; statistic 0 and p-value 1 are reported by policy.",
        )

    previous, current = values[:-1], values[1:]
    transitions = {
        "n00": int(np.sum((previous == 0) & (current == 0))),
        "n01": int(np.sum((previous == 0) & (current == 1))),
        "n10": int(np.sum((previous == 1) & (current == 0))),
        "n11": int(np.sum((previous == 1) & (current == 1))),
    }
    n00, n01, n10, n11 = (transitions[key] for key in ("n00", "n01", "n10", "n11"))
    transition_total = sample_size - 1
    pooled_probability = (n01 + n11) / transition_total
    ll_null = _bernoulli_log_likelihood(n01 + n11, n00 + n10, pooled_probability)

    state0_total = n00 + n01
    state1_total = n10 + n11
    ll_alternative = 0.0
    if state0_total:
        ll_alternative += _bernoulli_log_likelihood(n01, n00, n01 / state0_total)
    if state1_total:
        ll_alternative += _bernoulli_log_likelihood(n11, n10, n11 / state1_total)

    statistic = max(0.0, 2.0 * (ll_alternative - ll_null))
    p_value = float(chi2.sf(statistic, df=1))
    notes: tuple[str, ...] = ()
    if count in (0, sample_size):
        notes = ("The breach sequence is constant; independence is not identifiable. The boundary MLE gives statistic 0 and p-value 1.",)
    return VaRBacktestResult(
        test_name="Christoffersen independence",
        statistic=statistic,
        degrees_of_freedom=1,
        p_value=p_value,
        breach_count=count,
        sample_size=sample_size,
        observed_breach_rate=observed_rate,
        expected_breach_rate=expected_rate,
        null_hypothesis="Breach indicators are first-order independent (transition probability is unchanged after a breach).",
        interpretation=_interpret(p_value, significance_level),
        transition_counts=transitions,
        notes=notes,
    )


def christoffersen_conditional_coverage(
    breaches: Iterable[bool | int],
    expected_rate: float,
    *,
    significance_level: float = 0.05,
) -> VaRBacktestResult:
    """Joint unconditional-coverage and independence LR test (chi-square, df=2)."""
    values = _validated_breaches(breaches)
    unconditional = kupiec_pof(
        values, expected_rate, significance_level=significance_level
    )
    independence = christoffersen_independence(
        values, expected_rate, significance_level=significance_level
    )
    statistic = unconditional.statistic + independence.statistic
    p_value = float(chi2.sf(statistic, df=2))
    return VaRBacktestResult(
        test_name="Christoffersen conditional coverage",
        statistic=statistic,
        degrees_of_freedom=2,
        p_value=p_value,
        breach_count=unconditional.breach_count,
        sample_size=unconditional.sample_size,
        observed_breach_rate=unconditional.observed_breach_rate,
        expected_breach_rate=expected_rate,
        null_hypothesis="The unconditional breach probability is correct and breaches are first-order independent.",
        interpretation=_interpret(p_value, significance_level),
        transition_counts=independence.transition_counts,
        notes=unconditional.notes + independence.notes,
    )


def _degenerate_independence_result(
    count: int,
    sample_size: int,
    observed_rate: float,
    expected_rate: float,
    transitions: dict[str, int],
    note: str,
) -> VaRBacktestResult:
    return VaRBacktestResult(
        test_name="Christoffersen independence",
        statistic=0.0,
        degrees_of_freedom=1,
        p_value=1.0,
        breach_count=count,
        sample_size=sample_size,
        observed_breach_rate=observed_rate,
        expected_breach_rate=expected_rate,
        null_hypothesis="Breach indicators are first-order independent.",
        interpretation="Not testable from the available transitions; statistic 0 and p-value 1 are policy placeholders.",
        transition_counts=transitions,
        notes=(note,),
    )


def _validated_breaches(breaches: Iterable[bool | int]) -> np.ndarray:
    raw = np.asarray(list(breaches))
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("breaches must be a non-empty one-dimensional sequence.")
    if not np.isin(raw, [False, True, 0, 1]).all():
        raise ValueError("breaches must contain only boolean or binary 0/1 values.")
    return raw.astype(np.int8, copy=True)


def _validate_probabilities(expected_rate: float, significance_level: float) -> None:
    if not math.isfinite(expected_rate) or not 0.0 < expected_rate < 1.0:
        raise ValueError("expected_rate must be a finite probability strictly between 0 and 1.")
    if not math.isfinite(significance_level) or not 0.0 < significance_level < 1.0:
        raise ValueError("significance_level must be a finite probability strictly between 0 and 1.")


def _bernoulli_log_likelihood(successes: int, failures: int, probability: float) -> float:
    """Log likelihood with 0*log(0) treated as its continuous limit, zero."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("Bernoulli probability must lie in [0, 1].")
    value = 0.0
    if successes:
        if probability == 0.0:
            return -math.inf
        value += successes * math.log(probability)
    if failures:
        if probability == 1.0:
            return -math.inf
        value += failures * math.log1p(-probability)
    return value


def _interpret(p_value: float, significance_level: float) -> str:
    if p_value < significance_level:
        return f"Reject the null at significance level {significance_level:.3g}."
    return (
        f"Fail to reject the null at significance level {significance_level:.3g}; "
        "this is not proof that the VaR method is valid."
    )

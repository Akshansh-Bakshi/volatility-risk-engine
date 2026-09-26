"""Statistical hypothesis tests for return series diagnostics.

This module sits in the ``src.statistics`` layer and consumes only a
:class:`~src.preprocessing.return_series.ReturnSeries`.  It never imports
``yfinance``, never downloads data, and never touches the data or
preprocessing layers.

Available tests
---------------
* :func:`run_adf_test` — Augmented Dickey-Fuller stationarity test.
* :func:`run_kpss_test` — Kwiatkowski-Phillips-Schmidt-Shin stationarity test.
* :func:`run_ljung_box_returns` — Ljung-Box test on returns (autocorrelation).
* :func:`run_ljung_box_squared` — Ljung-Box test on squared returns (ARCH effect proxy).
* :func:`run_arch_lm_test` — Engle ARCH-LM test (volatility clustering).
* :func:`run_all_diagnostics` — run all five tests in one call.

Gap-flagged return policy
--------------------------
``ReturnSeries.spans_gap`` marks observations that span a calendar gap
(dropped row or run of missing sessions).  Such observations carry a
potentially spurious extreme return.  This module adopts the following
documented policy:

1. **Default (``exclude_gaps=False``)**: run diagnostics on the full
   series.  The number of flagged observations is always reported in every
   :class:`TestResult`.
2. **Sensitivity path (``exclude_gaps=True``)**: silently drop the
   flagged observations before running the test.  Callers should compare
   results from both paths and note any material difference.

No trading calendar is used.  The decision is recorded in every result.

Scale
-----
All tests are run on the **percent-scale** log-return series
(``ReturnSeries.percent``, i.e. decimal × 100) to stay consistent with
the modeling convention of the rest of the project.  The scale is
documented in every :class:`TestResult`.
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# Suppress known FutureWarnings from statsmodels 0.15.x about result_object kwarg.
# We call every function with result_object=False to opt into the current tuple API
# and silence the warning explicitly.
import statsmodels.tsa.stattools as _sm_ts
import statsmodels.stats.diagnostic as _sm_diag

from src.exceptions import InsufficientDataError, InvalidProfileInputError, StatisticsError
from src.logging_config import PACKAGE_LOGGER_NAME
from src.preprocessing.return_series import MODELING_SCALE, ReturnSeries

logger = logging.getLogger(f"{PACKAGE_LOGGER_NAME}.statistics.diagnostics")

# Default lag choices — standard practice values.
_DEFAULT_LB_LAGS = 10        # Ljung-Box: 10 lags for daily returns
_DEFAULT_LB_SQ_LAGS = 10     # Ljung-Box on squared returns
_DEFAULT_ARCH_LAGS = 5       # ARCH-LM: 5 lags
_MIN_OBSERVATIONS = 20       # Minimum series length to run any test meaningfully

# Significance level used for conclusions (not hardcoded into p-values, just for narrative).
_ALPHA = 0.05


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TestResult:
    """A single hypothesis-test result, fully self-describing and JSON-serialisable.

    Attributes:
        test_name: Short, human-readable test name.
        null_hypothesis: The null hypothesis of this test in plain English.
        statistic: The test statistic value.
        p_value: The p-value.  ``None`` if the library cannot compute one
            (e.g. KPSS p-values are capped at the table boundary — the
            reported value is a bound, recorded in ``notes``).
        lags_used: Number of lags included in the test, or ``None``.
        n_observations: Number of observations the test was run on.
        n_gap_flagged: Number of observations in the *original* full
            series that were flagged as spanning a gap.
        gaps_excluded: Whether gap-flagged observations were excluded
            before running the test.
        series_scale: Scale of the series the test was run on
            (always ``"percent"`` for this module).
        conclusion: Machine-readable conclusion string.  One of:
            ``"reject_null"`` or ``"fail_to_reject_null"``.
        interpretation: One or two sentences explaining what the
            conclusion means in plain English, including what rejecting or
            failing to reject the null implies for this specific test.
        notes: Optional list of additional strings (warnings, boundary
            conditions, caveats).
    """

    test_name: str
    null_hypothesis: str
    statistic: float
    p_value: float
    lags_used: int | None
    n_observations: int
    n_gap_flagged: int
    gaps_excluded: bool
    series_scale: str
    conclusion: str          # "reject_null" | "fail_to_reject_null"
    interpretation: str
    notes: tuple[str, ...] = field(default_factory=tuple)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def rejects_null(self) -> bool:
        """``True`` if the test rejects the null hypothesis at α = 5 %."""
        return self.conclusion == "reject_null"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "test_name": self.test_name,
            "null_hypothesis": self.null_hypothesis,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "lags_used": self.lags_used,
            "n_observations": self.n_observations,
            "n_gap_flagged": self.n_gap_flagged,
            "gaps_excluded": self.gaps_excluded,
            "series_scale": self.series_scale,
            "conclusion": self.conclusion,
            "interpretation": self.interpretation,
            "notes": list(self.notes),
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


@dataclass(frozen=True)
class ReturnDiagnostics:
    """Container for all five diagnostic test results for one return series.

    Build with :func:`run_all_diagnostics`; never construct directly.

    Attributes:
        ticker: Asset symbol the diagnostics were run on.
        n_observations_total: Observations in the full series (including
            any gap-flagged ones).
        n_gap_flagged: Observations flagged as spanning a gap.
        gaps_excluded: Whether gap-flagged observations were excluded.
        series_scale: Scale used for the tests (always ``"percent"``).
        adf: ADF test result.
        kpss: KPSS test result.
        ljung_box_returns: Ljung-Box test on returns.
        ljung_box_squared: Ljung-Box test on squared returns.
        arch_lm: Engle ARCH-LM test result.
    """

    ticker: str
    n_observations_total: int
    n_gap_flagged: int
    gaps_excluded: bool
    series_scale: str
    adf: TestResult
    kpss: TestResult
    ljung_box_returns: TestResult
    ljung_box_squared: TestResult
    arch_lm: TestResult

    @property
    def all_results(self) -> tuple[TestResult, ...]:
        """All five results in a consistent order."""
        return (
            self.adf,
            self.kpss,
            self.ljung_box_returns,
            self.ljung_box_squared,
            self.arch_lm,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "ticker": self.ticker,
            "n_observations_total": self.n_observations_total,
            "n_gap_flagged": self.n_gap_flagged,
            "gaps_excluded": self.gaps_excluded,
            "series_scale": self.series_scale,
            "adf": self.adf.to_dict(),
            "kpss": self.kpss.to_dict(),
            "ljung_box_returns": self.ljung_box_returns.to_dict(),
            "ljung_box_squared": self.ljung_box_squared.to_dict(),
            "arch_lm": self.arch_lm.to_dict(),
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Public test runners
# ---------------------------------------------------------------------------


def run_adf_test(
    returns: ReturnSeries,
    *,
    exclude_gaps: bool = False,
    max_lags: int | None = None,
) -> TestResult:
    """Run the Augmented Dickey-Fuller test for a unit root.

    **Null hypothesis**: the series has a unit root (is non-stationary).
    Rejecting H₀ is evidence of stationarity.

    Args:
        returns: The return series to test.
        exclude_gaps: If ``True``, gap-flagged observations are dropped
            before the test is run.
        max_lags: Maximum lag length for lag selection (AIC).  ``None``
            lets statsmodels choose automatically.

    Returns:
        A :class:`TestResult` describing the outcome.

    Raises:
        InvalidProfileInputError: If the series is too short to be tested.
    """
    values, n_gap = _prepare_series(returns, exclude_gaps=exclude_gaps)
    notes: list[str] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        result = _sm_ts.adfuller(
            values,
            maxlag=max_lags,
            autolag="AIC",
            result_object=False,
        )

    stat = float(result[0])
    pval = float(result[1])
    lags_used = int(result[2])
    n_obs = int(result[3])

    # Critical values are available but we just use the p-value for the conclusion.
    conclusion, interp = _adf_conclusion(pval)
    notes.append(f"Lag order selected by AIC: {lags_used}.")
    notes.append(f"Effective observations used by ADF: {n_obs}.")

    logger.info(
        "ADF test: ticker=%s stat=%.4f pval=%.4f lags=%d conclusion=%s",
        returns.ticker, stat, pval, lags_used, conclusion,
    )
    return TestResult(
        test_name="Augmented Dickey-Fuller (ADF)",
        null_hypothesis="The series has a unit root (is non-stationary).",
        statistic=stat,
        p_value=pval,
        lags_used=lags_used,
        n_observations=len(values),
        n_gap_flagged=n_gap,
        gaps_excluded=exclude_gaps,
        series_scale=MODELING_SCALE.value,
        conclusion=conclusion,
        interpretation=interp,
        notes=tuple(notes),
    )


def run_kpss_test(
    returns: ReturnSeries,
    *,
    exclude_gaps: bool = False,
    nlags: str = "auto",
) -> TestResult:
    """Run the KPSS test for stationarity around a constant.

    **Null hypothesis**: the series is stationary (around a constant mean).
    Rejecting H₀ is evidence of non-stationarity.

    Note: KPSS and ADF have *opposite* null hypotheses.  Failing to reject
    ADF (non-stationary) while also failing to reject KPSS (stationary) is
    contradictory evidence — consider both results together.

    Args:
        returns: The return series to test.
        exclude_gaps: If ``True``, gap-flagged observations are dropped.
        nlags: Lag selection method for KPSS (``"auto"`` or an integer).

    Returns:
        A :class:`TestResult` describing the outcome.

    Raises:
        InvalidProfileInputError: If the series is too short.
    """
    values, n_gap = _prepare_series(returns, exclude_gaps=exclude_gaps)
    notes: list[str] = []

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _sm_ts.kpss(
            values,
            regression="c",
            nlags=nlags,
            result_object=False,
        )

    stat = float(result[0])
    pval = float(result[1])
    lags_used = int(result[2])

    # Detect boundary clamping (common with short series).
    for w in caught:
        if issubclass(w.category, (UserWarning, FutureWarning)):
            msg = str(w.message)
            if "outside" in msg.lower() or "interpolation" in msg.lower():
                notes.append(
                    f"KPSS p-value is clamped at a table boundary ({pval:.4f}); "
                    "the true p-value may be more extreme."
                )

    conclusion, interp = _kpss_conclusion(pval)
    notes.append(f"Lags used by KPSS: {lags_used}.")
    notes.append(
        "KPSS H₀ is the *opposite* of ADF H₀: rejecting here means evidence "
        "of non-stationarity, not stationarity."
    )

    logger.info(
        "KPSS test: ticker=%s stat=%.4f pval=%.4f lags=%d conclusion=%s",
        returns.ticker, stat, pval, lags_used, conclusion,
    )
    return TestResult(
        test_name="KPSS (Kwiatkowski-Phillips-Schmidt-Shin)",
        null_hypothesis="The series is stationary around a constant mean.",
        statistic=stat,
        p_value=pval,
        lags_used=lags_used,
        n_observations=len(values),
        n_gap_flagged=n_gap,
        gaps_excluded=exclude_gaps,
        series_scale=MODELING_SCALE.value,
        conclusion=conclusion,
        interpretation=interp,
        notes=tuple(notes),
    )


def run_ljung_box_returns(
    returns: ReturnSeries,
    *,
    exclude_gaps: bool = False,
    lags: int = _DEFAULT_LB_LAGS,
) -> TestResult:
    """Ljung-Box test for serial autocorrelation in the returns.

    **Null hypothesis**: the first ``lags`` autocorrelations are all zero
    (no serial autocorrelation in the returns).

    The test is run on the **percent-scale** return series.

    Args:
        returns: The return series to test.
        exclude_gaps: If ``True``, gap-flagged observations are dropped.
        lags: Number of lags to include (default 10).

    Returns:
        A :class:`TestResult` at the requested lag order.

    Raises:
        InvalidProfileInputError: If the series is too short.
    """
    values, n_gap = _prepare_series(returns, exclude_gaps=exclude_gaps)
    _check_enough_for_lags(len(values), lags, returns.ticker, "Ljung-Box on returns")
    notes: list[str] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        lb_df = _sm_diag.acorr_ljungbox(values, lags=[lags], return_df=True)

    stat = float(lb_df["lb_stat"].iloc[-1])
    pval = float(lb_df["lb_pvalue"].iloc[-1])

    conclusion, interp = _lb_conclusion(pval, target="returns")
    notes.append(f"Test uses the joint null over {lags} lags.")

    logger.info(
        "LB-returns test: ticker=%s lags=%d stat=%.4f pval=%.4f conclusion=%s",
        returns.ticker, lags, stat, pval, conclusion,
    )
    return TestResult(
        test_name="Ljung-Box on returns",
        null_hypothesis=(
            f"The first {lags} autocorrelations of the return series are all zero "
            "(no serial autocorrelation)."
        ),
        statistic=stat,
        p_value=pval,
        lags_used=lags,
        n_observations=len(values),
        n_gap_flagged=n_gap,
        gaps_excluded=exclude_gaps,
        series_scale=MODELING_SCALE.value,
        conclusion=conclusion,
        interpretation=interp,
        notes=tuple(notes),
    )


def run_ljung_box_squared(
    returns: ReturnSeries,
    *,
    exclude_gaps: bool = False,
    lags: int = _DEFAULT_LB_SQ_LAGS,
) -> TestResult:
    """Ljung-Box test for serial autocorrelation in the *squared* returns.

    Significant autocorrelation in squared returns is a classic sign of
    volatility clustering (conditional heteroscedasticity), but this test
    is **not** the ARCH-LM test — it is a Box-Pierce-type test applied to
    the squared series rather than a regression-based LM test.

    **Null hypothesis**: the first ``lags`` autocorrelations of the squared
    returns are all zero (no volatility clustering / ARCH effect).

    Args:
        returns: The return series to test.
        exclude_gaps: If ``True``, gap-flagged observations are dropped.
        lags: Number of lags (default 10).

    Returns:
        A :class:`TestResult` at the requested lag order.

    Raises:
        InvalidProfileInputError: If the series is too short.
    """
    values, n_gap = _prepare_series(returns, exclude_gaps=exclude_gaps)
    _check_enough_for_lags(len(values), lags, returns.ticker, "Ljung-Box on squared returns")
    notes: list[str] = []
    notes.append(
        "This is a Ljung-Box test applied to squared returns — not the ARCH-LM test. "
        "Both detect volatility clustering but use different regression frameworks."
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        lb_df = _sm_diag.acorr_ljungbox(values ** 2, lags=[lags], return_df=True)

    stat = float(lb_df["lb_stat"].iloc[-1])
    pval = float(lb_df["lb_pvalue"].iloc[-1])

    conclusion, interp = _lb_conclusion(pval, target="squared returns")
    notes.append(f"Test uses the joint null over {lags} lags.")

    logger.info(
        "LB-squared test: ticker=%s lags=%d stat=%.4f pval=%.4f conclusion=%s",
        returns.ticker, lags, stat, pval, conclusion,
    )
    return TestResult(
        test_name="Ljung-Box on squared returns",
        null_hypothesis=(
            f"The first {lags} autocorrelations of squared returns are all zero "
            "(no volatility clustering / ARCH effect in squared returns)."
        ),
        statistic=stat,
        p_value=pval,
        lags_used=lags,
        n_observations=len(values),
        n_gap_flagged=n_gap,
        gaps_excluded=exclude_gaps,
        series_scale=MODELING_SCALE.value,
        conclusion=conclusion,
        interpretation=interp,
        notes=tuple(notes),
    )


def run_arch_lm_test(
    returns: ReturnSeries,
    *,
    exclude_gaps: bool = False,
    nlags: int = _DEFAULT_ARCH_LAGS,
) -> TestResult:
    """Run Engle's ARCH-LM (Lagrange Multiplier) test for conditional heteroscedasticity.

    This is a regression-based LM test distinct from the Ljung-Box-on-squared-returns
    test.  The ARCH-LM test regresses squared residuals on their own lags and tests
    whether the regression coefficients are jointly zero.

    **Null hypothesis**: there is no ARCH effect up to order ``nlags``
    (the conditional variance is constant — no volatility clustering).

    Args:
        returns: The return series to test.
        exclude_gaps: If ``True``, gap-flagged observations are dropped.
        nlags: Order of the ARCH test (number of lags; default 5).

    Returns:
        A :class:`TestResult` describing the outcome.

    Raises:
        InvalidProfileInputError: If the series is too short.
    """
    values, n_gap = _prepare_series(returns, exclude_gaps=exclude_gaps)
    _check_enough_for_lags(len(values), nlags, returns.ticker, "ARCH-LM")
    notes: list[str] = []
    notes.append(
        "ARCH-LM uses a regression-based LM statistic (χ² distributed). "
        "This is distinct from the Ljung-Box test on squared returns, which "
        "uses an autocorrelation-based Q statistic."
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        result = _sm_diag.het_arch(values, nlags=nlags, result_object=False)

    # het_arch returns (lm_stat, lm_pvalue, f_stat, f_pvalue) — use the LM pair.
    stat = float(result[0])
    pval = float(result[1])

    conclusion, interp = _arch_lm_conclusion(pval)
    notes.append(f"LM test uses {nlags} lag(s); χ² statistic reported.")

    logger.info(
        "ARCH-LM test: ticker=%s nlags=%d stat=%.4f pval=%.4f conclusion=%s",
        returns.ticker, nlags, stat, pval, conclusion,
    )
    return TestResult(
        test_name="Engle ARCH-LM",
        null_hypothesis=(
            f"There is no ARCH effect up to order {nlags} "
            "(conditional variance is constant — no volatility clustering)."
        ),
        statistic=stat,
        p_value=pval,
        lags_used=nlags,
        n_observations=len(values),
        n_gap_flagged=n_gap,
        gaps_excluded=exclude_gaps,
        series_scale=MODELING_SCALE.value,
        conclusion=conclusion,
        interpretation=interp,
        notes=tuple(notes),
    )


def run_all_diagnostics(
    returns: ReturnSeries,
    *,
    exclude_gaps: bool = False,
    lb_lags: int = _DEFAULT_LB_LAGS,
    lb_sq_lags: int = _DEFAULT_LB_SQ_LAGS,
    arch_lags: int = _DEFAULT_ARCH_LAGS,
) -> ReturnDiagnostics:
    """Run all five diagnostic tests and return a single :class:`ReturnDiagnostics`.

    Args:
        returns: The return series to diagnose.
        exclude_gaps: Passed to every individual test.
        lb_lags: Lag count for Ljung-Box on returns.
        lb_sq_lags: Lag count for Ljung-Box on squared returns.
        arch_lags: Lag count for the ARCH-LM test.

    Returns:
        A :class:`ReturnDiagnostics` aggregating all five results.

    Raises:
        InvalidProfileInputError: If the series is too short.
    """
    values, n_gap = _prepare_series(returns, exclude_gaps=exclude_gaps)
    n_total = int(returns.return_observations)

    adf = run_adf_test(returns, exclude_gaps=exclude_gaps)
    kpss = run_kpss_test(returns, exclude_gaps=exclude_gaps)
    lb_ret = run_ljung_box_returns(returns, exclude_gaps=exclude_gaps, lags=lb_lags)
    lb_sq = run_ljung_box_squared(returns, exclude_gaps=exclude_gaps, lags=lb_sq_lags)
    arch = run_arch_lm_test(returns, exclude_gaps=exclude_gaps, nlags=arch_lags)

    diag = ReturnDiagnostics(
        ticker=returns.ticker,
        n_observations_total=n_total,
        n_gap_flagged=n_gap,
        gaps_excluded=exclude_gaps,
        series_scale=MODELING_SCALE.value,
        adf=adf,
        kpss=kpss,
        ljung_box_returns=lb_ret,
        ljung_box_squared=lb_sq,
        arch_lm=arch,
    )
    logger.info(
        "ReturnDiagnostics complete: ticker=%s n=%d gaps=%d excluded=%s "
        "adf=%s kpss=%s lb=%s lb_sq=%s arch=%s",
        diag.ticker,
        diag.n_observations_total,
        diag.n_gap_flagged,
        diag.gaps_excluded,
        adf.conclusion,
        kpss.conclusion,
        lb_ret.conclusion,
        lb_sq.conclusion,
        arch.conclusion,
    )
    return diag


# ---------------------------------------------------------------------------
# ACF / PACF figure builders  (added to figures.py contract)
# ---------------------------------------------------------------------------
# These live here (not in figures.py) because they depend on statsmodels.
# figures.py intentionally has no statsmodels dependency so it stays importable
# without the heavy scientific stack for the dashboard's basic EDA page.


def plot_acf(
    returns: ReturnSeries,
    *,
    exclude_gaps: bool = False,
    lags: int = 40,
    alpha: float = 0.05,
    figsize: tuple[float, float] = (12, 4),
    title: str | None = None,
    squared: bool = False,
) -> "matplotlib.figure.Figure":  # type: ignore[name-defined]
    """Plot the sample ACF of returns (or squared returns).

    Args:
        returns: The return series.
        exclude_gaps: Exclude gap-flagged observations before computing ACF.
        lags: Maximum lag to display.
        alpha: Significance level for the confidence band (default 5 %).
        figsize: Width × height in inches.
        title: Override the auto-generated title.
        squared: If ``True``, compute the ACF of squared returns instead.

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    import matplotlib.figure as _mfig  # local import keeps module lightweight at test time
    values, _ = _prepare_series(returns, exclude_gaps=exclude_gaps)
    series = values ** 2 if squared else values

    acf_vals, confint = _sm_ts.acf(series, nlags=lags, alpha=alpha, fft=True, result_object=False)
    lags_arr = np.arange(len(acf_vals))

    conf_upper = confint[:, 1] - acf_vals
    conf_lower = acf_vals - confint[:, 0]

    fig = _mfig.Figure(figsize=figsize, tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(lags_arr[1:], acf_vals[1:], color="#1f77b4", alpha=0.7, width=0.4)
    ax.fill_between(
        lags_arr[1:],
        -conf_lower[1:],
        conf_upper[1:],
        alpha=0.2,
        color="#ff7f0e",
        label=f"{int((1 - alpha) * 100)}% CI",
    )
    ax.axhline(0, color="black", linewidth=0.6)
    label = "squared returns" if squared else "returns"
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
    ax.set_title(
        title or f"{returns.ticker} — ACF of {label} (percent scale)"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return fig


def plot_pacf(
    returns: ReturnSeries,
    *,
    exclude_gaps: bool = False,
    lags: int = 40,
    alpha: float = 0.05,
    figsize: tuple[float, float] = (12, 4),
    title: str | None = None,
) -> "matplotlib.figure.Figure":  # type: ignore[name-defined]
    """Plot the sample PACF of returns.

    Args:
        returns: The return series.
        exclude_gaps: Exclude gap-flagged observations before computing PACF.
        lags: Maximum lag to display.
        alpha: Significance level for the confidence band (default 5 %).
        figsize: Width × height in inches.
        title: Override the auto-generated title.

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    import matplotlib.figure as _mfig
    values, _ = _prepare_series(returns, exclude_gaps=exclude_gaps)

    # Use Ywm (Yule-Walker) method for partial ACF — well-conditioned for typical n.
    pacf_vals, confint = _sm_ts.pacf(values, nlags=lags, alpha=alpha, method="ywm")
    lags_arr = np.arange(len(pacf_vals))

    conf_upper = confint[:, 1] - pacf_vals
    conf_lower = pacf_vals - confint[:, 0]

    fig = _mfig.Figure(figsize=figsize, tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(lags_arr[1:], pacf_vals[1:], color="#2ca02c", alpha=0.7, width=0.4)
    ax.fill_between(
        lags_arr[1:],
        -conf_lower[1:],
        conf_upper[1:],
        alpha=0.2,
        color="#ff7f0e",
        label=f"{int((1 - alpha) * 100)}% CI",
    )
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Lag")
    ax.set_ylabel("PACF")
    ax.set_title(title or f"{returns.ticker} — PACF of returns (percent scale)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _prepare_series(
    returns: ReturnSeries,
    *,
    exclude_gaps: bool,
) -> tuple[np.ndarray, int]:
    """Extract the percent-scale series, optionally dropping gap-flagged rows.

    Returns:
        ``(values, n_gap_flagged)`` where ``values`` is a 1-D float64 array
        and ``n_gap_flagged`` is the count from the *original* full series.

    Raises:
        InsufficientDataError: If the (possibly filtered) series is too short.
    """
    n_gap = int(returns.spans_gap.sum())
    series = returns.percent  # percent scale: R_t × 100

    if exclude_gaps:
        keep = ~returns.spans_gap
        series = series[keep]

    values = np.asarray(series, dtype="float64")

    if len(values) < _MIN_OBSERVATIONS:
        raise InsufficientDataError(
            f"Cannot run diagnostic tests on {returns.ticker}: "
            f"only {len(values)} observation(s) available after gap filtering "
            f"(minimum required: {_MIN_OBSERVATIONS})."
        )
    return values, n_gap


def _check_enough_for_lags(n: int, lags: int, ticker: str, test_name: str) -> None:
    if n <= lags + 1:
        raise InsufficientDataError(
            f"Cannot run {test_name} for {ticker}: "
            f"{n} observations is not enough for {lags} lags."
        )


# ---------------------------------------------------------------------------
# Conclusion helpers — each test has its own null, so conclusions differ.
# ---------------------------------------------------------------------------


def _adf_conclusion(pval: float) -> tuple[str, str]:
    """ADF: H₀ = unit root (non-stationary).  Reject → evidence of stationarity."""
    if pval < _ALPHA:
        return (
            "reject_null",
            f"p = {pval:.4g} < {_ALPHA}. The ADF test rejects the null hypothesis of a unit "
            "root at the 5% level. This is evidence that the return series is stationary "
            "(mean-reverting), as expected for log returns of a traded asset.",
        )
    return (
        "fail_to_reject_null",
        f"p = {pval:.4g} ≥ {_ALPHA}. The ADF test does not reject the null hypothesis of "
        "a unit root at the 5% level. This is unusual for daily log returns and may indicate "
        "a very persistent series, a structural break, or insufficient data.",
    )


def _kpss_conclusion(pval: float) -> tuple[str, str]:
    """KPSS: H₀ = stationary.  Reject → evidence of non-stationarity.
    Note: opposite null to ADF."""
    if pval < _ALPHA:
        return (
            "reject_null",
            f"p = {pval:.4g} < {_ALPHA}. The KPSS test rejects its null hypothesis of "
            "stationarity at the 5% level. This is evidence that the series is non-stationary "
            "(e.g. a unit root or structural break). Note: this is the *opposite* "
            "of what ADF rejection means — rejecting KPSS indicates non-stationarity, "
            "not stationarity.",
        )
    return (
        "fail_to_reject_null",
        f"p = {pval:.4g} ≥ {_ALPHA}. The KPSS test does not reject its null hypothesis of "
        "stationarity. This is consistent with the series being stationary around a constant "
        "mean. Together with ADF rejection, this is the standard expectation for daily log returns.",
    )


def _lb_conclusion(pval: float, target: str) -> tuple[str, str]:
    """Ljung-Box: H₀ = no autocorrelation.  Reject → significant autocorrelation."""
    if pval < _ALPHA:
        extra = (
            " This is consistent with volatility clustering (conditional heteroscedasticity)."
            if "squared" in target
            else ""
        )
        return (
            "reject_null",
            f"p = {pval:.4g} < {_ALPHA}. The Ljung-Box test rejects the null of no "
            f"autocorrelation in {target} at the 5% level. There is statistically significant "
            f"serial dependence in {target}.{extra}",
        )
    return (
        "fail_to_reject_null",
        f"p = {pval:.4g} ≥ {_ALPHA}. The Ljung-Box test does not reject the null of no "
        f"autocorrelation in {target}. No significant serial dependence detected in {target} "
        "at the tested lag order.",
    )


def _arch_lm_conclusion(pval: float) -> tuple[str, str]:
    """ARCH-LM: H₀ = no ARCH effect.  Reject → significant volatility clustering."""
    if pval < _ALPHA:
        return (
            "reject_null",
            f"p = {pval:.4g} < {_ALPHA}. The ARCH-LM test rejects the null of no ARCH effect "
            "at the 5% level. There is evidence of significant conditional heteroscedasticity "
            "(volatility clustering) — volatility is not constant over time. "
            "This motivates GARCH-family modelling.",
        )
    return (
        "fail_to_reject_null",
        f"p = {pval:.4g} ≥ {_ALPHA}. The ARCH-LM test does not reject the null of no ARCH effect. "
        "No significant volatility clustering detected at the tested lag order. "
        "A constant-variance model may be adequate.",
    )

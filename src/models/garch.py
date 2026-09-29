"""GARCH(1,1) volatility model wrapper.

**Model**: Generalised Autoregressive Conditional Heteroscedasticity,
order (1,1).  Implemented via the ``arch`` library.

**Return-scale convention**: The model is fitted on **percent-scale** log
returns (``ReturnSeries.percent``).  The ``arch`` library is optimised for
data whose standard deviation is of order 1, not 0.01; fitting on decimal
returns produces numerically identical results but with omega scaled by 10⁻⁴,
which makes standard errors harder to inspect.  The scale is documented in
every result object so there is no ambiguity downstream.

GARCH(1,1) specification
--------------------------
The model is::

    r_t = μ + ε_t,    ε_t = σ_t · z_t,    z_t ~ D(0,1)
    σ²_t = ω + α · ε²_{t-1} + β · σ²_{t-1}

Parameters:
- ``μ`` (mu): constant mean return.
- ``ω`` (omega): long-run variance baseline.
- ``α`` (alpha[1]): ARCH coefficient — reaction to recent shocks.
- ``β`` (beta[1]): GARCH coefficient — persistence of past variance.

For stationarity of the variance process, α + β < 1 is required.

Defaults
--------
- Mean: ``"Constant"`` — a constant mean is appropriate for daily log returns.
- Distribution: ``"Normal"`` — Gaussian innovations.  A heavier-tailed
  distribution (e.g. ``"t"``) may fit daily equity returns better but is
  deferred to a later configuration option.
- Starting variance: estimated from the data by ``arch``.

Usage::

    from src.models.garch import GARCHModel

    model = GARCHModel(ticker="^NSEI")
    result = model.fit(return_series)
    print(result.daily_vol_pct.tail())
    print(result.aic, result.bic)
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd

from src.exceptions import (
    VolatilityModelConfigError,
    VolatilityModelDataError,
    VolatilityModelFitError,
)
from src.models.result import ModelFitResult, _TRADING_DAYS_PER_YEAR
from src.preprocessing.return_series import MODELING_SCALE, ReturnSeries
from src.logging_config import PACKAGE_LOGGER_NAME

logger = logging.getLogger(f"{PACKAGE_LOGGER_NAME}.models.garch")

_MIN_OBSERVATIONS = 50  # arch needs enough data to estimate variance dynamics


class GARCHModel:
    """Fit a GARCH(1,1) model to a :class:`~src.preprocessing.return_series.ReturnSeries`.

    Args:
        p: ARCH order (number of lagged squared error terms).  Must be ≥ 1.
        q: GARCH order (number of lagged variance terms).  Must be ≥ 0.
        mean: Mean specification.  ``"Constant"`` (default) fits a constant
            mean return.  ``"Zero"`` constrains the mean to zero.
        dist: Innovation distribution.  ``"Normal"`` (default) uses Gaussian
            innovations.
        ticker: Asset symbol (informational; stored in the result).
        annualization_factor: Number of trading days used to annualise
            daily volatility.  Default is 252.

    Raises:
        VolatilityModelConfigError: If any parameter is invalid.
    """

    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        *,
        mean: str = "Constant",
        dist: str = "Normal",
        ticker: str = "UNKNOWN",
        annualization_factor: int = _TRADING_DAYS_PER_YEAR,
    ) -> None:
        _validate_garch_config(p, q, mean, dist, annualization_factor)
        self._p = p
        self._q = q
        self._mean = mean
        self._dist = dist
        self._ticker = ticker
        self._ann = annualization_factor

    @property
    def model_name(self) -> str:
        return f"GARCH({self._p},{self._q})"

    @property
    def config(self) -> dict[str, Any]:
        return {
            "p": self._p,
            "q": self._q,
            "mean": self._mean,
            "dist": self._dist,
        }

    def fit(
        self,
        returns: ReturnSeries,
        *,
        arch_lm_note: str | None = None,
    ) -> ModelFitResult:
        """Fit the GARCH model and return a frozen :class:`~src.models.result.ModelFitResult`.

        Args:
            returns: A validated :class:`~src.preprocessing.return_series.ReturnSeries`.
                The percent-scale series (``returns.percent``) is used.
            arch_lm_note: Optional note from Stage 5 ARCH-LM diagnostics.
                If supplied, it is included in the result's ``notes`` without
                re-running the test.  This avoids duplicating statistical logic.

        Returns:
            A frozen :class:`~src.models.result.ModelFitResult`.

        Raises:
            VolatilityModelDataError: If the return series is too short or invalid.
            VolatilityModelFitError: If ``arch`` fails to converge or produces
                non-finite outputs.
        """
        _validate_return_series(returns, _MIN_OBSERVATIONS)
        pct = returns.percent  # percent-scale: unit travels with the series name

        from arch import arch_model as _arch_model  # local: heavy import

        notes: list[str] = []
        if arch_lm_note:
            notes.append(arch_lm_note)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                am = _arch_model(
                    pct,
                    vol="Garch",
                    p=self._p,
                    q=self._q,
                    mean=self._mean,
                    dist=self._dist,
                )
                res = am.fit(disp="off")
            except Exception as exc:
                raise VolatilityModelFitError(
                    f"GARCH fitting failed for {self._ticker}: {exc}"
                ) from exc

        for w in caught:
            msg = str(w.message)
            if "convergence" in msg.lower() or "not positive definite" in msg.lower():
                notes.append(f"arch warning: {msg}")

        converged = int(res.convergence_flag) == 0
        if not converged:
            notes.append(
                f"GARCH optimiser did not fully converge "
                f"(convergence_flag={res.convergence_flag}).  "
                "Parameter estimates may be unreliable."
            )

        cond_vol_pct = pd.Series(
            res.conditional_volatility,
            index=pct.index,
            name="daily_vol_pct",
        )
        ann_vol_pct = cond_vol_pct * np.sqrt(self._ann)
        ann_vol_pct.name = "annualized_vol_pct"

        std_resid = pd.Series(
            res.std_resid,
            index=pct.index,
            name="std_resid",
        )

        if not np.all(np.isfinite(cond_vol_pct.values)):
            raise VolatilityModelFitError(
                f"GARCH produced non-finite conditional volatility for {self._ticker}."
            )

        params = {str(k): float(v) for k, v in res.params.items()}
        stderr = {str(k): float(v) for k, v in res.std_err.items()}

        logger.info(
            "%s fitted: ticker=%s n=%d converged=%s aic=%.2f bic=%.2f",
            self.model_name, self._ticker, len(pct), converged, res.aic, res.bic,
        )

        return ModelFitResult(
            model_name=self.model_name,
            model_type="garch",
            config=self.config,
            ticker=self._ticker,
            frequency=returns.frequency,
            return_scale="percent",
            n_observations=len(pct),
            fit_start=pd.Timestamp(pct.index[0]).date(),
            fit_end=pd.Timestamp(pct.index[-1]).date(),
            converged=converged,
            log_likelihood=float(res.loglikelihood),
            aic=float(res.aic),
            bic=float(res.bic),
            params=params,
            std_errors=stderr,
            daily_vol_pct=cond_vol_pct,
            annualized_vol_pct=ann_vol_pct,
            annualization_factor=self._ann,
            std_residuals=std_resid,
            notes=tuple(notes),
        )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_garch_config(
    p: int, q: int, mean: str, dist: str, ann: int
) -> None:
    if not isinstance(p, int) or isinstance(p, bool) or p < 1:
        raise VolatilityModelConfigError(
            f"GARCH p (ARCH order) must be a positive integer, got {p!r}."
        )
    if not isinstance(q, int) or isinstance(q, bool) or q < 0:
        raise VolatilityModelConfigError(
            f"GARCH q (GARCH order) must be a non-negative integer, got {q!r}."
        )
    _valid_means = ("Constant", "Zero", "LS", "ARX", "HAR", "HARX", "AR")
    if mean not in _valid_means:
        raise VolatilityModelConfigError(
            f"mean must be one of {_valid_means}, got {mean!r}."
        )
    _valid_dists = ("Normal", "StudentsT", "SkewStudent", "GED")
    if dist not in _valid_dists:
        raise VolatilityModelConfigError(
            f"dist must be one of {_valid_dists}, got {dist!r}."
        )
    if not isinstance(ann, int) or isinstance(ann, bool) or ann < 1:
        raise VolatilityModelConfigError(
            f"annualization_factor must be a positive integer, got {ann!r}."
        )


def _validate_return_series(returns: ReturnSeries, min_obs: int) -> None:
    """Raise VolatilityModelDataError if the series cannot be modelled."""
    if not isinstance(returns, ReturnSeries):
        raise VolatilityModelDataError(
            f"Expected a ReturnSeries, got {type(returns).__name__}."
        )
    if returns.return_observations < min_obs:
        raise VolatilityModelDataError(
            f"ReturnSeries for {returns.ticker} has only "
            f"{returns.return_observations} observations "
            f"(minimum required: {min_obs})."
        )

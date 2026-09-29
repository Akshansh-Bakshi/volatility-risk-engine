"""EGARCH(1,1) volatility model wrapper.

**Model**: Exponential GARCH, order (1,1).  Implemented via the ``arch`` library.

**Return-scale convention**: Identical to GARCH — the model is fitted on
**percent-scale** log returns (``ReturnSeries.percent``).  See
:mod:`src.models.garch` for the rationale.

EGARCH(1,1) specification
---------------------------
The Nelson (1991) EGARCH model specifies the log of the conditional variance::

    ln(σ²_t) = ω + α·[|z_{t-1}| - E|z_{t-1}|] + γ·z_{t-1} + β·ln(σ²_{t-1})

Parameters:
- ``ω`` (omega): intercept of the log-variance equation.
- ``α`` (alpha[1]): magnitude effect (symmetric response to shocks).
- ``β`` (beta[1]): persistence of log-variance.
- ``γ`` (gamma[1]): asymmetry / leverage effect.  A negative γ indicates
  that negative returns increase volatility more than positive returns of
  the same magnitude (the "leverage effect" common in equity returns).

Key difference from GARCH
--------------------------
1. **Asymmetry**: EGARCH captures the leverage effect via ``γ``.  GARCH(1,1)
   assumes symmetric responses.
2. **No non-negativity constraint**: Because the log of variance is modelled,
   ω, α, β need not be constrained to be positive.
3. **Stationarity**: The variance process is stationary if |β| < 1.

Usage::

    from src.models.egarch import EGARCHModel

    model = EGARCHModel(ticker="^NSEI")
    result = model.fit(return_series)
    print(result.params)   # includes gamma[1] for leverage effect
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
from src.models.garch import _validate_return_series
from src.models.result import ModelFitResult, _TRADING_DAYS_PER_YEAR
from src.preprocessing.return_series import ReturnSeries
from src.logging_config import PACKAGE_LOGGER_NAME

logger = logging.getLogger(f"{PACKAGE_LOGGER_NAME}.models.egarch")

_MIN_OBSERVATIONS = 50


class EGARCHModel:
    """Fit an EGARCH(1,1) model to a :class:`~src.preprocessing.return_series.ReturnSeries`.

    Interface is intentionally parallel to :class:`~src.models.garch.GARCHModel`
    so downstream code can treat both models uniformly via their shared
    :class:`~src.models.result.ModelFitResult` output.

    Args:
        p: Order of the ARCH component (lagged absolute innovations).
            Must be ≥ 1.
        o: Order of the asymmetric innovation component (leverage).
            Must be ≥ 0.
        q: Order of the GARCH component (lagged log-variance terms).
            Must be ≥ 1 for EGARCH (unlike plain GARCH where q ≥ 0 is valid).
        mean: Mean specification.  ``"Constant"`` (default).
        dist: Innovation distribution.  ``"Normal"`` (default).
        ticker: Asset symbol (informational).
        annualization_factor: Trading days per year for annualisation
            (default 252).

    Raises:
        VolatilityModelConfigError: If any parameter is invalid.
    """

    def __init__(
        self,
        p: int = 1,
        o: int = 1,
        q: int = 1,
        *,
        mean: str = "Constant",
        dist: str = "Normal",
        ticker: str = "UNKNOWN",
        annualization_factor: int = _TRADING_DAYS_PER_YEAR,
    ) -> None:
        _validate_egarch_config(p, o, q, mean, dist, annualization_factor)
        self._o = o
        self._p = p
        self._q = q
        self._mean = mean
        self._dist = dist
        self._ticker = ticker
        self._ann = annualization_factor

    @property
    def model_name(self) -> str:
        return f"EGARCH({self._p},{self._o},{self._q})"

    @property
    def config(self) -> dict[str, Any]:
        return {
            "p": self._p,
            "o": self._o,
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
        """Fit the EGARCH model and return a frozen :class:`~src.models.result.ModelFitResult`.

        Args:
            returns: A validated :class:`~src.preprocessing.return_series.ReturnSeries`.
                The percent-scale series (``returns.percent``) is used.
            arch_lm_note: Optional note from Stage 5 ARCH-LM diagnostics.

        Returns:
            A frozen :class:`~src.models.result.ModelFitResult`.

        Raises:
            VolatilityModelDataError: If the return series is too short or invalid.
            VolatilityModelFitError: If ``arch`` fails or produces non-finite outputs.
        """
        _validate_return_series(returns, _MIN_OBSERVATIONS)
        pct = returns.percent

        from arch import arch_model as _arch_model

        notes: list[str] = []
        if arch_lm_note:
            notes.append(arch_lm_note)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                am = _arch_model(
                    pct,
                    vol="EGARCH",
                    p=self._p,
                    o=self._o,
                    q=self._q,
                    mean=self._mean,
                    dist=self._dist,
                )
                res = am.fit(disp="off")
            except Exception as exc:
                raise VolatilityModelFitError(
                    f"EGARCH fitting failed for {self._ticker}: {exc}"
                ) from exc

        for w in caught:
            msg = str(w.message)
            if "convergence" in msg.lower() or "not positive definite" in msg.lower():
                notes.append(f"arch warning: {msg}")

        converged = int(res.convergence_flag) == 0
        if not converged:
            notes.append(
                f"EGARCH optimiser did not fully converge "
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
                f"EGARCH produced non-finite conditional volatility for {self._ticker}."
            )

        params = {str(k): float(v) for k, v in res.params.items()}
        stderr = {str(k): float(v) for k, v in res.std_err.items()}

        logger.info(
            "%s fitted: ticker=%s n=%d converged=%s aic=%.2f bic=%.2f",
            self.model_name, self._ticker, len(pct), converged, res.aic, res.bic,
        )

        return ModelFitResult(
            model_name=self.model_name,
            model_type="egarch",
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
# Validation
# ---------------------------------------------------------------------------


def _validate_egarch_config(
    p: int, o: int, q: int, mean: str, dist: str, ann: int
) -> None:
    if not isinstance(p, int) or isinstance(p, bool) or p < 1:
        raise VolatilityModelConfigError(
            f"EGARCH p must be a positive integer, got {p!r}."
        )
    if not isinstance(o, int) or isinstance(o, bool) or o < 0:
        raise VolatilityModelConfigError(
            f"EGARCH o must be a non-negative integer, got {o!r}."
        )
    if not isinstance(q, int) or isinstance(q, bool) or q < 1:
        raise VolatilityModelConfigError(
            f"EGARCH q must be a positive integer (≥ 1), got {q!r}."
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

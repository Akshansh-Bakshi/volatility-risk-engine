"""Frozen, JSON-serialisable result object for a fitted volatility model.

:class:`ModelFitResult` is the common output type for GARCH(1,1) and
EGARCH(1,1).  It stores no raw ``arch`` objects — all information is
extracted at fit time and stored as plain Python types or
:class:`pandas.Series` (which are themselves serialisable via their
``.to_dict()`` method).

Return-scale convention
-----------------------
The model is fitted on **percent-scale** log returns (``ReturnSeries.percent``).
All conditional-volatility series stored here are therefore expressed in
**percentage units** (same scale as the returns).

Daily vs. annualised volatility
--------------------------------
Two volatility series are exposed:

* :attr:`daily_vol_pct` — the raw model output; same unit as the returns
  (percent per day).
* :attr:`annualized_vol_pct` — scaled by ``sqrt(252)`` where 252 is the
  conventional number of trading days per year.  This is a standard
  approximation; the true number of trading days varies by market and year.
  The annualization factor is stored in :attr:`annualization_factor` so
  users can verify or override it downstream.

No silent unit changes are ever applied.

Comparison compatibility
------------------------
Because GARCH and EGARCH produce results using this same schema, a later
comparison layer can compute::

    garch_aic vs egarch_aic
    garch_bic vs egarch_bic
    garch_loglik vs egarch_loglik

without model-specific branches.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd


_TRADING_DAYS_PER_YEAR: int = 252  # standard annualisation convention


@dataclass(frozen=True)
class ModelFitResult:
    """The output of fitting a volatility model to a return series.

    Attributes:
        model_name: Human-readable model name (e.g. ``"GARCH(1,1)"``).
        model_type: Short family string: ``"garch"`` or ``"egarch"``.
        config: Model configuration dictionary (orders, mean spec, dist,
            etc.).  Must be JSON-serialisable.
        ticker: Asset symbol.
        frequency: Observation frequency (``"1d"`` for daily).
        return_scale: Always ``"percent"`` — the model is fitted on
            percent-scale log returns.
        n_observations: Number of return observations used in fitting.
        fit_start: Date of the first observation in the fitting window.
        fit_end: Date of the last observation in the fitting window.
        converged: ``True`` iff the optimiser reported successful convergence
            (``arch`` convergence flag == 0).
        log_likelihood: Value of the log-likelihood at convergence.
        aic: Akaike Information Criterion.
        bic: Bayesian Information Criterion.
        params: Parameter estimates as ``{name: value}``.
        std_errors: Parameter standard errors as ``{name: se}``.
        daily_vol_pct: Fitted conditional volatility series in **percentage
            units per day** (same scale as the input returns), indexed by date.
            This is the raw model output — not annualised.
        annualized_vol_pct: Conditional volatility scaled by
            ``sqrt(annualization_factor)`` to approximate annual percentage
            volatility.  Uses the ``sqrt(252)`` convention by default.
        annualization_factor: The number of trading days used for annualisation
            (default ``252``).  Stored here so downstream code can verify or
            change the convention.
        std_residuals: Standardised residuals ``(return - mean) / cond_std``,
            indexed by date.  Useful for diagnostics.
        notes: Informational messages (convergence warnings, weak-ARCH notices).
    """

    model_name: str
    model_type: str
    config: dict[str, Any]
    ticker: str
    frequency: str
    return_scale: str   # always "percent"
    n_observations: int
    fit_start: date
    fit_end: date
    converged: bool
    log_likelihood: float
    aic: float
    bic: float
    params: dict[str, float]
    std_errors: dict[str, float]
    daily_vol_pct: pd.Series      # index: DatetimeIndex, values: float (% per day)
    annualized_vol_pct: pd.Series  # index: DatetimeIndex, values: float (% per year)
    annualization_factor: int
    std_residuals: pd.Series      # index: DatetimeIndex, values: float
    notes: tuple[str, ...] = field(default_factory=tuple)

    # ------------------------------------------------------------------ #
    # Serialisation                                                        #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary.

        :class:`pandas.Series` are converted to ``{date_iso: value}`` mappings.
        Non-finite floats (rare but possible at numerical boundaries) are
        represented as ``null`` in JSON via Python ``None``.
        """

        def _ser_to_dict(s: pd.Series) -> dict[str, float | None]:
            return {
                str(pd.Timestamp(idx).date()): (float(v) if math.isfinite(v) else None)
                for idx, v in s.items()
            }

        def _safe_float(v: float) -> float | None:
            return float(v) if math.isfinite(v) else None

        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "config": self.config,
            "ticker": self.ticker,
            "frequency": self.frequency,
            "return_scale": self.return_scale,
            "n_observations": self.n_observations,
            "fit_start": self.fit_start.isoformat(),
            "fit_end": self.fit_end.isoformat(),
            "converged": self.converged,
            "log_likelihood": _safe_float(self.log_likelihood),
            "aic": _safe_float(self.aic),
            "bic": _safe_float(self.bic),
            "params": {k: _safe_float(v) for k, v in self.params.items()},
            "std_errors": {k: _safe_float(v) for k, v in self.std_errors.items()},
            "daily_vol_pct": _ser_to_dict(self.daily_vol_pct),
            "annualized_vol_pct": _ser_to_dict(self.annualized_vol_pct),
            "annualization_factor": self.annualization_factor,
            "std_residuals": _ser_to_dict(self.std_residuals),
            "notes": list(self.notes),
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

"""Target-aware validation helpers shared by forecast and risk experiments.

These helpers score saved out-of-sample observations. They do not fit models.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def fz0_var_es_score(
    realised_loss: pd.Series | np.ndarray,
    var: pd.Series | np.ndarray,
    expected_shortfall: pd.Series | np.ndarray,
    tail_probability: float,
) -> np.ndarray:
    """Return the FZ0 joint VaR/ES score for positive upper-tail losses.

    ``VaR`` and ``ES`` use positive loss units and ES must be strictly positive.
    Lower scores are better. Inputs are paired positionally and must share shape.
    Formula: I(L>VaR)(L-VaR)/(alpha*ES) + VaR/ES + log(ES) - 1.
    """
    if isinstance(tail_probability, bool) or not math.isfinite(tail_probability) or not 0 < tail_probability < 1:
        raise ValueError("tail_probability must be finite and strictly between 0 and 1.")
    loss = np.asarray(realised_loss, dtype="float64")
    threshold = np.asarray(var, dtype="float64")
    es = np.asarray(expected_shortfall, dtype="float64")
    if loss.ndim != 1 or threshold.shape != loss.shape or es.shape != loss.shape or loss.size == 0:
        raise ValueError("realised_loss, VaR and ES must be non-empty one-dimensional arrays of equal length.")
    if not np.isfinite(loss).all() or not np.isfinite(threshold).all() or not np.isfinite(es).all() or np.any(es <= 0):
        raise ValueError("loss, VaR and ES must be finite and ES must be positive.")
    alpha = float(tail_probability)
    return ((loss > threshold) * (loss - threshold) / (alpha * es)
            + threshold / es + np.log(es) - 1.0)


def classify_volatility_regime(
    realised_volatility: pd.Series,
    *,
    lookback: int = 252,
    minimum_history: int = 60,
) -> pd.Series:
    """Classify dates against the expanding median of *prior* volatility values.

    The current observation is excluded from its threshold. Dates without the
    requested prior history are marked missing, so the classifier cannot use
    future observations or a threshold chosen from test performance.
    """
    if not isinstance(realised_volatility, pd.Series) or not isinstance(realised_volatility.index, pd.DatetimeIndex):
        raise ValueError("realised_volatility must be a Series with a DatetimeIndex.")
    if realised_volatility.empty or realised_volatility.index.has_duplicates or not realised_volatility.index.is_monotonic_increasing:
        raise ValueError("realised_volatility must be non-empty, unique and chronological.")
    if isinstance(lookback, bool) or not isinstance(lookback, int) or lookback < 1:
        raise ValueError("lookback must be a positive integer.")
    if isinstance(minimum_history, bool) or not isinstance(minimum_history, int) or minimum_history < 1:
        raise ValueError("minimum_history must be a positive integer.")
    values = pd.to_numeric(realised_volatility, errors="coerce").astype("float64")
    if not np.isfinite(values.dropna().to_numpy()).all():
        raise ValueError("realised_volatility must contain only finite values or missing warm-up values.")
    history = values.shift(1).rolling(lookback, min_periods=minimum_history)
    threshold = history.median()
    result = pd.Series(pd.NA, index=values.index, dtype="string", name="volatility_regime")
    valid = values.notna() & threshold.notna()
    result.loc[valid & values.ge(threshold)] = "high"
    result.loc[valid & values.lt(threshold)] = "low"
    return result


def summarize_by_regime(
    values: pd.DataFrame,
    regime: pd.Series,
    *,
    metrics: dict[str, str],
) -> pd.DataFrame:
    """Summarize named numeric metrics and sample counts by aligned regime."""
    if not values.index.equals(regime.index):
        raise ValueError("values and regime must have identical indices.")
    missing = sorted(set(metrics.values()) - set(values.columns))
    if missing:
        raise ValueError(f"values are missing metric column(s): {', '.join(missing)}.")
    frame = values.copy()
    frame["regime"] = regime
    rows = []
    for label, group in frame.dropna(subset=["regime"]).groupby("regime", sort=True):
        row: dict[str, object] = {"regime": label, "observations": len(group)}
        for output_name, column in metrics.items():
            numeric = pd.to_numeric(group[column], errors="coerce").dropna()
            row[output_name] = float(numeric.mean()) if len(numeric) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows, columns=["regime", "observations", *metrics])

"""Transparent historical and hypothetical portfolio stress scenarios."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Portfolio:
    """A fully invested, long-only portfolio with an explicit starting value."""

    name: str
    starting_value: float
    weights: Mapping[str, float]
    currency: str = "INR"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("portfolio name must be a non-empty string.")
        if isinstance(self.starting_value, bool) or not isinstance(self.starting_value, Real) or not math.isfinite(self.starting_value) or self.starting_value <= 0:
            raise ValueError("starting_value must be positive and finite.")
        if not isinstance(self.weights, Mapping) or not self.weights:
            raise ValueError("portfolio weights must include at least one asset.")
        normalized: dict[str, float] = {}
        for asset, weight in self.weights.items():
            if not isinstance(asset, str) or not asset.strip():
                raise ValueError("asset names must be non-empty strings.")
            if isinstance(weight, bool) or not isinstance(weight, Real) or not math.isfinite(weight) or weight < 0:
                raise ValueError(f"weight for {asset!r} must be finite and non-negative.")
            normalized[asset] = float(weight)
        if not math.isclose(sum(normalized.values()), 1.0, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("long-only portfolio weights must sum to 1.0.")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "starting_value", float(self.starting_value))
        object.__setattr__(self, "weights", MappingProxyType(normalized))
        if not isinstance(self.currency, str) or not self.currency.strip():
            raise ValueError("currency must be a non-empty string.")
        object.__setattr__(self, "currency", self.currency.strip())


def hypothetical_scenarios(
    portfolio: Portfolio,
    *,
    shocks: Iterable[float] = (-0.05, -0.10, -0.20),
    scope: str = "portfolio_wide",
    asset: str | None = None,
) -> pd.DataFrame:
    """Apply simple-return shocks to every holding or to one specified holding.

    Shocks are simple returns as decimals (for example ``-0.10`` is a 10%
    decline). These are deterministic what-if scenarios, not forecasts or
    probability statements.
    """
    if scope not in {"portfolio_wide", "single_asset"}:
        raise ValueError("scope must be 'portfolio_wide' or 'single_asset'.")
    if scope == "single_asset" and asset not in portfolio.weights:
        raise ValueError("single_asset scope requires an asset held by the portfolio.")
    rows: list[dict[str, object]] = []
    for shock in shocks:
        if isinstance(shock, bool) or not isinstance(shock, (int, float)) or not math.isfinite(shock) or not -1.0 <= shock <= 0.0:
            raise ValueError("hypothetical shocks must be finite declines between -100% and 0%.")
        asset_returns = {name: float(shock) if scope == "portfolio_wide" or name == asset else 0.0 for name in portfolio.weights}
        name = f"Hypothetical {scope.replace('_', ' ')} {abs(float(shock)):.0%} decline"
        assumption = (
            "Same simple-return shock applied to every portfolio holding."
            if scope == "portfolio_wide"
            else f"Shock applies only to {asset}; other holdings are unchanged."
        )
        rows.append(_scenario_row(
            portfolio, name=name, scenario_type="hypothetical", asset_returns=asset_returns,
            assumptions=assumption, start_date=None, end_date=None, horizon=None,
            gap_flagged_observations=0,
        ))
    return pd.DataFrame(rows)


def historical_scenarios(
    portfolio: Portfolio,
    log_returns_decimal: pd.DataFrame,
    *,
    horizon: int = 1,
    worst_n: int = 5,
    gap_flags: pd.Series | None = None,
) -> pd.DataFrame:
    """Return the worst ``worst_n`` observed portfolio shocks over trailing windows.

    Input columns are decimal log returns. The horizon is compounded per asset
    as ``exp(sum(log_return)) - 1``; portfolio value then uses a buy-and-hold
    weighted sum with no rebalancing, fees, cash flows, or leverage.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer.")
    if isinstance(worst_n, bool) or not isinstance(worst_n, int) or worst_n < 1:
        raise ValueError("worst_n must be a positive integer.")
    if not isinstance(log_returns_decimal, pd.DataFrame) or not isinstance(log_returns_decimal.index, pd.DatetimeIndex):
        raise ValueError("log_returns_decimal must be a DataFrame with a DatetimeIndex.")
    if log_returns_decimal.empty or log_returns_decimal.index.hasnans or log_returns_decimal.index.tz is not None or log_returns_decimal.index.has_duplicates or not log_returns_decimal.index.is_monotonic_increasing:
        raise ValueError("historical returns must be non-empty, unique, and chronological.")
    missing_assets = sorted(set(portfolio.weights) - set(log_returns_decimal.columns))
    if missing_assets:
        raise ValueError(f"historical returns are missing portfolio asset(s): {', '.join(missing_assets)}.")
    sample = log_returns_decimal.loc[:, list(portfolio.weights)].apply(pd.to_numeric, errors="coerce").astype("float64")
    if not np.isfinite(sample.to_numpy()).all():
        raise ValueError("historical returns contain missing or non-finite values.")
    if len(sample) < horizon:
        raise ValueError(f"Need at least {horizon} historical observations; got {len(sample)}.")
    if gap_flags is None:
        flags = pd.Series(False, index=sample.index)
    else:
        if not isinstance(gap_flags, pd.Series) or gap_flags.dtype != np.dtype(bool) or not gap_flags.index.equals(sample.index) or gap_flags.isna().any():
            raise ValueError("gap_flags must be a complete Series aligned with historical returns.")
        flags = gap_flags.astype(bool)

    rows: list[dict[str, object]] = []
    for end_position in range(horizon - 1, len(sample)):
        start_position = end_position - horizon + 1
        window = sample.iloc[start_position:end_position + 1]
        with np.errstate(over="raise", invalid="raise"):
            try:
                asset_returns = np.expm1(window.sum(axis=0).to_numpy(dtype="float64"))
            except FloatingPointError as exc:
                raise ValueError("historical scenario return overflowed.") from exc
        if not np.isfinite(asset_returns).all():
            raise ValueError("historical scenario return is non-finite.")
        start_date = sample.index[start_position]
        end_date = sample.index[end_position]
        rows.append(_scenario_row(
            portfolio,
            name=f"Historical {horizon}-session {start_date.date()} to {end_date.date()}",
            scenario_type="historical",
            asset_returns=dict(zip(sample.columns, asset_returns.tolist(), strict=True)),
            assumptions=(
                f"Observed {horizon}-session compounded log-return window; buy-and-hold weights, "
                "no rebalancing, fees, cash flows, or leverage."
            ),
            start_date=start_date,
            end_date=end_date,
            horizon=horizon,
            gap_flagged_observations=int(flags.iloc[start_position:end_position + 1].sum()),
        ))
    result = pd.DataFrame(rows).sort_values("loss_amount", ascending=False, kind="stable")
    return result.head(worst_n).reset_index(drop=True)


def _scenario_row(
    portfolio: Portfolio,
    *,
    name: str,
    scenario_type: str,
    asset_returns: Mapping[str, float],
    assumptions: str,
    start_date: pd.Timestamp | None,
    end_date: pd.Timestamp | None,
    horizon: int | None,
    gap_flagged_observations: int,
) -> dict[str, object]:
    missing = set(portfolio.weights) - set(asset_returns)
    if missing:
        raise ValueError(f"scenario is missing asset return(s): {', '.join(sorted(missing))}.")
    contributions: dict[str, float] = {}
    gross_value = 0.0
    for asset, weight in portfolio.weights.items():
        simple_return = float(asset_returns[asset])
        if not math.isfinite(simple_return) or simple_return < -1.0:
            raise ValueError(f"scenario return for {asset!r} must be finite and greater than -100%.")
        allocation = portfolio.starting_value * weight
        gross_value += allocation * (1.0 + simple_return)
        contributions[asset] = -allocation * simple_return
    loss = portfolio.starting_value - gross_value
    loss_pct = loss / portfolio.starting_value * 100.0
    return {
        "scenario_name": name,
        "scenario_type": scenario_type,
        "assumptions": assumptions,
        "scenario_start": None if start_date is None else start_date.date().isoformat(),
        "scenario_end": None if end_date is None else end_date.date().isoformat(),
        "horizon_sessions": horizon,
        "starting_value": portfolio.starting_value,
        "loss_amount": loss,
        "loss_pct": loss_pct,
        "stressed_value": gross_value,
        "asset_simple_returns": json.dumps({asset: float(asset_returns[asset]) for asset in portfolio.weights}, sort_keys=True),
        "asset_contributions": json.dumps(contributions, sort_keys=True),
        "gap_flagged_observations": gap_flagged_observations,
        "currency": portfolio.currency,
    }

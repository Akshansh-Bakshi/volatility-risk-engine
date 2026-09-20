"""Deterministic test doubles for the data layer: no network, no randomness."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from yfinance import exceptions as yf_exceptions

from src.data.market_data import INDEX_NAME, MarketDataRequest

_NO_OVERRIDE: Any = object()

# Yahoo's unadjusted close sits above the dividend-adjusted one for historical dates.
UNADJUSTED_FACTOR = 1.1


def make_prices(*, periods: int = 300, start: str = "2023-01-02") -> pd.DataFrame:
    """Standard-schema price frame on consecutive business days with fixed values."""
    index = pd.bdate_range(start, periods=periods, name=INDEX_NAME)
    k = np.arange(periods, dtype="float64")
    close = 100.0 + 0.25 * k + (k % 7) * 0.1
    return pd.DataFrame(
        {
            "open": close - 0.2,
            "high": close + 0.6,
            "low": close - 0.7,
            "close": close,
            "volume": 1_000_000.0 + 10.0 * k,
        },
        index=index,
    )


def datetime_index(frame: pd.DataFrame) -> pd.DatetimeIndex:
    """Return ``frame.index`` narrowed to ``DatetimeIndex`` (fails the test otherwise)."""
    assert isinstance(frame.index, pd.DatetimeIndex)
    return frame.index


def make_request(**overrides: Any) -> MarketDataRequest:
    """A request matching :func:`make_prices` unless overridden."""
    fields: dict[str, Any] = {"ticker": "TEST", "start": date(2023, 1, 2), "end": None}
    fields.update(overrides)
    return MarketDataRequest(**fields)


@dataclass
class StaticProvider:
    """A minimal :class:`~src.data.provider.MarketDataProvider`: fixed frame or error.

    Deliberately not a subclass of anything, to show the contract is structural.
    """

    frame: pd.DataFrame | None = None
    error: Exception | None = None
    name: str = "static_test_provider"
    requests: list[MarketDataRequest] = field(default_factory=list)

    def fetch(self, request: MarketDataRequest) -> pd.DataFrame:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        assert self.frame is not None
        return self.frame.copy()


class _FakeTicker:
    def __init__(self, owner: FakeYahoo, symbol: str) -> None:
        self._owner = owner
        self._symbol = symbol

    def history(self, **kwargs: Any) -> Any:
        owner = self._owner
        owner.calls.append({"symbol": self._symbol, **kwargs})
        if kwargs.get("raise_errors"):
            # yfinance 1.x emits this on every raise_errors=True call.
            warnings.warn(
                "'raise_errors' deprecated, do: yf.config.debug.hide_exceptions = False",
                DeprecationWarning,
                stacklevel=2,
            )
        if owner.error is not None:
            raise owner.error
        if owner.override is not _NO_OVERRIDE:
            return owner.override
        stored = owner.frames.get(self._symbol)
        if stored is None:
            raise yf_exceptions.YFTzMissingError(self._symbol)

        # Documented yfinance semantics: start inclusive, end exclusive, end=None -> now.
        keep = stored.index >= pd.Timestamp(kwargs["start"])
        if kwargs.get("end") is not None:
            keep &= stored.index < pd.Timestamp(kwargs["end"])
        window = stored.loc[keep]
        if not kwargs.get("keepna", False):
            window = window.dropna(how="all")
        if window.empty:
            if kwargs.get("raise_errors"):
                raise yf_exceptions.YFPricesMissingError(self._symbol, "(1d window)")
            return pd.DataFrame()
        return self._vendor_frame(window, kwargs)

    def _vendor_frame(self, window: pd.DataFrame, kwargs: dict[str, Any]) -> pd.DataFrame:
        owner = self._owner
        index = datetime_index(window).tz_localize(owner.tz).rename("Date")
        adjusted = kwargs.get("auto_adjust", True)
        scale = 1.0 if adjusted else UNADJUSTED_FACTOR
        vendor = pd.DataFrame(index=index)
        for name in ("open", "high", "low", "close"):
            vendor[name.capitalize()] = window[name].to_numpy() * scale
        if not adjusted:
            vendor["Adj Close"] = window["close"].to_numpy()
        vendor["Volume"] = window["volume"].fillna(0).astype("int64").to_numpy()
        if kwargs.get("actions", True) or owner.always_include_actions:
            vendor["Dividends"] = 0.0
            vendor["Stock Splits"] = 0.0
            vendor["Capital Gains"] = 0.0
        return vendor


class FakeYahoo:
    """Callable standing in for ``yfinance.Ticker``, faithful to its documented behaviour.

    It returns Yahoo-shaped frames (capitalised columns, timezone-aware
    exchange-local midnight index named ``Date``), applies inclusive-start /
    exclusive-end filtering, honours ``auto_adjust``/``actions``/``keepna``, raises
    yfinance's own exception types, and records every ``history`` call.
    """

    def __init__(
        self,
        frames: dict[str, pd.DataFrame] | None = None,
        *,
        tz: str = "America/New_York",
        always_include_actions: bool = False,
    ) -> None:
        self.frames: dict[str, pd.DataFrame] = {} if frames is None else frames
        self.tz = tz
        self.always_include_actions = always_include_actions
        self.calls: list[dict[str, Any]] = []
        self.error: BaseException | None = None
        self.override: Any = _NO_OVERRIDE  # set to return an arbitrary object verbatim

    def __call__(self, symbol: str) -> _FakeTicker:
        return _FakeTicker(self, symbol)

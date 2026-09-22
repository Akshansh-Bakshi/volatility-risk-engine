"""Preprocessing of validated market data.

Responsibility: turn a validated price series into the inputs of the statistical and
modelling layers.  Currently that is one step: authoritative prices -> log returns ->
:class:`ReturnSeries` (see :mod:`src.preprocessing.returns` for the rules and
:mod:`src.preprocessing.return_series` for the scale conventions).  Any
transformation that could leak future information must be causal by design.

The :class:`ReturnSeries` carries no reference to prices, providers or ``MarketData``:
downstream layers depend on it alone.
"""

from src.preprocessing.return_series import (
    MODELING_SCALE,
    STORED_SCALE,
    PriceBasis,
    ReturnScale,
    ReturnSeries,
)
from src.preprocessing.returns import build_return_series

__all__ = [
    "MODELING_SCALE",
    "STORED_SCALE",
    "PriceBasis",
    "ReturnScale",
    "ReturnSeries",
    "build_return_series",
]

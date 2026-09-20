"""Preprocessing of raw market data.

Responsibility: clean and align price series and derive return series.  Any
transformation that could leak future information must be causal by design.
"""

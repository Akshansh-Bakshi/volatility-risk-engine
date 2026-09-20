# Volatility Analytics & Market Risk Engine

A modular, research-grade market risk system built in Python. Its goal is to take
market data, model how volatility evolves, forecast future risk, translate those
forecasts into Value-at-Risk (VaR), test them rigorously out of sample, and present
the results in a professional Streamlit application.

> **Status: stage 2 of the build: market data ingestion.** The system can obtain,
> validate and cache daily price history. It does **not** yet compute returns,
> diagnostics, volatility models, forecasts, VaR or backtests, and the Streamlit app
> is still a shell. See [Current implementation status](#current-implementation-status).

## Research problem

Given the price history of a traded asset or index, can we produce **out-of-sample
forecasts of conditional volatility** and convert them into **Value-at-Risk estimates
that are demonstrably well calibrated**, using only information that was available
at each forecast date?

"Well calibrated" is the operative requirement. A 99% one-day VaR should be exceeded
on roughly 1% of days, and those exceedances should not cluster. Whether a model
achieves that is an empirical question, and this project is built to answer it with a
clean, leakage-free backtest rather than an in-sample fit.

## Why model volatility instead of price direction?

- **Direction is close to unpredictable; scale is not.** Daily returns show little
  exploitable serial correlation in their mean, so directional forecasts have a very
  low signal-to-noise ratio and are notoriously prone to overfitting and look-ahead
  bias. Volatility, by contrast, is one of the most robust regularities in financial
  data.
- **Volatility has structure.** Large moves tend to follow large moves and calm
  periods follow calm ones (*volatility clustering*), volatility is persistent and
  mean-reverting, falls are often followed by higher volatility than equally sized
  rises (the *leverage effect*), and return distributions have fat tails. The ARCH /
  GARCH family of models was developed to capture exactly this.
- **Risk management needs the distribution's scale, not its direction.** VaR is a
  quantile of the return distribution; its size is driven by conditional volatility
  and tail shape.
- **It is testable.** Risk forecasts make falsifiable predictions (exceedance
  frequency and independence) that can be evaluated statistically out of sample.

This project therefore contains **no machine-learning price prediction**.

## Planned pipeline

```
market data → returns → statistical diagnostics → volatility models
            → forecasting → VaR → backtesting → dashboard
```

| Stage | Package | Responsibility |
| --- | --- | --- |
| Market data | `src/data` | Fetch, validate and cache raw price data from external sources |
| Returns | `src/preprocessing` | Clean and align prices; derive return series |
| Statistical diagnostics | `src/statistics` | Characterise returns before modelling (distribution, autocorrelation, ARCH effects) |
| Volatility models | `src/models` | Specify, estimate and select conditional volatility models behind a common interface |
| Forecasting | `src/forecasting` | Produce forecasts using only information available at the forecast origin |
| VaR | `src/risk` | Turn forecasts into risk measures at configurable confidence levels |
| Backtesting | `src/backtesting` | Rolling / expanding-window re-estimation and statistical evaluation against realised outcomes |
| Dashboard | `app.py` | Streamlit interface over the packages above |

## Current implementation status

| Area | Status |
| --- | --- |
| Repository structure, layered package skeleton, dependency-rule tests | Implemented |
| Validated, environment-overridable configuration (`src/config.py`) | Implemented |
| Logging (`src/logging_config.py`) and exception hierarchy (`src/exceptions.py`) | Implemented |
| **Market data ingestion (`src/data`)**: provider contract, Yahoo Finance provider, validation, local cache, loader | **Implemented, verified offline** |
| Live verification against real Yahoo Finance | Opt-in tests provided (`pytest -m live`); **not yet run against Yahoo** in the development environment because outbound access was blocked |
| Streamlit entry point | Application shell only (shows the active configuration); no data or analytics pages |
| Returns and preprocessing | Not started |
| Statistical diagnostics | Not started |
| Volatility models (GARCH family) | Not started |
| Forecasting | Not started |
| Value-at-Risk | Not started |
| Backtesting | Not started |
| Dashboard | Not started |

Nothing in the repository forecasts volatility or computes risk yet.

## Market data layer (`src/data`)

### Data source

Daily prices come from Yahoo Finance through the open-source
[`yfinance`](https://github.com/ranaroussi/yfinance) library, an unofficial,
community-maintained wrapper around Yahoo's public endpoints. Availability, data
quality and terms of use are Yahoo's, not this project's; treat the data as suitable
for research and education. `yfinance` is imported in exactly one module
(`src/data/yahoo.py`), and a test enforces that.

### Architecture

```
MarketDataRequest ──► MarketDataLoader ──► MarketData
   (what)               │   ▲                (validated frame + provenance)
                        │   └── MarketDataCache   (validated JSON entries, freshness rules)
                        ▼
                MarketDataProvider  (Protocol: the contract)
                        ▲
                YahooFinanceProvider  (only module importing yfinance)
```

| Module | Role |
| --- | --- |
| `market_data.py` | `MarketDataRequest`, `MarketData`, schema constants: the domain types |
| `provider.py` | `MarketDataProvider`, a small structural `Protocol`; new sources implement it and nothing else changes |
| `yahoo.py` | Yahoo Finance implementation: request translation, option pinning, structural normalisation, error translation |
| `validation.py` | Provider-independent validation applied to every frame, downloaded or cached |
| `cache.py` | Local JSON cache with deterministic naming and freshness rules |
| `loader.py` | `MarketDataLoader`: cache, then provider, then validation, plus logging |
| `factory.py` | Composition root: builds a loader from `Settings` |

Usage:

```python
from src.config import get_settings
from src.data import MarketDataRequest
from src.data.factory import create_market_data_loader

settings = get_settings()
loader = create_market_data_loader(settings)
data = loader.load(MarketDataRequest.from_config(settings.data))

data.price        # the authoritative price series (pandas Series)
data.prices       # open/high/low/close/volume DataFrame
data.fetched_at, data.from_cache, data.rows_dropped, data.source
```

Application code depends on the `MarketDataProvider` contract and the domain types,
never on `yfinance`.

### The authoritative price series

**Decision:** the modelling series is the `close` column requested with
`auto_adjust=True`: **adjusted for splits and dividends (a total-return basis)**.
`close` is the only column later stages may use to compute returns; `open`, `high`,
`low` and `volume` are preserved as delivered, on the same adjustment basis, for
range-based analysis.

**Why.** The risk measured here is the risk of *holding* the asset. On an ex-dividend
date the price falls by roughly the dividend, but the holder is not poorer, because
they receive the cash. An unadjusted series records that fall as a negative return,
which biases returns downward and adds small, systematic, calendar-driven shocks to
every dividend payer, most visibly for high-yield assets. The adjusted series is the
one whose returns equal the holder's returns.

**What the choice does *not* concern.** Yahoo's plain `Close` is already adjusted for
splits, so the two settings differ only in dividends. (This is Yahoo's documented
column definition; the opt-in live tests include a check against a real split.)

**Consequences to keep in mind.**

- Adjusted history is *restated* whenever a new dividend is paid: every earlier price
  is rescaled by a constant. Returns between two dates before the new ex-date do not
  change, but **absolute price levels do**. Do not use adjusted `close` where actual
  traded price levels matter (for example currency-denominated position values).
- Indices such as `^GSPC` are price indices; they have no dividend adjustment
  (adjusted and unadjusted are identical), so their returns exclude dividends. Total
  return index symbols exist separately (for example `^SP500TR`).
- The basis is explicit and selectable: `adjust_prices` in the request, `VRE_ADJUST_PRICES`
  in configuration. `False` requests split-adjusted prices only. The setting is part of
  the cache identity, so the two bases never mix.

### Date semantics

- `start` is **inclusive**, and `end` is **inclusive** in this project's API.
  Yahoo's `end` is exclusive, so the Yahoo provider sends `end + 1 day`; without that
  shift the final requested session would be silently lost.
- `end=None` means "the latest available session". It is resolved **at fetch time**:
  the provider omits the end bound and `yfinance` resolves "now" when the call is made.
  No date is computed at import time or stored in configuration.
- Rows outside the requested range indicate a fault; the validator rejects such output
  instead of trimming it, which would also expose a `yfinance` change in end-date
  semantics.
- A "latest" download made while a market is open may end with a *provisional*
  current-session bar. For reproducible research, pin `end_date` to a past date.

### Index and timezone convention

The index is a `DatetimeIndex` named `date` that is **timezone-naive, midnight-normalised,
strictly increasing and unique**. Each label is the *exchange-local session date* of a
daily bar, not an instant. Yahoo delivers timezone-aware exchange-local timestamps; the
provider *drops* the timezone (it never converts to UTC, which would move an Asian
session to the previous calendar day). Naive session dates keep series from exchanges in
different timezones alignable by calendar date and avoid DST artefacts. The resolution is
always `datetime64[ns]`, and columns are always `float64`.

Only daily bars (`1d`) are supported for now. The rest of the pipeline (returns,
annualisation, daily VaR) assumes one observation per session; `SUPPORTED_INTERVALS`
widens only together with that code.

### Validation policy

Every frame passes `validate_prices`, whether downloaded or read from the cache. The
policy is strict and has exactly one repair.

| Condition | Action | Exception |
| --- | --- | --- |
| Not a DataFrame; MultiIndex, missing, extra or duplicated columns; non-numeric columns | Reject | `DataValidationError` |
| Index is not a `DatetimeIndex`, is tz-aware, has NaT, or has time-of-day components | Reject | `DataValidationError` |
| Duplicate timestamps | Reject (never de-duplicated) | `DataValidationError` |
| Timestamps not sorted ascending | Reject (never sorted) | `DataValidationError` |
| Observations outside the requested date range | Reject | `DataValidationError` |
| Infinite values | Reject | `DataValidationError` |
| Non-positive `close` | Reject | `DataValidationError` |
| No rows | Reject | `EmptyDataError` |
| Every `close` missing (no usable observations) | Reject | `EmptyDataError` |
| Fewer usable observations than `min_observations` | Reject | `InsufficientHistoryError` |
| Unknown symbol (when Yahoo signals it) | Reject | `InvalidTickerError` |
| Provider unreachable, rate-limited or failing | Reject | `DataFetchError` |
| A row whose `close` is missing | **Remove**, count, log with dates | none (`MarketData.rows_dropped`) |
| Missing `open`/`high`/`low`/`volume` on a row that has a close | Keep as delivered | none |

Why removing (and never forward-filling) rows without a close is safe: such a row carries
no price information, so removing it invents nothing, whereas forward-filling would
fabricate observations that later appear as artificial zero returns and bias volatility
downwards. The consequence for return construction is that the return across a removed
row spans more than one session; that is visible through `rows_dropped` and the log.

Not validated on purpose: cross-field plausibility (for example `high >= low`) and outlier
detection. Those judgements belong to later, explicitly configured stages.

`InvalidTickerError` is best-effort. `yfinance` reports a symbol whose timezone cannot be
resolved as missing, and it uses the same signal when Yahoo cannot be reached, so the error
message says so. Blocked or failed HTTP requests can surface from `yfinance` as a bare
`JSONDecodeError`; the resulting `DataFetchError` adds a connectivity hint.

### Cache policy

Downloads are cached as validated JSON under `<data_dir>/cache/market/` (default
`data/cache/market/`, git-ignored). It is a local research cache, nothing more.

- **Identity and naming.** One file per request identity, named from ticker (percent-encoded),
  interval, start, end (or `latest`) and price basis, for example
  `%5EGSPC__1d__2005-01-01__latest__adjusted.json`. The identity is also stored in the file
  and checked on read; a renamed or copied file never answers the wrong request.
- **Only validated data is stored**, with source, download time and `rows_dropped`, and it is
  re-validated on every read. A corrupt, mismatched, or no-longer-valid entry is logged,
  ignored and replaced by a fresh download.
- **Freshness.** An entry with an explicit `end` that had clearly passed (more than one day)
  when it was downloaded is *settled* and never expires, which suits reproducible research and
  offline work. Everything else, including every `end=None` request, expires after
  `VRE_CACHE_MAX_AGE_HOURS` (default 6). Stale data therefore cannot pass as current, and
  `MarketData.fetched_at` / `from_cache` always show what you are looking at.
- **Bypass and refresh.** `VRE_USE_CACHE=false` (or a loader built without a cache) never reads
  or writes the cache; `loader.load(request, refresh=True)` skips the read and overwrites the entry.
- A failed cache write is logged and does not fail the load. Files are written atomically.

### Assets

The API is generic: any Yahoo symbol works, and `ticker` is the only asset-specific input.
The default development ticker is `^GSPC` (S&P 500 index: long, liquid history and no corporate
actions). Examples:

| Kind | Example | Notes |
| --- | --- | --- |
| Equity index | `^GSPC`, `^NSEI`, `^N225` | Price indices; no dividend adjustment; volume is often 0 |
| ETF | `SPY`, `QQQ` | Dividend-adjusted `close` differs from the raw close |
| Equity | `AAPL`, `RELIANCE.NS` | Splits and dividends; exchange-local session dates |
| Crypto | `BTC-USD` | Trades every calendar day (about 365 observations a year, day boundaries in UTC) |
| FX | `EURUSD=X` | Volume is not meaningful |

Not solved yet: asset-class metadata, trading calendars and per-asset annualisation factors.
Nothing in the API prevents adding them, and `MarketData` already carries the full request.

### Known limitations

Yahoo data is unofficial and can be revised or unavailable; there are no retries or fallback
providers; one ticker per request; daily bars only; the latest bar can be provisional
during a live session.

## Architecture

### Layout

```
volatility-risk-engine/
├── app.py                  # Streamlit entry point
├── requirements.txt
├── pytest.ini
├── .env.example            # Template of supported VRE_* environment variables
├── src/
│   ├── config.py           # Central validated configuration
│   ├── logging_config.py   # Reusable logging setup
│   ├── exceptions.py       # Project exception hierarchy
│   ├── data/               # Market data ingestion (implemented)
│   │   ├── market_data.py  #   domain types and schema constants
│   │   ├── provider.py     #   MarketDataProvider contract
│   │   ├── yahoo.py        #   Yahoo Finance provider (only yfinance import)
│   │   ├── validation.py   #   provider-independent validation
│   │   ├── cache.py        #   local cache and freshness rules
│   │   ├── loader.py       #   MarketDataLoader
│   │   └── factory.py      #   composition root
│   ├── preprocessing/      # Cleaning and return construction
│   ├── statistics/         # Statistical diagnostics
│   ├── models/             # Volatility models
│   ├── forecasting/        # Volatility forecasts
│   ├── risk/               # VaR and related measures
│   ├── backtesting/        # Out-of-sample evaluation
│   └── utils/              # Small shared helpers
├── tests/
├── data/                   # Generated datasets and cache (git-ignored, created on demand)
└── notebooks/              # Exploration only; never the application architecture
```

### Dependency rule

Dependencies point one way, down the pipeline. Each layer may import from itself and
from layers below it, never from layers above:

```
config · logging_config · exceptions · utils      (foundation)
  ↑ data
    ↑ preprocessing
      ↑ statistics
        ↑ models
          ↑ forecasting
            ↑ risk
              ↑ backtesting
                ↑ app.py
```

This keeps each stage replaceable and makes look-ahead paths (for example, data
ingestion reaching into a model) structurally impossible. The rule is enforced by
`tests/test_architecture.py`, which also verifies that `yfinance` is imported only by
`src/data/yahoo.py` and that only the composition root imports that module.

### How the system will evolve

Development proceeds in small, incremental stages, each adding one capability behind a
stable interface and leaving earlier layers untouched:

1. **Foundation**: structure, configuration, logging, tests, app shell. *(done)*
2. **Data**: ingestion, validation and caching of market data. *(done)*
3. **Preprocessing and diagnostics**: returns and statistical characterisation.
4. **Modelling**: a common volatility model interface, then concrete model families
   and model selection.
5. **Forecasting and risk**: leakage-free forecasts, then VaR.
6. **Backtesting**: out-of-sample evaluation and formal coverage tests.
7. **Dashboard**: the Streamlit application on top of the completed pipeline.

### Design principles

- **No look-ahead bias by design:** anything computed for date *t* may only use
  information available up to *t*.
- **Reproducibility:** configuration is explicit and immutable; behaviour does not
  depend on hidden global state or on the day the process started.
- **Fail loudly and informatively:** validated inputs, a project exception hierarchy
  rooted in `VolatilityRiskEngineError`, and logs with module, function and line.
- **Generated artifacts stay out of git** (`data/`, `artifacts/`, `logs/`, ...).

## Setup

Requires **Python 3.10 or newer** (the test suite has been run on 3.10 and 3.12).

```bash
git clone <repository-url> volatility-risk-engine
cd volatility-risk-engine

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Configuration

Defaults live in `src/config.py` and can be overridden with environment variables
(all prefixed `VRE_`). `.env.example` documents them; the application reads the
process environment and does not load `.env` automatically, so export the values
first (for example `set -a && source .env && set +a`) or set them in your IDE.
Blank values are treated as unset. No secrets are required or stored.

| Variable | Default | Meaning |
| --- | --- | --- |
| `VRE_DEFAULT_TICKER` | `^GSPC` | Data-provider symbol requested by default |
| `VRE_DEFAULT_START_DATE` | `2005-01-01` | Default start of the date range (`YYYY-MM-DD`, inclusive) |
| `VRE_DEFAULT_END_DATE` | *(unset)* | Default end date (inclusive); unset means "latest available", resolved at fetch time |
| `VRE_CONFIDENCE_LEVELS` | `0.95,0.99` | Comma-separated VaR confidence levels in (0, 1) |
| `VRE_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL` |
| `VRE_LOG_FILE` | *(unset)* | Optional rotating log file, e.g. `logs/engine.log` |
| `VRE_DATA_DIR` | `<repo>/data` | Root for generated datasets; the cache lives in `<dir>/cache/market/` |
| `VRE_ADJUST_PRICES` | `true` | `true`: adjusted for splits and dividends (total-return basis); `false`: split-adjusted only |
| `VRE_MIN_OBSERVATIONS` | `250` | Minimum usable price observations (integer, at least 2) for a download to be accepted |
| `VRE_USE_CACHE` | `true` | `false` never reads or writes the on-disk cache |
| `VRE_CACHE_MAX_AGE_HOURS` | `6` | How long a "latest" download counts as current (positive number of hours) |

Invalid values fail fast with a `ConfigurationError` that names the offending setting.

## Running the tests

```bash
python -m pytest              # deterministic suite; Yahoo Finance is never contacted
python -m pytest -m live      # opt-in checks against the real Yahoo Finance (needs network access)
```

The normal suite replaces the `yfinance` boundary with a fake that follows its documented
behaviour, so it runs offline. The `live` tests are deselected by default and fail (rather
than skip) when Yahoo cannot be reached.

## Launching the application

```bash
streamlit run app.py
```

At this stage the app is a shell that shows the active default configuration; the
data and analytics pages arrive in later stages.

## Disclaimer

This is a research and educational project. Its outputs are not investment advice and
must not be used as the sole basis for trading or risk-capital decisions.

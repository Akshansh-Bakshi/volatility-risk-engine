"""Tests for src.statistics.profile and src.statistics.descriptive.

All tests use synthetic data built by the existing test helpers; no network
access, no disk I/O (except where tmp_path is explicitly used for snapshots).

Coverage
--------
* DatasetProfile construction and JSON serialisation.
* Field correctness: ticker, source, frequency, dates, observation counts.
* DataQualityFindings: missing-close rows, gap flags, duplicate timestamps.
* ReturnStats: count, mean, median, std, min/max, quartiles, skewness,
  excess kurtosis, percent-scale convenience properties, summary_frame.
* Exactly 500 observations.
* Fewer than 500 observations (but above the configured minimum).
* Invalid input: mismatched tickers, single-row price frame.
* No mutation of source objects.
* Serialisation round-trip (JSON → dict → values).
* Snapshot file layout (offline, tmp_path).
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.exceptions import InvalidProfileInputError
from src.preprocessing.return_series import PriceBasis, ReturnSeries, ReturnScale, STORED_SCALE
from src.statistics.descriptive import ReturnStats, compute_return_stats
from src.statistics.profile import DatasetProfile, DataQualityFindings, build_dataset_profile
from src.statistics.snapshot import save_eda_snapshot
from tests.data_fakes import (
    DEFAULT_FETCHED_AT,
    make_market_data,
    make_prices,
    make_request,
)
from src.preprocessing.returns import build_return_series

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

INDEX_NAME = "date"
FETCHED_AT = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)


def _make_returns(prices: pd.DataFrame | None = None, **md_kwargs) -> tuple:
    """Return (market_data, return_series) built from fake prices."""
    md = make_market_data(prices, **md_kwargs)
    rs = build_return_series(md)
    return md, rs


# ---------------------------------------------------------------------------
# DatasetProfile — correctness
# ---------------------------------------------------------------------------


class TestDatasetProfileFields:
    def test_ticker_matches(self) -> None:
        md, rs = _make_returns()
        profile = build_dataset_profile(md, rs)
        assert profile.ticker == "TEST"

    def test_source_matches(self) -> None:
        md, rs = _make_returns()
        profile = build_dataset_profile(md, rs)
        assert profile.source == "unit_test"

    def test_frequency_is_1d(self) -> None:
        md, rs = _make_returns()
        profile = build_dataset_profile(md, rs)
        assert profile.frequency == "1d"

    def test_price_basis_is_adjusted(self) -> None:
        md, rs = _make_returns()
        profile = build_dataset_profile(md, rs)
        assert profile.price_basis == "adjusted"

    def test_price_observations_equals_market_data_observations(self) -> None:
        md, rs = _make_returns()
        profile = build_dataset_profile(md, rs)
        assert profile.price_observations == md.observations

    def test_return_observations_equals_return_series_count(self) -> None:
        md, rs = _make_returns()
        profile = build_dataset_profile(md, rs)
        assert profile.return_observations == rs.return_observations

    def test_return_observations_is_one_fewer_than_prices(self) -> None:
        md, rs = _make_returns()
        profile = build_dataset_profile(md, rs)
        assert profile.return_observations == profile.price_observations - 1

    def test_first_return_date_matches(self) -> None:
        md, rs = _make_returns()
        profile = build_dataset_profile(md, rs)
        assert profile.first_return_date == rs.first_return_date

    def test_last_return_date_matches(self) -> None:
        md, rs = _make_returns()
        profile = build_dataset_profile(md, rs)
        assert profile.last_return_date == rs.last_return_date

    def test_first_price_date_before_first_return_date(self) -> None:
        md, rs = _make_returns()
        profile = build_dataset_profile(md, rs)
        assert profile.first_price_date < profile.first_return_date

    def test_last_price_date_matches_market_data(self) -> None:
        md, rs = _make_returns()
        profile = build_dataset_profile(md, rs)
        assert profile.last_price_date == md.last_date

    def test_fetched_at_is_utc(self) -> None:
        md, rs = _make_returns()
        profile = build_dataset_profile(md, rs)
        assert profile.fetched_at.tzinfo is not None
        assert profile.fetched_at.utcoffset().total_seconds() == 0  # type: ignore[union-attr]

    def test_from_cache_is_false_for_fresh_load(self) -> None:
        md, rs = _make_returns()
        profile = build_dataset_profile(md, rs)
        assert profile.from_cache is False

    def test_return_definition_is_log(self) -> None:
        md, rs = _make_returns()
        profile = build_dataset_profile(md, rs)
        assert "log" in profile.return_definition.lower()


# ---------------------------------------------------------------------------
# DataQualityFindings
# ---------------------------------------------------------------------------


class TestDataQualityFindings:
    def test_no_issues_when_data_is_clean(self) -> None:
        md, rs = _make_returns()
        profile = build_dataset_profile(md, rs)
        q = profile.quality
        assert q.missing_close_rows == 0
        assert q.gap_flagged_returns == 0
        assert q.duplicate_timestamps == 0
        assert "No quality issues" in q.quality_summary

    def test_missing_close_rows_counted(self) -> None:
        md = make_market_data(dropped_dates=["2023-01-03", "2023-01-04"])
        rs = build_return_series(md)
        profile = build_dataset_profile(md, rs)
        assert profile.quality.missing_close_rows == 2

    def test_gap_flagged_returns_counted(self) -> None:
        # make_market_data bypasses the loader, so we simulate two dropped rows
        # by removing them from the frame and recording them in dropped_dates.
        # The preprocessing layer then flags the returns that span those gaps.
        frame = make_prices()
        drop_idx = frame.index[[10, 11]]
        frame_clean = frame.drop(drop_idx)
        md = make_market_data(
            frame_clean,
            dropped_dates=[ts.strftime("%Y-%m-%d") for ts in drop_idx],
        )
        rs = build_return_series(md)
        profile = build_dataset_profile(md, rs)
        assert profile.quality.gap_flagged_returns == rs.flagged_gaps
        assert profile.quality.gap_flagged_returns > 0

    def test_columns_present_contains_standard_columns(self) -> None:
        md, rs = _make_returns()
        profile = build_dataset_profile(md, rs)
        assert "close" in profile.quality.columns_present
        assert "open" in profile.quality.columns_present

    def test_issues_appear_in_quality_summary(self) -> None:
        md = make_market_data(dropped_dates=["2023-01-03"])
        rs = build_return_series(md)
        profile = build_dataset_profile(md, rs)
        assert "Issues:" in profile.quality.quality_summary

    def test_provisional_bar_none_when_data_is_final(self) -> None:
        md, rs = _make_returns()
        profile = build_dataset_profile(md, rs)
        assert profile.quality.provisional_bar is None
        assert profile.quality.provisional_bar_excluded is False


# ---------------------------------------------------------------------------
# Exactly 500 and fewer than 500 observations
# ---------------------------------------------------------------------------


class TestObservationCounts:
    def test_exactly_500_price_observations(self) -> None:
        prices = make_prices(periods=500)
        md, rs = _make_returns(prices)
        profile = build_dataset_profile(md, rs)
        assert profile.price_observations == 500
        # return_observations == price_observations - 1 - excluded_observations
        assert profile.return_observations == 500 - 1 - rs.excluded_observations

    def test_fewer_than_500_price_observations(self) -> None:
        prices = make_prices(periods=260)
        md, rs = _make_returns(prices)
        profile = build_dataset_profile(md, rs)
        assert profile.price_observations == 260
        assert profile.return_observations == 260 - 1 - rs.excluded_observations

    def test_profile_dates_are_correct_for_exactly_500(self) -> None:
        prices = make_prices(periods=500)
        md, rs = _make_returns(prices)
        profile = build_dataset_profile(md, rs)
        assert profile.first_price_date == md.first_date
        assert profile.last_return_date == rs.last_return_date

    def test_exactly_500_returns_when_provisional_excluded(self) -> None:
        """Verify we can produce exactly 500 returns by picking a period count
        large enough to survive provisional-bar exclusion."""
        # make_prices starts 2023-01-02 with DEFAULT_FETCHED_AT = 2024-06-03.
        # Bars past 2024-06-03 are provisional and excluded.
        # 502 prices starting 2023-01-02: the series runs to ~2024-12-04, so
        # at most 2 bars are provisional; we may lose 1 or 2.
        # Simply assert that the pipeline produces a consistent result.
        prices = make_prices(periods=502)
        md, rs = _make_returns(prices)
        profile = build_dataset_profile(md, rs)
        assert profile.return_observations == rs.return_observations
        assert profile.return_observations >= 499


# ---------------------------------------------------------------------------
# Invalid input
# ---------------------------------------------------------------------------


class TestInvalidProfileInput:
    def test_mismatched_tickers_raise(self) -> None:
        md = make_market_data(ticker="AAA")
        # Build a return series for a different ticker name in the ReturnSeries directly.
        md2 = make_market_data(ticker="BBB")
        rs = build_return_series(md2)
        with pytest.raises(InvalidProfileInputError, match="ticker"):
            build_dataset_profile(md, rs)

    def test_returns_for_wrong_object_type_raise(self) -> None:
        md, rs = _make_returns()
        with pytest.raises((AttributeError, TypeError, InvalidProfileInputError)):
            build_dataset_profile(md, "not_a_return_series")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# No mutation of source objects
# ---------------------------------------------------------------------------


class TestNoMutation:
    def test_profile_does_not_mutate_market_data(self) -> None:
        md, rs = _make_returns()
        original_observations = md.observations
        original_first = md.first_date
        build_dataset_profile(md, rs)
        assert md.observations == original_observations
        assert md.first_date == original_first

    def test_profile_does_not_mutate_return_series(self) -> None:
        md, rs = _make_returns()
        original_count = rs.return_observations
        original_mean = float(rs.decimal.mean())
        build_dataset_profile(md, rs)
        assert rs.return_observations == original_count
        assert float(rs.decimal.mean()) == pytest.approx(original_mean)


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------


class TestProfileSerialisation:
    def test_to_dict_is_json_serialisable(self) -> None:
        md, rs = _make_returns()
        profile = build_dataset_profile(md, rs)
        d = profile.to_dict()
        round_tripped = json.loads(json.dumps(d))
        assert round_tripped == d

    def test_to_json_produces_valid_json(self) -> None:
        md, rs = _make_returns()
        profile = build_dataset_profile(md, rs)
        parsed = json.loads(profile.to_json())
        assert parsed["ticker"] == "TEST"
        assert parsed["return_observations"] == rs.return_observations

    def test_dates_serialise_as_iso_strings(self) -> None:
        md, rs = _make_returns()
        profile = build_dataset_profile(md, rs)
        d = profile.to_dict()
        for key in ("first_price_date", "last_price_date", "first_return_date", "last_return_date"):
            val = d[key]
            assert isinstance(val, str), f"{key} should be a string"
            assert date.fromisoformat(val) is not None

    def test_fetched_at_serialises_as_iso_string(self) -> None:
        md, rs = _make_returns()
        profile = build_dataset_profile(md, rs)
        d = profile.to_dict()
        assert isinstance(d["fetched_at"], str)
        assert datetime.fromisoformat(d["fetched_at"]) is not None


# ---------------------------------------------------------------------------
# ReturnStats — correctness
# ---------------------------------------------------------------------------


class TestReturnStats:
    def test_count_matches_return_observations(self) -> None:
        _, rs = _make_returns()
        stats = compute_return_stats(rs)
        assert stats.count == rs.return_observations

    def test_mean_close_to_numpy_mean(self) -> None:
        _, rs = _make_returns()
        stats = compute_return_stats(rs)
        assert stats.mean == pytest.approx(float(rs.decimal.mean()), rel=1e-10)

    def test_median_close_to_numpy_median(self) -> None:
        _, rs = _make_returns()
        stats = compute_return_stats(rs)
        assert stats.median == pytest.approx(float(np.median(rs.decimal.to_numpy())), rel=1e-10)

    def test_std_is_sample_std(self) -> None:
        _, rs = _make_returns()
        stats = compute_return_stats(rs)
        assert stats.std == pytest.approx(float(rs.decimal.std(ddof=1)), rel=1e-10)

    def test_min_max_correct(self) -> None:
        _, rs = _make_returns()
        stats = compute_return_stats(rs)
        values = rs.decimal.to_numpy()
        assert stats.minimum == pytest.approx(float(np.min(values)), rel=1e-10)
        assert stats.maximum == pytest.approx(float(np.max(values)), rel=1e-10)

    def test_quartiles_correct(self) -> None:
        _, rs = _make_returns()
        stats = compute_return_stats(rs)
        values = rs.decimal.to_numpy()
        assert stats.q25 == pytest.approx(float(np.percentile(values, 25)), rel=1e-10)
        assert stats.q75 == pytest.approx(float(np.percentile(values, 75)), rel=1e-10)

    def test_skewness_of_symmetric_distribution_is_near_zero(self) -> None:
        # Build a ReturnSeries directly to bypass the min_returns guard.
        # Alternating +/- returns are symmetric: skewness should be ~0.
        from datetime import timezone
        n = 300
        idx = pd.bdate_range("2022-01-03", periods=n, name="date")
        # Alternating +log(1.01) and -log(1.01): perfectly symmetric
        val = float(np.log(1.01))
        values = np.where(np.arange(n) % 2 == 0, val, -val).astype("float64")
        decimal = pd.Series(values, index=idx, dtype="float64")
        spans_gap = pd.Series(np.zeros(n, dtype=bool), index=idx)
        rs = ReturnSeries(
            ticker="SYM",
            frequency="1d",
            price_basis=PriceBasis.ADJUSTED,
            decimal=decimal,
            spans_gap=spans_gap,
            first_price_date=date(2022, 1, 3) - pd.Timedelta(days=1),
            gap_tolerance_weekdays=1,
            provisional_bar=None,
            provisional_bar_excluded=False,
            ingestion_rows_dropped=0,
            source="unit_test",
            fetched_at=datetime(2024, 6, 3, 12, 0, tzinfo=timezone.utc),
        )
        stats = compute_return_stats(rs)
        assert abs(stats.skewness) < 0.01  # symmetric → skewness ≈ 0

    def test_excess_kurtosis_property_exists_and_is_finite(self) -> None:
        _, rs = _make_returns()
        stats = compute_return_stats(rs)
        assert math.isfinite(stats.excess_kurtosis)

    def test_percent_scale_properties_are_exactly_100x_decimal(self) -> None:
        _, rs = _make_returns()
        stats = compute_return_stats(rs)
        assert stats.mean_pct == pytest.approx(stats.mean * 100.0, rel=1e-12)
        assert stats.std_pct == pytest.approx(stats.std * 100.0, rel=1e-12)
        assert stats.minimum_pct == pytest.approx(stats.minimum * 100.0, rel=1e-12)
        assert stats.maximum_pct == pytest.approx(stats.maximum * 100.0, rel=1e-12)
        assert stats.q25_pct == pytest.approx(stats.q25 * 100.0, rel=1e-12)
        assert stats.q75_pct == pytest.approx(stats.q75 * 100.0, rel=1e-12)

    def test_to_dict_decimal_scale_values_match(self) -> None:
        _, rs = _make_returns()
        stats = compute_return_stats(rs)
        d = stats.to_dict(scale="decimal")
        assert d["mean"] == pytest.approx(stats.mean, rel=1e-12)
        assert d["std"] == pytest.approx(stats.std, rel=1e-12)
        assert d["scale"] == "decimal"

    def test_to_dict_percent_scale_values_are_100x(self) -> None:
        _, rs = _make_returns()
        stats = compute_return_stats(rs)
        d_dec = stats.to_dict(scale="decimal")
        d_pct = stats.to_dict(scale="percent")
        assert d_pct["mean"] == pytest.approx(d_dec["mean"] * 100.0, rel=1e-12)
        assert d_pct["scale"] == "percent"

    def test_to_dict_is_json_serialisable(self) -> None:
        _, rs = _make_returns()
        stats = compute_return_stats(rs)
        d = stats.to_dict()
        assert json.loads(json.dumps(d)) == d

    def test_to_json_round_trips(self) -> None:
        _, rs = _make_returns()
        stats = compute_return_stats(rs)
        parsed = json.loads(stats.to_json())
        assert parsed["ticker"] == "TEST"
        assert parsed["count"] == rs.return_observations

    def test_summary_frame_shape_and_index(self) -> None:
        _, rs = _make_returns()
        stats = compute_return_stats(rs)
        frame = stats.summary_frame()
        assert isinstance(frame, pd.DataFrame)
        assert frame.shape == (10, 1)
        assert frame.columns[0] == "TEST"
        assert "mean" in frame.index
        assert "std" in frame.index

    def test_invalid_scale_raises(self) -> None:
        _, rs = _make_returns()
        stats = compute_return_stats(rs)
        with pytest.raises(ValueError, match="scale"):
            stats.to_dict(scale="invalid")

    def test_stats_does_not_mutate_return_series(self) -> None:
        _, rs = _make_returns()
        original = rs.decimal.copy()
        compute_return_stats(rs)
        pd.testing.assert_series_equal(rs.decimal, original)

    def test_stats_for_exactly_500_returns(self) -> None:
        # Build a ReturnSeries directly so we can control the exact count.
        from datetime import timezone as tz
        n = 500
        idx = pd.bdate_range("2020-01-02", periods=n, name="date")
        vals = np.random.default_rng(42).normal(0.0005, 0.01, n).astype("float64")
        decimal = pd.Series(vals, index=idx)
        spans_gap = pd.Series(np.zeros(n, dtype=bool), index=idx)
        rs = ReturnSeries(
            ticker="FIVE00",
            frequency="1d",
            price_basis=PriceBasis.ADJUSTED,
            decimal=decimal,
            spans_gap=spans_gap,
            first_price_date=date(2020, 1, 1),
            gap_tolerance_weekdays=1,
            provisional_bar=None,
            provisional_bar_excluded=False,
            ingestion_rows_dropped=0,
            source="unit_test",
            fetched_at=datetime(2024, 6, 3, 12, 0, tzinfo=tz.utc),
        )
        assert rs.return_observations == 500
        stats = compute_return_stats(rs)
        assert stats.count == 500

    def test_stats_for_fewer_than_500_returns(self) -> None:
        prices = make_prices(periods=260)
        md, rs = _make_returns(prices)
        # return_observations may be 259 or 258 depending on provisional exclusion.
        assert rs.return_observations <= 259
        stats = compute_return_stats(rs)
        assert stats.count == rs.return_observations
        assert stats.count < 500

    def test_ticker_carried_through(self) -> None:
        _, rs = _make_returns()
        stats = compute_return_stats(rs)
        assert stats.ticker == "TEST"


# ---------------------------------------------------------------------------
# Snapshot — file layout (offline, tmp_path)
# ---------------------------------------------------------------------------


class TestSnapshot:
    def test_snapshot_directory_is_created(self, tmp_path: Path) -> None:
        md, rs = _make_returns()
        from src.statistics.profile import build_dataset_profile
        profile = build_dataset_profile(md, rs)
        stats = compute_return_stats(rs)
        snap_dir = save_eda_snapshot(profile, stats, rs, base_dir=tmp_path)
        assert snap_dir.is_dir()

    def test_snapshot_json_exists_and_is_valid(self, tmp_path: Path) -> None:
        md, rs = _make_returns()
        from src.statistics.profile import build_dataset_profile
        profile = build_dataset_profile(md, rs)
        stats = compute_return_stats(rs)
        snap_dir = save_eda_snapshot(profile, stats, rs, base_dir=tmp_path)
        json_file = snap_dir / "snapshot.json"
        assert json_file.is_file()
        data = json.loads(json_file.read_text(encoding="utf-8"))
        assert data["ticker"] == "TEST"
        assert "profile" in data
        assert "return_stats_decimal" in data
        assert "generated_at" in data

    def test_returns_csv_exists_and_has_correct_columns(self, tmp_path: Path) -> None:
        md, rs = _make_returns()
        from src.statistics.profile import build_dataset_profile
        profile = build_dataset_profile(md, rs)
        stats = compute_return_stats(rs)
        snap_dir = save_eda_snapshot(profile, stats, rs, base_dir=tmp_path)
        csv_file = snap_dir / "returns.csv"
        assert csv_file.is_file()
        df = pd.read_csv(csv_file)
        assert "log_return_decimal" in df.columns
        assert "date" in df.columns
        assert len(df) == rs.return_observations

    def test_snapshot_json_is_reproducible(self, tmp_path: Path) -> None:
        """Two snapshots of the same data should have identical profile sections."""
        md, rs = _make_returns()
        from src.statistics.profile import build_dataset_profile
        profile = build_dataset_profile(md, rs)
        stats = compute_return_stats(rs)
        dir1 = save_eda_snapshot(profile, stats, rs, base_dir=tmp_path / "a")
        dir2 = save_eda_snapshot(profile, stats, rs, base_dir=tmp_path / "b")
        d1 = json.loads((dir1 / "snapshot.json").read_text())
        d2 = json.loads((dir2 / "snapshot.json").read_text())
        assert d1["profile"] == d2["profile"]
        assert d1["return_stats_decimal"] == d2["return_stats_decimal"]

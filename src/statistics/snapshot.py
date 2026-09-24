"""Synopsis snapshot: persist a reproducible EDA record to disk.

For the university synopsis phase this module saves a lightweight package of
files that records exactly what data was used and what the descriptive EDA
found.  Everything written here is **generated data** and lives under
``data/snapshots/`` which is git-ignored.

Contents of a snapshot directory
---------------------------------
``snapshot.json``
    The :class:`~src.statistics.profile.DatasetProfile` and
    :class:`~src.statistics.descriptive.ReturnStats` merged into one
    JSON envelope, together with a ``generated_at`` timestamp.

``returns.csv``
    The decimal log-return series as a two-column CSV
    (``date``, ``log_return_decimal``), suitable for offline inspection or
    import into another tool.

Both files are written atomically to a subdirectory whose name encodes the
ticker and the snapshot timestamp, so repeated runs do not overwrite each other.

Usage::

    from src.statistics.snapshot import save_eda_snapshot
    path = save_eda_snapshot(profile, stats, returns)

The returned :class:`pathlib.Path` points to the snapshot directory.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.config import get_settings
from src.logging_config import PACKAGE_LOGGER_NAME
from src.preprocessing.return_series import ReturnSeries
from src.statistics.descriptive import ReturnStats
from src.statistics.profile import DatasetProfile

logger = logging.getLogger(f"{PACKAGE_LOGGER_NAME}.statistics.snapshot")

_SNAPSHOT_SUBDIR = "snapshots"


def save_eda_snapshot(
    profile: DatasetProfile,
    stats: ReturnStats,
    returns: ReturnSeries,
    *,
    base_dir: Path | None = None,
) -> Path:
    """Write the EDA artefacts to a timestamped directory under ``data/snapshots/``.

    Args:
        profile: Dataset profile produced by
            :func:`~src.statistics.profile.build_dataset_profile`.
        stats: Descriptive statistics produced by
            :func:`~src.statistics.descriptive.compute_return_stats`.
        returns: The underlying return series (its decimal values are saved as
            CSV).
        base_dir: Root for the snapshot tree.  Defaults to the configured
            ``data_dir`` (``data/snapshots/`` inside the project root).

    Returns:
        :class:`pathlib.Path` to the snapshot directory that was created.

    Raises:
        OSError: If the directory cannot be created or a file cannot be written.
    """
    root = (base_dir or get_settings().data.data_dir) / _SNAPSHOT_SUBDIR
    timestamp = datetime.now(timezone.utc)
    snapshot_name = (
        f"{profile.ticker.replace('^', '').replace('/', '_')}"
        f"_{timestamp.strftime('%Y%m%dT%H%M%SZ')}"
    )
    snapshot_dir = root / snapshot_name
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # --- snapshot.json ---
    envelope: dict = {
        "generated_at": timestamp.isoformat(),
        "ticker": profile.ticker,
        "source": profile.source,
        "profile": profile.to_dict(),
        "return_stats_decimal": stats.to_dict(scale="decimal"),
        "return_stats_percent": stats.to_dict(scale="percent"),
    }
    json_path = snapshot_dir / "snapshot.json"
    json_path.write_text(json.dumps(envelope, indent=2), encoding="utf-8")

    # --- returns.csv ---
    csv_path = snapshot_dir / "returns.csv"
    returns.decimal.rename("log_return_decimal").reset_index().rename(
        columns={"index": "date", "date": "date"}
    ).to_csv(csv_path, index=False)

    logger.info(
        "Saved EDA snapshot: ticker=%s path=%s",
        profile.ticker,
        snapshot_dir,
    )
    return snapshot_dir

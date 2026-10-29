"""Transactional persistence for related experiment artifacts."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path

ArtifactWriter = Callable[[Path], None]


def write_artifact_bundles(
    bundles: Mapping[Path, Mapping[str, ArtifactWriter]],
) -> None:
    """Stage complete artifact directories, then replace them with rollback on failure.

    Existing directory contents are copied into the staged version so unrelated
    files are preserved. All writers finish before any published directory moves.
    If a publish step fails, already-moved directories are restored from backups.
    """
    if not bundles:
        return

    staged: dict[Path, Path] = {}
    backups: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        for raw_output, writers in bundles.items():
            output = Path(raw_output)
            output.parent.mkdir(parents=True, exist_ok=True)
            stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
            staged[output] = stage
            if output.exists():
                shutil.copytree(output, stage, dirs_exist_ok=True)
            for relative_name, writer in writers.items():
                relative = Path(relative_name)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"Artifact name must stay within its output directory: {relative_name!r}")
                target = stage / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                writer(target)

        for output, stage in staged.items():
            if output.exists():
                backup = Path(tempfile.mkdtemp(prefix=f".{output.name}.backup-", dir=output.parent))
                backup.rmdir()
                output.rename(backup)
                backups[output] = backup
            stage.rename(output)
            published.append(output)
    except BaseException:
        for output in reversed(published):
            if output.exists():
                shutil.rmtree(output)
            backup = backups.get(output)
            if backup is not None and backup.exists():
                backup.rename(output)
        for output, backup in backups.items():
            if output not in published and backup.exists() and not output.exists():
                backup.rename(output)
        raise
    finally:
        for stage in staged.values():
            if stage.exists():
                shutil.rmtree(stage, ignore_errors=True)

    for backup in backups.values():
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)

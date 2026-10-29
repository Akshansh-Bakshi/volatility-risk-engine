from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.artifacts import write_artifact_bundles


def test_artifact_bundle_replaces_complete_directory_and_preserves_other_files(tmp_path: Path) -> None:
    output = tmp_path / "results"
    output.mkdir()
    (output / "notes.txt").write_text("keep", encoding="utf-8")
    (output / "old.csv").write_text("old", encoding="utf-8")

    write_artifact_bundles({output: {
        "forecasts.csv": lambda path: path.write_text("new forecasts", encoding="utf-8"),
        "metadata.json": lambda path: path.write_text("{}", encoding="utf-8"),
    }})

    assert (output / "notes.txt").read_text(encoding="utf-8") == "keep"
    assert (output / "old.csv").read_text(encoding="utf-8") == "old"
    assert (output / "forecasts.csv").read_text(encoding="utf-8") == "new forecasts"
    assert (output / "metadata.json").read_text(encoding="utf-8") == "{}"
    assert not list(tmp_path.glob(".*.stage-*"))
    assert not list(tmp_path.glob(".*.backup-*"))


def test_failed_artifact_writer_leaves_previous_bundle_intact(tmp_path: Path) -> None:
    output = tmp_path / "results"
    output.mkdir()
    (output / "metadata.json").write_text('{"generation": 1}', encoding="utf-8")

    def fail(path: Path) -> None:
        path.write_text("partial", encoding="utf-8")
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        write_artifact_bundles({output: {
            "forecasts.csv": fail,
            "metadata.json": lambda path: path.write_text('{"generation": 2}', encoding="utf-8"),
        }})

    assert (output / "metadata.json").read_text(encoding="utf-8") == '{"generation": 1}'
    assert not (output / "forecasts.csv").exists()
    assert not list(tmp_path.glob(".*.stage-*"))


def test_artifact_bundle_rejects_paths_outside_output_directory(tmp_path: Path) -> None:
    output = tmp_path / "results"
    with pytest.raises(ValueError, match="stay within"):
        write_artifact_bundles({output: {"../escape.txt": lambda path: path.write_text("bad")}})
    assert not (tmp_path / "escape.txt").exists()


def test_multi_directory_publish_failure_restores_every_previous_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    risk, stress = tmp_path / "risk", tmp_path / "stress"
    risk.mkdir()
    stress.mkdir()
    (risk / "metadata.json").write_text("old risk", encoding="utf-8")
    (stress / "metadata.json").write_text("old stress", encoding="utf-8")
    original_rename = Path.rename

    def fail_stress_publish(path: Path, target: Path) -> Path:
        if path.name.startswith(".stress.stage-") and Path(target) == stress:
            raise OSError("publish failed")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_stress_publish)
    with pytest.raises(OSError, match="publish failed"):
        write_artifact_bundles({
            risk: {"metadata.json": lambda path: path.write_text("new risk", encoding="utf-8")},
            stress: {"metadata.json": lambda path: path.write_text("new stress", encoding="utf-8")},
        })

    assert (risk / "metadata.json").read_text(encoding="utf-8") == "old risk"
    assert (stress / "metadata.json").read_text(encoding="utf-8") == "old stress"
    assert not list(tmp_path.glob(".*.stage-*"))
    assert not list(tmp_path.glob(".*.backup-*"))


"""Every project module must import cleanly and the expected structure must exist."""

from __future__ import annotations

import importlib
import pkgutil

import pytest

import src

PIPELINE_SUBPACKAGES = (
    "data",
    "preprocessing",
    "statistics",
    "models",
    "forecasting",
    "risk",
    "backtesting",
    "utils",
)

ALL_MODULES = sorted(
    module.name for module in pkgutil.walk_packages(src.__path__, prefix=f"{src.__name__}.")
)


def test_package_declares_a_version() -> None:
    assert isinstance(src.__version__, str) and src.__version__.count(".") == 2


@pytest.mark.parametrize("subpackage", PIPELINE_SUBPACKAGES)
def test_pipeline_subpackage_exists_and_is_documented(subpackage: str) -> None:
    module = importlib.import_module(f"src.{subpackage}")

    assert (module.__doc__ or "").strip(), f"src.{subpackage} needs a docstring stating its role"


@pytest.mark.parametrize("module_name", ALL_MODULES)
def test_module_imports_cleanly(module_name: str) -> None:
    assert importlib.import_module(module_name).__name__ == module_name


def test_foundation_modules_are_discovered() -> None:
    assert {"src.config", "src.logging_config", "src.exceptions"} <= set(ALL_MODULES)

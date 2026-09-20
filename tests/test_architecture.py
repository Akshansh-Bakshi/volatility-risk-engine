"""Architecture guard: dependencies may only point down the pipeline.

The layers mirror the pipeline documented in the README.  A module may import
from its own layer or from any layer below it, never from one above.  This keeps
each stage replaceable and makes look-ahead paths (e.g. ``data`` reaching into
``models``) structurally impossible.  When a new top-level component is added,
assign it a layer in ``LAYERS``; ``test_every_component_is_assigned_a_layer``
fails until you do.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import src

PACKAGE = src.__name__
SRC_ROOT = Path(src.__file__).resolve().parent

# Lowest layer first.
LAYERS: tuple[frozenset[str], ...] = (
    frozenset({"config", "logging_config", "exceptions", "utils"}),
    frozenset({"data"}),
    frozenset({"preprocessing"}),
    frozenset({"statistics"}),
    frozenset({"models"}),
    frozenset({"forecasting"}),
    frozenset({"risk"}),
    frozenset({"backtesting"}),
)
LAYER_OF = {name: rank for rank, names in enumerate(LAYERS) for name in names}


def _components() -> set[str]:
    packages = {p.parent.name for p in SRC_ROOT.glob("*/__init__.py")}
    modules = {p.stem for p in SRC_ROOT.glob("*.py") if p.stem != "__init__"}
    return packages | modules


def imported_components(source: str, module_name: str, *, is_package: bool = False) -> set[str]:
    """Return the top-level ``src`` components imported by ``source``."""
    parts = module_name.split(".")
    package_parts = parts if is_package else parts[:-1]
    known = _components()
    found: set[str] = set()

    def record(target: list[str], names: list[str]) -> None:
        if not target or target[0] != PACKAGE:
            return
        if len(target) > 1:
            found.add(target[1])
        else:  # `from src import models, __version__`
            found.update(name for name in names if name in known)

    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                record(alias.name.split("."), [])
        elif isinstance(node, ast.ImportFrom):
            base = package_parts[: len(package_parts) - (node.level - 1)] if node.level else []
            target = base + (node.module.split(".") if node.module else [])
            record(target, [alias.name for alias in node.names])
    return found


def layer_violations(module_name: str, source: str, *, is_package: bool = False) -> list[str]:
    """Describe every import in ``source`` that points to a higher layer."""
    module_parts = module_name.split(".")
    if len(module_parts) < 2:
        return []
    own = module_parts[1]
    own_rank = LAYER_OF.get(own)
    if own_rank is None:
        return []
    return [
        f"{module_name} ({own}) imports {dep}, which sits in a higher layer"
        for dep in sorted(imported_components(source, module_name, is_package=is_package))
        if LAYER_OF.get(dep, -1) > own_rank
    ]


@pytest.mark.parametrize(
    ("source", "module_name", "expected"),
    [
        ("import src.models.garch", "src.risk.var", {"models"}),
        ("from src.risk import var", "src.backtesting.engine", {"risk"}),
        ("from src import backtesting, __version__", "src.utils.helpers", {"backtesting"}),
        ("from ..models import base", "src.risk.var", {"models"}),
        ("from .. import forecasting", "src.risk.var", {"forecasting"}),
        ("from . import sibling", "src.risk.var", {"risk"}),
        ("import numpy as np\nfrom src.config import get_settings", "src.data.loader", {"config"}),
        ("import statistics\nimport logging", "src.data.loader", set()),
    ],
)
def test_import_extraction(source: str, module_name: str, expected: set[str]) -> None:
    assert imported_components(source, module_name) == expected


def test_violations_are_detected() -> None:
    assert layer_violations("src.data.loader", "from src.models import base")
    assert layer_violations("src.config", "from src.data import loader")
    assert layer_violations("src.risk.var", "from ..backtesting import engine")


def test_downward_and_same_layer_imports_are_allowed() -> None:
    assert not layer_violations("src.models.garch", "from src.statistics import diagnostics")
    assert not layer_violations("src.models.garch", "from src.models import base")
    assert not layer_violations("src.logging_config", "from src.config import get_settings")


def test_every_component_is_assigned_a_layer() -> None:
    assert _components() <= set(LAYER_OF), "assign new components to a layer in LAYERS"


def _module_name(path: Path) -> tuple[str, bool]:
    relative = path.relative_to(SRC_ROOT.parent).with_suffix("")
    is_package = relative.name == "__init__"
    return ".".join(relative.parent.parts if is_package else relative.parts), is_package


def test_no_module_imports_from_a_higher_layer() -> None:
    violations: list[str] = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        module_name, is_package = _module_name(path)
        violations += layer_violations(
            module_name, path.read_text(encoding="utf-8"), is_package=is_package
        )

    assert not violations, "\n".join(violations)


# --- vendor isolation --------------------------------------------------------------------------


def imported_modules(source: str, module_name: str, *, is_package: bool = False) -> set[str]:
    """Return every dotted module name ``source`` imports (relative imports resolved)."""
    parts = module_name.split(".")
    package_parts = parts if is_package else parts[:-1]
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = package_parts[: len(package_parts) - (node.level - 1)] if node.level else []
            target = ".".join(base + (node.module.split(".") if node.module else []))
            found.add(target)
            found.update(f"{target}.{alias.name}" for alias in node.names)
    return found


def _importers_of(prefix: str) -> set[str]:
    """Project files (including app.py) that import ``prefix`` or anything below it."""
    importers: set[str] = set()
    files = [*sorted(SRC_ROOT.rglob("*.py")), SRC_ROOT.parent / "app.py"]
    for path in files:
        if path.name == "app.py":
            module_name, is_package = "app", False
        else:
            module_name, is_package = _module_name(path)
        source = path.read_text(encoding="utf-8")
        names = imported_modules(source, module_name, is_package=is_package)
        if any(name == prefix or name.startswith(prefix + ".") for name in names):
            importers.add(path.relative_to(SRC_ROOT.parent).as_posix())
    return importers


@pytest.mark.parametrize(
    ("source", "module_name", "expected"),
    [
        ("import yfinance as yf", "src.data.x", {"yfinance"}),
        ("from yfinance import exceptions", "src.data.x", {"yfinance", "yfinance.exceptions"}),
        ("from .yahoo import Provider", "src.data.factory", {"src.data.yahoo",
                                                             "src.data.yahoo.Provider"}),
        ("from src.data import yahoo", "src.app", {"src.data", "src.data.yahoo"}),
    ],
)
def test_import_name_extraction(source: str, module_name: str, expected: set[str]) -> None:
    assert expected <= imported_modules(source, module_name)


def test_yfinance_is_imported_only_by_the_yahoo_provider() -> None:
    assert _importers_of("yfinance") == {"src/data/yahoo.py"}


def test_the_yahoo_provider_is_wired_only_by_the_composition_root() -> None:
    assert _importers_of("src.data.yahoo") == {"src/data/factory.py"}

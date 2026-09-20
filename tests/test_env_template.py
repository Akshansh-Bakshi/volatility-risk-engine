"""The documented configuration surface must match what the code actually reads."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.config import ENV_PREFIX, PROJECT_ROOT, load_settings

CONFIG_SOURCE = (PROJECT_ROOT / "src" / "config.py").read_text(encoding="utf-8")
CODE_VARIABLES = {
    f"{ENV_PREFIX}{name}" for name in re.findall(r'_read_env\(env, "([A-Z_]+)"\)', CONFIG_SOURCE)
}


def _template_assignments() -> dict[str, str]:
    lines = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    return dict(line.split("=", 1) for line in lines if line and not line.startswith("#"))


def test_the_code_reads_a_non_trivial_set_of_variables() -> None:
    assert len(CODE_VARIABLES) >= 11, "the pattern used to discover variables has gone stale"


def test_env_example_defines_exactly_the_variables_the_code_reads() -> None:
    assert set(_template_assignments()) == CODE_VARIABLES


def test_env_example_is_a_valid_configuration_equal_to_the_defaults() -> None:
    # Blank values mean "unset", so loading the template must reproduce the built-in defaults.
    assert load_settings(_template_assignments()) == load_settings({})


@pytest.mark.parametrize("variable", sorted(CODE_VARIABLES))
def test_readme_documents_every_variable(variable: str) -> None:
    readme = Path(PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert f"`{variable}`" in readme

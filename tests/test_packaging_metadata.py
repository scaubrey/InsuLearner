from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.unit


def test_pyproject_contains_console_entrypoint():
    content = (REPO_ROOT / "pyproject.toml").read_text()
    assert "[tool.poetry.scripts]" in content
    assert 'insulearner = "InsuLearner.insulearner:main"' in content


def test_pyproject_declares_package_include():
    content = (REPO_ROOT / "pyproject.toml").read_text()
    assert '{ include = "InsuLearner" }' in content


def test_readme_contains_pip_install_and_cli_usage():
    readme = (REPO_ROOT / "README.md").read_text().lower()
    assert "pip install insulearner" in readme
    assert "usage (cli)" in readme

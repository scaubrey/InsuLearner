import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.unit


def _run_cli(args):
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    return subprocess.run(
        [sys.executable, "-m", "InsuLearner.insulearner"] + args,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )


def test_cli_help_smoke():
    result = _run_cli(["--help"])
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()
    assert "--num_days" in result.stdout


def test_cli_missing_required_args_fails():
    result = _run_cli([])
    assert result.returncode != 0
    output = (result.stderr + result.stdout).lower()
    assert "tp_username" in output or "tidepool" in output


def test_cli_invalid_gender_fails_fast():
    result = _run_cli(["user@example.com", "pw", "--gender", "invalid"])
    assert result.returncode != 0
    assert "invalid choice" in (result.stderr + result.stdout).lower()

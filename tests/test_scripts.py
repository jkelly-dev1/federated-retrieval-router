"""The commands the README documents, run the way the README documents them.

Python puts the SCRIPT's directory on `sys.path`, not the working directory, so
`python scripts/run_demo.py` cannot import the package beside `scripts/`. An
in-process suite never catches that, because pytest supplies its own path.
These run as SUBPROCESSES with a clean environment, which is the only way to
reproduce what a reviewer actually types.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _clean_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}


def _run(args, cwd=ROOT):
    return subprocess.run(
        args, cwd=cwd, env=_clean_env(), capture_output=True, text=True, timeout=180
    )


def test_the_documented_demo_command_runs():
    result = _run([sys.executable, "scripts/run_demo.py"])
    assert result.returncode == 0, result.stderr
    assert "federated-retrieval-router demo" in result.stdout


def test_the_demo_runs_from_any_working_directory(tmp_path):
    """Mutation check for the sys.path bootstrap."""
    result = _run([sys.executable, str(ROOT / "scripts" / "run_demo.py")], cwd=tmp_path)
    assert result.returncode == 0, result.stderr


def test_the_gate_is_runnable_as_a_module():
    result = _run([sys.executable, "-m", "router.gate"])
    assert result.returncode == 0, result.stderr
    assert "GATE PASSED" in result.stdout


def test_the_demo_is_byte_identical_across_runs():
    a = _run([sys.executable, "scripts/run_demo.py"])
    b = _run([sys.executable, "scripts/run_demo.py"])
    assert a.returncode == 0 and b.returncode == 0
    assert a.stdout == b.stdout


def test_the_demo_output_fits_eighty_columns():
    """A capture that wraps in a terminal or on GitHub is unreadable, and
    SAMPLE_RUN.md is pasted verbatim."""
    result = _run([sys.executable, "scripts/run_demo.py"])
    wide = [l for l in result.stdout.splitlines() if len(l) > 80]
    assert not wide, f"{len(wide)} lines over 80 columns: {wide[:2]}"

"""Preparing pinned upstream source remains safe after Python imports."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REVISION = "834635e4cf277ed8ca92049fc375b17c3dc20748"
PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "examples" / "fireredasr_llm" / "prepare_source.sh"


@pytest.fixture()
def firered_checkout(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git is required to prepare FireRedASR source")
    source = Path(os.environ.get("FIRERED_ASR_REPO", PROJECT / ".vendor" / "FireRedASR"))
    if not (source / ".git").exists():
        pytest.skip("Prepare the pinned FireRedASR checkout first, or set FIRERED_ASR_REPO")
    checkout = tmp_path / "FireRedASR"
    subprocess.run(
        ["git", "clone", "--shared", "--no-checkout", str(source), str(checkout)], check=True, capture_output=True
    )
    subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", REVISION], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(checkout), "remote", "set-url", "origin", "https://github.com/FireRedTeam/FireRedASR.git"],
        check=True,
        capture_output=True,
    )
    return checkout


def prepare(checkout):
    return subprocess.run(["bash", str(SCRIPT), str(checkout)], capture_output=True, text=True)


def test_prepare_is_idempotent_after_import(firered_checkout):
    assert prepare(firered_checkout).returncode == 0
    env = {**os.environ, "PYTHONPATH": str(firered_checkout)}
    env.pop("PYTHONDONTWRITEBYTECODE", None)
    env.pop("PYTHONPYCACHEPREFIX", None)
    subprocess.run(
        [sys.executable, "-c", "import sys; sys.dont_write_bytecode = False; import fireredasr.utils.wer"],
        env=env,
        check=True,
        capture_output=True,
    )
    assert list(firered_checkout.rglob("*.pyc")), "Import must create real upstream Python caches"
    result = prepare(firered_checkout)
    assert result.returncode == 0, result.stderr
    assert list(firered_checkout.rglob("*.pyc")), "Preparing must preserve generated caches"


@pytest.mark.parametrize("tracked", [True, False])
def test_prepare_refuses_source_changes(firered_checkout, tracked):
    source = firered_checkout / "fireredasr" / "utils" / ("wer.py" if tracked else "custom.py")
    contents = (source.read_text() if tracked else "") + "\n# Local source changes must be retained.\n"
    source.write_text(contents)
    result = prepare(firered_checkout)
    assert result.returncode == 1
    assert "dirty checkout" in result.stderr
    assert source.read_text() == contents
    current = subprocess.check_output(["git", "-C", str(firered_checkout), "rev-parse", "HEAD"], text=True).strip()
    assert current == REVISION

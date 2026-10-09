"""Launch the installed Swift engine with caller-supplied native arguments."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from collections.abc import Sequence


def build_sft_command(swift_args: Sequence[str]) -> list[str]:
    """Keep model, plugin, dataset and training configuration with the caller."""
    return [sys.executable, "-m", "swift.cli.main", "sft", *swift_args]


def run_sft(swift_args: Sequence[str]) -> int:
    if importlib.util.find_spec("swift") is None:
        raise RuntimeError("Swift is not installed. Run uv sync --locked first.")
    return subprocess.call(build_sft_command(swift_args))

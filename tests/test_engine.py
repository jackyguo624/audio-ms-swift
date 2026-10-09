import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from audio_swift.backends.swift.engine import build_sft_command, run_sft
from audio_swift.cli import main


def test_launcher_does_not_inject_configuration():
    assert build_sft_command([]) == [sys.executable, "-m", "swift.cli.main", "sft"]


@pytest.mark.parametrize(
    "native",
    [
        ["--model", "any/model", "--dataset", "train data.jsonl", "--learning_rate", "3e-5"],
        ["--model-type=custom", "--template", "custom", "--external-plugins", "my plugin.py"],
        ["--padding-free=true", "--packing", "false", "--truncation-strategy=left"],
        ["--gradient_accumulation_steps", "7", "--max_steps", "2", "--run_name", "a b;$(literal)"],
    ],
)
def test_native_arguments_are_forwarded_verbatim(native):
    original = list(native)
    assert build_sft_command(native)[4:] == original
    assert native == original


def test_exit_code_and_environment_are_inherited(monkeypatch):
    monkeypatch.setenv("CUSTOM_PLUGIN_DATA", "/caller/data with spaces.jsonl")
    monkeypatch.setattr("audio_swift.backends.swift.engine.importlib.util.find_spec", lambda name: object())
    captured = []

    def call(argv):
        captured.append(argv)
        assert os.environ["CUSTOM_PLUGIN_DATA"] == "/caller/data with spaces.jsonl"
        return 7

    monkeypatch.setattr("audio_swift.backends.swift.engine.subprocess.call", call)
    assert run_sft(["--model", "custom"]) == 7
    assert captured == [build_sft_command(["--model", "custom"])]


def test_missing_engine_has_a_generic_installation_hint(monkeypatch):
    monkeypatch.setattr("audio_swift.backends.swift.engine.importlib.util.find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match="uv sync --locked"):
        run_sft([])


def test_cli_forwards_native_help_and_configuration(monkeypatch):
    calls = []
    monkeypatch.setattr("audio_swift.cli.run_sft", lambda native: calls.append(native) or 0)
    assert main(["sft", "--help"]) == 0
    assert main(["sft", "--model", "custom", "--external_plugins", "plugin path.py"]) == 0
    assert calls == [["--help"], ["--model", "custom", "--external_plugins", "plugin path.py"]]


def test_dry_run_preserves_native_arguments_without_loading_engine(capsys):
    native = ["--model", "custom", "--dataset", "does not need to exist.jsonl", "--run_name", "a b;$(literal)"]
    assert main(["sft", "--dry-run", *native]) == 0
    assert shlex.split(capsys.readouterr().out) == build_sft_command(native)


def test_separator_is_removed_only_at_the_wrapper_boundary(monkeypatch):
    calls = []
    monkeypatch.setattr("audio_swift.cli.run_sft", lambda native: calls.append(native) or 0)
    native = ["--model", "custom", "--", "literal"]
    assert main(["sft", "--", *native]) == 0
    assert calls == [native]


def test_help_outside_checkout_does_not_import_framework_or_plugins(tmp_path):
    source = Path(__file__).resolve().parents[1] / "src"
    code = "from audio_swift.cli import main; import sys; main(['--version']); assert not any(k in sys.modules for k in ['torch', 'swift', 'audio_swift.plugins'])"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(source)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0.1.0"

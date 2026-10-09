"""Bash examples own configuration and pass paths/arguments without shell evaluation."""

import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def example_environment(tmp_path):
    binary = tmp_path / "bin"
    binary.mkdir()
    uv = binary / "uv"
    uv.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "keys = ['FIRERED_ASR_DATASET_PATH', 'FIRERED_ASR_VAL_DATASET_PATH', 'CUDA_VISIBLE_DEVICES', 'NPROC_PER_NODE']\n"
        "print(json.dumps({'argv':sys.argv[1:], 'env':{key:os.environ.get(key) for key in keys}}))\n"
    )
    uv.chmod(0o755)
    clear = {
        "MODEL_DIR",
        "DATASET",
        "VAL_DATASET",
        "OUTPUT_DIR",
        "INPUT",
        "OUTPUT",
        "ASR_LANGUAGE",
        "LIMIT",
        "DRY_RUN",
        "TRAIN_BS",
        "EVAL_BS",
        "GRAD_ACC",
        "LEARNING_RATE",
        "NUM_TRAIN_EPOCHS",
        "MAX_LENGTH",
        "TORCH_DTYPE",
        "ATTN_IMPL",
        "TUNER_TYPE",
        "SPLIT_DATASET_RATIO",
        "PLUGIN_PATH",
        "CUDA_VISIBLE_DEVICES",
        "NPROC_PER_NODE",
    }
    env = {key: value for key, value in os.environ.items() if key not in clear}
    env["PATH"] = str(binary) + os.pathsep + env["PATH"]
    return tmp_path, env


def invoke(script, args, cwd, env):
    result = subprocess.run(["bash", str(script), *args], cwd=cwd, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def option(arguments, name):
    return arguments[arguments.index(name) + 1]


@pytest.mark.parametrize("example", ["qwen3_asr", "fireredasr_llm"])
@pytest.mark.parametrize("use_env", [False, True])
def test_sft_examples_declare_native_configuration(example_environment, example, use_env):
    cwd, env = example_environment
    script = ROOT / "examples" / example / "sft.sh"
    model, dataset, validation = (
        "/models/model with spaces",
        "/data/train with spaces.jsonl",
        "/data/dev with spaces.jsonl",
    )
    env.update(VAL_DATASET=validation, GRAD_ACC="3", LEARNING_RATE="2e-5", OUTPUT_DIR=str(cwd / "output space"))
    args = []
    if use_env:
        env.update(MODEL_DIR=model, DATASET=dataset)
    else:
        args = [model, dataset]
    payload = invoke(script, [*args, "--run_name", "a b;$(literal)"], cwd, env)
    argv = payload["argv"]
    assert option(argv, "--project") == str(ROOT)
    native = argv[argv.index("audio-swift") + 1 :]
    assert native[0] == "sft" and "--recipe" not in native
    assert option(native, "--model") == model
    assert option(native, "--model_type") == example
    assert option(native, "--gradient_accumulation_steps") == "3"
    assert option(native, "--learning_rate") == "2e-5"
    assert option(native, "--split_dataset_ratio") == "0"
    assert native[-2:] == ["--run_name", "a b;$(literal)"]
    if example == "fireredasr_llm":
        assert option(native, "--external_plugins") == str(script.with_name("plugin.py"))
        assert payload["env"]["FIRERED_ASR_DATASET_PATH"] == dataset
        assert payload["env"]["FIRERED_ASR_VAL_DATASET_PATH"] == validation
    else:
        assert option(native, "--dataset") == dataset
        assert option(native, "--val_dataset") == validation


@pytest.mark.parametrize("example", ["qwen3_asr", "fireredasr_llm"])
def test_bash_dry_run_precedes_native_flags(example_environment, example):
    cwd, env = example_environment
    env.update(DRY_RUN="1", MODEL_DIR="/model", DATASET="/data")
    payload = invoke(ROOT / "examples" / example / "sft.sh", [], cwd, env)
    argv = payload["argv"]
    assert argv[argv.index("audio-swift") + 1 :][:2] == ["sft", "--dry-run"]


@pytest.mark.parametrize("example", ["qwen3_asr", "fireredasr_llm"])
def test_prepare_examples_supply_model_specific_format(example_environment, example):
    cwd, env = example_environment
    env.update(ASR_LANGUAGE="English", LIMIT="8")
    payload = invoke(
        ROOT / "examples" / example / "prepare_data.sh", ["raw data.jsonl", "prepared data.jsonl"], cwd, env
    )
    argv = payload["argv"]
    assert option(argv, "--input") == "raw data.jsonl" and option(argv, "--output") == "prepared data.jsonl"
    assert "--recipe" not in argv and "--language" not in argv
    assert option(argv, "--limit") == "8"
    if example == "qwen3_asr":
        assert option(argv, "--response-prefix") == "language English<asr_text>"
        assert "--response-prefix-pattern" in argv
    else:
        assert option(argv, "--prompt") == "<audio>请转写音频为文字"
        assert "--response-prefix" not in argv


@pytest.mark.parametrize("example", ["qwen3_asr", "fireredasr_llm"])
@pytest.mark.parametrize("script", ["sft.sh", "prepare_data.sh"])
def test_example_help_does_not_invoke_uv(example_environment, example, script):
    cwd, env = example_environment
    result = subprocess.run(
        ["bash", str(ROOT / "examples" / example / script), "--help"], cwd=cwd, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0 and "Usage:" in result.stdout
    assert '"argv"' not in result.stdout


def test_transcript_language_is_independent_of_system_locale(example_environment):
    cwd, env = example_environment
    env["LANGUAGE"] = "zh_CN:zh"
    payload = invoke(ROOT / "examples/qwen3_asr/prepare_data.sh", ["raw.jsonl", "prepared.jsonl"], cwd, env)
    assert option(payload["argv"], "--response-prefix") == "language Chinese<asr_text>"

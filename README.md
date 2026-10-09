# audio-ms-swift

一个面向语音模型训练的独立 Python 项目。模型结构、音频特征、数据处理和语音任务由本项目维护，`ms-swift` 作为安装依赖提供通用训练能力。LLM 主干继续复用 Qwen 等开源模型系列。

第一版提供两套通过 Bash 配置的 SFT 示例：

| 示例 | 模型 | 默认训练方式 |
| --- | --- | --- |
| [`qwen3_asr`](examples/qwen3_asr/README.md) | Qwen3-ASR-1.7B / 0.6B | LLM LoRA，冻结音频编码器和投影层 |
| [`fireredasr_llm`](examples/fireredasr_llm/README.md) | FireRedASR-LLM-L + Qwen2-7B-Instruct | `lora_llm`，LLM LoRA，音频侧全量训练 |

本仓库不再维护 Swift 原生源码。安装后导入名为 `audio_swift`，命令为 `audio-swift`。TTS、codec 模型训练和统一推理接口留待后续版本实现。

## 安装

使用 Python 3.11 和 [uv](https://docs.astral.sh/uv/)，在仓库根目录运行：

```bash
uv sync --locked --all-extras
uv run --locked audio-swift --help
```

只使用一个模型时可分别安装 `uv sync --locked --extra qwen3-asr` 或 `uv sync --locked --extra fireredasr`。训练脚本会选择对应 extra，并使用项目内的 uv 环境。

Swift 固定到提交 [`410a5d18930725d1d3cfe3acdd611eb9795c403b`](https://github.com/modelscope/ms-swift/tree/410a5d18930725d1d3cfe3acdd611eb9795c403b)，由 uv 安装；无需复制 Swift 代码或执行 `pip install -e` 安装旧仓库。`uv.lock` 固定完整的依赖解析结果。当前兼容基线为 PyTorch 2.10.0、Transformers 4.57.6、datasets 4.8.4、PEFT 0.18.1、Accelerate 1.12.0；Qwen3-ASR 使用 qwen-asr 0.0.6。GPU 驱动须能运行所安装的 PyTorch CUDA 构建。

FireRedASR 还需准备其官方模型实现：

```bash
bash examples/fireredasr_llm/prepare_source.sh
export FIRERED_ASR_REPO="$PWD/.vendor/FireRedASR"
```

此脚本检出提交 [`834635e4cf277ed8ca92049fc375b17c3dc20748`](https://github.com/FireRedTeam/FireRedASR/tree/834635e4cf277ed8ca92049fc375b17c3dc20748)。完整权重布局见 [FireRedASR 示例](examples/fireredasr_llm/README.md)。源码与模型权重不提交到本项目 Git。

## 准备 tiny 数据

可从 AISHELL-1 或 LibriSpeech 中选取少量音频，并先整理成每行一个样本的 JSONL：

```json
{"audio":"/data/AISHELL-1/wav/utterance.wav","text":"今天天气真好"}
```

两个模型共用原始音频和转写，分别通过示例脚本转换为训练格式。例如中文数据使用：

```bash
ASR_LANGUAGE=Chinese bash examples/qwen3_asr/prepare_data.sh \
    data/raw/train.jsonl data/qwen3_asr/train.jsonl --limit 32
ASR_LANGUAGE=Chinese bash examples/qwen3_asr/prepare_data.sh \
    data/raw/dev.jsonl data/qwen3_asr/dev.jsonl --limit 8

bash examples/fireredasr_llm/prepare_data.sh \
    data/raw/train.jsonl data/fireredasr_llm/train.jsonl --limit 32
bash examples/fireredasr_llm/prepare_data.sh \
    data/raw/dev.jsonl data/fireredasr_llm/dev.jsonl --limit 8
```

LibriSpeech 的 Qwen3-ASR 数据设置 `ASR_LANGUAGE=English`。原始 JSONL 中的相对音频路径按该 JSONL 所在目录解析，输出转换为绝对路径。训练集和验证集应使用不同音频；`--limit` 仅限制条数，不自动下载或划分数据。音频文件必须实际存在，准备后的 JSONL 和音频文件需一起保留。

## 单卡 SFT

Qwen3-ASR 可使用 Hub 模型 ID 或本地目录：

```bash
CUDA_VISIBLE_DEVICES=0 VAL_DATASET=data/qwen3_asr/dev.jsonl \
bash examples/qwen3_asr/sft.sh \
    Qwen/Qwen3-ASR-0.6B data/qwen3_asr/train.jsonl \
    --max_steps 4 --eval_steps 4 --save_steps 4 --logging_steps 1
```

FireRedASR 使用准备好的本地模型目录：

```bash
CUDA_VISIBLE_DEVICES=0 VAL_DATASET=data/fireredasr_llm/dev.jsonl \
bash examples/fireredasr_llm/sft.sh \
    /models/FireRedASR-LLM-L data/fireredasr_llm/train.jsonl \
    --max_steps 4 --eval_steps 4 --save_steps 4 --logging_steps 1
```

两套脚本默认单卡、每卡 batch 为 1、梯度累积 16。输出分别放入 `output/qwen3_asr` 和 `output/fireredasr_llm`，可用 `OUTPUT_DIR` 修改。脚本支持环境变量、位置参数、含空格路径和从其他工作目录调用；运行 `bash examples/qwen3_asr/sft.sh --help` 不安装依赖。

模型 ID、模型类型、模板、冻结范围、batch、学习率等训练配置均写在各自的 `sft.sh`，可修改脚本、设置环境变量或追加 Swift 参数。通用 Python 启动器不选择模型，也不注入这些训练默认值。可预览传给 Swift 的参数：

```bash
DRY_RUN=1 bash examples/qwen3_asr/sft.sh \
    /models/Qwen3-ASR-0.6B data/qwen3_asr/train.jsonl --max_steps 4
```

`audio-swift sft` 后面使用 Swift 原生参数，如 `--model`、`--model_type`、`--template`、`--dataset`、`--val_dataset` 和 `--output_dir`；启动器完整保留参数顺序。直接预览时把 `--dry-run` 放在 `sft` 之后的第一个位置，其他参数交给 Swift 解释。恢复训练追加 `--resume_from_checkpoint /path/to/checkpoint-N`。多卡、全量训练和模型限制分别见两个示例的 README。

预览不包含环境变量。FireRed 还依赖 Bash 配置的数据与源码环境；正式运行时保留这些配置、取消 `DRY_RUN`，重新执行同一 Bash 脚本，不要把预览输出直接复制到新 shell 执行。

## 项目边界

```text
模型与训练参数                               examples/*/*.sh
                   ↓
音频数据 / 特征 / 语音模型                   audio_swift
模型与模板注册                               audio_swift.plugins
                   ↓
通用训练启动器                               audio_swift.backends
                   ↓
LoRA / Trainer / DDP / checkpoint             ms-swift 依赖
                   ↓
张量计算 / CUDA                              PyTorch
```

纯模型与音频处理模块不依赖 Swift；模型注册集中在可选 plugin 中。FireRed 示例通过 `--external_plugins` 显式加载自己的 plugin，Qwen3-ASR 使用 Swift 现有模型支持。项目结构和后续扩展约束见 [架构说明](docs/architecture.md)。版权与来源见 [NOTICE](NOTICE) 和 [LICENSE](LICENSE)。

## 开发验证

安装完整依赖并准备固定的 FireRed 源码后，可运行：

```bash
uv run --locked --all-extras pytest
uv run --locked --all-extras ruff check src tests
uv run --locked --all-extras ruff format --check src tests
uv run --locked --all-extras pyright
uv build
```

测试包含本地构建的 tiny encoder/Qwen、真实 fbank/CMVN、LoRA 保存恢复，以及数据和命令行边界检查；不下载预训练权重。

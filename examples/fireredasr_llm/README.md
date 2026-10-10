# FireRedASR-LLM SFT

使用可选的外部 plugin，把本项目的 **FireRedASR-LLM-L** 模型封装、特征与模板接入安装的 ms-swift 引擎。LLM 主干为 Qwen2-7B-Instruct。本示例面向原版 FireRedASR，未适配 FireRedASR2-LLM。

## 环境和源码

在项目根目录安装 Python 依赖并准备固定版本的官方源码：

```bash
uv sync --locked
bash examples/fireredasr_llm/prepare_source.sh
export FIRERED_ASR_REPO="$PWD/.vendor/FireRedASR"
```

默认安装同时包含本项目 Qwen3-ASR 与 FireRedASR-LLM 的 Python 依赖，Swift 引擎固定为 PyPI 发布版本 [`ms-swift==4.5.3`](https://pypi.org/project/ms-swift/4.5.3/)。独立的 FireRed 官方源码固定到 [`834635e4cf277ed8ca92049fc375b17c3dc20748`](https://github.com/FireRedTeam/FireRedASR/tree/834635e4cf277ed8ca92049fc375b17c3dc20748)。也可将 `FIRERED_ASR_REPO` 指向这个提交的已有源码目录；目录下须包含 `fireredasr` 包。Python 依赖统一由 uv 管理，避免再用上游 requirements 改写已锁定环境。

## 模型布局

分别下载 [FireRedASR-LLM-L](https://huggingface.co/FireRedTeam/FireRedASR-LLM-L) 和 [Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct)，保留完整文件与以下布局。Qwen 子目录可用软链接：

```text
/models/FireRedASR-LLM-L/
├── config.json
├── model.pth.tar
├── asr_encoder.pth.tar
├── cmvn.ark
└── Qwen2-7B-Instruct/
    ├── config.json
    ├── tokenizer.json
    ├── model-*.safetensors
    └── ...                    # tokenizer、权重索引等全部文件
```

Swift 进入自定义 loader 前需要根目录的模型元信息。如果原始 FireRed 目录缺少 `config.json`，复制 Qwen 配置：

```bash
cp /models/FireRedASR-LLM-L/Qwen2-7B-Instruct/config.json \
    /models/FireRedASR-LLM-L/config.json
```

训练参数 `--model` 必须指向本地 FireRed 根目录。模型加载时会使用当前目录的权重、CMVN 和 Qwen 路径；原预训练 LoRA 如存在会先合并，再创建训练用的新 LoRA。官方 loader 会先在 CPU 上构建模型，主机内存也须能容纳初始化过程。

官方 `model.pth.tar` 和 `asr_encoder.pth.tar` 为兼容 checkpoint 中的 `args`，需要使用 `torch.load(weights_only=False)` 加载；这会反序列化 Python 对象，因此只使用可信的官方权重。`FIRERED_ASR_REPO` 中的 Python 源码也会被执行，应使用上述固定提交的可信官方 checkout。

## 准备数据

原始 JSONL 使用音频与转写，例如 AISHELL-1：

```json
{"audio":"/data/AISHELL-1/wav/utterance.wav","text":"甚至出现交易几乎停滞的情况"}
```

从现有训练集和验证集分别准备 32、8 条 tiny 样本：

```bash
bash examples/fireredasr_llm/prepare_data.sh \
    data/raw/train.jsonl data/fireredasr_llm/train.jsonl --limit 32
bash examples/fireredasr_llm/prepare_data.sh \
    data/raw/dev.jsonl data/fireredasr_llm/dev.jsonl --limit 8
```

`prepare_data.sh` 显式给通用数据工具传入 `--prompt '<audio>请转写音频为文字'`，不添加响应前缀。可设置 `INPUT`、`OUTPUT` 和 `LIMIT` 替代位置参数，额外数据参数原样追加。

原始 JSONL 中的相对音频路径按输入文件所在目录解析，输出转换为绝对路径。音频应为 16 kHz 单声道 WAV，建议控制在 30 秒以内。模型使用原始 fbank + CMVN；音频 token 数由编码器与投影层实际下采样比例确定。转写清洗沿用 FireRed 的 `clean_text`，包括标点、中文空格和大小写处理。

训练数据也支持单轮 Swift `messages` + `audios` 格式：

```json
{"messages":[{"role":"user","content":"<audio>请转写音频为文字"},{"role":"assistant","content":"甚至出现交易几乎停滞的情况"}],"audios":["/data/AISHELL-1/wav/utterance.wav"]}
```

每条样本只有一个音频；不要在输入文本中手工添加 `<speech>`。序列长度包括展开后的语音 token，超长样本采用丢弃策略，避免截断破坏对齐。tiny 数据应指定独立验证集；未指定时默认从训练数据切出 1%。

## 单卡训练

脚本可从任意目录调用。相对模型和数据路径按调用时的工作目录解析；路径含空格时加引号：

```bash
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 \
VAL_DATASET=data/fireredasr_llm/dev.jsonl \
bash examples/fireredasr_llm/sft.sh \
    /models/FireRedASR-LLM-L data/fireredasr_llm/train.jsonl
```

也可以只设置环境变量：

```bash
MODEL_DIR="/models/FireRedASR-LLM-L" \
DATASET="/data/tiny/train.jsonl" VAL_DATASET="/data/tiny/dev.jsonl" \
OUTPUT_DIR="/data/training runs/fireredasr_llm" \
bash examples/fireredasr_llm/sft.sh
```

默认采用 `lora_llm`：LLM 做 LoRA，audio encoder 和 projector 全量训练，并随 checkpoint 保存。因此显存需求高于只训练 LLM LoRA；checkpoint 依赖原始模型目录，不能作为独立的 FireRed 发布模型。

模型选择、数据集别名、冻结与训练默认值均放在 `sft.sh`。脚本通过 `--external_plugins` 加载本目录的 `plugin.py`，后者导入包内的 `audio_swift.plugins.fireredasr_llm.plugin`；可用 `PLUGIN_PATH` 替换入口。脚本把训练与验证 manifest 放入 `FIRERED_ASR_DATASET_PATH`、`FIRERED_ASR_VAL_DATASET_PATH`，并使用 plugin 注册的 `firered_asr_jsonl` 和 `firered_asr_jsonl_val` 数据集。通用 Python 启动器没有 FireRed 分支或训练默认值。

当前只支持 `lora` / `lora_llm` 的适配器训练与保存；若设置 `TUNER_TYPE=lora`，冻结与目标模块按 Swift 的 LoRA 配置解释。模型、模板及 plugin 负责音频对齐与保存约束，full checkpoint 导出不受支持。

默认每卡 batch 为 1、梯度累积 16，BF16、eager attention。`TRAIN_BS`、`EVAL_BS`、`GRAD_ACC`、`LEARNING_RATE`、`VIT_LR`、`ALIGNER_LR`、`NUM_TRAIN_EPOCHS`、`MAX_LENGTH`、`TORCH_DTYPE`、`ATTN_IMPL`、`TUNER_TYPE`、`SPLIT_DATASET_RATIO` 可覆盖对应配置，其他 Swift 原生参数直接追加在末尾。

设置 `DRY_RUN=1` 可预览传给 Swift 的参数，例如：

```bash
DRY_RUN=1 bash examples/fireredasr_llm/sft.sh \
    /models/FireRedASR-LLM-L data/fireredasr_llm/train.jsonl
```

预览不加载模型或执行 plugin 校验，也不包含 `FIRERED_ASR_DATASET_PATH`、`FIRERED_ASR_VAL_DATASET_PATH`、`FIRERED_ASR_REPO` 等数据与源码环境变量。正式运行时保留同一 Bash 配置、取消 `DRY_RUN` 后重跑，不要把预览输出直接复制到新 shell 执行。直接调用 CLI 时需把 `--dry-run` 紧跟在 `audio-swift sft` 后。

四步流程验证：

```bash
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 \
VAL_DATASET=data/fireredasr_llm/dev.jsonl OUTPUT_DIR=output/fireredasr_smoke \
bash examples/fireredasr_llm/sft.sh \
    /models/FireRedASR-LLM-L data/fireredasr_llm/train.jsonl \
    --max_steps 4 --eval_steps 4 --save_steps 4 --logging_steps 1
```

检查训练和验证 loss 以及 checkpoint 中的 LoRA 和音频侧权重。继续做一个优化步以验证恢复：

```bash
VAL_DATASET=data/fireredasr_llm/dev.jsonl OUTPUT_DIR=output/fireredasr_smoke \
bash examples/fireredasr_llm/sft.sh \
    /models/FireRedASR-LLM-L data/fireredasr_llm/train.jsonl \
    --resume_from_checkpoint /absolute/path/to/checkpoint-4 \
    --max_steps 5 --eval_steps 1 --save_steps 1 --logging_steps 1
```

使用训练日志中的实际 checkpoint 路径；Swift 可能创建带版本号的输出子目录。恢复时保留同一基础模型、FireRed 源码和项目依赖。

## 可选多卡与限制

支持每进程一张 GPU 的单机 DDP，例如：

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 GRAD_ACC=8 \
VAL_DATASET=data/fireredasr_llm/dev.jsonl \
bash examples/fireredasr_llm/sft.sh \
    /models/FireRedASR-LLM-L data/fireredasr_llm/train.jsonl
```

模板使用普通 padding，保持 `packing=false`、`padding_free=false` 和 `truncation_strategy=delete`。当前不覆盖量化、自动跨卡切分、Megatron 或推理。可设置 `ATTN_IMPL=sdpa` 使用 PyTorch SDPA。FlashAttention 路径依赖上游实现及硬件，默认的 eager/BF16 路径不需要安装 flash-attn。

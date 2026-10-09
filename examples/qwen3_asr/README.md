# Qwen3-ASR SFT

使用安装的 ms-swift 引擎训练 `Qwen/Qwen3-ASR-1.7B`、`Qwen/Qwen3-ASR-0.6B` 或对应的本地模型。默认对 LLM 做 LoRA，冻结音频编码器和投影层。模型与 tokenizer 继续使用官方实现。

## 环境

在项目根目录运行：

```bash
uv sync --locked --extra qwen3-asr
```

需要 Python 3.11 和 CUDA GPU。默认 BF16 与 SDPA；安装兼容的 FlashAttention 后可设置 `ATTN_IMPL=flash_attention_2`。

## 准备数据

可从 AISHELL-1 或 LibriSpeech 选取 tiny 子集。原始 JSONL 每条记录为音频与转写：

```json
{"audio":"/data/AISHELL-1/wav/utterance.wav","text":"今天天气真好"}
```

数据格式由本示例的 Bash 脚本配置；中文和英文分别设置 `ASR_LANGUAGE=Chinese`、`ASR_LANGUAGE=English`：

```bash
ASR_LANGUAGE=Chinese bash examples/qwen3_asr/prepare_data.sh \
    data/raw/train.jsonl data/qwen3_asr/train.jsonl --limit 32
ASR_LANGUAGE=Chinese bash examples/qwen3_asr/prepare_data.sh \
    data/raw/dev.jsonl data/qwen3_asr/dev.jsonl --limit 8
```

准备后的训练样本包含语言前缀和音频占位符：

```json
{"messages":[{"role":"user","content":"<audio>"},{"role":"assistant","content":"language Chinese<asr_text>今天天气真好"}],"audios":["/data/AISHELL-1/wav/utterance.wav"]}
```

`prepare_data.sh` 给通用数据工具传入 `--prompt '<audio>'`、语言响应前缀和 `--response-prefix-pattern '^language [^<\r\n]+<asr_text>'`，匹配已有前缀时不重复添加。也可设置 `INPUT`、`OUTPUT` 和 `LIMIT` 替代位置参数，额外数据参数原样追加。

原始 JSONL 中的相对音频路径按输入文件所在目录解析，输出转换为绝对路径。直接提供这种 Swift 格式也可以。不要手工添加 EOS；裸转写应先经过数据脚本添加前缀。训练模板读取音频并重采样到 16 kHz。`MAX_LENGTH` 包括展开后的音频 token，超长样本默认丢弃，应根据音频时长调整。tiny 数据应提供独立验证集；没有验证集时，训练脚本默认切出训练数据的 1%。

## 单卡训练

脚本可从任意目录调用；相对的模型和数据路径按调用时的工作目录解析，含空格的路径需要加引号。

```bash
CUDA_VISIBLE_DEVICES=0 VAL_DATASET=data/qwen3_asr/dev.jsonl \
bash examples/qwen3_asr/sft.sh \
    Qwen/Qwen3-ASR-1.7B data/qwen3_asr/train.jsonl
```

或只设置环境变量，使用本地模型：

```bash
MODEL_DIR="/models/Qwen3-ASR-0.6B" \
DATASET="/data/tiny/train.jsonl" VAL_DATASET="/data/tiny/dev.jsonl" \
OUTPUT_DIR="/data/training runs/qwen3_asr" \
bash examples/qwen3_asr/sft.sh
```

默认每卡 batch 为 1、梯度累积 16、LoRA rank 8、alpha 32、学习率 `1e-4`。环境变量 `TRAIN_BS`、`EVAL_BS`、`GRAD_ACC`、`LEARNING_RATE`、`NUM_TRAIN_EPOCHS`、`MAX_LENGTH`、`TORCH_DTYPE`、`ATTN_IMPL`、`TUNER_TYPE`、`SPLIT_DATASET_RATIO` 可覆盖对应配置。`MODEL_DIR`、`DATASET`、`VAL_DATASET`、`OUTPUT_DIR` 分别指定模型、训练数据、验证数据和输出目录。

全部模型选择和训练默认值都在 `sft.sh` 中，包括 `model_type`、模板、冻结范围和 gradient checkpointing；通用 Python 启动器只转交 Swift 原生参数。设置 `DRY_RUN=1` 可预览传给 Swift 的参数，例如：

```bash
DRY_RUN=1 bash examples/qwen3_asr/sft.sh \
    /models/Qwen3-ASR-0.6B data/qwen3_asr/train.jsonl
```

预览不会加载模型，也不包含环境变量；正式运行时保留 Bash 配置并取消 `DRY_RUN`。直接调用 CLI 时需把 `--dry-run` 紧跟在 `audio-swift sft` 后。脚本的额外训练参数保留原始顺序，使用 Swift 原生的 `--val_dataset`、`--output_dir` 等下划线拼写。

脚本末尾的参数按原始顺序传给 Swift，可覆盖 recipe 默认值。例如四步训练、验证和保存：

```bash
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 \
VAL_DATASET=data/qwen3_asr/dev.jsonl OUTPUT_DIR=output/qwen3_asr_smoke \
bash examples/qwen3_asr/sft.sh \
    /models/Qwen3-ASR-0.6B data/qwen3_asr/train.jsonl \
    --max_steps 4 --eval_steps 4 --save_steps 4 --logging_steps 1
```

检查 loss、验证日志及 `checkpoint-4` 是否保存。短程训练只验证流程；需要实际恢复一次，才能确认可续训：

```bash
VAL_DATASET=data/qwen3_asr/dev.jsonl OUTPUT_DIR=output/qwen3_asr_smoke \
bash examples/qwen3_asr/sft.sh \
    /models/Qwen3-ASR-0.6B data/qwen3_asr/train.jsonl \
    --resume_from_checkpoint /absolute/path/to/checkpoint-4 \
    --max_steps 5 --eval_steps 1 --save_steps 1 --logging_steps 1
```

使用日志中实际的 checkpoint 路径；Swift 可能在输出目录下创建带版本号的子目录。LoRA checkpoint 仍依赖原始基础模型。

## 全量与多卡

全量训练 LLM，同时保持音频编码器和投影层冻结：

```bash
TUNER_TYPE=full LEARNING_RATE=1e-5 OUTPUT_DIR=output/qwen3_asr_full \
VAL_DATASET=data/qwen3_asr/dev.jsonl \
bash examples/qwen3_asr/sft.sh \
    /models/Qwen3-ASR-1.7B data/qwen3_asr/train.jsonl
```

若需训练音频编码器和投影层，再追加 `--freeze_vit false --freeze_aligner false`，并重新评估显存。可选的单机多卡 DDP：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 GRAD_ACC=4 \
VAL_DATASET=data/qwen3_asr/dev.jsonl \
bash examples/qwen3_asr/sft.sh \
    /models/Qwen3-ASR-1.7B data/qwen3_asr/train.jsonl
```

这里总 batch 为 `每卡 batch × 进程数 × 梯度累积`；多卡 4 × 1 × 4 与默认单卡 1 × 1 × 16 相同。

# 单卡全量 SFT 基准（2026-10-10）

**四组全量训练及对应完整测试集评测均已通过，并独立复算全部测试句子的 S/D/I。** 本次分别从同一模型初始权重独立训练 AISHELL-1 和 LibriSpeech，各 1 epoch，没有串联训练、tiny 子集或步数上限。

| 语料 | 全量训练条数 | 实测音频小时 | 每模型 optimizer steps | 完整验证集 | 完整测试集 |
| --- | ---: | ---: | ---: | ---: | --- |
| AISHELL-1 | 120,098 | 150.6929 | 7,507 | 14,326 | test：7,176 |
| LibriSpeech 100+360+500 | 281,241 | 961.0529 | 17,578 | dev-clean+dev-other：5,567 | test-clean：2,620；test-other：2,939 |

LibriSpeech 三个训练划分分别为 28,539、104,014、148,688 条；表中时长按音频实测求和，保留官方划分和转写。数据检查覆盖样本 ID、重复、训练/验证/测试隔离及音频可访问性。训练目标仅对 AISHELL 去空白、LibriSpeech 转小写；测试 reference 保留官方原文。

每组使用 **1 张 NVIDIA A800-SXM4-80GB**，BF16、LLM SDPA。引擎为 PyPI `ms-swift==4.5.3`，通过 `uv sync --locked` 安装；Python 3.11.11、PyTorch 2.10.0、Transformers 4.57.6、PEFT 0.18.1、datasets 4.8.4、Accelerate 1.12.0、qwen-asr 0.0.6。项目训练源码为 `4d032cc69910ef4184ffd36b0577dc2582d046b2`；FireRed 官方实现固定于 `834635e4cf277ed8ca92049fc375b17c3dc20748`。

| 实际训练配置 | Qwen3-ASR-0.6B | FireRedASR-LLM-L + Qwen2-7B-Instruct |
| --- | --- | --- |
| Swift tuner | `lora` | `lora_llm` |
| LoRA rank / alpha / 实际 dropout | 8 / 32 / 0.05 | 8 / 32 / **0.0** |
| 音频编码器 / 投影层 | 冻结 | 全量训练，另存 `vit.safetensors` |
| 每卡 batch × 梯度累积 | 16 × 1 | 8 × 2 |
| 梯度 checkpointing | 关闭 | 开启，`use_reentrant=false` |
| LLM / 音频编码器 / 投影层学习率 | 1e-4 / — / — | 1e-4 / 1e-5 / 1e-5 |

两者 LoRA 都作用于 LLM 的 q/k/v/o 和 gate/up/down 线性投影层。共同使用 AdamW、weight decay 0.1、cosine 调度、warmup ratio 0.05，有效 batch 16，seed/data_seed 20261010，最大序列长度 2048，`strict=true`、`drop_last=false`。每 1,000 步保存，最多保留 2 个 checkpoint；使用完成 1 epoch 的最后一步 checkpoint，`load_best_model_at_end=false`，不按测试集选模型。

**FireRed 参数解释：** `args.json` 的 `lora_dropout=0.05`、`freeze_vit=true`、`freeze_aligner=true` 并非该 custom tuner 的实际行为。Swift 4.5.3 的 `lora_llm` 构造 PEFT 配置时未传 dropout，实际 adapter 配置为 0.0；它同时显式解冻音频编码器与投影层。以上表格以 adapter 配置、tuner 实现和保存/重载权重检查为准。

测试均为 batch 16、greedy（不采样、单 beam）、最多新增 512 tokens。Qwen 使用官方 SDK、空 context 和语料对应语种；FireRed 使用固定转写提示、真实音频长度与显式 attention mask，加载官方预训练增量后再加载本次 LoRA 和音频侧权重。模型输入不含 reference；reference 仅在生成完成后用于评分。

| 模型 | AISHELL-1 test CER↓ | LibriSpeech test-clean WER↓ | LibriSpeech test-other WER↓ |
| --- | ---: | ---: | ---: |
| Qwen3-ASR-0.6B | **1.6656%** | **1.9971%** | **4.4249%** |
| FireRedASR-LLM-L | **0.8113%** | **1.5879%** | **3.5362%** |

六项测试均覆盖完整官方 ID，原始 reference 逐条一致，生成截断数为 0。CER 对 reference/hypothesis 同施 NFKC、小写、去 Unicode 标点和空白，不做 ITN；WER 使用 Transformers 4.57.6 的 `EnglishTextNormalizer({})`（含其数字等正规化，拼写映射为空）。指标为语料级 `100×Σ(S+D+I)/Σ参考字数或词数`，不是逐句比例平均；另保留不做 ITN 的 basic WER。

`output/asr_benchmark_20261010/` 下的 `experiment-plan.json`、`data_inventory.json`、`performance.json` 和逐组工件记录源码/脚本、manifest、预测及 adapter 的 SHA-256；FireRed 还记录音频侧权重 hash。大型原始模型文件主要记录大小和修改时间，未对全部基座权重或全部 WAV 做内容 hash，不能宣称完整字节级复现校验。

FireRed AISHELL 的首次测试因 fresh `--resume` 状态判断失败；修复解码启动逻辑并通过 11 项回归检查后，复用已完成的 `checkpoint-7507` 单独测试，没有重新训练或替换训练结果。过程保存在 `evaluation-retry.json`。

这组数字描述上述训练和评分条件；未评测未微调 baseline，不主张 SFT 带来提升，也不据不同模型规模、音频冻结范围和 dropout 的结果推断架构优劣。


[机器可读结果与运行时哈希](asr-benchmark-20261010.json)包含完整精度指标、错误计数、数据划分、实际 adapter 配置和任务记录。训练入口为 [Qwen Bash 示例](../examples/qwen3_asr/sft.sh) 与 [FireRed Bash 示例](../examples/fireredasr_llm/sft.sh)；完整日志、评分脚本、逐句预测和 checkpoint 保存在本地 `output/asr_benchmark_20261010/`。GitHub 保存实验记录与汇总结果。

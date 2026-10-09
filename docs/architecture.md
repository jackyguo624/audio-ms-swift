# 架构与第一版边界

本项目把 Swift 作为训练引擎安装，把语音模型、特征和任务实现放回项目自身。PyTorch 是计算后端，Swift 提供通用训练流程。训练实例的选择和参数由 Bash 示例配置，通用 Python 代码无需随着模型数量增加而扩展分支。

## 责任划分

| 层 | 本项目维护的内容 | 外部依赖 |
| --- | --- | --- |
| Bash 示例 | 模型 ID、模板、LoRA/冻结范围、学习率、batch、验证集、输出目录、plugin 路径 | 项目 CLI |
| 通用数据工具 | 音频与转写 schema、路径解析、按传入 prompt/prefix 转换格式 | 音频读取库 |
| 音频与模型 | 特征、CMVN、语音编码器、投影层、音频和文本对齐、模型封装 | PyTorch、Transformers、官方模型实现 |
| 模型 plugin | Swift 模型/模板/数据集注册、该模型要求的约束 | 本项目模型、Swift API |
| 通用启动器 | 原样转交 Swift 参数与执行训练进程 | 已安装的 ms-swift |
| 通用训练 | 优化器、混合精度、梯度累积、DDP、checkpoint | ms-swift / PyTorch |

模型和音频模块不能导入 `swift`。plugin 可以导入模型模块并调用 Swift 注册 API；通用启动器不导入具体 plugin，不检查具体模型的权重布局，也不定义模型列表或训练超参数。新增模型时，新增示例配置和必要的 plugin，避免改动通用 CLI 与启动器。

## Bash 配置与通用入口

`examples/qwen3_asr/sft.sh` 和 `examples/fireredasr_llm/sft.sh` 显式设置此前验证过的训练参数，末尾原样追加用户参数。冻结范围、batch、梯度累积、学习率、attention、gradient checkpointing、验证切分与输出目录都能在脚本中看到。

`audio-swift sft` 只接收 Swift 原生参数并调用安装的 Swift CLI。它不选择 recipe，也不注入模型或训练默认值。`audio-swift sft --dry-run ...` 可预览命令，`--dry-run` 放在 `sft` 之后第一个位置；这个预览只生成命令，不加载模型或验证 plugin 的模型约束。

通用 `prepare-data` 接收 `--prompt`、`--response-prefix` 和 `--response-prefix-pattern` 等格式化参数，不知道 Qwen 或 FireRed 的名称。Qwen 的 Bash 数据脚本传入 `<audio>`、语言前缀及前缀匹配规则；FireRed 的脚本传入转写提示词，不添加响应前缀。模型专用字符串留在对应示例中。

## 模型 plugin

Qwen3-ASR 复用 qwen-asr 与 Swift 现有的模型支持，因此示例只配置 Swift 的 `qwen3_asr` 模型和模板。LLM 与 tokenizer 继续使用官方实现。

FireRedASR-LLM 的注册实现位于 `audio_swift.plugins.fireredasr_llm`。示例中的 `plugin.py` 是外部插件入口，Bash 通过 `--external_plugins` 加载它；包内模型与音频代码仍独立于 Swift。FireRed 的源码准备脚本也放在自己的示例目录中。

FireRed 模型布局、单音频输入、padding、截断与 checkpoint 范围等约束由该 plugin、模板或模型封装保证，不写进通用启动器。示例通过环境变量把本地训练与验证 manifest 提供给 plugin 注册的数据集别名。默认 `lora_llm` 对 LLM 做 LoRA、对音频编码器和投影层做全量训练；当前支持 LoRA 适配器保存，不提供 full checkpoint 导出。

外部源码、权重、数据与训练产物不提交到 Git。Swift 引擎按 PyPI 发布版本 [`ms-swift==4.5.3`](https://pypi.org/project/ms-swift/4.5.3/) 安装；独立的 FireRed 官方源码仍固定到自己的提交。uv 管理环境与依赖解析，`uv sync --locked` 默认安装本项目当前两套模型的 Python 依赖，无需按模型分别安装。发布版本、`uv.lock` 与 FireRed 源码提交组成复现基线。升级 Swift 或 FireRedASR 时，应重新检查模型注册、数据对齐、保存和恢复。

## 第一版范围与后续扩展

第一版提供 Qwen3-ASR 和 FireRedASR-LLM-L 的 SFT 示例。FireRed 使用普通 padding、单个音频输入和单设备或每进程单设备的 DDP；packing、padding-free、量化、自动跨卡切分和统一推理接口不在此版范围内。

codec、TTS、评测和推理是后续扩展方向，目前尚未实现。自回归 codec-token TTS 可以优先沿用类似的外部 plugin 接口。flow matching、diffusion 或 vocoder 应先确认 batch、loss 和训练步骤是否能通过 Swift 接入，再决定是否增加其他引擎。第一版保留一个通用 Swift 启动器。

## 验证要求

每次调整 Bash 配置、plugin 或引擎版本后，检查 tiny 数据转换、完整训练参数、真实 GPU 前后向、checkpoint 保存和恢复。短程 smoke run 可使用不同的 32 条训练样本、8 条验证样本和 4 个优化步，只验证训练流程，不判断精度或收敛。此前版本的 GPU 记录属于历史证据，修改配置边界后需要重新验收。

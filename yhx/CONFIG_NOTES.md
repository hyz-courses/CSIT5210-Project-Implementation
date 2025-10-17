# 配置与脚本改动备忘（保留被删内容的说明）

为便于后续检查，这里记录近期对脚本/配置做出的改动以及原本准备删除但无法以注释保留的内容。

## 脚本

- `run_LLM2Rec_CSFT.sh`
  - 保留为注释的命令：
    - `cp ${model_path}/*token* ./output/Qwen2-0.5B-CSFT-${category}/`
    - `latest_ckpt=$(ls -d ./output/Qwen2-0.5B-CSFT-${category}/checkpoint-* | sort -V | tail -n 1)`
    - `cp ${model_path}/*token* ${latest_ckpt}/`
  - 说明：保留为注释，避免路径不一致或版本不匹配；实际保存由 `run_csft.py` 的 `tokenizer.save_pretrained(...)` 完成。

- `run_LLM2Rec_IEM.sh`, `run_LLM2Rec_MNTP.sh`, `run_LLM2Rec_SimCSE.sh`
  - 关系说明：`IEM.sh` 是一个工作流控制脚本，用于按顺序自动执行 MNTP 和 SimCSE 两个训练阶段。`MNTP.sh` 和 `SimCSE.sh` 则是用于独立、灵活地运行这两个子阶段的脚本，更便于调试。修复了 `run_unsupervised_SimCSE.py` 中的错误后，`IEM.sh` 已能正常使用。
  - `IEM.sh` 更新：为其中的 `run_mntp.py` 和 `run_unsupervised_SimCSE.py` 命令分别添加了 `--run_name "mntp_from_iem"` 和 `--run_name "simcse_from_iem"` 参数，以便在 wandb 中清晰追踪其训练过程。
  - `IEM.sh` 中保留的注释：
    - `cp ${model_path}/*token* ...`
    - 说明：保留为注释；MNTP/SimCSE 链路会自行保存/加载 tokenizer，不需手工拷贝。

- `llm2rec/run_csft.py`
  - 新增：训练结束后执行 `tokenizer.save_pretrained(output_dir)`，并尝试保存到最新检查点目录。
  - 目的：确保 CSFT 输出中始终包含与训练模型一致版本的 tokenizer。

- `llm2rec/run_unsupervised_SimCSE.py`
  - 删减：移除了 PEFT/LoRA 相关的逻辑，包括 `initialize_peft` 函数、`CustomArguments` 中的 `lora_r` 和 `lora_dropout` 参数，以及 `main` 函数中动态应用 PEFT 的分支。
  - 新增：取消了 `for param in model.model.parameters(): param.requires_grad = True` 的注释。
  - 目的：简化 SimCSE 阶段的训练脚本，移除冗余的 PEFT 逻辑，确保在该阶段训练并保存的是完整的模型，而非仅 LoRA 适配器。这解决了原始脚本仅保存适配器导致后续阶段无法加载完整模型的问题。

- `run_LLM2Rec_SimCSE.sh` & `run_LLM2Rec_SimCSE_original.sh`
  - 新增：创建了 `run_LLM2Rec_SimCSE_original.sh` 作为原始脚本的运行文件，其输出目录指向 `...-original` 以作区分。
  - 新增：为两个脚本的 `torchrun` 命令分别添加了 `--run_name "simcse_modified"` 和 `--run_name "simcse_original"` 参数。
  - 目的：为了在 Weights & Biases (wandb) 上进行清晰的训练过程对比，通过指定不同的 `run_name` 来区分两个版本的实验。

## 配置（JSON 无法添加注释，改动在此记录）

- `llm2rec/train_mntp_config.json`
  - 新增：`"tokenizer_name": "Qwen/Qwen2-0.5B"`

- `llm2rec/train_simcse_config.json`
  - 移除：`"tokenizer_name"` 字段（此前曾临时添加，已撤回）。
  - 说明：当前脚本解析该字段会报错；SimCSE 会从 `model_name_or_path` 加载模型，再使用内部的 tokenizer；同时 MNTP 检查点会自带保存的 tokenizer，链路自洽。

如需恢复上述被注释的命令或字段，请在对应脚本/配置中解除注释或重新添加字段，并确保路径与版本匹配。
# Commerce-RSI

在 Amazon-C4 商品检索任务上，让研究 Agent 生成负采样程序，根据真实训练反馈更新自身 LoRA 参数。业务训练器、候选库、初始化和预算固定；目标是研究 `A0 → A1 → A2` 的经验与参数学习。

当前已跑通真实 `A0 → A1 → A2` 控制流程：OpenRSI 选择进化操作，Qwen 生成采样程序，在内核沙箱中执行，固定业务模型真实训练并反馈奖励，然后更新 1,843,200 个 LoRA 参数。两次更新均已保存、磁盘重载并核验参数哈希，第二次恢复了优化器状态；后续独立 rollout 核对了 A1/A2 身份。单卡训练峰值约 8.11 GiB。已完成数据审计、固定候选库、业务基线和有界对照；正式 meta-test 仍未评估。

同一 meta-dev Toys 任务、相同种子、每代 4 次候选的试跑结果如下。程序库分别从空库开始，每代都保留自己的执行反馈；评估期间没有更新 Agent 参数。

| Agent | 合法完成 / 候选数 | 最佳验证完整 MRR |
| --- | ---: | ---: |
| A0 | 2 / 4 | 0.008236 |
| 一次更新后冻结 A1 | 0 / 4 | 未测得：全部执行失败 |
| 两次更新后冻结 A2 | 3 / 4 | 0.008932 |

这些结果只有一个任务、一个种子，A1 未显示收益，A2 的小幅数值差异不足以证明 RSI。训练任务上的固定 A0 与递归流程均为 2/4 个候选成功，最佳 MRR 同为 0.013229。各方法前置训练算力不同，不能声称等总算力优胜。第一版接口的 6 次失败、重试和所有额外成本均保留，详见 [实验记录](docs/EXPERIMENTS.md)；当前主要瓶颈是程序合法率与业务反馈质量。

验证结果：86 项测试通过，另含 9 个子用例；源码编译和依赖一致性检查通过。实验已结束，没有后台训练。后续应先提升合法程序率，再进行多任务、多种子和等总预算比较；正式最终测试继续封存。

原始数据为 21,223 条合成购物查询、1,058,417 个商品；这不是线上搜索日志。本项目使用自定义训练协议，不复现 Amazon-C4 原评测协议。`ori_review/ori_rating` 不进入模型或 Agent；meta-test 和 final-evaluation 保持封存。

## 在租用服务器重建

所有操作只在服务器 `/root/autodl-tmp/commerce-rsi` 内执行。服务器已有 CUDA PyTorch，独立 venv 只读复用它；已有 `.venv` 可直接使用，无须重新创建。

```bash
cd /root/autodl-tmp/commerce-rsi
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -e '.[dev,agent,upstream]'
.venv/bin/python scripts/preflight.py
.venv/bin/python scripts/bootstrap_upstream.py
```

bootstrap 固定官方 commit `71ae803a035d5e3b78c19fa49ed9f67d0550cbaa`，校验仓库内 patch SHA256 后应用。已有 checkout 必须匹配来源、commit、开发分支；不覆盖已有修改。只检查可加 `--check-only`。

官方 Hugging Face 元数据通过服务器现有 `127.0.0.1:7890` 代理读取，大文件可用镜像传输，按官方 revision、字节数和摘要验证，不信任镜像自行提供的校验值。代理仅对下面命令生效：

```bash
HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 \
  .venv/bin/commerce-rsi download --output artifacts/data/raw \
  --revision 39322697749a88d179f88d322a2fe4765b655c98 \
  --transfer-endpoint https://hf-mirror.com
.venv/bin/commerce-rsi audit --raw artifacts/data/raw --output artifacts/data/audit
.venv/bin/commerce-rsi prepare --raw artifacts/data/raw --output artifacts/data/prepared \
  --candidate-count 30000 --seed 20261007 --minimum-task-queries 150
HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 \
  .venv/bin/python -m commerce_rsi.agent_training.model_source \
  --output artifacts/models/Qwen2.5-Coder-3B-Instruct --transfer-endpoint https://hf-mirror.com
```

下载会验证已存在的正确文件。准备数据和导出任务应使用新输出目录；保留旧 manifest 与实验日志，避免改变进行中实验的输入。

```bash
.venv/bin/python - <<'PY'
from commerce_rsi.openrsi_adapter.upstream import export_task_package, read_task_package
for split in ("meta_train", "meta_dev"):
    output = f"artifacts/tasks/{split}_v2.parquet"
    print(export_task_package("artifacts/data/prepared", output, split))
    print("validated task count:", len(read_task_package(output)))
PY
.venv/bin/python -m pytest -q
```

任务输入包含上游 `prompt/metadata` 结构、本项目 sampler executor 标记和 prompt schema 版本，训练/开发分别导出。当前 schema v2 明确 batch 为字典列表、结果为嵌套列表、禁止 numpy；仅澄清接口，没有提供采样算法。旧 pilot 的失败和成本保留，新版本需另存任务输入与实验目录。程序库、operator 选择和经验卡实际调用固定 OpenRSI 源码；Parquet 导出本身不代表训练接通。

## 运行有界实验

业务模型校准只用 meta-dev。每个策略重置相同初始化和 batch 顺序，保存配对种子、指标、负例抽检和成本：

```bash
.venv/bin/python scripts/run_business_pilot.py --prepared artifacts/data/prepared \
  --output artifacts/business_pilot/reproduce_shared_100 --task-id toys_7a17d0 \
  --shared-towers --steps 100 --seeds 17 29 43
```

正式 Agent rollout 前冻结 [pilot 配置](configs/pilot.json) 与 [实验协议](docs/PROTOCOL.md)。预试验发现 MRR@10 反馈常为零，因此主奖励为完整候选排名的未截断 MRR；仍报告 MRR@10、Hit@10、NDCG@10。以下是有界两轮控制路径命令，成功与否以日志为准，每次使用新输出目录：

```bash
.venv/bin/python -m commerce_rsi.recursion.pilot \
  --package artifacts/tasks/meta_train_v2.parquet \
  --model artifacts/models/Qwen2.5-Coder-3B-Instruct \
  --output artifacts/recursion/reproduce_pilot_v2 \
  --task-id garden_03c966 --mode recursive --rounds 2 --max-attempts 3 \
  --group-size 2 --business-steps 100 --max-new-tokens 768 --seed 17
```

`--mode fixed` 保留程序库和执行反馈而冻结 Agent；`--mode evaluate` 只接收 meta-dev，可通过 `--input-adapter` 加载某代已保存 adapter。正式能力比较还需统一新任务搜索预算、记忆条件与总体成本；不能把 pilot 输出直接当作已完成公平 RSI 对照。

输出目录包含 `pilot.json` 总状态/谱系/成本，`programs.sqlite` 上游程序库，各轮 `rollout.json`、`executions.json`、`group_report.json`，候选的 `business/run.json`，以及更新目录的 `update.json` 和成功保存的 adapter。两次真实更新、磁盘重载及参数 hash 核验均通过，才可报告控制路径完成；即使完成也不等于能力递归提升。

单卡后端采用 Transformers/PEFT LoRA 与同策略 RLOO，顺序执行生成、业务训练、参数更新。它不是上游 8 × H200 的 SLIME/Megatron/GSPO 复现。候选源码只在已测试的 Landlock/seccomp 沙箱内运行，失败、非法负例、超时和截断都会记为失败。

## 证据与归属

- [真实数据审计](docs/DATA_AUDIT.md)、[固定协议](docs/PROTOCOL.md)、[复用差异](docs/UPSTREAM_CHANGES.md)、[实际隔离验证](docs/ISOLATION_FEASIBILITY.md)、[审查及负例抽检](docs/REVIEW.md)。
- 原始运行入口：`artifacts/recursion/pilot_v2/pilot.json`；对照汇总：`artifacts/controls/comparison.json`。运行 `.venv/bin/python scripts/report_experiments.py` 可从记录重建汇总，无须重新训练。
- 来源版本、大小、SHA256：`artifacts/data/raw/source_manifest.json` 与模型目录 `source_manifest.json`；划分和候选池：`artifacts/data/prepared/manifest.json`。
- Draft/Improve/Debug/Crossover、程序库和经验机制来自 [OpenRSI](https://github.com/FrontisAI/OpenRSI)，不作为个人原创。上游原始材料为 CC BY-NC 4.0，第三方组件保留各自许可证。
- Built with Qwen。实际选用的 **Qwen2.5-Coder-3B-Instruct 使用 Qwen Research License，仅限非商业研究/评估**，不是默认 Apache 2.0；许可证随模型下载保存。模型/adapter 分发还须保留相应声明。

不提交原始大数据、模型权重、环境凭证或实验大文件。不声称 SOTA、线上 CTR/GMV 收益或已证实 RSI 假设。

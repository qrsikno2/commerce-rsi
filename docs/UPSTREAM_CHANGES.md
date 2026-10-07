# OpenRSI 来源、复用和单卡适配

本记录对应 2026-10-07 在租用服务器上的实际源码检查与接口测试。

## 固定来源

- 官方：<https://github.com/FrontisAI/OpenRSI>
- 实际 `git ls-remote ... HEAD` 与 clone HEAD：`71ae803a035d5e3b78c19fa49ed9f67d0550cbaa`。
- checkout：`third_party/OpenRSI`；独立开发分支：`commerce-rsi-adapter`。
- 上游原始材料许可证为 CC BY-NC 4.0；AIRA-Dojo、SLIME 等组件遵守各自 LICENSE/NOTICE。项目需要保留上游署名，不将继承模块视为个人原创。
- `third_party/` 不进入本项目 Git。可复现的修改保存在 `patches/openrsi-single-card.patch`。
- patch SHA256：`5825091d125cd08b37e81b4df0609fe37c09735aaea06521b31f374f72cd95d9`。

重建 checkout：

```bash
git clone https://github.com/FrontisAI/OpenRSI.git third_party/OpenRSI
git -C third_party/OpenRSI checkout -b commerce-rsi-adapter 71ae803a035d5e3b78c19fa49ed9f67d0550cbaa
git -C third_party/OpenRSI apply ../../patches/openrsi-single-card.patch
```

## 实际复用的最小接口

`src/commerce_rsi/openrsi_adapter/upstream.py` 动态加载固定 checkout 中的真实模块；不是同名本地替代实现。加载前检查 Git HEAD，`upstream_manifest()` 同时给出实际源码 SHA256。

| 上游文件/接口 | 本项目用途与边界 |
| --- | --- |
| `OpenMLE-ERL/RL/program_database.py:Program` | 记录源码、执行状态、真实验证分数、奖励、父程序、operator、checkpoint 等 metadata |
| 同文件 `ProgramDatabase.add/get_all/get_by_id/merge_program_metadata/close` | 实际 SQLite 程序库；`max_per_task=0` 保留失败与完整祖先链 |
| 同文件 `AIRAInferenceEvoSearch.select` | 实际 Draft/Improve/Debug/Crossover 选择、父节点效用采样、上游经验记忆；默认 `rl_mixed` |
| `OpenMLE-ERL/RL/airaevo_experience.py:build_experience_card` | 实际生成并持久化上游经验卡片；父节点变化、失败、运行时间和程序关系 |
| `OpenMLE-Gym/docs/task-inputs.md` 与 `openmle_gym/prepare_input.py` 的 `prompt` / `metadata` schema | 自定义任务输入的列名、messages 结构、绝对 `data_dir`、uuid 与指标方向 |

`SamplerSearch.next_prompt()` 返回 `SearchRequest`，包含 `messages`、`operator`、`parent_ids` 与 selection metadata；`record()` 接收可信训练器算出的 reward-validation MRR（完整候选排名、不截断）。成功的零分与失败分开，失败必须 `score=None`，奖励为零。`record()` 不接受候选程序自报的分数。MRR@10、Hit@10、NDCG@10 仍作为辅助报告指标。

本桥接未调用完整的 `OpenMLE-Evo/third_party/aira-evo/src/dojo/solvers/evo/evo.py` 主循环。检查确认该实现已有 Draft/Improve/Debug/Crossover、islands 与经验机制；本项目复用的是上游已经移植到 ERL 程序数据库的对应 `AIRAInferenceEvoSearch`。不能宣称个人重新发明这些能力，也不能称当前桥接复现了全部 Evo 配置。

## 上游仅两处修改

1. `ProgramDatabase.__init__` 建立实例自己的 `threading.local()`。原类变量 `_local` 导致同线程打开不同 SQLite 文件时复用第一个连接，可能污染独立任务/对照程序库。测试同时打开两个数据库，并确认分别持久化和重载。
2. `AIRAInferenceEvoSearch.__init__` 新增可选 `prompt_builder` 参数，`select()` 优先调用该 renderer。未提供时保持上游原 prompt builder。这样实际保留原 operator/parent/experience 逻辑，同时使用 sampler-only prompt，避免默认要求候选训练完整模型并输出 `submission.csv`。

修改后的源码 SHA256：

- `OpenMLE-ERL/RL/program_database.py`：`55a1b708ccc3f7a5cadbcdb74b3f3c542bc44ec2fa3305dcd2360b868e6b2d3c`
- 未修改的 `OpenMLE-ERL/RL/airaevo_experience.py`：`a7675a29fad120154e5337a6824b9b4d2f415284cb7e8c8e6c9c88c9ea7781f2`

## 任务输入与实际消费者

`export_task_package(prepared_dir, output, outer_split)` 使用已固定的 prepared manifest，只导出 `meta_train` 或 `meta_dev`，分别生成独立 Parquet。字段含原 schema 与 `executor=commerce-rsi-sampler-only-v1`、固定上游 commit、catalog SHA256、prepared manifest SHA256。提示不包含评价查询、答案、评论或评分字段；只有任务类别和接口描述。

`read_task_package(path)` 是本地 sampler 后端的实际入口：检查 executor、commit、固定 system prompt、manifest hash、重复 UUID 与单一外层 split。task name、outer split、category、catalog hash、protocol、确定性 UUID 均须与实际 manifest 一致；任务路径解析后须准确位于 prepared/tasks 内，不能靠更改 Parquet metadata 将 dev 改标为 train。`primary_metric=mrr` 固定未截断反馈。pilot 不支持导出/消费 meta-test，避免协议和 checkpoint 冻结前误用。Parquet 导出与加载的结果，必须和后续 Agent 生成、可信训练/评分、参数更新分别报告。

没有直接使用 Gym `prepare_input()`，因为它固定读取 `selfvalid0327-sft4.json` 模板，要求可改动完整训练程序、写文件、输出 `submission.csv` 和自报验证指标。原 Evo 的 `/api/v1/jobs` sandbox dispatcher 也按完整训练程序执行。这里的 Parquet 是同结构的自定义 sampler 消费输入，并非可不经适配直接交给原 dispatcher 的完整 Gym task package。

原消费者路径经源码确认：Evo 读取 `OPENMLE_EVAL_DATA`；RL 用 `PROMPT_DATA` 与 `EVAL_PROMPT_DATA`；parallel/evolutionary SFT rollout 各有 task Parquet 入口。SFT 权重训练消费 `id/messages` 轨迹，不直接消费 task Parquet。

## 奖励、轨迹和训练后端差异

上游 `generate_mle.generate(args, sample, sampling_params, evaluation)` 与 `reward_func(args, sample, **kwargs)` 依赖 SLIME/SGLang，并将执行请求发给 sandbox service。`reward_func_utils.py` 另含全程序 hack-check API、leaderboard reward transform、self-validation gap penalty。这些没有直接接入本地 sampler 实验；这里用固定业务训练器的实际 reward-validation 未截断 MRR，暂不应用 leaderboard 或自报验证分数的 shaping。主反馈在正式 Agent rollout 前按 meta-dev 预试验发现的 MRR@10 零分稀疏问题调整并冻结，未查看最终测试。

已阅读 SFT selection：`select_evolutionary.py` 构造 Draft/Improve/Crossover + Debug 后继片段，检查相对父程序的成绩，并要求 medal gate；多步成功片段还需要因果继承 annotation。Amazon-C4 自定义任务没有 Kaggle medals，因此不能直接伪造 medal 后使用其筛选。上游因果筛选、Debug 恢复和去重是可复用的已有能力，不是当前已证实的项目贡献。失败经历保留在程序库用于 Debug；重复失败程序不能自动当正向 SFT 标签。

上游 RL 文档只验证了 8 × H200 单节点异步 profile。四份 launcher 均面向 8 卡或更多卡，默认模型是 30B/35B MoE；没有本 release 单 A100 40GB + 3B LoRA 的验证配置。默认异步需要独立 actor/rollout GPU 配额。`generate_mle.py` 导入 SLIME `Sample` / `GenerateState`，原优化器与权重同步属于 Megatron/SGLang 路径。

因此本项目的 Transformers/PEFT 单卡顺序 rollout/LoRA 更新是明确的新后端适配，不能声称复现了原 SLIME/Megatron/GSPO 训练实现。其真实训练、保存重载和代际效果须以运行日志另行证明。这里只完成源码/接口复用与安全执行验证，没有通过源码阅读证明单卡训练一定可行。

## 已执行的接口验证

```bash
.venv/bin/python -m pytest tests/test_sandbox.py tests/test_upstream.py -q
```

当前结果为 **25 passed**（2026-10-07）。包含实际上游四种 operator、父节点/失败选择、经验卡持久化、两个数据库隔离/重载、Parquet round-trip/独立输入/manifest 校验、拒绝 dev 改标 train 和 symlink 越界，以及真实内核 sandbox 测试。测试中 0.2/0.3 为合成执行分数，不是 Amazon-C4 实验指标。真实业务与 Agent 结果由 `docs/EXPERIMENTS.md` 记录。

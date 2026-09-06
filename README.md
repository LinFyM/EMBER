# EMBER

EMBER研究能否把exact task language和一条或多条action-hidden教学视频，在rollout前一次性编译为冻结π0.5 source policy的
一套完整task-conditioned LoRA，使policy从未见初始化闭环完成任务。唯一正式性能目标为validation8 strict paired
single-checkpoint correct **>145/400**，同时满足相邻/跨视频稳定、breadth、四suite与Goal/Long贡献及最终视频因果controls。

当前候选是**分层局部关系视频到完整LoRA Writer**：冻结图文prefix → 单probe Action Expert共享观察Meta-LoRA → 保留18层×50-horizon
的窗口内帧对软对应 → 两端的内容/相对对应模式消息 → 逐帧聚合与同型block堆叠 → 多视频集合共同读取 → 原生坐标条件MLP生成全部38-target A/B。
9月7日数学设计已收口；owner随后明确授权接管者理解后连续实现、验证、训练、评测和证据支持的修正。
当前主线为Writer与读取侧Meta fresh端到端联合训练；尚无新架构性能证据。
准确授权见 [progress.md](progress.md)。

## 接手阅读

1. [AGENTS.md](AGENTS.md) 与 [长期要求](docs/current_owner_requirements.md)：目标、方法边界、数据/评测/资源合同和协作方式。
2. [当前计划](task_plan.md)、[当前状态](progress.md)、[持久发现](findings.md)：区分待研究问题、已完成证据和执行授权。
3. [科学动机](docs/concept.md) 与 [完整新设计](docs/layered_relation_video_writer_design.md)：理解完整推导、shape、梯度算法、实现边界。
4. [分层历史](docs/research_history.md)：先读全局脉络，再按当前问题展开原评审、旧账本和formal evidence；不默认恢复旧实验。
5. 下方代码地图及设计§11。临时HANDOFF已消费；持续授权、实施状态和下一步以正式账本为准。

## 代码与运行入口

活动树保留共同基础设施，旧P/Q、Natural Program、bank compiler与Stage 0专用执行面已退役；新图入口为scripts/train_layered_writer.py，正在真实机制与成本验证。
旧实现与原始专家意见可由 `fcdb6e43706c5fcedf10eaa5d2d459602b263016` 恢复，具体索引见历史§9。

| 责任 | 当前代码 | 使用边界 |
|---|---|---|
| 新图过程/集合/坐标生成 | `src/ember/writer/relation.py`、`layered.py`、`coordinate.py` | 完整H消费后压缩，一次生成全部A/B |
| 原生图文prefix、KV、Action Expert层捕获 | `src/ember/ecp/policy_effects.py`、`observer.py` | 由`writer/native.py`接入Meta与R-leaf/observer分块VJP，真实验证进行中 |
| Meta-LoRA与执行LoRA | `src/ember/writer/meta_lora.py`、`src/ember/pi05_lora.py`、`batched_lora.py` | 观察Meta与执行adapter作用域分离，最终执行只装一套完整LoRA |
| functional query VJP与Writer重放 | `src/ember/writer/functional.py`、`replay.py` | 已有同condition query microbatch VJP；未实现跨condition批量VJP或新Meta重放 |
| 数据、视频、任务采样与GPU placement | `src/ember/writer/data.py`、`functional_data.py`、`task_schedule.py`、`task_execution.py` | 复用读取/调度，正式allowlist、真实K1/2/4和episode角色须为新run明确登记 |
| checkpoint、NUMA与分布式 | `src/ember/ecp/checkpoint.py`、`src/ember/writer/topology.py`、`src/ember/pi05_source_setup.py` | 新架构fresh/new schema；exact-resume锁world topology |
| source与task-local专家 | `src/ember/source_sft/`、`src/ember/expert_manifold/` | 保留source来源和训练侧容量参照；不能部署held字典 |
| 官方闭环评测 | `src/ember/pi05_eval/`、`pi05_evaluation.py`、`pi05_eval_contract.py`、`static_task_lora.py` | 动态队列/long-first/persistent workers；旧静态adapter格式只用于证据读取 |
| 训练侧行为诊断 | `src/ember/reward/` | 保留已有rollout/occupancy工具；不授权held reward梯度或生成LoRA后的task-local RL |

当前Python入口均可使用 `--help`：

```text
scripts/seal_pi05_source_corpus.py
scripts/seal_pi05_target_data.py
scripts/train_source_sft.py
scripts/train_task_experts.py
scripts/train_layered_writer.py
scripts/materialize_layered_writer.py
scripts/evaluate_source_sft_validation_loss.py
scripts/evaluate_pi05.py
```

`scripts/bootstrap_env.sh`与`scripts/zig-cxx`保留既有环境/编译职责，不为交接重复安装环境。新架构的过程模块、集合compiler、坐标decoder与
观察Meta梯度编排已接线，唯一物化/评测入口已集成；不把工程测试当作新模型科学通过。

多checkpoint/arm物化可在同一进程复用source policy，使用原入口的`--requests-json /absolute/requests.json`，
同时指定公共`--asset-root`与`--device`。JSON为请求列表，例如：

```json
[
  {"checkpoint": "/run/checkpoints/macro_00000048", "output": "/output/correct48",
   "role": "development_train", "task_ids": [7, 12, 20, 35], "k": 1, "arm": "correct",
   "selection_mode": "fixed_per_task", "video_pool": [46, 47, 48, 49], "state_count": 10, "seed": 20260907},
  {"checkpoint": "/run/checkpoints/macro_00000096", "output": "/output/other96",
   "role": "development_train", "task_ids": [7, 12, 20, 35], "k": 1, "arm": "same_task_other",
   "selection_mode": "fixed_per_task", "video_pool": [46, 47, 48, 49], "state_count": 10, "seed": 20260907}
]
```

每项沿用单次参数，`fixed_videos`可直接提供按global task ID索引的JSON对象；路径按当前工作目录解析，建议使用绝对路径。
一批共用asset root/device及其LoRA/tokenizer/normalization，source、model和observer合同必须相同；不兼容会在加载policy前拒绝。
每项严格加载完整Writer/Meta/probe checkpoint，独立生成原格式manifest；不跨请求缓存R、prefix或生成LoRA。
输出目录必须互不重复且尚不存在；旧单次CLI保持可用。正式执行仍要求clean pushed detached worktree与实时GPU/存储准入。

## 资产与证据入口

Canonical workspace为 `/data1/user/ymdai/projects/EMBER`；`data/`、`models/`、`runs/`、`.venv/`均为本地ignored资产，不提交远端。
复用现有资产，不新建重复模型、数据或环境。

- `configs/libero_24_8_8_v1/protocol.json`：固定24/8/8 development split。
- `configs/pi05_source_corpus_v1/`：LIBERO-90去重审计、71-task source、冻结normalizer；`libero90_nonheld_meta_v1/protocol.json`约束额外meta来源。
- `configs/pi05_writer_data_v1.json`：当前source checkpoint、tokenizer、数据和既有functional panels的统一来源索引。
  其中历史73/18角色与episode分配只是provenance，**不是新run训练合同或allowlist授权**。
- `configs/pi05_lora_v1.json`、`pi05_target_evaluation_v1.json`：LoRA目标与官方rollout配置；首版rank16按新设计登记。
- `configs/pi05_task_expert_lineages_v1.json`：已存在专家的来源与角色；专家容量不能代替共享Writer能力。
- `runs/analysis/`与`runs/outputs/`：原始分析、run contract、completion、metrics、raw rows与checkpoint；先从历史索引找对应root。
- `runs/analysis/ember_handoff_cleanup_20260906/storage_cleanup.json`：本次已退役派生缓存的精确范围、重建依赖与保留例外。
  历史cache manifest描述当时生成状态，部分tensor payload已释放，重新使用须按原配方重建。

## 本地验证与正式执行

仓库使用Python 3.12与现有`.venv`。共享代码变动后可运行：

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```

只修改文档时检查相应diff即可。正式GPU train/eval须在已授权合同下，从clean pushed commit的detached frozen worktree启动，
launch前现场检查两个GPU节点与strg01对应独立quota；单节点至多6张真正提高吞吐的GPU，保持任务权重与正常BF16/TF32语义。

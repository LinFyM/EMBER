# EMBER

EMBER研究能否把机器人教学视频中的任务知识一次性编译为策略参数：

```text
exact task language + one or more action-hidden correct teaching videos
                    -> one shared Writer
                    -> one complete task-conditioned LoRA
                    -> frozen π0.5-LIBERO source policy
                    -> closed-loop execution from unseen initializations
```

现实启发是人观看正确示范后，结合任务名称理解对象、目标状态、必要动作阶段和有向顺序，再在不同身体、视角与
初始状态下尝试；他不是逐像素或逐关节复制示范轨迹。EMBER因此让Writer只在rollout前读取一次语言与视频，
生成的LoRA随后独立驱动冻结policy闭环执行。

## Scientific target

当前底线是同一shared method、同一single checkpoint的strict paired correct严格`>150/400`，并继续提高：

- absolute closed-loop success与task breadth；
- 多task能力在同一checkpoint共同积累，而非checkpoint之间换手；
- correct视频实质优于wrong、shuffled、reversed和no-video；
- same-task不同视频保持鲁棒；
- 视频语义和有向过程真实传到LoRA、effective BA与policy action。

历史最好single checkpoint仍是v6-fast macro400：
`correct/same/wrong/shuffled/reversed=143/135/125/128/129`。训练loss、functional loss、LoRA norm/rank/cosine、
reconstruction与内部margin只作诊断，正式选择只认严格配对的single-checkpoint closed-loop结果。

## Latest result and active successor

最新完成的Full-Factor rank8在macro50 K1 strict为`91/400`、breadth5、per-task=
`4/1/38/0/0/37/11/0`。它相对matched fixed-A macro50只净增3，且effective BA只有fixed-A约`.245x`并近乎
正交；因此终局non-pass，不resume、不做mapper小修。

当前active successor是**V6 Dynamic Slot-Set Bridge**：

```text
exact language + K=1..4 same-task ordered action-hidden videos
    -> each video independently runs frozen native v6 evidence/Core/Procedure
    -> each video independently compiles 320 policy/rank-aligned Program slots
    -> per-slot permutation-invariant mean backbone + selected centered residual
    -> native v6 factor heads decode once
    -> complete 38-target rank-16 LoRA
```

它保留昨晚对齐的逐video保序、跨video集合处理和单一LoRA部署原则，同时恢复唯一闭环证明过143的v6
Core/Procedure/compiler/factor路径。Slot-Set只有约197k参数，K=1严格恒等于原v6；首轮只训练该层，快速判断
few-shot共同程序能否在强底座上带来净增。warm start只作机制开发，若成功仍需从零训练。完整authority见
[`docs/action_forecast_writer_v6_dynamic_slot_set_bridge_design.md`](docs/action_forecast_writer_v6_dynamic_slot_set_bridge_design.md)。
canonical实现、370项CPU测试和真实GPU机制门已通过，当前进入full24 B20吞吐profile。
实时run identity和下一裁决只取
[`docs/active_session_handoff.md`](docs/active_session_handoff.md)。

## Information wall

Writer可读取exact language与同task action-hidden videos。它不能读取teacher action、proprio/state、reward、
terminal、task ID、filename、object pose、hidden normalization或policy outcome。训练action只进入冻结source
policy的train24 functional loss；validation/test actions或reward不产生梯度。

每个condition只生成一套完整task LoRA。不挑video，不平均frames、raw features、分别生成的LoRAs或checkpoints，
不使用held expert dictionary。视频与action query同task但跨episode采样，要求生成的LoRA跨初始化有效，而不是
复制某条teacher trajectory。

## Data and evaluation

- foundation从generic`lerobot/pi05_base`开始，不使用读过目标40 actions的`pi05_libero`；
- source policy只在与目标40 specification-only去重后的71个LIBERO-90 tasks上训练并冻结；
- 目标40固定split为24 train / 8 validation / 8 test，位于`configs/libero_24_8_8_v1/`；
- normalization只来自source corpus并冻结；
- official evaluator严格配对correct/same/wrong/shuffled/reversed/no-video的state、RNG与video identity；
- checkpoint union、挑task checkpoint、多checkpoint融合和80-row screen都不能冒充正式结果。

## Repository map

```text
EMBER/
├── configs/                 # frozen split, source, Writer and evaluation contracts
├── data/                    # canonical datasets and LIBERO assets
├── models/                  # tokenizer/model assets
├── runs/                    # formal checkpoints, raw rows and retained evidence
├── src/ember/writer/        # one canonical active Writer implementation
├── scripts/                 # canonical training/evaluation entrypoints
├── tests/
├── docs/current_owner_requirements.md
├── docs/active_session_handoff.md
├── docs/execution_brief.md
└── docs/research_history.md
```

主要文档：

- [`AGENTS.md`](AGENTS.md)：仓库科学、信息墙、GPU、存储、Git与实验规则；
- [`docs/current_owner_requirements.md`](docs/current_owner_requirements.md)：昨晚讨论形成的目标、原则、完整架构
  推导、方法/目标边界和当前协作要求；
- [`docs/active_session_handoff.md`](docs/active_session_handoff.md)：唯一实时状态与run identity；
- [`docs/execution_brief.md`](docs/execution_brief.md)：当前实验合同和下一裁决流程；
- [`docs/research_history.md`](docs/research_history.md)：所有历史路线的精炼连续认知与禁止重复项；
- [`docs/concept.md`](docs/concept.md)：稳定问题定义；
- [`docs/novelty_and_landscape.md`](docs/novelty_and_landscape.md)：论文claim与baseline边界；
- [`task_plan.md`](task_plan.md)：长期迭代阶段；
- [`findings.md`](findings.md)：当前保留的第一性原理结论。

## Runtime principles

GPU launch前同时live检查gpu01/gpu02，单节点使用至多6张真正合适且能提高吞吐的A40；有几张用几张，不等凑满、
不跨节点、不dummy占卡。少量显存或低利用率进程不自动排除，只要余量足够且不干扰他人。多卡训练固定
`NCCL_P2P_DISABLE=1`、GPU-local NUMA mapping与deferred NCCL。

吞吐优先，接受正常BF16/TF32、batch和kernel低位差异；不为防御性安心固定batch1、重复forward、扩dtype、逐tensor
扫描或新增内容hash。正式launch仍保证信息墙、pairing、finite、OOM、asset、checkpoint与resume语义正确。

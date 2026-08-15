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

性能继续追求同一shared method、同一single checkpoint的strict paired correct严格`>150/400`并继续提高；
约145若由相邻checkpoints稳定保持、低换手、跨同任务视频鲁棒且视频因果性明确，也属于有价值的成立结果：

- absolute closed-loop success与task breadth；
- 多task能力在同一checkpoint共同积累，而非checkpoint之间换手；
- correct视频实质优于wrong、shuffled、reversed和no-video；
- same-task不同视频保持鲁棒；
- 视频语义和有向过程真实传到LoRA、effective BA与policy action。

历史最好single checkpoint仍是v6-fast macro400：
`correct/same/wrong/shuffled/reversed=143/135/125/128/129`。训练loss、functional loss、LoRA norm/rank/cosine、
reconstruction与内部margin只作诊断，正式选择只认严格配对的single-checkpoint closed-loop结果。

## Latest result and current boundary

最新完成的
[`V6-LPCP Native Probe-Value Commitment`](docs/action_forecast_writer_v6_lpcp_native_probe_value_commitment_design.md)
在cycle1 K4 strict paired correct400为`136/400`、breadth6、per-task=`1/2/48/33/0/34/18/0`、per-suite=
`3/81/34/18`。相对直接底座LPCP143严格为`120 retained / 16 gained / 23 lost / 241 both-fail`、churn39、
Jaccard`.754717`；correct、breadth与retention三项预注册门失败，因此终局不续cycle2，也不补六臂或参数小扫。

NPVC解决了前序CCT在held native compiler处消失的问题：all400 NPVC/LPCP effective-BA relative-L2 mean=
`.0004683`，validation8每task四个不同correct K4的pure-NPVC cosine/energy平均=`.40870/.54227`，natural→
reversed的probe/BA relative-L2=`1.84084/1.60518`。这证明有向视频证据已形成跨视频较一致、原生尺度且能穿过
compiler的LoRA写入。

但健康写入没有变成闭环增益：gained/lost改写mean=`.000412/.000436`，持续失败样本反而最大至`.000549`；
full24后train task4的four-view cosine/energy从`.5929/.6792`漂到`.0569/.2951`。最早缺口因此已后移到
**reward credit如何选择native Value中真正改善held occupancy的组件与符号，以及这些异质task方向如何在同一个
Writer checkpoint中共存**。下一设计不会只加scale、capacity、coherence或support guard。memory token、rank8
和其它可扩展hypernetwork形式仍开放，但只有直接针对这个接口才值得引入。

最新裁决的是
[`V6-LPCP Pre-Addressed Factor-Selective Native Value`](docs/action_forecast_writer_v6_lpcp_preaddressed_factor_selective_native_value_design.md)：
保留LPCP143、NPVC native Value和rank16，只把所有tasks/q-v-action共享的zero-init router替换为固定语言
pre-address下的factor-owned zero-init selectors。真实task4写出、顺序与吞吐健康，但train24 address effective
rank仅`2.1575`，validation8跨视频共同方向仅`.1681/.3729`、3/8过门，显著低于NPVC，故在full24前终局且没有
strict结果。当前active successor是
[`V6-LPCP Shared Joint Native-Value Gate`](docs/action_forecast_writer_v6_lpcp_shared_joint_native_value_gate_design.md)：
保留NPVC native Value与LPCP/rank16强路径，只用所有factor共享的512参数joint language-video gate替换固定地址
和family selectors；canonical实现与完整CPU合同已通过，当前无GPU run，下一步先做task4到validation8的真实
机制门。实时run identity和下一裁决只取
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

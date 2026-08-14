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

最新完成的**V6 Shared-Core Ordered-Procedure Common-Value Bridge**在macro25 K4 strict为`139/400`、breadth6、
per-task=`1/2/46/32/0/36/22/0`。相对matched parameter-free Shared-Core139严格配对为
`120 retained / 19 gained / 19 lost`，Long1净`+4`完全由Spatial/Object各净`-2`支付；Goal3与Long2仍为0，
按`<140`与breadth`<7`双门终局non-pass。

该负结果不是“Procedure没写进去”：raw ordered Procedure correction relative-L2=`.09601`，effective-BA改写=
`.01397`，q/k/output全部训练且action响应改变。但同一checkpoint在train-seen 8 tasks×10 states上的严格
output-zero反事实也是trained/zero=`64/64`、`4 gained / 4 lost`。因此full24 B20 functional credit在train和
held on-policy上都只制造能力换手；继续放大Value、改rank或移动compiler没有证据。

当前active successor是
[`V6 Ordered-Procedure On-Policy Preference Writer`](docs/action_forecast_writer_v6_ordered_procedure_on_policy_preference_design.md)：
完整保留K4有序视频、shared Core、Procedure Common-Value、native rank16 compiler和强139底座，把macro25作为
短AS cold start，随后关闭target action入口，用train24真实闭环success/failure preference只优化同一个shared
Writer的19.7万FP32 Procedure参数。部署仍是一次语言+视频生成初始LoRA，不是生成LoRA后的task-local RL。
该reward链已完成full CPU=`395 passed`和真实mixed task smoke：Writer gradient、q/k/output、effective BA与fixed action
均产生明确非零响应，跨过历史Reward-Credit的sub-ULP写出断点。首个world5 formal在保留compiler graph期间OOM，
已按历史SRTP的同类根因改为“detached LoRA求cotangent后单次重解compiler”；同一task4复测的全部科学量逐位不变、
B8 exit0。这仍只是机制证据，下一正式裁决是fresh full24 cycle1后的single-checkpoint K4 strict paired400。
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

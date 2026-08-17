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

紧邻MCTC之前，
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

PAFS-NV用fixed language address预分流native Value，但validation8跨视频共同方向仅`.1681/.3729`、3/8过门；
后继SJNV-Gate改用共享joint language-video gate，continuous hidden虽达到`.9412/.9240`，经过冻结W2与native
BF16 public factors后却降至raw factor `.0214/.2659`、effective BA `.2019/.3964`。两者均在full24前终局，
没有strict结果。

最新完成的[`MCTC`](docs/action_forecast_writer_v6_lpcp_cfmg_median_capped_task_tangent_commitment_design.md)保留
carrier-exact V6-LPCP、37个逐层one-way memory tokens、K4集合、rank32 LoRA和unit-secant reward，只在shared
update前把高于active-task norm中位数的tangents截到中位数。从sealed LPCP fresh连续训练三个exact-resume
cycles并逐checkpoint跑strict400，score/breadth=`142/7 -> 142/6 -> 136/7`；相邻churn=`36/34`，未形成稳定
共同积累。cycle2→3仍有all400 BA relative-L2 `.001199`和first4同task更新cosine/energy
`.988724/.990306`，证明memory、跨视频共同Program和native写出都在工作；但持续失败样本改写最大。最早缺口已
定位为shared reward update不能选择held on-policy有用方向并保留多task support。MCTC终局，不cycle4或参数小扫；
实时run identity和下一裁决只取[`docs/active_session_handoff.md`](docs/active_session_handoff.md)。

最新终局是[`SEOD`](docs/action_forecast_writer_v6_lpcp_cfmg_successful_expert_occupancy_distillation_design.md)：
它保留上述memory/K4/rank32 Writer，只把稀疏binary reward差换成train24 task expert自身真实成功轨迹上的
on-policy动作蒸馏。四个相邻checkpoints的strict score/breadth=`129/6 -> 135/6 -> 143/5 -> 136/5`，相邻
churn=`36/38/41`；多训后的cycle4证明143只是暂时峰值。same-task不同K4更新已高度一致，但cross-task gradient
仍接近正交且held support继续换手，故SEOD不再续训或补六臂。

终局审计还确认正式reward cache切断了输入`memory_tokens`的反向链：四轮该组gradient与Adam moments严格为0，
实际训练的是固定随机memory queries之后的temporal/set/M2P/gate。这不否定已完成的closed-loop结果，也不能被
表述为可学习SHINE式memory token失败。当前没有active GPU run；实时successor判断只取
[`docs/active_session_handoff.md`](docs/active_session_handoff.md)。

当前设计authority是[`GOMQ`](docs/action_forecast_writer_v6_lpcp_cfmg_gradient_open_memory_query_design.md)：
保持SEOD的部署前向、K4、rank32、expert-occupancy credit和optimizer，只打开同一次native context forward中
37个input memory queries的反向链。它先做two-cycle机制门，证明learned queries确实形成跨task共同读取基并穿过
完整Writer→LoRA→action，而不是因为发现旧参数漏训就直接宣称新架构有效。

## Information wall

Writer可读取exact language与同task action-hidden videos。它不能读取teacher action、proprio/state、reward、
terminal、task ID、filename、object pose、hidden normalization或policy outcome。任何获授权的train24
functional/reward信号都与teacher video信息墙分离；validation/test actions或reward不产生梯度。

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

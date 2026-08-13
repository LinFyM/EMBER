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

最新完成closed-loop节点与当前active方法是**Dynamic-K Task-Grounded Visual-Value Rank-8 Writer**：

```text
exact language + K=1..4 same-task ordered action-hidden videos
    -> 每帧真实π0.5 image/language/Action-probe context + 8 memory tokens
    -> task token查询raw patch Value并按输入顺序形成visual transition/terminal goal
    -> Action-memory transition、terminal goal与Query-only semantic address
    -> causal temporal encoder（video内保序）
    -> permutation-invariant set attention（video间提取共同程序）
    -> 20 policy groups × 8 rank coordinates M2P
    -> shared projector + four direct shape-family B readouts
    -> complete 38-target rank-8 LoRA
```

它综合了三类依据：owner关于“语言说明任务、正确视频说明完成方式”的完整要求；SHINE/Doc2LoRA类成熟
Hypernetwork的少量memory、layer-aligned状态与结构化LoRA生成原则；以及EMBER从v4到v6、K4、task drift和
mapper逐接口probe的历史证据。它不是对外部方法的机械复刻，也没有引入额外target-task数据。

上一代Dynamic-K backbone-memory和semantic-address macro50 strict分别为`100/400`和`101/400`。最新32-point
probe显示task差异在M2P/final/shared projector仍较健康，首个明显common-direction增长出现在旧family
hidden/GELU；因此当前Direct-Family-B只删除这个已定位接口，不重新推翻视频前端。退役架构的完整公式和裁决门见
[`docs/action_forecast_writer_dynamic_k_semantic_address_direct_family_b_design.md`](docs/action_forecast_writer_dynamic_k_semantic_address_direct_family_b_design.md)。

前代Direct-Family-B在macro50的K1/K4 strict为`102/98`，K4把same-task effective-BA方差约降`6.3x`却没有
解锁task；因此当前方法只新增同一次joint forward中的task-grounded raw visual goal/transition Value，不改
dynamic K、set、M2P、direct mapper、rank8或B20 recipe。完整active design见
[`docs/action_forecast_writer_dynamic_k_task_grounded_visual_value_design.md`](docs/action_forecast_writer_dynamic_k_task_grounded_visual_value_design.md)。
该方法从clean `caa2e30` fresh完成macro0→100；macro50/100 K1 strict分别为`88/86`、breadth`5/6`，macro100
per-task=`1/3/34/0/0/35/12/1`。两点严格同episode比较为`62 retained/24 gained/26 lost`、churn50，success
union=`112`而single best只有88。同期effective BA平均cosine`.809`、relative-L2`.696`，action norm几乎不变但
方向cosine仅`.739`。因此当前最早断点仍是visual evidence经B20 functional credit得到的policy方向缺少held
on-policy usefulness，而且训练在继续task-specific重写能力；不是视频未读、K-set失效、近identity或整体scale。

这两个早期点不提前终止预注册曲线。同一root正以相同world3 topology exact-resume到150；随后继续strict
150/200，分析absolute、breadth和相邻checkpoint换手，再决定下一单变量。
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

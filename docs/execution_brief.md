# EMBER Current Execution Brief

更新时间：2026-08-03 18:05 UTC。本文是操作层authority；科研结果取
`docs/active_session_handoff.md`，迁移取`docs/a100_to_bci_migration_handoff.md`，
长期边界取`AGENTS.md`。

## 0. 当前BCI运行事实（覆盖下文旧A100操作细节）

- repo：`/data1/user/ymdai/projects/EMBER`，Python环境为项目`.venv`。模型、data、
  tokenizer、checkpoint和output由CLI显式传入；`EMBER_STORAGE_ROOT`、容量上限与
  `EMBER_LIBERO_ASSETS_ROOT`也必须在进程环境显式设置，不能依赖`.env.local`猜测。
- 当前A40正式候选配置：
  `configs/pi05_as_writer_semantic_direction_store_full24_decay400_bci_v1.json`，
  6 ranks×4 tasks、16-frame encoder microbatch、logical B20、policy microbatch2，
  formal teacher-video seed固定`20260722`且loader与sealed profile字段强制一致；
  不固定物理GPU编号。
- BCI A40/NCCL2.28必须由launcher显式设置`NCCL_P2P_DISABLE=1`走SHM，并由代码
  fail-fast，不能依赖`.env.local`。rank-local CUDA构造完成非NCCL ready rendezvous后
  才建立process group；六卡collective、fresh训练和exact resume均已实跑通过。
- 多卡analysis的任务ownership和最终result sealing必须读取实际`world_size`；
  `f82c7cd`/`a115b06`已消除历史4-rank默认，并在6 ranks、8 tasks、5 conditions真实
  规模通过。不得通过少用卡绕开缺失rank。
- 评测不再递归扫描整个个人目录或执行旧个人容量硬门；只检查目标文件系统余量和
  本次选择的GPU。
- 四卡迁移验收后，六卡logical-B20冻结profile在clean pushed`391f183`完成
  fresh0→1/exact-resume1→3；峰值allocated/reserved为
  `34,970,270,720/47,108,325,376` bytes，最长105帧，合同`31ea4bc9...55de0`。
- profile checkpoint没有warm-start到formal。有效VR fresh 0→200和
  50/100/150/200 paired correct400均已完成，曲线为`76/88/126/107`。
- Direction Store clean`91feeef`也已完成fresh0→200与四点paired correct400，曲线
  `129/107/120/129`；winner macro50=129、breadth7，低于v6-fast143和严格门151。
- `6f18499`首次formal因A40 overlay误保留profile seed`172`而在首个checkpoint前停止；
  10个partial宏步只作aborted审计，禁止resume/评测。修复与fail-close回归通过后，
  必须从新clean pushed commit和全新retry1 root重新fresh启动。
- 详细运行证据和精确指标见`docs/active_session_handoff.md`第0节。下文所有
  `/data/ymdai`、A100 GPU4--7和“BCI尚未验收”描述仅是历史状态。
- owner在VR结果后恢复推进：保持one-shot，取消Writer参数量上限，优先重构条件生成
  方向存储/组合并允许配套训练修改。Direction Store rollout与全部分析现已完成，
  owner最新边界是再次暂停了解现状。仍不使用subagent；效率优先，不重复全量hash或
  无关旧artifact扫描。

## 1. 当前操作状态

- owner已于`2026-08-02 19:18 UTC`开放约十小时post-seal研究窗口；允许Target-Bound
  及其根因迭代的最短CPU vertical path、GPU4--7 B20 profile/resume、formal训练、
  paired rollout与内部分析。
- 效率优先：不重复全仓仪式、全量hash或无关旧smoke；只做会改变实验可信度的shape、
  identity、freeze、causal、gradient、OOM、resume检查。
- A100窗口GPU工作已于`02:42 UTC`停止；其delta ledger只作历史provenance。
- owner已另行授予BCI研究权限：每次比较`gpu01`/`gpu02`，只用空闲卡、合计最多6张，
  不干扰他人；当前推进不使用subagent。
- BCI VR与Semantic Direction Store的正式训练、四点rollout、完整性和内部分析均已
  完成。Direction Store曲线`129/107/120/129`，四点union/intersection=`174/65`、
  envelope gap45，漂移仍在；winner内部有效LoRA stable rank仅`1.000043`，首奇异值
  能量`.999957`。owner最新要求在rollout和全部分析后暂停；当前不启动下一实验，
  长期`>150`目标未完成。

Target-Bound已完成首小时与四点correct400=`75/120/90/110`，不续训；内部反事实证明
其视频路径到达BA/action，剩余瓶颈定位到shared factor conditional coexistence。
Semantic Factor-Basis完整correct400为`69/91/118/127/117/81/126/120`；single
winner仍是macro200，第二小时出现显著跌落与恢复，未提高上限。VR estimator正式
correct400为`76/88/126/107`，single winner126仍低于SFB127和v6-fast143；breadth
`7/4/7/5`且150→200 gained/lost=`21/40`，同样没有解决漂移。

## 2. Canonical Git state

迁移准备前EMBER：

```text
post-seal baseline main/origin-main = f9a144c94e71bb44373d7247ed0fded2ed835305
current BCI write branch = codex/bci-continuation
Target-Bound formal commit = cfd26df63d08f29d8bfaac58f585387134ed680b
BCI VR formal code commit = d9130c9fbe0d68b6a83c1a356f51f7a684845275
Direction Store formal code commit = 91feeef
six-rank internal-analysis final fix = a115b06
```

`f9a144c`是另一迁移session已经封存的基线，不回写其内容。post-seal分支与所有新
artifact作为第二批增量交付；Target-Bound已封存为负结果。Semantic Factor-Basis
显示task routing有效但没有超过v6 best，也没有解决checkpoint换手，不能宣称成功。

迁移已完成；BCI环境、路径、assets、source checkpoint、四卡验收、六卡collective、
logical-B20冻结profile、formal训练和四点评测均已核验。当前恢复架构/训练研究。

## 3. Canonical assets

### Frozen source policy

```text
source run:
/data1/user/ymdai/projects/EMBER/runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722

source checkpoint:
/data1/user/ymdai/projects/EMBER/runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000

selected raw policy SHA256:
60ea7ee898629321cf34522e5f0e45f4f1c2659c5f5dbc7b02ed9eb46a8cdf36
```

rejected EMA与optimizer/DDP/scheduler训练态已经删除；raw policy、trainer state和
原manifest保留，并通过formal source inspector。它仍能作为所有Writer/SFT/eval的
frozen source asset，但不能exact-resume source training。

### Data and tokenizer

```text
tokenizer SHA256:
8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6

tokenizer path:
/data1/user/ymdai/projects/EMBER/models/tokenizers/openpi/paligemma_tokenizer.model

LIBERO dataset revision:
f13aa24a3da8c43c7225569f28c562979fa0e35a

LIBERO dataset path:
/data1/user/ymdai/projects/EMBER/data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a

canonical feature cache:
pi05_writer_feature_cache_v2_development32_raw_e4c19f9_b32_20260722

LIBERO simulation assets:
lerobot/libero-assets@0b3ea86be5fe169d0fd036ae63d1070ec09e90f6
/data1/user/ymdai/projects/EMBER/data/simulation/ember_assets/datasets/libero-assets/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6
586 files; file-list SHA256 721aa2484de396be5267e936f115ddd5f03ffd12e0849cc1cd05bb17454996b9
```

BCI路径和容量上限不得从A100字符串猜测；先设置`EMBER_STORAGE_ROOT`与owner给出的
`EMBER_STORAGE_CAP_BYTES`，再由CLI显式传入source/checkpoint/tokenizer/data/output，
并用`EMBER_LIBERO_ASSETS_ROOT`指向精确simulation snapshot。历史sealed artifacts中的
旧绝对路径不改写。

## 4. Latest scientific decision

CV-ADR RAW在macro50/100/150/200/250/300/350/400的paired correct400为：

```text
76 / 111 / 99 / 117 / 77 / 69 / 80 / 82
```

single winner为macro200；第二小时出现明显能力崩落，未做五臂。normalized GROUP4
在cycle50/100/150/200为`82/77/73/110`，四点均值和winner均低于RAW，且内部A/E/D
职责进一步变弱，不作为默认operator。

matched诊断把late gradient方差定位为video主效应约`.1%`、query/flow主导；
functional loss继续改善而closed-loop`117→82`。因此后续必须联合处理semantic
carrier、causal write value、single-video低SNR和functional/closed-loop错位。

v5.2/v6的old/task-complete四格又证明recipe作用依赖架构。不得把post-v5低分架构
整体判死、不得简单退回old six-update，也不得恢复CP-24、gate、scale、rank loss、
multi-video或checkpoint fusion作为默认解。

Semantic Factor-Basis的八点correct400为：

```text
69 / 91 / 118 / 127 / 117 / 81 / 126 / 120
```

八点success union=`193`而single-best=`127`；250→300 lost52、300→350 gained60。
后半段raw mean对24 tasks均非负，但保留能量约`4.2%`，factor占约`97%`，Adam一阶
moment每50 macros近正交。这同时降低“CP负投影”“只加task router”和“继续降低LR”
三种解释。SFB保留为canonical结构证据，formal estimator/closed-loop错位成为下一
训练假设。

Variance-Reduced estimator的正式四点correct400为：

```text
76 / 88 / 126 / 107
```

四点均值`99.25`低于ordinary SFB同期`101.25`；single winner macro150=`126`，
比严格门151少25。breadth=`7/4/7/5`，四点union/intersection=`158/49`，150→200
gained/lost=`21/40`。全200步matched same-task CountSketch cosine只提高`.00263`
（factor`.00510`），energy retention只提高`.00191`且分阶段反复；macro200 held
functional loss改善到`.12915`时closed-loop反而降到107并比SFB同点少20。正式拒绝
“可约flow Monte Carlo方差是主要根因”，下一设计边界转为functional action surrogate
与source-policy closed-loop有效流形错位。

Semantic Direction Store的正式四点correct400为：

```text
129 / 107 / 120 / 129
```

macro50与200同分，按breadth`7 vs 5`选择50为single winner。四点
union/intersection=`174/65`，相邻gained/lost=`17/39,43/30,27/18`；早期比SFB
macro50高60，但没有超过v6-fast143或严格门151。winner内部route与Core/Program/A/E
路径均工作，真正失败发生在多维功能写出：16个rank坐标全部active却stable rank仅
`1.000043`、top singular energy`.999957`、B-column cosine`.999971`。独立stores
解决了参数所有权，未解决public A/B几乎共线的生成几何，正式负裁决且不续到400。

## 5. Current Writer state and pause boundary

Semantic Factor-Basis的完整design已在main：

```text
docs/action_forecast_writer_semantic_factor_basis_design.md
```

当前执行顺序固定：

1. 不再在A100启动训练、评测或GPU分析；
2. 最终代码/文档及34个post-seal `must-transfer` roots已形成Git与增量台账交付；
3. logical-B20六卡profile已从clean pushed commit重放并seal；
4. VR fresh 0→200、四点correct400与全部预注册分析已完成并负裁决；
5. owner已恢复推进；新设计为frozen language semantic top2八个full-capacity
   direction stores，authority见
   `docs/action_forecast_writer_semantic_direction_store_design.md`；
6. canonical替换、profile、fresh0→200、四点paired correct400和winner refs1均已完成；
7. 当前在owner要求的结果后暂停边界，不启动下一架构、training target或GPU分析。

VR的设计、BCI适配和正式负结果统一见
`docs/action_forecast_writer_variance_reduced_functional_estimator_design.md`。不得续训
该root到400、不得做五臂，也不得从其checkpoint warm-start下一方法。

profile/smoke root必须全新，不得使用A100上的smoke权重或正式output路径。B20只有
真实OOM或连续非有限才降，不扫描B17–B19/B21。

## 6. Formal training contract

owner恢复研究并封存新的training-target design前，下列通用合同保持：

- one teacher video → one complete rank-16 LoRA；
- frame stride=5；
- 24 train tasks每cycle完整覆盖，每task一条video；
- B20同task跨episode独立action queries，task内mean后task等权；
- task-query RAW full24，一次clip、一次AdamW、一次scheduler/cycle；
- 每25 cycle checkpoint；fresh identity；
- validation/test actions不读、不产生梯度；test video不进入development训练；
- frozen source policy trainable parameters=0；
- cost-balanced long-first，真实最长105-frame验证；
- exact resume不得改写既有checkpoint payload，task/video/query/RNG/cursor连续。

任何operator改变必须fresh incompatible schema并在看到outcome前seal。pairwise负
gradient cosine本身不授权projection；无真实candidate conflict时投影必须退化为raw
mean。

## 7. Paired evaluation contract

- 8 validation tasks×50 states，共400 rollouts；
- 每task 50 teacher videos无放回，每条一次；
- candidate arms严格配对state、video ordinal、env/policy/noise schedule；
- dynamic cost-balanced queue，long shards优先，persistent model/env；
- 只选择single checkpoint；不融合或挑video；
- candidate报告aggregate、per-task、per-suite、gained/lost、breadth、top-task集中度、
  success-set Jaccard、train/held loss、LoRA norm和数据曝光量；
- 只有strong single winner才做correct/same/wrong/shuffled/reversed五臂；
- shuffled/reversed必须真实改变输入帧顺序后完整forward。

内部分析优先effective BA、Gram、singular spectrum、norm和fixed-query policy action，
不用raw A/B gauge符号做跨模型结论。每task报告Core、Program、compiler、factor、
effective BA与action，定位最早失效接口。

## 8. Launch preflight on BCI

每个expensive formal run只做一次live preflight并记录：

1. canonical repo、branch/commit、clean status、origin同步；
2. frozen worktree commit与sealed config/command；
3. owner新授权的GPU IDs、进程、显存、温度和CUDA runtime；
4. 不查询或触碰未授权GPU；
5. BCI个人storage root、独立quota和预计峰值；
6. source、checkpoint、tokenizer、data root存在且identity正确；
7. output root此前不存在；tmux/log名称无冲突；
8. DDP ranks与`CUDA_VISIBLE_DEVICES`只包含获批设备；
9. 不干扰他人进程。

A100上的NUMA node1和物理GPU4–7只是历史合同，不得复制到BCI。

## 9. Evidence and retention

- formal run保留config、command、commit、parameter count、macros/videos/queries、
  wall/GPU-hours、checkpoint curve、per-task、五臂（若过门）、internal analysis、
  root cause和retain/reject。
- 原封存保留60个checkpoint roots和406个complete eval roots；post-seal新增2个正式
  训练root和12个formal correct400 roots。它们用于训练漂移与架构×recipe审计，
  迁移时不能只留winner。
- A100历史writer/eval LoRA caches、profile/resume、reseal和退役SmolVLA已按manifest
  清理；本轮四个BCI formal eval root仍保留各自cache与完整结果，当前暂停交接不做
  额外cleanup。历史文档引用已删工程root不是重新运行指令。
- 不提交checkpoint、dataset、cache或大binary。formal artifacts经SSH迁移，源码和
  文档经GitHub迁移。

## 10. Git and handoff

meaningful状态更新：

- `AGENTS.md`
- `docs/a100_to_bci_migration_handoff.md`
- `docs/active_session_handoff.md`
- 本文件
- 对应architecture design
- `task_plan.md`
- `findings.md`
- `progress.md`

commit只含任务相关改动并push。A100 Codex不迁移；后续session从Git文档、formal
artifacts和migration manifests接手。owner结束本次阶段前，保持无GPU作业状态。

# EMBER Current Execution Brief

更新时间：2026-08-04 UTC。本文是操作层authority；科研结果取
`docs/active_session_handoff.md`，迁移取`docs/a100_to_bci_migration_handoff.md`，
长期边界取`AGENTS.md`。

## 0. 当前BCI运行事实（覆盖下文旧A100操作细节）

- 当前操作主线已切换到Task-Relative Flow-Credit Writer。AS侧恢复唯一v6 Writer并做
  fresh独立cold start；reward侧不再做success-filtered self-imitation，而是保留成功和
  失败executed prefixes，用同task K4 leave-one-out binary advantage、per-sample
  old/current CFM ratio、positive PPO与negative SPO更新Writer。精确科研合同见
  `docs/action_forecast_writer_relative_flow_credit_design.md`。
- BCI A40六卡AS profile和独立exact-resume已通过：logical B20/B2、最长105帧、三步
  `33.464/30.886/30.977s`，峰值allocated/reserved
  `34,948,858,880/44,816,138,240` bytes；fresh0→1/resume1→3保持1,440 queries、72
  videos与五主block可达。sealed config为
  `configs/pi05_as_writer_v6_relative_flow_coldstart_bci_v1.json`。
- fresh AS同一root已完成0→75：75个full24宏步、36,000 queries、1,800 one-shot
  videos、0 OOM/clip，step75原子checkpoint见第0.1/0.2节。canonical step75 reward
  profile完成96条pre-update K4 rollout：47 successes、18/24 tasks至少一次成功、13个
  mixed tasks，coverage仍未过。下一正式动作是从同一AS root exact-resume75→100；不得
  把任何RL profile checkpoint当成正式cold start继续训练。

- Policy-Target-Owned Factor本轮已完成并负裁决；此前暂停已由owner解除。其源码不再
  是canonical活动路径，历史结果只由Git、artifact与design authority保留；不能resume
  其step200或历史Direction Store checkpoint。
- clean`34be4a0`的fresh0→200完成200次full24 update、96,000 queries、4,800 videos、
  8 checkpoints；wall`6678.957s`，峰值allocated/reserved`33.696/38.729GiB`，0
  clip/OOM和0 validation/test action reads。四点paired correct400=`99/76/86/68`，
  breadth=`6/6/7/5`，union/intersection=`136/37`；winner99低于Direction Store129与
  v6-fast143，不续400。
- macro50五条件内部分析证明76 heads已把q/v跨层BA余弦从`.932/.967`降到约0，但
  LoRA层能量过度集中、norm下降，Program差异扩大的BA未变成闭环有效action方向。
  训练内部factor task梯度近随机正交且同task+demo不稳定重现；最新根因边界是
  condition-to-policy credit，而不是继续增加head、强制rank/SFT几何或调gate/scale。

- repo：`/data1/user/ymdai/projects/EMBER`，Python环境为项目`.venv`。模型、data、
  tokenizer、checkpoint和output由CLI显式传入；`EMBER_STORAGE_ROOT`、容量上限与
  `EMBER_LIBERO_ASSETS_ROOT`也必须在进程环境显式设置，不能依赖`.env.local`猜测。
- 当前A40正式候选配置：
  `configs/pi05_as_writer_v6_relative_flow_coldstart_bci_v1.json`，
  6 ranks×4 tasks、16-frame encoder microbatch、logical B20、policy microbatch2，
  formal teacher-video seed固定`20260722`且loader与sealed profile字段强制一致；
  不固定物理GPU编号。
- BCI A40/NCCL2.28必须由launcher显式设置`NCCL_P2P_DISABLE=1`走SHM，并由代码
  fail-fast，不能依赖`.env.local`。rank-local CUDA构造完成非NCCL ready rendezvous后
  才建立process group；六卡collective、fresh训练和exact resume均已实跑通过。
- reward credit的outcome-dependent本地反向结束后也必须先经独立FileStore all-rank-ready，
  再统一进入NCCL gradient sum。`e5bca71`已在原六卡96-rollout两epoch失败规模上重放：
  rollout 96/96字节级不变、两轮finite update、完整cycle1 checkpoint且0 watchdog。
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
  owner已再次授权继续推进并要求遇到科学问题自行深入分析。仍不使用subagent；效率
  优先，不重复全量hash或无关旧artifact扫描。

### 0.1 v6 Relative-Flow AS fresh 0→25 launch contract

- canonical workspace为`/data1/user/ymdai/projects/EMBER`，branch为
  `codex/bci-continuation`，实现commit为`b75cb19`；实际launch snapshot必须是包含本段
  的clean commit且与`origin/main`一致，精确HEAD由自动`run_contract.json`记录。
- sealed config为`configs/pi05_as_writer_v6_relative_flow_coldstart_bci_v1.json`；source
  是既有formal raw step1000，tokenizer/data/assets均取本文第3节canonical BCI路径。
  fresh identity，不传`--resume`或`--initialize-writer-checkpoint`，profile权重不得进入。
- 首段scale为25个full24 macros：12,000 logical queries、600 one-shot video conditions、
  6,000 physical B2 frozen-policy forwards，step25保存完整checkpoint。formal总schedule仍
  是400、every25；本次只执行sealed selected stop25，不缩短scheduler或数据量。
- output root固定为
  `runs/outputs/pi05_as_writer_v6_relative_flow_coldstart_formal_r6_b20_seed7_b75cb19_20260804`，
  启动前必须不存在；log固定为
  `runs/logs/pi05_as_writer_v6_relative_flow_coldstart_formal_r6_b20_seed7_b75cb19_20260804.log`，
  tmux为`ember_v6_rf_as_s25_b75cb19_20260804`。单checkpoint实测约133MB，完整400-step
  root按3GiB峰值预留，首段低于0.5GiB；2026-08-04 live `/data1` quota为
  `283,087,724/1,073,741,824 KiB`，容量充分。
- live设备裁决为gpu01八卡均有cyzhao进程，gpu02物理0/6有yfwang进程；只选
  `gpu02:1,2,3,4,5,7`六张空闲A40，DDP6、每rank 4 tasks，显式
  `NCCL_P2P_DISABLE=1`，rank按物理GPU自动绑定NUMA0/1。launch前再做一次短现场复核；
  任一所选卡被占用即不启动并重新选择，不挤占。
- exact inner command为：

```bash
PYTHONPATH=$PWD/src CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=1,2,3,4,5,7 NCCL_P2P_DISABLE=1 \
OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false \
EMBER_STORAGE_ROOT=/data1/user/ymdai EMBER_STORAGE_CAP_BYTES=1099511627776 \
EMBER_LIBERO_ASSETS_ROOT=$PWD/data/simulation/ember_assets/datasets/libero-assets/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6 \
.venv/bin/torchrun --standalone --nproc-per-node=6 scripts/train_as_writer.py \
  --config configs/pi05_as_writer_v6_relative_flow_coldstart_bci_v1.json \
  --mode formal \
  --source-run runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722 \
  --checkpoint runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000 \
  --tokenizer-path models/tokenizers/openpi/paligemma_tokenizer.model \
  --data-root data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a \
  --output-dir runs/outputs/pi05_as_writer_v6_relative_flow_coldstart_formal_r6_b20_seed7_b75cb19_20260804 \
  --num-workers 0 --log-every 1 --skip-data-sha
```

- 监控finite/OOM、每macro 24 tasks/480 unique queries/24 videos、teacher seed20260722、
  source trainable=0和写入边界。无step25完整checkpoint的partial root不得resume或用于
  coverage；正常step25后只允许同合同exact-resume。cold-start选择不看validation/test
  outcome，下一步只读24-train official random-reset K4 pre-update coverage。

### 0.2 step25/50/75 reward profile裁决与下一段

- AS root已完成step25，contract=`ad7ba631...b3d1`，`metrics_rows=25`、
  `global_policy_samples=12000`、`global_writer_video_conditions=600`，最终checkpoint为
  `checkpoints/step_00000025`。续到50必须传该checkpoint、同一output root、
  `--stop-after-step 50`和`--allow-contract-compatible-code-resume`；后者只容许AS科研
  contract不变而运行代码commit前进。
- 有效reward profile为
  `runs/outputs/pi05_rl_writer_relative_flow_profile_from_v6_macro025_r6_bci_retry2_6ff7599_20260804`，
  contract=`f0425002...b112`。96/96 official random-reset ledgers、28,085 actions、25
  successes；12/24 tasks有success，9 mixed、3 all-success、12 all-failure。suite success为
  spatial/object/goal/libero10=`7/7/10/1`，覆盖task数=`4/4/3/1`，故step25 coverage失败。
- 两epoch ratio范围=`[.9860,1.0174]`和`[.8902,1.0629]`，positive clip fraction=
  `0/.001781`，grad norm=`.04016/.03035`；机制非零且未被clip吞没。wall=`1066.374s`，
  max CUDA reserved=`45,183,139,840` bytes，A40可运行但不再扩大K、Nmc或replay batch。
- 启动时修复了两个真实多卡runtime缺口：RL环境池必须绑定sealed LIBERO asset cache；
  `MUJOCO_EGL_DEVICE_ID`必须取`CUDA_VISIBLE_DEVICES[LOCAL_RANK]`的物理卡号。前两个失败/
  中止root没有参数更新，只作工程诊断；后者的15条ledger不得进入科研比较。
- AS随后从同一step25 checkpoint exact-resume到50，合同仍为`ad7ba631...b3d1`；累计
  `metrics_rows=50`、24,000 queries、1,200 videos，step50 checkpoint完整。step50有效
  reward profile root为
  `runs/outputs/pi05_rl_writer_relative_flow_profile_from_v6_macro050_r6_bci_retry1_e5bca71_20260804`，
  contract=`f8552406...698c`：96 rollouts、25,878 actions、38 successes、14/24 coverage、
  10 mixed、4 all-success、10 all-failure。相对step25严格配对gained/lost/retained=
  `19/6/19`，coverage`12→14`；任务7/15/20/34新获得success，5/16失去coverage。
- 首次step50 run在96条ledger后因0-mixed rank提前进入NCCL credit sum而触发480秒
  watchdog，未产生update。`e5bca71`用每epoch独立FileStore ready rendezvous根修；原
  规模重放的96条JSON与失败run逐文件一致，完成两epoch、cycle1 checkpoint和summary，
  0 watchdog。ratio=`[.9905,1.0094]`/`[.8555,1.0559]`，clip均0，grad=
  `.02872/.02697`，max reserved=`40,342,913,024` bytes。
- AS同root继续exact-resume50→75，累计36,000 queries/1,800 videos、75 finite macros，
  segment wall=`805.356s`。首次尝试因live选卡形成`4+2` NUMA rank分布、不同于root封存
  `3+3` topology而在训练前正确fail-close；改用同节点`1,2,3,4,5,7`保持原topology后
  正常完成，无partial metrics/checkpoint混入。
- step75 reward profile root为
  `runs/outputs/pi05_rl_writer_relative_flow_profile_from_v6_macro075_r6_bci_04dbc4d_20260804`，
  contract=`166cf8b4...e8d5`：96 rollouts、25,223 actions、47 successes、18/24 coverage、
  13 mixed、5 all-success、6 all-failure。相对step50严格配对gained/lost/retained=
  `21/12/26`；coverage获得task`9/16/19/25/37`、失去task4。suite coverage=
  `4/6/5/3`，object已6/6，主要余缺在spatial与libero10。
- 两epoch ratio=`[.9778,1.0266]`/`[.9117,1.0812]`，positive clip=`0/.000247`，
  grad=`.03184/.02709`，max reserved=`40,340,815,872` bytes；0 watchdog并封存完整
  cycle1 checkpoint。coverage仍未过，下一段只从AS step75续到100。

## 1. 当前操作状态

- owner已于`2026-08-02 19:18 UTC`开放约十小时post-seal研究窗口；允许Target-Bound
  及其根因迭代的最短CPU vertical path、GPU4--7 B20 profile/resume、formal训练、
  paired rollout与内部分析。
- 效率优先：不重复全仓仪式、全量hash或无关旧smoke；只做会改变实验可信度的shape、
  identity、freeze、causal、gradient、OOM、resume检查。
- A100窗口GPU工作已于`02:42 UTC`停止；其delta ledger只作历史provenance。
- owner已另行授予BCI研究权限：每次比较`gpu01`/`gpu02`，只用空闲卡、合计最多6张，
  不干扰他人；当前推进不使用subagent。
- BCI VR、Semantic Direction Store和Policy-Target-Owned Factor的正式训练、四点
  rollout、完整性与winner内部分析均已完成。Target-Owned曲线`99/76/86/68`，低于
  Direction Store`129/107/120/129`；它修复跨layer硬同向却没有修复task换手或闭环
  credit。当前活动路径为上述Relative-Flow cold-start门控，长期`>150`目标未完成。

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
Target-Owned Factor formal code commit = 34be4a0b8804f9d0c9f64d66af2f4bf8327f59e9
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

Policy-Target-Owned Factor的正式四点correct400为：

```text
99 / 76 / 86 / 68
```

breadth=`6/6/7/5`，union/intersection=`136/37`，相邻gained/lost=
`15/38,35/25,18/36`。76个tensor heads确实把q/v跨层BA余弦降到约0，却把q/v
top-4能量集中到`.733/.853`，correct LoRA norm降至`19.03`。same Program变化`.909`
到BA`.091`但action只`.032`；factor的24-task gradient cosine`.0040`、负pair`.4457`，
相同task+demo方向也不稳定重现。因此policy-target硬共享只是旧几何的原因，不是task
漂移主因；当前最早接口是condition-to-policy credit没有形成闭环有效、可累积的
task/video方向。

## 5. Current Writer state and pause boundary

最新完整design与负裁决已在main：

```text
docs/action_forecast_writer_target_owned_factor_design.md
```

当前执行顺序固定：

1. 不再在A100启动训练、评测或GPU分析；
2. 最终代码/文档及34个post-seal `must-transfer` roots已形成Git与增量台账交付；
3. logical-B20六卡profile已从clean pushed commit重放并seal；
4. VR fresh 0→200、四点correct400与全部预注册分析已完成并负裁决；
5. Direction Store canonical替换、profile、fresh0→200、四点rollout和winner分析已
   完成并负裁决；
6. Target-Owned Factor的canonical替换、longest105 profile、fresh0→200、四点paired
   correct400和winner五条件分析也已完成并负裁决；
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

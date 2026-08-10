# EMBER

EMBER研究能否把目标机器人的无action教学视频一次性编译为完整task-specific LoRA，使同一个frozen
VLA在该任务不同初始化上闭环工作：

```text
exact task language + exactly one action-hidden teaching video
                    -> shared Writer
                    -> one complete rank-16 task LoRA
                    -> frozen π0.5-LIBERO source policy
                    -> closed-loop execution
```

Writer只在rollout前运行一次。它不读取teacher action、proprio、reward、terminal、task ID、filename、
object pose或hidden normalization；video是唯一dynamic value，不能存在language-only LoRA bypass。

## Current status

- canonical workspace：`/data1/user/ymdai/projects/EMBER`；主写分支：`codex/bci-continuation`。
- 长期Goal未完成：同一shared method/single checkpoint strict paired correct必须严格`>150/400`并继续
  提高，同时保持视频时序因果、same-task鲁棒、breadth和低checkpoint漂移。
- 历史最好single checkpoint是v6-fast macro400：五臂
  `correct/same/wrong/shuffled/reversed=143/135/125/128/129`。

### Latest active boundary（2026-08-11）

- Reward-Credit已从clean frozen`e3857f7`完成formal cycle0→1和预注册correct400。训练natural exit0、
  24 tasks/96 rollouts、B8、0 OOM/nonfinite；strict仍为`134/400`、breadth6、per-task=
  `1/4/46/31/0/38/14/0`，相对zero-Program macro0为`14 gained / 14 lost`。它没有达到cycle2或controls门，
  因此不续训、不扫reward scale/K/Nmc/RLS参数。
- 该non-pass的首断点不是video/Program：correct的task-common和same-video结构从P256 Program保留到analytic
  FactorHead tangent与continuous effective BA；wrong/shuffled/reversed Program量级约correct的`1.15%`。
  真正失败发生在q/v原生BF16 factor materialization：约`1e-8 RMS`的factor delta小于非零A/B约`1e-4`
  的局部ULP，native own-target cosine仅约`.037`；action FP32 factors保持健康。
- FP16 direct、dither、local-CD、gauge/global scale和absolute rank16 refactor都有直接non-pass。当前唯一active
  design是`docs/action_forecast_writer_qv_rank_reserved_native_reward_design.md`：36个q/v targets保留14个
  pivot-selected原生B columns并重解A，两个physical zero-B slots写condition-local rank2 Reward residual；
  两个action targets保持原full-rank16 FP32。对外仍是一套38-target public rank16 LoRA。
- full80 generation-only门已通过：q/v base error约`.0007523`、task max≤`.001302`，rank2 capture
  `.9997088`，dynamic cosine`.9975247`、video-centered cosine`.950556`，action exact。但artifact明确为
  0 policy forward/0 rollout/0 update，不能作为性能结果。
- canonical load-only compiler、v9 family/config、Program load-only reference、commit-bound native cache和
  ordered Gate B/C现已实现；旧Reward训练入口在distributed/runtime初始化前fail closed。活动config为
  `configs/pi05_v6_qv_rank_reserved_native_reward_v1.json`。当前CPU回归为`386 passed`。
- 下一顺序是clean pushed/frozen后的单卡同32-request panel B8/16/32吞吐profile，以及五臂四suite
  fixed-action + native cache/release + 8-row rollout vertical；更大profile候选OOM只作ineligible，vertical的
  full/qv-only action必须使用cache重新加载的q/v state，并分别报告configured selected batch与实际8-entry
  cache batch。profile+vertical通过后，回到tracked主分支由`rank-reserved-seal`自动组装config evidence，
  commit/push并新建sealed frozen worktree。随后先跑新rank14
  zero-Program strict400。若correct<130、breadth<6或
  相对旧134 lost>10，直接reject并省去第二个400；只有base过门才读取原84MB Program做rank14+2 cycle1
  strict400。只有correct≥144、breadth≥6、lost≤6且gained>lost才算load-only通过并补同checkpoint controls；
  140--143是诊断性non-pass，不授权fresh训练。两项行为门前不启动训练。
- 当前没有EMBER GPU进程。每次同时live检查`gpu01/gpu02`，选择一个节点并使用该节点所有真正空闲、健康且
  提高有效吞吐的A40；没有固定6卡上限，不等待凑卡、不dummy占位、不为跨节点碎片改launcher，也不触碰
  他人compute进程。训练多卡继续遵守NCCL/NUMA/deferred-NCCL合同。

### Earlier completed evidence

- v6-Prior whole-LoRA objective已完成formal 0→50和同一schedule四点strict correct400：
  macro0/10/25/50=`134/127/105/123`。macro0仍最佳，四点逐task envelope=`147`；该objective已停止，
  不续训、不扫权重、不为loser补五臂。
- 第34节ECP已完成formal0→10→25并退役：strict correct=`133/120`，macro25相对同schedule
  macro0=`134`的paired gained/lost=`13/27`、`p=.038477`。它成功把`a_correct`和expert component推高，
  却伴随更大的expert-orthogonal drift和显著closed-loop退化，因此不续50/100、不扫权重、不补六臂。
- 第35节v6 Condition-Local Dynamic Expert Tangent Tube也已完成正式裁决并退役。clean frozen
  `b308941`的fresh0→10自然exit0，10 macros总step wall=`207.444s`、input wait=`.265s`、peak
  allocated/reserved=`43.316/47.113GB`，0 OOM/nonfinite。macro10 correct/negative的
  `||Delta_perp||/||G0||`中位=`.01390/.01408`，说明半径受控；但方向比中位=
  `108.93/126.88`且两臂`0/24` tasks过门，task median `|a_correct-1|=.25229`，completion同样
  `0/24`。它主要压小更新，没有把更新转进expert方向。
- macro10 one-shot strict correct400=`131/400`、correct80=`27/80`、breadth5，per-task=
  `0/3/46/31/0/40/11/0`、per-suite=`3/77/40/11`。相对同schedule macro0=`134`的精确
  gained/lost=`16/19`、churn35、net`-3`；相对ECP10=`133`为`19/21`、net`-2`。因此不续25、
  不补六臂、不扫tube weight/LR/WD。该结果只淘汰当时的tangent recipe/window；completion从未成立，
  不能扩大成“expert component本身无效”。
- 第36节matched Expert-Flow Teacher
  Viability Audit已从clean frozen`e8e4728`完成：step2000 expert/macro0/tangent10的matched真实7维flow
  loss=`.098631/.091802/.091843`，expert只在`2/24` tasks、`0/4` suite means同时优于两baseline，明确未过
  `18/24+3/4` teacher-quality门。compiler/factor gradient residual=`.6864/.8387`虽非冗余，但不能把整体
  更差teacher变成有效监督，因此`authorize_cefd=false`，不实现CEFD、不做weight profile。
- audit完整覆盖480/480 queries、144次policy forward、0 update/rollout/OOM/nonfinite，wall=`39.698s`；
  一次性mode/flow-teacher/effective-objective执行路径和tests现已删除，历史只留Git/formal artifact。
- 第37节Frozen-v6 Counterfactual-Null Program Residual v1已完成唯一macro49 profile并退役：13项门中
  10项通过，correct retention=`.807966`且24/24，但DC-dominated key使condition=`1315.33`、negative/
  correct=`.264351`、null仅15/24；production ratio=`1.115458`。不训练v1、不降lambda、不扫seed/P/阈值。
- 第38节Balanced DC--Causal v2在当时是唯一active implementation：strict freeze v6 macro400全部600
  tensors，在fused Program后保留同一zero-init `[256,320,256]` FP32 memory和full48 update，只把video-DC
  static与centered sqrt-causal-prefix dynamic分别fixed-JL到128、各自zero-L2后拼成P256。无optimizer、
  expert target、negative policy forward或第二LoRA。
- v2 implementation CPU seal为聚焦`52 passed`、全仓`281 passed`。mechanism写回与deployment双root
  fail-close修复后当时全仓为`283 passed`，compileall/Black/JSON/diff-check和architecture 0 hard violation。随后
  clean frozen`5d93434`的macro49 mechanism profile **13/13通过**：condition=`106.114`、correct/cotangent=
  `.968254`、negative/correct=`.0218514`、24/24 correct和24/24 null；A/B、4/4 fixed-action、closure、
  0 negative forward全部通过。production ratio=`.949122`，无checkpoint/OOM/nonfinite，六卡已释放。
- mechanism与deployment artifact随后均seal；该阶段尚无v2训练或formal strict成绩。clean frozen`2af82aa`
  在空闲`gpu02:0`上以同一32-request/1093-frame panel实测batch8/16/32=`.911238/.901898/.906482 LoRA/s`，
  三者稳定且显存余量约32.4GiB，按最高实测吞吐选择batch8。随后validation8×state0 correct vertical smoke
  真实生成8套完整LoRA并完成8条LIBERO闭环，`4/8` success；该小样本只作执行证据，不是性能结论。
- 双root verifier当时已共同通过同一commit的profile root、vertical `results.json`和native LoRA cache manifest，
  并核对单次launcher、8 rows/entries、76 tensors、Writer release/source reuse和零禁止读取。config/runtime
  当时为formal ready。写回后的clean frozen worktree首次CPU-only formal prepare在CUDA初始化前发现一个纯工程
  问题：`runs`软链接解析到canonical仓库后被旧verifier误判为越界。`af7b101`已把路径owner收窄修复，
  nested symlink逃逸仍拒绝；全仓`285 passed in 21.38s`。同一clean frozen commit的prepare现已exit0并确认
  8×50 correct/no-replacement、historical-v6+exact-zero residual macro0、18 rollout workers + 18 Writer
  generators和batch8合同；临时prepare root已清理，它未启动
  GPU或产生性能结果。当时的下一门是从包含当前authority的新clean frozen seal评测zero-memory macro0
  strict400；现已由下项正式完成。
- zero-memory macro0现已从clean frozen`6b5f7a6`在空闲`gpu02:0--5`正式完成：`134/400`、breadth6，
  per-task按Spatial/Object/Goal/Long为`0/5/48/34/0/35/11/1`。72/72 shards和18 workers均attempt1/exit0；
  wall=`867.152s`，400套LoRA由18 generators以54 batches、max batch8全部fresh生成，0 retry/OOM/nonfinite/
  forbidden reads，Writer释放/source复用且GPU已释放。与历史native macro0的400个paired rows在state、RNG、
  video和success上逐行完全相同，gained/lost=`0/0`；400套cache的30,400个LoRA tensors、514,867,200
  values也全部bit-exact，不是只碰巧aggregate相等。当时的下一步是formal fresh0→10；该训练与
  macro10/25 strict结果现已完成，裁决以上方latest active boundary为准。
- 首次A40 batch8 smoke只发现普通BF16 batch-shape roundoff（max`.001953125`、mean约`4.70e-5`，direct
  repeat为零）。此前固定batch1和重复direct forward的决定已经撤回；当前吞吐优先，从稳定且有显存
  余量的候选中选择实测LoRAs/s最高的batch，并使用原生BF16/F32 LoRA cache、action prefetch和更少
  host sync。
- 历史v6 fixed-panel profile在同一32 requests/1093 sampled frames上得到batch8/16/32吞吐
  `.911427/.905107/.906432 LoRA/s`；新v8 residual graph的独立复测为
  `.911238/.901898/.906482 LoRA/s`。两者都由实测选择batch8，且新graph没有借用历史seal。
- logical B20保持不变；physical B20和B16已由A40容量实证排除，balanced B10+10以FP32 leaf-gradient
  加权累积完成train24×20=`480/480` queries。旧whole-LoRA gradient seal的expert/ranking weights为
  `.008355172068998324/.28570466890490887`；ECP重新实测的projection/ranking weights为
  `.006883349605446485/.010514451404229894`。已退役Tangent Tube当时从自己的live gradient seal得到
  `.00686480847114155/.010514453175708578`。第38节没有auxiliary weight或optimizer，不继承任一旧seal。
- formal训练root为
  `runs/outputs/pi05_v6_prior_formal_r6_lb20_mb10_eff15db_20260809`；四点paired分析保存在
  `runs/outputs/pi05_v6_prior_checkpoint_curve_strict_paired_eff15db_20260809/analysis.json`。

当前科研结论、完整历史实验谱系和关键不确定性见
[`docs/active_session_handoff.md`](docs/active_session_handoff.md)；精确执行协议见
[`docs/execution_brief.md`](docs/execution_brief.md)；当前计划见[`task_plan.md`](task_plan.md)。

## Cumulative evidence in one view

- frozen source base为`48/400`；privileged mixed-task Source-SFT best为`109/400`。
- v5.2 old的`132/138/74/82/83`仍是最强视频特异性形态；v6-fast task-complete的
  `143/135/125/128/129`是最高absolute。两者recipe交叉结果证明架构与训练方式耦合，不能整体判死某一
  architecture，也不能简单退回old recipe。
- CV-ADR、Target-Bound、Semantic Factor-Basis、variance reduction、Direction Store、Target-Owned、
  Policy-Lane/Atom、Condition-Kernel、K4/few-shot、trace/expert routing和多条reward路线逐步证明：视频
  sensitivity、LoRA健康几何、较低functional loss、独立parameter ownership或few-shot任一项都不是
  closed-loop成功的充分条件。
- 24个task experts统一step2000的development-train direct-expert成绩为`658/1200`、23/24 tasks非零，
  证明它们是有用但不完美的privileged train targets；soft/hard bank在held panel只有`15/80`和`3/80`，明确否定把train experts
  直接当deployment字典。
- 连续因果链已经收窄：whole-LoRA主要径向收缩；ECP补足expert分量却让大量正交方向漂移；Tangent
  控制相对半径却没有把shared update旋进expert方向；matched audit又证明expert flow在22/24 tasks上
  比macro0更差。当前把历史Condition-Kernel已证明的condition credit隔离与v6高增益decoder结合，直接
  检验shared update geometry。v2已证明condition隔离有效但blind-add不保留旧能力；当前第39节只修
  cumulative reconciliation，仍先过真实机制/吞吐门，再以strict closed-loop裁决。

## Data and evaluation

- 起点是generic`lerobot/pi05_base`，不是读过目标LIBERO-40 actions的`pi05_libero`。
- LIBERO-90 specification-only audit排除19个与目标40 exact semantic/composition重合的source tasks；
  71 tasks×50成功episodes训练frozen source base。
- 目标40固定development split为24 train / 8 validation / 8 test，不按outcome改task IDs。
- Writer训练只读train24 actions；validation/test actions不产生梯度。
- official evaluation严格配对correct/same-task-other/cross-suite-wrong/shuffled/reversed/no-video的state、
  env/policy RNG、video ordinal和输入处理；shuffled/reversed真实重排frames后完整forward。
- checkpoint只由真实closed-loop选择。loss、smoke、LoRA norm/rank/cosine和内部路径只能作机制证据。

## Runtime and paths

BCI项目资产按canonical roots归并：

```text
EMBER/
├── data/       # datasets and LIBERO assets
├── models/     # tokenizer/model assets
├── runs/       # training/evaluation/checkpoints/logs
├── evidence/   # migration and retained evidence
├── .venv/
└── .cache/
```

进入仓库后使用项目`.venv`；`.env.local`提供BCI本地默认路径，训练与评测仍通过CLI显式传入关键资产。
主要入口：

```text
scripts/train_task_experts.py
scripts/train_v6_prior_writer.py
scripts/evaluate_pi05.py
```

GPU工作必须实时检查`gpu01/gpu02`，选择一个节点并使用该节点所有真正空闲、健康且提高有效吞吐的A40；
没有固定6卡上限，不等待凑卡、不dummy占位、不为跨节点碎片改launcher、不干扰他人。训练多卡显式
`NCCL_P2P_DISABLE=1`并遵守NUMA physical/local
rank和deferred-NCCL合同。不得为验证身份生成或比较SHA-256/MD5；吞吐、有效显存利用和尽快获得真实
严格评测优先。

## Required reading

修改或运行项目前必须完整遵守[`AGENTS.md`](AGENTS.md)中的阅读清单。最小当前入口是：

1. `AGENTS.md`
2. `README.md`
3. `docs/active_session_handoff.md`
4. `docs/execution_brief.md`
5. `docs/action_forecast_writer_video_expert_manifold_design.md`
6. `docs/action_forecast_writer_qv_rank_reserved_native_reward_design.md`
7. `task_plan.md`
8. `findings.md`
9. `progress.md`

历史设计保留为证据而非活动实现；改变其拥有的接口前，按handoff实验谱系读对应design到EOF。

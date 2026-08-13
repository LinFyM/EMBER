# EMBER Active Session State

更新时间：2026-08-13。本文是唯一实时实验状态入口；旧文档、Git快照与formal artifacts中的“当前/下一步”只
表示当时时点。稳定目标与owner要求见`docs/current_owner_requirements.md`，历史结果见`docs/research_history.md`。

## 1. Current truth

- 长期Goal处于active：同一shared Writer、同一single checkpoint的strict paired correct严格`>150/400`，同时
  保持高breadth、低能力换手和正确教学视频的内容/顺序因果性；
- 历史最好仍是v6-fast macro400：`143/135/125/128/129`；
- 唯一主工作树：`/data1/user/ymdai/projects/EMBER`；唯一主写分支：`codex/bci-continuation`；
- 最新完成closed-loop architecture：Dynamic-K Semantic-Address Direct-Family-B Rank-8 Writer；其同一macro50 checkpoint
  K1/K4 strict为`102/98`、breadth均5，按预注册门终局non-pass，不resume；
- K4已正式证明cross-video set会显著过滤same-task demo nuisance，但不会自动产生更有用的task mean；下一fresh
  successor只前移到per-video高层视觉语义Value/Procedure接口，不继续修改K、set、mapper或LoRA几何；
- active implementation：Dynamic-K Task-Grounded Visual-Value Rank-8 Writer；design、canonical实现、CPU机制与
  matched world4 profile已通过；正式fresh 0→50已完整结束，50→100 exact-resume与macro50 K1 strict400正在
  并行运行；尚无新closed-loop成绩；
- 当前暂不使用subagents；实现、训练、评测和分析由当前主任务持续完成；

## 2. Latest completed architecture

完整设计authority：
`docs/action_forecast_writer_dynamic_k_semantic_address_direct_family_b_design.md`。

数据流：

```text
exact language + K=1..4 same-task action-hidden ordered videos
-> 每帧真实joint image/language/50 Action-probe context + 8 memory tokens
-> per-video signed adjacent transitions D + terminal goal residual G
-> absolute memory mean只作temporal Query semantic address
-> causal temporal encoder
-> permutation-invariant cross-video set attention + symmetric reduction
-> 20 policy groups x 8 rank coordinates M2P
-> shared bias-free 256->1024 projector
-> four bias-free zero-init direct family-B readouts
-> one complete 38-target rank-8 task LoRA
```

当前方法保留Dynamic-K semantic-address的全部输入、memory、temporal、set、M2P、fixed A和B20 recipe，只删除
旧mapper的四个family `1024->1024` hidden/GELU与未启用dynamic-A heads。最新mapper为5个trainable matrices、
`3,702,784`参数；整个Writer共`9,987,840`个trainable参数，输出76 tensors / `643,584`个LoRA scalars。

单变量依据：semantic-address macro50 strict=`101/400`，但逐接口probe中correct task-mean off-diagonal cosine从
M2P/final/shared-project的`.492/.529/.530`到family hidden的`.634`和dynamic-B/effective-BA的`.779/.779`。
因此首个新增common-direction接口是旧nonlinear family mapper，不是继续重写已经能保留task/order差异的视频
前端。上一代结果和probe只支持这个窄假设；下文的closed-loop结果已经否决其充分性。

## 3. Completed formal training

第一次world6 formal attempt：

`runs/outputs/pi05_dynamic_k_semantic_address_direct_family_b_rank8_formal_fresh0to50_r6_b20_c5353f3_20260813`

owner要求停止时停在完整macro16，无macro25 checkpoint、无completion、无strict评测。它只记录一次用户中止的
非完整run，不得resume、不得作为正式成绩，也不得覆盖。

完整fresh run：

`runs/outputs/pi05_dynamic_k_semantic_address_direct_family_b_rank8_formal_fresh0to50_r5_b20_c5353f3_retry1_20260813`

- frozen worktree：`/data1/user/ymdai/worktrees/EMBER-direct-family-b-formal-c5353f3`；
- clean commit：`c5353f3442a88565eded3b968dda104df5acc5cb`，与origin一致；
- host/devices：`gpu01`物理GPU`0,4,5,6,7`，world5；启动时五卡空闲健康，gpu01 1/2/3属于他人；
- launch：fresh macro0，formal total400，当前段stop-after50，B20，checkpoint every25，num_workers0；
- dynamic K：每macro的24 tasks中K1/K2/K3/K4各6，task等权、跨episode action queries；
- environment：BF16/TF32、`NCCL_P2P_DISABLE=1`、GPU-local NUMA、deferred NCCL；
- 原训练tmux `ember_dfb_r5_retry1`已正常退出；
- log：`runs/logs/pi05_dynamic_k_semantic_address_direct_family_b_rank8_formal_fresh0to50_r5_b20_c5353f3_retry1_20260813.log`；
- output是fresh root，不从world6中止run、旧semantic checkpoint或profile checkpoint迁移任何state；
- storage preflight：`/data1` user quota约`493 GiB / 1 TiB`，两个约185MiB checkpoint加run metadata远低于余量。

训练已完整结束：`metrics.jsonl`有50条，`completion.json.completed_macro=50`，macro25/50两个checkpoint均有
完整manifest，tmux/torchrun正常退出；总耗时`2138.7067s`。macro50 functional/consistency loss=
`.115038/.005875`、gradient norm=`.050324`且K1--K4各6，只证明训练合同健康，不是性能结论。

## 4. Sealed profile evidence

- canonical implementation：`3866f50`；runtime profile seal：`c5353f3`；完整CPU回归=`372 passed`；
- world5 full24 B20 profile：`39.4234s/macro`，相对matched semantic-address world5=`1.00476x`，K1--K4各6，
  loss/gradient finite，峰值allocated/reserved=`39.093/45.445 GB`；
- fixed validation8x4 deployment B8/B16/B32 LoRA/s=`.97732/.96489/.96513`，全部覆盖最长视频且0 OOM，正式
  evaluator锁B8；
- source policy、normalization、24/8/8 split、official LIBERO preprocessing和38-target topology不变。

## 5. Completed K1 strict400 and terminal analysis

正式root：

`runs/outputs/pi05_dynamic_k_semantic_address_direct_family_b_rank8_correct400_noreplacement_seed7_macro0050_trainr5_evalr6_c5353f3_gpu01_retry1_20260813`

- frozen eval worktree：`/data1/user/ymdai/worktrees/EMBER-direct-family-b-eval-c5353f3`，clean detached `c5353f3`；
- host/devices：gpu01物理GPU`0,2,4,5,6,7`，六卡；每卡3个persistent rollout replicas、1个Writer generator；
- arm：validation8×50、correct K1、without-replacement seed7、macro50 single checkpoint、generation B8；
- 原tmux `ember_dfb_correct400_m50_retry1`已正常退出；
- log：`runs/logs/pi05_dynamic_k_semantic_address_direct_family_b_rank8_correct400_noreplacement_seed7_macro0050_trainr5_evalr6_c5353f3_gpu01_retry1_20260813.log`；
- 400-entry LoRA cache估算peak新增`535,986,176` bytes，仍在已检查quota内；
- 第一份无`retry1`的eval root只完成prepare；启动瞬间GPU1被他人新占约34GB，原子preflight拒绝启动，因而没有
  Worker、LoRA cache或rollout结果。它不得冒充失败实验或活动root。

完成边界：

- `completion.json`且`completed_macro=50`；
- `metrics.jsonl`覆盖macro1--50；
- macro25与macro50 checkpoint完整；
- launcher/torchrun正常退出，无failure artifacts或nonfinite；
- macro50 checkpoint schema、world5 rank states和run contract一致。

- 72/72 shards complete，18/18 worker return code为0；
- 400 rows及`(suite, task, init_state)` keys均唯一，无failure artifact；
- strict=`102/400`、breadth5，per-task=`0/1/40/11/0/43/7/0`，per-suite=`1/51/43/7`；
- top3=`94/102`，说明能力仍集中；相对semantic101为`82 retained/20 gained/19 lost`、churn39；
- 相对Dynamic-K100为`79/23/21`、churn44；相对old134为`80/22/54`、churn76；相对compiler138为
  `79/23/59`、churn82；相对online128为`79/23/49`、churn72；
- 相对v6-fast143的per-task差=`0/-2/-6/-26/0/+7/-13/-1`。

exact effective-BA对照中，task-mean offdiag cosine从semantic的`.77947`降到Direct-B的`.74895`，但K1
closed-loop只`101→102`且breadth`6→5`。因此删除family hidden/GELU只轻微缓解几何压缩，不能解决
Program到policy direction、held on-policy usefulness或shared capability coexistence。该方法按门终局non-pass，
不resume到100、不做小超参sweep、不补K1五臂controls。

分析artifact：同一root下`benchmark_comparison.json`与`effective_ba_task_geometry_comparison.json`。分析工具同时
修复了一个窄工程问题：registered historical Dynamic-K families应使用各自episode schema，而runtime通用
writer-input事实不应被方法名措辞误拒；该修复不改变任何raw row、score或科学合同。

## 6. Completed K4 nested video-dose adjudication

formal K4 root：

`runs/outputs/pi05_dynamic_k_semantic_address_direct_family_b_rank8_k4_correct400_noreplacement_seed7_macro0050_trainr5_evalr5_73b9514_gpu01_retry1_20260813`

- frozen worktree clean detached `73b9514`；gpu01物理`2,4,5,6,7`，5个Writer generators、15个persistent
  rollout workers；启动时同节点只有这5张卡适合使用，未等待或跨节点拼卡；
- validation8×50、correct、without-replacement seed7、显式K4、每condition总frame budget64；K1视频是K4集合
  的严格nested第一个元素；generation B8来自sealed K4 profile；
- 60/60 shards、400/400 rows、15/15 workers exit0、0 failed；wall=`1098.3835s`，overall=`.36417`
  episodes/s；Writer 400 entries由50个B8 batches一次生成，无重复forward；
- strict=`98/400`、breadth5，per-task=`1/0/42/8/0/41/6/0`，per-suite=`1/50/41/6`，top3=
  `91/98=92.86%`；相对v6-fast143逐task差=`+1/-3/-4/-29/0/+5/-14/-1`；
- nested K1→K4为`80 retained/18 gained/22 lost/280 retained failure`，net`-4`、churn40、Jaccard
  `.6667`、exact McNemar `p=.635828`；没有新task被解锁；
- 全400对effective-BA K1→K4 cosine mean/median=`.98787/.99225`、relative-L2 mean/median=
  `.15325/.13764`、norm ratio mean=`.99876`，排除K4能量坍缩；gained与lost没有可分离的有用方向；
- validation每task前4 states中，same-task centered variance/sample从K1 `.021674`降到K4 `.003438`
  （约`6.3x`），task-mean K1→K4 cosine=`.99604`；跨task mean offdiag只从`.74895`轻降到`.73816`。

因此set aggregator和动态K训练合同本身已工作：更多视频成功滤掉单demo偶然性，却主要稳定同一个窄而错误的
task mean。当前最早失效接口是set之前的task-grounded高层evidence/Procedure及其task-level functional credit，
不是视频数量、集合不变性、mapper、rank或LoRA能量。正式分析为同一root下
`k1_k4_nested_dose_analysis.json`。

第一次formal root（commit `ca28da8`）在任何GPU工作前因resume/start adapter inspection漏传`evaluation_k=4`
而fail closed；queue保持60 pending、无rollout/cache。`73b9514`只修复该窄工程合同并有回归测试，不改变Writer、
checkpoint、schedule或科学panel。

## 7. Active Task-Grounded Visual-Value implementation and profile

- canonical实现`9d43e82`；数学等价吞吐优化`690dea5`；均已push，完整CPU=`378 passed`；
- 唯一主变量是同一次joint forward中的task-token query/raw-patch Value，按真实输入顺序重算visual D/G，再以
  semantic address + layer/rank route写入原18x8 Program；无额外backbone forward、prediction/negative loss、
  expert target、reward或language-only Value；
- 全新config/launch/checkpoint/evaluator schema，不能误resume Direct-Family-B或profile state；
- 首次world4 profile=`58.2544s/macro`，同gpu02物理0--3、同world4/B20/K调度的Direct-Family-B matched为
  `46.2242s`，即`1.2603x`，超过`1.15x`门，因而没有启动formal；
- 只截断frozen prefix无用backward、合并bias-free evidence projection GEMM和共享visual reader调用；优化后
  `49.0775s/macro`，matched=`1.061727x`，通过门。K1--K4各6，functional/consistency=`.156108/.009484`，
  gradient norm=`.068287`，峰值allocated/reserved=`39.303/45.561GB`，无OOM且完整checkpoint/completion；
- profile只证明机制与吞吐，不是closed-loop成绩。正式训练必须fresh macro0，不加载任何profile checkpoint。

## 8. Retained canonical assets

- source policy：
  `runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000`；
- tokenizer：`models/tokenizers/openpi/paligemma_tokenizer.model`；
- target data：`data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a`；
- split：`configs/libero_24_8_8_v1/`；
- LIBERO assets：`.env.local`中的`EMBER_LIBERO_ASSETS_ROOT`；
- task experts：`runs/outputs/pi05_task_expert_bank_formal_step1000_r6_81101fe_20260807`中的统一step2000；
- current config：`configs/pi05_as_writer_dynamic_k_task_grounded_visual_value_rank8_v1.json`；
- historical exact roots与逐方法negative boundaries：`docs/research_history.md`和retained formal artifacts。

## 9. Immediate next evidence and continuous loop

K4裁决已经完成，不能再通过增加K、调set或mapper救这个checkpoint。下一fresh design保留动态K1--K4、真实
image/language/Action-probe context中的8个memory tokens、逐video保序、跨video置换不变set、20×8 M2P、
direct family-B与完整rank8 LoRA；只改变set之前的evidence owner：让冻结VLM中由exact language定位的真实视觉
内容成为LoRA Value，并同时提供稳定Semantic Core与有向视觉transition，而不是只让absolute Action memory作Q。
该设计必须继承v5.2/v6已经验证的task-grounded patch/value原则，又避免恢复其多套旧前端或language-value旁路。
正式公式、机制门、吞吐门和macro0→200裁决见
`docs/action_forecast_writer_dynamic_k_task_grounded_visual_value_design.md`。

正式fresh root：

`runs/outputs/pi05_dynamic_k_task_grounded_visual_value_rank8_formal_fresh0to50_r3_b20_caa2e30_gpu01_20260813`

- frozen worktree：`/data1/user/ymdai/worktrees/EMBER-tgvv-formal-caa2e30`，clean detached
  `caa2e3045fee7d4a5a2bbcfcc126fad9ec61832f`且commit已push；
- host/devices：gpu01物理GPU`4,5,6`，world3；launch前双节点live检查后，这三张卡是同节点当时真正适合且
  稳定的设备，不等待凑6卡；
- fresh macro0、B20、stop-after50、checkpoint25/50、num_workers0；不加载profile或Direct-Family-B state；
- `NCCL_P2P_DISABLE=1`，三rank均映射到physical GPUs 4--6对应的NUMA1；
- 首个macro为`65.1458s`，K1--K4各6，functional/consistency=`.156108/.009484`、gradient norm=
  `.068287`，三卡约100%利用率且峰值allocated/reserved=`39.298/45.546GB`；只证明正式合同已健康运行；
- tmux：gpu01 `ember_tgvv_formal_caa2e30`；精确命令与环境以该root的`run_contract.json`为准。

fresh 0→50现已完整结束：`metrics.jsonl`恰好50条、macro1--50唯一连续，K1--K4全程各6，全部loss/gradient/
seconds finite；macro25/50两个checkpoint均为world3且5个声明文件尺寸完整；`completion.completed_macro=50`，
总耗时`3398.5443s`。macro50 functional/consistency/gradient=`.114084/.005499/.049901`，只作训练健康证据。

同一root已从macro50以相同gpu01物理`4,5,6`和world3 exact-resume到100，tmux
`ember_tgvv_resume50to100_caa2e30`；首个续训macro51已完成，证明optimizer/scheduler/RNG/cursor恢复链工作。

macro25 checkpoint已完整写出且训练继续。用该checkpoint在gpu02物理GPU1完成K1部署定标：

`runs/outputs/pi05_dynamic_k_task_grounded_visual_value_rank8_k1_writer_generation_profile_val8x4_correct_gpu02p1_caa2e30_macro0025_retry1_20260813`

- fixed validation8×4、longest-first B8/B16/B32 LoRA/s=`.984266/.976097/.971736`；三者均stable、包含最长64帧、
  0 OOM，peak reserved=`12.973/13.451/13.455GB`；按规则锁B8；
- profile使用clean frozen `caa2e30`和同一正式macro25 checkpoint，只决定相同生成图的deployment batch；不读取
  closed-loop结果、不选择checkpoint；
- 第一份无`retry1` root在任何GPU forward前因误传LIBERO assets旧路径fail-closed；没有profile结果，不得冒充
  方法失败或正式定标。

macro50完成后立即做K1 strict paired correct400。历史v6-fast macro50也只有106，因此不以单个50点低于120提前
杀死整个0→200裁决；按design继续100/150/200并分析相邻checkpoint共同积累。

macro50 K1 strict400已从clean frozen `99c2323`启动：

`runs/outputs/pi05_dynamic_k_task_grounded_visual_value_rank8_correct400_noreplacement_seed7_macro0050_trainr3_evalr5_99c2323_gpu02_20260813`

- gpu02物理`1,2,3,4,6`，5个Writer generators、15个persistent rollout workers；validation8×50、correct K1、
  without-replacement seed7、generation B8；
- GPU0约36GB、GPU5/7高util而未选；所选1--4约349MiB/0% util，GPU6约4.9GB/0% util，符合owner允许安全
  低util共驻的边界，不等待第6卡；
- evaluator准入窄修复`99c2323`只允许util≤10%、used≤8GiB、free≥32GiB，超门仍fail-closed；全量CPU
  `383 passed`；不改变scientific panel、Writer、随机性或rollout；
- 5个generator已封存400-entry LoRA cache并release Writer，15个workers已扩出，queue正在执行；尚无结果。

之后继续：真实结果 -> 深入接口分析 -> 一个主要因果变量 -> authority -> canonical实现/机制/吞吐 -> fresh训练 ->
single-checkpoint strict评测。memory token、rank8和Dynamic-K都是方法变量，不是Goal。

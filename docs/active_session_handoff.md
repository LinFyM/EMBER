# EMBER Active Session State

更新时间：2026-08-13。本文是唯一实时实验状态入口；旧文档、Git快照与formal artifacts中的“当前/下一步”只
表示当时时点。稳定目标与owner要求见`docs/current_owner_requirements.md`，历史结果见`docs/research_history.md`。

## 1. Current truth

- 长期Goal处于active：同一shared Writer、同一single checkpoint的strict paired correct严格`>150/400`，同时
  保持高breadth、低能力换手和正确教学视频的内容/顺序因果性；
- 历史最好仍是v6-fast macro400：`143/135/125/128/129`；
- 唯一主工作树：`/data1/user/ymdai/projects/EMBER`；唯一主写分支：`codex/bci-continuation`；
- 最新完成architecture：Dynamic-K Semantic-Address Direct-Family-B Rank-8 Writer；其K1 macro50 strict为
  `102/400`、breadth5，按预注册门终局non-pass，不resume；
- 当前没有已经授权fresh训练的successor；下一项有信息量工作是为同一Dynamic-K checkpoint补齐正式K2--K4
  deployment/evidence合同，先判断多视频聚合是否产生真实few-shot增益，再选择下一训练单变量；
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

## 6. Retained canonical assets

- source policy：
  `runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000`；
- tokenizer：`models/tokenizers/openpi/paligemma_tokenizer.model`；
- target data：`data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a`；
- split：`configs/libero_24_8_8_v1/`；
- LIBERO assets：`.env.local`中的`EMBER_LIBERO_ASSETS_ROOT`；
- task experts：`runs/outputs/pi05_task_expert_bank_formal_step1000_r6_81101fe_20260807`中的统一step2000；
- current config：`configs/pi05_as_writer_dynamic_k_semantic_address_direct_family_b_rank8_v1.json`；
- historical exact roots与逐方法negative boundaries：`docs/research_history.md`和retained formal artifacts。

## 7. Immediate next evidence and continuous loop

当前Writer从fresh训练起真实覆盖K1/K2/K3/K4，但现有formal evaluator只执行过K1。下一步先补齐一个canonical、
fail-closed的K2--K4 video-set generation/cache/evidence/evaluation合同，并用同一macro50 checkpoint先做K4 strict
correct400。它回答的是“当前cross-video set是否真正产生few-shot增益”，不是续训、换checkpoint或反向救K1结果。
若K4仍无material improvement，下一训练设计不得继续在mapper上做小修，而应把最早失效接口前移到高层Program的
可识别性与train24 functional credit对held on-policy方向的错位。若K4显著更强，则诚实把当前setting视为few-shot，
随后补K1--K4 scaling与视频因果controls。

之后继续：真实结果 -> 深入接口分析 -> 一个主要因果变量 -> authority -> canonical实现/机制/吞吐 -> fresh训练 ->
single-checkpoint strict评测。memory token、rank8和Dynamic-K都是方法变量，不是Goal。

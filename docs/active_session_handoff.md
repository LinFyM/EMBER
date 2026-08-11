# EMBER Active Session Handoff

## 1. Current truth

这是唯一跨session科研入口，覆盖历史文档、Git快照和artifact中的旧“当前/下一步”。

- 长期Goal未完成：同一shared method、同一single checkpoint的strict paired correct必须严格超过
  `150/400`，并继续提高absolute、breadth、稳定共同积累和teacher-video时序因果性。
- 历史最好single checkpoint仍是v6-fast macro400：
  `correct/same/wrong/shuffled/reversed=143/135/125/128/129`。
- 最新uniform pivot-rank14路线已经完成全部预注册裁决并退役。不存在待跑Gate B/C；当前唯一active
  successor是PICK，但只完成design authority，尚未实现、profile、训练或rollout，也没有EMBER GPU进程。
- 本次仓库整理已经完成：退役可执行路径、重复历史文档、旧worktree/branch与明确临时资产已清理，正式
  evidence和可复用基础保留。PICK必须从canonical owner替换与前序机制门推进，不得从旧命令恢复实验。
- canonical workspace是`/data1/user/ymdai/projects/EMBER`，主写分支是`codex/bci-continuation`。正式GPU
  工作以后仍须来自clean pushed commit的detached frozen worktree。

## 2. Latest formal decision

### 2.1 Online Gate B

online-regenerated rank14 zero-Program root：

`runs/outputs/pi05_v6_qv_rank_reserved_native_reward_correct400_macro0000_20260811`

- clean frozen commit：`0fd823f8cb5ab45164b185c0a42cb358044b095d`
- strict=`128/400`、breadth7
- per-task（Spatial1/3, Object1/3, Goal3/6, Long1/2）：`1/1/47/29/0/36/13/1`
- 相对immutable old full-rank macro0 `134/400`：retained/gained/lost=`113/15/21`、churn36
- 400 rows/cache、48 shards、task/state/video/env/policy RNG pairing和launcher completion完整

这是可信的端到端non-pass，但old/new分别使用18/12 generators，旧cache在worker内部局部拼B8，改变了
co-batch、position、padding和tail；因此不能把全部退化归于rank14 compression。

### 2.2 Compiler-only deconfounding

一次性clean去混杂root：

`runs/outputs/pi05_v6_qv_rank_reserved_compiler_only_old134_to_rank14_correct400_20260811`

- evaluation commit：`6db37c1138e1357108d07a3be3b3af5449a72932`
- decision evidence：`compiler_only_diagnostic_evidence.json`
- schema：`ember_pi05_v6_qv_rank_reserved_compiler_only_decision_evidence_v1`
- single-A40 transform：400/400 entries、50×B8、36 q/v targets；1600 action tensors bit-exact，400 video
  identities exact；0 teacher read、0 Writer/policy forward、0 rollout/update；wall=`69.464s`
- strict replay：`gpu02:2,3,4,5`，4 cards×3 persistent workers，无NCCL；48/48 shards、400/400 rows、
  12/12 return0；wall=`1037.920s`，overall约`1387.39 episodes/hour`
- result=`138/400`、breadth7、per-task=`1/1/46/32/0/35/22/1`
- old→compiler retained/gained/lost=`119/19/15`、net`+4`、churn34、Jaccard`.777778`、p=`.607591`

预注册门是correct`>=130`、breadth`>=6`、lost`<=10`。前两项通过，但lost15，所以
`counterfactual_gate_passed=false`。总分`+4`完全由Long1净`+11`掩盖Spatial/Object净`-3/-4`；breadth7中的
Spatial1仅`1/50`，Goal3仍`0/50`。这属于target-heterogeneous capability rotation，不是稳定累积。

compiler→online retained/gained/lost=`115/13/23`、net`-10`、churn36，其中Long1净`-9`。compression与
regeneration都是独立换手源；修好batching不能自动恢复old support。

正式终局：

- `original_gate_b_passed=false`
- `counterfactual_gate_passed=false`
- `retroactively_changes_original_gate_b=false`
- `authorizes_cycle1=false`

uniform pivot-rank14 base、rank14+2 cycle1、Gate C、controls与fresh训练全部关闭。该结论不外推为“视频、
Reward、continuous tangent或所有rank-reserved topology无效”。

## 3. Fixed scientific foundation

### 3.1 Data and split

- benchmark：LIBERO Spatial/Object/Goal/Long共40 tasks。
- development split：`configs/libero_24_8_8_v1/`，每suite 6 train / 2 validation / 2 test，共24/8/8。
- source corpus：LIBERO-90 specification-only audit排除19个与目标40 exact semantic/composition重合tasks，
  保留71 tasks×50成功episodes。
- source policy不能使用读过目标40 actions的`pi05_libero`；normalization只来自filtered source并冻结。
- validation选定方法后才合并32 source / 8 test；不得按outcome改task、video、state或topology。

### 3.2 Information wall

- canonical deployment：exact task language + exactly one action-hidden teacher video。
- video是唯一dynamic value；language不能形成LoRA bypass。
- 禁止teacher action/proprio/reward/terminal、task ID、filename、object pose、hidden normalization和其它元数据。
- 每episode一条video生成一套完整38-target public rank-16 LoRA；不平均/融合video、LoRA或checkpoint。
- frame stride5；frozen source policy无trainable parameters；no-video/step0 functional identity。
- held每rollout从正确task的50条videos无放回取一条，不挑最好video。

few-shot可作为未来新设计，但K4历史只证明它能减少部分单video偶然性；没有证明它解决共享credit、时序语义
或task drift。任何few-shot必须与one-shot matched比较并保持action-hidden。

### 3.3 Evaluation

- official preprocessing：render256/model224、两相机180° rotate、7D state/action、10 flow steps、执行前5
  actions后replan、dummy settling10、成功即停、horizon 220/280/300/520。
- correct/same-task-other/cross-suite-wrong/shuffled/reversed/no-video严格配对state、env/policy RNG、video
  ordinal和处理；shuffle/reverse对真实frames重排后完整forward。
- evaluator使用cost-balanced dynamic queue、long-first、persistent model/env；不静态task/GPU分配。
- checkpoint只认single paired400；报告aggregate、per-task/suite、breadth、retained/gained/lost与churn。

## 4. Canonical assets

- frozen source policy：
  `runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000`
- task-expert bank：
  `runs/outputs/pi05_task_expert_bank_formal_step1000_r6_81101fe_20260807`
- split：`configs/libero_24_8_8_v1/`
- source/target manifests：`configs/pi05_source_corpus_v1/`、`configs/pi05_target_data_v1/`
- tokenizer/model/data/simulation paths：由CLI或`.env.local`提供；历史A100绝对路径只作provenance
- formal results：`runs/outputs/`；迁移/retained manifests：`evidence/`

source policy是下游inference asset，不支持source-SFT exact resume。历史formal outputs、400-row results、
task experts、数据和checkpoints默认保留；screen/profile临时权重只有在确认无consumer后清理。

## 5. Task experts: what they do and do not do

同一formal root中的24 tasks已统一续到step2000。step250/500/1000/1500/2000的development-train
direct-expert closed-loop为`432/557/624/638/658` of 1200；step2000有23/24 tasks非零、task9仍0。

Experts能：

- 给出“该train task的什么public LoRA确实能闭环工作”的policy-effective task-level target；
- 提供正常task-local SFT LoRA的能量、rank坐标、跨target分配和有效方向参考；
- 作为train24 privileged机制监督，降低纯functional query的歧义。

Experts不能：

- 证明held task从video生成LoRA；
- 区分同task的不同videos，因为同task expert target对所有video恒定；
- 提供correct/shuffled/reversed的时序监督；
- 直接作为held dictionary。soft/hard bank的`15/80`/`3/80`已否定当前部署方式。

因此expert reconstruction和LoRA健康度只能辅助，最终仍由paired closed-loop视频controls裁决。

## 6. High-level causal chain

完整逐方法表见`docs/research_history.md`。必须继承的连续认识是：

1. v4证明视频与顺序会影响行为，但模型利用absolute-time/action-phase shortcut，shuffle可更好。
2. v5/v5.2把task identity与Procedure分开并增强wrong-video区分，但正确顺序没有稳定传进compiler。
3. v6-fast达到143，说明特定architecture×task-complete recipe有效；后期却退化，累积仍不稳定。
4. v7/v8/v10及Loom/Core/Prior说明内部时序更漂亮、去DC或更强fusion都不保证policy-effective direction。
5. Target-Spectral、Policy-Lane/Atom/Owned-Factor说明高rank、正交、均匀能量、更多capacity不是目标。
6. SFB union193但single127；VR改善functional evidence却闭环退化：surrogate与union都不能选模型。
7. K4/few-shot、trace与video routing证明视频能被读取，但不能自动解决full24 credit retention和正确时序。
8. Task experts有效但不是video teacher；address/barycentric/bank证明reconstruction、route和expert复现仍不足。
9. Balanced residual显示不同video correction近正交；RLS显示offline feature-row保留不等于held occupancy。
10. Reward-Credit显示on-policy reward能形成健康continuous tangent，但q/v native BF16 factor写出低于ULP。
11. rank14最终显示uniform compression本身损伤support，online regeneration又额外换手；tiny几何误差不够。

稳定规则：视频被使用不等于正确使用；LoRA健康度是约束不是目标；task drift没有单一原因；small panel和
checkpoint union会误导；新topology必须从train24机制推导，不能按held得失设计。

## 7. Active successor and remaining questions

当前已选后继为
`docs/action_forecast_writer_policy_innovation_consensus_key_design.md`中的PICK。它只替换Balanced-v2的condition
evidence owner：用frozen source-policy zero-image-subtracted high-level video innovation构造相同的balanced
static/causal key；frozen-v6 600 tensors、Program memory、full-rank16 compiler、blind full48 update和one-shot
部署保持不变。

选择依据是sealed train24×50 cache中causal descriptor的same/cross cosine约`.9195/.1565`、reverse约
`-.9755`、shuffle约`-.0053`，而Balanced-v2最新最早失败正是同task 50-video correction近随机正交。该cache
只作train-only design evidence，不能替代online机制或closed-loop。

PICK必须回答：

- online raw-frame key能否重现cache中的same-task与有向顺序结构；
- 一致condition neighborhood能否让Program/effective-BA correction脱离随机正交，而不抹掉held task差异；
- blind AS credit在key修复后是否仍是closed-loop首因；
- 额外frozen innovation forward能否在A40上保持可接受吞吐；
- strict400能否同时提高absolute、breadth和retention，并让correct优于视频controls。

仍未授权的其它候选包括pivot15+1、train-derived mixed topology、few-shot set encoder和新的reward credit。
它们不能与PICK并行实现或作为失败时自动fallback。特别是pivot15+1不能恢复旧balanced-rank15，rank1 tangent
capture历史约`.9185`，也不能因base多保留一列直接启动cycle1。

## 8. Runtime and GPU boundary

任何未来GPU launch前：

- 同时live检查`gpu01/gpu02`，选一个节点；
- 使用该节点所有真正空闲、健康且能提高吞吐的A40，没有6-card cap；
- 不等待凑卡、不dummy占位、不跨节点拼碎片、不干扰他人；
- evaluator无NCCL；多卡训练用`NCCL_P2P_DISABLE=1`、NUMA physical/local rank mapping和deferred-NCCL；
- fresh按live world size设计，exact-resume锁定原world size；
- 先查`/data1`独立user quota与峰值预算。

吞吐优先：原生BF16/TF32和batch低位差异可接受，不用batch1、重复forward、扩dtype、ULP/dither或内容hash
追微小复现。科学门只保护信息墙、pairing、shape/finite、串样、OOM、asset、checkpoint和resume语义。

## 9. Repository state and handoff

2026-08-11 compiler-only结果由commit`3a6f801d08facb3e855ab24f84e0b53cb8802e88`封存。随后cleanup完成：

- `04ed4e6`把约3.65万行追加式Markdown压成约3400行authority/history，删除错误train-only normalization与
  两个无caller旧配置负载；
- `4f172dd`删除退役rank/compiler的3个configs、14个专用runtime/contract/launcher/compiler owners、孤立
  artifact hasher及专用tests，净删8408行；results-only历史schema与可复用v6/Reward基础保留；
- 主工作树外12个旧worktrees与全部topic worktree已删除；本地和origin现在都只保留`main`与
  `codex/bci-continuation`，旧refs仍由6.15MB、138-head灾备bundle和Git历史覆盖；
- 删除38个已结束profile/smoke roots中的非formal checkpoint payload，实际约74.69GB；保留各root的contract、
  metrics、summary/completion。另清理34个uv临时目录、17个零字节日志、3个空/错误roots和3个空转tail进程；
- 为未来效率保留可复用`.cache/uv`主体、`.venv`、formal results/checkpoints、paired raw rows、task experts、
  source policy、data/models、feature caches、acceptance与migration evidence；
- 加载现有`.env.local`资产映射后的完整CPU回归为`361 passed`；compileall、CLI、Markdown links和diff-check通过。

本次收尾没有启动GPU、训练、rollout或profile，也没有修改formal实验结果。删除的源码/设计仍可从`3a6f801`
精确恢复；profile checkpoint payload没有保留副本，但它们从未是formal/consumer asset。

PICK实现前先确认当前HEAD、clean和origin同步，并按新design的owner/retirement边界替换旧key。GPU前必须重新
live检查双节点、quota、frozen commit与峰值；CPU/cache与live mechanism门不过不得创建formal训练root。

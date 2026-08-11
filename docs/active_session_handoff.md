# EMBER Active Session Handoff

## 1. Current truth

这是唯一跨session科研入口，覆盖历史文档、Git快照和artifact中的旧“当前/下一步”。

- 长期Goal未完成：同一shared method、同一single checkpoint的strict paired correct必须严格超过
  `150/400`，并继续提高absolute、breadth、稳定共同积累和teacher-video时序因果性。
- 历史最好single checkpoint仍是v6-fast macro400：
  `correct/same/wrong/shuffled/reversed=143/135/125/128/129`。
- 最新uniform pivot-rank14路线已经完成全部预注册裁决并退役。PICK随后完成canonical implementation、
  `345 passed`回归、raw-frame门和world6 discarded full48 profile；profile仅因condition=`483.61515>200`
  non-pass，其余14项机制、动作与吞吐门通过，未获formal训练资格。
- PICK-GC已完成formal fresh`0→10`及strict paired correct400并正式退役：`138/400`、breadth6、per-task
  （Spatial1/3, Object1/3, Goal3/6, Long1/2）=`1/3/48/33/0/39/14/0`；相对immutable macro0
  retained/gained/lost=`118/20/16`、churn36。它未过`correct>=144`与`lost<=8`门，不得resume到25、补controls
  或做参数sweep。
- 当前唯一active successor design是OSG-PC：保留PICK-GC的B20 proposal，只将其投影到“不增大train24成功
  on-policy executed-prefix loss”的parameter-free可行锥，再走同一full48 Program write。它已完成唯一canonical
  实现、fresh schema与CPU/synthetic门；尚未做discarded live profile、formal或评测，当前没有EMBER GPU进程。
- 本次仓库整理已经完成：退役可执行路径、重复历史文档、旧worktree/branch与明确临时资产已清理，正式
  evidence和可复用基础保留。任何successor必须原位替换canonical owner并重过对应机制门，不得从旧命令恢复实验。
- canonical workspace是`/data1/user/ymdai/projects/EMBER`，主写分支是`codex/bci-continuation`。正式GPU
  工作以后仍须来自clean pushed commit的detached frozen worktree。

## 2. Latest formal decision

### 2.1 PICK-GC formal macro10

formal训练root：
`runs/outputs/pi05_pick_gc_goal_causal_formal_fresh0to10_r4_b20_c2e1ff8_20260811`；训练commit
`c2e1ff878b6b68cb5bc45bb5443cdbd54ab8e62a`。10个macro全部rank48，condition=`83.61--152.88`，最终
20,971,520值FP32 Program memory RMS=`3.5493e-6`且全部非零。functional loss约`.093--.100`无下降趋势，
所以机制写入健康但offline surrogate没有形成更好趋势。

strict root：
`runs/outputs/pi05_pick_gc_goal_causal_correct400_noreplacement_seed7_macro0010_retry1_398425e_20260811`；
48/48 shards、400/400 rows、12/12 workers exit0。结果`138/400`、breadth6；macro0→macro10严格相同
task/state/video/env/policy RNG pairing下 retained/gained/lost/both-fail=`118/20/16/246`、churn36。各基线同一
task顺序为：v6-fast143=`0/3/46/37/0/36/20/1`、macro0 134=`0/5/48/34/0/35/11/1`、compiler138=
`1/1/46/32/0/35/22/1`、online128=`1/1/47/29/0/36/13/1`、PICK-GC138=
`1/3/48/33/0/39/14/0`。

paired缓存的effective-BA整体范数中位比=`1.000016`、cosine=`.99999724`、相对L2=`.002397`；action端
相对L2=`.003968`。这套小而非零的切向写入已造成20 gains与16 losses，排除identity/能量塌缩，也不支持靠
扩大scale或rank补救。结合pre-formal condition/closure门，最早失效接口被推进到blind train24 offline
source-action cotangent→held on-policy useful support/coexistence。正式decision evidence位于eval root的
`pick_gc_formal_decision_evidence.json`。只有PICK-GC+blind-credit组合被淘汰；ordered goal-causal key、
condition-local FP32 Program、native rank16 compiler、few-shot与新的on-policy credit均未被该结果否定。

### 2.2 Online Gate B

online-regenerated rank14 zero-Program root：

`runs/outputs/pi05_v6_qv_rank_reserved_native_reward_correct400_macro0000_20260811`

- clean frozen commit：`0fd823f8cb5ab45164b185c0a42cb358044b095d`
- strict=`128/400`、breadth7
- per-task（Spatial1/3, Object1/3, Goal3/6, Long1/2）：`1/1/47/29/0/36/13/1`
- 相对immutable old full-rank macro0 `134/400`：retained/gained/lost=`113/15/21`、churn36
- 400 rows/cache、48 shards、task/state/video/env/policy RNG pairing和launcher completion完整

这是可信的端到端non-pass，但old/new分别使用18/12 generators，旧cache在worker内部局部拼B8，改变了
co-batch、position、padding和tail；因此不能把全部退化归于rank14 compression。

### 2.3 Compiler-only deconfounding

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

## 7. Active OSG-PC design and boundary

PICK-GC的formal结果把因果链进一步收窄：frozen-policy goal/causal innovation能读取
same-task与顺序，full48 key可解，condition-local FP32 Program会连续积累，native compiler会把它传到effective
BA与action；但blind source-action cotangent没有在held rollout occupancy上稳定积累support。下一设计必须保留
这些已通过接口，只改变credit/occupancy这一项主要变量。

当前已选的OSG-PC定义为：对每task保持`d0=-grad_H L_source(B20)`，用同一correct-video LoRA做K4 train24
random-reset rollouts；每条成功episode的executed-prefix CFM cotangent `r_e`形成`<r_e,d><=0`，然后取离`d0`
最近的Euclidean feasible-cone projection。无success或raw已可行时逐元素退化为blind proposal；非零projection
仍严格保持source descent。binary reward只选择已有support，不用作sub-ULP improvement gradient。

它回答：

1. train24奖励或on-policy occupancy如何提供比source-action B20更接近真实闭环的support约束；
2. 如何保持exact language + one action-hidden video是部署时唯一动态输入，并让正确顺序成为有用更新的必要key；
3. 如何在每个task-complete更新中避免牺牲另一个task的已有reward support，而不读取held actions/outcomes；
4. 如何直接写入现有FP32 Program并复用native rank16 compiler，避免Reward-Credit的BF16 ULP和uniform rank14
   compression/regeneration损伤；
5. 哪个小规模train-only reward/occupancy gate能快速否决它，以及何时必须做strict paired400。

完整公式、owner替换、CPU/live/profile/formal门见
`docs/action_forecast_writer_on_policy_success_guarded_program_credit_design.md`。Reward-Credit、RLS和rank14历史
只能作为边界证据，不能直接恢复。few-shot、task-level manifold supervision与heterogeneous topology仍是
开放方向，不能与OSG-PC并行实现。

OSG-PC已经原位实现为唯一active path：`v6_reward_credit.py`只保留per-success retention VJP与最多4条约束的
解析Program锥投影；`reward/rollout.py`只让成功episode进入replay，失败episode replay/gradient严格为0；每个
task仍只做一次正确视频encode，rollout后只重解FactorHeads。训练继续复用同一full48 solver，并在profile中
额外报告“实际full48写入”对每条success guard和source descent的影响，以暴露task-local safe RHS在shared write
中是否重新被破坏。唯一config为`configs/pi05_v6_on_policy_success_guarded_program_credit_v1.json`，状态
`active_cpu_ready_awaiting_live_profile`。加载`.env.local`后的fresh完整CPU回归为`340 passed`，compileall与
`git diff --check`通过；这只证明实现合同，不构成GPU机制或closed-loop证据。
retention sampling按历史完整K4 task panel生成后只索引success row ordinal，因此失败actions仍不保留，且其它lane
的success集合不会使已有guard换Monte Carlo样本。deployment adapter/episode已升fresh v9，formal status与
checkpoint-curve registry显式支持OSG-PC的`0/5/10`，不再误用PICK-GC v8或固定`0/10/25/50`。

## 8. Runtime and GPU boundary

任何未来GPU launch前：

- 同时live检查`gpu01/gpu02`，选一个节点；
- 使用该节点所有真正空闲、健康且能提高吞吐的A40，没有6-card cap；
- 不等待凑卡、不dummy占位、不跨节点拼碎片、不干扰他人；
- evaluator无NCCL；多卡训练用`NCCL_P2P_DISABLE=1`、NUMA physical/local rank mapping和deferred-NCCL；
- fresh按live world size设计，exact-resume锁定原world size；
- 先查`/data1`独立user quota与峰值预算。

2026-08-11 deployment launch前的旧快照曾有`gpu02:0-5`空闲，单卡profile/vertical使用`:1`并已释放。
18:47+08:00新快照中`gpu02:1-5`空闲而`:0/:6/:7`属于他人，`gpu01`只有`:2/:4/:6/:7`空闲；没有单节点
world6。train24 full48按可整除且不跨节点的world4/local6重过profile；实际physical topology为gpu02
`:2,:3,:4,:5`，两侧NUMA各2卡；`:1`有历史corrected ECC/remap而未选，第五张空卡也不能保持固定local task数。
PICK-GC formal与strict400均已完成并释放四卡；完成后selected physical2--5均0MiB/0%且无本次compute process。
`strg01`最近一次报告`/data1`用量`508362308 KiB`、quota`1073741824 KiB`；这是漂移快照，任何新launch仍需
同时重查两节点、quota与峰值预算，不能把旧空闲状态当预约。

OSG-PC实施后的新快照为2026-08-11 22:22--22:23+08:00：`gpu01`只有物理`:2/:4/:6`空闲，未选；
`gpu02:0--5`均为`0 MiB/0%`、P8且无compute process，`:6/:7`由他人占用。深度健康检查显示`:0--5`均无
volatile/uncorrectable ECC、pending repair或remap failure；`:1`只有历史累计6次已纠正DRAM与1条已纠正remap，
当前pending/failure均为no，故可用但须发射前后监测。OSG-PC train24 profile据此只改执行拓扑为单节点
world6/local4，NUMA0为物理0--3、NUMA1为4--5，继续设置`NCCL_P2P_DISABLE=1`；matched基线为world6/local4
`507.30541240703315s`，production wall门为`<=1.25x`。当时`strg01 /data1`用量`509424560 KiB`、quota
`1073741824 KiB`，shared available约85T，fresh profile峰值估计`<2 GiB`且目标root不存在。以上仍不是预约；
clean pushed topology seal和detached frozen worktree完成后，真正launch前必须再同时复查两节点、进程、健康、
quota与fresh root。

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
- cleanup封存时加载`.env.local`的完整CPU回归为`361 passed`。PICK替换退役Reward/RLS测试与实现后，当前完整
  回归为`345 passed`；新增真实encoder合同验证zero-image baseline只算一次、每个物理frame只编码一次。
  compileall、config fail-close、diff-check通过；architecture gate无hard violation、无parallel family，active
  source净减少1984行；`5200bee`的formal evidence seal后完整回归为`346 passed`。

PICK与PICK-GC的raw/full48 roots、contracts、logs、metrics和completions均已保留，discarded profile没有memory
checkpoint。PICK-GC部署证据root为
`runs/outputs/pi05_pick_gc_goal_causal_writer_profile_val8x4_correct_gpu02p1_717b561_20260811`和
`runs/outputs/pi05_pick_gc_goal_causal_zero_memory_vertical_retry1_val8x1_correct_b32_gpu02p1_717b561_20260811`。
首次vertical的staging-path失败及retry完成后的CPU finalizer字段失败都保留为engineering evidence；两者未改变
科学合同或GPU结果。`5200bee` deployment seal与`09bbed3` world4 profile authority均已push并有detached frozen
worktree；world4 profile exit0、14 checks全true且没有checkpoint。formal macro10 checkpoint、strict400 raw rows、
cache、launcher completion和决策证据也已保留；结果138/breadth6/lost16已封存并关闭resume与controls。OSG-PC
canonical实现已由`e22cff1`封存并push；当前只在该科学commit上追加上述live world6/local4执行拓扑authority，
完成CPU复验、push与detached frozen worktree后做一次discarded full24 K4+B20 profile。profile未过全部预注册
机制、吞吐及实际guard传递证据，不得formal。

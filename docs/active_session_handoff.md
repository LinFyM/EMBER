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
- 当前唯一active successor PICK-GC只把PICK的static block换为terminal goal residual并保留causal-prefix。
  implementation阶段`345 passed`、exact raw full48、world6 discarded mechanism、B8/16/32吞吐和zero-memory
  deployment vertical已全部通过，seal后完整回归为`346 passed`并由`5200bee`push封存。随后live资源没有单节点
  world6；执行拓扑只改为world4/local6。`09bbed3` discarded profile已经逐字段exact复现world6机制payload并
  通过归一吞吐门，formal资格现已重新打开。尚未formal训练或做strict paired400，目前没有EMBER GPU进程。
- 本次仓库整理已经完成：退役可执行路径、重复历史文档、旧worktree/branch与明确临时资产已清理，正式
  evidence和可复用基础保留。PICK-GC必须从canonical owner替换与前序机制门推进，不得从旧命令恢复实验。
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
`docs/action_forecast_writer_policy_innovation_goal_causal_key_design.md`中的PICK-GC。PICK本身已证明frozen
source-policy zero-image-subtracted innovation能在raw frames读取same-task与顺序、full48 solver能把credit传到
Program/LoRA/action且吞吐可接受；它最早只在48-key conditioning失败。

PICK-GC只把static mean换为`terminal quartile - whole-video mean`，保留PICK原有centered causal prefix。
sealed train24×50 cache中complete same mean/median=`.90260/.91604`、cross=`.13455/.11375`、
reverse=`-.80305/-.80877`、shuffle约0，50个correct24 panels均rank24且condition max=`21.62`。相比PICK，
same-task略降但过既定门，cross与condition明显改善。terminal-minus-initial版本会漏掉保留首尾的中间阶段
shuffle，reversal-symmetric endpoint则same mean仅`.86177`；两者均已拒绝，不实现。

PICK-GC继续保持historical v6 600 tensors、完整rank16 compiler、B20 blind full48 credit和单个FP32 Program
memory。前序结果为：

- exact raw full48 rank48、condition=`152.45803`、same=`.94501`、reverse=`-.81318`、shuffle=`-.09783`、
  target-language wrong=`.12235`；shuffle只以约`.00217`余量过门，不夸大其分离强度；
- world6 mechanism condition=`152.61008`、correct motion/cotangent=`.96457`、negative/correct=`.03901`、
  retained/null=`24/24`，Program/LoRA/action与四suite breadth闭合，wall ratio=`1.13558`；
- B8/B16/B32吞吐=`.47119/.47244/.47299` LoRAs/s，均stable并含67帧最长video，按规则选B32；
- zero-memory四suite LoRA `76/76` tensors与action均bit-exact，Program memory为0；canonical 8-entry cache、
  Writer释放、source policy复用和8/8 rollouts通过，smoke观测`4/8` success不作选择门。

因此raw conditioning、Program→LoRA→action和部署图已不再是最早科学接口。world4已经证明同一full24 solve在
新执行拓扑保持机制/吞吐；下一科学裁决只剩blind AS credit能否在held on-policy occupancy上让同一checkpoint
共同积累support，由formal fresh`0→10`及其strict paired400判断。

仍未授权的其它候选包括pivot15+1、train-derived mixed topology、few-shot set encoder和新的reward credit。
它们不能与PICK-GC并行实现或作为失败时自动fallback。特别是pivot15+1不能恢复旧balanced-rank15，rank1 tangent
capture历史约`.9185`，也不能因base多保留一列直接启动cycle1。

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
该profile已完成并释放四卡；formal fresh launch前仍必须重查，不能把快照当预约。
`strg01`最新报告`/data1`用量`508269652 KiB`、quota`1073741824 KiB`，profile/vertical实际新增远低于1GiB，
formal加随后的correct400峰值仍估计不足2GiB。

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
worktree；world4 profile exit0、14 checks全true且没有checkpoint。当前下一步是commit/push formal-ready reseal、
建立新frozen worktree后fresh`0→10`并立即strict paired400；未过`144/breadth6/lost8/gained>lost`不得resume到25。

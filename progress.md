# EMBER progress

更新时间：2026-08-25。本轮G1实现起点：clean pushed `main@13ca36662e430d9e33afa9e26fa3835d859abc78`。

## 当前状态

专家回复已经收到、完整阅读并固化。active design现为`docs/event_conditioned_policy_compiler_design.md`中的
**ECP Native-Factor Compiler**，核心架构、数据角色、阶段Gate、最终controls和停止条件均已明确。

专家1416行原始回复已完整保存为`docs/expert_review_20260824_native_factor.md`，逐行内容与附件一致，仅换行从CRLF标准化为LF；
active design明确是解释/执行层，不能替代原文。

全仓库orientation和两轮owner复核已完成，owner已正式许可推进。当前只推进G1 task-local free-code capacity oracle；没有恢复
旧Writer/realizer/GOMQ/PECS/人工process路线，也没有联系专家。

G1 canonical实现面已接通。首轮formal held5 free-code优化与strict250已完成：唯一rank16 candidate为`88/250`，relative recovery
`45/67=0.6716`、breadth`3/5`、高于carrier`2/5`、carrier retention`30/43`，逐task为`33/18/37/0/0`；因此Gate为
`non_pass`。全部250 paired rows、47 shards与15 workers完整，Action Meta关闭，失败是科学结果而非运行故障。

按Gate合同完成的read-only span/response分析定位到当前scalar output pooling的结构性上限：无bias q/v组合位于base weight列空间；
q只能覆盖`1024/2048`输出维。action-in带bias且可跨output types相减，精确上限为`span(column_space(W),bias)`、至多`33/1024`。
15个known-success mobile-rank4
reference整体只保留约55--56% update energy。将independent member正交投影到该上限后的paired strict250为`109/250`，逐task
`34/30/45/0/0`；原independent mobile authority为`120/250`且Goal/Long为`11/8`，投影单独抹掉了两个process-sensitive suite。

q-head修正的formal optimization与strict250已经完成：唯一rank16 candidate为`84/250`，逐task`28/21/35/0/0`、relative recovery
`0.6119`、breadth3/5、高于carrier2/5、carrier retention`24/43`，Gate non-pass。step500 generated update与known-success references
整体cosine仅约`0.06`，所以增加的q自由度没有被随机近均匀dense logits实际利用。

随后稳定native-bank投影诊断把latest member materialize为同一唯一rank16，在strict250达到`94/250`、逐task`24/24/44/1/1`；
relative recovery、breadth、Goal/Long和四task高于carrier均成立，但retention仅`22/43`，故不是Gate pass。该结果证明稳定bank内存在
process-sensitive闭环方向，并把最早失效接口推进到free-logit可达优化与retention。

latest-only解析free logits的精确step0 strict250已完成：`100/250`、逐task`24/28/45/3/0`，relative recovery`0.851`；breadth4/5、
Long 0、仅3/5高于carrier且retention`22/43`，Gate仍non-pass。step0与解析投影residual cosine为`0.952--0.964`，第一次Adam更新后即
降至`0.039--0.070`；五task 500-step formal也未在预注册内部effect/update证据上恢复step0，因此没有用held reward选择被扰动checkpoint。

set-valued formal与strict250已完成：每task按fixed50 count选择carrier/independent/latest/independent/independent，结果`111/250`、
逐task`35/29/45/2/0`，relative recovery`1.015`且retention`34/43`；breadth4/5、Long 0、仅3/5高于carrier，Gate non-pass。

最早接口现为signed-measure闭式初始化的数值稳定性：`1e-3` span threshold使scatter inverse condition number可达约`1e6`，task94
FP32实际direction cosine最低为input `0.978`、output `0.883`。只把小型初始化solve的sufficient statistics改为FP64后，真实task94
forward/materialization两侧minimum cosine均恢复到`>=0.99999988`；candidate、rank、pooling、loss、38 hooks、唯一rank16和Action Meta 0
均不变。clean pushed formal与同一strict250已经完成：`116/250`、逐task`35/34/44/3/0`、relative recovery`1.090`、
retention`35/43`；但breadth4/5、Long0且仅3/5高于carrier，Gate仍non-pass。

FP64已排除数值失真后，最早剩余结构接口是action-in whole-vector output pooling：`32 -> 1024`真实Y共享一个scalar signed measure
时必然受限于`span(column_space(W),bias)`、至多`33/1024`。paired response只把task94的action-in target恢复为known-success
independent mobile，其它37 targets保持当前native candidate，Long从`0/50`变为`1/50`。当前canonical修正因此按native input width
把action-in真实Y切成32个32D blocks，各block独立signed pooling；完整response counterfactual为`118/250`、逐task
`35/35/44/3/1`、breadth5/5、4/5高于carrier、retention`35/43`，数值上满足全部G1门，但因task94 action-in来自privileged
reference而不是native pooling，不能冒充G1 pass。候选索引、四类bank、rank、scale、唯一rank16和G1/G3边界不变。

专家复核锁定的是远程`main@7ab5a04`。其后`6fdaeb8`只删除退役代码/人工资产并整合文档，没有新增实验结果；专家指出的当前
Stage 0实现缺口已在瘦身后的代码中复核：q/v owner仍来自layer input/residual，尚无真实38-target input/output hooks。因此该科学
裁决可直接应用于当前活动树。

## 专家裁决已固化

- ECP继续推进，名称细化为ECP Native-Factor Compiler；
- 取消neural `q_pi -> fixed effect-code realizer -> LoRA`前置链；
- privileged experts/effects只作nonparametric set-valued training critic；
- Video Program固定为owner-specific language/scene/ordered events及`rho/tau/sigma`；
- 第二pass读取38个target的真实native inputs/outputs与动态differences；
- Program通过signed pooling产生mobile rank4，与frozen rank12 carrier拼成唯一rank16；
- 当前唯一下一步是fold0 held5 task-local free-code strict250，不先训练fresh Program/compiler；
- 通过后依次进行Natural Program、frozen-Program shared compiler、joint Writer、conditional outer credit和final fresh；
- validation8与完整video controls的资格门、Test8 sealed规则及ECP根本失败条件均已固定。

owner已接受专家的Action Meta门槛：只有base Writer先产生明确闭环增量，matched Action Meta又有明确净收益且无breadth/retention
损害时才加入，否则保持关闭。rank12 carrier + mobile rank4是首版有证据配置，不是永久锁死；active design保留了rank-ceiling
诊断通过后重开分配的正式分支。

owner最新取消所有人为阶段工期、固定修正次数、结构版本和训练轮数上限。Gate与失败定位仍保留；有新机制证据可继续修正，
无信息超参小扫不算推进。执行应积极复用、并行和提升吞吐，顺利时力争数天内完成整体架构实现并推进关键Gate。

owner最新进一步明确：唯一正式性能目标线是validation8 strict paired correct严格`>145/400`，且必须同时满足
相邻稳定性、breadth、四suite非零、Goal/Long贡献、same-task鲁棒性和视频因果controls，不能用偶然峰值通过。
shuffled/reversed只在最终selected checkpoint选定并冻结后测试时序特异性，不进入训练、loss、checkpoint选择、
G1--G5 Gate或架构修正依据。

## 本轮仓库整理结果

- 退役Writer、functional decoder、ECP v1--v24后继、MDCO/PECS、fixed/two-sided realizers与人工process模块已删除；
- evaluator保留source/task-expert adapter、dynamic queue、occupancy diagnostics和strict aggregation；
- canonical基础模块为source/corpus/SFT、LoRA、task experts、Stage 0、policy effects、functional loss、reward/occupancy与evaluation；
- 旧41份Markdown、87份分散证据JSON、退役配置/测试及约11.6GB可重建人工datasets/runs已清除；recovery Gate A残留
  作为历史formal evidence保留，不删除也不恢复为当前路线；
- 瘦身提交`6fdaeb8`的126项活动CPU测试、compile、脚本入口与引用审计均通过；
- orientation清理节点当时只有`main`一个worktree、无task-owned branch或GPU job；后续G1按合同使用隔离实现面与detached formal
  worktree，动态状态以本节“当前下一步”和live检查为准。

## 当前可复用资产

- 固定24/8/8 split、71-task source corpus、五fold meta manifests与target fold0 manifests；target其余folds在G4多fold验证前补齐，
  不阻塞G1；
- frozen source PI0.5 authority、rank16 LoRA topology/materialization；
- task-expert bank、independent successful members、mobile-rank4解析容量与effect calibration；
- Stage 0 v3 full-layer/horizon observer、transition matcher、event binding/segmenter；
- cross-episode video/action schedule、functional flow loss与detached LoRA gradient bridge；
- natural reward rollout、occupancy capture、BDDL progress与cost-balanced strict evaluator；
- ignored `runs/`中的唯一formal checkpoints、raw rows和aggregate。

## G1真实smoke证据

- 使用纯`load_frozen_native_observer`路径，`action_meta_lora=None`、`install_action_meta_lora=False`；实际对象图中无
  `MetaLoRAStack/MetaLoRAProjection`，policy与Stage 0 trainable列表均为空；
- 38个target均从identity LoRA wrapper的真实`base_layer`捕获X/Y；输入候选不含output type，输出候选含四类bank；
- task 90的一步真实优化中`rank_queries/event_logits/input_logits/output_logits/scale_logits`均有有限非零梯度；
- 峰值allocated约27.24GB A40显存；输出checkpoint为single complete rank16、76 tensors、carrier slots`[0,12]`与task slots`[12,16]`；
- profile输出已核对后删除，不作为formal evidence。
- q-head修正后task93一步真实profile中`output_logits`的16,793,600个元素全部获得非零梯度，其余四类free variables也全部非零；
  peak allocated为28,332,442,624 bytes，single complete rank16与纯Native/Action-Meta-off合同保持不变。
- reference-projected初始化后task93的pre-update latest loss从旧随机路径约`1.32`降至`0.817`，global-member effect为`0.107`；
  全部五类free variables仍有非零梯度，峰值28,676,537,344 bytes，真实chunk cache为521,625,600 bytes。
- action-in 32×32D修正的task94真实profile中，32个output blocks均为stable rank32，input/output minimum direction cosine仍为
  `>=0.99999988`；一步真实loss backward使全部26,208,000个`output_logits`及其余四类free variables获得有限非零梯度，
  peak allocated为29,771,734,528 bytes。纯Native loader、Action Meta module/parameter 0、38 hooks和76-tensor唯一rank16均保持。

## 当前下一步与延期漂移

1. 审查action-in 32×32D native-block修正diff、配置合同与全量CPU回归，集成clean pushed main；
2. 从该commit建立detached frozen worktree，live检查gpu01/gpu02、
   prohibited设备、进程、显存/util与独立storage quota，fresh生成五个task-local step0；
3. 以该single step0唯一rank16 bank重跑同一strict250和four-arm Gate；若仍non-pass，按最早失败task的signed-measure/span response
   定位，不以步数、seed、LR、threshold或group-count小扫替代机制证据；
4. Action Meta在G1必须继续实际关闭；现有旧loader/config漂移不在本轮扩建Action Meta架构；
5. target当前只有fold0 manifests；在G4需要至少两个train24 folds前补齐，不阻塞G1；
6. 32-task fresh refit与71 meta+train24 development recipe的精确顺序延迟到Final前解决，不阻塞G1--G5。

# EMBER progress

更新时间：2026-08-25。本轮G2实现起点：clean pushed `main@5617a4e21ce91c931b658f7c995c4b536b412b49`。

## 当前状态

专家回复已经收到、完整阅读并固化。active design现为`docs/event_conditioned_policy_compiler_design.md`中的
**ECP Native-Factor Compiler**，核心架构、数据角色、阶段Gate、最终controls和停止条件均已明确。

专家1416行原始回复已完整保存为`docs/expert_review_20260824_native_factor.md`，逐行内容与附件一致，仅换行从CRLF标准化为LF；
active design明确是解释/执行层，不能替代原文。

全仓库orientation和两轮owner复核已完成，owner已正式许可推进。G1 task-local free-code capacity oracle已经通过，当前进入G2
Natural Program；没有恢复旧Writer/realizer/GOMQ/PECS/人工process路线，也没有联系专家。

G2实现面现已接通并通过最小真实检查。Program严格输出`P_lang/P_scene/P_process/rho/tau/sigma`固定schema；每条video分别运行
两条fixed antithetic native probes，再做monotonic canonical alignment与`beta_k=1/K`集合聚合。首轮真实held检查发现把多视频frames
先扁平后按全局chunk分批会使K4集合置换最大误差达`0.132`；按阶段合同改为每条video完全独立native forward后，同一检查降为
`2.38e-7`，K1为bitwise exact identity。该修正针对真实失效接口，不引入learned video reliability或task/frame route。

95-task training-only dynamic label authority已升级并在
`runs/outputs/pi05_ecp_natural_program_labels_g2_v2_cpu_20260825`封存完成：meta56/held15、target-fit19/held5共735,519 frames，
BDDL goal predicates为80个单predicate、14个双predicate、1个三predicate task；`obs[i]`按真实`state[i+1]`恢复，缺失terminal
post-action contact显式mask，`rising[0]`明确比较`states[0] -> states[1]`。全量复核发现4750个demo均无首action完成goal，故v2相对v1
rising数值不变，schema升级防止旧语义被静默复用。四个旧LIBERO-90 scene4任务的HDF5 XML使用pre-rename `salad_dressing_1`，当前BDD model使用
`new_salad_dressing_1`；只在内存中对模型identifier做已验证alias后补齐，未修改原始数据。

formal前代码复核发现并修正两项会污染解释的问题：旧rank-local negative queue与`local_index % 4` robustness使辅助loss随rank分配
不等权；现改为每task一次robustness及固定8个、target/meta fit各4个且与rank/world-size无关的content negatives。旧action target
先按video长度取整再映射action episode，而其它动态标签直接按action episode取整；现统一使用唯一action-episode query grid。修正后
architecture guard无hard violation，全仓库149 tests通过。修正后的真实K4 profile在v2 label authority上完成：macro time 23.53s、
peak allocated 18,853,217,280 bytes、84/84 trainable tensors均进入optimizer state、loss/gradient finite；run contract实际记录固定
role-balanced negatives、每task robustness、Action Meta module/parameter 0及source trainable 0。

最新真实K4 forward/backward smoke完成：84/84 Natural Program trainable parameter tensors均产生optimizer state，gradient norm有限，
peak allocated为18,853,213,184 bytes；纯Native loader实际记录Action Meta module/parameter 0、source policy trainable 0、held gradient 0。
下一步是完成diff复核与main集成，从clean pushed detached worktree启动macro10 formal及
meta-held15+target-held5 Gate。

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

action-in native-block修正由clean pushed `main@31f0053`完成，142项CPU回归及task94真实forward/gradient/materialization smoke通过。
从detached frozen worktree生成的五task step0 bank完成同一four-arm strict250：`114/250`、逐task`35/31/45/2/1`，relative
recovery为`71/67=1.060`，breadth5/5、四suite非零、Goal2、Long1、4/5 task高于carrier、carrier retention`35/43`，全部Gate
checks通过。54/54 shards、250/250 rows与18/18 workers完整，Action Meta module/parameter为0，adapter为唯一完整rank12+4 rank16，
没有使用shuffled/reversed。该pass只回答native X/Y banks与signed pooling形式的capacity问题；shared Program query-key attention仍由G3验证。

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
- 当时唯一下一步是fold0 held5 task-local free-code strict250；该Gate现已通过；
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

1. 完成G2 task-scoped diff与CPU回归，把唯一实现面合并到main并推送；
2. 从clean pushed main建立detached frozen worktree，live检查双节点GPU和storage后启动macro10 formal；
3. 在meta-held15+target-held5运行same-task separation、probe stability、event non-collapse、full-vs-endpoints、K1 identity和K4集合
   置换Gate；失败按最早的native capture、event grounding或owner-specific language/scene接口修正；
4. G2不引入learned video reliability；`beta_k=1/K`，learned bounded K correction只在G3根据G2证据决定；
5. target当前只有fold0 manifests；在G4需要至少两个train24 folds前补齐，不阻塞G2/G3；
6. 32-task fresh refit与71 meta+train24 development recipe的精确顺序延迟到Final前解决，不阻塞G2--G5。

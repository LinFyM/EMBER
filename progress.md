# EMBER progress

更新时间：2026-08-25。当前G2 formal authority：clean pushed `main@49e7769c560289623850b729bcf6b645042997d5`。

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

clean pushed `main@141a110`的首轮G2 macro10 formal及meta-held15+target-held5 Gate已经完成。same-task nearer为`1.0`、probe
margin为`0.9`、one-event为`0`、median active events为`6`、K1 exact identity与K4 permutation均通过；唯一non-pass是full相对
endpoints的action/progress改善仅`0.0226%`，低于`10%` Gate，因此没有进入G3。

read-only held20机制诊断显示，full相对endpoints的`P_process/rho/tau`差异分别是same-task不同视频差异的约`2.20x/13.77x/60.00x`，
说明最早失效接口不是native动态捕获。相反，decoder action时序标准差仅`0.00060`而target为`0.33789`，清零`P_process`的静态路径
combined loss反而由`0.39574`降到`0.39088`。当前唯一修正因此移除`P_lang/P_scene`对`P_process` fusion及action/progress等时序heads的
直接加性旁路；静态scene-only head保持独立。修正后的task92 K4真实forward/backward已完成，84/84 trainable tensors均进入optimizer
state，gradient有限、peak allocated 18,851,367,936 bytes，Action Meta module/parameter 0且source trainable 0；下一步是集成后fresh
训练并复评同一held20 Gate。

该静态旁路修正已由clean pushed `main@30b98ef`完成，fresh macro10 formal与同一held20 Gate也已完成，但仍为non-pass：
same-task nearer `1.0`、K1 identity、K4 permutation与median active events `3`通过；full相对endpoints改善为`-0.0570%`，
one-event fraction `0.30`，probe margin `0.65`。无梯度temporal诊断显示event weights虽随时间变化（std `0.04398`），最终pooled
state std仅`0.00111`，action预测std仅`0.00093`而target为`0.32725`；hard-nearest、uniform event measure和mean-repeated
process消融几乎不改变loss，最早接口已推进到event/owner内容而不是query measure。

进一步在target-held5对比初始frozen Stage 0与macro10 encoder状态：Stage 0的event/owner relative RMS为`0.06069/0.36992`，
训练后的raw encoder降为`0.02601/0.22824`，fusion后owner进一步降至`0.14837`。这证明G2梯度在Program读出成立前先侵蚀了已有
Stage 0 event grounding。当前唯一机制修正因此冻结已验证的Stage 0 v3 observer，仅训练新的language/scene/process readers、alignment
与training-only heads；不同时修改owner readout、slot、width、seed、LR或Gate。run contract将实际强制Action Meta 0、source trainable 0
及native observer trainable 0。

该修正的task92真实K4 forward/backward smoke已经完成：46/46个新增Program parameter tensors进入optimizer state，loss与gradient
有限，peak allocated 10,016,652,800 bytes；实际run contract记录native observer trainable 0且training mode为eval、source trainable 0、
Action Meta module/parameter 0。checkpoint中39个encoder tensors与Stage 0 v3 authority逐tensor相等，证明optimizer/forward没有改写
frozen observer。profile输出核对后将删除，不冒充formal evidence。

clean pushed `main@db84a50`的frozen-observer formal已从fresh训练到macro10，并按原world5 topology exact-resume到macro20。macro10
held20 Gate中same-task nearer `1.0`、K1/K4、median active events `5`与one-event `0`通过，但full相对endpoints仅`+0.0051%`、
probe margin `0/40`；macro20仍为non-pass，full相对endpoints为`-0.0207%`、probe margin `0/40`、one-event `0.025`，其它上述
资格项保持通过。fit total从macro10的`1.17260`降至macro20的`0.97637`，因此内部loss下降没有转化为视频动态因果增量。

macro20无梯度诊断确认冻结修正确实保存了native结构：raw full event/owner relative RMS为`0.06252/0.36771`，fused为
`0.05590/0.26447`，full与endpoints的fused Program RMS差异仍有`0.00618`。最早失效接口现为temporal owner readout：当前共享
`Linear(128,1)`对38-owner轴严格置换不变，owner entropy为`0.99898`，action prediction temporal std仅`0.00173`，而target为
`0.32725`；从macro10继续到20没有改善该比例。两个raw antithetic branches仍不稳定，但把零均值residual只在辅助Gate路径缩小、
而不改变部署`P_process/rho/tau/sigma`，会成为Gate-only旋钮，已明确不采用。

当前隔离实现只把training-only temporal readout改为38个固定LoRA owner各自的linear query；38条query从旧共享Linear完全相同的
向量初始化，保持其余head的旧RNG序列，之后只能由owner-specific梯度分化。queries跨task共享且只读取
`P_process` content，不是task-ID route。scene head、probe、Stage 0、Program schema、数据、loss、seed/LR/slot/width与Gate均不变。
task92真实K4 profile已通过：owner-query gradient norm `0.01827`，一步后query row centered RMS为`3.23e-5`，证明共享初始化已由
owner-specific梯度分化；46个Program parameter tensors/915,554 parameters trainable，native observer/source policy/Action Meta
trainable均为0，39个observer tensors逐tensor不变，peak allocated 10,016,671,744 bytes；profile只作机制smoke，核对后删除。

clean pushed `main@407340b`的owner-specific scalar-query formal已从fresh macro10按同一world2 topology exact-resume到macro20。
macro10/macro20 held20 Gate的full相对endpoints改善分别为`+0.0158%/-0.0340%`，probe margin均为`0/40`；same-task、K1/K4、
active-event范围继续通过，所以仍未进入G3。owner queries的row-centered RMS从自身RMS的`1.58%`增长到`2.94%`，但macro20
actual与强制shared-query的held combined loss只差约`4.9e-5`，action prediction temporal std仍只有`0.00171`，target为
`0.33589`；hard-owner readout同样不改善。该结果淘汰owner-specific scalar selection作为充分修正，也排除继续到macro40只等待
query分化的解释。

进一步无梯度反事实把frozen Stage 0 raw process与其已训练action head重新配对：held action absolute loss从当前fused/current的
`0.25511`降至`0.20767`，说明Stage 0坐标与head包含可复用信息；但full相对endpoints仍只改善`0.2467%`，prediction temporal std
仅`0.00298`，所以只复用旧head不足以满足G2动态门；该反事实也没有提供直接增加owner value map的充分证据。当前最早接口是
absolute cross-episode MSE被trajectory mean解主导，未单独约束query-time action/progress residual。当前机制修正保留absolute losses并新增等权query-centered action/progress
residual MSE，不改模型、Program schema、数据、K、seed/LR或Gate。task92真实K4 profile得到`action_temporal=0.14324`、
`progress_temporal=0.08779`、owner-query gradient norm `0.01839`，39个observer tensors不变，Action Meta/observer/source trainable均为0，
peak allocated 10,016,671,744 bytes；profile已删除。

clean pushed `main@68f8705`的temporal-residual objective已从fresh训练到macro10。held20 Gate中same-task nearer `1.0`、K1/K4、
median active events `5`与one-event `0`继续通过，但full相对endpoints只改善`0.0381%`，probe margin为`0/40`，因此明确non-pass，
没有进入G3，也没有用fit loss下降或继续同一低更新数run冒充进展。

该轮结果冻结后的read-only根因分析排除了新的表示架构猜测：固定Program的full-owner temporal readout相对endpoints可改善
`15.17%`；tied-query与independent-query初始化曲线近乎相同；cross-episode监督可识别。旧trainer每macro访问38个task却只执行
一次Adam更新，所以macro10仅10次更新。同一frozen readout temporal loss从`0.311873`开始，10/60步仅为
`0.311827/0.311164`，到200/500步才降至`0.294034/0.257824`。最早接口由此定位为optimizer cadence，而不是再次增加Program
slot、width或readout结构。

当前隔离实现保持模型、数据、loss、K、seed/LR峰值和Gate不变，把每macro拆为10个role-balanced optimizer steps：常规
2 target-fit+2 meta-fit，尾部1+1并随macro轮换；scheduler与exact-resume cursor按真实optimizer step计数。单卡及gpu02 world4
真实profile均完成：world4实际聚合4个互异task、role为2+2、finite owner-query/全局gradient，46/46 Program tensors进入Adam，
四个rank checkpoint齐全；run contract记录source/observer trainable 0、Action Meta module/parameter 0。profile只作执行证据，
核对后删除，不冒充formal。

该cadence修正已由clean pushed `main@49e7769`完成，并从fresh训练至macro10/100 optimizer steps。held20 Gate仍为non-pass：
full相对endpoints改善`0.3080%`、probe margin `13/40=0.325`，低于`10%/0.75`门；same-task、K1/K4、event范围与tau资格项均通过。
相对旧10-update checkpoint，动态增量由`0.0381%`提高约`8.1x`，20个held task中17个方向改善，meta/target-held分别为
`0.2781%/0.3891%`，所以这是宽泛但幅度不足的真实信号，不是偶然峰值，也不能进入G3。

冻结macro10后仅用12个fit task（target/meta各半，K=1/2/4等量）做gradient diagnostic；held gradient为0。full与endpoints的
`P_process`差异仍有`0.07296 RMS`，但full action/progress prediction temporal std仅`0.00379/0.00160`，而target为
`0.35248/0.32500`。temporal梯度相对non-temporal在Program process参数上为`0.01031/0.10345`，在temporal decoder上为
`0.00885/0.18567`；cosine仅`-0.065/-0.071`，说明问题不是方向性强抵消，而是近常数读出使temporal梯度小约`10--21x`。
结合frozen readout在100--500步才开始明显展开的既有曲线，下一步按同一commit/world4 topology exact-resume到预注册macro20；
这是对“有效但尚未跨过学习时标”的可证伪检验。若held增量和prediction temporal std没有实质继续增长，该解释即被否定，下一修正
必须直接针对Program-to-temporal-readout的梯度饥饿/近常数结构，而不能靠继续训练或超参小扫。

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

1. 在同一clean detached `49e7769`、world4 topology与run目录上exact-resume到预注册macro20/200 optimizer steps，复评完全相同的
   held20 Gate；不改变模型、loss、数据、K、LR、seed或checkpoint选择合同；
2. 若macro20使held full增量、prediction temporal std与probe margin按既有学习曲线实质增长，则继续依预注册节点判断；若仍近常数，
   即否定单纯学习时标解释，先冻结证据并针对Program-to-temporal-readout结构做机制修正后fresh复评；
3. 任何non-pass都不得用连续架构版本、probe-only旋钮或LR/seed/width小扫替代根因分析；结构修正允许且应在证据指向结构接口时实施；
4. G2不引入learned video reliability；`beta_k=1/K`，learned bounded K correction只在G3根据G2证据决定；
5. target当前只有fold0 manifests；在G4需要至少两个train24 folds前补齐，不阻塞G2/G3；
6. 32-task fresh refit与71 meta+train24 development recipe的精确顺序延迟到Final前解决，不阻塞G2--G5。

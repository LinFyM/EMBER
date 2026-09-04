# EMBER research history

本文件只保存影响当前决策的历史事实。精确旧实现、配置和逐条证据通过Git提交与保留的formal run artifacts恢复；退役代码、
单轮卡片和重复JSON不再保留在活动树。

## 1. 固定基线与能力上界

| 证据 | strict结果 | 结论 |
|---|---:|---|
| frozen source PI0.5 | 48/400 | generic source在validation8上远不足以完成目标 |
| validation8 task-local rank16 experts | 250/400 | 只改Action Expert的完整rank16 LoRA有真实闭环容量；四suite为73/78/58/41 |
| train24 fold0 held5 source | 21/250 | leave-task-out机制门的低基线 |
| held5 stable shared carrier | 43/250 | 跨任务共享prior能提供有限支持，但不是完整EMBER |
| held5 direct successful members | earliest 74/250；latest 108/250；independent 113/250 | 五个task含Goal/Long都存在task-local成功策略与可捕获policy effects |

因此核心瓶颈一直不是“Action Expert LoRA能否完成任务”，而是language+video能否通过共享Writer稳定地产生正确LoRA。

## 2. 早期Writer家族

2026-07至2026-08先后尝试了action-memory、belief、LOOM、CVADR、LMMPC/LPCP、layer-matched memory、reward credit、
gradient/open-memory query等多种Writer。共同问题是：训练loss或内部结构可以变化，但closed-loop提升低、churn高、Video
controls弱，且Goal/Long长期为0。

后期最有代表性的自然视频裁决为correct/language-only/video-only/first+final约`41/39/40/39`，Goal和Long均为0；单向outer
credit还从41降到39。完整视频没有形成超越language、video-only或端点信息的必要增量。这淘汰了当时具体Writer，不等于淘汰
视频条件参数生成。

这些实现由2026-08-18之前的Git历史恢复。活动树不保留其并行模块。

## 3. GOMQ的边界

GOMQ是独立历史机制，不是ECP阶段、Program schema或后续实现依赖。它曾显示较强但不充分的闭环适配：

- 历史rank32 correct为151/400；same-task-other 139、wrong 131、shuffled 127、reversed 115；
- 按统一rank16合同重建后的strict correct为136/400。

rank32结果说明更强adapter可以达到目标附近，controls说明存在一定视频特异性；rank16下降说明其能力部分依赖容量。GOMQ不能
证明ECP，也不应像每个历史架构一样被重复重跑。关键提交：`f2f9290`、`ac233fa`、`3075b3c`。

## 4. 专家第二轮后的functional路线

task experts与Action Expert policy responses建立了更明确的动态几何：完整成功轨迹在held5上可形成5/5 task separation；fixed
phase decoder曾把held5从source 21提高到44/54，但只保留约26%--31%的direct expert successes。stable shared prior达到43，
而不可靠的task residual降到37/33，说明“错误的条件更新”可能比不更新更坏。

这一阶段还建立了train24/non-held LIBERO-90 expert banks、跨episode video/action采样、occupancy capture、functional flow loss和
严格评测队列。这些基础设施仍在活动树；旧fingerprint decoder、phase decoder、outer-credit Writer和shared residual实现已退出。

## 5. ECP Stage 0

ECP将视频表示改为PI0.5-native owner/layer/horizon结构，并用有序event slots形成Program候选。native v3通过了non-degeneracy、
task separation、event occupancy与基本训练图检查；Action Meta-LoRA matched arm效果中性。它只证明候选observer不是常量，未证明
完整video-to-Program-to-LoRA链。

关键提交：

- `e675b87`：Stage 0 v3 formal结果；
- `bb06676`：冻结Stage 0 observer authority；
- 当前保留实现位于`src/ember/ecp/stage0*.py`、`observer.py`、`events.py`。

## 6. ECP Stage 1 v1--v24与MDCO

Stage 1连续尝试了deterministic/mean privileged code、fixed/free Program、learned A/B hyperdecoder、owner-local activation、policy
support、outcome binding、program-locked compiler、direct absolute surface和mapping-diverse compiler。v24之后继续扩大width、rank、
heads、fusion、LR、seed或mapping数量没有改变最早失效接口：Program或privileged evidence无法通过共享compiler在held task上产生
广泛闭环增量。

这一阶段的问题不是“版本不够多”，而是多次在没有重新核对ECP因果链时继续修改同一类decoder。retrospective在`1852d04`，
专家对全过程复核的仓库快照在`bfcc917`附近。所有v1--v24和MDCO执行代码现已从活动树删除。

## 7. Privileged policy evidence与realization门

为区分“evidence不足”与“realizer失败”，后续建立了独立successful lineages和多状态policy-effect banks：

- 独立successful members在held5达到113/250；五个task均有成功轨迹；
- source/shared/direct基线为21/43/74--113；
- structured privileged fixed-A solver达到78/250，相对carrier +35，但breadth仅3/5，Goal/Long为0，只恢复约35/115的best-member
  gap；
- 三个known-success member投影到fixed-A坐标仅49/41/35，Goal/Long全0，证明fixed-A在该实现中是行为瓶颈；
- mobile-rank4 residual具有更高表达容量，但12-step raw-factor solver只有49/250，未进入known-success basin；
- fixed effect realizer的两个裁决点只有33/37；
- centered two-sided coordinate达到80/250，但仍breadth3/5、Goal/Long0。

当时结论是：policy effect证据本身包含有效策略，但历史coordinate/solver/realizer不能稳定跨任务实现它。这个负结果淘汰
fixed-A、raw-factor短solver和当时的fixed realizer；它尚未检验distributional `q_pi`。2026-08-24专家在综合后续fit-span结果后
进一步判断，不应再为未经标定的latent增加神经`q_pi`，并以native-factor critic路线取代。

关键提交：`e8dba3c`、`ceae794`、`083ed98`、`44bd6f0`、`f75bafc`、`8acc5b4`、`1774a9e`、`01a96b6`、`8aab214`、
`7aa0bed`。

## 8. 人工process路线及终止

为构造same-scene/opposite-order最小对，曾手工建立复合任务、primitive/recovery experts和distillation数据。这条路线最终没有
提供可靠teacher：recovery Gate A为14/100（9+5），相对A3 gained0/lost23；所有worker和pairing检查正常，因此是科学non-pass，
不是运行故障。

owner随后明确：没有时间继续制作人工数据集，后续只用现成LIBERO tasks直指ECP核心。人工Gate B、process suite、controller
acquisition和SFT式recovery family全部取消，不再作为Stage 1前置。对应代码、配置、文档以及约12GB可重建人工数据/运行产物已
在2026-08-24仓库瘦身中删除。

关键提交链：`24c5bdc`、`4bf5039`、`b8fb0bf`、`38dbffd`、`7527568`、`d8eca79`、`a06a3ba`、`342620a`、`ebdd509`、
`7ab5a04`。

## 9. 2026-08-24 ECP Native-Factor专家裁决

专家在远程`main@7ab5a04`上复核全部人工process non-pass、Stage 0、successful members、mobile projection和两类shared realizer结果，
形成以下新主线：

专家回复原文完整保存在`docs/expert_review_20260824_native_factor.md`；以下仅为历史索引，若措辞存在疑义，以原文和owner后续明确
裁决为准。

1. 保留ECP输入输出目标，但取消canonical neural `q_pi -> fixed effect code -> inverse realizer`；
2. privileged successful policies/effects改为nonparametric set-valued functional critic，不输出Program；
3. Video Program固定为owner-specific language/scene与最多8个ordered events，并新增`tau[8,2]`；
4. 同一视频第二pass直接捕获38个LoRA targets的native inputs/outputs及动态differences；
5. Program-conditioned signed pooling首版生成mobile rank4，与frozen rank12 carrier拼成唯一rank16；
6. 12+4来自110/120/76且5/5非零的解析容量证据，不恢复fixed-A或旧rank4 solver；专家同时保留了rank-ceiling证据成立后重开
   carrier/task rank分配的分支，并未把12+4说死；
7. 当前唯一下一步是fold0 held5 task-local free-code oracle，门槛约90/250、breadth5/5、Goal/Long非零与高retention；
8. 通过后才训练Natural Program、frozen-Program shared compiler、joint Writer和conditional outer credit；
9. final使用全部71 meta+train24 fresh训练，validation只看三个预注册相邻checkpoint，冻结后补controls并最后打开Test；
10. 只有所有必要Gate与完整controls均完成仍系统失败，才可判断当前自然LIBERO/zero-interaction合同存在根本问题。

完整shape、loss、门槛与阶段逻辑已经结构化进入`docs/event_conditioned_policy_compiler_design.md`。专家原文中的工期和固定尝试
次数随后被owner明确取消为当前硬约束；专家原始审查快照早于
`6fdaeb8`仓库瘦身；后者没有新增科学结果，因此不影响裁决。当前保留代码也验证了其指出的q/v native target hook缺口。

## 10. G1 Native-Factor首轮与scalar-output失效证据

G1首轮canonical实现提交为`9a6f434`。fold0 held5 task-local free-code formal优化后，唯一rank12+4 rank16 adapter的paired
strict250为`88/250`，逐task`33/18/37/0/0`；relative recovery为`0.6716`，breadth3/5、高于carrier 2/5、carrier retention
30/43，因此Gate non-pass。250 rows、47 shards、15 workers、Action Meta关闭与single-adapter合同均正常。

提交`822147b`加入的只读分析证明，scalar signed output pooling把q factor限制在base q weight的1024维列空间（输出2048）。
action-in带bias且不同output-bank types可跨类型相减，后续精确修正其结构上限为`span(column_space(W),bias)`、至多33维，而非该
分析最初简写的32维纯列空间；这不影响无bias q target的主结论。15个known-success mobile-rank4 reference只保留约55--56%总update energy。将
independent member正交投影到该输出上限后，paired strict250为`109/250`、逐task`34/30/45/0/0`；未投影authority为`120/250`，
Goal/Long为`11/8`。因此被排除输出方向对Goal/Long具有闭环必要性，最早问题是scalar q-output bank，不是实现故障、seed或训练时长。

当前机制修正在每个真实q attention head内独立做signed pooling，再拼回原生2048维；候选索引、真实Y、rank4、carrier12和唯一rank16
合同不变。

关键artifacts：

- `runs/outputs/pi05_ecp_native_factor_g1_held5_formal_9a6f434_gpu01p34567_20260824/`；
- `runs/outputs/pi05_ecp_native_factor_g1_free_code_step500_strict250_9a6f434_gpu01p34567_r3_20260824/`；
- `runs/analysis/pi05_ecp_native_factor_g1_output_span_response_822147b_gpu01p3_20260824/`；
- `runs/outputs/pi05_ecp_native_factor_g1_output_span_independent_strict250_822147b_gpu01p34567_r3_20260824/`。

## 11. q-head formal与稳定bank投影

q-head修正由clean pushed `main@a8ec468`执行。五个task均完成500 steps、全部五类free variables有非零梯度、Action Meta module/
parameter为0，checkpoint仍为76 tensors的唯一rank12+4 rank16。paired strict250为`84/250`，逐task`28/21/35/0/0`；relative
recovery `0.6119`、breadth3/5、高于carrier 2/5、carrier retention `24/43`，因此Gate non-pass。

endpoint审计显示，step500 generated residual与latest/independent/earliest references的整体effective-update cosine约`0.05--0.07`；
Goal task的latest/independent update loss仍为`1.18/1.17`，虽然global-member effect从约`0.72`降到`0.24`。这表明effect critic的内部
改善没有把千万级随机近均匀free logits带入known-success update basin，不能用内部loss冒充closed-loop。

随后对五个真实K=1视频bank按relative singular threshold `1e-3`构造稳定中心子空间，将latest mobile rank4的input factors与q-head-grouped
output factors作正交投影并按冻结`s_ref`截断。该read-only唯一rank16 response诊断在strict250达到`94/250`，逐task
`24/24/44/1/1`：relative recovery超过0.70、breadth5/5、Goal/Long非零且四task高于carrier；carrier retention仅`22/43`，所以不是
G1 Gate pass。该结果直接约束下一修正为free-logit可达优化与retention，而不是扩rank、加slot或宣告native bank根本失败。

基于此证据，当前实现把稳定子空间的最小范数系数解析分解为positive/negative simplex，初始化实际task-local signed logits；q每个真实
head用共同背景质量保持跨head相对幅度。task93真实一步profile的pre-update latest loss为`0.817`（投影下界`0.813`），effect loss
`0.107`，五类参数均有非零有限梯度，缓存真实chunk后峰值约28.68GB。该初始化只属于G1 privileged capacity oracle，不进入G3共享
compiler。

clean pushed `fc53249`进一步保留未受optimizer扰动的step0。其residual相对解析projection cosine为`0.952--0.964`，而第一次Adam
更新后为`0.039--0.070`；配套五task 500-step run的最终effective-update loss均差于step0。step0 paired strict250为`100/250`、逐task
`24/28/45/3/0`，relative recovery`0.8507`，但breadth4/5、Long 0、tasks-above-carrier 3/5、retention`22/43`，故Gate non-pass。

该结果与固定50 evidence共同定位到set-valued选择：task90 carrier 38强于latest/independent/earliest的`27/26/17`；其余四task最强
member依次为independent/latest/independent/independent，成功数`32/40/13/5`。下一修正据此选择verified member，carrier胜出时用
zero rank4 residual；candidate bank、signed pooling、rank12+4和唯一rank16合同不变。

clean pushed `873af85`完成该set-valued solve，paired strict250为`111/250`、逐task`35/29/45/2/0`，relative recovery`1.0149`、
retention`34/43`；breadth4/5、Long 0、tasks-above-carrier 3/5，故仍non-pass。task94的实际初始化报告同时显示FP32
inverse-scatter在`1e-3` threshold下把input/output direction cosine最低降至`0.978/0.883`。仅将小型初始化sufficient statistics改为
FP64后，真实task94 profile两侧minimum cosine均恢复到`>=0.99999988`；该profile只证明机制接通，closed-loop结论待clean formal。

关键artifacts：

- `runs/outputs/pi05_ecp_native_factor_g1_qhead_held5_formal_a8ec468_gpu01p34567_20260825/`；
- `runs/outputs/pi05_ecp_native_factor_g1_qhead_free_code_step500_strict250_a8ec468_gpu01p34567_r3_20260825/`；
- `runs/analysis/pi05_ecp_native_factor_g1_bank_span_latest_r1e3_a8ec468_gpu01p34567_20260825/`；
- `runs/outputs/pi05_ecp_native_factor_g1_bank_span_latest_r1e3_strict250_a8ec468_gpu01p34567_r3_20260825/`。
- `runs/outputs/pi05_ecp_native_factor_g1_exact_init_held5_formal_fc53249_gpu02p1_20260825/`；
- `runs/outputs/pi05_ecp_native_factor_g1_exact_init_step0_strict250_fc53249_gpu01p34567_r3_20260825/`。
- `runs/outputs/pi05_ecp_native_factor_g1_set_oracle_held5_formal_873af85_gpu02p1_20260825/`；
- `runs/outputs/pi05_ecp_native_factor_g1_set_oracle_step0_strict250_873af85_gpu01p34567_r3_20260825/`。

## 12. action-in native-block修正与G1正式通过

clean pushed `main@31f0053`把action-in真实1024D Y按native input width分为32个连续32D blocks，各block独立signed pooling；候选索引、
四类output banks、真实Y、rank4 residual和唯一rank12+4 rank16合同均不变。task94真实profile确认32个blocks均为stable rank32，
两侧minimum direction cosine `>=0.99999988`，全部26,208,000个output logits有有限非零梯度，Action Meta module/parameter为0。

五task formal step0 bank的paired strict250达到`114/250`，逐task`35/31/45/2/1`；relative recovery`71/67=1.060`、breadth5/5、
Goal2、Long1、4/5 task高于carrier、carrier retention`35/43`，全部G1 checks通过。54/54 shards、250/250 rows与18/18 workers完整，
没有使用shuffled/reversed。该结果只证明真实native X/Y banks加signed pooling存在强闭环rank4 residual；共享Program到content attention
仍未验证，按顺序进入G2 Natural Program。

关键artifacts：

- `runs/outputs/pi05_ecp_native_factor_g1_action_in_groups_held5_formal_31f0053_gpu02p1_20260825/`；
- `runs/outputs/pi05_ecp_native_factor_g1_action_in_groups_step0_strict250_31f0053_gpu01p234567_r3_20260825/`。

## 13. G2 Natural Program首轮formal与静态旁路定位

clean pushed `main@141a110`的首轮G2 macro10 formal在meta-held15+target-held5上通过same-task separation、probe stability、event
non-collapse、K1 exact identity与K4 permutation invariance，但full相对first+final的action/progress loss改善仅`0.0226%`，低于
`10%`资格门，因此G2 non-pass且未进入G3。

无梯度机制诊断显示full与endpoints的`P_process/rho/tau`已经明显分开，但action/progress预测近乎时间常数，且清零`P_process`后
loss反而更低。历史结论因此只淘汰当时把`P_lang/P_scene`直接加到每个event、同时注入process fusion的training decoder实现；它不
淘汰Stage 0 native capture、Natural Program schema或ECP Native-Factor路线。后继修正保留scene-only head，切断动态heads的静态旁路，
并要求fresh训练后复评同一Gate。

关键artifacts：

- `runs/outputs/pi05_ecp_natural_program_g2_fold0_m10_141a110_gpu01p123457_r6_20260825/`；
- `runs/analysis/pi05_ecp_natural_program_g2_macro10_endpoint_shortcut_diagnostic_141a110_gpu01p1_20260825/`。

## 14. G2静态旁路移除后的formal与Stage 0结构侵蚀定位

clean pushed `main@30b98ef`移除静态旁路后，从fresh初始化完成macro10 formal。held20 Gate仍为non-pass：same-task nearer、K1、K4
与active-event median通过；full相对endpoints改善`-0.0570%`，one-event fraction `0.30`，probe margin `0.65`。

无梯度readout诊断表明event-time weights确实变化，但owner/event内容已经近乎均匀，最终prediction temporal std远低于target；改变
event query measure不能恢复动态。随后target-held5对照显示初始Stage 0 v3的event/owner relative RMS为`0.06069/0.36992`，G2训练后
raw encoder降为`0.02601/0.22824`。该轮只淘汰“在G2一开始联合训练Stage 0 observer与Program heads”的实现；后继修正冻结已有
Stage 0 v3，仅训练新增Program层，待fresh复评后再决定是否需要owner-structured readout修正。

关键artifacts：

- `runs/outputs/pi05_ecp_natural_program_g2_static_free_fold0_m10_30b98ef_gpu01p123457_r6_20260825/`；
- `runs/analysis/pi05_ecp_natural_program_g2_static_free_macro10_temporal_readout_diagnostic_30b98ef_gpu01p1_20260825/`；
- `runs/analysis/pi05_ecp_natural_program_g2_static_free_macro10_event_grounding_diagnostic_30b98ef_gpu01p1_20260825/`。

## 15. G2 frozen-observer formal与owner readout定位

clean pushed `main@db84a50`冻结Stage 0 v3后，fresh macro10并按原world5 topology exact-resume到macro20。held20 Gate中same-task、
K1/K4、active-event范围保持通过，但full相对endpoints改善仅`+0.0051%/-0.0207%`，macro20 probe margin为`0/40`，因此G2仍
non-pass。fit total从macro10的`1.17260`降至macro20的`0.97637`，不能用内部loss下降替代held视频增量。

无梯度层级诊断确认raw Stage 0 full event/owner relative RMS保持`0.06252/0.36771`，full/endpoints fused Program RMS差异为
`0.00618`；失败不再来自observer侵蚀或动态缺失。最早接口是training-only decoder对38个固定LoRA owners共享同一
`Linear(128,1)`，其owner加权和对owner轴严格置换不变。owner entropy `0.99898`、action prediction temporal std `0.00173`
而target为`0.32725`，继续训练没有修复。后继只允许把该readout改为跨task共享的固定owner-specific content queries；raw probe
margin仍须独立处理，不能用不改变canonical Program或action/progress utility的Gate-only residual缩放。

关键artifacts：

- `runs/outputs/pi05_ecp_natural_program_g2_frozen_observer_fold0_m10_db84a50_gpu01p34567_r5_20260825/`；
- `runs/analysis/pi05_ecp_natural_program_g2_frozen_observer_macro10_layer_diagnostic_db84a50_gpu01p3_20260825/`；
- `runs/analysis/pi05_ecp_natural_program_g2_frozen_observer_macro20_layer_diagnostic_db84a50_gpu01p3_20260825/`。

## 16. G2 owner-specific scalar readout与时间均值坍缩定位

clean pushed `main@407340b`把training-only owner score改为38个fixed-owner、shared-across-task content queries，并从fresh macro10
按同一world2 topology exact-resume到macro20。held20 Gate的full相对endpoints改善分别为`+0.0158%/-0.0340%`，probe margin均为
`0/40`，其它same-task、K1/K4与event资格项保持通过，因此G2仍non-pass。

query rows从自身RMS的`1.58%`分化到`2.94%`，但macro20 actual与强制shared-query held loss只差约`4.9e-5`，hard-owner同样无效；
action prediction temporal std仍为`0.00171`而target为`0.33589`。raw Stage0 process与其既有action head反事实将absolute action loss
从`0.25511`降至`0.20767`，却只产生`0.2467%` full增量和`0.00298` temporal std。该证据淘汰owner-specific scalar selection、
softmax温度和单纯旧head转移作为充分解释；下一修正只把同一授权action/progress监督分解出query-centered temporal residual MSE，
不改变模型容量、数据、K、seed/LR、Gate或Program schema。

关键artifacts：

- `runs/outputs/pi05_ecp_natural_program_g2_owner_readout_fold0_m10_407340b_gpu02p03_r2_20260825/`；
- `runs/analysis/pi05_ecp_natural_program_g2_owner_readout_macro10_diagnostic_407340b_gpu02p3_20260825/`；
- `runs/analysis/pi05_ecp_natural_program_g2_owner_readout_macro20_diagnostic_407340b_gpu02p3_20260825/`；
- `runs/analysis/pi05_ecp_natural_program_g2_stage0_transfer_macro20_diagnostic_407340b_gpu02p3_20260825/`。

## 17. G2 temporal-residual formal与optimizer cadence定位

clean pushed `main@68f8705`保留absolute action/progress监督，并加入等权query-centered temporal residual MSE。fresh macro10的
held20 Gate仍为non-pass：same-task nearer `1.0`、K1/K4、median active events `5`与one-event `0`通过；full相对endpoints改善
`0.0381%`，probe margin `0/40`。

冻结该checkpoint的Program做可证伪readout诊断后，full-owner temporal readout相对endpoints可改善`15.17%`，tied/independent
owner-query初始化曲线近乎相同，cross-episode监督可识别。旧trainer每macro聚合38个task后只做一次Adam更新，故macro10实际上仅
10次更新；同一frozen readout temporal loss在10/60步几乎不变，到200/500步才明显下降。该证据把最早接口定位为optimizer
cadence，而不是Program动态bank不存在或需要再次增加架构容量。后继实现只把每macro改为10个role-balanced microsteps，并按真实
optimizer step推进scheduler/resume；模型、数据、loss、K与Gate保持不变。

关键artifact：

- `runs/outputs/pi05_ecp_natural_program_g2_temporal_residual_fold0_m10_68f8705_gpu02p0123_r4_20260825/`。

## 18. G2 cadence formal与temporal gradient starvation定位

clean pushed `main@49e7769`把每macro的一次update改为10个role-balanced optimizer steps，并从fresh完成macro10/100 updates。
held20 Gate仍为non-pass：full相对endpoints改善`0.3080%`、probe `13/40`；same-task、K1/K4、event范围与tau资格项通过。相对旧
10-update checkpoint，动态增量约提高`8.1x`且17/20 held task方向改善，证明cadence修正有效但远不足以满足`10%` Gate。

冻结checkpoint后只用12个fit task做gradient diagnostic，held gradient为0。full/endpoints `P_process` delta RMS为`0.07296`；
action/progress prediction temporal std为`0.00379/0.00160`，target为`0.35248/0.32500`。Program process和temporal decoder上的
temporal/non-temporal gradient norm分别为`0.01031/0.10345`与`0.00885/0.18567`，cosine为`-0.065/-0.071`。因此最早接口不是
动态bank缺失或强梯度反向冲突，而是近常数readout造成的temporal gradient starvation。下一步保留同一科学合同exact-resume到
macro20，作为既有frozen-readout学习时标的可证伪节点；若动态幅度不实质增长，再依据该证据修改readout结构并fresh复评。

关键artifacts：

- `runs/outputs/pi05_ecp_natural_program_g2_cadence_fold0_m10_49e7769_gpu02p0123_r4_20260825/`；
- `runs/analysis/pi05_ecp_natural_program_g2_cadence_macro10_gradient_diagnostic_49e7769_gpu02p5_20260825/`。

## 19. G2 cadence macro20与canonical alignment坍缩定位

clean detached `49e7769`从macro10 exact-resume到macro20/200 optimizer updates。held20 Gate中same-task、probe、K1 identity、
K4 permutation与tau均通过；full相对endpoints改善达到`8.6878%`，但median active events为`1`、one-event fraction为`1.0`，
且动态增量仍未严格超过`10%`，所以G2明确non-pass。

fit-only no-gradient temporal panel确认readout已按预期展开：full action/progress temporal std为`0.03393/0.04789`，相对macro10
分别增长约`9x/30x`，full相对endpoints改善`15.82%`。进一步按K分解显示K=1仍保留平均`6.42`个active events，而全部K=2/K=4
条件都坍成1个；local native presence未坍缩，learned alignment却把多数path mass集中到单一canonical slot。boundary-only
counterfactual在不改checkpoint参数和decoder的情况下恢复K>1为3个active events，并保持`16.47%` fit视频增量；因此后继只给
monotonic DP增加首尾canonical边界锚点，保留中间stay/skip与content/time score，并从fresh复评。

关键artifacts：

- `runs/outputs/pi05_ecp_natural_program_g2_cadence_fold0_m10_49e7769_gpu02p0123_r4_20260825/`；
- `runs/analysis/pi05_ecp_natural_program_g2_cadence_macro20_alignment_counterfactual_49e7769_gpu02p5_20260825/`。

## 20. G2 boundary-anchored Natural Program正式通过

clean pushed `main@c1493a1`只增加monotonic DP首尾canonical边界锚点，从fresh运行到macro10/100 updates，再按相同world4
topology exact-resume到macro20/200 updates。macro10 held动态增量为`0.8268%`，但event结构已恢复为median 2、one-event 0；
macro20 held20 Gate全部通过：full/endpoints action+progress loss为`0.28167/0.36207`，相对改善`22.2047%`，probe `38/40`，
median active events 4、one-event 0、same-task/K1/K4均为1.0，K4 max abs `4.77e-7`，tau violation `0.00357`。

该结果证明旧macro20 non-pass的最早接口是未约束的K>1 alignment path，而非Natural Program动态容量或readout时标。冻结
`macro_00000020`作为G3唯一Program authority；G3继续验证共享Program-query到native content-key signed attention，不能使用G1
task-local free logits。

关键artifact：

- `runs/outputs/pi05_ecp_natural_program_g2_boundary_fold0_m10_c1493a1_gpu02p0123_r4_20260825/`。

## 21. G3 frozen-Program shared compiler macro5 formal non-pass

clean detached `5140362`以冻结的G2 `c1493a1/macro20` Program、75个fit tasks/93个verified members训练shared native-content
compiler到macro5/95 optimizer updates。compiler只训练共享Program-query/content-key attention、event weights、target scales与bounded
video correction；source policy、Stage 0与Program均冻结，Action Meta module/parameter为0。held5由同一macro5 checkpoint分别物化
correct full、first+final与disjoint same-task K4条件，每个条件是一套完整rank12+4 rank16；learned language control不读held video/action/reward。

五臂paired strict250为carrier/language/full/endpoints/same-task=`43/42/35/40/44`。full逐task为Spatial0 `27`、Spatial9 `4`、
Object8 `4`、Goal5 `0`、Long6 `0`；breadth`3/5`、carrier retention`28/43`、相对language/endpoints为`-7/-5`，只有same-task
retention `33/35=94.3%`通过。250个task/state rows、source model、tokenizer、normalization、environment/policy RNG、三个bank、唯一
compiler checkpoint和single complete rank16均通过配对与authority检查；shuffled/reversed未使用。因此macro5是明确G3 non-pass。

无梯度几何显示full与same-task residual update cosine为`0.992--0.999`，而full与endpoints已有约`38--47%`相对差异；对四个G1非零
held residual，full相对G1可行方向的整体cosine仅约`0.001--0.005`。这把最早接口定位为shared selection方向尚未学会，而非Pass A
没有动态、多视频置换/同任务鲁棒性失败或native bank无容量。macro5仍含50步warmup且fit metrics广泛改善，所以后继先执行原schedule的
macro10训练时标证伪；若closed loop与方向不显著改善，再修改signed-pooling confidence/initialization或shared supervision，而不是
盲续macro20/40。

关键artifacts：

- `runs/outputs/pi05_ecp_shared_compiler_g3_fold0_m5_5140362_gpu01p12_r2_20260825/`；
- `runs/analysis/pi05_ecp_shared_compiler_g3_gate_m5_5140362_990557d_20260825/report.json`；
- 四条新strict250结果分别位于对应`pi05_ecp_shared_compiler_g3_*strict250*20260825/`目录。

## 22. G3 v1 macro10与fit-native-span定位

clean detached `5140362`按相同v1合同从fresh训练至macro10/190 optimizer updates。五臂paired strict250为
carrier/language/full/first+final/same-task=`43/42/38/39/40`；full逐task为Spatial0 `32`、Spatial9 `2`、Object8 `4`、
Goal5 `0`、Long6 `0`，breadth`3/5`、carrier retention `32/43`、相对language/endpoints `-4/-1`，same-task retention
`32/38=84.2%`。全部bank、single-checkpoint、single-rank16、配对与信息墙authority通过，shuffled/reversed未使用；科学Gate为
non-pass。该结果只淘汰v1间接functional监督和共享global clip在原训练时标内足以学会shared selection的假设，不淘汰native bank或
Native-Factor整体。

同一冻结v1训练面显示全部190 steps被global gradient clip，macro10 scale path gradient约比input/output query高一个数量级；
macro5到10的query-key参数只发生约`1.7--2.1%`相对变化。与此同时，fit-only K1 span diagnostic对6 tasks/9 verified members得到
full-to-native update cosine median`0.7029`、native named/global functional retention median`0.7855/0.7981`和positive action
benefit `9/9`。因此后继活动修正转为fit-only K1 native-feasible factor supervision与selection/scale分离clip；它不把task/video/member
键变成deployment路由，K2/K4不读取teacher，最终结论仍由held closed loop决定。

关键artifacts：

- `runs/outputs/pi05_ecp_shared_compiler_g3_fold0_m10fresh_5140362_gpu02p45_r2_20260825/`；
- `runs/analysis/pi05_ecp_shared_compiler_g3_gate_m10fresh_5140362_4770c5e_20260826/report.json`；
- `runs/analysis/pi05_ecp_shared_compiler_g3_functional_span_4770c5e_gpu01p27_20260826/aggregate.json`；
- `runs/analysis/pi05_ecp_shared_compiler_g3_native_teacher_v2_formal_20260826/manifest.json`（clean pushed
  `93dffc7`，fit-only K1 authority：50 tasks、451 task-video、662 teacher states；不是Gate结果）；
- 三条macro10 video arms位于对应`pi05_ecp_shared_compiler_g3_*strict250*m10fresh*20260826/`目录。

## 23. G3 v2 macro5与teacher-credit冲突定位

clean detached `2a7f760`从fresh完成v2 macro5/95 optimizer updates。五臂paired strict250的
carrier/language/full/first+final/same-task=`43/42/41/38/37`；full逐task为Spatial0 `34`、Spatial9 `5`、Object8 `2`、Goal5 `0`、
Long6 `0`，breadth`3/5`、carrier retention`33/43`、相对language/endpoints`-1/+3`、same-task retention`30/41=73.2%`。
所有bank、single checkpoint、single rank16、配对和Action Meta 0 authority通过，shuffled/reversed未使用，故这是科学non-pass。

同一fit K1真实bank的固定条件审计显示，macro5的paired update cosine仍仅`0.00299`，spectrum较step0更差；teacher-selection与旧
functional selection梯度范数为`0.3235/21.8015`，teacher spectrum与旧scale梯度cosine为`-0.989657`，显式gradient wall泄漏为0。
teacher-only反事实能沿正确方向下降，因此失败不是teacher cache、native capture或autograd断路，而是v2在同一步内让旧functional
credit覆盖direct mapping credit，且等权subspace/update objective低估最终paired update方向。后继转向隔离fit-K1 mapping acquisition，
随后再以非干扰阶段恢复K2/K4 functional职责；不续训v2、不做seed/LR/rank/K小扫，也不进入G4。

后续同一固定bank的free-query/解析反事实进一步定位了selection内部接口：full-native query普通梯度500步只有`0.1624` update cosine，
而固定G2 measure下的FP64 weighted inverse-covariance dual为`0.99750`；该dual用现有small-logit antithetic exponential softmax仍为
`0.99749`。这证明当前bank和signed-pooling形式可以实现teacher，失败来自约`1e6`条件数dual的shared acquisition。该结果只授权下一步
fit-only leave-task-out dual-basis oracle；在解析压缩门通过并更新active schema前，不把dual label或full-width query写成已定架构。

关键artifacts：

- `runs/outputs/pi05_ecp_shared_compiler_g3_native_teacher_v2_fold0_m5fresh_2a7f760_gpu02p45_r2_20260826/`；
- `runs/analysis/pi05_ecp_shared_compiler_g3_gate_native_teacher_v2_m5_2a7f760_20260826/report.json`；
- 三条video strict250结果位于对应`pi05_ecp_shared_compiler_g3_*strict250_native_teacher_v2*m5*20260826/`目录；
- 一次性固定bank梯度审计位于ignored `.codex/tmp/g3_v2_fixed_probe_meta9_*`，其结论已固化在当前持久文件中。

## 24. G3四family task-LOTO dual-basis formal non-pass

clean pushed detached `main@e7d86b0`对formal40 schedule中实际有K1 authority的50个fit tasks运行四family解析probe。每task按
预注册口径最多取两条确定性video，共98 conditions；代表targets为q20、v21、action-in36、action-out37。capture与replay均使用冻结
G2 `c1493a1/macro_00000020`的fixed measure、真实native X/Y、四类动态Y bank、antithetic bounded signed pooling和verified member
rank4 factors，Action Meta与held reads均为0。

完整FP64 analytic dual reference的task-mean update cosine median/p10/min为`0.996949/0.995468/0.993884`，worst-video为
`0.996487/0.994944/0.991649`，因此数据、dual、bank与回放合同有效。用另外49 tasks的rank-block projector构造同一套task-balanced
LOTO basis后，最大128维overall task-mean median/p10只有`0.288444/0.249615`，50 tasks中没有一个达到`0.95`；family median为
action-in `0.999983`、action-out `0.146885`、q `0.000490`、v `-0.000586`。预注册Gate明确non-pass，故没有扩到38 targets，也没有
把compact dual code接入active compiler。

随后只读同一formal authority的fit-only反事实区分了失败层次。对最差q/v/action-out bank在固定LOTO span内做正确的functional
least-squares，128维update cosine由原欧氏投影的约`-0.012/-0.029/-0.118`提高到约`0.684/0.507/0.805`，证明欧氏realizer几何
确实错误；但要到约384--512维才接近强回放，v即使到当前稳定span仍约`0.933`。selected-task effect-basis反事实也需要相近宽度，
所以该结果不授权恢复fixed raw/effect basis或历史fit-span realizer。

直接content-key机制screen显示，现有width-shared 64维近线性key在两条同task视频上拟合后，第三条video的q/v/action-out均值只约
`0.52`；owner-specific复制只改善seen fit。按解析容量选取512维owner-specific key时，自由functional image在三条held-video条件
约`0.99`，但其最小/最大奇异值比约`1e-8--1e-6`，尚未证明shared Program query能稳定取得scores。因此后继只沿直接
Program-query/content-key attention继续：先验证截断谱下exact bounded signed-softmax的可用conditioning，再针对owner-specific key、
非线性高容量query生成和output-group相对幅度修正唯一canonical compiler；training-only dual/score、task/video键和解析系数不得进入
deployment，held5 closed loop仍是G3结论。

随后selected-condition exact-score反事实比较了两种content key。随机512维key必须保留约`1e7`条件数的tail才使q/v/action-out达到约
`0.993/0.984/0.994`，在`1e6`内只有约`0.956/0.936/0.966`，且同task跨video query方向不稳定；因此没有把“加宽key”当成
充分修正。直接以真实native X/Y作为key时，`1e6`截断与固定`0.01` small-logit scale即可使三family跨三video的update cosine均值达到
`0.99886/0.99551/0.99788`、minimum达到`0.99810/0.99447/0.99703`，无需逐bank读取后校准。q的八个output groups若分别单位化会降到
约`0.967--0.985`，保留一个公共score scale和`[0,1]`相对group gains则恢复约`0.999`。这些值只授权下一步owner-native direct key、
非线性高容量Program query与bounded group-gain mapping acquisition，仍不是shared mapping或closed-loop Gate结果。

关键artifact：

- `runs/analysis/pi05_ecp_g3_dual_basis_four_family_loto_e7d86b0_gpu01p012345_20260826/`。

## 25. 当前保留结论

1. EMBER输入输出目标不变，ECP核心尚未被完整实验反证。
2. task-local LoRA与mobile-rank4容量充足；native video basis和shared selection mapping是顺序待验证的最早接口。
3. policy effects保留为critic；neural `q_pi`、fit-span/fixed-effect realizer、GOMQ、PECS与v24均不进入active路线。
4. Program schema、native bank和compiler角色已经固定，不能共同自由旋转或退回全局task code。
5. Stage 0必须证明full video超越endpoints；shared compiler必须直接用held closed loop证明跨task mapping。
6. 分阶段冻结后必须进行冻结backbone、冻结carrier的全Writer联合训练。
7. shuffled/reversed只用于最终冻结checkpoint的时序特异性评测。

## 26. 证据恢复方式

- 活动科学合同：`AGENTS.md`、`docs/current_owner_requirements.md`、`docs/concept.md`。
- 当前架构：`docs/event_conditioned_policy_compiler_design.md`。
- 当前状态：`task_plan.md`、`findings.md`、`progress.md`。
- 精确旧实现与配置：以上Git提交或`git log`中相邻提交。
- 大型formal结果：本地ignored `runs/`中的唯一checkpoint、raw rows与aggregate；人工process资产除外，已明确删除。

任何旧提交中的“active”“next”或“current”只代表当时状态，不能覆盖当前owner要求。

## 27. G3 current-bank-conditioned F1 operator formal pass

第二位全新专家在远程`main@ed2883b`的完整可达历史上复核后，确认G3此前最早失效接口是current-bank gauge：稳定任务功能被写成
随video candidate covariance旋转、条件数约`1e6`的minimum-norm dual coordinates。活动Pass B因此修正为B0流式累计每视频
unit-mass mean/covariance及Program-conditioned native anchors，regularized solve形成query，再由B1重放同一真实bank执行exact
positive-minus-negative softmax pooling。owner同时明确Final必须保留整套Writer随机初始化、从头端到端fresh联合训练的正式选项；
这不改写专家原文。

clean pushed detached `main@435cb4a`随后先隔离验证operator capacity。固定authority沿用四family task-LOTO的50 fit tasks、98 K1
conditions与targets q20/v21/action-in36/action-out37；使用真实native X、abs/adj/init/goal Y、冻结G2 measure、bounded analytic
native anchors、FP64 covariance截断谱solve、output-group bounded relative gain及materialized/streaming exact replay。q/v/action-in/
action-out task-mean update cosine median分别为`0.999871/0.999824/0.999960/0.999884`，minimum分别为
`0.999757/0.999544/0.999951/0.999743`；536 member-family rows的streaming-to-materialized minimum为`0.99999988`。
全部预注册门通过，Action Meta为0且held reads为0。

该pass只证明bank statistics、截断谱solve、group gain和chunked signed replay能保存analytic Native-Factor上限；没有训练或证明shared
Program-to-anchor mapping，也不是closed-loop G3 Gate。关键artifact：

- `runs/analysis/pi05_ecp_g3_bank_operator_f1_435cb4a_gpu01p123456x2_20260826/`。

## 28. G3 F2 candidate-local `C=I` formal non-pass

clean pushed detached `main@2199a76`完成active design预注册的一次性`global_statistics_off`消融。该模式令`C=I`并关闭current-bank
covariance/preconditioning，但仍保留B0每video单位measure centered first-moment native anchor与B1对真实X/Y的exact signed replay；
Program/source/scale冻结、Action Meta为0，只训练shared anchor scorer。六卡world6从fresh训练到macro5/25 optimizer updates，mean
recovery从macro1 `.000639`升至macro5 `.019690`。

同一macro5由六个独立只读worker完整评估451个唯一条件：329 fit、40 held-video、82 task-holdout。task-equal fit、held-video、
task-holdout recovery median分别为`.022243/.022858/.018919`；held-video action-in/action-out/q/v median分别为
`.039958/.022185/.004722/.023158`。F2要求held-video overall至少`.75`、每family至少`.65`、task-holdout至少`.60`，故三个
primary checks全部失败。所有worker自然完成，Action Meta module/parameter、held gradient及shuffled/reversed use均为0。

fit本身接近零，故这不是held泛化的次级掉点，而是candidate-local compatibility加first-moment anchor未能建立teacher mapping。
按照预注册因果顺序，不续训F2且不做LR/seed/width小扫；该结果只淘汰`C=I` off假设，不反证F1已证明有容量的current-bank
statistics/solve/B1 operator。下一步从fresh进入F3。关键artifacts：

- `runs/outputs/pi05_ecp_shared_compiler_g3_f2_fold0_m5_2199a76_gpu01p023456_r6_20260826/`；
- `runs/analysis/pi05_ecp_shared_compiler_g3_f2_macro5_mapping_eval_2199a76_gpu01p023456_w6_20260826/`。

## 29. G3 F3 macro5/macro10 current-bank mapping formal non-pass

clean pushed detached `main@c1e26ce`开启current-bank covariance/preconditioning，F3从fresh训练至macro5/25 updates，再以同一world6
topology、optimizer和scheduler exact-resume至macro10/50 updates。训练mean recovery依次为`.002204/.010863/.019536/.031414/
.042865/.058599/.066219/.073079/.082339/.087444`；后半程仍增长但斜率趋缓。

macro5与macro10分别由六个独立只读worker完整评估同一451条件。macro10 fit、held-video、task-holdout task recovery median为
`.089915/.089704/.096849`，held/fit为`.997650`；held-video p10为`.072144`。macro5到10 held median提高`.041271`，相邻
同任务median absolute delta `.041962`且held没有下降，但两个checkpoint都未达到F3 primary的median `.75`和p10 `.50`，所以
adjacent Gate不能通过。

macro10 held-video四family median action-in/action-out/q/v为`.125947/.177230/.013288/.052761`。信息墙、451唯一条件、split、
Action Meta 0、held gradients 0及shuffled/reversed 0均通过。该结果证明current-bank solve相对F2有增量且held泛化成立，但shared
anchor acquisition的绝对能力、尤其q/v family仍有结构瓶颈；不继续盲训macro20，先定位anchor表达与梯度接口。关键artifacts：

- `runs/outputs/pi05_ecp_shared_compiler_g3_f3_fold0_m5_c1e26ce_gpu01p023456_r6_20260826/`；
- `runs/analysis/pi05_ecp_shared_compiler_g3_f3_macro5_mapping_eval_c1e26ce_gpu01p023456_w6_20260826/`；
- `runs/analysis/pi05_ecp_shared_compiler_g3_f3_macro10_mapping_eval_c1e26ce_gpu01p023456_w6_20260827/`。

## 30. G3 F3 shared-anchor credit根因分解与修正资格

对上述macro5/macro10 formal checkpoint做只读、同bank的factor几何与单family backward后，最早接口从笼统的q/v anchor弱进一步
定位为update-only双线性credit下的input/output rank4 subspace acquisition starvation。task93/video31 macro10的q/v/action-in/
action-out实际update recovery为`.012892/.046297/.125741/.094190`；对应input/output one-sided span ceiling为
`.192547/.094122`、`.260870/.282059`、`.775570/.145645`、`.237240/.657518`。q的key gradient norm仅
`.0643/.0313`，action-out则为`2.6973/.6383`；held task2 q也复现同样两侧ceiling不足。F1 operator、solve residual、fit/held
比例与信息墙均已通过，故这不是operator、pairing或泛化问题，也没有用macro20续训掩盖。

后继只改变mapping credit：仍以完整四family update选择一个set-valued global member，将posterior detach后对同一member的
gauge-invariant input subspace、output subspace与paired update direction固定等权；scale保持冻结。六卡真实5-macro qualification中
三项loss从`.939056/.922342/.999256`降至`.923254/.902963/.997695`，Action Meta module/parameter 0且source/Program/scale
冻结。该qualification只证明新credit能直接作用于最早接口；它不是formal 451-condition Gate，正式结论须来自fresh clean pushed
detached run。

## 31. G3 F3 equal-subspace formal non-pass与family/fixed-owner修正

clean pushed detached `main@84903aa`从fresh训练equal input/output subspace加paired update credit到macro5/25 updates，再以同一
world6 topology、optimizer和scheduler exact-resume到macro10/50 updates。mean recovery由macro1 `.001262`升至macro5
`.021016`和macro10 `.070337`；macro10 input/output/update三项loss为`.771307/.825056/.930808`，证明梯度图确实优化了新目标。

macro5与macro10分别由六个独立只读worker完整覆盖同一451 conditions。macro10 fit、held-video、task-holdout task median为
`.073151/.073029/.087636`，held p10 `.057174`、held/fit `.998320`。macro5到10 median absolute task delta `.049104`且held没有
下降，但两个checkpoint都未达到`.75/.50` primary，故相邻Gate仍不通过。macro10 held action-in/action-out/q/v为
`.098990/.146806/.008482/.040693`；equal-subspace credit没有超过旧`c1e26ce`，不续到macro20。

只读gradient decomposition随后在task93/video31与独立task94/video11复现：四family与q的18个固定layer owner目标大多近正交，
共享output scorer被action-out及少数高敏感层支配；q target aggregate gradient只有per-target norm和约`.28--.32`，最大层间norm差
约`20.5x`。macro10 q mean input/output span ceiling为`.235654/.093766`而实际update cosine仅`.008407`。因此最早接口从loss
credit继续收窄到shared scorer parameter ownership。

后继canonical修正遵循第二位专家原文：四family各自共享Program/query/candidate trunks，并用固定38-target LoRA topology的
zero-init bounded FiLM调制native candidate hidden direction；禁止task/video/member/frame lookup，且不改变width、rank、loss、data、
optimizer或Gate。真实GPU profile完成一组3+3任务forward/backward/update/checkpoint，222/222 trainable tensors进入optimizer state，
Action Meta/source/Program/scale trainable均为0；全仓181 tests通过。它只获得fresh formal资格，尚不构成F3 pass。关键artifacts：

- `runs/outputs/pi05_ecp_shared_compiler_g3_f3_subspace_fold0_m5_84903aa_gpu01p012346_r6_20260827/`；
- `runs/analysis/pi05_ecp_shared_compiler_g3_f3_subspace_macro5_mapping_eval_84903aa_gpu01p012346_w6_20260827/`；
- `runs/analysis/pi05_ecp_shared_compiler_g3_f3_subspace_macro10_mapping_eval_84903aa_gpu01p012346_w6_20260827/`；
- `runs/analysis/pi05_ecp_shared_compiler_g3_f3_subspace_credit_diagnostics_84903aa_20260827/`。

## 32. G3 family/fixed-owner F3 formal non-pass

clean pushed detached `main@c3fc8e3`将四family独立trunk与38个fixed owner bounded candidate modulation从fresh训练到macro5/25
updates，并按原world6 topology exact-resume至macro10/50 updates。macro5与macro10均由六个独立worker完整覆盖同一451 conditions；
macro10 fit、held-video、task-holdout task median为`.074715/.074620/.081644`，held p10 `.058381`、held/fit `.998724`。
held q/v/action-in/action-out median为`.027938/.066509/.044464/.164942`。相邻checkpoint变化稳定但两个primary都远低于
`.75/.50`，故不继续macro20。

该结果说明family/owner parameter ownership隔离没有解决绝对mapping acquisition。它没有推翻F1 operator、G1/G2、真实native
banks或signed replay。关键artifacts：

- `runs/outputs/pi05_ecp_shared_compiler_g3_f3_familyowner_fold0_m5_c3fc8e3_gpu01p012346_r6_20260827/`；
- `runs/analysis/pi05_ecp_shared_compiler_g3_f3_familyowner_macro5_mapping_eval_c3fc8e3_gpu01p012346_w6_20260827/`；
- `runs/analysis/pi05_ecp_shared_compiler_g3_f3_familyowner_macro10_mapping_eval_c3fc8e3_gpu01p012346_w6_20260827/`。

## 33. fixed-key/raw/FiLM image诊断与direct-native修正终止

对`c3fc8e3` macro10冻结candidate map并允许task-local free event query后，稳定`1e-3`谱下q/v/action-in/action-out joint teacher
update ceiling约为`.226/.315/.975/.629`；raw-native key约`.250/.336/.960/.600`，FiLM tangent约
`.280/.381/.973/.645`。降到`1e-6`时q/v/action-out大多恢复到`.97--.99`，说明方向仍存在但被压入极弱奇异尾。action-in已经有
`.975`容量而formal held只有`.044`，证明shared stable selection也独立失败。

clean pushed `main@4117117`曾把direct-native Program query接入同一B0 solve/B1 replay并通过真实F0工程Gate；但随后的解析复核指出，
若native anchor写成`a=Cq`，再求`C^-1a`会在retained span返回raw `q`，所以它退化为已失败的跨video raw-query transfer而非新的
bank-stable mapping。该commit没有启动formal F3，活动实现随后回退。关键artifact：

- `runs/analysis/pi05_ecp_shared_compiler_g3_f3_familyowner_macro10_fixed_key_image_c3fc8e3_gpu01_20260827/`。

## 34. same-task feature-common-code oracle与task-stable anchor修正资格

在task93同一teacher members上，以videos 31/32拟合一个共同feature code并把video46作holdout。full inverse、global
inverse-square-root及8-event block inverse-square-root的两video inductive q/v/action-out均近零或极低；event-block action-in held
为`.986`。然而把video46只加入共同code估计的transductive正控制后，q/v/action-out分别达到约`.905/.912/.929`，action-in为`1.0`；
全部q/v/action-out event covariance均满rank128。该只读oracle证明共享feature chart中存在强same-task code，失败是两video
minimum-norm selection落入巨大train-only nullspace，不是继续调谱floor、loss、width或训练长度可解释。

冻结G2的`P_lang`是exact-language-only且same-task不同video确定性相同，因此后继canonical修正以`P_lang`产生固定task anchor query，
让`P_scene/P_process/rho/tau/sigma`及canonical assignment只控制video-specific event/frame measure；每个video、target/group/event先
建立detached candidate-feature symmetric inverse-square-root，再形成native anchors、native-bank solve和exact signed replay。该选择
没有task/video/member/frame lookup，仍以真实X/Y为唯一factor Value路径。关键artifacts：

- `runs/analysis/pi05_ecp_g3_feature_anchor_canonicalizer_c3fc8e3_gpu01_20260827/`；
- `runs/analysis/pi05_ecp_g3_feature_anchor_invsqrt_c3fc8e3_gpu01_20260827/`；
- `runs/analysis/pi05_ecp_g3_feature_anchor_eventblock_invsqrt_c3fc8e3_gpu01_20260827/`。

## 35. task-stable feature anchor正式增量与fixed-owner query修正资格

clean pushed detached `main@20acc33`先通过stable-anchor F0：K1有效更新cosine最低`.99999826`、相对误差最高`.001863`，K4置换
误差`1.91e-6`，Action Meta 0、38 targets/76 tensors且唯一rank16被policy消费。随后F3从fresh训练到macro5并按world6 topology
exact-resume到macro10/50 updates；训练mean recovery由macro1 `.000902`连续升至macro5 `.038661`和macro10 `.127151`。

macro5与macro10均完整评估451 conditions。macro10 fit、held-video、task-holdout task median为
`.141080/.142120/.145828`，held p10 `.116653`、held/fit `1.00737`；macro5到10 held median增加`.092745`且40/40 held tasks改善。
held q/v/action-in/action-out为`.030186/.110266/.180031/.253562`。相对`c3fc8e3` held `.074620`，stable anchor获得约一倍增量并
消除了fit/held/task-holdout落差，但仍远低于F3 `.75/.50` Gate，两个checkpoint均不能通过primary/adjacent。

为定位绝对获取瓶颈，对meta tasks `1/32/52`和target tasks `92/93/94`各缓存三条真实K1 native banks，从macro10只在单task/
单fit-video上继续优化anchor scorer 20步；另一fit与held video只读。六task train overall从`.113--.163`升到`.203--.251`，fit/
held probe均为`.200--.247`，但q train update只到`.0197--.0277`，action-in/out通常达到`.31--.49`。这证明stable code能跨bank迁移，
q低值并非仅由多任务语义竞争或held-video泛化造成。q/v同层teacher input subspace median overlap约`.07--.18`，也不支持强行共享
q/v input factors。

同一macro10 task93/video31对18个q targets逐target反传：family-shared output-query的aggregate-to-norm-sum为`.26769`，153对中
74对为负；input-query为`.27231`，76对为负。output/input key trunk分别为`.60173/.36429`，但candidate侧已有fixed-owner bounded
FiLM。语言只读panel显示raw exact-language masked mean effective task rank `4.30`，`P_lang`去owner-only baseline后的四family rank约
`3.20--3.56`且same-task跨video差为0；这保留了可用task信号，当前证据不足以把G2 language reader列为更早接口。

因此下一单变量修正是给family-shared query trunks补上与candidate侧对称的zero-init bounded fixed-owner input FiLM及fixed-owner/
output-group FiLM。固定行只对应真实LoRA target/group拓扑，task dependence仍来自`P_lang`，无task/video/member/frame lookup；banks、
feature gauge、native solve、loss、rank、data、optimizer与Gate不变，旧checkpoint不兼容。实现通过184项CPU回归且architecture guard无
hard violation；正式结论仍须clean pushed detached F0及fresh F3。

关键artifacts：

- `runs/outputs/pi05_ecp_shared_compiler_g3_f3_stable_anchor_fold0_m5_20acc33_gpu01p012346_r6_20260827/`；
- `runs/analysis/pi05_ecp_shared_compiler_g3_f3_stable_anchor_macro5_mapping_eval_20acc33_gpu01p012346_w6_20260827/`；
- `runs/analysis/pi05_ecp_shared_compiler_g3_f3_stable_anchor_macro10_mapping_eval_20acc33_gpu01p012346_w6_20260827/`；
- `runs/analysis/pi05_ecp_g3_stable_anchor_single_task_probes_20acc33_gpu01p012346_w6_20260827/`；
- `runs/analysis/pi05_ecp_g3_stable_anchor_query_gradient_diag_20acc33_gpu01p1_20260827/`。

## 36. fixed-owner/group query FiLM formal F0

fixed-owner/group query FiLM首先在clean pushed `7e232b0`通过184项CPU回归与architecture hard checks。首次detached F0尚未进入
GPU计算即失败：新模块内部helper命名为`_apply`，覆盖了`torch.nn.Module._apply`，导致`.to(device)`调用签名冲突。该失败只揭示
PyTorch模块生命周期工程错误，不构成scientific non-pass。唯一修复将helper改名为`_modulate`并新增显式`.to(cpu)`回归，形成
clean pushed `main@d64f7ad`；代码、配置和科学合同除此之外未变。

clean pushed detached `d64f7ad`随后完成formal F0并通过全部qualification checks。真实task93 K1 backward给出input/output
owner-query gradient norm `.015828/.000958`；source与Program trainable为0，Action Meta modules/parameters为0。K4四条视频保持
均匀`.25` measure，置换最大误差`1.91e-6`且teacher tensor reads为0。相同cached X/Y bank的chunk4相对one-chunk有效更新cosine
最低`.99999826`、median近1，相对误差最高`.001863`、median `6.04e-5`，feature metric误差`5.96e-8`。最终仍只materialize
38 targets/76 tensors的一套完整rank16并由policy真实消费。总时长`592.16s`，峰值allocated/reserved约`34.00/42.81GB`。

该结果只解封该checkpoint-incompatible修正从fresh运行同一F3 mapping Gate；它不证明shared mapping已通过，也不改变stable-anchor
macro10 non-pass、F1 operator结论或后续`.75/.50` primary与相邻checkpoint口径。

关键artifacts：

- `runs/analysis/pi05_ecp_shared_compiler_g3_owner_query_film_f0_7e232b0_task93_gpu01p0_20260827.log`；
- `runs/analysis/pi05_ecp_shared_compiler_g3_owner_query_film_f0_d64f7ad_task93_gpu01p2_20260827.json`；
- `runs/analysis/pi05_ecp_shared_compiler_g3_owner_query_film_f0_d64f7ad_task93_gpu01p2_20260827.log`。

## 37. fixed-owner query F3 non-pass与candidate-compatibility image定位

clean pushed detached `main@3e4e9a0`的fixed-owner/group query FiLM从fresh训练到macro5，再按同一world6 topology exact-resume到
macro10/50 updates；两个checkpoint均完整评估451 conditions。macro10 fit、held-video、task-holdout task median为
`.162011/.163128/.164562`，held p10 `.133783`、held/fit `1.00689`。相对`20acc33` stable-anchor macro10，40/40 held tasks改善且
held median增加`.02007`；但held q/v/action-in/action-out为`.032001/.111951/.256629/.256391`，仍远低于`.75/.50` primary，
相邻Gate不能通过。新增幅度主要来自action-in，不能把overall增量解释为38-target query ownership已解决。

同checkpoint四臂ablation中，移除input owner-query路径使overall仅下降`.000193`；移除output路径下降`.014820`，其中action-in下降
`.056749`而q/v约为`1e-3`。六个fit-only tasks（meta `1/36/43`、target `85/93/94`）随后分别只优化fixed input或output FiLM 20步。
即使query相对base移动约`.48--.56` RMS，input probe的q/v完整update由`.02983/.10730`降到`.02686/.10292`，output probe只到
`.03356/.08511`。因此不是formal训练把FiLM幅度压得太小。

更宽松的task-local正控制把q/v两侧所有owner/rank/event queries直接设为free tensors，保留相同candidate keys、B0 covariance solve、
B1 exact signed replay和唯一rank16合同。六task 20步后的q/v update median仅为`.06519/.14487`，q input/output为
`.18546/.09746`、v为`.19614/.29710`；所有task方向一致。该probe仍不是收敛后的严格上界，但已说明fixed FiLM不是唯一或最早充分
瓶颈，不能直接把它替换成更大的per-target query projection并期待通过。结合F1 analytic operator约`.9998`，下一机制裁决应缓存同一
真实bank，对当前candidate-key/compatibility image做收敛或解析容量审计；若该image本身不足，则修复functional candidate
canonicalization或bank-global statistic到稳定functional anchor的映射，而不是继续扫LR、seed、width或训练长度。

关键artifacts：

- `runs/outputs/pi05_ecp_shared_compiler_g3_f3_owner_query_film_fold0_m5_3e4e9a0_gpu01p012346_r6_20260827/`；
- `runs/analysis/pi05_ecp_shared_compiler_g3_f3_owner_query_film_macro10_mapping_eval_3e4e9a0_gpu01p012346_w6_20260827/`；
- `runs/analysis/pi05_ecp_g3_owner_query_path_ablation_3e4e9a0_m10_gpu01p012346_w6_20260827/`；
- `runs/analysis/pi05_ecp_g3_input_owner_local_capacity_3e4e9a0_m10_gpu01p012346_20260827/`；
- `runs/analysis/pi05_ecp_g3_output_owner_local_capacity_3e4e9a0_m10_gpu01p012346_20260827/`；
- `runs/analysis/pi05_ecp_g3_free_owner_query_capacity_3e4e9a0_m10_gpu01p012346_20260827/`。

## 38. additive joint compatibility F3 non-pass与三层根因定位

在exact current-key image证明深层q/v线性点积失容后，`main@55710bb`保留真实X/Y、B0 covariance solve、B1 exact replay、
frozen G2 Program、rank/data/loss/Gate，只给四family加入bounded additive joint compatibility并保留dot residual。该实现先以
antithetic signed rows和small-nonzero joint scalar通过formal F0，随后从fresh训练到macro5并按world6 exact-resume至macro10。
两个checkpoint各由六个独立worker完整评估451 conditions。macro10 fit/held-video/task-holdout task median为
`.126205/.128720/.129465`，held p10 `.103610`、held/fit `1.019925`；held q/v/action-in/action-out为
`.025341/.095210/.121352/.276225`，Gate明确non-pass且低于前一owner-query FiLM `.163128`。

fit-only checkpoint path ablation覆盖tasks `16/72/85/93`。关闭joint后16个family update cosine median/minimum为
`.999752/.999450`，recovery平均变化仅`-.000175`；joint-only recovery为`.0018--.0116`。wrong-language与wrong-dynamic替换
也几乎不改变最终update。对task85进一步读取内部Program：same-task full `rank_event` cosine `.99750`，wrong-task dynamic为
`.92811`；但wrong-task `P_lang`本身为`.99704`，经当前P_lang-only query trunk后各family为`.9947--1.0`。因此该formal实际仍
依赖旧dot路径，并把G2 full Program中的主要task-dynamic信息排除在content query之外；非零gradient不足以证明joint路径被使用。

同一task85又完成两fit-video训练、第三video零梯度的task-local final-factor对照。200步后current frozen key的train/held为：
target0 q `.4235/.4237`、target1 v `.4849/.4852`、target18 q `.2920/.2425`、target19 v
`.2905/.2212`、target34 q `.2741/.1345`，action-in target36为`.9963/.9961`。owner-local raw-native projection没有
改善深层held。另一个score-supervised对照中，约`.983` score cosine对action-in产生约`.991` factor，但对浅层q/v只产生
`.125/.075`，直接显示高条件数solve对小anchor误差的放大。解析direct reference仍约`.99--1.0`，所以F1算术与native span没有
失败；失败的是从稳定Program和candidate content识别可跨video工作的近精确functional anchor。

该结果淘汰`55710bb`的P_lang-only/small-residual joint scorer及“换成raw native key即可”的直接修正，不淘汰G1/G2、
bank-conditioned两阶段Pass B、signed pooling、rank4或所有nonlinear scorer。关键artifacts：

- `runs/outputs/pi05_ecp_shared_compiler_g3_f3_joint_compatibility_fold0_m5_55710bb_gpu01p023456_r6_20260827/`；
- `runs/analysis/pi05_ecp_shared_compiler_g3_f3_joint_compatibility_macro10_mapping_eval_55710bb_gpu01p023456_w6_20260827/`；
- `runs/analysis/pi05_ecp_g3_joint_path_causality_t{16,72,85,93}_55710bb_macro10_20260827.json`；
- `runs/analysis/pi05_ecp_g3_joint_path_context_t85_55710bb_macro10_20260827.json`；
- `runs/analysis/pi05_ecp_g3_input_content_capacity_t85_j{0,1,18,19,34,36}_s200_factor_55710bb_macro10_20260827.json`。

## 39. fit-only functional consensus与target-native谱诊断

`55710bb` non-pass之后，没有继续macro20或做超参扫，而是先问两个更早的可证伪问题。第一，使用40个mapping-fit tasks、每task全部
mapping-fit videos且明确排除预注册held video，对每个verified member的真实rank4 updates做small-core均值与rank4 truncated SVD。
该fit-only consensus在held video上的overall recovery median/p10/min为`.945820/.935622/.908717`，mean为`.945704`；q/v/action-in/
action-out family median分别为`.931557/.924778/~1/.926581`。set-valued member upper bound几乎相同，说明member歧义也不是主要限制。

第二，对同一fit teacher authority读取native input方向谱。family-shared q/v top128 energy均约`.7966`，top256约`.88`，达到90%需要
约306/309维；固定target自己的top128在target0/1接近1、target18/19约`.966`、target34约`.976`、target37约`.959`，action-in
完整32D为1。由此淘汰“先family压缩、后owner调制足以保留深层方向”，但不支持简单扩大全局width；证据指向38-target native basis后
再family-share。

当前replacement据此使用full G2 Program query、target-native basis、无residual bypass的normalized bilinear compatibility，以及
fit-video consensus paired-update-only credit。真实K1/K4 smoke通过chunk、K4 permutation、76 tensors、policy consume和Action Meta 0；
world6一步profile覆盖6 tasks/12 K1 conditions，`91.44s`、peak allocated约`25.65GB`，Q/K/owner/gain gradients均finite/nonzero。
首次profile在同步前发现旧P_lang-only稳定神经支路64个参数无梯度；该冗余旁路被删除并以确定性P_lang+owner/rank稳定视图替代，第二次
profile完整通过。以上仍是fit-only机制与工程qualification，不是451-condition F3 Gate。关键artifacts：

- `runs/analysis/pi05_ecp_g3_fit_consensus_update_held_video_ceiling_20260827.json`；
- `runs/analysis/pi05_ecp_g3_fit_teacher_native_input_subspace_spectrum_20260827.json`。

## 40. full-Program functional-anchor F3 formal non-pass与双重结构根因

clean pushed detached `main@3062de8`的full-Program、38-target native basis、primary normalized bilinear functional-anchor实现从fresh
完成macro5/25 steps；完整451-condition评测的fit/held-video/p10/task-holdout为
`.084298/.082754/.072027/.093856`，held/fit `.981684`。四family held q/v/action-in/action-out为
`.020707/.065711/.084290/.171636`，F3 Gate明确non-pass。train、held-video与task-holdout等量级，F1 operator/solve、信息墙、
Action Meta 0、chunk replay与唯一rank16均正常；因此没有续macro10，也没有做超参扫。

后续fit-only几何发现一套由40 fit tasks构造的task-independent universal rank4在held-video/task-holdout上已达
`.825054/.835443` median，足以绕过原`.75/.50` mapping Gate而不证明Program/video因果。把它与现有carrier12合并后再压回rank12的
update cosine为`.998741`，但直接代数相减后的task residual在task85 q/v真实native input bank中只保留约`.828/.765`解析可达性；
所以这只是新carrier假设，必须从完整expert-minus-new-carrier重新投影验证，不能直接作为新teacher。

四task因果干预中，wrong-task full Program仍保留q/v/action-in/action-out平均`.973/.981/.992/.948`的最终update cosine；wrong-task
bank则使q/v/action-out降到`.863/.834/.569`，说明当前模型主要响应bank common content而非task Program。task85两fit-video的
task-local current-key scorer对held q/v input subspace只到`.188/.177`，新target-native pointwise projection也只有`.171/.130`，
原teacher direct native reference约`.997`；去掉universal项没有修复。一次fit-only backward又显示约`99.88%`原始gradient energy
位于candidate encoders/trunks。综合证据把最早接口定位为：旧carrier遗漏公共修正，同时pointwise Program-to-functional-content
canonicalizer没有获得task-conditioned selection；两者均需机制正对照后再形成唯一下一实现。

关键artifacts：

- `runs/outputs/pi05_ecp_shared_compiler_g3_f3_functional_anchor_fold0_m5_gpu01p012345_r6_20260827/`；
- `runs/analysis/pi05_ecp_shared_compiler_g3_f3_functional_anchor_macro5_mapping_eval_3062de8_gpu01p012345_w6_20260827/`；
- `runs/analysis/pi05_ecp_g3_fit_consensus_geometry_v3_20260828.json`；
- `runs/analysis/pi05_ecp_g3_functional_anchor_causality_t{16,72,85,93}_3062de8_macro5_20260828.json`；
- `runs/analysis/pi05_ecp_g3_functional_anchor_input_content*_t85_j{18,19}_s200_factor_3062de8_macro5_20260828.json`；
- `runs/analysis/pi05_ecp_g3_functional_anchor_gradient_groups_t85_3062de8_macro5_20260828.json`。

## 41. IEEE fresh F3 non-pass与functional-polar根因见证

对F1与canonical B1的数值路径复核发现旧runtime继承TF32，导致约`1e6`条件数native dual的score cancellation。`main@78b7e58`
把native dual FP32 matmul固定为IEEE并通过真实F0；随后从fresh完成macro5/25 steps及完整451-condition评估。fit/held-video/p10/
task-holdout为`.086508/.083131/.072629/.096191`，held/fit `.960958`；held q/v/action-in/action-out为
`.021698/.065269/.085933/.173804`。因此IEEE是必要数值修正，但没有改变shared mapping的`.08`量级non-pass；该run未续训。

同一post-`Wk`真实bank的后续解析将actual mapping写为`J_r=C_r C_0^+ H`，其中`C_0`是B0 base covariance、`H`是event-centered
native/key image、`C_r`是rank-specific B1 replay covariance。task93/video2的深层q target34、v target19、action-in target36、
action-out target37在该functional image中的task-local replay update cosine约为`.996/.999/1.000/.998`，证明bank/key内容本身有
容量。per-event polar对q只有约`.947`而其它families接近1；跨rank共享global polar对v/action-out降至`.915/.831`；不用per-event
feature whitening时q约`.911`。这些read-only witnesses指定了首版rank-specific global cross-event functional polar与保留feature
whitening的修正，不构成F3 Gate pass。

active v4据此删除旧`C=I`与Euclidean normalized-bilinear开关，只保留当前bank在线计算的detached polar gauge、bounded B0 anchors、
native solve和B1 exact signed replay。最终weights仍由共享Program query与content keys计算并对真实X/Y加权，不保存task/video状态。
该实现必须先通过真实K1/K4 F0，再从fresh接受相同451-condition F3；内部解析cosine不能代替Gate。

关键formal artifact：

- `runs/outputs/pi05_ecp_shared_compiler_g3_f3_ieee_fold0_m5_gpu01p012345_r6_20260828/`；
- `runs/analysis/pi05_ecp_shared_compiler_g3_f3_ieee_macro5_mapping_eval_78b7e58_gpu01p012345_w6_20260828/`；
- `runs/analysis/pi05_ecp_g3_native_dual_precision_audit_abff0a7_20260828/`；
- `runs/analysis/pi05_ecp_shared_compiler_g3_f0_ieee_precision_78b7e58_gpu01p1_20260828.json`。

## 42. functional-polar执行优化与吞吐资格non-pass

在没有启动新formal run的情况下，对v4做真实task93/video K1 profile。初始可运行condition为`82.114s`，其中compiler forward
`59.595s`、backward `11.615s`、Pass A/native capture约`5.283/5.622s`。`da3fd3e`保持同一个compiler和科学公式，只加入单次
frozen X/Y replay cache、同shape polar batching、IEEE FP32 statistics/solve、global/per-event解析分工及thin-QR small SVD；全仓
`189 passed`。最终condition降为`58.332s`，compiler forward `35.753s`，polar由约`38.15s`降为`14.15s`，peak allocated/reserved
约`29.34/30.70GB`。full cache coalescing与去activation checkpoint均在A40上真实OOM并已回退。

macro5只有25 optimizer updates，却固定执行300个K1 conditions；六卡理想训练下限仍约`49min`，并未包括451-condition Gate评测。
owner据此明确墙钟成本必须与规模相称。当前full per-bank polar执行形态的吞吐资格non-pass，未运行K4 F0、formal F3、训练或评测，
没有新checkpoint或scientific Gate结果。该结论保留functional witness的科学证据，只要求在下一formal前获得数量级结构性降本；后继
设计先交由全新专家复核，不在历史记录中预判。

## 43. 第三次专家复核与low-dimensional bank-adaptive sketch路线

第三位全新专家锁定远程`main@9b52e59`及其可达历史，审计了G1/G2、全部G3 formal/diagnostic证据、full functional-polar源码与
`da3fd3e` profile。1033行原始回复逐字保存为`docs/expert_review_20260828_g3_functional_sketch.md`。专家判断native bank容量、
current-bank global context必要性与functional coordinate mismatch均已有可靠证据，但连续shared F3没有取得task-specific Program
ownership；fit-only universal rank4又证明旧absolute recovery可被common residual绕过。full functional-polar因此降为fit-only
teacher/reference，不再获得另一轮deployment formal资格。

后继按一次无训练low-dimensional nested sketch rank curve、一次12-task free-query/shared student容量—因果Gate、再恢复完整451-condition
F3的顺序推进。owner接受该路线并补充leave-one-task-out universal、meta/target各一个true task-holdout、首个student冻结sketch
basis/statistics及在shared结果前一次性校准causal Gate。该节点只形成新active contract，尚未产生新GPU run、checkpoint或Gate结果。

## 44. low-dimensional native-Q sketch S1 formal non-pass

clean pushed detached `main@27bde62`实现sealed fixed nested projection、current-bank native/key cross-image、rank16/32/64共享前缀、
流式`C_rQ` full-native free-query与exact X/Y signed replay。预注册S1包含任一row至少`.95`的必要条件，因此沿此前固定使用的
task93/q20 actual-operator witness运行全部两条sealed K1 conditions作为formal early disqualifier；该早停可否决合取Gate，但不估计
50-task分布。

四个rank64 task/video/member rows的sketch-to-teacher为`.156687--.157438`，input/output full-native free-query最低为
`.413974/.253733`；同一exact key的sealed F1 analytic-to-teacher为`.995560--.997907`。streaming/materialized最低`.9999769`，
Action Meta与所有trainable parameters为0。故S1容量Gate formal non-pass，不继续其余96 conditions，也不训练native-Q sketch shared
student。exploratory完整`H`最佳top64仍远低于门，补充排除fixed random projection偶然性。按专家预先规定的fallback，下一资格改为
不经`Q_g q_tilde`瓶颈的pure low-dimensional set-summary candidate-logit执行面，并先做12-task free-query positive control。

关键artifact：

- `runs/analysis/pi05_ecp_functional_sketch_s1_early_q20_27bde62_gpu01p2_20260828/`。

## 45. S2 set-summary task93/q20两轮formal non-pass

专家S1失败分支首先在固定task93/q20建立pure low-dimensional set-summary witness：两条video共同拟合、第三条video严格zero-gradient，
最终仍对真实native X/Y做eventwise signed pooling。clean detached `main@4d84dee`的v1错误地把fresh seeded 128D projection当作existing
candidate encoder；1000步后fit/held为`.328188/.175318`。同bank nested global/eventwise free logits分别达`.9999996/.9999861`，
证明bank、signed pooling和两种reduction没有硬容量上限，也暴露v1的candidate authority错误。

clean detached `main@6b97100`的v2显式加载并冻结`78b7e58/macro5`旧F3中fit-trained的candidate encoders、family trunks、metadata与
key projections，共609 tensors/8,006,400 parameters；其余函数类、数据、loss、1000步与Gate不变。v2 fit median仅`.349191`，held
`.131624`、held/fit`.37694`、held input/output`.112037/.038104`，五项Gate全部失败；相对v1仅有小幅fit改善且held更差。
两轮均实测Action Meta 0、source/G2/compiler 0 trainable、无held outcome/gradient及shuffled/reversed。

因此S2尚未进入12-task或shared Program训练。历史证据只淘汰随机chart组合，以及冻结旧25-step F3 chart足以支持当前学生的假设；不把
它扩大为native bank、signed pooling、set summary或所有candidate encoder失败。下一阶段先以fit-only可训练chart诊断区分“旧chart没有
获得合适credit”和“candidate score函数类不足”，再决定唯一formal修正。

关键artifacts：

- `runs/analysis/pi05_ecp_g3_set_summary_s2_witness_task93_q20_v1_gpu01p0_20260828/`；
- `runs/analysis/pi05_ecp_g3_set_summary_s2_witness_task93_q20_v2_gpu01p0_20260828/`。

## 46. Program-primal/current-bank-global-dual的P0与P1通过

后续behavior-identifiability证据表明task93/q20的cross-episode flow gradient存在有效rank4 descent，而旧selector和bank-independent
dual无法跨视频保持；同一稳定primal经每条当前bank的global covariance对偶化并用同一measure replay后，fit/held立即恢复到约
`.904--.911/.901`。active v5据此改为共享Program预测native primal、当前video bank确定性产生dual，再对真实X/Y做antithetic
signed pooling；旧functional-polar与low-dimensional scorer均未恢复。

clean pushed detached `main@e2f9d33`的真实38-target K1/K4 P0通过。固定candidate microblock修复了近`1e6`条件数下外部chunk改变
FP32 covariance/replay归并树的问题；最终chunk4/one-chunk raw/solve/conditioning error为0，minimum update cosine
`0.9999999999999998`，K4 permutation error`2.384185791015625e-07`。Action Meta 0、source/G2冻结、全部梯度、四类Y边界、
uniform K、76-tensor唯一rank16与真实policy consumption均成立。

随后clean pushed detached `main@c9e8198`在六个预注册fit tasks、q/v浅中深及action-in/out八targets上完成P1。每task只优化跨两条
fit video共享的task-local primals，预注册held video零梯度，fit-only scale固定。fit/held median recovery为`.971731/.954539`，
held/fit`.982308`，held/optimistic`.992193`；四family held medians为`.939825/.941630/.995402/.945222`，minimum task
held`.935001`，全部Gate通过。该结果只证明current-bank primal→dual/replay的跨task/video容量，未证明shared Program mapping；下一
历史阶段是P2 frozen full-Program-to-primal scorer。

关键artifacts：

- `runs/analysis/pi05_ecp_shared_compiler_g3_v5_f0_e2f9d33_task93_gpu01p1_20260828.json`；
- `runs/analysis/pi05_ecp_primal_capacity_p1_v1_c9e8198_gpu01p012345_20260829/`。

## 47. 95-task behavior sufficiency把最早接口上移到G2跨task表征

P1通过后，P2 frozen-Program shared mapping在25 steps只缓慢升至约`.154`，Program-to-primal task-held读出长期约`.20--.26`；继续
修改operator已与P1证据冲突。按第三次专家复核预注册的“task-local/behavior strong而shared low则重开Program sufficiency”分支，补齐
95 tasks的两组disjoint cross-episode flow-gradient panels，并从`c1493a1/macro20`提取每task两套disjoint K4及一套K1 Program。

75 fit tasks按meta/target role各50%建立的rank16 behavior manifold，在meta-held15+target-held5对独立panel-B/consensus为
`.7160/.8006`；universal为`.1908`，rank32只比rank16增加约`.0526`。四个固定1000-step shared readers在fit75可达约`.97--.98`，
但held exact rank4仅full`.2695`、process`.2470`、dynamic`.2368`、language-only`.2687`。full的q/v/action-in/action-out为
`.4335/.2873/.1583/.1886`，meta-held/target-held为`.3684/.1707`。wrong-Program margin`.2054`说明旧Program有task identity，
但没有把未见task映射到正确policy-behavior方向。

该结果不推翻G2原动态Gate，也不恢复旧Writer；它把原结论精确收窄为“Program保留视频动态，但尚未证明policy-behavior sufficiency”。
活动计划因此暂停G3 operator迭代，保持Program schema、Stage0 v3、uniform K与原loss/Gate，从旧qualified model tensors初始化、fresh optimizer
增加process-only training decoder和fit-only rank16 behavior alignment；20 held tasks只作exact-rank4 Gate。真实task74 K4一步已证明新增
loss对decoder与既有process fusion均有有限非零梯度，source/Stage0/Action Meta为0；formal结果尚未产生。

关键artifacts：

- `runs/analysis/pi05_ecp_g2_behavior_manifold95_rolebalanced_5781694_gpu01p6_20260829.json`；
- `runs/analysis/pi05_ecp_g2_behavior_sufficiency_probe75_20_5781694_gpu01p6_20260829/`；
- `runs/analysis/pi05_ecp_g2_behavior_sufficiency_program95_combined_5781694_20260829/manifest.json`。

## 48. G2-B pointwise decoder formal non-pass与decoder-free topology修正

clean detached `main@5cbe76e`的pointwise behavior-aligned G2-B以fresh optimizer运行到macro60。behavior loss由`1.2723`降至
`.7080`，旧动态full-vs-endpoints由macro10的`31.85%`增至macro60的`39.40%`；但panel-B exact rank4在macro10/20/40/60仅
`.1837/.2622/.2938/.2828`，最终consensus`.3027`，meta/target held为`.3803/.1853`。除behavior alignment外旧动态、K、probe、
event和same-task checks均通过；因此formal结论为科学non-pass而非工程故障或明显训练不足。

冻结macro60 Program后的fresh reader、fit-only kernel/linear CV及old-vs-new geometry一致表明：fit code可被reader近乎完全拟合，task-held
仍约`.26--.30`；full Program的fit pairwise behavior correlation只从`.1610`变为`.1694`，official held仍约0。负结果只淘汰“通过
pointwise training reader把坐标loss间接传给Program即可形成跨task behavior geometry”这一实现；不淘汰Stage0、固定Program schema、
rank16 behavior basis或G3 primal/current-bank dual。

随后在原fit75内预注册role-stratified train60/internal-held15，原meta-held15+target-held5作为official held20不参与本轮训练、选择或
修正。train60 basis对internal15的panel-B/consensus oracle为`.6184/.7158`，四family为`.6556/.7373/.4550/.6676`。当前活动实现
删除pointwise decoder及其v2 config，改为无新参数的完整Program behavior-kernel loss：六个固定block-equal Program fields保留
owner/event次序，两组disjoint same-K views的Program cosine kernels直接对齐panel-A与consensus factor-cosine kernels。internal Gate
同时检查train/internal两role topology、旧动态资格，并用train60-only fixed kernel-ridge执行exact rank4 readout；official held20继续冻结。
三卡一步已验证distributed gradient、纯Native Stage0和Action Meta 0，正式internal结果尚未产生。

关键artifacts：

- `runs/outputs/pi05_ecp_natural_program_g2_behavior_fold0_m10_5cbe76e_gpu01p012345_r6_20260829/`；
- `runs/analysis/pi05_ecp_g2b_frozen_program_probe75_20_5cbe76e_gpu02p7_20260829/`；
- `runs/analysis/pi05_ecp_g2b_program_geometry_old_vs_macro60_5cbe76e_20260829.json`；
- `runs/analysis/pi05_ecp_g2b_internal_task_holdout_basis60_15_5cbe76e_gpu02p7_20260829.json`。

## 49. G2-B role-local kernel formal non-pass与joint-role credit graph修正

clean detached `main@60fb18b`的v3以五卡、fresh optimizer运行macro5/15 updates并完成预注册internal Gate。最后一个
5+5 batch的panel-A/B correlation为`.7036/.7037`；旧动态Gate也以full-vs-endpoints `13.945%`改善、active-events中位数4、
one-event 0、same-task/probe/K1/K4全部通过。然而全量train60 topology仅`.2315/.2358`，internal meta仅
`.2152/.2332`；internal target的`.7842/.7930`为旧Program已有的高基线，不是本轮产生的新能力。train60-only
kernel-ridge的panel-B/consensus exact role-equal只`.1207/.1253`，wrong Program margin为负。故v3为明确scientific non-pass，
official held20没有被读取。

确定性schedule审计给出了比“训练不够”更早的解释：v3的role-local 5-task kernels在meta45上只约束126/990 task
pairs，并分成5个互不连通components。每个component可在自己的相对坐标中取得高local correlation，而objective不会给它们之间
的几何梯度。这一证据只淘汰role-local minibatch topology objective，不淘汰decoder-free Program credit、Stage0、Program schema或behavior authority。

新v4利用authority中稳定的meta-target cross-role关系（panel-A对consensus平均`.8629`），在原5 meta+5 target batch内以
`.5/.25/.25`等质量对齐joint/meta/target kernels。这把预注册15个batches的fit60监督图改为483/1770 edges、minimum
degree 9、唯一connected component，却不增加reader、模型参数、forward或deployment路径。dirty三卡真实profile保持`18.33s`
一步与`9.98GB` peak，joint loss对Program梯度非零，source/Stage0/Action Meta仍全部冻结。该profile只作执行证据，v4仍须重跑
同一macro5/internal Gate。

关键artifacts：

- `runs/outputs/pi05_ecp_natural_program_g2_behavior_kernel_fold0_m5_c8fee96_gpu01p01234_r5_20260829/`；
- `runs/outputs/pi05_ecp_g2_joint_behavior_kernel_v4_profile_dirty_gpu01p012_r3_20260829/`。

## 50. G2-B joint-role v4 formal non-pass与global-calibrated v5

clean detached `main@4eb8b8c`的v4完成五卡macro5/15 updates。旧动态Gate继续通过：full相对endpoints改善
`14.6553%`、active events中位数4、one-event 0、same-task/probe/K1/K4全部成立；但train60 topology A/B仅
`.2360/.2362`，internal meta为`.2064/.2257`，internal target为`.7512/.7634`，exact panel-B/consensus role-equal
仅`.1129/.1177`且wrong margin为负。v4因此明确non-pass，official held20未被读取；joint credit graph没有带来比v3更强的
全局behavior identifiability。

随后对固定v3/v4 checkpoint的六个Program blocks做只读geometry审计。v4 full Program off-diagonal cosine均值/标准差约
`.965/.020`，teacher behavior kernel为`.145/.316`；有限变化主要在language/scene，dynamic/process geometry基本没有改写。
原loss对每个mini-batch分别双中心化并按Frobenius norm单位化，因而对batch-local affine变换近似不敏感，也允许near-collapse
Program在中心化后取得局部相关。该证据把最早接口从pair graph推进到监督坐标的global calibration，不支持解冻Stage0、续训v4或
做普通超参扫描。

v5不增加参数或模型路径，确定性使用`(1+K_behavior)/2`作为raw Program Gram target，并按完整joint/meta/target fit scope的
固定per-owner teacher dispersion缩放off-diagonal误差与cross-view误差。v4 config被v5替换；数据、5+5 role权重、两组video、
Program schema、initialization、旧动态Gate、internal15/official20边界、纯Native Stage0与Action Meta 0均保持。dirty三卡真实一步为
`18.35s`、peak `9.98GB`，behavior/Program梯度`1.7323/2.7450`，初始Program/teacher std `.0141/.1478`；该profile只证明
执行与梯度，不是Gate。formal若仍无数量级改善，当前约定停止新增G2版本并整理专家复核材料。

关键artifacts：

- `runs/outputs/pi05_ecp_natural_program_g2_joint_behavior_kernel_fold0_m5_37885a6_gpu01p01234_r5_20260829/`；
- `runs/analysis/pi05_ecp_g2_behavior_block_geometry_init_v3_v4_20260829.json`；
- `runs/outputs/pi05_ecp_g2_global_kernel_v5_profile2_dirty_gpu01p012_r3_20260829/`。

## 51. G2-B global-calibrated v5 formal non-pass与专家复核暂停点

clean pushed detached `main@7f4df1b`的v5在gpu01物理`0--4`完成macro5/15 optimizer updates、唯一checkpoint及全部预注册
internal Gate，formal进程自然结束。source policy与Native Stage0 trainable为0，Action Meta argument为null且module/parameter为0；
official held20没有被读取。旧动态职责全部通过：full相对endpoints改善`20.860189%`、median active events 3、one-event 0，
same-task/probe/K1/K4/tau checks均成立。

新增behavior Gate则明确non-pass。train60 role-equal topology A/B为`.216014/.220817`，meta fit为
`.190522/.189068`；internal meta为`.202199/.216887`，四个internal target的`.750762/.766971`继续复现旧Program已有
偶然高值。fit60-only exact rank4 reader对panel-B/consensus role-equal仅`.105413/.128861`，四family panel-B为
q`.138678`、v`.089924`、action-in`.160061`、action-out`.078307`；wrong-Program margin overall`-.046567`。
同一internal basis的top16 span oracle仍为panel-B`.618395`、consensus`.715843`，所以资格目标本身没有失去容量。

冻结block geometry给出了v5机制实际发生了什么。相对v4，full Program off-diagonal cosine均值/标准差从约`.96484/.01978`
移到`.92578/.04590`，process从`.89844/.08643`移到`.75000/.21973`；两view cross-correlation分别`.97026/.99427`。
也就是说固定raw target确实扩大了task spread并保持跨video稳定。然而full/process对teacher consensus的平均相关从
`.14972/.13494`降到`.14242/.13111`。训练中固定lifted-teacher std约`.14780`，Program joint std在macro5仍仅约
`.02954/.03111`；behavior alignment宏平均在macro4--5为`12.61899/12.61955`，梯度有限非零但已经平台化。v5学到的是
“把tasks区分开”，不是“依照真实policy效果区分”。

同口径v3/v4/v5的train A/B依次为`.2315/.2358`、`.2360/.2362`、`.2160/.2208`；exact panel-B依次
`.1207/.1129/.1054`，wrong margin始终为负。按预注册停止条件及owner本轮要求，v5不延长、不读official held20、不新增v6，
G3 P2继续暂停。该阶段只淘汰pointwise decoder和v3--v5直接Program behavior-credit实现；Stage0、Program schema、G1/native bank及
current-bank primal-to-dual operator没有被这一负结果整体否定。仓库在完整证据固化后停于专家复核点。

关键artifacts：

- `runs/outputs/pi05_ecp_natural_program_g2_global_behavior_kernel_fold0_m5_2d859f0_gpu01p01234_r5_20260829/`；
- `runs/analysis/pi05_ecp_g2_behavior_block_geometry_v5_2d859f0_gpu01p01234_r5_20260829.json`；
- `runs/analysis/pi05_ecp_g2_behavior_block_geometry_init_v3_v4_20260829.json`。

## 52. 第四次专家复核与J2 joint Program--primal functional裁决

第四位专家锁定远程`main@910fb204e8e3a5374ec988aa5e1da5bc042754aa`及`9b52e59..910fb20`的25个提交，重新审计S1/S2、
Program-primal/current-bank dual、P0/P1/P2、95-task behavior authority及G2-B pointwise/v3--v5。1075行原始回复逐字保存为
`docs/expert_review_20260829_joint_program_primal.md`。

专家确认P0/P1已把native bank、rank4、current-bank dual和exact signed replay从当前首因中排除；V5则是合法protocol non-pass，但
不能证明Program schema或Stage0结构性失败，因为它只有15次optimizer updates、前9次处于warmup，且监督的是固定block-equal、单位化、
等权Program Gram，而不是Program经primal、真实bank和唯一rank16 LoRA后的functional effect。V3--V5只淘汰各自的独立Program
behavior-geometry目标。

owner采纳主裁决：取消独立Program behavior-Gram硬Gate，当前唯一活动资格改为joint Natural Program--shared primal functional
training。generated完整38-target rank16 LoRA在cross-episode teacher action/flow panel上的真实功能损失同时给Program与primal scorer
credit；source、Native Stage0、已通过P1的current-bank operator、carrier、scale与Action Meta冻结。首个决定性实验固定10个gradient
tasks与两个true task-held，先做同loss的task-local跨video共享free-primal正控，再按train/held-video/task-held、四family、wrong
Program/bank、interaction、same-task和相邻checkpoint形成Gate。只有train与held-video强而task-held弱时，才进入matched raw frozen
Stage0 sufficiency probe；不恢复v6、旧P2 frozen-Program formal、realizer或functional-polar deployment。

## 53. J2 task-local functional positive control formal通过

clean pushed detached `main@f3677a5`在gpu01完成10个gradient tasks各100次task-local functional updates。每task只训练跨两条fit K1
videos共享的free primal；第三held video、panel B和held teacher零梯度，Action Meta 0，source/Stage0/Program/shared scorer/scale冻结，
实际生成唯一38-target rank16。held/fit functional benefit retention median/min为`1.0144/.8896`，held factor recovery median
`.8078`，四family medians为`.7973/.7722/.8436/.8481`；10/10 held panel-B经paired t与sign test均显著优于carrier。

formal physical microbatch4的评价/训练最大墙钟比`.1957`，但最长task93峰值`37.07GiB`违反`<35GiB`系统合同。保持logical16、keyed
randomness、banks与loss不变，仅将physical microbatch改为2的task93真实一步将peak降到`32.41GiB`，step1 loss相对差`.060%`、
step`13.32s`；active runtime据此锁定physical2，无需重复10-task科学formal。该阶段排除functional panel、scale与free-primal可达性
为当前首因，但没有检验shared Program mapping；下一步仍是六卡joint profile与12-task joint Gate。

关键artifacts：

- `runs/outputs/pi05_ecp_j2_functional_positive_control_10task_c4704cb_gpu01p012345_20260829/`；
- `runs/analysis/pi05_ecp_j2_pc_task93_mb2_memory_profile_f3677a5_gpu01p3_20260829/`。

## 54. J2 joint Program--primal formal non-pass与task-specific routing根因

clean detached `5fd80b6`在gpu01六卡完成10 warmup+100 effective joint updates，step70/110均由修正后的独立sealed video authority
执行完整12-task Gate。train recovery为`.159588/.170800`，held-video为`.148662/.164623`；step110 task-held task2/task74为
`.122798/-.109179`，q/v/action-in/action-out medians为`-.001007/.001435/.008965/-.000434`，full相对language/endpoints
`.086578/.033435`，wrong Program/bank margins`.008033/.007142`、interaction`.002387`。held/train、same-task retention、event、
K1与信息墙通过，但primary和全部task-specific causal指标明确non-pass；相邻checkpoint仅小幅改善，不支持继续同一训练。

训练日志与checkpoint110零step审计排除clip和断图。Program/primal task-pair cosine median约`.93--.95`，生成update仍高度公共；成功
free-primal input/output code task-pair median仅`.203/.149`。真实functional task-gradient median cosine`-.0229`、`62.22%` pairs
为负，六task组cancellation ratio`.4208--.5360`，且各Program/scorer有效组均有有限非零梯度。故最早接口是近同条件表示下的
task-specific functional routing：纯correct-pair loss允许common residual shortcut。后继只获得一次配对counterfactual functional
qualification，训练时交替wrong Program与wrong bank；若仍无train数量级提升，则停止该credit修正而重开函数类/representation，不把
问题错误上移到只适用于“train高、task-held低”分支的raw Stage0。

关键artifacts：

- `runs/outputs/pi05_ecp_j2_joint_program_primal_12task_s110_1d775a4_gpu01p012345_r6_20260829/`；
- `runs/outputs/pi05_ecp_j2_joint_gate_step110_2cd4091_gpu01p012345_w6_20260829/`；
- `runs/analysis/pi05_ecp_j2_checkpoint110_gradient_geometry_2cd4091_gpu01p5_20260829/report.json`。

## 55. J3 counterfactual functional routing formal non-pass

clean detached `f8bfb7a`在gpu01六卡完成10 warmup+100 effective J3 updates，step70/110随后由clean detached `3f6f94e`的六个
独立worker完成paired Gate。训练counterfactual gap持续增大且全部Program/scorer梯度有限非零，证明配对分支不是no-op；但step70/110
correct train recovery仅`.136913/.148649`、held-video`.131572/.147689`，都没有超过J2。step110 q/v/action-in/action-out为
`.000466/-.004513/.008217/-.001500`，wrong Program/bank margins`.010192/.012540`、interaction`.005426`，核心Gate全部non-pass。

逐task比较显示correct fit只有task52/72/75三项优于J2，而wrong Program、wrong bank和endpoints controls分别有8/10、7/10、9/10改善。
因此J3学到的主要是让错误组合变坏，而不是提高正确Program+bank的task-specific方向；少数outlier抬高mean，median仍远离`.10`。
该结果淘汰当前pairwise cyclic negative hinge作为充分routing credit，并触发active design中“train低于`.40`且routing未展开时停止继续
contrastive/normalization/optimizer技巧、重开representation/function class”的分支；它不淘汰P0/P1已通过的native bank/operator/rank4，
也不触发只适用于train/held-video已高而task-held低的raw Stage0 probe。

关键artifacts：

- `runs/outputs/pi05_ecp_j3_counterfactual_program_primal_12task_s110_9af7c19_gpu01p012345_r6_20260829/`；
- `runs/outputs/pi05_ecp_j3_counterfactual_gate_step70_3f6f94e_gpu01p123456_w6_20260829/`；
- `runs/outputs/pi05_ecp_j3_counterfactual_gate_step110_3f6f94e_gpu01p123456_w6_20260829/`。

## 56. R1 orthogonal routing-token control partial/non-pass与R2 critic分解

clean detached `8c213c5`的R1以10个固定、非参数化正交token替代Natural Program内容，只训练原
`ProgramNativePrimalScorer`。step110 train/held-video recovery为`.267809/.279828`，held/train`1.044879`、wrong-token margin
`.238352`、same-task retention`.990982`；相对J2 fit中位提升约57%且9/10 tasks改善，证明清晰task route有真实因果作用。但
q/v/action-in/action-out仅`.003698/.007820/.001111/.033335`，完整Gate non-pass，固定token也不构成deployment或G3通过。

后续只读几何显示scorer内部task hidden未坍缩，q/v能以8--9/10检索正确task-local code，却只产生极弱的coupled primal alignment；
q/action-in grouped-output在固定hidden的最优128D last-head拟合上限约`.658/.363`。同时J2 task-local正控由teacher consensus初始化，
首步已获得最终functional benefit的中位约`.431`，所以正控没有检验从随机方向发现解。当前根因被细分为：Natural Program route不清晰、
functional-only direction discovery credit不足，以及q/action-in grouped-output函数类限制，而不是bank/operator/rank4、video泛化、断图或
普通训练长度。

R2保持R1真实banks、current-bank dual/exact signed replay、scorer、functional primary、rank12+4与Gate不变，只给gradient-task fit views
加入已有fit-only set-valued paired-update critic。weight1真实六卡一步为`15.094s`、最大reserved`20.29GiB`，联合gradient norm`.1201`；
相同初始化functional-only为`.02575`，故formal一次性固定critic weight`.2`。R2仍是training-only边界对照，held/panel B/task2/74/
validation/test零梯度；结果只用于裁决下一canonical Natural Program shared compiler是否需要稠密privileged critic或更换primal decoder。

关键artifacts：

- `runs/outputs/pi05_ecp_routing_token_control_r1_s110_ec86fdb_gpu01p123456_r6_20260829/`；
- `runs/outputs/pi05_ecp_routing_token_control_r1_gate_step110_8c213c5_gpu01p123456_w6_20260829/`；
- `runs/analysis/pi05_ecp_routing_critic_r2_weight1_profile_dirty_gpu01p013456_r6_20260829/`。

## 57. R2 set-valued critic formal non-pass与R3 grouped-output裁决

clean detached `a4b91bb`的R2完成10 warmup+100 effective updates及step70/110完整Gate。step110 train/held-video recovery为
`.205796/.193603`，低于R1的`.267809/.279828`；held/train`.940749`、same-task retention`.978070`，wrong-token margin
`.090559`。与此同时q/v/action-in/action-out family recovery从R1近零提高到`.220453/.407617/.166808/.663453`，训练critic recovery
最后20步稳定在约`.322`。故critic真实进入并恢复了v/action-out，但未形成四family完整功能解；结果不能用续训或weight小扫修饰。

固定R2 hidden对10个成功free-primal code做FP64 least-squares反事实：当前共享group output head的q/action-in median/min上限为
`.690917/.654797`与`.391981/.325906`，v/action-out为`1.0`；每个owner×native group使用独立head后，四family median/min均为
`1.0/1.0`，每组hidden rank为`40/40`。因此R3只移除该错误参数共享，新增约426万参数，不改Program、bank、critic、rank、scale、
数据或Gate。R2 evaluator另暴露旧frame-count cost proxy把长任务两两配对，单checkpoint长尾约`400s`；R3只用R2 task wall time
预注册timing cost map，预计最长worker约`303s`，该调度不读取科学结果。

关键artifacts：

- `runs/outputs/pi05_ecp_routing_token_critic_r2_s110_6c41926_gpu01p013456_r6_20260829/`；
- `runs/outputs/pi05_ecp_routing_token_critic_r2_gate_step70_a4b91bb_gpu01_w6_20260829/`；
- `runs/outputs/pi05_ecp_routing_token_critic_r2_gate_step110_a4b91bb_gpu01_w6_20260829/`。

## 58. R3 grouped-output formal non-pass与functional-code初始化资格

clean detached `main@19dbf0e`完成R3 10 warmup+100 effective updates及step70/110 Gate。step110 train/held-video recovery为
`.305293/.287486`，q/v/action-in/action-out为`.285260/.277323/.656235/.668922`，wrong-token margin`.118578`、same-task
retention`.986958`。因此owner×group decoder明确修复action-in并维持action-out，但q/v与primary仍non-pass；内部family提升不能替代
真实functional。

随后六task只读梯度审计显示fit-only critic相对真实functional gradient的全局cosine median`-.148903`，q为`-.269630`；成功
task-local code监督梯度相对functional也仅`.032449`。同一成功方向只换R3冻结shared scale的真实policy recovery中位`.939783`、
范围`.753420--1.023163`，6/6均过`.60`。这排除了继续调critic/family权重或解冻scale，支持一次fixed-route、training-only
functional-code head初始化对照。

R4把十个formal成功task-local primal按fixed-token下现有hidden做FP64 minimum-norm owner×group head插值，不读取其task-local scale；
初始化后critic删除，只训练真实functional loss。dirty gpu01六task真实step0显示233 heads均rank`40/40`，最大FP64/FP32 fit error
`1.50e-14/.003992`，相对原positive code recovery为`.980/1.155/1.012/.944/.754/.815`，Action Meta 0、native teacher reads 0、
唯一rank16及全部冻结边界成立。该profile只给予clean formal资格，不是R4 Gate或G3通过。

关键artifacts：

- `runs/outputs/pi05_ecp_routing_grouped_decoder_r3_s110_67a49f8_gpu01p016_r3_20260829/`；
- `runs/outputs/pi05_ecp_routing_grouped_decoder_r3_gate_step110_19dbf0e_gpu01_20260829/`；
- `runs/analysis/pi05_ecp_routing_grouped_decoder_r3_gradient_allocation_19dbf0e_gpu01_20260830/`；
- `runs/analysis/pi05_ecp_routing_grouped_decoder_r3_utility_code_gradients_19dbf0e_gpu01_20260830/`；
- `runs/analysis/pi05_ecp_routing_grouped_decoder_r3_shared_scale_transfer_19dbf0e_gpu01_20260830/`。

## 59. R4 functional-code initialization formal与moving-chart根因

clean detached `0b51c57`完成R4 10 warmup+100 effective updates及step70/110 paired Gate。step110 train/held-video recovery为
`.819437/.839139`，q/v/action-in/action-out为`.439578/.388131/.249310/.400750`，wrong-token margin`.913637`、same-task
retention`1.002751`。除action-in外全部主检查通过；step70到110 train median继续提高`.041888`，但相邻pass因两个checkpoint的
action-in都未过而为false。R4因此是明确formal non-pass，同时首次证明强fixed-route shared functional解可跨训练与held video稳定保留。

随后从同一deterministic initialization封存pre-step state，并对step70/110做head/chart/module graft。action-in raw outer recovery从
step0约`.999988`降至`.298330/.301140`；在当前checkpoint hidden上只重拟合33个action-in heads可恢复到约`1.0`，hidden仍full rank
`40/40`。initial到step110的all-head relative drift仅`.000810`，feature-chart drift`.008930`；initial heads接checkpoint chart仍
`.301099`，checkpoint heads接initial chart保留`.998320`。program/rank context与input/output trunk各自微小移动都可单独破坏初始化
坐标，故这是distributed moving-coordinate问题，不是head、scale、bank、训练长度或单模块bug。

下一R5只冻结初始化后的feature chart、训练全部233 native heads，保持数据、loss、optimizer、budget、Gate及信息墙不变。dirty单卡真实
step已证明`10,297,344` trainable、全部heads有限非零gradient、chart零gradient、Action Meta和native teacher reads为0、唯一rank16及
真实policy consumption成立；该profile不构成formal结果。

关键artifacts：

- `runs/outputs/pi05_ecp_routing_functional_code_init_r4_s110_69a6b24_gpu01p012345_r6_20260830/`；
- `runs/outputs/pi05_ecp_routing_functional_code_init_r4_gate_step110_0b51c57_gpu01p012456_w6_20260830/`；
- `runs/analysis/pi05_ecp_routing_functional_code_init_r4_initial_state_capture_0b51c57_gpu01p0_20260830/`；
- `runs/analysis/pi05_ecp_routing_functional_code_init_r4_action_in_code_audit_0b51c57_20260830/`；
- `runs/analysis/pi05_ecp_routing_functional_code_chart_frozen_r5_profile_dirty_gpu01p0_20260830/`。

## 60. R5 fixed feature-chart formal pass与Natural Program接回

clean detached `9e6b6a7`在gpu01物理`0,1,2,3,4,6`完成R5 10 warmup+100 effective updates。110条metrics连续，actual
step70/110 world6 checkpoints完整，训练墙钟`1503.63s`、最大peak reserved`32,916,897,792` bytes；Action Meta、source/
Stage0/scale trainable和native teacher reads均为0。六个独立worker随后在物理`0--5`完成同一paired Gate。

step70/110 train recovery为`.933583/.940336`，held-video为`.957202/.963277`；step110 q/v/action-in/action-out为
`.815834/.839439/.820583/.837113`，wrong-token margin`.895772`、same-task retention`1.006591`、held/train
`1.024396`。两点全部primary checks通过，step110相邻稳定性通过，R5正式pass。它只证明冻结feature chart可保留utility-aligned
shared heads与真实functional优化，不证明fixed routing token可部署或Natural Program mapping已成立。

基于该单变量证据，下一R6加载R5 step110共享scorer，移除fixed route，恢复`c1493a1/macro20` Natural Program；只训练Program
readers/fusion/aligner和233 native heads，完整feature chart继续冻结。训练只用两fit-video生成的唯一rank16在disjoint panel A上的
PI0.5 flow，不恢复J3 counterfactual或其它探索loss；完整12-task Gate仍是唯一资格。

关键artifacts：

- `runs/outputs/pi05_ecp_routing_functional_code_chart_frozen_r5_s110_9e6b6a7_gpu01p012346_r6_20260830/`；
- `runs/outputs/pi05_ecp_routing_functional_code_chart_frozen_r5_gate_step70_9e6b6a7_gpu01p012345_w6_20260830/`；
- `runs/outputs/pi05_ecp_routing_functional_code_chart_frozen_r5_gate_step110_9e6b6a7_gpu01p012345_w6_20260830/`。

## 61. R6 Natural Program接回non-pass与content-chart根因

clean detached `1a6a59b`在gpu01物理`0,1,2,3,4,6`完成R6 10 warmup+100 effective updates，step70/110随后执行完整
12-task paired Gate。step70/110 train recovery为`.145063/.165181`，held-video为`.138406/.143114`，task-held mean为
`.012597/-.034333`；step110 q/v/action-in/action-out为`.038887/-.025071/-.007718/.319160`，wrong Program/bank margin
`.077886/.001131`、interaction`.000018`。两点均明确non-pass，R5 shared scorer没有因接回Natural Program而形成G3。

同一R5 heads的只读对照把fixed token functional-code cosine测为`.998514`，G2 Natural Program仅`.010736`；R6最终Program通过
R5/R6 heads分别只有`.020074/.020914`。同task三video在共同mapping下输出却约`.9994`稳定，排除普通video variance。两fit-video
minimum-norm head solve可精确插值80/80行，但第三held view仅`.353777`，task2/74约零。历史结论因此收窄为：R5建立的是无Natural
Program内容几何的fixed-token utility chart；R6 functional-only credit未能完成Program到该chart的获取，继续训练或简单head refit没有依据。

下一活动R7冻结R5通过的native heads，只训练Natural Program和共享feature chart，以gradient-task validated positive-control
outer-update directions作fit-only training labels。它不读held/action reward作为deployment输入，仍须由原12-task真实bank/唯一rank16
functional Gate裁决，不能以内部分数替代。

关键artifacts：

- `runs/outputs/pi05_ecp_natural_program_chart_reconnect_r6_s110_1a6a59b_gpu01p012346_r6_20260830/`；
- `runs/outputs/pi05_ecp_natural_program_chart_reconnect_r6_gate_step110_1a6a59b_gpu01p01246_w5_20260830/`；
- `runs/analysis/pi05_ecp_r6_program_chart_alignment_v2_49ec54b_gpu01p0_20260830/result.json`。

## 62. R7冻结chart acquisition formal non-pass

clean detached `024fc55`在gpu01物理`0,1,2,3,4,6`完成R7 10 warmup+100 effective updates；110条metrics连续，
step70/110 world6 checkpoints完整，训练墙钟`114.12s`，最大peak reserved `19,367,198,720` bytes。Action Meta、source、
Stage0、scale trainable与native teacher reads均为0。六个独立worker随后在同六卡依次完成两个checkpoint的12-task paired Gate。

step70/110 train recovery为`-.183186/-.133386`，held-video为`-.177017/-.129792`；step110 task2/74为
`.554032/-.576694`，q/v/action-in/action-out内部outer direction为`.725096/.737788/.743253/.642195`，wrong Program/bank
margin为`.091741/-.003368`，interaction`.001064`。两个checkpoint均primary non-pass；短训练导致的evaluation/training wall
ratio失败只是附加系统项，不改变科学结论。

R7证明dense functional-code direction supervision仍不能让Natural Program适配冻结R5的任意feature chart：内部方向中等偏高却不够
精确，target role闭环统一为负，且训练已平台；它没有反证validated code、bank、rank4、shared scale或exact replay。下一修正保持
同一labels、数据、预算和真实Gate，只取消冻结chart约束，让Program与完整primal scorer在绝对outer-update target下联合取得坐标；
先做短程真实fit/gradient资格，不以内部loss替代闭环。

关键artifacts：

- `runs/outputs/pi05_ecp_functional_code_chart_acquisition_r7_s110_89131fe_gpu01p012346_r6_20260830/`；
- `runs/outputs/pi05_ecp_functional_code_chart_acquisition_r7_gate_step70_024fc55_gpu01p012346_w6_20260830/`；
- `runs/outputs/pi05_ecp_functional_code_chart_acquisition_r7_gate_step110_024fc55_gpu01p012346_w6_20260830/`。

## 63. R9稳定chart联合获取与code-to-utility non-pass

clean pushed detached `7b614da`从R5 passed shared chart初始化scorer、从G2初始化Natural Program，随后用10个gradient tasks的固定
outer-update labels联合训练全部Program/scorer参数。110条metrics连续，step70/110 loss为`.354164/.334220`，两个world6 checkpoint
完整；Action Meta、source、Stage0与scale trainable均为0。相同实现的直接task-held诊断曾得到`.6404`中位outer方向，因此获得原
12-task functional Gate资格。

六个独立worker完整评价step70/110。step110 train/held-video/true-task-held functional recovery为
`-.131825/-.129718/-.011724`，task2/74为`.561846/-.585295`，五个target-role gradient tasks全部为负；q/v/action-in/action-out
outer medians却为`.728694/.744085/.745741/.642526`。full-over-endpoints、wrong-bank与interaction分别为
`-.082736/-.003829/.001946`，故primary与adjacent Gate明确non-pass。信息墙、K1、event、唯一rank16、worker/checkpoint覆盖均通过；
evaluation/training wall ratio失败是附加系统项，不是科学失败原因。

该轮同时保留两条结论：稳定chart初始化解决了fresh joint的部分可优化性，但outer-code相似度不足以表示真实policy utility。后继不续训
outer loss、不触发raw Stage0，也不重开bank/operator/rank/scale；只把R9 model tensors作为training-only初始化，按R4/R5坐标漂移证据
冻结feature chart，再用真实cross-episode functional flow细化Natural Program与native heads。

关键artifacts：

- `runs/outputs/pi05_ecp_functional_code_stable_chart_joint_r9_s110_43be484_gpu01p012346_r6_20260830/`；
- `runs/outputs/pi05_ecp_functional_code_stable_chart_joint_r9_gate_step70_43be484_gpu01p012346_w6_20260830/`；
- `runs/outputs/pi05_ecp_functional_code_stable_chart_joint_r9_gate_step110_43be484_gpu01p012346_w6_20260830/`。

## 64. R10真实functional refinement取得数量级跃升但task-held与bank interaction non-pass

clean pushed detached `f3baa81`从R9 step110完整Writer model tensors初始化，冻结feature chart，移除outer-code loss，只训练Natural
Program与233 native heads。gpu01物理`0,1,2,3,4,6`完成10 warmup+100 effective updates，110条metrics连续、step70/110 world6
checkpoints完整，训练墙钟`1442.70s`，最大peak reserved `32,937,869,312` bytes；Action Meta、source、Stage0、operator、scale
trainable和native teacher reads均为0。六个独立worker随后完成两个checkpoint的12-task paired panel-B与全部controls。

step70/110 train recovery为`.532227/.559896`，held-video为`.500728/.544189`，held/train为`.940817/.971946`；step110
q/v/action-in/action-out为`.645745/.614858/.717575/.548006`，same-task retention`.990228`、wrong-Program margin`.279494`。
这相对R9负值是明确功能跃升，证明稳定content initialization经真实flow refinement可转化为policy utility。两个checkpoint仍primary
non-pass：step110 task-held mean`.151475`（task2 `.375386`、task74 `-.072436`），wrong-bank`.007864`、interaction`-.002683`、
full-over-endpoints`.061382`，train也低于`.60`。

per-task结果显示meta gradient tasks总体强而target gradient tasks仅`.186--.499`；同一target tasks在R5 fixed-route正控均强，故剩余
最早接口是跨target-task的共享表示/generalization及Program--current-bank配对，而不是bank/operator/rank/scale。按专家预注册结果解释，
下一步是同scorer、loss、预算与Gate的raw frozen Stage0 sufficiency probe；它只作根因诊断，不登记为deployment Writer，也不授权在结果前
改Program schema或解冻Stage0。

关键artifacts：

- `runs/outputs/pi05_ecp_r9_initialized_functional_refinement_r10_s110_731a769_gpu01p012346_r6_20260830/`；
- `runs/outputs/pi05_ecp_r9_initialized_functional_refinement_r10_gate_step70_f3baa81_gpu01p012346_w6_20260830/`；
- `runs/outputs/pi05_ecp_r9_initialized_functional_refinement_r10_gate_step110_f3baa81_gpu01p012346_w6_20260830/`。

## 65. R11 matched raw-Stage0 sufficiency明确non-pass

clean pushed detached `25f38ce`保持R10除上游表示外的全部合同，只用exact-language embedding、scene transition、raw
owner/event process、presence、uncertainty和normalized time构造同shape输入。gpu01物理`0,1,2,3,4,6`完成10 warmup+100 effective
updates；110条metrics连续，step70/110 world6 checkpoints完整，训练墙钟`1435.70s`，最大reserved `32.94GB`。随后六个独立worker
完成两个checkpoint的12-task paired panel-B、四family与全部controls；Action Meta、native teacher reads、held backward和
shuffled/reversed use均为0。

step70/110 train recovery为`.218691/.292321`，held-video为`.232166/.288053`，true task-held为`-.139011/-.092369`；step110
meta/target gradient medians为`.470816/.110012`，task2/74为`.116054/-.300793`。q/v/action-in/action-out为
`.550257/.101550/.494693/.474379`，wrong-Program、wrong-bank和interaction为`.077357/-.003253/-.001410`。raw task-held相对
R10不增反降`.243844`，故Program schema/压缩首因判据明确失败；同时q/action仍可读、失效集中在v与target role，不能据此停止整个
frozen Stage0或解冻VLM/source/全Stage0。

关键artifacts：

- `runs/outputs/pi05_ecp_raw_stage0_sufficiency_r11_s110_0590f63_gpu01p012346_r6_20260830/`；
- `runs/outputs/pi05_ecp_raw_stage0_sufficiency_r11_gate_step70_25f38ce_gpu01p012346_w6_20260830/`；
- `runs/outputs/pi05_ecp_raw_stage0_sufficiency_r11_gate_step110_25f38ce_gpu01p012346_w6_20260830/`。

## 66. R5成功primal的cross-bank functional upper bound

clean pushed detached `2090799`复用R5 step110通过Gate的10个task-local成功primals，对每个task保持原panel B、carrier、scale、
policy RNG和完整rank16，只把K1 native bank换为同role cyclic错误task。六个独立workers完成全部10 task；无backward、Action Meta、
teacher tensor或shuffled/reversed使用。

正确bank recovery中位`.930860`，错误bank中位`.945799`，correct-minus-wrong中位`-.003819`；正确bank仅`2/10`更好、`0/10`
达到`.10` margin，错误bank`10/10`保持正收益，收益保留中位`1.003960`。因此P0/P1的capacity结论仍成立，但当前global covariance
`C^+d`接口不能建立wrong-bank可识别性；R10/R11的bank因果失败不能继续归因Program/scorer。后续先修operator-level bank interaction
并通过task-local正控，再恢复shared G3。

关键artifact：

- `runs/analysis/pi05_ecp_g3_r5_successful_primal_cross_bank_2090799_gpu01p012346_w6_20260830/`。

## 67. symmetric half operator的fit-only cross-bank bridge diagnostic

clean detached诊断先把R5旧primal直接用于`C_B^{-1/2}` replay，correct recovery中位降至`.076821`，排除“只换inverse幂次即可”这一
错误解释。随后每task只用两条fit bank分别对teacher-initialized primal做inverse-square-root transport并平均，第三条same-task held
video及same-role wrong bank不参与构造并保持零梯度。10-task held correct/wrong recovery中位为`.647543/.134170`，margin中位
`.480161`；正确bank在`10/10` task更好且全部达到`.10` margin，但correct中位未达到预注册`.75`，因此总体non-pass。

这条结果证明half operator可保留真实X/Y bank的功能特异性，却没有证明未经优化的fit-transport code具有足够same-task held capacity，
更没有证明shared Program mapping。后续只授权固定half operator下的task-local fit-only functional capacity control。

关键artifacts：

- `runs/analysis/pi05_ecp_g3_symmetric_whitening_cross_bank_3cd4ecf_gpu01p012346_w6_20260830/`；
- `runs/analysis/pi05_ecp_g3_fit_transport_half_cross_bank_3297dbd_gpu01p0_w6_20260830/`。

## 68. half-operator task-local functional formal与bank-interaction Gate

clean pushed detached `55fded4`在gpu01物理`0,1,2`及gpu02物理`4,7`完成10-task task-local formal。每task仅以两条fit K1
videos和panel A真实functional flow优化一个shared-across-video code 100步；全部sealed checkpoint为`321,792`个task-local参数，
Writer/source/Stage0/Action Meta均为0，held/wrong/panel-B backward为0，输出仍是唯一完整rank16。五个独立worker随后在第三条
same-task held video与same-role wrong bank上完成panel B评价。

held correct/wrong recovery中位为`.725204/.188873`，margin中位`.541238`，正确bank在`10/10` task更好且`10/10`达到
`.10` margin。bank interaction已强过门，但correct未达到预注册`.75`，所以总Gate non-pass。fit-video recovery中位为`.950541`；
meta fit/held为`.997452/.898189`，target为`.796767/.614878`，而旧full inverse target correct中位仍为`.945032`。
该分解把最早接口锁定为half operator的same-task fit-to-held谱坐标转移：不是bank没有因果增量，也不是单纯延长task-local优化可修复。

下一只读bridge预注册唯一inverse power`.75`与补幂fit transport`.25`，对应在replay中保留current-bank quarter-root作用；不做其它
幂次小扫。若同一10-task原Gate不能同时恢复correct capacity与margin，则停止继续调谱幂并转向common-coordinate/operator结构。

关键artifacts：

- `runs/outputs/pi05_ecp_g3_bank_interaction_positive_control_10task_89b130a_gpu01p012_gpu02p47_20260830/`；
- `runs/analysis/pi05_ecp_g3_bank_interaction_positive_control_gate_55fded4_gpu01p012_gpu02p47_w5_20260830/`。

## 69. tempered `.75` fit-transport cross-bank bridge

clean pushed detached `db88418`使用R5成功primal、两条fit video的`C_B^{-1/4}` transport和held bank的
`C_B^{-3/4}` replay，在同一10 tasks上完成唯一预注册tempered bridge。没有backward、Action Meta、teacher tensor、
shuffled/reversed或checkpoint选择。

held correct/wrong recovery中位为`.925312/.885043`，correct-minus-wrong中位`.054500`；correct bank在`8/10`
更好，但margin达`.10`仅`2/10`，wrong bank `10/10`仍有正效用。结果严格non-pass，并与half/full端点一起终止
谱幂调参：单一幂次无法同时保留correct capacity与wrong-bank specificity。

关键artifact：

- `runs/analysis/pi05_ecp_g3_fit_transport_tempered_cross_bank_db88418_gpu01p012_gpu02p47_w5_20260830/`。

## 70. full-inverse normalized-energy审计与operator-level正控

clean pushed `2551f7c`基于R5 step110成功primals审计10 tasks、30个same-task video pairs与50个same-role wrong-bank pairs。
普通raw dual energy不能区分正确bank；gauge-free input retained projection p10却达到AUC `1.0`、逐task严格分离`10/10`，全局正确
minimum `.907248`高于错误maximum `.905998`。input sorted projection第12--20位均值同样AUC `1.0`，成为后续训练的平滑低分位
calibration统计。该值只由当前primal与当前bank计算，不使用成对比值、task ID或文件名。

三项固定因果检查随后区分了信号的作用位置。support只缩放最终rank4 residual时correct/wrong/margin为
`.954661/.930365/.031766`，不能建立必要性；用阈值`.906622976064682`在full/half inverse query之间hard switch时为
`.950915/.005173/.908899`，correct-better与margin达`.10`均为`10/10`；按sigmoid在线性空间混合两套query时correct降到
`.238736`，明确non-pass。由此终止幅值gate、soft query interpolation、temperature与谱幂小扫，保留near-binary content route。

关键artifacts：

- `runs/analysis/pi05_ecp_g3_normalized_energy_grid_2551f7c_gpu01p012_gpu02p47_w5_20260830/`；
- `runs/analysis/pi05_ecp_g3_projection_energy_gate_cross_bank_2551f7c_gpu01p012_gpu02p47_w5_20260830/`；
- `runs/analysis/pi05_ecp_g3_projection_switched_operator_cross_bank_2551f7c_gpu01p012_gpu02p47_w5_20260830/`；
- `runs/analysis/pi05_ecp_g3_soft_operator_mixture_cross_bank_2551f7c_gpu01p012_gpu02p47_w5_20260830/`。

## 71. R10 hard-route compatibility qualification诊断

同一固定p10阈值被原样用于R10 step70/110两个sealed checkpoints的完整12-task panel-B Gate；五个独立workers完成所有原qualification
arms。step70 train/held/task-held为`-.455230/-.556757/-.401290`，step110为`-.482993/-.631937/-.533894`，两个checkpoint均
strict non-pass。step110 q/v/action-in/action-out为`.079345/.127917/.495778/.190120`，correct-wrong-bank margin
`-.087550`、interaction`.071561`。

根因由support rows直接给出。step70/110 matched-vs-mismatched AUC只有`.551215/.558160`，逐task严格分离均为`0/12`；step110
matched/mismatched中位为`.867304/.864599`，大多数正确pair被送入half端点。选中full的正确conditions与原R10结果吻合，证明hard
operator正控没有实现错误；R10 Natural Program primals本身未学到R5成功primals已存在的bank兼容几何。后续不再把固定门直接套在
R10上，而从R10 step110 model tensors与fresh optimizer启动shared compatibility qualification：cross-video same-task positive、
same-role cyclic negative的projection calibration与原correct functional loss共同训练；held、task2/74和panel B零梯度，Action Meta 0。

关键artifacts：

- `runs/analysis/pi05_ecp_g3_r10_hard_operator_switch_gate_step70_2551f7c_gpu01p012_gpu02p7_w5_20260830/`；
- `runs/analysis/pi05_ecp_g3_r10_hard_operator_switch_gate_step110_2551f7c_gpu01p012_gpu02p7_w5_20260830/`。

## 72. R12 shared Program--bank compatibility formal

clean pushed detached `fdab4ae`从R10 step110 functional tensors与fresh optimizer启动，用same-task cross-video positive、same-role
cyclic negative训练共享retained-projection calibration，同时在正确训练分支teacher-force full operator保护functional direction。
10 warmup + 100 effective updates连续完成，保存step70/110两个完整checkpoints；Action Meta/source/Stage0/scale冻结，仍只生成一套
carrier12+residual4 rank16。

两个checkpoint均完成六个独立worker的12-task paired Gate并strict non-pass。step70/110 matched full-route fraction为
`.444444/.527778`，mismatched均为`.083333`；step110 paired support margin中位`.021072`，correct-wrong-bank与interaction已达
`.145007/.578436`，但train/held/task-held只有`.298505/-.504329/-.129071`。正确video按实际route分组后，full的functional recovery
中位`.583340`，half为`-1.092634`；说明operator正控仍有效，失败来自正确bank召回不足。task52/72及held task74还存在correct support
低于wrong-bank support的排序冲突，不能通过移动全局阈值修复。

该结果只淘汰“同一functional primal兼任compatibility classifier”的R12实现，不淘汰Natural Program、真实X/Y、signed pooling、
rank4或hard operator正控。随后登记的独立Program-conditioned compatibility probe只作credit-ownership诊断：保持functional primal
唯一负责LoRA residual，检查support与held-task泛化，但不把full/half二值坐标选择登记为最终G3架构。

关键artifacts：

- `runs/outputs/pi05_ecp_bank_compatibility_r12_s70_fdab4ae_gpu01p012_r3_20260830/`；
- `runs/outputs/pi05_ecp_bank_compatibility_r12_gate_step70_fdab4ae_gpu01p012_gpu02p457_w6_20260830/`；
- `runs/outputs/pi05_ecp_bank_compatibility_r12_gate_step110_fdab4ae_gpu01p012_gpu02p457_w6_20260830/`。

## 73. R13 decoupled compatibility probe有部分增量但不能泛化

clean pushed detached `0489da362508a199236583ad9f910c73a1dd5c5c`从R12 step110初始化38个独立compatibility input
heads，冻结Natural Program、functional input/output primals、scale、source、Native Stage0、carrier12与Action Meta。gpu01物理
0/1/2完成10 warmup + 100 effective updates，actual step70/110两个world3 checkpoints完整，训练墙钟`415.093s`；实际trainable
只有`4,853,760`个probe参数，native teacher reads为0。六个独立workers随后在gpu01物理0/1/2与gpu02物理4/5/7完成两个
checkpoint的12-task paired panel；全部自然exit 0，validation/test及shuffled/reversed均未读取。

step70/110均strict non-pass。train recovery为`.298505/.483082`，same-task held均为`.048744`，true task-held均为
`.032951`；step110 q/v/action-in/action-out为`.262289/.333634/.570474/.318540`。matched/mismatched full-route从
`.638889/.166667`变为`.666667/.166667`，support AUC从`.826389`变为`.831019`，逐task严格正确高于wrong均为`9/12`。
按角色拆分，step110 gradient fit、same-task held、true task-held正确放行为`16/20`、`5/10`、`3/6`。task52/72/74的
正确support minimum仍低于wrong bank；在wrong full-route不超过`.20`时，任何全局阈值最多放行`.722222`的正确pairs。

二值route对功能结果具有决定性且脆弱的影响。step110的36条正确conditions中，24条full recovery中位`.572070`且minimum
`.181790`，12条half中位`-.893770`且maximum`-.300972`。step70到110只新增task8 video6一条full route：support从`.906201`
升到`.906683`，仅越过固定阈值约`.000060`，却令task8 fit recovery从`-.207688`跳到`.963754`；其held video仍为
`-1.597779`，整体held与task-held逐值不变。correct-wrong-bank与interaction虽升至`.688182/.843922`，仍不能补偿正确视频的
误路由。

该结果证明把compatibility credit从functional primal拆开有真实但有限的帮助，因此R12的credit ownership冲突是一个问题；但独立
共享线性probe连部分gradient tasks都无法完整拟合，也没有跨video/task泛化，所以它不是充分解。R13只淘汰当前probe + binary
full/half route实现，不否定真实X/Y、signed pooling、rank4或强full方向；也不授权阈值、temperature、weight、LR、seed、谱幂或
同类probe容量小扫。owner已暂时关闭持续性goal，当前停在专家咨询节点；后继问题是Program与当前bank如何共同生成唯一functional
direction，而不是继续精修二值门卫。

关键artifacts：

- `runs/outputs/pi05_ecp_decoupled_compatibility_r13_s110_82607b6_gpu01p012_r3_20260830/`；
- `runs/outputs/pi05_ecp_decoupled_compatibility_r13_gate_step70_82607b6_gpu01p012_gpu02p457_w6_20260830/`；
- `runs/outputs/pi05_ecp_decoupled_compatibility_r13_gate_step110_82607b6_gpu01p012_gpu02p457_w6_20260830/`。

## 74. 第五次专家复核与Program--Bank候选级联合交互裁决

第五位专家锁定远程`main@b59d7bdd5fd7c2990c2f6e0eb28f170419ac7a84`及其可达历史，完整复核J2--R13、
R5 cross-bank与当前代码。1132行原始回复逐字保存为
`docs/expert_review_20260830_program_bank_interaction.md`。专家确认wrong-bank证据正确重开bank/operator接口，但指出full-inverse
`C_B^+d_P`把当前bank主要当作坐标系而非方向证据；R12/R13再以condition级support选择full/half，只会把小匹配误差放大成离散功能
跳变。R13已经充分终止当前binary门卫、support classifier、threshold及谱端点选择，但没有终止真实X/Y、signed pooling、rank4、
Natural Program或Native-Factor主线。

owner采纳唯一后继：保留full-inverse base query作为zero-init容量保持项，在exact signed pooling前加入Program event query、当前video
local event context与每个真实native candidate的共同交互，产生bounded、measure-centered的逐candidate positive/negative branch
logit correction。最终只有一套continuous signed measure、一个rank4 residual及carrier12+residual4的完整rank16，不输出route类别或
第二operator。首个资格固定R5通过的fixed route/chart/heads与其余authority，只训练interaction scorer，用correct functional flow与
bounded wrong-bank neutralization通过同一deployment forward裁决；通过后立即接回Natural Program并联合训练Program、interaction与
native heads。此节点只形成新的active contract，尚未产生代码、GPU smoke或formal结果。

## 75. candidate-level interaction首轮formal与balanced-credit修正

clean pushed detached `c7874f39fe927993adb3b67a3e8b9767892c608b`完成fixed-routing-token candidate interaction的110步训练、
step70→110 exact resume及两个checkpoint的十task完整Gate。step70/110 correct fit为`-.388363/-.386363`、same-task held为
`-.392916/-.393941`、unseen wrong-on为`-.405269/-.398702`、correct-minus-wrong为`-.018599/-.020456`，correct更好均为
`5/10`；interaction-off为`.940432`。Action Meta、deployment teacher reads及held/panel-B backward均为0，每个condition只物化一套
完整rank16。结果是明确scientific non-pass，并显示interaction对正确和错误bank施加了高度相关的共同破坏。

最早接口是专家原式在实际数值尺度下的量纲失衡：correct使用raw flow loss，而wrong hinge除以很小的free-primal benefit；代码对应
两条correct合计`1/6`与active wrong `-1/[6(B_free+eps)]`，后者被放大约`15.7--359.6x`。训练轨迹中wrong hinge迅速关闭、correct
loss恶化且无法恢复，global clip不能改变支配方向。因此该结果淘汰normalized-gradient loss实现，不淘汰candidate-interaction函数类。
下一次fresh资格只改变这一机制变量：wrong改为raw-unit固定`-1/6`，normalized recovery仅报告；其余architecture/data/seed/LR/steps/Gate
均保持。step110另有diagnostic teacher cache使物理read delta为0的aggregation记账问题，deployment信息墙本身未违反。

## 76. raw-unit 1:1 interaction formal与positive-anchor修正

clean pushed detached `cbe3124f1e48a1b0f51e8ab2faeba00e98bceebd`完成110个连续optimizer steps、step70→110 exact resume及
五worker两个checkpoint完整Gate。step70/110 correct fit为`.652284/.672942`、same-task held为`.642756/.663154`、unseen wrong-on
为`.346082/.345229`、correct-minus-wrong为`.189253/.185745`，correct优于wrong均为`9/10`；interaction-off保持`.940432`，
四family与信息墙通过。两个checkpoint稳定但strict non-pass，不能接回Natural Program。

本轮相对首轮负recovery取得大幅恢复，说明raw-unit修正确实去除了wrong量纲支配，也证明candidate interaction具有部分bank分离能力。
但1:1 objective仍有解析缺口：两条correct各`+1/12`、active wrong `-1/6`，使三臂共同恶化同一损失时目标严格不变。训练轨迹与此
一致，correct/wrong从接近`1.00/.98`共同下降，只产生约`.18` margin。下一fresh资格只把wrong降为`-1/12`，形成positive:negative
总质量2:1并惩罚共同破坏；架构、数据、seed、LR、步数、Gate与信息墙不变。该修正仍是fixed-route局部资格，不冒充shared G3。

关键artifacts：

- `runs/outputs/pi05_ecp_program_bank_candidate_interaction_v2_balanced_s110_6694e99_gpu01p012_r3_20260831/`；
- `runs/outputs/pi05_ecp_program_bank_candidate_interaction_v2_balanced_gate_step70_cbe3124_gpu01p012_gpu02p47_w5_20260831/`；
- `runs/outputs/pi05_ecp_program_bank_candidate_interaction_v2_balanced_gate_step110_cbe3124_gpu01p012_gpu02p47_w5_20260831/`。

## 77. positive-anchor formal、shared-correction审计与free-delta reachability

clean pushed detached `248d7689308ae96a959d44f0a8a7a1adbe3596b7`完成2:1 positive-anchor candidate interaction的110步训练、
step70→110 exact resume及五worker相邻Gate。step70/110 correct fit为`.922565/.929101`、same-task held为
`.931639/.953285`，但unseen wrong-on为`.932045/.934305`、correct-minus-wrong为`-.002346/-.005576`，正确bank更好只有
`5/10`和`4/10`。四family、absolute correct、same-task与信息墙通过；unseen wrong、margin、correct-better及wrong off-on失败。
因此positive anchor解决了此前共同破坏，却没有产生bank因果分离。

step110 full10逐层诊断显示wrong bank在input feature、learned LayerNorm、MLP correction与pooled update中并未不可区分；相应
separation-ratio median约为`3.105/3.299/3.142/4.306`。但shared correction gauge RMS约`1.5e-5`，base-score RMS约`.0202`，
pooling KL约`1e-8`，说明learned interaction在功能上近于关闭。同一十task、相同真实X/Y和signed-pooling operator上的task-local
free-delta反事实以absolute delta p95 median`.0019996`、零bound saturation把wrong panel-B recovery压到`-.5277`中位，10/10 task
均低于`.25`。这证明现有bound/operator能表达强bank suppression，但不证明shared Program mapping。

基于该最早失效接口，下一fresh候选只增加B1 base-score feature
`stop_gradient(q0·(value-global_B0_mean))/replay_score_rms`；input/output candidate索引、event assignment、candidate measure、真实
X/Y values、positive/negative signed pooling、loss、optimizer、data、rank、scale与Gate均不变。旧v1--v3 checkpoint/config保留为历史
证据但不再由active loader执行。

关键artifacts：

- `runs/outputs/pi05_ecp_program_bank_candidate_interaction_v3_anchor_s110_fd20251_gpu01p012_r3_20260831/`；
- `runs/outputs/pi05_ecp_program_bank_candidate_interaction_v3_anchor_gate_step70_248d768_gpu01p012_gpu02p47_w5_20260831/`；
- `runs/outputs/pi05_ecp_program_bank_candidate_interaction_v3_anchor_gate_step110_248d768_gpu01p012_gpu02p47_w5_20260831/`；
- `runs/analysis/pi05_ecp_program_bank_candidate_interaction_v3_anchor_bank_separation_full10_step110_248d768_gpu01p012_gpu02p47_w5_20260831/`；
- `runs/analysis/pi05_ecp_g3_free_delta_reachability_full10_248d768_gpu01p012_gpu02p47_20260831/`；
- `runs/analysis/pi05_ecp_g3_free_delta_reachability_retry4_248d768_gpu01p012_gpu02p7_20260831/recovered_full10_summary.json`。

## 78. base-score-conditioned v4 formal仍不产生bank选择性

clean pushed detached `b7d2638fa33ee620e488eef59c4ef2aff76c2122`完成v4 110步fresh训练、step70→110 exact resume及五worker
相邻Gate。step70/110 correct fit为`.922509/.929947`、same-task held为`.926447/.953521`，但unseen wrong-on为
`.930806/.933331`、correct-minus-wrong为`-.001784/-.006375`，正确bank更好只有`5/10`和`4/10`。信息墙与四family通过，
bank因果分离失败；两个checkpoint均strict non-pass。

相对v3唯一新增的B1 base signed score没有破坏correct容量，也没有让shared scorer取得free-delta证明存在的微小有效correction。
六task首步梯度分解显示v2/v3/v4 correct-vs-wrong functional gradient cosine中位为`-.961291/-.961291/-.966288`，对应
wrong/correct norm ratio中位`1.0597/.5298/.5038`。因此此前三轮并非随机训练波动：在当前local candidate chart里，两臂要求的
共享参数方向近乎相反，loss质量只决定落在哪个坏端点。

关键artifacts：

- `runs/outputs/pi05_ecp_program_bank_candidate_interaction_v4_base_score_s110_90cd380_gpu01p012_r3_20260831/`；
- `runs/outputs/pi05_ecp_program_bank_candidate_interaction_v4_base_score_gate_step70_b7d2638_gpu01p012_gpu02p47_w5_20260831/`；
- `runs/outputs/pi05_ecp_program_bank_candidate_interaction_v4_base_score_gate_step110_b7d2638_gpu01p012_gpu02p47_w5_20260831/`。

## 79. 32维vector interaction的pointwise task-local目标不是可靠资格

`codex/g3-vector-interaction@2295f481dcd284e4bae92afeaf2cf5c4b2d3e5c2`在scalar/local chart外加入32维Program/native-query与
candidate-key逐元素interaction；候选集合、真实X/Y value、positive/negative signed pooling、rank4 residual与唯一rank16均不变。
该分支已推送但未合并到`main`，因为后续资格没有通过。

首个task-local诊断逐candidate拟合free-delta teacher的normalized gauge。task1/task93的wrong recovery可降至约
`-.548/-.301`，correct却也降至约`-.555/-.525`与`-.425/-.410`。审计发现这个目标与最终功能错配：candidate logits存在softmax
常数gauge，且相同pointwise误差经真实X/Y、softmax与small-core SVD后可形成很不同的effective rank4。该结果只淘汰pointwise
free-delta imitation，不足以淘汰vector interaction函数类。

## 80. exact-effective-rank4 task-local资格在两个task暴露同一capacity--specificity冲突

最终诊断直接监督最终effective rank4矩阵：wrong fit0追随task-local free-delta teacher，两个correct fit views追随interaction-off R5
强方向；四family按teacher displacement等质量。correct held、wrong fit1与panel B零梯度。完整vector scorer在单task内自由优化80步，
仍使用真实native X/Y、signed pooling、rank4、carrier12+residual4和唯一完整rank16；Action Meta未安装，deployment native teacher
reads、validation/test与shuffled/reversed均为0。

task1的correct fit0/fit1/held panel-B recovery为`.720904/.717564/.711262`，wrong fit0/fit1为`-.527627/-.519287`；task93为
`.591613/.601969/.569709`与`-.379331/-.418162`。zero-gradient wrong fit1与correct held均泛化到各自fit结果，证明vector scorer
确实按bank内容产生不同更新；但它在两个代表task都无法同时恢复R5级correct方向。task1 correct/wrong effective-rank4 normalized MSE
mean约`.323/.100`，task93约`.242/.111`，其中wrong action-in仍是明显弱项。

这项task-local上界没有shared或deployment成功含义，但它比shared formal更严格地排除了“只需跨task学习更好”：首版逐candidate
local query/key interaction本身就没有展现足够的capacity--specificity组合。按预先约定，不进入task-LOTO、Natural Program joint G3、
v5或普通超参小扫；当前停在专家咨询节点，待判断是否引入bank-set/global-event级摘要或改变交互因子化，还是终止该函数类。

关键artifact：

- `runs/analysis/pi05_ecp_g3_effective_rank4_tasklocal_2295f48_gpu01p01_20260831/`。

## 81. 第六次专家复核终止local pointwise家族并指定EBSRI

第六次专家锁定远程`main@92617d070e3a573f640875b9f5bd355c162177d7`与未合并诊断分支
`codex/g3-vector-interaction@2295f481dcd284e4bae92afeaf2cf5c4b2d3e5c2`，确认第五次candidate-interaction方案已被正确实现并达到其
预设失败边界。scalar/base-score formal在十task上correct/held与wrong同时约`.93/.95/.93`，vector task-local则以correct降至约
`.72/.60`换取wrong负recovery；结合约`-.96`的correct--wrong功能梯度、free-delta容量与基础执行合同，足以从canonical停止当前
set-independent pointwise函数类及其普通超参小扫。

专家没有把该结果外推为所有逐candidate连续函数的数学不可能，也没有淘汰真实X/Y、exact signed pooling、antithetic correction、
rank4 residual、full-inverse capacity anchor、Natural Program、Stage0或Native-Factor整体。新假设指出旧learned correction缺少
Program-conditioned、event-conditioned whole-bank相对分布：单candidate局部特征相似时，correct/wrong bank产生相似参数Jacobian，
无法形成只改变wrong的第三方向。

唯一推荐架构Event-Conditioned Bank-Set Relative Interaction以Program的4 rank x 8 event native queries构造每candidate的32维
相对坐标；B0按video/event流式累计mean、dispersion与少量antithetic induced summaries，B1由summary-conditioned FiLM调制同一套
continuous correction，再对真实X/Y形成唯一signed measure、rank4 residual和完整rank16。推进顺序固定为S0 free-summary
factorization、S1 real-summary task-local、S2 fixed-route shared task-LOTO、S3 Natural Program joint；每一步只裁决最早接口。
1320行原始回复逐字保存于`docs/expert_review_20260831_event_conditioned_bank_set_relative_interaction.md`。

## 82. EBSRI S0 direct-summary双task formal通过

首版S0间接FiLM+共享zero-head的correct与wrong均保持约`.92--.99`，correction近零且free-token swap无因果效应。Panel-B exact teacher
与scale-matched wrong-only对照证明同一真实bank、signed pooling、rank4及factor网络可以产生强抑制，最早失效是summary条件未取得candidate
head控制。后继单变量修正由Program/event context与summary直接生成每family candidate linear head，仍严格zero-init并保持其它数据、
teacher、loss、rank、scale、carrier与信息墙不变。

clean pushed detached `3b7124e87514454387cf6e575c8526f470f2a8de`完成task1/93各110步与五臂Panel-B。task1 correct
fit0/fit1/held recovery为`.948785/.922930/.929913`，wrong fit0/fit1为`-.535224/-.491055`；task93为
`.905449/.909439/.894417`与`-.161546/-.169201`。两task全部absolute、wrong、margin、all-pairs、saturation与family checks通过，
Action Meta和validation/test reads为0，held/wrong-fit1/Panel-B无反传，未使用shuffled/reversed，且只生成一套完整rank16。双task
aggregate正式`pass`：

- `runs/outputs/pi05_ecp_event_bank_set_direct_s0_task1_s110_3b7124e_gpu01p0_20260831/`；
- `runs/outputs/pi05_ecp_event_bank_set_direct_s0_task93_s110_3b7124e_gpu01p1_20260831/`；
- `runs/outputs/pi05_ecp_event_bank_set_direct_s0_gate_s110_3b7124e_gpu01p01_20260831/`。

该结果只证明给定whole-bank condition时factorization可行，不证明真实set summary、跨task共享或Natural Program。下一阶段S1从fresh R5
初始化，只把training-only free summary替换为真实B0 set encoder。

## 83. EBSRI S1 real-summary task-local双task formal通过

EBSRI S1从fresh R5初始化，以真实B0 set encoder替换training-only free summary，并保持direct condition-generated head、真实X/Y、
exact signed pooling、rank4 residual、carrier12和其它Gate合同不变。formal aggregate根为
`runs/outputs/pi05_ecp_event_bank_set_s1_gate_s110_a1f14e4_gpu01p01_20260831`。task1 correct fit0/fit1/held recovery为
`.942/.953/.962`，wrong fit0/fit1为`-.529/-.517`；task93分别为`.928/.905/.881`与`-.188/-.180`。两task全部checks通过，
Action Meta为0、Panel B backward为0，并且只生成一套38-target carrier12+residual4 rank16。

S1只证明真实whole-bank summary在task-local训练中可以保持correct并抑制wrong，不证明跨task shared mapping。当前下一阶段因此是S2
fixed-route shared task-LOTO；只有该Gate通过，才可用全部10 tasks fresh refit并继续S3 Natural Program joint。随后
`main@cdcae8b`解耦B0 summary与B1 replay chunk：wrong profile约`12s -> 6.3s`、task1 correct约`35.5s -> 13.3s`，峰值约
`41.1GB`；这是系统吞吐事实，不改变S1科学裁决。

## 84. S2 effective代表元稳定non-pass，直接功能梯度审计支持一次同图修正

首轮S2在8个gradient tasks共享同一EBSRI set encoder/interaction，并hold out task1/93。step70/110相邻结果稳定，但step110
meta/target gradient correct中位仅约`.604/.639`，held task1/93 correct为`.507/.731`，task93 wrong为`.676`，两个checkpoint均
primary non-pass。60个correct jobs的effective recovery与正式Panel-B只有Pearson `.417`、Spearman `.433`；高内部恢复没有稳定转化为
真实policy功能，因此不能靠继续训练该surrogate通过。

随后在同一真实native banks、唯一rank16与8个gradient tasks上做无optimizer-step梯度审计。fresh状态16个correct/wrong条件的旧
factor gradient与直接Panel-A policy gradient cosine中位`.0219237`，最小`-.16748`且6个为负，证明错配并非训练后才产生。
与此同时，四拍schedule对应的2:1 correct/wrong direct-functional raw mean对8个task的投影全部为正，范围`.1318--.6323`；归一化
minimum-norm共同下降方向最小投影`.112935`。所以证据没有否定shared EBSRI参数化，允许的最小修正是保持图、数据、rank、split和Gate，
将训练代表元换成真实Panel-A功能VJP。

实现提交`25477c9`采用两遍内存调度、correct质量1与wrong有界neutralization质量.5，LR `1e-4`。gpu01物理`0,1,3,4,5,6`的world6
两步profile覆盖wrong与correct两种分支，步时`27.697/17.719s`、peak allocated `32.931GB`，finite nonzero aggregate gradient、Action Meta 0、
held/Panel-B backward 0、target-cache build 0与唯一rank16全部成立。profile只证明实现与资源合同；科学裁决仍需fresh step70/110及完整
100-job Panel-B Gate。

## 85. direct-functional S2稳定non-pass与interaction-bootstrap诊断

clean pushed detached `25477c922c66b5167520cac8c7a20776ff1f2ec1`完成fresh direct-functional S2 110步，独立detached
`3e156324c84b4f0908be06ec7b588f7132d6ccc5`随后完成step70/110共100个
Panel-B jobs。step110 meta/target gradient correct为`.879627/.930968`，task1/93 correct为`.949350/.899343`；但meta/target wrong为
`.444409/.904698`，task1/93 wrong为`.931056/.899985`，对应margin`.265943/.017216/-.002605/-.015191`。两个checkpoint均primary
non-pass，而相邻稳定性通过。所有job保持Action Meta 0、唯一38-target rank16、held/Panel-B backward 0、forbidden reads 0且未使用
shuffled/reversed：

- `runs/outputs/pi05_ecp_event_bank_set_s2_direct_functional_s110_25477c9_gpu01p013456_r6_20260901/`；
- `runs/outputs/pi05_ecp_event_bank_set_s2_direct_functional_gate_25477c9_eval3e15632_gpu01p013456_w6_20260901/`。

随后对direct step110的8 task×correct/wrong做无optimizer-step梯度审计。每task两臂的oriented cosine约`-.917,-.861,-.884,-.935,
-.927,-.700,-.659,-.705`；raw与简单normalized均值均非所有条件下降。16-condition normalized MGDA存在最小投影`.00675`，只说明
局部可行，不授权把MGDA引入最终Writer。对旧effective-surrogate step110重复同一审计时，unit-normalized简单均值对全部16条件正投影，
最小`.03279`；排除已inactive的task52 wrong后仍为`.03680`。旧delta本身对direct gradients近正交且8/16 uphill，所以它不是正确解，
但已形成从fresh direct路径缺少的bank-discriminative表示：

- `runs/analysis/pi05_ecp_event_bank_set_s2_gradient_geometry_direct_step110_25477c9_gpu01p6_gpu02p456_20260901/`；
- `runs/analysis/pi05_ecp_event_bank_set_s2_gradient_geometry_step110_f8d33fa_gpu01p013456_20260831/`。

由此下一次限定检验保持EBSRI架构、split、rank与Gate不变，只加载旧interaction作为training-only初始化，重置optimizer/scheduler/cursors，
并以全部8 task的paired correct/wrong unit-gradient direct polish训练。该组合不是fresh、不能拆分归因，也不证明Natural Program；若真实
Panel-B仍不能把wrong选择性迁移到held task，应停止继续精修该组合，而把原因落到shared coordinate/跨task interaction泛化。

## 86. functional-polish S2正式non-pass并提出absolute chart直达假设

clean pushed detached authority完成旧effective-surrogate interaction初始化、fresh optimizer/cursors及8 task paired unit-gradient
direct-functional polish，随后完成step70/110共100个Panel-B jobs。正式aggregate为
`runs/outputs/pi05_ecp_event_bank_set_s2_functional_polish_gate_s70s110_bb98b81_gpu01p0256_w4_20260901/aggregate.json`。
step110 meta/target gradient correct为`.827073/.771768`、same-task held为`.810918/.756753`、wrong为
`-1.059941/-.082424`、margin为`1.743702/.714894`；held task1 correct/held/wrong/margin为
`.514425/.497590/.097308/.380736`，task93为`.612724/.613756/.566386/.029681`，held/train为
`.621982/.793923`。step70→110相邻稳定，10/10 task全部correct view严格优于wrong，但两个checkpoint和两个role均primary non-pass。
Action Meta、held/Panel-B/validation-test backward、forbidden reads和shuffled/reversed均为0，且每condition仍只有一套完整rank16。

随后四组无训练context-path intervention显示旧checkpoint同时依赖B1 raw event context与真实B0 summary；因为置零后的输入处于训练分布外，
这些结果只作为依赖定位，不作Gate或模型选择。代码审查确认fixed Hadamard token经冻结R5映射为`rank_event`后，B0 inducing及B1
condition-generated head均直接读取该absolute chart state，同时native query另行形成Program-relative `kappa`。这与专家原式的`z`
条件一致，不是实现偏差。结合S1 task-local通过而S2 held迁移失败，当前提出“absolute route state允许interaction按已见code专门化”的
可证伪结构假设；但gradient correct仍未过线且两个held task表现不同，故尚非正式根因。下一检验只删除可训练B0/B1的absolute state直达，
保留R5 base/event weights/native query→`kappa`与全部native-factor执行合同，并从R5 fresh依次重跑S0、S1、S2。

## 87. rank/event-only absolute quotient S0 non-pass与target-owner混杂定位

clean pushed detached `6c33760c01acac0ce4c66a927465104ba293d091`完成首版absolute-route quotient的task1/93 fresh S0。正式
aggregate为
`runs/outputs/pi05_ecp_event_bank_set_quotient_s0_gate_s110_0d8d901_gpu01p56_retry1_20260901/aggregate.json`。task1
correct fit0/fit1/held recovery为`.850909/.845630/.877463`，wrong fit0/fit1为`-.371508/-.368927`；task93为
`.709234/.727288/.725272`与`-.182036/-.189203`。task1仅第二fit略低于`.85`，task93 correct/held明确失败，双task aggregate
为`non_pass`。两个run均自然完成110步、五臂Panel-B和checkpoint70/110，Action Meta、held/wrong-fit1/Panel-B backward及
validation/test/shuffled/reversed reads为0，仍只生成一套完整rank16，故结果不是工程故障。

随后只读加载冻结R5 scorer，在10个fixed Hadamard tasks上形成完整`rank_event[10,38,4,8,128]`并做平衡分解。跨task均值结构占
raw energy约`37.49%`；该结构的centered variance中，rank+event无owner近似解释`4.10%`，owner单轴解释`90.86%`，
owner+rank+event解释`94.96%`，owner+自由rank-event pair为`94.97%`。这说明首版quotient在切除absolute task state时也切除了固定
target ownership，形成了明确实验混杂。后继修正只补回无task轴的owner38 slot，并继续以rank4/event8 slots表示其余固定结构；B0/B1
仍不读取`program_event_state`数值，不恢复task route，也不改变训练或Gate合同。

## 88. owner-preserving quotient S0 non-pass与relational B1定位

clean pushed detached `21caa20e228f0a7e4f523a0cf939a61a014381e0`完成owner-preserving quotient的task1/93 fresh S0，aggregate为
`runs/outputs/pi05_ecp_event_bank_set_owner_quotient_s0_gate_s110_03b7314_gpu01p56_20260901/aggregate.json`。task1 correct
fit0/fit1/held为`.881943/.865835/.890028`、wrong为`-.412478/-.394324`并正式通过；task93为
`.685079/.695883/.662119`与`-.167709/-.178537`，correct与held正式non-pass。两run均自然exit 0、110步、五臂Panel-B及
checkpoint70/110完整finite，Action Meta与held/wrong-fit1/Panel-B backward为0，未读validation/test或shuffled/reversed，且每condition
仍只生成一套完整rank16，因此结果是结构性负证据。

同合同full-`z`、rank/event-only和owner版本的task93逐step/family审计显示，owner补回后correct均值从rank版`.721`反降至`.681`；
第8--10步已首先出现q-family correct-vs-wrong干扰，最终correct q-NMSE为`.197`而wrong q为`.166`。slot范数、正交性、移动量、
free-token norm和correction幅度均排除塌缩、饱和或简单训练不足。冻结R5重建又显示，对每个target从`rank_event[4,8,128]`减去其
rank/event均值后，task1/93全局RMS为`.153/.163`，per-target RMS约`.141--.183/.146--.183`，内部slot RMS保留约
`.075--.289`的有效强度差。由此后继只允许一次无新增参数的relational B1检验：每target中心化并按整个tensor单RMS归一，保留内部
rank/event关系，B0仍用task-independent inducing。该检验若fresh S0失败，就停止这一quotient机制，不靠普通超参修补。

## 89. target-centered relational quotient S0正式non-pass

clean pushed detached `1b0833744363d69bf843aa2a9efb1b7243c72bdf`在gpu01物理3/4完成relational quotient task1/93 fresh S0，
aggregate为`runs/outputs/pi05_ecp_event_bank_set_relational_quotient_s0_gate_s110_ad64757_gpu01p34_20260901/aggregate.json`。
task1 correct fit0/fit1/held为`.882725/.845604/.870278`、wrong为`-.410601/-.395834`，仅correct-fit1未过`.85`；task93为
`.704350/.717381/.723679`与`-.202943/-.216522`，correct/held未过。两run均自然exit0、110步、checkpoint70/110、五臂各16 visits
及全部tensor finite，Action Meta、held/wrong-fit1/Panel-B backward与validation/test/shuffled/reversed reads为0，每condition仍只生成一套
完整rank16。

task93与rank/event-only quotient结果几乎相同，说明target-centered rank×event relation未恢复full-`z`容量；owner-only也已正式失败。
由此预注册的relational quotient修正链结束，不进入S1，不组合owner+relation，也不做普通超参小扫。后继必须先审计global free token与
真实B0 target/group/type summary的拓扑差异、condition网络是否实际利用relational维度，以及专家停止当前correction factorization并重开
bank-conditioned primal的条件，形成新的机制证据后才能继续。

## 90. post-quotient接口审计与专家复核边界

后续只读parameter/Jacobian复核证明relational输入实际进入condition head并获得非零优化状态，但task93 q层间关系近共线，且centered
部分只保留raw Program state约`2.3--2.7%` energy；这解释了其能改变wrong suppression却不能恢复correct。代码逐版对照同时发现原
full-`z`不仅在B1使用Program state，也在更早的B0 inducing使用其rank mean；三个quotient共同删除了该B0 Program-conditioned读取，
所以既有S0无法把B0与B1责任分开归因。

另一个混杂来自S0正控：一个global free token广播到全部target/group/type，而真实B0 summary具有逐target input及逐target/group
all/by-type拓扑。该差异在full-`z`存在时可被逐target Program context补偿，删除`z`后可能成为额外容量瓶颈。owner随后要求停止未定
架构并咨询专家；咨询范围必须覆盖自第六次意见以来S0/S1通过、三类S2失败及三次quotient S0反证的完整链条，解释为何仍未跨过G3并给出
具体后续计划。Program直接query、whole-bank context、B0/B1职责和absolute-code旁路是关键接口之一，但不是唯一问题。截至该边界，
没有新的S0/S1/S2启动，未提交的候选草案不属于active history；formal evidence均保留。

## 91. 第七次专家复核与Program-through-bank bottleneck裁决

专家对`main@d6f5715bf49277f1d8618e34fa9da84981eb827c`及其可达历史完成复核，1359行原文保存于
`docs/expert_review_20260901_program_through_bank_bottleneck.md`。复核确认S0/S1 task-local通过与三类S2 shared失败并不矛盾：前者证明
每任务存在解，后者证明当前共享、可迁移选择规则尚未被识别。fresh direct-functional保持correct但wrong同样高；surrogate bootstrap
加unit-gradient polish在八个训练任务形成bank specificity，却降低correct absolute capacity并不能稳定迁移到task1/93。

代码审计进一步纠正了quotient归因：S0的`with_condition()`把一个training-only `[E,S]` token覆盖到所有input targets、output groups和
all/by-type scopes，inducing-dependent summary字段在B1前被覆盖；full-z另有逐target/rank/event raw `z`直达B1，而quotient没有。因此三次
quotient只证明“删除raw code且仍使用过窄global condition”不能恢复task93，不能证明scope-matched summary-only B1或真实
Program-conditioned set read不可行。

owner据此采纳唯一后继Program-through-Bank Bottleneck EBSRI：Program仍产生native queries、base primal与event weights，但只能通过
query真实candidate set形成逐target/rank/event及output group/type匹配的bank response；B1只读该response和固定结构，最终仍以一个exact
signed measure pool真实X/Y、形成一个rank4 residual与唯一完整rank16。预注册顺序为topology-matched free-summary S0、real
Program-through-bank S1、fresh direct-functional shared S2；失败按最早接口分别转向bank-conditioned primal或停止当前shared coordinate。

## 92. Program-through-bank bottleneck S0双task正式通过

clean pushed implementation authority为`b11dc3eec4de47c0861f92bab7fb9a331b90fce4`，formal从只增加launch record的clean detached
`bc5c34a8400e571cee58a259ce039eea32b0a318`在gpu01物理2/3并行完成task1/93各110步。aggregate为
`runs/outputs/pi05_ecp_program_through_bank_bottleneck_s0_gate_s110_b11dc3e_gpu01p23_20260901/aggregate.json`，结论`pass`。

task1 correct fit0/fit1/held recovery为`.989173/.974203/.989332`，wrong fit0/fit1为`-.565043/-.565553`；task93 correct为
`.946634/.939966/.916589`，wrong为`-.341756/-.393537`。两task的correct、held、wrong、margin、all-pairs、family与saturation
全部通过，near-bound fraction为0；110步、checkpoint70/110、五臂Panel-B、Action Meta 0、zero-gradient held/wrong-fit1、唯一完整
rank16及全部信息墙合同完整。

这一结果只回答topology-matched free-summary S0：切断raw Program旁路后，summary-only B1与fixed base+bounded correction仍有充分
capacity和bank specificity。它不证明真实B0或shared mapping。下一阶段按专家顺序从R5 fresh运行real Program-through-bank S1，S0只提供
aggregate准入证据，不能加载其interaction、free summaries、optimizer或cursor。

## 93. Program-through-bank bottleneck S1双task正式non-pass

scientific implementation/config authority为`90472301990f3c86d6aabc6acae214f175ab017b`；formal从只增加launch record的clean detached
`1cdfbfa7bf05ebe369807afdf6d8c99313a9d458`在gpu01物理2/3并行完成task1/93各110步。aggregate为
`runs/outputs/pi05_ecp_program_through_bank_bottleneck_s1_gate_s110_9047230_gpu01p23_20260901/aggregate.json`，结论`non_pass`。

task1 correct fit0/fit1/held recovery为`.826825/.855228/.797545`，wrong fit0/fit1为`-.659529/-.628416`；task93 correct为
`.776511/.792673/.719798`，wrong为`-.178132/-.151166`。两task的wrong、margin、all-pairs和correction saturation checks均通过，
near-bound fraction为0；失败只在correct absolute与zero-gradient same-task held保持。两run自然exit0，step0误差0、110步、五臂Panel-B、
Action Meta 0、冻结source/Native Stage0、唯一完整rank16及信息墙均成立。最长视频的checkpointed chunk replay已以真实forward/backward和
非chunked合同验证，故先前OOM修正不改变该科学结论。

S0与S1唯一科学变化是scope-matched free summary tree换成真实Program-query/current-bank B0。S0强通过而S1同时损伤两task correct/held、
仍压低wrong，按专家第5.2与7.1节把最早失效接口定位为real B0 representation或Program→set query，而非B1、真实native X/Y、signed
pooling或rank4容量。原fresh shared S2因此没有启动；当前fixed-base+summary-only bounded correction函数类停止。后继只按专家失败分支
让whole-bank response更早参与bank-conditioned primal形成，再做current-bank dual与唯一exact replay，不恢复quotient/surrogate/polish，
也不做普通超参小扫。

## 94. bank-conditioned primal恢复correct但原query仍缺乏wrong specificity

按第七次专家§7.1失败分支，`eb9f295638e4f2f2f47c472cb57bb8607aae21e7`把whole-bank response前移为
`d_B=d0+sum_e G(S)A_B`，再对当前bank做full inverse与唯一exact signed replay。双task formal aggregate为
`runs/outputs/pi05_ecp_bank_conditioned_primal_gate_s110_eb9f295_gpu01p12_20260901/aggregate.json`。task1 correct
fit0/fit1/held为`.951/.931/.925`、wrong为`.428/.477`；task93 correct为`.917/.923/.888`、wrong为`.627/.654`。两taskcorrect、held与
all-pairs通过，wrong与margin失败；Action Meta、source/Stage0梯度、held/wrong-fit1/Panel-B backward和forbidden reads均为0，唯一rank16
与checkpoint合同完整。该结果证明whole-bank前移恢复capacity，但当前Program query/native anchor/family gate不能提供充分bank specificity。

## 95. Q_free与base-LR A_free依次排除query under-travel并暴露anchor优化混杂

首轮task93 Q_free因direct query位移只有普通Program网络约`1/38`而不能裁决容量；固定`4 rank × 8 event=32`步长校准后，formal correct
为`.808/.826/.795`、wrong为`.526/.534`，形成capacity--specificity权衡。对应roots为
`runs/outputs/pi05_ecp_bank_conditioned_primal_qfree_task93_s110_5cb1be6_gpu01p2_r1_20260901/`与
`runs/outputs/pi05_ecp_bank_conditioned_primal_qfree_calibrated_task93_s110_fdc669f_gpu01p0_20260901/`。

随后nested A_free保持真实candidate anchor，并加跨全部arms/banks/videos共享的逐target/rank/event full-native basis。base-LR formal root为
`runs/outputs/pi05_ecp_bank_conditioned_primal_afree_task93_s110_b0d81bb_gpu01p0_20260901/`，correct为`.815/.833/.797`、wrong为
`.512/.524`。checkpoint中233个anchors全部进图且optimizer moments非零，但合并RMS仅`.0094`、约为candidate的`3.7%`；同checkpoint
F=0只改变correct至多`.0027`、wrong约`.011`。因此该轮只淘汰小幅A_free，不淘汰充分行使的full-native span。

## 96. calibrated A_free正式non-pass并结束第七次专家执行链

科学实现`e02f4ca`只把既有free anchors移到固定32倍坐标步长组；formal从clean pushed detached
`144d59b6e1d47fb7a3725cac5a4a708ad1b66001`在gpu02物理4完整运行，root为
`runs/outputs/pi05_ecp_bank_conditioned_primal_afree_calibrated_task93_s110_e02f4ca_gpu02p4_20260901/`。correct fit0/fit1/held为
`.853296/.858892/.818467`并全部通过，wrong fit0/fit1为`.611592/.668511`，wrong和margin失败，all-pairs通过；110步、checkpoint70/110、
五臂各16次Panel-B、Action Meta 0、唯一rank16及信息墙完整，Gate为`non_pass`。

step110 free-anchor合并RMS为`.17664`，input/output为`.21929/.12706`，与candidate anchor`.188--.192`同量级，排除under-travel。
同checkpoint F=0使correct变为`.879708/.883433/.849663`、wrong变为`.750229/.756445`：F把wrong压低`.088--.139`，同时把correct压低
`.025--.031`，margin只从`.123`增至`.185`。逐层审计显示bank-dependent candidate delta的correct/wrong cosine约`.718--.772`，但
幅度占主导的free delta cosine约`.993`，summary/gate约`.991/.965--.971`。正式审计保存在
`runs/analysis/pi05_ecp_bank_conditioned_primal_afree_causal_audit_144d59b_gpu02p46_20260901/`。

至此第七次专家建议的scope-matched S0、real S1及其失败后的bank-conditioned-primal分支均已实际执行。S0通过，S1触发预注册停止并未进入
shared S2，primal pivot虽恢复correct却在原query、calibrated Q_free、base-LR与充分校准A_free下持续无法满足wrong/margin。当前停止
`summary→family-scalar G`加event-additive native anchor这一具体parameterization；该负结果不外推为Program schema、真实native X/Y、
signed pooling、rank4或整个ECP根本失败。

## 97. PNBTT首个单key-chart E1 formal稳定non-pass

PNBTT E1科学authority为`2664e0d3705da3cdfb4bde2e7633317e0b102b4a`，formal root为
`runs/outputs/pi05_ecp_pnbtt_e1_free_query_s110_2664e0d_gpu01p12_20260902/`。task1/93在gpu01物理1/2完成110步；训练期间
macro70 Panel-B在物理3/4并行执行，随后macro110完成相同五臂各16次评测。`evaluations/qualification.json`及逐task结论均为相邻一致
`non_pass`。

macro70 task1 correct fit0/fit1/held为`.628597/.639663/.606223`、wrong为`.126684/.192265`；task93 correct为
`.676496/.711490/.668680`、wrong为`-.000097/.257616`。macro110 task1为`.641984/.660311/.622909`与
`.122637/.186146`；task93为`.713247/.737497/.685649`与`.006121/.269427`。all-pairs、all-correct > all-wrong与near-bound
均通过，task1 wrong通过；主要缺口是correct/held和`.50` margin。70到110改善只有`.013--.037`，near-bound最大值为macro70
task1/task93 `.022005/.008557`、macro110 `.017115/.009780`，不是饱和或训练不足。

两枚checkpoint、五臂Panel-B、Action Meta 0、held/wrong-fit1/Panel-B backward 0、validation/test读取0、shuffled/reversed未运行及
唯一完整38-target rank16合同全部成立。Natural Program在E1冻结，所以只淘汰当前single-key-chart tangent parameterization，不进入E2。
下一步按专家§5.10计算`T=Cov(v,k)`功能梯度投影谱，再在`m`扩展或family-shared trunk + target-specific低秩key projection之间做有
机制依据的唯一修订并fresh重跑E1。

## 98. PNBTT tangent spectrum排除key-width截断并进入family-key修订

诊断实现authority为`8306a4cb43ee612671955354fbe0c508de996344`，retained root为
`runs/analysis/pi05_ecp_pnbtt_e1_tangent_spectrum_m128_step110_8306a4c_gpu01p12_20260902/`。它加载首个E1 macro110，
只在task1/93 Panel-A的correct fit0/fit1与wrong fit0上各取16个visits，计算38 targets × input/四类output × 两task共380个
`T=Cov_mu(v,k)L^{-T}`谱及功能梯度投影；held、Panel-B、validation/test读取均为0，`completion.json`自然完成，耗时`376.97s`。

当前`m=128`给每个operator 1024列。99%谱能量rank远低于该上限且末端10%能量通常只有`0--1e-6`量级，故不触发专家允许的
`m`扩展。q input/output的correct-preserve-wrong梯度保留率中位约`.555/.175--.240`，v约`.463/.620--.808`；q/v/action
abs或input的correct--wrong operator cosine约`.922--.979`，说明正确与错误bank在当前线性key坐标下仍高度重合且各family可达功能
梯度显著不均。该证据按专家§5.10触发family-shared nonlinear trunk + target-specific低秩key projection，而不是width、LR、seed、
head或rank扫。

后继v2只改key parameterization：四个family各有共享input/output nonlinear trunk与side heads，每target/side增加rank16线性低秩
residual projection；`m=128`、rank4、free query、E1数据、三项loss、Gate和formal cadence均保持不变。fresh E1真实profile与formal
结果尚未产生，不能把该结构修订写成科学通过。

## 99. PNBTT family-key E1 formal相邻一致non-pass

v2实现authority为`02633a3964ecfd9d40f9827ba98456c87c07552b`，formal从只增加launch记录的clean detached
`75db5f847e849c8953d4afeae4b7682e185ee734`运行，root为
`runs/outputs/pi05_ecp_pnbtt_e1_family_key_s110_02633a39_gpu01p12_20260902/`。task1/93在gpu01物理1/2完成110步；
macro70与macro110 Panel-B依次在物理3/4执行，两个checkpoint、raw metrics、completion和相邻qualification均完整。

macro70 task1 correct fit0/fit1/held为`.598648/.599961/.581859`、wrong为`.028320/.041884`；task93 correct为
`.693744/.706930/.650097`、wrong为`.036270/.224452`。macro110 task1 correct为`.616630/.620958/.601512`、wrong为
`.027332/.051458`；task93 correct为`.707775/.725727/.655429`、wrong为`.047247/.223365`。wrong、all-pairs、near-bound和
task1 margin通过；correct/held与task93 margin失败。70到110的correct/held改善仅`.0053--.0210`，总体与逐task均相邻一致
`non_pass`。

该结果只淘汰当前family-shared nonlinear key chart + target-specific rank16 key residual + rank4 PNBTT参数化。它比首版更强压低wrong，
但没有恢复correct absolute capacity，故不进入E2、不续训、不做width/LR/seed小扫。按专家§5.10，下一证据是同checkpoint的train-only
`T=Cov(v,k)`功能梯度投影谱；只有同构PNBTT task-local full-rank16 oracle明显优于rank4时才重开carrier/task rank分配。

## 100. Family-key v2 tangent spectrum停止继续修改key chart

retained root为
`runs/analysis/pi05_ecp_pnbtt_e1_family_key_tangent_spectrum_m128_step110_75db5f84_gpu01p12_20260902/`，加载同一v2
macro110，只使用task1/93 Panel-A correct fit0/fit1与wrong fit0，每task 16 visits，共380个target-side
`T=Cov_mu(v,k)L^{-T}`谱；held、Panel-B、validation/test读取均为0，`completion.json`自然完成，耗时`381.48s`。

`m=128`仍未截断有效谱。相对首版，q/v input的correct-preserve-wrong中位只从`.555/.463`变为`.566/.476`；q四类output为
`.174/.235/.220/.224`，v为`.643/.769/.685/.693`，其中v adj/init/goal还低于首版。family chart把action-out
adj/goal correct--wrong operator cosine从`.839/.748`降至`.712/.627`，解释了formal wrong改善，但q/v input仍约`.958`且
abs仍约`.927/.963`，没有恢复correct方向可达性。尾端10%谱能量继续近零。

该证据停止`m`扩展和更多key-chart变体，也不授权LR、seed或rank sweep。它本身不证明rank4 ceiling；按专家§5.10只准入一次同构
PNBTT task-local full-rank16 oracle，且输出仍为唯一38-target rank16。只有该oracle相对rank4 residual明显改善才重开carrier/task
rank分配，否则停止rank扩展。

## 101. PNBTT full-rank16 oracle稳定non-pass并结束已授权的E1扩展序列

oracle实现authority为`57969a6895adfe2e336e5d83a30d1a80c12d47d2`，formal从只增加launch记录的clean detached
`1897b8dceecf93d1b3063b6f42a78f286cb699b2`运行，root为
`runs/outputs/pi05_ecp_pnbtt_e1_fullrank16_oracle_s110_57969a68_gpu01p12_20260902/`。task1/93在gpu01物理1/2完成110步，
macro70在物理3/4与训练并行评测，macro110在训练结束后使用同样两卡完成；两个checkpoint、五臂各16次Panel-B、raw metrics、
run contracts、completion和相邻qualification均完整，所有launcher exit为0。

macro70 task1 correct fit0/fit1/held为`.953328/.933839/.941449`、wrong为`.648060/.719726`；task93 correct为
`.557237/.561168/.411465`、wrong为`-.001312/-.007719`。macro110 task1 correct为
`.960297/.941644/.948351`、wrong为`.634156/.711548`；task93 correct为`.586174/.595686/.449605`、wrong为
`-.006466/-.021862`。在task1，correct/held、all-pairs和near-bound通过，wrong/margin失败；在task93，wrong/margin、all-pairs和
near-bound通过，correct/held失败。两枚checkpoint总体和逐task均相邻一致`non_pass`。

该oracle只将`carrier12+task4`替换为`carrier0+task16`，其余family-key PNBTT、free query、E1数据、三项loss、LR、seed、110步
cadence与Gate均不变；仍只物化一套38-target rank16，Action Meta、held/wrong-fit1/Panel-B backward和validation/test读取均为0。
70到110的结果稳定表明：task16可在单task上分别恢复capacity或specificity，但两者在task1/93上反转，没有相对rank4形成一致、广泛、明显
更优的rank分配证据。按专家§5.10，rank扩展至此停止，不运行中间rank、scale、seed、LR或额外chart变体。

E1的free-query real-bank transport仍未通过，所以E2不启动；专家次选B所要求的“E1通过但换成frozen G2 Program后系统失败”未发生，
whole-Writer D的A路线前置Gate也未满足。因此PNBTT不再是active implementation route，需要owner返回新专家裁决或明确扩展authority才能继续。
这一停止只覆盖已实际测试的PNBTT E1函数类与扩展序列，不裁决冻结的Natural Program、G2、native X/Y、signed pooling或整个ECP。

## 102. Train-side loss audit撤销“PNBTT已无active route”的过早裁决

后续逐步审计三个PNBTT E1 formal root的`metrics.jsonl`：

- `pi05_ecp_pnbtt_e1_free_query_s110_2664e0d_gpu01p12_20260902`；
- `pi05_ecp_pnbtt_e1_family_key_s110_02633a39_gpu01p12_20260902`；
- `pi05_ecp_pnbtt_e1_fullrank16_oracle_s110_57969a68_gpu01p12_20260902`。

三者的`normalized_necessity_margin`均为`.10`，而formal E1 Gate要求minimum correct minus maximum wrong至少`.50`。首个run在step2后、
后两个run在约step10后`active_necessity_fraction`长期为0；当Panel-A normalized separation超过`.10`后，wrong-video necessity hinge就停止提供梯度，
即使Panel-B formal `.50` margin仍失败。这与专家§7.2将wrong-video necessity列为三项最小loss之一的意图直接不对齐。

因此第101节中full-rank16不重开rank分配的裁决仍有效，但“已授权E1序列耗尽”与“需要新专家authority”被本节后续证据取代。新的唯一
active config为`configs/pi05_ecp_pnbtt_e1_gate_aligned_necessity_v1.json`；它保持family-key rank4、LR、seed、数据、loss权重、Gate和评测口径不变，
只将`normalized_necessity_margin`对齐为`.50`，从fresh重跑E1。

## 103. Gate-aligned PNBTT E1稳定non-pass并耗尽现有专家分支

唯一gate-aligned实现authority为`e65c63888033639c58d29f285aed6cd8331c07e8`，formal从只增加launch记录的clean detached
`2050de9e7583955fa0c62eaeb375eb5b3847500a`运行，root为
`runs/outputs/pi05_ecp_pnbtt_e1_gate_aligned_necessity_s110_e65c6388_gpu01p12_20260902/`。task1/93在gpu01物理1/2完成110步；
macro70在物理3/4与训练并行评测，macro110在训练结束后完成。两个checkpoint、五臂各16次Panel-B、raw metrics、run contracts、
completion、logs与相邻qualification均完整，所有最终launcher exit为0。macro70初次错误的NUMA0/p3--4 evaluator在加载模型、建立输出root或
产生科学结果前被formal NUMA guard拒绝；改用NUMA1后fresh成功，不污染证据。

macro70 task1 correct fit0/fit1/held为`.585596/.592489/.541733`、wrong为`-.176695/-.153551`；task93 correct为
`.707213/.715694/.676823`、wrong为`-.055836/.018941`。macro110 task1 correct为`.607645/.609189/.561628`、wrong为
`-.171164/-.149315`；task93 correct为`.710657/.721565/.686395`、wrong为`-.086657/.006107`。两checkpoint的task1/93均通过wrong、
`.50` fit-margin、all-pairs与near-bound，只失败correct/held；overall与per-task conclusion consistent及training complete均为true。
70到110改善很小，结果不是训练未完成。

训练中`active_necessity_fraction`在step1--10为`.95`、11--70为`.3083`、71--110为`.05`；末步task1/93 free-query梯度为
`.1701/.1801`、shared-key梯度为`.04281`，preservation平均激活率`.9909`。这证明第102节发现的`.10` hinge错配已被充分修正；
`.50` objective先强力压低wrong，随后在train Panel-A separation达标时自然关闭，但没有恢复absolute correct capacity。

后续train-only spectrum root为
`runs/analysis/pi05_ecp_pnbtt_e1_gate_aligned_tangent_spectrum_m128_step110_2050de9e_gpu01p12_20260902/`。它包含task1/93各16个
Panel-A visits、三条gradient arms与380个target-side spectra，held/Panel-B/validation/test均未使用；耗时`382.57s`。最大末端10%谱能量
`1.3664e-5`与旧v2的`1.3675e-5`等价，q/v input correct--wrong operator cosine仍约`.9580/.9577`，没有`m`截谱、family chart新方向或
rank扩容信号。

因此第102节恢复的唯一修订也已稳定`non_pass`。专家规定E1通过后才进入E2；次选B只在E1通过而真实frozen Program E2系统失败时触发；
whole-Writer joint同样要求上游A路线取得资格。三者均未满足，当前没有active implementation route，不运行E2、B、joint、续训或
LR/seed/width/rank/scale/chart扫描。该停止只覆盖已实际检验的PNBTT E1 transport函数类，不裁决Natural Program、G2、native X/Y、
signed pooling、rank4、整个ECP或zero-interaction目标。

## 104. 完整历史复核与Policy-Response Event-to-Factor Writer启动

PNBTT E1耗尽后，owner要求从最早EMBER到当前ECP重新复核全部路线，并重点判断：如何吸收ECP已证明有效的原生证据，同时把Writer改成
可通过复制标准block扩展、而非连续堆叠专用数学坐标的架构。完整专家原文逐字归档为：

- `docs/expert_review_20260902_full_history_policy_native_meta_writer.md`；
- `docs/expert_review_20260902_policy_response_event_to_factor_writer_clarification.md`。

专家补充意见撤回上一版的negative-control训练与task-expert factor dictionary主张，并将新主案收敛为Policy-Response
Event-to-Factor Writer：

1. 冻结PI0.5逐帧捕获image/language、19层Action Expert boundaries、50 horizons、正负probe、flow velocity与38-target真实X/Y；
2. learned Video Process Encoder沿teacher-frame time形成task-grounded、boundary-anchored ordered events；
3. learned Factor Composer以38x4 queries读取events与当前视频bank，通过一个ragged native-group signed-attention operator直接pool
   raw X/Y；
4. rank4 factors只做一次small-core canonicalization并与frozen rank12 carrier拼成唯一rank16；
5. 训练只使用正确视频的cross-episode functional、positive causal policy-response prediction与轻量preservation；所有负controls在
   selected checkpoint冻结后评测。

该路线吸收G1 native-factor容量、G2 ordered dynamics、P0/P1 current-bank operator、chunked replay、functional runtime与materializer，
同时退役固定Natural Program到summary、covariance、whitening、transport、anchor和family scalar gate的硬接口。G2 Program仍作初始化与
机制证据，不再是deployment唯一schema。

主任务最后复核发现并修正四个未定义点：process auxiliary必须strictly prefix-only；预测target必须冻结以避免learned target collapse；
新Composer先在task1/task93做task-local正控；full/coarse matched对照只裁决前端表示，因为两臂都保留完整dynamic X/Y bank。这些修正
进入`docs/policy_response_event_to_factor_writer_design.md`。

owner于2026-09-02确认整体设计、要求建立持续goal并立即推进，强调尽快获得真实Writer闭环性能信号，但不得作弊或用随意改动替代有依据
的接口修正。系统goal已建立；active计划从最新clean pushed main创建唯一实现worktree，先完成真实forward/gradient/materialization与
task-local Composer正控，随后立即运行12-task full/coarse matched GPU实验并尽早进入held5 correct-only closed loop。

## 105. Policy-Response Writer真实图与Composer正控通过

首个实现提交为`66df197495d730a026c02883cbede9042f461a98`，正式正控从包含launch合同的clean detached
`a7d84d78523ad1d5e5afb2823c5872b7ae020281`运行。真实smoke一次捕获task1的51个stride-5 frames、19 layers、50 horizons、2 probes及
38-target native X/Y；初始mobile update严格为零，打开scale后functional梯度到达Frame、Event与Composer，prefix-only frozen-target
process梯度到达Frame/Event/predictor，76个LoRA tensors物化为唯一rank16。

task-local Composer-only正式正控均使用两条fit视频交替接受Panel-A correct-only梯度，第三条same-task held视频与全部Panel-B零梯度；
每个step70/110 checkpoint分别在三条视频上完成16次Panel-B。结果为：

- task1 root `runs/outputs/pi05_ecp_policy_response_writer_tasklocal_task1_full_s110_66df1974_gpu02p5_20260902/`：step70/110 fit
  recovery `.260876/.276421`，held-video `.207341/.244598`；
- task93 root `runs/outputs/pi05_ecp_policy_response_writer_tasklocal_task93_full_s110_66df1974_gpu01p2_20260902/`：step70/110 fit
  recovery `.337207/.346604`，held-video `.300885/.280724`。

两task、两个相邻checkpoint、每个checkpoint的两条fit与一条held视频全部优于carrier。该结果证明去掉PNBTT solve/transport后，新的
event-conditioned current-bank Composer仍能从真实native X/Y学得跨视频功能，不证明shared mapping或closed-loop目标已经成功。

shared positive-only运行面随后接通固定3 meta + 3 target/task-equal循环、稳定evidence-cache ownership、deferred NCCL、exact LoRA-leaf
VJP、prefix-only process loss、单侧preservation、step70/110 checkpoints与零梯度Panel-B评估。单卡两步和双卡单步profile均通过；双卡
profile明确覆盖一个rank执行真实task、另一个rank以显式零梯度参加all-reduce的拓扑，没有deadlock或参数分叉。profile只作工程证据，
不能用于方法选择。下一项科学实验仍是同配置、同参数量、同数据与训练预算的12-task K1 component-init full/coarse matched formal。

## 106. 12-task matched前端裁决与自然mapping扩规模

12-task K1 component-init full/coarse均完成110步、macro70/110零梯度Panel-B和held5 correct-only strict250。full的gradient-task
fit/held benefit在macro70为`.00054184/.00019916`、macro110为`.00103244/.00058589`；coarse分别为
`.00068417/.00059560`与`.00082845/.00071538`。两臂的task2/74 true-task-held在两个checkpoint全部为负，说明小幅train-task
functional改善没有转成task-disjoint映射。

closed-loop结果为：

| 表征/checkpoint | strict250 | Long/Goal/Object/Spatial0/Spatial9 | carrier retention |
|---|---:|---:|---:|
| full 70 | 33 | 0/0/1/32/0 | 29/43 |
| full 110 | 31 | 0/0/1/25/5 | 25/43 |
| coarse 70 | 43 | 0/0/2/40/1 | 37/43 |
| coarse 110 | 41 | 0/0/4/34/3 | 35/43 |

全部1000 rows与launcher完整、错误为0；四点均不超过carrier43，breadth最多3/5且Goal/Long全0。full没有证明19层x50 horizon
response表示相对coarse final-layer horizon mean的增量，且carrier破坏更强；因此后续首选coarse。这不裁决两臂共同使用的完整current-video
dynamic X/Y bank。

task-local Composer强、10-task shared弱且true-task-held为负，最早接口是task-disjoint shared mapping与自然factorial coverage。下一实验
保持coarse、K1、Writer、rank12+4、functional/process/preservation和学习率不变，只把gradient mappings扩大到55个审计meta与18个
target-fit；task2/74仍为true-task-held。每task选择最前两条outcome-independent正确视频作fit、最后一条作held，10 warmup + 1200
effective updates，macro610在训练继续时先做held5 correct250，macro1210作相邻裁决。这个scale实验检验mapping diversity，而不是恢复
full前端或进行超参数小扫。

## 107. 73-task scale macro610 closed-loop显著退化

scale component-init formal从clean pushed detached `df7a7f5a837192027ea4d9e8f566d0d2459e26df`在gpu01物理2/4/5/6继续运行；
固定coarse、K1、同一Writer/rank/LR/seed与positive-only objectives，梯度mapping为55 meta + 18 target-fit，五个held target和其它held
tasks均零梯度。macro610 checkpoint完整保存后，使用每个held task固定correct demo5各调用Writer一次并物化唯一38-target rank16 bank：

- materialization：
  `runs/outputs/pi05_ecp_policy_response_writer_scale73_coarse_m610_held5_correct_k1_materialized_df7a7f5a_gpu02p4_20260903/`；
- strict250：
  `runs/outputs/pi05_ecp_policy_response_writer_scale73_coarse_m610_held5_correct_k1_strict250_6ddceff5_gpu02p46_r2_20260903/`。

评测包含250个严格配对状态、20个cost-balanced shards和四个persistent workers，全部return code为0。结果为`26/250`，逐task
Long/Goal/Object/Spatial0/Spatial9=`0/0/1/25/0`，breadth`2/5`。相对carrier43，retained/gained/lost为`22/4/21`、success-set
Jaccard约`.4681`、paired exact p约`.00091`；相对旧coarse macro70同样为`22/4/21`，相对macro110为`21/5/20`。因此扩大自然
mapping在macro610没有恢复shared closed-loop，反而显著损伤carrier支持。

这是有效科学non-pass而非信息墙或评测故障：held action/reward/state、validation/test、wrong、shuffle和reverse读取均为0；每task-condition
只有一次Writer调用，部署端无Action Meta、第二adapter或runtime teacher-video读取。为支持scale checkpoint，evaluator中只接受旧
`{70,110}`的实验专用硬编码由`6ddceff5f30f02282d75c95a0ba9bae8d20a5652`修正为正数macro且checkpoint目录名必须精确匹配；构造测试和
真实macro610 bank reinspection均通过，该修复没有改变policy、LoRA或rollout合同。

train-side functional benefit从前100步约`.00012`持续升至601步后的约`.0038--.0042`，但held closed-loop下降。物化矩阵进一步显示
macro610 mobile4整体函数范数为carrier12的`1.81--2.15`倍；旧macro70/110为`.49--.66`倍，正式G1四个非零mobile held tasks约
`.77--.80`倍。放大遍及38 targets，尤其多层q-proj，而非单target爆炸；不同held task mobile方向余弦约`.03--.49`，也不是所有输出
完全塌成同一向量。当前定位因此是shared train functional credit向held task外推时的幅度/功能错位，而不是残差没训练、单层数值错误或
纯task-collapse。

为避免把低参数cosine误判成必然功能正交，随后在封存G1 effect bank的48个固定observation/noise anchors上，对carrier、G1正式
adapter和macro610 Writer adapter重新执行相同PI0.5 10-step denoising，比较owner/flow/action response；这是零梯度、使用privileged
reference的post-hoc定位，不参与checkpoint选择。raw root为
`runs/analysis/pi05_ecp_policy_response_writer_scale73_m610_g1_effect_alignment_5df9406_gpu02p46_20260903/`。五task
successful-member effect loss均值依次为carrier `.914596`、G1 `.238841`、Writer `1.023186`；Writer只在Spatial9上相对carrier
略降，另外四task均恶化，平均净恶化`.108590`。四个G1非零mobile task上，Writer相对G1的member-scale-whitened联合功能方向cosine
范围`.050440--.304702`、中位`.147526`，联合effect norm ratio为`.456257--.744368`、中位`.663001`。因此参数余弦近零确实漏掉
少量功能等价分量，但macro610的大参数norm没有形成过强的正确功能效应；主要容量位于低敏感或错误方向，不能靠事后缩放恢复G1。

macro610不冻结checkpoint、不运行negative controls，也不准入mixed-K、fully-random或validation。训练继续到预注册macro1210，使用相邻
closed-loop判断该失败是否稳定或进一步加剧。

## 108. macro610暴露缺失的完整scale边界与方向梯度预算

专家澄清稿§7.5规定每target只使用fit-only task-equal全局scale reference、网络预测bounded relative gains，§9.4又明确要求
“每target effective-update RMS cap”。首版Composer只实现了逐rank的`s_ref * tanh(gain)`；四个rank合成后没有完整矩阵边界。
对macro610五套held adapters只读重算表明，`94/190`个task-target的mobile `B@A` RMS超过各自`s_ref`，最大比值`2.2433`，
其中90个是18层q targets、另4个是action-in。作为参照，fit-only shared rank template的38个target均不超过`1 x s_ref`，正式G1
held5也只有`5/190`个轻微超过。

零梯度事后把每target全部四个mobile ranks共同乘以`min(1, s_ref/RMS(B@A))`，small-core方向、carrier12和配对状态均保持不变。
诊断root为
`runs/analysis/pi05_ecp_policy_response_writer_scale73_coarse_m610_sref_effective_cap_b8ad986_gpu02p46_20260903/`；其中
`cap_report.json`和`materialized/manifest.json::posthoc_diagnostic`明确登记不用于训练或checkpoint选择。strict250完整结果为
`33/250`、breadth`1/5`，Long/Goal/Object/Spatial0/Spatial9=`0/0/0/33/0`。相对原macro610，retained/gained/lost为
`20/13/6`、net `+7`、paired exact `p=.167068`；局部恢复Spatial0的8条净成功，但丢掉Object唯一成功，未恢复跨suite方向。

另对原训练前804条metrics分解梯度：global clip触发`.878109`，scale head与其余方向norm中位为`2.59920/.583903`；在相同
clip norm `1.0`下分组时，scale/方向预计触发率为`.822139/.038557`，方向有效倍率中位恢复`2.653265 x`。因此下一fresh matched
parameterization保持Writer、coarse、K1、73个gradient tasks、rank、loss、LR与seed不变，只补两个同一scale--direction接口修正：

1. 完整per-target mobile `B@A` RMS固定不超过`1 x s_ref`；该倍率是对专家全局reference加cap要求的直接实现解释，并由上述
   fit-only/G1分布支持；
2. `scale_head`与其余全部Writer参数各自沿用原clip norm `1.0`，不改objective或task权重。

旧未限幅/global-clip运行继续到预注册macro1210，只提供其实际parameterization的相邻证据；无论结果如何，它都不能替代上述
修正版fresh训练，也不能单独停止整个positive-only Event-to-Factor Writer函数类。

## 109. Ordered Event的static-repeat合同修复

在scale/gradient-boundary修正后、fresh 73-task启动前，逐项对照专家澄清稿§5.3和§7.2发现首版
Ordered Event存在一个硬合同偏差：真实`frame_positions`未进入event encoder，可学习slot position却直接
进入event value，且slot-specific candidate logits同时选择relation value。CPU零梯度构造检查证明，完全重复的
static frames/policy-response/native evidence仍产生`.19244/.13996`的event/frame innovation RMS，并可在scale
打开后让4个构造target全部达到`.20` mobile RMS cap。

修正不改变模块边界或loss：observed relative position只路由emission/transition/attention QK，relation value在
slot间共享，并围绕frame-common中心化聚合。同一构造检查降至`7.23e-8/6.17e-8`的
event/frame innovation RMS，4个合成target mobile RMS最大`4.50e-5`，在浮点舍入范围内有效只返回carrier。
修正后task1/task93真实shared两步profile均自然完成，步耗时为`17.42/18.18s`与`40.28/30.95s`；
Frame/Event/Process/Composer梯度均finite nonzero，峰值allocated/reserved为`23.47/34.81GB`与
`33.30/41.56GB`，信息墙计数全部正确。这两条profile只验证真实图与资源，不使用两步内部数值做方法选择。
旧macro610/1210因同时缺少这项动态必要性、完整`s_ref`边界与独立方向梯度预算，只保留为旧实现的
正式证据；fresh matched才能裁决修正后的active Writer。

## 110. 旧scale macro1210终点与corrected matched启动

旧73-task scale formal从clean detached `df7a7f5a837192027ea4d9e8f566d0d2459e26df`自然完成1210 optimizer steps，
总/train/evaluation耗时为`35173.12/34855.78/170.97s`。macro610到1210的10个gradient task fit/held benefit均值从
`.000700/.000107`增至`.001419/.000761`，但全部视频优于carrier仍只有`6/10`；两个true-task-held task的fit/held均值从
`-.003674/-.003763`变为`-.003806/-.004313`，仍为`0/2`全部视频优于carrier。训练、Panel-B与物化均未读取wrong、shuffle、
reverse、held action/reward/state或validation/test信息。

macro1210使用每个held task固定correct demo5各调用Writer一次，物化唯一完整38-target rank16；materialized root为
`runs/outputs/pi05_ecp_policy_response_writer_scale73_coarse_m1210_held5_correct_k1_materialized_df7a7f5a_gpu02p4_20260903/`。
gpu02物理4/6上四worker完成20个cost-balanced shards和250个严格配对rows，return code全0；结果root为
`runs/outputs/pi05_ecp_policy_response_writer_scale73_coarse_m1210_held5_correct_k1_strict250_6ddceff5_gpu02p46_r2_20260903/`。
总分`30/250`，Long/Goal/Object/Spatial0/Spatial9=`0/0/2/27/1`，breadth`3/5`。相对macro610 retained/gained/lost为
`18/12/8`、Jaccard `.473684`、paired exact `p=.503445`；相对carrier43为`26/4/17`、净丢13、Jaccard `.553191`、
`p=.007197`；相对旧coarse macro70同样净丢13、`p=.010622`。五套adapter的mobile/carrier聚合函数范数为
`1.7148/2.1191/2.0846/2.4009/2.4364 x`，高于macro610的`1.5812/1.8516/1.8771/1.9648/1.9679 x`。
故延长训练只带来不稳定的小幅回升，没有恢复carrier或Goal/Long breadth。

corrected matched formal随后从clean pushed detached `aed7b5244cce91df440d0d4a453b1e3b1be8a346`于2026-09-03
10:27 CST启动，使用gpu01物理2/4/5/6、world-size4、同一73-task/coarse/K1/component-init/schedule与positive-only loss；相对旧arm只
增加完整per-target mobile `B@A` RMS cap、scale/direction独立gradient clip和static-repeat动态必要性修复。前两个optimizer step已完成，
全部模块梯度finite nonzero且资源稳定。旧formal detached worktree随后删除，commit与全部formal evidence保留。

## 111. owner锁定full-only与corrected full task-local容量

owner于2026-09-03明确否决coarse/final-layer horizon mean作为active路线：Action Expert的50个relative horizon positions必须完整保留到
task/relation-conditioned learned read，旧coarse实现与结果只作历史审计。收到边界后，当时运行的corrected coarse 73-task shared与
task1/task93 task-local分别在step121/47/29主动中止；均未完成预注册checkpoint评估，不形成新的科学裁决。active config、CLI、
materialization与static adapter provenance随后只接受full，horizon-mean forward被删除；当前formal checkpoint生命周期结束前仅保留一个
不参与forward的旧embedding以维持RNG/state-dict兼容。

同一三项接口修正随后以full fresh运行。task1与task93正控分别从clean pushed detached
`3a342b6ca2d2d88dddc380e4ae943fd28d00bba9`在gpu02物理4/6完成110步、macro70/110与每checkpoint三条正确视频的零梯度Panel-B：

| task/checkpoint | fit recovery | held-video recovery | all videos above carrier |
|---|---:|---:|---:|
| task1 macro70 | `.207146` | `.169391` | yes |
| task1 macro110 | `.223986` | `.157630` | yes |
| task93 macro70 | `.417139` | `.409146` | yes |
| task93 macro110 | `.425338` | `.418759` | yes |

task1 root为
`runs/outputs/pi05_ecp_policy_response_writer_corrected_tasklocal_task1_full_s110_7d435ea3_v1_gpu02p4_20260903/`，task93 root为
`runs/outputs/pi05_ecp_policy_response_writer_corrected_tasklocal_task93_full_s110_7d435ea3_v1_gpu02p6_20260903/`；两者result/completion与
launcher exit0完整，held/Panel-B backward为0，source policy冻结且输出仍为唯一完整rank16。相对旧full，task1 recovery有所减弱，
task93 held则从`.300885/.280724`提高到`.409146/.418759`。因此三项修正不是对所有任务统一缩小update，full task-local跨视频容量仍然
明确；最早未解决接口继续是shared task-disjoint mapping，而不是允许回退coarse。

## 112. full native-bank horizon合同修复

在corrected full 73-task shared运行期间继续逐层核对owner新边界，发现Process的full response前端虽然保留每owner的
`50 horizon x 8 channel` tokens，Composer的辅助bank context仍在input/output keys上调用`.mean(2)`，该维正是Action Expert
horizon。最终signed pooling仍逐horizon处理真实native values，但rank query的bank read已提前丢失horizon，因此不满足active design
“native bank不得为方便提前平均probe或horizon”的合同。对应formal root
`runs/outputs/pi05_ecp_policy_response_writer_corrected_73task_k1_component_full_s1210_7d435ea3_gpu01p2456_20260903/`在optimizer
step156/effective146主动停止；没有macro checkpoint，不能作为full函数类的科学裁决。

修复提交`89833c235dea8df069418642e13400587dede385`删除两处horizon mean。现有process-conditioned rank query直接读取完整
frame/probe/horizon/bank-type key集合；实现用与dense MHA等价的chunked online-softmax累计分子、分母和最大值，并在反向重算attention
激活，避免百万token的连续副本。每个frame chunk内的output groups只做保持group/probe/horizon/type各轴的向量化projection，不做
均值、抽样或近似。dense/streaming等价、完整token计数、梯度、static-repeat与既有Writer/static-adapter定向测试共`18 passed`。

真实task93 smoke覆盖最长fit demo3的87个stride-5 frames、19 layers、2 probes、50 horizons、38-target native X/Y、functional与
process backward及唯一76-tensor rank16 materialization；峰值allocated/reserved为`42.42/47.14GB`，无OOM/NaN。单卡两步shared
profile分别使用87/79帧视频，用时`33.29/28.54s`，Frame/Event/Process/Composer梯度均finite nonzero。该profile只证明完整horizon
实现可训练；下一科学证据必须来自该提交合并推送后的fresh 73-task formal和held5 closed-loop。

## 113. 完整horizon 12-task短资格的稳定non-pass

owner拒绝在架构尚无闭环增量时直接付出约10小时73-task训练后，原73-task full-horizon fresh run于optimizer25/effective15安全停止，
未形成checkpoint或科学裁决。相同clean detached `e7278f1b22176b53025166dc5015a4463b819ecd`随后运行5 meta + 5 target梯度任务、
task2/74 true-task-held、K1、component-init、10 warmup + 100 effective updates。正式root为
`runs/outputs/pi05_ecp_policy_response_writer_full_horizon_12task_k1_component_s110_e7278f1b_gpu01p2456_20260903/`；110步自然完成，
train/total wall为`4212.19/4875.42s`，macro70/110 checkpoints、optimizer/rank state、Panel-B、contracts与completion均完整。

macro70的gradient fit/held benefit均值为`.000891/.000924`、`8/10` task全部视频优于carrier；true-task-held fit/held为
`-.000535/.000153`、`0/2`全视频优于。macro110 gradient为`.001283/.000949`、仍为`8/10`；true-task-held为
`-.000131/-.000135`、仍为`0/2`。因此Writer能在有梯度任务上形成小幅功能改进，却没有得到task-disjoint共享映射。

两个checkpoint均用held5每task固定correct demo5一次物化唯一完整38-target rank16，并各完成250条严格配对closed-loop：

- macro70 materialization：
  `runs/outputs/pi05_ecp_policy_response_writer_full_horizon_12task_m70_held5_correct_k1_materialized_e7278f1b_gpu02p4_20260903/`；
  strict root：
  `runs/outputs/pi05_ecp_policy_response_writer_full_horizon_12task_m70_held5_correct_k1_strict250_e7278f1b_gpu02p4_r3_20260903/`；
  `35/250`，Long/Goal/Object/Spatial0/Spatial9=`0/0/1/32/2`，breadth`3/5`，相对carrier retained/gained/lost=`32/3/11`。
- macro110 materialization：
  `runs/outputs/pi05_ecp_policy_response_writer_full_horizon_12task_m110_held5_correct_k1_materialized_e7278f1b_gpu01p2_20260903/`；
  strict root：
  `runs/outputs/pi05_ecp_policy_response_writer_full_horizon_12task_m110_held5_correct_k1_strict250_e7278f1b_gpu01p2_r3_20260903/`；
  `35/250`，逐task=`0/0/0/33/2`，breadth`2/5`，相对carrier=`31/4/12`。

两点总分相同、Goal/Long均为0且稳定低于carrier `43/250`。所以失败不是正式`>145/400`门槛太高，而是当前方法连较低的carrier保持与
跨suite breadth也未达到；不追加同构训练、不恢复73-task规模、不进入mixed-K/random Final，也不为失败checkpoint运行negative controls。
该结果只停止当时“relation已混合的frame innovation -> shared signed candidate logits”参数化，不否定task-local Composer、full
policy-response、ordered events、native X/Y、signed pooling或rank4。

## 114. full Writer训练吞吐的8.48倍优化

同一4卡、10-step、旧12-task资格schedule的逐层profile固定科学task group、权重、K1、functional rows和完整50-horizon。原始
streaming实现为`34.394s/step`；只加入有限期schedule上的outcome-independent CPU evidence cache选择性复制与动态task placement后为
`26.306s`。再将完整bank context在显存允许时改为dense exact SDPA、超限时以bounded fused blocks执行同一online softmax，并融合
整视频frame与output groups的exact signed pooling后为`8.699s`。最终将frame construction提高到整视频、streaming block设为128K tokens、
functional microbatch按可用rows提升到4，10步平均/中位/最小/最大为`4.054/4.175/3.436/4.872s`，相对原始实现提速`8.48x`。

最终4卡累计task计算占总device wall约`78.0%`；最长task单步约`4.54s`，其余rank在典型首步约`3.69--4.13s`，剩余尾部主要来自
task不可切分而非固定cache owner。实时dmon显示计算段各卡经常达到`78--100%` SM，最长视频卡显存峰值约`44.24GB allocated / 46.97GB
reserved`。所有dense/streaming、chunk/fused、output-group与K2 unequal-video输出/梯度等价测试通过；没有平均、抽样或删除任何
frame/probe/horizon/bank-type/raw X/Y。执行器只改变已选task的设备位置，每task仍执行一次；meta/target比例和每step task数由实验配置
决定，不固定`3+3`或6。ZeRO-1/2不对约347万可训练Writer参数和task-level尾部对症；当前不引入FSDP或Writer/Policy流水线复杂度。

## 115. Process--Composer显式event-relation绑定缺口

在上述相邻non-pass后逐字段对照专家澄清稿§5.3--5.4与实现：`PolicyResponseProcessOutput`确实产出
`assignment=alpha(e,t,m)`，但Composer没有任何`.assignment`读取；它只收到四类relation按单一概率混合后生成的
`frame_innovation(t,j)`。因此relation type虽然参与Process内部压缩，却未作为显式轴到达最终native candidate scorer，违背专家列出的
Composer输入合同。

该缺口不能仅解释为12-task组合覆盖不足。零梯度target authority task74为“pick up the black bowl on the ramekin and place it on the
plate”；三个有梯度target tasks 72/73/75使用同一verb、object和goal，仅把初始关系改为table center、top drawer或stove。task74仍无
稳定功能增量，正好符合“初始scene relation在进入signed scorer前被混合”的失效模式。

下一matched correction已登记为：直接用既有`D(e,j)`与`alpha(e,t,m)`形成显式frame x relation innovation，让relation type经一个共享
learned projection/FiLM调制后参与正负candidate logits；raw X/Y仍是唯一vector value，四relation等base mass，最终exact pooling覆盖完整
frame x relation x probe x 50-horizon x bank-type。relation只调制bias-free D路径，故`D=0`仍严格zero mobile。除此之外不改loss、task
split、rank、scale、carrier、materializer或部署输入；真实smoke后立即以同一短资格和held5相邻closed-loop裁决。

## 116. 显式event-relation scorer接通及最长视频工程准入

从clean pushed `main@ca623dbfe6aab9ca2a021d8d0456570a8cebc7be`建立唯一
`codex/policy-response-writer-relation-binding`实现分支。Composer现在直接读取`assignment=alpha(e,t,m)`，与既有
`D(e,j)`形成`I(t,m,j)=sum_e alpha(e,t,m)D(e,j)`；四个relation各占`1/4`固定base mass，relation embedding只乘性调制
bias-free动态路径。raw X/Y、完整frame x probe x 50-horizon x bank-type candidates、rank12 carrier + rank4 mobile、唯一38-target
rank16 materialization及positive-only目标均未改变。新embedding在forked RNG作用域初始化，已用独立同seed进程确认全部旧Composer
参数初始化逐tensor完全不变。

最初同时展开relation x native-token x hidden-width的实现，在gpu01物理1、task93 fit demo3的Writer chain-rule backward出现工程OOM；
当时卡上仅余约36MiB，不形成科学non-pass。实现随后保持同一数学分布，逐relation计算logits并用`logaddexp`做精确log-space
marginal，同时在梯度路径重算scorer激活；没有平均、抽样或删除relation、frame、probe、horizon与native value。

修正后的真实smoke覆盖task93 demo3的429 raw / 87 stride-5 frames、38 targets、完整50-horizon、functional leaf gradient、两次
Writer chain-rule backward、causal process backward及76 tensors唯一rank16 materialization，自然exit0。最大CUDA
allocated/reserved为`40178697216/47244640256 bytes`；Frame/Event/Composer与process prediction对应梯度均finite，static-repeat和
显式assignment置换定向测试通过。随后task93两步task-local profile的稳态step为`8.6865/8.7940s`，最大allocated/reserved为
`34309713920/46785363968 bytes`，第二步input/output branch与task query梯度均已打开；两步后的fit/held内部诊断不作科学裁决。
该证据只准入同一12-task短资格，shared功能与closed-loop结果仍未知。

## 117. relation-summed与event-measure两次matched shared裁决

relation-summed版本从clean detached `33ee1330` 完成12-task K1 component-init、110步训练与macro70/110全部
Panel-B。两点gradient task fit/held benefit为`.000995/.001146`与`.001305/.001007`，task2/74 true-held均为负。
held5 correct-only strict250为`42/34`，breadth均`3/5`、Goal/Long均为0，macro70到110为`16 lost/8 gained`。因此
`I(t,m,j)=sum_e alpha(e,t,m)D(e,j)`这一“先求event期望、后非线性score”接口正式non-pass。

后续matched版本从clean detached `a049f61e17ad9e5eae55b67b8de7be4aa686bfc9` 运行：以`log alpha(e,t,m)`作为
event x relation candidate base measure，用未求和`D(e,j)`生成bias-free动态logit；其他数据、K1、full 50-horizon、
positive-only loss、carrier12 + mobile4与唯一rank16输出均不变。formal训练root为
`runs/outputs/pi05_ecp_policy_response_writer_event_measure_12task_k1_component_s110_a049f61e_gpu01p0156_cache8g_20260903/`；
110步、两枚checkpoint、Panel-B、optimizer/rank state、contracts和completion均完整，train/evaluation/total wall为
`1109.73/576.16/1725.81s`。

macro70的gradient fit/held benefit为`.001329/.001114`、recovery `.11773195/.08567050`、`9/10`任务全视频优于carrier；
macro110为`.001617/.001701`、`.13651004/.13192627`与`9/10`。task2/74 true-held在macro70为
`-.001575/-.001798`，macro110为`-.002420/-.001998`，两点都是`0/2`任务全视频优于carrier。

macro70物化root为
`runs/outputs/pi05_ecp_policy_response_writer_event_measure_m70_held5_correct_k1_materialized_a049f61e_gpu02p6_20260903/`，
strict250 root为
`runs/outputs/pi05_ecp_policy_response_writer_event_measure_m70_held5_correct_k1_strict250_a049f61e_gpu02p46_r3_20260903/`；结果
`40/250`，Long/Goal/Object/Spatial0/Spatial9=`0/0/2/36/2`，breadth`3/5`。macro110对应物化root为
`runs/outputs/pi05_ecp_policy_response_writer_event_measure_m110_held5_correct_k1_materialized_a049f61e_gpu01p0_20260903/`，
strict250 root为
`runs/outputs/pi05_ecp_policy_response_writer_event_measure_m110_held5_correct_k1_strict250_a049f61e_gpu01p0156_r3_20260903/`；结果
`42/250`，逐task=`0/0/2/37/3`，breadth`3/5`。两点间retained/gained/lost=`33/9/7`、Jaccard `.673469`、
paired exact `p=.803619`；macro110相对carrier43为`35/7/8`、Jaccard `.700000`、`p=1.0`。两次评测的18个worker均
exit0，各自250条paired rows和launcher completion完整。

这两枚历史可比checkpoint的名称来自J2的记账：optimizer 1--10为warmup，global 70/110分别是post-warmup
effective 60/100；它们不是架构推导的最优步数。event-measure在已训练task上的functional收益从70到110继续小幅上升，
而true-held恶化、闭环几乎不变且Goal/Long始终为0，因此失败不用“刚好停早了”解释。该结果停止当前event-measure
matched接口的同构续训、mixed-K、fully-random和negative controls；它不否定task-local Composer、full policy-response、ordered
events、native X/Y、signed pooling或rank4。专家失败映射将下一问题定位为task-disjoint shared mapping/credit/
identifiability与自然task组合覆盖，应先做factorial coverage audit再决定最小扩展映射，不盲目恢复73-task长跑。

## 118. event-measure训练与评测运行时证据

同一event-measure formal显式给出`8GiB`只读cache replica预算后，110步训练wall均值/中位/p95为
`10.056/10.044/10.643s`，相对漏传预算的前6步可丢弃基线`13.624s`快`26.2%`；rank load `max/mean`
从`1.619`降到`1.216`，gpu01物理0/1/5/6训练稳定段平均SM为`88.2/88.5/89.4/90.2%`。m110 strict250以
4 GPU x 3 replicas x 8 envs在`1040.52s`内完成250条，rollout execution window `815.24s`；满载显存段四卡平均SM约
`90.1--94.1%`。该执行证据与科学non-pass分开：运行面已可复用，不因当前架构负结果回退exact/full优化。

## 119. 自然task factorial coverage与扩展短资格合同

在event-measure 12-task相邻non-pass后，先按专家失败映射做只读factorial coverage审计，而非直接恢复73-task长跑或继续修改网络。
脚本与配置由clean pushed `0b9450f94409a366b5b0fb700ddbd92e058652a0`固定，正式root为
`runs/analysis/pi05_ecp_policy_response_writer_factorial_coverage_v1_20260903/`。审计只读取固定split、language、task metadata与
`libero90_nonheld_meta_v1`人工登记的process contrast groups；不读取视频pixels、actions、states、reward、model outputs或
evaluation outcomes。仓库状态在报告中为clean。

95-task authority由71个审计non-held meta与24个target-train tasks组成。新实验使用其中全部55个meta-fit与18个target-fit产生梯度，
task2与task74仍为零梯度true-task-held。7组同语言跨场景组合覆盖17 tasks，其中5组至少有两个gradient tasks、4组形成
gradient-to-held bridge。人工protocol的same-language/same-procedure/order-or-relation三类contrast分别有`5/9/5`组train pair与
`3/7/3`组held bridge。task2有exact-language peer23；task74有同“black bowl -> plate”verb/object/goal而初始scene relation不同的
72/73/75；held5 Spatial、Object与Long也都有自然component重组依据。唯一明确缺口是held Goal task25的`push plate`procedure没有任何
Writer-gradient task以`push`开头。

因此该数据并非完全欠识别，足以运行一次扩大task-disjoint mapping的有信息量正样本实验；但metadata本身不证明video-dependent最优
adapter已经可学，Goal闭环失败也不能与覆盖缺口混为一谈。新配置保持full 50-horizon、K1、component-init、event-measure、
positive-only objectives、真实native X/Y、signed pooling与唯一rank16不变，只扩大gradient-task覆盖。每update显式采样
`9 meta + 3 target`，因为`9/55`与`3/18`使73个tasks近似等权；这只是当前配置，不固定未来role比例或batch size。
optimizer step200/400对应post-warmup effective190/390，每task分别获得`32--34/65--67`次暴露；400点对齐旧12-task短资格的约
66次/task。故新节点按实际数据暴露量产生，而不是把历史J2的70/110当成理论门槛。正式训练前先用六卡、12-task/update做两步真实
profile；若资源与图稳定，再启动预计小时级而非十小时级的短资格，并在200/400两点直接运行Panel-B与held5 correct-only strict250。

## 120. factorial Writer四卡执行profile

2026-09-03 20:39 CST从clean pushed detached `248d3efa56236986eaa28d8124abbb6f6e74157c`在gpu01物理0/1/5/6启动
73-task、12 tasks/update、full K1 event-measure两步profile。launch前四卡显存为0且util为0；gpu01物理2/3/4由他人约98%利用，
gpu02虽有四张近空闲卡，但其它卡已有4.7--31GB占用，不能安全容纳full最长样本峰值，因此没有跨节点拼碎片或干扰他人。
`/data1` quota blocks为`774956448/1073741824KiB`，limit `1084227584KiB`。正式root为
`runs/outputs/pi05_ecp_policy_response_writer_factorial_73task_k1_component_profile2_248d3efa_gpu01p0156_cache8g_20260903/`。

两步optimizer wall为`10.5972/9.2597s`，每步12 tasks都由cost-balanced placement分成每rank 3 tasks。第一步四rank预测cost为
`123/101/113/114`，第二步为`113/118/131/122`；第二步Frame、Event、Process prediction、Composer、relation与scale梯度均finite
nonzero。峰值allocated/reserved为`27368414208/36937138176 bytes`。146个唯一fit-video frozen evidence合计
`105020606660 bytes`；8GiB replica预算仅选择task61与task85的3个确有收益副本，额外`3111731600 bytes`，两步预测总cost由
272降至254，理想为234。两步训练wall为`19.97s`，从cache/normalizer准备开始的total为`150.20s`；进程冷启动另有约4分钟冻结
model/data加载，后续同节点运行可复用OS页缓存。

该profile故意使用2 functional rows，只证明73-task cache、12-task调度、完整50-horizon图、反向和资源安全，不能冒充formal
rows16吞吐。结合此前同实现rows2与rows16的真实差值，四卡正式400步保守预计约2--2.5小时，仍远低于已否决的十小时级探索。
因此无工程阻塞，下一步是保持同一科学配置从clean pushed detached authority启动400步资格，并在optimizer200/400进行预登记
Panel-B及随后held5 correct-only strict250。

## 121. factorial Writer正式短资格启动

2026-09-03 20:49 CST，full K1 event-measure component-init扩展资格从clean pushed detached
`5534cb140b90ac20e9143dd20a7ed8e11c539f19`在gpu01物理0/1/5/6、world-size4 fresh启动。tmux为
`ember_prw_factorial_s400`，formal root为
`runs/outputs/pi05_ecp_policy_response_writer_factorial_73task_k1_component_s400_5534cb14_gpu01p0156_cache8g_20260903/`。
launch前同步检查gpu01与gpu02：所选四卡均0MiB、0% util；gpu01物理2/3/4由他人98--100%使用，gpu02除四张近空闲卡外没有
另外两张能安全容纳full最长样本，故不跨节点拼卡或干扰他人。gpu01 available host memory约`269.2GB`；`/data1` quota blocks为
`774957644/1073741824KiB`、limit `1084227584KiB`，输出root不存在，预计新增远低于余量。

正式命令固定`NCCL_P2P_DISABLE=1`、GPU-local NUMA、8GiB outcome-independent frozen-evidence replica预算、55 meta + 18 target
gradient tasks、task2/74 true-task-held零梯度、每update 9 meta + 3 target、functional rows16/microbatch2、10 warmup + 390
effective、optimizer200/400 checkpoints、positive-only loss与唯一rank16。训练不读取wrong、shuffle、reverse或validation/test
outcomes。若m200形成时gpu02仍有两张安全卡，将在总EMBER卡数不超过6的边界内并行运行m200 held5 correct-only strict250；这只
提前获取已预登记checkpoint的闭环结果，不改训练、checkpoint选择或m400自然完成。

## 122. Panel-B任务均衡与microbatch实测

factorial profile的73-task base ownership为每rank 18/19/19/19 tasks，但12个预登记Panel-B tasks在其中实际落成
`2/4/5/1`；functional evaluation每task都有3条视频 x 16 visits x 16 rows，因此该布局会让两checkpoint持续等待5-task rank。
提交`e74b653961e6d7bf088348f88d95eeba95b74921`把Panel-B执行ownership与training cache owner解耦：只用每task三条视频的
sampled-frame总数，加每视频固定`4 * functional_rows * evaluation_visits`输入成本，再调用既有deterministic LPT。当前12-task/
4-rank合同稳定为`3/3/3/3`；task集合、视频、rows、visits、模型和数值均不变，result新增明确的
`evaluation_task_ownership` provenance。Writer定向测试`23 passed`。该优化晚于正在运行的`5534cb14` frozen formal，不会中途
改变它。

为判断是否还应增大evaluation policy microbatch，gpu02物理0/1并行运行同task93、两步相同profile图；训练仍因rows2实际使用
microbatch2，只有随后三条视频各一次16-row Panel-B分别使用2与8。microbatch2 root为
`runs/outputs/pi05_ecp_policy_response_writer_panelb_mb2_task93_profile2_e74b6539_gpu02p0_20260903/`，evaluation为
`11.6923s`；microbatch8 root为
`runs/outputs/pi05_ecp_policy_response_writer_panelb_mb8_task93_profile2_e74b6539_gpu02p1_20260903/`，为`11.5774s`，只快
`.98%`。两者exit0；因收益远小于运行波动且更大batch只增加峰值风险，正式保留microbatch2，不继续扫16。

## 123. node-local单份frozen evidence mmap

73-task frozen policy-response evidence的146条fit videos合计`105020606660 bytes`。rank-private cache即使只保存一份总数据，也会把
每个task固定在持有它的rank；8GiB选择性复制只能为少数task购买额外eligibility。新执行面把每个task/video原子保存为一份
safetensors文件，同节点所有torchrun ranks mmap同一物理页，于是每步可在全部ranks间做exact cost-balanced assignment。它完整保留
原tensor、frame/probe/50-horizon/bank-type、task group、权重、K1、functional/process loss与唯一rank16；cache不是checkpoint或
scientific evidence，成功后自动清理。

同一clean detached candidate `a2e40700762e8f8ff7e7ed5aea29ce6c8b1cc972`在gpu02物理0/1完成三条7-step matched profile；三者的
84个task、video demo、Panel visit、functional RNG、causal cutoff与weight逐项相同：

| cache布局 | train wall | mean step | max step | mean rank load gap | peak allocated/reserved |
|---|---:|---:|---:|---:|---:|
| private，0GiB replicas | `150.039s` | `21.3745s` | `28.9829s` | `8.678s` | `39.976/46.787GB` |
| private，8GiB replicas | `130.139s` | `18.5403s` | `26.2068s` | `3.122s` | `39.976/46.785GB` |
| shared mmap，0 physical replicas | `124.870s` | `17.8110s` | `19.8142s` | `.338s` | `39.975/46.964GB` |

shared相对当前8GiB路径平均快`4.05%`、最坏step快`24.39%`，实际rank load `max/mean`从`1.0897`降至`1.0096`；相对0GiB
平均快`16.78%`。短两卡实验低估了四卡长schedule收益：对正在运行的rows16 formal optimizer20起126步，以每task真实耗时重做
全eligibility assignment，平均wall从`23.441s`估为`17.955s`；再计入两卡观测的约3% mmap单task开销约为`18.4s`，仍约快`21%`。
首次105GB capture/build与private冷启动同量级；cache-hit profile的`total_seconds`不用于该比较。最终实现修复短profile只覆盖部分
task时的planner inventory，并把原数值等价测试的随机输入纳入forked fixed RNG；Writer定向套件`25 passed`。因此后续fresh同节点
多卡Writer训练采用shared mmap；已冻结的`5534cb14` formal不改变cache、world topology或执行分配。

## 124. 73-task factorial资格完成但名义rank4实际坍缩为近rank1

`5534cb140b90ac20e9143dd20a7ed8e11c539f19`的full K1 component-init formal自然完成400 optimizer steps、macro200/400、
两点Panel-B和两次held5 correct-only strict250。训练root为
`runs/outputs/pi05_ecp_policy_response_writer_factorial_73task_k1_component_s400_5534cb14_gpu01p0156_cache8g_20260903/`；
train/evaluation/total wall为`9000.24/757.08/9894.15s`，completion、result、两枚checkpoint、400条metrics与全部rank状态完整。

macro200/400的gradient task Panel-B fit/held benefit为`.00074035/.00031607`与`.00102278/.00054707`，recovery为
`.09587/.06039`与`.15516/.13105`，两点各有`7/10`任务的全部视频优于carrier；但task2/74 true-task-held均值分别为
`-.00233354/-.00173280`与`-.00232097/-.00208995`，两点仅`1/2`任务全部视频为正。m200物化与strict250 roots为
`runs/outputs/pi05_ecp_policy_response_writer_factorial_m200_held5_correct_k1_materialized_5534cb14_985265ae_gpu02p0_20260903/`与
`runs/outputs/pi05_ecp_policy_response_writer_factorial_m200_held5_correct_k1_strict250_5534cb14_985265ae_gpu02p01_r3_20260903/`；
结果`30/250`，Long/Goal/Object/Spatial0/Spatial9=`0/0/3/27/0`、breadth`2/5`。m400对应roots为
`runs/outputs/pi05_ecp_policy_response_writer_factorial_m400_held5_correct_k1_materialized_5534cb14_985265ae_gpu02p1_20260903/`与
`runs/outputs/pi05_ecp_policy_response_writer_factorial_m400_held5_correct_k1_strict250_5534cb14_985265ae_gpu02p01_r3_20260903/`；
结果`32/250`，逐task=`0/0/1/30/1`、breadth`3/5`。m200到m400为`20 retained/12 gained/10 lost`、Jaccard `.476190`、
paired exact `p=.831812`；m400相对carrier43为`27/5/16`、Jaccard `.5625`、`p=.026604`。因此扩大task覆盖和把每task exposure
近似加倍都没有恢复carrier或Goal/Long，不能再归因于m200停早；functional proxy继续改善却未迁移到task-disjoint闭环。

冻结activation诊断位于
`runs/analysis/pi05_ecp_policy_response_writer_factorial_m200_rank_collapse_task18_5534cb14_985265ae_20260903.json`与对应m400文件。
task18的Process common约`67`范数，owner/language约`11`，四个rank query仅约`1`；直接相加后m200/m400各target从初始query到第二个
Composer block的rank centered/mean RMS均约`1.1%`，末端pairwise cosine分别为BF16记录的`1.0`与约`.99983`。m200的
q/v/action-in/action-out有效update participation rank为`1.0006/1.0008/1.0013/1.0062`，m400为
`1.0001/1.0011/1.0006/1.0011`。额外200步没有修复名义rank4的实际近rank1坍缩，最早失效接口是rank/shared query seed的
数值尺度，而不是rank4理论容量或末端幅度。

## 125. parameter-free rank/shared query balance进入fresh因果裁决

m200冻结反事实
`runs/analysis/pi05_ecp_policy_response_writer_factorial_m200_rank_collapse_task18_normalized_add_5534cb14_985265ae_20260903.json`
分别对rank context和shared task context作LayerNorm后再相加。它把初始rank centered/mean RMS恢复到`.83--.84`、第二个block后保留
`.60--.65`；q/v/action-in participation rank恢复到`1.364/1.077/1.190`，rank posterior TV从约`.01--.02`增到约`.12--.18`。
action-out仍只有`1.010`，所以这只是授权fresh训练的因果证据，不是结果替代。

提交`3e589695779be4a78d5f5bde6059e39e178bd146`在Composer唯一query seed处实现同一parameter-free balance：rank context与
owner+language+Process common shared context各自LayerNorm后相加。没有新增参数、block、loss、正交/熵/rank regularizer、solve或
第二adapter；full 50-horizon、真实X/Y、event-measure、signed pooling、per-target cap、positive-only目标和唯一rank16不变。
26项Writer测试、真实task1 forward/functional/process gradient/materialization smoke与task1/task93两步shared profile均通过。

2026-09-03 23:52 CST从clean pushed detached `3e589695`在gpu01物理0/1/5/6启动fresh 73-task、400-step shared资格，root为
`runs/outputs/pi05_ecp_policy_response_writer_rank_balance_73task_k1_component_s400_3e589695_gpu01p0156_sharedmmap_20260903/`。
第一次launcher在任何output/cache/scientific step产生前因错误地把整个torchrun父进程固定到NUMA0，使物理5/6无法取得本地CPU集合而
被仓库guard拒绝；撤掉外层绑定后，各rank由现有runtime分别正确绑定为GPU0/1 -> NUMA0、GPU5/6 -> NUMA1，未绕过检查或改科学代码。
正式run contract确认commit clean且等于origin/main、world4、full/positive-only、无Action Meta、无negative reads。105020606660-byte
node-local单份mmap cache已完成；最初三个optimizer steps约`17--18s`，每步四rank wall完全对齐，相比旧formal末段约`23s/step`
符合预期提速。

旧m400评测释放gpu02物理0/1后，第一次task1/task93 launcher复用了只定义shared pool的factorial配置，两个进程在生成run contract、
checkpoint或optimizer step前由`task_local_positive_control`缺失而退出；这不是科学结果。提交
`8bdd9595e94cb9f7893b6192f05a98a85551159e`只在同一active factorial配置中补齐`[1,93]`正控声明，config loader与26项Writer测试通过，
模型、数据、loss和shared run均不变。2026-09-04 00:09 CST从该clean pushed detached authority重启后，两条任务又在相同的step 0前
以`KeyError: functional_panel_config`退出：task-local formal contract仍把旧单一panel config路径硬编码为provenance，而实际runtime
已正确从两个completed roots解析task-specific panel records。两次失败均未生成output root、run contract、checkpoint、capture或
optimizer step，因此不是科学non-pass。

提交`89ca865d`把task-local run schema升为v2，并直接记录`runtime.panels[task]`实际resolved record的task、path与bytes；它同时兼容
旧单config和当前multi-source roots，而且比记录间接配置入口更准确。该修复不改变Writer forward、训练数据、loss、optimizer、步数或
任何信息墙，新增回归测试后27项Writer测试全部通过，factorial配置中task-local正控、reference、全部optimization字段和两个panel
sources也已完整解析。

2026-09-04 00:19/00:21 CST，两条fresh正控从当时最新clean pushed detached `ef00f446`依次启动。launch前同时live检查两节点：
gpu01物理0/1/5/6为同一shared formal；gpu02物理0/1分别仅`209/162MiB`、util 0%，各有同一他人gqma约`186/148MiB`
低占用进程。先让task1生成有效v2 contract，再复查物理1仍为`162MiB/0%`后启动task93，因此总EMBER物理卡恰为6且未触碰其它
高负载卡。gpu02 available host memory约`341505284KiB`；`/data1` quota为`877711568/1073741824` blocks，两条同构小run的预计
增长远低于余量。task1 root为
`runs/outputs/pi05_ecp_policy_response_writer_rank_balance_tasklocal_task1_full_s110_ef00f446_gpu02p0_20260904/`，task93 root为
`runs/outputs/pi05_ecp_policy_response_writer_rank_balance_tasklocal_task93_full_s110_ef00f446_gpu02p1_20260904/`。两份contract均锁定
v2 schema、commit、full、component-init、110 steps、实际task panel、Action Meta false与single-process topology；两task都已进入真实
optimizer且input/output/relation/scale/task-query梯度全部finite nonzero。当前只等待macro70/110 Panel-B，不用启动期loss选模型。

## 126. rank-balanced task-local正控保留容量并显著提速

两条`ef00f446`正控均自然完成110步及macro70/macro110的三视频、16-visit Panel-B复评。task1在macro70的fit/held recovery为
`.222388/.115343`，macro110为`.328291/.228193`；task93分别为`.304725/.311541`与`.357007/.322264`。四个
task-checkpoint组合的两条fit和第三条zero-gradient held视频全部自发优于carrier。两份result均确认held、Panel-B、wrong backward
为0，source policy与Process可训练参数为0，Action Meta未安装，输出为唯一完整rank16。

相较同数据/目标的前一corrected task-local，task1 macro110由`.223986/.157630`提高，task93则由`.425338/.418759`下降；
因此parameter-free balance不是对所有task的单调数值增益，但已排除“恢复rank identity会破坏Composer容量”，并显示至少一个task的
跨视频功能恢复得到实质改善。更重要的是它不回答task-disjoint shared映射是否成立，后者只由当前fresh m200/m400闭环裁决。
工程上，task1训练/总wall从`1815.64/2016.31s`降至`629.36/812.76s`，task93从`2853.88/3076.06s`降至
`876.82/1061.80s`；20秒采样中shared四卡平均SM为`81.5/83.6/83.9/82.2%`，task-local两卡为`83.6/89.0%`。

## 127. rank-balanced shared m200显著修复旧版但尚未超过carrier

fresh `3e589695` shared训练在继续前往m400时，并行从m200为held5每task固定correct demo5物化一次唯一完整rank16。物化root为
`runs/outputs/pi05_ecp_policy_response_writer_rank_balance_m200_held5_correct_k1_materialized_3e589695_gpu02p0_20260904/`；
manifest确认full、K1、outcome-independent、无Action Meta、validation/test action/reward/state reads为0，且每condition只有一次
Writer调用。correct-only strict250 root为
`runs/outputs/pi05_ecp_policy_response_writer_rank_balance_m200_held5_correct_k1_strict250_3e589695_gpu02p01_r3_20260904/`；
22/22 shards、250/250 rows与6/6 persistent workers完整，结果`45/250`，Long/Goal/Object/Spatial0/Spatial9=
`0/0/4/38/3`、breadth`3/5`。

相对旧raw-query m200的`30/250`，新m200为`23 retained/22 gained/7 lost`、Jaccard `.442308`、paired exact
`p=.008130`；相对旧m400的`32/250`为`26/19/6`、Jaccard `.509804`、`p=.014633`。这证明parameter-free
rank/shared query balance产生了显著真实闭环恢复，而非只改善内部几何。相对stable carrier `43/250`则只有
`33 retained/12 gained/10 lost`、Jaccard `.600000`、`p=.831812`：Spatial0同为38但内部`8 gained/8 lost`，Object同为4，
Spatial9由1到3，Goal/Long仍均为0。因此m200是部分shared信号，不是路线通过，也不授权负controls或Final扩展。

只读对已物化A/B做small-core谱分析时，旧m200四family参与秩中位均约1；新m200的q/v/action-in/action-out中位约
`1.164/1.218/1.050/1.006`，说明q/v rank容量已经真实打开，但action-out仍近rank1。其action-out mobile update范数仅约
`.006--.008`，而G1四个非零成功正控约`.091--.120`且参与秩约`2.16`；这只是定位线索，因为functional等价方向不必复现G1
factor。m200评测释放gpu02物理0/1后，在launch前同时live检查gpu01/gpu02；前者仍为本shared四卡，后者两卡仅有
`209/162MiB`低占用且util 0%。随后并行启动Object18与Long36的正确视频activation诊断，训练继续到m400，不读取wrong、
shuffle或reverse。

## 126. m200正确样本几何与functional VJP把残余瓶颈定位到共享family readout

rank-balanced m200的三条冻结activation证据位于
`runs/analysis/pi05_ecp_policy_response_writer_rank_balance_m200_rank_geometry_task18_3e589695_20260904.json`、对应
`task25`与`task36`文件；分别覆盖held Object、Goal、Long的固定correct demo5。三者均为diagnostic-only、零梯度、零checkpoint
selection且不读取negative。初始rank centered/mean RMS约`.83`，到第二Composer block仍约`.47--.56`；q/v参与秩中位为
`1.08--1.51/1.13--1.49`，确认parameter-free balance已经打开rank与bank read。action-out则仍仅`1.00--1.04`，平均绝对scale约
`1.13--1.22e-5`，比q约`3.69--3.71e-4`小约30倍；cap factor恒为1。Object/Goal/Long的Process innovation均值为
`.589/.380/.328`，最大occupancy比例为`.518/.424/.672`，说明Long动态偏弱但三者都没有统一事件坍缩。

六个gradient-authorized正确Panel-A任务的冻结functional归因保存在
`runs/analysis/pi05_ecp_policy_response_writer_rank_balance_m200_positive_family_credit_a_3e589695_20260904.json`与对应`_b_`文件。
它使用真实PI05 functional LoRA leaf gradient，把每个family的VJP链回同一个scale head，但optimizer更新为0、negative reads为0且不
参与checkpoint选择。action-out scale-head梯度范数均值`.09209`，与q的`.10237`同量级；4/6任务沿当前action-out mobile方向的
径向导数为负。但action-out聚合梯度/各task梯度范数和仅`.247`，q/v/action-in也仅`.501/.334/.315`，共享head上q--v和
q--action-in聚合梯度cosine为`-.733/-.879`。因此action-out弱不是缺functional credit、断图或cap，而是首先指向跨family共享
末端readout的梯度所有权冲突与跨task相消。m400闭环与同构几何仍是相邻裁决；若稳定复现，再只检验family-local可复制readout，
不以额外loss、人工幅度或超参扫描替代。

## 128. rank-balanced m400显著退化并把最早失败收敛到typed Composer边界

同一fresh `3e589695` shared run完整保存m200/m400 checkpoint。m400 held5 correct-only strict250为`35/250`，逐task
Long/Goal/Object/Spatial0/Spatial9=`0/0/1/33/1`、breadth`3/5`；相对m200的`45/250`为
`30 retained/5 gained/15 lost`、paired exact `p=.0413895`。10个gradient tasks的Panel-B held benefit由
`.000429`升至`.000526`，但两个true-task-held由`-.000438`恶化到`-.002605`、全视频通过数由`1/2`降为`0/2`。
训练Panel-A正benefit比例持续升至最后50步的`.852`，故该相邻结果排除了“尚未训够”：见过task的functional拟合增强时，真正
task-disjoint和闭环反而同步退化。

m400 Object18/Goal25/Long36冻结activation显示matching target/rank query cosine仍为`.99925--.99952`。Process common范数
约`70--74`而language约`11.35`；Composer owner约`0.99`又被family约`11.66`淹没，同family q/v owner-bias cosine达
`.99247/.99318`。无梯度的per-source normalization反事实把四family跨task centered/mean RMS由
`.01783/.02237/.01880/.02241`提高到`.03009/.03515/.03503/.03767`；单纯追加memory没有同等效果。
显示owner/family独立pre-norm可把两个block后q/v跨target centered/mean RMS由`.03713/.04595`提高到
`.18353/.20492`，Object/Goal/Long一致复现。m400六任务VJP同时
显示四family独立聚合平方范数和为当前共享scale-head梯度平方范数的`2.0034x`，action-out在`5/6`任务已要求缩小当前方向却仍随
共享head增长。因此下一fresh修正统一定义为Composer typed-boundary ownership：输入分别保留rank/owner/family/common/language，
输出relative rank gains按family拥有参数；它不改变full bank、真实X/Y、signed pooling、rank4、cap、positive-only loss或唯一rank16。

## 129. m400收尾失败是NFS mmap生命周期错误，checkpoint与独立Panel-B证据完整

原训练进程完成optimizer400、两枚checkpoint和两checkpoint Panel-B后，在rank0删除105GB共享safetensors cache时触发NFS
open-file `.nfs*`语义，`rmtree`报`ENOTEMPTY`，其它rank随后在barrier超时。因此原root保留checkpoint/metrics但不伪造
`result.json`或`completion.json`。clean detached `3e589695`上的只读恢复文件为
`runs/analysis/pi05_ecp_policy_response_writer_rank_balance_s400_panelb_recovery_3e589695_gpu01p0156_20260904.json`，明确记录
零gradient、零optimizer update、零wrong reads以及两checkpoint完整Panel-B。工程修复要求所有rank先清空mmap tensors并GC、
barrier后再由rank0删除，最后再barrier；删除异常同步到全部rank而不再遗留barrier timeout。新增回归后Writer 29项测试通过。
该故障不改变任何训练或科学结果。

## 130. typed-boundary ownership实现与真实smoke

从clean pushed main `2d1fa6e6`建立唯一`codex/policy-response-writer-typed-boundary`工作树。实现只改现有Composer：对
rank/owner/family/common/language各自做parameter-free pre-norm并以固定方差系数组合；把共享relative-gain row改为四个
family-owned rows，每target仍以完整bank-conditioned query选择本family row。没有新Writer、task table、anchor、loss、rank、cap、
bank reader或部署输入，旧checkpoint明确不兼容并必须fresh。

30项Writer测试与新旧config互斥预检通过。gpu01物理0的task1 correct demo5真实smoke消费51帧full policy-response，functional梯度
到达Frame/Event/Composer/relation，causal process梯度到达Frame/Event/predictor；初始A非零/B严格零，打开scale后生成38-target、
76-tensor唯一rank16，峰值allocated/reserved为`27.35/33.98GB`。首个shared资格采用optimizer100/200而非400步，m200仍与前一
资格保持约33次/task的matched总暴露；task-local 50/100与其并行，只作容量定位而不阻塞shared结果。

## 131. typed-boundary task-local 50/100正控

clean detached `682f7ecf`在gpu02物理0/1并行完成task1/task93 formal。task1 macro50/100 fit/held recovery为
`.173162/.163290`、`.241859/.258089`；task93为`.346355/.332496`、`.382004/.364330`，四个checkpoint各自两条fit与一条
held视频全部strict benefit为正。两run均100条metrics、exit0、completion完整，禁止backward计数为0且输出唯一rank16。
该结果保留task-local容量；73-task shared 100/200同时运行，其closed-loop才裁决跨task映射。

## 132. typed-boundary macro100 held5闭环

clean detached `682f7ecf`的shared macro100为五个held5 task分别从固定correct K1视频一次物化唯一76-tensor rank16 LoRA；
`runs/outputs/pi05_ecp_policy_response_writer_typed_boundary_m100_held5_correct_k1_strict250_682f7ecf_gpu02p01_r3_20260904/`
已完成22/22 shards与250/250 rows，结果`39/250`，Long/Goal/Object/Spatial0/Spatial9=`0/0/3/33/3`、breadth`3/5`。
相对stable carrier43为`32 retained/7 gained/11 lost`，相对前一rank-balanced m200=45为`30/9/15`；task/state/env/policy
seed及policy-noise共同前缀逐条配对。该点保留基本闭环能力但没有超过carrier，不能授权Final或negative controls；macro200
相邻strict250与同run两checkpoint Panel-B继续按预注册合同裁决。

## 133. typed-boundary shared训练与Panel-B完成

shared run自然完成200步，训练/Panel-B/总wall为`3348.04/448.99/4072.12s`，两枚checkpoint、200条metrics、result与
completion完整；峰值allocated/reserved约`39.99/46.75GB`。所有rank先释放safetensors mmap后，98GiB临时cache正常删除，
没有复现旧NFS `.nfs*`收尾故障。m100/m200的10个gradient tasks fit/held benefit由
`.0001963/-.0000465`升到`.0005009/.0002095`；两个true-task-held则分别为
`-.0006155/-.0008589`与`-.0008069/-.0007323`，全视频为正由`1/2`降为`0/2`。这保留了seen-task学习证据，
但没有证明task-disjoint shared映射；macro200 held5 strict250的最终裁决见下一节。

## 134. typed-boundary相邻non-pass与common--innovation消费边界定位

macro200 held5 correct-only strict250最终为`32/250`，Long/Goal/Object/Spatial0/Spatial9=`0/0/2/28/2`、breadth`3/5`；
相对m100为`28 retained/4 gained/11 lost`，相对carrier43为`28/4/15`、paired exact `p=.01921`。所以typed source与
family-owned readout没有形成task-disjoint闭环优势，且不能以更多训练解释：m100到m200时gradient tasks的held functional改善，
true-task-held仍全部为负，闭环从39降到32。

六个正确Panel-A任务的VJP确认四个family-owned scale rows梯度支持严格正交，旧跨family冲突已被消除；Object18、Goal25、Long36
的冻结几何却显示matching target/rank跨task query仍约`.9984` cosine。继续向前定位发现，Process innovation `D`跨task并不相似
（median cosine约`-.02--.18`、centered/mean约`1.1`），但RMS仅`.04--.06`，common `C`约`5`。现有predictor直接使用
`C+D_last`，Composer event memory拼接`[E=C+D,D]`，dynamic branch也读取raw `D`，使同一约百倍量纲差在两个consumer边界重复
淹没动态。

causal预测在task-disjoint m100/m200、六个gradient-authorized任务及已有强functional recovery的task-local task1上都不如直接预测
零，证明该辅助目标尚未学得专家要求的时序职责。冻结、零梯度反事实只把`C/D`分别做无affine LayerNorm，并用信息等价的
`[LN0(C),LN0(D)]`替换`[E,D]`、让signed branch读取`LN0(D)`，就把两个Composer blocks后的跨task centered/mean由
`.03172`提高到`.07324`，dynamic branch TV由`.004--.019`提高到`.138--.411`。下一fresh因此只修这一处typed consumer
boundary；不同时修改teacher、loss、rank、scale、bank、数据或正样本合同，并用optimizer50/100短资格作真实闭环裁决。

## 135. Process consumer-boundary闭环non-pass与random-delta恢复

clean pushed `f33f2955`把Process common/innovation分源规范化接入causal predictor、Composer event memory与signed dynamic
scorer，未改变主架构其它变量。73-task fresh optimizer50/100 formal完整结束；held5 correct-only strict250为`40/35`，逐task
Long/Goal/Object/Spatial0/Spatial9分别`0/0/2/38/0`与`0/0/5/29/1`，Goal/Long仍为0。gradient fit/held functional随训练改善，
两个true-task-held却进一步变负；所以该consumer修正本身正式non-pass，不以续训或负控挽救。

冻结诊断确认新边界确实放大了动态并改变了真实LoRA，但causal auxiliary在gradient与held任务上都没有学会预测，且对共享Frame/Event
施加约`1.6--2.8x`、与functional近正交的梯度。进一步审计发现首版把专家明确要求的随机`t,delta>0`实现成了固定
`future_offset=1`。同一m100的六task正样本multi-gap probe保持完整38x50x2输出，within-video最优尺度MSE解释量从delta1的
`.0094`升到delta2/4/8的`.1382/.3268/.4718`，跨task双向均值也从约`.0093`升到约`.0752`。将teacher投到input projection
可见子空间只有次要增益，故下一matched fresh先只恢复random legal delta与parameter-free interval conditioning，固定teacher及
Process/Composer主图不变。不同delta的target RMS近似按`sqrt(delta)`增长，loss对prediction与target作同一可逆
`sqrt(delta)`标准化，避免无关的interval权重漂移。`38d51bab`完成该实现，31项定向测试和task1真实full-horizon delta8 smoke
通过，主functional图保持逐项一致。formal roots、全部diagnostic文件与更细数值见`findings.md`第138节。

## 136. random-delta闭环持平，但机制诊断定位到process objective优化失真

clean detached `eec024f8`的73-task random-delta optimizer50/100正式资格完整结束；两个held5 correct-only strict250均为
`41/250`、breadth`3/5`，Long/Goal仍为0。它比前代m100恢复6分，但没有超过carrier43且相邻换手16行，所以不运行negative
controls。Panel-B同样只显示seen-task微弱改善、两个true-task-held持续为负。

冻结m100的correct-only诊断显示正式predictor相对zero只改善约`.35%`，但同一状态上的head-only容量在直接预测标准化target时，
100步已对未见同task video解释`.144`、250步解释`.217`；delta4/8状态线性probe的同视频解释量为`.278/.410`。因此状态和head
不是零容量，正式优化没有学成。根因包括loss外`sqrt(delta)`对长间隔prediction梯度的衰减，以及每task两随机pair normalizer把
73-task有效权重数降到约`37.2`。下一fresh把head输出定义成标准化delta，以每fit视频8个target-only pair稳定normalizer，并按累计
步长容量证据给纯辅助readout `20x`学习率；实现提交为`df1e8c6e`，固定teacher、主Writer、loss权重、full 50-horizon/native bank/rank/data均不变。
34项测试、task1真实full smoke和两步shared profile通过，精确证据见`findings.md`第139节。

## 137. Process优化修复后仍为37/250，根因推进到causal prefix的event坐标

clean detached `f20a5299`的process-conditioned 73-task formal完成optimizer50/100、两点Panel-B、物化及held5 correct-only
strict250。m50/m100均为`37/250`，逐task分别`0/0/3/31/3`与`0/0/3/32/2`，breadth均`3/5`且Goal/Long为0；相邻为
`31 retained/6 gained/6 lost`。m100相对carrier43为`32/5/11`。所以更稳定的target-only normalizer与充分移动的prediction head
没有产生闭环增益，不支持继续同构训练或negative controls。

正确视频只读诊断却证明时序预测已经学到非零视频信号：m100在108个fit/held pair上的标准化Smooth-L1比zero改善`6.49%`，把Process
state置零反而比zero差`2.36%`，完整状态在`100/108` pair上优于zero-state。失败不是prediction capacity或无视频捷径。代码与张量
联合审计发现，每个严格causal prefix仍调用完整视频的hard-final posterior，导致任意cutoff当前帧`108/108`都被强制映射到slot7；
同一帧在full video只有`15/108`属于slot7，assignment重合仅`.136962`。target侧Process梯度又是functional的`3.73x`、cosine
`-.0572`，Event子集为`3.16x`与`-.2037`，说明错误坐标的辅助更新足以覆盖真实功能credit。

冻结权重反事实仅把prefix改成first-anchored monotone forward filtering，同帧assignment重合即升至`.692016`；完整视频的G2首尾
anchors完全未改。下一matched实现因此只区分“真实完整视频终点”和“辅助训练的人工截断点”：前者保持hard final，后者不施加假终点。
该结论不淘汰Policy-Response Process、positive causal objective、ordered events、Composer、full 50-horizon、native X/Y、signed
pooling、rank4或整个ECP，只淘汰当前对causal prefix错误复用full-video final anchor的推断语义。

## 138. Causal filter闭环non-pass与Composer group-credit定位

`f6b58aac`的causal-prefix filter在clean detached `db354581`完成73-task optimizer50/100 formal。filter把m50/m100人工prefix与完整
视频同帧event assignment重合恢复到约`.774/.743`，m100 predictor相对zero的fit/held/all Smooth-L1改善为
`5.58%/4.39%/5.19%`，所以实现与正时序监督均真实工作；但held5 correct-only strict250只有`38/36`，breadth均`3/5`且
Goal/Long为0。10个gradient tasks m100 fit/held benefit为`+.000341/+.000353`，两个true-task-held却为
`-.002125/-.001937`。该matched实例正式non-pass，不续训或运行negative controls。

冻结正确视频诊断把最早剩余接口推进到Composer gain/credit。task1/72/75/93只优化rank gain时，100步fit/held恢复为
`.151/.093`、`.166/.147`、`.122/.116`、`.150/.099`；恢复专家原文规定的ragged target-native output-group gains后为
`.244/.198`、`.222/.189`、`.146/.126`、`.240/.202`。group自由度是现实现遗漏，但仍非充分修复。相同group-gain对照下，
component-init方向四任务均值`.178/.135`，causal m100为`.213/.179`，说明训练方向有小幅正学习，不支持推倒Process/Composer。

当前零gain启动却让第一次functional backward只能更新gain，后续方向梯度始终被小gain衰减；100步Composer梯度几乎都在gain head，
其它Composer参数只移动低千分量级。G1 free native oracle的既有实现实际以scale logit `0.1`启动。下一matched函数类因此恢复一个
195-row ragged target-native group readout，并以同一`0.1`小幅非零logit让首步functional credit到达direction；完整target BA cap、
static-repeat/no-innovation零mobile、positive-only loss、full 50-horizon、Process、bank、rank、数据、LR与task采样均不变。只有该
边界仍non-pass且方向继续落后，才进入Process/Composer分阶段优化。完整数值与artifact见`findings.md`第142节。

隔离实现的47项Writer/native测试及task1真实full-horizon smoke随后通过。初始非零gain使第一次functional backward的
Frame/Event/Composer-direction/group-gain梯度达到`.056881/.053071/.098988/.221375`，而旧零启动同一输入的Frame/Event约为
`.002898/.002652`；完整50 horizon、38 targets、76 tensors、唯一rank16、冻结policy/observer与约`33.99GB`reserved峰值均保持。
这只验证方向信用恢复，不提前宣称科学性能。

task1两步shared optimizer profile随后以`3.686/3.497s`自然完成；Composer总/gain梯度为
`6.398/6.116`与`3.123/2.732`，非gain方向约`1.879/1.512`，峰值reserved `38.50GB`。profile rows2 loss不与完整carrier
混作科学比较；该证据只补齐optimizer和资源运行面。

gpu01出现6张真正空闲A40后，同一`aebd9d74`又完成73-task world6 rows2 profile。两步均为每rank 2 tasks，wall
`6.361/6.695s`、预测cost范围`73--78/78--86`，峰值allocated/reserved `27.36/37.04GB`；direction与gain梯度均非零。
相对既有73-task world4 rows2均值约快`34.2%`，因此下一optimizer50/100短资格使用world6；它仍是执行拓扑选择，不改变task batch、
权重或科学方法。

## 139. Composer-functional持平carrier并定位factor-conditioned gain readout

`45b63c97`实现的冻结Process阶段从clean detached `a9baa7a4`完成73-task、K1、component-init、optimizer50/100正式训练与两点
held5 correct-only strict250。m50/m100为`39/43`，breadth均`3/5`且Goal/Long为0；m100恰与stable carrier持平，success set
仍有`7 gained/7 lost`，不构成性能提升。Process全程零梯度；Composer direction/gain真实移动且seen functional继续改善，故该结果
淘汰“联合causal auxiliary或Process漂移是当前唯一瓶颈”，不支持续训或negative controls。

冻结正确视频几何显示最终mobile directions仍高度task-specific，query-only的195个raw gains却在不同task间几乎相同。m100上
task2/task74的actual gain cosine为`.9991`，exact per-group descent方向cosine为`-.5846`；只调当前gain存在足以覆盖task74负增量的
局部下降空间。由此下一fresh不修改Process、signed candidate direction或loss，而以共享factor-conditioned ragged group token
readout替换target-owned输出rows：每个token显式读取当前query与signed X/Y factor，同一可复制GatedMLP和scalar head应用于全部
target/rank/group。实现与科学裁决继续见active design、`findings.md`第144节及后续提交。

## 140. Factor-conditioned pointwise readout non-pass并推进到set-relative factor协调

clean detached `ef066789`的73-task factor-conditioned formal完成optimizer50/100、Panel-B与两点held5 strict250。闭环为
`40/44`，m100相对carrier43仅净增1且Goal/Long仍为0；seen-task functional继续提高时两个true-task-held均值更负。
与此同时，task1/task93 task-local m50/m100的fit/held recovery均约`.36--.53`且所有fit/held视频为正，证明该函数类有local容量与
跨视频保持，失败在共享映射。

六task correct-only参数信用诊断进一步显示，task74/task73所需gain-logit变化cosine为`+.590`，pointwise readout全参数梯度却为
`-.418`，而实际gain跨task cosine仍约`.994--.998`。下一fresh因此只在当前factor token之后加入同target
`rank x ragged-group`标准self-attention + GatedMLP块，使gain可读取相对组合；不跨target，不改Process、signed bank direction、
loss、full-50、rank、cap、data或初始化。实现、61项测试与两步真实GPU smoke已通过；正式结果见后续记录。

## 141. Role-equal终局与Axial Writer架构重置

role-equal formal root
`runs/outputs/pi05_ecp_policy_response_writer_factor_set_relative_gain_role_equal_73task_k1_component_s100_28b4eb05_gpu02p013_sharedmmap_20260904/`
完成100 optimizer steps并删除临时mmap cache。m50/m100 held5 correct-only strict250为`39/45`；m100逐task
Long/Goal/Object/Spatial0/Spatial9=`0/0/2/41/2`、breadth`3/5`，仍无Goal/Long。m100 evaluation root为
`runs/outputs/pi05_ecp_policy_response_writer_factor_set_relative_gain_role_equal_m100_held5_correct_k1_strict250_73380ec8_gpu01p036_r3_20260904/`，
250 rows全部完成。Panel-B虽在gradient tasks上小幅改善，两个true-task-held仍总体为负。因此role weight与cursor覆盖联合修复未解决
shared task-disjoint mapping，不再续训、跑negative controls或扫描比例。

2026-09-05 owner基于两个月历史与ECP两周的累积non-pass，授权在EMBER科学边界内实质重构，同时要求新架构必须优雅、可按复制层扩展，
不能再形成连续数学变换。新的active合同为`docs/axial_policy_response_native_factor_writer_design.md`。实现整体删除旧relation/HMM、
C/D、factor normalization、event/relation marginal、独立gain和causal auxiliary；主图只保留Frame、Temporal、Event、RankBank
attention/MLP blocks与direct signed raw-X/Y pooling。原生full-50、真实X/Y、rank12+4、unique rank16、positive-only与信息墙不变。

开发分支上的25项Writer/native tests通过。task72真实full-50 smoke确认全部learned模块有functional gradient、冻结policy无梯度、
76 tensors与完整rank16；5-step task-local profile三条fit/held视频均微弱高于carrier，但幅度尚小。该证据只解封clean detached
25/50-step task72容量控制，尚未形成shared或closed-loop结论。

## 142. Axial task72短容量控制

clean detached `89d912b7`完成task72、full-K1、component-init、Composer-only 50-step formal，root为
`runs/outputs/pi05_ecp_policy_response_writer_axial_factor_tasklocal_task72_full_s50_3cc4dbfc_gpu01p0_20260905/`。macro25 fit/held
functional recovery为`.09362/.08602`，macro50为`.07660/.06156`；两点三条视频平均均优于carrier，但后半回落且低于历史
set-relative task-local容量。不续训该控制。

同authority的两步whole-Writer shared profile在task72上以`1.81/1.50s`完成，约3.74M Writer参数全部可训，所有learned模块均有
functional梯度，冻结policy/observer无梯度。它解封73-task joint 25/50-step短资格，尚不构成shared性能证据。

## 143. Axial短shared与frame-local Composer缺口

Axial 73-task whole-Writer 25/50-step formal完成，但总计只有300次task exposure；m25/m50 gradient fit/held与两个true-task-held均
non-pass，未进入held5。随后task1/72/93相同50-update whole-Writer对照呈弱正、正、负三种行为；旧shared correct-only几何则证明
events仍有task差异，输出B却被训练到更共同的方向。task93 Composer-only formal m50 held benefit/recovery仅
`+.000600/.04547`，远低于free primal与历史frame-local Composer。

选择性Git复核定位到旧强实现以candidate所属frame的innovation参与signed score，而当前Axial把一个global dynamic query广播到
全部frame。active图因此以可重复`FrameAlignedFactorBlock`整体替换RankBank职责：rank读events、frame按真实相对位置读本视频events，
frame-specific dynamic直接对完整raw X/Y做一次exact signed pooling；首次完整bank预读被删除。该变化保持full-50、positive-only、
rank4、真实native X/Y及唯一rank16，不增加loss、solve、normalization、gain或校准链。25项CPU合同与task93真实full smoke通过，
正式task-local/shared裁决待后续记录。

## 144. Frame-aligned task-local与输出几何裁决

clean detached `7b42cdf6`的task1/task93 Frame-Aligned Composer-only正式控制完成25/50步，两任务相邻checkpoint的三条正确
fit/held视频均优于carrier；m50 fit/held recovery分别为`.0761/.0958`与`.0868/.0793`。task93 held相对前一global-broadcast
实现提高约74%，但绝对恢复仍仅约8%，所以只证明frame-local修正有效而非Composer容量已充分。

同checkpoint的correct-only响应曲线进一步裁决两条旧式输出修正。统一contrast放大在task93的2倍有益、4倍灾难性，在task1则fit从
1倍起即不再改善，故不存在跨task固定温度。逐rank单位化A/B并施加统一`s_ref`步长在两任务上都很快转负，证明低幅方向不能被无差别
放大。active路线因此不恢复normalization/gain链，而以10 gradient + 2 true-held、100-step的短shared资格直接检验完整可复制主干。

## 145. Frame-Aligned 12-task non-pass与Frame-Bank替换

clean detached `da1657ef`完成10 gradient + 2 zero-gradient held、K1、whole-Writer 100-step资格。m50/m100 gradient
fit/held benefit为`+.000475/+.000383`与`+.000561/+.000477`，两个true-task-held两点均为负且`0/2`全视频通过，故没有运行
held5、负controls或续训。正式root为
`runs/outputs/pi05_ecp_policy_response_writer_frame_aligned_12task_k1_component_s100_da1657ef_gpu02p012_sharedmmap_20260905/`。

两轮post-run correct-only几何见`findings.md`第150节。核心证据是Process events并未坍缩，而global Composer query跨task约
`.996` cosine；task74生成update与相邻72/73/75过度同向，甚至在改善这些邻近task时增加自身真实functional loss。两层event read
仅为强global residual的约`.28--.45%`。因此实际淘汰的是“global structural query读取events后广播到所有frame，再由bank末端线性
打分”的函数类，不是full horizon、PI0.5时序、native X/Y、signed pooling或rank4。

后继active实现用单一可复制`FrameBankFactorBlock`整体替换该职责：每个真实frame的rank state先读取同frame完整native X/Y bank，
再读取ordered events、做rank attention和bias-free MLP，末端仍只做raw-value signed pooling与target cap。逐视频bank read在frame轴
中心化以结构性保持static-repeat零mobile；没有增加gain、normalization、whitening、solver、temperature或calibration链。它不同于
历史whole-video `RankBankContextBlock`的global bank summary/broadcast。由于定位诊断在formal non-pass后读取了task2/74的授权
Panel-B gradient，后继shared资格必须选择新的unseen held tasks。

新实现的task93真实full-50 smoke保持79 sampled frames、2 probes、38 targets、76 tensors与唯一rank16，全部Writer阶段梯度有限非零、
冻结policy零梯度。exact frame chunk从8增至128使相同第一步由`8.02s`降至`3.95s`，峰值reserved约`28.55 GiB`；这只是完整候选集的
等价执行分块，不改变科学图或full-only边界。

## 146. Frame-Bank task-local部分改善与fresh-held shared预注册

clean detached `471592f4`完成task1/task93各50步Composer-only formal。task1 m25/m50 fit/held recovery为
`.0540/.0526`与`.0841/.0702`；task93为`.0446/.0425`与`.1378/.1360`，两任务两相邻点的三条正确视频聚合均为正。相对
Frame-Aligned，task93 m50明确改善而task1混合；两任务绝对恢复仍仅约`5--14%`。因此同frame bank read是有效增量，但冻结Process的
Composer-only容量不充分，不以续训、统一gain或校准链挽救。

下一50-step whole-Writer shared在查看结果前固定task3/77为fresh zero-gradient held，选择规则为post-hoc暴露task2/74后每个role的
最小eligible未读ID；task2/74转入gradient，与原10 tasks形成6 meta + 6 target。每个gradient task在50步内精确暴露25次，m25/m50
只读Panel-B。该资格只检验可训练Process与FrameBank的task-disjoint共同适配；出现正信号前不运行held5或negative controls。

## 147. Frame-Bank shared终局与Native-Temporal Axial替换

clean detached `07804433`的Frame-Bank 12-gradient + 2-held whole-Writer资格完成50步。m25/m50 gradient-task fit/held benefit由
`+.000157/+.000117`升至`+.000240/+.000200`，全视频为正由`6/12`升至`8/12`；fresh held task3为正、task77为负，两点均只有
`1/2`通过。该实例有微弱seen-task学习但没有稳定task-disjoint映射，未运行held5或negative controls。

冻结正确视频VJP排除普遍梯度灾难：m50六task整体pairwise cosine mean为`.0556`，而冲突集中在event readout及signed-X head；后者
pairwise mean为`-.0578`，task93相对其它task和为`-.663`，signed-Y则为正。task1/task93路径消融又显示frame-only和event-only没有
任何一条跨两类任务充分。active设计据此整体删除`Temporal -> Event -> late FrameBank -> shared X/Y state`接口，改为逐帧完整
PI0.5 response编码后，以显式X/Y factor-side states在同一可复制NativeTemporalFactorBlock内完成same-frame native read、真实frame-time
attention与rank/side协调，再直接signed-pool raw X/Y。learned图收敛为Frame与NativeTemporalFactor两种block；没有恢复summary、solve、
normalization、gain、temperature或calibration链。正式与诊断roots及完整数值见`findings.md`第152节。

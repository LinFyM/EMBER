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

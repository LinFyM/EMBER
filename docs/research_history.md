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

## 22. 当前保留结论

1. EMBER输入输出目标不变，ECP核心尚未被完整实验反证。
2. task-local LoRA与mobile-rank4容量充足；native video basis和shared selection mapping是顺序待验证的最早接口。
3. policy effects保留为critic；neural `q_pi`、fit-span/fixed-effect realizer、GOMQ、PECS与v24均不进入active路线。
4. Program schema、native bank和compiler角色已经固定，不能共同自由旋转或退回全局task code。
5. Stage 0必须证明full video超越endpoints；shared compiler必须直接用held closed loop证明跨task mapping。
6. 分阶段冻结后必须进行冻结backbone、冻结carrier的全Writer联合训练。
7. shuffled/reversed只用于最终冻结checkpoint的时序特异性评测。

## 23. 证据恢复方式

- 活动科学合同：`AGENTS.md`、`docs/current_owner_requirements.md`、`docs/concept.md`。
- 当前架构：`docs/event_conditioned_policy_compiler_design.md`。
- 当前状态：`task_plan.md`、`findings.md`、`progress.md`。
- 精确旧实现与配置：以上Git提交或`git log`中相邻提交。
- 大型formal结果：本地ignored `runs/`中的唯一checkpoint、raw rows与aggregate；人工process资产除外，已明确删除。

任何旧提交中的“active”“next”或“current”只代表当时状态，不能覆盖当前owner要求。

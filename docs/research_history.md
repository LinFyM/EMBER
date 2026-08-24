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

## 10. 当前保留结论

1. EMBER输入输出目标不变，ECP核心尚未被完整实验反证。
2. task-local LoRA与mobile-rank4容量充足；native video basis和shared selection mapping是顺序待验证的最早接口。
3. policy effects保留为critic；neural `q_pi`、fit-span/fixed-effect realizer、GOMQ、PECS与v24均不进入active路线。
4. Program schema、native bank和compiler角色已经固定，不能共同自由旋转或退回全局task code。
5. Stage 0必须证明full video超越endpoints；shared compiler必须直接用held closed loop证明跨task mapping。
6. 分阶段冻结后必须进行冻结backbone、冻结carrier的全Writer联合训练。
7. shuffled/reversed只用于最终冻结checkpoint的时序特异性评测。

## 11. 证据恢复方式

- 活动科学合同：`AGENTS.md`、`docs/current_owner_requirements.md`、`docs/concept.md`。
- 当前架构：`docs/event_conditioned_policy_compiler_design.md`。
- 当前状态：`task_plan.md`、`findings.md`、`progress.md`。
- 精确旧实现与配置：以上Git提交或`git log`中相邻提交。
- 大型formal结果：本地ignored `runs/`中的唯一checkpoint、raw rows与aggregate；人工process资产除外，已明确删除。

任何旧提交中的“active”“next”或“current”只代表当时状态，不能覆盖当前owner要求。

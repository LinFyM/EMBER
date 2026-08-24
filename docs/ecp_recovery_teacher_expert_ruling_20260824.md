# ECP composite recovery teacher 专家复核裁决

状态：2026-08-24，复核基准为远程
`main@d8eca7987a4ad2a59c5d27738b29a8d4d9bfd161`。本裁决取代
`docs/ecp_composite_teacher_distillation_gate_20260824.md`中“直接训练现有phase-expert
distillation数据”的执行授权，但不改写既有采集数据和历史实验结果。

## 1. 总体裁决

1. **现有on-policy phase-expert distillation不得启动。** 它覆盖了composite student自己的访问状态，但没有证明
   primitive phase expert在这些状态上是能够继续完成整个组合任务的可靠oracle。
2. 路线在GOMQ归档、process Gate A/A2/A3、37条成功轨迹bootstrap、两个order-specific composite SFT、
   effect-path calibration和两种principled realizer裁决之前总体仍与专家方案一致。
3. 真正的偏移发生在两个composite SFT policy闭环失败后：仓库重新把已经在post-sibling-goal状态上失败过的
   primitive experts升级成student occupancy上的完整动作oracle。
4. 当前唯一合理的下一科学步骤，是用A3真实成功轨迹的第二阶段动作训练两个
   **composite-context recovery experts**，并让它们与冻结的第一阶段primitive experts组成新的privileged
   composite controller，重新接受不变Gate A。
5. 当前shared-realizer family保持关闭；fresh Stage 0、`q_pi`、`q_V`、新realizer、joint Writer和outer credit均不启动。

## 2. 当前distillation为什么不能训练

### 2.1 标签oracle没有成立

当前采集满足`state ~ d(student)`，但没有验证`primitive expert(state)`是正确恢复动作。采集过程只在student当前状态
查询phase expert，没有从该状态执行teacher continuation，也没有检查它能否完成当前phase、保持已完成goal或最终完成
composite task。A2/A3恰好已经证明primitive expert在另一个goal完成后的状态上不可靠。

### 2.2 完整50步标签跨越了潜在phase边界

真实phase controller在event完成时会丢弃旧chunk、切换expert并基于新观测重新规划。当前HDF5却把查询时phase expert的
完整`[50,7]` chunk全部作为有效标签；reader不读取phase boundary，`action_is_pad`也没有遮掉切换后的tail。若event在
第6--20步完成，后续标签仍来自旧phase expert，因此监督目标与真实composite controller语义不一致。

### 2.3 失败轨迹按query等权会放大最不可靠区域

长失败episode自然贡献更多query；当前采样不依据continuation validity、episode outcome或invalid proximity加权，且没有
混入原37条成功轨迹保护已知success support。在oracle未验证时，这会让最偏离成功basin、teacher最不可信的状态占据较大权重。

因此，现有`2773/3998` query数据只归档为**student occupancy与weak-teacher response资产**。无论其训练后分数提高还是
下降，都无法清楚区分oracle正确性、phase-tail错位、状态覆盖、SFT优化和static LoRA能力，不构成可解释的formal实验。

## 3. 已有结果的正确边界

- GOMQ真实rank16为`136/400`：保留视频因果机制证据，但失去`145+` absolute baseline资格。
- process Gate A/A2/A3分别为`19/100、44/100、37/100`：否定当前primitive phase composition，不裁决视频是否能识别顺序；
  Gate B从未运行。
- 两个order-specific composite SFT从28/9条成功episode训练后，在完整固定100-row matched采集中的真实闭环baseline均为
  `5/50`；训练loss下降只证明成功轨迹局部拟合，不能证明闭环composite policy成立。
- Phase 2A的15/15 known-success effect paths通过，证明当前policy-effect objective能识别已知成功修正方向。
- balanced-SVD learned realizer和centered two-sided fit-span coordinate均未过held closed-loop门。关闭的是当前fit90
  mapping authority上的shared mapping family，不是`carrier rank12 + mobile residual rank4`容量、Program、`q_pi/q_V`
  或整个ECP。

## 4. 唯一下一步：composite-context recovery teacher Gate

### 4.1 两个recovery experts

- `red -> yellow-white`：第一阶段使用冻结的red primitive expert；第二阶段切换到
  `yellow-white-after-red recovery expert`。
- `yellow-white -> red`：第一阶段使用冻结的yellow-white primitive expert；第二阶段切换到
  `red-after-yellow-white recovery expert`。

每个recovery expert从对应原primitive expert初始化；source PI0.5完全冻结；只训练一套38-target rank16 task-local LoRA；
两个方向不共享参数。它们是训练期privileged teacher acquisition，不是EMBER部署架构。

### 4.2 固定训练数据

- 50%：A3真实成功single-episode中第一event完成后的second-phase执行片段；red-first有28条，yellow-first有9条；
- 50%：对应原primitive成功数据，用于保持原技能；
- 只使用真实执行且最终成功的动作；action chunk不得跨越未建模phase或episode结束；
- 不使用当前`2773/3998`条未验证phase-expert预测作标签；
- 不使用A3失败状态，除非未来另有经过continuation验证的正确标签。

训练前固定两方向相同的混合比例、steps/epochs、优化合同和唯一checkpoint；不按方向单独调参。

### 4.3 固定privileged controller

```text
first event:
    frozen original primitive expert
phase transition:
    discard old action chunk and replan
second event:
    direction-specific composite-context recovery expert
```

两个方向继续使用统一公开language、相同initial states、temporal wrapper、RNG和information wall。

### 4.4 不变Gate A

- red-first至少`20/50`；
- yellow-first至少`20/50`；
- total至少`50/100`；
- wrong-first invalid、phase/expert route mismatch、state/noise mismatch和public-video privileged leakage均为0；
- 每个success必须保持第一predicate并最终完成conjunction；
- 不按state、checkpoint或结果选择teacher。

通过后唯一授权是运行原Gate B。若本轮recovery teacher仍不通过，则关闭task65/68上的primitive composition、composite SFT、
current phase-expert distillation和second-phase recovery SFT；不做第二轮DAgger、不延长训练、不扫step/LR/rank/seed，也不换
task66/67重复同一机制。下一次process推进必须先取得独立成立并先通过同一Gate A的composite privileged controller，例如
scripted planner、teleoperation/human demonstrations、privileged MPC或task-local simulator RL。

## 5. 对ECP主链的含义

ECP没有被整体否定，但当前有两个独立前置条件没有建立：

1. 可靠的process-identifying composite teacher；
2. 可跨任务泛化、deployment-compatible且能保留低能量闭环关键方向的fixed realizer。

recovery teacher若通过，只解决第一个条件的teacher acquisition并授权Gate B，不等于ECP Writer已经开始，也不自动重开
旧realizer。Gate B通过后才扩充process suite并重建fresh owner-specific Stage 0 Program；之后仍须依次建立新的realizer
authority、distributional `q_pi`、privileged full chain、同构`q_V`、joint Writer和outer credit。

## 6. 当前执行状态

- 已封存的`347/500`步phase-expert distillation training未启动，现取消执行授权；
- 已采集数据与原launch contract保留为历史和weak-teacher诊断证据；
- 当前没有active GPU job；
- 当前仅记录专家裁决并向owner解释项目状态；在owner理解并确认前，不实现或启动recovery teacher训练。

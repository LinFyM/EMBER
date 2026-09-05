# Native-Temporal Axial Policy-Response Writer

状态：sealed direct predecessor；当前active见`docs/unified_policy_native_factor_writer_design.md`
owner边界：2026-09-04至2026-09-05
前身裁决：Frame-Bank实例证明同frame native read有局部增量，但显式Event瓶颈与晚期X/Y共享状态没有学得稳定shared task-disjoint映射

## 1. 目标

Writer只接收exact language与一条或多条same-task、action-hidden、内部有序正确视频，在rollout前运行一次，生成唯一一套覆盖
38 targets的rank16 LoRA。正式目标仍是validation8 strict paired correct稳定严格大于`145/400`，并满足相邻checkpoint稳定、
高breadth、四suite非零、Goal/Long贡献、same-task不同视频鲁棒性，以及checkpoint冻结后的视频因果controls。

本设计保留ECP已由实证支持的PI0.5原生响应、真实native X/Y、signed pooling与rank4 mobile边界，整体替换反复失败的
`temporal trunk -> event bottleneck -> late bank fusion -> shared X/Y state`接口。主干只有两种职责清楚的learned block；容量通过
复制同构block或增加width/heads扩展，不再串联专用摘要、门控、归一化、求解或校准步骤。

## 2. 不可改变的科学边界

- deployment输入只有exact language与正确、action-hidden、内部有序teacher videos；不得读取action、state/proprio、reward、terminal、
  task ID、filename、pose或policy outcome。
- 冻结source PI0.5逐帧使用原生image-language prefix和固定antithetic probes，保留真实19 layer boundaries、2 probes、完整50
  Action Expert horizons、flow velocity及38-target native X/Y；`full`是唯一active representation。
- teacher-frame time、Action Expert horizon、flow time、network depth和probe是不同轴；不得用coarse、horizon mean、抽样或`t+h`
  伪造绝对控制时间。
- 每条视频独立保序编码；视频之间没有index，集合聚合必须置换不变。不得把frames、raw features或最终LoRA平均成部署表示。
- 当前视频真实native X/Y是mobile factor唯一原始vector value来源。language与learned state只能决定读取方式，不能独立写出residual。
- 每个condition只生成一套rank12 carrier加rank4 mobile组成的完整rank16 LoRA；不部署第二adapter、expert或task dictionary。
- 首轮训练只使用授权task的correct cross-episode functional梯度。wrong、no-video、shuffled和reversed不进入训练、loss或checkpoint选择。

## 3. 唯一数据流水线

```text
ordered video frames + exact language
        |
frozen PI0.5 native capture
  prefix + response[19 layers, 2 probes, 50 horizons, state/residual/noise/velocity] + raw X/Y bank
        |
repeat Nf x FramePolicyResponseBlock
  each target-rank reads the same frame's native prefix and its full owner-matched policy response
        |
add an explicit factor-side axis {X, Y}
        |
repeat Nc x NativeTemporalFactorBlock
  same-frame side-matched native-bank read -> ordered frame-time attention -> rank/side attention -> MLP
        |
one final per-video frame centering
        |
factor-side two-branch queries -> one exact signed pooling of untouched full raw X/Y
        |
one target-level BA cap -> rank12 + rank4 -> unique 38-target rank16 LoRA
```

没有显式event压缩、global video summary、第二条carrier、并行fallback，也没有
`summary -> solve -> recenter -> whitening -> transport -> gain -> calibration`一类连续接口。

## 4. Learned blocks

### 4.1 FramePolicyResponseBlock

每个target-rank state依次以标准cross-attention读取当前frame的完整native image-language prefix和该owner对应的完整
`2 probes x 50 horizons x 8 channels` policy-response，再在同一frame内对target-rank states做self-attention和GatedMLP。
exact language直接留在native prefix memory中，不再先另做owner-language summary。antithetic probe只做even/odd可逆换基；没有删除
probe或horizon信息。层对应关系沿用真实LoRA owner，38个owners共同覆盖Action Expert目标层。

初始state只由rank、LoRA owner与owner family的learned embeddings相加得到；它们描述输出结构而非task identity。视频语义必须经当前
frame的prefix与完整policy-response进入state。加深时只复制相同Frame block。

### 4.2 NativeTemporalFactorBlock

Frame states进入Composer时扩成显式`factor side = {X, Y}`轴。每个block保持同一固定拓扑：

1. X-side逐帧读取同frame完整native input-X candidates；Y-side逐帧读取同frame完整native output-Y
   `absolute/adjacent/initial/goal-relative` candidates；
2. 每个target-rank-side沿真实teacher-frame time做标准self-attention；frame position只进入Q/K，不进入value；
3. 每个frame内的`rank x side` states做标准self-attention和GatedMLP，使四个rank以及X/Y两侧能协调但不被迫共享同一个晚期向量。

同一block同时拥有native敏感性与视频时间建模责任，因此动态理解不会先被压成少量events再晚期接回bank。所有视频各自独立执行同一
block，block不读取video index。扩大模型只复制这个block，不增加另一套event、relation、gain或calibration模块。

### 4.3 Direct native-factor readout

最后一个block后，每条视频只做一次frame-axis content centering。该操作不聚合或删除frame；全部centered frame states仍分别为原始
candidate打分。它只保证完全静态重复序列不能靠owner、rank、side或position形成mobile update。

X与Y各有一个bias-free two-branch query projection。对应side的每个逐帧state直接产生正、负两组logits，再对完整
`video x frame x probe x 50-horizon x bank-type` raw native values做G1同类exact signed pooling。没有额外base/contrast串联，
没有全视频dynamic query广播，也没有factor normalization、gain、temperature或方向校正。静态重复视频时centered state为零，
正负分布相同，A、B与完整mobile update严格为零。

唯一post-pooling操作是既有per-target `B@A` RMS cap；small-core canonicalization只用于稳定物化rank4并与rank12 carrier合并，
不改变功能矩阵。

## 5. 这次替换由什么证据支持

- G1的`114/250`、breadth`5/5`、Goal2、Long1证明真实native X/Y、signed pooling和task-local rank4能产生有效闭环变化，因此保留。
- G2的full相对endpoints改善`22.2047%`、probe`38/40`和same-task/K1/K4通过，证明PI0.5的完整Action Expert response与有序视频动态
  有信息，因此保留full response和真实frame time；删除的是固定event schema，不是时序信息。
- Frame-Bank task-local task1/task93在m25/m50均跨fit/held视频保持正方向，证明same-frame bank read是有效增量；但只恢复free primal
  约`5--14%`，说明把bank接在冻结Event接口之后仍不是充分函数类。
- Frame-Bank 12-gradient shared从m25到m50只由`6/12`升至`8/12` gradient tasks全视频为正；fresh held task3为正而task77持续为负，
  task93/94也没有稳定恢复。继续训练原图不能解释这种family/task选择性失败。
- m50六task functional-gradient几何的整体pairwise mean为`.0556`、负比例`.40`，排除“全部共享参数天然不可联合训练”。冲突集中在
  event消费与X readout：event readout pairwise mean仅`.0140`，task93/94相对其余任务和为负；signed-X head pairwise mean
  `-.0578`，task93相对其余任务和为`-.663`，而signed-Y为`+.0875`。因此X/Y在末端才由同一state线性分叉过晚，显式factor-side
  state是针对最早失效接口的结构替换，不是附加补丁。
- 冻结m50的task1/task93正确视频消融显示，frame-only与event-only都不是普适解：task1 frame-only微弱为正而event-only为负；task93
  frame-only为负，event-only仅有极小且不稳定的独特贡献。结论不是删除时间，而是让native-bank read与时间建模在同一个block内共同形成
  factor state。
- 历史v5/v6类neural Writer证明可扩展神经主干能获得一定收益，但video necessity与稳定性不足；本设计吸收其可复制learned trunk，
  同时保留ECP的信息墙、full PI0.5 response、raw native value路径、task-disjoint裁决与最终因果controls。

## 6. 简洁性与生命周期规则

首个短资格配置采用`2 Frame + 2 NativeTemporalFactor` blocks、宽度128和4 heads；数字只是资格实例，不是owner永久要求。

- active learned图只有两种block和两个直接factor-side heads。不得为负结果追加event summary、analytic solve、whitening、gain、temperature、
  calibration或等价专用链。
- 扩容优先增加width/heads或复制现有block。若某一接口被复核为最早失效点，应整体替换其责任，而不是保留旧接口再补偿。
- 手工运算只服务明确边界：axis/mask、可逆probe换基、一次最终dynamic centering、exact chunk reduction、signed pooling、最终cap与
  rank materialization。
- active tree只保留这一条canonical Writer图；旧Temporal/Event/FrameBank runtime实现随替换删除，历史由Git、sealed configs、formal
  artifacts与research history保存。

## 7. 实现、训练与裁决

component initialization只复用G2已验证的prefix/response projections、owner/family/layer/horizon embeddings及第一个response
attention；NativeTemporalFactor blocks与factor-side heads为fresh。Final仍必须比较同拓扑fully-random、fresh optimizer/scheduler候选。

首轮按最短有信息量成本递增：

1. CPU shape/order/K/chunk/static不变量检查，以及真实full-50 forward、functional VJP和唯一rank16 materialization smoke；
2. task1/task93各25/50-step Composer-only容量控制，确认显式X/Y side与native-temporal block在普通与Long任务上都能跨video学习；
3. task-local图成立后，以12个gradient tasks和预先冻结的新held task4/78运行50-step whole-Writer shared资格；每个gradient task在
   m50前获得25次暴露。task batch和meta/target比例只是该配置选择；
4. 只有出现task-disjoint正信号才运行held5 correct-only strict250；明显坏结果不靠长跑、seed/LR/scale小扫挽救；
5. shared信号成立后再覆盖mixed-K、fully-random Final与更长训练，并进入validation8 strict paired400；
6. 只在correct-only选定并冻结checkpoint后运行same-task-other、wrong、no-video、language-only、first/final、shuffled/reversed controls。

task3/77已经被Frame-Bank资格读作zero-gradient held，不能重复充当新架构的unseen证据；按相同outcome-independent规则，在读取任何
Native-Temporal结果前固定每个role的下一个eligible未读task4/78。任一负结果先定位最早失效接口并检查真实closed-loop/functional
正控，再以一个有证据的责任替换推进；不以内部loss门槛代替闭环结果，也不无限扩散搜索。

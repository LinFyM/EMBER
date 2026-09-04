# Axial Policy-Response Native Factor Writer

状态：active design
owner边界：2026-09-04至2026-09-05
前身裁决：Policy-Response Event-to-Factor Writer的set-relative/role-equal实例未学得shared task-disjoint映射

## 1. 目标

Writer只接收exact language与一条或多条same-task、action-hidden、内部有序正确视频，在rollout前运行一次，生成唯一一套覆盖
38 targets的rank16 LoRA。正式目标仍是validation8 strict paired correct稳定严格大于`145/400`，并满足相邻checkpoint稳定、
高breadth、四suite非零、Goal/Long贡献、same-task不同视频鲁棒性，以及checkpoint冻结后的视频因果controls。

本设计不是放弃ECP，而是保留已被正证据支持的原生观测与factor边界，直接替换反复失败的Program--bank/shared utility接口。
它的工程目标同样明确：主干只由少量职责清楚的attention/MLP blocks组成，增大容量时复制同构块，不再串联专用数学补丁。

## 2. 不可改变的科学边界

- deployment输入只有exact language与正确、action-hidden、内部有序teacher videos；不得读取action、state/proprio、reward、terminal、
  task ID、filename、pose或policy outcome。
- 冻结source PI0.5逐帧使用原生image-language prefix和固定antithetic probes，保留真实19 layer boundaries、2 probes、完整50
  Action Expert horizons、flow velocity及38-target native X/Y；`full`是唯一active representation。
- teacher-frame time、Action Expert horizon、flow time、network depth和probe是不同轴；不得用coarse、horizon mean、抽样或`t+h`
  伪造绝对控制时间。
- 每条视频先独立保序编码；视频之间没有index，集合聚合必须置换不变。不得平均frames、raw features或最终LoRA。
- 当前视频真实native X/Y是mobile factor唯一原始vector value来源。language与结构token只能形成query，不能独立写出residual。
- 每个condition只生成一套rank12 carrier加rank4 mobile组成的完整rank16 LoRA；不部署第二adapter、expert或task dictionary。
- 训练只使用授权task的correct cross-episode functional梯度。wrong、no-video、shuffled和reversed不进入训练、loss或checkpoint选择。

## 3. 唯一数据流水线

```text
ordered video frames + exact language
        |
frozen PI0.5 native capture: prefix + owner-matched response[2 probes, 50 horizons] + X/Y
        |
repeat Nf x FramePolicyResponseBlock
        |
repeat Nt x TemporalPolicyResponseBlock over real frame time
        |
one content centering + ordered event readout + repeat Ne x OrderedEventBlock
        |
repeat Nc x FrameAlignedFactorBlock aligning events to real teacher frames
        |
per-frame base +/- dynamic logits -> one exact signed pooling of full raw X/Y
        |
one target-level BA cap -> rank12 + rank4 -> unique 38-target rank16 LoRA
```

没有并行fallback，也没有`summary -> solve -> recenter -> whitening -> transport -> gain`一类连续接口。

## 4. Learned blocks

### 4.1 FramePolicyResponseBlock

每个target-rank query依次读取当前frame的完整native prefix与该owner对应的`2 x 50` policy-response channels，再在同一frame内对
target-rank tokens做self-attention和MLP。antithetic probe以even/odd坐标表示；对两个probe这是可逆换基，不删除probe信息。
层对应关系沿用真实LoRA owner：每个target读取其native layer，38个owners共同覆盖Action Expert目标层。

### 4.2 TemporalPolicyResponseBlock

对每个target-rank沿真实teacher-frame time做标准self-attention和MLP。frame position只进入attention的Q/K，不进入value，避免
静态重复frame仅靠位置产生伪动态。深度只通过复制同一block扩展。

### 4.3 Ordered event readout

时序trunk之后只做一次跨frame content centering。首、尾content作为边界事件，内部event queries从完整有序frame memory读取，其后
使用同构event attention/MLP blocks。该设计吸收G2“有序动态与边界有信息”的正证据，但不继续使用固定
`P_lang/P_scene/P_process/rho/tau/sigma` schema、HMM、relation marginal或手工event assignment。

### 4.4 FrameAlignedFactorBlock

每个target的四个rank queries先以标准cross-attention读取动态events，再沿rank轴self-attention。随后每条视频的每个真实frame
以自身`frame_innovation + rank query + relative frame position`为query，只读取本视频有序event tokens；event slot position只进入
Q/K，dynamic event本身是value。这是一个可重复的Transformer block：加深时只复制同构block，不新增坐标或规则。

该逐帧read将event语义重新对齐到产生native X/Y的真实frame，恢复专家原合同中的soft temporal assignment。静态重复视频的
frame innovation与event value都为零；所有dynamic value projection与MLP均无bias，因此language、owner或位置不能凭空制造动态。
多视频先独立执行frame-event alignment，最后在signed pooling的集合归约中等权聚合，保持视频置换不变。

### 4.5 Direct native factor readout

最终rank query产生静态base logits；每个frame对齐后的dynamic state分别产生该frame的X侧与Y侧contrast logits，再对完整
frame x probe x 50-horizon x bank-type raw native values做一次G1同类exact signed pooling。没有在此之前对bank做第二次attention，
也不把全视频压成一个dynamic query广播回所有frame。静态重复视频时两侧正负分布都相同，A、B和完整mobile update均为零。
不存在独立gain网络、factor normalization、family scalar或事后方向修正。

唯一post-pooling操作是既有per-target `B@A` RMS cap；small-core canonicalization只用于把rank4稳定物化并与rank12 carrier合并，
不改变功能矩阵。

## 5. 为什么保留或删除这些部分

- G1的`114/250`、breadth`5/5`、Goal2、Long1证明真实native X/Y、signed pooling和task-local rank4可产生有效闭环变化，因此保留。
- G2的full相对endpoints改善`22.2047%`、probe`38/40`和same-task/K1/K4通过，证明PI0.5 response和有序视频动态有信息，因此保留
  原生response、真实frame time和ordered events，但不把G2固定schema当成下游瓶颈。
- task72 task-local从carrier`34/50`到m100`40/50`，且functional恢复跨held video保持，证明functional-to-LoRA-to-behavior与
  evaluator可工作，因此继续使用correct cross-episode functional作为直接目标。
- 多个shared family从约`35--45/250`徘徊且Goal/Long为0；最新role-equal m100也仅`45/250`。它们共同指向shared映射接口，
  不支持继续修补gain、relation、normalization或采样比例。
- causal process auxiliary即使学到时序信号也没有带来闭环收益，并多次与functional信用冲突；当前首版删除辅助目标，让最终功能
  梯度直接训练整个Writer。时序理解仍由PI0.5 response和temporal blocks承担，不等于删除视频时序。
- 历史v5/v6类neural Writer说明神经主干能获得一定收益，但稳定性和video necessity不足；本设计吸收其可扩展learned trunk，
  同时用ECP原生响应、信息墙、raw bank value路径和因果controls约束捷径。

## 6. 简洁性与生命周期规则

首版固定为`2 Frame + 2 Temporal + 1 Event + 2 Frame-Aligned Factor` blocks、宽度128。这里的数字只是短资格配置，不是owner永久要求。

- 扩容只允许优先增加width、heads或复制现有block；不能为一个负结果追加新的专用摘要、解析solve、归一化链、校准器或gate。
- 如果某一接口被复核为最早失效点，下一版本应删除并替换该责任模块；不得保留旧模块再在其前后补偿。
- 手工运算只允许服务明确边界：axis/mask、可逆probe换基、一次content centering、exact chunk reduction、signed pooling、最终cap与
  rank materialization。
- active tree只保留这一条canonical Writer图。历史实现由Git、sealed configs、formal artifacts和research history保存。

## 7. 训练与裁决

component initialization只复用G2已验证的prefix/response projections、owner/family/layer/horizon embeddings及第一个response
attention；Frame/Temporal/Event/Composer主体为fresh。Final必须同时保留同拓扑fully-random、fresh optimizer/scheduler候选。

首轮按成本递增：

1. 真实full-50 forward/functional VJP/materialization smoke，确认所有learned模块有非零梯度、冻结policy零梯度、唯一rank16；
2. task1/task93短task-local Composer容量控制，检查相反难度任务上的frame-aligned readout容量；它不单独裁决shared end-to-end trunk；
3. 通过正控后使用最小task-disjoint shared资格，观察gradient tasks与真正leave-task-out tasks的fit/held functional方向；
4. 只有出现task-disjoint正信号才运行held5 correct-only strict250；明显坏结果不靠长跑挽救；
5. 共享信号成立后再覆盖mixed-K、fully-random Final与更长训练，并进入validation8 strict paired400；
6. 只在correct-only选定并冻结checkpoint后运行same-task-other、wrong、no-video、language-only、first/final、shuffled/reversed controls。

task batch、meta/target是否同时出现及比例都属于具体配置，不是架构常量。任一阶段的负结果先定位最早失效接口、检查指标与正控，再以
一个主要因果变量作优雅替换；不以人为内部loss门槛代替闭环结果，也不无限扩散搜索。

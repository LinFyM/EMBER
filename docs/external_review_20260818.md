# EMBER Fixed-Commit External Independent Review

状态：2026-08-18归档的外部独立复核记录。复核对象为
`codex/bci-continuation@947c0e308c0b16bea97f0a3d157a3fe7b570a074`。本文忠实保留专家的核心判断、反证、
实验建议和证据缺口，并单独标记仓库侧已经核验的代码事实。它不是active design、run contract或自动launch授权。

## 1. 证据边界

- 专家能读取固定提交的canonical Writer、训练入口、配置、测试和tracked文档；
- 专家不能读取本地`runs/`、checkpoint、逐行rollout、optimizer state或ignored artifacts；
- 本文中的实验结果来自仓库tracked文档，不能被外部独立复算；
- McNemar数值由专家根据公开的paired gains/losses重新计算，仓库侧已用SciPy复核；
- “代码事实”“实验观察”“因果推断”“建议干预”必须分开解释。

## 2. 专家的一页式裁决

专家认可EMBER问题定义与信息墙：exact language指定任务、对象、关系和目标；action-hidden videos提供过程、状态变化、
时序和policy-relevant evidence；Action representation把过程放入frozen Action Expert可解释的坐标；memory states提供
按policy layer组织的动态Value；Writer一次生成一个完整LoRA并在未知初始化闭环执行。

当前LMMPC已经证明：

1. 信息墙和source freeze基本正确；
2. action-hidden视频能影响内部表示；
3. Core-addressed Reader具有真实closed-loop正收益；
4. fresh Writer能生成非平凡完整LoRA。

当前LMMPC尚未证明：

1. Program表示高层过程而非task identity、static cue或nonzero-video carrier；
2. memory layer/rank index具有policy-functional语义；
3. FactorHeads稳定覆盖所需policy-effective方向；
4. static expert-state functional loss支持generated-policy rollout occupancy；
5. 一个shared checkpoint能保留广泛task support；
6. correct视频在closed loop中优于wrong、shuffle、reverse和no-video。

专家将当前问题分成三个优先级：

1. fresh过程前端的functional credit被切断，且correct-only合同允许language/static shortcut；
2. offline expert-state credit与rollout occupancy和support retention不一致；
3. Program到完整LoRA的共享FactorHead坐标存在可达性和co-drift风险。

第一项有最直接代码证据；第二项最能解释macro25到50崩落；第三项最能解释“新增support存在，但丢失更多旧support”。
专家明确认为123低上限与后续checkpoint漂移不应被一个单一根因吞并。

## 3. 最重要的代码发现：fresh前端输出被再次detach

### 3.1 专家指出的路径

`LayerMatchedBackboneMemoryEncoder`先将frozen backbone的`prefix_hidden`和`action_hidden`detach，再送入
Writer-local `_project_joint_evidence`。这个第一次detach符合source freeze。该投影内部运行：

```text
language_projection(packed language hidden)
language_projection(image patch hidden)
patch_grounding(text query, projected patch content)
interaction_projection(mean(Action hidden over 50 tokens))
```

但`forward()`在返回`LayerMatchedVideoEncoding`时又执行：

```text
frame_evidence=evidence.detach()
grounded_evidence=grounded.detach()
interactions=interactions.detach()
```

### 3.2 仓库侧核验结果

仓库侧逐行核验确认专家判断成立：

- `patch_grounding`和`interaction_projection`是fresh、`requires_grad=True`的Writer参数；
- 第二次detach使二者不能从functional objective获得任何downstream gradient；
- `language_projection`仍通过独立text-query和后端language gate获得梯度，但逐帧视觉evidence分支对其不提供信用；
- Core、visual transition、Procedure和后端模块仍可在这些固定投影特征上训练，因此不能推成“视频完全没用”；
- optimizer按`requires_grad=True`收集这些参数，trainable contract只核对数量，不核对每个intended module是否真的有梯度；
- 当前所谓“all dynamic path receive gradients”测试使用fake backbone，并只检查FactorHeads、memory tokens、Reader、
  K-set等，没有检查`patch_grounding`或`interaction_projection`；
- 历史V6直接使用semantic encoder返回的evidence，没有这层输出detach；
- Git blame显示该行为随capacity-matched backbone memory路径引入，而非继承自V6。

这个事实把最早可疑接口前移到：

```text
frozen native hidden
    -> fresh grounding / interaction projection
    -> Core / Procedure
```

它是强工程证据，但仍未做matched closed-loop干预，不能直接宣称它单独造成123、Long/Object缺口或123到84漂移。

## 4. 流水线逐接口审计

### 4.1 Language

专家认为language应做task/object/relation/goal消歧，为Core和Reader提供Query，但不得单独写出task LoRA。当前没有literal
zero-video language-only LoRA，因为zero dynamic memory保持identity；但language gate对所有nonzero memory主值和
Core修正逐维相乘，存在“任意非恒定视频只提供通用carrier，language决定task方向”的条件式科学旁路。

固定提交的tested architecture仍使用rank4 Text Meta-LoRA、VL Meta-LoRA为0。owner在收到复核后明确：后续
canonical Writer移除Text Meta-LoRA；exact language仍通过冻结原生text/VLM表示和Writer-local读取进入Program。
该决定不自动取消Action Meta-LoRA。

### 4.2 Video与Core

Core使用逐帧evidence均值和task-query对中心化视觉evidence的注意力。由于evidence来自language-conditioned frozen
multimodal hidden，Core适合承载task语义，也容易被task identity、static object/scene cue主导；K个同task Core又在
condition内均值。仅有Core稳定不能证明高层过程形成。

### 4.3 Action representation

使用真实image/language prefix和固定noise/time Action suffix，不读取teacher action，符合信息墙。但最终把50个
Action token hidden直接均值再投影到256维，会丢失horizon内部位置、接触/释放阶段和多模态动作方向。专家将其视为
合法state probe，不自动承认它是完整动作过程表示。

### 4.4 Procedure

Procedure读取相邻grounded visual差分，由Action probe作为Query，再经过causal temporal encoder。它具有顺序敏感、
相邻状态变化、policy context和causal organization，且tracked数据证明reverse/stage perturbation会改变表示。

这些只证明路径live，不证明变化方向有用，也不证明correct order改善闭环。历史v4的shuffled优于correct是直接警告：
内部order sensitivity可能编码错误shortcut。

### 4.5 Native layer/rank memory

one-way mask与source freeze通过；未发现teacher action、reward、terminal、outcome、expert route或old Writer leakage。
memory token保存确定性的layer/rank index，但当前存在多套未绑定坐标：memory token index、Reader layer/rank identity、
Core fusion identity、M2P identity。成立的是index preservation，不成立的是“第r个token天然是第r个policy skill”。

### 4.6 Reader时间Value

Reader执行：

```text
relative[t] = projected[t] - projected[first]
dynamic[t]  = relative[t] - mean_t(relative)
```

代数上等价于`projected[t] - mean_t(projected)`，首帧项最终抵消。因此它是时间中心化，不是保留相对首帧位移。
优点是静态memory被消除，迫使Value使用动态差异；代价是绝对状态、终点和持续目标信息从memory Value消失，只能由
Core另路补充。Procedure只能选择native memory已有Value，不能创造native memory中不存在的policy direction。

### 4.7 K-video set

K=1恒等，K>1以same-address均值为anchor，correction受每cell RMS限制，结构上优于无界set mixer。现有证据证明
permutation invariance、K1 identity、可微和same-task consistency，尚未证明它提高strict closed-loop均值。

### 4.8 Core fusion、M2P和endpoint rows

bounded M2P解决了历史unbounded overwrite，但带来独立group/rank identity。最终per-cell RMSNorm消除Program
magnitude，若magnitude表示置信度或一致性，FactorHeads无法直接读取。18个expert rows有native layer correspondence，
action-in row由第一层expert cell线性派生，action-out由最后一层派生，二者不是真正独立native endpoint memory。

### 4.9 FactorHeads

八个family共享`256 -> 256 -> output_width`头，末层零初始化。专家指出：

- 对宽度2048的q-B row，给定head参数时输出位于末层至多256维子空间；
- LoRA identity为`A=A0, B=0`，因此identity点B有一阶policy gradient而A没有；
- head末层又从零开始，使B-family先打开，上游Program和A-family信用较晚；
- macro25到50的cross-decode中heads-only变化`1.320`高于Program-only`.582`，与early decoder-dominated dynamics相容；
- 历史V6/LPCP使用同一FactorHead类达到143，反驳“FactorHead形式先验不可能”。

专家的准确假设是当前fresh front end、native memory coordinates和共享heads的联合校准/保留可能有问题，而不是直接
要求扩大head、改变rank或宣称decoder唯一有错。

### 4.10 Training objective

B20并非反复固定20行；sampler跨50 episodes和progress strata轮换，且排除teacher-video episode。functional
autograd链本身没有被专家发现算错。更关键分布差异是：训练状态来自expert demonstration occupancy，部署状态来自
generated policy自己的occupancy；早期动作误差会把后续状态推离监督分布。

## 5. 对absolute缺口和checkpoint漂移的统计复核

专家复算并由仓库侧确认：

| paired comparison | gains | losses | two-sided exact McNemar p |
| --- | ---: | ---: | ---: |
| current123 vs LPCP143 | 23 | 43 | `.018657` |
| current123 vs GOMQ151 | 23 | 51 | `.001516` |
| current macro25 -> 50 | 13 | 52 | `1.1688e-6` |
| macro50 -> 75 | 30 | 25 | `.59005` |
| macro75 -> 100 | 17 | 19 | `.86794` |
| v4 104 -> current123 | 38 | 19 | `.016348` |
| fixed memory135 -> GOMQ151 | 29 | 13 | `.019520` |

所以：

- current123并非没有新support，而是丢失旧support显著多于新增；
- macro25到50是结构性崩落，后两段是在低位换手而非持续统计显著下降；
- macro25 success=`3/3/44/25/1/43/3/1`，top3贡献91.1%；breadth@1=8掩盖实质breadth集中；
- 专家建议同时报告breadth@1/@5/@10、per-task histogram、suite minimum和top3 concentration；
- macro25低上限更可能受前端credit/shortcut与decoder可达性限制；25到50崩落更可能受occupancy/retention影响；二者
  可以相互放大但不应强行同因。

## 6. 三个根因的证据、反证和不确定性

### 6.1 前端credit切断与task/static shortcut

直接证据：三项输出detach、fresh模块、旧V6无同样detach、梯度测试缺口、correct-only监督、language gate权限。

尚属推断：Program主要学习task identity、wrong nonconstant video可作通用carrier、static Core/language主导123、
detach是143/151差距的主要来源。

反证/限制：constant video严格identity；reverse/shuffle改变内部Program；Reader单变量带来+19；frozen source表示和
随机投影可能已经保留足够信息。

### 6.2 Offline occupancy与closed-loop retention错配

直接证据：13 gains/52 losses；同期19/24 train task B20改善；global loss下降而strict净丢39；只有49 rows四点
始终成功；52 lost到macro100只恢复15；sampler并非重复窄B20。

尚属推断：lost rows在generated-policy特有状态失败、早期误差造成occupancy shift、expert-state direction牺牲窄
support、Adam moment交叉污染。

反证/限制：offline panel覆盖50 episodes/strata；它曾产生123/143/151；macro50到75恢复部分support；历史若干
occupancy/reward尝试未稳定解决。

### 6.3 FactorHead可达性与co-drift

直接证据：23 gains对43/51 losses；shared head与256维末层；B-first冷启动；25到50 heads-only变化最大；endpoint
rows非独立native memory。

尚属推断：有效LoRA在head manifold之外、head条件数差、共享dictionary改写旧task、absolute低点主要由decoder造成。

反证/限制：同一head类历史143；当前仍新增23 rows；所有family有梯度和非零输出；后两段Program/head变化接近。

### 6.4 专家对既有归因用语的修正

- `task drift`只是结果描述，必须定位task、state、失败阶段和具体模块；
- raw Procedure趋同有LPCP143反例，不能单独解释失败；
- `functional loss mismatch`方向合理但要拆为offline state cotangent、shared Writer update、generated occupancy和retention；
- 只证明当前objective+parameterization+AdamW组合失败，不能直接归罪arithmetic mean或AdamW；
- `memory correspondence`应限定为index preservation；
- “全动态路径梯度接通”对当前fresh grounding/interaction并不成立。

## 7. 专家提出的最小判别实验

以下全部是外部advisory proposal；未写入active design或run contract。

### F.0a 全前端梯度审计

在fresh initialization和macro25各取一个canonical task step，记录每个intended module的gradient是否`None`、是否
非零、是否finite。至少覆盖：

- `patch_grounding.query/key/output`；
- `interaction_projection`；
- `language_projection`；
- Text/Action Meta-LoRA；
- Core、Procedure、memory tokens、Reader、K-set、M2P；
- 八个FactorHeads；
- source policy nonzero gradient tensors必须为0。

专家预期当前`patch_grounding`与`interaction_projection`无gradient。该审计只纠正mechanism事实，不是性能gate。

### F.0b macro25完整视频因果面板

对同一macro25、fixed states、policy RNG和K运行correct、same-task-other、cross-suite-wrong、shuffled、
shuffled-keep-first、reversed、no-video strict paired400。

专家建议的过程因果支持门：correct相对每个negative净gains至少15、exact McNemar`p<.05`、无suite大幅反向、
same-task-other总分在correct±10且保留correct success rows至少90%。shortcut警报包括wrong nonconstant video与correct
差小于10，或correct对reverse/shuffle不显著。上述数值是专家建议，不是既有项目资格合同。

### F.1 只恢复本地过程前端gradient

唯一代码变化是移除返回处三项输出detach，同时保留frozen backbone hidden detach、source policy freeze、Writer拓扑、
rank16、Reader、K-set、M2P、FactorHeads、数据、B20、optimizer、LR、seed和K schedule。新增测试要求source policy
零梯度且`patch_grounding`/`interaction_projection`非零梯度。

专家建议根因支持门：相对当前macro25 paired net至少+10、`p<.05`、实质breadth增加、视频controls改善。更严格方法门：
macro25接近143、macro50与macro25差不超过10、续训不再显著losses>gains、至少6 tasks达到5/50、same-task-other
保留率至少90%。若梯度恢复后strict净变化<5且controls/breadth/stability均不改善，则停止围绕detach调参。

### F.2 Occupancy mismatch固定状态反事实

先不训练新模型。保存macro25到50的52 lost、13 gained与retained rows的两checkpoint rollout states，构造固定
`S25 union S50`，用冻结task expert或既定teacher policy作为action reference，比较两个LoRA在同一状态并集上的
action/flow error。

支持occupancy解释：macro50在offline B20更好，但在lost-row rollout states更差，且差异在failure前出现，gained
rows相反。若macro50在offline、macro25 occupancy和macro50 occupancy都更好却仍失败，则转向action loss与success
错位、execution/replanning、decoder关键tail或非均值failure。只有审计支持后才考虑以预冻结occupancy-matched panel
单变量替换B20；最终仍用strict paired400裁决。

### F.3 FactorHead co-drift

从macro25作诊断性resume，唯一变化为冻结八个FactorHeads，继续Program/Reader到macro50。支持标准建议为lost≤20、
保留macro25 success至少90%、macro50≥110且breadth不降。若仍接近原规模崩落，则主要责任转向upstream、objective/
occupancy、shared gradient或fixed heads已有reachability问题。该分支是机制诊断，不是fresh最终方法。

### F.4 Decoder reachability oracle

只在train24 task-expert bank上：冻结macro25 FactorHeads，为每task直接优化自由`20x16x256` Program以逼近已知
policy-effective expert LoRA，并在train-task closed loop比较投影前后。若充分优化后仍明显丢失expert success，支持
head reachability bottleneck；若保留≥90%，问题主要在video-to-Program或credit。该oracle不得成为deployment carrier
或正式held分数。

### F.5 Shared-gradient conflict

只有前述证据仍不能解释时，保持per-task gradient、AdamW、LR、tasks和数据，只把arithmetic mean替换为预注册、
不扫参的conflict-safe aggregation。macro25相当而25到50 lost显著减少才支持shared-gradient conflict；若仍漂移，
回到occupancy或decoder。专家明确建议最后才做。

### 专家建议顺序

`F.0 -> F.1 -> 条件化F.2/F.3 -> F.4 -> 最后F.5`。不优先改rank16、memory token数量、BF16、LoRA scale、
M2P blocks、seed或做大规模超参数搜索。

## 8. 专家认为远程仓库仍缺少的证据

1. 四checkpoint与LPCP/GOMQ的逐行paired400矩阵、video IDs和RNG reference；
2. current macro25完整视频controls与不同K4 sets；
3. intended trainable modules的实际gradient表及首次非零macro；
4. 每个formal run的commit、dirty diff、source/config/checkpoint/metrics provenance；
5. Program、Reader、Core、Procedure、memory tokens、M2P和各FactorHead family的per-module delta及closed-loop
   cross-decode矩阵；
6. lost/gained/retained rows的rollout occupancy、首次行为分歧、固定状态union error；
7. task-expert LoRA在current head manifold上的投影误差、投影后closed-loop和family-wise reachability；
8. 区分objective、gradient mean、Adam moment、shared heads和shared Program的matched conflict intervention；
9. breadth@1/@5/@10、per-task histogram、suite minimum与top3 concentration。

## 9. 仓库侧综合边界

外部复核最重要的新增价值不是提出了又一套大架构，而是发现了一个比现有“B20 cotangent到retention”更靠前、可由
代码直接证明且可单变量修复的credit断点。它不推翻Core-addressed Reader、Dynamic-K、rank16、bounded K-set/M2P
或memory主链，也不授权返回V6/LPCP/GOMQ。

同时，它没有证明：移除detach必然回到143、Text Meta-LoRA是123的唯一shortcut、FactorHeads需要扩大、occupancy
一定是漂移唯一根因，或其建议的所有数值门槛必须成为项目合同。任何实施前仍需由`task_plan.md`和`progress.md`
登记active design与单变量边界。

# EMBER Current Evidence Synthesis for Independent Review

状态：2026-08-18远程复核版。本文只综合当前可证实事实、分析边界和仍未区分的问题；**不推荐下一步架构、
objective或实验**，也不授权恢复任何历史执行路径。

外部reviewer只需读取远程仓库即可理解当前问题。大型checkpoint、raw rollout视频和`runs/`本地artifact没有提交；
所有影响当前判断的aggregate、paired transition和stage结果在本文与`research_history.md`中重述。

## 1. 研究问题与资格标准

EMBER研究以下映射：

```text
exact task language
+ one or more internally ordered, action-hidden teacher videos
    -> one shared Writer runs once
    -> one complete 38-target task-conditioned LoRA
    -> one frozen PI0.5-LIBERO source policy
    -> unseen-initialization closed-loop success
```

语言定义任务关注点和目标，视频必须提供正确动态过程。Writer不能读取teacher action、state、reward、task ID、
filename或policy outcome。video与action监督episode同task但跨episode，避免逐帧轨迹复制。

方法资格不由单一峰值决定。长期目标是strict paired correct超过`150/400`；稳定约145也可构成有价值结果，但必须
同时具有相邻checkpoint低churn、高breadth、same-task不同video鲁棒，以及correct相对wrong、shuffled、reversed和
no-video的因果优势。当前最新方法未达到absolute门，因而未做六臂controls。

## 2. 当前被测架构：Core-Addressed Reader

当前代码实现和最新正式训练均属于EMBER-LMMPC Core-Addressed Reader。它是从fresh Writer初始化生成完整rank16
LoRA的统一架构，不加载旧V6/LPCP/GOMQ Writer，也没有第二套adapter或checkpoint融合。

### 2.1 完整数据流水线

```text
exact language L + K same-task ordered videos, K in 1..4, frame stride 5

for each video independently and each sampled frame:
  native image/language context
  + 50 fixed Action probes
  + 16 one-way memory queries
    -> frozen native VLM/Action context evidence
    -> Action representation A[t]
    -> layer/rank memory M[t, layer=18, rank=16]

task-grounded visual evidence over frames
    -> order-invariant Semantic Core C_video

adjacent grounded visual transitions queried by A[t]
    -> ordered causal Procedure P_video[t]

for every fixed (layer, rank) address:
  task Core supplies the Query
  ordered Procedure supplies temporal Keys
  centered native memory supplies Values
    -> per-video parameter memory H_video[layer, rank]

same-address K-video aggregation
    -> mean-anchored bounded H_set[layer, rank]

nonzero dynamic memory gates Semantic Core content
    -> 18 x 16 addressed grid
    -> add action-in/action-out boundary rows
    -> bounded 20 x 16 group/rank axial M2P
    -> eight native factor-family heads
    -> one complete 38-target rank16 A/B LoRA
```

### 2.2 四条信息流的职责

| 信息流 | 当前职责 | 被禁止的旁路 |
| --- | --- | --- |
| language | task grounding、Core Query和必要gate | language-only直接写LoRA |
| ordered video | 对象变化、阶段和有向Procedure | first-frame、absolute-time或video-presence shortcut |
| Action representation | 让source policy先验解释当前视觉变化 | teacher action或fake/no-context Action forward |
| memory states | 为每个Action-Expert layer与LoRA rank提供native contextual Value | flat payload或第二套parameter-address bank |

memory token在这里解决的是layer/rank correspondence，不等价于保证对应地址上写出的LoRA方向正确。数值上的
`20 x 16 = 320`只是最终parameter-grid cells，不是额外320个memory tokens。

### 2.3 聚合和decoder

- 每条video先独立形成有序Procedure和`18 x 16`parameter memory；
- K轴仅在相同layer/rank地址上做置换不变聚合，K1严格返回单video表示；
- learned K-set correction被限制在per-video mean每cell RMS的`.5x`以内；
- M2P correction同样被限制在输入addressed grid每cell RMS的`.5x`以内；
- Core只在非零动态memory和language gate共同存在时进入Value；
- 八个bias-free FactorHeads分别生成q/v/action-in/action-out的A/B，最终仍是一套rank16 LoRA。

当前实现位置：

- `src/ember/writer/model.py`：完整Writer和一次性encode/decode；
- `src/ember/writer/backbone_memory.py`：真实native context与one-way layer memory；
- `src/ember/writer/temporal.py`：Semantic Core和causal Procedure；
- `src/ember/writer/parameter_grid.py`：Core-addressed reader、K-set、Core fusion和M2P；
- `configs/pi05_writer_layer_matched_memory_program_compiler_v5.json`：正式冻结recipe。

config顶层`status=active_formal_ready`是该formal run启动时写入并随run冻结的历史字段，不表示当前仍有active run；
当前authority是`progress.md`。

## 3. 当前训练与评测合同

训练只用固定train24，每macro对24 tasks等权；K1/K2/K3/K4各覆盖6 tasks。每个condition使用同task跨episode的
20个action queries，在冻结source policy上优化correct-order dense functional loss。没有matching、reverse/shuffle
训练臂、reward、expert reconstruction或LoRA几何正则。

正式run从fresh Writer训练到macro100，在相同world6/topology下于macro25/50/75/100保存完整checkpoint。评测为同一
K4 teacher schedule、同一400个task/state/RNG rows的strict closed-loop panel，因此四点之间可以逐行配对。

## 4. 历史强结果与结构差异

| 方法 | strict轨迹 | 结构上与当前方法最重要的差异 |
| --- | ---: | --- |
| v6-fast | best`143`; 后续`131/130/132/126` | 历史Core/Procedure与native rank16 compiler；训练仍漂移 |
| LPCP | `143`, breadth7 | 在强V6图上形成小幅layerwise Procedure改写；BA仍接近历史carrier |
| GOMQ | `151 -> 135 -> 131`, breadth始终6 | 保留LPCP rank16载体，另加rank16 direct-B residual；不是fresh完整LoRA重建 |
| 当前Core-Addressed Reader | `123 -> 84 -> 89 -> 87` | fresh统一生成全部rank16 LoRA；当前reader和decoder共同训练 |

历史151是有效的absolute证据，但不是合格稳定方法：它随后连续回落且没有六臂controls。其结构差异也意味着，不能
把151直接解释成“同一fresh完整Writer比当前更强”；它证明的是learned memory residual曾在一个已有强support上
产生真实增益。

## 5. 当前正式结果

### 5.1 四checkpoint轨迹

| macro | strict | breadth | per-task | per-suite Spatial/Object/Goal/Long |
| ---: | ---: | ---: | --- | --- |
| 25 | `123/400` | 8 | `3/3/44/25/1/43/3/1` | `6/69/44/4` |
| 50 | `84/400` | 5 | `0/1/45/1/0/29/8/0` | `1/46/29/8` |
| 75 | `89/400` | 6 | `3/0/36/1/2/44/3/0` | `3/37/46/3` |
| 100 | `87/400` | 4 | `0/4/38/0/0/42/3/0` | `4/38/42/3` |

相邻transition：

| transition | retained | gained | lost | churn | net |
| --- | ---: | ---: | ---: | ---: | ---: |
| 25 -> 50 | 71 | 13 | 52 | 65 | -39 |
| 50 -> 75 | 59 | 30 | 25 | 55 | +5 |
| 75 -> 100 | 70 | 17 | 19 | 36 | -2 |

400个固定rows中只有49行四点始终成功，150行至少成功一次。macro25到50丢失的52行，到macro100仅恢复15行；
macro25到50新增的13行，到macro100只保留6行。后期churn下降发生在更窄的breadth上，不能解释为共同积累。

macro25的breadth8也高估了有效覆盖：Object1、Object3和Goal6贡献`112/123=91.1%`的成功，其余五task各只有1到3次。

### 5.2 当前123与两个同schedule强基线的严格配对

| reference -> current123 | retained | current gained | current lost | net | churn |
| --- | ---: | ---: | ---: | ---: | ---: |
| LPCP143 -> current123 | 100 | 23 | 43 | -20 | 66 |
| GOMQ151 -> current123 | 100 | 23 | 51 | -28 | 74 |
| GOMQ131 -> current123 | 92 | 31 | 39 | -8 | 70 |

当前123相对GOMQ151的suite差为：Spatial `+3`、Object `-12`、Goal `+4`、Long `-23`。逐task最大缺口是Long1
`26 -> 3`，其次Object3 `34 -> 25`。因此28分缺口并非所有task均匀变弱，而是长程/多阶段与部分Object support被
明显换出，同时Spatial/Goal有少量新增。

相对共同LPCP143基线，GOMQ151为`126 retained / 25 gained / 17 lost`；当前123为
`100 retained / 23 gained / 43 lost`。两者新增成功数接近，差异主要来自当前完整Writer丢失了更多已有success rows。
这项分解描述的是观测到的support replacement，不能单独确定责任位于Program、decoder、objective还是optimizer。

## 6. 表示链与LoRA证据

### 6.1 已经成立的正证据

- 从旧reader到Core-addressed reader，只改变reader接口，macro25由`104`提高到`123`；严格配对为
  `85 retained / 38 gained / 19 lost`。reader改动具有真实closed-loop收益。
- validation8中raw Procedure的correct/reverse relative-L2约`.574`，reader后的H_set约`1.012`，compiled约
  `1.133`；当前reader没有继续抹掉已存在的有向差异。
- macro25同task四个K4 conditions在H_set/compiled/effective-BA上的cosine约`.989/.986/.993`；当前失败不是
  same-task不同video updates彼此正交相消。
- macro25到50的LoRA norm从`27.225`增至`43.999`，effective targets从`16.88`增至`19.50`，说明Writer仍在产生
  material update，而非停在identity附近。
- bounded K-set和bounded M2P关闭了早期revision中已经实测的representation overwrite。

### 6.2 functional loss与closed-loop分离

四点固定K4+B20 loss为：

```text
macro25  .112124
macro50  .099353
macro75  .098427
macro100 .101337
```

25到50期间19/24 train tasks的固定panel loss改善，strict却净丢39。75到100时固定panel和held strict才同时恶化。
因此训练确实优化了当前functional objective，但该objective的改善不能选择held closed-loop checkpoint。

### 6.3 Program与FactorHeads都在移动

相邻compiled Program relative-L2为`.770/.730/.710`。交叉解码结果：

| transition | 仅换FactorHeads的BA rel-L2 | 仅换Program的BA rel-L2 |
| --- | ---: | ---: |
| 25 -> 50 | 1.320 | .582 |
| 50 -> 75 | .676 | .583 |
| 75 -> 100 | .585 | .575 |

早期norm扩张主要由FactorHeads放大；后两段Program和FactorHeads贡献接近。现有证据不支持“只冻结FactorHeads”或
“只修Procedure”即可解释整条轨迹。

### 6.4 Procedure趋同的准确边界

raw Procedure随训练确实趋同：between-task cosine从macro25约`.954`升到macro50约`.972`，temporal centered
energy下降。但历史LPCP的raw Procedure更趋同（between-task cosine约`.997`）仍达到143；当前reader又把较小的
correct/reverse差异放大到H_set/compiled。因此Procedure趋同是风险和视频因果性疑点，但不是当前123到84之间最早
被观测到的task/order抹除接口。

### 6.5 外部复核发现的fresh前端gradient断点

固定提交外部复核后，仓库侧确认当前layer-matched memory路径在已detach frozen backbone hidden并完成fresh
`language_projection`、`patch_grounding`和`interaction_projection`之后，又把返回的
`frame_evidence/grounded_evidence/interactions`全部detach。因此`patch_grounding`和
`interaction_projection`不能从functional objective获得gradient，`language_projection`的逐帧visual分支也被切断；
现有“dynamic path gradients”测试没有覆盖这些模块。历史V6 semantic forward没有这层输出detach。

这项代码事实说明当前Core/Procedure是在固定fresh投影特征上学习，而非整个process front end端到端接受policy credit。
它是比此前归因更靠前的未解接口，但尚无matched closed-loop干预，不能据此单独解释123、Long/Object缺口或后续漂移。
完整外部判断、反证和建议实验见`docs/external_review_20260818.md`。

## 7. 当前证据支持、排除和未决的解释

| 类型 | 结论 |
| --- | --- |
| 已确认 | 当前recipe绝对峰值只有123，且四点没有恢复到该峰值；它同时存在弱峰值和checkpoint漂移两个问题 |
| 已确认 | reader是正机制；memory layer/rank index preservation、Dynamic-K、bounded K-set/M2P和rank16完整输出均已工程接通 |
| 已确认 | 当前123相对强基线获得约23个新success rows，但丢失43到51个旧success rows；主要缺口集中在Long和Object |
| 已确认 | 继续优化静态B20 loss没有形成held多task共同积累；Program和FactorHeads都承载了有限长更新 |
| 已确认 | fresh `patch_grounding`和`interaction_projection`被输出detach切断functional gradient；现有gradient门没有检查它们 |
| 已排除为单因 | Writer没有写出、LoRA仅仅太小、同task视频相消、K-set/M2P仍无界覆盖、训练只是不够久 |
| 已排除为单因 | raw Procedure趋同、FactorHeads单独漂移、rank16容量不足或LoRA norm不足 |
| 未决 | 静态cross-episode B20 query occupancy与真实rollout occupancy的错配占多大责任 |
| 未决 | 当前Program是否提取了高层过程，还是主要稳定了task identity/static cues |
| 未决 | 恢复fresh前端gradient能改善多少absolute closed-loop、breadth与视频因果性 |
| 未决 | Program到native A/B坐标是否存在系统性可达性、条件数或高维输出先验问题 |
| 未决 | shared optimizer如何在同一checkpoint保存跨task support，及其与objective/decoder的交互 |
| 未决 | 当前方法的correct视频是否真实优于wrong/shuffle/reverse/no-video；因absolute不足尚无六臂数据 |

## 8. “Task drift”不能覆盖全部现象

当前至少要区分两个时间尺度：

1. **单checkpoint的support形成/替换**：最佳点123已经比143/151低，且配对分析显示新增量接近、丢失量更大；
2. **跨checkpoint retention**：123之后继续训练跌到84/89/87，success rows和tasks持续轮换。

历史151也有第二种问题，但它通过保留已有rank16 carrier减轻了第一个问题。当前方法从fresh生成完整LoRA，因此它的
绝对缺口不能仅用“后续漂移”解释。反过来，结构差异也不能证明旧carrier是正确答案：151仍不稳定、没有视频因果
资格，并且owner明确禁止后续直接回退旧Writer。

## 9. 独立复核边界

- 后续判断必须以当前Core-Addressed Reader架构为基础，不得把直接恢复V6/LPCP/GOMQ当作回答；
- 历史checkpoint只作为comparison/provenance，不是当前参数、初始化或并行执行分支；
- 本文没有写入preferred successor、推荐objective或实验顺序；
- reviewer应独立判断上述未决项的因果优先级，并区分事实、推断和方法偏好。

精确历史演进见`docs/research_history.md`；当前架构公式与formal事实见
`docs/layer_matched_memory_program_compiler_design.md`；稳定目标见`docs/current_owner_requirements.md`。

# EMBER Architecture Reasoning After Consolidation

状态：2026-08-17认知重建与证据裁决已完成；本文不自动授权实现、训练或GPU实验。

本文不从某个模块名出发，而从EMBER要传递的信息和历史最早失败接口逐步推理。任何具体架构只有在这里的分支问题
被证据裁决后，才应单独建立design authority。

## 1. 先固定问题，不先固定方法

输入：

```text
exact task language + K action-hidden correct videos
```

输出：

```text
one complete task-conditioned LoRA for one frozen source policy
```

部署：Writer运行一次，policy在未见初始化闭环执行。当前评价的是初始LoRA，不包含生成后的task-local RL。

这里没有预先规定K、memory token、rank、Action Expert调用方式或decoder。它们必须由信息流和失败证据推导。

## 2. 输入里真正存在的四类信息

### 2.1 Language query

语言确定任务对象、关系、关注点和目标。它适合提供语义query或address，不应独立成为LoRA Value。

### 2.2 Static visual context

单帧可识别对象、颜色、位置和场景，但无法单独确定“怎样从初态演化到目标态”。静态信息需要服务于语言query，
不能成为language+first-frame shortcut。

### 2.3 Directed process

视频内部的状态变化、阶段边界、接触事件和先后依赖，是correct/shuffle/reverse区别的来源。每条video必须在这一维
保序建模。

### 2.4 Cross-video invariants

多个同task videos之间共同保留的对象关系、目标和阶段结构，更可能是可迁移Program；不同的路径、速度和视角更
可能是nuisance。聚合应发生在有语义的Program证据上，而非raw frames或最终LoRA。

## 3. 输出端真正需要表达什么

LoRA不是一串任意参数。38 targets位于不同policy layers和q/v/action families，具有不同输入/输出维度和功能。
task-local experts和v6已经证明：native rank16、低stable rank、q-dominant与跨列coherent都可能policy-effective。

因此输出问题更准确地写为：

```text
task Program
    -> layer/family/target/rank-aware policy coordinates
    -> native A/B factors
    -> effective BA and action change
```

直接回归百万参数不是唯一方式；但降低rank、增加atoms或追求正交也不是自动解法。关键是Program坐标与policy功能
坐标是否对应，以及写出是否保留已有support。

## 4. 历史证据对四个接口的裁决

| 接口 | 已有正证据 | 尚未解决 |
| --- | --- | --- |
| Evidence extraction | v5/v6、LPCP能读语义、动态和顺序 | correct过程是否被理解成可迁移因果Program |
| Cross-video Program | K4、CMBG、GOMQ可形成高coherence同task表示 | 高coherence可能只是稳定错误mean；cross-task separability不稳 |
| LoRA compilation | task experts、v6/LPCP rank16、direct native heads均可material | material更新是否沿held policy-useful方向并保留support |
| Shared training | reward、endpoint和expert occupancy均可产生task-local下降 | 多task在相邻single checkpoints持续共同积累 |

这张表排除几种过度简化解释：

- 不能只说“视频没有被读取”；
- 不能只说“LoRA参数太多”；
- 不能只说“memory token失败”；
- 不能只说“训练轮数不够”；
- 不能只说“Writer容量不够”。

## 5. 为什么类SHINE路线既不能照搬，也不能简单否定

过去Dynamic-K rank8 Backbone-Memory路线在真实图文/Action-probe context中加入memory并直接生成LoRA，工程上成立，
但strict只有100左右。失败组合同时包含：弱化absolute Semantic Core、rank8 fixed-A/Direct-B tail、特定mapper和
functional credit。因此它不能否定memory correspondence一般。

成熟Hypernetwork工作真正值得继承的是（结构参考分别见
[SHINE](https://arxiv.org/html/2602.06358v3)与
[Doc-to-LoRA](https://arxiv.org/html/2602.15902v1)）：

- 输入内容由一个本来就能理解该模态的backbone处理；
- 少量memory states与目标层/参数有明确对应；
- memory通过正常内容计算获得信息，而不是无context运行；
- decoder随backbone层数/target数可扩展。

EMBER不同于纯文本Doc2LoRA：π0.5的VLM/Gemma与Action Expert原生处理图像、语言、state和action denoising。任何
memory设计都必须说明它在真实native context中读什么、写什么、是否改变原policy计算；不能为了模仿论文在缺失
图像/文字prefix时空跑Action Expert。

## 6. Memory token应被问成三个问题

不是先问“要不要memory”，而是：

1. **放置**：memory位于输入内容encoder、policy layer observer还是LoRA target grid？
2. **通信**：它只读native context，还是双向改变原policy token？如果改变，step0 identity和source行为如何保证？
3. **职责**：它承载per-video有序证据、cross-video共同Program，还是LoRA target address？一个token不应同时承担
   四个未分离职责。

历史证据偏向一个约束：memory必须处于真实图文/Action context中，且最好有layer correspondence；但它是否作为
encoder token、one-way observer或compiler query仍未裁决。

GOMQ证明learned input query有用；其151→135→131说明不能沿用当前independent rank32 Direct-B tail并期待稳定。

## 7. 多视频数据流应先分轴，再决定注意力

自然数据结构有四个不同轴：

```text
video set K
  × ordered time/stage within each video
  × policy layer/family/target
  × LoRA rank/factor coordinates
```

这些轴不意味着每一对都要做全注意力。每个交互必须有职责：

- 时间轴：提取一条video的有向过程；
- K轴：寻找same-task不变量，置换不变；
- policy轴：把Program放入冻结policy可利用的层/功能坐标；
- parameter轴：生成native LoRA factors。

横向/纵向attention只有在对应上述职责时才合理。无目的地让四个轴全互联会同时增加复杂度、数据需求和归因难度。

## 8. Stage-wise证据矩阵

下面只使用已经封存的formal artifacts；几何指标用于定位，不用于选择checkpoint。

| 路线 | ordered / K-set Program | compiler / effective BA | closed-loop裁决 | 最早失效接口 |
| --- | --- | --- | --- | --- |
| Dynamic-K Semantic-Address rank8 | validation8的video Program、M2P input、final Program跨task mean cosine分别约`.495/.492/.529`；same-task比wrong更近为8/8，reverse/shuffle在Program中material | family hidden升到`.634`，dynamic-B/effective-BA进一步升到约`.779` | strict约`101` | Program仍有中等task区分，nonlinear family readout首先把task means压得更同向；“能区分顺序”仍不等于有用方向 |
| V6-LPCP | 同一native context的18层Action probes对reverse和constant都有material响应；same-task correction coherence均值`.618` | 相对AS139的all400 BA relative-L2仅`.002653`、cosine`.9999948` | `143`，相对AS139 gained/lost=`23/19`、churn42 | carrier通过；conditioned Procedure经过冻结fusion/compiler后只形成AS139邻域小修 |
| SJNV / SFMC类hidden commitment | task4 four-view hidden residual cosine约`.991` | 同一信号经冻结factor W2后raw-factor cosine约`.054`；held仅2/8通过 | 未进入formal或单点不稳定 | frozen nonlinear factor head可以直接毁掉已经共同的hidden方向 |
| Direct Joint Native-Factor residual | 绕过W2后validation8 BA 8/8健康 | native q/v/action写出material | `136`，相对LPCP lost23 | 生成端接通后，reward credit仍选择错误held方向 |
| GOMQ | learned memory相对fixed query带来真实`+16`；但isolated held memory contribution的same-task cosine仅`.127`，高`.983`主要属于完整residual；不能用`.993`相邻BA增量单独证明高层Program | rank32 direct-B使BA/action material；cycle2→3、3→4更新的same-task four-K4 BA cosine均约`.993` | `151→135→131`；cycle2→3=`122/13/29`，cycle3→4=`116/15/19` | direct tail解决“写不出”，但shared successful-occupancy Adam不保留held support；memory有用，不等于当前Program/credit已解决 |

证据路径分别落在：

- `semantic_mapper_stage_localization.json`：Program到family/B的between-task collapse；
- `v6_lpcp_macro0025_effective_ba_analysis.json`：LPCP到AS139邻域小改写；
- `sjnv_gate_stage_localization.json`：hidden到冻结W2的coherence断裂；
- GOMQ `metrics.jsonl`、相邻effective-BA和strict adjudication：direct写出后的shared retention失败。

## 9. 对三个首因假设的裁决

### 9.1 “视频没有形成任何Program”不是当前首因

它不成立为总解释。Dynamic-K在M2P/final Program仍保留中等between-task结构，LPCP能读layerwise顺序，GOMQ learned
query还产生过显著closed-loop增益。视频、顺序和policy context都能进入表示。

但“高层Program已经解决”同样不能成立。GOMQ isolated memory的held same-task cosine只有`.127`，且历史多次出现
reverse差异很大而correct不更好。因此Program质量仍需验证，只是不应再把下一轮主要变量放在换carrier或加更强
negative上。

### 9.2 Program到LoRA commitment是下一架构应改变的最早接口

两种极端tail都已暴露问题：

- 冻结V6 tail保留强support，却把新Procedure压成千分之几的小修，hidden residual还可能被W2旋散；
- independent direct-B tail让写出material，却脱离V6已验证的rank16 slot/family协同，并在连续shared update中换手。

下一步应把memory-derived Program放入**已有policy layer/rank坐标**，再由一套可共同训练的native rank16 A/B compiler
生成完整LoRA；既不只调旧Procedure Query，也不另开rank32 B-only bank。

### 9.3 Shared credit是第二个真实缺口，但不能与新compiler同时改

GOMQ cycle2--4的memory/downstream跨task gradient cosine约`.15→.13`与`.09→.05`，gained/lost BA幅度又不可分，
说明successful-expert occupancy不是可靠shared selection rule。可是如果下一轮同时更换compiler、reward、optimizer和
rank，结果无法归因。

因此首轮不续GOMQ reward，而以v6已验证的dense、task-complete、cross-episode functional training提供policy credit；
Program不可识别性另由下一节的结构与matching处理，不同时重写reward/optimizer。Writer reward仍开放，但只能在新
架构先证明absolute和相邻稳定性后另立authority；生成LoRA后的task-local RL继续不属于当前目标。

### 9.4 普通causal encoder仍没有解决positive-only不可识别性

v4根因审计已经给出形式化结论：训练样本是同task但条件独立的`(video_d, action_e)`，所有video又始终correct；因此
functional loss只规定task controller，不规定哪种video过程解释必须被使用。给输入加相邻差分、causal mask或位置
编码，只是提供使用顺序的能力，模型仍可退化成task identity或generic video-presence gate。

统一架构必须额外满足两点：

1. video additive Value本身去掉order-even/static旁路；LMMPC用
   `P(V)=0.5(F(V)-F(reverse(V)))`使reverse严格反号；
2. exact language必须匹配对应的directed Program；同task correct videos为正对，cross-task/shuffle/reverse只在
   Program matching中作为反例，不通过破坏negative LoRA制造闭环margin。

dense functional credit仍不可删除，因为matching只能保证“内容相符”，不能保证LoRA沿policy成功方向。

## 10. 三个可行分支与选择

### A. 原样续GOMQ direct-B

优点是曾到151且写出material；缺点是`151→135→131`已终局、rank32第二bank与低cross-task gradient coherence均未
解决。继续训练或小改LR不是新假设，淘汰。

### B. 只把memory加回LPCP Procedure Query

它最保守、step0容易完全等于143；但LPCP已经证明Query差异经冻结tail只形成`.002653`邻域改写，SJNV又证明冻结
W2会破坏共同hidden方向。若仍只走同一Query入口，最可能重复已定位的衰减，不推荐为主线。

### C. Layer-Matched Memory Program Compiler

每条frame在真实image/language/Action context中由rank-matched memory queries读取18层状态；video内以
`forward-minus-reverse`形成反转反对称Program，K轴再做保符号的置换不变集合聚合；language/Core只作Query，视频
过程是唯一additive Value；最终进入V6的320个layer/rank policy slots，并由共同训练的native rank16 factor heads
生成A/B。反对称约束止于Program/slot；输出端不硬编码negative LoRA，避免用结构人为制造control margin。

该分支同时保留GOMQ“learned memory有真实价值”和V6“rank16 policy topology有效”两条正证据，避免GOMQ flat
direct-B与LPCP frozen-tail两个已测极端。它是当前推荐架构；完整数据流与实验合同单独写入
`layer_matched_memory_program_compiler_design.md`。

## 11. 训练方式的结论

最终方法必须可fresh训练；开发阶段允许sealed LPCP作为step0 anchor隔离接口，但不能把warm-start结果冒充最终
recipe。第一轮不加入expert reconstruction、reward、LoRA negative margin、LoRA norm/rank、subset consistency或
gradient solver。

但只用correct functional loss仍没有解决owner指出的positive-only不可识别性。因此统一recipe保留两个职责分开的
信号：

1. 固定train24、每macro task-complete等权；
2. K1--K4真实均衡曝光，每video独立保序；
3. video与action query同task跨episode；
4. frozen source policy的dense functional loss只优化correct LoRA的policy方向；
5. language—directed-Program multi-positive matching区分same-task correct、cross-task wrong和重算后的shuffle/reverse，
   但不读取negative action/reward，也不规定negative LoRA必须失败；
6. single-checkpoint strict paired400及时裁决。

same-task跨video一致性、between-task separability、Program→BA transmission和correct/reverse差异先作stage diagnostics，
不再用漂亮surrogate代替closed-loop。

## 12. 设计问题现在如何闭合

1. **高层知识**：跨episode functional credit阻断轨迹复制；反对称动态Program消去static/order-even旁路；同task多video
   是matching positives，K-set只聚合Program而非低层帧。
2. **正确顺序**：`P(V)=0.5(F(V)-F(reverse(V)))`使reverse严格反号；shuffle必须重算相邻变化和causal state，不再只是
   一个希望模型自行使用的输入维度。
3. **language边界**：language/Core只进入Query、gate和route；factor slot的additive Value只能来自视频动态，缺少任一
   输入均为identity；language—Program matching阻断generic video-presence gate。
4. **多video**：每video独立形成反对称Program，K轴以mean、dispersion和odd centered correction置换不变聚合，不
   平均frames/raw features/LoRA，也不挑video。
5. **有效LoRA**：320个layer/rank slots和八个共同训练的native heads生成一套rank16 A/B，不经过冻结W2或flat
   Direct-B第二bank；Program符号必须能传到factor/BA/action，但不强迫reverse LoRA等于correct的负数。
6. **多task共存**：task-complete joint训练；cross-task matching先保持Program可分，condition-specific slot address再
   分流到共享family heads，不让一个global B grid承担全部task。
7. **历史继承**：保留V6 Core、native context、layer/rank topology、factor ownership和task-complete recipe；保留GOMQ learned
   memory；删除旧generic Procedure、rank32 residual bank和首轮reward。
8. **证伪**：先查Program→slot→factor→BA/action是否再次衰减；有信息量训练节点立即strict400，首次约145且retention
   合理即补六臂，并继续相邻checkpoint判断稳定性。

这是一份已完成证据裁决的认知文档，不直接授权GPU。对应最终提案见
`layer_matched_memory_program_compiler_design.md`；是否进入实现属于下一决策。

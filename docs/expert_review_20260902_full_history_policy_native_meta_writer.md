完整逐行证据、历史 blob 定位与 Git-wide 清单分别放在以下文件中。正文中的主判断与下面的决策摘要一致；详细稿保留了 `commit:path:line`、formal evidence root 和“原始 artifact 未找到”的标记。

* [完整 EMBER 全历史架构复核报告](sandbox:/mnt/data/EMBER_full_history_architecture_review_8f2e859_verified.md)
* [复核执行摘要](sandbox:/mnt/data/EMBER_review_executive_brief_8f2e859.md)
* [Git-wide 机械证据索引：提交、历史设计文档、逐行 authority 与关键字符串命中](sandbox:/mnt/data/EMBER_evidence_inventory_8f2e859.md)
* [压缩版机械证据索引](sandbox:/mnt/data/EMBER_evidence_inventory_8f2e859.md.gz)

下面给出架构决策层面的完整结论。

# 1. 核心结论

## 1.1 EMBER没有证明“video-to-LoRA不可行”，而是逐步证明了三个问题不能再被混为一谈

EMBER近两个月的历史不是一条架构连续调参史，而是三个相互耦合的研究问题：

1. **视频中是否存在可恢复、对策略有用、并具有正确时序含义的信息。**
2. **这些信息能否跨任务映射为策略功能，而不是只在 task-local 条件下成立。**
3. **功能意图能否稳定地落到完整 38-target rank16 LoRA 的正确方向、容量和尺度上。**

早期路线经常同时改变视频表示、credit 和 LoRA 几何，因此高分和失败都存在混杂。ECP最大的价值不是某个单独版本的分数，而是把三个问题拆开，并通过 Stage0、G1、G2、G3逐级建立了更可靠的局部证据。

## 1.2 最终科学目标仍然合理，而且不能因为G3失败而下调

固定目标——稳定 validation8 strict paired correct 大于 145/400，同时满足相邻 checkpoint 稳定、低 churn、四 suite breadth、Goal/Long、same-task 多视频鲁棒和完整因果 controls——是合理的。

原因不是“历史最好结果距离145很近”，而是三类互补证据：

* frozen source 的 48/400 说明任务确实需要显著适配；
* task-local rank16 oracle 的 250/400 说明**同一 frozen π0.5、同样38 targets和rank16容量下，策略具有远高于145的可达行为上界**；
* v6 的143和GOMQ rank32的151说明 shared Writer 并非完全无法从视频产生行为增益。

但是，v6的单点143不稳定，GOMQ的151使用rank32且缺少完整因果 controls，因此两者都不是“差一点正式通过”。它们证明的是存在共享生成信号，不证明已得到稳定、正确的视频条件映射。

## 1.3 post-a185的PNBTT裁决，失败点应被限定在G3

PNBTT之后最重要的判断是：

> 当前证据否定的是“由冻结的 Natural Program 和 native bank，经当前这组 compiler／transport／factor materialization 接口，可以建立一个跨任务共享、同时具备 correct capacity 与 wrong-video specificity 的映射”。

它没有否定：

* Stage0的信息墙；
* G1的policy-native factor证据；
* G2对有序视频事件和Natural Program的证据；
* native X/Y；
* signed pooling；
* rank4局部机制；
* task expert；
* bank本身；
* 整个ECP研究方向。

Program-through-bank、bank-conditioned primal和PNBTT构成了一条很清楚的反证链：

* 强制功能完全通过bank，specificity容易控制，但correct容量不足；
* 恢复更直接的primal路径，correct容量增加，同时wrong/common方向泄漏；
* gate-aligned PNBTT把wrong和margin压下去以后，correct和held仍然不足；
* full-rank16在task1与task93上呈现稳定相反的capacity–specificity变化，说明这不是一个统一scale或rank的小调参问题。

因此，当前G3 non-pass是**结构性non-pass**，但结构范围是“冻结Program到bank的当前共享编译链”，不能外推成ECP所有前置机制失败。

## 1.4 当前ECP确实已经过度串联化

ECP中的很多单步变换都有合理动机，但“每一步各自合理”不能推出“整条链合理”。当前主要风险不是数学操作太多本身，而是：

* 每一层都建立新的坐标或语义接口；
* 后一层只能看到前一层压缩后的结果；
* 某一层丢失的信息无法由后续训练恢复；
* 梯度必须跨越多个固定或弱可学习变换；
* bank公共方向、canonicalization和pooling可能反复强化任务共有成分；
* 输出support和任务specificity被要求在冻结链的末端同时解决。

历史上应保留的归纳偏置可以压缩成四个统一操作：

1. **相对顺序感知的policy-native视频编码；**
2. **language、video和target query之间的注意力交互；**
3. **对policy-native候选方向的signed选择、组合和残差化；**
4. **一个确定的rank16 LoRA边界materializer。**

这四种操作足以承载G1、G2、native bank、signed evidence、task experts和完整target结构，不需要十几种连续的专用变换。

## 1.5 下一步不应继续做PNBTT-R14，也不应回到无约束direct hypernetwork

主建议是建立一个：

> **统一的 policy-native evidence-to-LoRA Meta-Writer：用重复的标准block共同更新视频事件token、bank token和38×16个target-rank query；Natural Program退化为可探测的分布式latent，而不再是冻结的显式中间合同；native bank作为soft attention memory和输出先验；最终由target-rank query生成bank-anchored、带小型有界残差的rank16 LoRA。**

这不是“删除数学后换普通Transformer”。它保留了历史上真正有证据的结构边界，同时取消没有独立必要性证据的串行接口。

唯一后备分支应由一个明确实验触发：

> 只有当bank-anchored输出在task1和task93的task-local条件下都明显无法逼近已有rank16 oracle，而相同Writer加一个有界free residual后能够恢复时，才开放更大的free residual空间。

shared held失败不应触发任意A/B扩容；那更可能是credit、identifiability或shortcut问题。

---

# 2. 完整EMBER架构谱系

以下按“表示、credit、LoRA生成几何”的实质变化整理。逐提交版本、首次出现位置、最后可达历史blob和相应结果根见完整报告与机械索引。

| 历史阶段                                                | 实质变化                                     | 得到的正证据                                                                      | 主要失败或混杂                                                         | 后续应保留／避免                                          |
| --------------------------------------------------- | ---------------------------------------- | --------------------------------------------------------------------------- | --------------------------------------------------------------- | ------------------------------------------------- |
| 最早SmolVLA sparse Writer                             | 从有限视频特征直接生成稀疏策略适配量                       | 证明视频到参数更新的训练链能接通                                                            | 视频覆盖有限；策略、表示和输出几何同时受限                                           | 保留端到端可微链；避免把稀疏局部结果当完整LoRA证据                       |
| 全视频层级Writer                                         | 用层级视频聚合替代少量帧或局部特征                        | 更充分读取完整视频                                                                   | “读取更多视频”没有自动转化为正确时序使用；容易被静态场景支配                                 | 保留多尺度事件压缩；必须配套reversed/static controls            |
| π0.5 AS-Writer v1/v2                                | 切换到目标source policy与更完整adapter生成          | 证明π0.5可通过生成LoRA改善                                                           | direct hypernetwork容易依赖language/task identity；输出factor gauge不稳定 | 保留π0.5原生目标；避免无约束dense A/B head                    |
| reward Writer与task-local适配                          | 引入行为reward或单任务优化                         | task-local能力上界显著，最终形成rank16 oracle证据                                        | task-local拟合不等于shared generalization；reward可能掩盖错误视频使用           | task expert作为teacher和上界，不作为shared能力结论             |
| Action-Memory、temporal-RoPE                         | 显式表示视频内部顺序与历史状态                          | 顺序信息有必要，absolute frame pooling不足                                            | absolute time可能绑定录制速度；长序列聚合仍弱                                   | 保留相对事件顺序，不绑定绝对时间                                  |
| Action-Forecast、Belief、Visual-State                 | 从静态外观转向状态变化、隐变量和未来变化                     | 强化“视频应表达过程而非场景”的认识                                                          | 内部预测loss可能与策略闭环功能脱节                                             | 可作辅助表征，不应成为正式Gate                                 |
| v5、v5.1、v5.2、v5.3                                   | direct Writer逐步结构化，增加target/layer/rank条件 | shared Writer出现明显闭环增益                                                       | recipe、表示和几何同时变化；峰值与邻近checkpoint不一致                             | 保留target-conditioned共享参数；避免用单点峰值选架构               |
| v6                                                  | 当时最强rank16附近行为能力，峰值143                   | shared视频Writer可逼近正式分数区间                                                     | 不稳定、churn高、controls不足，不能称正式近通过                                  | 用作行为上界和回归基线，不恢复整套旧结构                              |
| v7、v8、v10                                           | 尝试解决v6稳定性、泛化或输出结构问题                      | 暴露支持集、公共方向和训练credit问题                                                       | 修补式分支增加，未形成可复制主干                                                | 不应逐版本复活专用分支                                       |
| Loom、Recenter、Core-Program                          | 分离共享prior与任务innovation，建立显式program       | 正确指出公共方向会压制任务差异                                                             | 硬recenter/program可能过早压缩任务证据                                     | 保留prior–innovation思想；由学习block实现                   |
| Prior-Innovation、Target-Spectral                    | 对公共先验、target频谱和输出坐标进一步分解                 | 输出并非所有targets共享同一方向                                                         | 多次谱变换和canonicalization引入新接口                                     | 只保留一次边界规范化或固定native字典                             |
| SPG、UCP、AP-ADR、CV-ADR、Target-Bound                  | 用几何约束控制方向、目标归属、尺度和跨视频一致性                 | 明确scale、direction、target ownership是不同问题                                     | proxy几何正确不保证闭环；硬约束可能压低correct容量                                 | 约束降级为正则和诊断，不作为主体网络                                |
| factor basis、direction store、target ownership       | 从直接A/B转向factor或方向字典                      | policy-native候选空间比任意输出更稳定                                                   | 固定support可能覆盖不足；ownership错误会跨target泄漏                           | 保留native dictionary和target索引；允许学习残差               |
| atom、lane、kernel路线                                  | 把LoRA拆成可复用原子、通道或kernel                   | 改善参数共享和输出规模                                                                 | 手工lane/atom语义可能不对应真实功能                                          | 把它们统一为bank token，由attention学习组合                   |
| K4、layer trace                                      | 多视频与策略层内证据进入Writer                       | K1/K4和多视频机制得到局部支持                                                           | 层trace不自动等于功能credit                                             | 保留policy-native taps和Dynamic-K                    |
| energy/evidence factorization、video expert          | 区分证据强度、方向和视频专家                           | signed evidence和视频specificity更明确                                            | 固定energy分解可能重复normalization                                     | 用signed gate统一表达                                  |
| reward credit、task experts                          | 用真实行为或task-local expert改善credit          | task experts提供了最强task-local功能teacher                                        | reward噪声和expert manifold覆盖限制shared学习                            | task experts作训练期teacher，不进入推理                     |
| expert manifold、tangent、guard                       | 限制Writer输出落在已知可用邻域并防止破坏策略                | 说明随机LoRA方向风险很大                                                              | guard可能只抑制wrong，同时抑制correct                                     | 保留native残差先验；guard变为软边界                           |
| Dynamic-K、V6 bridges                                | 支持不同数量视频，并尝试连接v6能力与新结构                   | 多视频不是必须固定K                                                                  | bridge兼容旧坐标，容易继承旧包袱                                             | 使用mask/recurrent set aggregation，不保留版本桥           |
| LPCP                                                | language、policy evidence与program更紧密结合    | 为后续Program路线提供基础                                                            | 仍存在显式中间合同和shared mapping问题                                      | language-conditioning保留，显式compiler不保留             |
| post-LPCP native value、occupancy、commitment         | 强调真实策略native value及support使用             | occupancy揭示collapse和公共方向                                                    | 高occupancy不代表正确行为；commitment可能变成内部目标                            | occupancy仅作监控和轻正则                                 |
| memory-grid、GOMQ                                    | 用更大native support和组合机制提升容量               | rank32达到151，证明native/support扩容有行为价值                                         | 后续回落、controls不足；统一rank16约136                                    | bank作为先验有价值；不能靠rank32掩盖方向错误                       |
| LMMPC v1–v5                                         | 显式program/controller、phase和occupancy建模   | 进一步确认任务过程具有阶段结构                                                             | 固定phase/program decoder成为新瓶颈                                    | phase可由latent attention学习                         |
| functional decoder、shared-prior                     | 尝试把内部program直接解码成策略功能                    | 正确聚焦functional mapping                                                      | decoder proxy与闭环功能仍不一致                                          | 训练credit必须穿过frozen policy或可靠teacher               |
| ECP Stage0                                          | 固定information wall和正式任务定义                | 把action-hidden、same-task、ordered video变成可审查合同                               | 不直接证明生成能力                                                       | 应保留                                               |
| ECP Stage1 v1–v24、MDCO、PECS、privileged realization  | 分解表示、事件、program和功能监督                     | 建立大量组件级机制证据                                                                 | 版本间专用接口逐渐累积                                                     | privileged只作训练信号；合并重复算子                           |
| G1 Native-Factor                                    | 验证policy-native factor在held任务上的功能性       | strict250为114/250，breadth 5/5，Goal2、Long1                                   | 不证明video→factor shared映射                                        | native bank与factor basis应保留                       |
| G2 Natural Program                                  | 验证视频能形成有序、same-task稳定的program            | endpoint held改善22.2047%，probe38/40，median active events 4，same-task、K1、K4通过 | 不证明program可被统一编译为LoRA                                           | event/program encoder可初始化新Writer；Program不应冻结为唯一瓶颈 |
| G3 F1/F2/F3、sketch、Program-primal、G2-B、J2/J3、R1–R13 | 系统测试冻结Program到bank和LoRA的共享编译             | 逐步排除了简单scale、margin和wrong泄漏解释                                               | 长期没有建立泛化Program–bank功能映射                                        | 终止继续堆compiler补丁                                   |
| Program–bank interaction、EBSRI                      | 强化Program与native evidence的交互             | specific方向可被识别                                                              | 交互仍受冻结坐标和公共bank方向限制                                             | 交互应在统一block内联合学习                                  |
| Program-through-bank                                | 所有功能必须经bank                              | wrong specificity改善                                                         | correct capacity不足                                              | bank不能成为唯一硬通路                                     |
| bank-conditioned primal                             | 恢复direct/primal功能路径                      | correct容量恢复                                                                 | wrong/common方向同步上升                                              | 需要learned residual，不是无条件primal bypass             |
| PNBTT                                               | 用gate-aligned目标统一capacity和specificity    | wrong、margin等约束可压好                                                          | correct、held仍显著不足；task1/93呈相反权衡                                 | 当前冻结compiler路线正式non-pass                          |

---

# 3. 跨历史反复出现的成功机制与失败模式

## 3.1 反复得到支持的机制

### Policy-native坐标优于完全自由的参数生成

G1、post-LPCP、GOMQ以及后续bank路线共同说明：从π0.5内部真实出现的方向、factor、value或expert更新中构造候选空间，比让Writer从零输出高维A/B更容易获得有效闭环行为。

真正应保留的是“输出锚定在策略已知可用方向附近”，而不是某一种固定basis、lane或transport公式。

### 视频内部顺序必须保留，但不应绑定absolute time

Action-Memory、temporal-RoPE、Forecast、G2 event和same-task/K1/K4结果共同支持：

* 帧集合不足；
* 静态场景不是主要答案；
* 视频之间可以作为集合处理；
* 单个视频内部必须保留事件先后；
* 顺序应按相对位置、事件邻接或状态变化表达，而不是绝对秒数。

### Signed evidence与task-specific innovation是必要的

多个recenter、prior–innovation、energy/evidence和PNBTT结果均表明，只生成“某个方向应被加强”的非负权重不够。Writer必须能够：

* 增强某些native方向；
* 抵消共享prior；
* 对不同target给出不同符号；
* 在正确视频和错误视频之间形成方向性差异。

### target/rank条件化比整套LoRA一次性平坦生成更可靠

38 targets并不共享完全相同的功能角色。历史上的target ownership、Target-Spectral、factor、lane和G1证据都支持共享主干加target/rank query，而不支持一个不区分target的全局summary直接广播到全部A/B。

### task expert有价值，但其价值是credit和初始化

task-local rank16 oracle与task expert路线证明了有效更新存在，也能提供行为teacher。它们没有证明shared Writer已经知道如何从视频选择相应更新。

因此task expert最合理的角色是：

* 构造native bank；
* 提供task-local行为KL或action teacher；
* 为Writer输出建立正样本；
* 提供representability上界。

它不应成为推理期模块，也不应通过task ID泄漏任务答案。

## 3.2 反复失败的模式

### language/task identity旁路

exact language是最终输入的一部分，不能删除。但当训练集中一种language几乎总对应一个task和一种LoRA时，Writer无需使用视频也能降低loss。

仅加入video token、提高video attention或让video branch非零，不能证明正确使用视频。需要同language下的correct/wrong/shuffled/reversed/no-video对照进入训练分布或至少进入高频闭环验证。

### proxy指标与行为脱节

历史中反复出现：

* reconstruction更好；
* cosine更高；
* occupancy更高；
* factor或Program匹配更好；
* wrong margin更漂亮；

但真实correct closed-loop没有同步改善。

这些指标可以诊断具体接口，却不应阻塞早期真实rollout，也不能替代最终裁决。

### 单checkpoint峰值与support churn

v6和GOMQ说明偶然峰值能超过或接近分数门槛，但相邻checkpoint不稳定、正确任务集合不断变化。其常见原因不是简单过拟合，而是多个近等价参数方向在训练中旋转，导致LoRA行为离散跳变。

因此新Writer必须减少factor gauge、使用native字典或规范化边界，并把相邻checkpoint闭环轨迹作为一等指标。

### rank扩容同时放大正确容量和错误方向

GOMQ rank32与PNBTT task1/task93共同提示：rank不足不是唯一瓶颈。增加rank或开放primal路径会扩大可表达空间，但如果shared mapping没有正确识别任务specificity，wrong视频同样受益。

所以不能看到correct不足就直接增加rank、width或free A/B。

### 显式中间接口形成“局部都通过、串联不工作”

G1证明factor有用，G2证明Program有信息，并不推出冻结的Program→factor函数容易学习。ECP当前最重要的教训正是：

> 两个分别有效的表示之间，仍可能不存在由当前数据、credit和函数类可学习的共享映射。

继续增加canonicalization、transport或projection只能改变函数类，不能自动解决可辨识性。

---

# 4. 当前ECP逐层必要性和复杂度审计

## 4.1 当前流水线

当前概念上的完整链可概括为：

```text
exact task language
+ ordered, action-hidden teacher videos
        │
        ▼
policy / visual feature capture
        │
        ▼
temporal event extraction
        │
        ▼
language-conditioned evidence
        │
        ▼
signed evidence / expert or phase organization
        │
        ▼
canonicalization / pooling / program formation
        │
        ▼
Natural Program                       ← Pass A
        │
        ▼
native X/Y capture and factor bank
        │
        ▼
bank summary / occupancy / commitment
        │
        ▼
Program–bank interaction
        │
        ▼
pooling / canonicalization / transport
        │
        ▼
factor or primal materialization
        │
        ▼
per-target, per-rank normalization and scale
        │
        ▼
complete 38-target rank16 LoRA          ← Pass B
        │
        ▼
frozen π0.5 closed-loop behavior
```

## 4.2 必要性分类

| 变换                                   | 判断                  | 原因与处置                                         |
| ------------------------------------ | ------------------- | --------------------------------------------- |
| action-hidden information wall       | **直接支持，应保留**        | 属于最终科学问题本身，避免退化为动作复制                          |
| exact task language输入                | **直接支持，应保留**        | 最终目标要求，但必须用paired视频阻断language-only解           |
| policy-native video capture          | **直接支持，应保留**        | G1、layer trace、native value与bank路线共同支持        |
| 视频内相对顺序                              | **直接支持，应保留**        | G2、temporal路线和reversed需求支持                    |
| 多视频Dynamic-K                         | **已有支持，应保留**        | K1/K4和same-task结果支持；实现不应绑定固定K                 |
| event化／局部时间压缩                        | **有支持，但不必固定现有形式**   | event比全局frame pooling合理；应由重复block学习           |
| signed evidence                      | **直接支持，应保留**        | prior抵消、wrong specificity和task innovation需要符号 |
| task/target条件化                       | **直接支持，应保留**        | target ownership和native factor结果支持            |
| 显式Natural Program监督                  | **可保留为辅助和probe**    | G2证明其有诊断价值；不应成为唯一信息通道                         |
| 固定维度、冻结语义的Program瓶颈                  | **缺少必要性证据，建议取消**    | G3表明其与bank之间的共享功能映射未成立                        |
| Pass A与Pass B严格冻结分离                  | **历史实验合同，而非最终必要机制** | 有助定位失败，但阻止表示和输出几何共同适配                         |
| native bank                          | **直接支持，应保留**        | 提供policy-native候选方向和稳定输出先验                    |
| bank作为唯一输出support                    | **已有反证，避免**         | Program-through-bank表现为specific但容量不足          |
| 无约束primal bypass                     | **已有反证，避免**         | bank-conditioned primal恢复correct同时增加wrong     |
| 多层bank summary                       | **动机合理、独立证据不足**     | 可由target query对bank的attention替代               |
| 重复canonicalization                   | **功能重复、可能损失信息**     | 多次改变坐标；只需一次factor gauge边界                     |
| 多级pooling                            | **可能损伤specificity** | 公共视频／task方向容易在平均中占优                           |
| 固定transport                          | **G3主要失败候选**        | 把表示学习与功能映射割裂，梯度无法修正上游语义                       |
| factor materialization               | **最终需要，但应只保留一次**    | rank16 A/B必须生成；无需成为多阶段主体网络                    |
| occupancy                            | **诊断或轻正则**          | 能发现collapse，不是功能目标                            |
| commitment／guard                     | **软边界条件**           | 可防止破坏source policy；过强会压低correct容量             |
| wrong/margin约束                       | **训练信号应保留**         | PNBTT证明可控制specificity；它只是必要非充分条件              |
| exact rank16、38 targets、scale/norm边界 | **必须保留**            | 属于最终接口和稳定性条件                                  |

## 4.3 对owner七个疑问的直接回答

1. **十几次连续变换不是不可避免。**
   它们主要是历史诊断接口的累积，不是由最终问题数学上强制产生。

2. **有效归纳偏置可以压缩成四种算子。**
   相对顺序attention、policy-native bank attention、signed residual mixing、LoRA boundary materialization。

3. **Pass A与Pass B不应继续严格冻结分开。**
   研究裁决时分开有价值；最终Writer中，应允许target-rank query反向影响事件表示，使“什么视频信息有用”由功能credit共同决定。

4. **Program不必是固定显式瓶颈。**
   它可以成为一组latent event/program token，并保留G2 probe头。这样仍能检查时序和same-task性质，但不会强迫全部功能经过单一冻结向量。

5. **native bank最合适的角色是attention memory加输出先验。**
   它既不是唯一中间监督，也不是唯一support；其职责是提供策略已知可用方向、降低搜索空间和factor churn。

6. **Writer不应从第一天直接生成任意A/B。**
   主路径应在policy-native候选空间中选择、组合和残差化。只有task-local覆盖实验明确证明bank空间不足，才开放有界free residual。

7. **固定数学约束只应留在边界。**
   包括rank16、target映射、factor gauge、RMS/norm、LoRA scale和精确materialization。Program语义、bank interaction和target ownership不应继续由层层硬公式承担。

---

# 5. 最终目标及内部Gate合理性判断

## 5.1 最终目标

最终目标应保持不变。145不是由oracle上界决定的极限，而是一个要求shared Writer取得明显高于source、接近历史能力峰值、同时具备稳定性和因果性的最低标准。

正式PASS必须是合取条件：

* strict paired correct >145/400；
* 预先指定的相邻single checkpoints均保持能力，而非checkpoint cherry-pick；
* churn低且正确任务集合不是不断替换；
* Spatial、Object、Goal、Long均非零；
* Goal与Long不是各一例的偶然点；
* 同任务不同正确视频保持；
* correct显著优于wrong、shuffled、reversed和no-video；
* language-only与static-scene不能解释结果。

仅跨过145而controls失败，仍然是non-pass。

## 5.2 内部Gate

| Gate类型                             | 判断                                            |
| ---------------------------------- | --------------------------------------------- |
| Stage0 information-wall Gate       | 合理，是问题定义和泄漏检查                                 |
| G1 native-factor Gate              | 合理，回答“候选方向是否具有功能性”                            |
| G2 Natural Program Gate            | 合理，回答“action-hidden ordered video能否产生可复现事件表示” |
| G3 frozen compiler Gate            | 对“当前冻结compiler是否成立”合理                         |
| 用G3 non-pass否定G1/G2                | 不合理，超出Gate裁决范围                                |
| 用Program reconstruction/cosine替代闭环 | 不合理                                           |
| wrong/margin必须合格                   | 合理但仅是必要条件                                     |
| 要求冻结Program→bank映射先完美，再允许联合训练      | 对下一代架构过早                                      |
| v6的143视为“只差2分”                     | 不合理；稳定性和controls差距远大于2分                       |
| GOMQ的151视为正式完成                     | 不合理；rank与controls合同不同                         |
| 因多次失败而降低145                        | 没有证据支持                                        |

G3最值得调整的不是数值门槛，而是Gate所绑定的科学问题。它已经充分回答了“当前冻结编译链是否值得继续”。下一步不应再要求新架构先通过同一个冻结compiler问题；新架构要尽早接受真实functional credit。

---

# 6. 推荐的可扩展整体架构

## 6.1 总体结构：统一 Policy-Native Meta-Writer

推理时只有一个Writer，输入仍严格是：

```text
exact language
+ K条same-task、action-hidden、各自内部有序的视频
```

输出仍是：

```text
唯一一套完整 38-target rank16 LoRA
```

建议的数据流为：

```text
language tokens ─────────────────────────┐
                                         │
video frames → frozen π0.5 native taps   │
              → relative event tokens    │
                                         ▼
native factor bank ───────────────→ repeated unified meta-writer blocks
                                         │
38 × 16 target-rank queries ─────────────┘
                                         │
                                         ▼
signed bank coefficients
+ bounded learned residual coefficients
                                         │
                                         ▼
single canonical LoRA materializer
                                         │
                                         ▼
38-target rank16 LoRA
                                         │
                                         ▼
frozen π0.5 rollout
```

## 6.2 四种token

### Language token

保留exact task language，但不增加task ID、文件名、数据集名或其它隐式标签。

### Event token

从冻结π0.5的policy-native视觉／中间层特征生成。每个视频先独立编码：

* 使用相对帧距离、事件邻接和视频内顺序；
* 不使用绝对秒数作为主要位置；
* 视频之间不强加顺序；
* K条视频通过mask和共享聚合处理。

### Bank token

每个bank atom包含：

* 所属target或可作用target；
* policy-native factor／direction；
* 来源层和尺度；
* 由task expert或G1得到的功能metadata；
* learned key，但value保持native几何。

bank不先被压成一个summary。每个target-rank query直接attention到相关bank atoms。

### Target-rank query

共38×16个query。它们共享同一套block参数，通过target embedding和rank-slot embedding区分职责。这样：

* 不需要38套专用Writer；
* 不会把同一全局向量广播给所有target；
* 增加depth只需复制block；
* 增加bank容量只需增加token；
* rank固定为16，不改变最终合同。

## 6.3 标准重复block

每层只包含常见、可复制的操作：

1. event/language self-attention或局部时序attention；
2. target-rank query对language、event和bank的cross-attention；
3. target-rank query之间的self-attention，用于跨layer/target协调；
4. gated MLP；
5. pre-norm和residual。

不再为新能力增加新的Program类型、transport公式或专用branch。扩大模型能力主要通过：

* 增加block层数；
* 增加hidden width；
* 增加attention heads；
* 增加native bank atoms；
* 增加训练meta-task和same-task视频多样性。

## 6.4 输出几何

主输出不是任意dense A/B，而是每个target-rank slot产生：

* 对target-specific native basis的signed系数；
* 一个标量gain；
* 一个初始很小、有明确norm cap的residual系数。

可以把每个rank slot写成：

$$
a_{t,r}=D^A_t c^A_{t,r}+R^A_t z^A_{t,r},
\qquad
b_{t,r}=D^B_t c^B_{t,r}+R^B_t z^B_{t,r},
$$

其中：

* \(D^A_t,D^B_t\) 是由G1、task experts和历史有效LoRA构造的policy-native字典；
* \(R^A_t,R^B_t\) 是小型共享残差basis；
* \(c,z\) 由统一Writer产生；
* 最终只做一次norm/gauge规范化和LoRA materialization。

这比直接输出任意A/B更稳定，也比Program-through-bank更有容量。

## 6.5 Natural Program的角色

G2 encoder可以用于初始化event block。G2的probe仍保留，用来回答：

* 是否识别active events；
* 是否保留顺序；
* K1/K4是否一致；
* same-task不同视频是否形成兼容表示。

但Program不再是冻结的、唯一的Pass A输出。功能梯度可以继续更新event token，使其只保留真正有助于策略行为的信息。

## 6.6 最小充分loss

主训练loss只保留三类：

1. **task-expert functional distillation**
   在匹配状态分布上，对生成LoRA后的frozen π0.5动作分布与task expert做KL或action loss。

2. **paired causal specificity loss**
   同language下，correct video必须优于wrong、shuffled、reversed、no-video。优先在策略行为或经过持续校准的functional evaluator上比较，而不是仅比较参数cosine。

3. **轻量边界正则**
   LoRA norm、rank-slot collapse、bank occupancy和residual比例。它们只防止退化，不定义成功。

G2 program loss、factor reconstruction等可以在初始化期低权重使用，随后衰减，不能长期与functional目标竞争。

## 6.7 task expert、functional evaluator、occupancy和reward

| 组件                   | 推荐角色                                                                        |
| -------------------- | --------------------------------------------------------------------------- |
| task expert          | 构造bank、提供行为teacher、定义task-local上界；推理期不存在                                    |
| functional evaluator | 对昂贵rollout做排序、采样和低方差辅助；必须用真实rollout持续校准                                     |
| occupancy            | 检测bank collapse、公共方向垄断和dead ranks；仅诊断或轻正则                                   |
| reward               | 在shared supervised mapping已稳定后做Action Meta/RL精修；不能用于拯救未建立的video specificity |

## 6.8 component initialization与fresh random比较

必须比较两个预注册arm：

* **Evidence-init arm**：加载G1 bank、G2 event encoder和已有task expert字典；
* **Fresh-joint arm**：相同结构、数据、batch、优化步数和policy，仅Writer可训练部分随机初始化。

比较完整学习曲线、correct/wrong separation、邻近checkpoint稳定性和最终闭环，不只比较终点。

若evidence-init显著加快收敛但终点相同，历史组件主要是优化帮助；若终点也更高，说明其归纳偏置仍有不可替代价值。

## 6.9 Action Meta何时打开

Action Meta不应在当前G3 non-pass上直接打开。它至少要等到新Writer满足：

* 多个shared held任务上correct稳定高于source；
* wrong、shuffled、reversed和no-video没有同步上升；
* task1和task93不再表现为稳定相反的权衡；
* 两个或以上相邻checkpoint保持相似breadth；
* same-task多视频与K1/K4成立。

届时Action Meta可作为提升闭环上限的精修，而不是寻找基本映射的工具。

## 6.10 1–6张A40扩展

架构不依赖固定GPU数：

* 冻结π0.5的视频native taps可预计算和缓存；
* language embedding与bank K/V可缓存；
* variable-K视频使用packed sequence和mask；
* 38×16 query天然适合批处理；
* 训练样本按task/video episode做数据并行；
* frozen policy可按可用GPU数使用FSDP或tensor shard；
* Writer保持小型复制，功能forward按episode做gradient accumulation；
* 1卡使用microbatch与activation checkpoint；
* 2–6卡增加policy shard或task data-parallel group，不改变数学语义。

最需要实现优化的是“每个样本一套LoRA”的batched grouped-LoRA kernel，而不是再重构表示链。

---

# 7. 新架构与全部相关历史路线的防重复对照

| 旧家族                               | 旧表示                                | 旧credit                  | 旧LoRA几何                     | 新架构的实质变化                                                   |
| --------------------------------- | ---------------------------------- | ------------------------ | --------------------------- | ---------------------------------------------------------- |
| sparse／hierarchical Writer        | raw或层级视频summary                    | 直接参数／弱行为loss             | 稀疏或平坦输出                     | policy-native event token；target-rank query；paired功能credit |
| AS-Writer v1/v2                   | language+video全局表示                 | shared或task-local        | 较自由A/B                      | native字典锚定加有界残差                                            |
| Action-Memory／temporal-RoPE       | 显式时序状态                             | 内部时序loss                 | 未改变主体几何                     | 相对顺序成为统一block的一部分，并直接接收功能梯度                                |
| Forecast／Belief／Visual-State      | 预测隐状态                              | reconstruction／forecast  | 间接参数生成                      | 预测loss降为可选辅助，不再是主任务                                        |
| v5–v10                            | 多专用branch和summary                  | shared行为训练               | 逐版结构化                       | 一个重复block替代版本专用branch                                      |
| Loom／Recenter／Prior-Innovation    | 显式prior与innovation分解               | 参数或内部目标                  | 手工recenter                  | signed residual attention学习同一作用                            |
| Target-Spectral／SPG／UCP／ADR       | 多重固定几何约束                           | cosine、alignment、bound   | 特定投影空间                      | 只保留一次输出边界；其它转为软正则                                          |
| factor／direction／atom／lane／kernel | 固定原子及ownership                     | factor监督                 | 字典或分槽输出                     | 原子统一为bank token，组合由attention学习                             |
| K4／layer trace／video expert       | 多视频和策略层证据                          | expert或video loss        | 多分支                         | policy-native event token原生支持Dynamic-K                     |
| expert manifold／tangent／guard     | 在expert邻域内更新                       | expert距离／防破坏             | tangent限制                   | native bank是先验，残差提供可检验的逃逸空间                                |
| LPCP／V6 bridge                    | program与旧Writer桥接                  | 混合credit                 | 兼容旧坐标                       | 不保留版本桥，event和output联合训练                                    |
| native value／occupancy／commitment | bank使用统计                           | occupancy等proxy          | native support              | bank保留，occupancy不再主导                                       |
| GOMQ                              | 更大组合support                        | 闭环加内部目标                  | rank32能力较强                  | 固定正式rank16；用残差测试coverage，不靠rank扩容                          |
| LMMPC v1–v5                       | 显式phase/program/controller         | decoder或functional proxy | controller到LoRA             | phase/program变成分布式latent                                   |
| ECP Stage0/G1                     | information wall与native factor     | 分级Gate                   | native factor               | 原样保留为输入合同与bank初始化                                          |
| G2 Natural Program                | 冻结显式Program                        | program probe            | 尚未直接生成完整LoRA                | Program改为可probe latent，并接收functional梯度                     |
| G3 frozen compiler                | Program经多级接口到bank/LoRA             | compiler与gate loss       | hard bank或primal路径          | 取消冻结compiler和唯一硬路径                                         |
| PNBTT                             | gate-alignedProgram-bank transport | correct/wrong/margin     | capacity-specificity受固定结构约束 | paired causal loss保留，mapping由重复block联合学习                   |

它不是任何早期架构的简单加深：

* 与direct hypernetwork相比，它不在任意参数空间搜索；
* 与G1相比，它解决shared video-conditioned mapping；
* 与G2相比，它不冻结Program语义；
* 与GOMQ相比，它遵守rank16并正式处理controls；
* 与G3/PNBTT相比，bank不再是必须经过的编译通道，而是可注意、可残差化的policy-native先验。

---

# 8. 最小决定性实现与实验序列

## 第0步：只做必要smoke

不先重构通用框架。只验证：

* 38 targets × rank16的shape与target对应正确；
* Writer梯度能经过frozen policy的functional forward返回；
* correct/wrong视频确实进入不同event token；
* grouped LoRA materialization和现有单样本实现数值一致；
* 1、2和多卡的同一小batch结果一致；
* 无task ID、文件名或action输入。

控制在数十个样本和极少量rollout内。

## 第1步：task1/task93二维判别实验

选择task1和task93，是因为PNBTT full-rank16在两者上显示相反的capacity–specificity行为，最适合快速识别结构是否真正改变。

固定：

* rank16；
* 同一task expert；
* 同一correct/wrong视频；
* 同一functional loss；
* 同一训练预算。

比较：

1. PNBTT冻结基线；
2. 新统一Writer的bank-anchored主路径。

此阶段不扫LR、width、rank和seed。只观察：

* correct是否在两个任务上都持续上升；
* wrong是否保持接近source；
* 两个相邻checkpoint是否同向；
* task-local训练集能否被明显拟合。

判定：

* 两任务correct都不升：表示或credit链未接通，停止扩规模；
* correct和wrong同步升：specificity／数据可辨识性失败，不增加rank；
* 一个任务升、另一个仍相反：输出几何仍存在task-dependent冲突；
* 两任务均形成稳定correct>wrong：进入shared实验。

## 第2步：bank coverage决定实验

只在第1步correct容量不足时，增加一个变量：开启有界free residual。

* residual恢复两个任务的correct且wrong不升：native bank覆盖不足，保留小残差；
* residual使correct/wrong一起升：不是coverage，而是任务识别失败；关闭残差；
* residual也无效：当前video evidence或functional credit不足，停止LoRA几何修补。

这是唯一允许的主架构后备分支。

## 第3步：shared held小规模实验

使用已有正式held任务与task expert数据，固定架构后训练共享Writer。每个评估点同时运行：

* correct；
* wrong；
* shuffled；
* reversed；
* no-video；
* language-only；
* 至少两条same-task正确视频；
* K1与K4。

内部Program probe、occupancy和bank attention只用于解释失败，不作为进入下一步的硬门槛。

继续条件：

* held aggregate相对source为正；
* 多数任务同向而不是少量任务拉动；
* correct与各负control形成一致差距；
* 相邻checkpoint breadth稳定；
* Goal和Long开始出现非偶然贡献。

## 第4步：component-init与fresh-joint比较

在完全相同的shared held设置中比较G1/G2初始化与fresh joint。

只改变初始化，避免把“新架构效果”与“历史组件warm start效果”混合。

根据结果决定：

* init只提升速度：后续大规模训练可以使用，但论文结论应是优化优势；
* init提升终点和controls：G1/G2提供关键归纳偏置；
* fresh更好：冻结历史组件存在表示包袱，应只保留native bank合同；
* 两者均失败：停止，不做宽度或LR小扫。

## 第5步：validation8 strict paired400

在架构、loss、数据合同和checkpoint规则全部冻结后运行。预先确定：

* 评估checkpoint间隔；
* 允许报告的相邻checkpoint窗口；
* strict paired统计单位；
* suite breadth规则；
* control差值；
* same-task多视频选择方式。

只有在至少一个checkpoint超过145且邻近checkpoint保持相似水平、breadth和controls时才算正式通过。

## 第6步：扩大meta-task与视频规模

通过validation8后再增加：

* meta-task数量；
* 每task视频多样性；
* Dynamic-K范围；
* Writer层数或宽度；
* native bank覆盖。

扩大顺序应优先是数据，其次block深度，最后才是残差support。不得因为held失败直接同时增加数据、width、rank和loss项。

---

# 9. 主要风险、停止条件和仍缺失的原始artifact

## 9.1 主要风险

### native bank覆盖不足

task-local rank16 oracle可能位于当前bank span之外。必须通过bank-only与bounded residual的task-local对照判断，不能仅看bank occupancy。

### 训练数据不可辨识

当language几乎唯一标识task，而wrong视频与correct视频差异又不充分时，任何大模型都会选择language shortcut。模型容量越大，旁路可能越严重。

### task expert状态分布偏差

在expert轨迹上做action KL可能没有覆盖生成LoRA实际访问的状态。需要周期性把Writer rollout状态加入functional训练集。

### functional evaluator被利用

evaluator只能作为辅助。其预测必须持续与真实闭环校准，并监控correct/wrong上的系统偏差。

### factor gauge与checkpoint churn

即使行为近似，A/B可以发生缩放或旋转。输出应以native字典系数和单一边界规范化减少等价解。

### residual退化为旧direct hypernetwork

free residual必须有：

* 小初始化；
* 明确norm cap；
* 独立占用统计；
* bank-only对照；
* 只有coverage证据才扩大。

## 9.2 停止条件

应停止当前主设计，而不是继续小扫的情形：

1. bank加有界residual仍无法在task1和task93上得到基本task-local correct提升；
2. 加入same-language paired负例后，correct和wrong仍在多个任务上同步上升；
3. shared训练集拟合良好，但held长期接近source且增加meta-task后不改善；
4. 相邻checkpoint持续大幅churn，而native系数、norm和functional loss均未显示对应变化；
5. 需要放松action-hidden、使用task ID或依赖文件名才能工作；
6. 需要rank32才能恢复能力，而rank16下task-local oracle仍明显可达——这意味着几何或credit仍错，不是正式目标应改。

## 9.3 仍需要的决定性原始证据

完整报告第9节给出了仓库中实际找到和未找到的artifact。无论文件是否物理缺失，下一轮最关键的统一证据表应至少包含：

* v6与GOMQ所有报告checkpoint的原始400-rollout逐任务表，而不只是best summary；
* 每个checkpoint的correct集合、churn和四suite分布；
* task-local rank16 oracle的逐任务LoRA、policy hash、video manifest和rollout结果；
* G1 bank atom／factor到task-local oracle方向的coverage投影；
* G2 same-task视频交叉矩阵，而非只比较平均Program；
* PNBTT task1与task93逐checkpoint的correct、wrong、margin、norm、bank/primal贡献；
* Program-through-bank、bank-conditioned primal和PNBTT使用的完全一致数据manifest；
* correct/wrong/shuffled/reversed/no-video的原始配对关系；
* frozen π0.5 checkpoint、38-target映射和LoRA scale的版本指纹。

缺少这些原始表时，只能得出组件级或formal-summary级结论，不能追溯某个峰值究竟来自表示、rank、seed还是评估差异。

---

# 10. 事实、推断、建议与工程细节的明确区分

## 仓库事实

以下是本次架构判断依赖的核心事实类型，逐行来源见完整报告：

* source validation8为48/400；
* task-local rank16 oracle为250/400；
* v6出现143峰值但不稳定；
* GOMQ rank32出现151，后续回落且controls不足；统一rank16约136；
* G1 held5 strict250为114/250，breadth 5/5、Goal2、Long1；
* G2 endpoint held改善22.2047%，probe 38/40，median active events 4，same-task、K1、K4通过；
* G3长期没有建立可泛化的Program–bank共享功能映射；
* gate-aligned PNBTT能压低wrong和margin，但correct与held不足；
* full-rank16在task1和task93上显示相反的capacity–specificity行为。

## 由证据支持的推断

* rank16本身不是最终能力上限；
* 当前主要失败不应归因于“Writer完全看不懂视频”；
* bank有功能价值，但作为唯一support会限制容量；
* primal bypass提供容量，同时引入公共方向泄漏；
* 冻结Program与冻结compiler之间的接口是当前最可能的结构瓶颈；
* ECP前置组件通过，不意味着它们的串联映射可学习；
* serial canonicalization、pooling和transport可能造成信息与梯度损失；
* task1/task93相反行为意味着不能用一个全局scale修复PNBTT。

## 我的建议

* 终止PNBTT后续同类Recovery/R14补丁；
* 不恢复早期无约束direct hypernetwork；
* 取消严格冻结的Pass A/Pass B与固定Program瓶颈；
* 建立统一、重复、可扩展的policy-native Meta-Writer；
* native bank作为attention memory与输出先验；
* Program作为可probe的distributed latent；
* 输出采用bank-anchored signed组合加有界残差；
* functional distillation和paired causal specificity作为主要credit；
* 尽快运行task1/task93真实闭环判别，再进入shared held和validation8。

## 需要实现者自行决定的工程细节

以下选择不改变科学合同，可以根据现有代码和显存决定：

* block使用标准Transformer attention还是等价的高效attention；
* event token每视频的最大数量；
  -具体policy-native tap层；
* bank atom数量和检索top-k；
* relative position采用RoPE、bias还是显式event edge；
* frozen policy采用FSDP、tensor parallel还是单卡量化；
* grouped-LoRA kernel的具体实现；
* gradient accumulation与1–6张A40的shard策略；
* residual basis维度和初始norm cap。

这些工程选择不应被升级为新的架构Gate。下一轮最重要的不是再证明某个内部tensor满足更多约束，而是尽快回答一个决定性问题：

> **当视频事件、policy-native bank和target-rank输出在同一重复网络中接受直接functional credit时，是否能够在task1与task93上同时打破PNBTT的capacity–specificity对立，并把这种改善带到shared held任务。**

这应当成为EMBER下一阶段的第一项正式实验。

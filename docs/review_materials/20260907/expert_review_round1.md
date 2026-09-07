> Owner提供的第一轮独立专家意见。此副本仅规范化机器绝对路径前缀；论证、数值和建议未改写。专家意见属于待讨论材料，不是新实验或代码修改的授权。

# EMBER 全项目科研审查

**审查基准：`ec02710b169b3dc624b6dfca998a4ad9bdc8dd14`**

## 一、直接结论

**EMBER已经证明了“教学视频经Writer生成LoRA，可以实质改变并改善冻结policy的闭环行为”；尚未证明的是：同一个共享方法、同一个checkpoint，能够稳定地把正确视频中的可迁移过程知识编译成策略，同时满足绝对能力、任务覆盖、跨视频鲁棒性和视频因果性。**

历史强结果不应被近期弱结果抹掉。v5.2、v6-fast、LPCP和GOMQ分别提供了真实而不同的正证据；但不能把v5.2的较大视频差异、v6-fast的143、LPCP的breadth、GOMQ的151以及其它checkpoint的成功集合，拼成一个从未存在过的合格方法。仓库保存的结果本身就反对这种拼接。

我的核心判断有五项：

| 判断                                | 本次结论                                                                             |
| --------------------------------- | -------------------------------------------------------------------------------- |
| 项目是否只有工程接通、没有科学能力？                | **不是。** 历史上存在显著闭环能力，以及部分相当有区分力的干预证据。                                             |
| 反复出现的因子趋同、能量集中、公共表示是否属实？          | **属实，但不是同一个现象，也没有被证明共享一个根因。** 必须区分native坐标、rank、task和公共carrier等不同轴。              |
| 当前方法是否只是未见任务迁移差？                  | **不是。** 熟悉训练视频上的闭环能力也弱，说明缺口已经出现在训练任务的共享学习层面；但这仍不能唯一定位到decoder。                   |
| 过去是否反复采用了证据不足的解释？                 | **是。** 最突出的是把局部oracle通过、几何指标变化、模块替换结果，升级成“某一接口已经排除”或“最早根因已经找到”。部分结论随后被项目自己的实验削弱。 |
| 现在是否应直接换decoder、恢复旧v6或改成全任务batch？ | **现有证据不支持直接批准任何一个作为唯一主线。** 应先完成能够改变路线判断的现存对照，避免又从一个几何症状跳到下一张图。                   |

这里最需要纠正的并不是“项目完全不重视行为”。实际上，后期合同对配对、成功集合、冻结controls、曝光和信息墙越来越认真。问题在于：**这些行为约束经常用于最终否决一个方法，却没有同样严格地约束前面选择下一种因果解释的过程。**

---

## 二、实际审查范围与证据等级

### 2.1 我实际访问到了什么

本次固定在上述参考提交。当前核心源码对应的正式训练提交是：

`e4ca59984b5e20112d6a448c11ca23de10a4c038`

我核对了它与参考提交之间的差异：参考提交领先11个提交，**没有修改Writer、核心训练器或训练配置**。科学执行代码的相关差异是一处评测准备修改：允许明确登记的development-train子集使用5个state；这对应`97a8a24a13bc06f4df74896f038c8057e26ea07f`，不是对validation400合同的改变。

实际读取包括：

* 当前指定文档、当前历史索引、配置，以及Writer、原生读取、关系模块、坐标输出、数据读取、采样、梯度重放、训练、评测和相关测试。
* `fcdb6e43…`中的旧设计、完整历史账本的关键分叉、原始专家意见，以及完整P/Q源码。
* `ac233fa0`的早期详细历史。
* `3a6f801d`的v5.2/v6设计、v6实际编码源码，以及更早Action-Forecast咨询、设计、因果诊断和后续修正文档。
* `8553b613…`中的LPCP、GOMQ原始设计。
* 提交内保留的GOMQ rank16投影合同和正式结果汇总。

容器未能直连Git，因此**没有克隆完整可达Git对象，也没有逐个提交、逐行检查全部旧代码或181节旧账本**；历史审查通过固定commit的文件和GitHub接口完成。主要路线和关键分叉已覆盖，但不能把这个访问范围写成全仓逐行形式化审计。

### 2.2 本报告采用的证据等级

| 等级          | 含义              | 本次实际情况                            |
| ----------- | --------------- | --------------------------------- |
| **代码／定义**   | 直接读到的源码、配置、数学结构 | 当前图、v6关键输入路径、完整P/Q输出、训练与评测合同等     |
| **实验原件—汇总** | 提交内的机器可读正式汇总    | GOMQ有效rank16重物化结果；不是完整逐条rollout记录 |
| **仓库报告**    | 文档对实验结果的记录      | 本文绝大多数成功数、学习曲线、梯度统计和耗时            |
| **审查推断**    | 根据以上材料形成的解释     | 明确写为判断、可能性或待检验假设                  |

例如，我直接请求了：

`runs/outputs/layered_relation_train24_joint_e4ca5998_gpu01p0235_20260907/run_contract.json`

远程返回404。因此，**当前训练实际使用的run contract、逐步metrics、exposures和checkpoint内容，并没有被本次读取或复算**。下文的67/400、151/400、克隆对照等，都不能理解成我重新运行或重新统计了原始rollout。

引用中使用固定文件版本；部分GitHub返回把文件正文包装在JSON的同一行，因而引用标记中的行号是返回正文位置。涉及关键代码和合同处，我另外给出文件、函数或原文件读取范围，避免把包装行号冒充源码行号。

### 2.3 文档存在时间层次冲突

当前README、设计和历史速览中仍有“新图尚无实现／性能证据”的旧状态文字，而后续章节已记录short4、train24和384步止损。当前`progress.md`内部也保留了已经完成事项的“下一步”表述。应按**运行提交、时间、实际结果章节**解释，不能把某个当前文件中的所有句子都当作同一时刻的有效状态。

---

## 三、科学目标、数据合同和已选方法必须分开

### 3.1 科学目标并非从项目第一天起完全不变

最初README允许语言、视频或两者作为条件，目标包含“最低可用能力的初始化，再通过task-local RL改善”。后来逐步收紧为：正确语言与action-hidden视频，在rollout前生成完整LoRA，零交互地从未见初始化完成任务。**今天的目标比最初目标更强。** 因此，早期结果既不能直接算作完成当前目标，也不能因为不满足今天的全部标准而被说成毫无贡献。

当前应区分以下五层：

| 层次          | 当前内容                                                           | 不应混入的判断                         |
| ----------- | -------------------------------------------------------------- | ------------------------------- |
| **科学目标**    | 把教学所提供的任务知识一次性写入完整策略LoRA，并在新初始化闭环发挥作用                          | 某一种token、bank、关系矩阵或decoder天然正确  |
| **数据与信息墙**  | teacher输入仅语言和RGB视频；监督动作只在授权训练侧进入policy loss；validation/test无梯度 | task ID用于数据调度就等于task ID进入Writer |
| **评测合同**    | 固定划分、严格配对、单checkpoint、相邻稳定性、跨视频和冻结后controls                    | 用内部loss、union或不同配方的优点代替合格模型     |
| **已选方法**    | 当前单probe、Action Meta、全horizon跨帧关系、集合compiler、坐标输出              | 这些选择是科学目标的逻辑必然                  |
| **实现默认／假设** | width256、四个局部block、邻域4、坐标宽64等                                  | 默认值已经达到充分容量，或关系矩阵已经具有物理语义       |

例如，“不得在首次learned跨帧消费前压平horizon”是当前已选的方法约束；它具有研究动机，但**不是“任何有效视频→LoRA方法都必须如此”的已证定理**。旧v6在这一点上不同，却确有闭环能力。

### 3.2 source与监督权限

旧配置和当前来源记录一致指向：

* foundation是generic `lerobot/pi05_base`，明确禁止`pi05_libero`。
* 对LIBERO-90与目标任务做重叠审计后保留71个source tasks。
* 先用这些source任务建立控制接口，再冻结source。
* target40按每suite的6/2/2分为24训练、8验证、8测试。
* 当前Writer训练使用授权训练任务的跨episode动作监督。

这里还有一个容易混淆的基线分类：

**train24 Source-SFT的109/400，是无视频共享适配参照；不能仅因为它读取训练动作，就把它和held任务独立expert归为同一种privileged上界。** Writer训练本身也读取train侧动作来形成functional loss。真正强特权的容量证据，是对被评估任务直接优化的独立expert、task-local解或解析初始化。另一方面，109对应rank128共享LoRA，也不是与rank16 Writer等输出容量的严格架构消融。

### 3.3 当前部署与资格合同

当前teacher读取器读取第三人称`agentview_rgb`及实际帧索引，stride5并保留末帧；没有从该路径读取teacher动作、proprio或reward。执行policy读取自身当前观测和state，则属于另一条合法路径。评测适配器只消费已经生成的完整LoRA，不在rollout中加载Writer、observer或Action Meta。

当前train24资格在`docs/layered_relation_video_writer_design.md §13.3.1`，本次核对原文件922–980行，明确要求：

* 至少两个连续登记checkpoint：correct均严格大于145/400，breadth至少7/8，四suite非零，Goal和Long各至少10。
* 相邻correct下降不超过10，churn不超过20/400，Jaccard至少0.85。
* 两点各自的same-task-other相对correct下降不超过10，跨视频churn不超过40/400，Jaccard至少0.80。
* 模型选择完成并冻结之后，才做最终视频必要性和shuffled/reversed评测；这些不能反向用于本次训练、选择或结构修正。
* 方法冻结后才进入32/8 fresh最终训练与Test。

这些数值是当前方案的操作化合同。历史曾使用不同的门槛和稳定性规定，不能追溯写成所有时期完全相同的Owner原话。

---

## 四、历史比较：架构与训练必须一起看

下列成功数除特别说明外均为**仓库报告**，不是本次重算。

### 4.1 从最早Writer到v6：能力和视频依赖很早就发生过分离

| 路线                              | 输入与输出结构                                                                    | 正证据                           | 当时没有解决的问题                                         |
| ------------------------------- | -------------------------------------------------------------------------- | ----------------------------- | ------------------------------------------------- |
| **Pooled AS Writer**            | 每帧视觉tokens全局池化，再压成4个episode tokens；learned queries和带bias的heads生成完整rank16   | 119/400；correct/wrong 119/115 | 接近公共adapter；视频变化对有效LoRA影响极小                       |
| **Conditional Spatial／早期RL**    | 保留4×4空间网格，去query直达残差，condition-only读取；一度加入paired contrast                  | 条件版本99/55；reward-only 94/87   | 视频差异部分由loss直接规定；correct能力下降，后来不再接受该contrast作为最终解法 |
| **Action-Memory／Temporal-RoPE** | 16个memory tokens读取Action Expert图文流；之后加入真实时间RoPE                            | 曾有105、108等闭环结果                | 内容敏感不等于过程敏感；加时间位置没有自然建立正确顺序语义                     |
| **Action-Forecast v1**          | 每帧完整10-step flow生成50×7 forecast；跨帧绝对时间Plan/Revision；320queries输出LoRA       | 125/400；correct/wrong 125/67  | shuffled/reversed 121/124，顺序差异几乎未传到最终LoRA         |
| **v2／Belief-v3**                | content-only路由，修正Revision；Plan与Revision分配固定表示                              | 某些内部有向差异恢复                    | 公共成分放大；没有独立完成一个新的闭环能力上限裁决                         |
| **Visual-State v4**             | visual-state→forecast→绝对时间Plan/Revision→Temporal→LoRA                      | correct109；顺序变化确实能显著改变行为      | shuffled148、reversed126，变化方向与目标相反                 |
| **v5／v5.1／v5.2**                | 转向Semantic Core与有向Procedure；v5.2增加task-token对patch的grounding               | v5.2 old132，较大视频内容／顺序差异       | 架构与recipe强耦合，绝对能力和稳定性仍不足                          |
| **v6**                          | 可训练Text/VL/Action Meta；直接patch语义与视觉转移；Core/Procedure分工；全slot编译及完整A/B heads | task-complete版本143            | 后续下降，较强absolute与较大视频差异不在同一recipe同时成立              |

早期各路线及其机制、数字直接保存在`3a6f801d:docs/action_forecast_writer_expert_consultation.md`。这份材料尤其重要，因为它表明：**“公共LoRA—封堵旁路—增强视频差异—correct下降—再改表示”这个循环，在7月已经出现。**

v5.2/v6的交叉recipe结果不能省略：

| 版本                    | Correct | 同任务另一视频 | Wrong | Shuffled | Reversed |
| --------------------- | ------: | ------: | ----: | -------: | -------: |
| v5.2 old              |     132 |     138 |    74 |       82 |       83 |
| v5.2 task-complete    |     120 |     109 |   107 |      111 |      124 |
| v6 old                |     121 |     122 |   111 |       84 |       47 |
| v6-fast task-complete |     143 |     135 |   125 |      128 |      129 |

v6-fast后续450/500/550/600为131/130/132/126。**全任务recipe并不普遍优于old recipe；更大的顺序差异也不普遍带来更高correct。**

### 4.2 多视频、稳定化、LPCP与GOMQ

这一阶段并非只反复换encoder，也大量修改了监督、优化和参数保留方式。

| 路线／干预                                              | 真实改变                                                      | 行为证据与边界                                               |
| -------------------------------------------------- | --------------------------------------------------------- | ----------------------------------------------------- |
| Target-Spectral、target ownership、atom／lane等        | 谱、能量分布、输出共享方式和容量                                          | Target-Spectral34、Target-Owned99、Atom80等；改善几何没有普遍改善行为 |
| Expert manifold、projection、tangent tube、RLS与guards | expert方向、局部约束、成功点保护或更新合成                                  | 多次保持了被约束的量，却继续丢失held成功；不能据此推断所有expert监督都无效            |
| Dynamic-K、Trace、Shared-Core                        | 改变多视频表示、聚合层次及不同视频的一致性                                     | Shared-Core约139；减少跨视频变化有正证据，但不保证增加任务能力                |
| **LPCP**                                           | 从AS139继承完整强图；新增逐层、逐rank的probe-conditioned Procedure query | 143、breadth7；相对139保留120、新增23、丢失19，仍非稳定完成              |
| **SEOD→GOMQ**                                      | 在相同LPCP载体上，真正打开37个memory input queries的奖励梯度               | GOMQ cycle2达到151；随后135、131，未保持                        |

这些负结果中，有些是科学non-pass，有些只是OOM、watchdog或吞吐合同失败。后者不能被算成相应科学函数类已经失败。早期详细账本对此有明确记录。

**LPCP和GOMQ的“fresh”尤其不能误读。**

LPCP从AS139开始，冻结既有V6、集合聚合、compiler和factor heads，只训练新增读取／conditioning路径；optimizer fresh不等于整个模型从零训练。GOMQ又从封存的LPCP143载体开始，fresh的是本轮新增训练路径和状态，不是从generic source重新学出全部151能力。

GOMQ cycle2五臂为151/139/131/127/115，确有视频差异；但cycle2→3→4为151→135→131，相邻churn42、34。这既不能叫“已经完成”，也不能叫“只是一个无视频公共adapter”。

另一个需要永久纠正的解释是rank：

$$
A_{32}=\begin{bmatrix}A_0\\A_0\end{bmatrix},\qquad
B_{32}=[B_0,\Delta B]
$$

所以实数代数上：

$$
B_{32}A_{32}=(B_0+\Delta B)A_0,
$$

有效rank本来就不超过16。后来的rank16重物化得到136/400，保留123、新增13、丢失28，是本次直接读到的提交内正式汇总；**它不是“有效容量由32减到16导致掉分”的证据**。数值表示、舍入和闭环敏感性可以解释这种差异，但本次没有adapter字节和逐条轨迹，不能进一步把它们宣布为已验证原因。

### 4.3 Privileged容量、G1/G2、G3及后继编译器

| 路线                                              | 架构／训练职责                                                            | 获得的证据                                                                 | 不能推出什么                                        |
| ----------------------------------------------- | ------------------------------------------------------------------ | --------------------------------------------------------------------- | --------------------------------------------- |
| **Privileged experts／effects／realizer**         | 独立任务优化、phase/effect监督、fixed-A及共享realizer                           | validation任务独立expert250/400；部分effect decoder从低source提高；若干投影确实破坏已知成功方向 | 不能作为共享Writer的性能，也不能证明固定effect表示已足够可学          |
| **G1**                                          | 真实native X/Y，signed pooling，carrier12+mobile4；分组读出与privileged解析初始化 | step0达到114/250、breadth5/5；局部rank4更新存在行为容量                             | 不是共享语言／视频映射已经学会，更不是从随机初始化学会                   |
| **G2**                                          | 原生response、Natural Program、单调event alignment、动态K                   | boundary修正后full比endpoints的辅助功能loss改善22.2%，probe38/40                  | 不等于完整Writer闭环成立，也不等于固定Program已经充分             |
| **G3早期／dual basis**                             | 冻结G2，学习Program到native因子／dual坐标                                     | 某些局部解析解和宽key读出可达；共享闭环仍弱                                               | 不能把oracle可达性当作共享映射的可学习性                       |
| **F1/F2/F3、functional sketch、primal**           | 当前bank几何、协方差、两侧credit、functional子空间及primal表达                       | 若干operator容量很强；shared recovery多轮仍低                                    | F1解析构造与F2有限步学习不是纯粹的单变量因果比较                    |
| **Joint Program／routing／candidate interaction** | 解冻Program，fixed route与functional-code正控，候选级交互                      | 特定固定route可恢复强方向；correct/wrong往往耦合                                     | 固定route正控不是未见任务的真实Program正控                   |
| **EBSRI／quotient**                              | Program-relative set统计、summary条件化候选修正                              | S0/S1局部容量通过；S2真实共享映射和specificity仍未同时成立                                | 不能据此宣布只缺少一种set summary或只坏在absolute code       |
| **PNBTT**                                       | query通过当前bank的key几何、whitening和signed transport决定因子                 | free-query E1多版本稳定non-pass                                            | **实际Program驱动的E2没有运行，不能归罪于冻结Natural Program** |

G1有一项值得保留的强证据：将已知成功更新投影到当时受限的native输出空间后，strict250从120降至109，Goal/Long由11/8降至0/0。这是对“被删掉的方向有行为价值”的直接干预，而不是仅凭低rank猜原因。但随后114的通过同时包含分组、数值处理和privileged初始化，不能唯一归功于某一个修改。

G3的实际结果反复展示的是：

$$
\forall c,\;\exists \Delta W_c\text{ 有用}
\quad\not\Rightarrow\quad
\exists W_\phi(c)\text{ 能由有限共享训练学到这些更新}.
$$

后续不少诊断正确指出了这个区别，却又在下一轮选择中把某个局部通过的接口当成暂时“排除项”。F1解析容量、F2/F3共享学习、固定route正控、真实Natural Program，必须始终保留为不同层次。

PNBTT原设计对此非常清楚：E1训练free query与key projections；E2才训练真实Program-to-query。已完成的失败发生在E1。把它写成“Natural Program经bank转换失败”，已经越过实际实验。

### 4.4 Full-response、Axial、Unified、P/Q与完整输出

这一阶段逐步保留完整layer/probe/horizon、velocity和native X/Y，取消固定event瓶颈，显式区分X/Y side，再从Process→Composer转向重复统一block，最后引入共同P/Q并恢复完整A/B生成。

但这些修改不能概括成“只是换decoder”：

* full-response改变输入及其轴结构；
* Axial／Native-Temporal改变时间、target和factor-side之间的通信；
* Unified改变每层是否直接读取证据，以及softmax的分组；
* P/Q增加共同过程状态与整策略状态的反复交互；
* 完整输出同时移除carrier12+mobile4分配、raw-X/Y-only限制和mobile静态零约束，并换成完整factor heads。

关键结果是：

| 方法／对照                                        | 结果                                | 正确解释                                 |
| -------------------------------------------- | --------------------------------- | ------------------------------------ |
| Unified v3                                   | held5相邻35/31，对carrier43           | task-local正控没有转化为共享闭环                |
| Unified v4，拆分language/patch/response softmax | 45/40；Goal/Long仍0                 | 修复attention分配没有带来稳定净增                |
| A2 meta73                                    | strict400主视频80/81/79，另一视频78/81/75 | 较低churn不能补偿绝对能力不足                    |
| 有限输出P/Q                                      | 四任务64步fit/held41/39，各150          | P/Q本身尚无胜出证据                          |
| **完整输出P/Q**                                  | 同面板64/62                          | 耦合输出重构有实质收益，主要来自Goal；Spatial和稳定性仍有代价 |
| 完整输出meta73／target18                          | 训练18任务42/180→55/180；validation仍窄  | 目标组成影响学习，但移除meta未解决整体问题              |
| 同图whole-Writer clones                        | 两训练任务14/20，共享3/20；随机共享4/20        | 存在共享学习代价；不是仅未见任务迁移                   |
| 同图width256                                   | 训练完成，四checkpoint保存；**无闭环结果**      | 扩容成败尚未裁决                             |

上述P/Q实际输出源码也与设计相符：family共享heads生成全部A/B，A有非零模板、B零初始化；不是历史native signed pooling的换名。

### 4.5 训练口径对照：最容易导致错误归因的地方

| 方法                    | 初始化／冻结                                  | 每次真实更新及曝光                                                           | 关键区别                            |
| --------------------- | --------------------------------------- | ------------------------------------------------------------------- | ------------------------------- |
| v5.2原设计               | Writer及Meta fresh，source冻结              | 原设计每更新4个task、每task B21                                              | 不是v6 task-complete的同一更新语义       |
| v6-fast               | Text/VL/Action Meta与Writer联合学习          | 每更新24tasks×20queries；step50/400每task约1000/8000queries；50-episode视频池 | task数、视频支持、observer、结构与预算一起变化   |
| LPCP                  | AS139主体冻结，新增路径和optimizer fresh          | train24 task-complete、B20；真实K1–4轮换                                  | 强能力大量继承自AS139                   |
| GOMQ                  | 继承LPCP143；打开memory-query训练              | full24、每task四组K4条件，successful-expert occupancy credit               | cycle不是普通AS step；成功轨迹支持也不同      |
| 完整P/Q meta73          | 部分G2组件初始化，source/observer冻结             | **每次更新73个task，各8queries**；128updates                                | 已经是全任务batch，不是少任务更新             |
| 完整P/Q target18/random | component或全随机Writer；observer冻结          | **每次更新18个task，各8queries**；每task1024queries                          | 同图、同曝光的初始化和任务组成对照               |
| 当前layered train24     | **Writer和Action Meta全部fresh**，不继承short4 | 每更新4tasks×16queries；384步每task1024queries；视频0–15，真实K1/2/4            | 比近期P/Q视频池大、Meta可训，但每更新任务更少，图也不同 |

这些口径分别来自原设计、当前配置和仓库实际曝光报告；仍需原run contract核验实际执行字节。

由此有三个明确结论：

**更多GPU不等于更多任务参与一次更新；更多steps不等于更多每task监督；“fresh”不等于整套模型从头训练。**

此外，“全任务batch尚未试过，因此应先试它”的叙述不符合历史。它可能对当前图有帮助，但完整P/Q的73-task和18-task更新已经表明：**全任务聚合不是通用充分条件。**

---

## 五、历史强方法为什么强，又为什么仍没有完成目标

### 5.1 可以确认的“为什么强”

从源码和干预看，强方法至少具备三类真实资产。

**第一，完整自由A/B生成已经进入policy-effective区域。** 早期Writer不依赖raw X/Y signed bank也能获得119、125、132、143等能力。因此，“必须让最终因子来自原生bank，才能生成有效LoRA”从未被历史支持。反过来，这不说明任意hypernetwork都容易学会。

**第二，v6保留了一组后来逐步拆开的职责。** 在`3a6f801d:src/ember/writer/video_program.py`原文件320–520行，实际代码同时包含：

* 可训练Text Meta产生的contextual language queries；
* VL Meta参与的图文context；
* task query读取patch内容；
* Action Meta参与的Action Expert响应；
* 直接patch evidence与语义evidence的组合。

它确实对最终50个suffix hidden做mean，但并不是“只从一个压平动作向量生成全部LoRA”。直接视觉与图文语义路径是重要组成部分。

**第三，LPCP/GOMQ优先保留已经有效的强载体。** 它们的短期收益是在强行为邻域里改变读取和更新，而不是要求一张全新图同时重新学会grounding、过程抽象和参数生成。这解释了它们为什么不能与当前fresh全图只按cycle数比较。

这些是结构与训练层面的解释，**不是已经做完所有正交消融后的独立因果分解**。

### 5.2 为什么仍未完成

未解决的问题并不相同：

* Pooled Writer的强分很大程度可由公共adapter解释。
* Forecast v1和部分v5/v6配方依赖视频内容，却未表现足够顺序必要性。
* v4顺序显著影响行为，但符号错误。
* v5.2 old的较大视频差异没有同时达到更高absolute。
* v6-fast、LPCP和GOMQ的成功不能稳定共同保留。
* GOMQ的151还没有成为等价rank16物化下可直接复现的151。
* 各强方法的Goal/Long贡献、不同video及相邻checkpoint必须在同一运行版本上核对，不能从不同报告拼接。

所以，历史强方法应当作为**行为能力锚点和职责参照**，而不是直接恢复为正确答案。

---

## 六、重复诊断与推进决策：哪些解释成立，哪些被过度扩张

### 6.1 “趋同”至少有四种，不能混写

当前材料中反复出现的“公共”包括：

1. **native坐标上的常量分量**：例如一个B列在输出坐标上近似常数。
2. **rank列之间趋同**：不同rank产生相似B列，可能降低有效rank。
3. **任务之间趋同**：不同语言／视频生成近似相同有效更新。
4. **公共载体掩盖残差**：完整LoRA高度相似，但小残差仍决定关键行为。

这四者不等价。对A/B因子本身，还存在：

$$
BA=(BQ)(Q^{-1}A)
$$

这样的可逆换基自由度。因子cosine或norm可以随表示改变，而功能矩阵不变；即使比较的是有效\(BA\)，闭环行为仍可能对低能量方向和状态分布高度敏感。

### 6.2 把现象、干预与行为接起来

| 原因假设                            | 已做干预                                                     | 现象是否改变        | 行为是否按假设改善          | 本次裁决                              |
| ------------------------------- | -------------------------------------------------------- | ------------- | ------------------ | --------------------------------- |
| 能量过于集中／rank太低导致弱能力              | 谱正则、atom、target ownership等                               | 多次改变          | 经常没有，甚至明显下降        | 低rank不是普遍根因；只能在特定可达方向丢失时形成强解释     |
| 公共表示淹没任务差异                      | 去均值、静动态分路、owner/rank分工、不同memory读法                        | 多次恢复内部差异      | 不稳定                | 高公共相似度不能单独判坏；可能是有用共享结构，也可能掩盖小有效差异 |
| 视频顺序没有传到输出                      | RoPE、Revision、Temporal、event、full-response等              | 多次显著增强顺序敏感性   | v4甚至使错误顺序更好        | 顺序敏感性是条件，不是正确过程理解的充分证据            |
| 共享梯度冲突导致漂移                      | projection、guards、成功key、occupancy credit、task-complete更新 | 局部约束／cosine改善 | held成功仍换手          | 共享学习代价存在；尚不能唯一归为梯度冲突              |
| native输出来自真实policy，因此没有固定span问题 | signed X/Y pooling                                       | 最初假设本身不成立     | G1后续修正才获得局部容量      | 数学限制必须先准确推导，不能靠“native”命名豁免       |
| memory输入没被学到                    | 修复SEOD no-grad边界，形成GOMQ                                  | 梯度与实际参数更新恢复   | cycle2有明显正收益，但后续掉分 | 这是有效的具体工程因果解释，但不解释全部后继失败          |
| 当前B常量／rank共享是主因                 | std0.02→1、末读出按target/rank独立                              | 不同轴的常量性显著变化   | 小面板有增益，完整train24仍弱 | 支持局部读出干预，不支持“上游已正确，只差decoder”     |

历史Target-Spectral、ownership、guard和energy-preserving反例见早期账本；它们共同削弱的是这些指标的**普遍充分性**，不是说这些指标永远没有用途。

### 6.3 当前读出干预尤其说明：不能把所有常量性统称“坍缩”

当前short4报告中：

* 坐标初始化std增大后，native常量分量下降，但rank方向仍高度趋同。
* 将末读出从全局共享改为target/rank独立后，rank常量分量明显下降，96步两组正确视频都达到11/40。
* **同时，native坐标常量分量反而更高。**
* 后续完整train24仍只有较弱闭环能力。

这并不否定读出共享曾有问题。它否定的是更强的故事：

> “一个统一的常量坍缩指标下降，就会带来能力提升；指标仍高，就说明该接口仍是根因。”

对当前坐标MLP，可以从代码推导小坐标扰动容易产生公共项。例如：

$$
B_{m,o,r}=u_{m,r}^{\top}\sigma(z_{m,r}+e_{m,o}).
$$

当\(e_{m,o}\)很小时，一阶展开确实包含主导的、与native坐标\(o\)无关的项。这能够解释一种**初始化附近的几何偏置**，但它没有证明该偏置是384步低分的主要原因。也不能仅凭中间宽度64，就把这个非线性坐标网络等同于固定64维线性输出子空间。

### 6.4 v4的前后修正，是整个项目最值得记住的一条历史

最初forecast-order移植显示：固定逐帧forecast，只改变后续时间槽位，就能复现大部分行为差异。这支持绝对时间Plan/Revision是一个直接放大器。

但后续`action_forecast_writer_v4_root_cause.md`明确撤回“保留上游，只修下游”的过早结论：

* visual-state neutral化几乎不改变forecast；
* 它可能已经被raw-image/Meta路径旁路；
* AS loss下降时，forecast的“最新更准”和“残差是纠错方向”假设反而恶化；
* 同任务独立video/action配对不能自动识别内部forecast的过程语义。

**这意味着“下游忠实地传递了差异”不能证明上游语义正确。这个教训在当前关系表示→compiler→decoder链上仍然适用。**

### 6.5 需要修正的专家推理

这里有几项不是观点差异，而是推理范围问题。

**第一，8月24日把native Y pooling视为自动摆脱固定输出空间，数学上不成立。** 若\(Y=WX+b\)，标量权重的线性组合仍在\(\operatorname{span}(W,b)\)内。G1随后发现并修正q与action-in的相关限制，恰好说明原论证不够。

**第二，8月28日把F1解析容量与F2有限步共享学习的差距称为很干净的因果对照，说得过强。** 这同时改变了oracle求解与学习算法。它不能独立证明“全局bank context是必要原因”。F2/F3更匹配的比较可以支持某种context有帮助，但仍不是普遍必要性结论。

**第三，部分意见把GOMQ的rank32书写当作更高有效容量，把PNBTT E1失败延伸到冻结Program。** 这两项已被后来的历史修正；它们不应在新图讨论中重新出现。

**第四，“换回某个未训练模块后变好／变坏”，不足以单独判定该模块职责失效。** 训练模块之间可能共适应；模块交换本身也可能破坏坐标约定。因此Axial／Unified阶段的whole-module和block替换，应被视为交互敏感性证据，而非自动定位唯一故障模块。

**第五，v5.2阶段的B整体放大试验只能削弱“统一幅度太小”的解释，不能排除decoder的方向、局部Jacobian、条件分辨率或共享可学习性问题。** 同理，某个attention类别的概率质量接近token比例，不足以直接说明它是行为瓶颈。

不过，后续确有实质进步：同图clone去掉额外task query；记录真实曝光；区分熟悉视频与新视频；纠正rank；冻结controls；禁止union；保留相邻成功集合。这些不是纯文档修辞。问题是它们尚未完全阻止“局部诊断通过→上游暂时免检→换下一接口”的推进模式。

---

## 七、当前layered relation Writer：实际图、贡献证据和能力缺口

### 7.1 实际前向图

当前源码对应的数据流是：

```text
exact language + K条独立有序RGB视频
    ↓
冻结图像／语言prefix
    ↓  单个固定noise probe，flow time=1
Action Expert + 仅观察侧可训练Meta
    ↓
R[视频帧, 18层, 50 horizon, 1024]
    ↓
投影到width256
    ↓
4个同层、局部双向帧对关系block
    ↓
learned horizon read
    ↓
E[视频帧, 18层, 256]
    ↓
38 targets × 16 ranks的整策略queries
读取所有视频的过程集合，并进行query间交互
    ↓
target/rank-conditioned坐标MLP
    ↓
完整38-target rank16 A/B
    ↓
冻结source闭环执行；不再读取teacher视频
```

关系模块正确区分了视频时间、action horizon和网络层；帧对两端分别对score与转置score做row-softmax，不是转置已经归一化的attention；每个block同步读取旧状态。当前代码还保留对应内容与horizon位移分布，而非只取内容差。

但以下语义都还没有由结构本身证明：

* horizon对应不是teacher真实动作时钟。
* 同probe下跨帧匹配不自动是物理运动对应。
* 四层局部关系不自动形成高层过程表示。
* 保留更多轴，不自动意味着更容易被functional训练利用。

### 7.2 实际梯度与缓存

我没有在所读代码中发现一个足以统一解释弱结果的明显断梯度或权重错误：

* source被冻结。
* Action Meta只在观察侧安装。
* policy对生成LoRA leaves产生VJP，再重放Writer和observer。
* 持久缓存属于冻结prefix；没有把更新后的Meta响应R永久当作冻结证据复用。
* 每task权重已在本地形成，跨卡SUM后没有再错误除以world size。
* 真实K与任务权重、microbatch拆分被分开处理。

identity初始化时第一步Meta零梯度是可解释的：B和最终读出从零起步，上游一阶梯度尚未打开。报告中第二步Meta出现梯度，与这个结构一致；不能把第一步零梯度重演为SEOD的永久断梯度问题。真实full/staged一致性数字目前仍只是仓库报告，而本次读到的CPU测试主要验证代数、依赖范围与toy梯度，不等于闭环贡献。

### 7.3 当前并没有结构性保证“必须用视频”

Compiler查询包含语言，并保留query residual直达最终code。因此，动态Value路径存在，不等于模型不能借语言和公共查询生成大部分LoRA。

另外，当前直接language-query路径读取的是冻结词嵌入，再用单query聚合；它不是旧v6的可训练contextual text路径。**这不意味着整个Writer不理解词序**——Action Expert响应仍受contextual prefix影响——但说明不能把当前直接语言接口视为旧接口的等价继承。

### 7.4 当前行为缺口已定位到什么程度

| 面板                      |     Source | 192步 | 384步 |
| ----------------------- | ---------: | ---: | ---: |
| validation400，correct   |         47 |   69 |   67 |
| validation400，同任务另一正确视频 | 同一source参照 |   72 |   64 |
| train120，held视频         |         16 |   22 |   18 |
| train120，熟悉视频           |         16 |    — |   21 |

384步correct的suite分布是Spatial3、Object33、Goal27、Long4。相邻correct保留46、新增21、丢失23，churn44，Jaccard约0.511；384步两正确视频之间保留52，Jaccard约0.658。**不仅absolute没过，成功集合稳定性也远未过。**

这里已经可以作出三个有边界的判断：

**一是存在真实局部能力，而不是全图完全无效。** Object增益在两组正确视频上重复出现。

**二是存在能力取舍。** 当前source主要Goal成功与后来Object增益之间出现明显交换，不能只看总分增加约20。

**三是问题不只在未见task或未见video。** 熟悉视频也只有21/120。但是，熟悉组中89个条件曾以K1出现，另31个视频只曾参与K2/K4；而rollout初始化仍不同。因此这组结果不能进一步证明“训练分布上都完全拟合不了”，也不能排除闭环状态支持问题。

### 7.5 当前各组件的处置建议

| 组件                               | 证据状态                   | 建议                                       |
| -------------------------------- | ---------------------- | ---------------------------------------- |
| source、信息墙、唯一完整LoRA、配对评测         | 有明确工程合同和长期复用价值         | **保留。** 不通过换source或放宽输入来掩盖当前问题           |
| 冻结prefix／Meta重放                  | 代码合理，报告有机制验证           | **保留实现资产；不宣称Meta已证明有行为贡献**               |
| 单probe与完整18×50响应                 | 是合法观测选择                | **保留为候选，不升格为已充分证据**；不能据此排除输入／grounding问题 |
| 局部帧对关系、位移分布、同步更新                 | 主要是工程接通证据              | **暂不扩window、depth或给关系矩阵追加“更物理”的约束**      |
| horizon read及集合compiler          | 结构合理，职责未独立裁决           | **不宣布上游正确**；K置换不变也不等于多视频有行为收益            |
| 当前target/rank独立末读出               | 有匹配short4局部正证据         | **保留这一版本作为当前被审查基线**，不退回已弱于它的共享读出         |
| 坐标MLP整体                          | 有初始化偏置和局部可学习性疑点，尚非唯一根因 | **不批准仅凭常量性启动又一次decoder主线**               |
| 旧直接patch／contextual text／全局P/Q职责 | 历史上处于强路径；当前不等价继承       | **作为可能丢失的职责记录，不自动全部加回，也不宣布已经不需要**        |

---

## 八、目前最有根据的竞争解释

我不建议给这些解释编造精确概率。现有材料支持的是不同范围的竞争关系。

| 解释                                 | 支持证据                                                               | 反证／限制                                            | 仍缺什么                                       |
| ---------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------ | ------------------------------------------ |
| **共享条件映射在当前预算下未学会：表示、参数共享与优化共同限制** | 同图clone14/20对shared3/20；当前熟悉视频仍弱；完整输出和末读出解共享曾改善局部能力                | clone可以学近似固定LoRA；不能区分容量、干扰、初始化、条件辨识；全任务batch也失败过 | matched完整图容量结果、真实每task学习轨迹与行为配对            |
| **functional目标和监督支持不足以形成稳定闭环控制**   | 多次loss／内部recovery改善而行为退化；guards保护局部点却不保护held成功                     | 不能把所有问题统一叫occupancy；部分teacher-state接续也失败，失败阶段异质  | 相同checkpoint、相同任务的监督损失、访问状态和阶段失败原件         |
| **任务组成与能力保持产生取舍**                  | meta73→target18改善训练目标；当前Object增益伴随Goal损失                           | 加train24、更大视频池、真实K仍未解决；并非所有弱任务都曾学会后遗忘            | 哪些任务是从未学起，哪些确实获得后丢失；不能仅凭terminal分数判断       |
| **数据对“视频过程知识”缺乏充分识别约束**            | exact language常能标识任务；跨episode阻断复制，却允许公共／语言条件controller；7月v4已经指出这一点 | 历史正确视频与wrong/order controls确实存在行为差异，不能说视频毫无用处    | 冻结候选上的完整controls，以及对差异究竟来自任务内容、终态还是过程的谨慎解释 |

最后一项尤其重要。给定正确语言、执行观测和训练任务动作监督，模型可能通过多种内部机制降低同一个loss。**“输入始终是正确有序视频”并不唯一识别“内部学到可迁移过程”。**

同任务另一视频表现相近，证明鲁棒性；wrong变差，证明某种内容匹配有用；shuffle变差，证明顺序扰动有行为影响。它们合起来很有价值，但仍应避免把这些结果直接扩大为任意跨身体、高层程序迁移。当前LIBERO设置主要检验的是同一具身接口上的新初始化／新任务适应。

统计上也需要保留一个边界：400条rollout来自8个任务，不能把它们当成400个独立“新任务”样本；长期重复使用validation进行架构选择，会使最终sealed Test的重要性更高。

---

## 九、最优先的下一项行动

### 9.1 当前只读授权下：先恢复一个最小、可核对的证据包

不是要求上传整个`runs/`，也不是等待所有材料齐全才继续研究。

首先应将**当前384步、历史强方法、完整P/Q width对照**所需的最小contract、曝光和配对结果放到可审查位置。其作用是确定接下来比较的真的是对应运行，而不是又比较了文档中的名称。

### 9.2 若重新授权科研执行：我优先建议关闭“完整P/Q同图width对照”这个未裁决分支

我不建议把“原生线性decoder”直接作为下一个主线。

优先对象是已经训练完成的：

`runs/outputs/pi05_ecp_prw_complete_target18_width256_s128_14bc7605_gpu02p012356_20260906/`

与其已有的width128 **fully-random**对照，而不是component初始化对照。

原因不是已经花过训练费用，而是它具有当前许多新方案没有的区分力：

* 同一完整P/Q拓扑；
* 同样18任务、两条fit视频、K1；
* 每task同样1024queries；
* 相同随机初始化原则和优化／采样合同；
* 同样完整rank16、无独立carrier；
* 只把整体width128→256、heads4→8，并同时扩充主干和heads；
* 四个checkpoint已经存在，至今没有闭环裁决。

**这不是检验“当前layered再加宽”，也不是恢复旧方法为正式方案。** 它检验的是：上一张完整图的共享失败，是否在一个已经训练好的、匹配曝光的更充分函数类上仍然存在。当前图本身已经是width256，更不能把这个历史结果省略后继续笼统讨论“宽度不足”。

#### 保持什么

保持原checkpoint、source、normalization、视频ordinal、state、policy/env RNG、动作执行合同；不重新训练，不换seed，不重新选择视频，不引入negative controls或Test。

#### 实际观察什么

先恢复原登记的四点validation80 screen，以及固定terminal训练诊断。为避免只靠两个根据失败挑出的任务下结论，若结果需要支撑路线选择，应把**两个terminal模型放到相同18-task训练breadth面板**比较；已有匹配rows直接复用。

80-row screen继续只决定投入，不选择最终模型，不线性外推400。只有广泛能力和相邻保持支持时，才按原规则扩strict400。

#### 不同结果分别意味着什么

| 结果                               | 可以改变的判断                               | 下一步决定                                   |
| -------------------------------- | ------------------------------------- | --------------------------------------- |
| width256在训练breadth和未见任务上都明显、持续改善 | “完整P/Q路线已经没有价值”的裁决过早；整体共享函数类／可学习性值得重开 | 先完成原资格，不同时启动layered decoder重构           |
| 训练任务显著改善，validation仍弱            | 共享任务获取与任务迁移可以被分开                      | 把后续问题转向条件泛化、输入职责和任务支持，而不是继续把所有失败压到一个末端头 |
| functional明显改善，训练与validation闭环仍弱 | 扩充同图函数类没有弥合功能监督与控制的差距                 | 优先做基于真实失败阶段的目标／支持诊断，不再追加几何正则            |
| 所有行为仍弱                           | 这一幅度的同图扩容、在匹配预算下没有解决问题                | 保留负结果；它不否定所有容量扩展，也不证明当前decoder是根因       |
| contract、采样或checkpoint无法核对       | 比较无效                                  | 记为inconclusive，不用重训或改seed自动填补           |

这项行动与历史重复试验的区别是：**它不是又做一次“更大模型试试看”，而是完成一项已经登记、已经训练、但在暂停时没有裁决的matched实验。** 它不保证解决EMBER，却能改变是否继续抛弃整图、是否把资源继续押在局部decoder上的判断。

对于当前“native linear output”建议，只有在上述证据整理之后仍有明确必要时，才值得登记为局部对照。届时必须说明它同时改变了哪些东西：共享输出basis、非线性、初始化Jacobian、参数量和target/rank ownership。不能把一个明显不同的函数类包装成“只修正常量问题”。

---

## 十、可能改变结论的最小补充材料

以下不是要求全部大模型和视频原始资产。第一轮应优先给JSON／JSONL和小量代表性记录。

| 材料                         | 准确位置／定位入口                                                                                                                                       | 最小字段及其影响                                                                                                                                                  |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **当前正式训练合同与曝光**            | `runs/outputs/layered_relation_train24_joint_e4ca5998_gpu01p0235_20260907/`                                                                     | `run_contract.json`、`metrics.jsonl`、`exposures.jsonl`；实际trainable keys、source/checkpoint身份、每step task/K/demo/query episode-frame、权重、LR、clip；验证“相同曝光”和梯度统计 |
| **当前192/384配对结果**          | `runs/analysis/layered_relation_writer_20260907/train24_shared/`                                                                                | `registration.json`、`decision_after192.json`、`decision_after384.json`，四组validation结果及source参照；逐条task/state/video/env RNG/policy RNG/success/terminal step |
| **熟悉视频诊断**                 | 同目录的`video_novelty_diagnostic_registration.json`、`video_novelty_diagnostic_results.json`                                                        | 每条熟悉视频实际K1或仅K2/K4历史、配对source及两组120 rows；决定“训练支持不足”能排除到哪一步                                                                                                 |
| **当前读出诊断**                 | `runs/analysis/layered_relation_writer_20260907/coordinate_init_control/`与`target_rank_readout_control/`                                        | registration、panel summary、相同query／noise核对、几何指标精确定义；尤其区分native轴、rank轴和task轴，不能仅给统一“collapse”数值                                                            |
| **历史强方法的同checkpoint证据**    | v6原run由`historical_v6_recipe_context.json`定位；GOMQ源root见下文                                                                                       | selected/相邻checkpoint身份、五臂逐条rows、逐task/suite、初始化继承链、真实query曝光；避免把不同强项拼接                                                                                   |
| **同图clone/shared与width对照** | `runs/analysis/pi05_ecp_prw_complete_single_task_20260906/`；`…complete_target18_20260906/`；`…complete_target18_random_20260906/`；width256正式root | 实际采样匹配、32/64/96/128功能曲线、已有行为rows、width256完成记录和checkpoint元数据                                                                                               |
| **少量能区分失败机制的轨迹**           | 当前Goal保留失败、Object新增成功，以及旧同图诊断中有强expert参照的任务                                                                                                     | 相同初始化下source／Writer／可用参照的执行观测、动作、终止点；用于区分未接触、抓取失败、放置失败和后期恢复，不需全部视频                                                                                        |

GOMQ原始结果根为：

`runs/outputs/pi05_v6_lpcp_cfmg_gomq_cycle2_k4_correct400_noreplacement_seed7_trainr6_evalr6_8553b61_gpu01p012457_b16_retry1_20260817`

rank16重物化根为：

`runs/outputs/pi05_gomq_cycle2_effective_rank16_correct400_seed7_ac233fa_gpu01p123457_20260824`

这两者最少需要adapter身份、转换记录及配对success rows；不需要先把所有历史checkpoint上传。原件定位由提交内复现卡和正式汇总支持。

---

## 总结

EMBER最值得保留的资产，是**真实的历史闭环能力、若干具体接口的可达性证据、严格的信息墙，以及已经成熟的采样、重放和配对执行基础**。

最需要停止重复的，是以下推理：

> 发现某种公共／低rank现象 → 把它命名为最早瓶颈 → 修复指标或通过局部oracle → 暂时认定该接口正确 → 闭环不升 → 更换下一接口。

已有历史表明，这个流程有时修复了真问题，例如绝对时间构造、native span和memory断梯度；但它也多次把**相关症状、有限预算学习失败和结构偏置**当成更强的因果结论。

**当前最稳妥的科学结论不是“layered错了”，也不是“只差decoder”，而是：共享条件学习尚未建立，能力保持与闭环监督仍有独立缺口，视频过程的识别性也没有被最终裁决。** 下一项行动应当缩小这些解释之间的差别，而不是再让某个内部指标变得更好看。

# 条件读写 Writer：完整架构设计

状态：Owner于2026-10-01授权按最终版本实施并自主推进；本稿现为active design，首批范围见§13。
对应实现已进入canonical源码；实际运行、冻结版本与结果只看[progress](../../progress.md)，不把实现完成写成科学阳性。
本文承接[专家原文](../review_materials/20260930_t_architecture/EXPERT_RESPONSE.md)、
[讨论整合](../review_materials/20260930_t_architecture/PUBLIC_BASE_REVIEW.md)及Owner随后对执行能力和元学习的澄清。
下文规定当前条件读写计算；旧operator实验成绩保持原适用范围，不作为本设计结果。

## 1. 这次定下什么

从一条正确、action-hidden教学视频和exact language出发，在rollout前一次编译一套完整38-target LoRA。
source复用既有canonical资产：由generic lerobot/pi05_base在过滤后的71-task source corpus上建立，随后冻结；
不使用读过目标40-task actions的pi05_libero，也不为本设计重算normalization。
冻结source与该LoRA共同承担具体操控；Writer学习从教学获得、表达和编译适配的规则，执行时退出。
A₀/B₀是跨任务共同学习的执行参数与native读取参照，S/M是此次教学生成的参数调整。
不要求知识在四个因子中互斥分区，也不额外要求无视频公共函数独立完成任务。

本稿采用单一完整FM目标，共同训练公共参数、视频解释器和两侧生成头。
公共分数保留为诊断；不加入public FM、随机public/full分支、条件作用收缩、因子吸收或辅助teacher动作监督。
这项取舍有原T与公共辅助/Joint正反证据支持，但不证明公共FM在所有后继模型上无效。

本稿保留Owner此前明确要求的前序编码不读后帧：使用帧块因果mask和后向差分。
专家认为离线编译也可用双向解释；那是可讨论的替代，本稿没有据此撤销已有因果要求。
因果性只约束视频编码；片尾得到的最终A及随后编译的M可以使用整段教学。

为使设计具体，首版实例采用K=1、每target rank128、4个独立1024宽解释层、16个attention heads、4096宽FFN、dropout=0。
这些是本稿配置选择，不是永久容量限制或已验证最优值。两台相机属于一条视频，不是K=2；不声称支持未训练的dynamic K。

## 2. 完整数据流

```text
一条教学：同步双相机RGB + exact language
  │ stride5，保留真实末帧；不读teacher action/state/reward
  ▼
冻结source + 可学习公共A₀/B₀，逐帧native读取
  ├─ 38处真实层输入 X_l[t] ──────────────────────────────────┐
  └─ 完整动作响应 H[t,50,1024]                              │
       │ token内归一化；后向差分；时间/槽位分别编码             │
       ▼                                                     │
  4层因果解释器：上下文c + 零保持动态d                         │
       │                                                     │
       ├─────────────┐                                       │
       ▼             │                                       ▼
  S头读取 c/d、真实X与A₀X ──按序关联写入── S_l[r,d_in]
       │
       ▼
  A_V,l = A₀,l + S_l              片尾形成每层最终A
       │
       ├─ 对保存的X重新计算 K=normalize(A_V X)
       └─ 计算读取变化 δz=S X
               │ c/d、K、δz
               ▼
          M头 + 按序关联写入 ── M_l[d_out,r]
               │
               ▼
      唯一LoRA：{A_V,l, B_V,l=B₀,l+M_l}，共38处
               │ Writer结束；固定参数
               ▼
自身图像/state/语言 + 自身动作噪声 → 完整policy十步flow
               │ 预测50步动作，执行前5步
               ▼
          环境反馈 → 新的自身图像/state → 再规划

训练专用支路：同task、不同episode的query + query动作标签
  → 完整policy的一次FM预测 → 完整FM误差
  → A/B直接执行路径 + M编译 + S编译 + c/d解释 + 公共native
  → 更新A₀/B₀及Writer参数；source基础权重不更新
```

## 3. 输入、轴和张量

以下使用列为horizon槽的矩阵写法；外部展示H用N×50×1024，计算H[t]用1024×50。
N是stride5并保留末帧后的帧数，合法教学至少有2帧。视频时间t、动作horizon j、flow time τ和层编号l是四个不同轴。
50个槽是该帧原生动作网络的内部预测位置，不是未来50帧或实际teacher动作。

| 对象 | 形状/来源 | 消费者 |
| --- | --- | --- |
| 教学RGB | native入口N×2×3×H_render×W_render uint8；内部resize224并映射到模型范围，两相机同步 | 每帧真实视觉/语言prefix |
| exact language | 原任务文本与token mask | 教学native及自身policy；不另加整数task ID |
| native probe | 50×32，固定seed1729的公开Gaussian噪声；τ=1 | 公共action expert |
| H[t] | 1024×50，公共action expert的真实末端hidden | 上下文与动态解释器 |
| X_l[t] | d_in,l×50，对应LoRA target的真实输入 | S写入、A_V寻址及M写入 |
| c[t],d[t] | 各1024×50 | 38组S/M生成头 |
| A₀,l / S_l | 各128×d_in,l | 组成同一A_V,l |
| B₀,l / M_l | 各d_out,l×128 | 组成同一B_V,l |
| 自身query | 双RGB、8维state、语言、50步7维action | 训练期完整FM；动作标签不进入Writer |

38 targets延续以下精确分组。rank=alpha=128，所以LoRA乘法scale=alpha/rank=1；不另加输出scale。

| target | 数量 | d_in → d_out | A₀/S | B₀/M |
| --- | --- | --- | --- | --- |
| action expert Q | 18 | 1024 → 2048 | 128×1024 | 2048×128 |
| action expert V | 18 | 1024 → 256 | 128×1024 | 256×128 |
| action_in | 1 | 32 → 1024 | 128×32 | 1024×128 |
| action_out | 1 | 1024 → 32 | 128×1024 | 32×128 |

每处只输出一对因子，LoRA配置rank仍为128，不是把公共与条件两套rank相加；
矩阵乘积的实际秩还受d_in/d_out限制，action_in/out尤其不能按实际rank128理解。

## 4. 公共native：用真实策略计算读取教学

令β={A₀,l,B₀,l}。每帧在W_source,l+B₀,l A₀,l下计算真实双RGB/语言prefix和action suffix，
使用同一公开probe、固定τ=1，不输入teacher state或由真实teacher动作构造的latent。
teacher语言prompt省略State段，不是把state填零；双视角共512个视觉tokens，语言序列按既有200位置/mask合同处理。
H是18层action expert和最终norm之后、action_out之前的50×1024 hidden。
Q/V的X是各自projection的真实输入，action_in的X是固定probe，action_out的X就是H；不是38份相同的末层H。
这样相邻帧的H差异不由重新抽取probe造成；固定probe/端点是本版读取选择，不是信息墙或未来所有方法的原则。

对各帧读取X/H后，同一组特征依次用于S编译和M编译。没有先安装S/M再做第二遍native，
也不把FM query的hidden回灌Writer。训练期X/H随β更新重新计算，不能跨optimizer step沿用旧特征。
重放反向所需的同参数版本native不构成第二个科学读取阶段。

source的基础参数冻结，但其算子对输入/LoRA的导数仍须保留；不能把整条native放进no_grad后切断β信用。
与β无关的冻结视觉/语言prefix可按既有实现复用。
各target的间接路径不同：native执行action_out后丢弃velocity，该处A₀/B₀不会经被丢弃的输出改变X/H；
它们仍有完整执行及本target生成头/寻址中的相关信用，不能宣称76个公共张量都有同样的native梯度路径。

## 5. 因果解释器：证据可以离开发生位置

逐token对1024个通道归一化，得到H̄[t,j]；禁止用整个视频的均值或方差改变前序token。
上下文初值c⁰=H̄+时间位置编码+独立horizon位置编码，动态初值为

    d⁰[0,j] = 0
    d⁰[t,j] = H̄[t,j] − H̄[t−1,j], t≥1.

时间位置取真实采样索引，不以最终N或未来内容归一化；horizon位置仅标识同帧内部的50槽。
H的完整网格保留到学习算子，禁止先平均horizon或把视频压成一个均值。

每层从c产生Q/K，以相同帧块mask做上下文self-attention和动态传递：

    α_ij = softmax_j(Q(c_i)K(c_j)ᵀ/√64 + mask_ij)
    mask_ij = 0 if frame(j)≤frame(i), else −∞
    c ← 普通pre-norm attention残差 + 上下文FFN残差
    d ← d + W_o Σ_j α_ij W_v RMS₀(d_j)
    d ← d + W₂ [GELU(W₁ RMS₀(d)) ⊙ sigmoid(W_g c + b_g)].

多头按16×64计算，再合并为1024维。这里RMS₀(0)=0；动态W_v/W_o/W₁/W₂均无bias，
动态流不加位置向量、语言常量或独立静态残差。上下文流可以有bias，因其只选择/调制动态内容。
4层参数各自独立；两流共享每层的注意力权重，不共享Value/输出投影。
生成头可额外读取对应真实H的幅度信息作上下文门控，但不能把它绕过动态乘法直接写成Value；
本版明确把对应H与c共同送入门控，动态内容只来自d。

整段d⁰=0时，每层d及两侧Value都为0，最终S=M=0。
局部d⁰[t]=0并不要求d[t]=0：过去的真实变化可沿attention传到后来静止状态。
最后Value不再乘回本地原始差分，避免把证据来源与写入位置重新绑死。
这是输入依赖与传播性质，不等于已证实物理因果理解；运动仍可能被学成任务模板开关。

## 6. 明确转移索引，再编译S

本版采用转移起点寻址，与既有§90的因果版本一致。转移(t−1,t)在t时可知，
写入使用保存的X_l[t−1]，而c[t]/d[t]是在转移到达时形成的表示。两次编译均按t=1…N−1处理。
末帧进入最后一个真实转移和上下文；不虚构末帧后的转移，也不额外对最后地址做零Value覆盖。
这与按X[t]寻址的终点版本不同，不声称仅加mask就精确包含原T函数。

省略层编号，令X=X[t−1]、ξ_j=x_j/max(||x_j||₂,ε)、Ξ=[ξ₁,…,ξ₅₀]、z₀=A₀Ξ。
S头的具体形式为

    U^A = O_A { GELU(P_X Ξ + P_z z₀ + C_A [c[t]; H[t]]) ⊙ D_A d[t] }
    S^(0) = 0
    S^(t) = S^(t−1) + (U^A − S^(t−1) Ξ) Ξᵀ / 50
    S_V = S^(N−1); A_V = A₀ + S_V.

生成头内部宽度沿当前Value头采用256：P_X为256×d_in，P_z为256×128，C_A为256×2048，
D_A为256×1024，O_A为128×256；
所有直接处理动态内容及最终输出的线性层无bias，门控内的静态bias不影响零保持性质。
U^A为128×50，是网络预测的读取响应修改，并非给定的正确中间标签。
S以真实高维X为右侧关联对象，生成128×d_in矩阵；不是仅生成128×128矩阵再左乘A₀。

A₀Ξ让生成头知道当前映射在这些输入方向上的响应。归一化只用于编译寻址，最终执行输入不做这种单位化。
S的行空间仍受该视频X张成空间限制；action_in的输入维度小于rank，不能把所有target都解释为增加行空间维数。
递推旧内容的传播有界不保证最终S/M幅度、动作稳定或迁移。

## 7. 使用最终A，重新编译M

S全部完成后，再按同一转移顺序读取保存的X以及c/d；不重复生成另一套A，不对不同A的M直接平均。

    z = A_V X[t−1]
    K = column_normalize(z)
    δz = S_V X[t−1]
    V^B = O_B { GELU(P_K K + P_δ δz + C_B [c[t]; H[t]]) ⊙ D_B d[t] }
    M^(0) = 0
    M^(t) = M^(t−1) + (V^B − M^(t−1)K) Kᵀ / 50
    B_V = B₀ + M^(N−1).

P_K/P_δ为256×128，C_B为256×2048，D_B为256×1024，O_B为d_out×256。
V^B处于该target实际输出坐标，M为d_out×128；δz保留读取改变量的幅度，补充单位化key所没有的信息。
生成后的A不detach：它改变执行读取，也改变K、覆盖与Value，须让这两类作用共同接受真实信用。

到此输出唯一集合{A_V,l,B_V,l}，其层内改动为

    ΔW_l = (B₀,l+M_l)(A₀,l+S_l)
          = B₀,l A₀,l + B₀,l S_l + M_l A₀,l + M_l S_l.

B₀在视频适配后仍通过B₀S参与控制，A₀仍通过MA₀被利用；公共参数不是仅在无视频模式才工作。
诊断无视频公共函数必须同时令S=M=0；只关M仍是视频条件策略。

## 8. 部署：编译结束后，Writer退出

从未见初始化开始，完整policy只消费自身双RGB、8维state、exact language及自身动作生成latent。
自身state用冻结source normalization归一化并转成256-bin文本tokens，进入带State段的执行prompt；与teacher省略State的条件不同。
从新噪声开始做10步flow，得到50×7动作chunk，执行前5步后读取新观测并重新规划。
动作去归一化、相机处理、settling和各suite horizon延续官方评测合同。
生成的LoRA整段rollout固定；不在闭环中读teacher视频、改变S/M、调用第二adapter或做任务内梯度更新。

自身hidden h_l随环境反馈及前层完整适配改变。该层作用是B_V,l A_V,l h_l，
Q/V处先改变attention的query/value，经过后续网络再形成动作；不能把每个B列直接解释成一个已知技能。
教学X_β与自身h_full的分布仍可能不同；共用同一个A只保证同一线性映射，不保证语义已经对齐。

## 9. 训练：唯一完整FM目标与全部信用路径

在已授权、经审计的non-held任务集合中等权采样task。取一条teacher视频，
同task另取严格不同episode的自身query，重复使用同一次生成的完整LoRA监督多个query。
目标train24及登记的support任务范围、source normalization、Validation/Test隔离沿既有合同；本文不扩充allowlist。

query的动作a只参与query侧的真实FM：

    x_τ = τ ε + (1−τ)a
    y = ε−a
    L_full = E_task E_(V,q | task, different episodes) mean_(50×7) ||v_full(q,x_τ,τ) − y||².

实际接口先把7维动作pad为32维，再采样32维噪声；预测为50×32，loss只取前7个真实动作维。
50步真实动作窗口和末尾处理复用canonical数据消费者，不另改为只监督执行前5步。
teacher编译不使用query action/state、x_τ或对应hidden；τ<1含真实动作的FM hidden不能送回Writer，detach也不能消除泄漏。
每条query只算完整策略的一次FM预测；不是十步rollout训练，不再分公共分支。
梯度可跨query/microbatch累计，但须对应同一参数版本的LoRA与native，不能提前更新β再重放旧图。

直接执行路径训练A₀/B₀及生成的A/B；信用继续穿过M递推、S递推、生成头、解释器和公共native。
固定A₀/X/c/d并以S=A−A₀重参数化时，A侧合成信用包含两项：

    G_A = B_Vᵀ G_W + (D_A M_V)* [G_W A_Vᵀ].

第一项来自自身执行，第二项来自A改变M的生成，含key与δz两条依赖。
实际计算图若把δz直接接在S上，须把这条旁路的信用与经A返回的信用相加；之后还要继续反传X/c/d对β的依赖。
若使用现有cotangent replay工程形式，必须与完整图等价、保持同一参数版本，不把计算切分变成stop-gradient。
source基础权重始终不更新；训练共享参数为β、4层解释器及38组S/M头；S/M本身是本次前向结果，不是任务专属optimizer参数。

## 10. 初始化、计算与生命周期

全部共享可学习模块fresh初始化，optimizer/scheduler fresh，不载入成熟T或MT作分阶段课程。
A₀按合法非零初始化，B₀=0；O_A=O_B=0，其他投影非零初始化，动态路径不再叠加第二个零标量门。
初始S=M=0、完整LoRA是identity，policy等于冻结source。首步A/S及部分上游信用可为零；
B₀/O_B先接到完整误差后，后续信用才能流向A/S及解释器。这是初始化的计算结果，不是分阶段冻结策略。

推理成本为N次逐帧native、一次4层网格解释、一次S顺序编译和一次M顺序编译，全部在rollout前支付。
全网格attention有O((50N)²)配对；优先用不物化完整注意力矩阵的实现，长视频仍须按真实长度profile，不能声称已验证可行吞吐。
本稿不通过horizon均值、截断长视频、丢帧或改task权重来隐性降低成本。
训练可用activation checkpoint或同版本重放降低峰值内存；这会增加重算，不改变定义的特征和梯度。
X/H只在同次编译/更新内有效，38层S/M可按实际依赖分块执行；先得到最终A再编译对应M的顺序不能省略。
图文prefix可复用，teacher graph及生成器有额外反向成本，因此full-only也不能据此宣称总训练成本恰好减半。

## 11. 机制预期、历史约束和判定

本设计具体改变两条联系：动态内容可以从证据发生处传到后续写入位置；S能调整读取方向，并把改变传给M和自身执行。
前者解除本地差分的末端门控，后者开放原T固定A所不能直接适配的部分输入作用。
这是可表示与学习路径的判断，尚未证明合法视频提供了正确修改、主要历史缺口源于这些限制或闭环一定提高。

原T的有益视频作用应保留；公共辅助/Joint不能被解释为所有共同学习不可能；Context/self_read不能被改写成没有真实上下文或梯度。
LocalField已有X关联生成A，Pullback已有条件A与执行绑定，P/I已有公共与条件A/B；相关负证据继续降低无条件成功预期。
该候选的差别在可学习公共native、动态内容传播、A₀感知的S，以及新A→M→执行的共同信用，不在于第一次出现这些模块名。

实施前仍需实际消费者的图/shape/信息墙/identity与最长视频成本检查；这些不构成性能资格。
真实方法判断使用同预算同龄强参照、严格single-checkpoint paired400、任务/suite覆盖及相邻能力保持。
视频causal controls在selected checkpoint冻结后补充；Test、shuffled/reversed封存结果不反哺设计。
若同解释器和监督下重新训练的S=0取得相当或更好能力，应去掉S；成熟模型临时关S不是该对照。
若完整新机制不能保持匹配T能力，不以S非零、attention变化、FM下降或增加rank/深度维护它。
上述架构讨论本身没有产生实验结果；Owner随后授权的首批实施、科学读出与投入边界见§13。

## 12. 可核对来源

- [专家原文](../review_materials/20260930_t_architecture/EXPERT_RESPONSE.md)：Owner提供；审阅身份为3bb3ad1a。
- [原件证据与面板边界](../review_materials/20260930_t_architecture/EVIDENCE.md)：原T、Joint、Context、self_read。
- [机制分析§88–93](../analyses/feature_to_operator_mechanism_20260926.md)：条件A、因果索引、信用、公共目标的既有推导。
- [native](../../src/ember/operator_writer/native.py)、[model](../../src/ember/operator_writer/model.py)、[target registry](../../src/ember/pi05_lora.py)：共享真实读取/形状与历史原T算子；[新解释器及S/M编译](../../src/ember/operator_writer/conditional_read_write.py)由canonical model调用。
- [prompt与state处理](../../src/ember/pi05_processing.py)：teacher省略State、自身policy读取state的实际区别。
- [FM消费者](../../src/ember/writer/function_credit.py)：真实query功能监督及cotangent。
- [Owner要求](../current_owner_requirements.md)、[研究历史](../research_history.md)：当前边界与最近似反例。

## 13. 首批实施与完整学习检验（2026-10-01授权）

### 13.1 最终选择、问题与竞争解释

Owner要求按最终确定版本开始并由主讨论持续自主推进，次日检查；不需再次请示启动或正常接续。
最终保留本稿的S/M残差编译，接口直接输出A_V/B_V；不采用随后讨论的以B₀为初值、覆盖完整B的替代递推。
直接写B_V或B₀+M只是出口表达；本批始终从零编译M，再与B₀相加。唯一目标为完整FM，无public项。

所要解释的实际不足是：原T已有有益教学作用，但成熟完整161/400尚未形成相对强MT153的可靠广泛优势，
Context/Joint/self_read未解决主要能力缺口。当前候选检验完整动态内容传播与条件读取共同学习能否产生更有效的执行修正。
已有局部结构见证只证明原局部差分门控的限制和条件A的可表示方向，不证明它们是实际主要根因。

竞争解释至少保留：

- 动态证据必须在发生位置之外写入，且读取方向需要按教学改变；新图应改善完整能力及需要多阶段调用的任务。
- native特征中关键关系不足，或有限meta-task支持未教会可迁移编译；新增结构可以接通甚至拟合训练，而held仍不改善。
- 新容量/训练参数化改变了学习速度或已见任务拟合；早期分数差不能自动归因动态传播或S。

本批是Owner选定完整方法的一次联合检验，包含解释器、条件A和full-only取舍；与原T/Context的差额不能隔离任一模块因果效应。
保留LocalField/Pullback/P-I及原T/Joint/Context/self_read的完整正反证据，不把部件重组视为首次解决旧问题。

### 13.2 训练范围与事件

唯一运行根为`/data1/user/ymdai/ember_runs/conditional_read_write_fresh_20261001`，代码仍由canonical `operator_writer`拥有。
source1000、normalization、tokenizer、固定24/8/8协议及已审36task支持复用原合同，不扩充数据或改变task等权。
明确训练allowlist为
`[0,1,2,4,5,7,12,13,14,15,17,19,20,21,22,25,28,29,32,34,35,36,37,38,42,43,51,55,56,62,64,73,95,96,97,101]`；
前24项为target train24，后12项为既有经审non-held support。
事件复用`configs/operator_read_write_v1/self_conditioned_native_fresh_spec.json`中的events定义，
只继承数据/事件身份，不继承其self_read、双目标、旧模式或初始化权重。

首窗fresh450宏步，每步4个task、各28跨episode query，共50,400个完整FM query，
每task首轮50条teacher各访问一次；query action offset1、同task跨episode、全部50×7监督及flow随机流保持。
新解释器与两侧头全部fresh、公共LoRA合法identity、fresh optimizer/scheduler，不加载旧T/MT/Context/profile权重。
初始沿用AdamW lr3e-4、betas(.9,.95)、eps1e-8、weight decay1e-4、clip1、warmup150、decay1200、floor1e-5。
这固定比较所需的初始优化条件，不授权后续LR/seed/rank扫描，也不把旧训练时长当新架构上限。

90/180/270/360/450保留完整ECP；450是本批唯一科学评测节点，中间点只供恢复，不按loss/内部量挑选。
ECP保存新架构/schema、全部共享参数、optimizer/scheduler/scaler、sampler/cursor、rank RNG与拓扑。
旧checkpoint与新图不兼容，必须fresh；完整边界可按AGENTS显式迁移物理卡数，保持逻辑事件、有效batch和权重。
物理并发由真实吞吐决定，不把四个task或一次smoke的world size变成永久设备上限；无收益的rank不占卡。

### 13.3 实际图与成本检查

执行者在现有消费者内验证：真实38-target输出、信息墙、帧块因果、整段零动态时S=M=0、
转移(t−1,t)以X[t−1]寻址、先得到最终A再编译M、合法identity及同参数版本完整cotangent/replay。
实际梯度须包含A的执行、M的key/δz和native依赖；source基础参数冻结不等于切断其算子导数。
只做与新计算有明确关系的针对性检查，不为工程形式扩展通用测试平台或逐tensor低位一致检查。

在最长已授权train38/demo36的105采样帧、28跨episode query上完成最多3次fresh完整FM更新，
记录编译/FM/replay耗时、峰值、finite和必要的分组信用；第一步上游为零按identity解释，后续验证路径打开。
profile更新不进入正式初值或正式曝光。训练内存/吞吐可用正常BF16/TF32、高效attention、checkpoint与同版本重放优化，
须保持4层独立参数、完整50槽与真实视频长度、同帧全可见/跨帧因果mask及零保持动态内容；不能静默换成均值或局部门控。
GPU工程检查预算包含在本批总额中，预计不超过1GPUh；若计算定义无法按现有硬件执行，报告具体科学/成本取舍给main。
检查通过且总预算可执行后，实验session直接启动fresh450，不等待main代码复审、重复测试或二次放行。

### 13.4 450节点的最小科学读出

1. **完整correct validation400**：固定validation `[3,6,11,16,23,26,31,39]`、init0..49；
   复用`demonstration_transfer_learning_20260927/scenes/manifest.json`及原state–video/env/policy RNG，
   每task50条teacher各一次，固定schedule seed7，继承官方预处理/10flow/前5执行和long-first persistent动态队列。
   保存8个init0 full、392compact及全部success、goal/continuous/action/RNG。报告per-task/suite、breadth、
   对T450122、Context450126、self_read450116和强MT153的paired R/G/L、churn、Jaccard及whole-task不确定性。
   Joint450114、Context900151、成熟T2340161作为学习年龄/目标不同的完整背景；不隐去强点，不重评参照。
2. **完整train144**：36task×init32..35，使用同一450完整视频条件策略，复用既有训练诊断场景与teacher调度，
   明确有限池/训练重用范围，不能冒称held或整轮50视频无放回。36个init32 full、108compact、完整原行与goal保存。
   与同场景MT93原件配对，分别报告train24/support12；Context900完整103仅作不同年龄背景。
   本批不另做公共train144或held公共400。它检验训练侧获取，不能由该读数直接推断跨任务泛化。
3. **固定train-only A28功能读出**：沿用task0:40/11、12:25/14、20:38/42、32:17/43的8条teacher与
   原28query/noise/tau/target；保存8份完整和4份公共FM预测，报告full50/first5/motion6/gripper及逐query原件。
   公共只作冻结诊断，不产生梯度或选点。仅在这8条实际编译中被动保留H、最终c/d及
   Q8/V8/action_out的真实X、A₀/S/B₀/M和frame indices，供核对传播、读取变化与功能作用；
   不额外做层扫描、反向探针、teacher标签输入或新增query面板。

合计544条新环境episode、12份固定FM预测；工程profile单列。所有新GPU消费者含物化/诊断/评测共同计卡、计费。
本批不跑other/wrong/shuffled/reversed、Test、RL、task-local优化或S消融；视频controls待正式selected checkpoint冻结后登记。
不由一个新点宣布稳定性或视频机制确证；相邻checkpoint能力保持仍需后继明确合同。

### 13.5 结果分支、投入及职责

450是有信息量的首轮教学覆盖节点，不是预设的最终模型或普遍训练上限。
若完整训练获取与held能力、任务分布共同支持继续学习，main结合实测曲线与成本登记最小后继和相邻保持证据。
若仅内部量改变、训练/held能力均弱，先核实际干预及学习阶段，再依据竞争解释裁决；不自动900、无限续训或参数小扫。
若train获取明确而held仍弱，降低“结构接通就能迁移”的支持，不能只通过加深/加rank延长同一假设。
若出现已定位工程违约，执行者自行修复并保留有效原件；科学non-pass按科学证据处理，不修成预期答案。
停止本批不等于停止研究；main仍须按Owner长期授权形成并执行有依据的下一判断，不能等Owner催促。

首批硬限**40GPUh / 80GiB新增峰值**，包括profile、加载/保存、训练、物化、读出、失败/恢复及临时输出。
先验预计训练12–24GPUh、物化与544episode/功能读出4–8GPUh，整批含工程约6–12小时；
依据原Context450训练8.419GPUh、self_read整批约18GPUh及新增4层网格解释/S编译的成本不确定性，
不是新模型实测。profile后由执行者修订一次ETA；预计超12小时或累计30GPUh仍距收尾较远时向main报告具体剩余量。
超过40GPUh/80GiB前须由main作科学/预算裁决，不擅自缩短视频、减少query或修改评测口径。
所有新增data1；启动前由执行者live核strg01独立quota、目录实占、共享容量及完整峰值，复用canonical source/data/env。
每次launch/resume同时核gpu01/gpu02，按AGENTS总8/6、单节点6及NUMA/NCCL合同选择真实吞吐；不另加整批4卡硬上限。

唯一实验执行session仍为`01a0f018-69af-7b00-b614-7e117540051b`（接管并继续实验）。
它负责实现、针对性工程检查、资源准入/调度、运行排障、Git集成push和clean detached冻结、整批原件与计费交付；
main负责科学合同、原件解释、机制判断、后续实验及预算裁决，不默认接管工程或重复QA。
canonical tracked写入及Git集成由实际冲突方直接串行协调；窗口当前归属见progress，不向main例行广播。
代码保持一个canonical实现；历史执行面由Git/冻结版本/正式原件保留，不另起长期并行Writer或fallback。
正常长任务由执行者持续等待退出；工程交付、冻结、profile修复/重启、窗口释放及普通调度记入已有记录。
只在整批科学结果完成或实质权限/科学合同/预算边界、原件有效性未明且需裁决时回报main；自行可修复事项不暂停、不通知。
整批结果在main活跃时Steer、idle时Queue，只发一次；main没有独立有价值工作时可以结束回合，回报后主动验收与接续。
Owner对主讨论在无独立工作、已派发实验后的不设goal要求仅适用于该场景；不扩展为独立任务的禁止。
仓库整理session按Owner最新明确要求使用goal；不另起实验session，不使用自Queue维持主讨论运行。

# Policy-Response Event-to-Factor Writer

状态：active design  
owner确认：2026-09-02  
科学基线：main@7f5fefe134f7ebba2d09b906186eb5c140cccce9

## 1. 目标与裁决范围

本设计从exact language与一条或多条same-task、action-hidden、内部有序正确视频出发，在rollout前一次生成唯一38-target
rank16 LoRA。最终目标仍是validation8 strict paired correct稳定严格大于145/400，并同时满足相邻checkpoint稳定、低churn、
高breadth、四suite非零、Goal/Long贡献、same-task不同视频鲁棒性以及selected checkpoint冻结后的视频因果controls。

本设计保留ECP已验证的PI0.5原生观测、ordered events、真实native X/Y、signed pooling、rank4与唯一LoRA物化；停止把冻结
Natural Program送入专用Program--bank compiler。新的主要可学习路径只有Policy-Response Video Process Encoder与
Current-Video Native Factor Composer，两者由可重复attention/MLP blocks扩展。

本设计不恢复普通视频embedding到高维LoRA的任意hypernetwork，也不把task experts或历史LoRA做成部署字典。

## 2. 不可改变的科学合同

- 输入只有exact language与K条正确、action-hidden、内部有序teacher videos。
- teacher-video path不得读取state、proprio、teacher action、reward、terminal、task ID、filename、pose或policy outcome。
- source PI0.5、frozen rank12 carrier和首版Action Meta全部冻结。
- 每条视频独立保序编码；video index没有顺序，跨视频处理对排列等变，最终LoRA对排列不变。
- teacher-frame time、Action Expert horizon、flow time、layer depth和probe轴不得混淆。
- 当前视频的真实native X/Y是mobile factor的唯一原始vector value来源。
- language与静态context可以ground或调制query，但不能独立产生mobile factor。
- 每个condition只生成一套完整38-target rank16 LoRA；不平均多套video LoRA，不部署第二adapter或expert。
- validation/test不产生梯度。
- wrong、no-video与language-only不构造专门训练contrast；shuffled/reversed只在selected checkpoint冻结后评测。
- checkpoint选择只使用正确视频功能与相邻稳定性，内部loss、attention、factor cosine或Program差异不能替代closed loop。

## 3. 五个相互独立的轴

1. teacher-video frame time t：视频观察的真实先后与阶段推进。
2. Action Expert horizon h=0...49：同一静态观测和probe下的50个future-action response positions，不是teacher未来50帧。
3. flow time s：flow matching内部积分坐标；首版固定在s=1。
4. Action Expert depth ell：18层输入边界加final norm前状态，是网络拓扑而非物理时间。
5. antithetic probe p：同一frame上的固定正负suffix noise测量，不是两个视频或训练正负样本。

禁止使用t+h把不同teacher frames上的relative horizons映射到一个机器人绝对时钟。

## 4. Module 0：Frozen PI0.5 Evidence Capture

### 4.1 输入

对每个视频k、frame t：

- image(k,t)；
- exact language；
- fixed antithetic probes epsilon与-epsilon，单个probe形状为[50,32]；
- flow time固定为1。

prefix使用原生image-language路径且不含state/proprio。PI0.5与全部原始policy参数保持冻结。

### 4.2 冻结raw evidence

逐帧捕获：

- image patch hidden P(k,t)，256个patch positions；
- exact-language embedding与contextualized language hidden；
- H(k,t,p,ell,h)，19个Action Expert layer boundaries、2 probes、50 horizons；
- layer residual increments Delta-H(ell)=H(ell+1)-H(ell)；
- flow velocity V(k,t,p,h)；
- probe noise epsilon(p,h)；
- 38个LoRA targets的真实native input X与output Y。

raw tap capture与PI0.5 forward严格冻结。任何可训练compact projection归Module 1所有，不能把可训练target同时称为冻结
process-prediction teacher。

### 4.3 target-native bank

每个target j保留：

- X_j(k,t,p,h)；
- Y_j_abs(k,t,p,h)；
- Y_j_adj(t)=Y_j(t)-Y_j(t-1)；
- Y_j_init(t)=Y_j(t)-Y_j(0)；
- Y_j_goal(t)=Y_j(T-1)-Y_j(t)。

所有跨frame差分比较相同relative horizon h，不声称这些positions属于同一绝对控制时刻。bank按frame/target chunks流式读取，
不得为方便提前平均probe或horizon。

## 5. Module 1：Policy-Response Video Process Encoder

Module 1包含Frame Policy-Response Blocks和Ordered Event Blocks。两部分共同接受最终functional gradient。

### 5.1 compact channel ownership

raw layer state、layer residual、probe noise和flow velocity先分别投影到共享宽度d，首版以128为默认profile起点。它们作为不同
channel type保留，不使用旧Stage0的state + gate x residual提前相加。

两个probe不提前平均，而做可逆变换：

\[
Z_{even}=(Z_+ + Z_-)/2,\qquad Z_{odd}=(Z_+ - Z_-)/2.
\]

even与odd作为不同channel type继续进入frame-response block。odd表示有限probe幅度下的方向敏感性，不解释为严格Jacobian。

### 5.2 Frame Policy-Response Blocks

exact language query先读取当前frame patches，形成task-grounded视觉context。随后构造四类frame relation：

1. adjacent；
2. short-window previous；
3. initial-to-current；
4. current-to-final。

对每个frame、relation和owner形成query，第一次learned读取完整probe x horizon x channel evidence。每个重复block只包含：

1. horizon/probe/channel cross-attention；
2. owner-axis self-attention；
3. gated MLP；
4. pre-norm residual。

输出F(k,t,m,j)，其中m为四种relations。50-horizon只允许在这次task/relation-conditioned读取后压缩。

owner于2026-09-03进一步锁定：full policy-response是唯一active representation。final-layer horizon mean、coarse response或
任何等价的无条件horizon平滑均不得用于后续训练、选择、初始化或部署；既有实现与结果只保留为历史消融证据。

### 5.3 Ordered Event Blocks

每条视频独立把F编码为最多8个ordered event slots：

- E(k,e,j)；
- soft frame/relation assignment alpha(k,e,t,m)；
- occupancy rho(k,e)。

单调stay/forward、首尾boundary anchor、relative frame positions和mask属于硬结构；旧Natural Program的固定tuple、canonical code与
下游decoder不是硬接口。

relative frame position可以进入event emission、transition以及Event Block attention的Q/K，决定“事件在哪里”；它不能作为value直接
加进E，否则slot编号本身会在静态重复帧上伪造动态。slot query同样只路由单调posterior；用于聚合event value的四类relation权重在同一
frame内计算并对slot共享。这样完全相同的frame-policy-response重复成视频时，各slot得到同一content value，event centering后的D只剩
浮点舍入量；真实内容变化仍通过frame relations、posterior与可复制Event Blocks进入D。

每个视频与owner只做一次event centering：

\[
C_{k,j}=\operatorname{weighted\ mean}_e E_{k,e,j},
\qquad
D_{k,e,j}=E_{k,e,j}-C_{k,j}.
\]

C表示task、scene与共有过程context；D表示event-relative process innovation。多视频在Module 1后仍保持为无序集合，不先平均为单个
Program。每条视频的candidate base mass按1/K正规化，避免长视频或更多frames改变video权重。

## 6. Module 2：Current-Video Native Factor Composer

### 6.1 输入与查询

Composer读取：

- 每条视频的E、C、D、alpha和rho；
- exact language，仅作query或FiLM条件；
- 当前视频真实X与四类Y；
- frame、probe、horizon、relation、owner、layer和family metadata。

首版使用38 targets x 4 mobile ranks，共152个target-rank queries。固定rank12 carrier不需要query。

### 6.2 可扩展block

每个Bank Context Block包含：

1. target-rank query读取同target ordered event innovations；
2. query读取当前target bank的低维projected candidate context；
3. rank-axis self-attention；
4. gated MLP。

能力扩展通过复制同一block、增加hidden width或增加授权meta-task mappings完成，不增加summary、covariance solve、whitening、
transport、anchor或family scalar gate。

第一遍bank read只形成低维whole-bank context；第二遍产生最终candidate logits并流式对raw X/Y做exact pooling。

### 6.3 signed native pooling

2026-09-03首个corrected full 12-task shared相邻closed-loop稳定为`35/250`，而task-local Composer继续有容量。逐层核对发现
Process虽然产出并声明传递`alpha(k,e,t,m)`，Composer实际只读取四类relation已经混合后的单个`frame_innovation(k,t,j)`；
`assignment`与relation type从未进入bank candidate scoring。这与专家明确要求Composer读取event assignment和relation type不一致，
也会让“同一动作、对象和goal，只改变初始scene relation”的task-disjoint组合在进入signed pooling前失去显式绑定。

下一matched实现因此只修正这一处最早接口。对每个frame、relation和owner直接从既有event输出构造：

\[
I_{ktmj}=\sum_e \alpha_{ketm}D_{kej}.
\]

relation type通过一个共享learned projection/FiLM进入该innovation；每个raw native candidate在logit语义上保留relation轴，
但其X/Y value不复制成新的向量来源。四类relation共同按固定`1/4` base mass归一，最终仍由exact online softmax对全部
frame x relation x probe x horizon x bank-type candidates归约。该改动不增加Program、summary、solve、anchor或新loss，
不平均或抽样50-horizon，也不改变rank、carrier、materializer或部署输入。

对target j、mobile rank r、relation m和native candidate n，A侧与B侧分别预测两组softmax logits。输出为：

\[
a_{jr}=\sum_{m,n}(w^{A,+}_{jrmn}-w^{A,-}_{jrmn})X_{jn},
\]

\[
b_{jrg}=\sum_{m,n}(w^{B,+}_{jrgmn}-w^{B,-}_{jrgmn})Y_{jng}.
\]

signed pooling的正负softmax branches与输入antithetic probes是不同轴；每个softmax branch都可以读取两个probe产生的candidates。

native output grouping固定为：

- q：8个256D groups；
- v：1个256D group；
- action-in：32个32D blocks；
- action-out：1个32D group。

实现使用一个ragged native-group operator和family masks，不复制四条compiler。

### 6.4 幅度与固定边界

每个target只使用一个由fit19、task-equal expert-minus-carrier effective-update RMS得到的冻结全局`s_ref`；网络只预测
`tanh`有界的相对rank gains，不使用task-specific scale表。四个rank合成后的完整mobile update为

\[
\Delta W_j=B_j^\top A_j.
\]

在factor输出边界统一施加

\[
\gamma_j=\min\left(1,\frac{s_{ref,j}}{\operatorname{RMS}(\Delta W_j)}\right),
\qquad B_j\leftarrow\gamma_j B_j.
\]

该边界约束完整rank4更新，而不是分别约束四个可能相互对齐的rank；它不读取held outcome，不是loss，也不增加SVD、projection或
transport。后续small-core canonicalization保持同一有效更新不变。专家明确规定了每target effective-update RMS cap，但没有给出
额外倍率；首版采用`1 x s_ref`，因为fit-only shared rank template的最大比值为`.846`，通过G1的held5中`185/190`个
task-target不超过该值，而未设完整边界的shared macro610已有`94/190`个超过、最大`2.243`。

### 6.5 视频必要性参数化

两条signed logits共享common base b，只有bias-free D路径产生branch-specific offset：

\[
\ell^\pm=b+\delta^\pm(D).
\]

当全部event innovations为零时，delta+与delta-都严格为零，两组权重相同，mobile residual为零。language、C和bank context可以决定
关注什么，但不能在没有D时独立产生mobile update。relation embedding只能调制由D与alpha共同形成的innovation，不能添加独立
branch bias，因此显式relation轴也不能成为static旁路。static-repeat只作这一结构不变量的无梯度实现检查，不进入训练、loss或
checkpoint选择。

这只是排除最直接language-only旁路，不保证正确视频自然优于wrong或shuffled。最终因果性仍由冻结后的closed-loop controls裁决。

首版不使用free learned residual。只有task-local current-bank composer已充分优化、仍稳定低于同task direct-factor正控，且差距跨
至少两个suite并包含Goal或Long时，才重新审查event-conditioned residual basis。

## 7. Module 3：Canonical Materializer

Composer为每个target输出rank4 A/B factors。每个target只执行一次现有small-core balanced canonicalization：

- QR分解A/B；
- 对4x4 small core做SVD；
- 平衡A/B scale；
- 以A最大绝对pivot固定符号；
- 保持有效BA不变。

随后与frozen task-independent rank12 carrier直接concat，形成唯一38-target rank16 LoRA。中间不做额外SVD、projection、
transport或factor reconstruction。

## 8. Positive-only训练目标

主目标为：

\[
L=L_{func}+L_{process}+\lambda_{pres}L_{pres}.
\]

### 8.1 Correct cross-episode functional loss

由正确视频episode A生成LoRA，在同task独立episode B的observations/actions或授权task-expert policy response上计算真实PI0.5
functional loss。generated LoRA必须实际安装到frozen policy；不拟合teacher A/B，不以parameter cosine为目标。

### 8.2 Causal policy-response prediction

从截至frame t的prefix-only表示预测冻结PI0.5在未来frame t+delta的policy-response变化。合同为：

- target使用冻结raw evidence或固定teacher projection；
- predictor侧的learned projection不能同时充当可漂移target；
- causal auxiliary forward不得读取t之后的frames、current-to-final relation、future event assignment或target interval；
- 预测保留38 owners、50 horizons和2 probes；
- 不读取teacher actions，不制造wrong、shuffle或reverse样本。

主Writer在部署时可以读取完整视频；prefix-only约束只属于该causal auxiliary view。

### 8.3 Preservation

在carrier/source已有正功能的训练states上使用轻量单侧preservation bound，防止mobile residual无故破坏已有能力。它不能压制真实
task-specific增量，也不要求generated LoRA接近carrier。

三项loss在首个optimizer step前固定无量纲化：每task的`L_func`与`L_pres`使用该task冻结Panel-A carrier loss的RMS；`L_process`
使用同步后的初始Writer在两条fit视频固定prefix上的平均causal loss。formal run将这些数值一次性冻结到`normalizers.json`，resume复用而不
随训练漂移。无量纲后的`L_process`系数固定为`1.0`，`lambda_pres`固定为`.05`，不做weight/LR小扫。

G2已有positive temporal heads只允许作为component-init短暂辅助，并在functional优化稳定后退火到零；它们不进入最终Writer forward。

### 8.4 scale与方向的优化预算

shared训练保持同一optimizer、LR与loss权重，但将`scale_head`和其余全部Writer参数分别按既有norm `1.0`裁剪。该分组不改变
objective或task权重；它只防止幅度梯度占用同一个global clip后持续缩小Frame、Event和Composer方向更新。未分组的73-task
macro610轨迹中global clip触发率为`.8781`，scale norm中位`2.5992`而其余方向norm中位`.5839`；若独立使用同一边界，方向侧仅
`.0386`的step需要裁剪，方向更新倍率中位可恢复`2.6533 x`。

## 9. 首轮实现与证据顺序

### 9.1 最小真实smoke

接通后只验证：

- 真实PI0.5 forward与19 x 50 x 2 taps；
- no state/proprio/action进入Writer；
- L_func梯度到达Frame、Event与Composer；
- L_process使用冻结target且无未来泄露；
- relative frame positions只路由Event segmentation/QK，不进入event value；static-repeat只返回carrier；
- K1和K4、video permutation invariance；
- chunked与one-chunk有效BA一致；
- 38 targets、76 tensors及唯一rank16被policy真实消费；
- source、carrier与Action Meta冻结。

### 9.2 task-local Composer正控

在task1与task93的当前视频bank上，用task-local event-conditioned Composer验证去掉PNBTT solve后仍可接近G1可行功能。该正控与实现
smoke一起完成，不作为文档门槛。若G1 free logits仍强而新Composer task-local接近零，最早失败接口就是Composer，不应直接解释为
shared representation或数据问题。

### 9.3 shared实验

复用J2的gradient/true-task-held划分、cross-episode Panel A/B和两条fit加一条same-task held video。唯一active输入为
full-response：owner对应layer input与residual、38 owners、50 horizons、2 probes、probe noise和flow velocity，且完整horizon证据
保留到task/relation-conditioned attention。旧coarse matched arm已经完成历史定位，此后不得重启或用于模型选择。

首轮使用component initialization、10 warmup加100 effective updates，并保存相邻节点。训练图有效且出现有意义correct功能信号后，
立即运行held5 correct-only strict250，不以一长串内部小数阈值阻塞闭环。

首轮12-task资格实验采用每步3 meta加3 target只是该次配置，不是Writer或owner的固定要求。后续运行可按证据只使用其中一类、采用
任意显式比例并改变每步task数。科学sampler先确定task group与权重；多卡执行器随后按视频帧数与functional rows的
outcome-independent成本，在已有frozen evidence cache副本上最小化最长rank耗时。选择性cache复制和动态设备放置不得增删task、改变
task权重、K、optimizer cadence或完整50-horizon内容，formal resume冻结其执行计划。

checkpoint依据correct-only表现、task breadth与相邻稳定性选定。选定并冻结后才一次性运行same-task-other、wrong、no-video、
language-only、first+final、shuffled与reversed controls；这些结果不回流训练或loss设计。

## 10. 后续扩展与Final

首轮shared信号成立后：

1. 混合K属于1、2、4的真实训练；
2. 完成component-init与同拓扑fully-random fresh matched比较；
3. 扩大到train24与经审计的non-held LIBERO meta tasks，保持task或显式role等权；
4. 在方法稳定后运行validation8相邻single-checkpoint strict paired400；
5. selected checkpoint冻结后补齐完整因果controls；
6. 只有base Writer已有稳定闭环增量、剩余错误明确集中在action control detail时才评估Action Meta。

模型扩展优先复制Frame/Event/Bank blocks、增加有效meta-task mappings和提高吞吐；不得以LR、seed、width、rank、scale小扫替代接口
判断。

## 11. 失败定位与停止边界

- process prediction也失败：检查frozen target、frame-response或event表示。
- process prediction好而functional fit低：Composer、functional credit或D-path使用失败。
- task-local Composer强而shared低：shared mapping、task组合覆盖或positive-only可辨识性失败，bank support未失败。
- train高、same-task held低：video-specific overfit。
- gradient tasks高、true task-held低：task-disjoint组合泛化失败。
- full的task-local容量强而shared task-held弱：检查relation-conditioned horizon attention的内容/位置利用、process-to-Composer
  credit与task-disjoint组合泛化；修正必须继续保留全部50-step horizon，不能以均值或coarse规避。
- correct高但冻结wrong也高：positive distribution未形成bank/task specificity，不能用wrong loss修补。
- correct高但shuffle/reverse更好：时序语义错误，不能将control改成训练标签。
- 相邻checkpoint持续换手：不能用单点峰值宣称成功。

一次non-pass只淘汰实际检验的组合。当前函数类只有在task-local支持、shared正功能、matched前端、K、component/random joint及合理规模
训练均获得充分证据后仍系统失败，才停止；EMBER总体停止需要进一步证明zero-interaction static-LoRA合同或数据可辨识性本身构成根本
限制。

## 12. 工程与生命周期

- 新Writer保持一个canonical训练/评测入口；合法差异放入同一配置或窄strategy边界，不复制runner。
- active source ownership固定为：`capture.py`只拥有冻结PI0.5/native taps；`process.py`只拥有Frame/Event表示与causal target；
  `composer.py`只拥有current-bank signed factor生成；`model.py`只拥有组合与唯一rank16物化；`training.py`拥有共同asset/data runtime与
  唯一CLI dispatch；`tasklocal.py`/`tasklocal_contract.py`拥有task-local正控；`shared_schedule.py`拥有可扩展task/video采样；
  `shared_execution.py`只拥有outcome-independent cache复制与task-to-rank放置；`shared.py`拥有evidence cache与shared
  orchestration；`shared_training.py`只拥有多卡positive-only optimizer steps；
  `shared_evaluation.py`只拥有零梯度Panel-B评估；
  `shared_contract.py`只拥有formal authority与resume合同；`materialization.py`只把冻结shared checkpoint与固定正确视频物化成
  每task唯一完整rank16 adapter。唯一Writer CLI仍是`scripts/train_ecp_policy_response_writer.py`；closed-loop继续复用通用
  `scripts/evaluate_pi05.py`及其static task-LoRA bank、dynamic queue与paired rollout，不复制Evaluator。这些文件不是平行Writer、
  平行Evaluator或版本化fallback。
- 复用observer hooks、native X/Y capture、events、chunked replay、materializer、J2 data/functional infrastructure与evaluator。
- PNBTT、EBSRI、旧G3与Natural Program完整实现只保留为历史复现和kernel来源，不得作为active fallback。
- 当前不删除这些tracked历史实现；待新Writer完成matched shared裁决并冻结论文方法后，才审计删除无独占复现价值且已由Git/formal
  artifacts覆盖的旧可执行入口。此前它们不能被新runtime导入为备选Writer路径。
- shared运行面拆分为schedule/cache、optimizer、只读评估和authority四个owner，是因为多卡训练、Panel-B信息墙与exact-resume各自已有
  独立生命周期；模型数学图仍只有Process与Composer两个可复制模块。该拆分避免把约千行distributed/evaluation代码重新堆进一个
  runtime，也不增加部署模块或数学变换。
- 新active source按Frozen Capture、Video Process、Factor Composer、Training Runtime四项责任组织；避免继续增长现有超大runtime文件。
- 首个真实profile按LoRA/s、最长视频稳定性、GPU利用率和峰值显存选择frame/target microbatch。
- 每次GPU launch前同时live检查gpu01与gpu02，使用1至6张真正提高吞吐的A40，可在不干扰他人的前提下安全共驻。
- formal训练来自clean pushed detached worktree；等待训练时优先完成cache、分析、评测准备和下一科学节点，只有没有实质性推进工作时
  才做可随时中断的增量workspace清理，结果一到立即返回科学推进。

## 13. Authority

本设计由以下证据与owner裁决共同形成：

- docs/expert_review_20260902_full_history_policy_native_meta_writer.md；
- docs/expert_review_20260902_policy_response_event_to_factor_writer_clarification.md；
- docs/expert_review_20260824_native_factor.md；
- docs/research_history.md中的G1、G2、G3、J2/R、EBSRI、Program-through-bank与PNBTT formal结果；
- owner于2026-09-02确认整体设计并授权建立goal后立即推进。

若本文与owner后续明确表达冲突，以owner最新表达为准。

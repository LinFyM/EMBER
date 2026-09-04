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

首尾anchor只属于一条真实完整视频：deployment/full-video posterior继续硬锚真实第一帧与真实最后帧。causal prediction训练使用的
`video[:t]`只是人工观察prefix，不是另一条在`t`处自然结束的视频；其posterior从第一帧anchor开始做标准monotone forward filtering，
不把当前cutoff硬锚为最终slot。这样不读取未来frame、future assignment或current-to-final relation，同时让prefix slot与完整视频
slot保持同一过程语义。把每个cutoff都硬当最终帧会系统性把当前状态别名成slot7，不能算作G2 boundary anchor的合法复用。

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

typed-boundary资格进一步证明，仅在Composer query入口保留source所有权还不够：`D`虽有task-specific内容，其RMS约
`.04--.06`，而`C`约`5`。因此下一fresh在所有直接消费Process common/innovation的边界使用同一个无参数合同。令

\[
\mathcal N(x)=\operatorname{LN}_0(x)\frac{\operatorname{Var}(x)}{\operatorname{Var}(x)+10^{-5}},\qquad
M_{k,e,j}=W_M[\mathcal N(C_{k,j}),\mathcal N(D_{k,e,j})],
\]

其中event memory读取`M`，signed dynamic scorer读取`\mathcal N(D)`，causal predictor的prefix state见8.2。`C,D`与原来的
`E=C+D,D`信息等价，但不再重复注入大尺度`C`。方差可靠度项与LayerNorm使用同一个固定epsilon：真实`D`的RMS约`.04--.06`
时系数约`.994--.997`，而static-repeat约`7e-8`的浮点残差被平滑压回零；它没有affine、新参数或硬阈值，避免普通LayerNorm反而
放大roundoff。该修正不改变event assignment、完整50-horizon bank、X/Y、signed pooling、rank或LoRA幅度规则。

## 6. Module 2：Current-Video Native Factor Composer

### 6.1 输入与查询

Composer读取：

- 每条视频的E、C、D、alpha和rho；
- exact language，仅作query或FiLM条件；
- 当前视频真实X与四类Y；
- frame、probe、horizon、relation、owner、layer和family metadata。

首版使用38 targets x 4 mobile ranks，共152个target-rank queries。固定rank12 carrier不需要query。

target-rank identity、target结构身份与共享task context必须在进入首个Bank Context Block前保持独立数值所有权。定义

\[
s_j=\frac{\operatorname{LN}_0(o_j)+\operatorname{LN}_0(f_{\phi(j)})}{\sqrt 2},
\]

\[
c_j=\frac{s_j+\operatorname{LN}_0(C_j)+\operatorname{LN}_0(L_j)}{\sqrt 3},
\qquad
q^{(0)}_{jr}=\operatorname{LN}_0(q_r+q^{local}_{jr})+c_j.
\]

其中`o_j`与`f_phi(j)`分别是owner与family结构source，`C_j`与`L_j`分别是Process common与exact-language source；`LN_0`
是无可训练affine的parameter-free LayerNorm，shared模式没有`q_local`。`sqrt(2)`与`sqrt(3)`只保持各残差组的初始化方差，
不学习或人为指定任一task/family的输出幅度。分源pre-norm只防止合法但大范数的source在相加前抹掉其它type；它不要求输出rank
正交、非零或等权，也不增加rank loss、entropy Gate、solve、transport或新参数。若正确视频证据只需要一个rank，后续attention
与signed pooling仍可自发产生低秩结果。

这一接口来自直接失败定位，而不是结构偏好：73-task旧实现把范数约`1`的rank token直接加到范数约`67--69`的Process common
state上，m200与m400进入Composer的rank相对差异都只有约`1.1%`，经过两个block后仍未恢复；物化的q、v、action-in与
action-out mobile update中位参与秩均约为`1.00`。同一冻结m200权重的分源归一化反事实把block后rank差异恢复到约`63%`，
并把rank-conditioned signed posterior明显拉开。该修正保持完整frame x probe x 50-horizon x bank-type read和所有
positive-only训练合同不变；最终资格仍只由fresh训练后的closed-loop决定。

m400进一步暴露同一类、但不等同于rank轴的typed-source问题：Process common范数约`70--74`而language约`11.35`；Stage0复用的
owner约`0.99`又被family约`11.66`淹没。三条Object/Goal/Long correct视频上，matching target/rank跨task cosine仍为
`.99925--.99952`，同family q/v的18个target在两个block后区分比仅`.03713/.04595`。冻结反事实把owner/family/common/language
分别pre-norm后，跨task区分比提高约`1.6--1.9x`，q/v跨target区分比提高到`.18353/.20492`，且三个task近乎逐位复现。
因此当前公式是对同一个边界所有权错误的完整修复，不是为了追内部指标追加一串特征变换。

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

首个matched修正曾先构造`I(k,t,m,j)=sum_e alpha(k,e,t,m)D(k,e,j)`，再对relation候选做非线性打分。该版本虽然恢复了
relation轴，却仍在打分前消掉event轴；12-task macro70/110的held5 correct-only结果为`42/34`，breadth均为`3/5`、
Goal/Long均为0，并且两个true-task-held在macro110继续为负，因此该具体“先求event期望、后打分”接口正式non-pass。

当前matched实现只修正这个仍然最早的接口：把soft assignment本身作为候选base measure，而不是先乘入D并求和。对video k、
event e、frame t、relation m、target j、mobile rank r和raw native candidate n，动态logit为：

\[
\ell^{\pm}_{kretmjn}
=b_{krjtn}+q^{\pm}_{krj}\cdot
\left(W_DD_{kej}\odot s_m\odot \tanh K_{jtn}\right),
\]

对应的固定base mass为：

\[
\log \mu_{ketmn}=\log \alpha_{ketm}-\log K-\log T_k-\log P-\log H
\]

（output bank另含其真实bank-type基数）。`alpha`已经在每个frame对event x relation归一，因此不再额外除以event数或固定
`1/4` relation质量。raw X/Y只在逻辑上作为event x relation候选重复，仍是唯一vector value来源；relation type只能乘性调制
bias-free D路径。该改动不增加Program、summary、solve、anchor或新loss，不平均或抽样50-horizon，也不改变rank、carrier、
materializer或部署输入。

实现按relation顺序计算并对event做`logsumexp`，再用`logaddexp`合并relations；有梯度路径重算scorer激活。它与显式展开全部
event x relation x raw native candidates严格等价，只避免同时驻留大激活，不是event、relation、frame或horizon的平均、抽样或近似。

对target j、mobile rank r和扩展候选`c=(k,e,t,m,n)`，A侧与B侧分别预测两组softmax logits。输出为：

\[
a_{jr}=\sum_c(w^{A,+}_{jrc}-w^{A,-}_{jrc})X_{jn(c)},
\]

\[
b_{jrg}=\sum_c(w^{B,+}_{jrgc}-w^{B,-}_{jrgc})Y_{jn(c)g}.
\]

signed pooling的正负softmax branches与输入antithetic probes是不同轴；每个softmax branch都可以读取两个probe产生的candidates。

native output grouping固定为：

- q：8个256D groups；
- v：1个256D group；
- action-in：32个32D blocks；
- action-out：1个32D group。

实现使用一个ragged native-group operator和family masks，不复制四条compiler。

### 6.4 幅度与固定边界

每个target只使用一个由fit19、task-equal expert-minus-carrier effective-update RMS得到的冻结全局`s_ref`。每个rank/group仍有
独立relative gain，但当前函数不再让195个target-native groups各自拥有一个只读whole-target query的输出row。对signed pooling
得到并按最终物化口径规范化的当前因子`a_jr`与`b_jrg`，使用只按真实native width共享的输入/输出encoder构造token：

\[
z_{jrg}=[q_{jr},E_X(a_{jr}),E_Y(b_{jrg}),e_g],
\]

\[
h^{(0)}_{j}=\operatorname{reshape}_{r,g}\{W_z z_{jrg}\},
\]

\[
u^{(l)}_j=h^{(l)}_j+
\operatorname{MHA}_l(\operatorname{LN}(h^{(l)}_j)),\qquad
h^{(l+1)}_j=\operatorname{GatedMLP}_l(u^{(l)}_j),
\]

\[
g_{jrg}=s_{ref,j}\tanh(w^\top\operatorname{LN}(h^{(L)}_{jrg})+b).
\]

`E_X`只按1024/2048等真实input width共享，`E_Y`只按32/256 native block width共享；同一个fusion、可复制的标准
self-attention + GatedMLP blocks和唯一scalar output应用于所有target。attention scope严格限制为一个target内部的
`4 ranks × ragged groups`，不做跨target通信；q/action-in可比较rank与group的相对组合，v/action-out至少可比较四个rank。
target/layer扩展因此增加token而不是新增私有输出row。`q_jr`仍携带target、rank、exact language与完整Policy-Response时序
context，`a/b`则把当前视频实际选出的signed native factor直接交给幅度决策。该readout只能缩放同一个当前视频X/Y已经产生的
B子向量，不能独立写出方向或adapter，也没有task-specific scale表、anchor、第二Writer或负样本输入。

这一变更来自冻结Process后的直接反证。前一195-row query-only实现从fresh component initialization只训练Composer，optimizer
50/100的held5 correct-only strict250为`39/43`，后一点仅与carrier持平，Goal/Long仍为0；gradient tasks的Panel-B继续改善，
true-held task74继续为负，故联合Process干扰不是充分解释。跨task最终mobile update已经高度task-specific，但raw group gain
cosine仍为`.99973`、task-specific fraction仅约`.013`。更直接地，在同一m100方向上task2与task74所需的exact group-logit下降
方向总体cosine为`-.585`，v/action-out分别为`-.521/-.623`，而实际gain cosine为`.9991`；task74只调现有rank/group gain的
10% logit-norm局部可用下降约`.00193`，远大于当前方向的`-.000094`。首个pointwise factor-conditioned版本虽然读取了这些
token，却在m50/m100只得到`40/44`，后一点相对carrier43为`35 retained/9 gained/8 lost`且Goal/Long仍为0；actual gain仍跨task
近一致。六个bridge task又显示，task74与task73所需logit变化cosine为`+.590`，同一pointwise readout的参数梯度却为`-.418`，
说明问题已从“factor不可见”推进到“独立token Jacobian不能表达可迁移的相对协调”。因此当前最早缺口是readout缺少同target
rank/group集合的相对坐标，而不是signed bank没有候选方向、cap普遍截断、Process漂移或训练未移动。

首版误把专家§7.5的`rank/group gains`缩成每family一个rank gain。causal-filter m100冻结方向上的正确视频task-local反事实显示，
自由rank gain在task1/72/75/93的fit/held恢复只有`.151/.093`、`.166/.147`、`.122/.116`、`.150/.099`；恢复真实group gain后
分别升至`.244/.198`、`.222/.189`、`.146/.126`、`.240/.202`，全部第三条视频仍优于carrier。因此group ownership是已证实的
函数接口遗漏；它带来约`1.2--2.1x`恢复，但仍远低于G1，不能单独解释为最终修复。

gain scalar output初始化固定为`w=0,b=0.1`，直接复用G1 `TaskLocalNativeFactorOracle`的已存在选择，不做数值扫描。这样初始
所有group仍严格为同一小幅logit；第一次functional backward给共享output weight条件化信用，同时非零gain让梯度到达signed
direction，第二步后native factor encoders与GatedMLP开始共同学习。首版严格零初始化曾导致第一次
functional backward只有gain head有梯度，随后direction梯度又始终乘以很小的gain；causal-filter 100步中Composer总梯度几乎由
gain head占据，Composer方向权重相对移动只有低千分量级。非零小幅启动仍受完整target BA cap与preservation保护，并且当event
innovation为零时positive/negative weights仍完全相同，mobile residual仍为零；它只让正确视频functional credit从第一步到达
Frame/Event/Composer direction。

四个rank合成后的完整mobile update为

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

\[
S_{k,t,j}=\frac{\mathcal N(C_{k,j})+\mathcal N(I_{k,t,j})}{\sqrt 2},
\]

其中`I_{k,t,j}`是只由不晚于`t`的frames形成的prefix innovation。固定系数只保持两条source的方差，不学习或人工规定预测幅度。

- target使用冻结raw evidence或固定teacher projection；
- predictor侧的learned projection不能同时充当可漂移target；
- causal auxiliary forward不得读取t之后的frame content、current-to-final relation、future event assignment或中间target interval；
- 预测保留38 owners、50 horizons和2 probes；
- 不读取teacher actions，不制造wrong、shuffle或reverse样本。

主Writer在部署时可以读取完整视频；prefix-only约束只属于该causal auxiliary view。

首版实现曾把上述`delta>0`静默固定成`future_offset=1`。`f33f2955`的consumer-boundary资格与后续multi-gap诊断已经证明，固定相邻
目标主要是高频、弱可预测变化：在同一m100 Process state上，保持完整38x50x2 target时，within-video held-cutoff的最优尺度MSE
解释量从delta1的`.0094`升到delta2/4/8的`.1382/.3268/.4718`，跨task双向均值也从约`.0093`升到约`.0752`。
因此下一matched fresh恢复专家原合同：先从所有合法prefix endpoints中确定性均匀采样`t`，再从该prefix剩余future offsets中
确定性均匀采样`delta>0`；采样只依赖optimizer step、task与video demo，不读取outcome。predictor只额外接收delta的标准无参数
sinusoidal interval encoding，不读取future frame、路径中间帧或future event assignment。target继续使用现有固定teacher projection，
prediction与target同时除以`sqrt(delta)`后计算loss，以抵消实测近似random-walk的跨interval方差增长；该已知delta缩放可逆，不改变
方向、owner、probe或horizon信息。loss权重与固定normalizer合同不变；不平均50-step Action Expert horizon，也不改变主Writer
deployment forward。

该matched变更由`38d51bab`实现；31项定向测试和task1真实full-horizon delta8 smoke通过，且主functional loss、梯度与输出
和前代smoke保持逐项一致。它只修复causal auxiliary的interval监督合同，不借机改变deployment函数类。

clean detached random-delta formal已经完成optimizer50/100与两点held5 correct-only strict250；闭环均为`41/250`，逐task均为
Long/Goal/Object/Spatial0/Spatial9=`0/0/4/33/4`、breadth`3/5`。它相对前代m100的`35/250`有恢复，但仍低于carrier
`43/250`，两个相邻checkpoint换手`8 gained/8 lost`，且Goal/Long仍为0，因此该训练实例稳定non-pass，不运行negative controls。
更早的机制检查表明，它并没有真正完成本节赋予的时序职责：m100预测头对多任务正确视频只比零预测改善约`.35%`，delta1仍劣于
零预测；m50到m100预测头权重只移动约`.1%`。

冻结同一m100 Process state后的两个正样本诊断进一步区分了容量与优化。只训练已有预测头时，直接预测标准化target在100步的
train/同视频未见pair/同task未见video MSE解释量为`.540/.302/.144`，250步为`.833/.389/.217`，证明现有状态与readout并非零容量；
多间隔线性probe也在delta4/8上给出同视频`.278/.410`与跨task约`.043/.074`的方向信息。正式训练没有得到这些信号的两个直接原因是：

1. 网络先输出原尺度prediction、再在loss外除以`sqrt(delta)`，使长间隔的prediction反向梯度额外衰减；
2. 每task normalizer只用两条fit视频各一个随机pair，73 task的inverse-weight有效数仅约`37.2`，top5偶然占约`24.2%`；实际
   16--17条训练pair形成的稳定代理则约为`64`个有效task。

提交`df1e8c6e`的matched修正因此让predictor直接输出`Delta Z / sqrt(delta)`，target仍是同一个可逆标准化冻结量；每task normalizer改为两条fit
视频各8个确定性随机pair上的zero-predictor target Huber均值，不再读取可训练Writer状态。辅助readout的独立学习率固定为主Writer的
`20x`：100步余弦调度的累计步长由容量实验换算为约100个`1e-3`常学习率head-only更新，不是LR sweep；Frame/Event/Composer仍使用
原学习率。固定teacher、主deployment forward、functional/process权重、数据、完整50-horizon与所有factor合同不变。若这次让
process prediction成立而闭环仍失败，下一定位才转向Process-to-Composer credit或task-disjoint mapping，而不是继续改预测优化。

该process-conditioned fresh已完成。m50/m100 held5均为`37/250`且Goal/Long为0，但m100在六task、fit+held共108个正确pair上的
标准化prediction Smooth-L1比zero改善`6.49%`；zero-state prediction反而比zero更差，说明视频Process state真实承担了预测。
同时发现causal prefix仍复用full-video hard-final posterior：任意cutoff当前帧`108/108`都成为slot7，同一帧在完整视频只有
`15/108`如此，assignment重合仅`.137`。target侧Process梯度为functional的`3.73x`，Event子集方向cosine为`-.204`，足以让错误
prefix坐标压过功能credit。冻结反事实改用first-anchored forward filter后重合升到`.692`，故下一matched fresh只修这一posterior
语义；完整视频首尾anchor、teacher、loss权重、head优化与主deployment图全部不变。

该matched修正已由`f6b58aac`实现：`causal=False`完整保留原前向--后向posterior，只有`causal=True`在同一Event Encoder内直接读取
forward-filter posterior；没有新增网络、参数、并行实现或兼容fallback。41项定向及materialization相邻测试通过。task1 demo5真实
smoke继续得到与前代完全相同的functional loss `.150360`，并保留Frame/Event/Composer功能梯度、Frame/Event/Predictor过程梯度、
完整50 horizon、38 targets、76 tensors与唯一rank16。实现图已有资格进入同规模optimizer50/100 fresh裁决。

固定teacher的矩阵rowspace只有约`12.4--12.5%`落在ResponseTokenizer projection可见子空间，但事后把teacher对齐到该子空间只让
delta1跨task解释量达到约`1--2%`，没有单独形成足够修复证据。为保持一次只改变一个主要因果变量，本轮仍不同时更换teacher。

### 8.3 Preservation

在carrier/source已有正功能的训练states上使用轻量单侧preservation bound，防止mobile residual无故破坏已有能力。它不能压制真实
task-specific增量，也不要求generated LoRA接近carrier。

三项loss在首个optimizer step前固定无量纲化：每task的`L_func`与`L_pres`使用该task冻结Panel-A carrier loss的RMS；`L_process`
使用两条fit视频各8个确定性随机合法pair的冻结标准化target相对zero predictor的Huber均值。它不运行Process forward，也不依赖
Writer初始化或可训练prediction。formal run将这些数值一次性冻结到`normalizers.json`，resume复用而不随训练漂移。无量纲后的
`L_process`系数固定为`1.0`，`lambda_pres`固定为`.05`；只有纯辅助prediction probe/horizon/head使用上述有容量实测依据的`20x`
参数组学习率，主Writer学习率与调度不变。

G2已有positive temporal heads只允许作为component-init短暂辅助，并在functional优化稳定后退火到零；它们不进入最终Writer forward。

### 8.4 gain与方向的优化预算

shared训练保持同一optimizer、LR与loss权重，但将完整`gain_readout`和其余全部Writer参数分别按既有norm `1.0`裁剪。该分组不改变
objective或task权重；它只防止幅度梯度占用同一个global clip后持续缩小Frame、Event和Composer方向更新。未分组的73-task
macro610轨迹中global clip触发率为`.8781`，scale norm中位`2.5992`而其余方向norm中位`.5839`；若独立使用同一边界，方向侧仅
`.0386`的step需要裁剪，方向更新倍率中位可恢复`2.6533 x`。

causal-filter的后续对照进一步说明不能把现有方向全部丢弃：在同一四任务free-group-gain反事实中，component-init方向的100步
fit/held平均恢复为`.178/.135`，shared m100方向为`.213/.179`，说明联合训练已经学到小幅可泛化方向；但增量不足以越过当前
函数瓶颈。所以下一fresh先在同一个optimizer与全模块图中恢复group ownership和首步方向信用，不先冻结Process、不改loss、LR、
task比例或训练时长。只有该边界仍non-pass且方向学习继续落后于gain，才依据新证据进入Process/Composer分阶段训练。

## 9. 首轮实现与证据顺序

### 9.1 最小真实smoke

接通后只验证：

- 真实PI0.5 forward与19 x 50 x 2 taps；
- no state/proprio/action进入Writer；
- L_func梯度到达Frame、Event与Composer；
- L_process使用冻结target且无未来泄露；
- relative frame positions只路由Event segmentation/QK，不进入event value；static-repeat只返回carrier；
- 初始small nonzero gain下第一次L_func backward同时到达Composer direction与ragged group-gain rows；
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

首轮历史协议使用component initialization、10 warmup加100 effective updates；effective 60/100对应global macro70/110，目的只是
与J2/R系列直接比较，不是由架构推导出的最优训练时长。新接口先按真实profile与早期功能轨迹采用可续跑的短段，出现有意义correct
功能信号后立即运行held5 correct-only strict250；有希望才补齐70/110历史可比节点，不以一长串内部小数阈值阻塞闭环，也不在
架构未证明时先付出长训练。

首轮12-task资格实验采用每步3 meta加3 target只是该次配置，不是Writer或owner的固定要求。后续运行可按证据只使用其中一类、采用
任意显式比例并改变每步task数。科学sampler先确定task group与权重；多卡执行器随后按视频帧数与functional rows的
outcome-independent成本，在已有frozen evidence cache副本上最小化最长rank耗时。选择性cache复制和动态设备放置不得增删task、改变
task权重、K、optimizer cadence或完整50-horizon内容，formal resume冻结其执行计划。

checkpoint依据correct-only表现、task breadth与相邻稳定性选定。选定并冻结后才一次性运行same-task-other、wrong、no-video、
language-only、first+final、shuffled与reversed controls；这些结果不回流训练或loss设计。

typed-boundary首个fresh资格已经完整结束。task-local task1/task93的50/100四点均在fit/held视频上优于carrier，证明容量保留；
shared m100/m200 held5 strict250却为`39/32`，m200显著低于carrier43，且seen functional继续改善时true-task-held与闭环退化。
其后只替换`C/D` consumer boundary的clean `f33f2955` fresh资格也已完整结束：m50/m100 held5仅`40/35`，逐task为
`0/0/2/38/0`与`0/0/5/29/1`，Goal/Long仍为0；gradient fit/held functional继续改善时两个true-task-held更负，故不续训或运行
negative controls。random legal delta资格随后在m50/m100均为`41/250`且Goal/Long为0，机制诊断确认预测状态有信息、预测头有容量，
但正式训练的随机normalizer和累计readout更新量不足以学成目标。process objective conditioning随后让prediction真实改善，却在
macro50/100都只有`37/250`。causal-prefix forward-filter资格也已完成：m50/m100 strict250为`38/36`，breadth均`3/5`且Goal/Long
仍为0；gradient-task Panel-B略正而两个true-task-held均明显为负，故filter语义修正本身non-pass，不续训也不运行negative controls。

这次non-pass后的正样本诊断将最早剩余接口推进到Composer gain/credit边界。free rank/group gain与component-init/m100方向对照见
§6.4和§8.4；family finite ablation又显示action-out在5/5任务为正，而q只在部分任务为正，说明不是一个family scalar可统一解决。
下一资格仅恢复ragged target-native group gain与G1非零小幅启动，仍用optimizer50/100；50点在训练继续时尽快物化，100点作相邻
裁决。架构未证明前不扩成长跑。

该group-gain-credit资格现已完整结束。macro50/100的held5 correct-only strict250为`37/35`，breadth均`3/5`、Goal/Long均为0，
低于carrier `43`；gradient tasks功能增量继续上升，true-held task74却随训练恶化。冻结正样本诊断表明Process innovation与最终
mobile direction仍有明显task差异，但group-scale跨task cosine达`.99941`，action-out只获得G1成功task-local幅度约
`.016--.020`；task74的q-only有限幅效应为负，v与action-out为正。与此同时，共享Process上的process/functional聚合梯度总和
cosine为`-.114`，Event子集为`-.291`，process范数约为functional的`1.49--1.80x`。checkpoint movement确认主模块已经移动且
m50后增量有限，因此不能以断图或训练不足解释。

下一资格进入本文已预留的分阶段优化，但不增加部署模块或专用坐标：从相同component initialization fresh开始冻结整个Process，
只让现有Composer接受correct cross-episode functional与preservation梯度；冻结Process后不计算对参数无作用的causal auxiliary。
full 50-horizon、真实native X/Y、signed pooling、ragged group gain、rank12+4、73-task数据、task权重、Panel与materialization
均保持不变，首轮仍保存optimizer50/100并做相邻held5 correct-only strict250。这一单变量直接检验Process辅助/联合漂移是否压坏
Composer；若gradient tasks仍正而true-held与闭环仍低，下一根因才是Composer自身的shared dynamic gain/readout函数类，而不是
继续续训或修改Process。

该阶段的窄实现、真实工程profile与科学裁决均已完成。配置显式封存`composer_functional_process_frozen`，优化器只拥有Composer的
`1,141,187`个参数，Process保持eval且所有梯度为0；旧joint配置继续使用原两参数组语义。53项Writer/native-factor/PI0.5 LoRA
测试通过。gpu01物理0上的task1、K1、full-50-horizon两步shared profile中，Composer direction/scale梯度分别为
`1.367/4.732`与`1.790/4.423`，step为`3.476/3.355s`，峰值allocated/reserved为`23.89/32.14GB`，真实functional VJP、Writer
重算与完整rank16 policy消费全部接通。clean detached optimizer50/100的held5 correct-only strict250为`39/43`，breadth均
`3/5`且Goal/Long为0；m100相对carrier恰为`36 retained/7 gained/7 lost`，不是性能增益。训练后段seen-task functional benefit
继续升高且约79%记录为正，true-held task74却继续恶化；Process冻结与Composer真实移动均已确认，所以该阶段正式non-pass，
不续训或运行negative controls。

随后correct-only诊断显示最终adapter方向并未坍缩：四个task的mobile update跨task cosine低而task-specific fraction约
`.85--.87`；相反，195个query-only gains的跨task cosine约`.9997`。task2/task74的exact gain-gradient强烈相反且仅调gain存在
足够局部下降空间，故下一fresh只用§6.4的共享factor-conditioned group token readout替换195-row head。Process继续冻结、
full 50-horizon、signed candidate direction、rank、cap、数据、task权重、positive-only objective与optimizer50/100全部保持，
直接裁决“当前factor可见的共享utility rule”能否把task-specific方向转化成task-disjoint闭环功能。

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

当前set-relative实例已经给出这一分叉的直接证据：authority72 task-local m100在严格配对50初态上由carrier `34/50`提高到
`40/50`，并与历史task expert `38/50`处于同一量级；而相同函数类的shared m100在held5只有`44/250`且target-role held功能
`0/4`明确为正。随后审计发现，旧`9 meta + 3 target`的target task每6个global steps才出现一次，而video与Panel游标都使用
global step，导致全部target task永久只见1/2 fit video与8/16 visits。shared sampler因此必须用per-task occurrence cursor驱动
video、Panel visit与causal pair，并在resume时从同一确定性task schedule重建；task batch大小、role比例和world size只能改变
exposure数量，不能改变单task数据支持集。

当前已启动的`6 + 6`恰好同时修复该覆盖别名并把角色总质量从75%/25%改为50%/50%，所以只能解释为target训练质量/覆盖联合诊断，
不能把结果单独归因于role weight。它仍不规定未来必须包含meta或target、固定比例，也不规定每step必须12个task；若该更强联合
修复失败，不再以比例小扫替代共享条件表示、task-disjoint可辨识性或positive-only信用分析。

一次non-pass只淘汰实际检验的组合。当前函数类只有在task-local支持、shared正功能、matched前端、K、component/random joint及合理规模
训练均获得充分证据后仍系统失败，才停止；EMBER总体停止需要进一步证明zero-interaction static-LoRA合同或数据可辨识性本身构成根本
限制。

## 12. 工程与生命周期

- 新Writer保持一个canonical训练/评测入口；合法差异放入同一配置或窄strategy边界，不复制runner。
- active source ownership固定为：`capture.py`只拥有冻结PI0.5/native taps；`process.py`只拥有Frame/Event表示与causal target；
  `composer.py`只拥有current-bank signed factor生成；`gain_readout.py`只拥有所有target/rank/group共享的current-factor utility block；
  `model.py`只拥有组合与唯一rank16物化；`training.py`拥有共同asset/data runtime与
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

# 过去定向完整 Horizon Writer：正式接续设计

日期：2026-09-08。本文是当前唯一 active design，登记与实施状态见 [progress.md](../progress.md)。
**完整首版已实现，正在真实联合profile与数值合同验证，尚未开始formal训练。**旧layered运行面已退役；当前状态以progress.md为准。

权威顺序为 Owner 最新决定、[长期要求](current_owner_requirements.md)、[AGENTS](../AGENTS.md)、正式状态与本文。
[专家原文及 Owner 裁决](review_materials/20260908/README.md)保存完整推导来历。
专家最后原文采用双向长程；**Owner 最终选择过去单向长程，本文已落实这一覆盖，不能照原文恢复双向。**

## 1. 目标、已定选择和证据边界

输入 exact language 与 K 条同 task、action-hidden、内部有序正确视频，一次生成唯一完整 38-target LoRA，
使冻结 source 从自己的未见初始化闭环执行。视频只参与 rollout 前的编译；source 保持冻结。
最终科学资格为 validation8 single-checkpoint strict paired correct **>145/400**及稳定、breadth、Goal/Long、
跨视频与最终视频因果性，具体见 §8。代码能跑、梯度非零、训练结束或一次高分都不是科学完成。

| 已定项目 | 本设计 |
| --- | --- |
| 观察响应 | `action_out_proj` 实际输入，最终归一化之后的 50×1024 hidden；没有 18 层输入轴或多层融合 |
| 视频证据 | 同一次真实冻结图文 prefix 的最终有效 Z 与 KV；保留视觉 tokens 和 exact task span，禁止 teacher state |
| 局部方向 | 当前 t 只读过去四个实际存在的采样位置 u；真实时间差用原始帧索引 |
| Horizon 查询 | 每帧对先形成 50 条关系，再经一层双向 H attention＋FFN，仍输出 50 个视觉 query |
| 视觉与聚合次序 | 每帧对分别读取前序/当前 Z，形成经核实消息，再按 u 从早到晚用短 GRU 聚合 |
| 局部—长程 | 四组；每组临时 H-read 后一层 **past+self 长程 attention**；前三组逐 H 非线性回写，第四组直接进 compiler |
| 集合与输出 | 每视频独立编码；608 个 paired target/rank queries 联合读取，native D 生成完整 A/B，rank16/alpha16 |
| 训练 | Writer 与观察 Meta fresh、端到端；首版 FM 辅助真实 Writer RL，同版本采样与求梯度后一次联合更新 |

没有新的性能证据。历史分层图在 192/384 的 correct=69/67、other=72/64，熟悉/held 视频 train120=21/18；
旧 run 永久止于 384。基础 SFT 109/107、早期 Writer143、GOMQ151及其非稳定边界，见
[research_history](research_history.md)，不能合并成一个新方案的先验成功。

### 1.1 基线压力、历史教训与后续自主权

以下是已存正式validation8结果，本次按导出rows复算计数，未重新执行环境：

| 参照 | 成功数 | 比较边界 |
| --- | --- | --- |
| 当前旧图配对使用的frozen source | 47/400（11.75%） | 另一历史面板为48/400；不同rows/schedule不混算 |
| train24 source SFT step400 | 109/400（27.25%） | rank128，不读取teacher视频 |
| 同一SFT step425 | 107/400（26.75%） | 相邻点；不是当前新后端重跑 |

来源为 `docs/review_materials/20260907/records/current_source_reference/`、`historical_source_sft/`，
精确目录与历史兼容性核查见[research_history基线](research_history.md#baseline)。
SFT不是rank16 Writer的同参数量对照，但它提供明确的行为参照，参数量差异不能成为长期低性能的免责理由。
新面板需核对task/state、normalization、执行与RNG口径；不能拿80行screen和400行分数直接比较。

在预先定义的有信息量学习节点，若持续不及或仅略高于source/SFT，应当作严重科学non-pass信号，重新定位
表示/条件信息、完整参数生成、共享优化、监督与实际控制之间的机制缺口；不能用source局部增益或内部指标开脱。
这不是唯一根因的证明，也不是初始identity或短smoke必须超过SFT的门槛；不能在失败后无限推迟“有信息量”的判断。
历史旧图67/64虽超过source47却远低于SFT109/107，已经提供了不应继续自我安慰的例子。

每次实质改动先查最近等价历史：完整H/梯度接通不等于理解，Factor几何不等于闭环，clone强不等于shared可学，
熟悉视频也弱不能只归因迁移，多视频/低参数方差不等于高breadth，单峰151不等于稳定，重复小补丁不等于定位根因。
同时检查重复大forward、逐token循环和无效batch等吞吐问题；保留已验证的批量/缓存/VJP资产，不重犯已有工程退化。

Owner授权后续依据充分证据自主改变具体读取、关系、时序、回写、读出、监督/RL、优化和执行布局，必要时做实质重构，
不局限于微调参数。核心思想、信息墙、固定数据/评测与完整一次策略编译的科学目标不变。
本设计是完整首版起点；证据支持修订时记录“旧合同—失败事实—假设—改动—验证—结果”，同步设计/config/schema并维护唯一canonical路径。
不静默缩水，不把“忠实实现”解释为坚持已知错误，也不因局部失败无依据地180度转向。合同内科学迭代无需逐项请求批准。

## 2. 全图与三个有序轴

```text
exact language + K条正确有序视频
  → 每帧同一次 frozen vision/Gemma prefix
      ├─ 最终图文 Z[t,token,2048]：可重读证据
      └─ prefix KV → AE + shared Meta，固定 probe / flow_time=1
                     → R[t,50,1024]，实际 action_out_proj 输入
  → U0[t,50,256]
  → 四组，每条视频独立：
      过去帧对软对应 → 完整H-query → 两端Z读取
      → 经核实的M[t,u,h] → u按早到晚短GRU → 完整局部U
      → 临时H-read E[t,256] → 一层过去单向长程 P[t,256]
      → 前三组逐H条件化回写；第四组P直接送出
  → 多视频集合compiler，38×16 paired queries
  → target/rank/side独立、跨task共享native D
  → 完整76个A/B张量 → 一套LoRA → source闭环执行
```

H 是相对动作位置；u 是固定当前帧 t 下的历史起点序列；t 是完整视频时间。H-query 允许 H 内双向交互，
长程层只允许视频位置 `s≤t`；不得把视频因果 mask 套到 H 上，也不得用 layer depth 或 h 指定人工阶段。
GRU 的各条消息覆盖 `[u,t]` 的重叠区间，不是首尾相接的动作片段。

## 3. 冻结证据与末层响应

固定 `H=50, d_R=1024, d=256, heads=8`；FFN 扩展比4，pre-norm、dropout0。
语言条件 `ell_bar[256]`由 exact task tokens 的带位置、mask 读取形成，不再做一次 Gemma 视觉前向。
该共享语言条件只依赖exact language，可复用当前静态token embeddings的masked learned read；不能从整段视频Z池化得到它，
否则会经语言旁路破坏过去前缀合同。Z内图文融合后的task tokens仍按各自帧归属供视觉查询读取。

\[
R_t=\operatorname{PreActionOut}\big(\operatorname{AE}_{\theta_0,\mu}
(\operatorname{KV}_t,\xi_0,s=1)\big),\quad
U^0_{t,h}=W_RR_{t,h}+e_h.
\]

- `theta0` 基础参数冻结；Meta 沿用18层 AE 的 q/k/v/o、rank4，所有可训练模块 fresh 初始化。
- 固定 probe 为 CPU FP32 `torch.randn([50,32], generator seed=1729)`，每 task/frame 共用；存入 checkpoint。
- `R`为归一化后 hidden，不是一次 denoise_step 返回的 flow，不是十步采样后的动作，也不是校准的教师未来轨迹。
- stride5，按 canonical include-last-frame 规则保留真实索引；不重算 source normalizer。
- 冻结 Z/KV 可有界缓存；R、learned Z 的 K/V、U/P、生成 LoRA 只在同一参数版本内有效。
- 新接口应 hook **action_out_proj 的实际输入**。旧 `ActionLayerStateCapture` 最后项是 final norm 的输入，不能直接取其末项替代。
- Z 应来自同一次真实 prefix 的最终图文输出；现有 helper 只保留 KV，需扩展其返回合同，不能拿静态词 embedding 冒充 Z。

## 4. 一组局部核实

第 b 组读 `U^(b−1)`；所有帧对读同一旧状态，不按视频遍历顺序原地更新。

### 4.1 过去方向与软对应

\[
\mathcal N^-(t)=\{\max(0,t-4),\ldots,t-1\},\quad
\tau_t=n_t,\quad\Delta_{tu}=\tau_t-\tau_u>0.
\]

名义控制记录步下，当前位置 h 对应前序位置 g：
\[
\tau_u+g\approx\tau_t+h\iff g-h-\Delta_{tu}\approx0.
\]

当前为行 h、前序为列 g，每 head a：
\[
C^a_{tu}(h,g)=
\frac{F_b^a(\operatorname{LN}U_{t,h})^\top F_b^a(\operatorname{LN}U_{u,g})}{\sqrt{32}}
+B_b^a\left(\frac{\Delta_{tu}}5,\frac{g-h-\Delta_{tu}}{50}\right).
\]

B 为小型 joint-gap MLP，末投影零初始化。残差坐标与 `(gap,offset)`是可逆线性重参数化，不宣称新增表达能力；
不加速度混合、warp、固定窄对齐带、远端必低可靠性衰减。`/5`与`/50`仅缩放数值，不能再乘 stride。
只计算 `t←u` 一种条件消息；由旧方向迁移时转置未归一化 score，不能转置已 softmax 的概率充当另一条件分布。

\[
[\Pi^a_{tu}(h,:),\pi^a_{\varnothing,tu}(h)]
=\operatorname{softmax}[C^a_{tu}(h,:),c^a_{\varnothing,tu}(h)].
\]

空匹配 logit 依赖当前状态和 gap；不是校准置信度，也不关闭整条视觉消息。

### 4.2 新帧对模式沿完整 H 形成查询

\[
m_{tu,h}=W_m\operatorname{Concat}_a\sum_g\Pi^a_{tu}(h,g)V_b^a(U_{u,g}),\quad
w^a_{tu,h}=\sum_g\Pi^a_{tu}(h,g).
\]
\[
\rho^a_{tu,h}(q)=\sum_{g:g-h=q}\Pi^a_{tu}(h,g),\quad q=-49,\ldots,49.
\]

对 rho 做 learned 线性读取为 r[d]，不压成一个平均位移；直接使用含非空质量的 Pi，不除以趋零的 w。

\[
X_{tu,h}=f_{Q,b}[U_{t,h},m_{tu,h},U_{t,h}-m_{tu,h},r_{tu,h},w_{tu,h},e_h,\bar\ell,\Delta_{tu}/5].
\]
\[
Y=X+\operatorname{MHSA}_{H,b}(\operatorname{LN}X;\operatorname{RoPE}(h)),\quad
Q=Y+\operatorname{FFN}_{H,b}(\operatorname{LN}Y).
\]

每帧对 `Q[50,256]`。H attention 双向，保留所有 H 位置；它使某行新对应模式能够影响其它 h 的视觉 query。
原生 AE 已有 H 内交互，但其生成时没有这次帧对的 Pi/rho；新增模块处理的是新形成的关系信息。
hidden 差不解释成物理速度，匹配斜带不能单独证明过程推进。

### 4.3 两端图文证据与消息

\[
z^{past}_{tu,h}=\operatorname{Attn}_{Z,b}(Q_{tu,h}+e_{past},Z_u,Z_u),\quad
z^{now}_{tu,h}=\operatorname{Attn}_{Z,b}(Q_{tu,h}+e_{now},Z_t,Z_t).
\]
\[
M^b_{tu,h}=f_{M,b}[Q_{tu,h},U_{t,h},m_{tu,h},r_{tu,h},w_{tu,h},\Delta_{tu}/5,
z^{past}_{tu,h},z^{now}_{tu,h},z^{now}_{tu,h}-z^{past}_{tu,h}].
\]

两端共享读取参数和 K/V 投影，以角色身份区分。每组每帧 Z 的 K/V 只投影一次，供相关 edge 复用。
Z_u 是前序起点的实际画面，不是前序 horizon g 对应的未来图像。Pi 是 H×H，不能直接乘图文 token 轴当视觉权重。
不把 M 硬乘 w；动作延续不可靠时，视觉仍必须有机会揭示实际变化和修正。

### 4.4 按历史起点排序的短 GRU

对固定 `(t,h)`，实际邻帧 `u1<…<uL<t, L≤4`，沿 u 从早到晚处理：
\[
\gamma_i=[(\tau_t-\tau_{u_i})/5,
\mathbf1_{i>1}(\tau_{u_i}-\tau_{u_{i-1}})/5,\mathbf1_{i>1}],\quad
x_i=\operatorname{LN}M_{t,u_i,h}+\eta_b(\gamma_i).
\]
\[
c_0=\tanh(W_{0,b}\operatorname{LN}U_{t,h}),\quad c_i=\operatorname{GRUCell}_b(x_i,c_{i-1}).
\]

采用标准 GRUCell；reset gate 作用于 hidden-side candidate affine，与 PyTorch 语义一致。每组共享一个256维 GRU，
跨 t/h/task/video 共享，最多4个 u 步内共享参数。所有 `(t,h)` 批量处理，只有 u 维短递推。

\[
U'_{t,h}=U_{t,h}+W_{O,b}(c_L-c_0),\quad
\widetilde U^b_{t,h}=U'_{t,h}+\operatorname{FFN}_{U,b}(\operatorname{LN}U'_{t,h}).
\]

WO 无 bias；L=0 时邻帧增量严格0，逐位置 FFN 仍可运行。不构造假邻帧，GRU 不跨 `(b,t,h)`保留状态。
有序递推提供消息间的条件依赖，不声称自动消除重叠、一定相信最近帧或成为贝叶斯滤波。

## 5. 四组单向长程与逐 H 回写

每组先临时读取：
\[
E^b_t=\operatorname{HRead}_b(q_b(\bar\ell),\widetilde U^b_{t,0:49})\in\mathbb R^{256}.
\]

完整 U 仍保留。每组一层标准 pre-norm temporal self-attention＋FFN，时间 RoPE 使用 `tau/5`，
attention score 对 `s>t`设为不可见，对 `s≤t`可见；不进行逐帧生成或额外预测 loss。
\[
P^b_t=\operatorname{TemporalBlock}_b(E^b_{\le t};\operatorname{RoPE}(\tau/5)).
\]

前三组：
\[
U^b_{t,h}=\widetilde U^b_{t,h}+
W_{o,b}\operatorname{GELU}\left(W_{u,b}\operatorname{LN}\widetilde U^b_{t,h}
+W_{p,b}\operatorname{LN}P^b_t\right),\quad b=1,2,3.
\]

回写中间宽512、输出256；`Wp P_t`可按帧预计算供50个h复用。不同 h 由自身状态得到不同增量。
不能用只含一个 Key 的 cross-attention 冒充 query-dependent 回写；也不把 P 简单复制50份作为新的完整 U。
第四组 P4 直接送 compiler，没有无用途的最后回写。

每组参数独立；每组内部共享。总计4次局部对应/H-query/视觉核实/GRU/H-read/长程，3次回写。
过去局部、逐帧源输入、过去单向长程与同 t 回写共同保持每组 `U_t`只依赖原视频前缀。
GRU 某条 `[u,t]`消息可以包含当前 t 的证据；不宣称它是在 u 时刻作出的预测。
上述前缀依赖不等于视频对最终行为的因果必要性；compiler 最终合法读取所有 t。

## 6. 集合编译与完整 native A/B

每视频独立得到 P4[k,t,256]；608个 `(target m,rank r)` queries，2层 cross-attention＋query self-attention＋FFN。
每视频基础质量通过读取 logit 减 `log T_k`修正；内容决定实际权重。不设 video ordinal embedding，
不跨视频建立时间边，不平均原始视频或最终 LoRA。得到 `c[38,16,256]`。

\[
a_{mr}=a^0_{mr}+D^A_{mr}\operatorname{GELU}(U_Ac_{mr}),\quad
b_{mr}=D^B_{mr}\operatorname{GELU}(U_Bc_{mr}),\quad
\Delta W_m=\sum_{r=1}^{16}b_{mr}a_{mr}^\top.
\]

D 按 target/rank/side 独立、跨 task 共享；没有 task ID 输入。width256、rank16、alpha16，完整76个 A/B tensor。
D 全零初始化、A0 用 canonical 非零 identity 模板，初始 BA=0；其它新模块 fresh，optimizer/scheduler fresh。
初始 B/D 为0可使首步部分上游梯度为0，须在正常更新后判断通路，不能把这一代数现象当成永久断梯度。
新的架构/参数空间不兼容旧 checkpoint，禁止从旧 short4、384 或 native-head 草稿 checkpoint warm start。

## 7. FM 辅助 Writer RL：完整首版更新合同

### 7.1 数据、随机性与权重

固定 train24，global IDs 来自 `configs/libero_24_8_8_v1/protocol.json`，每 suite 六 task。
每次尝试迭代每 suite 均匀抽1个 task，共4个；每 task 1个 condition，K 从1/2/4等概率抽取，视频无放回且互异。

| 数据/配置 | 首版 |
| --- | --- |
| 主 sampler/optimizer seed | 7；独立子 RNG 流和状态存入 checkpoint，不复用 episode 随机流 |
| Teacher 视频 | episodes 0–15，完整有序，stride5，include-last-frame |
| FM actions | episodes16–41；每 condition 64 queries，episode均匀后frame均匀，有放回；与teacher角色互斥 |
| 冻结动作诊断 | episodes42–45；无梯度、非选点依据 |
| 同task held视频 | episodes46–49；只作训练侧诊断，不替代validation资格 |
| RL | 每 condition 4条相互独立完整episode；初态从该训练task的0–31均匀独立抽样 |
| 训练侧行为诊断初态 | 32–49，与本run RL初态互斥；不是历史上从未使用过的盲集 |
| RL反传采样 | 每episode均匀无放回 `M=min(16,Q)`个decision，使用 `Q/M`修正 |
| 额外meta tasks | 首版为空；后续增加须精确语义/specification审计并预登记allowlist和权重 |

每 suite 一 task 等权，跨迭代对 train24 均匀目标无偏；不是旧cycle采样，不保证每6步恰好覆盖所有task。
task平均、task内episode平均、episode内decision求和，不能再除Q。报告实际 task/condition/K/query/rollout曝光，
不能以 optimizer 步数或 GPU 数代替监督量。

### 7.2 FM 是训练动作监督

\[
x_s=(1-s)a+s\epsilon^{FM},\quad u_s=\epsilon^{FM}-a,\quad
L_{FM}=\mathbb E\frac{\|v_{\theta_0+G_\psi(C)}(o,\ell,x_s,s)-u_s\|^2}{50\cdot7}.
\]

沿用冻结 normalization、flow-time 分布与末动作补齐定义；不顺带改变 padding mask/loss 口径。
监督来自同task另一episode的 actions；teacher部署输入仍 action-hidden。FM系数1，RL系数0.1是可试默认，
不是两种梯度实际占比，也不是 Owner 要求永久固定损失混合。首版从第一次迭代计算两者，不采用G1–G3冻结课程。

### 7.3 实际探索策略与可计算密度

固定本次参数版本 `psi_n=(Writer,Meta)`，生成 LoRA `Lambda_i=G_psi_n(C_i)`。
每个执行decision独立采样原生 `epsilon_q ~ N(0,I[50,32])`，运行完整10步 Euler flow：
\[
x_0=\epsilon_q,\quad
x_{k+1}=x_k-0.1v_{\theta_0+\Lambda_i}(o_q,\ell,x_k,1-0.1k),\quad k=0,…,9.
\]
\[
m_{n,q}=\operatorname{vec}(x_{10}[0:5,0:7])\in\mathbb R^{35},\quad
z_q\sim\mathcal N(m_{n,q},\Sigma).
\]

完整积分50×32后再取前5×7，不能提前缩小flow内部形状。探索加在归一化动作上；执行复用canonical反归一化、
范围与夹爪处理，不额外加sign/tanh/阈值。log probability使用裁剪前完整 z：
\[
\log q_\psi(z_q\mid o_q,C_i,\epsilon_q)
=-\tfrac12(z_q-m_{\psi,q})^\top\Sigma^{-1}(z_q-m_{\psi,q})
-\tfrac12\log|\Sigma|-\tfrac{35}{2}\log(2\pi).
\]

这是外生 epsilon 与 z 的扩展动作策略；`p(epsilon)`不依赖psi，不求未知flow边缘密度，不把FM loss当logprob。
不对环境反传；保存观测在重放中固定。

\[
\Sigma=C_{.8}\otimes\operatorname{diag}(.05^2,.05^2,.05^2,.05^2,.05^2,.05^2,.10^2),\quad
(C_\rho)_{ab}=\rho^{|a-b|},\ a,b=0,…,4.
\]

这是时间优先的35维flatten顺序。五步内相关，不同decision独立；首段固定Sigma，不自动退火或学习噪声。
正式执行J0去掉额外Sigma，保留正常flow noise。训练J_Sigma增益不能替代J0。
若五步中途成功终止，仍可使用采样的完整35维联合score；相关噪声下不能按终止结果随意删尾部密度项。
dummy settling不计score，不改变官方成功终止和suite horizon。

### 7.4 Reward、baseline与无偏decision采样

仅官方成功 `Y_ie∈{0,1}`，超时失败，不bootstrap。不给视频预测、人工阶段、shaping、critic或expert纠错默认支路。
\[
b_{ie}=\tfrac13\sum_{e'\ne e}Y_{ie'},\quad A_{ie}=Y_{ie}-b_{ie}.
\]

4条episode在给定condition/固定策略下独立，包括初态、env、flow noise、额外噪声。baseline/advantage停止梯度。
同组全失败或全成功时advantage均0，RL梯度确实为0；必须报告有效混合组比例，不能虚构成功信用。

对Q个decision均匀无放回抽M个：
\[
\widehat g_{RL}=\frac1{4\cdot4}\sum_{i,e}A_{ie}\frac{Q_{ie}}{M_{ie}}
\sum_{q\in S_{ie}}\nabla_\psi\log q_{\psi_n}(z_q\mid o_q,C_i,\epsilon_q).
\]

可用独立于reward的reservoir sampling，仅保存最多16组完整观测、epsilon、z、m_old与必要标量证据。
不能按接触、最大分歧或成功前最后几步挑点后继续声称相同无偏估计。Q是实际产生探索动作的decision数。

### 7.5 LoRA cotangent 与同版本联合更新

采集、FM求梯度、RL重放期间参数全部保持psi_n。可以并行累积FM梯度，但不能提前做FM optimizer step。
对保存的decision，固定o/epsilon/z/m_old/Y/b/A，动作均值的最小化目标cotangent为：
\[
g_{m_q}=-0.1\frac1{4\cdot4}\frac QM A_{ie}\Sigma^{-1}(z_q-m_{old,q}).
\]

完整10步flow反传到A/B，各步之间不能detach x_k；推理的no_grad包装不能直接充当训练路径。
每condition先合并 `g_Lambda = (1/4) grad_Lambda L_FM,i + sum VJP_flow(g_m)`，
再只组织一次Writer→R→Meta重放。全组、临时H-read、GRU和回写不detach；冻结source不更新但仍参与LoRA可微计算。
所有task/episode系数只乘一次；跨卡对已全局加权梯度SUM，不再按world_size平均。

默认AdamW：LR3e-5，betas(.9,.95)，eps1e-8，weight_decay1e-4；前8个**接受更新**线性warmup，之后先常数。
合并梯度后global norm clip1，不分别把FM/RL或各task归一到同norm，不沿用旧672步cosine。

### 7.6 有限候选trust检查与拒绝处理

只在psi_n求一次 `grad L_FM − .1*g_RL`，形成一个AdamW候选方向d和待提交moments。
依次检查 `alpha∈{1,1/2,1/4,1/8,1/16,1/32,1/64,1/128}` 的完整候选 `psi'=psi_n+alpha*d`：
\[
K_q(\psi')=\tfrac12(m_{\psi',q}-m_{old,q})^\top\Sigma^{-1}(m_{\psi',q}-m_{old,q}).
\]

每episode从已均匀保存的decision中再均匀取至多4个；各candidate复用同一检查子集。按episode/task平均，
本批task最大均值≤0.02方可接受；单位是完整35维动作块。候选用自己的Meta重新生成R和LoRA，不能复用旧R。
只做候选前向，不在候选点对旧轨迹重复普通score梯度。这个检查不是off-policy修正，也不是成功单调保证。

2026-09-08真实profile发现提前成功后的活动batch缩小会污染重放：同参数task21的KL为0.07968，超过0.02；
始终batch4的五个task均为0。因此每decision额外记录采集flow的实际batch尺寸，RL VJP与trust按原尺寸分组重放。
尾组用已选真实记录补齐，补齐项梯度为0且不进入KL/episode/task权重；不重算或替换采集m_old，不扣底噪，
不扩dtype或固定batch1，Sigma与0.02阈值不变。候选仍用自己的Meta完整重读R。该修复由同参数自比较和完整联合profile验证。

2026-09-08进一步profile在消除batch伪KL后仍1/8接受。极小非零执行B的动作差为0，有限B响应不单调，
没有发现任何非零B都引入固定扰动的分支；也未区分剩余响应的模型非线性与精度效应。
本次只检验优化假设：首版四候选可能过早截断可行步搜索，因此同一Adam方向的有限回溯扩为上述八个尺度。
不改变LR、gradient、Sigma、原始m_old、0.02或任何科研性能线；候选不额外求梯度，也不以接受更新当作行为通过。
修订后的fresh四轮profile仍1/4接受，新增四尺度没有贡献有效更新；step2/3/4在1/128仍maxKL=.02300/.07222/.03424。
该范围扩展没有解除停滞，停止同样续试；正式学习尚不启动。下一步用固定A、原B/完全同B重放/相邻BF16 B的
只读完整flow诊断，区分执行端量化响应与生成端变化；不替换训练m_old、不扩大dtype或据此放宽科学资格。

接受缩放步时提交本次moments并统一缩放实际参数步；全部拒绝则参数、moments、optimizer step和warmup计数不前进。
**已消费的sampler/RNG和尝试迭代计数仍前进**，下一次重新采样，不能恢复同一随机流而无限重播同一拒绝批次。
记录 attempted/accepted/rejected、alpha、各task KL及梯度/奖励统计。接受规则、clip和Adam之后不声称实际更新无偏。
本批轨迹不进入后续参数版本的actor训练；保留紧凑证据及必要选中观测，不默认保存每一步大图像副本。

### 7.7 Checkpoint与恢复

新schema必须显式区分末层R、Z、过去窗口、H-query、GRU、四组单向长程、native D及FM/RL更新协议；旧checkpoint明确拒绝resume。
保留Writer/Meta/optimizer、LR状态、sampler/cursor、各rank RNG、probe、world topology、attempt/accepted计数、
Sigma、更新/数据版本和schema。正式checkpoint在完整迭代边界保存；exact-resume锁world size/topology。
在途rollout若无完整可恢复状态，不能伪称exact-resume。新run fresh，与旧384/旧draft无resume关系。

## 8. 实施、行为节点与正式资格

### 8.1 实现后进行必要真实验证

1. 末层读取确为action_out_proj实际输入；Z来自真实最终prefix；Meta范围与teacher信息墙正确。
2. 过去窗口/score方向/H-query跨行依赖/GRU顺序/四组同步与单向长程；修改未来原始输入不影响先前U/E/P。
   结构测试可用合成张量，不用真实shuffled/reversed视频结果选架构。
3. 真实FM与RL梯度到A/B、Writer、Meta；同版本、10步flow无detach、LOO与Q/M及跨卡权重正确。
   identity首步零上游梯度不作为失败；正常更新后核对实际通路。
4. 最长真实K1/K4、真实FM/rollout/replay下测LoRA/s、queries/s、每尝试更新墙钟、各卡峰值。
   合理复用prefix/Z和帧级投影，批量edge/H/decision，不逐token循环；只做与声明相关的检查。

不把每个模块拆成冻结课程或全面消融矩阵。机制通过后以完整候选获得共享学习和行为证据。

### 8.2 首批学习与随后预登记

首版每24个接受联合更新做训练侧 `Sigma / 0`严格配对行为诊断，保留官方flow noise。
可先使用train24×states32–36共120行，从held视频46–49预先固定K1条件，两个噪声arm完全配对。
源参照必须同实际task/state/env/policy合同，旧source16/120对应states0–4不能冒充新面板基线。
该120行细化是本次交接的执行默认，不写成Owner原话或旧专家实测。

新session在真实profile后、首次正式学习结果之前登记首段命令、checkpoint/行为节点、继续/止损规则和资源预算。
首段以24个接受更新及其训练侧诊断为起点；profile若显示明显失衡，先优化执行，不能直接默认约10小时长跑。
不按被拒绝次数无限重复；多次拒绝或长期全失败时区分实现错误、步幅/探索支持和有效non-pass。
具体后续strict400节点依据投入和曝光预先冻结；有信息量且出现广泛能力后及时做strict400，不靠内部loss选点。
未预填GPU、world size、最终总步数、实际吞吐和绝对run目录，因为这些需要新session的实时证据；这是执行工作，不是额外审批。

### 8.3 资格与最终controls

正式资格只认同一checkpoint的validation8×states0–49=400行，correct与same-task-other严格配对，
K1、每个ordinal内部从同task全部50条合法视频无放回取两个不同视频；跨ordinal遵循canonical teacher schedule，
teacher seed沿用20260907以便核对既有schedule。
支持dynamic K的训练覆盖1/2/4真实不同视频；其它K能力报告需明确面板，不能因K1通过就声称所有K均通过。

沿用旧预登记中对Owner定性目标的操作化口径，首个新资格分数前写入新run registration：

- 至少两个相邻登记checkpoint各自correct>145，breadth≥7/8，四suite非零，Goal/Long各≥10。
- 相邻correct下降≤10，churn≤20/400，成功集Jaccard≥.85。
- 支持稳定性的每点other相对correct下降≤10，churn≤40/400，Jaccard≥.80。
- eligible single checkpoint按correct、other、较晚步数依次选择；不使用union、checkpoint融合或内部loss。

这些阈值是沿用的研究操作化，不冒称Owner逐字指定；修改须在看到相关分数前说明科学理由，不事后放宽门槛。
selected checkpoint冻结后才做language/no-video、static端点、cross-suite wrong及最终shuffled/reversed等因果controls；
shuffled/reversed只在真实frame重排后完整forward，绝不进训练、checkpoint选择或架构调整。
方法冻结后才按32/8合同fresh重训与最终Test。所有面板报告per-task/suite、breadth、R/G/L、churn、相邻/跨视频重合。

## 9. 实现owner与生命周期

当前源码、入口和职责地图见[README](../README.md#当前代码与运行入口)，实现验证状态见[progress](../progress.md)。
`writer/native.py`只保留post-norm完整H读取和同forward的最终Z/KV；`writer/horizon.py`/`relation.py`/`attention.py`
负责完整过程图与集合compiler，`writer/native_factor.py`负责76张量输出。旧18层capture、layered/coordinate图及旧FM-only入口退役。

`writer/joint.py`组织每condition同版本的采集、FM/RL合并cotangent和一次Writer/Meta反传；`flow.py`保留完整10步可微执行，
`rollout.py`复用canonical环境池和预处理，`rl_math.py`负责Gaussian信用、reservoir和单方向trust事务。
`learning_data.py`持久化独立随机流；`training.py`维护attempt/accepted、接受更新warmup和完整边界checkpoint。
这些owner复用现有FM、LoRA注入、checkpoint与评测，不保留旧antithetic信用或cycle/cosine作为新训练fallback。

唯一入口为 `scripts/train_horizon_writer.py` / `scripts/materialize_horizon_writer.py`，配置为
`configs/pi05_horizon_writer_v1.json`。旧checkpoint明确拒绝新schema resume。图、读取、联合更新和评测的测试随实际合同替换；
不继续保留旧18层/双向断言。新图的实现与CPU测试不替代§8规定的真实机制、成本和行为证据。

未合并 `codex/native-factor-readout` 草稿只有旧上游末端替换，不整支合并；其dirty worktree与旧正式detached worktree保留。
旧实现由Git、sealed configs、formal artifacts和research_history追溯，不恢复旧run。

## 10. 资源、Git与交付

源资产入口是 `configs/pi05_writer_data_v1.json`及README，不复制dataset/model/tokenizer/env。
初始edge chunk8、observer frame chunk4、policy microbatch8只作profile起点，不锁死低吞吐batch。
约329.5M参数的native D，FP32参数/梯度/两Adam moments约4.91GiB，不含激活、checkpoint和LoRA banks。
大输出前从strg01检查data0/data1各自quota及真实个人用量，计入完整checkpoints、候选参数、选中rollout观测、物化与临时空间；
不能沿用旧32GiB预算或共享df空闲量。

实现从最新main出发，按隔离需要使用 `codex/<topic>` worktree；验证后集成main并push，清理已合并task-owned worktree。
formal train/eval只从clean pushed commit的detached frozen worktree启动；launch前同时live检查两节点，
单节点1–6张真正提高吞吐的A40，固定NCCL_P2P_DISABLE=1/NUMA/deferred NCCL，独立evaluator不用NCCL。
不继承历史GPU空闲、固定卡号或旧tmux中的待执行意图。

本设计没有必须先重开讨论的首版架构选择。Owner要求新session先全面阅读仓库、完整阅读并忠实实现本设计，不静默缩水或偷换关键依赖；
创建覆盖整个科研过程的goal，不设置未经要求的token预算。新session负责真实机制与成本验证、结果前预登记、正式实验、
依据结果实质迭代、strict400/稳定性与最终交付。后续结构修订按§1.1自主开展，不限于小补丁；必须基于真实证据并同步更新正式合同。
遇到明确工程错误就修，遇到科学non-pass只淘汰实测假设，不靠无限小扫或再一轮全架构专家审查代替科学推进。

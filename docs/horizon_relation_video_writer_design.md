# 过去定向完整 Horizon Writer：正式接续设计

日期：2026-09-08。本文是当前唯一 active design，登记与实施状态见 [progress.md](../progress.md)。
**完整首版已实现；Owner最新要求先纯监督FM达到有证据的平台，再接独立共享Writer RL。监督正式启动状态见progress。**旧layered运行面已退役；当前状态以progress.md为准。

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
| 训练 | Writer（含内部读取模块Meta） fresh、端到端纯FM监督；有证据平台后从单个监督checkpoint接独立RL |

设计初稿不构成性能证据；当前实测由progress与formal artifacts登记。历史分层图在 192/384 的 correct=69/67、other=72/64，熟悉/held 视频 train120=21/18；
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
当前按§8.2.5采用`first_query_only_v1`：首块残差内容从task-independent target/rank identities开始，language只加到首cross的检索query；任务条件内容经真实P4 Value进入。第二块按实际首块输出继续。旧裸language残差只由原冻结提交和100–600结果保存，不作为当前运行开关。

\[
a_{mr}=a^0_{mr}+D^A_{mr}\operatorname{GELU}(U_Ac_{mr}),\quad
b_{mr}=D^B_{mr}\operatorname{GELU}(U_Bc_{mr}),\quad
\Delta W_m=\sum_{r=1}^{16}b_{mr}a_{mr}^\top.
\]

D 按 target/rank/side 独立、跨 task 共享；没有 task ID 输入。width256、rank16、alpha16，完整76个 A/B tensor。
D 全零初始化、A0 用 canonical 非零 identity 模板，初始 BA=0；其它新模块 fresh，optimizer/scheduler fresh。
初始 B/D 为0可使首步部分上游梯度为0，须在正常更新后判断通路，不能把这一代数现象当成永久断梯度。
新的架构/参数空间不兼容旧 checkpoint，禁止从旧 short4、384 或 native-head 草稿 checkpoint warm start。

## 7. 先纯监督，再独立共享 Writer RL

### 7.1 Owner最新阶段安排与历史地位

2026-09-08 Owner明确覆盖“从第一轮FM/RL同一步联合更新”的此前默认。完整图不变；先使Writer（含内部读取模块Meta）
从fresh联合学习纯监督FM。source基础冻结。监督阶段不计算RL loss、不采集用于RL更新的rollout，
不做RL trust/KL候选检查、接受筛选或整步回滚。正常闭环评测继续作为真实能力裁决。
既有joint profiles仅为历史机制、执行与成本证据，其checkpoint不初始化正式监督训练，也不算监督学习结果。
精度/布局/批量累加等已验证的执行一致性修复保留；RL信用、trust或探索敏感性不阻塞监督启动。

### 7.2 数据、随机性与公平权重

固定train24；实际已执行且本轮冻结诊断沿用的学习合同为每suite均匀抽1 task、共4条件×64queries，每条件权重1/4。§8.2.3的24-task候选尚未运行，当前后置，不作为活动训练配方。
每task条件恰好一条完整teacher video，K=1、内部保序、stride5、include-last-frame。teacher episodes0–15；FM actions episodes16–41，与teacher严格不交叠，均匀episode再均匀frame、有放回。
训练侧独立动作验证episodes42–45，held teacher videos46–49，均不用于梯度；validation/test不得产生梯度。额外meta tasks为空。
video/query使用独立持久随机流，seed7；多卡只分配完整条件，已经乘1/4的梯度跨rank SUM，不再除world size。每个条件完整64-query随机性先生成，再按实际physical microbatch切块；卡数、分工与microbatch不改变逻辑batch或权重。
混合K历史保留；K1全部通过后再开展few-shot，集合compiler保持完整，暂不进行K2/4训练或测试。

### 7.3 同task跨episode FM与完整端到端梯度

\[
x_s=(1-s)a+s\epsilon^{FM},\quad u_s=\epsilon^{FM}-a,\quad
L_{FM}=\frac14\sum_{i=1}^{4}\mathbb E\frac{\|v_{\theta_0+G_\psi(C_i)}(o,\ell,x_s,s)-u_s\|^2}{50\cdot7}.
\]

复用冻结normalization、原生Beta flow-time分布、独立Gaussian noise及末动作补齐；不改变监督目标或padding口径。
按完整逻辑query批次固定随机性，microbatch只改变执行，不重复生成不同noise或改变task/query权重。
每condition生成一次完整LoRA，在冻结policy求FM的A/B cotangent；随后完整Writer重放一次，R cotangent沿观察Meta重放。
末层完整H、两端Z、全部关系/GRU/长程/前三回写/集合compiler/native A/B都保留，不能detach或缩视频。
只有冻结prefix/Z/KV可跨更新缓存，读取R不能跨Meta更新复用。合法identity初始化首步上游零梯度由正常更新后再核实。

Writer/FM保持已验证BF16 autocast，trainable参数FP32，source保持原生权重类型。
正式执行不加外层autocast；生成A/B适配真实物理参数dtype/连续布局，批量LoRA先相加再cast。
原生execution修复的非零LoRA物理parity已验证；允许正常kernel/reduction差异，不扩source dtype。

AdamW LR3e-5，betas(.9,.95)，eps1e-8，weight_decay1e-4，global norm clip1；前8个optimizer updates线性warmup后常数。
每轮直接更新一次，无RL项和trust回滚。后续优化修改以学习证据登记，不能无限小扫LR/rank/seed来回避能力问题。

### 7.4 Checkpoint与恢复

监督stage=`horizon_relation_writer_fresh_supervised`，run schema=`ember_horizon_relation_writer_supervised_run_v1`，
update_version=`supervised_fm_writer_meta_v1`。保存Writer、optimizer、scheduler、sampler/cursor、每rank RNG、
probe、source/data版本及world topology；macro就是已完成optimizer updates。exact-resume锁原config、source和topology。
正式监督初始run从fresh开始，与joint profiles和旧384无resume关系；同run同合同完整边界可exact-resume。
本轮K1按Owner补充纠正明确fresh：旧混合K run在安全完整边界停止，checkpoint和结果仅作历史探索证据，
不得成为K1初始化。Writer全部可训练参数重新初始化，LoRA采用合法identity；optimizer、scheduler、sampler/RNG全fresh，
从step0独立正式run开始，冻结source/资产复用。原exact-resume仍要求同run/config/topology；未来换卡数须先有受控迁移实现与登记。
每个update共4条件×64queries=256；每个条件梯度先乘1/4，跨rank SUM，整个逻辑batch后统一裁剪和更新一次。§8.2.3曾提出的受控分叉仍未执行，不改变已存checkpoint的恢复语义。

### 7.5 独立RL阶段在监督平台后另登记

用§8.2的真实学习与行为证据判断平台；监督充分仍弱时先定位原因并允许实质改进，不能把“饱和”当作交给RL救场的理由。
监督阶段选定并保留单个checkpoint后，从其Writer初始化共享RL训练；新建独立optimizer/scheduler与stage记录。
RL阶段默认仅RL目标，不自动混回FM，不部署task-local优化。探索、信用与更新约束依据监督后的真实行为重新审视，
不机械恢复已造成停滞的Gaussian/trust设置。具体合同在看到监督证据后、RL启动前另登记，不现在盲选。
报告相对固定监督起点的收益、遗忘、breadth、相邻稳定与J0正式执行；监督checkpoint保留为可回退参照。

## 8. 实施、行为节点与正式资格

### 8.1 实现后进行必要真实验证

1. 末层读取确为action_out_proj实际输入；Z来自真实最终prefix；Meta范围与teacher信息墙正确。
2. 过去窗口/score方向/H-query跨行依赖/GRU顺序/四组同步与单向长程；修改未来原始输入不影响先前U/E/P。
   结构测试可用合成张量，不用真实shuffled/reversed视频结果选架构。
3. 真实FM梯度到A/B、Writer、Meta；跨episode采样、全局task权重与直接optimizer更新正确。
   identity首步零上游梯度不作为失败；正常更新后核对实际通路。
4. 当前最长真实K1、真实FM与完整Writer replay下测LoRA/s、queries/s、每尝试更新墙钟、各卡峰值。
   合理复用prefix/Z和帧级投影，批量edge/H/decision，不逐token循环；只做与声明相关的检查。

不把每个模块拆成冻结课程或全面消融矩阵。机制通过后以完整模型获得监督共享学习和行为证据。

### 8.2 监督学习节点、独立验证与平台期

最新Owner安排覆盖旧24/64/128/192及每点correct/other密集评测。旧配置运行在可恢复完整checkpoint边界结束；
其混合K结果与原注册保持历史身份。后续K1每段连续训练约一小时，按优化后实际吞吐选择中间和末尾两个近等间隔节点，
checkpoint使用50或100的倍数，并在看到成绩前写入配置及segment registration。每段评测后依据证据自主继续。
首个fresh K1段登记0→200：完整逻辑更新实测约18s/step，100/200保存单checkpoint，段末依次correct400；
独立held-action诊断在0/200，train96按分数反映的获取/泛化需要安排，早期不跑other。后续用CLI显式登记两个节点，原run合同不重写。

当前以validation8 K1 correct strict paired400为主；沿用§8.3的task/state/video/policy RNG合同。
已有correct raw rows持续计算per-task/suite、breadth、R/G/L、churn和相邻Jaccard。
训练侧后续闭环诊断登记为train24×states32–35 J0 paired96：K1 held teacher46–49，每task四条各一次；从canonical排列中过滤合法四条，按固定state origin32分配。
other若需要则在同一合法四条排列内偏移17 mod4，仍各一次且逐行不同。旧train120在四条视频上重复复用，保留历史；source19/120不能直接作96行基线，需同状态子集重新核对或重跑source96。
独立held-action诊断保留train24等权、actions42–45、每task128queries、teacher46+(task mod4)、seed20260908+task，
全程no_grad、不消耗训练sampler；节点按同段需要登记，不能代替闭环或选模型。
绝对分数仍低且持续获取时不重复other。correct接近/超过目标并出现相邻稳定候选后，再补other资格；最终controls在选点冻结后执行。

记录累计optimizer updates、FM queries、每task条件曝光和墙钟，fresh K1单独统计，不将旧混合K曝光计入本轮。
历史监督曝光参考v5.2=75600、v6-fast=192000、SFT=230400queries；优化过程与计算量不同，不承诺必达分数。
64steps=16384queries仍早；192steps也不能自动视为充分。结合充分曝光、held FM、训练task表现和多个有信息量correct节点判断平台。
至少三个有信息量节点中validation最佳改善≤5/400、训练侧同口径面板改善≤2.5个百分点（新96行面板不与旧120行直接比较）、held FM相对改善≤2%，且无持续breadth/suite获取，
只能构成平台候选，还需判断监督量；去掉原固定128update充分性暗示。有改善继续，弱平台先定位真实能力缺口，不机械交给RL。
不追求数学完全收敛，不无限续训或无依据超参小扫。RL阶段独立登记，K1限制同样适用。

### 8.2.1 冻结200/400训练任务诊断的裁决合同

Owner恢复后第一轮原样400→600、保留500/600 correct400；600是信息节点，不是整个任务终点。
与其并行比较冻结200/400的§8.2 held-video train96，唯一训练历程变量是checkpoint；两点task/state/video/environment/policy RNG完全相同，无梯度、无选点。
source参照为已完成source120中预登记states32–35的固定96行（15成功），先核对source/policy/environment/normalization/RNG合同，不能称作新执行的source96。

本轮要区分共享训练任务能力获取不足、同task视频泛化失败、跨task迁移不足及FM改善未转成闭环行为。
历史强专家与旧Writer排除了“合法LoRA或视频到LoRA原则上不可能”的说法；旧图熟悉视频也弱、旧v6条件组织差异和当前非零梯度，均未给当前完整图确定根因。
现有train24 held-action FM下降只证明固定动作面板的拟合改善，不证明训练task已闭环学会，也未排除后期回升。

- held-video训练任务出现广泛能力、而validation仍弱：支持优先检查跨task迁移，但不同task难度使原始分数差本身不是因果量；结合各自source增量、suite/breadth与相邻保留判断。
- held-video训练任务仍弱：尚不能区分共享获取和视频泛化，才追加200/400熟悉视频诊断。固定同96状态/RNG，只改变teacher视频；视频必须来自截至200已实际曝光的共同池，映射在看结果前冻结，各task四条不同视频各一次，不按成功选视频。
- 熟悉视频广泛强而held弱：支持视频条件泛化为下一定位层；若两者都弱，优先定位共享系统能力获取。单task局部成败不足以定分支，不能自动称梯度冲突或容量不足。
- 若200→400 FM改善而训练闭环不改善，先定位任务动作阶段及当前状态条件，观察失败究竟发生在接近、接触/抓取、保持/搬运还是释放/后续子目标；任何新增诊断须说明它如何改变监督或接口修正决策。

面板每task仅4状态，报告绝对成功、source和相邻的配对R/G/L、churn/J、suite/breadth与条件曝光；保留配对差异的不确定性和task聚类边界。
差异不清晰时围绕具体疑点补必要证据，不强行贴根因标签，也不自动展开clone/free-code/probe/消融矩阵。
只有机制被支持后才做对应受控修正：条件组织对照保持query预算和task权重；接口重构说明首次失效接口、新职责及行为预测；每次一个主要因果变量。
纯FM/source冻结、信息墙及最终controls顺序不变；不以小扫或RL代替定位。诊断请求、实际映射和对照记录保存在当前run的`k1_fresh/train96_diagnostic/`。

### 8.2.2 冻结600训练任务诊断：区分后续整体退化与跨task退化

在读取600闭环分数前登记：500 correct70/400，400为87；训练held-video证据目前止于400的59/96。因此不能把200→400“训练增而validation降”的解释直接外推至600。
补冻结600同一train96，与400逐行配对，source仍为固定15/96子集。唯一变量为checkpoint；task、states32–35、held videos46–49、K1、policy/env RNG及预处理不变；无梯度、不选点。
历史已排除当前所有训练任务完全未学会，但未排除400之后整体行为退化。600训练仍广泛保持而validation继续弱，支持下一步聚焦跨task迁移；二者一起退化，优先定位后期能力保持/优化；若变化集中Long，保留局部获取与视频泛化分支。
小面板差异不够时不强行归因；熟悉视频仅针对仍未区分的实际缺口按§8.2.1补必要证据。此项不自动启动clone/free-code、修改配方或恢复训练，600 strict400主线照常并行。

### 8.2.3 条件组织受控诊断：完整task覆盖、相同query预算与期望任务权重

**状态更正（2026-09-09）：本项为未运行、当前后置的历史候选。Owner已重新授权分析实验及证据驱动推进，但没有把24-task组织认定为修复；首批诊断见§8.2.4，当前不启动本分叉。已产生100–600结果的实际合同仍为冻结b6d70d98的4task×64queries。**

**依据与竞争解释。** 原样500/600 correct为70/82，600仍只覆盖4/8且78/82成功集中11/26；训练held-video200/400/600为52/59/67。没有观察到整体训练行为退化，跨task迁移为优先层级，但尚未识别根因。要区分：每步少量teacher/task条件的更新组织是否是可干预因素，还是在改变该组织后仍存在任务支持/表示/编译接口的泛化缺口。
旧v6的条件数、任务覆盖、视频池和其它实现同时不同，不能因历史分数直接归因；旧meta73/target18还改变task权重与总queries。当前诊断不新增meta tasks，不提前使用最终视频controls。

**唯一主要变量。** 从原fresh K1完整macro400学习状态分叉，把每update“四suite各随机一task、每条件64queries”改为“全部24 tasks各一个K1条件、总256queries”。每suite六task共64queries：按固定task顺序，两task取10、四task取11；低query任务按分叉后步数每步轮换一对，三步各task累计32queries。
各task FM先在自身10/11 queries上取均值，再乘1/24，不能把256行直接混成样本等权均值。原方法每task被抽中概率1/6、条件权重1/4，期望权重也是1/24；新方法每步确定为1/24。总query预算、suite权重与期望task目标保持，改变的是这个目标的条件组织与梯度估计。
条件数、每步task覆盖和每条件query数在固定预算下联动，不能由本实验进一步唯一命名“梯度冲突”或“视频条件数”根因；它们属于同一个预算分配干预。实际视频、query和噪声轨迹也不会逐样本匹配原随机task调度。

**继承与信息墙。** 保持架构、全部episode pools、K1、source/normalization、optimizer参数顺序与状态、scheduler、precision、clip、seed和world4/global-rank顺序。原注册继承gpu02物理1/2/3/6；因1/3后来被占用，Owner在2026-09-09明确允许仅本次分叉改用0/2/4/6，保留各rank完整RNG，按新GPU重新绑定NUMA/CPU affinity，并记录父子UUID映射、重做profile；子run后续exact-resume仍锁自身完整拓扑。显式`controlled-fork-from`加载完整400，继承rank RNG、video/query随机流和历史task occurrences；不伪称fresh，也不称原run exact-resume。新run只写401起日志，父400/1600条件/102400queries由父记录引用，不复制或重标。
新sampler保存分叉边界、父occurrence offsets及轮换相位所需状态；每task计数=父计数+(当前step−400)，总条件=1600+24×(step−400)，总queries仍=256×step。源task随机流不再参与新调度。子run exact-resume验证自己的配置、topology、完整学习状态与日志偏移。输入/输出/梯度信息墙完全保持。

**节点与执行。** 用真实四卡完整逻辑update profile确认峰值和吞吐，profile状态不作为正式起点。正式仍从原400重新加载，保存完整节点后在500/600各做canonical correct strict400，与已有原样500/600同task/state/video/RNG比较；终点600补同口径held-video train96，与原600的67/96配对。
可按实测速率拆成450/500与550/600两段，450/550仅为恢复边界，不进入正式checkpoint选择。新增条件计算量须如实报告，不能宣称同query即同FLOPs或墙钟；不复制大资产。

**结果分支。** 两个匹配节点出现实质绝对增益并扩展跨task/suite能力，支持条件组织是有效干预，继续按原资格判断相邻稳定；只有低churn或原少数task波动不算解决。训练侧改善而validation仍弱，说明本次组织调整未解除迁移缺口，应再定位独立task支持或具体接口；两侧都弱只否定本次已检验分配，不证明完整图无容量。不盲扫query档位、LR/rank/seed，不由负结果自动转RL或机械展开消融矩阵。
实现由现有`learning_data.py`、`supervised.py`、`training.py`承担；原四条件运行由冻结提交和formal artifacts保留，活动树不增设第二套训练器或永久fallback。

### 8.2.4 原200/600冻结诊断与后续修正排序（2026-09-09）

**授权与先后。** Owner允许直接开展分析实验和内部结构拆解；先获得能区分原因的证据，再决定正式模型修改与训练。休息期间在既有科学精神、信息墙和资源范围内持续自主推进。常规换卡/调度无需再问，学习状态迁移仍按真实合同处理。
原K1不原样续训700/800，不自动执行24-task分叉。以下诊断不更新Writer/Meta/source，不挑新的checkpoint，不提前shuffle/reverse，也不改变正式400分数。

**新增历史反证改变了优先级。**

- `34be4a0` Target-Owned Writer的76个native factor heads已经在同target内跨16 rank共享、跨target和A/B不共享；其末投影20,594,688参数与拟议rank-shared D完全相同，正式50/100/150/200为99/76/86/68。旧前端、1024维factor输入、W_in ownership、初始化/优化和删除DirectionStores等同时不同，不能作为当前图的单变量否证；但共享D并非未试过的新机制，当前不凭省参数就优先重训。
- 当前checkpoint header确认独立D为329,515,008参数，占Writer368,675,520的89.38%；只共享同target内rank会变成20,594,688，整Writer59,755,200。该约束保持单套rank16 LoRA的理论可达秩，但收紧跨条件/跨rank的联合native span；不是无损压缩，也不证明过拟合。
- 旧v5.2/v6实际让纯language只作Q检索、content来自视频语义；它与较强能力共存。另一方面Dynamic-K Semantic-Address、task-grounded D/G和DirectFamilyB有类似路由/Value限制而correct仍约100。删除裸语言残差只消除一条代码通路，不保证动态必要性、可迁移过程或高闭环。
- 不从这些历史方案的最终shuffled/reversed结果选择当前架构。完整原件及资格臂来源由research_history索引。

**A. 真实训练输入下的内部接口分析。** 固定原checkpoint200与600；24个train tasks全部覆盖，每task使用正确完整teacher46及同task diagnostic actions42–45固定16queries，复用已有诊断采样/flow RNG定义，跨checkpoint完全相同。共48个条件/768个FM queries，不消费训练sampler、不创建优化器更新。

同次真实视频forward记录P4、language及compiler的实际初始query与两层cross-attention/self-attention/FFN分量。得到正常生成LoRA的训练侧FM cotangent后，只向输入P4/ell和这些中间分量求导，统计局部增益导数 `<dL/dx,x>`、敏感性及分量大小。输入、source、全部Writer和Meta数值冻结；不用zero/no-video输入，不替换checkpoint模块，也不合成shuffle/reverse。

要回答的是当前参数下哪些接口对动作拟合实际有作用、200→600是否出现一致变化；梯度/范数小不等于视频被忽略，贡献项也不可相加为完整性能归因。这项诊断不单独识别R→P4的过程职责，不由FM的优劣选模型或宣称闭环修复。

**A2. 根据A结果追加单支路冻结干预。** A完成48条件/768queries：固定16-query FM .115832→.107259，20/24 tasks下降；P4 cotangent全非零，cross/self局部敏感度同数量级。梯度观察不足以判断实际支路依赖，故复用完全相同输入与200/600，比较normal、仅query_language输出置零、仅block1.cross输出置零、仅block2.cross输出置零四臂，共48×4×16=3072 FM queries。
每臂都保留真实完整P4，至少一层cross仍读视频，P4生成时的exact language和全部原生图不变；不做两个cross同时置零、整模型language/no-video或最终视频controls。使用原functional loss的真实velocity/time/query-loss输出，统计有效50×7动作维度的配对FM及同FM时刻implied clean chunk变化；后者不是10-step动作采样或closed-loop。
这是已训练冻结checkpoint的activation lesion，临时hook在调用后撤除，不写checkpoint、无optimizer更新或新的正式架构。大幅删除支路会产生训练分布之外的中间状态，结果只说明原参数的使用依赖，不能推断fresh删支路必然好/坏，不选择最佳gain或将这些训练侧分数当正式选点。原A数据保留，新预注册和结果单独存入同一internal diagnostic目录。

**A3. 语言首层路由与残差内容分离。** A2已完成：query_language置零使200/600平均FM增加.028474/.029431，23/24与24/24任务变差；单cross移除增加约.0012–.0030，600作用更大。这说明现参数依赖语言初始项，但A2同时改变首层视频检索和残差内容，不能据此选择删除内容路径。
因此沿用相同48条件、每条件16固定queries，追加normal、仅移除首层cross相加时的language残差、仅移除首层cross检索query内的language三臂，共2304queries。设原初始query `q=e_target+e_rank+l`、`c(q)=Cross(LN(q),LN(P)+time,LN(P))`：原首层相加为`q+c(q)`；内容移除为`q-l+c(q)`；路由移除为`q+c(q-l)`。首层后self/FFN及第二个compiler block全部按实际受干预的中间状态继续原forward；所有真实P4、前端语言、weights和FM样本保持。只改第一层的一个位置，不把第二层后续变化误写成也被钳制固定。
这两臂是配对冻结activation干预，用于区分原参数的首次读取与内容依赖；不是完整条件贡献分解，也不证明fresh修改优劣或video dynamic必要性。与A2跨checkpoint对齐同一真实视频/query/time/noise，保留正常基线，无梯度/optimizer/held数据/最终controls；临时hooks仅存在于诊断脚本。实现前核验实际CompilerBlock公式，原A/A2原件保持。若仍不能分离有效修正，不再机械追加language/cross消融矩阵，应转入一个明确预登记、保持其它合同的学习干预。

**B. 一次性sealed held行为轨迹诊断。** 在看到新轨迹前固定validation8全部任务、init states `[0,12,25,37]`、原200/600两点，共64 episodes；复用各自原canonical correct400中的LoRA、teacher/state映射、source/normalization、environment/policy RNG。视频不重物化，不挑成功案例。原完整400结果仍是唯一正式成绩；新小面板只用于行为解释，无训练梯度、无checkpoint选择、无Test。

复用现有persistent evaluator、cost-balanced queue和真实policy输入/动作chunk捕获；仅新增显式登记的冻结reference重放入口，核对原run已完成、同policy/manifest、固定cases和read-only用途，不绕过原保护或扩展为新训练入口。记录每次replan的实际双相机/状态输入、预测action chunk，以及每个环境step的原BDDL目标谓词变化。目标谓词不是人工完整阶段标签，接触/抓取需结合真实画面与动作，不把未满足终局谓词直接命名为未抓住。

首先核对新重放与历史逐行的success、steps、teacher和noise一致程度；接受正常kernel/分组低位差异，不为复刻某次成功重复挑run。有变化的案例按本次真实轨迹解释，不能冒充原历史轨迹。该固定状态小面板不能估计每task总体成功率，也不能因行为阶段相同就定责某一个Writer模块。

要区分：200已有行为是否在600最初目标获取阶段消失；是否接近/抓住正确对象但放置、释放或后续子目标失败；还是在接触前就没有形成正确运动。对Spatial长期零、Object13已获取后丢失和Long组合缺口分别给出观察与不能推出的结论。若固定面板没有覆盖某种失败，不临时按新结果换状态来放大叙事。

**下一步的结果分支。**

| 新证据 | 后续动作 | 不能推出 |
|---|---|---|
| 出现可复现的采样/执行/信息接口合同违反 | 修复该明确缺陷，最小复验后重新判断原假设 | 所有低分都来自该bug |
| 训练FM有用的接口明确、行为主要在后期子目标丢失 | 沿可观察缺口分析训练侧动作支持与过程压缩，必要时做固定train-side局部诊断 | 直接判定occupancy或某个GRU故障 |
| 语言query残差与后端对动作拟合主导、视频P4作用弱且跨任务行为弱 | 优先准备一次只改compiler内容初始化/残差来源的fresh对照，保留视频条件共同语义Value及全部原生响应前端 | 当前已证明语言捷径或删除残差必会改善 |
| P4实际有功能作用，但仍无稳定迁移 | 保留其作用证据，分析从条件代码到native因子的共享/任务支持；rank共享需明确相对Target-Owned的新变量 | 仅因有梯度就宣布视频理解完成 |
| 两项诊断仍不分竞争机制 | 如实保留不可识别项，选择能区分解释的一项训练侧干预并预登记；不机械跑整个消融矩阵 | 把“未排除”改写为“已定位” |

正式架构对照前补全确切公式、参数初始化、fresh学习状态、仍保持的4×64数据/优化口径和匹配节点；不得同时改language path、D sharing和task组织。每段约一小时，以真实profile决定50/100倍数的中间/末节点，日志同时报updates、conditions、queries/每task曝光和墙钟。先获得行为证据，再决定相邻继续；目标/资格仍为§8.3。

本批轨迹与内部分析新增峰值预算8GiB；source、checkpoint和已有LoRA均引用canonical资产。2026-09-09 strg01实测data1 576,355,896KiB/soft1,073,741,824KiB，shared84TiB，原run38GiB；正式启动时刷新对应GPU现场。诊断原件位于 `k1_fresh/internal_diagnostic_20260909/` 与 `k1_fresh/behavior_replay_20260909/`，本节登记不是完成声明。

### 8.2.5 单变量fresh对照：保留首次语言检索，移除直接语言内容残差

**依据与待检验解释。** A/A2/A3共6144个固定train-side FM queries（含各臂正常基线），实际P4/cross均有作用；A3仅移除首层language残差使200/600 FM增加.026782/.028808，23/24及24/24变差，而仅移除首层language检索增加.000188/减少.000107，任务方向混合。B的预登记64条真实回放定位到错误对象/实例及组合执行混合缺口，并保留成功例和历史数值分叉。完整证据见[horizon_k1_frozen_diagnostics_20260909.md](horizon_k1_frozen_diagnostics_20260909.md)。
这些证据支持首次compiler内容路径是一个已确认被使用、且可单独干预的接口，但没有证明它就是泛化根因。下一轮检验：直接language残差是否让共享映射过多依赖容易拟合的任务语义内容；让task-conditioned内容经现有真实视频memory进入，是否在相同学习口径下改善未见任务能力及保持。反向可能是移除有效语义prior导致能力更差，本实验允许此non-pass。

**唯一主要变量与完整公式。** 记`e=e_target+e_rank`，`l=query_language(ell)`，原首cross及后续self/FFN组合为`F`。旧第一块是`F(e+l+Cross(LN(e+l),LN(P)+time,LN(P)))`，新第一块是`F(e+Cross(LN(e+l),LN(P)+time,LN(P)))`。其后的第二compiler block完全采用既有公式，输入为新第一块实际输出；不强行把后续query钳制为旧图数值。新模型中`l`只向首次cross的query加偏置，不直接进入该块残差内容。
原有language encoder、每组内部语言引导、真实native prefix、单probe/Meta、完整50H、四组过去局部/单向长程、K1集合memory、双cross、自注意力/FFN、target/rank identities、38-target rank16 A/B、每target/rank/side独立native D、source冻结全部保留。P4仍可含视频静态语义与动态内容，此项不把删除语言残差等同于证明video dynamic必要性。共享task-independent identities/biases允许继续学习，不增加second adapter、gate、额外loss或head。

**最近等价历史与新增变量。** 旧v5.2/v6已经使用语言query检索、视频Value和视频残差，但其原生读取、前端、decoder与优化配方不同；强分数是可行性边界，不能借来声称本轮已验证。旧semantic-address/Task-Grounded等弱结果也未在当前Horizon图中单独隔离这一处残差。本轮保留当前全部其它接口，首次隔离当前compiler的内容来源，不重命名复做旧整套方案。rank-sharing与24-task组织不混入。

**fresh与身份。** 从合法identity step0重建整个Writer/Meta及AdamW/scheduler/sampler/RNG，seed7；不加载200/600或任何profile学习状态。参数数量、初始化分布与随机创建顺序不变，source和public probe复用原定义。新增显式model identity `compiler_language_mode=first_query_only_v1`；当前runtime/物化入口拒绝未标记或旧模式的Writer checkpoint，旧架构只由原frozen worktree/commits与artifacts解释。即使张量shape相同也不得exact-resume旧权重。

**学习与评测。** 保持原四suite各随机一task、4conditions×64queries=256/update、各条件1/4（全task期望1/24）、teacher0–15/actions16–41跨episode、Meta端到端、纯FM、AdamW3e-5、warmup8与其它超参。仅物理microbatch按现场与真实profile调整，SUM归约/clip/一次optimizer step不变。以原约18s/update为依据预登记首段200更新，100/200完整checkpoint各自correct strict400；完整profile若显示无法接近一小时，只在看到新行为前调整为50/100倍数附近节点并记录原因。
100/200累计400/800条件、25600/51200queries，与旧同节点正式55/110比较task/state/video/RNG及per-task/suite/breadth/RGL/churn/J。新200补固定held-video train96（states32–35、46–49各一次）对比原200的52/96；固定held-FM仍是训练侧定位，不用于选点。训练新run root与所有新LoRA独立，不复用旧Writer物化条件；policy、dataset、tokenizer和source checkpoint继续复用canonical资产。

**裁决与后续。** 若实质扩展能力并保持相邻获取，继续原架构的新学习历程以判断稳定性；若只改善内部FM、不改善closed-loop，不称修复。两点弱但持续学习时结合训练侧与曝光继续观察，不用单点直接淘汰；经有信息量节点未形成新能力且无支持继续的证据，则本次内容移除non-pass，不通过LR/rank/seed小扫或立刻叠rank-sharing挽救。不再机械扩充冻结language/cross消融矩阵。资格、other与最终controls仍按§8.3，最终目标不降格。

**工程与资源。** 由现有`horizon.py`与`attention.py`承接这处内容/检索分离，不新增平行Writer。旧未执行24-task草稿已撤回，当前baseline仍4×64。正式前完成针对新接口、language与P4有效梯度、完整identity LoRA、checkpoint身份拒绝和分块重放的检查；从clean pushed detached运行面启动。profile不成为正式权重起点；现场quota、峰值估计、双节点GPU与命令登记在本轮独立launch contract，GPU/NUMA/P2P和exact-resume约束保持。

### 8.3 资格与最终controls

正式资格只认同一checkpoint的validation8×states0–49=400行，correct与same-task-other严格配对，
K1正式每task的50个init states对应全部50条合法teacher videos各一次，每臂各自整轮无重复；validation8共400个不同task-video条件。
复用`expert_manifold/video_schedule.py`中每task固定排列，correct用reference_demo_index(without_replacement)，other用canonical +17 mod50规则，逐行不同。
teacher seed为20260907；同seed/task/init state跨checkpoint、worker顺序、分片与恢复保持映射一致。旧per-ordinal独立随机抽样不是canonical schedule。
当前只实施K1。K1全部资格与因果验证通过后再登记K>1训练及测试；声称dynamic K时须真实覆盖相应cardinalities。

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

`writer/supervised.py`组织每condition的FM A/B cotangent和一次完整Writer反传；`functional.py`复用原生FM。
`learning_data.py`持久化task/video/query随机流与固定无梯度动作诊断；`training.py`维护直接AdamW更新及完整边界checkpoint。
旧joint orchestrator退役到Git及profile原件，RL数学/flow辅助代码不构成活动训练入口；独立RL阶段登记时重新审视并清理失效部分。

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

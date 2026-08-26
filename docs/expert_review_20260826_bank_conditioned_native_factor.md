已确认 `main@ed2883bdcbd63ddd767a6ad36596aaf30090e5f9` 可访问。以下审查严格锁定该提交及其可达历史。我完整读完了指定的 mandatory reading、1416 行专家原文、active design、README，并核对了当前 G3 代码、测试、配置和代表性历史提交。

我会明确区分三类信息：**代码和 Git 可直接核验的事实**、**仓库中固化但本次无法从 raw run 重跑的实验记录**、以及**我的独立判断和建议**。

# A. Executive verdict

## A.1 G3 真正卡在哪里

我的判断是：

> **G3 当前卡住的不是 Native-Factor bank 没有容量，也不是 G2 Program 完全没有信息，而是把“稳定的目标功能”表示成了“依赖当前 video bank 协方差的、不稳定 inverse dual/score 坐标”。**

更具体地说，当前架构混淆了两个对象：

1. **应当跨同任务视频相对稳定的对象**：最终 paired low-rank update、其 policy function、或 set-valued successful policy equivalence class。
2. **不应要求跨视频稳定的对象**：从某一条具体 candidate bank 中恢复该 update 的 minimum-norm query、positive/negative score、或 analytic dual。

后者不是功能本身，而是功能在当前 bank 上的一种坐标表示。bank 的候选分布、测度和协方差一变，这个坐标就可以大幅旋转，即使最终功能基本不变。

当前证据非常符合这一解释：

* same-task teacher effective update 的跨视频 cosine 仍约为 q/v/action-out=`0.873/0.866/0.884`；
* G2 Program flattened cosine 更高，约 `0.9971`；
* 但一个视频的 raw query 迁移到另一个视频时，q/v update 接近零，action-out 只有约 `0.086`；
* 逐 video analytic score 在训练视频可以被拟合到较高 cosine，却不能在 held video 保持最终 paired update；
* 50-task/98-condition 的 frozen-Program decoder 也表现为 train 可记忆、task/video holdout 基本失效。

因此我**基本同意当前仓库的工作假说**，但要增加两个限定：

* “约 \(10^6\) 条件数导致 score cosine 约 0.9 仍不能保留 factor”是高度可信的解释，但 raw score cosine 本身没有 quotient 掉 softmax 的常数 gauge、positive/negative 分解 gauge，也没有按 bank Jacobian 加权。要彻底坐实，应查看 score 误差在 covariance 特征谱各方向上的分解，而不能只看普通 cosine。
* `Program flattened cosine≈0.9971` 不能单独证明 Program 所有细粒度 event/owner 信息都充分，因为大幅度的 `P_lang/P_scene` 或冗余维度可能主导 flattened cosine。需要补看 `P_process/rho/tau/sigma` 的 owner/event 分解与 whitening 后稳定性。但现有证据已经足以说明：**Program 不是目前最早、最强的嫌疑接口。**

## A.2 Native-Factor 主线是否仍有合理成功路径

**有，而且目前不应终止。**

理由不是乐观推测，而是三项已经分离的证据：

* task-local rank16 oracle `250/400` 证明唯一 Action Expert LoRA 的输出合同有闭环容量；
* G1 `114/250` 证明真实 native X/Y bank 与 signed pooling 中存在 process-sensitive、Goal/Long 非零的 rank4 residual；
* G2 证明自然视频可以产生有动态增量、跨视频可对齐的 Program。

真正尚未通过的是：

$$
\text{Program}+\text{当前 video bank}
\longrightarrow
\text{正确 signed functional selection}
$$

而不是：

$$
\text{视频无信息}
\quad\text{或}\quad
\text{native bank 无容量}.
$$

G1、G2 的通过提交和当前 G3 non-pass 状态均能从 Git 历史核实；`ed2883b` 只更新了诊断文档，没有把任何新架构写进 active code，因此当前代码确实仍是 G3 v2。

## A.3 最推荐的下一动作

我建议下一步不是：

* 继续拟合逐 video dual/score；
* 把 key 从 512 再加宽；
* 调 teacher loss 权重；
* 再叠一个 direct factor decoder；
* 或直接恢复旧 realizer。

最推荐的是：

> **把 Pass B 改为 bank-conditioned、两阶段流式的 set-equivariant compiler：先累计当前 bank 的全局 sufficient statistics 和 Program-conditioned native anchors，再形成 bank-conditioned query；随后重放同一 bank，执行原有 exact signed softmax pooling。**

同时，在这一实现中保留一个明确的 `global_statistics_off` 消融，对“严格 one-pass、candidate-local functional canonicalizer”做最后一次决定性验证。它若通过，就保留更简单的一遍接口；它若失败，则正式淘汰严格 candidate-local 假设，而不是继续做 pointwise key 变体。

唯一 fallback 应是：**当显式协方差/低秩统计无法数值恢复 analytic reference，但 matrix-free covariance-vector product 可以时，改用 block-CG/Lanczos 的多遍流式求解。**

## A.4 我所理解的不可破坏合同

本审查采用的合同是：

* 部署输入只有 exact language 和 \(K\) 条同任务、action-hidden、内部有序的正确视频；
* Writer 只在 rollout 前运行一次，但 Writer 内部允许有多个只读子 pass；
* 部署不得读取 actions、state/proprio、reward、terminal、task ID、filename、pose、hidden normalization、policy outcome；
* 每个 condition 最终只产生一套覆盖 38 targets 的完整 rank16 LoRA；
* 不挑 video、不平均多套 LoRA、不融合 checkpoint、不加载第二 adapter/expert；
* full 必须相对 language/no-video、static/endpoints、wrong-video 形成必要增量；
* shuffled/reversed 只在最终 checkpoint 选定并冻结后测试；
* 固定 24 train / 8 validation / 8 test，validation/test 不产生梯度；
* 正式目标是 validation8 strict paired correct 严格 `>145/400`，并同时满足相邻 checkpoint 稳定、breadth、四 suite 非零、Goal/Long、retention、same-task robustness 和视频因果 controls。

---

# B. Historical reconstruction

## B.1 主要路线、因果问题与真正留下的结论

| 路线                                      | 它试图回答的因果问题                                                 | 核心输入、表示与监督                                                                                    | 真实结果与最早失败接口                                                                     | 当前仍有效的约束                                               |
| --------------------------------------- | ---------------------------------------------------------- | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------ |
| Frozen source / task-local oracle       | Action Expert LoRA 本身有没有能力？                                | source policy；每任务 privileged rank16 LoRA                                                      | source `48/400`，task-local oracle `250/400`，四 suite `73/78/58/41`               | 输出为唯一完整 Action Expert rank16 LoRA 是可行合同；瓶颈是共享生成        |
| Held5 基线与 successful members            | held task 上是否存在可实现的成功策略和可观察 policy effect？                 | source、shared carrier、直接 task experts/member lineages                                         | source `21/250`，carrier `43/250`，successful members 最终约 `113/250`               | 错误 residual 比不写更坏；policy effect 有信息，但必须 set-valued     |
| 早期 Writer 家族                            | language/video summary、memory、reward credit 能否直接生成更新？      | action-memory、belief、LOOM、CVADR、LMMPC/LPCP、layer-matched memory 等                             | 代表性 full/language/video/endpoints 约 `41/39/40/39`，Goal/Long 0；outer credit 甚至下降 | full-video 必须以闭环证明超越静态/语言；内部 loss 无权替代                 |
| GOMQ                                    | gradient-open memory query 能否用视频快速生成 task update？          | 与 ECP 不同的 memory/query 结构；历史 rank32/rank16 adapter                                            | rank32 correct `151/400`，rank16 重建 `136/400`；controls 有视频特异性但低于当前 `>145` 线      | 它证明过某种视频条件适配可能接近目标，但不证明 ECP；容量依赖明显                     |
| ECP Stage 0                             | PI0.5 native owner/layer/horizon observer 是否非退化？           | native probes、owner/event candidates                                                          | representation 非常量、可分离；Action Meta matched 结果中性                                 | observer 可复用；旧 q/v owner 不是完整 target-native X/Y        |
| ECP v1–v24                              | Program/privileged code 能否通过 hyperdecoder 生成 A/B 或 effect？ | deterministic/mean codes、free/fixed Program、A/B heads、direct surface、mapping-diverse compiler | 多轮 proxy 可改善但 held closed-loop 不广；v24 仍不能识别 own-policy mapping                  | 不能用更多 decoder 版本替代对最早接口的识别；latent mapping 欠定           |
| MDCO                                    | 扩大任务与 mapping 数量是否足以学会共享映射？                                | 90 tasks、540 visits、structured simulator                                                      | held candidate 约 `20/250`，低于 source21 和 carrier43，Goal/Long0                    | “更多 mapping 数”本身不解决 under-identification               |
| PECS                                    | 若 privileged trajectory effect 很精确，能否直接求解 residual？        | local/complete trajectory effect、精确求解                                                         | 两个裁决点约 `58/59`，breadth3/5、Goal/Long0                                            | effect matching 不等于完整闭环策略实现                            |
| fixed-A                                 | 是否可把任务更新限制在共享固定 A 行空间？                                     | fixed-A structured solver、successful member projection                                        | solver `78/250`；known-success 投影仅 `49/41/35`，Goal/Long0                         | fixed-A 在当前实现中是行为容量瓶颈，不能恢复                             |
| mobile-rank4 raw solver                 | 放开 A 后，短优化能否进入成功 basin？                                    | mobile rank4 raw factors                                                                      | 约 `49/250`，没有进入 known-success basin                                             | rank4 容量与 solver 可达性必须分开                               |
| balanced-SVD / fixed effect realizer    | policy effect code 能否经固定 inverse map 恢复 LoRA？              | fixed effect code、balanced-SVD realization                                                    | `33/37`，低于 carrier43                                                            | 跨 task 固定 inverse realizer 失败；不能恢复为当前 fallback         |
| centered two-sided span                 | 若允许正负 innovation span，高 cosine 是否能保留闭环？                    | centered two-sided fit-task coordinate                                                        | `80/250`，breadth3/5、Goal/Long0；update cosine 仍约 `.877–.960`                     | 高参数/update cosine 不保证低能量但行为关键方向保留                      |
| neural \(q_\pi\)、full-width FactorHeads | 能否先学 privileged latent，再直接吐高维 factors？                     | policy encoder、video encoder、latent、direct A/B/effect heads                                   | 缺少 canonical Program 标签，latent 可旋转；train 拟合不转化为 held mapping                    | privileged evidence 只作 functional critic；不再产生部署 latent |
| 人工 process / primitive / recovery       | 人工 opposite-order/minimal-pair 能否补齐可识别监督？                  | 手工任务、primitive/recovery expert、distillation                                                   | recovery Gate A `14/100`，相对 A3 gained0/lost23，执行检查正常                            | 路线科学 non-pass；owner 明确只用现成 LIBERO                      |
| Native-Factor G1/G2/G3                  | native bank 容量、Program 动态、共享 selection 能否逐接口成立？            | 真实 X/Y、signed pooling、Natural Program、rank12+4                                                | G1、G2 通过；G3 v1/v2 和 dual mapping 连续 non-pass                                    | 当前唯一未解决接口是共享、跨视频可泛化的 native selection                  |

这些历史结果及提交索引均被集中记录在 `research_history.md`；代表性历史 diff 进一步验证了 fixed-A、MDCO、PECS、fixed effect 和 two-sided coordinate 各自淘汰的是不同接口，不能被概括成“所有共享 Writer 都失败”。

## B.2 GOMQ 为什么是独立路线

GOMQ 不是 Natural Program 的早期版本，也不满足当前 G1/G2/G3 分解：

* 它没有当前固定 Program schema；
* 不以 38 个真实 target X/Y 的 exact signed pooling 为唯一 factor 来源；
* 历史强结果部分依赖 rank32；
* rank16 重建为 `136/400`，虽接近但仍未越过正式线；
* 它的 controls 表明视频信息可能进入更新，却不能回答 Native-Factor 的 bank-conditioned selection 问题。

因此它只应作为“视频条件参数适配可能达到较高闭环”的历史证据，不应在 G3 困难时被恢复为并行 fallback。

## B.3 为什么人工 process、primitive、recovery 路线终止

终止原因有两层：

1. **科学上**：recovery teacher Gate A 已是正常执行后的 non-pass，不是 worker、pairing、loader 故障。
2. **资源和 authority 上**：owner 已明确不继续制造人工任务、人工 controller trajectory 或额外场景，只用审计后的现成 LIBERO。

这意味着当前不能通过自制“更容易识别的过程任务”改变问题分布；G3 必须面对自然 same-task video variation。

## B.4 为什么内部指标不能代替 closed-loop

EMBER 历史上反复出现：

* loss 下降；
* effect 变近；
* update cosine 很高；
* latent 分离更强；
* reconstruction 更好；

但 Goal/Long 仍为 0，或闭环低于 carrier。centered two-sided span 在 `.877–.960` update cosine 下仍只有 `80/250`，就是最直接的反例。

因此内部指标的职责只能是：

$$
\text{定位最早失败接口}
$$

不能变成：

$$
\text{替代真实任务成功}.
$$

这一原则与 owner authority 一致。

---

# C. G1/G2/G3 audit

## C.1 当前 Native-Factor 基线的完整数据流

### Pass A：形成 Natural Program

输入是 exact language、每条视频的 ordered frames，以及两个固定、task-independent 的 antithetic probes。

输出固定为：

$$
\begin{aligned}
P_{\text{lang}} &\in \mathbb{R}^{38\times128},\\
P_{\text{scene}} &\in \mathbb{R}^{38\times128},\\
P_{\text{process}} &\in \mathbb{R}^{8\times38\times128},\\
\rho &\in \mathbb{R}^{8},\\
\tau &\in \mathbb{R}^{8\times2},\\
\sigma &\in \mathbb{R}^{8\times38\times128}.
\end{aligned}
$$

这里 \(E=8\) 是容量上限，不是每条视频固定有八个事件。实际 active event 数由 `rho`、分段 posterior 和 frame-to-slot assignment 决定。

每条视频先独立保序编码；K>1 时才进行 boundary-anchored monotonic canonical alignment。K=1 必须严格 identity，K 集合换序不得改变聚合结果。Action Meta 在当前 Pass A 路径中显式关闭。

### Pass B：真实 target-native banks

对 38 个 LoRA target 重读真实 linear input/output：

| target     | X input | Y output |
| ---------- | ------: | -------: |
| q_proj     |    1024 |     2048 |
| v_proj     |    1024 |      256 |
| action_in  |      32 |     1024 |
| action_out |    1024 |       32 |

输入候选严格为：

$$
n_A=(k,t,p,h)
$$

输出候选为：

$$
n_B=(k,t,p,h,u),\qquad
u\in\{\mathrm{abs,adj,init,goal}\}.
$$

其中：

$$
\begin{aligned}
Y_{\mathrm{abs}}(t)&=Y(t),\\
Y_{\mathrm{adj}}(t)&=Y(t)-Y(t-1),\\
Y_{\mathrm{init}}(t)&=Y(t)-Y(1),\\
Y_{\mathrm{goal}}(t)&=Y(T)-Y(t).
\end{aligned}
$$

X 没有 \(u\) 轴，也不能为了实现方便复制四份改变输入 attention measure。

当前输出侧还保留了 G1 证明必要的 native grouping：

* q output：8 个 query-head group，每组 256D；
* v output：1 个 256D group；
* action-in output：32 个 32D native block；
* action-out output：1 个 32D group。

chunked runtime 为每条视频单独维护 `previous/first/final`；跨 chunk 保持，跨视频严格重置。现有测试覆盖了 chunk/non-chunk 等价、Y_adj/Y_init/Y_goal 边界、X/Y 原生轴和 rank12+4 materialization。

### 当前 G3 signed compiler

当前代码中：

* key width 固定为 64；
* key 按 native width 共享，而不是 owner-specific；
* Program context 经一次线性层产生 positive/negative query；
* candidate key 是每个 X/Y 向量的逐点映射；
* query 在完整 bank 被读取之前已经形成；
* bank-global 交互只有 softmax 的标量 normalizer；
* pooled raw difference 之后无条件执行 `rms_normalize`；
* K videos 先分别池化，再做有界、置换不变聚合。

这说明当前代码没有隐含的 covariance、Gram、DeepSets summary 或 set-conditioned query。它确实是严格的 candidate-local one-pass compiler。

最终每个 target 的四个 \((a_{jr},b_{jr})\) outer products形成 rank4 update，经 small-core balanced SVD 后与 frozen rank12 carrier 拼接：

$$
A_{\mathrm{final}}=
\begin{bmatrix}
A_{\mathrm{carrier}}\\
A_{\mathrm{task}}
\end{bmatrix},\qquad
B_{\mathrm{final}}=
\begin{bmatrix}
B_{\mathrm{carrier}} & B_{\mathrm{task}}
\end{bmatrix}.
$$

仍是一套 38-target、76-tensor 的完整 rank16 LoRA，没有第二 adapter。

## C.2 G1 已经证明与没有证明的内容

### 已经证明

action-in native-block 修正后的 strict250 为：

* total `114/250`；
* per-task `35/31/45/2/1`；
* breadth `5/5`；
* Goal、Long 非零；
* `4/5` tasks 高于 carrier；
* carrier retention `35/43`；
* Action Meta module/parameter 为 0；
* 唯一 rank12+4 rank16 adapter。

因此 G1 证明：

> 对每个 held task，当前真实 native X/Y bank 中至少存在一个 signed-pooling 可达、具有闭环效用的 rank4 residual。

这排除了“native bank 完全不含 Goal/Long 需要的方向”。

### 没有证明

G1 使用的是 task-local free code、analytic projection 和 privileged member selection。它没有证明：

* Program 能确定选择；
* shared query/key 能找到该选择；
* 同一 mapping 可跨 task/video 泛化；
* optimizer 能从随机共享参数进入该 basin。

把 G1 解释成 shared Writer capacity 已被 active design 明确禁止。

## C.3 G2 已经证明与没有证明的内容

G2 authority 是 `c1493a1/macro_00000020`：

* full 相对 endpoints 动态 loss 改善约 `22.2047%`；
* probe `38/40`；
* median active events `4`；
* one-event `0`；
* K1 identity、K4 permutation、same-task 等合同通过。

它证明：

* frozen PI0.5 的自然视频观察中有可读出的时序信息；
* Program 没有退化成单事件；
* 多视频 alignment 可以在保持置换不变的同时保留顺序；
* full 中间帧相对 endpoints 有实际训练目标增量。

但它没有证明：

* Program 的每个 128D channel 对 LoRA selection 有自然语义；
* Program 足以生成 high-condition native dual；
* G2 loss 所要求的动态信息与闭环策略更新所需信息完全相同。

因此 G2 是必要资格，不是 G3 的充分条件。

## C.4 G3 v1、v2 的真实结论

### v1 macro10

五臂 strict250：

$$
43/42/38/39/40
$$

依次为 carrier/language/full/endpoints/same-task。

此外：

* breadth `3/5`；
* Goal/Long 为 0；
* carrier retention `32/43`；
* full-language `-4`；
* full-endpoints `-1`；
* same-task retention `84.2%`。

正式报告路径为：

`runs/analysis/pi05_ecp_shared_compiler_g3_gate_m10fresh_5140362_4770c5e_20260826/report.json`

190 optimizer steps 已足以证伪“只是在 warmup 后训练不够”；selection query/key 相对变化很小，scale 路径和全局 clip 主导。

### v2 macro5

五臂为：

$$
43/42/41/38/37.
$$

此外：

* breadth `3/5`；
* Goal/Long 为 0；
* retention `33/43`；
* full-language `-1`；
* full-endpoints `+3`；
* same-task retention `73.2%`。

所以 v2 也明确 non-pass。

### v2 teacher credit 为什么不是最终根因

固定 K1 条件的梯度分解为：

* teacher-selection gradient norm `0.3235`；
* 其它 selection gradients `21.8015`，约大 `67x`；
* teacher spectrum 与其它 scale gradient cosine `-0.989657`；
* teacher-only 反事实可优化。

代码也确认所有这些职责在同一次 backward 中相加；虽然 selection 和 scale/video 做了不同 clip，但多个 selection loss 仍共同更新同一 query/key 参数。

因此 v2 有一个真实的优化问题：

> direct mapping credit 在同一步中被旧 functional/scale credit 覆盖。

但隔离 teacher credit 后，跨视频 dual 仍然旋转，说明：

> **梯度冲突是 v2 的近端失败原因，不是 G3 的最深结构原因。**

## C.5 当前仓库结论是否有逻辑跳跃

总体上，当前 tracked 结论是克制的：它只淘汰“逐 video analytic dual/score 作为 shared label”以及已经测试的 raw/event/anchor/candidate-local 实现，没有宣称所有 content attention 不可能。这个范围是正确的。

我发现四个需要收紧的表述：

1. **Program `0.9971` 不是充分性证明。**
   应补 owner/event/field-wise 稳定性，尤其是 `P_process/rho/tau/sigma`，以及 whitening 后的相对变化。

2. **raw score cosine 不应作为主要几何指标。**
   softmax branch 的加常数不改变输出；positive/negative 分解也非唯一。应使用 measure-centered score、pushforward factor error 或 Jacobian-induced metric。

3. **同任务 teacher update 的 `.87` 不是完全相同。**
   cross-video consistency loss 不应强迫 student updates 相等，而应允许不超过 teacher-teacher dispersion 的合法变化。

4. **pointwise canonicalizer 尚未被数学上完全排除。**
   当前 512D 结果和 50-task decoder 已使它成为低概率路线，但因直接 factor-supervised、跨多任务/多视频、完全不使用 dual label 的最终测试尚未严格完成，仍值得做一次预注册的决定性消融。

---

# D. G3 root-cause analysis

## D.1 signed pooling 中的 bank covariance

对一个分支，令：

* candidate values \(v_n\in\mathbb R^d\)；
* candidate keys \(h_n\in\mathbb R^m\)；
* base measure \(\pi_n\)；
* query \(q\in\mathbb R^m\)。

简化写成：

$$
p^\pm_n(q)=
\operatorname{softmax}
\left(\log \pi_n \pm h_n^\top q\right),
$$

$$
z(q)=\sum_n
\left(p^+_n(q)-p^-_n(q)\right)v_n.
$$

在 small-logit 邻域：

$$
p^+(q)-p^-(q)
\approx
2\left(D_\pi-\pi\pi^\top\right)Hq,
$$

于是：

$$
z(q)\approx
2V^\top
\left(D_\pi-\pi\pi^\top\right)Hq.
$$

若 key 直接取候选本身或其近似线性表示 \(H=V\)，则：

$$
z(q)\approx 2C_{V,\pi}q,
$$

其中 \(C_{V,\pi}\) 是当前 bank 在测度 \(\pi\) 下的 centered covariance。

要产生稳定目标 \(z^\star\)，minimum-norm query 是：

$$
q_{\mathcal B}
\approx
\frac12 C_{\mathcal B}^{+} z^\star.
$$

因此即使 \(z^\star\) 跨视频稳定，若：

$$
C_{\mathcal B_1}\neq C_{\mathcal B_2},
$$

则：

$$
q_{\mathcal B_1}\neq q_{\mathcal B_2}
$$

完全是预期现象。

当 \(\kappa(C)\approx10^6\) 时，小奇异方向上的轻微 score 误差会被显著放大，最终 factor 可能接近正交。仓库中的 fixed small-logit exact replay 约 `.995–.999` 进一步说明：这不是“大 logits 下 softmax 非线性失效”，因为小 logits 已能表现 analytic factor。

## D.2 这里至少有四种 gauge

### 1. LoRA rank gauge

$$
A\mapsto RA,\qquad B\mapsto BR^{-1}
$$

不改变 \(BA\)。当前 subspace/update/small-core losses 已经有意识地处理这一 gauge。

### 2. softmax 常数 gauge

每个分支：

$$
\ell_n^\pm\mapsto\ell_n^\pm+c^\pm
$$

不改变 softmax。因此 raw score cosine 会把无功能差异的常数平移当成差异。

### 3. positive/negative 分解 gauge

给定 signed measure：

$$
w=p^+-p^-,
$$

通常有多组 \((p^+,p^-)\) 产生同一个 \(w\)。解析 solver 选出的分解只是一个代表元。

### 4. candidate null-space gauge

若 \(N\gg d\)，存在大量：

$$
\delta w\neq0,\qquad V^\top\delta w=0.
$$

这些 score/measure 差异不改变 factor。minimum-norm dual 依赖 solver 使用的度量和当前 bank。

所以逐 video analytic dual 不是一个天然可监督标签；它是一个强烈依赖 gauge fixing 的坐标。

## D.3 逐 candidate 函数为什么很难产生 bank-conditioned inverse

当前 query 在读取完整 bank 前由 Program 决定：

$$
q=q(P).
$$

key 是逐候选函数：

$$
h_n=\phi(v_n).
$$

softmax denominator 确实依赖整个集合，但它只提供每个 query/sign 的**标量 partition function**。它不能一般性地表达：

$$
C_{\mathcal B}^{-1}
$$

这样的矩阵算子。

一般的 permutation-equivariant set function 至少需要类似：

$$
s_n=f\left(P,v_n,\sum_m \psi(v_m)\right),
$$

即 candidate-local 内容和 bank-global summary 的共同作用。当前实现缺少最后一项。

严格来说，一个足够强的 pointwise \(\phi\) 有可能把不同 bank 映到近似统一、已白化的分布，从而不再需要显式 \(C^{-1}\)。这就是 bank-independent functional canonicalizer 的唯一合理解释。但现有证据已经表明：

* width-shared 64D key 不足；
* owner-specific 512D key 的 functional image 病态；
* event query、sparse anchor 不迁移；
* direct score label train 可拟合但 held q/v 失败；
* frozen Program dual decoder task holdout 几乎为零。

因此这一假设只值得一次严格、直接 factor-supervised 的淘汰性测试，不值得继续做宽度或 scorer 版本序列。

## D.4 对候选根因的排序

| 候选根因                                      | 我的判断                                                                   |
| ----------------------------------------- | ---------------------------------------------------------------------- |
| 监督标签选错                                    | **主要原因。** per-video minimum-norm dual/score 不是 stable functional label |
| candidate key 缺少 canonicalization         | **主要原因。** 当前 key 只做 pointwise direction+magnitude 映射                   |
| one-pass streaming 缺少 bank-global context | **主要结构原因。** query 形成时不可访问 covariance                                   |
| Program 信息不足                              | 暂非主要原因，但不能由 flattened cosine 完全排除                                      |
| exact signed softmax 设计错误                 | 不是当前首因；small-logit exact replay 已很强                                    |
| measure normalization 错误                  | 当前长视频 quadrature 与 per-video normalization 基本合理；需要保持                   |
| v2 mixed losses                           | 真实近端失败原因，但隔离后仍有跨视频几何问题                                                 |
| 无条件 `rms_normalize`                       | 重要放大器：低置信随机 selection 被变成全幅方向                                          |
| rank4 / carrier12                         | 当前无证据是 G3 首因，G1 已通过                                                    |
| Action Meta 缺失                            | 不是 G3 根因；当前应继续关闭                                                       |

## D.5 训练期、部署期与禁止路径的边界

| 计算                                       | 是否允许     | 边界                                                              |
| ---------------------------------------- | -------- | --------------------------------------------------------------- |
| FP64 analytic dual、teacher LoRA/factors  | 训练/诊断允许  | 不能成为 checkpoint 参数、forward 输入或部署 cache                          |
| verified member set-valued critic        | 训练允许     | member identity 不进入 Writer；一个 trajectory 由一个 global member 解释   |
| 当前视频 bank 的均值、协方差、Gram、谱、低秩 sketch       | **部署允许** | 它们只由 exact language/video 和 frozen policy native activations 导出 |
| 多次读取同一视频                                 | 允许       | 仍是 rollout 前一次 Writer 调用；不能在 rollout 中适配                        |
| authority_id/demo ID 用于 loader 配对        | 训练基础设施允许 | 不得进入模型 forward                                                  |
| task/video/frame embedding 查表            | 禁止       | 会形成 held dictionary                                             |
| Program 直接输出完整 A/B factors               | 不应恢复     | 等价于已失败的 full FactorHead/hyperdecoder 风险                         |
| fixed effect code → fixed inverse → LoRA | 禁止恢复     | 已被 realizer/two-sided 证据否定                                      |
| 第二 adapter 或 expert                      | 禁止       | 最终只能有一套 rank16                                                  |

---

# E. Recommended architecture

我建议把新接口称为 **bank-conditioned Native-Factor compiler**。它保留 ECP 主体，只修改 Pass B 中 query 的形成方式。

## E.1 保持不变的部分

以下全部保留：

* frozen G2 `c1493a1/macro_00000020` Program；
* exact language、ordered action-hidden videos；
* 两个固定 antithetic probes；
* Program schema；
* 每视频独立编码、monotonic alignment、K permutation invariance；
* 真实 38-target X/Y hooks；
* 输入无 type 轴；
* 输出 abs/adj/init/goal；
* q 8 groups、action-in 32 blocks；
* exact positive/negative softmax；
* per-video measure normalization；
* chunk 中的 previous/first/final；
* rank4 outer products；
* small-core balanced SVD；
* frozen rank12 carrier；
* 最终唯一 rank16；
* Action Meta 关闭。

## E.2 修改：从 pointwise query-key 改为 bank-conditioned native anchors

对 video \(k\)、target \(j\)、rank \(r\)、event \(e\)，构造 Program context：

$$
c_{jre}=
\operatorname{Concat}
\left(
P_{\text{lang},j},
P_{\text{scene},j},
P_{\text{process},e,j},
\rho_e,\tau_e,\sigma_{e,j},
E_j,E_r,E_e
\right).
$$

### 输入分支

候选：

$$
X_{j,n}\in\mathbb R^{d^{\text{in}}_j},
\qquad n=(t,p,h).
$$

先定义 video/target 级均值与协方差：

$$
\mu^A_{kj}
=
\sum_n\bar\pi_n X_{j,n},
$$

$$
C^A_{kj}
=
\sum_n\bar\pi_n
(X_{j,n}-\mu^A_{kj})
(X_{j,n}-\mu^A_{kj})^\top.
$$

这里 \(\bar\pi\) 是每条视频单位质量的基础测度；不按 event/rank 重复建立协方差，从而控制内存。

Program 与 candidate-local 内容只生成**有界标量 anchor compatibility**：

$$
g^{A,\pm}_{kjre,n}
=
\tanh f^{A,\pm}_j
(c_{jre},
\widehat X_{j,n},
t/T,p,h,
M_{k,t,e}).
$$

它不生成高维 factor。

形成 native anchor：

$$
a^{A,\pm}_{kjr}
=
\sum_e\alpha_{jre}
\sum_n
\pi_{e,n}
(X_{j,n}-\mu^A_{kj})
g^{A,\pm}_{kjre,n}.
$$

然后做 bank-conditioned solve：

$$
q^{A,\pm}_{kjr}
=
\left(C^A_{kj}+\lambda^A_{kj}I\right)^{-1}
a^{A,\pm}_{kjr}.
$$

最终 candidate logits：

$$
\ell^{A,\pm}_{kjr,n}
=
\left(q^{A,\pm}_{kjr}\right)^\top
(X_{j,n}-\mu^A_{kj})
+b^{A,\pm}_{jre,n}.
$$

再执行现有 exact signed softmax：

$$
w^A_n
=
\operatorname{softmax}(\log\pi+\ell^{A,+})_n
-
\operatorname{softmax}(\log\pi+\ell^{A,-})_n.
$$

最终：

$$
a_{kjr}^{\mathrm{final}}
=
\sum_n w^A_n X_{j,n}.
$$

### 输出分支

对每个 native output group \(g\)：

$$
Y_{j,g,n}^{u}
\in
\mathbb R^{d^{\text{out}}_{jg}},
$$

完全同样累计：

$$
C^B_{kjg},
\quad
a^{B,\pm}_{kjgr},
\quad
q^{B,\pm}_{kjgr}.
$$

最后对真实 \(Y^u\) exact pooling。

这意味着统计矩阵形状为：

* input：`C_A[k,j,d_in,d_in]`；
* q output：18 targets × 8 个 `256×256`；
* v output：18 个 `256×256`；
* action-in output：32 个 `32×32`；
* action-out output：1 个 `32×32`。

协方差可在一条视频内一次累计并在形成 query 后释放，不需要为 8 events 或 4 ranks 复制矩阵。

## E.3 为什么这个算子针对了当前证据

若 local compatibility 隐含一个跨视频相对稳定的 score function：

$$
g_n\approx \theta^\top (v_n-\mu),
$$

则：

$$
a
=
\sum_n\pi_n(v_n-\mu)g_n
\approx C\theta.
$$

所以：

$$
(C+\lambda I)^{-1}a
\approx \theta.
$$

这正好消除了不同视频 \(C\) 对 raw coefficient 的旋转影响。

换句话说，网络学的是：

> “什么样的 candidate content 与当前 Program 功能相关”

而不是：

> “这条具体视频的 minimum-norm analytic dual 数值是多少”。

## E.4 candidate anchor 网络应使用什么输入

建议 anchor 网络按 family 共享 trunk、按固定 owner/group 使用 FiLM 或 embedding，不按 task/video 建表。

输入最小集合：

* `P_lang[j]`；
* `P_scene[j]`；
* `P_process[e,j]`；
* `rho[e], tau[e], sigma[e,j]`；
* owner、family、rank、event、group embedding；
* candidate native direction 与 log-magnitude；
* normalized time；
* probe、horizon；
* output 分支的 `u∈{abs,adj,init,goal}`；
* frame-to-event canonical assignment。

不输入：

* authority ID；
* task index；
* video/demo index；
* frame absolute ID；
* member identity；
* teacher factors；
* actions/reward/state。

anchor 网络只输出两个 bounded scalar compatibilities及可选的 bounded relative group gain；高维 native 方向始终由当前 bank 的加权和产生。

## E.5 q output 和 action-in 的 group gain

最新解析证据显示，q 的 8 个 output groups 若各自单位化，会把 update 从约 `.999` 降到 `.967–.985`；保留 bounded relative group gain 才恢复强 update。

因此输出分支应采用：

$$
\gamma_{jgr}\in[0,1],
$$

配合一个共同 target/rank score scale，而不是每组独立强制 RMS=1。

action-in 的 32 blocks 同理保留独立 measure，但其幅度必须通过同一个 target/rank small-core 共同约束，避免 32 个隐式 adapter。

## E.6 confidence 与 `rms_normalize`

当前代码无条件把 pooled raw difference RMS-normalize。即使 signed selection 尚未识别，接近零的随机向量也会被放大为完整方向。

建议改为：

$$
\widehat z
=
\frac{z}{\sqrt{\operatorname{RMS}(z)^2+\epsilon}},
$$

$$
c_{\mathrm{id}}
=
\min\left(
1,\frac{\operatorname{RMS}(z)}{z_{\mathrm{ref}}}
\right)
\cdot
\exp(-r_{\mathrm{solve}})
\cdot
\sqrt{\frac{\operatorname{trace}(C_{\mathrm{retained}})}
{\operatorname{trace}(C)}}.
$$

最终 scale：

$$
s_{jr}
=
s_{\mathrm{ref},j}
\tanh(\hat s_{jr})
c_{\mathrm{id}}.
$$

其中：

* \(z_{\mathrm{ref}}\) 只能由 fit authority 冻结；
* \(r_{\mathrm{solve}}\) 是当前 bank 的相对线性求解残差；
* confidence 不得读取 held outcome；
* 低置信时应自然退回 carrier，而不是产生全幅随机 residual。

## E.7 数值正则化

建议：

* native activations detached；
* 均值/协方差 FP32 累计；
* solve 使用 FP64 或稳定 FP32 Cholesky；
* 固定相对谱 floor：

$$
\lambda
=
\max(\lambda_{\mathrm{abs}},
10^{-6}\lambda_{\max}),
$$

其中 \(10^{-6}\) 来自当前 formal conditioning 证据，不再作为 held sweep 参数；

* branch logits 在 measure 下显式中心化，去掉常数 gauge；
* operator residual 作为 hard qualification，不用 loss 掩盖不精确 solve。

## E.8 Pass B 的系统执行

```text
Pass A
  language + videos + probes
  -> frozen G2 Program

for each video independently:

  Pass B0: statistics / anchors
    stream native X/Y chunks
    maintain first/final/previous
    accumulate:
      mean
      covariance / Gram
      Program-conditioned native anchors
      per-video unit measure
    do not materialize full bank

  solve
    q_plus/q_minus = regularized bank-conditioned solve
    compute group gains and confidence

  Pass B1: exact pooling replay
    reread same native X/Y chunks
    rebuild abs/adj/init/goal exactly
    exact online positive/negative softmax
    pool real X/Y values

permutation-invariant K aggregation
rank4 outer products
small-core balanced SVD
concatenate frozen rank12 carrier
one complete rank16 adapter
```

这增加一次内部 native read，但仍是 rollout 前的一次 Writer 调用，不构成 task-local adaptation，也不读取禁用信息。

## E.9 为什么这不是历史 fixed realizer 或 FactorHead

它与旧路线有本质区别：

* 没有跨 task 固定 effect code；
* 没有 fixed inverse map；
* Program 不直接输出 \(d_{\text{in}}\) 或 \(d_{\text{out}}\) factor；
* 所有 native anchor 和最终 factor 都是当前视频真实 X/Y 的加权和；
* 协方差每个部署 condition 重新计算；
* 没有 fit-task PCA/span；
* 没有 task/video lookup；
* 最终仍是 exact signed pooling 和唯一 rank16。

## E.10 唯一 fallback

只有在以下条件同时成立时触发：

1. 完整 materialized FP64 operator 能达到约 `.995–.999`；
2. 显式 covariance/Cholesky 实现因内存、条件数或低秩截断不能恢复该结果；
3. 失败不是 anchor 网络泛化，而是 operator replay 本身。

fallback 使用 matrix-free：

$$
Cq=
\sum_n\pi_n
(v_n-\mu)
\left((v_n-\mu)^\top q\right)
$$

的流式 covariance-vector product，配合 block-CG 或 Lanczos。可以多遍重读同一 bank，不物化矩阵；固定相对 residual 和谱 floor。

若 matrix-free exact operator 仍不能恢复 analytic reference，则停止这一接口，不再做 width/seed/LR 变化。

---

# F. Experiments and Gates

## F.1 最小、有因果顺序的验证序列

| 阶段                                  | 数据                                                                  | 训练/冻结                                                    | 输出                                       | 推荐通过条件                                                                                        | 失败解释                                  |
| ----------------------------------- | ------------------------------------------------------------------- | -------------------------------------------------------- | ---------------------------------------- | --------------------------------------------------------------------------------------------- | ------------------------------------- |
| F0 信息墙与算术 smoke                     | 1 个真实 K1、1 个 K4 条件                                                  | 仅新 compiler 可训练；Program/source/carrier frozen            | native stats、query、单 adapter             | shape/finite；Action Meta0；无 ID 输入；K2/K4 teacher reads0；76 tensors；policy 实际消费                 | 工程合同错误                                |
| F1 bank-operator capacity           | 50 tasks/98 conditions 的 q/v/action-in/out representative authority | free anchor 或 analytic anchor；不训练 shared Program mapping | materialized 与 streaming operator replay | 每 family median `>=0.995`，minimum `>=0.99`；chunk 与 full 等价                                    | 若 analytic 强而新 operator 弱，是实现/正则化错误   |
| F2 strict candidate-local 消融        | 50 K1-covered tasks、451 task-video；预注册 task/video holdout           | `C=I` 或关闭 global stats；只训练 anchor scorer                 | paired factor/update                     | held-video oracle-normalized recovery median `>=0.75`，每 family `>=0.65`，task-holdout `>=0.60` | 不通过则正式淘汰严格 one-pass candidate-local   |
| F3 bank-conditioned mapping         | 同 F2                                                                | 开启 \(C^{-1}\)；冻结 Program/source/carrier                  | shared bank-conditioned query            | held-video recovery median `>=0.75`、p10 `>=0.50`；train/held ratio `>=0.8`；两个相邻 checkpoint 稳定  | train 高 held 低：functional seed/内容泛化失败 |
| F4 scale 与 functional qualification | 全部 75 fit tasks；teacher 仅 K1-covered 50 tasks                       | selection 受 mapping loss保护；scale/video 单独更新              | 完整 K1 adapter                            | teacher paired update不退化；functional、flow、preservation改善；低置信退 carrier                          | 若 selection 被旧 loss 覆盖，参数职责仍未隔离       |
| F5 K 恢复                             | K1→K2→K4                                                            | K2/K4 不读 teacher                                         | full/endpoints/same-task adapters        | K1 identity；K2/K4 permutation；bounded beta；same-task mapping retention `>=80%`                | 多视频聚合或视频边界问题                          |
| F6 held5 strict250                  | fold0 held5，所有 actions/reward 梯度0                                   | 单一 frozen checkpoint                                     | 五臂 strict250                             | 使用现有 G3 Gate                                                                                  | closed-loop shared compiler裁决         |
| F7 G4 两 folds                       | 至少两个 train24 folds                                                  | fresh joint Writer                                       | closed-loop checkpoints                  | 使用现有 G4 Gate                                                                                  | joint stability或整体因果问题                |

## F.2 数据切分

### Mapping acquisition

使用已经封存的：

* 50 个 K1-covered fit tasks；
* 451 个唯一 task-video conditions；
* 68 个 covered-task verified members；
* 662 个 teacher states。

建议预注册：

* 40 tasks 用于 mapping fit；
* 10 tasks 作为 fit-authority 内部 task holdout，按 meta/target role 和 suite 分层；
* 对具有多条视频的 condition，至少留出一条 held video；
* ID 只用于构造 split，不进入模型。

架构确定后，再从 fresh 使用全部 50 K1-covered tasks 训练，并只在正式 held5 做 G3 closed-loop。

### Functional phase

使用全部：

* meta56；
* target-fit19；
* K=1/2/4 natural videos；
* 现有 teacher actions、effects、successful members。

其中 K2/K4 不读取 native teacher factors。

validation8 和 Test8 此时仍完全不参与。

## F.3 mapping acquisition 的最小 loss

### 1. 主要 loss：set-valued paired update

对一个任务的 verified members \(m\)：

$$
L_{\mathrm{update}}
=
-\eta\log
\sum_m
w_m
\exp\left[
-\frac{
D_{\mathrm{family-balanced}}
(\Delta W_{\mathrm{pred}},\Delta W_m)
}{\eta}
\right].
$$

要求：

* 一个 logical multi-video condition 使用一个 global member responsibility；
* 不按 event 混合不同 member；
* q/v/action-in/action-out 等权；
* student scale 在 selection 分支 stop-gradient；
* 不监督 raw dual 或 raw score。

### 2. cross-video functional consistency

不直接要求：

$$
\Delta W(v_1)=\Delta W(v_2).
$$

而要求 student dispersion 不超过对应 teacher set 的合理 dispersion：

$$
D(\Delta W_{\mathrm{pred},1},
\Delta W_{\mathrm{pred},2})
\le
D(\Delta W_{\mathrm{teacher},1},
\Delta W_{\mathrm{teacher},2})
+\delta.
$$

### 3. spectrum/scale

只更新 scale 和 group-gain：

$$
L_{\mathrm{spectrum}}
=
\left\|
\log \sigma_{\mathrm{small-core,pred}}
-
\log \sigma_{\mathrm{teacher}}
\right\|_1.
$$

### 4. operator checks

以下应主要是 assertion/metric，而不是可被其它 loss 抵消的软目标：

$$
\frac{\|(C+\lambda I)q-a\|}{\|a\|},
$$

retained trace、effective condition number、chunk/full equivalence。

### 暂时删除或阻断

mapping acquisition 阶段不让以下 loss 更新 selection：

* old global-member effect；
* cross-episode flow；
* carrier preservation；
* same-task response consistency；
* direct analytic score；
* raw factor reconstruction。

这些职责可以在 F4 只更新 scale/video，或在 G4 joint 阶段重新引入。

## F.4 最小真实 forward/gradient/materialization 检查

在任何 formal launch 前，必须一次性验证：

1. source、Stage 0、G2 Program、carrier 全部无梯度；
2. Action Meta module/parameter 为 0；
3. teacher lookup 仅 K1，K2/K4 reads=0；
4. authority ID/demo/member 不在 compiler forward signature；
5. anchor scorer、group gain、scale 都有 finite nonzero gradient；
6. covariance solve residual finite；
7. stats pass 与 materialized reference 一致；
8. pooling replay与现有 non-chunk reference 一致；
9. previous/first/final 不跨视频；
10. K4 video permutation error处于正常浮点范围；
11. 输出恰好 38 targets、76 tensors、rank16；
12. frozen policy 实际加载和使用该 adapter；
13. checkpoint 不包含 task/video tables、teacher factors、analytic dual 或 per-condition covariance。

## F.5 K1/K2/K4 恢复顺序

严格顺序应是：

1. **K1 mapping acquisition**
   先排除多视频组合干扰。

2. **K1 held-video generalization**
   同任务不同视频必须保持最终 factor/update，而非 query。

3. **K2 uniform combination**
   每视频单独统计、单独 solve、单独 pool；先固定 \(\beta_k=1/K\)。

4. **K4 uniform combination**
   验证 permutation、长视频 measure、显存和 chunk。

5. **bounded K correction**
   只有 K1/K2/K4 资格均成立后才训练；不得形成 video selection。

## F.6 正式 G3 Gate

沿用 active authority：

* full `>=60/250`；
* breadth `>=4/5`；
* carrier retention `>=33/43`；
* Goal 或 Long 至少一个非零；
* full-language `>=+5`；
* full-endpoints `>=+5`；
* same-task retention `>=80%`；
* 相邻 checkpoint 不应出现大幅 success-set churn；
* shuffled/reversed 不进入此 Gate。

## F.7 失败后的最早接口路由

### 情况 1：analytic operator 强，新 streaming operator 弱

进入唯一 fallback：matrix-free solve。不要改 Program、loss 或 key width。

### 情况 2：candidate-local 消融失败，bank-conditioned 通过

正式淘汰严格 one-pass interface；修订 active design，保留两阶段 Pass B。

### 情况 3：bank-conditioned train 强、held-video/task holdout 弱

说明 local anchor/Program-to-functional-content mapping 仍在记忆 fit distribution。停止加宽；检查：

* Program field-wise sufficiency；
* candidate content normalization；
* task/video leakage；
* same-task teacher set 定义。

### 情况 4：mapping qualification通过，但 held5 full 仍低于 carrier

说明 teacher projected update 与 held closed-loop utility之间存在差距，或 scale/confidence/critic错误。先做 per-family closed-loop response counterfactual，不回到 dual label。

### 情况 5：full 低于 carrier或 breadth≤2，且 mapping、operator均通过

按照现有合同，应判断当前 source-unseen mapping 数据不足以支持共享 compiler；不能用 G4 joint training掩盖 G3 失败。

## F.8 明确停止条件

在以下条件同时成立后停止当前 G3 interface：

* bank-conditioned analytic operator通过；
* 无 task/video leakage；
* factor-supervised selection在训练条件可拟合；
* 两个相邻 checkpoint 的 held-video 与 task-holdout仍显著失败；
* 失败跨 q/v/action-out，不是单一 owner 或 numerical bug；
* matrix-free fallback不改变结论。

此时被淘汰的是：

> 当前 Natural Program + current native candidate content 足以识别 shared functional selection

而不是整个 EMBER 目标、LoRA 输出合同或所有视频条件参数生成。

---

# G. G4/G5/Final、Action Meta 与 loss 建议

## G.1 G4/Final 是否应 fresh 联合训练

**是。owner 的方向正确。**

G1–G3 的冻结是为了回答分离的因果问题：

* bank 有没有容量；
* Program 有没有动态；
* compiler 有没有 shared mapping。

这些组件 Gate 通过后，继续机械冻结会人为限制整体系统。

不过，“fresh”不应默认解释成丢弃已经通过的组件权重、全部随机初始化。更合理的是：

* fresh run；
* fresh optimizer/scheduler；
* 从通过 Gate 的 Program/compiler 参数初始化；
* backbone、carrier 继续冻结；
* 全部 Writer 组件联合可训练；
* checkpoint 重新按完整 closed-loop 选择。

完全随机重置已通过组件只能作为显式 ablation，而不应成为默认正式训练。

## G.2 Final 很可能没有目标 LoRA，应用什么监督

Final 不应假定每个授权任务都有 target LoRA。

更一般的监督是：

1. **cross-episode teacher action/flow**
   视频和 action query 来自不同 episode，阻断轨迹复制。

2. **set-valued functional equivalence**
   在 verified successful states 上比较 generated policy response 与多个成功 member。

3. **natural on-policy evidence**
   对 student 实际访问状态使用短 continuation、BDDL progress 和最终成功信号。

4. **视频因果与 same-task consistency**
   full 必须优于 endpoints/language，同时不同正确视频不应导致策略崩落。

G3 teacher factors 只用于 mapping warmup和组件资格，不是 Final 的永久数据依赖。owner authority 已明确这一点。

## G.3 最小充分 loss 集合

我建议正式 joint training 最终只保留四类主要职责：

$$
L_{\mathrm{joint}}
=
L_{\mathrm{behavior}}
+
L_{\mathrm{functional}}
+
L_{\mathrm{video}}
+
L_{\mathrm{preserve}}.
$$

### \(L_{\mathrm{behavior}}\)

授权 fit/meta teacher actions 的 cross-episode flow/action loss。

### \(L_{\mathrm{functional}}\)

whole-trajectory、single-member、set-valued policy equivalence。

### \(L_{\mathrm{video}}\)

包含：

* same-task video consistency；
* full 相对 endpoints 的动态 margin；
* language/full 的必要增量约束。

不使用 shuffled/reversed。

### \(L_{\mathrm{preserve}}\)

carrier/source 的 guardrail，防止错误 residual 大面积破坏已有 success。

## G.4 哪些旧 loss 应删除或只用于 warmup

### 删除

* direct analytic dual loss；
* direct score cosine loss；
* task/video classification；
* raw A/B reconstruction；
* hidden separation 作为优化目标；
* 多个重复表达同一 factor 误差的 reconstruction loss；
* 无机制证据的 scale/norm matching。

### 只用于 G3 warmup

* input/output subspace teacher loss；
* paired teacher update；
* small-core teacher spectrum。

一旦：

* held-video mapping qualification连续两个 checkpoint 通过；
* G3 小型 closed-loop 不退化；

这些 teacher-factor loss 应退出，而不是永久叠加。

### 作为监控或 conditional anchor

* G2 action/progress temporal diagnostics；
* Program proximal/EMA；
* operator whitening residual；
* probe robustness。

只有 joint training 实际破坏相应组件时才进入优化目标。

## G.5 warmup 的退出条件

warmup 不应按固定 steps 退出，而应同时满足：

1. G2 dynamic Program Gate 在两个相邻 checkpoint 保持；
2. G3 held-video mapping qualification保持；
3. K1/K2/K4信息墙通过；
4. 小型 closed-loop screen 不低于 carrier；
5. teacher-factor gradient不再是产生 selection 的唯一来源。

满足后 teacher subspace/update/spectrum loss衰减到 0。

## G.6 如何防止 joint training 破坏已通过组件

* 每个 checkpoint 同时跑 G2 Program panel、G3 mapping panel 和 closed-loop screen；
* checkpoint 只有在三者都满足时才具有选择资格；
* 先观察真实 gradient conflict，再决定使用更低 Program LR、proximal、EMA、alternating updates 或 gradient projection；
* 不预先永久冻结 Program；
* 不以训练 total loss 最低选 checkpoint；
* 监控相邻 success-set Jaccard，而不只看总分。

现有 G4 authority要求至少两个 folds：

* oracle-normalized recovery `>=0.40`；
* breadth5/5；
* Goal/Long 均非零；
* carrier retention `>=75%`；
* same-task retention `>=85%`；
* 相邻 checkpoint 分差不超过10、Jaccard `>=0.75`；
* full 相对 language/endpoints 为正。

## G.7 G5 outer credit 应何时启动

只有出现以下自然 on-policy 证据才启动：

* G4 已有 broad、Goal/Long 非零的真实成功；
* full 已稳定高于 carrier/language/endpoints；
* student rollout 中出现可复现的特定失败状态；
* verified expert member 在这些 student-visited states 上的固定短 continuation：

  * 最终成功，或
  * 明显提高 BDDL progress，
  * 且不撤销已完成 predicate；
* 同一状态下存在足够的正/负对照，可估计 credit，而不是只看到 rollout failure。

outer credit 只能更新：

* event posterior；
* Program；
* rank attention；
* target scale。

不应直接对高维 A/B 输出 unrestricted gradient。

现有 G5 Gate 要求相对 G4 净增至少10，且 breadth、Goal/Long、same-task retention 不下降。

## G.8 Final fresh 与 Test8

建议顺序：

1. 先解决 owner 文档中的 Final 数据 authority 冲突；
2. 使用全部授权 meta tasks + train24 做 fresh joint run；
3. validation8 只评预注册的相邻 checkpoints；
4. selected checkpoint 必须 strict `>145/400` 且满足全部 breadth/retention/control；
5. 冻结 selected checkpoint；
6. 此时才跑 shuffled/reversed；
7. 方法完全冻结后打开 Test8；
8. Test8 结果不得再用于架构、loss 或 checkpoint 修改。

## G.9 Action Meta LoRA

### 最合理加入时机

不是 G3，也不是 G4 初始阶段。

只有当：

* base Native-Factor compiler 已通过 G3；
* G4 有明确闭环增量；
* q/v mapping 已较强；
* 剩余误差明确集中在 action-in/out 的短程控制、动作幅值或 transition bandwidth；

才做 matched ablation。

### 它可能解决的残余误差

最合理的机制目标是：

* native Action Expert observation/readout 对控制细节的带宽不足；
* action block 的局部变化不能由现有 frozen probe/state 表示；
* 不是补救 Program→native mapping 失败。

### 部署合同

Action Meta 必须满足以下之一：

* 只存在于 rollout 前 Writer 的 observation path，rollout 时不加载；
* 或其作用被解析折叠/蒸馏进同一套 38-target rank16 adapter。

若它在 rollout 时作为第二套 LoRA 与 generated adapter 同时加载，则违反合同。

### 最小 ablation

同一 base Writer、相同数据、相同 checkpoint budget：

* Action Meta off；
* Action Meta on；
* Program/compiler 其余完全 matched。

报告：

* held5/validation8 strict；
* per-suite；
* breadth；
* retained/gained/lost；
* full-language/endpoints；
* same-task；
* 相邻 checkpoint 稳定；
* 最终唯一 adapter inventory。

### 永久关闭条件

满足任意一项即永久关闭：

* 无相邻 checkpoint 可复现的净增；
* 改善小于预注册的实际意义阈值；
* breadth、Goal/Long 或 retention 下降；
* full-video causal margin 下降；
* 只有内部 action loss 改善，closed-loop不变；
* 依赖 mandatory loader、第二 adapter 或禁用信息。

---

# H. Risks, conflicts and missing evidence

## H.1 文档—代码—历史冲突或滞后

### 1. `progress.md` 顶部 evidence 标记滞后

顶部仍写最新 G3 formal evidence 为 `e7d86b0`，但正文与 canonical `ed2883b` 已包含更晚的 cross-video score diagnosis。它是 header 更新遗漏，不应覆盖正文。

### 2. README 和 `concept.md` 的当前状态滞后

README 仍把 G1 写成“下一阶段/等待许可”，而 G1、G2 已通过，G3 已运行多轮。它们可作稳定概念文档，不能作动态 authority。

### 3. active design 顶部 provenance 文字滞后

顶部仍写当前仓库在 `6fdaeb8` 后没有新科学结果，但正文已增加 G1–G3 后续结果。这是文档头部未同步，不影响后部明确的 G3 分叉。

### 4. 当前 code 与最新诊断有意不同步

当前 `shared_compiler.py` 仍是 v2：64D pointwise key、Program-only query、无 bank-global stats。`ed2883b` 没有把任何新 architecture 写入代码。这不是隐性冲突，而是当前仓库明确暂停实现的状态。

### 5. world-size authority 漂移

现有 v2 formal surface源于 world1/2 配置，而最新 owner authority 要求后续 formal 可在1–6卡弹性分片，同时保持全局 task group、role 权重和 optimizer cadence不变。历史 run 不因此失效，但新实现不能继续把 world2 写死。

### 6. Final 数据 recipe 尚未裁决

owner requirements 提到方法冻结后 fresh 使用32 tasks并评测 Test8；active design又记录71 meta+train24 development/final recipe。`findings.md` 已承认这是 unresolved authority。Final launch 前必须明确：

* 32 tasks 的准确组成；
* validation8 是否进入 final refit；
* 71 meta 的角色；
* 如何保持 Test8 sealed。

在 owner 裁决前不能静默选择任一解释。

## H.2 本次无法从远程 Git 独立验证的 raw evidence

仓库中的 aggregate 和 commit 可核实，但以下结论仍依赖 ignored raw runs。

| 所需 artifact                                                                                                                            | 需要的字段                                                                                                                                                         | 原因                                                                       |
| -------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| G1 final action-in-block formal 的实际 run root                                                                                           | resolved config、run contract、五 task checkpoints、38-target adapters、per-row strict results、worker/shard manifest                                               | 独立重算 `114/250`、retention、unique adapter 与 Action Meta0                   |
| `runs/outputs/pi05_ecp_natural_program_g2_boundary_fold0_m10_c1493a1_gpu02p0123_r4_20260825/`                                          | `macro_00000020` state、held20 rows、full/endpoints Program tensors、alignment matrices、rho/tau/sigma                                                            | 验证 `22.2047%` 并排除 flattened cosine 被静态字段主导                               |
| `runs/analysis/pi05_ecp_shared_compiler_g3_gate_m10fresh_5140362_4770c5e_20260826/report.json` 及对应 run/checkpoint                      | per-arm row IDs、RNG、adapter manifests、per-task success sets、parameter/gradient logs                                                                           | 独立复核 v1 macro10 五臂与 churn                                                |
| G3 v2 clean detached `2a7f760` 的实际 run root；tracked docs只给出 `pi05_ecp_shared_compiler_g3_*strict250_native_teacher_v2*m5*20260826/` 模式 | teacher root manifest、451 condition index、662 states、checkpoint、per-loss gradients、strict rows                                                                | 验证 v2 score、teacher coverage与信息墙                                         |
| `runs/analysis/pi05_ecp_g3_dual_basis_four_family_loto_e7d86b0_gpu01p012345_20260826/`                                                 | candidate measures、covariance spectra、FP64 dual、basis projectors、task/video splits、per-family replay                                                          | 验证 `<=128` basis non-pass和384–512维需求                                     |
| 最新 cross-video mapping diagnosis 的实际 analysis root                                                                                     | 三视频 raw X/Y或充分统计、Program各字段、teacher factors、analytic scores、\(C\) 的 eigenvalues/eigenvectors、2000-step checkpoints、train/held split、per-eigenmode score error | 这是验证“score误差是否集中在小奇异方向”和排除泄漏的关键；canonical文档未记录实际 root                    |
| `.codex/tmp/g3_v2_fixed_probe_meta9_*`                                                                                                 | 固定 condition 输入、每项 loss 单独梯度、teacher-only trajectory                                                                                                          | 当前 tracked 文档只固化 aggregate；需要原始梯度确认 `.3235/21.8015` 与 cosine `-0.989657` |

特别是最新 cross-video 诊断，我最需要的不是完整巨大 native bank，而是以下压缩字段：

$$
\pi,\quad
C,\quad
\operatorname{eig}(C),\quad
q_{\mathrm{analytic}},\quad
s_{\mathrm{analytic}},\quad
s_{\mathrm{pred}},\quad
V^\top w(s),\quad
\Delta W.
$$

有了这些即可判断：

* 普通 score cosine 是否被 softmax gauge 污染；
* 误差是否确实落在高增益小奇异方向；
* ridge/truncation 能否稳定修复；
* bank-conditioned sufficient statistics 应采用全矩阵、低秩、对角还是 matrix-free。

---

# 最终裁决

EMBER 当前不应恢复任何旧主线，也不应直接进入 G4。

最合理的下一轮研究合同是：

1. **保留 G1/G2、真实 native banks、exact signed pooling、rank4、small-core SVD、carrier12 和唯一 rank16；**
2. **将 Pass B 改成先统计当前 bank、再形成 query、再重放 exact pooling 的 bank-conditioned compiler；**
3. **在同一实现中用 `global_statistics_off` 做一次严格 candidate-local factor-supervised 消融；**
4. **先通过 operator replay、held-video/task-holdout mapping qualification，再恢复 K2/K4 和功能职责；**
5. **最后回到现有 held5 五臂 strict250 Gate；**
6. **若 bank-conditioned operator 与 shared mapping均通过而 closed-loop仍失败，再定位 Program-to-function 或 teacher-to-utility，而不是继续修 dual。**

这条修改直接针对最新证据揭示的 covariance/gauge 问题，同时没有重新引入 fixed realizer、FactorHead、task lookup、第二 adapter 或禁用部署信息。

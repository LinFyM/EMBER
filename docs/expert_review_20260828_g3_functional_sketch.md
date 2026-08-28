# 1. 一页以内的结论

## 我们现在到底卡在哪里

EMBER 当前 **不是卡在 native bank 没有容量，也不是卡在 current-bank inverse 不可计算**。F1 已经证明：对真实 q/v/action-in/action-out bank，只要给出合适的 anchor，B0 statistics、谱截断 solve、group gain 和 B1 exact signed pooling 几乎可以无损恢复解析上限；F2 又证明，关闭 current-bank covariance、只保留 candidate-local first moment 后，完整 451-condition recovery 降到约 `.02`。所以“必须读取 current-bank global context”已经成立。

真正尚未解决的是：

> **如何让一个跨任务共享、真正受 Program 控制的映射，在当前 video bank 中取得 task-specific functional selection，而不是只取得 common residual、bank-common content 或某个 per-condition inverse oracle。**

这一判断来自最稳定的现象：从初版 current-bank F3 到 subspace credit、family/owner ownership、stable anchor、owner-query FiLM、joint compatibility、full-Program functional anchor和 fresh IEEE，fit、held-video、task-holdout 基本总在同一数量级一起低；它不是先学会 fit 再泛化失败，而是共享映射连训练条件自身都没有取得足够的 task-specific update。尤其 q/v 长期只有约 `.01–.11`，而 action-in/out 的偶发改善经常主导 overall。

## functional-polar 是否抓住了主因

**它抓住了一个真实且重要的子因，但没有抓住当前全部主因。**

它正确揭示了：普通 Euclidean query-key 坐标没有按照真实算子

$$
J_r=C_rC_0^+H
$$

的 metric 定标。task-local functional-polar 在 task93 的深层 q/v/action families 上接近 `.996/.999/1.000/.998`，说明真实 bank、post-\(W_k\) keys、rank-specific replay measure 和 signed pooling 的组合函数类中确实存在强方向。

但这个结果只证明：

$$
\forall\text{ condition }B,\quad
\exists\text{ 一个 bank-conditioned functional coordinate}
$$

并没有证明：

$$
\exists\text{ 一个 shared }f_\theta(P,B)
\quad
\text{能在新 task/video 上找到它。}
$$

它与 G1 free-code 的边界类似：都是非常有价值的**条件内 representability upper bound**，不是 shared mapping 证明。v4 进一步把每个 condition 的 \(C_0\)、四个 \(C_r\)、八个 event 的 native/key cross-images 和 polar/SVD 全部搬进部署 Writer；这在信息墙上合法，但在科学含义上接近“把 per-condition inverse solver 作为 Writer 的一部分”，在系统上也已被真实吞吐否决。

所以在用户给出的 A/B/C 之外，我选择：

> **D：B + C，并保留一个条件式 A。**
> full functional-polar 当前应降级为 fit-only privileged teacher、容量 oracle 和诊断 reference；functional coordinate mismatch 是真实问题，但不是 shared Program acquisition 的充分解。只有一个 bank-adaptive、低维、数量级更快的数学近似先通过容量与吞吐 Gate，才能重新成为 deployment 候选。

## 当前路线应该继续、简化还是改变

**继续 Native-Factor 主线，但改变 G3 的执行层和资格口径。**

应继续保留：

* G2 Natural Program；
* 真实 38-target X/Y；
* input 无 type 轴；
* output 的 abs/adj/init/goal；
* q 八组和 action-in 32 blocks；
* exact positive-minus-negative softmax；
* rank4、small-core SVD；
* frozen carrier12；
* 唯一完整 rank16；
* Action Meta 关闭。

应改变：

1. 不再把 full 233-bank polar 放进每次部署 forward；
2. 不再只用 absolute recovery `.75/.50` 判断 task-specific mapping；
3. 不蒸馏逐 video raw dual 或 raw scores；
4. 先建立轻量、current-bank set-conditioned student；
5. 用 universal-centered 和 Program×bank 交叉干预证明 task-specific causal ownership。

## 下一步最值得做什么

最值得做的不是启动 v4 formal，也不是继续优化 CUDA kernel，而是按以下顺序：

1. **一次无训练的低维 bank-adaptive functional sketch 容量与吞吐 Gate**；
2. **一次 12-task、带 task-local正控和 Program×bank 交叉因果对照的轻量 student Gate**；
3. 前两项都通过后，才重新启动完整 451-condition F3，并使用加强后的因果 Gate。

---

# 2. 证据审计

## 2.1 `ed2883b..9b52e59` 的结构—证据—归因链

该区间共 33 个提交。总体上，它们不是简单超参扫描；大部分结构改动有上一轮 formal 或固定-bank 诊断作为直接依据。但从 stable-anchor 以后，若只看 overall recovery，容易高估 action-in 带来的局部改善，并忽略 q/v 和 Program 因果所有权一直没有解决。

| Git 阶段                                            | 实际修改与结果                                                                                 | 独立审计                                                                                                     |
| ------------------------------------------------- | --------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `ea8bb226` → `435cb4a` → `c6d047f6`               | 接受 bank-conditioned 两阶段 Pass B；隔离 F1 operator。四 family median/min 均约 `.9995–.99996` 以上  | **归因成立。** current-bank stats/solve/B1 在 analytic anchor 下不是瓶颈；但不证明 shared mapping                        |
| `82dfb94f`                                        | 建立 v3 mapping compiler、329/40/82 split、role-balanced schedule、F2/F3 Gate                | 结构与信息墙合理；当前 head 已删除 v3 config，应按历史提交理解，不能把 v3 当当前实现                                                     |
| `917287a4` → `f8d285a1` → `19b5b3f` → `2199a76`   | F0 从 raw factor-slot 比较修正为有效更新比较；chunk/non-chunk cosine约 `.99999976`                    | **正确的工程纠偏。** small-core SVD 有 rank gauge，raw A/B slot 不能做功能 Gate                                         |
| `c1e26ce`                                         | F2 \(C=I\) 完整451条件约 `.022`                                                              | **强负证据。** candidate-local first moment 不足，且 fit 本身就失败，不是泛化问题                                             |
| `2a210365`                                        | 初版 current-bank F3 macro10：fit/held/task-holdout约 `.0899/.0897/.0968`；q/v `.0133/.0528` | **归因成立。** covariance 有增量，但 absolute acquisition 才是首因，不应续长训                                               |
| `84903aad`                                        | 增加 input/output subspace 等权 credit；held约 `.0730`                                        | 诊断“update-only credit starvation”合理；formal 证明 subspace credit **不是充分修正**                                 |
| `c3fc8e39`                                        | family-shared trunk + fixed-owner FiLM；held约 `.0746`                                    | family/owner 共享竞争确实存在，但拆 ownership 不够；compressed key 的稳定谱也不足                                             |
| `41171171`                                        | 删除 compressed key，尝试 direct-native anchor                                               | 正确发现其代数上会退化为 raw-query transfer；只完成 F0 而未浪费 formal，这个停止是合理的                                              |
| `4c0e72f8` → `20acc338`                           | task-stable `P_lang` anchor + per-event feature gauge；held提高到 `.1421`                   | **真实正证据。** bank gauge 与 task-stable code 有帮助；但 q 仍约 `.030`，不能说 shared selector 已解决                       |
| `7e232b03` → `d64f7ade` → `3e4e9a09` → `f2e1ecb4` | fixed-owner/group query FiLM；生命周期 `_apply` bug修复；held `.1631`                           | 工程修复正确；科学增量几乎主要来自 action-in，q/v仍`.032/.112`，所以“owner query路径已接通”和“q/v mapping成立”必须分开                     |
| `a2a56a7b` → `e784eb9d` → `55710bbd` → `ce56017d` | additive joint compatibility、antithetic 初始化；held约 `.1287`                               | joint path 有梯度但 path ablation 显示实际输出仍由旧 dot path 拥有；P_lang-only query近 task-agnostic。formal non-pass归因成立 |
| `5f61b0ba` → `3062de89` → `abff0a7a`              | full Program、target-native basis、fit-only consensus、primary bilinear；held约 `.08275`     | 发现两项更早问题：universal rank4 shortcut 与 pointwise canonicalizer 不具 task ownership。这个归因比“网络不够强”更可信            |
| `78b7e587` → `766a81b7`                           | 关闭 TF32，强制 IEEE FP32；fresh复评仍约 `.08313`                                                 | **必要数值修复，但不是根因修复。** 旧 checkpoint 不能 post-hoc 救回，fresh non-pass很关键                                        |
| `8e1eb878`                                        | 实现 full functional-polar v4                                                             | 数学上针对实际 operator metric；但仍只是待资格实现，不是新F3结果                                                                |
| `9e63f271` → `e139bc7e`                           | 加入 40-task correct-vs-wrong Program Gate 并平衡评测负载                                        | 加因果 Gate 的方向正确，但当前口径仍不足，详见下文                                                                             |
| `da3fd3e0`                                        | 单次 X/Y capture、同shape polar合批、IEEE FP32、thin-QR small SVD、per-event/global模式            | 属于有效执行优化，不改变科学变量；完整 diff 没有添加第二 Writer 或 fallback                                                        |
| `9b52e59f`                                        | 真实 profile 后判定吞吐 non-pass，无 K4 F0、无F3、无checkpoint                                       | **裁决正确。** 29%端到端加速不足以消除结构性成本，不能靠多GPU掩盖                                                                   |

## 2.2 已经可靠成立的结论

### 可靠结论一：current-bank global context 是必要的

F1 和 F2 构成了很干净的因果对照：

* full current-bank operator 可近乎无损；
* \(C=I\) first-moment 路径 fit/held/task-holdout 全部约 `.02`。

所以 candidate-local pointwise score 加 softmax denominator 并不能隐式替代 bank covariance/global context。

### 可靠结论二：旧 F3 首先失败在 absolute acquisition，不是 shared generalization

初版 F3、subspace、family-owner、stable-anchor、query-FiLM 和后继版本普遍表现为：

$$
\text{fit}\approx\text{held-video}\approx\text{task-holdout}\ll \text{Gate}.
$$

这排除了“模型只记住 fit task/video，然后在 held 崩溃”作为主要解释。

### 可靠结论三：candidate/functional coordinate 确实有问题

compressed key 的深层 q/v compatibility image 在稳定谱内明显失容；普通 Euclidean query 即便放宽谱也对深层 target 不稳定。actual-operator polar witness 又能在同一 bank 接近1。这证明“当前 candidate chart/query metric 错”是实际问题，不只是梯度或 loss。

### 可靠结论四：TF32 是严重错误，但不是最终根因

TF32 在约 \(10^6\) 条件数下产生 `.52–.68` 量级误差，而 IEEE FP32 与 FP64 很近；fresh IEEE F3仍约 `.083`。因此以后 native dual/score path 必须保持 IEEE，但不能再把 IEEE 当成待释放的巨大性能变量。

### 可靠结论五：现有 F3 absolute Gate 被 universal residual shortcut 污染

fit-only task-independent universal rank4 本身在 held-video/task-holdout达到约 `.825/.835`，已经超过旧 `.75/.50` Gate，却完全不证明 Program 或 video task selection。因此旧 primary recovery只能是容量条件，不能是因果充分条件。

### 可靠结论六：当前 v4 执行结构不合格

current v4 对每个 input/output native group保存 base covariance、四个 rank-specific replay covariance、八个 event cross-image，再进行 global/per-event polar。38个 input groups加195个 output groups共233组大算子。优化后单K1 condition仍约58秒，其中 polar约14秒、feature whitening约7.2秒、B0 solve约5.8秒、B1 replay约2.7秒；full cache和移除 checkpoint 均 OOM。没有新 formal 结果。

## 2.3 存在过度解释或仍未回答的结论

### “Program 已经足够”——尚未证明

G2 证明的是：

* Program保留动态；
* full相对endpoints有训练目标增量；
* event不坍缩；
* K对齐和置换合同成立。

它没有证明 Program 足以唯一确定 native factor selection。stable `P_lang` anchor带来真实提升，full `rank_event` 对 wrong task 也有一定分离；但 full-Program F3仍低，wrong Program对旧模型影响很小。因此目前最稳妥的判断是：

> Program中存在可用 task/dynamic 信号，但当前 mapping/credit 没有取得它；Program是否包含足够的闭环策略细节仍需交叉因果实验。

G2代码确实保持每视频独立编码、Action Meta显式关闭、K1 identity和uniform K aggregation。

### “functional-polar 证明 shared mapping函数类成立”——过度解释

task-local witness证明的是 bank-local operator image，不是 Program-to-query 的 task-unseen learnability。当前 polar本身由当前 bank 的 \(C_0,C_r,H\) 构造，会主动把 raw query投到条件内最有功能意义的像空间；这已经承担了大量 per-condition inverse geometry。它比 G1 free logits约束更强，但仍属于条件内 oracle，不能越过 G1/G3 的科学边界。

### “universal rank4 应立即并入 carrier12”——证据不足

universal component 确实说明当前 teacher/Gate含有很大公共修正，但把 `carrier12+universal4` 重压回 rank12 后，代数差分得到的 task residual 在真实 q/v native bank中可达性只有约 `.828/.765`。因此：

* **已证明**：teacher target和Gate受到 common residual污染；
* **未证明**：现有 carrier12 应立即替换；
* 新 carrier 必须从完整 expert-minus-new-carrier 重新进行 native projection/free-code和closed-loop验证，不能直接减 factor。

### “当前 correct-vs-wrong Program Gate 已足以排除 shortcut”——不足

当前代码只做：

* primary bank和teacher固定；
* 替换 complete Program；
* 要求 meta/target 两个 role 的 median correct-minus-wrong recovery ≥`.10`。

它没有：

* 与 universal rank4比较；
* 交换 native bank；
* 计算 Program×bank interaction；
* 要求 q/v 分别有因果增量；
* 要求多数 task为正；
* 检查 own-task functional target相对 wrong-task target更近。

因此它能排除“完全不读 Program”，但不能排除：

$$
\text{universal/common update}
+
\text{很小的 Program perturbation}.
$$

## 2.4 文档—代码—执行状态冲突

1. `configs/pi05_ecp_shared_compiler_g3_v4.json` 和 active design 仍将 v4 标成 active functional-polar compiler；但最新 owner authority、`progress.md`顶部和 `9b52e59` 已明确其吞吐 non-pass、无F0/F3、仅保留为可审查 reference。执行状态应以后者为准。
2. 用户要求重点阅读 `configs/...g3_v3.json`，但该文件在 head 已不存在；它是 `82dfb94` 到 `78b7e58` 阶段的历史配置，当前只有 v1/v2/v4。这不是缺失资产，而是历史配置已被 v4替换。
3. current bank-conditioning tests覆盖了 synthetic chunk equivalence、polar、spectral solve和因果 Gate逻辑，但不能替代真实 v4 K1/K4 F0、451-condition mapping或吞吐。
4. `native_factors.py`、`natural_program.py`、`lora.py`、`batched_lora.py`和 PI0.5 target合同在这段历史中没有被改成并行 adapter或新 factor head；新增问题集中在 G3 bank-conditioning/compiler surface。

---

# 3. 根因模型与推荐架构

## 3.1 我的根因模型

把 task \(\tau\) 的 teacher residual写成：

$$
T_\tau=U+S_\tau,
$$

其中：

* \(U\)：跨大量 task共享的 common residual；
* \(S_\tau\)：真正需要 language/video/Program识别的 task-specific residual。

当前 student可以抽象为：

$$
\widehat T_{\tau,v}
=
\mathcal R_{B_{\tau,v}}
\left(
q_\theta(P_{\tau,v},B_{\tau,v})
\right),
$$

其中：

* \(P_{\tau,v}\) 是 G2 Program；
* \(B_{\tau,v}\) 是当前 video native bank；
* \(\mathcal R_B\) 是 B0 solve + B1 signed pooling；
* \(q_\theta\) 是共享 Program/content mapping。

现有证据分别说明：

* F1：\(\mathcal R_B\) 有容量；
* F2：\(B\) 的 global context不可省；
* 多轮F3：\(q_\theta\) 没有取得足够 absolute functional selection；
* universal rank4：旧 objective/Gate 很容易优先学习 \(U\)；
* wrong-Program：模型主要由 bank-common content驱动；
* polar witness：存在一个条件内坐标可把 bank映射到强方向；
* throughput：把完整坐标求解塞入每次 forward 不现实。

## 3.2 根因排序

| 排名 | 候选根因                                                       | 状态           | 判断                                                                        |
| -: | ---------------------------------------------------------- | ------------ | ------------------------------------------------------------------------- |
|  1 | **task-specific Program→selection 的可识别性/credit ownership** | 强证据支持，尚未完全证明 | 当前最可能首因；shared模型连fit绝对值也低，且常被common/bank路径拥有                              |
|  2 | **teacher/common residual 与 Gate 污染**                      | 已证明          | universal rank4 可绕过旧Gate；这是比继续改scorer更早的科学口径问题                            |
|  3 | **candidate/functional coordinate错误**                      | 已证明为重要子因     | Euclidean/pointwise chart不对；但修成full polar仍未证明shared acquisition           |
|  4 | **full current-bank inverse/polar执行过重**                    | 已证明          | 当前v4 deployment形态系统non-pass                                               |
|  5 | Program本身信息不足                                              | 未决           | Program有动态和一些task separation，但是否足够支撑q/v闭环selection尚未回答                    |
|  6 | carrier12/rank4/scale                                      | 次要、条件式       | common residual和equal-scale ceiling存在；但direction目前约`.08`，不是先恢复scale/F4的理由 |
|  7 | Action Meta缺失                                              | 当前无证据        | 当前问题不是动作语义带宽，而是q/v/task causality；应继续关闭                                   |

## 3.3 对 functional-polar 的精确裁决

当前 v4 的核心：

$$
J_{r,e}=C_rC_0^+H_e,
$$

再对 \(J\) 做 polar/SVD，把 Program raw query变换到 actual B0-solve/B1-replay functional metric。

这证明了两件事：

1. 当前 candidate bank内有强 functional image；
2. rank/event共享方式必须符合 family：q需要更多 cross-event global信息，v/action更适合 per-event。

它没有证明：

1. Program能预测正确的 polar coordinate；
2. 40/50个训练任务足以识别这个共享映射；
3. 当前 teacher decomposition是task-specific的；
4. 这个 per-condition operator值得部署。

所以：

> full polar保留为 oracle/teacher；deployment只保留它的**低维、bank-adaptive sufficient-statistics近似**，并且必须先通过容量和吞吐。

## 3.4 推荐的轻量架构：Sketched Bank-Conditioned Native-Factor Student

### Pass A：保持不变

输入：

* exact language；
* K条 ordered action-hidden videos；
* 两个 fixed antithetic probes。

得到：

$$
P_{\text{lang}},P_{\text{scene}},P_{\text{process}},
\rho,\tau,\sigma.
$$

每条视频仍独立保序，K1 identity，K>1 monotonic canonical alignment；Action Meta关闭。

### Pass B0：当前 bank 的低维 functional sketch

对每个 target/native group \(g\)，真实候选为 \(v_n\)：

* input：\(X_{j,n}\)；
* output：\(Y^u_{j,g,n}\)。

先计算低维 candidate key：

$$
k_n=\phi_g(v_n,\text{time,probe,horizon,type,event})
\in\mathbb R^m.
$$

然后流式累计：

$$
\mu_v=\sum_n\pi_n v_n,
\qquad
\mu_k=\sum_n\pi_n k_n,
$$

$$
H_e
=
\sum_n\pi_{e,n}
(v_n-\mu_v)(k_n-\mu_{k,e})^\top
\in\mathbb R^{d_g\times m}.
$$

为了保留 functional-polar 而不形成全矩阵，使用固定、task-independent 的 nested projection \(R\)：

* q global：对8 event concat后的 key/event轴一次投影；
* v/action：每event投影。

构造 bank-adaptive native basis：

$$
Q_g=\operatorname{orth}(H_{\text{concat}}R)
\in\mathbb R^{d_g\times r_s}.
$$

随后只累计小矩阵：

$$
\widetilde C_0=Q_g^\top C_0Q_g,
\qquad
\widetilde C_r=Q_g^\top C_rQ_g,
\qquad
\widetilde H_e=Q_g^\top H_e.
$$

小型 functional operator：

$$
\widetilde J_{r,e}
=
\widetilde C_r
\widetilde C_0^+
\widetilde H_e.
$$

所有 eig/SVD 都在 \(r_s\le64\) 的小空间内完成，最后把 native query lift回：

$$
q^{\text{native}}_{jr}
=
Q_g\widetilde q_{jr}.
$$

### 为什么这个 sketch比直接 top-eigen covariance 更合适

关键方向可能位于 \(C_0\) 的低能量尾部；直接保留 \(C_0\) top eigenspace会重演 fixed-span失败。这里的 \(Q_g\) 从当前 bank 的 native/key cross-image \(H\) 产生，因此优先覆盖“candidate content实际能寻址的 functional image”，而不是只覆盖最大方差方向。

### Program与current-bank summary共同产生 query

完整 Program，而不是 `P_lang`-only，形成：

$$
z_{jre}^{\pm}
=
f_\theta(
P_{\text{lang},j},
P_{\text{scene},j},
P_{\text{process},e,j},
\rho_e,\tau_e,\sigma_{e,j},
s_{B,j,e},
E_j,E_r,E_e,E_g
).
$$

其中 \(s_B\) 是低维 bank summary，不包含 task/video ID。

得到 candidate logits：

$$
\ell_{n}^{\pm}
=
\left(q_{jr}^{\pm}\right)^\top(v_n-\mu_v)
+
b_\theta(P,k_n,\text{metadata}).
$$

### Pass B1：保持 exact signed pooling

严格执行：

$$
w_n
=
\operatorname{softmax}(\log\pi+\ell^+)_n
-
\operatorname{softmax}(\log\pi+\ell^-)_n,
$$

$$
a_{jr}=\sum_nw^A_nX_{j,n},
\qquad
b_{jr}=\sum_nw^B_nY^u_{j,n}.
$$

不蒸馏或近似最终 weighted sum；最终 Value始终是真实当前 X/Y。

随后：

1. native output group gain；
2. rank4 outer products；
3. small-core balanced SVD；
4. frozen carrier12拼接；
5. 唯一38-target、76-tensor rank16 LoRA。

### 哪些 full v4 计算可以删除或共享

当前233组来自：

* 38个 input groups；
* q output：\(18\times8=144\)；
* v output：18；
* action-in output：32；
* action-out output：1。

推荐实现中：

* 每group的 \(C_0\) 在4个ranks、两个branches和8 events间共享；
* 不物化任何 \(d\times d\) \(C_0/C_r\)；
* 不做233次大尺寸 eig/SVD；
* \(Q_g\) 在同group全部rank/event间共享；
* \(C_r\) 只以 \(Q^\top C_rQ\) 的小矩阵存在；
* q仅保留一个cross-event global small operator；
* v/action保留per-event small operator；
* fixed projection \(R\)、owner/group topology和Program heads可预计算；
* mapping训练时，因G2/source全部冻结，可把每个 condition的Program和X/Y cache一次性封存为训练期临时cache；deployment仍跑真实流式路径。当前代码已经有ephemeral X/Y cache机制，可扩展到run-local cache而不进入checkpoint。

复杂度由近似的：

$$
O(Nd^2+d^3)
$$

降为：

$$
O(Ndr_s+Nr_s^2+r_s^3),
$$

显存由 \(O(d^2)\) 降为 \(O(dr_s+r_s^2)\)，B1 exact pooling仍是 \(O(Nd)\)。

以现有 profile 为基准，目标不是声称一定从58秒降到某个数字，而是设置资格：

* post-capture compiler forward不超过一次 native capture wall；
* 即约5–6秒，而不是35.8秒；
* K1 deployment condition总墙钟约不超过15–20秒；
* peak reserved低于35GB；
* mapping run使用frozen Program/X/Y cache后，单condition student forward/backward必须是秒级，而非几十秒。

达不到这一量级，full polar只能留作teacher。

## 3.5 teacher→轻量 student 的评价

### 它与旧 typed-landmark、fixed realizer、direct score supervision 的区别

只有同时满足以下条件才算真正不同：

* student对**每个当前 candidate**打分，而不是选固定landmark；
* bank summary来自当前video set，具有set-equivariance；
* Program参与 query；
* 最终factor由当前真实X/Y exact pool产生；
* teacher不进入checkpoint和deployment；
* 没有固定 effect code→fixed inverse→LoRA；
* 没有 task/video lookup；
* 没有直接输出高维 A/B 的 FactorHead。

### 不应蒸馏什么

不应蒸馏：

* raw logits：有softmax加常数gauge；
* positive/negative branch probabilities：分解不唯一；
* raw dual：随bank covariance旋转；
  -逐candidate signed measure本身：存在巨大 \(V^\top\delta w=0\) null space。

### 应蒸馏什么

首选目标顺序：

1. **family-balanced effective update direction**；
2. deterministic small-core balanced-SVD 后的 input/output subspace，作为诊断或短warmup；
3. scale/spectrum单独监督，stop-gradient隔离；
4. policy effect只作为后期functional guardrail。

若需要 measure-level辅助，应该使用 pushforward：

$$
L_{\text{push}}
=
\left\|
X^\top(w_s-w_t)
\right\|^2
+
\left\|
Y^\top(w_s-w_t)
\right\|^2,
$$

而不是 raw score cosine；本质上只惩罚对factor有功能影响的measure误差。

### 如何防止 universal/common residual shortcut

由40个fit tasks、且只用fit视频构造冻结 universal update \(U\)。正式同时报告：

$$
R_{\text{abs}}
=
\cos(\widehat T,T),
$$

$$
R_{\text{specific}}
=
\cos(\widehat T-U,T-U).
$$

训练可保留 full target职责，但必须加入：

* universal-centered task-specific loss；
* own-task consensus优于wrong-task consensus的contrast；
* Program×bank crossed causality；
* task-equal weighting；
* q/v单独因果检查。

这不会要求立即改变 carrier；它先把 common component从“证明task selection”的指标中移除。

## 3.6 建议的加强版 F3 Gate

对40个fit causality tasks，构造四臂：

$$
R_{CC}=R(P_{\text{correct}},B_{\text{correct}}),
$$

$$
R_{WC}=R(P_{\text{wrong}},B_{\text{correct}}),
$$

$$
R_{CW}=R(P_{\text{correct}},B_{\text{wrong}}),
$$

$$
R_{WW}=R(P_{\text{wrong}},B_{\text{wrong}}).
$$

teacher始终是primary task的 set-valued consensus；wrong Program/bank来自同role下一task，不读outcome。

除了现有 absolute Gate，要求：

1. meta和target两个role分别满足

$$
\operatorname{median}
\left[
R_{CC}-
\max(R_{WC},R_{CW},R_U)
\right]
\ge0.10;
$$

2. 交互项

$$
I=R_{CC}-R_{WC}-R_{CW}+R_{WW}
$$

role median至少`.05`；

3. q和v各自 `correct - wrong Program` 与 `correct - wrong bank` median至少`.05`；

4. 至少70% tasks的overall task-specific margin为正；

5. predicted update对own-task set-valued teacher比wrong same-role teacher更近的task比例至少70%；

6. 再满足原有 held `.75`、p10 `.50`、held/fit `.8`和相邻checkpoint稳定。

这样才能同时排除：

* universal residual；
* Program-only lookup；
* bank-only common content；
* 小Program perturbation；
* overall被action-in单独抬高。

shuffled/reversed仍完全不进入该 Gate。

---

# 4. 按因果顺序排列的三个决定性实验

## 实验一：低维 bank-adaptive functional sketch 的容量与吞吐 Gate

### 数据

* 现有F1的50 tasks / 98 K1 conditions；
* q20、v21、action-in36、action-out37；
* 另加入既有诊断中浅层、layer9、layer17 q/v targets；
* 不读取held outcome；
* 不训练任何shared模型。

### 训练与冻结

全部冻结。full FP64/current v4 polar只作 reference。

使用同一次捕获构造一个 nested sketch：

$$
r_s\in\{16,32,64\}.
$$

这不是三个训练版本，而是一条共享最大sketch的只读rank curve；事前规定选最小通过者。

### 输出

* sketched operator的A/B和effective update；
* full polar reference；
* per-family/per-depth recovery；
* pergroup spectrum；
* wall time与显存；
* streaming/materialized一致性。

### 容量 Gate

每个family：

* task median `>=.98`；
* minimum `>=.95`；
* 深层q/v median `>=.95`、minimum `>=.90`；
* streaming/materialized cosine `>=.9999`；
* IEEE FP32。

### 速度 Gate

单A40真实deployment路径：

* post-native-capture compiler forward不超过 native capture wall；
* 目标约`<=6s`；
* K1总condition约`<=20s`；
* peak reserved `<35GB`；
* 不靠增加GPU通过。

### 通过后

固定最小通过rank和sketch公式，进入实验二。

### 失败后

若到64维仍不能保留深层q/v：

* full polar永久降级为fit-only teacher/diagnostic；
* 不再把其任何大 covariance/eig/SVD放入deployment；
* 实验二只能采用纯低维 set-summary student，并且其task-local正控必须先通过。

---

## 实验二：12-task轻量student的容量—因果联合资格

### 数据

选12个fit tasks：

* 6 meta；
* 6 target；
* 覆盖q/v深层困难和action controls；
* 每task至少3条K1视频；
* 两条fit、一条video-holdout；
* wrong Program和wrong bank按同role固定配对。

### 训练与冻结

冻结：

* source；
* G2 Program；
* carrier；
* full polar teacher；
* teacher factors；
* native X/Y。

只训练：

* full Program→低维bank-summary query；
* fixed owner/rank/event/group topology；
  -轻量candidate compatibility；
* scale暂时冻结。

### privileged teacher

full polar或现有fit-only consensus只在fit videos上离线产生 canonical effective-update teacher；held视频不参与teacher构造。teacher tensor、dual、covariance、authority ID均不进入student forward或checkpoint，因此不违反deployment信息墙。

### 最小loss

$$
L=
L_{\text{set-valued paired update}}
+
L_{\text{task-specific excess}}
+
L_{\text{cross-video dispersion}}.
$$

不使用：

* raw score；
* raw branch probability；
* polar reconstruction；
* equal subspace长期loss；
* policy effect；
* flow；
* carrier preservation。

### 容量正控

同一student执行面，将 Program query换成每task free low-dimensional query，只在两条fit video训练，第三条video零梯度。

要求 held video：

* overall `>=.90`；
* q/v各 `>=.80`；
* action families各 `>=.90`。

不通过即淘汰该轻量summary/score函数类，不允许shared训练掩盖。

### shared student资格

* fit median `>=.60`；
* held-video median `>=.50`；
* q/v held各 `>=.35`；
* held/fit `>=.8`；
* 相对universal的task-specific margin role median `>=.10`；
* Program margin、bank margin各 `>=.10`；
* crossed interaction `>=.05`；
* 70%以上task为正；
* own-task teacher retrieval `>=70%`；
* 满足实验一速度Gate。

这些门低于正式 `.75/.50`，但显著高于现有 `.08–.16`，用于决定是否值得完整451。

### 失败解释

* free query失败：candidate/summary函数类不足；
* free query通过、shared fit低：Program或credit不可识别；
* fit高、held低：shared task/video泛化失败；
* absolute高、causal低：universal/common shortcut；
* overall高但q/v低：action family掩盖。

只有 shared资格全部通过，才允许实验三。

---

## 实验三：加强因果口径后的完整451-condition F3

### 数据

严格恢复现有split：

* 329 fit；
* 40 held-video；
* 82 task-holdout；
* 两个相邻single checkpoints；
* 40-task crossed Program×bank因果panel。

### 训练

只训练已通过实验二的轻量student。G2/source/carrier/scale继续冻结。

### Gate

必须同时满足：

* held median `>=.75`；
* held p10 `>=.50`；
* held/fit `>=.8`；
* adjacent稳定；
* universal-centered recovery median `>=.50`、p10 `>=.25`；
* correct相对wrong Program、wrong bank、universal的role median margin均 `>=.10`；
* interaction `>=.05`；
* q/v family causal margin各 `>=.05`；
* 多数task own-teacher胜wrong-teacher；
* 速度资格继续成立。

### 通过后

才恢复：

1. scale/spectrum的隔离credit；
2. K2；
3. K4；
4. same-task robustness；
5. held5五臂 strict250；
6. G4 joint Writer。

### 失败后

* absolute与causal都低：重开 Program sufficiency/credit，不再改operator；
* absolute高、causal低：停止当前teacher/Gate decomposition；
* fit高held低：处理shared generalization；
* sketch teacher与task-local control均强、shared仍低：当前 Natural Program 对任务策略选择的信息不足，需以明确证据重开G2，而不是恢复旧Writer。

这三个实验之外，不应再发射 full v4 functional-polar F3。

---

# 5. G4、Final、随机fresh、简化loss与Action Meta

## 5.1 G1–G3作为组件因果验证，而不是Final课程

owner的要求正确。

G1/G2/G3分别用于确认：

* native X/Y的条件内容量；
* Program动态；
* shared selection。

它们的冻结关系和teacher-LoRA只是科学诊断工具，不代表最终模型应按同样的课程机械训练。owner authority已经明确要求保留完整Writer随机初始化、fresh端到端联合训练的matched候选。

## 5.2 G4建议

G3通过后，G4应有一个joint run：

* source PI0.5冻结；
* carrier冻结；
  -完整Writer可训练；
* Program、candidate mapping、bank-summary query、selection、scale共同更新；
* 不要求每个任务有target LoRA。

最小目标：

$$
L_{\text{G4}}
=
L_{\text{cross-episode behavior}}
+
L_{\text{set-valued functional}}
+
\lambda_{\text{preserve}}L_{\text{carrier guardrail}}.
$$

其中：

### 1. Cross-episode behavior

视频与action query来自不同episode，以teacher action/flow直接监督policy行为，避免逐帧复制。

### 2. Set-valued functional

在授权fit/meta states上，让generated policy response接近任一完整successful member；一个logical trajectory仍由一个global member解释。

### 3. Carrier guardrail

只防止错误residual大面积破坏已有支持，不要求逼近某个raw A/B。

same-task video consistency和correct-vs-endpoints/wrong可作为paired sampling/causal regularizer；只有实际joint崩落时才升级为额外loss。

## 5.3 哪些探索期loss不应进入Final

默认删除：

* raw polar重建；
* raw dual/score；
* branch probability；
* equal subspace永久loss；
* fixed teacher-LoRA reconstruction；
* candidate hidden separation；
* condition-number辅助loss；
  -每轮探索遗留的多个scale/norm loss。

允许的短warmup只有：

* G3通过组件初始化分支上的paired update；
* 明确的Program/selection causal qualification；
* 有预注册退出条件。

退出条件不是固定step，而是：

* mapping causal Gate保持；
  -小型closed-loop不低于carrier；
* teacher-factor gradient不再是selection唯一来源。

## 5.4 Final的matched初始化对照

Final至少保留两个正式matched候选：

### 候选A：组件初始化

从通过Gate的G2/G3组件参数初始化，fresh optimizer、scheduler和数据顺序。

### 候选B：整个Writer随机fresh

随机初始化：

* language/scene/process readers；
* Program；
* alignment；
* bank-summary/student compiler；
* query/key/scale heads。

仍冻结：

* source PI0.5；
  -原生38 target weights；
* frozen carrier。

两者必须：

-相同架构；
-相同参数量；
-相同授权数据；
-相同loss；
-相同checkpoint节点；
-相同validation8 closed-loop选择合同。

不得因为候选A内部loss更低就排除候选B。

## 5.5 Final不应依赖target LoRA

Final的主监督应来自：

* teacher actions；
* cross-episode flow；
* successful-policy functional evidence；
  -必要时的student-visited-state short continuation；
  -后期on-policy success/progress。

teacher LoRA、full polar和consensus factor可以用于G3组件验证或有限warmup，但不能成为“所有Final任务必须有target LoRA”的数据前提。

## 5.6 G5 outer credit的启动条件

只有G4已经显示：

* full高于carrier；
* full高于language/endpoints；
* breadth成立；
* Goal/Long非零；
* same-task稳定；

才在student-visited states上启动outer credit。

一个expert member只有在固定短 continuation中：

-最终成功，或
-明显提升BDDL progress，
-且不撤销已有predicate，

才可成为credit target。无valid member的state只用reward/progress，不制造伪action label。

## 5.7 Action Meta触发条件

当前继续关闭。

只有同时满足以下机制证据才测试：

1. G3 task-specific q/v selection已经通过；
2. G4有明确、跨suite闭环增量；
3. 剩余误差集中在action-in/out、short-horizon action semantics或控制幅值；
4. native action-side representation的task-local正控明显高于base Writer；
5. 加入Action Meta不会形成第二adapter或部署信息泄漏。

matched ablation必须比较：

* off；
* on；

且保持Program/compiler、数据、checkpoint和唯一rank16合同不变。

只有当相邻checkpoint中：

* closed-loop净增明确；
* breadth不降；
* Goal/Long不降；
* carrier retention不降；
  -视频因果margin不降；

才保留。否则永久关闭。

---

## 最终技术裁决

当前 Native-Factor 主线仍值得继续，但 **full functional-polar v4 不应以当前形态继续 formal**。

最合理的科学状态是：

* G1：bank/value容量成立；
* G2：Program动态资格成立；
* F1：current-bank operator容量成立；
* F2：candidate-local first moment被淘汰；
* 多轮F3：shared task-specific acquisition未成立；
* TF32：已修复的必要数值错误；
* universal rank4：证明旧Gate与teacher decomposition存在shortcut；
* functional-polar：证明actual operator metric重要，但目前只是高成本条件内upper bound；
* v4 throughput：明确non-pass。

下一唯一合理路线是：

> **用 full polar作fit-only teacher/reference，先验证 bank-adaptive低维 functional sketch；再训练读取full Program和current-bank低维summary的轻量student；最终仍以真实X/Y exact signed pooling生成唯一rank16，并用 universal-centered、Program×bank crossed causality 和完整451-condition Gate裁决。**

在这条链通过前，不应恢复旧Writer、fixed realizer、FactorHead、task lookup、Action Meta，也不应启动G4或完整validation开发。

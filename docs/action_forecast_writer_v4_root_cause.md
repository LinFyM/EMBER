# Action-Forecast Writer v4 根因复审

状态：2026-07-26历史机制证据，2026-08-11归档。本文不定义活动架构；后续完整演进与最终结论见
[`research_history.md`](research_history.md)。精确v5历史设计可由
`git show 3a6f801:docs/action_forecast_writer_v5_design.md`读取。

本文覆盖此前“已拍板 frame-local Intent + adjacent Transition”的结论。新证据证明：absolute-time
Plan/Revision只是最直接的行为放大器，不是完整根因；保留v4 visual-state和两个Meta-LoRA原样不动，不能
解决更上游的表示旁路与训练可识别性问题。

本轮诊断当时的边界：

- v4 代码、checkpoint 和正式结果保持封存；
- 当时没有实现或训练 v5，没有继续 AS，也没有进入 RL；
- 本轮只做 train-split 隐藏语义诊断、既有 validation 输出的内部反事实，
  以及 Object-1/Object-3 的定向 official rollout；
- train teacher action/proprio 只在 Writer 完成推理后作为诊断 target 读取，
  从未进入 Writer、optimizer 或 validation/test；
- 后续任何新架构必须原位替换 v4 owner，不保留并行 runner 或双活动 schema。

## 1. 最终判断

当前异常需要分成四层，而不能再归到单个模块：

1. **科学可识别性不足。** AS 中 teacher video 与 action episode 只共享 task，
   并非同一 episode；policy 又在执行时看到正确 language 和 current
   observation。functional action loss 能稳定监督“这个 task 需要什么
   controller”，却没有监督“视频第 \(i\) 帧表示哪个阶段、该帧应对应什么
   future action、同任务不同 demo 的哪些差异应保留”。输入始终为正序，只说明
   normal order 是训练分布，不会自动把内部 forecast 校准成 teacher future
   action。
2. **visual-state 没有成为必要表示。** 32 个 visual-state tokens 只是加入
   frozen PaliGemma prompt 的一条小分支；raw image tokens 加上可训练 VL/Action
   Meta-LoRA 仍能绕过它，直接产生 forecasts。训练因此可以忽略 visual-state，
   通过更容易的 raw-image/Meta 路径生成 task/action-shaped latent。
3. **Meta-LoRA 把弱语义 source forecast 改造成低层 demo/phase code。**
   forecasts 的确越来越能区分同任务不同 teacher 的具体平移轨迹，但
   “latest forecast 更准”和“residual 是误差修正/置信度”这两个 Plan/Revision
   前提反而随 AS 训练持续恶化。
4. **absolute-time Plan/Revision 把错误语义放大成 controller。** v4 把各帧
   local chunks 投到未经识别的共享 robot clock，再把主要来自 frame phase 和
   translation 的错配解释为 Revision。Temporal 和 content-only decoder
   随后忠实保留这条差异；它们不是本轮差异坍缩的来源。

所以最准确的根因链是：

```text
同 task 独立 video/action 的 positive AS 目标
        ↓ 不能识别 demo 的高层过程语义
非瓶颈 visual-state + raw-image/Meta 旁路
        ↓
低层 demo/phase/translation action-shaped forecasts
        ↓
未经校准的 absolute-time Plan/Revision
        ↓
稳定但目标外的 translation controller perturbation
        ↓
Object-1/Object-3 闭环阈值被跨过，shuffled 偶然优于 correct
```

这里“偶然”不表示随机抽样噪声。shuffle 扰动本身跨 permutation、video 和 task
高度一致；偶然的是 AS objective 没有决定它对 closed-loop success 的符号。

## 2. 诊断合同

所有新增诊断固定使用：

- v4 observed-best `step_00000825`，并补查 step75、step300 的演化；
- 同一 frozen π0.5-LIBERO source base、rank-16 public LoRA schema；
- validation 内部反事实复用相同 references、flow noise 和 action queries；
- 新 rollout 只用 Object-1/Object-3 各 50 个既有固定 states；
- rollout 只读 official reward/success；
- 语义审计只使用 24 development-train tasks × 每 task 4 demos；
- train teacher action/proprio 仅作 post-inference measurement，不更新模型；
- validation/test hidden action、proprio、reward 和 outcome 均未读取；
- 全部新增 GPU launch 只使用物理 GPU 4–7。

本轮没有新增 full400 condition，也没有借助对比/顺序 loss 制造预期差距。

## 3. 为什么“训练时一直输入正序”仍不能校准 forecast

### 3.1 正序只是常量条件，不是监督关系

对一个 task，训练样本实际是：

\[
(V_d,\; A_e), \qquad d \perp e \mid task,
\]

其中 \(V_d\) 是 teacher demo，\(A_e\) 是另一条独立 action episode。loss 只要求：

\[
F(V_d,\ language)
\]

生成的 LoRA 能降低 \(A_e\) 上的 functional action loss。它没有要求：

\[
\hat A_i[\ell] \approx A^{teacher}_{i+\ell},
\]

也没有要求同一视频的相邻帧在真实动作时钟上如何对应。所有输入都是正序，
意味着模型只见过 normal distribution；但 task latent、demo style latent、
固定 action template 和真正的过程理解都能在这个分布上降低 loss，目标无法在
它们之间选择。

从期望风险看，独立配对确实会鼓励同 task 不同 demo 的输出趋同；这解释了
same-task-other 比 wrong/shuffle 更接近 correct。但“鼓励平均”不等于建立高层
视频语义：language 已经直接给出 task，policy 执行时又看到 current
observation，因此 task-common controller 可以不依赖视频过程；剩余
demo-specific directions 只要落在较平的 AS 子空间，也不会被目标可靠消除。
当前正是这种情况：same-task aggregate 接近，但低层 translation 差异仍足以在
closed loop 跨过抓取阈值。

### 3.2 loss 下降时，forecast 基本假设反而恶化

24 train tasks × 4 demos 的隐藏语义审计如下。MSE 越低越好；
`latest/earlier MSE ratio < 1` 才支持“最新帧信息最充分”；correction cosine
越高，才越支持 Revision residual 指向真实误差修正。

| checkpoint | trailing-25 AS loss | exact forecast MSE | Pearson | latest/earlier MSE | latest-better pairs | residual→true-correction cosine |
|---|---:|---:|---:|---:|---:|---:|
| step75 | 0.13387 | 0.36745 | 0.26850 | 0.96625 | 0.5090 | 0.33529 |
| step300 | 0.12011 | 0.34499 | 0.26094 | 1.04300 | 0.4194 | 0.25424 |
| step825 | 0.10646 | 0.38592 | 0.28798 | 1.08721 | 0.4042 | 0.23847 |

source-base-like 路径（关闭 visual-state、VL Meta 和 Action Meta）在同一审计上
为 latest/earlier `0.98624`、latest-better `0.54337`、correction cosine
`0.33851`。它不强，但至少没有 v4 step825 那样系统性违反 Plan 假设。

step825 单独关闭 VL Meta、Action Meta、两者全部时，per-task forecast MSE
相对完整 trained path 的中位比为 `0.9409/0.9037/0.8902`；小于 1 表示隐藏
teacher future-action 误差更低。VL/Action Meta 的 LoRA-B RMS 同时从 step75
的 `0.00126/0.00121` 增长到 step300 的 `0.00510/0.00474`，再到 step825 的
`0.00638/0.00594`。

因此不是“normal-order 训练会自然语义校准，只是下游搞坏了”。到 step300，
AS loss 仍在改善，但 Plan/Revision 所需的 forecast 排序已经反转；到 step825
更严重。

## 4. visual-state 是主要结构问题之一

### 4.1 它不是信息瓶颈

当前前向同时存在：

```text
32-token visual-state offset ─┐
                              ├─ frozen PaliGemma + trainable VL Meta
raw image tokens ─────────────┘
                                      ↓
                         trainable Action Meta + Action Expert
```

所以 visual-state 即使完全无效，raw image 和两个 Meta-LoRA 仍可承担全部
video→forecast 映射。架构没有强迫模型通过 visual-state 理解视频。

step825 把 visual-state 替换为 neutral 后：

- forecast relative L2 中位数只有约 `0.00855`；
- cosine 约 `0.999963`；
- exact forecast MSE 比值约 `0.99993`。

也就是说这条设计上最重要的路径，最终对 Action Expert forecast 几乎没有
功能影响。

### 4.2 它从弱状态信号退化为时钟/进度信号

同 task 不同 demo 的 pairwise distance 与真实 teacher action distance 的
Spearman 相关随训练变化为：

| representation | step75 | step300 | step825 |
|---|---:|---:|---:|
| visual coordinates | 0.324 | 0.105 | 0.107 |
| full forecasts | 0.144 | 0.470 | 0.587 |
| frame-main forecasts | 0.327 | 0.561 | 0.652 |
| frame-main translation | 0.503 | 0.625 | 0.695 |
| frame×lead interaction vs teacher translation | 0.336 | 0.658 | 0.740 |

visual coordinate 的 leave-one-demo-out probe 在 step825 为：

- normalized video progress \(R^2 \approx 0.548\)；
- robot state \(R^2 \approx 0.046\)；
- action \(R^2 \approx 0.038\)。

所以 visual-state 最终主要表示“视频大约走到哪里”，并没有成为可供 Action
Expert 使用的机器人视觉状态。与此同时，raw-image/Meta forecast 路径越来越
贴近具体 demo 的低层平移轨迹。

这也修正了先前“visual-state 不是主因”的判断：shuffled context 在 step825
不先改变 per-image forecast，只能说明当前 visual-state 已经被旁路，不能证明
visual-state 设计正确。

## 5. forecast 学到了什么

可用近似分解表示：

\[
A_{i,\ell}
=G_{task,\ell}
+F_{demo,i}
+J_{demo,i,\ell}
+\epsilon_{i,\ell}.
\]

- \(G_{task,\ell}\)：每帧反复出现的 task-level lead/action template；
- \(F_{demo,i}\)：随 frame phase 变化的主效应，主要是 translation；
- \(J_{demo,i,\ell}\)：frame×lead 交互；
- \(\epsilon\)：flow noise 和剩余误差。

step825 的 ANOVA energy 中位占比约为：

- frame main `0.425`；
- lead main `0.324`；
- frame×lead interaction `0.174`。

同任务四个 teacher demos 的距离几何进一步显示：

- teacher action 的 within/cross-task distance ratio 为 `0.531`；
- full forecast 为 `0.276`；
- lead profile 为 `0.170`，但它与同 task teacher-action distance 的相关仅
  `0.181`；
- frame main、translation 和 interaction 与真实 demo 低层动作差异的相关为
  `0.652–0.740`；
- visual coordinates 只有 `0.107`。

所以 v4 不是完全“看不到当前图片”。它通过 raw image/Meta 路径看到了
demo-specific 运动和阶段，而且越来越明显；问题是这些差异主要是低层轨迹/
translation，而不是经同任务多 demo 抽象后的统一高层逻辑。task-level lead
profile 又非常容易充当公共 action template。

## 6. shuffle 为什么产生稳定的同向扰动

### 6.1 不是 permutation lottery

64 条 validation references × 每条 8 个独立 random permutations 上：

- shuffle delta 的两两 cosine 中位数 `0.94095`；
- 对 sealed shuffled delta 的 cosine 中位数 `0.94679`；
- mean-delta norm / mean individual norm 为 `0.96556`。

既有 400-panel LoRA 中，Object-1/Object-3 的 random-video shuffle deltas
pairwise cosine 均值为 `0.8688/0.8676`；各自 mean-delta 保留 individual
norm 的 `0.939/0.936`。没有一个 shuffled adapter 更接近 correct task
consensus，因此 shuffle 不是“去掉 demo 噪声、回到 task 均值”。

原因是 normal order 保留 frame phase 与 time slot 的协方差；任意充分随机的
permutation 都会把这项协方差压到近似零。对带有相似单调 phase/translation
主效应的 videos，不同 permutation 因而产生相似的“去相关后”下游方向，而不是
每次随机得到完全不同的 LoRA。

### 6.2 主效应是 frame phase 和前三维 translation

16-reference forecast 分量移植相对完整 slot shuffle 的 LoRA delta：

| 只重排的 forecast 分量 | relative L2 | cosine to full shuffle |
|---|---:|---:|
| frame main only | 0.208 | 0.899 |
| frame×lead interaction only | 0.182 | 0.550 |
| translation dims only | 0.247 | 0.996 |
| rotation only | 0.0445 | 0.491 |
| gripper only | 0.0015 | 0.124 |

translation-only adapter 与完整 shuffled adapter 的 cosine 为 `0.99747`，
距离仅为完整 shuffle delta norm 的 `0.0848`。所以 true frame×lead
forecast revision 不是必要成分；主要是 frame phase 中的前三维平移内容经过
absolute-time slot 错配后，被 Plan/Revision 编成一个 controller。

### 6.3 闭环 rollout 完整复现

Object-1/Object-3 各 50 episodes：

| arm | Object-1 | Object-3 | total |
|---|---:|---:|---:|
| correct | 38 | 11 | 49 |
| no VL Meta at inference | 40 | 8 | 48 |
| no Action Meta at inference | 35 | 15 | 50 |
| lead-only / remove all frame detail | 31 | 9 | 40 |
| shuffle frame-main only | 41 | 31 | 72 |
| normal forecasts in shuffled slots (`N→S`) | 43 | 29 | 72 |
| shuffle translation dims only | 44 | 35 | 79 |
| true shuffled | 45 | 37 | 82 |

translation-only 相对 correct 为净 `+30`，paired exact McNemar
`p=5.30e-6`；相对 true shuffled 仅差 3，churn `15/100`、
`p=0.607`。这几乎闭合了 `49→82` 的全部异常。

关闭 Meta-LoRA 虽改善隐藏 forecast calibration，却不能作为 step825 的
inference hotfix：下游 Belief/Temporal/decoder 已在完整 Meta 语义上共同训练，
直接移除一支只得到 `48/50`。lead-only 更差为 `40`，也排除了“shuffle 只是
删除所有 demo detail”。

## 7. 为什么这种扰动恰好提高 success

normal→shuffle 的 AS functional loss 在 64 个 validation action queries 上：

- normal `0.133615`；
- true shuffle `0.135925`，增加 `0.002310`；
- Object-1/Object-3 只增加 `0.000414/0.000156`，近似 objective-flat；
- shuffle LoRA delta 与 negative local AS gradient 的 cosine 中位数约
  `0.00164`。

因此 shuffle 不是另一个 AS descent step，也不是 normal 训练漏掉的显然更优
offline solution。它在 AS objective 看来整体略差，在两个 Object tasks 上近乎
平坦且与梯度正交。

已有 25 个分阶段 current-observation probes 显示，异常 action delta 主要位于
approach、pre-grasp、close 和 transport 的 end-effector translation：
translation RMS 中位数约 `0.0197–0.0341`，而单轴 action 上限约 `0.05`。
这种幅度足以改变接近角度和抓取位置。精细抓取是明显的 closed-loop threshold：
一个 offline loss 近似等价的 controller perturbation 可以把大量 episode 从
抓取阈值一侧推到另一侧。

所以可以解释到两层：

- **扰动从哪里来、为什么跨随机 shuffle 稳定、为什么主要改 translation：**
  已有直接因果证据；
- **为什么它在 Object-1/Object-3 的符号恰好为正：** 当前 AS objective
  不识别 closed-loop success，这个符号是 objective-unidentified 的偶然补偿，
  不是随机测量噪声，也不是 Writer 理解了 shuffled video。

在不引入 privileged pose/reward 训练的前提下，不能诚实地把这一步再解释成
某个普适物理定律。它可能在另一个 task 上同样稳定地变坏。

### 7.1 成败翻转复放把 Object 收益具体定位到错误空间绑定

为避免只从参数和 action delta 猜测行为原因，使用 sealed correct/shuffled
cache 原样复放 Object-1/Object-3 的全部固定 50-state panel，并只为两臂成败
翻转的 episode 保存 agentview、wrist、EEF position 和 gripper qpos。未读取
object pose、teacher action/state、reward shaping 或任何隐藏目标。四个
task/condition 的 success 与 termination step 均 `50/50` 精确复现原结果。

两任务共有：

- `correct fail → shuffled success`：`40` 条；
- `correct success → shuffled fail`：`7` 条；
- 其中 Object-1 为 `9/2`，Object-3 为 `31/5`；净增 `33`，解释完整
  400-panel `+39` 中的主要部分。

Object-3 的 31 条 shuffled-only 成功给出了最明确的行为证据：

- `23/31` 条 correct-order rollout 明确接近、闭合并通常抬起深绿色干扰瓶，
  而不是语言指定的红橙色 BBQ sauce；
- `7/31` 条接近了红橙色目标，但在抓取或后续运输中失败；
- `1/31` 条发生多物体碰撞，无法可靠归为单一目标；
- shuffled 在相同 init、language、policy RNG 下均把红橙色目标放入篮中。

这不是“correct 不会抓东西”。在这 31 条中，correct/shuffled 首次闭合后
60 steps 内的最大抬升中位数分别为 `0.2165/0.2316 m`；correct 经常成功抓起并
运输了**错误物体**。两臂首次闭合 EEF 位置的配对距离中位数为 `0.1119 m`，
shuffled-minus-correct 的均值为
`[-0.0451, +0.0924, -0.0176] m`，即主要是稳定的平面抓取点切换，而不是
无结构抖动。

Object-1 的主要模式不同但同样属于低层控制绑定：

- shuffled-only 9 条的首次闭合 step 中位数由 correct 的 `122` 提前到 `91`
  （配对差值中位数 `-43`）；
- correct 的前 60-step 抬升中位数仅 `0.0744 m`，`9/9` 小于 `0.10 m`；
- 画面中反复出现极晚到达、空夹/错物体接近以及抓后掉落，shuffled 则更早
  到达 cream cheese 并完成运输。

反向翻转验证了 shuffle 不是普遍更好的语义变换。Object-3 的 5 条
correct-only episode 中，两臂首次闭合 step 中位数同为 `78`，但
correct/shuffled 的 60-step 抬升中位数为 `0.2392/0.0654 m`，shuffled 有
`4/5` 低于 `0.10 m`；它通常仍朝红橙色目标运动，只是破坏了有用的抓取/运输
控制。Object-1 的两条反向翻转也把闭合从 correct 的中位 `98` 推迟到
shuffled 的 `155.5`。

该现象不集中在少数坏视频。Object-3 的 31 条 shuffled-only episode 跨
`22` 个 teacher demos；demo `14/30/32/43` 生成的同一个 cached LoRA 在不同
init states 上还同时出现了两个翻转方向。这说明 teacher LoRA 携带的是会与
当前几何相互作用的空间控制偏置，不是某几条视频被错误标注。

因此 owner 提出的解释方向成立，但“释放 LoRA 容量”不是最准确的机制：

1. correct-order 视频产生连贯、可被 AS 稳定利用的低层 phase/translation
   controller code；
2. teacher video 与监督 action episode 独立配对，而生成的又是部署到任意
   新初态的静态 LoRA，这个 code 没有被约束为相对物体、相对当前 observation
   的可迁移操作意图；
3. 在 Object-3，它常把当前 policy 的到达点绑定到绿色干扰瓶一侧，压过已有
   language/object semantics；在 Object-1，它常造成到达时机和抓取几何偏差；
4. shuffle 改写前三维 translation 的时间绑定，破坏/旋转这条有害控制偏置，
   已存在于 frozen source base、language 和稳定视频内容中的高层任务信号因而
   重新占主导。

这不是 shuffled video 新生成了更多高层信息：lead-only 更差、shuffled adapter
也没有向 task consensus 收缩。它更像对错误低层 controller 的结构化消融。
此前“success 正号只能称为 objective-unidentified 偶然补偿”的表述需要收窄为：
AS objective 仍不能预测这条扰动是否有益，但在当前主要 Object 增益上，正号的
具体行为来源已经定位为目标选择、到达点和抓取/运输几何的纠正。

复放证据位于：
`/data/ymdai/outputs/ember/pi05_action_forecast_v4_step0825_correct_shuffle_flip_replay_object13_g4567_20260726`。
四个 trajectory JSON SHA256 为
`ee87166f...d1e35a`、`f5bc893c...e1da31`、
`ebae6587...38003`、`2f6942bf...aaee0`；47 对 contact-sheet manifest 为
`4275d01e...a75b4e`。

## 8. 已排除或降级的解释

现有证据不支持把以下因素写成主根因：

- 某一次幸运 random permutation；
- 随机首帧 anchor；
- 视频尾帧或单一 endpoint；
- flow noise；
- Revision strength 数值爆炸；
- Temporal 只有两层；
- content-only LoRA decoder 再次把差异压没；
- shuffle 把 adapter 拉回 identity 或 correct task consensus；
- 只删除全部 demo-specific detail；
- 只在 inference 关闭任一个 Meta-LoRA；
- 直接把 Revision 置零。

其中 absolute-time Plan/Revision 仍是**已证明的直接行为放大器**，只是不能再
被写成唯一上游根因。

## 9. 对旧 v5 决定的修正

旧候选：

```text
frame-local Intent I_i
adjacent Transition ΔI_i = I_i - I_(i-1)
```

有一个明确优点：它删除未经识别的 shared robot absolute-time overlap，不再把
不同 frames 的不同 lead positions 假装成同一控制时刻。因此这部分原则仍可
保留为候选。

但旧决定原样保留以下结构：

- 非瓶颈 32-token visual-state；
- raw image 到 full Meta-LoRA forecast 的旁路；
- 只受 task-level AS 间接训练的 VL/Action Meta-LoRA；
- 可直接把 frame-local low-level action chunk 编成 `I_i` 的路径。

新证据表明，这样做最多移除放大器，不能保证：

- `I_i` 是 visual state 或高层 intent，而不是 translation/phase latent；
- `ΔI_i` 表示任务步骤，而不是相邻低层轨迹差；
- same-task demos 汇聚到共同高层逻辑；
- positive AS 不重新学习 task latent shortcut。

所以 **Intent+Transition 不再是已拍板 v5，只是下一轮设计时可复用的一个局部
约束。** 当前不得据此直接实现、训练 75 step 或进入 RL。

## 10. 下一版在设计前必须满足的合同（已由 v5 设计落实）

以下是当时尚未决定具体结构时提出的必要合同。现均由活动 v5 设计逐项落实：

1. **必要性：** learnable visual-state 必须成为 video→forecast 的必要 owner，
   不能被 raw-image/Meta 旁路到 action-shaped latent；neutral 或 permuted
   visual-state 应产生可解释的上游变化。
2. **可学习但不退化：** 允许保留 task、object、scene 和绝对状态信息，不做
   “删除所有恒定成分”；但变化分支必须 zero-preserving、有向，并且不能仅靠
   language/static query 生成。
3. **Meta-LoRA 职责：** 仍要支持 observer/human teacher 到机器人执行视角的
   转换，但不能在没有中间语义约束时直接自由改写 future-action clock。
4. **forecast gate：** 在进入任何 Plan/Revision/Intent 前，必须先证明：
   normal 的 teacher-future calibration 优于 shuffle/reverse；latest forecast
   统计上更准；residual 与真实修正方向相关。这个检查只作 train-only
   post-inference 诊断，不进入训练。
5. **抽象层级：** same-task different correct demos 的高层 representation/
   LoRA 应明显近于 wrong-task，同时不能随低层速度、抓取角度和具体轨迹大幅
   摇摆。
6. **下游克制：** 两层 Temporal 和 content-only decoder 目前没有失败证据，
   不先加深或重写；任何顺序结构都不能假设未被数据识别的共同 robot clock。
7. **训练信号：** 不使用 contrast/order loss 强行制造 correct-shuffle 差距。
   若 task-level AS 本身仍不可识别过程语义，优先考虑 action-hidden、
   positive-only 的 causal/predictive visual objective，但需先完成架构设计，
   不能把它当作已经批准的答案。

## 11. 本地证据与 SHA256

| evidence | summary/file SHA256 |
|---|---|
| 400-panel LoRA shuffle consensus | `390fcad15cb43bd4d7f588b2a45cd658d73b01f16b932f9a8e232e84aae7f9a6` |
| 64×8 permutation、AS loss、endpoint/time-warp | `edbb86c8d7a6788fb9311f41357bcf982521e0500fb987ea4ef23e0bd1f0916e` |
| forecast component decomposition | `2bd6ae54125d4313b8d0a9a6e59ff7eba32c88e25dd467b23b80c6f8d61c7186` |
| step75 hidden forecast semantics | `99f341c26dd07927f6648ca867dd614d42aa0b7f39342e8cfeb26fd1e09b2baa` |
| step300 hidden forecast semantics | `de5a4529f9a8563dae7a03bff322e749c5360d4f20422a45620cef7bb20b763c` |
| step825 hidden forecast semantics | `a1633aa51a075c7ef30ed132f2df29fd9b3bd2d4b47f74340f37e3de9bedb4bf` |
| step75/300/825 same-task demo geometry | `3507a53f...5169` / `90917cd1...3a84` / `846b5f50...b1e6e9` |
| Writer parameter evolution | `6d1e26687ee7010779a2efb229a8d9e126f29acd2761c4f1c57c3df9425a7f36` |
| root-cause LoRA geometry | `3d0b667962080977ead3a16ef1533e4fdc94b03ea925ae45eca3af92e95565c1` |
| Object-1/Object-3 causal rollout | `d384219c73f6c59ff94a93e69f81d2894408930ab1140adba52aab7e3fc5662d` |

此前 forecast-order transplant、Revision factor、phase-action probes 仍是有效
provenance；本文新增证据只是把“唯一根因”和“v5 已决定”两项过度结论撤回。

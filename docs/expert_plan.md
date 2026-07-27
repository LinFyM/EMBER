# EMBER 独立专家评审与完整研究计划

> **历史原文，不是活动 authority。** 其中的 60/15/15、bank、geometry、shared
> update subspace、per-task oracle、validation-frozen test adaptation 和旧 backbone
> 均已退役；只能用于追溯思想来源。当前合同见根目录 `AGENTS.md` 与
> `docs/execution_brief.md`，当前Writer架构见
> `docs/action_forecast_writer_v5_design.md`，实时执行见
> `docs/active_session_handoff.md`。不要从本文启动实验或恢复实现。

> 评审日期：2026-07-17  
> 评审范围：将仓库当前文件视为完整上下文，不假定任何未提交的历史讨论、实现或结果。  
> 研究状态：**假说与设计评审，不是结果背书。**  
> 证据标记：**[已验证事实]** 表示可由论文、官方项目页、代码、模型卡或数据卡核查；**[基于证据的推断]** 表示由现有证据导出的判断；**[待验证设计假设]** 表示必须由实验裁决的设计选择。

---

## 1. Executive judgment

### 1.1 总体裁决

**结论：只在三个具名先决条件通过后推进（conditional go）。当前不应直接实现完整 EMBER。**

1. **原始命题在受限、可识别的任务分布内科学上成立。** 一个共享的 amortized learner 可以从任务信息 `x_T`（语言、动作隐藏视频）推断任务后验，并输出在控制分布上有用的参数状态。但它不是把“任意非监督信息”神奇地变成梯度；它依赖跨 source tasks 的 executable bridge supervision，学习的是
   
   ```text
   task specification -> posterior over useful policy changes -> parameter state
   ```
   
   若两个需要不同策略的任务在 Writer 可见输入上不可区分，任何 Writer 都不可能同时输出正确更新。
2. **EMBER 瞄准的冷启动问题是真实的。** 对稀疏奖励的新任务，通用 VLA 可能理解语义却没有足够初始成功率，普通在线 RL 因零成功探索而无法启动。一个有零目标交互效用的 adapter center 可能把策略移入可探索区域；一个任务条件化局部度量可能进一步提高固定交互预算下的适应效率。
3. **但主实验中的 video 不是“免费互联网视频”。** 本计划首先使用成功的、同 Franka embodiment、同 LIBERO dynamics 的机器人轨迹，只对 Writer 隐藏 action/proprio/reward。它检验的是跨信息与监督分布的编译，而不是证明 action-free video 比机器人示范更便宜，也不支持 human-to-robot transfer。
4. **最大的科学风险是 benchmark 可识别性，而不是网络容量。** vanilla LIBERO 中场景、初始布局和语言模板可能泄露任务；一个无语言策略或 scene-to-action lookup 可能获得高分。没有同初始状态反事实任务、spec-swap、no-language probe 和场景内 hard negative，成功率提升不能证明 Writer 使用了语言或视频，更不能证明信息被编译成参数更新。
5. **创新空间已经很窄。** DAML 已覆盖“视频诱导机器人策略更新”，Watch, Try, Learn 已覆盖“示范初始化后用奖励适应”，HyPoGen、Hyper-GoalNet 和 DISC 已覆盖“任务描述经 hypernetwork 生成策略参数”，NOLA/LEO/policy-subspace work 已覆盖低维参数坐标，ConRFT、DSRL、TMRL、RL Token、EXPO-FT 等已覆盖 VLA 或机器人策略的 RL refinement。因此，LoRA、video conditioning、hypernetwork、subspace 或 RL 任一单独成分都不是创新。

### 1.2 三个先决条件

- **Gate -1：benchmark/spec 有效性。** 同一初始状态下交换 instruction/video 必须显著改变目标行为；no-language/scene-only probe 不得接近完整模型。
- **Gate 0：useful-update oracle。** 与计划相同注入位置、使用 source-task action supervision 训练的 per-task LoRA oracle，必须在独立 query episodes 和 fresh rollouts 上稳定提高成功率。
- **Gate 1：表示可行性。** 固定的 canonical adapter bank 必须保留 unrestricted oracle 至少 90% 的功能增益；否则 Writer 没有值得学习的坐标。

只有三门均通过，才授权训练直接 Writer；只有直接 Writer 在 locked validation 上有 zero-interaction utility，才授权 task-local RL；只有 ordinary RL 从 Writer center 获益，才授权 task-conditioned geometry 和 reward outer loop。

### 1.3 最强可辩护论文 claim

> **Within a predeclared same-embodiment compositional task distribution, a frozen shared VLA and a task-specification-conditioned hypernetwork can map an instruction and one action-hidden robot video to a low-rank adapter center with positive zero-interaction utility on held-out tasks, and to a soft local adaptation metric that improves matched-budget task-local RL sample efficiency; all shared parameters are trained only on source tasks and frozen during held-out evaluation.**

中文表述：

> 在预先声明的同一机器人 embodiment 与动力学、但任务组合 held-out 的分布内，共享 VLA 冻结时，一个由语言与单段动作隐藏机器人视频条件化的 hypernetwork，可以生成在未进行目标任务环境交互前就提高 held-out 成功率的低秩 adapter center，并生成在相同参数量、环境步数和优化器预算下提高 task-local RL 适应效率的软局部度量；所有共享参数仅使用 source tasks 训练，并在 held-out evaluation 时冻结。

这一定义**不**支持以下更强说法：通用 human-video-to-robot transfer、跨 embodiment/dynamics 迁移、任意文档到任意模型更新、通用 learned optimizer，或优于所有直接条件策略。

### 1.4 最强与最弱假设

**最强、最可接受的假设：** source 与 held tasks 共享感知、动力学、动作原语；source tasks 提供 action/reward bridge；held tasks 只组合已见原子技能；base VLA 已有足够表征能力；shared parameters 在 held 上严格冻结。

**最弱、最需要证伪的假设：** 单段 RGB video 在 language 之外含有可执行增量；有用 adapter 的任务间结构可压缩到 32 维；从 spec 可预测的不只是 center，还包括 RL 的局部几何；一阶、截断的 reward outer estimator 足以训练 Writer；LIBERO 中的增益不是场景或 task-ID 捷径。

### 1.5 Go / no-go

- **Go：** Gate -1、0、1 通过；直接 Writer 在 locked validation 上超过 frozen base、平均 adapter、最近邻 retrieval、直接条件控制和 factorized task descriptor；随后 geometry 在 matched-budget RL 中提高 adaptation AUC。所有方法与阈值冻结后，held tasks 只运行一次确认。
- **No-go / 负面结论：** oracle 本身无用；canonical bank 丢失大部分 oracle utility；Writer 只输出通用 adapter；spec 对换不影响行为；视频时间打乱/错配不降；geometry 不优于普通 LoRA RL；reward outer 只提高 source 而伤害 validation；或完整系统不优于同预算直接条件控制。出现这些情况，应收窄或否定 claim，而不是扩大模型、rank 或 RL budget 掩盖失败。

---

## 2. Correct abstraction and terminology

### 2.1 建议术语

| 术语 | 裁决 | 使用边界 |
|---|---|---|
| **amortized task-conditioned parameter-update generator** | **最准确的总称** | 描述一次性把 task specification 编译为参数状态的功能。 |
| **amortized meta-learner** | 准确 | Writer 跨 source tasks 学习，在新任务上一次前向推断；强调跨任务摊销。 |
| **task-conditioned hypernetwork** | 准确的架构名 | Writer 生成 adapter coefficients/metric；不能把 hypernetwork 本身作为创新。 |
| **adapter initializer and local adaptation preconditioner** | **最准确的具体实现名** | 分别对应 zero-step center 与 task-conditioned geometry。 |
| **learned update rule** | 有条件准确 | 只有 Writer 读取当前梯度、奖励历史或适应轨迹并迭代地产生更新时才使用。 |
| **meta-optimizer / learned optimizer** | 当前不建议 | 经典 learned optimizer 反复读取 gradient/loss/history。静态 `spec -> adapter` 不满足该含义。 |
| **policy subspace** | 不充分 | 固定 bank 只是共享坐标；单个 LoRA update 只是一个点。必须证明 task-conditioned affine center 和 metric。 |

建议正式方法名：

> **EMBER: an amortized task-specification-conditioned adapter initializer and local adaptation preconditioner.**

### 2.2 数学抽象

令任务 `T ~ p_meta(T)`，Writer 可见信息为 `x_T=(l_T,V_T)`，source-only bridge supervision 为 `D_T^bridge={(o,a,r)}`，共享 base policy 为 `pi_theta`。Writer 输出：

```text
(c_T^0, g_T, s_T, tau_T) = H_psi(l_T, V_T)
p_T = normalize(g_T * softplus(s_T))
c_T^k = c_T^0 + diag(p_T) u_T^k
Delta W_l(c) = alpha_l * sum_i c_i b_{l,i} a_{l,i}^T
```

其中：

- `c_T^0 in R^32` 是 adapter center；
- `{a_{l,i}, b_{l,i}}` 是 source-only 学得并冻结的 canonical physical operator bank；
- `p_T in R^32_+` 是任务条件化局部预条件器；
- `u_T^k`、rank-4 residual、critic 和 optimizer state 均为 task-local；
- `tau_T` 包含 KL budget、residual gate 和初始 exploration scale。

目标至少区分：

```text
J_zero = E_T[R_T(theta + Phi(c_T^0))]
J_post = E_T[AUC_k R_T(theta + Phi(c_T^0 + p_T ⊙ u_T^k) + residual_T^k)]
```

参数距离不是主目标；source bootstrap 的主目标是独立 query observations 上的 action loss，最终裁决是 closed-loop success 和 success-versus-interaction AUC。

### 2.3 必须明确的五种分布

1. **信息模态分布：** Writer 读取 `q_spec(language, video)`；视频没有 action/proprio/reward。
2. **监督分布：** source tasks 的 `q_bridge(observation, action, reward)` 用于训练；spec episode 与 bridge/query episode独立。
3. **任务分布：** `p_source(T)`、`p_validation(T)`、`p_held(T)` 在精确 task ID 上不相交；本计划只测试已见原子技能的新组合/关系。
4. **环境与 embodiment 分布：** 首篇固定 Franka、相机、控制接口和 LIBERO dynamics；不测试跨 embodiment。
5. **部署反馈分布：** held task 先做 zero-target-interaction Writer inference；之后只允许 reward 驱动 task-local state，不允许更新 shared Writer/base/bank。

### 2.4 成立所需假设与边界

1. **任务信息可辨识性。** 若 `p(x|T1)=p(x|T2)` 而两任务最优策略不相容，Writer 的最优输出只能是后验混合。必须让 `x_T` 对策略差异有互信息。
2. **共享结构。** 有用 policy deltas 必须落在低复杂度函数族；否则 Writer 只能记忆 source task 或输出平均 adapter。
3. **bridge 可学习。** 没有 source robot actions、reward 或成功判别，RGB video 不唯一指定 7-DoF 控制。
4. **base representation 覆盖。** 小 adapter 不能创造缺失的传感、接触信息或控制带宽。
5. **canonical 坐标稳定。** LoRA 存在缩放、符号和 factor gauge；必须用物理算子归一化、固定符号和功能损失，而不是 raw factor MSE。
6. **局部性。** 有效适应方向位于 center 附近；若新任务需要跨越大策略盆地，soft geometry 会成为上限。
7. **零交互的精确定义。** 它表示没有目标任务环境 rollout；但 instruction 和成功 expert video 是部署输入，具有采集成本，不能称为“无示范”。

### 2.5 与相邻范式的本质区别

| 相邻范式 | 它做什么 | EMBER 必须额外证明什么 |
|---|---|---|
| 普通 language/video 条件控制 | 每个控制步把 spec 作为输入，权重不变 | 在同 encoders、数据、推理 FLOPs 下，参数化 task state 带来 zero-step 或 RL AUC 增益；并通过 neutral-prompt 审计证明任务信息确实进入参数。 |
| 行为克隆 | 用目标任务动作标签拟合策略 | held zero-step 阶段没有动作标签；source bridge 与 spec/query episode 分离。 |
| adapter retrieval | 选择或混合 source adapters | Writer 超过平均、最近邻、soft k-NN，并在新组合上产生非库内功能增益。 |
| hypernetwork | 条件输入生成权重 | EMBER 本身就是 hypernetwork；贡献只能来自完整训练/部署契约。 |
| LoRA initialization | 为目标任务给初始 LoRA | center-only EMBER 就是条件 LoRA initialization；geometry 和 reward outer 需单独证明。 |
| policy subspace | 在固定低维坐标适应 | `p_T` 必须随任务改变并在 matched-budget 下提高 AUC；否则退回固定 subspace。 |
| MAML / meta-RL | 学共享初始化或快速适应程序 | EMBER 是其 multimodal、parameter-generating 实例，不是范式之外的新类别。 |
| latent-action RL | 固定生成策略，在 action/noise latent 中优化 | EMBER 改变参数 center 和参数局部度量；必须与更便宜的 residual/latent actor 比较。 |

一个正结果只能证明：在一个同 embodiment、组合式任务分布上，非标签 task specification 可以经 source bridge 被摊销为有用参数状态。它不能证明通用跨模态 optimizer。

---

## 3. Problems and improvements

### 3.1 高优先级失败模式、修改与证伪实验

| 优先级 | 问题 | 为什么严重 | 具体修改 | 可证伪实验 / 停止条件 |
|---:|---|---|---|---|
| P0 | **benchmark shortcut / language 无效** | vanilla LIBERO 的场景和布局可能泄露 task；Writer 可学 scene-to-adapter lookup。 | 建立 `EMBER-LIBERO-C`：同 init-state 反事实 goal pairs、spec-swap、no-language probe、同 scene 错视频、paraphrase。 | full-spec 相对 no-spec gap ≥20 pp；correct 相对 swapped spec ≥20 pp。no-language 若在 10 pp 内接近完整模型，停止用该设置作核心证据。 |
| P0 | **信息不足或 language 已充分** | 若 instruction 已唯一指定目标，video 可能零增量；若两者都不含摩擦/力信息，零交互不可能恢复。 | 预注册 full-language、underspecified-language、video-only、multimodal 和 motion-sensitive 子集。 | 在 motion/underspecified 子集，正确视频相对同 scene 错视频 ≥10 pp；时间打乱/反转不降则不能声称利用 motion。 |
| P0 | **video 与 action/dynamics 缺口** | RGB 轨迹不唯一确定控制、接触力、顺应性或隐藏状态。 | 第一篇只用同 embodiment robot video；后续 human video 必须有 embodiment/latent-action bridge。 | 对相同视觉轨迹改变摩擦或控制增益；若所需动作不同且 Writer 无法区分，明确记录边界。 |
| P0 | **“action-hidden”不等于低成本** | held spec video 仍来自成功专家轨迹，原始 action 只是对 Writer 隐藏；终态帧可能泄露答案。 | 单列视频获取成本；做首帧、末帧、去终态、失败/部分视频和同终态不同轨迹控制。 | 仅末帧达到完整视频 95% 性能时，claim 收窄为 goal-image conditioning；没有成功视频就失效时明确部署前提。 |
| P0 | **held 数据或 checkpoint 泄漏** | public LIBERO-finetuned VLA、held normalization stats、文件名/task index、视频 metadata 都可泄露。 | 从 `openvla/openvla-7b` 基础 checkpoint 做 source-only OFT；禁止任何 `*-finetuned-libero-*` checkpoint；source-only action stats；strip IDs/actions/reward/terminal。 | checkpoint ancestry、manifest 和 cache key 自动审计；发现 held actions/stats 或 all-task fine-tune 即整次实验作废。 |
| P0 | **参数不可辨识** | 多组 LoRA factors 实现同一 `Delta W`；raw MSE 奖励任意 gauge。 | 从合并后的物理 `Delta W` 学 rank-1 operator bank；单位范数、固定符号；以 query action loss、policy KL、return 为主，coefficient Huber 仅辅助。 | 随机 gauge 变换保持行为却显著改变 factor MSE；若训练依赖 MSE 而功能 loss 不降，停止该监督。 |
| P0 | **Writer 输出尺度灾难** | 小 coefficient 误差映射到 4096×4096 权重可导致动作饱和、NaN 或通用能力崩溃。 | empirical-Fisher/KL whitening；`tanh` center；positive metric clamp；spectral/Frobenius、action KL 和 saturation barrier。 | 99% source/validation 输出 KL ≤0.02；任一 spec 产生 NaN、>5% saturation 或 KL>0.1 即失败。 |
| P1 | **退化成通用 adapter** | Writer 可忽略 spec，只输出平均修正仍提高弱 base。 | correct-vs-wrong functional contrastive loss；average/random adapter；coefficient task variance 和 spec-swap。 | Writer 比 average adapter <3 pp、specificity gap <10 pp，或 task explained variance <20%，判定 collapse。 |
| P1 | **task identity shortcut** | 文本模板、scene ID、视频长度或背景可索引 task。 | 去 filename/task ID，统一长度编码；同 scene hard negatives；paraphrase；factorized semantic tuple baseline。 | scene-only、raw-ID 或 semantic lookup 达到 Writer 95% 功能时，不能声称内容转换；重设 split。 |
| P1 | **geometry 成为性能上限** | 32 维 bank 可加速早期却阻止最终最优策略。 | 使用软几何：bank coefficients + rank-4 residual escape；KL 约束逐步放松；报告 hard/soft。 | 20k-step final success 比 unrestricted rank-4 LoRA RL 低 >5 pp，则 geometry 不合格；只能声称 early AUC。 |
| P1 | **内外循环梯度不可行/有偏** | simulator 不可微，7B 二阶 meta-gradient 成本过高；stop-gradient 又忽略 `p_T` 如何改变 inner trajectory。 | 首篇冻结 backbone；使用 detached-inner FOMAML-style score-function surrogate；小模型上与 finite difference/ES 对比。 | 一阶方向与 finite-difference 方向长期余弦 ≤0 或 outer return 不稳定，则改用 antithetic ES/implicit low-dim gradient，不扩大 7B autograd。 |
| P1 | **Writer 与 base 共同更新不稳定** | base 漂移会使 bank/Writer 坐标成为 moving target，并破坏 zero-step utility。 | 分阶段：先永久冻结 base 和 bank；Stage 6 才允许约 65k 的 shared rank-4 action-head adapter，LR 为 Writer 0.1×、5:1 更新频率、source replay+KL。 | locked validation zero-step 降 >2 pp 或 KL>0.03，回滚并永久冻结 shared base adapter。 |
| P1 | **PPO 概率模型缺失** | OFT L1 head 是确定性动作；粗暴加噪没有可靠 log-prob。 | 将归一化动作均值 `m` 映射为 `atanh(clamp(m))`，使用带精确 Jacobian 的 tanh-squashed diagonal Gaussian；所有 RL baseline 同包装。 | 检查 base-action preservation、ratio、clip fraction、entropy、bound saturation；不可信则切 SAC/AWAC，不报告 PPO 结果。 |
| P2 | **计算成本低估** | 多任务 oracle、5 seeds、数百万 env steps 和 7B online inference 远超一次 SFT。 | 冻结特征缓存、batched actor server、predeclared racing、前置门淘汰失败方法；200k-step pilot 回填吞吐。 | 预测 >1,800 A100 GPUh 或 >16M env steps，先删除次要 ablation，再切 SmolVLA fallback；不减少 primary seeds。 |
| P2 | **统计功效与选择性报告** | 15 tasks 平均值可能由少数场景驱动，大量 checkpoint/ablation 容易过拟合。 | 5 independent seeds、task-level paired bootstrap、唯一 primary endpoint；validation 选模型，held 只运行一次。 | 95% task-bootstrap CI 不跨 0 才称胜；帧/rollout 级伪重复显著不接受。 |

### 3.2 对“bootstrapping”和“adaptation geometry”的裁决

- **Bootstrapping** 只有在 generated center 于 0 target interaction 时已有正效用，且在相同 RL 算法、参数和交互预算下提高 AUC 或降低 steps-to-threshold 时才成立。只有零步提高、随后被普通 RL 立即抹平，应称为 *conditional adapter initialization*。
- **Adaptation geometry** 只有在 center 相同、局部参数数、optimizer、env steps 和 residual escape 相同的比较中，task-conditioned `p_T` 优于单位/全局 metric 时才成立。单个 LoRA update 不是 subspace，固定 bank 也不自动是“任务几何”。
- **共同更新 shared Writer/base** 在资源上只能使用一阶或黑盒 estimator。首篇默认只 reward-update Writer；7B backbone 永久冻结，最多追加一个极小 shared base adapter ablation。

---

## 4. Model, data, and benchmark selection under 8 x A100 80GB

### 4.1 明确选择

**主栈：OpenVLA-OFT + LIBERO-90 + source-only benchmark-native demonstrations。**  
**低成本 fallback：SmolVLA `lerobot/smolvla_base` + 同一 LIBERO split。**  
**不采用 Meta-World 作为主实验或必经阶段。** Meta-World 可用于验证 outer estimator 的小网络单元测试，但它没有真实 VLA language/video interface，不能替代 EMBER claim；再维护一套 simulator 会分散资源。SmolVLA+LIBERO 是更便宜且忠实的 Gate 0。

### 4.2 主模型与 checkpoint

- **Checkpoint：** `openvla/openvla-7b`，启动时解析并记录 immutable Hugging Face revision SHA；MIT license。
- **代码：** `moojink/openvla-oft`，固定 exact Git commit；MIT license。
- **规模与精度：** 约 7.5B 参数，bf16。
- **输入/输出：** third-person RGB + wrist RGB + 8-D proprio；连续 7-D action；chunk length 8；online RL 执行 4 actions 后重规划。
- **Source-only base recipe：** `use_l1_regression=True, use_diffusion=False, use_film=False, num_images_in_input=2, use_proprio=True`。在 source tasks 的 base-fit episodes 上训练 official-style rank-32 `all-linear` VLA LoRA、完整 L1 action head 和 proprio projector；训练后全部冻结。
- **预计 source-fit trainable 参数：** L1 head 约 151M，VLA rank-32 LoRA 约 80–110M，projector 约数百万，总计约 235–270M；实际由代码枚举写入 manifest。
- **显存/并行：** 8-way DDP，不需要模型并行。官方 OFT 报告 LIBERO 双图像+proprio、batch 8/device 约 62.5GB/GPU；推理约 16GB。预计本计划 source fit 62–70GB/GPU，冻结后 Writer/RL replica 20–35GB/GPU。
- **禁止：** 任何 `openvla-7b-finetuned-libero-*`、OFT LIBERO checkpoint 或使用所有 90 tasks 统计量的公开模型。

### 4.3 Simulator、数据与许可证

- **Simulator/benchmark：** LIBERO-90，Franka/Panda、MuJoCo/robosuite stack，固定 upstream commit、MuJoCo version、BDDL files、init-state files 和 camera config。
- **Code license：** MIT。
- **Dataset license：** CC BY 4.0。
- **数据：** benchmark-native HDF5/RLDS demonstrations。标准发布通常为每 task 约 50 条成功轨迹；ingestion 必须断言每个 task 的实际 episode 数、长度、成功标记和 hash，不能静默补齐。
- **其他数据集：** BridgeData、DROID、Open X-Embodiment 不进入 task-level bridge supervision。OpenVLA 已在大规模机器人数据上预训练是 base prior，但 EMBER source/held authority 只来自声明的 LIBERO split。
- **外部 stress：** LIBERO-X（RSS 2026，CC BY 4.0）和 LIBERO-Plus 可做 reporting-only robustness；不用于训练、选 checkpoint 或补救主结果。

### 4.4 精确 task split

以官方 `libero_90` task map 的 **0-based index** 为准，固定 task-map commit `08144b4dd01d91fb0ca40e2c1d93ccaa85025fbc`：

```text
validation = [3, 5, 10, 12, 16, 19, 24, 32, 36, 42, 49, 59, 63, 75, 89]
held_out   = [1, 4, 8, 13, 17, 21, 23, 27, 34, 37, 45, 54, 64, 70, 80]
source     = sorted(set(range(90)) - validation - held_out)  # 60 tasks
```

该 split 特意保留同 scene 的关系/对象/组合 hard negatives，例如 front/back、left/right、单步/复合 drawer、stack order、stove activation + placement。它测试的是：

- 已见 embodiment/dynamics；
- 已见原子谓词和大部分对象家族；
- 未见精确 task ID、关系组合或多步组合；
- **不**测试全新技能 primitive 或新机器人。

启动前解析全部 BDDL，生成 `(verb, object, receptacle, spatial relation, order, scene)` 因子表并断言每个 validation/held 原子技能在 source 中至少出现两次。断言失败则在读取任何结果前终止并重新定义整个 split，不允许事后换 task。

### 4.5 每个 source task 的 50-episode 权限划分

```text
episodes  0..7   Writer spec pool：language + third-person video；actions永不进入Writer/loss
          8..27  source-only base OFT fit
         28..39  unrestricted per-task oracle support
         40..45  Writer functional query / oracle selection
         46..49  locked source report；不参与任何拟合
```

Validation/held tasks 只开放 episodes `0..7` 的 language 和处理后 RGB video；其 action、proprio、reward、terminal 和 normalization contribution 全部封存。Validation reward 只用于 shared-frozen 模型选择和局部 RL 协议调试；held reward 只用于最终 task-local RL。

### 4.6 Action-hidden video 构造

- 从成功轨迹 third-person camera 均匀采样 16 帧，覆盖动作开始至成功前一帧；224×224 或 OpenVLA native resolution。
- Writer 不读取 wrist camera、proprio、action、reward、terminal、文件名、task index、trajectory length metadata。
- 视频统一 16 帧；短轨迹使用时间插值而非 padding pattern；删除音频与容器 metadata。
- Primary spec 为 language + 1 video；训练时 modality dropout 产生 language-only、video-only 和 multimodal。
- 必做 controls：首帧、末帧、首尾帧、去最后 20%、时间打乱、反转、同 scene 错 task、全局随机 video、language/video conflict。
- 建立 paired counterfactual evaluation：固定 init state，在同 scene 运行两个可执行目标；primary deployment 保留普通 instruction，另做 neutral-prompt 审计，在 Writer 生成后把 online policy prompt 统一替换为通用 prompt，以检查任务状态是否真正进入参数。

### 4.7 Writer bridge supervision 与适配目标

**Writer-visible：** instruction、16-frame action-hidden third-person video、modality mask。  
**Source training-only：** episodes 28–45 的 robot actions、observations、success 和 source environment rewards。  
**主监督：** generated center 在独立 query observations 上的 8-step action L1/NLL、policy KL 和 fresh rollout success。  
**辅助监督：** canonical oracle coefficient Huber，权重不超过总 loss 的 20%。

Writer/geometry 只作用于 OFT L1 action head 的两个 `4096 -> 4096` residual linear layers：

- unrestricted useful-update oracle：rank 8 LoRA，`2 × 8 × (4096+4096) = 131,072` trainable scalars；
- shared canonical bank：32 个 rank-1 physical directions/层，`2 × 32 × (4096+4096) = 524,288` shared scalars；
- Writer 只输出 32-D center 和 geometry，不输出 524k 权重；
- task-local residual escape：rank 4，`2 × 4 × (4096+4096) = 65,536` scalars；
- task-local bank coordinate `u_T`：32 scalars；local log-std：7 scalars。

若 Gate 0 表明这两个 residual layers 无法承载有用更新，只允许一次预注册扩展：加入 action-head `fc1` 或把 oracle rank 提到 16。仍失败则停止主栈，不把 adapter 扩到整个 7B 模型。

### 4.8 低成本 fallback

- **Checkpoint：** `lerobot/smolvla_base`，约 450–500M 参数，Apache 2.0；固定 Hugging Face revision 和 LeRobot commit。
- **禁止：** `lerobot/smolvla_libero`，因为它已接触 LIBERO target tasks。
- **默认冻结：** vision encoder 与 VLM；source-only 训练 action expert、state projection 或官方 PEFT targets。
- **Writer target：** action expert 的 `q_proj/v_proj` 与 `action_out_proj` 中预注册的两到四个矩阵；oracle rank 8，bank `m=24`，task residual rank 4，实际 shape/参数量由模型枚举。
- **RL：** SmolVLA flow policy 没有简单 exact likelihood；fallback 使用 fixed-noise deterministic flow rollout + external Gaussian exploration，并用 AWAC/TD3+BC-style task-local actor update。不得把近似 PPO log-prob 用作主结果。
- **用途：** 低成本检验 oracle、表示、Writer acquisition 和 geometry 的机制；正结果不能替代 OpenVLA 主 claim，负结果可阻止昂贵主栈。
- **预计资源：** 1–4 A100，20–35GB/GPU，220–390 GPUh，2.5–4M env steps，300–600GB scratch，5–10 days critical path。

### 4.9 为什么这不是普通 VLA 微调

1. shared base 只在 60 source tasks 上训练，held action labels 从未进入 shared training；
2. held zero-step 只有 language/video spec，Writer 一次生成 task-local parameter state；
3. spec episode 与 action/query episodes 独立；
4. shared Writer/base/bank 在 held 上冻结；
5. task-local RL 只更新 `u_T`、rank-4 residual、critic 和 local optimizer state；
6. 评测同时要求 correct-vs-wrong spec specificity、zero-step utility 和 matched-budget adaptation AUC，而不只看最终 fine-tuned success。

---

## 5. Exact first claim and minimum falsification experiment

### 5.1 第一篇候选 claim

> On a predeclared 60/15/15 split of LIBERO-90 that holds out exact task compositions but preserves the Franka embodiment, simulator dynamics, and source-covered atomic skills, a source-trained multimodal Writer maps an English instruction and one action-hidden robot demonstration video to 32 canonical adapter coefficients for a frozen source-only OpenVLA-OFT policy. Before any target-task rollout, the generated center improves locked-task success over the frozen base, an average adapter, matched-norm random adapters, nearest-adapter retrieval, factorized semantic lookup, and a capacity-matched direct-conditioning policy. Starting from that center, a task-conditioned positive diagonal metric plus a soft rank-4 escape residual improves success-versus-interaction AUC over ordinary matched-parameter LoRA RL. All shared parameters are trained only on source tasks and are frozen on held-out tasks.

精确 distribution shift：**information-to-supervision shift + cross-task compositional generalization**；不包含 environment、dynamics 或 embodiment shift。

### 5.2 最低成本但忠实的实验

不先做完整 reward outer。按以下最小闭环运行：

1. 在 60 source tasks 上训练 source-only SmolVLA fallback 或 OpenVLA-OFT base；主 claim 最终必须用 OpenVLA。
2. 训练 source unrestricted rank-8 oracles，证明两个 action-head residual layers 存在 useful updates。
3. 从 source physical deltas 学 32-D canonical bank，并只拟合 coefficient oracle，证明表示不丢 utility。
4. 训练 static Writer，仅输出 center；不实现 geometry、PPO 或 outer reward。
5. 在 15 locked validation tasks 上比较 base、random、average、nearest retrieval、semantic tuple、direct conditioning 和 Writer center，并运行 modality/spec controls。
6. 所有设计冻结后，held zero-step 只运行一次；若项目继续完整系统，则把这次 held 解封延后到 Stage 7。

### 5.3 最小实验的量化门

**Benchmark validity：**

- full-spec vs no-spec paired success gap ≥20 pp；
- correct vs swapped spec gap ≥20 pp；
- ≥80% counterfactual pairs 的行为随 spec 正确切换。

**Useful-update oracle：**

- locked source report 上 median success gain ≥15 pp；
- median query action loss 降低 ≥20%；
- ≥70% source tasks 有正 success gain；
- median action KL ≤0.02，无 NaN/明显 saturation。

**Representation：**

- 32-D bank coefficient oracle 保留 unrestricted oracle ≥90% 的 action-loss gain；
- median success 与 unrestricted oracle 差 ≤5 pp；
- ≥80% tasks 的 success gap ≤10 pp。

**Direct Writer：**

- locked validation zero-step mean gain vs base ≥10 pp，task-bootstrap 95% CI 下界 >0；
- 超过最强 average/retrieval/semantic baseline ≥5 pp；
- ≥70% tasks non-harm，且受害 >10 pp 的 tasks 不超过 20%；
- correct-vs-same-scene-wrong specificity gap ≥10 pp；
- motion-sensitive/underspecified subset 中 correct video 相对 wrong/shuffled video ≥10 pp。

任一 predecessor gate 失败即停止后续组件。Non-VLA 小网络只能用于验证 gradient estimator 或代码，不可替代上述 VLA claim。

### 5.4 交互、reset 与 seeds

- zero-step：Writer 推断后直接评测；评测 rollout 不更新参数，不计为训练 interaction，但单独报告评测成本。
- local RL primary budget：20,000 实际 simulator action steps/task；budgets `{0, 2k, 5k, 10k, 20k}`。
- reset：使用 LIBERO fixed init-state lists 和 upstream horizon，episode 完成/超时后精确 reset。
- 5 independent shared-training seeds；每 task/seed/adaptation point 10 fixed eval rollouts，zero/final 各 20；环境 seeds 在方法间 paired。

### 5.5 为什么 pass 不只是 task recognition

必须同时满足：

- 超过 factorized semantic tuple hypernetwork 和 k-NN adapter mixture；
- 在同 scene、同 init state 的反事实 pair 中，spec 交换改变目标行为；
- held coefficient 不等于任何单个 source adapter，且功能优于最近邻；
- neutral-prompt 审计仍保留 correct-vs-wrong Writer 差；
- 时间打乱/错配视频在预注册 motion subset 上明显降低；
- 后续 geometry 在 center、参数量、optimizer 和 interaction 相同的条件下超过固定 metric。

---

## 6. Complete model design

**首个实现明确选择方案 3：center + Writer-predicted gates/scales over a shared basis bank。** 不直接生成完整 LoRA tensor；Stage 2 只验证 center，Stage 3 加 ordinary local RL，Stage 4 才加入 task-conditioned geometry，Stage 5 才加入 reward outer。

### 6.1 Shared base policy

```text
third RGB, wrist RGB             [B, 2, 3, H, W]
proprio                           [B, 8]
instruction tokens                [B, L<=64]
OpenVLA-OFT frozen features       [B, sequence, 4096]
L1 action head                    [B, 8, 7]
```

Source-only rank-32 VLA LoRA、proprio projector 和 L1 head 在 base fit 后冻结。Writer bank 注入 L1 head 两个 residual `Linear(4096,4096)`，避免生成全 VLA adapter；若表示门失败才扩展。

### 6.2 Language/video encoders

- **Language：** 复用冻结 OpenVLA tokenizer/LLM；instruction 只前向一次，mean/last valid token hidden `4096`，trainable projection `4096 -> 512`，约 2.10M 参数。
- **Video：** 16 帧分别通过冻结 OpenVLA vision backbone；每帧 patch-pool 后得到约 `D_v`，trainable projection `D_v -> 512`，约 0.5–1.0M。
- **Temporal aggregation：** 4-layer Transformer encoder，`d=512, heads=8, MLP=2048`，16 frame tokens + 1 language token + 1 CLS + modality tokens；约 12.6M。
- **Output MLP：** `512 -> 1024 -> 105`，约 0.64M。
- **Writer 总量：** 约 15.8–16.5M trainable；冻结 encoders 不计入。
- **Missing modalities：** learned `[NO_LANGUAGE]`、`[NO_VIDEO]` tokens，加显式 mask；训练 modality dropout `{L,V,L+V}={0.25,0.25,0.50}`。

### 6.3 Writer 输出

```text
c0          [B, 32] = c_max * tanh(raw_c)
log_scale   [B, 32]
gate        [B, 32] = sigmoid(raw_gate)
p           [B, 32] = normalize_geomean(gate * (softplus(log_scale)+1e-3))
res_gate    [B,  1] = 0.1 + 0.9*sigmoid(raw)
kl_budget   [B,  1] = 0.005 + 0.045*sigmoid(raw)
log_std     [B,  7] = clamp(raw, -5, -1)
```

合计 105 个主要标量。`p` 的几何均值归一到 1，避免 Writer 用全局尺度替代 optimizer learning rate；`gate` 表示方向重要性，`log_scale` 表示局部条件数。

### 6.4 Canonical physical operator bank

对两个目标层 `l in {1,2}`：

```text
a[l,i]      [4096]  unit norm
b[l,i]      [4096]  unit/Fisher norm
U[l,i]      [4096,4096] = b[l,i] a[l,i]^T
DeltaW_l(c) = alpha_l * sum_{i=1}^{32} c_i U[l,i]
```

- 共享 bank 参数：524,288；
- 从 source unrestricted physical `Delta W` 通过 alternating functional dictionary learning 学得；
- 每个 direction 固定 scale、order 和 sign：最大绝对值的 `a` 分量为正；
- 用 source calibration buffer 做 empirical-Fisher/KL whitening；
- bank 在 Stage 1 后冻结，首篇不 joint-train，避免坐标漂移；
- task center 最多 rank 32，但 Writer 只预测 32 coefficients。

### 6.5 Soft adaptation geometry 与 residual escape

```text
c_k = c0 + p ⊙ u_k                           # u_k in R^32, task-local
DeltaW_k = Phi(c_k) + lambda_res * LoRA4_k   # rank-4 residual, task-local
```

- `u_0=0`，使用 SGD+momentum 而不是 Adam，以免 coordinate-wise normalization 抵消 `p`；
- 若 `g_c = dL/dc`，则近似 `Delta c = -eta * p^2 ⊙ g_c / sqrt(g_c^T diag(p^2) g_c + eps)`；
- residual 函数值初始化为 0：`A` 用固定 seed orthogonal/Kaiming，`B=0`，禁止 `A=B=0` 导致第一步梯度死亡；
- residual rank 4 共 65,536 actor scalars，受 KL/trust region 约束；`res_gate` 有 0.1 floor，防止错误 Writer 完全封死 escape；
- `p`、`res_gate` 和 trust budget 在 held 上由 Writer 一次产生，shared Writer 不更新。

### 6.6 Policy distribution 与 critic

OFT head 输出归一化确定性 action chunk `m in R^{B×8×7}`。为保持 base mean：

```text
mu_z = atanh(clamp(m, -1+1e-5, 1-1e-5))
z ~ Normal(mu_z, exp(log_std))
a = tanh(z)
log pi(a) = log Normal(z; mu_z, sigma) - sum log(1-a^2+eps)
```

RL 执行前 4 个 actions，macro-action log-prob 为 4×7 维之和；zero-step/eval 用确定性 `tanh(mu_z)`。若 gripper 在实际 wrapper 中离散化，使用 6-D squashed Gaussian + Bernoulli gripper 并单独审计。

Critic：冻结 pooled VLA feature `512` + proprio projection `64` + `c_k` 32 + previous reward/done 2，输入约 610；`MLP(610,512,512,1)`，约 0.58M。使用 source-trained initialization，每个 task clone 后仅本地更新；所有 RL methods 共用相同 critic 容量和训练预算，critic 参数不计 actor matched budget但必须报告。

### 6.7 参数权限表

| 模块 | Source base/boot | Source reward outer | Validation/Held |
|---|---|---|---|
| OpenVLA original weights | frozen | frozen | frozen |
| source-only rank-32 OFT LoRA | train then freeze | frozen | frozen |
| L1 head + proprio projector | train then freeze | frozen；仅 Stage 6 可加独立 shared rank-4 adapter | frozen |
| canonical bank | Stage 1 source-only train | frozen | frozen |
| Writer | supervised train | source reward update | frozen |
| `u_T`, rank-4 residual, logstd | task-local | task-local | task-local |
| critic/optimizer/replay/GAE | task-local clone | task-local | task-local |

### 6.8 Feedback-aware Writer（仅后续）

首个实现是 static Writer，不读取当前 task failure history。只有 static center/geometry 已通过、且 Stage 5 显示 feedback 可预测改进方向时，才增加约 0.5–1.0M 的 recurrent correction head：

```text
per-block summary h_k = [return, success, entropy, KL, clip_fraction,
                         critic_loss, ||u||, ||residual||,
                         32-D random projection of policy gradient]  # ~48 dims
history H               [B, K<=8, 48]
2-layer GRU/causal TF   [B, K, 256]
output                   delta_c[32], delta_logp[32]
```

加入后才可称为 learned update rule；只有它反复读取 gradients/history 并输出更新时才接近 meta-optimizer。它不是首篇必要组件，不能绕过 static Writer 的失败门。

---

## 7. Training and evaluation algorithm

### 7.1 数据与梯度边界

| 类别 | Source | Validation | Held-out |
|---|---|---|---|
| Writer-visible input | language、16-frame RGB、mask | 可见 | 可见 |
| bridge supervision | actions、query behavior、reward | 不反传 shared params | 永不用于 shared training |
| task-local state | 每 meta-episode 重置 | 每 task 重置 | 每 task/seed 重置 |
| shared state | 按阶段训练 | frozen | frozen |
| checkpoint selection | source metrics | 允许 | 禁止 |
| reporting surface | 非最终 | locked | 只运行最终一次 |

首篇**不**通过 inner RL 做二阶反传。Simulator 不可微，且 7B 二阶轨迹梯度不现实。采用两类 estimator：

1. supervised bootstrap：query action loss 可直接通过 `Phi(c0)` 到 Writer；
2. reward outer：inner `u_K/residual` 和其轨迹依赖 stop-gradient，在 fresh source rollouts 上用 score-function gradient 更新 `c0,p` 和 Writer。这是有偏 FOMAML-style surrogate，必须在小策略上与 finite difference/antithetic ES 比较。

### 7.2 Useful-update oracle 与 bank

```python
def build_oracles_and_bank(source_tasks, frozen_base):
    records = []
    for T in source_tasks:
        support = load_actions(T, episodes=range(28, 40))
        query   = load_actions(T, episodes=range(40, 46))

        delta = functional_zero_rank8_lora(
            targets=TWO_ACTION_HEAD_RESIDUAL_LINEARS,
            A_init="orthogonal", B_init="zeros"
        )
        opt = AdamW(delta.parameters(), lr=3e-4, weight_decay=1e-4)

        best = None
        for step in range(MAX_ORACLE_STEPS):
            batch = sample(support)
            loss = action_l1(frozen_base + delta, batch)
            loss += 0.01 * policy_kl(frozen_base + delta, frozen_base, batch.obs)
            backward_clip_step(loss, opt, max_norm=1.0)
            if step % EVAL_EVERY == 0:
                best = keep_best_safe_on_independent_query(best, delta, query)

        records.append(merge_to_physical_delta_and_scores(best, T))

    bank = alternating_functional_dictionary_learning(
        records, num_directions=32, rank_per_direction=1,
        objective="query_action_loss + Fisher_reconstruction"
    )
    bank = normalize_order_sign_and_fisher_whiten(bank, source_calibration_buffer())
    freeze(bank)

    coeff_oracles = {}
    for rec in records:
        c_star = optimize_32_coefficients_only(bank, rec.query)
        coeff_oracles[rec.task] = c_star
        evaluate_on_locked_source(rec.task, episodes=range(46, 50), fresh_rollouts=True)

    return records, bank, coeff_oracles
```

Oracle 和 bank pass/fail 只能看 episodes 46–49 与 fresh simulator seeds，不能看 support/query training loss。

### 7.3 Supervised Writer bootstrap

```python
def bootstrap_writer(writer, frozen_base, bank, coeff_oracles):
    for step in range(BOOTSTRAP_STEPS):
        T = sample_source_task_balanced_by_scene_verb_relation()
        lang, video = sample_spec(T, episodes=range(0, 8))
        query = sample_action_query(T, episodes=range(40, 46))
        wrong = sample_same_scene_hard_negative(T)

        mode = sample(["language", "video", "multimodal"], [.25, .25, .50])
        c0, p, trust = writer(mask_modalities(lang, video, mode))
        pi = attach_center(frozen_base, bank, c0)

        L_func = action_l1(pi, query)                         # primary
        L_coeff = huber(c0, coeff_oracles[T])                 # <=20% weight
        c_wrong = writer(wrong).c0
        L_spec = relu(MARGIN + action_l1(pi, query)
                      - action_l1(attach_center(frozen_base, bank, c_wrong), query))
        L_safe = kl_barrier(pi, frozen_base, query.obs, trust.kl_budget)
        L_safe += action_saturation_penalty(pi, query.obs)

        loss = L_func + .2*L_coeff + .2*L_spec + L_safe
        backward_clip_step(loss, WRITER_OPT, max_norm=1.0)
        periodically_evaluate_on_locked_validation_without_updates()
```

### 7.4 一个 source meta-training episode

```python
def source_meta_episode(T, shared):
    lang, video = sample_spec(T, episodes=range(0, 8))
    c0, p, trust = shared.writer(lang, video)

    zero_rollouts = rollout(shared.base + Phi(shared.bank, c0), T,
                            seeds=fresh_source_seeds(), update_shared=False)

    local = LocalState(
        u=zeros(32),
        residual=functional_zero_rank4_lora(A_init="orthogonal", B_init="zeros"),
        log_std=trust.log_std,
        critic=clone_source_critic(),
        optimizers=new_local_optimizers(),
    )

    for _ in range(4):
        local = task_local_rl_block(T, shared, c0, p, trust, local,
                                    env_steps=500)

    post_rollouts = rollout(
        shared.base + Phi(shared.bank, c0 + p * stopgrad(local.u))
        + stopgrad(local.residual),
        T, seeds=fresh_outer_seeds(), update_shared=False
    )
    outer_update(shared, zero_rollouts, post_rollouts, local)
```

### 7.5 Task-local RL

Primary algorithm：PPO；若概率审计失败，预注册切换 SAC/AWAC。Primary reward **只用 simulator binary success**，因为命题正是跨过 zero-success barrier。privileged potential shaping 只可做 source-side optimizer sanity 或 secondary ablation，不进入 held primary curve。

```python
def task_local_rl_block(T, shared, c0, p, trust, local, env_steps):
    buffer = collect_macro_action_rollouts(
        policy=compose_policy(shared.base, Phi(shared.bank, c0 + p*local.u),
                              trust.res_gate*local.residual, local.log_std),
        task=T, env_steps=env_steps, replan_horizon=4,
        reward="sparse_binary_success"
    )
    adv, ret = GAE(local.critic, buffer, gamma=.99, lam=.95)

    for _ in range(4):
        mb = minibatch(buffer)
        ratio = exp(logp_current(mb) - mb.logp_behavior)
        L_actor = -mean(min(ratio*adv, clip(ratio, .8, 1.2)*adv))
        L_value = mse(local.critic(mb.obs), ret)
        L_trust = beta(trust.kl_budget) * policy_kl_to_center(mb.obs)
        L_reg = 1e-4*physical_fisher_norm(local.residual) + 1e-4*mean(local.u**2)
        loss = L_actor + .5*L_value - .01*entropy(mb) + L_trust + L_reg

        # u uses SGD+momentum; residual/critic use AdamW.
        backward_clip_step_grouped(loss, local.optimizers, max_norm=.5)

    log_ratio_entropy_kl_saturation_gradient_variance()
    return local
```

### 7.6 Shared reward outer update

```python
def outer_update(shared, zero_rollouts, post_rollouts, local):
    # local inner trajectory is detached; c0 and p are recomputed from spec.
    L_zero_pg = score_function_policy_loss(zero_rollouts)
    L_post_pg = score_function_policy_loss(post_rollouts)
    L_auc = score_function_auc_surrogate(zero_rollouts, post_rollouts)
    L_anchor = source_action_replay_loss(shared)
    L_kl = policy_kl_to_pre_outer_checkpoint(shared, source_replay_obs())
    L_mod = modality_consistency_and_hard_negative_loss(shared.writer)

    loss_writer = L_post_pg + .5*L_zero_pg + .25*L_auc
    loss_writer += .5*L_anchor + adaptive_outer_kl()*L_kl + .1*L_mod

    update_base = STAGE6_SMALL_SHARED_ADAPTER and outer_step % 5 == 0
    total = loss_writer
    if update_base:
        total += .1 * (L_post_pg + L_anchor + 2.0*L_kl)

    total.backward()
    clip_and_step(shared.writer, WRITER_OUTER_OPT, .5)
    if update_base:
        clip_and_step(shared.small_base_adapter, BASE_ADAPTER_OPT, .25)
```

Outer rollouts 只能来自 source task IDs；validation/held reward 不得进入 shared optimizer。Stage 6 也只更新新增的小 rank-4 shared adapter，不更新 7B backbone、source OFT adapter或 bank。

### 7.7 Held-out evaluation

```python
def evaluate_held_task(T, sealed_shared, seed):
    assert all_shared_parameters_frozen(sealed_shared)
    lang, video = locked_spec(T, spec_episode=seed % 8)
    c0, p, trust = sealed_shared.writer(lang, video)

    zero = evaluate_fixed_rollouts(
        sealed_shared.base + Phi(sealed_shared.bank, c0),
        T, seeds=REPORT_SEEDS, n=20
    )

    local = fresh_local_state(seed)
    curve = [(0, zero)]
    for budget in [2_000, 5_000, 10_000, 20_000]:
        local = run_local_rl_until(T, sealed_shared, c0, p, trust, local, budget)
        curve.append((budget, evaluate_fixed_rollouts(
            policy(local), T, seeds=REPORT_SEEDS, n=10 if budget < 20_000 else 20)))

    assert shared_hash_unchanged(sealed_shared)
    return zero, curve, safety_and_specificity_diagnostics()
```

逻辑上先锁定全部 zero-step 结果，再允许 held local RL。任何人在看到 held curve 后改变 architecture、threshold、normalization、seed 或 checkpoint，都必须废弃该 held split 并使用新的 reporting-only split。

---

## 8. Baselines, ablations, and metrics

### 8.1 Matched-budget baselines

“Matched”至少表示：相同 source demos、frozen base、spec episode、online env steps、action distribution、replan horizon、critic、actor trainable parameter上限和 eval seeds。使用额外 held supervision 的方法标为 oracle。

| ID | 方法 | Test-time task 信息 | Task-local actor state | 隔离的问题 |
|---|---|---|---:|---|
| B0 | Frozen source-only OpenVLA-OFT | ordinary instruction | 0 | base performance |
| B1 | No-language / scene-only probe | image+proprio | 0 | benchmark shortcut |
| B2 | Direct L/V/L+V conditioning | cached spec tokens/FiLM，每控制步可用 | 0 | 参数生成是否胜过普通条件控制 |
| B3 | Source multitask BC/OFT | instruction+observation | 0 | 普通 VLA source training |
| B4 | Average source oracle adapter | no target content | 0 | generic adapter |
| B5 | Random matched-norm adapter | no target content | 0 | 仅尺度效应 |
| B6 | Nearest adapter retrieval | target language/video embedding | 0 | retrieval 是否足够 |
| B7 | Soft k-NN adapter mixture | target embedding | k weights | Writer 是否只是插值 |
| B8 | Factorized semantic-tuple hypernetwork | verb/object/receptacle/relation | 32 coeffs | task recognition/结构标签基线 |
| B9 | Writer center only | L/V/L+V | generated center | zero-step claim |
| B10 | Base init + ordinary rank-4 LoRA PPO | reward | 65,536 + logstd | 普通 RL adaptation |
| B11 | Writer center + ordinary rank-4 LoRA PPO | spec+reward | 同 B10 | center 的初始化价值 |
| B12 | Mean center + fixed global diagonal metric | reward | 32 `u` + residual | 固定 subspace/preconditioner |
| B13 | Writer center + fixed global metric | spec+reward | 同 full | task-specific metric 增量 |
| B14 | Full EMBER center + task metric + residual | spec+reward | 32 `u` + 65,536 residual | 完整方法 |
| B15 | Frozen-feature residual actor | pooled VLA feature+reward | 2-layer 512→64→7，≤40k | 更便宜 action-space RL 是否足够 |
| O1 | One action-labeled demonstration LoRA | held actions | rank 8 | practical supervised upper bound |
| O2 | Per-task unrestricted action oracle | held multi-demo actions | rank 8 | representability upper bound；最终表后计算 |
| O3 | Reward-only full action-head adaptation | held reward | >matched | performance ceiling，不用于效率 claim |

B8 的 held 主结果使用可组合 semantic tuple，不使用未训练的 90-way one-hot。raw one-hot ID 只作 seen-ID shortcut 诊断；若用 held ID 训练 embedding，它就是 oracle。B10/B11 增加 32 个 dummy/gain scalars，做严格参数匹配。

直接条件 baseline 复用相同 frozen encoders 和 4-layer fusion Transformer，但输出 8 conditioning tokens 或两个 action-head block 的 FiLM，而非 coefficients；spec 可在 episode 开始缓存，不能用重复编码 FLOPs 人为削弱它。

### 8.2 关键 ablations

1. **模态/时间：** L、V、L+V、underspecified-L+V、same-scene wrong V、global wrong V、shuffle、reverse、first/last frame、drop-final、language/video conflict。
2. **表示：** `m={8,16,32,64}`；rank-1 vs rank-2 directions；两个 residual layers vs 加 `fc1`；Fisher whitening vs Frobenius；physical bank vs raw LoRA factor MSE。
3. **center：** zero/mean/retrieved/random center；same norm random direction；coefficient stop-gradient；center scale sweep只在 validation。
4. **geometry：** `p=1`、global diagonal、task diagonal、`D+UU^T`；shuffle `p_T`；hard bank-only vs soft residual；gates-only/scales-only。
5. **RL：** no RL、PPO vs predeclared SAC/AWAC fallback、replan 1/4/8、sparse-only、trust budgets；主要比较保持 env steps。
6. **outer：** supervised only、zero-return outer、post-return outer、combined；FOMAML surrogate vs antithetic ES；small shared base adapter off/on；bank始终 frozen为主。
7. **泛化：** paraphrase、camera/layout perturbation、LIBERO-X Level 1–2、LIBERO-Plus；均为 reporting-only。

### 8.3 Primary endpoint 与 metrics

唯一 primary endpoint：15 held tasks 上 full EMBER 相对最强 matched-budget baseline 的 normalized success-versus-interaction AUC，task-level paired bootstrap 95% CI 下界 >0；zero-step gain 是必须通过的机制 endpoint。

```text
budgets = [0, 2k, 5k, 10k, 20k]
AUC_norm = trapezoid_integral(success(b), b) / 20,000
zero_step_gain = S_0(method) - S_0(base)
final_gain = S_20k(method) - S_20k(baseline)
episodes_to_50 = first interaction count with smoothed success >= .50
control_harm = P(base succeeds but generated-update policy fails)
specificity_gap = S(correct spec) - S(same-scene swapped spec)
```

同时报告：

- mean/median zero-step success、改善/无害/受害 task 覆盖率；
- AUC、steps/episodes-to-30/50/70%，未达到按 censoring；
- fixed-budget final success，不只报 best checkpoint；
- task/seed variance、worst-quartile task success；
- policy KL、action L1 drift、gripper/bound saturation、episode length和 safety violations；
- coefficient norm、task explained variance、wrong-spec sensitivity、nearest-neighbor novelty；
- critic error、gradient variance、PPO ratio/clip fraction/entropy；
- GPU-hours、peak VRAM、wall-clock、env steps、rollout throughput。

### 8.4 统计与封存协议

- 5 independent shared-training/meta seeds；每 task/condition/seed 每 curve point 10 fixed rollouts，zero/final 20。
- hierarchical paired bootstrap：先重采样 tasks，再在 task 内重采样 seeds/rollouts；task 是主要独立单位。
- 报 absolute pp、relative AUC、95% CI；primary comparison 不校正，其余 families 用 Holm。
- validation 选择唯一 checkpoint 和阈值；held table 只允许 sealed run。
- Writer 使 >20% held tasks 下降 >10 pp 时，即使均值上升也判定机制不可靠。
- 不把 50×15×5 rollout 当成 3,750 个独立 task samples。

---

## 9. Resource and systems plan

### 9.1 资源假设

- 单节点最多 8×A100 80GB、NVLink；≥128 CPU cores、≥512GB RAM、2TB local NVMe；EGL/MuJoCo headless rendering。
- bf16、FlashAttention 2、DDP；不做模型并行。
- online RL 每 GPU 一个 frozen VLA inference replica，多 CPU vector env clients；只在 block 边界同步 32 coeffs 和 residual。
- 冻结 language/video/query features 预计算；online observation 实时编码。
- 所有数字是工程估计，必须用 200k env-step + 2k gradient-step pilot 回填。

### 9.2 主栈预算

| 阶段 | GPU/并行 | Peak VRAM/GPU | GPU-hours | Env steps | 关键路径 wall-clock |
|---|---|---:|---:|---:|---:|
| 数据、manifest、Gate -1 probe | 1–2 | 20–35GB | 24–48 | 0.2–0.4M | 1–2 days |
| Source-only OpenVLA-OFT base | 8 DDP | 62–70GB | 288–384 | 0.2M eval | 36–48h |
| Unrestricted oracle + bank | 4 task-parallel | 22–38GB | 70–120 | 0.3–0.5M | 1–2 days |
| Supervised Writer bootstrap | 4 DDP/cached | 24–38GB | 70–130 | 0.4–0.8M | 1–2 days |
| Writer-init RL baselines | 8 task/seed-parallel | 22–35GB | 200–320 | 2.5–3.5M | 2–4 days |
| Geometry + residual | 8 task/seed-parallel | 22–36GB | 180–300 | 2–3M | 2–4 days |
| Reward outer + optional base ablation | 8 | 26–40GB | 150–250 | 1.5–2.5M | 2–3 days |
| Final sealed held + strongest baselines | 8 | 22–35GB | 180–300 | 4.5–6.5M | 3–6 days |
| **总计（若所有门通过）** | — | — | **约 1,160–1,850** | **约 12–17M** | **约 16–30 days** |

设置硬重估线：pilot 后预测 >1,800 GPUh 或 >16M env steps 时，依次删除 bank joint-training、low-rank SPD、额外 stress seeds和非最强 RL curves；仍超限则切 SmolVLA fallback。不得削减 primary 5 seeds 或用 held pilot 选择方法。

### 9.3 Baseline racing 与环境步数控制

- 所有方法先做 zero-step；validation RL pilot 淘汰显著劣于 B10/B11 的方法。
- held 完整 5-point curves 只运行：full EMBER、最强 ordinary-LoRA-RL、最强 residual/latent actor 三种。
- 其他 baselines 只运行其机制所需的 zero/final endpoints。
- shortlist 在看 held 前冻结；训练和固定评测 rollouts 均计入 env-step ledger。

### 9.4 存储

| 内容 | 估计容量 |
|---|---:|
| LIBERO HDF5/RLDS、init states | 50–120GB |
| 16-frame videos与controls | 80–180GB |
| frozen language/video/query features | 250–600GB |
| rollouts、PPO buffers、diagnostics | 200–400GB |
| OpenVLA/OFT checkpoints | 250–450GB |
| Writer/bank/local checkpoints | 50–120GB |
| **scratch target** | **1.3–2.0TB** |

只保存 PEFT/action-head/Writer 和 manifest，不反复保存 merged 7B。长期归档 300–500GB：sealed checkpoints、task manifest、environment lockfile、seeds、metrics、代表性失败视频和 compute ledger。

### 9.5 并行与依赖

```text
data seals -> source-only base -> unrestricted oracles -> canonical bank -> Writer
                                                        -> local-RL harness (可并行开发)
Writer + RL harness -> geometry -> reward outer -> sealed held evaluation
```

- 数据转换、no-language probe 和小模型 RL 单元测试可并行；
- source oracles 可按 task 并行；validation baselines 可按 method/task 并行；
- source base、bank、Writer、geometry、outer 有严格依赖；
- held RGB/language 的正式评测必须在 architecture、hyperparameters、baseline shortlist 和 checkpoint 冻结后进入；看到 held 后不得回到上游修改共享方法；
- LIBERO-X/Plus stress 只在主 held table 后运行。

### 9.6 主要系统瓶颈

预期瓶颈依次为：LIBERO reset/render/IPC、7B online forward、baseline rollout 数、最后才是 Writer。若 GPU utilization <40%，先提高 observation batching 和 env workers；若 CPU/EGL 饱和，增加 CPU/render workers而不是 GPU。Source SFT OOM 时将 per-device batch 8 降至 4，并用 gradient accumulation 保持 global batch，不静默改变优化预算。

### 9.7 一次性与每任务成本

- **一次性：** source base、oracles、bank、Writer bootstrap、source reward outer；论文必须单列约 1,000+ GPUh 的 meta-training，不可隐藏在“秒级适应”后。
- **新 task zero-step：** 16-frame Writer forward，预计 1–3s/A100；生成 state <2MB（共享 bank不重复）。
- **task-local RL：** primary 20k actual env steps；pilot 假设有效 8–20 steps/s/GPU，约 20–60min/task/seed，但必须以实测报告。

---

## 10. Staged plan to complete the full design

每阶段只有 predecessor 通过才授权下一阶段；路线按科学门而不是日历组织。

### Stage -1 — Benchmark/spec validity

- **Artifact：** `benchmark_validity_report.md`、counterfactual BDDL/init-state pairs、no-language probe、spec-swap harness。
- **组件：** split seals、hard-negative map、paraphrase、wrong-video controls。
- **资源：** 1–2 A100，24–48 GPUh，0.2–0.4M steps。
- **Pass：** full-vs-no-spec ≥20 pp；correct-vs-swapped ≥20 pp；≥80% pairs 正确切换。
- **失败解释：** benchmark主要测场景识别/动作先验。
- **决策：** 失败则停止 vanilla LIBERO 主张，迁移到自建 paired goals 或 LIBERO-X；不训练 Writer。

### Stage 0 — Useful-update oracle

- **Artifact：** 60 个 source rank-8 functional oracles、locked query/rollout report。
- **组件：** source-only base、两目标层 LoRA、独立 query evaluator。
- **资源：** 4 A100，40–80 GPUh（不含 base），0.2–0.3M eval steps。
- **Pass：** median success +15 pp；action loss -20%；≥70% tasks 改善；KL/saturation安全。
- **失败解释：** 目标层/rank或base没有有用局部更新。
- **决策：** 只允许一次 rank16或加`fc1` redesign；仍失败则切 fallback或停止。

### Stage 1 — Representation feasibility

- **Artifact：** 32-D rank-1 operator bank、per-task coefficients、functional reconstruction report。
- **组件：** physical delta extraction、Fisher whitening、sign/gauge canonicalization。
- **资源：** 2–4 A100，40–80 GPUh。
- **Pass：** 保留 unrestricted gain ≥90%；median success gap ≤5 pp；≥80% tasks gap ≤10 pp；Fisher Gram condition number <100。
- **失败解释：** useful updates 不共享低复杂度坐标或字典病态。
- **决策：** 尝试 `m=64` 或 rank-2 direction 一次；仍失败则删除 geometry 研究并重新定义方法。

### Stage 2 — Direct Writer zero-interaction utility

- **Artifact：** source-trained static Writer、locked validation zero-step table和 modality controls；不读取 held。
- **组件：** frozen encoders、fusion Writer、functional bootstrap、hard-negative loss。
- **资源：** 4 A100，70–130 GPUh，0.4–0.8M eval steps。
- **Pass：** validation +10 pp vs base，CI>0；+5 pp vs best retrieval/semantic；≥70% non-harm；specificity和video门通过。
- **失败解释：** oracle acquisition不可摊销、spec信息不足或collapse。
- **决策：** only-language成功则收窄为language-conditioned initialization；video失败则删除video claim；全部失败则停止完整EMBER。

### Stage 3 — Writer initialization + ordinary task-local RL

- **Artifact：** base/mean/retrieval/Writer initialization 的 matched rank-4 adaptation curves。
- **组件：** Gaussian wrapper、critic、PPO/AWAC fallback、KL trust；无 task metric。
- **资源：** 8 A100，160–260 GPUh，1.5–2.5M steps。
- **Pass：** validation AUC +10% relative 或 steps-to-50% -20%；20k final 不低于最强 baseline >5 pp；control harm可接受。
- **失败解释：** center不在更易优化盆地，或RL迅速抹平初始化。
- **决策：** 只zero-step有益则停在initializer claim；RL受害且一次exploration修正仍失败则停止。

### Stage 4 — Predicted adaptation geometry

- **Artifact：** unit/global/task-conditioned metric + residual 的 matched table。
- **组件：** `p_T`、32-D `u`、soft rank-4 escape、metric-swap controls。
- **资源：** 8 A100，180–300 GPUh，2–3M steps。
- **Pass：** validation AUC +10% relative 或 steps-to-50% -20% vs Writer-center+ordinary-LoRA-RL；final不低>5 pp；shuffled `p_T`下降；task metric超global metric。
- **失败解释：**适应方向无法从spec预测、diagonal太弱或residual已足够。
- **决策：** 只允许一次 `D+UU^T` 尝试；仍失败则删除geometry claim。

### Stage 5 — Reward-trained shared Writer

- **Artifact：** source reward outer-trained Writer、before/after locked validation、estimator sanity report。
- **组件：** detached-inner score-function surrogate、zero-step preservation、source replay/KL；base frozen。
- **资源：** 8 A100，120–220 GPUh，1–2M source steps。
- **Pass：** validation post-AUC +8% relative或+5 pp；zero-step下降≤3 pp；source/validation gap不扩大>5 pp；小模型上方向余弦>0.2。
- **失败解释：** estimator偏差/方差、source reward overfit或supervised bridge已饱和。
- **决策：** 尝试antithetic ES/implicit low-dim一次；仍失败则保留supervised Writer并报告负结果。

### Stage 6 — Optional shared base outer adapter

- **Artifact：** frozen-base vs约65k shared rank-4 adapter ablation。
- **组件：**小LR、5:1 Writer/base更新、source replay和KL。
- **资源：**4–8 A100，40–100 GPUh，≤0.8M source steps。
- **Pass：** validation post-AUC +3 pp；zero-step regression≤2 pp；KL≤0.03；≥60% tasks受益。
- **失败解释：** moving coordinates/catastrophic interference。
- **决策：**失败即永久冻结base；完整EMBER不要求此模块成功。

### Stage 7 — Complete multimodal frozen-held evaluation

- **Artifact：**最终主表、curves、failure taxonomy、compute/data cards、sealed checkpoints。
- **组件：**通过前置门的 multimodal Writer、center、metric、local RL和source outer；held shared frozen。
- **资源：**8 A100，180–300 GPUh，4.5–6.5M train+eval steps；5 seeds。
- **Pass：** primary AUC对最强matched baseline的task-bootstrap CI下界>0；zero-step、geometry、video必要性门均保持；无单scene驱动；预算内。
- **失败解释：**组件组合干扰、outer overfit或强baseline解释掉增益。
- **决策：**按失败组件收窄claim；完整系统不胜则不得用cherry-picked task宣称成功。

### Stage 8 — Human-video / cross-embodiment extension（仅后续）

- **Artifact：** WatchAct或受控human-to-LIBERO paired subset外部验证。
- **组件：** perspective/embodiment adapter、latent-action/plan bridge；比较WALA/UniVLA/WatchAct-style pipeline。
- **资源：**单独预算，不计首篇主计划。
- **Pass：** human video相对language-only、oracle plan和wrong-video有增量，同时execution可控。
- **停止：**同 embodiment主命题未通过或oracle execution过低时不进行。

### 10.1 明确的负面结论标准

满足任一项即必须给出负面结论或收窄 claim：

1. useful oracle 在独立 rollouts 上无稳定正效用；
2. canonical bank 不能保留 90% oracle 功能增益；
3. Writer 不超过 average/retrieval/semantic baseline，或 spec-swap 不改变行为；
4. video shuffle、错配、反转在 motion subset 不降低；
5. zero-step center 不提高 RL AUC或造成明显 control harm；
6. task-conditioned metric 不优于单位/全局 metric 和普通 LoRA RL；
7. reward outer 只改善 source、损害 locked validation；
8. 完整方法不胜最强直接条件或 residual/latent-action RL baseline；
9. 任何结果依赖含 held tasks 的 public checkpoint、held actions/stats或开发期 held tuning；
10. benchmark validity gate失败且没有迁移到可识别设置。

---

## 11. Literature and novelty gaps

### 11.1 同行评审工作

| 工作 | Venue / status | 与 EMBER 的重叠 | 对 claim 的约束 |
|---|---|---|---|
| Learning to learn by gradient descent by gradient descent | NeurIPS 2016 | recurrent learned optimizer | 说明“meta-optimizer”通常读取梯度/历史；静态Writer不应使用该术语。 |
| MAML | ICML 2017 | shared initialization + fast adaptation | EMBER属于条件化meta-learning实例，而非新范式。 |
| Meta Networks | ICML 2017 | meta information生成fast weights | 参数生成已有基础先例。 |
| DAML | RSS 2018 | human video诱导robot policy gradient update | “video产生policy update”不能单独声称创新。 |
| LEO | ICLR 2019 | latent parameter space adaptation | 低维适应坐标已有强先例。 |
| Watch, Try, Learn | ICLR 2020 | demonstration bootstrap + sparse-reward trial-and-error | “先看再RL”不是新组合。 |
| Hypernetworks in Meta-RL | CoRL 2022 / PMLR 2023 | task-conditioned policy parameters | EMBER需靠multimodal、zero-step和geometry契约区分。 |
| VIMA | ICML 2023 | multimodal prompt-conditioned robot control | direct conditioning是必需强baseline。 |
| NOLA | ICLR 2024 | basis-composed LoRA | shared basis LoRA不是创新。 |
| LAPA | ICLR 2025 | action-free video latent-action pretraining | video利用不是创新；EMBER需证明per-task parameter state。 |
| HyPoGen | ICLR 2025 | task specification生成policy parameters | language Writer最直接竞争者之一。 |
| UniVLA | RSS 2025 |跨embodiment video/latent actions | action-free video bridge已有强证据。 |
| Hyper-GoalNet | NeurIPS 2025 | goal-conditioned hypernetwork生成机器人策略参数 | goal-to-weight generation不能作为新点。 |
| ConRFT | RSS 2025 | VLA reinforced fine-tuning | VLA+RL不是创新。 |
| DSRL | CoRL 2025 | diffusion latent-space robot RL | 低维适应/探索空间的强实用竞争者。 |
| DISC | RSS 2026 | instruction生成task-specific visuomotor policy weights | 几乎排除“语言生成策略参数”作为EMBER新颖性。 |
| LIBERO-X | RSS 2026 | 600 tasks、100 scenes的扩展benchmark | 可作外部stress，但不能替代因果controls。 |

Primary links：

- MAML: <https://proceedings.mlr.press/v70/finn17a.html>
- Meta Networks: <https://proceedings.mlr.press/v70/munkhdalai17a.html>
- DAML: <https://www.roboticsproceedings.org/rss14/p42.html>
- Watch, Try, Learn: <https://openreview.net/forum?id=H1lXvCNtPr>
- Hypernetworks in Meta-RL: <https://proceedings.mlr.press/v205/beck23a.html>
- NOLA: <https://proceedings.iclr.cc/paper_files/paper/2024/hash/66b99dbf9ed172abac5cb5ccfc82d1e2-Abstract-Conference.html>
- LAPA: <https://proceedings.iclr.cc/paper_files/paper/2025/hash/45d74e190008c7bff2845ffc8e3facd3-Abstract-Conference.html>
- HyPoGen: <https://openreview.net/forum?id=CJWMXqAnAy>
- Hyper-GoalNet: <https://openreview.net/forum?id=aWWRPyGMie>
- DISC: <https://roboticsconference.org/program/papers/147/> and <https://github.com/ReNginx/DISC>
- DSRL: <https://proceedings.mlr.press/v305/wagenmaker25a.html>
- LIBERO-X: <https://arxiv.org/abs/2602.06556>

### 11.2 近期 preprints / workshop work（接受状态需在投稿前复核）

| 工作 | 日期/状态 | 最接近的重叠 |
|---|---|---|
| ViVLA / See Once, Then Act | 2025 preprint | single expert video test-time task learning；直接视频条件VLA强竞争。 |
| Human Demonstration Video as a Prompt | 2025 preprint | human video直接作为policy prompt，无target fine-tuning。 |
| VLA-RL / SimpleVLA-RL | 2025 preprints | VLA RL post-training和zero-success barrier。 |
| Latent Weight Diffusion | NeurIPS 2025 workshop | 生成closed-loop policy weights。 |
| CLAW | 2026-06 preprint | action-free video联合latent actions/world model。 |
| World Pilot | 2026-06 preprint | video/world-action priors直接注入VLA。 |
| WatchAct | 2026-06 preprint/dataset | human video、language与LIBERO task对齐，适合后续human-video test。 |
| Benchmark audit: What Are We Actually Benchmarking in Robot Manipulation? | 2026-06 preprint | 指出LIBERO/CALVIN shortcut和统计分辨率问题；使Gate -1成为必要。 |
| TMRL | 2026-05 preprint | diffusion timestep modulation改善探索。 |
| RL Token | 2026-04 preprint |紧凑VLA readout + small actor-critic online RL。 |
| EXPO-FT | 2026-05 preprint | sample-efficient、stable online VLA RL。 |
| WALA | 2026-07-13 preprint | action-labeled demos + action-free videos学习可执行latent actions；是截止日最新强竞争之一。 |

Primary links：

- ViVLA: <https://arxiv.org/abs/2512.07582>
- Human Demonstration Video as a Prompt: <https://arxiv.org/abs/2505.20795>
- CLAW: <https://arxiv.org/abs/2606.04130>
- World Pilot: <https://arxiv.org/abs/2606.12403>
- WatchAct: <https://arxiv.org/abs/2606.26443>
- Benchmark audit: <https://arxiv.org/abs/2606.04233>
- TMRL: <https://arxiv.org/abs/2605.12236>
- RL Token: <https://arxiv.org/abs/2604.23073>
- EXPO-FT: <https://arxiv.org/abs/2605.25477>
- WALA: <https://arxiv.org/abs/2607.11397>

### 11.3 每个潜在创新点的最强竞争

| 潜在说法 | 最强竞争 | 裁决 | EMBER 还必须证明 |
|---|---|---|---|
| 从语言生成policy参数 | HyPoGen、Hyper-GoalNet、DISC | **不能单独声称** | action-hidden video增量、zero-step center、task-local geometry、frozen-held契约。 |
| 视频产生policy update | DAML、ViVLA | **不能单独声称** | VLA上的canonical adapter center、独立query、时序controls和后续RL。 |
| 看一次再用RL | Watch, Try, Learn | **不能单独声称** | 参数center+metric、source reward outer、held shared frozen。 |
| 利用action-free video | LAPA、UniVLA、CLAW、WALA、World Pilot | **不能单独声称** | 测试时per-task parameter state，而非共享pretraining/direct conditioning。 |
| LoRA/basis/subspace adaptation | NOLA、LEO、policy-subspace、DSRL | **不能单独声称** | task-conditioned center+metric、matched AUC、soft escape且无final ceiling。 |
| VLA reinforcement learning | ConRFT、VLA-RL、TMRL、RL Token、EXPO-FT | **不能单独声称** | Writer在RL前有utility，metric在相同RL budget下有增量，outer仅source。 |
| hypernetwork减少视觉捷径 | DISC | 只能是次要分析 | same-init counterfactual、spec-swap、no-language probe和neutral-prompt审计。 |
| reward训练共享Writer | MAML/meta-RL/hypernetwork meta-RL | 组合上可能 | source-only outer、estimator sanity、zero-step preservation、held冻结。 |

### 11.4 MemLLM 经验是否正确迁移

**正确迁移：**

- representability 不等于 amortized acquisition；oracle、bank、Writer 必须分门；
- raw parameter target 病态，功能行为目标优先；
- update diversity 不等于 task specificity；average/wrong-spec controls 必须；
- credit assignment 到 Writer 不等于应用安全或有效；
- mechanics pass 不等于 science pass；
- 一次性 meta-training cost 与部署成本必须分开。

**具身场景的新增差异：**

1. 功能目标必须从 offline action loss 升级为 closed-loop return、control harm和adaptation curve，因为covariate shift会让低L1 adapter在rollout中失败。
2. video到action的不可辨识性比文档到QA更强，缺失的是 embodiment、contact、timing和dynamics，而不只是标签格式。
3. 受限policy geometry会直接影响探索和安全，必须保留residual escape并比较final ceiling。

### 11.5 最终 novelty judgment

**[已验证事实]** 现有工作已经分别覆盖任务描述生成策略参数、视频驱动适应、action-free video预训练、低维策略空间和VLA RL。  
**[基于证据的推断]** 截至 2026-07-17，本评审没有在已核查的同行评审论文或近期 preprint 中发现完全相同的端到端实验契约：

```text
action-hidden task video + language
    -> zero-interaction useful adapter center
    + task-conditioned soft parameter adaptation metric
    -> task-local RL
    -> source reward outer update of shared Writer
    -> frozen-shared held-task evaluation
```

这不是优先权证明。创新只能落在该**组合、因果controls和matched-budget评测契约**，不能落在单一 ingredient。

### 11.6 结果收窄规则

- center正、geometry负：只能声称 multimodal amortized adapter initialization；geometry报告为负结果。
- language正、video负：删除multimodal/action-hidden claim；相对DISC/HyPoGen的新颖性显著减弱。
- zero-step正、RL无加速：只能声称immediate utility，不能称bootstrapped adaptation。
- RL加速来自center、`p_T`无增量：保留center+ordinary local RL，删除adaptation geometry。
- reward outer只在source正：判定meta-overfitting，不声称shared reward learning generalizes。
- Gate -1失败：所有vanilla LIBERO平均分只算工程结果，必须换可识别benchmark。

### 11.7 最终建议

项目值得推进，但立即工作应严格限于：

1. 建立 leakage-proof split、counterfactual tasks和benchmark validity report；
2. 训练 source-only base并证明 useful oracle；
3. 证明 canonical bank 保留 oracle utility；
4. 只到此时训练 direct multimodal Writer；
5. Writer 在 locked validation 有 zero-step utility 后，再授权 local RL、geometry、reward outer和可选shared base adapter。

最可能成功的形式是一个**受限、同 embodiment 的 amortized adapter initializer**。最不可信的部分是从单段RGB视频推断可执行动力学细节，以及让reward-trained Writer与共享base长期共同稳定更新。若这些组件失败，冻结base、删除geometry/video过强claim并给出明确边界，比构建一个不可解释且昂贵的完整系统更有科学价值。

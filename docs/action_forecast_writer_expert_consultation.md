# EMBER Action-Forecast Writer：外部专家咨询材料

状态：2026-07-26，历史外部咨询材料。本文面向只能访问远程 GitHub 仓库的
外部专家，完整保存v4问题；它不再是当前架构authority。理解本文不需要历史
聊天、本地主机checkpoint或`/data/...`实验目录。专家复核后的根因见
[`action_forecast_writer_v4_root_cause.md`](action_forecast_writer_v4_root_cause.md)，
随后v5设计见
[`action_forecast_writer_v5_design.md`](action_forecast_writer_v5_design.md)；
当前v5.1见
[`action_forecast_writer_v5_1_proposal.md`](action_forecast_writer_v5_1_proposal.md)。

需要先强调两点：

1. 本文记录的是一个真实的科学负结果，而不是已知软件 bug。最新架构确实明显
   使用了视频和顺序信息，但正确时序产生的策略反而不如 shuffled/reversed。
2. 我们目前主要寻求对现象和可辨识性的分析，而不是要求专家立即给出一个复杂
   新架构。尤其不希望用 contrastive/order loss 人工迫使 correct 与 shuffled
   拉开距离，从而掩盖原始信息与目标是否足以支持所期待的语义。

---

## 一、EMBER 的主要思想、核心出发点、问题与方法

### 1.1 想解决什么问题

目标机器人 action trajectories 昂贵，而第三人称机器人示范、互联网人类视频
等 action-hidden teaching videos 更容易获得。EMBER 研究：

> 对一个已经具有基本机器人视觉、语言和控制能力，但没有看过目标任务 action
> 的 frozen VLA，能否只给一条目标任务教学视频和任务语言，就直接生成一个
> 可执行的 task-specific LoRA？

核心映射是：

```text
task language + exactly one action-hidden teaching video
                    -> shared Writer
                    -> complete task-specific LoRA
                    -> frozen VLA executes the task
```

这不是在执行时把视频作为每一步 policy 的额外上下文，而是把视频中的任务知识
一次性“编译”进参数。生成后的 LoRA 可以 zero-interaction 直接执行，也可以
作为以后只用环境 reward 继续练习的初始化。

我们希望 Writer 从视频提取的是可迁移的高层执行逻辑，例如：

```text
识别目标与相关物体
-> 接近目标
-> 抓取或操作
-> 搬运、对齐或改变状态
-> 完成终态
```

它不应机械复刻某条 teacher 的逐点轨迹、速度、抓取角度或偶然视角变化。因为
zero-interaction rollout 的初态与 teacher episode 并不配对，这些低层细节通常
不适合当前执行实例。

### 1.2 公平地基和数据协议

Backbone 是 generic `lerobot/pi05_base`。它在目标 LIBERO panel 上原始为
`0/400`，因此不能合理要求 Writer 同时从一条视频学会 LIBERO embodiment 和新
任务。

我们先只按 task language/BDDL/specification 审计 LIBERO-90 与目标 LIBERO-40
的 exact semantic/composition overlap，排除 19 个重合 source tasks，保留
71 个 source tasks。随后使用这 71 tasks、每 task 50 条成功 action episodes，
从 generic base fresh 联合 action-SFT 1,000 optimizer steps，并冻结共享
π0.5-LIBERO source base。该 raw source base 在目标 40 tasks × 8 states 的
快速能力检查中为 `46/320`，成功覆盖 13 tasks 和全部四个 suites；它只提供
基本 embodiment/control interface，不声称已经解决目标任务。

目标 LIBERO-40 固定拆为：

- 24 development-train tasks；
- 8 validation tasks；
- 8 test tasks；
- 每个 suite（Spatial、Object、Goal、Long）均为 6/2/2。

当前所有 Action-Forecast 结果都只来自 24-train/8-validation development
阶段，尚未进入 final-32 或 test。

### 1.3 AS-Writer 如何训练

每个 Action-Supervised Writer update：

1. 在 24 个 train tasks 中均衡采样 task；
2. 随机采一条该 task 的 action-hidden teacher video；
3. 从同一 task 独立采另一条 observation/action episode 或 action chunk；
4. Writer 用正确 language 和 one video 生成完整 task LoRA；
5. LoRA functional 地装入 frozen source policy；
6. action loss 只通过 generated LoRA 回传到 Writer，source policy 不更新。

teacher video 与监督 action episode 只保证同任务，不保证同一 episode。这个
独立配对是有意设计：Writer 应学习同任务示范共享的执行逻辑，而不能逐帧复制
隐藏 action。

Writer 严禁读取 teacher action、proprio/state、reward、terminal、task ID、
filename、隐藏 normalization 或 policy outcome。当前 teacher 只使用
`obs/agentview_rgb`，另加正确 task language。AS 训练只有 normal positive
functional action loss，没有 shuffled/reversed negative、contrast、margin 或
order classification loss。

### 1.4 比较基线和成功标准

主要无视频基线是 Source-SFT：从同一 frozen source base 出发，在 24 train
tasks 上联合训练一套 shared LoRA；held task 执行时只看 language 和当前机器
人观察，不看 teacher video。四卡 rank-128 Source-SFT 的 validation
observed-best 是 `108/400`。旧八卡结果 `122/400` 只作参考，不是必须超过的
门槛。

Writer 的核心因果对照保持 action-query task、正确 language、official init
state、environment seed 和 policy RNG 不变，只替换 Writer 视频：

- `correct`：正确任务的正常顺序 teacher；
- `same-task other`：同正确任务的另一条正常 teacher；
- `cross-suite wrong`：另一 suite 的错误任务视频；
- `shuffled`：同一视频完整随机打乱；
- `reversed`：同一视频倒序；
- `shuffled_keep_first`：原始首帧固定，只打乱后续帧。

对 `shuffled/reversed`，采样时刻网格及传给 forecast alignment 的
`frame_indices=[0,5,10,...]` 保持不变；只把图像帧重新放入这些时间位置，
不会让某张图像携带它原来所在位置的 index 一起移动。因此该对照确实在问：
“若把另一阶段的画面当作当前时刻，仍让它预测向前的 action，再按正常绝对时间
对齐，会发生什么？”

我们期待同任务不同正确 teacher 的影响相对小，因为它们应共享任务逻辑；
wrong、shuffled、reversed 应产生更大且有害的变化。仅仅看到 LoRA 随视频变化
不算成功；变化必须在实际 rollout 中对应正确视频的行为优势。

---

## 二、从最开始到现在的架构演进、设计动机与结果

本节按时间顺序总结所有对当前问题有因果意义的 Writer 版本。历史版本命名有两
次 “v2”，因此这里明确区分 `Conditional Spatial Writer` 与后来的
`Action-Forecast v2`。

### 2.1 初始 Pooled AS Writer：高性能但几乎是公共 LoRA

**思路。** 每帧 frozen PaliGemma 的 256 个视觉 tokens 被全局平均，整条视频
再压成 4 个 episode tokens。learned parameter queries 读取这些 tokens，并经
带 bias 的共享 factor heads 生成完整 rank-16 LoRA。

**为什么当时合理。** 先验证“一条视频经共享超网络生成完整 LoRA，再通过
functional policy action loss 训练”是否机械可行，并尽快建立非零能力。

**结果。**

- Writer 约 `12.48M` 参数；
- validation best：step 250 `119/400`；
- 同一 fixed400 panel：source base `48/400`，当时 matched-scale
  Source-SFT `61/400`；
- correct/wrong：`119/115`，只有 `+4`；
- fixed video/change language 的 effective-LoRA 相对差约 `4.02e-4`；
- fixed language/change video 只有 `7.52e-6`。

**结论。** 高成功率主要来自近似 input-independent 的公共/domain adapter。
functional policy 本身收到正确 language 和当前 observation，所以 Writer 完全
可以忽略视频，生成一套训练 tasks 通用 LoRA；全局池化、静态 query residual
和 head bias 又进一步打开这条捷径。

### 2.2 Conditional Spatial Writer：用结构和 contrast 强迫视频依赖

**思路。** 保留每帧 `4×4` spatial grid，分别压缩 language/video memory，
去掉 learned-query 到输出的直接 residual，使用 condition-only attention 和
bias-free identity heads。历史训练还交替使用 normal、full-language paired
contrast 和 generic-language paired contrast。

**为什么这样设计。** 直接封堵公共 LoRA 旁路，并让正确/错误视频在同一 action
query 和 policy RNG 下产生可测的 functional 差异。

**结果。**

- 第一轮 step250 correct/wrong `83/63`；
- 充分探索后的 observed-best step500 correct/wrong `99/55`；
- 当时 Source-SFT observed-best `87/400`；
- correct 相比最初版本下降 20，但 wrong 下降 60，视频因果性明显增强。

**结论。** 视频特异性与绝对能力可以分离；但这个版本用 paired contrast 直接
定义了“错误视频应更差”。owner 后来明确拒绝把这种 loss 当作最终解法：我们
希望顺序与视频语义由架构和任务本身自然产生，而不是在 loss 中预设答案。
因此 contrast 路线只保留为历史证据，不是当前允许的修复。

同一历史阶段曾从 fresh Writer 只用 source environment reward 训练
RL-Writer；它的 validation correct/wrong 为 `94/87`。这证明 reward-only
Writer 可以学到部分 competence，但视频因果证据很弱，与当前异常无直接结论。

### 2.3 Action-Memory Writer：让 Action Expert 读取视频，而非直接池化视觉

**思路。** 把语言理解留给 frozen PaliGemma；16 个 memory tokens 从
Action Expert 流中读取每帧图文 prefix。memory 由确定性 action codes 初始化，
VL/Action Meta-LoRA 只帮助 teacher encoder 适配，而 public execution LoRA
仍由 Writer 生成。

**为什么这样设计。** 与其让任意视觉池化器直接生成 LoRA，不如利用 source
VLA 已学到的语言—图像—动作接口，让中间表征更接近“看到当前画面后机器人会
怎么动”。

**结果。**

- bias-free 轨迹 step300/500：`105/89`；
- clean query-scaled 轨迹 step300/400/500/600/800：
  `57/91/86/87/88`；
- best 没有超过当时 Source-SFT；
- 换 cross-suite 视频会明显改变 LoRA；
- 但倒序/乱序的 effective-LoRA 相对差只有约 `0.036/0.027`，远小于
  单帧或重复端点帧的 `0.237–0.312`。

**结论。** 它会看视频内容，却基本把视频当成无序状态集合，没有表达连续动作
如何展开。

### 2.4 Temporal-RoPE Writer：显式加入时间位置仍不够

**思路。** 保留 Action-Memory 主体，只把 temporal self-attention 改为使用
真实 frame index 的 1D RoPE，并让 4 个 condition-only memory queries 读取
多阶段摘要。

**为什么这样设计。** 最小地测试“缺少位置编码”是否是顺序不敏感的主要原因。

**结果。**

- step400/500：`108/98`；
- cross-suite、same-task other、reversed、shuffled 的 effective-LoRA 相对差
  分别约 `0.2267/0.0403/0.00937/0.00699`。

**结论。** 加位置编码并不会让 positive functional loss 自动学会过程语义。
模型仍主要识别“这是什么任务/场景”，而不是“动作怎样随时间推进”。

### 2.5 Action-Forecast Writer v1：把视频翻译成 receding-horizon forecasts

**思路。** 每个 teacher frame 与 imagined visual-state 一起进入
PaliGemma 和 Action Expert，经过可学习 VL Meta-LoRA/Action Meta-LoRA 及完整
10-step flow，得到 `50×7` future-action forecast。不同帧对同一绝对未来时刻
的预测被对齐：

- `Plan_u` 表示离时刻 `u` 最近、信息最充分的 forecast；
- `Revision_u` 表示随着新帧到来，对同一时刻预测的修正/一致性；
- Temporal Transformer 读取 Plan/Revision；
- 320 个 LoRA queries 读取 temporal memory 并生成完整 task LoRA。

**为什么这样设计。** 不再要求普通 hidden states 自发学会过程。显式暴露
“当前最可信的未来计划”和“计划怎样被后续观察修正”，希望把连续动作语义变成
模型容易使用的中间状态。

**结果。**

- Writer `10,161,217` 参数，public LoRA 为 76 tensors、
  `1,287,168` scalars；
- correct 曲线 step150/300/450/600/750/900/1050/1200 为
  `75/99/93/118/104/113/117/125`；
- 后续曾回落又回升，step2550 仍为 `124`；
- observed-best step1200 `125/400`，correct/wrong `125/67`，
  6/8 tasks 净受益，视频任务内容特异性明确；
- 但 shuffled/reversed 为 `121/124`；
- shuffled/reversed effective-LoRA 相对差只有
  `0.001101/0.001787`。

**结论。** 这是迄今 absolute performance 最好的 Writer，也确实依赖正确任务
视频的内容；但它几乎不保留视频顺序。其高层 Action-Forecast 想法仍有吸引力，
失败更像发生在 forecast 后的 Revision/Temporal/query 信息路径。

曾短暂 profile 过 shuffled/reversed negative functional gradient，但 owner
拒绝以 loss 强行制造差异；它从未成为接受的正式训练路径。

### 2.6 Action-Forecast v2：修正 Revision 和静态 query 旁路

对 v1 的无训练内部诊断发现：

- raw directed revision events 的 reversed/shuffled time-centered 相对差为
  `0.223/0.230`；
- 旧 Revision token 只剩 `0.028/0.032`。

旧 Revision 把 old/new absolute actions、delta 和 count/mean/std/max stability
混在一起；稳定的绝对动作内容和统计量淹没有向 revision。LoRA decoder 的静态
query residual 又能不依赖 memory 生成公共成分。

**改动。**

- 28 个 visual-state slots、Revision read、320 个 LoRA queries 都使用
  content-only routing：静态 identity 只进入 attention Q/K，不进入 value、
  residual 或 factor head；
- Revision 的 stability 只作为有限乘法 gate，不再 additive 覆盖 directed
  content；
- 保持 public LoRA schema 和参数规模。

**结果。** Writer `10,125,376` 参数，真实 shape/gradient/resume/profile
通过；该版本主要用于验证机制，随后迅速被更严格的 Belief-v3 取代，没有独立
完成 closed-loop ceiling。

### 2.7 Belief-v3：Plan/Revision 合成正确，但共同成分淹没顺序差异

**思路。** 每个绝对控制时刻只有一个固定布局的 256D Belief：

```text
Belief_u = concat(Plan_u[128], Revision_u[128])
```

Plan 只编码最新 `7D` action。Revision 使用所有更早 forecasts 相对 Plan 的
residual：

```text
direction_u = RMSNorm(bias_free_MLP(
    signed_mean_7(residuals),
    per_dimension_rms_7(residuals)
))

m_u = raw 7D residual RMS
Revision_u = stopgrad(m_u) * direction_u
```

这里没有人工温度 `tau`；一致时严格为零，分歧翻倍时强度自然翻倍。Temporal
和 LoRA decoder 都采用 content-only、zero-preserving Q/K routing 与 raw
content V。

**结果。**

- step600 内部 reversed/shuffled 相对差：
  action forecasts `0.0725/0.0678`，
  Belief `0.0523/0.0464`；
- Revision 的 time-centered 差为 `0.175/0.160`，说明 Revision 合成已有效；
- Belief 总 RMS `0.834` 经两层 Temporal 增至 `9.62`，差异的绝对 RMS
  基本仍在，但 raw relative difference 降到 `0.00479/0.00425`；
- LoRA query 后最终 effective LoRA 只有 `0.000297/0.000169`；
- 只在反事实中减去 temporal mean，可恢复到 `0.0543/0.0401`。

**结论。** Revision 本身不再是首要问题。Plan/Revision 里跨时刻相似的任务与
动作共同成分经 Temporal 放大，单路 query read 几乎只读取共同成分。直接移除
temporal mean 虽能恢复顺序差异，但会删除有价值的任务、场景、物体和稳定动作
信息，因此没有被接受为 canonical 解法。

进一步追到上游后发现旧 virtual-state 总 RMS 约 `0.652`，跨帧变化只有
`0.0057`。每帧 Action Expert 实际容易输出近似同一条 task-level action chunk。
这把最大嫌疑从下游移到了 visual-state：如果每帧没有可靠表示任务阶段，后面的
Plan/Revision 无法凭空创造正确 forecast。

### 2.8 当前 Visual-State v4：结构上恢复动态信息，但行为方向异常

v4 的设计与结果很多，下一节单独完整介绍。它解决了 v3 的内部顺序差异塌缩：
step75 的 final effective-LoRA reversed/shuffled 差已经达到
`0.0420/0.0468`。然而充分训练后的正确时序 rollout 不但没有优于顺序破坏，
反而显著更差。这是当前寻求专家意见的核心现象。

---

## 三、当前最新架构：设计思想、每个模块与全部现有结果

当前 canonical 实现由
[`action_forecast_writer_design.md`](action_forecast_writer_design.md)
精确定义，配置为
[`pi05_as_writer_action_forecast_v4.json`](https://github.com/LinFyM/EMBER/blob/73c419137b6004d6578c3c784633a711dfb95e0c/configs/pi05_as_writer_action_forecast_v4.json)。
这些v4源码现已退役，链接固定到最后一个保留完整v4实现的远程commit。
下面给出完整前向链路和每个模块存在的理由。

### 3.1 总体链路

```text
correct task language + one action-hidden teacher video
  -> frozen projected image tokens X_t
  -> anchored 8D visual-state
  -> native 32-token State: block
  -> PaliGemma + VL Meta-LoRA
  -> Action Expert + Action Meta-LoRA + 10-step flow
  -> per-frame 50×7 future-action forecasts
  -> same-absolute-time alignment
  -> Plan_u + Revision_u
  -> Belief_u = concat(Plan_u, Revision_u)
  -> two-layer Temporal Transformer
  -> content-conditioned 320-query LoRA decoder
  -> 76 rank-16 tensors = complete task LoRA
```

不存在独立的 visual-state/task-global `G -> LoRA` 旁路。所有视频信息，无论
稳定还是动态，都必须先影响 Action Expert forecasts，再经过 Plan/Revision、
Temporal 和 query decoder。

### 3.2 视频采样与 frozen 图像 tokens

- 只用 `agentview_rgb`；
- `frame_stride=5` 固定，视频长度 `T` 可变；
- 每帧保留 frozen SigLIP/PaliGemma projector 的完整
  `X_t[N_img,2048]`，不做旧式全局池化；
- canonical sampled-time grid `frame_indices=[0,5,10,...]` 始终固定并用于
  absolute-time alignment/RoPE；反事实只重排图像，不随图像重排 indices；
- 不调整为 stride 10。

**意图。** 保留物体、机械臂和空间布局；让顺序变换只改变实际帧次序与其产生的
状态/forecast，而不是丢掉原始时间索引。

### 3.3 Anchored visual-state

我们没有 teacher proprio supervision，也不把 8 个 coordinates 声称为真实
机器人关节轴。它们是 source Action Expert 可使用的低维、连续、
video-conditioned state-and-motion coordinates。

```text
h_0 = InitialStateReader(X_0)

D_anchor_t = X_t - X_0
D_local_t  = X_t - X_(t-1)

c_t = ChangeReader(X_0, X_(t-1), X_t,
                   D_anchor_t, D_local_t)

h_t = h_0 + c_t
z_t = tanh(h_t)             # z_t has 8 scalars
```

**为什么不是逐帧独立预测。** 在当前 task-level AS loss 下，独立
`AbsoluteReader(X_t)` 容易把每帧都映射为同一 task latent。

**为什么不是递归累计。** `h_t=h_(t-1)+delta_t` 会累计误差和漂移。

**当前折中。** 每帧相对同一首帧重新计算 anchor change，同时读取局部相邻
change；`h_t` 不依赖上一个估计值，所以不会积累 state-estimation error。

InitialStateReader 和 ChangeReader 都不读 language。coordinate identities
只进入 attention Q/K；content 只能来自 image values。动态 value 只来自有符号
差分，使用 bias-free、odd `tanh` 路径，因此 identical pair 产生严格零 change，
交换有符号 pair 时输出相应反向。8-scalar bottleneck 避免 visual-state 直接
变成任意高维 soft prompt。

代码：
[`visual_state.py`](https://github.com/LinFyM/EMBER/blob/73c419137b6004d6578c3c784633a711dfb95e0c/src/ember/writer/visual_state.py)。

### 3.4 32-token native state renderer

source PI05 原生中性状态文本：

```text
" 128 128 128 128 128 128 128 128"
```

恰好 tokenizes 为 `[space,1,2,8]×8`，即 32 tokens。8 个 whitespace
positions 固定为 frozen native embeddings；每个 `z_(t,d)` 只控制随后 3 个
digit positions，且可学习偏移被限制在 frozen digit `0..9` embedding 的子空间。

**意图。**

- step0 与 source model 熟悉的合法原生 prompt 完全对齐；
- visual-state 可学习但容量受控；
- 不再用随机 8/28-token soft prompt 造成位置和分布偏移；
- 仍允许不同初态和未来 observer/human videos，不假设所有 teacher 共享同一
  物理初始 state。

### 3.5 VL Meta-LoRA 与 Action Meta-LoRA

- VL Meta-LoRA：PaliGemma 18 layers 的 q/k/v/o，rank 4；
- Action Meta-LoRA：Action Expert 18 layers 的 q/k/v/o，rank 8；
- 两者均 identity initialization、可学习，只服务 teacher-video forecast
  路径，不成为 public execution adapter。

**意图。**

- VL Meta-LoRA 适配视觉域、observer viewpoint 以及 image/language/state
  融合；
- Action Meta-LoRA 让机器人理解：“假如我是视频中的 teacher，此刻接下来会
  怎么动”，把第三人称或未来人类示范转换到机器人动作空间。

保留它们是因为仅靠 frozen source model 未必能理解新的 visual-state 表示或
observer-to-robot 映射。风险是：AS loss 没有逐帧 forecast supervision，它们
也可能学到一种对最终 action loss 有用但语义不校准的内部 forecast。

代码：
[`action_forecast.py`](https://github.com/LinFyM/EMBER/blob/73c419137b6004d6578c3c784633a711dfb95e0c/src/ember/writer/action_forecast.py)。

### 3.6 Per-frame future-action forecast

每帧 native prompt 顺序为：

```text
image
+ Task: <correct language>, State:
+ 32 visual-state tokens
+ ;\nAction:
```

它经过 PaliGemma、Action Expert 和完整 10-step flow，输出 source-normalized
`P_i[50,7]` future-action forecast。同一视频 condition 的所有帧使用同一条
可恢复 `[50,32]` Gaussian flow noise，避免不同帧的随机 flow noise 伪造
revision。

**意图。** 让后续模块处理明确的动作预测，而不是自己从 arbitrary hidden
states 中发明“过程”的含义。

### 3.7 Absolute-time alignment、Plan 与 Revision

若第 `i` 帧对应真实控制时刻 `t_i`，它的第 `k` 个 future action 覆盖绝对时刻
`u=t_i+k`。对每个 `u`，选择离 `u` 最近、信息最充分的 forecast 作为 Plan：

```text
plan_action_u = latest covering P_i[k]       # 7D
Plan_u = bias_free_linear(plan_action_u)     # 128D
```

所有更早 covering forecasts 都与最新 Plan 比较：

```text
r_(i,u) = plan_action_u - P_i[k]

signed_mean_u = mean_i(r_(i,u))              # 7D
per_dim_rms_u = sqrt(mean_i(r_(i,u)^2))       # 7D

direction_u = RMSNorm(
    bias_free_MLP([signed_mean_u, per_dim_rms_u])
)

m_u = sqrt(mean_(i,dim)(r_(i,u)^2))
Revision_u = stopgrad(m_u) * direction_u      # 128D
```

若没有更早 forecast 或所有 forecast 一致，Revision 严格为零。没有人工
`tau`、count/mean/std/max additive statistics，也不把 old/new absolute action
直接塞进 Revision。

**意图。**

- Plan 表示当前最可信的未来动作；
- Revision 表示多次 rolling forecast 对同一绝对时刻的一致性、修正方向与
  原始无量纲分歧强度；
- `stopgrad(m_u)` 防止 AS loss 通过任意放大/缩小 forecasts 来操纵“置信度”；
- 方向仍可通过 forecast 路径学习。

Plan 和 Revision 不是交错两个 tokens，而是固定 concat 成同一
`Belief_u[256]`，避免把本来属于同一绝对时刻的状态拆开：

```text
Belief_u = concat(Plan_u[128], Revision_u[128])
```

代码：
[`temporal.py`](../src/ember/writer/temporal.py) 中
`ForecastBeliefEncoder`。

### 3.8 两层 Temporal Transformer

- width 256，8 heads，2 blocks；
- 对 absolute control time 使用 1D RoPE；
- Plan/Revision type、latest lead、revision count、detached strength 等
  routing 只进入 Q/K；
- V 和 residual 传递 raw Belief content；
- 不减 temporal mean；
- 初始化为 identity-safe/zero-preserving。

**为什么只有两层。** 当前目标不是让 Temporal 重新推理视觉，而是对已经对齐
到绝对时间的 Belief 做有限时序整合。step75 内部证据显示两层会压缩但不会消灭
顺序差异，所以没有证据表明深度是当前首要瓶颈。专家仍可审视这一判断。

**为什么 value 不做普通 RMSNorm。** Revision 的范数本身表达 forecast 分歧
强度；若所有 value 都归一到单位长度，`m_u` 的含义会丢失。Q/K 可以归一化以
稳定 attention，V 需要保留 magnitude。

代码：
[`temporal.py`](../src/ember/writer/temporal.py) 中
`VariableTimeTemporalEncoder`。

### 3.9 Content-conditioned LoRA query decoder

共有 320 个 public-LoRA routing queries，对应 module/layer/rank identity。
每个 query 有静态 routing `R` 和从零开始的 content state `Z`：

```text
Z_0 = 0

CrossAttention:
    Q = Norm(Z) + R
    K = Norm(temporal memory)
    V = raw temporal memory
    Z <- Z + attention(...)

SelfAttention:
    Q/K = Norm(Z) + R
    V = Z
    Z <- Z + attention(...)

Z <- Z + FFN(Z)
factor_heads(Norm(Z)) -> LoRA factors
```

静态 `R` 只决定“这个 module/layer/rank 应读什么”，不能进入 factor heads。
因此 decoder 不读取 temporal content 就不能生成公共 LoRA。当前为 2 blocks；
如果 temporal memory 已明显区分而 query output 再次坍缩，才有理由加深或拆
read。step75/v4 没有出现这种内部坍缩。

代码：
[`temporal.py`](../src/ember/writer/temporal.py) 中 `LoRAQueryDecoder`，
以及 [`model.py`](../src/ember/writer/model.py) 中 factor heads/schema。

### 3.10 参数量与训练配置

| 模块 | trainable parameters |
|---|---:|
| visual-state | 756,992 |
| VL Meta-LoRA | 921,600 |
| Action Meta-LoRA | 1,253,376 |
| Plan/Revision | 37,376 |
| Temporal | 1,640,192 |
| LoRA query decoder | 2,191,104 |
| factor heads | 3,498,432 |
| **Writer total** | **10,299,072** |

四卡 rank-128 Source-SFT 是 `10,297,344` 参数；Writer 只多 1,728
（`0.017%`）。public output 仍是固定 rank-16、38 targets、76 tensors、
`1,287,168` scalars。

正式训练使用 GPU 0–3、4 DDP ranks、batch 20/rank、frame microbatch 32、
stride 5、bf16、AdamW、100-step warmup 后 cosine schedule。v4 从 fresh
identity 训练到 step2400 后已停止；累计 `192,000` action queries，24 个
train tasks 各 `8,000`，每 task 50 条 teacher demos 全覆盖。当前不继续训练，
也不进入 RL。

### 3.11 75-step 内部机制 gate：通过

在正式学习率时间轴前 75 steps，8 validation tasks × 2 videos 的配对内部诊断：

| normal → condition | same-task other | shuffled | reversed | cross-suite wrong |
|---|---:|---:|---:|---:|
| effective-LoRA relative L2 median | 0.0250 | 0.0468 | 0.0420 | 0.0714 |

补充结果：

- normal forecasts 的跨帧变化 RMS 约占总 RMS 的 `60.4%`，不再是近似重复
  task-level chunk；
- permutation 对齐后，reversed/shuffled visual coordinates 仍变化
  `0.1610/0.1463`，future forecasts 变化 `0.0231/0.0152`；
- Belief 差异 `0.8217/0.7852`；
- Temporal 差异 `0.6902/0.6428`；
- query output 差异 `0.0528/0.0593`；
- final effective LoRA 差异 `0.0420/0.0468`。

所以 v4 解决了“内部顺序信息最终只剩万分之一”的工程/表示问题。

### 3.12 正式 validation 曲线：绝对性能下降

弃用波动较大的 80-episode screen 后，在完全相同的固定
8 tasks × 50 episodes panel 上：

| step | 675 | 825 | 900 | 1200 | 1275 | 1500 | 1875 | 2100 | 2400 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| success/400 | 100 | **109** | 82 | 96 | 94 | 92 | 90 | 90 | 89 |

本轮 observed-best 是 step825 的 `109/400`。它：

- 比四卡 Source-SFT `108/400` 高 1；
- 比旧 Action-Forecast v1 best `125/400` 低 16；
- 与旧版逐 episode 配对后，新增成功 30、丢失成功 46；
- 净下降 16 全部集中于 Object-1 `45→38` 和 Object-3 `20→11`；
- 其余六个 tasks 合计净变化为 0。

因此不是换 panel 或随机抽到更难视频造成的普通波动；新表示在 object 精确
识别、接近、抓取类任务上付出了真实代价。

### 3.13 Step825 内部特异性：层级看似合理

8 validation tasks × 2 reference videos，共 64 个 paired comparisons：

| normal → condition | same-task other | shuffled | reversed | cross-suite wrong |
|---|---:|---:|---:|---:|
| effective-LoRA relative L2 median | 0.0955 | 0.2598 | 0.3255 | 0.8762 |
| Temporal relative L2 median | 0.1196 | 0.9348 | 0.9085 | 0.7533 |
| Revision-strength candidate/reference | 1.014 | 1.419 | 1.340 | 1.038 |

内部层级符合直觉：

- 同任务另一正确 teacher 的变化最小；
- shuffle/reverse 明显增加 forecast disagreement；
- wrong 主要改变 LoRA 语义/方向，而不是简单增加 Revision 强度；
- shuffled/reversed LoRA RMS 相对 correct 为 `0.988/0.964`，并未把 adapter
  缩回 identity。

换言之，v4 确实读取并传播视频内容与顺序，失败不再是公共 LoRA 坍缩。

### 3.14 Step825 完整 rollout 特异性：行为 gate 反向失败

同一固定 400-episode panel 的结果：

| condition | total | Long-1 | Long-2 | Goal-3 | Goal-6 | Object-1 | Object-3 | Spatial-1 | Spatial-3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| correct | 109 | 6 | 2 | 0 | 40 | 38 | 11 | 0 | 12 |
| same-task other | 104 | 6 | 2 | 1 | 37 | 35 | 11 | 0 | 12 |
| cross-suite wrong | 99 | 0 | 3 | 2 | 44 | 39 | 7 | 2 | 2 |
| shuffled | **148** | 8 | 1 | 0 | 45 | 45 | 37 | 2 | 10 |
| reversed | 126 | 6 | 2 | 0 | 41 | 48 | 20 | 3 | 6 |
| shuffled_keep_first | 136 | 9 | 1 | 0 | 45 | 45 | 26 | 1 | 9 |

Paired 统计：

| correct vs condition | both success | correct-only | condition-only | both fail | exact McNemar p |
|---|---:|---:|---:|---:|---:|
| same-task other | 80 | 29 | 24 | 267 | 0.583 |
| cross-suite wrong | 69 | 40 | 30 | 261 | 0.282 |
| shuffled | 85 | 24 | 63 | 228 | `3.48e-5` |
| reversed | 79 | 30 | 47 | 244 | 0.0675 |
| shuffled_keep_first | 91 | 18 | 45 | 246 | `8.98e-4` |

关键事实：

1. `same-task other=104` 与 `correct=109` 最接近，说明同任务高层共性部分存在；
   但仍有 53/400 个 deterministic outcome flips。
2. `wrong=99` 只净低 10，且各 task 方向不一致，没有形成稳定语义伤害。
3. `shuffled=148` 比 correct 高 39，显著性很强，而且明显超过四卡
   Source-SFT `108` 和旧 Action-Forecast v1 best `125`。
4. `reversed=126` 也比 correct 高 17，改善集中在 object tasks。
5. shuffle/reverse 的收益不是把 LoRA 缩回 identity，而是生成了不同且更有效
   的 adapter 内容。

### 3.15 固定原始首帧的 shuffle 归因

v4 visual-state 使用 `X_0` 作为 anchor，因此完整 shuffle 同时改变了：

- 哪一帧充当 initial anchor；
- 后续帧顺序和 local transitions。

为隔离第一项，`shuffled_keep_first` 复用 full-shuffle 完全相同的 deterministic
permutation，只把原始 frame0 移回首位，其他非零 frames 的相对顺序不变。

结果：

- correct `109`；
- keep-first shuffle `136`；
- full shuffle `148`。

correct vs keep-first 净增 27，`p=8.98e-4`。所以即使初始 anchor 完全正确，
打乱后续时序仍显著更好；随机 anchor 不是主因。

full vs keep-first 的 paired counts 为 both/full-only/keep-only/both-fail
`116/32/20/232`，full 净高 12，`p=0.126`。这 12 的差异几乎全部集中在
Object-3（`37→26`，full-only/keep-only `14/3`，`p=0.0127`）。两项干预可能
存在非线性交互，所以不能把 39 严格做因果加法分解；但直接观测上，恢复原始
anchor只把148降到136，而固定anchor下仍保留相对correct的27次净增。随机anchor
不是异常的必要条件，也不足以成为主要解释。

有 14 个 episodes 的 full-shuffle permutation 本来就以 frame0 开头；它们在
full 与 keep-first 下产生完全相同 LoRA hashes，且 14/14 rollout outcome
一致，验证反事实没有引入额外随机变化。

---

## 四、目前的困惑，以及希望专家重点分析的问题

我们不把下面任何解释当作既定结论。希望专家首先判断问题是否被正确表述、目标
是否可辨识、现有证据更支持哪些解释，以及下一组最有区分力的诊断是什么。

### 4.1 最核心的悖论

v4 只用正常顺序 correct videos 做 positive AS 训练。它已经：

- 让每帧 visual-state 和 action forecasts 显著变化；
- 让 normal/shuffled/reversed 的差异穿过 Plan、Revision、Temporal、query
  decoder 到最终 LoRA；
- 在内部量上呈现 same-task < shuffled/reversed < wrong 的合理层级；
- 使 shuffled/reversed Revision strength 明显上升，说明模型确实认为它们的
  rolling forecasts 更不一致。

但实际 rollout 中，correct 不是最佳，反而 shuffled 显著最佳。为什么一个只在
correct order 上训练的模型，会让破坏顺序后的 adapter 更有效？

我们希望专家区分至少以下可能性：

1. **Forecast 语义未校准。** Action Expert 的 per-frame outputs 虽不同，却
   不一定真是“此 observer state 下机器人未来应怎样动”。Meta-LoRA 可能只学到
   一种能降低 task-level AS loss 的 latent code，Plan/Revision 随后忠实处理了
   语义错误的 forecasts。
2. **训练目标不可辨识。** 每个 task 有唯一 language，policy 本身收到正确
   language/current observation；teacher video 与 action episode 不配对。
   Positive task-level action loss 只要求生成对该 task 有用的 LoRA，并没有
   直接说明哪部分视频变化是高层因果阶段，哪部分是低层 nuisance。结构能禁止
   最简单的静态旁路，但未必能保证 learned coordinates/forecasts 具有我们给它
   命名的语义。
3. **正确连贯时序被错误映射。** stable task/object information 本身有用；
   coherent order 路径可能额外加入了有害的视角估计、低层 trajectory 或
   forecast bias。shuffle/reverse 破坏这个分量后，task/object information
   反而占主导，产生更强 policy。
4. **Revision magnitude 被错误使用。** shuffle/reverse 让 Revision strength
   增加约 34%–42%。decoder 是否可能把“大分歧”系统性解释成某种更激进、
   更强或更接近 Source-SFT 的 object manipulation adapter，而不是把它理解为
   低置信度？
5. **顺序 gate 的科学定义可能需要重审。** 最终输出是一个静态 task LoRA，
   某些 LIBERO task 的目标、物体和终态或许主要由 frame set/endpoints 决定。
   正常顺序是否理论上必须优于 shuffled，还是只应要求 wrong task 有害、
   same-task demos 稳定？另一方面，当前 shuffled 的显著大幅提升集中在需要
   精确操作的 object tasks，不能简单写成“顺序无关”。

### 4.2 为什么 wrong-video 的内部变化极大，行为伤害却不稳定

wrong-video effective LoRA relative L2 中位数 `0.876`，远大于 shuffle/reverse，
但 rollout 只有 `109→99`，`p=0.282`，而且 task 间方向相反。

需要解释：

- relative L2 是否主要发生在 policy function 不敏感的参数方向；
- frozen policy 的正确 language/current observation 是否覆盖了错误 LoRA
  内容；
- Writer 是否生成了“不同但都像通用 LIBERO adapter”的参数；
- 现有内部 specificity 指标是否需要改为 function-space/Jacobian-weighted
  差异，而不是裸 `B@A` L2；
- 如何不读取 forbidden signals 地验证 LoRA 变化究竟改变了哪些 action
  semantics。

### 4.3 同任务另一 teacher 稳定，但 correct order 仍有害，说明什么

same-task other 只有净 `-5`，远小于 wrong/shuffle/reverse。这不支持“v4 对
所有 demo-specific 低层轨迹都普遍过敏”这一过强解释。更具体的嫌疑似乎是：

- 正常 order 映射本身有系统性偏差；
- local transition、absolute-time alignment、Action Expert receding-horizon
  forecasts 或 Temporal read 中某一处把连贯变化解释错了；
- 两条正常同任务 demos 都共享这种错误，所以彼此接近。

我们希望专家判断这一推理是否成立，以及应怎样进一步定位：

- 固定 visual states，只交换 forecast order；
- 固定 per-frame forecasts，分别消融 anchor/local change；
- 保留 frame set 和 original indices，只改变 Temporal adjacency；
- 将 Plan-only、Revision-only 或 magnitude/direction 分别做无训练
  counterfactual；
- 比较 LoRA 的 function-space action changes，而非仅比较参数；
- 观察 object tasks 中 gain/loss episodes 的 action phase、gripper timing 和
  approach behavior。

以上只是候选诊断，不代表我们希望全部实现。请优先给出最少、最有判别力的一组。

### 4.4 visual-state 的第一性原理是否足够

当前 visual-state 用 8D anchored nonrecursive representation 避免两种失败：

- 逐帧独立 reader 坍缩为 task latent；
- 递归 transition 积累漂移。

但仍有根本问题：

- `X_t-X_0` 和 `X_t-X_(t-1)` 是高维 projected tokens 的位置对应差分；图像中
  物体/机械臂发生空间移动时，同一 token index 的差是否真有状态/运动语义？
- odd/zero-preserving 只约束代数形式，不保证 coordinate 的物理可解释性；
- 8D bottleneck 是否太小、刚好，或仍足以编码 task ID；
- native digit embedding renderer 对 frozen PaliGemma 是否真等价于可理解的
  state，还是一个受限但仍任意的 soft prompt；
- `h_0+c_t` 中共享 `h_0` 是否让稳定场景内容压过 stage change；
- anchor/local 两种变化一起进入 ChangeReader，是否导致无法识别哪一路有益。

我们尤其希望专家判断：在没有 visual-state/forecast 对齐 supervision 的前提
下，是否存在一种既可学习、又能从结构上避免 task-latent、同时仍能表达绝对
state 和有向 transition 的更简单方案；或者这个目标本身不能只靠架构保证。

### 4.5 Plan/Revision/Temporal/decoder 是否仍符合第一性原理

在 v3/v4 诊断后，我们暂时认为：

- Plan 选最新 covering forecast 符合“信息最充分的当前预测”；
- Revision 对所有较早 forecasts 与 Plan 比较，比只比较相邻 forecasts 更直接
  表达一致性；
- raw residual RMS 乘 direction 是无超参数、保强度的表示；
- Plan/Revision concat 成同一 absolute-time Belief 比拆成两个 tokens 更自然；
- 两层 Temporal 和 content-only query decoder 在 v4 内部没有再消灭差异；
- 有价值的 temporal mean/task-global information 不应被直接删除。

但 behavior 结果说明“内部差异保留”并不等于“表示语义正确”。希望专家重新从
目标作用出发审查：

1. Plan/Revision 是否对独立 teacher/action pairing 有正确统计含义；
2. 以 latest forecast 为真值参考是否会把 observer-view estimation error 写入
   Revision；
3. `stopgrad(m_u)` 是防止上游操纵置信度，还是也阻断了必要校准；
4. 两层 Temporal 是否足以理解阶段关系，或者任何深度都无法补足监督缺口；
5. 单个静态 LoRA 是否适合承载视频中的时序方法，还是会天然把顺序压成
   task-level average；
6. content-only decoder 虽封堵静态 query bypass，是否还有更隐蔽的
   task-latent shortcut。

### 4.6 高层任务逻辑与低层 teacher 细节应如何分离

当前数据合同实际上给出一种 multi-instance learning 信号：同任务 video 和
action episode 独立配对，所以只有跨 demo 稳定的任务逻辑才与 action supervision
一致。理论上这应抑制 demo-specific trajectory；实测 same-task other 也确实
相对稳定。但 object tasks 上正常时序仍有害。

我们希望专家分析：

- 独立配对是否足以在有限 24 tasks/50 demos 下识别高层逻辑；
- unique task language 是否让模型根本没有必要从视频学习共同逻辑；
- 需要的是更好的 inductive bias、更多跨任务组合、不同数据采样，还是更直接
  但仍不泄漏 action 的 intermediate objective；
- 怎样判断一个候选改动是在恢复“靠近—抓取—搬运—放置”的抽象逻辑，而不是
  再次制造一个高性能公共 adapter 或人为 order classifier。

### 4.7 希望专家最终提供什么

优先级从高到低：

1. 对上述反常结果给出一到数个最合理、彼此可区分的解释，并明确哪些是数据支持、
   哪些仍是推测；
2. 判断 EMBER 当前的视频→forecast→Plan/Revision→LoRA 思路是否在信息论和
   优化目标上可辨识，哪些语义只是我们对 hidden states 的命名；
3. 指出当前架构中最不符合第一性原理或不必要复杂的部分，以及哪些模块其实应
   保留；
4. 给出少量、低成本、能最大区分候选解释的诊断/消融，说明每种结果分别意味着
   什么；
5. 在诊断之后，再给后续结构或训练修改的方向。可以明确建议“不应立即改架构”；
6. 不以 contrastive/order loss 强行制造差异，不依赖 forbidden teacher
   actions/proprio/reward，也不以增加任意容量作为默认答案。

我们完全接受专家结论可能是：“现有证据还不足以提出可靠修复，应该先做某个
诊断”，或“某项原先设定的成功 gate 并不合理”。当前最需要的是可信解释和探索
方向，而不是表面完整的解决方案。

---

## 远程仓库阅读路径与代码地图

仓库：<https://github.com/LinFyM/EMBER>，分支：`main`。

建议按以下顺序阅读：

1. **本文**：问题、历史、最新架构、结果与咨询问题；
2. [`action_forecast_writer_design.md`](action_forecast_writer_design.md)：
   v4 的 canonical 数学与执行合同；
3. [`concept.md`](concept.md)：EMBER 的信息墙、比较方法与 claim boundary；
4. [`pi05_as_writer_action_forecast_v4.json`](https://github.com/LinFyM/EMBER/blob/73c419137b6004d6578c3c784633a711dfb95e0c/configs/pi05_as_writer_action_forecast_v4.json)：
   v4 sealed schema/参数；
5. 核心实现：
   - [`visual_state.py`](https://github.com/LinFyM/EMBER/blob/73c419137b6004d6578c3c784633a711dfb95e0c/src/ember/writer/visual_state.py)；
   - [`action_forecast.py`](https://github.com/LinFyM/EMBER/blob/73c419137b6004d6578c3c784633a711dfb95e0c/src/ember/writer/action_forecast.py)；
   - [`temporal.py`](https://github.com/LinFyM/EMBER/blob/73c419137b6004d6578c3c784633a711dfb95e0c/src/ember/writer/temporal.py)；
   - [`model.py`](https://github.com/LinFyM/EMBER/blob/73c419137b6004d6578c3c784633a711dfb95e0c/src/ember/writer/model.py)；
   - [`as_step.py`](https://github.com/LinFyM/EMBER/blob/73c419137b6004d6578c3c784633a711dfb95e0c/src/ember/writer/as_step.py)；
6. inference/evaluation：
   - [`inference.py`](https://github.com/LinFyM/EMBER/blob/73c419137b6004d6578c3c784633a711dfb95e0c/src/ember/writer/inference.py)；
   - [`evaluation_cache.py`](https://github.com/LinFyM/EMBER/blob/73c419137b6004d6578c3c784633a711dfb95e0c/src/ember/writer/evaluation_cache.py)；
   - [`evaluate_pi05.py`](https://github.com/LinFyM/EMBER/blob/73c419137b6004d6578c3c784633a711dfb95e0c/scripts/evaluate_pi05.py)；
7. [`findings.md`](../findings.md)：
   从 “Fresh 1k source base” 到 “v4 step825固定首帧shuffle归因” 的逐阶段
   证据 ledger；
8. 如需核查 stable contracts：
   [`test_writer_model.py`](../tests/test_writer_model.py)、
   [`test_pi05_evaluation_runtime.py`](../tests/test_pi05_evaluation_runtime.py)、
   [`test_writer_evaluation_cache.py`](../tests/test_writer_evaluation_cache.py)。

更早阶段的 `expert_plan.md` 已随旧 SmolVLA 执行簇从工作树退役，历史原文保留
在 Git commit `149badc` 及其之前；它不是当前 authority，不应据此恢复旧
split、旧 runner 或旧 Writer。

本文已把咨询所需的本地实验 aggregate、逐任务结果、paired counts、显著性和
内部量嵌入远程仓库。外部专家不需要访问本地主机 output 目录即可分析当前问题。

---

## 外部复核后的全面根因诊断（2026-07-26）

专家建议之后，我们先用forecast-order移植和Revision因子交换定位到
absolute-time Plan/Revision是直接行为放大器；随后继续做hidden teacher-future
语义演化、visual-state neutralization、same-task demo几何、random permutation
共识、AS loss/gradient和forecast分量/维度移植。更完整的结论是：

> v4的问题不是单一模块。positive task-level AS与同task独立video/action pairing
> 无法识别demo的高层过程语义；32-token visual-state不是必要信息瓶颈；
> raw-image/Meta路径逐渐学习了低层demo phase/translation latent；
> absolute-time Plan/Revision最后把它放大成OOD translation controller。

最关键的新证据：

- step75/300/825的latest/earlier forecast MSE ratio为
  `0.966/1.043/1.087`，residual到真实误差修正的cosine为
  `0.335/0.254/0.238`，但AS loss同期持续下降；
- step825 neutral visual-state只改变forecast约`0.855%`；visual coordinates
  最终主要预测video progress，而raw-image/Meta forecast对同demo真实低层
  translation差异的相关达到`0.587–0.740`；
- 8个独立random permutations产生高度共识的LoRA delta，并非一次幸运shuffle；
- Object-1/Object-3固定100 episodes上：

| arm | success |
|---|---:|
| correct | 49 |
| no VL Meta at inference | 48 |
| no Action Meta at inference | 50 |
| remove all frame detail / lead-only | 40 |
| shuffle frame-main only | 72 |
| normal forecasts in shuffled slots | 72 |
| shuffle translation dims only | 79 |
| true shuffled | 82 |

translation-only相对correct净`+30`、`p=5.30e-6`，相对true shuffled只差3、
`p=0.607`。normal→shuffle的AS loss整体略差，且LoRA delta与negative AS
gradient近乎正交。因此扰动生成机制是稳定确定的，但它改善closed-loop
Object success的正号没有被AS objective识别；这不是shuffled视频被理解得更好。

第一轮曾据absolute-time证据拍板frame-local Intent + adjacent Transition。
全面复审证明该方案只删除最后一层放大器，仍会保留visual-state旁路和
action-shaped Meta latent，现已撤回为局部候选。完整数字、解释、排除项和
证据SHA见
[`action_forecast_writer_v4_root_cause.md`](action_forecast_writer_v4_root_cause.md)。

该咨询阶段之后，owner已批准不再预测7D action trajectory的Semantic Core +
Causal Procedure v5；本文上述“没有批准的v5”只描述咨询结束当时的历史状态，
不得作为当前停止条件。活动合同见
[`action_forecast_writer_v5_design.md`](action_forecast_writer_v5_design.md)。

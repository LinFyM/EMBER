# Action-Forecast Writer v4 因果诊断与 v5 架构决策

状态：2026-07-26。本文记录外部专家复核后完成的最小因果诊断，以及由直接证据
得到的下一版架构决定。

本文的边界是：

- 当前仓库代码和已有 checkpoint 仍是已封存的 v4；
- 本轮没有实现或训练 v5，没有继续 AS，也没有进入 RL；
- 本文覆盖 v4 文档中仍把 absolute-time Plan/Revision 当作未来活动路径的表述；
- v4 的实现、训练和历史结果仍以
  [`action_forecast_writer_design.md`](action_forecast_writer_design.md) 为准；
- 后续若获准实现，v5 必须原位替换 v4 的 belief owner，不保留两套活动路径。

## 1. 最终判断

step825 的异常不是由随机首帧、visual-state 对 shuffled context 的反应、
Revision strength 数值爆炸、Temporal 层数不足，或 LoRA query decoder 再次
丢失差异所主导。

直接证据定位到：

> **v4 把每帧局部 action chunk 按 teacher frame index 放到一个假定共享的
> robot absolute-time 轴上，再把错位后的 forecast residual direction 当作
> Revision 内容。这个“同一绝对时刻”的对应关系没有被数据或目标识别。
> Shuffle 主要通过改写这条 Revision direction 通道，偶然产生了更好的
> end-effector translation controller。**

因此下一版不应：

- 继续修补 Revision strength 的归一化或裁剪；
- 增加 Temporal 层数；
- 删除稳定的 task/scene 内容；
- 优先重做 v4 visual-state；
- 直接删除 Revision 后只保留当前 Plan；
- 加入 contrast/order loss 强迫 correct 与 shuffled 拉开。

下一版应：

> **保留逐帧 visual-state、两个 Meta-LoRA 和 frame-local Action Expert
> forecasts；删除跨帧 absolute-time overlap、latest-covering Plan、
> multi-forecast Revision 及其 count/strength/routing；改为
> frame-local Intent + ordered adjacent Intent Transition。**

## 2. 诊断合同

所有新诊断固定使用：

- v4 observed-best `step_00000825`；
- 同一 frozen π0.5-LIBERO source base、rank-16 public LoRA schema；
- 原 fixed validation task、language、init state、env seed 和 policy noise；
- Object-1/Object-3 固定各 50 states 作为唯一新增定向 rollout panel；
- teacher action/state、proprio、reward 和 outcome 不进入 Writer 诊断；
- GPU 只使用物理 `4,5,6,7`，不使用或干扰正在被他人占用的 `0,1,2,3`。

没有运行新的 full400 条件，没有训练模型，也没有读取 forbidden teacher
signals。阶段动作诊断只使用执行时 policy 本来会看到的 current observation。

以下记号中：

- `N→N`：normal context 产生的 per-image forecasts 放在 normal slots；
- `N→S`：同一批 normal-context forecasts 按 shuffle permutation 放入
  shuffled slots，不重新计算 visual-state/Action Expert；
- `S→N`：shuffled-context forecasts 按图像 identity 放回 normal slots；
- `S→S`：真实 shuffled condition。

这里的 `S` slot 始终保留原始递增 `frame_indices`。所以 `N→S` 精确隔离的是
forecast 之后的 slot、absolute-time alignment、Plan/Revision、Temporal 和
decoder 响应。

## 3. Forecast-order 四臂交叉移植

### 3.1 内部和动作函数

16 条 validation reference videos 上：

- `N→S` 相对 `S→S` 的 effective-LoRA delta：
  cosine 中位数 `0.999978`、magnitude ratio `0.99905`、
  residual ratio `0.00712`；
- 同一比较在真实 π0.5 action function 中为
  `0.999864 / 1.00020 / 0.01668`；
- `S→N` 相对 `N→N` 的 effective LoRA relative L2 中位数只有
  `0.00337`；
- `S→N` action chunk 相对 `N→N` 的 relative L2 中位数只有
  `0.00156`。

这说明按 image identity 对齐后，shuffled visual context 对每张图像 forecast
的影响很小。真实 shuffle 的绝大部分参数和动作变化，都来自把基本相同的
per-image forecasts 放到错误时间 slots。

### 3.2 Object-1/Object-3 定向 rollout

| condition | Object-1 | Object-3 | total |
|---|---:|---:|---:|
| correct / `N→N` reference | 38 | 11 | 49 |
| `S→N` | 36 | 11 | 47 |
| `N→S` | 43 | 29 | 72 |
| true shuffled / `S→S` | 45 | 37 | 82 |

配对结果：

- `S→N` 相对 correct 净 `-2`，churn `4/100`，`p=0.625`；
- `N→S` 相对 correct 净 `+23`，`p=4.31e-4`；
- true shuffled 相对 `N→S` 再净 `+10`，`p=0.0129`。

所以主效应明确位于 forecast 之后。shuffled context 仍有较小的非线性交互，
尤其在 Object-3，但它不是异常的必要条件或主要原因。当前没有证据支持优先做
anchor/local visual-state 分路。

## 4. Plan / Revision 因子交换

固定同一批 normal-context forecasts，只在 downstream 交换：

- `P`：Plan；
- `D`：Revision direction；
- `V`：乘在 Revision value 上的 residual RMS strength；
- `R`：进入 Temporal Q/K routing 的 strength。

`N` 表示 normal slot 因子，`S` 表示 `N→S` shuffled-slot 因子。

### 4.1 内部和初态 action-function 证据

- 只交换 routing strength：effective-LoRA target-delta magnitude
  `0.0061`；action target-delta magnitude `0.0112`，可忽略；
- 把 target 的 routing strength 恢复为 normal，action delta 仍与 target
  基本相同：cosine `0.99989`、magnitude `1.0013`；
- Plan、direction 和 value strength 都能改变动作，但具有明显非线性交互；
- executed first-5 action 上，完整 shuffled Revision 相对 `N→S` delta 的
  cosine/magnitude 中位数约 `0.972/0.969`；
- direction-only 约 `0.906/0.446`；
- value-strength-only 约 `0.366/0.412`；
- Plan-only 约 `0.818/0.696`。

shuffled strength 不是无界 OOD 爆炸：

- train-normal token strength median/p95/max 为
  `0.333/0.614/0.858`；
- shuffled-slot 为 `0.454/0.670/<0.858`；
- `14.4%` shuffled tokens 高于 train p95，但没有 token 超过 train max。

### 4.2 Object 行为因果结果

| factor arm | 含义 | Object-1 | Object-3 | total |
|---|---|---:|---:|---:|
| correct | normal 全部因子 | 38 | 11 | 49 |
| `PS_DN_VN_RN` | 只换 Plan | 37 | 24 | 61 |
| `PN_DN_VS_RN` | 只换 value strength | 39 | 15 | 54 |
| `PN_DS_VN_RN` | 只换 direction | 39 | 28 | 67 |
| `PN_DS_VS_RS` | 只换完整 Revision | 45 | 30 | 75 |
| `N→S` | 全部 shuffled-slot downstream | 43 | 29 | 72 |
| true shuffled | 重新计算 shuffled context | 45 | 37 | 82 |

相对 correct 的 paired 结果：

- Plan-only 净 `+12`，`p=0.0501`；
- value-strength-only 净 `+5`，`p=0.359`；
- direction-only 净 `+18`，`p=0.00143`；
- 完整 Revision 净 `+26`，`p=6.16e-6`。

direction-only 与 `N→S` 只差 5，`p=0.458`；完整 Revision 与 `N→S`
只差 3，`p=0.690`。完整 Revision 比 direction-only 高 8，但该配对差异
`p=0.200`。因此：

1. shuffled Revision direction 是主要行为中介；
2. value strength 单独不是主因，但与 direction 存在次级协同；
3. Plan 有次级贡献；
4. Q/K routing strength 没有可测的实质作用。

## 5. Policy-action 阶段诊断

在五条 Object 成功轨迹上，选择经过画面与 gripper qpos 核验的：

```text
approach
pre-grasp
gripper close
transport
place / terminal
```

共 25 个 current observations。每个 observation 使用完全相同的正确
language 和 flow noise，对 12 个 LoRA 反事实同时计算动作。driver 单臂执行
与多臂 probe 分开，probe 不影响环境轨迹。

主要结果：

- true `S→S` 与 `N→S` action delta 在五个阶段的 cosine 为
  `0.99927–0.99980`，magnitude ratio 为 `0.9979–1.0014`；
- `S→N` 只保留 `N→S` delta 的 `1.9%–4.6%`；
- 完整 shuffled Revision 在 approach/pre-grasp/close/transport 的
  action-delta cosine 为 `0.986/0.936/0.938/0.951`，
  magnitude 为 `0.827/0.821/0.628/0.701`；
- direction-only 的对应 cosine 为
  `0.867/0.830/0.927/0.849`；
- direction-only 对 executed translation 的 cosine 在
  pre-grasp/close/transport 为 `0.925/0.954/0.982`，
  magnitude 为 `0.601/0.529/0.662`；
- value-strength-only 的 behavior 和阶段动作贡献都明显更弱。

`N→S` 相对 normal 的 executed-action delta 主要位于 end-effector
translation。五阶段 translation RMS 中位数为 `0.0197–0.0341`，而
rotation 为 `0.00217–0.00770`、gripper 为 `0.00155–0.00459`。
考虑 LIBERO translation command 单轴上限约 `0.05`，这不是微小 null-space
变化，而是足以改变 approach 和 transport 轨迹的控制修正。

把 Revision 完全置零也不是合理修复。`normal Plan + Revision=0` 在五阶段
产生相对 `N→S` target delta 的 `2.1–5.8×` 动作变化，而且多阶段方向低相关
或反向。现有 Plan 和 Revision 已共同适配成一套耦合表示；直接删除其中一支
不会恢复干净的“Plan-only policy”。

## 6. 根因解释

v4 对 frame `i` 生成局部 action chunk：

\[
A_i[\ell],\qquad \ell=0,\ldots,H-1.
\]

随后使用：

\[
u=t_i+\ell
\]

把来自不同 teacher frames 的局部 chunks 放进同一个 absolute-time 轴，并把
最新覆盖 `u` 的 action 当作 Plan，其他覆盖 `u` 的 action 相对它的 residual
当作 Revision。

这要求：

\[
A_i[u-t_i]\quad\text{和}\quad A_j[u-t_j]
\]

确实是对同一个未来机器人控制时刻的可比预测。当前训练没有校准这一假设：

- teacher video 与监督 action episode 只共享 task，不共享 episode；
- observer video stage、robot rollout state 和控制时钟并不对齐；
- 以后的人类视频更不具有相同 embodiment、速度或 action clock；
- Action Expert forecasts 只受最终 task-level functional loss 间接约束。

四臂移植进一步证明，shuffle 并不是先让每张图像产生了更好的 forecast。它主要
把同一批局部 chunks 的不同 lead positions 错配成“同一时刻的多次预测”。
由此得到的 residual direction 不是校准的 confidence/revision，而是一条新的、
训练分布外的 action-shaped controller code。它在两个 Object tasks 上恰好给出
更好的 translation 修正。

因此“Revision direction 有用”不能被解释为 shuffled 更懂视频；它证明的是
当前 absolute-time residual 可以控制 policy，而且 normal residual 的语义没有
被识别。

## 7. v5：Frame-Local Intent + Ordered Transition

### 7.1 保持不变

第一版 v5 保留：

- task language + exactly one action-hidden video 的信息墙；
- frame stride `5`，不重新比较 stride 10；
- v4 的 32 个 native-subspace visual-state tokens；
- anchor/local visual change reader；
- trainable identity-init VL Meta-LoRA 和 Action Meta-LoRA；
- 每帧共同 flow noise；
- frozen π0.5-LIBERO source base；
- 完整 rank-16 public task LoRA schema；
- 两层 content-only Temporal；
- routing/content 分离的 LoRA query decoder；
- positive AS functional action loss和同任务独立 video/action pairing；
- stable task、language、scene、object 信息，不做时间均值删除。

保留两个 Meta-LoRA 是必要的：Action Meta-LoRA 仍负责把 observer-view 或未来
human teacher 理解为“假如机器人是 teacher，会怎样执行”。本轮证据没有把主要
问题定位到这个上游适配器，因此不先冻结或删除它。

### 7.2 删除

v5 原位删除：

- `_time_layout` / shared absolute robot-time expansion；
- latest-covering forecast `Plan_u`；
- same-absolute-time residual `Revision_u`；
- revision count、residual RMS strength 和 strength routing；
- `Belief_u=[Plan_u|Revision_u]`；
- lead/count/strength 的 Temporal routing features。

这些不是并行 ablation path；v5 实现后 v4 belief 只通过 Git 和封存结果保留。

### 7.3 Frame-local Intent

对第 `i` 帧，只在自己的局部 lead 坐标中解释 Action Expert chunk：

\[
A_i\in\mathbb R^{H\times 7}.
\]

使用一个 bias-free、zero-preserving local chunk encoder：

\[
I_i =
W_2\,\mathrm{GELU}\!\left(
W_1\,\mathrm{vec}(A_i)
\right)\in\mathbb R^{256}.
\]

其中：

- `H=50` 和 7 维 action 是 sealed π0.5 interface；
- flatten 的列位置只表示本帧 forecast 的 local lead，不声称跨帧共享绝对时刻；
- `bias=False` 使全零 chunk 严格生成零 content；
- 不在 `I_i` 上做会抹掉整体动作幅度的普通 RMSNorm；
- source-normalized action 值直接作为 content，frame/type identity 不进入 value。

一枚 256D `I_i` 足以保留固定 `50×7=350` 的局部 chunk 结构，同时比把
`T×50` action tokens直接送入全局 self-attention更高效、更简洁。

### 7.4 Ordered Transition

相邻帧只比较同一 local-chunk 表示：

\[
\Delta I_i=I_i-I_{i-1},\qquad i\ge1.
\]

它结构上满足：

\[
\Delta I(I,I)=0,\qquad
\Delta I(I_a,I_b)=-\Delta I(I_b,I_a).
\]

这不再声称两个不同 lead 的 robot actions属于同一 absolute time。它只表达：

> 随着 teacher 视频从上一阶段推进到当前阶段，机器人从该画面想象出的局部执行
> intent 怎样有向变化。

### 7.5 Temporal memory

Temporal 输入按视频顺序交错：

```text
I_0,
ΔI_1, I_1,
ΔI_2, I_2,
...,
ΔI_(T-1), I_(T-1)
```

- `I_i` 保留任务、物体、场景和当前阶段的绝对内容；
- `ΔI_i` 显式保留有向过程变化；
- Intent/Transition type 只作为 Q/K routing identity；
- frame ordinal 通过 RoPE 进入 Q/K；
- V 和 residual 只传递 `I_i` 或 `ΔI_i` content；
- 不增加 confidence、strength、temperature、clip threshold 或训练集分位数；
- 两层 Temporal 先保持不变，因为 v4 已证明两层足以把输入差异传到 decoder。

### 7.6 LoRA decoder

当前 content-only query decoder 原样保留：

```text
static module/layer/rank identity -> Q/K routing only
temporal memory                  -> V/content only
factor heads(Norm(content))      -> complete rank-16 LoRA
```

四臂和阶段动作证据均表明 decoder 已忠实传递 downstream memory；没有理由恢复
静态 query bypass、增加第二套 decoder，或先加深网络。

### 7.7 参数预算

v5 仍以 rank-128 Source-SFT 的 `10,297,344` trainable parameters 为容量参考。
删除 v4 belief/routing 后加入 `350→256→256` bias-free chunk encoder。实现时只
允许通过既有 factor-head hidden width 做一次机械容量校准，使 Writer 总参数量
与 comparator 差异保持在 `0.1%` 内；不得为匹配参数新增科学分支或公共 adapter。

## 8. v5 不能保证的事情

这一改动修复的是已被直接证明的错误对应关系。它仍不能仅靠命名保证：

- Action Expert chunk 已经是真实 calibrated future action；
- `I_i` 一定只编码高层逻辑而不含 task/style latent；
- positive task-level AS 一定会让 correct order 在所有 OOD shuffle 上占优。

因此 v5 的主张必须收紧为：

> 它让顺序信息来自 frame-local intent 的有向演化，而不是未经识别的跨帧
> absolute-time residual。

若 v5 仍出现 `shuffled > correct`，不得继续用 strength clipping、更多
Temporal 层或 contrast/order loss补洞。下一项原则性候选应是 action-hidden、
positive-only 的 causal video objective，例如让只看当前帧的局部 intent 预测
后续 frozen visual features；是否加入必须由 v5 的新证据另行决定，不是本次
已拍板实现的一部分。

## 9. 后续实现与实验门

本轮在架构决定处停止。后续若 owner 允许实现：

1. 原位替换 belief owner，不新增平行 v5 runner；
2. 做最小 shape/gradient/identity/freeze/resume/parameter checks；
3. 固定 stride 5，在实时核验后的 GPU 4–7 profile batch 和 frame microbatch；
4. fresh 训练到 75 step；
5. 先做低成本内部 normal/same/wrong/shuffled/reversed 检查；
6. 重新做 forecast/order transplant，确认顺序差异来自
   `I/ΔI` sequence，而不是重新出现错误 absolute-time slot code；
7. 内部门通过后只在 Object-1/Object-3 做必要 paired rollout；
8. 只有 correct-order 行为语义与绝对性能都通过，才进入充分 AS 训练；
9. 不通过且没有明确可纠正错误时停下讨论，不自动进入 RL。

当前仍禁止自动推进 cold-start RL、final-32、task-local RL、joint oracle 或
ViVLA。

## 10. 本地证据与校验哈希

远程读者不需要访问本地主机即可根据本文数字理解结论。主机上的 canonical
summary 及 SHA256 为：

| evidence | summary SHA256 |
|---|---|
| forecast-order internal transplant | `6b45475d111b66e8d091d4105e36976c466f4c46b3435db4935fbe6018ec413d` |
| forecast-order policy function | `e996caf0360fa331503684c4431929cb9696b42dd6ff57c20ea23dfc77b50df3` |
| forecast-order Object rollout | `517fbb2ffb6282e3290ba9ad4e5849d77c97a3dbef7d4af97a99b7cfab223000` |
| Revision factor internal exchange | `e4ff0be9c172db7049dc71e74b6ff05388129a6ce6256d546e7b7800180d88c6` |
| Revision factor policy function | `9b6a41dfba112db75e9911529c060b3286004e6adc1140fa6e17bea6d4506b48` |
| Plan/full-Revision Object rollout | `da2707ffa7104dccbfc0aadcfb116a3cb97390b29cbe7a589c747041fa457bfb` |
| direction/value-strength Object rollout | `92baa928c5ec4cfb4aca45234a4f573958805ef35aa3fa424e3d912694709f5a` |
| phase-specific policy actions | `24595de9981f19678cff47f69de55ab4b5f354e7bc9a99ddd12a826a943cb11f` |

本轮全部 GPU launch 使用物理 4–7；最终阶段探针峰值 reserved
`12,530,483,200` bytes/GPU。0–3 上的他人进程未被停止、重置或干扰。

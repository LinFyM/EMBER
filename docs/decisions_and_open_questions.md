# Decisions and Open Questions

## 已拍板

- 目标benchmark为LIBERO-Spatial/Object/Goal/Long；活动split为每suite 6 train / 2 validation / 2 test，总计24/8/8，final合并为32 source / 8 test。
- generic `pi05_base` 的0/400 feasibility已完成，只作原始校准。
- 先用与目标40 exact semantic/composition去重后的LIBERO-90 source corpus训练共享π0.5-LIBERO source base；每个active source task使用全部50条成功action episodes。
- source base训练若使用LoRA，完成后merge进policy；所有下游方法共享同一个冻结base和source-only normalization，不叠加shared source adapter。
- source base在全部目标40 tasks上快速测试，只要求已开始在多个tasks出现部分真实成功，不先追求高ceiling，也不能只靠单个易task aggregate。
- `Action-Supervised Writer (AS-Writer)`输入正确language+恰好一条action-hidden video；source video/action episode在同task内独立采样。
- 32-token Action-Forecast v4的75-step内部门已通过；fresh正式轨迹已训练到
  step2400并停止，不再继续训练或使用波动过大的80-episode快筛。固定400 panel
  的现有observed-best为step825的`109/400`。
- step825完整特异性为correct/same-task other/cross-suite wrong/shuffled/
  reversed=`109/104/99/148/126`；固定原始首帧只打乱后续帧仍为`136`。
  因此v4解决了内部视频/顺序差异塌缩，却没有正确使用这些差异；行为硬门失败。
- 外部复核后的forecast-order四臂移植、Revision因子交换和阶段动作诊断已完成。
  主因是v4未经识别的跨帧absolute-time forecast alignment：同一批normal
  per-image forecasts只要换到shuffled slots，就在Object-1/Object-3上把
  `49/100`提高到`72/100`；shuffled-context forecasts放回normal slots只有
  `47/100`。主要行为中介为Revision direction，而非strength、Q/K routing、
  Temporal深度或decoder差异坍缩。
- 下一版原位删除absolute-time Plan/Revision/Belief，改为frame-local Intent
  与adjacent ordered Intent Transition；保留32-token visual-state、两个
  trainable identity Meta-LoRA、frame-local forecasts、两层content-only
  Temporal和content-only LoRA decoder。完整决定见
  [`docs/action_forecast_writer_v5_decision.md`](action_forecast_writer_v5_decision.md)。
- frame stride固定为5，不再把stride 5/10作为待选变量；后续任何GPU工作只使用
  物理GPU 4–7，0–3不进入visible set也不被干扰。
- 当前工程推进以效率优先：最短垂直切片通过必要的shape/gradient/
  identity/freeze/resume检查后立即真实profile/训练，不用广泛全仓校验、
  重复流程门槛或文档整理延迟GPU启动。
- `Reward-Trained Writer (RL-Writer)`从新架构规定初态做短、task-balanced AS cold start，直到24个train tasks各在official random-reset rollout中至少一次success，再关闭action入口转pure reward；不从完整AS-Writer best继续。
- Source-SFT是在24/32 source tasks上联合训练的一套shared LoRA；它与AS-Writer各自按validation选最佳，不要求相同训练steps或数据量。
- 第一轮完整流程只跑一个training seed；有足够性能差异后再考虑独立seeds。
- development和final都必须报告seen-task performance。
- wrong-video直接来自另一suite；保持正确language、task、init state和policy RNG，只替换Writer输入视频。
- validation后AS-Writer、RL-Writer（若成立）、Source-SFT均在32 source tasks上从规定初态重训一次。
- task-local RL只在test打开后做，不在validation上预冻结算法。每个test task上直接调优并训练到接近最佳。
- task-local RL三臂为identity、AS-Writer、RL-Writer初始化；两个Writer臂每task/adaptation seed共用一条随机video并固定初始化LoRA。
- 所有RL rollout与checkpoint选择使用官方random reset/BDDL初态；fixed50 states只用于fresh evaluation。
- RL reward使用官方env reward/success，不手工读取内部object pose构造privileged shaping。
- direct target-action oracle使用8 test tasks×50 episodes联合训练一套shared LoRA，不是8套task-local LoRA；第一轮只做50/task。
- 已提前看过generic/source-base feasibility不再作为“test purity”阻塞理由。
- ViVLA-style matched reproduction有时间再做；outer learning仅为核心之后optional。
- 不用`pi05_libero`、bank、geometry、shared update subspace、residual escape、额外shared adapter、旧SmolVLA活动checkpoint/runner或MemLLM。

## 已完成的source overlap audit

本地官方task suite的只读audit先发现：

- LIBERO-90 task44与目标`libero_goal` task7均为`turn on the stove`，scene/BDDL不同；后者当前是test task。
- LIBERO-90 task77与目标`libero_10` task5均为`pick up the book and place it in the back compartment of the caddy`，scene/BDDL不同。

完整3600-pair semantic/composition audit随后已在看新policy outcome前封存；
最终排除19项、保留71个active source task及其manifest/hash。新session不得
重新修改source IDs或把这两条初步发现误写成audit仍未完成。

## 外部复核后的机制定位与下一版决定

Phase A、source base、public rank-16 LoRA合同、functional per-sample注入、
Source-SFT comparator、v4训练、现有checkpoint选择和step825完整特异性均已
封存，不得重新开启。本轮新诊断也已在架构决定处停止：

- image identity对齐后，shuffled context对per-image forecasts及policy action
  的残余影响分别只有千分位量级；把normal forecasts放入shuffled time slots
  几乎完整重现真实shuffle的LoRA和action delta；
- Object定向rollout中`correct/S→N/N→S/S→S=49/47/72/82`，证明主效应位于
  forecast之后，真实shuffled context只提供较小的额外交互；
- `Plan-only/strength-only/direction-only/full Revision=61/54/67/75`，
  Revision direction是主要行为中介，strength单独不是主因，Q/K strength
  routing可忽略；
- 阶段动作探针显示该方向主要改写pre-grasp、close和transport的end-effector
  translation，不是参数空间差异却行为无效；
- 直接删除Revision会产生比目标shuffle delta大`2.1–5.8×`且多阶段反向的
  动作变化，说明现有Plan/Revision已经共同适配，不能做Plan-only热修。

据此，v5只修复被直接证明的错误对应关系：每帧`50×7` local action chunk先编码
为256D Intent，相邻帧用`ΔI_i=I_i-I_{i-1}`表达有向演化，再按
`I_0, ΔI_1, I_1, ...`进入两层Temporal。它不引入confidence/strength超参数、
时间均值删除或contrast/order loss，也不声称positive task-level AS已能保证
高层语义可辨识。

当前仍开放的科学问题只有：在移除错误absolute-time语义后，positive AS和独立
video/action pairing能否使Intent演化对应可迁移的任务过程，而不是另一种
task/style latent。若后续75-step v5仍出现inversion，原则性下一候选是
action-hidden、positive-only的causal future frozen-visual-feature prediction；
它尚未被批准为v5第一版的一部分。

## 不再开放的问题

- 是否需要共享source base：需要，且来自过滤后的LIBERO-90，不是目标24/32 action-SFT。
- Source-SFT是否机械匹配AS-Writer step budget：不匹配，各自按validation选最佳并报告实际成本。
- task-local RL是否必须先在validation冻结：不需要；只在test task上训练与调优。
- direct oracle是否per-task：不是；8个test tasks联合一套shared LoRA。
- wrong-video是否需要hard same-suite negative：第一轮不需要，直接cross-suite。
- 第一轮是否做多seed或action-budget curve：不做。

## 当前设计覆盖核验

以下 owner 细节均已进入活动 authority、执行计划或 canonical design：

- one-video train/test 信息墙；source 内 video/action 独立抽样；held 每 rollout 随机正确视频；
- 四 suites 的 6/2/2 development split、validation 合并后的 32/8 final split；
- 过滤 LIBERO-90 overlap 后训练并 merge/freeze 共享 π0.5 source base，40-task 快速能力筛查；
- 新visual-state已完成75-step内部闭环和0→2400正式轨迹；现有best与完整
  correct/same/wrong/shuffled/reversed/fixed-anchor证据均已封存；外部复核后的
  最小因果诊断已把主因定位到absolute-time Plan/Revision并形成v5决定，当前在
  实现和训练前停止；
- RL-Writer 使用独立短AS cold start取得24-task逐task成功覆盖后转pure reward，并完整报告action消耗；
- Source-SFT 是一套 shared target-source LoRA，独立按 validation 选最佳；
- seen/source panel 与 cross-suite wrong-video 对照；
- final 第一轮单 training seed，不提前扩多 seed；
- test-only 三臂 task-local RL 直接训练到各自平台，不在 validation 预冻结；
- 所有 RL/adaptation rollouts 使用官方随机 BDDL/reset，三臂匹配 seeds/初态序列并保存
  worker RNG、schedule 与 interaction cursor；fixed 50 states 只作 fresh evaluation；
- direct oracle 是 8 test tasks × 50 episodes 联合训练的一套 shared LoRA；
- 动态 cost-balanced rollout 调度、每卡相同 CUDA 进程数、GPU0 无额外角色、等待长任务时
  继续推进互不污染工作，以及不影响科学结论时效率优先；
- ViVLA-style 与 outer learning 只在核心之后可选；禁止 `pi05_libero`、旧 SmolVLA 活动路径、
  bank/geometry/shared subspace/residual escape/额外 shared adapter 和 MemLLM。

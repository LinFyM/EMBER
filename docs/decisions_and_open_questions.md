# Decisions and Open Questions

## 已拍板

- 目标benchmark为LIBERO-Spatial/Object/Goal/Long；活动split为每suite 6 train / 2 validation / 2 test，总计24/8/8，final合并为32 source / 8 test。
- generic `pi05_base` 的0/400 feasibility已完成，只作原始校准。
- 先用与目标40 exact semantic/composition去重后的LIBERO-90 source corpus训练共享π0.5-LIBERO source base；每个active source task使用全部50条成功action episodes。
- source base训练若使用LoRA，完成后merge进policy；所有下游方法共享同一个冻结base和source-only normalization，不叠加shared source adapter。
- source base在全部目标40 tasks上快速测试，只要求已开始在多个tasks出现部分真实成功，不先追求高ceiling，也不能只靠单个易task aggregate。
- `Action-Supervised Writer (AS-Writer)`输入正确language+恰好一条action-hidden video；source video/action episode在同task内独立采样。
- AS-Writer单次训练不超过约2小时；根据loss斜率和val成本尽快找到接近饱和点，不频繁做昂贵完整val。
- `Reward-Trained Writer (RL-Writer)`先完全无AS warm-up直接用source reward训练；无信号时加极少warm-up，仍失败就暂停，不从完整AS-Writer继续。
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

## 已核验但待封存的source overlap

本地官方task suite的只读audit发现：

- LIBERO-90 task44与目标`libero_goal` task7均为`turn on the stove`，scene/BDDL不同；后者当前是test task。
- LIBERO-90 task77与目标`libero_10` task5均为`pick up the book and place it in the back compartment of the caddy`，scene/BDDL不同。

这两个至少必须进入完整semantic/composition overlap audit；最终过滤规则、active source IDs和数量在看任何新policy outcome前封存。

## 尚需通过官方/成熟实现确定

这些是下一session的Phase A工作，不需要owner重新选择研究方向：

- π0.5 source action-SFT是full fine-tune、action expert还是LoRA-then-merge；以成熟recipe和实际8-A100 profile决定。
- 下游LoRA targets/rank/alpha/dropout与functionally identity initialization。
- Writer对π0.5的feature extraction、functional per-sample LoRA注入和checkpoint格式。
- RL-Writer与task-local RL在π0.5 flow policy上的最佳可执行reward-update算法；task-local算法直接在test tasks上调优。
- 训练loss斜率触发val screen的具体steps，以及完整validation候选数；目标是最少昂贵rollouts下可靠早停。
- seen panel的精确task IDs/episode数量；必须specification-only覆盖四suites并兼顾评测成本。
- 高吞吐evaluator选择batched functional LoRA还是每卡统一多policy replicas；按真实rollouts/s profile决定。
- ViVLA matched implementation的具体范围，仅在核心闭环后处理。

## 不再开放的问题

- 是否需要共享source base：需要，且来自过滤后的LIBERO-90，不是目标24/32 action-SFT。
- Source-SFT是否机械匹配AS-Writer step budget：不匹配，各自按validation选最佳并报告实际成本。
- task-local RL是否必须先在validation冻结：不需要；只在test task上训练与调优。
- direct oracle是否per-task：不是；8个test tasks联合一套shared LoRA。
- wrong-video是否需要hard same-suite negative：第一轮不需要，直接cross-suite。
- 第一轮是否做多seed或action-budget curve：不做。

## 交接覆盖核验

以下 owner 细节均已进入活动 authority、执行计划或 handoff prompt：

- one-video train/test 信息墙；source 内 video/action 独立抽样；held 每 rollout 随机正确视频；
- 四 suites 的 6/2/2 development split、validation 合并后的 32/8 final split；
- 过滤 LIBERO-90 overlap 后训练并 merge/freeze 共享 π0.5 source base，40-task 快速能力筛查；
- AS-Writer 约 2 小时上限、loss 驱动的低成本 validation 与早停；
- RL-Writer 从随机初始化和零 warm-up优先，极少 AS warm-up 仅作失败恢复；
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

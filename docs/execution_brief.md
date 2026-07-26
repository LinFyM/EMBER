# EMBER Current Execution Brief

状态：2026-07-26。共享 π0.5-LIBERO source base 与 Source-SFT comparator 已
封存。四卡 Action-Forecast Writer v4 已训练至step2400并停止；现有checkpoint
observed-best为step825的`109/400`。内部视频/顺序特异性通过，但完整rollout
出现`shuffled=148/400 > reversed=126/400 > correct=109/400`的反常排序；
固定原始首帧后shuffle仍为`136/400`。

外部专家复核后的最小因果诊断现已完成。forecast-order四臂交叉移植证明，
shuffle的主要行为效应发生在per-image forecast之后：v4把frame-local action
chunks投到一个没有被训练合同识别的共享robot absolute-time轴，再把错位
residual direction解释为Revision。Object-1/Object-3上，normal forecasts放入
shuffled slots把`49/100`提高到`72/100`，而shuffled-context forecasts按image
identity放回normal slots为`47/100`。Revision direction-only为`67/100`，
完整shuffled Revision为`75/100`；strength-only只有`54/100`。因此主因不是
visual-state、strength爆炸、Temporal层数或decoder坍缩。

下一版决定删除absolute-time Plan/Revision/Belief，保留32-token visual-state、
两个Meta-LoRA和content-only下游，改为frame-local Intent + adjacent ordered
Transition。当前只完成架构决策，不实现或训练v5，不进入RL。完整证据、数学和
后续gate见
[`docs/action_forecast_writer_v5_decision.md`](action_forecast_writer_v5_decision.md)；
原咨询材料见
[`docs/action_forecast_writer_expert_consultation.md`](action_forecast_writer_expert_consultation.md)。

## 1. 研究问题

机器人 action trajectories 对目标任务稀缺时，一条 action-hidden teaching video 是否能让共享 Writer 为一个已有通用控制能力的 VLA 生成更好的完整 task-specific LoRA？直接 target-action SFT 是“教练拉手”的 privileged oracle；Writer zero-interaction 是“看过教程后的第一次尝试”；test-task reward adaptation 是后续自行练习，不是 EMBER 必须存在的尾巴。

## 2. 目标 benchmark 与 split

- 目标任务来自 `libero_spatial`、`libero_object`、`libero_goal`、`libero_10` 四 suites，共 40 tasks。
- development split 为每 suite 6 train / 2 validation / 2 test，总计 24/8/8，封存在 `configs/libero_24_8_8_v1/`。
- split 只使用 task identity、language 和 BDDL/specification；不得按新 outcome 改 IDs。
- development 完成后将 8 validation tasks 合入 source，最终为 32 source / 8 test；第一轮完整流程只跑一个 training seed。

## 3. 共享 π0.5-LIBERO source base

所有主方法的共同起点统一定义为：

```text
generic lerobot/pi05_base
→ specification-only 过滤 LIBERO-90 与目标40 exact semantic/composition overlap
→ 在剩余 source tasks、每 task 全部50条成功action episodes上联合SFT
→ 若使用source LoRA则merge进policy
→ 冻结共享、多任务、语言条件的 π0.5-LIBERO source base
```

已知必须审计的同语言 overlap 至少有 LIBERO-90 task 44 对目标 `libero_goal` task 7，以及 LIBERO-90 task 77 对目标 `libero_10` task 5；完整 audit 决定最终 source task 数。禁止使用已经在目标40 actions上训练的 `pi05_libero`。

source action/state normalization 只由过滤后的 LIBERO-90 source corpus计算，并随 base 冻结供所有下游方法共用。source-base LoRA recipe、targets与训练参数必须先参考官方或成熟 π0.5 fine-tuning 项目，不能猜。

base 不追求高 ceiling。训练期间用全部目标40 tasks做小型快速screen，确认它已开始在该 benchmark 上出现跨多个task的部分真实成功；不要求每task已有高成功率，但不能只靠一个易task的aggregate。generic π0.5 的正式结果 `0/400` 仅为原始校准，不能替代新 source base 结果。

owner于2026-07-22将source-base正式训练改为从generic base fresh运行1,000 optimizer steps，333-step线性warmup后到官方peak LR；旧30k attempt在step2880停止且无checkpoint，不得resume。这里的目标只是轻量获得LIBERO embodiment/control interface，而不是在LIBERO-90上收敛或过拟合。

## 4. Development methods

### AS-Writer

- 输入恰好一条 action-hidden teacher video + 正确 task language；输出完整 task-specific LoRA。
- 在24 train tasks上做均衡混合。每个 update 同 task 内独立随机采 video 与 action episode/chunk，不要求配对；action只进 functional behavior loss。
- source base冻结，只有Writer更新。Writer不得看到action、proprio、reward、terminal、task ID、filename或隐藏stats。
- Action-Forecast v4的75-step内部门已通过，正式fresh轨迹已训练至step2400
  并按owner要求停止。固定400 panel的现有checkpoint observed-best是step825
  `109/400`，不再追加训练或80-episode快筛。
- step825已完成correct、same-task other、cross-suite wrong、shuffled、
  reversed及fixed-anchor shuffle的内部和rollout证据。行为特异性硬门失败：
  shuffled/reversed显著或方向性优于correct；随机首帧anchor只能解释部分
  shuffle收益。外部复核后的因果诊断进一步定位到未经识别的absolute-time
  forecast alignment及其Revision direction；不以contrast/order loss挽救，
  也不进入RL。
- v4实现、参数预算、32-token visual-state、Plan/Revision、Temporal、
  LoRA decoder、特异性门和训练合同完整记录在
  [`docs/action_forecast_writer_design.md`](action_forecast_writer_design.md)。
  下一版活动架构决定由
  [`docs/action_forecast_writer_v5_decision.md`](action_forecast_writer_v5_decision.md)
  覆盖其中的Plan/Revision未来口径；v4结果只作provenance。

### RL-Writer

- 与完整AS-Writer best分开，从新架构规定初态做短、task-balanced AS cold
  start；持续用官方random-reset reward screen，直到24个train tasks每个至少
  一次真实success，再关闭action数据入口并跨source tasks做纯reward训练。
- rollout初态来自官方随机reset/BDDL机制，不来自fixed `.pruned_init`。
- cold-start必须报告teacher-action queries、每task first-success step和wall；
  不能偷换成完整AS-Writer continuation。
- 使用官方reward/success，不额外读取object pose构造privileged shaping。

### Source-SFT

- 从同一frozen source base开始，在24 train tasks上联合训练一套shared multi-task LoRA。
- test不看held video/action；它控制“只加强source policy”与“额外读取held video”的差异。
- 它与AS-Writer各自在validation上选最佳，不要求相同步数或数据量；必须报告各自steps、action chunks、参数量、GPU-hours和搜索上限。

所有方法共享同一frozen source base、normalization和policy接口，但不机械要求
相同LoRA rank。Writer生成sealed rank-16 task LoRA；capacity-matched
Source-SFT可使用rank128，并以其`10,297,344`个trainable参数约束Writer本体
参数预算。各自targets/rank/alpha/dropout、identity initialization和参数量均
需显式报告；不能把LoRA两因子同时全零造成无梯度。

## 5. Seen-task 与 wrong-video 机制证据

- 在看outcome前，按specification从source tasks预声明覆盖四suites的seen panel。
- 比较frozen source base、Source-SFT、AS-Writer，以及可用的RL-Writer。
- Writer同时跑correct-video与cross-suite wrong-video。wrong-video实验保持正确language、evaluation task、init state和policy RNG，只替换为另一suite的视频。
- 主要报告 `correct-video - source base`、`wrong-video - source base` 和 `correct-video - wrong-video`，防止把通用或language-only adapter误写成视频利用。
- held zero-interaction每个rollout随机抽一条正确task teacher video，不挑最好视频。

## 6. Final retraining and test

development在24/8上选定AS-Writer、RL-Writer（若成立）和Source-SFT配置后，把validation合入形成32 source。三个方法从规定初态各自重训一次；先做final seen comparison，再打开test。

zero-interaction test比较：

- 新frozen source base；
- Source-SFT shared LoRA；
- AS-Writer one-video task LoRA；
- RL-Writer one-video task LoRA（若成立）；
- correct-video与cross-suite wrong-video Writer controls。

旧generic base已打开过并为0/400，owner明确不要求继续争论test purity；但新source base是不同模型，必须另行测量。

## 7. Test-only three-arm task-local RL

task-local RL不在validation上预先冻结算法，也不在test打开前运行。test阶段开始后，每个test task本身就是adaptation training domain，可直接根据官方随机-reset reward rollouts调优并训练到曲线接近最佳。

三臂：

1. frozen source base + functionally identity LoRA；
2. frozen source base + AS-Writer LoRA；
3. frozen source base + RL-Writer LoRA。

每个task/adaptation seed随机选一条该task视频；两个Writer臂用同一条，并固定由它生成的初始化LoRA。三臂匹配task、env/policy seed schedule、随机初态序列、RL实现与可比资源上限，保存完整interaction cursor和exact-resume状态。adaptation、调参和checkpoint选择使用该test task随机reset rollouts；固定50 states只做训练分离的fresh evaluation。

## 8. Joint target-action oracle

无action主结果和三臂RL封存后，从同一source base出发，使用8个test tasks × 每task全部50条teacher action episodes，联合训练一套shared multi-task LoRA。它不是8套task-local LoRA。第一轮只做完整50/task，不做action-budget curve；它是privileged oracle，不能反向修改其他方法。

## 9. Evaluation efficiency

- 保持官方π0.5/LIBERO preprocessing：render256/model224、两相机旋转180°、7维state/action、10 flow steps、执行5步后replan、dummy settling10、成功终止、horizon 220/280/300/520。
- 旧“一task/一GPU”使两个horizon-520进程严重拖尾。下一实现先调研成熟项目，再按 `episodes × horizon` cost-balanced切state shards，使用动态队列、持久model/env和work stealing。
- Writer每rollout LoRA不同，profile batched functional LoRA与每卡统一1/2/3个policy replicas。所有卡CUDA process数相同，GPU0不得额外放controller/server/model。
- batch8到16只有约0.9% per-episode提升；不靠盲目加batch伪装效率。
- 训练最多8张A100，一卡一DDP rank为默认，真实batch尽量利用显存且平均预留约10GB。task-local RL按每个初始化方法在全部8个test tasks上的合计训练wall-clock执行约2小时上限，而不是每task各给一份上限；三臂分别报告合计time/interactions与budget-censored状态。
- 上述短周期、证据驱动原则适用于所有训练阶段；约2小时仅为预算guardrail。到上限仍明显未训练充分时不自动延长，保存证据供owner事后分析判断。

## 10. Optional work and hard boundaries

- 核心闭环完成后有时间再做同base/split/one-video信息墙的ViVLA-style matched reproduction。
- source-only reward/meta outer learning只作更晚的可选增强，不阻塞Goal complete。
- 不使用bank、geometry、shared update subspace、residual escape、额外shared adapter、旧SmolVLA checkpoint/runner或MemLLM。
- meaningful状态更新 `task_plan.md`、`findings.md`、`progress.md`，验证、commit、push；等待长任务时继续推进不污染运行的后续工作。

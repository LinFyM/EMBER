# Prompt for the Next EMBER Session

你现在在 BCI A100 主机上接手独立研究项目 EMBER，工作区是：

```text
/data/ymdai/projects/EMBER
```

远程仓库：

```text
git@github.com:LinFyM/EMBER.git
https://github.com/LinFyM/EMBER
```

## 先建立完整长期 Goal

先调用 `get_goal`。若没有 active Goal，调用 `create_goal`，不要设置
`token_budget`，objective 原文使用：

> 在最多使用 8 张 NVIDIA A100 80GB、每卡训练平均预留约 10GB、统一 π0.5-LoRA 空间、exact-resume、one-video 信息墙和高效多卡调度约束下，完成 EMBER 从共享 π0.5-LIBERO source base 到完整单 seed 核心实证：以 specification-only 方式审计并排除 LIBERO-90 与目标 LIBERO-40 exact semantic/composition overlap，使用剩余 source tasks 每 task 全部 50 条成功 action episodes 从 generic lerobot/pi05_base 联合训练、merge 并冻结共享 source base，快速 screen 全部 40 个目标 tasks 确认部分真实成功；在封存的 24 train / 8 validation / 8 test split 上完成 Action-Supervised Writer、从零 action warm-up 优先的 Reward-Trained Writer 和 shared Source-SFT LoRA 开发，完成 seen-task 与 correct-video/cross-suite-wrong-video 对照，并根据 validation 快速早停选择；将 validation 合入形成 32 source 后从规定初态重训并完成 seen 与 zero-interaction test；test 阶段直接在每个 test task 上将 identity-init、AS-Writer-init、RL-Writer-init 三臂 task-local LoRA RL 训练到各自接近最佳，使用官方随机 BDDL 初态并与 fixed-50 fresh evaluation 隔离；最后使用 8 个 test tasks 每 task 全部 50 条 action episodes 联合训练一套 shared target-action LoRA oracle 并统一报告逐任务 rows、learning curves、seeds、interaction/data counts、runtime 与 hashes。第一轮只跑一个 training seed；RL-Writer 在零 warm-up 和极少 AS warm-up 后仍无信号可带完整证据关闭；ViVLA-style matched reproduction 和 source-only outer learning 仅在核心闭环后有时间再做，不阻塞 Goal complete。全过程不使用 pi05_libero、bank、geometry、shared subspace、residual escape、额外 shared adapter、旧 SmolVLA 活动路径或 MemLLM，任何 source base、smoke、loss 或局部结果都不能单独触发 Goal complete。

创建后再次调用 `get_goal`，核验 objective 完整且没有 token budget。不要把已经完成的
generic `pi05_base` 0/400 feasibility audit 当成新 Goal。

## 必读与 authority

先执行：

```bash
cd /data/ymdai/projects/EMBER
git pull --ff-only origin main
git status --short --branch
```

然后完整阅读根目录 `AGENTS.md`，并严格按其中顺序完整阅读：

1. `README.md`
2. `docs/execution_brief.md`
3. `task_plan.md`
4. `findings.md`
5. `progress.md`
6. `docs/concept.md`
7. `docs/decisions_and_open_questions.md`
8. `docs/novelty_and_landscape.md`

这些文件已经按 owner 的最新决定更新，是活动 authority。旧 70/10/10、SmolVLA、
Writer cold start、validation 预冻结 test-RL、task-local direct oracle、静态一 task 一 GPU
等内容只可作为历史 provenance，不能恢复成活动路径。不要使用或混入 MemLLM。

## 当前已知事实

- 固定目标 split 是四个 LIBERO suites 各 `6 train / 2 validation / 2 test`，即开发期
  `24/8/8`；确认配置后把 8 个 validation tasks 合入，最终为 `32 source / 8 test`。
- generic `lerobot/pi05_base` 已按官方口径在 8 个 sealed test tasks、每 task 50 个固定
  init states 上完成评测，结果是 `0/400`。这是校准结果，不再重复，也不因此立刻改用
  `pi05_libero`；`pi05_libero` 因见过目标四 suites 的 actions 而禁用。
- 现有 source corpus 候选是 LIBERO-90，但必须先做 specification-only exact
  semantic/composition overlap audit。已经确认至少两处语义重合：LIBERO-90 task 44 与
  Goal test task 7 都是 `turn on the stove`；LIBERO-90 task 77 与 Long train task 5 都是
  `pick up the book and place it in the back compartment of the caddy`。不得机械假定剩余数目
  就是 88；先完成全量审计并封存规则、manifest 和 hashes。
- 当前没有正式 π0.5-LIBERO source base、AS-Writer、RL-Writer 或可续用的新协议
  checkpoint。旧 SmolVLA/70-task checkpoint 与新协议不兼容。
- 之前静态 task/GPU evaluator 在最后两个 Long tasks 上出现明显拖尾；未来评估必须改为
  cost-balanced state shards、动态队列/工作窃取、持久化 model/env，而不是一 task 固定一卡。
  GPU0 不得额外堆 controller/model 进程，每张卡 CUDA 进程拓扑一致。

## 活动实验合同

### 1. 共享 source base

从 generic `lerobot/pi05_base` 出发，在经过 overlap 过滤的 LIBERO-90 source tasks 上，
每 task 使用全部 50 条成功 action episodes，按成熟/官方 π0.5 action-SFT/LoRA 配方进行
联合多任务训练。若用 LoRA 训练 source base，完成后必须 merge 成一个普通共享 policy，
避免下游再堆一层 shared adapter。只从 source 数据计算并冻结 action/state normalization；
不得使用 target-specific 或 test normalization。

随后快速 screen 全部 40 个目标 train/validation/test tasks，确认这个地基已经能在目标
benchmark 上跨多个 task 产生部分真实成功、而不是仍然完全不会控制或只靠单个易 task
支撑 aggregate。这里是低门槛能力筛查，不要求每个 task 已有高成功率。source base 冻结后，所有下游方法从同一权重、
normalization 和相同 LoRA target names/rank/alpha/dropout/capacity 出发。

### 2. Action-Supervised Writer（AS-Writer）

输入固定为 `correct task language + exactly one action-hidden teacher video`，输出完整
task-specific LoRA。开发期在 24 train tasks 上联合训练；同一 task 内，Writer 可见的
video episode 与 functional action loss 使用的 episode/chunk 独立随机抽取，不做成对模仿。
actions 只进入 functional loss，绝不进入 Writer。总训练不超过约 2 小时：先 profile
throughput 与 loss，按 loss 斜率安排稀疏而便宜的 validation screens，只对少数候选 checkpoint
做完整 validation，并在接近饱和时早停。所有 checkpoint 必须 exact-resume。

### 3. Reward-Trained Writer（RL-Writer）

这是与 AS-Writer 独立的一条路线，从随机初始化 Writer 开始，优先完全不使用 action
warm-up，直接在 source tasks 上用联合 reward/RL 更新共享 Writer。不要从训练完整的
AS-Writer 继续。RL rollouts 使用 LIBERO 官方 BDDL/reset 随机初态与官方 reward/success；
不使用自定义 privileged object-pose shaping。若完全没有 reward signal，可加入极少量
AS warm-up；若仍无信号，保存完整失败证据并暂时关闭这条路线，不让它阻塞核心闭环。

### 4. Source-SFT baseline、seen 与错误视频

Source-SFT 是在 24/32 个目标 source tasks 上联合 action-SFT 的**一套 shared multi-task
LoRA**，不是 source base，也不是每 task 一套 LoRA。它独立用 validation 选到近饱和；
不再要求和 AS-Writer 严格匹配 optimizer steps 或数据量，但必须报告实际 steps、action
chunks、GPU-hours、参数量和搜索规模。

必须预先按 specification 选一个覆盖四 suites 的 seen/source panel，比较 source base、
Source-SFT、AS-Writer 以及可用的 RL-Writer。还必须做错误视频对照：保持 task language、
evaluation task、init state 和 policy RNG 不变，给 Writer 一条来自**不同 suite**的 teacher
video，比较 source base、correct-video LoRA 与 cross-suite wrong-video LoRA。不要为了造
hard negative 改成同 suite。

### 5. 最终合并、test 与 RL

开发期根据 8 个 validation tasks 选择模型与普通训练超参数后，把 validation 合入，形成
32 source tasks；从规定初态重训 AS-Writer、可用的 RL-Writer 和 Source-SFT。第一轮完整
流程只用一个 training seed。先完成最终 seen comparison，再统一进行 zero-interaction test。
held/test rollout 每次从该 task 的 50 条 teacher videos 中随机抽一条；不得挑最好视频。

task-local RL **只在 test 打开以后、只在 8 个 test tasks 上做**，不需要先在 validation
冻结算法。对每个 test task，把该 task 当作 adaptation training domain，允许根据官方随机
BDDL reward rollouts 调参、训练、选 adaptation checkpoint，直到各臂接近各自最佳/平台。
三臂为：

1. source base + identity-init LoRA；
2. source base + AS-Writer-init LoRA；
3. source base + RL-Writer-init LoRA（仅在 RL-Writer 可用时）。

每个 task/adaptation seed 为 Writer 两臂随机选一条目标视频，并在本次 adaptation 内固定。
三臂匹配 task、env/policy seeds、官方随机 BDDL 初态序列、RL 实现和可比资源；保存 worker
RNG/seed schedule、optimizer、interaction cursor 和完整 exact-resume 状态。固定 50 个
`.pruned_init` states 只能做与 RL 数据隔离的 fresh evaluation，沿用官方 dummy settling、
成功即终止及 suite 官方 horizon。

### 6. 最后的 action-supervised oracle

所有 action-free 与 RL 结果封存后，才读取 8 个 test tasks 的 teacher actions。从同一 source
base 出发，用 `8 tasks × 50 episodes` 联合训练**一套 shared multi-task target-action LoRA**；
它不是 task-local LoRA。第一轮只做全部 50 episodes/task，不做 1/5/10 demonstration 曲线。

ViVLA-style matched reproduction 和 source-only outer learning 都只是在上述核心闭环完成后
有时间再做的可选项，不阻塞 Goal complete。

## 计算、信息墙与记录要求

- 最多使用 8 张合法空闲 A100 80GB；启动前只读检查 GPU owner/process、driver/CUDA、
  Python/PyTorch、磁盘和现有 cache。不得干扰无关进程。
- substantial download/run 前核算 `/data/ymdai` 当前占用和峰值增量；个人硬上限 500GB。
- 训练默认一卡一 DDP rank，使用真实 batch/data parallelism，每卡平均预留约 10GB；GPU0
  不得多一个 controller/model 进程。
- 评估优先有效 rollout/秒：先研究成熟 π0.5/LIBERO 项目的 evaluator，再使用 cost-balanced
  state shards、动态调度、持久化 model/env；Writer 逐 rollout 变化 LoRA 时，实测 batched
  functional LoRA 与每卡统一 1/2/3 个 policy replicas 的真实吞吐后再选择。
- Writer 永远只看 language、恰好一条 RGB teaching video 和允许的公开视觉输入；不得接收
  actions、proprio、reward、terminal、task ID、文件名或隐藏 normalization。
- 禁止 bank、geometry、shared update subspace、residual escape、额外 shared trainable
  adapter、MemLLM、旧 SmolVLA 活动 runner 或从 Git 历史恢复已退役协议。
- 每个 meaningful 阶段保存 command、config、代码/data/model revision、manifest/hash、raw
  episode rows、逐 task successes、steps/interactions、seeds、GPU/process topology、wall-clock、
  checkpoint 与 sampler/RNG cursor；更新 `task_plan.md`、`findings.md`、`progress.md`，验证后
  commit 并 push。
- smoke 只验证 mechanics，不用小分母性能做科学判断。source base、环境、cache、代码、loss
  下降或任一局部正结果都不能单独完成 Goal。

## 现在直接开始

完成 Goal 核验、authority 阅读和只读 live checks 后，不要停在复述，也不要等待形式确认，
直接推进 `task_plan.md` 的 Phase A：

1. 查清并记录成熟/官方 π0.5 action-SFT/LoRA 的确切模型加载、processor、action head、
   optimizer、scheduler、precision 与 LoRA target 方式；脚本参数不要猜。
2. 对 LIBERO-90 与目标 40 tasks 做完整 specification-only semantic/composition overlap audit，
   只使用 language、BDDL/scene/object/role/composition specification，不读取 policy outcomes；
   封存规则、task IDs、algorithm/seed、manifest 和 hashes。
3. 在不改变科学合同的前提下修正 evaluator 的拖尾与多卡利用率，并做最小真实 smoke。
4. 做 live GPU/storage preflight，形成可恢复 launch contract，随后训练并验证共享
   π0.5-LIBERO source base，继续按 Phase B 以后顺序推进。

遇到具体阻塞时，先定位并尝试最小修复；只有确实无法继续时才报告一个可复现失败。效率与
精度需要取舍且不影响科学结论时，以持续推进和反馈速度优先。

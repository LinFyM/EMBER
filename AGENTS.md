# EMBER Repository Instructions

## Authority

本文件和 `docs/execution_brief.md` 是当前活动 authority。owner 在 2026-07-21 明确将旧 SmolVLA + LIBERO-90 70/10/10 主线替换为 π0.5 + 四个标准 LIBERO suites 的单视频协议。旧配置、旧 checkpoint、旧 Phase A–F 结果和 `docs/expert_plan.md` 只作 provenance，不得自动续跑。

修改代码、split、数据或实验前完整阅读：`README.md`、`docs/execution_brief.md`、`task_plan.md`、`findings.md`、`progress.md`、`docs/concept.md`、`docs/decisions_and_open_questions.md`、`docs/novelty_and_landscape.md`。

## Current terminal task

当前唯一执行任务是：

1. 修正并封存最新协议；
2. 从通用预训练 π0.5 出发，不做任何权重训练，在预封存的 8 个 test tasks 上完成 zero-shot policy evaluation；
3. 保存逐任务 50-rollout 原始成功数、seeds、配置、runtime 和 hashes；
4. 结果产生后立即停止。

不得在本轮结果后继续 source-base action-SFT、Writer、LoRA、RL、baseline 或其他实验。是否需要 source base 由 owner 在看到结果后另行决定。

## Active scientific protocol

- Backbone：通用预训练 π0.5；不默认建立 action-SFT source base。
- Benchmark：`libero_spatial`、`libero_object`、`libero_goal`、`libero_10`（文档中可称 LIBERO-Long）。
- Development split：每 suite 6 train / 2 validation / 2 test，总计 24/8/8；按 specification-only deterministic hash 封存，不用任何 policy outcome。
- Final retraining：未来只有在 validation 完成方法选择后，才把 8 个 validation tasks 合入 source，形成每 suite 8 source / 2 test、总计 32 source / 8 test；当前不得启动。
- EMBER 输入：训练和测试均恰好一条 action-hidden teacher video + task language。
- Source functional training：同一 task 内独立随机抽一条 teacher video和一条 action-supervised agent episode/chunk，不要求 episode 配对；action 只进 functional loss，不进 Writer。
- Zero-interaction evaluation：每个 rollout 从该 task 的 teacher videos 中独立随机抽一条，Writer 生成完整 task-specific LoRA 后执行。
- Writer 可以直接产生很强的 task LoRA；RL 不是必选尾巴。若做 shared Writer RL，可跨 source tasks 联合更新 Writer。
- `Action-Supervised Writer (AS-Writer)` 是原“Writer cold start”的正式名称；它只在 source tasks 通过 functional action loss 监督训练。
- `Reward-Trained Writer (RL-Writer)` 是独立路线：从随机 Writer 初始化，或只做预声明的极短 AS warm-up 后直接用 source reward 联合训练；默认不从已完成的 AS-Writer 继续，以检验没有 teacher actions 能否训练 Writer。
- Future task-local RL：一条 adaptation run 开始时抽一条 teacher video并固定其 Writer LoRA；identity/zero-init 与 Writer-init 两臂匹配 task、env seeds、随机初态和 budget。
- Optional source-only outer learning 只能放在 Phase F 之后，且不阻塞核心结果。

## Information wall

- split 只读 suite task language、BDDL filename/specification 和 task identity；不得用 action、reward、proprio、terminal、normalization 或 policy result 选 task。
- 当前 π0.5 zero-shot 测试不读取 validation/test teacher actions，也不使用在 40 个 LIBERO tasks 上 action-finetuned 的 `pi05_libero` checkpoint。
- 通用 base 没有可直接执行 LIBERO action space 的 norm stats；若推理接口必须 normalization，只能从 24 个 development-train tasks 的 action/state 数据计算并封存。这是接口校准，不更新模型权重。
- 旧 70/10/10 normalization、SmolVLA source base、Writer 和 RL checkpoint 与新协议不兼容。

## Evaluation contract

当前 π0.5 feasibility test 服从 Physical Intelligence 官方 LIBERO inference recipe：

- `pi05_base` 模型结构与权重；LeRobot official conversion 保持 `chunk_size=50`、`n_action_steps=10`，evaluator 每次只执行前 5 actions 后重规划；
- render 256×256，模型 resize-with-pad 到 224×224；agentview 与 wrist image 均旋转 180°；
- state 为 EEF position + quaternion-to-axis-angle + gripper qpos；输出前 7 维 action；
- `replan_steps=5`、inference seed 7、每 task 50 个 official fixed init states；
- reset 后 10 个 dummy settling steps，成功即终止；
- suite horizon 使用官方 OpenPI runner：Spatial 220、Object 280、Goal 300、LIBERO-10 520。

未来 RL 更新与 adaptation checkpoint 选择只能用 LIBERO official reset/BDDL random initialization，禁止 fixed `.pruned_init`；matched arms 必须保存可恢复 RNG/seed schedule 与 interaction cursor。固定 50 states 只作与 RL 分离的 fresh evaluation。

## Model and method constraints

- 核心仍是 `language + one action-hidden teaching video -> shared Writer -> complete task-specific LoRA`。
- Writer 是共享 hypernetwork；默认每次输出 task-specific LoRA。若未来实证支持一套 LoRA 覆盖多 task，可以研究，但不得无证据写成已定合同。
- 只用 LoRA；不得引入 bank、geometry、shared update subspace、residual escape 或额外 shared trainable adapter。
- 不依赖、不修改、不混入 MemLLM。
- action-supervised direct LoRA 仅作 privileged oracle；不能用它否定 action-hidden-video setting 的必要性，也不能混入同信息墙主对照。
- 最终增加 `Source-SFT π0.5`：在合并后的 32 source tasks 上，以和 AS-Writer 相同的 optimizer-step budget 做 action-SFT，不读取 test video/action，随后直接测 8 test tasks；它用于检验 EMBER 多看到 held teacher video 是否优于单纯加强 source-policy SFT。

## Compute and execution

- 最多 8 张 A100；启动前实时检查 telemetry、进程 owner 和存储预算，不干扰无关进程。
- 每张使用中的 GPU 恰好一个同角色 policy CUDA process；GPU0 不得额外放 server、controller 或模型进程。
- 评测优先有效 rollout/s；每个 test task 分配一张卡正好 8 卡并行，持久化模型/env，避免无意义渲染。
- 精度问题不影响科学结论时效率优先。等待下载、加载或 rollout 时可推进互不污染的文档、hash、结果聚合与离线检查。
- substantial download 前确保 `/data/ymdai` 预计峰值低于 500GB；不复制已有 cache。
- meaningful state 后更新 `task_plan.md`、`findings.md`、`progress.md`，验证、commit、push。

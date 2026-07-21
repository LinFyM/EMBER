# EMBER Current Execution Brief

状态：2026-07-21 owner 协议。它替换此前 SmolVLA + LIBERO-90 70/10/10 活动线。

## 1. 研究问题

机器人 action trajectories 稀缺时，互联网上或其他数据源中的 action-hidden teaching videos 是否仍能提供足够的任务知识，使共享 Writer 生成一个比无信息初始化更好的 LoRA？直接 action-SFT 是 privileged upper bound，不是 EMBER 必须击败的同信息墙方法。

乒乓球类比是活动叙事：教练拉手对应 target action-SFT；看教学视频后第一次尝试对应 Writer zero-interaction LoRA；实践和环境反馈对应可选 reward adaptation。第一次尝试已经强时，不必为了流程完整而强加 RL。

## 2. Benchmark 与 split

只使用四个标准 LIBERO suites：`libero_spatial`、`libero_object`、`libero_goal`、`libero_10`。每 suite 10 tasks。

开发期永久封存：

- 6 train/source tasks；
- 2 validation tasks；
- 2 test tasks。

总计 24/8/8。精确 task IDs、language、BDDL/init-state identity、hash seed、排序算法和文件 hashes 位于 `configs/libero_24_8_8_v1/`。算法只对 `seed + suite + task name + language + BDDL filename` 做 SHA256 排序：前 6 train、中间 2 validation、后 2 test；没有读取任何 action、reward、normalization 或 policy outcome。

validation 完成架构、checkpoint 和预算选择后，未来最终训练才把 8 个 validation tasks 合入 source，形成 32 source / 8 test。当前 feasibility evaluation 之后必须停住，不执行该步骤。

## 3. π0.5 feasibility gate

先直接测通用预训练 π0.5 是否具备 LIBERO Panda 的基本 zero-shot control：

- 权重：generic `pi05_base`，不得使用 `pi05_libero` 或其他读过 40 个目标 tasks actions 的 fine-tuned checkpoint；
- 不训练任何模型参数；
- 不读取 validation/test teacher video、action、reward outcome；
- 若动作接口必须 normalization，只从 24 development-train tasks 的 state/action frames 计算并封存；
- 在 8 个 test tasks 各跑 50 个 official fixed init states，共 400 episodes；
- 报告每 task raw successes/50 和 aggregate，不用结果反选 task。

该结果只回答“generic π0.5 是否已经能控制 LIBERO”。结果产生后立即停止。是否建立 source base 由 owner 讨论后另开工作。

## 4. 官方评测参数

参数来自 Physical Intelligence `openpi/examples/libero/main.py` 与 `pi05_libero` config：

- render resolution 256；resize-with-pad 224；
- agentview、wrist view 均 180° rotate；第三视角补零；
- state：EEF xyz + quaternion axis-angle + two gripper qpos；
- LeRobot official π0.5 conversion 保持 model chunk 50、`n_action_steps=10`；每次规划只执行前 5 actions；10 flow inference steps；
- seed 7；每 task 50 fixed init states；
- reset 后 10 dummy actions `[0,0,0,0,0,0,-1]`；
- done/success 即终止；
- Spatial/Object/Goal/LIBERO-10 horizons 分别 220/280/300/520。

8 个 task 正好各分配一张 GPU。每卡一个等价 policy CUDA process，禁止 GPU0 额外 server/model。

## 5. 后续 EMBER 设计（仅记录，不在本轮运行）

### Action-Supervised Writer (AS-Writer)

- 输入：task language + 恰好一条 action-hidden teacher video；
- 输出：完整 task-specific LoRA；
- development source step：先从 task 均匀混合的 24 source tasks 采 task，再独立随机采 video episode 与 agent action episode/chunk；两者只要求同 task，不要求同 episode；最终冻结方案后在合并的 32 source tasks 上从规定初态重训；
- video 不含 action/proprio/reward/terminal/task ID/filename；actions 只进入 frozen π0.5 + functional LoRA 的 behavior loss；
- held evaluation：每个 rollout 从 50 条 teacher videos 随机采一条，Writer 生成 LoRA 后执行。

### Reward learning

`Reward-Trained Writer (RL-Writer)` 是与 AS-Writer 分开的路线：从随机初始化 Writer 开始，或只允许预声明的极短 AS warm-up，然后直接跨 source tasks 用 reward 联合更新 Writer。默认禁止把完整 AS-Writer 当作 RL-Writer 起点；核心问题是没有 teacher actions 时能否仅靠 source reward 训练出 video-to-LoRA Writer。生成 LoRA 是否原位更新取决于明确实验，不把 Writer 限定成单一算法。

matched task-local RL 作为第二实验：每个 adaptation run 抽一条 teacher video并固定 Writer initialization；identity/zero-init 与 Writer-init 使用相同 task、env seeds、BDDL random reset sequence、interaction/update budget 和 checkpoint rule。RL 与 checkpoint selection 不得使用 fixed `.pruned_init` states；保存 worker RNG/seed schedule 和 interaction cursor。fixed 50 states 只用于 fresh evaluation。

optional source-only outer learning 必须放在 Phase F 之后。

## 6. Baselines 与 claim

主同信息墙比较：generic/frozen π0.5、AS-Writer one-video LoRA、RL-Writer one-video LoRA、ViVLA-style one-video direct conditioner，以及未来匹配的 zero-init vs Writer-init reward adaptation。目标 action 可见的 direct LoRA SFT 只作 privileged oracle。

最终另设 `Source-SFT π0.5`：在合并后的 32 source tasks 上，从同一 generic π0.5 出发，以和 AS-Writer 相同的 optimizer-step budget 做 action-SFT；test 时不读取 teacher video 或 action，直接测同一 8 tasks。它控制“同样在 32 个 source tasks 上训练，但只有 EMBER 额外读取 held video”这一差异。

EMBER 的核心优势不是“必须做 RL”，而是把 action-hidden video 编译成可复用参数。若生成的 LoRA zero-interaction 已强，这是更好的结果；若其后 reward learning 更快，再增加 adaptation-efficiency claim。

## 7. 明确禁止

- 恢复或依赖旧 SmolVLA/70-10-10 checkpoint；
- 使用 `pi05_libero` 作为 generic base feasibility result；
- 用 test actions 计算 normalization；
- 本轮结果后自动继续 source base、Writer 或 RL；
- bank、geometry、shared update subspace、residual escape、额外 shared adapter；
- CUDA0 额外堆积进程或干扰其他用户作业。

# Decisions and Open Questions

## 已拍板

- Backbone：generic pretrained π0.5；当前不先训练 source base。
- Benchmark：LIBERO-Spatial/Object/Goal/Long (`libero_10`)。
- Development split：每 suite 7 train / 1 validation / 2 test，总计 28/4/8。
- Final source set：未来 validation 选完方案后才合并成 32 source / 8 test。
- EMBER train/test 都只输入一条 action-hidden teacher video。
- Source 训练同 task 内 independently sample video 与 action episode/chunk，不要求 paired episode。
- held zero-interaction 每 rollout 随机抽一条 teacher video。
- Writer 直接生成强 LoRA 时不强制 RL；Writer RL 可以跨 source tasks 联合进行。
- matched task-local RL 保留为后续：每 run 一条固定 video initialization，identity/Writer 两臂严格 matched。
- optional outer learning 放在 Phase F 之后。
- 只用 LoRA，不用 bank、geometry、shared update subspace、residual escape 或额外 shared adapter。
- 当前先测试 generic π0.5 在 8 test tasks 的 raw zero-shot performance，结果出来立即停止。
- 当前 test 任务可以按 owner 指令直接评；不再以“reporting-only test”异议阻塞。

## π0.5 feasibility 的精确口径

- 不用 `pi05_libero` fine-tuned weights。
- 不训练 generic base 权重。
- validation/test teacher actions 不读取。
- 必要 action/state normalization 只来自 28 train tasks。
- official OpenPI preprocessing/replan/horizon/seed/fixed init states；每 task 50 episodes。
- 8 tasks 对应 8 张 GPU；每卡相同 CUDA process topology。

## 后续尚未拍板

这些问题必须等当前 8-task result 后和 owner 讨论，当前不得自行执行：

- generic π0.5 接近零时，是否训练一个 28-task source base；
- 若训练 source base，更新 full model、action expert 还是 LoRA；
- π0.5 LoRA 的 targets/rank/alpha/dropout；
- Writer architecture 如何适配 π0.5；
- one-video Writer 训练 steps、feature cache 与 RL algorithm；
- ViVLA matched baseline 的实现细节；
- 最终是否在 validation 后合并成 32 source 重训。

## 需要 owner 决策的边界

- 当前 π0.5 result 后任何新训练；
- 扩大 benchmark、换 backbone 或改变 one-video 核心；
- 使用 test actions、fine-tuned target checkpoint 或新 shared trainable module；
- 不可逆数据清理或跨越 500GB storage cap。

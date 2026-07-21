# Prompt for the Next EMBER Session

你在 `/data/ymdai/projects/EMBER` 接手 EMBER。先调用 `get_goal` 并完整阅读 `AGENTS.md` 及其指定 authority。不要依赖 MemLLM，不要恢复旧 SmolVLA/70-10-10 runner 或 checkpoint。

当前协议：generic π0.5；LIBERO-Spatial/Object/Goal/Long 四 suites；每 suite 6 train / 2 validation / 2 test；EMBER train/test 都只看一条 action-hidden teacher video；source training 同 task 内独立随机采 teacher video 与 action episode/chunk；不默认 source base。当前唯一任务是按官方参数完成 generic `pi05_base` 在 8 个预封存 test tasks 上的 zero-shot 50-state evaluation，结果出来立即停止。

后续但本轮禁止执行：原 Writer cold start 改名 `Action-Supervised Writer (AS-Writer)`；另设从随机初始化或仅极短 AS warm-up 开始的 `Reward-Trained Writer (RL-Writer)`，检验无 teacher action 的 source-reward 训练。validation 后合并成 32 source tasks 重训，并增加与 AS-Writer 相同 optimizer-step budget、test 不看视频的 `Source-SFT π0.5` baseline。

禁止使用 `pi05_libero` fine-tuned weights，禁止读取 validation/test teacher actions。generic base 所需 action/state normalization 只能从 24 development-train tasks 计算，不更新模型权重。使用 Physical Intelligence 官方 LIBERO preprocessing、replan 5、seed 7、50 fixed states、10 dummy settling 与 suite horizons 220/280/300/520。

启动前 live 检查 GPU owner、进程、driver/CUDA、Python/PyTorch 和 storage；8 个 task 各用一张 GPU，每卡一个同角色 policy CUDA process，GPU0 不得额外堆进程。保存 raw rows、per-task success、command/config/revisions/hashes/runtime。结果写回 `task_plan.md`、`findings.md`、`progress.md`，验证、commit、push，然后停止，不继续 source base、Writer、RL 或 baseline。

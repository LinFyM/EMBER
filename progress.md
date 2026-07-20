# EMBER Progress and Handoff

最后更新：2026-07-20。

## 工作区状态

- 当前活动科研合同已统一到同分布 70/10/10 + SmolVLA source embodiment base + full-video task-specific LoRA Writer。
- 旧 60/15/15 config、Gate recovery runner、自定义 Gate0 RL、旧 Writer runner 和对应 tests 已从活动树删除；完整版本可由父提交 `999df28` 追溯。
- 保留的代码仅是通用 LIBERO 审计、runtime/gallery、可变长度 Writer model/data/topology 内核。
- 新 70/10/10 task IDs、manifest、normalization、source base config 和新 Writer config **尚未生成**。
- 旧 checkpoint 全部与新协议不兼容，不得 exact-resume。
- 当前没有活动 EMBER GPU 训练/评估进程。
- 当前没有活动 Goal；下一 session 应按 `docs/new_session_prompt.md` 创建无 token budget 的完整 Goal。
- 已删除 179 个非 canonical `.codex/longrun` wrapper 目录（约 65MB）；原始科学证据仍在外部 checksummed output，完整旧执行树在 Git 父提交。
- 已核验并删除四个干净、已 supersede 的 worktree/local branches：`d25544e`、`cc4ba36`、`1722b9d`、`dbfaa59`。它们没有未提交用户改动；相关活动实现已由当前历史取代。

## 已完成且仍有效

- Gate -1：passed with residuals。
- Gate 0：passed with limited coverage。
- pinned SmolVLA/LIBERO 环境与 HDF5/task factor 审计经验。
- 37-target rank-32/alpha-16/dropout-0 LoRA 支持的实现经验。
- full-video variable-episode Writer architecture。
- GPU/NUMA/EGL 进程放置和评估吞吐诊断。
- exact-resume 所需状态清单和运行纪律。

## 已明确退役

- 60/15/15 IDs 和 normalization。
- canonical bank、soft geometry、shared subspace、residual escape、旧 Gate 1。
- task3/task4 作为未来主开发任务。
- h16 作为新主评估。
- 三帧 Writer、固定 episode 数、0–7/8–39 人为 context/query 划分。
- last-two q/v rank-8/16 作为最终 LoRA。
- standalone Language-only Writer / Video-only Writer。
- 旧 Gate0 custom chunk-level PPO/FPO recovery tree。
- 旧 source base 与所有旧 Writer/direct checkpoint 作为新协议起点。

## 下一 session 的第一批动作

1. 调用 `get_goal`，确认没有 active Goal；创建 `docs/new_session_prompt.md` 中的完整长期 Goal，不设置 token budget。
2. 完整阅读权威顺序。
3. `git status --short --branch`、`nvidia-smi`、磁盘和现有 cache 只读检查。
4. 用 `libero_task_factors.py` 建立 90-row specification factor table。
5. 在任何新 policy outcome 前生成并封存同分布 70/10/10。
6. 写最小单一路径 source-base config/runner。
7. 8-GPU 真实吞吐 smoke 后运行约 30 分钟 source base。

不得先做：

- 恢复旧 runner；
- 评估旧 checkpoint 来选新 split；
- 开 Writer/RL；
- 重构未来框架；
- 打开 reporting-only test 来调方法。

## 交接验证入口

```bash
cd /data/ymdai/projects/EMBER
git status --short --branch
.venv/bin/python -m pytest -q
rg -n '60/15/15|Gate 1|canonical bank|soft geometry|horizon 16|three.*frame|0--7|8--39' \
  README.md AGENTS.md task_plan.md findings.md progress.md docs \
  --glob '!docs/expert_plan.md'
```

`docs/expert_plan.md` 预期会命中旧词；它是明确标注的历史原文，不是活动 authority。

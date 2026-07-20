# EMBER Progress and Handoff

最后更新：2026-07-20。

## 工作区状态

- 当前活动科研合同已统一到 Phase A–F：70/10/10 → shared source embodiment base → Writer cold start → Writer-only RL → matched task-local LoRA RL → 全部冻结后统一 reporting-only test；outer learning 只可在 Phase F 之后作为可选增强。
- 旧 60/15/15 config、Gate recovery runner、自定义 Gate0 RL、旧 Writer runner 和对应 tests 已从活动树删除；完整版本可由父提交 `999df28` 追溯。
- 保留的代码仅是通用 LIBERO 审计、runtime/gallery、可变长度 Writer model/data/topology 内核。
- 新 70/10/10 task IDs、factor table、data manifest 和 train-only normalization 已封存在 `configs/libero90_70_10_10/`；source-base config/runner 正在当前工作树实现，尚未完成验证或提交，新 Writer config 尚未生成。
- 旧 checkpoint 全部与新协议不兼容，不得 exact-resume。
- 当前没有活动 EMBER GPU 训练/评估进程。
- 完整长期 Goal 已建立且保持 active；Phase A 不能单独触发 Goal complete。
- 已删除 179 个非 canonical `.codex/longrun` wrapper 目录（约 65MB）；原始科学证据仍在外部 checksummed output，完整旧执行树在 Git 父提交。
- 已核验并删除四个干净、已 supersede 的 worktree/local branches：`d25544e`、`cc4ba36`、`1722b9d`、`dbfaa59`。它们没有未提交用户改动；相关活动实现已由当前历史取代。

## 已完成且仍有效

本文后续的 frozen source embodiment base 统一指：通用预训练 `lerobot/smolvla_base` → 在 70 个 train tasks、每任务全部 50 条成功 teacher episodes 上联合训练 → 得到共享、多任务、语言条件的 source embodiment base → 训练完成后冻结。它只能按 train/source evidence 选择，并作为 EMBER、target direct LoRA oracle 和 ordinary task-local LoRA RL 的共同起点。

- specification-only 70/10/10 seal：seed `20260720`，validation `[0,8,15,28,40,56,61,71,85,88]`，test `[4,7,11,32,41,59,60,70,84,86]`。
- 90-task HDF5/BDDL/init-state audit：4500 demos、669,043 frames；normalization 只读 70×50 train episodes。
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

## 当前下一批动作

1. 核验 pinned `lerobot/smolvla_base@c83c316...` 的成熟 trainable set、pre/postprocessor、37-target LoRA 挂载空间和数据 normalization 接口。
2. 写最小单一路径 source-base config/runner；只消费 sealed train IDs 和全部 70×50 episodes。
3. 在启动前重新做实时 GPU/storage preflight，随后 8-GPU DDP 真实吞吐 smoke。
4. 根据实测 samples/s 冻结约 30 分钟的 total steps、full-task-cycle thirds 和 exact-resume command，启动 source base。
5. source base 训练等待期间，在不读取 validation/test outcome、不并发修改同一 owner 的前提下，推进 validation evaluator/direct-LoRA oracle 与 Writer 接线的只读检查和独立代码准备。

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
cd configs/libero90_70_10_10 && sha256sum -c checksums.sha256
rg -n '60/15/15|Gate 1|canonical bank|soft geometry|horizon 16|three.*frame|0--7|8--39' \
  README.md AGENTS.md task_plan.md findings.md progress.md docs \
  --glob '!docs/expert_plan.md'
```

`docs/expert_plan.md` 预期会命中旧词；它是明确标注的历史原文，不是活动 authority。

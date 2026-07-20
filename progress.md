# EMBER Progress and Handoff

最后更新：2026-07-20。

## 工作区状态

- 当前活动科研合同已统一到 Phase A–F：70/10/10 → shared source embodiment base → Writer cold start → Writer-only RL → matched task-local LoRA RL → 全部冻结后统一 reporting-only test；outer learning 只可在 Phase F 之后作为可选增强。
- RL 数据合同已冻结：更新与 adaptation checkpoint 选择使用官方 reset/BDDL 随机初态，matched 两臂共享 task/env seed/初态序列；固定 50 个 `.pruned_init` states 只用于独立 fresh evaluation。
- 旧 60/15/15 config、Gate recovery runner、自定义 Gate0 RL、旧 Writer runner 和对应 tests 已从活动树删除；完整版本可由父提交 `999df28` 追溯。
- 保留的代码仅是通用 LIBERO 审计、runtime/gallery、可变长度 Writer model/data/topology 内核。
- 新 70/10/10 task IDs、factor table、data manifest 和 train-only normalization 已封存在 `configs/libero90_70_10_10/`；source-base 首段 seed-1 trajectory 与 thirds source-development h50 已完成，已据 source-only 上升趋势封存 step630→945 短续段。新 Writer 训练 config 尚未生成。
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
- source-base trainable contract：98,880,992 trainable / 450,046,176 total parameters；只更新 SmolVLA action expert、state/action/time projections，VLM 保持冻结。
- source-base 数据路径：537,946 个 train frames，70 tasks × 50 episodes；跨 rank task slots 使用 deterministic no-replacement cycles，episode/frame 选择是 global-step 的纯函数。
- rank-local device 修复后的 8-GPU profile：batch/rank 352、global batch 2816、2.590s/step、1087.2 samples/s、每卡峰值 allocated/reserved 65.05/67.35GiB；steady-step 进程表显示每张卡恰好一个 policy CUDA rank、69,124–69,132MiB process memory，GPU0 与其余卡一致。
- 8-rank interrupted/resumed smoke 的最终 policy SHA256 位级一致，optimizer/scheduler 与每 rank RNG 逐值一致。根因修复是固定 DDP static graph，避免恢复后的首轮 bucket 布局与连续运行不同。
- 首次 formal 启动在首个 checkpoint 前由进程表发现所有 ranks 的无索引 `"cuda"` 构造路径会在 GPU0 留额外 context；作业被主动停止且失败目录不复用。policy config、processor 和模型构造现显式使用 `cuda:{local_rank}`，随后 batch=1 load smoke 与 batch=352 steady smoke 均验证 8 卡进程数一致。
- 正式 source-base seed-1 trajectory：commit `72eb10d`、630/630 steps、退出码 0、约 28 分钟；210/420/630 三个 checkpoint 均完整。最终累计 1,774,080 examples、5,040 task slots，70 tasks × 50 episodes 全覆盖；launch contract `22c4ffb5...2e8`，最终 checkpoint manifest `89e9f493...ed22c`。
- step 630 当前只按 source loss 选为候选；正式冻结仍等待 8 个预声明 source-development tasks 的官方 h50 闭环结果，validation 不参与 checkpoint 选择，test 保持封存。
- shared LoRA 合同已实现并封存：37 targets、rank 32、alpha 16、dropout 0、1,485,312 parameters，支持 in-place injection 和 differentiable functional application；Writer/direct/RL 将共用同一挂载空间。
- source-base evaluator 已通过真实 8-rank smoke：固定 state IDs 0–7 各出现一次、每卡一个 policy CUDA process、显存完全一致、退出后全清；完整测试为 28 passed。该 smoke 的 `0/8` 不作行为结论。
- 初始 source-base thirds 使用同一 8 tasks × 50 states 得到 step210/420/630=`3/400, 8/400, 15/400`；420→630 为 11 paired gains、4 paired losses。绝对 competence 仍低，故不冻结并只追加一次 315-step exact continuation。

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

1. 从 step-630 checkpoint exact-resume 315 steps；保持 batch/rank 352、8 ranks、原 optimizer/RNG/sampler 和 floor LR，保存 735/840/945。
2. 先同口径评估 step945；若无增益保留630，含混时才补735/840。选定后冻结 base；validation 不回选 checkpoint，test 继续封存。
3. 生成 10 validation tasks 的 frozen-base reference，并训练各自 50-episode matched direct-LoRA oracle，复用同一 37-target LoRA/evaluator 合同。
4. 训练等待期间推进 frozen VLM feature cache 与 Writer functional-loss 接线；先 profile 真实完整视频路径，再只优化会影响 90 分钟反馈预算的瓶颈。

## Canonical runner ownership

- `scripts/train_source_base.py` 是 Phase B 唯一活动入口；`src/ember/source_base.py` 负责训练编排，`source_base_checkpoint.py` 只拥有 launch provenance 和 exact-resume 原子 checkpoint，现有 `writer/data.py` 提供共享的 HDF5/sampler owner。没有保留平行或版本化 runner。
- `scripts/evaluate_source_base.py` 是 Phase B frozen-base fresh evaluation 的唯一入口；`src/ember/libero_evaluation.py` 只拥有 split/RNG/state schedule 和结果聚合，test role 在 Phase F 前不存在。`src/ember/lora.py` 是 Writer/direct/RL 的共享 37-target LoRA owner。
- 这些文件在 source base 冻结后继续作为可复现入口保留，不再复制出下一版 runner；只有出现第二个当前消费者时才提炼公共抽象。profile 和 resume-smoke 大权重是可删除的临时产物，正式 checkpoints、manifest、metrics 和 hashes 才是 retained evidence。

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

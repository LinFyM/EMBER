# EMBER Progress and Handoff

最后更新：2026-07-20。

## 工作区状态

- 当前活动科研合同已统一到 Phase A–F：70/10/10 → shared source embodiment base → Writer cold start → Writer-only RL → matched task-local LoRA RL → 全部冻结后统一 reporting-only test；outer learning 只可在 Phase F 之后作为可选增强。
- RL 数据合同已冻结：更新与 adaptation checkpoint 选择使用官方 reset/BDDL 随机初态，matched 两臂共享 task/env seed/初态序列；固定 50 个 `.pruned_init` states 只用于独立 fresh evaluation。
- 旧 60/15/15 config、Gate recovery runner、自定义 Gate0 RL、旧 Writer runner 和对应 tests 已从活动树删除；完整版本可由父提交 `999df28` 追溯。
- 保留的代码仅是通用 LIBERO 审计、runtime/gallery、可变长度 Writer model/data/topology 内核。
- 新 70/10/10 task IDs、factor table、data manifest 和 train-only normalization 已封存在 `configs/libero90_70_10_10/`；source base 已依据 train/source-development evidence 冻结为 step630，选择合同在 `configs/source_base_selected_v1.json`。正式 Writer cache 与 interrupted/resume profile 已完成，formal cold-start 参数已封存。
- 旧 checkpoint 全部与新协议不兼容，不得 exact-resume。
- GPU 进程与占用必须以每次 launch 前的实时快照为准；任何阶段都保持一张卡一个 policy CUDA rank，GPU0 不放额外 controller/model 进程。
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
- source base 已正式冻结为 step630：step945 续训候选同口径仍为 `15/400`，相对630配对 `5 gains / 5 losses / net 0`，因此按预声明规则保留630且不再评735/840。选择未读取 validation/test；selected checkpoint manifest 为 `89e9f493...ed22c`，policy SHA256 为 `eb7e01f2...c1f159f`。
- 冻结后 source-base validation reference 已完成：10 tasks × 50 fixed fresh states 为 `56/500`，per-task `{0:28,8:0,15:0,28:14,40:0,56:1,61:0,71:0,85:0,88:13}`；结果 SHA256 `3d19f00f...0cac9`，test 未打开。
- shared LoRA 合同已实现并封存：37 targets、rank 32、alpha 16、dropout 0、1,485,312 parameters，支持 in-place injection 和 differentiable functional application；Writer/direct/RL 将共用同一挂载空间。
- source-base evaluator 已通过真实 8-rank smoke：固定 state IDs 0–7 各出现一次、每卡一个 policy CUDA process、显存完全一致、退出后全清。该 smoke 的 `0/8` 不作行为结论。
- 初始 source-base thirds 使用同一 8 tasks × 50 states 得到 step210/420/630=`3/400, 8/400, 15/400`；420→630 为 11 paired gains、4 paired losses。绝对 competence 仍低，故不冻结并只追加一次 315-step exact continuation。
- frozen-VLM cache 正式产物已完成：70 tasks、3500 full episodes、537,946 frames，70 个 tensor 独立 size/SHA 全通过；manifest `ae5854a6...be127`。提取与 smoke/resume 均保持 8 卡各一个 CUDA rank。
- Writer cold-start canonical path 已接通并通过真实 8-rank interrupted/resume profile：每 rank batch 384、steps 2–35 平均 3.426s、峰值 reserved 70.54GiB；step17 的 10 个 checkpoint 文件均验 SHA，恢复至 step35 后 70 tasks × 50 episodes 全覆盖。formal 已封存为 1575 steps 与 525/1050/1575 checkpoints，预计纯训练 89.93 分钟。
- Writer cold-start formal seed 1 已在 commit `69bbdee` 完成 1575/1575 steps、exit 0，约 92.9 分钟、4,838,400 global queries；70 tasks 各精确消费 69,120 queries 并覆盖全部 50 episodes。最终 manifest `c30c49af...3357`、consumed identity `2029f311...4112`，每卡仍恰好一个 CUDA rank。
- validation Writer cache 已完成 10×50 episodes / 63,544 frames；profile step35 的 8-rank fresh-eval smoke 生成完全一致的 task0 adapter SHA，并完成 8 条唯一 fixed-state rows。该 smoke 只证明 mechanics。
- cold Writer 首个完整 validation RNG 已完成：step525/1050/1575=`58/500,63/500,60/500`，按预封存规则选择 step1050。相对 base `56/500`，selected 为 `31 gains / 24 losses / net +7`，正增益在 task28/task88 两类但 task0 `-9`；故 selection 完成而机制结论仍待独立 policy RNG 确认。选择 seal 为 `configs/writer_cold_start_selected_v1.json`，RNG2 合同已在 outcome 前封存且不得重选 checkpoint。
- 独立 policy RNG2 已完成：base/selected Writer=`51/500,57/500`，配对 `30/24/+6`。两 RNG 合并为 `107/1000 → 120/1000`；task28 `+22/100`、task88 `+12/100` 且方向逐 RNG 复现，构成 KITCHEN-actuation 与 STUDY-pick-place 两个不同未见类别上的真实功能信号。覆盖仍有限：task0 `-20/100`，aggregate 仅 +1.3pp，多数任务双方为零。确认 seal 为 `configs/writer_cold_start_rng2_confirmation_v1.json`，test 未打开。
- validation direct-LoRA formal 已完成：10 tasks 各自使用全部 50 action episodes、batch384 × 180 steps = 69,120 queries，60/120/180 的 30 个 manifests/files 全部验证；约 17.5 分钟、峰值 reserved 69.09GiB、每卡一个 CUDA rank。固定 final checkpoints 的 500-row fresh evaluation 为 `186/500`，per-task `{0:48,8:1,15:17,28:36,40:21,56:11,61:9,71:2,85:11,88:30}`；相对 base `56/500` 配对净 `+130`，相对 cold Writer `63/500` 净 `+123`。seal 为 `configs/direct_lora_validation_reference_v1.json`，test 未打开。
- Writer-only RL 已从 update1 exact-resume 到 update9 并完成一个 70-task cycle：每 task 4 个官方随机 reset rollouts，总计 `280` interactions / `87` successes / `90,391` env steps，72 个 worker ledgers、280 个唯一 seed rows和两个 10-file checkpoints 全部通过 cursor/manifest 审计。max-rank wall 405.50 秒；formal 已封存 12 cycles，即 108 updates、每 task 48 rollouts、thirds 36/72/108，预计 81.1 分钟。

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

1. 继续当前已启动的 108-update Writer-only RL formal；保持一 GPU 一 rank、official random reset、source-only reward，并在 36/72/108 保存 full-cycle checkpoints。
2. 依预封存规则对 cold step1050 与 Writer-RL 36/72/108 做同口径 validation，选择最佳 Writer 后自动进入 matched task-local RL profile。
3. 保留 cold Writer 两 RNG 已确认的“两类正效用、覆盖有限、task0 回退”和 direct oracle 的“大 acquisition gap”原始结论，不增 policy RNG、不重选 cold checkpoint。
4. optional outer learning 只可在 Phase F 完成后考虑，不阻塞核心 Goal。

## Canonical runner ownership

- `scripts/train_source_base.py` 是 Phase B 唯一活动入口；`src/ember/source_base.py` 负责训练编排，`source_base_checkpoint.py` 只拥有 launch provenance 和 exact-resume 原子 checkpoint，现有 `writer/data.py` 提供共享的 HDF5/sampler owner。没有保留平行或版本化 runner。
- `scripts/evaluate_source_base.py` 是 base/Writer/direct/task-local-RL 共用的唯一 fresh-evaluation 入口；`src/ember/libero_evaluation.py` 只拥有 split/RNG/state schedule 和结果聚合，各 inference 模块只拥有各自产物 → task LoRA materialization，test role 在 Phase F 前不存在。`src/ember/lora.py` 是 Writer/direct/RL 的共享 37-target LoRA owner。
- `scripts/train_writer_cold_start.py` 是 Phase C 唯一训练入口；`src/ember/writer/training.py` 只编排 frozen source policy、feature cache、Writer DDP 和 functional loss，`writer/checkpoint.py` 独占 exact-resume checkpoint。没有第二套 Writer runner。
- `scripts/train_direct_lora.py` 是 action-supervised direct reference 的唯一训练入口；`direct_lora_protocol.py` 固定 validation-only split、69,120 matched queries/task 和 8-rank assignment，`direct_lora_checkpoint.py` 独占 task-local LoRA/optimizer/scheduler/RNG/sampler 恢复状态。真实 profile/resume 通过后 formal 已按 batch384、180 steps、60/120/180 打开。
- `scripts/train_writer_only_rl.py` 是 Phase D 唯一训练入口；只对 shared Writer 做 8-rank DDP，生成 LoRA 不原位更新，70-task no-replacement source cycles 与 official-random-reset interaction ledger 可 exact-resume。
- `scripts/train_task_local_lora_rl.py` 是 Phase E 唯一训练入口；identity/Writer 两臂共享不含 arm 的 env/policy seed schedule，只更新各目标任务自己的 LoRA。`task_local_rl_checkpoint.py` 独占 optimizer/scheduler/RNG/interaction cursor，selected adaptation checkpoint 由预算内随机-reset reward segment 决定；固定 50 states 只在 shared fresh evaluator 中使用。formal 在真实 profile 前关闭。
- 这些文件在 source base 冻结后继续作为可复现入口保留，不再复制出下一版 runner；只有出现第二个当前消费者时才提炼公共抽象。profile 和 resume-smoke 大权重是可删除的临时产物，正式 checkpoints、manifest、metrics 和 hashes 才是 retained evidence。

不得先做：

- 恢复旧 runner；
- 评估旧 checkpoint 来选新 split；
- 在 cold-start validation 证据前开 Writer-only RL 或 task-local RL；
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

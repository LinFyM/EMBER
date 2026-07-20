# Prompt for the Next EMBER Session

将下面整段复制给新的 Codex session。

---

你现在在 bciA100 上接手独立研究项目 EMBER。

工作区：

```text
~/EMBER
/data/ymdai/projects/EMBER
```

远程：

```text
git@github.com:LinFyM/EMBER.git
https://github.com/LinFyM/EMBER
```

不要修改、依赖或混入 MemLLM。不要从 Git 历史恢复已退役的 60/15/15、Gate recovery、bank/geometry 或旧 runner。

## 先建立完整 Goal

先调用 `get_goal`。若没有 active Goal，调用 `create_goal`，不要设置 `token_budget`，objective 原文使用：

“在最多使用 8 张 NVIDIA A100 80GB、每卡训练平均预留约 10GB、matched fairness、exact-resume、train/validation/test 隔离和 shared-frozen held 约束下，完成 EMBER 在 SmolVLA + LIBERO-90 上的可复现实证：以 specification-only、outcome-blind 方式封存同分布 70/10/10 task split；从通用预训练 lerobot/smolvla_base 出发，在全部 70×50 条成功 teacher episodes 上联合训练一个共享 source embodiment base，随后冻结它，作为 EMBER 和核心对照的共同起点；训练 language + 任意数量和长度的 action-hidden teaching videos → shared Writer → complete task-specific LoRA 的 cold start，并在多个不同类别 validation tasks 上评估和选择 Writer；随后只在 source tasks 上完成只更新 Writer、不原位更新生成 LoRA 的 Writer-only RL，并在 validation 上选择最佳 Writer；在 source embodiment base 和 Writer 冻结条件下，完成 matched zero/identity-init 与 Writer-init ordinary task-local LoRA RL，并在 validation 上冻结算法、交互预算和 checkpoint 选择规则；所有方法、checkpoint、预算和 baseline 冻结后，统一在 reporting-only test 上评估 frozen source embodiment base、最佳 EMBER zero-interaction、ordinary LoRA RL、Writer-init matched LoRA RL 以及必要的同信息墙强 baseline，其中共享状态始终冻结，只允许预声明的 task-local LoRA reward adaptation。Target-action-supervised direct LoRA 在 validation 和最终 test 上单独作为使用目标 teacher actions 的 oracle/reference，不属于与 EMBER 信息条件相同的主 baseline。EMBER 在多个不同类别 validation tasks 上明显优于 frozen source embodiment base，即可证明核心机制产生功能价值。最终合同冻结后保留完全符合该合同的开发训练轨迹，仅重训受合同变化影响的方法，并补齐必要独立 seeds、baselines 和统一 test。source-only reward/meta outer learning 只作为核心机制成立后的可选增强，不是 Goal 完成条件；当前 Goal 不包含 OpenVLA-OFT、不包含 source direct-LoRA localization，也不使用 bank、geometry、shared subspace 或额外 shared trainable adapter。”

创建后再次 `get_goal`，核验 objective 完整、状态 active 且没有 token budget。

## 必读

先完整阅读根目录 `AGENTS.md`，然后按其顺序阅读：

1. `README.md`
2. `docs/execution_brief.md`
3. `task_plan.md`
4. `findings.md`
5. `progress.md`
6. `docs/concept.md`
7. `docs/decisions_and_open_questions.md`
8. `docs/novelty_and_landscape.md`

`docs/expert_plan.md` 是历史原文，不是活动 authority。不要重新引入其 bank/geometry/60-15-15。

## 固定定义

本文后续的 frozen source embodiment base 统一且只指：

```text
通用预训练 lerobot/smolvla_base
    → 在 70 个 train tasks、每任务全部 50 条成功 teacher episodes 上联合训练
    → 得到一个共享、多任务、语言条件的 source embodiment base
    → 训练完成后冻结
```

它不是原始通用 checkpoint。EMBER、target-action-supervised direct LoRA oracle 和 ordinary task-local LoRA RL 都以它为共同起点；source base 只能按 train/source evidence 选择，validation/test 不得用于选择或调整它。

## 当前事实

- Phase A 已完成：specification-only 70/10/10 split、factor table、manifest、train-only normalization 和 hashes 已永久封存在 `configs/libero90_70_10_10/`；不得根据 outcome 重新搜索 IDs。
- 当前进入 Phase B；以 `git status`、`progress.md` 和活动代码为准确认 source-base runner 的最新验证状态。
- 旧 source base、旧 Writer 和全部旧 checkpoint 与新协议不兼容。
- 保留的 Writer 内核支持 full-video、variable-episode、complete task-specific LoRA；不得恢复旧 executable tree。
- reporting-only test 尚未解封。

## 阶段顺序

1. Phase A：封存 specification-only、outcome-blind 70/10/10 split（已完成）。
2. Phase B：全部 70×50 source episodes 联合训练并冻结 shared source embodiment base；只按 train/source evidence 选择；随后只在 validation 建立 frozen source embodiment base 与 target-action-supervised direct LoRA reference。
3. Phase C：Writer cold start；在多个不同类别 validation tasks 评估和选择。
4. Phase D：只在 source tasks 做 Writer-only RL；validation 比较并选择最佳 Writer。
5. Phase E：在 validation 完成 matched zero/identity-init 与 Writer-init ordinary task-local LoRA RL，冻结算法、interaction/update budget 和 checkpoint selection rule。
6. Phase F：冻结全部方法、checkpoint、预算、selection rule 和 baseline；保留完全合规 trajectory，只重训受合同变化影响的 arms，补齐必要 seeds/baselines，最后统一打开 test。
7. Phase F 完成后，才可按价值探索 optional source-only reward/meta outer learning；它不阻塞 Goal complete，也不改写核心 test。

## Test 禁区与最终口径

Phase F 解封前：

- 不运行 test policy evaluation；
- 不训练 test direct LoRA；
- 不读取 test actions、reward outcomes 或成功率；
- 不提前生成 frozen source embodiment base/direct-LoRA test reference；
- 不根据 test 修改任何方法。

最终 test 统一运行 frozen source embodiment base、validation 选出的最佳 EMBER zero-interaction、target-action-supervised direct LoRA oracle、zero/identity-init ordinary LoRA RL、best Writer-init matched LoRA RL 和已冻结必要强 baseline。test direct LoRA 可使用目标 task 的全部 50 条 teacher action episodes，但只作 oracle/reference。test task-local RL 只按 validation 已冻结的 reward budget 和 selection rule 适应，并用 fresh rollout 报告最终性能。

## 当前 Phase B 执行要求

- `cd ~/EMBER && git pull --ff-only origin main && git status --short --branch`；先保护任何未提交用户工作。
- 只读检查 GPU owner/telemetry、CUDA/driver、Python/PyTorch、磁盘和现有 cache；启动前再次实时 preflight。
- 从锁定的通用 `lerobot/smolvla_base` 开始，只消费 sealed 70 train IDs 和全部 3500 条成功 episodes。
- 一个共享 language-conditioned multi-task base；成熟 SmolVLA action expert + 必要 projections；8-GPU DDP，一卡一 rank。
- deterministic no-replacement task cycles；每个完整阶段覆盖全部声明 episodes。
- 先短测真实 batch 的吞吐/显存，再冻结约 30 分钟的 steps；checkpoint 在 full-task-cycle 对齐的 1/3、2/3、3/3。
- checkpoint 保存 model、optimizer、scheduler、scaler、sampler/data cursor、每 rank RNG、step、episode 和 consumed-data state，并验证 exact resume。
- 训练结束后冻结 source base；先做 train/source development 与 validation，不打开 test。

## LoRA、Writer 与核心对照

- Writer 输入只有 language + 任意数量/长度 action-hidden teaching videos；action 只进 source functional loss。
- Writer/direct/RL 使用相同 37 targets、rank32、alpha16、dropout0 和参数量。
- 核心对照：frozen source embodiment base、EMBER zero-interaction、validation/final-test direct LoRA oracle、zero/identity-init ordinary task-local LoRA RL、best Writer-init matched LoRA RL。
- 不运行 source direct-LoRA localization、standalone Language-only Writer 或 standalone Video-only Writer。
- 当前 Goal 不包含 OpenVLA-OFT。

## 效率与执行风格

- 真实实验优先，最小化脚手架；smoke 只看 mechanics，不读小分母性能。
- training 用真实 batch、缓存、task/data parallelism 提升 samples/s，尽量使用合法空闲的 8 张卡并保留约 10GB headroom；不用 dummy tensors。
- evaluation 以有效 rollout/秒为目标：每卡一个 policy CUDA process、持久化 env pool、task/arm/seed 动态分片，减少模型/env 重建和非必要视频渲染。
- 等待训练或评估时，推进下一阶段的只读检查、独立 owner 代码和离线准备；不得并发修改同一文件/产物，不得提前读取 validation/test outcome 或让后续选择污染当前阶段。
- meaningful 结果后更新 `task_plan.md`、`findings.md`、`progress.md`，验证、commit、push。
- 如果不能启动，只报告一个具体可复现失败和最小修复；不要停在复述，也不要等待形式确认。

完成 Goal 核验和只读状态检查后，从 `progress.md` 的当前阶段继续自主推进。

---

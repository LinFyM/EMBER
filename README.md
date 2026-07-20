# EMBER

EMBER 是一个面向机器人操作的任务条件 LoRA 生成与后续适应项目。当前核心假设只有一条：

```text
language + action-hidden teaching videos
                    ↓
                 Writer
                    ↓
       complete task-specific LoRA
```

Writer 不是通用优化器，也不生成 bank、basis、geometry、mask、metric、radius、学习率或第二层搜索空间。它生成预先固定挂载位置和 rank 下的一整套任务专属 LoRA；后续普通 task-local RL 只在原位更新这套 LoRA。

## 当前状态

- Gate -1 已按“通过但带残差”封存。原始 action-hidden-video recovery 为 ordered/wrong-video `19/24`、paired `15/24`，旧 `0.80` 阈值和 drop-last 残差没有被改写。
- Gate 0 按“通过但证据覆盖有限”处理。历史 task 3/4 的 action-supervised LoRA 相对当时 base 呈一致正向点估计，但只覆盖两个近似任务、每臂 32 次评估；它不再是 Writer 的前置门槛。
- 旧 60/15/15 split、旧 source base、旧 Writer checkpoint、旧 recovery runner 与旧 h16 主评估合同均已退役。它们只保留在 Git 父提交 `999df28` 和外部证据包中，不能作为下一轮实验的活动输入。
- 下一步从 specification-only 信息重新封存同分布 70/10/10 split，随后在全部 70×50 条成功 teacher episode 上训练一个共享 source embodiment base，再重训 Writer。
- 当前没有活动 Goal、GPU 作业或可继续沿用的正式新协议 checkpoint。新 session 必须先建立覆盖完整生命周期的 Goal。

## 权威阅读顺序

新进入者先完整阅读：

1. [AGENTS.md](AGENTS.md)
2. [docs/execution_brief.md](docs/execution_brief.md)——唯一活动科研与执行合同
3. [task_plan.md](task_plan.md)——分阶段推进顺序与完成条件
4. [findings.md](findings.md)——已经成立的证据、负结果与局限
5. [progress.md](progress.md)——当前工作区状态和下一条动作
6. [docs/concept.md](docs/concept.md)——概念和训练对象
7. [docs/decisions_and_open_questions.md](docs/decisions_and_open_questions.md)
8. [docs/novelty_and_landscape.md](docs/novelty_and_landscape.md)
9. [docs/new_session_prompt.md](docs/new_session_prompt.md)——可直接交给下一 session 的 prompt

[docs/expert_plan.md](docs/expert_plan.md) 是 2026-07-17 的历史专家建议，不是活动 authority。它的 60/15/15 前置关系、canonical bank、soft geometry、residual escape 和旧阶段顺序已被 owner 明确 supersede。

## 数据与实验骨架

- 数据：只用 LIBERO-90；每个任务 50 条成功 teacher episode。
- 新 split：70 train / 10 validation / 10 reporting-only test，同分布、任务不重叠、只依据语言/task factor/scene 设计。
- source base：从 `lerobot/smolvla_base` 出发，在 70 个 train task 的全部 3500 条 episode 上联合训练一个多任务 embodiment base。
- Writer：跨 70 个 train task 混合训练；每次根据某任务的语言和全部 50 条 action-hidden 完整视频生成一套该任务 LoRA。
- validation/test：Writer 冻结，只看目标任务 language + action-hidden video；不看目标 action。
- target-task RL：val/test 均可进行，但 base 和 Writer 冻结，只更新 task-local LoRA；预算和选择规则先在 validation 冻结。
- direct LoRA：目标任务 teacher action 可见的 task-local oracle/reference，不伪装成与 EMBER 信息条件相同的 baseline。

标准闭环评估采用 LIBERO 官方 task suite 和固定 init states；环境最大 horizon 为 400，SmolVLA 标准 action execution horizon 采用 50。开发期通常先覆盖每任务全部 50 个标准 init states；若 flow sampling 方差需要，再增加独立 policy RNG，而不是拼接不同 checkpoint。

## 代码状态

工作树只保留下一阶段仍有明确用途的最小内核：

- `src/ember/libero_data.py`：LIBERO HDF5/normalization 审计原语；
- `src/ember/libero_task_factors.py`：90 条语言 specification 的 role-aware factor parser；
- `src/ember/writer/model.py`：完整 LoRA Writer；
- `src/ember/writer/temporal.py`：不限制 episode 数量和视频长度的层次注意力编码器；
- `src/ember/writer/data.py`：完整 action-hidden 视频流与 functional-query 数据；
- `src/ember/writer/topology.py`：GPU/NUMA 绑定；
- `src/ember/eval_artifacts.py`：紧凑视频 gallery；
- `src/ember/runtime_env.py`：锁定环境的兼容修复。

旧 runner 没有被“藏进 archive 目录”；它们由 Git 历史保存。下一 session 应在完成 70/10/10 封存后复用这些内核，建立一条新的单一 canonical 训练/评估入口。

## 工作原则

真实科学实验优先。配置、manifest、测试、文档和 runner 只做到保证机械正确、exact resume、matched fairness、数据隔离和可解释结果所需的最小程度。训练尽量使用全部 8 张合法空闲 A100，每卡平均保留约 10GB；评估优化有效 rollout/秒，不用虚假显存占用代替吞吐。

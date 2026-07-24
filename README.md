# EMBER

EMBER 研究能否把廉价、没有目标机器人 action 标注的教学视频编译成可直接执行、也可继续用环境反馈优化的 VLA 参数：

```text
task language + exactly one action-hidden teaching video
                    -> shared Writer
                    -> complete task-specific LoRA
```

## 当前主线

- Backbone 从 generic `lerobot/pi05_base` 开始，但不直接以其 `0/400` LIBERO表现作为Writer地基。
- 先对 LIBERO-90 与目标 LIBERO-40 做 specification-only semantic/composition overlap audit，排除重合 source tasks；在剩余 tasks × 每 task 50 success episodes 上联合 action-SFT，并冻结一个共享 π0.5-LIBERO source base。
- source base 快速覆盖测试全部40个目标tasks，只要求开始出现跨多个task的部分真实成功，不追求先把base训到高ceiling，也不能只靠单个易task的aggregate。
- 当前source base从generic base fresh训练1,000 steps；其余适用阶段先短profile，再以廉价screen和曲线斜率快速筛选，完整validation只给少量候选。约2小时是防预算暴走的上限而非训练目标；到上限仍未充分训练则记录后停止。
- 目标 benchmark 为 LIBERO-Spatial/Object/Goal/Long 四 suites。development split 每 suite 6 train / 2 validation / 2 test，共24/8/8；final将validation合入形成32 source / 8 test。
- `Action-Supervised Writer (AS-Writer)` 在source tasks上以一条视频生成LoRA，同task action episode/chunk只进functional loss，视频/action独立随机采样。
- `Reward-Trained Writer (RL-Writer)` 与完整AS best分开：新架构先做短、均衡AS cold start，直到24个train tasks各有至少一次official random-reset success，再关闭action入口并转纯source reward训练。
- `Source-SFT` 从同一source base在24/32 source tasks上联合训练一套shared LoRA，不看held video；它独立按validation选最佳，不强制匹配AS-Writer训练步数。
- development和final都增加seen-task comparison；AS/RL Writer还必须比较correct video与另一suite的wrong video。
- 第一轮只跑一个training seed。开发选定后，AS-Writer、RL-Writer（若成立）和Source-SFT都在合并后的32 source tasks上重新训练，再统一做seen与zero-interaction test。
- test打开后，直接在每个test task上将identity-init、AS-Writer-init、RL-Writer-init三臂task-local LoRA RL训练到各自接近最佳；不在validation上提前冻结这段RL。
- 最后使用8个test tasks、每task全部50条action episodes，联合训练一套shared target-action LoRA oracle；不是每task一套LoRA。
- 有时间再做ViVLA-style matched reproduction；outer learning不是核心完成条件。

## 已完成证据

generic `pi05_base` 在预封存8个test tasks、每task50个官方fixed states上为 `0/400`。该结果只说明原始模型没有可用LIBERO控制能力，不评价EMBER。合同与result seal位于 `configs/libero_24_8_8_v1/`。

## 约束

- 不使用 `pi05_libero`，因为它读过目标40 tasks actions。
- 不使用bank、geometry、shared subspace、residual escape、额外shared adapter、旧SmolVLA活动路径或MemLLM。
- 所有下游方法从同一冻结source base、normalization和policy接口出发；Writer生成sealed rank-16 task LoRA，capacity-matched Source-SFT可用rank128，比较时显式报告各自参数量而不机械强制相同rank。
- 训练最多8张A100；GPU0不得堆额外CUDA角色。评测改用cost-balanced state sharding和动态调度，避免horizon-520长任务拖尾。
- 详细阶段、信息墙与执行合同见 `docs/execution_brief.md` 和 `task_plan.md`。

## 阅读顺序

1. `AGENTS.md`
2. `docs/execution_brief.md`
3. `task_plan.md`
4. `findings.md`
5. `progress.md`
6. `docs/concept.md`
7. `docs/decisions_and_open_questions.md`
8. `docs/novelty_and_landscape.md`

`docs/expert_plan.md` 是历史原文，不是活动 authority。

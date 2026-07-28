# EMBER

EMBER 研究能否把一条没有目标机器人 action 标注的教学视频，一次性编译成
能让 frozen VLA 完成对应任务的完整 task-specific LoRA：

```text
task language + exactly one action-hidden teaching video
                    -> shared Writer
                    -> sealed rank-16 task LoRA
                    -> frozen π0.5-LIBERO source policy
```

Writer 只在 rollout 前运行一次；policy 随后依据实时 observation/state 闭环
执行。Writer 不读取 teacher action、proprio、reward、terminal、task ID、
filename 或隐藏 normalization。

## 当前主线

- 共同起点是 generic `lerobot/pi05_base`，不是读过目标 LIBERO-40 actions 的
  `pi05_libero`。
- LIBERO-90 与目标 LIBERO-40 的 3,600-pair specification-only audit 已封存：
  排除 19 个 exact semantic/composition 重合 source tasks，在余下 71 tasks ×
  每 task 50 条成功 teacher episodes 上联合 action-SFT，并冻结共享
  π0.5-LIBERO source base。
- 目标 benchmark 是 LIBERO-Spatial/Object/Goal/Long 共 40 tasks。固定
  development split 为每 suite 6 train / 2 validation / 2 test，总计
  24/8/8；方法选定后合并成 32 source / 8 test。
- v4 的 `shuffled=148/400 > correct=109/400` 暴露了低层
  phase/translation 旁路；v5 虽在内部形成顺序表征，却没有把它稳定传到
  effective LoRA/action。v5.2 首次同时通过 wrong/shuffled/reversed 行为门，
  step900 五臂为 `132/138/74/82/83`，但 absolute 与跨 task 稳定性仍不够。
- 当前唯一活动 Writer 是
  [`v6`](docs/action_forecast_writer_v6_design.md)：Semantic Set 使用
  `mean backbone + task-selected centered residual`；Procedure 联合 frozen
  Action-Expert probe 与 task-grounded adjacent visual transition；两路再经
  canonical compiler/refiner 生成完整 LoRA。v5.3 仅作 v6 的直接设计来源。
- v6 fresh 训练直接使用 task-complete update：4 ranks × 6 tasks/rank/macro，
  task 内 `B_a=20` action queries 求均值，24 tasks 等权；前 5 个 local task
  backward 使用 `no_sync`，第 6 个同步，随后只做一次 AdamW update。首段
  200 macros、每 25 macros 保存；除非 absolute 明显下降，默认续到 macro400。
- v6 用 fixed correct400 选择 observed-best，再对 best 做
  correct/same-task-other/wrong/shuffled/reversed full400 与内部
  Core/Procedure/effective-LoRA/action 传递检查。
- v6 确认后，必须从同一 frozen source base fresh 训练 corrected mixed-task
  rank-128 Source-SFT；随后做 matched one-shot baseline，最后才进入独立
  short-AS cold-start → pure-reward RL-Writer。

## 硬约束

- 当前 focused GPU 工作只使用物理 GPU4–7；GPU0–3 不进入 visible set。
- frame stride 固定为 5；正式 LIBERO preprocessing、horizon、dummy settling、
  success termination 与 frozen normalization 不变。
- 不使用 bank、geometry、shared update subspace、residual escape、额外 shared
  adapter 或 MemLLM。
- 旧 SmolVLA、70/10/10、Phase A–F、flat task-local RL 和 flat Writer-RL
  可执行路径已从工作树退役；provenance 只保留在 Git 历史与 evidence ledger。
- 当前运行状态和下一动作只看
  [`docs/active_session_handoff.md`](docs/active_session_handoff.md)；架构与长期
  科学合同分别看 v6 design、`AGENTS.md` 和 `docs/execution_brief.md`。

## 阅读顺序

1. `AGENTS.md`
2. `docs/active_session_handoff.md`
3. `docs/execution_brief.md`
4. `docs/action_forecast_writer_expert_consultation.md`
5. `docs/action_forecast_writer_design.md`
6. `docs/action_forecast_writer_v4_root_cause.md`
7. `docs/action_forecast_writer_v5_design.md`
8. `docs/action_forecast_writer_v5_1_proposal.md`
9. `docs/action_forecast_writer_v5_2_design.md`
10. `docs/action_forecast_writer_v5_3_design.md`
11. `docs/action_forecast_writer_v6_design.md`
12. `task_plan.md`
13. `findings.md`
14. `progress.md`
15. `docs/concept.md`
16. `docs/decisions_and_open_questions.md`
17. `docs/novelty_and_landscape.md`

外部专家所需的 v4 结构、实验 aggregate、逐任务结果和根因证据集中在
[`docs/action_forecast_writer_expert_consultation.md`](docs/action_forecast_writer_expert_consultation.md)
与 [`docs/action_forecast_writer_v4_root_cause.md`](docs/action_forecast_writer_v4_root_cause.md)。

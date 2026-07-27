# EMBER

EMBER 研究能否把廉价、没有目标机器人 action 标注的教学视频编译成可直接执行、也可继续用环境反馈优化的 VLA 参数：

```text
task language + exactly one action-hidden teaching video
                    -> shared Writer
                    -> complete task-specific LoRA
```

## 当前主线

- Backbone 从 generic `lerobot/pi05_base` 开始，但不直接以其 `0/400` LIBERO表现作为Writer地基。
- 已完成 LIBERO-90 与目标 LIBERO-40 的3600-pair specification-only
  semantic/composition overlap audit：排除19个重合source tasks，保留71个；
  在这71 tasks × 每task 50 success episodes上联合action-SFT，并冻结一个
  共享π0.5-LIBERO source base。
- source base 快速覆盖测试全部40个目标tasks，只要求开始出现跨多个task的部分真实成功，不追求先把base训到高ceiling，也不能只靠单个易task的aggregate。
- source base已从generic base fresh训练并封存1,000-step raw policy。
  Action-Forecast Writer v4已从fresh identity训练至step2400并停止；固定
  400-episode validation observed-best为step825的`109/400`。完整rollout中
  `shuffled=148/400`、`reversed=126/400`反而高于`correct=109/400`。
  根因复审定位为：positive AS无法识别demo高层过程、visual-state被旁路、
  Meta路径学成低层phase/translation code，再被absolute-time
  Plan/Revision放大。
- Semantic Core + Causal Procedure v5已fresh训练至step1800并停止。
  correct-video observed-best为step1000/1400的`115/400`；step1400五臂
  `correct/same/wrong/shuffled/reversed=115/108/74/113/114`。wrong-video
  paired效应显著，但shuffle/reverse与correct等价；内部Procedure虽有强顺序
  差，downstream fusion却把它压弱到effective LoRA和policy action，因此v5
  顺序行为硬门失败。
- 当前唯一活动架构是
  [`Writer v5.1`](docs/action_forecast_writer_v5_1_proposal.md)：
  text-only Gemma task-token queries与multimodal per-frame task-token evidence
  经token-aligned frame-set attention形成language-axis Semantic Core；固定
  native Action Expert suffix形成causal Procedure；中心化Procedure通过
  zero-init AdaLN调制Core slots，再由一个post-fusion slot block生成完整LoRA。
  机械预算`10,244,872`，低于rank-128 Source-SFT。训练仍为每rank每step
  1 task / 1 video / 1 LoRA / 完整`B_a` action batch，不加辅助顺序loss。
  v5.1必须重新profile显存和step吞吐，首段按实测效率取约一小时；第二/第三段
  只有在上一段特异性、absolute和曲线共同支持时才单独启动。
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
- 训练最多8张A100；当前focused v5.1工作只使用物理GPU4–7，0–3不进入visible
  set。评测使用cost-balanced state sharding和动态调度，避免horizon-520长任务
  拖尾。
- 当前v5.1设计见`docs/action_forecast_writer_v5_1_proposal.md`；已封存v5见
  `docs/action_forecast_writer_v5_design.md`，已封存v4实现见
  `docs/action_forecast_writer_design.md`；完整v4根因复审见
  `docs/action_forecast_writer_v4_root_cause.md`；详细阶段、信息墙与执行口径
  见`docs/execution_brief.md`和`task_plan.md`。

## 外部专家咨询入口

Action-Forecast Writer 的研究动机、全部关键架构演进、已封存 v4 的每个模块、
完整内部/rollout证据和当时待分析问题，集中记录在
[`docs/action_forecast_writer_expert_consultation.md`](docs/action_forecast_writer_expert_consultation.md)。
该文档面向只能访问远程仓库的读者，不依赖历史聊天或本地主机结果目录。
专家建议之后完成的因果反事实与根因复审记录在
[`docs/action_forecast_writer_v4_root_cause.md`](docs/action_forecast_writer_v4_root_cause.md)；
后续v5设计与失败证据见
[`docs/action_forecast_writer_v5_design.md`](docs/action_forecast_writer_v5_design.md)
和`docs/active_session_handoff.md`；当前v5.1见
[`docs/action_forecast_writer_v5_1_proposal.md`](docs/action_forecast_writer_v5_1_proposal.md)。

## 阅读顺序

1. `AGENTS.md`
2. `docs/active_session_handoff.md`
3. `docs/execution_brief.md`
4. `docs/action_forecast_writer_expert_consultation.md`
5. `docs/action_forecast_writer_design.md`
6. `docs/action_forecast_writer_v4_root_cause.md`
7. `docs/action_forecast_writer_v5_design.md`
8. `docs/action_forecast_writer_v5_1_proposal.md`
9. `task_plan.md`
10. `findings.md`
11. `progress.md`
12. `docs/concept.md`
13. `docs/decisions_and_open_questions.md`
14. `docs/novelty_and_landscape.md`

`docs/expert_plan.md` 是历史原文，不是活动 authority。

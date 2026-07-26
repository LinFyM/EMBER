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
- source base已从generic base fresh训练并封存1,000-step raw policy。
  Action-Forecast Writer v4已从fresh identity训练至step2400并停止；固定
  400-episode validation observed-best为step825的`109/400`。它通过内部视频/
  顺序差异门，但完整rollout中`shuffled=148/400`、
  `reversed=126/400`反而高于`correct=109/400`，行为特异性硬门失败。
  fixed-anchor shuffle仍为`136/400`，排除随机首帧anchor是主因。进一步的
  train-only隐藏语义审计、随机permutation共识、forecast分量移植和Object定向
  rollout证明：v4 visual-state未成为必要信息瓶颈；同task独立video/action的
  positive AS目标不能识别demo高层过程语义；两个Meta-LoRA逐渐把forecast变成
  低层demo/phase/translation code；未经校准的absolute-time Plan/Revision再把
  它放大成controller。只重排前三维translation已得到`79/100`，几乎复现完整
  shuffled的`82/100`。此前frame-local Intent + adjacent Transition只能修复
  最后一层放大器，现已从“已拍板v5”降为局部候选；下一版设计重新开放，尚未
  实现或训练，也未进入RL。
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
- 已封存v4实现见`docs/action_forecast_writer_design.md`；完整根因复审和下一版
  未决合同见`docs/action_forecast_writer_v5_decision.md`；详细阶段、
  信息墙与执行口径见`docs/execution_brief.md`和`task_plan.md`。

## 外部专家咨询入口

当前 Action-Forecast Writer 的研究动机、全部关键架构演进、最新 v4 每个模块、
完整内部/rollout证据和待分析问题，集中记录在
[`docs/action_forecast_writer_expert_consultation.md`](docs/action_forecast_writer_expert_consultation.md)。
该文档面向只能访问远程仓库的读者，不依赖历史聊天或本地主机结果目录。
专家建议之后完成的因果反事实、根因复审与下一版未决状态记录在
[`docs/action_forecast_writer_v5_decision.md`](docs/action_forecast_writer_v5_decision.md)。

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

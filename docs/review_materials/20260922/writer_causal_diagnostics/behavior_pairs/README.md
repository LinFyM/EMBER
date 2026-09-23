# O1200 成功 / C600 失败的既有执行轨迹复核包

这是专家要求的**事后失败定位材料**，不是新的成功率样本、评测、训练或模型推理。它从已完成的 D1 `O1200/CC` 和 `C600/CC`
原始行中，按每个 suite 的 `task_id`、`init_state_id` 升序，固定选择第一个 `O1200 success=True`、`C600 success=False`
且 teacher 条件完全相同的条件。因此恰好四对；该选择使用已知结果，绝不能用于新的性能率或方法优劣统计。

所有四条保存轨迹均为 compact capture：原件保存了每个 replan 的 Pi05 state 和原始 50×7 normalized action chunk，
没有逐时刻 RGB。导出器只将已保存 chunk 的前五个实际执行动作以原生 source normalization 反归一化后重放到原环境，
没有加载或调用 Writer、source policy、checkpoint、action sampler 或模型 forward。每一条 replay 都与原件的 replan
Pi05 state、success/steps 和 stage-predicate transitions 对照；若任一对照失败，`replay_verification.csv` 会标记
`illustrative_only_replay_drift`，画面不可代替原始执行结果。

## 四个固定对照

| suite | task/state | O1200 | C600 | teacher | paired execution | replay fidelity |
| --- | --- | ---: | ---: | ---: | --- | --- |
| libero_spatial | 5/1 | success, 143 | failure, 220 | 48 | [`libero_spatial_task_05_state_001.mp4`](paired_execution/libero_spatial_task_05_state_001.mp4) | O=verified_saved_action_replay; C=verified_saved_action_replay |
| libero_object | 2/0 | success, 202 | failure, 280 | 46 | [`libero_object_task_02_state_000.mp4`](paired_execution/libero_object_task_02_state_000.mp4) | O=verified_saved_action_replay; C=verified_saved_action_replay |
| libero_goal | 0/1 | success, 116 | failure, 300 | 48 | [`libero_goal_task_00_state_001.mp4`](paired_execution/libero_goal_task_00_state_001.mp4) | O=verified_saved_action_replay; C=verified_saved_action_replay |
| libero_10 | 4/1 | success, 254 | failure, 520 | 48 | [`libero_10_task_04_state_001.mp4`](paired_execution/libero_10_task_04_state_001.mp4) | O=verified_saved_action_replay; C=verified_saved_action_replay |

## 文件与阅读顺序

- `pairs.csv`：固定选择、原始 trajectory 路径、结果和对应的并排视频。每个视频一帧对应一个 control step；早结束的一侧保持最终画面，时间轴不再压缩。布局为上排 agentview、下排 wrist，左 O1200、右 C600，均是策略看到的 180° 旋转视角。
- `executed_action_and_state.csv`：每一步真正传给环境的 7 维 action、保存的 normalized action、以及 action 前后 8 维 Pi05 robot state。
- `predicate_timeline.csv`：step 0 和每个 control step 的所有目标 predicate；`original_predicate_transitions.csv` 是 D1 原始行直接保存的变化点，二者不应混淆。
- `replay_verification.csv`：replay 与保存 state / 原始结果 / 原始 predicate transitions 的逐轨迹一致性检查。
- `teacher_visuals/`：同一正确 teacher demo 的单相机轻量视频和 contact sheet；它是 action-hidden RGB，不包含 teacher action。
- `selected_original_rows_O1200_CC.csv` 与 `selected_original_rows_C600_CC.csv`：未改写的原始 D1 CSV 行；`selection_contract.json` 记录选择和重放边界。

## 不应使用的既有 full capture

D1 的预保存 full RGB 条件 `global_task_id=5/state=0` 和 `global_task_id=37/state=1` 都不是本包的正反对照：
O1200 在这两个条件也失败。它们没有被作为“旧成功、新失败”的证据导出或引用。

## Teacher 轻量画面

| pair | demo | sampled/raw frames | video | contact sheet |
| --- | ---: | ---: | --- | --- |
| libero_spatial_task_05_state_001 | 48 | 23/108 | [`libero_spatial_task_05_state_001_teacher_demo_48.mp4`](teacher_visuals/libero_spatial_task_05_state_001_teacher_demo_48.mp4) | [`libero_spatial_task_05_state_001_teacher_demo_48_contact_sheet.png`](teacher_visuals/libero_spatial_task_05_state_001_teacher_demo_48_contact_sheet.png) |
| libero_object_task_02_state_000 | 46 | 25/118 | [`libero_object_task_02_state_000_teacher_demo_46.mp4`](teacher_visuals/libero_object_task_02_state_000_teacher_demo_46.mp4) | [`libero_object_task_02_state_000_teacher_demo_46_contact_sheet.png`](teacher_visuals/libero_object_task_02_state_000_teacher_demo_46_contact_sheet.png) |
| libero_goal_task_00_state_001 | 48 | 30/143 | [`libero_goal_task_00_state_001_teacher_demo_48.mp4`](teacher_visuals/libero_goal_task_00_state_001_teacher_demo_48.mp4) | [`libero_goal_task_00_state_001_teacher_demo_48_contact_sheet.png`](teacher_visuals/libero_goal_task_00_state_001_teacher_demo_48_contact_sheet.png) |
| libero_10_task_04_state_001 | 48 | 48/235 | [`libero_10_task_04_state_001_teacher_demo_48.mp4`](teacher_visuals/libero_10_task_04_state_001_teacher_demo_48.mp4) | [`libero_10_task_04_state_001_teacher_demo_48_contact_sheet.png`](teacher_visuals/libero_10_task_04_state_001_teacher_demo_48_contact_sheet.png) |

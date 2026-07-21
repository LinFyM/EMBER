# EMBER Progress and Handoff

最后更新：2026-07-21。

## 当前状态

- active Goal 已替换为 π0.5 feasibility test；没有 token budget。
- Git `main` 在开始本轮时为 `215fb3c`，与 `origin/main` 同步且工作树干净。
- 当前主线是四个标准 LIBERO suites，每 suite 7 train / 1 validation / 2 test；不是 LIBERO-90 70/10/10。
- 当前 backbone 是 generic π0.5；先不建立 source embodiment base。
- EMBER train/test 都只输入一条 action-hidden teacher video；source video 与 action episode在 task 内独立采样。
- 当前尚未运行 π0.5 test，也没有读取 8 test tasks 的 teacher actions。
- `/data/ymdai` 当前约 333GB；在 500GB operator cap 下有约 167GB headroom。新模型、环境、28-task source-only numeric data 与 outputs 的预计峰值必须保持在该余量内。
- 当前 `.venv` 有 PyTorch 2.11/CUDA 12.8、8 张可见 GPU、LeRobot 0.6.0 和 LIBERO simulator；尚无本地 π0.5 checkpoint cache。

## 已核验的官方事实

- Physical Intelligence 发布 generic `pi05_base`（用于 fine-tuning）和 separate `pi05_libero`（在 LIBERO 上 fine-tuned，用于直接 inference）。当前只允许前者。
- 官方 π0.5 LIBERO config：`pi05=True`、action horizon 10、`discrete_state_input=False`、10 flow inference steps。
- 官方 evaluator：256 render、224 resize、两相机 180° rotate、replan 5、seed 7、50 trials/task、10 dummy settling steps；suite horizons 220/280/300/520。
- generic base 没有 LIBERO action unnormalization stats，不能把空 postprocessor 当有效控制接口；当前只允许用 28 source tasks 计算接口 stats，不更新模型权重。

## 新 split（已封存）

| suite | train | validation | test |
| --- | --- | --- | --- |
| `libero_spatial` | 0,2,3,4,5,7,9 | 1 | 6,8 |
| `libero_object` | 1,2,4,5,6,8,9 | 3 | 0,7 |
| `libero_goal` | 0,1,2,5,6,8,9 | 3 | 4,7 |
| `libero_10` | 2,4,5,6,7,8,9 | 1 | 0,3 |

算法 seed `20260721`；SHA256 key 为 `seed\0suite\0task_name\0language\0bddl_file`，按 digest 升序前 7 / 第 8 / 后 2。canonical seal 为 `configs/libero_28_4_8_v1/protocol.json`。

## 下一动作

1. 完成活动 docs 与 split seal。
2. 建立最小 official-compatible π0.5 base evaluator及 train-only normalization。
3. mechanics smoke。
4. live GPU preflight 后 8 卡并行评测 8 test tasks。
5. 结果、raw rows 和 hashes写回，commit/push 后停住。

## 历史边界

旧 SmolVLA 70/10/10 已真实完成到 Phase F freeze，包含 base `56/500`、cold Writer `63/500`、identity-RL `54/500`、Writer-RL `74/500`、direct oracle `186/500` 等 validation 结果。这些仍是可引用 provenance，但与新 split/backbone/data semantics 不兼容，不得复用其 checkpoint、normalization 或继续旧 test。

## 禁止自动续跑

π0.5 的 8-task result 一旦生成，必须停止。不得据结果自行训练 source base、Writer、RL 或 baseline。

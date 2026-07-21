# EMBER Progress and Handoff

最后更新：2026-07-21。

## 当前状态

- active Goal 已替换为 π0.5 feasibility test；没有 token budget。
- 本次 owner correction 前的最新已推送 commit 为 `6bb13de`；当前工作树正在实现 24/8/8 protocol 与 π0.5 evaluator，正式 launch 前必须再提交并保持 clean。
- 当前主线是四个标准 LIBERO suites，每 suite 6 train / 2 validation / 2 test；不是 LIBERO-90 70/10/10。
- 当前 backbone 是 generic π0.5；先不建立 source embodiment base。
- EMBER train/test 都只输入一条 action-hidden teacher video；source video 与 action episode在 task 内独立采样。
- 原 Writer cold start 已改名 `Action-Supervised Writer (AS-Writer)`；`Reward-Trained Writer (RL-Writer)` 改为随机初始化或仅极短 AS warm-up 后的独立 source-reward 路线，不默认继承 AS-Writer。
- 最终 baseline 增加 `Source-SFT π0.5`：合并后的 32 source tasks、与 AS-Writer 相同 optimizer-step budget、test 不看 teacher video。
- 当前尚未运行正式 π0.5 test，也没有读取 8 test tasks 的 teacher actions；单卡 mechanics/throughput smoke 已完成且不作性能判断。
- `/data/ymdai` 当前约 333GB；在 500GB operator cap 下有约 167GB headroom。新模型、环境、24-task source-only numeric data 与 outputs 的预计峰值必须保持在该余量内。
- 当前 `.venv` 有 PyTorch 2.11/CUDA 12.8、8 张可见 GPU、LeRobot 0.6.0 和 LIBERO simulator；generic π0.5 checkpoint revision `7de6639` 已下载，14,467,165,872-byte weights SHA256 为 `0eb11ca9...ca59b0f`。

## 已核验的官方事实

- Physical Intelligence 发布 generic `pi05_base`（用于 fine-tuning）和 separate `pi05_libero`（在 LIBERO 上 fine-tuned，用于直接 inference）。当前只允许前者。
- 官方 OpenPI LIBERO config 使用 action horizon 10；当前 LeRobot official conversion 的 checkpoint config 为 model chunk 50、`n_action_steps=10`，evaluator 与官方 OpenPI runner 相同只执行前 5 actions 后重规划；10 flow inference steps。
- 官方 evaluator：256 render、224 resize、两相机 180° rotate、replan 5、seed 7、50 trials/task、10 dummy settling steps；suite horizons 220/280/300/520。
- generic base 没有 LIBERO action unnormalization stats，不能把空 postprocessor 当有效控制接口；当前只允许用 24 development-train tasks 计算接口 stats，不更新模型权重。
- train-only normalization 已完成：只对 377 个 parquet 先读取 `task_index`，随后只在 task IDs 全属于 24 development-train role 的 62 个文件读取 state/action；共 43,785 source rows，24 tasks 均有贡献，8 validation 与 8 test actions 未读取。artifact SHA256 为 `a97857dc...3b1f1`。
- LeRobot 默认 tokenizer loader 会访问 gated Google repo；改用同一 OpenPI revision 明确引用、可匿名读取的 `gs://big_vision/paligemma_tokenizer.model`，4,264,023 bytes、SHA256 `8986bb4f...168fc6`。prompt/state discretization 逐 token 对官方 OpenPI 格式核验通过。
- evaluator mechanics smoke 已验证模型加载、预处理、batched action、LIBERO reset/step 与结果落盘。吞吐 profile 在同一 Spatial task 的 full-horizon 失败 episodes 上为：1 env 27.52 秒/episode、8 env 19.76 秒/episode、16 env 19.58 秒/episode；8→16 只提升约 0.9%，且峰值显存约从 20.1GB 增至 23.2GB，因此正式使用每 policy process 8 个 env。这里的 0/1、0/8、0/16 不作科学性能证据。
- 首次 8 卡 formal launch 在 rollout 前暴露 EGL rank 映射错误：旧 evaluator 固定 `MUJOCO_EGL_DEVICE_ID=0`，导致物理 GPU1–7 的 robosuite import 明确拒绝，GPU0 未完成即主动终止；该批输出标为 invalid，不进入 aggregate。现改为从每个单卡进程唯一的 numeric `CUDA_VISIBLE_DEVICES` 派生 EGL device，并已在物理 GPU1 完成一条 load/env/rollout smoke。

## 新 split（已封存）

| suite | train | validation | test |
| --- | --- | --- | --- |
| `libero_spatial` | 0,2,4,5,7,9 | 1,3 | 6,8 |
| `libero_object` | 2,4,5,6,8,9 | 1,3 | 0,7 |
| `libero_goal` | 0,1,2,5,8,9 | 3,6 | 4,7 |
| `libero_10` | 4,5,6,7,8,9 | 1,2 | 0,3 |

算法 seed `20260721`；SHA256 key 为 `seed\0suite\0task_name\0language\0bddl_file`，按 digest 升序前 6 / 中间 2 / 后 2。canonical seal 为 `configs/libero_24_8_8_v1/protocol.json`。这次调整发生在任何 π0.5 test rollout 之前，8 个 test task 未变。

## 下一动作

1. 完成活动 docs 与 split seal。
2. 完成已实现的 official-compatible π0.5 base evaluator及 24-task train-only normalization；评测用每 GPU 一个 policy process、进程内 8 个持久 env 做 batched planning。
3. mechanics smoke。
4. live GPU preflight 后 8 卡并行评测 8 test tasks。
5. 结果、raw rows 和 hashes写回，commit/push 后停住。

## 历史边界

旧 SmolVLA 70/10/10 已真实完成到 Phase F freeze，包含 base `56/500`、cold Writer `63/500`、identity-RL `54/500`、Writer-RL `74/500`、direct oracle `186/500` 等 validation 结果。这些仍是可引用 provenance，但与新 split/backbone/data semantics 不兼容，不得复用其 checkpoint、normalization 或继续旧 test。

## 禁止自动续跑

π0.5 的 8-task result 一旦生成，必须停止。不得据结果自行训练 source base、Writer、RL 或 baseline。

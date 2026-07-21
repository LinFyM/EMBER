# EMBER Task Plan

最后更新：2026-07-21。当前 Goal 是一次有明确停止点的 π0.5 feasibility evaluation，不是旧 Phase F 的自动续跑。

## 当前完成定义

只有以下全部完成，当前 Goal 才完成：

1. 活动 authority、概念、决策、handoff 与新 session prompt 全部对齐 π0.5 / 24-8-8 / one-video；
2. specification-only 6/2/2-per-suite split、任务文本、算法、seed 和 hashes 封存；
3. generic `pi05_base` 与官方 LIBERO 推理参数核验，不使用 `pi05_libero`；
4. 只用 24 development-train tasks 计算必要的 interface normalization，validation/test action read count 为零；
5. 8 个 test tasks × 50 fixed states 完整评测，保存 400 条 raw rows 与逐任务成功数；
6. 运行时 8 张卡进程角色一致、GPU0 无额外 CUDA process；
7. 结果写回、验证、commit、push；
8. 立即停止，不启动后续实验。

## Phase A：协议与 split

状态：已完成，seal 为 `configs/libero_24_8_8_v1/`。

- [x] 确定四 suites 与 24/8/8 数量。
- [x] 确定 one-video train/test semantics。
- [x] 确定 source video/action episode 独立随机采样。
- [x] 确定 generic π0.5 先测、结果后停。
- [x] 写完全部活动 authority。
- [x] 封存 `configs/libero_24_8_8_v1/` 并验证 hashes。

## Phase B：π0.5 official-path mechanics

状态：已完成；小分母 rollout 只用于 mechanics/throughput，不作性能判断。

- [x] 固定 Physical Intelligence/openpi revision、generic base checkpoint revision 与文件 hash。
- [x] 取得 24 development-train tasks 的必要 metadata/state/action columns，不下载 image/video payload。
- [x] 计算 train-only π0.5 LIBERO normalization；验证 val/test teacher action access 为零。
- [x] 实现单一最小 evaluator；固定 official preprocessing、replan、horizon、seed、init states 和 raw row schema。
- [x] 单卡 mechanics smoke 通过；未据小分母判断性能。
- [x] 吞吐 profile：batch 8 为 158.07 秒/8 episodes，batch 16 为 313.24 秒/16 episodes，仅快约 0.9%；正式锁定 8 env/process，避免更大 batch 的尾批和稳定性代价。

## Phase C：8-task test

状态：已完成；generic π0.5 为 0/400，现按 owner 要求停止。

- [x] GPU/storage live preflight。
- [x] 8 GPU 同构 launch：一 task / 一 GPU / 一 policy CUDA process。
- [x] 每 task 50 fixed init states，共 400 rows。
- [x] 保存 command、revisions、normalization provenance、GPU/process snapshot、wall-clock、raw rows、summary 和 hashes。
- [x] 更新 `findings.md`、`progress.md` 和本计划。
- [x] 验证、commit、push，当前 Goal complete 并停住。

## 未来计划（本轮禁止执行）

- 开发：24 source tasks 训练 one-video 方法，8 validation tasks 选方法。
- 最终：合并 validation，32 source tasks 重训，8 test tasks 统一评估。
- Writer 路线：`Action-Supervised Writer (AS-Writer)`；独立的 `Reward-Trained Writer (RL-Writer)` 从随机初始化或仅极短 AS warm-up 开始，不默认继承 AS-Writer。
- 可选：matched task-local identity/Writer-init RL；Phase F 后 outer learning。
- Baselines：generic/frozen π0.5、在最终 32 source tasks 上与 AS-Writer 匹配 optimizer steps 的 `Source-SFT π0.5`、ViVLA-style one-video direct conditioning、direct target-action LoRA oracle，以及相同 reward budget 的 matched RL。

## 每次运行前

- [ ] Git revision/status 明确。
- [ ] live `nvidia-smi` 与 owner/process audit。
- [ ] `/data/ymdai` 当前与预计峰值低于 500GB。
- [ ] 8 卡进程拓扑相同，GPU0 无额外 CUDA role。
- [ ] exact config、checkpoint、data surfaces、output root 和 stop condition 明确。

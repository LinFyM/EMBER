# EMBER Findings

## 2026-07-21 最新 owner correction（活动 authority）

- development split 在任何新 π0.5 rollout 前从每 suite 7/1/2 改为 6/2/2，总计 24 train / 8 validation / 8 test；沿用同一 specification-only SHA256 排序，8 个 test task 不变。validation 选定方案后合并为最终 32 source / 8 test 并从规定初态重训。
- 原“Writer cold start”正式称为 `Action-Supervised Writer (AS-Writer)`。
- `Reward-Trained Writer (RL-Writer)` 是独立路线：从随机初始化 Writer，或只做预声明的极短 AS warm-up，然后直接用跨 source-task reward 联合训练；默认不从完整 AS-Writer 继续，明确检验没有 teacher actions 能否训练 Writer。
- 最终新增 `Source-SFT π0.5` baseline：在 32 source tasks 上以和 AS-Writer 相同的 optimizer-step budget 做 action-SFT，test 不看 held video/action。它与 AS-Writer 的差异是后者额外得到一条 held teacher video。
- 当前执行边界不变：只运行 generic π0.5 的 8 test-task zero-shot feasibility；结果出来立即停止，不自动运行上述后续训练。
- generic `lerobot/pi05_base` revision `7de663972b7817d2c4cf2d84c821153dfea772e9` 已完整下载；14,467,165,872-byte weights SHA256 为 `0eb11ca9587678c1d2ef8cf32807c29f8ce53a2bfdfc1aa4a4c96f16fca59b0f`。
- 24-task train-only quantile normalization 使用 `HuggingFaceVLA/libero@8695891` 的 column-range reads：先读 377 个 parquet 的 task IDs，只对 62 个纯 source 文件读取 state/action，共 43,785 rows；24 train tasks 均有贡献，validation/test action 列从未在 mixed/held 文件上打开。normalization SHA256 为 `a97857dc...3b1f1`。
- LeRobot processor 默认会向 gated `google/paligemma-3b-pt-224` 请求 tokenizer，匿名环境得到 401。当前改为 OpenPI 同 revision 明确使用、匿名公开的 `gs://big_vision/paligemma_tokenizer.model`；其 SHA256 为 `8986bb4f...168fc6`，本地预处理的 prompt、state binning、BOS、padding/mask 已逐 token 对官方 OpenPI 实现核验。
- π0.5 evaluator mechanics 已通过。单卡吞吐 profile（同一 Spatial task、full-horizon smoke，不作性能判断）：batch 1 为 27.52 秒/episode，batch 8 为 158.07/8=19.76 秒/episode，batch 16 为 313.24/16=19.58 秒/episode；batch 8→16 仅约 0.9% 提升，峰值显存约 20.1→23.2GB。正式评测锁定每卡一个 policy CUDA process、每进程 8 个持久 env；瓶颈是官方 π0.5 推理计算而不是显存容量。
- 首次 8 卡 formal launch 在任何 task 产出结果前失败：robosuite 要求 `MUJOCO_EGL_DEVICE_ID` 是该进程 `CUDA_VISIBLE_DEVICES` 中的物理编号，而 evaluator 固定写 0，使 GPU1–7 import 失败；GPU0 被主动终止。修复为每 rank 派生自身唯一物理 ID 后，GPU1 独立 smoke 已通过。失败 root 保留为 invalid failure packet，不复用、不聚合。
- 修复后 commit `bf27ebc` 上的 8 卡 formal run 全部 exit 0；每卡恰好一个 policy CUDA process，GPU0 无额外进程。8 个 test tasks × 50 fixed states 的 generic `pi05_base` 结果全部为 `0/50`，总计 `0/400`；400 个 `(suite, task, init_state)` 唯一，所有 rows 均到达对应 220/280/300/520 horizon。
- 该结论只说明 generic π0.5 在当前官方 LIBERO 接口下没有直接 zero-shot competence，因此若继续需要 source-side action adaptation/base；它不构成对 one-video Writer 的评价。aggregate SHA256 为 `8ffa816e...7776`，tracked result seal 为 `configs/libero_24_8_8_v1/pi05_base_feasibility_results.json`（SHA256 `c78e92e9...20c2`）。按 owner stop condition，不自动启动任何后续训练。

## 2026-07-21 protocol reset

owner 已将活动研究协议改为 generic π0.5 + 四个标准 LIBERO suites + 每 suite 6/2/2 + one-video Writer。下文全部 SmolVLA/70-10-10 数字仍是真实历史证据，但从本节开始只作 provenance，不能作为新协议 checkpoint、normalization、split 或完成状态。

generic `pi05_base` 在预封存 8 test tasks 的 400 个 official fixed-state rollouts 上为 `0/400`，每 task 均为 `0/50`。该 feasibility 问题已经关闭，当前等待 owner 决定是否建立 source-side action adaptation/base。

已核验官方实现事实：generic `pi05_base` 是 fine-tuning base；`pi05_libero` 是另一个在 LIBERO 上 action-finetuned 的 inference checkpoint。当前禁止后者。官方 generic Hugging Face pre/post processors没有可执行 LIBERO action space 的 normalization state，因此有效的 generic-base test 必须从 24 development-train tasks 计算 interface-only state/action stats，同时保持 validation/test action read count 为零。

---

本文件只保留会影响当前科学解释的证据。父提交 `999df28` 保存 2026-07-17 至 2026-07-20 的完整逐次日志、旧配置、runner 和测试；外部 checksummed artifacts 保存原始 rows、metrics、视频和 failure packets。这里不把历史过程重新伪装成活动合同。

## 当前结论

### 新 70/10/10 protocol：已永久封存

- 在读取任何新协议 policy outcome 前，使用 90 条官方 language、scene 和 role factors，通过 `scipy_milp_highs_three_stage_lexicographic_v1`、seed `20260720` 封存 70 train / 10 validation / 10 test。
- validation IDs 为 `[0, 8, 15, 28, 40, 56, 61, 71, 85, 88]`；test IDs 为 `[4, 7, 11, 32, 41, 59, 60, 70, 84, 86]`；其余 70 个为 train。
- validation/test 各自含 10 个不同 scene，并恰好共享相同 scene 分布：5 Kitchen、3 Living Room、2 Study；两者均为 5 个单步/5 个双步、2 actuation / 3 single-place / 4 pick-place / 1 compound。
- exact composition group 不跨 split；所有 held task 的每个精确 role atom 在 train 中至少保留 2 个实例。stacking 因这一严格支持约束全部保留在 train。
- 90 个 pinned HDF5 共 `66,658,085,995` bytes，4500 demonstrations、669,043 frames、每 task 50 demos；90 个 official init-state 文件均为 50 states。controller 为 OSC_POSE/20Hz，camera 为 agentview + eye-in-hand、128×128。
- validation/test HDF5 只读取 metadata/shape/hash；normalization 仅从 70×50 train episodes 读取 state/action 数值。producer `env_args` 有 90 个 legacy suite 注记和 6 个 legacy basename 注记，但 canonical HDF5 BDDL basename/language 均通过。
- canonical hashes：factor table `73828b1b...015`、split `996a3061...77e`、data manifest `b18f1cfa...be7e`、train-only normalization `5141e4b3...2d28`；完整值在 `configs/libero90_70_10_10/checksums.sha256`。

### Source-base 正式训练与 source-only 选择：完成，冻结 step630

- 从 pinned `lerobot/smolvla_base` 严格加载 450,046,176 parameters；98,880,992 个 action-expert/projection parameters 可训练，冻结 VLM trainable 泄漏为零。
- 全部 70×50 episodes 对应 537,946 frames；sampler 在跨 rank global task slots 上做 deterministic no-replacement cycles，并保证每个 checkpoint 边界前 3500 episodes 全覆盖。
- 显式 rank-local device 修复后的 8-A100 profile 使用一张卡一个 rank、batch/rank 352：稳态 2.590s/step、1087.2 global samples/s，每卡峰值 allocated/reserved 65.05/67.35GiB，data wait 0.13ms。
- 首次 formal 启动暴露出无索引 `device="cuda"` 会让非零 ranks 在 GPU0 留额外构造 context；它在首个 checkpoint 前被停止且不复用。改为 `cuda:{local_rank}` 后，满 batch steady-step 进程表为每卡恰好一个 CUDA PID、69,124–69,132MiB，GPU0 不再额外堆积。
- 8-rank continuous/resume 对照中，step-1 policy、optimizer、scheduler 和每 rank RNG 起点一致；启用 DDP static graph 后，step-2 policy 文件 SHA256 位级一致，optimizer/scheduler/RNG 逐值一致。默认 DDP 首轮后 bucket 重建是此前微小漂移的工程原因。
- commit `72eb10d` 上的正式 seed-1 trajectory 在约 28 分钟内完成 630/630 steps、退出码 0；210/420/630 三个 checkpoints 均通过 15-file size/SHA manifest 校验。最终 step loss 为 `0.483089`，吞吐 `1084.95 samples/s`，峰值 allocated/reserved 为 `65.05/67.35GiB`。
- step 630 累计 1,774,080 global examples 和 5,040 global task slots；每个 checkpoint 边界均覆盖全部 70 tasks 和每 task 50 episodes。最终 policy SHA256 为 `eb7e01f2...c1f159f`，checkpoint manifest SHA256 为 `89e9f493...ed22c`，launch contract SHA256 为 `22c4ffb5...2e8`。
- source-only loss 三段均值为 `0.80098 → 0.53083 → 0.49775`。同一 8-task × 50-state source-development panel 的 step 210/420/630 h50 成功数为 `3/400 → 8/400 → 15/400`；420→630 是 11 paired gains、4 paired losses，行为改善不只来自总数拼接。
- step-630 per-task 原始成功数为 `{1:0, 2:3, 6:6, 16:4, 46:1, 63:1, 65:0, 73:0}`，说明它在 5/8 个预声明 source tasks 出现 competence，但 3.75% 绝对成功率仍低；因此只允许一次短续训检验是否仍在改善。
- step 210/420/630 `results.json` SHA256 分别为 `b1445ec2...b3893`、`ba8fdbb5...e8cdb`、`b901f758...5ffa4`；每份均为 400 个唯一 `(task_id, init_state_id)` rows，8 ranks、50 states/task、horizon 400，无跨 checkpoint 拼接。
- 只据上述 train/source evidence，封存一次 315-step continuation：从 step 630 完整恢复 optimizer、sampler/RNG 与 interaction-free data cursor，原 cosine scheduler 保持在 decay LR `2.5e-6`，相对 thirds 为 735/840/945。没有读取 validation/test outcome，也不重启高 LR。
- continuation 在 step735/840/945 均完整覆盖70×50，最终累计2,661,120 examples，8卡仍各恰好一个70,176MiB CUDA rank，exit 0；三个 checkpoint 的15个文件均通过 size/SHA 校验。
- step945 source-development 仍为 `15/400`，per-task `{1:0,2:3,6:6,16:5,46:1,63:0,65:0,73:0}`；相对630为5 paired gains、5 paired losses、10 kept successes，净增益0。按预声明停止规则冻结step630，不再评735/840；该选择及完整 artifact hashes 已封存在 `configs/source_base_selected_v1.json`。

### 新 h50 fresh evaluator：mechanics 通过，未打开 test

- evaluator 只暴露 specification-only 预声明的 8 个 train/source-development tasks 和 10 个 validation tasks；reporting-only test role 在 Phase F 前结构性不可解析。
- 使用官方 LIBERO suite/BDDL/controller/camera/normalization、固定 `.pruned_init` states、dummy settling 10、horizon 400、成功即终止和 SmolVLA h50；固定 states 只服务 fresh evaluation，不会进入 RL update 或 adaptation checkpoint selection。
- step-630 mechanics smoke 在 8 ranks 上各跑 1 个不同固定 state，共 8 条唯一 rows；运行时每卡恰好一个 policy CUDA process、显存一致为 3347MiB，退出后全部归零。`0/8` 只是 smoke 小分母，不作性能证据。
- 完整 source-development/validation 评估按 task 同步、state rank-strided、每 rank 4 个持久 async env workers；这使八卡 policy 进程拓扑完全对称，同时把 MuJoCo rollout 吞吐作为优先优化对象。
- source base 冻结之后才打开 validation reference：step630 在 10 tasks × 50 fixed fresh states 上为 `56/500 = 11.2%`，per-task `{0:28,8:0,15:0,28:14,40:0,56:1,61:0,71:0,85:0,88:13}`。结果集中在三个任务，既提供非零 competence，也要求 Writer 的增益必须跨多个类别而非只追随单一易任务；`results.json` SHA256 为 `3d19f00f...0cac9`。该结果没有参与 source-base 选择，test 仍未打开。

### Frozen Writer feature cache：正式 70×50 cache 完成

- 只读取每条 source teacher episode 的 `obs/agentview_rgb` 和 language；不读取 action、proprio、reward、terminal、task ID/file-name features。每帧按 source-base 相同 OpenGL transform 进入 frozen SmolVLA VLM，64 个 960-d 空间 tokens 经固定 `sqrt(960)` normalization 后确定性均值池化为一个 960-d BF16 frame feature。
- 8-rank smoke 为每 rank 1 个不同 train task/episode，共 1,194 frames；所有视觉/语言 features finite，episode offsets 与原 episode lengths 一致。单 task 108–197 frames 的提取 wall time 为 0.63–0.83 秒，按正式 LPT 调度估计全 537,946 frames 约 5 分钟。
- resume 再运行时 8/8 ranks 均验证既有文件 size/SHA 后 `new=0`；模型加载阶段每卡恰好 1 CUDA PID、414MiB，GPU0 无额外 context，退出后 8 卡全清。
- selected step630 的正式 cache 已完成：70 tasks、3500 episodes、537,946 frames、1,034,531,040 tensor bytes、825 language tokens。70 个 task tensor 均独立通过 size/SHA；cache manifest SHA256 为 `ae5854a6...be127`，run contract 内部 SHA256 为 `7b7fb765...e03ce`。全程每卡一个 3900MiB CUDA rank，GPU0 无额外进程。
- validation action-hidden cache 使用同一冻结 VLM 合同，仅读取预封存 10 tasks 的 RGB/language：500 full episodes、63,544 frames、122,236,320 tensor bytes。10 个 task tensor 均独立通过 size/SHA；manifest SHA256 为 `06087541...05221`，extraction SHA256 为 `65d275d0...4a11`。8 卡仍各一个约 3900MiB CUDA rank。

### Writer functional cold-start：真实 profile/resume 通过，formal 已封存

- Writer 生成的全部74个 LoRA A/B tensors 已通过 `torch.func.functional_call` 接入冻结 SmolVLA 标准 flow/action loss；单元 backward 验证 policy 所有物理参数无梯度、Writer 获得 finite gradient。
- source query sampler 继续使用跨rank 70-task no-replacement cycles，并可对每个 checkpoint 生成精确 `(step, rank, batch offset, task, episode, frame)` identity SHA256；不是只保存不可审计的 step 数。
- feature cache 训练侧使用有界 task LRU，每个 task 首次载入验证 size/SHA，换入时不重复做15MB级哈希；这是吞吐优化，不改变 features。
- canonical runner 保存 Writer、optimizer、scheduler、sampler cursor、consumed identity、每rank RNG和完整 launch contract。
- 真实 8-rank functional profile 选择每 rank batch 384（global 3072）：steps 2–35 平均 3.426s、898.1 queries/s，峰值 allocated/reserved 68.00/70.54GiB；8 卡各恰好一个约 72.6–73.4GiB CUDA rank，GPU0 没有额外 context。更大的 448/512/768/896 batch 均在 8 卡对称 OOM，因此不再为小幅吞吐继续挤压约 10GiB headroom。
- step17 checkpoint 的 Writer、optimizer/scheduler 与 8-rank RNG 共 10 个文件均独立通过 size/SHA；从它恢复至 step35 后 loss、吞吐和显存连续稳定。35 steps 恰好为 4 个完整 70-task cycles，每 task 1536 queries、50/50 episodes 覆盖，consumed identity SHA256 为 `59804c03...6db01`，step35 manifest SHA256 为 `835f9758...ec15`。
- 正式 cold-start 已封存为 1575 steps、525/1050/1575 thirds；按 profile 预计纯训练 89.93 分钟。profile 只证明 mechanics 和资源合同，不能作为 Writer 行为结论。
- fresh evaluator 已能从 checkpoint 与 validation cache 生成、注入同一 37-target LoRA。profile step35 的真实 smoke 中，8 ranks 对 task0 生成的 adapter SHA256 均为 `067780eb...aa421`，8 个固定 state 各一条、设备/EGL 映射完整并 exit 0；小分母 `6/8` 明确不作性能证据。

### Writer functional cold-start：formal seed 1 完成

- commit `69bbdee` 的正式 trajectory 完成 1575/1575 steps、exit 0，约 92.9 分钟；累计 4,838,400 global queries。最终 70 个 train tasks 各精确消费 69,120 queries，并各覆盖全部 50 episodes。
- 525/1050/1575 三个 checkpoint 均在完整 70-task cycle 边界保存；最终 checkpoint 的 10 个 Writer、trainer 与 rank-RNG 文件共 150,436,331 bytes，逐文件 size/SHA 校验通过，manifest 为 `c30c49af...3357`，consumed identity 为 `2029f311...4112`。
- 全程每卡一个 Writer CUDA rank，GPU0 无额外模型/controller context；最终 peak allocated/reserved 为 68.00/70.71GiB，最后一步吞吐 916.1 queries/s。训练 loss 与机械完成本身不作为 Writer 功能价值结论，行为结论只取预封存 validation rows。

### Writer cold-start：首个 validation policy RNG 选择 step1050，但增益仍边缘

- frozen source base 为 `56/500`；Writer step525/1050/1575 分别为 `58/500, 63/500, 60/500`。全部结果各含 500 个唯一 `(task_id, init_state_id)` rows，环境与 policy seed 逐 row 匹配 base，每个 task 的 adapter hash 在 8 ranks 间唯一一致。
- 预封存排名选择 step1050。它的 per-task 原始成功数为 `{0:19,8:0,15:0,28:24,40:0,56:1,61:0,71:0,85:0,88:19}`；相对 base `{0:28,8:0,15:0,28:14,40:0,56:1,61:0,71:0,85:0,88:13}` 是 `31 gains / 24 losses / net +7`。
- 正增益落在 KITCHEN-actuation task28 `+10` 与 STUDY-pick-place task88 `+6`，但 KITCHEN-actuation task0 同时 `-9`，aggregate 只增加 1.4pp。因此这足以确定后续 cold initialization，不足以声称 Writer 已明显跨类别优于 base。
- selection 与所有 artifact hashes 已封存在 `configs/writer_cold_start_selected_v1.json`。根据既有“policy sampling 方差可能左右判断时加第二 RNG”规则，已在查看新 outcome 前封存 `configs/source_base_eval_rng2_v1.json`；它只比较 base 与已选 step1050，不能重新选择 checkpoint，test 仍未打开。

### Writer cold-start：第二 policy RNG 复现两类功能信号，同时确认覆盖有限

- RNG2 的 frozen base/selected Writer 为 `51/500, 57/500`，配对 `30 gains / 24 losses / net +6`；RNG1 对应为 `56/500, 63/500` 与 `31/24/+7`。两组使用同一 50 fixed states 和 env seeds，policy seeds 完全不相交，selected checkpoint 没有重选。
- 两 RNG 合并后 base/Writer 为 `107/1000, 120/1000`。task28 为 `26/100 → 48/100`，配对 `32 gains / 10 losses / net +22`；task88 为 `24/100 → 36/100`，配对 `18/6/+12`。它们分别属于 KITCHEN-actuation 与 STUDY-pick-place，且增益方向在两个 policy RNG 中逐次复现。
- 同时，task0 从 `55/100` 降到 `35/100`，配对净 `-20`；task85 净 `-1`，其余六个任务净零且多数双方均无成功。故可支持“Writer 在两个不同未见类别产生真实即时功能价值”，不能支持“已广泛泛化”。aggregate 只增加 1.3pp，Phase D/后续 matched RL 需要原样报告这项局限。
- 完整 RNG2 合同、result hashes、逐任务配对数与解释封存在 `configs/writer_cold_start_rng2_confirmation_v1.json`；test 从未打开。

### Validation direct-LoRA：formal oracle 完成，LoRA acquisition ceiling 明确非零

- 8 个 validation tasks 各由一个独立 GPU rank 从同一 frozen source base 训练自己的 37-target LoRA；每卡恰好一个 CUDA process，batch384 的 peak allocated/reserved 为 67.09/69.09GiB，实测最慢 step 2.816 秒。
- 每 task 在 step1 保存后由新进程 exact-resume 到 step10；两个边界的 16 个 checkpoint manifests 均逐文件验证 LoRA、trainer 与 RNG state。step1 已覆盖全部 50 teacher episodes，step10 每 task 消费 3,840 queries。
- profile 只验证 mechanics、恢复和资源合同，不看小步性能。formal 每 task 消费 69,120 matched queries，即 batch384 × 180 steps，checkpoints 60/120/180；10 个 validation task 都固定使用 final step180，不按 policy outcome 选择。正式训练约 17.5 分钟，30 个 checkpoint manifests/files 全部通过哈希与 episode coverage 审计。
- 同一 500 条 validation rows 上，frozen base / cold Writer / direct oracle 分别为 `56/500, 63/500, 186/500`。direct per-task 为 `{0:48,8:1,15:17,28:36,40:21,56:11,61:9,71:2,85:11,88:30}`，相对 base 配对 `141 gains / 11 losses / net +130`，相对 cold Writer 为 `138/15/+123`；它在 10 个 task 上都取得正的 raw count gain。
- 因而当前 37-target LoRA 空间并非整体无效，且 target-action acquisition ceiling 充足；cold Writer 与 oracle 的主要差距是跨任务 acquisition/generalization coverage。direct 使用目标 action，只是 oracle/reference，不属于同信息墙主结论，也不驱动 Writer checkpoint 或 test 选择。合同与结果 hashes 封存在 `configs/direct_lora_validation_reference_v1.json`，test 未打开。

### Writer-only RL：formal 完成，但 source reward 使 validation 单调退化

- cold step1050 起点从 update1 checkpoint 由全新 8-rank 进程恢复到 update9，完整覆盖 70 source tasks。每 task 恰好 4 个官方随机 reset rollouts，共 280 interactions、87 successes、90,391 env steps 和 9 个 Writer optimizer updates；生成 LoRA 没有原位更新。
- 72 个 rank/update ledgers 全部声明 `official_random_reset=true`、`fixed_init_state_id=null`；70 个 active task ledgers 合计 280 个唯一 `(task, env_seed, policy_seed)` rows。update1/update9 checkpoints 各含 Writer、trainer 和 8-rank RNG 共 10 个文件，逐文件验证通过，最终 interaction cursor 精确为一个 full cycle。
- max-rank cycle wall 为 405.50 秒，最慢 update 49.73 秒；reward updates 的 peak reserved 5.04GiB。该阶段是 rollout/CPU 受限，增加 dummy 或无科学作用的 batch 只会浪费时间，因此保留一 GPU 一 policy rank、以有效 interactions/秒为准。
- formal 完成 12 个 full cycles：108 declared updates、107 个有成功信号的 optimizer updates、每 task 48 rollouts、总 3,360 source interactions、679 successes、1,176,874 env steps；max-rank wall 4,862.10 秒。864 个 ledgers、3,360 个唯一 seed rows 和 36/72/108 三个 10-file checkpoints 全部通过 cursor/hash 审计。
- 相同 500 条 validation rows 上，cold step1050、update36、update72、update108 依次为 `63,56,36,15` successes；相对 frozen base 的 paired net 依次为 `+7,0,-20,-41`。逐任务原始数已封存在 `configs/writer_only_rl_selected_v1.json`，预封存排序明确选择 cold step1050。
- 因而本轮 Writer-only RL 是真实完成但未带来 held 泛化收益的负结果；source binary-success self-imitation 随交互增加破坏了 cold Writer 的窄 validation 效用。它不被解释成工程失败，也不通过改算法、加 RNG 或重选 checkpoint 来追求正结果；Phase E 使用未经过 Writer-RL 的 cold step1050。
- 首次 validation launcher 因漏传 sealed `--writer-rl-config` 被 canonical evaluator 在 rollout 前拒绝；失败 packet 保留。重试只补齐该 authority 参数，未改 evaluator、checkpoint、rows 或选择规则，三个候选均 exit 0。

### Task-local RL：Writer initialization 赢得 matched fresh evaluation，但覆盖集中

- 4 tasks × 2 arms 恰好映射到 8 卡，每卡一个 CUDA policy process；update1 后由全新进程 exact-resume 到 update3。共 24 ledgers、96 trajectories、16 checkpoints，所有 `fixed_init_state_id=null`，两臂的 task/env/policy seed block 逐项一致。
- 24 个 `task_local_reward_update.step_seconds` 的线性插值 p90 为 `49.8926s`。按读取 reward outcome 前已封存的纯吞吐规则选择 formal `U=18`、每 task/arm `K=72`、checkpoints `6/12/18`；20 单元共 1,440 interactions，含 180 秒开销的投影总 wall 为 2,874.20 秒。
- formal 实际 wall 1,982 秒；360 ledgers、1,440 trajectories、720 个唯一 matched seed rows、60 个 checkpoint manifests/files 全部通过审计。所有 adaptation 与 selection rollouts 均 official random reset、`fixed_init_state_id=null`；fixed 50 states 只在训练结束后的 fresh evaluator 使用。
- adaptation reward identity/Writer 为 `89/720`、`110/720`，AUC 为 `0.1236/0.1528`。成功主要来自 task0/task28/task88，六个任务两臂都没有 reward signal；这已提示 coverage 有限。
- 同一 500 条 fresh validation rows 上，base/cold/identity-RL/Writer-RL/direct 为 `56/63/54/74/186`。identity-RL 对 base 为 `11 gains / 13 losses / net -2`；Writer-RL 对自身 cold J0 为 `37/26/+11`；Writer-RL 对 matched identity-RL 为 `43/23/+20`，exact paired-binomial two-sided `p=0.0187`。
- Writer-RL per-task 为 `{0:30,8:0,15:0,28:10,40:0,56:1,61:0,71:0,85:0,88:33}`；相对 identity-RL 的 raw delta 为 task0 `+3`、task88 `+20`、task28 `-3`，其余为零。相对 cold 则 task0 `+11`、task88 `+14`、task28 `-14`。因此可支持 Writer initialization 在相同 K72 ordinary RL 下带来真实终点优势，但优势主要由 task88 驱动，不能宣称 reward adaptation 已广泛覆盖未见任务。
- 不追加第二 policy RNG：matched +20 的方向同时出现在 reward AUC 与 fresh evaluation，足以支持上述有限结论；第二 RNG 可能改变小的 task0/28 波动，却不会把它升级为广泛覆盖。完整原始 counts、hashes、J0/JK、curve/AUC/time-to-threshold 与 selection 封存在 `configs/task_local_lora_rl_validation_v1.json`，test 未打开。

### Gate -1：通过但带残差

- 初始 action-hidden-video probe 未达到预声明标准。
- 有界 temporal representation recovery 得到 ordered balanced accuracy `19/24 = 0.7917`，same-scene wrong-video 同为 `19/24`，bidirectional paired both-correct `15/24`。
- ordered 明显优于 static/reversed/shuffled controls，说明视频中存在有用时序任务信息。
- 原 `0.80` 内容阈值、paired 不足和 drop-last sensitivity 未被改写。
- owner 接受它作为当前阶段“通过但带残差”，不再烧算力凑 0.80。

完整历史报告在父提交 `999df28:docs/benchmark_validity_report.md`。

### Gate 0：通过但覆盖有限

历史正式 n=32 h16 packet 中：

- task 3：base `22/32`，action-supervised LoRA `28/32`；
- task 4：base `16/32`，action-supervised LoRA `20/32`。

两个任务点估计均为正，分别 +18.75pp 和 +12.5pp，但任务近似、每臂只有 32 episodes，区间较宽。最新 owner 定义下，它足以说明一个成熟 task-local LoRA 空间可以获得有用行为更新；它不再要求 LoRA 在一个已经 source-trained 的 base 上继续跨过人为门槛。

Gate 0 不证明：

- Writer 有效；
- 跨类别普适性；
- ordinary task-local RL 有效；
- 旧 source base 是新协议应使用的起点。

## 旧 Writer 证据

### Source utility 确实出现过

旧 foundation/full-video Writer 在 16 个旧 source tasks、h50、每 task/arm 32 episodes 上：

| 方法 | 成功 |
| --- | ---: |
| generic foundation base | 0/512 |
| frozen Writer LoRA | 55/512 |
| action-supervised direct LoRA | 51/512 |

Writer/base paired gain 为 +10.74pp，说明当前 full-video hypernetwork 结构能够从 source language/video 学到真实闭环行为；它不是只有离线 loss 的空壳。

### Validation transfer 很弱且集中

旧五类 validation comparison：

| 方法 | 成功 |
| --- | ---: |
| foundation base | 0/160 |
| Writer | 1/160 |
| validation-action-supervised direct LoRA | 18/160 |

额外八任务 frozen Writer 为 `4/256`，四次成功全部在 task 22。task 22 的同口径三臂是 base `0/32`、Writer `4/32`、direct LoRA `12/32`。

这说明旧 Writer 有零星未见任务泛化，但远未达到稳健跨类别泛化。

### 不能简单归因于“只是不泛化”

另一组旧 source-trained-base source localization 在五个旧 source tasks、h16 上得到：

| 方法 | 成功 |
| --- | ---: |
| source-trained base | 141/320 |
| Writer | 127/320 |
| direct source LoRA | 137/320 |

旧 Writer 和 direct LoRA 在这个较强 base/recipe 下都未形成稳定 aggregate gain。因此旧实验混合了：

- base 定义不同；
- source split 不同；
- LoRA teacher recipe 不稳定；
- Writer acquisition/normalization 不充分；
- validation generalization 弱。

新 70/10/10 计划必须用统一 source embodiment base 和全 50 episodes 重训，不能只引用对自己有利的一组旧结果。

## 评估系统发现

- 早期 GPU0 进程/UTL 偏高的主因是 MuJoCo/EGL workers 未绑定各 rank 的物理 GPU，以及某些 direct fit 与 eval 同时被调度到 GPU0。
- 修复后每 GPU 只有一个 policy CUDA model；resource tracker、forkserver 和 env worker 是 CPU/仿真进程，不应创建额外 CUDA context。
- SmolVLA h50 一次 policy inference 后可执行至多 50 个 simulator actions，因此闭环评估本质上常由 CPU/MuJoCo 限速，GPU UTL 低且脉冲化是正常现象。
- persistent env pools、NUMA/EGL binding、只渲染少数视频分别带来小幅但真实提速。下一实现应保留这些原则，不恢复旧 runner。
- 训练阶段可以通过真实 batch 把显存提高到约 70GB/card；评估阶段不应为了“看起来满”分配无用显存。

## Source base 为什么曾经表现差

通用 `smolvla_base` 是预训练 VLA，但没有保证掌握 LIBERO-90 的具体 Panda embodiment、camera/controller、scene/object 和 action normalization。预训练提供视觉语言和部分动作先验，不等于对每个 LIBERO task 有高成功率。

旧对话中“foundation base”与“source-trained base”曾混用：

- foundation base 未在当时 source tasks 上 action-train，很多任务接近 0；
- source-trained base 已在旧 source tasks 上全 action expert/projection 训练，可在相似 source tasks 上很强。

新计划通过明确的 70×50 source embodiment base 消除这个混淆。

## Writer architecture 已成立的机械事实

- `VariableEpisodeTaskEncoder` 接受任意有限正数、任意正长度 episode；帧不被固定为三帧。
- full video 通过 chunk temporal attention、episode memory 和 task-level set attention 聚合。
- 语言用完整 token embeddings；视频保留 episode boundaries。
- `CompleteLoRAWriter` 使用 module/factor/rank-aware queries 和 width-typed heads 输出每个 LoRA A/B tensor。
- 冻结 SmolVLA/VLM features 可以缓存；Writer 自身 encoder/fusion/decoder 仍可训练。

历史训练 checkpoint 不可续用，因为新协议将改变 split、source base、全 50 episode 输入、normalization 和数据 authority。

## 数据与 benchmark 事实

- LIBERO-90 正好提供 90 个大规模 task 数据文件，每 task 50 条成功 teacher demonstrations。
- LIBERO-10/Spatial/Object/Goal/Long 是另外的标准 benchmark suites；不应和 LIBERO-90 task IDs 当作同分布池混划。
- 旧 60/15/15 specification-only parser 证明 role-aware factorization 可行，但旧 split 已因最新 owner 设计改为待生成的同分布 70/10/10。

## 不允许从历史证据推出的结论

- 不能声称 EMBER 已有稳健 validation 泛化。
- 不能声称 Writer-only RL 改善了 validation；本轮已验证的是其完整负结果。也不能把 task-local RL 的 task88 集中增益说成广泛覆盖，不能声称 identity-init ordinary RL 改善 aggregate，或声称 outer learning 已验证。
- 不能声称 direct LoRA 是信息匹配的 held baseline。
- 不能声称 h16 是标准 SmolVLA/LIBERO 主评估。
- 不能把旧 source task 结果、旧 validation 结果或 task 22 用于新 split。
- 不能重新引入 bank/geometry 作为“修复”。

## 当前 RL 实现边界

- Writer-only RL 与 task-local RL 都采用 binary-success on-policy
  success-weighted flow regression；这是为 SmolVLA 没有可直接使用的 exact action
  likelihood 而选的 ordinary reward adaptation，不含 critic、teacher action、外部
  exploration adapter 或伪 PPO。
- task-local 两臂只允许同一 37-target LoRA 可训练；base、Writer、encoders 和所有
  shared state 冻结。其配对 seed schedule 明确排除 arm，官方随机 reset rollout
  ledger 与 worker RNG/interaction cursor 进入 checkpoint；`.pruned_init` 只由独立
  fresh evaluator 使用。
- Writer-only RL 已有完整 source formal 与 4 候选 validation 的真实负结果；task-local RL 已完成 matched formal、预算内选择和独立 fresh validation，支持 Writer-init 相对 identity-init 的有限终点优势。

## 证据定位

外部结果根不写入公开仓库；使用本地 `EMBER_OUTPUT_ROOT` 查找。关键历史目录名：

- `gate_minus1/specification/video_information_recovery1_*`
- `gate_zero/task_local_lora_rl_formal_development/formal_n32_recovery3_*`
- `foundation_source_screen/source_three_arm_eval_v2_*`
- `foundation_source_screen/validation_three_arm_*`
- `foundation_source_screen/additional_val_writer_probe_*`
- `foundation_source_screen/task22_base_direct_*`

精确旧命令/config/code 由 Git commit `999df28` 保存。不要把这些目录复制回仓库。

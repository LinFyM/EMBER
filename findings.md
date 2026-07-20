# EMBER Findings

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

### Source-base 正式训练：完成，闭环行为结论待 source-development 评估

- 从 pinned `lerobot/smolvla_base` 严格加载 450,046,176 parameters；98,880,992 个 action-expert/projection parameters 可训练，冻结 VLM trainable 泄漏为零。
- 全部 70×50 episodes 对应 537,946 frames；sampler 在跨 rank global task slots 上做 deterministic no-replacement cycles，并保证每个 checkpoint 边界前 3500 episodes 全覆盖。
- 显式 rank-local device 修复后的 8-A100 profile 使用一张卡一个 rank、batch/rank 352：稳态 2.590s/step、1087.2 global samples/s，每卡峰值 allocated/reserved 65.05/67.35GiB，data wait 0.13ms。
- 首次 formal 启动暴露出无索引 `device="cuda"` 会让非零 ranks 在 GPU0 留额外构造 context；它在首个 checkpoint 前被停止且不复用。改为 `cuda:{local_rank}` 后，满 batch steady-step 进程表为每卡恰好一个 CUDA PID、69,124–69,132MiB，GPU0 不再额外堆积。
- 8-rank continuous/resume 对照中，step-1 policy、optimizer、scheduler 和每 rank RNG 起点一致；启用 DDP static graph 后，step-2 policy 文件 SHA256 位级一致，optimizer/scheduler/RNG 逐值一致。默认 DDP 首轮后 bucket 重建是此前微小漂移的工程原因。
- commit `72eb10d` 上的正式 seed-1 trajectory 在约 28 分钟内完成 630/630 steps、退出码 0；210/420/630 三个 checkpoints 均通过 15-file size/SHA manifest 校验。最终 step loss 为 `0.483089`，吞吐 `1084.95 samples/s`，峰值 allocated/reserved 为 `65.05/67.35GiB`。
- step 630 累计 1,774,080 global examples 和 5,040 global task slots；每个 checkpoint 边界均覆盖全部 70 tasks 和每 task 50 episodes。最终 policy SHA256 为 `eb7e01f2...c1f159f`，checkpoint manifest SHA256 为 `89e9f493...ed22c`，launch contract SHA256 为 `22c4ffb5...2e8`。
- source-only loss 三段均值为 `0.80098 → 0.53083 → 0.49775`。同一 8-task × 50-state source-development panel 的 step 210/420/630 h50 成功数为 `3/400 → 8/400 → 15/400`；420→630 是 11 paired gains、4 paired losses，行为改善不只来自总数拼接。
- step-630 per-task 原始成功数为 `{1:0, 2:3, 6:6, 16:4, 46:1, 63:1, 65:0, 73:0}`，说明它在 5/8 个预声明 source tasks 出现 competence，但 3.75% 绝对成功率仍低，不能直接冻结。
- step 210/420/630 `results.json` SHA256 分别为 `b1445ec2...b3893`、`ba8fdbb5...e8cdb`、`b901f758...5ffa4`；每份均为 400 个唯一 `(task_id, init_state_id)` rows，8 ranks、50 states/task、horizon 400，无跨 checkpoint 拼接。
- 只据上述 train/source evidence，封存一次 315-step continuation：从 step 630 完整恢复 optimizer、sampler/RNG 与 interaction-free data cursor，原 cosine scheduler 保持在 decay LR `2.5e-6`，相对 thirds 为 735/840/945。没有读取 validation/test outcome，也不重启高 LR。

### 新 h50 fresh evaluator：mechanics 通过，未打开 test

- evaluator 只暴露 specification-only 预声明的 8 个 train/source-development tasks 和 10 个 validation tasks；reporting-only test role 在 Phase F 前结构性不可解析。
- 使用官方 LIBERO suite/BDDL/controller/camera/normalization、固定 `.pruned_init` states、dummy settling 10、horizon 400、成功即终止和 SmolVLA h50；固定 states 只服务 fresh evaluation，不会进入 RL update 或 adaptation checkpoint selection。
- step-630 mechanics smoke 在 8 ranks 上各跑 1 个不同固定 state，共 8 条唯一 rows；运行时每卡恰好一个 policy CUDA process、显存一致为 3347MiB，退出后全部归零。`0/8` 只是 smoke 小分母，不作性能证据。
- 完整 source-development/validation 评估按 task 同步、state rank-strided、每 rank 4 个持久 async env workers；这使八卡 policy 进程拓扑完全对称，同时把 MuJoCo rollout 吞吐作为优先优化对象。

### Frozen Writer feature cache：真实 smoke 与 resume 通过

- 只读取每条 source teacher episode 的 `obs/agentview_rgb` 和 language；不读取 action、proprio、reward、terminal、task ID/file-name features。每帧按 source-base 相同 OpenGL transform 进入 frozen SmolVLA VLM，64 个 960-d 空间 tokens 经固定 `sqrt(960)` normalization 后确定性均值池化为一个 960-d BF16 frame feature。
- 8-rank smoke 为每 rank 1 个不同 train task/episode，共 1,194 frames；所有视觉/语言 features finite，episode offsets 与原 episode lengths 一致。单 task 108–197 frames 的提取 wall time 为 0.63–0.83 秒，按正式 LPT 调度估计全 537,946 frames 约 5 分钟。
- resume 再运行时 8/8 ranks 均验证既有文件 size/SHA 后 `new=0`；模型加载阶段每卡恰好 1 CUDA PID、414MiB，GPU0 无额外 context，退出后 8 卡全清。正式 cache 尚未生成，smoke artifact 不作为训练数据。

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
- 不能声称 Writer-only RL、task-local RL 或 outer learning 已验证。
- 不能声称 direct LoRA 是信息匹配的 held baseline。
- 不能声称 h16 是标准 SmolVLA/LIBERO 主评估。
- 不能把旧 source task 结果、旧 validation 结果或 task 22 用于新 split。
- 不能重新引入 bank/geometry 作为“修复”。

## 证据定位

外部结果根不写入公开仓库；使用本地 `EMBER_OUTPUT_ROOT` 查找。关键历史目录名：

- `gate_minus1/specification/video_information_recovery1_*`
- `gate_zero/task_local_lora_rl_formal_development/formal_n32_recovery3_*`
- `foundation_source_screen/source_three_arm_eval_v2_*`
- `foundation_source_screen/validation_three_arm_*`
- `foundation_source_screen/additional_val_writer_probe_*`
- `foundation_source_screen/task22_base_direct_*`

精确旧命令/config/code 由 Git commit `999df28` 保存。不要把这些目录复制回仓库。

# EMBER Findings

## 2026-07-21 当前 π0.5 协议与已验证事实

- 活动目标split为四个标准LIBERO suites、每suite 6 train / 2 validation / 2 test，总计24/8/8；final合并为32 source / 8 test。
- generic `lerobot/pi05_base` revision `7de663972b7817d2c4cf2d84c821153dfea772e9` 已完整下载；weights SHA256 `0eb11ca9587678c1d2ef8cf32807c29f8ce53a2bfdfc1aa4a4c96f16fca59b0f`。
- generic base在8 test tasks×50 official fixed states上全部为0/50，总计0/400；400个`(suite, task, init_state)`唯一且全部到suite horizon。该结果只说明原始π0.5没有LIBERO执行能力，不评价EMBER。
- result aggregate SHA256为`8ffa816e...7776`；tracked seal为`configs/libero_24_8_8_v1/pi05_base_feasibility_results.json`（SHA256 `c78e92e9...20c2`）。
- 24-task interface normalization和公开OpenPI tokenizer已完成并核验，但它们只属于generic feasibility合同；新source base将从过滤后的LIBERO-90 source corpus重新计算并冻结自己的source-only stats。
- batch1/8/16 profile分别约27.52、19.76、19.58秒/episode，8→16只快约0.9%。正式静态task/GPU运行中Spatial约1004秒，而horizon-520 task最长约2169秒，证明后续必须优化cost-balanced sharding而不是只加batch。
- EGL rank映射错误已在commit `bf27ebc`修复；正式8卡运行每卡一个CUDA process且全部exit0。

## 2026-07-21 LIBERO-90 source overlap seal

- 完成90 source×40 target共3600对full-task specification audit。只读language、BDDL、objects、roles、initial predicates和ordered composition；没有读取任何numeric action/state、reward、terminal、normalization或policy outcome。
- 排除19个完整semantic/composition等价source IDs：`8,9,10,20,25,27,30,31,44,46,47,48,49,50,51,52,53,54,77`。除已知44/77外，audit捕获Goal与Object中的别名和不同scene复现；过滤规则与逐pair理由已封存。
- 保留71 tasks；primitive containment、不同object multiplicity、source/destination selector或额外ordered subgoal不算full-task overlap。IDs `2,29,12,13,14,15,38`是人工复核并保留的关键near misses。
- 71×50 successful episodes全部存在，共3550 episodes、529,173 frames、52,710,755,898 bytes，aggregate HDF5 SHA256 `81bdb358...a1a50e`；source-only normalization读取这些source numeric rows且validation/test numeric reads为0。
- seal hashes为audit `fe731127...cc003`、manifest `75453a20...2e54`、normalization `e259ee6e...f7c4`和recipe `4c537067...281734`。recipe hash更新只封存pinned OpenPI缺失右腕的zero-image/false-mask相机合同，不改变source IDs或任何outcome；后续outcome不得反向修改source IDs。

## 2026-07-21 π0.5 source-base recipe、profile与resume

- 官方full-SFT anchor来自pinned OpenPI/LeRobot：global batch256、30k steps、AdamW betas `(0.9,0.95)`、eps `1e-8`、weight decay `1e-10`、clip1、peak LR `5e-5`、10k warmup后constant、EMA `0.999`。EMBER source base采用full-SFT，不使用`pi05_libero`，不叠未merge shared adapter。
- 修正后的canonical 8×A100 m32+EMA smoke使用pinned OpenPI的`q99-q01+1e-6` quantile公式，并省略右腕feature key，使LeRobot产生zero padding且`image_mask=false`。3/3 steps的loss/gradient finite，steps2–3平均47.75 global examples/s；峰值allocated/reserved为67,178,351,616/71,179,436,032 bytes，约保留10.7GB稳定余量，故锁定global batch256且不做gradient accumulation。旧m1/m4/m16/m32对比profile把显式zero右腕误标为`mask=true`，只保留工程provenance，不再作活动launch证据。
- 第一次formal启动的CUDA进程拓扑正确，但live `PSR`显示rank未受GPU-local NUMA约束；在step12、首个checkpoint前主动终止，exit130。该root仅保留20KB failure evidence，run contract/metrics/log hashes分别为`997af43a...8b2`、`81dbfcbc...4ca`、`7a169300...118`，不得resume或作科学结果。
- 修复后每rank在CUDA初始化后立即绑定sysfs GPU NUMA cpulist，DataLoader children继承：rank0–3为`0-37,76-113`/NUMA0，rank4–7为`38-75,114-151`/NUMA1。相机mask修正后的3-step m32+EMA smoke exit0，contract/metrics/summary/log SHA256分别为`90fbe1da...0458`、`de2d9889...50d9`、`0a590a29...e1bc`、`26bb5aad...c10`。
- formal attempt2在step316、80,896 global examples处被主动终止：根因是训练与评测都显式传入zero右腕，LeRobot因此把第三相机标为`mask=true`，而pinned OpenPI LIBERO policy要求zero image加`mask=false`。该run无checkpoint、不得resume或作科学结果；failure packet/run-contract/metrics/log SHA256为`2d2a9e40...9b80`、`e79e1c84...e7d8`、`fb0b2edc...f918`、`3f0eb65f...76f7`。修复是省略feature key而不是另建相机路径，训练和评测共用这一唯一合同。
- canonical runner严格加载weights，避免上游异常时静默返回随机模型；每个checkpoint封存policy、EMA、optimizer、scheduler、8个rank RNG/sampler states、metrics cursor、contract和文件hash，并在新checkpoint原子发布后才清理旧状态。
- 两个独立8-rank进程均从同一step1 manifest `0461dee1...5953`恢复；step2 loss、grad norm、LR、cursor和8个rank state文件完全一致。4,143,404,816个policy元素中0.0308%仅有独立NCCL启动末位差，max `1.49e-8`；EMA max `3.73e-9`。这支持state/cursor exact且numerically reproducible的resume合同，不虚假宣称跨新distributed process bitwise identical。
- 三套约32GB probe checkpoint在compact evidence封存后按500GB cap永久清理；保留evidence packet为444KB，comparison SHA256 `16137fa1...b1e`。清理后个人占用379,942,686,720 bytes，atomic双checkpoint峰值估计约447.62GB。
- 2026-07-22 owner将训练预算口径明确为短周期、证据驱动：先profile学习速度/吞吐，按曲线斜率安排廉价固定screen，只给少量候选完整validation，接近饱和即停。约120分钟只是所有适用训练阶段的防失控上限；到上限仍未充分训练则封存曲线与budget-censored判断，不自动追加。task-local RL按每个初始化方法在全部8个test tasks上的合计训练时间计费。

## 2026-07-21 canonical π0.5 target evaluator

- 唯一活动入口为`scripts/evaluate_pi05.py`；旧静态`scripts/evaluate_pi05_base.py`已退役，不从Git历史恢复。`pi05_eval_contract.py`拥有sealed authorities、source final-EMA门与seed schedule，`pi05_eval_queue.py`拥有cost-balanced SQLite队列，`pi05_evaluation.py`拥有persistent policy/env与rollout，`pi05_eval_results.py`拥有worker证据与strict aggregate；这些是单一runner内的故障边界，不是并行实现。
- 40-task screen按`states × suite horizon`切成近等cost shards，8卡使用相同1/2/3 replicas并动态claim/work-steal；每worker持久加载一套policy和当前task env pool，GPU0没有额外CUDA controller。
- formal/screen只接受与当前source config、全部model/tokenizer/recipe authorities完全一致的final step1000 EMA；test `.pruned_init`逐项对sealed protocol hash。worker在load前重算model/tokenizer SHA，raw shard、DB counts与producer/claim均交叉核对。
- launcher在任何queue recovery前独占lock；局部spawn失败只终止本launcher创建的workers并保存PID、logs、failed jobs与hash。吞吐主指标从首worker进程spawn到全体退出，包含model load与首次env/EGL创建；另报shard-only window，避免1/2/3 replicas profile偏置。

下文全部SmolVLA/70-10-10证据仍是真实历史，但只作provenance，不能驱动当前π0.5训练或复用旧checkpoint/normalization/runner。

---

本文件只保留会影响当前科学解释的证据。父提交 `999df28` 保存 2026-07-17 至 2026-07-20 的完整逐次日志、旧配置、runner 和测试；外部 checksummed artifacts 保存原始 rows、metrics、视频和 failure packets。这里不把历史过程重新伪装成活动合同。

## 历史 70/10/10 结论（已退役）

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

当时的 70/10/10 计划要求统一 source embodiment base 和全 50 episodes 重训；该要求已被
当前 π0.5 / 24-8-8 方案替代，但“不能只引用有利旧结果”的证据原则仍保留。

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

当前方案通过明确的“过滤后 LIBERO-90 × 每 task 50 episodes”π0.5 source base 消除这个
混淆；最终 task 数由新的 specification-only overlap audit 决定。

## Writer architecture 已成立的机械事实

- `VariableEpisodeTaskEncoder` 接受任意有限正数、任意正长度 episode；帧不被固定为三帧。
- full video 通过 chunk temporal attention、episode memory 和 task-level set attention 聚合。
- 语言用完整 token embeddings；视频保留 episode boundaries。
- `CompleteLoRAWriter` 使用 module/factor/rank-aware queries 和 width-typed heads 输出每个 LoRA A/B tensor。
- 当前活动`CompleteLoRAWriter`在temporal encoder之前额外要求恰好一个非空episode，故只接受`offsets=[0,L]`；底层多episode能力不构成活动输入权限。
- 历史SmolVLA/VLM feature cache只证明冻结features可缓存；当前π0.5必须重建pure-language、按单demo切片且与action query独立的新cache，旧cache不可复用。

历史训练 checkpoint 不可续用，因为新协议将改变 split、source base、全 50 episode 输入、normalization 和数据 authority。

## π0.5统一LoRA与Writer接口事实（2026-07-21）

- 真实`pi05_base` safetensors metadata与meta-device模型结构独立确认同一拓扑：18层action expert的`q_proj`为1024→2048、`v_proj`为1024→256，另含`action_in_proj` 32→1024和`action_out_proj` 1024→32。
- rank/alpha 16、dropout 0时共38 targets、76个A/B tensors和1,287,168 trainable parameters；合同文件SHA256为`1dcf58f7...cb07`，canonical payload SHA256为`42d5919e...94dd7`。
- PEFT注入后的36个expert adapter tensors为BF16，而action in/out adapter为FP32。Writer必须逐tensor保留template dtype；全部强转FP32会让functional训练与materialized推理走不同数值路径。
- mixed-dtype toy policy已逐值验证Writer-generated functional state与copy到物理adapter后的loss完全一致；B=0 identity仍为确定性物理恒等。
- LeRobot `PI05Policy.forward`只接受`batch,reduction`，不接受旧Smol functional路径的`noise/time`参数。训练随机flow noise/time由PI05内部RNG产生，exact-resume保存并恢复每rank RNG。
- Writer language不得复用带当前normalized state的PI05 action prompt，否则会泄漏proprio；新feature owner必须单独做pure task-language tokenization。policy functional action loss仍使用完整PI05 observation/action processor，且缺失right-wrist key以保持false mask。
- `lora.py`只拥有跨backbone PEFT注入、identity、state validation/hash与functional-call机械接口；`pi05_lora.py`是唯一活动科学拓扑authority。旧Smol contract/import只作历史模块provenance，不得进入π0.5 runner。

## Target40数据墙与π0.5 AS-Writer机械合同（2026-07-21）

- immutable Hub revision `f13aa24...e35a` 的四个标准suites共40个HDF5已本地齐备：2,000 episodes、338,575 frames、33,784,856,577 bytes。`configs/pi05_target_data_v1/manifest.json` SHA256为`1b28547f...049d`，40个本地文件均与Hub LFS SHA一致，HDF5 identity aggregate为`6342f5d9...78a6`。
- target封存只读取task specification、Hub metadata、HDF5 schema/shape metadata与opaque file bytes用于SHA；没有解码trajectory/video值，action/state/reward/terminal/video value reads均为0。manifest中的24/8/8 global IDs逐项等于既有protocol，policy outcome reads与task-selection changes均为0。
- development feature cache只授权24 train + 8 validation的action-hidden agentview视频；pure-language prompt固定为`Task: {cleaned}\n`，target40实测最长23 tokens，小于sealed max64。PI05投影后每帧256×2048 tokens只做spatial mean，不沿用SmolVLA的`sqrt(dim)`缩放；缓存BF16并保留50条episode边界。
- AS action dataset从同一32-task manifest显式筛到恰好24 train tasks；validation/test actions永不进入dataset。policy functional loss仍使用冻结LIBERO-90 source normalization，Writer路由所需task/demo identity不会作为tensor输入Writer。
- action query与teacher video分别由不同seed的deterministic no-replacement schedule产生；每个rank/step只取同task的一条video并传`offsets=[0,L]`。checkpoint保存两套schedule identity、全部rank RNG、optimizer/scheduler、metrics cursor，并先验证canonical manifest及每文件SHA再读取pickle。
- feature cache formal配置SHA256为`3e3a8ea7...429e`、AS-Writer配置SHA256为`971cac43...f807`；两者分别保持`pending_source_base`/`pending_profile`。这只是机械authority，不是训练或性能结果。
- 当前架构owner为：`pi05_target_data.py`负责held-data seal；`feature_cache.py`负责PI05 cache schema/tensor store；`cache_pi05_writer_features.py`负责唯一8-rank extraction；`as_contract.py`负责24-task action wall与source/cache/hash联锁；`training.py`只负责AS模型与step loop；`checkpoint.py`负责atomic exact-resume。旧`cache_writer_features.py`与`train_writer_cold_start.py`已删除，剩余历史Smol推理/训练入口因新schema fail-close，待对应PI05 owner具备功能对等后删除。

## canonical evaluator中的one-video Writer证据（2026-07-21）

- `writer/inference.py`已原位替换旧Smol/cold-start实现，只接受`ember_pi05_as_writer_launch_v1`、PI05 38-target LoRA、同一final raw source policy及与训练逐字段相同的formal feature cache；checkpoint先核验canonical manifest和全部文件SHA，legacy schema不再有活动分支。
- 每个rollout的视频selection seed为`sha256([namespace, seed, eval suite, eval task, init state])`前63 bits，demo为`seed mod 50`。该纯函数不依赖worker、shard、重试、queue顺序或outcome；correct/wrong不把arm或video task写入seed，因此使用完全相同的demo ordinal。
- wrong map在每个split role内按Spatial→Object→Goal→Long→Spatial循环，并按该role中排序后的task ordinal一一映射；它是跨suite双射且role-preserving。run contract保存显式map及SHA，避免final_source的train/validation混排导致越墙。
- materialized backend在episode开始只运行一次Writer并固定完整LoRA；由于并行env可能使用不同LoRA，每个replan前为对应slot重装其state并逐slot推理。policy noise仍由原有`(task,init,replan)`schedule生成，correct/wrong不变。
- raw row保存Writer checkpoint step/manifest、Writer state、LoRA contract、逐rollout LoRA、video selection seed/suite/task/role/demo、map SHA、teacher-video pairing SHA和generation timing；run contract另保存排除condition/map但覆盖source、tasks/states、env/policy RNG、topology、Writer/cache的`paired_control_sha256`。聚合重新计算schedule/map并报告每task唯一视频数与频数。当前只是机械合同，尚无Writer行为结果。

## π0.5 Source-SFT机械合同（2026-07-21）

- development authority精确选择target manifest中的24个train global task IDs，四suite各6个、共1,200条可用action episodes；validation/test actions与teacher video均不进入训练。当前配置`configs/pi05_source_sft_development_v1.json` SHA256为`32e927c...8a641`，formal仍为`pending_profile`，不能被误当成正式配方。
- 每次fresh stage都从同一final formal raw source policy注入确定性B=0 identity的PI05 38-target LoRA；只有76个LoRA tensors、1,287,168 parameters可训练。八rank各取task-pure batch后由DDP聚合为一套shared multi-task LoRA，不使用functional Writer、per-task adapter或额外shared adapter。
- checkpoint只保存shared `lora.safetensors`，不复制约14.5GB source policy；同时封存optimizer/scheduler、scaler-disabled声明、optimizer/micro cursor、metrics cursor、每rank Python/NumPy/CPU/CUDA RNG、DataLoader seed、deterministic sampler identity与逐task episode coverage。所有文件在读取pickle前先验证bytes/SHA，正式末checkpoint必须覆盖每task全部50 episodes。
- development config只能启动development；validation选择后必须另封final config，32-source formal run从同一确定性identity fresh开始，development checkpoint因stage/config/contract hash不同而无法resume。这避免后续修改配置破坏development provenance。
- evaluator仍只有`evaluate_pi05.py`一条canonical runner。Source-SFT LoRA在每个worker初始化时只安装一次，随后保持原有multi-env batched replan；它不会走Writer的逐rollout生成/重装路径。raw row保存固定`policy_adapter_sha256`，resume重新核验source/run/checkpoint/config/LoRA全部hash；Writer专属`paired_control_sha256`不套用于Source-SFT。
- 新代码按当前故障边界归入`ember.source_sft`子包（contract/data wall、training、checkpoint、inference），`eval_adapters.py`只拥有两类adapter的runtime分派和row证据；这是独立baseline所需owner，不是第二套evaluator。architecture guard无hard violation、无parallel family。旧Smol `direct_lora*`仍只作provenance；待本PI05 owner完成真实8卡finite/exact-resume smoke后，迁移其仍被旧task-local模块使用的通用helper并删除旧direct CLI/module，而非长期保留双活动路径。

## 数据与 benchmark 事实

- LIBERO-90 正好提供 90 个大规模 task 数据文件，每 task 50 条成功 teacher demonstrations。
- LIBERO-10/Spatial/Object/Goal/Long 是另外的标准 benchmark suites；不应和 LIBERO-90 task IDs 当作同分布池混划。
- 旧 60/15/15 specification-only parser 证明 role-aware factorization 可行；其后的
  70/10/10 也已退役。当前目标 split 是已封存的四 suites 24/8/8，并在 final 合并为
  32 source / 8 test。

## 不允许从历史证据推出的结论

- 不能声称 EMBER 已有稳健 validation 泛化。
- 不能声称 Writer-only RL 改善了 validation；本轮已验证的是其完整负结果。也不能把 task-local RL 的 task88 集中增益说成广泛覆盖，不能声称 identity-init ordinary RL 改善 aggregate，或声称 outer learning 已验证。
- 不能声称 direct LoRA 是信息匹配的 held baseline。
- 不能声称 h16 是标准 SmolVLA/LIBERO 主评估。
- 不能把旧 source task 结果、旧 validation 结果或 task 22 用于新 split。
- 不能重新引入 bank/geometry 作为“修复”。

## 历史 SmolVLA RL 实现边界（不约束当前 π0.5 算法）

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

## π0.5 reward adaptation机械事实（2026-07-21）

- π0.5每次规划产生50步chunk，但活动LIBERO runner只执行前5步后重规划。reward replay不能把未执行的后45步计入成功credit；新`Pi05ExecutedPrefixFlowLoss`直接保留逐action-step flow loss并按每个replan实际执行长度mask，successful episodes先各自平均再等权。仅传旧`action_is_pad`字段不足以建立这一合同。
- reward rollout唯一owner使用raw `OffScreenRenderEnv`的`env.seed(seed) -> reset()`，随后10个dummy settling steps；不调用`set_init_state`，不读取`.pruned_init`。环境seed和逐replan flow-noise seed只依赖task/adaptation/rollout cursor，显式排除arm、rank、worker、queue order与outcome。
- RL-Writer从同seed的fresh随机Writer开始，zero输出头使初始generated LoRA在功能上为identity；source policy物理冻结，只有共享Writer经functional LoRA loss更新。全局没有成功episode时optimizer/scheduler cursor不前进；有成功时8 ranks按全局successful-episode数做等权DDP缩放。
- micro-AS分支当前明确fail-close：只有完整zero-warmup run封存为无信号后才允许从同seed fresh Writer做恰好24个action chunks（每development train task一个）的短warm-up；永不载入完整AS-Writer checkpoint。当前没有运行reward实验，也没有正/负科学结果。
- task-local合同严格绑定8个test global IDs `6,8,10,17,24,27,30,33`。同一`(task, adaptation seed)`的AS/RL Writer臂共享一条hash选定video，初始化LoRA只生成一次并固定；identity臂不读取video。rollout、environment-action、optimizer三类cursor和初始化/ledger hashes均进入checkpoint，fixed-50不能选adaptation checkpoint。
- 活动代码所有权：`ember.reward`只拥有共享random-reset/seed/ledger/executed-prefix mechanics；`ember.rl_writer`拥有fresh shared Writer的authority/runtime/update/checkpoint；`ember.task_local`当前拥有test-only unit/initialization/update/checkpoint mechanics。三个旧Smol可执行入口已删除，避免双canonical runner；剩余历史模块/配置只作provenance，等π0.5 task-local runtime与fresh evaluator功能对等后删除，不加兼容分支。
- seen/source panel已在任何新source/Writer/Source-SFT outcome产生前按specification-only规则封存：每suite只在6个development-train tasks内按`SHA256(tag, seed, suite, task_id, language, BDDL SHA)`排序取前2个，得到global IDs `0,2,15,12,21,28,39,37`。该panel只用于source acquisition诊断，不能替代validation/test泛化。
- frozen RL-Writer不需要第二套evaluator：它与AS-Writer共用task/video mapping、逐rollout materialization、每次replan重装同一LoRA及row validation；adapter明确记录`writer_method`与`reward_update` checkpoint axis。RL checkpoint inspector会重算24-task task/video full-cycle coverage并核验source/cache/config/hash，formal状态未seal时不能进入非smoke评测。

## Fresh 1k source base与evaluator吞吐证据（2026-07-22）

- owner把source acquisition明确限定为从generic base fresh训练1,000 optimizer steps；因此旧30k attempt在step2880无checkpoint停止，不能resume或参与比较。新run在8×A100、global batch256、333-step warmup、EMA下完成256,000 examples，训练loop用时91.58分钟。
- 50-step mean loss由0.26213降到0.08659，后半程下降幅度逐渐缩小但并未数学收敛。科学解释只能是“1k预算下获得轻量LIBERO interface acquisition且末段趋平”，不能写成LIBERO-90已过拟合或已完全收敛；到预算仍缓降按全局规则记为budget-censored。
- step1000 checkpoint的完整manifest/hash验证通过，包含raw policy、EMA和1000条唯一finite metrics；冻结后所有下游方法必须共享同一raw source policy与source-only normalization，不能追加source adapter或因target outcome回改source IDs。
- all-40-task×1-state公平panel显示1/2/3 replicas/GPU的有效吞吐为0.1556/0.1818/0.1897 rollout/s。3 replicas比1 replica高约22.0%、比2 replicas高约4.35%，每卡约31GB且GPU0无额外CUDA角色，因此40-task source screen锁定3 replicas/GPU。
- 上述3个40-episode profile均为吞吐smoke且全部0 success，分母太小，不能用来断言source competence；正式行为判断只来自随后预定的40 tasks×8 states screen及其逐taskraw rows。
- 原正式40-task×8-state screen的0/320只验证了滞后EMA，不能解释为source acquisition负结果。参数比较显示EMA只走完raw更新位移的28.62%（action expert 33.52%，action I/O/time/state 40.13%，VLM 26.87%）；匹配source closed-loop为raw 4/4、EMA 0/4，匹配offline flow loss为raw 0.06165、EMA 0.17775、generic 0.29302。
- raw step1000的正式40-task×8-state screen为`46/320 = 14.375%`，成功覆盖13 tasks和全部四个suites：Long 2/80、Goal 28/80、Object 1/80、Spatial 15/80。它满足“多个tasks有部分真实成功、aggregate不由单task支撑”的source acquisition条件；canonical下游使用raw `policy/`，EMA只保留为训练状态和负诊断。
- 正式raw screen绑定commit `cab2edf72a8b7d5173503735ef33bdd8fc4c2a50`、raw weights SHA256 `60ea7ee8...cdf36`、corrected source summary SHA256 `473ae3dc...f874`。320 rows/task-state唯一、24 workers均exit0；results SHA256为`4e2defaf...db3a`，wall-clock 412.372秒。
- development Writer feature cache在同一raw source policy上完成8卡batch32 smoke：8 tasks各1条video、共1,033 frames全部成功，单task 89.77–113.70 frames/s，按并行critical path为689.47 frames/s；输出仅4.92MB且没有OOM/nonfinite。该证据足以直接锁定batch32，不再做无科学收益的batch sweep。
- 随后的formal cache覆盖development train+validation共32 tasks、每task全部50 videos，总计1,600 episodes/274,523 frames；32个task tensors逐文件SHA全部通过，目录1.127GB，launch-to-manifest 248秒。其information wall明确记录test video与trajectory action/state/reward/terminal读取均为0。

## AS-Writer profile结论（2026-07-22）

- AS-Writer的显存主因是对generated LoRA保留functional policy反传图，而不是Writer本体参数量。batch1只分配约12.9GB；batch16分配63.53GB、reserve 68.17GB，并把稳态吞吐从约32提高到约122 global action queries/s，因此batch16是当前实测有效且保留稳定余量的点。
- 极短训练必须保持预期正式scheduler horizon；把`total_steps`也缩成4会令LeRobot按`4/30000`缩放warmup并取整为0，造成首步直接使用peak LR。这是profile协议伪影，不是Writer科学发散。
- 在1,000-step horizon下执行前128步时，首/末16-step mean functional loss为0.14714/0.11930，后64步线性斜率约`-1.58e-4/step`，gradient norm在warmup后降至约0.2–0.3且无nonfinite。曲线尚未饱和，但实测完整1,000步仅约17.5分钟净训练，故选择1,000步并以四分点做稀疏validation候选，不把120分钟guardrail当目标。

## AS-Writer选择、source-base validation与Source-SFT固定预算（2026-07-22）

- AS-Writer正式训练完成1,000 steps、global batch128；cheap screen选择step250与step500进入完整8-task×50 validation。step250为`119/400`、step500为`99/400`，因此development AS-Writer冻结step250。step250逐任务成功为Spatial 1/3=`0/0`、Object 1/3=`40/36`、Goal 3/6=`0/27`、Long 1/2=`16/0`。
- 同一8-task×50 fixed-state配置下，frozen source base为`48/400`：Spatial 1/3=`0/0`、Object 1/3=`5/0`、Goal 3/6=`0/41`、Long 1/2=`2/0`。AS step250 aggregate增加71，但Goal 6从41降到27，说明增益来自任务重分配而非所有任务一致改善。
- owner选择先做不调step的matched-scale Source-SFT。比较基准是被选中的AS step250所消耗的`250 × 128 = 32,000` action queries，而非完整1000-step曲线的累计消耗。Source-SFT保持profile选定的batch64/rank（global512），固定63 steps=`32,256` queries，与32,000相差0.8%；该匹配不要求batch size或optimizer updates相同，也不声称forward compute、参数量或监督路径相同。
- Source-SFT profile显示batch64/rank稳定finite且峰值allocated/reserved约32.16/42.04GB、约71.88 queries/s；batch128/rank约54.92/67.93GB、约72.05 queries/s，没有有效吞吐增益。因此正式run保留batch64/rank，只保存并验证step63。
- Source-SFT正式run完成63/63 steps、32,256 queries，训练loop wall-clock `450.263s`。首/末8-step mean loss为`0.15475/0.13883`，全程线性斜率约`-3.00e-4/step`；仍有下降迹象，按预先固定的同规模预算记为可能未充分训练，不增加steps。checkpoint的10个exact-resume文件与63条连续finite metrics均通过hash验证。
- 唯一step63完整validation为`61/400 = 15.25%`，5/8 tasks非零；逐任务Spatial 1/3=`5/1`、Object 1/3=`20/0`、Goal 3/6=`0/32`、Long 1/2=`1/2`。相对source base `48/400`增加13，但同时Goal 6从41降到32、Long 1从2降到1；增益主要来自Object 1 `+15`和两个Spatial tasks `+6`，不是全任务一致改善。
- 在400个匹配task/init rows上，Source-SFT与source base/AS都保持相同env seed、policy seed root和共享长度内全部policy-noise seed前缀。AS step250仍明显更高：`119/400`对`61/400`，主要差异为Object 1 `40 vs 20`、Object 3 `36 vs 0`和Long 1 `16 vs 1`；Source-SFT在Goal 6为`32 vs 27`并在两个Spatial tasks合计`6 vs 0`。
- Source-SFT train contract/metrics/checkpoint-manifest SHA256为`4e113268...dd87`/`02ca6611...64bb`/`2dce01f0...fb3e`；validation results SHA256为`92e3e667...3f6d`，24 workers均exit0、38 shards完整、400 raw rows唯一。三方法对照证据文件SHA256为`c376ef9c...a1f`。

## AS-Writer cross-suite wrong-video结果（2026-07-22）

- step250 cross-suite-wrong完整validation为`115/400 = 28.75%`，而correct-video为`119/400 = 29.75%`、source base为`48/400 = 12%`。核心correct−wrong差值仅`+4/400 = +1pp`，因此当前checkpoint虽明显优于base，却没有建立强teacher-video任务内容依赖。
- 逐任务correct/wrong为Spatial 1/3=`0/0` vs `0/1`、Object 1/3=`40/36` vs `37/33`、Goal 3/6=`0/27` vs `0/28`、Long 1/2=`16/0` vs `16/0`。400个paired rows中both-success 102、correct-only 17、wrong-only 13、both-fail 268；不是少数任务的大幅正负效果恰好抵消。
- 两臂400/400 rows的task/init、env seed、policy seed root及共享长度内policy-noise seed前缀完全匹配；noise列表长度仅因成功终止改变replan次数。wrong run 24 workers均exit0、38 shards和400 raw rows完整，results SHA256为`0e6ee518...a9ce`，correct/wrong对照证据SHA256为`d4a4f9f7...eaac`。
- 科学解释应保持克制：结果与Writer主要使用language、或视频编码在该训练下近乎不敏感相一致；在没有进一步干预证据前，不能声称AS-Writer从正确视频恢复了task-specific visual information。

## AS-Writer视频塌缩的根因与Writer-v2决策（2026-07-22）

- `fixed video / changed language`相对有效LoRA差约`4.02e-4`，`fixed language / changed video`约`7.52e-6`；所谓53倍只表示视频残差更弱，两者绝对量都很小，不能作为language conditioning有效的证据。当前v1输出应解释为近乎input-independent shared LoRA。
- 根因首先是目标不可辨识，而非Writer参数不足：functional PI05 policy本身收到正确language和observation；24个train tasks又各有唯一language；teacher video与action episode在同task内独立。因此一套共享domain/control LoRA加policy自身language就能降低loss，目标没有要求Writer使用视频。v1约12.48M参数生成1.287M LoRA参数，增加容量不会消除该捷径。
- 架构进一步放大捷径：每帧256个视觉token被全局平均、整条video只压为4个episode tokens，而parameter-query residual和有bias的共享head可绕过task memory直接产生公共LoRA。1000-step全程warmup可能影响checkpoint退化，但解释不了step250已经视频不敏感。
- owner决定效果优先、消融后补。因此Writer-v2一次性组合修复目标和架构：缓存固定4×4空间网格；language/video分别压到固定memory并使用不含learned-query residual的conditional attention；LoRA decoder只允许parameter query乘性寻址conditional task memory，最终head无bias并保持B=0 identity。
- owner最终口径明确否定零向量language和generic policy prompt。Writer-v2使用经同一PI05 tokenizer/embedding得到的固定中性Writer语言`perform the demonstrated task`（记为`g`）；frozen policy在全部分支始终接收action-query task的正确language A。主循环为`normal → full-language contrast → generic-language contrast`：分别训练`W(l_A,v_A)`、成对比较`W(l_A,v_A/B)`、成对比较`W(g,v_A/B)`。
- 两类contrast都只取半批独立A action queries并复制成correct/wrong两臂，因此每步总policy samples保持与normal相同；两臂从同一CPU/CUDA RNG状态开始，policy observation、language A、query、flow noise/time完全配对。correct臂各自有绝对functional action loss，bounded softplus matching只要求wrong loss高出margin，避免仅靠破坏wrong臂构成训练目标。
- 新cache合同的8卡真实smoke验证了generic不是零向量或运行时占位符：`perform the demonstrated task`经正常PI05纯语言tokenizer得到8个有效tokens，8个task cache中的embedding逐byte完全相同；视频tensor为`frames×16×2048 BF16`。8 tasks/1,033 frames均finite并完成，batch32 critical-path约666.25 frames/s，故只封存cache batch而不从中推断Writer效果。
- 只让Writer做language dropout是不充分的，因为policy language仍可承担任务；仅让wrong分支变差也不构成成功。后续判断必须同时看correct是否超过constant/shared adapter、同query functional loss是否按video-task正确排序、以及多validation tasks上的correct−wrong，而不是只看LoRA hash或相对倍数。
- Writer-v2 formal cache已完成32 development tasks×50 videos（1,600 episodes、274,523 frames），32/32 task tensor hashes、episode/frame counts与information-wall复核通过；generic embedding在全部tasks逐byte一致。该cache约17GB，保留4×4空间token而非旧全局平均。
- 8卡batch16学习profile在500-step scheduler horizon下实际执行30步，三种mode各10步。normal positive首/末3步均值为`0.16161/0.13554`；full-language wrong-minus-correct gap由`-3.44e-6`移至`+9.58e-5`，generic gap由`-1.04e-5`移至`+4.13e-5`。这证明训练方向已开始区分匹配关系，但幅度仍小，不能提前声称视频特异性成立。
- profile中normal/full/generic step中位耗时分别为`1.865/2.127/2.858s`，只作资源记录。owner明确不以约`1.3–1.6×`wall-clock倍率作为设计或启动门槛；四类概念分支、两个correct绝对行为目标、paired RNG与policy恒用正确language的科学合同优先。
- 首轮formal据此封存250 steps、batch16/rank与50-step checkpoints。每个checkpoint先做固定functional matching、generic-correct competence和adapter-specificity诊断；若step250仍明显未充分训练，按统一预算口径封存曲线并标记undertrained，不自动追加。
- Writer-v2首轮在commit `dcfb20689954225aa0cc92ae75f4103a7db6213c`上完成250/250 steps；训练段wall-clock `334.476s`，共产生32,000 policy samples与21,376 global action queries。24 tasks分别覆盖全部50条action episodes与全部50条teacher videos，validation/test action reads均为0；五个checkpoint manifests逐文件验证通过。
- 三种mode的首/末10步positive loss分别为normal `0.14501→0.11478`、full-language `0.14337→0.12463`、generic `0.13617→0.12040`。full/generic的wrong-minus-correct gap从首10步`+9.37e-5/+5.95e-5`增至末10步`+0.00707/+0.00788`，说明matching方向已明显增强且positive competence没有被训练目标主动牺牲；由于各step action query不同，这仍需fixed-query和rollout证据确认。
- canonical evaluator现在可表达`generic_correct`与`generic_cross_suite_wrong`：只有Writer language切换为cache中的中性`perform the demonstrated task` embedding，policy observation/prompt仍沿用evaluation task的正确language。旧`correct/cross_suite_wrong` adapter payload保持不变，已有评测的resume/reaggregation合同不受影响。

## Writer-v2首轮closed-loop结果与充分训练决策（2026-07-22）

- step250固定64-state screen中，full-language correct/wrong=`12/8`，generic correct/wrong=`12/8`；两组paired flips均为correct-only 7、wrong-only 3。generic-correct成功覆盖Goal与Object多个tasks，说明neutral Writer language下视频能生成有绝对competence的adapter，而不是只破坏negative臂。
- full-language完整validation为correct `83/400=20.75%`、cross-suite wrong `63/400=15.75%`，净差`+20/400=+5pp`。paired rows为both-success 43、correct-only 40、wrong-only 20、both-fail 297，exact McNemar `p=0.01349`；5/8 tasks正向、1负向、2持平。共享policy-noise seed前缀逐项一致，列表长度差只来自成功提前终止。
- 逐任务correct/wrong为Long 1/2=`4/2` vs `11/2`、Goal 3/6=`0/31` vs `0/20`、Object 1/3=`33/6` vs `29/0`、Spatial 1/3=`2/5` vs `1/0`。v2已比v1的119/115建立更强视频特异性，但correct由119降至83，当前科学问题从“无视频依赖”转为“特异性与绝对competence的权衡”。
- owner要求分别充分训练Writer-v2与Source-SFT并找各自validation最佳。Writer选择阶段只运行correct-video；只有唯一最强Writer checkpoint选定后才运行一次correct-language + cross-suite-wrong-video完整control，不运行per-checkpoint wrong或generic full arms。RL-Writer与seen继续暂停。
- Writer-v2 ceiling run从identity fresh训练1,500 steps，使用50-step warmup/1,500-step cosine并每250步保存；Source-SFT从identity fresh训练800 steps，使用100-step warmup/800-step cosine并保存100/200/400/600/800。两者独立选择，不匹配steps、queries、compute或参数；到120分钟guardrail仍未饱和则保留曲线并标记budget-censored。

## Development ceiling最终判断（2026-07-22）

- Source-SFT的完整validation在step200/400/600为`74/87/73`，最佳step400已经被前后候选夹住；恢复到计划上限800不会服务于“找最强已观测checkpoint”的当前判断，因此冻结step400且development不再重训。
- Writer-v2原run的step500/750/1000/1500 correct为`99/92/75/72`。独立dense-retention replay的350/400/450/500/550/600/650/700/750 cheap screens为`24/27/20/24/26/31/19/30/33`（各128），只将600/700/750提升为完整validation，得到`90/85/95`。它们均未超过原step500，后段亦持续退化，故没有证据支持再补800–950或更多细粒度训练。
- 选定的原run step500逐task correct为Long `5/0`、Goal `1/38`、Object `37/12`、Spatial `2/4`，合计`99/400`。其唯一cross-suite wrong-video arm为Long `6/2`、Goal `0/27`、Object `20/0`、Spatial `0/0`，合计`55/400`；correct-only/wrong-only=`56/12`，exact McNemar `p=6.21e-8`。
- v1的`119/115`说明高绝对correct主要来自几乎input-independent的公共adapter捷径；v2的condition-only架构与paired contrast使correct下降20、wrong下降60。v2仍有提升绝对competence的空间，但当前`99 > Source-SFT 87 > source base 48`且视频差`+44`跨6个tasks，已同时满足行为收益和视频特异性，当前最有价值的下一证据是RL-Writer而非继续AS消融。

## RL-Writer初始reward可学习性（2026-07-22）

- fresh identity Writer生成的初始LoRA功能上等于source base；首个完整24-task official random-reset cycle取得`7/24=29.17%`成功，成功来自7个不同tasks并覆盖Spatial、Goal、Long。因而zero-AS分支并不缺初始binary reward，当前没有科学理由消耗24条teacher-action warm-up。
- 成功episode需要26–57左右replan chunks时，整轨迹functional反传既逼近80GB，又因成功/失败ranks走不同DDP图而互等。8-chunk proxy-state微批把同一个episode mean loss精确拆分：每chunk权重为`1/(global_successes × episode_chunks)`，生成LoRA梯度汇总后只回传Writer一次；固定顺序all-reduce等价于对全局成功episodes取均值。
- 修复后的三次global updates均完成，successes为`4/1/2`，global gradient norms为`0.0535/0.0729/0.1341`且全部finite，峰值reserved仅40.84GB。这排除了“有reward但更新机械不可执行”的工程解释；下一科学问题是多cycle reward coverage与held validation是否随Writer训练改善。

## RL-Writer development选择与视频对照（2026-07-23）

- canonical zero-AS run从fresh identity出发，经update3→24→36→54三次真实8-rank exact-resume完成；累计`432`条official random-reset source rollouts、`81` successes、`131,354` environment actions，optimizer cursor为44，teacher-action consumption、fixed-pruned-init reads和validation/test reward reads均为0。净训练wall为`2261.716s`，远低于120分钟guardrail。
- 18个完整24-task cycles的后六轮successes为`5/2/4/3/5/3`，没有持续上升。固定同一64-state validation screen在update12/24/36/54依次为`6/11/15/14`；source base同subset为`7/64`，因此选择最早峰值update36并停止到54，不继续72/96/120。
- update36完整correct-video validation为`94/400=23.5%`，逐task为Long 1/2=`1/3`、Goal 3/6=`0/47`、Object 1/3=`40/0`、Spatial 1/3=`3/0`，成功覆盖5/8 tasks。development绝对性能排序为AS-Writer `99`、RL-Writer `94`、Source-SFT `87`、source base `48`。
- 同一checkpoint唯一cross-suite wrong-video arm为`87/400=21.75%`，逐task为Long `0/1`、Goal `0/44`、Object `40/0`、Spatial `2/0`。400对rows的correct-only/wrong-only/both/neither为`10/3/84/303`，exact McNemar `p=0.092285`；方向为正但不足以宣称强视频特异性。
- 科学结论因此分开表述：reward-only Writer确实学到可泛化且优于source base/Source-SFT的held competence，但其增益主要仍可由language/common adapter解释，视频因果控制较弱。已有明确source reward，故不以结果不够正为由启用micro-AS或改协议；后续RL-init task-local arm可以保留，但不得写成已证明强依赖teacher video。

## Sealed seen-panel比较（2026-07-23）

- 在任何这些outcome产生前封存的8-task panel（global IDs `0,2,15,12,21,28,39,37`）上，四方法均完成每task 50个相同official fixed states：source base `137/400`、Source-SFT step400 `182/400`、AS-Writer-v2 step500 correct-video `204/400`、RL-Writer update36 correct-video `164/400`。因此source acquisition在seen任务上成立，绝对排序为AS > SFT > RL > base；seen诊断不替代held validation。
- 按`Long 7/9, Goal 1/8, Object 2/5, Spatial 0/2`顺序，逐task successes为base `2/0,50/34,0/0,22/29`；SFT `0/0,48/41,10/0,41/42`；AS `1/0,49/47,23/1,41/42`；RL `1/0,41/45,1/0,35/41`。AS相对SFT的主要额外收益来自Object-2，而Long-9对四方法均为0，不能把aggregate写成全suite普遍提升。
- 四份results SHA256依次为base `91a9a31f...fb833`、SFT `05c4c0d1...d889b`、AS `3d640e57...d97479`、RL `92a958a3...3f2c8`；对应evaluation wall分别为`504.007/510.400/736.591/815.813s`，每份均保留400条raw rows和逐task aggregation。
- Source-SFT训练因owner在development选择step400后于step600手动停止，原runner未走到terminal summary发布；这不影响已原子发布的step400 checkpoint，但当前fail-closed evaluator要求训练summary。修复只从不可变run contract、600条连续metrics和step600 manifest重建`run_summary.json`，没有GPU forward、optimizer update或权重改写；summary/recovery provenance SHA256为`887ae816...ab2e`/`c7f29ae7...803c`。第一次seen启动在任何rollout前因此失败并保留，成功结果来自新root，未把失败目录续作正式证据。
- 该比较已回答Phase E的source acquisition问题；不再补seen wrong-video、额外checkpoint或generic arms。下一步使用development已选普通配置从规定fresh初态进行final 32-source重训。

## Final 32-source训练合同（2026-07-23）

- final角色只把封存的8 validation global IDs机械并入24 train IDs，四suite各8 source tasks，8 test IDs不变。AS与Source-SFT读取32 tasks各50条source action episodes；RL-Writer只读取同32 tasks的official random-reset reward和action-hidden videos；三者test action/reward/video reads在训练合同中均为0。
- AS复现development最优step500时必须保留原1,500-step cosine horizon；若把formal total直接改成500，LeRobot会把50-step warmup自动缩到16并把decay压到500，已不再是同一配置。因此final合同以`total_steps=1500`封存scheduler，机械`selected_stop_step=500`，只实际训练到并发布step500。
- Source-SFT同理保留800-step horizon与100-step warmup，机械停在development选定的step400；final不是把scheduler重缩到400。RL-Writer development update36在24 tasks上等于每task 12个完整cycles，32-task final据此固定为48 updates=`384` rollouts、每task仍12次，不用test outcome重新选择预算。
- 三份final配置SHA256为AS `ebe269ea...e299e`、Source-SFT `25e99628...d10c2`、RL `32dd979b...2ab30`。AS/RL扩展同一canonical runner的sealed stage和source roles，没有增加第二套入口；现有32-task feature cache完整覆盖train+validation，可直接复用而不生成重复17GB cache。

## Final AS-Writer完成（2026-07-23）

- final AS从同seed fresh identity在32 source tasks上实际完成500/500 selected steps，保持原1,500-step scheduler horizon；训练loop wall为`634.671s`。checkpoint coverage证明32 tasks均使用全部50条action episodes和全部50条teacher videos，累计64,000 policy samples、42,688 unique action queries，test action/video reads均为0。
- normal/full/generic positive loss首20到末20均值分别为`0.13805→0.12183`、`0.14542→0.11595`、`0.13450→0.11863`；末20步full/generic wrong-minus-correct gap为`0.00729/0.00783`。所有500 metrics连续唯一且finite，峰值reserved `68,344,086,528` bytes。
- final step500 checkpoint manifest payload SHA256为`b30b2e1d...c395`；run-contract/metrics/corrected-summary SHA256为`36207182...2de`/`0d208b15...b619`/`a4f76fb2...9de7`。
- 初始summary错误继承了development字段`validation_action_reads=0`，与final source角色冲突；训练contract、checkpoint和coverage均正确。已仅修正summary为400个validation-source action/video episodes available并保存零权重改动的correction provenance，SHA256 `ebc1bed8...414e`；代码同步按stage生成正确字段。

## Final Source-SFT完成（2026-07-23）

- final Source-SFT从fresh identity在32 source tasks上完成development已选的400/400 steps，保留原800-step cosine horizon与100-step warmup；训练loop wall为`2852.793s`，累计204,800 action queries，400条metrics连续且全部finite。
- step400 coverage证明32 tasks均覆盖全部50条action episodes，每task 6,400 examples，共138,952 unique query rows；test action/video reads均为0。首/末20-step平均loss为`0.15139→0.11531`，稳态吞吐约`71.88 queries/s`，峰值reserved `42,037,411,840` bytes。
- step400 checkpoint manifest payload SHA256为`0012ffb6...52bd`，run-contract/metrics/summary/file-manifest SHA256为`bc136964...da31`/`c0d91c9b...6211`/`ff0a33f7...d472`/`cd2f0766...d034`；10个exact-resume文件全部通过size/hash校验。

## Action-Memory Writer设计与资源结论（2026-07-23）

- 冻结PaliGemma的图文prefix可以预计算；但按当前stride-4开发集估算，保存pre-transformer图文prefix约50–60GiB，保存18层KV约250GiB，后者不符合当前收益/存储比。由于直接训练预计低于一小时，本轮不让cache工程阻塞科学结果；若后续确认encoder是主瓶颈，优先缓存language-independent image embeddings而非完整KV。
- Action-Memory Writer将语言理解留给冻结π0.5的PaliGemma，并让16个memory tokens从Action Expert流读取每帧图文prefix；初始化使用16个确定性正交32D action codes经冻结`action_in_proj`投到1024D后detach。Meta-LoRA仅增强teacher encoder对该输入的读取，不成为共享execution adapter。
- 最终profile证明per-rank batch16可执行，10.10M训练参数与rank128 Source-SFT 10.30M基本等量，因此后续AS对SFT的比较不再有约10倍训练参数容量差这一明显混杂。当前尚无closed-loop性能结论；必须由多checkpoint validation和唯一best correct/wrong arm决定。

## bias-free初测、bias恢复与新上限判据（2026-07-23）

- bias-free Action-Memory run的完整correct-video validation在step300/500为`105/400`与`89/400`；step300逐task为Long `15/3`、Goal `1/35`、Object `25/26`、Spatial `0/0`。step500为Long `5/0`、Goal `0/39`、Object `22/23`、Spatial `0/0`。paired step300-only/step500-only=`41/25`，exact binomial约`p=0.064`：300更好，但两个点不足以证明真实峰值或架构上限。
- 同参数口径rank128 Source-SFT先前完整validation曲线为step100/200/300/400/600=`90/105/65/122/111`。step300后又在400大幅恢复，证明一个下降点不能作为饱和证据；旧`122/400`是当前最佳观测值，不是已充分探索的上限。
- `condition-only`只要求完整LoRA经language/video条件路径产生、没有独立公共LoRA输出支路；它不要求所有内部线性层无bias。此前全局`bias=False`会降低约束网络的平移自由度并增加优化难度，属于额外实现限制。owner据此选择保持拓扑不变而恢复conditional path普通bias。
- 恢复bias不会自动创建显式共享adapter：temporal/layer/slot block和factor head仍只处理条件hidden states；factor-head最终bias从零初始化，与最终weight一起保证fresh task LoRA为identity。它仍可能通过共享参数学出近公共输出，这必须由correct/wrong视频行为和生成LoRA差异实证判断，而不是靠`bias=False`宣称排除。
- 新validation functional-loss panel固定512个task-balanced、video/action不配对query，可低成本观察loss斜率、train–val gap和候选checkpoint；由于teacher-forced action loss与closed-loop恢复能力可能错位，它不能单独选best。真实峰值要求完整8-task×50 success曲线，且“峰后持续下降”不能由单一相邻checkpoint判断。
- bias恢复只增加`21,696`个训练参数：Writer从`10,097,601`变为`10,119,297`，仍仅为rank128 Source-SFT容量的`98.27%`。四卡真实profile的显存与旧bias-free八卡profile几乎相同（reserved均约78.87GB），说明恢复bias没有引入隐藏模型副本或新执行支路。
- 每rank batch16/global64的第二个稳态profile step为`1.930s`，对应约`33.16` global queries/s；四卡相对旧八卡global128约`1.93–2.48s/step`的单步时延相近、总吞吐约减半，符合相同单卡工作量和world-size缩减预期。后续step数按实际action-query量解释，不能把四卡step与旧八卡step直接等同。
- rank128 Source-SFT从8 ranks切到4 ranks时无法宣称逐rank RNG与sampler完全相同的bitwise exact-resume，但这不意味着必须从零重训。合理做法是优先延续已有权重与optimizer、封存拓扑切换和重分片后的cursor，并将其标记为`topology-transition continuation`；只有完全相同的world size与合同才称exact-resume。batch size本身不是科学门槛，跨轨迹比较需同时给出optimizer updates、累计action queries和独立task-condition visits。
- 为充分检验`122/400`是否真实上限，新SFT最大horizon可延到2400，但不把cosine decay从800拉长：这样前800步不因扩大上限而获得更高LR，800后只是固定低LR tail。若完整validation已在800前后显示多点持续退化，便无需机械跑满；若仍波动或回升，则同一合同可恢复到1600/2400。
- bias-restored AS的封存512-row validation functional loss在step100–800依次为`0.135237/0.138363/0.134698/0.141123/0.134224/0.138690/0.139285/0.140583`。step400的单点回升随后在500完全恢复，验证了“单点不能早停”；而500后连续三个checkpoint回升、同时train loss继续下降，已经把closed-loop候选区间收缩到step500附近，但不能单独证明closed-loop最优。
- 独立backfill与resident monitor在step300/400/500的1,536条逐query loss完全一致，排除了训练进程内切换eval数据导致数值漂移的实现疑点。validation过程无gradient、无optimizer update，结束后恢复完整RNG与Writer train mode。

## bias-restored AS首轨迹结果与四卡scheduler混淆（2026-07-23）

- decay-6400首轨迹的完整correct-video validation为step300/500/800=`62/77/80`（各400）。逐任务分别为Long `0/0`、Goal `0/36`、Object `16/8`、Spatial `0/2`；Long `2/0`、Goal `0/27`、Object `33/15`、Spatial `0/0`；Long `3/0`、Goal `0/38`、Object `26/12`、Spatial `0/1`。results SHA256依次为`3c2643cf85c1a33a8335fd96636b46e55deef9f1839747c88cd7d62d30fa8334`、`db01087c00b2dd162f6900cead653d553d7c9e2e8ae8c9e20535c5902624fce6`、`f2ef8786ffb536b03483de3900a9fcab3fa3b6e417862c73cc89532174a8af10`。
- paired closed-loop比较中，step500对800为500-only `27`、800-only `30`、exact `p≈0.791`，两点实质持平；step300对800为23/41、`p≈0.0328`。因此functional val loss从step500的`0.134224`连续升至step800的`0.140583`可提示train–val分叉和候选区间，但不能精确排序77与80个闭环成功，最终best仍必须由完整rollout决定。
- 该轨迹存在决定性的scheduler混淆：旧八卡global batch128实验使用warmup50/decay1200；四卡global batch64若按action-query数保持同一学习率轨迹，应机械换算为warmup100/decay2400。现有warmup100正确，但decay6400令step500–800仍接近peak LR，所以`80/400`既不能归因于bias恢复，也不能作为当前架构上限。
- 干净修正只把cosine decay改为2400；冻结prefix、Action Expert memory、Meta-LoRA、temporal/layer/slot架构、全部已恢复conditional bias、数据、sampler和loss均不变。fresh首段到step1200并每100步在驻留进程测封存val-loss panel；若仍未建立闭环峰后持续下降，再exact-resume到1800/2400。

## query-scaled bias-restored AS训练曲线（2026-07-23）

- fresh四卡轨迹在step100–800的task-balanced val functional loss依次为`0.135237/0.141384/0.135191/0.134058/0.134964/0.135579/0.141342/0.139462`。step200上冲在300完全恢复，证明单点不能早停；step400后500/600/700连续上升，800虽较700回落但仍比400高`0.005403`，已形成可执行的候选谷底与峰后区间。
- 同期每100步train-loss mean为`0.138046/0.128935/0.122111/0.117919/0.116702/0.114383/0.110524/0.110282`，持续下降而validation自400后恶化，支持真实train–val分叉。按owner要求用趋势避免无意义长训，run在完整step800 checkpoint和validation后停止，不机械执行原首段1200。
- step800保存了全部24 train tasks的50 action episodes和50 teacher videos覆盖，累计51,200 global action queries；checkpoint manifest payload SHA256为`4198c15cd82c0acc000951462ec6c410273c6d2ea474f5f9673369173fb963a1`，Writer state SHA256为`e680c4f2f45acf4a35ea664ae7078958345bf6b601e0b4ced21880eba880debf`。中断信号到达前额外完成step801–809，但没有覆盖latest checkpoint；这些rows保留作透明运行证据，科学候选只使用原子step800及以前checkpoint。
- 完整correct-video closed-loop在step300/400/500/600/800依次为`57/91/86/87/88`；逐任务step400为Long `2/0`、Goal `0/43`、Object `31/15`、Spatial `0/0`。step400相对step300的paired flips为`49/15`、exact `p=2.44e-5`；相对500/600/800分别为`33/28`、`26/22`、`29/26`，后三个差异均不显著。故step400是该轨迹已观测真实峰值，step300已显著较差，无需再补step200。
- val-loss的用途需要保持克制：本轮最低val-loss和最高closed-loop都在step400，且400后的train–val分叉正确提示停止长训；但五个候选的loss-success Spearman仅`-0.10`，step300的loss接近400而success低34。因此它能判断继续/早停并收缩候选区间，不能独立精确排序checkpoint，最终best仍由完整rollout决定。
- 当前Action-Memory AS首轨迹没有通过相对SFT门槛：step400 AS为`91/400`，旧rank128 Source-SFT step400为`122/400`；paired AS-only/SFT-only=`25/56`、exact `p=7.52e-4`。在充分探索新的Source-SFT ceiling同时，下一轮AS应优先修正训练统计效率，而不是继续当前已经过拟合的轨迹：目前每个task-condition每步只有16个action queries，而rank128 SFT为128个；这会让约10.12M参数Writer的functional梯度方差远高于约10.30M参数SFT。

## 四卡rank128 Source-SFT ceiling profile（2026-07-23）

- 在物理GPU0–3上用4个对称DDP ranks、batch128/rank完成4步真实forward/backward；global batch保持512，与旧八卡rank128轨迹完全相同。四步均finite，step wall为`14.919/14.109/14.152/14.190s`，后3步平均吞吐约`36.18 queries/s`。
- 峰值CUDA allocated/reserved为`54,998,429,696/67,979,182,080` bytes，保留约14GB物理余量；无需降低batch。run contract、metrics、summary SHA256依次为`9a15add8...8479`、`ff033c3f...d2a0`、`c6b3c1f6...70cd`。
- 正式轨迹沿用已建立的rank128 optimizer：warmup100、cosine decay800、每100步checkpoint。首段到step800；之后只在val-loss或闭环候选仍未充分时按300步exact-resume。每个checkpoint在同一驻留policy上原地测封存512-query validation action loss，不卸载模型、不更新参数；完整8×50 closed-loop仍决定最终best。

## AS query-matched训练修正（2026-07-23）

- 首轨迹每rank每个task/video condition只用16个action queries，而容量匹配的rank128 SFT每rank为128；两者虽然训练参数约10.12M/10.30M相当，单步functional梯度统计精度并不相当。这是先于改Writer架构需要消除的混杂。
- 修正保持Writer、16 memory tokens、Meta-LoRA、temporal/layer/slot、rank16输出LoRA和信息墙全部不变：一次生成adapter后，将128 queries拆成8个16-query policy microbatches；每个microbatch独立求adapter梯度，按真实query数加权平均，再只对同一Writer图反传一次。峰值显存保持原batch16量级，计算量约增至8次policy forward。
- 该行为仍由现有`ember.writer.as_step`和同一canonical runner拥有，没有并行入口。normal以及已有contrast模式都复用同一微批机制；paired contrast仍在整组microbatch前后恢复并核验相同policy RNG。正式配置在真实四卡profile前保持pending。
- owner随后明确batch size不应成为公平性门槛。最终待profile方案不再用每condition 64 queries去机械匹配SFT；改为四卡每rank顺序累计2个独立conditions、每condition保留16 queries，使每update合计8 conditions/128 queries，与旧八卡AS逻辑单位一致。上面的128-query/condition方案保留为未启动的设计provenance，不进入正式实验。

## rank128 Source-SFT step100–400训练–validation分叉（2026-07-23）

- 新四卡fresh run的固定512-query validation loss从step100的`0.1330666`连续升到step200 `0.1333360`、step300 `0.1341674`、step400 `0.1371306`；100→400相对恶化约3.05%，最后一段恶化约2.21%。同期train-loss区间均值持续下降，形成了比单点波动更强的早停证据。
- loss曲线只把closed-loop候选收缩到step100–400，不能独立宣布step100最优。旧rank128完整success高度非单调，因此必须比较四个checkpoint的同seed 8×50 rollouts；若真实success仍在最晚点上升，则从step400 exact-resume继续。
- 中间formal checkpoint已由run-contract hash、四rank RNG/optimizer state、LoRA hash和原子manifest封存。正式validation不需要伪造整个run完成；缺失的最终summary保持为null并明确标注checkpoint-before-completion，同时final/test仍禁止打开。

## Source-SFT八卡/四卡训练量口径与step100–800闭环（2026-07-24）

- 旧八卡rank128轨迹每个optimizer update覆盖8个task小批和512个action queries；其`step400=122/400`对应400次更新与204,800 queries。当前四卡轨迹用batch128/rank保持每步512 queries，所以同为step400时主要训练量大体可比。checkpoint实数进一步确认两者均消费`204,800` examples；旧/新每task范围分别为`8,512–8,576`与`8,448–8,576`。每次更新内4个更大task小批与8个较小task小批主要改变梯度方差和顺序，不能据此将四卡step机械乘2或把step800称为旧step400的等价点。
- 当前四卡fresh轨迹step100/200/300/400/500/600/700/800的完整validation success为`81/95/68/78/94/99/108/97`。step700是该轨迹当前best；600→700 paired flips为`19/28`，700→800为`31/20`，均不足以证明显著上升或持续下降。旧八卡step400的`122/400`仍是全部SFT候选的incumbent，而不是可移植的早停step。
- 对应四卡functional val loss为`0.133067/0.133336/0.134167/0.137131/0.134146/0.134832/0.135634/0.135192`，与闭环曲线只呈弱对应，因此后续仅微弱参考。四卡run已从完整step800原地exact-resume到1100；候选判断继续以同seed 8×50 rollout为主，不因batch或卡数变化机械重启。

## Action-Memory时间顺序诊断与最小结构修正（2026-07-24）

- 旧Action-Memory checkpoint在固定语言下换跨suite视频会明显改变有效LoRA，但完整视频倒序/打乱的相对变化仅约`0.036/0.027`，远小于单帧或重复端点帧的`0.237–0.312`；因此问题不是公共LoRA塌缩，而是temporal路径近似将视频作为无序状态集合。
- 当前最小修正不预设functional loss与closed-loop错位，也不增加额外训练目标：只让顺序通过RoPE直接进入temporal Q/K，并用4个不传播query residual的learned memory queries替代单一pool。显式帧差分、手工phase和order auxiliary均暂不采用，以便500-step实验直接回答正常AS监督是否会利用可表达的时间顺序。

## temporal-RoPE Writer 500-step结果（2026-07-24）

- bias-enabled Action-Memory、Meta-LoRA、信息墙和完整rank16 LoRA均未改变；唯一结构变化是temporal self-attention使用原始frame index的1D RoPE，并以4个condition-only temporal memory queries保留多阶段摘要。Writer为`11,252,737`个训练参数，约为rank128 Source-SFT的`1.093×`。
- 四卡native global64 fresh训练到step500，训练body wall为`1188.6s`。封存512-query functional validation loss在step400/500为`0.1364674/0.1369167`，几乎持平且后者略差；它正确提示没有继续改善，但最终选择仍由closed-loop决定。
- 优化后的`per_sample_lora_batched_replan` evaluator在step400/500分别得到`108/400`和`98/400`。paired rows中step400-only/step500-only=`24/14`、both=`84`；逐task step400→500为Long `5→4, 3→0`、Goal `0→0, 37→35`、Object `37→35, 26→24`、Spatial `0→0, 0→0`，故step400是明确observed-best。它仍低于rank128 Source-SFT incumbent `122/400`以及旧v1 AS的`119/400`，本次结构修正没有恢复AS绝对泛化上限。
- 仅在step400进行post-selection、无action/reward/outcome的特异性诊断。保持正确language不变，仅换跨suite错误视频时，temporal feature、LoRA参数和有效LoRA更新的中位相对L2分别为`0.1228/0.1595/0.2267`；同task另一demo对应`0.0255/0.0368/0.0403`。8/8 tasks的跨suite有效更新变化均为`0.1770–0.2715`，说明视频任务内容已经稳定进入adapter，不是公共LoRA塌缩。
- 对同一视频倒放或确定性乱序时，有效LoRA更新的中位相对L2却仅为`0.00937/0.00699`，cosine中位数为`0.999957/0.999976`；而只保留首/中/末帧时为`0.1745/0.1124/0.3339`。因此模型使用了多帧内容和端点状态，但normal functional监督几乎没有让新RoPE路径学习动作顺序。这个结论是representation/adapter诊断，不冒充错误视频的closed-loop performance arm。
- 当前最直接的科学结论不是“视频没用”，而是“视频语义内容有用、时间顺序仍未被当前监督识别”。由于correct性能未超过SFT，按owner的快速子任务合同不启动contrast、更多checkpoint或RL，先停止汇报。

# EMBER Task Plan

最后更新：2026-07-20。状态词只描述当前新 70/10/10 主线；旧 60/15/15 实验不计为这些阶段完成。

本文后续的 frozen source embodiment base 统一且只指：通用预训练 `lerobot/smolvla_base` → 在 70 个 train tasks、每任务全部 50 条成功 teacher episodes 上联合训练 → 得到一个共享、多任务、语言条件的 source embodiment base → 训练完成后冻结。EMBER、target-action-supervised direct LoRA oracle 和 ordinary task-local LoRA RL 都以它为共同起点；它只能按 train/source evidence 选择。

## 完成定义

长期工作只有在以下链条全部完成后才可标记 Goal complete：

1. specification-only 同分布 70/10/10 split 与新 manifest 永久封存；
2. 70×50 shared source embodiment base 可复现训练、按 train/source evidence 选择并冻结；
3. Writer cold start 在多个不同类别 validation tasks 上明显超过 frozen source embodiment base；
4. Writer-only RL 保存独立产物并报告相对 cold-start 的 validation 原始结果；
5. Writer-init task-local LoRA RL 与 matched zero-init ordinary LoRA RL 在相同 interaction/update 预算下完成；
6. 全部方法、checkpoint、预算、selection rule 和 baseline 冻结后，补齐必要独立 seeds、尚缺 matched baselines，并统一完成 reporting-only test；
7. shared-frozen held/test 合同成立。

完全符合最终合同的开发 trajectory 可直接作为正式 seed 1；只重训受 split、source base、architecture、LoRA、loss、data、optimizer、RL 或 evaluator 变化影响的 arms。source-only reward/meta outer learning 只可在 Phase F 完成后作为可选增强，不是完成条件；当前 Goal 不包含 OpenVLA-OFT 或 source direct-LoRA localization。

代码完成、环境完成、一个 checkpoint、Gate -1、Gate 0、一次 source 结果或“可以进入 Writer”都不能完成长期 Goal。

## Phase A：永久封存 70/10/10

状态：已完成；canonical seal 为 `configs/libero90_70_10_10/`。

动作：

- [x] 从 pinned LIBERO-90 task language/scene 重建 90-row role-aware factor table。
- [x] 明确 verb、moved object、target receptacle、target relation、order/composition 和 specification difficulty。
- [x] 未知/歧义 instruction fail closed。
- [x] 在只读 specification surface 上确定 deterministic algorithm、seed 和 objective。
- [x] 搜索并冻结 70 train / 10 validation / 10 test。
- [x] 验证 task IDs 互斥且覆盖 0..89。
- [x] 验证每个 val/test task-relevant exact role atom 在 train 至少有 2 个支持，并封存同 family top-neighbor audit。
- [x] 验证 val/test 各有 10 个不同 scene、相同 scene/category/difficulty 配额。
- [x] 在任何新协议 policy outcome 前封存 factor rules、IDs、algorithm/seed、manifest 和 SHA256。
- [x] 重新审计 HDF5、BDDL、init states、controller、camera、50 success demos/task。
- [x] normalization 只使用新 train tasks；validation/test numeric access 为零。

停止条件：存在多个实质改变论文解释、无法用 specification-only objective 排序的 split 时才询问 owner。不得用旧 policy performance 选 split。

## Phase B：Source embodiment base

状态：source-only 选择已结束并冻结 step 630。短续段 step 945 同口径仍为 `15/400`，相对 step 630 为 `5 gains / 5 losses`，因此按预声明规则不评 735/840，也不读取 validation/test 选择 base。

合同：

- 起点：`lerobot/smolvla_base` 固定 revision。
- 数据：全部 70×50 teacher episodes。
- 训练对象：官方成熟 SmolVLA action expert + 必要 projections；一个共享多任务 base。
- 混合抽样：70-task deterministic no-replacement cycles。
- 资源：8 GPU DDP，一卡一 policy CUDA rank；GPU0 不放额外 CUDA controller/model；每卡约 10GB headroom。
- 初始 wall-clock：约 30 分钟。
- checkpoint：估算总 steps 后在约 1/3、2/3、3/3，且每个边界完整覆盖 70 tasks。
- 选择：只用 train/source development，不看 validation/test。

已锁定首轮运行：每 rank batch 352、global batch 2816、630 optimizer steps、checkpoint 210/420/630；沿用官方 action-expert recipe 的 peak LR `1e-4`，不做未经验证的大 batch LR 放大。显式 rank-local device 修复后的 8 卡 profile 为 2.590s/step、1087.2 samples/s、每卡峰值 reserved 67.35GiB。

正式 seed-1 trajectory 已在 commit `72eb10d` 完成：630 steps、退出码 0、约 28 分钟；最终 loss `0.483089`，三个 checkpoint 均含 policy、optimizer/scheduler/scaler、sampler/data cursor 和 8-rank RNG。最终 checkpoint manifest SHA256 为 `89e9f493...ed22c`，launch contract SHA256 为 `22c4ffb5...2e8`；step 630 累计 1,774,080 global examples、5,040 global task slots，70 tasks 均覆盖全部 50 episodes。

同一组 8 个预声明 train/source-development tasks、每 task 50 个固定 fresh states、相同 env/policy seeds 下，step 210/420/630 分别为 `3/400`、`8/400`、`15/400`。420→630 有 11 个 paired gains、4 个 paired losses；因此只用 source evidence 封存一次相对 thirds 为 735/840/945 的 continuation。它保留 step-630 policy、optimizer、8-rank RNG、sampler cursor 和原 scheduler state，LR 继续钳在 `2.5e-6`，不重启高 LR。

动作：

- [x] 核验官方 recipe、trainable names、normalization 和 exact loss。
- [x] 写唯一 canonical config/runner，不恢复旧 Gate0 runner。
- [x] 1–5 分钟吞吐/显存 smoke，只读 mechanics。
- [x] 固定 global batch 与学习率合同。
- [x] 运行约 30 分钟 exact-resume trajectory。
- [x] 第一个完整阶段结果前确认 3500 条 episode 均贡献训练信号，并记录完成的 corpus epochs/consumed chunks。
- [x] 标准 LIBERO h50 闭环测初始 thirds 的 train/source development；continuation 后重测最终候选。
- [x] 冻结 source base、完整训练状态、hash 和 compute ledger。
- [x] 冻结后在 10 个 validation tasks 建立 frozen source embodiment base reference。
- [x] 每个 validation task 用自己的全部 50 条 teacher action episodes 训练 matched direct LoRA oracle，并明确标注 action-supervised reference。
- [ ] Phase F 解封前不运行任何 test policy evaluation、不训练 test direct LoRA，也不读取 test actions/reward/success。

direct oracle 的单一 canonical runner 已接通并完成：每个 validation task 独立从 selected step630 注入同一 37-target LoRA，只用自己的 50 条 action episodes 做标准 action/flow loss；每 task 固定消费 69,120 queries，与 Writer cold-start 的 `1575×8×384/70` 完全相等。正式运行采用 batch384、每 task 180 steps、60/120/180 checkpoints；30 个 checkpoint manifests/files 均验证通过，每个边界覆盖全部 50 episodes，8 卡始终各一个 rank，峰值 reserved 69.09GiB，10 tasks 总 wall-clock 约 17.5 分钟。固定 final checkpoint 的 500-row fresh evaluation 为 `186/500`，per-task `{0:48,8:1,15:17,28:36,40:21,56:11,61:9,71:2,85:11,88:30}`；相对 frozen base `56/500` 为配对 `141 gains / 11 losses / net +130`。完整训练、checkpoint 与评估 hashes 封存在 `configs/direct_lora_validation_reference_v1.json`。

已触发且只触发一次上述停止规则：从 step630 exact-resume 到step945，新增315 steps、约14分钟，checkpoints735/840/945。step945 与630 均为 `15/400`，配对 `5 gains / 5 losses`，故已按规则冻结step630且不补测735/840；selection seal 为 `configs/source_base_selected_v1.json`。

## Phase C：Writer cold start

状态：train/validation 正式 feature cache、真实 8-rank interrupted/resume profile、Writer fresh-eval smoke 和 1575-step formal cold start 已完成；525/1050/1575 三个候选按首个 policy RNG 选择 step1050，随后预封存的独立 policy RNG 确认也已完成。

固定结构：

- language + 任意数量/长度 action-hidden videos；
- 冻结 VLM features 可缓存；
- trainable temporal/episode/set attention、fusion、task memory、layer-aware LoRA decoder；
- 输出完整 37-target rank-32/alpha-16/dropout-0 LoRA。

同空间 LoRA 合同已封存为 `configs/smolvla_lora_v1.json`：精确37 targets、74个A/B tensors、1,485,312 parameters；Writer、direct oracle 和后续 matched RL 共用这一实现。`scripts/train_writer_cold_start.py` 已把 selected source base、full-video cache、mixed sampler、functional loss 和 exact-resume checkpoint 接成唯一活动路径；真实 profile/resume 通过后 formal gate 已按测量参数打开。

frozen-VLM cache 的唯一入口与合同已封存为 `scripts/cache_writer_features.py` / `configs/writer_feature_cache_v1.json`。正式 cache 已由 selected step630 生成：70 tasks、3500 full episodes、537,946 frames，70 个 task tensors 均独立通过 size/SHA 校验；提取期间每卡恰好一个 CUDA rank、GPU0 无额外进程。

真实 functional profile 采用每 rank batch 384、global batch 3072；step 17 checkpoint 的 10 个文件独立通过 SHA，随后 exact-resume 至 step 35。35 steps 恰好覆盖 4 个完整 70-task cycles，每 task 1536 queries 且全部 50 episodes 覆盖；steps 2–35 平均 3.426s，峰值 allocated/reserved 68.00/70.54GiB。正式合同据此封存为 1575 steps、checkpoints 525/1050/1575，预计纯训练 89.93 分钟。

正式 seed-1 trajectory 已在 commit `69bbdee` 上完成 1575/1575 steps、exit 0，wall-clock 约 92.9 分钟；累计 4,838,400 queries。最终每个 70 train task 精确消费 69,120 queries 且覆盖全部 50 episodes，8 卡始终各一个 CUDA rank，峰值 reserved 70.71GiB。三个 thirds checkpoint 的 Writer、optimizer/scheduler 与 8-rank RNG 文件均由 manifest 做 size/SHA 封存。

同一 500 条 validation rows 上，frozen base 与 Writer step525/1050/1575 的成功数分别为 `56/500` 与 `58/500, 63/500, 60/500`。step1050 按预封存规则获选；其 per-task 为 `{0:19,8:0,15:0,28:24,40:0,56:1,61:0,71:0,85:0,88:19}`，相对 base 配对 `31 gains / 24 losses / net +7`。正增益来自 KITCHEN-actuation task28 与 STUDY-pick-place task88，但 aggregate 仅 +1.4pp 且 task0 回退 9，因此 checkpoint 选择已完成、核心机制结论仍未成立；已在读取第二 RNG outcome 前封存只复核 base 与 selected step1050、不得重选 checkpoint 的独立 policy-RNG 合同。

第二 RNG 同口径得到 base/Writer `51/500, 57/500`，配对 `30 gains / 24 losses / net +6`。两个 RNG 合并为 `107/1000` 对 `120/1000`；task28 配对净增益 `+22/100`、task88 `+12/100`，且两次 RNG 中方向分别为 `+10/+12` 与 `+6/+6`，因此 language+action-hidden-video Writer 在 KITCHEN-actuation 和 STUDY-pick-place 两个未见类别上的即时功能信号可复现。覆盖仍有限：总增益仅 +1.3pp，task0 为 `-20/100`，其余多数任务双方均为零；Phase D 必须检验 source reward 是否扩展而非只移动这项窄效用。

validation action-hidden cache 覆盖预封存 10 tasks × 50 full episodes、63,544 frames。profile step35 仅作 fresh-eval mechanics smoke：8 ranks 对 task0 生成的完整 LoRA SHA 完全一致，8 个固定 state 各出现一次并成功闭环退出；该小分母结果不用于任何行为判断。

动作：

- [x] 将 `writer/model.py`、`temporal.py`、`data.py` 接入新 source base 与新 manifest。
- [x] 构建每 task 全部 50 条完整视频的 frozen feature cache，保留 episode boundaries。
- [x] 70-task mixed/no-replacement sampler。
- [x] 70-task cycle 之上再保证每 task 50 episodes 的最小覆盖；记录 consumed episode/chunk identity。
- [x] functional action/flow loss；action 不进入 Writer。
- [x] 只做 shape/OOM/NaN/gradient/freeze/resume/leakage smoke。
- [x] 8 GPU 真实训练，初始总预算约 90 分钟。
- [x] checkpoints 位于估算总 steps 的 1/3、2/3、3/3，每个边界覆盖 70 tasks。
- [x] 在 10 个跨类别 validation tasks 上按预封存规则选择 Writer cold-start checkpoint。
- [x] 在 10 个跨类别 validation tasks 比 frozen source embodiment base、Writer、direct LoRA oracle。
- [x] 保存原始 per-task successes、rollouts、seeds、steps、数据、GPU 和 wall-clock。

判断：跨多个类别明显超过 frozen source embodiment base 即可证明核心机制产生功能价值并进入下一阶段；低于 direct LoRA 是优化信号，不是一票否决。效果差时先看实现、数据规模和 source base，再从同一 checkpoint 加训；数据充分仍失败才改 architecture/loss。

## Phase D：Writer-only RL

状态：完整 formal 与四候选 validation 选择均已完成；Writer-only RL 没有改善 validation，故按预封存规则保留 cold step1050 进入 Phase E。

- [x] 只用 70 train/source tasks。
- [x] base 冻结；生成 LoRA 不原位更新；reward 只更新 Writer。
- [x] 70-task balanced mixed rollout/update cycles。
- [x] 总 wall-clock 目标 ≤约 90 分钟。
- [x] checkpoints 在 1/3、2/3、3/3 full-task cycles。
- [x] 每个候选在 validation 做 frozen-Writer zero-interaction h50 评估。
- [x] 报告 cold-start 与 Writer-only-RL 的原始数字。
- [x] 在 validation 选择一个最佳 Writer initialization 进入首轮 task-local RL。

profile 从 update1 checkpoint 由新进程恢复到 update9，完成一个 70-task cycle：每 task 4 个官方随机 reset rollouts，共 `280` interactions、`87` successes、`90,391` env steps 和 9 个真实 Writer updates；72 个 rank/update ledgers、280 个唯一 env/policy seed rows 与两个 10-file checkpoints 全部通过恢复审计。max-rank wall 为 405.50 秒，因此 formal 封存 12 个 full cycles，即 108 updates、每 task 48 rollouts、总 3,360 interactions，thirds 为 36/72/108，预计约 81.1 分钟。

cold step1050 与 Writer-RL update36/72/108 的 validation 选择规则已在任何 Writer-RL validation outcome 前封存为 `configs/writer_only_rl_selection_rule_v1.json`：先最大化 500 rows 总成功数，再比较相对 frozen base 的正增益任务数与配对净增益，最后偏好更少 source reward interactions。全部候选复用 cold/base 的 fixed states、env/policy seeds 和 evaluator；选择与机制声明分开，test 保持封存。

formal 实际完成 12 个 full cycles、108 declared updates、107 个有 reward signal 的 optimizer updates；70 tasks 各 48 条官方随机 reset rollouts，共 `3,360` interactions、`679` successes、`1,176,874` env steps，max-rank wall `4,862.10s`。864 个 ledgers、3,360 个唯一 seed rows 与 36/72/108 三个 10-file checkpoints 全部通过审计。相同 500 条 validation rows 上，cold/update36/update72/update108 分别为 `63/56/36/15` successes；逐任务为 cold `{0:19,8:0,15:0,28:24,40:0,56:1,61:0,71:0,85:0,88:19}`、u36 `{0:20,8:0,15:0,28:20,40:0,56:0,61:0,71:0,85:0,88:16}`、u72 `{0:19,8:0,15:0,28:15,40:0,56:0,61:0,71:0,85:0,88:2}`、u108 `{0:12,8:0,15:0,28:3,40:0,56:0,61:0,71:0,85:0,88:0}`。因此该 source-only reward recipe 是已完成的负结果，不改算法补救，也不阻塞 matched task-local RL；选择与 hashes 封存在 `configs/writer_only_rl_selected_v1.json`。

## Phase E：Task-local LoRA RL

状态：cold step1050 已锁定；matched formal adaptation、预算内 checkpoint 选择和独立 fixed-50 fresh evaluation 均已完成。Writer-init arm 明显优于 matched identity-init arm，但收益集中而非广泛覆盖。

Arm A：

- source base；
- zero/identity LoRA；
- ordinary task-local LoRA RL。

Arm B：

- source base；
- best Writer-generated LoRA；
- 完全相同 ordinary task-local LoRA RL。

动作：

- [x] 在 validation 选择成熟 RL algorithm、K interactions、U updates、optimizer 和 selection rule。
- [x] 所有 RL 更新与 adaptation checkpoint 选择 rollouts 通过 LIBERO 官方 reset/BDDL 随机化初态；禁止从固定 50 个 `.pruned_init` states 取样。
- [x] zero/identity-init 与 Writer-init 两臂使用相同 task、env seeds 和初态序列，并保存可恢复的 worker RNG/seed schedule 与 interaction cursor。
- [x] 初始 profile 约 10–15 分钟/task/arm 等价预算，8 卡并行。
- [x] validation 的 10–15 分钟/task/arm 预算已产生可判读 matched contrast，未触发约 20 分钟扩展；不得按 task outcome 临时改预算。
- [x] checkpoints K/3、2K/3、K。
- [x] 每 task 可按预算内 adaptation reward 选择 checkpoint。
- [x] 用与 RL 数据分离的固定 50 个 `.pruned_init` states 做 fresh evaluation，保留 dummy settling、horizon 400、成功即终止。
- [x] 报告 J0、curve、AUC、time-to-threshold、JK、JK−J0、interactions、updates、wall-clock。

活动实现把 4-task profile 映射成 8 个 task×arm 单元，恰好每卡一个 policy
进程；正式 20 单元按 rank-strided 分片。两臂的 rollout seed 函数不含 arm，
ledger 保存每次 official random reset 的 env/policy seed、interaction cursor 和
`fixed_init_state_id=null`。checkpoint 只允许在完整 rollout block 后保存，恢复时
机械核验 `interaction_cursor = update × rollouts_per_update`；固定 50 states 仅由
同一个 fresh evaluator 在 RL 外读取。

在读取 profile reward outcome 前，formal U/K 的选择已限定为纯吞吐规则：只用
两个 exact-resume segments 中 `task_local_reward_update.step_seconds` 的 p90，选择
满足每 task/arm 约 10–15 分钟、20 单元总 wall-clock 不超过 100 分钟的最大 3
的正整数倍，formal 最多 3 units/rank 并计 180 秒启动余量；若可行需达到 10 分钟
下限，checkpoint 固定为 thirds。profile successes 不参与预算选择。

真实 profile 在全新进程由 update1 checkpoint exact-resume 到 update3，8 个 task×arm 单元共生成 24 个 ledgers、96 条 official-random-reset trajectories 和 16 个已验证 checkpoints；matched 两臂的 env/policy seed blocks 逐 task 完全相同，所有 `fixed_init_state_id=null`。24 个 update timing 的线性插值 p90 为 `49.8926s`，纯吞吐规则选择 `U=18`、每 task/arm `K=72` interactions、thirds `6/12/18`；20 单元总 interactions 为 `1,440`，按最多 3 units/rank 加 180 秒启动余量投影 `2,874.20s`，低于 100 分钟上限。

formal 实际 wall `1,982s`，完成 360 ledgers、1,440 trajectories、720 个唯一 matched seed rows 和 60 个逐文件验证 checkpoints；所有 RL 初态仍为官方随机 reset 且 fixed ID 为 null。adaptation reward identity/Writer 为 `89/720` 与 `110/720`，AUC `0.1236/0.1528`。独立同一 500 rows fresh evaluation 的 base/cold/identity-RL/Writer-RL/direct 为 `56/63/54/74/186`；identity-RL 对 base 配对净 `-2`，Writer-RL 对 cold 净 `+11`，Writer-RL 对 matched identity-RL 为 `43 gains / 23 losses / net +20`。逐任务 Writer-RL 为 `{0:30,8:0,15:0,28:10,40:0,56:1,61:0,71:0,85:0,88:33}`，相对 identity 的正增益在 task0/task88，task28 回退。结果足以支持“Writer initialization 提高 matched task-local RL 的终点”，但不支持广泛 reward adaptation；完整 seal 为 `configs/task_local_lora_rl_validation_v1.json`。效应与 adaptation reward 方向一致且 matched paired net 为 +20，第二 policy RNG 不会改变这项有限结论，故为效率不追加。

所有 target tasks 的总 wall-clock 是预算对象，不允许每 task 各跑 90 分钟。

## Phase F：合同冻结与统一 reporting-only test

进入条件：Writer cold start 已在多个类别 validation tasks 上明显优于 frozen source embodiment base，Writer-only RL 和 matched task-local RL 已完成 validation 选择。

状态：最终方法、预算、selection rule、evaluator 语义和 baseline 集已封存在 `configs/phase_f_protocol_v1.json`；test 仍未打开。完全合规的开发 trajectory 保留为 seed1；打开 test 前只补一个固定 step1050、不能重选方法/checkpoint 的独立 Writer training seed2，并完成其 validation confirmation 与产物封存。

- [x] 冻结 split、source base、Writer/architecture、LoRA、loss、data、optimizer、steps、RL algorithm、interaction/update budget、checkpoint selection rule、evaluator 和 baseline 集合。
- [x] 保留从一开始就使用完整 70×50 数据且完全符合最终合同的开发 trajectory 作为正式 seed 1。
- [ ] 只重训因合同变化失去可比性的 arms，并补齐必要独立 training seeds、尚缺 matched baselines 和 validation rows。当前只缺固定协议的 Writer seed2 confirmation；没有合同变化 arm，也没有未完成的核心 matched baseline。
- [x] 冻结并完成 frozen source embodiment base、最佳 EMBER zero-interaction、zero/identity-init ordinary LoRA RL、best Writer-init matched LoRA RL，以及必要同信息墙强 baseline。identity-init ordinary RL 是与 Writer-init 臂除初始化外逐项相同的强 matched baseline；不为 core test 新增未预验证的 frontier 方法。
- [ ] 最后统一解封 test，运行上述方法和 target-action-supervised direct LoRA oracle；test direct LoRA 每 task 可用全部 50 条 teacher action episodes，但不进入信息匹配主结论。
- [ ] test task-local RL 只按 validation 已冻结的 reward budget/selection rule 适应，最终性能使用 fresh rollouts；test 结果不反向改方法。
- [ ] 统一 task/init/RNG/h50/precision/budget，报告 data、steps、interactions、GPU-hours、wall-clock 和原始 rows。

当前不做 standalone Language-only Writer、standalone Video-only Writer、source direct-LoRA localization、full/action-expert task-local RL upper bound 或 OpenVLA-OFT。

## Phase F 之后的可选增强：Source-only reward/meta outer learning

状态：只在 Phase F 完成后按价值决定，不阻塞 Phase F 或 Goal complete。

- [ ] 若执行，inner/source RL 只更新 task-local LoRA。
- [ ] outer source reward/meta objective 只更新 Writer；shared source embodiment base 始终冻结。
- [ ] 只用 source 训练、validation 选择，不影响已经冻结并报告的核心 Phase F 结果。
- [ ] 有收益时作为后续额外 arm；未实现或负结果不否定核心 EMBER。若未来要给该 arm 增加 test，必须另行冻结并声明为 Phase F 之后的扩展。

## 每次运行前的最小检查

- [ ] Git clean、正确 main revision。
- [ ] `nvidia-smi` 实时确认设备和所有者。
- [ ] 不超过 8 张 EMBER A100。
- [ ] 个人数据占用预计峰值低于 500GB。
- [ ] exact command/config/revisions/output root 唯一。
- [ ] 不覆盖已有 artifact。
- [ ] split/surface/visible fields 正确。
- [ ] checkpoint/resume stop condition、worker RNG/env seed schedule 和 interaction cursor 清楚。
- [ ] 视频/gallery 保留和清理策略清楚。

## 每次 meaningful 结果后

- [ ] 更新 `findings.md`：事实、指标、局限、解释边界。
- [ ] 更新 `progress.md`：命令入口、产物、当前 Git、下一动作。
- [ ] 更新本计划状态，不复制长日志。
- [ ] targeted verification。
- [ ] commit + push。

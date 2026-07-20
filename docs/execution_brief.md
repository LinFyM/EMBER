# EMBER 当前完整执行合同

状态：2026-07-20 最终 owner correction 对齐后的唯一活动版本。

执行状态：Phase A 已封存于 `configs/libero90_70_10_10/`；当前进入 source embodiment base。

本文件回答三个问题：EMBER 到底研究什么；每个阶段究竟看什么数据、更新什么参数；下一 session 应按什么顺序运行。若任何旧文档、旧配置或旧 checkpoint 与本文件冲突，以本文件为准。

## 1. 研究问题

EMBER 要验证的是：一个共享 Writer 能否从任务语言和不含动作数值的 teaching video 中，直接生成一套具有即时功能价值的任务专属 LoRA；这套 LoRA 随后还能像普通 task-local LoRA 一样继续用环境 reward 适应。

第一阶段只研究同一 Panda embodiment、同一 LIBERO simulator dynamics family、LIBERO-90 内的组合式未见任务。暂不声称 human video、cross-embodiment、真实机器人或任意互联网视频迁移。

EMBER 不是：

- 通用 optimizer；
- 预测更新方向的 learned optimizer；
- 多任务通用 LoRA；
- canonical bank、共享更新子空间或 geometry；
- 限制后续 RL 的 mask、metric、radius 或 learning-rate generator。

共享的结构约束只有预先固定的 LoRA target modules、rank、alpha、dropout 和参数量。Writer 输出该完整空间中的 task-specific initialization；task-local RL 可沿这个 LoRA 参数空间的任何梯度方向原位更新。

## 2. 任务、episode 与可见信息

### 2.1 LIBERO-90

当前只使用 LIBERO-90 的 90 个任务。LIBERO-Spatial/Object/Goal/Long 等十任务 suite 是 LIBERO-100 的标准 benchmark suites，与 LIBERO-90 的数据组织和任务分布用途不同；本阶段不把它们混入 70/10/10。

每个 LIBERO-90 task 有 50 条 teacher episode。每条 episode 是一条完整的成功机器人示范轨迹，包含：

- 完整时序 camera observations；
- robot/state/proprio observations；
- teacher actions；
- reward/done/terminal 和 simulator state；
- task language 与相关 metadata。

数据集中的 teacher episodes 是成功示范。训练 batch 中的“80”一类数字指从轨迹中抽出的 observation/action chunks，不是 80 条 episode；多次 optimizer step 对同一 50-episode corpus 重采样是标准 VLA 微调方式。

### 2.2 EMBER 信息墙

在 source/train tasks 上，所有 50 条 episode 的 action 可作为监督，因为这里本来就是训练 Writer 的地方。Writer 自身的输入仍只有：

- 完整任务语言；
- 每条 episode 的 action-hidden 视频序列；
- episode boundaries/order。

action、proprio、reward、terminal、hidden normalization、task ID 和文件名不能进入 Writer 输入。

同一条 source episode 可以同时贡献：

1. action-hidden video 作为 Writer 输入；
2. observation/action chunk 作为 functional loss 的训练样本。

这不构成泄漏，因为 action 只出现在 loss target，永远没有送进 Writer。当前合同不再做人为的“0–7 条 episode 看视频、8–39 条 episode 做监督”切分，也不把 12/32/40 条当固定规模；每任务 50 条都可用。

在 validation tasks 上，Writer 冻结且只能看到 language + action-hidden video。目标 task 的 action 不得用于生成或选择 EMBER LoRA；目标 reward 只可用于预声明的 task-local LoRA RL。test 的 language/scene/specification factors 只用于 outcome-blind split；全部方法、checkpoint、预算、selection rule 和 baseline 冻结前，不运行 test policy evaluation、不训练 test direct LoRA，也不读取 test actions、reward outcomes 或成功率。

## 3. 新 70/10/10 split

旧 60/15/15 已退役。新 split 已在读取任何新协议 policy outcome 前，只根据 specification 设计并永久封存为：

- 70 train/source tasks；
- 10 validation tasks；
- 10 reporting-only test tasks。

“同分布”不是简单随机：

- 按 verb、moved object、target receptacle、target relation、order/composition、scene 和 specification-derived difficulty 分层；
- val/test 每个 task-relevant role 在 train 中至少有多个相似任务；
- val/test 不堆积 train 从未见过的 primitive/category；
- 三个集合保持大致相同的类别、scene 和难度比例；
- exact task 与完整 composition 互斥，仍然是未见任务；
- validation/test 各自覆盖多个明显不同类别。

canonical factor table、算法、seed、task IDs、manifest、train-only normalization 和 hashes 位于 `configs/libero90_70_10_10/`。split 不能读取 action、proprio、reward、terminal、normalization 或任何 policy result；不得根据后续 outcome 重新搜索。旧 IDs 和旧 normalization 未被继承。

## 4. Source embodiment base

### 4.1 它是什么

本文后续的 frozen source embodiment base 统一且只指：

```text
通用预训练 lerobot/smolvla_base
    → 在 70 个 train tasks、每任务全部 50 条成功 teacher episodes 上联合训练
    → 得到一个共享、多任务、语言条件的 source embodiment base
    → 训练完成后冻结
```

训练采用成熟 SmolVLA recipe 的 action expert 及必要 state/action/time projection 范围。它不是原始通用 checkpoint、70 个独立模型或人为 Gate；EMBER、target-action-supervised direct LoRA oracle 和 ordinary task-local LoRA RL 都以这同一个 frozen source embodiment base 为共同起点。checkpoint 只能按 train/source evidence 选择，validation/test 不得用于选择或调整 source base。

### 4.2 数据和预算

- 使用全部 70×50=3500 条成功 teacher episode，不再为省时删 episode。
- task/trajectory 混合随机抽样；不得把 70 个任务按顺序各训完再换下一个。
- “使用全部”至少意味着每条 episode 在该阶段第一次完整结束前提供过训练信号；对 frame/chunk SFT 优先完成至少一个全 corpus shuffle epoch。30 分钟是 wall-clock 目标，不是遗漏数据的许可。
- 第一轮目标 wall-clock 约 30 分钟。先做极短吞吐测量，据真实 samples/s 估算 optimizer steps/epochs；不预先写死一个无依据的 step 数。
- 不故意把 base 训弱，也不盲目刷满。选择一个对 source tasks 已形成基本 embodiment competence、但训练成本有限的较早充分 checkpoint。
- source base checkpoint 只看 train/source development evidence，绝不看 validation/test 来挑。

估算总 steps 后，把 checkpoint 放在约 1/3、2/3、3/3 的完整 optimizer-step 边界。每个边界必须对齐 deterministic full-task cycle，确保自上个完整边界以来 70 个 task 至少都提供一次有效训练信号。这里不是“每 30 分钟存一次”。

## 5. 统一 LoRA 合同

第一轮从已经核验的成熟 SmolVLA-compatible support 开始：

- 37 个 target modules；
- rank 32；
- alpha 16；
- dropout 0；
- 1,485,312 个 task-local trainable parameters。

精确 target names 必须在新 source base 上重新核验并随 config/hash 封存；若模型 revision 不变，优先沿用官方默认式的 all action-expert q/v 加 state/action/time projections。不得因为 Writer 生成困难就缩成 last-two q/v 或低 rank。

以下方法必须共享完全相同的 LoRA contract：

- Writer 输出；
- direct task-local LoRA SFT；
- zero/identity-init ordinary task-local LoRA RL；
- Writer-init task-local LoRA RL；
- average/retrieval 等以 LoRA 为输出的对照；
- HyPoGen/DISC-style direct parameter generator。

若未来修改 target/rank，所有 matched arms 同步修改并重新比较。

## 6. Writer 架构

### 6.1 输入长度

架构接受任意有限正数的 episode，每个 episode 接受任意正长度的视频。训练时给每任务全部 50 条视频；“50”是当前数据量，不是模型上限。一条短视频或 50 条以上长视频在接口层都合法。

### 6.2 特征缓存与可训练部分

可缓存冻结 SmolVLA/VLM 已计算的：

- 每帧视觉 tokens/features；
- 完整语言 token embeddings；
- episode boundaries 和顺序。

缓存目的是避免每个 Writer step 重复跑冻结 VLM，不是把整个 Writer encoder 冻结。以下 Writer 部分保持可训练：

- visual projection；
- chunk/temporal attention；
- episode aggregation；
- cross-episode set attention；
- language/video fusion 与 task memory；
- layer/module/rank-aware LoRA decoder。

当前 `VariableEpisodeTaskEncoder` 对完整帧序列做分块层次注意力，对 episode memories 做集合聚合；`CompleteLoRAWriter` 用 module/factor/rank embeddings 和 type-specific heads 生成所有 A/B tensors。不得退回三帧平均或巨大扁平 MLP。

### 6.3 输出

输出是一整套 task-specific LoRA A/B tensors。LoRA-B 可从物理零模板开始，Writer 负责产生完整有效状态。输出不是：

- source adapter 的索引或线性组合；
- task bank coordinates；
- 后续 RL 的 preconditioner；
- 仅几个 layer 的补丁。

## 7. Writer cold start

### 7.1 训练方式

- source base 冻结；
- frozen VLM features 可缓存；
- Writer 在 70 个 train tasks 上联合、多任务、混合训练；
- 每个 global step 的 8 个 DDP ranks 可各处理一个 task，使该 rank 内复用同一 task LoRA；
- 全局按 deterministic no-replacement task cycle 轮换，DDP 聚合跨任务梯度。

访问某个 task 时：

1. 读取该 task 50 条完整 action-hidden 视频的缓存特征和语言；
2. Writer 生成一套 LoRA；
3. 从同一 task 的 50 条 teacher episodes 抽 observation/action chunks；
4. 将 LoRA functional 地装入 frozen source embodiment base；
5. 计算标准 action/flow-matching/behavior loss；
6. 梯度穿过 policy 和 functional LoRA 回到 Writer。

这就是 functional loss。validation 与最终 test 的 target-action-supervised direct LoRA oracle 使用同类行为监督 loss，但直接更新目标 task 的 LoRA tensors；EMBER 则在 source 上更新共享 Writer，使其从 language/video 生成这些 tensors。raw factor MSE 或 oracle-delta imitation 至多作辅助诊断，不能替代 functional loss。

### 7.2 数据、预算和保存

- 使用全部 70 source tasks、50 episodes/task 和兼容的 action/query chunks；当前不要求训练 source direct-LoRA arm。
- 每 task 的 50 条视频全部作为 Writer context；functional sampler 在阶段完成前也要让 50 条 episode 各自至少贡献一个 action/query chunk。无需穷举每个 timestep，并记录实际 consumed chunks 供 matched arms 对齐。
- 初始总 wall-clock 目标不超过约 90 分钟；优先把真实全量数据/合理 steps 压进 8 卡，而不是缩模型或缩 LoRA。
- 先基于吞吐估算总 steps，再在 1/3、2/3、3/3 保存完整 exact-resume checkpoint。
- 每个 checkpoint 边界覆盖所有 70 tasks。
- validation 选择 Writer checkpoint；test 不参与。

阶段结束保存 Writer、optimizer、scheduler、scaler、sampler/data cursor、每-rank RNG、step/episode/consumed-data state，并报告跨类别 validation 的原始成功率。

## 8. Writer-only RL

这是与 task-local RL 严格分开的共享训练阶段，只在 70 个 train/source tasks 上进行：

- base 冻结；
- Writer 生成 LoRA；
- 直接用该 LoRA rollout；
- 生成的 LoRA 不是独立 optimizer variable，不做 task-local 原位更新；
- source reward 只更新 Writer。

科学问题是：reward 是否能让 Writer 以后生成更有 zero-interaction utility、也更利于适应的初始化。

训练同样跨 tasks 混合、balanced/no-replacement；不是逐任务各训一个 Writer。总 wall-clock 目标约 90 分钟，按估算 steps 的 1/3、2/3、3/3 保存。每个候选用冻结 Writer 在合法 validation tasks 上测 zero-interaction 原始成功率。

当前可选方案是 cold-start Writer 和 Writer-only-RL Writer。为了节约后续 RL 时间，先在 validation 选择其中表现更好的一个 initialization 进入主 task-local RL；另一个 arm 可在机制成立后补做，不要求第一轮两套都跑。

## 9. Task-local LoRA RL

task-local RL 本来就是每个目标任务单独适应，这在 meta-learning/adaptation 设置中很常见。它不意味着 Writer 应改成生成一套多任务通用 LoRA。

对每个 validation/test task：

- base 冻结；
- Writer 冻结；
- 先为该 task 生成一套完整 LoRA，或构造 matched zero/identity LoRA；
- 只原位更新这套 task-local LoRA；
- 不创建 shared adapter 或其他参数对象。

主 matched arms：

- Arm A：source base + zero/identity LoRA + ordinary task-local LoRA RL；
- Arm B：validation 选出的最佳 Writer LoRA + 完全相同 ordinary task-local LoRA RL。

两者的 LoRA、RL algorithm、optimizer、hyperparameters、reward、task、init states、policy RNG、interaction K、update U、replay epochs 和 checkpoint schedule完全相同，唯一因果差异是初始化。

记录：

- zero-step success (J_0)；
- success-vs-interactions curve；
- AUC；
- time-to-threshold；
- matched-budget final success (J_K)；
- (J_K-J_0)；
- interaction 数、独立 trajectories、optimizer updates、wall-clock 和 GPU-hours。

### 9.1 时间预算

task-local RL 的预算是所有目标 tasks 的总预算，不是每个 task 90 分钟。初始工程目标是在 10 个 task 上每 task/arm 约 10–15 分钟等价 interaction/update budget，通过 8 卡 task 并行把两臂总 wall-clock 压在约 45–100 分钟。先在 validation profile；若明显不足，可在看 test 前统一提升到约 20 分钟/task。最终冻结的是 interaction K 和 optimizer updates U，而不是松散的 wall-clock。

### 9.2 每任务选择最好 RL checkpoint

允许每个 task 根据预算内 adaptation reward/rollouts 选择自己的最好 checkpoint，因为这本身是部署时的 adaptation algorithm。必须满足：

- 选择规则、K/U 和候选节点先在 validation 固定；
- 用于选择的 reward interactions 全部计入预算；
- 最终性能用新的、未参与选择的 rollout 评估；
- 不得观察最终 test evaluation rows 后反选 checkpoint。

Arm A 和 Arm B 使用完全相同选择规则。checkpoint 典型放在 K/3、2K/3、K。

## 10. Validation 与 test

### 10.1 Validation

10 个 validation tasks 用于：

- 选择 source-independent Writer checkpoint；
- 选择 cold-start 或 Writer-only-RL 版本；
- 选择 task-local RL algorithm/hyperparameters/budget/checkpoint rule；
- 调整 Writer architecture、functional loss 和训练规模；
- 决定是否 exact-resume 增加数据/steps。

每次评估覆盖多个类别，不能用两个近同构 task 代替普适性。

### 10.2 Test

10 个 test tasks 是 reporting-only：

- 只有全部方法、架构、checkpoint、预算、selection rule、baseline、split 和 evaluator 冻结后才打开；
- Writer 只看 language + action-hidden video；
- zero-interaction 先评估；
- 之后可按预声明规则做 task-local LoRA RL；
- base、Writer、encoder 和所有 shared state 冻结。

test reward 可以作为 task-local adaptation 数据，但不得更新 shared Writer/base，也不得用最终 evaluation rollout 选模型。

解封前不得运行 test policy evaluation、训练 test direct LoRA 或读取 test actions/reward/success。解封后统一评估 frozen source embodiment base、validation 选出的最佳 EMBER、target-action-supervised direct LoRA oracle、zero/identity-init ordinary LoRA RL、best Writer-init matched LoRA RL 和已冻结必要强 baseline。test direct LoRA 可使用每个 test task 的全部 50 条 teacher action episodes，但只作 oracle/reference，不进入同信息墙主结论，也不反向影响方法选择。

## 11. 闭环评估标准

使用 LIBERO 官方 benchmark 语义：

- official task suite/BDDL；
- 标准固定 init states；
- 官方 controller、camera、normalization；
- reset 后 dummy settling；
- simulator success predicate；
- 成功即提前结束；
- max episode horizon 400；
- SmolVLA action chunk/execution horizon 50。

旧实验中的 h16 是历史 credit diagnostic，不是新主测试协议。标准评估先覆盖每任务全部 50 个官方 init states。若 flow sampling 的 policy RNG 方差可能左右结论，再用第二个独立 policy RNG，在相同 50 states 上形成 100 episodes/task；不能把不同 checkpoint 拼成一个分母。

同一比较中的 task/init-state/policy RNG/evaluator/precision/horizon/run mode 完全一致。报告：

- per-task successes / independent episodes；
- macro average；
- category/primitive summaries；
- paired differences 与合理区间；
- 原始逐 episode rows。

不使用“每个 task CI 下界严格大于零”一票否决。综合看跨类别总体效应、实际幅度、方向一致性、严重退化和不确定性。EMBER 在多个不同类别 validation tasks 上明显超过 frozen source embodiment base，就证明核心机制有功能价值；暂时低于 direct LoRA 说明尚有优化空间。

## 12. Baseline 与公平性

### 12.1 快速机制阶段

只跑当前问题必需的：

1. frozen source embodiment base；
2. EMBER Writer zero-interaction；
3. direct task-local LoRA SFT；
4. source base + ordinary task-local LoRA RL；
5. best Writer init + matched ordinary task-local LoRA RL。

direct LoRA 在 validation 和最终统一 test 看 teacher actions，是 oracle/capability reference，不是信息匹配的主 baseline；owner 明确要求保留这个参考。

不跑单独的 Language-only Writer 和 Video-only Writer 内部消融。暂不跑 action-expert/full-weight task-local RL 上界。

当前不要求在 source tasks 训练 direct LoRA 或运行 base/EMBER/source-direct 三臂 localization。

### 12.2 最终前沿 baseline

与 held information wall 最接近的候选：

- HyPoGen/DISC-style：source action demonstrations 训练 task-spec-to-parameters，在 val/test 只读 specification；
- ViVLA/DAML-style：source 配对 action 数据训练，在 val/test 读取 action-hidden video 做直接条件化或 learned adaptation；
- direct language+video-conditioned policy：不生成参数但消耗相同目标 language/video；
- average/retrieval/nearest source LoRA：低成本参数初始化对照；
- WTL/RIPT-style reward adaptation：与 EMBER 后续 RL 比较，必须匹配并报告 interaction budget。

这些论文的原始协议并不完全一致；公平比较不是机械复制论文数字，而是在 SmolVLA/LIBERO-90 上实现它们的核心机制，并施加相同 70/10/10 信息墙。

### 12.3 训练预算匹配

“baseline 也最多 90 分钟”不是硬规则。主要匹配：

- 相同 source tasks 与每任务 50 episodes；
- 相同可见字段；
- 对应阶段相同 optimizer update count/consumed examples，或原生方法可比的 interaction budget；
- 对 LoRA arms 相同参数空间和容量；
- 相同 validation/test evaluator。

source baseline 的训练 steps 要与对应 EMBER source stage 等同；target direct LoRA 若训练，则各 target task 使用相同 50 episodes 和相同 per-task steps。val/test 上没有参数更新的 zero-interaction 方法不伪造“训练 steps”。RL 方法匹配 interactions、updates 和 selection rule。额外 wall-clock/GPU-hours照实报告。

direct LoRA 和 task-local RL 是彼此独立的 task states，不应把一个 task
拆成八卡 DDP 后错误平均成“共享 LoRA”。它们优先按 task 并行：一张卡训练
或评估一个/一组 task，十个目标任务分 waves 完成。source base 和共享
Writer 才使用八 rank DDP 聚合跨任务梯度。

## 13. 8-GPU 执行

### 13.1 Training

- 启动前实时检查 8 张 GPU；不得干扰无关进程。
- DDP 通常一张 GPU 一个 training rank；GPU0 不额外加载 controller/model。
- 使用真实 per-GPU batch、global batch、gradient accumulation、feature cache 和 data loading 提升吞吐。
- global batch 改变时按成熟 recipe 合理调整学习率并记录。
- 目标每卡平均约 70GB、预留约 10GB，前提是这些显存承载真实 batch/activation；不以 dummy allocation 伪装利用率。
- 如果 DDP 对某阶段扩展差，保持同一入口，改用独立 task/seed/arm 并行。

### 13.2 Evaluation

评估通常由 MuJoCo/CPU/EGL 限速，显存和 GPU UTL 不会像训练一样持续满载。优化目标是 valid episodes/min：

- 每 GPU 一个 policy CUDA process；
- 每个 policy 复用持久化 env pool；
- 多个 CPU/MuJoCo workers 可以共享该 policy；
- task/arm/seed 使用动态 work queue，避免慢 task 拖住全局；
- rank 与子进程绑定 GPU-local NUMA/EGL；
- 只为少量代表性 episode 渲染视频，其余禁用昂贵渲染；
- 复用模型、LoRA、feature 和 env，不在每个 seed 重建进程池。

多进程数量不要求“一卡总共只有一个 OS process”；要求的是一卡只有一个 policy CUDA model，CPU resource tracker/forkserver/env workers 不应创建额外 CUDA context。GPU0 不能再承担所有 EGL worker。

## 14. Exact resume 与证据

每个训练阶段 checkpoint 必须包含：

- model/Writer/LoRA parameters；
- optimizer；
- scheduler；
- scaler；
- sampler/data cursor；
- 每 rank RNG；
- global step；
- episode/interaction/update count；
- consumed-data identity/cycle position；
- world size/topology 和 config/hash。

operationally sufficient 的 resume 验证是：checkpoint 可加载，模型/optimizer/scheduler/step/data cursor 正确，短 resume 不崩溃，loss/functional behavior 在预声明容差内。与科学状态无关的 ambient RNG digest、日志字节或 telemetry 尾数不能无限阻塞。

每个 stage 产物还保存：

- raw per-episode evaluation rows；
- aggregate metrics；
- exact command/config/revision；
- compute ledger；
- 少量代表性视频和本地 gallery；
- failure packet（若失败）。

当视频、临时 cache 或 checkpoint 积累过多时，只保留 selected/final checkpoint、必要 recovery checkpoint、metrics、hash、raw rows 和代表性媒体；删除可再生 features、重复视频、未选 checkpoint 和旧 longrun wrapper 日志。

## 15. Phase F：合同冻结与最终正式实验

前面的 30/90 分钟训练是真实开发实验，不是“假实验”。validation 可能促使修改 architecture、LoRA config、loss 或 training recipe。Phase F 按以下顺序执行：

1. 冻结 final split、source base、Writer/architecture、LoRA、loss、data、optimizer、steps、RL algorithm、interaction/update budget、checkpoint selection rule、evaluator 和 baseline 集合；
2. 保留从一开始就使用完整 70×50 数据且完全符合最终合同的开发 trajectory 作为正式 seed 1；
3. 只重训因 split、source base、architecture、LoRA、loss、data、optimizer、RL 或 evaluator 改变而失去可比性的 arms；
4. 补齐必要独立 training seeds、尚未完成的 matched baselines 和完整 validation rows；
5. 最后统一打开 test，运行 frozen source embodiment base、最佳 EMBER zero-interaction、target-action-supervised direct LoRA oracle、zero/identity-init ordinary LoRA RL、best Writer-init matched LoRA RL 和已冻结必要强 baseline；
6. test direct LoRA 可使用每个 test task 的全部 50 条 teacher action episodes，但只作 oracle/reference；test task-local RL 只按 validation 已冻结的 reward budget 和 checkpoint rule 适应，最终性能使用 fresh rollouts；
7. test 结果不反向修改方法，不把旧 split、数据子集、旧 source base 或旧合同 checkpoint 冒充正式结果。

当前 Goal 到 SmolVLA + LIBERO-90 的统一 test 为止，不包含 OpenVLA-OFT。

## 16. Phase F 之后的可选 outer learning

完成 Phase F 和当前 Goal 的核心证据后，才可选择探索 source-only reward/meta outer learning：

- inner loop 只在 source tasks 更新 task-local LoRA；
- outer objective 只用 source reward 更新 Writer；
- shared source embodiment base 始终冻结；
- 只用 source 训练、validation 选择，不影响已经冻结并报告的核心 Phase F 结果；
- 有收益时作为后续额外 arm，未实现或结果为负都不否定核心 EMBER。

Writer-only RL 与 outer learning 不能混写：前者没有 LoRA 内层原位更新，后者显式通过或估计 inner adaptation 对未来 Writer initialization 的影响。若未来要给 outer arm 增加 test，只能另行冻结并声明为 Phase F 之后的扩展，不能反向改写核心 test。

## 17. 当前历史证据如何解释

- Gate -1：passed with residuals，不再重跑凑 0.80。
- Gate 0：passed with limited coverage，不再要求“LoRA 必须超过一个已经 source-trained 的 base”。
- 旧 source base 和旧 Writer 都基于 60/15/15、部分 episode 或不同 foundation/source-base 合同；不能直接续到新 70/10/10。
- 历史 full-video Writer 在 16 个旧 source tasks 上对 generic foundation base 得到 base/Writer/direct `0/512, 55/512, 51/512`，说明架构能学到 source 行为。
- 同一旧 Writer 在 validation 上只有 sparse transfer：五任务 base/Writer/direct `0/160, 1/160, 18/160`；额外八任务 Writer `4/256`，全部成功集中于 task 22；task 22 为 `0/32, 4/32, 12/32`。
- 另一组旧 source-trained-base 三臂诊断得到 h16 base/Writer/direct `141/320, 127/320, 137/320`，说明旧结果强烈依赖 base/data/target recipe，不能简单说“EMBER source 一定好、只是不泛化”；该一次性历史诊断不是当前阶段要求。

这些结果必须保留并诚实引用，但新协议要重新训练，不能选择性挑一个旧 checkpoint 当活动起点。

## 18. 下一条 canonical 动作

1. 为 source base 写一个最小 canonical config/runner，复用 sealed split 与现有 LIBERO audit 原语。
2. 八卡短吞吐窗口后，运行约 30 分钟 source base，按 full-task-cycle thirds 保存，train/source 选择 checkpoint。
3. 用标准 LIBERO h50 评估 frozen source embodiment base 在 train development、validation，并在 validation 训练 target-action-supervised direct LoRA oracle；不打开 test。
4. 适配现有 full-video Writer core 到新 split/full-50 feature cache，训练 cold start。
5. validation 选 Writer，随后完成 Writer-only RL 和 matched task-local RL，并冻结算法、interaction/update budget 与 checkpoint rule。
6. 保留完全合规 trajectory，只补受合同变化影响的重训、必要 seeds/baselines；全部冻结后统一打开 test，完成 Phase F。
7. Phase F 之后再按价值决定是否探索 optional outer learning。

任何真实阻塞必须是具体可复现的命令/错误；“还可以写更多 schema、测试或文档”不构成阻塞。

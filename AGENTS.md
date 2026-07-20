# EMBER Repository Instructions

## Authority

本文件和 `docs/execution_brief.md` 是当前活动 authority。与它们冲突的旧 Git 提交、聊天摘要、历史报告和 `docs/expert_plan.md` 只作 provenance，不得驱动运行。

在修改代码、数据、split、环境或实验状态前，完整阅读：

1. `README.md`
2. `docs/execution_brief.md`
3. `task_plan.md`
4. `findings.md`
5. `progress.md`
6. `docs/concept.md`
7. `docs/decisions_and_open_questions.md`
8. `docs/novelty_and_landscape.md`

只有需要追溯思想来源时才读 `docs/origin_and_general_thesis.md`、`docs/prior_work_memllm_lessons.md` 和历史 `docs/expert_plan.md`。不要把历史专家方案里的 bank/geometry/60-15-15 重新带回活动路径。

## Core contract

EMBER 的核心固定为：

```text
language + action-hidden teaching videos
    -> shared Writer
    -> complete task-specific LoRA
```

- Writer 是跨 source tasks 训练的共享 hypernetwork，但每次生成一套任务专属 LoRA；不是生成一套多任务通用 LoRA。
- Writer 输入支持任意有限正数的 episode 和任意正长度的视频。训练用每任务全部 50 条 episode，但架构不得把 50 写成上限。
- Writer 不得接收 action、proprio、reward、terminal、隐藏 normalization、task ID 或文件名特征。
- source 上同一条 teacher episode 可以同时提供 Writer 可见的 action-hidden 视频和 functional loss 的 action label；action 只进入 loss，不进入 Writer。
- 只使用 LoRA。不得添加 bank、basis、shared update subspace、soft geometry、residual escape、mask、metric、radius、learning-rate generator、shared LoRA 或额外 shared trainable adapter。
- Writer、direct LoRA 和相关 matched RL arm 必须使用相同 target names、rank、alpha、dropout、参数量和挂载空间。

## Current data protocol

- 只用 LIBERO-90。
- 活动 split 是 `configs/libero90_70_10_10/` 中永久封存的 specification-only 同分布 70/10/10；不得重新搜索或替换 IDs。
- split 只能使用 task language、role-aware factors、scene identity 和 specification-derived composition/difficulty。不得使用任何 policy outcome、action、reward、proprio、terminal 或 held privileged surface。
- train/validation/test 的 task IDs 必须互斥；val/test 中每个 task-relevant primitive/role 要在 train 中有多个近邻，三个集合的类别和难度分布要相近，同时 exact task/composition 保持未见。
- 每任务 50 条成功 teacher episode 全部可用，不再人为截成 8/32/40 条。
- 每个完整训练阶段结束前，全部声明 episode 都应至少贡献一次有效训练信号；无需把所有 timestep 都当一次独立样本，但 matched baseline 必须匹配 consumed examples/chunks。
- train 用于 source base、Writer cold start 和 Writer-only RL；validation 可用于模型/超参数/预算选择；test 只在方案冻结后打开。

## Stage boundaries

本文后续的 frozen source embodiment base 统一且只指：

```text
通用预训练 lerobot/smolvla_base
    → 在 70 个 train tasks、每任务全部 50 条成功 teacher episodes 上联合训练
    → 得到一个共享、多任务、语言条件的 source embodiment base
    → 训练完成后冻结
```

EMBER、target-action-supervised direct LoRA oracle 和 ordinary task-local LoRA RL 都以这同一个 frozen source embodiment base 为共同起点。它只能根据 train/source evidence 选择；validation/test 不得用于选择或调整它。

1. source embodiment base：完成上述 70×50 联合训练并冻结共享 base。
2. Writer cold start：base 冻结，functional action/flow/behavior loss 更新 Writer。
3. Writer-only RL：base 冻结，生成 LoRA 不原位更新，source reward 只更新 Writer。
4. task-local RL：base/Writer 冻结，只原位更新目标任务的同一套 LoRA；先在 validation 冻结算法、预算和选择规则。
5. held/test：全部方法、checkpoint、预算、selection rule 和 baseline 冻结后统一打开；shared base、Writer、encoders 和所有共享状态冻结，只有预声明的 task-local LoRA reward adaptation 可更新。

source-only reward/meta outer learning 仅可在 Phase F 合同冻结与统一 test 完成后作为可选增强：若执行，inner LoRA 在 source tasks 适应，outer source reward/meta objective 只更新 Writer，不更新 shared base；它不阻塞 Goal complete，也不改写已报告的核心 test。

每个 EMBER 阶段保存产物，并在合法 validation tasks 上报告原始成功率、任务、rollout、seed、数据、step、interaction、GPU 和 wall-clock。训练 loss 或代码完成不是阶段证据。

## Evaluation

- 使用 LIBERO 官方 task suite、BDDL、controller、camera 和 normalization。
- 环境最大 horizon 400；reset 后保留官方 dummy settling；成功即终止。
- SmolVLA 主 action execution horizon 为 50，和其 50-step action chunk 对齐。旧 h16 只属历史诊断，不得作为新主指标。
- 所有 RL 更新与 adaptation checkpoint 选择 rollouts 必须通过 LIBERO 官方 reset/BDDL 随机化机制生成初态，禁止从每任务 50 个固定 `.pruned_init` states 取样。
- matched zero/identity-init 与 Writer-init 两臂必须使用相同 task、env seeds 和随机初态序列；保存可恢复的 worker RNG/seed schedule 与 interaction cursor。
- 每任务全部 50 个固定 `.pruned_init` states 只用于与 RL 数据分离的 fresh evaluation；仍采用官方 dummy settling、horizon 400、成功即终止。若 policy sampling 方差仍影响判断，再加第二个独立 policy RNG。
- 同轮 fresh evaluation 必须匹配 task/init-state、policy RNG、evaluator、precision、horizon 和运行模式。
- 不拼接不同 checkpoint 形成分母，不把 replay transition 当独立 episode。
- checkpoint/model/hyperparameter 只能在 validation 选择；test 最终 evaluation rows 不参与选择。
- 在全部方法、架构、checkpoint、预算、selection rule 和 baseline 冻结前，不运行 test policy evaluation、不训练 test direct LoRA，也不读取 test actions、reward outcomes 或成功率。
- 最终 test 才统一运行 frozen source embodiment base、最佳 EMBER、target-action-supervised direct LoRA oracle、zero/identity-init ordinary LoRA RL、Writer-init matched LoRA RL 和已冻结强 baseline。
- test task-local RL 可以依据预算内 adaptation reward 选择该 task 的 checkpoint，但选择规则、interaction budget 和 fresh post-selection evaluation 必须先在 validation 冻结。最终评估 episode 不参与选择。

## Baselines

快速核心对照：

- frozen source embodiment base；
- EMBER zero-interaction Writer LoRA；
- direct task-local LoRA SFT（目标 action 可见的 oracle/reference）；
- source base + zero/identity LoRA + ordinary task-local LoRA RL；
- validation 选出的最佳 Writer init + 完全相同 ordinary task-local LoRA RL。

不要运行独立的 Language-only Writer 或 Video-only Writer 内部 arm。最终前沿 baseline 可包括 HyPoGen/DISC-style language-to-parameters、ViVLA/DAML-style action-hidden-video adaptation、direct language+video conditioned policy，以及必要的 WTL/RIPT-style reward adaptation。避免把核心阶段已运行的 Base+RL 或 EMBER+RL 换名重复运行。

## Time, compute, and checkpoint rules

- 当前最多使用 8 张 A100 80GB；启动前实时检查，绝不干扰无关作业。
- 训练尽量一张卡一个 DDP rank；不要在 GPU0 额外放 controller/model 进程。
- 训练通过真实 batch、data/task parallelism 和缓存使每卡平均约 70GB、预留约 10GB。不要用 dummy tensor 填显存。
- 评估通常受 MuJoCo/CPU/EGL 限制；一张卡一个 policy CUDA process，持久化 env pool，把 task/arm/seed 动态分片到 8 卡。多个 CPU worker 不等于多个 GPU model。
- source base 第一轮约 30 分钟；共享 Writer 阶段通常总 wall-clock 不超过约 90 分钟；一次盲态科学反馈通常不超过 1–2 小时。
- 先估算总 optimizer steps，再在约 1/3、2/3、3/3 保存 checkpoint；不是机械每 30 分钟保存。
- shared multi-task 阶段使用 deterministic no-replacement task cycles。每个完整 checkpoint 边界前，70 个 train tasks 都至少提供一次有效信号。
- task-local RL 的 wall-clock 是所有目标 tasks 的总预算，不是每个 task 90 分钟。先在 validation profile，再统一冻结每 task interaction/update 预算。
- checkpoint 保存 model、optimizer、scheduler、scaler、sampler/data cursor、每 rank/worker RNG、env seed schedule、interaction cursor、step、episode、interaction 和 consumed-data state；不足时从同一 checkpoint exact-resume。

## Engineering style

- 真实运行优先；只为下一段实验的启动、机械正确、恢复或解释写工程。
- 禁止第二套 runner、未来猜测抽象、重复 schema/validator 或论文级脚手架。
- smoke 只查 shape、OOM、NaN、gradient flow、冻结对象、resume、数据隔离；不读小分母性能。
- 新协议尚未封存前，不得把旧 60/15/15 config 或旧 checkpoint 偷换成新协议输入。
- meaningful state change 后简洁更新 `task_plan.md`、`findings.md`、`progress.md`。
- 可视化保留代表性视频和一个 gallery；累计过多时删除可再生临时缓存、未选 checkpoint 和重复视频，保留选定模型、原始 episode rows、metrics、config/hash 与 failure packet。

## Git and privacy

- EMBER 独立于 MemLLM，不引用其实现或数据。
- 不提交 dataset、weights、checkpoint、缓存、凭据、私有主机信息或大文件。
- 每个可复现里程碑验证后提交、推送；只保留一条 canonical 活动路径。
- Git 历史是已退役 runner/config 的 provenance。不要为了“恢复”而把旧可执行树重新复制回主线。

# EMBER Task Plan

最后更新：2026-07-28 UTC。

本文件只保存尚未完成的长期闭环与当前执行顺序。历史实验过程见
`findings.md`、`progress.md` 和 Git；实时进程见
`docs/active_session_handoff.md`。不得从历史 ledger 中恢复已退役 recipe、
runner、split 或 GPU 权限。

## 长期完成定义

只有以下核心项全部完成，长期 Goal 才可完成：

1. 使用过滤后的 LIBERO-90 corpus 训练并冻结共享 π0.5-LIBERO source base；
2. 在固定 24 train / 8 validation 上完成并选择 AS-Writer、RL-Writer（若成立）
   与 corrected mixed-task Source-SFT；
3. 完成 source/seen、correct/same-task-other/wrong/shuffled/reversed 机制证据；
4. 合并 validation 后在 32 source tasks 上从规定初态重训已选方法；
5. 完成 final seen comparison 与 8-task zero-interaction test；
6. 在 8 test tasks 上完成 identity/AS/RL Writer 三臂 task-local RL；
7. 用 8 test tasks × 50 action episodes 联合训练一套 shared target-action
   privileged oracle；
8. 原始 rows、逐 task counts、learning curves、seeds、interaction/data counts、
   GPU-hours、参数量、runtime 与关键 hash 齐全，代码验证、commit、push。

ViVLA-style matched reproduction 和 source-only outer learning 是核心闭环后的
可选项，不阻塞 Goal complete。

## 已封存基础

- [x] 固定 LIBERO-Spatial/Object/Goal/Long 40-task benchmark 和 24/8/8 split。
- [x] 完成 LIBERO-90 × target40 的 3,600-pair specification-only audit：
  排除 19 个 exact semantic/composition 重合 tasks，封存 71 active source
  tasks × 50 successful episodes。
- [x] 从 generic `lerobot/pi05_base` fresh 训练 1,000-step shared source base；
  raw step1000 在 target40×8 screen 为 `46/320`，覆盖 13 tasks 与四 suites。
- [x] 冻结 source-only action/state normalization、source policy、tokenizer、
  model/data manifests 和 canonical π0.5 evaluator。
- [x] evaluator 支持 cost-balanced dynamic queue、persistent model/env、
  Writer per-rollout LoRA、无放回 video schedule 与逐 row paired RNG evidence。
- [x] v4/v5/v5.1 失败根因和 v5.2 五臂成功证据已封存；旧可执行路径已退役。

## Phase C：v6 AS-Writer development

- [x] 封存
  [`docs/action_forecast_writer_v6_design.md`](docs/action_forecast_writer_v6_design.md)：
  Task-Grounded Semantic Set + Visual-Transition Procedure。
- [x] 在唯一 canonical Writer/runner 内实现 v6，参数 `10,775,296`，step0 public
  LoRA 精确 identity；不保留 v5.3 平行 executable path。
- [x] 实现 task-complete macro：4 ranks × 6 tasks/rank、每 task 一视频一 LoRA
  和 B20 queries、24 tasks 等权、前 5 次 `no_sync`、一次 DDP sync/AdamW。
- [x] 实现 selected-video cost balancing、rank 内 long-first、跨 macro rank
  rotation、macro-boundary checkpoint/resume 和 every-25 retention。
- [x] GPU4–7 最长 105-frame B20 profile 连续 3 macros finite；稳态约
  `25.793 queries/s`、`193.447 macros/hour`，选择 B20，不触发 B16。
- [x] step1→3 resume smoke 恢复 task/video/query/LR/cursor；真实
  visual-transition gradient 可达。
- [x] fresh macro0→200 正式段完成：200 条连续 finite metrics、8 个 every-25
  checkpoint、24-task 等权消费与终点全文件 SHA 均已核验。
- [x] 在 GPU4–7 对 macro50/100/150/200 做并行 fixed correct400；每卡一个
  checkpoint，6 Writer generators + 6 persistent workers，视频 50 条无放回，
  全局 long-first。结果为 `114/77/120/129`，paired 输入合同全部通过。
- [x] 选择 macro200 为 absolute observed-best；其 129/400 仅覆盖 5/8 tasks，
  与覆盖 7/8 tasks 的 macro150（120/400）差异不显著，保留 breadth 风险。
- [x] 对 macro200 做 correct/same-task-other/wrong/shuffled/reversed
  full400：`129/131/108/111/105`；same同档，correct对后三臂 paired
  `p=.011/.0198/.00094`，方向门通过但 margin 弱于 v5.2。
- [x] 完成 macro200 的 16-reference 内部传递分析：顺序差异由新增
  visual-transition 路径进入 Procedure，并在 fixed-Core 反事实下传到
  effective LoRA/action；无 Semantic Core 顺序旁路。相对 v5.2，Procedure
  差异更强但下游 LoRA/action 差异更弱，需由续训判断是成熟度还是新瓶颈。
- [x] exact-resume macro200→400 与 macro250/300/350/400 correct400 已封存；
  后四点为 `117/118/125/125`，均未超过 macro200=`129`。第二小时提升部分
  breadth但aggregate不涨，能力继续在tasks间迁移；不继续同一full-24 recipe。
- [x] v6拓扑与机制证据封存：Semantic Set、Visual Transition、Causal
  Procedure职责成立，macro200五臂通过方向门；但absolute、margin和跨task
  稳定性未达最终满意门，后续训练粒度/下游compiler仍需改进。

## Phase D：corrected mixed-task Source-SFT

- [x] 从 frozen source base 实现唯一canonical corrected rank-128 Source-SFT；
  每个physical batch一次普通同步forward/backward/clip/AdamW，无gradient
  accumulation或Writer式micro-round。
- [x] 每rank physical batch包含全部24 tasks等量样本；按
  task→episode→chunk分层无放回周期采样，跨rank row不重复、exact resume，
  task-balanced普通batch mean。
- [x] GPU4–7 B144真实fresh step1→resume step3通过；每步全球576 queries、
  峰值allocated/reserved `60.69/74.07GB`，稳态`34.52–36.35 queries/s`。
  B144稳定，未触发B120 fallback。
- [x] 用sealed config从identity fresh训练step0→225（约一小时训练body，
  冷加载另计），每25步checkpoint；225条metrics连续finite，9个checkpoint
  和完整resume state均已核验。
- [x] GPU4/5/6/7各加载step50/100/175/225之一，并行完成四个fixed
  validation correct400；结果为`60/75/77/56`，每点400 rows、36 shards、
  6 workers、零错误，paired seeds和noise prefix完全一致。
- [x] 从step225 exact-resume到450并完成12点dense correct400；
  step400/425为`109/107`同档，step450降到`74`且paired显著，封存full-24
  observed-best step400=`109/400`，不再续训该recipe。
- [x] 实现global-8 cyclic mixed替代sampler：4 ranks×2 tasks、每update
  8个disjoint tasks、连续3 updates完整覆盖24 tasks；保持B144/global576、
  rank-128 LoRA、LR/scheduler与平均task/sample clock不变。
- [x] GPU4–7完成global-8 B144 fresh0→3→resume6 profile；两轮完整cycle、
  3,456 query identities唯一，稳态`36.27–36.38 queries/s`，峰值
  allocated/reserved `60.69/74.07GB`，无OOM或nonfinite。
- [ ] 在clean/pushed main上启动global-8 fresh step0→240，保留每30步
  checkpoint并评测step60/120/180/240；除非可信下降，否则续到480。
- [ ] 为global-8在validation找到observed-best并看到充分峰后证据；报告
  steps、consumed examples、GPU-hours、参数量与搜索上限。
- [ ] 与 v6 使用同一 frozen source base、normalization、policy interface 和
  validation rows；不机械匹配 optimizer steps。

## Phase E：matched π0.5 action one-shot baseline

- [ ] 在看 outcome 前，每个 validation task 用固定 seed 从 50 episodes 中抽
  1 条 action episode。
- [ ] 对每 task 只训练一次 one-shot LoRA，不做 50 次 one-shot。
- [ ] EMBER 比较臂使用与该 episode 对应的 action-hidden video；保持 task、
  state、env/policy RNG 和评测预算 paired。
- [ ] 比较 absolute performance、训练/适配 wall、GPU-hours、action supervision、
  trainable parameters 和 deployment-time forward 成本。
- [ ] EMBER 只看 video 且一次 Writer forward，因此不把“必须绝对超过 action
  one-shot”设成唯一成立条件；若能超过则作为更强结果。

## Phase F：RL-Writer development

- [ ] 从 v6 架构规定初态做独立、短且 task-balanced AS cold start；不得从完整
  AS observed-best 继续。
- [ ] 直到 24 个 development-train tasks 各在 official random-reset rollout
  中至少成功一次，才关闭 action 入口并进入 pure-reward。
- [ ] reward 阶段只用官方 binary reward/success；不读 object pose、
  privileged shaping、validation/test reward 或 `.pruned_init`。
- [ ] 保存 Writer/optimizer/scheduler、worker RNG、env/policy seed schedule、
  interaction cursor、video schedule、完整 reward ledger 与 exact-resume state。
- [ ] 用 correct/wrong-video、source/seen 和 absolute validation 选择；
  RL 不能用来掩盖 AS 的绝对性能或逻辑漏洞。

## Phase G：32-source final 与 zero-interaction test

当前 focused v6/SFT/one-shot/RL 完成并向 owner 汇报后才进入；不得自动启动。

- [ ] 将 8 validation tasks 机械合入形成 32 source / 8 test。
- [ ] AS-Writer、Source-SFT、RL-Writer（若成立）各自从规定初态单 seed 重训。
- [ ] 打开 test 前完成 final seen comparison。
- [ ] zero-interaction test 比较 source base、Source-SFT、AS-Writer、RL-Writer
  及 correct/wrong-video；每 rollout 随机抽正确 task 的一条 teacher video。

## Phase H：test-only task-local RL 与 oracle

- [ ] test 打开后，identity/AS/RL Writer 三臂在每个 test task 上使用相同
  official random-reset sequence、同一 cohort video 和可比预算训练到各自最佳。
- [ ] fixed 50 `.pruned_init` states 只作训练分离的 fresh evaluation。
- [ ] 三臂结果封存后，才读取 8 test tasks × 50 action episodes，联合训练一套
  shared multi-task target-action LoRA oracle；不是 8 套 task-local LoRA。

## 每次 GPU 运行前

- [ ] 只读核验 workspace/branch/HEAD/origin/status、现有进程和输出根。
- [ ] 只查询并只使用物理 GPU4–7；GPU0–3 不进入命令。
- [ ] 检查 `/data/ymdai` 当前占用、峰值新增量与 500GB hard cap。
- [ ] 封存 exact command、config/model/data paths、output root、process topology、
  checkpoint cadence、停止与继续判据。
- [ ] 正式昂贵 run 前做 live GPU preflight；不杀、暂停、reset 或干扰他人进程。
- [ ] output 不覆盖；resume 核验完整 state。stage stop 只可在 sealed total axis
  内单调延长，其它 scientific contract 变化必须 fresh。
- [ ] 评测按 `episodes × horizon` 动态调度；所有 worker 先处理 long task，
  long 耗尽后再取其它 task；任何 checkpoint/GPU 分配都遵守。
- [x] 等训练/rollout 时推进不污染运行的代码、分析和仓库清理；已退役旧路径
  18,853 行、约 3.8 MiB 仓库缓存和 87.49 GB 已完成评测 LoRA 中间 cache；
  活动运行环境、checkpoint、rollout rows/results 和 contract 证据完整保留。

## 当前继续/停止判据

- absolute 低于可信满意区间或尚未形成充分峰后下降时，继续训练、诊断或 fresh
  架构实验；不能因单点略涨结束。
- correct 提升若依赖 wrong video、shuffle/reverse、validation 泄漏或其它违反
  EMBER 映射的捷径，一律判为机制失败。
- focused AS-Writer的absolute硬门统一为
  `correct400 >= max(150, corrected Source-SFT observed-best + 30)`；两个条件
  必须同时满足。还必须same≈correct、correct显著优于wrong/shuffled/reversed、
  多tasks共同贡献并通过独立RNG/video permutation复测。
- `122/400`旧八卡Source-SFT只是背景，不是独立硬门；`+30`是最低研究里程碑，
  不是达到后强制停止。新corrected Source-SFT必须重新训练和选峰后再比较。

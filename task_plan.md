# EMBER Task Plan

状态：2026-08-20 **active；Gate 2 qualified，Writer macro10 formal与generation profile已完成，当前在Phase 3 matched推断裁决。**

## Goal

全面落实第二轮独立专家意见与owner随后确认的授权：停止在当前LMMPC 320-cell joint-moving compiler上做增量修补，先建立
跨任务、policy-functional且坐标固定的adaptation manifold，再学习exact language与action-hidden有序视频到compact
skill code的推断，并以train/meta-task closed-loop credit改善一次性完整LoRA。所有方向必须进入实现、实验裁决或明确的
条件转向，不得静默遗漏；最终选择仍由held zero-interaction closed-loop absolute、breadth、相邻稳定性与视频净因果增量共同决定。

## Done when

- `docs/expert_second_round_implementation_ledger_20260819.md`中A--N及五个替代方向均不再是`queued`：核心路线已有实现和
  formal证据，条件方向已有触发条件、实际裁决或实施结果；
- non-held meta-task allowlist、语义去重、source/meta/architecture-validation/final-test边界与provenance可由仓库直接复核；
- 目标task的task-local/rank16 ceiling与source primitive coverage已被测量；若source prior不足，已在不触碰held
  action/reward梯度的前提下强化并重建基线；
- 固定decoder在task-level leave-out上能把compact code解码为policy-effective完整38-target LoRA，而不是只在raw A/B
  L2或train-task free-Program上可达；
- learned language-only、video-only、language+video、first/final/endpoints/middle/order controls可以在同一推断与评测
  合同下比较，视频净增量不会由架构性identity开关自证；
- language prior + video posterior保留每视频内部有序过程与完整Action probe结构，并在声明Dynamic-K时真实覆盖各K；
- functional warm-start后的train/meta closed-loop outer objective已有严格对照，held部署仍只在rollout前生成一次完整LoRA；
- 每个正式候选均报告single-checkpoint paired400、per-task/suite、breadth@1/@5/@10、suite floor、retained/gained/lost、
  churn、相邻success-set重合、same-task不同视频以及process-identifying controls；
- 方法若通过则在多个task-level split复现并冻结后进入sealed held诊断和最终Test；若按预注册停止条件失败，则完成广义
  video-to-LoRA路线的证据化裁决并实际转入最有根据的替代问题，而不是继续小扫；
- 验证后的代码、配置、文档与remote-safe证据均已及时合并到`main`并推送，formal artifacts保留完整provenance。

## Fixed scientific boundaries

- deployment输入仍是exact task language与一条或多条same-task、action-hidden、内部有序teacher videos；rollout前只运行
  Writer一次，输出并部署唯一一套完整38-target LoRA；source policy冻结；
- validation/test actions或reward不产生梯度。train24及显式allowlist中的non-held LIBERO-90 meta tasks可使用teacher
  action alignment、task-local experts和simulator reward训练共享表示、decoder与Writer；
- learned language-only adapter是必要baseline；它不取消“视频必须提供必要Value增量”的最终科学要求；
- shared language/base prior + video-conditioned residual是允许的，只要rollout前merge成唯一完整LoRA；不得部署第二adapter、
  held expert route、task-ID字典、挑video、checkpoint fusion或多LoRA平均；
- sealed held action/reward只可用于预注册、冻结、无梯度、无checkpoint选择的诊断；Test默认留到方法冻结后；
- 当前LMMPC增量路线停止。可复用其严格评测、per-video/set数据流、Core/Procedure概念、Core-addressed Reader、bounded
  K-set/M2P等经验证机制，但不把旧320-cell grid或moving FactorHeads当成后继架构前提；
- closed-loop absolute首先选择方法；functional fingerprint、code reconstruction、norm/rank/cosine和内部margin只作定位；
- 一次尽量只改变一个主要因果变量；不以rank/LR/scale/seed/dtype小扫挽救明确坏结果，不添加hash、重复forward、
  大量防御性检查或为正常BF16/TF32低位差异牺牲吞吐；
- formal训练/评测来自clean pushed commit的detached frozen worktree。日常结构性开发在`codex/<topic>`隔离分支完成，
  验证后及时合并`main`并推送；
- launch前live检查gpu01/gpu02与独立存储quota。资源紧张时最多跨两个节点合计6张合适GPU；只有空闲资源明显充裕时
  才可合计最多8张。允许与低显存、低util进程安全共驻，不等待凑满、不占卡、不抢占或干预他人进程。

## Evidence ladder and decision gates

1. **Contract gate**：信息墙、task split、allowlist、pairing、single-LoRA和冻结source policy正确；失败只修合同。
2. **Functional realizability gate**：fixed decoder在未参与decoder拟合的meta tasks上复现expert的closed-loop功能；raw
   LoRA reconstruction不构成通过。
3. **Inference gate**：language/video可预测leave-task-out code，full video相对learned language-only与first+final有净增量；
   失败先定位source coverage、process representation或code inference，不解冻decoder追逐latent gauge。
4. **Outer-credit gate**：functional warm-start固定后，train/meta closed-loop objective相对matched functional-only提高
   held absolute/breadth/retention且不破坏process controls；否则淘汰该outer objective实现。
5. **Method qualification gate**：single-checkpoint paired400达到有意义absolute并同时满足breadth、suite floor、Long、
   same-video robustness、相邻稳定与视频因果资格；约145可构成有价值结果，150+仍需上述联合证据。
6. **Stop/redirect gate**：更强clean source prior、更多meta tasks、fixed functional decoder、process representation、outer
   closed-loop objective和多split评测都完成后仍`<=120`、top3贡献`>80%`、full不优于first+final、Long无收益、
   same-task换手且checkpoint漂移，则停止广义zero-interaction video-to-LoRA并转入ledger中证据最强的替代方向。

## Phase 0 — Successor contract、覆盖账本与现状基线（active）

- [x] owner确认train/meta experts、learned language baseline、outer RL、non-held LIBERO-90、sealed diagnostics与
  mergeable base+residual均被允许；
- [x] 从最新`main`建立`codex/functional-adaptation-successor`独立worktree；
- [x] 建立A--N及五个替代方向逐项ledger，登记依赖、触发条件、证据与状态；
- [x] 扫描现有`expert_manifold`、Writer、reward、evaluation和数据合同，标记可复用owner、退役路径与新模块边界；
- [x] 审计LIBERO-90与target40语义映射，产生显式non-held meta-task allowlist与去重证据，不复制大数据；
- [x] 定义task-level交叉验证split、process-control panel、learned language-only基线和source/task-expert ceiling协议；
- [x] 完成结构基线检查，冻结后继design与最小首轮实现范围，再进入Phase 1。

**Gate 0：** 数据边界、功能decoder训练/leave-out边界和每条专家意见的去向必须可审计；不能以“后续再考虑”代替状态。

## Phase 1 — 数据可识别性、ceiling与process controls

- [ ] 在现有成功expert/adapter上建立统一policy-functional probe panel，优先复用action response、flow response、policy
  Jacobian response与stage behavior；只保留能预测closed-loop等价性的必要表示；
- [x] 用train/meta leave-task-out测量task-local LoRA ceiling与source primitive coverage，分reach/grasp/place/open/toggle/
  multi-object sequence/recovery，区分“Writer不会推断”与“source不存在能力”；
- [x] 实现learned language-only、video-only、language+video共享评测面；no-video不得再由结构强制identity；
- [ ] 补first-only、final-only、first+final、endpoints-fixed-middle-shuffle、monotone-sparse、flow-only、static-only、
  robot/object mask、同endpoint异procedure、跨view、paraphrase、同对象异目标与stage success中当前数据可诚实支持的项；
- [x] 对暂时缺少配对数据的controls登记具体数据缺口并在non-held meta pool构造，而不是伪造对照；
- [x] 若source primitive/target ceiling不足，先强化clean source policy并重新冻结baseline；否则不无条件扩大source训练。

**Gate 1：** full video是否含超出语言与端点的可识别过程、source policy是否具有目标控制能力必须分开回答。

## Phase 2 — Policy-functional manifold与fixed decoder

- [x] 将完整PI0.5 flow decoder拟合收敛为单一formal/exact-resume入口，冻结56/15 task-equal schedule及下游authority门；
- [x] 在train24与新增non-held meta tasks训练/整理task-local successful adapters；每个任务允许多成功adapter用于估计
  功能等价类，不把raw A/B gauge当标签；
- [ ] 用统一probe panel形成每个adapter的functional fingerprint，检验同功能不同参数与不同功能的可分性；
- [x] 学习whitened/gauge-fixed compact code与`code -> complete LoRA` decoder，固定decoder后做task-level leave-out；
- [ ] 比较fully fixed与有明确two-timescale/EMA合同的decoder；默认主线为fully fixed，只有固定版明确欠拟合才启用慢更新；
- [ ] 分析shared language/base prior + video residual是否提高code效率与功能保真；若采用，训练/部署前merge并验证唯一LoRA；
- [ ] decoder若在leave-task-out closed loop不能接近对应expert功能，先修manifold/data，不进入视频推断。

**Gate 2：** 通过的是未见meta task的policy功能，不是train-task reconstruction、free Program或参数相似度。
default fold0还必须先证明uniform-step direct experts相对`646/750` source baseline产生有信息量的跨task功能增量；decoder
以paired source/direct/projected的retained、gained、lost、churn和per-task delta裁决，不能由identity/source高分过门。
若direct增量不足，当前source/meta任务重叠使Gate 2未识别，转入role-disjoint meta-task构造而不是继续训练Writer。

**Default fold0裁决：** direct experts相对source在56/15任务面板分别净增`+247/+38`，因此当前pool具有信息量；
fixed-decoder projected为`2451/2800`与`659/750`，分别保留direct successes的`92.62%/90.79%`。train侧广泛净增，
held侧仅净`+13`且`p=.18208`、只复现54.67% direct gains，故登记为`qualified_pass_to_writer_inference`而非方法通过。
立即进入Writer推断，但多fold fixed-decoder复现仍是后续选择门，不能用本fold privileged free code替代video inference证据。

## Phase 3 — Language prior + video process posterior

- [x] 构建`z_L=f(language)`与`z_LV=z_L+delta(language, videos)`，同时保留video-only反事实；
- [x] 每条video独立编码initial relations、goal relations、contact/events、ordered subgoals、completion conditions与
  uncertainty；跨videos只在event/relation/code-evidence层置换不变聚合；
- [x] 不再平均50个Action tokens；保留flow-time/horizon-phase/noise-probe维，使用phase-specific policy response地址；
- [x] 接入frame-count-matched static-first-repeated与同episode eye-in-hand cross-view；确认现有HDF5无可用mask authority，
  不从teacher state派生mask绕过信息墙；
- [ ] 在train/meta actions可用处加入inverse dynamics、visual transition-to-latent-action与phase correspondence辅助目标，
  held输入继续action-hidden；
- [ ] Dynamic-K若进入论文候选，训练与formal评测真实覆盖各cardinality；不平均frames/raw features/final LoRAs；
- [ ] 固定decoder，仅训练language/video inference；按task-level split报告code与closed-loop泛化。

**Gate 3：** full video必须稳定优于learned language-only与first+final，收益不能只集中Object，Long不得系统性反向。

## Phase 4 — Train/meta closed-loop outer objective

- [ ] 以Phase 3 functional/code distillation为warm start，保持decoder固定；
- [ ] 在train24/non-held meta tasks用simulator rollout reward或经验证progress critic优化video encoder/code posterior；
- [ ] 与matched functional-only、language-only、video-only比较，明确outer RL改变的是shared zero-interaction Writer；
- [ ] reward采样按task等权，保存可exact-resume的sampler/topology状态；held validation/test reward不产生梯度；
- [ ] 若需要stability regularization，约束policy-functional response或已验证成功support，不再用raw parameter gradient
  PCGrad作为默认解；
- [ ] task-local post-generation RL单独保留为扩展评测，绝不混入zero-interaction分数。

**Gate 4：** outer RL必须带来held closed-loop净收益与breadth/retention改善；train reward或critic loss不能选方法。

## Phase 5 — Formal迭代、相邻稳定与多split复现

- [ ] 每个有信息量candidate在clean pushed commit的detached frozen worktree训练；机制smoke后尽快strict paired400；
- [ ] 报告base/language/video/L+V/process controls的严格matched矩阵，含per-task/suite、breadth、floor、集中度与Long；
- [ ] 好结果继续相邻checkpoint，检查same-task different-video retention达到至少90%--95%、Jaccard与suite floor；
- [ ] 首次达到约145且breadth合理时立即补全部因果controls和多split复现，不用继续训练掩盖峰值；
- [ ] 明确坏结果按最早失效接口转回对应phase，不做无关小扫；
- [ ] 只有architecture-validation完成并冻结方法后才进入final Test；sealed held诊断不回流模型选择。

## Phase 6 — 条件方向与最终收口

- [ ] 若静态一次编译在完整主线后触发stop gate，按证据选择并实际启动video->reward/progress、skill composition、
  object-centric plan、runtime video-conditioned policy、video-initialized task-local RL或observation-only offline imitation；
- [ ] richer sensing只有在明确改变纯RGB问题合同后才启动，结果与核心EMBER分开报告；
- [ ] 每条A--N和替代方向更新为`implemented-pass / implemented-fail / not-triggered-with-evidence / superseded`；
- [ ] 将最终设计、formal结果、停止边界和可复用机制写入`findings.md`、`progress.md`与`research_history.md`；
- [ ] 验证、合并`main`、推送远程、清理已集成worktree与临时输出；目标真实完成后再标记goal complete。

## Current next actions

1. 使用已封存macro10 checkpoint与B8 generation authority，在15-task meta-validation固定states上先运行correct、
   learned language-only、video-only、first+final与关键order/static controls的matched screen；共享LoRA cache只在配对合同一致时复用；
2. 对screen报告per-task、breadth、retained/gained/lost、churn、正确视频净增量及task73/Study局部表现；不由训练loss或
   internal code距离选择方法，也不把free held code分数当Writer结果；
3. 若Writer inference出现跨task真实净增量，补相邻checkpoint、same-task-other及多fold复现，并进入train/meta closed-loop outer objective；
   若失败，按language prior、video posterior、fixed decode或process representation的最早失效接口处理，不做小扫；
4. 只有核心A--J/H完成并触发预注册stop gate，才实际启动runtime policy、task-local RL或其它替代问题；这些方向继续保留，
   不因推进速度而静默丢弃。

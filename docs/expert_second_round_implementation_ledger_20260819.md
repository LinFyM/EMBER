# EMBER第二轮专家意见实施账本

更新时间：2026-08-20。本文把第二轮独立专家审查中的诊断、架构批评、方向A--N、替代研究问题和成功/停止证据逐项
映射到当前后继goal。它是覆盖账本，不是把所有候选同时并入一个模型的设计清单。

状态词：

- `active`：当前phase正在落实；
- `scheduled`：已有依赖顺序和进入门；
- `conditional`：只有明确前置证据触发，未触发不等于遗漏；
- `implemented-pass/fail`：已经实施并有formal裁决；
- `not-triggered-with-evidence`：触发条件经证据判定不成立；
- `superseded`：被更直接且已验证的机制取代，并保留原因。

## 1. 核心因果判断

| ID | 专家核心判断 | 当前接受程度 | 落实方式与裁决证据 | 状态 |
| --- | --- | --- | --- | --- |
| C1 | 24个task上的correct-only、同task恒定expert target使video-to-skill统计欠识别 | 高 | 71-task fold0 direct相对source在56/15面板净增+247/+38，当前pool具有功能增量；但source/meta identity仍重叠，fold0只允许进入推断，same-endpoint/different-procedure与多fold/role-disjoint复现仍是显式要求 | active |
| C2 | Program与完整LoRA decoder联合移动造成latent gauge/坐标漂移 | 高 | fold0已完成policy-functional code-to-complete-LoRA decoder训练、冻结与held free-code闭环；held保留90.79% direct successes但净增仅+13且不显著，故只作qualified Gate 2，Writer训练中decoder继续固定 | active |
| C3 | expert-state functional matching与closed-loop success外目标错位 | 高 | functional distillation只做warm start；随后在train/meta simulator上用closed-loop reward/progress credit优化code posterior | scheduled |
| C4 | source skill prior尤其对Long可能不足 | 中高 | 71-task source为2918/3550；direct及projected在56-task Study分别较source净增+117/+99、pick-place净增+126/+102，当前全局强化触发条件不成立；后续只按Writer结果定位局部source缺口 | not-triggered-with-evidence |
| C5 | 当前模型更容易学object/affordance/direction/template而非多阶段过程 | 高 | object/relation/event/subgoal representation与full-vs-endpoints/middle/order controls共同裁决 | scheduled |
| C6 | FactorHead静态range与moving decoder需分别裁决 | 高 | 正确F4仅307/1200；后继fixed decoder在56/15面板保留92.62%/90.79% direct successes并获qualified推断资格，但held仅净+13、task58 loss退化，故不恢复旧FactorHeads，也不把单fold结果冒充range已解决 | active |
| C7 | raw parameter PCGrad不等于功能冲突解法 | 高 | 不把PCGrad作为后继默认；如需稳定约束，使用policy-functional response与retained support | accepted |
| C8 | validation8已被长期使用，不能当全新独立证据 | 高 | 保持固定24/8/8 ID不按结果改；validation8明确作architecture panel，新增non-held meta folds，Test8留到冻结后 | active |
| C9 | 150不是唯一科学门，约145也需breadth/stability/causality | 高 | 继续追求150+，但联合报告absolute、suite floor、breadth、same-video retention、adjacent stability与process controls | accepted |

## 2. 现有机制的保留、重构与退役

| ID | 专家意见 | 当前处理 | 状态 |
| --- | --- | --- | --- |
| P1 | 保留严格evaluation、deployment信息墙与single complete LoRA | 原样继承；只放宽授权train/meta与sealed diagnosis墙 | accepted |
| P2 | 每视频先保序，跨视频再置换不变 | successor按video独立编码、再对完整video summary做集合聚合；4-task训练profile已接通，等待formal closed-loop裁决 | active |
| P3 | Core/Procedure概念区分有价值 | 继承语义/目标与过程/阶段的职责，不强制继承旧tensor拓扑 | accepted |
| P4 | Core-addressed Reader、bounded K-set/M2P有局部正证据 | 作为可复用组件候选；只有与fixed-code接口匹配才保留 | active |
| U1 | language-only被架构强制identity会让no-video对照自证 | learned language-only已独立生成complete LoRA且部署时不读视频；Text/VL Meta-LoRA均固定为0并由authority guard保护，等待formal baseline | active |
| U2 | LoRA rank index不是policy-functional coordinate | 新decoder使用functional code/response地址；rank16只作为最终LoRA参数化 | active |
| U3 | Program与complete LoRA decoder不应持续共同移动 | 主线固定decoder；慢EMA/two-timescale仅在fixed版明确欠拟合时作为对照 | active |
| U4 | 50个Action token直接mean丢失horizon/noise/phase结构 | successor保留完整50-token序列，以phase queries读取并输出phase-specific alignment；等待消融 | active |
| U5 | 时间中心化Value不等于过程表示 | 后继以initial/goal/events/transitions显式编码；centered memory不再承担唯一过程语义 | scheduled |
| U6 | action-in/out由首末layer派生不是已知endpoint correspondence | 不把该对应作为canonical地址；以policy response probe验证地址功能 | active |
| U7 | 最后RMSNorm会抹除cell magnitude | 在复用旧组件时做功能消融；新fixed decoder默认不无条件抹除有意义幅度 | active |
| U8 | 20x16x256 grid复杂且缺少清晰可识别语义 | 停止其增量路线；只复用经过闭环验证的独立机制，不保留平行canonical grid | active |
| U9 | frozen native memory、Dynamic-K、rank16有价值但未证明必要 | 分别由ablation、真实cardinality训练和closed-loop性能决定，不升级为架构信条 | scheduled |

## 3. 专家方向A--H：核心合同内

| 方向 | 内容 | 实施/裁决门 | 状态 |
| --- | --- | --- | --- |
| A | 功能锚定的固定adapter decoder | default fold0已完成71 experts、56-task decoder fit、15-task frozen-decoder code fit及两侧closed loop；Gate 2为qualified而非方法pass，多fold仍必须复现 | active |
| B | language prior + video posterior | 一次性完整LoRA生成、统一evaluator/cache和训练面已接通；4-task profile通过且decoder/VLM保持冻结，当前进入56-task formal与matched矩阵 | active |
| C | object-centric explicit Program | 表示objects、initial/goal relations、contact events、ordered subgoals、completion与uncertainty；用paired controls裁决而非只看latent | scheduled |
| D | 保留完整Action probe结构 | `frame x 50 x hidden`进入phase-specific读取，不再直接mean；等待有/无alignment对照 | active |
| E | train-task closed-loop outer objective | fixed decoder与functional warm-start之后，在train/meta simulator优化encoder/code；held仍zero-interaction | scheduled |
| F | 扩展meta tasks并分离四类数据角色 | 71-task allowlist、5 folds与完整source已建立；direct在56/15面板均产生增量，因此fold0可继续，但source/meta identity未完全分离，多fold与必要时role-disjoint构造仍保留为方法选择门 | active |
| G | process-identifying controls | first/final/endpoints/middle/order/sparse已接通；新增真实首帧等长重复static与同episode eye-in-hand cross-view；HDF5无depth/segmentation，mask不读取state伪造；flow/procedure/paraphrase/goal/stage继续补可信数据 | active |
| H | 强化clean source policy | 71-task source总体82.20%；direct/projected已在Study与pick-place产生大幅正增量，当前无差别source重训不触发，保留局部失败时的定向强化条件 | not-triggered-with-evidence |

## 4. 专家方向I--N：owner已授权或保留的合同变化

| 方向 | owner边界与当前决定 | 进入/停止条件 | 状态 |
| --- | --- | --- | --- |
| I train/meta teacher-action alignment | 允许；validation/test action不训练 | meta-train phase alignment已实现为同task、不同episode的归一化过程相位配对；部署与meta-validation不读action，仍待有/无对照 | active |
| J sealed held actions/reward diagnosis | 允许冻结、无梯度、无checkpoint选择诊断 | validation可在预注册点诊断；Test默认方法冻结后一次性使用，结果不回流设计 | scheduled |
| K runtime video-conditioned policy | 改变Writer-once部署主张，当前不混入核心分数 | 只有A--J/H主线完整后触发广义video-to-LoRA stop gate，才作为明确替代实验 | conditional |
| L generation后task-local RL | 允许但必须与zero-interaction分开 | 先报告初始化分数，再比较达到成功的episodes与base/language/video样本效率 | conditional |
| M shared base adapter + video residual | 允许；rollout前merge为唯一complete LoRA | 作为B的自然实现候选；需证明比单code提高功能保真/稳定且merge不改变行为 | scheduled |
| N RGB-D/proprio/object pose | 会改变当前纯RGB合同，不作为核心结果偷换输入 | 仅在纯RGB路线触发stop gate且owner接受独立研究问题时启动 | conditional |

## 5. 欠识别时的替代研究问题

| ID | 替代方向 | 触发后的首个比较 | 状态 |
| --- | --- | --- | --- |
| X1 | video -> reward/progress | progress generalization与用其训练policy的样本效率；不把reward metric冒充task success | conditional |
| X2 | video -> skill selection/composition | 既有primitive库上的subgoal/order泛化与closed-loop completion | conditional |
| X3 | video -> object-centric plan/state machine | relation/subgoal graph正确性及由冻结executor闭环执行的成功率 | conditional |
| X4 | video -> fast-RL exploration prior | 初始成功、安全失败与达到阈值所需episodes，相对base/language/video controls | conditional |
| X5 | observation-only offline imitation | 多步occupancy matching能否优于一步functional distillation，保持action-hidden test输入 | conditional |

替代方向不会为了“覆盖意见”而在核心路线尚未被诚实检验前并行消耗资源；一旦stop gate触发，必须根据已有失效接口
选择最直接的一项实际推进，不能只在文档中列名。

## 6. Process-identifying control可行性账本

| Control | 所需配对 | 当前目标 |
| --- | --- | --- |
| learned language-only | 相同task/state/env/policy RNG，无视频但允许learned prior | Phase 1实现 |
| video-only | 去掉language Value但不改变视频与decoder | Phase 1实现 |
| first-only / final-only / first+final | 同一真实video重取帧并完整forward | Phase 1实现 |
| endpoints-fixed, middle-shuffled | 同一真实video保留首末帧，仅重排真实中间帧 | Phase 1实现 |
| monotone sparse | 同一真实video按预注册稀疏率保序抽帧 | Phase 1实现 |
| optical-flow-only / static-only | 不泄露动作的动态/外观分解 | static-first-repeated已用真实RGB等长接入；当前无经验证的RGB-only flow owner，不能用帧差冒充optical flow |
| robot-mask / object-mask | 一致mask及不改变task配对的渲染 | HDF5无segmentation/depth；由teacher state重渲染会越过信息墙，当前明确不伪造 |
| same endpoint, different procedure | 成功且端点关系匹配的non-held任务对 | meta allowlist/配对审计后构建 |
| cross-view / paraphrase / same-object-different-goal | 同过程异view/语言/目标的真实配对 | eye-in-hand对agentview已按同episode同帧接通；paraphrase/goal只使用可证实metadata，不按结果挑选 |
| stage-level success | stage boundary或可靠progress标注 | train/meta reward/diagnostic可用；held只sealed诊断 |

## 7. 正证据、负证据与更新规则

核心路线的正证据必须联合包含：fixed-decoder leave-task-out、full优于language与first+final、Long一致正收益、same-task
视频success-row retention至少90%--95%、breadth@5改善、相邻checkpoint稳定、多task split复现和source-policy oracle
可实现性。任何单一internal metric、aggregate p值或一次峰值都不能替代该组合。

广义zero-interaction video-to-LoRA只有在更强clean source prior、更多meta tasks、fixed functional decoder、object/process
representation、train/meta on-policy outer objective和多split评测均完成后，仍同时表现为低absolute、top-task集中、
full不胜endpoints、Long无收益、same-video换手与checkpoint漂移，才进入停止/替代裁决。

每个phase结束更新本ledger的状态、证据路径和结论。负结果只关闭实际检验的组合；未触发的条件方向必须写清触发证据，
不能标成“已完成”。

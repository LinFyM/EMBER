# EMBER第二轮专家意见实施账本

更新时间：2026-08-21。本文把第二轮独立专家审查中的诊断、架构批评、方向A--N、替代研究问题和成功/停止证据逐项
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
| C1 | 24个task上的correct-only、同task恒定expert target使video-to-skill统计欠识别 | 高 | role-disjoint与validation8 250/400排除整体ceiling不足；direct action/JVP仅2/4与1/4后，train24的47/47多成功checkpoint完整occupancy经fit19-only固定坐标达到held5 `5/5` mutual-nearest并通过phase门，已具备重建Decoder资格 | implemented-pass |
| C2 | Program与完整LoRA decoder联合移动造成latent gauge/坐标漂移 | 高 | 统一fingerprints已把train/held std修复到1.000/.7248、平均norm修复到5.570/4.144；held只做train-only PCA变换且不再自由优化，坐标问题implemented-pass | implemented-pass |
| C3 | expert-state functional matching与closed-loop success外目标错位 | 高 | fit19 learner-state aggregation把两套44提高到54/47，证明closed-loop state bank有效；但direct success仍只保留31.08%/27.78%，Jaccard降到.4028。本轮state aggregation implemented-fail，不外推成occupancy无关，后续outer reward仍保留 | implemented-fail |
| C4 | source skill prior尤其对Long可能不足 | 中高 | 71-task source为2918/3550；validation8 local oracle250/400、八task全正且Long41/100，source仅48/400。全局source/local ceiling不足分支不成立；Long trace仍保留第二子目标与保持的局部缺口定位 | not-triggered-with-evidence |
| C5 | 当前模型更容易学object/affordance/direction/template而非多阶段过程 | 高 | macro10 correct131低于reversed134/shuffled133/static132，首轮process inference明确失败；先修functional坐标，再复用同一controls检验显式过程表示 | implemented-fail |
| C6 | FactorHead静态range与moving decoder需分别裁决 | 高 | flow/probe/exact与shared-zero已完整分解；exact仅638且task code相对shared-zero净-2。当前单expert/fingerprint Decoder range没有通过，不恢复旧FactorHeads并停止objective变体 | implemented-fail |
| C7 | raw parameter PCGrad不等于功能冲突解法 | 高 | 不把PCGrad作为后继默认；如需稳定约束，使用policy-functional response与retained support | accepted |
| C8 | validation8已被长期使用，不能当全新独立证据 | 高 | 保持固定24/8/8 ID不按结果改；validation8明确作architecture panel，新增non-held meta folds，Test8留到冻结后 | active |
| C9 | 150不是唯一科学门，约145也需breadth/stability/causality | 高 | 继续追求150+，但联合报告absolute、suite floor、breadth、same-video retention、adjacent stability与process controls | accepted |
| C10 | 旧A/B/C缺少`Text Meta-LoRA + repaired front-end`第四格，不能据此判定语言先验无用 | 高 | 不把旧三臂当完整析因证据。owner后续明确禁止canonical继续使用额外Text/VL Meta-LoRA，因此不补跑已退役LMMPC的D臂；科学问题由共享decoder上的learned language-only、video-only与language+video matched比较正面重做，保留语言prior而非把它架构性置零 | superseded |

## 2. 现有机制的保留、重构与退役

| ID | 专家意见 | 当前处理 | 状态 |
| --- | --- | --- | --- |
| P1 | 保留严格evaluation、deployment信息墙与single complete LoRA | 原样继承；只放宽授权train/meta与sealed diagnosis墙 | accepted |
| P2 | 每视频先保序，跨视频再置换不变 | successor按video独立编码、再对完整video summary做集合聚合；4-task训练profile已接通，等待formal closed-loop裁决 | active |
| P3 | Core/Procedure概念区分有价值 | 继承语义/目标与过程/阶段的职责，不强制继承旧tensor拓扑 | accepted |
| P4 | Core-addressed Reader、bounded K-set/M2P有局部正证据 | 作为可复用组件候选；只有与fixed-code接口匹配才保留 | active |
| U1 | language-only被架构强制identity会让no-video对照自证 | learned language-only已独立生成complete LoRA且部署时不读视频；Text/VL Meta-LoRA均固定为0并由authority guard保护，等待formal baseline | active |
| U2 | LoRA rank index不是policy-functional coordinate | 71-task unified expert-source fingerprints与train-only PCA已实现；held transform-only且尺度接近train，rank16只作为输出参数化 | implemented-pass |
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
| A | 功能锚定的固定adapter decoder | phase表示门通过；fresh Decoder由source21到44/44，learner-state聚合再到54/47，但direct retention、latest breadth与member stability仍未联合过门。保留多成功phase表示与state-bank正增量，不把当前Decoder升级到Writer | implemented-fail |
| B | language prior + video posterior | macro10七臂formal screen已完成并失败：correct131、language130、video134、first+final130、reversed134、shuffled133、static132；旧Writer封存为反事实，不续训，等待新fixed coordinates | implemented-fail |
| C | object-centric explicit Program | 表示objects、initial/goal relations、contact events、ordered subgoals、completion与uncertainty；用paired controls裁决而非只看latent | scheduled |
| D | 保留完整Action probe结构 | official denoised `50x7` action与exact `50x32` JVP均已接通；direct八strata失败后改为全部replan、完整50-token响应和固定功能弧长对应，held5达到`5/5`。JVP维持辅助，Action为当前primary标签 | implemented-pass |
| E | train-task closed-loop outer objective | fixed decoder与functional warm-start之后，在train/meta simulator优化encoder/code；held仍zero-interaction | scheduled |
| F | 扩展meta tasks并分离四类数据角色 | 71-task allowlist与5 folds已建立；source未见的target train24保持19/5边界，47/47 multi-checkpoint成功occupancy在held5通过表示门。后续Decoder仍只拟合fit19，validation8继续architecture panel、Test8 sealed | implemented-pass |
| G | process-identifying controls | first+final/reversed/shuffled/static已完成matched closed loop并共同否定macro10过程优势；validation8 BDDL stage trace已完成并定位Long第二子目标，但只是final-goal代理。其它已接通controls保留到新坐标Writer，HDF5无depth/segmentation且不从state伪造mask | active |
| H | 强化clean source policy | 71-task source总体82.20%；direct/projected已在Study与pick-place产生大幅正增量，当前无差别source重训不触发，保留局部失败时的定向强化条件 | not-triggered-with-evidence |

## 4. 专家方向I--N：owner已授权或保留的合同变化

| 方向 | owner边界与当前决定 | 进入/停止条件 | 状态 |
| --- | --- | --- | --- |
| I train/meta teacher-action alignment | 允许；validation/test action不训练 | target train24 privileged experts的47条successful trajectories已构成完整Action response与monotone functional phase correspondence；fit19拟合坐标、held5固定变换并以`5/5`通过 | implemented-pass |
| J sealed held actions/reward diagnosis | 允许冻结、无梯度、无checkpoint选择诊断 | validation8独立rank16 oracle与step2000-only strict400已完成：250/400对source48/400，八task全正、四suite非零；stage trace同轮收集。没有更新共享模型、选择checkpoint或读取Test | implemented-pass |
| K runtime video-conditioned policy | 改变Writer-once部署主张，当前不混入核心分数 | 只有A--J/H主线完整后触发广义video-to-LoRA stop gate，才作为明确替代实验 | conditional |
| L generation后task-local RL | 允许但必须与zero-interaction分开 | 先报告初始化分数，再比较达到成功的episodes与base/language/video样本效率 | conditional |
| M shared base adapter + video residual | 允许；rollout前merge为唯一complete LoRA | learner-state聚合后direct success仍仅31.08%/27.78%，触发条件再次成立。原始`Delta_shared + D(z)`现落实为rank16内互斥的shared rank12/task rank4，避免full-rank factor相加产生`BA`交叉项；zero code严格退化为shared-only，最终只物化一套LoRA并加入matched baseline | active |
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

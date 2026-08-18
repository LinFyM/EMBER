# EMBER External Review Claim Ledger

状态：2026-08-18 active。本文是`docs/external_review_20260818.md`的逐项执行账本；目标是让专家报告中的每个
重要判断、限定、建议和证据缺口都有明确回应。它不是新架构设计，也不把专家建议门槛自动升级为owner硬合同。

## 1. 状态与证据规则

- `confirmed-code`：当前canonical代码可直接证明；只确认机制事实，不自动确认性能因果。
- `confirmed-artifact`：已由本地原始artifact重算；若尚未形成remote-visible evidence，仍标记对应G项未完成。
- `supported` / `refuted`：有matched实验或直接反事实证据支持/反驳因果主张。
- `underdetermined`：现有证据不足，已登记最小裁决实验。
- `queued`：无条件建议已进入本goal且尚未完成。
- `conditional`：只在专家明确前提成立时执行；前提不成立时必须写`not-applicable`及证据。
- `owner-adjusted`：owner原则改变了实现合同，但不改写历史事实。

每项最终至少包含：原意见、类型、当前裁决、证据或待执行项、反证/边界。专家建议的`+15`、`90%`等阈值按
advisory threshold并列报告，EMBER最终方法仍由single-checkpoint strict paired400、稳定性和视频因果性裁决。

## 2. A — 总体裁决与问题分解

| ID | 专家意见 | 类型 | 当前裁决与执行状态 |
| --- | --- | --- | --- |
| A1.01 | exact language、action-hidden video、Action representation、layer-addressed memory和一次性完整LoRA的职责划分合理 | scientific framing | `accepted`；与owner information wall一致，后续每个实验保持该合同 |
| A1.02 | 当前信息墙与source freeze基本正确 | code/scientific fact | `confirmed-code`；仍需F0a以实际source gradient表补强 |
| A1.03 | action-hidden视频影响内部表示 | experiment fact | `confirmed-artifact`于历史内部扰动证据；不等价于correct视频提升闭环，F0b待完成 |
| A1.04 | Core-addressed Reader有真实closed-loop正收益 | experiment fact | `confirmed-artifact`：matched 104→123；不证明整套方法达标 |
| A1.05 | fresh Writer生成非平凡完整LoRA | experiment fact | `confirmed-artifact`；health指标只作诊断，不作方法资格 |
| A1.06 | 未证明Program是高层过程而非task identity/static cue/carrier | negative claim | `supported`；F0b显示correct与shuffled相当、shuffled-keep-first更高，但wrong/no-video显著更低：视频内容必要，正确中间阶段顺序未成为必要evidence |
| A1.07 | 未证明memory layer/rank index有policy-functional语义 | negative claim | `confirmed evidence gap`；index preservation与functional correspondence分开 |
| A1.08 | 未证明FactorHeads稳定覆盖policy-effective方向 | negative claim | `confirmed evidence gap`；F3/F4待完成 |
| A1.09 | 未证明expert-state loss支持generated-policy occupancy | negative claim | `confirmed evidence gap`；F2待完成 |
| A1.10 | 未证明shared checkpoint保留广泛support | experiment fact | `confirmed-artifact`：123→84→89→87、breadth 8→5→6→4 |
| A1.11 | 未证明correct优于wrong/shuffle/reverse/no-video | negative claim | `mixed/refuted as a joint claim`；correct显著优于wrong/reverse/no-video，但不优于shuffle或shuffled-keep-first |
| A2.01 | frozen hidden后的fresh projected outputs被第二次detach | code fact | `confirmed-code`，见`backbone_memory.py`返回路径 |
| A2.02 | patch grounding与interaction projection因此无functional credit | code fact | `confirmed-code`；F0a须给实际grad表 |
| A2.03 | language projection的逐帧视觉支路信用也被切断 | code fact | `confirmed-code`；其独立text/gate路径仍有gradient |
| A2.04 | Core/Procedure仍能在固定随机投影特征上学习，不能说视频路径全死 | limitation | `confirmed-code`；任何报告不得把detach夸张为video-off |
| A2.05 | optimizer/现有测试没有发现intended fresh modules无梯度 | coverage gap | `confirmed-code`；新增完整gradient regression属于F0a |
| A2.06 | 旧V6没有相同output detach | provenance fact | `confirmed-code/history`；只用于定位，不授权恢复V6 |
| A3.01 | 123低上限与25→50漂移必须分开解释 | causal framing | `accepted`；所有干预分别报告absolute support与retention |
| A3.02 | current123相对143为23 gains/43 losses | experiment fact | `confirmed-artifact`，raw-row remote evidence待G1 |
| A3.03 | current123相对151为23 gains/51 losses | experiment fact | `confirmed-artifact`，raw-row remote evidence待G1 |
| A3.04 | 25→50为13 gains/52 losses且显著 | experiment fact | `confirmed-artifact`；后段低位换手另报 |
| A3.05 | macro25 support高度集中，breadth@1不充分 | experiment fact | `confirmed-artifact`；G9扩展breadth待完成 |
| A4.01 | 首要问题是fresh前端credit断点+correct-only shortcut空间 | ranked hypothesis | `underdetermined`；F0a/F0b/F1裁决 |
| A4.02 | 第二问题是offline occupancy与retention错配 | ranked hypothesis | `underdetermined`；F2裁决 |
| A4.03 | 第三问题是FactorHead reachability/co-drift | ranked hypothesis | `underdetermined`；F3/F4裁决 |

## 3. B — 流水线与四条信息流

| ID | 专家意见 | 类型 | 当前裁决与执行状态 |
| --- | --- | --- | --- |
| B1.01 | language应消歧task/object/relation/goal并作为Query，不得单独写LoRA | method boundary | `accepted`；后续canonical移除Text Meta-LoRA，但保留frozen native language |
| B1.02 | video应提供状态变化、先后关系、阶段、因果和nuisance过滤 | method boundary | `accepted`；F0b验证其是否在当前checkpoint真正发挥该作用 |
| B1.03 | Action probe使用真实image/language prefix且不读取teacher action | code/info-wall fact | `confirmed-code` |
| B1.04 | 50个Action token直接mean会丢horizon内阶段/多模态方向 | architectural hypothesis | `confirmed-code`其mean操作；性能后果`underdetermined`，不在首轮同时改 |
| B1.05 | memory states应提供按policy layer组织的dynamic Value | method interpretation | `accepted as intended role`；functional语义尚未证明 |
| B1.06 | 输出是一套完整38-target LoRA且Writer只运行一次 | code/deployment fact | `confirmed-code/contract` |
| B2.01 | 每帧四流先在同一帧交互，再做视频内聚合，再做K-set聚合 | pipeline fact | `confirmed-code`；逐模块细节由sealed design提供 |
| B2.02 | Core聚合全视频语义，容易受task/static cue主导 | code + hypothesis | `confirmed-code`其输入；“主导失败”`underdetermined`，F0b裁决 |
| B2.03 | Procedure用相邻grounded差分、Action query和causal encoder | code fact | `confirmed-code` |
| B2.04 | Reader中`relative - mean(relative)`代数等于time centering | algebra/code fact | `confirmed-code`；首帧reference最终抵消 |
| B2.05 | Reader去除静态Value但也丢绝对状态/终点/持续目标 | representational consequence | `confirmed-algebra`；是否限制closed loop仍`underdetermined` |
| B2.06 | Procedure只能选择native memory已有Value，不能创造缺失direction | structural interpretation | `accepted`；F4可达性与F0b共同约束该判断 |
| B2.07 | K-set K1恒等，K>1 same-address mean anchor且RMS有界 | code fact | `confirmed-code/tests` |
| B2.08 | K-set的invariance/consistency不证明strict均值提升 | evidence boundary | `confirmed gap`；不把内部指标写成closed-loop收益 |
| B3.01 | Core适合task语义但可能成为identity/static shortcut | hypothesis | `underdetermined`；wrong/no-video/carrier controls待F0b |
| B3.02 | Procedure有真实顺序敏感性 | internal experiment fact | `confirmed-artifact`；方向是否有用待F0b |
| B3.03 | v4 shuffled优于correct警告order sensitivity可学错shortcut | historical fact | `confirmed-history`；F0b必须看方向而非hidden margin |
| B3.04 | 当前最早可证明失败接口是fresh projection→Core/Procedure | causal localization | `confirmed-code`为credit断点；“最主要性能根因”仍`underdetermined` |
| B4.01 | 未发现teacher action/reward/terminal/task-ID/file/old Writer leakage | code/data fact | `confirmed-static`；正式run仍复核provenance与data manifest |
| B4.02 | literal constant/no-video路径保持identity | code/test fact | `confirmed-code/tests` |
| B4.03 | nonzero video可作carrier、language gate决定task方向的条件旁路存在 | structural possibility | `confirmed-code possibility`；实际利用程度待F0b |
| B5.01 | one-way memory mask和source freeze成立 | code/test fact | `confirmed-code/tests`，F0a补实际梯度 |
| B5.02 | memory/Reader/Core/M2P各有独立layer/rank identity | code fact | `confirmed-code` |
| B5.03 | 当前只证明index preservation，不证明functional skill correspondence | evidence boundary | `accepted`；F4和family分析补证 |
| B6.01 | bounded M2P避免历史unbounded overwrite | code/history fact | `confirmed-code/history` |
| B6.02 | M2P输出RMSNorm抹去Program magnitude | code fact | `confirmed-code`；其性能影响`underdetermined`，暂不作为首要变量 |
| B6.03 | action-in/out由首/末expert cell线性派生，不是独立native endpoint memory | code fact | `confirmed-code`；F4按endpoint family单列 |
| B7.01 | 每个FactorHead为256→256→output且末层zero init | code fact | `confirmed-code` |
| B7.02 | 给定head时宽输出处于至多256维末层子空间 | mathematical fact | `confirmed`；不等价于专家LoRA不可达 |
| B7.03 | A=A0/B=0 identity造成B一阶、A零阶信用；zero head使上游信用延后 | autograd fact | `confirmed-code/math`；F0a记录首次信用，F4测实际可达性 |
| B7.04 | macro25→50 heads-only cross-decode变化大于Program-only | artifact fact | `confirmed-artifact`；G5补remote原始结果 |
| B7.05 | 历史同类FactorHead达到143，反驳形式上必然不可行 | counterevidence | `confirmed-history`；禁止未做oracle就扩大decoder |
| B8.01 | B20会跨50 episodes/progress strata轮换并排除teacher episode | sampler fact | `confirmed-code` |
| B8.02 | functional LoRA autograd链没有发现结构性计算错误 | code audit | `confirmed-code`；F0a仍查实际module credit |
| B8.03 | expert-demo occupancy与generated-policy occupancy不同 | distribution fact | `confirmed by contract`；是否解释漂移待F2 |
| B9.01 | 流水线有真实新机制，但当前不能声称高层过程理解 | overall verdict | `accepted`；最终由F0b与closed-loop结果更新 |

## 4. C — 历史证据边界

| ID | 专家意见 | 类型 | 当前裁决与执行状态 |
| --- | --- | --- | --- |
| C1.01 | Core-addressed Reader相对matched旧Reader有+19收益 | positive evidence | `confirmed-artifact` |
| C1.02 | GOMQ learned memory相对fixed memory有135→151收益 | positive evidence | `confirmed-history/artifact summary` |
| C1.03 | V6/LPCP/GOMQ证明旧共享Writer能达到143/151附近 | positive evidence | `confirmed-history`；不等于应返回旧架构 |
| C2.01 | permutation/representation margins、LoRA health、loss等只证明内部机制 | evidence boundary | `accepted`；一律不单独选模型 |
| C2.02 | 更漂亮rank/energy/cosine/functional evidence历史上可能降低闭环 | historical fact | `confirmed-history` |
| C3.01 | v4只否定当时absolute-time shortcut，不否定一切时序建模 | negative-result boundary | `accepted` |
| C3.02 | LMMPC v1只否定对应实现，不否定memory token/layer correspondence整体 | negative-result boundary | `accepted` |
| C3.03 | v2/v3/v4负结果分别只约束实际测试的Reader/Core/M2P组合 | negative-result boundary | `accepted` |
| C3.04 | reward/manifold/gradient旧实验不否定所有occupancy/RL/credit方法 | negative-result boundary | `accepted`；当前目标仍是zero-interaction初始LoRA |
| C4.01 | `task drift`只是表象，不是根因 | terminology correction | `accepted`；后续必须定位task/state/stage/module |
| C4.02 | Procedure趋同不能单独解释失败，LPCP143构成反例 | causal correction | `accepted`；仍作为接口诊断，不作为充分根因 |
| C4.03 | functional mismatch必须拆为state cotangent/shared update/generated occupancy/retention | causal correction | `accepted`；F2/F5分别裁决 |
| C4.04 | 不能由当前失败直接归罪arithmetic mean或AdamW | causal correction | `accepted`；F5 matched实验前保持未知 |
| C4.05 | “全动态路径已接通”不成立 | mechanism correction | `confirmed-code`；F0a补表 |
| C4.06 | “memory correspondence成立”只能写index preservation | terminology correction | `accepted` |

## 5. D/E — 根因排序、证据与反证

| ID | 专家意见 | 类型 | 当前裁决与执行状态 |
| --- | --- | --- | --- |
| D1.01 | absolute上限与retention可由不同首因、相互放大 | causal model | `accepted working decomposition` |
| D1.02 | front-end credit/shortcut更像absolute问题 | hypothesis | `underdetermined`，F0/F1 |
| D1.03 | occupancy/support retention更像25→50问题 | hypothesis | `underdetermined`，F2 |
| D1.04 | decoder co-drift可能同时影响两者 | hypothesis | `underdetermined`，F3/F4 |
| E1.01 | detach、freshness、测试缺口、correct-only与language权限是直接证据 | evidence bundle | `confirmed-code/contract` |
| E1.02 | Program主要学identity/carrier、detach解释分数仍是推断 | inference boundary | `accepted`，不得提前写confirmed |
| E1.03 | identity constant、内部order sensitivity、Reader+19和frozen feature质量削弱单因解释 | counterevidence | `confirmed`，纳入F1判读 |
| E2.01 | loss下降同时52 lost、49始终成功、恢复少是直接retention证据 | artifact evidence | `confirmed-artifact` |
| E2.02 | generated occupancy shift、cotangent牺牲support、Adam moment污染仍是推断 | inference boundary | `accepted`，F2/F5分开 |
| E2.03 | B20覆盖广、后段恢复、历史occupancy尝试不稳是反证/限制 | counterevidence | `confirmed`，F2不得预设结论 |
| E3.01 | shared 256 bottleneck、B-first、head变化大、endpoint派生是直接结构证据 | evidence bundle | `confirmed-code/artifact` |
| E3.02 | head manifold不可达、条件数差、dictionary改写仍是推断 | inference boundary | `accepted`，F3/F4裁决 |
| E3.03 | 旧143、当前新增23 rows、所有family非零、后段模块变化接近是反证 | counterevidence | `confirmed`，禁止先验认定decoder唯一根因 |
| E4.01 | 对仓库六项解释的修正需进入后续措辞 | reporting requirement | `queued`；最终专家报告逐项引用C4/E4结论 |

## 6. F — 最小判别实验逐项执行账本

| ID | 建议 | 类型 | 状态与完成条件 |
| --- | --- | --- | --- |
| F0a.01 | fresh canonical task step全模块gradient audit | unconditional diagnostic | `completed`；见`gradient_audit_before_fix.json` |
| F0a.02 | macro25 canonical task step同样审计 | unconditional diagnostic | `completed`；同task/K/query/RNG matched |
| F0a.03 | 覆盖patch grounding q/k/out、interaction、language projection | coverage requirement | `completed`；前两者在macro25均为no-gradient，language为nonzero |
| F0a.04 | 覆盖Text/Action Meta-LoRA、Core、Procedure、memory、Reader、K-set、M2P、8 heads | coverage requirement | `completed for sealed A`；上述各组macro25均nonzero finite，新B/C后续分别复核 |
| F0a.05 | source policy nonzero grad tensors必须为0 | safety/contract | `completed`；fresh/macro25均0 |
| F0a.06 | 新增稳定regression，不能再以fake部分路径代表全路径 | code recommendation | `completed`；真实functional audit覆盖全部参数组，稳定测试另覆盖projected outputs局部credit、source freeze与intended-path分组完整性 |
| F0b.01 | macro25 correct strict paired400 | unconditional diagnostic | `completed`；123/400，clean sealed raw rows |
| F0b.02 | same-task-other paired400 | unconditional diagnostic | `completed`；125/400，correct-only 18 / control-only 20 |
| F0b.03 | cross-suite-wrong paired400 | unconditional diagnostic | `completed`；81/400，correct净+42，p=`1.3816e-5` |
| F0b.04 | shuffled paired400 | unconditional diagnostic | `completed`；122/400，correct净+1，p=`1.0` |
| F0b.05 | shuffled-keep-first paired400 | unconditional diagnostic | `completed`；131/400，correct净-8，p=`.27996` |
| F0b.06 | reversed paired400 | unconditional diagnostic | `completed`；90/400，correct净+33，p=`2.1680e-4` |
| F0b.07 | no-video paired400 | unconditional diagnostic | `completed`；48/400，correct净+75，p=`2.1689e-16` |
| F0b.08 | fixed rows/state/env/policy RNG/K/video ordinal与完整paired统计 | execution contract | `completed`；7面板逐行核对mismatch全0，见`video_causality_evidence.json` |
| F0b.09 | +15、p<.05、same-task±10/90%等 | advisory thresholds | `completed-reporting`；wrong/reverse/no-video达absolute与显著性门槛，same-task总分达标但success retention=85.37%，shuffle方向门槛失败 |
| F1.01 | 仅移除fresh输出detach的matched干预 | training recommendation | `queued after no-Text attribution chain` |
| F1.02 | 保持frozen hidden detach、拓扑、rank16、data/B20/optimizer/LR/seed/K | single-variable contract | `queued` |
| F1.03 | source零梯度、fresh modules非零梯度测试 | regression requirement | `queued` |
| F1.04 | 专家root-cause/method/negative thresholds | advisory thresholds | `registered`；与owner稳定性合同并列报告 |
| F2.01 | 收集25/50 lost/gained/retained rollout states和首次分歧 | unconditional diagnostic | `completed`；52 lost / 13 gained / 71 retained均采集两checkpoint occupancy；136/136在首次replan已分歧，初态action RMS最小`.04284` |
| F2.02 | 构造固定`S25 union S50`并用expert/teacher reference比较同状态error | unconditional diagnostic | `underdetermined-after-audit`；fixed union与两checkpoint逐replanaction已完成，但validation task expert不存在且读取held teacher action违反信息墙，因此只能比较checkpoint disagreement，不能伪称expert error |
| F2.03 | 分别检查offline更好、lost occupancy更差、failure前分歧、gained反向 | causal criteria | `refuted for the simple occupancy-divergence claim`；offline loss改善，但lost在macro50 occupancy的checkpoint disagreement均值反而低`.00655`，gained高`.01129`；所有行从首个replan已分歧，缺少合法reference不能判断哪一方向更正确 |
| F2.04 | 若所有fixed states也更好却失败，转查loss-success/replan/tail | alternative branch | `not-applicable as stated`；没有合法expert reference，不能建立“所有fixed states也更好”的前提；剩余解释保留为未裁决边界，不据此启动新干预 |
| F2.05 | 只有证据支持才用预冻结occupancy panel替换B20训练 | conditional intervention | `not-applicable`；F2方向未支持该前提，不替换B20 |
| F3.01 | 从有support checkpoint冻结8 heads续训到下一节点 | unconditional mechanism diagnostic | `completed`；A macro25冻结八个FactorHeads续到macro50，16个head tensors不变、404/481 upstream tensors变化 |
| F3.02 | lost≤20、retention≥90%、score≥110、breadth不降 | advisory thresholds | `completed-reporting`；117分满足score≥110，但33 lost、73.17% retention且breadth下降，其余门槛失败 |
| F3.03 | 若仍崩落，责任后移upstream/objective/occupancy或fixed-head reachability | inference rule | `supported`；冻结heads显著优于正常84（49 gained / 16 lost，`p=5.08e-5`），证明head co-drift是放大器；仍相对起点丢33，证明不是充分根因并已进入F4 |
| F4.01 | 固定heads、每task自由优化20x16x256 Program逼近train24 expert | unconditional oracle | `queued` |
| F4.02 | 在train-task closed loop比较expert与投影后LoRA | oracle validation | `queued` |
| F4.03 | family-wise q/v/action-in/out reachability与endpoint单列 | analysis requirement | `queued` |
| F4.04 | ≥90% expert success则停止扩大head/rank，否则支持不可达 | advisory decision rule | `registered`；只作诊断，不成为deployment carrier |
| F5.01 | 最后做matched conflict-safe aggregation | final diagnostic | `conditional but this goal must resolve`；前因充分解释时以`not-applicable`回应，否则执行 |
| F5.02 | 保持per-task gradients、AdamW、LR、tasks、data，只换aggregation | single-variable contract | `queued if applicable` |
| F5.03 | macro25相当且25→50 lost显著下降才支持conflict | causal threshold | `registered` |
| F6.01 | 执行顺序F0→F1→F2/F3→F4→F5 | sequencing recommendation | `accepted`，owner的no-Text要求以A/B/C attribution嵌入F1前 |
| F6.02 | 不优先扫rank/memory数量/BF16/scale/M2P/seed | efficiency boundary | `accepted` |

## 7. G — 远程可见证据缺口

| ID | 专家要求 | 当前状态与完成条件 |
| --- | --- | --- |
| G1 | 四checkpoint、LPCP、GOMQ逐行paired400、video IDs、RNG reference | `completed`；见`docs/evidence/external_review_20260818/paired_evidence.json`及README |
| G2 | current macro25全部video controls与不同K4 sets | `completed`；`video_causality_evidence.json`公开7面板aggregate、paired统计、逐行success与video/RNG reference |
| G3 | intended modules实际gradient表与首次非零macro | `partial`；fresh/macro25完整表与first-observed state已公开，新B/C和精确首次macro随训练补齐 |
| G4 | 每个formal run的commit/dirty diff/source/config/checkpoint/metrics provenance | `completed for six reviewed panels`；training/eval commit差异、schema、manifest和contract均显式保留 |
| G5 | per-module delta与Program/FactorHeads cross-decode原始结果 | `completed`；见`writer_drift_evidence.json`，包含4点module gradients、3区间stage/FactorHead cross-decode与fixed-B20/strict transitions |
| G6 | lost/gained/retained occupancy、首次行为分歧、fixed union error | `completed with scientific boundary`；`occupancy_evidence.json`公开136行轨迹统计、初态首次分歧和fixed-union checkpoint disagreement；validation expert/teacher error因information wall明确记为不可判定，不用代理量冒充 |
| G7 | expert LoRA在head manifold投影误差、投影后closed-loop、family reachability | `queued`；F4 |
| G8 | objective/mean/Adam moment/shared heads/shared Program的matched conflict evidence | `conditional`；F5执行或证据化not-applicable |
| G9 | breadth@1/@5/@10、task histogram、suite minimum、top3 concentration | `completed for six reviewed panels`；公共metric helper已扩展，新strict结果继续沿用 |

## 8. Owner调整与不做的事

| ID | 调整 | 裁决 |
| --- | --- | --- |
| O1 | 后续canonical Writer不再使用Text Meta-LoRA | `owner-adjusted`；sealed A历史事实保留，新B/C均无Text |
| O2 | VL Meta-LoRA同样移除，Action Meta-LoRA不自动取消 | `owner-adjusted` |
| O3 | 不直接回到V6/LPCP/GOMQ | `fixed boundary`；旧实现只作provenance/counterfactual |
| O4 | 继续基于当前Core-Addressed Reader主架构 | `fixed boundary` |
| O5 | 不使用subagents，效率优先，不做防御性校验或小超参扫 | `execution boundary` |

## 9. 最终逐项收口格式

每项完成后将状态更新为以下之一，并给出artifact路径、commit和核心数值：

1. `confirmed-code`：机制事实成立，但性能归因仍单列；
2. `supported`：matched证据支持；
3. `refuted`：matched证据反驳；
4. `not-applicable`：只允许用于专家明确的条件建议，并写清未满足的前提；
5. `underdetermined-after-audit`：只有确实无法由现存资产或在授权边界内实验判定时使用，同时记录具体缺失证据。

最终面向专家的报告必须按A、B、C、D/E、F、G顺序引用本ledger，确保没有建议被静默略过。

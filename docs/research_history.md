# EMBER research history

当前咨询重点为结合全部历史证据重新推导有益视频特异性，暂不强制绝对性能，也不直接回到v5.2。
新专家先读[机制证据地图](review_materials/video_specificity_20260911/EVIDENCE_MAP.md)及[最新证据](review_materials/video_specificity_20260911/LATEST_EVIDENCE.md)，
再沿本页进入各阶段完整论证；旧“根因”措辞仍须接受证据审查，不能直接作为已证实结论。

本文把可复核的2026年7月至9月研究整理为三层：先读本页的阶段结论；遇到具体接口问题再读对应细节；只有摘要无法裁决时才进入
immutable Git原件和formal artifacts。历史中的资格、假设和“下一步”都属于当时时点，不恢复执行。当前状态见
[progress.md](../progress.md)，当前方法见 [设计记录](horizon_relation_video_writer_design.md)。

本次整理前的完整账本是 `fcdb6e43706c5fcedf10eaa5d2d459602b263016:docs/research_history.md`，含旧§1--181（有两个旧§126）；
逐轮findings及所有旧设计、专家原文也在同一提交。下文“旧§”均指这一冻结账本，避免重编号后误定位。
[查看完整旧账本](https://github.com/LinFyM/EMBER/blob/fcdb6e43706c5fcedf10eaa5d2d459602b263016/docs/research_history.md)。

## 研究主线速览

| 时段 / 路线 | 真正获得的证据 | 未解决或失效的范围 | 对接续工作的约束 |
|---|---|---|---|
| 7月下旬 source / v5.2 / v6 | 视频→完整LoRA可带来真实闭环能力，v6曾143/400 | 后续训练下降，视频必要性与稳定性不同时成立 | 保留强基线；不能以后期弱结果概括全部历史 |
| 8月上中旬多视频、稳定化、LPCP、GOMQ | 多视频可降低同task变化；GOMQ曾151/400 | 低参数变化仍有success churn；151未保持 | 参数稳定、多K和单点高分都不足以选方法 |
| 8月中下旬privileged effect / realizer | 独立task专家、固定policy effects具有容量 | fixed-A、短solver、shared mapping无法稳定实现功能 | 区分任务本身可学、证据可解码、共享映射可泛化 |
| 8月24--25日 G1 / G2 | native-factor局部容量与有序response动态各有正证据 | 不是一套已通过闭环的完整Writer | 继承正证据，不能把机制Gate当最终能力 |
| 8月26--9月2日 G3 / primal / EBSRI / PNBTT | 一些task-local operator/replay通过 | 多轮shared或specificity/capacity冲突未解 | 不再沿固定接口堆专用数学链；负结果只限实际函数类 |
| 9月2--5日 full-response / Axial / Unified | full-horizon合同修复，真实加速，若干task-local正控 | shared闭环弱或训练后漂移 | 完整输入、梯度和标准模块不等于机制兑现 |
| 9月5--6日 P/Q、完整输出、覆盖与clone对照 | 完整输出改善训练侧Goal；单task clone明显更强 | 共享与迁移仍弱，容量/优化/数据未被单独定位 | 不能继续围绕猜测小扫，也不能把全部差距归到某一个模块 |
| 9月6—7日重新推导与验证 | 分层局部关系图已实现；真实机制、short4读出对照及train24相邻节点完成 | train24 correct69→67、other72→64，熟悉/held训练视频21/18，未通过科学资格 | 当前run止于384；Owner随后要求先补齐已有证据，与专家讨论清楚再决定实施 |

<a id="baseline"></a>
## 1. 基线、口径和不能混用的数字

| 证据 | 结果 | 适用范围 |
|---|---:|---|
| 历史frozen source | 48/400 | 对应历史validation8面板；另一独立实测面板为47/400，不混为同一组rows |
| validation8 task-local rank16 experts | 250/400，suite 73/78/58/41 | privileged容量上界，不能成为held任务字典或部署第二adapter |
| train24 rank128 source SFT | step400 109/400，step425 107/400 | 不读取teacher video的跨任务参照；不是rank16 Writer同参数量对照 |
| train24内部fold0 held5 source / carrier | 21/250 / 43/250 | 机制面板；不与validation400或后来train-side150面板混算 |
| 早期v6 single checkpoint | 143/400 | 强能力参照，未满足后续稳定性 |
| 历史GOMQ单点 | 151/400 | 未通过相邻稳定性，不是selected checkpoint |

SFT复核已纠正文字误差：实际运行一直是8维state（position3、axis-angle3、gripper qpos2）与7维action，不是7维state。
SFT与当前冻结source的normalization一致，400组task/state/language/env seed和policy-noise prefix曾逐项核对；source SFT的真实rank是128。
它仍是历史运行结果，不冒充新后端重跑。精确复核在
`runs/analysis/pi05_ecp_prw_complete_shared4_20260906/sft_historical_compatibility.json`（旧§172）。

正式目标始终是single checkpoint的validation8 paired correct严格 >145/400及稳定/breadth/四suite/GoalLong/跨视频/因果合同。
80-row screen只判断后续投入；held5或训练任务面板只用于机制诊断。未见初始化、未见视频和未见task是不同的泛化问题。

<a id="early-writers"></a>
## 2. 早期端到端Writer与多视频稳定化

2026-09-11讨论补充：[专家回复原文](review_materials/20260911/expert_review_round1.md)与[Owner要求的合并追问](review_materials/20260911/FOLLOWUP_PROMPT.md)重新核对v5.2普通监督的视频依赖正证据，要求解释旧架构/recipe与当前方法的差异。它们是咨询记录，没有新实验或架构采纳；跨轮解释边界见findings§50。

### v5.2 / v6：有能力，尚未稳定

五臂顺序为correct / same-task-other / wrong / shuffled / reversed：

| 历史版本 | strict400五臂 |
|---|---|
| v5.2 old | 132 / 138 / 74 / 82 / 83 |
| v5.2 task-complete | 120 / 109 / 107 / 111 / 124 |
| v6 old | 121 / 122 / 111 / 84 / 47 |
| v6-fast task-complete | 143 / 135 / 125 / 128 / 129 |

v6-fast step450/500/550/600为131/130/132/126。较好的absolute、较强的视频差距和较低churn来自不同配方时，不能拼成一个
同时具有这些优点的checkpoint。这里是已有历史controls，不授权接续路线用shuffle/reverse参与设计或checkpoint选择。

原v6配方包含完整train24、50-episode视频池、width256，以及当时的Text/VL/Action三组Meta适配。step50/400每task实际约
1000/8000 action queries，对应106/143；新近18task、两视频、1024 queries与observer冻结的短学习同时改变了多项因素。
不能把性能差距简化为“只需更多steps”或“只需开Meta”。新候选只适配Action Expert，且不恢复旧coarse/horizon-mean前端。

v6使用learned FactorHeads生成完整LoRA，说明显式raw X/Y bank不是产生任何闭环能力的必要条件；这不证明后继可以随意丢弃
有用信息，也不证明新的坐标MLP优于旧heads。原设计为 `3a6f801d:docs/action_forecast_writer_v6_design.md`，早期详细流水账为
`ac233fa0:docs/research_history.md`旧§2及§3.1--3.9。

### 输出共享与纯语言路由：近等价反例的补充索引（2026-09-09）

本次对接续假设的历史核查只使用实际源码与correct资格臂，不把旧最终controls转为新架构选择信号。

- **Target-Owned Factor (`34be4a0`)**：76个native tensor各一个1024→256→native head，同target跨16 rank共享，不跨target/layer/side共享。末投影20,594,688参数；50/100/150/200为99/76/86/68。历史原文曾把梯度低重合和层几何称作condition-to-policy credit的定位；这些观察不能唯一证明现代Horizon同一根因。该拓扑不能当新发现，旧前端/初始化/配方与删除DirectionStores的联动也不能被省略。完整设计、实现和结果：`3a6f801d:docs/action_forecast_writer_target_owned_factor_design.md` §1–8，正式root `runs/outputs/pi05_as_writer_target_owned_factor_bci_rawfull24_decay400_formal_r6_b20_micro2_seed7_formalvideo20260722_34be4a0_20260804T051244Z/`。
- **旧v5.2/v6**：纯text queries只进入Q，内容输出是视频证据mean加attention读出；没有纯语言query残差。真实代码 `3a6f801d:src/ember/writer/temporal.py` 83–141、234–254；v5.1 proposal §6.1/13及v5.2 design §3。存在较强closed-loop，但不能单独归因于该路径限制；视频条件共同语义Value不等于裸语言Value。
- **Dynamic-K Semantic-Address (`9e70b81`)**：semantic address只进Q，V/残差来自有向D/G，correct100→101（R/G/L84/17/16）；DirectFamilyB (`c5353f3`) 改直接B后K1/K4为102/98；后续Task-Grounded Visual-Value增加真实patch差分，四点88/86/86/96。说明这类信息通路限制不自动形成高性能；它们的动态差分only/fixed-A/rank8/旧前端不是当前Horizon。源码 `9e70b81:src/ember/writer/memory_program.py` 85–110、242–271，各完整设计在 `8553b61:docs/action_forecast_writer_*_design.md`。
- **Common-Value Bridge与Direct Joint Native Factor Residual**：前者把冻结v6的set Value由centered Core换common Core，135→133；后者在LPCP143之上新增共享linear完整A/B residual为136。共同语义Value、直接共享head均有具体正负边界，不能拿其几何或模块名字保证后继性能。原件 `12311bd`/`0ead61e` 与 `8553b61` 对应设计。

这些反例降低“单改头部共享/删语言残差即自然修复”的先验，但不否定当前图上经充分机制分析后的单变量对照。当前已授权的冻结分析实验见active design §8.2.4。

### 多视频、LPCP与参数稳定：必须看行为集合

历史LPCP K4达到143/400、breadth7，但teacher schedule与另一143不同，不能伪造paired比较。若干DynamicSlotSet/SharedCore
实验将same-task有效更新变化降低约9.26倍/9.69倍，分数分别约130/139；更晚的聚合改善了波动却未恢复task-mean能力。
DirectFamilyB的K1/K4为102/98，即使同task变化降低约6.3倍，更多视频也没有保证更高分。

一次LPCP配对比较相对139保留120、新增23、丢失19；更新cosine接近0.99999479仍有42个success-set变化。这要求研究者将
视频导致的变化与训练相邻checkpoint的变化分开，把R/G/L、churn和breadth作为实际行为证据。

### GOMQ：纠正rank解释

GOMQ cycle2/3/4为151/135/131，相邻churn42、34。cycle2五臂151/139/131/127/115，存在视频特异性，但未稳定。
旧物化为 `A32=[A0;A0]`、`B32=[B0,deltaB]`，实数代数上等于 `(B0+deltaB)A0`，有效rank本来不超过16。
后来按rank16合同重物化得到136/400，123 retained / 13 gained / 28 lost；不能把151→136解释成已证实的有效rank容量损失。

依据：`ac233fa0:docs/gomq_rank16_archival_card_20260824.md` 与
`3075b3c7:docs/evidence/gomq_20260824/gomq_cycle2_effective_rank16_strict400.json`（旧§3）。这不授权dtype、rank、seed扫描或旧run恢复。

<a id="native-capacity"></a>
**2026-09-09冻结诊断补充：** 原fresh K1的200/600已完成A/A2/A3共6144 train-side FM queries及预登记64条validation行为回放，全部无参数更新。首层language内容残差比仅该层language检索的移除影响大；P4/cross有实际功能，不能据此宣布动态理解、捷径根因或fresh删除有效。回放9/32→7/32，与各自历史子集7/32有成功集合分叉，原正式110/82不改；图像定位了错误目标/实例、组合子目标和具体抓取混合缺口，并保留正确双目标成功。完整报告与原件索引：[冻结诊断](horizon_k1_frozen_diagnostics_20260909.md)。后续fresh内容路径对照只以当前progress登记为准，不从此历史条目恢复执行。

## 3. 从privileged容量到G1/G2机制证据

### Privileged effects与人工process的边界（旧§4--8）

独立successful members在held5达到113/250；固定phase decoder曾从source21提高到44/54，却只保留约26%--31%的direct expert
successes。shared prior43加上不可靠残差后跌到37/33，说明条件更新可能破坏已有能力。

fixed-A solver为78/250、breadth3且Goal/Long0；将三个known-success成员投影到固定A后仅49/41/35，说明该具体坐标限制确实丢失
行为。短raw-factor solver49、fixed realizer33/37、centered two-sided80仍没有解决稳定实现与跨任务迁移。
这些结果排除的是当时的参数化和预算，不排除policy effects中存在有效信息。

人工复合任务/primitive/recovery路线最终teacher只有14/100（9+5），相对参照gained0/lost23。owner终止人工process数据与新场景制作，
后续回到现成LIBERO。这是有效科学non-pass，不是运行错误；数据/控制器问题不能由人为任务循环掩盖。

### G1：X/Y约束能有容量，也能限制必要方向（旧§9--12）

最初scalar signed-output pooling使无bias q层B只能处在base q weight的列空间，原生输出2048维却至多1024维；action-in含bias时
上限是span(W,bias)，至多33维。将已有独立成功更新投影到此限制后，strict250从120降到109，Goal/Long由11/8降为0/0。
这是对“丢掉的方向有行为必要性”的直接干预证据。

随后在原生q attention heads内分组，以及在action-in原生输入宽度对应的blocks内独立读出，结合privileged解析初始化，step0达到
114/250、breadth5/5、Goal2、Long1，retained carrier35/43，G1通过。authority为 `31f0053`，正式root：
`runs/outputs/pi05_ecp_native_factor_g1_action_in_groups_held5_formal_31f0053_gpu02p1_20260825/`。

这一结果证明特定native X/Y与rank4 residual有局部可达性；free-logit优化曾走坏，shared Program→content attention还未被证明。
它既不要求后继继续signed pooling，也不支持宣称native-bank路线整体没有容量。

### G2：有序response确有动态，具体DP alignment曾坍缩（旧§13--20）

多次内部non-pass分别涉及静态旁路、分段freeze、scalar/temporal readout与optimizer cadence，但最终关键干预是monotonic DP的
首尾canonical边界。K>1原路径常退化成单event；边界锚定后macro20的full/endpoints action+progress loss为0.28167/0.36207，
相对改善22.2047%，probe38/40，median active events4、one-event0；对应Gate通过。

authority `c1493a1`，冻结macro20，正式root：
`runs/outputs/pi05_ecp_natural_program_g2_boundary_fold0_m10_c1493a1_gpu02p0123_r4_20260825/`。

这支持完整有序PI0.5响应包含动态信息，以及当时alignment的具体修正；不证明Natural Program已足够让共享Writer在held task
产生有效LoRA。新候选没有恢复DP、固定语义event schema或G1--G3强制课程。

<a id="program-compilers"></a>
## 4. G3与一系列共享编译器：保留容量，停止补丁链

此阶段旧§21--103细节多，但主问题反复相同：局部可达的功能坐标，经过共享表示、跨bank转换或readout之后，能否保住真实功能。
不要把所有试验归为同一次失败，也不要由一个operator通过宣称完整映射通过。

| 子阶段 | 已有正证据 / 实际裁决 | 边界与教训 | 旧账本 |
|---|---|---|---|
| frozen Program / dual basis | held-task native span与shared credit有缺口 | native span、训练credit、可泛化mapping是不同接口 | §21--26 |
| current-bank operator F1 | family task-mean update cosine约0.9998以上；streaming保住analytic capacity | 只证明operator，不是shared mapping或closed-loop Gate | §27--42 |
| low-dimensional sketch / set summary | 减少专用坐标规模仍不能稳定恢复功能 | 压低内部loss不等于保住必要方向 | §43--45 |
| primal/current-bank dual P0/P1 | P1 fit/held median recovery .9717/.9545，跨视频容量通过 | task-local primals通过，不代表共享Program预测通过 | §46--51 |
| joint Program-primal J2/J3 | task-local正控通过，shared routing仍non-pass | 不把route token或特权task code当部署接口 | §52--55 |
| R1--R13 chart/routing/refinement | fixed chart R5通过，接回真实Program R6失败；R10功能大幅改善但held/bank交互仍失败 | 学到或给定一个code与从自然视频得到同等功能不是同一件事 | §56--73 |
| candidate-level interaction | 多种标量/向量交互、effective-rank资格暴露capacity–specificity冲突 | 不再在同一弱接口前后堆gain、gate或recenter | §74--80 |
| EBSRI / quotient | S0/S1 task-local通过，S2/direct-functional/quotient相继non-pass | 正控不能替代最终真实路径，共享角色/owner混杂需实证 | §81--90 |
| Program-through-bank / calibrated A | free-summary/过拟合正控通过，实际query未同时恢复correct与wrong specificity | 局部可达与真实共享条件获取分别裁决 | §91--96 |
| PNBTT E1 | 单key/family-key/full-rank16/gate-aligned均相邻一致non-pass | 始终停在free-query E1，真实Program E2没有启动 | §97--103 |

P0/P1曾在6个预注册fit tasks、8个浅中深/边界targets上验证：每task只训练共享于两条fit视频的primals，held video不产生梯度，
held/fit为.9823，四family held medians .9398/.9416/.9954/.9452。原件 `c9e8198` 和
`runs/analysis/pi05_ecp_primal_capacity_p1_v1_c9e8198_gpu01p012345_20260829/`。这是真实应保留的局部容量证据。

PNBTT最后一次gate-aligned E1纠正了原necessity hinge与Gate margin的错配，wrong/margin通过，但correct/held依然失败；末端谱没有
新的width/rank方向信号。最终原件 `e65c6388`，launch commit `2050de9e`，root：
`runs/outputs/pi05_ecp_pnbtt_e1_gate_aligned_necessity_s110_e65c6388_gpu01p12_20260902/`。

后来复核纠正了两种过度解释：PNBTT没有完成真实Program E2，因此其失败不能否定G2；某些曾被命名为“根因”的局部数值现象，
也不足以解释整个共享闭环问题。当前禁止恢复这些退役专用链为默认fallback；原数学论证和试验边界保留在Git。

<a id="full-response"></a>
## 5. 完整原生响应、Process–Composer、Axial与Unified

### 输入完整性与工程错误要先分清（旧§104--123）

9月2日后路线转向完整Policy-Response与事件/因子共同学习。起初task-local Composer正控和部分12-task比较有收益，扩大到73task
之后明显退化。后续检查确认过真实horizon完整性/scale/梯度预算问题，owner因此锁定full50，不再把coarse或horizon mean作为候选。
另一次m400收尾失败是NFS mmap生命周期错误，checkpoint与独立Panel-B证据完整；不能把退出错误当成科学non-pass，反之不能用修复
运行错误自动宣称能力恢复。

### 多次职责替换没有解决共同学习（旧§124--163）

| 路线 | 当时观察 | 不能扩大成的结论 |
|---|---|---|
| rank-balanced / typed Composer | 某些m200明显改善，m400又退化 | 名义rank变好不是稳定性能保障 |
| process/causal-filter/functional/gain | 若干内部接口修正，闭环仍低或持平carrier | 梯度接通、loss更低不等于动态成为必要Value |
| Axial / Frame-aligned / Frame-Bank | 任务局部有容量，shared仍不充分 | 不证明整条视频到LoRA目标不可学 |
| Native-Temporal | 早期部分泛化随后漂移 | 新增一个时序模块不等于保留历史行为 |
| Unified common-base v3 | shared held5 m100/m200为35/31，低于carrier43 | 模块职责合理不代表共享行为成立 |
| source-separated v4 | m25/m50为45/40，breadth3/5，Goal/Long0 | m25净增+2未保持，不能直接进入长跑或Final |

语言曾与大量patch/response tokens共用softmax，多数层语言质量仅约2.2%；分源读取改进了内部grounding，但没有同时解决绝对能力。
因此新候选继续区分查询条件和视频Value职责，同时不把“分源attention”本身当成足够的方案。

v4终局证据root：
`runs/outputs/pi05_ecp_policy_response_writer_source_separated_73task_k1_component_s50_f02f9148_gpu01p036_sharedmmap_20260905/`，
物化/strict250 roots由旧§163列出。该图及专用配置已经退役，G1/G2和task-local正证据仍有效。

<a id="recent-learning"></a>
## 6. 9月5--6日：同图学习、完整输出与共享缺口

### 从同图正控到有限改进（旧§164--171）

9月5日专家复核保留v5.2/v6/G1/G2的正证据，强调absolute、breadth、视频必要性与相邻漂移须分别验证；纠正GOMQ有效rank和PNBTT
E1/E2范围。owner随后接受共同过程P/Q候选，但暂停讨论期间不准启动；完成对齐并明确授权后才实施。

| 实际干预 | 配对结果与边界 |
|---|---|
| whole-Writer clone/shared，去掉额外task query | 首轮shared4 fit32/64=39/44、held40/41（各150）；局部新增仍伴随旧成功丢失 |
| 原Panel-A fresh query覆盖 | 实际episode从8到16、unique rows从115--126到455--499；fit41/41、held39/45，混合改善，非普遍迁移 |
| 非对称A上下文 | 同batch对称64=41/44，非对称64=44/45；Goal/Long相邻仍弱，未识别整体根因 |
| 两条fit视频池扩大到四条，K仍为1 | 固定64 updates将单视频曝光减半，fit41、held39，低于两视频44/45；这不是K1→K4实验 |
| 共同P/Q替换有限输出主干 | 32/64 fit38/41、held37/39，低于A2；更稳定的低分不构成胜出 |
| A2 meta73充分曝光 | strict400 primary32/48/64=80/81/79，other78/81/75；Goal/Long弱且未稳定，未达资格 |

A2 meta73的32→48→64 primary Jaccard .769/.818并没有改变绝对弱分。Long跨视频和相邻成功缺少保持，不能只按整体Jaccard通过。
完整run与配对在 `runs/analysis/pi05_ecp_prw_meta73_equal_exposure_20260906/`。

### 完整输出是有收益的耦合变化（旧§172--173）

owner重新对齐早期强能力参照，允许P/Q联合生成全部38-target rank16 A/B，无独立carrier，不再把raw X/Y signed span作为输出硬约束。
同四任务、同query/video/noise/optimizer的短学习中，新32 fit/held=50/54，64=64/62；旧P/Q为38/37、41/39，A2为43/41、44/45（均150）。

收益主要来自Goal；64时Spatial比A2两视频都少7次，Long只在新视频出现4次成功。新64跨视频Jaccard .680，相邻fit/held .606/.589。
这支持“完整输出职责重构”这一组合，不单独识别解除span、移除carrier、增加mobile rank或head变更的效果。

证据：`b2bb03ce`，`runs/analysis/pi05_ecp_prw_complete_shared4_20260906/`。

### meta73、target18与fully-random（旧§174--181）

三组完整输出图都使用128 updates、每target1024 action queries、两条fit视频；mixed对照实际保持对应target采样、noise与权重审计。
这些是有限预算研究，不是整个函数类被穷尽。

| 训练组合 | validation8预登记四点screen80 | terminal训练18task诊断 / 边界 |
|---|---|---|
| component，55meta+18target | 15 / 19 / 19 / 19 | 42/180，breadth9/18；Object/Goal少数任务占主要成功 |
| component，仅同18target | 17 / 17 / 20 / 16 | 55/180，breadth13/18；受监督Object仍5/40，未见task不是唯一缺口 |
| 同18target，whole-Writer随机初始化 | 16 / 16 / 17 / 19 | 未补全18task180；固定两弱task仅4/20，不能外推全部任务 |

同prefix SFT是24/80。三组都没有显示足够广泛而稳定的趋势，因此未扩strict400、未选模型；80分不能线性当作正式400结果。
移除meta后目标functional大多改善，但meta诊断变差；这证明特定组合下的取舍，不证明所有额外meta任务有害或梯度冲突是根因。

同两个训练任务Spatial7/Object2，whole-Writer独立clone为6/10+8/10=14/20，共享component为3/20，随机共享4/20；原3个成功被clone保留。
Clone使用相同video/query/noise/每task预算，区别涉及共享目标、有效容量与优化，不能仅凭该差距命名容量或梯度冲突。
共同component32/64/96/128在这两个任务为2/3/4/3，没有先强学会再遗忘的阶段。

训练轨迹与teacher-state接续提示接触、抓取、放置等异质失败，不能统一归因occupancy漂移。Long38/meta93专家本身弱（5/50），
不能代表其它更强Long任务；另一Long35专家40/50提示短面板应有充分任务代表性。晚期状态接续一锅完成0/3、更后contact2/3，
也不能说明所有失败都来自初始状态分布。

主要完整证据：

- `runs/analysis/pi05_ecp_prw_complete_meta73_20260906/`，authority `041aff55`；
- `runs/analysis/pi05_ecp_prw_complete_target18_20260906/`，authority `351feb48`，含180-row和轨迹/续接诊断；
- `runs/analysis/pi05_ecp_prw_complete_single_task_20260906/`，authority `6efdd2e0`；
- `runs/analysis/pi05_ecp_prw_complete_target18_random_20260906/`，authority `f3717836`。

### width256训练完成，但没有闭环结果

同图随机width128→256、heads4→8，其余18task/两视频/128updates保持，Writer从4,750,208增至15,660,800参数。
19:40启动后owner19:50明确暂停自主后续，允许当时训练自然结束；没有再启动物化或闭环评测。
本次交接只读核验：128步正常完成、train.exit=0；训练732.48秒、内部functional Panel-B498.24秒、总计1283.05秒，32/64/96/128
四个checkpoint均保留。内部Panel-B不是闭环性能，不能给本轮填写validation分数或称加宽已成功/失败。

root：`runs/outputs/pi05_ecp_prw_complete_target18_width256_s128_14bc7605_gpu02p012356_20260906/`，authority `14bc7605`。
旧run现已封存；补评估并非新session的默认第一步，须由新架构的实际诊断需求决定且遵守新授权。

<a id="design-alignment"></a>
## 7. 9月6—7日从科学动机到分层局部关系

owner要求在正式推进前充分讨论，并以另一段“从π0.5静态图像信息走向视频过程”的对话为共同基础。核心动机是无兼容action标签的
教学与跨身体技能迁移；owner明确不能只因为full-horizon已捕获、梯度已接通就宣称兑现了科学精神。

### 7.1 9月6日初稿

以下是9月6日的历史推导顺序。其中past-only及内容差分的定义已被下一小节覆盖，不能恢复为当前实施合同。
初稿原文保留在Git `12d9689c:docs/causal_layered_video_writer_design.md`；最新完整公式在
[新设计记录](layered_relation_video_writer_design.md)：

1. 保留exact language与原生Gemma图文prefix；Action Expert共享Meta适配观察侧，vision/Gemma冻结。
2. 将video时间T与relative action horizon H分开，在H压缩前真正进行跨帧处理；不把H位置命名为物体/阶段。
3. 用后帧查询过去，建立有向过程；同一block只读取上一层状态，禁止未来帧或全视频长度泄漏到早期E。
4. 当前相对过去的响应差分表达变化，当前内容提供条件；不使用t+h的假时钟，不要求严格单调物理对应。
5. 增加显式计算层J，跨帧先保持同层匹配，共享block加层身份；J不等于动作阶段。
6. learned H-read得到E[t,j]，最终全局整策略queries读取所有视频的过程集合；单个E仍只有有限历史范围。
7. 从LoRA的功能作用推导A/B配对与原生坐标条件MLP；不把小latent维度本身误认为限制所有输出线性span的证明。
8. 多视频在过程集合阶段共同读取，按视频基础质量归一，独立K采样训练；多K的Bayes/信息/噪声推导只在所述假设下成立。
9. 原双probe没有新的需求依据，owner明确选单固定probe；删除无独立职责的noise/初始boundary/velocity/层差分旁路。
10. 不额外恢复raw X/Y bank，不限制因子span；真实执行功能梯度学习原生坐标。专家全量复审不再作为开工前置。

这些决定形成了一个已对齐、可实现、可证伪的新候选，尚无性能证明；width、层数、窗口与MLP宽度只是首版默认。
owner安排新session接手，当前session负责完整记录和仓库清理。正式推进仍须新session理解仓库后获得owner明确同意。

### 7.2 9月7日帧对关系重推与再次交接

owner明确希望在局部窗口内先建立每个帧对的50×50关系，随后每帧聚合自己的证据，不预先把整个关系规定为某一方向的query。
确认半径4时前后两侧最多8个邻居均可使用，同型模块应当可以堆叠；视频在rollout前完整可用，早期表示读取后续教学帧是合法的。

随后owner进一步明确：新帧较靠前的horizon可能对应旧帧较靠后的horizon，关系矩阵可能呈现偏移带，但应由内容与时间间隔学习，
不做人为平移或单位矩阵监督。重新推导据此收口：

1. 每个无序帧对建立一次共享F的score；联合bias依赖帧间隔与horizon位移，避免独立gap常数被帧对内softmax抵消。
2. 两端分别归一化C与其转置，不能转置已归一化的A；signed gap与接收端相对位置保留方向。
3. 对应内容m与相对位移分布rho一同进入关系MLP。完美对齐可使内容差为零，rho仍保留对应位置的推进证据。
   rho只是A行的索引重排，其MLP第一层等价为相对位置向量的加权读取，无需独立位移摘要网络。
4. 每个关系先经过非线性解释，再由每帧对自己的邻居消息做attention、residual和FFN；所有horizon一直保留到最终H-read。
5. 每层只读旧U并同步更新，blocks间重新计算关系；上下文是前后各Bw，替换旧的past-only及未来帧不变性检查。
6. 单probe/观察Meta、集合Q、完整坐标A/B与真实functional链式梯度继续保留。对应的物理意义、视频必要性和闭环收益都未被公式证明。

owner在完整重推后要求更新仓库并重新给出新session prompt。当前图具备明确可实施的候选定义，尚未实现或产生新实验结果；
新session仍先充分理解并报告计划，得到owner明确同意后才能正式推进。旧初稿通过Git保留，active tree只保留最新设计。
这一修正记录的是设计判断，不是新的性能或根因结论。

<a id="throughput"></a>
## 8. 应当复用的工程经验与实际加速范围

| 变更 | 可核验收益 | 对新图的适用边界 |
|---|---|---|
| full Writer exact batch/SDPA/fused pooling/placement | 同4卡10step从34.394降到4.054秒/step，8.48倍 | 同旧图完整输入、task权重和梯度语义；不是新图加速承诺 |
| node-local共享frozen evidence mmap | 105,020,606,660-byte/146视频；两卡8GiB replica→mmap均值18.5403→17.8110秒，最坏26.2068→19.8142 | 解决cache ownership造成的负载不均；Meta更新时不能复用旧R |
| 同run物化复用policy residency | 每条件重复准备约115--116秒降到0.11--0.24秒 | 这是消除重复加载，不是每次完整Writer forward的相同倍数加速 |
| 真实policy inference batching | 独立forward约2.9倍；同strict150端到端34.09→23.46分钟，1.45倍 | 带环境/排队后收益小于算子收益，正常BF16差异可改变少量success rows |
| 真实batch8/16/32 policy profile | 7.83/8.18/8.31 observations/s | 该场景增batch只有约6%，不能无证据宣称还有数量级收益 |

旧§114/123/166/168保存完整配方与数字。关键原则是批量和布局先行、每步task权重不随GPU位置改变、按真实长视频profile选择配置，
不能用最低显存、静态task绑卡或dummy占用作为效率目标。

交接清理时复核的旧shared trainer实际按condition执行：A/B leaf forward → policy query microbatch VJP → Writer replay。
它没有现成的跨condition batch VJP，不能把新设计中的候选优化写成已有能力。已提取的通用helper见当前代码地图；旧原件为
`fcdb6e43:src/ember/ecp/policy_response_writer/shared_training.py` 与 `shared_execution.py`。
新图的Action Meta梯度需要继续实现R的VJP与分块observer重放；只缓存冻结prefix，不可沿用旧永久detached response cache。

<a id="archive-index"></a>
## 9. 按问题恢复原件，不重读所有历史

### 完整账本的分组入口

| 要回答的问题 | 冻结旧账本范围 |
|---|---|
| 最强基线、早期Writer、多视频、GOMQ | §1--3；更早细节 `ac233fa0:docs/research_history.md` §2、§3.1--3.9 |
| privilege/effect/solver为什么失败、人工process为何终止 | §4--8 |
| X/Y native span、q/action-in grouping、G1容量 | §9--12 |
| 视频动态、G2 cadence与DP alignment | §13--20 |
| frozen Program/fit-span/dual basis | §21--26 |
| current-bank operator、anchor、compatibility、polar | §27--42 |
| functional sketch、primal P0/P1、behavior kernel | §43--51 |
| joint Program、routing、chart acquisition/refinement | §52--73 |
| candidate interaction、EBSRI、quotient | §74--90 |
| Program-through-bank与PNBTT完整适用条件 | §91--103 |
| full horizon输入合同、scale与执行吞吐 | §104--123 |
| rank/typed/process/gain接口反复修订 | §124--141（原文有两段§126） |
| Axial/Native-Temporal/Unified与shared漂移 | §142--163 |
| 最新整体专家复核及同图、覆盖、完整输出 | §164--173 |
| meta73/target18、轨迹、clone、random | §174--181 |

读取方法，例如：

```bash
git show fcdb6e43706c5fcedf10eaa5d2d459602b263016:docs/research_history.md
git show 3a6f801d:docs/action_forecast_writer_v6_design.md
git show ac233fa0:docs/research_history.md
```

要核对某一次实验，先从对应段落取得authority/run根，再读该run的contract、completion、metrics/raw rows与必要analysis；
不要反过来扫描所有run寻找“最好数字”。旧生成缓存删除后，原始评测证据与生成配方仍保留，见下节。

### 专家原文索引

以下文件均按原样保留在 `fcdb6e43706c5fcedf10eaa5d2d459602b263016:docs/<filename>`；从活动树删除是文档生命周期整理，
不是抹除原始意见。意见必须与owner后续修正及实际结果一起解释，不能把旧建议当新硬约束。

| 原件文件 | 重点与后续边界 |
|---|---|
| expert_review_20260824_native_factor.md | X/Y native-factor与分接口验证；工期/固定尝试数后来被owner取消 |
| expert_review_20260826_bank_conditioned_native_factor.md | current-bank几何与operator；capacity不能替代shared mapping |
| expert_review_20260828_g3_functional_sketch.md | low-dimensional sketch，后续specificity/capacity仍不兼得 |
| expert_review_20260829_joint_program_primal.md | joint Program-primal；routing/functional-code正控的适用范围 |
| expert_review_20260830_program_bank_interaction.md | candidate-level交互；后续effective-rank资格non-pass |
| expert_review_20260831_event_conditioned_bank_set_relative_interaction.md | EBSRI；S0/S1通过不代表S2实际条件路径通过 |
| expert_review_20260901_program_through_bank_bottleneck.md | 强制信息路径与real-bank transport；后续PNBTT只到E1 |
| expert_review_20260902_global_route_reassessment.md | 全路线复核，避免局部补丁扩大为根因判断 |
| expert_review_20260902_full_history_policy_native_meta_writer.md | 原生policy响应与自然视频主线 |
| expert_review_20260902_policy_response_event_to_factor_writer_clarification.md | full axes、事件到factor职责与正视频训练边界 |
| expert_review_20260905_full_history_joint_process_policy_writer.md | 保留早期正证据、同图正控、P/Q候选；新设计已按owner讨论进一步修订 |

旧设计原件同样位于该commit：event_conditioned_policy_compiler_design、program_conditioned_native_bank_tangent_transport_design、
policy_response_event_to_factor_writer_design、axial_policy_response_native_factor_writer_design、unified_policy_native_factor_writer_design、
joint_process_policy_writer_design（均为docs下的.md）。后者已被本次完整单probe设计替换，不能恢复其旧运行清单。

<a id="cleanup"></a>
## 10. 交接整理与证据保留

2026-09-06本次授权为完整设计记录、文档/源码/存储清理及新session交接，不启动新架构实现或科学训练/评测。
旧专用执行面通过Git保存，保留source、数据、task专家、评测、正常LoRA应用及必要的通用训练基础；当前代码地图以README和新设计为准。

运行产物只删除已经确认可重建的派生payload：从保留Writer/source/数据与原generation recipe生成的episode LoRA缓存、
两种旧冻结特征缓存、已结束的编译profile缓存。原run contracts、所有entry/cache JSON、指标、raw rows、正式checkpoint与唯一轨迹保留。
原cache manifest记录的是历史生成状态；payload退休的名单、依赖链、实际删除量和保留例外在
`runs/analysis/ember_handoff_cleanup_20260906/storage_cleanup.json`，不能把“文件不再驻留”误读成当时未生成或未评测。

两个缺上游Writer checkpoint的小smoke缓存保留，不能证明可重建；两个约44GiB的正式训练root及全部独特occupancy/teacher-state
诊断证据也保留。所有旧已完成且HEAD已集成main的detached工作树移除，历史代码仍可从其commit重建。
清理完成量、验证与Git交付状态在 [progress.md](../progress.md)，本文不重复动态现场。

## 11. 2026-09-07持续科研执行授权

Owner在交接基线9ea20340之后明确授权：完整理解后立即实现、验证、训练、评测、诊断及证据支持的修正；此前等待批准的
交接状态被覆盖。当前Writer与读取侧Meta采用fresh optimizer/scheduler直接端到端联合训练，不实施G1--G3冻结课程，
也不为历史候选措辞额外创建阶段初始化路线。合法LoRA初始化保持。旧评审、历史结果和当时暂停记录继续作为历史事实保留。

接手时main干净，HEAD为9ea2034037e5c70b514198a70910aac5c2fb18f5；临时HANDOFF消费后删除，入口改指正式账本。
长期goal按完整稳定性、因果controls与最终32/8评测定义，尚未获得新架构实现或性能证据。当前进度以progress为准。

## 12. 新图首轮真实机制与成本（2026-09-07）

唯一layered关系/集合/坐标Writer已实现，Writer14,112,544参数、读取Meta626,688。真实两帧初始化启动后，Meta B及A先后获得
实际flow loss梯度；source无梯度。限定抽查的full/staged VJP loss相同，三个接口梯度cosine均>0.99998，正常BF16误差下链式语义成立。
完整105帧K1及406帧K4含真实16queries的联合前反传为8.27/25.78秒，prefix另3.60/13.37秒；K4峰值allocated34.75GiB。
首轮CUDA BF16消息scatter dtype错误在第一个功能更新前修复；不作科学non-pass。以上均为exploratory机制/profile，不证明闭环。
原始脚本、日志、JSONL和设备快照在runs/analysis/layered_relation_writer_20260907/mechanism/。据此预登记short4 fresh联合训练与
16/48/96行为节点，含有实质专家容量Long35；实际launch、完成和分数另由progress及formal证据登记。

## 13. Fresh joint short4的第16步行为节点（2026-09-07）

从clean pushed detached8d934408，world3完成16updates、每task256queries与真实K1/2/4，全部恢复状态保存、exit0。固定global7/12/20/35
的states0–9与两组held K1视频，d5b8119e物化及三组动态队列全部exit0。Source/correct/other分别4/4/6（各40）；按Spatial/Object/Goal/Long
分别2/0/0/2、2/0/0/2、4/0/0/2。Correct保留source4，无新增/丢失；other保留4、新增Spatial2、丢失0，跨视频Jaccard2/3。
每组breadth2/4，未证明广泛基础能力，更不是validation资格。该剂量不足以否定共享图，按预登记短学习继续48/96，不扩长训。

完整raw rows、same-RNG/环境/source-normalizer合同与RGL：runs/analysis/layered_relation_writer_20260907/下的short4_source_screen40、
s16_correct_screen40、s16_same_task_other_screen40及s16_correct_vs_source.json、s16_other_vs_source.json、s16_cross_video.json。正式训练root为runs/outputs/layered_relation_short4_joint_8d934408_gpu01p235_20260907。
原生frame批量16的真实最长K4另测joint16.88s、peak34.65GiB，无optimizer更新；它是布局成本证据，不是新的学习结果。


## 14. 原始坐标初始化short4完成与最早输出接口线索（2026-09-07）

原8d934408 fresh joint run已完成96步，384conditions/6144queries，每task1536queries、K1/2/4各32条件。
16→48→96完整exact-resume均通过；实际更新总1288.78秒，平均13.42秒/step，含三次加载总1639.82秒。
每task训练视频覆盖16/16，独立query episode-frame数为1230/1248/1269/1310；不能把这些episode曝光算作更多meta-task mappings。

| 固定节点 | queries/task | correct/40 | other/40 | correct相对source R/G/L | other相对source R/G/L | 跨视频Jaccard |
|---|---:|---:|---:|---|---|---:|
| source | 0 | 4 | — | — | — | — |
| 16 | 256 | 4 | 6 | 4/0/0 | 4/2/0 | 2/3 |
| 48 | 768 | 6 | 5 | 4/2/0 | 4/1/0 | 5/6 |
| 96 | 1536 | 4 | 4 | 3/1/1 | 3/1/1 | 1/3 |

所有K1点的breadth均2/4、Object/Goal均0；Spatial/Long依次为16c2/2、16o4/2、48c3/3、48o3/2、96c2/2、96o3/1。
相邻48→96 correct RGL4/0/2，other3/1/2；96跨视频RGL2/2/2、churn4/40。
96步追加K4correct事先登记，仅使用全部held46–49，得到6/40（Spatial3、Long3，其它0），对source RGL3/3/1；
对同点K1 RGL2/4/2、churn6/40、Jaccard1/4。K变化是明确的视频集合干预，不能冒充相同video schedule；任务/状态/RNG经共同source比较核对。
这些是训练侧诊断，无validation/Test资格或最终选点意义；原配置未形成稳定广泛增益，不继续无依据长训。

无更新固定query面板在96步的correct功能benefit约[1.59e-4,1.26e-4,3.48e-4,-3.18e-5]，不能替代行为。
在两个授权训练任务7/35、expert0q/0v/action_out三代表target上，B的native-channel常量方向能量>99.995%，
真实flow梯度在该方向仅约0.003%–2.23%；code RMS1.10–1.16，而坐标RMS0.02、rank间code差约0.0116。
这定位到输出接口的一项可检验学习条件问题，不能唯一归因旧共享缺口，不能否定仍未被充分检验的关系图或视频过程。
由此启动880bde5e的单变量fresh短对照：native A/B坐标改独立标准正态，其它科学与曝光合同不变；初始source flow loss与原run同为0.1250352208。
新初始化能否有效仍待同一闭环判据，不把方向或梯度变大当作性能通过。

所有原始rows/completion/RGL/launch、short4_exposure_cost.json、functional_s*.json、decoder_gradient_s96.json/.safetensors保留于
runs/analysis/layered_relation_writer_20260907；原formal run及三个完整checkpoint路径见§13。K4补充为s96_k4_correct_screen40、s96_k4_vs_source.json、s96_k4_vs_k1.json。

## 15. Native坐标初始化对照的局部收益与边界（2026-09-07）

880bde5e单变量fresh run完成96updates，384条件/6144queries；任务权重、K/视频集合、action episode/frame和policy RNG逐项匹配§14。
更新总1257.37秒，单段含加载1376.95秒，exit0；Writer/Meta均直接联合训练。三个完整checkpoint及七组物化保留。

| 节点 | correct/40 | other/40 | correct对source R/G/L | other对source R/G/L | 跨视频Jaccard |
|---|---:|---:|---|---|---:|
| 16 | 4 | 4 | 3/1/1 | 3/1/1 | 3/5 |
| 48 | 5 | 8 | 2/3/2 | 3/5/1 | 5/8 |
| 96 | 6 | 6 | 2/4/2 | 3/3/1 | 5/7 |

各点breadth2/4、Object/Goal均0。Spatial/Long依次为16c2/2、16o2/2、48c2/3、48o3/5、96c2/4、96o3/3。
48→96 correct保留5、新增1、丢失0、Jaccard5/6；other保留6、新增0、丢失2、Jaccard3/4。
96K4correct为8/40（Spatial3/Long5），对source RGL3/5/1，对原初始化96K4的6/40为4/4/2；
对本次96K1为5/3/1、churn4/40、Jaccard5/9。不能把不同节点/视频臂拼成一个更高分模型。

固定query面板三点均完成，query/noise/time与§14一致、source loss完全相同。96步correct benefit为
[0.0019035,0.0040148,0.0109558,0.0010109]，other也均正。新初始化改善了功能学习并带来局部Long收益，
但未解决广泛行为及两组视频的稳定保持；未进行validation/Test或最终controls，不构成正式qualification。

同两训练任务三代表target的96步冻结诊断表明：native通道常量能量降至70.7%–85.5%，rank常量能量仍>99.995%；
真实B梯度的rank常量能量仅2.56%–30.06%。共享末读出同时汇合不同target/rank的梯度：三个代表target中，
action_out读出梯度norm约0.135–0.316，expert0q约0.000556–0.000974、0v约0.000056–0.000345。
这支持参数共享方式的竞争解释，不是全38target的完整梯度归因。单target局部Jacobian中解除rank共享也并非处处改善方向，
故仍须以受控学习/闭环裁决，不能由几何推定根因或疗效。

原件在runs/analysis/layered_relation_writer_20260907/coordinate_init_control/：七个*_screen40的raw rows/完整completion、
各*_vs_source/baseline/adjacent/correct比较、exposure_cost.json、functional_s*.json及functional_comparison.json、
decoder_gradient_s96.json/.safetensors、readout_tangent_diagnosis.json。正式run为
runs/outputs/layered_relation_short4_coordinate_init_880bde5e_gpu01p235_20260907，eval/批量物化冻结fa0b7b43。

据此保留坐标初始化局部正证据和其它已验证图，预登记一个离散fresh对照：仅将全局共享的A/B末读出[p64]改为
按输出target/rank各自学习[38,16,64]，不引入task-specific参数或旁路。预计增加77,696参数，科学曝光及短闭环口径完全匹配。
注册在runs/analysis/layered_relation_writer_20260907/target_rank_readout_control/registration.json；此处只是实验决定，尚无该改动结果。

## 16. 末读出target/rank共享约束的受控检验（2026-09-07，已完成）

- 接续§15保留坐标初始化的局部Long正证据。只将全局[p]末读出改为[target,rank,p]，新增77,696参数；其它图、std1坐标、
  identity、随机初始化抽样、种子/数据/优化器/96步曝光与world3保持。不是rank/scale/LR扫描，也不据局部Jacobian宣称唯一根因。
- 实现前registration在runs/analysis/layered_relation_writer_20260907/target_rank_readout_control/registration.json；
  canonical实现6ae406ea，182 CPU tests/17.08s通过，包含单输出target/rank更新隔离。
- 新真实GPU机制检查完成exit0：identity后Meta B、随后A梯度可达，source无梯度；full/staged loss同0.1256087869，
  Meta0qB/rho0/decoderB的余弦0.999993/0.999998/1.0。这不构成科学行为结论。
- formal fresh run layered_relation_short4_target_rank_6ae406ea_gpu01p235_20260907从clean pushed detached6ae406ea启动，
  gpu01physical2/3/5，完整合同在该analysis目录launch.json。结果待16/48/96双视频K1及96K4固定配对40裁决；未用validation/Test或负视频controls。

- 本对照96步训练已完成exit0，完整384条件/6144queries与坐标对照的采样/权重/随机数一致，K1/2/4各32条件/task；
  实际更新1263.39秒、总1384.12秒、peak allocated34.323/reserved38.109GiB，新增读出参数未带来明显吞吐代价。
- 最先完成的16correct为7/40（Spatial2/Object0/Goal0/Long5），对source4及坐标16correct4均RGL3/4/1、Jaccard3/8。
  只作为Long早期信号；16other及48/96/K4结果执行中，尚无完整相邻/跨视频结论。

全部七组固定40面板已完成exit0：

| 节点 | correct | same-task-other | correct S/O/G/L | other S/O/G/L |
|---|---:|---:|---|---|
| 16 | 7 | 10 | 2/0/0/5 | 4/0/0/6 |
| 48 | 6 | 8 | 1/1/0/4 | 1/1/0/6 |
| 96 | 11 | 11 | 3/2/0/6 | 3/2/0/6 |
| 96 K4 | 10 | 未设（held池仅4） | 2/2/0/6 | — |

- 两视频16/48/96的Jaccard为.70/.75/1.0；96为完全同一11个成功。对source，16c RGL3/4/1、16o4/6/0、48c2/4/2、48o3/5/1、96c/o4/7/0。
- 16→48 c/o RGL4/2/3、5/3/5；48→96为5/6/1、6/5/2，后者churn均7/40、Jaccard5/12和6/13。相邻仍存在明显变化。
- 96c/o对坐标对照6→11均RGL4/7/2；K4对坐标K4为8→10、RGL4/6/4；本run K1→K4为11→10、RGL9/1/2、Jaccard.75。
  所有普通比较经过完整source/RNG/env/normalizer及video schedule检查；K变更分别与同一source核对后比较success keys，未放宽常规配对器。
- 同一两训练任务/三个target的冻结既有LoRA输出诊断：rank常量能量从>99.995%降至1.55%–6.55%，native常量能量反升到91.2%–99.3%。
  行为提高与几何指标并非整体同向。此结果支持保留末读出干预进行共享学习，不支持宣称唯一根因或继续无依据坐标/rank/scale扫描。
- **裁决：** 96K1两视频11/40且集合完全一致、breadth3/4，提供进入完整train24学习的基础；尚无Goal、未见task迁移、正式相邻稳定或video必要性证据。
  下一阶段保持当前图、完全fresh Writer/Meta及optimizer，将固定train24引入共享训练，先用K1 qualification并继续真实K1/2/4训练。
  未选择最终checkpoint，未运行validation/Test或最终负视频controls。所有raw rows、aggregate、exposure/cost和比较见target_rank_readout_control/。

## 17. train24共享阶段预登记与训练侧参照（2026-09-07）

当前图从fresh Writer+Meta扩展到固定train24，训练frozen e4ca5998，672步完整schedule、首段停192；
完整节点和资格口径在active design§13.3.1与`runs/analysis/layered_relation_writer_20260907/train24_shared/registration.json`。
尚无当前图的validation闭环结果，不能把short4基础行为解释为迁移达标。

新source训练侧参照使用固定train24×states0–4，共120行；frozen evaluator97a8a24a，单GPU3persistent workers，
36分片全部完成exit0，耗时1234.63秒。结果16/120，Spatial6/Object0/Goal10/Long0，breadth7/24；
非零任务为Spatial0/2/4/5分别2/1/2/1、Goal1/2/8分别5/1/4，其他17任务0。
完整结果在`train24_shared/source_train120/`，配对证据在`source_train120_integrated_reference.json`；
只用于当前192/576训练侧correct诊断，不替代validation400、旧short40或其它视频/初始化口径。

首段192步已完成exit0：768条件/12288queries，每task32条件/512queries，K1/2/4各10或11次，任务等权及episode角色验证通过。
22tasks覆盖全部16条训练视频，task15/34各15条。更新3221.41秒、段总3347.23秒，3.814queries/s；
rank0 allocator峰值allocated34.326/reserved38.201GiB，未登记为全rank历史峰值。
`train24_shared/exposure_cost.json`与`completion_to192.json`保留实际曝光、成本及独立分段耗时；
checkpoint为train24 run的`macro_00000192`，尚未产生validation分数或选点资格。

192步训练侧correct诊断完成exit0：22/120，Spatial11/Object1/Goal10/Long0，breadth8/24；对source16为RGL11/11/5、
churn16/120、J11/27，单GPU3workers耗时1316.09秒。训练侧改善有限，未获得Long行为；不据此唯一归因为容量或共享冲突。

192步首个validation correct strict400完成exit0并与历史source47通过完整policy/environment/RNG/normalization配对：
69/400，Spatial3/Object29/Goal37/Long0，breadth5/8，RGL34/35/13、churn48/400、J34/82，四GPU12workers耗时1063.75秒。
per-task（每task50行，source→candidate）：Spatial1:0→2、Spatial3:0→1；Object1:5→29、Object3:0→0；
Goal3:0→1、Goal6:41→36；Long1:1→0、Long2:0→0。
这保留未见Object任务的局部性能增益，但未达到>145/400、breadth、四suite和Goal/Long资格，也尚无相邻稳定性。
完整证据在`train24_shared/s192_correct_strict400/`、`s192_train120_screen120/`及`*_vs_source.json`；
same-task-other随后完成exit0，72/400，Spatial4/Object31/Goal37/Long0，breadth5/8；
对source47为RGL35/37/12、churn49/400、J35/84。对同点correct为RGL60/12/9、churn21/400、J60/81=.740741；
未达到登记的跨视频J≥.80。per-taskother为Spatial1/3:3/1、Object1/3:31/0、Goal3/6:1/36、Long1/2:0/0。
五GPU15workers耗时928.99秒；未开展负视频controls或Test。

192裁决：两组正确视频均有相对source的重复Object增益，保留局部正证据；absolute、breadth、Long及跨视频均未达资格。
每task512queries仍早于short4同task1536曝光，按原预登记协议exact-resume至384（每task1024queries），
模型、优化器、完整schedule、数据、K和world4不变。384双视频strict400用于判断扩展与相邻行为，不凭内部loss选择。
完整原件`decision_after192.json`及`launch_resume384.json`；并未把曝光不足或共享冲突宣布为唯一根因。

## 18. train24相邻384节点：局部Long出现，Goal损失与成功集合不稳定（2026-09-07）

按同一frozenE4、world4和完整672 schedule exact-resume至384，训练完成exit0。累计1536条件/24576queries，
每task64条件/1024queries，24tasks各16训练视频全部覆盖；K、任务权重及跨episode角色验证通过。
累计更新6385.93秒，本段含加载3292.72秒；完整checkpoint、分段completion与曝光成本保留。
两组LoRA物化255/259条件完成exit0，分别约1.32/1.34GB、576.28/597.26秒（含启动，近似）。

两组400行均完成exit0，实际source、环境、normalization、RNG与视频schedule配对通过；每task50行：

| 面板 | 总分 | Spatial1/3 | Object1/3 | Goal3/6 | Long1/2 | breadth |
|---|---:|---|---|---|---|---:|
| 384 correct | 67 | 1/2 | 33/0 | 0/27 | 4/0 | 5/8 |
| 384 other | 64 | 3/1 | 31/0 | 0/26 | 2/1 | 6/8 |

对source47，correct RGL30/37/17、churn54/400、J30/84；other28/36/19、churn55/400、J28/83。
192→384 correct从69到67，RGL46/21/23、churn44/400、J46/90=.511111；other从72到64，47/17/25、
churn42/400、J47/89=.528090。384跨视频67→64，RGL52/12/15、churn27/400、J52/79=.658228。
两面板均使用gpu01四GPU×3persistent workers，墙钟1111.27/1072.95秒；所有raw rows、aggregate、completion与比较保留。

Object1局部增益仍在，Long出现少量成功，但Goal6在两个arm均回落，未扩大总分、correct breadth或稳定性。
该checkpoint未满足absolute/breadth/Long/相邻/跨视频资格，未选selected checkpoint；这不是工程错误或整条图无效的证明。
完整裁决在`train24_shared/decision_after384.json`；不继续以“还没跑完schedule”解释续训。

在任何384训练诊断分数出现前，已登记冻结384 train24×相同states0–4、K1、seed20260907的两组正确视频：
seen0–15与held46–49，分别107/68个唯一条件覆盖120行。已见组120行视频均曾训练，89行也曾作为精确K1条件，
31行视频曾在K2/K4出现；因此不能将整个已见面板称为逐条件训练拟合测量。两组独立与同一source120做完整配对，
然后在不修改raw rows、不放宽canonical视频schedule检查的前提下报告预登记的video-familiarity干预。
这只区分已训练task/视频、新视频与新task的行为缺口，不做最终选择、梯度或因果controls；Test未打开。
原件`video_novelty_diagnostic_registration.json`与`materialization_s384_novelty_launch.json`；此处尚无诊断分数。

冻结诊断随后全部完成exit0：seen21/120（Spatial10/Object3/Goal8/Long0），breadth11/24；
held18/120（10/1/7/0），breadth7/24。对同source16分别RGL10/11/6、11/7/5，churn17/12，J10/27与11/23。
seen→held为RGL14/4/7、churn11/120、J14/25；held192→384为22→18、RGL11/7/11、churn18/120、J11/29。
两组完整task/state/source/normalization/env/policy RNG共同前缀及预登记视频角色均验证，raw rows未修改；
两eval墙钟495.01/503.80秒，物化107/68条件为325.36/256.92秒、约552/351MB。

Owner指出当前分数远低于基础SFT约108（历史相邻109/107），执行裁决当前run永久止于384，不再运行576/624/672。
熟悉训练视频也未建立广泛行为，Long均0，不能仅归因为新视频或未见task迁移；也未唯一定位某个模块。
同两个训练task/三个代表target的已有输出中，native常量能量约64.8%–95.4%，较short4部分改善而行为仍弱，
继续美化坐标几何没有裁决依据。下一步是保留读取/关系/compiler与target/rank独立性，检验直接native线性因子输出的
离散参数化对照；增加参数量与固定投影子空间都是须报告的取舍，不能预告性能或把本诊断称为唯一decoder根因。
原件`video_novelty_diagnostic_results.json`、`s384_train120_*_vs_source.json`、`s384_train120_held_video_vs_s192.json`、
`decoder_output_representatives_s384.json`与更新后的`decision_after384.json`。

## 19. 独立审查补证据与第二轮推导讨论（2026-09-07）

Owner要求先与专家讨论清楚，不能把一轮分析直接转化为实施。此前native线性因子读出提案及专家提出的旧P/Q width256补评均停留在讨论候选；
科研实现、训练与评测暂停，最新授权以progress为准。现存width256训练和checkpoint不等于已有闭环证据。

[审查补充包](review_materials/20260907/README.md)提供当前图、两次读出修正、train24、早期v5.2/v6、SFT、多视频/LPCP/GOMQ和P/Q对照的
现有合同、metrics、exposures、checkpoint元数据、诊断与raw rows公共副本；保留源路径索引并标注转换、缺口和未完成面板。
本次仅整理既有证据，没有新增训练、rollout、模型选择或因果结论。

[第一轮专家意见](review_materials/20260907/expert_review_round1.md)作为待核验论证保留；[第二轮prompt](review_materials/20260907/ROUND2_PROMPT.md)
要求先依据原件修正证据等级，再从需求、信息条件和合法监督逐步推导架构与训练，区分数学性质、假设、归纳偏置与待验证能力，
并用历史成功和失败约束选择。讨论稿及最小验证建议不自动构成执行授权。

## 20. 专家连续推导、Owner裁决与新session交接（2026-09-07—08）

在§19补充原件后，Owner继续要求从证据、信息依赖和实际控制学习推导完整方法。四份后续完整回复保存在
[20260908讨论索引](review_materials/20260908/README.md)，包含第二轮事实修订、过程条件化视觉回读、完整FM/RL协议与最终交替架构。
原文不被改写成最终裁决，也不把专家每一轮建议都作为待实施路线。

最终保留末层PreActionOut而非18层轴；过去4帧软对应后，用一层完整H-query让新关系模式共同影响视觉查询；
每帧对核实两端Z，再按历史u排序短GRU。四组临时H-read/长程/前三组逐H回写，最后集合compiler与完整native D。
专家最后原文推荐双向长程；Owner明确选择单向后，正式设计四组全部改为past+self，H-query沿H双向不受影响。
这一变化只选择了视频表示的可见范围，不声称由前缀因果mask证明视频对行为的必要性。

训练选择为fresh Writer/Meta的FM辅助共享Writer RL，同版本采集和求梯度后一次更新，Gaussian动作探索、LOO baseline、
Q/M decision子采样、实际10步flow反传和有限候选KL检查。旧GOMQ的成功expert occupancy蒸馏不能当作这条新信用路径已通过。

[正式设计](horizon_relation_video_writer_design.md)登记了完整输入/shape/公式/初始化/梯度/数据/采样/恢复/验证/迁移合同；
旧18层设计正文由Git e868de525fda0a20c597dee2c7bffe5717f5e2fd追溯，活动路径仅保留历史入口。
代码仍是旧layered实现，384永久停止，未选selected checkpoint。未补width256闭环、未整支合并native-head草稿。

Owner要求后续由新session推进，本session完成启动前准备、关键资产存在性和代码接口核对及Git交付；无新增科研代码、训练、
rollout、模型选择、GPU准入或配额证据。新session按正式设计落实实现、真实机制/成本与预登记学习，不能恢复旧暂停前清单。

Owner随后明确接班prompt须要求全面理解整个仓库、完整忠实实现最终架构、正式实验与结果驱动迭代，并在新session创建覆盖全程的goal。
这项goal授权属于接班科研任务，不把本session的交接文档工作扩为立即启动实验；没有指定token预算。

Owner又强调不能重犯已知架构/性能问题，必须正视source和SFT参照，并给予新session在整体思想与硬合同内基于证据实质调整方法的充分自主权。
本次从20260907补证据包的三组各400行outcome重新计数，确认为source47、SFT step400为109、step425为107；没有新rollout。
source48属于另一历史面板。接续设计§1.1规定在有信息量的可比正式节点判断基线差距，不能以内部指标或仅高于弱source开脱，
也不能把低分直接命名为唯一根因；已有证据或新实验指向必要改变时，更新正式合同并完整实现，不局限于小补丁。


## 21. 2026-09-08接管与既有证据复核

Owner启动全程科研goal，无token预算、总工期或总尝试次数，授权完整实现、真实验证、正式实验与证据驱动迭代至最终达标。
接管基线4f1686ab；授权/goal与已消费HANDOFF的状态整理于92a2673f提交推送。旧384永久停止，尚无新架构分数。
系统阅读当前源码、测试、脚本、配置与文档，完整阅读20260908四篇专家原文及Owner最终裁决。程序连接index的843个导出，
核对511份本地原件存在，重算95个rollout面板24100行的success/suite/breadth并与原始outcome向量核对一致。

source47原件为`runs/analysis/pi05_ecp_prw_meta73_equal_exposure_20260906/source_strict400/results.json`；
source48原件为`runs/outputs/pi05_source_base_validation8x50_raw_b649ba5_r3_20260722/results.json`。
48→47的R/G/L为45/2/3，五条success不同；400行task/state/language/env/policy RNG共同前缀及policy字段对应。
因此不是仅少一次成功，也不能归因teacher schedule（source没有teacher输入）；后台/批量/数值原因本次没有独立定位。
SFT400/425对应109/107，各400组上述配对字段再次核对匹配source47；保持历史normalization兼容性记录的边界，不冒充新后端评测。

新图最接近的历史约束是：v5.2/v6具备直接视觉/语言与动作响应及自由FactorHeads，143后下降；GOMQ从强carrier继承且成功expert occupancy信用，
151未保持；P/Q/Unified交互未解决shared学习；旧末读出target/rank干预局部改善，但train24 67/64及熟悉21/120仍不足。
新首版新增过程条件的逐H两端Z核实、历史u递推和长程回写，再接直接native因子通道及真实当前policy Gaussian成功信用；
这些新增依赖不证明视频必要性或能力保持，必须由预登记行为节点和最终controls裁决。

旧native-factor草稿只替换18层上游末端，默认factor width64且缺正式GELU；不能整支集成。旧P/Q width256无闭环仍保留该事实。
当前canonical数据/source/tokenizer关键资产核对存在并匹配manifest字节；没有加载大checkpoint、复制数据或刷新GPU/quota。

## 22. 2026-09-08完整Horizon首版与真实联合profile

新canonical图、最终R/Z、native A/B、真实Gaussian Writer RL、同版本FM/RL联合更新、物化/evaluator/checkpoint已集成。
单卡真实机制证明末层post-norm接口、同prefix最终Z、完整10flow VJP与学习后Meta梯度；最长合法K1/K4整段可运行。
这是机制/工程证据，不是科学通过。原件：`runs/analysis/horizon_relation_writer_20260908/mechanism/`。

9c5a1c2b四卡完整profile两轮exit0，8条件/512FM queries/32episodes、两轮各一个mixed reward group，
两轮150.73/149.88秒、含加载保存456.06秒，allocated峰值22.94GiB；0/2更新接受，完整checkpoint1.378GiB。
前两次仅assets路径失败，零rollout/零更新；准确revision经四suite真实env reset/settling修复。
完整profile日志、环境和数值诊断：`runs/analysis/horizon_relation_writer_20260908/joint_profile/`。

同参数trust KL非零恰好在提前成功后活动batch缩小的task2/5/21，分别.01265/.01472/.07968，另外五task均0。
task21全成功且RL梯度0仍超0.02，因此不是信用或参数更新导致其自比较差异。真实复现四episode的成功和步数一致，
64保存decision覆盖batch1/2/3/4；按原size重放及重排行顺序均KL0，强制batch4为.02439。
原profile随机子集16与诊断全64不同，.07968/.02439不作为同口径前后值。修复只记录原batch尺寸并按其重放，
尾组以真实记录补齐、零cotangent/零计分；保持原m_old、Sigma、threshold、候选Meta和完整图。尚无修复后正式学习结论。

07871988 batch修复后两段profile共8attempts、1accepted，所有current-version task KL0，接受后Meta梯度非零；
完整同拓扑exact-resume的下一抽样与checkpoint全部随机流吻合。计算2048 queries，仅256参与接受更新，不能等同有效学习。
8次只有1次接受，尚无formal学习或qualification结果。固定旧A、new B有限探针的0/tiny输出与旧均值一致，
有限B响应不单调，没有证明固定非零B分支；详见joint_profile/numerical_limit/。后续登记只扩同方向有限回溯到1/128的
优化假设，保持Sigma/原m_old/0.02及科研性能线，先验证可行更新和成本，不以接受率代替行为。

source训练任务新初态32–36的J0/JΣ120各完成，19/22，S/O/G/L=8/0/9/2和10/0/10/2、breadth7/24；
strict paired R/G/L=15/7/4、churn11/120、J=.5769。原件`runs/analysis/horizon_relation_writer_20260908/source_train120/`。
这是新train诊断baseline，不与旧初态train16或validation47混用。

八档有限回溯ae9507b5完整四轮仍1/4接受，与前版前四轮相同；新增1/16至1/128没有贡献接受更新。
step2/3/4在1/128的最大task KL=.02300/.07222/.03424，全部16个同版本自比较为0；四轮1024计算queries仅256进入接受更新。
该证据不支持“只扩展有限搜索范围即可解除停滞”，停止同样续试。后续固定原B/相同重放/相邻BF16 B诊断定位执行端，
不以内部数值代替闭环资格，也不根据此处训练诊断调整科学性能线。原件joint_profile/finite_ae9507b5/。

相邻B固定诊断（joint_profile/adjacent_b/）完成exit0：同B重放KL0，B整体朝+infinity一个BF16相邻值
(relative L2 .57214%, max delta9.53674e-7)时KL=.034327、max动作差=.0512085。固定A/输入/epsilon/原batch，
Writer只生成一次；这定位到执行端足以产生变化，未证明精度缺陷。后续仅动作expert FP32/TF32因果对照另登记，尚无新formal协议或性能。

限定expert FP32/TF32诊断（expert_tf32/）相邻B KL=.000257，但baseline对旧训练外层BF16 KL=.921861，
不能视为原policy或据此直接采用。随后静态与真实native_boundary/验证发现更早的合同差异：evaluator无外层autocast，
旧训练flow有；旧机制parity把两边都包入同一context，未覆盖实际evaluator。非零LoRA canonical-vs旧训练KL=.509307，
禁用flow/prefix外层autocast后vs canonical=.014033，旧batched-vs physical=.016349，相邻B仍.030222。
据此先恢复原生执行类型，不扩大模型dtype；CPU非零LoRA舍入边界另外复现batched预先cast delta与物理PEFT不同，
v4一并按PEFT相加后cast修复。正常低位差异不作为新增逐元素一致要求；完整学习与行为仍待新合同验证。

e1ea3596 native_fixed/非零LoRA64行真实flow对物理canonical KL/max/mean均0，13.29秒/10.70GiB；
真实batched仍KL .016349、max .044685，12.67秒/10.72GiB。物理36 expert targets A/B为BF16，2 head targets为FP32；
先前全state.float的诊断不代表物理参数类型。没有扩大source权重dtype或强求kernel逐元素一致。
同commit完整fresh四卡profile正常exit0：2attempts/2accepted、alpha1及1/16，同版本8个task KL全0，
每轮144.49/156.98秒、总446.30秒；512 FM queries全部进入接受更新、32 episodes、1 mixed RL group，第二步Meta梯度1.17e-7。
峰值allocated25.69GiB/reserved30.37GiB，完整macro2 checkpoint4.13GiB。原件joint_profile/native_e1ea3596/；
这是机制/成本证据，没有新架构formal score。此后Owner明确切换为先纯监督后独立RL，联合formal未启动。

## 23. 2026-09-08 Owner改为监督平台后独立共享RL

Owner明确覆盖首轮FM/RL混合默认：完整架构保持，fresh纯FM端到端Writer/Meta，无RL rollout/loss/trust/回滚。
全部joint profiles及唯一checkpoints保留为历史机制/成本证据，不能初始化正式监督或冒充监督结果；执行一致性修复保留。
平台结合实际曝光、held-action验证、训练task闭环与预登记validation相邻节点；充分监督仍弱要先定位并允许实质改进，
不以饱和为由交给RL救场。后续独立RL从单个保留监督checkpoint初始化，新optimizer/scheduler，默认不混FM，探索/信用/约束另审。
本次正式监督节点/判据见active design §8.2；未报告任何监督科学分数。

265ef31b正式pure-FM首段24正常exit0：96conditions/6144queries，K1/2/4=34/30/32，实际23/24task有曝光，task4尚未抽到。
平均30.40秒/update，总1027.99秒；完整macro24为4.13GiB，optimizer/scheduler/sampler均24，21/24task固定held-action FM改善，
等权mean .151447→.131235。该节点仅早期监督获取，不能宣称平台或闭环增益；J0 train120随后执行。
原件`runs/analysis/horizon_relation_writer_20260908/supervised/step24_summary.json`及run根目录完整metrics/exposures/diagnostics/checkpoint。

同checkpoint24的held-video J0 train120全部完成：17/120，S/O/G/L=6/4/6/1、breadth8/24；source19/120为8/0/9/2、breadth7。
strict paired retained/gained/lost=7/10/12，churn22/120，success-set J=.24138。五卡×两replicas耗489.22秒，全部exit0。
这是监督loss下降而总体闭环未增益的早期节点，不是平台或方法成功；继续预登记64节点并首次strict400。
原件`supervised/train120_step24/paired_source_comparison.json`及run下`evaluation/train120_step24_J0/`。

同一监督run从macro24完整恢复至64，源码仅将不可变query全局索引移出逐query循环，samples/RNG/目标保持；
frozen9ab1e710段40updates平均32.03秒，总1532.41秒，未见相对首段的明显整体提速。
累计256conditions/16384queries，24tasks各6–17次曝光，K1/2/4=87/77/92；完整macro64为4.13GiB，正常exit0。
固定held FM .131235→.123122，24/24tasks相对24改善；相对初始.151447为23/24task改善。
这是持续监督学习证据，尚无64节点闭环分数，不判平台；原件`supervised/step64_summary.json`及run根完整记录。

64节点train120 held-video J0完整exit0，471.70秒：34/120，S/O/G/L=12/7/12/3，breadth14/24。
vs source19的R/G/L=13/21/6、churn27/120、J=.325；vs step24的17，R/G/L=14/20/3、churn23/120、J=.37838。
相比24，Spatial/Goal成功全部保留并新增；Object丢2新增5，Long丢1新增3，仍须正视成功集合变化。
这是训练task获取增益，不能替代validation资格。原件`supervised/step64_evaluation/train64_vs_source19.json`和`train64_vs_step24.json`。

64首个validation correct 400行完整exit0（后审计为重复视频抽样，不合规）（949.35秒）：99/400，S/O/G/L=9/53/35/2、breadth8/8。
global tasks1/3/11/13/23/26/31/32分别5/4/31/22/1/34/1/1。vs source47的R/G/L=35/64/12、churn76/400、J=.31532。
对历史SFT109/107，R/G/L分别60/39/49与55/44/52，churn88/96；SFT400 S/O/G/L=0/69/22/18，
因此当前Long/Object弱于SFT，Spatial/Goal较高，不能将不同成功集合压成仅差10分。旧SFT v1/后端/rank边界明确保留。
尚未>145、Long2<10，无相邻稳定性或视频因果证据；这是旧重复视频分布下的探索增益，不能作正式泛化或qualified checkpoint证据。
原件`supervised/step64_evaluation/validation_correct_vs_source47.json`与`validation_correct_vs_historical_sft.json`。

64 same-task-other 400行完整exit0（后审计为重复视频抽样，不合规）：95/400，S/O/G/L=4/51/36/4、breadth7/8；
global tasks1/3/11/13/23/26/31/32分别1/3/31/20/1/35/4/0。correct99→other95的R/G/L=77/18/22、
churn40/400、J=.65812；总分只降4但J不达.80，Long无correct成功保留。vs source47为34/61/13、churn74。
对SFT109/107，R/G/L分别64/31/45与58/37/49，仍保留历史后端/不同rank边界。
完整64裁决：持续监督学习与train/held获取，但未>145、Long弱、无相邻稳定且跨视频J不足，因此继续预登记128，不转RL。
原件`supervised/step64_evaluation/decision_after64.json`及全部strict comparisons。

两worker correct/三worker other分别949.35/1405.26秒；启动124.97/548.84秒，shard窗口820.09/852.64秒，
实际env steps120809/120820，后者15worker全部权重成功、无OOM，实测最拥挤42372/46068MiB。
不同视频arm并非纯replica因果实验，但三worker没有实际吞吐收益，后续回到两worker，不继续资源档位扫描。
只读加载路径核查发现大Gemma/PaliGemma重复构造；GPU allocation发生在checkpoint加载前，不能当作已加载权重。
未改第三方loader、未引入no-init；准确边界与候选限制记在`evaluation_replica_choice.json`，不改变原始计算记录；视频schedule的科学适用范围见下文更正。

## 2026-09-08 Horizon视频schedule合同更正

旧混合K64的correct99/other95保留为重复teacher抽样下的探索结果，不满足每task50视频各一次的正式合同；撤销其合规strict400身份。
实际correct为255个task-video条件/400行、other259/400；八task每臂仅28–36条唯一视频（other29–36），存在重复。
原success、R/G/L、churn和当时决策记录不改，但关于泛化、SFT/source比较及跨视频鲁棒性的结论仅适用于旧重复视频分布，不能作为资格证据。
旧train24/64的17/120与34/120均为held池46–49在5个init中重复复用，共71条件/120行；这是同一旧映射下的诊断变化，不代表整轮视频覆盖。
Fresh K1的100/200训练checkpoint保持有效；其初次物化也错误地只有255条件，已停止错误schedule评测（rollout前SIGINT、无结果），将复用合法LoRA并补齐新canonical映射。

修复后f1330697复用canonical排列及+17规则；fresh100/200各255个合法旧LoRA hardlink复用、145个新编译，正常exit0、总墙钟357.89/357.22秒。最终manifest实检各400条件/400行、八task各50视频一次、两checkpoint完全相同mapping；相关74项回归通过。原件`k1_fresh/schedule_repair/final_manifest_coverage.json`。

## 2026-09-08 Fresh K1首个canonical correct400节点

macro100（fresh K1累计25600 FM queries）correct400完整exit0：55/400，S/O/G/L=0/22/27/6、breadth6/8，global tasks1/3/11/13/23/26/31/32为0/0/21/1/2/25/4/2。相对同source47的R/G/L=25/30/22、churn52/400、J=.32468。实际rollout400行的teacher身份匹配新canonical bank，每task50条各一次；六worker全部exit0，完整墙钟1391.43秒（launcher1324.93秒）。
未达>145、Spatial为零、Long6<10；这是单个早期监督节点，不能判断平台。旧mixed64的99/95同时具有不同训练条件和错误视频分布，不从两者分数差异归因K1或schedule。原件`k1_fresh/schedule_repair/step100/completed_summary.json`、`correct_vs_source47.json`及`evaluation/validation_correct_step100_schedule_v2_gpu3_J0/`；继续预登记200并做相邻paired分析。

同一macro100对历史SFT400=109与SFT425=107，R/G/L分别39/16/70和38/17/69，churn均86/400、J=.312/.30645。固定source/policy/environment/RNG及normalization的既有历史兼容性检查通过；仍保留历史rank128、旧后端且未在当前backend重跑的边界，不作为隔离架构或K因素的实验。原件`k1_fresh/schedule_repair/step100/correct_vs_historical_sft.json`。

## 2026-09-08 Fresh K1 macro200及首个相邻比较

macro200（累计51200 FM queries）canonical correct110/400，S/O/G/L=1/61/41/7、breadth6/8；global tasks1/3/11/13/23/26/31/32为0/1/37/24/0/41/4/3。全部400条实际视频身份匹配同一canonical schedule，六worker exit0，完整墙钟1375.21秒（launcher1300.95）。vs source47 R/G/L=41/69/6、churn75/400、J=.35345。
100→200从55升110，R/G/L=42/68/13、churn81/400、J=.34146；S/O/G/L净增1/39/14/1。Long R/G/L=0/7/6，旧成功完全未保留；Goal为24/17/3，Object18/43/4。breadth仍6，200的global1与23为零。总分增长不能替代稳定性；仍未>145，Long7<10、相邻churn/J不合格。
对历史SFT109/107，R/G/L为71/39/38和68/42/39，churn77/81、J=.47973/.45638；总分相近但成功集合不同，历史rank128/旧backend限制保持。原件`k1_fresh/schedule_repair/step200/completed_summary.json`、`correct_vs_historical_sft.json`、`correct200_vs_correct100.json`和`decision_after200.json`。
因直接闭环仍显著获取且未形成平台，保持纯FM与全部学习状态，预登记200→400约一小时、300/400两个correct节点；不做架构重启或超参小扫，other与最终controls暂缓。

## 2026-09-08 Fresh K1 macro300及200→400续训完成

原b6d70d98学习状态续训200→400完整exit0，墙钟3287.09秒，段内均值15.583秒/update，峰值38.170GiB；累计1600个真实K1条件/102400queries，24task各46–86次曝光。300/400完整checkpoint均通过正式检查，未重置或改变训练超参。
300/400 banks各400个新条件、分别791.57/818.26秒完整生成，正式检查及实际50视频整轮覆盖通过，与100/200同一canonical state-video映射。
macro300 correct400=86，S/O/G/L=0/37/45/4、breadth5；global1/3/11/13/23/26/31/32为0/0/36/1/0/45/2/2。全部400rows实际视频身份验证通过，六worker exit0，完整墙钟1348.54秒（launcher1322.565）。vs source47 R/G/L=43/43/4、churn47、J=.47778。
200→300从110降86，R/G/L=71/15/39、churn54、J=.568；S/O/G/L的R/G/L分别0/0/1、30/7/31、40/5/1、1/3/6。Object global13原24个成功全部丢失，仅新增1；Object global11净-1，Goal净+4。这是合规映射下的行为退化，未建立工程错误或监督平台，不能由单次下降推翻整个图。
对历史SFT109/107的R/G/L为53/33/56和52/34/55，churn均89、J=.37324/.36879；历史rank128与旧backend边界保持。原件`k1_fresh/segment200_400/step300/completed_summary.json`、`correct_vs_historical_sft.json`、`correct300_vs_correct200.json`和`decision_after300.json`。继续预登记400 correct400后再裁决，不提前启动other/final controls。

## 2026-09-08 Fresh K1 macro400及后续观察

macro400 canonical correct87/400，S/O/G/L=0/40/42/5、breadth4；global1/3/11/13/23/26/31/32为0/0/37/3/0/42/5/0。六worker exit0，全部400rows实际视频映射验证通过，完整墙钟1351.46秒（launcher1284.764）。vs source47 R/G/L=40/47/7、churn54、J=.42553。
300→400 R/G/L=70/17/16、churn33、J=.67961；S/O/G/L分别0/0/0、30/10/7、39/3/6、1/4/3。总分86→87但breadth5→4，尚未恢复200的110，更未通过绝对性能、Spatial、Long和相邻稳定资格。
历史SFT109/107比较R/G/L为54/33/55和54/33/53，churn88/86、J=.38028/.38571，原rank128/旧backend边界保持。原件`segment200_400/step400/completed_summary.json`、`correct_vs_historical_sft.json`和`correct400_vs_correct300.json`。
同checkpoint另行复用原固定train24 held-action面板，原video/action demos/action frames/policy RNG/128queries逐task一致；原run config及0/200共48条diagnostics未修改，无梯度/optimizer、sampler未消费。
held FM0/200/400=.15145673/.11135264/.10627193，200→400改善4.563%、23/24task下降，仅task16微升.000386；完整墙钟427.57秒，峰值11.484GiB。400采用单卡micro4，0/200原world4 micro8/4/8/8；接受正常执行低位差异。诊断不选择checkpoint、不替代闭环，原件`segment200_400/step400/held_action/comparison_0_200_400.json`。
Owner明确要求继续观察400及以后，因为历史强架构也可先升后降再升；当前held FM仍改善，未建立监督平台。完整保留原学习状态与图，预登记400→600、500/600两个correct400节点，不因300/400回落重启、换架构或小扫超参。下一段注册`k1_fresh/segment400_600/launch_contract.json`，裁决原件`segment200_400/decision_after400.json`。

400→600刚启动后Owner要求暂停并仔细分析；已在首个新update前SIGINT退出，末checkpoint/metrics仍400。后续节点准备转为inactive，不构成继续执行授权。本次只读分析显示成功集中同一对象簇、训练条件曝光与历史task-complete方案明显不同，但没有单因果结论；原件`k1_fresh/paused_review_20260908/analysis.md`。


## 2026-09-08 恢复原样至600与冻结200/400训练任务诊断

Owner解除暂停并授权持续推进，600只是一轮节点。400→600保持科学配方与学习状态完整恢复；因共驻余量将物理FM分块改为2/4/8/6，逻辑4task×64queries及权重不变。首次恢复401前反向OOM未更新，第二次从完整400成功exit0。
200次更新完整墙钟3562.43秒、均值16.821秒/update；500/600完整checkpoint保留且通过public检查，累计2400个K1条件/153600queries、382/384种task-video。原件`k1_fresh/segment400_600/training_completion_summary.json`，500/600 closed-loop另记，不用训练完成代替性能。

冻结200/400的train24×states32–35、每task held videos46–49各一次，实际teacher/state/env-policy RNG与source身份配对通过；无梯度，不选checkpoint。
200/400为52/59成功（各96），S/O/G/L=12/15/16/9与17/18/17/7，breadth20/21；source同状态固定子集15/96、breadth7。
200→400 R/G/L=45/14/7、churn21/96、J=.681818；vs source分别13/39/2与12/47/3、churn41/50、J=.240741/.193548。完整墙钟465.90/554.68秒，两个8worker面板全部exit0。
训练侧总体获取与validation110→87不同向，支持优先审视跨task泛化而非全局训练失败；Long9→7仍有局部缺口。4状态/task只作定位，不直接证明根因或稳定。
条件曝光200/400=800/1600、queries51200/102400，实际task-video341/378（总池384）。原件`k1_fresh/train96_diagnostic/{analysis.md,comparison.json,evidence_statistics.json}`，历史边界见同目录`history_boundaries.md`。


## 2026-09-08 Fresh K1 macro500/600：局部回升，没有广度恢复

500/600 canonical correct70/82（各400），S/O/G/L=1/32/34/3与1/32/46/3，breadth均4/8；global1/3/11/13/23/26/31/32分别0/1/32/0/0/34/3/0与0/1/32/0/0/46/3/0。两节点8worker均exit0，实际每task50视频各一次且跨checkpoint配对一致，完整墙钟1192.66/1252.58秒。
400→500 R/G/L=62/8/25、churn33、J=.652632；500→600=56/26/14、churn40、J=.583333。600净增12全部来自Goal26；其它三个suite总分不变且成功集合仍变化。400→600=68/14/19、churn33、J=.673267；200→600=63/19/47、churn66、J=.488372。
vs source47，500/600分别33/37/14与41/41/6，churn51/47、J=.392857/.465909。历史SFT109比较为43/27/66与49/33/60，SFT107为41/29/66与51/31/56；保留rank128/旧backend限制。
500/600累计2000/2400条件、128000/153600queries、379/382种task-video。完整原样续训一轮没有恢复200的110或任务广度；600有78/82成功集中11/26，加31为81/82。不支持无限原样续训，也不能证明架构整体无容量。
原件`k1_fresh/segment400_600/{round_analysis.md,round_evidence.json}`及step500/step600完整比较。在600分数前登记的同口径train96检验后期训练行为是否也退化，不用于选点；据此继续机制诊断。


## 2026-09-09 冻结600训练面板完成与只读原因审计

原K1冻结600 held-video train96正常完成67/96，S/O/G/L20/17/18/12，breadth20；400→600 R/G/L51/16/8、churn24/96、J=.68；200→600为44/23/8、churn31/96、J=.586667。原始96行state/video/RNG配对与8worker完成均已检查，完整墙钟542.38秒。训练600四个零task9/15/38/39分别已有115/97/117/95个条件，不能简单归为未曝光；12个task在该小面板4/4，不代表普遍解决。

Owner随后停止条件组织分叉，并要求仅深入分析已有证据，正式及其它新实验均未启动。只读审计核对原b6d70d98训练、f1330697评测、44个旧400行面板、专家原文与机制历史，CPU重算当前原始结果/采样/模块学习状态及已存LoRA几何，没有新增forward或rollout。
历史四任务v5.2/v6已有132/121，全任务配方效果又方向相反；不能只凭旧v6-fast143将当前弱分优先归因4task。600的81/82成功集中source原成功对应的三个任务，而200这些任务外28成功到600只剩1；同时train held52→59→67。最受支持的是训练分布学习未稳定转化为跨task能力，具体架构/表示/共享与FM/优化作用未被单独识别。
完整报告`docs/horizon_k1_evidence_review_20260909.md`；CPU派生统计`runs/analysis/horizon_relation_writer_20260908/k1_fresh/evidence_review_20260909/`。已写但未运行的24task源码/config/tests保留为暂停草稿，不算任何科学结果，不由本次报告恢复。

## 2026-09-09 首层语言内容对照首段：早期增益未成为200整体提高

`fea45593`从fresh seed7仅去掉首compiler的直接language内容残差，保持首cross语言检索及全部其它模型/4×64 pureFM口径。实际800条件/51200queries与原基线task/video/query/seed/权重一致，完整200更新与100/200 correct400、200 train96全部exit0。

新100/200为75/110，对照原55/110；breadth均6/8。新相邻R/G/L55/55/20、churn75、J=.4231；原200→新200为86/24/24，同分110。新S/O/G/L1/36/31/7→3/59/38/10，global1/23未获取。新200 train96为46、breadth18，原52、breadth20，R/G/L39/7/13。相同3072query held FM新/原.111184/.111353均值接近，不能替代行为结论。

只支持早期validation增益及部分能力交换，未证明整体修复、唯一根因或资格。完整报告`docs/horizon_k1_first_query_only_20260909.md`，原件`runs/analysis/horizon_relation_writer_20260908/k1_first_query_only/first_segment_evidence.json`及新run `runs/outputs/horizon_k1_first_query_only_v1_seed7_20260909/`。保留既有近等价历史边界；后续状态只由progress/active design解释。

## 2026-09-09 首层语言内容对照至400：局部保持收益，目标未扩展

同一`fea45593`完整200状态exact-resume到400；实际201–400与原模型800条件/51200queries逐条匹配，累计1600条件/102400queries。300/400完整single checkpoints、各400新LoRA bank、400 train96与冻结held3072均保留并正常结束。200更新均值14.771秒，完整3138.75秒；300/400/train96 wrapper1555.43/1398.37/465.89秒。

新300/400 correct106/103，S/O/G/L0/64/35/7→3/52/39/9、breadth4→6；原86/87。新200→300 R/G/L77/29/33、churn62/J=.554；300→40077/26/29、churn55/J=.5833。BBQ新28→24→10，原24→1→3，回落部分缓解但仍复现；其它任务补偿使总分只由110→106→103。两个validation task始终0，未满足目标线、breadth、Long或稳定性。

400 train96=59、breadth22，对新20046为R/G/L35/24/11，对原40059为46/13/13；S/O/G/L12/21/19/7，对原17/18/17/7。训练侧获取仍在，validation未持续扩大。冻结held FM.105737594，对原.106271931，11/24任务更低；无梯度/optimizer且sampler不变。严格配对、六worker exit0和全部原件见`k1_first_query_only/segment200_400/round_evidence.json`及[完整报告](horizon_k1_first_query_only_20260909.md)。

本轮保留内容移除的局部收益，未认定唯一根因或整体修复，不原样追加500/600。下一步机制分析与任何新实验由progress及后续登记解释；本历史段不恢复执行。

## 2026-09-09 首层内容对照后的冻结功能对应矩阵

完整按§8.2.7执行新200/400×teacher46/47×train24完整adapter，应用于固定train24的32query；加source共74,496query预测。所有actual time/noise配对，无Writer forward、梯度、optimizer及新rollout；sampler不变，GPU02p2单卡449.10秒、peak11.750GiB、exit0。首个重复full-prefix实现于首行完成前终止，日志保留；冻结前缀缓存执行首行source/own核对在预登记数值容差内。

自身FM200→400=.111621→.106557，source=.154875；其它23task−自身margin .009396→.015491，同suite .003599→.005578。400两teacher两半面板均优于其它23task均值为24/24，同suite为18/24，训练内任务特化继续增强。此结果支持把持续训练获取与验证退化分开，不证明充分语义理解、不用微小FM排名选择adapter、不替代视频因果或闭环。

[完整报告](horizon_k1_functional_assignment_20260909.md)与`runs/analysis/horizon_relation_writer_20260908/k1_first_query_only/functional_assignment/`保存注册、两执行版本、完整24行及汇总。后续学习干预尚须另登记，本段无新结构或训练启动。

## 2026-09-09 逐帧上下文过程条件首段：52/103，尚无整体收益

冻结运行面`9abc9b95`只把四组静态语言条件换成同次Gemma逐帧exact task-token状态的learned read，复用原参数；compiler保持first-query-only，完整H/四组/过去依赖/视觉核实/native D与采样保持。fresh200更新800条件/51200queries与前轮实际曝光逐条匹配，完整3413.33秒；100/200完整checkpoint和896个新bank条件均通过检查。全部训练、物化和闭环exit0。

100/200 correct52/103，对前轮75/110；S/O/G/L0/26/22/4→1/54/37/11、breadth6→7，200两个弱task各仅1/50。相邻R/G/L38/65/14、churn79/J=.3248；前轮200→本轮76/27/34、churn61/J=.5547，其中Long仅2/9/8。200 train96=41，对前轮46为35/6/11，S/O/G/L11/11/13/6、breadth18。配对held FM200 .111031413，对前轮 .111183766，均值接近没有变成行为收益。

本首段未证明语义访问有效、稳定breadth或模块根因。相对缺口23→7、自身52→103仍获取，被登记为补300/400区分较慢获取与保持的依据，而非正结果。进一步执行状态只看progress/active design，不由本历史段恢复。完整[报告](horizon_k1_frame_contextual_20260909.md)、`k1_frame_contextual/first_segment_evidence.json`、两个`step*/completed_summary.json`及`train96_step200/`保留逐task/suite、成功集合、配对与退出证据。

## 2026-09-09 上下文条件完整400：90/400，保持干预无整体收益

同一9abc9b95完整200 exact-resume至400，全部学习状态/采样/topology保持，新增800条件/51200queries与前轮逐条配对。300/400 correct79/90，前轮106/103；S/O/G/L1/33/36/9→2/44/35/9，breadth均6。200→400 R/G/L56/34/47、churn81/J=.4088，BBQ25→3→1且原25成功全丢失。

400 train96=49、breadth20（前轮59），对本轮20041为30/19/11；held3072 FM .105074533（前轮.105737594）。训练、checkpoint、全部896新bank条件、闭环配对与worker退出均通过；续段wrapper3308.24秒，300/400 validation1981.63/2055.91秒，train96673.07秒。本轮不追加500/600，转Owner最新授权的诊断分析，禁止正式架构/训练改动和正式launch。

完整[100–400报告](horizon_k1_frame_contextual_20260909.md)，原件`k1_frame_contextual/segment200_400/round_evidence.json`；诊断另见[报告](horizon_k1_causal_diagnostics_20260909.md)，不从本历史段恢复实验。


## 2026-09-09 完整原因诊断：676行闭环与固定接口反证

在Owner禁止正式架构/训练改动和正式launch的范围内，完成监督/历史审计、B1冻结分支、B2完整native FM与10步采样、B3 480行分支闭环、八task三层final64局部求解/新time-noise复核、192行接口闭环及四条正常/专家目标回放。所有最终worker exit0、实际配对与冻结边界通过，source运行面9abc9b95不变。

B3 normal/H-read零/Compiler零/visual零/all-language零为51/54/52/54/40（各96），单分支净效应小而得失混合。C source/normal/P4/C/A-B/expert为4/18/17/20/15/16（各32）；task7 free-C局部4/4对normal1/4不构成整体修复。A/B原fit改善在新噪声下仅保留6.8%，独立episode八task全部变差，局部fit不能当容量或有效学习的上界。两条Long正常回放已经选择第二对象后执行失败，旧专家同状态也未完成相同目标。

[完整报告](horizon_k1_causal_diagnostics_20260909.md) §13给出原因排序与方案，原件总索引`runs/analysis/horizon_relation_writer_20260908/causal_diagnostics_20260909/summary.json`。未使用最终视频controls/Test或held梯度，没有正式模型/配方修改或训练启动；本历史段不恢复实验。

## 2026-09-09 后端条件fresh学习：联合检索删除有收益，保持缺口仍在

隔离探索运行面e60a7ca0，none关闭local/H-read/Compiler额外条件；local_only保留local、关闭两检索额外条件。两臂各fresh400、固定200/400，与contextual实际task/video/query/RNG曝光逐项匹配，source/信息墙/纯FM/完整50H保持；完整checkpoint、8个新闭环共1984行、bank与同节点/相邻strict配对均通过，所有训练和评测exit0。

validation contextual/none/local_only为103→90、108→92、114→110；train96为41→49、39→57、46→56。400 local S/O/G/L0/57/36/17、breadth6，对contextual R/G/L74/36/16，自身相邻70/40/44/churn84；none自身相邻65/27/43/churn70。BBQ none23→1（0/1/23）、local34→18（12/6/22）；其它7task85→91、80→92。local相对first-query-only验证200/400仅+4/+7，400训练56对59，不能将恢复contextual损失称为整体能力突破。

首轮支持local保留、联合去掉两额外检索条件的总分收益；不分辨H-read与Compiler单独作用，不证明主要原因或视频动态机制已解决。none未修复训练/迁移分歧，local仍有弱任务和保持问题。未读取Test、未用held梯度、未使用最终视频controls、未正式采纳候选；后续分离学习只由progress的新登记解释。完整原件`runs/analysis/horizon_relation_writer_20260908/causal_learning_20260909/language/completed_language_matrix.json`及各节点比较。

## 2026-09-09 检索条件2×2：仅关闭Compiler取得126，转入固定候选复核

隔离运行面45e16633完整训练local_h_read/local_compiler，各fresh400及200/400 checkpoint；保留local上下文，补足all/local_only的检索条件矩阵。两个新臂合计8个面板1984行完整exit0，全部worker0、实际训练曝光及bank/闭环strict配对通过；没有Test、held梯度或最终时序controls。

local_h_read validation108→126，train40→56；400 S/O/G/L1/70/36/19、breadth7，对all90为R/G/L74/52/16，对自身200为82/44/26。BBQ29→32（23/9/6），其它7task79→94。local_compiler validation98→90、train43→65，BBQ21→2且原21成功全丢。Compiler关闭在两个H-read背景、两个验证节点均正效应；H-read关闭方向不一致，联合删除并非400最优。候选仍有Spatial和其它弱任务缺口、高churn；single seed结果不能等同可靠整体修复。

完整原件总索引`runs/analysis/horizon_relation_writer_20260908/causal_learning_20260909/retrieval/completed_retrieval_matrix.json`。此批完成后优先复核固定Compiler简化候选，未正式采纳、未合入canonical科学行为；动态launch只按progress，不从本历史段恢复任何实验。


## 2026-09-10 Compiler固定确认：净增方向保留，稳定保持未复现

固定init11 all/local_h_read fresh400及200/400 validation400/train96共1984行完整；冻结init7两臂两节点换正确视频关联共1600行完整，另有预选BBQ八次描述回放。全部实际同节点/相邻strict配对通过、worker均0，训练/物化/评测进程全部正常结束。探索科学实现来自45e16633，评测0e2a3a44仅修正资源准入；不构成正式方法采纳。

init7原关联all103→90、候选108→126；新关联99→82与101→126；init11为100→108与104→119。init11训练任务all41→56、候选44→61。删除Compiler额外仿射query支路的验证增益方向保留，但400净效应从init7+36变为init11+11。init11候选BBQ35→19、保留15/丢20，整体相邻67保留/52新增/37丢失、breadth7→6、Spatial1/100，稳定保持修复没有复现。

该结果识别局部支路贡献，不解释全部获取/保持缺口或旧v5.2/v6配方分差。本节保留结论，完整原件见`runs/analysis/horizon_relation_writer_20260908/causal_learning_20260909/compiler_confirmation/completed_confirmation_matrix.json`；后续实验与授权仅看progress，不从此历史条目恢复。

## 2026-09-10 当前D绑定与机制深化：训练获取改善，迁移保持未修复

当前前端、全语言路径、rank16、source、合法teacher/query池、纯FM与实际采样协议保持，只将同target跨rank的A/B两侧D绑定；fresh seed7固定400。科学运行面c63f55dc，400及后段评测e7447291仅修正按worker显存准入；探索科学实现未合入canonical方法。训练完整exit0、两个checkpoint及1600条件/102400queries实际配对通过，held3072无梯度、mean .114600358→.107955018。四个200/400行为面板992行完整，全部worker/父进程exit0和实际strict配对通过。

共享validation115→82，对all103→90，400同节点R/G/L63/19/27、breadth6→5；共享自身59/23/56、churn79/J=.4275、breadth7→5，S/O/G/L3/61/38/13→1/42/35/4。BBQ26→3丢23，Long早期13个成功全丢。训练侧却32→60，对all41→49，400同节点41/19/8；自身29/31/3、保留90.6%、breadth16→20。该具体绑定的早期验证优势未保持，后期训练获取与未见任务保持继续分离，不作为本轮修复采纳。

结果前登记的CPU只读机制分析进一步定位：八个Compiler checkpoint的额外共用语言位移在LayerNorm前强烈压缩第一层608个query的差异（相对同权重置零反事实只剩3.5%–4.6%），但没有实际K/V attention或独立中介行为证据，不能称注意力饱和或唯一根因。共享两节点的生成B/BA近单方向，D_B字典自身有多方向且后期更分散；形式rank16未被强制降为1。该几何与训练闭环60并存，不能代替性能或被直接命名为缺陷。绑定的梯度聚合/AdamW共同改变，五个相关400日志均未实际触发clip1。

Compiler候选init7/init11的验证108→126、104→119与训练40→56、44→61都仍有净学习，保持未修复不等于平台或再训无效。本轮停止来自Owner明确完整训练上限与复核边界，不是参数/性能穷尽证明。未识别的查询中介、读取侧可学习性、任务覆盖和旧新训练配方仍保留；不以本负结果否定全部共享、FM或未来组合。

独立原因报告已按Owner要求删除，结论直接在对话中解释；逐task/suite与成功集合原始证据保留，共享总索引`runs/analysis/horizon_relation_writer_20260908/causal_learning_20260909/sharing/completed_matrix.json`，机制原件同root的`mechanism/`。全部本任务GPU与观察进程已结束；未正式修改/采纳方法、未续401或新增完整训练、未用Test/held梯度/最终视频controls。本历史段不恢复执行，后续须Owner复核后重新授权。


## 2026-09-10 真实读取、信用及执行机制补充分析

Owner要求把原因分析继续做深，在已完成的最后共享训练之后，仅新增有明确问题的小预算冻结诊断。全部10checkpoint×32task×2teacher=640真实视频forward完成，all/off第一层真实读取分工差异得到确认；第二层仍广播，C在task间仍有明显变化。attention宽分布，不能称饱和。共享D的局部SGD核H H^T在rank差异子空间/共同方向的平均响应约3.2e-5，解释代码相似与输出绑定的学习耦合，不能独自证明近单方向造成泛化失败。

32条件/64组train-only reader FM信用分解，加400两初始化8条件/16组固定Adam度量，均未支持强稳定支路冲突。已有BBQ四state、两轨迹节点上的768次固定观测/原噪声10-flow动作预测定位Q相关功能路径；all初始平移漂移在两节点均删Q后剩.200倍，但直接400删Q距200参考反而1.96倍，未形成修复。正常动作重建通过预登记容差，全部无参数更新、无新rollout，无held梯度/Test/最终controls。

四项GPU诊断均complete/exit0，进程内1564.96/414.79/111.00/90.47秒；原预算内完成，所有句柄退出。首个actual-video辅助SVD的BF16错误修正为FP32代数后重启，失败证据保留，不改变原生attention。训练/视频运行面c63f55dc、执行重放e7447291保持clean pushed冻结版本，canonical科研方法未改。

跨轮解释与候选等级见findings§39–43；原件`runs/analysis/horizon_relation_writer_20260908/causal_learning_20260909/mechanism/{actual_video,reader_credit,execution_projection}/`各summary及注册。当前最有学习证据的候选仍仅去Compiler额外仿射query；地址/内容分离是进一步机制候选，尚未验证完整可学习性或闭环。旧强FM与共享负例继续约束解释，不能宣布唯一根因、完整修复、平台或必须RL。Owner要求直接在对话中交付，不新建报告；正式修改/采纳和未来训练仍停在复核前。

## 2026-09-10 视频控制、查询中介及真实更新深化：动态增量与条件映射保持

Owner重新要求深入原因，明确允许本轮shuffle/reverse/wrong-video诊断及小预算真实更新，禁止再次400步完整训练。全部沿用已训练agentview单视角，clean pushed detached9abc9b95；canonical科研源码与正式方法未改。新视频控制1728行、双层Q中介640行、train8 P/C/D端点256行、有限更新36行、sealed BBQ端点32行、橙汁描述回放9行，共2701条实际配对闭环，51个worker全部exit0。原件总索引`runs/analysis/horizon_relation_writer_20260908/causal_learning_20260909/mechanism/deep_causal_20260910/summary.json`。

视频对照200/400正确为41/51（各96），同task另一视频43/53，同suite错误43/48、跨suite错误46/46、乱序42/47、倒序42/50、重复首/中/末45/52、43/51、43/48。400有局部正确视频增量，却没有正确动态过程的稳定整体优势；静态首帧对完整为R/G/L39/13/12，不能由52≈51推逐例等价。原生R/P4已包含exact language条件，经过video Value并不能保证动态图像是必要信息；纯task记忆或完全忽略视频仍不是唯一解释。完整per-task/suite、得失和局部反例见findings§47及`video_controls/dependence_summary.json`。

四个预注册真实macro从all7完整400临时恢复，在单GPU串行保持四task各0.25权重、64跨episode queries；16条件1024queries，四个预选观察任务均未参与这些更新。另从同初态执行四次零当前梯度AdamW，保留m/v、decay与scheduler。生成274.20秒、峰值26.97GiB、exit0。原400/M均4/4，FULL404为3/4，橙汁task19转向BBQ瓶；P-only复制该目标转移，C+D联合则产生另一种操作失败。九臂冻结轨迹重放逐臂复现，初始实际native输入相同；这是新梯度造成跨任务功能损害的实例，不能全归历史动量，也不当全局遗忘率。无额外训练复跑或完整更新模型保存，原正式checkpoint/optimizer/sampler只读；详见findings§44。

train8的全200/40016→18/32仍R/G/L10/8/6，旧P/C+新D可达22却也丢5。直接沿用BBQ四实例、原teacher的sealed P/C/D八组合全部完成，两个正常端点重现3/4、0/4及正确BBQ→绿色瓶。旧P/C只换新D即使3/4实例转向绿色瓶，只换P为2/4，只换Compiler为0/4；state25则需D与P或C联合变化才转移。新decoder在部分训练任务有用、在held条件有害，与实际跨task更新共同定位共享条件映射的获取/保持耦合，不唯一归Compiler，也不证明冻结某块就是修复。32条完整轨迹和首步相同输入证据保存；详见findings§45–46。

查询中介真实改变Q及Value槽间差异，但normal/LL/HL/LH/HH成功数（各32）为all20016/16/18/15/16、all40018/14/18/14/17、off20016/15/17/14/15、off40020/18/18/19/18。首层作用依赖背景，双层HH四面板均未超过normal；降低强制槽分工作为即时修复的优先级，仍保留Compiler删除的fresh学习净收益及完整学习中介未识别的边界。详见findings§48及`query_mediator/behavior_mediation_summary.json`。

全部新增证据5.6GiB，在strg01核定的总32GiB新增峰值预算内；独立worker复用释放设备、最多六张实际工作GPU，最终双节点无本任务句柄。Q首次子集队列准备的断言错误修正后完成，失败记录保留；无held梯度/Test/正式checkpoint选择或正式方案采纳。双视角学习、Compiler关闭候选的后续增长与保持、每100步strict400及合法language/static prior对照等仅记录为后续正式验证事项。本历史段不恢复执行。

## 2026-09-11 最强Compiler-off macro400冻结视频检查补全

此前九臂只测all，不能概括当前最强候选。Owner要求后，固定validation126/400的`horizon_causal_local_h_read_seed7_20260909` macro400补完整216 LoRA/864配对rows；agentview、train24、states32–35、teacher46/other47与all400完全同口径。正确/另一正确/同suite错误/跨suite错误/乱序/倒序/首帧/中帧/末帧=53/60/59/59/55/52/58/53/53（各96），对应all51/53/48/46/47/50/52/51/48。完整all→off保留43/新增10/丢8，改善不限于正确过程；当前最强候选在该面板亦未形成稳定动态优势。输入仍导致具体成功得失，不能外推完全忽略视频或纯task记忆。

18个worker complete/exit0、216 jobs完整，实际视频输入及env/policy RNG配对通过；无更新、Test、正式选点或方法采纳。生成用原45e16633，执行用9abc9b95；数据量791MiB，未超过登记3GiB。原件`runs/analysis/horizon_relation_writer_20260908/causal_learning_20260909/mechanism/deep_causal_20260910/best_model_video_controls/{registration.json,step400/closedloop/summary.json,comparison_with_all400.json}`；解释与边界见findings§49，当前授权只看progress。

## 2026-09-11 普通FM语义条件化消费C首轮负结果

Owner授权恢复自主科研后，登记video_consumption_writer_design并实现R/C/S与双K1条件分配，formal运行面clean pushed detached7fedbe85。C fresh100/200分别25,600/51,200queries和800/1,600条件，strict400=43→50，四suite0/37/6/0→3/29/16/2；相邻R/G/L27/23/16、churn39、J=.40909，breadth4→6。两组各6worker完整exit0、实际teacher及执行配对通过，约29.5/29.6分钟。相对source47分别R/G/L9/34/38、18/32/29。当前组合弱且改善有限，不追加C训练；不从该负结果推翻整个过程函数类。R/S与train96尚待完成。

原件runs/outputs/video_consumption_c_seed7_20260911/{checkpoints,materialized,evaluation}，分析runs/analysis/video_consumption_20260911/c/step{100,200}/，设计/实现验证和资源记录在同一analysis根；跨轮解释见findings§51，当前状态看progress。

同轮R的strict400两点为37→83，四suite1/26/10/0→0/41/37/5，R/G/L23/60/14、churn74、J=.23711。R200比同51200queries旧off200108少25，保留63/新增20/丢45；比C20050多33。训练任务诊断旧off/R/C=40/46/49（各96），R/C均breadth18；旧→R的R/G/L33/13/7，旧→C34/15/6。R仍获取但双视频条件未在当前节点带来迁移净收益；C也有训练任务能力，不能把低validation等同于完全学不动。所有R/C面板完整exit0，实际配对通过；S仍在执行，本段不代替最终三臂裁决。

首轮最终S=59→45/400，四suite2/37/16/4→0/21/24/0、breadth7→3，相邻R/G/L32/13/27、J=.44444；train96=52，四suite22/10/14/6。S100优势没有保持，C/S均不追加。R/C/S最终83/50/45低于同查询预算旧off108，训练任务46/49/52却高于旧40；新增条件分配及语义消费未改善未见任务迁移。六组strict400及三组train96共2688rows全部exit0，训练合计153600queries，原件总入口runs/analysis/video_consumption_20260911/first_round_summary.json。该负结果不证明监督平台、全部有序视频不可行或强旧候选已穷尽。


## 2026-09-11 off7受控续训：训练任务提升与未见任务下降

按video_consumption_writer_design§7，从冻结45e16633探索性off7 macro400完整保留Writer/Meta、AdamW/scheduler、sampler和三rank RNG，显式迁移gpu01 1/5/6→gpu02 0/1/2，微批8/8/8与逻辑4×64不变；原checkpoint文件不重写，父run不改，新root horizon_off7_continuation_20260911记录lineage并保持探索性标记。原生合同/400恢复及401实际采样匹配通过，200新增更新/51200queries完整exit0，5216.95秒，500/600 checkpoint通过检查。

预登记correct500/600各400条完整为73/54，四suite1/51/12/9→1/24/22/7，breadth均6。父400为126；400→500 R/G/L60/13/66，500→600为31/23/42，400→600为43/11/83、churn94/J=.31387。两个面板分别1218.65/1205.90秒、9worker/3GPU，均42shards/400rows/所有worker exit0。600 train96=64（20/16/16/12、breadth22），父40056→64保留46/新增18/丢失10，3worker/1GPU822.95秒。独立动作FM400/500/600为.104946/.104097/.104786。

全部896闭环rows及输入/执行配对通过。该轨迹增加训练task获取却明显降低未见任务能力，不支持进一步原样续训，原126及完整证据保留；不是全部FM平台或某个模块唯一根因的证明。没有新wrong/shuffle/reverse、Test或held梯度。原件runs/analysis/video_consumption_20260911/continuation/continuation_summary.json及step500/step600内逐task/suite、相邻、parent400和train96比较。另因R37→83仍获取而登记的R300/400只看当前plan/design，不由本历史段恢复其它任务。

## 2026-09-11 R追加300节点：早期增长未保持

冻结7fedbe85、原gpu02 4/6/world2/micro6/6，从formal macro200原生完整exact-resume。R300经formal inspector通过，累计76800queries/2400条件，新增100updates均32.28秒；物化400条件sealed/exit0、615.33秒。correct strict400=63（四suite0/24/34/5、breadth4），9worker/42shards全部exit0、1206.51秒。200的83→300的63保留43/新增20/丢失40、churn60/J=.41748；Long5→5却全部更替。相对旧off400126为51/12/75、churn87/J=.36957，只作不同曝光的能力参照。实际teacher、source/normalizer、执行RNG的canonical配对通过；未触碰Test或新增因果controls。该节点未保持双K1条件的早期增长，后续按已登记400收尾。原件runs/analysis/video_consumption_20260911/r/step300/{completed_summary.json,r200_vs_300.json,old400_vs_r300.json}，解释见findings§53，动态授权只看当前plan/progress。

R400终点与train96随后完整结束：correct85，四suite1/45/32/7、breadth6；R300→400 R/G/L50/35/13、churn48/J=.51020；R200→400为53/32/30、churn62/J=.46087；同102400queries旧off400126→R85为69/16/57、churn73/J=.48592。train96=56（17/17/14/8、breadth20），R20046→56为36/20/10，旧off40056→R56为43/13/13。独立动作FM=.105399/3072queries/无梯度/289.10秒。200新增更新6422.31秒，400验证15worker/60shards798.74秒、train3worker/36shards888.14秒，496条件物化747.14秒；新增896闭环rows和全部配对通过。结束该双K1条件配方，未增加500节点；原件r/continuation/continuation_summary.json及r/step400/（均位于runs/analysis/video_consumption_20260911）。更多教学条件未提高终点训练能力或迁移，不将这个负结果外推为所有独立meta-task扩展无效。


## 55. 2026-09-11：完整形成链复核后，C冻结机制面板完成

Owner要求先还原科学动机、数学与专家形成链，再定位并持续修正正确视频失效，补充correct必须高于source。形成链及旧6000条rows复核记录见[机制复核](video_mechanism_reassessment.md)。随后C100/200六臂、匹配S及固定八task路径替换1472新闭环和14336配对动作预测完成，无训练或held梯度。C200 correct42/96、两错46/46、乱序45、静态49、S51、source15；固定正确S时correct/静态过程同为15/32。C在训练任务上获得source增量，却未使正确过程产生净收益；两个节点不证明收敛。

具体表示缺陷为重复画面仍产生较强中心化P4。依据该性质与功能/闭环干预，准备局部真实历史GRU减同gap/窗口静态参照GRU，只改过程更新、保留正样本FM，尚无新学习结果。详细边界、原件和相邻集合见findings§54；候选设计与后续授权只能从当时progress确认。本历史段不恢复运行。


## 56. 2026-09-11：无变化参照200步首段与冻结行为

冻结64eba75b的局部无变化参照候选完成fresh200更新/51200queries（3327.11秒）；100、200完整checkpoint与六卡恢复状态保留。唯一主要改动是C局部过程更新的同内容参照，未新增wrong训练、辅助loss、RL或task覆盖。24项固定留出动作FM均改善，平均.151451→.109323。

固定train100/200正确视频36/45，各96；200正确/换视频/同suite错/跨suite错/乱序/静态45/42/41/37/41/40。所有视频差额的逐task bootstrap95%区间含0。200 validation34/400低于配对source47及旧C20050；四suite4/24/5/1，breadth5。证据支持学习与初步训练条件分化，未证明可信视频因果优势或source以上迁移能力；首段观察不能作为收敛结论。详见findings§55。

证据根`runs/analysis/video_change_reference_20260911/`中的`launch_contract.json`、`first_segment_action_diagnostic.json`、`step200/paired_summary.json`与`train_mechanism/step100`、`step200`；formal输出`runs/outputs/video_change_reference_seed7_20260911/`。本条只记已完成首段事实，不把开发视频面板当最终资格。

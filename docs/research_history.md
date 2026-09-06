# EMBER research history

本文把可复核的2026年7月至9月研究整理为三层：先读本页的阶段结论；遇到具体接口问题再读对应细节；只有摘要无法裁决时才进入
immutable Git原件和formal artifacts。历史中的资格、假设和“下一步”都属于当时时点，不恢复执行。当前状态见
[progress.md](../progress.md)，新候选见 [设计记录](layered_relation_video_writer_design.md)。

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
| 9月6—7日重新推导 | 单probe、分层局部帧对关系、集合编译、坐标生成；9月7日明确双向读取与对应模式消费 | 新图尚无实现和性能证据 | 按最新数学定义获得真实证据，进入实现前不必重复完整专家复审 |

<a id="baseline"></a>
## 1. 基线、口径和不能混用的数字

| 证据 | 结果 | 适用范围 |
|---|---:|---|
| 历史frozen source | 48/400 | 对应历史validation8配对；另一teacher schedule下曾47/400，不混为同一组rows |
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

## 16. 末读出target/rank共享约束的受控检验（2026-09-07，执行中）

- 接续§15保留坐标初始化的局部Long正证据。只将全局[p]末读出改为[target,rank,p]，新增77,696参数；其它图、std1坐标、
  identity、随机初始化抽样、种子/数据/优化器/96步曝光与world3保持。不是rank/scale/LR扫描，也不据局部Jacobian宣称唯一根因。
- 实现前registration在runs/analysis/layered_relation_writer_20260907/target_rank_readout_control/registration.json；
  canonical实现6ae406ea，182 CPU tests/17.08s通过，包含单输出target/rank更新隔离。
- 新真实GPU机制检查完成exit0：identity后Meta B、随后A梯度可达，source无梯度；full/staged loss同0.1256087869，
  Meta0qB/rho0/decoderB的余弦0.999993/0.999998/1.0。这不构成科学行为结论。
- formal fresh run layered_relation_short4_target_rank_6ae406ea_gpu01p235_20260907从clean pushed detached6ae406ea启动，
  gpu01physical2/3/5，完整合同在该analysis目录launch.json。结果待16/48/96双视频K1及96K4固定配对40裁决；未用validation/Test或负视频controls。

# Horizon K1 现状与可能原因：已有证据审计

日期：2026-09-09。范围截止原 fresh K1 macro600。Owner要求暂停实验，本轮只读取已有源码、Git、日志、checkpoint与物化LoRA，并在CPU重算既有数据；没有训练、模型forward、物化新条件、闭环评测、干预或新增样本。

## 1. 判断先行

**尚不能认定“训练方法出了问题”，也不能证明某个架构模块就是唯一根因。当前最有支持的判断是：系统已经能为不少训练任务生成可用的策略调整，但它学到的条件映射没有形成广泛、稳定的跨任务迁移。我的首要怀疑是现有条件表示与参数生成的归纳偏置，在当前正样本FM监督下偏向训练任务内足够用的语义/外观关联，而没有可靠学成设计所期待的可迁移操作过程。** 这是有结构与行为依据的工作解释，尚未由当前视频因果对照或中间接口行为证据唯一识别。

这个判断包含三个不同强度的层次：

- **事实很强：** validation在200步达到110/400，后续300/400/500/600为86/87/70/82；600有81/82成功集中在source原本全部成功所属的三个cream-cheese相关任务。训练任务held-video面板则52→59→67/96。两类任务上的行为走势不同。
- **解释较受支持：** 后期学习更像继续适配训练任务分布，同时没有保留一部分早期出现的未见任务能力；它不符合“整个系统都没学会”或“后期全部训练能力崩溃”的简单解释。训练侧仍有局部能力缺口，不能被平均分掩盖。
- **具体机制仍待区分：** 新的语言直达残差、任务语义压缩、过程读出、输出参数共享方式、观察侧适配范围，与条件/数据/优化配方都可能参与。现有证据不足以在这些因素中单独定责。

**旧强模型构成实质反证。** v5.2 old和v6 old也用每步4个任务、单视频、多action queries、纯正样本FM，达到132/121。因此“4-task更新本身不行”“纯FM本身只能低分”“24个训练任务必然不够”都不成立。旧v5.2换task-complete配方后最好点反而降到120；v6则升到143。历史更支持架构与配方耦合，不能机械地把全任务覆盖当修复。[E03–E05]

**本次没有找到足以解释当前分数的确定性工程缺陷。** 采样/恢复、权重/分块、缓存/Meta梯度、物化/执行、视频schedule均有对应检查和原始证据；这提高了科学机制解释的优先级，但不等于所有代码路径已被穷尽证明。[E01、E10]

此前我优先实施24-task受控分叉，给训练组织假设的权重过高，未充分使用旧四任务强结果及同架构配方反向效果。该分叉未运行、保持暂停；已有草稿代码和测试不构成任何科学证据，也不是本报告的既定下一步。

## 2. 已执行对象、比较边界与证据覆盖

| 对象 | 本报告采用的权威记录 |
|---|---|
| 当前训练 | `runs/outputs/horizon_k1_supervised_v1_seed7_20260908`；原始run_contract；冻结训练提交`b6d70d98` |
| 当前物化/评测 | 冻结提交`f1330697`；修复后的canonical无放回schedule；100–600六个correct400 |
| 训练侧闭环 | 同一train24×states32–35、held teacher46–49各一次的200/400/600三组96行；同状态source15/96 |
| 当前动作拟合 | 原训练600行metrics、2400行condition exposures；固定held-action 0/200/400，尚无同口径600 held-FM |
| 旧强模型 | v5.2 old、v6 old、两者task-complete的原合同、完整曲线、已存配对分析；44个历史400行面板重新计数，共17,600行 |
| 机制历史 | G1/G2原件、同图完整输出对照、旧共享/clone和meta73/target18边界；四份9/8专家原文及Owner覆盖 |
| 本轮CPU分析 | 现有逐行结果/曝光重算；少数模块组的checkpoint漂移与Meta Adam统计；既有LoRA的观测性几何统计（见E12） |

当前活动树有未提交的24-task sampler/config/tests草稿，**不能用它描述已经产生82/400的训练**。本报告源码审计读取冻结运行面。旧mixed-K64的99/95来自重复teacher抽样，不能当作合法canonical成绩，也不用于把K1与mixed-K作因果比较。[E01、E10]

当前train96与validation400属于不同任务集合、状态数量和视频分布。不能把69.8%与20.5%直接相减叫“纯泛化损失”，也不能将训练侧breadth20/24理解为20个任务已经解决。历史与当前复用同一source checkpoint及源normalization，但跨代teacher日程并未完整配平，backend亦有变化；旧成绩在这里是能力参照和历史反证，不冒充当前严格paired反事实。

## 3. 现状：能力具体在哪里获取、在哪里丢失

### 3.1 validation的主要问题是能力收窄和保持失败

每任务50次，下表均为同一canonical task/state/video/RNG映射上的single-checkpoint结果。[E01]

| global ID与任务简述 | source | 100 | 200 | 300 | 400 | 500 | 600 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1：ramekin旁黑碗→盘 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3：cookie box上黑碗→盘 | 0 | 0 | 1 | 0 | 0 | 1 | 1 |
| 11：cream cheese→篮 | 5 | 21 | 37 | 36 | 37 | 32 | 32 |
| 13：BBQ sauce→篮 | 0 | 1 | 24 | 1 | 3 | 0 | 0 |
| 23：开上抽屉并放碗 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| 26：cream cheese→碗 | 41 | 25 | 41 | 45 | 42 | 34 | 46 |
| 31：cream cheese与butter→篮 | 1 | 4 | 4 | 2 | 5 | 3 | 3 |
| 32：开炉并放moka pot | 0 | 2 | 3 | 2 | 0 | 0 | 0 |
| **合计** | **47** | **55** | **110** | **86** | **87** | **70** | **82** |
| 有成功任务数 | 3 | 6 | 6 | 5 | 4 | 4 | 4 |

600的suite S/O/G/L为1/32/46/3。78次成功来自11与26两个任务；加31后为81/82。200在这三个任务有82次成功，600为81；**三个任务之外从28降到1**。这比“总分从110降到82”更明确：后期没有扩大source薄弱区域，早期出现的其它能力基本未保留。不能据此证明source知识是唯一原因，或模型仅复制source；Object11相对source的5→32仍是实际新增能力。

| 相邻/跨点比较 | 保留 | 新增 | 丢失 | churn/400 | success-set Jaccard |
|---|---:|---:|---:|---:|---:|
| 100→200 | 42 | 68 | 13 | 81 | .341 |
| 200→300 | 71 | 15 | 39 | 54 | .568 |
| 300→400 | 70 | 17 | 16 | 33 | .680 |
| 400→500 | 62 | 8 | 25 | 33 | .653 |
| 500→600 | 56 | 26 | 14 | 40 | .583 |
| 200→600 | 63 | 19 | 47 | 66 | .488 |

200→300时，BBQ sauce原24个成功全部丢失，只新增1；500→600净增12全部由Goal26贡献。故“相邻总分相近”不等于稳定，“600回升”也不等于广泛能力恢复。当前仍远未满足绝对分数、Spatial/Long、breadth、相邻稳定和视频因果目标。

历史train24 rank128 SFT相邻为109/107，当前600的82仍明显低于这一能力参照。SFT的rank与旧backend不同，不是同架构因果对照；但这些差别不能把当前严重能力缺口降格为“比source好就基本成功”。[E01]

### 3.2 训练侧有真实学习，也有未解决的局部缺口

| held-video train96 | source | 200 | 400 | 600 |
|---|---:|---:|---:|---:|
| Spatial /24 | 6 | 12 | 17 | 20 |
| Object /24 | 0 | 15 | 18 | 17 |
| Goal /24 | 7 | 16 | 17 | 18 |
| Long /24 | 2 | 9 | 7 | 12 |
| 合计 /96 | 15 | 52 | 59 | 67 |
| 有成功任务数 /24 | 7 | 20 | 21 | 20 |
| 本面板4/4的任务数 | — | 4 | 6 | 12 |

200→600保留44、新增23、丢8，churn31/96、J=.587；400→600保留51、新增16、丢8，churn24/96、J=.680。67/96和总体增长是此固定小面板上的事实，不是所有训练任务/初始化的性能证书。[E02、E12]

600有四个零成功任务：global9（cabinet上黑碗）、15（tomato sauce）、38（两把moka pots）、39（放杯入微波炉并关门）。它们分别已曝光115、97、117、95次；相比之下global18只有81次曝光却4/4。**没有观察到“只因没抽到这些任务”的解释。** Long38的action起点覆盖48.6%，低于短任务，保留长序列支持不足的可能；但它的条件次数比平均值高，不能合并成简单的训练量不足。

两个有区分力的现象：

- 训练Spatial0（碗在plate与ramekin之间）、5（碗在ramekin上）到600均4/4；validation1（碗在ramekin旁）始终0/50。这与关系/语义组合迁移弱一致，但状态、场景难度不同，不能据此独自定位语言或视觉模块。
- 训练Long34（两杯分别放两盘）与37（alphabet soup和cream cheese入篮）均4/4，validation31只有3/50，训练38/39又为0。它反对“整个图完全不能做多步任务”，同时保留组合迁移和特定操作获取缺口。

当前没有相同状态下“熟悉训练视频→held视频”的完整配对，也没有当前600同状态换另一条视频的资格结果。因此不能量化纯视频更换代价，不能宣称视频鲁棒性已解决。训练task的未训练视频仍有较广成功，只是使“全局一换视频就崩溃”不足以解释全部差距。

### 3.3 动作拟合持续改善，但它没裁决所需的行为

训练每100步mean FM依次为.12443/.11152/.10509/.10150/.09667/.09520。固定同query/video/noise的held-action 0/200/400为.1514567/.1113526/.1062719；200→400有23/24任务改善。后者是更直接的动作泛化证据，仍只覆盖训练任务分布。[E01、E11、E12]

训练task9、15、38的窗口FM总体也下降，却分别在600为0/4。由此不能直接命名接触、抓取、状态分布漂移等具体原因。当前formal artifacts保留success/steps/RNG及条件身份，未发现这批run留存逐步动作/状态轨迹或rollout视频；仅靠终局行不能知道最先在哪个动作阶段失败。旧其它架构的接触/续接诊断不补足这个缺口。

## 4. 历史强模型究竟排除了什么

### 4.1 “同类训练”确实已经成功过，但完整配方不相同

| 项目 | v5.2 old | v6 old | v6 task-complete | 当前Horizon K1 |
|---|---|---|---|---|
| 训练任务 | 固定train24 | 同左 | 同左 | 同左 |
| 每步任务组织 | 全24随机置换循环中取4，每6步轮遍 | 同左 | 每步全24 | 每suite有放回抽1，共4 |
| 单条件视频数 | 1 | 1 | 1 | 1 |
| queries/条件；每步总数 | 21；84 | 20；80 | 20；480 | 64；256 |
| teacher池/task | 50条，无放回循环 | 同左 | 同左 | 16条，有放回抽取 |
| action episode池/task | 50 | 50 | 50 | 26，与teacher严格不交叠 |
| action batch组成 | 21个不同episode各1个起点 | 20个不同episode各1个起点 | 同左 | 独立抽episode再抽frame，有放回 |
| optimizer目标 | 正样本纯FM，共享Writer，source冻结 | 同左 | 同左 | 同左 |
| LR/scheduler | peak3e-4，warmup100，cosine12000 | 同左 | peak3e-4，warmup17，cosine400至1e-5 | 3e-5，warmup8后恒定 |
| 最好已见correct | 132@900 | 121@500 | 143@400 | 110@200；当前82@600 |
| 该点条件/queries | 3600 / 75,600 | 2000 / 40,000 | 9600 / 192,000 | 800 / 51,200；600为2400 / 153,600 |

旧teacher和query两个采样器各用同一个50-episode池，没有显式排除同episode；每batch至多一条query可能与teacher重合。原日志无action-demo trace，不能报告实际重合率，更不能把这项口径差别直接当成绩来源。旧结果也不能写成与当前严格cross-episode完全相同。[E03–E05]

同架构配方对照有相反方向：旧最好点v5.2为132→120，v6为121→143；匹配每task150次条件时，v5.2为132→51、v6为95→111。这些对照同时改变每次Adam聚合的任务数、Adam更新次数、LR阶段，v5.2还改变21/20 queries，因此不单独隔离task覆盖。它们足以说明不能把task-complete当作无条件更好，也说明只拿旧v6 143论证当前4-task问题不完整。

本轮还修正一处旧分析的过宽caveat：v6 old900与task-complete150按(task,visit)归一后的3600条teacher记录实际完全一致；冻结query采样公式亦一致。不同的是更新分组、flow RNG时间线与LR阶段，不能凭包含step/rank的旧hash不同便断言teacher内容也不同。这提高了历史对照的可解释性，仍没有把优化因素逐一分离。

### 4.2 旧强证据不允许被新弱结果抹掉，也不能被拼成不存在的成功

旧v5.2/v6说明同source、rank16完整LoRA、共享Writer和纯FM可以形成比当前600更强的absolute能力；因此冻结source、rank16、单probe或FM不能被一概判死刑。旧v6 task-complete后续450/500/550/600为131/130/132/126，尚非稳定>145；旧各配方视频controls的强弱也不同，不能把一个配方的absolute与另一个配方的视频必要性拼成同一强checkpoint。[E03–E05]

G1只证明特定task-local自由码/privileged初始化的局部容量；G2采用固定antithetic probe pair、冻结stage0 observer，以及action/progress/temporal等直接监督，证明特定条件下动态可读；它没有证明当前single-probe、四组关系图、仅末端FM已经学出同一机制。旧完整输出短学习有明确耦合收益，但同时改变span/carrier/mobile rank/head，后续共享结果仍弱。旧clone强于共享则未分离共享目标、有效容量与优化。所有这些证据都保留其正结果和适用边界，不能拿来机械复用诊断流程或宣布当前某层已被验证。[E08、E09]

## 5. 架构审计：哪些差异确实存在，为什么值得怀疑

### 5.1 当前图没有少算设计要求，但“有计算”并不证明“学会职责”

实际路径为：真实agentview+exact language→冻结vision/Gemma prefix Z/KV→单固定Gaussian probe、flow time1、仅Action Meta→最终action_out_proj前完整50×1024 hidden→四组过去4帧软对应/H-query/两端Z读取/短GRU→逐帧H-read与单向长程→前三组逐H回写→最后P4→608个target/rank compiler queries→完整A/B。

代码中的跨帧依赖、H-query、两端原生视觉读取、GRU次序、长程回写和末端梯度都存在。可疑之处是：**这些被赋予操作含义的中间量，全部只由末端action FM间接学习，没有现有行为证据证明软对应代表真实过程对应、视觉查询核实了关键状态、GRU处理了冗余，或P4保留了可迁移操作关系。** 更完整的输入和更多关系算子并不自动带来更可用的任务表示。[E06、E07]

单probe hidden是噪声端点的条件响应，不是正确未来动作。旧强v6也用single probe@time1，故单probe本身不是独有缺陷；但当前架构把跨H软对应作为视觉查询的重要入口，对这类响应的可对齐性承担了更多职责。这里是一个较早、尚未被当前证据检验的语义接口，不是已发现的forward错误。

### 5.2 裸语言残差扩大了不依赖视频的可达路径

当前语言先由static token embeddings压到一个256维代码；compiler初始query直接相加`target + rank + query_language(language)`，两层均保留query残差。存在这样一个明确的参数子类：cross-attention输出投影为零时，仍能输出依赖语言的完整LoRA。[E06]

旧v6的compiler主体先从视觉Core读出内容，再由Procedure作调制；routing只进Q/K，Value和残差都是内容。旧系统仍可能依靠多模态token中的语言或静态外观，所以它也没有自动保证动态必要性。**不同点是新图新增了更直接的裸语言到输出残差通路；当前checkpoint实际走了多少这条路，现有证据没有测定。** 这不是task-ID/filename泄漏的指控，输入language本来合法；问题在于它是否让FM更容易满足于训练task内的条件关联。

正样本cross-episode训练的统计结构也允许这种解：对固定训练task，teacher video与action-query episode独立采样，所有视频都对应同一个任务目标。一个只凭该任务语言生成合适task adapter的函数，可以降低同样的FM，而不必区分视频过程。训练目标没有要求它证明视频增量。旧强模型同样有这种目标非唯一性，所以**不能据此宣布FM应被替换或转RL**；值得怀疑的是该目标与当前输入/残差/共享方式的组合。

### 5.3 参数共享和语义组织的变化比“4个任务”更大

| 信息/参数组织 | 旧强v6 | 当前Horizon |
|---|---|---|
| 可训练读取侧 | Text/VL/Action三组rank4 Meta | 仅Action Meta |
| 语言语义 | contextual task-token轴保留 | 主language分支压为单代码；Z中仍有contextual任务信息 |
| 视觉/过程组织 | task-grounded Core静态骨架，与Procedure分开 | 经关系视觉读取后形成统一P4 |
| compiler内容起点 | 从Core的Value读内容，再Procedure调制 | target/rank/裸语言query残差加P4读取 |
| 输出参数共享 | 8个family/side heads跨layer/rank共享 | 每target/rank/side独立native D |
| 参数规模 | 整个Writer含三组Meta约10.775M；heads2.179M | Writer368.676M，另Meta0.627M；D末层329.515M |

新D约占Writer的89.4%；条件代码的最后一层因子基在不同target/rank之间不再硬共享。这样增加了表达自由度和直接接收native factor梯度的能力，也把跨层/槽/任务相似性更多交给上游代码与各自D去学习。它与“训练任务学会、未见组合迁移不足”的现象相容；**不能仅凭参数量大或24个任务就认定实际过拟合，也不能因此恢复旧的受限span。**

旧和新在固定checkpoint上的每个因子输出都处于至多256维的learned native span，因此“新D只有256维span”不是独有缺陷。新D可以随梯度改变这个空间；G1的原生X/Y span边界不能直接套到这里。零初始化D/A0 identity的首步只有输出B学习是预期行为，实际后续上游与Meta已经更新，不是永久关门。[E06、E08、E12]

### 5.4 时间压缩与单向反馈是已接受的取舍，仍未被行为验证

每组H-read把50×256状态读成每帧一个256维token；最后compiler只读P4，不读最终完整U或原始Z。前缀因果合同还意味着很晚出现的证据不能回到早期原始视觉重新提出查询：某帧Z只在其附近四帧关系中被直接读取，更晚只能利用已经保留的压缩内容。最终compiler可以综合全视频P4，但不能保证恢复早期已丢信息。

这是Owner选择的过去单向合同，不是漏实现双向长程；专家原文关于“末尾信息帮助回查早期抓取”的双向例子不能作为当前能力说明。该取舍可能影响复杂组合/Long任务，但训练Long已有成功、Spatial也有严重缺口，所以现有结果不能把主要原因锁到因果mask或Long压缩，更不授权自动恢复双向、18层或追加模块。[E07]

## 6. 训练、数据与工程解释的逐项核查

| 候选解释 | 现有证据 | 本报告判断 |
|---|---|---|
| 任务/episode/query采样错位，resume重置 | 从冻结采样公式重建全部2400条件及153600个query episode/frame，与日志匹配；600步连续 | 未发现此类缺陷，不能用来解释退化 |
| 四任务随机覆盖不均 | 每task81–122条件，最大连续缺席18–53个updates；旧为每6步完整轮遍 | 实际差别，可能增加更新波动；弱task也有较高曝光，旧四任务强结果反对将其定为主因 |
| queries太少/完全没看过数据 | 153600次query，77701个不同(task,episode,frame)起点，占当前107825个允许起点约72.1%；382/384种teacher条件已见 | 已有实质监督；未穷尽数据，也不能按query总数宣称“充分到架构已被否定” |
| 视频/任务条件学习不足 | 当前16视频池，600累计2400条件；旧v5.2强点3600条件/50视频池，v6 old强点仅2000条件 | 条件支持是可能的放大因素，不能单独解释当前弱分或推出全24一定有益 |
| LR/schedule不合适 | 当前warmup8后恒定3e-5，旧peak高10倍且衰减；无隔离对照 | 保留优化与架构耦合解释；没有依据指定升LR、降LR或延长训练哪一个正确 |
| 梯度爆炸/频繁裁剪 | 六个百步窗口总梯度最大值均<.40，clip阈值1，600步无一次触发裁剪 | 没有支持；正常学习中成功集变化不等于数值爆炸 |
| Meta断梯度、实际没更新 | 除identity第1步外2–600非零；checkpoint中A/B实际变化，Adam有效更新见下文 | 排除全局冻结/永久断开；不证明学到了有效视频理解 |
| 缓存旧R造成读取Meta失效 | 缓存只含冻结prefix KV/Z及语言/索引，每步重算R，Writer/observer完整VJP后才step | 未发现此类错误 |
| microbatch改变权重或噪声 | 先生成完整64条随机量再切片，chunk/64均值，条件×.25，跨rank SUM | 未发现重复加权/重采样；正常低位差异仍存在 |
| padding mask回归 | 当前repeat-last且全50×7取均值；旧强Writer处理器也丢弃action_is_pad | 是共有目标语义，不能作为当前独有bug；没有证据已定位其行为代价 |
| 相机/旋转/state/normalization错误 | teacher agentview180°，queries双相机；8维state、7维action；复用source统计 | 未发现串线。demo128→224、rollout render256→224是实际已有数据域差别，并非新相机bug |
| 旧autocast/LoRA dtype bug仍未修 | 修复代码在当前运行面；已有非零LoRA native/physical PEFT证据支持修复 | 旧缺陷不能照搬解释当前；证据不是当前600所有条件全量parity |
| checkpoint错用、第二adapter、平均LoRA | 完整Writer/Meta严格恢复；每condition物化一套38-target/76张量；source基础identity B零 | 未发现这种实现错误 |
| 无放回视频schedule或成功统计错误 | 当前六点各400条件、每task50视频，跨点mapping一致；成功来自LIBERO谓词，不以timeout当成功 | 旧schedule错误已更正，不解释当前200→600配对退化 |
| 冻结source/rank16根本无法支持广泛能力 | 旧相同source的rank16 Writer132/121/143；旧局部容量和SFT亦有更广行为 | 不是当前已证实硬上限；source能力簇仍可能影响学习偏向 |
| source先验被机械复制 | Object11相对source5→32，600有41个source之外的新成功 | 不能说完全没学；新增仍高度集中于source原成功相关任务 |
| 泛化问题全由同task换视频导致 | 训练任务held-video有67/96，但无同状态familiar-vs-held当前配对 | 全局换视频崩溃不足以解释；局部视频代价和当前动态必要性未识别 |
| 单纯训练动作记忆 | 固定held-action与held-video训练任务行为都有改善 | 逐episode记忆不是充分解释；更高层的任务语义特化仍可能 |
| 所有失败都是occupancy/contact | 当前只有终局结果，没有这批rollout的阶段轨迹 | 现有证据不能定位动作阶段；旧其它图的续接诊断不可移作证明 |
| K1不行，应换K或RL | mixed-K64旧分数不合规且配方不同；旧K1模型已有较强行为 | 不支持以此绕开当前缺口；当前也没有RL诊断证据 |

### 6.1 读取Meta的实际变化：不能只看原始梯度大小

只读当前已存200/400/600 checkpoint，按少数模块组统计；没有模型执行，也没有扫描329M D做“逐tensor一致性验证”。Meta A/B在200→400的相对L2变化为2.64%/62.45%，400→600为1.83%/30.28%。600保存的Adam moments对应最后一步自适应更新分量RMS约为A 2.98e-6、B 4.08e-6；epsilon主导坐标占21.1%/8.82%，不支持“全部被eps压成近冻结”。各local、temporal、compiler等被统计模块也有实际变化。[E12]

这些是学习状态证据，不能按不同参数化的相对变化大小排名因果重要性；特别是B从零起步，分母小会放大相对值。它们不能告诉我们R里哪种信息变得有用，也不能证明四组关系已经学会设计职责。

### 6.2 precision检查的适用范围

既有真实非零LoRA parity来自旧joint macro4、task21、64个decision：修复后的native flow对保存physical-PEFT oracle差异为0；batched mean|Δaction|=.000514、max=.044685、KL=.01635。相关实现延续到当前，但这不是fresh K1六个checkpoint全量parity，也不能保证闭环对低位差异完全不敏感。反过来，也没有证据支持把未测量数值误差当当前低分主因；不能通过扩大dtype、强行逐元素一致或重做大量检查制造一个工程解释。[E10]

### 6.3 已存LoRA：没有全条件恒定输出的证据，几何仍不裁决视频必要性

本轮仅读取600已经物化的400套LoRA，按存储数值的有效更新`BA`计算Frobenius内积，利用rank16 Gram积避免形成大矩阵；不用有gauge歧义的单独A/B cosine，也不模拟policy执行时的dtype/kernel。每task预先固定demo0、25两个参照，与该task全部50条视频比较，另比较不同task的这两组参照；没有重新运行Writer或按成功选参照。[E12]

同task有效更新cosine的中位数为.99764–.99994，归一化距离中位数为1.13%–7.93%；不同task参照的cosine中位数为.87259、归一化距离中位数50.78%。个别同task条件距离可达42.0%，所以也不能说每条视频输出严格一样。上述统计是两个固定参照的观测，不是所有视频对的穷举。

这排除了“所有task/视频都编译成完全同一个LoRA”的简单说法，但**同task相近本身不是问题**：好的编译器也应从不同正确视频得到一致任务策略。不同task/视频导致参数变化，不说明这些变化对成功有效，更不区分语言、静态外观、时长/帧索引和动态证据的贡献。不能把几何当视频因果证明，也不能据此选择checkpoint或修改架构。

## 7. 我认为最可能的原因及置信边界

这里排序的是解释优先级，不能把它当成已完成因果实验后的概率估计。

| 排位 | 原因解释 | 为什么目前更可疑 | 反证与不确定性 |
|---|---|---|---|
| **1** | **当前架构与正样本FM耦合，学成的任务条件映射偏向训练内可用关联，跨任务关系/组合迁移不足** | train held总体改善、validation收窄；新旧语义/视觉内容路径与输出共享实质不同；新裸语言残差允许更短的task条件解；所声称过程职责仅末端监督 | 未测当前语言/static/video因果贡献；参数多不直接证明过拟合；旧纯FM也有非唯一解却能更强。对“迁移缺口”把握高，对具体语言捷径或某层定责仅中低 |
| **2** | **后期继续优化训练分布时，早期获得的未见任务能力未被保持；动作FM改善与目标闭环表现脱节** | BBQ sauce从24到0、其它source簇外28→1，held FM/训练侧仍改善；存在训练task本身FM下降却零闭环的局部例子 | 这是支持较强的行为层解释；原因仍可能是表示更新、共享参数耦合、LR/Adam轨迹或状态覆盖，不能单称梯度冲突/灾难遗忘/occupancy |
| **3** | **较窄视频支持、每条件64queries、随机任务更新及不同LR，放大了上述表示学习与保持问题** | 当前条件/视频池与旧强模型不同，随机轮遍确有长缺席间隔；同query数对Writer条件学习不等量 | 旧四任务强模型和v5.2 task-complete反向结果限制此解释；当前高曝光task仍可能零成功。训练组织是候选贡献因素，不是已定位主因 |
| **4** | **特定训练任务/多阶段操作仍存在监督到闭环的局部缺口** | 训练9/15/38/39零成功，Long仍12/24；复杂组合能力并不普遍 | 没有当前动作阶段轨迹，不能定位抓取、接触、释放、关门或长程压缩；也不能用局部失败否定整图容量 |
| 较低 | 确定性工程回归、全局梯度断开、没有训练到这些task、固定source/rank16绝对无能力 | 这些是必须检查的解释 | 本轮针对性核查和历史强结果没有支持它们；保留未覆盖路径的一般不确定性 |

这不是“已经证明架构错、训练无关”。更准确的说法是：**当前最可疑的是学到什么样的条件表示和共享参数映射，以及它为何在继续FM学习时没有保留跨任务能力；现有证据不足以把问题降格为任务batch的单项调整。**

## 8. 现有证据回答不了什么

1. 600实际有多少行为来自语言/静态外观，有多少必须依赖动态视频；正确视频下的成功、参数变化、attention依赖都不回答必要性。
2. 相同当前checkpoint、task、state换视频后的因果差别；现有correct每state对应不同video，没有资格other配对，train96也没有当前familiar对照。
3. R→对应→视觉核实→GRU→P4→compiler→D中哪一个首先丢失了对行为必要的信息。代码依赖证明与旧局部机制证据不能代替当前接口证据。
4. LR、任务轮遍、视频池、query分配和参数共享各自的独立贡献；现有跨代和配方bundle比较不够。
5. 当前失败最先发生在哪个动作阶段，或某种action/state区域是否缺监督。终局success/steps无法恢复未保存的轨迹。
6. 如果从fresh起点使用另一组织是否更好。此前拟从400分叉，即使运行，也只能回答“沿原轨迹学到400后改变组织”的效果，不会直接裁决fresh训练的最优配方，更不自动解释原200→600为何下降。

这些是本报告的可识别边界，不是自动安排的新实验清单。shuffled/reversed仍保持最终冻结后用途；不因分析需要提前生成它们。所有实验继续暂停，等待Owner基于报告讨论。

## 9. 证据索引

下列原件均为既有证据；E12为本轮对既有数据的CPU再分析。源码引用优先指向实际冻结运行面，避免混入暂停的草稿。

- **E01 当前六点与source、曝光、配对：** [round_evidence.json](/data1/user/ymdai/projects/EMBER/runs/analysis/horizon_relation_writer_20260908/k1_fresh/segment400_600/round_evidence.json)、[round_analysis.md](/data1/user/ymdai/projects/EMBER/runs/analysis/horizon_relation_writer_20260908/k1_fresh/segment400_600/round_analysis.md)、[原run_contract](/data1/user/ymdai/projects/EMBER/runs/outputs/horizon_k1_supervised_v1_seed7_20260908/run_contract.json)、[metrics](/data1/user/ymdai/projects/EMBER/runs/outputs/horizon_k1_supervised_v1_seed7_20260908/metrics.jsonl)、[exposures](/data1/user/ymdai/projects/EMBER/runs/outputs/horizon_k1_supervised_v1_seed7_20260908/exposures.jsonl)。round_evidence内逐项给出原始results.json路径。
- **E02 train96与source配对：** [analysis.md](/data1/user/ymdai/projects/EMBER/runs/analysis/horizon_relation_writer_20260908/k1_fresh/train96_diagnostic/analysis.md)、[200/400 comparison](/data1/user/ymdai/projects/EMBER/runs/analysis/horizon_relation_writer_20260908/k1_fresh/train96_diagnostic/comparison.json)、[400/600 comparison](/data1/user/ymdai/projects/EMBER/runs/analysis/horizon_relation_writer_20260908/k1_fresh/train96_diagnostic/comparison400_600.json)、[600 summary](/data1/user/ymdai/projects/EMBER/runs/analysis/horizon_relation_writer_20260908/k1_fresh/train96_diagnostic/step600_completed_summary.json)。
- **E03 旧强四配方完整曲线、匹配曝光与原件引用：** [matched_exposure/analysis.json](/data1/user/ymdai/projects/EMBER/runs/outputs/pi05_as_writer_v52_v6_recipe_matched_exposure_seed7_20260801/analysis.json:998)。[v5.2 old合同](/data1/user/ymdai/projects/EMBER/runs/outputs/pi05_as_writer_v5_2_patch_grounded_dev_r4_seed7_s12000_529da6b_20260728/run_contract.json:112)、[v6 old合同](/data1/user/ymdai/projects/EMBER/runs/outputs/pi05_as_writer_v6_oldrecipe_rankrotating_dev_r4_b20_seed7_s12000_bad9a96_20260729/run_contract.json:124)、[v6 task-complete合同](/data1/user/ymdai/projects/EMBER/runs/outputs/pi05_as_writer_v6_decay400_taskcomplete_dev_r4_b20_seed7_s2400_4efa737_20260729/run_contract.json:118)、[v5.2 task-complete合同](/data1/user/ymdai/projects/EMBER/runs/outputs/pi05_as_writer_v52_taskcomplete_decay400_formal_dev_r4_b20_seed7_60f4508_20260731/run_contract.json:118)。
- **E04 已存同架构/配方交互分析：** [architecture_recipe_interaction/analysis.json](/data1/user/ymdai/projects/EMBER/runs/outputs/pi05_as_writer_v52_v6_architecture_recipe_interaction_paired_seed7_20260801/analysis.json:1251)。旧源码通过Git读取：`4efa737:src/ember/writer/model.py` 97–114、227–235、499–520；`4efa737:src/ember/writer/temporal.py` 438–461、521–535、637–654；`4efa737:src/ember/writer/video_program.py` 290–447；`4efa737:src/ember/pi05_processing.py` 179–185。源文件名称须按该冻结提交解释。
- **E05 已存旧五臂及视频动态边界：** [video_causality_audit/analysis.json](/data1/user/ymdai/projects/EMBER/runs/outputs/pi05_as_writer_v52_v6_recipe_video_causality_audit_seed7_20260802/analysis.json:1258)。只解释历史，不创建或提前使用当前最终controls。
- **E06 当前图实际源码：** [native](/data1/user/ymdai/projects/EMBER/.codex/worktrees/horizon-k1-runtime/src/ember/writer/native.py:86)、[language encoder](/data1/user/ymdai/projects/EMBER/.codex/worktrees/horizon-k1-runtime/src/ember/writer/horizon.py:97)、[compiler query](/data1/user/ymdai/projects/EMBER/.codex/worktrees/horizon-k1-runtime/src/ember/writer/horizon.py:144)、[query residual](/data1/user/ymdai/projects/EMBER/.codex/worktrees/horizon-k1-runtime/src/ember/writer/attention.py:81)、[relation与视觉核实](/data1/user/ymdai/projects/EMBER/.codex/worktrees/horizon-k1-runtime/src/ember/writer/relation.py:62)、[native factor](/data1/user/ymdai/projects/EMBER/.codex/worktrees/horizon-k1-runtime/src/ember/writer/native_factor.py:14)。
- **E07 专家原文与Owner取舍：** [讨论索引](/data1/user/ymdai/projects/EMBER/docs/review_materials/20260908/README.md:23)、[round2 evidence review](/data1/user/ymdai/projects/EMBER/docs/review_materials/20260908/expert_round2_evidence_review.md:119)、[process visual loop](/data1/user/ymdai/projects/EMBER/docs/review_materials/20260908/expert_process_visual_loop.md:112)、[FM/RL protocol](/data1/user/ymdai/projects/EMBER/docs/review_materials/20260908/expert_fm_rl_protocol.md:17)、[past/interleaved](/data1/user/ymdai/projects/EMBER/docs/review_materials/20260908/expert_past_interleaved.md:230)。后续Owner的单向长程、纯FM→独立RL覆盖原文相关建议。
- **E08 G1/G2边界：** [G1 gate](/data1/user/ymdai/projects/EMBER/runs/outputs/pi05_ecp_native_factor_g1_action_in_groups_held5_formal_31f0053_gpu02p1_20260825/g1_gate_step0.json:11)、[G2 gate](/data1/user/ymdai/projects/EMBER/runs/outputs/pi05_ecp_natural_program_g2_boundary_fold0_m10_c1493a1_gpu02p0123_r4_20260825/gates/macro_00000020.json:17)、[G2实际监督合同](/data1/user/ymdai/projects/EMBER/runs/outputs/pi05_ecp_natural_program_g2_boundary_fold0_m10_c1493a1_gpu02p0123_r4_20260825/run_contract.json:103)。
- **E09 同图输出与共享历史：** [complete-output comparison](/data1/user/ymdai/projects/EMBER/runs/analysis/pi05_ecp_prw_complete_shared4_20260906/comparison.md)、[历史meta73/target18/clone适用边界](/data1/user/ymdai/projects/EMBER/runs/analysis/horizon_relation_writer_20260908/k1_fresh/train96_diagnostic/history_boundaries.md)。
- **E10 工程合同与已有真实验证：** [训练与恢复](/data1/user/ymdai/projects/EMBER/.codex/worktrees/horizon-k1-runtime/src/ember/writer/training.py:222)、[FM分块](/data1/user/ymdai/projects/EMBER/.codex/worktrees/horizon-k1-runtime/src/ember/writer/functional.py:599)、[VJP重放](/data1/user/ymdai/projects/EMBER/.codex/worktrees/horizon-k1-runtime/src/ember/writer/supervised.py:27)、[cache](/data1/user/ymdai/projects/EMBER/.codex/worktrees/horizon-k1-runtime/src/ember/writer/runtime.py:73)、[数据读取](/data1/user/ymdai/projects/EMBER/.codex/worktrees/horizon-k1-runtime/src/ember/writer/data.py:319)、[processor](/data1/user/ymdai/projects/EMBER/.codex/worktrees/horizon-k1-runtime/src/ember/pi05_processing.py:141)、[旧非零LoRA真实parity](/data1/user/ymdai/projects/EMBER/runs/analysis/horizon_relation_writer_20260908/joint_profile/native_fixed/summary.json)、[600 bank coverage](/data1/user/ymdai/projects/EMBER/runs/analysis/horizon_relation_writer_20260908/k1_fresh/segment400_600/step600/final_manifest_coverage.json)、[evaluation](/data1/user/ymdai/projects/EMBER/.codex/worktrees/horizon-schedule-runtime/src/ember/pi05_evaluation.py:227)。
- **E11 当前固定held-action：** [comparison_0_200_400.json](/data1/user/ymdai/projects/EMBER/runs/analysis/horizon_relation_writer_20260908/k1_fresh/segment200_400/step400/held_action/comparison_0_200_400.json)。
- **E12 本轮CPU既有数据再分析：** [quantitative_evidence.json](/data1/user/ymdai/projects/EMBER/runs/analysis/horizon_relation_writer_20260908/k1_fresh/evidence_review_20260909/quantitative_evidence.json)、[checkpoint_module_statistics.json](/data1/user/ymdai/projects/EMBER/runs/analysis/horizon_relation_writer_20260908/k1_fresh/evidence_review_20260909/checkpoint_module_statistics.json)、[existing_lora_geometry.json](/data1/user/ymdai/projects/EMBER/runs/analysis/horizon_relation_writer_20260908/k1_fresh/evidence_review_20260909/existing_lora_geometry.json)。这些统计不新增模型运行或闭环样本，不作为正式选点；方法与限制写在各JSON内。

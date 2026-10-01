# 公共基座与T架构审阅：证据及解释边界

本页区分仓库中可直接复算的结果、由本地大原件读回后保存的小型统计，以及尚未确诊的解释。
2026-10-01更新公共/完整对照与远程入口，没有新增实验。模型与讨论状态先读
[README](README.md)及[最新讨论整合](PUBLIC_BASE_REVIEW.md)。下方原始学习曲线和历史诊断保留其各自日期与范围。

## 本次最相关的公共与完整结果

所有分数均来自既有完成记录；不能跨行相减、跨学习年龄归因或把已见任务面板当held。

| 比较 | 同一口径结果 | 可远程核对的字段/原行 | 解释边界 |
| --- | --- | --- | --- |
| 原T2340公共/完整 | Validation400：公共103、完整161；full对public保留79/新增82/丢失24 | [selected summary](selected_condition_summary.json)的arms.public_beta、arms.correct、pairs；[逐行CSV](selected_condition_rows.csv) | 同一冻结checkpoint删除M，不是独立训练的public模型；差58不能全部称为新操作知识 |
| 成熟T追加public FM | Validation400：aux1890为148、control1890为158；保留136/新增12/丢失22 | [validation summary](validation_summary.json)的arms.public_aux1890、arms.control1890、pairs；[CSV](validation_rows.csv) | 同一1800父点各续90步，继承optimizer；只凭该结果不能否定fresh共同学习 |
| fresh Joint完整能力 | Validation400：Joint450为114、原T450为122、MT300为153；Joint对T保留88/新增26/丢失34 | [Joint evidence](../../analyses/operator_joint_public_evidence_20261001.json)的closed_loop.joint、closed_loop.T450、comparisons.full_joint_vs_T450 | 双目标从初始化开始，不能仅以晚加解释；分数不能直接诊断梯度冲突 |
| fresh Joint公共能力 | 已见144：Joint80、T68；保留60/新增20/丢失8 | 同一[Joint evidence](../../analyses/operator_joint_public_evidence_20261001.json)的closed_loop.joint_public、closed_loop.T450_public、comparisons.public_joint_vs_T450 | train24＋support12，每task4个固定初态；没有Joint held-public，不能计算114−80 |
| Context900公共/完整 | Validation400：公共134、完整151、MT153；full对public保留112/新增39/丢失22 | [Context public evidence](../../analyses/operator_context900_public_evidence_20261001.json)的summaries.public900、summaries.full900、comparisons.full900_vs_public900、method | 同一checkpoint消融；Long为13/14/24，不支持关M即可修复。相对T的架构与年龄改变，不能归因于public FM |
| self_read450 | Validation400完整116，对Context450126/T450122/MT153；已见144公共83 | [self_read evidence](../../analyses/operator_self_read_evidence_20261001.json)的closed_loop.summaries、closed_loop.official_comparisons、closed_loop.public_comparisons、decision.scope_limit | 公共83=目标train24的38/96＋support12的45/48；无held-public、M0闭环或相邻稳定性，116与83不可相减 |

原T的same-task-other为150/400，仍见[selected summary](selected_condition_summary.json)。
这支持有效条件作用与视频依赖，但不能把所有条件收益认定为新知识，或把所有条件损失归给共同适配。
fresh Joint的held差额也有任务/seed不确定性；本材料将其表述为未观察到完整提升，而非已证明普遍有害。

关于裸Source：Owner指出T已有公共基础能力，但本次当前T的strict配对导出中未找到相应裸Source臂。
历史coverage协议的Source51/400及其它旧数值不能移入当前strict MT153这张比较表；
本材料不补一个无法按当前配对口径独立复算的Source数字，也不把“公共103”直接等同相对Source提升103。

以上JSON/CSV是Git跟踪的轻量证据，可核分数、task/suite分布和success-set得失；
其中sources/source_paths是原件provenance，不表示远程专家能打开服务器文件。
没有weights、H/X/cotangent、完整RGB或continuous轨迹时，不能声称重新核验其全部内容或重跑模型。
路径/seed字段本身也不独立证明全部scene内容匹配。Test与shuffled/reversed不进入本次方法论据。

补充机制统计可读[整体综合](../../analyses/t_public_basis_synthesis_evidence_20261001.json)，
以及[Context900已见面板](../../analyses/operator_context900_seen_evidence_20261001.json)。这些是派生证据，不是新候选的训练结果。

## 完整能力曲线

[validation_rows.csv](validation_rows.csv) 导出29个已完成评测臂、每臂400行的成败与必要标识，共11600行。
[per_task.csv](per_task.csv)给出逐任务结果及相对 MT 的 R/G/L/N；[validation_summary.json](validation_summary.json)
保存来源、suite/breadth、相邻得失及指定参照比较。R=双方成功，G=候选新增，L=候选丢失，N=双方失败。
这些都是同一固定 Validation8 的单 checkpoint 结果，不是 checkpoint union；不同更新点不是独立训练 seed。

| 更新点 | 原 T | U | change_clock |
| ---: | ---: | ---: | ---: |
| 270 | 116 | 128 | 124 |
| 450 | 122 | 143 | 123 |
| 810 | 144 | 132 | — |
| 900 | 148 | 132 | — |
| 1080 | 147 | 128 | — |
| 1350 | 154 | — | — |
| 1710 | 154 | — | — |
| 1800 | 154 | — | — |
| 1890 | 158 | — | — |
| 1980 | 156 | — | — |
| 2070 | 152 | — | — |
| 2160 | 155 | — | — |
| 2250 | 156 | — | — |
| 2340 | 161 | — | — |
| 2430 | 147 | — | — |
| 2520 | 154 | — | — |
| 2610 | 160 | — | — |
| 2700 | 154 | — | — |
| 2790 | 159 | — | — |

每格分母400；强 MT300 为153。T1890 是公共风险小试的原目标 control 分支，随后是 T1980 等的实际父，
不是 public_aux 分支。T2340 为固定选择点，不能把后续节点抹去后描述成没有观察成熟相邻表现。
U 在早期高于 T，后期排名反转；这一事实不支持“绑定 A 从一开始必然更好”。
T2340 比 MT 多8例，T2790 为159；这些小差额、任务数与单训练轨迹不支持已建立稳健绝对优势的强结论。

主要分布可直接从 CSV 复算：

| 模型 | task3 | 6 | 11 | 16 | 23 | 26 | 31 | 39 | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MT300 | 41 | 7 | 36 | 10 | 0 | 35 | 24 | 0 | 153 |
| T2340 | 45 | 6 | 45 | 5 | 0 | 36 | 24 | 0 | 161 |
| T450 | 24 | 2 | 42 | 1 | 1 | 38 | 14 | 0 | 122 |
| change_clock450 | 17 | 3 | 46 | 0 | 0 | 37 | 20 | 0 | 123 |

机制分析 §47–55/75 记录行为和学习阶段。能力交换不可只看总分：clock270→450 保留100、得23、失24，
churn47；相对同节点 T450 保留94、得29、失28。Object 正例保留，Spatial 目标获取和 Long 第二目标仍有交换。
原 T 和 MT 都未完成的任务不能自动归为 Writer 独有失败。

## 选定 T 的视频条件确有作用

固定 T2340 后的三种视频条件与公共策略结果为：correct161、same-task-other150、cross-suite-wrong65、public β103。
[selected_condition_rows.csv](selected_condition_rows.csv)和 [summary](selected_condition_summary.json)只导出这四臂。
它们不用于重新选择 checkpoint，也不能将 wrong 的全部差额都算成 correct 的收益：correct−β=+58，wrong−β=−38。

同task两个正确视频共同带来 β之外的64个成功，也共同丢失14个 β成功。task31 的24/19个成功只有9个重合，
所以同task换视频总分接近不等于执行作用相同。来源为机制 §63/findings §239；具体正反个例须按其适用范围解释。
这些证据支持当前视频条件具有真实功能依赖，未证明任意新任务的操作关系均可被正确编译。

Test 和 shuffled/reversed 未导入本材料，也不应从历史中抽取后作为本轮改进依据。

## 哪些解释已经受到约束

下表是已完成证据，不是对候选架构的普遍否决。Frozen 干预测得的当前依赖，与 fresh 共同学习后的可能性分开。

| 问题 | 已有观察 | 能支持和不能支持的判断 | 仓库原论证 |
| --- | --- | --- | --- |
| 共享 A 是否保证对齐 | T/U早期与后期排名反转；教学与执行的 state、noise/time、条件参数不同 | 共享坐标改变学习约束，未保证语义匹配或性能 | 机制 §42–44、§48–50 |
| 视频条件 M 是否整体多余 | T1800 full154，删除 M 后自身 β100；full 得79、失25 | M承担真实能力，同时有局部损害；不是独立训练的两模型消融 | 机制 §56；本目录 beta1800 原行 |
| 补强公共策略是否直接修复 | 同父90步 control158、public_aux148；固定训练公共/完整FM均略改善 | 公共风险被改变但没有匹配闭环收益；不证明 B₀应删或必须保留 | 机制 §56.2–56.5 |
| 让最终条件参数参与重新读取 | 固定 T2340 的 joint/X/H/other 四种回读，匹配生成参照后整体无一致收益 | 未支持冻结替换的直接修复；不能否定两遍图重新共同学习 | 机制 §64–65；设计 §27 |
| 简单删除 Q/V/IO 条件写入 | 完整十步生成中无整体直接收益，V有局部task0前缀正例 | 不采纳全局删减；不把共适配被破坏当新架构学习失败 | 机制 §66–67；设计 §28 |
| B₀教学与执行路径严重互相抵消 | 四task/八条件，A50 native间接/直接梯度范数比约.061–.175，多数近正交 | 没有普遍严重抵消证据；不证明不存在关键局部牵制 | 机制 §71.1；learning_credit JSON |
| 训练面板主要误差来自换teacher方差 | 同task两teacher前5动作MSE的离散项约.14%–.38% | 这两个teacher主要共同犯错；不能外推成全部teacher或闭环不敏感 | 机制 §71.2；同上 |
| 简单末层读出能把回读特征变成修复 | 零初始线性残差拟合 A 改善，跨episode/留task不迁移；直接B闭式控制也失败 | 该小拟合没有分离“自由修正可迁移而Writer不行”；不能据此给完整Writer判死刑 | 机制 §71.3–71.4 |
| 输出秩或子空间是否构成主要硬阻断 | Q8计入公共 B₀/共享O/任意条件δZ的宽松局部切空间覆盖约98.45%的B风险余切能量 | 固定O确有限制，但不能忽略其它学习自由度；不是实际优化可达率或整体能力比例 | 机制 §64.2 |
| 宽谱地址是否让有用方向写不进去 | 真实十步 Aq/独立B余切未普遍落入 key-transport 弱方向 | 不支持只凭条件数选择白化、扩大空间或重新生成A | 机制 §72；learning_credit JSON |
| 逐帧覆盖是否是主因 | change_clock修正静态覆盖/路径细分性质，270信用分布确变，闭环124→123 | 局部机制变化成立；本窗口未兑现主要能力修复。450不是更晚学习上限证明 | 机制 §69、§73–75 |

上表中的 [learning_credit JSON](../../analyses/operator_learning_credit_evidence_20260930.json)、
[clock JSON](../../analyses/operator_clock_evidence_20260930.json)、
[270 evidence](../../analyses/operator_change_clock_270_evidence_20260930.json)、
[450 evidence](../../analyses/operator_change_clock_450_evidence_20260930.json) 均为 Git 中的小型证据。
数值来自已保存原件，不意味着原始大张量也在仓库中。原文集中在
[机制分析](../../analyses/feature_to_operator_mechanism_20260926.md)及 [findings](../../../findings.md)。

## 最新 CPU 几何分析的有限含义

[learning_geometry.json](learning_geometry.json)补入本轮只读已有 T2340 原件的派生结果，零新增模型/GPU/环境或参数更新。
它不是当前 S/P/D 的结果。全部来自四个 train tasks、每task两teacher，以及 Q8/V8/action_out 三处；
原始 H/X/余切不在远程仓库，专家应将数字视作有来源的读回统计，而非自己已重新计算的事实。

固定 β/A/native 时可精确写出 `M_i=O Z_i`。只训练 O、采用普通 SGD 时：

```text
δO = −η Σ_i w_i G_Ai Z_iᵀ
δM_j = −η Σ_i w_i G_Ai Z_iᵀ Z_j
```

这揭示当前 Value 坐标如何缩放、耦合不同条件的写入，但不代表同时训练 P/C/D 的完整 Jacobian，更不等于实际 Adam 更新。
实际 Z 的谱有效秩中位数约 Q8 2.47、V8 2.94、out 3.24；key-only transport 相应为12.68、22.44、29.19。
低于最大特征值1e−3的方向承载的 B风险余切能量中位数约10.8%、22.3%、30.7%，并非全部为零；
但24个site/condition的梯度 Rayleigh quotient 均高于各自等方差平均值，没有显示普遍的有用方向不可写。
有效秩不是代数秩，谱集中也不自动证明语义塌缩或白化必要。

为检查 §33固定 B₀是否把有用方向全部锁死，同Q8八条件、允许任意δZ和共享δO的宽松切空间仍覆盖约97.71%的B余切能量。
这不是实际 P/C/D 参数化或有限优化的保证；该统计只降低“诊断已被固定公共输出方向完全卡死”的简单解释。

## 最近似历史不能清零

- LocalField 已有完整双向时序上下文、逐位置 `U r Xᵀ` 和真实功能信用；当前 T 改变的是实际 A 绑定与公共原生读取，
  不是首次使用时序内容或同一消费者。完整比对见机制 §42.1及 [研究历史](../../research_history.md)的 LocalField条目。
- ProcessPullback及共享 L/R 已有差分 Value、递归矩阵、功能监督；不能用“加入差分/记忆”单独论证新收益。
- 条件速度 V151→102、公共/条件对应 P124→108，保留了已有正例及后续能力丢失；存在公共路径、普通FM或共享执行状态基并不充分。
- 当前 Writer 已按同task跨episode训练；旧缓存拟合的负例和完整训练不是同样曝光。不能把“增加跨episode监督”当作目前完全没有的功能。

具体旧模型的源码快照、数据、标签、实际训练和停止范围，沿研究历史索引读取；不能只比较模块名或最终总分。

## 已完成的有限学习诊断

design §33从成熟 T2340比较共享 Writer S、task私有 Writer P、condition私有直接 B 修正 D。
冻结公共 β/A，使用64步新抽帧的同曝光跨episode监督；独立于本次更新的B episode与有限train闭环作末点读回。
P/D使用特权参数隔离，是机制诊断；它们不是部署候选、等函数步幅比较或理论能力上界。
每条件1792次query，没有中间结果选点；B只独立于本次64步，父T已经使用过这些train数据。

[learning_limit_rows.csv](learning_limit_rows.csv)保留128条原行，[summary](learning_limit_summary.json)
给出每task/teacher/init得失、实际目标谓词、动作风险及来源。main从八个原始results文件重算成败，
并从24份末点预测/target重算动作风险；未新增模型或环境前向。

| 臂 | 成功/32 | 相对父 R/G/L | task0 | 12 | 20 | 32 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 父T2340 | 17 | 17/0/0 | 8 | 3 | 3 | 3 |
| S共享Writer | 20 | 16/4/1 | 8 | 4 | 3 | 5 |
| P任务私有Writer | 20 | 15/5/2 | 8 | 4 | 4 | 4 |
| D条件私有直接B | 23 | 16/7/1 | 8 | 4 | 5 | 6 |

32条只有16个不同物理初态，两个teacher重复同task/init，不是32个独立样本，也不是官方400资格。
P对S得4/失4；D对S得5/失2、对P得6/失3。任务私有化没有提高总分，D也没有包含S/P的全部成功。
task32 teacher17的父/S/P/D为1/2/2/4，D补齐三例“开炉后未放壶”；teacher43为2/3/2/2，D没有学到S/P新增的init3。
task0已全成功；task20三臂各损失一个不同的父成功，正反例和teacher不对称都不能被总分抹去。

固定A-FM由.11077084到S/P/D .10897617/.10827229/.10795090；独立B20十步生成的前5 MSE相对父
分别−.00818877/−.00835784/−.00742204，改善条件6/8、8/8、6/8。
但full50差为+.00531628/+.00928010/+.00748608，valid-future也变差；前5主要改善来自gripper。
因此“独立B误差更低”不能解释D较高闭环，更不能单独定位生成器。完整论证见机制§76。

§33保存的全部38-target初始/S/P末点Z允许检验实际D修正的参数表达性，
[learning_limit_geometry.json](learning_limit_geometry.json)给出定义与逐site统计：
固定Z，拟合`B_D−B_origin≈δO Z`；以诱导权重变化`(δB)A`的Frobenius能量计，
初始Z的共享O可表达95.86%（B坐标为81.64%），38site中位数92.43%，提高截断后仍94.71%。
这是unbounded最小二乘的可表达性，未评测拟合模型；剩余方向可能对控制关键，不等于95.86%收益可学。
私有O的近乎完整表达依赖部分巨大系数，不能据此推荐直接求逆。

S/P末点的共享拟合比例虽99.17%/99.45%，绝对BA残差却为2.6313→2.6267/2.6628，几乎不变；
比例升高主要因为拟合目标能量分母扩大，不能当成Z学习改善。原始权重/Z不在远程，数值是有来源的本地读回。
P末点本身包含task私有参数；这里的“共享”仅指附加δO，不能把它解释成恢复了一套完全共享的Writer。

这一批支持现有图仍可学出有限真实修正，降低单纯task共享冲突和巨大输出不可达解释；
D优势同时含参数化、teacher共享及优化坐标的变化，尚未证明唯一主因或改进方法。
成本2.100774/4GPUh，全部计算退出，按Owner要求暂停新增实验。

## 主讨论的可反驳判断

主讨论目前更怀疑的是：在合法特征下，局部变化怎样被共同学成可在不同自身状态调用、并跨任务保持的作用映射。
这是待解释联系的定位，仍不是唯一模块根因。共享 A、较早信用或自由输出方向存在，都没有单独证明这条联系已学成。
当前降低了单纯遗忘时钟、A未学习、公共 B₀普遍强冲突、解除task共享和简单输出扩容的优先级；它们的限定反例与未测范围仍保留。
Value只直接读取当前K/H/ΔH，过去M不进入Value内容计算；这是具体依赖限制，却只在局部信息需要历史消歧时构成信息缺口。
递推仍有顺序作用；而旧LocalField已有完整时序上下文，因此不预设换Transformer就足够。
专家应独立判断这些优先级是否有依据，特别检查是否遗漏了实际特征形成、完整非线性消费者、学习条件或竞争解释。

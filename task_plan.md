# EMBER task plan

## 当前目标与授权

Owner于2026-09-07最新明确要求先与专家把证据、架构和训练推导讨论清楚，再决定开工。
**科研实现、训练与评测暂停**；此前自主科研执行授权在这一范围内被最新要求覆盖。
本轮只整理并上传已有证据、校准历史事实、准备第二轮讨论，不向外部专家发送消息。
准确状态见 [progress.md](progress.md)，讨论材料见 [审查补充包](docs/review_materials/20260907/README.md)。

已登记设计仍为 [分层局部关系视频到完整LoRA Writer](docs/layered_relation_video_writer_design.md)，当前run止于384。
该设计是已实现候选的记录，不代表可以恢复执行。Writer与读取侧Meta fresh联合训练的现有合同未被改写；
任何新路线须在讨论后登记。旧P/Q width256补评和未合并native-heads草稿均只是候选，不自动启动或集成。

原接管任务记录过长期goal；这一历史记录不覆盖当前暂停。科学完成标准为validation8 strict paired single-checkpoint correct严格 >145/400，并满足相邻稳定、低churn、
高breadth、四suite非零及Goal/Long贡献、same-task另一组正确视频鲁棒性；selected checkpoint冻结后完成必要因果controls，
方法冻结后按32/8合同fresh最终训练与Test，交付可复核代码与正式证据。实现或一次高分不算完成。
不自行设token预算、总工期或固定尝试数；停止无信息重复，依据新机制证据推进。

## 当前阶段：补齐现存证据，讨论架构与训练推导

1. 整理当前train24与读出对照、早期强Writer、SFT、多视频/LPCP/GOMQ及P/Q共享/clone/宽度对照的现存合同、metrics、曝光和raw rows。
   公共副本去除非科学现场信息，保留原始记录索引；标注缺失与未完成面板，不生成新的实验结果。
2. 请专家先根据原件修正第一轮的事实口径和证据等级，再从需求、信息条件、合法监督逐步推导表示、模块职责、参数生成与联合训练。
   每一步区分数学性质、假设、归纳偏置、实现选择和待验证能力；同时面对历史成功与失败。
3. 形成可讨论的优先路线、竞争解释和最小区分性验证计划，先与Owner讨论，再登记是否实施。
   不把专家建议、数学自洽或材料上传直接转换为实验授权。

## 暂停前执行计划快照（历史记录，不构成待执行清单）

1. **理解（已完成）。** 指定当前文档已完整阅读、Git已核对；相关原始专家评审、修正、Git快照与正式证据已核对，
   核对现有代码职责和资产。完成科学目标/数据流/历史教训/实现缺口的简要说明后立即继续；临时HANDOFF已消费删除。
2. **实现唯一canonical新图（已完成并通过真实验证）。** 原生prefix与18×50单probe读取Meta；独立双向局部帧对关系、rho/内容MLP、同步更新，
   learned H-read、多视频共同compiler与坐标A/B；复用LoRA/functional基础，补R-leaf VJP及observer分块重放、完整checkpoint/resume。
   具体bias/probe/采样和文件owner按完整设计落实。只在真正独立且节省总时间时使用最少subagents并隔离写入。
3. **真实机制与成本（已完成首轮）。** 信息墙、方向/两端归一化/rho等价性、同步更新及有限双侧上下文、真实K1/2/4置换不变、
   Meta作用域及真实功能梯度、staged VJP权重与缓存有效期。最长真实K1/K4加真实action queries测阶段成本/峰值/吞吐。
4. **短学习与行为（本轮已完成）。** 预登记跨Spatial/Object/Goal及有实质专家容量Long的训练侧任务、跨episode角色、曝光和行为节点。
   原初始化96步K1为4/4、K4为6；坐标对照为6/6、K4为8，对source4，仍无Object/Goal。保留局部Long正证据与完整图，只检验末读出按输出target/rank分别学习；同曝光fresh、无额外旁路。
   末读出对照96K1双视频均11/40、success集合一致、breadth3/4；K4为10。具备推进完整共享学习的基础，Goal仍0、相邻churn未稳。
   小面板用于投入判断，不选择最终模型；保持当前图，不继续按几何小扫。
5. **完整train24共享与迁移（当前run384止损，冻结训练诊断待完成）。** 当前图全部fresh，672步完整schedule，首段192；
   192/384/576/624/672双视频K1 strict400，后两点按正向/近目标趋势决定；192/576 train120/source诊断。
   资格数值及采样固定见设计§13.3.1与train24_shared/registration.json。真实不同视频K1/2/4，任务权重独立于K/长度/卡数；额外non-held meta有必要时先审计allowlist、
   provenance及权重。检查训练行为、same-task新视频、未见task三层接口，依据历史和新证据做最小区分性干预。
   384correct未扩展总分或breadth，Goal回落而Long首次4；已追加预登记同120初始化的已见/held正确视频诊断，
   固定384、不产生梯度或参与模型选择；熟悉视频21/120、Long0，当前配置已确定止于384，不执行剩余节点。
   held诊断已完成18/120、Long0；针对已训练任务上的行为缺口，准备直接native因子线性heads，保留上游与target/rank独立性。
   新旧参数/子空间不同，按matched短学习和闭环先裁决，不预告根因或长跑；基础SFT约108也是必须正视的性能参照。
6. **strict400与稳定性。** 在有信息量且预登记的single-checkpoint节点及时评测，强候选继续相邻点，报告per-task/suite、breadth、
   retained/gained/lost、churn、success-set overlap、实际曝光和成本；完成same-task另一组正确视频。80-row screen/union/融合不选模型。
7. **冻结与最终证据。** qualification arms和相邻口径事先登记；selected checkpoint选定冻结后做必要视频controls，
   shuffled/reversed最后测试且不进入训练、loss、Gate、checkpoint选择或架构修正。方法冻结后32/8 fresh最终训练与Test。

## 决策和交付边界

- 工程合同错误修复后重做受影响验证；科学non-pass只否定实际检验组合；证据不足不补写结论。
- clone/shared差距不能唯一命名容量或梯度冲突，functional/参数几何不代替闭环；好趋势判断相邻稳定，坏结果不靠小扫续命。
- 每次launch live检查两GPU节点，EMBER同时≤6张物理卡、同节点1--6张；大增长核验strg01独立quota、实际用量与峰值预算。
- formal train/eval从clean pushed commit的detached frozen worktree启动。隔离实现验证后集成main、推送、清理已合并task worktrees。
- 持续更新正式账本和历史；保留正式证据、唯一checkpoint、原始数据及所有权不明资产。非必要整理不阻塞可运行科学节点。
- 不联系外部专家。只有改变科学目标/信息墙、未授权资源数据、不可安全裁决的重大投入分歧或越权破坏性操作才需要owner决定。

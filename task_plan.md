# EMBER task plan

## 当前目标与授权

Owner于9月7日完成局部关系模块的重新推导对齐，并要求更新仓库与新session启动prompt。本session负责数学/实现说明、
长期要求、当前状态、历史及交接材料的同步、文档核对和main交付；前次代码/存储清理已完成。
新架构实现及科学训练/评测尚未启动；新session先理解仓库、报告完整认识与执行计划，得到owner明确同意后再自主推进。
旧goal、旧design或历史未完成清单不构成启动授权。当前记录见 [progress.md](progress.md)。

已登记设计：[分层局部关系视频到完整LoRA Writer](docs/layered_relation_video_writer_design.md)，状态为已对齐、尚未实现。
最终目标仍是validation8 strict paired single-checkpoint correct严格 >145/400，并满足相邻稳定、低churn、高breadth、四suite非零、
Goal/Long贡献、same-task另一组正确视频鲁棒性，以及selected checkpoint冻结后的必要性/时序因果controls；方法冻结后按32/8合同fresh最终训练与Test。

## 本次交接交付

1. [x] 完整记录科学精神、推导、T/H/J职责、单probe、观察Meta、多视频、坐标decoder与GPU梯度算法。
2. [x] 按所有权与重建依赖清理大派生缓存，保留唯一checkpoint、数据、正式结果和不可重建小缓存。
3. [x] 退役旧专用代码/脚本/配置，整合必要公共运行基础；更新长期要求、分层历史与文档入口。
4. [x] 前次主工作区验证、工作树及临时材料清理、main交付已完成。
5. [x] 9月7日更新双向局部关系、对应模式消费、堆叠及GPU算法/验证定义，刷新正式文档与临时HANDOFF；本次文档核对见progress。

## Owner同意后的执行路线

下列是后续计划，不是当前启动指令。预算与具体节点根据实现和真实profile登记，不预先虚构速度、分数、根因或固定总尝试数。

1. **确认共同理解与实施合同。** 读完当前authority、完整设计和分层历史，检查相关现有代码与canonical资产。
   解释每个模块如何服务科学问题、哪些是已知事实/归纳偏置/默认/待检验假设；给出文件归属、数据/采样、验证与行为节点计划。
   新session在此获得owner明确启动指令后，连续完成以下已授权工作，不为每个常规步骤重复确认。
2. **实现唯一canonical新图。** 保留真实prefix、18层×50-horizon，单probe只读Action Expert Meta；实现同层双向帧对关系、
   内容/对应模式关系MLP、逐帧聚合与同型block堆叠、learned horizon read、多视频共同compiler与原生坐标A/B decoder。复用现有LoRA和functional基础，完成Meta-on R-leaf VJP/observer分块重放与新checkpoint schema。
   incompatible架构fresh；首版不追加raw X/Y bank、双probe或无独立职责旁路。
3. **验证真实机制与执行成本。** 检查信息墙、两端分别归一化/方向、rho索引与等价读出、block同步更新及有限双侧上下文，
   以及真实K1/2/4集合语义、观察与执行作用域、实际功能梯度、staged VJP权重与缓存有效期。
   用最长真实视频及query batch选择布局/attention/chunk，保留H和Meta梯度；CPU smoke不能代替这些GPU机制证据。
4. **尽快判断能否学习与形成行为。** 用具代表性的训练侧任务覆盖各suite和有实质容量上界的Long；真实跨episode视频/query训练，
   在登记节点检查训练行为、同task新视频及共享学习代价。比较实际曝光、task权重、预算和R/G/L；既不只报loss，也不直接先跑约10小时长训练。
5. **推进共享训练与未见task迁移。** 新合同明确完整train24；有科学理由使用额外non-held meta时保留精确allowlist与分层权重。
   如声称dynamic K，训练必须覆盖真实不同视频的K1/2/4。保留同拓扑fully-random fresh端到端候选；组件初始化只是另一候选。
   定位最早失败接口，按历史已排除的解释和新证据选择实质修正；不无依据小扫或恢复退役fallback。
6. **用strict400裁决强候选。** 有接近强基线的广泛行为后及时做预登记single-checkpoint400；正式选择只看qualification arms与相邻稳定。
   报告per-task/suite、breadth、R/G/L、churn与success-set overlap，完成同task另一组正确视频。80-row screen不选最终模型。
7. **冻结后完成因果证据与最终评测。** selected checkpoint冻结后运行登记的视频必要性controls；shuffled/reversed最后测试且不得反哺架构或选择。
   方法冻结后按规定32 source/8 test从fresh训练并进行最终Test；整套证据共同判断是否达成最终目标。

## 失败与继续的依据

- 明确工程合同错误：修复实际根因并重做受影响验证；不把故障当科学non-pass。
- 有效科学non-pass：只否定实际检验组合。区分单task学习、共享代价、同task视频变化、未见task迁移和functional/闭环分离。
- 有希望的结果：继续到足以判断相邻稳定性；明确坏结果停止无信息续训，先用有限干预区分竞争解释。
- 查阅 [findings.md](findings.md) 和 [research_history.md](docs/research_history.md) 后选择诊断；不用一次gradient cosine或局部问题命名整个系统根因。
- 真正可并行的实现、审计或分析由主agent主动委派，隔离写入；主agent负责科学决策、集成、验证与main推送。

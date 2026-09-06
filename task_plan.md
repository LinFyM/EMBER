# EMBER task plan

## 当前目标与授权

Owner于2026-09-07明确授权接管者充分理解后连续自主科研执行，覆盖实现、验证、训练、评测、诊断及证据支持的修正。
旧交接中的未授权/等待同意已失效；进度说明不等待回复。持续授权及当前现场见 [progress.md](progress.md)。
已登记active design：[分层局部关系视频到完整LoRA Writer](docs/layered_relation_video_writer_design.md)。
当前训练主线为Writer和读取侧Meta fresh端到端联合训练，fresh optimizer/scheduler；基础source冻结，LoRA采用合法identity初始化。
不实施G1--G3分阶段冻结课程，不额外建立阶段初始化候选，不恢复旧width256或无关旧运行。

环境长期goal已创建。完成标准为validation8 strict paired single-checkpoint correct严格 >145/400，并满足相邻稳定、低churn、
高breadth、四suite非零及Goal/Long贡献、same-task另一组正确视频鲁棒性；selected checkpoint冻结后完成必要因果controls，
方法冻结后按32/8合同fresh最终训练与Test，交付可复核代码与正式证据。实现或一次高分不算完成。
不自行设token预算、总工期或固定尝试数；停止无信息重复，依据新机制证据推进。

## 当前阶段：formal short4学习与闭环比较

1. **理解（已完成）。** 指定当前文档已完整阅读、Git已核对；相关原始专家评审、修正、Git快照与正式证据已核对，
   核对现有代码职责和资产。完成科学目标/数据流/历史教训/实现缺口的简要说明后立即继续；临时HANDOFF已消费删除。
2. **实现唯一canonical新图（已完成并通过真实验证）。** 原生prefix与18×50单probe读取Meta；独立双向局部帧对关系、rho/内容MLP、同步更新，
   learned H-read、多视频共同compiler与坐标A/B；复用LoRA/functional基础，补R-leaf VJP及observer分块重放、完整checkpoint/resume。
   具体bias/probe/采样和文件owner按完整设计落实。只在真正独立且节省总时间时使用最少subagents并隔离写入。
3. **真实机制与成本（已完成首轮）。** 信息墙、方向/两端归一化/rho等价性、同步更新及有限双侧上下文、真实K1/2/4置换不变、
   Meta作用域及真实功能梯度、staged VJP权重与缓存有效期。最长真实K1/K4加真实action queries测阶段成本/峰值/吞吐。
4. **短学习与行为（进行中：16步诊断已完成，48步已训练完并开始诊断）。** 预登记跨Spatial/Object/Goal及有实质专家容量Long的训练侧任务、跨episode角色、曝光和行为节点。
   Writer/Meta联合更新；小面板用于定位与投入判断，不选择最终模型。未见基础行为前不默认投入约10小时长训练。
5. **完整train24共享与迁移。** 真实不同视频K1/2/4，任务权重独立于K/长度/卡数；额外non-held meta有必要时先审计allowlist、
   provenance及权重。检查训练行为、same-task新视频、未见task三层接口，依据历史和新证据做最小区分性干预。
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

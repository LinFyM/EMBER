# EMBER 新专家入口：可实施主设计与首批完整学习

更新于2026-10-08。本轮已纳入第六轮架构、主讨论审阅和owner最新澄清，供新专家独立接续。
本目录是研究讨论的永久归档与阅读入口，不是active design，也不授权启动新实验。

**最新委托：选定一套立足现有数据、经过实质推导、能够完整实施的主设计，并给首批小而完整学习方案，
使这一轮回答足以支持冻结科学合同、进入实施。** 整个教学—实践—固定LoRA过程替代Writer；
视频必须贡献实际有用的操作知识，不能以形式读取掩盖额外任务内训练。第一阶段先超过强MT、不前置全部视频controls的原则保持。
Owner基本接受教学引导实践、发现有效行为再学进LoRA的方向，没有逐项接受第六轮模块、目标或规模。

## 建议阅读顺序

1. [最新架构审阅](ARCHITECTURE_REVIEW.md)：第六轮具体建议、主讨论认可、四条待补连接、实际代码/数据入口和本轮交付目标。
2. [第六轮架构正文](EXPERT_RESPONSE_ROUND6.md)和[owner原话与实际后续请求](OWNER_FOLLOWUP_REQUESTS.md)第4、5节：
   理解刚刚讨论的主设计及最新核心，不能只根据审阅摘要推断专家原意。
3. [讨论衔接与最新边界](DISCUSSION_BRIEF.md)：目标、原机制、历史正反证据和六轮意见的变化；
   旧“零交互/无奖励”规则按历史解释。
4. [长期要求](../../current_owner_requirements.md)、[概念](../../concept.md)、[当前状态](../../../progress.md)：
   最新信息边界、科研原则、职责及授权。当前无新active design；早期阶段条款只解释对应实验。
5. [历史推理路线](../20261007_research_reassessment/HISTORY_MAP.md)、
   [实际T/C机制](../20261007_research_reassessment/MECHANISM.md)、
   [核心证据](../20261007_research_reassessment/EVIDENCE.md)：建立完整历史，不只对上一版候选做局部修补。
   与所提方法最近似的路线，再沿设计、Git和原行索引核对。
6. 需要理解建议如何改变时，读[第一轮](../20261007_research_reassessment/EXPERT_RESPONSE.md)、
   [第二轮](../20261007_research_reassessment/EXPERT_RESPONSE_ROUND2.md)、
   [第三轮](../20261007_research_reassessment/EXPERT_RESPONSE_ROUND3.md)、[第四轮](EXPERT_RESPONSE_ROUND4.md)及[第五轮](EXPERT_RESPONSE_ROUND5.md)。

[新对话可直接使用的完整提示词](EXPERT_PROMPT_FRESH.md)是当前委托入口。
原[EXPERT_PROMPT](EXPERT_PROMPT.md)是第四轮的历史请求，保留原文，不承担当前委托。
此前的新对话提示词版本由Git保存，不再建立并行的“最新”文件。

## 归档边界

- 第四、第五轮回复和owner已发送的后续请求按原文归档；第六轮保留架构、训练、规划与困难处理正文，略去末尾总结句。
- [架构审阅](ARCHITECTURE_REVIEW.md)是主讨论判断，不能当作专家已经承认的结论；候选设计、成本估计和首批规模均未执行。
- 讨论衔接由主讨论整理，明确区分事实、专家假说、主讨论判断与owner要求；它不代替原文或原始证据。
- 旧会话的专用引文标记可能无法在新会话解析；使用正文Git链接、上述证据包与论文原文。
- 远程材料包含设计、代码身份、主要指标及可导出的原行，并不包含全部模型、数据、轨迹或大张量。
  依赖缺失材料的判断须标明未核实；不能声称重新跑过实验。
- 历史Test和shuffled/reversed封存结果不用于新方法设计或选点；旧协议分数不与当前协议混排。

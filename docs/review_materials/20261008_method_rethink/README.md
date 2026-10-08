# EMBER 新专家对话入口：完整架构设计

更新于2026-10-08。旧专家对话过长且加载异常，owner要求将主要内容存入仓库，供新对话独立接续。
本目录是研究讨论的永久归档与阅读入口，不是active design，也不授权启动新实验。

**最新委托：先形成一套具体、完整的方法架构，贯通共享训练、教学与实践适应、最终固定LoRA执行；
再由该架构推导渐进推进规划和关键困难的处理原则。** Owner基本接受“教学引导实践、发现有效行为、
再将能力训练进最终LoRA”的方向，但没有逐项接受第五轮的模块或首批规模。

## 建议阅读顺序

1. [讨论衔接与最新边界](DISCUSSION_BRIEF.md)：任务全貌、原机制、历史正反证据、五轮专家意见的变化、
   owner后续修正及尚未完成的要求。先读本页链接，避免把旧“零交互/无奖励”规则当作最新限制。
2. [第五轮完整回复](EXPERT_RESPONSE_ROUND5.md)和[owner后续澄清及最后一轮完整请求](OWNER_FOLLOWUP_REQUESTS.md)：
   理解已接受的方向与真正待补的架构连接，由新专家完成最后一轮要求。
3. [长期要求](../../current_owner_requirements.md)、[概念](../../concept.md)、[当前状态](../../../progress.md)：
   最新信息边界、科研原则、职责及授权。当前无新active design；早期阶段条款只解释对应实验。
4. [历史推理路线](../20261007_research_reassessment/HISTORY_MAP.md)、
   [实际T/C机制](../20261007_research_reassessment/MECHANISM.md)、
   [核心证据](../20261007_research_reassessment/EVIDENCE.md)：建立完整历史，不只对上一版候选做局部修补。
   与所提方法最近似的路线，再沿设计、Git和原行索引核对。
5. 需要理解建议如何改变时，读[第一轮](../20261007_research_reassessment/EXPERT_RESPONSE.md)、
   [第二轮](../20261007_research_reassessment/EXPERT_RESPONSE_ROUND2.md)、
   [第三轮](../20261007_research_reassessment/EXPERT_RESPONSE_ROUND3.md)及[第四轮](EXPERT_RESPONSE_ROUND4.md)。

[新对话可直接使用的完整提示词](EXPERT_PROMPT_FRESH.md)是当前委托入口。
原[EXPERT_PROMPT](EXPERT_PROMPT.md)是第四轮的历史请求，保留原文，不承担当前委托。

## 归档边界

- 第四、第五轮专家回复及owner已发送的后续请求按原文归档，最后一轮请求就是新专家本次要完成的任务。
- 讨论衔接由主讨论整理，明确区分事实、专家假说、主讨论判断与owner要求；它不代替原文或原始证据。
- 旧会话的专用引文标记可能无法在新会话解析；使用正文Git链接、上述证据包与论文原文。
- 远程材料包含设计、代码身份、主要指标及可导出的原行，并不包含全部模型、数据、轨迹或大张量。
  依赖缺失材料的判断须标明未核实；不能声称重新跑过实验。
- 历史Test和shuffled/reversed封存结果不用于新方法设计或选点；旧协议分数不与当前协议混排。

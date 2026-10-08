# EMBER 新专家入口：视频到LoRA编译的完整设计

更新于2026-10-09。Owner要求新专家充分理解最新边界与已有证据，独立设计完整方法，给出具体数学推导及详细推进规划。
[EXPERT_PROMPT_FRESH](EXPERT_PROMPT_FRESH.md)是唯一当前提示词，旧版本由Git保留。本目录是讨论与证据入口，不是active design，不授权实验。

## 先明确方法边界

**任何先准备动作轨迹数据，再通过微调得到最终LoRA的路线都不接受。** 恢复、合成、搜索、规划、自身真实采集，
以及可学习数据准备器、交错采集／微调、先生成后末端微调，都不豁免。性能更好或视频确有增益也不能改变这个判定。
共享Writer训练时使用合法动作／功能监督允许；面对新教学条件构造动作训练集后拟合最终LoRA不允许。
自身实践和反馈仍可帮助视频理解与参数编译，不据此另设单网络、单次前向或全面禁止梯度的限制。

第五至第七轮“发现有效行为再直接训练LoRA”的主线已因上述形式被否决，未实施的完整组合不记作实验阴性。
此前材料中的“基本接受”“当前方案”“可以实施”只代表原时点。新专家应独立设计，不负责补齐或维护这些被否决的模块。

## 建议阅读顺序

1. [当前完整提示词](EXPERT_PROMPT_FRESH.md)、[Owner原话第6、7节](OWNER_FOLLOWUP_REQUESTS.md)，理解排除条件与本次交付要求。
2. [长期要求](../../current_owner_requirements.md)、[概念](../../concept.md)、[AGENTS](../../../AGENTS.md)及[当前状态](../../../progress.md)，
   理解信息来源、编译与执行边界、数据／评测合同和授权。旧条款按其时点解释，最新Owner表达优先。
3. [讨论衔接](DISCUSSION_BRIEF.md)，区分现有资产、关键正反证据、七轮专家建议和实际实施边界。
4. [历史图谱](../20261007_research_reassessment/HISTORY_MAP.md)、[T/C实际机制](../20261007_research_reassessment/MECHANISM.md)、
   [原件入口](../20261007_research_reassessment/EVIDENCE.md)及[研究历史](../../research_history.md)。
   沿索引核对与新候选最接近的完整论证、修订和原件，不必无差别重读所有历史。
5. 需要核对历史建议原意时，读[第一轮](../20261007_research_reassessment/EXPERT_RESPONSE.md)、
   [第二轮](../20261007_research_reassessment/EXPERT_RESPONSE_ROUND2.md)、[第三轮](../20261007_research_reassessment/EXPERT_RESPONSE_ROUND3.md)、
   [第四轮](EXPERT_RESPONSE_ROUND4.md)、[第五轮](EXPERT_RESPONSE_ROUND5.md)、[第六轮](EXPERT_RESPONSE_ROUND6.md)、[第七轮完整原文](EXPERT_RESPONSE_ROUND7.md)。
   [第六轮主讨论审阅](ARCHITECTURE_REVIEW.md)保留当时判断，其认可不构成现在的方案约束。

## 本次希望得到的结果

一套有明确方法选择的完整架构，贯通共享训练、教学与实践的参数编译、最终固定策略执行。
数学须落实到实际特征、算子、各层A/B、真实自身hidden与动作、标签和梯度消费者；区分可表示、可学到、可迁移。
给出训练与编译伪代码，用一个现有任务贯通，并说明相对最近似历史方法改变了什么、仍需检验什么。

详细规划从小而完整的学习／闭环循环开始，按依赖说明产物、判断分支、wall-clock／GPU小时／存储、扩大及停止条件。
分别计算共享训练、单condition编译／实践和最终评估成本，不能用8次任务适应代替正式400个不同教学条件。
首批合同是待主讨论科学审阅的草案，不是直接实验授权；专家由Owner自行联系，后续实现由实验session承接。

## 原件与可访问范围

第七轮使用Owner提供的完整文本归档；第六轮保存架构、训练、规划与困难处理正文，略去末尾总结句，其余轮次保持原存档范围。
讨论衔接及审阅是主讨论整理，不替代专家原文或实验原件。旧专家建议中的实施指令均不直接执行。
远程材料有设计、代码身份、统计与可导出原行，不包含全部模型、数据、轨迹或大张量。缺失材料须明确，不假称重新核对或运行。
历史Test及shuffled／reversed封存结果不反哺新设计或选点，不跨协议混排分数。
原[EXPERT_PROMPT](EXPERT_PROMPT.md)只保留第四轮历史请求，当前委托统一使用[EXPERT_PROMPT_FRESH](EXPERT_PROMPT_FRESH.md)。

# 新 Writer 专家审阅与修订入口

状态：2026-10-11，专家意见及原八rank128教师结果已消费。Owner定案不用T、fresh双meta读取及自主实施，
进一步开放MT合并基底和小rank修订；[当前设计§13](../../designs/video_guided_proposal_writer_20261010.md#13-当前正式实例rank8多起点完整链2026-10-11)
登记rank8多起点完整后继，实际状态见progress。专家审阅对象仍是原快照，不冒称已审阅本次全部修订。

## 当前应读什么

1. [专家审阅后的完整设计](../../designs/video_guided_proposal_writer_20261010.md)：当前唯一完整架构定义。
2. [专家原文](EXPERT_REVIEW.md)：Owner提供，审阅对象为 `be0d469a`，建议不自动成为执行合同。
3. [主讨论的修订论证](../../analyses/architecture_rethink_discussion_20261010.md#12-消费专家意见保留分工补齐可执行定义与因果归因)：接受、进一步修正和未机械采用的内容。
4. [Owner要求](../../current_owner_requirements.md)、[concept](../../concept.md)、[当前状态](../../../progress.md)：科学目标、信息墙与实际授权。

[原送审稿](https://github.com/LinFyM/EMBER/blob/be0d469a33deb4d5dcc8deb078609b67ab4a1042/docs/review_materials/20261010_writer_design_review/README.md)
和[原提示词](EXPERT_PROMPT.md)保留审阅时点，不是修订后的当前算法。

## 当前完整链

```text
合法有序视频/语言 ── 双meta原生读取 ─┬─ G自己的Reader ─ 完整参数条件流 ─ 候选LoRA
                                   └─ π自己的Reader ─┐
实际历史/完整参数/真实观测上的候选动作预测 ──────────────┤
                                                    └─ parent/STOP、practice、final
真实实践追加历史 → 可继续修订 → 唯一固定完整LoRA → 独立新初态
```

G及两组读取meta由合法训练任务中的实际教师编辑监督学习；π的final用途由真实候选后果预训练，
π学习期间固定全部G与meta，用完整编译路径的类别score训练父策略、实践、停止和最终选择。
最终执行原生官方ODE；新condition没有训练教师、正确动作轨迹拟合或第二adapter。
视频应该帮助理解操作和选择有用控制，不能用形式读取、参数差异或额外实践总收益替代视频证据。

## 已补齐的定义

- 教师从真实parent复制完整A/B，目标、恢复专家适用性、成功保持来源、固定坐标和预定节点均明确。
- preferred/coverage真正决定G标签分布；负向和不确定编辑不被冒称有益，专门探测行为不是首版已解决的能力。
- G有完整64数值块通路和block/rank/target双向通信；固定16步参数积分，零速度头不等于identity。
- Owner最新改为fresh Gemma/action meta，区分contextual prefix与raw patch；训练时梯度贯通，更新后缓存失效。
- 读取meta与执行完整LoRA使用明确参数上下文，不加载T，也不把读取meta叠入最终执行。
- π空历史有常驻query、三个阶段分别定义用途、绝对动作响应不依赖虚构parent。
- 原样parent可实践；最终候选资格与训练一致；score包含完整路径，缓存和版本更新边界明确。
- U/P/selected区分生成、实践漏选和最终误选；selection标签池与独立audit报告分开。

## 判断到哪里为止

专家和主讨论均未发现推翻G＋π核心分工的硬数学错误；已识别的定义缺项已在具体设计中处理。
这不证明教师供给、G控制传递、视频操作收益、有限预算选择或跨任务泛化已经成立。
弱起点可定义、可获得共享监督，不等于现有强起点附近的模型已能在弱起点学成。
不能因保留MT就宣称实际不退步，也不能因类别score正确就宣称实践信息和有效候选充足。

首版只保留强MT执行起点，K1及合法36任务总范围不变；首批为四任务完整pilot，随后按实际证据展开。
正式400节点由完整训练与吞吐证据登记；没有因本次定案开放held离线标签或提前Test。

## 相关历史原件

| 材料 | 必须继承的边界 |
| --- | --- |
| [Experience设计](../../designs/experience_conditioned_compiler_20261009.md)／[结果](../../analyses/experience_conditioned_compiler_20261009.md) | 已有完整参数、经历及真实RL；明确实践分布semi-gradient，135/400对MT153不足 |
| [Parameter原结果](../../analyses/parameter_conditioned_compiler_20261009.md)／[完整读出](../../analyses/parameter_compiler_fullreadout_20261009.md) | 127/400、后续130；Long1局部正例不能抹去，整体与训练域净修订不足 |
| [固定incoming诊断](../../analyses/parameter_edit_credit_diagnostic_20261009.md) | I/E+/E0同为13/24，经历有增有失；专家FM更优仍可丢成功 |
| [Functional设计](../../designs/functional_revision_compiler_20261010.md)／[正式结果](../../analyses/functional_revision_learning_20261010.md) | 实际是最多16×5×7响应的完整导数；正式FM360得143/400、未训练正式PG，不能简化成单个7维出口 |
| [对齐教师原件](../../analyses/aligned_teacher_recovery_20261008.json) | 四任务有限初态/接手正例，不是覆盖36任务的恢复oracle |
| [更早功能监督审计](../../analyses/flow_supervision_history_20260926.md)／[历史索引](../../research_history.md) | Native Correction、phase/decoder与learner-state监督的负证据不因新名称清零 |

当前[源码](../../../src/ember/proposal_writer/)已承接G/π及训练教师，旧experience算法退役；原八教师有实际控制收益，
G/meta/π尚无formal学习。rank8参数上下文、尺度和多起点实例由实验session按新合同承接，不能把已有机制当作新能力证据。

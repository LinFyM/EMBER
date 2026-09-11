# EMBER：有益视频特异性的综合理论重构

这是供**没有旧对话上下文的全新外部专家**阅读的当前入口。请求是综合全部已有正负证据，提出机制理论，并从原始输入、表示、
策略参数作用一路推导到架构与训练；不是再给当前模型加一轮补丁，也不是先扩大实验量再寻找解释。
先读事实、代码和比较边界，形成独立判断；以前专家的原文按需查阅，不作为本轮结论或必须采纳的建议。

## 1. Owner最新目标与本次边界

- **阶段重点：恢复有益的视频特异性，暂不强制绝对性能。** 正确任务内容及内部顺序应使生成LoRA获得真实执行价值，
  错任务或非正序视频不应产生同样的教学收益。请协助精确定义可操作、可证伪的验收，而非把所有错误条件强制为零。
- 项目长期目标仍是validation8 strict single-checkpoint correct>145/400及稳定、任务广度、换视频与视频因果资格，
  **该数值及历史SFT/source总分门槛不作为本阶段的额外完成条件**。应继续报告绝对成绩和参照，区分正确条件获得收益与仅让错误条件退化。
- **明确不直接回到v5.2，也不预设在v5.2底座上改进。** v5.2的性能和稳定性不达标；它的普通监督视频依赖正例是必须解释的重点。
  Owner认为后继没有整体超过它；证据上存在v6/GOMQ更高单项结果，尚无同时覆盖能力、稳定和特异性的整体胜出方案。
  应结合全部历史重新设计，不把不同配方的长处拼成一个虚构模型，也不预设保留Horizon全部组件。
- 当前只整理材料和咨询，由Owner转发。本次没有新训练、模型forward、rollout或checkpoint选择；科研执行仍暂停，95-task不恢复。
  没有获准执行的新active design。最新权限见[progress](../../../progress.md)，长期原则见[owner requirements](../../current_owner_requirements.md)。

研究任务仍是：准确任务语言＋action-hidden有序教学视频，在rollout前一次生成冻结π0.5 source的完整task-conditioned LoRA，
随后依靠机器人自身观测从新初始化闭环执行。语言、视频和执行观测各自的职责需要推导清楚。跨具身是科学动机，目前LIBERO不证明跨身体泛化。

## 2. 建议阅读路线

1. 本页和[咨询问题](EXPERT_PROMPT.md)：理解目标，避免沿用旧145门槛或旧持续实验授权。
2. [历史机制证据地图](EVIDENCE_MAP.md)：按信息、输出与学习机制寻找正例、反例、混杂和原件。
3. [最新结果与证据定位](LATEST_EVIDENCE.md)：R/C/S、续训、冻结诊断及局部参照；该页把事实与解释边界分开。
4. [concept](../../concept.md)、[研究历史](../../research_history.md)、[findings](../../../findings.md)，按具体问题查相应原设计与源码。
5. 当前源码从`writer/native.py → relation.py/horizon.py → semantic.py/native_factor.py → supervised.py/functional.py/training.py`阅读。
   当前完整Horizon设计及局部参照设计是待分析方法的合同，不能视为本次新方案已确定。
6. 查原始证据时使用本包[index.json](index.json)和[panel_summary.json](panel_summary.json)，再沿下面的旧包索引读取。
   不必把全部JSON一次性放入上下文。旧专家意见仅在核对形成过程或某项论证时补读。

| 证据范围 | 远程入口 | 能直接读取的内容 |
| --- | --- | --- |
| v5.2、task-complete、v6-fast、SFT、多视频、LPCP、GOMQ、P/Q/clone、9月7日关系图 | [9月7日索引](../20260907/index.json)、[重算汇总](../20260907/rollout_summary.json) | 既有95个outcome面板、24100条rows及训练/诊断记录；不是全部历史原件 |
| Horizon all/off、两初始化、D绑定、查询/信用/实际更新、最强模型九臂 | [9月11日旧索引](../20260911/index.json)、[汇总](../20260911/panel_summary.json) | 119个面板、13901条rows及相关方法、分析和已有轨迹图 |
| R/C/S、off/R续训、C完整冻结诊断、无变化参照首段 | [本包索引](index.json)、[本包汇总](panel_summary.json) | 45个面板、7216条rows；其中192条是明确复用的参照，不是新增执行 |
| G1/G2/G3等较早接口与条件边界 | [历史地图](EVIDENCE_MAP.md)、[分层历史](../../research_history.md) | 当前摘要、冻结Git账本、原设计/代码入口；未声称完整原始张量或每个面板均已导出 |

## 3. 模型身份：当前main不是旧126候选，也不是原C

| 身份 | 原科学commit | 合同与状态 |
| --- | --- | --- |
| 旧Horizon off7/off11 | `45e16633394e1baa9af8cefba698661b41aab539` | 126/119候选；[旧合同](../20260911/model_records/off7/run_contract.json)，统一P4消费；未获最终资格 |
| R/C/S | `7fedbe85b5eb6835e9fa50c313068a276a3fb63b` | [R](model_records/r/run_contract.json)、[C](model_records/c/run_contract.json)、[S](model_records/s/run_contract.json)；R到400，C/S到200 |
| off7继续至500/600 | `45e16633394e1baa9af8cefba698661b41aab539` | [续训合同](model_records/off_continuation/run_contract.json)；保留exploratory lineage，未重标formal fresh |
| 当前保留的无变化参照实现 | `64eba75bcf316a1f6792ab8ddf5eb1aaefded378` | [合同](model_records/nochange/run_contract.json)；fresh100/200，200 val34；100验证未完成，300/400未运行 |

审阅仓库使用Owner提供的冻结提交；实验各自使用表中科学版本。当前源码默认局部更新为`paired_nochange_reference_v1`，
与原C不同；不能把旧C checkpoint套入当前runtime，再把行为变化当旧模型结果。

## 4. 科学约束与可重新推导的选择

固定LIBERO train24/validation8/test8；source从generic `lerobot/pi05_base`和排除目标重合后的source71建立，normalization冻结。
部署Writer不读取teacher actions/state/reward/terminal/task ID/filename/pose，执行policy可读取自己的观测与state；
输出唯一完整38-target A/B，不部署任务字典、第二expert或闭环中反复编译，不做部署时task-local优化。Test继续封存，held不产生梯度。

当前集中K1、stride5；Action Expert的动作知识参与理解，完整50-H保留到真实task-conditioned跨帧learned read。
当前过去窗口/单向长程/GRU/四组/native D/Meta范围/条件分配与纯FM，是有来源的具体方法及待解释的实验因素；
本次允许提出有依据的实质架构和训练重构，不把它们全部视作永久不变量。若建议改变既定科学约束，请明确指出，不能静默恢复旧coarse或horizon mean。
旧G1–G3机制分段不是当前训练课程；独立RL只是历史后续方向，不能预设它能解决视频利用。

开发诊断曾获Owner授权使用wrong/shuffle作正常输入干预；它们未用于错误惩罚训练，不能冒充未触碰的最终冻结验证。
建议验收应同时区分任务内容特异性、顺序特异性、同task换视频稳定、训练任务获取和未见task迁移。具体阈值与面板尚未替新方案登记。

## 5. 复算与数据解释

在仓库根目录运行`python docs/review_materials/video_specificity_20260911/verify_evidence.py`。
它只读取已提交数据，复算面板总分、逐task/suite、breadth及登记的配对成功集合；不会运行模型或环境。
旧包有各自的汇总与9月11日验证脚本，可按需要复算。

本包源记录导出约32.75 MiB。逐行保留task/state、目标语言、teacher/frame信息、success、steps、完整env/policy RNG及已有行为字段。
每个面板的恒定method/checkpoint/source内容存于metadata的`horizon_writer_lora_common`，合回各行可还原规范化内容；
功能诊断表的分片在metadata的`review_export_rows`中列出。导出时已把序列化结果与规范化原件核对，没有新增hash。

主机、进程、凭据、启动命令与资源快照移除，机器绝对路径使用占位符；运行ID仅作科学provenance。
`index.source`表示本地原件身份，实际远程文件由`exports`给出；摘要里保留的`runs/`引用不保证远程存在。
方法脚本以不可执行的`.py.txt`保存，用于检查真实RGB变换、路径替换和统计计算；其本地依赖不表示专家可以直接重跑GPU。

未提供模型/optimizer张量、数据集、完整视频、大型激活数组和全部历史原始面板。本包不伪造缺失的100验证、300/400或95-task成绩。
历史状态、decision与“下一步”字段保留原意，只描述当时；当前权限始终以根目录progress为准。

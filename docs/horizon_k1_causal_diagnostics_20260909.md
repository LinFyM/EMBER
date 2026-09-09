# K1能力缺口：冻结接口与监督诊断

日期：2026-09-09。当前是诊断协议与执行记录，尚无新机制结论。Owner允许深入分析、修正及非正式验证，明确禁止正式架构/训练方式修改和正式训练启动。已有上下文400评测收尾继续；canonical Writer、Meta、source与全部checkpoint保留。

## 1. 已知问题与竞争解释

- 原始K1 correct55/110/86/87，移除Compiler直接语言内容残差后75/110/106/103；本轮逐帧上下文条件52/103/79，400待完整。前一干预有局部收益，未修复获取/保持；本轮至300无整体优势。
- 上一轮train96由46升59，训练内adapter功能对应增强；本轮41升49，400仍低于前轮59。不支持普遍未学会条件编译，也不证明训练总体已解决。
- 本轮400 held FM .105074533，比前轮.105737594低，但train96反而少10；平均误差不能作为闭环收益代理。
- 既有回放包含错误对象/实例、正确子目标后未完成、抓取/时限与干扰物等多种缺口；不能统一归因motor或单个过程模块。

本轮检验四类解释：额外语言条件影响信息使用；有用过程关系未形成或在读出中未保留；Compiler/native读出未把表示转成正确功能修正；监督分布/共享更新与闭环关键决策存在不一致。它们可能共同存在，不预先要求唯一根因。

## 2. 证据等级与边界

1. 完整已有闭环、逐任务配对及实际运行合同是行为事实。
2. 冻结接口干预确认特定参数状态下的功能依赖；大幅干预可能偏离训练时激活分布，不能替代fresh学习或证明删除后有益。
3. 局部可达性/oracle若实施，只能说明固定接口及预算内能否改善，不自动说明共享可学习、跨任务可迁移或deployment合法。须在执行前另记数据、变量与预算。
4. 梯度、几何、attention图或微小FM差异只作定位。最终方案以证据支持程度和可区分下一步表达，不把结构名字当机制证据。

不访问validation/test actions或产生其梯度；不使用最终wrong/static/no-video/shuffled/reversed controls反哺设计。模型诊断只用train24。已有正式validation只按登记口径完成分析。

## 3. B1冻结分支干预（结果前登记）

对象固定为同架构上下文checkpoint200/400，冻结运行面`9abc9b95`。全部train24、正确teacher46/47；每task固定seed=`20260908+task`的32条独立动作queries，来自原diagnostic episodes42–45。两个teacher、两个checkpoint和全部arms共享实际query/frame/time/noise。训练sampler不推进，source/Writer/Meta无参数更新或梯度。

九臂每条件均完整执行：

| 臂 | 精确定义 | 保留 |
|---|---|---|
| normal | 原生完整运行 | 全图 |
| local_language_zero | 四组local入口的language张量置零 | 原生语言、上下文Z、H-read、Compiler语言 |
| hread_language_zero | 四组read_language线性层输入置零，保留其已学bias | H-read attention和数据依赖K/V；不是重新学习的Query |
| compiler_language_zero | query_language输出置零 | target/rank首次query、P4和全部残差/解码 |
| all_backend_language_zero | 同时上述三项 | 原生语言/Z task tokens/R任务条件化 |
| visual_read_zero | 四组两端visual_read的最终output置零 | 原生Z语言条件/R、对应与局部组织 |
| local_neighbor_zero | 四组neighbor_output输出置零 | 当前H、local后FFN、H-read/长程 |
| temporal_attention_zero | 四组temporal.attention.output置零 | temporal残差与FFN、局部关系 |
| writeback_zero | 前三组writeback返回其输入states | 所有组局部/H-read/长程及第四组P4 |

临时hook只存在于独立分析进程，不改正式源码/配置、checkpoint或输入视频；视频不重排、不删帧，H=50与原生prefix保持。干预支持的是特定分支的边际依赖，不能把visual_read零当全部视觉消除，也不能把all language零当无语言模型。

每条件记录原生50×7 loss的逐query、horizon与动作维度统计，前5/后45对照，实际速度预测扰动，hook调用覆盖与输入配对。全量统计到24task才解释；不由单task正负选checkpoint。两teacher分别报告再汇总，防止一条视频偶然性。附同query source参照。

可复用此前已核验的冻结query-prefix缓存与原生denoise_step，避免每arm重复冻结prefix。首task每checkpoint的normal与source各做一次8query完整native forward核对；沿用逐点loss平均绝对误差≤.002且相对≤2%的既有界限。该核对不是全体误差上界，接近数值差异尺度的结果保持不确定；必要后续仅对有决策价值的差异使用完整native forward复核。

## 4. 并行分析与后续选择

只读CPU量化实际监督的episode进度、50horizon尾部padding、前5动作权重与夹爪事件覆盖；结合真实数据轴解释，不把episode进度当精确语义阶段。历史复核聚焦专家/clone、G1/G2与接口oracle，防止把不同监督/初始化的正结果当当前模块保证。

B1及上述量化结束后，按结果登记有信息量的下一项：实际10-step采样与FM的差异、冻结接口局部可达性、或train任务诊断闭环。避免重复已有functional assignment和几何敏感性，不默认全面重训或盲扫参数。后续具体执行和原件由本报告及progress记录。

## 5. 执行与结果

协议已登记，B1尚未启动。400 train96已完整49/96、breadth20，全部配对/六worker退出通过；原件`runs/analysis/horizon_relation_writer_20260908/k1_frame_contextual/segment200_400/train96_step400/completed_summary.json`。400 validation尚待结束。

# EMBER task plan

## 当前目标（2026-09-20，Owner授权单次覆盖重训）

保留后继论文实验计划，先完成新覆盖协议的一次fresh EMBER与MT-BC训练，得到唯一选定模型。

1. **完成：数据与执行合同。** 审计新24/8/8与12辅助任务操作覆盖、等价泄漏、canonical数据；锁定等权采样和动态早停。
2. **完成：统一协议入口。** 训练、物化、评测共用显式协议，完整恢复与增量事件计划，补必要接口验证。
3. **进行中：一次fresh正式训练/方法。** Profile已完成；Source复用，Writer从200节点恢复并续训，MT-BC首段与Source新Validation运行中。两节点合规调度；固定完整Validation间隔，独立早停与最佳checkpoint。
4. **条件执行：冻结性能及视频对照。** 新Test paired400相对MT-BC至少+40，视频内容/顺序证据与same-task-other无明显问题；否则停止商量。
5. **条件执行：原后继计划。** 独立演示选点的单support FT及必要三support复核、两臂task-local RL、合格外部比较，交付图文报告。

## 边界

只各一条fresh轨迹，不做k折、额外seed、Unpaired重训；不改架构loss，不合并Validation再训练。
旧协议及负结果保留；新协议是受旧结果启发的受控泛化实验，不称全项目盲测。
充分使用双节点可用及安全共驻GPU，遵守总卡上限；完整阶段结束后读结果，不反复轮询。
维护main并推送，formal从clean pushed detached runtime运行，及时清理确认无用资产，保留正式证据。

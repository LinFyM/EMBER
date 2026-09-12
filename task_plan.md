# EMBER task plan

## 当前goal：有益视频特异性先行，再提升绝对性能

Owner授权依据综合正负证据自主高效推进理论、设计与实验，取得正确视频在唯一完整LoRA中的可重复闭环增量，
并验证跨同task视频、初始化、相邻checkpoint和固定validation迁移。暂不要求145/400；当前goal未完成。

Owner已明确要求重新设置自主推进goal并已激活。依证据自主承担理论分析、方法修正与验证；
先完成当前队列，负结果后优先审计冻结正例复核的判别力与合同，再决定执行。整体目标未达时不标记完成。

**当前active design：**[冻结正例正确收益复核](docs/frozen_positive_replication_audit.md#6-条件性正确收益复核合同新outcome产生前登记)§6。

### 当前执行计划

1. **时间对齐实验已完整关闭：**两臂fresh200及8面板／1,984rows全部完成。train100/200为44/51对36/53；
   validation为61/72对65/81。训练侧早期优势未保持，validation两节点有序差额均负，无资格候选；不续训或扫描。
2. **已激活预注册冻结复核：**固定旧611770d1 ordered200与frame_set200，24task×4video×8state，
   两模型各768加source192，共1,728rows。仅复用原LoRA／运行面，新映射不覆盖原件；先完成live资源准入再启动。
3. 完整报告task/suite、四video、source、R/G/L、churn、换视频成功集合；source只计192个实际outcome，
   使用预定的固定task交叉、多层交叉、task均值三种20,000次配对bootstrap，不使用IID行重采样。
4. 按预定下界、suite及四video方向条件判定跨条件支持；不确定／负向／异质结果均遵守一次预算，不自动扩大面板。
   阳性也不等于顺序因果、相邻保持或validation迁移；后续修正须先提出实质机制证据、历史差异与可失败预测。
5. 整体goal保持：可重复有益视频增量、跨视频／初始化／相邻及固定validation迁移，并在合法冻结后确认最终controls。
   当前未选checkpoint，不启动顺序干预、Test或RL；不会把本次固定模型诊断当作正式方法选择。

用卡遵循Owner上限：两节点合计最多8张，空闲卡总数不超过10张时最多6张；训练与全部评测共享额度。
formal来自clean pushed frozen commit，后续新增长重新检查独立quota与两节点实时资源。

## 已完成历史与最终口径


1. **已完成：具体设计与实现。** 保留T×L到长程过程表示，明确完整H首次读取；实现辅助真实FM和分组cotangent。
   复用官方source/data/完整LoRA出口/evaluator，退役旧活动Writer路径；旧结果留在Git与formal artifacts。
2. **已完成：**验证梯度分配、identity启动、teacher墙、query/noise配对和checkpoint身份；真实长视频profile决定物理batch与段长。
3. **已完成：**主方案、纯FM首段与匹配frame_set共40面板、9,920闭环rows；所有节点尚未获得可信有序增益。
4. **已完成有界200/300窗口；目标未达。** main与frame_set均76,800queries，后段两者均validation退化，原训练已结束。
   **已完成：**固定表示读出64epochs诊断；留出FM改善但所有train24任务仍落后原LoRA学生，不形成部署资格。
   **已全部完成：**active design第8节去蒸馏fresh200比较；100训练42/38、validation57/61；200训练55/55、validation72/67。
   训练获取提高，四项validation均低于原main，相邻及换视频保持未获可信改善；不延长或自动补rho0无序训练。
   **已完成：**第9节原生中层读出32epochs；held .13979对旧头 .14028，差额CI跨0，仍24/24落后LoRA学生 .11000。
   **已完成：**第10节teacher侧VL Meta fresh200及8面板；validation100=61/61、200=62/64，未形成可信增益与保持，不延长或自动补frame_set。
   **已完成：**第11节单独移除辅助表示FM；fresh200、800条件/51,200queries及8面板匹配。validation100=60/55、200=64/54，未形成可信迁移收益。
   当前候选关闭，转入过程获取的综合机制判断；不由本项单个涨跌自动触发下一轮模块修改。
   不延长读出probe或扫描层位；所有旧Writer/probe冻结，新干预从fresh开始且不继承probe。
5. 达到设计登记的正确视频收益、换视频/相邻保持及迁移后，冻结方法和选点，完成视频内容/顺序因果确认，才完成当前goal。
6. 下一阶段另以保持视频收益并提升绝对性能为goal，恢复长期145/400及完整稳定/breadth资格。

每轮只根据真正检验的因素修正；不以辅助loss、wrong退化或单点峰值完成goal。不启动95-task、不恢复旧候选；
新实验live资源与精确命令写run contract，阶段状态写progress，跨轮结论写findings，历史结果写research_history。

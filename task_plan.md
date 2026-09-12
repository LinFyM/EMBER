# EMBER task plan

## 当前goal：有益视频特异性先行，再提升绝对性能

Owner授权依据综合正负证据自主高效推进理论、设计与实验，取得正确视频在唯一完整LoRA中的可重复闭环增量，
并验证跨同task视频、初始化、相邻checkpoint和固定validation迁移。暂不要求145/400；当前goal未完成。

Owner已明确要求重新设置自主推进goal并已激活。依证据自主承担理论分析、方法修正与验证；
当前时间对齐队列与冻结正例复核已完整结束，转入基于全部证据的机制判断。整体目标未达时不标记完成。

**当前无active实验design。** 时间对齐与冻结正例复核均已关闭；当前阶段为机制理论裁决，整体goal保持。

### 当前执行计划

1. **时间对齐已完成：**fresh200两臂及8面板／1,984rows。train100/200为44/51对36/53，
   validation为61/72对65/81；早期有序优势未保持，validation无资格，不续训或扫描。
2. **冻结正例复核已完成：**固定旧200双模型，24task×4video×8state加source，共1,728rows；
   有序359/768、静态361/768、source32/192。三种有序−静态95%CI均跨零，四video净额0/+4/+2/−8。
   新条件下有序对source能力仍在，原大幅有序优势未复现；按预注册上限关闭，不增加训练或面板。
3. **理论边界复核已形成：**[视频信息与可识别性](docs/video_information_identifiability.md)核对了全帧frame_set、
   train24当前状态目标和原受益9task的新条件净额−1/288；区分独立训练臂比较与固定模型因果干预。
   跨episode FM允许同task视频无关最优解，但不证明有限共享模型不可能从视频学习迁移知识。
   **当前理论任务：**依据文中竞争预测，明确下一机制究竟增加何种影响闭环的知识及其合法判别方式；
   不预设E充分、Compiler唯一失败或共享竞争已被隔离，不用终态目标替换有益过程要求。
4. 只有新增机制证据与相对近邻历史的实质区别成立，才登记下一项最小、合法判别实验的预算与停止分支；
   不因CI跨零、局部正数、总分高于source或代码准备好而自动启动完整Writer、参数扫描或最终controls。
5. 整体目标仍是可重复有益视频增量及跨视频／初始化／相邻和固定validation迁移，在合法冻结后确认最终controls。
   当前未选checkpoint，不使用Test、held梯度或RL；局部诊断不构成方法选择或goal完成。

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

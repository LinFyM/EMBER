# EMBER task plan

## 当前goal：有益视频特异性先行，再提升绝对性能

Owner 2026-09-11授权自主高效实施，2026-09-12再次明确核心科学精神内的理论／架构修正自由度。
结合全部历史重新设计，不恢复v5.2底座；暂不要求145/400，但正确视频闭环收益、跨视频/初态/相邻保持及validation迁移不可替代。
用卡遵循Owner上限：两节点合计最多8张，空闲卡总数不超过10张时最多6张；训练与全部评测共享额度。

**当前无active design。** [Local Action Grounded](docs/local_action_grounded_writer_design.md)的fresh两臂200步、
两类学习诊断与16闭环面板全部完成，共3,968rows。validation100有序两视频均劣于匹配frame_set，200差额0/+1且区间跨零；
两臂后段均退化，局部动作FM优势未成为有益任务过程。当前配方关闭，整体goal未完成；详见findings§69及progress。

### 当前执行计划

1. **已完成并封存：**局部post-action配对、共享encoder/native信用重放、信息墙和真实profile；两臂各51,200主queries，
   完整checkpoint、两类动作留出、全部16登记面板及逐task/suite、相邻/换视频配对统计保留。所有进程已退出。
2. **停止当前配方投入：**不续训、扩大局部头或扫描loss/LR/rank/seed；未获资格，不启动pureFM第三臂或最终controls。
   不将局部头优势当作共享E充分、Compiler唯一失效或goal完成的证据。
3. **下一工作：**在综合正负证据上推导实质不同的过程表示与学习机制，明确完整输入到唯一LoRA的因果路径、
   相对近等价历史新增什么，以及不同结果如何改变后续投入；再登记唯一active design并按既有授权实现、profile和有界验证。
   任务覆盖和外部预训练先验已有边界核查，不能仅凭本轮负结果自动启动；不恢复95-task、旧候选或改低行为门槛。
4. 达到登记的正确视频收益、换视频/相邻保持及迁移后，冻结方法与single checkpoint，完成最终视频内容/顺序因果确认，才完成当前goal。

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

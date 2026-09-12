# EMBER task plan

## 当前goal：有益视频特异性先行，再提升绝对性能

Owner授权自主高效推进新视频表示与执行状态条件化功能监督，取得正确视频在唯一完整LoRA中的可重复闭环增量，
并验证跨同task视频、初始化、相邻checkpoint和固定validation迁移。暂不要求145/400；当前goal未完成。

**当前无active design。** Pretrained Video Grounded的fresh200与8面板／1,984rows全部完成并关闭。
train100有序／静态41/41，200为52/40且配对CI为正；validation100为53/48但下界0，200为33/48且CI为负。
有序Long5→0，静态48→48却有52次成功状态变动；没有稳定迁移或跨视频资格。完整证据见progress及findings§70。

### 当前执行计划

1. **已完成：**完整有界学习、实际曝光核验、两节点train96／validation400及成功集合裁决。
2. **已关闭：**不续训、不扫局部参数、不触发other／image／最终controls；保留formal evidence，全部进程已退出。
3. **下一步机制判断：**同时解释train200局部正例、validation反转、相邻漂移和历史强正例；核对最近等价历史，
   在信息路径、参数作用或学习关系上提出实质可辨别的新机制，避免重命名已有共享prior、辅助loss或任务扩展。
4. 登记有明确竞争预测、停止条件和行为证据的新active design后，自主完成实现及验证；不默认启动另一轮同图小改。
5. 只有正确视频收益、跨视频／初态／相邻保持与迁移均成立，且selected后的因果controls通过，才完成当前goal。

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

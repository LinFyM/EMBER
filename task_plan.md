# EMBER task plan

## 当前goal：有益视频特异性先行，再提升绝对性能

Owner 2026-09-11已授权自主高效实施，2026-09-12再次明确核心科学精神内的理论／架构修正自由度。
旧[Video Functional设计](docs/video_functional_writer_design.md)已关闭；唯一active design为[Local Action Grounded Writer](docs/local_action_grounded_writer_design.md)。
结合全部历史重新设计，不恢复v5.2底座；暂不要求145/400，但闭环真实收益、跨视频/初态/相邻保持及validation迁移不可替代。

用卡遵循Owner最新上限：两节点合计最多8张，空闲卡总数不超过10张时最多6张；训练与全部评测共享额度。

Owner 2026-09-12要求改变连续负结果后的推进方式：停止自动串联局部消融。第11节200步及100/200全部面板已完成。
按findings§62/65，当前主候选在训练侧亦未形成可重复有序增量；不再默认先修保持、扩大辅助读出或改loss。
下一设计应正面解释并检验过程获取，说明近等价历史、竞争预测及各结果的停止/保留决策；未经此分析不启动新候选。
外部视频预训练仅为待分析的知识来源，不自动采纳。这不是等待额外人工审批，也不改变既有评测门槛。

已完成[过程获取的监督关系复核](docs/video_process_acquisition_analysis.md)：推荐具体审视action训练池的
同episode局部观察—动作配对，区别于已执行的跨episode功能／phase监督。该新支路的跨episode
合同适用范围现由Owner最新授权覆盖；数据流、停止条件及信息墙已登记为active design，实现与真实profile已通过，ordered正式学习已启动，frame_set按同拓扑随后接续。

### 当前执行计划

1. 已完成隔离实现局部post-action数据、动作FM头及共享encoder/native重放；旧执行读出与蒸馏已退役。
2. 已通过信息墙、索引、无序参照、梯度、采样恢复及checkpoint定向检查；93帧真实profile选frame8/policy8。
3. 进行中：从clean pushed frozen 5f4f440c开展ordered/frame_set匹配学习，预登记100/200、两类动作留出及train96/validation400两臂。
4. 按新设计分开裁决局部获取、LoRA收益与迁移；只有可信正结果才进入无辅助归因参照与最终冻结后的sealed controls。

## 计划与完成口径

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

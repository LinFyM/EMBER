# EMBER task plan

## 当前goal：有益视频特异性先行，再提升绝对性能

Owner授权依据综合正负证据自主高效推进理论、设计与实验，取得正确视频在唯一完整LoRA中的可重复闭环增量，
并验证跨同task视频、初始化、相邻checkpoint和固定validation迁移。暂不要求145/400；当前goal未完成。

Owner已明确要求重新设置自主推进goal并已激活。依证据自主承担理论分析、方法修正与验证；
先完成当前队列，负结果后优先审计冻结正例复核的判别力与合同，再决定执行。整体目标未达时不标记完成。

**当前active design：**[Execution-Aligned Video Writer](docs/execution_aligned_writer_design.md)。
前轮8面板已关闭，训练200局部有序正例与validation反转完整保留。综合审计见findings§71。

### 当前执行计划

1. **已完成机制选择：**排除重命名的endpoint／专家蒸馏；依据post-action生产时序校正主执行监督。
2. **实现与验证已完成：**唯一Writer强制obs[i]对应actions[i+1:]；训练／动作留出共用，末query排除，无teacher截断。
   验证真实下一状态转移、padding、显式旧消费者口径及新训练／物化身份；历史checkpoint只用原冻结runtime。
3. **正式学习已完成：**2ecf1770 clean pushed冻结树，两臂各world3完成fresh200，各800条件／51,200queries；100/200完整checkpoint与全800条件配对已验证，两节点动作留出有序差额CI均跨零，不能替代闭环。
4. **睡前自主执行已启动：**后台先完成100/200两节点train96与validation400 correct共8面板／1,984rows，并自动汇总。按新标签合同匹配ordered／frame_set；有相邻正向候选才评估后续other与强静态参照资格。
5. 保持全部task/suite、breadth、R/G/L、churn及相邻/换视频资格；选点冻结后的sealed内容／顺序controls通过才能完成goal。
   时间对应的正确性不是有益过程的性能证据；无资格则关闭有界学习，不扫描offset或局部参数。

本轮未通过则关闭，不追加完整Writer训练或参数扫描。下一项优先准备旧52/96对40/96冻结正例复核：
先确定跨视频／初始化面板、统计辨别力、四臂合同及停止分支；尚不自动启动新rollout或改变最终controls用途。
夜间后台队列范围止于本轮完整correct证据与初步裁决，异常保留日志，整体目标不因队列完成而完成。

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

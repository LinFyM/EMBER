# EMBER task plan

## 当前goal：有益视频特异性先行，再提升绝对性能

Owner 2026-09-11已授权自主高效实施。以[active design](docs/video_functional_writer_design.md)为唯一运行合同，
结合全部历史重新设计，不恢复v5.2底座；暂不要求145/400，但闭环真实收益、跨视频/初态/相邻保持及validation迁移不可替代。

用卡遵循Owner最新上限：两节点合计最多8张，空闲卡总数不超过10张时最多6张；训练与全部评测共享额度。

## 计划与完成口径

1. **已完成：具体设计与实现。** 保留T×L到长程过程表示，明确完整H首次读取；实现辅助真实FM和分组cotangent。
   复用官方source/data/完整LoRA出口/evaluator，退役旧活动Writer路径；旧结果留在Git与formal artifacts。
2. **已完成：**验证梯度分配、identity启动、teacher墙、query/noise配对和checkpoint身份；真实长视频profile决定物理batch与段长。
3. **已完成：**主方案、纯FM首段与匹配frame_set共40面板、9,920闭环rows；所有节点尚未获得可信有序增益。
4. **已完成有界200/300窗口；目标未达。** main与frame_set均76,800queries，后段两者均validation退化，原训练已结束。
   **已完成：**固定表示读出64epochs诊断；留出FM改善但所有train24任务仍落后原LoRA学生，不形成部署资格。
   **已全部完成：**active design第8节去蒸馏fresh200比较；100训练42/38、validation57/61；200训练55/55、validation72/67。
   训练获取提高，四项validation均低于原main，相邻及换视频保持未获可信改善；不延长或自动补rho0无序训练。
   **当前：**第9节已登记固定表示的原生中层功能读取诊断；实现及smoke已通过，正在运行32epochs有界拟合，区分旧动作头与原生控制消费函数类。
   原Writer冻结；probe不会回写、直接部署或自动触发新Writer训练。
5. 达到设计登记的正确视频收益、换视频/相邻保持及迁移后，冻结方法和选点，完成视频内容/顺序因果确认，才完成当前goal。
6. 下一阶段另以保持视频收益并提升绝对性能为goal，恢复长期145/400及完整稳定/breadth资格。

每轮只根据真正检验的因素修正；不以辅助loss、wrong退化或单点峰值完成goal。不启动95-task、不恢复旧候选；
新实验live资源与精确命令写run contract，阶段状态写progress，跨轮结论写findings，历史结果写research_history。

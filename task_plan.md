# EMBER task plan

## 当前goal：有益视频特异性先行，再提升绝对性能

Owner 2026-09-11授权自主高效实施，2026-09-12再次明确核心科学精神内的理论／架构修正自由度。
结合全部历史重新设计，不恢复v5.2底座；暂不要求145/400，但正确视频闭环收益、跨视频/初态/相邻保持及validation迁移不可替代。
用卡遵循Owner上限：两节点合计最多8张，空闲卡总数不超过10张时最多6张；训练与全部评测共享额度。

**当前active design：**[Pretrained Video Grounded Writer](docs/pretrained_video_grounded_writer_design.md)。
冻结V-JEPA2.1过去四帧dense表示，经task-conditioned Value读取进入完整native H及唯一LoRA；仅跨episode主FM。
实现与真实profile已通过，两臂fresh200均完整结束且实际曝光匹配；有序train100/200为41/52（各96），有序validation100为53/400。静态100训练完整41/96，与有序100配对差额CI跨零；全部LoRA库已sealed，剩余静态200训练和三项validation评测并行进行，尚无有益过程或迁移结论。Local Action Grounded已关闭，其完整证据保留于findings§69与progress。

### 当前执行计划

1. **已完成设计登记：**综合功能信用、局部监督、覆盖扩展负证据与v5.2正例，检验预训练视频知识来源。
   主比较为同一视频编码器的ordered／逐帧重复静态；仅出现完整正向资格时，追加原生image全帧静态参照确认强度。
2. **实现已集成：**唯一入口接入冻结先验与learned task grounding，保留完整H、source冻结、teacher信息墙和跨episode主FM。
   退役局部动作头／loss／数据支路／专属诊断，复用现有trainer、重放、checkpoint与动态evaluator。
3. **正式学习已完成：**实现／单份资产／最长teacher profile完成；学习前固定100/200与每臂51,200主FM queries。
   ordered与frame_set在clean pushed frozen `611770d1`上各完成world4 fresh200；实际800条件／51,200queries及18个采样字段匹配。
4. **行为裁决：**先两节点train96和validation400 correct；有正向候选再补other与已登记强静态检查。
   报告绝对能力、source、task/suite、breadth、成功集合及相邻／换视频保持；明确坏结果不扫参数或追加局部补救。
5. 全部资格成立后冻结方法与single checkpoint，完成sealed内容／顺序controls；当前goal仍未完成。

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

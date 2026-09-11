# EMBER progress

## 当前状态：Video Functional Writer设计与实现（2026-09-11）

Owner最新明确“开始推进”，给予足够自由度并要求高效率。已创建活动goal：先实现有益视频特异性及validation迁移，
暂不要求145/400；达到后再保持视频收益提升性能。此前暂停只继续适用于旧C/无变化参照、95-task等历史路线，
不阻止本次新设计的实现、profile、正式学习、匹配对照和证据驱动修正。

**唯一active design：[Video Functional Writer](docs/video_functional_writer_design.md)。**
选择时间×任务token有序表示＋执行query条件化辅助真实FM；完整LoRA从第一步真实FM，
仅Compiler/native D接受归一化蒸馏，encoder的真实LoRA FM系数固定1。部署仍只有唯一完整38-target A/B。

### 当前工作

- 历史证据包已在main d1b62535交付；专家已确认实际读取，最终三轮原文保存在该包并由active design索引。
- 主实现工作树codex/video-functional；独立视频编码及评测适配子任务已完成，两个提交已集成至主实现树。
- 主任务负责原生FM执行读出、辅助头、分组VJP、runtime/config/schema和集成；旧未提交95-task工作树不动。
- 已核对/data1独立quota（786.6GiB/1TiB）及共享空间，代码工作树新增不到1GiB；训练前另核定输出峰值。
- 新模型、分组信用、runtime/config/schema及评测适配已实现；旧图运行路径退役。完整CPU测试378 passed。
- 最长合法训练视频93frames真实profile通过：原生/捕获FM均.106989；rho=.25时batch4/8为3.10/3.20queries/s、峰值25.32/37.84GiB；batch16 OOM。未保存profile权重。
- 首段在看到正式学习结果前登记50/100更新；主方案拟用gpu01:4,5,6，microbatch8；pureFM拟用gpu02:3,6，microbatch8,4，后者给低负荷服务留显存。正式训练尚未启动。
- /data1实时独立quota使用787.0GiB/1TiB，runs实测636GiB；两配方首段checkpoint及全配对物化预算40GiB，预计峰值827GiB，不复制基础资产。
- 训练与评测均从clean pushed detached commit；不把profile权重作为formal起点。

## 暂停时点的已完成实验与未完成范围

无变化参照候选从冻结`64eba75bcf316a1f6792ab8ddf5eb1aaefded378`完成fresh200更新/51200queries，
用时3327.11秒，100/200完整checkpoint保留。唯一主要变化为C局部更新采用同gap/角色/窗口的无变化参照，
保留train24、每task两个K1条件各32queries、普通正样本FM、source冻结；没有RL、错误视频惩罚或任务覆盖变化。

训练task独立留出动作FM由.151451降至.109323，24/24任务改善。固定teacher46、states32–35的train100→200
correct为36→45，各96；相邻保留22/新增23/丢失14，churn37/J=.37288。200正确/换视频/同suite错/跨suite错/乱序/静态
为45/42/41/37/41/40，各视频差额的task-bootstrap95%区间均含0。200 validation完整为34/400，source47、旧C20050；
Spatial/Object/Goal/Long为4/24/5/1，breadth5。初步训练条件分化未成为可信视频机制收益或迁移改善。

100 validation在Owner暂停时终止，没有可报告的strict400成绩；300/400未启动。首段不能证明候选收敛或整个机制被否定。
`runs/analysis/video_change_reference_20260911/owner_pause.json`记录定向停止及checkpoint保留；本节不是实时GPU进程检查。

此前完整C冻结诊断共1472条新闭环及14336次配对动作预测：C200 correct42、两错46/46、乱序45、静态49、S51、source15，
各96；固定语义S后correct/static过程同为15/32。静态首帧仍产生正常视频.692–.935倍中心化P4，支持一个具体表示性质，
尚非全局根因。无变化参照修正已在真实native输入上削弱该性质，但行为收益未被证明。

| 暂停前路线 | validation /400 | 训练任务观察与边界 |
| --- | --- | --- |
| R，off图+两个独立K1条件 | 100/200/300/400=37/83/63/85 | train200/400=46/56，各96；同query预算旧off400=126/56，未获迁移收益 |
| C，语义条件化过程消费 | 100/200=43/50 | 轮换held视频train96=49；固定teacher46机制面板42，不能混用 |
| S，无序帧集合参照 | 100/200=59/45 | 轮换held视频train96=52；固定teacher46机制面板51；不是单帧static模型 |
| off7续训 | 400/500/600=126/73/54 | 轮换held视频train96=56→64；训练获取与未见task能力分离 |

R/C/S及两项续训共4480条闭环，原件根`runs/analysis/video_consumption_20260911/`；C诊断根
`runs/analysis/video_mechanism_20260911/`；无变化参照根`runs/analysis/video_change_reference_20260911/`。
远程副本统一由新专家材料的index定位，`runs/`路径本身不表示已经发布。

95-task提案未启动GPU实验，其两个未提交worktree原样保留，不集成或清理。精确历史和旧合同由Git与
[research_history](docs/research_history.md)保留。后续执行只能依据Owner新授权及新登记，不能沿本文历史段恢复。

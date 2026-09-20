# EMBER task plan

## 当前目标与裁决顺序（2026-09-20）

在已冻结的[覆盖重训合同](docs/coverage_retraining_design.md)下，各完成一条fresh EMBER Writer和MT-BC轨迹，按完整Validation选出各自唯一checkpoint。资格通过后依次做固定模型Test、[单演示FT与task-local RL](docs/paper_experiments_design.md)、合格的外部比较，交付图文报告及原始统计材料。科学门槛失败或结论无法定夺时停止下游，先交付当阶段证据并询问Owner。实际进程、exit与产物只在[progress](progress.md)记录；本文件保留执行顺序和通过条件。

Owner提供的专家原始对话解释了两项已采纳决策：held所需对象与基本操作须有真实训练支持；训练终点由固定完整Validation的趋势决定。后续讨论撤回了五折、额外seed和固定2000/600硬终点。当前协议、审计边界及早停纯函数以仓库active design为准，不从较早对话恢复建议。

| 阶段 | 执行与交付证据 | 进入下一阶段的条件 |
| --- | --- | --- |
| 1. 协议与训练入口：已完成 | 唯一24/8/8＋12协议、BDDL/HDF5 metadata覆盖与等价泄漏审计、36任务采样及两方法fresh配置；保留Long9、Goal6、Long8、Spatial9等未完整覆盖的边界。 | 协议在fresh训练前冻结；不得以对象仅出现在画面或初态已满足谓词冒充操作监督。 |
| 2. 训练与完整Validation：进行中 | 复用Source-71；Writer每200更新、MT-BC每50更新完成correct400，保留每节点raw rows、checkpoint、sampler/optimizer恢复状态及早停历史。后台控制器负责分段和早停；只在完整节点或故障退出后处理。 | 两条轨迹分别按登记的持续下降或平台规则结束。非零退出先按工程层诊断和原合同恢复；不能把资源截断写成科学收敛。 |
| 3. 唯一选点与Validation资格 | 审核完整节点、任务/suite曲线及相邻success-set；MT-BC取correct最高且同分最早。Writer先对所有correct并列最佳补other400，按other最高、再按最早选定。冻结checkpoint及视频映射，完成selected correct/other/wrong/shuffle/reverse配对400与Source参照。 | correct和other的有效能力、相邻稳定性及视频内容/顺序证据可解释；other明显退化或特异性不清楚则停止询问，不重选或调参。 |
| 4. Test主表与硬门槛 | 用冻结模型完成Source、MT-BC、EMBER correct各400；检查exit、manifest、固定state-video映射和raw rows，报告per-task、per-suite、breadth、R/G/L、churn及task-cluster bootstrap。 | EMBER成功数减MT-BC成功数至少40/400，才进入Test视频对照。若不足，保留正式负结果并停止下游。 |
| 5. Test视频对照与零交互封存 | 门槛通过后完成EMBER other/wrong/shuffled/reversed各400，按同task、state、RNG与video ordinal配对；冻结零交互模型、表格及成本。 | correct/other相对错误内容和时间干预的证据清楚；优势消失或区间不明确则停止讨论，不依据Test换模型。 |
| 6. 单演示FT比较 | 先完成merged MT-BC底座＋fresh rank16接口的功能等价检查和Train配方资格；在held适应前锁定每task support、独立选点集、固定查询、预算与节点。默认Validation/Test全16任务，各用相同一条support比较一次编译的EMBER与监督FT，并各评50个独立初态。 | 完整报告每task结果、FT曲线和动作标签成本。EMBER明显落后时按登记规则追加两条独立support，不融合或挑优；持续落后或混杂则停止询问。 |
| 7. Task-local RL | 按固定RLinf版本审计实现Source＋生成A/B和merged MT-BC＋fresh A/B两臂；先在Train任务验证零步行为、38-target trainables、flow10/horizon50/replan5、chain logprob、成功即停、真实control-step预算及恢复语义。之后预注册held预算、节点与初态分隔，再做正式比较。 | 每个预定预算点的任务等权平均曲线按Owner目标比较；不学习、持续退化或结论不理想时停止下一段并报告，不删除不利节点。 |
| 8. 外部比较与交付 | 先核对WIZARD、ViVLA原论文、官方实现、专家标签及可运行性，再按可比协议实施合格基线；保留原方法能力、成本和不能复现的具体原因。汇总新旧协议区别、既有Test使用记录、Validation选点、全套视频对照、FT/RL结果及局限、失败案例、编译/训练/评测成本、CSV/JSON和可复用图表源文件。 | 报告区分成立、失败与未执行的结论；每项数字能追溯到完整正式产物。 |

## 边界

两方法各只fresh训练一次；不做k折、额外seed、Unpaired、双相机修补、合并Validation重训或因Test改划分/配方。Source及normalization不重训。旧Test的78/74/82与新协议分开；新划分受旧Test启发，不称全研究过程盲测。

Writer部署只读exact language和action-hidden有序视频，一次生成完整38-target A/B；主监督跨episode。Writer训练用demo0..45，MT-BC用0..49，报告时披露。正式选择只用single-checkpoint完整Validation400；shuffled/reversed只作冻结诊断。

每次真正launch/resume前查双节点实时GPU与data0/data1独立quota、预计峰值；总物理卡和单节点上限按AGENTS执行。正式训练/评测来自clean pushed detached runtime，不干扰其他用户进程。保留运行依赖worktree，结束且确认无引用后再清理。

阶段切换时用已登记profile和完整阶段用时检查训练更新/秒、LoRA/秒、评测rows/秒、GPU利用率及峰值显存；优先填满合同允许的合格物理卡和安全共驻余量。MT-BC在完整checkpoint边界可按真实两节点卡况迁移到更有吞吐的单节点、调整物理卡数；必须保持同一条576查询轨迹和optimizer/scheduler状态，并登记换拓扑RNG边界。训练用任务交错分片检查rank等待，评测用动态队列防止长任务拖住全队；慢条件仍完整执行。仅针对实测瓶颈优化物理batch、评测副本/动态队列或kernel，并验证逻辑任务权重、配对映射与数值稳定性；活跃frozen runtime不原地改写。

结果不达标是科学负结果，工程失败按层定位并在原合同内恢复。只删确认可再生成且无后继依赖的载荷；checkpoint、raw rows、manifest与旧正式证据保留。修改集成`main`并推送，文档按实际阶段更新，不用频繁轮询代替现有控制器。

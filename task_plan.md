# EMBER task plan

## 当前 active design：Writer 因果路径与辅助梯度诊断（2026-09-22，preflight 已完成，待 launch）

Owner 要求按专家后续意见完成有界分析实验；合同见
[Writer 因果路径与辅助梯度诊断](docs/writer_causal_diagnostics_20260922.md)。它只在冻结 O1200/C600/C1200、M300、
Source 上运行 D1 Core/Procedure 路径消融、D2 主/辅助梯度与 AdamW 虚拟更新，并仅在预登记120分钟总 GPU 预算仍能
完整覆盖时做 C600 的 D3 三臂各18步微训练。禁止新 fresh 训练、Validation/Test、FT、RL、外部比较、自动选点、自动
调参或无限延长。实现、测试、clean pushed detached runtime、GPU/quota preflight 与正式原始行齐全后才开始；D3 不会
按部分分数选择性执行。首次 live preflight 已登记 gpu01:0 与低负载共驻的 gpu02:1--3；不预留空卡，120分钟绝对截止
独立于分数，D3 仅在 D1/D2 都完整成功且保留至少65分钟时整体启动。

首次 launch 的四个 D1 worker 均在首次数据 authority 校验前后 exit=1，未产生任何性能或闭环行；已确认是 detached
runtime 错当 asset root 的工程路径错误。该失败原件保留，修复只令 runner 通过 canonical 只读 asset root 读取 authorities/
assets，重新从 clean pushed runtime 执行同一合同；不得把此类 pre-probe exit 记为 D1 科学结果。

## 当前状态：36任务 Writer 辅助 episode 配对 fresh 对照已完成（2026-09-22）

本轮[辅助配对合同](docs/writer_auxiliary_pairing_design.md)已执行并封存，当前**没有 active design 或获授权的后继
训练**。study根为`/data0/user/ymdai/ember_runs/coverage_retraining_cross_episode_aux_20260922`；完整报告为
`analysis/report/pairing_report.md`。候选只把7个辅助查询从`same_video`改为同任务另一episode，固定保留
`tau=1`、前5步和权重`1/3`。

clean pushed detached runtime `64947492`在gpu02:0--3、world4/frame8完成1200更新和六个correct Validation400。
controller exit为0；每个正式面板均复核400条唯一task/state行、materialization manifest、固定配对映射和12个worker
全零退出。correct曲线为`200..1200 = 151, 120, 154, 137, 129, 124`；原早停在1200以`sustained_decline`触发，
所以没有post-stop extension。按`correct → other → earliest`选中step600的154/400；same-task-other=160、
cross-suite-wrong=153，二者都未参与选点。154比同节点same-video115高39、比原Writer最佳117高37，但仍低冻结
MT-BC300=155一分。

cross-episode在六个匹配节点都超过same-video（差`+41,+28,+39,+50,+12,+44`），但相邻成功集不稳定：200→400、
400→600、600→800的R/G/L分别为84/36/67、80/74/40、101/36/53。选中点相对same-video的R/G/L=78/76/37，
Spatial净+45而Goal/Object各净−8/−7；不能把绝对增益写成普遍保持或全面恢复。由于154落在118..155的预登记分支，
仅完成other/wrong，未执行shuffle/reverse；正确视频相对wrong只+1，other反而+6，不能声称恢复视频内容或顺序特异性。

[PR #3](https://github.com/LinFyM/EMBER/pull/3)已合入`main`，其状态/选点重复路径已删除并复用现有controller。真实
1800步、7200条件事件重放以及全部完成后的正式完整性检查均通过。该结果支持“在本合同下跨episode**辅助配对**改善了
same-video参照”，不证明端点前缀辅助项整体、`tau=1`、前5步或`1/3`必要；纯跨episode随机flow-time完整50-horizon
主FM仍是未执行的独立假设。Test、FT、RL、外部比较和任何LR/seed/rank/head补救均未启动；等待Owner决定下一设计。

## 已完成目标：Writer输出空间与code投影诊断（2026-09-21）

Owner授权按专家最新收敛方案执行[输出空间投影诊断合同](docs/writer_output_space_projection_design.md)。本轮只做
A0三checkpoint CPU空间检查、A1十六次无梯度视频编译，以及SELF/NEWSPACE/SHRINK三臂共48条冻结闭环；
复用既有O1200原始16条参照。禁止反向传播、训练、Test、FT、RL、完整E2及自动拆head后继。

执行顺序：先登记资产和实现可复算投影/code接口；完成CPU与最小合成检查后提交推送并创建clean detached runtime；
GPU launch前实时检查两节点与quota；A1按模型并行、A2按三臂并行使用安全可用GPU；只在完整exit后汇总。
最终交付head谱、code几何、逐层有效更新误差、48条原始闭环、O1200配对统计、SELF首replan数值检查和图文报告，
随后停止等待Owner/专家裁决。

A0/A1/A2已经按合同完整结束。O1200/SELF/NEWSPACE/SHRINK在固定16条件上为11/11/11/9；NEWSPACE
没有造成净成功下降，因此不支持“新B空间删除旧有效方向”的解释。N1000 code与实际B/BA更趋同，只登记为
系数映射候选定位。当前没有获授权的后继实验，不自动恢复Test、FT、RL、完整E2或split-head训练。

## 已完成阶段：Writer训练稳定性修订诊断

按Owner转交的专家最后修订执行[Writer稳定性修订诊断](docs/writer_stability_diagnostics_design.md)。复用已完成
E0/E1，取消且不恢复原E2；只从O1200/N1800完整父状态各做高/低固定LR的36步短窗，随后完成每臂36条
compact动作探针、四个held任务各state0..3的闭环及既有16条E1轨迹轻量导出。四臂结果只交付事实证据，
不选新正式模型、不自动续训，也不启动Test、FT、RL或外部比较。

四臂、终点探针、闭环和轨迹导出均已完整exit0，脱敏原始包见
[corrected diagnostics](docs/review_materials/20260921/writer_stability_diagnostics/corrected/README.md)。当前没有运行中
诊断实验；Owner/专家已据此选择正式N1800低LR修复，执行agent仍不把O-H/O-L/N-H/N-L作为正式父状态。

原覆盖重训主线保留如下；当前修复阶段结束前不能自动恢复第4阶段。

## 覆盖重训裁决顺序（2026-09-20，当前下游暂停）

在已冻结的[覆盖重训合同](docs/coverage_retraining_design.md)下，各完成一条fresh EMBER Writer和MT-BC轨迹，按完整Validation选出各自唯一checkpoint。资格通过后依次做固定模型Test、[单演示FT与task-local RL](docs/paper_experiments_design.md)、合格的外部比较，交付图文报告及原始统计材料。科学门槛失败或结论无法定夺时停止下游，先交付当阶段证据并询问Owner。实际进程、exit与产物只在[progress](progress.md)记录；本文件保留执行顺序和通过条件。

Owner提供的专家原始对话解释了两项已采纳决策：held所需对象与基本操作须有真实训练支持；训练终点由固定完整Validation的趋势决定。后续讨论撤回了五折、额外seed和固定2000/600硬终点。当前协议、审计边界及早停纯函数以仓库active design为准，不从较早对话恢复建议。

| 阶段 | 执行与交付证据 | 进入下一阶段的条件 |
| --- | --- | --- |
| 1. 协议与训练入口：已完成 | 唯一24/8/8＋12协议、BDDL/HDF5 metadata覆盖与等价泄漏审计、36任务采样及两方法fresh配置；保留Long9、Goal6、Long8、Spatial9等未完整覆盖的边界。 | 协议在fresh训练前冻结；不得以对象仅出现在画面或初态已满足谓词冒充操作监督。 |
| 2. 训练与完整Validation：两方法已完成 | 复用Source-71；Writer每200更新、MT-BC每50更新完成correct400，保留每节点raw rows、checkpoint、sampler/optimizer恢复状态及早停历史。后台控制器负责分段和早停；只在完整节点或故障退出后处理。 | 两条轨迹分别按登记的持续下降或平台规则结束。非零退出先按工程层诊断和原合同恢复；不能把资源截断写成科学收敛。 |
| 3. 唯一选点与Validation资格：两方法已选；Writer视频内容特异性待Owner裁决 | 审核完整节点、任务/suite曲线及相邻success-set；MT-BC取correct最高且同分最早。Writer先对所有correct并列最佳补other400，按other最高、再按最早选定。冻结checkpoint及视频映射，完成selected correct/other/wrong/shuffle/reverse配对400与Source参照。 | correct和other的有效能力、相邻稳定性及视频内容/顺序证据可解释；other明显退化或特异性不清楚则停止询问，不重选或调参。 |
| 4. Test主表与硬门槛：暂停等待Owner裁决 | 用冻结模型完成Source、MT-BC、EMBER correct各400；检查exit、manifest、固定state-video映射和raw rows，报告per-task、per-suite、breadth、R/G/L、churn及task-cluster bootstrap。 | EMBER成功数减MT-BC成功数至少40/400，才进入Test视频对照。若不足，保留正式负结果并停止下游。 |
| 5. Test视频对照与零交互封存 | 门槛通过后完成EMBER other/wrong/shuffled/reversed各400，按同task、state、RNG与video ordinal配对；冻结零交互模型、表格及成本。 | correct/other相对错误内容和时间干预的证据清楚；优势消失或区间不明确则停止讨论，不依据Test换模型。 |
| 6. 单演示FT比较 | 先完成merged MT-BC底座＋fresh rank16接口的功能等价检查和Train配方资格；在held适应前锁定每task support、独立选点集、固定查询、预算与节点。默认Validation/Test全16任务，各用相同一条support比较一次编译的EMBER与监督FT，并各评50个独立初态。 | 完整报告每task结果、FT曲线和动作标签成本。EMBER明显落后时按登记规则追加两条独立support，不融合或挑优；持续落后或混杂则停止询问。 |
| 7. Task-local RL | 按固定RLinf版本审计实现Source＋生成A/B和merged MT-BC＋fresh A/B两臂；先在Train任务验证零步行为、38-target trainables、flow10/horizon50/replan5、chain logprob、成功即停、真实control-step预算及恢复语义。之后预注册held预算、节点与初态分隔，再做正式比较。 | 每个预定预算点的任务等权平均曲线按Owner目标比较；不学习、持续退化或结论不理想时停止下一段并报告，不删除不利节点。 |
| 8. 外部比较与交付 | 先核对WIZARD、ViVLA原论文、官方实现、专家标签及可运行性，再按可比协议实施合格基线；保留原方法能力、成本和不能复现的具体原因。汇总新旧协议区别、既有Test使用记录、Validation选点、全套视频对照、FT/RL结果及局限、失败案例、编译/训练/评测成本、CSV/JSON和可复用图表源文件。 | 报告区分成立、失败与未执行的结论；每项数字能追溯到完整正式产物。 |

## 边界

两方法各只fresh训练一次；不做k折、额外seed、Unpaired、双相机修补、合并Validation重训或因Test改划分/配方。Source及normalization不重训。旧Test的78/74/82与新协议分开；新划分受旧Test启发，不称全研究过程盲测。

Writer部署只读exact language和action-hidden有序视频，一次生成完整38-target A/B；主监督跨episode。Writer训练用demo0..45，MT-BC用0..49，报告时披露。正式选择只用single-checkpoint完整Validation400；shuffled/reversed只作冻结诊断。

每次真正launch/resume前查双节点实时GPU与data0/data1独立quota、预计峰值；总物理卡和单节点上限按AGENTS执行。正式训练/评测来自clean pushed detached runtime，不干扰其他用户进程。保留运行依赖worktree，结束且确认无引用后再清理。

阶段切换时用已登记profile和完整阶段用时检查训练更新/秒、LoRA/秒、评测rows/秒、GPU利用率及峰值显存；优先填满合同允许的合格物理卡和安全共驻余量。Owner要求后续专门判断当前Writer有无提速必要：上一轮双卡同microbatch的frame chunk8→16使最长视频更新步时约缩短7.7%，本轮是四卡frame8，不能直接外推；在安全资源窗口先用同任务/视频长度、同逻辑更新的短profile比较墙钟、rank等待和峰值，再决定是否值得做物理执行优化。MT-BC在完整checkpoint边界可按真实两节点卡况迁移到更有吞吐的单节点、调整物理卡数；必须保持同一条576查询轨迹和optimizer/scheduler状态，并登记换拓扑RNG边界。训练用任务交错分片检查rank等待，评测用动态队列防止长任务拖住全队；慢条件仍完整执行。仅针对实测瓶颈优化物理batch、评测副本/动态队列或kernel，并验证逻辑任务权重、配对映射与数值稳定性；活跃frozen runtime不原地改写。

结果不达标是科学负结果，工程失败按层定位并在原合同内恢复。只删确认可再生成且无后继依赖的载荷；checkpoint、raw rows、manifest与旧正式证据保留。修改集成`main`并推送，文档按实际阶段更新，不用频繁轮询代替现有控制器。

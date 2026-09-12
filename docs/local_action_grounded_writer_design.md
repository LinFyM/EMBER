# Local Action Grounded Writer：真实局部转移监督

2026-09-12，Owner再次明确给予核心科学精神内的理论／架构修正自由度后登记。
本设计的200步有界比较已完成并关闭，不再是active design；结果与停止范围见§7。
前置推导与历史核查见[过程获取分析](video_process_acquisition_analysis.md)。
当前目标及最终资格不缩减：正确action-hidden教学内容与顺序须通过一次生成的完整LoRA产生可重复闭环收益，
跨同task视频、初态、相邻checkpoint保持并迁移到validation8。暂不要求145/400。

## 1. 本次检验的因果变量

新增的监督关系是action训练episode中，实际RGB变化与其间已经执行的动作配对。
旧辅助FM与G2跨episode phase监督都没有提供这个局部对应关系。
本次保留已实现的任务token有序表示、完整H、Compiler及native D；它们现在是待检验的表示／编译器，
不继承“已经学会过程”的结论。区别是训练标签对应的信息，而不是扩大旧读出头或修改辅助权重。

两项竞争预测：局部动作监督能为共享表示提供可用的有序操作信息；或者它只能学习一般机械臂运动，
仍无法给完整LoRA带来任务过程增量。即使局部可解码性成立，策略编译与跨task迁移仍须单独证明。
v5.2正例继续说明普通FM有时足以利用视频，不能把本监督宣称为所有方法的必要条件。

## 2. 信息与数据边界

- 固定train24/validation8/test8，source及source normalization不变，不新增meta tasks／外部模型。
- 主LoRA：teacher0–15完整K1 action-hidden视频，action queries16–41，同task跨episode；每update四suite各一task，
  每task64主FM queries，权重1/4，总256。主FM沿用既有obs[i]/actions[i:]合同，不同时变更其时间对齐。
- 新局部监督：只从action pool16–41取同一episode的四个RGB帧，位置p,p+5,p+10,p+15。
  当前数据为post-action RGB，目标严格取actions[p+1:p+16]，15×7，选p使区间完整、不padding。
  episode均匀，再在该episode合法p中均匀；不使用任务阶段、动作幅度、成功状态或terminal筛样。
- task、episode、p只用于授权、采样与记录，不作为learned输入。读取器输入仍只有exact language及RGB/native响应；
  标签和加噪标签只能到训练专用局部头，不能进入encoder／Compiler。
- diagnostic42–45无梯度，held teacher46–49保持action-hidden；validation/test动作不读取、不训练。
  允许读取动作训练episode自身RGB进行共享表征监督，是Owner最新授权下登记的训练期用法；不取消主LoRA跨episode要求。

## 3. 唯一部署图与新增训练出口

```text
完整K1 teacher RGB + exact language
 → frozen vision + teacher Gemma VL Meta → contextual Z/task tokens
 → frozen Action Expert + Action reading Meta → 每帧完整R[50,1024]
 → 逐任务token视觉grounding、相邻完整H读取、两轮语言／过去时间交互
 → E[T,L,256] → 两层Compiler → native D → 一套38-target rank16 A/B
 → 冻结source依据自己的观测闭环

action池四帧局部RGB + 相同exact language
 → 同一个observer和encoder → E[4,L,256]
 → 训练专用15步局部动作FM头 → 实际15×7动作标签
```

局部头不生成LoRA，不向闭环提供动作，不做task-local BC或优化。没有第二个视频编码器、在线教学记忆或部署adapter。
局部头width256、heads8，两层已有标准cross/self attention+MLP。
15个动作token由7维加噪动作、flow time的正弦编码及局部动作位置形成，输出7维速度；末投影零初始化。
全片段E作Value，局部相对视频时间仅作有序臂的Key路由；不读取裸task ID、执行state或source执行query。
局部路由属于局部头，不复用／更新Compiler的时间投影。静态参照将此时间路由置零。

每条局部片段使用8个独立noise/time样本，noise/time由局部采样key决定，不改变主FM随机流。
采用与source FM相同的Gaussian噪声及Beta(1.5,1)经.999/.001变换的时间分布，速度目标noise−action；
局部动作只按冻结source quantiles归一化，不需要teacher state或重算统计。
对8×15×7取均值，每task权重1/4；总目标为主FM加局部FM，局部系数1。
此数值是两项均值损失同单位下的固定实现默认，不能将它称作理论最优或在本窗口内扫描。

Writer、两组reading Meta、Compiler/D及局部头全部fresh，fresh AdamW/scheduler；联合更新、无阶段冻结。
source基础权重始终冻结。局部梯度只更新共享encoder、两组reading Meta及局部头；主FM更新完整生成链。
不保留旧执行辅助头、函数蒸馏、rho课程或第二trainer。主LoRA cotangent重放与native Z/R重放复用已有实现。

## 4. 匹配参照与有限学习窗口

首个比较为ordered+localFM与frame_set+localFM。两臂共有参数按同一fresh初始化、任务／teacher／query／局部片段／noise流匹配，
主FM曝光、局部样本数、optimizer及LR一致；唯一表示差异是当前有序／全帧无序结构。
frame_set的帧独立native处理、无有向pair、无视频时间位置；所有帧经置换不变集合消费。
局部头也不能指定末帧、绑定端点身份或从原帧号恢复时间；15个动作输出位置在两臂相同，不映射到输入帧位置。

先profile最长93帧teacher及真实四帧局部支路，选择物理batch；首段预登记100/200两个完整checkpoint，
每臂200updates、800主条件、51,200主FM queries、800局部片段、6,400局部noise/time样本。
若现场拓扑使其显著偏离约一小时，须在看到学习分数前更新节点及曝光登记，不以profile冒充学习结果。
每个checkpoint进行下述诊断与行为验证，不增加seed/rank/LR/层位扫描。

固定局部留出诊断：train24每task16个片段，来自diagnostic42–45，各episode4个均匀合法起点；
seed20260913，每片段8个配对noise/time样本，0/100/200无梯度，报告逐task及两臂配对FM差额。
主动作留出沿用原train24、teacher46–49、action42–45、128queries/task、seed20260908，0/100/200。
二者分别记录，不把局部拟合与跨episode主动作拟合混作同一指标。

闭环节点100/200：train96 correct/other与validation400 correct/other，沿用已有seed20260911的严格配对state-video映射。
validation K1每臂50合法teacher/task各一次；train96是登记的四视频有限池面板，不声称跨整轮无复用。
所有per-task/suite、breadth、R/G/L、churn、相邻与换视频Jaccard保留；不以80-row screen或融合选模型。
若仅局部FM更好而主LoRA无获取／迁移收益，本窗口关闭；不改头容量继续追逐局部分数。
若局部与闭环都无有序增益，停止该监督在当前图上的投入；不把负结果外推为所有逆动力学不可能。

若两节点行为满足下节资格，才追加同一实现／共同初始化的fresh ordered纯主FM参照，匹配实际曝光，
用于判定局部监督相对同图无辅助目标的贡献；本段未运行前不把历史pureFM当作完全匹配的消融。
纯FM参照若已同样有用，优先去掉无增量的局部复杂度。不得依据新低分自动启动该第三臂。

## 5. 当前goal的完成条件保持

沿用已登记的行为条件：single-checkpoint validation correct相对充分匹配训练的全帧无序参照，
task-cluster paired bootstrap95%下界>0；other同向，至少两个suite有净收益。
相邻节点correct增量同向，同时报告绝对分数、source、所有task与成功集合；不靠弱参照退化制造差额。
train侧获取不能代替validation迁移，局部FM不能代替LoRA闭环。

配方及selected checkpoint冻结后，sealed seed20260912重新配对合法50视频池，进行correct/other/wrong/shuffled/reversed。
correct−wrong及correct−shuffled的task-cluster95%下界>0，reverse作为方向特异性支持；
最终controls不进入训练、局部诊断、选点或架构修正。bootstrap为20,000次、seed20260911、双侧percentile95%。
Test继续封存。独立RL、K>1与长期145/400阶段均不混入当前结果。

## 6. 工程与资源所有权

`video.py/native.py/native_factor.py`复用主表示、原生读取和完整LoRA出口；`function_reader.py`替换为局部动作头，
`function_credit.py/supervised.py`保留唯一主FM重放并增加独立片段的共享encoder信用；旧执行辅助／蒸馏代码退役。
`data.py/learning_data.py`只在明确的action数据所有者加入局部配对读取，不让RawTeacherVideoStore获得动作读取能力。
`runtime.py/training.py`仍唯一构建／训练／checkpoint入口，旧schema与旧checkpoint由其冻结Git实现保留。
配置及materialization身份更新后再formal执行；不保留兼容旧辅助头的活动fallback。

实现采用隔离codex工作树，合main并push；formal必须clean pushed detached commit。
启动前两节点GPU、总额度、NUMA/NCCL和strg01独立quota按现有合同实查，记录增长预算与精确命令。
不复制source/data/models，不保留整action池新cache。四帧局部prefix只作当次工作量，必要缓存仅冻结输入embedding。
峰值与吞吐按实际profile裁决，不用11.27%的输入帧数比例代替运行时间证据。

### 首次执行参数（学习分数产生前登记）

真实93帧profile两次完整更新通过，第二次17.558秒，allocated37.837GiB、reserved40.936GiB。
局部4帧支路约0.5秒，frame chunk8 / policy microbatch8保持；不继承profile权重。
按当前总额度，ordered与frame_set各使用同一四卡拓扑顺序fresh学习，同时在其它合法设备处理节点评测，
每臂200updates约一小时，两个节点与曝光不变。精确GPU/commit/命令与动态资源只记launch contract/progress。

## 7. 有界比较结论：关闭当前局部监督配方

两臂各完成200更新、51,200主queries及800局部片段；16个登记面板共3,968闭环rows全部完成，所有进程退出。
validation100 correct/other为ordered45/48、frame_set58/53；有序差额95%CI分别[−.06,−.01]、[−.025,−.0025]。
validation200为ordered30/27、frame_set30/26；差额CI分别[−.015,.0125]、[−.005,.01]，均无可信正增量。
四个训练侧匹配差额区间也均跨零。两臂validation后段退化，未满足相邻、跨视频有益过程资格。

局部FM两节点存在小幅有序优势，但主FM无可靠优势；局部头拥有独立时间Key路由，不能据此断言共享E已学会完整任务过程，
也不能将Compiler指定为唯一失败点。按§4停止当前配方，不延长训练、不扫局部头／权重／LR／rank／seed，
不启动条件触发的pureFM第三臂或sealed controls。未选checkpoint，Test未用，整体研究goal仍未完成。
没有同图无局部目标的归因结果，不能把与旧配方的分数差异单因归于局部监督；负结果不外推为所有逆动力学方法无效。

完整逐task/suite、source、R/G/L、churn、相邻与换视频统计在
`runs/analysis/local_action_grounded_20260912/paired_summary.json`，两类学习证据在`step100/learning_comparison.json`及`step200/learning_comparison.json`，
统一裁决为`bounded_200_decision.json`；历史摘要见[research_history](research_history.md#local-action-grounded)。

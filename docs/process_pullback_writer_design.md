# Process Pullback Writer

## 1. Authority and whole-method hypothesis

Owner在2026-09-14完成五个问题的讨论后明确恢复推进并要求设置goal；随后纠正：是否回到v5.2由owner决定，
agent只专注本方案的实施、训练、有依据的迭代与结果汇报。允许多次训练与工程修复；持续缺少正向信号时须降低
对实际检验组合的支持并明确报告，不自动回退、不无限追加无信息修补。Active design登记与执行状态以progress为准。

完整假设：exact language引导同步双路教学视频的状态／过程读取；变化驱动的共享网络输出动作空间作用码；
裸冻结source的固定导数把作用码编译为唯一完整LoRA；纯跨episode执行FM可以共同学得可迁移且保持的视频依赖能力。
这是一项联合科学假设，接口存在、梯度非零、局部oracle通过均不代替合法视频到闭环的证据。

当前优先级为有用的视频特异性与能力保持；>145/400保留为长期目标和参照，本轮不额外强制。
Owner已接受在选定冻结模型上correct明显优于wrong／shuffled／reversed的证据，不要求另训独立frame_set模型。
没有改动历史实验的原资格、分数或关闭结论。

## 2. Inputs and information wall

- 固定train24／validation8／test8。仅train24动作产生梯度；无额外meta tasks、Test、RL或task-local优化。
- K=1是一条episode的同步agentview＋eye_in_hand两路RGB，两路共同进入学习读取器及裸source编译器，不是K=2。
- exact language、真实stride5全视频并保留真实末帧；不按结果挑视频／帧，不读取teacher action/state/reward/terminal/ID/pose。
- 主教学池与action pool复用demo16–41；每个条件的执行queries排除该teacher episode，post-action观测对应future control。
- 固定独立训练侧诊断：动作42–45、teacher46–49；train闭环states32–35与teacher46–49明确登记有限池复用。
- validation正式每臂400行：8task×50init，每task的50合法teacher videos整轮各用一次，沿用video_schedule。
- 身份字段仅供数据加载、调度、审计，绝不进入模型。教学video自身不要求动作标注；执行FM的训练queries需要动作。

## 3. Reading and process content

真实双相机prefix经冻结vision与Gemma基础权重及fresh observer Meta产生逐帧Z和完整50-horizon Action Expert H。
observer Meta属于共享Writer，所有读取／过程模块端到端共同更新；裸source导数读取在所有Meta与执行LoRA作用域之外。
仅冻结的pre-Gemma输入嵌入以及裸source量可跨更新缓存；Z/KV/H不得跨参数版本复用。

exact language为语义查询、实体／关系角色及原生动作horizon读取提供条件。每帧先形成具有一致角色对应的语义状态e_t，
真实相邻状态差d_t=e_t-e_(t-1)作为过程内容。H的全部50位置在实际learned read前保持；horizon位置不是video时间。
允许整个视频的前后文辅助理解，但须保留真实方向与前置关系。

变化驱动递推为m_t=a_t*m_(t-1)+b_t*d_t，m_0=0；a/b可依赖当前语义、原生动作估计和已有过程记忆。
另有反方向上下文通道，保持与正向不同的前态／后态角色。静态语义与动作估计只条件化查询／门，不直接加到过程Value
或作用码。所有内容变换保持零输入为零；无绝对视频时钟Value、bias造出的动态或language-only参数旁路。
在相同合法画面重复的理想条件下，d=0，过程内容与作用码必须为零。该性质不证明真实视频的变化已经被有益消费。

作用码q_t在每个真实视频状态的全部50个horizon位置保留，使用7维实际动作输出坐标；完整32维原生输出中的其余维度
补零，不让无执行含义的padding输出成为额外自由字典。native source的公开probe固定[50,32]、seed1729、flow time=1。
q只读取过程Value；静态状态可决定怎样读取，不能独立产生非零q。末端q投影零初始化，其上游正常fresh初始化。

## 4. Fixed source pullback and rank16 outlet

F0(V_t)是同一合法双路RGB／language prefix下裸source的原生flow velocity，全部基础权重固定。
裸source动作估计probe-F0仅作读取条件。对38个目标层：

    G_l = (1/T) sum_t J_(W_l) F0(V_t)^T q_t
    A_l = top16 right singular vectors of full-video bare native X_l (orthonormal rows)
    B_l = G_l A_l^T
    DeltaW_l = B_l A_l = G_l P_l,  P_l = A_l^T A_l

所有真实frame／horizon参与X_l；不以observer Meta坐标替代裸source坐标。相同实际X的q/v目标可以复用计算。
不需要显式物化完整G：可逐chunk把输出余切传至每层，再与X_l A_l^T收缩为B_l。A/B一起形成唯一38-target rank16 LoRA，
alpha=rank，部署没有自由高维译码头、第二adapter、loss、optimizer、task-local候选或重新评估更新后的teacher。
一次Writer调用允许固定多阶段只读重放；编译明确使用冻结模型导数，不宣称为纯forward。

q到B是给定视频下的固定线性映射。训练反传使用其精确伴随，并分块重放裸source；source参数不成为可训练叶子。
不能把固定q detach后仅训练旁路。每个optimizer更新前完成所有FM、Writer与observer信用，沿用分离激活生命期的训练器。
导数精确描述F0处的局部关系，有限LoRA实际行为由完整非线性policy检验；不声称输出变化恰等于q。

PCA投影保留输入能量，未保证保留功能方向。旧fixed-A反例仍有效。启动完整学习前，仅做一次有界的训练侧功能核验：
同一批固定真实纠正q经本次G P出口，是否在独立episode保留可用作用；使用已有授权训练数据／已有面板与最小必要计算，
登记具体条件及判断后执行。不能拿旧G闭环或旧局部rank16场的结果替本出口背书，也不把核验扩为新的oracle课程。

本次固定使用`native_corrective_transfer_20260913/formal`的train24 × teacher16–19，共96条件；每条件使用原固定
动作42–45的16个跨episode queries及原noise。保留原四个teacher支持位置、前15×7动作、source velocity和eta，
以`q=-2*eta*residual/(15*7)*T/4`嵌入完整T×50×7，其余位置为零；T/4抵消新编译器1/T与原四支持平均的差异。
唯一主要变量为完整视频native X决定的PCA16投影出口；不重新拟合eta，不重跑或选择source／原G参照。
两个读出为t1 endpoint和full10，均使用query自身执行state。每task先等权聚合4teacher与16queries，再对24task等权；
20,000次task-cluster配对bootstrap，seed20260914。只有两个读出的source−G P改善95%CI下界均严格>0、
各至少两个suite净正且全部预测finite才通过。报告原G差额和各task／suite，不以本privileged结果选择Writer或宣称能力。
入口为`scripts/check_process_pullback_function.py`，输出`runs/analysis/process_pullback_writer_20260914/functional`；
默认只保留小型预测／元数据，预算128MiB，不保存可重建LoRA或完整G。判断已在本次GPU读出前登记。

## 5. Acquisition and retention through ordinary FM

唯一训练loss为同task跨episode主执行FM，无q标签辅助项、局部场监督、重建、蒸馏、顺序loss、负视频或RL。
Writer及两个observer Meta fresh，optimizer/scheduler/sampler/RNG fresh；合法identity q=0给出零LoRA。
初次更新主要学习q末层，之后实际FM信用须到达整个读取／过程网络，不实施分段冻结G1–G3课程。
递推保留门初始化依据训练视频长度覆盖有意义时间尺度，避免默认0.5的连续衰减；不把这一设计理由当旧实验根因。

每个logical update为4tasks、每suite一个，长期task等权；每task一个K1条件与64个跨episodequeries。
GPU／帧数成本分配不改变1/4任务权重。保持现有task/query随机性及执行精度合同，可按实测优化物理microbatch／chunk。
固定q→LoRA关系减轻共享译码漂移，但读取器仍可能遗忘或不泛化；必须由闭环获取／保持证据裁决。

## 6. Evidence, learning windows and iteration decisions

先验证合法输入、全38-target／完整H、identity、q→LoRA伴随信用、第二次更新的联合梯度，以及最长真实视频吞吐／显存。
这些只防止无效工程投入。新出口功能前提通过后，用真实吞吐与历史曝光尺度登记初始学习窗口与中间／末尾闭环节点，
在看到正式分数前落实更新数、teacher条件数、queries、wall-clock和保存点。不得机械继承旧50/100硬截断，也不无限延长。
历史v5.2的3600条件／75600queries、v6的9600条件／192000queries只用于学习尺度判断，不是新方法的充分性证明。

- 可信正向信号：correct闭环相对paired source出现明确获取，且增益跨任务／suite而非一个峰值；训练／held趋势及保留支持继续。
- loss／q拟合／参数几何单独改善不授权追加预算。训练改善但held持续退化须降低迁移／保持假设的支持。
- 可复现合同错误可修复并标记受影响证据；按原算法继续的吞吐修复不是新的科学候选。
- 架构、监督、数据范围或参数作用关系的实质变化是新假设；须说明历史近邻、新增证据与不同结果如何改变投入。
- 合理窗口后持续缺少正向信号时完成有限原因分析并明确报告，不自动切换v5.2；owner保有是否停止坚持的决定权。

所有正式分数为single-checkpoint strict paired400，不用screen、union、融合或挑视频选模型。
按per-task／suite、breadth、R/G/L、churn、相邻success-set overlap与真实曝光判断；相邻节点检验保持，same-task-other检验视频鲁棒性。
能力与相邻资格成立后补完整other400并选定冻结单checkpoint，再做同配对合同的wrong/no-video/shuffled/reversed最终controls。
shuffle/reverse重排真实双路frames后重新完整读取和编译；这些结果不进入训练、checkpoint选择或架构修正。
最终完整报告正确视频能力、same-task-other、三类错误视频、source参照及不确定性；不把错误条件退化当正确条件获益。
正向学习、能力但视频依据不足、未获可迁移能力与证据不足分别报告；不把局部non-pass扩大成所有video-to-LoRA不可能。

## 7. Engineering ownership and launch

复用writer/runtime、native observer、data、supervised、training、materialization和唯一strict evaluator。
video.py拥有任务条件化过程与q预测；correction.py/factor.py拥有固定source坐标、投影及线性编译／伴随。
替换退役的LocalFieldLoRADecoder与LocalFieldSupervisor，不保留新旧两个active Writer或辅助field分支。
旧代码与schema由Git、sealed configs和formal artifacts保存；结束运行的clean worktree可删除并按原commit重建。
旧checkpoint不由新runtime加载。唯一入口为scripts/train_writer.py与scripts/materialize_writer.py。
隔离并发实现使用codex worktree，集成main并push；正式train/eval来自clean pushed detached frozen commit。
每次GPU launch前同时检查两节点；新大run root前按strg01独立quota及实际占用估计峰值，不复制source／数据／模型。
完整训练checkpoint保存Writer、optimizer、scheduler/scaler、sampler/cursor、rank RNG、world topology及schema。

## 8. First learning window registration（2026-09-14，正式训练／闭环前）

96条件的G P功能前提通过：t1的source MSE .11977705→.11523266，改善CI [.00142637,.00853139]；
full10 .16493043→.15792518，改善CI [.00191831,.01375751]，分别4／3个正suite且全部finite。
相对原完整G平均保留72.56%／76.29%的改善，故投影有实际损失，不能宣称等效保留全部纠正。
原件与逐task／suite在`runs/analysis/process_pullback_writer_20260914/functional/summary.json`。

初始窗口固定fresh **900个logical updates**，每更新四task、四K1条件、256个跨episodequeries，
合计**3,600教学条件／230,400 queries**。100的倍数保存完整checkpoint；**300／600／900**各做完整correct
validation400和train96，同时在0／300／600／900做既定独立动作诊断。300达到约76,800 queries，接近v5.2
历史query曝光；900达到其条件曝光并超过v6历史query数量。这给予更有信息量的学习机会，不保证新方法已充分学习。

执行按300／600／900三个段边界衔接，及时完成对应闭环再exact-resume同一run；不重新初始化、改数据或改变梯度权重。
每段继续原Writer／optimizer／scheduler／sampler／rank RNG与同一world size／topology，只改变已预登记的段停止点。

三个节点共1,488条闭环rows，跨checkpoint固定state/video/RNG映射。既有source train17/96与validation47/400只在
当前执行合同和逐episode pairing一致时复用原rows，来源登记到本轮运行记录；不以旧Writer结果代替source。
source的两个参照为`native_correction_writer_20260913/oracle_rollout/evaluation/source`与
`semantic_path_writer_20260914/source400`。

继续与裁决服从§6：看实际获取、跨task／suite分布及相邻保持，报告source差额CI、breadth、R/G/L、churn和重合；
真实窗口后持续接近source或退化则降低本组合的支持，完成有限原因分析，不自动追加到1200／1500或小扫。
有明确获取且仍在保持／改善时可登记有依据的后续窗口；loss或本节privileged正例不构成延长理由。
same-task-other和最终controls仍须先满足前置资格并冻结单checkpoint；本节不提前打开sealed controls。

最长task38/demo36（105个真实采样帧）的两次联合更新与部署编译均通过。缓存裸source前缀、直接q伴随及官方FM
prefix-KV路径后，frame_chunk4／8／16第二次更新分别31.95／26.72／23.99秒；原full-prefix基线46.86秒。
选用**frame_chunk16、FM microbatch16**，峰值37.01GiB、部署7.78秒；FM batch32相对16的收益不足1%，保持16。
全部模块第二次获得finite信用，source冻结；真实nonidentity全38 FM对照梯度cosine .999899、norm比1.00363、loss差0.155%。
原严格数值probe的阈值失败保留；带指标对照和安装版运算核对未显示语义变化，不据此关闭BF16或追求逐tensor一致。
profile仅为一次条件更新，正式logical update仍须汇总四task。真实900更新采样共129,188帧，四卡最长rank平均58.53帧；
按实测估计更新本体3.86–4.19小时，正式准备按约4–5小时及额外诊断／评测时间执行，并以实际墙钟更新估计。
九份完整checkpoint加首轮1,488个物化LoRA约8GiB，初段预留12GiB；连同有资格才触发的controls，整轮预计峰值24GiB。
这在清理后约217GiB独立data1配额余量内；正式launch仍登记当时实际quota、两节点GPU与单节点拓扑。

## 9. 900更新窗口完整结果与裁决（2026-09-14）

预登记900更新、3,600条件／230,400queries完整结束，实际覆盖623/624个task/video条件；独立task仍为24。
训练来自d6defa69，物化／评测来自f5d9db78的clean pushed frozen trees。九份checkpoint、六面板／1,488rows保留，
训练和全部最终worker exit0；source、完整38-target、曝光、固定视频／初态／RNG和normalization配对审计通过。

| 更新 | Train /96 | Validation /400 | Validation S/O/G/L | Validation breadth /8 |
| ---: | ---: | ---: | --- | ---: |
| 300 | 24 | 64 | 0/26/36/2 | 3 |
| 600 | 22 | 72 | 0/29/40/3 | 4 |
| 900 | 26 | 64 | 2/18/44/0 | 4 |

Source为17/96和47/400。900的train净增9、task-bootstrap CI[+2.08,+17.71]pp，保留真实局部学习。
Validation相邻R/G/L为56/16/8、52/12/20，Jaccard .7000／.6190；Object task1为5→26→29→18，Long最终归零。
900的validation相对source区间为[0,+10.75]pp，三个节点未形成广泛且保持的suite能力；本阶段不以>145/400作为额外硬门槛。

本轮未获能力／相邻前置资格，不选择checkpoint，不开展same-task-other及wrong／no-video／shuffled／reversed。
这些未运行的条件仍是未识别范围，不能报告成视频无效、顺序已验证或因果controls失败。
固定动作诊断19/24task改善、均值下降约3.08%；Writer和两组Meta在首步后899次梯度记录均finite非零，source冻结。
梯度与loss不弥补闭环缺口。当前联合组合的稳定广泛迁移解释被降级，但q、投影、读取器、任务支持及优化尚不能单独定责。
train96用teacher46–49，不能单凭它区分训练池拟合与同task换视频／初始化泛化；900更新不是普遍收敛或不可能性证明。

按本轮窗口结束原样续训及小扫，保留实现和证据。q辅助尚未实施；后续实质修订或回到v5.2由owner依据结果决定。
完整逐task／suite、配对集合、成本及有限原因分析见本轮`READOUT.md`、`paired_readout.json`与`bounded900_decision.json`；
原件根为`runs/analysis/process_pullback_writer_20260914/`。训练／物化／闭环墙钟分别4.565／.462／1.091小时，前期profile另计。

## 10. 原因诊断授权与预登记（2026-09-14）

Owner在看过三个问题的原因分析及拟议方案后明确要求按方案充分诊断。仅授权下述诊断／冻结评测，
不启动新一轮正式Writer训练，不做q辅助、RL、Test或方法切换；结果在对话中说明，不新增用户报告。
原§9的科学non-pass保留；下述注册在任何新增outcome前完成，原件集中于本研究根的`causal_diagnostics/`。

### 10.1 视频特异性与跨视频保持

- 冻结窗口末尾900，不按300／600／900分数选点；沿用seed20260911及全部validation8×50的state/video/RNG配对。
- correct复用原完整400。same-task-other复用兼容完整LoRA，改变逐state视频映射；每task整轮各50视频且逐行不同。
- cross-suite-wrong使用共享video_schedule的同split循环donor映射，保留目标exact language；shuffled／reversed
  重排同步双路真实frames后完整重做observer与裸source编译。绝不打乱latent或LoRA冒充真实输入干预。
- no-video定义为零视频作用的identity/source参照，不做fake zero-image forward；检查实际零LoRA与官方执行合同后复用paired source。
- 每臂报告完整paired差额、per-task／suite及不确定性。这些controls只用于描述冻结模型，不能选择或调整下一方法。
- 另对300／600／900做train24×states32–35，teacher16–19按原seed的有限池ordinal规则各用一次；
  与既有teacher46–49的相同初态面板比较。新旧池不是新task，不把两者成功率差直接称唯一因果泛化损失。
- 行为回放固定全部validation8×init0／12／25／37×三个checkpoint，共96条；复用原bank与RNG，保存逐replan输入／动作和谓词。
  不按成败挑病例，报告对原结果的复现情况；对象选择、抓取／运输、放置及组合任务进展只作有范围的行为定位。

### 10.2 固定出口与共享获取的有界可达性参照

- 全train24，每task固定完整teacher16一条，共24条件。以900的同一套q及完整A/B为起点；原Writer、两组Meta及source全部冻结。
- `writer900`保持原输出；`free_q`只优化每视频的完整T×50×7变量，经过原冻结source/PCA16编译；
  `free_lora`只优化同起点的完整76个A/B factors。两个诊断参照均为唯一38-target rank16，绝不部署为Writer或后继初始化。
- 支持集固定64个同task跨episode动作query，来自demo17–41并明确排除teacher16；使用两套独立官方FM time/noise，
  共128个固定预测条件，保存实际位置、time/noise和seed。独立动作42–45、独立噪声不参与优化、停止或选择。
- 两臂采用同一原生LBFGS，max_iter32、soft max_eval48、history_size10、strong_wolfe；记录真实closure次数、
  原生退出信息和有限值。线搜索可以超过soft max_eval，不自造中途硬截断；不根据held表现改步数、LR或选中间点。
- 先单独做首个train task的2步机制／成本smoke，结果不合并正式诊断。工程错误可修复，科学预算变更须在完整outcome前解释。
- 报告起点／末尾支持及独立动作功能。三个最终参照统一在train24×states32–35上评测，每task复用teacher16，
  明示有限视频池；状态从不参与拟合，source17/96只在执行合同及逐行配对相同时复用。
- 直接拟合只是受给定优化预算限制的可达性参照：free_q成功不能证明共享RGB可学；失败不能证明数学容量上界；
  free_lora与free_q的差同时受参数化及优化影响。只按已实际区分的接口提出修正，不由此恢复旧solver或追加扫描。

每次GPU启动按现有两节点、单节点有效卡数、NUMA、quota和clean pushed frozen commit合同登记。
本次训练侧拟合不更新正式模型；视频控制、seen-video和行为回放均无梯度。正式源码变更独立集成，原训练运行树保持冻结。

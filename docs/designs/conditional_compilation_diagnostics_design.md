# 给 Luna 执行任务：条件参数生成的四臂诊断

> 状态：2026-09-23被Owner暂停。以下为冻结方案及当时执行授权，不能据此恢复实验；最新授权见progress.md。实现草稿未通过GPU验证。

任务标识：`conditional_compilation_diagnostics_20260923`。
配套机器规格：`configs/conditional_compilation_diagnostics_v1/experiment_spec.json`。两者共同定义本批工作；不得自行修改科研参数。
这是执行合同，不是给 Owner 阅读的结果报告。

## 0. 授权、分工与消费顺序

Owner 最新指令是：“详细规划好具体的实验流程，然后可以把活继续派给luna session。不过luna只能干说清楚明确要求和指令的活，所以你在安排的时候要足够详细。”
本交接据此授权：实现本合同、完成必要工程验证、运行四条有界 fresh 诊断及规定评测、保存证据和回报。先前全局暂停对这四臂的对应部分已被本次指令取代；除此之外不恢复其它实验。

接收任务：`01a0cc37-2740-71a1-b470-153548d41f87`，现名“了解 EMBER 仓库”，gpu02，仓库 `/data1/user/ymdai/projects/EMBER`。
主讨论/科研决策任务：`01a0c8da-0058-7951-8d1c-9c2233079f7e`。

按顺序执行：

1. 检查本批是否已经登记/执行；同一批消息不重复启动。
2. 继续完成已经启动的 Source1000 与 MT-BC300 Test400 补测，保留原合同、选点、输出。不要中断它们。CPU 实现准备可以并行，但新 GPU 作业必须等它们完成并释放资源。
3. 本批任务/配方已经在 baseline Test 结果被主讨论任务读取前固定。无论该 Test 分数如何，都不据它改本批划分、目标、节点或参数；只将 baseline 事实单独报告。
4. 将本执行合同原文提升为 `docs/designs/conditional_compilation_diagnostics_design.md`，将 JSON 提升为 `configs/conditional_compilation_diagnostics_v1/experiment_spec.json`。如需更正矛盾，先通知主任务，不静默改科学字段。
5. 在 `progress.md` 和 `task_plan.md` 顶部登记本次授权、active design、阶段和明确边界。旧暂停段落保留并标注适用范围，不把历史“下一步”复活。
6. 你负责实现、验证、调度、记录和 Git 集成；主任务负责解释科研结果和决定后继改法。遇到任务身份/方法定义等科学矛盾时停受影响阶段，给主任务列出精确矛盾；普通工程问题在既定合同内自行解决。
7. 按下述工程检查全部通过后自行启动这四臂，不再向 Owner 申请一次启动批准。完成本批后停止新增实验，通知主任务。不要自行追加第5臂、换超参、选新 seed、重训旧v5.2、恢复旧shuffle/reverse、跑新的正式Validation/Test、FT/RL或36任务正式候选。

此前主任务给你的“baseline结束后清理旧worktree”要求撤销；你专注本批实验。运行目录及旧worktree善后由主任务负责协调。不要为此打断实验。

本主任务只写此临时交接目录，不会与你同时修改代码或仓库状态文件。你将材料提升到已提交的正式位置、确认收到后，可以删除这份已消费的临时交接目录；正式合同是后续唯一引用。不要删除未消费的旧任务证据。

## 1. 科学问题与本批能够识别的范围

比较链是：直接优化一套rank16 LoRA → 用语言生成LoRA → 加入教学视频生成LoRA → 改变第二组功能损失。

四臂均 fresh，共同 Source1000、rank16、38 targets、动作查询流、task权重、实际标签曝光及优化时钟。它们不互相初始化，不构成“先公共底座、再视频残差”的课程。

- A与B：直接参数优化与语言条件参数生成的整体差异，包含参数化/共享/优化几何。不能凭这一差值唯一归罪于某个头。
- B与C：可训练的语言参照与完整视频生成链的差异。它包含增加的视频通路、Meta和相关参数，不能宣称严格参数量匹配的纯输入消融。
- C与D：同一视频架构、相同原始查询和权重；第二组目标由随机tau/全50步变成tau1/前5步。这是一个明确的目标定义干预，不包含更换episode、删标签或减少更新。

A名为“匹配直接LoRA参照”，不是历史MT-BC的只改rank复刻：这里采用4任务/112查询和共同Writer时钟。不得用A的任何分数替代原rank128 MT-BC300的155/400，也不得由A不够强宣称rank16上限低。

所有新分数属于当前Train池内的开发诊断。新的held8是对本批四条学习轨迹未见；不声称研究者过去从未观察过它们。本批没有给官方Validation8/Test8增加模型结果。

### 已完成baseline Test后的解释补充

本批划分、四臂、采样、优化和评测节点固定后，Owner要求主任务审阅Luna刚完成的baseline Test并判断是否调整方案。核验结果：Validation Source51/400、MT-BC155/400；Test Source75/400、MT-BC121/400。Test内MT-BC相对Source保留46、新增75、丢失29。Source变强而MT-BC变弱，因此不能把两者差异概括成Test统一更难，也不能继续假定只有EMBER存在跨任务迁移或能力保持问题。8个任务簇不足以由总分差判定划分有错或无错；两个划分不是逐行配对条件。

**由此只补充分析口径，不改变本批训练/评测科学参数，也不增加GPU实验。** Source原定的held400/seen64已足以在每臂、每节点分解“继承能力保留”与“新能力获取”。§10规定对此机械统计。既有Test结果独立登记为背景，不用于选择本批任务、训练目标、checkpoint、超参或后继修复；不尝试让Validation/Test分数接近而重划数据。

## 2. 固定数据：28任务拟合、8任务诊断留出

唯一父协议仍为 `configs/libero_24_8_8_coverage_v1/protocol.json`，数据身份仍来自同目录manifest及canonical revision。
不更改父协议、manifest的官方24/8/8角色、Source71、normalization、dataset revision或HDF5。
本批增加的是显式训练子集/评测子集登记，所有36个任务在父协议中都仍是Train。

诊断留出8个global IDs：`[0,1,14,15,20,21,36,38]`。

| suite | local IDs | 内容与支持边界 |
|---|---|---|
| Spatial | 0,1 | between plate/ramekin、next to ramekin选碗到盘；拟合池保留搬碗到盘，具体选择关系留出 |
| Object | 4,5 | ketchup、tomato sauce入篮；拟合池有其它对象入篮，这两种对象不作为fit28动作目标；Source71有完全相同BDDL对象类型的ketchup-to-drawer/tray与tomato-to-tray |
| Goal | 0,1 | 开中间抽屉、碗放炉上；拟合池有开顶抽屉、搬碗及moka放炉上；中间抽屉几何和碗—炉配对留出 |
| Long | 6,8 | 杯放盘＋pudding放盘右、两个moka上炉；拟合池有对应对象操作与多物体任务，完整组合留出；Long8炉灶初始已开 |

拟合28个global IDs：
`[2,4,5,7,12,13,17,19,22,25,28,29,32,34,35,37,42,43,51,55,56,62,64,73,95,96,97,101]`。

其中16个target任务用于seen闭环面板：
`[2,4,5,7,12,13,17,19,22,25,28,29,32,34,35,37]`。
12个辅助任务仍按原始global=40+LIBERO90 local ID解释，不向Writer输入这些ID。

实施前只读核验：

- 28与8互斥、并集准确为父协议36 Train；每个target suite各留出2个；官方Val/Test与二者都不交。主任务已提供`configs/conditional_compilation_diagnostics_v1/partition_audit.json`，提升到配置目录并按原始身份核对即可，不让你自行重新选划分。
- 复用 `configs/pi05_source_corpus_v1/overlap_audit.json` 的完整BDDL字段核验8个任务与fit28不存在完整任务等价。必须同时看操作、对象/装置类型、selector、初始谓词、目标及组合；不是只比language或文件名。
- Source71既有排重覆盖目标40。Object4/5的Source层对象支持local32/58、59已经由主任务核对；区分Source知识与Writer fit28操作监督，不能把二者写成相同的训练覆盖。
- 以上支持不是“全部几何/组合都已见”。若发现完整等价泄漏或登记ID错误，回报主任务；不得自行换留出任务。
- 可以读取训练数据的metadata和原有审计，不能为了选分组先跑不同分组的成功率。

梯度数据只来自fit28 demo0..45，video和action池都为0..45。demo46..49保留给本批独立诊断，A也不能使用。
diagnostic-held8全部episode禁止进入optimizer、蒸馏、统计量拟合、数据驱动初始化或学习率校准。
其RGB可在冻结checkpoint评测时作为合法action-hidden教学输入；其动作/state标签只允许在§8规定的训练全部结束、选点冻结之后作无梯度动作探针。
官方Val/Test action/state/reward始终不进入本批分析或训练。

## 3. 四臂的精确计算定义

### A_direct16

直接训练一套覆盖38 targets的rank16 A/B。Source基础权重冻结，执行policy继续正常读取当前观测、state和exact language。
使用与Writer相同的合法rank16 identity template，A初始化按现有template、B为物理零。禁止SVD现成MT-BC128作为初始化。
所有任务共享同一套可训练A/B；teacher编号仅存在于共同采样元数据，用来让动作查询排除规则一致；不得加载teacher RGB到A，也不得使teacher/language控制A/B参数值。
复用现有完整LoRA functional credit计算，通过identity参数出口累加同一宏步的四任务梯度，一次SUM、一次clip、一次AdamW。
不要复制一个source-SFT训练器来做本臂。可以在共同训练入口增加显式direct parameterization，使用外部可训练A/B leaves保持物理source冻结。

### B_language

这是fresh训练的语言参照，不是现成视频Writer推理时置零图片。
唯一条件输入为exact task language，沿现有真正text-only原生读取路径：

1. 用同一tokenizer及task span，调用/提取 `Pi05LanguageAxialEncoder._encode_text` 的实际text-only逻辑：真实语言embedding、Text Meta、原生语言模型、已有language_projection得到256维task tokens。
2. 对这些tokens运行 `LanguageSemanticCore.blocks` 的两层语言token处理，位置为真实文本顺序，valid mask一致。跳过frame_attention，不制作假frame evidence。
3. 复用compiler的routing、CoreSlotReader及core_norm得到normalized core slots。
4. 直接将这些slots交给现有post_fusion和完整FactorHeads，生成与C/D相同38-target rank16完整A/B并加同一identity template。

没有图像编码、VL Meta、Action Meta、Procedure、ProcedureSlotReader或AdaLN gamma/beta计算；不制作zero-image、fake action query或假的零Procedure去触发native forward。
保持其余共享模块定义和初始化不变。B训练Text Meta、language_projection、两层Core blocks、routing/Core reader/core norm、post_fusion、FactorHeads；不消费的模块显式冻结并排除optimizer，记录有效可训练参数数目。
优先给已有类增加有明确条件模式的入口/共享helper；不要fork完整模型。允许模型容器保留未使用模块以保持初始化序列，禁止用“挂着未使用参数”声称参数量匹配。
同一语言在不同teacher编号下必须生成同一LoRA，因为根本不读取teacher。这是接口属性验证，不新增闭环实验。

### C_video_fm

复用当前canonical完整video Writer，单agentview、K1、stride5含真实末帧、完整H50、三组Meta、Core/Procedure/Compiler/完整A/B。部署一次编译，不局部优化。
训练中两组query都使用官方随机flow time、独立Gaussian noise、全50步×真实7维FM均方误差。

### D_video_aux

Writer计算、可训练参数、初始状态、teacher流、所有query episode/frame、Gaussian noise流与C相同。
第一组21条仍是随机tau/全50步FM。第二组7条固定tau=1、只对前5步真实7维求均值。权重见§4。
第二组严格来自同task另一episode，不能切回same-video。这是在当前最佳C600所用目标上的fresh参照，不是昨晚12条件配方。

初始化：四臂seed7；C/D共同模块初始化必须由相同构建顺序与seed保证。B保留共享模块的同一初始化规则。不要为了“逐bit证明”做全张量扫描；合法identity、共享构建配置、实际图与少量行为检查足够。

## 4. 固定宏步、查询流与loss

每个宏步4个不同任务。28个任务每7宏步各出现一次，逐轮随机排列，任务等权；GPU分片不得改变该逻辑。
随机种子：model7、sampler20260721、teacher_video20260722、teaching20260919。
复用 `WriterTrainingData` 现有event算法，显式支持本登记的4条件与fit28子集；不能把当前全局常量12临时monkeypatch成4再冒充同一合同。

每个条件一个teacher demo，21条主query来自21条不同且非teacher的episode；额外7条来自同task另一条非teacher episode的合法stride5位置。
21与7的episode、frame、offset、Gaussian seed应在一个共享event plan中提前生成，四臂读取同一计划。第二组7条允许按现有算法在合法位置不足时有放回，但必须记录。
参考实现是 `learning_data._build_events/_episode_queries/_teaching_event`；保持seed派生、query offset1、噪声逻辑batch21/7和抽样规则。

设主query集合Q21，额外query集合Q7：

```
A/B/C 每宏步 L = (1/4) sum_task [mean_Q21 FM_random_tau_H50
                                 + (1/3) mean_Q7 FM_random_tau_H50]
D     每宏步 L = (1/4) sum_task [mean_Q21 FM_random_tau_H50
                                 + (1/3) mean_Q7 FM_tau1_prefix5]
```

在A/B/C里，每条query权重均为1/84，故整个112query的总loss尺度为4/3，不是未经说明地改成mean112。四臂保留同一总尺度，以免C/D比较混入loss整体缩放。
A/B/C的额外7条仍使用与D相同的episode/frame。不能为了“纯FM”把它们改采成7条独立episode，那会同时改变原始监督数据。
D与C共享第二组Gaussian noise；C另外按同一个合法FM time采样函数产生tau，D令tau=1。不得改变第一组任何随机流。
每宏步全部四任务及两组余切求和，再clip1、一次AdamW。绝不按物理world size再除一次，不对不同GPU上的任务各自单独Adam。

## 5. 统一学习窗口与选点

每臂1260个真实optimizer宏步，不继承任何checkpoint/Adam/scheduler。每7步一轮，共180轮。
每臂总计5040个任务条件，141120个动作query：105840条第一组＋35280条第二组；每fit任务180次条件访问、5040条query。

四臂共同AdamW：lr3e-4，betas(.9,.95)，eps1e-8，wd1e-4，global clip1。
共同调用现有Writer `_learning_rate_multiplier`：warmup150、decay_updates18000、floor1e-5、tail1350..2250、tail_final_ratio0.1。实际执行到1260；这段没有进入tail。记录真实applied LR序列，不让scheduler多走一步或每个microbatch走一步。
这个共同优化时钟用于参数化诊断，不是对A重新搜索最优MT-BC配方。

完整checkpoint每105步保存一次，含optimizer/scheduler/scaler状态、sampler游标、各rank RNG、topology/schema；评测只在：
`210,420,630,840,1050,1260`。

各臂完整跑完1260及六个指定节点；不根据某个差点临时换配置/加密checkpoint/选择其它节点。异常nonfinite、数据墙错误、OOM/资源变化按工程/资源规则暂停和恢复，不把这些写成科学负结果。
同拓扑恢复按完整状态；如物理拓扑变动必须沿已有显式转换合同记录，不声称bitwise exact。

每臂的diagnostic selected checkpoint为六个完整held400中correct成功最多者；同分取更早。seen64、loss、other/wrong都不参与选点。
选点只用于本诊断最终对照，不产生官方方法资格。先冻结selection文件再开视频controls。
固定窗口不足以宣称某方法的容量/收敛上限；结果报告保留此边界，不自行追加训练。

## 6. 工程实施与启动前检查

使用 `codex/conditional-compilation-diagnostics` 独立开发worktree；先从最新clean pushed main创建。baseline的frozen evaluator继续原树运行，严禁修改它。
你可以在baseline运行期间做CPU实现/检查；GPU profile要等baseline两臂完成。开发树和正式detached runtime均遵守现有存储放置及Git合同。
本批完成验证后及时集成main并push；四条正式诊断都从同一clean pushed detached commit运行。运行中不热改源文件。

已有owner及具体改动位置：

- `src/ember/writer/learning_data.py`：显式登记4任务、fit28子集与共享event plan；保留canonical父协议data wall，不伪造父manifest为28任务。
- `runtime.py/model.py/video_program.py/temporal.py`：受控的direct/language/video三种诊断parameterization，复用同一真实文本读取与输出头。
- `supervised.py/function_credit.py`：两组余切共用，第二组目标由明确enum切换；canonical FM仍由 `paired_functional_credit` 负责。
- `training.py`及checkpoint/materialization owner：给本批schema明确注册权限和恢复语义。不能用删assert、伪造旧experiment声明、忽略strict load来绕过合同。
- evaluator复用 `scripts/evaluate_pi05.py` 的 `--role development_train --task-subset-selection`；现已支持screen4和formal50，不需要改官方split。
- `src/ember/expert_manifold/video_schedule.py` 是视频映射唯一owner；动态队列/persistent workers及官方执行流程不另写一份。

必要的新诊断配置可放 `configs/conditional_compilation_diagnostics_v1/`。共同代码用有界配置差异表达，禁止复制四个trainers或把历史快照整份拷为新运行面。
本批涉及保留的入口和模型模式，应用 `code-architecture-gate` 的结构检查；使用 `using-git-worktrees` 做写入隔离。守住owner职责，超过尺寸信号时做合理抽取/解释，不为行数机械拆文件。

CPU检查应覆盖真实合同，不为了填检查数另造大测试框架：

1. fit28/held8/official Val/Test互斥，任务身份及上述完整等价审计成立；训练构造器遇到held8应拒绝进入event/optimizer查询。
2. 六个节点预算以及每7步28任务等权；四臂同一event plan，21主query跨episode且distinct，额外7条非teacher，offset1、实际权重和RNG分片一致。
3. C/D配置diff仅有声明的第二组目标；B的active parameter名单明确；A只训练共享A/B。
4. 分片后的条件/余切权重不丢失、不重复、不多除world；现有合并与恢复测试尽量复用。
5. held400/seen64 subset selector、50视频无放回、other偏移17、wrong donor、期望行集合正确。
6. checkpoint schema/加载器/物化器可识别本批，不能把A/B的无video情况假装成读取了teacher。

GPU disposable smoke/profile（独立输出，权重绝不用于正式初始化）：

- 每臂最多4个连续宏步验证实际图和更新；所有source基础参数requires_grad=False，实际LoRA shape/finite，梯度进入该臂应训练路径。identity初始化第一步上游为零允许，后续须有真实信用。B不要求冻结的视频模块有梯度。
- B在相同language、改变teacher编号情况下结果不变；A不同任务条件的输出一致。检查实现接口即可，不启动额外闭环来验证这种结构属性。
- C/D第一组在同一状态/同一query/noise/tau下的loss和功能预测一致；第二组原始标签/noise配对，目标差异如定义。用正常dtype容差，不追逐低位一致。
- 做一次实际保存→加载→下一完整宏步的恢复检查，核对游标/学习率/有效更新语义及合理数值一致性，不要求bitwise。
- video臂选择fit28中元数据最长的视频测试真实full-H/可微重放显存；复用现有frame microbatch/profile机制，不删帧或改stride。
- 测实际queries/s、condition/s、update秒数及峰值，固定能通过最长视频的物理microbatch。最长视频OOM可降低物理batch，不改4×(21+7)、模型宽度或有效梯度。

完成CPU与smoke/profile、Git集成及资源launch合同后，向主任务发一次简短ready消息（commit、命令位置、estimated wall time、是否有未解决问题）。全部通过即可继续启动，不等待Owner再次批准。

如果科学字段的实现需要改合同、某个预期活动路径持续无梯度、恢复失败或有任务等价问题，不能自行“近似执行”。报告确切接口/证据并停止相关launch；独立且不依赖该问题的准备可继续。

## 7. 固定闭环评测

执行合同原样保持：render256/model224、双执行相机rotate180、8维state、7维action、10flow、执行前5动作再replan、settling10、成功终止、horizon Spatial220/Object280/Goal300/Long520、环境与policy seed7。
Teacher仍仅agentview；执行policy使用双相机与自身state。禁止将两者混淆。

### Source参照

Source1000固定跑本批held400和seen64各一次，共464条。允许复用已有完全同条件的完整原件，但逐行任务/初态/RNG和执行合同必须吻合；不拿旧小面板近似替代。
Source参照只用于定位基础能力，不选择训练臂。

### 每个checkpoint

- diagnostic-held400：held8各state0..49；teacher pool0..49，每task50条全部各一次。复用video_schedule、seed20260911、without_replacement。跨模型/节点固定同一映射。
- seen64：fit target16各state0..3；分别teacher46/47/48/49。每task四条新episode各一次。明确这是有限诊断面板，不声称正式400。
- A对所有任务使用同一套A/B；B对同一task可以物化一次复用。为配对保留teacher schedule元数据，但记录teacher values read=0，不能假称用了视频。
- C/D各条件生成一套完整LoRA，运行前固定。不得平均、挑teacher、checkpoint union或引入第二adapter。

六节点×四臂×464＝11136条。Source464及末尾controls1600后，共13200条注册闭环（smoke无新增成功率面板）。
每个面板等期望唯一行全部完成、所有worker exit0、聚合与配对通过后才读成功率。不读部分得分；工程活性/OOM/退出检查正常进行。

evaluator使用现成cost-balanced动态队列、long-first、persistent workers。若物化需要新manifest schema，保持static task-LoRA bank合同的来源、rank、target、task/state/video/RNG可追溯，显式登记此诊断schema，不能伪造旧Writer资格标记。
`--task-subset-selection` 的role仍是development_train，held8表示诊断子集身份，不把它改写成官方validation。seen16同理。

### 选点冻结后的controls

仅C和D各自选定的一点，分别执行same-task-other400、cross-suite-wrong400。
other使用同canonical函数的numeric demo offset17，整轮仍50条各一次，逐行不同于correct。可按同checkpoint/同teacher复用correct LoRA bank的精确条件，必须重新做全部rollout。
wrong donor由canonical suite轮转及suite内排序生成，global映射固定为：
`0→14, 1→15, 14→20, 15→21, 20→36, 21→38, 36→0, 38→1`。
target language/执行state/env RNG/policy RNG保持，teacher ordinal按canonical paired schedule。wrong必须对替换后的真实RGB完整编译。
不对A/B重复运行没有输入变化的other/wrong，直接用它们对应完整correct参照并标注是同一结果。
本批不跑shuffle/reverse/no-video，不用control结果重选checkpoint、改训练或补实验。

### 行为保存

全部闭环保留现有compact执行记录，足以还原task/state、teacher、success、steps、实际action chunk/执行前缀与配对RNG。
预先固定held8每task state0和25，在630与1260节点四臂都保存同步画面与goal/stage predicates；不能根据成功失败事后换case。
使用现有capture/轨迹保存能力。未实现的阶段谓词不要编造“抓住/放好”等标签；直接保留实际可验证的BDDL谓词和画面。
不要求Luna主观判根因；返回固定case索引，由主任务解释。

## 8. 全部训练与选点结束后的无梯度动作探针

固定每臂selected及Source1000，共5个模型。任务是held8＋seen target16，共24个。
每任务两组：teacher46/query-demo47，teacher48/query-demo49。每条query episode取两个位置：
`p=floor(0.2*(N-2))` 与 `p=floor(0.6*(N-2))`，观测为p，动作从p+1开始，尾部处理仍用canonical规则。
每模型96条query；policy noise由root seed20260923、task ID、组号、位置ordinal确定，各模型完全配对。
只运行actual10-step flow预测，报告前5步与完整50步的真实7维MSE，按task和seen/unseen分开。
无优化、无teacher动作进入Writer、无probe选点。不要新增梯度余弦、Jacobian矩阵或LoRA范数扫描来扩展本批。
动作MSE只辅助定位，不用它取代闭环排名，也不把读到的held8动作回灌任何训练。

## 9. 资源、运行和异常处理

计划输出：`/data0/user/ymdai/ember_runs/conditional_compilation_diagnostics_20260923`。不存在才创建；已存在先查合同/完成状态，不覆盖、不另起重复run。
正式新资产峰值预留data0≤96GiB，开发/运行树data1额外≤2GiB；此为保守上限，launch前按checkpoint、banks、capture的实际实现细化估计。
大资产复用canonical模型、dataset、tokenizer、环境；不复制Source或HDF5。每个训练/评测launch前分别查strg01 user quota与共享容量，不以df代替quota。

已恢复常规GPU限制：两节点总计最多8张，空闲卡总数≤10时上限6张，每节点≤6张；训练、物化、评测和共驻统一计物理卡。双节点live核验；不沿用昨夜无限卡授权。
低利用率、显存充足的卡可共驻；不kill/pause/reset别人的任务。单条DDP留在一个节点，固定NCCL_P2P_DISABLE=1、deferred NCCL与GPU-local NUMA。

推荐优先保证一个video臂4卡训练，另一节点资源用于较轻的A/B或完整评测；若profile证明其它物理划分明显更快，可在不改逻辑更新的前提下采用。具体GPU index、物理microbatch、replicas是按live资源选择的工程项，不是让你选择科研配方。
不要等满卡，不为占卡开dummy进程，不让评测无限积压到最后。C/D也可以在资源满足时分别同节点3卡训练，不能跨节点拼一个训练进程组。

采用已有launcher/tmux等机制，保存exact command/env、实际commit、完整输出和exit状态。不为此新增一套调度框架或监控子代理。
在进度未变时不刷屏；阶段结束、失败、真实阻塞再回报。物理资源不足可等释放或选择合格卡，不能改样本数/节点/源模型来省事。
工程重试保持同一数据与checkpoint合同，保留失败原件；按完整状态resume，不能从新seed或更好的中间点另开一条轨迹。

## 10. 汇总、交付与裁决边界

在study下保留机器可读小文件：

- `registration.json`：spec、active design、实际code commit、允许/禁止数据、launch前读过哪些历史结果。
- `launch/`：实际命令、CPU检查、smoke/profile、资源快照、训练与评测completion。
- `training/<arm>/`：event plan引用、完整checkpoint及训练loss/实际LR/每task曝光。
- `evaluation/<arm>_<step>_{held,seen}/`：contract、rows、aggregate、completion和配对依据。
- `selection/<arm>.json`：六完整节点、唯一selected与freeze时点。
- `evaluation/<C或D>_<selected>_{other,wrong}/`：完整controls。
- `analysis/learning_curve.csv`、`per_task.csv`、`paired_comparisons.json`、`video_controls.json`、`action_probes.csv`、`case_index.json`、`completion.json`。

分析必须机械按预定义口径做，不替主任务决定修复：

1. 每臂六个节点的held400/seen64、四suite、逐task、breadth和相邻retained/gained/lost/churn。
2. 同一更新节点的B−A、C−B、D−C；630与1260作为预定阶段参照，其它节点完整保留。各臂最佳单checkpoint另外列，不用“甲最佳对乙最后”替代整条配对曲线。
3. 同一held面板任务/初态/RNG配对；seen与held任务不同，不能逐行配对，分别报告。
4. 对任务簇做20000次bootstrap、seed20260923、百分位95%区间；8任务/单training seed明确是描述性不确定性，不当等价性检验或大样本独立400试验。
5. C/D各自correct/other/wrong及成功集合交换，联系A/B参照。wrong降低但correct/other不提高，不能写成有益视频修复。
6. 利用原定Source held400/seen64，对每臂六节点分别计算相对Source的retained/gained/lost/churn，逐task、suite及整面板都输出。保留率=retained/Source成功数，获取率=gained/Source失败数；分母为0记NA，必须同时列整数计数。净增益=gained−lost；不能仅凭净分区分“没学会”与“学会后损害已有能力”，也不能将跨不同任务集合的成功率相减称作配对遗忘。将Source能力、Object留出的Writer训练对象支持不足、其它几何/组合留出边界一并列明；不能由某个困难task全零直接归因于模块。此项只汇总已有注册面板，不增加训练/评测。
7. 动作误差和固定行为case只作解释材料；不依据probe选择新checkpoint。

正式科学结论只写“这个固定诊断窗口观察到什么”。不宣布找到唯一根因、不宣布模型上限、不自动启动任何修复训练。
正常更新progress/task_plan；已完成事实进入research_history，跨轮有效结论进入findings。不要新增面向Owner的长报告，最终直接在对话里给分数、关键比较与未解决项。
小型合同/指标/源码按项目Git规则提交main并push，不提交checkpoint、bank、dataset或大视频。主任务稍后统一解释并安排下一批。

完成全部规定项目后，向主讨论任务发送简短完成消息，包含commit、run root、四臂六节点总表、C/D各自selected的correct/other/wrong分数及是否有缺项；如果部分被工程/数据问题阻塞，准确列出，不冒称整批完成。
可以用已经验证的 `codex queue --thread 01a0c8da-0058-7951-8d1c-9c2233079f7e --message '...'` 回传；长内容用文件路径或安全的引用变量。保留双方模型配置，不另建任务、不使用exec resume代发消息。

通知时点仅为：收到并登记、工程ready/正式启动、整批完成或需要科研决定的阻塞。不发送例行心跳，不额外承担仓库大清理。

## 11. 暂停前讨论的解释边界（2026-09-23）

本节不改变冻结spec，也不授权追加实验。A本质上是已有MT-BC类别的匹配参照，不是新方法。
旧MT-BC用过本批diagnostic-held8的动作，不能直接充当本批未见任务参照；差别不以rank为主要辩护。
旧有效rank16 SVD为冻结功能诊断，昨晚自由A/B是C600起点task-local短更新，都不是本批fresh共享fit28学习。

A/B比较包括参数化、条件映射和优化；B/C还改变活动模块/容量；C/D同时改变第二组tau与horizon。
四臂固定数据，因此不能由此独立识别数据构成的因果作用；不能凭终点排名唯一归罪某个头、某条视频路径或某类数据。
应同时看已见任务获取、未见任务迁移、Source相对新增/丢失及correct/other/wrong成功集合。
若未来开展机制验证，应先提出竞争解释和可反驳预测，再单独登记干预；此处提出的端点/前缀拆分、条件学习几何、
定向任务支持对照仅为讨论中的后继思路，均未执行、未授权，不得自动追加到本四臂。
Owner随后停止研究及派发；所有后继工作须由接手session按最新owner授权决定。

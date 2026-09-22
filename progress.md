# EMBER progress

## 2026-09-22 夜间：Owner授权有界根因探索，正式训练前须停下确认

Owner要求集中分析新划分绝对性能与视频特异性，允许约1–2小时探索/小实验，后续时间可至9/23 08:00 CST；稳定性暂降优先级。
现存MT-BC300=155与C600=154继续保留。最新补充指令要求在任何正式训练启动前停下，先报告具体改法、验证证据、预算与方案，
待Owner确认；这条边界覆盖此前较宽的夜间推进授权。目前只有探索授权，没有正式训练授权；Test/FT/RL不启动。

起点main为`a54940c4552180a7e83a619ca89a72ba0ab919a8`，工作区干净。三个只读任务分别核对训练目标/实现、视频到LoRA通路、
MT-BC窗口与数据比较；主agent负责全部写入和GPU启动。首次资源查询修正了本节点不可解析的BCI-GPU别名，实际使用gpu01/gpu02；
快照显示gpu01:0/4空闲，gpu02:1/2/3仅外部148MiB、0% context，可在启动前复核后按共驻合同使用。未启动GPU工作，也不预留卡。
strg01的`xfs_quota`实测data0/data1用量为144,965,576/901,627,152 KiB，soft quota均1,073,741,824 KiB。
新探索预计额外峰值不超过10GiB，输出优先data0，复用模型/数据；具体实验与launch在确定后另记，不能从此快照推定之后设备仍空闲。
当前active exploration：[夜间根因探索](docs/overnight_root_cause_analysis_20260922.md)。首批登记一次旧MT-BC200冻结400评测、
同训练侧面板的LoRA因子有效作用/flow时间轴两项冻结诊断，以及MT-BC有效秩CPU检查；没有正式训练授权。

运行源码冻结在已推送`131ab2f1`的detached `.codex/tmp/overnight-analysis-runtime`，只读symlink复用data/models/runs。
20:04 CST左右，旧MT-BC200在gpu01:0/4启动原旧validation400冻结评估；gpu02:1/2/3共驻分别执行C600/C1200/O1200
的因子诊断，各约147秒完整保存。O1200/C1200 exit0；C600保存完整行但原D1的一个Object2/cohortB/25% query重放
maxabs=.1148，超过预先粗容差.05，故exit1并保留`complete_reference_mismatch`。其64行总体relative L2=.00646，
其余最大差约.00506；不修改阈值、不丢行，也不把此次值替换历史D1。当前同次运行WW的BA相对变化C600/C1200为.2362/.1827，
两项线性因子变化的全局cos为正，动作差额MSE却为.005045/.0002314；主要衰减不支持简单因子相消，进一步按真实flow轨迹定位。
三模型flow检查在前一批进程退出与双节点live复核后启动，仍与旧MTBC合计5卡；记录见study/launch。

M300薄QR/SVD的rank16保留总BA能量.9968455，最低action_in为.8859357，几何结果不足以判断功能；已追加上方合同中的
唯一rank128/有效rank16冻结功能比较（Train32query+16闭环/臂），仍不训练。静态审计与MTBC统计原件分别在study的
flow_time_probe/audit.md、mtbc_comparison/report.md。后者显示新MTBC成熟非峰200–500的6节点均值131.17，旧晚期评估
不能解释为对新曲线同曝光的完整观察，旧200诊断尚待结束。

20:33 CST前，旧MTBC200完整400与有效rank16功能比较均exit0。旧200=102/400，共同五个held任务91/250；旧425为89/400、
共同五任务86/250，新M300为134/250。旧晚期选点确实遗漏较强早期点，但新旧共同五任务差额43中Spatial3占38，不能把差异
全部解释为采点密度，也不能把单个节点补测当成完整旧曲线。M300原rank128与有效rank16在固定训练16闭环各8成功，待配对复核；
保留99.68%权重能量与小面板分数不等价于完整排除Writer共享头的表达限制。

flow三模型检查均完整exit0；C1200在全部被测τ的演示拟合均优于C600，不支持仅噪声端点与其余时间的拟合冲突。
实际采样中视频差额随flow步骤扩大，未出现普遍途中抵消；C600/C1200最终动作差额均值约21.8倍，而中位数仅2.3倍，
前者主要由少数夹爪动作分歧贡献。不能继续把二十余倍均值概括成整条视频路径塌缩。因子诊断单query旧参考偏差如实保留。

第二批已在active exploration登记P/J/F各训练侧72闭环+144query：P冻结C600，J原目标，F保留辅助7的采样与权重但改为
完整50步随机flow时间FM。J/F只有54固定事件更新，不是正式训练，不产生正式选点；正式训练仍须先向Owner汇报并等待确认。

20:37 CST左右P/J/F已启动：P=gpu01:4，J/F=gpu02:2/3，共3卡，单臂90分钟上限；原第一批进程全部退出。
启动前quota为data0/data1 144,993,884/901,883,160 KiB，study仅15MiB，新增≤10GiB；选定两张共驻卡均只有148MiB、0% context。
CPU接口核验显示F/J辅助调用唯一差异为noise_endpoint/prefix_steps，原backward复用；601–654重放曝光22tasks×6、7×5、7×7。

额外比较审计确认旧O1500与C600的共同held5正确成绩148/130，旧早期MTBC200与新300为91/134；这些是不同配方的描述比较。
Spatial3绝对正确35→41、Object1 42→33，故绝对能力退化与视频增量减弱不是同一任务模式。原wrong对照的donor发生变化：
Spatial3 old13/new11，Goal6 old32/new39，Long1 old1/new3；Object1/Goal3映射相同。已登记唯一Spatial3冻结交叉补核，
不训练、无held动作、无checkpoint/架构反馈；不能在核验前把旧新wrong差直接当成模型变化的因果量。

20:58 CST左右Spatial3两格冻结补核在gpu01:0/2启动，各50条、30分钟硬上限，总占用5卡。设备live身份/共驻与quota均复核，
脚本CPU检查确认所有100条RGB采样/原state-video/RNG映射；逐臂完成后再统一读分。P/J/F继续原54更新，不追加或按部分成绩修改。
可读的第一批分析与两张图暂存在study/analysis/report.md（明确标记后续尚在运行）；其中新增CPU联合输出空间计算表明M300最优
共享216维空间保留q/v有效BA能量99.687%/99.939%，仍只是几何，不能证明当前Writer能实现或执行同等策略。

Owner两次纠正代理对“固定LoRA”的误读：这是EMBER理想模型类的退化解/理论可达下界论证，绝非要求先学公共能力、
预训练公共底座后再加视频、或建立分阶段训练。已撤回该表述并更新稳定需求；同面板M300只是测量参照，不改变联合端到端目标。
21:08左右M300训练侧72条冻结参照在P释放的gpu01:4启动，上限30分钟，无额外训练；总占用仍≤5卡。
Spatial3交叉补核两臂已完整退出0：O1500在donor13/11均0/50，C600分别40/45，correct为41。
两种wrong来源的差异不足以解释该任务旧新模型的40/45条差额；这仅校正历史比较解释，不据此选模型或改架构。

21:31 CST前P/J/F与M300参照全部完整exit0；每臂72条唯一闭环，P/J/F各144probe，J/F各54更新、216事件。
逐事件teacher、main/aux episode/frame与policy RNG，以及逐闭环language/env/policy RNG共同前缀核对通过。
P/J/F/M300全36任务成功分别39/46/44/44，目标24为18/23/23/21（各48），辅助12为21/23/21/23（各24）。
F−J为−2/72，R/G/L=36/8/10，任务簇95%区间[−15.28,+9.72]pp；目标24净0，仅Spatial对J为正。
未满足预登记明显改进条件；J也有Spatial−4、Long+5的交换，不改选J作为正式候选。当前不建议基于该干预启动正式训练。
P共同八任务重放为3/16，旧D1为2/16，差异仅Object2/state0；配对随机流未变，原件保留。M300为8/16，与既有检查一致。
训练记录同时确认主21查询已经来自21条不同episode，216事件均无teacher重叠；辅助7来自另一条episode的7位置。

本轮新增闭环820条，诊断更新108次；旧MT-BC400、rank32、P/J/F216、M30072、wrong donor100均完整正常退出。
3个flow资产完整；因子C600的单query旧参考偏差仍为exit1例外，不称全部检查无例外通过。没有canonical源码改动、Test、FT或RL。
Owner再次明确视频特异性用于检验正确视频经Writer改善闭环的机制链，非独立刷高差值；已纳入稳定需求和报告。
已只读审阅既有四对行为画面与谓词时间线，记录抓取后衔接及子目标保持的不同失败，不据选例定位唯一训练根因。
原始表、脚本与图保留在上述study，按Owner最新要求直接对话汇报，不新增专门报告或远程复核包。
已登记两批探索完成但总体性能目标未达成，尚无已验证修复；不自动启动正式训练。

21:46 CST后登记唯一补充检查：用原D1固定Train8 [5,7,12,14,20,25,34,37]，各从相同C600完整状态复制，
只保留本task在events601–654的原J梯度，其余新梯度为零；全部54个AdamW step和固定LR不跳过，保留动量与decay。
再设Z分支54步全零新梯度。独立task副本各评本task state0/1、teacher46/48（合计16条），Z评8task×2（16条），
复用现有P/J配对参照，不新增held/动作probe/选点；所有分支完成后统一分析。新增总32闭环，预计四卡10–20分钟，上限30分钟。
保存实际梯度事件、参数位移、生成LoRA、compact轨迹和退出；不保存8份完整checkpoint、不拼成部署字典。
它识别其它任务新梯度被移除后的整体影响，不能单独区分步幅、共享容量或更新方向，不直接批准PCGrad或formal训练。
源码为study/task_interference_probe一次性脚本，复用冻结131ab2f1生产SupervisedEngine；源模型始终冻结。
21:46双节点实测gpu01:4空闲，gpu02:1/2/3为148MiB低负载context可共驻；启动仍核验占用，总量≤4/6卡。
data0/data1 quota用量145,342,072/901,883,228 KiB，study340MiB，新增预计≤1GiB，原≤10GiB预算保持。

## 2026-09-22：O1200/C600 既有执行轨迹专家失败定位材料

专家要求停止新增训练/梯度/模块消融，只读取既有 D1 `O1200/CC` 与 `C600/CC` 真实执行，以便定位最早动作错误。按事后
固定规则（每 suite 中 `task_id`、`init_state_id` 升序的第一个 O1200 success/C600 failure 且同 teacher 条件）选择
Spatial 5/1、Object 2/0、Goal 0/1、Long 4/1；它们不是新的 success-rate 样本。原件均为 compact capture，故以保存的
normalized action chunk 的实际五步执行前缀，经 source normalization 反归一化后在原环境重放，未加载或调用 Writer、
source policy、checkpoint、action sampler 或模型 forward。gpu01:4 的一次 EGL 只读重放完成后，8/8 trajectory 的每个
保存 replan Pi05 state（最大绝对误差0）、success/steps和全部 stage-predicate transitions 都与原始行一致。

[远程行为复核包](docs/review_materials/20260922/writer_causal_diagnostics/behavior_pairs/README.md)包含四个同步并排执行视频、
2,035 条逐控制步 action/state、完整 predicate timeline、原始 predicate transition、逐轨迹 fidelity 表、原始选中行，
以及同一 action-hidden teacher 的轻量视频/contact sheet。预保存 full RGB 的 global5/state0、global37/state1 都因
O1200 本身失败而明确排除，不被误作正反比较。材料仅供专家判定错误类别；不在此归因性能下降或启动后继实验。

## 2026-09-22：Writer 因果路径与辅助梯度诊断 D1/D2/D3 完整交付

专家后续的有界 D1/D2/D3 已结束，未启动 fresh 训练、正式 Validation/Test、FT、RL 或外部比较。最终
`completion.json` 为 `status=complete`、`d3.status=complete`；canonical 原始表有544条路径 probe、112条路径闭环、
16条虚拟 AdamW 候选、54条微训练更新、192条微训练 probe 与48条微训练闭环。最终 retry6 的 controller、D2 C600、
D2 O1200、finalize 及 detached checkout 都 exit=0；D1/D3 已在 retry5 全零退出完成，retry6 只重放 D2/aggregate。

D1 的 Procedure donor replacement 在 O1200/C600 造成比 Core donor replacement 更大的动作/Compiler/LoRA 改变，说明
两条路径都被真实消费者使用；但 C600 的 CC/CO/CW/WC/WW 训练侧面板为2/5/3/5/4，不能主张正确视频内容优于其它路径。
D2 的全 Writer `||g_A||/||g_Q||` 在 C600/O1200 的 B4/B36 为.978/.728/.890/.526，且全局cos为.113/.192/.243/.286；
所有 J preclip norm<1。D3 从C600各18步的CC闭环为 J=6/16 breadth4、Q=5/16 breadth5、M=5/16 breadth4，success set
不同，不能选择统一优胜臂。完整表格、逐task闭环、完整性与成本已推送为远程
[专家审计包](docs/review_materials/20260922/writer_causal_diagnostics/README.md)；canonical 本地报告仍在
`/data0/user/ymdai/ember_runs/writer_causal_diagnostics_20260922/analysis/causal_diagnostics_report.md`。登记 stage-worker
成本10,478.47秒，累计有效launch墙钟82.6分钟，低于120分钟预登记上限。

早期 asset-root、encoder、旧VJP等价性与D2 CSV字段问题的 exit、stderr 和无效原件均被保留。旧 VJP 问题已修为直接记录
native `g_J`、`g_Q` 和参数空间 `g_A=g_J-g_Q`；一事件残差`7.276e-12`。本诊断仅提供路径和辅助项整体的定位证据，
不恢复视频内容/顺序特异性主张、不定位唯一历史根因，也不自动授权纯主FM、LR/seed/rank/head补救或后继训练。

## 2026-09-22：Writer 因果路径与辅助梯度诊断中间记录（已由上方完整交付取代）

Owner 转交专家后续方案后，已把它收敛为
[有界 D1/D2/D3 合同](docs/writer_causal_diagnostics_20260922.md)：D1 用八个训练 task 的 CC/CW/WC/WW/CO 固定路径
比较；D2 用 C600 原始 events604..612 分离 `g_Q`、已含1/3的`g_A`、AdamW J/Q/M/Z；D3 仅在完整 preflight 和总
120分钟 GPU 预算可覆盖时，从 C600 做 J/Q/M 各18更新及终点测量。当前仅创建隔离实现 worktree，未启动任何 GPU
进程、训练、正式 Validation/Test、FT、RL 或外部比较，也未读取部分性能结果。后续结果只会在所有预登记行、manifest、
worker completion 与 exit 核对后写入本文件和 findings。一次性 live GPU/配额预检已完成：gpu01:0 空闲；gpu02:1--3
为0% util、约148 MiB外部 context 的短时共驻，均有约45 GiB余量；data0/data1 quota 已用分别142,199,872/
902,103,016 KiB，soft quota 均1,073,741,824 KiB，额外峰值登记为不超过8 GiB。launch 起连续120分钟绝对截止，
D3 只在 D1/D2 全部成功且至少剩余65分钟时三臂整体启动；该判断只取时间和完整性，不读取任何部分分数。

首次 controller 在任何 D1 probe 或 rollout 前以 exit=1 停止；四个首批 worker 都报 `task authority changed: 0`，没有生成
性能行、rollout、D2/D3 或 `completion.json`。原因已由最小复现验证：runner 把 detached code runtime 当 asset root，而
该 worktree 不含 canonical `runs` authority 链接。修复仅把 code root 与只读 canonical asset root 分离；`current_training_data`
在 canonical root 的36任务直接重放成功，9项相关单测通过，architecture guard=PASS。初始 launch 的 exit/stderr/preflight
均保留为工程故障证据；将从新的 clean pushed detached runtime 重新开始同一注册合同，不改变任务、资产、预算或科学变量。

后续 retry 已完整写出 D1：函数 probe 共544条，闭环面板共112条，所有对应 worker exit=0；这里只核对行数、manifest
与 exit，尚未读取或汇总性能值。retry4 的两个 D2 worker 都在生成任何 gradient/virtual-update 行之前因同一旧完整性
断言退出。最小复现显示旧断言比较了“分别 main/teaching Writer VJP 后相加”和“先合并 LoRA cotangent、以 native BF16
做一次 joint Writer VJP”；C600 的相对L2差异约0.76%，来自 cotangent cast/reduction 位置，而非任务、事件或冻结资产。
因此修正为 native `g_J` 与 `g_Q` 的参数空间分解 `g_A=g_J-g_Q`，并令 J/M 用直接 `g_J`；这是对原训练计算图的忠实
重放，不改变辅助目标的科学定义。D1 原始行和所有失败 launch 原件均保留；D2/D3 会从新的 clean pushed detached runtime
续跑，D3仍只由完整性和预登记时间门控决定，不能依据D1/D2的部分分数启动或取消。

native-J 版 retry5 已完成：D2 C600/O1200、D3 J/Q/M、finalize 与 controller 都 exit=0，`completion.json` 登记
544/112 D1、16 virtual updates、54 micro updates、192 micro probes、48 micro rollouts。分析时发现 D2 的 group 记录中
`joint_grad_norm` 被同名 draw-level 全 Writer 指标覆盖；这是 CSV 合并错误，不影响任何梯度、AdamW、D3或闭环计算。
retry5 原始 D2/final 载荷会以明确原因保留；修复后仅重放两条 D2并重建最终 aggregate，不重算已经完整的D1/D3。

## 2026-09-22：辅助配对正式结果专家复核包

按Owner要求，已从sealed正式产物机械导出并推送[轻量复核包](docs/review_materials/20260922/auxiliary_pairing_fresh/README.md)，不重跑视频、GPU诊断或训练。包中保留六个correct400节点2400条逐行结果、C600 other/wrong各400条严格配对行、C600与same-video/MT-BC300/Source各400条配对行、11个面板的rows/manifest/worker完整性、1200步训练主/辅助loss、LR和既有梯度统计、36任务exposure汇总、成本及合同/停止/选点/C600完整resume资产索引。

C600相对MT-BC300的逐行配对为共同成功111、C600-only 43、MT-BC-only 44；相对wrong为共同成功111、correct-only 43、wrong-only 42。所有配对task/state、language、environment seed与policy seed root一致；因不同成功时点而长度不同的policy-noise序列均在共同前缀一致。正式manifest未记录materialization elapsed，包中显式标作`not_recorded`。这只是供专家判断能力共享、交换和后继可行性的材料，不新增科学结论或后继授权。

## 2026-09-22：36任务 Writer 辅助 episode 配对 fresh 对照正式完成

本轮唯一变量为`data.teaching_episode: same_video → cross_episode`，其余Source-71、normalization、36任务、
K=1、38-target rank16 A/B、fresh Writer/三Meta、每更新4任务、每task主21＋辅助7、`tau=1`前5步/`1/3`、seed、
采样、优化器和LR时钟均不变。PR #3已合入`main`；它删除平行status/selection路径，复用coverage controller。真实
coverage manifest与`exposures.jsonl`的1800更新、7200条件preflight通过teacher/主21、辅助噪声、teacher排除、未来5动作、
无有放回及历史曝光逐行核验；该检查只证明事件合同。

正式运行来自clean pushed detached runtime `64947492`。03:10 CST资源/配额preflight后，controller在gpu02:0--3
（已知`gqma`小显存共驻、world4/frame8）完成至预登记上限1200；没有启动纯主FM、Test、FT、RL或外部比较。controller
exit=0，六个correct和两个分支评测面板均有400条唯一Validation行、正确manifest、固定配对映射与12个worker全零退出。
correct400为`151, 120, 154, 137, 129, 124`（step200..1200）；早停规则在1200以`sustained_decline`首次触发，故原
early-stop选点与extended-run选点都为step600=154，post-stop extension为空。

600的same-task-other=160、cross-suite-wrong=153；154处于118..155分支，因而不执行shuffle/reverse。全部同节点相对
same-video为`+41,+28,+39,+50,+12,+44`，但600仍比冻结MT-BC300=155低1，且相邻R/G/L为84/36/67、80/74/40、
101/36/53、107/22/30、98/26/31，未建立稳定保持。选中600相对same-video=115的R/G/L=78/76/37、J=.4084，
Spatial净+45而Goal/Object净−8/−7。视频对照也不支持恢复性主张：correct−wrong=+1，correct−other=−6，且无顺序
controls。结论严格限为本合同下跨episode**辅助配对**相对same-video的性能改善；不证明辅助项整体或其tau/前缀/权重，
也不支持视频内容/顺序特异性恢复。

正式原件、完整性汇总、CSV/JSON和图表在
`/data0/user/ymdai/ember_runs/coverage_retraining_cross_episode_aux_20260922/analysis/report/`；训练loop为11720.8秒，
六个correct评测5228.9秒、含两个后继面板6994.0秒，物化3200条件但manifest未记录elapsed。本轮已结束，当前无运行中
EMBER任务或active design；不自动继续训练、调参或启动后继实验。

## Writer输出空间与code投影诊断完整交付（2026-09-21）

Owner要求执行专家最新方案。Active design为[Writer输出空间投影诊断](docs/writer_output_space_projection_design.md)：
A0读取O1200/N1000/N1800三份checkpoint；A1在global5/7/12/37的demo46/48完成16次无梯度编译；A2复用
held global3/11/26/31、state0..3和既有O1200参照，对SELF/NEWSPACE/SHRINK各运行16条冻结闭环。
不训练、不反向传播，不启动Test/FT/RL/完整E2或拆head后继。

A0三checkpoint、A1两模型各8次编译、A2三臂各16条闭环均已完整exit0；无Test、反向传播、optimizer更新或
checkpoint选择。O1200/SELF/NEWSPACE/SHRINK成功为11/11/11/9；相对O1200的R/G/L分别为10/1/1、
8/3/3、8/1/3。NEWSPACE将q-B有效更新范数平均保留69.5%、方向`rho`改变量0.719，v-B分别为96.3%和
0.267；总成功仍与O1200相同。SHRINK匹配逐层范数却降至9，故本轮不支持新B空间删除旧有效方向的解释。

A1初版发现并修复明确工程错误：hook调用遵循词典序tensor specs，却被误当成数值layer顺序。旧A1完整封存，
A0/A2不受影响；修复提交`36e7f791`已推送并从clean detached runtime重跑，`C @ code`重建relative-L2最大
0.1745%，两模型正式stderr为空。N1000的rank-centered code能量约0.8%–0.95%，O1200约5.35%；实际B/BA
也更单方向化，只作为系数映射候选定位，不自动授权拆head或新训练。

[图文报告和轻量原始统计](docs/review_materials/20260921/writer_output_space_diagnostics/report.md)已生成；正式study保留
code safetensors、48条compact轨迹、launch合同与无效A1事故证据。当前没有运行中实验，按专家合同停止等待裁决。

## 当前状态：Writer低学习率修复已由Owner停止（2026-09-21）

Owner判断性能已经无法提升并明确要求停止。唯一低学习率phase已终止，双节点均无本实验训练、物化或评测进程；没有启动Test、FT、RL、外部比较或其它补救分支。正式完整correct Validation400节点为：1900=101、2000=92、2100=95、2200=98、2300=101。phase最高101/400，未严格超过原N1000的117/400，因此按冻结资格规则保留N1000，不执行same-task-other或cross-suite-wrong。

训练已完成到global2400并保留完整checkpoint，但Owner停止时对应Validation仍在执行，未形成`results.json`或`launcher_completion.json`，所以正式证据截止global2300。中断竞态使stage/controller的exit文件写成0；该exit不能覆盖正式产物完整性要求，global2400不得作为完整评测节点或科学结果。停止事实与边界记录于study的`launch/owner_stop_20260921.json`，选点记录为`writer_low_lr_selection.json`。

图文报告与原始统计材料位于[Writer低学习率修复报告](docs/review_materials/20260921/writer_low_lr_repair/report.md)，包括完整曲线、逐任务、逐suite、相邻R/G/L、选中correct失败行、训练成本、精简原始rows、选点和Owner停止记录。phase共执行600次更新（1801..2400），累计67200 queries；训练循环合计6400.99秒，峰值reserved 22.803 GiB/卡。本实验结论是固定低学习率续训未恢复到N1000水平，属于科学负结果，不按工程故障继续补救。

## Writer稳定性修订诊断完整交付（2026-09-21）

Owner要求阅读最新专家意见并按修正方案继续。复用已完成E0/E1，原E2及其不完整文件正式取消；
O-H/O-L/N-H/N-L四臂各36更新、每臂36条终点compact probe、每臂16条四任务闭环，以及既有E1
原件的16条轻量轨迹均已完成。controller、四个train和四个rollout全部exit0；两节点已核对无本诊断进程。

修订合同冻结当前36任务事件1801..1836；高/低LR分别为`1.6279650115e-4`/`2.959936e-5`。终点闭环
固定global 3/11/26/31、state0..3、teacher demo0..3；每分支只有Spatial3-state0与Long1-state0保存
完整图像，其余行只保存state/action与谓词。实现将在既有official `rollout_shard`的诊断捕获接口上增加
向后兼容compact模式，不复制评测循环。正式执行来自clean pushed commit `b75ab08e`；启动前data0/data1独立
quota满足额外2GiB峰值，因gpu01:0/1被他人新任务占用，live分配调整为gpu01:2/4与gpu02:0/4，四臂仍两节点
各两卡、逻辑合同不变。四臂各36 steps、144 task events、36 probes、16 rollouts全部验收；合计144/576/144/64，
四周期平衡且仅8条终点轨迹含图像，另16条E1轻量视频含5536个实际执行动作记录。

四任务小面板的父节点/高LR/低LR成功数为：O1200 11/16、O-H 11、O-L 10；N1800 8/16、N-H 7、
N-L 10。终点36任务probe的mean actual-flow MSE为O-H .054112、O-L .053502、N-H .052721、N-L .050463；
这些只是登记面板事实，不作因果归因或分支选择。脱敏原始包、8条全图像轨迹索引和16条轻量视频位于
`docs/review_materials/20260921/writer_stability_diagnostics/corrected`。Test/FT/RL继续暂停等待Owner/专家裁决。

## Writer训练稳定性诊断启动（2026-09-21）

Owner要求完整执行专家最后登记的E0–E3，不采用对话中较早且已撤回的续训或补救建议。Active design为
[Writer训练稳定性诊断](docs/writer_stability_diagnostics_design.md)。已一次性核对O1200、O1500、N1000、
N1800四个Writer checkpoint均含完整51,033,768-byte权重、约97MB trainer state与对应2/4-rank RNG文件；
O1200/O1500/N1000/N1800的optimizer/scheduler cursor分别为1200/1500/1000/1800，Writer AdamW均为
545个参数状态。M300和S1000正式资产也存在。旧训练frame chunk16/world2，新训练frame chunk8/world4；
诊断更新将恢复各父节点独立Adam历史，但用统一当前36任务事件。新Test、FT、RL和外部比较继续暂停。

实现隔离在`codex/writer-stability-diagnostics`；正式GPU执行前仍需clean pushed frozen runtime、双节点live
GPU检查及data0/data1独立quota检查。运行根预定为
`/data0/user/ymdai/ember_runs/writer_stability_diagnostics_20260921`，尚未启动GPU任务。

首个E0启动在任何模型forward前因工程路径错误exit1：新detached runtime没有canonical主仓库的`runs`资产
链接，而诊断入口误把代码runtime当作asset root。stderr与初始exit保留；没有生成`e0_integrity.json`，GPU已释放，
checkpoint未写入。修复只把代码runtime与canonical只读asset root分离，不改变任务、节点、梯度或优化合同。
修复后E0在gpu01:2完整exit0：O1200/N1800的main与teaching四条完整Writer梯度均finite，两个父节点分别
恢复545/545个AdamW状态，Adam step为1200/1800；原生frame chunk16/8及原checkpoint world2/4均与
资产一致，正式stderr为空。E0没有更新参数，准许按固定合同进入E1。

E1已完整结束。E1-A六模型各288条固定动作探针全部exit0，共1728行；合并表逐条件加入相对Source的
10步flow前5步动作差值。E1-B先以Source的52条正式批次验证闭环、轨迹与阶段谓词链，再在五张物理卡上
并行其余模型；六模型共280条、280个轨迹，全部exit0、stderr为空，成功标记与最终目标谓词一致。首次
Source闭环在任何episode前因未设置sealed LIBERO assets环境exit1，故障栈与旧exit保留；修复只补运行资产
authority，没有改变面板、状态、视频、RNG或模型。E1合并表、轨迹索引与事实汇总位于正式study根。

E2在E1完整验收后启动，同用当前36任务事件1801..1809及固定`1.6279650115e-4`。首次两父节点单进程
执行被一次错误的Gram瓶颈判断人为中断；事后文件时间戳证明Gram只用约25秒，该判断及部分输出已作为
无效尝试封存，没有改变科学合同。随后按E1-A实测约6分钟/完整288行探针，将每父节点21个彼此独立的
更新分成3个shard，六张卡并行且每shard仍重算同一组完整任务梯度。

Owner认为剩余评测时间不可接受并明确要求暂时停止后，六个E2 shard均已中断，双节点已核对没有本诊断
E2进程。E2没有形成完整shard集合，不能作为科学结果；E3及下游均未启动。暂停记录在正式study的
`launch/owner_pause_20260921.json`。若恢复且合同不变，E2须从两个原父checkpoint重新运行全部登记shard；
若要缩减规模，必须先由Owner裁决新的诊断范围，不能把现有部分文件冒充完成结果。

Owner要求先把已有结果交给专家。已在
`docs/review_materials/20260921/writer_stability_diagnostics`整理E0/E1脱敏结果包：1728条动作探针、280条
闭环原始行、逐状态配对表、280条轨迹索引、汇总CSV/JSON、PNG/SVG及绘图/导出源码。checkpoint、轨迹
张量、主机绝对路径和未完成E2部分文件均未进入Git；报告明确小闭环不是Validation400，根因在E2/E3暂停后
仍未解决，并附需要专家裁决的缩减问题。

## 远端差异审计材料补交（2026-09-21）

Owner要求把专家分析旧/新Writer差异所需及相邻材料推送远端。已从现有正式原件只读导出[诊断包](docs/review_materials/20260921/coverage_retraining/diagnostics/README.md)：旧Writer5、新Writer9、新MT-BC10及Source节点和已完成的视频对照共34个Validation400面板的13,600行精简成功标记；逐节点逐任务、相邻成功集合、旧1200/新1800五个共同held任务配对；旧/新Writer3,900更新指标及15,600任务曝光、按step/occurrence分组汇总、训练/评测Git与轮转位置；新MT-BC500更新指标。导出脚本和数据口径一并保留。完整checkpoint、视频与原始run contracts仍在study，不复制进Git。该补交不改科学资格裁决：新Test、FT、RL和外部比较仍等待Owner决定。

## Writer冻结与MT-BC续行（2026-09-21）

两条训练与五个Writer Validation视频臂结束后，核对gpu01/gpu02无本轮runtime进程，五个clean detached frozen worktree已用`git worktree remove`清理；Git现只保留main worktree，运行代码仍由已记录并已推送的commit追溯。selected Writer1000的other/wrong/shuffled/reversed四个bank各400个可再生LoRA payload退休，合计名义8,257,888,000字节，逐bank`payload_retirement.json`记录完整checkpoint及保留物；manifest、正式400行结果、worker日志与checkpoint未删。最终停止节点Writer1800 bank仍按原保留合同持有payload。`.codex/tmp`其它约402MiB内容未确认归属，保持不动。两模型训练/Validation阶段图文报告和原始CSV/JSON已推送main；Writer内容特异性不清楚的新Test暂停裁决未变。

MT-BC step500完整correct Validation400、15个worker与全训练控制器均exit0：119/400；最近400/450/500为130/138/119，正式持续下降规则`stop=true`，没有再启动新段。10个完整节点依次82/112/99/133/136/155/131/130/138/119；唯一correct最高step300为155，已写`mtbc_selection.json`并冻结checkpoint，MT-BC同分规则无需触发。450→500配对保留/获得/丢失102/17/36、churn53、覆盖5→5。Source→选定MT-BC300为51→155、保留/获得/丢失36/119/15、覆盖4→6。与Writer选中1000的117严格配对，以MT-BC为参考Writer保留/获得/丢失87/30/68、覆盖6→4，Writer−MT-BC为−38/400；8任务簇bootstrap95%百分位区间[−29.75,+6.5]pp。此仍是Validation，不替代Test的+40门槛；Writer视频内容资格待Owner裁决，新Test及FT/RL/外部比较均未启动。MT-BC十节点CSV、完整双方法曲线PNG/SVG/源码、逐任务/逐suite CSV及成本JSON已汇入study/analysis和阶段报告。500次更新最大rank步时合计30349.24秒、累计288000查询；五卡阶段均值约31.3秒/更新，逻辑36×16保持。

MT-BC step450五卡完整correct Validation400和15个worker exit0：138/400；当前最佳仍step300的155，正式早停`stop=false`。最近350/400/450为131/130/138，出现回升、线性斜率为正，不能因连续三个低于最佳就误触发持续下降。400→450配对保留/获得/丢失98/40/32、churn72，覆盖5→5。五卡401–450平均31.342秒/更新、峰值reserved32.934GiB，36任务×16查询不变，累计259200查询。控制器继续原合同，无手工重复launch。

MT-BC step400五卡完整correct Validation400和15个worker exit0：130/400，当前最佳仍step300的155，正式早停`stop=false`。350→400配对保留/获得/丢失96/34/35、churn69，覆盖7→5；step350/400均低于最佳，但登记的持续下降规则要求最近3个完整节点，故控制器继续。五卡351–400平均31.324秒/更新、峰值reserved32.934GiB，36任务×16查询不变，累计查询230400。没有按两点下降提前停止或调参。

MT-BC step350五卡完整correct Validation400与15个worker均exit0：131/400；相对当前最佳step300的155下降24，正式早停仍`stop=false`，最近最佳未刷新窗口只有一个节点。300→350严格配对保留/获得/丢失95/36/60、churn96，覆盖6→7；分数下降同时覆盖扩展，如实保留。五卡301–350平均31.339秒/更新、峰值reserved32.934GiB/卡，36任务各16查询和累计201600查询连续。控制器按原登记规则继续，不以一次下降自行停训或调参。

MT-BC step300在gpu01五卡续行及完整Validation400均退出0，15个worker全部0退出：155/400，刷新step250的136，正式早停`stop=false`、最佳300。250→300配对保留/获得/丢失111/44/25、churn69，覆盖6→6。Writer冻结1000 correct117相对当前MT-BC300为−38/400；严格配对以MT-BC为参考，保留/Writer获得/Writer丢失87/30/68，覆盖6→4。此为Validation阶段的当前最佳对比，不把MT-BC300提前选成最终模型，也不替代未来Test硬门槛。五卡251–300每步仍36任务各16查询，平均31.369秒/更新、峰值reserved32.934GiB/卡；先前双卡201–250为75.303秒/更新，实测约2.40倍更新吞吐。正式Validation评测墙钟由双卡250的1640.64秒降至五卡300的701.37秒，约2.34倍吞吐。控制器继续登记的下一完整节点；不因Writer差距自行调配方或提前停训。

MT-BC step250完整correct Validation400与6个worker均exit0：136/400，刷新step200的133，正式早停`stop=false`，最佳为250。200→250严格配对保留/获得/丢失94/42/39、churn81，覆盖仍为6/8；Source→MT-BC250为51→136、保留/获得/丢失24/112/27、覆盖4→6。双卡201–250平均75.303秒/更新、峰值reserved32.934GiB，全部更新仍36任务各16查询，累计查询在step250为144000。Writer对照完成并释放gpu02后，MT-BC唯一控制器在step300的live分卡快照选择gpu01:0,1,2,5,6共5卡；已登记为物理拓扑切换，需在完整step300节点再核验真实吞吐与评测，不以分卡数本身宣称加速。Writer科学资格问题不改变MT-BC早停合同；新Test仍暂停待Owner裁决。

选中Writer1000的五个完整Validation视频臂correct/other/wrong/shuffled/reversed已全部退出0、各400行、各12个worker退出0：117/119/107/86/62。correct→wrong保留/获得/丢失88/19/29；wrong−correct为−2.5pp、8任务簇bootstrap95%百分位区间[−6.5,+0.5]pp。correct→shuffled为64/22/53、−7.75pp、区间[−16.5,+0.51]pp；correct→reversed为52/10/65、−13.75pp、区间[−24.75,−3.5]pp。完整manifest确认wrong8个donor均跨suite，shuffle/reverse各400条件重排真实RGB并完整forward；后两臂与correct逐行同teacher demo。时间倒序敏感性明确，错误内容的必要增量不清楚，这是科学资格问题而非已证实工程故障。按Owner边界暂停新Test及FT/RL/外部比较并请求裁决；MT-BC唯一控制器继续完成原授权训练/完整Validation，不因Writer结果停训。正式比较、bootstrap与图文报告在新study/analysis及`docs/review_materials/20260921/coverage_retraining`。

选中Writer1000的same-task-other Validation400已正式退出0：119/400，12个worker全部0退出；correct为117/400。两臂各覆盖8任务×50无放回视频，逐行400个other视频均不同于对应correct。正式比较器核对400配对行，保留/获得/丢失100/19/17、churn36，覆盖同为4/8，suite依次Long27→26、Goal39→43、Object43→45、Spatial8→5。未出现严重same-task换视频下降，已登记`writer_selected_00001000_other_qualification.json`，仅准许继续冻结的Validation wrong/shuffled/reversed诊断；特异性尚待这些完整结果，不因此启动Test。三项对照在gpu02独立控制器顺序执行，按每次live双节点GPU资源选卡，与MT-BC后台控制器并行。首次controls launcher的Bash局部变量在赋值前展开，GPU预检前exit1；已保留旧exit/故障JSON并原子修复，重新启动的唯一session已经进入wrong物化，科学合同未变。

已从正式完整面板导出选中Writer的九节点CSV、可复用SVG/PNG与绘图源码、实测训练循环成本和逐任务/逐suite配对CSV，原件在新study/analysis。选中Writer1000相对Source Validation为117对51/400，严格配对保留/获得/丢失36/81/15，覆盖同为4/8任务。新增成功主要集中在Object1（1→43）、Long1（5→27），Spatial3为0→8；Goal6为42→39、Spatial6为3→0，Long9、Goal3、Object6仍是0。Goal6的BDDL实际为cream cheese On bowl，不能从语言中的“in”推断训练已覆盖In。Writer1800更新循环实测合计17684.20秒、均值9.825秒/步、峰值reserved22.803GiB；不包含物化、评测与阶段开销。冻结Writer的Test correct声明已经用新eval runtime完成只读元数据准入校验，尚未启动Test。

Writer step1800完整correct Validation400及控制器均退出0，92/400。1600→1800严格配对保留/获得/丢失82/10/29、churn39、覆盖5→5；最近完整节点满足登记的持续下降规则，控制器正式`stop=true, reason=sustained_decline`，没有再启动下一段。9个完整节点依次为110、92、115、87、117、80、97、111、92；唯一最高为step1000的117，已写`writer_selection.json`与`method_freeze.json`，无并列other破同分。选中step1000的same-task-other Validation400正在gpu02用0/1/2/4物化及评测；对应原始correct400与checkpoint均保留。首个对照launcher因冻结runtime提交号抄漏四位，在任何物化/评测前exit1；保留旧exit及故障JSON，原子修正后重新启动同一checkpoint、seed、映射与评测合同。此次工程错误不改变科学结果。

MT-BC step200完整correct Validation400退出0，133/400，刷新最佳且正式`stop=false`；150→200严格配对保留/获得/丢失77/56/22、churn78、覆盖6→6。唯一控制器已自动进入step250，当前两卡阶段不为重新分卡而中断；下一阶段按live卡况重新选择。单卡首50更新平均148.902秒/步，双卡51–150更新约75.3–75.4秒/步，约1.98倍更新吞吐。本轮Writer已结束，暂无继续调整其物理训练配置的收益；释放的gpu02四卡立即用于选中Writer的Validation视频对照。对照及MT-BC均按完整exit和400行结果裁决，不读取部分成功率。

## 吞吐续行与Source完整节点（2026-09-20）

Writer step1600完整correct Validation400退出0，111/400；仍低于历史最佳1000的117，但1200/1400/1600依次80/97/111呈上升，正式早停`stop=false`。1400→1600配对保留/获得/丢失64/47/33、churn80、任务覆盖5→5，原始JSON保留。不得只因连续三点未刷新最佳而忽略斜率条件提前停训。

MT-BC step150完整correct Validation400退出0，99/400；正式早停`stop=false`，最佳仍为step100的112。100→150配对保留/获得/丢失73/26/39、churn65、任务覆盖5→6，原始JSON保留。双卡101–150更新平均75.432秒/步、峰值reserved32.934GiB，全部50步累计查询均为step×576，与51–100的75.331秒/步相近；没有为一次分数下降改优化配置。

Writer step1400完整correct Validation400退出0，97/400；高于1200的80，仍低于历史最佳1000的117，正式早停`stop=false`。1200→1400配对保留/获得/丢失51/46/29、churn75、任务覆盖5→5，比较JSON保留在新study/analysis。最近未改善窗口只有两个点且出现回升，不触发登记规则；控制器继续。

Writer step1200完整correct Validation400退出0，80/400；正式早停`stop=false`，当前最佳仍为step1000的117。1000→1200配对保留/获得/丢失65/15/52、churn67，任务覆盖4→5，原始JSON在新study/analysis。此为刷新高点后的第一个下降节点，不自行提前停止或按结果改训练合同。

MT-BC step100双卡续行及完整Validation400退出0：112/400，刷新step50的82，正式早停`stop=false`、当前最佳100。50→100配对保留/获得/丢失58/54/24、churn78、任务覆盖5→5；Source→MT-BC100为51→112、27/85/24，原始JSON在新study/analysis。双卡50更新平均75.331秒/步、峰值reserved32.934GiB/卡；首段单卡50更新平均148.902秒/步、峰值33.635GiB，实测约1.98倍更新吞吐。两段各更新36任务×16=576查询，step50→51累计micro-step450→455，step100为700，optimizer/LR连续；原step50 checkpoint与run合同保持。控制器已自动进入step150，动态分卡快照在Writer仍运行时选择gpu01:0,1，合计6张物理卡。后续若Writer结束则下个完整阶段重新live选卡，实际扩卡收益仍须测量。

Writer step1000完整correct Validation400退出0，117/400，正式history刷新最佳为1000且`stop=false`。800→1000保留/获得/丢失63/54/24、churn78，但至少成功一次的任务覆盖5→4；前一个高点600→1000为115→117，保留/获得/丢失82/35/33、覆盖6→4。两个比较均用完整400行及正式配对比较器，原始JSON在新study/analysis。总分小幅刷新同时覆盖收窄，后续必须如实报告相邻稳定性，不能只按117宣称广泛改善；继续由登记的完整节点早停规则推进。

按Owner要求对已完成的Writer前800更新做只读吞吐检查：四个200更新区段平均步时依次9.996、9.848、10.163、9.815秒，最近601–800中位9.244秒、95百分位15.141秒，峰值reserved22.787GiB，梯度同步均值约0.029秒/步。profile均值9.44秒/步与正式记录同量级；记录没有逐rank计算等待分解，较长尾步还不能归因。上一轮两卡的frame chunk8→16在最长视频上缩短步时约7.7%，本轮四卡frame8尚无匹配对照。当前Writer4卡＋MT-BC2卡已按6卡上限运行，不另起并发GPU profile；若训练仍需较长时间且出现安全资源窗口，再做同逻辑更新的短对照，不静默改活跃Writer的exact-resume配置。

Writer step800完整correct Validation400退出0，87/400；正式早停`stop=false`，历史最佳仍为step600的115/400。600→800严格配对成功集保留/获得/丢失62/25/53、churn78，任务覆盖6→5；比较JSON在新study/analysis。当前四个完整Writer节点为110、92、115、87；不因这一次下降提前终止或改变配方。

MT-BC step100首个双卡恢复在准入修复后仍于optimizer更新前拒绝：旧run合同与新冻结runtime候选合同归一后的唯一差异，是同内容tokenizer manifest在两个worktree里的绝对路径。已用只读双卡诊断捕获实际候选合同，核对两个manifest字节相同；`6950f761`只在其它tokenizer字段及两个manifest内容均相同时接受路径迁移，真实失败候选与新runtime路径均能严格归一到原step50合同。9项聚焦测试通过，内容/模型路径变化仍拒绝。第二次失败exit、训练stderr、准入快照、候选合同和独立故障JSON保留；step50 checkpoint未变、step100无checkpoint。新的clean pushed detached runtime `.codex/tmp/coverage-mtbc-resume-runtime` 为`6950f761`，阶段脚本再次原子替换。双节点实时GPU与strg01额度通过后，唯一控制器重新续接；正式日志已出现`resume_step=50, stop_after_step=100, tasks=36`和原run合同hash，双卡训练进程在运行。完整step100结果和真实吞吐仍待确认；不将这次工程拒绝解释为科学结果。

MT-BC step50恢复correct Validation已完成400行、3个worker退出0：82/400，正式readout`stop=false`。Source→MT-BC50逐行配对51→82，保留/获得/丢失36/46/15，原始比较在新study/analysis。控制器随后自动启动step100，但新增stage内嵌Python预检的`else6`语法错误使两次准入都在训练进程启动前失败，stage和控制器exit1；旧exit、两次GPU快照、storage记录和故障JSON均已封存，step50 checkpoint与完整结果未变。已原子修为`else 6`，编译两段内嵌Python，并在gpu01用原失败快照复跑准入均通过；双节点实时GPU和strg01 data0额度也通过。唯一MT-BC控制器已从完整step50重新续接，实际gpu01双卡torchrun训练进程核验存在，原科学合同不变；真实step100退出和完整评测仍待确认。故障与重启记录在study/launch/mtbc_step100_physical_transition.json及mtbc_step100_embedded_preflight_failure.json。

Writer step600完整correct Validation400已正常退出：115/400，刷新200节点的110/400；正式history裁决`stop=false`、当前最佳600，控制器继续登记的下一段。400→600严格配对成功集保留/获得/丢失70/45/22、churn67，任务覆盖6→6；Source→Writer600为51→115、39/76/12，任务覆盖4→6。原始比较JSON已写入新study/analysis，不以单节点峰值提前冻结模型。

已用正式比较器对全部完成的paired Validation行做阶段统计，原始比较JSON保存在新study/analysis：Source→Writer200为51→110/400，保留/获得/丢失37/73/14；Writer200→400为110→92/400，保留/获得/丢失52/40/58，success-set churn98，任务覆盖5→6。两次比较均通过完整评测完成记录、共同source/normalization、逐行task/state/RNG与Writer视频ordinal检查。这只是已完成节点的邻近证据，不替代后续400节点、早停或最终选点，也不按这一轮下降调配方。

已原子安装后续MT-BC物理分卡入口：step100仍固定gpu01:0,1、micro64/accum5；step150及以后每段在旧评测进程释放后读取双节点实时GPU快照，只从gpu01选择状态合格、无进程、空余至少38GiB且利用率不高于10%的卡。Writer控制器未以exit0结束时，持续预留其gpu02四卡；结束后释放该预留，按当时空闲卡数决定总上限6或8，单节点最多6。选择结果及原始快照逐段留在study/launch，训练和评测前仍分别做完整GPU与存储准入。保存的真实快照演练：Writer活动时选gpu01:0,1；模拟其结束后选gpu01:0,1,2,5,6，均符合当前6卡总上限。此为物理调度，科学合同、逻辑576查询、选点与早停不变；实际多卡提速须看完整正式段。原双卡入口备份于study/launch/stage_mtbc_before_dynamic_allocation_20260920.sh，`bash -n`与选卡演练通过；脚本使用原子替换，未触碰正在执行的控制器或恢复评测inode。

MT-BC恢复评测完成后，现有唯一控制器将自动从step50启动step100；未来stage_mtbc.sh已原子切换为gpu01:0,1单节点DDP双卡，来自clean pushed detached `e0d60f75`，micro64/accum5、task-striped物理分片、显式v4→v5完整节点恢复，同一576查询与optimizer/scheduler时钟。实际启动仍由stage先检查双节点GPU总上限、峰值余量与storage gate。准确命令/环境、输入、输出、恢复语义、失败处置记于study/launch/mtbc_step100_physical_transition.json；目前状态为等待step50完整评测，尚未launch。两卡一次性profile与并行Writer4卡＋MT-BC评测1卡会超过6卡上限，因此step100正式段将提供首次真实多卡吞吐/显存证据；CPU旧v4→两卡更新测试已过，工程失败保留step50。旧stage_mtbc脚本备份于study/launch/stage_mtbc_before_two_card_resume_20260920.sh。训练/评测准入均加一次10秒有界重试，显式传播storage/GPU工具失败；活跃旧inode不改。

MT-BC首段50更新训练已完成并通过v4完整checkpoint/合同校验。训练进程释放gpu01:0后，19:19的评测准入快照显示该卡无进程、空闲45,752 MiB但瞬时util=100%，原stage在任何评测worker启动前退出1，控制器随即退出1；尚无MT-BC Validation分数。原exit、控制器exit和GPU快照已归档，事实与未证实的采样原因记录于study/launch/mtbc_first_stage_evaluation_preflight_failure.json。双节点重新live预检显示gpu01:0空闲且util=0后，独立`recover_mtbc_evaluation.sh`仅用原step50 checkpoint启动正式400行评测和readout，实际launcher与3个worker已核对；不重训、不换配方。原MT-BC控制器已作为唯一续行者重新启动，并确认只等待恢复的stage-complete信号；恢复脚本完整exit/400行readout成功后由它自动裁决并从step50继续，失败则退出。两份未来stage launcher已原子更新：评测准入首次失败时保存原快照、等待10秒、再做一次完整双节点预检；仍失败则正常退出并保留证据。原文件备份在study/launch/stage_before_gpu_quiescence_retry_20260920.sh。

后续全新评测面板已接入clean pushed detached evaluator runtime `.codex/tmp/coverage-eval-runtime`（`bee2d5e8`），两份stage launcher经`bash -n`与最小diff核对后以`os.replace`原子替换；已启动的Writer600阶段继续读旧inode，不被中断。MT-BC50原stage在评测前已经失败，独立恢复评测使用新runtime。未来stage只在评测命令使用新runtime；训练、checkpoint与物化仍来自各自原冻结runtime，evaluation/MT-BC配置字节一致。旧launcher备份于study/launch/stage_before_eval_queue_upgrade_20260920.sh；回退须先写临时副本再原子替换两份stage文件，不能原地覆盖活跃inode。新分片实际吞吐待后续完整400面板验证，既有完整队列不迁移。

已完成的Writer400面板给出具体评测尾部：12个persistent workers的400行/48分片耗时1000.34秒；最后一个分片完成前约200秒已有5个worker退出领取，尾部主要是16初态的普通Spatial分片，单分片耗时约244–279秒。当前动态队列有效但普通分片相对Long优先分片过大。主树已按实测针对性收紧：当最长普通分片估算成本超过最长预平衡优先分片1.5倍时，普通分片限为一个env batch；对应4卡×3副本400面板由48分片改为66分片，仍覆盖原400个task/state一次。105项queue/horizon测试通过。此项只用于以后全新输出目录的评测；正在运行的冻结runtime及现有队列未改，实际墙钟收益待新完整面板核验。

多卡MT-BC的独立clean pushed detached runtime已备于`.codex/tmp/coverage-mtbc-flex-runtime`，commit`e0d60f75`；一次性候选profile入口study/launch/profile_mtbc_striped.sh未启动。Writer4卡＋MT-BC1卡同时运行时另起至少2卡profile会超过当时6卡总上限，因此正式step100作为首个多卡吞吐观察。旧step50训练来自原冻结runtime，后续完整阶段由控制器调用新runtime，不并行启动第二条训练轨迹。

Writer第二个完整correct Validation节点400更新已退出0：400行、12个worker均退出0，成功92/400；200节点为110/400。两节点不足以触发登记的早停，正式history裁决`stop=false`、当前最佳仍为200。控制器已从400完整checkpoint自动续训至600，实际训练进程核对存在；此下降不作科学终止或改配方依据。当时MT-BC首段50仍在训练，尚无正式完整Validation结果。

Source释放gpu01:1后，在两节点合计6卡上限内完成独立MT-BC microbatch72/accum8的3更新profile，退出0。与旧micro64/accum9同为每更新576 queries；旧/新平均149.760/146.437秒，steady两步148.668/146.726秒，峰值reserved 33.635/36.707 GiB。不同物理卡、仅3更新的约2%优势不足以证明稳定提速，正式运行暂保留micro64，优先等待卡数变化后测多卡。一次性profile的6个权重/优化器载荷共127,000,368字节已退役，run contract、metrics、summary、checkpoint manifest、launch log/exit和`payload_retirement.json`保留；不能用它resume。

Source在新协议上的Validation400已完成：`source_validation.exit=0`，正式results有400个不同task/state行，worker退出0，成功51/400。此前`exit=1`仅为已修复的准入失败，不是本次结论。截至Source完整节点时Writer和MT-BC仍由各自后台控制器推进，未为代码变更打断运行或启动重复轨迹。

Owner要求训练恢复不绑定物理卡数、节点内减少慢条件拖累。MT-BC已在`1c36908e`实现完整optimizer节点的显式物理拓扑恢复：原36×16逻辑查询、loss权重和optimizer/scheduler沿用，checkpoint记录新物理拓扑；新rank建立独立RNG，换卡后不称bitwise exact。36任务交错分片把慢任务分摊到各rank，评测仍用原动态队列。旧v4单卡checkpoint迁到双卡CPU DDP后，下一步参数与单卡参考一致；相关52项测试及编译通过。正式使用前仍须从clean pushed frozen runtime做真实多卡吞吐/显存检查，完整阶段边界live核验双节点GPU及quota；现有运行中的frozen worktree和脚本不原地改写。

## 接手续行快照（2026-09-20 18:14 CST）

`main`工作区干净，本地与远端`main`均为`07467977`；两个detached runtime仍由正式任务使用。当前Codex任务已建立继续推进的goal。Source新Validation评测与MT-BC首段50更新在gpu01仍有实际进程；`source_validation.exit=1`是17:07首次准入失败留下的旧文件，当前运行结束后才可据新exit及完整产物裁决。MT-BC控制器正在等待首段完整评测。

Writer 200节点已完成正式correct400，110/400；原200 checkpoint和物化bank用于恢复，没有重训。其控制器续接400时因gpu02:2被其他用户短时占用，在训练前GPU预检退出1；旧exit保留为`*.gpu_contention_initial`。18:13双节点实时预检后，原定gpu02:0,1,2,4满足准入；从`macro_00000200` exact-resume启动400段，正式训练进程已确认。恢复入口为`launch/resume_writer_controller_400.sh`，完成后由同一控制器自动执行完整评测、早停与后续区段。没有启动第二条Writer轨迹；不得将此次工程准入失败或旧Source exit解释为科学结果。

下一次只在完整结束信号或明确工程退出后读取对应exit、正式400行和早停历史；不读取部分分数。新Test及FT/RL仍受既定资格门槛约束。

## 当前状态（2026-09-20，新goal：单次覆盖重训）

Owner已授权采用专家最后的24/8/8＋12辅助方案重训EMBER与MT-BC，Source复用；性能及特异性正常后继续原后继实验。
协议与训练入口接入完成。17:03启动一次性配置profile：gpu02:0,1,2,4 Writer9updates；gpu01:0 MT-BC3updates。两节点tmux与真实训练PID已核对，Writer9步覆盖全部36task，随后完整resume至10通过；均值9.44s/update、peak21.38GiB。配置冻结于1a32a0cf，已提交正式Writer首段200→完整correct400（gpu02:0,1,2,4）；Source新Validation400在gpu01:1共驻重启。Active design：[覆盖重训合同](docs/coverage_retraining_design.md)。任务规格与36个训练HDF5 metadata覆盖审计通过，未读held动作。
coverage_audit已完成36任务规格/metadata审计；训练组件分别完成并集成。CPU集成94项中原fixture缺data字段造成5项失败，修复fixture后45项视频测试通过；另8项协议/早停测试通过。
配置profile与日志：`/data0/user/ymdai/ember_runs/coverage_retraining_20260920`。合计5卡；Writer含低负载共驻卡。完整结束后读结果，等待期间接入评测编排。
新训练无固定2000/600硬终点，仅固定观察间隔；不追加fresh seed或k折；不达预期停下询问。
低负载GPU可共驻，双节点live选卡；不监控代理、不读取部分评测成绩。

### 当前执行与资产

Source新Validation初次提交因旧入口固定32GiB要求在worker启动前拒绝；已按source单worker与materialized policy共用12GiB+2GiB预算修复，在1a32a0cf冻结runtime重启，无已完成rollout被丢弃。

MT-BC profile完整3updates通过，每步36task各16queries，平均149.76s/update、预留33.64GiB。已从clean pushed detached3ebb979b启动fresh首段50→完整400（gpu01:0），正式训练PID已核验；不复用profile权重。Writer已按profile冻结正式首段，Source共驻评测在worker启动前的限制已修复。

两个训练控制器已在各自节点tmux独立运行：`ember-coverage-writer-controller` / `ember-coverage-mtbc-controller`。
首段完成后由`study/launch/continue_training.sh`读取完整400裁决，按原间隔自动exact-resume直到登记早停；不再由主agent逐段读成绩或手工重复launch。
最终完成信号为`ember-coverage-writer-training-complete`（gpu02）和`ember-coverage-mtbc-training-complete`（gpu01）；Source仍为`ember-coverage-source-validation-complete`。
任一失败立即结束对应控制器，exit/log留在study/launch。每段live选卡合同不变；已评测且继续训练的Writer bank仅退役可再生载荷，manifest/rows/完整checkpoint保留。

本轮已清理400个已结束旧Test物化载荷及18个完成profile参数文件，释放实际占块2,489,434,112字节；完整清单在新study/asset_retirement.json。全部旧formal checkpoint、raw rows及profile合同/指标/恢复证据保留。


Test other/wrong/shuffle/reverse接口已通过canonical编译/评测路径开放，需同method_freeze及paired_correct_manifest。两处旧fixture补data字段后，horizon/video-controls合计148项CPU检查通过；尚未授权越过性能门槛运行Test controls。

Writer首段200及correct400物化完成后，最初shell launcher退出2，未启动评测；其运行期间曾原地改写launcher，疑似解析位置受影响，原stderr未保留，不能确证。
已保存`writer_first_stage_launch_failure.json`，使用独立不可变`recover_writer_evaluation.sh`仅启动该bank的完整评测，不重训或重新物化。后续脚本变更采用原子替换，现有进程继续读原inode；阶段恢复通过后再接回控制器。

## 前阶段已封存：旧Test未达门槛

冻结Source1000 / MT-BC425 / EMBER1500的Test8各400已完成：78 / 74 / 82。
EMBER只领先MT-BC8条（2pp），低于Owner至少40条的推进门槛；任务簇bootstrap95%CI为[-8.51,11.25]pp。
1200行、18个worker退出、共同source/normalization、逐行state/RNG与50条teacher无放回映射全部核对通过。
Owner已选择先看报告并与专家讨论；controls、FT、RL及外部比较全部停止，等待新指示，不重选模型。
[图文报告](docs/review_materials/20260920/test_capacity/report.md)附逐任务、suite、breadth、配对成功集合与CSV原始行。
Active design登记[paper_experiments_design](docs/paper_experiments_design.md)，当前仅为已冻结待裁决合同，不授权自动进入下一阶段。
Study：runs/analysis/paper_experiments_20260920；正式结果、manifest及readout.py保留；400个旧物化LoRA载荷已在新训练启动后退役，可从保留的formal1500 checkpoint重新生成。
首次nohup提交未存活；tmux重新提交后正常完成。EMBER原定gpu02:3忙，物化前拒绝后改gpu02:0,1，科学合同不变。
证据见study/initial_launch_failure.json、allocation_retry.json及每组launcher_completion.json。
临时detached runtime 45a39ba1已在完成后删除；无本轮分支。原checkpoint与数据不动，未为本轮创建新训练checkpoint。
后续等待采用完成信号，不读中间成绩，不使用监控代理。24任务Test入口的13项Source-SFT与45项Writer tests已通过。

## 上轮已封存结果（2026-09-20）

Owner授权的专家最终修订、主候选、唯一匹配消融、后续趋势、消融视频特异性和条件性双相机全部完成。
该轮没有剩余作业；[合同](docs/video_teaching_writer_design.md)已封存。
先读[专家讨论总报告](docs/review_materials/20260919/final_report.md)与[可复制提示词](docs/review_materials/20260919/expert_discussion_prompt.md)。

| 更新 | 同视频 correct /400 | 消融 correct /400 | 双相机 correct /400 |
| ---: | ---: | ---: | ---: |
| 900 | 149 | 125 | 117 |
| 1200 | 174 | 136 | 108 |
| 1500 | 165 | 147 | 108 |
| 1800 | 160 | 136 | 未运行 |
| 2100 | 158 | 159 | 未运行 |

原1500相邻qualification选定主checkpoint后保持冻结；1200的174不作selected结果，追加窗口不重选。
主组1500→2100 correct165→158、other165→156；消融147→159、other155→161。
主组原窗口的正确能力增量未保持到末点，不证明消融更好或两者等价；[续训报告](docs/review_materials/20260919/continuation_report.md)。
固定1500的主组correct/other/wrong/shuffle/reverse/source为165/165/124/113/113/50，消融147/155/122/104/86/50。
两组都有顺序敏感性，但六个匹配DID区间均跨零，尚未证明教学项增强视频特异性；[视频检查](docs/review_materials/20260919/ablation_controls_report.md)。
双相机fixed1500 other105，correct/other相对单相机均下降；train52/64/67及FM下降未转为验证能力；[相机对照](docs/review_materials/20260919/dual_camera_report.md)。
2100和双相机没有额外controls，不能继承旧因果结论。无Test、RL、融合、挑视频或旧A/v5.2重训。

全部40个正式面板、12,048条新闭环、5700实际更新及638,400主＋辅助queries核对通过，共同source与续训父历史只计一次。
保留57个唯一arm/step完整checkpoint、run contract、manifest、raw rows和worker日志；[完成清单](docs/review_materials/20260919/completion.json)。
双相机按真实四卡profile采用world4/frame8/microbatch16，训练与物化／闭环流水并行，末轮三面板在两节点六卡并行。
全部训练/driver/正式workers正常退出；08:50评测收齐，08:51两节点核验无本轮GPU或训练／物化／评测作业。
source始终冻结，完整resume、6000/8400事件、全池teacher映射、逐行state/RNG和真实RGB变换核对通过。
实现阶段CPU检查主教学图累计246项、相机相关222项通过（覆盖有重叠）；后续分析只读取已有证据，不新增模型forward。

正式主/消融/续训/双相机的冻结commit为39c3919c/bd497edc/d1474ce0/35124aa9；临时runtime树已移除，可按commit重建。
本地原件统一在`runs/analysis/video_teaching_20260919/`，远程保留方法、配对CSV/JSON、曲线、输入示例和报告。
findings§121–122与[研究历史](docs/research_history.md)保存完整解释及边界；后继选择供Owner与专家讨论，需新的明确授权。

## 历史证据入口

[A的learned frame-set报告](docs/review_materials/20260918/frameset_report.md)保留上轮匹配诊断；
[46组证据审计](docs/v52_evidence_audit_20260917.md)、[findings](findings.md)§117–122及
[研究历史](docs/research_history.md)索引统一Writer、A900机制、Core/Procedure交叉与更早实验。
历史结果、旧源码与配置均按各自封存口径解释，不恢复已结束路线。

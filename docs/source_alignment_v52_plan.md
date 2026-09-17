# 对齐A的后续学习、问题分析与一轮改进B（2026-09-16）

## 当前目标与授权

执行暂停（owner 2026-09-17最新要求）：当前已启动A3000训练完成并保存后暂停，不再启动物化、评测、续训或B。
下述为未完成计划及先前授权，当前执行状态以progress为准，后续须等owner新指示。

Owner于2026-09-16调整goal：先把对齐A的情况弄清楚，继续训练，观察性能是否像原始v5.2一样不稳定、
视频特异性会改善还是恶化；随后结合原始v5.2至今近两个月实验，分析存在的问题、原因与改进依据；
最后把改进称为本轮B，完成一轮正式训练和评测，无论解决与否都分析清楚并直接汇报，不另写报告。
期间正常推进共享SFT的训练与评测。此前A900后暂停、SFT仅训练至400和不运行新B的限制已解除。
本轮B尚未设计，不自动继承历史双视角／learned H-read B；旧C及其它关闭窗口不自动恢复。

措辞按owner纠正：旧v5.2的强视频特异性没有在新A900复现出来，不能改写成只是“没有稳定建立”。
必要纠错和合理训练条件变化不能替方法表现开脱；这些差异只限制对具体根因的归属。
当前A900为correct／other／wrong／no-video／shuffled／reversed=140／136／116／48／128／139，
各400；完整结果见findings§109和`runs/analysis/source_alignment_20260915/A/video_specificity_step900`。

## 第一阶段：继续A并弄清变化

在新结果产生前登记A1200／1500／1800，保存每100更新；沿已有A900精确恢复，不重置Writer、三Meta、
optimizer、原100-warmup／12000-decay学习率时间轴、sampler／RNG、source或四rank／GPU UUID拓扑。
原1800窗口覆盖历史A900后1200和1800已经暴露的保持问题，原登记追加900更新／3600个视频条件／75600 queries，
累计1800更新／7200条件／151200 queries。2026-09-17 owner追加：先看1800，若反弹则继续，希望看到明显过拟合趋势再停止；
因此1800不再是自动终点，后续按300更新节点登记和执行，不修改配方或重启学习率。
保持每步4task×21queries、train24、46/4分池、offset1、单agentview／full50 mean、rank16完整LoRA及纯FM。

停止判断依据完整correct400与train96的共同趋势：至少两个相邻正式节点显示验证能力持续退化／无法保持前段获取，
同时训练任务能力保持或提高、两者差距扩大，并结合逐task获取／丢失和任务bootstrap判断是否为清楚的过拟合趋势。
最后一段出现验证反弹就继续观察；单点回落、训练和验证一起变坏或仅FM下降均不足以单独裁定过拟合。
不能为得到预设结论强行贴标签；若证据仍是波动，登记下一个300更新节点。每次追加1200条件／25200queries，
launch前重核预算、原完整事件前缀和完整状态连续性。视频controls用于特异性演化解释，不参与这个续训／停止判断。

先用原冻结575c189a完成900→1200；1500／1800需要显式、仅延长预算的resume支持，
须验证已登记1200事件的前缀完全保持、原状态连续、架构／优化语义和所有既有证据不变后才使用。
不篡改旧checkpoint、旧config或原run contract来绕过resume检查；必要实现从新的clean pushed detached树运行。
若1800后续训，同样保留原1800扩展登记与已完成数据；不能修改当前正在执行的冻结树或共用launcher。

每个新节点完成validation correct400和固定train96，再完成other／cross-suite-wrong／shuffled／reversed各400；
时序臂在该节点最后测试，重排真实RGB后完整forward。沿用seed20260911的全50视频无放回state–video映射，
以及env／policy seed7；no-video是同一冻结source的完整零LoRA，复用已审计48/400，不冒充新rollouts或language-only。
比较相邻节点及900基准的逐task／suite、breadth、R/G/L、churn、Jaccard、任务bootstrap区间，
并分别判断correct与other相对wrong／shuffled／reversed的差额及其变化；不能仅凭对照总分下降宣称视频利用改善。

本轮owner明确授权按节点观察特异性，并结合A全貌分析改进。它覆盖此前仅在末点做封闭controls的阶段限制；
不按controls选择checkpoint、延长A预算或构造训练loss，不使用validation/test动作产生梯度，Test保持关闭。
A900原诊断的原始目的和冻结声明保留为历史，不倒改原件；本轮新的比较与解释另记在既有ledger和formal evidence。
长任务按顺序后台运行，只在完整阶段结束时核验；不持续盯看队列、分数或逐分钟汇报，实际失败才介入。

## 第二阶段：证据分析与本轮B

A节点完成后综合历史原件，覆盖原始v5.2／v6、后续读取／出口／任务共现／过程路径／Pullback等实际实验；
保留正负证据的适用条件，区分获取不足、已获能力丢失、未见任务迁移与视频特异性未复现。
先给出完整数据与信用路径，再定位最早有证据支持的失效接口；纯loss、norm或架构名称不能代替闭环事实。
明确哪些原因已有定位、哪些只是竞争假设，改进将改变什么预测以及为何不等价于已失败方法。
必要的限定机制检查服务于区分解释，不把全部历史重跑，不借此自动添加多轮候选。

形成一个有依据的改进，称为本轮B。保持科学目标、冻结source、split、信息墙和正式配对合同，
具体架构、训练曝光、节点、预测及停止边界在B实现／profile后、formal训练前登记；默认与A有可比较曝光和映射。
B采用fresh Writer和fresh优化状态，完成一轮正式训练、闭环、保持与视频对照；无论结果正负都给出同样完整分析。
本goal结束条件是A弄清、原因与改进论证完成、B一轮实测及解释完成和SFT既定基线读出完成；
>145仍是方法资格，但不是将一轮B自动扩大成多轮搜索的理由。最终直接向owner汇报，不另生成报告文档。

## 并行共享SFT及资源

原SFT400训练继续，按既定400→评测→425→评测→450→评测顺序完成后停止；原rank128／global576、
2400步LR时间轴、raw source1000、offset1与无held-action读取不变。使用独立冻结16c81e29及原gpu02两rank。
SFT操作由独立agent负责既有SFT输出，我方负责A和主线文档；各自launch前检查双节点实时资源及独立quota，
不修改运行中的共用launcher、不重叠使用彼此正在运行的GPU。新阶段登记后无需再次向owner请求逐项许可。

原阶段准备时data1用量970873968KiB，soft1073741824KiB，余量约98.1GiB，共享83TiB；这是准备快照，launch前刷新。
原A三个节点预留32GiB（checkpoint、四类实际视频LoRA banks、train banks及logs；other复用同视频完整LoRA），
SFT当时预留3GiB、B40GiB，总75GiB小于当时余量。已完成1500节点的checkpoints和去重LoRA banks实测约8.56GiB，
1800后每个新增300更新完整节点按12GiB预留；SFT完成后其剩余预算为零，B继续预留40GiB，设计／profile后再测并重核。
不复制source或dataset。
3000登记时data1余量已不足同时放置下一A节点和B预留，B的40GiB改在既有`/data0/user/ymdai/ember_runs`独立预留；
该filesystem的strg01 user quota、个人用量与共享容量已核验，具体快照归progress及launch contract。
A已有与后续输出仍在data1，不合并两侧quota、不移动已有资产；实际B设计／profile后再次估价，每次launch检查适用预算。
当前执行状态归progress，历史科学事实归findings／research_history；以下保留source重建和A原始训练规格。

## 已确认的时间关系

固定LIBERO生产代码先执行action再保存RGB／proprio，原states/actions保留同一原索引：
`states[i]=s_i, actions[i]=a_i, obs[i]=o_(i+1)`。本地train0/12/20/34及source allowlist首task的demo0前11行
关节观测均与下一行simulator state精确对应，间隔0.05秒；核验未使用held动作。
原source commit e2cc238和历史v5.2从同一frame_index取action chunk，现行Writer已采用offset1。
官方评测读取实时观测、执行预测chunk第0–4条后重规划，没有跳过第一条预测。

修正监督为`obs[i] -> actions[i+1:i+51]`，RGB和8维proprio保持同一时刻。
无未来动作的末观测不作执行query；不足50条沿原末动作padding，teacher完整视频仍保留真实末帧。
旧闭环分数仍是真实行为证据，不能由此认定错位解释全部历史负结果。

## S：同规格fresh source

原参照为`runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722`，新source保持：

- 同revision generic `lerobot/pi05_base`、审计后71个LIBERO-90 tasks、每task全部50个episodes，不引入target40 actions。
- 全参数SFT，包括vision／VLM／Action Expert；seed7、1000 updates、effective global batch256，共256000个queries。
- 原task／episode均衡采样、AdamW betas(.9,.95)、eps1e-8、weight_decay1e-10、global clip1。
- Linear warmup333后constant5e-5，BF16及原模块dtype、双相机、processor、chunk50、source normalization与tokenizer。
- EMA decay .999及完整恢复状态；预先固定raw step1000供后续使用，不在raw／EMA／中间点间选优。
- 只改未来动作标签与必要的有效frame支持；不声称与旧训练逐query完全相同。

原8×A100、物理batch32、约71GB峰值属于历史执行条件。当前以live A40资源和等效microbatch／梯度累积保持global256。
允许rank0 CPU保存不参与梯度的EMA，以及减少optimizer临时张量／共享DDP gradient bucket；这些须由真实更新profile验证。
不冻结参数、不压缩曝光、不降低训练dtype。旧120分钟属于旧机器窗口；本轮严格1000更新，墙钟由profile另行登记。
采样保留原8个logical task slots／每槽32queries，再按logical global query stream打包到物理rank／microbatch。
不能因累积次数增加而改成更多task、每task更少queries。原任务／episode／frame调度与历史实现已用71tasks、
1000 logical updates共256000 queries核对一致；offset1使可执行frame支持变化，仍不声称与旧标签逐query等同。
允许256不能整除world时按query数分配（如三rank为86／85／85），末microbatch取实际剩余queries，
每个mean loss乘`local_query_count * world_size / 256`后由DDP平均，保持每个query严格等权。
每个rank的每次microbatch均为真实非空query；world、GPU UUID及完整物理batch计划进入resume合同。
原生未参与action loss的language-output heads继续按原图处理；累积DDP使用unused-parameter发现，不人为增加额外forward／loss。
DDP保留已建立的gradient bucket views并原位清零；fresh／resume的首个真实microbatch先同步以建立views，
其余累积仅在末microbatch同步。首项已平均的梯度再次取平均不改global256权重，不增加训练样本或optimizer更新。

恢复一个source训练入口及cohesive训练owner，复用现存setup／contract／checkpoint／dataset，不复制source-SFT或Writer orchestrator。
新config单独登记，原sealed config及source保留。数据／模型复用固定revision、manifest、file sizes与provenance，
不重复整库hash扫描或复制资产。Profile验证实际全参数、下一步标签、finite更新、EMA和峰值，随后验证多卡global256及resume。
正式运行来自clean pushed detached树；中途恢复点只用于resume，允许新完整点发布后按keep_latest=1退休，raw1000完整保留。
物理拓扑、checkpoint interval与命令在profile后封存于唯一launch contract。

完成后固定raw1000做validation400及train96官方闭环，旧source47/400、17/96的同合同面板可明确复用。
不使用held动作诊断或梯度，不按验证分数延长source或挑选source checkpoint。

## A原始训练规格与历史双视角B参照

| 臂 | 教学输入 | 完整50-H读取 | 共有模型 |
| --- | --- | --- | --- |
| A结构复现 | 单agentview | fixed mean | v5.2 Core／Procedure／三Meta／完整38-target rank16 A/B |
| B | 同步agentview＋eye_in_hand | 现行learned H-read | 与A相同 |

Owner明确要求A结构参照，单视角／mean在本计划内获授权；不重新引入旧错位或同episode query。
两臂fresh，不直接给旧Writer换底座；保留raw frame indices RoPE、stride5＋末帧、原Gaussian probe和纯跨episode FM。
不增加RL、辅助loss、第二adapter或其它表示路径。Mean组保留相同零初始化q/b并固定为零，learned组学习它们。
共有参数保持相同构建顺序／RNG；保留B已有初始化流，不因诊断模式额外初始化或重设seed。

统一沿B：train24、teacher/query demo0–45严格跨episode、demo46–49固定train96；每更新4task、每task21queries，
baseline task permutation分组，不用C分组。共同事件、flow噪声、task权重、optimizer和全局LR时间轴一致。
每臂最多1200更新／4800条件／100800queries；每100保存，300／600／900／1200做correct400＋train96。
各节点复用B seed20260911的state–video映射、env/policy seed7及全50视频无放回合同。

此前source基线计划已被上方三阶段goal替代；下表B仅指历史双视角参照，不是本轮待设计B。旧B已有offset1，是source效应的历史下游参照。
旧A125／132只作历史定位；新A还修正了旧采样／标签合同，不能将其全部差额单归source。
旧A正式映射使用video seed7，本轮A／B沿用已登记的seed20260911；两者都在每task使用全部50条视频各一次，
但400组state–video配对仅8组相同。因此跨历史A的逐行R/G/L只在task／初态／env和policy RNG层面配对，
同时包含teacher分配变化；不得称作完整video-paired因果差额。本轮A的相邻节点及本轮A／B仍完全复用同一映射。
实测边界与旧A900→旧B900读数见study的`archival_A_B_comparison_boundary.json`。

## 共享SFT基线补齐（2026-09-16 owner追加）

历史109／107来自train24单套共享rank128 LoRA，而非source71全参数SFT或逐task experts。
固定原实际run `pi05_source_sft_rank128_mixed_dev_r4_b144_seed7_s2400_20260728`为训练规格参照：
24tasks、每task全部50episodes、每更新各24个query、global576、seed7、原episode／chunk均衡调度，
原AdamW／warmup100／cosine时间轴、相同38-target rank128参数化和冻结normalization。
原实际完成450更新／259200queries；400／425两个配对参考分别109／107，不把旧模板config的rank16／800步当实际运行。

新SFT从对齐source固定raw1000出发fresh初始化共享LoRA及optimizer／scheduler，正确使用offset1。
固定同样450更新，保存100／200／300／400／425／450，400／425／450均做完整validation400并比较相邻成功集合；
保留原声明2400训练时间轴的真实LR序列，不把调度压缩至450。物理rank／microbatch／累积由A40实测确定，
保持global576及24task等权；部署只有source加这一套共享LoRA，不读取teacher video。
旧run使用过在线validation action-loss监控，本次不复制该历史做法；validation/test actions不读取、不产生梯度，
不根据held action loss停训或选择。全部闭环只用现行固定官方400初态／RNG，Test保持关闭。

复用唯一`train_source_sft`及现有checkpoint／评测适配；修正过时的硬编码offset与训练调度，避免另建SFT runner。
先恢复科学合同、验证实际更新及恢复，再封存配置与formal launch contract。SFT的rank和动作曝光多于Writer，
它是普通共享监督的能力参照，不宣称同参数量或同query预算公平比较。

实测执行登记：两张A40、physical micro64，每rank按64/64/64/64/32累积五次，global576；
三次真实更新及step1→3完整恢复通过，原生混合BF16/F32 LoRA dtype与历史rank128一致。
后两步平均74.596秒、reserved峰值32.934GiB，450次纯更新估计9.324小时，另计加载、保存与三轮闭环。
正式仍从fresh identity开始；profile参数不复用，配置封存后从独立clean pushed detached运行树启动。
小型CPU／profile与完整恢复证据保存在`runs/analysis/source_alignment_20260915/SFT/`。

## 统一解释边界

裸source、A、SFT和本轮B分别报告absolute、相对各自source增量、逐task／suite、breadth、R/G/L、churn、
相邻overlap与任务bootstrap区间。Source同时影响观察与执行，裸source接近不能证明其对Writer影响小。
旧A与新A存在训练及视频分配差异，不能把全部差额单归source；宽区间不等于等效。
历史双视角／learned-read B的两个架构变量也不能由两个角唯一分开；是否采取其某项做法须服从本轮证据分析。
本轮授权和完成判断以上方三阶段为准，不恢复先前“是否补旧B”的自动路线。

## 工程、存储与交付

主集成为main，Writer模式改动使用task-owned worktree；主agent拥有source及整合，formal jobs来自clean pushed detached树。
模型／data／config身份覆盖source引用、offset、camera与H-read；不同架构fresh，exact-resume锁world/topology。
运行差异以配置表达，活动树只有一个Writer实现；历史configs、source、checkpoints及正式证据保留。

首次strg01 data1 quota为908275108KiB used／1073741824KiB soft，约866.2／1024GiB；项目实测834GiB，共享约83TiB可用。
S+A+B追加峰值预算112GiB，包括旧＋正在写的source恢复点、EMA／optimizer、两Writer checkpoints／banks／rows和profiles。
本地已无generic权重，按原revision补回约13.5GiB并纳入上述预算；数据、tokenizer、旧source和环境复用。
下载期间允许旧source的同尺寸全参数容量smoke；它只定A40显存／吞吐，不作为fresh source或科研结果，
formal前仍须generic初始化的真实更新及resume核验。launch前按实时quota及实测checkpoint复核，后续旧架构另估预算。
当前状态进入既有ledger，完成证据进入research_history；验证后集成推送，清理已集成开发worktree及明确临时文件。

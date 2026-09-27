# 条件速度算子：共同状态基与单LoRA编译候选

2026-09-27。科学依据与最近完整历史比较见[机制分析§29](../analyses/feature_to_operator_mechanism_20260926.md#29-共同学习状态反馈基直接编译条件速度场一个有明确代价的完整候选)。
这是候选及其渐进合同，不是已验证修复。是否实际启动只看progress；§4–5工程已完成，
§6–10的270首批已完成；§11登记450后继，§12记录CPU验收及当前正式执行范围。
实际排队、冻结commit、资源准入和运行状态只看progress及launch原件。

## 1. 假设、输出与信息墙

完整假设：可从合法教学学得任务相关系数R(C)，与共同学习的自身状态函数U h_β(x)组合，形成有用的条件FM速度场。
β提供完整共享LoRA，基础source冻结；β、视频encoder、R读出和U全部fresh、同一真实跨episode FM共同学习。
不是先训/冻结MT-BC，不加载其权重，不建立Reader教师，也没有语义/纠正/几何辅助loss。
“共同反馈基+条件系数”是受限的完整方法，不能由恒等式保证视频被理解、held泛化、绝对能力或保持。

输入仍为exact language和一条完整agentview RGB教学，stride5及真实末帧；K1，不声称Dynamic-K。
Writer不读teacher动作/state/reward/terminal/pose/task ID/文件名或query。动作和query自身state只用于训练policy。
source、normalization、tokenizer和预处理复用现有canonical资产，不复制。部署自身双相机/8维state/7维动作合同不变。

输出一套完整38-target LoRA；共同部分β的rank128，视频部分仅位于action_out、有效rank≤7。
统一部署rank135/alpha135，其余37目标以零算子补齐额外7rank。它们是公共参数，不能声称逐层都由视频改变。
rollout前生成一次，无第二adapter、运行时教师memory、额外控制器或task-local优化。

## 2. 固定的完整图

```text
L + 完整RGB → 原生Text/VL/Action读取及各rank4 Meta → q / E / 完整50×1024 H
           → 现有LanguageSemanticCore C与RecurrentProcedure P（width256、各2 blocks）
           → 七个动作维度query读取C和P的真实Value → R(C) [7,256]

自身真实query + noise/time → source + 自由共同rank128 LoRA β → h_β [50,1024]
U [256,1024]与R(C)相乘得到M(C) [7,1024]
共同action_out + E7 M(C) → 原生32维FM速度（真实前7维监督）
```

教学native读取不装公共β，只沿现有fresh三Meta读取合法prefix/fullH，避免在本候选又增加参数回读因果变量。
复用既有q/E/H、完整帧打包、Core与Procedure的所有实际读取规则；不恢复初帧广播、状态补全、时序重排或额外先验。
Core不启用`language_content_path`。P保留其原始内容/顺序，不使用旧Compiler的中心化调制。
不实例化旧FactorHeads/Compiler后丢弃其输出来取得memory。

七个learned query仅为动作行的检索身份，shape[7,256]、正常非零初始化；不是task身份。
用现有ContentCrossAttention（width256/8heads）先读C得到c，再以query+RMSNorm(c)读P得到p。
C读取无时间RoPE，P使用真实frame positions的既有RoPE；mask沿用真实有效token/frame。
两个Value都来自真实C/P。令d=GELU(Wc c+Wp p)，Wc/Wp为无bias正常初始化的256→256线性层。
R=Wo d，Wo为无bias、零初始化256→256线性层，各行动作行共享这些映射；query本身不旁路进入d的Value。
不增加softmax层位/深度/中心化对照，不由profile分数选择这套读取。

U为无bias自由线性投影，非零fan-in初始化；它不是fixed source/PCA span，也不接受独立标签。
β为自由完整LoRA因子、A非零/B零identity；全部参数fresh，seed7，公共identity seed沿canonical contract。
U、R及β的初始化顺序和RNG需在机器规格中固定，完整恢复保存实际RNG；不要求复现旧MT-BC的逐元素随机轨迹。

设E7=[I7;0]∈R^(32×7)，实际合成：

```text
out: A=[Aβ; R U], B=[Bβ,E7]
others: A=[Aβ;0], B=[Bβ,0]
```

公共和条件算子相加，无同rank因子交叉项；生成state只含76个canonical因子。
初始R=0，虽然out额外B=E7非零，BA仍为零。source物理参数requires_grad数始终0。
训练按现有完整LoRA cotangent/VJP接回β/U/R/encoder；共享β在各condition下的梯度正常求和，不重复计权。
真实7维、完整50-horizon、随机noise/time普通FM；不启用端点/前5辅助、KD、RL或回滚。

## 3. 最接近的失败与候选边界

旧Video Functional的末端query来自未挂LoRA的冻结source，reader与encoder联训，但query特征不学；有另一个LoRA学生。
本候选共同训练执行特征且从第一步就是可精确编译的单policy，没有强教师前置条件。
旧LocalField已让同一U/r受真实纠正标签并组成LoRA；本项不以“直接消费者”作为新理由，改变的是实际条件速度场和共同状态基。
旧公共prior/Unified/DJNFR反对“加公共路径自动保持”；β自由且fresh共同学习并不能独自消除这条反例。
当前完整Writer/private正例反对“生成全层本来就不可学”。这里主动收窄条件函数，可能牺牲所需早期条件读取。

可失败预测：若该共同状态基能承接可迁移教学关系，完整模型应取得强共享参照以外的正确/换正确视频能力，
而不靠大范围丢失和单个峰值；若主要获得公共能力、只降低FM或R有变化而没有正面视频收益，假设不通过。
训练条件不同导致的总分差不能唯一归因于本参数化；正式比较要锁同一数据/评测协议，并如实保留强MT-BC。
若完整有信息量窗口失败，关闭本组合，不逐层加入条件出口、不扫rank/辅助/冻结公共分支继续维护它。
未测功能基的可用性，不用“未试过”或常量可达性称其已获科学资格。

## 4. 当前仅可派发的有界工程

唯一执行者：`01a0dd6c-f2e5-7971-821a-56766e1c0f22`。
主讨论负责科学裁决、机器规格的实质变更和后继学习合同；执行者不得自动进入正式训练或held。

**时间预期**：CPU实现/核对45–60分钟，GPU部分约12–25分钟wall；本批总计约60–90分钟，120分钟为工程判断上限。
这是待实测的工程估计，不是训练速度证据。明显超时或资源不足时回报具体完成面与阻碍，不继续堆叠实现或扫描。
GPU硬限0.75完整GPU-hour，含加载、失败、恢复、profile及case；上限优先于任何预定步骤。最多同节点2张物理卡。
旧Reader的15.53秒宏步不是本候选profile，不能直接据它启动后继长训。

**存储**：所有新增在data1。study root为
`/data1/user/ymdai/ember_runs/conditional_velocity_operator_engineering_20260927`。
原件峰值上限4GiB，开发+冻结代码峰值768MiB；创建前由执行者在strg01核data1独立quota及共享容量，并按实际参数/保留checkpoint估算。
不得新增data0写入或复制canonical大资产。未通过容量核算不得创建大run root。

**当前工程梯度只使用固定train条件**：global2/12/22/32，各teacher demo0；query从同task demo1–49取，严格跨episode。
四task每宏步各28query、等权，总112；固定sampler seed20260927，沿既有offset1与50步chunk规则。
这四task均为coverage train24，且不涉及当前diagnostic-held8；profile还允许coverage train的global29/demo0。
不读任何official validation/test动作/state标签，不把工程case列入科学比较。
工程优化仅用于图/恢复核验：AdamW lr3e-5、betas(.9,.95)、eps1e-8、wd1e-4、clip1、8步warmup后固定；所有参数同一loss。
该短跑配方不预设正式学习LR或曝光窗，profile loss不参与后继超参选择。

CPU核对：完整因子合成/identity、实际7维嵌入、参数梯度消费者、checkpoint字段、官方接口与信息墙。
需要的是少量有真实失败含义的检查，不构建另套测试框架或逐tensor/逐bit验收。
实现必须先clean pushed，GPU使用detached frozen树；启动前按项目规则同时live检查两个GPU节点、NUMA/NCCL和全项目用卡数。

GPU允许范围固定为：

1. 一个fresh world2的4宏步，保存2/4；从完整2恢复到4，额外2宏步。共6实际更新/672训练query，不重跑fresh。
   核验真实梯度、source冻结、共享β计权、恢复sampler/optimizer/RNG/topology和日志前缀；允许正常浮点差异。
2. global29/demo0的完整合法长视频，一次28-query反传profile，无optimizer更新。原记录347帧/stride5含末帧71帧，
   实际资产若不符则先核身份/配置，不按长短或loss改选视频。
3. global2与global32各state0/teacher0一次train-only canonical episode，共2条；2全RGB、32 compact，保存真实T+1动作/body/EEF/gripper/BDDL接口。
   成功与否只作原始smoke行，不能称科学收益、不能据失败补episode或改模型。

默认真实query microbatch28；仅为OOM/已证吞吐问题调整物理microbatch，不改变112逻辑query、task权重或精度合同。
不增加语言臂、冻结β臂、局部oracle、policy probes或最终顺序controls。不得以工程throughput差恢复旧Reader。
结束后核全部退出状态、原件、实际资源计费和自身进程清理；只回报一次完整完成/实质阻碍。

## 5. 工程所有权与后继裁决

优先复用执行者已空闲的适合隔离树；从交付的main commit开始，分支`codex/conditional-velocity-operator`。
主讨论只改main文档，执行者不改main。架构技能由执行者按实际结构应用，不把其review signal变成人工批准关卡。
视频/native输入与C/P拥有一个共享owner；条件R/U/β合成拥有一个owner；训练/恢复与canonical case尽量复用现有接口。
不能创建旧Writer/Reader副本或新通用runner框架。允许工程分支暂有一个明确命名的有界入口，执行者是owner。
科学否决时由Git保留并退役其活动入口；若进入正式学习，须先完成canonical替换/旧输出路径退役及main集成，不留下平行fallback。
源码超过500新增行或3新文件须按skill完成代理自审，报告主要所有权/增长与退役触发；不机械拆文件。

工程完成时不自动授权正式学习：工程回报提供真实秒/宏步、query吞吐、最长峰值、物化和两类case墙钟。
主讨论据此在学习前冻结数据/采样/学习时标、首段最多两个节点及完整比较预算；不从4步loss选checkpoint或决定配方。
正式选择仍须single-checkpoint strict paired400、相邻保持及same-task-other；小screen只作有界早期读出，不能选模型。
当前coverage强MT-BC与C0 diagnostic-held属于不同合同，不能拿155−120解释本候选，也不重跑旧实验补排列组合。
测试集和最终shuffled/reversed保持关闭；本合同不允许自动蒸馏、第二架构、长训或其它实验。

## 6. 工程验收后的完整学习比较

主讨论已独立核对实现`50559080`、逐宏步原metrics、恢复查询前后缀和两条NPZ。实际6宏步/672query，
source物理参数冻结；完整因子FM cotangent通过同一次合成VJP到达β/U/R/教学路径，各task权重1/4后跨rank求和。
四次GPU启动均exit0，总0.1269607541 GPUh；两条接口失败不作科学阴性。原件根为§4的data1路径。

首轮比较完整视频方法V与真正训练的语言方法L；不设独立Reader、冻结β、R置零或几何辅助臂。
二者均由fresh source identity开始，β/U/读出/实际使用的教学模块共同接受相同的普通跨episode FM。
不继承工程checkpoint或MT-BC权重。V保持§2的函数；L的执行policy仍使用自己的双相机/state与exact language。

L的条件路径使用已有`encode_text_only`和`LanguageSemanticCore.language_only`读取真实语言token，
第一、第二次attention都读取该真实语言memory；第二次使用真实token位置和mask，不把第二Value置零，也不造RGB/动作H。
L不运行VL/Action Meta或Procedure，未使用的模块不进optimizer；不得用空视频运行V来冒充L。
这不是严格参数量匹配的信息消融，故V/L差额与最终冻结controls共同解释，不能单独宣布动态视频根因。

β、U和七行读出在两臂中有相同初始化身份，不允许因先构造视频模块而移动共享参数RNG。
正式模型把共同β/读出/U初始化置于条件模块之前。β的identity A沿canonical LoRA合同的identity_seed20260721，
B为零；读出/U的全局module seed为7，共享Text/Core使用明确的module seed7。
机器记录的common_seed7指共同模块RNG，不能覆盖identity_lora_state内部按canonical identity_seed建立的独立generator；
此处明确§2既定的identity合同，不改变本批实际初始化或冻结代码。
可以复用原生教学owner并冻结L未用模块；不能新增一份原生视频/文本读取器。正式fresh顺序与工程顺序的区别须登记，
不把工程checkpoint称为兼容正式初始化。每rank初始化相同共享权重，训练RNG按原有rank合同保存。

数据使用唯一coverage协议`configs/libero_24_8_8_coverage_v1/protocol.json`及其审计manifest：
24 target train加既有12个合法non-held source meta tasks，共36；不扩大任务支持。
显式训练ID为`[0,1,2,4,5,7,12,13,14,15,17,19,20,21,22,25,28,29,32,34,35,36,37,38,42,43,51,55,56,62,64,73,95,96,97,101]`。
official validation为`[3,6,11,16,23,26,31,39]`，Test继续封闭。C0 diagnostic-held8不作为本轮评测协议，
其120/121不与本轮强MT-BC155作因果差。两臂训练demo0–49均合法，teacher与query严格不同episode；不沿用旧46/4诊断池。

## 7. 唯一事件流、优化时钟与逐段判断

每宏步4个不同task、每task一条教学与28个query，总112，loss为四个任务完整50×7普通FM均值的平均。
无21+7两组、无4/3尺度、无端点/前5步项；噪声/时间仍由既有FM owner产生，query使用自己的真实state与offset1动作chunk。
两个模式共用只含编排身份的确定事件流，task/demo/帧索引均不得进入条件网络。

事件seed为20260927，零基update u对应round=u//9、slot=u%9：

- 每round用NumPy SeedSequence `[20260927, round, 0x5441534B]`打乱排序后的36任务，连续每4个组成一宏步。
  完成9步时各task恰好一次，不按长度更改抽样或权重；只按实际frame cost分配rank。
- task的visit等于round；teacher在每50次visit的周期中，按
  `[20260927, task, visit//50, 0x564944]`生成demo0–49排列，取visit%50项。
- query用`[20260927, task, visit, 0x515259]`从其它49个demo无放回选28个，再各均匀取frame∈[0,T−2]；
  action起点为frame+1。L只复用这些query事件，不读教学RGB。
- FM seed复用`task_logical_batch_policy_rng_seed`，optimization_seed7、task/visit及实际query demo/frame相同；
  事件与GPU分配、物理microbatch和恢复顺序无关。保存sampler游标与逐rank RNG，不要求低位/逐bit一致。

全部活动参数使用同一个AdamW：peak lr3e-4、betas(.9,.95)、eps1e-8、wd1e-4、clip1；
直接复用强MT-BC的`source_sft.control.clamped_lr_multiplier`：150更新warmup、绝对step1200到1e-5后保持floor。
沿原函数step0计数，warmup乘数为(step+1)/151，step150达峰；不解释为warmup后再衰减1200步。
V/L共用该优化时钟，不新增分支LR、尾部放大、公共底座课程或训练中配方切换。
这沿用有实际成功记录的普通FM量级，不从四步loss选择。4-task与MT-BC的36-task宏步不同，
相同update数不等于相同查询曝光或优化轨迹；旧Writer不同loss/时钟也不构成单因素架构对照。

**第一批仅到270更新/臂**：30个完整task rounds，每task840 query；每臂30,240 query、两臂60,480。
保存90/180/270完整checkpoint，但只对270做本批正式评测，不用90/180的loss选点或补评最高点。
第二个预期判断节点为450更新/臂（每task完整50个不同teacher均已出现），尚未授权自动续到450。
270回报后由主讨论决定是否沿完全同一轨迹推进该节点；不能用第一点尚低于155宣布理论不可能，也不因此换结构/重启种子。

270裁决关注整个学习程序：V/L与strong MT-BC/Source的绝对能力、配对得失、suite/task覆盖。
V共同能力和L都未形成时，不用内部非零梯度或小loss称方法有效；V明显毁掉L已有能力时，也不靠局部辅助补洞。
如果需要450来判断视频条件获取及保持，只续这一预注册轨迹，不增加层出口、rank、冻结臂、loss或新采样。
第二点后仍无正面视频能力证据且完整能力持续较弱时，应关闭实际检验的组合，不循环宣布“只是某模块未过关”。
若完整策略确在持续取得并保持新能力，后继也必须另有明确节点/预算；450不是优化充分性的数学保证。

## 8. 首批完整评测、证据和产物

V270 correct400与L270 correct400，共800条新episode；无80-row模型选择screen。
两臂及旧强MT-BC使用同一official validation8、state0–49、env/policy seed7及canonical keyed noise，
render256/model224、双相机180度rotate、state8/action7、10 flow、前5动作replan、settling10与原suite horizon不变。
V使用既有`video_schedule.py`，映射seed20260911、K1、全50teacher各一次；L保留配对ordinal元数据但不读teacher。
source/normalization/tokenizer与已核MT-BC300一致；旧MT-BC155与Source51原400行只读复用，不重跑。

同时报告8个task、4个suite、非零任务广度、与L/MT-BC/Source的retained/gained/lost及churn；
保存逐行key和success-set，后继450须报告相邻重合与得失。配对不确定性可以描述，不能把小样本差额说成确定机制。
第一批只有一个节点，不能宣称稳定性、最终模型选择或视频必要性。后继单checkpoint选择必须接受相邻保持及
same-task-other400；本批不跑wrong/no-video/shuffled/reversed，最终controls只在模型选定并冻结后另行登记。
不能用R置零代替learned L：公共action_out与R常量项可互相转移，分支消融不识别视频增量，见机制分析§30。

各episode保存compact真实T动作及T+1 body/EEF/gripper/BDDL；固定每task state0、两臂均保存full双相机RGB，
共16条full，选择与结果无关。无额外episode、环境分支或held teacher action/state标签读取。
沿用cost-balanced long-first/persistent evaluator与已有BatchedLoRAInference；不重写环境/随机流/批量LoRA后端。
多replica/物理microbatch只按显存与吞吐选择，不改变单行配对；任何故障重做仅补缺失row并留失败计费。

bank采用本方法精确共享项存储：每checkpoint的β/U只存一次，每合法condition存FP32 R及实际输入provenance。
由唯一`compile_velocity_state`在该episode首次执行前重建完整76因子，随后整条episode固定；不在线读视频/优化/重生成。
L每个exact language生成一次R并复用50个状态，逻辑仍是一套完整LoRA。无最终LoRA平均或第二adapter。
保存schema/rank135/source/checkpoint/condition身份，保留必要共同权重依赖；不能把R文件单独称完整可部署权重。
这是对相同完整算子的无损存储，不更改模型函数；不建立支持旧/新实验的一般插件或fallback平台。

## 9. 分段资源、canonical集成与当前派发边界

工程实测world2平均16.822秒/112query；270更新的V update-only约1.262h wall/2.523 GPUh。
36-task元数据中完整stride5视频P50/P90/P99/max为29/51/87/105帧，最大raw517（global38/demo36），
高于工程profile的71帧。需要将实际分布/负载和加载保存计入，不能称71已覆盖新协议最长视频。
主讨论CPU核完冻结事件：270时各task恰30个不同teacher/840query，450时恰50个不同teacher/1400query，
均严格跨episode；270实际教学最大102采样帧。用工程第2–4步12个condition的时长作简单frame-cost外推，
270的world2训练约1.216h/2.433 GPUh，最大宏步约24.6秒；只有4条不同教学的校准，故不据此缩减下方预算。
计算原件为`.codex/tmp/conditional_velocity_learning_20260927/event_cost_review.json`，不是模型运行或实际formal吞吐。
L尚无本模型实测吞吐；预算先按不快于V的保守输入计，不把旧Reader-L速度当新证据。
历史官方400约1 GPUh/面板仅是调度参照；本模型800评测另留约2–3 GPUh，物化/加载/失败留余量。

**首批270两臂及800完整评测总硬限9 GPUh**，包括全部加载、训练、物化、评测、故障重试与等待占卡。
合理预期为约5.5–8.8 GPUh，冻结后约2–3h wall（两臂各同节点world2可并行，评测使用实际空闲卡）；
这仍为外推，不是保证。若实测投影越过硬限，保留完整checkpoint/已有原件并回报，不悄然取消L、缩成小面板或超预算。
执行者报告必须区分训练实际进度与完整面板；未完成面板不能选模型。无GPU空闲时不占卡等待。
两节点项目总卡数仍服从AGENTS动态6/8上限，每臂单节点world2锁拓扑；首批不授权训练拓扑切换。

新增root为`/data1/user/ymdai/ember_runs/conditional_velocity_operator_learning_20260927`，
首批产物峰值8GiB，开发+冻结代码768MiB；须在strg01重新核data1独立quota/共享容量与真实峰值才建大root。
使用紧凑bank约两份40MiB公共项加R/manifest；完整训练checkpoint及16条full轨迹是主要增长，不能生成400份公共LoRA副本。
不复制大source/数据、不写data0；工程原件保持原路径，不改写为正式结果。

**已完成的CPU实现/集成准备：原估45–75分钟、90分钟判断上限，当时无GPU许可**：
从工程分支纳入最新main文档，保留单一参数合成owner，加入真实L路径、上述sampler/优化时钟、紧凑bank和官方评测接入。
正式入口替换工程固定4task/4步入口，退役其专用case/profile CLI，工程历史由`50559080`与原件保存。
不得把旧CompleteLoRAWriter/Compiler作为另一可选输出路径；共享native/C/P/FM/ECP/环境工具继续复用。
优先关闭旧默认训练/物化出口与实验flags；若某个共享文件仍是历史路径的依赖，明确最小暂留范围、owner和本次集成时的移除触发，
不能为全树无关历史清理拖住本合同，也不能留下可静默回退的旧Writer。无需复制通用trainer或第二batch evaluator。
源码结构按code-architecture-gate自审，超过规模信号报告实际owner/复用/退役理由，不额外请求Owner仪式批准。
CPU验证实际合成代数（含非零公共项）、V/L共享初始化与合法信息路径、36-task等权事件/跨episode/恢复、
紧凑bank重建与官方episode配对。只核metadata/CPU合成，不跑新模型或环境probe。

CPU交付不自动允许GPU；主讨论已经独立审阅、集成并在§10登记后继启动。文档规模不代替launch记录。
正式train/eval必须来自另行明确的clean pushed detached commit，
并依formal-training-launch登记精确命令/环境/数据/配置/资源。整体Owner授权有效，无需再向Owner索取本范围内许可。

## 10. 首批启动登记（2026-09-27）

CPU交付`7f5374a7`已独立核对；随后修正学习率函数和L第二Value两处事前实现偏差，最终源码为`18b7e4b7`。
L两次均读同一Core语言memory，位置仍为native真实task token位置；优化时钟直接复用强MT-BC owner。
主讨论已经将隔离分支fast-forward集成main，相关22项CPU检查通过；正式冻结采用本登记随后的clean pushed main commit。
没有使用工程权重，没有读取本模型验证结果来修改图/超参，尚无新的能力收益结论。

准许唯一指定执行者按§6–9完成V/L各270更新、各correct400，随后一次完整回报并停止新增GPU；
不自动进入450/other/最终controls/RL/新架构。启动前按formal-training-launch保存一个完整launch记录，
包括精确命令/环境、source与数据路径、mode/事件/optimizer、冻结commit、实际GPU及storage预算。
先查strg01 data1独立quota与共享容量，再建本批run root；每次launch同时live核gpu01/02和本项目总卡数。
训练默认microbatch28、每臂同节点world2；source基础权重冻结，fresh初始化，无模型挑选。

两臂可在不同的同节点world2组并行。已完成L可先物化/评测，同时V继续固定训练；
评测结果不能影响尚在运行的固定事件或参数，主讨论统一验收完整批次。依实际余量选卡和共驻，
不为固定util阈值或凑齐预定卡数无故等待，不干扰其它任务；硬件与全项目总卡数仍服从AGENTS。
评测优先复用已证明有效的每GPU多replica、每replica8-env动态批量配置，在峰值余量/CPU吞吐许可时取2–3replica，
不将命令模板里的单replica当吞吐最优。物理并行变化不改变canonical行/随机流/400数量。

完整9 GPUh和8GiB硬限不变。按launch/退出事件记录实际卡时和失败，必要的超时守护按剩余完整预算设置，
不靠频繁扫描共享metrics估计预算。正常运行持续等待退出；故障、资源变化或预算投影越界才做针对性检查。
若不能在预算内完成，保留已有checkpoint/原件并明确回报，不少算加载/失败、不缩成80行、不以部分面板选模型。
失败仅修复明确工程合同违例，不据负分改模型、seed、rank、学习率或辅助项。

正式运行入口为`scripts/train_writer.py`和`scripts/materialize_writer.py`；工程case/profile入口已由Git保留并退役。
图/合成、数据事件、ECP训练、bank/官方适配各有一个owner。旧generic CLI已经关闭，
旧model的native packing方法与旧materialization的selection/file/source工具暂为共享依赖，主讨论负责在本候选首次完整裁决时
随保留/退役决定处理剩余内部依赖；不新增其调用，不将旧Compiler恢复为fallback。当前不做无关全树清理。

验证限制：扩展Source-SFT测试`test_formal_mtbc_resolves_four_cards_only_for_explicit_checkpoint_migration`
在加载历史MT-BC配置时报告LoRA/source-base authority不一致。主讨论在集成前main `2a34b5dd`独立复现同一错误，
相关旧配置/contract源码无本轮diff；新训练只复用纯scheduler函数，并不调用该旧配置loader。
该既有问题留有记录，不因此改写历史MT-BC权威或重跑旧实验，也不宣称全库测试通过。

## 11. 270科学裁决后的450续训登记（2026-09-27，CPU准备已完成）

270原件已由主讨论独立验收，V151/L147/MT155/Source51；V对L的R/G/L=124/27/23，
V对MT=113/38/42，两臂都损失Goal26和Object16能力。首点不通过最终视频方法资格，
但已有较强完整能力，每task仅覆盖30/50条教学，故执行§7预留的450完整覆盖节点有判断价值。
原理解释和停止边界见机制分析§31；不是根据局部失败换架构、增加辅助或重启。

**唯一学习变化是原轨迹增加180更新/臂**，每臂新增20,160query、累计450更新/50,400query；
每task累计50条不同teacher/1400query。模型、合成、数据/采样、FM/noise、loss尺度、优化器/学习率与部署均不改。
从各自270完整ECP承接所有权重、AdamW状态、scheduler、sampler及rank RNG，既不fresh重训也不重置warmup。
world2及gpu02原6/7顺序、UUID、NUMA、microbatch28保持；两节点live准入和全部资源规则照常。
可保存360作恢复点，450为唯一新增正式读出；不评360，不自动630/other/controls/Test/RL或新probe。

**完整比较**：V450/L450各official correct400，同270的task/state/video ordinal/environment/policy RNG，
每臂仍固定8条state0 full、392条compact及实际T/T+1数组；不重跑任何270或旧基准。
报告V450对L450、各自对270、两臂对MT/Source的每task/suite、breadth、R/G/L、churn及成功集合重合。
弱视频差额不因共同能力高就变成视频通过；第二点仍主要V≈L且没有改变判断的完整正面证据时不自动延长。
正面结果也只允许另行裁定相邻资格/换正确视频，不能从450单点直接声称稳定或选最终checkpoint。

**本节最初派发仅CPU：预计20–35分钟、45分钟判断上限；完成后由§12单独授权GPU。** 复用执行者隔离开发树，从最新main接入本节，
仅扩展canonical训练/物化的登记节点和兼容续训检查。现有程序有意锁270，不能用runtime monkeypatch、
绕过CLI、跳过合同校验或未推送脚本直接调用私有训练函数到450。禁止模型/环境/GPU运行及新增工程smoke。
图与张量计算owner `conditional_velocity.py`、实际FM/VJP、data events/采样、通用ECP、source/评测后端均不改。
在现有training/bank owner与learning_spec内完成，不另建trainer、版本分支或一般迁移平台；相关CPU检查比例适当。

**显式兼容续训，不能冒称跨代码版本逐bit exact。** 旧270父commit固定0c4ea636；新代码差异必须仅为控制/记录。
同版本恢复仍使用完整合同相等。唯一270→450跨版本入口须显式核定父270节点、旧commit及合法父目录，
比较source/mode/model/LoRA/optimizer/initialization/sampler/trainable参数名/physical microbatch/topology等全部非迁移字段。
只允许已登记的Git身份、spec所在冻结目录及本续训登记信息变化；读旧/新spec核实所有数值学习字段不变。
不得简单删掉git/spec检查或放宽任意schema。270模型/optimizer/scheduler/cursor/rank RNG语义承接和270行metrics前缀均保留；
360同版本恢复也必须可用。保留明确父checkpoint与迁移记录，拒绝不相干父checkpoint、模式或其它数值变化。

新输出固定在原study根下`continuation_450/{V,L}`；bank各在该mode的`banks/450`，评测在`evaluation/correct400`。
旧首批目录不覆盖、不搬走。旧270 bank的spec引用旧冻结树，所以保留
`/data1/user/ymdai/projects/EMBER-conditional-velocity-formal`在0c4ea636；不能checkout新commit使旧证据依赖改变。
CPU交付后主讨论先独立审阅/集成并给出新clean pushed commit，再由指定执行者用另一个detached frozen树运行。
本节的CPU任务本身不授权GPU，后继启动按§12。

实测外推追加训练约2.14 GPUh，物化约0.196，两400约2.258；加加载/保存/失败余量预计4.5–5.6，
本后继硬限6完整GPUh，现场条件近似时GPU阶段约2–3h wall。原首批已用5.7032另行如实累计，不从后继账目隐藏它。
后继新增原件峰值4GiB，整个study根仍不超过8GiB（旧首批终值2.674GiB）；不复制大资产。
新冻结代码预计约253MiB、上限320MiB；开发+旧/新冻结三树预计约760MiB、总硬限1GiB。
建立新冻结树/运行目录及GPU前重新核data1独立quota和共享容量，原有无dummy、live两节点、全项目卡数上限保持。
原first-batch launch把Torch版本误记在python字段；已由evaluator preflight核明Python3.12.3、Torch2.11.0+cu128。
执行者已于00:19 UTC在launch_contract加入record_annotation并正确分列版本；运行环境未改变。
后继沿用已核明环境，不为文字勘误重跑，也不再修改首批原件。

本轮仍保留最小native packing/identity/selection内部共享依赖；旧generic CLI与旧Compiler不恢复。
因本次裁决选择继续既定学习程序，依赖退休触发明确顺延到450完整保留/关闭决定；不在学习途中混入无关源码重构。
只有指定执行者01a0dd6c-f2e5-7971-821a-56766e1c0f22承接具体实现/实验，主讨论负责审阅与科学裁决。

## 12. 450 CPU验收与第二批正式授权（2026-09-27）

CPU源码79c36003已由主讨论逐项审阅并通过a38d2822合入main；4文件+230/−34行，生产净增80行，无新模块。
改动只在既有training/bank owner、learning_spec与相关CPU测试；模型/真实FM/VJP、事件/数据、optimizer/LR、
通用ECP及官方evaluator数值实现无diff。主讨论独立运行canonical测试7/7通过，diff检查通过。
执行者只读核验真实V/L父270的迁移准入、270游标和旧bank可用性；这不等于已经实际GPU恢复。
跨版本严格限定固定父270和全部非迁移合同字段，360同版本恢复保持完整相等；不允许fresh或270重新训练。

准许唯一指定执行者以本登记随后的clean pushed main commit，在新的detached frozen树完成§11全部范围：
两臂分别从各自270完整ECP承接至450，每臂仅新增180步/20,160query；保留360恢复点，随后450 bank和各correct400。
不得改变旧0c4ea636冻结树或旧首批任何结果，不评360，不自动450之后节点、other/controls/Test/RL或额外模型/环境probe。
主讨论给出精确冻结commit后再开始；首次launch记录全部精确命令/环境、父节点、新目录、GPU/存储及预算。
更新次数/恢复语义/配对/报告要求沿§11，不重复建另一份通用训练或评测平台。

world2的gpu02/6,7、原UUID顺序、NUMA1和microbatch28固定；每次launch仍须同时live检查两节点，
选卡和共驻须满足余量、项目动态6/8卡上限及单节点至多6卡，不等待凑卡、不干扰他人。
若原训练拓扑无法合法使用，保留CPU交付并回报实质资源阻塞，不私自跨卡恢复或重新训练。
评测并行沿已证实有效的动态批量后端，按live余量安排，不改400行、官方算子、随机流或固定full集合。

本批预计4.5–5.6完整GPUh、硬限6，约2–3h wall；原首批5.703205 GPUh保持独立并报告总累计。
新增原件≤4GiB、study总≤8GiB；新增冻结代码≤320MiB、开发+两冻结树≤1GiB。
新树/产物前在strg01重核data1独立quota、共享容量及实际增长；不复制大资产，所有新增只写data1。
环境正确分列Python3.12.3/Torch2.11.0+cu128，并显式带现有EMBER_LIBERO_ASSETS_ROOT和新冻结PYTHONPATH。
包括加载、恢复、保存、物化、失败和评测全部卡时；用实际退出/完成事件登记，不反复读取运行中metrics作选择。
首批只记录过可用显存/无OOM，没有精确连续CUDA峰值；本批不为补齐该列额外跑profile。

完整交付包括180新增事件及270前缀/恢复信息、450模型与bank、两400和各自对270/MT/Source的保持/新增/丢失、
每task/suite/广度/成功集合重合、所有退出/失败/账目及最终本批无残留。完成或实质阻塞一次回报；
主讨论持续等待真实执行事件并独立裁决，不安排自通知。负分只淘汰实际检验的条件，不自动改模型或加步数。

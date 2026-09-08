# EMBER progress

更新时间：2026-09-08 CST。最新Owner要求：暂停后续执行，仔细分析已有结果；当前不训练、不评测、不自动恢复原goal。
长期K1/完整图/纯FM后独立RL目标保留，执行授权以以下暂停状态为准。

## 当前：Owner暂停执行，仅分析至400的情况

Owner最新明确要求“先别继续，仔细分析情况”：训练和评测全部暂停，当前只读分析已有证据与相关代码，不启动新实验或自动恢复旧goal/准备脚本。
此前“继续观察400及之后”的授权已被本条覆盖；500/600准备仅保留，不再属于当前执行计划。

canonical correct100/200/300/400为55/110/86/87。400 S/O/G/L=0/40/42/5、breadth4/8，尚未恢复200水平，也未qualified。
300→400 R/G/L=70/17/16、churn33/400、J=.67961；200→300为71/15/39、churn54、J=.568，退化主要集中Object global13。
两个节点均完整400rows/六worker exit0，实际每task50视频各一次、与100/200同一固定state-video映射；完整墙钟300/400为1348.54/1351.46秒。

200→400训练完整exit0，墙钟3287.09秒（54分47.09秒），均值15.583秒/update，峰值38.170GiB；400完整checkpoint约4.13GiB。
累计1600个真实K1条件/102400queries，各suite400条件，各task46–86次曝光。300/400 banks均全新400条件、正式检查通过，墙钟791.57/818.26秒。
400固定train24 held-action独立诊断已exit0，墙钟427.57秒、峰值11.484GiB；实际video/action frames/policy RNG与0/200一致。
FM0/200/400=.15145673/.11135264/.10627193，200→400改善4.563%，23/24task下降；task16微升.000386。尚不满足held FM平台条件，不能由闭环回落认定监督充分。
诊断复用f133原`_validate_actions`、每task128queries、micro4，无梯度/optimizer，sampler未消费；原config和48条0/200 `diagnostics.jsonl`保持，独立原件`segment200_400/step400/held_action/`。

400→600刚发起后按Owner要求SIGINT停止，torchrun3190898及四rank3192409–3192412均退出。未执行401：metrics仍400行、末步400；checkpoints仅100/200/300/400，原diagnostics仍48行。
本次外层exit1来自用户要求的SIGINT，不是训练数值失败。`segment400_600/launch_contract.json`已标记`owner_stopped_before_first_update`，prepared500/600脚本明确inactive。
当前无本任务训练、评测或物化GPU进程。已有400及以前checkpoint/rawrows/manifest/held诊断均保留。
最新结果与分析原件`k1_fresh/segment200_400/`；接下来核对当前任务成功集合、source增量、训练目标与闭环脱节，以及历史强路线的适用条件，结论出来后与Owner讨论。

## 本次暂停分析结论

400的87个成功中，两个cream cheese任务占79，加上同对象Long为84；source47也全部集中这三个task。存在Object11的真实增益（5→37），但跨对象/任务广度没有稳定扩展。
Object13从200的24→300的1→400的3，200全部110成功到400只保留67、丢失43、另获20。33个200成功连续在300/400都未恢复。
历史v6-fast确有106→64→111和132→117→138→143；但其每update24teacher条件×20queries，当前4条件×64queries。相近96k/102.4k queries下，历史4800条件对当前1600；旧每task50video池、当前16，实际当前400已见378/384种task-video。
这些是优化/数据组织的重要差异，尚未被受控实验证明为本次回落原因。旧架构、LR/scheduler及backend也不同，不能从历史回升推出当前原样续训必会恢复。
没有发现可直接解释回落的确定性执行错误；当前主要证据缺口是fresh K1训练task的真实闭环能力，不能用同task held FM替代。建议讨论冻结200/400同口径train96诊断后再决定恢复或调整，当前不执行。
完整只读分析与精确表格：`k1_fresh/paused_review_20260908/analysis.md`、`current_evidence.json`。历史核对由read-only history_audit完成，未新增GPU实验或实现。

## 已完成：整轮视频schedule修复与首段评测

Owner2026-09-08明确要求恢复canonical每task50视频各一次。fresh K1训练200已完成、100/200 checkpoint保留，不重训或重置。
初次100/200 banks各255条件不合规，已保留原manifest并添加sampling_issue.json；correct100启动后在rollout前SIGINT停止，exit130，无评测输出。
旧混合K64的99/95只作重复teacher分布下的探索分数，不满足正式400合同；旧train120四条held视频重复复用（71条件/120行），相关历史结论限于该分布。
canonical schedule、旧LoRA身份复用与启动前实际视频覆盖检查已实现，74项相关回归通过。真实旧bank被拒绝；每个checkpoint的255个已有LoRA均通过身份/生成合同核对，两份最终sealed manifest各400个实际task-video条件/400行，每task correct/other各50条各一次且逐行不同，两checkpoint映射完全相同；各255个hardlink复用和145个新生成已实物验证。
训练侧未来诊断改为states32–35的96行，单独建立source比较；真实核验与修复启动记录在`k1_fresh/schedule_repair/`。
修复已由clean pushed f1330697进入detached `horizon-schedule-runtime`；GPU02 tmux `ember-k1-schedule100`/`ember-k1-schedule200`分别在p1/p3补齐145条件，旧255各自hardlink复用。两份物化exit0，完整墙钟357.89/357.22秒；最终覆盖检查通过。correct100首次六卡请求被现有evaluator准入限制（共驻使用≤8GiB、free≥32GiB、util≤10%）在worker前拒绝，exit1、无rollout；原准备目录保留。已改GPU02 p1/3/6、每卡两worker重新发起（tmux `ember-k1-correct100-v2-gpu3`，日志`step100_gpu3/`、输出`validation_correct_step100_schedule_v2_gpu3_J0`）。该次正式结果及后续状态见下文。
source同口径已核对：原source120完整记录固定筛出states32–35得到15/96，S/O/G/L=6/0/7/2、breadth7/24。仅为原记录的预登记状态子集，不是新run；后续train96比较仍须验证候选source/policy/environment/RNG与逐行pairing。证据`k1_fresh/schedule_repair/source96_reference.json`。
fresh K1 macro100 canonical correct400已完整exit0：55/400，S/O/G/L=0/22/27/6、breadth6/8；按global tasks1/3/11/13/23/26/31/32为0/0/21/1/2/25/4/2。vs source47 R/G/L=25/30/22，churn52/400、J=.32468；完整墙钟1391.43秒。实际400行各task50视频各一次、逐行匹配最终manifest。尚未达到目标/Spatial覆盖/Long≥10，单节点不判断平台。macro200及相邻分析也已完成，当前结果与续训见顶部。
macro100对历史SFT109/107的R/G/L分别39/16/70、38/17/69，churn均86/400、J=.312/.30645；复用已审计历史compatibility检查，保留旧后端/rank128、非当前backend重跑的边界。原件`k1_fresh/schedule_repair/step100/correct_vs_historical_sft.json`。
以下较早“正在评测/已封存”描述仅指修复前阶段，不授权继续旧schedule。

## 当前授权与阶段

- 唯一active design：[过去定向完整Horizon Writer](docs/horizon_relation_video_writer_design.md)。完整图、集合能力、信息墙和科学目标保持。
- Meta是Writer内部读取模块；Writer整体端到端共同更新，source冻结。监督阶段无RL更新rollout、RL loss或trust/KL回滚。
- 当前训练/诊断/评测固定K1；K1绝对性能、相邻稳定、换视频鲁棒性和最终因果验证全部通过后再推进few-shot。
- goal持续active，无总预算/尝试上限。strict correct>145/400与全部资格、最终方法冻结32/8 fresh/Test尚未完成。
- 合同内实现、优化、正式训练、评测与依据结果继续已授权。无需重新审查全仓，也不等待逐项批准。

## K1切换：已落实

- 旧混合K run已到macro128完整保存并正常exit0，已停止；旧学习状态只作历史探索证据。
  原代码/config/frozen9ab1e710保持，原run、已完成99/95和混合K曝光证据保留。128的旧评测准备不自动执行other/train120。
- 实际sampler已改为真实K1视频抽样，移除K随机流；已验证，不只是修改cardinalities配置。
- Owner补充纠正覆盖前条checkpoint承接要求：本轮K1从fresh step0独立开始；旧混合K checkpoint只保留历史。实际停止128，共32768queries。
  Writer全部可训练参数重新初始化、LoRA合法identity，fresh AdamW/scheduler/sampler/RNG；冻结source/架构/资产复用。
  不继承旧训练学习状态；未提交的mixed→K1迁移实现已撤除，原exact-resume topology合同仍保持。
- 每步四个suite各抽一task、每条件64queries，共256；每条件梯度乘1/4后跨rank SUM，全局一次clip/step/scheduler。
  1/2/3/4卡仅改变条件分配；不会用5/6空rank或为凑卡扩大batch。18项训练检查覆盖三卡不均匀任务数、SUM权重与global cursor；9项严格加载、49项评测合同及1项异构物理microbatch检查通过。
- 当前优先correct strict400及廉价相邻raw-row分析；早期other后移，train120按实际能力诊断需要安排。
- K1条件吞吐短测已完成：仅为执行测量读取历史64checkpoint，不更新或保存训练状态；已比较FM microbatch4/8及frame4→8、edge8→32，
  使用完整训练池中位/最长视频，无截短或flow/horizon变更。既有混合K中77个K1条件均耗16.45s（FM11.91s），只是条件成本，不能冒充K1整步。
- 条件profile已完整结束：baseline4/4/8中位30帧17.72s、最长93帧23.85s；8/8/32分别11.43s、15.00s，
  提速1.55×/1.59×，每条件仍64queries。最长峰值35.404GiB（未含Adam），预留约2.56GiB moments后还须真实更新验证。
  profile在模块导入后计时232.23s，其中旧loader初始化/加载128.09s；选择microbatch8/frame8/edge32，保留既有activation checkpoint。
  原件`runs/analysis/horizon_relation_writer_20260908/k1_fresh/throughput_condition_profile.json`；profile未更新或保存checkpoint。
- 首个fresh K1段预登记0→200：短测完整逻辑更新约18s，预计约一小时，取100/200两个single checkpoints correct400。
  held FM0/200，train120按能力诊断需要；其他资格后移，旧24/64/128/192不再是未来安排。
- 双节点profile现场已检查，gpu02 p4有40314MiB余量、util2；四张旧训练卡保持，新增单卡短profile总共5卡。
  strg01 data1使用527970796KiB，soft1073741824KiB，当前旧run12GiB；临时profile只写小日志，不生成checkpoint或新cache文件。
- 两项独立实现已集成main并推送，任务工作树已清理；主线程继续正式训练、评测与证据裁决。

## Fresh K1集成与真实更新短测（2026-09-08）

主代码clean pushed f94c8b62；fresh profile在detached `.codex/worktrees/horizon-k1-runtime` 上实际运行。
GPU02 physical1/2/3/6、microbatches8/4/8/8，frame8、edge32；GPU2既有低util约8.8GiB作业，采用micro4保留余量。
前3updates为18.66/17.00/18.90s，完整256queries/update，SUM同步约.43s、Adam约.06s；含Adam峰值38.13GiB。
profile已在8步完整保存并exit0，32条件全部K1、2048queries，正式仍重新fresh step0、不继承任何profile状态。
完整8步均值17.345s，即14.759queries/s、0.2306训练LoRA/s；峰值38.147GiB，SUM同步.425s、Adam .0645s。
外层完整墙钟334.27s，包含导入/初始化/加载/保存/退出；短段摊销后6.127queries/s，不能只报稳态吞吐。
按真实update时间200步约57.82分钟，再加本段加载、held诊断与保存。
数据准备均值2.53s/condition、冻结prefix1.37s、FM9.55s、读取前/反向.212/.474s、图前/反向.090/.526s；
已取得实际提速，先启动正式训练，不继续扩大profile；数据准备可在真实运行中继续观察。启动/退出外层`/usr/bin/time`包含Python导入、加载、保存的完整墙钟。
新增24GiB预算覆盖第一正式小时与临时profile checkpoint、两正式checkpoint/atomic临时写、两validation banks和日志；
strg01 data1已用532410484KiB/soft1073741824，shared84TiB。GPU双节点现场与精确命令在`k1_fresh/`。

正式fresh K1已启动：clean pushed b6d70d98，detached `horizon-k1-runtime`；GPU02 tmux `ember-horizon-k1-supervised`、pane1042996。
新root=`runs/outputs/horizon_k1_supervised_v1_seed7_20260908`；run_contract确认formal、K1、world4、source trainable=0，
启动日志确认segment_start0/resume=null。held-action baseline0已结束，实际optimizer持续推进；没有继承mixed/profile checkpoint。
精确注册/命令/外层完整墙钟与日志在`runs/analysis/horizon_relation_writer_20260908/k1_fresh/formal_launch.json`、`formal.sh`、`formal.log`。

第2–16步整步均值15.469s；microbatch4/8的FM均值11.109/9.136s（不同条件，非配对速度试验）。
GPU02 physical2在15步中7步条件耗时最慢，超过其余三卡最慢条件的正差均值.570s；不能据此直接声称改8可提速多少。
现场该卡总46068MiB、他人8806+148MiB，约36.2GiB可用；micro8训练allocated峰值38.147GiB尚未计CUDA额外开销，直接改8不可行。
micro4为已验证可运行配置，micro6尚未测，不能称已找到最优值；本段结束后可按现场余量做一次限定测量。
100/200各自correct400请求、外层完整墙钟脚本和source/SFT比较准备在`k1_fresh/step100_evaluation/`及`step200_evaluation/`；
200额外比较100的严格配对成功集合。只准备correct，训练到200暂停后再启动；不把旧混合K节点充作fresh相邻证据。

Fresh K1 macro100已完整发布，继续同段训练200；checkpoint共4434752868bytes（约4.13GiB），manifest文件大小匹配。
trainer/scheduler/sampler/training游标均100，643项Adam状态及四rank RNG/topology已保存；仅核对序列化元数据，未重复restore或逐tensor扫描。
首100updates=400个真实K1条件/25600queries，四suite各100条件、24tasks均覆盖（8–25次）；整步均值16.029s，峰值38.160GiB。
原件`k1_fresh/step100_summary.json`；correct400尚未执行，不从训练loss或checkpoint保存推断性能通过。

## Fresh K1首段200完成，正在物化correct400（2026-09-08）

正式0→200已正常exit0，四个训练rank均已退出；100/200两个完整checkpoint保留。200的trainer/scheduler/sampler/training游标均200、四rank RNG/topology匹配，643项Adam状态保存。
200updates=800个真实K1条件/51200queries；平均15.848s/update，后100步15.667s，峰值38.160GiB。
外层完整墙钟3522.76s（58分42.76秒，含导入/初始化/训练/held诊断/保存/退出），吞吐14.534query/s；更新阶段16.153query/s、0.2524训练LoRA/s。
同一固定held-action query/video/RNG配对已核对，FM0→200=.15145673→.11135264，24/24训练task下降；不据此宣称闭环通过。
完成与训练汇总在`k1_fresh/step200_summary.json`，原始metric/exposure/diagnostic和完整checkpoint在正式root。

双节点与strg01已刷新，data1使用540974176KiB/soft1073741824KiB，shared84TiB；末checkpoint和两bank增长在原24GiB预算内。
clean pushed detached b6d70d98不变。GPU02 physical1/3各162MiB、util0；physical2既有8974MiB、util5，余37094MiB。
现在1号卡物化100 correct bank（tmux `ember-k1-bank100`）、3号卡物化200（`ember-k1-bank200`）；每bank255个独立条件/400配对rows。
2号卡执行micro4/6限定短测（`ember-k1-micro6`），只读200 Writer后在临时进程做完整反向和诊断Adam，不保存或继承学习状态。
注册与现场证据`k1_fresh/post200_launch.json`、`post200_gpu_preflight.json`；三个入口均记录外层完整墙钟与退出码。
待物化完成后用现场合适GPU、每卡2个persistent workers分别执行100/200 strict correct400，再做相邻成功集合、source47和历史SFT比较。
尚未启动closed-loop评测，未做other/final controls/Test，尚未qualified。下一段纯FM/RL裁决仍以真实闭环证据为准。

## 历史：初始纯FM安排与原注册

以下24/64/128/192、混合K与密集other是切换前的历史执行合同，不覆盖上面的最新安排。

纯监督相关15项FM/native/训练检查、物化49项检查通过；增加独立held动作不消耗训练sampler的测试后，训练10项通过，CLI导入通过。
结构检查REVIEW：监督路径替换joint而非增加并行训练分支，active source净减111行（包含测试重命名）；现有复杂合同检查职责保持。
正式首段计划gpu02 physical0/1/3/6，world4，每卡真实一task，FM microbatch4、observer chunk4、edge chunk8，保留完整图。
最新双节点现场核验这些卡util0，p0/1/3仅小context、p6既有4.6GiB且有充分余量；不干扰其它进程。
data1 quota使用516182860KiB/soft1073741824KiB，shared84TiB；新监督至192节点预算48GiB，计入4个完整Adam checkpoint、
atomic临时写入、train/validation LoRA banks、raw rows/logs。当前joint末profile4.2GiB、analysis1.2MiB、scratch98MiB；其余既有用量由quota覆盖。
新root=`runs/outputs/horizon_supervised_v1_seed7_20260908`；现场证据/精确命令登记在`runs/analysis/horizon_relation_writer_20260908/supervised/`。
已从clean pushed `265ef31b` 的detached `.codex/worktrees/horizon-supervised-265ef31b` 正式启动；
gpu02 tmux `ember-horizon-supervised`，torchrun3925038、四个rank3925226–3925229已完成首段并正常退出。
24updates/96conditions/6144queries，K1/2/4=34/30/32，覆盖23/24tasks（task4尚未抽到）；平均30.40秒/update，总1027.99秒。
独立held-action FM由.151447降至.131235（21/24task改善），不代表闭环或平台。macro24完整4.13GiB，sampler/scheduler及643个Adam状态step均24。
summary=`supervised/step24_summary.json`。已刷新双节点/strg01，data1使用520203820KiB；step24 bank预算512MiB、评测1GiB计入48GiB总预算。
train120_step24已物化71条件（350MiB）并完成J0：17/120 vs source19/120，S/O/G/L=6/4/6/1，breadth8/24（source7）。
strict paired R/G/L=7/10/12，churn22/120，J=.24138。5卡×2persistent workers耗489.22秒，全部完成exit0。
held FM下降没有转化为总体闭环增益；24只是早期节点，不视为平台，不转RL。按预登记继续监督64再做strict400。
另修复监督数据采样对同一全局episode索引的逐query重复复制，10项训练检查通过；sample/RNG/目标不变，当前frozen首段未热改。

## 历史64节点结论与正在收尾的128旧配置段

监督64已完整exit0；累计256conditions/16384queries，24tasks各6–17次曝光，K1/2/4=87/77/92。
held FM0/24/64=.151447/.131235/.123122，24→64全部24tasks改善。macro64完整4.13GiB，checkpoint检查通过。
24→64段平均32.03秒/update、总1532.41秒；原run_contract保留初始265ef31b，续训/物化实际frozen9ab1e710另有segment记录。
summary=`supervised/step64_summary.json`；训练仍只有pure FM，source始终冻结。

train120 held-video J0=34/120（24步17，source19），S/O/G/L=12/7/12/3、breadth14/24。
vs24 R/G/L=14/20/3，churn23/120，J=.37838；vs source为13/21/6，churn27/120，J=.325。
Spatial task5仍0/5而source4/5；不能只报告新增而忽略遗忘。原件`step64_evaluation/train64_vs_*.json`。

旧重复teacher抽样的validation探索correct/other=99/95（不合规正式400）（source47，历史SFT109/107），均完整exit0。
correct S/O/G/L=9/53/35/2、breadth8/8；other为4/51/36/4、breadth7/8。
global tasks1/3/11/13/23/26/31/32，correct分别5/4/31/22/1/34/1/1，other为1/3/31/20/1/35/4/0。
correct→other R/G/L=77/18/22、churn40/400、J=.65812；总分下降4，但J未达≥.80，Long成功无保留。
correct vs source R/G/L=35/64/12、churn76；other vs source为34/61/13、churn74。
SFT400 S/O/G/L=0/69/22/18，当前主要弱项是Long/Object，Spatial/Goal较高；旧SFT backend/rank边界保留。
correct99未达严格>145目标，Long2<10，无相邻资格或视频因果证据；仍未qualified，也不是监督平台。
完整裁决、比较和输入配对证据均在`supervised/step64_evaluation/decision_after64.json`及该目录。

两replica correct总949.35秒（启动124.97、shard窗口820.09，120809 env steps）；三replica other总1405.26秒
（启动548.84、shard窗口852.64，120820 env steps）。15worker均有完整权重成功日志，无OOM或队列错误；
最拥挤卡实测42372/46068MiB。增加replica未带来实际收益，后续恢复每卡2replica，不另开扫描。
只读源码核查发现初始化会重复构造大模型，GPU分配早于checkpoint加载；没有修改第三方loader或引入no-init路径。

根据held FM持续改善、train闭环17→34与首次held99/95，按已登记继续纯监督至128，不转RL、不做最终controls。
同root、同config、同world4与GPU02 physical0/1/3/6，从macro64恢复完整Writer/Meta/Adam/scheduler/sampler/RNG；
clean pushed frozen=`9ab1e710`，tmux `ember-horizon-supervised` pane573577，精确命令/provenance在`supervised/resume_to128.json`和`.sh`。
新日志/退出码为`resume_to128.log`/`.exit`；已完成恢复后的65–67步，Writer/Meta梯度非零，optimizer/sampler继续推进；不能把旧completion64当作128完成。
新双节点现场已刷新，所选四卡util0、低context且有余量；strg01 data1使用527869508KiB/soft1073741824KiB，
监督run约12GiB，原48GiB注册预算尚余36GiB覆盖后续checkpoint、atomic写入、banks及评测。未使用Test。
128节点物化/评测及对64相邻比较已准备于`supervised/step128_evaluation/`；尚未启动，默认2replica，现场资源另核。

## 已结束的联合profile

最后e1ea3596 profile已正常exit0，全部训练/验证GPU进程已退出；两次更新均接受（alpha1、1/16），共512FM queries/32episodes，
每轮144.49/156.98秒，总446.30秒，完整macro2约4.13GiB；仅作为历史机制与成本证据保留。
非零LoRA真实native flow对64个physical PEFT decisions输出一致；真实physical36 expert targets A/B为BF16，2 head targets为FP32。
实际batched仍KL .016349，记录其kernel差异，不要求逐元素一致。原件joint_profile/native_fixed/与native_e1ea3596/。
此前joint阶段的24/48/72登记草案及拒绝暂停规则被最新Owner阶段选择覆盖，未启动对应formal run。

以下保留本轮早前历史执行记录；其中“下一步/正在”等时态只适用于当时，不能覆盖上述最新监督授权与计划。

## 已完成：八档回溯与数值接口诊断

`ae9507b5`已通过23项针对性CPU检查、提交推送，并从clean detached `.codex/worktrees/horizon-profile-ae9507b5`启动。
gpu02 physical0/1/3/6、world4，fresh profile目标2 accepted、最多4 attempts；保持完整图、64 FM queries/task、
同一个Adam方向、原始采集均值与KL≤0.02，只检验回溯到1/128是否解除原四档搜索的接受停滞。
输出`runs/outputs/horizon_joint_profile_ae9507b5_gpu02p0136_20260908`；精确命令/log与两节点现场证据在
`runs/analysis/horizon_relation_writer_20260908/joint_profile/finite_ae9507b5/`。本次data1 quota使用507313652KiB，
soft1073741824KiB；新增峰值预算16GiB（完整Adam checkpoint、可能的同合同恢复及临时写入），共享剩余84TiB。
既有v2 run8.3GiB、其余EMBER444GiB、其余个人projects27GiB（du按此前已统计路径排重）。全部大资产复用。
四轮迭代与完整checkpoint已完成exit0（总804.43秒），四个主训练进程已退出：1 accepted/3 rejected，1024计算FM queries中仅256进入接受更新，64 episodes。
四轮耗时140.66/172.43/175.74/179.46秒，allocated峰值25.69GiB、reserved31.22GiB；全部同版本自比较KL0。
新增四个更小尺度没有增加任何接受更新：step2/3/4在1/128仍maxKL=.02300/.07222/.03424，且响应不单调。
本次有限回溯范围假设未获得支持，不继续添加尺度或同样续试，不启动formal。
下一固定只读诊断在joint_profile/adjacent_b/：同一已接受Writer生成的A/B，固定A与全部输入，对比原B、
完全相同B重放、每个非零B朝+infinity改一个BF16可表示单位。只定位执行端最小量化改变是否足以产生KL变化，
不替换训练采集m_old、不训练、不放宽阈值。profile和此诊断都不是formal资格证据。
相邻B诊断已完成exit0（152.35秒）：原B重复完整重放KL/动作差均0；仅移动一个BF16相邻值时，
task KL=.0343270、max动作差=.0512085、mean abs=.000832334。680448个非零B元素各改一个相邻值，
聚合relative L2=.57214%、max元素差9.53674e-7。执行端单独足以产生此量级变化，但这还不能证明dtype缺陷或解释全部拒绝。
据此登记下一个限定因果对照：同一固定生成A/B及真实64 decisions，只将动作expert运算改FP32并启用TF32，
视觉/语言prefix仍BF16，KV和B仅同值上转适配，无新权重信息；比较原B/同B重放/相邻B，并报告对BF16 baseline的动作差和成本。
这是针对接受停滞的数值分辨率假设，不是正式协议已变更；不训练、不扫多个dtype、不更改Sigma/0.02。
若后续采用此执行合同，必须覆盖采集、VJP、trust、物化执行和同口径baseline，不能混用旧BF16分数或旧采集均值。
证据入口joint_profile/expert_tf32/。
本task已完成且clean的ba556b98/07871988 detached worktrees已移除，Git与所有run/checkpoint原件保留。

## 前一batch修复profile与exact-resume结果

clean pushed `07871988` 的batch修复profile及exact-resume已全部完成exit0：
同gpu02 physical0/1/3/6、world4，目标2 accepted、该诊断segment最多4 attempts，必要时依据结果继续，不是全程上限。
输出 `runs/outputs/horizon_joint_profile_07871988_gpu02p0136_20260908`，launch/log在joint_profile/matched_07871988/。
现场四卡依旧低util且有足够显存，data1 used498649088KiB；按两个完整Adam checkpoint及恢复/临时写入重算新增峰值16GiB。
旧9c5a1c2b task-owned clean worktree移除，失败证据/唯一checkpoint和Git保留。

以下是修复前已完成profile的合同与证据：

从clean pushed `9c5a1c2b` 的detached `.codex/worktrees/horizon-profile-9c5a1c2b` 已完成两次尝试迭代的完整联合profile，exit0；
gpu02 physical0/1/3/6、world4，每task一张卡，NUMA-local、deferred NCCL、NCCL_P2P_DISABLE=1。
现场四卡util0，p0/1/3仅小context，p6既有约4.6GiB；FM microbatch4，完整图与64 queries不变。
启动前再次核验两节点和strg01：data1使用497104064KiB/soft1073741824KiB，shared84TiB；新增峰值预算8GiB，
计入一个完整checkpoint和临时写入；另有source基线预算1GiB。复用全部模型/data/env。profile不参与科学选点。
精确命令、完整现场证据和日志：`runs/analysis/horizon_relation_writer_20260908/joint_profile/retry_9c5a1c2b/`。
输出：`runs/outputs/horizon_joint_profile_9c5a1c2b_gpu02p0136_20260908`，tmux `ember-horizon-joint-profile`。
profile每轮额外测当前参数版本的trust子集KL，定位已观察到的跨batch数值底噪；formal不默认重复该forward。
完成后核对实际episode/RL信用、FM曝光、全局SUM、候选接受/拒绝、checkpoint恢复与墙钟，再登记正式节点。

前两launch均未采集或更新：ba556b98缺资产环境变量；d956956d只设缓存父目录，真实env创建时缺scenes。
两次failure log/run contract均保留；已把准确固定revision路径登记到数据authority，首轮四task
（34/25/15/2，各suite一个）已实际创建四env、reset初态并10步settling，RGB256×256，exit0。
证据`joint_profile/assets_environment.log`；不能把前次仅元数据检查称为环境检查通过。
本task创建的已退出d956956d clean worktree移除，代码由Git和failure合同保留；其它历史worktree未动。

独立gpu02 physical4已完成source train24×初态32–36的J0/JΣ paired120，先J0后JΣ，
冻结ba556b98、2 persistent replicas；登记和命令在 `runs/analysis/horizon_relation_writer_20260908/source_train120/`。
错误父目录的0-row失败保留，修正固定revision后重新prepare；living-room/study真实reset/step另验证通过。
source J0/JΣ均完整完成：19/120与22/120，S/O/G/L=8/0/9/2与10/0/10/2，breadth均7/24；
strict paired R/G/L=15/7/4、churn11/120、J=.5769，summary/paired_comparison原件在source_train120/。
两组全部worker exit0，p4释放，不再运行baseline。
这是初态32–36的训练任务诊断参考，不能与历史validation47/400或旧初态的train16/120混用。

采样节点规划（非学习结果）见`joint_profile/exposure_node_planning.json`：假设所有attempt接受，
24次只覆盖23/24 tasks、6144 queries，48次覆盖24 tasks、12288 queries（每task4–16个conditions），
96次24576 queries（每task8–25个conditions）。真实曝光以exposures为准，拒绝也推进采样。
24节点仅作早期获取诊断，不能当作已充分学习所有task；strict400节点待本profile后、formal开始前冻结。

## 2026-09-08 完整联合profile结果与最小数值修复

9c5a1c2b profile两轮分别150.73/149.88秒；含模型加载/保存总456.06秒。8条件/512FM queries/32真实episodes，
2个mixed reward groups，各有真实非零RL credit；接受0/2。峰值allocated22.94GiB，reserved约30.3GiB。
完整macro_00000002为1.378GiB，Writer/Meta/4rank RNG、sampler next_step2、attempted2/accepted0、optimizer空moments、
warmup未前进、metrics_rows8均与全拒绝状态相符。真实非空moments恢复和接受后Meta学习尚待验证。

current-version trust KL：task2=.01265，task5=.01472，task21=.07968，其它五个task均0。
非零恰好发生在提前成功、采集active batch从4缩小的tasks；task21四episode全成功、RL梯度为0仍超0.02，
证明至少存在采集/重放数值batch接口缺陷，不能靠无期限重试或放宽阈值进入formal。
静态复核两侧均no-grad/BF16/同flow functional substitution，未发现独立dtype或B=0分支；实际底层数值由诊断裁决。

最小修复正在main验证：每decision记录原flow_batch_size；RL VJP与trust按该size重放，尾组复用真实记录补齐，
补齐项zero cotangent且不进KL/episode/task权重。保持采集m_old、Sigma、0.02、完整10步flow与候选自己的Meta读取。
update_version改为same_version_fm_rl_collected_batch_replay_v2；不将旧profile作为此版本正式训练或exact-resume证据。
22项相关CPU检查通过（包含batch数值oracle和完整信用权重）。gpu02p1真实task21复现exit0：
四episode均成功，步数128/118/194/140与原profile一致。64保存decision含原batch4/3/2/1共54/4/1/5条；
原size重放KL0，改变记录排列及补齐行后仍0，统一batch4则.02439。该统计使用全部64保存decision，
不能与原profile随机选16的.07968直接视为同一子集。接下来从推送frozen版本重跑完整联合profile，
路径joint_profile/trust_replay.{log,json}；启动前双节点和data1 quota498549752KiB复核，预计小于0.3GiB临时观测，仍在8GiB预算内。

## 2026-09-08 batch修复后的接受停滞与下一个优化假设

07871988两段共8attempts、1accepted/7rejected，32conditions/2048计算FM queries，只有256 queries进入被接受更新，
128真实episodes；所有32个current-version task KL均0，接受后Meta梯度非零。完整macro4/macro8各4.130GiB保留。
首段741.75秒、恢复段803.27秒（均含加载/保存）。峰值allocated约25.7GiB、reserved32.1GiB；没有OOM或source梯度。
exact-resume真实核对通过：从macro4读出的四task、video、query_seed、episodes全部随机流、occurrence/frames与step5逐项一致，
恢复Adam643parameter states step1、scheduler1、attempt4/accepted1；拒绝不回退sampler。
仅一次接受不能构成足够有效学习，未进入formal，不能报成新图性能失败或成功。

限定数值诊断用train21固定旧A/new B×{0,1e-12,1/8,1}，完整64保存decisions及原batch，source/Writer冻结：
0及tiny非零B（max1.709e-16）KL和动作差均0；1/8 KL.05858/maxdelta.09534，1×KL.03123/maxdelta.04779。
没有发现任何非零B导致固定扰动的分支；剩余非单调响应尚不能区分模型非线性与精度效应。
原件joint_profile/numerical_limit/；不与随机16decision trust值混用，不作为模型选择或完整参数候选插值。

已登记下一优化假设：保持同一个Adam方向、LR/梯度/Sigma/原m_old/0.02不变，将有限回溯从四尺度扩至八尺度1…1/128，
检验原候选范围是否过早截断可行步。update_version=v3，源码仍是一条canonical路径；不扩dtype、不改变科学性能线。
先fresh完整profile检验实际alpha/接受率/成本，再冻结formal节点；若仍停滞，停止同样续试并重新定位。
这是运行更新问题的有据检验，不是靠低步长掩盖已经发生的正式性能失败；新图尚无formal qualification分数。

## 2026-09-08 单卡真实机制结果与联合profile准备

真实机制已完成exit0，成功源代码7da77fb1；证据`runs/analysis/horizon_relation_writer_20260908/mechanism/summary.json`。
R为真实action_out_proj输入同一对象、FP32[2,50,1024]，同一次prefix产出最终Z[2,270,2048]和18层KV。
同batch2/4的完整10步flow分别与canonical推理一致；真实FM小步后A/B、过程和Meta梯度均非零，source无梯度。
最长合法pool为task38：K1 demo0共93帧，K4 demos0/1/3/6为93/89/87/85帧；64FM queries整段VJP含prefix
耗时19.38/47.68秒，峰值35.40/35.72GiB。该数值不含rollout、Adam moments和trust，不能用作formal iteration成本。
初次scratch统计后缀错误已修复并保留失败记录；没有source缺陷、截视频或正式学习证据。

flow VJP batch4为1.451s/11.24GiB，batch2为1.488s/10.70GiB，采用RL microbatch4以提高实际吞吐。
FM microbatch8已接近单卡峰值，加Adam/NCCL及其它用户context后余量不足；联合profile先用FM microbatch4、
其余完整图不变，测是否支持四task四卡并行与实际吞吐。不是降低FM query64或以最低显存为目标。
跨batch2→4的identity动作KL实测0.028335，须用实际收集/候选重放的检查子集判断数值底噪和误拒绝；
不为逐元素一致强制batch1/扩dtype或放宽科学阈值。尚未开始formal训练或得到新闭环分数。

## 2026-09-08 新图接通与真实验证（进行中）

已集成完整Horizon过程图、native D、最终R/Z读取、真实四episode采集、十步可微flow、Gaussian信用和trust候选，
同版本FM/RL合并AB cotangent后一次Writer/Meta重放；独立采样、runner/config、checkpoint和Sigma/0评测也已集成。
旧layered/coordinate、旧FM-only入口与旧18层边界capture已退役，历史工作树/证据保留。
主树全量CPU检查262项通过，覆盖过程依赖、解析flow VJP、Gaussian/Q/M/LOO、候选回滚、拒绝曝光、checkpoint恢复与评测配对。
这是工程合同证据，不是闭环科学通过；尚无新正式checkpoint或行为分数。

结构自审：新职责分为完整过程图、native因子、十步执行、环境采集、信用/trust、同版本联合调度；复用现有FM、
LoRA、checkpoint和evaluator。源码净增长超过1000行来自原来缺失的真实RL与过程模块，不保留旧平行训练路径。
`joint.py`的单condition更新保持一个清楚的同版本生命周期；共享大evaluator只增加必要接线，探索计算归独立owner，
没有复制大forward或按token执行。尺寸/复杂度review信号由上述职责界面承接，无新增超过800行的实质计算owner。

2026-09-08现场存储：strg01 data1使用497206440KiB、data0使用57652716KiB，各soft1073741824KiB，
hard1084227584KiB；shared分别84TiB与1.9TiB。du workspace443GiB、其余data1个人32GiB、data0个人55GiB。
机制/profile只写小日志（峰值<1GiB），不存大中间tensor；首段完整checkpoint暂估4–5GiB，
相邻checkpoint加物化/评测暂估新增峰值32GiB以内，正式launch前按实测和实际节点重算。
GPU02 physical1（GPU-19cb2d30-d2ca-77b5-54a5-6b23fd2eede4）现场162MiB/util0，仅其它用户148MiB context，
NUMA0；按共驻合同用于单卡机制验证，准确命令和结果归`runs/analysis/horizon_relation_writer_20260908/mechanism/`。
这是现场调度选择，不是reservation；正式launch前再次核验所有将用设备。

## 当前执行记录

- 全程goal已启用。当前完整设计为首版起点；保留证据支持的具体方法修订自主权，修改前查最近等价历史并更新合同。
- 已消费HANDOFF；长期要求、方法、计划、证据均在正式文档，该临时入口已随状态提交92a2673f移除并push。
- 初始architecture guard基线无改动（+0/-0）；新架构将替换旧layered/coordinate行为，复用现有source、FM、LoRA及评测基础。
- 只读审计覆盖全部当前src/tests/scripts/configs与项目文档；大型manifest按程序核对结构/任务/episode/frame及文件字节，不逐字展开重复元数据。
  历史index全部843导出、511原件存在，95面板24100行的成功数/suite/breadth与本地原件outcome向量完全一致；这是复核，不是新评测。
  source/tokenizer/71source与40target HDF5存在且匹配manifest；没有新hash或held trajectory梯度读取。
- 纯过程图在`codex/horizon-graph`独立worktree；物化/评测迁移在`codex/horizon-evaluation`独立worktree；主agent在main负责native/runtime/FM+RL/数据/训练与集成。
  两worktree都基于92a2673f，写范围不重叠；使用using-git-worktrees隔离，验证后由main统一集成push。
- 关键实现风险：旧capture取final norm输入，必须换action_out_proj实际输入；冻结prefix新增最终Z，静态语言读取补位置；
  RL必须是显式action Gaussian、4独立episode、LOO和Q/M、完整10步可微重放，同版本单次更新与trust拒绝不倒退sampler/RNG。
  旧reward antithetic/成功expert occupancy与旧cycle/cosine均不作为新训练路径。
- 下一项：真实source的R/Z、完整flow VJP、最长K1/K4、真实联合更新及环境检查；按实测成本登记strict400节点并冻结首段合同。

## 暂停前执行记录（以下“当前/下一步”只表示当时时点）

下文关于旧双向图、旧未来帧不变性检查及旧待执行对照的表述均是历史合同，不能覆盖本页当前状态和新设计的过去单向选择。

held-video train120已完成exit0：18/120（Spatial10/Object1/Goal7/Long0），breadth7/24；对source16为RGL11/7/5、
churn12/120、J11/23。seen21→held18为RGL14/4/7、churn11/120、J14/25；held192→384为22→18、11/7/11、churn18/120。
两诊断eval耗时495.01/503.80秒，所有比较验证已完成，结果见video_novelty_diagnostic_results.json与decision_after384.json。
当前没有GPU作业。原run永久止于384；下一项为native线性因子读出实质对照，保留target/rank独立性与完整上游图，
必须fresh共同训练并报告参数/子空间/成本取舍。尚未实现或启动该对照，不能宣称已定位唯一decoder根因。

Owner指出当前性能连基础SFT约108都未达到；历史SFT两个相邻点为109/107，当前67/64应按性能失败处理，
相对source47的局部增益不足以支持继续本配置。执行裁决：本train24 run止于384，不执行576/624/672；长期目标继续。
熟悉视频train120已完成exit0并通过完整source配对：21/120（Spatial10/Object3/Goal8/Long0），breadth11/24，
对source16为RGL10/11/6、churn17/120、J10/27。缺口已出现在训练任务与熟悉视频，不能仅归因为未见task迁移。
seen/held两组LoRA物化均exit0，分别107/68条件、约552/351MB、325.36/256.92秒（含启动，近似）。
held-video train120在gpu01physical0/2/3/5评测已完成，frozen97a8a24a、12workers；启动前两节点live核验四卡空闲，
quota496858640KiB，现场与命令见eval_s384_train120_held_video_launch.json。完成后做接口区分性实验，不再无依据续训或几何小扫。

384 other已完成exit0，64/400（Spatial4/Object31/Goal26/Long3），breadth6/8；对source47为RGL28/36/19、
churn55/400、J28/83；对192other72为RGL47/17/25、churn42/400、J47/89。
384跨视频correct67→other64为RGL52/12/15、churn27/400、J52/79=.658228；完整两arm均未达资格，decision_after384.json已封存。
两组eval实际墙钟1111.27/1072.95秒，均12workers。当前没有训练作业或validation评测作业。
冻结384 seen/held训练视频诊断在gpu01physical0/2并行物化已完成；两节点live核验所用卡空闲0MiB/process0，
quota495972916KiB、run5.7GiB、analysis29MiB，预计额外峰值1.1GiB仍在原32GiB预算内。
两组各120行、107/68个唯一条件；命令与现场在materialization_s384_novelty_launch.json，seen21/held18均已完成。

384 correct strict400已完成exit0并通过完整配对：67/400（Spatial3/Object33/Goal27/Long4），breadth5/8；
对source47为RGL30/37/17、churn54/400、J30/84；192→384为RGL46/21/23、churn44/400、J46/90=.511111。
Object1保留局部改善，Long1首次4/50；Goal6从36降至27，Goal3由1降至0。尚无广泛、稳定迁移，未满足qualification。
same-task-other从frozen97a8a24a、gpu01physical0/2/3/5启动并已完成，3replicas/card；两节点现场显示四卡空闲，
p1被其它任务使用，未等待第五卡；quota495966384KiB，命令/现场见eval_s384_same_task_other_launch.json。
在任何384训练侧诊断分数出现前，已登记train24同120初始化的seen-video0–15与held-video46–49两组正确视频诊断，
固定checkpoint384/K1/seed20260907，不更新参数，不作模型选择。已见组全部视频出现于训练，89/120行也曾作为K1条件。
登记及准备命令在video_novelty_diagnostic_registration.json；other已完成，现已live核验资源并执行两组诊断，
依此区分已训练task/视频、新视频与新task接口，再做实质、可区分的修正；当前run已决定止于384。未选selected checkpoint。

384步exact-resume已完成exit0，完整checkpoint与completion_to384保留；累计1536条件/24576queries，
每task64条件/1024queries，实际K、任务权重、视频/query跨episode及连续step/cursor检查通过。
累计实际更新6385.93秒，本段含加载3292.72秒；exposure_cost.json保留192与384的分段和累计成本。
384 correct/other两组LoRA已从clean pushed frozen evaluator97a8a24a完成物化exit0，分别使用gpu01physical0/2；
255/259套文件分别1,316,582,728/1,337,237,521bytes，含启动的近似墙钟576.28/597.26秒。
两节点live检查所用GPU均0MiB/process0；data1 quota493190120KiB、run3.0GiB、analysis16MiB、sharedfree84TiB，
预计新增2.7GiB，仍在32GiB阶段峰值预算内；现场与命令见materialization_s384_launch.json。
384双qualification与冻结训练视频诊断均已完成，无selected checkpoint。

当前train24首段192步完成exit0：768条件/12288queries，每task32条件/512queries，真实K1/2/4各10或11次；
全部task权重.25与video/query跨episode角色核对通过。22个task覆盖16条训练视频，task15/34各15条；
实际更新3221.41秒、含加载3347.23秒，3.814queries/s；rank0 allocator峰值34.326/38.201GiB（不冒充全rank峰值）。
完整checkpoint及completion_to192/曝光成本已保留，后续exact-resume仍锁原world4和完整672步schedule。
三组物化（validation correct/other与train120correct）从frozen evaluator97a8a24a完成exit0，255/259/68套唯一完整LoRA，
分别覆盖400/400/120行；含SSH启动的近似墙钟613/629/260秒。约3.00GB生成文件，仍在预算内。
192correct strict400与train120均完成exit0并通过真实配对比较：
correct69/400（Spatial3/Object29/Goal37/Long0），breadth5/8；对source47为RGL34/35/13、churn48/400、J34/82。
train120为22/120（Spatial11/Object1/Goal10/Long0），breadth8/24；对source16为RGL11/11/5、churn16/120、J11/27。
correct增益主要来自未见Object1（global11）5→29；Goal6从41→36，Long仍0。仅有局部改善，未满足目标或相邻资格。
实际eval墙钟correct四卡1063.75秒、train120单卡1316.09秒；完整per-task/RGL与source合同见train24_shared/panel_summary.json及各*_vs_source.json。
same-task-other strict400已完成exit0，72/400（Spatial4/Object31/Goal37/Long0），breadth5/8；
对source为RGL35/37/12、churn49/400；对correct为RGL60/12/9、churn21/400、J60/81=.740741，未达到跨视频J≥.80。
五GPU15workers耗时928.99秒；两组总分接近，不等于成功集合稳定或视频必要性成立。
本段从macro_00000192 exact-resume至384已完成，frozenE4、gpu01physical0/2/3/5、world4与完整672步schedule不变；
命令launch_train_to384.sh，日志train_to384.log/exit，tmux ember-layered-train24。两节点live核验四卡均空闲0MiB/process0；
quota493189140KiB，当前run3.0GiB/analysis16MiB/sharedfree84TiB，仍在32GiB新增峰值预算内。
续训依据：两组正确视频均有相对source重复的Object局部增益，当时每task512queries；按已登记节点增加至1024queries，
观察增益扩展与相邻稳定性。仍未满足absolute/breadth/Long/跨视频资格；不唯一归因为曝光不足或共享冲突。
完整裁决与launch在decision_after192.json、launch_resume384.json。
物化曾使用gpu01physical0/2/3；
launch前两节点live核验三卡空闲0MiB/process0，quota490060680KiB，预计新增物化3.2GiB，处于32GiB阶段预算内。
命令与现场见train24_shared/materialization_s192_launch.json、eval_s192_correct_launch.json及eval_s192_train120_launch.json；
192双视频strict400和train120均已结束并裁决续训；下一384双视频strict400将形成首组相邻证据。

当前train24协议已登记：保持当前图、fresh Writer+Meta，固定24训练tasks，每update4tasks×16queries，真实K1/2/4；
warmup24/full cosine672，首段停192。节点192/384/576/624/672的双视频K1 strict400与相邻资格见设计§13.3.1及
train24_shared/registration.json；192/576另有train120 correct/source诊断。无validation梯度、无Test/负视频controls。
预算新增峰值32GiB，data1 quota490051060KiB/soft1073741824/sharedfree84TiB，复用全部大资产。正式训练已从clean pushed frozen e4ca5998启动：
`.codex/worktrees/layered-train24-e4ca5998`，gpu01physical0/2/3/5，world4，tmux ember-layered-train24；
run为`runs/outputs/layered_relation_train24_joint_e4ca5998_gpu01p0235_20260907`，精确命令在train24_shared/launch_train_to192.sh。
launch时两节点检查完成，四卡显存0/process0，最新quota490054904KiB；前两步正常，Meta第二步出现梯度，首步identity的Meta零梯度符合合同。
训练侧source120准备被旧screen只支持10/50的入口拒绝，未产生rollout；已仅为显式训练子集加入5-state支持，
14 targeted tests通过，validation/Test边界仍拒绝；已用frozen eval97a8a24a补齐源参照，不改变训练配置。
source_train120已全部exit0：16/120，Spatial6/Object0/Goal10/Long0，breadth7/24；完整raw rows和120行配对检查保留。
实际单GPU3workers耗时1234.63秒。该新训练侧参照仅配对192/576 train120诊断；validation参照仍是历史严格400的47。

三轮short4的训练/物化/闭环及冻结diagnostic均已完成；这些short4作业均已自然结束。初始化对照16/48/96的correct为4/5/6、other为4/8/6，
96K4为8/40；所有breadth2/4，Object/Goal均0。48→96 correct RGL5/1/0、other6/0/2；96跨视频5/1/1、Jaccard5/7。
完整事实及边界见research_history§15。保留局部Long正证据，不恢复原配置无依据续训，不将其宣称为广泛共享能力。
下一单变量fresh对照已在target_rank_readout_control/registration.json登记：只将末读出[p]改为[target,rank,p]，+77,696参数；
坐标std1、完整图/Meta、identity、数据/K/queries/优化器/96步节点与world3均保持。canonical源码及模型合同已更新；182 CPU tests/17.08s通过，包括单target/rank更新隔离、完整identity和分块VJP。
Writer总14,190,240、Meta626,688；真实GPU机制检查exit0，Meta在identity后A/B梯度均可达，source无梯度。
full/staged loss同0.1256087869，Meta0qB/rho0/decoderB余弦0.999993/0.999998/1.0；这是工程机制证据。

已完成short4 formal run：runs/outputs/layered_relation_short4_target_rank_6ae406ea_gpu01p235_20260907；clean pushed frozen6ae406ea，
worktree .codex/worktrees/layered-readout-6ae406ea，gpu01physical2/3/5、world3、tmux ember-layered-readout-control。
launch前两节点live检查，所用三卡均无计算进程；data1 quota489371696KiB/soft1073741824，已有两run各653MiB、analysis15MiB，
新增峰值<1GiB，共享free84TiB，全部大资产复用。精确命令/环境/采样/预算见target_rank_readout_control/launch.json及launch_train.sh。
96步训练完成、train.exit0，三个checkpoint保留；384条件/6144queries的task权重、K/视频、query episode/frame与policy RNG完全匹配坐标对照。
每task1536queries，K1/2/4各32；实际更新1263.39秒、含加载1384.12秒，峰值allocated34.323/reserved38.109GiB。exposure_cost.json保存完整曝光/成本。
16correct闭环完成exit0：7/40（Spatial2/Long5，Object/Goal0），breadth2/4；对source及坐标16correct均RGL3/4/1、churn5/40、Jaccard3/8。
全部七组物化/闭环均exit0：16correct/other7/10、48为6/8、96为11/11，96K4correct10；完整比较见target_rank_readout_control/panel_summary.json及*_vs_*。
96K1两arm同为Spatial3/Object2/Goal0/Long6，success集合完全一致（RGL11/0/0，Jaccard1）；对source均RGL4/7/0。
48→96 correct RGL5/6/1、other6/5/2，churn均7/40；仍有明显相邻变动，未满足正式稳定性。K4相对K1 RGL9/1/2，Jaccard.75。
同预算比坐标对照96K1从6/6到11/11、breadth2到3；保留基础学习正证据，Goal仍0，未证明未见task迁移或video必要性。
代表target的rank常量能量从>99.995%降至1.55%–6.55%，native常量却升至91.2%–99.3%；几何并非单向变好，不据此追加decoder修补。
下一步保持当前图和fresh联合训练，扩展固定train24，以K1为初始qualification setting，真实K1/2/4训练保持；预算和strict400节点登记后launch。
历史source strict400（47/400）policy/environment/RNG/source checkpoint已与当前合同核对；旧normalizer路径随worktree退役失效，
比较工具现只在clean recorded commit与原bytes匹配时从同一Git配置恢复，并报告provenance。5 targeted tests及真实400行比较通过；未改写原始formal artifacts。

## 已完成的初始化formal对照

- 已完成short4对照：runs/outputs/layered_relation_short4_coordinate_init_880bde5e_gpu01p235_20260907。
  Clean pushed frozen commit880bde5e，workspace .codex/worktrees/layered-coordinate-880bde5e；gpu01 physical2/3/5、world3，
  tmux ember-layered-coordinate-control已自然结束，train.exit=0；96步数值与真实Meta梯度正常，三个checkpoint完整保留。
  初始source flow loss与原run同为0.1250352208；384条件的task权重、K/视频集合、query episode/frame及policy RNG逐项匹配原run。
  更新总1257.37秒、单段总1376.95秒；原run含两次resume加载，不能把总时差全部解释为方法吞吐改善。完整曝光/成本见coordinate_init_control/exposure_cost.json。
- 唯一科学改动是native A/B坐标由std0.02初始化改为标准正态；原图、参数量、public A0、零readout、Meta、seed、
  optimizer/schedule、任务/视频/query采样、曝光、frame_chunk4和world3不变。175 CPU tests/17.04s通过。
- Fresh联合训练96updates、checkpoints16/48/96；每task1536queries，真实K1/2/4各32组。
  固定四训练任务global7/12/20/35、states0–9，held46–49无放回correct/other K1；96步补K4correct全4视频。
  这些均为训练侧学习/初始化诊断，不选择最终checkpoint，不使用validation/Test或负视频controls。
- 新对照的命令、环境、资源、预算与裁决在runs/analysis/layered_relation_writer_20260907/coordinate_init_control/launch.json及launch_train.sh。
  Launch前两节点live复核；data1独立quota使用488688112KiB/soft1073741824，原run653MiB、analysis6.9MiB，
  新增峰值预算<1GiB，共享free84TiB，复用全部大资产；初期合计<5GiB预算仍满足。baseline K4已完成并释放p0，后续新对照16步物化/评测可用该卡；正式launch前重新live核验。
- 16步correct已完成exit0：4/40，Spatial2/Long2/Object0/Goal0，breadth2/4；对source及原16步correct均RGL3/1/1、churn2/40。
  新结果与原run分目录保留在coordinate_init_control/。剩余六组物化已用resident batch完成exit0；eval frozen fa0b7b43。
  七组闭环、固定functional三点面板与96步冻结梯度诊断均已完成exit0，曾使用gpu01p0/2/3/5，现已释放。完整新增比较见历史§15。
- 批量物化入口在隔离worktree实现并集成：同一次source加载复用runtime，各请求严格重载完整Writer/Meta/probe，独立输出既有manifest。
  不改变单条件compiler、采样、模型或评测；原单次入口保留。这只优化准备成本，未宣称科学收益。
- 三个既有诊断target（task7/35的expert0q/0v/action_out）显示：96步native通道常量能量由原>99.995%降至70.7%–85.5%，
  rank常量能量仍>99.995%。原已冻结功能梯度中rank常量分量仅2.64%–21.40%。这保留readout共享/槽区分的候选解释，
  不凭几何选点或立即叠改；先完成本单变量的真实行为裁决。原件coordinate_contrast_s16/s96.json及original_gradient_rank_contrast.json。

## 原始初始化short4：训练与全部短面板已完成

- 原formal run：runs/outputs/layered_relation_short4_joint_8d934408_gpu01p235_20260907，frozen train8d934408、eval d5b8119e。
  完成96steps、384conditions/6144queries，16→48→96两次完整exact-resume通过，所有checkpoint保留。
  三段含加载时间339.39/545.03/755.39秒；实际更新总1288.78秒、平均13.42秒/step。
  各task覆盖全部16条训练视频，K4各32个不同集合；独立query episode-frame数1230/1248/1269/1310。
- 固定screen40结果：source4；16步correct/other4/6，48步6/5，96步4/4。所有点breadth2/4，Object/Goal均0。
  suite顺序Spatial/Object/Goal/Long：source2/0/0/2；16c2/0/0/2、16o4/0/0/2；48c3/0/0/3、48o3/0/0/2；96c2/0/0/2、96o3/0/0/1。
- 对source，16c RGL4/0/0、16o4/2/0；48c4/2/0、48o4/1/0；96c/o均3/1/1。
  相邻16→48：correct4/2/0、other5/0/1；48→96：correct4/0/2、other3/1/2。
  同点跨视频Jaccard16:2/3、48:5/6、96:1/3；96跨视频RGL2/2/2、churn4/40。
  原K1短学习没有形成稳定且广泛的行为改善，停止原配置无依据续训。
- 原96步K4correct已exit0，6/40（Spatial3/Long3，其它0），对source RGL3/3/1；对同点K1 RGL2/4/2、churn6/40、Jaccard1/4。
  只用全部held46–49，固定同40初始化，不做K4other（held池只有4）。
  该项在读96闭环分数前登记，不用于最终模型选择。
- 原fixed functional panel均已完成，无参数梯度/更新，action42–45各8query/task，固定noise/time，不能选点。
  correct benefit（source loss减candidate）16步[8.90e-5,2.86e-5,5.11e-5,-1.87e-5]，
  48步[7.73e-5,7.05e-5,8.45e-5,-1.03e-5]，96步[1.59e-4,1.26e-4,3.48e-4,-3.18e-5]；功能小变化不能代替闭环。
- 事后冻结输出梯度诊断（仅授权训练task7/35、无参数更新）：expert0q/0v/action_out的96步B常量能量>99.995%，
  真实policy梯度常量分量约0.003%–2.23%，code RMS1.10–1.16而native坐标约0.02；rank间code差约0.0116。
  这支持坐标初始化条件假设，不唯一归因共享学习缺口，也不证明新初始化有效；active design§8.4定义单变量fresh检验。
- 完整原件均在runs/analysis/layered_relation_writer_20260907：各s16/s48/s96_*_screen40 raw rows、*_vs_source、*_vs_s16/s48、
  cross_video比较、short4_exposure_cost.json、functional_s*.json、decoder_gradient_s96.json/.safetensors及各registration/launch/log/exit。
  同state/env/policy RNG、source与normalizer合同经比较CLI确认；未运行validation/Test或最终因果controls。

## 当前实现与验证（2026-09-07）

- 授权记录已提交推送70b194ec；纯Writer实现4fa19c3a已集成main。训练/读取/数据于8d934408集成，物化评测于d5b8119e集成；各自formal运行使用对应clean pushed frozen authority。
- 唯一实现owner：writer/relation.py负责局部帧对和同步邻居更新；layered.py负责语言/H-read/集合compiler；coordinate.py负责完整坐标A/B；
  native.py负责原生Meta读取与R-leaf/observer VJP；learning_data.py负责固定split与跨episode采样；runtime.py负责加载和有界冻结prefix缓存；
  training.py及薄CLI负责全局任务权重、调度、checkpoint/resume；materialization/evaluation负责逐episode条件物化及已有队列接线。没有恢复旧Writer/fallback。
- 这是退役后从空缺重建部署图及必要训练面，训练加物化/评测源代码增长约2k行；各模块职责独立，主文件均小于400行，复用现有LoRA、functional、
  source、checkpoint与队列。现有checkpoint函数的局部增长仅添加sampler状态，trainer.run作为单一生命周期编排保留，避免机械拆分。
- 首版Writer14,112,544参数，读取Meta626,688参数；两者直接fresh联合训练，source0 trainable。配置入口configs/pi05_layered_writer_v1.json。
- 训练侧短面板global7/12/20/35覆盖Spatial/Object/Goal/Long；Long35历史专家40/50（完整原件已核）。
  新采样定义video0–15、action16–41、diagnostic action42–45、held video46–49，互斥；每task真实K1/2/4轮换、独立无放回抽K组，
  query按episode再frame分层抽样。4task×3visits真实数据读取已验证，源normalizer冻结，梯度normalizer明确为1（未继承旧carrier task reweight）。
- 新跑全CPU suite153 passed/20.06s；后续checkpoint/sampler与相关检查26 passed/14.85s。纯CPU通过不代表真实功能或行为通过。
- gpu01p3（GPU-e59426ed-ed41-cb75-2190-f50841cff288）实际两帧native smoke：[2,18,50,1024]、finite、requires_grad；真实最长train视频为
  global38/demo36，raw517、stride5含尾帧105。此smoke只验证读取接口，未给出Meta functional梯度结论。
- 完整GPU首轮在第一次功能更新前暴露BF16消息与FP32 scatter buffer dtype不匹配，已以消息dtype分配修复；重跑已通过真实功能VJP与最长K1/K4。
  临时记录：.codex/tmp/layered/joint_mechanism_retry.log、joint_mechanism.jsonl；任务包含identity启动后的真实Meta梯度、full/staged VJP限定比较、
  最长真实K1/K4（38的36/41/28/35；query另取0–15）及真实16条action queries。只作工程机制/profile，不能选择checkpoint或声称科学收益。
- 实测strg01：data1约465.4GiB、data0约54.8GiB，分别soft1T/hard约1.01T；du workspace约434GiB，data1其它约32GiB，data0约55GiB。
  共享空间data1约84TiB、data0约1.9TiB；初期新增预算<5GiB（完整模型/optimizer checkpoints和小证据），复用全部大资产。
  prefix只做每rank2GiB有界CPU缓存，临时R每step失效。launch前已同时核验两节点，未干扰其它用户作业。
- 真实functional检查：identity第0步Meta A/B均0，第1步B非零，第2步A/B均非零；source无梯度。
  full/staged loss同为0.1337032914；抽查Meta0q-B/rho0/decoderB的cosine为0.9999877/0.9999966/1.0，相对误差0.00502/0.00260/0。
  这是BF16链式语义验证，不是训练能力结论。采样只使用授权train任务与跨episode真实action queries。
- 最长profile：K1[105frames] prefix3.60s、joint8.27s、peak allocated34.23GiB；K4[105,102,102,97] prefix13.37s、joint25.78s、
  peak allocated34.75GiB/reserved36.10GiB。K4分项observer3.83、Writer1.80、policyVJP2.26、WriterVJP8.18、observerVJP9.70秒。
  完整原件保留runs/analysis/layered_relation_writer_20260907/mechanism/，明示exploratory，不选模型。
- 四任务formal短学习预登记96updates，每task1536真实queries、K1/2/4各32条件，global4task等权；16/48/96固定checkpoint。
  各节点global7/12/20/35×states0–9×correct/other两组K1 held视频，seed20260907，固定每task无放回分组；仅判断基础行为/新视频泛化，
  不能选择最终checkpoint。未见基础行为则诊断最早接口，不默认长训；后续完整train24的strict400/邻接口径在读分数前另登记。
- Subagent在隔离worktree完成纯图后，已交付物化与原有评测队列接线，验证后已集成；主agent负责GPU机制、训练及科学决策。

## 已封存的最近科学结果

- 旧complete P/Q short4 m64 fit/held为64/62（各150）；mixed meta73四点validation screen80为15/19/19/19。
- 相同18target对照四点screen为17/17/20/16；terminal128训练侧held视频breadth为55/180、13/18tasks，对meta73的42/180、9/18。
  Object仅5/40；Goal/Long改善仍不足以证明共享与迁移问题解决。
- 相同预算task75/77 whole-Writer clones为14/20，对shared18的3/20；shared相邻四点为2/3/4/3。
  说明共享训练存在代价，但不能单凭该差距确定容量、梯度冲突或优化根因。
- 同图fully-random target18四点screen为16/16/17/19；固定训练两task4/20，未消除共同学习缺口。
- width256原run已自然结束：128updates、15,660,800参数、train.exit=0；训练732.48秒，functional Panel-B498.24秒，总计1283.05秒。
  32/64/96/128 checkpoint均保留；暂停后未启动物化或闭环评测，**没有width256闭环分数**。

完整原件、样本/预算口径与边界在 [历史§6](docs/research_history.md#recent-learning)；跨轮解释在 [findings.md](findings.md)。
上述都是旧图结果，新候选不继承其性能结论。

## 前次仓库整理

- 新设计记录涵盖科学动机、完整数学与因果推导、张量shape、多视频、单probe选择、X/Y与坐标decoder、GPU staged VJP/cache与现有代码迁移。
- 重写concept、长期要求、分层history、findings、README与当前账本；原6份旧设计、11份专家原文和181节完整旧账本保留在Git
  `fcdb6e43706c5fcedf10eaa5d2d459602b263016`，历史§9可按问题定位原件。
- 退役旧P/Q、bank conditioning、Natural Program、joint primal、G1/G3与Stage 0专用训练面及专用配置/测试；
  `src/scripts/tests/configs`文件共476→123（src195→69、scripts42→8、tests42→24、configs197→22）。
- 整理通用functional重放、panel读取、task/K调度与成本分配；source/data/expert/evaluation/normalization/LoRA基础保留。
  `configs/pi05_writer_data_v1.json`集中可复用来源，历史角色不是新实验授权。新图仍无可运行训练入口。
- 本次由设计记录、代码整理、存储审计三个subagents并行完成各自范围；通用委派规则随后按owner纠正统一维护在用户级AGENTS，删除项目内重复要求。
- 删除222组可重建派生缓存，共82,122个payload文件、245,347,917,824 allocated bytes（228.50 GiB）；
  保留所有cache/entry JSON、生成配方、run contracts、metrics、raw rows、正常化参数和唯一checkpoints。
  缺上游Writer的两个小smoke缓存、两个各约44GiB正式run root及独特轨迹证据保留。
- strg01清理后quota复查：data1 488,035,348 KiB（465.43 GiB），soft1,073,741,824 KiB、hard1,084,227,584 KiB；
  清理前727,683,088 KiB。data0 57,471,972 KiB，使用独立quota。此为当时快照，下次大增长前须现场复查。
- 精确删除范围、保留例外、重建依赖、width256完成核验与工作树清理记录位于
  `runs/analysis/ember_handoff_cleanup_20260906/storage_cleanup.json`。

## 前次整理的验证与交付

- 在集成后的主工作区新跑 `PYTHONPATH=src .venv/bin/python -m pytest -q`：139 passed，20.75秒。
- 6个保留Python CLI的`--help`全部exit0；当前Python源码语法、22个JSON配置与两份shell脚本语法检查通过。
- 当前Markdown本地文件链接无缺失，17份退出活动树的设计/评审原件仍可由冻结Git读取；`git diff --check`通过。
- 10个已完成工作树已移除；两个写入subagent的交付范围与main集成内容一致，task分支已删除。临时启动/诊断副本与Python/pytest缓存清理完成。
- 代码整理与设计记录已合入main；本文及其余文档随交接提交推送。最终提交与remote一致性以Git实际状态为准。
- 验证覆盖当前保留的工程基础；新架构GPU forward、Meta梯度、profile和闭环尚未运行，不能据此宣称新方法有效。

## 本次设计修订与交接

- 完整设计更名为layered_relation_video_writer_design.md，保留一个canonical方法文档；重写过程推导、shape、GPU布局/成本与验证定义。
  原past-only文本在Git 12d9689c及此前提交中保留，历史§7记录修正原因与边界。
- 同步concept、长期要求、findings、README、task_plan、分层历史和HANDOFF；内容差与对应位置变化均有明确消费者，
  允许双向教学帧读取，禁止重新使用旧的未来帧不变性测试。
- 已审阅9份文档的相关差异，核对55个本地文件引用均存在，原初稿Git引用可恢复；旧单向定义只保留在历史或明确的退役说明中。
  本次git diff --check通过；只更新文档，未重跑前次139项工程测试，未进行GPU/新模型验证。
  main交付随本次文档提交推送，Git实际状态为准。

## 下一步

落实并验证已登记的末读出共享单变量改动，fresh重跑同短预算和闭环口径；原初始化对照全部完成且不再恢复。
通过基础训练行为后再登记完整train24与strict400；不能把几何或loss代替闭环。
按task_plan持续执行，不因例行检查、阶段汇报或一次实验结束停止。

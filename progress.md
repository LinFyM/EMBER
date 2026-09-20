# EMBER progress

## Writer冻结与MT-BC续行（2026-09-21）

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

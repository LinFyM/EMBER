# EMBER progress

更新时间：2026-09-08 CST。专家讨论与 Owner 最终选择已收口，当前完成新 session 的科研接续准备。

## 当前授权与方法状态

- Owner 最新要求：“后续交给一个新session去推进……把新session启动前的所有工作都做好，然后给我一个启动prompt”。
  本 session 执行方法登记、原文保存、状态对齐和 Git 交付；**不在本 session 实现科研代码或启动训练/评测**。
  新 session 接到 Owner 的启动 prompt 后，按已定设计连续完成实现、验证和科学推进，无需再次请求整体架构审批。
  Owner补充四点：先全面阅读仓库；完整阅读并忠实实现新架构、不打折扣；正式实验并据结果迭代到性能达标；
  **新session创建覆盖整个过程的goal**，遵守合同并实质解决难点。未指定token预算，不擅自添加。
- **唯一 active design：**[过去定向完整 Horizon Writer](docs/horizon_relation_video_writer_design.md)。
  **已定、尚未实现：**末层 post-norm PreActionOut；过去四帧 soft correspondence；完整 H-query；两端 Z 核实；
  按历史 u 从早到晚短 GRU；四组局部—**过去单向长程**交替，前三组逐 H 非线性回写；集合 compiler 和 native D 完整 A/B。
  长程四组都只读 past+self；H-query 可在完整50个h内双向交互。不要从专家原文恢复双向长程或18层输入。
- 首版训练采用 fresh Writer/Meta、固定 source，FM 辅助真实 Writer RL；同版本采集/求梯度、一次联合更新。
  具体 Sigma、LOO baseline、Q/M、10-step replay、trust/拒绝/RNG/checkpoint合同全部见 active design §7。
  FM+RL 是可试的训练选择，不把固定系数说成 Owner 的永久规定。
- **当前源码仍是旧 layered Writer。**最新 handoff 准备前代码基线 e868de525fda0a20c597dee2c7bffe5717f5e2fd；
  旧 `configs/pi05_layered_writer_v1.json`、训练/物化 CLI 不是新方案可直接运行的配置。新图无 checkpoint、无学习或闭环分数。
- 旧 train24 run 永久止于384：correct69→67、other72→64；熟悉/held训练视频21/18（各120）。未通过资格，未选selected checkpoint。
  旧 P/Q width256 尚无闭环，不自动补评；旧 native-head draft 不整支集成、不单独开旧上游对照，不恢复旧672 schedule。
- 当前科学目标仍为validation8 strict paired single-checkpoint correct>145/400，并满足相邻稳定/低churn/high breadth/
  四suite及GoalLong/同task视频鲁棒性；冻结后完成视频因果controls，方法冻结后32/8 fresh与Test。准备工作不代表科学完成。
- 本session不创建新Codex task或启动科研goal，不联系外部专家。新session按Owner本次要求创建并推进科研goal；
  不自行设置token预算或自动化。历史goal/暂停与旧文件中的“下一步”不覆盖这次新session交接安排。

## 交接入口与实测范围

1. [HANDOFF.md](HANDOFF.md)是临时接续入口；长期目标、方法、计划、证据已在正式文件，消费后可删除该入口。
2. [task_plan.md](task_plan.md)给出实现→真实机制/成本→结果前登记→共享学习/闭环的执行顺序。
3. [原文与 Owner 裁决](docs/review_materials/20260908/README.md)保存四份完整回复；其中最后专家稿的双向长程被 Owner 的单向决定覆盖。
   上一份完整 FM/RL 原文用于补齐最新稿引用的训练细节，不恢复其旧架构。
4. [research_history.md](docs/research_history.md)索引历史事实；[20260907补证据包](docs/review_materials/20260907/README.md)仍保留95面板/24,100行现存结果。
5. 本次仅核对 Git/worktree、代码接口及关键资产存在性；未跑科学测试、未启动GPU、未刷新两节点GPU状态或strg01配额。
   旧 GPU/quota/吞吐快照不作为新run准入，真实现场由接班者在相应操作前检查。

## 接班前代码与工作树快照

准备时 main 干净、与本地origin/main一致；交接文档随后在main提交并推送，最终精确提交以Git为准。
共有主workspace、7个历史detached工作树与一个dirty native-factor-readout草稿工作树；未删除或重置它们。
草稿分支 `codex/native-factor-readout` 基于ec02710b，相对准备前main无独有提交、落后1提交；未提交变化为
删除coordinate.py、修改layered.py/其测试、新增native_factor.py，仍使用旧上游。其所有权与内容须保留，不能当新实现直接合并。

source模型、tokenizer、manifest对应40目标/71source HDF5、LIBERO BDDL/init及仿真assets在本地存在；精确入口见README与design §9。
这些只做存在性核对，没有复制大资产、读取held action产生梯度或做全树完整性扫描。

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

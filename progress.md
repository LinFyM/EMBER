# EMBER progress

更新时间：2026-09-06。

## 当前快照

- 2026-09-05 23:24 CST owner明确结束事前对齐并要求设置goal持续自主推进；goal已建立且active。本条新授权取代此前“仍在对齐”的
  暂停状态。同图clone/shared、action-query覆盖和非对称A三轮对照均已完成；当前只增加正确fit视频池2→4的matched训练运行于
  clean pushed detached `13339ac1`，数据与下一步详见最新节点和active design。
- 专家附件已原文保存为`docs/expert_review_20260905_full_history_joint_process_policy_writer.md`。
  `docs/joint_process_policy_writer_design.md`现登记为active design，首项同部署图、whole-Writer、无task query的clone/shared对照；
  P/Q主干及非对称读出依证据决定，旧v4结果继续sealed，不续跑旧checkpoint。
- 后续接管原则保留：两个节点合计最多6卡、遵守科学/信息墙；性能不佳须认真分析竞争解释，不能随意命名根因并立即修补。
  常规分析与实验自主推进，维护本文件方便owner查看；需要owner决定时提出具体问题，继续不受影响工作，真正受阻才标记blocked。

## 最新节点（2026-09-06 04:45 CST）

- 四视频formal已从clean pushed detached `13339ac1`在gpu01物理3/4/5启动；实际contract确认A2、四fit池、K1与world3/NUMA1，
  当前21步、约11--12秒/update。run为`runs/outputs/pi05_ecp_prw_shared4_four_videos_s64_13339ac1_gpu01p345_20260906/`；
  launcher为`.codex/tmp/prw_four_video_launch/train.sh`。本次/data1 quota721109192KiB，新增<25GiB预算不变。
- 32物化（GPU0）与held（0/2）/fit（6）strict150的launcher已准备，checkpoint出现且live检查后启动，仍合计不超过六卡。
  功能汇总已准备原两fit匹配平均与新四fit平均两个独立口径；闭环汇总以原非对称四结果为直接参照，当前6/10为已存在参照。
- 上一轮02a/387 frozen worktrees确认clean、没有进程引用后移除，只保留当前13339ac1；全部formal数据、checkpoint、raw rows和
  exact launch evidence保留。README中旧“没有active goal、等待交接”描述已纠正为读取当前计划/状态，历史不恢复执行。

## 四视频 formal启动合同（2026-09-06 04:37 CST）

- 四update真实profile exit0，16个task executions覆盖全部四fit视频并与预注册query/video/noise逐行匹配；2,970,368 Writer参数，
  source/observer/task-local参数均0，完整38-target rank16。各step11.64--12.35秒，最长87帧/新增66帧均通过；峰值allocated35.77GiB、
  reserved36.61GiB。单份capture tensor约14.99GiB，临时mmap已自动移除。profile只作资源和真实梯度证据。
- 正式训练从本合同所在的下一clean pushed commit的detached tree
  `/data1/user/ymdai/projects/EMBER-worktrees/prw-four-video-20260906`启动，config为
  `configs/pi05_ecp_prw_samegraph_shared4_four_videos_v1.json`；`--phase shared --mode formal --representation full --initialization component`
  `--stop-after-step 64`，fresh optimizer/scheduler，warmup5+effective59、checkpoint8/32/64，query/video/Panel-B与四条件strict150见active design。
- 计划gpu01物理3/4/5、NUMA1、world3、NCCL_P2P_DISABLE=1；启动前再次live核对两节点与/data1 quota。exact resume仅限原commit/config/
  world/物理GPU顺序/root；不恢复两视频run的optimizer。run root计划为
  `runs/outputs/pi05_ecp_prw_shared4_four_videos_s64_{commit}_gpu01p345_20260906/`，analysis为
  `runs/analysis/pi05_ecp_prw_samegraph_four_video_20260906/`；新增峰值<25GiB，复用全部大资产，单份mmap退出清除。
- 32/64 first-fit/held各strict150继续在checkpoint生成后与训练并行，使用同eager batch入口、固定状态/RNG与原视频；原非对称四结果
  为直接参照。functional须另报原两fit视频匹配平均，不与新四fit平均混淆。精确launcher/log/exit、profile与采样审计保留至formal root。

## 前一执行节点（2026-09-06 04:32 CST）

- 非对称四条件与两个同batch参照全部exit0，900新行已完整汇总。对称64 fit/held41/44，非对称32为43/41、64为44/45；
  64非对称逐Spatial/Goal/Long为fit40/4/0、held38/6/1。相邻fit churn13、held18；64跨视频R/G/L=39/6/5、Jaccard.78，
  但Goal fit相邻只保留1/5、Long2→0。完整表和所有配对保留在asymmetric analysis；没有最终资格或根因结论。
- 下一单变量已登记：以闭环略高的非对称A为工作起点，fit视频池2→4、K1/4tasks/64updates/8rows及其它合同不变。
  原held/first-fit不变；全部256个query/policy-noise与normalizer已逐occurrence核对匹配，四条fit视频各16 exposures。
  三个新配置为`pi05_ecp_prw_samegraph_shared4_four_videos*_v1.json`，正式launch等待当前真实profile。
- runtime只保留A2/B1，删除临时对称A构造开关和接受分支，原A2初始化及RNG不变；三个现有源文件净减8行，无新source。
  composer835→832、training791→788，既有大文件未增长；7项有针对性的shape/gradient/static wall/signed-query/set/config检查通过。
- 四update真实profile已在gpu01物理3/4/5、NUMA1启动，恰覆盖每个task的四条视频；输出`.codex/tmp/prw_four_video_profile*`。
  两节点已live核对，原评测全部释放。/data1 quota721100456KiB，独立额度1073741824KiB，项目实际656GiB、shared84TiB空余；
  本阶段新增峰值预算<25GiB（单份约16GiB共享cache及run/eval），profile cache释放后才formal训练，复用全部大资产。

## 前一执行节点（2026-09-06 03:59 CST）

- 非对称训练与Panel-B均exit0：训练781.62秒、功能评估239.25秒；共有normalizers和全部256条task executions的query/video/noise/权重
  与fresh-query审计一致，8.03GB临时mmap已自动清除，三checkpoint与launch/profile evidence保留。
- 功能结果混合：64-step的task1/72/83/93 held benefit依次为`-.000437/+.001764/+.000817/+.000652`；相比对称fresh-query，
  Goal约翻倍，但meta task1转负，Long没有改善。task6/79诊断仍负；32时也没有普遍正收益，因此不能据此确认A侧干预有效。
- 64两bank已sealed并exit0，首次runtime116.18秒、驻留第二次0.093秒，物化分别14.09/13.88秒。64 held在gpu01物理3启动；
  对称64同执行参照分别在4/5运行，32 held在0/2、fit在6，总计六卡。64 fit的0/2 launcher已准备，待32 held自然完成后live核对启动。
  32 held的首个完整Long50为0成功，其余未完成；最终判断等四条件及两个同执行参照收齐。

## 前一执行节点（2026-09-06 03:53 CST）

- 非对称64-step训练更新完成，预注册Panel-B仍在运行；32 fit/held闭环已进入真实执行，三个评测GPU均100%利用率。
- 为分开小幅科学增量与BF16批处理差异，已登记补跑fresh-query symmetric64的first-fit/held各strict150，直接复用原sealed banks、
  相同状态/noise与387f6d0b入口。两个launcher已准备，待训练释放设备并live核对后与64物化并行；不重新训练、选视频或改变评测口径。
  非对称64横向归因将以这两个同执行参照为主，32/64相邻仍为同batch入口，不把历史serial差异直接归于A侧。

## 前一执行节点（2026-09-06 03:49 CST）

- 32-step两条件物化均exit0，首次runtime114.60秒、驻留第二次0.147秒，三task物化14.17/12.80秒。完整rank16 banks已sealed。
  held strict150已在gpu01物理0/2、fit strict150已在6启动；训练继续3/4/5，总共六卡。两组均来自387f6d0b的eager batch运行面，
  两节点实时检查及精确launcher保存在`.codex/tmp/prw_asymmetric_launch/`。64-step两条件在checkpoint/训练设备释放后继续同一矩阵。
- 已完成且不再被进程使用的078 query-coverage frozen worktree经clean/引用检查后移除；只保留当前02a训练和387评测两个frozen trees。
  原训练checkpoint、raw rows、functional/配对汇总、readout与完整性能probe evidence全部保留，可按pushed commit重建历史运行面。

## 前一执行节点（2026-09-06 03:44 CST）

- 32 checkpoint已完整生成，GPU0正在驻留物化held与first-fit两条件；训练继续3/4/5。物化与后续eager batch eval authority为
  clean pushed detached `387f6d0b`，tree为`/data1/user/ymdai/projects/EMBER-worktrees/prw-asymmetric-eval-20260906`；只新增同task静态
  LoRA batch入口，Writer模型/训练配置与`02a85314`一致。两节点live核对通过，GPU0仅有1260MiB无PID静态占用、峰值余量充足。
- 32 held计划使用0/2两GPU，32 fit使用6一GPU，各3 persistent workers与8env/worker；精确launcher已准备在
  `.codex/tmp/prw_asymmetric_launch/`，物化完成并再次live检查后启动，届时与训练合计六卡。
- 单denoise-step dynamic compile三case全部exit0：Goal8为10.54 obs/s、首次79.8秒；切到Spatial7为10.27 obs/s、首次0.685秒；
  回到Goal1为5.96 obs/s、首次35.4秒。各正确task adapter与相同观察/noise下输出finite，RMSE约.0016--.0020。
  由于当前完整batch8额外收益约10%且首次/特殊batch有编译成本，本轮首批科学比较采用eager batching。完整batch与两种compiler探测
  的script/log/result/exit均保留到query-coverage analysis的`inference_profile/`；没有新增编译backend配置或全局环境修改。

## 前一执行节点（2026-09-06 03:39 CST）

- 非对称A正式训练已从clean pushed detached `02a85314`在gpu01物理3/4/5启动；run为
  `runs/outputs/pi05_ecp_prw_shared4_asymmetric_s64_02a85314_gpu01p345_20260906/`，launcher为`.codex/tmp/prw_asymmetric_launch/train.sh`。
  实际contract已确认A context branches2、common动态B、原采样与world3，8-step checkpoint已生成，训练约12秒/update。
  本次/data1 quota720889376KiB，预估<15GiB峰值不变；32出现后继续与训练并行物化和闭环。
- static adapter已补齐evaluator现有batch prediction接口，只允许同task完整LoRA的env集合，逐env noise/order保持；混task、空或
  noise数量不符会拒绝。既有7项static-bank/信息墙检查通过。真实Goal/Spatial8观察，在同TF32、同adapter/输入/noise下，
  serial为3.28/3.27 observations/s，batch8为9.59/9.55（约2.9倍），完整50步outputs finite；RMSE分别.00481/.00534，
  对应serial output RMS.556/.492。这个量级与正常BF16 batch/reduction差异相符，不做逐元素一致要求。
- 新batch接口复用唯一evaluator协议，无新runner；两现有文件净增46行，架构检查仅既有review、无hard。本轮后续评测采用batch入口，
  在结果中另行记录执行commit；训练、checkpoint、noise/状态及scientific preprocessing不变，不把执行效率修复称为科学方法收益。
- 只编译单个denoise step的动态batch探测正在物理6继续，script/log/JSON位于`.codex/tmp/profile_static_batch*`；Goal8→Spatial7→Goal1
  都使用各自正确task adapter和观察，检验变batch与adapter切换。无formal backend改变；本次科学评测先使用已验证的eager batching。

## 非对称A formal启动合同（2026-09-06 03:32 CST）

- 三rank真实profile已exit0，2,970,368 Writer参数、source/observer/task-local均0，原38-target rank16与信息墙通过；实际8个task
  executions的query/video/noise逐行与原fresh-query审计一致。峰值allocated35.77/reserved36.58GiB，无OOM/non-finite。
  同预算训练与Panel-B预计约17分钟，加首次资产准备；profile只作工程证据，不作方法选择。
- 正式训练使用包含本合同与实现的下一clean pushed main，detached worktree计划为
  `/data1/user/ymdai/projects/EMBER-worktrees/prw-asymmetric-20260906`；config为`configs/pi05_ecp_prw_samegraph_shared4_asymmetric_v1.json`，
  `--phase shared --mode formal --representation full --initialization component --stop-after-step 64`，fresh optimizer/scheduler、
  warmup5+effective59，8/32/64 checkpoints、4tasks等权，完整输入/采样/Panel-B及四条件strict150见active design。
- 计划gpu01物理3/4/5、NUMA1、world3、NCCL_P2P_DISABLE=1、单份共享mmap；launch前再次live核对双方节点与quota。
  exact resume仅限同commit/config/world/物理卡顺序/root，旧head-shape checkpoint不兼容，不加载旧optimizer。
  run root为`runs/outputs/pi05_ecp_prw_shared4_asymmetric_s64_{commit}_gpu01p345_20260906/`；精确命令/log/exit保留到run的launch。
  本阶段新增峰值<15GiB、结果另存`runs/analysis/pi05_ecp_prw_samegraph_asymmetric_20260906/`，大资产全部复用。
- 工程优化另发现：目前static-task adapter没有实现evaluator既有的batched prediction接口，因此实际闭环对每个env单独forward。
  已有batch8吞吐probe不能直接代表该运行路径；下一步补齐同task静态LoRA的批处理并用真实推理验证，独立记录执行改变。
  full sample_actions compile在变batch时超过15分钟总时限（batch8结果有效、batch4未完成），不能据此直接部署整图编译。

## 前一执行节点（2026-09-06 03:28 CST）

- 真实correct-video读出诊断18/18完成，零action/reward/梯度/修改视频。64 held的A/B signed相对branch RMS中位数：
  Spatial `.0319/.0087`、Goal `.0097/.0071`、Long `.0367/.0147`，raw BA/cap中位`.0298/.00634/.0472`，仅7/4/8个target触cap。
  这与近零双侧因子的局部解释相符，但不是行为根因；完整值与汇总保留在`actual_readout_probe/`，汇总明确用常规midpoint median。
- 下一单变量对照已在active design登记：A侧使用独立正/负context query，B侧仍由dynamic差关闭静态mobile，其他主干、4tasks、
  fresh query、64updates/8rows、视频/噪声/normalizer、rank与loss保持匹配。仅增加16,384参数，新增head采样隔离CPU RNG，
  旧A base/dynamic初始值保留；两个eval配置继续32/64 fit/held strict150。CPU配置、匹配初始化和7项既有定向测试通过。
- 三rank真实profile已在gpu01物理3/4/5启动（均NUMA1、两节点live检查通过）；log为`.codex/tmp/prw_asymmetric_profile.log`，
  profile只验证真实功能梯度、吞吐和峰值。独立compile仍在6，总计四卡。最新/data1 quota720880912KiB，约336GiB余量，
  新训练单份mmap+run/四eval仍保守<15GiB，复用原source/data/normalization，shared filesystem84TiB可用。
- 架构检查：三个现有源文件与一个既有测试改动，无新source；composer由824增至835行，为本次A head增加与统一signed-query形状的
  11行净增。采用skill允许的窄范围cohesive exception：改变仍归同一个读出owner，当前抽离整个bank/readout反而扩大无关重构；
  不增长原高复杂度block forward，不创建第二训练器，比较结束由主agent移除未选定head分支。training保持800行以内。
- compile batch8已得到有效结果：eager9.60→compiled13.67 observations/s（约+42.4%），首次编译359.7秒，output RMSE约.00398、
  eager RMS.645。batch4仍编译中；没有把探测结果混入已完成科学分数，尚未改正式evaluator backend。

## 前一执行节点（2026-09-06 03:19 CST）

- query覆盖四个strict150全部exit0，完整新600行及6个复用参照汇总为10/10。最终fit32/64=`41/41`，held32/64=`39/45`；
  fit逐Spatial/Goal/Long为`36/5/0→34/7/0`，held为`38/1/0→37/5/3`。fit相邻37 retained/4 gained/4 lost，held为38/7/1，
  两者churn均8/150。但64时fit→held的Goal只保留3/7、Long仅held出现3次；不能把45单点解读为稳定跨视频能力。
  完整结果见新analysis的`comparison.md`和`closed_loop_comparison.json`，历史第167节登记。没有扩大task或继续64步训练。
- m64 fit原`_r2`已自然完成并释放GPU0，确切launcher与exit已保留；其曾取消的重调度计划记录已补充最终完成状态。所有科学评测均已结束。
- 下一项为真实correct-video读出工作区间的只读诊断：相同三tasks、first-fit/held视频，在component初始化与32/64 checkpoint下，
  记录当前context/innovation、两signed branch均值与差、raw BA相对原cap及cap因子。只读取实际未修改的视频/native输入，不改变动态信号、
  不做negative controls、不读actions/rewards、不产生梯度或选择checkpoint。目的只在判断专家局部双零因子解释是否适用于实际工作区间，
  不以norm/梯度替代闭环。临时script为`.codex/tmp/probe_actual_readout.py`，结果保留在新analysis的`actual_readout_probe/`。
  两节点已live核对，probe在空闲gpu01物理4/NUMA1启动，复用现有模型与内存capture，新增结果<1MiB。
- 独立compile probe已通过命令级`CPATH`使用现有3.12 headers，并对齐正式TF32；eager batch8实测9.60 observations/s，
  编译仍进行中，尚无compiled吞吐结论。该probe在物理6；当前总共只用两卡，没有改变正式evaluator设置。

## 前一执行节点（2026-09-06 03:10 CST）

- 三个新strict150已完整exit0：m32 fit/held=`41/39`，m64 held=`45`。各Spatial/Goal/Long分别`36/5/0`、`38/1/0`、`37/5/3`。
  held32→64为38 retained/7 gained/1 lost，churn8/150、Jaccard.826；旧固定查询held40→41的相邻churn为13/150。
  新查询有更好的相邻行为积累迹象，但本轮还差m64 fit，尚不作路线或最终checkpoint选择。
- m64 fit仍是GPU0的原`_r2`单卡执行，已完成7/11 shards，3 claimed、1 pending。其间曾计划使用新释放的4/5/6与0四卡重启；
  在实际重启前4/5已被其它用户占用，因此该计划取消，`_r3`从未启动。一次SIGINT请求没有使原nohup run退出，随后保持原评测继续，
  没有force terminate任何process，也没有丢弃完成的rollout。相关计划取消原因记录在r2的`execution_interruption.json`，不是模型non-pass。
- 在空闲GPU6进行独立、有15分钟时限的torch.compile吞吐探测，正式评测代码与配置未变。首次探测引用已移除旧worktree中的normalization，
  在加载前拒绝；已改为当前078a2e68的同task正式资产。随后编译遇到缺少`/usr/include/python3.12/Python.h`，尚无有效加速结论。
  当前venv为系统CPython3.12.3；现有`/data0/soft/anaconda3/include/python3.12/Python.h`存在，待核对是否可作临时编译依赖。
  正式evaluator启用TF32，而最初独立probe未对齐此设置；后续仅在对齐环境/TF32并获得有效实测后考虑保留任何推理修改。
  probe脚本、日志与临时cache均在`.codex/tmp/`；本次/data1 quota720761444 KiB、约336GiB余量，编译缓存保守预算<10GiB。

## 前一执行节点（2026-09-06 02:40 CST）

- 首个新closed-loop完成：m32 held为39/150，Spatial/Goal/Long=`38/1/0`；旧shared4同条件40、carrier38。相对carrier为
  33 retained/6 gained/5 lost，churn11/150、Jaccard.750；相对旧shared4为36/3/4、churn7、Jaccard.837。尚不能据单条件裁决干预。
- 64 fit原计划接0/2，但gpu02及gpu01物理2此时出现其它用户的新作业。一次外层检查与启动衔接未正确拦截，evaluator自身admission
  检查在任何worker启动前拒绝该次启动（exit1、0 rollout）；尝试停止时launcher已自然退出，没有终止其它用户进程。
  原prepared root及完整失败log保留，不作科学non-pass。后续launch只在单独读取并确认live检查通过后执行。
- 已重新live检查两节点，并用gpu01物理0单卡3workers fresh启动同一m64 fit条件，root为
  `runs/analysis/pi05_ecp_prw_samegraph_query_coverage_20260906/fresh_queries_m64_first_fit_video_strict150_r2/`，launcher为
  `.codex/tmp/prw_fresh_queries_launch/eval64_fit_r2.sh`。checkpoint/video/50states/RNG/执行合同完全相同；summary仅把该明确runtime重试映射
  回m64 fit条件，原失败root没有results不会计入。当前32 fit在3、64 held在4/5/6、64 fit在0，共5张物理GPU；物理2留给其它用户。

## 前一执行节点（2026-09-06 02:29 CST）

- 64-step fit/held两个条件已完成驻留物化，exit0。held strict150已在gpu01物理4/5/6启动，每卡3 persistent workers；
  32 held继续0/2，32 fit继续3，合计六卡。64 fit的两卡launcher已准备，待下一组设备释放后立即运行。
- 32 held的Long部分6个shards已完成，Goal/Spatial仍推进；32 fit尚在Long部分。三组均只沿用预注册correct-video条件、50固定状态
  与原环境/策略RNG，未形成可用于本轮裁决的完整四组aggregate。所有物化与训练launch记录已保留，完整训练/功能结果见上条。

## 前一执行节点（2026-09-06 02:23 CST）

- query覆盖shared4正式run已完成64/64与所有8/32/64 Panel-B评估，completion与launcher均exit0；实际training775.71秒、
  Panel-B240.08秒，64次更新无clip。完整256个task executions的pairs/video/noise与预先审计逐occurrence匹配，实际覆盖仍为
  16episodes/task、473/455/459/499 unique rows。launch、CPU审计、profile证据已自动保留到formal root，8.03GB临时mmap已删除。
- 配对功能结果并非普遍改善。32/64 held benefit分别为：task1 `+.000138/+.000266`，Spatial72 `+.001121/+.001496`，
  Goal83 `+.000250/+.000400`，Long93 `-.000022/+.000730`；64时四task全部fit/held视频为正，但Goal/Long的held收益低于旧固定查询。
  留出task6/79两点仍负；完整表、相对旧shared4差值和真实覆盖计数写入新analysis的`functional_comparison.json`与
  `actual_training_query_coverage.json`。这仍只是功能证据，不提前代替四条件闭环。
- 64-step fit/held驻留物化已在释放的gpu01物理4启动；32 held继续物理0/2、32 fit继续物理3。双方节点已live复核，4/5/6已无训练process。
  64 held准备使用三张释放卡4/5/6，提高150状态吞吐；64 fit在下一组设备释放后启动。矩阵与科学预算不变，任何时点总物理卡不超过六。

## 前一执行节点（2026-09-06 02:15 CST）

- 正式训练已到57/64，8/32 checkpoint已sealed，每步约12秒；两种32-step视频条件均已驻留物化完成、exit0。首次加载114.11秒，
  后续0.163秒，完整三task Writer分别13.75/12.83秒。fit视频为global2/20/38的3/5/2，held为49/49/48，均来自原注册条件。
- 32 held strict150已在gpu01物理0/2启动（每卡3 persistent workers），32 fit strict150在物理3启动；训练继续物理4/5/6，合计六卡。
  两eval的精确launcher为`.codex/tmp/prw_fresh_queries_launch/eval32_{held,fit}.sh`，所有work来自同一078a2e68 frozen authority。
  live检查确认0/2/3无其它compute process；0原1260MiB静态占用仍有足够峰值余量，gpu02已同时核对且未使用。尚无新closed-loop aggregate。
- 64-step两个条件在checkpoint和资源就绪后继续同一矩阵；不因某个先返回结果增减条件。新的功能与行为汇总脚本都只引用上一轮raw基线，
  保存在新analysis root，状态仍partial；真实功能结论等待固定Panel-B evaluation，闭环结论等待四个完整strict150。

## 前一执行节点（2026-09-06 02:02 CST）

- 正式shared4 query覆盖对照已从clean pushed detached `078a2e68`启动，gpu01物理4/5/6、NUMA1、三rank；精确run root为
  `runs/outputs/pi05_ecp_prw_shared4_fresh_queries_s64_078a2e68_gpu01p456_20260906/`。launcher位于`.codex/tmp/prw_fresh_queries_launch/train.sh`，
  输出/commit/log/exit集中在同目录`run.json`；完成后自动将launch、CPU采样审计与真实profile证据保留到formal root。
- 32-step fit/held驻留物化launcher已准备，checkpoint sealed后立即与剩余训练并行；所有新工作合计最多6卡。物化/评测仍从同一新frozen
  worktree运行。两个节点live检查已通过，选定4/5/6无其它process；最新/data1用量720598052 KiB，原预算不变。
- 旧d35e66bf batch worktree经clean与进程引用核对后已移除，其18个完整eval和checkpoint evidence仍保留，历史可按pushed commit重建。

## query覆盖对照启动合同（2026-09-06 02:00 CST）

- 三rank真实profile已完整exit0，四tasks每步等权各1/4，2次更新为12.629/12.588秒。rank实际任务为`[1] / [83,72] / [93]`，
  Long单task约11--12秒，与两短task合计接近，增加第四卡无法明显降低当前主要计算下界。最大allocated35.76/reserved36.58 GiB；
  Writer2,953,984 trainable、source/observer/task-local参数均0、Panel-B/held/wrong/shuffle/reverse backward均0。
  本次采样实际pairs与预先审计序列吻合，profile不参与科学比较；原始result、metrics和log保存在`.codex/tmp/prw_fresh_queries_profile*`。
- 正式authority为包含本合同的下一clean pushed main，detached worktree为`/data1/user/ymdai/projects/EMBER-worktrees/prw-query-coverage-20260906`。
  复用相同source step1000、stage0 macro10、frozen rank12 carrier、normalization、tokenizer与dataset；配置为
  `configs/pi05_ecp_prw_samegraph_shared4_fresh_queries_v1.json`，`--phase shared --mode formal --representation full --initialization component
  --stop-after-step 64`，fresh optimizer/scheduler、K1、8rows/microbatch2、warmup5+effective59、checkpoints8/32/64。
- 计划gpu01物理4/5/6、world-size3、均NUMA1，固定`NCCL_P2P_DISABLE=1`、OMP/MKL4和既有deferred NCCL，launch前再次live核对双方节点。
  使用单份共享mmap，profile实测8.03GB native evidence；训练结束自动删除临时cache。新run+缓存+物化+四次strict150峰值仍<15GiB，
  独立/data1预算见前条。exact resume只允许相同commit/config/world topology/physical顺序与run root，不覆盖旧run或借旧optimizer续训。
- run root为`runs/outputs/pi05_ecp_prw_shared4_fresh_queries_s64_{commit}_gpu01p456_20260906/`；完整命令/NUMA wrapper/log/exit、
  audit与profile在run的`launch/`保留。新的闭环分析根为`runs/analysis/pi05_ecp_prw_samegraph_query_coverage_20260906/`。
  四个新eval为32/64 × first-fit/held，各Spatial2/Goal20/Long38 strict150；只引用上一轮source32/carrier38与四shared4条件的raw rows，
  不重跑基线、不新增validation/test消费、不以此train-side面板选最终checkpoint。原视频/RNG配对、所有50 horizon与执行合同保持不变。

## 前一执行节点（2026-09-06 01:55 CST）

- 新配置`configs/pi05_ecp_prw_samegraph_shared4_fresh_queries_v1.json`只改变原Panel-A授权episode/frame抽样；两份32/64 fit/held
  eval配置沿用原task子集、固定状态与视频。既有shared trainer中增加一个采样helper，原batch装配器限制覆盖只用于Panel-A且episode不重复、
  frame有效；原panel loader继续拒绝Panel-A/B/video episode交叠。sampling RNG只取config seed/task/occurrence，policy RNG仍取原visit。
- 对真实dataset index直接核对4tasks全部64 occurrences：每task覆盖16 episodes与473/455/459/499 unique rows，原为8episodes与
  123/115/122/126；normalizer完全相同，video/noise序列未改变，任意occurrence可复现。错误Panel-B/video episode与Panel-B override均被拒绝。
  记录见`.codex/tmp/fresh_query_schedule_audit.json`及同目录`check_fresh_queries.py`；formal启动时保留到run证据。
- 4项既有config、optimizer、task权重与positive-only目标CPU检查通过，架构检查只有review、无hard。两现有源文件净增39行，无新源文件；
  787行runtime仅增加query override的信息墙，sampling归属shared training；单一训练器以显式实验变量承载两采样条件。
  训练日志现在记录真实query pairs，并把原16-row carrier loss明确标为reference panel loss；移除与本次8-row generated loss不配对的
  伪benefit字段。真正paired benefit仍仅来自未改变的Panel-B评估，不为新随机rows额外执行carrier forward。
- gpu01物理4/5/6（全NUMA1）已启动2-update、四task/full50/8rows的三rank profile；最长87 sampled frames，沿用单份mmap缓存与
  `NCCL_P2P_DISABLE=1`。profile只作输入/梯度/吞吐/峰值验证，不作科学分数。gpu02已同时live检查，未使用；gpu01物理1的他人任务保持原状。
- 本次`strg01`独立/data1 quota为720589628/1073741824 KiB（hard1084227584），约336.8 GiB软额度余量；shared filesystem84TiB可用。
  上轮5个训练run各103MiB、完整分析131MiB；本次profile/单份mmap/新run/checkpoints/四个strict150总新增保守<15GiB，复用全部大资产。

## 前一执行节点（2026-09-06 01:35 CST）

- 首轮18/18个预注册eval、1500 paired rows全部自然完成，所有worker与launcher exit0。最终shared4 fit32/64=`39/44`，held32/64=`40/41`，
  carrier=`38`，source=`32`（各150状态）；逐task、suite、breadth、paired sets和相邻重合全部写入
  `runs/analysis/pi05_ecp_prw_samegraph_local_b89ee997_20260906/comparison.md`与`closed_loop_comparison.json`，历史第166节登记。
  Goal clone64 fit最终为8/50；其余分数见下方快照及完整表。全部materialization/eval的精确script/log/exit已保存在formal roots。
- 实测四worker Goal50约666.8--672.6秒，三worker同task约693.6--698.7秒；视频不同，暂不严格归因速度差。
  四worker显存约40--41GiB、无OOM。额外一次性推理profile复用真实train-side观测，batch8/16/32为7.83/8.18/8.31 observations/s，
  所有50-step outputs finite；batch增大收益约6%。profile只测吞吐，不参与模型选择，完整脚本/日志/JSON保存在同root `inference_profile/`。
- 当前所有GPU工作已完成，未启动后继训练。已清理不再被进程使用且clean的训练6624127b、carrier54312cf1与旧evalb89ee997三个detached
  worktrees；仅d35e66bf batch frozen tree暂留作最近复现。checkpoint、source/data、raw rows与formal evidence均保留。
- 下一项已在active design登记：同4tasks、component-init、64 updates、8rows、原视频/16组policy noise/normalizer/optimizer，
  仅从原Panel-A全部16个授权action episodes重新均匀采样每次8个episode/frame。原方案每task512次row使用只有8episodes、115--126个
  unique states，且两fit视频关联不同固定row子集；这提供可检验的覆盖假说，尚不是已确认原因。不同时扩task、K、训练时长或换P/Q。
  下一步实现既有shared runtime中的最窄采样变化，检查Panel-A/Panel-B/video信息墙与可重现occurrence，然后真实profile、冻结提交、启动。

## 前一执行节点（2026-09-06 01:10 CST）

- 9/18个预注册strict eval已完成。source32/150，carrier38/150，逐Spatial2/Goal20/Long38分别`32/0/0`、`34/4/0`。
  全部clone的held条件已完成：m32=`41/6/1`，m64=`39/6/1`；这是三个独立诊断模型，不是一个48/46分的部署模型。
  Spatial相邻38 retained/1 gained/3 lost、Jaccard.9048；Goal为5/1/1、.7143；Long两个1/50成功状态不同、重合0。
- shared4 m32 held为40/150（36/4/0），相对carrier32 retained/8 gained/6 lost、churn14/150、Jaccard.6957。
  Spatial为30/6/4，Goal为2/2/2；故当前只有小净增，不能据此宣布稳定积累或直接扩大训练。其余shared条件与全部fit视频仍继续。
- 16/16条件已物化。clone72最后两个fit条件在同一个驻留runtime完成，首次115秒、后续0.135秒、实际物化2.7/2.3秒，exit0。
  当前gpu01：shared4 m64 fit(0)、clone93 m32 fit(2)、shared4 m64 held(3)、clone93 m64 fit(4)、clone83 m32 fit(5)、
  shared4 m32 fit(6)。还未启动的评测为clone72两fit、clone83 m64 fit；全部adapter已ready。
- clone83 m32 fit使用evaluator现有`--replicas-per-gpu 4`，每卡仍8 environments/worker；其余已启动条件为3 workers。
  已观测静态LoRA每worker约10--10.3 GiB，预计4 workers约41 GiB、低于A40可用45 GiB；本轮实测吞吐与峰值，未改变模型/视频/状态/RNG
  或scientific preprocessing。该执行选择在该fit条件结果产生前作出，只用于增加同卡有效并行，不能按结果选择runtime。
- 五训练run及carrier相关eval均已完成并保留完整证据，清理了clean且无进程使用的task-owned训练`6624127b`和carrier`54312cf1`
  detached worktrees。相应历史launch可按已pushed commit重建；checkpoint、raw rows、contracts、logs均未删除。

## 前一执行节点（2026-09-06 00:50 CST）

- 驻留物化已从clean pushed detached `d35e66bf`完成shared4剩余三个条件、clone83剩余三个条件和clone93两fit条件，均exit0。
  首次runtime准备115--116秒，后续0.11--0.24秒；shared三task每条件完整物化12.6--13.6秒、Goal约3秒、Long约7.4秒。
  后续条件仍重新完整capture与一次Writer，独立checkpoint/video metadata已按预注册配置核对。当前14/16条件已物化，只余clone72两fit条件。
  时间证据见同analysis root的`materialization_timing.json`；复用仅消除重复加载，没有合并条件、截断输入或缓存最终LoRA。
- 当前5/18个strict eval已完成：source32/150；clone72 m32/m64 held为41/39，配对38 retained/1 gained/3 lost、churn4/50、Jaccard.9048；
  clone83 m32 held6/50；clone93 m32 held1/50。carrier以及shared多条件仍运行，不提前据此作路线裁决。
- 正在gpu01：clone93 m64 held(0)、clone83 m64 held(2)、shared4 m64 held(3)、carrier150(4)、shared4 m32 held(5)、
  shared4 m32 fit(6)。下一释放卡优先shared4 m64 fit，其余clones fit已ready后连续调度；两个节点合计不超过6卡。
- 额外分析只复用本轮原result：shared的零梯度task6/79在32/64 held functional仍负，因而不宣称task迁移；normalizer完全匹配，
  clone有少量clip、shared无clip。两者容量/聚合/AdamW语义仍不同，不能径直归因梯度冲突，详见findings和对应JSON。

## 前一执行节点（2026-09-06 00:40 CST）

- 五组同图正式训练均完成64/64、8/32/64 checkpoints、Panel-B与completion，全部exit0。
  functional汇总见`runs/analysis/pi05_ecp_prw_samegraph_local_b89ee997_20260906/functional_comparison.json`；四task在clone/shared
  两组32/64的fit/held平均benefit均为正。当前继续预注册闭环，不据内部收益宣布视频因果或task-disjoint能力。
- 已完成strict结果：source为32/150，Spatial2/Goal20/Long38=`32/0/0`；clone72 m32 held为41/50，clone83 m32 held为6/50。
  后两者尚须同状态carrier和shared参照；本面板只作train-side定位，不作为validation selector。原始行与完整配对汇总位于同analysis root的
  `closed_loop_comparison.json`，未完成时明确标记partial。共18个预注册eval：12 clone条件、4 shared条件、source/carrier各一。
- 评测沿用clean pushed detached `b89ee997`与支持原carrier静态bank的`54312cf1`；所有worker保持同一预处理与RNG。
  source已释放物理3，clone72 m64 held/clone93 m32与m64 held/shared4 m32 held/carrier分别在gpu01物理2/6/0/5/4运行；
  已完成launch的精确script/log/exit已复制到各输出`launch/`。gpu02没有EMBER新任务。
- 初期source launcher遗漏assets/storage环境变量，均在rollout前失败；已补齐canonical路径并从prepared root继续start成功。
  原尝试日志保留；这是执行环境遗漏，未改变科学条件。carrier直接引用既有canonical adapter，零Writer调用，不复制或重训。
- 实测native捕获每视频约1.9--6.0秒，而每condition重复加载冻结policy/observer耗费数分钟。现在同一物化器增加可重复的
  `--additional-materialization EVAL_CONFIG CHECKPOINT OUTPUT`，复用同run、同task集合的驻留冻结模型；各条件仍独立校验原run/checkpoint/
  fit或held split、重设原seed、完整捕获和一次Writer、单独seal。无新runner、无模型或loss变化。后续用真实剩余条件测加载与物化时间。
  源码净增长约53行、无新文件；632行物化文件与90行准备函数保持同一资产/信息墙职责，架构检查只有review信号、无新增hard。
  training仅增加声明式CLI与非物化phase拒绝，未改变训练语义。7项相关CPU合同、8个预注册配置、CLI phase与重复输出保护检查通过；真实batch计时待下一节点登记。
- 本次`strg01`独立/data1 quota为720506924/1073741824 KiB（hard1084227584），分析root36MiB，shared filesystem84TiB可用；
  剩余物化/1500总rollout仍在原<25GiB新增预算内，无数据或模型复制。

## 当前执行与吞吐（2026-09-06 00:03 CST）

- 五组正式训练均成功进入同图shared runtime：clone1/72/83/93使用gpu01物理0/2/3/4，shared4使用5/6。
  运行根目录与精确launch记录对应`6624127b`；canonical训练worktree为`/data1/user/ymdai/projects/EMBER-worktrees/prw-samegraph-20260905`。
- clone72/83已完成64步、8/32/64 checkpoints及Panel-B，均exit0；task72 m32/m64 fit/held benefit为
  `+.001292/+.001381`与`+.001279/+.001339`，task83为`+.000786/+.000234`与`+.000891/+.000435`。
  两任务两个后期节点三条视频均优于carrier；这是train-side functional学习证据，还没有本轮closed-loop结果。
- 实际训练短任务约5--7秒/update、task93约12秒、双卡shared约17秒/四task update；采用真实8-row、full50、最长87帧。
  owner补充强调训练、推理及后续算法实现都要充分利用GPU、节约实验时间，已纳入稳定要求。后续模块设计同时考虑batch张量、
  高效attention、数据布局、传输/重复大算子，按真实吞吐与长视频峰值裁决执行方案。
- 为执行已预注册的Spatial2/Goal20/Long38 first-fit/held闭环，复用现有物化器与evaluator，放开历史held5/demo5常量为显式配置，
  并校验task归属、训练run中fit/held视频、checkpoint与条件的对应关系。无新训练器、无模型或loss更改；固定val/test依旧排除。
  8个条件配置、4个train24子面板和7项相关CPU合同已通过。源改动集中于现有物化、条件与任务选择/恢复边界；无新增源文件。
  架构自查保留124行的现有声明式manifest测试fixture（新增真实video字段、两种配置参数化），它主要是测试数据，拆碎会降低一致性；
  此为局部规模例外，不增加并行执行路径。其余增长阈值已通过合并到既有职责与抽出任务归属校验消解。

## 首轮同图 clone/shared 启动合同（2026-09-05 23:42 CST）

- 配置为`configs/pi05_ecp_prw_samegraph_{clone1,clone72,clone83,clone93,shared4}_v1.json`；输入与局部行为面板固定在
  `configs/pi05_ecp_prw_samegraph_panel_v1.json`。同一v4拓扑、component-init、seed20260905、whole Writer、无task query，
  所有组运行`--phase shared --mode formal --representation full --initialization component --stop-after-step 64`。
  每task每update一次曝光、8 action rows、lr1e-4/decay1e-6、warmup5+effective59；checkpoint为8/32/64。
  shared每次task1/72/83/93等权各1/4，clones各1；每task视频、Panel-A visit和policy RNG序列已核对64次完全匹配。
- 视频fit/held：task1=`5,6 / 39`，72=`3,8 / 49`，83=`5,7 / 49`，93=`2,3 / 48`。
  task6/79保留零梯度复用诊断；不是fresh selector。首轮32/64局部闭环固定Spatial2、Goal20、Long38各50状态，分别第一fit视频与
  held视频，对照source/carrier，严格复用canonical evaluator配对及执行合同。meta1先看functional，不把该train侧面板作为最终selector。
- 唯一执行修正是共享评估器从写死2个checkpoint改为核对配置中的完整名单；没有模型/损失/训练图改变。4项相关CPU合同通过，
  配置、视频split、全部64次task occurrence及video/row schedule直接核验通过。
- 最长task93真实profile2步自然exit0：79/87 sampled frames，full50、38 targets、8 rows/microbatch2，约11.95/12.25秒每update，
  max allocated35.76/reserved36.58 GiB；冻结模块零可训练参数，whole Writer2,953,984参数，task query0。
  该profile只作执行证据，不作科学资格。输出`.codex/tmp/prw_samegraph_profile_task93_20260905/`。
- 计划GPU分配：gpu01物理0/2/3/4分别clone1/72/83/93，NUMA0/0/1/1；shared使用5/6、NUMA1/1两rank，
  `NCCL_P2P_DISABLE=1`、rank-local NUMA与deferred NCCL。两个节点合计六卡；每次实际启动前刷新双方GPU。
- `/data1` strg01 quota核对719920236 KiB已用、1073741824 KiB quota、1084227584 KiB limit，软额度约337 GiB余量；
  项目实际655 GiB，shared filesystem约84 TiB可用。五run checkpoint+共享mmap+局部物化/rollout总新增保守<25 GiB；
  单卡clone使用进程内CPU缓存，shared仅一份临时mmap，复用全部canonical source/data/model。
- 正式运行从包含本合同的下一clean pushed commit的detached worktree启动，run roots使用
  `runs/outputs/pi05_ecp_prw_samegraph_{arm}_s64_{commit}_gpu01p{devices}_20260905/`；精确命令、NUMA wrapper、log、exit及
  `run_contract.json`保存在各run。fresh optimizer/scheduler；exact resume锁原commit/config/world-size/physical顺序/NUMA与output。

## 前次交接快照（历史记录）

- 当前没有active goal、active design、active配置或运行中的实验。owner于2026-09-05在v4 held5结果完成后要求停止推进、全面清理并交由
  他人接手。`docs/unified_policy_native_factor_writer_design.md`与
  `configs/pi05_ecp_policy_response_writer_unified_factor_v4.json`是最新已裁决设计及配置，均为sealed而非自动续跑入口。
  本文件后续较早条目中的“active/当前/下一步”只表示对应历史时点，不能覆盖本条交接状态。
  较早条目中的`.codex/tmp`路径只是当时的临时launcher/log位置，交接清理后不保证继续存在；科学证据以对应formal root、Git和
  `docs/research_history.md`为准。

- 最新已裁决架构是source-separated common-base Unified Policy-Native Factor Writer。输入端只做token projection；显式
  `frame x target x rank x X/Y-side` latent在每一层用同一query与同一policy-attention
  权重分别读取exact language、同frame image patches和完整PI0.5 response，三者各自softmax；side-matched native bank另作独立
  standard attention，四个读出直接相加，再做teacher-frame time及rank/side attention。learned主干
  仍只有一种可按深度复制的`UnifiedPolicyNativeFactorBlock`；末端同一个head把frame-common context作为两signed分支共享的bank定位
  query、把frame-relative innovation作为两分支偏移，再做direct signed raw-X/Y pooling、target cap和
  唯一rank16。没有独立Process/Composer坐标、source gate、token-count校正或summary/gain/solver/calibration链；full 50-horizon、
  positive-only与信息墙不变。v1/v2/v3及v4均已sealed。

- common-base v3的73-task formal已自然完成200/200并exit0；m100/m200 held5 correct-only strict250为`35/250`与`31/250`，逐task
  Long/Goal/Object/Spatial0/Spatial9为`0/0/3/29/3`与`0/0/4/22/5`，breadth均`3/5`、Goal/Long均0。相对carrier43，m100为
  `30 retained / 5 gained / 13 lost`，m200为`23 / 8 / 20`；m100到m200为`22 / 9 / 13`，所以这是相邻稳定non-pass而不是门槛过高。
  formal root为
  `runs/outputs/pi05_ecp_policy_response_writer_common_base_73task_k1_component_s200_f214fefa_gpu02p46_sharedmmap_20260905/`。

- non-pass后的correct-only责任诊断显示：m200 learned evidence projection在true-held task6/79的held benefit分别给出
  `+.0000363/+.0000336`，而learned factor Writer分别给出`-.0000889/-.0002809`；seen task1则两者均为正，full为
  `+.0003345`。进一步把factor Writer拆成structure、blocks、signed heads后，重复blocks在task1为`+.0002331`，在task6/79为
  `-.0002371/-.0000809`；逐子层替换又没有找到三个task共同的单一坏算子。成功task-local task1/93的factor有效rank也降到约1，
  因而不能用rank regularization、冻结evidence或单层补丁解释失败。最早缺口是整个shared block没有形成可转移的task grounding。

- 闭环责任反事实已自然完成且exit0：从m200只保留trained evidence projection，把完整factor Writer恢复为component initial，零优化
  物化五个唯一rank16后，held5 correct-only strict250为`39/250`，Long/Goal/Object/Spatial0/Spatial9=`0/0/3/35/1`。相对carrier43
  为`36 retained / 3 gained / 7 lost`、Jaccard `.7826`；相对完整m200的31为`19 / 20 / 12`、Jaccard `.3725`。因此去掉trained
  factor Writer确实恢复了大量carrier-aligned成功，但trained evidence + initial Writer仍没有超过carrier、没有增加breadth或Goal/Long；
  该结果定位shared block的破坏性，不构成旧架构候选或v4性能证据。formal analysis root为
  `runs/analysis/pi05_ecp_policy_response_writer_common_base_m200_evidence_only_held5_correct_k1_strict250_d557dffc_gpu02p46_20260905/`。

- information-flow审计确认旧policy read仍大量消费response，但language与256 patch、400 owner-response共用一个softmax时，多数层的
  language质量只有约`2.2%`，接近token-count占比；task-local依赖自由task query绕过了这一缺口。当时的v4只拆开这三个policy source
  的softmax并复用同一MHA参数，再与原side-native read相加，不新增网络、参数、loss或数学阶段。当前分支17项CPU合同全部通过，包含
  source-cardinality不变性与旧v3 config fail-closed。clean `da964fad`上的task93真实full smoke已自然exit0：完整消费79 sampled
  frames、2 probes、50 horizons与38 targets；patch/language/response/policy-read/native-read/unified/signed-X/signed-Y梯度分别为
  `.03624/.03405/.02191/.01390/.00575/.02862/.02099/.01095`，冻结policy/observer零梯度，生成76 tensors和唯一rank16；峰值
  allocated/reserved为`36.15/36.91 GiB`。当时下一步是不追加结构，直接进入25/50短科学资格；该资格现已完成并non-pass。

- source-separated v4 task-local formal launch contract：科学实现固定为clean `da964fad`，配置固定为7,495-byte
  `configs/pi05_ecp_policy_response_writer_unified_factor_v4.json`；formal authority为包含本合同的下一clean pushed main。task1/task93
  各自fresh K1 component-init，冻结source policy、Native Observer与evidence tokenizer，只训练同一Unified Factor Writer及正控专用
  task query；warmup5 + effective45、optimizer25/50 checkpoints、每步8条correct cross-episode functional rows，两条fit视频反传、
  第三条same-task held视频只读。wrong/shuffle/reverse、validation/test、reward与Panel-B均不产生梯度，内部恢复率不设置成人为续跑门槛。
  2026-09-05合同检查时gpu01物理0/3分别约`1.27 GiB/1 MiB`且util均0，二者无compute process；task93固定物理0/NUMA0、task1固定
  物理3/NUMA1，并相对v4最长smoke `36.91 GiB`留有余量。同期gpu02物理4/6正在执行本项目diagnostic rollout，故不挤占；总EMBER卡数
  不超过4。输出固定为
  `runs/outputs/pi05_ecp_policy_response_writer_source_separated_tasklocal_task93_full_s50_da964fad_gpu01p0_20260905/`与task1 corrected fresh
  `runs/outputs/pi05_ecp_policy_response_writer_source_separated_tasklocal_task1_full_s50_da964fad_gpu01p3_r2_20260905/`；
  logs固定`.codex/tmp/prw_source_separated_task93_tasklocal_da964fad_gpu01p0.log`与
  `.codex/tmp/prw_source_separated_task1_tasklocal_da964fad_gpu01p3_r2.log`；首次NUMA fail-fast另保留原无`r2`日志。`strg01` live `/data1` quota为
  `778990080/1084227584 KiB`，limit余量约291.1 GiB，两run复用canonical assets且合计保守小于1 GiB。单进程使用
  `NCCL_P2P_DISABLE=1`、GPU-local NUMA与fresh output；只允许同commit、同config、task、节点/物理卡和topology exact resume。

- v4 task-local formal已完成并exit0。task93 m25/m50 fit/held recovery为`.155260/.147577`与`.218747/.218805`，两个checkpoint
  三条correct视频全部自发优于carrier；m50高于v3的`.206582/.184789`。task1 m25为`.010424/.009039`且一条fit仅
  `-5.6e-6`，m50转为`.043085/.048998`且三条视频全部为正；它保留容量但弱于v3的`.078978/.081318`。两run均完成50/50、
  保存25/50 checkpoints、冻结policy/evidence encoder且held/Panel-B/wrong backward均为0，输出唯一rank16。task1首次launch把物理3
  误绑NUMA0，formal在创建output前fail-fast；失败日志保留，corrected `r2`以NUMA1 fresh运行，未覆盖任何状态。该混合结果只支持立即
  运行73-task 25/50短资格，不支持长跑或新增结构。

- source-separated v4 73-task shared formal launch contract：从包含本合同的下一clean pushed main，以相同7,495-byte v4 config、fresh
  component initialization和optimizer/scheduler运行；K1、full-50、correct cross-episode functional positive-only，55 meta + 18 target
  产生梯度，task6/79只作zero-gradient held诊断。本配置每update采样`9 meta + 3 target`共12 task且各占`1/12`，仅为当前短实验的
  覆盖/成本选择，不是架构或owner固定要求；只运行warmup5 + effective45并保存optimizer25/50，不预先扩长。live preflight时gpu01
  物理0/3/6分别为`1260/1/2 MiB`、util均0；以visible `0,3,6`运行3-rank DDP，rank0绑定物理0/NUMA0，rank1/2绑定物理3/6/NUMA1，
  rank-local wrapper已用3进程真实affinity probe验证为node0的`0-27,56-83`与node1的`28-55,84-111`各56 CPUs。选择这三张空闲卡避免gpu02物理4/6的他人显存，
  且相对实测v4峰值均有余量。fresh output/cache为
  `runs/outputs/pi05_ecp_policy_response_writer_source_separated_73task_k1_component_s50_f02f9148_gpu01p036_sharedmmap_20260905/`与
  `.codex/tmp/prw_source_separated_v4_75task_cache_f02f9148_gpu01_20260905/`；log/exit为
  `.codex/tmp/prw_source_separated_v4_73task_s50_f02f9148_gpu01p036.{log,exit}`。上一matched raw cache实测
  `118247162700` bytes，本轮fresh重建；`strg01` live `/data1` quota为`779098484/1084227584 KiB`，约291.0 GiB limit余量足够cache与
  小于2 GiB output。固定`NCCL_P2P_DISABLE=1`、dynamic cost-balanced assignment和single-copy safetensors mmap；exact resume锁相同
  pushed commit、config、world-size3、visible顺序、rank-local NUMA和cache/output roots。

- source-separated v4 73-task shared已从clean detached `2e3a9612`自然完成50/50并exit0。m25/m50的12个gradient诊断任务
  全视频高于carrier由`5/12`升到`7/12`，fit/held benefit由`-0.000001/+0.000002`升到
  `+0.000029/+0.000045`；两个完全不反传的true-held task6/79在两个相邻checkpoint均为`2/2`全视频正向。m25两task
  fit/held benefit均值为`+0.000100/+0.000141`，m50为`+0.000108/+0.000136`；task6的m25/m50 held为
  `+0.000020/+0.000009`，task79为`+0.000261/+0.000263`。这是相对v3首次出现的相邻稳定task-disjoint正信号，但幅度很小，
  尚不证明闭环收益。训练/Panel-B/总耗时为`1305.66/384.84/1989.10s`，约`26.11s/step`；三rank早期平均工作量差
  约`1.21s`、最大`3.89s`，峰值allocated/reserved为`38.41/39.32GB`。cache已自动删除，checkpoint、raw metrics、
  completion和launch bundle均完整；冻结policy/observer、Action Meta 0、held/wrong backward 0及唯一rank16合同全部满足。

- v4首次held5 closed-loop launch contract：只比较预登记的m25/m50 single checkpoints，不续训、不改结构或loss。两个checkpoint
  分别用held5固定correct K1、每task一次Writer调用物化唯一rank16，再各运行strict paired250；只读train24 held5，不读
  validation/test、wrong/shuffle/reverse或reward。2026-09-05 19:02 CST同时live检查两节点：gpu01物理0/3/6为
  `1260/1/2 MiB`且util 0，gpu02物理2/3为`162/162 MiB`且util 0，物理6有他人`4749 MiB`、util 0的轻量进程；按既有
  evaluator峰值余量可安全共驻且不触碰该进程。m25在gpu01物理3/NUMA1物化后用`0,3,6`动态评测，m50在gpu02物理2/NUMA0
  物化后用`2,3,6`动态评测，两条pipeline并发且EMBER总占6卡。formal roots固定为
  `runs/outputs/pi05_ecp_policy_response_writer_source_separated_m25_held5_correct_k1_materialized_f02f9148_gpu01p3_20260905/`、
  `runs/outputs/pi05_ecp_policy_response_writer_source_separated_m25_held5_correct_k1_strict250_f02f9148_gpu01p036_20260905/`、
  `runs/outputs/pi05_ecp_policy_response_writer_source_separated_m50_held5_correct_k1_materialized_f02f9148_gpu02p2_20260905/`与
  `runs/outputs/pi05_ecp_policy_response_writer_source_separated_m50_held5_correct_k1_strict250_f02f9148_gpu02p236_20260905/`；launch时均须
  不存在。`/data1` quota为`779170288/1084227584 KiB`，约290.9GiB limit余量，新增物化与两份250-row结果保守小于2GiB。
  evaluator保持cost-balanced dynamic queue与每GPU 3 replicas；正式裁决直接对carrier43及v3/v2相邻点报告逐task、breadth、
  retained/gained/lost、churn和配对成功集合，不以微小functional信号替代行为。

- v4 held5两条formal pipeline均自然完成并exit0。m25为`45/250`，逐task Long/Goal/Object/Spatial0/Spatial9=
  `0/0/4/37/4`、breadth`3/5`；相对carrier43为`41 retained / 4 gained / 2 lost`、churn6、Jaccard`.87234`、
  paired exact `p=.6875`。m50为`40/250`，逐task=`0/0/3/34/3`、breadth`3/5`；相对carrier为
  `37 / 3 / 6`、churn9、Jaccard`.80435`、`p=.5078125`。m25到m50为`38 retained / 2 gained / 7 lost`、
  churn9、Jaccard`.80851`、`p=.1796875`。两点Goal/Long均0，早期+2净增没有相邻稳定性，故预登记v4短资格non-pass；
  不解封续训、mixed-K、fully-random Final、validation8或negative controls。

- 该non-pass不是工程断路或输出坍缩：m25到m50五task mobile范数增长约`1.21--2.05x`，同task方向cosine约
  `.433--.930`；跨task mobile cosine仍低，全部Writer参数组均有finite movement。内部true-held functional保持微弱正向而闭环回落，
  把最早未解问题留在shared task-disjoint mapping及correct functional到held closed-loop的信用对齐。它不撤销G1/G2、真实native
  X/Y、完整50-horizon、signed pooling、rank4 task-local容量或v4分源改善证据，也不支持增加手工gain、calibration或数学变换链。

- parallel-read v2的73-task shared formal已自然完成200/200 optimizer steps并exit0；m100/m200的12个gradient Panel-B任务全视频为正
  `5/12 -> 8/12`，fit/held benefit由`+.000111/+.000058`升到`+.000302/+.000184`，但task6/79两点均为`0/2`且均值由
  `-.000319/-.000310`恶化到`-.000691/-.000649`。对应held5 strict250为`40/250`与`38/250`，逐suite分别
  `0/0/3/35/2`与`0/0/2/34/2`，breadth均`3/5`、Goal/Long均0，稳定低于carrier43。训练、两次物化与两次评测均完整，信息墙计数为0。

- non-pass后的correct-only物化几何显示五个held task的nominal rank4 mobile有效参与rank在m100约`1.012--1.023`，m200降至
  `1.007--1.014`；同task m100到m200方向cosine约`.806--.857`而norm放大`1.47--1.74x`，top3 targets承载约`77--88%`能量。
  代码复核确认旧末端先删除`C=mean_t z_t`，再只以`D_t=z_t-C`形成native query，因而共同context中的language、owner、family和rank
  无法直接决定“去当前bank哪里读”。active v3只在同一个线性signed-query head恢复专家原式`b(C)+delta(D)`；base在正负分支共享，
  所以静态视频仍严格零mobile。没有增加模块阶段、loss、gate、归一化、温度或校准链。

- common-base v3实现已完成：X/Y各自原有signed-query linear由`2d -> 3d`，一次输出共享base与两个bias-free innovation offsets，
  Writer仅增加32,768个参数；没有新增模块类型或并行fallback。16项CPU合同与config/schema互斥检查通过。gpu01物理0上的task93
  profile smoke自然exit0，完整消费79 sampled frames、2 probes、50 horizons与38 targets；policy/native read梯度为
  `.002008/.002443`，unified/signed-X/signed-Y为`.006853/.003875/.002644`，冻结policy/observer无梯度，生成76 tensors和唯一
  rank16；峰值allocated/reserved约`33.79/35.84 GiB`。

- common-base v3 task-local formal launch contract：科学实现固定为clean pushed `7d6f2d3a`，配置固定
  `configs/pi05_ecp_policy_response_writer_unified_factor_v3.json`，formal authority为包含本合同的下一clean pushed main。task1/task93
  各自fresh、K1、component initialization；warmup5 + effective45、optimizer25/50 checkpoints、每步8条correct cross-episode
  functional rows，两条fit视频产生梯度、第三条same-task held视频只读；不读取wrong/shuffle/reverse、validation/test或reward。
  task93使用gpu01物理0/NUMA0，task1使用gpu02物理6/NUMA1；launch检查时前者无compute process、后者仅有约4.75 GiB低util他人进程，
  两者相对对应实测峰值均有充分余量且不触碰他人进程。输出固定为
  `runs/outputs/pi05_ecp_policy_response_writer_common_base_tasklocal_task93_full_s50_7d6f2d3a_gpu01p0_r2_20260905/`与
  `runs/outputs/pi05_ecp_policy_response_writer_common_base_tasklocal_task1_full_s50_7d6f2d3a_gpu02p6_r2_20260905/`，launch前均不存在；
  stdout/stderr分别写入`.codex/tmp/prw_common_base_task93_tasklocal_r2_7d6f2d3a.log`与
  `.codex/tmp/prw_common_base_task1_tasklocal_r2_7d6f2d3a.log`，不预先创建formal output root。
  `/data1` quota为`778716248/1084227584 KiB`，limit余量约`291.4 GiB`，两run复用canonical assets且合计新增保守小于1 GiB。
  单进程固定`NCCL_P2P_DISABLE=1`与GPU-local NUMA；只允许同commit、同节点/物理卡、config、task与topology exact resume。

- 首次两条launch在optimizer step0前按合同fail-fast：launcher为写日志提前创建了原无`r2`后缀的output root，runtime因fresh root非空
  拒绝封存run contract。两个无效root都只有一份1,044-byte traceback log，没有run contract、metrics、checkpoint或科学结果；原样保留
  审计且绝不resume/overwrite。修正只改变日志位置和fresh output名，不改变代码、config、GPU、数据或科学合同。

- corrected common-base v3 task-local formal均自然完成50/50并exit0。task1的m25/m50 fit/held recovery为
  `.025794/.035692`与`.078978/.081318`，相对matched v2的`.021949/.023451`与`.059224/.064798`在两点全面改善；task93为
  `.156805/.137100`与`.206582/.184789`，也全面高于v2的`.114529/.106962`与`.161922/.153538`。两任务、两相邻checkpoint、
  两条fit与一条未反传same-task held正确视频全部自发高于carrier。task1与task93 train/eval/total分别为
  `324.96/85.43/421.44s`与`554.31/90.52/665.85s`，峰值reserved约`26.00/36.23 GiB`；冻结墙与held/Panel-B/wrong backward
  计数均为0，输出均为唯一rank16。该matched结果支持共同context确实承担bank定位，而非只靠innovation学习通用近rank1方向；它是
  task-local接口正证据，不冒充shared或闭环收益。下一步直接进入matched 73-task optimizer100/200 shared与两点held5。

- common-base v3 73-task shared formal launch contract：科学代码固定为`7d6f2d3a`，active training config固定为clean pushed
  `main@f214fefa`中的`configs/pi05_ecp_policy_response_writer_unified_factor_v3.json`（6,869 bytes），formal authority为包含本合同的
  下一clean pushed main。实验从fresh optimizer/scheduler与component initialization训练whole Writer；K1、full-50、correct
  cross-episode functional positive-only、55 meta + 18 target、该配置每update `9 meta + 3 target`、rows8、optimizer100/200
  checkpoints保持与v2 matched。task6/79只作已消费zero-gradient诊断；same-task held、Panel-B与true-held均不反传，不读取
  wrong/shuffle/reverse、validation/test outcome或reward。每update任务数及meta/target比例仅为本次matched配置，不是架构或owner固定要求。
  输出固定为
  `runs/outputs/pi05_ecp_policy_response_writer_common_base_73task_k1_component_s200_f214fefa_gpu02p46_sharedmmap_20260905/`，single-copy
  mmap cache固定`.codex/tmp/prw_common_base_75task_cache_f214fefa_gpu02_20260905/`，stdout/stderr固定
  `.codex/tmp/prw_common_base_73task_s200_f214fefa_gpu02p46.log`；三者launch前均不存在，formal output root不由日志提前创建。
  2026-09-05 launch前live检查中，gpu02物理4/6分别约`5.75/4.75 GiB`、`2%/0%`，均为低util他人进程；相对最长task93
  v3实测`36.23 GiB`峰值仍留约`4--5 GiB`，不触碰其它任务。两卡同属NUMA1，使用world2 dynamic cost queue、rank-local NUMA、
  deferred NCCL及`NCCL_P2P_DISABLE=1`。同一时点gpu01物理0空闲，保留给m100封存后的并行物化/评测，不跨节点拼训练卡。
  gpu02 available host RAM约266 GiB；`/data1` quota为`778834860/1073741824 KiB`，quota余量约281.2 GiB；约105 GiB cache加
  保守小于2 GiB formal输出安全，cache在训练/物化消费后删除。若他人显存动态增长触发OOM，只按同commit/topology exact resume，
  不降低full horizon或改变科学合同。

- combined-softmax v1的task1/task93正式25/50-step控制均从clean detached `06f3b465`完成。task1 m25/m50 fit/held recovery为
  `-.007016/-.002323`与`-.003950/+.005152`，两个checkpoint均没有三条视频全为正；task93为
  `+.024906/+.026362`与`+.097318/+.068702`，两点三条视频均为正。相对前代Native-Temporal，v1参数更多、速度更慢且两task容量
  都没有改善，因而没有直接进入73-task shared。

- correct-only、零优化attention-mass诊断进一步定位了首版根因：每frame prefix/response分别为456/400 tokens，X bank为100，Y bank
  则随target为400、3200或12800；在同一softmax中，X bank只获得约`8--14%`质量，而Q/action-in的Y bank约占`70--96%`。
  task1训练还普遍降低bank质量，task93仅在后层局部恢复。因此失败不是统一latent断图，而是实现定义的token cardinality让语义来源
  竞争同一归一化。

- parallel-read v2已由`831d9d6c`集成并推送到`main`，只在同一block内以独立policy/native cross-attention替换
  combined softmax，其余block、depth、full bank、loss、rank与readout完全不变。15项定向CPU回归通过，包含native tokens成倍复制不改变
  读出的合同。gpu01物理3上的task93真实full smoke自然exit0：79 sampled frames、2 probes、完整50 horizons、38 targets；
  policy/native read梯度分别为`.002554/.003015`，prefix/response/unified/signed-X/signed-Y分别为
  `.008025/.006184/.008155/.001529/.005662`，冻结policy/observer无梯度，生成76 tensors和唯一rank16；峰值
  allocated/reserved约`34.90/38.77 GiB`。下一步是clean pushed detached authority上的matched task1/task93 25/50控制。

- parallel-read v2 task-local formal launch contract：科学实现固定为clean pushed `831d9d6c`，配置固定
  `configs/pi05_ecp_policy_response_writer_unified_factor_v2.json`，formal authority为包含本合同的下一clean pushed main。task1/task93
  各自fresh、K1、component initialization；冻结policy、Native Observer和evidence tokenizers，只训练同一parallel-read Factor Writer与
  task-local query；warmup5 + effective45、optimizer25/50 checkpoints、每步8条correct cross-episode functional rows。两条fit视频
  产生梯度，第三条same-task held视频只读；不读取wrong/shuffle/reverse、validation/test或reward，也不把内部恢复率设置为续跑门槛。
  输出固定为
  `runs/outputs/pi05_ecp_policy_response_writer_parallel_read_tasklocal_task1_full_s50_831d9d6c_gpu01p3_20260905/`与
  `runs/outputs/pi05_ecp_policy_response_writer_parallel_read_tasklocal_task93_full_s50_831d9d6c_gpu01p6_20260905/`，launch前必须不存在。
  最近live候选为gpu01物理3/6，但正式launch前仍同时刷新gpu01/gpu02；`/data1` quota为
  `778441340/1084227584 KiB`，limit余量约`291.6 GiB`，两run复用canonical assets且合计新增保守小于1 GiB。命令为detached authority下
  `NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=<3|6> PYTHONPATH=src ... scripts/train_ecp_policy_response_writer.py --phase task-local
  --representation full --initialization component --mode formal`。

- 上述两条parallel-read v2 formal均从clean detached `810f32d3`自然完成并exit0。task1 m25/m50 fit/held recovery为
  `.021949/.023451`与`.059224/.064798`；task93为`.114529/.106962`与`.161922/.153538`。两个任务、两个相邻checkpoint的
  两条fit与一条held正确视频全部高于carrier；相对combined-softmax v1，task1由两点不通过变为两点通过，task93 m50由
  `.097318/.068702`提高到`.161922/.153538`，且高于前代Native-Temporal约`.125/.126`。task1/task93 train/eval/total分别为
  `345.13/85.65/441.88s`与`517.19/89.57/627.12s`；峰值reserved为`26.00/42.01 GiB`。冻结policy/evidence、held/Panel-B/wrong
  backward计数均为0，输出均为唯一rank16。这证明parallel source ownership修复了task-local接口，但尚不构成shared或闭环结论。
  shared前只做保持逐元素语义等价的frame-chunk内存/吞吐profile，随后立即进入73-task optimizer100/200短资格。

- task93前两步exact frame-chunk profile使用相同初始化、视频、functional rows与full candidate轴。chunk128基线为
  `10.06/10.56s`、峰值reserved `42.01 GiB`；chunk64为`10.92/11.42s`、`36.22 GiB`；chunk32为
  `11.85/11.91s`、`33.57 GiB`。分支级activation checkpoint在chunk128下只降到`41.94 GiB`且更慢，已完整丢弃，不进入runtime。
  选择chunk64作为唯一active执行点：约8%单task开销换取`5.79 GiB`最长视频余量，使gpu02物理4/6可与各自约5.6/4.6 GiB低util
  进程安全共驻并使用6卡；所有frames、50 horizons、native candidates、softmax与输出保持不变，CPU chunk-equivalence合同通过。

- parallel-read 73-task shared formal launch contract：科学实现固定`831d9d6c`，exact chunk64执行配置固定于clean pushed
  `38329b92`，formal authority为包含本合同的下一clean pushed main；配置为
  `configs/pi05_ecp_policy_response_writer_unified_factor_v2.json`。从fresh optimizer/scheduler和component initialization训练whole Writer；
  K1、full-50、correct cross-episode functional positive-only、55 meta + 18 target、每step本配置`9 meta + 3 target`、rows8、
  optimizer100/200 checkpoints均固定。task6/79只作已消费zero-gradient诊断；Panel-B、same-task held与true-held均不反传，且不读取
  wrong/shuffle/reverse、validation/test outcome。输出固定
  `runs/outputs/pi05_ecp_policy_response_writer_parallel_read_73task_k1_component_s200_38329b92_gpu02p012346_sharedmmap_20260905/`，
  single-copy mmap cache固定`.codex/tmp/prw_parallel_read_75task_cache_38329b92_gpu02_20260905/`，launch前均不存在。缓存沿用此前同一
  95-task authority实测约105 GiB，formal输出保守小于2 GiB；`/data1` quota为`778547352/1084227584 KiB`，limit余量约291.5 GiB；
  gpu02 available host RAM约322 GiB，峰值增长安全。正式launch前同时刷新gpu01/gpu02，计划在gpu02物理`0,1,2,3,4,6`使用world6
  dynamic cost queue、rank-local NUMA、`NCCL_P2P_DISABLE=1`、0 GiB private replication和node-local single-copy mmap；物理4/6只有
  约5.6/4.6 GiB低util他人进程，chunk64最长实测36.22 GiB仍留约3--4 GiB余量，不触碰物理5/7。预计约1--1.5小时得到两点Panel-B，
  随后立即并发物化和运行两点held5 correct-only strict250。

- clean detached `07804433`的Frame-Bank 12-gradient + 2-held whole-Writer 50-step资格已完整结束。m25/m50 gradient-task
  fit/held benefit为`+.0001573/+.0001166`与`+.0002399/+.0001997`，全视频为正由`6/12`升至`8/12`；fresh held task3持续为正、
  task77持续为负，两点都只有`1/2`，故没有运行held5、negative controls或续训。train/eval/total为`262.29/293.95/631.46s`，
  peak reserved约`34.42 GiB`，信息墙计数全部为零。

- non-pass后的correct-only VJP显示m50六task整体梯度pairwise mean `.05557`、负比例`.40`，不是普遍共享冲突；event readout只有
  `.01402`且task93/94对其它task和为负，signed-X head mean `-.05783`、task93对其它和`-.66264`，signed-Y则为`+.08754`。
  task1/task93冻结路径消融中，frame-only只对task1微正，event-only对task1为负且对task93也只有不稳定微量贡献。因此最早失效点是
  “独立event压缩后晚期接bank、X/Y到末端才共享分叉”的接口；下一实现是整体责任替换，而非增加event权重或输出校准。

- Native-Temporal实现`78a0ca6a`已集成并推送到`main`。active runtime删除独立Temporal/Event classes、event readout、Composer二次
  query seed和四个base/contrast heads，新增显式X/Y side的单一`NativeTemporalFactorBlock`；source + tests净删除约146行，learned图
  仅保留Frame与NativeTemporalFactor两种block。25项Writer/native定向测试和py_compile通过，旧Frame-Bank config被新loader拒绝，
  没有兼容alias或并行fallback。

- clean detached `78a0ca6a`在gpu02物理0完成task93真实profile smoke：79 sampled frames、2 probes、完整50 horizons、38 targets；
  Frame/NativeTemporal/Signed-X/Signed-Y functional梯度分别为`.00407/.00314/.00271/.00188`，冻结墙通过，生成76 tensors与唯一rank16。
  峰值allocated/reserved约`32.02/35.57 GiB`。随后2-step Composer-only真实profile为`3.679/3.464s`，train/eval/total
  `7.21/6.99/35.01s`，两步NativeTemporal和X/Y heads梯度均非零；峰值allocated/reserved约`29.54/38.22 GiB`。因此50-step
  task-local资格预计训练约3分钟、含加载评测约5分钟，不存在架构未证明前的长跑成本问题。

- Native-Temporal task-local formal launch contract：科学实现固定`78a0ca6a`，formal authority为包含本合同的下一clean pushed main；
  配置固定`configs/pi05_ecp_policy_response_writer_native_temporal_12gradient_2held_v1.json`。task1/task93各自K1、component initialization、
  冻结Frame Encoder且只训练Native-Temporal Composer、warmup5+effective45、optimizer25/50 checkpoints、每步8 correct cross-episode
  functional rows、fit视频训练与Panel-B/held视频只读；不读取wrong/shuffle/reverse，不产生held或Panel-B梯度。输出固定为
  `runs/outputs/pi05_ecp_policy_response_writer_native_temporal_tasklocal_task1_full_s50_78a0ca6a_gpu02p2_20260905/`与
  `runs/outputs/pi05_ecp_policy_response_writer_native_temporal_tasklocal_task93_full_s50_78a0ca6a_gpu02p0_20260905/`，launch前均不存在。
  `/data1` live quota为`778392860/1084227584 KiB`、limit余量约`291.67 GiB`，两run复用canonical assets且保守新增小于1GiB。
  gpu02物理0的最长实测峰值reserved约`38.22 GiB`；正式launch前仍同时刷新gpu01/gpu02并只选余量安全的卡，不干扰他人进程。

- 上述两条Native-Temporal task-local formal均已从clean detached `934bd82c`自然完成并exit0。task1在m25的fit/held recovery为
  `-.00633/-.01934`，到m50转为`+.03197/+.01693`，两条fit与一条held视频均高于carrier；task93在m25/m50的fit/held为
  `+.12288/+.12213`与`+.12536/+.12624`，两个相邻点三条视频全部为正。task1显示该图对普通任务学习较慢且m50容量弱于
  Frame-Bank，task93则相邻稳定并接近Frame-Bank的`.13780/.13596`；因此不能宣称task-local容量全面提高，但已经直接证明显式X/Y
  side与NativeTemporal blocks不是零容量接口。两run的train/eval/total分别为`145.75/82.02/238.37s`与
  `172.96/81.54/275.25s`，峰值reserved约`21.34/30.19 GiB`；source policy、wrong、held及Panel-B backward计数均为0，输出均为
  唯一完整rank16。

- Native-Temporal 12-gradient + 2-held shared formal launch contract：科学实现保持`78a0ca6a`，formal authority为包含本合同的下一
  clean pushed main；配置仍为`configs/pi05_ecp_policy_response_writer_native_temporal_12gradient_2held_v1.json`。gradient meta
  固定`[1,2,8,9,32,52]`、gradient target固定`[72,73,74,75,93,94]`，在读取本架构结果前已固定的fresh zero-gradient held为
  task4/78。K1、component initialization、whole Writer、warmup5+effective45、optimizer25/50 checkpoints、每步本配置选择
  `3 meta + 3 target`，使每个gradient task到m50恰有25次暴露；Panel-B、same-task held与true-task-held均只读，wrong、shuffle、
  reverse均不读取且不反传。使用node-local single-copy safetensors mmap、动态cost-balanced task assignment、
  `NCCL_P2P_DISABLE=1`与单节点4卡；输出固定为
  `runs/outputs/pi05_ecp_policy_response_writer_native_temporal_12gradient_2held_k1_component_s50_78a0ca6a_gpu02p0123_sharedmmap_20260905/`，
  cache固定为`.codex/tmp/prw_native_temporal_14task_cache_78a0ca6a_gpu02_20260905/`，launch前二者均必须不存在。该短资格直接检验
  显式factor-side是否改善shared task-disjoint映射；不以task1内部恢复率另设人为准入线，也不在本轮追加输出补丁。

- 上述Native-Temporal shared formal已从clean detached `53131aae`自然完成并exit0。m25/m50 gradient-task全视频为正为`7/12`
  与`8/12`，fit/held benefit由`+.00013531/+.00008523`增至`+.00030324/+.00018066`；但fresh zero-gradient held task4/78在
  m25均为正，到m50均转负。task4三视频平均benefit由`+.00023626`降至`+.00002188`，task78由`+.00005355`降至
  `-.00005258`；task74训练视频改善而held视频恶化到`-.00070093`。train/eval/total为`270.16/294.03/629.32s`，peak
  reserved约`34.79 GiB`；source policy、true-held、same-task-held、Panel-B及wrong backward均为0，输出为唯一完整rank16。

- 冻结m25/m50后的correct-only诊断已先校正为与formal完全相同的canonical rank4和每visit完整16 Panel-B rows；task4单点复算与
  formal loss精确吻合。task4/78的m25->m50真实mean loss分别增加`+.00021437/+.00010613`。Simpson路径积分显示伤害分散：task4
  最大项为Process tokenization`+.00009075`，两层Frame合计`+.00003895`，两层Native MLP合计`+.00003029`，signed-output
  `+.00001424`；task78同样在Process、Frame、bank tokenization、Native MLP与两侧head上广泛同向。task74路径高度非线性且大项
  相消：bank tokenization、bank read与signed-input有益，而Frame、Native MLP、Process tokenization与signed-output抵消。
  六task梯度几何的整体pairwise mean为`-.00588`、负比例`.667`。结论是12-task后半共享更新整体过拟合/冲突，不是单一head故障；
  不据此追加末端数学补丁，先做m25闭环并以同图扩大task覆盖。

- Native-Temporal m25 held5 correct-only launch contract：checkpoint固定为上述shared run的`macro_00000025`，只因预注册Panel-B
  task-disjoint证据在任何held5 rollout前选择；不使用checkpoint union或m50结果融合。held5固定global task `0/9/18/25/36`、每task
  correct demo5、K1、一次Writer调用、唯一38-target rank16；不读取wrong、shuffle、reverse、validation/test action/reward/state。
  scientific implementation仍为`78a0ca6a`，formal authority为包含本合同和active held5 config的下一clean pushed main。物化与
  strict250 roots固定为
  `runs/outputs/pi05_ecp_policy_response_writer_native_temporal_m25_held5_correct_k1_materialized_78a0ca6a_gpu02p0_20260905/`和
  `runs/outputs/pi05_ecp_policy_response_writer_native_temporal_m25_held5_correct_k1_strict250_78a0ca6a_gpu02p012_r3_20260905/`，
  launch前均不存在。2026-09-05 live同时检查两节点后，gpu02物理0/1/2仅约`209/162/162 MiB`、0% util，只有同一共享gqma
  约`148--186 MiB`低占用，选择三卡materialize后`3 replicas/GPU`动态Evaluator；总EMBER占卡3张。`/data1` quota为
  `778540680/1084227584 KiB`，limit余量约`291.5 GiB`，本次复用canonical assets且新增远小于1GiB。固定launcher为
  `.codex/tmp/launch_prw_native_temporal_m25_held5_78a0ca6a_gpu02p012_20260905.sh`；launch前还会刷新两节点且不干扰他人进程。

- 下一scale科学合同已冻结为`configs/pi05_ecp_policy_response_writer_native_temporal_v1.json`，原12-gradient配置标记sealed。模型、
  initialization、width128、2 Frame + 2 NativeTemporal blocks、K1、full-50、raw native X/Y、positive-only functional、rank12+4、
  cap和唯一rank16完全不变；唯一主要因果变量是gradient task从12扩到73。gradient meta为55个、target为18个，fresh zero-gradient
  held按事先规则固定task6/79；Panel-B仍评估同一组6 meta + 6 target并加两个fresh held，便于与本轮直接比较。每step `9 meta +
  3 target`只是在55:18规模下近似task等暴露的本配置选择，不是架构或owner固定比例；optimizer100/200时每task分别恰有`16--17`
  与`32--34`次暴露。warmup5+effective195、rows8、K1下预计为一小时内短跑，不扩到十小时；两个checkpoint都将直接做correct-only
  held5 strict250。若fresh held和闭环仍随训练反向，则不再用更多步数、LR/seed或数学校准挽救这一函数类。

- m25 single-checkpoint held5 correct-only strict250已自然完成并exit0：总分`39/250`，Long/Goal/Object/Spatial0/Spatial9为
  `0/0/3/35/1`，breadth`3/5`。相对stable carrier `43/250`为`37 retained / 2 gained / 6 lost`、Jaccard `.82222`、paired
  exact `p=.28906`。这说明early task4/78 functional正信号没有迁移为闭环净增；同时它保留37个carrier成功、churn仅8，故也不是
  完全失控的随机adapter。物化manifest确认五task各一次Writer调用、每task唯一完整rank16、held action/reward/state与validation/test
  reads均为0；Evaluator 29/29 shards、250/250 rows、9/9 workers完整，三卡active wall `1235.66s`。正式roots即上一合同登记的
  materialized与strict250路径。

- Native-Temporal 73-task optimizer100/200 formal launch contract：科学实现仍为`78a0ca6a`，配置固定
  `configs/pi05_ecp_policy_response_writer_native_temporal_v1.json`，formal authority为包含本合同的下一clean pushed main；fresh
  optimizer/scheduler、component initialization、whole Writer、K1、rows8、full-50、correct-only functional、55 meta + 18 target、
  每step本配置`9 + 3`及fresh zero-gradient held6/79均锁定。coverage audit来自clean `541f2a5f`，确认95-task authority、73
  gradient tasks、optimizer100/200时每task`16--17/32--34`次暴露且存在factorial signal；不读取wrong/shuffle/reverse，不产生
  Panel-B、same-task-held或true-held梯度。输出固定为
  `runs/outputs/pi05_ecp_policy_response_writer_native_temporal_73task_k1_component_s200_78a0ca6a_gpu02p012346_sharedmmap_20260905/`，
  single-copy cache固定为`.codex/tmp/prw_native_temporal_75task_cache_78a0ca6a_gpu02_20260905/`，launch前均不存在。此前同数据面
  world6 cache约`97.8--105 GiB`，本轮保守峰值小于110GiB；`/data1` quota为`778214672/1084227584 KiB`、limit余量约
  `291.8 GiB`，gpu02 available host RAM约`314.8 GiB`，余量充足。双节点live检查后选择gpu02物理`0,1,2,3,4,6`：前四张仅
  `162--209 MiB/0%`，物理4/6约`5.75/4.75 GiB`且连续观测`2%/0%`，均有同图shared实测peak reserved `34.79 GiB`所需余量；
  使用world6动态cost queue、rank-local NUMA、`NCCL_P2P_DISABLE=1`、0GiB private replication及node-local single-copy mmap。
  总EMBER占卡6张，不触碰物理5/7或gpu01任务；固定launcher为
  `.codex/tmp/launch_prw_native_temporal_73task_s200_78a0ca6a_gpu02p012346_20260905.sh`。预计一小时内得到两点Panel-B，科学图无变化。

- 上述73-task formal已从clean detached `7b89da92`自然完成并exit0。200/200 metrics、macro100/200两枚完整checkpoint、
  `result.json`与`completion.json`齐全；train/eval/total为`1264.68/232.67/1725.42s`，step均值约`6.30s`，peak
  allocated/reserved约`34.26/34.80 GiB`，105GB shared mmap cache已在成功收尾后自动删除。m100/m200的12个见过Panel-B任务
  fit/held benefit为`+.00027657/+.00029709`与`+.00043504/+.00039299`，全视频高于carrier为`8/12`与`9/12`；
  fresh zero-gradient task6/79两点都为`0/2`，fit/held聚合由`-.00064467/-.00038182`恶化到
  `-.00082644/-.00068045`。source policy、Native Observer、Action Meta、same-task held、true-held与Panel-B均为零梯度，
  wrong/shuffle/reverse读取为零，输出合同为唯一rank16。该结果否定“12-task数据量过小是主要解释”，但按预注册仍须用两点
  correct-only闭环直接裁决。

- m100/m200 held5 correct-only strict250 launch contract：固定使用上述两枚single checkpoint、active held5 config、global task
  `0/9/18/25/36`、每task correct demo5、K1、一次Writer调用及唯一38-target rank16；不做checkpoint union，不读取wrong、
  shuffle、reverse或validation/test outcome。物化/评测分别固定为
  `runs/outputs/pi05_ecp_policy_response_writer_native_temporal_m100_held5_correct_k1_materialized_78a0ca6a_gpu02p0_20260905/`、
  `runs/outputs/pi05_ecp_policy_response_writer_native_temporal_m100_held5_correct_k1_strict250_78a0ca6a_gpu02p012_r3_20260905/`以及
  `runs/outputs/pi05_ecp_policy_response_writer_native_temporal_m200_held5_correct_k1_materialized_78a0ca6a_gpu02p3_20260905/`、
  `runs/outputs/pi05_ecp_policy_response_writer_native_temporal_m200_held5_correct_k1_strict250_78a0ca6a_gpu02p346_r3_20260905/`；launch前均不存在。
  2026-09-05同时live检查两节点：gpu01仅物理0/6空闲、其余多卡高负载；gpu02物理0--3仅`162--209MiB/0%`，物理4/6约
  `5.75/4.75GiB`且`2%/0%`，因此m100使用`0,1,2`、m200使用`3,4,6`并发，最多6张EMBER卡，不触碰物理5/7或他人任务。
  `/data1` quota为`778281120/1084227584 KiB`，limit余量约`291.8GiB`，新增远小于1GiB。唯一参数化launcher为
  `.codex/tmp/launch_prw_native_temporal_held5_checkpoint_7b89da92.sh`，避免为两个checkpoint复制运行面。

- 上述m100/m200 strict250均已自然完成并exit0。m100为`42/250`，Long/Goal/Object/Spatial0/Spatial9为
  `0/0/4/35/3`、breadth`3/5`；相对carrier43为`36 retained / 6 gained / 7 lost`、Jaccard`.73469`、paired exact
  `p=1.0`。m200为`35/250`，逐task为`0/0/4/29/2`、breadth`3/5`；相对carrier为`28 retained / 7 gained / 15 lost`、
  Jaccard`.56`、paired exact`p=.13380`。m100到m200自身为`33 retained / 2 gained / 9 lost`、Jaccard`.75`、paired exact
  `p=.06543`，训练后期呈明确方向性退化。两run均为29/29 shards、250/250 rows、9/9 workers且所有worker exit0；active wall
  分别`1409.06/1392.87s`。这与seen functional继续改善、fresh held继续恶化一致，故task diversity不是主要缺口，也不再续训或
  扫LR/seed/数学校准。

- 下一correct-only职责诊断已于2026-09-05 09:40 CST从同一clean detached `7b89da92`启动。对gradient task1/93与已消费为
  zero-gradient true-held诊断的task6/79，逐一比较initial、m100/m200 full、仅移入trained Process及仅移入trained Composer；每个
  状态只读取两条fit正确视频和一条same-task held正确视频的固定8-visit Panel-B，不训练、不rollout、不读取wrong/shuffle/reverse、
  validation/test或reward。gpu02物理0--3 launch前仅`162--209MiB/0%`，四task各占一卡；`/data1` quota为
  `778335648/1084227584 KiB`。该诊断只决定冻结/替换哪个完整职责，不会向当前图追加变换。

- 四task职责诊断均在约`323--344s`内自然exit0。held benefit的`initial / trained Process only / trained Composer only / full`在
  m100为：task1 `-.000015 / +.000138 / -.000021 / +.000195`，task6 `+.000022 / -.000219 / +.000012 / -.000306`，
  task79 `+.000097 / -.000149 / -.000108 / -.000458`，task93 `-.000038 / -.000535 / -.000117 / -.000646`；m200保持同一
  责任模式，分别为task1 `-.000015 / +.000117 / -.000007 / +.000264`，task6 `+.000022 / -.000277 / +.000002 /
  -.000697`，task79 `+.000097 / -.000226 / -.000208 / -.000661`，task93 `-.000038 / -.000291 / -.000165 /
  -.000416`。fit视频结论一致。full臂与原formal在正常BF16跨卡归约差异内复现相同符号与排序，carrier逐值相同；四root均明确
  optimizer/rollout/negative/validation/test reads为0。

- 诊断不支持把问题缩成“只冻结Process”：Process确实是主要task-specific正负源，但Composer checkpoint单独在四task均没有形成
  可复用正增量，joint interaction又只在task1为正、在task6/79为负。结合历史Process-frozen Composer-functional只持平carrier的
  证据，下一变量是整体删除Process--Composer learned-coordinate handoff。新图使用一种重复的Unified Policy-Native Factor block，
  让显式frame x target x rank x X/Y latent在每层直接读取frozen prefix、full policy response与same-frame native bank，再做time及
  rank/side axial attention；末端科学边界不变。不新增summary/gate/normalization/transport/calibration链。

- 等待闭环时完成一次verified workspace cleanup：逐一确认11个旧formal detached worktree均clean且gpu01/gpu02/mgt无进程引用后，
  只移除其checkout登记；Git提交、formal artifacts、checkpoint和当前Native-Temporal worktree均保留。现在现场只有canonical main
  与当前formal detached worktree。

- role-equal formal已完整结束且non-pass。m50/m100 held5 correct-only strict250为`39/45`；m100逐task Long/Goal/Object/Spatial0/
  Spatial9=`0/0/2/41/2`、breadth`3/5`，仍没有Goal/Long。m50/m100 gradient-task fit/held benefit为
  `+.0001670/+.00009662`与`+.00034273/+.00015433`，但两个true-task-held均值仍为负。覆盖修复和target质量提高有小幅作用，
  没有解决shared Program--bank utility映射，因此不再扫meta/target比例或修补旧gain链。

- Axial实现已由`3cc4dbfc`集成进clean pushed `main`；相对前代净删除约`1.8k`行，独立gain module已删除。25项
  Writer/native tests、py_compile和diff check通过。synthetic覆盖full-only、order sensitivity、K2视频置换不变、chunk等价、
  dense/streaming value+gradient等价及静态重复视频mobile近零。

- gpu01物理3的真实task72 smoke完整消费22 sampled frames、2 probes、50 horizons、38 targets与真实native X/Y；functional loss
  `.0667395`，Prefix/Response/Frame/Temporal/Event/Composer/Signed X/Signed Y梯度均非零，冻结policy零梯度，生成76 tensors和唯一
  rank16。峰值allocated/reserved约`15.34/19.50 GiB`。随后5-step task-local profile约`2.4--2.6s/step`；fit videos 3/8和
  held video43的benefit分别为`+.0002928/+.0001267/+.00008453`，三条均同向，但恢复仅`.02118/.00971/.00609`，只证明短图已
  接通，尚不构成容量或shared性能结论。

- 73-task whole-Writer 25/50-step formal已结束：300次总task exposure使55个meta task各仅约2--3次、18个target各约8--9次；
  m25/m50 gradient fit/held benefit均约`-1e-4`，true-task-held non-pass，按资格规则没有运行held5。该短跑排除“极少暴露即可学会”，
  但没有以低暴露结果否定可扩展图。

- 进一步的正确视频反事实把最早失效点定位到旧Composer。相同50次whole-Writer更新下，task72三条fit/held benefit约
  `+.0010--.0012`，task1仅`+.0001--.0003`，task93为`-.0013--.0015`。旧shared m25/m50的event仍保持task差异，但训练后B侧
  跨task cosine明显升高；A侧初始化时已约`.90`同向，最终task差异主要由B决定。去掉Composer的首次bank-context读取仅小幅降低
  同质化，证明主因不是重复bank read本身。

- clean `04b22550`上的task93 Composer-only 25/50-step正控最终三条fit/held视频均为正，但m50 held benefit仅`+.000600`、recovery
  `.04547`，相对free primal约`.01320` benefit只实现很小一部分，也远低于历史frame-local Composer约`.28--.30` held recovery。
  历史代码与专家合同共同指出：旧强实现把每个candidate chunk绑定到同frame innovation；当前Axial实现却以一个global dynamic
  query给全部frame打分，丢失了event/native candidate的frame-local对应。

- 隔离分支`codex/frame-aligned-factor-decoder`已用单一`FrameAlignedFactorBlock`整体替换该职责，并净删约103行Composer代码：
  rank query读取events，每个frame按自身位置读取本视频events，frame dynamic同时驱动X/Y signed contrast；完整bank只在最终exact
  pooling读取一次。25项Writer/native CPU测试全部通过。gpu02物理2上的task93真实smoke消费79 sampled frames、2 probes、完整50
  horizons与38 targets；全部learned模块梯度有限非零，生成76 tensors和唯一rank16，峰值allocated/reserved约`30.20/34.43GB`。

- clean detached `da1657ef`的Frame-Aligned 12-task shared已完成100步。m50/m100 gradient fit/held benefit为
  `+.000475/+.000383`与`+.000561/+.000477`，全视频为正仅`6/10`与`7/10`；两个true-task-held两点均`0/2`且fit/held聚合约
  `-.0025/-.0037`与`-.0025/-.0033`。m50到m100结论稳定，故没有运行held5、负controls或续训。训练`581.76s`、评测
  `294.37s`，三卡动态调度下step均值约`5.79s`、峰值allocated/reserved约`30.01/30.17GiB`。

- non-pass后的两轮零优化correct-only几何把最早失效接口锁定到Composer：Process event pairwise cosine median约`.552`、
  task-specific fraction median约`.590`，而global query cosine median约`.996`、task-specific仅约`.055`；两层event read norm只有
  输入残差的`.28--.45%`。task74 update与72/73/75 cosine为`.740/.540/.380`，真实functional gradient却仅
  `.144/.076/-.244`；task74当前update的一阶benefit为`-.000698`，并同时改善三个邻近task，证明是bank-relative方向误路由，
  不是幅度、训练不足或上游时序坍缩。由于post-hoc诊断读取了task2/74的授权Panel-B gradient，它们不再作为下一设计的unseen held。

- 隔离分支`codex/frame-bank-factor-decoder`已用单一`FrameBankFactorBlock`替换失效职责；同一逐帧bank token既供block标准
  attention又供末端raw-value pooling，position不进入bank memory，逐视频frame centering保证静态repeat不能打开mobile。
  14项当前Writer合同测试与task93真实full-50 forward/VJP/materialization smoke均通过；79 sampled frames、2 probes、38 targets、
  76 tensors与唯一rank16全部保留，Prefix/Response/Frame/Temporal/Event/Composer-bank/Signed-X/Signed-Y梯度均有限非零。
  严格等价地把frame chunk从8增至128后，相同task93第一步由`8.02s`降至`3.95s`，峰值reserved约`28.55 GiB`。
  下一动作是固定并集成实现后，并行运行task1/task93 25/50-step Composer-only正控；局部容量成立后才以fresh held tasks做短shared资格。

- Frame-Bank task-local launch contract：科学实现固定为`1323f8ed`，formal authority为包含本合同的下一clean pushed `main`；配置固定
  `configs/pi05_ecp_policy_response_writer_frame_bank_v1.json`。task1/task93各自K1、component initialization、Composer-only、
  warmup5+effective45、optimizer25/50相邻checkpoint、8 functional rows、正确cross-episode fit与只读held video；不读取wrong、
  shuffle或reverse，不产生held梯度。两条run从同一clean pushed detached worktree在gpu02物理0/2并行运行，输出固定为
  `runs/outputs/pi05_ecp_policy_response_writer_frame_bank_tasklocal_task1_full_s50_1323f8ed_gpu02p0_20260905/`与
  `runs/outputs/pi05_ecp_policy_response_writer_frame_bank_tasklocal_task93_full_s50_1323f8ed_gpu02p2_20260905/`，launch前均不存在。
  2026-09-05准备时`/data1` quota为`778200976/1084227584 KiB`，limit余量约`291.9 GiB`；两run复用canonical assets且只新增小型
  checkpoints/metrics。双节点live检查时gpu02物理0/2仅约`209/162 MiB`、0% util，实测最长task93峰值reserved约`28.55 GiB`，
  均有安全余量；launch前再次同时刷新gpu01/gpu02，不等待凑卡、不干扰其它进程。

- 上述Frame-Bank task1/task93 formal已从clean detached `471592f4`并行自然完成。task1 m25/m50 fit/held recovery为
  `.05404/.05262`与`.08408/.07016`；task93为`.04461/.04253`与`.13780/.13596`，两任务、两相邻点的三条正确视频聚合均优于
  carrier。相对Frame-Aligned，task93 m50明显改善，task1则fit略升、held下降；整体仍只恢复free primal约`5--14%`，所以
  same-frame bank参与方向形成是有效增量，但Composer-only容量仍不充分。task1/task93 train/eval/total分别约
  `145.39/83.03/239.34s`与`184.18/79.07/283.91s`，峰值reserved约`20.92/29.46 GiB`。

- 下一短shared资格不再把task2/74伪装为unseen held：在读取任何新架构task3/77结果前，按“每个role取post-hoc暴露后最小eligible
  未读ID”的outcome-independent规则固定task3/77为zero-gradient held；task2/74移入gradient，与原10 tasks组成6 meta + 6 target。
  `configs/pi05_ecp_policy_response_writer_frame_bank_12gradient_2held_v1.json`只运行50步、m25/m50，每步3+3只是该配置选择；12个
  gradient task均精确获得25次暴露。这个whole-Writer实验直接判断可训练Process与FrameBank能否共同形成task-disjoint方向，不靠延长
  task-local训练或增加专用数学补丁。

- Frame-Bank 12-gradient + 2-held shared launch contract：科学配置固定为`83109a33`，formal authority为包含本合同的下一clean pushed
  `main`。K1、component initialization、whole-Writer correct-only functional、12个gradient tasks、fresh zero-gradient held task3/77、
  每步该配置的3 meta + 3 target、每task 8 rows、warmup5+effective45及m25/m50均锁定；不读取negative controls，不运行held5。
  输出固定为
  `runs/outputs/pi05_ecp_policy_response_writer_frame_bank_12gradient_2held_k1_component_s50_83109a33_gpu02p0123_sharedmmap_20260905/`，
  单份node-local mmap固定为`.codex/tmp/prw_frame_bank_14task_cache_83109a33_gpu02_20260905/`，两者launch前均不存在。预计cache约
  `20--25 GiB`、formal输出远小于1 GiB；`/data1` quota为`778303288/1084227584 KiB`，limit余量约`291.8 GiB`。
  双节点live检查后gpu02物理0/1/2/3仅约`209/162/162/162 MiB`且0% util，选择同一NUMA0的world4动态cost调度；四卡都有最长
  task93正式峰值`29.46 GiB`以上的余量，总EMBER占卡4张。launch前再次同时刷新gpu01/gpu02，固定`NCCL_P2P_DISABLE=1`。

- Frame-Aligned task-local launch contract：科学实现固定为`e2f38c2a`，formal authority为包含本合同的下一clean pushed main；
  科学变量仅为上述Composer职责替换；配置仍使用
  `configs/pi05_ecp_policy_response_writer_axial_factor_v1.json`，task1/task93各自K1、component initialization、Composer-only、
  warmup5+effective45、optimizer25/50 checkpoints、每次8 functional rows、正确cross-episode fit与只读held video。两run从包含本合同
  的clean pushed commit建立detached worktree，输出分别固定为
  `runs/outputs/pi05_ecp_policy_response_writer_frame_aligned_tasklocal_task1_full_s50_e2f38c2a_gpu01p0_20260905/`与
  `runs/outputs/pi05_ecp_policy_response_writer_frame_aligned_tasklocal_task93_full_s50_e2f38c2a_gpu02p2_20260905/`且launch前不存在；
  不覆盖旧run、不读取wrong/shuffle/reverse、不产生held梯度。复用canonical assets，预计每run仅小checkpoint/metrics。
  2026-09-05 launch准备时`/data1` quota为`777973108/1084227584 KiB`，limit headroom约`292.1 GiB`。两节点live检查中gpu01物理0
  约`1.29GB/0%`、gpu02物理2约`.16--.43GB/0%`，均有task93实测峰值所需余量；正式launch前再次刷新，最多并行占这两张卡。

- Axial task72 formal launch contract：科学实现为`3cc4dbfc`，authority为包含本合同的下一clean pushed main；配置固定
  `configs/pi05_ecp_policy_response_writer_axial_factor_v1.json`，K1、component initialization、task-local Composer-only、
  warmup5+effective45、optimizer25/50 checkpoints、每次8 functional rows，输出固定
  `runs/outputs/pi05_ecp_policy_response_writer_axial_factor_tasklocal_task72_full_s50_3cc4dbfc_gpu01p0_20260905/`，launch前不存在且不覆盖。
  2026-09-05 launch前`/data1` quota为`777765776/1084227584 KiB`，limit headroom约`292.3 GiB`；本run复用canonical assets，
  仅产生小checkpoint/metrics，峰值远低于余量。gpu01/gpu02同时live检查后，gpu01物理3与6均为约1MiB、0% util；选择已实测
  峰值reserved约18.9GiB的物理3单卡，不占用或干扰其它进程。正式launch前再次刷新时，物理3/6已被他人新占约21.5GiB，
  因此按live state改选物理0；它约1.3GiB、1% util且无计算进程，余量明显更安全。固定命令为
  `NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /data1/user/ymdai/projects/EMBER/.venv/bin/python scripts/train_ecp_policy_response_writer.py --config configs/pi05_ecp_policy_response_writer_axial_factor_v1.json --asset-root /data1/user/ymdai/projects/EMBER --data-root /data1/user/ymdai/projects/EMBER/data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a --output-dir /data1/user/ymdai/projects/EMBER/runs/outputs/pi05_ecp_policy_response_writer_axial_factor_tasklocal_task72_full_s50_3cc4dbfc_gpu01p0_20260905 --phase task-local --task 72 --representation full --initialization component --mode formal`。

- 上述task72 formal已在clean detached `89d912b7`自然完成。macro25的fit/held recovery为`.09362/.08602`，fit videos3/8与
  held43 benefit均约`+.00125--.00132`；macro50为`.07660/.06156`，三条仍全部为正但弱于m25。训练50步`127.6s`、总计
  `214.6s`，峰值allocated/reserved约`15.60/17.56 GiB`。相邻点证明direct readout不是断图，但后半回落且明显弱于旧task72
  set-relative约`.3`恢复；不靠延长Composer-only训练挽救。由于该控制冻结全新的Temporal/Event Process，下一项直接训练完整
  Writer，以区分“Composer函数类上限”与“冻结随机时序表示限制”。

- clean detached whole-Writer task72两步shared profile已自然完成：训练全部`3,735,168` Writer参数，step约`1.81/1.50s`；
  Prefix/Response/Frame/Temporal/Event/ComposerContext/Signed-X/Signed-Y梯度全部有限非零，冻结policy/observer零梯度，峰值reserved
  约`17.86 GiB`。该profile只验证端到端训练和资源路径，不用两行rows2结果作科学选择。

- Axial 73-task shared短资格的首次world3执行从clean detached `eaa26fe0`启动；146份共享evidence cache已完整生成，约`98 GiB`。
  step1为`7.837s`，但step2在其它rank等待时长期没有完成。独占gpu01物理0复现最长task92后确认：51 sampled frames的整步仅
  `2.412s`，但whole-Writer训练峰值allocated/reserved为`25.62/29.16GB`；world3所用物理6因他人低util进程约占`22GB`，实际
  余量只有约`24GB`。因此失败是执行调度错误地用短task72约`18GB`峰值估计所有任务，而非full算子慢或科学non-pass。该run在
  checkpoint前主动终止；60KB不完整root和旧cache authority已可恢复地移至
  `.codex/tmp/aborted_prw_axial_world3_memory_pressure_20260905/`，98GiB safetensors保留复用，没有删除正式证据或触碰他人进程。

- 修正后的Axial 73-task shared短资格launch contract：科学实现仍为`3cc4dbfc`，formal authority为包含本合同的下一clean pushed
  `main`。配置、73个gradient tasks、K1、component initialization、joint correct-only functional、warmup5+effective45、
  optimizer25/50 checkpoints、每步6 tasks与本run的`3 meta + 3 target`、每task 8 functional rows及只读Panel-B全部不变；这些batch
  设置不是owner或架构常量。唯一执行变化是在具有足够最长样本峰值余量的gpu01物理0上world-size1串行执行六个task，避免不安全
  共驻和collective等待；单卡不会改变task权重、optimizer cadence或样本序列。输出固定为
  `runs/outputs/pi05_ecp_policy_response_writer_axial_factor_73task_k1_component_s50_3cc4dbfc_gpu01p0_sharedmmap_20260905/`，复用cache
  `.codex/tmp/prw_axial_factor_73task_cache_3cc4dbfc_gpu01_20260905/`。launch前再次同时live检查gpu01/gpu02、quota、authority、
  output不存在及cache的146份safetensors。固定命令为
  `NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /data1/user/ymdai/projects/EMBER/.venv/bin/python scripts/train_ecp_policy_response_writer.py --config configs/pi05_ecp_policy_response_writer_axial_factor_v1.json --asset-root /data1/user/ymdai/projects/EMBER --data-root /data1/user/ymdai/projects/EMBER/data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a --output-dir /data1/user/ymdai/projects/EMBER/runs/outputs/pi05_ecp_policy_response_writer_axial_factor_73task_k1_component_s50_3cc4dbfc_gpu01p0_sharedmmap_20260905 --phase shared --representation full --initialization component --mode formal --shared-evidence-cache-root /data1/user/ymdai/projects/EMBER/.codex/tmp/prw_axial_factor_73task_cache_3cc4dbfc_gpu01_20260905`。

## 2026-09-04及以前进度账本

- Set-relative shared已完整裁决为non-pass，但authority72正控证明同一函数类的functional信用可以转化为闭环行为。clean detached
  `3686baec`的shared m50/m100 held5 correct-only strict250为`37/44`，m100与前一pointwise m100同为`44/250`、breadth仅`3/5`
  且Goal/Long为0；10-task correct held-video功能诊断只有3/10明确为正，四个target task为`0/4`。相反，task72 task-local
  optimizer50/100的fit recovery为`.31231/.35311`、held recovery为`.28411/.32669`，三条视频全优于carrier；同一50初态上
  carrier/m50/m100为`34/35/40`，m100相对carrier为`30 retained / 10 gained / 4 lost`（exact `p=.179565`），并达到历史
  task expert `38/50`的同一行为量级。50/50 env seed与共同policy-noise prefix已逐项一致，成功episode提前终止导致完整noise
  序列长度不同，不是pairing错误。因此当前没有证据支持“Evaluator完全失真”或“functional-to-LoRA-to-behavior链路根本断裂”；
  最早缺口收窄为shared task-conditioned/task-disjoint mapping及其训练信用。

- 当前frozen role-equal短跑保持73个gradient tasks、每步总12 task、模型、数据、full-50、K1、loss、LR、100步与m50/m100不变，
  把`9 meta + 3 target`改为`6 meta + 6 target`。launch后的逐task sampler审计发现，它不只是把target角色总权重由25%提高到
  50%：旧18-task target组每步取3个，使每task严格每6步出现；global-step驱动的2-video与16-visit游标因此发生周期别名，100步内
  每个target只见一条fit video和8/16个Panel-A visits。改为6个后每task每3步出现，恰好恢复两条video与16/16 visits。因此该run
  重新解释为“target训练质量/覆盖联合修复”，不能把可能收益只归因于role weight。它仍不是固定meta/target比例、固定每步task数
  或最终规则；若该更强联合修复仍失败，就不再扫比例。

- Role-equal formal launch contract：科学配置与task72证据固定为`28b4eb05`，formal authority为包含本合同的下一clean pushed main；
  配置固定`configs/pi05_ecp_policy_response_writer_factor_set_relative_gain_role_equal_v1.json`，数据仍为canonical filtered source、
  73个gradient tasks与原Panel，输出固定
  `runs/outputs/pi05_ecp_policy_response_writer_factor_set_relative_gain_role_equal_73task_k1_component_s100_28b4eb05_gpu02p013_sharedmmap_20260904/`，
  单份临时mmap固定`.codex/tmp/prw_factor_set_relative_gain_role_equal_73task_cache_28b4eb05_gpu02_20260904/`；两者launch前均不存在。
  预计同构100步约55分钟（其中训练约38分钟），m50完成即可与后半训练并行物化/评测。2026-09-04 launch前`/data1` quota为
  `777636444/1084227584 KiB`，limit headroom约`292.4 GiB`，预计cache约`98 GiB`、formal output约`0.1 GiB`，峰值安全。
  两节点live检查后gpu02物理`0,1,3`仅约`.21/.16/.16 GiB`且0% util，选择同一NUMA0的world3、
  `NCCL_P2P_DISABLE=1`与shared mmap；gpu01虽也有三张候选，但不跨节点拼卡。正式启动前再次同时刷新两节点；总EMBER占卡先为3，
  m50评测与训练重叠时至多6，不触碰其它用户进程。只在m50/m100运行correct held功能与held5 strict250，不运行negative controls。

- 已在独立`codex/prw-task-cursor`修正最早责任点：video、Panel visit与causal pair不再读global optimizer step，而读该task自己的
  occurrence cursor；resume从确定性task schedule重建cursor，execution cost plan与实际选中video使用同一cursor。这样任意task数、
  meta/target比例与world size都不会因公因数而永久漏掉某条fit video或一半Panel。旧`9 + 3`的精确复现是target每task仅1/2 video、
  8/16 visits；修复后在相同100步、相同16--17次target exposure下为2/2与16/16。最终改动相关5项测试通过；全量重跑在17项
  已通过时因与正式训练争用CPU而主动停止，不把非必要测试置于科学吞吐之上。当前frozen run不改代码，结果到达后再决定是否需要
  同权重的fresh cursor-only裁决。gpu01物理3上的task72真实full-50两步profile也自然exit0：记录的task occurrence为`0/1`、
  fit video为`3/8`、Panel visit为`0/1`，step为`2.668/2.407s`，峰值reserved约`16.73GB`；functional VJP、Writer重算、
  Composer有限梯度与唯一rank16路径均接通。

- 同时发现当前cross-episode functional batch虽在原始dataset中生成`action_is_pad`，processor却未把它传给已有的
  `pi05_mean_flow_loss(..., action_is_pad=...)`路径；因此episode尾部重复最后动作的padding参与了损失。73个gradient task的
  Panel A/B平均padding约`17.61%/18.46%`，约`35.35%/36.81%`的query row含padding，且六条历史shared Writer曲线中
  padding比例与功能benefit的rank correlation均为正。对m100的task2/53/72/74/78/93使用同一LoRA、correct held视频、query、noise
  和flow time直接比较后，legacy benefit为`+.001050/+.001062/+.000806/-.002431/-.000645/-.000377`，valid-action-only为
  `+.001181/+.001093/+.000703/-.002814/-.000708/-.000358`：六个符号全部不变，task74剔除padding后反而更差。因此padding会改变
  数值但不是当前shared失败根因，不修改训练loss。诊断保存在
  `runs/analysis/pi05_ecp_policy_response_writer_factor_set_relative_gain_m100_masked_functional_tasks2_53_72_74_78_93_3686baec_gpu02p3_20260904.json`。

- Task72 functional-to-closed-loop formal launch contract：科学配置变更固定为`60fca3a5`，只把已有reference与Panel均完整的
  authority72加入`task_local_positive_control`；模型、full 50-horizon、correct cross-episode functional、两fit/一held视频、
  component initialization、rank12+4、唯一rank16、optimizer与100-step cadence均不变。formal authority为包含本合同的下一
  clean pushed main，单卡从gpu01/gpu02 launch前live状态中选择，不占用超过1张EMBER卡。输出固定为
  `runs/outputs/pi05_ecp_policy_response_writer_factor_set_relative_gain_tasklocal_task72_full_s100_60fca3a5_gpu01p3_20260904/`；完成后
  使用held correct video在同一50个environment states上比较Writer与精确carrier。2026-09-04 launch前`/data1` quota为
  `777549876/1084227584 KiB`，约余`292.5 GiB`；同构task-local与物化实测仅约`57/10 MiB`，不创建dataset/model副本。

- 当前正在实现下一单变量`within-target set-relative factor gain readout`。它保留每个rank/group的contextual query、normalized
  signed X、signed Y group与shared group embedding，但不再让token各自独立通过pointwise MLP；同一target的`rank x ragged-group`
  tokens先经过一个标准pre-norm self-attention + GatedMLP block，再由唯一shared scalar output产生gain。attention不跨target，
  新增target/层仍只增加token；full 50-horizon、Process冻结、signed direction、rank12+4、cap、数据、task权重、positive-only loss
  与信息墙均不变。配置为`configs/pi05_ecp_policy_response_writer_factor_set_relative_gain_v1.json`，旧pointwise config由loader明确拒绝。
  隔离分支`codex/prw-factor-set-relative`的61项相关CPU测试通过；task1真实51-frame、2-probe、full-50两步GPU smoke自然exit0，
  Process梯度严格为0，Composer direction为`2.722/1.466`，set conditioner在零output-weight的step1为0、step2为`.0007293`，
  wall为`3.499/3.282s`，峰值allocated/reserved为`23.88/24.12GB`，唯一38-target、76-tensor rank16被policy functional VJP消费。
  科学实现已固定为`534303cf`；下一步形成clean pushed authority，并在同规模optimizer50/100 shared训练期间并行完成本函数类
  task-local正控。

- Set-relative formal launch contract：科学实现固定为`534303cf`，formal authority为包含本合同的下一clean pushed main；训练配置固定
  `configs/pi05_ecp_policy_response_writer_factor_set_relative_gain_v1.json`。shared保持与pointwise前驱相同的73个gradient tasks、当前
  run-specific `9 meta + 3 target`等权近似、K1、component initialization、Process冻结、correct cross-episode functional、轻
  preservation、full 50-horizon、100 optimizer steps及m50/m100 checkpoints，唯一科学变量为pointwise token MLP改为同target
  `rank x ragged-group` self-attention + GatedMLP。gpu02使用同一NUMA0的物理`0,1,3`、world3、`NCCL_P2P_DISABLE=1`与shared mmap；
  gpu01物理`3/6`分别并行运行task1/task93相同100-step task-local正控，总EMBER占卡5张。启动前两节点均已live核验：所选卡空闲，
  无EMBER残留；`/data1` quota已用`777298356 KiB / 1073741824 KiB`，约余`283 GiB`，预计shared临时cache约`98 GB`、checkpoint与
  task-local增量远小于余量。shared输出固定
  `runs/outputs/pi05_ecp_policy_response_writer_factor_set_relative_gain_73task_k1_component_s100_534303cf_gpu02p013_sharedmmap_20260904/`，
  cache固定`.codex/tmp/prw_factor_set_relative_gain_73task_cache_534303cf_gpu02_20260904/`；task-local输出固定
  `runs/outputs/pi05_ecp_policy_response_writer_factor_set_relative_gain_tasklocal_task{1,93}_full_s100_534303cf_gpu01p{3,6}_20260904/`。
  只在原生checkpoint节点做correct-only功能/held与held5闭环；不运行wrong/shuffle/reverse、不读取held action/reward、不续训坏曲线。

- 前一pointwise factor-conditioned gain formal已完整裁决为non-pass。clean detached `ef066789`的73-task m50/m100 held5
  correct-only strict250为`40/44`，逐task由Long/Goal/Object/Spatial0/Spatial9=`0/0/5/32/3`变为`0/0/3/38/3`；m100相对
  carrier43仅`35 retained/9 gained/8 lost`、paired exact `p=1.0`，且Goal/Long仍为0。gradient-task fit/held benefit从
  `+.000278/+.000254`升到`+.000412/+.000333`，两个true-task-held却从`-.000266/-.000299`恶化到
  `-.000316/-.000406`，因此不续训或运行negative controls。相同函数类的task1 task-local m50/m100 fit/held recovery为
  `.388/.407`与`.532/.439`，task93为`.385/.355`与`.440/.370`；四点全部fit/held视频都优于carrier，明确保留local容量与
  同task跨视频泛化。

- 已完成的Factor-conditioned gain formal launch contract：scientific implementation固定为clean pushed `3de3fcb9`，formal authority为包含
  本合同的下一clean pushed main。配置固定
  `configs/pi05_ecp_policy_response_writer_factor_conditioned_gain_v1.json`，仍为73 gradient tasks、当前matched的
  `9 meta + 3 target`、K1、component initialization、冻结Process、correct cross-episode functional、preservation、full 50-horizon、
  真实native X/Y、rank12+4、完整target cap与唯一rank16。相对上一Composer-functional run的唯一科学变量是把195个
  target-owned query-only rows替换为共享current-factor-conditioned group token readout；signed candidate direction、LR、task权重、
  data、Panel与checkpoint cadence均不变。运行100 optimizer steps并保存50/100；m50 checkpoint完整后即可在另一节点开始物化/评测，
  与m50到m100训练重叠但总EMBER占卡不超过6，训练自然完成后补m100并作相邻closed-loop裁决。

  输出固定为
  `runs/outputs/pi05_ecp_policy_response_writer_factor_conditioned_gain_73task_k1_component_s100_3de3fcb9_gpu02p123_sharedmmap_20260904/`，
  临时单份mmap为`.codex/tmp/prw_factor_conditioned_gain_73task_cache_3de3fcb9_gpu02_20260904/`，两者已确认不存在。2026-09-04
  17:49 CST两节点live检查时，gpu02物理`1,2,3`均仅有约`.16GB/0%`host公共进程，适合world-size3；物理0/5/7高显存或重载，
  4/6已有约`5.75/4.75GB`他人进程，而前一正式run最长task reserved可到约`43.1GB`，故不冒险共驻。gpu01物理`0,3,6`可留给
  m50并行物化/rollout，其余为他人高负载任务。launch前仍同时刷新两节点；若上述三张训练卡漂移则只换同节点安全卡并先更新合同。
  `strg01` live quota为`777081864/1084227584 KiB`，limit headroom约`292.92GiB`；同构single-copy cache实测约`97.81GiB`、formal
  output小于`1GiB`，峰值安全低于/data1独立quota，不复制dataset或model。

  该formal已于2026-09-04 17:56 CST从clean detached `ef066789`实际启动，tmux为`ember_prw_fcg_s100`。启动前再次同时live
  检查gpu01/gpu02并复查quota、authority与不存在的output/cache/exit roots；gpu02物理`1,2,3`状态仍为约`.16GB/0%`，故按合同
  使用world-size3、`NCCL_P2P_DISABLE=1`与GPU-local NUMA。torchrun父进程和3个rank均已建立，当前处于一次性model/cache冷启动；
  不触碰gpu02其它卡上的他人进程。m50完整checkpoint出现后，最多再使用gpu01物理`0,3,6`做并行物化/strict250，总EMBER卡数不超过6。

- 冻结Process的Composer-functional正式资格已经完整裁决为non-pass。scientific implementation为`45b63c97`、clean detached
  authority为`a9baa7a4`，训练root为
  `runs/outputs/pi05_ecp_policy_response_writer_composer_functional_73task_k1_component_s100_45b63c97_gpu01p023456_sharedmmap_20260904/`。
  100条metrics、macro50/100、Panel-B、completion、两点物化与held5 correct-only strict250均完整；m50/m100为`39/43`，逐task
  Long/Goal/Object/Spatial0/Spatial9=`0/0/3/35/1`与`0/0/4/37/2`，breadth均`3/5`且Goal/Long为0。m100只是与carrier43持平，
  相对carrier为`36 retained/7 gained/7 lost`；m50到m100为`33/10/6`。因此冻结Process恢复了上一joint m100=35的一部分，
  但没有形成新性能，且不续训或运行negative controls。

- Composer-functional的根因诊断排除了“没训练够、全局方向坍缩、cap普遍截断”。seen functional最后25步平均benefit约
  `+.000889`且约79%为正；m50/m100的Composer direction相对初始化移动`.719%/.825%`，gain head移动`9.29%/10.88%`，Process
  严格为0。四个held/bridge task的最终mobile update task-specific fraction约`.85--.87`，但raw group gain跨task cosine
  `.99972/.99973`、task-specific fraction仅约`.013`。task2/task74的exact group-logit下降方向总体cosine为`-.5846`，
  `v/action-out`为`-.5208/-.6231`，而实际gain总体cosine为`.9991`。task74在固定当前directions下仅以10% logit-norm调整gain的
  局部可用下降约`.001932`，远大于当前first-order benefit `-.0000935`；这直接支持先更换readout输入/参数共享方式，而不是修改
  signed candidate生成器或扩大训练。关键诊断为
  `runs/analysis/pi05_ecp_policy_response_writer_composer_functional_m50_m100_checkpoint_movement_a9baa7a4_gpu01p3_20260904.json`、
  `runs/analysis/pi05_ecp_policy_response_writer_composer_functional_m100_group_logit_credit_tasks2_74_a9baa7a4_gpu01p6_20260904.json`及
  对应m50/m100 geometry、family finite-ablation文件。

- Group-gain-credit的正式科学裁决已经完成。clean detached authority为`a0797488`，formal root为
  `runs/outputs/pi05_ecp_policy_response_writer_group_gain_credit_73task_k1_component_s100_aebd9d74_gpu01p023456_sharedmmap_20260904/`；
  100条metrics、macro50/100、Panel-B、completion、物化与两点held5 correct-only strict250均完整。m50/m100分别为`37/250`与
  `35/250`，逐task Long/Goal/Object/Spatial0/Spatial9=`0/0/5/29/3`与`0/0/3/30/2`，breadth均`3/5`，都低于carrier
  `43/250`且Goal/Long为0；相邻为`28 retained/7 gained/9 lost`。因此不续训、不运行negative controls。

- non-pass后的correct-only诊断把根因进一步收窄。m50/m100的gradient-task fit/held功能增量仍为正，但true-held聚合为负；task2
  随训练改善，task74由`-.001095`恶化到`-.002365`。task2/74的Process innovation cosine为`.706/.686`，mobile update cosine
  由`.493`降到`.258`，说明上游动态和bank方向没有整体坍缩；然而Event assignment、Composer query和group-scale ratio在m100
  的跨task cosine分别为`.99644/.99143/.99941`，实际gain几乎是task-invariant平均解。有限幅家族消融显示task74 m100的
  q-only为`-.002379`，v-only/action-out-only为`+.000074/+.000034`，但action-out有效更新仅为其`s_ref`的`.01365`，并且仅约
  G1成功task-local更新幅度的`.016--.020`。当前训练的process与functional共享Process梯度总和cosine为`-.1141`、Event为
  `-.2913`，process范数分别为functional的`1.49x/1.80x`；相对旧random-delta checkpoint冲突明显增强。参数移动又确认m50前
  已完成主要学习，m100的Process/Composer/scale相对初始化移动为`3.03%/.77%/9.43%`，不是断图或单纯训练不足。

- 下一单变量是active design已预留的Composer-functional阶段，而不是增加数学链：从component initialization fresh冻结Process，
  只更新同一Composer，并只使用correct cross-episode functional与preservation。模型前向、full 50 horizon、native X/Y、
  group gain、rank、数据与task权重不变。该短实验先回答联合Process辅助/漂移是否压坏共享Composer；若仍在gradient tasks正而
  true-held/闭环低，则才有证据修改Composer的动态gain readout函数类。

- held5 m50 evaluator曾在250/250 rows和全部worker exit0后因硬编码`GPU 0--3 -> NUMA0`拒绝聚合；gpu01真实拓扑为0--2在
  node0、3--6在node1。`6ab0c518`已改为校验worker报告的非负NUMA节点而不猜GPU编号，14项evaluator测试通过；原始rows已无重跑
  恢复为`37/250`。这是执行层缺陷，不改变科学non-pass。

- causal-filter 73-task fresh资格已经完整裁决为non-pass。clean detached authority为`db354581`，formal root为
  `runs/outputs/pi05_ecp_policy_response_writer_causal_filter_73task_k1_component_s100_f6b58aac_gpu01p0156_sharedmmap_20260904/`；
  optimizer50/100、Panel-B、result/completion和两点held5 correct-only strict250均完整。m50/m100为`38/36`，逐task
  Long/Goal/Object/Spatial0/Spatial9=`0/0/2/33/3`与`0/0/3/31/2`，breadth均`3/5`且Goal/Long为0；相邻
  `25 retained/11 gained/13 lost`。m100 gradient fit/held benefit为`+.000341/+.000353`，两个true-task-held为
  `-.002125/-.001937`。filter已使event坐标与完整视频明显对齐、predictor仍比zero改善约`5.19%`，但没有转化成shared闭环，
  因此不续训或运行negative controls。

- non-pass后的正样本根因诊断已经完成。task1/72/75/93冻结m100方向、只训练rank gain时，100步fit/held恢复为
  `.151/.093`、`.166/.147`、`.122/.116`、`.150/.099`；恢复专家明确要求的ragged output-group gains后为
  `.244/.198`、`.222/.189`、`.146/.126`、`.240/.202`。相同group-gain下component-init方向四任务均值为
  `.178/.135`，m100方向为`.213/.179`，证明shared训练学到小幅方向但远不充分。formal Composer梯度几乎由gain head占据，
  原因是严格零gain让首步只有gain得到功能信用、后续direction梯度继续被小gain衰减；而G1既有free oracle实际从logit `0.1`
  启动。family ablation又显示action-out在5/5任务为正、q只有2/5为正，排除统一family幅度修补。

- group-gain credit修正已由`aebd9d74`集成并推送到canonical main，临时实现分支/worktree与已结束的causal formal worktree均已
  清理。实现只把family-rank gain恢复为195-row target-native ragged
  group gain，并以G1已有`0.1` logit小幅启动，使第一次functional backward同时到达gain与direction；完整target BA cap、
  zero-innovation零mobile、Process、teacher、loss、LR、task比例、full 50-horizon、真实X/Y、rank12+4与数据不变。新config为
  `configs/pi05_ecp_policy_response_writer_group_gain_credit_v1.json`及对应held5 config；47项Writer/native定向测试通过。gpu01物理0
  的真实task1 demo5 smoke完整消费51帧、2 probes、全部50 horizons和38 targets，初始A/B均非零；functional loss为`.152187`，
  第一次反向的Frame/Event/Composer-direction/group-gain梯度为`.056881/.053071/.098988/.221375`，相对旧零启动的Frame/Event约
  提高20倍。process loss`.163691`且Frame/Event/Predictor梯度`.065089/.058663/.202458`；76 tensors、唯一rank16与峰值
  allocated/reserved `27.35/33.99GB`均通过。task1单卡两步profile也已自然exit0：step为`3.686/3.497s`，peak allocated/reserved为
  `27.36/38.50GB`；Composer总/gain梯度在step1为`6.398/6.116`、step2为`3.123/2.732`，对应非gain方向约
  `1.879/1.512`，确认optimizer、独立clip与方向更新均真实接通。profile rows2与完整carrier统计口径不同，loss不作科学选择。

- gpu01恰有6张真正空闲A40后，`aebd9d74`又完成73-task、12 tasks/update、shared-mmap、rows2的world6两步profile；root为
  `.codex/tmp/policy_response_group_gain_credit_world6_profile2_aebd9d74_gpu01p023456_20260904/`。两步均严格为每rank 2个真实task，
  step为`6.361/6.695s`，预测cost范围`73--78/78--86`，峰值allocated/reserved为`27.36/37.04GB`；方向梯度为
  `.4478/.3196`，gain梯度为`1.7117/1.8523`。相对既有同73-task rows2 world4的`10.597/9.260s`均值快约`34.2%`；虽非逐参数
  matched，但新增gain仅约2.5万参数，且2-task/rank的真实临界路径和低all-reduce代价直接支持world6。完整294份、`97.81GiB`
  action-hidden mmap在成功后已删除，信息墙计数均为0。

- Group-gain-credit formal launch contract：scientific implementation为`aebd9d74`，formal authority为clean pushed
  `a0797488`。配置固定`configs/pi05_ecp_policy_response_writer_group_gain_credit_v1.json`，保持73个gradient tasks、当前实验的
  `9 meta + 3 target`、K1、component-init、correct-only cross-episode functional、positive causal process、preservation、
  full 50-horizon、真实native X/Y、rank12+4与唯一rank16。该比例和12-task batch只为与前代作单变量匹配，不是owner或未来固定
  要求。唯一科学变化是195-row ragged target-native group gain与G1既有`0.1`初始logit；Process、teacher、loss权重、LR、数据和
  materialization均不变。运行100 optimizer steps并保存50/100；输出固定为
  `runs/outputs/pi05_ecp_policy_response_writer_group_gain_credit_73task_k1_component_s100_aebd9d74_gpu01p023456_sharedmmap_20260904/`，
  临时mmap为`.codex/tmp/prw_group_gain_credit_73task_cache_aebd9d74_gpu01_20260904/`。2026-09-04 14:12 live检查确认gpu01物理
  `0,2,3,4,5,6`均为`0MiB/0%`，物理1的他人任务约`1.8GB/60%`保持不动；gpu02卡均有他人常驻或重载任务。因此使用gpu01上述
  6卡、world-size6、`NCCL_P2P_DISABLE=1`与GPU-local NUMA。训练使用6卡期间不并发评测；自然完成后m50/m100可各用3卡并行做
  held5 correct-only strict250，总EMBER卡数仍为6。只允许相同authority/config/world topology exact resume，任何fresh root不覆盖。

  同次`/data1` user quota为`776813584/1084227584 KiB`，limit headroom约`293.17GiB`；实测mmap峰值`97.81GiB`，retained
  checkpoint/metrics预计低于`1GiB`，安全低于独立quota。目标output/cache root在launch前必须再次确认不存在。

  该formal已于2026-09-04 14:23 CST从上述detached authority实际启动，tmux为`ember_prw_group_gain_s100`；启动前再次同时live
  检查两节点、确认目标root/cache/exit均不存在，6个rank均已建立且未触碰gpu01物理1的他人进程。当前处于一次性model/cache冷启动；
  不把低利用率初始化阶段或rows2 profile loss当作科学结果。

- causal-prefix event filter已由`f6b58aac`接通并推送。实现只在`causal=True`时返回从真实首帧anchor开始的monotone forward
  posterior；完整/deployment视频仍逐行复用原hard first/final前向--后向posterior，参数与state dict均未改变。41项Writer与
  materialization相邻测试通过。gpu01物理0上的task1 demo5真实smoke完整消费51帧、2 probes、全部50 horizons和38 targets，
  functional/process loss为`.150360/.163691`；functional梯度到达Frame/Event/Composer
  `.002898/.002652/.187827`，process梯度到达Frame/Event/Predictor`.065089/.058663/.202458`，生成76 tensors与唯一rank16，
  峰值allocated/reserved约`27.35/33.98GB`。与前代相同的functional loss直接确认full deployment forward未漂移。

- Causal-event-filter formal launch contract：scientific implementation为`f6b58aac`，formal authority为包含本合同的下一clean
  pushed main。继续使用`configs/pi05_ecp_policy_response_writer_process_conditioned_v1.json`、73个gradient tasks、每update
  `9 meta + 3 target`、K1、component-init、correct-only cross-episode functional、positive causal process、preservation、
  full 50-horizon、真实native X/Y、rank12+4及唯一rank16。这里保留task采样比例只为与上一run作单变量匹配，不将其固定为后续要求。
  唯一科学变化是人工prefix不再hard-final；teacher、loss权重、prediction优化、Frame/Composer和完整视频posterior均不变。运行100
  optimizer steps并保存50/100；输出固定为
  `runs/outputs/pi05_ecp_policy_response_writer_causal_filter_73task_k1_component_s100_f6b58aac_gpu01p0156_sharedmmap_20260904/`，
  临时单份mmap为`.codex/tmp/prw_causal_filter_73task_cache_f6b58aac_gpu01_20260904/`。若launch时live状态仍允许，使用gpu01物理
  `0,1,5,6`、world-size4、`NCCL_P2P_DISABLE=1`与既有NUMA映射；状态漂移则只换安全设备、不改科学合同。macro50出现后可在EMBER
  总卡数不超过6时并行held5 correct-only strict250；macro100随后同合同评测。只允许相同authority/config/world topology exact
  resume，任何fresh root不得覆盖。

  2026-09-04存储预检：`/data1` user blocks为`776687164/1084227584 KiB`，limit headroom约`293.29GiB`；上一同构run的146份
  gradient-fit mmap实测`97.81GiB`、retained run仅`.08GiB`，本轮含日志/checkpoint预计峰值新增低于`100GiB`，明显低于独立quota
  余量。目标output与cache root在launch前必须再次确认不存在。

- process-conditioned optimizer50/100资格与两点held5闭环已经全部自然完成。clean detached authority为`f20a5299`，formal root为
  `runs/outputs/pi05_ecp_policy_response_writer_process_conditioned_73task_k1_component_s100_df1e8c6e_gpu01p0156_sharedmmap_20260904/`；
  100条metrics、两枚checkpoint、Panel-B、result/completion与信息墙完整，训练/总wall为`1702.73/2467.33s`。m50/m100
  correct-only strict250均为`37/250`，逐task分别`0/0/3/31/3`与`0/0/3/32/2`，breadth均`3/5`且Goal/Long为0；相邻
  `31 retained/6 gained/6 lost`。m100相对carrier43为`32/5/11`，所以优化修复没有转化成闭环增益，当前实例正式non-pass。

  深入诊断同时排除了“process没学到”和“预测头绕过视频”。m100在六task、fit+held共108个正确视频pair上的标准化Smooth-L1由
  zero `.065969`降到`.061689`，改善`6.49%`；zero-state predictor反而为`.067523`，完整状态在`100/108` pair上更好。
  但causal prefix把每个任意cutoff当前帧都硬锚到最终slot7，同一帧在完整视频中只有`15/108`仍属slot7，assignment平均重合仅
  `.136962`。target聚合Process梯度为functional的`3.73x`且cosine `-.0572`；Event子集为`3.16x`与`-.2037`。
  无权重反事实把prefix改为首锚定的monotone forward filter后，m100同帧assignment重合升到`.692016`，同时完整视频首尾anchor
  完全不变。下一唯一matched修正因此是：完整/deployment视频继续hard first+final；严格causal auxiliary的人工prefix只做forward
  filtering，不把cutoff伪装成真实视频终点。fixed teacher、loss权重、prediction优化、Frame/Composer、full 50-horizon、真实
  native X/Y、rank12+4、数据与task比例均不变。诊断细节见`findings.md`第140节；以下条目保留此前执行时间线。

- random legal-delta fresh资格已经从clean detached `eec024f8`完整结束。73-task optimizer50/100的held5 correct-only strict250
  均为`41/250`，逐task Long/Goal/Object/Spatial0/Spatial9均为`0/0/4/33/4`、breadth`3/5`；相邻为
  `33 retained/8 gained/8 lost`。m100相对前代consumer-boundary m100的`35/250`有`12 gained/6 lost`，但相对carrier43仍为
  `33/8/10`且Goal/Long为0，所以该实例稳定non-pass，不运行negative controls。gradient-task Panel-B fit/held由
  m50的`-.000010/+.000089`到m100的`+.000223/+.000202`，两个true-task-held在m50/m100均为负，后者为
  `-.000531/-.000872`；seen functional改善没有形成task-disjoint增益。

  机制诊断说明这不是random delta方向本身已失败，而是正式优化仍未学成它。m100实际predictor相对zero的总体MSE改善仅约`.35%`，
  delta1仍明显劣于zero；预测头m50到m100参数只移动约`.1%`。但冻结同一Process state、只训练同一readout时，直接预测标准化
  target在100步的train/同视频未见pair/同task未见video解释量为`.540/.302/.144`，250步为`.833/.389/.217`。多间隔
  state probe在delta4/8同视频解释量为`.278/.410`，跨task方向信息约`.043/.074`；因此Frame/Event state与head都有非零容量。
  当前正式normalizer却只由每fit视频一个随机pair估计，73 task的inverse-weight有效数仅`37.23`、top5占`24.20%`，个别task平均
  normalized process达到`4.89`；按实际16--17条pair估计则有效task约`64`。同时loss外除`sqrt(delta)`会衰减长间隔的prediction
  梯度，而原100步余弦schedule对head的累计步长只相当于容量诊断约5步。

  `df1e8c6e`的唯一matched修正已经接通：predictor直接输出标准化delta；normalizer用每fit视频8个确定性pair的冻结target-only Huber；纯辅助
  prediction参数组使用主Writer `20x`学习率，使累计步长与head-only 100步容量证据对齐。固定teacher、Frame/Event/Composer、
  主deployment forward、loss权重、完整50-horizon/native X/Y/rank4/唯一rank16均不变。34项Writer测试通过；task1真实smoke的
  functional/process loss为`.150360/.161288`，对应functional Frame/Event/Composer梯度`.002898/.002652/.187827`，process
  Frame/Event/Predictor梯度`.065754/.057292/.192145`，峰值allocated/reserved约`27.35/33.98GB`。两步shared profile又确认
  task1 normalizer`.075211`、学习率严格`20x`、全部梯度组非零，step约`3.67/3.48s`。该同规模optimizer50/100资格随后已从
  clean pushed detached authority完成；process prediction成立而闭环失败，并进一步定位到本节首条的causal event坐标错位。

- 已执行的Process-objective-conditioning formal launch contract：scientific implementation为`df1e8c6e`，formal authority为包含本合同的
  下一clean pushed main。配置固定`configs/pi05_ecp_policy_response_writer_process_conditioned_v1.json`，保持73 gradient tasks、
  每update显式`9 meta + 3 target`、K1、component-init、correct-only cross-episode functional、positive random-delta process、
  preservation、full 50-horizon、真实native X/Y、rank12+4及唯一rank16；唯一变化是direct standardized predictor、每fit视频8个
  target-only normalizer pairs与prediction-only `20x` LR。运行100 optimizer steps并保存50/100；输出固定为
  `runs/outputs/pi05_ecp_policy_response_writer_process_conditioned_73task_k1_component_s100_df1e8c6e_gpu01p0156_sharedmmap_20260904/`，
  node-local单份mmap固定为`.codex/tmp/prw_process_conditioned_73task_cache_df1e8c6e_gpu01_20260904/`。计划使用gpu01物理
  `0,1,5,6`、world-size4、`NCCL_P2P_DISABLE=1`及既有NUMA映射；launch前必须再次同时live检查gpu01/gpu02，若状态漂移则按同一
  科学合同更换安全卡。只允许相同commit/config/world topology exact resume；任何失败fresh root不覆盖。macro50出现后，在总EMBER
  占卡不超过6的前提下可用gpu02两卡并行held5 correct-only strict250；训练结束后macro100可使用至多4卡评测。checkpoint选择与
  negative controls仍按correct-only闭环、breadth与相邻稳定性决定。

  2026-09-04启动前存储检查：`/data1` user blocks为`776537756/1084227584 KiB`，limit headroom约`293.44GiB`；共享filesystem
  尚余`84TiB`。上一同构formal retained output仅`83MiB`，本轮临时mmap预计约`105GiB`、checkpoint与日志小于`1GiB`，峰值明显低于
  独立quota余量；目标output与cache root均确认不存在。

- 当前最前沿已经完成Process common--innovation consumer-boundary的真实裁决。clean detached `f33f2955`保持完整50-horizon、
  真实native X/Y、positive-only与唯一rank16，73-task shared optimizer50/100、两点Panel-B、物化和held5 correct-only strict250
  均完整结束；m50/m100仅为`40/35`，逐task Long/Goal/Object/Spatial0/Spatial9=`0/0/2/38/0`与`0/0/5/29/1`，breadth
  `2/5`与`3/5`，Goal/Long仍为0。两点间`28 retained/7 gained/12 lost`；m100相对carrier43也只有`30 retained/5 gained/13 lost`。
  gradient tasks的fit/held functional benefit由`.000285/.000181`升到`.000495/.000336`，两个true-task-held却从
  `-.000081/-.000432`恶化到`-.000630/-.000983`。因此这不是门槛太高或只差续训，consumer-boundary参数化正式non-pass，
  不运行negative controls。

  深入定位发现两个独立事实。第一，修正已让Process state的时间cosine从旧约`.9999`降到`.825--.987`并把跨task final query
  cosine降到`.99276`，所以动态确实进入主图；但最终mobile factor的有效参与秩中位仍只有`1.032`，rank坍缩已存在于signed
  selection weights而非物化之后。第二且更早，当前`L_process`实现把专家要求的随机`(t,delta>0)`错误收缩成固定相邻
  `future_offset=1`。六个gradient task的m100因果预测在36个cutoff上仅`5/36`优于零，process与functional共享梯度范数比约
  `1.6--2.8`而cosine接近0；固定teacher约`87.5%`矩阵能量虽落在ResponseTokenizer rowspace之外，但把teacher事后对齐只把
  单步跨task最优解释量提高到约`1--2%`，不足以单独修复。

  同一checkpoint的正样本多尺度反事实随后给出明确方向：保持38 owners、50 horizons与2 probes不变，within-video线性probe的
  最优尺度MSE解释量从delta1的`.0094`升至delta2/4/8的`.1382/.3268/.4718`；跨task双向均值从约`.0093`升至
  `.0173/.0416/.0752`。因此下一fresh只恢复专家原本的随机合法prefix/future interval，并以parameter-free encoding告知
  predictor相对delta；prediction与target同步除以`sqrt(delta)`，抵消实测target RMS近似按`sqrt(delta)`增长而造成的无关loss
  权重漂移。固定teacher、Process/Composer主图、bank、rank、数据与closed-loop合同全部不变。实现提交`38d51bab`已通过31项
  定向测试和task1真实full-horizon delta8 smoke：functional/process loss为`.150360/.142832`，Frame/Event/Composer functional
  gradient均非零，Frame/Event/Predictor process gradient均非零，38 targets、50-step horizon、rank4 materialization与唯一rank16
  输出保持完整；主functional loss、梯度及输出与前代smoke逐项一致。现在立即复用optimizer50/100短资格。以下条目保留此前执行与
  审计时间线，不覆盖本条当前状态。

- canonical集成目标为clean pushed `main`。上一轮full event-measure
  73-gradient-task资格已完成训练、m200/m400 Panel-B、物化与两次held5 correct-only strict250：闭环仅`30/32`，breadth
  `2/5`与`3/5`且Goal/Long均为0，m400仍显著低于carrier43。根因审计进一步定位到Composer最早query融合处：约`67`范数的
  Process common淹没约`1`范数的rank token，使名义rank4在m200/m400都退化为近rank1；冻结反事实表明分别归一rank/shared
  context可恢复rank区分与部分有效factor谱。该最小修正已集成推送。当前gpu01物理`0,1,5,6`正从clean detached `3e589695`
  运行fresh 73-task m200/m400 shared资格；已完成105.02GB单份mmap cache和NUMA审计，稳定step约`15--17s`且四rank
  wall几乎完全对齐，当前没有读取negative controls。task1/task93正控的前两次启动都在run contract、checkpoint、capture和
  optimizer step前退出：先后暴露factorial配置缺少正控声明，以及task-local合同仍假设旧单一panel config。前者由`8bdd9595`
  补齐，后者由`89ca865d`把实际resolved task panel封存进v2合同；模型、数据、loss与训练步数均未改变，27项Writer测试及
  task-local配置全字段预检通过。两条失败启动不构成科学状态。task1/task93从clean pushed detached `ef00f446`在gpu02
  物理`0/1`完成正控：macro70 fit/held recovery分别为`.2224/.1153`与`.3047/.3115`，macro110为
  `.3283/.2282`与`.3570/.3223`，四个checkpoint的三条视频均自发优于carrier；禁止路径backward全为0。相较前版，task1
  macro110改善而task93略降，所以修正确认保留并可改善task-local容量，但不是跨task统一增益。总wall由旧版约`2016/3076s`
  降到`813/1062s`。shared m200现已完成唯一rank16物化与held5 correct-only strict250：`45/250`，逐task
  Long/Goal/Object/Spatial0/Spatial9=`0/0/4/38/3`、breadth`3/5`。相对旧raw-query m200的`30/250`为
  `23 retained/22 gained/7 lost`、paired exact `p=.00813`，确认修正具有真实闭环因果收益；但相对carrier `43/250`
  仅为`33/12/10`、`p=.83181`，Goal/Long仍为0，因此尚未通过shared路线。m400训练已自然完成且两枚checkpoint完整；
  correct-only strict250反而降到`35/250`，逐task为`0/0/1/33/1`、breadth`3/5`，相对m200为
  `30 retained/5 gained/15 lost`、paired exact `p=.04139`。Panel-B的10个gradient tasks held benefit均值虽由
  `.000429`升到`.000526`，两个零梯度true-task-held却由`-.000438`恶化到`-.002605`、全视频通过数由`1/2`降到`0/2`。
  因而已排除“只差更多训练”：当前Writer在继续改善见过task的离线functional时，task-disjoint映射与闭环同时退化。m200冻结、
  correct-only的Object18/Goal25/Long36诊断均已完成：query从初始约`.83`到第二Composer block仍保留约`.47--.56`的
  rank centered/mean RMS，q/v参与秩中位约`1.08--1.51/1.13--1.49`，说明最早rank坍缩确已修复；但action-out参与秩仍仅
  `1.00--1.04`，其平均绝对scale约`1.13--1.22e-5`，比q约`3.69--3.71e-4`小约30倍，且cap恒为1。随后在六个
  gradient-authorized正确Panel-A任务上做冻结VJP、零optimizer更新诊断：action-out scale-head梯度范数均值约`.0921`，与q的
  `.1024`同量级，4/6任务局部希望沿当前方向放大，但跨task聚合/逐task范数和仅`.247`，且共享scale head的q--v、q--action-in
  梯度cosine为`-.733/-.879`。m400同构VJP进一步显示action-out已有`5/6`任务希望缩小当前方向，但它的实际幅度仍随共享head
  增长；四family聚合梯度之和只保留各family范数和的`.480`，独立family readout的一阶下降信号平方和为共享head的`2.00x`。
  另外，m400跨Object/Goal/Long的matching target/rank query cosine仍为`.99925--.99952`：Process common范数约`70--74`，
  language约`11.35`，而`owner_bias`内部又是约`1`范数的owner被约`11.7`范数的family淹没；同family q/v owner bias的平均
  cosine为`.9925/.9932`。冻结、零梯度的typed-source反事实把四family最终跨task centered/mean RMS由
  `.0178/.0224/.0188/.0224`提高到`.0301/.0351/.0350/.0377`；单纯把source追加到event memory没有同等作用。
  owner与family也独立pre-norm后，q/v的18个target在两个标准block后的区分比又从`.0371/.0459`提高到
  `.1835/.2049`，三条Object/Goal/Long视频一致复现。
  因此下一fresh修正不再只看action-out幅度，而统一处理Composer边界的typed ownership：独立pre-norm并方差平衡
  rank/owner/family/common/language，末端relative rank gain按family拥有参数；不改bank reader、loss、rank、scale上限或数据。
- 400步训练的模型计算与Panel-B已经完成，但原进程在删除NFS上的105GB mmap cache时先unlink仍被其它rank映射的文件，产生
  `.nfs*`句柄并使rank0 `rmtree`报`ENOTEMPTY`，其余rank随后在barrier超时；故原root没有伪造`result/completion`。
  两checkpoint的Panel-B已从clean detached `3e589695`以零梯度、零optimizer、零wrong reads独立恢复到
  `runs/analysis/pi05_ecp_policy_response_writer_rank_balance_s400_panelb_recovery_3e589695_gpu01p0156_20260904.json`。
  cleanup现改为所有rank先清空mmap tensor并`gc.collect`、barrier后再由rank0删除，回归测试同时验证映射已释放和双barrier顺序；
  29项Writer测试通过，删除失败也会同步转成明确错误而不再留下其它rank超时。该故障只影响收尾文件，不改变m200/m400
  checkpoint或任何科学结论。
- typed-boundary修正已以clean pushed `main@682f7ecfcc233d486d7a4ce41e776562f2e6890e`封存；实现分支已删除。它把上述
  根因收敛为两个同属Composer边界所有权的改动：入口对
  rank、owner、family、Process common与language逐source做parameter-free pre-norm并按残差组保持方差；末端把同一个
  `Linear(width,1)`改为`Linear(width,4)`，每个target只选择其native family row产生query-conditioned relative rank gain。
  它没有task表、anchor或第二Writer，仍只能缩放当前视频真实X/Y signed pooling所得方向。30项Writer测试和新旧config互斥预检
  通过。gpu01物理0上的真实task1 smoke已完整消费51帧、38 targets与full 50-horizon，生成76 tensors/唯一rank16；functional梯度
  到达Frame/Event/Composer/relation，causal梯度到达Frame/Event/predictor，峰值allocated/reserved为`27.35/33.98GB`。
  新资格配置使用圆整的task-local 50/100与shared 100/200 checkpoints；shared m200与前一m200保持约33次/task的matched暴露，
  不在架构未证明时再次先付出400步。
- 2026-09-04 03:08--03:09 CST从同一clean detached `682f7ecf`并行启动三条fresh formal：gpu01物理`0,1,5,6`、
  world-size4运行73-task shared optimizer100/200，root为
  `runs/outputs/pi05_ecp_policy_response_writer_typed_boundary_73task_k1_component_s200_682f7ecf_gpu01p0156_sharedmmap_20260904/`；
  gpu02物理`0/1`分别运行task1/task93的task-local optimizer50/100。shared使用node-local单份mmap、零额外replica、deferred
  NCCL、`NCCL_P2P_DISABLE=1`及rank-local NUMA；三者均为component-init、K1、full 50-horizon、positive-only、唯一rank16。
  launch前gpu01四卡完全空闲；gpu02两卡各有他人约`0.15--0.19GB`、0% util的轻量常驻进程，已测峰值仍留约12GB余量。
  `/data1`为`775964240/1073741824` blocks，约105GB共享cache与小型checkpoint/metrics不越过独立quota。两条task-local
  已自然完成且exit0：task1 macro50/100的fit/held recovery为`.1732/.1633`与`.2419/.2581`，task93为
  `.3464/.3325`与`.3820/.3643`；四点各自三条fit/held视频均自发优于carrier，held、wrong与Panel-B backward均为0，输出保持
  唯一rank16。task93以少10步同时高于前一rank-balanced macro110的`.3570/.3223`；task1 fit较前一`.3283`低，但held高于
  前一`.2282`。因此typed ownership保留了task-local容量并改善held-video保持，尚不构成shared跨task性能结论。shared已完成
  98GiB/146-video单份mmap cache并进入稳定训练，前20步平均`17.17s`，与前一实现`16.82s`近似，四rank wall对齐。
- shared macro100已完整保存并在训练不停顿时物化held5 correct K1：五个task各生成一套76-tensor完整rank16，Writer调用恰为
  每task一次，held action/state/reward、wrong与negative controls读取均为0。2026-09-04 03:49 CST在gpu02物理`0/1`启动
  6-worker、每task50初态的correct-only strict250，root为
  `runs/outputs/pi05_ecp_policy_response_writer_typed_boundary_m100_held5_correct_k1_strict250_682f7ecf_gpu02p01_r3_20260904/`；
  queue采用long-first动态调度，结果不回流正在运行的macro200训练。该评测已自然完成并exit0，22/22 shards、250 rows与
  run summary完整：总分`39/250`，Long/Goal/Object/Spatial0/Spatial9=`0/0/3/33/3`、breadth`3/5`。相对stable carrier
  `43/250`为`32 retained/7 gained/11 lost`、churn`18`；相对前一rank-balanced m200 `45/250`为
  `30/9/15`。所有250个task/state/env/policy seed严格配对，按episode实际长度截断的policy-noise序列共同前缀也逐条一致。
  所以m100改变了真实闭环行为并保留基本breadth，但净低于carrier与前一m200，不是通过性结果。shared训练随后完整保存
  macro200；live检查gpu01/gpu02后，gpu01四卡继续零梯度Panel-B，gpu02物理`0/1`仅与他人约`0.15--0.19GB`、0% util进程
  安全共驻，并于04:18 CST启动macro200的同合同物化与strict250，未超过6张EMBER GPU。
- typed-boundary shared run已自然exit0：200条metrics、macro100/200 checkpoint、`result.json`与`completion.json`完整，训练/
  Panel-B/总wall为`3348.04/448.99/4072.12s`；98GiB mmap cache在所有rank释放映射后已成功删除，验证NFS cleanup修复。
  Panel-B从m100到m200时，10个gradient tasks的fit/held benefit均值由`.0001963/-.0000465`改善到
  `.0005009/.0002095`，全视频为正由`4/10`到`5/10`；但两个true-task-held的fit/held仍为
  `-.0006155/-.0008589`与`-.0008069/-.0007323`，全视频为正从`1/2`降到`0/2`。所有held、wrong与Panel-B backward
  均为0。该证据说明typed ownership改善见过task的shared functional拟合，却尚未解决task-disjoint映射；macro200闭环正在运行。
- owner于2026-09-02完成最后审查，正式确认Policy-Response Event-to-Factor Writer并要求立即推进。系统goal已重新建立并保持
  active，不设置token或阶段工期预算。
- 当前唯一active design为`docs/policy_response_event_to_factor_writer_design.md`。它保留PI0.5
  layer x horizon x probe原生响应、G2 ordered events、当前视频真实X/Y、G1 signed pooling、rank4和唯一rank16物化；主要learned
  模块收敛为可复制的Video Process Encoder与Native Factor Composer。
- 完整horizon 12-task短资格已从clean detached `e7278f1b22176b53025166dc5015a4463b819ecd`自然完成110步。macro70/110的
  10个gradient task fit/held benefit均为正，且均有`8/10` task全部视频优于carrier；但两个零梯度true-task-held在两点都没有
  全视频优于carrier，macro110 fit/held benefit均值为`-.000131/-.000135`。对应held5 correct-only strict250相邻结果均为
  `35/250`：macro70逐task Long/Goal/Object/Spatial0/Spatial9=`0/0/1/32/2`、breadth`3/5`，macro110为
  `0/0/0/33/2`、breadth`2/5`。两点稳定低于carrier `43/250`且Goal/Long均为0，因此这是当前shared parameterization的正式
  non-pass；不追加训练、不恢复73-task长跑，也不对失败checkpoint运行negative controls。
- 该non-pass后对照专家输入合同与实际数据流，确认Process产出的`assignment=alpha(e,t,m)`从未被Composer读取，四类relation在
  signed candidate scoring前已混为一个`frame_innovation(t,j)`。这不是单纯数据覆盖不足：零梯度target task74与已训练
  task72/73/75具有相同“black bowl -> plate”动作/对象/goal，只改变初始scene relation，仍未迁移。当时的matched改动先让
  event innovation与soft assignment以显式relation轴进入exact signed pooling；它的完整non-pass及后续event-measure修正见下一项。
- 首个matched relation-summed修正`I[t,m,j]=sum_e alpha[e,t,m]D[e,j]`已完成全部裁决。macro70/110 Panel-B gradient
  fit/held benefit分别为`.000995/.001146`与`.001305/.001007`，但两个true-task-held在两点都为负；held5 correct-only
  strict250为`42/34`，breadth均`3/5`且Goal/Long为0，macro70到110为`16 lost/8 gained`。因此不追加训练或controls。
  复核专家§7.1后确认该版本仍在非线性score前消掉event轴，soft assignment也没有成为base candidate measure。下一唯一科学修正是
  event x relation候选：`log alpha(e,t,m)`直接进入base log-mass，未求和`D(e,j)`产生bias-free动态logit；其余科学变量不变。
  该实现的显式enumeration输出/梯度等价、归一base measure、assignment消费、static-repeat及融合pooling测试已通过。真实task93
  formal-rows16两步profile为`8.934/8.205s`，最大allocated/reserved为`39944498688/46433042432 bytes`；第二步Frame、Event、
  Process、Composer与relation参数梯度均finite nonzero，Panel-B零反向、输出仍为唯一rank16。
- event-measure 12-task短资格已于2026-09-03 19:08 CST从clean pushed detached
  `a049f61e17ad9e5eae55b67b8de7be4aa686bfc9`在gpu01物理`0,1,5,6`、world-size4 fresh启动，root为
  `runs/outputs/pi05_ecp_policy_response_writer_event_measure_12task_k1_component_s110_a049f61e_gpu01p0156_cache8g_20260903/`。
  它保持K1、component-init、full 50-horizon、12-task正样本合同、110步与macro70/110不变，显式提供`8GiB`只读evidence-cache
  replica预算。execution plan只依据冻结cache大小、预定task/video schedule与计算量，未使用outcome；预测总成本由owner-only的
  `14047`降到`10131`，理想无所有权限制为`10098`，尾部从`157`降到理想`104`。前两步真实wall为`10.556/10.137s`，而同一实现
  漏传replica预算的已中止基线首步为`13.637s`；后者只运行6步、无checkpoint，保存在
  `runs/outputs/pi05_ecp_policy_response_writer_event_measure_12task_k1_component_s110_a049f61e_gpu01p0156_budget0_interrupted_m6_20260903/`
  作执行诊断，不形成科学结论。
- 该event-measure训练与两checkpoint Panel-B现已自然完成，optimizer110、completion/result、两枚checkpoint及110条metrics完整；
  train/evaluation/total为`1109.73/576.16/1725.81s`。macro70的10个gradient task fit/held benefit为
  `.001329/.001114`、recovery为`.11773/.08567`、`9/10` task全部视频优于carrier；macro110分别为
  `.001617/.001701`、`.13651/.13193`与`9/10`。但两个零梯度true-task-held在两点均为`0/2`：macro70 fit/held benefit
  `-.001575/-.001798`，macro110为`-.002420/-.001998`。held5 correct-only strict250也已全部完成：macro70为
  `40/250`，逐task Long/Goal/Object/Spatial0/Spatial9=`0/0/2/36/2`；macro110为`42/250`，逐task
  `0/0/2/37/3`；breadth均为`3/5`。两点间retained/gained/lost=`33/9/7`、Jaccard `.67347`、paired exact
  `p=.80362`；macro110相对carrier `43/250`为`35/7/8`、`p=1.0`。因此event-resolved measure增强了已训练task内
  的functional方向，但没有解决task-disjoint泛化、carrier保持或Goal/Long breadth；不追加同构训练、mixed-K、
  fully-random或negative controls。`70/110`仅表示10步warmup后的effective `60/100`历史可比取点，不是理论推导的
  最优训练时长。
- full训练吞吐优化已完成可复现实测：同一4卡、10-step、6-task旧资格schedule的基线为`34.394s/step`；选择性CPU cache复制后为
  `26.306s`，融合完整bank attention与pooling后为`8.699s`，最终加入bounded streaming blocks、整视频frame融合、output-group
  归约与functional microbatch 4后为`4.054s/step`，总提速`8.48x`。最终10步最小/最大为`3.436/4.872s`，四卡有效task计算占
  总device wall约`78.0%`；剩余主要是最长task不可切分尾部。所有优化完整保留50 horizon与exact online softmax，不改变task
  group、权重、K或optimizer cadence；新sampler也不再把当次`3 meta + 3 target`误当owner固定要求。
- 进一步exact scorer contraction把同卡task93 rows2从`7.43/6.40s`降到`4.78/4.10s`，约快`36%`。microbatch4仅再快
  `.8%`且reserved升到约`46.76GB`，CPU saved-tensor offload显存减半却让step慢到`24.79/16.41s`，梯度all-reduce打包收益
  小于`.04%`，三者均不保留。Evaluator `3 replicas x 8 envs`的50-row rollout-only为`.07056 rows/s`；`4x8` OOM，
  `2x16`只有`.06567 rows/s`且SM利用率更低，因此继续保留`3x8`。
- 旧动态放置把一条functional policy row与一帧full bank等价计价，低估了冻结policy VJP。对relation-summed 110步逐步回放显示，
  用`4 * functional_rows + sampled_full_bank_frames`作纯执行成本后，相对真实最优makespan的平均比值由`1.0365`降到`1.0135`，
  p95由`1.0868`降到`1.0420`，44步改善、0步变差；预计总wall约再降`2.30%`。对应实现不假定每步6 task或固定meta/target比例，
  完整Writer定向测试`22 passed`，已集成并推送；当前event-measure 110步在既定replica eligibility下新旧模型给出相同分配，因此不影响
  正在运行的frozen authority，主要改善未来不同task batch、role比例、K与cache拓扑。
- 当前event-measure完整110步的训练wall均值/中位/p95为`10.056/10.044/10.643s`，对比遗漏replica预算的前6步基线均值
  `13.624s`快`26.2%`；真实rank load的`max/mean`由`1.619`降到`1.216`，平均最长--最短gap由`8.01s`降到`4.19s`。
  gpu01物理`0/1/5/6`稳定训练段平均SM为`88.2/88.5/89.4/90.2%`，活跃采样比例均超过`99.6%`。剩余gap主要是task93等单task
  不可切分下界，不是某些rank长期空闲。
- factorial coverage正式root为
  `runs/analysis/pi05_ecp_policy_response_writer_factorial_coverage_v1_20260903/`，读取范围仅限固定split、language、metadata与
  人工审计protocol groups，未读取视频、action/state/reward、模型输出或评测结果。55个meta-fit与18个target-fit全部作为
  gradient tasks，task2/74继续true-task-held。7组同语言跨场景组合中5组有train pair、4组有held桥；
  same-language/same-procedure/order-relation三类contrast分别有`5/9/5`组train pair及`3/7/3`组held桥。task2有同语言peer23，
  task74有同verb/object/goal而scene relation不同的72/73/75；held Spatial/Object/Long也都有component重组来源。唯一明确缺口是
  held Goal task25的`push` procedure没有Writer-gradient peer。因此数据足以让扩大mapping成为可识别实验，但metadata不证明
  video-dependent最优adapter必然可学。
- 新资格配置为`configs/pi05_ecp_policy_response_writer_factorial_v1.json`：保持full 50-horizon、K1、component-init、
  event-measure、positive-only与唯一rank16不变，只扩大到全部73个eligible gradient tasks。每update的`9 meta + 3 target`
  按55:18池大小近似task等权，只是本实验配置而非owner或未来固定比例；总batch为12也不是长期合同。optimizer step200/400
  分别对应post-warmup effective190/390，并使每task获得约`32--34/65--67`次暴露；400点对齐旧12-task 3-of-5 x 110步的
  约66次/task，因此检查点来自暴露量而不是沿用J2的70/110历史编号。当前可用单节点四卡profile结果与正式短跑合同见下一项。
- 73-task/12-task-per-update两步真实profile已从clean detached `248d3efa`在gpu01物理`0,1,5,6`完成，root为
  `runs/outputs/pi05_ecp_policy_response_writer_factorial_73task_k1_component_profile2_248d3efa_gpu01p0156_cache8g_20260903/`。
  当时gpu01其余2/3/4由他人约98%利用、gpu02没有额外两张能安全容纳full峰值的卡，故不跨节点拼卡或干扰他人。两步
  profile rows2 wall为`10.597/9.260s`，每步四rank均恰好3 tasks；第一步预测cost为`123/101/113/114`，第二步为
  `113/118/131/122`。第二步Frame/Event/Process/Composer/relation梯度均finite nonzero；峰值allocated/reserved为
  `27.37/36.94GB`。73 tasks的146个唯一fit videos冻结evidence约`105.02GB`；8GiB预算只选择实际有收益的3个replicas、
  额外`3.11GB`，没有为耗尽预算盲目复制。训练两步`19.97s`，cache/normalizer准备加训练为`150.20s`；此外约4分钟是
  冻结模型与数据冷加载，正式运行紧接同节点可复用页缓存。该profile使用rows2只证明执行图与调度，正式rows16预计约
  2--2.5小时/400步而非十小时；下一步立即从clean pushed detached authority正式启动。
- factorial full K1 component-init正式短资格已于2026-09-03 20:49 CST从clean pushed detached
  `5534cb140b90ac20e9143dd20a7ed8e11c539f19`在gpu01物理`0,1,5,6`、world-size4启动，tmux为
  `ember_prw_factorial_s400`，root为
  `runs/outputs/pi05_ecp_policy_response_writer_factorial_73task_k1_component_s400_5534cb14_gpu01p0156_cache8g_20260903/`。
  launch前两节点同步live检查仍确认所选四卡为0MiB/0% util；gpu01物理2/3/4为他人98--100%任务，gpu02其它卡没有安全
  full峰值余量，故未跨节点凑6。`/data1` quota blocks为`774957644/1073741824KiB`、limit `1084227584KiB`；root fresh，
  预计新增远低于剩余额度。命令固定`NCCL_P2P_DISABLE=1`、GPU-local NUMA、8GiB只读cache replica预算、55 meta + 18 target
  gradient tasks、task2/74零梯度、rows16、10 warmup + 390 effective、optimizer200/400 checkpoints；没有wrong/shuffle/reverse
  读取或loss。训练达到m200后若gpu02仍有两张安全卡，将在不超过总6卡且不影响训练的前提下并行物化和运行m200 held5
  correct-only strict250，以更早获得阶段科学结果；m400仍由同一fresh run自然完成。
- 性能线在formal冻结后继续独立推进。73-task的training-cache ownership会让12个Panel-B任务落成每rank `2/4/5/1`，因此
  `e74b653961e6d7bf088348f88d95eeba95b74921`新增outcome-independent evaluation cost：每task三条完整视频的frame数加固定
  `4 * functional_rows * visits`成本，再用既有LPT均衡；本轮结构稳定得到`3/3/3/3`，且result显式记录
  `evaluation_task_ownership`。23项Writer测试通过。该提交晚于当前frozen formal，不改变其训练或结果，只服务后续运行。
- gpu02物理0/1在总EMBER卡数恰为6时并行完成task93 Panel-B microbatch2/8真实profile，roots分别为
  `runs/outputs/pi05_ecp_policy_response_writer_panelb_mb2_task93_profile2_e74b6539_gpu02p0_20260903/`与
  `runs/outputs/pi05_ecp_policy_response_writer_panelb_mb8_task93_profile2_e74b6539_gpu02p1_20260903/`。三条视频各一次16-row
  functional evaluation分别用`11.692/11.577s`，microbatch8只快`.98%`；不值得增加峰值风险，故保留2且不继续盲测16。
  两profile均exit0，使用的两张卡原有约148--186MiB、0% util进程未受干扰。
- node-local单份safetensors mmap cache已实现并完成严格匹配实测。它只保存deployment-visible、action-hidden frozen video
  evidence，每个task/video一份原子文件；各local rank mmap同一物理页，因此全部task都可动态放置而不按rank复制105.02GB。
  shared、private 0GiB和private 8GiB三条两卡profile均来自同一clean detached candidate `a2e40700`，使用完全相同7-step、
  84-task/video/Panel/RNG/weight schedule，roots分别为
  `runs/outputs/pi05_ecp_policy_response_writer_factorial_73task_k1_component_profile7_sharedmmap_a2e40700_gpu02p01_20260903/`、
  `runs/outputs/pi05_ecp_policy_response_writer_factorial_73task_k1_component_profile7_private_a2e40700_gpu02p01_20260903/`与
  `runs/outputs/pi05_ecp_policy_response_writer_factorial_73task_k1_component_profile7_private8g_a2e40700_gpu02p01_20260903/`。
  train wall为`124.870/150.039/130.139s`，即shared为`17.811s/step`，相对当前8GiB方案平均快`4.05%`且最坏step快`24.4%`；
  rank实际load gap从`3.122s`降到`.338s`，峰值allocated/reserved仍约`39.98/46.96GB`。当前四卡formal rows16前126步的逐task
  timing反事实重排给出`23.441 -> 17.955s`，计入实测mmap开销后约`18.4s`，预计长期收益约`21%`。缓存成功后自动删除；首次
  105GB capture/build与private冷启动处于同一量级，不能用复用cache的`total_seconds`冒充端到端提速。完整Writer测试固定输入RNG后
  `25 passed`。后续fresh同节点多卡训练以shared mmap为canonical；当前`5534cb14` frozen formal不中途改拓扑。
- 最后审查已把causal process auxiliary严格prefix-only、预测target冻结及task1/task93 Composer容量正控写入合同。owner于
  2026-09-03进一步明确：full是唯一active representation，50-step horizon必须完整保留到task/relation-conditioned learned read；
  coarse/final-layer horizon mean及等价无条件平滑均不得继续用于训练、选择、初始化或部署。
- 收到该边界后，已立即停止正在运行的corrected coarse任务：73-task shared停在step121，task1/task93 task-local分别停在
  step47/29；三者均未完成预注册checkpoint评估，不形成科学裁决，保留小型root只记录主动中止。远端精确训练进程确认消失，gpu01
  物理2/4/5/6回落到`15/15/98/15MiB`，gpu02物理4/6回落到既有低占用`6020/4751MiB`。
- corrected full合同随即从同一clean pushed detached authority fresh启动：73-task/K1/component-init shared使用gpu01物理2/4/5/6、
  world-size4，task1/task93 full task-local分别安全共驻gpu02物理4/6，总EMBER物理卡为6。三条命令都显式锁定
  `--representation full`；输出root分别为
  `runs/outputs/pi05_ecp_policy_response_writer_corrected_73task_k1_component_full_s1210_7d435ea3_gpu01p2456_20260903/`、
  `runs/outputs/pi05_ecp_policy_response_writer_corrected_tasklocal_task1_full_s110_7d435ea3_v1_gpu02p4_20260903/`和
  `runs/outputs/pi05_ecp_policy_response_writer_corrected_tasklocal_task93_full_s110_7d435ea3_v1_gpu02p6_20260903/`。launch前双节点live
  检查确认所选gpu01四卡全空闲，gpu02两卡只有低util既有进程；`/data1` quota为
  `774455720/1073741824KiB`、limit `1084227584KiB`，三个root均为fresh。
- 唯一`codex/policy-response-full-only`实现worktree已从最新clean pushed main建立：两个active Writer config只声明`full`，训练CLI、
  shared checkpoint物化与static adapter provenance均拒绝coarse，`ResponseTokenizer`删除horizon-mean forward。仅保留不参与任何forward的
  旧`coarse_embedding`参数以维持当前full formal的RNG/state-dict兼容，待本轮checkpoint生命周期结束后再删除；定向Writer与static
  adapter测试为`16 passed`。
- corrected full task1正控已自然完成110步与macro70/110正式Panel-B；result/completion与launcher exit0完整。macro70的两条fit平均
  recovery为`.207146`、未训练held video24为`.169391`，macro110为`.223986/.157630`；两个checkpoint的三条视频全部自发优于
  carrier，Panel-B backward为0，source policy冻结且输出仍为唯一完整rank16。相对旧full正控幅度有所减弱、held后段回落，但明确保留
  full task-local跨视频功能容量；task93与73-task shared继续运行，不以task1单点裁决shared或closed loop。
- corrected full task93正控也已自然完成，result/completion与launcher exit0完整。macro70 fit/held recovery为
  `.417139/.409146`，macro110为`.425338/.418759`；两个checkpoint的全部三条视频均优于carrier，held/Panel-B backward为0，
  source policy冻结且仍为唯一rank16。相对旧full task93的`.337207/.300885`与`.346604/.280724`明显提高；结合task1，修正并非
  对所有任务统一缩小update，而是在完整horizon路径上保留并改善部分current-bank有效方向。两条task-local进程已结束并释放gpu02
  物理4/6；当前不以不匹配world topology的额外arm填卡。
- 对full路径继续逐层审计时发现，Process前端虽完整保留`50 x 8` response tokens，但Composer的辅助native-bank context仍通过
  `.mean(2)`提前平均了50-step horizon；最终signed pooling虽逐candidate保留horizon，该context read仍违反owner与active design的
  full-only合同。因此当时的73-task full formal在optimizer step156/effective146主动中止，未到macro610且不形成科学裁决。修复删除
  两处horizon mean，让process-conditioned rank query对完整frame/probe/horizon/bank-type keys做数学等价的chunked online-softmax
  cross-attention；output groups只作保持全部轴的向量化kernel融合，activation checkpoint避免百万token连续副本。定向Writer/static
  adapter测试为`18 passed`。task93最长fit demo3含87个采样帧，真实smoke的38-target forward、functional/process backward与唯一
  rank16物化全部通过，峰值allocated/reserved为`42.42/47.14GB`；两步shared profile为`33.29/28.54s`，全部梯度finite nonzero。
  下一步从该clean pushed commit fresh重启同一73-task/K1/component-init full formal。
- full-horizon 73-task formal已于2026-09-03 13:29 CST从clean pushed detached
  `e7278f1b22176b53025166dc5015a4463b819ecd`在gpu01物理2/4/5/6、world-size4 fresh启动；tmux为
  `ember_prw_full_horizon_scale73`，root为
  `runs/outputs/pi05_ecp_policy_response_writer_full_horizon_73task_k1_component_s1210_e7278f1b_gpu01p2456_20260903/`。
  launch前四卡分别为`15/15/98/15MiB`且util全0，gpu01 available host memory约`238.1GiB`；gpu02没有能容纳最长视频约
  `42.42GB allocated`且更合适的四卡集合，故不与已有4--31GB进程冒险共驻。`/data1` quota blocks为
  `774540872/1073741824KiB`、limit `1084227584KiB`，root为fresh且参考旧run保守新增小于`2GB`。模型、数据、carrier、config均存在；
  固定K1、component-init、10 warmup + 1200 effective、macro610/1210、positive-only loss、`NCCL_P2P_DISABLE=1`及完整horizon
  implementation。exact command为：
  `cd /data1/user/ymdai/projects/EMBER-worktrees/policy-response-writer-full-horizon-formal-e7278f1b && NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=2,4,5,6 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /data1/user/ymdai/projects/EMBER/.venv/bin/torchrun --standalone --nproc_per_node=4 scripts/train_ecp_policy_response_writer.py --config configs/pi05_ecp_policy_response_writer_scale_v1.json --asset-root /data1/user/ymdai/projects/EMBER --data-root /data1/user/ymdai/projects/EMBER/data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a --output-dir /data1/user/ymdai/projects/EMBER/runs/outputs/pi05_ecp_policy_response_writer_full_horizon_73task_k1_component_s1210_e7278f1b_gpu01p2456_20260903 --phase shared --representation full --initialization component --mode formal`。
  owner指出架构尚未用闭环证据证明，不应直接付出约10小时的扩展训练；因此该运行在optimizer25/effective15安全停止，四个worker
  全部退出且gpu01物理2/4/5/6回落到`7/6/7/6MiB`。运行未到macro，只作owner-directed early interruption记录，不作科学结论。
- 同一full-horizon实现改为先做12-task x 110-step短资格实验：2026-09-03 13:50 CST从同一clean detached `e7278f1b`
  在gpu01物理2/4/5/6、world-size4 fresh启动，tmux为`ember_prw_full_horizon_12task`，root为
  `runs/outputs/pi05_ecp_policy_response_writer_full_horizon_12task_k1_component_s110_e7278f1b_gpu01p2456_20260903/`。该设置保持完整50-step
  horizon、native X/Y、positive-only loss、component-init与唯一rank16不变，只把gradient/panel范围缩至已有可直接对比基线的
  5 meta + 5 target + 2 true-held，并把更新缩至10 warmup + 100 effective、macro70/110。launch前双节点live检查确认所选四卡全空；
  gpu02无另一组四张能安全容纳最长视频`42.42GB allocated`的卡，不跨节点拼碎片。`/data1` quota blocks为
  `774541080/1073741824KiB`、limit `1084227584KiB`，root为fresh。旧合同的直接可比结果为full macro70/110
  `33/31`、carrier `43/250`；本macro70一出立即物化并运行held5 correct-only strict250，只有真实闭环增量才恢复73-task长跑。
- PNBTT E1及其single/family chart、两次spectrum、full-rank16和gate-aligned necessity均已完成并稳定`non_pass`。PNBTT、
  EBSRI、Program-through-bank和旧summary/gate/anchor均不是active fallback，历史证据与formal artifacts继续保留。
- 已从clean pushed `main@194b91b2ae34efcb042a6c838973ba5d57ceda55`建立唯一
  `codex/policy-response-writer`实现worktree。Frozen Capture、repeatable Frame/Event blocks、Current-Video Native Factor Composer、
  task-local functional runtime、formal checkpoint/resume与唯一rank16 materialization已经接通；旧PNBTT/J2 runtime只复用数据、
  functional与checkpoint基础设施，不是active fallback。
- 新Composer沿用G1已证明的非对称启动：native A方向在初始化存在，`tanh(scale_head)`令有效B与mobile update严格为零；真实功能梯度
  第一步只打开scale，第二步再进入Frame/Event/Composer，避免随机新Writer在训练前破坏carrier。相同projected native keys在context
  read与signed pooling间复用，task-local阶段缓存冻结Process输出，不改变任何科学变量。
- 相关定向测试为`23 passed`；完整`tests/ecp`排除既有失败后为`141 passed`，唯一未通过项
  `test_shared_compiler_mapping.py::test_mapping_credit_is_set_valued_family_balanced_and_scale_stopped`在未修改的canonical main上同样失败，
  是退役SharedCompiler测试与当前`TangentTransportResult`字段不一致的既有问题，不由本实现引入。
- task1真实smoke保存在disposable
  `.codex/tmp/policy_response_smoke_zero_scale_task1_20260902/`：51帧、19 layer x 50 horizon x 2 probe与38-target X/Y一次捕获，
  初始A非零/B严格零；打开scale后functional梯度到达Frame `.002874`、Event `.003822`、Composer `.237721`，prefix-only frozen-target
  process梯度到达Frame/Event/predictor，76 tensors、rank12+4到唯一rank16全部通过，峰值allocated约`30.59GB`。
- task1/task93两步task-local profile只作工程证据。step1均只有scale-head梯度，step2 input/output branches与task query均有非零梯度；
  task93三个correct Panel-B视频两步后均自发略优于carrier。去重后task93 step耗时由`34.80/34.15s`降至`28.47/26.24s`，
  allocated峰值由`39.63GB`降至`32.89GB`、reserved由`47.30GB`降至`40.22GB`。该实现随后已提交、合并并推送，并从clean
  detached authority并行运行task1/task93的110步formal正控及step70/110只读Panel-B。
- 上述两条正式正控现已完整结束。task1 step70/110的Panel-B fit recovery为`.260876/.276421`、held-video为
  `.207341/.244598`；task93 fit为`.337207/.346604`、held-video为`.300885/.280724`。两task、两checkpoint的全部fit/held视频均
  优于carrier，确认新Composer不是零容量接口；task93 held后段轻微回落，不把单点峰值当结论。
- shared K1运行面已在同一Writer与唯一CLI内接通：每步固定3 meta + 3 target、task/role等权；每task固定一个CPU evidence-cache owner；
  functional使用正确视频跨episode Panel-A exact LoRA-leaf VJP，process使用strict prefix-only frozen target，preservation只作同一正样本
  单侧hinge；task-held、same-task held与Panel-B均零梯度。单卡两步profile和双卡单步profile均通过，后者覆盖一个rank有task、一个rank
  显式零梯度仍完成deferred NCCL/all-reduce/gather。结构审查已从单个851行shared driver拆为orchestration、optimizer、只读评估与
  authority四个owner，当前无hard architecture violation。
- 专家规定的process Huber loss与无量纲权重`1.0`已在最终代码上完成额外真实单卡一步profile：task1冻结process normalizer为
  `.0558371`，首步normalized process loss为`1.17047`；Frame/Event/Predictor/Composer梯度范数分别为
  `1.23987/.991191/2.50437/3.34321`，峰值allocated/reserved仍为`26.64/35.52GB`。该profile只验证最终目标图与资源，不作方法选择。
- shared实现及最终Huber目标图已经完成回归并提交为`0c5c7e99`；launch合同随clean pushed
  `main@1290673a0b51158c0a4f1fc02ff0f32a729996e0`冻结。12-task K1 component-init full/coarse两条matched formal及四个held5
  correct-only strict250均已完整结束、零错误。full step70/110的Panel-B gradient fit/held benefit为
  `.000542/.000199`与`.001032/.000586`，true-task-held两点均为负；对应closed-loop为`33/31`。coarse为
  `.000684/.000596`与`.000828/.000715`，true-task-held仍均为负；closed-loop为`43/41`。四个checkpoint的Goal/Long均为0。
  coarse逐task按Long/Goal/Object/Spatial0/Spatial9为`0/0/2/40/1`与`0/0/4/34/3`，相对carrier分别保留`37/43`与
  `35/43`；full只有`29/43`与`25/43`。因此full没有证明复杂19层response前端的增量，coarse更稳定但仍只达到carrier，四臂均
  为科学non-pass。
- 等待训练期间已在唯一Writer CLI内接通`materialize` phase：冻结shared checkpoint后，用held5每task固定correct demo5各调用Writer
  一次，物化五套独立且完整的38-target rank16 adapters，并交给既有`evaluate_pi05.py` static task-LoRA运行面做correct-only
  strict250。配置固定为`configs/pi05_ecp_policy_response_writer_held5_eval_v1.json`；deployment runtime不实例化functional action/state
  dataset或processor，不读取held action/reward，不生成wrong、shuffle、reverse、no-video或language-only条件，也不复制Evaluator。
  clean pushed `main@e7631247`上的单卡真实runtime smoke确认authority IDs `71/76/81/86/91`精确对应global
  `0/9/18/25/36`，language tokens同集合、Panel数为0、query dataset/processor均为`None`；冻结资源峰前常驻约
  `9.38GB allocated / 19.23GB reserved`。该路径随后已物化四个checkpoint并完成上述1000条正式rollout；没有为失败checkpoint运行
  wrong、shuffle、reverse、no-video或language-only controls。
- 首个有依据修正不改模型、functional/process目标、rank或scale：选择较简单的coarse与K1，把梯度映射从5 meta + 5 target扩大为
  55个经审计meta与18个target-fit，task2/74仍作true-task-held，另外15个meta held和5个target held不产生梯度。每task只缓存两条
  outcome-independent fit正确视频，最后一条正确视频作held；6卡时按role与cache owner平衡，绝大多数update同时使用6个真实任务。
  配置为`configs/pi05_ecp_policy_response_writer_scale_v1.json`，10 warmup + 1200 effective updates，保存macro610/1210；macro610
  一出现即可在训练继续时并行做held5 correct250，不新增人为内部门。单卡两步真实profile已解析两套completed Panel roots中的75个
  task authority，完成真实capture、functional反传、process反传和唯一rank16物化；step2 Frame/Event/Predictor/Composer梯度均非零，
  峰值allocated/reserved为`27.55/38.17GB`。profile当时为资源上界读取4 fit + 1 held视频；正式首轮已缩为2 fit，profile不作方法选择。
- 73-task scale component-init formal launch contract：scientific implementation为`1a11115bd719aa00ec4a80ac61ee09a200944443`，
  formal从包含本条合同的最新clean pushed detached `main`运行。唯一科学变量是gradient mapping数量；固定coarse、K1、component-init、
  同一Writer/model/rank/seed/LR、correct-only cross-episode functional、positive prefix-only process与preservation。55 meta与18 target按role
  各占每update一半，task在role内等权；task2/74及所有其它held任务零梯度。10 warmup + 1200 effective updates，保存macro610/1210；
  macro610出现后训练继续，另行物化并做held5 correct-only strict250。00:05第一次尝试在gpu02物理4/5/6装载时，另一用户在物理5的
  新任务从`1.48GB`增长到`24.4GB`并开始计算；EMBER在生成run contract、output root、checkpoint或任何科学step前主动停止，未触碰
  对方进程，也没有产生可恢复或可解释的科学状态。修正后输出root固定为
  `runs/outputs/pi05_ecp_policy_response_writer_scale_73task_k1_component_coarse_s1210_1a11115b_gpu01p2456_20260903/`且launch前不存在。
  2026-09-03 00:23 CST同时live检查两节点：gpu01物理2/4/5/6分别为`15/15/98/15 MiB`、util全`0%`且无compute process；gpu02
  物理5已有`26520MiB/74%`动态任务，其余可共驻卡不比gpu01四张空卡更合适，故不跨节点拼卡。四卡真实75-task schedule owner各持
  `18/19/19/19` tasks、采样frame cost为`1818/1850/1815/1807`；只依据静态task/cache owner自动选择两role固定phase后，完整1210步
  每rank最多2个task，`905/1210`步四卡都有真实task，其余305步为三卡各2个task，meta task恰为66次、target为201或202次，不读取
  outcome且不改变科学权重。gpu01 available host memory为`247500165120 bytes`；按75-task真实video长度共5285个采样帧、已测
  `22164019 bytes/frame`估计训练加Panel-B冻结cache峰值`109.1GiB`，四个model process仍有充分余量。`/data1` quota blocks为
  `773575340/1073741824`、limit `1084227584 KiB`；旧shared run约`82MB`，本run保守新增小于`2GB`且不复制dataset/model。
  固定`NCCL_P2P_DISABLE=1`、world-size4与每rank GPU-local NUMA绑定；exact resume只允许同commit、同物理卡、world size、config、输入与
  output root。exact process command为：
  `cd /data1/user/ymdai/projects/EMBER-worktrees/policy-response-writer-scale-formal-1a11115b && NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=2,4,5,6 PYTHONPATH=src /data1/user/ymdai/projects/EMBER/.venv/bin/torchrun --standalone --nproc_per_node=4 scripts/train_ecp_policy_response_writer.py --config configs/pi05_ecp_policy_response_writer_scale_v1.json --asset-root /data1/user/ymdai/projects/EMBER --data-root /data1/user/ymdai/projects/EMBER/data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a --output-dir /data1/user/ymdai/projects/EMBER/runs/outputs/pi05_ecp_policy_response_writer_scale_73task_k1_component_coarse_s1210_1a11115b_gpu01p2456_20260903 --phase shared --representation coarse --initialization component --mode formal`。
- 上述scale formal已于2026-09-03 00:27 CST从clean pushed detached `df7a7f5a`启动，tmux session为
  `ember_prw_scale73_coarse`。run contract精确锁定同一commit、world-size4、物理2/4/5/6、73个gradient tasks、K1及positive-only
  信息墙；四rank冻结cache完成后首个optimizer step在00:34结束。step10完成时最近5步平均`28.92s`，Frame/Event/Predictor/Composer
  梯度均finite nonzero，最大CUDA reserved为`40.883GiB`，gpu01 available host memory仍为`122612895744 bytes`。当前训练继续；
  macro610 ETA只作运行调度估计，不作科学证据。已完成旧12-task formal的clean detached worktree已删除，commit与全部formal outputs保留。
- scale macro610 checkpoint已完整保存并在训练继续时完成held5 correct-only物化与strict250。物化root为
  `runs/outputs/pi05_ecp_policy_response_writer_scale73_coarse_m610_held5_correct_k1_materialized_df7a7f5a_gpu02p4_20260903/`；
  五个held task各只调用Writer一次，held action/reward/state、validation/test、wrong、shuffle和reverse读取均为0，输出均为唯一完整
  38-target rank16。旧evaluator把合法Writer checkpoint硬编码为`{70,110}`；`6ddceff5`将其收敛为正数macro且目录名必须精确匹配，
  相关测试`5 passed`并对真实macro610 bank完成reinspection。该修复不改变model、adapter或评测科学合同。
- macro610 strict250 root为
  `runs/outputs/pi05_ecp_policy_response_writer_scale73_coarse_m610_held5_correct_k1_strict250_6ddceff5_gpu02p46_r2_20260903/`；
  250个严格配对状态、20个shards和四worker均完整，return code全0。结果为`26/250`，逐task Long/Goal/Object/Spatial0/Spatial9为
  `0/0/1/25/0`，breadth `2/5`。相对carrier43保留`22`、获得`4`、丢失`21`，paired exact p约`.00091`；相对coarse
  macro70同样为`22/4/21`，因此是显著净退化。materialized mobile4整体函数范数已从旧macro70/110约为carrier12的
  `.49--.66`倍增至macro610的`1.81--2.15`倍；38 targets普遍放大而非单target爆炸。训练functional benefit仍持续上升，故当前最早
  警报是train functional proxy与held closed-loop脱节并伴随shared residual外推过强，而不是Writer没动或评测故障。
- 随后的48-state固定observation/noise只读功能诊断直接重算carrier、G1与macro610的PI0.5 owner/flow/action response，raw evidence在
  `runs/analysis/pi05_ecp_policy_response_writer_scale73_m610_g1_effect_alignment_5df9406_gpu02p46_20260903/`。五task
  successful-member effect loss均值为carrier/G1/Writer=`.914596/.238841/1.023186`，Writer只在`1/5` task略优于carrier、平均反而
  恶化`.108590`。四个G1非零mobile task上，Writer相对G1的member-scale-whitened功能方向cosine仅
  `.05044--.30470`、中位`.14753`，功能响应norm却只有G1的`.4563--.7444`。因此参数残差虽过大，真正policy effect并非“正确方向
  过强”，而是主要落在低效或错误功能方向；事后缩放只会趋近carrier，不能补出G1效应。该诊断零梯度、使用封存G1 privileged
  effect reference，只作non-pass定位，不参与checkpoint选择。
- 对照专家§7.5与§9.4继续审计后确认，首版只实现逐rank bounded gain，却遗漏完整per-target effective-update RMS cap。macro610
  held5有`94/190`个task-target完整mobile `B@A`超过fit19 task-equal全局`s_ref`、最大`2.2433 x`；fit-only shared template为
  `0/38`，正式G1 held5只有`5/190`个轻微超过。固定压回`1 x s_ref`的post-hoc strict250已经完整结束：`33/250`、breadth
  `1/5`，Long/Goal/Object/Spatial0/Spatial9=`0/0/0/33/0`。相对原macro610 retained/gained/lost=`20/13/6`、paired exact
  `p=.16707`；它局部恢复Spatial0净8条，却丢掉Object唯一成功，证明scale boundary有保护作用但不能补出跨suite正确方向。root为
  `runs/analysis/pi05_ecp_policy_response_writer_scale73_coarse_m610_sref_effective_cap_b8ad986_gpu02p46_20260903/`，manifest明确登记
  `training_gradient_use=false`与`checkpoint_selection_use=false`，四worker、250 rows及return code均完整。
- 原训练前804步的global clip触发率为`.8781`；scale-head/其余方向norm中位为`2.5992/.5839`，同一norm分组后方向侧预计只有
  `.0386`的step触发、有效方向倍率中位恢复`2.6533 x`。唯一实现worktree现为
  `codex/policy-response-writer-scale-boundary`：Composer按完整rank4 Gram矩阵计算RMS并在不物化dense `B@A`时统一缩放B，shared
  optimizer把`scale_head`与其余Writer参数各自按原norm `1.0`裁剪；active scale config已显式记录两项。focused tests为
  `9 passed`，包含dense等价、cap内外、零初始化finite gradient、完整Composer integration和独立裁剪预算。task1/task93两步真实
  shared profile均已`exit 0`完成：step耗时分别为`15.73/13.81s`与`30.14/25.96s`，Frame/Event/Process/Composer梯度均finite且
  nonzero；峰值allocated/reserved分别为`23.47/34.81GB`与`33.30/41.56GB`。source policy、native observer与task-local参数均
  trainable=`0`，wrong/held/Panel-B backward及shuffle/reverse reads均为`0`，输出仍为唯一完整rank16；profile只验证真实图、边界与资源，
  两步内部functional数值不作科学选择。gpu02物理4/6随后释放，总EMBER物理卡回到旧训练使用的4张。
- 该单点不运行controls、不进入mixed-K、fully-random或validation。旧73-task训练继续到预注册macro1210，以相邻checkpoint判断其
  实际未限幅/global-clip parameterization是过渡还是稳定退化；它不再承担停止整个函数类的资格。修正版验证后及时merge/push，
  并从clean detached authority fresh运行同一73-task/coarse/K1/component-init matched实验。等待期间只删除了一个无
  invocation、无shard且已被完整retry1取代的80KiB旧full-macro110 prepare root；formal evidence未删除。
- fresh启动前对照专家动态必要性合同发现，首版Event未使用真实`frame_positions`，却把可学习slot
  position直接混入value，并用slot-specific logits选择relation value。完全重复的静态8帧因此仍产生
  event/frame innovation RMS `.19244/.13996`，scale打开后4个构造target全部打满`.20` cap，是可复现的
  架构合同违反。当前唯一`codex/policy-response-writer-dynamic-necessity` worktree正把position限于emission/
  transition/QK路由、使relation value在slot间共享并做frame-common中心化聚合。CPU结构检查已降至
  `7.23e-8/6.17e-8`，合成mobile RMS最大`4.50e-5`；focused tests当前`10 passed`。task1/task93两步真实shared
  profile均已`exit 0`：步耗时分别为`17.42/18.18s`与`40.28/30.95s`，Frame/Event/Process/Composer梯度均
  finite nonzero，峰值allocated/reserved分别为`23.47/34.81GB`与`33.30/41.56GB`。source policy、native observer与
  task-local参数全部冻结，wrong/held/Panel-B backward与shuffle/reverse reads均为0，输出仍是唯一完整rank16。
  旧macro1210因不含该修正，只作旧parameterization相邻证据，不能替代fresh corrected formal。
- 旧scale formal已自然完成到macro1210，`completion.json/result.json`完整，1210 optimizer steps总耗时`35173.12s`，
  train/evaluation分别为`34855.78/170.97s`；Panel-B、true-task-held、same-task-held、wrong和shuffle/reverse backward/read均为0。
  macro610到1210的10个gradient task fit/held benefit从`.000700/.000107`增至`.001419/.000761`，但仍仅`6/10`
  task全部视频优于carrier；两个true-task-held task的fit/held均值从`-.003674/-.003763`进一步变为
  `-.003806/-.004313`，二者仍为`0/2`全部视频优于carrier。因此旧参数化的训练proxy继续改善却没有跨task泛化恢复。
  macro1210的held5 correct-only strict250已从唯一完整38-target rank16物化结果完成：`30/250`、breadth`3/5`，逐task
  Long/Goal/Object/Spatial0/Spatial9=`0/0/2/27/1`，四worker return code全0。相对macro610 retained/gained/lost为
  `18/12/8`、Jaccard `.47368`、paired exact `p=.50344`；相对carrier43为`26/4/17`、净丢13、`p=.00720`，相对旧coarse
  macro70同样净丢13。旧端点虽从macro610净增4，却没有相邻稳定性、Goal/Long或carrier恢复；该结果只裁决旧参数化，不用于判断
  下面修正版。macro1210五个held adapter的mobile/carrier聚合函数范数为`1.71--2.44 x`，继续高于macro610的`1.58--1.97 x`。
- corrected 73-task component-init matched formal launch contract：scientific implementation固定为clean pushed
  `7d435ea3a6c78141a21fea60baeac6d321a174b3`，formal从包含本条合同的最新clean pushed detached `main`运行。与旧
  `df7a7f5a` arm相比只修正三项已证实的合同偏差：完整per-target mobile `B@A` RMS压在fit19 task-equal `s_ref`内、
  `scale_head`与其余Writer各自按norm `1.0`裁剪、真实frame position只路由emission/transition/QK且静态重复帧不能从slot
  identity制造dynamic value。固定同一`configs/pi05_ecp_policy_response_writer_scale_v1.json`、73个gradient tasks、coarse、K1、
  component-init、10 warmup + 1200 effective updates、macro610/1210、source/Stage0/carrier12/native bank、rank4 residual与唯一rank16、
  data/split/两条fit视频、functional/process/preservation loss、LR/seed/schedule及role/task权重；不引入wrong、shuffle、reverse或
  held gradient。2026-09-03 10:25 CST同时live检查两节点：gpu01物理2/4/5/6均无compute process，分别为
  `15/15/98/15MiB`、util全`0%`；gpu02物理4/6正运行本项目strict250，其他卡要么为他人高显存任务、要么无额外吞吐价值，故训练选
  gpu01物理2/4/5/6，评测占gpu02物理4/6，EMBER总占用正好6张。gpu01 available host memory为`253864224KiB`；旧同构formal
  实际仅`89412KiB`，`/data1` quota blocks为`774454692/1073741824`、limit `1084227584KiB`，fresh root保守新增小于`2GB`。
  输出root固定为
  `runs/outputs/pi05_ecp_policy_response_writer_corrected_73task_k1_component_coarse_s1210_7d435ea3_gpu01p2456_20260903/`
  且launch前必须不存在；固定`NCCL_P2P_DISABLE=1`、world-size4和GPU-local NUMA。exact resume只允许同运行commit、同物理卡、world
  size、config、输入与output root。exact process command为：
  `cd /data1/user/ymdai/projects/EMBER-worktrees/policy-response-writer-corrected-formal-<launch-commit> && NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=2,4,5,6 PYTHONPATH=src /data1/user/ymdai/projects/EMBER/.venv/bin/torchrun --standalone --nproc_per_node=4 scripts/train_ecp_policy_response_writer.py --config configs/pi05_ecp_policy_response_writer_scale_v1.json --asset-root /data1/user/ymdai/projects/EMBER --data-root /data1/user/ymdai/projects/EMBER/data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a --output-dir /data1/user/ymdai/projects/EMBER/runs/outputs/pi05_ecp_policy_response_writer_corrected_73task_k1_component_coarse_s1210_7d435ea3_gpu01p2456_20260903 --phase shared --representation coarse --initialization component --mode formal`。
- corrected formal已于2026-09-03 10:27 CST从clean pushed detached
  `aed7b5244cce91df440d0d4a453b1e3b1be8a346`启动，tmux session为`ember_prw_corrected_scale73`；实际run contract锁定该commit、
  origin/main authority、world-size4、物理2/4/5/6、73-task owner、三项修正与positive-only信息墙。前两个optimizer step已完成，
  step耗时为`35.84/20.75s`；Frame/Event/Process Predictor/Composer/scale梯度均finite nonzero，最大allocated/reserved为
  `28955342848/40542142464 bytes`，没有OOM、non-finite或其它任务干扰。旧`df7a7f5a` formal已经完整结束且无进程引用，其detached
  worktree已删除；commit、checkpoint、raw metrics、result与全部materialized/evaluation evidence均保留。
- corrected task-local capacity formal launch contract：在主shared formal继续期间，从包含本条合同的最新clean pushed detached
  `main`并行运行task1/task93，各自single-process、component-init、coarse、10 warmup + 100 effective updates、macro70/110、
  16 functional rows与每checkpoint 16次零梯度Panel-B；两task沿用sealed correct fit/held视频、task-local reference、source、Stage0、
  carrier12、native X/Y、rank4 residual及唯一rank16，唯一目的为确认动态必要性与完整`s_ref`边界后的实际coarse函数类仍有task-local
  容量，不作为shared checkpoint门槛。首次尝试从`2d3a0627`误用只服务73-task shared的scale config；task1/task93均在run contract、
  output root和optimizer step产生前以同一`KeyError: functional_panel_config`退出，只有外部launcher log/exit code，不构成科学状态。
  scale config的临时task-local allowlist随即删除，不为诊断扩展并行配置路径；fresh retry改用此前正式正控的canonical
  `configs/pi05_ecp_policy_response_writer_v1.json`，其task-local模型宽度、初始化、优化、视频与reference合同不变，当前代码仍强制执行
  dynamic value与完整`s_ref`边界。2026-09-03 11:00 CST同时live检查两节点：gpu01物理2/4/5/6由主训练占用；gpu02物理4为
  `5186MiB/2%`且仅有`982+982+3173MiB`低占用进程，物理6为`4749MiB/0%`且仅有`4584+148MiB`低占用进程，均有约`40GiB`
  余量并已由同路径真实profile验证可安全共驻；gpu02 available host memory为`245593816KiB`。`/data1` quota blocks为
  `774456800/1073741824`、limit `1084227584KiB`，两个约45MB同构formal保守合计小于`1GB`。task1 fresh retry用物理4，root为
  `runs/outputs/pi05_ecp_policy_response_writer_corrected_tasklocal_task1_coarse_s110_7d435ea3_v1_gpu02p4_retry1_20260903/`；task93用物理6，
  root为`runs/outputs/pi05_ecp_policy_response_writer_corrected_tasklocal_task93_coarse_s110_7d435ea3_v1_gpu02p6_retry1_20260903/`；两root均
  不存在。两者固定`NCCL_P2P_DISABLE=1`、`PYTHONDONTWRITEBYTECODE=1`与独立GPU-local NUMA；exact resume只允许同运行commit、
  node/GPU、config、输入和single-process topology。launch前必须再次同时live检查两节点，若共驻余量漂移则改用其它安全卡而不干扰他人。
- task-local fresh retry已于2026-09-03 11:11 CST从clean pushed detached
  `3a342b6ca2d2d88dddc380e4ae943fd28d00bba9`启动，sessions为`ember_prw_corrected_tl1_r1`与
  `ember_prw_corrected_tl93_r1`；launch瞬间gpu02物理4/6状态仍为`5186MiB/3%`与`4749MiB/0%`，总EMBER物理卡为6。
  两条run contract均完整锁定该commit、canonical task-local config、coarse、task1/93、single-process与110 steps。两条均已完成真实
  optimizer step；task1 step3的input/output/scale/task-query梯度为`.005468/.000698/.081094/.000058`，task93按非对称初始化首步仅
  scale梯度`.046751`，均finite。当前峰值allocated/reserved为task1 `23204407808/30383538176 bytes`、task93
  `30760676864/37891342336 bytes`，没有OOM或对其它进程的可见干扰。
- Policy-Response Writer shared matched formal launch contract：scientific implementation为`0c5c7e99`，formal从包含本条合同的
  最新clean pushed detached `main`运行；两臂共用唯一配置`configs/pi05_ecp_policy_response_writer_v1.json`、固定source、Stage0、
  carrier12、s_ref、J2 Panel A/B、mapping split与数据`data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a`。gradient tasks固定为
  meta `1/8/9/32/52`及target `72/73/75/93/94`，task2/74只作true task-held零梯度评估；每step固定3 meta + 3 target，K1、
  两条fit正确视频交替训练，第三条same-task视频只评估。两臂均为component-init、single-process/single-A40、10 warmup + 100
  effective updates、正确视频cross-episode Panel-A 16 rows、Huber positive process权重`1.0`、单侧preservation `.05`，保存
  step70/110并在12 tasks三条视频上各做16次零梯度Panel-B。两臂唯一差异是Process读取full 19-layer x 50-horizon x 2-probe
  response或coarse final-layer horizon mean；Composer与完整dynamic X/Y bank、参数量、seed、schedule和world size完全相同。
  full固定gpu01:2，输出`runs/outputs/pi05_ecp_policy_response_writer_shared_12task_k1_component_full_s110_0c5c7e99_gpu01p2_20260902/`；
  coarse固定gpu02:5，输出`runs/outputs/pi05_ecp_policy_response_writer_shared_12task_k1_component_coarse_s110_0c5c7e99_gpu02p5_20260902/`。
  合同记录时live状态分别为`15MiB/0%`与`159MiB/0%`，后者只有gqma `148MiB`低占用进程；launch前必须再同时检查两节点，若
  状态漂移则不盲目占用。`/data1` quota blocks为`773293620/1073741824`、limit `1084227584 KiB`，两个root均不存在；参考正式run
  每条约`45MB`，即使保守估计小于`1GB`也远低于余量，冻结evidence cache仅驻CPU内存。exact resume只允许同commit、同node/GPU、
  config、输入及world size；无效root不覆盖。两条exact commands为：
  `cd /data1/user/ymdai/projects/EMBER-worktrees/policy-response-writer-formal-0c5c7e99 && NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src /data1/user/ymdai/projects/EMBER/.venv/bin/python scripts/train_ecp_policy_response_writer.py --config configs/pi05_ecp_policy_response_writer_v1.json --asset-root /data1/user/ymdai/projects/EMBER --data-root /data1/user/ymdai/projects/EMBER/data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a --output-dir /data1/user/ymdai/projects/EMBER/runs/outputs/pi05_ecp_policy_response_writer_shared_12task_k1_component_full_s110_0c5c7e99_gpu01p2_20260902 --phase shared --representation full --initialization component --mode formal`；
  `cd /data1/user/ymdai/projects/EMBER-worktrees/policy-response-writer-formal-0c5c7e99 && NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=5 PYTHONPATH=src /data1/user/ymdai/projects/EMBER/.venv/bin/python scripts/train_ecp_policy_response_writer.py --config configs/pi05_ecp_policy_response_writer_v1.json --asset-root /data1/user/ymdai/projects/EMBER --data-root /data1/user/ymdai/projects/EMBER/data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a --output-dir /data1/user/ymdai/projects/EMBER/runs/outputs/pi05_ecp_policy_response_writer_shared_12task_k1_component_coarse_s110_0c5c7e99_gpu02p5_20260902 --phase shared --representation coarse --initialization component --mode formal`。
- Policy-Response Writer task-local formal launch contract：scientific implementation为clean pushed `66df1974`，formal从包含本条合同的
  最新clean pushed detached `main`运行；唯一配置`configs/pi05_ecp_policy_response_writer_v1.json`，复用其中固定source checkpoint、
  Stage0/native observer、carrier12、s_ref、J2 Panel A/B、mapping fit/held split与数据
  `data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a`。task1使用fit videos 5/6、held 24，task93使用2/3、held 46；
  每task独立single-process A40、10 warmup+100 effective updates、Panel-A correct-only 16 rows/step、checkpoints 70/110，每checkpoint
  对三条correct视频各做16次只读Panel-B。task1命令固定`CUDA_VISIBLE_DEVICES=5`在gpu02，输出
  `runs/outputs/pi05_ecp_policy_response_writer_tasklocal_task1_full_s110_66df1974_gpu02p5_20260902/`；task93固定
  `CUDA_VISIBLE_DEVICES=2`在gpu01，输出
  `runs/outputs/pi05_ecp_policy_response_writer_tasklocal_task93_full_s110_66df1974_gpu01p2_20260902/`；两者均设置
  `NCCL_P2P_DISABLE=1 PYTHONPATH=src`并使用canonical `.venv/bin/python scripts/train_ecp_policy_response_writer.py --phase task-local
  --representation full --mode formal`及相同asset/data roots。launch前live状态为gpu01:2 `15MiB/0%`，gpu02:5 `159MiB/0%`且只有
  gqma `148MiB`低占用进程；不会触碰他人进程。`/data1` user blocks为`772868852/1084227584 KiB`，两个run含四枚checkpoint
  保守峰值小于`2GB`，3.04/5.67GB frozen evidence与小型Process cache仅驻内存；两个目标root均不存在。裁决比较step70/110
  fit/held correct functional recovery与既有同task free-primal正控，不使用wrong、held或Panel-B梯度。只允许同commit、同节点/物理卡、
  config、输入与single-process拓扑exact resume；无效或superseded root不覆盖，另名保留。
  两条exact process commands分别为：
  `cd /data1/user/ymdai/projects/EMBER-worktrees/policy-response-writer-formal-66df1974 && NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=5 PYTHONPATH=src /data1/user/ymdai/projects/EMBER/.venv/bin/python scripts/train_ecp_policy_response_writer.py --config configs/pi05_ecp_policy_response_writer_v1.json --asset-root /data1/user/ymdai/projects/EMBER --data-root /data1/user/ymdai/projects/EMBER/data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a --output-dir /data1/user/ymdai/projects/EMBER/runs/outputs/pi05_ecp_policy_response_writer_tasklocal_task1_full_s110_66df1974_gpu02p5_20260902 --phase task-local --task 1 --representation full --mode formal`；
  `cd /data1/user/ymdai/projects/EMBER-worktrees/policy-response-writer-formal-66df1974 && NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src /data1/user/ymdai/projects/EMBER/.venv/bin/python scripts/train_ecp_policy_response_writer.py --config configs/pi05_ecp_policy_response_writer_v1.json --asset-root /data1/user/ymdai/projects/EMBER --data-root /data1/user/ymdai/projects/EMBER/data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a --output-dir /data1/user/ymdai/projects/EMBER/runs/outputs/pi05_ecp_policy_response_writer_tasklocal_task93_full_s110_66df1974_gpu01p2_20260902 --phase task-local --task 93 --representation full --mode formal`。

## 最新科学结论

### 仍成立的正证据

- frozen source validation8为`48/400`，validation8 task-local rank16 oracle为`250/400`。
- held5 source/carrier/independent successful members为`21/43/113`；mobile-rank4解析容量覆盖held5五个task。
- G1 action-in native-block free-code strict250为`114/250`，breadth5/5、Goal2、Long1，正式通过。
- G2 boundary-anchored Natural Program的held full相对endpoints改善`22.2047%`，probe`38/40`、median active events`4`，
  same-task/K1/K4均通过。
- P0/P1、R5等正控证明真实native bank、current-bank operator和task-local功能方向具有容量；它们不证明shared mapping。

### 已裁决G3/PNBTT最早接口

- Program-through-bank topology-matched free-summary S0双task正式通过：task1 correct/held约`.974--.989`、wrong约`-.565`；
  task93 correct/held约`.917--.947`、wrong约`-.342--.394`。
- fresh real Program-through-bank S1正式non-pass：task1 correct fit0/fit1/held为`.826825/.855228/.797545`，task93为
  `.776511/.792673/.719798`；wrong、margin、all-pairs、信息墙、Action Meta 0和唯一rank16均通过。按预注册条件没有运行shared S2。
- §7.1 bank-conditioned-primal恢复correct，但wrong specificity不足：原双tasktask1 wrong为`.428/.477`，task93为`.627/.654`。
- calibrated Q_free把wrong从`.815/.832`降到`.526/.534`，同时把correct降到`.808/.826/.795`，确认capacity--specificity权衡。
- base-LR A_free虽然233个anchors全部更新，但RMS仅`.0094`、约为candidate的`3.7%`，因此只淘汰under-travel版本。
- 最终calibrated A_free把free-anchor RMS提高到`.17664`，已与candidate anchor`.188--.192`同量级。task93 correct
  fit0/fit1/held为`.853296/.858892/.818467`，wrong为`.611592/.668511`；all-pairs通过，wrong和margin正式non-pass。
- 同checkpoint精确F=0后correct升至`.879708/.883433/.849663`、wrong升至`.750229/.756445`。F确实更强抑制wrong，
  但也伤害correct；candidate delta的correct/wrong cosine约`.718--.772`，占主导的free delta约`.993--.995`。
- 最早缺口因此是高相似summary经family-scalar gate调制共享event-additive anchor时只能近同向移动correct/wrong，无法把bank内容差异
  放大为所需功能分离。停止边界只覆盖这一具体parameterization。
- PNBTT的single-key chart、首次tangent spectrum、family-key chart、v2 spectrum和唯一full-rank16 oracle已按专家§5.10顺序
  全部执行。full-rank16 macro70上task1 correct fit0/fit1/held为`.953328/.933839/.941449`、wrong为
  `.648060/.719726`；task93 correct为`.557237/.561168/.411465`、wrong为`-.001312/-.007719`。macro110为
  task1 `.960297/.941644/.948351` vs `.634156/.711548`，task93 `.586174/.595686/.449605` vs
  `-.006466/-.021862`。两checkpoint均为`non_pass`，任务依赖反转稳定。
- 正证据是task16 transport在task1可恢复高correct，在task93可产生强specificity；gate-aligned rank4也能同时把两个wrong arm压到很低。
  负证据是这些性质不能在两个任务上同时恢复E1要求的absolute correct/held；full-rank16没有相对rank4呈现一致、广泛、明显更优，
  gate-aligned spectrum也没有扩容触发。因此不做中间rank或LR/seed/width/scale/chart小扫。
- 该已裁决路线的最早缺口是PNBTT E1 free-query real-bank transport函数类的absolute correct capacity。它不裁决冻结的Natural
  Program、G2、native X/Y、signed pooling、rank4或整个ECP。

完整历史及每个旧架构的结果在`docs/research_history.md`；长期跨轮结论在`findings.md`；全部专家原文均位于`docs/`。

## 已裁决的PNBTT实现

- PNBTT保留G2 Natural Program、真实38-target X/Y及四类output bank、frame quadrature、exact signed replay、small-core
  canonicalization与首版carrier12+residual4。
- Program只产生低维query；当前bank的真实candidate产生key并继续作为唯一native value。B0只做可微key-space whitening，B1在同一bank
  上执行一次联合measure的antithetic signed transport；没有base primal、bounded correction、family scalar gate或free anchor。
- 首个E1 single-key-chart与family-key v2均稳定non-pass。v2 spectrum相对首版的q correct-preserve-wrong中位仅从input
  `.555`到`.566`，四类output约为`.174/.235/.220/.224`；v input从`.463`到`.476`，abs改善到`.643`，但adj/init/goal反而从
  `.808/.727/.734`降到`.769/.685/.693`。尾端10%谱能量仍近零；family chart主要把action-out adj/goal operator cosine从
  `.839/.748`降到`.712/.627`，与formal wrong改善一致，却未补足correct容量。因此不增加`m`或继续改chart。
- 唯一full-rank16 oracle只比较rank分配端点：保持相同family-key PNBTT、free query、数据、loss、Gate和110步cadence，将
  `carrier12+task4`改为`carrier0+task16`；最终仍是单一38-target rank16，不形成rank28或第二adapter。task16冻结幅度先验由与`s_ref`
  一致的fit19、非held task-local rank16 Action Experts做exact small-core singular component RMS后task-equal median得到；不读取
  validation/test，也没有task/video lookup。该oracle已稳定non-pass，未触发rank重开条件。
- 最后裁决config为`configs/pi05_ecp_pnbtt_e1_gate_aligned_necessity_v1.json`；它从family-key v2机械派生，唯一科学改动是将
  `normalized_necessity_margin` 从`.10`对齐到formal Gate `.50`。该轮已稳定`non_pass`，所以没有重算E2所需G2 Program tensors。

## 最新formal evidence

- PNBTT E1 free-query transport：
  `runs/outputs/pi05_ecp_pnbtt_e1_free_query_s110_2664e0d_gpu01p12_20260902/`；110步、macro70/110 Panel-B与
  `evaluations/qualification.json`均完成，最终为相邻一致`non_pass`。
- PNBTT E1 tangent spectrum：
  `runs/analysis/pi05_ecp_pnbtt_e1_tangent_spectrum_m128_step110_8306a4c_gpu01p12_20260902/`；task1/93共380个
  target-side spectra、16个Panel-A visits、三条gradient arms，`completion.json`完整，耗时`376.97s`。
- PNBTT family-key E1：
  `runs/outputs/pi05_ecp_pnbtt_e1_family_key_s110_02633a39_gpu01p12_20260902/`；训练authority固定为clean detached
  `75db5f84`，gpu01物理1/2双rank；110步、macro70/110五臂各16次Panel-B、两个checkpoint与
  `evaluations/qualification.json`完整，最终为相邻一致`non_pass`。
- PNBTT family-key tangent spectrum：
  `runs/analysis/pi05_ecp_pnbtt_e1_family_key_tangent_spectrum_m128_step110_75db5f84_gpu01p12_20260902/`；同一v2 macro110、
  task1/93各16个Panel-A visits、共380个target-side spectra，held/Panel-B/validation/test均未使用，`completion.json`完整，
  耗时`381.48s`。
- PNBTT full-rank16 oracle：
  `runs/outputs/pi05_ecp_pnbtt_e1_fullrank16_oracle_s110_57969a68_gpu01p12_20260902/`；训练authority为clean detached
  `1897b8dceecf93d1b3063b6f42a78f286cb699b2`，110步、macro70/110 checkpoints、两次五臂各16次Panel-B、raw metrics、
  run contracts、completion与`evaluations/qualification.json`均完整；总体与逐task均相邻一致`non_pass`。
- PNBTT gate-aligned necessity E1：
  `runs/outputs/pi05_ecp_pnbtt_e1_gate_aligned_necessity_s110_e65c6388_gpu01p12_20260902/`；训练authority为clean detached
  `2050de9e7583955fa0c62eaeb375eb5b3847500a`，110步、macro70/110 checkpoints、两次五臂各16次Panel-B、raw metrics、
  run contracts、completion、训练/评测logs与`evaluations/qualification.json`均完整；总体与逐task均相邻一致`non_pass`。
- PNBTT gate-aligned tangent spectrum：
  `runs/analysis/pi05_ecp_pnbtt_e1_gate_aligned_tangent_spectrum_m128_step110_2050de9e_gpu01p12_20260902/`；同一step110、task1/93
  各16个Panel-A visits、共380个target-side spectra，held/Panel-B/validation/test均未使用，`completion.json`、`result.json`、shards、
  run contract与analysis log完整，耗时`382.57s`。

- Program-through-bank S0：
  `runs/outputs/pi05_ecp_program_through_bank_bottleneck_s0_gate_s110_b11dc3e_gpu01p23_20260901/`
- Program-through-bank S1：
  `runs/outputs/pi05_ecp_program_through_bank_bottleneck_s1_gate_s110_9047230_gpu01p23_20260901/`
- bank-conditioned-primal双task：
  `runs/outputs/pi05_ecp_bank_conditioned_primal_gate_s110_eb9f295_gpu01p12_20260901/`
- calibrated Q_free：
  `runs/outputs/pi05_ecp_bank_conditioned_primal_qfree_calibrated_task93_s110_fdc669f_gpu01p0_20260901/`
- base-LR A_free：
  `runs/outputs/pi05_ecp_bank_conditioned_primal_afree_task93_s110_b0d81bb_gpu01p0_20260901/`
- calibrated A_free：
  `runs/outputs/pi05_ecp_bank_conditioned_primal_afree_calibrated_task93_s110_e02f4ca_gpu02p4_20260901/`
- A_free逐层与F=0因果审计：
  `runs/analysis/pi05_ecp_bank_conditioned_primal_afree_causal_audit_144d59b_gpu02p46_20260901/`

以上formal evidence、唯一checkpoints、raw rows、aggregate、run contracts与completion均保留；没有因交接清理删除。

## 仓库与workspace整理

- 2026-09-05 Git收口：另一任务完成的v4结果封存与交接文档统一提交到`main`，保留待继任者消费的`HANDOFF.md`。
  本地与远端只保留`main`分支和一个canonical worktree；旧G3未合并提交`2295f481`由已推送的归档标签
  `archive/g3-vector-interaction`保留，原远程分支删除。下列旧分支保留记录只描述当时状态。
- 2026-09-05最终交接清理确认：只剩canonical worktree，无local `codex/*` branch或EMBER进程；`.codex/tmp`、pytest/Python cache
  均为空。删除唯一一份退役的2026-08-31 G3 frozen-condition cache
  `runs/caches/pi05_ecp_program_bank_candidate_interaction_v1_90pair_200a778_20260831/`，释放`61030186361` bytes；它不含checkpoint、
  program output或deployment input，90份可重建safetensors的manifest、重建launcher与provenance仍由formal analysis/log保留。
  `runs/logs`以及formal output/analysis roots作为科研证据保留。
- 交接前58个累积worktree已清理；首个E1、family-key E1、三次spectrum、full-rank16及gate-aligned formal结束后对应detached evidence worktree均已删除。
  PNBTT首轮结论完整快进至远程`main`后，已合并的`codex/pnbtt`实现worktree与本地/远程分支均已清理。目标函数错配审计后，已从最新clean pushed
  `main@9afca0bb`建立的`codex/pnbtt-gate-aligned-necessity`也已合并并清理；最后formal固定在detached `2050de9e`完成后，该worktree同样删除。
  训练、评测与分析日志均已移入各自formal root，`.codex/tmp`已为空；当前只剩canonical worktree且没有local `codex/*` branch。
- full-rank16 formal启动后已删除被提交记录取代的两步disposable profile及两个非运行worktree的Python/pytest cache。为保持一个
  canonical Writer运行面，删除4个只暴露已退役EBSRI/J3/routing-control路线的旧runner：`train_ecp_bank_set_shared.py`、
  `train_ecp_bank_set_tasklocal.py`、`evaluate_ecp_bank_set_shared.py`、`evaluate_ecp_joint_program_primal.py`；历史modules、configs、tests和
  formal artifacts仍保留审计，PNBTT的已结束训练与checkpoint评测均由唯一`train_ecp_joint_program_primal.py`运行面执行。
- 删除8个local `codex/*` branch：已合并分支由`main`保存；两个未合并EBSRI S2草案因S1预注册non-pass而失去执行资格；历史
  `g3-vector-interaction@2295f48`仍由`origin/codex/g3-vector-interaction`保存。
- 已删除完整并入`main`的远程`codex/g3-bank-set-relative-interaction`与`codex/g3-v4-evaluator-authority`；未合并的
  `origin/codex/g3-vector-interaction@2295f48`明确保留。
- 两个旧dirty worktree分别是已被clean S0/S1链和后续G3历史取代的实现草案；确认无运行进程、无formal authority引用后随worktree清理，
  未提交内容不可恢复。
- `.codex/tmp`中约`5.1GB`旧smoke/profile/script/cross-language临时cache已删除；其中影响决策的结论均已进入`findings.md`或
  `docs/research_history.md`。后续profile只作可删除工程证据，不与formal roots混存。
- 未删除或移动dataset、models、formal runs、checkpoints、raw rows、aggregate、source policy、task experts或ownership不清资产；
  仅删除了上文已经逐项证明可重建且不含formal状态的退役condition cache。
- tracked旧科学代码、测试和历史configs在新实现复用/退休审计前保留；`main`是canonical source，旧结果或仍存在的config不得自行
  恢复为路线。

## 已裁决PNBTT执行细节

- 第八次专家原文、代码/config/authority冲突和formal evidence已完成逐项复核；未发现推翻主路线判断的结果错误。
- PNBTT canonical compiler已接通：Natural Program只供query，real bank同时供key/value；包含可微batched key whitening、joint-K等video
  质量、exact chunked antithetic signed replay、四类output scope、38-target rank4 materialization及唯一carrier12+residual4 rank16。
- E1 task-local free-query训练与Panel-B evaluator已接通；policy、Program、carrier/scale、native values与Action Meta均冻结，correct fit0/fit1和
  wrong fit0产生梯度，held/wrong fit1/Panel-B为零梯度。task8/94只提供unrelated Panel-A states，preservation用同一keyed flow
  time/noise比较generated与carrier真实action velocity；wrong-video仍有单侧carrier上界。run contract从真实policy/Program模块审计
  Action Meta，而非声明式写零。
- E0 synthetic hard checks通过：zero native value给出zero residual；candidate/video排列与chunked replay误差仅为FP32低位；K2 video质量各
  `.5`；bank swap改变方向；forward/gradient finite；真实policy消费唯一38-target rank16。
- 首个真实双卡profile因一次性保留38-target covariance/Cholesky autograd图在A40约44GiB OOM；按target即时链式回传后不改变梯度
  （synthetic leaf-gradient最大误差`0`）。接入真实`D_policy`后的最新两步profile在task1/task93 microbatch 8/4下稳定完成，分别为
  `25.000/24.665s`；rank0/1峰值allocated为`39.773/36.154GB`、reserved为`46.376/44.109GB`。step1 free-query梯度非零且shared key按
  非对称LoRA零初始化预期为0；step2 shared-key梯度为`.293542`，task1/93 paired policy distance为`.003844/.002297`，correct/wrong已分离。
- 上述profile只验证工程图与吞吐，不参与E1科学Gate。E1 macro70/110均完成五臂各16次Panel-B；两枚checkpoint的task gate均为
  `non_pass`，总体与逐task结论一致。step110相对step70的correct/held改善仅`.013--.037`；near-bound最大值从未超过`.022005`，
  因此首个E1失败不是softmax饱和、训练过短或Natural Program；后续已按专家指定完成`T=Cov(v,k)`功能梯度投影谱。
- `T=Cov(v,k)`诊断已自然完成：380个谱均来自train-side Panel-A，operator列数固定1024；除结构性零bank外，99%谱能量rank远低于
  1024且末端10%能量通常不超过`1e-6`量级，因此不增加`m`。q/v的功能梯度保留与correct/wrong operator重合暴露的是chart
  表达问题。family-shared nonlinear trunk + target-specific rank16 low-rank projection已经接入；`m=128`、rank4、query、loss、
  数据与Gate未改，35项PNBTT/shared-compiler/joint-primal focused tests通过。
- v2 implementation `02633a3964ecfd9d40f9827ba98456c87c07552b`已在clean pushed main完成双A40两步真实profile。step2
  family-key aggregate gradient为`.155687`，task1/93 free-query gradient为`13.945/9.212`，correct/wrong已分离；单步
  `25.266s`，两rank峰值allocated为`39.789/37.260GB`、reserved为`46.272/44.082GB`，无OOM或non-finite。
- `0f052cccc9ddb96fbcaaa2a036fdc61ee190d945`在不改变当前K1 E1的前提下补齐E2前置硬合同：每条视频在每个有效
  event/scope先归一为等质量再按`1/K`混合，并缓存授权内容排序键以稳定集合归约；K2每半event mass精确为`.5`，相同Program
  context下的native内容换序测试通过。`a2c3fe9e`同时把canonical runner默认配置从退役J3收敛到当时PNBTT v2；两提交均已
  fast-forward并推送至`main`，family-key E1在运行期间仍固定在其祖先`75db5f84`。
- fresh E1 formal launch：从`02633a39`之后只增加本记录的clean pushed detached `75db5f84`运行；配置为
  `configs/pi05_ecp_pnbtt_e1_family_key_v2.json`，task1/93双rank DDP、110 optimizer steps、macro70/110 checkpoints，数据、
  Panel-A/B、loss与Gate完全复用首个E1。使用gpu01物理1/2，launch瞬间两卡均空闲，固定`NCCL_P2P_DISABLE=1`和NUMA0；输出
  `runs/outputs/pi05_ecp_pnbtt_e1_family_key_s110_02633a39_gpu01p12_20260902/`在launch时为fresh空目录。`/data1`当前user用量
  `772469868/1073741824 KiB`，参考上一E1的`257MB`，本轮含两个checkpoint峰值估计小于`1GB`。只允许同commit、同world-size2、
  同config exact resume；不覆盖无效root。科学裁决仍只认macro70/110五臂各16次Panel-B及相邻一致E1 Gate。
- family-key E1已经自然完成。macro70 task1 correct fit0/fit1/held为`.598648/.599961/.581859`、wrong为
  `.028320/.041884`；task93 correct为`.693744/.706930/.650097`、wrong为`.036270/.224452`。macro110 task1 correct为
  `.616630/.620958/.601512`、wrong为`.027332/.051458`；task93 correct为`.707775/.725727/.655429`、wrong为
  `.047247/.223365`。wrong、all-pairs与near-bound均通过，task1 margin也通过；两task correct/held和task93 margin稳定不足，
  70到110的correct/held改善只有`.0053--.0210`。因此family-key提高了specificity但没有恢复absolute capacity，不追加训练且不进入E2。
- v2 tangent spectrum也已自然完成：仍为380个train-side Panel-A spectra、每task 16 visits，耗时`381.48s`。相对首版，q/v input的
  correct-preserve-wrong中位只小幅变化为`.566/.476`；q四类output为`.174/.235/.220/.224`，v为
  `.643/.769/.685/.693`，没有形成correct容量所需的新可达方向。action-out adj/goal correct--wrong operator cosine降至
  `.712/.627`，解释了wrong specificity改善；但q/v input仍约`.958`，abs仍约`.927/.963`。全部非结构性operator的尾端谱能量仍远低于
  width上限，因此停止增加`m`或继续改key chart。该诊断本身不证明rank4 ceiling；只准入专家限定的一次同构full-rank16 oracle。
- full-rank16 oracle实现`57969a6895adfe2e336e5d83a30d1a80c12d47d2`保持一个参数化运行面：PNBTT residual rank由配置取4或16，
  rank4仍走原12+4拼接，唯一oracle直接物化task16；overcomplete action-out canonicalization以small-core SVD后零填充保持合法rank16
  shape。16项native/PNBTT与22项shared-compiler/functional focused tests通过。gpu01物理1/2两步真实profile自然完成，step1/2为
  `29.469/28.910s`；step2 task1/93 free-query梯度`9.883/11.488`、shared-key梯度`.205305`，全部finite。两rank峰值
  allocated为`39.841/38.584GB`、reserved为`45.722/44.080GB`，没有OOM；相对rank4约`25.3s`只增加约17%步时。
- full-rank16 formal已从clean detached `1897b8dceecf93d1b3063b6f42a78f286cb699b2`自然完成，root为
  `runs/outputs/pi05_ecp_pnbtt_e1_fullrank16_oracle_s110_57969a68_gpu01p12_20260902/`。配置固定
  `configs/pi05_ecp_pnbtt_e1_fullrank16_oracle_v1.json`，task1/93双rank、110步、macro70/110、五臂各16次Panel-B；除rank分配和对应
  fit19冻结task16 scale prior外，E1数据、三项loss、LR、seed与Gate均不变。两枚checkpoint、Writer、optimizer/trainer state、raw rows、
  run contracts、completion、evaluation logs和qualification均完整，所有launcher exit为0。
- formal launch preflight已同时检查两节点：gpu01物理1/2均仅`15MiB`、util `0%`，物理3/4也空闲；gpu02物理5空闲、4/6可共驻，
  0--3与7为他人高负载任务。训练选择gpu01物理1/2与NUMA0，因为两task一rank一卡已是有效拓扑且复用该节点23GB condition cache；
  不跨节点拼卡、不干扰他人。`/data1` user blocks为`772567180/1073741824 KiB`，参考上一formal仅`94684 KiB`且本轮两个更大
  Writer checkpoints仍预估小于`1GB`；目标root确认不存在。固定`NCCL_P2P_DISABLE=1`、world-size2、相同commit exact resume，
  macro70出现后可在仍空闲的物理3/4并行Panel-B以隐藏评测时间。
- full-rank16 macro70 task1 correct fit0/fit1/held为`.953328/.933839/.941449`、wrong为`.648060/.719726`；task93
  correct为`.557237/.561168/.411465`、wrong为`-.001312/-.007719`。macro110 task1 correct为
  `.960297/.941644/.948351`、wrong为`.634156/.711548`；task93 correct为`.586174/.595686/.449605`、wrong为
  `-.006466/-.021862`。两枚task gate均为`non_pass`，overall/per-task conclusion consistent与training complete均为true。该结果在
  task1通过correct/held却失败wrong/margin，在task93则通过wrong/margin却失败correct/held；两task的all-pairs和near-bound都通过。
  因此它是稳定科学non-pass，不是训练未完成、饱和、OOM或评测错误。
- `c992b3f0d1fc5954f55ad939368881aa7a78a52e`已删除430行仅绑定退役primal/gate/anchor拓扑的stale tests，保留active cache、
  set不变性、信息墙和member-effect合同；25项focused tests通过。该清理提交已fast-forward至`main`，不改变当时冻结的
  detached scientific authority，该formal运行现已完成。
- `50f876cb0e5e2e3623a4b77e768d67658960fccc`修正detached formal评测把会正常前进的`origin/main` tip误当训练身份的问题；
  现在仍锁定实际commit、clean/detached拓扑与全部科学合同，只允许包含该commit的authority tip前进。26项focused tests通过。
- 重新核对专家§6/7.2与三个formal metrics后，确认旧`.10` necessity hinge在formal `.50` margin失败时已关闭，先前的route/authority blocker裁决因此撤销。
  唯一`.50` config已由`e65c63888033639c58d29f285aed6cd8331c07e8`提交并推送。gpu01物理1/2双rank两步真实profile自然完成：step1/2
  `active_necessity_fraction`均为`1.0`，step2 task1/93 free-query梯度为`15.319/9.216`、shared-key梯度为`.171069`，correct/wrong已分离；
  单步`26.063/25.363s`，两rank峰值allocated约`39.79/37.26GB`、reserved约`46.27/44.08GB`，无OOM或non-finite。该profile只证明
  gate-aligned hinge真实接通，不作科学Gate证据。
- gate-aligned E1 formal launch contract：从包含`e65c6388`且clean pushed的detached `main`运行
  `scripts/train_ecp_joint_program_primal.py`，配置固定`configs/pi05_ecp_pnbtt_e1_gate_aligned_necessity_v1.json`，source checkpoint、
  tokenizer、dataset、G2/native authority与两个frozen cache均复用family-key E1；task1/93双rank DDP、110 optimizer steps、macro70/110，
  唯一科学变量是necessity margin `.10 -> .50`。环境固定`CUDA_VISIBLE_DEVICES=1,2`、`NCCL_P2P_DISABLE=1`、NUMA0与canonical venv；
  输出`runs/outputs/pi05_ecp_pnbtt_e1_gate_aligned_necessity_s110_e65c6388_gpu01p12_20260902/`在preflight时不存在。live检查时gpu01物理1--4
  均空闲，训练用1/2、保留3/4给checkpoint Panel-B并行；gpu02物理5空闲，4/6可安全共驻，其余为他人高负载任务。`/data1` user blocks为
  `772766460/1073741824 KiB`，参考同构rank4 formal峰值估计小于`1GB`。只允许同commit、world-size2、config和输入的exact resume；无效root
  不覆盖而另名保留。裁决只认macro70/110五臂各16次Panel-B与相邻一致E1 Gate。
- gate-aligned E1已从clean detached `2050de9e7583955fa0c62eaeb375eb5b3847500a`自然完成。110条raw training rows、macro70/110
  两枚checkpoint、五臂各16次Panel-B、contracts、completion与qualification完整。macro70 task1 correct fit0/fit1/held为
  `.585596/.592489/.541733`、wrong为`-.176695/-.153551`；task93 correct为`.707213/.715694/.676823`、wrong为
  `-.055836/.018941`。macro110 task1 correct为`.607645/.609189/.561628`、wrong为`-.171164/-.149315`；task93 correct为
  `.710657/.721565/.686395`、wrong为`-.086657/.006107`。两checkpoint每个task都通过wrong、margin、all-pairs与near-bound，
  只失败correct/held；总体与逐task均相邻一致`non_pass`。70到110的correct/held增益仅约`.003--.022`，不追加训练。
- `.50` necessity objective已真实行使：`active_necessity_fraction`在step1--10为`.95`、11--70为`.3083`、71--110为`.05`；
  末步task1/93 free-query梯度为`.1701/.1801`、shared-key梯度为`.04281`，preservation平均激活率`.9909`。因此结果不是旧`.10`
  hinge错配、梯度断开、anchor未移动、OOM或训练未完成。
- gate-aligned step110 train-only tangent spectrum随后在同一clean authority与gpu01物理1/2完成，380个target-side spectra、每task
  16个Panel-A visits、correct fit0/fit1与wrong fit0三条gradient arms均完整，held/Panel-B/validation/test读取为0。最大末端10%谱能量
  `1.3664e-5`，与旧v2的`1.3675e-5`等价；q/v input correct-preserve-wrong中位为`.5584/.4806`、operator cosine为
  `.9580/.9577`，action-out adj/goal operator cosine为`.7039/.6365`，均未形成新的correct可达方向。故不触发增大`m`、继续key chart
  或重开rank分配。
- 当时专家规定E1通过后才进入E2；次选B要求E1通过而真实frozen Program E2失败；whole-Writer joint也要求上游A路线Gate成立。
  这些条件均未发生，所以该PNBTT authority在当时没有后续active route。停止只覆盖已实际检验的PNBTT E1 transport函数类，不外推为
  Natural Program、G2、native X/Y、signed pooling、rank4、ECP或zero-interaction根本失败。
- 对专家原文§5.10、E1--E4及次选B做了第二次逐条件路线审计，并逐层复核当前query/key、Cholesky whitening、联合measure、antithetic
  real-value replay、固定type normalization、rank4物化与三项loss。没有发现工程合同偏离，也没有尚未执行且满足触发条件的专家分支：
  `m`未截谱、family chart已执行、full-rank16未明显优于rank4，E2/B/joint仍分别被E1前置条件阻断。
- workspace cleanup发现canonical joint runner仍默认指向已裁决的`.10` family-key config。该陈旧默认已删除，`--config`现在必须显式提供，
  从而保留sealed configs与复现实验能力但不会误把旧路线当作active；没有修改任何scientific config、模型或formal artifact。PNBTT与joint
  定向测试`30 passed`，CLI help及missing-config fail-fast均通过。
- `HANDOFF.md`已消费并删除；长期信息全部由authority、已裁决PNBTT design、本文件与Git保存。

## 2026-09-05 Frame-Aligned正控与下一shared资格

- clean detached `7b42cdf6`已完成task1/task93 Composer-only formal。m50 fit/held recovery为
  `.076140/.095793`与`.086793/.079274`，两任务m25/m50的全部fit/held正确视频均高于carrier；task93 held较旧global-broadcast
  `.04547`改善约74%，但距离free-primal仍远。
- 四个零梯度correct-only诊断root已完成：
  `runs/analysis/pi05_ecp_policy_response_writer_frame_aligned_task1_contrast_response_m50_7b42cdf6_gpu02p1_20260905/`、
  `runs/analysis/pi05_ecp_policy_response_writer_frame_aligned_task93_contrast_response_m50_7b42cdf6_gpu02p2_20260905/`、
  `runs/analysis/pi05_ecp_policy_response_writer_frame_aligned_task1_unit_factor_response_m50_7b42cdf6_gpu02p0_20260905/`、
  `runs/analysis/pi05_ecp_policy_response_writer_frame_aligned_task93_unit_factor_response_m50_7b42cdf6_gpu02p2_20260905/`。scale=1逐值复现
  正式评测；固定contrast与强制单位化均被排除，未读取wrong/shuffle/validation/test、未产生梯度或checkpoint选择。
- 下一formal使用`configs/pi05_ecp_policy_response_writer_frame_aligned_12task_v1.json`：gradient meta
  `[1,8,9,32,52]`、gradient target `[72,73,75,93,94]`、true-held meta/target `[2]/[74]`；100 optimizer steps、
  m50/m100、每步6 tasks且本配置为3+3。机械schedule审计确认每个gradient task在m50/m100恰有30/60次暴露，per-task cursor覆盖
  两条fit视频与Panel-A visits；这不是owner规定的固定batch或角色比例。
- 该shared只训练correct cross-episode functional，full 50-horizon、真实native X/Y、frame stride5、rank12+4和唯一rank16不变；
  true-held与Panel-B零梯度。只有task-disjoint正信号才解封held5，不以内部loss或人为门槛代替闭环。

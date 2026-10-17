# §17经历条件功能修订：实际完整交付

fresh ψ单轨180有效更新、720经历条件/20160跨episode FM query，随后36完整condition/144独立query已实际完成。八臂1152逻辑case/28配对比较验收，终点固定、不按loss选点；无held/Test/自动后继。

当前93/144，对Reader90089、旧§16 video86。实践83→100；12个实际改参数任务query24→28。超过Reader900的当前独立面板总量，仍需主讨论判断增益分布、保持和正式资格；本批不自动扩量或开held。
本批为训练task/已见初始H诊断；旧no-video59不是新ψ匹配消融，旧full是变化theta的历史方法，180不称充分，也不能据此认证H语义/视频因果或普遍否定H学习。

## 全部结果

|面板|case/臂|source|reference|Reader900|first_best|full|video|no_video|functional_feedback|
|---|---|---|---|---|---|---|---|---|---|
|全部|144|58|127|89|73|79|86|59|93|
|target24|96|14|80|43|29|35|41|14|48|
|source-seen support12|48|44|47|46|44|44|45|45|45|
|五identity|20|20|20|20|20|20|20|20|20|
|libero_10|24|3|15|5|2|2|4|2|6|
|libero_90|48|44|47|46|44|44|45|45|45|
|libero_goal|24|8|20|14|14|14|13|7|14|
|libero_object|24|0|21|9|1|3|10|0|11|
|libero_spatial|24|3|24|15|12|16|14|5|17|

breadth：{'Reader900': 29, 'first_best': 23, 'full': 25, 'functional_feedback': 30, 'no_video': 19, 'reference': 35, 'source': 20, 'video': 30}。参考新增70保留与完整28对R/G/L/churn/union/交并/Jaccard逐项见results.json。

|比较|before→after|R|G|L|churn|Jaccard|
|---|---|---|---|---|---|---|
|Reader900_to_functional_feedback|89→93|85|8|4|12|0.876289|
|video_to_functional_feedback|86→93|83|10|3|13|0.864583|
|source_to_functional_feedback|58→93|56|37|2|39|0.589474|
|reference_to_functional_feedback|127→93|91|2|36|38|0.705426|
|full_to_functional_feedback|79→93|73|20|6|26|0.737374|
|first_best_to_functional_feedback|73→93|69|24|4|28|0.711340|
|no_video_to_functional_feedback|59→93|58|35|1|36|0.617021|

|task|suite|source|reference|Reader900|first_best|full|video|no_video|functional_feedback|
|---|---|---|---|---|---|---|---|---|---|
|0|libero_spatial|1|4|4|3|3|4|1|4|
|1|libero_spatial|0|4|1|0|0|1|0|2|
|12|libero_object|0|4|4|0|1|4|0|4|
|13|libero_object|0|3|0|0|0|0|0|0|
|20|libero_goal|0|4|3|3|3|3|0|3|
|21|libero_goal|3|4|4|4|4|3|3|4|
|32|libero_10|0|3|0|0|0|0|0|0|
|34|libero_10|2|2|1|0|0|1|1|1|
|2|libero_spatial|1|4|4|4|4|4|3|4|
|4|libero_spatial|0|4|1|1|3|1|0|2|
|5|libero_spatial|1|4|3|1|3|2|1|3|
|7|libero_spatial|0|4|2|3|3|2|0|2|
|14|libero_object|0|4|4|1|1|4|0|4|
|15|libero_object|0|4|0|0|0|1|0|2|
|17|libero_object|0|3|0|0|0|0|0|0|
|19|libero_object|0|3|1|0|1|1|0|1|
|22|libero_goal|1|4|3|3|3|3|0|3|
|25|libero_goal|0|1|0|0|0|0|0|0|
|28|libero_goal|4|4|4|4|4|4|4|4|
|29|libero_goal|0|3|0|0|0|0|0|0|
|35|libero_10|0|4|1|0|0|1|0|3|
|36|libero_10|0|3|0|0|0|0|0|0|
|37|libero_10|1|3|2|2|2|1|1|1|
|38|libero_10|0|0|1|0|0|1|0|1|
|42|libero_90|4|4|4|4|4|4|4|4|
|43|libero_90|3|4|3|3|3|3|3|3|
|51|libero_90|3|4|4|4|4|4|4|4|
|55|libero_90|4|4|4|4|4|4|4|4|
|56|libero_90|4|4|4|3|3|3|4|3|
|62|libero_90|4|4|4|4|4|4|4|4|
|64|libero_90|4|4|4|4|4|4|4|4|
|73|libero_90|3|3|3|3|3|3|3|3|
|95|libero_90|3|4|4|3|3|4|3|4|
|96|libero_90|4|4|4|4|4|4|4|4|
|97|libero_90|4|4|4|4|4|4|4|4|
|101|libero_90|4|4|4|4|4|4|4|4|

Object12/14、Long及全部Reader/旧video/reference得失逐case列入query_changes.json；保持和新得分别报告，不以同分遮掩交换。未query候选全部保留，不虚构oracle收益。

## 实际学习与编译

source/full38r8/normalization/linear64 U/S/saved center/Reader900/code_units冻结，ψ fresh seed7/AdamW1e-4/.9,.95/constant；无旧meta9/H/Adam/MT/reference回退/第二adapter。只用36等权合法nonheld跨episode完整50×7 FM，经同套完整A/B、固定D回到H修订；自己的动作仅解释经历，无动作BC、PG、BA、code/producer/keep等辅助。
标签流承接实际Reader900 visits100及query cursor2844..2868，teacher排除；ψ从update0开始。720前缀来自§16 314实际完整面板（V156/noV158）和1256旧H引用，按当时全部既往面板重建m/c/remaining，包含失败、未选与terminal；新teacher与旧H off-policy重配，未取未来H/finalbest/query/R/ref。
实际原生FM LoRA cotangent和head非零，首更新body0符合零head初始化，后续H body梯度实测非零（最早2步）；CPU还验证各H消费者/因果性/完整chunk重算。此为信用接通证据，不是学会纠错或能力证明。
保存90仅完整恢复、180唯一终点，加原子rolling/上一恢复点；CP含ψ/Adam/scheduler/scalerNone、全部cursor/rankRNG/topology/frozen paths/units/physical/timing。物理拓扑在完整9更新边界迁移，保留logical112/四task全局权重，SUM不重复除world；新增rank RNG来源实记，不称bitwise exact。
终点从既有Reader900实际完整LoRA起步，初始video144实践/H（83成功/35207步）强制复用且占各8192，10个原4/4立即止。之后ψ只读当前condition实际H，u=m+δ确定性提案，完整四例严格提高才移动m；平局保持，失败/未选H继续可读，精确重复已实践64向量则复用证据并止。没有Gaussian、局部optimizer、固定提案轮数或新condition轨迹拟合。
合法方法实际126次ψ决策、126个候选完整实践/15严格接受（重复代码决策不虚增实践）；另75个未认证提案（71已实践、4未实践）封存，全ROOT共201个saved proposal receipt，不把事故决策省略为零成本。停止分布{'all_four_success': 15, 'insufficient_complete_worst_case_panel': 21}。最终文件就是实际完整实践过的38/r8；初始未变96 query固定复用§15 Reader900，余48真实执行。query只在最终冻结后打开，执行不读V/H/ψ或继续适应。全部决策输入/代码/receipt/接受曲线与完整H链接保留，不把训练用旧后续H塞进部署。

## 费用、资源与吞吐

累计新物理860提交，保守预算计费217602control+8600dummy=226202步，包含事故300行84110实际步与8 unknown上界1840；已完成物理行确切224362步，unknown实际观察lower155/upper1840，故实际动作步区间[224517, 226202]，不把单个failed operation0当整pack费用。合法方法的实际物理结果552行，其中200成功/352正常科学失败。精确GPUh16.555653970374，分阶段{'compilation_and_query': 4.375884972148471, 'training': 12.17976899822553}。状态{'complete': 852, 'reserved_unknown': 8}；普通工程额8/16，合法单condition最大8120≤8192，累计3004/control400015/dummy30040/联合404615分别验收；原76认证行未重跑，300事故行未混入方法结果。
旧initial144/35207只读复用且仍计条件预算，不重复算新物理费；新H504份/8.762GiB、主要原张量14.113GiB，压缩比0.6208551435777668。原§16三点二三GPUh、§14九点三二GPUh、§15 unknown上下界，以及source/reference/native144/operator1170/basis/video等全部旧精确量/区间均沿历史原件保留，未归零或强行摊精确单video GPUh。
完整教学M使用原纯图像Gemma prefix的148GiB级只读特征，再经冻结Reader900有序读取；缓存不等于零教学成本。H用真实pre/post raw vision、同次flow0/9 action hidden、facts63与真实行为u，不构造fake suffix或teacher动作条件。新M/H视觉编译、共享训练/读取、部署新H与query费用全部计入job wall/GPUh；操作seconds不是可直接相加的pipeline wall。第一次回执竞态修复后第二次跨node冻结JSON索引丢key，均保留真实失败与成本。main按实际builder直接返回分支认证76行19173步及13完整条件，300行84110步/75旧提案封存uncertified-cache-provenance，不称全部实际读错或H科学阴性；其中284实践79871步/16query4239步各自独立。旧query不参与恢复选择，23条件仅从认证前缀重新生成一次后缀。原CP180及训练12.17976899822553GPUh保持，0重训/补量/选90。精确清单、事故/恢复authority、原manifest/小witness/ledger/conditions/traceback在analysis/main_contract及engineering_attempt002；旧原件均保留。
教学费用分层：初始实际LoRA复用原36条v0编译/1179帧当量；本窗720 logical teaching访问对应720 unique视频/23655帧。物理rank分片真实消费M1368次/44909帧当量，冻结M构建720、命中648，不能当更多独立任务或条件。合法126次提案读4074帧当量，75旧未认证提案读2542帧当量另计；失败未形成完整receipt的部分计算包含GPUh，未伪称完整读取数。详细来源见teaching_cost_accounting.json；原纯图像148GiB特征不重建、初始m/文件不重新编译并不等于0教学。
有用180更新中比较micro7/14/28和更大真实FM批、H chunk16..512/image16..128及完整周期物理rank，实际FM批直方图{'7': 54, '14': 344, '21': 638, '28': 36, '56': 10}；allocated/reserved峰24044.1/39530.0MiB。各topology完整周期时延和选择见useful_throughput.json，不把不同H长度/命中率的顺序周期冒称同输入因果profile；未增加有效更新或query。
独立native评估沿§16实测16优于24的证据，动态long-first完整面板；全部物理槽数{'4': 25, '8': 17, '12': 10, '16': 31}，实测{'12': {'end_to_end_plans_per_second_median': 3.8316374628719663, 'native_plans_per_second_median': 7.028950563970974, 'packs': 10, 'peak_allocated_MiB': 10793.05078125, 'peak_reserved_MiB': 33422.0}, '16': {'end_to_end_plans_per_second_median': 5.537560760019859, 'native_plans_per_second_median': 8.793465261775015, 'packs': 31, 'peak_allocated_MiB': 11400.9501953125, 'peak_reserved_MiB': 37392.0}, '8': {'end_to_end_plans_per_second_median': 4.045786613617804, 'native_plans_per_second_median': 6.911222080307398, 'packs': 17, 'peak_allocated_MiB': 10183.31396484375, 'peak_reserved_MiB': 34444.0}}。释放inactive缓存后各有用面板的单独结果为{'4': {'packs': 18, 'end_to_end_plans_per_second_median': 2.1334612015987866, 'native_plans_per_second_median': 4.222149733937601, 'peak_allocated_MiB': 9578.03857421875, 'peak_reserved_MiB': 10862.0}, '8': {'packs': 6, 'end_to_end_plans_per_second_median': 3.5022794024169244, 'native_plans_per_second_median': 6.501415857106025, 'peak_allocated_MiB': 10182.65771484375, 'peak_reserved_MiB': 14718.0}, '12': {'packs': 3, 'end_to_end_plans_per_second_median': 3.8223453720524723, 'native_plans_per_second_median': 6.877627599010838, 'peak_allocated_MiB': 10641.91650390625, 'peak_reserved_MiB': 16946.0}, '16': {'packs': 6, 'end_to_end_plans_per_second_median': 5.069209520202069, 'native_plans_per_second_median': 8.95002314434651, 'peak_allocated_MiB': 11400.3837890625, 'peak_reserved_MiB': 27314.0}}；不同任务/长度不当作同输入因果profile。没有重探32或添加profile行；每launch live双node/process/真实余量/owner6或8/node6与nodeCPU分别准入，不挤foreign；多卡训练P2P禁用/NUMA/deferred，独立evaluator不用NCCL，正常BF16/TF32不逐bit追查。
冻结cache全局H≤16GiB、M≤1GiB，ready/reserved/partial及DB-journal控制余量共同计，当前{'H': {'capacity': 17179869184, 'charged': 17140436396, 'control_bytes': 970752, 'global_ready_reserved_partial_budget_verified': True, 'learned_embedding': False, 'ready': 204}, 'M': {'capacity': 1073741824, 'charged': 6979951, 'control_bytes': 110592, 'global_ready_reserved_partial_budget_verified': True, 'learned_embedding': False, 'ready': 23}}。新单SQLite DELETE事务索引与payload内schema/key/source/reader/观测身份绑定通过实际两node六并发180唯一key、中断恢复及错身份拒绝；旧tensor-only可再生payload退役仅留小witness/metadata，未复制整16GiB。只LRU本批可再生特征，完整H/V无截短，ψ变化不缓存旧learned H。全部物理新H788份/14.197GiB包含284份事故H；ROOT终值30.916GiB，64GiB合同包含原子临时/代码/恢复，末次storage-node独立quota/shared保留，未把df当quota或声称直接测OS峰。

## 时钟、释放与交接

实际接收08:30:38.195Z、准备08:30:46.693Z、首owner2200056启动2026-10-16T09:03:08.034317+00:00、真实退出2026-10-16T12:48:33.594620+00:00，累计owner实际执行185.46min（修复CPU/Git空档另计真实派发wall）；各训练/迁移/编译job实际开始退出在launch/exit，原2026-10-15T09:05:48.038Z参考首时钟不重置。起估2–4.5h，首有效长H更新22–47s后已调整到约2.5–4.5h；实际费用不是wall强杀门槛。
全部GPU子进程已退出；原两个callback失败（child exit0）、cache failure、六个configs启动失败和一个EGL0x8cdd失败与费用均完整保留。EGL事故8 reserved_unknown原1840上界继续占各condition和全局，原pending实际代码只补跑一次，消耗原16工程额度中的8，不重抽提案/不清零其它合法后缀。新冻结仅在panel前释放inactive Torch cache供EGL使用；实际pending及剩余流程完成是本批工程证据，不宣称唯一driver根因。runner完整close、逐node本批owner/GPU/env释放见resource_exit.json：[{'node': 'GPU-node-A', 'UTC': '2026-10-16T12:50:30.905850+00:00', 'user_processes': 37, 'user_zombies': 0, 'user_threads': 554, 'user_PSS_KiB': 344041, 'run_processes': [], 'remaining_recorded_environments': []}, {'node': 'GPU-node-B', 'UTC': '2026-10-16T12:50:30.943694+00:00', 'user_processes': 66, 'user_zombies': 0, 'user_threads': 976, 'user_PSS_KiB': 1254402, 'run_processes': [], 'remaining_recorded_environments': []}]。单owner直接等实际退出，无Queue自通知/心跳/第二waiter；无自动360/其它CP读出/held/Test/新参考。
正式代码身份['0b9b8080421a05ec32fd979d591f76707c91b4df', '26a8fbb37296d1228bcee4f68c2924cab1103a8b', '76d5480ba781012a7fd794af39e05e127a21c3ab', 'eac21f1c35c32cc274702ae047f8ef8b2f7b73cb']，clean pushed detached冻结保留；单一control_space active运行面，初实现新增五模块净1055行及后续claim/缓存身份/预算leg窄修的owner与生命周期已记录，事故恢复净增115行/0新模块/architecture0hard。第三owner因稀疏冻结漏configs，六worker在authority阶段退出，0新提案/环境，0.01624122010336982GPUh独立保留；同eac21f1c新完整src/configs冻结先通过实际authority CPU消费者后接续，并非改模型/科学输入。开发树仅在clean/完整集成核验后清理，旧formal/frozen/大资产/foreign保留。最终main clean/pushed commit和canonical记录写锁释放见git_delivery.json；整批一次交main01a13dd9科学消费，自主接续，无例行Owner许可或main工程ACK停点。

证据：results/report、acceptance、candidate_trajectories/query_changes、cost_accounting/useful_throughput、全部conditions/proposals/原始rows/H、training完整日志/CP和resource_exit；没有以loss或实现完成代替EMBER科研目标成功。

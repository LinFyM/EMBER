# EMBER progress

2026-10-04 main完成机制§123/findings§302：全部24对首次规划在任何新环境动作前已响应对象换位，不能只归因后续物理反馈。
首帧采样候选经真实强MT对照后不采用：T1943/262080=.741377%，MT1259/172800=.728588%；比例无Writer特异性，覆盖/预算差异保留。
新增仅既存首块CPU读取1.484240秒及原train数据/采样统计，0新模型/环境/GPU/训练。原root analysis两个小目录保存全量原件。
main仍独占canonical/Git，无active新实验或已选架构；实际A几何监督的只读子项已完成，教学坐标对应和有效M使用仍未由共用A保证。
下方§122等完成记录保持，不恢复旧位置扫描、首帧重训或辅助路线。

2026-10-04 main完成机制§122/findings§301科学消费：直接核实际XY变换/被动In、六双RGB、全部48新+24旧continuous。
C/init4/5转搬旧orange位置的butter与init6跨位置仍搬orange并存；T/init2原butter成功→换位同位置orange入篮，原成功不证明角色绑定。
全部第三对象/未成功及旧新行为分叉保留；按混合结果裁决，不扩位置/对象扫描或恢复grounding/原生key/旧辅助。
main CPU.235226秒、0新模型/环境/GPU，原root analysis/main_scientific_readback.py/.json；本批完整成本仍.259836593305GPUh。
canonical tracked/Git窗口已由main接回，无active实验或已选新训练；继续教学关系到自身情境控制的同一难题。
下方执行交付/接手/待退出均是已消费历史，不能恢复旧运行。

2026-10-04 `object_position_transport_20261004` 唯一固定48行与完整读回已完成：三模型各原/交换八初态，6 full/42 compact，全50×7 chunks/实际physical actions/T+1全对象/EEF/gripper/官方goal及butter-orange被动In齐备，缺项0。
C900旧/本批原/交换成功0/0/2（交换init3/4）；T2340 1/1/2（原2，交换3/4）；MT300 2/1/1（旧2/7、本批原7、交换3）。原→交换R/G/L分别0/2/0、0/2/1、0/1/1；旧→本批原C/T无churn，MT丢2，仍抬butter但280未入篮，未重跑择优。
行为不是单一对应：C原orange抬>3cm为0/1/4/5/6、In0/1/4/5，butter全8静止；交换butter抬3/4/5、In3/4，orange抬并In6/7。init4/5改搬原orange区域的butter与位置依赖相容；init6跨位置仍搬orange与错误实体跟踪相容；0/1及其它第三物体/轻微移动反例全部保留，不强行归类或以换位得分选模型。
八实际registry/free-joint/world XY物理检查和48前后状态通过；各自z/quat/qvel、机器人state8/controller、其它body/model/time保持，0额外step/settling，双RGB真实刷新、无明显穿透/悬空（最深contact−.020521mm）。新原布局对旧24初始body/EEF/quat/gripper/goal误差0，scene/teacher/env7/policy root7/共同绝对noise配对；实际前5与continuous逐行一致，只butter官方In终止。八物理初态重复，不冒称48独立/strict400。
六full实际双RGB已看；init0原C/T搬orange，换位C向ketchup并小移、T搬ketchup，MT原butter附近操作未入篮/换位orange入篮。旧orange In不存在、42 compact无轨迹RGB，均不造图/重算。中心/1cm/3cm/命令不证明抓持或内部根因，main负责进一步科学解释。
完整成本.259836593305/2GPUh：实际物理验证.005058317908、六首次preconsumer失败.004921007279、有效48读取.249857268118；加载/渲染/I-O/失败/退出全计。六读取消费者/六worker exit0，CPU完整读回4.180215秒exit0，两节点本用户GPU0及全部本批PID消失，不影响他人。
六独立persistent消费者并行，各将全部八合法case打包（capacity16/实测maxbatch8），source常驻；完整48GPUstage151.752520秒/.316304行每秒，reserved10.785156GiB。八case已全部打包且总cap6已用尽，不新增样本或重复profile填显存，未实测配置比较、不声称速度收益。Owner显存/吞吐长期要求current_owner_requirements§5和AGENTS§9按此执行。
新root+工程阶段观察高水1.297657GiB、准入保守峰6/硬12，非连续精确峰；strg01 data1独立quota2147483648KiB/实占1193873672KiB与shared86845632372KiB通过，旧source/MT/data0只读。所有新增data1。
物理/首次失败freeze8a2c8496，实际48 clean pushed detached读取dac0ad57fb2e10b86ecfc9224f6bb2af208554c1；C900学习/原correct40085919994、T学习e2afbfd7/原读取cb535c1e、MT学习3ebb979b/原读取83946ae1分列，0新训练/Writer/native/held动作标签。首六launch只因新prepare遗漏tokenizer manifest_path及固定子集恢复入口而前加载拒绝，真实模型/tokenizer未变，按原语义修复新push/freeze；CPU检查假设与最终snapshot摘要KeyError均修并留失败；首次退役检查直接import叶模块触发既有public回导入循环，按canonical public入口顺序通过，未改科研计算，0新科学计算，不写成科学阴性。
primary `/data1/user/ymdai/ember_runs/object_position_transport_20261004/` 的readback/completion、analysis/report/rows/per_case/行为/RGB/全8轨迹图、全部evaluation原件及launch预算/失败/退出/释放。专用两文件与五hooks已退役，canonical contract guard拒绝新执行，Git/两freeze/原件保留；交付集成push及一次有来源Queue后canonical/Git窗口回main，实验session停止新增计算，无自动其它位置/对象/层/强度/fresh/controls/Test/RL。下方启动/待退出仅为历史时点。

2026-10-04 `object_position_transport_20261004` 固定48行第二次实际启动：C900_original_v2 PID4065331, C900_swapped_v2 PID4065330, MT300_original_v2 PID4065332, MT300_swapped_v2 PID4065329, T2340_original_v2 PID4065334, T2340_swapped_v2 PID4065333。
新clean pushed detached读取dac0ad57；六份重新prepare均exit0/13.245543秒/0GPU，实际canonical validate_resume_inputs在GPU启动前六份全部通过。真实tokenizer未变，19份选中header/原视频/场景/完整A-B来源重查通过，旧混合BF16/F32 MT按原值保留。
再次现场双节点/独立data1 quota/shared/相关实占准入，本用户0→6、cap6/单节点6；六独立persistent消费者各打包八合法case，Source常驻，无额外例/profile或新模型。已有全部费用.009979325187 GPUh，六读取最多1.8，含加载/渲染/I/O/退出仍在2硬限。
此前六失败均0模型/新行，失败目录/冻结8a2c8496及工程回执保留；合法physics八初态也保留不重做。当前48行/6 full/完整读回仍待退出后确认，实验独占canonical窗口并在CPU侧准备读回，无阶段自Queue/日志轮询。

2026-10-04 `object_position_transport_20261004` 首次六个launcher已退出exit1，均在模型/worker/环境加载之前被canonical恢复检查拒绝，0新增诊断行；六次费用合计.004921007279GPUh（含启动/I/O/退出），全部失败日志/contract/queue/回执保留。
原因已CPU核实：Source模型身份无差异，只有tokenizer manifest_path历史冻结树与新消费树不同，真实tokenizer路径/字节/模型均相同。原prepare复用旧字段遗漏新消费者解析；同时需要把固定注册八行银行子集接入canonical恢复重查，而不能按原400全任务重查。
修正在独占分支：按canonical inspect_tokenizer记录新manifest来源；task16范围由冻结专用合同校验，原bank的选中task/episode/factor/source/LoRA及19份header按实物重查，source/normalization/model和旧原件保持。三模型实际CPU恢复子集检查通过；通用Source/git/assets/噪声/行验证保留，无物理/模型/评分语义变化。
另保留CPU MT缺少per-condition factors的元数据假定错误、progress编辑脚本语法错误，以及新header检查把旧MT的BF16/F32混合误要求为全F32；仅修正检查口径，原权重/dtype未改，均0GPU/环境。不把前启动工程失败写科学阴性。已有八物理初态验证仍有效，不重复模型或物理检查；新push/clean frozen继续唯一未执行48行。

2026-10-04 `object_position_transport_20261004` 窄接入已集成push，实际clean detached frozen=8a2c8496。原三个银行仅选task16八行元数据，C完整A/B、T sharedA+完整条件B、MT完整共同因素由原canonical loader消费；19份既存header/八teacher/官方来源核实，0新物化或模型候选。
同八授权初态的真实physics消费者已完成：全部registry/free joint/world XY、z/quat/qvel/robot-state8/controller/其它body/model/time不变检查通过；刷新双RGB已逐图核实，无明显穿透/悬空，最深接触−.020521mm。验证0 policy/native forward/0 rollout控制step，18.209944秒×1=.005058318 GPUh，exit0；初始前后完整状态/图像/contacts在root analysis/physical_consumer。
六个各八行的原/交换正式诊断消费者CPU prepare正在完成；容量16会由现有env池限制为八个合法case并整批打包，六独立读取按空闲卡并行，不造额外例/环境填显存。尚未启动48行模型读取，后续启动前再核双节点/配额/卡数。一个CPU原件检查曾误假定MT有per-condition factors，已按既存共同银行语义修正；0模型/环境/GPU失败。

2026-10-04 实验session 01a0fabb-f7a0-7100-93d8-6a0f66055553 已实际接手 `object_position_transport_20261004`，独占canonical tracked/Git窗口；main只读科学分析。
从clean pushed 1ac14681建立codex/object-position-transport隔离树，完整读取Owner要求/机制§121/findings§300/合同。仅task16八个post-dummy初态，冻结C900/T2340/MT300×原/交换世界XY共48行；6 full/42 compact，官方butter成功终止和两对象被动In分离。
预计2–3小时/2完整GPUh/12GiB，准入保守新增峰6GiB；strg01 data1实占1192509776KiB/quota2147483648KiB/shared86847001360KiB通过。只读源银行/Source/旧MT，不复制大资产，0训练/native/held动作。
正在沿canonical scene/episode/trajectory及persistent evaluator接入有界物理变换；尚未启动任何模型或环境诊断。专用入口/hooks交付后退役，所有失败/费用保留，无自动后继。以下仅登记/历史状态被本实际接手覆盖。

2026-10-04 main完成机制§121/findings§300，active design登记为[自身对象位置交换诊断](docs/designs/object_position_transport_diagnostic.md)。
唯一task16/init0..7，C900/T2340/强MT300×原/交换butter-orange世界XY，共48行；固定原视频/LoRA/语言/噪声，不训练或扩训练数据。
从真实错对象行为区分区域依赖与错误实体跟踪，保留混合/无诊断结果；不由此预选架构或把换位得分当官方资格。
预计2–3小时、2完整GPUh/12GiB，6 full/42 compact与两对象被动In；当前尚未派发/启动，main持canonical窗口，提交后交实验session。
main此前误把salad区域用于butter所产生的疑点已纠正：task16 BDDL/官方初态/原scene一致，无该恢复错误、无原件改动。
下方§120等均为已消费历史，不恢复其关闭分支。

2026-10-04 main完成机制§120/findings§299：既存B20的A修正仅14.14%能量是跨query共同项，变化项在12/20/32有益、0有损。
前5父/A风险.126520/.119086；J离线稍优却闭环更弱、两臂unmasked全50更差的反例完整保留。
不把query变化等同纯state反馈，不追加中心化loss或相关探针；继续解释已有输入相关控制为何未跨对象/场景迁移。
新增仅原root analysis/query_function_decomposition.py/.json，主CPU.163282秒/0GPU；无新模型/环境或已选后继。
main仍独占canonical tracked/Git。下方§120登记段是结果前记录，已被本完成条目消费。

2026-10-04 main按机制§120登记一次既存完整策略B20预测的CPU分解：父/A/J四task×两teacher，区分共同偏移与query相关修正。
只解释§117有限学习和§119迁移不足之间的联系，0新模型/环境/GPU；尚未得到结果，不自动追加loss/架构或正式训练。
原生key候选沿§25边界不采用；历史只读子项已结束。canonical tracked/Git仍由main独占，无active实验执行者。

2026-10-04 Owner在核对最近24小时推进后明确恢复继续，并提醒必须围绕同一难题有逻辑推进，不能遇难后换成容易的小问题。
该约束已进入current_owner_requirements；无新实验、训练或GPU。main继续承担教学关系到自身状态控制的机制推导。
任务LoRA加减尚缺跨对象/场景/阶段的功能可迁移依据，未登记或派发；只读历史子项核对现有对象监督与实际执行Q的联系。
canonical tracked/Git窗口仍由main独占。以下§119等记录为已完成历史，本次继续不重启其关闭分支。

2026-10-03 main完成机制§119/findings§298科学消费：核冻结实际消费者、四模型原始400行、200份task16 continuous和全部8个新full/关键父点RGB。
A64只143/400，有限获取没有成为广泛优势；Spatial净−8、task31得12失6，不能把净+3或局部成功当完整方法成立。
task31八个新增从cream-only补成双目标；task16父/A仍48/50目标butter位移≤.01cm，orange juice抬高>3cm从39增至42。
main独立读回在原root analysis/main_scientific_readback.py/.json，主CPU.280秒、0新模型/环境/GPU；阈值不定义抓持或根因。
本项不选A64、不续训/扫点或恢复冻结课程；main已接回canonical窗口，继续固定数据内推导，无active实验或已选后继。
以下实验交付/接手段落为本次已消费历史，不能恢复执行。

2026-10-03 `fixed_B_transfer_readback_20261003` 固定400新视频银行与paired400全部完成，8 full双RGB/state8、392 compact/全50×7 chunk/physical actions/goal/T+1 continuous齐备，缺项0。
A64 143/400，父900140、成熟T2340161、强MT300153；task3/6/11/16/23/26/31/39分别A64 29/2/46/1/2/44/19/0、父32/7/45/1/1/41/13/0、T45/6/45/5/0/36/24/0、MT41/7/36/10/0/35/24/0。
对父R121/G22/L19/churn41/Jaccard .746914；对T R115/G28/L46/churn74/Jaccard .608466；对MT R106/G37/L47/churn84/Jaccard .557895。8-task精确whole-task经验重采样净成功95%区间[-14,20]/[-56,16]/[-50,32]，无训练seed不确定性。总分小胜父3仍低于强参照，无稳定/selected/视频动态因果资格。
Source/归一化数值/官方口径/400 scene/env-policy seed7/root7与共同绝对noise一致，T teacher映射亦一致；三参照对新行初始body/EEF/quat/gripper/predicate误差均0。所有generated前5与实际physical/continuous一致，正反400逐行保留，无参照重评或退为background。
task31双目标13→19但R7/G12/L6；init0父仅butter=478/最终[0,1]，A64仅cheese=393/最终[1,0]，两者520失败，真实双RGB/goal/轨迹支持目标交换而非修复。task16父/A64均1/50同成功集合；init0父实际搬orange juice，A64仍向orange附近动作但目标butter未动、goalfalse。task23仍2/50、39全模型0/50；中心/3cm/命令不证明抓持/接触。
读取身份clean pushed detached d56f6fa2；A64学习b0df34ee/旧32环境a270fb01、Source1000b8ea00e9、900实际训练/原correct40085919994、450父a0e0248d、Original32读取923ff89b分列；旧sealed简称及原模型/结果保持，0训练/optimizer/held teacher action/FM/新标签/额外case。
物化225.033899秒×6=.375056499 GPUh；CPU prepare14.473874秒/0GPU；环境750.462311秒×6=1.250770519，合计1.625827017959/3完整GPUh，包含加载/编译/native/I/O/退出。三个stage及12worker均exit0，所有PID退出/双节点本用户GPU0；CPU legacy tuple解析/只读schema两次exit1按canonical捕获owner修复；退役检查另一次expected父目录索引写错、修正实际绝对endpoint后通过。三个CPU错误/回执保留，0额外model/env/GPU。
原规定案例内主动chunk32→128（最长91帧完整打包、native reserved21.345703GiB）；env容量8→16/两persistent replicas每卡/六卡，真实最大adapter batch16，worker max reserved12.96875GiB（单worker，非整卡峰）。实测.543901 rows/s，对旧父.548744未证明更快，实际GPUh更贵，保留效率不利项；不追加重复400/profile找赢家。
新增root+工程树阶段实占16.612556GiB，保守准入峰20/硬24，含冻结/临时/cache/所有输出，非连续精确磁盘峰；strg01 data1最终used1192765076KiB/quota2147483648KiB/shared余88950413602816B通过。所有新增data1，旧资产只读。
primary `/data1/user/ymdai/ember_runs/fixed_B_transfer_readback_20261003/` 的completion/readback、analysis/report.md/rows.jsonl/per_case.csv/完整配对/CPU读取/RGB与来源、banks/64、evaluation/correct400及launch预算/实际stage/释放原件。专用archive接入与两配置退役；Git/frozen/失败/原件保留，canonical仅保留实际batch缓存及worker最终资源统计，未新增trainer/evaluator/runtime mode。
退役针对性AST/实际导入/archived runtime guard核实后Git集成push/清理task-owned工程树/整批一次回报，窗口交回main；实验session停止新增计算。科学解释与下一判断由main消费原件负责，无自动续训/选点/controls/Test/RL或新方法。以下启动/接手段落为历史时点。

2026-10-03 `fixed_B_transfer_readback_20261003` 的400新视频完整38-target银行已完成：exit0，225.034秒×6卡=.375056499完整GPUh，factor共16480787200 bytes。chunk128覆盖最长91帧，实测reserved21.345703GiB；更大chunk不能增加实际帧，未重复编译/profile。
CPU canonical prepare通过（exit0/14.474秒/0GPU），400映射/原scene/RNG/official/归一化配对一致。correct400已实际启动，PID2367038，仍用clean pushed detached d56f6fa2。
launch再次live双节点、本用户0→6卡、data1独立quota/shared与银行后实占核实；env容量16替代旧8、每卡2个persistent replicas，当前授权400内按实际大小打包，worker最终保留峰值/最大真实batch/吞吐。六卡reader无Writer/gamma/新训练，完整64权重来源及900纠正分列。
本批400环境行/8 full/392 compact及CPU行为读回未完成；实验session继续独占canonical/Git，main只读，无自Queue/周期日志读取或其它arm。

2026-10-03 `fixed_B_transfer_readback_20261003` 的400新视频银行已实际启动：clean pushed detached d56f6fa2，GPU02物理0/1/2/3/7/6，bank owner PID2317299。
现场双节点本用户0→6卡/总cap6/单节点6，六卡live低util且最少38.5GiB空余；strg01 data1独立quota/shared复核准入。native framechunk32→128，授权最长视频91帧可整段打包，无额外视频/profile forward。
A64完整545-key Writer仅冻结读取；全部38-target A0+S/B0+M保持，0optimizer/训练/held动作。父/T/MT旧400原件的source、scene、policy/env RNG、官方口径及source normalization数值配对一致，T的完整teacher映射亦一致。
银行结束后用canonical persistent/dynamic队列读取固定400环境行，8 full/392 compact；工程packing为env容量16/每卡2 replicas，实际吞吐/峰值/有效batch在消费者退出记录核实，不以最低显存为目标。主讨论只读，实验session继续独占窗口；无自动其它矩阵。

2026-10-03 实验session 01a0fabb-f7a0-7100-93d8-6a0f66055553 已实际接手 `fixed_B_transfer_readback_20261003`，独占canonical tracked/Git；main只读科学分析。
从clean pushed e8134b5f建立codex/fixed-b-transfer-readback独占工程树，完整读取§117–118/findings§296–297及合同；仅接入原A64全Writer归档，复用canonical Compiler/bank/official evaluator。
固定8 validation任务/400不同视频、父900原scene/RNG及8 full/392 compact，无optimizer/新训练/held动作/额外面板。预计45–90分钟、3完整GPUh/24GiB，预计新增18–20GiB；strg01 data1 quota2147483648KiB/实占1175344300KiB/shared余88968256987136B已核，峰值预算准入。
来源链900续训/原correct400=85919994、450父=a0e0248d、Original32读取=923ff89b、A64学习=b0df34ee、原环境=a270fb01分列；旧sealed简称保留。本批尚未GPU启动，实施/实际退出随后更新。

2026-10-03 main完成机制§118/findings§297，登记唯一[既存A64共享B跨任务完整读回](docs/designs/fixed_B_transfer_readback.md)。
冻结原A64全部权重，原8val/400不同task-video条件及900映射/scene/RNG；8 full/392 compact，对比既有900140/T161/MT153。
无新训练/数据/held动作，完整读回用于判断已有有限控制修正的迁移价值；不自动追加模型、controls、相邻或正式selected声明。
预计45–90分钟、3完整GPUh/24GiB。原900实际训练/读取85919994、450父a0e0248d、原32bank读取923ff89b的来源简称已纠正，模型路径/结果未变。
当前仅登记，尚未派发/启动；main仍独占canonical tracked/Git，提交后交既有实验session独占实施。以下§117无后继为此前裁决时点。

2026-10-03 main完成机制§117/findings§296科学消费：核实际算子、24份B20和全部96原continuous及三组task32双RGB对照。
普通动作A在固定表示/A下13→24/32，J22/32没有额外收益；仅16物理init，不解释为冻结优于共同训练或新task迁移。
J全50示范位移有学习，前5有限、实际执行四task均未胜零位移；原学到/未学到的区别及全部能力交换已保留。
main派生读回在原root analysis/main_scientific_readback.py/.json，主CPU.385秒、0模型/环境/GPU。按预登记关闭J续训/扫描理由。
canonical tracked/Git由main独占，无active实验或已选后继；完整目标未完成，继续固定数据范围推导。以下本批执行段落为已消费历史。

2026-10-03 `joint_action_effect_credit_20261003` 全部固定学习/读回完成：两臂各64有效更新/完整恢复点，16份完整38-target bank；64新增环境行/12 full双RGB/52 compact，父32行复用，48份A28/B20功能引用（40份新PT）齐备，缺项0。
父/A/J成功13/24/22；task0/12/20/32分别父8/2/2/1、A7/4/6/7、J6/4/6/6。A对父R10/G14/L3，J对父R9/G13/L4；J对A R20/G2/L4/churn6，新增均在task20/t38 init0/2，丢task0/t40 init2、20/t38 init1、20/t42 init2、32/t43 init2。
task32全部父/A/J均开炉并最终保持，放置1/7/6；两新臂均丢父t17/init2。J对A少t43/init2的放置；A成功327，J无placement/结束520。body高度、中心/命令仅描述，未宣称抓持/接触。
J实际执行1..n≤5位移RMS按task为8.154/13.244/4.186/6.276mm，相对零位移风险1.029/2.256/1.069/1.847；四项均未胜零位移，保留所有成功/静止/失败。离线B20示范误差和实际自身效果分列，所有通道/有效mask/逐query/不利项保留。
场景/seed7/root7/共同绝对噪声及64条新增初始body/EEF/quat/gripper/predicate与父配对通过，初始误差0；实际actions与独立continuous一致，全部50×7 chunks、J全50×3被动预测/T+1点位和12 full齐备。CPU读回exit0，针对性完整性核验exit0；三份task32 RGB对照来自既存full，无新render。
物理标签06c49c53；训练/功能b0df34ee，环境读取a270fb01；父900/a0e0248d、Source1000/b8ea00e9分别登记。首次global_task_id接口失败为0新环境step、两臂exit1；合法64/银行/功能保持，新版本只续原环境面板，两消费者exit0，无重训练/native/功能推理。
整批完整GPUh=1.027844869627/4（首次.807849274079 + 环境.219995595548），含全部加载/profile/native/I/O/失败/退出；训练实测reserved高水25.478516GiB，micro14→28/物理suffix28→56与两臂并行依据在profile/contract，不以低显存为目标。
新增root+工程树阶段实占5.770306GiB、保守准入峰10/硬限12，非连续精确磁盘峰；strg01 data1独立quota/shared已最终核实。四消费者PID已退出，双节点本用户GPU卡数0，没有影响他人。
primary `/data1/user/ymdai/ember_runs/joint_action_effect_credit_20261003/`：completion.json/readback.json，analysis/report.md/rows.jsonl/per_case.csv/verification.json/pairing_initial_readback.json/functional_readback.json/control_readback.json/rgb，以及A/J checkpoint64、bank、cases/results和launch两份账本/失败/释放证据。
专用入口/四个诊断模块及临时owner hooks退役，canonical保持唯一运行面，旧合同执行需原frozen；Git/frozen/全部科学与失败原件保留。退役AST/实际canonical导入/两合同guard通过，未重跑全仓或额外模型。整批Git集成push及一次来源明确回报后交回窗口，实验session停止新增计算；main负责科学消费/后继，无自动续训/fresh/held/Test/controls/RL。
以下接手、运行和工程修复段落仅为历史时点，不覆盖完成状态。

2026-10-03 §116固定环境读回已实际启动：新clean pushed detached a270fb01，GPU02物理2/3分别A/J（PID1731130/1731326），每task八个已登记case打包、两臂并行，一份resident source/臂。
此次只读既存64末点完整bank；不重新训练、编译native或读取功能面板。原训练/功能身份b0df34ee与新环境读取身份分列，首次失败及.807849GPUh保持。
launch前双节点现场准入、本用户0→2卡/合计上限6、data1 quota2147483648KiB/实占1174468184KiB/shared余88994780536832B已核；阶段root+工程树4.682GiB，含全RGB仍在预计10/硬限12GiB内。
等待实际消费者退出后统一核64行/12 full/52 compact、配对场景/噪声、被动位移及正反读回；当前仍持canonical窗口，main只读，无自动后继。

2026-10-03 §116两臂固定64训练、完整checkpoint64、父/A/J固定A28/B20全部功能原件和16份新完整LoRA bank已完成，训练/功能读取b0df34ee。
首次环境消费在创建任何新环境step前因run contract无global_task_id而退出；全局ID本来在bank的suite/local任务表中，错误是工程元数据接入，不是科学阴性。
两臂exit1、累计完整GPUh=.807849（含全部加载/native/profile/功能/I/O/退出），原失败日志/launch/恢复点/预测保持；当前GPU消费者已退出。
仅修正canonical任务映射并以新clean pushed detached代码继续原定64环境行；复用已完成64恢复/LoRA/功能预测，0重训练、0新native、0重功能推理。原科学/样本/信息墙/预算不变。

2026-10-03 §116已实际启动：clean pushed detached b0df34ee，GPU02物理2/3分别A/J（PID1546900/1547112）；launch前两节点现场准入、data1独立quota/shared和当前/启动后卡数已登记。
A完成八teacher一次完整38-site public native并保存固定H/context/d/A/K/delta-z；J同期读取固定父B20功能面板，随后复用同一cache，未重复native或Teacher特权读取。
两臂各两次已登记输入的可丢弃完整更新通过实际FM消费者/全部38处B五组梯度检查。micro14→28并vmap两个teacher，物理suffix batch28→56、同query prefix只读一次；逻辑查询/32维noise-time/condition1/8/完整余切不变。
A实测20.645→19.156秒/更新、reserved18.730→25.156GiB；J18.374→18.314秒、reserved19.527→25.129GiB，均选择micro28。最大组已覆盖该task全部两teacher×28固定queries，且每臂两次profile已用尽，不额外生成工作或扩profile。
初始全部B参数/fresh optimizer/RNG已恢复，固定64正在运行；实测学习预计A1226秒/J1172秒。公共β/source/解释器/A冻结，J仅额外监督7:10合法位移、其余padding不加loss。
直接进程退出事件由本批budget owner持续等待，无自Queue/周期日志读取；唯一root内保留launch/profile/标签来源/计费。整批功能/64环境行与退役交付未完成，不把此时profile或loss当科学结果。

2026-10-03 §116工程与语义准入：192个合法query episode、21122个原有物理时点的CPU恢复完成（69.089秒、0GPUh），EEF最大误差1.407e−15m。
四个实名body/site与原7bd8a6f5 post-action缓存对应成立；task20 site挂在中抽屉body。仅固定A28训练/独立B20所需窗口，未读teacher特权状态作标签。
标签恢复来自clean pushed detached06c49c53；无新环境step/render/数据。共享native只缓存固定H/context/d/A/K/delta-z，不缓存可训练B Value/M。
工程复用ConditionalTarget B Value、原完整FM余切与canonical evaluator；专用有限面板/位移捕获/入口将在整批后退役。针对性CPU mask/零梯度/shape及导入检查通过。
尚未启动GPU；即将集成并冻结实际消费者，按两臂独立并行及真实micro14/28完整更新profile选择执行配置，全部失败/加载/profile计费。

2026-10-03 实验session01a0fabb-f7a0-7100-93d8-6a0f66055553实际接手 `joint_action_effect_credit_20261003`，独占canonical tracked/Git窗口；main只读科学分析。
基线09ee68f2；完整合同、机制§116/findings§295及Owner科学/资源要求已读。隔离分支codex/joint-action-effect-credit，独立data1工程worktree。
当前为实体/post-action时间对应的CPU语义准入及工程实施；尚无新GPU、模型forward、训练或环境执行。不恢复旧T/prefix/Gamma路线。
唯一root `/data1/user/ymdai/ember_runs/joint_action_effect_credit_20261003/`，固定64×两臂及64环境行；预计2–4小时，硬限4完整GPUh/12GiB。
strg01现场data1个人实占1169523776KiB、quota2147483648KiB、shared余89017670901760B；含标签/cache/两臂恢复/banks/RGB/冻结工程树预计10GiB。
只读旧源/原件；teacher state/action不作表示输入。后续按真实吞吐验证microbatch/frame chunk，正式消费来自clean pushed detached。

2026-10-03 main完成机制§116/findings§295，登记唯一[动作—物体位移联合信用](docs/designs/joint_action_effect_credit_diagnostic.md)。
当前900固定教学读取/解释器/A；两臂只学习共享38处B生成器，比较原动作FM与query自身实际位移的原生联合输出信用。
固定原四task/八teacher/64步A28、每臂32环境行/六full及原功能面板；预计2–4小时、4完整GPUh/12GiB，不扩数据。
此时尚未派发/启动模型、标签恢复或GPU；main持canonical记录/Git窗口，提交后交既有实验session独占实施。
阳性也不自动完整fresh，阴性不追加坐标/λ/层/时长扫描；下面无active/未选后继段落保留前一裁决时点。

2026-10-03 main完成机制§115/findings§294，只读历史实现与原task32交叉续行，0新模型/环境/GPU。
G2→G3确有谓词监督表示到完整LoRa/FM，G2时间平均与G3冻结边界保留；VisibleObject、LocalAction及SEOD/GOMQ的正负证据分列。
task32两前缀在180步目标谓词同为[true,false]，两策略却仅从s17成功；仅阶段标签不能区分这些成功续行成员，不推定唯一几何/接触原因。
main独占canonical记录/Git，无active实验或已选后继；下一推导继续解释可复用任务关系与实际自身控制，不扩数据或自动恢复旧训练。

2026-10-03 main完成机制§114/findings§293科学消费，直接核实际算子及全部96份预测；无新模型/数据/环境/GPU。
第二memory输入的常量作用在内部4各task均有益，新增风险−.002958；20的整体大收益97.57%仍保留在P0−F网络项。
事后F+b_input参照内部亦4/4改善（−.002831），不能把输入作用全归为抵消P0坏项；total仍未明确胜裸mu、32和各通道损失保留。
原件新增main_scientific_readback.py/.json；主CPU.604秒。停止小头/输入/比例扫描，不重开旧完整消费者，继续推导可调用控制关系。
canonical tracked/Git窗口已回main，无active实验或已选后继；以下执行记录已科学消费，不重复派发。

2026-10-03 `native_calibration_input_attribution_20261003` 完整执行与CPU读回完成：96/96视频、3199合法offset1区间、288不同episode有向pair及9597个成对query区间齐备，缺项0。
同一原P500/a4da650a冻结Calibration，在CPU FP32将每视频全部区间×P1/P0打包（实际batch34–182），完整50位置读出；全部预测保存后才读原标签评分，无Source/native/HDF/RGB/环境/梯度或新增数据。
fit20的network→total风险.074269888→.072928640，input增量−.001341249、18/20改善（28/37不利）；F=.072951705，因此总体对F仍近零。内部4的network→total为.133927320→.130969494，input增量−.002957826、4/4改善。
内部total相对F的有符号收益87.449%仍来自task20，该task收益97.569%保留在network项；input自身收益仅8.279%来自20，不能把全部P−F收益归于实际前后联系，也不能将P0−F称纯静态知识。
全部不利项保留：fit240/内部48pair的input all7分别50/2不利；内部两条为task20 demo42→43/44。task12/32 total motion6仍劣F，20的rotation/gripper仍劣F；32 total=.153714720仍劣mu=.111505513，四task total gripper均劣mu。
内部total对mu均值差−.001399031、描述95%[−.019071967,+.027847858]跨0；P0仍含真实出发视频及动态训练权重，重复memory可能偏离训练分布。本批不构成状态反馈、LoRA或闭环部署资格，不选择后继。
P1相对原P的全区间RMS=.000756863032，获取风险差/原§112/强mu/F/特权常量全部分列，无低位精度门槛或重复forward；float64精确风险恒等式最大残差4.16e−17。
实际小头/评分消费者PID178391含启动/加载/写入/退出13.165774秒exit0，forward3.964346秒；CPU后处理/原件核验exit0（主计算1.005468秒、不含启动），0/.25完整GPUh，CPU峰RSS902408KiB。本批无GPU分配，消费者已退出。
预读取CPU weights_only拒绝历史numpy RNG的失败已保留，随后只从可信完整checkpoint mmap消费P权重，未创建/恢复optimizer。所有旧原件不改，一次性脚本仅作外部原件，无新增或恢复canonical入口。
primary `/data1/user/ymdai/ember_runs/native_transition_action_calibration_20261003/analysis/input_attribution/`：completion/readback/verification、predictions、components.npz、episode_moments/episode_pairs、pair_interval_risks、comparison_table/per_task_channels、report及launch回执。
P500训练/native/原预测/冻结类=a4da650a，Source1000原训练=b8ea00e9；新一次性脚本留原件、实际读回仓库clean pushed2fbdf687，来源分别登记。新增原件约18.4MiB，含代码/文档/Git保守准入峰100MiB<2GiB，strg01 data1独立quota/shared复核通过。
整批科研记录Git推送与一次来源明确的回报后交回canonical tracked/Git窗口给main；实验session停止本项，无active计算/恢复或自动后继，main负责原件科学消费和下一取舍。

### 本批实际接手与登记历史（下述状态只表示当时时点）

2026-10-03 实验session01a0fabb-f7a0-7100-93d8-6a0f66055553已实际接手 `native_calibration_input_attribution_20261003` 的canonical tracked/Git独占窗口；main只读科学分析。
完整机制§113、findings§292、原生校准合同§6及当前Owner要求已读；本次只在原root analysis/input_attribution保存一次性脚本/预测/读回，不恢复训练入口。
P500/原native/预测实际来源均a4da650a；Source1000原训练b8ea00e9分列。96份demo42–45、3199合法offset1区间、288不同episode有向pair固定，尚未开始新小头forward。
CPU优先：P1/P0同FP32/同batch、全部先保存后评分；不加载source、不重建native/读HDF/RGB/环境或优化。预计含实现30–60分钟，硬限2GiB/至多.25完整GPUh。
strg01现场data1个人用量1169500940KiB、quota2147483648KiB、共享余89021464776704B；原root实占1896264KiB，新输出预计<100MiB，无大资产复制。整批Git/回报完成后交回窗口并停止本项。

以下§113登记段落保留派发前历史时点，实际授权/接手以上段为准。

2026-10-03 main登记机制§113/findings§292与原生校准设计§6的冻结输入归因；尚未派发/启动，main独占canonical记录/Git。
仅P500在已有96视频/3199区间读P1(H,H+)与P0(H,H)，精确分解第二memory作用及跨episode常量收益，无训练/新source/环境/扩数据。
预计含实现30–60分钟、2GiB、CPU可行则0GPU/硬限.25完整GPUh；P0分布改变与所有强参照/不利项均保留。
历史已核：Video Functional曾从未见target40的source学视频纠正，不能冒称新原理；没有选择cross-fitting或其它后继训练。
执行session接手后持独占窗口，整批完成/推送/回报后交回main；实际状态以随后接手记录为准。

2026-10-03 main完成机制§112/findings§291：用96份既存预测分解跨episode动作增量，0新模型/数据/环境/GPU。
fit20的常量分量收益近零；P/F未参与拟合的内部4task三益一损，但87.43%净收益来自task20，12/32运动项不利。
内部风险F .142491→.130964、裸mu .132369，未明显胜裸mu；保留有限任务偏差的可迁移正例，不称完整反馈或重开旧配方。
登记/脚本/逐episode及逐pair分量在native_transition_action_calibration_20261003/analysis/main_episode_innovation，主CPU.423秒、约1.31MiB。
CIRL原理/源码核对已结束；main独占canonical记录/Git，无active实验或已选后继，继续固定数据内推导。

2026-10-03 main完成固定17任务的源预测差转移CPU读回，机制§111/findings§290；没有模型/优化/环境或新增数据，0GPU。
F/P转移all7=.083982/.095743，较原匹配估计改善，但P仍17/17劣F；后补直接共享F_Q=.072443，两种转移均17/17不利。
事前登记、参照补项时点、逐行/逐task/分量及两次exit0原件在native_transition_action_calibration_20261003/analysis/main_residual_transfer。
两进程含启动/序列化合计约5.7秒、新增约3.21MB；不恢复动作残差、Pullback、校准门控或扫描，完整目标未完成。
当时main独占canonical记录/Git、无active实验；上述goal-inference核对现已结束，见§112，不代表采纳路线或训练许可。

2026-10-03 main完成本批科学消费（机制§110/findings§289），直接读取冻结干预算子、48份原continuous与全部12 full双RGB，0新增模型/环境/GPU。
图像选择具有有限因果作用，但10个原策略成败不同的条件仅救回弱recipient3条、同时丢强recipient4条；task13双向目标选择预测未完整成立。
actual pi_D依赖donor完整控制计算，图像V是上下文化prefix，不能把它当物体标签；读取与后续控制的共同作用仍须完整学习。
降低直接自身attention监督的主修复优先级，不增加蒸馏/层/质量/scale扫描，不从混合面板宣称部署收益。
新增原件为本root analysis/main_scientific_readback.json；canonical记录/Git窗口已回main，无active实验，继续固定数据内自主推导。

2026-10-03 `self_image_attention_transfer_20261003` 全部执行与CPU读回完成：48/48行、12 full双RGB/36 compact、全部goal/continuous/physical actions和18层×10flow紧凑作用齐备，无缺项。
task4/13/56按C、N、C←N、N←C成功数分别为[0,4,1,2]、[3,0,3,1]、[4,1,2,2]（各臂每task4行）；C←N总6/12、R5/G1/L2/churn3，N←C总5/12、R3/G2/L2/churn4。
C/N同批重放7/5与历史成功集合完全一致、churn0；所有teacher/scene/初始body/EEF/quat/gripper/predicate及绝对共同噪声配对通过，实际动作与独立continuous一致，保留25条失败及全部正例。
十二份init32 RGB已读：task4的C←N有晚段移出碗但未置盘，N←C失去原成功；task13的N←C/init32改为操作bbq却未入篮，init35新增成功，C←N三条成功保持；task56两混合臂同为{32,35}，保留C←N在33/34的丢失。
body中心/3cm/夹爪命令不当抓持证据；混合臂只有有限因果诊断身份，不能算部署分数。以上为执行/描述读回，科学解释与后继取舍由main消费全部原件后负责。
实际消费者clean pushed detached8e682f9045da22f8c6bad8fce92638f1821b3035；C训练/物化a0e0248d、N训练/物化2c630fb3、Source1000原训练b8ea00e9分别登记。48行消费者285.456734秒exit0、六worker每份source只加载一次。
本批含加载、失败profile、I/O与退出共.185489290357/3完整GPUh；正式已记录reserved峰10.0078125GiB/worker，失败profile峰未采集，不伪称全批VRAM峰已观测。合法batch4比batch1实测query/s快39.47%，正式两卡×3 persistent worker，按EGL/CUDA余量准入，遵守Owner显存吞吐要求。
新增root+工程树阶段实占1.801430GiB、暂存/代码/失败冻结树保守准入峰6.5/8GiB；strg01 data1独立quota/shared最终复核通过，所有消费者及六worker PID已退出，双节点本用户GPU进程0，未终止/暂停/reset他人。
primary `/data1/user/ymdai/ember_runs/self_image_attention_transfer_20261003/`：completion.json、readback.json、analysis/report.md/rows.jsonl/per_case.csv/effect_summary.json/identities.json/original_replay_rows.json/rgb，launch/gpu_ledger.json/final_release.json及四臂原results/captures。
三份任务专用入口/hooks及五处共享接入已退役，当前runtime拒绝旧诊断直接执行；Git/frozen/失败和合法原件保留。Git推送与一次来源明确整批回报完成后交回canonical窗口，实验session停止本项；无active计算/训练或自动后继，不恢复450/900配方。

### 本批此前工程过程（以下状态仅指其历史时点）

2026-10-03四臂48行已实际启动：PID/PGID3305517，clean pushed detached8e682f90，gpu02:0/1两卡、各3 persistent worker、完整task四init batch4。
启动前双节点/总卡数及strg01 data1独立quota/shared已现场核实；本用户0→2卡/上限6，个人用量1168940496KiB，新增root+工程树1.020GiB，保守峰6.5<8GiB。
实际消费者profile exit0：batch1=.716868秒、batch4=2.055971秒，query/s提高39.47%，reserved峰9.980469GiB；按EGL/CUDA余量与canonical每worker12GiB+2GiB准入使用每卡3份，未把torch峰值直接当整卡预算。
48行prepare exit0且固定teacher/scene/官方协议检查通过；所有阶段命令/环境/身份/失败回执在root launch，正式环境尚未读回。不训练、不重新物化，无自Queue或中间选点；当前窗口仍由实验session独占。

实际消费者8beacfe9已在gpu02:0运行并退出1（PID3193494、67.3346秒/.018704064GPUh，无环境行）。两可见相机各256 token、第三256 token false mask、200语言/共968 prefix、50 suffix与18层已实测。self-donor RMS .001342318/max .009053469被自设绝对RMS .001阈值拒绝；该阈值没有仓库依据且低于正常BF16量级，现按一个BF16 epsilon×原输出尺度修正检查，干预/模型计算保持原样，失败及原数值保留。新code重新push/freeze，仍仅原注册case内profile，无新环境smoke。

首次ad431240冻结CPU bank读取消费退出1、0GPUh，唯一差异为原bank spec绝对路径与新frozen路径（均6152B）；小JSON语义直接核对后只恢复历史provenance路径传给原inspector，其余source/factor/scope完整检查保持，旧原件不改。修复将重新push/freeze，不原地修改ad431240。

本批窄实现已接入canonical bank/evaluator：原sample_actions十步积分与18层原生RoPE/GQA保持，两bank因子每replan各打包一次，四panel由同一resident worker/source依次消费，动态队列仍由canonical owner执行。
针对CPU语法、四route、极小donor图像质量稳定条件分布、非图像概率保留和Value恒等式检查已执行；原bank metadata inspector要求clean detached，首次在工程树读取被其guard拒绝（0GPUh），将按原规范在冻结树消费。
结构自审：三份任务专用源码约600行和五处窄接入，复用官方积分、bank loader、run_worker、队列/launcher/aggregate；不构造第二policy/evaluator。旧validate_episode_adapter_fields复杂度32→33的单个诊断dispatch是有界局部例外，48行读回后连同专用hooks/入口和公共接入全部退役，Git/frozen原件保留。

2026-10-03 实验session01a0fabb-f7a0-7100-93d8-6a0f66055553已实际接手 `self_image_attention_transfer_20261003` 的canonical tracked/Git独占窗口；main只读科学分析。
完整合同、机制§108–109/findings§287–288及Owner资源/信息墙要求已读；独占工程树为`/data1/user/ymdai/projects/EMBER-self-image-attention-transfer`，基线d15cba34。
当前在实现同输入、同prefix的18层/50slot/10flow图像内部条件分布传递；原四臂48行/12 full/36 compact及teacher/scene/RNG固定。尚未启动模型或环境。
strg01 data1个人用量1167868424KiB、quota2147483648KiB、共享余89.023TB；本批预计新增峰6.5GiB<8GiB，硬限3完整GPUh。正式消费者将由clean pushed detached来源运行；完整交付后退役本次入口/hooks并交回窗口。

## 本批授权与登记（现已执行完成）

main已在机制§109/findings§288登记唯一[自身图像读取分布交叉诊断](docs/designs/self_image_attention_transfer_diagnostic.md)。
只交换两冻结策略同一自身输入下的图像内部attention，保留recipient图像总质量/非图像分量及其执行权重；不由错物体现象直接添加grounding loss。
固定task4/13/56×init32–35×四臂，共48行/12 full/36 compact，复用原C450/N450同teacher bank与scene/RNG。
不训练、不扩数据、无官方held/Test；混合臂只作因果辨识，不能当作一次LoRA方法成绩，结果不自动接新训练或扫描。
预计含工程3–5小时、硬限3完整GPUh/8GiB；实际接手/launch/完成以上方执行记录为准，本面板不再active。
整批Git回报后canonical窗口交回main，其消费并负责科学裁决。下方无active段落保留各自历史时点。

main完成机制§108/findings§287的CPU对应读回，无新模型/梯度/环境/GPU：17当前fit任务、272pair、9528行/2382个不同query帧。
复用旧绝对1NN索引，动作只读合法任务现有P/F预测；当前held与内部留task均排除。
相同选帧下P−F教师风险−.014253，跨episode+.008147，17task全部反转；夹爪贡献约85.9%，运动亦不利。
匹配本身相对各自teacher均值全部17task有益；获取与可迁移控制不能互相代替，具体限制见机制§108。
原件在native_transition_action_calibration_20261003/analysis/main_correspondence_transfer；main仍持窗口、无active实验，继续固定数据内推导。

main已完成控制内容调制读写的科学消费与裁决，见机制§107/findings§286；本fresh450配方关闭，没有active实验或新GPU计算。
直接核实际2c630fb3科学代码、12份FM/8份q与native原件、新旧1632份continuous及六份配对双RGB捕获；
派生原件为本批`analysis/main_scientific_readback.json`，没有新forward、标签或环境运行。
correct400=122、seen144=77；q八训练teacher均改善但98.977%的收益来自五步平均项，不能把旧独立P的内部留task正例移植成新Gamma的held获取资格。
A28前5的.001379改善由夹爪+.001675抵消运动六维−.000296；真实得失仍跨对象选择、获取及后续调用。
关闭本构造续900、辅助/门控小扫与Reader/Pullback回退；不能由一次阴性断言所有动作校准或视频编译不可能。
canonical tracked/Git窗口回main，实验session已停止。唯一实现及完整450恢复资产暂保留至下一方法的接口/生命周期取舍，当前不运行。
Owner自主推进与不扩数据约束持续；main继续推导视频中的状态—作用关系如何变成自身可调用的控制，不能以未形成后继为理由结束等待。

### 本批执行事实（科学裁决以上段为准）

2026-10-03控制内容调制读写整批执行与原件读回已完成：fresh450、六完整ECP、correct400=122/400、seen144=77/144（train24=36/96、support12=41/48），544行/44 full/500 compact及原A28的12 FM/8被动条件全部齐备、无缺项。
相对C450 correct R98/G24/L39、seen R69/G8/L22；绝对性能低于强MT153及成熟T2340的161，全部高低参照/逐task/逐行得失保留，不能由q拟合或内部MSE宣布有效。
训练/物化/A28/实际环境消费者均为clean pushed detached2c630fb3，均exit0；本批11.471727418/32完整GPUh，正式reserved峰41.296875GiB。三次真实profile含frame40容量OOM已计费，world6仅快1.36%且多50%卡；评测按4+2卡/每卡3 persistent replicas利用显存与吞吐，无额外案例。
新增实占含工程树51.444GiB，临时/缓存/冻结树保守准入峰72.06/80GiB；strg01 data1独立quota及shared核验齐备。16个实际consumer PID全部退出、双节点本用户GPU进程0，没有影响他人。
首次CPU prepare漏传既有asset环境路径、CPU读回误把成功停止后未执行的末尾planned前5全当physical，都已在原语义内修复并保留失败；原frozen/分数/产物未改、0新模型或环境补跑。所有544行实际prefix与独立continuous actions一致。
primary `/data1/user/ymdai/ember_runs/control_calibrated_read_write_20261003/`：completion.json、readback.json、analysis/closed_loop_readback.json/functional_readback.json/checkpoint_readback.json/report.md/comparison_table.csv/per_task_scores.csv/ability_changes.json及launch/gpu_ledger.json/final_release.json。
当前无active计算，保留唯一新canonical实现与完整450恢复能力，等待main科学消费；不自动selected/续900/扫描/Reader蒸馏/controls/Test/RL。本次整批Git推送与一次来源明确的回报完成时，canonical tracked/Git窗口交回main，实验session停止本批、不再写tracked或新增计算。

### 本批登记及执行过程（以下“正在/尚未/已接手”仅指历史时点）

main已在机制§106/findings§285登记唯一后继[控制内容调制读写](docs/designs/control_calibrated_read_write_design.md)。
完整fresh450：同一Gamma q既受合法5步动作监督，又直接调制实际A/B Value，公共native/c/d及跨episode真实FM保持。
原36task/50teacher/50,400主query流，不加载诊断P权重；固定450的correct400、seen144、原A28，强MT及成熟T高点完整保留。
预计含工程6–10小时、硬限32完整GPUh/80GiB；无自动续训、扫描、Reader蒸馏、Test/RL或controls。
2026-10-03实验session01a0fabb-f7a0-7100-93d8-6a0f66055553已实际接手canonical tracked/Git窗口，main从此只读科学分析。
完整新合同、机制§106/findings§285及当前科学/资源规则已读；开始从最新main隔离工程实施，先核完整缓存/标签/ECP/物化metadata峰值。
工程独占树`/data1/user/ymdai/projects/EMBER-control-calibrated-rw`已接入canonical Gamma/38组Ua/Ub与q余切；纯F0缓存和训练labels分离。
元数据1800 train/400 validation共73642帧，预计新增峰72.06GiB<80；strg01 quota2TiB、个人实际runs593542635520B/projects539566137344B、共享89.078TB，资源可准入。
针对输入墙/短gap/唯一辅助计数/本地及remote复合信用和400/144实际捕获路由检查通过；50,400 query/1800 teacher事件CPU审计通过。
main已核对原C450/C900/MT300全144行与当前scope，确认原scene/RNG/固定teacher映射完全一致；design中teacher46–49为文字误记，已勘误为scope seed20260928、0–49池固定四teacher及scene_canonical144，sealed原件未改。seen继续按原144行配对。
本批已实际启动pure F0缓存：clean pushed detached e314d2271a8fe1e18533c84fac4af77e811f8518，gpu02:0/1/2/3四worker，精确command/环境/PGID/准入见root launch/bare_cache_shard*_consumer.json；一次持续等待退出，无自Queue。176既存train-native只提取H0/mu/indices到action-free store（6200帧），旧labels/原件未改，不加载P/F权重。实际费用自加载起累计，完整fresh450尚未启动。仅执行fresh450和固定544环境行/A28，新增canonical实现及完整450交付后保留待裁决。
四cache实际exit0，2200视频/73642帧全部齐备（新增67442帧、复用6200），14.064GiB，含加载/写入/退出累计1.119620866GPUh。
原初始packing记录部分尾块不足256，不能声称它证明256更快；实际两worker峰40.42/41.08GiB、另两24.39GiB。
已窄修profile跨实际task帧组收齐完整候选、只计完整batch，CPU实际消费检查保持963帧各一次，无新Source forward、旧原件未改。
三个有界profile已结束：同macro85/105帧/28query，world4/frame32为33.097秒、峰39.289GiB；world6/frame32为32.652秒、只快1.36%，50%更多卡无充分持续吞吐收益。world4/frame40在native反传OOM（物理占用44.06GiB、余155MiB），exit1与.093356145GPUh保留，非科学阴性，无第4次profile。
正式fresh450已完成、PID/PGID1843944退出0：gpu02四卡/world4/micro28/frame32、clean pushed detached2c630fb3940c16365ac9a9a672b1f632e8d31e51（root/frozen_run）。全部状态fresh，无profile/P/F权重；0/90/180/270/360/450六完整ECP及Gamma/全部76 U权重、optimizer/scheduler/sampler/rank RNG已CPU实际读回。50400主query、55858合法aux区间/279290位置准确，辅助每condition一次、Gamma-only与same-version记录齐备。
正式消费者含加载/保存/退出7465.552秒、8.295057918GPUh，平均完整更新16.408秒、reserved峰41.296875GiB；加载/profile/失败/cache等本批累计9.691814519GPUh。未读中间分数，无选择或自动续训。
450读回实际已启动：400物化PID2356778用gpu02:0/1/2/3，144物化PID2358187用gpu02:6，A28 PID2359958用gpu02:7；三份准入/精确request在root launch，NoGrad真实native framechunk128、各resident一次source加载。双节点本用户0→6卡/总上限6与strg01 data1现场检查通过，root当前26.647GiB<80。seen使用原scope/scene/RNG，已就绪评测及时接续；全部消费者仍仅450，当前仍持写入窗口，main只读。
三读出均已exit0：400/144完整38-target bank sealed，A28为8 full/4 public FM及8被动条件，实际源码均2c630fb3；累计10.104028164GPUh。CPU原件功能读回12/8完整、全部399个不利项保留，没有新增预测。
首次correct400 CPU prepare因请求漏传既有LIBERO assets路径exit1（2.029秒、0GPUh），未启动模型/环境；只修启动环境引用原data1资产，不改源码、原件或科学范围。失败回执保留，重prepare与seen prepare均exit0。
固定544行现已实际启动：correct400 PID2427254用gpu02:0/1/2/3，seen144 PID2430407用gpu02:6/7；各卡3 persistent replicas/cost-balanced long-first，双节点现场本用户0→6卡、data1 quota/shared准入通过。相对历史5+1约729/1534秒，按真实负载分4+2提高整批吞吐；无额外smoke，原task/state/video/scene/RNG准备配对检查通过。当前root47.933GiB、保守峰72.06<80；一次持续等待各消费者退出，无自Queue或中间选点，窗口仍由实验session独占。
下方§105“没有新训练”属于其科学消费完成时点，不覆盖这个新登记；旧获取批次仍已完成，不恢复其运行。

main已完成本批源码与全部176份实际预测的科学消费，见机制§105/findings§284。
内部P对F风险降19.857%、对裸mu降13.728%；P−mu四task簇区间跨0，task32全部四视频较mu差。
精确五步误差分解表明内部P对F改善99.001%来自区间平均指令；取得部分可迁移控制内容，未建立精细时序或自身控制规律。
原件补充`analysis/main_action_content_readback.json`，无新forward/标签/环境/GPU。
canonical tracked/Git窗口已实际回main，实验session本项停止、无active计算；main继续推导同一LoRA如何消费这些内容。
没有选择部署P、Reader蒸馏、固定Pullback或新训练；本项通过不自动生成后继合同，也不结束自主推进。
下方为已完成获取批次的执行事实，原“待消费”状态以本段为准。

2026-10-03唯一native_transition_action_calibration_20261003批次已完成：176/176视频、6,200真实采样帧、5,882合法offset1五步区间。
P/F各fresh500及80,000区间曝光，0/250/500完整恢复点、全部mu/F/P预测/标签/索引/mask及CPU实际读回齐备；无Writer/LoRA/环境/官方Val/Test。
内部留task[0,12,20,32]无gamma梯度；等task all7为mu .132368525、F .142491026、P .114196534，4/4 task相对F改善，预注册获取预测通过。
fit20新episode为mu .119912357、F .072951705、P .060660000，20/20 task相对F改善；训练池为.122514996/.062229756/.049934530。
不利项完整保留：task32全部demo42–45的P劣于mu；内部整体rotation3 P−F=+.003823469，task12 motion6/rotation3及task20 gripper劣于F。
fit20新episode整体rotation3 P−F=+.000554396，4/80视频P劣于F；训练池2/80视频亦劣于F，未由task平均抹去。
内部P−F任务cluster描述95%[-.042266626,-.019228437]，P−mu[-.036102539,+.007058999]跨0（20,000、seed20261003）；四task不建立广泛迁移或LoRA有效性结论。
实际native/学习/固定预测clean pushed detached a4da650a4e1a00a2d4322569339fbf4bce4d5be4；裸aligned Source1000原训练b8ea00e9fbb86742ef076bac9dd35c5314cd5aed，只读复用。
一次source加载、共同H/mu缓存；合法帧64/128实测后选128，约21frame/s，native reserved峰25.078GiB，未显示相对暖64的稳定显著提速。
两gamma每臂全部160查询打包320，正式500平均.017647秒/更新、学习reserved峰1.918GiB；不增加逻辑样本或跨更新合批填显存。
累计.134531061782/3完整GPUh，含首NUMA接口失败15.510秒、加载/profile/训练/物化/退出；最终GPU及CPUexit0，双节点本用户GPU进程0、两个consumer PID均消失。
root阶段实占2.051GiB、保守新增峰2.3/8GiB；data1个人约1.141TB/2TiB quota、共享89.082TB；全部新增写data1，未复制大source/dataset。
任务专用入口/模型/测试三份tracked源码已退役，未改共享运行面；原失败frozen21b4d22b及实际frozen a4da650a/Git/全部科学原件保留。
primary root `/data1/user/ymdai/ember_runs/native_transition_action_calibration_20261003/`：completion.json、GPU_completion.json、analysis/readback.json/videos.json/intervals.jsonl/adverse_intervals.jsonl/task_scores.csv/task_cluster_intervals.json/checkpoint_readback.json、launch/gpu_ledger.json/gpu_release.json。
本交付commit推送及唯一整批回报后canonical tracked/Git窗口交回main，实验session停止本项；没有自动后继、选点/延长、Writer/fresh/环境/held/controls/Test/RL许可。
Owner持续自主推进；主讨论消费本获取原件后负责理论解释与下一科学取舍，完整EMBER目标尚未完成。

### 本批登记及实际实施过程（以下“正在/尚未”仅指历史时点）

Owner于2026-10-02在完成交接与现状讨论后明确恢复自主推进，不对分析设置时间限制；本项覆盖当日上午分析后暂停的要求。
主讨论当前负责在既有数据内统合机制与历史证据、选择可失败的有界干预，再由唯一实验session完成工程与运行。
目标是在相对稳定的情况下大幅超过强MT，不把严格保持T的架构、稳定性或逐例成功作为新门槛。
2026-10-03Owner再次要求没有推导出后继时继续推导，不能以此结束等待。main已完成机制§104/findings§283，
登记唯一[原生可见转移动作校准辨识](docs/designs/native_transition_action_calibration_diagnostic.md)。
P/F使用相同裸aligned Source1000、native点估计和残差读出，只改变第二memory为到达H/出发H。
现有train24中fit20×demo16–19，内部留task[0,12,20,32]无本次读出梯度，全部24×demo42–45一次读回；共176条已有视频。
两臂fresh500/各80,000区间曝光；仅gamma学习，无LoRA/Writer训练、环境、官方Val/Test或新数据来源。
预计70–140分钟含工程，3完整GPUh/8GiB硬限；原生动作信息获取不等于一次LoRA控制，后继没有自动许可。
2026-10-03实验session 01a0fabb-f7a0-7100-93d8-6a0f66055553已实际接手canonical tracked/Git窗口，main只读科学分析。
完整新合同及机制§104/findings§283已读，正在核算176条既存video的metadata帧数/合法区间与缓存峰值；尚未launch。
从clean pushed ed97fba3开始；只实施本500更新获取诊断，不恢复旧§103或任何Writer/环境计算。
metadata实际为6,200个采样帧、5,882个完整offset1五步区间，142个不足5的真实末段不产生标签；FP32缓存约1.184GiB，预计新增峰2.284/8GiB。
strg01已核data1个人约1.139TB/2TiB quota、共享约89.089TB；旧资产只读引用，无source/dataset复制。
独占codex/native-transition-action-calibration worktree复用原native、source/data/tokenizer owner；临时入口/模型/四项检查约410行，未改共享运行面。
CPU四项检查与实际入口导入通过，涵盖同seed7零残差、P包含F函数类、双臂打包梯度与500×160共同fit20事件流。
首次实际入口导入缺少既有facade初始化，0GPUh失败及修复保留；没有创建source/CUDA/环境，科学计算不变。
结构guard无新增hard违约；native分块/缓存接入复杂度和已有peer数为review信号，本任务有界使用并在500读回后退役全部三份临时源码。
启动前再次live核两节点与data1；将用既有合法帧比较64/128、每臂全部160区间比较双臂打包/顺序，profile后恢复fresh初始化/RNG。
当前实施/CPU验证已完成，待集成push及clean detached冻结；尚无本批GPU计算。
实现现已集成push，实际消费者为clean detached21b4d22b，PID1217776在gpu02:7启动同一次native/cache/双臂500/固定读回。
启动前双节点现场本用户0→1卡、合计上限6；所选卡只有他人148MiB且util0的context，45908MiB余量可共驻，未改变他人进程。
strg01 data1实占约1.140TB/2TiB quota、个人余量986.738GiB、共享89.086TB；当前root约.510GiB，预计新增峰2.284/8GiB。
一次source加载后两臂共用H/mu；真实64/128帧与每臂160区间profile计费并恢复初值/RNG，正常执行只持续等退出，不自Queue。
首21b4消费者在native/标签/更新前因NUMA绑定缺少device参数退出1，计15.510GPU秒（.004308GPUh）；失败frozen/log/ledger保留。
已核原NUMA owner接口并修正为可见设备0，实际调用签名及source元数据消费者CPU检查后新push/freeze继续；非科学阴性，无新增数据/模型/矩阵。
修复已集成push为a4da650a，新clean detached frozen_fixed的PID1251875已在gpu02:7启动；首失败15.510秒从同一3GPUh余额扣除。
重新现场双节点本用户0→1卡、data1 quota/shared准入通过；信息墙和完整176条offset1 metadata读回齐备，只等实际消费者退出。
下方§103的“停止/无active计算/下一判断”保留其收束时点，不恢复旧批次，也不覆盖上述新登记。
最新主讨论已完成自身功能信用的原件消费及机制§103/findings§282：128条continuous、24真实双RGB和32功能PT直接读回。
live/stop学生20/24、Reader19/23，live相对stop R18/G2/L6，新增query信用没有净控制收益；辅助头也未建立强功能教师。
stop24及其相对历史父17的全部保持/新增7保留，旧S20因native重构范围不能作为严格辅助因果对照。
关闭本构造延长、扩头/换层/调权、蒸馏和正式fresh理由；不由局部赢家自动接探针或选新方法。
本项实验与科学消费均完成，canonical tracked/Git窗口由main独占；实验session停止，无active计算。
自主推进授权持续；main下一判断回到可由新教学决定的动作修正如何经共享学习成为自身完整控制，完整目标尚未解决。
以下为本项及此前各批的实际执行过程，原有“正在/已接手”仅对应当时时点：
main现已直接消费下述视频来源批的16份continuous、四份RGB及真实构造代码，完成机制§101/findings§280。
后段单独造成“得到放壶、失去开炉”的预测未获支持；早段增量也能补出完整操作，但仍依赖完整父LoRA与后段地址。
关闭依此事件分区选择阶段门控、部署E或继续扫切点/层位/scale；历史完整S与新增量约.82%–1.23%重构差限制精确归因。
本批及其原件消费已完成，canonical tracked/Git窗口回main；该批无自动后继。
主讨论进一步完成机制§102/findings§281，登记唯一有界学习辨识
[自身执行特征的功能信用](docs/designs/state_coupled_functional_credit_diagnostic.md)。
旧T2340/固定公共native/A/B0/原四train任务八teacher与A28逻辑流，两臂各64次共享学习；
辅助头读同一Z和当前生成LoRA的执行hidden，仅live/stop其query反传，主真实FM和其它信用完全相同，无蒸馏。
固定64各student/Reader诊断32行，共128新增、24 full/104 compact；只有student计为一次LoRA形式，原parent/S/P/D复用。
预计75–135分钟含工程，硬限3完整GPUh/8GiB，所有输出data1；无held、Test、正式fresh或自动续训/扫描。
本次检验新增自身hidden信用是否改善完整学生，不把Reader拟合或非零梯度当作成功，也不声称主要根因已经确定。
2026-10-02实验session 01a0fabb-f7a0-7100-93d8-6a0f66055553已实际接手canonical tracked/Git窗口；main只读科学分析。
从clean pushed af1a2c0f创建独占codex/state-coupled-functional-credit工程worktree，正在实施同次FM隐藏捕获、LoRA/Z联合VJP和固定读回。
四项针对性CPU检查通过：原M/Z递推、gamma初始化/零Value、live/stop同forward与双余切对直接反传、7维诊断hook移除。
首次CPU fixture误用了所有memory token相同的Value，使query梯度合法为零；已修正为异质固定memory，0GPUh失败保留。
临时约770行增量由原T/FM/银行/环境owner承接；bank仅4行复用已注入LoRA以避免第二次source加载，
原长rollout仅窄接8-case矩阵，新127行冻结读取保留同一canonical环境。结构guard的旧bank>800/矩阵复杂度26信号采用此批有界例外；
整批交付移除入口、两个诊断模块、测试和两处临时接入，不建立长期平行trainer/evaluator。
启动前将实测microbatch14/28（每臂最多两次丢弃更新并恢复初值/RNG），native帧块64优于旧32且不增样本；六次完整native加两份既存H/K。
实现/实际消费者clean pushed detached 795e86a0；实际live PID160306在gpu02:7已启动native/profile，stop按native完成事件在gpu02:1接同一特征。
现场双节点核验本用户0→2卡、合计上限6；两卡原148MiB/util0 CUDA context可共驻，不动他人。
strg01 data1个人实占约1.135TB/2TiB quota，共享89.098TB，预计新增峰6/8GiB。
第一次等待器误用Python3.8无pidfd接口，退出后无自有消费者/native/更新；改用既有3.12，保守101GPU秒计入总3GPUh，失败回执保留。
六条新public native一次全帧打包、两份task32既存完整H/K复用已完成；两臂各两次真实profile更新后均恢复父Writer/gamma/optimizer/RNG。
2026-10-02正式64已实际启动，live gpu02:7 PID160306、stop gpu02:1 PID161725；八条件每步224 query、各1/8保持。
两臂实测选择micro28（单condition全部query），约21.93/21.78秒每完整更新，reserved峰17.70/17.55GiB；
原micro14为26.15/22.39秒含首轮启动差，不宣称稳定19%提速。两臂独立并行，native最大完整视频37帧、framechunk64，无新样本填显存。
两臂均已完成正式64、full checkpoint和全部A28/B20 student/Reader功能预测，实际训练代码795e86a0。
第一次闭环读取在任何case启动前因canonical environment tasks不含global_task_id而退出1；Prepared证据中冗余consumer字段同时修正。
原训练/预测/银行/日志保留，非科学阴性；独占分支针对128注册case映射、adapter身份与24 full/104 compact CPU核验通过。
修复集成push为1449c908，新clean detached frozen_readback仅读取完整64，不创建optimizer、不训练、不再native或重复已存功能预测。
2026-10-02T22:43:46+08:00新固定读取已实际运行并完成，live PID389135 gpu02:7、stop PID389136 gpu02:1，最后22:57:05退出。
两臂各student/Reader32行，共128/128；24 full/104 compact，continuous/goal/actual actions及全部功能PT齐备。
CPU实际读回通过：原scene/root7与绝对噪声时钟匹配，初态body/EEF/夹爪/谓词误差全0，50×7/十步/前5/成功停与实际命令对应。
成功live学生20/Reader19，stop学生24/Reader23；task0/12/20/32依次live学生8/4/3/5、stop学生8/4/6/6，Reader7/4/4/4与8/4/5/6。
live学生对stop R18/G2/L6、churn8：新增32/17/init2、3；丢失20/42/init0、1、2，32/17/init1，32/43/init0、3。
对历史父，live R14/G6/L3、stop R17/G7/L0；对S为16/4/4与18/6/2；完整逐task/teacher、P/D与不利行均保留。
全部32个新task32行最终开炉；失败均没有最终placement。live17成功[0,2,3]、stop17[0,1]，live43[1,2]、stop43[0,1,2,3]。
Reader17 live[1]/stop[0,1,2]，Reader43两臂均[0,1,3]；原S43/init2只place无stove，新两学生均完成炉/放置，但两Reader该行未放置。
live学生43/init3在253抬高>3cm、最大10.556cm仍520失败；live Reader17/init0抬高343、最大3.568cm仍失败。3cm/中心/命令不证明抓持。
全部24份真实双RGB已看四组contact sheet，图片仅来自保存的replan，末RGB可能比terminal早4步；不补造compact/旧S43图。
A28前5FM live学生/Reader=.127706/.127674、stop=.127755/.127724；B20十步前5=.109498/.109558与.109493/.109545。
旧parent/S/P/D同B20为.117854/.109665/.109496/.110432，全50/有效future、motion6/gripper1与逐query不利项均保存。
旧A28只剩端点aggregate，未假称逐query完整匹配；离线MSE不用于归因具体失败。历史parent B重构relative0.49–3.52%，两新臂同起点；OZ closure约.008–.010%。
两份完整64恢复点含Writer/gamma/optimizer64/sampler64/RNG/topology/schema，CPU读回确认；新读取无optimizer/native/新预测。
累计含失败/加载/profile/正式更新/退出1.377359781265GPUh，硬限3；root阶段观察4.451GiB、保守新增峰6/8GiB。
原795消费者metadata退出各1且0环境行，首Python3.8等待器失败保守100.862GPU秒，全部失败保留，非性能阴性；最终读取两臂及CPU均exit0。
双节点退出现场本用户GPU进程0，四个消费者PID已消失；个人data1实占约1.139TB/2TiB quota、共享89.093TB，未改变他人进程。
正式训练micro28约22秒/224逻辑query，native framechunk64全帧打包、两臂并行；读取最大batch8、5959完整生成chunk，合计forward约734.366s。
原训练/预测/银行795e86a0与新读取1449c908分列；原父训练e2afbfd7、旧学习092a0ae8、旧读取5313257c保持。
专用入口/两个诊断模块/测试及共享临时hooks已退役，归档guard拒绝本批合同；退役后两项原canonical CPU检查与真实contract拒绝检查通过。
本批计算和原件读回已完成，以下交付提交push及一次整批回报后释放canonical tracked/Git窗口，主讨论接续科学裁决。
无训练/环境active，无自动fresh/续训/扫描/额外teacher/init/held/controls/Test/RL。
原件唯一root `/data1/user/ymdai/ember_runs/state_coupled_functional_credit_20261002/`：completion.json、analysis/readback.json、rows.jsonl、pairing.json、functional_comparison.json、checkpoint_readback.json、RGB_contact_sheets.json、launch/gpu_ledger.json、gpu_release.json。

主讨论完成上一批原件与机制§100后，登记唯一后继冻结分析：
[task32已学修正的视频来源](docs/designs/task32_learned_video_segment_diagnostic.md)。
同一共享S64在teacher17/43上，仅按真实教学炉面首次变红的固定采样帧80/95，分解早/后转移的完整38处学习增量。
两teacher×E/L×原四init=16新行，父/S原16参照复用；四full/十二compact，只有两条旧父native重读，无学习/新标签/held。
它辨识S43放壶收益与开炉损害的实际视频来源；若不能按此事件定位，不继续切点/层位扫描或自动开训练。
预计35–75分钟含工程，硬限1完整GPUh/4GiB，预计新增峰值3GiB；唯一root为
`/data1/user/ymdai/ember_runs/task32_learned_video_segment_20261002/`。
本批已完成16/16新增行、4 full/12 compact与全行continuous/goal/actions；两条public native重读及四bank/H/全38 K齐备。
实验session 01a0fabb-f7a0-7100-93d8-6a0f66055553从clean pushed a2c39b5a接手窗口、独占分支实施；四项CPU检查通过。
构造/实际消费者clean pushed detached 8b90f31cd3af60045b14ea73e6320529edd73d5b，原父训练e2afbfd7、学习092a0ae8、旧读取5313257c分别保留。
2026-10-02T19:47:09.740844+08:00–2026-10-02T19:52:05.019625+08:00在gpu02:7运行PID3861322，消费者/CPU读回均exit0；GPU进程与占用已释放，无GPU失败。
一次source加载、native全部51/49帧分别18.920/20.463 frame/s、闭环最大batch16；1296个完整50×7生成chunk、forward9.190 chunk/s。
全部授权帧/case已打包，未为填显存增计算；未测试第二卡，不宣称其速度劣势。峰allocated13.983/reserved15.486GiB。
含加载/退出累计0.082021883726GPUh；root阶段观察1.280GiB、保守新增峰3/4GiB。现场strg01 data1个人实占约1.134TB、quota2TiB、共享89.099TB；本用户0→1卡，上限6，148MiB/util0共驻未动他人。
17 E/L成功[0,1,2]/[0,1]（3/4、2/4），43 E/L为[0,1,2]/[1]（3/4、1/4）。对父R/G/L依次1/2/0、1/1/0、2/1/0、1/0/1；对完整S为2/1/0、1/1/1、2/1/1、1/0/2。
全部16行最终炉目标真，7个不利行均为placement未满足；保留17 E3、L2/3、43 E3、L0/2/3。
原S43/init2仅放置无开炉，新E在133/296开炉/放置并成功，新L145开炉却未放置；43/init3完整S成功，E/L均未保留。
四份实际双RGB读回；E43/init3在474才描述性>3cm、最大4.628cm，520仍未放置；3cm/中心/闭合命令不证明抓持，旧S43无RGB仍缺。
封存初态body/EEF/夹爪/谓词误差全部0、scene/root7与绝对policy噪声时钟全行匹配；official256→224、十步、前5、成功停通过。
同一重读特征上的E+L closure relativeL2为0.000224/0.000203；对旧parent/S整组重构为0.78–1.07%，与旧S-parent增量合计差1.229%/0.824%。
正常BF16/TF32/batch/reduction差保留，不称逐bit原bank复现，也不重跑完整S或作dtype/batch小扫。
任务专用入口/测试与canonical episode-context hook已退役，原运行树/Git/原件/0GPUh CPU fixture失败保留；current runtime拒绝已归档本批合同。
此Git交付完成后仅一次整批回报主讨论并交回canonical tracked/Git窗口；实验session停止本项，无自动后继、训练、FM、held/controls/Test/RL。
原件索引：上述root的completion.json、construction_readback.json、analysis/readback.json、analysis/rows.jsonl、analysis/pairing.json、launch/gpu_ledger.json。


以下为更早已完成的冻结分析，当前合同与派发状态只看本节顶部：
[task32已学修正的执行投影分组](docs/designs/task32_learned_operator_groups_diagnostic.md)。
仅旧D17/S43的Q/非Q学习增量、两teacher×两臂×四init共16新行，复用原parent/完整学习行。
预算1完整GPUh/4GiB；16/16新增行、4 full/12 compact及全行continuous/goal/actions已齐，GPU消费者与CPU读回均exit0。
实验session（01a0fabb-f7a0-7100-93d8-6a0f66055553）当时从clean pushed c17d4c80接手并创建独占工程worktree；
整批已在8a6f8d6b交付并释放canonical tracked/Git窗口，现由main维护科研记录。
9项针对性CPU检查通过；实现/构造/读取clean pushed detached 22faa1966f43b14555088af61dc445f67f95ca5b。
四份完整38-target银行已构造；2026-10-02T18:55:21.952181+08:00实际在gpu02:7启动16合法case最大packing、一次source加载，
消费者PID3624517，含加载/退出预算1GPUh；四full/十二compact与全行continuous/goal/actions由同一canonical consumer捕获。
现场本用户既有GPU为0，新增1，总卡数准入上限6；gpu7低占用148MiB/util0可共驻，未改变他人进程。
data1现场XFS个人实占1.0T、quota2T/limit2.0T，共享82T；新增峰值预计2.5GiB、硬限4GiB。
整批实际完成：旧T2340、teacher17/D64的Q/R各2/4，成功init为[1,2]/[0,1]；teacher43/S64的Q/R为3/4、2/4，成功init为[0,1,2]/[0,1]。
对父分别R/G/L=1/1/0、1/1/0、2/1/0、2/0/0；对完整学习臂为2/0/2、2/0/2、2/1/1、2/0/1。
两组均未保留D17/init3及S43/init3的新增完整成功；D17/init0只R保留、init2只Q保留，不能净分选全局赢家。
S43/init2原完整S“放置但没开炉”，新Q开炉141/放置281并成功，新R开炉121但没有放置；原件正反均保留。
全部16条最终开炉为真，7条完整失败均未放置；Q17/init3、R17/init2/3、R43/init3有描述性>3cm抬高却未放置。
四条真实双RGB已读回，3cm/中心/闭合命令仍不证明抓持或接触；不据组名宣布感知/运动根因。
封存scene/body/EEF/夹爪/谓词初始误差均0、root7逻辑噪声全行匹配；实际消费者50×7/十步/前5/成功停合同通过。
GPU消费者18:55:21–19:00:05(gpu02:7) exit0，含加载与退出累计0.078771065871GPUh，无GPU加载/运行失败。
一次source加载、最大batch16、1,312个完整50×7生成chunk；实际forward约9.2093 chunk/s，rollout249.751s；
峰allocated11.4464755/reserved12.7695313GiB，一次NVML实际util100%。全部授权case已打包，无额外case可放大；未测试第二卡，不作其速度优劣声明。
root阶段观察约1.193GiB，保守新增峰值2.5/4GiB；CPU fixture/准入/读取脚本的失败均0GPUh并留回执。
双节点退出快照本用户GPU进程0，消费者PID3624517已消失，未修改他人进程；没有新训练、Writer/native编译、离线FM或held/controls读取。
9项实现CPU检查和实际16行验证完成；专用入口/测试及重复init hook已从active tree退役，canonical rollout接口恢复，原件与22faa196 frozen代码保留。
退役后旧owner两项检查及active运行拒绝已归档新合同的检查通过；只增加退役guard，不建立长期并行consumer。
completion/readback/全16行/四full/RGL/费用/释放/失败与实际冻结身份统一在
`/data1/user/ymdai/ember_runs/task32_learned_operator_groups_20261002/`。
本批工程、运行与读回结束；一次整批消息发给主讨论后释放canonical tracked/Git窗口并停止，不自动追加任何计算。
main已直接读取全部16份continuous及四份双RGB，完整解释见机制§100/findings§279。
D17/init0只R保留新增成功、init2只Q保留；D17及S43的init3均只有完整更新成功，S43/init2却只Q成功。
有限成功交互既有+1也有−1，不能把它等同神经Hessian或能力比例；关闭按Q/R分组选修复和继续拆投影的路线。
下一判断仍须把实际视频特征的学习改变与自身状态调用联系起来，不能从本次互补直接选择新架构或正式训练。
Owner最新明确：MT只是参照，EMBER应大幅超过MT；不能用MT同样失败降低EMBER自身失败案例的研究优先级。
主讨论此前据错误筛选标准提出的两条MT补测，在派发前撤销，0新增GPU/环境、无新run root或实现。
保留已有MT比较作为能力证据，继续从EMBER自身的绝对不足及已知可学改进解释教学—算子—自身控制，不要求参照先成功。
最近[task32自身状态与冻结LoRA交叉续行](docs/designs/task32_state_policy_crossover_diagnostic.md)已完成并由主讨论直接消费原件，
完整科学解释见机制§98/findings§277：两套策略都有从s17完成后段的控制，却均未从s43在原时限内完成；
不能把具体失败简化成缺少第二阶段Value，也不能从一个选例推出普遍状态/接触根因。
main另对已有四train任务、同query两teacher预测作CPU风险分解：十步前5的teacher差异项只占总风险.0961%–.7586%，
大部是平均预测残余；与具体闭环中的前缀敏感共同解释，不能归罪公共B0或重开一致性loss。
实际native→S/M→自身调用→真实FM信用、任务内获取/跨任务迁移/保持已合入同一推导，未选择新的架构或正式重训。

同query强MT/旧T的CPU审计已完成并停止，0GPUh、约9.6分钟含交付、1.75MiB，无模型/环境/新预测或tracked/Git写入。
四task A28两消费者完整配对，Current/T/MT前5FM=.128471/.128661/.131079；十步all7=.124717/.124228/.119610，
当前motion6更低、gripper更高；任务/逐query不利项及masked两种权重见机制§98.6与findings§277，不把离线通道差直接当闭环根因。
审计原件已归档至四格root `analysis/conditional900_reference_audit_20261002/`，原tmp路径仅保留别名，单一实体。
上述CPU参照审计结束时，canonical科研记录/Git窗口由main独占，实验session没有新GPU任务；
随后新16行合同另行移交，当前状态见本节顶部。
**原四行及CPU参照审计均已结束；两条MT补测未派发，不恢复任何历史训练。**

旧T2340父/S共享/P任务私有/D条件私有的task32全部32条既存轨迹CPU读回已完成并停止，0GPUh、约1.11MiB。
main已直接读16条关键continuous、双RGB及实际学习源码，完整解释见机制§99/findings§278。
D17从1/4到4/4，新增三行保留开炉并完成搬壶；D43仍2/4，S/P在其init3反而成功。
S17两行学会抬壶却未放置，S43/init2得到放置而丢失开炉；3cm仅为描述阈值，父有低于该阈值的完整成功。
两组D面对同一query/flow/功能目标，差别由视频生成的初始B及其后学习轨迹造成；不将D修正当唯一视频知识标签。
继承已完成PZ投影及其跨条件外推失败，不重复投影/共享解除/输出扩张，也不将旧T和当前900混成同一模型。
原件在旧run `analysis/task32_absolute_learning_readback_20261002/`；无新模型/环境/动作标签/held读取。
main另读已有S/P/D同B20预测作学习修正分解：task32修正差异能量仅.346%/.479%/.789%，两D修正余弦.9844，
几乎全部平均改善来自两teacher共有部分；task20与task12的不利/高差异项保留，不能外推为所有任务一致。
结果及执行Q/R的精确含义见机制§99.5。后继冻结分析不由共有分量直接宣称公共B0不足，也不重开一致性训练。
该合同已完成并由main消费，canonical窗口已经交回；没有新的架构/fresh训练资格。

以下保留最近四格批次的完整执行事实：
按(i,j)=(17,17)/(17,43)/(43,17)/(43,43)，success为真/真/假/假，结束步278/275/520/520；
首次抬壶3cm为210/210/509/无，放置谓词为278/275/无/无，最大抬高14.771/15.362/7.299/0.668cm。
两条对角重现原一成一败；原17为277步、本次278步，不追微小数值一致。同i两行anchor状态与新双RGB误差均0，
相对原continuous/compact的EEF、夹爪、物体、谓词、raw state8误差均0；原state2没有RGB，未编造历史RGB。
完整4 full/continuous/goal/physical action齐备：144个重放prefix chunk明确无新policy forward，
175个合法续行50×7 chunk，first consumer为batch4、index36、共同seed、官方十步；有限差D=D_W+D_s误差0。
W17在s43到509才抬壶且未放置的不利例保留；最终成功随前缀的事实不能直接宣称具体几何根因或新架构资格。

沿用canonical rollout、原FrozenOperatorAdapter和初始scene/full捕获；31项针对CPU及10项封存来源/数值/case检查通过。
实际有效读取为clean pushed detached `e84712d8b6f3641968d820a45500670a58196c66`，原900训练85919994与原读取923ff89b单列。
16:52首次加载因LIBERO新root缺配置退出1、0环境行，费用0.004473948GPUh；封存spec搬迁的CPU来源错误也保留。
复用原runtime配置owner修正后16:56:08–16:57:38在gpu02:7执行四行batch4、一次source加载，
有效consumer/CPU readback均exit0，0.024935183GPUh；累计**0.029409131GPUh**含失败/加载/退出，硬限1GPUh。
峰allocated9.424913/reserved9.968750GiB；四行已经是本科学范围最大packing，没有额外工作可加入。
data1 quota现场2T/limit2.0T、报告使用1.0T、个人实占1131008344064B、共享82T；
root阶段观察约1.489GiB、保守峰值2.5/4GiB。双节点核实本批owned GPU为0、两个consumer PID已消失，未动他人进程。
原source/spec/frozen与失败保留；任务专用入口/新case hooks在交付后从active tree退役，Git及实际冻结源码保留，
封存spec身份的通用读取修正与回归检查保留；退役后35项针对检查通过，canonical rollout/capture恢复为原单一路径。
completion/readback/report、原F2×2和分解、费用/退出/释放回执均在
`/data1/user/ymdai/ember_runs/task32_state_policy_crossover_20261002/`。整批报告一次交给主讨论并释放canonical写入/Git窗口。
Owner要求的决策错误已写入current_owner_requirements§3、findings§276和research_history，394b5f79已推送。

## 最近撤回批次及保留事实

[原生prefix变化Value的共同学习](docs/designs/native_prefix_change_value_design.md)曾获派发，
但Owner指出没有建立“具体task失败→有证据的方法缺陷→干预改变失败预测”的决定性链条，block17选择也仅有启发；
主讨论承担方法选择错误，立即撤回该批剩余训练、物化与评测。允许有界检验不等于可以跳过机制辨识、
让正式训练替方法选择寻找理由。**本批因方法选择依据不足撤回，非工程失败或性能阴性。**
Owner恢复自主推进的总体授权仍在；本实验session只完成撤回收束，不自行恢复本批或其它历史计算。

实际工程在独占worktree完成原生attention字段、零E、full-only合同及原消费者接入，29项CPU检查通过；
clean pushed detached训练/读取身份为`ab8d2c7ee22447da133cde59b975133b5988520f`，source aligned1000保持冻结。
最长视频517原帧/105采样帧两步profile完成，micro28/frame8、38.93/33.34秒、峰reserved22.805GiB；
导入失败exit1与有效profile exit0合计0.044224GPUh，profile权重只作一次性工程产物，未进入正式初始化。
正式fresh在gpu02的0/1/2/3运行到已登记update71（284条件、7,952个query），主讨论于2026-10-02 15:46:56
向已核实的本批PGID2709066发SIGTERM；实际torchrun exit1、四rank已退出。未到首个90 ECP，完整checkpoint为0，
无本批可恢复checkpoint；已登记1–71更新均未保存，第72步可能在途，无额外完整更新的证据。
270→450、全部bank/400/seen及后继未启动且已取消，三个task-owned启动器已加撤回退出保护，冻结源码未改。
新闭环0/944、full0/52；没有本批held native读取/物化/评测，也没有读取Val/Test动作或产生held梯度。
曾只读核对既存Val400结果元数据/成功原行及seen参照，不将其冒称完全没有接触held结果。
累计1.452021841GPUh（含加载/profile/启动失败/训练退出），root阶段观察峰值约0.69GiB；
双节点现场核实本批GPU占用为0，所有本批producer/rank PID消失；没有终止他人进程。
原source/spec/frozen、profile、日志、失败及停止回执均保留；completion/readback和精炼closure见
`/data1/user/ymdai/ember_runs/native_prefix_change_value_20261002/`，原停止回执见本树
`.codex/tmp/native_prefix_change_main_stop.json`。原944行合同未完成，不能将收束完成写成科学实验完成。

Owner同时纠正GPU利用：实测22.805GiB仍有余量，实验session沿用frame8却未验证更大物理分块，
未落实已有吞吐要求。Owner要求已加强为AGENTS§9长期规则：必须主动验证显存余量的吞吐用途，
以实测选择物理配置并记录未放大的依据，保持逻辑batch/权重/更新及科学范围；本批撤回后没有追加profile。
实验session的运行收束已在863eef2b推送并释放canonical窗口；主讨论随后在394b5f79完成稳定要求的科学纠正。
当前写入分工见顶部，已结束的CPU临时目录权限不延续为并发canonical写入权。

最近完成[条件A函数重表达诊断](docs/designs/conditional_A_reexpression_diagnostic.md)：Original13/32、Reexpressed14/32，
R13/G1/L0；448 FM＋448十步query、64配对闭环和16 full齐备，费用0.281819GPUh、阶段观察3.113960GiB。
主讨论直接消费后完成机制§97/findings§274：不部署解析C、不自动去S训练或追加投影探针，主要能力缺口仍在。
原生Value批次已撤回，不由该冻结保留结果自动推出后继；条件读写§13–§15仍不恢复。

同时维持Owner已明确的数据约束：现有数据规模固定，后继不增加训练任务、示范数量或引入额外数据来源扩量。
稳定约束已登记于`docs/current_owner_requirements.md`§4；不以增加任务、示范或额外数据源解决当前性能瓶颈。
分析不限人为时长不等于计算预算无限；每批仍须登记主要干预、完整参照、预计耗时、资源预算与停止线。

条件A诊断的交付事实见findings§273；前置判断见函数重表达设计、findings§272与两份20261002专家审计/条件A证据JSON。
固定数据审计未发现匹配当前source/标签/训练量的全36专家上界；不能以“同数据专家已经都学成”为前提恢复蒸馏路线。
现有900的四train×两teacher原件中，S在Q8/V8/out的作用大多能用A0响应重表达，但只证实教学native分布上的局部事实。
该诊断已完成完整38处重表达、自身query/十步动作/64有限闭环，原硬限3GPUh/12GiB、预计含工程1.5–2.5小时。
无优化器、Val/Test/controls或选点；结果已到齐并停止。通用图文支路/条件图去S仍未选择；没有自动恢复的Value图。

最近训练结果完整判断见机制§96、findings§271与`docs/analyses/conditional_support_diversity_evidence_20261002.json`：
扩支持只带来净+2、得27失25，未兑现广泛迁移/保持预测；原C12 137→154→140与seen91→99→115揭示获取和held保持分离。
native真实动作读回有小幅改善，仍未形成广泛收益；没有把对象/阶段行为定位冒称唯一神经根因，未选择新的架构或辅助目标。
主讨论未增加模型/环境前向或GPU分析，工程验证由实验session闭环。三批总40.215657GPUh，2,176条新增闭环原行。

§15整批已完成并停止新增计算：同父同龄固定630的D71为156/400、C12为154/400；配对R129/G27/L25、净+2、churn52、Jaccard0.712707，八task簇95%差额[-9,15]。原36task seen为D71 98/144、C12 99/144；target24为53/55，原support12为45/44。1,088环境原行、24份A28及16条被动native全部验收，missing/invalid为空，整批owner与全部消费者exit0。新增8.477763GPUh，阶段观察53.338696GiB、保守60/64GiB，旧31.737894GPUh单列。completion/readback与完整训练/读取身份见下方交付段；两个630仍为有界诊断，没有selected或后继训练资格声明。主讨论据整批原件作科学判断。

metadata/protocol先于新label读取封存；实际事件确认480个target条件/13,440query及flow完全保持，71项支持240条件按80/71或60/71直接进入原四条件损失。核心11项CPU、消费者37项CPU及集成后19项针对检查通过。唯一data/credit/trainer/scope与官方队列复用，新增support_diversity只拥有固定事件/权重；源码训练/读取796d7a9e来自clean pushed detached冻结，旧原件保持。新root/data1启动现场quota 996.3G/2T、个人实占1069732286464B、共享82T余量，全部新增data1，data0只读。普通工程事实只入launch记录，整批科学结果一次交付。

主讨论已完成900原件、行为、native实际动作读出及数据/历史分析；最近完成合同为
[条件读写设计§15](docs/designs/conditional_read_write_architecture.md#15-固定目标曝光的辅助任务分布短窗检验2026-10-02)；新批范围只按本文件顶部及新诊断设计。
当前问题是已见能力获取未转成足够的跨任务绝对能力与保持；数据支持不足未被证实为唯一根因，扩数据不再是可选后继。
完整解释见机制§95、findings§270和`docs/analyses/conditional_read_write900_evidence_20261002.json`。

§15从真实450父点训练唯一D71分支180更新到630，保持target24的原事件/次数/权重，仅将support12分布替换为审计source71。
复用既存C12_630为同父同龄对照，不重训control；两个固定630各读correct400、原seen144、A28及8条被动native。
新增20,160个完整query中13,440个target query保持、6,720个support query改变分布；每source task只有3/4个teacher条件。
这是有界学习诊断，不是95task fresh或selected资格，不恢复原同池路线选点。
唯一新root为`/data1/user/ymdai/ember_runs/conditional_support_diversity_pilot_20261002`；新增硬限12GPUh/64GiB，预计2–4小时。
科学合同及实际执行均已完成；新540/630完整ECP保留，540仅供恢复，两个630读出齐备后已停止。
没有自动D71→900、完整fresh、其它checkpoint/controls/Test/RL或小扫；后继由主讨论消费原件后裁决。

已完成§14为900 validation140/400、seen115/144（target70/96、support45/48），held相对450仅+3，task31丢失16条旧成功。
810分支未触发；全部544环境原行及A28/native验收，新增12.464601GPUh。观察存储35.598759GiB、保守60/64GiB。
已完成§13为fresh450、validation137/400、seen91/144，19.273293GPUh；两批合计31.737894GPUh，全部计算已结束。
两个原合同、完成记录及训练/读取身份均保持；§14同池继续学习的科学预测未通过，不因§15读取C12_630而反选旧峰值。

Owner于2026-10-02指定新的唯一实验执行者`01a0fabb-f7a0-7100-93d8-6a0f66055553`
（hostId：`remote-ssh-discovered:BCI-GPU02`），接替`01a0f018-69af-7b00-b614-7e117540051b`，负责代码、测试、排障、资源调度、Git及冻结运行。
继任主讨论`01a0faba-4bb9-7ca1-b3b4-6d05bec44e33`接替`01a0ed66-cda4-7a23-90f0-e0d3a06a1d36`，
负责科学合同、原件解释、机制及后继/预算裁决。继任后Owner已恢复自主推进；只读专家审计已完成，当前阶段如本文件顶部。
不从历史暂停或设计中的“下一步”推断现行状态；工程检查仍由实验session闭环。
沟通边界按[Owner要求§6](docs/current_owner_requirements.md#6-沟通与交接)：整批科学结果或确需裁决的实质边界只回报一次；
工程阶段、可自行修复的故障和普通调度记入已有记录。写入/Git窗口与实际冲突方直接串行协调。

## 条件A重表达诊断：整批交付（2026-10-02）

唯一root为`/data1/user/ymdai/ember_runs/conditional_A_reexpression_diagnostic_20261002`，
`completion.json`／`readback.json` complete，精炼结果在`report.md`与`delivery_summary.json`，实际原件读取在`verification.json`。
8条件×38处默认float64 gelsd，原S/M保留，16份完整rank128两格LoRA；448 FM与448官方十步query记录，
独立task-query只有112。64闭环为四train×两teacher×states0–3×两格，各格32、16不同物理初态；
全部continuous／goal／实际action／compact齐备，state0全16行full双相机，未作新训练、held/controls或选点。

Original13、Reexpressed14，R13/G1/L0、churn1、Jaccard0.928571；唯一得例12/14/0。
task0/12/20/32为8→8、2→3、2→2、1→1；每teacher成功集合及RGL完整在readback／report。
固定task内四state成组bootstrap保留两teacher重复结构，净差95%区间[0,3]，不是大样本泛化区间。
FM全50/前5 MSE .110917/.128471→.111018/.128774；十步全50/前5 .141694/.124717→.141308/.123937。
有效future、有效前5、motion6/gripper分别保留；两格前5直接输出差/原输出范数等条件均值FM0.899%、十步2.249%。

教学E/完整LoRA中位数Q8/V8/out为0.0364%/0.0682%/0.1734%；Original十步自身hidden为5.547%/9.870%/7.415%。
全38处最不利条件task0/teacher40/V7为FM69.032%、十步73.381%，不把焦点层位的小值概括成全部层近等价。
194/224个条件query至少一项所列风险恶化（包含微小数值变化），Original19／新格18共37失败行均保留。
教学拟合不能直接搬到自身hidden；本有限面板控制大体保留不证明S无用、fixed-A可重新学成或held能力修复。

实现／实际读取`923ff89b60f4f8a269e5352d8d82189d72a9cd0a`已push main，消费者来自root下clean detached `frozen`。
原900训练`85919994aef11c17b49b7d0e70a2c110158bff61`、aligned source1000和冻结normalization保持，原件只读。
37项针对CPU验证通过；完成后读回全64小continuous原件的实际action/EEF/物体/夹爪/goal形状及finite，
单条full的双256相机、normalized完整50×7与physical前5核实；policy仍官方256→224/10step/前5/settling10/成功终止。

现场strg01 data1 quota2T/limit2.0T、个人实占1,127,743,614,976B，共享82T；全部新增data1。
单producer后两卡×3persistent workers，总费用0.281818972GPUh，含加载／CPU解析期间占用／故障／恢复；
root阶段观察峰值3.113960GiB，保守估计7GiB，低于3GPUh/12GiB。不声称连续精确峰值。
首GPU14:02:43、末GPU约14:12:37，全部GPU释放，只保留原gqma低占用上下文，未中断其它用户。
首临时等待包装器误用系统Python而exit1；其producer因重父化OS退出码不可观测、明确null。
producer completion与全部原件已验证，未重启其有效工作；四闭环、CPU readback和恢复owner均exit0。
完整退出／费用／释放回执见root/launch；不伪报全部exit0。工程／计算已闭环，整批一次回报后释放canonical窗口并停止本项。

## §15科学登记与交付范围（2026-10-02）

同父450、模型/full-only/Adam/绝对LR保持；target24沿451…630全部480条件，support240个slot改为71task。
固定permutation seed `[20260928,15,cycle]`，task次数3/4分别以80/71、60/71校正，使target/support总权重仍480/240。
其余59source任务对当前Writer是新映射，对source policy不是；完整19个target40等价排除项不恢复。
source71不存在任何basket或In-microwave目标，不能把扩分布写成直接补齐held16/39。
metadata预算表在900分析JSON：teacher总帧24,188对原23,636，最大同95帧，每步4个不同task；无新模型执行。

C12_630引用§14 root的`conditional_read_write/train/attempts/continuation/checkpoints/macro_00000630`，
旧实际训练为85919994；D71从§13的完整450（a0末段）分叉。新读取/训练身份由执行者分别登记，不改旧冻结树。
新分支540/630完整ECP；两个固定630共1,088环境行、24份A28及16条被动native，不能按部分结果取消对照或补读其它点。
原36-task seen面板不扩分母，保留所有原能力和反例。正收益只提高当前数据解释支持；阴性不自动延期或开完整fresh。
预计8–10GPUh有用计算、硬限12；额外产物估计约53GiB、峰值限64。实际GPU和strg01 quota准入由原执行者现场核验。
canonical科研文档在本次登记后随具体派发一并交给执行者的tracked/Git窗口；不得与main重叠写入。

## §15整批交付（2026-10-02）

唯一root为`/data1/user/ymdai/ember_runs/conditional_support_diversity_pilot_20261002`。
`completion.json`与`analysis/readback.json`均为complete，全部44项读取/配对/来源/费用检查validated，missing/invalid为空；
`launch/batch_owner_exit.json`确认exit0。两个固定630各400＋144、8full＋4public A28及8条被动native，
共1,088条环境原行、24份速度预测、16条合法teacher机制字段，所有goal/continuous/action/RNG原件齐备。
每个400为8full/392compact、每个144为36full/108compact；参照无新GPU消费者。

Held task3/6/11/16/23/26/31/39的C12→D71为30→28、7→5、44→44、2→4、0→0、43→48、28→27、0→0。
对应R/G/L为24/4/6、4/1/3、39/5/5、1/3/1、0/0/0、41/7/2、20/7/8、0/0/0。
两臂breadth均6/8；Spatial37→33、Object46→48、Goal43→48、Long28→27。
主比较D71−C12为净+2，R129/G27/L25、churn52、Jaccard0.712707，whole-task簇95%差额[-9,15]，不是训练seed不确定性。
D71对共同450137为R110/G46/L27、净+19；对既有900140为R114/G42/L26、净+16；
对MT153为R109/G47/L44、净+3，对T900148为R113/G43/L35、净+8，对Context900151为R115/G41/L36、净+5。
C12对共同450为R110/G44/L27、净+17，对MT为R112/G42/L41、净+1；全部成功集合和其它配对保留readback。
630事先固定，两个诊断点均未selected；没有根据这次分数恢复§14选点或读810，成熟T2340仍是不同年龄背景。

原seen144为C12 99、D71 98，R86/G12/L13、净−1、churn25、Jaccard0.774775，36task簇95%差额[-10,8]。
Target24为55→53（R42/G11/L13），原support12为44→45（R44/G1/L0）；breadth34→31/36。
D71对45091为R80/G18/L11、净+7，对900115为R91/G7/L24、净−17；
对MT93为R84/G14/L9、净+5，对Context900103为R88/G10/L15、净−5。
四初态/task、原50teacher池及有限面板边界保持，seen分母没有扩为95task。

A28仍为原query/noise/tau的FM速度预测，无额外十步采样。两臂8full/4public及全部逐query、first5/full50、motion6/gripper记录已验收；
八full的D71−C12平均risk差first5 −0.00184794、full50 +0.00032395，四public为−0.00037572/+0.00029511。
这些是固定训练面板的功能读回，不构成闭环或视频因果结论。各臂8份真实H/c/d与Q8/V8/action_out的X/A0/S/B0/M保留，未增加native forward。
原件入口为`C12|D71/conditional_read_write/evaluation/630/correct400/results.json`、
`C12|D71/conditional_read_write_seen/evaluation/630/correct144/results.json`和`C12|D71/analysis/A28/conditional_read_write/`。
完整逐task/suite、原行得失、goal阶段和连续动作引用在`analysis/readback.json`，全部有利和不利样本保持。

仅新训D71的451..630共180更新、720条件、20,160完整query，full-only，public训练FM为0。
Actual target条件/visit/teacher/query episode/frame/flow与C12原事件完全相同；支持240slot按固定71task permutation和3/4次权重执行。
窗口target/support权重480/240，每source累计240/71，实际损失为sum(weight×mean28fullFM)/4，不按宏步权重和或物理rank归一化。
新增59项是Writer的新映射；每source只读3/4不同teacher，不声明数据普遍可行或不可能。
完整450的Writer/Adam/scheduler/scaler与五rank RNG恢复，绝对LR保持；sampler为登记科学分叉，不称原轨迹exact resume。
首451恢复证据在`launch/actual_resume451.json`，完整新540/630和实际事件审计在readback.training.D71。
旧父450为a0e0248d，C12_630真实训练85919994、新读取796d7a9e；D71训练/读取均796d7a9e。
冻结树`/data1/user/ymdai/projects/EMBER-support-diversity-formal`及两个arm的spec身份独立登记，旧冻结树/原件未改。

D71训练实际GPU02:7/1/2/3/0、world5、µ28/frame32，执行面支持1–6rank；180步mean17.4291s、median16.9755s。
C12 bank/A28/seen在GPU01:2与D71训练并行，D71 bank400及两个400/末点seen144在GPU02五卡动态队列运行，
官方评测每卡3 persistent replicas；D71 bank144/A28在GPU01:2与C12 400并行。全部有效进程exit0。
训练加载/保存4.621526GPUh，C12 bank400/144 .372669/.132272、A28 .025758、seen144 .307342、400 1.016590；
D71 bank400/144 .344510/.115134、A28 .020905、400 1.085736、seen144 .435324；总8.477763278407GPUh。
启动到整批owner退出约1小时33分；此数不包含启动前工程实施时间。原两批31.737894GPUh单列，无预算结转或扩大。
无GPU失败/hold/profile；一次首451 CPU包装读错键为record，已按真实row字段修正，0GPUh，原失败记录保留。
阶段du高水位53.338696GiB、保守60/64GiB，非连续精确峰值；所有新增data1，原data0只读。
新增计算已停止；整批科学交付一次发送主讨论，常规阶段无跨session广播。

## §14整批交付（2026-10-02）

唯一root为`/data1/user/ymdai/ember_runs/conditional_read_write_continuation900_20261002`。`completion.json`及`analysis/readback.json`验收complete，
整批readout owner exit0，全部有效训练/bank/evaluator/A28 exit0，原工程失败及计费保留。
仅从实际450完整ECP恢复451..900；新增450更新、1,800条件、50,400 full query，累计100,800，public训练FM为0。
36task每条原50teacher完成第二轮，累计100访次/50不同teacher；不是50条新视频。
Adam/scheduler/scaler/sampler/原五rank RNG和450 metrics前缀恢复，绝对LR不重启；540/630/720/810/900完整ECP齐备。
训练/读取实际身份均`85919994aef11c17b49b7d0e70a2c110158bff61`，来自clean pushed detached
`/data1/user/ymdai/projects/EMBER-conditional-read-write-continuation900-formal`；父450的a0及更早797/0e2/ebbb来源不改。

900完整validation为140/400，breadth7/8；task3/6/11/16/23/26/31/39依次32/7/45/1/1/41/13/0，
Spatial39、Object46、Goal42、Long13。对自身450137为R107/G33/L30、净+3、churn63、Jaccard0.629412，
整task簇95%差额[-30,36]；对T900148为R114/G26/L34、净−8；对Context900151为R117/G23/L34、净−11；
对MT153为R102/G38/L51、净−13、churn89、Jaccard0.534031、簇区间[-50,25]。
Task11净+11、26净+5，但3净−5、31净−11（原24只留8，得5失16）；16/23各仅1、39仍0，全部反例保留。
140≤153，按预登记不生成或读取810，不倒选其它checkpoint；没有相邻资格或selected声明。

900完整seen为115/144，breadth35/36；train24为70/96、support12为45/48。
对45091为R87/G28/L4、净+24、簇区间[12,37]；train净+25、support净−1。
对MT93为R87/G28/L6、净+22、簇区间[9,36]；train净+24、support净−2。
对Context900103为R95/G20/L8、净+12、簇区间[2,23]。四初态/task、已训练50池不能冒称held-video或正式400资格。
全部12份A28仍为固定query/noise/tau的FM速度预测；八条H/c/d、Q8/V8/action_out被动字段不增加forward或标签。
原行入口为`conditional_read_write/evaluation/900/correct400/results.json`、
`conditional_read_write_seen/evaluation/900/correct144/results.json`；A28/native为`analysis/A28/conditional_read_write/`。

训练GPU02:0/1/2/3/7、world5、µ28/frame32是现场选择，执行面仍支持1–6实际rank；
新增450步mean16.7869s/median15.7107s，训练含恢复/保存10.586330523091GPUh。
400 bank五卡与seen bank GPU01:2并行；400/144均五卡×每卡3 persistent replicas/dynamic long-first queue；
A28在GPU01:2与400独立并行。无dummy/hold/profile、无历史重评、无新增科学面板。
一次备用卡调度漏排除已派发但context尚未显存可见的卡，被launch准入拒绝；原bank无产物即退出−15，
0.007435357122GPUh全计费。已在独立读出owner窄修排除running receipts，通过原现场CPU复现，
未改冻结科学代码或重训，真实后段全exit0。细节保留`launch/scheduling_repair.json`和原/新owner记录。
全部真实新增12.464600929159463GPUh，旧19.273293210686425单列，两批31.737894139846GPUh。
阶段du高水位35.598759GiB、保守上界60/64GiB，未称连续精确峰值；全部新增data1，data0只读。
新增计算已停止，最终现场两节点无项目GPU context；完整科学交付只发主讨论一次，普通过程无广播。

## §13整批交付（2026-10-02）

唯一运行根为`/data1/user/ymdai/ember_runs/conditional_read_write_fresh_20261001`。
`completion.json`及`analysis/readback.json`已验收为complete，missing/invalid均为空；整批owner exit0。
450宏步、1,800条件、50,400完整FM query、36task各50不同teacher，以及90/180/270/360/450完整ECP全部核实。
公共训练FM为0，profile权重未进入正式训练。4×28有效query、事件流、绝对LR、Adam/scheduler及恢复边界保持。

完整validation400为137/400，breadth5/8；Spatial43、Object34、Goal36、Long24（各100行）。
对Context450126为R91/G46/L35、净+11；对T450122为R97/G40/L25、净+15；
对self_read450116为R87/G50/L29、净+21；对强MT153为R110/G27/L43、净−16。
对MT的whole-task-cluster95%差额为[-37,-1]，对Context为[-28,52]；不是seed不确定性。
Task31为24/50，但task16与23均0/50，保留全部有利及不利原行，不据单点或内部量宣布机制成立。

完整seen144为91/144，breadth29/36；train24为45/96，support12为46/48。
对同场景MT93为R78/G13/L15、净−2、churn28、Jaccard0.73585，whole-task-cluster95%差额[-13,9]。
Seen各task只有四个初态、来自已训练teacher池，不能冒称held-video或正式400资格。
全部544条环境原行、goal、连续trace、实际动作和RNG，以及固定A28的8full+4public速度预测、8条被动机制记录齐备。
原行入口为`conditional_read_write/evaluation/correct400/results.json`、
`conditional_read_write_seen/evaluation/correct144/results.json`；A28与机制入口为`analysis/A28/conditional_read_write/`。
分布、success sets、配对、簇区间和全部原件引用统一在`analysis/readback.json`，未重评参照或增开其它面板。

首段训练来源`797ae01f`（1..90），后续`0e2d6c79`（91..180）、`ebbb5e78`（181..270）、
`a0e0248d`（271..450及全部读取）；后续工程交付已承接整理后的`16241717`，旧冻结树与失败原件保持不变。
最终冻结树为`/data1/user/ymdai/projects/EMBER-conditional-read-write-mlp-formal`；
完整450在`conditional_read_write/train/attempts/resume360_native_packing/checkpoints/macro_00000450`。
真实来源、spec、topology/RNG与恢复链在`launch/code_identity.json`，不把今天文档提交改写成训练来源。

已交付按实际卡数自动target分片和完整cotangent返传，以及teacher作用域SDPA/冻结MLP checkpoint优化；
91..450由GPU02五卡真实训练，第五rank承担target与信用计算。最后段按实测峰值自动选择frame上限32、policy microbatch28。
实际361耗时12.93秒、当步最大帧批28，任务rank峰值约30–35GiB；这些是该事件的实测值，非最长视频或全程吞吐测量。
400 bank五卡耗时307.97秒，A28在GPU01独立并行，正式评测使用动态persistent queue。
已知NCCL、native OOM及profile replay工程失败均已独立修复并计费；失败attempt的181..185未保存更新从完整180重算。
针对实际源模块的8项attention及12项MLP CPU检查、实际通信/恢复/完整训练与读取消费者均有原件；未重复最长GPU profile。

整批成本19.273293GPUh，其中有效训练段10.876456GPUh、显式占卡5.085877GPUh，全部失败/加载/后段成本均包括。
最早登记占用到completion约4小时29分钟；不是不含工程的训练时长。
阶段边界观察新增最高37.058697GiB、保守峰值界64GiB，分别低于40GPUh/80GiB；不声称连续精确峰值。
strg01 data1独立quota2T现场准入及双节点launch检查在已有合同内，所有新增产物均data1，data0只读复用。
§13计算已停止，原合同不自动续900或其它checkpoint。主讨论消费完整原件后的独立后继已登记为上方§14；
原批完整科学分析、数据和成本保留，不因新合同覆盖其当时停止边界。

## 整仓整理交付

整理会话`01a0f7fa-63b3-7a42-a196-4c0fd145b10c`在独占`codex/ember-cleanup-20261001`完成源代码、
测试、脚本、配置、入口、文档及已核实临时内容整理；从`927b1498`建立，承接到`0e2d6c79`。
Owner纠正后的无token budget goal用于这项独立任务；没有新增session、子代理或GPU实验。
实验dev/frozen树未被修改，科学合同及必要原件保留。

已在本树退役旧Writer训练/生成与已关闭的专用诊断执行面，保留当前共享组件、sealed配置、实际bank/原行读取及科学原件；
统一README、稳定规则、当前状态和历史索引，修正Source配置导航变化导致的误拒绝。
源码提交`540fb773`；536项保留CPU测试及十个实际CLI help通过，承接96b07612后的实际消费者38项通过；
配置、import与文档引用已检查。57个退役文件及具体保留理由见research_history。
已实际修改个人workspace-cleanup skill及其inventory helper，symlink/边界/CLI验证通过；回滚在
`/data1/user/ymdai/skill-maintenance/ember-cleanup-20261001/workspace-cleanup/`。
已删除已消费的旧交接文件、canonical废弃pytest夹具及整理自产的大型临时检查内容。
交付以main包含本整理提交、与origin一致为准；task-owned临时树在交付后清除。
详细实改、验证、skill回滚及生命周期未明的保留范围见[research_history](docs/research_history.md#2026-10-02整仓整理与历史状态入口)。
实验后继窗口仍与实际冲突方串行协调；不将整理的工程检查转给主讨论。

## 已完成研究与原件可用性

最近的self_read450为116/400，Context450126、T450122、强MT153；完整正负证据、task/suite与机制边界见
[研究历史](docs/research_history.md#2026-10-01共享参数两次native的fresh450完整阴性与分析后讨论)、
[机制分析](docs/analyses/feature_to_operator_mechanism_20260926.md)§87及findings§261。
原400/144 scenes、MT/T/Context/self_read原始结果和固定A28输入
`operator_chain_diagnosis_20260929/functional_credit_transport/group0`均为当前依赖，继续保留。
已完成§40及更早路线不因历史“下一步”自动恢复；科学阴性、失败原因和有效正例完整留证。

2026-09-23历史存储裁剪账目为`runs/analysis/workspace_cleanup_20260923/storage_cleanup.json`及
`checkpoint_retirement/closeout.json`；当时checkpoint累计回收413.766GiB，94个weights_only、80个metadata_only。
这些是当时的裁剪事实，不是当前存储实测。Source1000 frozen policy可读取，原训练optimizer/EMA载荷已退休。
每项资产的当前可用性由其`checkpoint_retirement.json`、`payload_retirement.json`与实际文件共同说明；
weights_only不含完整训练恢复状态，metadata_only不含权重。不得仅按原manifest宣称可重放或exact resume。
本次源代码整理不删除source/dataset、formal原始rows/metrics、关键模型、运行根或冻结树。

## 历史状态的恢复入口

旧progress与task_plan累积了多轮互相覆盖的“当前/暂停/尚未实施”。完整当时过程保存在Git：
`git show 797ae01f:progress.md`、`git show 797ae01f:task_plan.md`。
按日期/方案追溯的入口为[research_history](docs/research_history.md)，跨轮结论在[findings](findings.md)。
这些历史快照记录当时授权、成本、失败与交接，不作为今天的执行authority；不新增平行状态或in-tree archive。

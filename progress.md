# EMBER progress

## 当前状态（2026-09-19，专家修订执行goal已启动）

Owner明确要求完成专家意见、推送并汇报。Active design为[同视频教学候选](docs/video_teaching_writer_design.md)。
Owner随后要求再观察训练趋势；已登记两臂各从完整1500继续至2100，1800/2100评测，2100双方换视频。
原窗口选择和1500 controls保持冻结；续训入口已实现，71项相关CPU检查通过，
真实8400事件核对保持两臂各自前6000事件、全部主事件/RNG及教学噪声种子；现有后台任务继续。
续训代码/合同已推送d1474ce0，并建立clean detached runtime；`continuation/driver.sh`已在tmux排队，
Owner要求利用空闲卡后，原串行控制器已在1500物化正常结束处退役，现改为训练与评测流水并行；
两臂训练保持原world2／GPU拓扑，1800 checkpoint发布即交额外合格卡物化与闭环，不再等整臂评测结束。
保留调度前脚本、状态及新分配记录；切换时旧控制器已进入1500 validation的worker启动，
该未完成尝试（0个完成shard）随旧终端关闭，已原样保存并在额外卡按原400条件重新开始；训练与已完成结果均保留。
续训仍严格以2100为终点，原科学合同、所有配对面板和1500冻结选择不变。
主实验先用单相机agentview；新增教学项与主FM经同一套LoRA联合训练全部Writer/Meta，source冻结。
主训练严格完成1500硬终点，15个完整checkpoint保留；1500条更新连续，6000条件、126000主query＋42000教学query完成。
900/1200/1500 correct为149/174/165（各400，A为140/135/112），train为50/67/65（各96，A54/62/56）。
1488对条件、RNG、source与完成状态核对通过，主driver及六个评测面板均正常退出。
相对A，验证R/G/L为106/43/34、112/62/23、93/72/19，churn77/85/91，breadth均6→5；差值任务簇95%CI为[-2,+9]、[-1.5,+22.25]、[+1,+29.25]百分点。
自身相邻R/G/L为118/56/31、135/30/39，churn87/69（A79/83）；四suite均非零，但验证breadth保持5/8，Long为32→17→23。
三个节点same-task-other400全部完成，为140/159/165；每节点复用全部400个已有条件，整轮50视频/task与逐行不同映射核对通过。
按预登记规则，胜出相邻对为1200/1500（四分数最小值159），主候选冻结1500（correct/other均165）；[冻结记录](docs/review_materials/20260919/selected_main.json)。
1500 correct→other保留143、获得22、丢失22，churn44；换视频未整体崩塌，Long仍有明显条件更替，不能把同分说成逐行等价。
已触发唯一匹配cross-episode前缀消融，配置与主训练仅教学episode关系这一科学变量不同；fresh1500，同三个主节点，并在主selected1500固定补other400。
匹配消融运行中。冻结1500的wrong/shuffled/reversed/no-video已全部完成，为124/113/113/50（各400）；
真实RGB重排、完整重编码、source与逐行state/RNG/video映射核对通过，全部worker正常退出；[独立读出](docs/review_materials/20260919/controls_report.md)。
shuffle/reverse各比correct净少52，任务簇95%CI分别[-27.50,-2.25]、[-30.75,-0.50]pp；wrong区间跨零。
效应较集中于Spatial task3，Long在wrong反而23→33，三个任务仍为零。支持此点的顺序敏感性，不证明普遍过程理解或教学项单独有效。
no-video是零LoRA/source，不是learned language-only。Controls不改变主checkpoint、架构、消融或已登记的续训配方；本轮不补双相机。
900/1200/1500冻结train held-action FM为0.104706/0.100551/0.098260（初始0.146830）；900与A同一24任务诊断输入已核对，A同点0.103778。
CPU累计246项通过：真实PI05主/五步端点损失、标签隔离、
联合一次重放、四任务更新、1–6卡不同分片归一、完整checkpoint/resume、物化与视频controls，
以及完整H/E、零残差退回有序P、BF16空分片与原生重放。首轮六个旧文案/旧架构fixture断言已修正复测。
源代码新增一个117行Procedure读取owner，退役frame-set和预算扩展分支；架构检查无新增hard，
既有native/temporal/物化大文件保持或缩小，较长shape校验仍与其张量接口同属一个owner。
真实profile及4→6完整恢复已完成；[摘要](docs/review_materials/20260919/profile_summary.json)。最长105帧在chunk16/micro16下约16.3–16.6秒，
双卡六更新均值16.75秒，分配显存峰值32.41GiB；四任务与84+28 queries语义保持，所有6000主事件及RNG与A逐项相同。
正式窗口已从clean pushed detached `39c3919c9dd54713f7bff6aa24d275e4a6231ff0` fresh启动：
gpu02两卡、frame chunk16、policy microbatch16，1500硬终点。实际run contract核对source可训练参数为0、主84＋教学28。
本地执行记录在`runs/analysis/video_teaching_20260919/launch_contract.json`，tmux driver按固定节点训练、物化和评测。
900/1200/1500各correct400/train96，selected及额外消融只按预注册条件触发。profile权重不进入正式训练。
新输出统一位于data0的`runs/analysis/video_teaching_20260919`链接目标；原预算60GiB，追加续训预留20GiB，总预算80GiB。
登记时data0 quota使用146206060KiB、额度1073741824KiB；续训前会重新核验，而非依赖此历史快照。

## 上轮完成记录（2026-09-18，A learned frame-set诊断）

Owner授权的唯一匹配诊断已完成1200更新、四节点strict paired400及train96；训练严格停在1200。
所有worker和driver正常退出；无运行中的训练/评测，无待自动启动的新实验，当时没有active design。
[封存合同](docs/learned_frameset_reference_design.md)与[专家原文](docs/review_materials/20260918/expert_review.md)保持登记时口径；
结果、解释和可复核精简原件见[完整报告](docs/review_materials/20260918/frameset_report.md)。

| 更新 | frame-set validation /400 | A /400 | frame-set train /96 | A /96 |
| ---: | ---: | ---: | ---: | ---: |
| 300 | 79 | 99 | 33 | 36 |
| 600 | 109 | 88 | 40 | 47 |
| 900 | 142 | 140 | 49 | 54 |
| 1200 | 118 | 135 | 57 | 62 |

集合参照900能达到142，但1200未保住该能力，不能宣称与A相邻等强，也不能说A在两个节点全面更好。
主节点A→set R/G/L为102/40/38、90/28/45，breadth6→7、6→5；差值任务簇95%CI为[-4,+5.75]、[-12,+1.75]百分点。
set900→1200保留91、获得27、丢失51，churn78（A79），Long29→10，主要是双物体放篮28→10。
train49→57继续提高而validation回落；有限单seed结果不证明顺序普遍无用，也不支持立即删除或加深旧时序处理。

正式版本为clean pushed detached `a9d8614964abfcddef40eb82f2862623f587ffa6`，历史A为`575c189a`；
4800条件/100800queries完成，12个完整checkpoint保留，profile权重未进入formal。
本地canonical study：`runs/analysis/a_learned_frameset_20260918/`；远程保留1984对精简CSV、汇总、图表及报告。
既有source/normalization、4800采样事件、逐行RNG和teacher映射均已核对；8个新面板全部完成。
本轮仅移除视频RoPE、因果mask及读取时间寻址，匹配历史A的agentview/fixed-mean H；不改变未来接口原则。
Test、RL、新controls及其它关闭实验均未运行。已完成的目标不构成下一实验授权。

## 已完成实验与证据入口

| 实验 | 已完成事实 | 解释边界与入口 |
| --- | --- | --- |
| 统一Writer | 600/900/1200/1500 correct为90/109/67/84，各400；train为38/54/54/52，各96。1500按原合同结束 | 有界non-pass，不采纳本次改造；不证明所有统一结构不可行。findings§117、[封存设计](docs/v52_evidence_based_writer_design.md) |
| A900机制诊断 | train24固定96条件，正确58、关Procedure34、固定视频保留目标语言38/38、固定LoRA13/10、Source12、SFT47 | 旧过程路径有行为贡献，不证明顺序理解或fresh删除效果。findings§118 |
| Core/Procedure交叉 | CC58、CW34/35、WC56/56、WW38/38，各96；新增384闭环，复用288对角线 | 正确P增量能跨两个donor Core发挥；不证明Core可删、未见任务迁移或下一架构应只改P。findings§119 |

完整正负历史先读[46组证据审计及补表](docs/v52_evidence_audit_20260917.md)，再读[findings](findings.md)§117–120。
源码版本、旧专家评审、各轮逐task/suite、R/G/L/churn及formal原件由[研究历史](docs/research_history.md)索引。
统一实验及两次诊断的详细报告分别位于本地：

- `runs/analysis/unified_writer_20260917/experiment_report.md`
- `runs/analysis/v52_mechanism_audit_20260918/report.md`
- `runs/analysis/v52_core_procedure_cross_20260918/report.md`

这些`runs/`原件是ignored本地资产；远程仓库保留源码、合同、历史审计和findings结论，不包含checkpoint或数据集。
frame-set实现已封存于`b9bd90ac`；当前canonical源码为同视频教学候选，正式训练版本见上文。
已封存统一正式训练版本为`184947cb`，旧A冻结诊断版本为`575c189a`。
保留复现入口不表示已经选择或恢复该方法。旧A3000评测、C及其它关闭窗口均未恢复；Test保持关闭。

## 此前仓库整理

已检查源码、12个脚本入口、34个测试文件、27份配置、文档和ignored临时目录；未发现需要本轮改动的第二套Writer实现。
已合并重复的当前计划/进度叙述，补齐可扩展结构要求，纠正最近诊断的推断范围，并更新README讨论入口。
删除12个Python/pytest缓存目录的156个可重建文件，文件占用3,678,208字节；不改科研代码或配置。
保留九个含未合入commit或未提交改动的旧工作树，以及全部正式证据、唯一checkpoint、数据和模型。
未核定为可删除的历史临时记录保留；不以本次整理改变历史实验结果或扩大资产删除范围。
本轮检查范围为文档diff、受影响链接、Git状态与远程同步；不重复运行训练、GPU检查或无关测试。

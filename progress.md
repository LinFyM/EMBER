# EMBER progress

## 当前状态：两臂200步已完成，继续配对闭环评测（2026-09-12）

Owner已授权自主高效推进有益视频特异性及validation迁移，暂不要求145/400。
**唯一active design为[Local Action Grounded Writer](docs/local_action_grounded_writer_design.md)，ordered/frame_set均完成fresh200步，当前补齐配对闭环证据。**
Owner已在具体提案后明确给予核心科学精神内的理论／架构修正自由度，覆盖动作训练池内的局部RGB—动作配对监督。
主LoRA跨episode、teacher动作隐藏、完整H、source冻结和部署零交互保持。旧[Video Functional Writer](docs/video_functional_writer_design.md)
及第7–11节比较全部结束，旧执行辅助头／蒸馏路线关闭；新假设是实际观察转移的局部动作标签能改善过程获取。
旧C/无变化参照、95-task等历史路线继续停用；以下旧暂停记录不是当前执行授权。

### 当前实现与执行证据

- 唯一训练入口已接通主跨episode LoRA FM与局部动作FM；原执行辅助头、函数蒸馏与rho调度退役。
  局部四帧读取由action数据owner负责，严格post-action偏移；RawTeacherVideoStore仍无动作读取能力。
  局部noise/time使用独立key，主sampler/查询/RNG流保持；局部梯度仅到共享encoder、两组Meta及局部头。
- 新runtime/run/training schema拒绝旧checkpoint混入；完整optimizer/scheduler/sampler/rank RNG/topology继续保存。
  142项定向测试通过；随后processor/native/main FM与配置恢复相关64项通过（部分覆盖与前组重叠）。
- 最长task38/demo0的93帧真实teacher完成两次完整主＋局部更新；第二次17.558秒、3.645queries/s，
  peak allocated37.837GiB、reserved40.936GiB。局部分支约0.5秒；第二次Writer、Action Meta、VL Meta、局部头梯度均非零，
  source trainable及梯度为0，局部无梯度诊断路径通过。原件`runs/analysis/local_action_grounded_20260912/profile/`。
  profile仅证明真实机制与容量，不作为正式初始化或性能证据；进程已退出。
- 保持frame chunk8、policy microbatch8，正式两臂各fresh200、checkpoint100/200；单臂四卡约一小时。
  按同一四卡拓扑顺序训练ordered与frame_set，同时用剩余额度处理已到节点的物化/评测。
  每臂51,200主queries、800局部片段/6,400noise draws；两个动作留出诊断及16个闭环面板按active design执行。
- 结构检查标记现有配置/评测协议条件复杂度、文件规模及测试增长，已按数据、头、重放、生命周期owner复核；
  生产源码净增约76行，没有新增生产模块或第二trainer，不因旧协议复杂度扩展无关重构。
- 本次/data1 quota886.5GiB/1TiB、现有旧study100GiB、共享83TiB；两臂checkpoint、banks、临时写出与执行树合计新增峰值预算40GiB，
  投影926.5GiB。不复制source/data/env；启动前再核对所用GPU及剩余预算。首个完整train96分数见下。

### 当前formal launch

- ordered已从clean pushed detached `5f4f440c992e470aaffcc873c847373ce6df43e5`启动；
  冻结运行面`.codex/worktrees/local-action-frozen`，gpu01物理0/4/5/6、world4、GPU-local NUMA及NCCL_P2P_DISABLE=1已由actual run contract确认。
  启动时使用该4张训练卡；到点评测与训练共享两节点总额度≤6，不dummy占用。
- 输出`runs/outputs/local_action_grounded_20260912/ordered/`；命令、launch contract、两节点preflight与日志在
  `runs/analysis/local_action_grounded_20260912/ordered/`，tmux `ember-local-action-ordered`。
  已完成200次更新、800条件、51,200主queries及800局部片段/6,400noise draws，用时3337.34秒；
  100/200完整checkpoint与0/100/200两类无梯度诊断齐备，保存合同检查通过，训练进程退出。
  peak allocated37.852GiB，所有Writer/Meta/局部头有非零梯度，source trainable=0。
- 本轮主留出明确采用128queries/task；旧study部分实际配置为32queries，旧FM均值不能当作完全匹配的跨轮对照。
  当前两臂的曝光、诊断、optimizer和初始化配置已逐字段核对，只有process_mode不同。
  条件触发的pureFM配置也已同步至100/200及相同曝光，目前不启动。
- ordered/frame_set两节点共16个train96/validation400 correct/other面板已准备，materialize/evaluate脚本及请求语义核对通过；
  主/局部学习对比已完成，paired成功集合持续汇总完整面板；未选checkpoint，Test及sealed controls未用。
- 最新启动前/data1 quota886.7GiB/1TiB，两臂新增峰值预算40GiB；3个已集成临时实现工作树清理完毕，
  source/data/env与正式冻结运行面保留。接续训练和各新GPU评测启动前按实际变化检查额度。

### 首个100节点与评测接续

- ordered第100步完整checkpoint已保存，并由formal materialization检查器验证；0/100主动作留出各24task×128queries，
  局部留出各24task×16clips×8noise均完整无梯度。主FM均值0.151461→0.114319，局部1.308081→1.164801；
  这是拟合进展，尚不能推断有序视频增益，匹配frame_set及闭环结果仍待完成。
- 100节点train/validation correct及same-task-other四个LoRA库均已封存，各96/400条件；
  other按固定映射复用同条件LoRA，不平均生成权重。物化进程已退出。
- train96 correct完整40/96、breadth19；配对source15/96，R/G/L=12/28/3，churn31、J=.27907，
  相对source task-cluster95%差额区间[.166667,.354167]。这是训练任务适配收益，尚无匹配frame_set有序增量结论。
- train96 other完整35/96、breadth17，source R/G/L=11/24/4。correct/other四suite S/O/G/L分别9/9/15/7、9/8/12/6；
  换视频成功集合重合33、churn9、J=.78571，correct−other差额95%CI[-.010417,.114583]。两臂进程均已退出。
- ordered 200节点四个LoRA库均已封存，物化进程退出；100/200所有train96/validation400 correct/other条件库齐备。
- validation100 correct完整45/400，source47/400；S/O/G/L=0/37/0/8、breadth4，source R/G/L=4/41/43，
  churn84、J=.04545，差额task-cluster95%CI[−.2825,.22]。source的41个Goal成功全部丢失；本节点未显示总体迁移改善。
  原件及逐task/suite记录在统一`paired_summary.json`。单臂不能判断有序结构增量，匹配frame_set及相邻200仍待完成。
- validation100 other完整48/400，S/O/G/L=0/39/1/8、breadth5，source R/G/L=4/44/43、churn87、J=.04396。
  correct/other成功重合36、churn21、J=.63158，差额task-cluster95%CI[−.025,.005]。两组正确视频总体均接近source，
  但成功任务组成大幅改变；仍未显示总体迁移改善。两项400评测及进程均已完成。
- ordered200 validation correct完整30/400，S/O/G/L=1/29/0/0、breadth3；source R/G/L=2/28/45、churn73、J=.02667。
  相较100步45/400，R/G/L=21/9/24、churn33、J=.38889，差额task-cluster95%CI[−.1075,.02]；
  Object37→29、Long8→0、Spatial0→1，Goal仍0。总体迁移继续偏弱；匹配frame_set及200换视频结果待齐。
  400rows/36jobs完整、评测1769.79秒，进程退出。
- frame_set100 validation correct完整58/400，S/O/G/L=0/43/1/14、breadth5，source R/G/L=5/53/42、churn95、J=.05。
  ordered100为45/400，相对frame_set R/G/L=39/6/19、churn25、J=.609375，task-cluster95%差额CI[−.06,−.01]。
  Object−6、Long−6、Goal−1、Spatial0，没有suite净收益；这一节点的有序结构迁移劣于充分匹配的无序参照。
  本节点不满足预登记迁移资格；继续补齐其余节点与换视频证据，不据此启动pureFM第三臂或最终controls。
  42jobs/400rows完整、评测1180.75秒，全部进程退出。
- frame_set200 validation other完整26/400，S/O/G/L=1/22/3/0、breadth4；source R/G/L=5/21/42、churn63、J=.07353。
  对应ordered200 other及frame_set200 correct仍在评测；与frame_set100 other的相邻比较见下。
  42jobs/400rows完整、评测1213.09秒，全部进程退出，所用3张GPU已释放。
- frame_set100 validation other完整53/400，S/O/G/L=0/43/1/9、breadth5，source R/G/L=5/48/42、churn90、J=.05263。
  ordered100 other48对53，R/G/L=40/8/13、churn21、J=.65574，task-cluster95%差额CI[−.025,−.0025]；
  Object−4、Long−1，其余suite相同。100节点两组正确视频的有序差额−13/−5，区间均严格为负，未兑现跨视频有益过程预测。
  frame_set100自身correct/other重合45、churn21、J=.68182；other100→200为53→26，R/G/L=21/5/32、churn37、
  J=.36207、95%CI[−.17,.005]，Long9→0、Object43→22。各项原件与逐task/suite统计在统一paired_summary。
  本次30jobs/400rows完整、评测3344.79秒，三worker exit0、全部进程退出；仍不启动pureFM第三臂或最终controls。
- 当前归因边界：局部FM优势属于共享读取器与专用局部头的整体；局部头拥有自己的时间Key投影，不更新Compiler的时间投影。
  因此它尚未单独证明共享E已获得可迁移的完整任务过程，不能把Compiler指定为唯一失败点。
  新局部RGB—动作真实配对区别于旧跨episode辅助FM；其增量贡献尚无同图纯FM消融，不能用跨配方历史分数代替。
- 收尾期间已复核扩展任务历史：75/73-task确有新增任务映射；更近的完整输出73-task／18-target比较
  也未支持更多任务自然解决共享迁移。证据与未排除范围见findings§68，不由本轮负结果自动恢复95-task。
- correct评测进程已退出，frame_set100/200物化随后均完成；当前资源接续见下节。
- 精确命令、两节点preflight与预算记录在`runs/analysis/local_action_grounded_20260912/ordered/step100/`。
  此次quota890.4GiB/1TiB，study已用4.4GiB，原40GiB新增峰值预算剩余35.6GiB，投影926.0GiB；
  gpu02可用RAM321GiB。两个共驻设备原仅有约0.2GiB低利用率context，未改动其它用户作业。
- 两个完整训练面板及全部配对统计在`runs/analysis/local_action_grounded_20260912/paired_summary.json`；
  首个correct面板完成时quota892.3GiB、study6.3GiB、原40GiB预算剩余33.7GiB，投影926.0GiB。
- ordered 200主留出FM为0.108236、局部0.810114；0/100/200各有完整24task主诊断及384局部片段，均无梯度。
  它们是学习证据，不能替代匹配frame_set及完整闭环比较。
- ordered退出后，frame_set已从同一clean pushed detached 5f4f440c在gpu01:0/4/5/6、world4 fresh启动，
  tmux `ember-local-action-frame-set`，精确命令及新preflight在`runs/analysis/local_action_grounded_20260912/frame_set/`。
  启动前四卡均无进程、0MiB，节点可用RAM460GiB；quota896.1GiB、study10.028GiB，40GiB预算剩余29.972GiB，投影926.072GiB。
  actual run contract确认fresh/world4/source trainable=0；0步24主诊断及384局部诊断完整，现已完成200步，匹配分析见下。
- validation100 other接续前，两节点preflight确认gpu02:0仅223MiB低利用率context；quota898.4GiB、study12.418GiB，
  原40GiB预算剩余27.582GiB，投影925.982GiB。精确命令与资源记录在`ordered/step100/eval_validation_other_launch.json`。
- 下一步补齐两臂100/200节点的剩余闭环面板。
  当前没有selected checkpoint、sealed controls或Test使用，goal未完成。

### 100/200步匹配学习对比与评测接续

- frame_set第100步完整checkpoint及两类留出诊断已保存，formal checkpoint检查通过。
  两臂前100步实际400次条件曝光、25,600主queries、400局部片段、3,200局部noise draws逐字段匹配，
  包括task、teacher/action episode、frame、局部起点、noise/query seed及权重。
- 主动作留出ordered/frame_set为0.114319/0.114132；以frame_set−ordered定义改善，差额−0.000186888，
  task-cluster95%CI[−0.000422633,0.000041479]，6/24任务正向。当前没有可信主动作拟合改善。
- 局部留出为1.164801/1.164866，ordered改善0.000065278，95%CI[0.000005210,0.000128710]，16/24任务正向。
  差异很小，不能单凭此宣称有益过程已获得。两臂0步诊断一致；均只读train24留出动作，不读取validation/test动作。
- 原件`runs/analysis/local_action_grounded_20260912/step100/learning_comparison.json`包含逐task/suite与匹配曝光。
  当前不据此扩大局部头、追加消融或选checkpoint。
- frame_set完整200更新已结束，用时3307.92秒；100/200 checkpoint、0/100/200两类诊断齐备，formal检查通过，进程退出。
  两臂各800次条件曝光、51,200主queries、800局部片段、6,400noise draws及实际采样字段全部匹配。
- 200主动作留出ordered/frame_set为0.108236/0.108274；ordered改善0.000037421，95%CI[−0.000154784,0.000225352]，
  12/24任务正向。两节点主动作拟合均未形成可靠有序增量。
- 200局部留出为0.810114/0.811286，ordered改善0.001171932，95%CI[0.000908669,0.001444574]，24/24任务正向。
  局部优势一致但幅度仍小；只有配对闭环才能判定是否成为有益过程。原件`step200/learning_comparison.json`与100节点同根。
- 两臂100/200共16个LoRA库全部封存，train/validation每库各96/400条件；所有物化进程已退出。
  ordered200 train correct完整46/96，S/O/G/L=15/12/12/7、breadth17，source R/G/L=13/33/2、churn35、J=.27083。
  相较自身100步40/96，R/G/L=31/15/9、churn24、J=.56364，差额task-cluster95%CI[−.052083,.1875]；
  总分增加但breadth19→17，尚无可信相邻改善，完整匹配frame_set结果待齐。96rows/36jobs全部完成，进程退出。
- ordered200 train other完整49/96，S/O/G/L=16/12/14/7、breadth18；source R/G/L=14/35/1、churn36、J=.28。
  自身100→200为35→49，R/G/L=28/21/7、churn28、J=.5，task-cluster95%CI[.052083,.25]，显示这一训练面板的相邻适配改善。
  200 correct/other重合43、churn9、J=.82692、差额95%CI[−.083333,.020833]；不能替代匹配有序增量及validation迁移。
  96rows/36jobs完成、进程退出。
- 首个匹配训练结果：frame_set100 correct完整39/96，S/O/G/L=8/9/15/7、breadth19，source R/G/L=11/28/4。
  ordered100 correct为40/96；相对frame_set R/G/L=35/5/4、churn9、J=.79545，task-cluster95%差额CI[−.03125,.052083]。
  两臂仅Spatial净差1，其余suite总数相同，尚无可信有序结构增益。原件及逐task数据在`paired_summary.json`；96rows/36jobs完成、进程退出。
- frame_set100 other完整38/96，S/O/G/L=7/9/15/7、breadth19；ordered100 other35/96，相对它R/G/L=30/5/8、
  churn13、J=.69767，差额task-cluster95%CI[−.09375,.03125]。有序组两套视频对照差额+1/−3均未形成可信增益。
  frame_set自身correct/other重合33、churn11、J=.75；36jobs/96rows完成、进程退出。
- frame_set200 correct完整42/96，S/O/G/L=14/12/12/4、breadth17；source R/G/L=13/29/2、churn31、J=.29545。
  ordered200 correct46/96，相对frame_set R/G/L=39/7/3、churn10、J=.79592，差额task-cluster95%CI[−.020833,.114583]。
  净收益Long+3、Spatial+1，另两suite相同；当前仍无可信有序结构增量。frame_set自身100→200 R/G/L=29/13/10、churn23、J=.55769。
  该correct面板36jobs/96rows完成、进程退出；200换视频匹配见下，validation继续评测。
- frame_set200 other完整45/96，S/O/G/L=13/11/14/7、breadth17；source R/G/L=15/30/0、churn30、J=.33333。
  ordered200 other49/96，相对frame_set R/G/L=41/8/4、churn12、J=.77358，差额task-cluster95%CI[−.03125,.114583]。
  净收益Spatial+3、Object+1，另两suite相同；frame_set自身换视频重合39、churn9、J=.8125，other相邻重合30、churn23、J=.56604。
  两臂8个训练面板全部完成；100两视频差额+1/−3、200两视频差额+4/+4，四项区间均跨零，尚无可信有序结构增量。
  原件与所有task/suite、相邻及换视频统计均在`paired_summary.json`；最后36jobs/96rows完整、进程退出。
- 当前已完成14/16完整闭环面板，16项均已启动。gpu02:0评测ordered200 validation other，
  gpu02:2评测frame_set200 validation correct，两项各3 workers；gpu01本轮全部评测进程已退出。
  剩余两项评测各使用一张GPU；精确命令与各次两节点preflight在对应step launch records，不追加实验占用已释放设备。
  最新接续时quota910.9GiB、study24.871GiB，原40GiB预算剩余15.129GiB，投影926.029GiB；大型checkpoint/bank写出已完成。
- 新接续前gpu01四卡均0MiB、无进程，节点可用RAM460GiB；quota906.8GiB、study20.803GiB，
  原40GiB预算剩余19.197GiB，投影925.997GiB。不存在额外训练或新配方，待全部登记闭环结果后裁决。

### Owner最新纠正与执行调整

Owner指出近几小时没有根本进展，要求调整负结果后的分析与修正方式。此前去蒸馏、额外读出与VL适配虽各有
限定结论，但没有充分收窄原主假设；“尚不能否定整体”不能继续充当追加投入的依据。综合初判见findings§62。
第11节已完成原登记200步及100/200配对证据。综合裁决见findings§65：主方案在训练侧也未显示可重复的有序增量，
不能再默认“有益过程已经学会、只需保持或编译”。停止读出容量、辅助loss、VL层位及一般保持小修的自动串联；
下一步应正面解释过程获取的缺口。文献中的额外视频先验仅为待分析候选，不自动启动；无需另设人工审批，不恢复旧任务或改变资格标准。

### 过程获取的监督关系复核（2026-09-12）

已完成[下一项机制分析与可审阅提案](docs/video_process_acquisition_analysis.md)：旧Action-Forecast预测未来，
v5/v6使用固定probe，Stage0/G2虽有真实动作标签，却明确来自另一episode并按进度映射；
所查证据没有建立“所观察的局部变化—造成它的实际动作”配对监督。这区别于再次扩大辅助执行读出。
推荐继续推导action训练池内的局部动作反演，共享表示最终仍由跨episode完整LoRA FM学习；
局部预测强弱、LoRA收益和跨task迁移分别裁决，不能以反演头成绩完成goal。

此前待确认的范围现已由Owner最新授权覆盖：辅助分支只使用action池16–41自身RGB及动作训练共享读取器，
标签不进入encoder／Compiler，主LoRA查询继续跨episode。已登记完整新设计，不再等待同一事项批准。
实现复用唯一运行面并退役旧辅助执行／蒸馏支路；其它旧候选保持关闭。

等待范围确认期间已完成只读准备：提案§8规定全帧无序参照不得通过末帧选取或端点标记泄漏先后；
§9依据固定官方生产代码及四suite训练样本确认post-action RGB，区间obs[p]→obs[q]应配actions[p+1:q+1]。
当前主FM仍使用原同索引合同，时间差的行为影响未测，不作为历史低分根因。仅长度元数据表明四帧局部候选
新增观察量约为平均teacher帧数的11.27%，不是墙钟实测。上述只读准备已完成，新实现与profile据最新授权推进。

### 当前执行与完整结果

- 第11节fresh200训练完成，正式用时3339.18秒；800条件/51,200queries与teacher_vl参照逐字段匹配，100/200完整checkpoints齐备，训练进程退出。source保持冻结，峰值37.83GiB。
- held student FM100/200=.114742605/.107176777；相对参照改善.000179162/.000175662，task-cluster95%CI分别[.000050021,.000325814]/[-.000226025,.000602552]，18/24和13/24任务改善。差额很小，200区间跨零，不能替代行为证据。原件`direct_fm_vl/step{100,200}/learning_comparison.json`。
- 100训练两臂完整37/40（各96），teacher_vl参照34/37；correct R/G/L28/9/6、breadth19，other33/7/4、breadth18；差额CI[-.052083,.114583]/[-.052083,.125]均跨零。自身换视频重合33、churn11、J=.75；两臂进程均退出。原件`direct_fm_vl/step100/train_paired.json`。
- 200训练两臂55/59（各96），主要参照56/56，差额CI[-.072917,.052083]/[-.03125,.09375]均跨零；S/O/G/L=17/15/14/9、17/18/14/10，breadth均20。自身100→200为37→55、40→59，R/G/L28/27/9、32/27/8；200换视频重合52、churn10、J=.83871。原件`direct_fm_vl/step200/train_paired.json`。
- 100 validation两臂60/55，主要参照61/61；差额CI[-.04,.035]/[-.0475,.02]，S/O/G/L=1/49/3/7、0/47/3/5，breadth7/6。Object+7/+2伴随Long−8/−5，未形成可信迁移收益；source47仅保留8/5。换视频重合47、churn21、J=.69118。原件`direct_fm_vl/step100/validation_paired.json`。
- 200 validation两臂64/54，主要参照62/64，差额CI[-.04,.0425]/[-.0575,.005]；S/O/G/L=2/46/6/10、1/41/6/6，breadth均7。相对auxiliary_fm为−8/−13，相对原main为−10/−20；没有可靠的跨视频迁移改善。
- 自身100→200 correct R/G/L38/26/22、churn48/J=.44186；other35/19/20、churn39/J=.47297。相较teacher_vl相邻J=.35165/.38889有局部提高，但总分与换视频收益不成立。200两视频重合45、churn28/J=.61644；correct−other的+10区间[.0025,.05]比较的是两组合法正确视频，不能称为内容/顺序收益。
- 100/200全部8个banks、8面板/1,984rows完成，study累计64面板/15,872rows；最后200两臂各42jobs/400rows、0 job errors，耗时1202.94/1213.93秒。两节点进程均退出，峰值6GPU；无selected checkpoint、Test或最终sealed controls。
- 原clean detached执行checkout已在全部worker退出、Git clean与main ancestry核对后清理；正式代码由`9b21b0ed`、全部checkpoint和结果保留，无运行树待恢复。
- **第11节及当前候选关闭：**不延长、扫描或自动补frame_set。完整逐task/suite、source、相邻、换视频统计保留于统一`paired_summary.json`，限定裁决`direct_fm_vl/bounded_200_decision.json`。局部训练获取与重合提高保留，但目标未完成；未训练本配方的匹配无序参照，不能扩大为所有纯FM/VL的时序否证。
- 最新launch前两节点live检查通过，/data1 quota886.0GiB/1TiB、当前run13GiB，全部banks落盘后剩余为小型rows/logs，仍低于原893.6GiB投影峰值，共享83TiB；精确命令与资源原件`direct_fm_vl/step100/{validation_correct,validation_same_task_other}_launch.json`及`step200/{materialization,train_correct,train_same_task_other,validation_correct,validation_same_task_other}_launch.json`。本轮未新增候选或额外面板。
- 第11节无辅助路径相关83测试通过，配置与8个评测请求的科学字段匹配。初始化按共同CPU构造顺序保持VL随机流；reader在进入GPU/state optimizer前丢弃，无新的执行模块或第二trainer。
- 真实最长task38/demo0 93frames：micro16 OOM已结束；micro8两次更新通过，第二次16.86秒/64queries（3.795queries/s）、峰值37.81GiB。Writer/Action/VL均有实际梯度，source冻结、source辅助forward=0。固定frame8/policy8；profile无checkpoint，原件`runs/analysis/video_functional_20260911/direct_fm_vl_profile/`。
- 第11节已从clean pushed detached `9b21b0ed34d5501f931b8c44f8922c2e6d254849`启动，执行checkout `.codex/worktrees/direct-fm-vl`；gpu01:0/4/5/6，world4，tmux `ember-direct-fm-vl`，fresh200/800条件/51,200queries，保存100/200。实际run contract确认reader0/source0、Writer337,568,000/Action626,688/VL921,600、NCCL/NUMA及物理8；初始held FM=.154849362与参照一致，已进入有效更新。
- 100/200全部评测请求已准备。启动前本人用卡0→4≤6，/data1 quota873.6GiB/1TiB，新增峰值预算20GiB、投影893.6GiB（含checkpoints/banks/临时与执行checkout），共享83TiB。精确命令与现场记录`runs/analysis/video_functional_20260911/direct_fm_vl/launch_contract.json`，输出`runs/outputs/video_functional_20260911/direct_fm_vl/`。
- 首次启动脚本误指旧output，fresh-run合同保护在任何训练更新前拒绝；已修正参数并刷新GPU后从fresh重启。旧formal结果未覆盖，失败日志与精确命令保留在新analysis目录`startup_output_path_rejected.{log,json}`，实际新run contract已核对。

- 第10节完整Z/R双路VJP已接入唯一observer；跨update只缓存pre-Gemma embeddings，新VL Meta在共同模块后初始化，新增921,600参数。旧KV缓存入口退役，model schema v2拒绝旧checkpoint；无第二trainer或执行adapter。
- CPU相关168项通过，配置小改后3项复核通过。真实native identity Z差0，两条VL梯度均非零，联合重放与直接梯度一致，source trainable=0；真实学习第2次后Writer/Action/VL/reader均有梯度。
- 最长合法视频task38/demo0，93frames/64queries：frame_chunk4/8分别24.97/21.55秒，后者2.970queries/s、峰值37.84GiB。选择frame8、policy micro8；profile只运行gpu01:4且已退出，无保留训练checkpoint。原件`runs/analysis/video_functional_20260911/teacher_vl_profile/`。
- 结构变更净增15行active source/test、无新文件；配置检查复杂度从28降为27，其余guard提示为既存大文件/函数与目录规模，未增长相关复杂职责。实现已集成并推送；formal冻结`a89648740c28d1a2f4aff07b6e42bc7b35e44223`；已结束的clean detached执行checkout在核对main ancestry后清理，全部formal资产保留。
- 第10节fresh训练已完成，原gpu01:0/4/5/6 world4进程全部退出；200updates/800条件/51,200queries与参照曝光匹配，训练4010.96秒，100/200完整checkpoints已保存。两组读取Meta共同学习、source trainable=0，NCCL/NUMA与逻辑batch已由实际run contract确认。
- 启动前两节点本人占用0，启动后4≤6；/data1独立quota861.2GiB/1TiB，新增峰值预算20GiB、投影881.2GiB，共享83TiB。精确命令与资源记录`runs/analysis/video_functional_20260911/teacher_vl/launch_contract.json`，日志`train.log`；输出`runs/outputs/video_functional_20260911/teacher_vl/`。100/200全部物化与评测请求已准备，训练及评测共享Owner全局额度。
- 100节点实际400条件/25,600queries与第8节逐字段匹配；held student=.114921766，参照.114956222，改善仅.000034455（task-clusterCI[.000004203,.000064595]）；reader=.153159816、差额CI跨0。差额很小，不能据此推定闭环收益；原件`teacher_vl/step100/learning_comparison.json`。
- 100四个banks已完整物化：train96与validation400各正确/同task另一视频，other复用相同条件LoRA而保留不同state-video映射。训练两臂已在gpu02:0/3各3 persistent workers完成并退出；原件`teacher_vl/step100/{materialization_launch,train_correct_launch,train_same_task_other_launch}.json`及对应preflight/logs。
- 100训练两臂完整34/37（各96），参照42/38；correct保留31/新增3/丢失11、差额CI[-.166667,0]，other33/4/5、CI[-.083333,.0625]。自身换视频重合30、J=.73171；100节点未见训练收益。原件`teacher_vl/step100/train_paired.json`。
- 200留出student=.107352440、相对参照改善−.000013986（CI[-.000078555,.000057563]）；reader=.149220013、微小改善.000016497，不能作为目标证据。全部800条件/51,200queries匹配，原件`teacher_vl/step200/learning_comparison.json`。
- 100/200全部8个banks、8面板/1,984rows完成；study累计56面板/13,888rows。两节点本轮进程已退出，峰值合计6张卡。最后200 validation两臂各2GPU×3 workers，均36jobs完整400rows；完整命令、资源与结果保留在`teacher_vl/step100`及`step200`。最近quota873.6GiB/1TiB、run13GiB，低于原881.2GiB投影峰值。
- 200训练两臂56/56（各96），参照55/55；净差均+1、CI均跨0，breadth21/20。100→200训练获取提高，但同节点换视频重合46、J=.69697，参照50、J=.83333。
- validation100两臂61/61，参照57/61；差额CI[-.02,.0425]/[-.0325,.025]。S/O/G/L=2/42/2/15、3/45/3/10，breadth8/7；correct的Long+7未在other复现（−3）。
- validation200两臂62/64，参照72/67；差额−10/−3，CI[-.0575,.0025]/[-.0375,.03]。S/O/G/L=2/42/6/12、2/45/4/13，breadth5/6；source47仅保留8/6，新增54/58、丢失39/41。
- 自身100→200 validation correct R/G/L32/30/29、churn59/J=.35165；other35/29/26、churn55/J=.38889。参照J=.32990/.34737，但correct保留仍32、获取更少，breadth8→5；不能把小幅J增加当作保持改善。200换视频重合44、churn38/J=.53659，差额CI跨0。
- **第10节裁决：**未形成值得追加的跨task收益与保持，不延长或扫描VL Meta rank/层位/LR，不自动补匹配frame_set。该因素并非input信息或全部VL学习的否证。原件`teacher_vl/bounded_200_decision.json`，全部逐task/suite、source及相邻配对在统一`paired_summary.json`。
- 当前goal未完成。第11节只移除L_R的fresh200及配对闭环全部结束；不从本段历史启动下一轮训练。
- 此前Writer学习、动作头诊断、LoRA生成和闭环评测均已完成，无待恢复任务。第9节中层读出诊断也已完成并退出；它们不是当前训练的初始化或恢复来源。
- 去蒸馏配方固定rho=0、mu=1，fresh200更新/800条件/51,200queries，训练3967.10秒，峰值37.82GiB。完整100/200 checkpoints及全部800条件曝光匹配证据保留；source trainable=0。
- formal代码为clean pushed detached `ba4d1f4da16759a5ea1c5d7dec0dce1b0bd1b5e4`，原执行checkout已清理；原main/frame_set代码为`a81a38edd055034a4a080215bdc4362310500678`，其已结束checkout亦清理。配置解析及相关49测试通过，未改变原科学实现。
- 新配方8面板/1,984闭环rows全部完成；连同原40面板，study共48面板/11,904rows。精确命令、preflight、raw rows、aggregate和completion保留；统一逐task/suite及配对统计见`runs/analysis/video_functional_20260911/paired_summary.json`，裁决见`auxiliary_fm/bounded_200_decision.json`。

| 配方/更新 | train correct /96 | train other /96 | validation correct /400 | validation other /400 |
|---|---:|---:|---:|---:|
| 原main/100 | 35 | 38 | 72 | 71 |
| 去蒸馏/100 | 42 | 38 | 57 | 61 |
| 原main/200 | 45 | 47 | 74 | 74 |
| 去蒸馏/200 | 55 | 55 | 72 | 67 |

- 去蒸馏200训练两臂S/O/G/L=17/16/13/9、17/17/13/8，breadth21/20；对main的correct净增10，task-cluster95%CI[.020833,.1875]，other净增8、CI[-.020833,.197917]。同checkpoint两视频重合50，J=.83333。训练获取改善保留为正证据。
- 200 validation correct/other S/O/G/L=3/47/5/17、2/47/6/12，breadth7/6。对main净差−2/−7、CI[-.0875,.06]/[-.0925,.0425]；R/G/L37/35/37、34/33/40。Spatial/Long有局部收益，但Goal较main少15/12，不能以总获取替代迁移保持。
- 自身100→200 validation correct57→72：R/G/L32/40/25、churn65/J=.32990、CI[-.0675,.14]；other61→67：33/34/28、churn62/J=.34737、CI[-.075,.1125]。原main相邻J=.32727/.39423；没有可信保持改善。
- 200同task换视频other→correct67→72：R/G/L48/24/19、churn43/J=.52747、差额CI[-.0025,.03]；原main200两视频J=.66292。source47→新72/67只保留9/8，Goal task6原41→5/5，广泛保留未形成。
- 100对pureFM的validation只+1/+1，CI跨0；留出动作0/100/200 student FM=.154849/.114956/.107338，reader=.154849/.153161/.149237。动作拟合继续改善，不能替代闭环迁移或过程资格。
- 最后两项200 strict400分别耗时1202.78/1204.78秒，各42个queue jobs完整400rows。原新增存储预算20GiB，最近quota861.4GiB/1TiB、投影869GiB；未新增数据或模型副本。
- **本项裁决：**不延长去蒸馏训练，不自动启动同rho0 frame_set。两个节点、两臂validation都低于main，局部Spatial/Long正证据尚不足以证明保持改善；训练收益不能成为追加资格训练的唯一理由。
- **已完成诊断：**第9节仅改变固定表示后的功能读出：在Action Expert中层读取E，由后续冻结原生层转成动作；原Writer完全冻结，32epochs已全部完成。新头未形成更强教师，不追加拟合、层位扫描或自动重训Writer。没有新的Writer训练、selected checkpoint或最终sealed controls；Test继续封存，goal保持未完成。
- 新诊断从clean pushed detached `b1d3fa25`执行，已结束checkout随结果归档清理；临时单模块335行已退役，复用既有reader与原生cache。语法/CLI和真实smoke通过，source identity保持、输出/query/Value梯度存在、source无梯度；缓存/完整forward的最大FM差.000381，峰值9.93GiB。
- 新launch与完整资源记录在`runs/analysis/video_functional_20260911/native_reader_launch.{json,sh}`，运行日志`native_reader.log`、输出`native_reader_diagnostic/`。32epochs/192updates固定，1536支持queries复用、768留出queries无梯度；没有新的Writer训练或部署checkpoint。
- 启动前两节点本人作业0，启动后1≤6。/data1 quota861.6GiB/1TiB，新增预算<1GiB、投影862.6GiB，共享83TiB；gpu01可用RAM444GiB，缓存预计数十GiB且不落盘。两个旧任务运行树清理已核对clean与main ancestry，全部正式资产保留。
- 结构复核曾允许一个有明确退出条件的诊断入口；cache helper的循环/字段拼装复杂度17及writer目录27个文件为临时例外。问题关闭后已删除入口，未创建第二Writer运行面。
- 最终32epochs/192updates学习用时325.06秒；真实profile选择micro32约143queries/s，profile最高显存17.95GiB（包含micro64试测）。48条件实际video/action/query/RNG与旧诊断匹配，缓存/完整source FM最大差.000389，接受正常BF16路径差异；所有参照在同缓存路径重算。
- 新中层读出held0/4/16/32=.154897/.154597/.144469/.139792；旧拟合头=.140281，原LoRA学生=.110005。最终对旧头平均改善.000488、95%CI[-.003509,.003975]，15/24更好；对学生劣化.029787、CI[.021072,.039954]，24/24仍更差。
- 相对source改善.015083、CI[.010621,.020128]，22/24更好；16→32仍改善.004677、CI[.002959,.006544]。这些学习事实保留，但固定预算下没有更强教师证据，不能据末段获取无限追加。
- held原E相对跨task E优势.001718、CI[.001015,.002516]，19/24；E含语言/静态，仍非视频过程资格。完整原件`native_reader_diagnostic/results.json`，配对汇总`native_reader_analysis.json`。临时诊断入口随问题关闭退役，原代码由b1d3fa25及artifacts保留。
- 综合复核后，第10节只开放teacher侧PaliGemma VL Meta，保留第8节其余图和学习配方。历史审计曾提出该因素；现已按上方合同完成匹配检验，未获可信收益。两类弱头不证明输入无信息或唯一编译失败，goal保持未完成。
- Owner双节点GPU总上限持续有效：最多8张，空闲总数不超过10张时最多6张；全部训练、物化与评测共享。

### 已完成学习与验证

- 原Writer正式实验代码保存在Git `a81a38edd055034a4a080215bdc4362310500678`，已结束执行checkout已清理；原CPU套件378 passed，真实native梯度和最长93frames profile通过，source trainable=0。新诊断入口另以语法、结构检查及真实smoke验证，未改原Writer训练路径。
- main与frame_set均按登记完成300更新、1,200条件、76,800queries，checkpoint50/100/200/300完整保存；全部task/video/action/query/RNG/weight实际匹配。证据`frame_set/step300/exposure_alignment.json`。pureFM仅完成原登记100更新，无追加续训。
- main200/300续训段8657.20秒，frame_set对应段10470.90秒；frame_set峰值39.05GiB。所有LoRA banks及正式rows、aggregate、completion与恢复证据保留在`runs/outputs/video_functional_20260911/`。
- 40个完整面板共9,920条保留闭环rows，source参考train15/96、validation47/400；task/state/env/policy RNG配对通过。逐task/suite、breadth、R/G/L、churn、Jaccard和task-cluster bootstrap95%CI均见`paired_summary.json`。

| 配方/更新 | train correct /96 | train other /96 | validation correct /400 | validation other /400 |
|---|---:|---:|---:|---:|
| main/50 | 26 | 29 | 69 | 71 |
| main/100 | 35 | 38 | 72 | 71 |
| main/200 | 45 | 47 | 74 | 74 |
| main/300 | 43 | 46 | 32 | 32 |
| pure_fm/50 | 30 | 22 | 69 | 64 |
| pure_fm/100 | 41 | 40 | 56 | 60 |
| frame_set/50 | 29 | 24 | 66 | 67 |
| frame_set/100 | 35 | 38 | 73 | 74 |
| frame_set/200 | 46 | 47 | 69 | 61 |
| frame_set/300 | 48 | 44 | 27 | 31 |

### 当前裁决与后续

- 本轮未获得视频资格。50/100/200/300的main相对frame_set correct差额+3/-1/+5/+5，other差额+4/-3/+13/+1；primary正确视频95%CI均跨0。预注册必要门槛核对见`bounded_300_decision.json`，不以差额方向或单点峰值代替可信收益。
- main200→300 validation两臂均74→32，correct/other差额95%CI分别[-.18,-.0325]/[-.1925,-.0225]。frame_set相邻correct69→27、other61→31，CI[-.195,-.025]/[-.1425,-.015]。两方案后段均迁移退化，训练任务获取近似保持不能代替迁移保持。
- 已有train24留出动作100步main source/reader/student FM=.154849/.153163/.116660；251–300同批为.157011/.148191/.102797。辅助读出明显落后学生，尚无“好教师已学会、仅编译失败”的证据；这是诊断线索，不是退化的单一根因。
- 固定表示probe的support reader FM .150983→.136589（24/24改善），held .149662→.140251（23/24改善，平均降幅.009411、task-cluster95%CI[.006285,.012963]）；原学生held .110025，24/24任务仍优于reader。当前接口能学习可迁移到同训练task留出动作的修正，但尚无更好功能教师。
- held换入跨task E后FM .148876，相对原条件高.008625、CI[.005880,.011906]；E含语言与静态内容，此接口依赖不等于视频动态因果资格。逐task/节点见`reader_fit_analysis.json`及probe原始results。
- 同批损失可直接恢复输出误差内积：`<S-y,S-T>=(L_C+L_D-L_R)/2`。main/frame_set 151–200、251–300各50/50更新为负，汇总cos≈−.15。它限定在预测空间，不能推出参数梯度或Adam更新相反，更不能把迁移退化单因归给蒸馏。现有pureFM同时移除了两种辅助作用，故已单独完成去蒸馏比较；其训练收益未转为可信validation收益，不增加reader优化轮数或无依据延长。
- 当前goal仍未完成；无selected checkpoint，最终sealed视频controls未执行，Test继续封存。完整阶段结论与历史细节已转入`docs/research_history.md`和`findings.md`。

## 暂停时点的已完成实验与未完成范围

无变化参照候选从冻结`64eba75bcf316a1f6792ab8ddf5eb1aaefded378`完成fresh200更新/51200queries，
用时3327.11秒，100/200完整checkpoint保留。唯一主要变化为C局部更新采用同gap/角色/窗口的无变化参照，
保留train24、每task两个K1条件各32queries、普通正样本FM、source冻结；没有RL、错误视频惩罚或任务覆盖变化。

训练task独立留出动作FM由.151451降至.109323，24/24任务改善。固定teacher46、states32–35的train100→200
correct为36→45，各96；相邻保留22/新增23/丢失14，churn37/J=.37288。200正确/换视频/同suite错/跨suite错/乱序/静态
为45/42/41/37/41/40，各视频差额的task-bootstrap95%区间均含0。200 validation完整为34/400，source47、旧C20050；
Spatial/Object/Goal/Long为4/24/5/1，breadth5。初步训练条件分化未成为可信视频机制收益或迁移改善。

100 validation在Owner暂停时终止，没有可报告的strict400成绩；300/400未启动。首段不能证明候选收敛或整个机制被否定。
`runs/analysis/video_change_reference_20260911/owner_pause.json`记录定向停止及checkpoint保留；本节不是实时GPU进程检查。

此前完整C冻结诊断共1472条新闭环及14336次配对动作预测：C200 correct42、两错46/46、乱序45、静态49、S51、source15，
各96；固定语义S后correct/static过程同为15/32。静态首帧仍产生正常视频.692–.935倍中心化P4，支持一个具体表示性质，
尚非全局根因。无变化参照修正已在真实native输入上削弱该性质，但行为收益未被证明。

| 暂停前路线 | validation /400 | 训练任务观察与边界 |
| --- | --- | --- |
| R，off图+两个独立K1条件 | 100/200/300/400=37/83/63/85 | train200/400=46/56，各96；同query预算旧off400=126/56，未获迁移收益 |
| C，语义条件化过程消费 | 100/200=43/50 | 轮换held视频train96=49；固定teacher46机制面板42，不能混用 |
| S，无序帧集合参照 | 100/200=59/45 | 轮换held视频train96=52；固定teacher46机制面板51；不是单帧static模型 |
| off7续训 | 400/500/600=126/73/54 | 轮换held视频train96=56→64；训练获取与未见task能力分离 |

R/C/S及两项续训共4480条闭环，原件根`runs/analysis/video_consumption_20260911/`；C诊断根
`runs/analysis/video_mechanism_20260911/`；无变化参照根`runs/analysis/video_change_reference_20260911/`。
远程副本统一由新专家材料的index定位，`runs/`路径本身不表示已经发布。

95-task提案未启动GPU实验，其两个未提交worktree原样保留，不集成或清理。精确历史和旧合同由Git与
[research_history](docs/research_history.md)保留。后续执行只能依据Owner新授权及新登记，不能沿本文历史段恢复。

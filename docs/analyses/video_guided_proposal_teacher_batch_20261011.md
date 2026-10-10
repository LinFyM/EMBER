# 原rank128八教师完整结果与参数化交接（2026-10-11）

原八条教师及全部160/480配对读出已完成，没有触发“全批缺有益供给”的停止线。独立audit中，MT为16/128，教师160为74/128，教师480为83/128。两节点分别相对MT净增58/67；480仍保留15个MT成功、增加68个、丢失1个。这个结果支持强MT在当前四训练任务上可由合法训练专用监督继续改善，尚不支持G生成控制、视频理解或held迁移。教师消费有标签train数据、只作为共享训练端点，其成绩不是EMBER部署成绩。

Owner已开放MT合并冻结基底及小rank任务残差，并要求后继从首次G更新混合真实不同parent/H。按main最新裁决，本次在完整教师交付点结束；即使有正例也不自动进入原rank128 G320/π。rank8是优先后继，16教师、bank尺度和G480等准备稿数值尚未定案或启动。

## 实际合同、覆盖与原件

- 唯一ROOT：`/data1/user/ymdai/ember_runs/video_guided_proposal_writer_20261010/pilot`。科学合同来自`aef01ff448e6ba81a896c8328f34aceb56622cd8`，后续52预算及G前交接、多parent要求逐条保存在ROOT/run_contract.json。
- source1000/冻结source normalization、MT300、原38-target/rank128、76因子/10,297,344值；仅合法train tasks12/29/32/38，按已有同source专家/接口覆盖预先选定，不按本次结果挑task或削弱MT。每task空H与一次真实MT episode后H两个事件，8条fresh教师各480更新；112跨episode action queries/步，4×28逻辑流，offset1。AdamW lr1e-4、betas(.9,.95)、eps1e-8、wd1e-4，clip1，固定MT坐标prox1e-3/D；rec/keep只按真实标签启用。
- 学习前`panel.json`固定视频/初态/各角色RNG。每事件parent/160/480各selection4+audit16，共480原行。selection决定q_T，audit从未用于教师拟合、H或偏好选择。两个教学事件复用task的初态池，128 audit配对单位仅64个不同task/init组合，不能称128个独立初态或正式paired400。
- 教学condition池与动作query不是全episode留出：每条教师仅排除其自身教学episode，其余49条合法train episode均被查询。没有Val/Test/held标签、T模型/缓存、新教学condition动作轨迹拟合。
- CPU读回、曝光/恢复/逐任务/suite与success集合使用上述ROOT下`analysis/teacher_decision_readback.json`、`analysis/teacher_raw_rows.jsonl`和`analysis/teacher_raw_rows.csv`。`teacher_summary.json`为冻结9a消费者原汇总；event readout/aggregate.json及单行JSON保留全部480行。

## 全部教师及不利相邻变化

下表每个三元数为固定MT / 教师160 / 教师480；相邻R/G/L为160→480，均没有挑选最好节点。

| Task / event | 教学ordinal | H / rec / keep | selection（/4） | audit（/16） | 相邻R/G/L | q_T(160,480) |
| --- | ---: | --- | --- | --- | --- | --- |
| 12 / 0 | 42 | 空 / 0 / 0 | 1 / 3 / 4 | 4 / 13 / 16 | 13 / 3 / 0 | 0.500, 0.500 |
| 12 / 1 | 9 | 真实MT / 1 / 0 | 1 / 2 / 3 | 5 / 14 / 15 | 13 / 2 / 1 | 0.500, 0.500 |
| 29 / 0 | 1 | 空 / 0 / 0 | 1 / 4 / 2 | 0 / 5 / 5 | 3 / 2 / 2 | 0.500, 0.500 |
| 29 / 1 | 33 | 真实MT / 0 / 0 | 0 / 0 / 1 | 0 / 11 / 6 | 6 / 0 / 5 | 0.125, 0.875 |
| 32 / 0 | 6 | 空 / 0 / 0 | 1 / 4 / 4 | 3 / 13 / 15 | 13 / 2 / 0 | 0.500, 0.500 |
| 32 / 1 | 16 | 真实MT / 1 / 0 | 0 / 3 / 3 | 2 / 13 / 12 | 10 / 2 / 3 | 0.500, 0.500 |
| 38 / 0 | 48 | 空 / 0 / 0 | 0 / 0 / 3 | 0 / 2 / 5 | 1 / 4 / 1 | 0.125, 0.875 |
| 38 / 1 | 29 | 真实MT / 1 / 0 | 0 / 1 / 1 | 2 / 3 / 9 | 2 / 7 / 1 | 0.500, 0.500 |

所有8事件的两个节点在该audit池均相对MT净正，但selection并非准确的用途排序：task29/event1的160在selection为0/4而audit为11/16，480在selection升至1/4、audit降至6/16，q_T按合法selection仍偏重480。task38/event0在selection偏好480，audit也由2升至5。保留这些错位，不以audit重新改标签分布。

全部selection的MT/160/480为4/17/21（/32）；audit为16/74/83（/128）。160对MT为R13/G61/L3、churn64；480为R15/G68/L1、churn69。160→480为R61/G22/L13、净+9、churn35、Jaccard0.635417。总量上升伴随真实成功交换，task29两事件合计16→11、task32/event1为13→12，不能称逐例稳定保持。q_T按selection权重对应的audit经验期望为77.75/128，只是标签bank的加权读回，不是一套部署LoRA的成绩或逐行success union。

| Task / suite | audit分母 | MT | 160 | 480 | 480对MT R/G/L |
| --- | ---: | ---: | ---: | ---: | --- |
| 12 / libero_object | 32 | 9 | 27 | 31 | 9 / 22 / 0 |
| 29 / libero_goal | 32 | 0 | 16 | 11 | 0 / 11 / 0 |
| 32 / libero_10 | 32 | 5 | 26 | 27 | 5 / 22 / 0 |
| 38 / libero_10 | 32 | 2 | 5 | 14 | 1 / 13 / 1 |

Object为9→27→31（/32），Goal为0→16→11（/32），Long为7→31→41（/64）；Spatial没有覆盖。四task在两节点均有相对MT净增，四task均有教师成功；这不外推成36任务、四suite或held泛化。

## 参数、原生函数与实际学习身份

8条教师均实做480更新，共3,840更新、430,080 demo query曝光；每条53,760 queries、49个合法episode。三个rec事件各480次真实功能目标，总计1,440 rec目标曝光；keep全部缺失。所有更新编号连续、逻辑112/offset1/4×28、教学episode排除和finite检查通过；每条的独特demo/frame数量、49episode曝光、首末40步loss及真实阶段吞吐保留读回原件。loss变化不作为能力判据。

| Task / event | MT尺度编辑RMS 160→480 | 同初态原生前5动作RMSE 160→480 | 480 ΔA / ΔB L2 |
| --- | --- | --- | --- |
| 12 / 0 | 0.182732 → 0.325995 | 0.065280 → 0.071231 | 6.072393 / 6.002479 |
| 12 / 1 | 0.172121 → 0.313541 | 0.108782 → 0.104784 | 5.849720 / 5.749277 |
| 29 / 0 | 0.228468 → 0.392482 | 0.204182 → 0.221209 | 8.082435 / 7.024669 |
| 29 / 1 | 0.227158 → 0.390205 | 0.193586 → 0.224931 | 8.077106 / 6.984168 |
| 32 / 0 | 0.199709 → 0.343339 | 0.124512 → 0.137410 | 6.550183 / 6.295495 |
| 32 / 1 | 0.192512 → 0.337500 | 0.142202 → 0.146402 | 6.411988 / 6.183253 |
| 38 / 0 | 0.222882 → 0.356305 | 0.120061 → 0.136646 | 6.996214 / 6.530084 |
| 38 / 1 | 0.207792 → 0.347310 | 0.112081 → 0.122077 | 6.760486 / 6.372103 |

参数比较使用全部76因子；原生函数比较使用相同初态、首次实际观测proprio、环境/政策RNG及first noise的第一组normalized 5×7动作。全部paired metadata、共同政策噪声前缀和首proprio对应已从原行核对。它证明实际函数改变，不能解释全episode行为、证明视频知识或把大编辑当更优控制。task29/event1编辑更大而后节点成功更少，已保留。

真正学习的是训练专用教师的完整任务A/B副本。source权重被冻结；教学RGB/新condition不进入教师动作拟合。Gemma/action expert meta、G的Reader/编码/速度网及π尚无formal更新。此前profile的双meta梯度、source/meta执行隔离、完整参数输出、10-flow及STOP消费者只证明机制；没有把它们算作共享学习或视频/经历已学会。

四条实际MT episode均失败，16次现场专家恢复验证中3次成功，rec仅task12/32/38各1，task29缺失；keep0。v4保存真实运动前缀快照及跨进程后续同5动作验证，不用旧RGB/proprio补造H。空H/非空H同时改变教学排除、query RNG及部分rec标签，未做隔离因果比较，不能把两事件差归因为经验或恢复监督。

当前完整因子编辑的稠密差为B′A′−B_MT A_MT，其秩上界可达min(256,d_in,d_out)；MT合并基底上的rank8残差是另一参数族。这里的有益供给不能截断后当rank8标签，也不能证明rank8容量充分或原G过拟合。后继须fresh合法低rank教师/坐标及真实多parent/H，教学meta保留原source上下文；不得用冻结G上的π信用补尚未学会的parent修订。

## 计算、恢复、费用与资源退出

fresh至160来自clean pushed detached `18666c1032e601260cdd9e9f86949109e309b7f4`；原optimizer继续160→480及全部readout来自clean pushed detached `9a4fa85a4eff3963cd5e891914eeaa4c7978b2d9`。各精确command/env/输入/物理GPU/UUID/开始退出/配额在jobs/*/attempt_*/launch.json和exit.json。冻结源码没有原地修改；docs的新授权不改变已运行教师。CPU汇总时main为315cd411，最终交付记录身份见ROOT/scientific_completion.json与消息回执。

每事件160/480均保留lora.safetensors、state.pt、manifest.json，共16个完整checkpoint。8个160已由真实480消费者读取恢复，8个480完整state均CPU实际加载：76 optimizer状态/step480、sampler next_query53760、Python/NumPy/CPU/CUDA RNG、world1、schema和event/names完整。无scheduler/scaler；不存在G/π formal checkpoint。迁移物理GPU及microbatch14→28保留逻辑查询/任务权重与optimizer，不称bitwise exact。旧128状态与rank8族不兼容。

教师实际平均更新约10.01–20.44秒；续段实际10.01–18.91秒，详见每event的phase记录。采用共驻与退出事件动态接续，实测最大同时6卡，gpu01最多4、gpu02最多3。正式八教师及读出从UTC14:10:30到18:00:27，共3.832529小时；本数不包含此前工程/profile，也不代表未运行G/π全链时长。

| 计费部分 | GPUh |
| --- | ---: |
| disposable_profiles | 0.397374855 |
| collect_failed | 0.142494201 |
| collect_success | 0.232244853 |
| teacher_0_160 | 5.370153963 |
| teacher_160_480 | 10.287997461 |
| paired_readout | 2.210146771 |

总预算记账18.640412104 GPUh：精确计时部分18.602912104，三次未包围计时的短render诊断保守占用0.0375；后者精确实际费用未知，未包装成实测值。当前全部GPU jobs退出，未结算在途0；按登记口径距52还有33.359587896 GPUh，不构成新rank8规模的自动授权。44退出回执覆盖46尝试/9失败尝试；8失败回执为profile01/02、snapshot01、含3子尝试的untimed snapshot记录及四初次collect。所有正式教师160/480与readout均exit0，保留失败、原40/52修订及丢失H的事实。

两节点live快照gpu_release.json无本用户GPU进程，所有当前launch均有exit。strg01 data1最后独立配额使用1409972884KiB/quota2147483648KiB/limit2157969408KiB，共享可用88182809952256B；本批ROOT在读回前4,293,750,540B（约4.00GiB），新增分析原件约2.6MiB，收尾实占另写storage_exit.json。64GiB新增峰预算有充分余量，但没有连续峰值测量。历史data0只读；没有为了交付新开GPU探针、下载或复制policy。必要教师状态/真实H/标签/原行/失败及冻结代码保留；非U旧path生命周期规则保存在工程记录，当前没有G/π path可退休。

## 尚未完成范围与交接

本次完成的是当前授权的八教师科学决策单元。原计划的G160/320/480、学生刷新最多4教师、πlocal128/RL16、两个48行报告、指定完整U/P/selected及other/wrong均未运行；没有任意parent修订、实际video因果、held稳定或正式400结论。没有自动补这些范围、追加教师或恢复旧functional/T/PG。

结论是有限non-held教师供给有效、当前参数与函数有真实修改，同时较晚节点并非逐task/逐例更强。主讨论据全部正负原件自主定案下一参数族、多parent事件、尺度、实际规模及累计预算；这一步不需要Owner例行确认。实验session在可靠整批消息实际处理后串行交回tracked/Git，收到明确后继实施合同再接回。

## main独立消费与后继取舍

main直接读取run contract、panel、event中H/标签的来源记录、教师实现及全部480行，独立复算四task的128组audit三元结果，
并核对实际task/init、环境/政策seed、first-noise和首proprio配对，结果与上文相同。没有新GPU探针或训练。
128组只有64个不同task/init，不能用128独立Bernoulli样本的区间掩盖复用；此处不报伪精确的总体泛化置信度。

最重要的正证据是：同一个已经训练过这36任务的MT，其当前四任务表现并未穷尽可学空间。
T480新增68而丢1，比单纯参数/函数差更直接支持任务专用监督可以形成新初态控制收益。
但教师每条消费53,760个合法动作query，G只见V/L、parent与实际H；监督可用性远强于部署信息，
所以“教师会做”不推出“G可从合法条件预测其完整参数”。八教师全以MT为parent，也没有检验继续修订已有候选。
经验项没有隔离：四MT episode均失败、keep0，只有三条rec事件；task29没有rec仍能学习，
不能把总体增益归因于恢复标签、视频或经历。

相邻+9伴随22 gains/13 losses，task29还出现selection偏480但audit反向的例子。
这削弱以训练步数或小selection池作为稳定用途排序的充分性，却不支持抛弃全部后期节点：task38明显从5升至14。
后继保留160/480与原q_T，完整记录误判；不按已看的audit倒改旧标签，也不立即追加更大选择面板。
π仍需由真实候选的新初态结果学习用途，且有限H可能无法区分全部用途差，不能把所有选择损失都当优化不足。

主讨论选择MT合并基底上的fresh rank8、四类parent/H联合训练，而不改用弱source或继续原rank128 G。
MT的完整能力保存在基底，G只生成新改变量；坐标由10,297,344降至643,584，减少必须预测的数值，能否更容易学到协调输出仍待检验，
但旧完整差`B′A′−B_MT A_MT`不受rank8限制，故本次不直接投影旧端点，而用真实rank8教师检验这个容量假说。
历史`ac233fa0`中共享12能力43/250、残差4降至37/33及坐标/exact-BA修正未恢复专家控制，继续限制小rank/坐标能救能力的解释。
本次成熟MT基底、原生双meta及多parent标签改变了具体学习条件；并未消除因子误差的交叉项`B E_A+E_B A+E_B E_A`或部署分布差。

尺度分开承担职责：S_prox以固定A0和MT稠密更新定义教师邻近单位，S_G只由合法训练bank的完整编辑统计确定。
这样避免B0的零RMS造成不合理尺度，也让G输入能明确区分不同parent；归一化只改善数值条件，不增加视频中不存在的信息。
G用实际V/L、参数与经历通过分层通信输出完整因子，监督回到Gemma/action读取meta；
这些因子在MT基底的原生hidden上改变q/v及输入输出投影，经10步动作流形成前5动作，实际控制才是传递是否成立的判据。
π之后冻结这条生成链，其完整回报更新选择与实践，不能反向补上G没学会的修订。

正式后继在[同一active设计§13](../designs/video_guided_proposal_writer_20261010.md#13-当前正式实例rank8多起点完整链2026-10-11)登记：
4task×seed/mid/late/return共16教师先准备齐，G从第一次更新共同学习480步，再π-local128/RL16及规定报告。
每个mid/late独立执行自身parent的一条完整episode，return复用H160但保留产生者；
不串接两条可能各530步的长任务历史后谎称仍有1024预算继续修订。真实G、多producer长H分布仍是待检验差距，
本批不自动追加学生教师；若实际结果要求，main再形成具体有界后继，不把一次刷新当保证。

累计预算52→96GPUh，含原18.640412104，新增峰值仍64GiB；工程4–8h、主要计算14–24h是依实际费用和阶段依赖的估计。
本批目标是完整训练内控制传递和选择证据，video/held结论仍未成立；不把教师成绩写成EMBER成绩或自动扩到36任务。

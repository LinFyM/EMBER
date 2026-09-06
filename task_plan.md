# EMBER task plan

更新时间：2026-09-06。

## 当前目标与启动授权

2026-09-05 23:24 CST owner完成事前对齐，明确表示“没啥问题了”，要求设置goal并持续自主推进；goal已建立且active。目标仍是一次视频到唯一LoRA的zero-interaction能力，
validation8 strict paired correct严格>145/400，并满足相邻稳定、breadth/suite、same-task鲁棒性和最终视频因果资格。

在既有硬约束内自主实验、深入分析与有依据的实现，两个节点合计最多6张物理GPU；普通non-pass不能随意命名根因并最小修补。
重大科学/权限取舍与外部联系回到owner。需要owner决定时明确说明，同时继续可独立工作；真正不能推进时按goal工具合同标记blocked。

## Active design与当前阶段

- Active design：`docs/joint_process_policy_writer_design.md`。
- 专家原文：`docs/expert_review_20260905_full_history_joint_process_policy_writer.md`，按owner附件原文保存。
- 当前阶段：同图、query覆盖、非对称A、视频池2→4和共同P/Q主干比较全部完成；四视频与P/Q均没有闭环增量，结果sealed。
- 已恢复较好的两fit A2主干；73-task全task等权、64曝光formal从4d57f816 frozen worktree完成，实际采样/信息墙检查通过。
  32/48/64的两组正确视频strict400全部完成：80/78、81/81、79/75，未通过资格，完整证据已sealed。
- 同拓扑fully-random候选已按owner原有要求预注册并于10:09 CST从7f6b1611启动六卡训练；只跳过组件继承。
  不继续P/Q的rank/scale/seed小扫或无依据续训。
- 旧Unified v4及其m25/m50 `45/40`结果保持sealed，不续跑旧实验。

## 推进顺序

1. [x] 保存专家原文，完成对齐；依据owner最新明确要求建立goal并登记active design。
2. [x] 复用shared runtime，锁定少任务名单、训练/held视频、action rows、per-task exposure、局部闭环及source/carrier参照。
3. [x] 完成必要配置/最小实现、比例相称的验证与真实吞吐检查，启动同图clone/shared对照。
4. [x] 联合分析fit、新视频、实际闭环与任务更新；首轮显示小共享行为增量和有限query覆盖。
   同图fresh Panel-A query对照也已完成：fit41→41、held39→45，Goal/Long跨视频仍弱。先用只读实际correct-video诊断核对双侧零因子假设，
   不提前扩task、继续64步或同时替换读出与主干。实际18-forward诊断已完成，下一项只改变A侧context正负query职责。
5. [x] matched检验非对称A职责：64 fit/held44/45相对同执行参照41/44，暂保留A2并退役双臂runtime；不宣称稳定或已识别根因。
6. [x] 完成正确视频池2→4对照：41/38与41/39，没有行为增量；停止扩大同task视频数和无依据续训。
7. [x] matched比较共同P/Q与两fit A2：四组闭环均更弱，封存本实例并恢复A2；参数、初始化和约2.30倍训练加速单列。
8. [x] A2的55meta+18target全task等权group、64曝光及32/48/64两正确视频validation8 paired400全部完成，source/carrier参照也已收齐。
   80/78、81/81、79/75相对carrier72只有有限净增，数值资格、breadth/Long及跨视频稳定性未通过；历史§171、findings§170和formal报告已sealed。
   不宣称任务数量是根因，不直接续训或恢复P/Q；当前仅声称K1，继续已约定初始化比较，最终冻结后才补controls。
9. [ ] 完成同拓扑整套Writer fully-random fresh联合候选：相同73-task/64-update合同、32/48/64两组正确视频strict400，
   source/carrier复用同执行原始rows；比较变量只为初始化。已从7f6b1611启动，等待64步/Panel-B及后续closed-loop；不把继承先验认作根因。

下文保留的是先前已完成的研究事实。其“当前/下一步/active”只属于历史时点，不恢复旧配置或未执行分支。

## 最新已裁决design

最新设计合同为：

- docs/unified_policy_native_factor_writer_design.md

最新配置为：

- configs/pi05_ecp_policy_response_writer_unified_factor_v4.json

二者均已sealed。Native-Temporal、v1--v3及更早配置同样只由对应Git authority和formal artifact复现，不是active入口。

专家原始依据为：

- docs/expert_review_20260902_full_history_policy_native_meta_writer.md
- docs/expert_review_20260902_policy_response_event_to_factor_writer_clarification.md

原Policy-Response Event-to-Factor Writer、PNBTT及此前Program--bank实现均已裁决，不是active fallback。

## 已确认的科学基线

- validation8 source为48/400，task-local rank16 oracle为250/400，输出合同本身有容量。
- held5 stable carrier为43/250。
- G1真实native X/Y、signed pooling与rank4 free-code正式通过：114/250、breadth5/5、Goal2、Long1。
- G2 boundary-anchored ordered Natural Program正式通过：full相对endpoints改善22.2047%，probe38/40、median active
  events 4、same-task/K1/K4通过。
- G3及后续多类Program--bank mapping长期没有学得稳定shared功能映射。
- Program-through-bank、bank-conditioned primal与PNBTT共同证明：旧summary/gate/anchor或whitening/transport函数类不能稳定同时
  保持correct capacity与bank specificity。
- full-rank16 PNBTT在task1与task93呈稳定相反行为，没有证明rank4是当前主要瓶颈。
- 以上负证据不淘汰Stage0原生观测、ordered events、真实native X/Y、signed pooling、rank4或整个ECP。

## 历史v4设计审查结论（sealed）

1. full 50-step horizon、真实native X/Y、signed pooling、rank4 mobile与唯一rank16继续保留；coarse及等价平滑永久不是active方案；
2. 旧`grounded relations -> HMM/events -> C/D -> normalization -> relation marginal -> factor gain`连续路径整体退休；
3. 新主图只含一种可复制Unified Policy-Native Factor attention/MLP block；显式X/Y factor latent在每一层以同一query和同一套
   policy-attention权重分别读取exact language、同frame image patches与完整PI0.5 response，三者各自softmax；side-matched native
   bank另作独立标准attention，四个读出直接相加，再沿真实frame time及rank/side交互；最终同一个signed-query head以frame-common
   context定位当前bank，以frame-relative innovation产生正负偏移，随后直接对raw X/Y做signed pooling；
4. 独立Process坐标、event bottleneck、Composer解释边界与causal process auxiliary均删除，整个Writer只接受correct cross-episode
   functional梯度；时序信息仍由完整PI0.5 response与统一block承担；
5. 若新接口失败，删除并替换责任模块，不在其前后堆叠summary、solve、recenter、whitening、transport、calibration或gate；
6. task-local控制、shared资格与正式closed-loop各自回答不同问题，不能用内部loss或人为阈值代替最终行为。

## 2026-09-05先前执行记录（sealed）

1. [x] role-equal/cursor联合修复完成：m50/m100 held5为`39/45`，breadth均仅3且Goal/Long为0，未形成决定性shared增益；
2. [x] 以Axial Writer整体替换旧Process/Composer/gain链，删除独立gain runtime与causal objective；
3. [x] 25项Writer/native定向测试及真实task72 full-50 forward/VJP/materialization smoke通过；
4. [x] task72 5-step task-local profile自然结束，三条视频均微弱高于carrier，约`2.4--2.6s/step`；
5. [x] 从clean pushed detached authority完成task72正式25/50-step Composer容量控制：两点三条视频平均均优于carrier，m25
   fit/held recovery约`.094/.086`，m50约`.077/.062`；证明direct readout可学但局部上限偏弱，且后半回落；
6. [x] 73-task whole-Writer 25/50-step formal完成；每步6 task但总训练仅300次task exposure，55个meta task各约2--3次、18个target
   各约8--9次。m25/m50 gradient fit/held benefit都约为`-1e-4`，true-task-held non-pass，未解封held5。它只淘汰该低暴露实例，
   不把结果误解释成完整规模shared裁决；
7. [x] 相同50次更新的task-local whole-Writer反事实显示task72三条fit/held约`+.0010--.0012`，task1约`+.0001--.0003`，task93
   约`-.0013--.0015`，说明同一图的学习能力强烈依赖任务。correct-only几何进一步定位：events跨task仍不同，但训练把B/完整residual
   推向共同分量；A本来就高度共享，最终差异主要由B决定；
8. [x] task93 Composer-only 25/50-step正控虽三条视频最终全部为正，m50 held benefit仅`+.000600`、functional recovery`.0455`，
   远弱于同task free primal约`.0132` benefit和历史frame-local Composer约`.28--.30` held recovery。结合历史实现，最早缺失接口是
   当前单一global dynamic query被广播到所有frame，丢掉event与native candidate的frame-local对应；
9. [x] 以一个更短的`FrameAlignedFactorBlock`整体替换旧RankBank职责：rank读取events，每个真实frame再按相对位置读取本视频events，
   frame-specific dynamic直接产生X/Y signed contrast；删除完整bank预读。没有新增loss、gain、solve、normalization或校准链。
   25项CPU合同测试及task93真实full-50 forward/VJP/76-tensor rank16 smoke通过；
10. [x] 从clean pushed detached authority完成task1/task93 25/50-step Composer正控。m50 fit/held recovery分别为
    `.0761/.0958`与`.0868/.0793`，两个任务、两相邻点的三条正确视频均高于carrier；frame-local修正使task93 held相对旧实现
    提高约74%，但绝对容量仍远低于free primal；
11. [x] correct-only输出几何反事实排除补丁式修正：统一contrast温度在task1/task93上的最优区间不一致，4倍已饱和并退化；将所有
    pooled A/B单位化会把低幅噪声强行放大，两任务均迅速转负。因此不恢复`normalize -> gain -> cap`链，也不设固定温度；
12. [x] `configs/pi05_ecp_policy_response_writer_frame_aligned_12task_v1.json`的100-step task-disjoint shared资格完整结束。
    m50/m100 gradient fit/held benefit仅为`+.000475/+.000383`与`+.000561/+.000477`，两个true-task-held两点均`0/2`
    全视频为正且聚合为负；不进入held5或续训。每步`3 meta + 3 target`只是本配置的成本/覆盖选择，不是owner或架构固定比例；
13. [x] non-pass后从正确视频做零优化几何定位：Process events保持明显task-specific，完整mobile update也未坍缩；但global Composer
    query跨task median cosine约`.996`，每层event read只有输入残差的`.28--.45%`。task74生成update与72/73过度同向并增加自身
    functional loss，而真实functional gradients要求弱相关乃至相反方向。最早缺口是当前bank没有参与逐帧非线性方向形成；
14. [x] 以唯一可复制`FrameBankFactorBlock`整体替换旧Frame-Aligned职责：每个frame-rank先读取同frame完整native bank，再读取
    ordered events并做rank attention/MLP；逐视频中心化保证静态零mobile，末端仍只做一次raw X/Y signed pooling和安全cap。
    不增加gain、normalization、solver、calibration或并行fallback。14项合同测试与task93真实full-50 smoke通过；严格等价的
    frame chunk从8增至128后，相同第一步由`8.02s`降至`3.95s`，峰值reserved仅约`28.55 GiB`；
15. [x] clean detached `471592f4`的task1/task93 25/50-step Composer-only正控完成。task1 m25/m50 fit/held recovery为
    `.0540/.0526`、`.0841/.0702`，task93为`.0446/.0425`、`.1378/.1360`，四点均三条正确视频聚合为正。task93改善但两任务仍只
    恢复free primal约`5--14%`，所以bank-local方向有效而局部容量不足；
16. [x] clean detached `07804433`完成50-step Frame-Bank whole-Writer shared资格。m25/m50 gradient全视频为正为`6/12`与`8/12`；
    fresh held task3为正、task77为负，两点均仅`1/2`。它有微弱seen-task学习但无稳定task-disjoint映射，不运行held5或续训；
17. [x] 冻结correct-only VJP与路径消融定位最早接口：整体梯度并非普遍冲突，冲突集中在event消费与signed-X head；frame-only/
    event-only又无法同时覆盖task1和task93。裁决整体替换`Temporal -> Event -> late bank fusion -> shared X/Y state`，不追加数学补丁；
18. [x] 实现并集成唯一Native-Temporal Axial运行面：删除旧Temporal/Event/FrameBank runtime，以显式X/Y side、同frame side-bank read、
    ordered frame-time attention及rank/side attention组成同构block；25项定向检查通过，代码净缩短且无兼容fallback；
19. [x] clean detached真实task93 full forward/VJP/materialization smoke及2-step profile通过；task1/task93 25/50-step Composer正控
    也已完成。task1由m25微负转为m50 fit/held `+.0320/+.0169`且三条视频为正；task93两点稳定约`.122--.126`且三条视频均为正。
    该结果证明新接口可学，但不把task1弱恢复或内部恢复率变成额外性能门槛；
20. [x] 以同12 gradient tasks和预先固定fresh held task4/78完成50-step shared资格。m25的两个true-task-held均跨全部视频为正，
    m50却同时转负；gradient tasks由`7/12`升至`8/12`且平均fit/held继续改善。精确m25->m50路径积分显示held退化分散在
    Process、Frame、bank tokenization、NativeTemporal MLP和signed heads，不能归结为单个末端head或靠补丁修复；
21. [x] 冻结预注册m25 single checkpoint，以held5固定correct K1各调用一次Writer并完成strict250。结果`39/250`，Long/Goal/
    Object/Spatial0/Spatial9为`0/0/3/35/1`、breadth`3/5`；相对carrier43为`37 retained / 2 gained / 6 lost`。早期
    task-disjoint functional正信号没有迁移成闭环优势；
22. [x] 保持完全相同的两种可复制block、full-50、correct-only目标与width/rank，从fresh把gradient task覆盖扩到73个，并完成
    optimizer100/200短裁决。m100/m200见过的12个Panel-B任务fit/held benefit由`+.000277/+.000297`升至
    `+.000435/+.000393`，全视频为正由`8/12`升至`9/12`；但fresh held task6/79两点均为`0/2`，held benefit均值由
    `-.000382`恶化到`-.000680`。因此任务多样性不是主要缺口，不再以更多步数、LR/seed或数学校准挽救该函数类；
23. [x] 预注册m100/m200 held5 correct-only strict250分别为`42/250`与`35/250`，breadth均`3/5`且Goal/Long为0；整模块
    checkpoint swap进一步显示Process只在task1产生正收益、在task6/79/93产生主要伤害，Composer单独在四task都没有学得稳定正映射，
    二者joint interaction又按task放大正负。停止当前Process--Composer learned-coordinate handoff，不以冻结或数学补丁掩盖；
24. [x] 以一个可复制Unified Policy-Native Factor block整体替换该handoff：显式frame x target x rank x X/Y token在每层直接读取
    frozen prefix、完整PI0.5 response与same-frame side-matched native bank，再做标准time及rank/side axial attention；末端只保留一次
    centering、raw signed pooling、target cap和唯一rank16，删除独立Process/Composer中间坐标。19项定向CPU回归及最长task93真实full-50
    forward/gradient/materialization smoke已通过；
25. [x] combined-softmax首版task1/task93 optimizer25/50控制完成。task93在m25/m50的fit/held recovery为
    `.0249/.0264`与`.0973/.0687`且三视频均为正；task1四点为`-.0070/-.0023`与`-.0040/+.0052`且没有全视频通过。
    attention-mass诊断显示X bank只约占`8--14%`，而Y侧Q/action-in bank因token数可占约`70--96%`；首版把语义来源放进同一softmax，
    概率质量被实现定义的cardinality支配，配置已sealed；
26. [x] 在同一可复制block内完成parallel policy/native reads：同一query分别对policy evidence和side-native bank做独立softmax，读出直接
    相加；不加gate、手工权重或token-count校正。15项CPU合同和task93真实full smoke通过，policy/native read均收到非零functional梯度；
27. [x] 从clean pushed detached authority完成matched task1/task93 optimizer25/50。两个任务、两个相邻checkpoint的三条正确视频全部高于
    carrier；m50 fit/held recovery分别为task1 `.0592/.0648`、task93 `.1619/.1535`，相对combined-softmax与前代接口均有改善；
28. [x] 严格等价chunk64把最长视频峰值从`42.01`降至`36.22 GiB`；随后55 meta + 18 target的component-init
    optimizer100/200 shared短资格与两点held5 correct-only strict250均完整结束。m100/m200 held5只有`40/38`、breadth均`3/5`、
    Goal/Long为0，低于carrier43；gradient任务虽由`5/12`升至`8/12`全视频为正，fresh held task6/79始终`0/2`且恶化，故该版
    parallel-read shared函数类正式non-pass；
29. [x] correct-only物化几何与代码路径联合定位：nominal rank4 mobile的有效参与rank约`1.01--1.02`并随训练下降，m100到m200主要沿
    相似方向放大；旧readout在frame centering后只让innovation直接决定最终native查询，丢掉了共同context中的language、owner、family
    与rank定位。该缺口与专家明确的`base(context,current-bank)+dynamic innovation`合同一致，不是追加校准的理由；
30. [x] 在同一个最终signed-query head中实现common-base修订：每视频保留`C=mean_t z_t`与`D_t=z_t-C`，令
    `q+/-=b(C)+delta+/-(D_t)`；共同base在两分支完全相同，因此`D=0`仍严格零mobile。16项CPU合同及最长task93真实full
    forward/gradient/materialization smoke通过；不新增阶段、gate、normalization、温度或辅助loss；
31. [x] 从fresh完成task1/task93 optimizer25/50正控；两任务两相邻点的fit与未反传same-task held视频均自发高于carrier，且相对
    innovation-only v2全面改善。task1 m25/m50 fit/held recovery为`.0258/.0357`与`.0790/.0813`，task93为
    `.1568/.1371`与`.2066/.1848`；
32. [x] common-base v3的73-task optimizer100/200及两个single-checkpoint held5 correct-only strict250均完整结束；m100/m200为
    `35/250`与`31/250`，逐task为`0/0/3/29/3`与`0/0/4/22/5`，breadth均`3/5`、Goal/Long均0，稳定低于carrier43且随训练退化；
33. [x] 零优化职责诊断排除rank坍缩与evidence projection：成功task-local解同样接近rank1；m200 learned evidence在true-held
    task6/79均给出正增量，反而重复factor blocks在seen task1为正、在task6/79为负。block子层替换没有找到单一坏算子，说明整个
    shared block学成seen-task expert；进一步只保留trained evidence、恢复initial factor Writer的held5 strict250为`39/250`，相对
    carrier43保留/新增/丢失`36/3/7`，相对完整m200的31为`19/20/12`。它确认block破坏性但仍未超过carrier或恢复Goal/Long；
34. [x] 信息流审计定位task grounding稀释：15--24个language token与256 patch、400 response竞争同一policy softmax，language多数层
    仅约2.2%质量。当时的v4只把language、patch、response改为同权重、独立softmax的标准parallel reads；没有新参数、阶段、gate、
    calibrator或数学链，CPU合同与schema互斥检查已通过；
35. [x] clean `da964fad`完成task93真实full forward/VJP/materialization smoke；79 frames、2 probes、50 horizons、38 targets全部消费，
    patch/language/response/native与统一block梯度均非零，生成唯一76-tensor rank16，峰值reserved `36.91 GiB`；
36. [x] 从clean pushed authority完成task1/task93 optimizer25/50正控；task93两点稳定通过且m50 fit/held为`.2187/.2188`，task1由m25
    近零到m50 `.0431/.0490`并三视频为正。证明图有容量但task1弱于v3，不把该内部差异设成人为停止门；
37. [x] 从clean pushed authority完成73-task optimizer25/50短资格；gradient全视频正向由`5/12`升到`7/12`，task6/79在两个相邻
    checkpoint均为`2/2`全视频正向，首次形成稳定true-held shared信号，但绝对增量仍小；
38. [x] 并发物化m25/m50并各完成held5 correct-only strict250。m25/m50为`45/40`，逐task
    Long/Goal/Object/Spatial0/Spatial9=`0/0/4/37/4`与`0/0/3/34/3`；相对carrier43分别为
    `41 retained / 4 gained / 2 lost`与`37 / 3 / 6`，m25到m50为`38 / 2 / 7`。两点breadth均`3/5`，没有Goal/Long，
    因而当前v4短资格non-pass；
39. [ ] 未解封：只有继任者在新owner授权下提出新的、可复核的shared闭环证据后，才可进入扩展训练、mixed-K、同拓扑
    fully-random Final和validation8 strict paired400；当前结果不允许直接续训v4；
40. [ ] 未解封：只有correct-only选定并冻结single checkpoint后才运行negative/causal controls；
41. [x] owner要求停止推进、全面清理并交接；取消active goal/design/run，封存v4结果，清除临时目录、Python/pytest cache、已完成
    detached worktree与61,030,186,361-byte可重建退役G3 condition cache，并建立消费后删除的临时`HANDOFF.md`。

## 历史执行账本

1. [x] 逐字归档并核验完整历史专家意见与补充澄清；
2. [x] owner确认Policy-Response Event-to-Factor Writer主案；
3. [x] 建立持续推进goal；
4. [x] 登记active design并校正owner requirements、concept、findings、task plan与progress；
5. [x] 从最新clean pushed main建立唯一codex/policy-response-writer分支和worktree；
6. [x] 盘点并复用observer、events、native X/Y、chunked replay、materializer、J2 data/functional runtime与evaluator，确定唯一
   runtime owner及旧路径退休触发；
7. [x] 实现Frozen Capture接口、Frame Policy-Response Blocks、Ordered Event Blocks、Current-Video Native Factor Composer与
   positive-only objectives；
8. [x] 完成最小真实forward/gradient/materialization smoke及task1/task93两步task-local工程profile；
9. [x] 从clean pushed detached authority正式运行task1/task93 task-local Composer正控并评估step70/110；两task、两checkpoint的
   fit与held-video均自发优于carrier，确认Composer保留可泛化current-bank功能容量；
10. [x] live检查gpu01/gpu02、storage quota与真实吞吐，立即运行component-init 12-task K1 full/coarse matched实验；
11. [x] 已对12-task full/coarse的step70/110全部运行held5 correct-only strict250；coarse为`43/41`，full为`33/31`，
    四个checkpoint均未超过carrier，且Goal/Long全部为0；
12. [ ] correct-only选定并冻结checkpoint后一次性运行same-task-other、wrong、no-video、language-only、first+final、shuffled与
    reversed controls；controls不回流训练；
13. [x] 首轮定位到10-task shared映射的task-disjoint方向与组合覆盖不足；当时曾以coarse、K1扩大到55 meta + 18 target，
    但后续证据表明该参数化仍不能恢复方向或闭环性能，且owner现已明确禁止以horizon mean规避full问题；
14. [x] 原coarse shared信号后的扩展节点未执行，并已由owner的full-only边界及下面的新full执行节点取代；
15. [x] 完成55 meta + 18 target的旧scale component-init训练。macro610 held5 correct250为`26/250`、breadth`2/5`，相对carrier
    retained/gained/lost=`22/4/21`，为显著non-pass；只读policy-effect对照又显示Writer仅`1/5` task优于carrier、四个非零G1
    task的功能方向cosine中位仅`.14753`。完整`s_ref`事后限幅为`33/250`、breadth`1/5`，只局部恢复Spatial0，确认缩放不能补出
    缺失方向。macro1210为`30/250`、breadth`3/5`、Goal/Long仍为0；相对macro610高churn净增4但相对carrier仍净丢13，完成对其
    实际未限幅/global-clip/static-slot parameterization的相邻裁决；
16. [x] 完成per-target `B@A` RMS cap、独立gradient clipping与dynamic-value合同实现和真实smoke；原corrected coarse shared/task-local
    在owner更新方法边界后分别于step121/47/29主动中止，不形成checkpoint裁决，也不再续跑；
17. [x] 只用full裁决corrected shared。task1/task93 task-local已完成且保持跨视频容量；首个shared在审计发现
    Composer辅助bank context仍提前平均horizon后于macro前主动停止。现已改为对完整frame/probe/horizon/bank-type
    tokens做exact chunked learned attention并通过最长视频真实profile。owner于2026-09-03要求架构未证明前不付出10小时扩展成本，
    因此73-task x 1210-step fresh运行在optimizer25/effective15主动停止，不形成科学裁决。先在同一clean pushed commit上
    运行12-task x 110-step四卡资格实验并完成macro70/110 held5 correct-only strict250；两点均为`35/250`，breadth分别`3/5`与
    `2/5`，Goal/Long均为0，稳定低于carrier `43/250`，所以不恢复旧参数化73-task长跑、不运行其负controls；
18. [x] 修正Process到Composer最早失效接口：首个relation-summed scoring已完成12-task相邻裁决，held5 macro70/110为
    `42/34`、breadth均`3/5`且Goal/Long为0，两个true-task-held继续为负，因此该“先跨event求和、后非线性score”的具体接口
    non-pass。当前只把soft `alpha(e,t,m)`改为event x relation候选的base measure，以未求和`D(e,j)`产生动态logit；full
    50-horizon、真实X/Y、positive-only loss、rank12+4和唯一rank16不变。显式枚举等价测试与最长task93 formal-rows16真实profile
    已通过，单步`8.93/8.20s`且峰值reserved `46.43GB`；clean pushed detached短资格的110步及两点Panel-B已完成。
    gradient task的m110 fit/held recovery为`.1365/.1319`，但true-task-held仍`0/2`且为负。held5 macro70/110为
    `40/42`，breadth均`3/5`、Goal/Long均为0，m110相对carrier43为`7 gained/8 lost`。该matched接口稳定non-pass；
    70/110是10步warmup后effective 60/100的历史可比节点，不是不可调整的理论步数；
19. [ ] 吞吐线继续并行推进：第一阶段已完成选择性CPU evidence cache复制、outcome-independent动态task放置、
    dense/streaming exact bank attention、整视频signed pooling与output-group归约，四卡10-step由`34.39s/step`降至`4.05s/step`
    （`8.48x`）；relation scorer等价收缩又使task93快约`36%`。microbatch4、CPU activation offload和gradient packing经实测
    收益不足或明显变慢而淘汰；Evaluator当前保留实测更快的`3 replicas x 8 envs`。选择性8GiB cache复制已把当前event-measure
    完整110步均值为`10.06s`且四卡稳定段平均SM约`88--90%`；`4 * functional_rows + sampled_frames`新cost模型在旧110步
    真实回放预计再省`2.30%`，已通过22项测试、集成并推送，且不改变当前frozen run的逐步分配。
    73-task profile又确认两步均为每卡3 tasks；Panel-B后续已从training-cache ownership解耦，按完整视频与functional工作量把
    本轮预期`2/4/5/1`任务布局改为`3/3/3/3`。microbatch2/8实测只差约1%，保留2。node-local单份safetensors mmap又使
    每个local rank无需物理复制即可读取全部105.02GB frozen evidence；同commit、同两卡7-step schedule相对0GiB/8GiB
    private cache由`21.37/18.54s`降到`17.81s/step`，相对当前8GiB方案平均快`4.05%`、最坏step快`24.4%`，rank实际工作
    gap由`3.12s`降到`.34s`。当前四卡rows16的126步真实task timing反事实估计约可再省`21--23%`。继续以真实phase timing、
    峰值显存、rank idle tail与Evaluator LoRA/s定位剩余瓶颈，彻底优化可复用执行层，但不得等待性能工程完美才获取阶段科学结果；
    task batch与meta/target比例始终由实验配置决定。最新73-task rows2 world6两步为`6.36/6.70s`、每rank恰好2 tasks，较已有world4
    均值快约`34.2%`且峰值reserved仅`37.04GB`，故本轮空闲6卡确有吞吐收益；
20. [x] 已按专家的task-disjoint失败映射完成train24 + 审计non-held meta tasks的factorial coverage审计。73个gradient tasks中，
    7组同语言跨场景组合有5组包含至少两个gradient tasks、4组形成gradient-to-held桥；三类人工protocol contrast分别有
    `5/9/5`组train pair与`3/7/3`组held桥。task2、task74及held Spatial/Object/Long均有可见component重组依据；held Goal的
    `push` procedure没有任何Writer-gradient peer，是明确覆盖缺口。故数据并非完全欠识别，允许最小full扩展映射，但审计本身
    不证明video-dependent最优adapter已经可学；
21. [x] 已用当前单节点全部4张安全空闲卡完成73-task、每update 12 tasks的完整event-measure两步真实profile，并从clean pushed
    detached authority完成full K1 component-init扩展资格。55 meta + 18 target近似task等权，每步显式`9 meta + 3 target`
    只属于该配置；optimizer step200/400各task约`32--34/65--67`次暴露。m200/m400 Panel-B的gradient fit/held benefit虽由
    `.000740/.000316`升到`.001023/.000547`，两个true-task-held均值仍为负；held5 strict250仅`30/32`，breadth`2/5`与
    `3/5`，Goal/Long均为0。m200到m400为`20 retained/12 gained/10 lost`，相对carrier43的m400仍为
    `27/5/16`，故数据扩大和训练加倍没有解决shared闭环；
22. [x] 已把最早失效接口定位为Composer query seed的数值rank坍缩：raw相加时约`67`范数的Process common压过约`1`范数的
    rank token，m200/m400的实际mobile update均接近rank1。只对rank context与shared task context分别做无参数LayerNorm后相加，
    冻结m200反事实恢复rank query与部分有效factor谱；不新增loss、正交约束、solve或网络分支。修正已通过定向测试、真实
    forward/gradient/materialization smoke并合并推送。当前从clean detached `3e589695`运行73-task fresh m200/m400 shared资格；
    task1/task93的两次启动在任何optimizer step前依次暴露缺少正控声明及旧单panel合同假设，均不构成科学结果。`89ca865d`已让
    task-local v2合同直接封存resolved task panel并通过27项Writer测试和配置全字段预检。两条正控已从clean detached
    `ef00f446`完成；macro70/110的task1 fit/held为`.2224/.1153`、`.3283/.2282`，task93为`.3047/.3115`、
    `.3570/.3223`，四点全部视频优于carrier并满足信息墙，确认修正后的task-local容量。全线继续使用full 50-horizon、
    positive-only、唯一rank16及node-local单份mmap。shared m200 held5 strict250已为`45/250`、逐task
    `0/0/4/38/3`、breadth`3/5`；相对旧m200的`22 gained/7 lost`具有显著改善（`p=.00813`），但相对carrier43仅
    `12 gained/10 lost`（`p=.83181`），Goal/Long仍为0。Object18/Goal25/Long36的correct-only几何已确认q/v的rank
    区分恢复，而action-out仍约rank1、scale比q小约30倍且不是cap所致。六个gradient-authorized正确任务的冻结VJP进一步显示
    action-out并不缺policy梯度，但共享scale head存在强跨family冲突及跨task相消。m400 strict250已降到`35/250`，相对m200
    `5 gained/15 lost`、`p=.04139`；10个gradient tasks的held functional略升，两个true-task-held却从`1/2`降到`0/2`且均值
    恶化到`-.002605`，正式排除续训解释。m400跨Object/Goal/Long matching query cosine仍约`.9993--.9995`；common淹没
    language、family淹没owner的第二层typed-source坍缩已由冻结反事实复现。下一fresh只修Composer边界所有权：分别pre-norm并
    方差平衡rank/owner/family/common/language，relative rank-gain head按family分行，保持完整bank、loss、rank、cap、数据和
    positive-only合同不变；实现、30项测试、config互斥预检及真实task1 forward/gradient/76-tensor rank16 materialization smoke
    已通过。task1/task93的50/100正控自然完成，四个checkpoint的三条fit/held视频全部优于carrier；73-task shared 100/200
    短资格也从clean `682f7ecf`完整结束。m100/m200 held5 strict250为`39/32`，后者相对carrier43为
    `28 retained/4 gained/15 lost`、`p=.01921`，Goal/Long仍为0。与此同时gradient tasks held Panel-B由
    `-.0000465`改善到`+.0002095`，两个true-task-held却始终为负并从`1/2`全视频通过降为`0/2`，正式判定该
    typed-boundary参数化non-pass。family-local readout已经消除跨family梯度争抢，但跨task query仍约`.9984` cosine；后续
    归因确认Process的`D`本身保留task-specific动态，而当前predictor的`C+D_last`及Composer的`[E=C+D,D]`和raw `D`
    scorer在消费边界重复让约百倍更大的`C`淹没`D`。causal predictor在shared、gradient-authorized乃至task-local强正控上均不如
    零预测，故失败不是训练不足，也不是task-local容量不足；
23. [x] 只修common--innovation typed consumer boundary：以无affine、带连续可靠度的`LN0`分别规范`C`与`D`，causal state使用
    `(LN0(C)+LN0(D_last))/sqrt(2)`，event memory使用信息等价但不冗余的`[LN0(C),LN0(D_e)]`，signed dynamic scorer读取
    `LN0(D_e)`；没有新增模块、loss、teacher、task table、scale规则或coarse路径。30项Writer测试、真实full forward/gradient/
    materialization smoke通过，clean `f33f2955`的73-task optimizer50/100 fresh资格与两点held5 correct-only strict250完整结束。
    m50/m100仅为`40/35`，逐task分别`0/0/2/38/0`与`0/0/5/29/1`，breadth`2/5`与`3/5`，Goal/Long均为0；两点间
    `28 retained/7 gained/12 lost`。gradient fit/held functional虽继续改善，两个true-task-held却由负变得更负，故该
    consumer-boundary参数化正式non-pass，不追加训练或negative controls；
24. [x] 修正首版实现偏离专家原合同的causal interval：此前错误固定`future_offset=1`，而专家要求随机选择prefix `t`与合法
    `delta>0`。冻结m100的六个gradient task多尺度诊断显示，完整38-owner、50-horizon、2-probe target的within-video最优尺度
    MSE解释量从delta1的`.0094`单调升至delta2/4/8的`.1382/.3268/.4718`，跨task双向均值也由约`.0093`升到约`.0752`；
    所以adjacent高频目标是已证实的最早监督错配。下一fresh只改为outcome-independent的随机合法`(t,delta)`、parameter-free
    interval encoding及可逆的`sqrt(delta)`方差标准化，避免长interval仅因target方差增长而取得更大loss权重；保持固定teacher、
    Process/Composer主图、完整bank、rank与数据不变。实现已在`38d51bab`完成，31项定向测试及task1真实full-horizon delta8 smoke
    均通过。该formal m50/m100 held5均为`41/250`、breadth`3/5`且Goal/Long为0，故不运行negative controls；
25. [x] 修正random-delta实际没有学成的优化根因：m100预测相对zero仅改善约`.35%`，但冻结状态只训练同一head可在100/250步对
    未见同task视频达到`.144/.217` MSE解释量，delta4/8状态probe也保留`.278/.410`同视频信息。`df1e8c6e`只让head直接输出
    标准化delta、把两随机pair normalizer改为每fit视频8个target-only pair，并按累计步长实测给纯辅助readout `20x`学习率；
    Frame/Event/Composer、teacher、loss权重、full bank和数据不变。34项测试、task1真实smoke及两步shared profile通过；clean
    `f20a5299`的73-task optimizer50/100 formal与两点闭环完整结束。m50/m100均仅`37/250`、Goal/Long为0；但m100多pair
    process prediction相对zero改善`6.49%`，zero-state反而更差，说明辅助职责已学成却没有到达deployment Composer；
26. [x] 修正并裁决causal event坐标错位：完整视频继续使用G2 hard first/final anchors；人工截取的严格causal prefix只使用
    首锚定monotone forward filter，不再把每个cutoff强制映射为slot7。当前m100 hard-prefix与full同帧assignment重合仅`.137`，
    filter反事实为`.692`；同时target Process梯度为functional的`3.73x`且Event方向cosine `-.204`。`f6b58aac`已只改posterior推断，
    41项定向/物化测试和task1真实51-frame、full-50-horizon forward/gradient/唯一rank16 smoke通过；所有loss、teacher、prediction优化、
    Composer、完整native bank、rank、数据与task权重保持不变。clean `db354581`的optimizer50/100与两点闭环完整结束；held5为
    `38/36`、breadth均`3/5`且Goal/Long为0，gradient Panel-B略正但两个true-task-held明显为负，故该matched实例non-pass；
27. [x] 修正并裁决Composer gain/credit边界：当前实现遗漏专家明确要求的ragged native-group gain，并以严格零gain造成首步方向信用饥饿。
    m100四任务free rank gain的fit/held恢复约`.122--.166/.093--.147`，group gain提高到`.146--.244/.126--.202`；m100方向又比
    component-init平均`.213/.179`对`.178/.135`略好，说明不应推倒Process。该修正只恢复195个target-native group
    rows，并复用G1的`0.1`初始logit；Process、loss、LR、task比例、full bank、rank和数据不变。47项定向测试及task1真实full smoke
    已通过；首个functional backward的Frame/Event/Composer-direction/group-gain梯度为`.0569/.0531/.0990/.2214`，76 tensors与唯一
    rank16完整。两步shared optimizer profile也以`3.69/3.50s`通过，非gain Composer方向梯度约`1.88/1.51`；实现已由
    `aebd9d74`集成推送，world6吞吐profile通过。clean `a0797488`的73-task optimizer50/100及两点held5 correct-only
    strict250已完整结束：`37/35`，breadth均`3/5`，Goal/Long均为0，且低于carrier `43`。gradient task持续改善而task74随训练
    恶化；Process innovation并未坍缩，但实际group scale跨task cosine达`.99941`，action-out更新仅约G1成功解的`.016--.020`，
    task74的q-only有限幅效应为负而v/action-out为正。共享Process上process与functional总梯度cosine为`-.114`、Event为`-.291`，
    前者范数约为后者`1.49x`；该实例不是没训练够或断图，正式non-pass；
28. [x] 执行冻结Process的Composer-functional阶段：从同一component initialization fresh开始，保持完全相同deployment Writer、
    full 50-horizon、真实native X/Y、ragged group gain、rank12+4、73-task数据与task权重，只冻结Process并仅以correct cross-episode
    functional加preservation更新Composer。首轮仍用可续跑optimizer50/100和两点held5 correct-only strict250，直接区分联合
    Process辅助/漂移干扰与Composer共享函数类不足；不引入新网络、task table、coarse、negative loss或第二Writer。窄实现、53项测试及
    task1两步真实full shared profile已通过；正式optimizer50/100及两点闭环也已完成，held5为`39/43`，breadth均`3/5`且Goal/Long
    为0。m100仅与carrier持平；seen functional继续改善而true-held task74为负，故Process干扰不是充分解释，该实例non-pass；
29. [x] 修正并裁决Composer共享dynamic gain/readout函数类：exact group-logit VJP显示task2/task74真正所需下降方向cosine为`-.585`，但
    query-only raw gains cosine为`.9991`；同时最终mobile directions已有task-specific内容，task74仅调rank/group gain存在约
    `.00193`局部下降空间。下一fresh把195个target-owned query-only rows替换为共享factor-conditioned group tokens，每个token
    读取当前query、signed X、signed Y-group与group identity，经同一个可复制GatedMLP和scalar output预测bounded gain。保持冻结
    Process、full-50、bank/direction、rank/cap、数据/权重、positive-only loss和optimizer50/100不变；当前56项CPU合同测试通过，
    task1真实两步GPU smoke也已exit0：step1只有scalar output接收gain信用且direction非零，step2 conditioner梯度有限非零；
    Process全零、唯一rank16与full-50合同完整。clean detached formal m50/m100 held5为`40/44`，m100相对carrier仅净增1且Goal/Long
    仍为0；seen functional上升时true-task-held继续为负。task1/task93本地正控四点fit/held recovery约`.36--.53`且全部视频为正，
    故该pointwise共享参数化non-pass，问题不是local容量、跨视频保持或训练不足；
30. [x] 修正并裁决同target factor集合的相对协调：六task bridge诊断显示task74/task73所需logit变化cosine为`+.590`，pointwise
    readout的全参数梯度却为`-.418`，而actual gains仍约`.994--.998`同向。当前只把独立GatedMLP换成同target
    `rank x ragged-group` self-attention + GatedMLP标准重复块，不跨target，不改loss、初始化、Process、direction、data或训练规模；
    61项测试和真实full-50两步smoke均通过；clean detached `3686baec`的matched m50/m100 held5为`37/44`，m100与pointwise同为
    `44/250`且Goal/Long仍为0，10-task held-video功能诊断又只有3/10明确为正，故该参数化non-pass；
31. [x] 裁决functional-to-closed-loop接口本身：legacy与valid-action-only功能benefit在六task上符号一致，padding不是shared失败根因；
    authority72 task-local m50/m100在同一50初态由carrier `34/50`提高至`35/50`与`40/50`，m100为`30 retained / 10 gained /
    4 lost`且达到task expert `38/50`量级。functional信用可转化成真实闭环变化，当前主因收窄为shared task-disjoint mapping；
32. [x] 完成当前`6 meta + 6 target`联合采样修复并裁决m50/m100：它同时把target角色质量从25%提高到50%，并把旧global-step
    周期别名造成的每task 1/2 fit video、8/16 Panel visits恢复为2/2、16/16。通用实现改用per-task occurrence cursor并已通过
    回归；最终m50/m100 held5为`39/45`、breadth均仅3且Goal/Long为0，两个true-task-held均为负，因此没有再扫比例；
33. [ ] full shared信号成立后进入mixed-K、fully-random fresh Final joint和validation8相邻single-checkpoint strict paired400；
34. [ ] selected checkpoint冻结后补齐最终因果controls；只有base Writer稳定且剩余错误集中在action detail时才评估Action Meta；
35. [ ] 达到最终合同，或在完整信息量证据下形成当前函数类乃至EMBER总体停止裁决。

## 推进与决策原则

- 效率优先：实现接通后立即获取真实GPU与closed-loop证据，文档、通用重构和冗余测试不能延迟科学结果。
- 缜密修正：每次架构改变必须对应已定位的最早接口，不以随手改结构、LR/seed/width/rank/scale小扫替代分析。
- 结构优雅：有证据时允许实质重构，但不把non-pass修成连续数学变换链；同一接口反复失败时优先整体替换责任模块。learned主干以少数
  职责清楚、可复制扩展的attention/MLP blocks为主，手工运算只承担不可避免的科学边界。
- 不作弊：只用正确视频训练；validation/test无梯度；负controls只在checkpoint冻结后评测。
- 一次non-pass只淘汰实际检验的组合，不因局部失败推翻全部正证据。
- 明确坏结果不无限续训；有新的机制证据时也不受人为版本次数限制。
- closed-loop absolute性能优先，内部loss、factor cosine、attention和representation指标只负责定位。
- Final必须包含component-init与fully-random同拓扑fresh候选。
- GPU launch前同时live检查gpu01/gpu02；有1至6张有效A40就使用，不等待凑卡，可安全共驻但不干扰他人。
- 长训练期间优先完成cache、分析、评测准备和下一科学节点；只有没有推进相关的实质性工作时才做可中断的增量workspace清理，
  训练或评测结果一到立即停止清理。

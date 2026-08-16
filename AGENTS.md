# EMBER Repository Instructions

## 1. Authority

当前authority按以下顺序解释：

1. owner最新明确表达；
2. `docs/current_owner_requirements.md`：owner目标、昨晚讨论形成的原则、方法/目标边界与协作要求；
3. 本文件：科学合同、信息墙、评测、GPU、存储、Git与工程边界；
4. `docs/active_session_handoff.md`：唯一当前实验状态、run identity和下一裁决点；
5. 当前active design（若已建立）；
6. `docs/research_history.md`与Git/artifacts：历史证据和负结果边界。

旧design、Git快照、formal artifact、日志和历史文档中的“当前/下一步/active/暂停”只表示当时时点，不得覆盖
上面三份当前authority，也不得直接恢复执行。owner主要使用语音输入；明显同音词、术语识别或断句错误应结合
EMBER上下文纠正理解。

## 2. Minimal mandatory reading

修改代码、配置、数据、split、模型或实验状态，或启动GPU工作前，主任务完整阅读：

1. `docs/current_owner_requirements.md`
2. `docs/active_session_handoff.md`
3. `docs/execution_brief.md`
4. 当前active design：
   `docs/action_forecast_writer_v6_lpcp_cfmg_unit_secant_endpoint_preference_design.md`
5. `task_plan.md`
6. `findings.md`
7. `docs/concept.md`
8. `docs/research_history.md`

涉及旧架构细节时先查`docs/research_history.md`；只有确需旧公式或实现时再从Git commit`3a6f801`选择性读取。
涉及A100到BCI迁移与路径恢复时才读`docs/a100_to_bci_migration_handoff.md`。不要把重复阅读数万行退役设计当成
科研推进前置步骤。

## 3. Current operation

长期目标尚未完成。当前active successor是**V6-LPCP CFMG Unit-Secant Endpoint Preference**（USEP），authority=
`docs/action_forecast_writer_v6_lpcp_cfmg_unit_secant_endpoint_preference_design.md`。它从sealed LPCP fresh开始，
完整保留CFMG memory/content grid、rank32、K4与global commitment，只把每个matched state的raw
`softplus(Dwinner-Dloser)`改为除以该winner/loser action RMS后的unit-secant preference。该pair-local公式不读
task ID或gradient norm，不改变task/view权重，也不做MSE反比放大。固定task4/34/38 world3将直接检验task38
支配能否从`58.73x`降至`<=15x`、raw shared能否3/3、task34能否4/4及native能否12/12；当前尚无USEP GPU结果。
canonical实现与fresh identity已原位完成，相关CPU=`143 passed`、完整CPU=`413 passed`、architecture guard无hard
violation，且没有增加forward、参数、模块或部署分支。

最新终局predecessor是**V6-LPCP Content-First Memory Grid**（CFMG）：clean`bb5341e` world5
full24 cycle1完整exit0，24 tasks/48 paired states/96 rollouts，candidate/reference=`32/32`、gains=`3/3`，6个
active tasks覆盖四suite，cycle=`467.783s`且0禁读/OOM/nonfinite。它保持validation8 held四view的
BA cosine/energy=`.982412/.985173`，但full24的11个global candidates最好仅`14/24` task-view margins下降，
最终恢复exact LPCP step0；parameter delta、q/v/action BA与fixed-action response全为0，故strict400按authority
跳过。相对CMBG，六task梯度只近似统一放大`1.78--1.92x`，cross-task cosine mean/min仍为
`.00923/-.20641`，task34 same-task cosine仍`-.0923`，task38相对次大梯度的支配由`54.45x`增至`58.73x`。
最早失败接口是content-conditioned random Program到幅度平衡、多task可共存的第一次public commitment；因gate
被拒绝，temporal/set/M2P从未获得下一步学习。CFMG终局，不cycle2、strict、controls或小扫；当前没有active
successor或GPU run。SFMC144仍是最高correct单点但lost15/churn31，不具稳定资格；v6-fast仍是有完整五臂的
历史最好：`143/135/125/128/129`。

当前最强zero-interaction carrier baseline是**V6 Layerwise Action-Probe Conditioned Procedure Reader**（V6-LPCP）：macro25 K4
strict=`143/400`、breadth7、per-task=`1/4/48/35/0/38/16/1`、per-suite=`5/83/38/17`。相对同schedule
AS139严格=`120 retained / 23 gained / 19 lost / 238 both-fail`、churn42、net`+4`、p=`.643969`；它
count-only追平不同teacher schedule的历史v6-fast143并把breadth从6增到7，但按`<144`和lost>10两项门终局
non-pass，不resume50、不补controls或扫memory/rank/LR/scale/seed。

该轮否决的不是layerwise视频carrier：同一次真实context forward的18层probe旁读满足one-forward，倒序使
query-delta/Program relative-L2=`2.0572/.40414`，constant query-delta近零；reader/controller也获得持续credit。
全400 effective-BA相对AS139只改`.002653` relative-L2、cosine`.99999479`、norm ratio`.99997391`，而first4
same-task correction coherence median`.56804`。Goal3虽有`.004224`改写和`.88373` coherence仍为0，Long1只改
`.001324`却净丢6。最早缺口是conditioned Procedure经冻结fusion/compiler承诺成AS139邻域小方向，以及blind
B20 functional credit不能选择held on-policy有用方向；只替换已通过的carrier为literal memory不直接针对该缺口。

封存authority=`docs/action_forecast_writer_v6_layerwise_probe_conditioned_procedure_design.md`。正式root、AS139严格
配对和effective-BA artifact均已保留。owner随后明确授权继续，并再次澄清memory token是候选机制而非必须形式。
V6-LPCP Paired Causal Success Distillation（PCSD）保留LPCP/AS139/rank16部署图，以同初态
reference/candidate两臂的唯一成功轨迹只校准65,536参数`query_delta`。精确authority=
`docs/action_forecast_writer_v6_lpcp_paired_causal_success_distillation_design.md`。canonical实现已经原位替换旧ADSP
runtime：同一K4 context只编码一次，K2两臂返回完整executed-prefix replay，ties为零，只训练
`query_delta.weight`；全量CPU=`387 passed`、architecture guard无hard violation。clean frozen `efc17be`在
gpu01物理`5/6/7`以world3完成full24 cycle1：24 tasks、48 pairs、96 rollouts，candidate/reference=
`34/33` successes、`5/4` gains、9 discordant/active tasks覆盖3 suites，gradient、parameter delta、BA/action
response均非零，0 forbidden read/OOM/nonfinite，wall=`837.694s`。

PCSD cycle1 K4 strict paired400已终局为`135/400`、breadth6、per-task=
`0/4/48/32/0/35/15/1`、per-suite=`4/80/35/16`。相对LPCP143严格=`121 retained / 14 gained /
22 lost / 243 both-fail`、churn36、net`-8`；相对AS139严格=`115/20/24/241`、churn44、net`-4`。
全400 PCSD相对LPCP effective-BA relative-L2 mean/median仅`.0006834/.0006767`，gained/lost幅度不可分；
FP64 first4显示同task四个不同K4 video sets的增量pairwise cosine平均`-.00187`、mean/sample energy ratio=
`.24860`，即几乎正交且平均后只剩约四分之一能量。reward credit有内容且LoRA→action链路工作，但稀疏
paired success经单一shared query commitment没有形成跨task/video可保留方向。PCSD按四项门终局non-pass，
不得cycle2、补controls或参数小扫。

最新完成successor是**V6-LPCP Cross-Video Causal Success Distillation**（CV-CSD），精确authority=
`docs/action_forecast_writer_v6_lpcp_cross_video_causal_success_distillation_design.md`。clean `c1d8952` full24 cycle1完整：
24 tasks/48 paired states/96 rollouts，33/34两臂成功、5/4单臂成功、9 active tasks；四view全部LoRA/query gradient
非零，wall=`863.432s`=`1.0307x` PCSD，3 ranks各8 tasks/3 active tasks、负载max/min=`1.0828x`。

CV-CSD K4 strict paired400终局=`134/400`、breadth7、per-task=`1/2/47/32/0/36/15/1`、per-suite=
`3/79/36/16`。相对LPCP143严格=`122 retained / 12 gained / 21 lost / 245 both-fail`、churn33、net`-9`，
四suite全降；相对AS139=`121/13/18/248`、相对PCSD135=`115/19/20/246`。correct、lost、net与suite四项门失败，
不续cycle2、controls或小扫。

全400 CV-CSD/LPCP effective-BA relative-L2 mean=`.00068370`，gained/lost约相同；FP64同task四K4 correct
conditions的增量pairwise cosine=`.000205`、mean/sample energy=`.250155`，相对PCSD也为`-.001908/.248578`。
所以四个正确视频下的exact成功credit经过shared `query_delta`均值后仍落成近正交局部方向。

最新完成架构是**V6-LPCP Semantic Factor-Memory Commitment**（SFMC），authority=
`docs/action_forecast_writer_v6_lpcp_semantic_factor_memory_commitment_design.md`，clean frozen commit=`8994180`。
full24 cycle1完成24 tasks/48 pairs/96 rollouts，8/8 family maps更新且wall=`920.555s`=`1.0662x` CV-CSD；
strict correct400=`144/400`、breadth7、per-task=`1/3/47/36/0/38/18/1`、per-suite=`4/83/38/19`。
相对LPCP143严格=`128 retained / 16 gained / 15 lost / 241 both-fail`、churn31、net`+1`、Jaccard
`.805031`，只有lost≤10门失败，故仍是终局non-pass，不续cycle2或六臂controls。

稳定FP64低秩差分证明训练时连续hidden residual没有有效写成部署LoRA：相对LPCP的all400 effective-BA
relative-L2 mean/median仅=`2.899e-7/1.066e-9`；249/400样本只有q-family跨过native factor量化边界，v仅16，
action仅1。first4同task修正pairwise cosine约`-8.10e-6`、mean/sample energy=`.249995`，仍是近正交且
稀疏的video-local ULP crossing；semantic query/basis-key参数delta也仅约`1.7e-9`，cycle1没有形成学到的
semantic route。最早失败接口是**SFMC continuous hidden residual -> frozen W2 -> native public LoRA**，不是
carrier未读视频、GPU负载或训练图未工作。

最新终局successor是**V6-LPCP Gradient-Open Semantic Commitment**，authority=
`docs/action_forecast_writer_v6_lpcp_gradient_open_semantic_commitment_design.md`。它不续SFMC checkpoint，只把
zero-init staged commitment改成step0严格等于LPCP、但family delta maps与semantic query首步同时有梯度的
V6-W1 anchored参数化；LPCP carrier、K4 four-view credit、rank16、optimizer与信息墙不变。task4真实smoke已
通过：semantic query delta=`1.1979e-4`，q/v/action native effective-BA response均非零，cycle wall为SFMC
`.9501x`。fresh world5 full24 cycle1已由clean detached `eb543d3`完整exit0：24 tasks/48 pairs/96 rollouts，10 active tasks
覆盖四suite，semantic query delta=`6.9499e-5`，5/5 probes的q/v与3/5的action native BA非零，cycle=
`581.924s`，完整world5 checkpoint/completion已保留。随后同checkpoint K4 strict=`141/400`、breadth7、
per-task=`1/3/48/29/0/36/23/1`；相对LPCP为`128/13/15`、churn28、Jaccard`.82051`，suite净变化=
`-1/-6/-2/+7`。稳定FP64显示BA改写较SFMC放大约`33.3x`且q/v/action几乎全样本非零，但同task four-view
增量cosine仍`.000144`、energy ratio`.250124`。因此梯度与native写出已打开，最早缺口后移到跨video共同
causal Program；不续cycle2或六臂，当前没有可resume的active checkpoint。
SFMC和Gradient-Open都没有六臂结果，因此不得宣称same-task-video鲁棒或视频特异性。该负结果只淘汰本轮`LPCP innovation memory
+ 4-way language route + zero-init family-hidden residual + one CV selected-success cycle`组合，不否定memory
token、rank8、few-shot或生成LoRA本身。

CCT authority=`docs/action_forecast_writer_v6_lpcp_causal_coefficient_transport_design.md`。formal full24 cycle1完整：
24 tasks/48 pairs/96 rollouts、candidate/reference=`33/32`、9 active tasks覆盖四suite，cycle=`577.729s`，完整
checkpoint/completion与0禁读/OOM/nonfinite均通过。strict结果如本节开头，故没有cycle2或六臂。

机制记录已纠正一个分析counterfactual错误：旧`.563803/.672852`实际混入AS139 reference，不是纯CCT增量；
同一task4 post-update state相对exact LPCP的正确CCT-only cosine/energy仍为`.575776/.681821`，所以train-seen
局部门仍成立。held first4则约为`0/.25`；live loader确认semantic query逐元素精确加载。train→held transported
coefficients、pre-W2 hidden residual与effective-BA L2分别缩小`1.63x/1.70x/249.92x`。最早失败接口是held
Program经native BF16 factor/compiler时未形成policy-effective commitment，而不是carrier、loader或reward链路。
最新完成的**V6-LPCP Native Probe-Value Commitment**（NPVC），authority=
`docs/action_forecast_writer_v6_lpcp_native_probe_value_commitment_design.md`。它保留LPCP143、rank16、native
FactorHeads、CCT language-policy axes与matched selected-success合同，唯一把factor Value从微小
`Procedure_LPCP-Procedure_AS139`差分换成已有320 slots的ordered native Action-probe delta，并用同一
Procedure-set attention做K轴聚合。trainable仍为67,072，不加memory token、rank变化、scale、normalization或
额外loss。canonical实现与fresh-incompatible config/checkpoint/eval schema已完成；定向CPU合同`43 passed`、
完整CPU在`.env.local`的LIBERO assets环境下`398 passed`。formal前必须先过train→held只读视频门；held若再现
约`0/.25`或BA `1/250`断裂，立即终局。该门现已通过：train task4 cosine/energy=`.5929/.6792`，validation8
平均=`.4494/.5715`、6/8 tasks过门，held/train BA L2=`.7525x`；reverse、constant与wall也过门。full24
cycle1随后完整完成，但strict仅136。post-train held8仍有`.40870/.54227`的跨视频共同写出，all400 BA改写
mean=`.0004683`，说明CCT的held compiler断裂已经解决；然而gained/lost改写不可分，持续失败样本改写最大，且
train task4共同方向在full24后从`.5929/.6792`漂到`.0569/.2951`。最早失败接口是reward-useful Value组件/符号
选择与多task共存，不是video carrier或LoRA量化。

最新终局的是**V6-LPCP Pre-Addressed Factor-Selective Native Value**（PAFS-NV），authority=
`docs/action_forecast_writer_v6_lpcp_preaddressed_factor_selective_native_value_design.md`。它保留LPCP、NPVC native
Value、rank16与matched reward，只把zero-init common router换成reward update前已存在的fixed language address，
并令八factor families各自用zero-init diagonal selectors选择native Value组件/符号；trainable=`16,384`。这直接
检验task/family在第一次full24 update前分流能否避免NPVC task4坍塌。真实task4机制健康，但train24 address
effective rank=`2.1575`，validation8 cosine/energy=`.1681/.3729`且仅3/8过门，故未启动full24或strict并终局。
随后终局的是**V6-LPCP Shared Joint Native-Value Gate**（SJNV-Gate），authority=
`docs/action_forecast_writer_v6_lpcp_shared_joint_native_value_gate_design.md`。clean `913d3d3` task4 smoke工程健康，
train cosine/energy=`.47227/.59781`；但validation8仅`.20190/.39645`、2/8过门，action cosine=`.04299`，故不
full24/strict。stage localization显示gate/continuous hidden cosine均约`.94`，经过冻结W2写成native raw factors后
骤降为cosine/energy=`.02135/.26592`，action factor cosine=`.00267`。最早断点因此是coherent hidden residual
到frozen W2/native public A/B，而不是carrier、顺序或joint gate未读视频。

最新终局的是**V6-LPCP Direct Joint Native-Factor Residual**（DJNFR），authority=
`docs/action_forecast_writer_v6_lpcp_direct_joint_native_factor_residual_design.md`。它已证明有序视频与语言joint Value可
绕过冻结W2直接稳定写成native public LoRA：post-full24 validation8 BA cosine/energy=`.790242/.785834`、8/8过门。
但cycle1 strict仅`136/400`、breadth7、per-task=`1/2/44/35/0/35/18/1`；相对LPCP143为`120 retained /
16 gained / 23 lost`，churn39、net`-7`。gained/lost/retained-failure BA改写依次=`4.522e-5/7.248e-5/
8.987e-5`，说明生成端已通但selected-success credit不选择held reward-useful方向；不续cycle2或controls。

最新终局successor是**V6-LPCP Direct-Factor Paired Common-State Preference**（DF-PCSP），authority=
`docs/action_forecast_writer_v6_lpcp_direct_factor_paired_common_state_preference_design.md`。它从sealed LPCP fresh，完整
保留DJNFR carrier、K4、rank16和八个direct factor heads，唯一把成功整轨迹正蒸馏改为candidate/reference discordant
pair在分叉前同一初始观测处的winner-vs-loser首段flow preference；四个disjoint correct K4 views等权。首次clean
`de6c812` task4 smoke在credit前证明相同seed顺序reset不能保证首观测逐元素一致；`07b764b`进一步证明flattened state
恢复后，hard reset仍改变未被state覆盖的model pose：差异只在两相机，language/state tokens完全相同。两次均为
工程exit1且无科学结果。canonical现每lane只做一次hard reset+settling，每臂以deterministic soft reset清空
controller/observables后恢复相同qpos/qvel，不增加rollout/forward。exact task4/task7均为tie；task9/15/18分别有
1/2/1个discordant pairs且margin均下降、八head/q-v-action均非零，但task9 held/train BA仅`.105x`，task18 train
跨video仅`.290/.428`，三个有效anchors只有task15全门通过。按门终局，不full24/strict/cycle2；当前无active
successor、GPU run或可resume checkpoint。最早失败接口是final success只归因给第一shared prefix后形成
task-dependent update。memory token与rank8仍开放。

最新终局successor是**V6-LPCP Direct-Factor Successful-Occupancy Counterfactual Preference**（DF-SOCP），
authority=`docs/action_forecast_writer_v6_lpcp_direct_factor_successful_occupancy_counterfactual_preference_design.md`。
它不改LPCP/DJNFR生成图、rank16、K4或rollout数；对每个exact discordant pair沿winner全部replan observations，
使用同一policy-noise批量查询loser arm的counterfactual actions，再逐状态做winner-vs-loser flow preference。
固定task9/15/18 outcomes与26/65/44 chunks均复现，三者train/held跨video、八head、q/v/action、reverse与
constant机制均健康；但task9/15的stored动态B2/B1 action到B8 requery差异分别是名义策略contrast的
`1.086x/1.693x`，positive/negative存在batch-shape混杂。三项wall又分别为matched DF-PCSP的
`3.083x/5.335x/3.887x`，瓶颈是完整轨迹四view functional credit而非counterfactual inference；task9
held/train BA仅`.1181x`。按预注册门终局，不full24/strict/cycle2或小扫。最早失败接口是
**stored动态batch winner action + B8 loser counterfactual -> 非matched-batch preference panel**；下一设计必须
同B8重查两臂并预声明时间分布的informative occupancy子集；该接口随后由MB-SOP实际检验。

最新终局successor是**V6-LPCP Direct-Factor Matched-Batch Stratified Occupancy Preference**（MB-SOP），authority=
`docs/action_forecast_writer_v6_lpcp_direct_factor_matched_batch_stratified_occupancy_preference_design.md`。它保留LPCP/DJNFR
carrier、K4、rank16、八direct heads、exact outcomes与四view，只替换successful-occupancy credit panel：在winner
完整occupancy上以相同B8 observation/noise顺序分别重查reference和candidate；每条成功轨迹等分8个进度strata，
每区只保留matched action RMS最大的一项。task9/15/18 functional pairs预定从`26/65/44`降为`8/16/8`；
trajectory、strata、views与tasks等权。clean `ad65347`三anchor复现全部outcomes/counts，wall相对DF-PCSP仅
`1.655/2.119/1.542x`，train四video BA cosine/energy=`.852/.854,.694/.694,.699/.723`，held8也约
`.77--.78/.78`。但task15/18同一panel margin在AdamW后分别增加`+.0003545/+.0004093`，task9 held/train仅
`.1096x`，故按门终局，不full24/strict/cycle2。

额外flat-gradient诊断显示三anchor四个view到raw等权均值的最小cosine=`.695/.629/.601`且下降覆盖均为`4/4`；
所以MB-SOP最早缺口是**已正确汇合的functional gradient -> coordinate-preconditioned finite AdamW delta**，不是
carrier、memory、rank、video聚合或LoRA写出。

最新终局successor是**V6-LPCP Direct-Factor Adam-Radius Euclidean Commitment**（AR-EC），authority=
`docs/action_forecast_writer_v6_lpcp_direct_factor_adam_radius_euclidean_commitment_design.md`，clean `b578d56`。它保留
Adam候选global L2但把final delta精确放回`-raw mean`。task9/15/18的raw coverage和final方向均为`4/4`/cosine1，
但post-margin每个任务都只有`1/4` views下降；完整delta分别为
`[-.000017,+.000007,+.000202,+.000061]`、`[+.000279,+.000018,-.000018,+.000003]`、
`[+.000417,-.000290,+.000213,+.000218]`。Adam radius是raw gradient L2的`6333/7988/4294x`。train/held BA
coherence、q/v/action、reverse/constant和core wall全部健康，且task15/18 coherence较MB-SOP更高；因此最早缺口
进一步定位为**全Adam半径超出四view共同局部下降区间**。AR-EC不full24/strict/cycle2或固定scale sweep。

最新终局successor是**V6-LPCP Direct-Factor All-View Monotone Backtracking Commitment**（AV-MBC），authority=
`docs/action_forecast_writer_v6_lpcp_direct_factor_all_view_monotone_backtracking_commitment_design.md`。它保留AR-EC全部
方向、optimizer state和科学图，只从Adam upper radius沿同一`-g`方向依次检验`1,1/2,...,1/1024`，接受同一panel/
noise下四个correct-video views全部严格下降的第一个candidate；不是挑最佳scale或多checkpoint sweep。首轮仅实现
固定三anchor机制门。修正版clean `202a64d`结果为：task18在`j=5`通过全部门；task9只在`j=10`接受但held4/8、
held/train`.18446x`；task15到`j=10`仍无共同candidate并恢复exact no-op。故scalar radius对task呈有效、
near-identity和空集三种状态，AV-MBC终局，不full24/strict/resume。

最新终局successor是**V6-LPCP Direct-Factor Maximum-Margin Common-Descent Commitment**（MMCD），authority=
`docs/action_forecast_writer_v6_lpcp_direct_factor_maximum_margin_common_descent_commitment_design.md`。它保留AV-MBC全部
carrier、matched reward、Adam upper radius、native backtracking、rank16与八heads，唯一从已有四view gradients的
`4x4` Gram确定性求maximum-margin direction，再保持原task mean norm与跨task等权。clean `fc3bdd7`固定
task9/15/18的continuous worst margin均提高`1.216/1.334/1.356x`，但native结果分别为j0大步且held/train仅
`.160558x`、到j10仍no-op、j6且全门通过；只有1/3 anchors通过。故first-order maximum margin不能稳定预测
native finite-step/held commitment，MMCD终局，不full24/strict/resume或小扫。当前没有active GPU run或可resume
checkpoint；下一变量必须针对native metric/finite-step接口，不能退回改已通过的carrier或继续换raw-gradient solver。

最新终局successor是**V6-LPCP Direct-Factor Preconditioned All-View Backtracking Commitment**（PAV-BC），authority=
`docs/action_forecast_writer_v6_lpcp_direct_factor_preconditioned_all_view_backtracking_commitment_design.md`。它保留MMCD前的
全部science graph与同路径acceptance，删除raw-gradient solver，唯一沿实际AdamW candidate delta依次检验
`1,1/2,...,1/1024`。clean `581140c`结果为task9 j5但held/train仅`.109466x`，task15/18到j10均无共同candidate
并exact no-op；0/3 anchors通过。raw equal-mean、raw maximum-margin与Adam-preconditioned三类parameter rays至此
均被固定anchors否决，PAV-BC终局，不full24/strict/resume或ray混合。当前没有active GPU run或可resume checkpoint；
下一变量必须转到LoRA输出/effective-BA参数化的native-safe线性Value路径，不能继续改gradient ray或trust scale。

最新终局successor是**V6-LPCP Anchored Linear-B Native Value Commitment**（ALB-NV），authority=
`docs/action_forecast_writer_v6_lpcp_anchored_linear_b_native_value_commitment_design.md`。LPCP143、MB-SOP四view credit、
rank16与PAV acceptance全部不变；唯一删除四个A residual heads，只训练四个zero-init B heads（860,160参数），使
`BA=B0A0+delta-B A0`对joint video Value严格线性且完整保留baseline项。选择B side来自LPCP correct400固定几何：
B侧单位增量的BA灵敏度相对A侧为q/v/action-in/action-out=`1.049/1.411/2.594/8.258x`，q/v native ULP不更差。
clean `0899166`固定task9/15/18结果仅1/3过门：task9的continuous four-view gradient为`.415/.559`且覆盖4/4，
但`j0--10`无native共同下降并exact no-op；task15从PAV no-op改善为j5，held BA=`.375/.513`、held/train=`.333x`，
但仅5/8 held tasks且raw-B=`.101/.323`；task18从PAV no-op改善为j0并全门通过，held BA=`.774/.785`、8/8、
held/train=`1.030x`。所以fixed-A删除gauge/cross term确有价值，但向condition-specific非零B0追加residual仍不能
稳定形成native-safe共同步长。ALB-NV终局，不full24/strict/resume、补A side、family mix或小扫。当前没有active
GPU run或可resume checkpoint；下一变量必须让小的共同reward方向从native-zero坐标进入effective BA且完整保留
LPCP rank16 carrier，不能压缩baseline或回到parameter-ray sweep。

最新终局successor是**V6-LPCP Native-Zero Residual Bank Commitment**（NZRB-C），authority=
`docs/action_forecast_writer_v6_lpcp_native_zero_residual_bank_commitment_design.md`。它保留ALB的上游、四B heads、
reward、optimizer与continuous新增`delta-B A0`，唯一把public LoRA从rank16 additive `B0+delta-B`改为一套rank32
state：`A=[A0;A0]`、`B=[B0,delta-B]`，其中第二B bank从native zero开始。`alpha=rank=32`故scale仍为1；原LPCP
rank16逐元素保留，无compression、SVD、第二adapter或新增trainable。clean `d4fc92e`固定task9/15/18训练与完整
held8分析均exit0；稳定结构审计证明carrier、second-B step0、base BA、连续`delta-B A0`等价与residual state五项
误差均精确0。task15/18纠正后全门通过：held BA分别`.95235/.93984`与`.93418/.92186`，raw-B分别
`.95322/.94073`与`.93272/.92047`，均8/8 held tasks；说明native-zero bank真正修复了ALB的小残差可见性与
跨video factor coherence。task9仍在j0--10无all-view candidate并exact no-op，且rank32计算shape虽FP64 BA等价，
paired outcome由预定`2/1,26 chunks`漂为`1/0,25 chunks`。三anchor合计wall/ALB=`1.16565x>1.15x`；故2/3门与
吞吐门失败，NZRB-C不full24/strict/resume或bank/rank/scale小扫。初次analysis的约`1e-3`结构报警来自跨autocast
重算carrier，已由每anchor的
`nzrb_stable_rank_bank_contract.json`纠正，不是模型结构失败。

最新完成successor是**V6-LPCP Native Endpoint Action-Preference Credit**（NEAP-C），authority=
`docs/action_forecast_writer_v6_lpcp_native_endpoint_action_preference_design.md`。它不改LPCP/NZRB carrier、rank32
public LoRA、MB-SOP同B8 action panel、四correct K4 views、optimizer或native backtracking，唯一把随机flow-time
CFM preference替换为冻结source policy完整10步部署求解后的action-endpoint preference。新objective直接比较
generated endpoint到reward-labeled winner/loser action的executed-prefix距离，gradient与acceptance使用同一个
deployment metric，不再换parameter ray。clean`33f69fd` task9 outcome/count=`1/0,25/8`，gradient
cosine/energy=`.846/.865`且j0一次接受，cycle=`97.107s`、reserved=`19.367GB`；held8全部通过且BA/raw-B/action
cosine=`.953/.955/.485`，reverse/constant与rank-bank健康。但held/train BA L2=`.234<.30`，26/27门仍按authority
终局，不运行task15/18、full24/strict或小扫。stage localization为probe/joint幅度`.671/.665x`到direct rows
`.223x`，最早缺口是one-task condition经shared direct-B head的跨task幅度。NEAP无可resume checkpoint。

最新终局successor是**V6-LPCP Task-Complete Endpoint Coexistence**（TCEC），authority=
`docs/action_forecast_writer_v6_lpcp_task_complete_endpoint_coexistence_design.md`。它不改NEAP endpoint objective、
LPCP/NZRB生成图、rank32、K4、MB-SOP、Adam或backtracking ray；唯一把commitment单位改为全部active tasks。
每task四view gradient先等权，active tasks再等权形成同一Adam candidate；所有ranks聚合每个active task×view的
endpoint margin，只接受全局全部严格下降的同一个首个scale。clean `9ed6a08` world3完整复现task9/15/18
outcome/count，cycle=`182.142s`且0禁读/OOM/nonfinite。三task各自four-view cosine/energy=
`.846/.865,.596/.645,.448/.557`且均4/4下降；但task间gradient cosine mean=`-.14513`，task15 norm分别是
task9/task18的`41.45x/10.43x`，故global raw mean只对task15下降。11个native scales最多覆盖8/12 margins，
全部拒绝并恢复exact step0；保存的860,160个B-head参数逐元素全零。TCEC按continuous 3/3与native 12/12两门
终局，不full24/strict/held controls或小扫。最早失败接口是task-local coherent endpoint credit汇成单一shared
direct-B commitment，而不是carrier、reward、global rank同步或held compiler。TCEC没有可resume checkpoint。

最新终局successor是**V6-LPCP Capacity-Matched Action-Probe Grid**（CAPG），authority=
`docs/action_forecast_writer_v6_lpcp_capacity_matched_action_probe_grid_design.md`。它保留LPCP/NEAP/K4/rank32和同一
真实joint forward，唯一把`320x256 -> four shared wide B heads`换成逐层Action-probe activations读取的
`18x37x1024`parameter grid；37由rank16 B-only payload精确推导。逐video adjacent/goal causal Program、K-set、
layer/token M2P后直接reshape native-zero B；`18x37x1024`zero-init elementwise payload gate使step0 exact LPCP、
首步gate gradient开放且不做大数相减。37个slots位于backbone之后，不是memory tokens；literal memory与rank8仍
开放。canonical实现已原位退休TCEC wide heads/config/schema：trainable=`3,008,384`，同一hook保留raw layer states，
step0/constant exact zero、gate-open全链gradient与K-set/direct-reshape合同通过；定向CPU=`79 passed`、完整CPU=
`405 passed`、architecture guard无hard violation。clean`878b5e4` world3完整复现固定outcomes/counts，cycle=
`179.973s`、0禁读/OOM/nonfinite。task9/15/18 same-task cosine/energy从TCEC的
`.846/.865,.596/.645,.448/.557`提高到`.983/.985,.898/.870,.982/.949`，raw shared coverage从1/3提高
到2/3，native最佳从8/12提高到10/12；但task15 norm仍为task9/task18的`36.29x/5.99x`，task18到shared mean
cosine=`-.1570`，11 scales无12/12，最终681,984 gate参数exact zero。故不full24/strict/held或小扫。最早失败
接口是task-local coherent capacity grid credit到shared task-conditioned first native commitment。

最新终局successor是**V6-LPCP Capacity-Matched Backbone-Memory Grid**（CMBG），authority=
`docs/action_forecast_writer_v6_lpcp_capacity_matched_backbone_memory_grid_design.md`。它严格保留CAPG已通过的
temporal/K-set/M2P/direct-B、NEAP、K4、rank32和global gate，只把backbone之后的37 query latents换成逐层读取
真实image/language/50 Action context的37个one-way memory tokens。memory走对应Action Expert update，原carrier
执行图完全不含memory；每层37 states直接形成同shape grid。37由B-only payload推导，不是阶段数。step0仍只有
coordinate gate开放，trainable=`2,828,928`。首版clean`38f7fc7`虽把cross-task cosine mean提高到`+.03865`、
native best到11/12，却令task15固定`2/0,65,16`漂成`1/2,47,8`，故不能科学裁决。修正版一次原生context forward
加one-way layer observer；真实task15 130帧的text/frame/grounded/interactions和18层Action states相对LPCP全部
max-abs0，task9完整梯度链与A40门通过。随后clean`2aecece` fixed world3完整通过：task9/15/18=
`1/0,2/0,1/2`，selected pairs=`8/16/8`，cross-task cosine mean=`+.09842`，raw/final=`3/3`、native=`12/12`，
accepted delta L2=`.168481`且q/v/action均非零。occupancy chunk数与跨运行逐元素差异只作诊断。validation8 held
8/8通过，BA cosine/energy=`.983541/.985926`、held/train=`.960650x`；该证据随后授权fresh full24 cycle1。
clean`a62348e`首次full24只是工程失败：旧formal路径只保留view0，与全局四视图
commitment合同冲突，rank0/4报`lost four video views`，其余ranks在all-gather等待后触发30分钟
watchdog；无checkpoint/completion或科学结果。canonical已最小修正为formal active task保留已算出的
4个conditioning states，不新增forward或改变科学合同；全量CPU=`411 passed`，待从新clean commit
fresh重跑。修正后clean`b4dbf84` full24已完整exit0：24 tasks/48 pairs/96 rollouts，candidate/reference=
`32/32`，6 active tasks覆盖四suite，cycle=`527.605s`。五task的same-task four-view cosine/energy约
`.988--.992/.956--.994`，但task34为`-.105/.339`；cross-task cosine mean/min=`.00681/-.21022`，task38
gradient norm是次大的`54.45x`且与shared mean cosine=`.99978`，task4则为`-.16081`。11个scale最好仅
`17/24` deployed margins下降，无候选被接受，final delta、q/v/action BA和fixed-action response全为0。按
`restore_step0_parameters_and_terminal_non_pass`合同终局；不跑只会重测LPCP carrier的strict400，不cycle2/
controls/小扫。精确artifact=`cmbg_full24_terminal_adjudication.json`。当前无active successor或GPU run；rank8、
完整A/B、Dynamic-K和memory token一般仍是独立开放变量。

## 4. Long-term objective and decision rule

EMBER研究能否从generic`lerobot/pi05_base`建立的冻结π0.5-LIBERO source policy出发，把目标task的语言和
action-hidden正确教学视频一次性编译为task-conditioned policy adaptation，使policy从未见初始化闭环完成任务。

当前主目标是Writer初次生成的adaptation本身立即有效。生成LoRA后的task-local RL是之后独立实验，不能混入
当前zero-interaction分数。性能继续追求`>150/400`并越高越好；owner最新明确补充，约`145`也可成为科学上
有价值的稳定方法，前提是它不是训练波动中的单点winner。方法资格要求同一shared method的相邻single
checkpoints同时具备：

- strict paired correct保持约`145+`或更高，而非只在一个checkpoint偶然出现；
- 高task breadth、相邻checkpoint低换手/高success-set重合、多个tasks共同积累；
- correct实质优于wrong、shuffled、reversed与no-video；
- same-task不同teacher videos鲁棒；
- 视频语义和有向过程经Program、LoRA、effective BA传到policy action；
- 高分不能主要来自language-only shortcut、挑video、expert route、checkpoint union或融合。

closed-loop absolute首先选择方法。LoRA norm/rank/cosine、reconstruction、functional loss、内部margin与hidden
差异只作诊断；不能为了几何漂亮接受明显更低的真实性能。

每轮结果必须与最接近历史架构、v6-fast143及old134/compiler138/online128逐task比较，报告per-suite、breadth、
retained/gained/lost与churn。先定位最早失效接口，一次尽量只改变一个主要因果变量。负结果只淘汰实际检验的
假设；局部问题不得无证据推翻整套已认可设计。

## 5. Data, split and shared foundation

- benchmark为LIBERO Spatial/Object/Goal/Long共40 tasks；
- development split固定在`configs/libero_24_8_8_v1/`：24 train / 8 validation / 8 test，不得按结果改task IDs；
- source corpus由LIBERO-90 specification-only audit排除19个与目标40重合tasks后保留71个，每task使用50条成功
  episodes训练共享source policy；
- 不得使用已经读过目标40 actions的`pi05_libero`；
- normalization只从过滤后source actions/states计算并冻结；validation/test不得重算；
- 选定方法后才合并32 source / 8 test并从规定初态重训。

## 6. Writer information wall and deployment

- 输入必须包含exact task language和一条或多条同task、action-hidden、内部有序teacher videos；
- language说明关注什么和目标是什么，但不能独立写LoRA；video dynamic evidence必须成为必要Value路径；
- 不得读取teacher action、proprio/state、reward、terminal、task ID、filename、object pose、hidden normalization
  或policy outcome；
- training action只属于冻结source-policy functional loss；validation/test actions或reward不产生梯度；
- 每个condition只生成一套完整38-target task LoRA；不分别生成多套video LoRA再平均，不挑video，不做checkpoint
  融合或第二套部署LoRA；
- Writer在rollout前运行一次，闭环期间不反复观看teacher video；
- frame stride保持5；frozen source policy无trainable parameters；step0/constant-dynamic路径保持functional identity；
- task experts可作train24 privileged teacher或几何诊断，但不能成为held expert dictionary、task-ID route或第二
  套LoRA。

Dynamic-K不是“多视频越多自动更好”的声明。若架构声称支持可变K，训练必须真实覆盖各cardinality；每条video
先独立保序编码，videos只在集合阶段置换不变地聚合，不平均frames、raw features或最终LoRAs。one-shot或
few-shot哪一个最终成为论文设定只由真实性能决定，不为形式公平故意削弱较强方案。

## 7. Training contract

- development只用24 train tasks产生梯度；每个完整macro按task等权；
- video与action query同task但跨episode采样，阻断逐帧低层复制；
- 多卡分配可按K、帧数和历史cost做负载均衡，但不能改变task权重；
- formal checkpoint包含Writer、optimizer、scheduler/scaler、sampler/cursor、rank RNG、world topology与schema；
- incompatible架构必须fresh；exact-resume锁原world size/topology，不能伪装fresh或跨run加载部分state；
- 机制/smoke只证明图接通。训练到有信息量的预注册节点后及时做strict paired400，不以loss替代真实性能；
- 不靠rank、scale、seed、dtype、temperature或小补丁sweep挽救失败checkpoint；
- Writer RL若未来恢复，须另立authority并保持信息墙、task balance与single checkpoint；当前Direct-Family-B阶段
  不混入RL。

## 8. Evaluation contract

- official preprocessing保持render256/model224、两相机180° rotate、state/action 7维、10 flow steps、执行前5
  actions后replan、dummy settling10、成功即终止、suite horizons 220/280/300/520；
- zero-interaction rollout从正确task的teacher videos无放回采样，不挑最好video；
- correct/same-task-other/cross-suite-wrong/shuffled/reversed/no-video严格配对task、state、env/policy RNG与video
  ordinal；shuffle/reverse必须重排真实frames后重新完整forward；
- evaluator使用cost-balanced dynamic queue、long-first和persistent workers，不静态task/GPU分配或dummy占卡；
- 正式选择只认single-checkpoint 400 paired rows。80-row screen、checkpoint union与内部surrogate不能选择模型。

任何首次达到约`145`且retention过门的checkpoint就必须补视频因果controls，并继续评测相邻checkpoint稳定性；
不能等到`>150`才检查correct是否沿有用policy direction获益。

## 9. GPU, throughput and numerical policy

- 每次GPU launch前同时live检查gpu01与gpu02，区分空闲、可共驻、忙碌与故障；
- 单节点使用至多6张真正能提高吞吐的A40。有几张合适卡就用几张，不等待凑6卡、不跨节点拼碎片、不dummy占卡；
- 少量显存占用或低利用率进程不自动排除设备，只要有足够峰值余量且不会明显干扰他人；
- 若合适空卡不足，owner已明确允许与`ycliu`用户的进程安全共驻；仍须按实时显存峰值余量与利用率判断，不得
  pause、kill、reset或明显干扰其任务。该授权不自动扩展到其他用户；
- 不reset、kill、pause、抢占或干扰他人进程；设备ownership与telemetry始终按实时状态判断；
- 多卡训练固定`NCCL_P2P_DISABLE=1`、GPU-local NUMA mapping和deferred NCCL；独立evaluator不用NCCL；
- 接受正常BF16/TF32、batch、kernel和reduction order低位差异；不为逐元素一致固定batch1、重复forward、扩dtype、
  关闭高效kernel或增加逐tensor扫描；
- 不新增SHA-256、MD5或大量防御性校验。只保留信息墙、shape、finite、OOM、asset、pairing、checkpoint和resume
  正确性所需检查；
- profile以真实LoRA/s、samples/s、最长视频稳定性和显存利用选择batch，不以最低显存为目标。

## 10. Storage, artifacts, Git and documentation

- 大资产位于`/data0/user/ymdai`或`/data1/user/ymdai`。大copy/cache/training前在`strg01`查询对应filesystem的
  独立user quota、测实际使用并估计峰值；`df -h`不是quota检查；
- 复用canonical source policy、dataset、tokenizer、assets和manifest，不复制大资产；
- formal结果保留run contract、checkpoint manifest、metrics、raw rows、aggregate、completion与必要analysis；
  profile/smoke不得冒充formal；
- 一个canonical active Writer implementation。旧实现由Git、frozen config、formal artifacts和
  `docs/research_history.md`保存，不保留可执行平行版本或兼容fallback；
- canonical workspace是`/data1/user/ymdai/projects/EMBER`，唯一主写分支是`codex/bci-continuation`。formal
  training/evaluation来自clean pushed commit的detached frozen worktree；
- 主工作树保持task diff聚焦，不提交dataset、cache、checkpoint、大binary、secret或host-private配置；
- meaningful状态只更新`current_owner_requirements`、`active_session_handoff`、`execution_brief`、当前design、
  `task_plan`、`findings`和必要README入口；历史精确结果进入`research_history`，不向退役design重复追加；
- 删除只针对生命周期明确的obsolete/temporary/duplicate内容；formal evidence、唯一checkpoint、dataset及所有权
  不清内容必须保留。

## 11. Collaboration

owner授权在上述边界内自主循环推进：历史综合 -> 单变量设计 -> 实现/机制验证 -> 吞吐profile -> 训练 -> strict
评测 -> 逐task/接口分析 -> 下一轮。当前最新要求是**暂时不使用subagents**；后续工作由主任务直接完成，直到
owner再次明确改变。不要把owner提供的启发机械照搬，也不要因一个局部建议整套摇摆；应独立判断并保留已经
对齐和有证据支持的部分。

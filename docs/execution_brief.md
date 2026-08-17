# EMBER Execution Brief

更新时间：2026-08-17。本文只定义当前实验与持续迭代的执行语义；实时进度见`active_session_handoff.md`，稳定
owner原则见`current_owner_requirements.md`，历史负结果见`research_history.md`。

## 1. Latest completed experiment and next decision boundary

当前active是**SEOD cycle4相邻稳定性扩展**，authority=
`docs/action_forecast_writer_v6_lpcp_cfmg_successful_expert_occupancy_distillation_design.md`。SEOD保留CFMG literal
memory、K4、rank32、four-view/median-cap/Adam与部署信息墙，只把MCTC稀疏binary arm credit换成train24
step2000 expert成功on-policy occupancy上的matched behavior distillation；expert不进入Writer输入、checkpoint或
held部署。clean frozen `d2f765c`、gpu01 world6已完成cycle1--3，训练wall=`383.342/391.596/378.032s`、
active tasks=`19/17/16`，且cycle2/3均有24/25 content/gate参数组gradient。

三个single checkpoints的strict score/breadth为`129/6 -> 135/6 -> 143/5`；cycle3 per-task=
`0/4/47/35/0/36/21/0`、per-suite=`4/82/36/21`。cycle2→3严格=`120 retained / 23 gained / 15 lost`、churn38、
net`+8`、Jaccard`.759494`；相对LPCP143=`121/22/22`、churn44、net0。count-only追平v6-fast143，但cycle3
top3占`118/143`且三个tasks仍为0，所以不能把aggregate143称为稳定资格。

cycle2→3 FP64 all400 effective-BA relative-L2 mean/median=`.002354/.002020`，first4同task不同K4更新
cosine/energy=`.993193/.992576`；gained/lost改写=`.002439/.002350`仍不可分。训练内跨task pairwise gradient
cosine从cycle1的`.1380`降到cycle3的`.0554`。因此同任务不同视频的共同native写出已经稳定打开，当前未决的是
继续训练能否形成更广的多task support，而非只把强task继续推高。

owner最新明确有好结果时应增加训练量再判断；aggregate连续`+6/+8`使原三轮止损上限不再机械终局。现不改任何
科学变量，只锁原world6 topology exact-resume cycle4并做第四次strict400。cycle4若低于143、继续丢breadth或
相对cycle3 gained<lost则终局；若至少143、breadth恢复至少6且gained>=lost，才可做最后cycle5。首次约145且
retention合理立即补视频因果controls；全部checkpoint都报告，不能挑cycle3峰值。

以下保留通往USFC/USDC的紧邻因果链。

此前执行的**CMBG full24 cycle1**：修正formal four-view retention后的clean`b4dbf84`在world5完整
exit0，24 tasks/48 paired states/96 rollouts，candidate/reference=`32/32`、gains=`3/3`，6 active tasks覆盖四suite，
cycle=`527.605s`，checkpoint/completion完整且0禁读/OOM/nonfinite。其preformal基础为clean`2aecece`：task9/15/18=
`1/0,2/0,1/2`、selected pairs=`8/16/8`，cycle=`114.732s`。三task four-view cosine/energy=
`.968/.748,.857/.864,.979/.977`；cross-task gradient cosine mean由CAPG的`-.13938`变为`+.09842`，shared
raw/final coverage=`3/3`、native=`12/12`，j0 actual Adam被接受并写出非零q/v/action LoRA response。world3功能
共存门因此通过。task15 occupancy chunks=`70`而历史65只作诊断；跨运行BF16/TF32、batch/kernel/reduction低位差异
不再作为门。validation8/reverse/constant held门随后全过：8/8 tasks，effective-BA cosine/energy=
`.983541/.985926`，raw/action cosine=`.982788/.986590`，held/train L2=`.960650x`。

full24中五task保持约`.99`的same-task four-view gradient cosine，但task34降到`-.105`；cross-task
cosine mean/min=`.00681/-.21022`。task38 gradient norm是次大的`54.45x`并与shared mean cosine=`.99978`，
task4则为`-.16081`。所有11 scales都无法使24个active task-view deployed margins同时下降，最好仅
`17/24`；final delta、q/v/action BA和fixed-action response全为0。按预注册合同CMBG终局，跳过只会
重测LPCP的strict400，不cycle2/controls/小扫。最早缺口是**task-local video-coherent memory credit不能
在full24形成幅度平衡、跨task兼容的shared native commitment**，不是carrier、held video或LoRA写出未接通。
精确authority=`docs/action_forecast_writer_v6_lpcp_capacity_matched_backbone_memory_grid_design.md`。

随后建立的successor是**CFMG**，authority=
`docs/action_forecast_writer_v6_lpcp_content_first_memory_grid_design.md`。代码定位发现CMBG的zero gate位于
temporal/K-set/M2P之前，因此首步这些模块只贡献零输入处的固定Jacobian，而没有先形成内容依赖的有序程序。
CFMG只把同一个gate移到完整grid之后：参数量、rank32、K4、reward、全局commitment和step0=LPCP全部不变；
language仍只提供context/address，新增B Value必须经过视频动态。canonical实现与fresh config/checkpoint/eval
schema已原位完成，完整CPU=`411 passed`、architecture guard无hard violation。clean`010487b` fixed world3
保持task9/15/18成功关系=`1/0,2/0,1/2`与selected pairs=`8/16/8`，cross-task gradient cosine全正、
raw/final=`3/3`、native=`12/12`且j0接受。validation8 held四view又以8/8、BA cosine/energy=
`.982412/.985173`、raw/action=`.981593/.987546`、held/train=`.960548x`通过。原始constant门保留
`.005144>.005`的单项false；exact constant-hidden为0且实际public残差仅`.514%`，按既有数值政策另存透明
adjudication后授权了一个fresh full24 cycle1。clean`bb5341e` world5 full24完整exit0：24 tasks/48 pairs/
96 rollouts、candidate/reference=`32/32`、6 active tasks，cycle=`467.783s`。所有11 candidates最好仅`14/24`
task-view margins下降，final delta及q/v/action response均为0；task38相对次大梯度=`58.73x`，task34跨video
cosine=`-.0923`。所以CFMG按合同终局并跳过只会重测LPCP的strict400，不得resume。

USEP clean`6033330` fixed task4/34/38 world3已经完整裁决：既有outcome/count复现，task38 gradient dominance从
`58.73x`降到`6.1538x`，cross-task cosine mean/min=`.16247/.08040`、raw shared=`3/3`，actual Adam j0使
12/12 normalized deployed margins下降并写出非零q/v/action response。task34 raw four-view仍只有`2/4`、
cosine/energy=`-.05217/.33803`，所以USEP依预注册门终局，不held/full24/strict且不得resume。

只读stage localization随后证明task34四组K4从endpoint到content grid均约`.984--.986`一致，carrier A/B/BA和
`content delta-B A0`也约`.99`一致；冲突位于共同BA经condition-local policy/action Jacobian回传，而非视频
Program、K聚合或LoRA Value。其fresh successor USFC随后只把raw局部coverage降为diagnostic，并已得到上文
`17/20`但被硬门恢复step0的终局结果。

最新完成closed-loop实验是**V6-LPCP Direct Joint Native-Factor Residual**（DJNFR），authority=
`docs/action_forecast_writer_v6_lpcp_direct_joint_native_factor_residual_design.md`。clean frozen `49a4129`从sealed LPCP
fresh完成full24 cycle1：24 tasks/48 paired states/96 rollouts，candidate/reference=`34/33`、gains=`5/4`、9 active
tasks，cycle=`717.940s`；完整world4 checkpoint、completion与0禁读/OOM/nonfinite均保留。

同一cycle1 checkpoint的K4 strict paired correct400=`136/400`、breadth7、per-task=
`1/2/44/35/0/35/18/1`、per-suite=`3/79/35/19`。相对LPCP143严格=`120 retained / 16 gained / 23 lost /
241 both-fail`、churn39、net`-7`、Jaccard`.754717`；suite net=`-2/-4/-3/+2`。correct≥142与lost≤15两门失败，
按authority终局：不resume cycle2，不补same/wrong/shuffled/reversed/no-video，不做参数小扫。

本轮真正推进了因果定位。post-train task4与validation8 four-view BA cosine/energy=`.802835/.800444`和
`.790242/.785834`，held8 8/8过门；raw factor/action cosine=`.677321/.686125`，held/train BA L2=`1.08903x`。
all400相对LPCP effective-BA relative-L2 mean/median=`7.773e-5/5.080e-5`且q/v/action均非零。因此SJNV的
coherent hidden到frozen W2/native public LoRA断裂已被关闭，carrier、视频顺序、direct写出与跨视频共同方向工作。

但写入内容与reward不对齐：gained/lost/retained-failure BA relative-L2 mean=`4.522e-5/7.248e-5/8.987e-5`，
持续失败样本最大。active-task gradient pairwise cosine mean仅`.002470`。所以最早失败接口是**selected-success
only credit怎样在一个shared direct-factor update中形成held on-policy reward-useful方向**，而不是carrier、LoRA
幅度、量化、W2或跨视频coherence。

PAFS-NV随后因validation8仅`.168111/.372863`、3/8过门在formal前终局。SJNV-Gate进一步把固定地址换成共享
joint gate；clean `913d3d3` task4达到`.472272/.597814`，但validation8只有`.201903/.396448`、2/8过门，action
cosine=`.042986`，同样没有full24/strict。关键新证据是stage localization：joint gate/continuous hidden cosine
约`.94`，冻结W2后的raw factor delta却只有`.021353/.265925`，action factor cosine=`.002672`。因此最早断点已
锁定为**coherent video-language hidden -> frozen W2 -> native public A/B**，不是carrier、时序或joint input。

最新终局successor是**V6-LPCP Direct-Factor Paired Common-State Preference**（DF-PCSP），authority=
`docs/action_forecast_writer_v6_lpcp_direct_factor_paired_common_state_preference_design.md`。它从sealed LPCP fresh并
完整保留DJNFR生成图，唯一把成功整轨迹正蒸馏改为candidate/reference discordant pair在分叉前同一初始观测处的
winner-vs-loser首段flow preference。两臂共享flow time/noise，只比较共同执行长度；四个disjoint correct K4
views等权。它与历史OPPP的不同是状态严格配对且不比较分叉后的occupancy。memory token、rank8与dynamic K仍
开放，但本轮没有同时改变。首次clean `de6c812` task4 smoke在credit前确认重复相同seed/reset
不能保证两臂首观测逐元素相同；`07b764b`又确认flattened state恢复后，hard reset仍改变未被state覆盖的model pose，
差异只出现在两相机，language/state tokens完全相同。两次都只是工程失败、没有科学结果。canonical现每lane只做
一次hard reset+settling，每臂以deterministic soft reset清空controller/observables后恢复相同qpos/qvel；不增加
rollout或forward。exact task4/task7均变为tie；task9/15/18分别有1/2/1个discordant pairs，margin均真实下降且
八head/q-v-action全通，但task9 held/train BA幅度仅`.105x`，task18 train跨video仅`.290/.428`，三个有效anchors
只有task15全门通过。按门终局，不full24、strict或cycle2。

最新终局successor是**DF-SOCP**：保留LPCP/DJNFR全部生成与部署图，只沿winner成功occupancy的每个replan
observation，用相同policy noise查询loser counterfactual action，再做逐状态paired flow preference。它复用DJNFR
完整成功trajectory并补同状态negative，不增加env rollout；固定task9/15/18三anchor全部过门才允许full24。
authority=`docs/action_forecast_writer_v6_lpcp_direct_factor_successful_occupancy_counterfactual_preference_design.md`；
clean `16cedb9`固定outcomes与26/65/44 replay均复现，三者train/held跨video、q/v/action及顺序门基本全过；但
stored动态B2/B1 action与B8 loser requery在task9/15的数值差异是名义策略contrast的`1.086x/1.693x`，故action
panel不是matched-batch因果比较。三项wall又为DF-PCSP的`3.083x/5.335x/3.887x`，完整轨迹functional credit占
主要成本；task9 held/train BA仅`.118x`。按预注册门终局，不full24、strict、cycle2或小扫。下一步只应修复同B8
双臂重查与时间分层informative occupancy压缩，不能改carrier、memory、rank或LoRA topology。当前无GPU run或可
resume checkpoint。

MB-SOP已用同B8 action panel与8段max-disagreement occupancy解决批形混杂和成本门：task9/15/18 wall仅为DF-PCSP
`1.655/2.119/1.542x`，train/held四video BA健康。但task15/18真实AdamW后同一panel margin反而增加，task9
held/train仅`.1096x`，故未full24/strict。额外四view flat gradients证明三任务raw等权均值均为`4/4`共同下降；
最早接口是raw functional gradient到coordinate-preconditioned finite parameter delta。

AR-EC clean `b578d56`已终局：三anchor raw gradient均为`4/4`共同下降且final到`-g` cosine1，但同Adam full radius
后每个任务仅`1/4` post margins下降；radius/raw-gradient L2=`6333/7988/4294x`。train/held BA coherence、
q/v/action、reverse/constant与core wall都健康，故不full24/strict，最早接口是finite trust radius。

最新终局successor是**AV-MBC**：完整保留MB-SOP/AR-EC matched panel、四video等权、direction、LPCP/DJNFR、
rank16与八heads，只沿`-g`从Adam upper radius确定性减半，接受第一个四view同panel/noise margin全部下降的candidate。
不挑最佳scale、不产生多checkpoint。authority=
`docs/action_forecast_writer_v6_lpcp_direct_factor_all_view_monotone_backtracking_commitment_design.md`。clean `aa819f2`
三anchor都完整exit0，但训练前gradient CFM margin与candidate inference CFM margin不是同一执行路径，导致小scale
不收敛到零；恢复step0后fixed-action仍有约`.0021--.0030` RMS，又暴露rollout batch与batch1 probe混比。因此
`aa819f2`只作工程证据，不能裁决AV-MBC。canonical已只修测量合同：同inference evaluator先测step0再比较candidate，
fixed-action前后使用相同batch1 query/noise；CPU=`404 passed`、architecture guard无hard violation。clean
`202a64d`真实结果为task9 `j10`但held4/8与held/train`.184x`失败，task15全部11个candidate拒绝并恢复exact no-op，
task18 `j5`全门通过。一个scalar radius因此无法成为shared稳定commitment，AV-MBC终局且无full24/strict。

最新终局successor是**MMCD**：不改video-language、reward、FactorHeads、Adam半径或backtracking，只把每task四
view raw mean方向换成对四个raw gradients具有最大worst-view一阶下降余量的共同方向；方向再缩放回原mean norm，
formal跨task仍等权。该方向由已有gradients的`4x4` Gram确定性求解，不增加forward或模型参数。authority=
`docs/action_forecast_writer_v6_lpcp_direct_factor_maximum_margin_common_descent_commitment_design.md`。clean `fc3bdd7`
固定task9/15/18的continuous worst margin提高`1.216/1.334/1.356x`，但native分别为j0且held/train`.160558x`、
j0--10全拒绝并exact no-op、j6且全门通过；只有1/3 anchors通过。formal未解锁，MMCD终局，不full24/strict/
resume或solver/rank/LR/scale sweep。当前没有active GPU run；下一设计必须直接针对native finite-step metric与held
amplitude，不能继续只优化continuous gradient geometry。

最新终局successor是**PAV-BC**：完整保留LPCP/MB-SOP/AdamW与同路径all-view acceptance，唯一把final ray从
raw/MMCD direction换成实际Adam candidate delta并固定减半。MB-SOP只测过该ray的full step，AV-MBC只回退raw ray；
clean `581140c`结果为task9 j5但held/train`.109466x`，task15/18到j10仍无共同candidate并exact no-op；0/3过门。
PAV-BC终局，不full24/strict/resume。raw equal mean、raw maximum margin与Adam preconditioned rays均已否决；下一变量
必须转LoRA输出/effective-BA参数化。authority=
`docs/action_forecast_writer_v6_lpcp_direct_factor_preconditioned_all_view_backtracking_commitment_design.md`。

最新终局是**ALB-NV**：保留LPCP143、MB-SOP、PAV acceptance与rank16，唯一固定LPCP A并只训练四个B residual
heads（860,160参数），令新增`BA=delta-B A0`严格线性、step0 exact LPCP。B side由LPCP correct400固定factor
geometry选择，不按task/held切换。authority=
`docs/action_forecast_writer_v6_lpcp_anchored_linear_b_native_value_commitment_design.md`；clean `0899166`固定task9/15/18
只1/3过门。task9连续梯度`.415/.559`但11个native steps均非四路共同下降，exact no-op；task15在j5写出并通过
held BA aggregate/held-train/action，但仅5/8且raw-B `.101/.323`；task18 j0全门通过，held BA `.774/.785`、8/8、
held/train`1.030x`。因此fixed-A线性化有效但不稳定，终局不full24/strict/resume或side/scale小扫。当前没有active
GPU run或可resume checkpoint；下一变量必须使共同Value从native-zero residual坐标写入且完整保留LPCP rank16
support，不能压缩baseline、重新混合A/B或继续parameter-ray路线。

最新终局是**NZRB-C**：不改ALB continuous方向，只把一套public LoRA写成rank32
`A=[A0;A0], B=[B0,delta-B]`。第二B bank从native zero开始，第一rank16 bank逐元素保留LPCP；alpha/rank仍为1，
四heads与trainable仍860,160。它是“同一`delta-B A0`、不同native origin”的严格反事实，不是两套LoRA、
compression或capacity sweep。authority=
`docs/action_forecast_writer_v6_lpcp_native_zero_residual_bank_commitment_design.md`。clean `d4fc92e`三anchor完整：稳定
结构五项精确0；task15/18 held BA=`.952/.940,.934/.922`、raw-B=`.953/.941,.933/.920`且均8/8，纠正后通过；
task9 paired outcome漂为`1/0,25`且11步仍no-op。总wall/ALB=`1.16565x>1.15x`，所以2/3门与吞吐门失败，终局
不full24/strict/resume。zero bank解决accepted update的native/held coherence，但没有解决reward方向是否存在
all-view finite policy step；不再改factor origin。初版结构假报警由稳定rank-bank artifact纠正。

最新终局是**NEAP-C**：完整保留LPCP/NZRB/MB-SOP与native backtracking，只把CFM surrogate credit替换为冻结
PI05完整10步求解后的deployed-action endpoint preference。每个selected共同observation用同一noise生成一次
condition action，同时比较reward winner/loser的executed-prefix距离；四views/tasks继续等权。先只跑task9的
clean`33f69fd` task9的gradient cosine/energy从NZRB`.286/.448`提高到`.846/.865`，j0一次接受，cycle=
`97.107s`、reserved=`19.367GB`；held8为8/8，BA/raw-B/action cosine=`.953/.955/.485`，时序与rank-bank均健康。
但held/train BA L2=`.234<.30`，故26/27门仍终局，不跑task15/18/full24。probe/joint幅度保持`.671/.665x`，
direct rows仅`.223x`，最早缺口是one-task condition到shared direct-head的跨task幅度。authority=
`docs/action_forecast_writer_v6_lpcp_native_endpoint_action_preference_design.md`；NEAP无可resume checkpoint。

最新终局successor是**TCEC**：不改NEAP endpoint、LPCP/NZRB、rank32、K4、MB-SOP、Adam或native ray，只把
commitment从一个task-local probe升级为所有active tasks全局共同裁决。固定task9/15/18将在world3中共享一次
equal-task-mean update；12个correct-view endpoint margins必须全部严格下降且三个ranks接受相同scale。随后用三个
train anchors的平均幅度重新裁决validation8 held/train>=`.30`。authority=
`docs/action_forecast_writer_v6_lpcp_task_complete_endpoint_coexistence_design.md`。canonical已原位实现world3固定anchor、
active-probe list和global scalar-row backtracking；定向=`41 passed`、完整CPU=`405 passed`、architecture guard 0 hard。
clean `9ed6a08` world3随后完整复现三anchor并在`182.142s`结束。每个task内部四video credit均为4/4共同下降，
但task间gradient cosine mean=`-.14513`，task15 norm比task9/task18大`41.45x/10.43x`，所以global shared mean
只对1/3 task下降。11个scale最多覆盖8/12 margins并全部拒绝，最终B-head state 860,160参数全零。TCEC在held前
终局，不full24/strict/controls。

最新终局design是**CAPG**：保留LPCP/NEAP/K4/rank32与同一真实joint forward，删除四个shared wide B heads；
从18层50个Action-probe states用37个capacity-matched post-backbone parameter latents读取context，经video内
adjacent/goal causal reducer、K-set和layer/token M2P直接reshape native-zero B。`18x37x1024`zero-init
coordinate payload gate使初始化exact LPCP、首步gate gradient-open，accepted update后其余链路打开；不做anchor
大数相减。37不是memory token；literal memory与rank8仍开放。canonical实现已原位完成，trainable=
`3,008,384`，定向/完整CPU=`79/405 passed`且architecture guard无hard violation。clean`878b5e4` world3
outcomes/counts与信息墙全通过，cycle=`179.973s`；三task same-task cosine/energy提升到
`.983/.985,.898/.870,.982/.949`，raw shared coverage=`2/3`、native best=`10/12`。但task15仍主导且task18
到shared cosine=`-.1570`，没有12/12 scale，最终exact no-op；不full24/strict/held或小扫。

最近完成的CMBG严格保留CAPG的有向video/K-set/M2P/direct-B与NEAP/K4/rank32/global gate，唯一把
post-backbone 37-query latents换成逐层读取真实image/language/50 Action context的37个one-way memory tokens，形成
`18x37x1024`grid。memory走真实Action Expert layer，carrier不看memory且保持原生执行shape；step0仍只有coordinate
gate开放，trainable=`2,828,928`。真实task15 130-frame carrier parity的五组输出均max-abs=`0`；task9 112-frame
CUDA检查确认exact-zero、gate-open全链gradient、wall=`121.251s`、peak reserved=`18,848 MiB`。fresh world3仍必须
复现固定outcome/count并达到raw3/3、native12/12；authority=
`docs/action_forecast_writer_v6_lpcp_capacity_matched_backbone_memory_grid_design.md`。该轮full24及其后继CFMG、USDC、
MCTC均已按上文终局；当前没有active GPU run。
稳定约145资格高于单点分数：至少需要相邻single checkpoints低churn/high-overlap、same-task-other鲁棒，并在
同一final checkpoint上证明correct相对wrong/shuffled/reversed/no-video的明确paired优势。

DJNFR训练root=
`runs/outputs/pi05_v6_lpcp_direct_joint_native_factor_residual_formal_cycle0to1_r4_k4_views4_nmc4_b8_49a4129_gpu02_20260815`；
strict与终局分析root=
`runs/outputs/pi05_v6_lpcp_direct_joint_native_factor_residual_cycle1_k4_correct400_noreplacement_seed7_trainr4_evalr6_49a4129_gpu01_20260816`。

## 2. Completed CV-CSD variable and exact conclusion

- 部署继续是exact language + dynamic K ordered action-hidden videos一次生成一套38-target rank16 LoRA；
- anchor K4只产生AS139/LPCP paired唯一成功trajectory；同一trajectory在4个disjoint same-task correct K4
  conditions下分别通过完整Writer→LoRA→policy CFM；
- 四view只在65,536参数`query_delta.weight`梯度处等权汇合，不平均video、Program或LoRA；
- 只训练query commitment，LPCP carrier、AS139 tail、source policy、optimizer与rollout数不变；
- 结果只否定“query-only map + four-view exact selected-success mean”，不否定few-shot、memory、reward或生成LoRA。

## 3. Closed-loop adjudication

Ordered-Procedure AS139、raw reward138、ADSP138、V6-LPCP、PCSD、CV-CSD、SFMC、Gradient-Open与CCT均已终局且
不得resume或小扫。
各自训练root、strict root、逐episode transitions和必要BA/跨video evidence均已封存。`>150`仍是更高性能追求；约145
也可成为有效结果，但必须由相邻single checkpoints低churn、same-task-other鲁棒和correct相对wrong/shuffled/
reversed/no-video的明确优势共同认证。单点145或151都不算完成。

报告aggregate、8项per-task、4 suite totals、breadth、retained/gained/lost、top-task concentration和K1→K4
success-set变化。不能用K1/K4 union、LoRA norm或functional loss冒充同一condition的能力。

SFMC与CCT均因cycle1预注册门失败而没有相邻checkpoint与controls；这不是缺失分析，而是预注册停止规则。下一架构仍须先以
cycle1 absolute/breadth/retention筛选；过门后必须评相邻checkpoint，再做same/wrong/shuffled/reversed/no-video。
单点145或151都不能跳过稳定性与视频因果资格。

## 4. Continuous adjudication loop

每轮strict结果完成后，按以下顺序分析：

1. absolute、per-task/per-suite、breadth；
2. 相对最接近方法和历史强基线的retained/gained/lost与能力集中；
3. 若有相邻checkpoint，分析persistent/gained/lost与union gap；
4. 沿`input evidence -> Core/Procedure -> set/compiler -> effective BA -> fixed action -> rollout`定位最早失效接口；
5. 分离科学non-pass与明确工程合同违约；
6. 只对最早接口提出一个主要因果变量，写可证伪authority后实现；
7. 做最小必要CPU/机制验证和吞吐profile，尽快回到真实paired400。

不得用loss、cosine、rank或漂亮五臂margin代替absolute，也不得通过rank/scale/seed/dtype/temperature小扫救一个
失败checkpoint。owner的局部建议不能导致整套已认可方案无证据重写。

## 5. GPU and efficiency

- launch前同时live检查gpu01/gpu02；单节点至多6张，有多少真正合适就用多少；
- 允许在显存峰值余量充足、低util且不明显干扰他人的卡上共驻；不抢占、kill、reset或dummy占卡；
- evaluator当前保守共驻门为util≤10%、已用≤8GiB且剩余≥32GiB；任一越界即拒绝；
- 多卡训练设置`NCCL_P2P_DISABLE=1`、GPU-local NUMA和deferred NCCL；
- fresh可用world1--6；exact resume锁原world topology；
- evaluator用动态cost queue和persistent model/env，不静态拆task；
- 以真实samples/s、LoRA/s、最长视频稳定性和显存峰值选择batch；
- 接受正常BF16/TF32低位差异，不重复forward、固定batch1、扩dtype、逐tensor scan或增加hash。

## 6. Storage, Git and artifacts

- 大run前查询`strg01`上的独立user quota，测canonical root并估计checkpoint/cache/temp峰值；
- formal训练与评测来自clean pushed commit的detached worktree；
- 新run使用fresh output root，不覆盖或部分复用中止/不兼容root；
- formal保留run contract、metrics、macro checkpoints、completion、400 raw rows、aggregate和decision analysis；
- profile/smoke roots只作机制/吞吐证据，不冒充formal；
- meaningful结果更新current state、execution brief、current design、task plan、findings和research history；
- 不把历史命令重新复制进多个文档，精确命令以run contract/invocations为准。

## 7. Collaboration boundary

owner授权在核心目标、信息墙与效率原则内持续自主迭代。当前暂不使用subagents。只有真正需要改变核心目标、
扩大权限、处理破坏性操作或遇到无法从本地证据解决的关键歧义时才向owner停下来请求决定。

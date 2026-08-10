# EMBER Current Execution Brief

## 0. Outcome and current operation

目标是把同一shared method、同一single checkpoint的strict paired correct从历史最好`143/400`推进到
严格`>150/400`并继续提高，同时保留真实视频时序因果、same-task鲁棒、breadth和稳定积累。当前没有
运行中的EMBER GPU任务。

**最新执行状态（2026-08-10）**：第39.5节Reward-Credit Program Cotangent已在唯一canonical path完成
实现和CPU seal，active config为`configs/pi05_v6_reward_credit_program_cotangent_v1.json`，当前只等待
full24×K4×Nmc4 live A40 discarded profile，尚无Reward-Credit GPU、训练或strict结果。部署继续是exact
language + exactly one action-hidden video、Balanced P256 key、frozen historical-v6 decoder、single Program和
一套完整38-target rank16 LoRA；fresh `M0=0,Lambda0=I`，禁止继承RLS10。

每task同一one-shot LoRA做K4 batch4 official random-reset rollouts，成功/失败都保留executed prefix；binary
LOO `A_e=(4R_e-sum R)/3`，全成/全败严格零且跳过replay，mixed task按episode等权和Nmc4 keyed CFM直接形成
Program cotangent。full48仍以24 correct + 24 exact-zero negative rows做single manual write；无old/current第二
forward、ratio、第二epoch、shared Adam、progress/SPSA或negative policy forward。六rank×4 tasks、BF16、四
persistent env lanes、deferred NCCL与K4 batch4固定；replay B2是历史A40容量上限的初始高吞吐点，不是数值精度
退让。全仓CPU回归`336 passed in 59.12s`；正式动作只有clean push/frozen后live preflight与一次profile。

**最新已完成实验（2026-08-10）**：第39节RLS已从clean frozen`25bbd52`完成formal fresh0→10与预注册
macro10 strict correct400。训练natural exit0、10 rows、step sum/mean=`199.425195/19.942519s`、input wait=
`.278241s`、peak allocated/reserved=`43,247,554,048/46,919,581,696B`，0 OOM/nonfinite/negative policy
forward。strict=`140/400`、breadth6、per-task=`2/3/47/35/0/34/19/0`；相对exact balanced macro0=`134`
严格paired retained/gained/lost=`119/21/15`、net`+6`、churn36，未过`lost<=6`门。相对blind-v2 macro10
同为140却有`17/17`换手。RLS没有保住closed-loop旧成功，不续25、不补六臂、不扫超参。

full400 transition已封存在
`runs/outputs/pi05_v6_exact_anchored_reconciliation_macro0010_historical_baseline_transition_866cca9_20260810/analysis.json`。
correct80会显示`31 vs 26`、gained/lost=`5/0`，与full400的`21/15`相反；以后不使用80-row screen选点。
RLS内部current/blind降至`.230340`、final precision condition约`8325`，而formal没有保存历史reference rows；
它付出了acquisition抑制却没有对应闭环retention。config/runtime已封为formal non-pass，fresh/restart/resume
均fail closed。其最早失败接口是source-action functional cotangent与closed-loop occupancy/reward credit错位，
这正是当前第39.5节只替换training cotangent、保持部署图不变的依据。

此前第38节v2已从clean frozen`abd8e08`完成formal0→25，root=
`runs/outputs/pi05_v6_balanced_causal_condition_residual_formal_r6_lb20_mb10_abd8e08_20260810`。25 macros
step sum/input wait=`535.464796/2.208183s`，peak allocated/reserved=
`43,247,029,760/46,917,484,544B`，0 OOM/nonfinite/negative policy forward。macro0/10/25 strict
correct400=`134/140/139`、breadth均6；0→10、0→25、10→25 gained/lost=`19/13,18/13,12/13`，m0∪m10=
`153`但不能融合。macro25内部72/72 shards、400 rows、18 return0均完整，外层wrapper exit记录缺失，状态必须
写为scientific complete / wrapper unobserved，而不是exit0。

v2机制、部署和短期`+6`均真实，但macro10 effective-BA delta/base中位仅`1.69498e-4`、stable rank仍约
`1.000022`；同task 50-video correction consistency=`.141539--.142175≈1/sqrt(50)`，fixed10 effective-BA
pair cosine跨8 tasks为`[-.001371,.003280]`。所以blind-add正在写入近正交video-specific小扰动，能产生
换手却不能保留旧能力；v2不续50、不补多臂、不扫超参。

第39节**Exact Anchored Reconciliation**当时保持Balanced v2 P256 one-shot graph；训练端
用FP64`Lambda_0=I_256`和RLS把每批target锚到更新前condition输出，再叠加`[-G;0]`。checkpoint增加
training-only precision/assimilated_rows，部署仍只加载Program memory。family fresh-incompatible。
旧`f0c3f51`按原合同保持16/18 non-pass。第39.4.1不改热路径；clean pushed/frozen`f28fc8b`随后独立
fresh0→3 natural exit0且17/17通过：old drift/blind=`.248611/.213872`、old rows改善全1、current/blind=
`.999980/.784334/.640650`，production ratios=`.947963/.983678/.925249`、mean=`.952297`，0 checkpoint/
OOM/nonfinite/negative forward。config已登记`100452B` raw artifact并formal-ready；profile权重永久弃用。
上述formal/strict现已执行并由full400否决；profile与启动合同继续保留为历史机制证据，不再授权RLS训练、
续训或评测。

以下为此前按时间发生的完整证据，不能覆盖上面的active边界。v6-prior whole-LoRA objective已完成formal
0→50，同一schedule macro0/10/25/50 strict correct400=`134/127/105/123`；macro0仍最佳，因此已退役。

第34节v6-Initialized Expert-Component Projection（ECP）已完成formal0→25并负裁决。
macro10/25 strict correct400=`133/120`；同schedule macro0=`134`。macro25对macro0的精确
paired gained/lost=`13/27`、net=`-14`、McNemar `p=.038477`，per-task=
`0/1/43/27/0/33/15/1`。内部`a_correct`已从`.736`增到`.884`且23/24 tasks向1移动，
expert component也在24/24 tasks上升；但macro10→25的expert-orthogonal norm增量`8.471`远大于
component增量`.228`。所以ECP的实现和机制生效，但held closed-loop明显退化；按预注册
不续50/100、不扫权重、不补六臂。

第35节**v6 Condition-Local Dynamic Expert Tangent Tube**也已正式完成并退役。clean frozen
`b308941`的fresh0→10自然exit0；10 macros总step wall=`207.444s`、input wait=`.265s`、peak
allocated/reserved=`43,316,440,064/47,112,519,680` bytes，0 OOM/nonfinite。macro10的correct/
negative relative-anchor tube中位=`.01390/.01408`，但directional ratio中位=`108.93/126.88`、
两臂均`0/24` tasks通过`≤1`；task median `|a_correct-1|=.25229`且completion为`0/24`。
共享Writer因此学到的是“小而几乎全正交的更新”，不是所设计的expert方向修正。

紧接着同一frozen checkpoint的one-shot correct400自然完成：`131/400`、correct80=`27/80`、breadth5、
per-task=`0/3/46/31/0/40/11/0`、per-suite=`3/77/40/11`。相对同schedule macro0=`134`严格
paired gained/lost=`16/19`、churn35、net`-3`、McNemar `p=.735879`。所以不续25、不补六臂、
不扫tube weight/LR/WD；current config/runtime已封为formal non-pass并fail-closed。该结果只淘汰当前
tangent recipe/window，不能把未达到completion的实验扩大解释为expert-component假设整体无效。

第36节matched Expert-Flow Teacher Viability Audit已从clean frozen`e8e4728`正式完成并自然exit0。
480/480 queries、24 tasks、suite 6×4、8/8/8 negatives、144 PI05 forwards、0 update/rollout/OOM/nonfinite；
wall/input wait=`39.698/.684s`，peak allocated/reserved=`43,418,974,720/47,133,491,200` bytes。
expert/macro0/tangent10 matched真实7维flow loss=`.098631330/.091801740/.091843160`。expert只在`2/24`
tasks和`0/4` suite means同时优于两baseline，明确未过`18/24+3/4`teacher门。compiler/factor gradient
residual=`.6864/.8387`虽通过非冗余门，但来自整体更差teacher，因此`authorize_cefd=false`。

第37节**Frozen-v6 Counterfactual-Null Condition-Kernel Program Residual v1**已完成唯一macro49 profile并
正式non-pass。clean frozen`6903ee6` root=
`runs/outputs/pi05_v6_condition_residual_mechanism_profile_macro49_r6_lb20_mb10_6903ee6_20260810`，不保留
checkpoint/weight。13门中10门通过：feature rank48、correct motion/cotangent=`.807966`且24/24 task、
application closure=0、A/B response与4/4 fixed-action均成立；0 OOM/nonfinite/negative policy forward。
所以显式kernel和frozen-v6 Program→LoRA→action路径有效，失败不是工程bug。

v1 key的regularized condition=`1315.33`，aggregate negative/correct=`.264351>.25`、task-local null=
`15/24<18/24`。shuffled/reversed/wrong feature cosine mean=`.98552/.95645/.90627`，null通过仅
`2/8,6/8,7/8`；全部失败row cosine均`>=.97099`。四时间矩的DC块使顺序反事实近重复，paired ridge
解析leakage与实测相关`.99021`。production ratio=`1.115458>1.10`也按原门保留non-pass，但跨host
input-wait差已大于超门`.326s`，不扩大解释为稳定结构慢化。禁止训练v1、降低lambda或扫seed/P/阈值。

第38节**Balanced DC--Causal Condition Key v2**随后成为当时的active实现。它严格冻结historical v6 macro400
600 tensors，保留同一个`[256,320,256]` FP32 memory、full48、`.01` damping、step1、B20/B10+10和0
negative policy forward；只把key改为video-DC static与phase-centered sqrt-causal-prefix dynamic两个
独立JL128 blocks，各自zero-L2后拼成P256。zero/no-video仍精确identity；reverse/shuffle共享static但
RHS为`g/0`，static不能单独拟合。deployment仍只输出一套38-target rank16 LoRA。

一次性teacher-audit/effective-objective/flow-teacher执行路径已删除；checkpoint只保存Program memory、
cursor和六rank RNG，historical v6 base与fixed projection不归checkpoint所有。训练、checkpoint、deployment
v8 adapter、one-shot六臂证据和cross-family analysis已经联锁；错误family不能继承本候选的profile seal。
profile artifact会从raw macro重算全部门并匹配完整scientific run，formal result必须绑定completion、50-row
metrics与macro10/25/50 manifests，trained deployment checkpoint必须位于active authority lineage；clean
detached frozen authority ancestor可直接运行v8 evaluator。v2聚焦`52 passed`、带LIBERO assets全仓
`281 passed in 21.34s`；同static/反dynamic两帧反例的natural/reversed unit keys内积为0。这是clean
implementation commit`5d93434`的CPU seal：compileall、26份JSON、diff-check通过；architecture guard相对
`6903ee6`为`+144/-126`、净增18行且0 hard violation，1243行legacy contract未增长。

clean frozen`5d93434`的v2 macro49 mechanism profile现已正式**13/13通过**。root=
`runs/outputs/pi05_v6_balanced_causal_condition_residual_mechanism_profile_macro49_r6_lb20_mb10_5d93434_20260810`；
rank48、condition=`106.114`、correct/cotangent=`.968254`、negative/correct=`.0218514`，24/24 correct与
24/24 null。24个tasks全部相对v1同时提高retention并降低leakage；shuffled/reversed/wrong cosine mean=
`.479565/.013732/.507178`，三臂leakage max=`.048462/.032562/.033571`。A/B、4/4 fixed-action、closure、
0 negative forward均通过。

production=`20.021842s`、对sealed baseline ratio=`.949122`；本次与baseline同为
`gpu01:0,1,2|4,5,7`，input wait=`.069295s`对`.076318s`，吞吐pass不是跨host wait假象。profile不保留
checkpoint，结束后六卡释放。artifact已由config verifier从raw result/run/completion重算并seal；formal
初始写回曾被标为ready，但v8 deployment graph当时仍无自己的live吞吐/smoke seal，且当时没有v2训练或
closed-loop成绩；这个状态缺口已由下一段的fail-close修正，不能按旧ready恢复执行。

GPU前复核又修正了formal提前ready与profile-only evidence两个执行缺口：新增唯一deployment-seal owner
共同重读profile/results/cache。全仓`283 passed in 26.10s`，architecture guard相对`5d93434`为
`+968/-318`、0 hard violation；旧contract缩到1101行，无parallel family或热路径变化。

clean frozen`2af82aa`现已在实时空闲`gpu02:0`完成该双root seal。固定validation8×4、32-request/
1093-frame panel的batch8/16/32=`.911238/.901898/.906482 LoRA/s`，三点均稳定、reserved约12.9GB（约12.0GiB），
选择实测最快batch8。validation8×state0 correct vertical smoke随后真实生成8套完整native LoRA并完成8条
LIBERO闭环：`4/8` success、总wall=`336.056s`、单次launcher、0 retry/runtime failure/forbidden reads，Writer释放且
source policy复用。三件raw artifact由assembler重算通过，config现为formal ready；该8-row结果只作执行证据。
deployment写回后的全仓CPU门为`284 passed in 26.86s`，compileall、Black、JSON、diff-check、raw seal重建和
formal runtime均通过；pre-deployment formal fail-close仍有独立负回归。

deployment seal由clean pushed`d228d0d`封存。其frozen worktree第一次CPU-only formal prepare在任何CUDA
初始化前触发`residual Writer config violates its sealed contract`：不是科学non-pass，而是evaluation
artifact路径先`.resolve()`到canonical仓库，再被旧逻辑强制要求位于frozen worktree内。`af7b101`把唯一
路径owner修为仅接受词法`runs/outputs/...`且resolved target仍在canonical outputs root；nested symlink
逃逸和vertical manifest越界继续fail closed。全仓新门为`285 passed in 21.38s`。clean frozen`af7b101`
上的同一prepare已exit0，登记8 tasks×50 states、correct/without-replacement、historical-v6 macro400
load-only + `[256,320,256]` fresh elementwise-zero residual、18 rollout workers + 18 Writer generators、batch8和约1.064GB新增root
预算。prepare为CPU-only，未生成rollout、LoRA cache或性能证据。
临时prepare root已清理，正式run必须使用全新不存在的output root。

该zero-memory macro0随后已从clean frozen`6b5f7a6`在实时空闲`gpu02:0--5`正式完成。root=
`runs/outputs/pi05_v6_balanced_causal_condition_residual_correct400_noreplacement_seed7_method_macro0000_6b5f7a6_20260810`；
strict correct=`134/400`、correct80=`26/80`、breadth6，per-task按Spatial/Object/Goal/Long=
`0/5/48/34/0/35/11/1`。72/72 shards、400 rows、18 workers全部attempt1/exit0；wall/rollout window=
`867.152/616.138s`。18 generators生成400 fresh LoRAs、54 batches、max batch8、0 reuse/redundant forward，
Writer全部释放且source policy全复用；0 retry/OOM/nonfinite/forbidden reads，GPU结束后0MiB/P8。

与历史native v6 macro0的400-row strict pairing得到所有state/language/env seed/policy noise/video ordinal与
order/selection seed 0差异，success逐行也完全相同：gained/lost=`0/0`、共同成功/失败=`134/266`；每task
demos0--49各一次。400 cache entries的30,400 LoRA tensors、514,867,200 values逐tensor bit-exact；仅一条
共同成功episode晚1 env step终止，不改变success集合。因此该结果是当前family可信的exact closed-loop
identity baseline，不是性能提高。

当前操作顺序（覆盖上面的历史顺序）：

1. v6-prior whole-LoRA、ECP与Tangent三条连续单变量实验的formal/strict/机制诊断均已封存并释放GPU；
2. Tangent只解决了局部半径，没有解决共享decoder Jacobian把更新旋进expert方向的问题；strict同时降到
   `131`，所以禁止靠续训、扫权重或硬加大auxiliary掩盖首个失效接口；
3. audit teacher-quality已方向性失败，禁止CEFD、weight profile、换expert step或把gradient novelty当价值；
4. 一次性audit、v1和v2 blind-add均已退役；v2的mechanism/deployment/identity、formal0→25、macro10/25
   strict与same-task诊断都已封存，不能从旧Tangent/audit/v1/v2恢复活动执行；
5. 第39节RLS的CPU/profile/formal0→10与strict400均已seal；full400 lost15未过门，config/runtime已退役，
   不得fresh、resume25、补六臂或用correct80重解释；
6. Reward-Credit Program Cotangent设计、canonical实现与CPU seal已完成；GPU前必须clean push/frozen和
   live preflight，不能从RLS10或任何retired checkpoint续权重；
7. 唯一下一动作是fresh0→1 discarded profile；通过后另commit seal，才允许formal cycle0→1与立即
   strict400。cycle2只由macro1 support gate授权，不从任何retired checkpoint伪resume。

不得从下文自行跳到later stage，也不得从历史文档恢复已退役命令。

## 1. Fixed scientific contract

- 方法：one-shot Video-Conditioned Writer总路线；whole-LoRA/ECP/Tangent/CEFD、第37节v1、第38节v2
  blind-add和第39节RLS均已退役或否决。唯一active implementation是第39.5节Reward-Credit Program
  Cotangent；保留balanced-key frozen-v6 Program deployment图，只替换训练cotangent。
- 输入：exact task language + exactly one action-hidden raw teacher video。
- 视频是唯一dynamic value；无language-only LoRA bypass、expert-bank部署、multi-video/LoRA/checkpoint
  平均或融合。
- 输出：一套完整38-target public rank-16 LoRA；Writer在rollout前运行一次后释放。
- source policy、normalization、split、frame stride5、LIBERO preprocessing与paired evaluator固定。
- historical v6-fast macro400是当前active candidate唯一允许的load-only Writer初始化；不得加载任何退役
  optimizer/scheduler/sampler/RNG，也不得用task expert输出作为cotangent。
- train24 task-complete、每task每cycle一条correct video和同一LoRA、K4 random-reset rollouts、24-task等权；
  mixed task才从success+failure executed prefixes形成Nmc4 cotangent。六rank各4 tasks，CUDA-complete readiness
  后固定gather 24 correct cotangents与24+24 condition features，独立形成同一manual write，不all-reduce
  80MiB Program。
- 不恢复global norm attraction、whole-LoRA cosine、ECP completion、Tangent dynamic anchor、CEFD、progress
  reward、critic、SPSA、ratio或parameter-distance代理。correct cotangent只读train24 on-policy reward与真实
  executed observation/action；negative只作zero incremental condition constraint，不读取wrong-task expert或
  最大化negative action error。
- step2000 task experts仅作历史分析/privileged参照；当前训练、部署和held选择均不读取其输出。
- state为FP32 Program `M[256,320,256]`与training-only FP64 precision `Lambda[256,256]`；每reward cycle
  同化full48 rows。checkpoint二者分文件原子保存，deployment只读M；fresh0与exact-resume严格区分，RLS10
  的Program/precision不能加载。
- profile必须从fresh0→1且不留权重。formal cycle0→1启动前固定macro0 reference与唯一future cycle1 strict
  root；cycle1后只认canonical raw400聚合、same state/RNG/language/actual-video pairing及checkpoint identity。
  只有correct`>=140`、lost`<=6`、breadth`>=6`且gained>lost才允许cycle2。

## 2. Throughput-first runtime contract

“精度”指科学合同和真实closed-loop，不指底层逐元素复现。执行中遵守：

- 接受正常BF16 kernel、batch shape和reduction order的低位差异；不因`.001953125`级LoRA roundoff固定
  batch1、重复single forward或逐tensor扫描。
- 只门禁shape、finite、信息墙、cache完整性、明显跨样本污染、OOM、错误asset和resume语义。
- Writer生成从batch8起profile，逐步增加到16/32/更高，直到吞吐进入平台、allocator明显抖动或接近
  OOM；选择samples/s最高且能稳定完成真实最长video batch的点。目标是让A40 reserved memory保持高
  利用而非留出大块空闲，不用dummy tensor填充。
- LoRA cache保持template原生72 BF16 + 4 F32 tensors，单entry tensor data=`2,641,920` bytes；不强制
  扩宽FP32。每batch集中nonblocking D2H，只同步一次。
- action query DataLoader默认2 spawn workers、persistent workers、prefetch2；profile若显示GPU仍等待
  data，再实测4 workers/prefetch而不是猜测。
- reward rollout固定K4 policy batch4；四条persistent env lanes按active-lane compaction每次只做一个batched
  replan。mixed replay固定Nmc4且physical B2；历史同类图B2峰值reserved约40.34GB，B4容量尚未实证，先用B2
  不是为低位一致性降速。full profile若证明B4安全且samples/s更高，formal前只上调、不下调科学K/Nmc。
- policy activation checkpointing当前保持关闭；现有Writer checkpointing不覆盖OOM所在的frozen PI05
  Gemma MLP。只有B16/B10都无法形成高吞吐有效配置时，才把policy重算作为独立正式候选。
- 不做SHA/MD5，不重复全仓hash或历史artifact扫描。CPU全仓回归只在代码合同变化后运行一次。
- mechanism profile的production wall计K4 rollout、mixed replay/cotangent与full48 gather/solve/write；
  application closure、LoRA A/B、4-task fixed-action及zero-credit shared-motion是一次性verification，不进入
  formal热路径。fixed-action使用同observation/同noise的before/after推理，不读取target action。

保留FP32 RMSNorm/softmax/ROPE/image normalization和policy-effective reduction，除非profile证明它们是
显著瓶颈且降低精度不伤真实闭环；吞吐优先不是盲目改变模型数学。

## 3. CPU seal before GPU

GPU前一次性要求：

- `git diff --check`和聚焦/全仓CPU tests通过；
- 16种K4 success pattern覆盖LOO零和、mixed符号、全成/全败zero fast path；episode等权、Nmc4 keyed
  time/noise、physical batch不变性和ASPO old=current首epoch一阶等价通过；
- BF16 frozen decoder输出到FP32 LoRA leaf gradient再VJP到Program，positive `G`进入内部负梯度solver；
  full48 order、negative zero RHS、Program/precision/cursor exact-resume与profile→formal fail-close通过；
- historical 600-tensor Writer strict-load并全部冻结：`10,775,296` base parameters、0 trainable
  parameters；新增`20,971,520`个FP32 memory values由manual update拥有，但仍不注册为optimizer parameter；
- validation8 real-asset inspect和CLI prepare产生正确one-shot requests，部署expert-bank reads=`0`；
- no-video不读frames并返回template-A/zero-B identity；wrong/shuffle/reverse只改变允许的视频输入；
- native LoRA storage descriptor从checkpoint metadata贯穿run contract与cache write/load；resident policy的
  destination dtype由已验证的同一template决定，正常路径不发生额外转换，不在每次replan加dtype扫描；
- 2-worker prefetched sampler与serial、prefix+resume逐row一致；
- retired Tangent config只继承未改变推理图的retained evaluation throughput seal；ECP gradient、aux
  weights和resume profile没有被复用；v3 gradient与resume evidence曾从各自clean frozen lineage重新
  实证并原样封存，formal0→10完成后已切为non-pass并关闭runtime。任何status-only、旧family evidence或
  stale config仍fail-closed。
- retired teacher-audit CPU oracle曾覆盖physical B20的3次forward与B10+10的6次forward；正式artifact已
  证明480/480 queries、144 forwards、real7、8/8/8 negatives和0 update/rollout。其一次性feature tests随
  runtime退役；canonical contract owner继续保留。该证据不能解锁CEFD或新候选训练。
- current v2 residual CPU oracle覆盖zero memory identity、真实frame order、wrong-video exact-language、full48
  solve/application、negative-null、A/B response、memory-only exact-resume、六rank RNG、v8 deployment
  asset和strict paired row evidence；带LIBERO assets全仓`281 passed`。随后live mechanism profile已验证真实
  A40 feature rank、task-local motion、fixed-action传递、wall和显存；CPU与该单步机制证据都不能代替
  deployment graph吞吐、真实closed-loop或多macro累积裁决。

CPU门不要求batched Writer与single Writer逐元素相同，也不解释性能。

## 4. Inherited single-A40 deployment seal（部署图不变，已绑定）

第38节已经为Balanced-key frozen-v6 residual deployment graph独立完成双root seal；第39.5只替换训练
cotangent，部署forward、checkpoint Program payload与online generation完全不变，因此active config可重读并
绑定这份既有证据。它不替代Reward-Credit profile或closed-loop性能：

- retained 32-request/1093-frame panel的batch8/16/32吞吐为
  `.911427/.905107/.906432 LoRA/s`，batch8峰值reserved约`12.8GB`且稳定；
- retained validation8×state0 vertical smoke为8 requests/8 LoRAs/8 rows，0 retry/failure/OOM/
  nonfinite/forbidden reads，Writer释放、source policy复用且GPU自然释放；
- active contract会重读raw profile/results/cache manifest、commit、family、batch、release/reuse与禁止读取，
  不能只信status或人工抄录；
- Reward-Credit profile只实测新增K4/replay/cotangent/full48路径，不重复已封存且未改变的deployment batch sweep。


## 5. Historical six-A40 profile provenance（已完成，不恢复）

旧whole-LoRA、ECP与Tangent依次完成了physical B20、B16+4容量裁决和balanced B10+10 gradient/resume/
throughput seal。B20与B16+4都在真实A40容量上失败；B10+10保持logical B20、480/480 unique queries和
task mean不变，并达到约21秒/full24 macro、43.3/47.1GB allocated/reserved。对应gradient assembler、
resume assembler、600/41-tensor ownership、6-rank RNG及NUMA/NCCL证据均已封存。

这些profile只解释为何当时audit继承B10+10、workers2、六rank与default allocator；它们不授权重跑v3
gradient profile、恢复profile checkpoint、复用旧auxiliary weights或启动训练。详细ECP和Tangent证据保留在
下列5.1/5.2；任何活动GPU命令只取Section 7。

### 5.1 Historical ECP resume evidence

clean pushed/frozen`5fbcb27`在live空闲`gpu02:0--5`完成retry1：resumed链fresh0→1再exact-resume1→3，
contiguous链独立fresh0→3。两run contracts逐字相同，所有invocation exit0；各3 metrics、macro1/3
checkpoints和completion，0 OOM/nonfinite/clip。contiguous/resumed合计step wall=`61.368/64.450s`，
input wait=`.203/1.153s`；steady-state macro3约`20.018/19.698s`且input wait约`.0006s`，故不测试workers4。
峰值allocated/reserved=`43,265,769,984/47,118,811,136` bytes。

cursor、checkpoint contract、6-rank RNG、scheduler、AMP、559 frozen Writer tensors精确相等；41个
trainable Writer与82个Adam moment tensors通过逐tensor`atol=2e-4, rtol=2e-3`，scientific metrics最大
tolerance ratio=`.233773`。macro3 Writer maxabs/relative-L2=`4.6033e-5/1.06393e-5`，差异L2仅为两步
真实update的`1.023%`；Adam maxabs=`2.6865e-6`，其`.007719` relative L2由近零moment分母放大。

原aggregate gate把Writer的同一阈值误用于Adam，造成工程false negative。只读v2比较器现要求Writer
global relative L2`≤.002`，并对Adam各moment要求symmetric norm ratio`≥.99`、cosine`≥.999`；raw
maxabs/relative-L2只作诊断，同时保留上述逐tensor科学门和所有exact语义门。训练kernel、dtype、
reduction、B10、objective及artifacts均未改变，也没有重跑GPU追逐逐元素一致。原roots重新assemble
通过并原样写入当时ECP config，profile/formal均为`sealed_from_live_a40_resume_profile_evidence`。
这些证据只说明B10/deferred-NCCL/checkpoint比较器曾按合同工作；不能解锁v3 formal，也不构成
closed-loop性能结论。

### 5.2 Historical tangent-tube gradient and resume seal

clean pushed/frozen`2616773`已完成一次macro49 gradient/whole-macro profile：root=
`runs/outputs/pi05_v6_tangent_tube_gradient_profile_macro49_r6_lb20_mb10_2616773_20260809`，24 tasks、
480/480 queries、8/8/8 negatives、最长105帧，wall/input wait=`21.53076/.60603s`，peak
allocated/reserved=`43,353,948,672/47,112,519,680` bytes，0 OOM/nonfinite。macro0双臂tube与delta
24/24 exact zero；唯一projection/ranking weights=`.00686480847114155/.010514453175708578`已由
assembler原样写回v3 config。相对ECP whole-macro raw wall仅增约`5.4%`、显存约`36/18MiB`，不启用cache。

clean pushed strict后继`c1bdcae`已在`gpu01:0,1,2|4,5,7`完成fresh0→1、same-root
exact-resume1→3和independent contiguous0→3。原自动chain在fresh结束后的inter-phase selected-GPU
preflight发现设备不再满足expected-idle合同并由live gate fail-close；重新live检查通过后分别启动剩余
两段。三段科学invocation均exit0，不能把原chain的exit1误写成训练失败，也不能写成整条chain自然exit0。

resumed/contiguous总step wall=`62.34061/61.95860s`，input wait=`.09366/.13220s`，peak
allocated/reserved=`43,316,387,840/47,137,685,504` bytes，0 OOM/nonfinite。assembler核对dynamic
anchor为41 tensors/`3,714,304` parameters且optimizer/checkpoint/deployment ownership全false；macro1/3
cursor、6-rank RNG、scheduler/AMP、559 frozen tensors和checkpoint contract语义相等。macro3 trainable
Writer maxabs/relative-L2=`8.5067e-6/1.14428e-6`，82个Adam moments的最低direction/norm门通过；scientific
metrics最大tolerance ratio=`.67790`。evidence已原样写回v3 config，当时的Section 6由此解锁，profile runtime
按状态机关闭且checkpoint永久弃用。

三步只作工程与早期机制证据：macro2有21/24 tasks把`a_correct`推向1，macro3却变为0/24；macro3
`a_correct=.71744`，task median `|a-1|≈.2799`，correct/negative orthogonal-relative-anchor median=
`.03158/.03173`且仅`10/24`、`6/24`低于`.03`，orthogonal-to-direction median约`60.98/61.2`，
`gradient_norm_before_clip≈1.45294>1`。quadratic tube在anchor处一阶梯度为零，首步正交漂移后才产生
回锚力，这是当时formal0→10必须直接证伪的结构风险。该formal和strict已完成，最新裁决见第0节与
第6.4节；不能从本段恢复训练。

## 6. Historical Tangent formal and truthful evaluation（已完成）

### 6.1 Completed baseline and training

Tangent已从clean pushed frozen`b308941`按预注册完成fresh0→10并自然停止；historical macro0=`134`来自同一
video schedule，未用历史143替代paired baseline。训练root、完整机制metrics、cursor/RNG和资源证据见6.4。
本段只作provenance，禁止从macro10续训。

### 6.2 Completed checkpoint decision

macro10已完成完整paired correct400，correct80仅由相同rows派生；native-family验证和cross-family historical
transition均通过。结果`131`虽处旧条件区间，但breadth5、directional/completion门失败，因此预注册决策是
不续25、不补六臂。旧checkpoint cadence全部关闭，不能恢复成活动命令。

### 6.3 Reusable rule for what counts as real improvement

方法改善至少需要absolute和任务分布证据，而不是一个suite换手。完整报告：

- aggregate/per-suite/per-task successes；breadth；
- 相对macro0和历史best的retained/gained/lost/both-fail；
- 相邻checkpoint成功集合Jaccard、union/intersection和single-envelope gap；
- paired correct-vs-same/wrong/shuffled/reversed/no-video；
- same-task不同teacher video方差；
- Core→Procedure→effective BA→fixed-action的条件传递；
- LoRA norm、stable rank、top energy、B-column cosine和跨target energy作为机制参照。

LoRA健康但分数低仍是失败；LoRA近rank1但分数提高不能仅因“不像SFT”淘汰。

### 6.4 Tangent formal结果与关闭状态

第6.1--6.3的预注册流程已完整执行。formal root、strict root和historical transition分别为：

```text
runs/outputs/pi05_v6_tangent_tube_formal_r6_lb20_mb10_b308941_20260810
runs/outputs/pi05_v6_tangent_tube_correct400_noreplacement_seed7_method_macro0010_b308941_20260810
runs/outputs/pi05_v6_tangent_tube_macro0010_historical_baseline_transition_b308941_20260810
```

macro10 correct=`131`落在条件续训区间，但breadth仅5、directional两臂`0/24`且completion`0/24`，
所以明确不满足续25门。完整六臂只在`≥144`触发，本点不运行。current config只保留formal result并
fail-closed；所有Section 6 launch语句现为已执行provenance，不是活动命令。

## 7. Current diagnosis and single active variable

Balanced v2已经证明condition key、full48 solve、frozen decoder、Program→LoRA→action和部署吞吐成立；RLS又
证明feature-space保留可在内部成立但无法保住held success。当前最早接口因此是offline source-action
functional cotangent与真实on-policy occupancy/binary success错位。第39.5只把这一项替换为同一one-shot LoRA
上K4 binary LOO、success+failure executed-prefix Nmc4 Program cotangent；feature、decoder、Program、RLS
full48 transport、rank、scale和部署输入输出均不变。CPU seal已完成，当前须由一次discarded profile先裁决
credit/closure/吞吐，再由fresh cycle1 strict400裁决价值；不能从内部loss或几何直接宣告改善。

按以下顺序定位最早接口：

1. **evidence extraction**：correct/shuffle/reverse/wrong在Core前是否含任务和时序信息；
2. **Core/Procedure**：信息是否形成正确阶段/顺序，而非仅对低层异常敏感；
3. **compiler**：Procedure差异是否保留到policy-effective BA，是否被task-common方向压缩；
4. **factor heads/topology**：能量是否进入正确targets/rank coordinates而非同向高能量；
5. **functional/expert/ranking credit**：梯度是否对task可累积、是否由auxiliary绑到train expert流形；
6. **optimizer/task aggregation**：full24更新是否抵消并导致checkpoint轮换；
7. **closed-loop mismatch**：内部改善是否落到真实action和成功集合。

相应决策：

- absolute升、视频margin弱：下一变量只改counterfactual credit或Procedure temporal objective；
- margin升、absolute降：ranking伤害policy，不解释成“还没训够”；
- expert cosine升、held迁移降：减弱/重构train-expert流形约束，不转回online expert bank；
- compiler梯度健康但BA传递弱：修改compiler ownership/topology，不换video encoder；
- BA与action传递健康但任务继续换手：处理task aggregation/credit coexistence，不靠增rank/scale；
- same-task跨video方差成为最早限制：才设计固定K few-shot invariant aggregation；
- 所有下游目标与传递成立而上游无正确顺序语义：证据才允许解冻/重构Procedure或更早表示。

每个新设计必须明确继承旧方法已证明有效的组件、只改变哪个假设、预期影响哪些内部和closed-loop指标、
何种结果淘汰该假设。禁止东一榔头西一棒。

## 8. Evidence, Git, and retention

- canonical branch：`codex/bci-continuation`；正式root绑定包含该run contract的clean pushed commit。
- retired config：`configs/pi05_v6_condition_local_tangent_tube_writer_v3.json`；Tangent与teacher-audit均已
  formal non-pass/fail-closed，不能作为活动入口。
- retired v2 config及formal root由Git/artifacts保留；其最终strict=`134/140/139`和近正交same-task correction
  已封存，不再是活动入口。
- retired RLS config：`configs/pi05_v6_exact_anchored_reconciliation_program_residual_v3.json`；formal macro10
  strict=`140`且lost15后已fail closed，`f0c3f51/f28fc8b`只保留历史profile provenance。
- active config：`configs/pi05_v6_reward_credit_program_cotangent_v1.json`；status=
  `awaiting_live_a40_reward_credit_profile`，profile=`awaiting_live_a40_fresh0_to1_reward_profile`，formal=
  `blocked_until_live_reward_profile_passes_and_is_sealed`。当前0 Reward-Credit GPU/strict结果。
- current design authority：`docs/action_forecast_writer_video_expert_manifold_design.md`第39.5--39.6节；第38节
  只作unchanged deployment/key/macro0 provenance，第39节只作retired RLS根因证据。
- training/evaluation entries为`scripts/train_v6_prior_writer.py`与`scripts/evaluate_pi05.py`；retired
  `--mode teacher-audit`及其owners/tests已经删除。
- fresh/profile要求HEAD等于当前remote authority；同root exact-resume固定原frozen commit且只要求它仍为
  当前authority ancestor。不得把remote后续文档提交写入旧run contract或冒充原训练commit。
- formal保留config、command/env、GPU topology、checkpoint schema、raw rows、aggregate、completion和必要
  mechanism analysis；不提交checkpoints/cache/data/binaries。
- failed `30b2ccf` smoke及其`batch_equivalence.json`保留为BF16诊断；不resume、不作为性能证据。
- 历史design/findings/progress/Git/formal artifacts保存所有已验证经验；当前authority不重复堆叠旧
  “下一步”。清理只移除已确认obsolete的活动path/worktree/temporary output。

## 9. Stop or ask conditions

只有以下情形暂停并请求owner：设备边界/ownership不明确、需要干扰他人进程、quota无法容纳、必须改变
split/信息墙/test边界、需要删除含唯一证据或不明ownership的资产、或同一实质阻塞在安全替代方案后仍
无法推进。普通科学负结果、一次OOM、工程bug或候选不达标不构成沟通门，应按上述证据链自主继续。

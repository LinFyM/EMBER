# EMBER Findings

## Reward-Credit首次A40 profile：主链成立，固定probe门设计non-pass（2026-08-10）

- clean frozen`c4507e9f4872a88cccca37ca7956371bd8a18bd4`在空闲`gpu02:0--5`完成root=
  `runs/outputs/pi05_v6_reward_credit_program_cotangent_profile_full24_k4_nmc4_r6_b2_20260810`，natural exit0，
  但stored/raw recomputed `passed=false`且未保留checkpoint。24 tasks、96 rollouts、11 mixed/13 homogeneous、
  60 success/36 failure、4452 replay chunks、22124 executed steps；wall=`554.268s`，peak allocated/reserved=
  `16,336,873,984/19,417,530,368B`，0 OOM/nonfinite/watchdog/old/negative forward。
- reward主链证据为：11/11 mixed cotangent finite/nonzero，13/13 homogeneous cotangent exact zero且0 functional
  forward；Program cotangent RMS=`5.219e-7`，full48 rank48、condition=`105.66`，predicted negative/correct=
  `.017081`，application closure relative RMS=`0`；LoRA A/B response=`5.736e-6/6.001e-6`，aggregate fixed-action
  RMS=`.0006390`。这证明单步reward RHS→Program solve→完整LoRA→至少一个真实BF16 action probe已通，不能
  推出closed-loop会提高，也尚不能推出每个mixed task均action-effective。
- 唯一false gate要求固定ordinals`0/7/14/21`全部action-nonzero；实际仅0是mixed，7/14/21均4/0 homogeneous，
  direct cotangent按设计为零。`1/4`因此与“唯一有RHS的probe通过、三个zero-credit probes不变”完全一致。
  旧artifact必须保持正式non-pass，不能post-hoc放宽；正确修复是把同一严格门对准全部有reward RHS的tasks。
- profile v2穷举所有mixed tasks，每task复用K4真实首状态和原始首noise，before/after各一次batch4 forward；
  raw rows必须精确等于mixed ordinals、每task K4/2 forwards、覆盖四suite且每task LoRA A/B/action RMS均finite
  nonzero。homogeneous继续exact-zero direct credit；其shared solve motion旧profile为mixed RMS的`1.182%`，
  是多cycle drift风险而非本次action gate目标。
- live B2显存远低于旧functional图的40GB类比，因此B8是吞吐优先的直接候选：keyed Nmc4逻辑panel不变，
  functional invocations预计约`3648→928`。旧profile六rank本地rollout+credit wall约
  `542.5/140.6/434.8/547.9/418.7/332.6s`；冻结one-task-per-suite静态map后，旧B2 critical cost预测降至
  `410.1s`且mixed counts变为`2/2/1/2/2/2`。mapping固定跨cycle、RNG排除rank、full48按ordinal排序，故只
  改吞吐调度，不按结果改task权重。
- all-mixed/B8实现的全仓CPU回归=`338 passed in 37.69s`；compileall、27 JSON、Black、diff-check、旧artifact
  v2 fail和architecture guard均通过，0 hard violation/parallel family。该seal只授权新discarded profile。

## Reward-Credit Program Cotangent canonical实现与CPU seal（2026-08-10，profile前状态）

- RLS full400已把最早失败接口从feature-row retention定位到offline source-action cotangent与真实on-policy
  occupancy/binary success错位。第39.5只替换这一项：Balanced P256、frozen v6 decoder、single Program、
  full48 transport、one-shot信息墙和完整rank16 LoRA均不改；fresh `M0=0,Lambda0=I`，不继承RLS10。
- 同一task/video的一套LoRA做K4 policy-batch4 random-reset rollouts；binary LOO在mixed task给success/failure
  executed prefixes相反符号，全成/全败严格zero并跳过replay。mixed replay按episode等权和Nmc4 keyed CFM
  形成完整LoRA gradient，再VJP到Program；没有old/current重复forward、ratio、第二epoch、shared Adam、critic、
  progress、SPSA或negative policy forward。
- 历史B2峰值约40.34GB来自不同旧functional图；已被上节active Reward-Credit live B2显存证据覆盖。
  K4 batch4、四persistent env lanes、BF16、六rank×4 tasks、CUDA-complete rendezvous和两次fixed gather保持。
  homogeneous不拼大replay，hot path不做hash或逐tensor防御检查。
- canonical owner与retired path已收敛：旧ledger/single-lane/success-only、RLS gate/deployment/nonpass owners和
  tests删除；Reward数学、K4 rollout、training/profile与strict decision各有单一owner。CPU覆盖16种outcomes、
  ASPO一阶等价、Nmc4、BF16→FP32 gradient、Program/full48、checkpoint/cursor和fail-close；全仓=
  `336 passed in 59.12s`。
- 当前config=`configs/pi05_v6_reward_credit_program_cotangent_v1.json`仍profile awaiting、formal blocked；
  尚无Reward-Credit checkpoint或strict分数。首次profile和修正见上节；只有全新v2 artifact过门才允许formal。

## RLS full400正式否决feature-row retention充分性（2026-08-10）

- clean frozen`25bbd52`的formal fresh0→10 natural exit0；10 macros总step wall/input wait=
  `199.425195/.278241s`，peak allocated/reserved=`43,247,554,048/46,919,581,696B`，feature rank恒48，
  0 OOM/nonfinite/negative policy forward。macro10 checkpoint的Program、FP64 precision、480 assimilated rows
  与六rank RNG完整。
- macro10 correct400=`140/400`、breadth6、per-task=`2/3/47/35/0/34/19/0`。相对exact macro0 `134`
  gained/lost=`21/15`、retained success119、churn36；blind-v2 macro10相对同一macro0为`19/13`、retained121、
  churn32。RLS没有减少旧成功丢失，反而多gain2也多lost2；RLS与v2同分之间仍`17/17`换手。
- full400的Long1从11→19看似`+8`，内部却是gained/lost=`13/5`；v2→RLS在该task更是`9/10`。Goal3仍0、
  Long2从macro0唯一1个成功降到0。aggregate、suite net和breadth都掩盖了episode能力轮换，不是多task共同累积。
- correct80恰好为macro0→RLS `5/0`，full400却为`21/15`；small-panel gate在这里给出方向错误。由于correct
  未达144且retention门失败，不续25、不补六臂。
- internal current/blind=`.999980→.230340`、negative/correct=`.020028→.291493`，precision logged/final
  condition约`7510.5/8325.5`；历史精度明显抑制新写入，却未保留validation闭环成功。formal设置
  `reference_correct_rows=0`，所以old-row improved=`1`是空集合，不能解释retention。
- 被否定的是“train24离线feature-row anchored RLS足以保护held closed-loop成功集合”，不是整个
  video-to-LoRA图。最早失败接口转为offline pointwise source-action cotangent对真实on-policy occupancy和
  binary success的credit错位；下一候选不得继续调RLS ridge/window，也不得跳到few-shot或LoRA健康度优化。

权威transition：
`runs/outputs/pi05_v6_exact_anchored_reconciliation_macro0010_historical_baseline_transition_866cca9_20260810/analysis.json`。

## RLS新合同profile通过：短历史保留成立，长期获取冲突仍待strict裁决（2026-08-10）

- clean pushed/frozen`f28fc8b`在实时空闲`gpu02:0--5`完成独立fresh0→3；root=
  `runs/outputs/pi05_v6_exact_anchored_reconciliation_profile_fresh0to3_r6_lb20_mb10_f28fc8b_20260810`。
  raw profile=`100452B`、schema v3、stored/recomputed 17/17通过，completion passed、exit0、0 checkpoint/
  OOM/nonfinite/negative policy forward；六卡随后释放。logical B20/physical B10+10、BF16、六rank与workers2
  均未降低，peak allocated/reserved=`43,261,790,208/46,919,581,696B`。
- 核心证据与旧f0c科学payload完全相同：correct/cotangent=`.969147/.738140/.621680`、current/blind=
  `.999980/.784334/.640650`；macro2/3 old drift/blind=`.248611/.213872`且旧rows改善均100%。task-local
  correct全程24/24，null=`24/24,22/24,22/24`；closure最大`.001033`，A/B=`1.09591e-5/1.13877e-5`，
  fixed-action 4/4。首步raw ppm偏差仍原样存在，证明新run不是靠重跑改变数值；旧f0c artifact仍为16/18
  non-pass。
- production=`19.9974/20.7508/19.5182s`，ratios=`.947963/.983678/.925249`、mean=`.952297`。可靠结论是
  RLS没有实用吞吐代价，不宣称结构性加速；timing相对f0c/v2的差异主要是cold/cache jitter。
- 风险也很具体：aggregate current motion连续下降，precision condition升到`1724.84`；macro2的
  Spatial5/Object6与macro3的Object5/Object8构成两项null失败，最弱Object5 correct retention仅`.46942`、
  shuffled leakage`.40127`。这可能是长期“保留旧功能—获取当前更新”冲突，也可能仍足以改善closed-loop；
  三macro不能裁决。config因此只解锁identity fresh0→10，必须立即以预注册correct400的absolute、lost和
  breadth裁决，不能直接续25或写成视频因果改善。
- 启动链另封住两个执行缝隙：historical transition现只接受RLS macro10/25；formal correct macro10
  evaluator在创建output前从training run contract/manifest核对唯一registered root、commit、next_macro和
  metrics rows；完整checkpoint cursor/payload由后续既有adapter验证。
  这避免完整400 rollout落到错误目录，不改变评测panel或模型。

## RLS首次live profile：机制成立、原测量合同non-pass（2026-08-10）

- clean pushed/frozen`f0c3f51`在实时空闲`gpu02:0--5`完成fresh0→3；root=
  `runs/outputs/pi05_v6_exact_anchored_reconciliation_profile_fresh0to3_r6_lb20_mb10_f0c3f51_20260810`。
  72 task visits、1,440 action queries、三轮8/8/8 negatives，exit0且0 checkpoint/OOM/nonfinite/negative
  policy forward；peak allocated/reserved=`43,261,790,208/46,919,581,696B`。旧artifact按预注册合同为
  16/18、`passed=false`，不会被后续代码重写或冒充通过。
- 核心reconciliation门全部成立：macro2/3 old-row drift/blind=`.248611/.213872`，旧correct rows改善=
  `1.0/1.0`；三个macro current/blind=`.999980/.784334/.640650`，correct/cotangent=
  `.969147/.738140/.621680`，negative/correct=`.020028/.126612/.130233`。task-local correct均24/24，null=
  `24/24,22/24,22/24`；A/B、4/4 fixed-action、closure和state `0→48→96→144`均通过。RLS确实解决了
  短历史feature保留接口，但该证据不等于episode success、same-video鲁棒或视频时序因果。
- 首步hard gate只因RLS/blind RMS相差`3.80e-11`、ratio偏离1=`1.97e-5`而失败；condition=`106.0`下它与
  FP32低位误差同阶。为这点差扩大FP64大RHS、重复forward或降低batch会直接违反吞吐优先；CPU FP64 oracle
  已负责代数等价，GPU保留ratio诊断、finite和`current>=.5x`硬门即可。
- wall三步ratio=`1.175588/.984891/.928918`，均值=`1.029799`；production总计`65.17118s`，同host/同卡/
  同schedule的v2前三步`64.99104s`，差`.277%`。原`all(each fresh macro <= warm macro49*1.10)`把cold
  jitter当结构退化；新合同改为三步production算术均值，baseline和`1.10`阈值均不变。旧root仍non-pass，
  当时要求从新clean commit再做一次fresh profile、不能post-hoc seal；该要求现已由上节`f28fc8b`新root完成。

## v2正式终局、最早瓶颈与RLS选择（2026-08-10）

- clean frozen`abd8e0826e52758eda53b1963f8b12db92bf3748`的v2 formal root=
  `runs/outputs/pi05_v6_balanced_causal_condition_residual_formal_r6_lb20_mb10_abd8e08_20260810`完成25 macros；
  step wall累计/均值=`535.464796/21.418592s`、input wait累计=`2.208183s`，peak allocated/reserved=
  `43,247,029,760/46,917,484,544B`，0 OOM/nonfinite/negative policy forward。logical B20/physical B10+10
  与六卡吞吐合同成立，没有理由为低位数值降低batch、dtype或并行度。
- 同一paired schedule的macro0/10/25 strict correct=`134/140/139`，breadth均6；per-task分别
  `0/5/48/34/0/35/11/1`、`1/2/48/31/0/38/20/0`、`2/4/48/30/0/38/17/0`。macro0→10
  gained/lost=`19/13`、union/intersection=`153/121`；macro10→25=`12/13`、`152/127`。v2只比自身
  baseline净增6，未超过历史single best`143`，更未达到`>150`；能力集合继续换手，因此不续50、不补多臂、
  不扫P/lambda/eta。
- macro25科学结果内部为72/72 jobs、400 rows、18/18 workers attempt1/return0；但外层wrapper exit没有
  retained record，只能封存`scientific_status=complete`与
  `external_wrapper_exit_status=unobserved_missing_record`，不得事后合成exit0。
- macro10的effective delta/base median=`1.69498e-4`、stable rank=`1.000022`、top1 energy=`.999978`，
  Program residual RMS=`3.818e-6`。这说明“能量太小”仍是真实几何现象，但`+6`已在极小、近rank1写入下发生；
  单纯增幅或强制rank健康度既没有证据能保留旧能力，也不是当前最早可证伪接口。
- 同task 50条正确视频的raw correction consistency=`.141539--.142175`，几乎等于独立方向参考
  `1/sqrt(50)=.141421`；fixed macro10 all-target pair cosine=`-.001371--.003280`，action-target=
  `-.009579--.014302`。视频路径不是零，但同任务不同正确视频产生的修正近乎正交；这不证明真实时序因果，
  也不自动授权few-shot平均。结合success union`153`与跨checkpoint换手，当前首因更符合更新只拟合本macro、
  未显式保留历史functional约束。
- 因此第39节只改变训练侧reconciliation：部署仍是exact language + exactly one action-hidden video、balanced
  `phi256`、frozen v6、single `M[256,320,256]`和完整rank16 LoRA；RLS累计FP64 precision并把每批target锚定
  到`F M_prev + E`。它直接检验retention根因，不引入expert bank、language bypass、scale、few-shot、额外
  policy forward或新LoRA图。该选择当时尚无RLS GPU/profile/strict结果，不能把CPU数学成立写成性能改善；
  后续正式结果已由本文件顶部覆盖。

## Exact Anchored Reconciliation CPU与决策合同（2026-08-10）

- 首步在`Lambda0=I`时严格退化为旧blind ridge；streaming RLS与显式累计最小二乘的纯FP64 oracle误差约
  `6.1e-16/3.5e-14`。runtime有意保持Program/gain/RHS FP32，随机实际差在普通roundoff量级；不扩FP64大写入、
  不牺牲吞吐追求bit-level一致。zero cotangent时`M`不变，但precision与`assimilated_rows`正常同化，防止
  “没有当前梯度就遗忘本批condition”的状态语义漏洞。
- checkpoint把deployment-owned Program与training-only FP64 precision分文件保存；联合fresh/resume同时恢复
  Program、precision、rows和六rank RNG。precision在checkpoint边界做finite+positive-definite验证，但部署
  只安装Program，不读取precision值。该检查只发生在预声明checkpoint，不进入训练热路径。
- fresh0→10必须在结果出现前预注册唯一macro0与macro10 strict roots；macro0固定为bit-exact `6b5f7a6`
  400-row root。10→25前会从immutable queue/shards重聚合两份panel，核对family、commit、macro10 checkpoint、
  manifest、state/RNG/language/actual video identity；只有macro10 `correct>=140`、相对macro0 lost`<=6`且
  breadth`>=6`才放行。这样既避免结果后择优选root，也避免把文档门当成无执行力的建议。
- formal evaluator仅接受config预声明的macro10/25 checkpoint；paired analysis新增独立RLS-v3 family，同时
  保留v2历史family。artifact路径被限制在canonical `runs/outputs`，absolute和`..`逃逸继续fail closed。
- 下一科学动作仍只有fresh0→3 A40 profile：首步blind误差、旧row drift/improvement、当前motion、现有
  condition/null/action/closure与wall ratio全部通过后才允许formal0→10。macro25是macro10 strict支持门后的
  条件动作，不是因config列出25就预授权。

## v2 zero-memory macro0与历史native行为逐行同一（2026-08-10）

- 正式macro0不是只有aggregate同为`134/400`：新旧400个paired episodes的state、language、env/policy RNG、
  teacher video ordinal/order/selection seed全部0差异，success也逐行完全相同，gained/lost=`0/0`。因此新
  v8 adapter、balanced key构造和zero residual fusion在完整closed loop中没有引入隐藏基线漂移。进一步
  逐tensor直比30,400 LoRA tensors、514,867,200 values全部bit-exact；一条共同成功episode晚1 step终止
  是不影响成功集合的模拟运行微差，不值得降低吞吐追逐。
- per-task仍是`0/5/48/34/0/35/11/1`、breadth6；这保留了historical v6的强Object/Goal6与Spatial1/
  Goal3盲点。`134`低于eligible历史best143，不是进步；它的价值是把下一次macro10的任何逐row变化干净
  归因到非零Program memory，而不是family/evaluator切换。
- 18 generators以batch8在六卡同时把显存推到约38GiB/card并生成400套LoRA，随后18 retained source policy
  workers完成72 shards；总wall仅14.45min，较约16min历史预算更快。0 retry/OOM/nonfinite，说明吞吐优先
  拓扑稳定，无需为低位浮点差异降batch或增加防御性扫描。

## Frozen-worktree formal prepare的工程边界（2026-08-10）

- CPU-only prepare在GPU前失败不是负科研结果：deployment raw evidence、Writer checkpoint和formal panel都
  一致，唯一差异是frozen worktree的`runs`软链接使resolved artifact位于canonical仓库。旧verifier把
  “不在worktree物理目录”错误等同于“越出canonical outputs authority”。
- 正确安全边界不是禁止软链接，而是双重限定：词法路径必须精确位于`runs/outputs/...`，resolved target
  必须仍包含于该worktree解析出的canonical outputs root。这样既支持正式frozen checkout复用唯一retained
  artifacts，也拒绝absolute/`..`/伪前缀/nested-symlink和manifest跨root逃逸；不需要SHA/MD5或逐tensor扫描。
- `af7b101`修复后同一CPU-only prepare登记8×50 correct/no-replacement和exact-zero residual macro0并exit0，
  全仓`285 passed`，临时prepare root已清理。它只证明启动合同可构造；没有CUDA、rollout、LoRA cache或
  性能样本，不能替代strict400。

## v2 deployment seal通过后的科学边界（2026-08-10）

- 新v8 residual graph在公平32-request/1093-frame panel上，batch8/16/32吞吐仅相差约1%，且三者均稳定、
  显存余量约32.4GiB。batch8以`.911238 LoRA/s`略高于batch32`.906482`和batch16`.901898`，所以选8是
  measured-throughput裁决，不是追求位级相同、保守显存或主观偏好；新graph也没有借用historical v6 seal。
- validation8×state0 vertical smoke证明完整部署生命周期成立：每个视频一次生成完整LoRA、native cache、
  Writer释放、同一source policy复用、真实LIBERO闭环8/8 rows执行完成、成功`4/8`且无重试/runtime failure。双root assembler从raw
  profile/results/manifest重算通过，排除了“只生成LoRA但真实policy/env路径没走通”的执行缺口。
- `4/8` success的样本太小、只含correct/state0且不是paired400；它不能说明absolute已改善，也不能说明
  correct优于same/wrong/shuffled/reversed/no-video。当前证据只把首个未知接口从部署可执行性推进到真实
  closed-loop能力；下一决定必须来自同schedule macro0 strict400，而不是放大解释smoke。
- 机制profile和deployment均通过仍不解决独立block L2可能放大same-task示范噪声、多macro memory累积、
  task coexistence或checkpoint drift。先测macro0可把“frozen v6+新key的zero-memory部署图”与历史v6严格对齐；
  随后fresh0→10才隔离Program memory学习的净作用。这一顺序维持单变量证据链。

## Balanced DC--Causal v2机制通过与剩余科学边界（2026-08-10）

- profile结果本身过门后又发现一处与科学结论无关、但会污染执行顺序的证据链缺口：formal runtime曾只
  检查mechanism seal，v8 evaluation verifier也只读throughput profile而未读required vertical smoke。
  这会让“文档禁止提前训练”和“机器实际允许训练”分裂。当时先把formal硬阻塞到mechanism+deployment
  双seal，并恢复唯一双root verifier共同重读profile、validation8×state0 results和native cache manifest；
  修复不进入GPU热路径，也不把8-row smoke success当性能证据。
- clean frozen`5d93434`在与sealed baseline相同的`gpu01:0,1,2|4,5,7` panel完成唯一macro49 profile；
  functional loss与cotangent RMS仍精确为`.091801740/2.1920664e-6`，所以v1→v2差异没有混入action panel、
  source target或training recipe。13项原门全部通过，0 checkpoint/OOM/nonfinite/negative policy forward。
- feature rank保持48，regularized condition从v1`1315.329`降到`106.114`；correct retention从`.807966`
  升到`.968254`，negative/correct从`.264351`降到`.0218514`。task-local由24/24 correct、15/24 null变成
  24/24 correct+24/24 null；最差correct retention仍`.942261`，最差negative leakage仅`.048462`。
  这直接验证“DC与causal动态分块等能”修复了首个失效接口，而不是靠降低lambda或更激进逆矩阵补门。
- shuffled/reversed/wrong的paired feature cosine mean从`.985525/.956451/.906269`变成
  `.479565/.013732/.507178`；各臂最大为`.851083/.023307/.762135`，leakage最大仅
  `.048462/.032562/.033571`。reverse近零而非接近`-correct`，说明static anchor确实打破纯causal的
  正负共线；shuffle/wrong保留content相似但已有足够动态/语义分离，三类都8/8过`.25` null门。
- value-delta RMS减小约45%到`1.16318e-6`，但A/B response略升到`1.37744e-5/1.38187e-5`且4/4
  fixed-action=`.001210888`；更小memory write实现更高correct motion，符合v1病态条件数曾浪费能量在
  pair差分方向。Program→A/B→action传递没有因“健康几何”而变弱。
- production=`20.021842s`、ratio=`.949122`；同host/panel input wait=`.069295s`与baseline`.076318s`
  匹配，kernel仅`.436306s`。所以v1的`.326s`边缘吞吐non-pass既没有被事后改门，也通过v2的少同步、小
  projection和健康solve在公平panel上真实消失。
- 当前结论只到“condition key、显式kernel与单步policy-effective传递可用”。profile每task仍只看一条correct
  video，没有测same-task-other或多步memory累积；独立block L2会让任何非零dynamic占最终key一半能量，
  可能放大same-task示范噪声。它是后续task drift/video鲁棒性的明确风险，必须由deployment smoke、macro0、
  fresh0→10和严格correct/same/wrong/shuffled/reversed/no-video裁决，不能因13/13机制门宣称EMBER已改进。

## v1 mechanism non-pass与Balanced DC--Causal key裁决（2026-08-10）

- clean pushed/frozen`6903ee6`的唯一macro49 root=
  `runs/outputs/pi05_v6_condition_residual_mechanism_profile_macro49_r6_lb20_mb10_6903ee6_20260810`自然
  exit，0 OOM/nonfinite/negative policy forward且没有checkpoint。13门中10门通过：full48 rank48、
  correct motion/cotangent=`.807966`、24/24 correct retention、application closure relative RMS=`0`、
  A/B response RMS=`1.27385e-5/1.26956e-5`及4/4 suite fixed-action response=`.00121293`。所以显式
  kernel、frozen-v6 decoder与Program→完整LoRA→action路径已经真实工作。
- 唯一科学non-pass是feature geometry：regularized Gram condition=`1315.33`、aggregate negative/correct=
  `.264351>.25`、task-local null=`15/24<18/24`。shuffled/reversed/wrong的feature cosine mean=
  `.98552/.95645/.90627`，leakage mean=`.38347/.20054/.12898`，过门=`2/8,6/8,7/8`。全部9个失败
  rows cosine均`>=.97099`；pairwise ridge解析leakage对实测Pearson=`.99021`、MAE=`.06044`。这排除
  gather/order/sign/FP64 solve bug，定位到旧`[1,tau,cos,sin]` descriptor的DC块压倒时序块。
- 最难shuffled pair距离`.07777`，任何线性memory要区分`[g,0]`至少放大约`12.86x`。把lambda从`.01`
  降到约`.002`虽可能机械过门，却把差分方向放大从约`3.0x`推到`7.8x`并增加held视频噪声与task漂移；
  hard-null/SVD/reweight也不能创造顺序信息。因此不训练v1、不扫lambda/seed/P/threshold。
- production=`23.530704s`、ratio=`1.115458>1.10`按预注册保留non-pass；但只超允许上限`.326083s`，
  跨host input-wait差却为`.633711s`，去wait诊断ratio约`1.086`。所以吞吐证据不支持“稳定慢化”，也不
  允许事后改门或单独重跑。v2移除每condition GPU sort/mask同步并把profile-only bookkeeping/zero allocation
  移出production timer，不降B20/B10、dtype或并行度。
- 历史phase16 cache已给出直接修正依据：DC能量占`.98057`，而phase-centered sqrt-causal-prefix的
  correct/reversed/shuffled template cosine=`.96263/-.94287/-.04463`。纯causal在reverse时又可能接近
  `-correct`，同一线性M仍难拟合`[g,0]`；当前v2因此把video-DC static与causal dynamic分别fixed-JL到128、
  各自zero-L2后拼成P256。按历史proxy组合，reverse/shuffle cosine预计约`.0286/.4777`，孤立leakage约
  `.0003/.0061`；真实结论仍只取新macro49 profile。
- static block不是language bypass：它来自`frame_evidence-text_queries`，zero/no-video时为0；同frame-set
  reverse/shuffle又与correct共享static但RHS为`g/0`，纯static memory无法满足full48。frozen `S0(c)`继续
  提供原v6完整Core/Procedure。v2保持memory、full48、`.01` damping、step1、B20/B10+10和0 negative
  policy forward完全不变，只替换最早失败key。
- v1 config/code从active tree退役、Git/artifact保留；v2 schema/checkpoint fresh-incompatible。CPU同static/
  反dynamic两帧反例得到natural/reversed unit keys内积0；聚焦`52 passed`、带LIBERO assets全仓
  `281 passed in 21.34s`。这些不构成v2 GPU或closed-loop成绩。
- 独立只读接线审查未发现v2数学或热路径阻断；上游schedule owner已保证frame permutation不跨video，
  所以删除重复GPU sort/mask同步成立。新增单测进一步证明feature-level `frame_order`与物理重排evidence
  生成完全相同key；正式evaluation family同步升级为`v6_condition_residual_v2`，避免未来artifact自报为
  已退役v1。独立block L2可能把很小但非零的dynamic提高到半数key能量，仍是same-video稳定性的待实证风险。

## 第37节Counterfactual-Null Program Residual实现与CPU裁决（2026-08-10）

- 当前实现不是又一套fresh Writer：它strict load historical v6-fast macro400的600 tensors/
  `10,775,296` parameters并全部冻结，只新增fixed-seed、zero-preserving时序condition feature和一个
  `[256,320,256]` FP32 Program memory。memory从零开始，因此完整76 LoRA tensors在step0保持historical
  v6 identity；A/B都继续由原八个FactorHeads生成，没有B-only/static/language bypass。
- 每个macro仍是train24×logical B20、每task一条correct video。correct只读真实functional cotangent；
  wrong/shuffled/reversed按8/8/8轮换只产生feature和exact-zero RHS，0 negative policy forward。full48
  小Gram/Cholesky与condition operator用FP64，大cotangent/coefficient/memory write保持FP32；六rank只
  gather local4 correct cotangents与8 features，各自形成同一约80MiB write，不all-reduce memory。
- 当前profile门同时防止aggregate掩盖task drift：aggregate correct-motion/cotangent`≥.25`、negative/
  correct`≤.25`之外，还要求至少18/24 task-local correct retained、18/24 paired negative null。四个suite
  各取一个task做before/after同observation与同noise的fixed-action probe，要求4/4非零；8次inference只作
  verification、不读target action、排除在production wall外，也不冒充negative functional forward。
- throughput门直接比较正式生产部分（24-task functional work + full48 gather/solve/write）与sealed v6
  macro49 B20/B10+10 `21.095109596s`，ratio`≤1.10`；task-local/application/LoRA/action verification另计时。
  这是吞吐门，不要求BF16逐元素同一，也不靠kernel fraction或空余显存判断。
- application evidence复算actual `feature @ delta`并与memory write后observed motion比较，证明生产delta完整
  应用；它不是独立solver的第二份数学证明。solver/sign/order/scaling由CPU algebra oracle独立覆盖，避免
  把相关量误写成更强证据。
- checkpoint family只拥有单个Program memory、cursor和六rank RNG；base600与fixed projection不持久化，
  没有optimizer/scheduler/scaler。fresh/profile必须HEAD等于当前remote authority；exact-resume仍绑定原
  frozen commit且只要求它是当前authority ancestor，从而后续文档提交不会破坏合法同root resume。
- 新v8 residual deployment graph不能继承旧v6/Tangent吞吐seal。artifact gate要求同目录run contract、
  actual clean commit、A40、正确adapter/family和batch8/16/32 fixed panel；其它family的synthetic负向测试
  明确fail closed。
- mechanism artifact不能靠写入`passed=true`过门：seal会从保存的raw macro重算全部13项机制门，并核对
  initialization、source identity、train24 task/language/data schedule、Writer、objective、ownership、A40
  topology和NCCL runtime。formal结果状态也不能靠字符串宣告，必须绑定completion、50-row metrics及
  macro10/25/50三个memory-only manifests；部署只接受active authority lineage内的training commit。
- 一次性teacher-audit/effective-objective/flow-teacher owner和tests已删除，历史由Git/formal artifact保留。
  纯paired statistics从artifact/family analysis拆为单向依赖owner；architecture guard无hard violation。
  聚焦测试、compileall、JSON/diff-check与全仓`280 passed in 21.02s`通过。当前仍是**0新GPU、0训练、
  0 rollout、0 strict成绩**；这些数值不授权宣称方法有效。

## Matched Expert-Flow正式non-pass与structured residual选择（2026-08-10）

- clean frozen`e8e4728`的formal root=
  `runs/outputs/pi05_v6_expert_flow_teacher_audit_r6_lb20_mb10_e8e4728_20260810`自然exit0：24 tasks、
  suite 6×4、480/480 queries、8/8/8 negatives、144 PI05 forwards、0 update/rollout/OOM/nonfinite；
  wall/input wait=`39.698/.684s`，peak allocated/reserved=`43.419/47.133GB`，六卡自然释放。
- expert/macro0/tangent10 matched真实7维flow loss=`.098631330/.091801740/.091843160`。expert比macro0/
  tangent平均差约`7.44%/7.39%`，仅`2/24` tasks同时胜两臂、`0/4` suite means过门；删最差global39仍比
  macro0差约`6.07%`。这是方向性teacher failure，不是边缘门、单outlier或视频长度单调效应。
- CEFD gradient相对existing span的compiler/factor residual=`.686410/.838727`且finite；novelty成立，但
  teacher在同一监督度量整体更差。distillation loss又与direct expert success负相关，意味着CEFD会在更弱
  experts上施加更大修正。按预注册`authorize_cefd=false`，不profile weight、不训练、不事后换expert step。
- task-expert最后50步loss与audit expert loss的Pearson约`.897`，证明audit没有量错；direct expert success
  与expert/macro0共同task difficulty均相关。`658/1200`衡量自身闭环状态分布上的序列成功，而audit衡量
  固定示范state/action/noise/time的pointwise velocity MSE，两者不矛盾，也不能互相替代。
- 下一单变量选择Frozen-v6 Counterfactual-Null Condition-Kernel Program Residual。历史Condition-Kernel已
  证明显式Gram能消除condition credit公共旋转，但fresh decoder LoRA norm仅`.176`导致`46--49/400`；v6
  提供`134/143`高增益起点。新设计冻结v6 600 tensors，以zero memory在fused Program后加入video-keyed
  residual，correct写真实functional cotangent、当前counterfactual写zero-motion，不再经过shared Adam或
  expert auxiliary。相比只训factor final layer的post-Adam QP，它直接控制cross-condition function motion、
  保留counterfactual且不修改historical decoder，因此作为首选；QP只保留为本候选失败后的窄后备证据。

## Expert-Flow Teacher Audit CPU implementation seal（正式结果前历史状态，2026-08-10）

- Tangent负裁决把最早失效接口收窄到`LoRA cotangent -> shared decoder/update kernel -> condition-specific
  output motion`，但在增加CEFD前仍需知道task expert是否真是更好的policy-flow teacher、其监督是否超出
  existing functional/completion/ranking span。因此当前实现的是一次零更新诊断，不是新训练架构。
- canonical CLI新增唯一`teacher-audit` mode；`writer/flow_teacher.py`只拥有matched real-action PI05
  velocity/cotangent，`v6_prior_teacher_audit.py`只拥有full24聚合与两门裁决，`v6_prior_run_contract.py`是
  唯一launch-contract owner。training通过显式frozen callbacks调用audit，无反向import环、第二runner或
  部署路径；失败/通过后的退役触发已写入第36节。
- 每task logical B20按A40容量切为B10+10；每slice依次运行expert和tangent10 `no_grad`、macro0 student
  differentiable，三臂重放完全相同noise/time/offset，正式共6次PI05 forward/task。actual action width必须
  为7，四类loss只在`B10×50×7` real-action slice上转FP32，不扩宽BF16主干或牺牲吞吐。
- 四类gradient先rank内4-task等权mean，再一次stacked all-reduce/world6。existing span用FP32 Gram、CPU64
  pinv `rtol=1e-5`，记录eigenvalues/effective rank/coefficients/residual并拒绝超阈值负特征值；near-collinear
  oracle证明不会把`1e-4` FP32扰动误算成新方向。
- CPU oracle分别覆盖physical B20的3次forward与B10+10的6次forward，测试轨迹合计9次；正式runtime仍固定
  6次/task。三臂逐tensor matched randomness、真实7维FP32 loss、same-memory comparison、8/8/8 negatives、
  480/480 queries和0 optimizer/scheduler/update/rollout均已封存；当时全仓`284 passed in 33.47s`。本段
  是正式运行前的实现事实；teacher-quality与CEFD最终裁决只取上节。

## Tangent Tube formal non-pass与下一证据门（2026-08-10）

- clean pushed/frozen`b308941`完成formal fresh0→10；10 macros总step wall=`207.4436s`、input wait=
  `.2655s`、peak allocated/reserved=`43,316,440,064/47,112,519,680` bytes，0 OOM/nonfinite，只有
  macro3一次clip。B10+10、logical B20、六卡并行和吞吐合同均不需要降级。
- macro10 correct/negative relative-anchor tube中位=`.013900/.014079`且两臂`24/24` tasks过`.03`；
  但directional ratio中位=`108.926/126.883`、两臂`0/24`过`≤1`，completion error task median=
  `.252295`且`0/24`过`.05`。Tangent确实压住总半径，却没有让共享decoder实际运动沿expert方向；
  学到的是小而几乎全正交的更新。
- one-shot strict correct400=`131`、correct80=`27`、breadth5、per-task=`0/3/46/31/0/40/11/0`、
  per-suite=`3/77/40/11`。相对同schedule macro0=`134`严格paired gained/lost=`16/19`、churn35、
  net`-3`、`p=.735879`；相对ECP10=`133`也是net`-2`。prefix略升但full400下降，再次证明及时真实
  closed-loop不可由小panel或内部几何替代。
- 裁决是不续25、不补六臂、不扫tube/projection weight、LR或WD，config/runtime formal non-pass后
  fail-closed。这个实验淘汰当前soft-tube recipe/window，但completion从未成立，所以不能写成
  expert-component已经完整写入后仍无效。
- 第一失效接口是`q -> J^T q -> Adam P -> J' P J^T q`。提高tube权重只能缩放cotangent，不能修复
  shared update kernel的方向旋转；hard tangent又会引入部署expert/scalar route并重复已负裁决路线。
- 下一步先做matched no-update Expert-Flow Teacher Viability Audit，而不是直接加CEFD。它用同一train24
  B20/noise/time检查step2000 expert flow是否在至少18/24 tasks、3 suites优于macro0和tangent10，并检查
  CEFD gradient在compiler/factor相对existing gradient span的残差比例是否都`≥.25`。任一门失败就以低成本
  否决CEFD，避免为冗余监督增加一次昂贵PI05 forward。

## Condition-Local Tangent Tube resume profile与一阶滞后风险（2026-08-10）

- clean pushed/frozen`c1bdcae`在`gpu01:0,1,2|4,5,7`完成fresh0→1、same-root
  exact-resume1→3和independent contiguous0→3。inter-phase selected-GPU preflight曾发现设备不再满足
  expected-idle合同并fail-close；重新live检查通过后余下两段分别exit0。三个科学invocation均有效，原
  chain exit1不是训练失败。
- resumed/contiguous三步step wall=`62.34061/61.95860s`、input wait=`.09366/.13220s`、peak
  allocated/reserved=`43,316,387,840/47,137,685,504` bytes，0 OOM/nonfinite。在线双decoder继续只增加
  很小显存与约3--5% compute/wall，没有证据支持用12.68GB anchor cache换复杂度，也没有降低B10+10、
  logical B20、六卡并行或BF16吞吐。
- 两轨run contract完全相等，scientific metrics最大tolerance ratio=`.67790`。macro1/3 cursor、
  checkpoint contract、6-rank RNG、scheduler/AMP语义相等，559 frozen Writer tensors exact；macro3
  trainable Writer maxabs/relative-L2=`8.5067e-6/1.14428e-6`。82个Adam moments的最低cosine/
  symmetric norm ratio均远高于`.999/.99`。这证明exact-resume与并行数值连续，不证明方法性能。
- 早期动态揭示soft quadratic penalty的关键时序：macro1在anchor处tube loss和gradient均为0；一次更新后
  macro2有21/24 tasks的`a_correct`向1移动且component上升，但macro3相对macro1为0/24，aggregate
  `a_correct=.71744`、task median `|a-1|≈.2799`。macro3 correct/negative的
  orthogonal-relative-anchor median=`.03158/.03173`，仅`10/24`与`6/24`通过`.03`，active
  orthogonal-to-direction median约`60.98/61.2`；gradient norm`1.45294`超过clip norm1。
- 第一性原理上，二次tube在原点没有一阶恢复力，首步functional/completion/ranking可以先造成正交位移，
  偏离后才出现随距离增大的回锚梯度。这是当前recipe的真实结构风险，但三步、轮换video/task panel和
  pre-update rows不足以判断macro10能否回锚或closed-loop怎样变化。连贯下一证伪仍是原recipe fresh0→10
  后立即strict correct400；若macro10仍不满足两臂tube门，即使correct为`130--134`也停止，不扫权重。
  只有裁决checkpoint同时满足median `|a_correct-1|≤.05`和tube门却仍不超macro0，才可干净证伪
  expert-component completion；否则只淘汰当前recipe/训练窗口。

## 2026-08-09 Tangent Tube六卡gradient/throughput seal

- clean pushed/frozen`2616773`在live比较双节点、核对`/data1` quota后只使用空闲
  `gpu01:0,1,2|4,5,7`；3+3 NUMA、physical/local rank、deferred NCCL与
  `NCCL_P2P_DISABLE=1`均由artifact复验。24 tasks、480/480 unique B20 queries、8/8/8 negatives、
  最长105帧，0 OOM/nonfinite，结束后六卡回到14MiB。
- live macro0上correct/negative的student与same-input frozen anchor在24/24 tasks完全一致，全部
  delta/tube/orthogonal指标exact zero。unweighted projection compiler/factor gradient=
  `.402617/1.670787`，与ECP的`.401533/1.667382`只差约`.27%/.20%`；ranking保持
  `.262866/.269814`。这关闭了“真实BF16路径破坏macro0 gradient identity”的风险。
- 预注册`.25`规则唯一给出projection/ranking weights=
  `.00686480847114155/.010514453175708578`；应用后compiler均`.25`，factor仅
  `.108659/.026876`。不继承旧权重，也不做weight sweep。
- whole-macro wall/input wait=`21.53076/.60603s`，相对ECP同图`20.42496/.17998s`的raw wall只增加
  `5.4%`，扣除input wait后约`3.4%`；peak allocated/reserved只增加约`36/18MiB`到
  `43,353,948,672/47,112,519,680` bytes。因此在线双decoder满足吞吐边界，cache没有end-to-end收益证据。


## 2026-08-09 Condition-Local Dynamic Expert Tangent Tube CPU封存

- ECP的正确臂expert component确实上升，但大量非expert effective方向同时漂移；因此第35节只改变
  一个变量：以historical v6对**同一language、同一实际video和同一frame order**产生的完整LoRA为
  condition-local原点，保留沿step2000 task expert的增量，只惩罚正交增量。correct和当前轮换negative
  各有自己的baseline，避免共享ranking更新通过破坏negative其他方向制造虚假margin。
- 三状态low-rank objective先跨全部38 targets求和，tube内使用expert合同保证非零的exact
  `D=sum||E||²`：`d=<G-G0,E>/D`，`R_perp=max(0,||G-G0||²-<G-G0,E>²/D)`；原ECP
  coefficient仍保留`D+epsilon`以维持初始gradient identity。两臂tube取算术均值，因此每臂梯度各为
  standalone的一半；不再声称等价于到某个单一dense target的平方距离。
- runtime只在六rank Writer同步后、任何resume load前冻结复制compiler+factor heads：41 tensors、
  `3,714,304` parameters。student与anchor对每个condition消费同一个memories对象；anchor不进optimizer、
  checkpoint或deployment。checkpoint仍保存完整600-tensor student以保留证据，但resume/evaluator均先加载
  immutable historical warm-start，再只恢复compiler/factor heads，伪造的frozen/template checkpoint值
  无法覆盖部署图。
- CPU oracle覆盖dense/low-rank、跨target全局投影、三状态独立gauge、macro0零tube、parallel/orthogonal
  perturbation、双臂mean、output-gradient chain rule、same-memory与所有权。冻结eval decoder与train-mode
  student在CPU同输入的最大普通kernel差约`4.77e-7`，只按`allclose`验证，没有为逐bit一致改变模式或
  增加重复forward。退役旧smoke执行路径后全仓`276 passed in 28.74s`，compileall和
  `git diff --check`通过；本阶段未启动GPU。
- 新方法使用独立`tangent_tube_v3` family；within-family curve仍只接收0/10/25/50，generic historical
  transition保留legacy+ECP并新增legacy+tangent 10/25/50，拒绝ECP+tangent混合；macro100必须等显式
  config续训授权后再开放。首次strict
  correct`≥144`即跑六臂，若不同checkpoint达到`≥151`再对goal winner重跑，不能只看tube或LoRA几何。

## 2026-08-09 ECP formal闭环负裁决

- clean pushed/frozen`450e688`的formal root=
  `runs/outputs/pi05_v6_ecp_formal_r6_lb20_mb10_450e688_20260809`完成fresh0→10与同root
  exact-resume10→25；25 metrics、macro10/25 checkpoints、完整六rank RNG/optimizer/scheduler/
  sampler和completion均在，0 OOM/nonfinite/clip。两段mean step wall=`20.447/20.631s`，
  B10+10保持吞吐优先的A40高显存利用。
- macro10 strict root=`runs/outputs/pi05_v6_ecp_correct400_noreplacement_seed7_method_macro0010_450e688_20260809`；
  72/72 shards、400 rows、18 workers和400 LoRAs全部成功，得分`133/400`、breadth6、per-task=
  `1/2/45/28/0/38/19/0`。对同video/state/RNG macro0=`134`的gained/lost=`22/23`、
  net=`-1`；Object净`-9`而Long净`+7`，是能力换手而非共同累积。
- macro25 strict root=`runs/outputs/pi05_v6_ecp_correct400_noreplacement_seed7_method_macro0025_450e688_20260809`；
  exit0、72/72、400 rows、18/18 return0、400 generated LoRAs、54 batches、batch上限8、0 retry/reuse/
  redundant forward。得分`120/400`、breadth6、per-task=`0/1/43/27/0/33/15/1`；suite=
  `1/70/33/16`，top3 share=`.85833`。对macro0精确paired gained/lost=`13/27`、net=`-14`、
  `p=.038477`；suite net=`-4/-12/-2/+4`。macro10→25也是`18/31`、net=`-13`，四suite
  全净下降；correct80恰好仍为`28`，再次证明80-row prefix会隐藏full400退化。
- 实现与优化确实生效：macro1→25的`a_correct=.736184→.884127`，23/24 tasks向1移动；
  component=`3.06189→3.67225`且24/24 tasks上升。但macro10→25的expert-orthogonal norm=
  `151.303→159.774`（`+8.471`），而component只`+0.228`。因而这不是“训练没动”或
  “权重不够”；它证伪了无约束共享参数下expert component是held共同改善的充分条件。
- negative同样漂移：macro1→10的counterfactual component只`+0.25062`，expert-orthogonal norm却
  `+8.26660`。因此第35节对correct和当前轮换negative各使用same-input frozen-v6 baseline；
  否则ranking可能靠破坏wrong/order LoRA其他方向伪造video margin。两臂tube使用condition mean而不
  翻倍权重，仍只是一个“condition-local orthogonal drift”变量。
- 按预注册门，ECP不续50/100、不扫aux weight、不补六臂。下一个有证据的单变量是
  以同一language+correct-video的frozen v6输出为dynamic baseline，把所需expert分量与有害正交漂移
  分开。这不能退化成parameter weight decay、static/language bypass、B-only residual或部署专家库。

## 2026-08-09 ECP resume门与早期方向证据

- clean frozen`fea3f40`在同一空闲`gpu01:0,1,2|4,5,7`完成resumed fresh0→1+exact-resume1→3和独立
  contiguous0→3。三次tmux launcher均exit0；两root各3 metrics、macro1/3 checkpoints和completion，
  0 OOM/nonfinite。三步step wall=`62.369/61.017s`，input wait=`.176/.231s`，peak reserved=
  `47,118,811,136` bytes；B10+10仍接近填满A40且没有I/O等待瓶颈，不扫workers或更小microbatch。
- 官方assembler确认run contracts、cursor、六rank RNG、scheduler/AMP和checkpoint语义一致。macro3 Writer
  maxabs/relative-L2=`1.304e-5/4.845e-6`，scientific metric最大tolerance ratio=`.4290`，Adam moment的
  scale/direction门均通过。这是正常parallel reduction低位差异；没有降低并行度或重跑GPU追逐逐元素一致。
- macro1→3的mean `a_correct=.736184→.754337`、absolute expert component=`3.06189→3.13618`、generated
  norm=`140.973→142.359`；23/24 tasks向1移动、23/24 component上升，未重演旧whole-LoRA objective的
  系统径向塌缩。它只证明ECP update短程方向符合构造，不证明held closed-loop会提高。
- 下一裁决只fresh训练到macro10并跑correct400。为吞吐不先重复运行行为相同的v2 macro0；历史macro0
  immutable rows按legacy native family验证，再与ECP v2 macro10做显式cross-family逐row transition。若
  `a`继续改善但macro10仍低于门，证伪的是expert-component共同迁移假设，不能靠加权或长训挽救。
- 新transition入口不接受任意mixed curve：只允许legacy macro0与ECP macro10 correct400；分别验证native
  adapter/episode family、clean sealed root和8×50 panel，再核对共同source/tokenizer/normalization、policy/
  RNG/video schedule及实际400行。只排除不同clean Git、family config identity和worktree-local manifest
  路径，artifact明确声明`not_checkpoint_curve`。全仓`262 passed`。

## 2026-08-09 ECP实现边界与首轮可证伪信号

- ECP不是“让整套LoRA更像SFT LoRA”：它只约束generated effective BA在task expert方向上的最小二乘
  coefficient。给generated update增加任意expert-orthogonal能量不会改变projection loss；因此旧路线
  94% loss降幅来自径向收缩的机制已被结构性移除。
- correct与reversed/shuffled/wrong都投影到exact-language task的同一个expert；wrong video来源task的
  expert从不进入objective。ranking只比较`a_correct-a_negative`并用temperature-scaled softplus平滑退火，
  same-task-other仍属于positive分布。
- v2 schema隔离是科学边界而非命名整理：旧whole-LoRA aux weights、optimizer/resume和trained checkpoint
  无法被ECP live loader接受，避免把`134→127→105→123`路线的状态伪装成新方法证据；历史strict结果仍可
  只读比较。
- 首轮真正裁决仍是closed-loop。若profile后`a_correct`大多数task向1、norm不塌缩，但macro10/25仍不超过
  134或只是任务换手，说明task-expert component本身不是held成功的共同方向，不能靠加权或长训挽救。
- clean frozen`de28157`的profile保持旧positive gradient近乎不变：compiler只差`1.86e-9`、factor完全
  相同，证明这次实证只换aux objective而没有暗改functional图。ECP projection/ranking未加权gradient在
  compiler/factor为`.40153/1.66738`和`.26287/.26981`，远大于旧cosine ranking的`.00967/.01475`；
  `.25`预算因此自动给出小得多的weights=`.00688335/.01051445`。这是coefficient对小expert norm更高的
  解析灵敏度，不是训练不稳定，也不能用旧`.28570` ranking weight。
- 初始化`a_correct` mean/median/min/max=`.73453/.73979/.48660/.96277`，24/24都低于1，所以projection
  对所有task的直接目标方向一致；margin mean=`.10324`且23/24为正。reversed/shuffled/wrong mean margin=
  `.09505/.05050/.16417`，唯一反向是object task2的shuffled `-.00968`，首轮应重点观察顺序辨识是否共同
  改善而不是只看wrong-video分离。
- 38 targets的absolute numerator fraction top1/top4 median=`.18084/.52988`，未出现单target垄断；
  generated/expert norm mean仍为`140.52/4.182`，这是ECP有意保留的expert-orthogonal历史v6能力，不应在
  profile阶段再加norm约束。上述均是机制起点，不是性能成绩。

## 2026-08-09 v6-prior formal四点裁决与Expert-Component Projection根因

- current-schedule strict correct400在macro0/10/25/50为`134/127/105/123`，correct80却为
  `26/26/24/27`；小panel会把macro50误示为提升，formal选择必须继续使用完整400。
- 0→10 gained/lost=`19/26`，0→25=`19/48`（McNemar `p=.000522`），0→50=`20/31`。四点
  union/intersection/per-task envelope=`172/77/147`；envelope仍低于150且不能作checkpoint融合。
- expert loss从约`1.7943→1.7191`，但约`94.2%`降幅来自log-norm径向项；generated norm
  `140.97→107.00`，correct expert cosine只`.02194→.02630`。更关键的绝对投影系数
  `a=<G,E>/||E||²`均值`.736→.662`且23/24 tasks下降，说明优化没有补足expert有效分量。
- paired held state0的macro50相对macro0 norm ratio/cosine/radial coefficient/orth residual/base/delta/base
  均值=`.7180/.9755/.7007/.1551/.3373`。训练主要沿原LoRA径向收缩，object任务损失明显；goal6虽增加
  6 states，但不足以形成共同提升。
- 结论只淘汰whole-LoRA direction+norm objective，不淘汰v6 representation或task expert作为局部方向。
  下一单变量是objective-only Expert-Component Projection：correct只把`a`推向1，negative做bounded
  `a_correct-a_negative` ranking，不约束global norm或expert-orthogonal v6动态分量。

## 2026-08-09 balanced B10 gradient seal与macro0几何诊断

- clean frozen `9c814ff`在同一空闲`gpu01:0,1,2,4,5,7` 3+3 NUMA拓扑完成macro49。artifact assembler
  独立复验24 tasks、480/480 unique queries、最长105帧、8/8/8 counterfactual、default allocator和完整
  Git/config/HDF5 provenance。wall=`21.0951s`，input wait=`.0763s`（`.36%`），peak allocated/reserved=
  `40.3318/43.8594GiB`，0 OOM/nonfinite；因此不扫workers4，也不为预防性余量降到更小microbatch。
- positive compiler/factor梯度=`.0110556/.105556`，expert=`.330800/.663721`，ranking=
  `.00967394/.0147533`。逐aux逐block `.25`规则唯一给出expert/ranking weight=
  `.008355172068998324/.28570466890490887`；加权后compiler各`.25`，factor仅`.05254/.03993`。auxiliary
  修正不会在初始化时压过真实functional signal。
- macro0的系统性矛盾很明确：generated correct effective norm mean=`140.52`，expert mean=`4.182`，
  比值mean约`33.72x`；correct→expert cosine mean仅`.02196`。wrong margin mean=`.00225`，而reversed/
  shuffled仅`.000832/.000634`，各仍有一个负margin。也就是说历史143 prior保留闭环能力，但LoRA能量/
  方向离task-local SFT流形很远，真实时序辨识尤其弱；当前实验正以受控小梯度修正这两个接口。
- 这些内部数值不等于性能结论。下一步先验证三宏步训练、optimizer state、resume和显存平台，再及时跑
  同schedule macro0/10/25/50 closed-loop；若energy/cosine改善但性能不升，必须判为expert surrogate错位，
  不能继续为“健康度”无限优化。

## 2026-08-09 physical B16仍超过A40容量，当前转为balanced B10+10

- clean pushed/frozen `eddba96`保持logical B20、完整20-query keyed randomness和两次policy forward，
  只把physical execution改为B16+4。live比较两节点后选择空闲`gpu01:0,1,2,4,5,7`；run contract实证
  local→physical=`0,1,2,4,5,7`、NUMA=`0,0,0,1,1,1`、deferred NCCL和default allocator均正确，
  他人占用的`gpu01:3`及`gpu02:6/7`没有触碰。
- 首个非持久SSH后台launcher只产生contract/invocation便exit0，没有start/gradient/completion；该root是
  无效进程托管证据，不能复用或解释容量。改用项目长期合同的tmux和全新root后完整进入start，并在六个
  ranks的第一条functional eager-attention一致OOM：请求`254MiB`，allocated=`42.49GiB`、
  reserved-unallocated=`1.25GiB`、free=`235.31MiB`。这证明B16没有稳定A40容量余量，不存在可比较的
  B16 whole-step吞吐。
- 先前`expandable_segments`在physical B20已证明减少碎片仍不能容纳active graph，因此不为B16再做
  allocator retry。按预声明决策直接进入balanced B10+10；logical B20、20条query及draw、task mean、
  train24和objective全部不变。policy activation checkpointing仍是只有B10也失败时才打开的较慢候选。

## 2026-08-09 physical B20 A40 OOM与logical-B20微批裁决

- clean frozen`a17805c`在当时live空闲`gpu01:0,1,2,4,5,7`完成两次六卡工程尝试。默认allocator在第一条
  PI05 functional B20的Gemma MLP申请`606MiB`时OOM：PyTorch allocated=`42.29GiB`、reserved-unallocated=
  `1.29GiB`、free=`395.31MiB`。一次有证据的`expandable_segments:True`重试把碎片降到约`157MiB`，但
  allocated=`43.43GiB`、free=`389.31MiB`，仍在同一MLP申请`606MiB`失败。故根因是active capacity，
  不是碎片；继续allocator/env重试无科学价值。
- OOM发生在frozen PI05 policy MLP，不在Writer。当前`writer.activation_checkpointing=true`只覆盖视频/
  文本Writer encoder；policy checkpointing被明确关闭。启用后者需要对整段Transformer重算，可能降低
  吞吐，且不是解决logical batch的必要条件，因此不作为首选。
- scientific estimator必须保持每task 20条跨episode query及等权mean。现有functional helper已经支持
  physical slices与FP32 LoRA leaf-gradient accumulation；缺口只是确保每个slice看到同一logical B20的
  随机draw集合。当前以optimization seed、task、visit以及完整有序20条demo/frame identity生成局部seed，
  通过固定SplitMix64整数mix而非SHA/MD5；每个
  physical slice重新生成完整20个独立Beta(1.5,1) time和Gaussian noise后取对应slice。因而B16+4只改变
  峰值显存与前向次数，不改变query集合、随机分布、loss权重或train24×20=`480/480`合同。
- 吞吐判断不能只看“forward越大越快”或“越小越安全”。B16+4与B10+10都只有两次policy forward；随后
  live证据已证明B16本身无法完成第一条functional forward，所以不存在B16/B10吞吐A/B。当前只运行
  balanced B10+10；若它成功，直接以其完整macro wall/input wait/peak作为A40可行点。所有容量失败root
  均无gradient/completion，不是方法负结果，也不能用于选择auxiliary weight。

## 2026-08-09 六卡profile证据门的第一性原理结论（CPU实现）

- gradient norm只有在它确实来自同一clean pushed canonical config、train24全覆盖、精确video/negative
  schedule和六卡拓扑时才能选择auxiliary weight；否则“两个block均不超过positive的`.25`”只是可伪造
  数字。新assembler因此从raw retained artifacts重算panel、norm、weight和applied fraction，并回查
  frozen manifest、HDF5 path/bytes与demo frame metadata；不相信status、stale config或人工摘要。
- exact resume的科学要求不是逐bit相同，而是同一cursor/contract、完整6-rank RNG、scheduler/AMP、
  Writer trainable state和Adam moments在预注册容差内等价。只比loss会遗漏optimizer二阶矩损坏，只比
  Writer会遗漏下一步更新轨迹；当前只读comparator同时封住两层，并接受正常parallel reduction roundoff。
- 吞吐证据应回答“GPU在等数据还是在算、显存是否逼近有效边界”，不需要在每个video/policy/backward
  阶段插入同步。retained path只记录已有宏步同步下的step wall、input wait和peak VRAM；若首个六卡
  gradient显示data wait显著才实测更高workers，若显存/计算证据指向checkpointing才做单变量候选。
  这保持了owner要求的吞吐优先，也避免为profile精度拖慢真正训练。
- 这些是工程/谱系门，不是方法有效性证据。当前仍无v6-prior新strict分数；`143/400`历史最好和
  `>150/400`目标均未改变，最终仍由及时paired correct400与五/六臂闭环裁决。

## 2026-08-09 v6-prior单卡吞吐与纵向链路的真实结论

- clean pushed/frozen `ded0c80`在live比较两节点并核对quota后只使用空闲`gpu02:0`。fixed panel为
  完全相同的32 requests、1093 sampled frames、最长67 frames；batch8/16/32的forward分组分别为
  `8×4/16×2/32×1`，吞吐`.911427/.905107/.906432 LoRA/s`，repeat变动低于1%且三者都有约
  34.85GB headroom。实测说明当前视频编码/固定panel工作量主导，增大LoRA forward分组没有吞吐收益；
  因此选8是吞吐结论，不是精度或显存保守结论。
- fresh validation8×state0 correct smoke用batch8一次生成8套LoRA，generation wall=`10.597s`、
  peak allocated/reserved=`11,651,564,544/12,811,501,568` bytes。Writer释放后allocated/reserved降为
  `9,370,872,832/9,628,024,832`，同一source policy未reload并完成8 rollouts。
- vertical总wall=`325.540s`，model load=`111.469s`、rollout execution window=`196.816s`；8 rows、
  single attempt、0 retry/failure/OOM/nonfinite/forbidden reads，cache每entry为72 BF16+4 F32、
  `2,641,920` bytes。退出后物理卡0MiB/0%，他人GPU未受影响。
- `4/8` success只是小样本execution信息，绝不能写成新性能或与历史`143/400`比较。真正科研信息是
  canonical video→LoRA→cache→release→policy闭环链路成立，六卡gradient profile现可开始；方法优劣仍
  必须由后续paired correct400与五/六臂裁决。

## 2026-08-09 吞吐优先纠偏与连续科研裁决框架

- Owner明确覆盖此前batch1决定：不得为了底层微小BF16/kernel差异降低效率。`30b2ccf`的
  direct-repeat=`0`、batch8 max-abs=`.001953125`、mean约`4.70e-5`证明没有随机性、padding或串样，
  但不构成把single-forward低位语义提升为科研变量的理由。逐episode direct重跑、`1e-5`逐tensor门和
  canonical batch1现已撤回；该失败root只保留诊断，不作为性能证据。
- 吞吐修正保持科学图不变：one-shot、video-only dynamic value、历史v6 macro400 load-only、冻结
  encoder/Core/transition/Procedure、只训练compiler+factor、train24/B20、positive functional +
  effective-expert + bounded ranking和strict paired evaluator均未改变。
- evaluator改为在同一32-request longest-first panel/同一总帧数上，从稳定且有显存余量的候选中选择
  实测吞吐最高batch，且不重复Writer forward；这避免大batch因同时加入更多短视频而获得混杂优势。LoRA cache保持template原生72 BF16 +
  4 F32，单entry tensor bytes=`2,641,920`而非强制FP32的`5,148,672`；batch D2H只同步一次。
  functional单physical batch绕过76个FP32 accumulation buffers；correct effective alignment只算一次；
  task metrics/gradient norms批量host transfer；action loader使用2个spawn persistent workers/prefetch2。
- Writer/video hot path还合并了offset、frame ordinal/order、task-span/condition ownership等重复host barrier，
  PI05 formal functional loss绕过只供日志使用的两次host sync。单卡profile和vertical evaluator均在模型
  load/worker spawn前拒绝忙卡和非A40，普通evaluator同时强制owner六卡上限；profile单卡worker再次live
  preflight并核对checkout，profile seal只能从真实artifact重建。
- 真实validation8×4-state CPU prepare得到32 requests、historical 600 tensors/12,064,064 values、
  deployment expert-bank reads=0和72 BF16 + 4 F32 cache descriptor；其后的live结论由本文件顶部取代。
  目前仍没有新训练或formal strict成绩。
- 后续任何实验按`docs/active_session_handoff.md`的统一谱系解释：与最邻近旧架构、历史143、逐task
  gained/lost、checkpoint churn、五臂和内部传递共同比较；只修改最早失效接口。负结果只淘汰实际检验
  的假设，未充分证明无效的结构不随单点结果丢弃。最终目标只有EMBER closed-loop性能，LoRA健康度和
  视频特异性只是机制参照。

## 2026-08-09 v6-prior batch复现失败的根因（batch1裁决已被上节撤回）

- `30b2ccf`首次A40 warm-start smoke在cache前被预注册`1e-5`门拦截；失败root保留0 cache、0 rollout，
  GPU自然释放，因此没有污染任何性能结论。
- 1,287,168个LoRA参数上，single-direct连续两次逐元素完全相同；同一样本复制batch8与8个异构样本
  batch8相对single-direct的max-abs都为`.001953125`，mean分别`4.703e-5/4.700e-5`。两种batch的量级
  几乎相同，排除跨样本串扰、可变长度padding和随机性，定位为BF16 batch-shape kernel数值路径。
- 当时曾据此拒绝把阈值放宽到`.002`并把Writer固定为batch1；owner随后明确裁决该做法把普通BF16
  low-bit差异误升格为科研变量且无谓牺牲吞吐。该决定已经撤回，不能作为当前实现或下一步依据；本节只
  保留失败定位的历史过程。

## 2026-08-09 v6-prior部署替换后的工程与科学边界

- clean pushed`bca3f6d`已把rejected hard-route从canonical evaluator原位替换为历史v6同构的raw-video
  `CompleteLoRAWriter`。部署不再读取expert bank或phase feature cache；这两类资产只能留在train24
  supervision与历史分析侧。该改变防止“用一个nearest train expert冒充video-to-LoRA学习器”继续污染
  held-task裁决。
- historical macro400的600-tensor state可由当前v6构造器逐名strict-load，真实state values=
  12,064,064。validation8真实asset inspector和CLI prepare均得到8个one-shot video/LoRA requests；旧
  deployment参数fail closed，no-video返回source identity且不读frames。
- shuffled/reversed的因果操纵语义已落实为“按错误展示顺序送入frame content，同时使用新的顺序位置”，
  而不是把content和原始时间戳一起置换后让模型恢复正确顺序。这与用户对人类教学视频的直观相同：
  倒放与乱序必须真正破坏动作先后关系。
- CPU门只能证明资产、shape、信息墙和cache handoff成立。当时A40结果曾错误导向batch1和`1e-5`
  direct门；该执行结论已由本文件顶部的throughput-first裁决撤回。当前单卡门只认真实batch吞吐、显存、
  finite/cache/release/reuse和vertical rollout合同，通过前仍不能进入六卡训练。
- v6-prior的科学假设因此保持单变量：上游video semantics/Procedure继续使用143起点，只修compiler+
  factor heads如何把它写入task-expert定义的policy-effective方向。若后续absolute没有共同提高，不能再
  把首因归给expert LoRA能量、hard/soft路由或evaluator部署错图。

## 2026-08-09 v6-prior transferable Writer的根因定位与设计决策

- 训练侧实现已由clean pushed`dd57edc`落地并通过全仓`215 passed`。真实data gate确认24 tasks、
  206,346 query rows；profile macro49覆盖480 unique跨episode action queries且包含最长105 sampled-frame
  视频。六卡profile在单卡warm-start输出复现前由config fail-close，当前这些仍是工程合同而非新性能证据。
- 历史v6 ownership的精确冻结数为`7,060,992` parameters；此前文档中的`7,062,592`是记录误差，
  trainable compiler+factor heads仍为`3,714,304`。

- 历史v6-fast macro400是当前唯一已验证的高绝对起点：correct=`143/400`、breadth6、五臂=
  `143/135/125/128/129`。其上游对wrong/reversed/shuffled的Procedure相对变化仍大；task-complete相对
  old recipe的主要收缩发生在Procedure之后，effective-LoRA传递只剩约`.42--.61`、action传递约
  `.34--.56`。因此下一最小接口是compiler，不是再换video encoder或rank形态。
- macro400 checkpoint含600 tensors；前四block encoder/Core/transition/Procedure共483 tensors、
  `7,060,992` parameters，compiler+factor heads共41 tensors、`3,714,304` parameters。下一轮冻结前者、
  只训练后者，既保护143已有语义/时序表示，也直接作用于已定位的输出增益接口。
- task experts解决train task上“什么LoRA是policy-effective”的监督问题，不解决held部署支撑和视频时序。
  所以它们不再在线选择/混合，而只通过exact effective`BA` cosine+norm监督correct输出；reversed/
  shuffled/wrong只做有界相对ranking，不能靠最大化错误动作loss或无限能量满足目标。
- step2000 experts的global effective norm范围`3.02--5.82`；q/v/action I/O能量占比范围分别约
  `78.2--87.9%/11.7--21.2%/.29--.57%`。几何监督保持真实energy weighting，positive functional action
  loss继续提供policy敏感度；不以手工等权放大小action targets，也不回到factor MSE。
- 本候选是从143 warm-start的单接口continuation，不是把旧v6冒充fresh新方法。正式必须用当前同一
  video schedule先评step0，再与10/25/50严格配对；若只提高margin而牺牲absolute，按有效负结果停止。

## 2026-08-09 Hard-route strict负裁决：task expert不是held-task部署字典

- 预注册correct80为`3/80`、breadth=`2/8`；逐task Long/Goal/Object/Spatial=
  `[0,2]/[0,1]/[0,0]/[0,0]`。运行合同完整且三卡已释放，因此是有效科研负结果。
- hard与soft15的80条state、env seed、policy RNG、teacher demo和真实frame order完全一致；hard只保留
  1条soft成功，新增2条、丢失14条，净`-12`，exact McNemar `p=.0041809`。这直接反驳“soft mixture
  稀释是当前主要瓶颈”。
- 80个cache LoRA的nearest step2000 raw expert effective cosine中位/最小=`.998544/.997096`，与第二名
  gap最小`.35133`；共覆盖11 experts。79/80与soft affine argmax一致，唯一数值flip就是预先发现的
  Long-2 state0 `.000664`边界。退化不是实现仍在soft混合、全局单expert塌缩或大量路由翻转。
- Object-1十条视频全部选ordinal10 Chocolate-pudding-to-basket expert，hard为`0/10`，而soft组合在同一
  panel为`8/10`；Object-3全部选ordinal8 Tomato-sauce-to-basket仍`0/10`。这说明soft affine组合偶尔
  产生训练expert之外的有用迁移方向，而单个语义近邻expert的task-local policy不能直接跨对象/场景执行。
- 结论不是丢弃task experts：它们已证明可定义健康、闭环有效的train-task policy update target；应继续
  作为policy-effective监督/先验。被否定的是把24个experts直接作为held部署时的hard/soft/sparse字典。
  因此停止top-k、temperature、global scale、rank和confidence修补，也不把few-shot用于平均错误字典。
- 下一方向必须恢复“可迁移生成器”而非“expert选择器”：以历史v6动态Writer中已经达到`143/400`的表示
  和优化轨迹为先验，直接学习video-conditioned policy-effective完整LoRA；同时保留视频为唯一dynamic
  value，并把expert target用于稳定task accumulation。先做CPU/历史合同核对和结构设计，再决定GPU门。

## 2026-08-09 Hard-route真实资产CPU判别

- 唯一部署路径已把soft affine composition替换为video-conditioned signed-argmax one-hot；soft scores只
  留作审计，不存在top-k/temperature/global scale选择。实现`1619631`已push，schema v4和`hard1` asset
  reference阻止旧soft cache冒充。
- train24上路由不是薄弱环节：24/24 centroids及1,200/1,200独立videos都选择本task expert，且top1-
  top2 margin中位`.630`；24 experts各50次，没有静态塌缩。ordered反转后1,200/1,200都换expert，固定
  phase shuffle后699/1,200换expert，说明现有causal representation确实含有强顺序敏感信号。
- one-hot compiler保持真实expert policy update：24×38 target effective cosine中位`.998982`，最差
  `.961962`；zero exact identity，所有state finite。该门只验证机制和runtime输入输出，不说明held
  expert能闭环执行validation task。
- correct80旧soft coefficients的argmax会选择11个train experts，各validation task有稳定或双峰路由；
  Object-1全部选Chocolate-pudding-to-basket，Object-3全部选Tomato-sauce-to-basket。held top1-top2
  margin中位仅`.0193`，所以hard screen是高干预、可归因的support试验，而不是对soft LoRA的小扰动。
- 下一步仍只能先做单卡online smoke；若同一80-row hard screen没有实质高于`15/80`并保住breadth，
  证据将直接否定“soft dilution是主因”，停止在24-expert mixture内调参并转向v6先验的可迁移Writer。
- 单卡online smoke的工程链路完整通过，且8个live生成LoRA的nearest one-hot effective cosine最小
  `.999999799`、nearest-vs-second factor-distance gap最小`.389`，排除“代码声称hard但在线仍在soft混合”。
  8条覆盖7 experts；只有Long-2 state0与旧soft implied argmax不一致，因为旧ordinal12/13 margin仅
  `.000664`，live为13。hard argmax在held边界附近可能受微小encoder/数值扰动翻转，这是下一screen必须
  记录的稳定性风险，不通过confidence gate或temperature事后修补。

## 2026-08-09 Policy-Effective correct80负裁决与下一判别

- 预注册validation8×states0--9 screen自然完成为`15/80`、breadth=`5/8`，逐task Long/Goal/Object/
  Spatial=`[1,0]/[0,2]/[8,0]/[1,3]`。36/36 jobs、80 unique LoRAs、9 workers、信息墙与释放合同全部
  有效；这是科学non-pass，不是工程失败。因score低于`22`，不扩跑160/400、不做五臂。
- exact same-video paired screen相对raw barycentric为gained/lost=`6/3`、`p=.5078`，相对source=
  `13/7`、`p=.2632`；相对v6-fast same-state/different-video为`5/18`、`p=.01062`。policy-effective
  compiler有小幅正向效应，但远不足以恢复历史上限。
- 80个输出的exact effective`BA`几何为norm/stable/top中位=`4.148/1.234/.847`，A/B RMS=
  `.018909/.008413`，q/v/action B-column cosine=`.610/.626/.372`，16/16 coordinates active。
  因此当前不是LoRA能量、公共rank、inactive coordinate或rank96压缩故障。
- 新compiler与同视频raw-factor输出的effective cosine中位仍为`.958`、relative-L2=`.302`、norm ratio=
  `1.055`。这把30节的cross-term错误重新定量为“真实但次要”：修复后只净增3条，不能继续把主要失败
  归给factor algebra。
- same-task不同video/cross-task/task-mean exact cosine中位=`.989/.703/.712`，最近step2000 expert=
  `.641`；每task视频中心化effective variance仅`.56%--2.54%`。Object-1与Object-3都路由到语义正确的
  basket-pick experts，闭环却分别`8/10`与`0/10`。最早不确定性是soft composition是否稀释可用expert，
  还是train expert对held object/scene根本不具备足够迁移支撑。
- 下一最小判别不是继续调scale/rank、训练大decoder或加入few-shot，而是保持reader不变，把每条视频的
  coefficients确定性argmax为one-hot并复用同一80-row panel。若hard route明显提高，后续研究
  video-conditioned sparse routing；若仍低，则停止在24-expert convex/affine support内修compiler，转向
  以v6强先验或更可迁移policy-effective target训练单一Writer。hard route仍由视频决定，不是task-ID/
  outcome oracle。

## 2026-08-09 Causal Barycentric strict负裁决与policy-effective compiler根因

- 正式correct400=`63/400`、breadth=`5/8`，逐task为Spatial`0/6`、Object`38/0`、Goal`0/17`、
  Long`1/1`。400 states/videos/LoRAs、72 jobs、18 workers和信息墙全部有效；相对same-video
  source/addressless的gained/lost=`46/31`、exact `p=.1100`，相对address-binding=`27/39`、
  `p=.1753`。这是有效科研non-pass，不是运行故障；不做其余五臂。
- 400套LoRA的norm/stable-rank/top-energy=`3.958/1.155/.894`、16/16 coordinates active、top4
  energy=`.271`，q/v/action B-column cosine=`.712/.744/.351`，已接近task experts的健康形态。
  因此“继续解决能量”不是当前方向。每条query却平均有效使用约13个experts，same-task/cross-task/
  task-mean cosine=`.988/.685/.697`，远未恢复expert bank跨task中位约`.100`的分离。
- 最早的数学错误是把policy-effective manifold误实现在factor manifold：分别加权A和B得到
  `B(c)A(c)=sum_{k,j}c_kc_jB_kA_j`，除期望的同expert项外还有大量`k!=j`交叉项；chunk-wise方向/
  log-scale归一化又使这一关系更非线性。因而one-hot exact、健康谱和语义合理coefficients都不能保证
  affine组合对应任何expert策略。
- task expert bank本身已用development-train random-reset闭环证明step2000=`658/1200`且23/24 tasks非零；
  它解决“监督目标是否是policy-effective更新”，不解决held task是否可由train experts组合、对象/
  场景变化是否可迁移，也不解决视频时序因果。Object-1与最邻近train Object-8语义对应且得`38/50`，
  Object-3与最邻近train Object-5同样语义对应却得`0/50`，正说明语义近邻不是闭环可组合性的充分条件。
- coefficient-reader反事实把两个接口拆开：contrastive reader能把correct/reversed/shuffled目标方向变为
  约`.394/-.392/-.008`，但held update norm ratio仅`.106`；rectified prototype可让reversed近identity，
  correct target cosine仍约`.381`且幅度偏低。视频顺序可辨识，不等于能生成足够强且可迁移的策略更新。
- 下一单变量是只修compiler：在有效`BA`空间先做`sum_k c_k B_kA_k`，再投回同一public rank-16；
  保持视频representation和coefficients不变。先用CPU比较shared joint subspace与per-query exact low-rank
  compression的投影保真，避免用另一个GPU实验猜测。
- CPU门否定了最朴素的pure affine版本：其effective norm中位`2.220`、相对expert仅`.527`，即使方向
  语义正确也会因13-expert平均而幅度不足。per-target normalized effective direction + affine log norm
  则为`4.155/.986`，既避开cross-factor项又保持真实expert target-energy profile，因此被选中。
- public rank16不是限制：8个task的full-span exact样本captured-energy中位`.99523`。shared rank96
  对真实400 queries的global captured-energy/cosine中位=`.99365/.99682`、最小=`.99065/.99532`；
  对24 one-hot experts captured-energy中位/最小=`.99677/.99331`。rank128只把query cosine中位从
  `.99682`提高到`.99712`，不值得额外basis/runtime成本。rank96是当前证据支持的最小充分内部容量。
- 真实资产实现确认CPU预测没有因factorization落空：one-hot expert/demo0 intended effective cosine中位=
  `.99838/.99836`，zero identity exact。template-A Procrustes gauge给出norm/stable/top=
  `4.179/1.125/.910`、A/B RMS=`.01891/.00846`、q/v/action B-column cosine=`.815/.813/.455`、16 active，
  与正常experts同量级；不是另一种低能量、单rank或99%列共线LoRA。train24 demo0跨task cosine中位
  `.203`，显著接近expert bank`.100`而非旧Writer公共方向。
- 专属A40 smoke进一步排除online工程错位：8/8不同validation task各生成一套唯一完整LoRA，v3 evidence
  显示每行确实读取correct action-hidden frames；0 forbidden reads、0 retry/failure/OOM/nonfinite，Writer
  release和source-policy原位复用均闭合。`1/8`只说明环境链路可执行，样本太小，不能支持或否定absolute
  性能；因此下一门仍是预注册strict correct小panel，而不是据此改reader或进入few-shot。

## 2026-08-09 Causal Barycentric online smoke与LoRA几何

- live双节点检查后只使用空闲`gpu02:0`完成validation8×1-state纵向链路；8套唯一FP32 LoRA、8 cache
  entries、2个batch4及3个rollout workers全部首次完成，0 retry/failure/OOM/nonfinite，四类forbidden
  reads均为0。Writer/encoder释放后同一source policy原位复用且没有reload，GPU随后完全释放。
- 8套held LoRA的norm/stable-rank/top-energy中位=`3.9802/1.1555/.89243`，16/16 rank coordinates
  active、top4 coordinate energy=`.27103`。cross-task effective cosine中位`.69277`、nearest step2000
  expert cosine中位`.65624`；这比learned address-binding Writer full400的`.94197/.12734`同时更分离且
  更靠近真实task-expert方向，说明闭式坐标并未在线退化成公共LoRA。
- 该结论只有8个task各一条video，不能估计same-task video方差，也不能从`1/8` smoke success推断
  closed-loop性能。它只解除工程门；config现为`sealed`，下一证据必须是400-state strict correct与
  400-LoRA task separation，之后才有资格运行same/wrong/shuffled/reversed/no-video。

## 2026-08-09 Causal Barycentric canonical实现结论

- clean pushed`1d9d030`把闭式流形坐标原位接入唯一evaluation runtime，并删除learned Writer
  trainer/checkpoint/model及旧checkpoint CLI。新方法没有可训练Writer参数，也没有checkpoint选择轴；
  shared资产是固定step2000 train24 experts、train24×50 action-hidden centroids和同一闭式规则。
- 真实资产的关键数值合同成立：24个one-hot coordinates逐一重建完整expert的最大误差`2.235e-8`；
  zero/phase-constant表示逐tensor精确回到source identity；非零query coefficients之和最大误差
  `1.192e-7`，24/24 demo0 ordered/reversed不同，task-pair coefficient L2中位`.97133`。
- 这直接避免learned decoder的“所有task向公共方向训练塌缩”，但并不证明held validation video落在
  正确expert convex/affine邻域，也不证明重构LoRA闭环有效。下一门必须是在线A40 smoke后strict
  correct400，而不是把CPU LOO或one-hot exactness当作性能结果。
- 全仓180 tests与结构门通过；无hard violation、无parallel implementation family，当前变更净删
  941 active lines。config曾故意blocked以防旧address-binding smoke冒充新图证据，现已由上面的专属
  online smoke精确解封。

## 2026-08-09 Address-binding 75/400与Causal Barycentric设计裁决

- address-binding macro50的正式strict correct400=`75/400`、breadth=`4/8`，逐task为Long
  `[2,0]`、Goal `[1,47]`、Object `[25,0]`、Spatial `[0,0]`。400-row/LoRA、72 jobs、18 workers和
  信息墙全部有效。相对exact同video schedule的旧addressless macro50，gained/lost=`31/4`、exact
  `p=3.47e-6`，说明乘性地址修复确实进入了closed loop；但净增几乎只形成Goal-6和Object-1两种能力，
  远低于v6-fast `143`与长期门，不能续训或做昂贵五臂。
- full400 LoRA不是旧式能量/秩坍缩：norm中位`3.20095`、stable rank`1.31757`、top singular
  energy`.77753`、16 coordinates active。真正病灶是方向同质化：same-task不同video cosine中位
  `.99791`，cross-task `.94197`，task-mean cross-task `.94270`；最近train expert仅`.12734`。
  每task video-centered variance只约`.00118--.01801`。macro3八task pairwise中位`.54184`到macro50
  反而升高，证明训练先吸收了24 targets的公共raw-factor均值，而没有成熟为task residual。
- 这与expert target统计吻合：raw expert mean占约`.414`能量，而centered target仍有约19.54 effective
  dimensions；原loss在129万坐标上优先降低公共均值误差，direction项到macro50仍约`.819`，所以“更大
  decoder/更多步”没有证据会自动学到centered manifold。最早剩余接口是
  `causal video representation → task-discriminative expert coordinates`。
- 闭式LOO把每个train task及其expert整折拿掉，只用其余23个video centroids与experts预测held task。
  causal ridge `.3`的直接raw-factor affine得到effective target cosine中位`.38838`，但近正交expert
  相消使norm仅`1.740`，不符合健康能量。改为168 chunks逐项混合normalized expert direction、affine
  插值chunk log-scale并限制在train-expert envelope后，cosine仍为`.38302`，norm恢复到`3.84385`，
  stable rank/top energy=`1.15056/.89540`、top4 coordinate energy`.27048`，接近真实expert
  `4.21249/1.12877/.90846/~.26`。
- 同一LOO中reversed/phase-shuffled cosine降为`.098995/.185395`，correct margin=
  `.284026/.197626`。因此因果phase representation不仅区分task，还能把正确顺序映到更接近held expert
  的完整LoRA；这比事后向loss加一个漂亮margin更接近用户要求的“正确视频提供有效动作知识”。但这仍是
  train-task机制代理：16-slot shuffle不等同formal raw-frame shuffle，Goal/Long部分task margin很弱，
  最终只认validation five-arm closed loop。
- 由此选择Causal Barycentric Topological Writer作为下一唯一canonical候选。它固定step2000 experts和
  50-video train centroids；部署时一条视频的phase-centered causal value求24个affine coefficients，
  再直接重构完整rank16 LoRA。zero/phase-constant value令coefficients全零，language没有独立value，
  所以没有language-only LoRA bypass。learned coefficient reader、few-shot和显式negative training均
  延后，只有闭式候选的strict结果证明哪一接口仍不足后再单变量引入。

## 2026-08-09 Address-binding macro50内部裁决

- fresh formal0→50本身工程健康，但最重要的结论来自同一train24 demo0纵向对照。cross-attention和
  axial输出的chunk/rank centered energy中位仍只有`4.60e-6/4.47e-6`与`5.64e-6/6.14e-6`；这证明
  上游动态值本身没有学会坐标身份，也确认旧根因诊断没有被推翻。
- 新乘性接口恰好在最早断点处恢复地址：addressed latent两轴energy=`.4930/.4765`，最终LoRA token
  output=`.4669/.6159`，而expert target=`.9936/.9364`。因此地址信息不再被permutation-equivariant
  decoder不可逆抹掉，且不是靠静态address单独输出。
- train24 raw-token/own-effective expert cosine中位由旧图约`.0233/.0108`提高到`.1177/.1342`；nearest
  expert cosine`.1393`，8/24 tasks最近的是自己。LoRA norm中位`3.360`接近expert`4.212`，stable rank
  `1.349`、top singular energy`.757`、16 coordinates active。结构修复带来真实target方向改善，不只是
  抬高谱或能量。
- 仍然最危险的结构风险是task分离：24套generated LoRA的pairwise effective cosine中位`.8686`，而
  step2000 experts约`.100`；top4 coordinate energy也高达`.8694`。Writer可能把所有视频写成一个
  高rank但公共的平均方向，只在其上做小幅task变化。这可以比旧近rank1随机方向好，却仍可能导致
  task rotation或有限closed-loop增益。
- 因此macro50 strict correct400是必要且充分的下一证据。若absolute没有material上升，不能因内部
  cosine提高而resume；若absolute上升但task集中或时序margin失败，下一接口才转向condition/task
  direction separation或显式order credit，而不是继续放大norm。

## 2026-08-09 Address-binding online smoke与早期LoRA几何结论

- 新图的纵向链路已经闭合：8个validation task/state各生成一套唯一完整rank-16 FP32 LoRA，随后释放
  Writer/encoder并原位复用同一source policy；3 workers首次完成全部8行，0 retry/failure/OOM/nonfinite。
  每行恰好一条correct action-hidden视频，teacher action/state/reward/terminal reads均为0。
- 对8套macro3 LoRA逐tensor检查没有nonfinite。effective norm中位`.70069`尚处于early-profile幅度，
  但stable rank中位`1.98260`、top singular energy中位`.51202`、16/16 rank coordinates active、top4
  coordinate energy中位`.31274`。相较旧macro50的stable rank`1.0000014`/top energy`.9999986`，
  地址绑定在仅3 macros时已打破结构性单lane塌缩；这不是“stable rank越高越好”的性能门。
- 八套不同task LoRA的pairwise effective cosine中位`.54184`，说明macro3尚未充分分离task方向；而且
  smoke只有8 tasks×1 state，不能估计same-task video variance、expert proximity或closed-loop能力。
  因此正确证据顺序仍是identity-fresh macro50后同时看strict correct400、train24 target cosine、
  rank/chunk retention和LoRA谱，再决定是否resume或进入时序五臂。
- `1/8` rollout success只证明执行链可运行，不进入性能比较。profile/smoke权重均弃用；config seal只
  解封fresh formal，不授权从工程checkpoint warm-start。

## 2026-08-09 Address-binding exact-resume profile结论

- 新乘性地址图没有破坏stateless flat-reduction的可恢复性：resume与contiguous在三步loss/raw/
  direction/log-scale/gradient/LR逐值一致，macro1全文件及macro3 Writer/RNG byte-exact；trainer只存在
  PyTorch容器serialization bytes差异，载入后的optimizer/scheduler/scaler逐项相等。
- 新owner不是形式参数：`address_norm`到macro3已有非零finite Adam状态，且权重从macro1到3发生
  `1.62e-5`最大绝对变化。这个结论只证明梯度和resume工程可达，不代表地址已学到expert geometry。
- 峰值reserved仍低于`.9GB`，所以结构修订没有引入A40训练内存风险。下一唯一工程不确定性是在线
  frozen video encoder→Writer→FP32 LoRA cache→释放模块→source-policy rollout的纵向兼容性。

## 2026-08-09 Address-binding reprofile证据边界

- 旧profile不能验证新forward图，但既有flat ordered all-reduce数学与资源结果仍给出规模上界；因此
  本轮只重做三步工程等价性，不改变world6、24-task mean、optimizer或scheduler，也不把profile权重
  用作warm-start。
- 预注册将“恢复后结果一致”拆成scientific metrics、Writer/RNG逐字节和optimizer/scheduler语义三层；
  同时要求新`address_norm`确实存在并收到finite/nonzero梯度。任何一层不通过都先定位工程合同，不能
  用容差把新结构封成formal。

## 2026-08-09 Topology-address修订的结构结论

- 根修不是增加LoRA全局能量，而是恢复“动态视频值写到哪个LoRA坐标”的可辨识性。canonical公式为
  `Z[b,c,r]=RMSNorm(D[b,c,r]) ⊙ RMSNorm(Qchunk[c]+Qrank[r])`，其中`D`仍完全来自
  phase-centered causal video value；`Z`才进入共享output projection。静态address只能调制动态值，
  不能在无视频动态时产生adapter，因此没有language-only或static-LoRA bypass。
- 乘法而非加法是identity合同的关键：zero/phase-constant输入令`D=0`，无论address学成什么，
  `Z=0`且template-A/zero-B逐tensor不变。它也避免把address当成另一套可独立记忆24 tasks的LoRA。
- CPU回归证实该修订直接消除了已定位的表示论断点：给所有chunk/rank完全相同的`D`，绑定后两轴
  centered energy都`>.1`；zero动态仍精确零，ordered/reversed不同，上游与新address norm在
  zero-head打开后均有梯度。这里证明的是机制可达性，不是已经学到expert manifold或闭环提升。
- 旧profile/smoke只验证没有address-value绑定的失败decoder，不能外推到新增参数和forward图。
  `cd95281`因此主动撤销meta formal seal并要求fresh六卡exact-resume profile与macro3 online smoke；
  只有这些工程门通过后才允许identity-fresh训练。one-shot、expert2000、feature cache、target、loss、
  optimizer、24-task等权与strict evaluator均未改变，便于把下一结果归因到地址接口。

## 2026-08-09 Expert-Manifold macro50 strict负裁决与topology-address根因

- replacement correct400完整自然结束：`48/400`，8-task依次为Spatial=`0/0`、Object=`4/0`、
  Goal=`0/42`、Long=`2/0`。72/72 jobs、400 unique rows、18 workers attempt1/exit0、0 retry/error/
  OOM/nonfinite；每条只读exact language加一条correct action-hidden video，400套LoRA与forbidden-read
  零计数闭合。这是有效科研non-pass，不是运行故障。
- 它与旧source-base同一400-state panel的aggregate恰同为`48`；逐state的env seed、policy seed与
  noise prefix全部匹配，both-success/source-only/writer-only/both-fail=`43/5/5/347`。Writer没有
  建立共同新能力，不能因Goal-6的42次成功把source原有能力误记为meta学习。
- 全400生成LoRA的effective norm中位=`4.54899`，与step2000 expert中位`4.21249`同量级；失败不是
  能量不足。相反stable rank中位=`1.00000144`、top singular energy=`.99999856`，q/v/action
  B-column cosine各task均约`.99999`。nearest-of-24 train-expert effective cosine中位只有
  `.007974`，说明高幅LoRA方向没有落在policy-effective expert manifold上。
- train24自身demo0复算也得到raw token/effective-BA target cosine中位仅`.02326/.01081`，排除
  “只在validation泛化失败”。rank/chunk address centered energy从静态query的约`.481/.486`，在
  cross-attention后降到中位`1.04e-6/1.08e-6`，经过四个axial blocks与共享output projection后降到
  `2.51e-8/4.67e-10`；expert target中位却为`.936/.994`。cross-attention只把query当权重、没有保留
  zero-preserving address value；一旦16-phase动态value近同，后续无位置编码的permutation-equivariant
  axial blocks无法重新创造chunk/rank身份，并进一步强化共同方向。
- 因此最早失效接口是`video dynamics → topology-addressed latent`，不是video encoder、expert能量、
  closed-loop evaluator或训练时长。原轨迹拒绝resume50→100；先在唯一路径加入“动态video latent ×
  静态chunk/rank address”的乘性零保持绑定，保证zero/phase-constant video仍精确identity，再fresh
  profile/train。order-negative loss与few-shot暂不同时加入，避免掩盖该单一结构因果。
- 400套cache实际保存FP32，共`2,064,364,800` tensor bytes；原descriptor按BF16估算约1.03GB。
  这不影响数值结论或quota安全，但后续资源预算必须按实际cache dtype计算。

## 2026-08-09 Formal checkpoint跨worktree身份根因

- 失败谓词精确位于`expert_manifold/inference.py::_training_checkpoint`：run contract中的training config
  absolute path必须等于当前evaluation config path。两个clean frozen worktree必然有不同前缀，因此
  该门与仓库自己的formal isolation规则冲突。
- 合法身份不是“绝对目录相同”，而是同一仓库相对authority、相同schema/bytes，以及run contract中
  method、information wall、topological writer、meta training、source和checkpoint manifest全部相同。
  修复只替换这一条路径表示，并额外补上此前未检查的config bytes；没有放宽科学配置或模型身份。
- 真实macro50 artifact在修复后通过，错relative path和错bytes均fail-close。首次评测没有GPU或结果，
  不能计为模型失败，也不能复用其root。
- 修复seal为clean pushed`d59841e`；replacement只更换evaluator commit/root，不改变checkpoint、panel、
  视频schedule、policy RNG或任何模型数值，因此后续r2结果仍是预注册macro50 correct arm。

## 2026-08-09 Macro50 formal训练观察

- formal0→50与profile前三步完全相同的identity初始化语义起步；macro1 loss低是direction项在zero output
  尚未定义为惩罚，macro2以后direction约`.986`并到macro50缓降为`.9764`，不能把总loss的跳升误判为
  训练发散。
- raw reconstruction从macro1`4.0655e-5`到macro50`7.8499e-5`并非单调下降，但log-scale项从
  `.03030`降到`.01855`，梯度始终finite且末步`.01526`。这表明当前复合目标主要先学习scale/非零
  direction，单看raw MSE不支持“已接近expert”或“应继续训练”的结论。
- 训练本体仅约10秒而strict correct400约1GiB cache和400条环境轨迹，科学瓶颈明确在
  generated-LoRA→closed-loop接口。下一证据必须是macro50 correct400和内部LoRA/action传递。

## 2026-08-09 Formal首段资源与证据顺序

- flat-reduction profile的稳态macro wall约`.13--.19s`，完整checkpoint约184MiB、峰值reserved低于
  `.9GB`；因此meta训练本体不是当前成本瓶颈，macro50后strict closed-loop才是主要证据成本。
- 首段保持formal scheduler总长800和warmup25，仅用`--stop-after-macro 50`分段；不会把50步重新解释
  为完整schedule，也不会加载profile权重。每macro仍覆盖24 tasks×1独立视频，0→50共1,200个
  video-conditioned full-LoRA reconstruction pairs。
- 选择macro50先评correct400，是为了尽早测量surrogate→policy闭环接口；reconstruction loss、expert
  proximity或健康LoRA几何都只作机制证据，不能触发续训或宣告性能。

## 2026-08-09 旧K4/AS/RL executable原位退役

- design第12节触发后的canonical cleanup已完成：删除旧K4/AS/RL model、training、checkpoint、
  live-generation模块和4个旧入口，并移除29份旧Writer配置/校验文件及8份专属测试；Git和formal
  artifacts继续保存历史，没有创建archive目录或兼容分支。
- 共享`writer/data.py`、`as_sampling.py`、`functional.py`、topology、LoRA rollout、cache与evaluation
  runtime保留；错误类型和prepared-LoRA schema被提升到两个最小共享owner。统一evaluator只接受
  Expert-Manifold动态Writer、静态Source-SFT或task expert，不再接受旧AS/RL adapter或rollout B-scale。
- active one-shot video mapping、shuffle/reverse和evidence owner已移到`expert_manifold`；新聚焦测试文件
  避免继续扩张既有超长runtime测试。`CUDA_VISIBLE_DEVICES=''`加正式LIBERO assets的全仓回归为
  `186/186`，compileall/diff check通过；architecture guard无hard violation、无parallel family，active
  source additions/deletions/net约`525/13,592/-13,067`。这只是架构收口，不是性能证据。

## 2026-08-09 Online generation/cached rollout通过并seal formal

- clean pushed`31d41d8`的replacement root完整产生8/8 unique validation rows和8个唯一rank-16 LoRA
  references。一个generator以两个batch4生成8套LoRA，generation wall=`12.634s`；三个rollout workers
  各完成`5/1/2` shards，全部attempt1/exit0，0 retry/failure/OOM/nonfinite。
- online阶段peak allocated/reserved=`10,576,054,272/11,182,014,464` bytes；释放Writer/encoder后为
  `9,391,467,520/9,651,093,504` bytes。`writer_modules_released=true`，已加载source policy直接复用于
  rollout且没有reload；cache manifest、episode evidence与8个唯一LoRA引用闭合。
- 每个episode只读exact language与一条correct action-hidden teacher video，teacher action/state/reward/
  terminal reads全0。smoke的`1/8` success只说明整条执行链可运行，不是方法性能证据。
- 六卡stateless flat-reduction profile与单卡online smoke两道门均已通过，meta formal config现已seal。
  旧K4 executable的设计移除触发已满足；正式identity-fresh训练前先原位退役旧路径并跑完整CPU回归。

## 2026-08-09 Online smoke source-descriptor fail-close根因

- 首次macro3 online smoke在CPU prepare、0 CUDA worker/0 scientific row时停止。训练profile通过formal
  source inspector记录非空`source_run_summary`；evaluation smoke对完全相同的final source checkpoint
  把这一模式相关字段表示为`null`。其余source字段逐项相同，故不是模型、checkpoint或数据漂移。
- scoped repair仅接受这一项缺省，并要求training contract中的summary descriptor仍指向存在且path/bytes/
  schema匹配的文件；其他任一source字段变化仍失败。真实macro3 profile authority现可被smoke接受。
- 首次root已ABORTED、不得resume。聚焦58/58、正式assets环境全仓224/224；replacement必须是新root、
  clean pushed commit并重新live看卡。该修复没有GPU或科研结论，也不改变Writer输出。

## 2026-08-09 Flat-reduction exact-resume core profile通过

- clean pushed launch-record`b00024b`的fresh0→1/resume1→3与独立contiguous0→3三步科学metrics逐值
  相同；macro1和macro3 Writer、六份macro3 RNG逐字节一致。它消除了首轮DDP profile的系统性A/B轨。
- macro3 `trainer.pt`原始序列化bytes不一致，但load到CPU后的optimizer/scheduler逐项0差异。这是容器
  序列化差异，不是训练状态差异；证据边界只写Writer byte-exact和trainer semantic-exact。
- resume/contiguous峰值allocated/reserved分别为`.736/.877GB`与`.736/.816GB`，0 OOM/nonfinite；
  run contract逐rank 3+3 NUMA/physical-local mapping正确，并封存无DDP wrapper、单flat ordered mean、
  P2P disable与Ring/Simple。由此确认旧DDP reducer生命周期是未checkpoint的隐藏状态。
- 这些仍只是core工程证据。profile权重不得warm-start；正式训练还必须先通过macro3 online video→LoRA→
  cache→release→rollout smoke。

## 2026-08-09 Meta exact-resume DDP reducer working root cause

- 首轮六卡fresh0→1/resume1→3与独立contiguous0→3都finite且资源健康；两个macro1 Writer、trainer、
  optimizer/scheduler和六份RNG逐字节一致。但macro3有45个Writer tensors、约681万值不同，最大绝对
  差约`1.30e-5`，所以不能用“数值很小”放宽exact-resume门。
- 启用PyTorch deterministic algorithms加cuBLAS deterministic workspace后分叉不变；再强制math-SDPA
  后仍稳定复现resume路径A/contiguous路径B。分叉不是checkpoint漏存、随机kernel或flash attention，
  而是在恢复后的首个optimizer update后系统性产生。
- 与已验证source-base/Source-SFT DDP对照后，唯一有代码差异支持的候选是meta trainer没有
  `static_graph=True`且仍
  broadcast immutable buffers。重启会重建“首次迭代”reducer，而连续run已经过该生命周期；reducer
  自适应状态没有checkpoint。修复沿仓库既有模式固定static graph、关闭buffer broadcast和unused
  parameter search，并将DDP语义写入run contract。新profile必须以逐字节parity确认或否决该解释；
  旧profile与两个diagnostic probes均不得复用。
- 候选修复已由clean pushed`12727b8`封存并通过46/46聚焦CPU合同；新的fresh/resume/contiguous roots
  与exact门已预注册，但尚无GPU结论。
- 真实static-graph root在macro1、0 optimizer step触发PyTorch 2.11 reducer
  `expect_autograd_hooks_`内部断言；dynamic graph只关闭buffer broadcast后仍复现完全相同A/B分叉。
  这否决了两个候选并把结论收紧为DDP reducer生命周期本身是未checkpoint隐藏状态。
- canonical replacement不再使用DDP wrapper：每rank顺序累积4-task local mean，按固定parameter order
  拼一个flat gradient，以固定Ring/Simple NCCL做一次all-reduce mean，再共同clip/AdamW。它与原24-task
  等权梯度数学等价，但去除reducer历史；新profile仍须证明byte parity，尚无GPU通过结论。
- retained seal为clean pushed`c33a16b`；新flat-reduction roots、Ring/Simple exact commands及byte门
  已预注册。CPU为49/49聚焦、223/223全仓；其后GPU core profile已通过，但仍不构成性能结论。

## 2026-08-09 Task-expert五点闭环终态与统一step2000选择

- clean pushed`1362d15`的step1500/2000 roots均自然完成1200 unique rows。本轮每点3 GPUs×3
  replicas，动态分片为126 jobs和9 workers；均126/126 complete、attempt1、exit0、0 retry/failure。
  五点task/state/env seed/policy seed与共同长度policy-noise prefix逐row一致，pairing mismatch=0。
- 五点success=`432/557/624/638/658`；suite顺序Spatial/Object/Goal/Long在1500→2000为
  `178→181/216→228/164→166/80→83`。paired gained/lost=`77/57`，24 tasks中17升、5降、2平；
  2000相对1000为`91/57`，净增34。五点state union/intersection=`801/312`，per-task oracle=`684`，
  只比统一2000多26。
- breadth并非单调：nonzero tasks=`21/23/24/23/23`，成功至少25次=`8/11/14/16/15`；task9在
  1500/2000仍为0，且1500→2000虽然parameter update energy仅`.000312`，仍有134个discordant states。
  这证明expert自身的policy boundary也高度敏感，task drift没有被task-local训练自动解决。
- 选择规则以direct closed-loop为主。2000虽然不改善target proxy且breadth比1000少1，却取得最高
  absolute、四suite全净增且多数task改善；因此统一选择step2000作为meta reconstruction target，不做
  task-specific混点。658/1200是privileged train-task expert target质量，不是Writer成绩。

## 2026-08-09 Meta profile live资源边界

- 00:01 CST实时比较：gpu01物理`0,1,2|4,5,7`均为空闲A40并满足3+3 NUMA，host available memory
  约516.5GB；物理3为`nlge` VLLM，明确不触碰。gpu02物理6、7有他人进程，空闲0--5只能形成4+2
  NUMA，故不用于六rank DDP。
- `/data1` quota=`552,249,764/1,073,741,824 KiB`；bank约938MiB、cache约113MiB，三条profile
  roots预计新增低于2GiB。profile只验finite/OOM、task等权、梯度、NUMA与exact-resume，再用macro3
  做online generation→释放Writer→cached rollout smoke；权重不进入formal。
- 统一step2000 config seal为clean pushed`d96f0fb`；exact roots/commands/acceptance已预注册在
  `task_plan.md`。root suffix指向科学seal，运行时clean/pushed commit另由run contract精确记录。

## 2026-08-08 Task-expert 2000终态与晚期target平台

- clean`81101fe`原root exact-resume1000→2000已自然完成：24/24 tasks、6/6 summaries、
  step1500/2000各24 checkpoints、0 error/OOM/nonfinite。最后50步24-task等权action loss从
  step1000 `.105372`降到1500 `.103881`和2000 `.103526`；这仍只是surrogate拟合。
- effective-LoRA norm中位=`4.170/4.212/4.212`，stable rank均约`1.129`，top singular energy均约
  `.909`，跨task cosine均约`.100`。1000→1500 update sample energy只剩`.05294`，1500→2000为
  `.000312`；expert参数流形在1500后实际上已收敛，没有出现更健康的rank或task分离。
- target raw centered effective rank=`19.45/19.54/19.54`、B cross-task cosine=
  `.10245/.10348/.10347`，同样稳定。causal-prefix one-shot B proxy correct=
  `.38820/.38685/.38678`，reversed=`.06042/.06399/.06425`，phase-shuffled=
  `.19110/.19195/.19199`；晚期训练没有提高视频可预测性，order margin反而微降。
- 因此2000不能靠更低loss、更大step或更大norm自动胜出。唯一剩余选择证据是与既有三点完全paired的
  1500/2000 development-train closed loop；若行为也平台，较早near-max target对meta学习更合理。

## 2026-08-08 Task-expert 1500/2000 live launch边界

- 22:41 CST双节点live preflight：gpu02物理0--5均0MiB/0%且host available memory约524GB；物理6有
  `yfwang`进程，物理7虽空闲但不使用。gpu01物理3有`nlge` VLLM且不触碰。选择gpu02:0,1,2与3,4,5
  两组并发，每卡3 replicas，总18 workers，和已验证有效主机内存规模相同。
- `/data1` quota=`552,236,168/1,073,741,824 KiB`，bank约938MiB，两个新root/log均不存在；从
  clean pushed`1362d15`运行，除统一expert step外保持source、50-state panel和RNG完全相同。

## 2026-08-08 Expert-Manifold cached-rollout schema接口根因

- 静态纵向追踪`evaluation_runtime → expected_writer_episode`发现，Expert-Manifold接入统一adapter
  dispatch时wrapper仍只有旧签名，没有接受调用方一直传入的`evidence_schema`。这会让昂贵的online
  LoRA generation先成功，随后在Writer释放、rollout scale-out开始前因unexpected keyword
  `TypeError`失败；单测generation或LoRA cache本身无法发现它。
- 根修让统一wrapper接受schema：旧Writer继续传给原evidence owner，Expert-Manifold生成canonical
  evidence后显式验证`schema_version`，不匹配就fail-close。正确schema等价和错误schema拒绝均有
  regression；聚焦62/62、全仓220/220及`py_compile`通过。
- 这是可复现工程合同缺口，不是scientific non-pass；它不改变teacher video、Writer输出、LoRA cache、
  source policy或随机数，也没有产生GPU/闭环性能证据。后续profile必须覆盖generation和cached rollout
  两段共存/释放边界，不能只证明模型forward。

## 2026-08-08 Expert-Manifold feature dynamics与causal-prefix value裁决

- full24×50 sealed cache的phase-DC能量占比中位`.98057`，temporal residual仅`.01943`；原decoder
  将完整projected memory同时作为key/value，会让同task恒定expert target通过约50倍更强的静态
  task摘要完成重建，reconstruction loss无法迫使它学习顺序。
- encoder并没有丢失时序：leave-one-video-out ordered temporal-template cosine中位`.88284`，
  reversed=`-.32402`、phase-shuffled proxy=`-.02194`、reversal relative-L2=`.21884`。temporal
  task geometry与raw/B target geometry Spearman=`.46046/.45087`，时序残差具有可预测expert方向。
- leave-one-task linear B proxy的phase-centered one-shot correct/reversed/shuffled中位=
  `.38607/.20667/.26386`，而DC-only correct=`.39500`且对反转/乱序完全不变。3/5-shot correct=
  `.39051/.39290`，边际很小；当前主要瓶颈是静态value捷径而非shot数量。
- 仅做phase centering仍不充分：若learned phase key被模型忽略，cross-attention对原始centered
  key/value仍是frame-set permutation invariant。CPU最小反例把phase key置零后，ordered/reversed
  只剩浮点求和级绝对差异；恒定task target没有目标函数压力替我们避免这条旁路。
- canonical value因此固定为`c_t=Wx_t-mean(Wx)`、`V_t=sum_{s<=t}c_s/sqrt(t+1)`，key仍为
  `RMSNorm(Wx_t)+phase_key_t`。causal-prefix uniform-pool template correct/reversed/shuffled=
  `.96263/-.94287/-.04463`，B proxy=`.38820/.06042/.19110`，相对简单centered proxy保持correct
  可预测性并把order margins从`.17940/.12221`提高到`.32778/.19709`；四suite margin均为正。
  centered/causal-prefix carrier RMS中位=`.14446/.19388`，后者能量约为前者`1.8875×`，但这不
  等于generated LoRA能量已解决，仍需真实训练后对expert形态和closed loop裁决。
- joint与Action-Expert完整特征仍控制phase routing，但只有固定顺序绑定的动态prefix能写LoRA content；
  zero或任意phase-constant输入精确identity。no-video保留严格paired元数据但不读frame，以zero
  innovation完整运行Writer。3/5-shot causal proxy只到`.39379/.39558`，仍不足以先切few-shot。
- CPU retained证据覆盖constant/zero identity、ordered≠reversed、zero-output第一步打开output head且
  第二步梯度到达input projection/cross-attention/phase keys；六卡入口逐rank封存NUMA/topology，
  profile step wall与累计显存峰值跨全部rank取`MAX`。architecture gate无hard violation，聚焦
  28/28和全仓220/220通过。审计还定位并修复了smoke evaluator误用formal checkpoint集合、从而
  拒绝合法profile macro1/3的问题；formal集合没有放宽。这只证明机制可运行，不是GPU profile、
  重建或closed-loop成绩。

## 2026-08-08 Expert-Manifold train24×50 formal feature cache

- clean pushed`222d3ac`上的正式cache已用6个独立workers自然完成并由canonical入口seal：
  `runs/outputs/pi05_expert_manifold_feature_cache_train24x50_r6_222d3ac_20260808`。它覆盖
  24 tasks×50 teacher videos，每task一个`[50,16,3072]` BF16 tensor，总量约113MiB。
- 24/24 records、6/6 summaries和24 feature files齐全；task ordinals恰好为`0--23`，每task
  demo indices恰好为`0--49`，文件尺寸与record宣告一致。peak allocated/reserved=
  `10,504,039,936/19,232,980,992` bytes，worker logs均正常完成。
- `cache_manifest.json`记录training commit、source/tokenizer/config和information wall；teacher action/
  state/reward/terminal reads合计0，validation/test video reads也为0。这证明frozen feature
  extraction完整且没有越墙，但不证明meta-Writer能重建expert、利用时序或提高闭环成绩。

## 2026-08-08 Task-expert三点闭环、漂移根因与feature profile

- 唯一有效development-train roots为
  `runs/outputs/pi05_task_expert_bank_devtrain24x50_step{0250,0500,1000}_formal_r3_1362d15_20260808`。
  每点1200 rows、24×50覆盖、108/108 shards、6 workers exit0、attempt1且task/state/env/policy-noise
  公共前缀严格配对。结果=`432/557/624`，即400-scale=`144/185.7/208`；四suite从250到500到1000
  为Spatial=`123/147/170`、Object=`125/191/208`、Goal=`142/163/164`、Long=`42/56/82`。
- 250→500与500→1000 paired gains/losses=`189/64`与`143/76`；task升/降/平分别=
  `19/1/4`与`18/4/2`。非零breadth=`21/23/24`，成功至少25次的task=`8/11/14`。三点state
  union/intersection=`731/332`，per-task checkpoint oracle=`636`，只比统一step1000的624高12；
  因此step1000是强而广的统一中间点，并正式触发全部24 experts exact-resume到2000，而不是混点。
- 独立experts仍有state turnover：Goal在500→1000总分`163→164`但gains/losses=`21/20`。
  这证明漂移不全来自shared Writer跨task梯度；action surrogate继续优化时，单task policy闭环边界
  本身也会换状态。last50 action-loss变化与success变化的Spearman仅`.034/.094`，LoRA norm变化与
  success变化也仅`.161/-.108`；loss和几何不能替代1500/2000闭环。
- 首批每点12 workers造成gpu02主机内存不安全，0 rows终止；每点8 workers时每卡4 replicas静态约
  37.7GB，首个inference activation OOM，0 complete shards终止。有效r3每卡3 replicas约30.3GB，
  总18 workers稳定完成；失败roots只保留`ABORTED.md` provenance，不得resume。
- feature cache profile在clean`1362d15`、`gpu02:4`完成task0×4 videos，task extraction wall=
  `4.372s`、peak allocated/reserved=`10.47/19.23GB`，feature=`[4,16,3072]` BF16，action/state/
  reward/terminal reads全0。formal六worker cache可按原24-task覆盖seal，不需降低batch或视频数。

## 2026-08-08 Task-expert full24三checkpoint geometry

- canonical CPU artifact为
  `runs/outputs/pi05_task_expert_bank_geometry_full24_steps0250_0500_1000_05d4868_20260808/analysis.json`；
  它读取24 tasks×step250/500/1000共72个formal adapters，training commit仍为clean`81101fe`，
  不读取held action/video、环境outcome，也不使用GPU。
- effective-LoRA norm中位随训练为`2.792/3.652/4.170`，public B RMS中位为
  `.00386/.00469/.00512`；scale仍增长。stable rank中位却保持
  `1.126/1.129/1.129`，top singular energy为`.903/.907/.909`，没有随训练形成更多矩阵有效秩。
- 16个rank coordinates对全部24 experts都active，top4 coordinate energy为
  `.262/.260/.258`；但mean absolute q/v/action B-column cosine中位从
  `.828/.843/.460`变为`.861/.853/.413`，rank-component cosine也从`.077`升到`.116`。
  因此coordinate能量均匀只说明raw factor gauge没有死坐标，不能当成16个policy-effective独立
  方向；真正BA仍是低stable-rank且q/v列高度同向。
- effective target count中位`18.79→17.17→17.04`、top4 target energy
  `.342→.368→.369`；q energy fraction中位`.822→.850→.859`，action projection能量始终只有
  `.0026/.0024/.0033`量级。写入随训练略向少数targets和q投影集中，但没有历史Writer那种单一
  target或单一rank-coordinate完全塌缩。
- 跨task effective cosine中位`.108/.095/.100`，negative pair fraction
  `.130/.080/.033`；bank含低幅共同方向，但大部分能量仍是task-specific。250→500与500→1000
  更新的task-mean/sample-energy比仅`.143/.128`，并未显示后期更新收敛成一个共享方向。
- 科学裁决：full24 geometry支持“task-local action credit确实建立了有差异的policy parameter
  targets”，但既不能证明step1000闭环更强，也不能证明继续到2000会修复低stable-rank或视频
  因果性。下一必要证据仍是三个统一step的development-train official closed loop。

## 2026-08-08 Task-expert bank完成与当前证据边界

- clean`81101fe`的正式task-expert bank已完成统一step1000：6个独立workers各4 tasks，24/24
  completion，step250/500/1000共72个checkpoint，root约562MiB：
  `runs/outputs/pi05_task_expert_bank_formal_step1000_r6_81101fe_20260807`。
- 三个统一checkpoint最后50步的24-task等权mean action loss为
  `.115355/.107207/.105372`。下降说明task-local AS拟合仍在改善，但此前证据已反复证明action/
  functional loss不能选择closed-loop checkpoint；它不能独立支持续到2000或选择step1000。
- 目前只有最早6个experts的探索性几何：LoRA norm median约`4.157`、stable rank约`1.131`、
  top singular energy约`.916`、16个rank坐标均active、top4 coordinate energy约`.258`；q/v/action
  B-column cosine约`.860/.874/.395`，跨task effective-LoRA cosine约`.260`。相比旧Writer的近
  rank-1/跨target同向输出，它初步显示task-local action credit能形成更丰富方向；但样本不完整，
  尚不能说明24-task expert流形、checkpoint趋势或视频可预测性已经成立。
- Expert目标对同一task的不同视频恒定，因此它只定义policy-effective task-level manifold，不能
  自行保证Writer学习时间顺序或same-task video specificity。最终仍必须由one-shot correct/same/
  wrong/shuffled/reversed严格配对closed loop裁决，而不是只看expert重建误差或几何。
- 尚无新的strict rollout，历史single-checkpoint最好仍为v6-fast`143/400`；`>150/400`未完成。
  下一决定应先看full24三checkpoint geometry与development-train expert closed loop，再决定统一
  resume2000还是进入feature/meta-Writer阶段。

## 2026-08-07 Video Expert-Manifold task-expert A40 profile

- 独立task expert不是新的部署输入或task-ID route：teacher action只用于train24各自的policy-effective
  LoRA坐标，后续Writer仍只能从language+action-hidden video生成LoRA。
- clean`174d292`在`gpu01:0`以B16完成fresh0→1、same-root resume1→3和独立contiguous0→3。
  三步loss=`.221725/.283785/.259915`、gradient norm=`.029505/.032996/.035243`，峰值
  allocated/reserved=`15,082,000,384/21,313,355,776` bytes，说明46GB A40有充分余量且
  不需改变task data、batch或LoRA topology。
- resume与contiguous的loss/grad/LR/query cursor完全一致，step3 adapter逐字节一致；checkpoint
  loader的CPU/CUDA RNG设备语义已在真实失败中根修。现在可在六张实时空闲卡上启动24-task
  expert bank的统一step1000 formal，不需DDP/NCCL。

## 2026-08-07 K4 Phase-Aligned最终根因与Expert-Manifold方向

- K4 Phase-Aligned的strict curve=`88/108/80/99`，union/intersection=`157/36`；恢复历史
  v6高层semantic/procedure图并增加K4对齐仍没有解决checkpoint能力轮换。
- winner五臂=`108/115/94/101/121`：correct显著优于wrong，但不优于same且低于
  reversed。内部wrong/shuffled/reversed对BA的relative-L2约`.330/.188/.165`，所以
  视频已真实改变LoRA与closed loop；不能用“视频被忽略”解释失败。
- 最早失败是视频条件到policy-effective parameter manifold的对齐。LoRA norm中位
  `91.12`但stable rank=`1.00021`、top energy=`.99979`；后50步factor/program retention
  仅`.0463/.0436`。functional credit学到高增益方向，却不能稳定组织多task参数。
- 新方向保持视频为唯一dynamic value：frozen π0.5 joint-video hidden减text/no-image
  baseline，不设language-only output path。先以24套task-local rank-16 experts建立可闭环验证的
  参数流形，再用168个topological chunks与chunk/rank axial memory从one-shot video重建。
- 真实policy inference实测调用456个Linear，若全部rank16则为37,812,736个LoRA参数；
  首轮仍保留已达143的38-target/1,287,168-parameter拓扑，用task-expert closed loop
  先分离“credit target”与“topology范围”，不一次混入两个大变量。

## 2026-08-07 Grounded-Video最终根因与K4 Phase-Aligned v6假设

- Grounded-Video Expert的负结果排除了三个常见解释：视频没有被完全旁路，LoRA不再严格rank1，
  完整parameter isolation也确实显著提高局部gradient retention。尽管如此，correct曲线最高仅88且
  五臂无margin，说明“视频决定hard parameter owner”不是condition-to-policy credit的充分结构。
- hard top1 route把同一task的credit收进一个完整expert，却同时切断v6中跨task可共享的高层语义与
  compiler迁移。该方法既没有稳定积累，也没有让wrong/order扰动与正确程序方向对齐；继续加expert、
  rank、scale或训练步数没有证据基础。
- 历史v6-fast的143仍证明trainable PI05 high-level encoder、task-grounded transition、causal
  Procedure和320-slot compiler组合具有当前最强closed-loop杆杆。此前K4各版都替换了其中至少一段，
  所以新的最小有据重构是保留完整v6语义写出，只在多演示组合接口解决one-shot歧义。
- K4 Phase-Aligned设计让每条video独立对齐到phase16；Core在四条等权语义集合上提取共同内容，
  Procedure只在video内保持因果、再按phase平均。它在program空间组合演示而不是平均LoRA，避免跨
  video假transition，同时保持AS/RL同一部署图。A40真实profile已保持K4/B20/B2/full24并exact-resume，
  peak reserved`47.02GB`、step3五个owner全可达；下一证据必须是fresh闭环曲线。

## 2026-08-07 Grounded-Video formal训练机制证据

- grounded-video top1 route的identity-fresh formal完整完成200步，没有工程或数值失败；因此后续
  四点rollout可以直接裁决“高层视频寻址+独立完整expert”是否改善single-checkpoint行为，而不再
  混入warm-start、batch缩减或训练中断。
- 487,415,808参数并未在46GB A40上迫使改变scientific batch：logical B20、K4、full24一次Adam、
  policy B2与16-frame chunk保持不变，最大reserved`42.73GB`。这确认参数增加本身不是A40适配
  障碍。
- 冻结route的局部共存机制达到预期：按每个expert实际拥有的task数归一后，Reader/axis八个25步
  窗口retention中位为Reader `.580/.598/.558/.531/.511/.483/.516/.453`，axis
  `.518/.532/.509/.489/.499/.493/.499/.486`。末窗仍接近`.5`，不再是shared map约1/24的
  credit cancellation；global padded值下降只反映八个互斥owner blocks的零梯度占位。
- functional loss从`.15038`降到`.10098`只证明AS目标可优化，不用于选checkpoint或推断rollout。
  接下来仍必须用50/100/150/200 strict paired correct400、breadth、换手和winner五臂决定视频
  route是否把正确高层语义组织到closed-loop有效方向。

## 2026-08-07 Grounded-Video Route input-only裁决

- 冻结PI05 final multimodal task-token video innovation在train24×50上形成稳定的高层视频地址：
  初始8-center top2对每task 250个随机K4的primary/exact/overlap为
  `1.0/.984833/.992417`，证明不是退回language-only寻址。
- top2的secondary并非可靠owner：task35在batch4与singleton路径间从expert0翻到expert5，虽不改
  primary，却会替换占`.5`权重的一整套Reader+axis参数。严格门因此没有被放宽。
- input-only证据表明primary在6,000个随机K4 set和24个batch/singleton对上全都一致，所以最终
  route改为top1 one-hot，而不是增加数值容忍或用rollout/outcome选route。最终8-expert usage=
  `2/6/7/3/1/1/2/2`，无空置，随机K4与batch合同均为`1.0`。
- 该改变加强了视频高层语义对parameter owner的因果作用：同一K4 address唯一选择一套完整
  Writer map，20-group时序trace仍生成全部LoRA value；AS与未来RL共享同一部署图。
- 六卡A40 profile确认该图在真实longest105、K4/B20/B2下可训练并exact-resume：三步均finite、
  0 clip/OOM，step2起8 experts的Reader和axis共16 blocks全部有实际更新；训练期每个expert的
  非零task-gradient集合与input-only route artifact逐项一致。peak reserved`45.24GB`，没有通过
  减batch或关闭科学路径适配46GB显卡。

## 2026-08-07 Sparse Semantic-Expert正式负裁决与视频寻址根因

- routefix identity-fresh formal完整curve=`74/74/78/75`、breadth=`6/5/5/5`；winner
  macro150五臂=`78/85/90/83/92`，correct最低。wrong/reversed相对correct gained/lost=
  `20/8`、`26/12`，五臂union/intersection=`123/55`。完整parameter isolation没有形成absolute
  或correct-video margin。
- 这不是视频被忽略。production-batch内部root为
  `runs/outputs/pi05_as_writer_k4_sparse_semantic_expert_trace_m2p_routefix_internal_macro0150_507ae6e_20260807`；
  wrong与reversed的`physical/Reader/program/BA/action`relative-L2中位分别=
  `.309/.194/.209/.279/.050`与`.335/.197/.205/.278/.044`。correct LoRA
  norm/stable-rank/top-singular-energy=`44.79/1.412/.791`，top4 target energy`.489`。
- parameter ownership机制本身有效：Reader四窗full-task local retention=
  `.2847/.2542/.2187/.2053`，axis=`.2419/.2146/.2034/.1959`，远高于shared版末窗约`.05`。
  但route完全由target language决定，所以同一task的correct/same/wrong/shuffle/reverse始终进入
  同两个parameter maps；wrong/order较大扰动反而偶然增强source policy已有能力。
- 最早接口因此从“是否隔离完整experts”前移为“高层视频语义是否拥有parameter address”。下一
  authority用冻结PI05 multimodal task-token video innovation产生K4 route，同时保留现有trace
  dynamic value。不能再通过加expert、加步数、调rank/scale或直接reward绕过。
- 旧raw visual feature cache的input-only CPU先验支持K4视频寻址可稳定：对train24 task-mean中心化
  frozen vision descriptors做8-center route，250个K4 sets/task的primary/exact/top2-overlap约
  `1.000/.982/.991`；这只证明视频地址可行，不作为新grounded multimodal route的效果或最终
  artifact，新route仍必须由train24×50当前descriptor重新生成并过门。

## 2026-08-07 Sparse Semantic-Expert实现与route审计

- 首次formal到macro28时，expert-local Gram给出可复现的route contract冲突：task9在expert1有
  material梯度、expert7为零，而旧artifact声明secondary=7。不是训练漂移，而是冻结text
  backbone在24-language BF16 co-batch与逐task runtime之间产生足以翻转secondary近邻的数值差。
  该formal/root与旧profile均已主动否决，不resume。
- task anchor现对每条exact language独立forward，route generator以singleton anchors拟合并
  复核co-batch调用；最大anchor差`1.49e-8`且top2完全相同。新primary usage=
  `5/7/6/1/1/2/1/1`，top2 usage=`7/11/6/5/4/4/3/8`，无expert空置。
- clean`bbe5cf2`六卡新profile完成fresh0→1与exact-resume1→3；step1从真实Gram反推的八组
  active task sets与route artifact逐项一致。三步finite、0 clip/OOM，step2起16 blocks全可达，
  peak reserved`45,592,084,480` bytes；fixed-route工程合同重新闭合，但仍不提前说明closed-loop。
- 八个完整Reader+axis owners真实enumeration为`487,415,808` trainable：Reader
  `218,980,352`、axis M2P `268,435,456`。language route只选owner，K4 trace仍是全部动态value；
  top2 memory在decode前组合成一套LoRA。
- tiny dense-reference与conditional gather/scatter数值一致；zero trace严格zero memory，未选expert
  无梯度，selected experts在zero-output bootstrap拥有预期梯度入口。下一证据必须来自A40真实
  fresh/exact-resume profile，不能由参数隔离本身宣称性能改善。
- 六卡A40真实profile已通过：fresh0→1、exact-resume1→3的三步均finite，0 clip/OOM；step1
  八个Reader owners可达，step2起16个Reader/axis blocks全部非零。峰值reserved`45.59GB`，说明
  top2 activation确实适配46GB卡但余量有限；这只证明工程/梯度合同，不证明closed-loop提升。

## 2026-08-07 Evidence-Factorized五臂/内部裁决与完整expert门

- macro200五臂=`84/85/66/83/78`；correct相对wrong gained/lost=
  `36/18,p=.0198343`。视频task identity真实影响closed loop，same与order controls却不降；
  结论是“视频有效但任务程序未对齐”，不是视频被忽略。
- 8-task内部root为
  `runs/outputs/pi05_as_writer_k4_evidence_factorized_layer_trace_m2p_macro0200_internal_refs1_r6_8c8b502_20260807`。
  same/wrong/shuffle/reverse的`physical→direction→attention→Reader→BA→action`中位分别为
  `.135/.995/.477/.134/.197/.050`、`.310/1.319/.647/.432/.620/.155`、
  `.251/1.375/.700/.406/.511/.122`、`.335/1.414/.650/.337/.443/.180`。
- direction/physical Reader分支能量中位`14.65/27.52`、cosine`.510`、合并能量ratio`1.484`；
  direction-only与physical-only相对full的BA变化`.838/.659`，两支均material。no-evidence的
  BA/action变化`.066/.020`，说明evidence参与但不是单独瓶颈。
- LoRA norm/stable-rank/top-energy=`60.31/1.291/.847`，identity action effect`.373`；高增益和
  视频差异已到policy。最后50步Reader/axis retention却只有`.05527/.04650`、cosine
  `.00954/.00266`、负pair`.47464/.48007`。最早剩余故障是完整共享condition→policy map的
  multi-task parameter ownership，不是继续改表示、放大LoRA或训练更久。
- 新authority因此打开固定semantic top2的完整Reader+axis experts，而不是恢复只隔离final
  heads的Direction Store。language只寻址，K4 traces仍是唯一动态value；该结构对AS与未来RL
  同样适用。

## 2026-08-06 Evidence-Factorized Trace fresh formal训练证据

- identity-fresh formal0→200自然完成：200 finite macros、96,000 action queries、19,200 K4
  action-hidden video conditions、8 checkpoints、0 clip，source trainable=0且validation/test
  action reads=0；wall=`7272.774s`，peak allocated/reserved=
  `18,203,289,600/20,304,625,664` bytes。
- 四个50步窗口的full24 raw gradient retention/cosine/negative-pair中位为
  `.10601/.06078/.36957`、`.08578/.05152/.38949`、`.06065/.02493/.44746`、
  `.05227/.00727/.47645`。双value/evidence Reader在前半段保持了比raw-only稍好的共存，
  但晚期仍接近正交抵消；它不能提前证明closed-loop有效或选择checkpoint。
- 当前固定裁决仍是macro50/100/150/200同panel strict correct400；只有四点行为、breadth、
  换手和single-winner五臂/内部传递能判断direction与physical evidence是否真正互补。
- 四点随后完成为`74/59/65/84`、breadth=`6/6/5/5`，相邻gained/lost=
  `19/34,23/17,29/10`，union/intersection=`122/32`。双证据没有形成unit/raw两个端点的
  闭环互补，macro200虽后段回升仍低于raw-only85与unit-only99；需用winner五臂和内部
  branch/readout分析定位是shared attention、fusion还是更晚的shared credit最先失败。

## 2026-08-06 从destructive normalization转向Evidence-Factorized Trace

- 两个同topology fresh反事实形成闭环：unit direction保留task specificity但将low-energy
  order modes放大约140倍；global raw amplitude使前150步task-gradient更共存，却把
  correct/wrong从`99/57`压成`85/80`并把Reader effective groups从13.97压到10.63。
- 两者都能产生高增益、非纯rank1 LoRA，故不能再把direction与evidence reliability折成一个
  scalar amplitude。下一设计从同一raw DCT显式保留normalized direction与physical value，
  用group/frequency energy share及K4 leave-one-out direction consensus只作key evidence，
  shared attention后用vector fusion组合两路content。
- 该表示不依赖AS loss或LIBERO outcome，未来reward/RL使用同一video→LoRA图；只有新Reader
  闭合而shared credit再次近1/24抵消时才打开sparse experts。

## 2026-08-06 Energy-Preserving Trace fresh formal训练证据

- identity-fresh formal0→200自然完成：200 finite macros、96,000 action queries、19,200 K4
  action-hidden video conditions、8个checkpoints、0 clip，source trainable=0且validation/test
  action reads=0；wall=`7373.955s`，peak allocated/reserved=
  `18,096,449,024/20,478,689,280` bytes。
- 四个50步窗口的full24 raw gradient retention/cosine/negative-pair中位为
  `.12497/.07199/.35870`、`.08564/.04393/.42391`、`.08050/.02884/.44022`、
  `.05079/.00555/.48007`。相对上一版，保留真实频谱能量显著改善前150步的task-gradient
  coexistence；最后50步仍向近正交抵消退化。
- 这是representation修复有效的机制证据，不是closed-loop结论。functional loss从约`.15`
  降至`.10511`也不得用于checkpoint选择；四点strict correct400仍是下一裁决。
- 四点随后完成为`67/83/74/85`、breadth=`5/6/7/7`；相邻gained/lost=
  `28/12,18/27,28/17`，union/intersection=`122/40`。虽然前150步gradient coexistence改善，
  closed-loop反而全线低于上一版对应`69/99/88/94`，故“弱高频只是应压低的噪声”不成立。
- 当前尚不能断言旧逐频率单位化正确：它可能同时放大噪声和提供Reader所需的弱判别方向。
  macro200五臂与内部幅度/path分析将区分视频身份、顺序线索和policy leverage何处衰减。
- 五臂最终`85/85/80/74/87`；correct相对wrong只有gained/lost=`25/30,p=.590`。上一版
  correct/wrong=`99/57`的task specificity被全局raw amplitude消除，而不是转化为更稳的能力。
- same/wrong/shuffle/reverse从trace到Reader到BA的中位差异为
  `.135/.030/.049`、`.310/.297/.478`、`.251/.060/.092`、`.335/.079/.117`；旧版为
  `.995/.135/.167`、`1.319/.547/.715`、`1.375/.406/.450`、`1.414/.342/.452`。
  物理幅度保留把弱的task/order direction在最早输入处就压低，M2P没有恢复它。
- LoRA norm58.71、stable rank1.410、top singular energy.793、identity action effect.581，
  且effective group memory只有10.63（旧版13.97）。失败是direction/evidence support被破坏性
  绑定，不是LoRA质量仅表现为低rank、低能量或共享Writer参数量不足。

## 2026-08-06 K4 Layer-Trace五臂与内部裁决：视频有效，频谱幅度被破坏

- macro100五臂`correct/same/wrong/shuffled/reversed=99/92/57/94/105`；correct相对wrong
  paired gained/lost=`61/19,p=2.73e-6`，证明video task identity已进入LoRA与closed loop。
  same、shuffled、reversed却与correct同档，顺序差异未成为有效任务程序。
- trace→reader→axis4→effective BA→fixed-action的same条件relative-L2中位约
  `.995/.135/.112/.167/.040`，wrong约`1.319/.547/.528/.715/.244`；LoRA norm中位
  `48.28`、stable rank`1.34`、top singular energy`.836`。因此不是忽略视频、低增益或
  layer/group collapse。
- 原始DCT trace的DC energy fraction中位`.95664`，high8总fraction仅`.003592`，
  effective frequencies仅`1.092`。旧实现对每频向量独立单位化，使high8占约一半
  token energy，相对放大约140倍；最弱频幅度相对最强频放大约73倍。
- reversal只翻转奇DCT项符号，却因上述单位化将normalized trace推到近正交
  relative-L2`1.414`，形成强BA/action操纵而closed-loop不受损。所以最早故障是
  temporal spectrum amplitude semantics，不是立即增加experts。
- 下一方法保留K4与完整video→LoRA链路，只用每视频一个全局scalar匹配旧总
  trace energy，保留原始group/frequency间相对能量与符号。这一接口对AS和未来RL相同。

## 2026-08-06 K4 Policy-Layer Trace四点：层对齐只有局部增益，task换手未解

- macro50/100/150/200 strict correct400=`69/99/88/94`、breadth=`5/6/6/6`；每点均为
  400 rows、42 shards、9 workers exit0。相邻gained/lost=`42/12,28/39,28/22`，四点
  union/intersection=`145/37`，K4 set、state、env seed与policy-noise common prefix为0 mismatch。
- 逐task为`4/0/0/33/29/2/0/1`、`5/3/0/34/41/12/0/4`、
  `12/1/0/34/26/13/0/2`、`15/1/0/34/27/11/0/6`；Goal-3与Spatial-1四点始终0，
  macro100→150净失11且150→200只恢复6。single winner macro100=99不是单调/双曲成熟曲线。
- 相对旧K4同点`70/94/99/108`，新layer alignment为`-1/+5/-11/-14`。它没有把错误的视频
  表示方向变成可共同累积的policy更新；结合最后50步retention`.04573`、pair cosine`.00400`
  与negative pair`.47464`，共享condition credit cancellation仍是必须检验的最早接口。
- 不能仅凭行为提前等同为“视频无效”：当前macro100还必须完成same-task/wrong/order控制和
  raw trace→reader→axis M2P→BA→action内部分析。只有这些接口闭合后才能按预注册证据打开
  sparse sharing/experts，不能用loss、续训或checkpoint挑选修补99分。

## 2026-08-06 K4 Policy-Layer Trace fresh formal训练证据

- 独立fresh0→200自然完成且工程合同闭合：200 finite macros、96,000 action queries、19,200
  action-hidden K4 videos、8个完整checkpoints、0 clip/OOM/nonfinite；source trainable=0且没有
  validation/test action读取。wall=`7350.114s`，peak reserved=`20,478,689,280` bytes。
- full24 raw gradient coexistence在训练前半段相对旧K4有实质但非持续的改善：四个50步窗口的
  retention中位依次`.07229/.06074/.05466/.04573`，median cosine依次
  `.03067/.01741/.01100/.00400`；最后50步已经接近旧K4的`.04326/.00038`正交抵消极限。
- 这不能提前证明层对齐行为无效，也不能写成task drift已解。step200 functional loss为
  `.10194`，但项目既有证据已否定用functional loss选择closed-loop checkpoint；必须完成
  50/100/150/200同panel strict correct400后再结合LoRA/path与视频因果性定位最早接口。

阅读规则：本文是按日期追加的证据账本。历史段落里的“当前”“下一步”和GPU
权限只描述其日期当时的状态，不覆盖后续owner决定。活动状态以
`docs/a100_to_bci_migration_handoff.md`、`docs/active_session_handoff.md`和本文顶部
最新段落为准；
不得从早期段落恢复旧runner、旧架构或旧训练合同。

## 2026-08-06 从final-layer通用M2P转向policy-layer trace M2P

- K4上一轮已经证明video-common representation、顺序反事实和高增益LoRA有效，因此下一版
  不能忽略视频或退回language-only route。新的根因假设更早：旧descriptor把PI05 final
  hidden固定随机压到128维，再让24 tasks上fresh的256维M2P重建38-target policy topology，
  condition credit的共享抵消可能部分来自semantic-to-parameter alignment缺失。
- 参考SHINE的all-layer memory与layer/token交替M2P、Doc-to-LoRA的显式layer/module/rank输出
  组织，新设计直接提取冻结PI05 action expert 20组baseline-subtracted video traces，以K4×16
  temporal tokens读取`20×68×1024`memory，并按layer/parameter slot双轴通信后直接reshape
  public LoRA。训练objective、信息墙与未来reward接口不变，不引入监督专用auxiliary loss。
- 暂不直接复制8个完整experts，因为这会把同一未对齐输入分桶并引入约2.5亿fresh参数；若
  layer-aligned方法仍在group-wise梯度上接近1/24抵消，再以该证据设计稀疏共享。
- 首个A40 diagnostic定位到实现级幅度不连续，而非科学负结果：direct memory后axis FFN使用
  pre-LayerNorm，step1任意小的nonzero bootstrap在step2被归一到O(1)，functional loss从
  `.1504`跃升`58.93/96.82`。显存仅20.48GB、步时约34.7秒；根因与容量、视频trace、NCCL或
  full24无关。`ed4f46e`让FFN直接处理raw value并用小幅度2×合同验证zero邻域连续。

## 2026-08-06 K4四点与内部裁决：视频共同程序成立，shared credit仍近正交抵消

- strict correct400曲线=`70/94/99/108`、breadth=`6/6/6/7`；相邻gained/lost=
  `42/18,30/25,25/16`，union/intersection=`150/42`。曲线单调且breadth提高，但macro200
  仍低于v6-fast143和严格门`>150`，Goal-3四点全0，single envelope gap42。
- macro200 refs1内部分析严格确认不是忽略视频：set permutation与zero-video identity为0；
  same-task另一K4 set的Program/M2P/BA/action差异中位`.150/.137/.115/.016`，LOO为
  `.062/.055/.043/.009`，cross-suite wrong为`1.092/.987/.730/.151`，shuffled/reversed
  又将约`1.28/1.44`的Program变化传为`.253/.274` action变化。
- effective LoRA norm中位27.59、identity→correct action差异.254，未重演Condition-Kernel
  的tiny-LoRA；stable rank约1本身也不构成失败，因为direct SFT同样低rank。
- 最后50步full24 task-gradient retention中位仅`.04326`，pair cosine`.000376`、negative
  pair`.49275`；Program、M2P shared、A/B heads均接近`1/24`正交极限。K4解决了one-shot
  视频共同性辨识和高增益写出，剩余最早接口是condition-specific credit在共享Writer参数
  中无法稳定共存。下一版必须保留K4 video-owned program，不续同一schedule、不warm-start。

## 2026-08-06 K4 formal与前两点rollout：早期acquisition回升但尚未过门

- fresh K4 M2P formal完整跑到macro200，训练规模为96,000 action queries与19,200条
  action-hidden teacher videos，0 clip/OOM/nonfinite；四视频联合条件不是训练期augmentation，
  每次task visit只生成一套完整LoRA并复用于B20。
- macro50/100 strict correct400为`70/94`、breadth均6；50→100 gained/lost=`42/18`，说明
  acquisition净增但已有明显能力换手。Goal-3与Spatial-1两task仍为0，当前证据不能宣称
  task drift解决，也不能以训练loss外推macro150/200。
- 每task的50个paired K4 sets覆盖全部50条teacher videos，每条set含4条且两个checkpoint
  使用相同set/state/env/policy RNG。聚合器最初只认识K1字段而在全部rollout完成后报错；
  根因是结果摘要层未迁移`teacher_demo_indices`，修复后直接聚合既有immutable shards，
  科学rollout、LoRA cache与success rows均未改变。

## 2026-08-06 从one-shot不可辨识性切换K4 video-owned invariant program

- owner明确纠正了任何“忽略视频”的错误方向：EMBER的任务就是从action-hidden teacher
  video提取可泛化的高层任务信息并生成LoRA；language只能ground和route，不能成为动态
  LoRA value的旁路。owner允许few-shot，因此当前方法不是削弱视频，而是增加跨演示统计轴。
- 历史one-shot把任务共同程序、单次初态、轨迹速度和偶然视觉细节绑在同一条video里；模型
  无法观测哪些变化应跨demo保持。K4让四条独立same-task videos在一次forward内共同决定
  一套LoRA，且不允许逐视频LoRA平均、挑选或ensemble。
- 新`InvariantProgramEncoder`把每条video的policy-aware时序descriptor保留为4个value
  tokens；首次cross read没有query residual，全部video values为零时Program与动态LoRA严格
  为零。task language只形成attention address，因此无法退化成language-only task adapter。
- 新M2P以38 policy targets×16 public rank lanes共608 tokens联合读32个invariant slots，
  在完整policy范围self-attend后由target-owned A/B heads写出。该设计直接针对此前
  Program差异到public BA/action被压低或碎片化的问题，不用SFT重建、rank/正交loss、scale
  trick或LIBERO outcome特化。
- 训练仍是24 tasks×B20的原full24 functional credit，但部署、AS与未来reward统一使用K4
  condition→Program→M2P接口；不读取held validation actions，不用functional loss选择
  checkpoint。CPU全仓`188 passed`确认身份、video-zero、set invariance、梯度、schedule、
  checkpoint和evaluation合同；性能结论必须等待fresh A40 formal与strict rollout。
- live A40 profile还发现了一个必须区分的优化时钟问题：若把profile的total axis压成3步，
  warmup会被压缩并让首步直接使用peak LR，step3 clip不能代表formal。改为正式200-step
  scheduler、仅early-stop到3后，LR为`1.154e-5/2.308e-5/3.462e-5`，三步0 clip且step2起
  invariant-program/M2P shared/A/B全部可达；因此最终seal的是新root，不使用早期诊断权重。

## 2026-08-06 Condition-Kernel formal负裁决：credit隔离成立但decoder增益坍缩

- clean `4038960`从functional identity完成fresh AS0→200：200 macros、96,000 queries、
  4,800 videos、wall=`3951.928s`、peak reserved=`19,344,130,048` bytes，0 clip/OOM和
  0 validation/test action reads。200步Gram全rank24、condition=`5.139--7.750`、cap scale
  始终1；predicted/observed Program update relative RMS在50/100/150/200为
  `.002184/.001731/.001718/.001304`，macro50后FactorHeads freeze violation为0。
- strict correct400曲线=`46/46/45/49`，breadth=`3/3/3/3`；adjacent gained/lost=
  `5/5,4/5,6/2`，四点union/intersection=`55/40`。共同state/video/env/policy seeds和
  policy-noise common prefix均严格配对。40个四点共同success中Goal-6占37、Object-1占3；
  macro200的49个success中Goal-6占42、Object-1占5、Long-1占2，其余5 tasks全0。低换手是接近source
  identity的静止，不是task漂移得到解决或多task能力单调累积。
- 六卡内部分析完整96 rows/6 payload，wall=`273.968s`、peak reserved=
  `19,277,021,184` bytes，0 action-wall reads。same-task demo1的fixed feature/Program/BA
  relative-L2约`.786/.784/.775→.767`，reversed/shuffled BA约`1.39/1.36`，Program
  centered/sample energy约`.347`；fixed地址、memory和方向传递没有抹掉视频内容或顺序。
- 真正坍缩在绝对policy leverage：LoRA norm中位仅`.1761→.1779`，比corrected direct SFT
  `35.7362`小约200倍。它的stable rank约`3.79→3.72`、top singular energy约`.28`、q/v
  B-column cosine约`.19/.205`，说明高rank和异质方向都真实存在；但identity、same-demo、
  reversed、shuffled的fixed-action效应一律只有约`.19--.24%`。漂亮LoRA几何再次不是充分条件。
- checkpoint update的task-mean energy fraction在Program为`.730/.718/.672`、BA为
  `.784/.781/.727`，低于旧Program-Credit `.830/.916`但仍占主导。explicit kernel确实
  部分解决condition credit的共享同向化，却不能弥补Program→LoRA basis本身的低增益。
- fresh zero-B使macro1 Program cotangent严格0；FactorHeads仅在0→50训练，freeze时public
  A/B RMS约`.01829/.000369`。之后M能按不同condition准确更新，但全部变化只能经过这一
  已冻结的弱decoder Jacobian。最早失败接口因此是policy-effective decoder cold start，
  而不是address、kernel、rank、scale或functional loss选择checkpoint。
- AS200未过预注册`correct≥120 && breadth≥6`，direct reward禁止实现/启动；不能延长同一
  bootstrap、调RFF seed/维度、挑checkpoint或用RL救活。精确汇总在
  `runs/outputs/pi05_as_writer_condition_kernel_memory_internal_all4_r6_2972f8f_20260806/experiment_analysis.json`。
  当时按owner要求暂停讨论；该暂停已于2026-08-06解除，结果现只作为K4设计证据。

## 2026-08-05 Condition-Kernel地址审计、实现与A40 profile

- action-hidden audit覆盖train24×50 videos及validation8×50 apply-only。固定1024维
  task×video RFF在全部50个no-replacement task schedules上均为rank24，最坏regularized
  condition=`7.5471`、最大off-diagonal=`.4270`；same-task video/cross-task demo0 feature
  距离中位=`.8718/1.4058`，reversed最小/中位=`1.1567/1.4064`。地址没有重复或顺序失活，
  且train/validation/test action与reward/outcome reads全0，因此按预注册规则固定feature、
  bandwidth、seed和P1024，不做held驱动的sweep。
- canonical AS Writer已原位替换为固定descriptor/address→完整83,886,080参数Program Value
  Memory→2,179,072参数fresh FactorHeads，总参数86,065,152。M不使用Adam，full24唯一owner
  在24×24 FP64 Gram上求解后做FP32 value write；FactorHeads只在macro0→50训练。旧v6 AS
  condition path、temporal parallel module与Program-Credit method-specific analysis runtime
  已删除，历史证据由Git/artifact保留。
- 六卡fresh0→1与同root exact-resume1→3通过。三步wall=
  `20.713/19.842/19.448s`，峰值allocated/reserved=`16,556,672,000/19,344,130,048` bytes；
  step1 Program cotangent为0符合zero-final-layer identity，step2/3变为
  `1.9946e-7/3.5717e-7`，predicted update RMS=`1.9684e-7/3.5240e-7`且cap scale始终1。
  Gram rank24、condition持续`7.523→6.632→6.023`，0 OOM/clip，六rank exact-resume、
  scheduler/sampler/RNG、1,440 queries/72 videos和0 validation/test action reads均闭合。
  因此46GB容量、固定地址、梯度可达性和resume不是正式AS障碍；profile权重弃用。

## 2026-08-05 Program-Credit机制负裁决与Condition-Kernel Memory决策

- clean`129cab6`的六卡只读机制分析已完成24 tasks×AS125/cycle1共48 rows、6/6 payload；
  wall=`272.876s`、peak reserved=`19,304,284,160` bytes，target-action与validation/test
  reads均0，全部GPU自然释放。532个冻结tensors逐元素不变，四个上游block确有微小更新，
  所以下述结果不是checkpoint、freeze或空梯度错误。
- train24 exact program cotangent在共享参数更新之前pair cosine mean/median=`.000107/0`、
  negative fraction=`.2464`、full24 energy retention=`.041874`，说明不同task的closed-loop
  credit本来几乎正交；一次共享Writer更新后的task-mean program delta却为
  `.5801/.6128`、negative fraction=`0`、retention=`.55537`。最早故障是condition-map
  Jacobian把不同credit压成公共更新，而不是raw task credit本身同向。
- same-task五video更新的task-mean energy fraction在program/BA中位=`.82990/.91623`，pair
  cosine=`.78826/.89916`；AS125→cycle1的same-task video centered/sample energy在program
  `.002153→.002149`、BA`.001154→.001178`。demo1/wrong/reversed/shuffled到program、BA、
  action的相对差异几乎没变，direct reward没有增加视频特异性。
- program→BA→fixed-action update relative-L2中位=`.006782/.004713/.002279`，decoder并非
  完全失活；400个held LoRA的BA变化中位`.005519`，gained/lost=`.004726/.004742`，
  retained failures反而`.005705`。stable rank`1.0000160→1.0000163`、top1与B-column也
  几乎不动，故不能再用rank/scale/decoder完全无响应解释。
- binary/semantic cotangent energy=`.00261635/.00003600`、cosine`.00184`，binary约
  `72.7×`主导。Program-Credit真实`+9`不是functional action loss或semantic tie-break造成，
  但共享condition map无法把它累积为task/video-specific policy方向，因此正式禁止cycle2。
- 下一唯一architecture为Factorized Condition-Kernel Program Memory：冻结foundation
  task hidden与policy-aware temporal video innovation，形成32×32 fixed RFF product feature；
  P1024每个coordinate拥有完整320×256 program value。full24用24×24 regularized Gram solve
  直接把program cotangent写入memory，使induced condition update可预测。fresh FactorHeads
  固定bootstrap到macro50后冻结，AS与reward都只更新同一M；这是objective-agnostic credit
  storage，不是监督auxiliary loss或LIBERO outcome trick。
- Direction Store只用language top2选择8个factor stores，video仍穿过共享trainable value
  path；新方法的地址显式含video，完整program value线性读取，且condition共享由固定kernel
  唯一决定。它不恢复历史store/router，也不加载任何旧Writer checkpoint。

## 2026-08-05 Antithetic Program-Credit架构与训练决策

- fresh formal cycle1与strict correct400已完成：cycle1=`106`、breadth5，相对AS125=`97/5`
  gained/lost=`18/9`、union/intersection=`115/88`、paired p=`.12208`。Long/Goal/Object净
  `+1/+1/+7`，Spatial总数不变但唯一成功从task1换到task3。它是真实但尚弱的共同改善，
  仍包含能力换手；因预注册续训门要求净增至少10，正式禁止resume cycle2。
- 400 rows、8×50、50条无放回videos、18/18 worker exit0且0 retry/error；state/env/video和
  共同policy-noise prefix严格配对。故`+9`不是漏行或面板不一致，也不能因“只差一条”改门。
- 一次full24 direct-program update使四个上游block参数relative-L2约
  `.000231/.000151/.000245/.000204`（Core/Visual/Procedure/compiler）；532个冻结Writer
  tensors逐元素不变。接下来只读机制分析要判断小幅held改善是video-program credit还是
  task-level校准，不能仅凭aggregate或train 54/96写成根因已解。

- clean`318b6f6`首次六卡原规模profile在68/96 rollout、任何update/checkpoint之前由共同
  随机数检查终止。task38两组pair的env seed与各104个policy-noise seed完全相同，但首帧
  hash不同；排除了Writer、LoRA、policy noise、NCCL和显存层。
- 真实LIBERO最小复现显示：同一默认hard-reset env重复`seed+reset`的MuJoCo XML不同，物体
  位置最大差约3--4cm；仅恢复47维sim state也不能逐像素恢复旧模型。两个独立persistent
  env若从构造起保持相同reset count/seed，即使中间执行不同动作，连续三次reset的XML、state、
  agent/wrist pixels都逐字节一致。因此根修是per-task plus/minus lockstep lanes，而不是放宽
  equality gate、容忍图像误差、固定init state或换成监督proxy。
- 双lane v2实现后项目正式activation全仓`221 passed`，compileall与diff check通过；失败
  root仍无参数更新，下一科学证据只能来自clean commit上的全新原规模重放。
- clean`f3f6b15`全新六卡v2 profile及exact-resume完整通过：cycle0/1各96 rollouts、24 task
  credits、48 CRN pairs且0 mismatch、54 successes；binary-discordant pairs=`6/2`，非零credit
  pairs=`24/22`。四个上游block两轮梯度都非零，冻结semantic encoder/FactorHeads/source
  policy均0；wall=`431.709/431.367s`，peak reserved约`19.31/19.33GB`，0错误。由此接受
  common-random reset、A40容量、direct-program梯度、all-rank ready和exact-resume机制，
  仍不把profile reward或权重当性能证据。
- Policy-Lane完整负证据把最早失败从LoRA容量/几何推进到conditional credit：约10个有效
  output lanes、stable rank`1.34--1.54`、direct-SFT量级跨层能量都没有带来闭环，same-task
  video在hidden/BA却仍只有`.05%/.02%`。因此下一方法不增加结构容量，而直接对生成LoRA
  之前的policy program分配closed-loop credit。
- 选择v6 compiler的`320×256`输出作为高层动作：它按18 layers×16 ranks加action-in/out
  组织，并共同驱动q/v/action的A/B FactorHeads，比raw LoRA或某个factor tensor更接近完整
  policy decision。确定性`forward`只拆成等价的encode/decode接口，不改变AS函数。
- K4改为两组共享environment/policy randomness的`+/-`Rademacher program扰动；success
  不同时用严格二值差、双成功为0、双失败才用冻结action-free semantic progress差。由pair
  差直接构造program cotangent并反传Core/Procedure/compiler，不再经过executed-action
  CFM ratio、PPO/SPO或teacher action。
- 唯一cold start为同一fresh v6 AS125，不是historical v6-fast best或reward checkpoint；
  FactorHeads、semantic encoder、source policy和normalization冻结。该选择把完整方法定义为
  generic source→fixed AS stage→direct program reward，并保持IL/RL都可用的根接口。
- canonical实现已经原位恢复v6并删除Policy-Lane/旧CFM executable family。确定性forward
  与encode/decode逐tensor等价；pair方向/credit、artifact/randomness cursor、四上游block
  ownership、冻结decoder/observer、full24多卡与checkpoint roundtrip均有聚焦合同。项目正式
  activation下全仓`220 passed`，py_compile/diff check通过；AS125 checkpoint身份定向成立。
- 失败run只证明旧单env配对合同无效，没有产生本方法性能结果。双lane v2修复后仍必须在
  全新root重放独立A40 cycle0→1/resume1→2，不能复用68条旧ledger或把修复写成性能改善。

## 2026-08-05 PWAD正式负裁决与Policy-Lane Coupled Hyperdecoder决策

- Policy-Lane在clean pushed`2aeb22a`完成BCI六卡longest105/logical-B20/full24 profile：
  三步`33.457/31.024/31.007s`，峰值allocated/reserved=`36,168,858,624/
  47,053,799,424` bytes，0 OOM/clip/nonfinite。step1只有Policy-Lane梯度，step2起五个
  声明主块全部非零；独立fresh0→1→exact-resume1→3也闭合optimizer/scheduler/RNG/
  sampler/六rank state。由此排除46GB容量、梯度路径与resume机械性障碍，但尚不构成
  closed-loop性能证据。
- Policy-Lane正式fresh0→200随后在clean pushed`244b677`完整结束：200 finite macros、
  96,000 queries、4,800 one-video conditions、8个checkpoint、0 OOM/clip/nonfinite/stall，
  validation/test action reads=0。该证据证明49M Writer能在46GB A40合同内稳定端到端训练，
  但不把functional loss下降写成闭环改善。
- 固定四点strict correct400=`70/63/37/61`、breadth=`6/4/6/6`；相邻gained/lost=
  `17/24,14/40,40/16`，union/intersection=`117/14`、single envelope gap=`47`。四点
  机械合同完整且严格配对，故低absolute与macro150崩落不是漏行、视频重用或worker故障。
  macro50 single winner仅70，正式禁止续400。
- 训练ledger的same-task相邻32维CountSketch方向在151--200段显示：Policy-Lane block
  mean/median cosine=`.0416/.0462`、negative fraction=`.4350`；PWAD policy-atom为
  `.0907/.0976/.3567`，历史v6 factor在76--125为`.1134/.1244/.3317`。该跨架构比较受
  block维度与sketch方差限制，只作为“扩大独立输出后credit复现可能更差”的待验证信号；
  后续真实内部分析确认输出容量打开，但不把CountSketch差异单独当因果量。
- clean`3869d20`四checkpoint内部分析完整封存96 rows、24 tasks、6 ranks，wall
  `318.446s`、peak reserved=`19,295,895,552` bytes，0 target-action/validation/test reads。
  storage effective lanes约`15.96--15.97`，condition output effective lanes=
  `10.85/9.87/9.57/9.80`，hidden row stable rank约`4.15--4.41`；容量没有失活。
- effective LoRA stable rank=`1.336/1.409/1.507/1.542`、top singular energy=
  `.809/.766/.727/.707`。四点400个held LoRA的精确跨layer复核给出q/v signed cosine约0、
  energy CV=`.75--.83/1.03--1.15`、top4 energy=`47--52%/58--61%`，与两套direct SFT
  `.705--.937/.707--1.052`及`46--59%`处于同一量级。Policy-Lane真实修复了PWAD/v6的
  层同向与伪rank问题，closed-loop却更低，故这些几何不是性能充分条件。
- cross-task hidden centered/sample energy从`.503→.660`，pair cosine从`.488→.313`；
  same-task video hidden/BA centered energy却只有`.046--.059%/.017--.023%`。macro50
  demo1/reversed/shuffled的BA relative-L2=`.0176/.0281/.0133`，fixed-action=
  `.00577/.00977/.00597`。任务分离增强而视频credit不增强，符合functional目标在正确
  policy language与同task独立query下可由task/common adapter满足的不可辨识捷径。
- Policy-Lane到corrected Source-SFT的effective-BA cosine overall为
  `.328/.328/.311/.307`、action为`.254/.281/.286/.287`；PWAD则overall从`.278`升到
  `.432`但correct只`77→80`。SFT方向相似度、rank、层能量外观与闭环均不单调对应。
  正式关闭继续增加lane/store/rank、调scale、强制SFT profile或续训的方向；下一方法必须
  直接重做Writer/LoRA生成层的closed-loop credit transport。

- PWAD fresh0→200训练健康完成，strict correct400=`77/71/80/80`、breadth=`5/6/5/5`；
  相邻gained/lost=`19/25,21/12,16/16`，四点union/intersection=`115/44`、single
  envelope gap=`35`。它既没有接近v6-fast143，也没有形成单调或共同累积能力，禁止
  resume400。
- 64/64 storage atoms active，storage effective count=`63.62→63.93`；condition combined
  effective participation=`50.50→54.19`，storage-weighted=`47.67→53.49`。因此负结果
  不能归因于dead atoms或K64容量没有使用。
- A/B mixing mean stable row rank四点约`1.000002`、top singular energy约
  `.999997--.999998`；public effective LoRA stable rank约`1.0000002`、top energy约
  `.9999998`、q/v B-column cosine约`.999998`。macro200 coordinate-query本身stable rank
  `10.73`，A/B key projection rank99=`58/59`，说明参数有容量，但condition readout把16个
  public lanes压成同一方向。
- same-task video mixing与effective BA centered/sample variance仅约`.022--.054%`与
  `.022--.047%`。identity→demo0 fixed-action从`.366`增至`.510`，demo1/reversed/shuffled
  则整体缩小；action energy share从`.1197%`降到`.0136%`，q始终约`88--89%`。PWAD
  学到强task/common q-dominant adapter，而不是视频条件program。
- direct SFT的mean target stable rank也只有`1.505/1.517`，所以不把near-rank1本身写成
  根因。区别是SFT跨layer方向与能量组织稳定；PWAD的A/B atom又经独立mixing，使真实BA
  包含所有`B_j A_k`交叉项，完整policy direction从未成为实际存储单位。
- 下一architecture让16个public lanes分别以同一32维condition hidden共同生成全部38
  targets的A/B向量。它保留lane内policy-wide协调、给lane独立输出ownership，并取消两套
  coefficient system；不强迫正交/高rank，也不复制SFT权重。AS与RL可对同一LoRA端到端
  提供credit，完整authority见
  `docs/action_forecast_writer_policy_lane_hyperdecoder_design.md`。
- canonical实现现已删除PWAD executable module/config family并原位切换到policy-lane。
  真实每lane A/B输出宽度=`37,920/42,528`，hyperdecoder=`41,320,448`参数，完整Writer=
  `49,041,664`。84项聚焦Writer合同确认identity、condition写出、BA梯度阶段、source
  freeze、checkpoint拒载和lane分析；没有借参数增长加入新输入、loss或并行runtime。

## 2026-08-05 Policy-Wide Atom Dictionary架构决策与CPU合同

- direct SFT表明单target LoRA可以低rank但跨layer/target能量组织稳定；Target-Spectral说明
  强制升rank会破坏有效方向，Direction Store/Target-Owned说明分开task或target存储也不会
  自动形成闭环协调。Tangent-Basis又否定局部basis旋转为主要漂移根因。因此最早未检验
  接口是：以完整policy方向为存储单位，并让一个condition坐标共同组合全部targets。
- 新Writer学习K64个跨38 targets共享索引的rank-one atoms；condition从Core/Procedure生成
  rank16×K64的signed A/B mixing。每个target保留自己的A/B向量，但同一atom index与mixing
  横跨q0--q17、v0--v17及action projections，兼顾target-specific内容与policy-wide协调。
- D_A/D_B exact-zero保证fresh public A=template、B=0。真实BA functional loss聚焦测试
  验证第一步只有D_B，B打开后composer/Core/Procedure与D_A可达，A atoms打开后A-side
  mixing可达；不是通过辅助loss或手工非零初始化制造路径。
- canonical参数量13,033,728；新launch/checkpoint family拒绝v6 checkpoint。clean`60e45f8`
  live六卡longest105 profile三步均finite，峰值allocated/reserved=`35,024,829,440/
  44,883,247,104` bytes，0 OOM/clip；step1只有policy atom，step2起semantic/Core/Program/
  composer/policy atom五block全部非零，source policy trainable=0。
- 独立fresh0→1后首次resume由restore validator fail-fast，因为family已在save/schema owner中
  建立但漏入cycle-normalized optimizer合法集合；0新增训练行。补齐唯一白名单并加回归后，
  同一六卡step1→3成功，累计1,440 queries、72 videos、六rank state与scheduler/RNG/data
  cursor完整。该问题是checkpoint schema ownership缺口，不改变PWAD科学机制。

## 2026-08-05 SFT-Anchored Tangent-Basis formal负裁决

- clean`059d40f`从v6-fast macro400 warm-start完成独立formal cycle0→1：96 rollout、
  61 successes、11 mixed、5 all-failure，two finite updates，五个系数侧block可达，
  basis/semantic encoder冻结，wall`2046.03s`、peak reserved`19.478GB`、0 OOM/watchdog。
- strict correct400为`142`，baseline v6 macro400为`143`；严格paired gained/lost/
  retained/both-fail=`20/21/122/237`，`p=1.0`，breadth`6→7`，union/intersection=
  `163/122`。Spatial由`3→6`并打开Spatial-1，但Long/Goal/Object分别净`-1/-1/-2`。
- aggregate只净降1掩盖了41个state换手。固定8个factor-output basis不能阻止能力轮换，
  因而basis旋转不是task drift的充分解释；更早问题仍在共享condition coefficient/composer
  的credit分配，或固定dictionary本身的policy覆盖。
- 该分数继承自143分的历史v6 SFT Writer，不能与fresh架构0→142混写；cycle1只有correct
  arm，不能声称视频特异性改善。预注册续训门未过，禁止cycle2与补跑一个暖启动主线。
  下一设计必须从functional identity fresh训练LoRA generator，RL只作后续闭环校准。

## 2026-08-05 SFT-Anchored Tangent-Basis profile通过

- clean`2f934bd`的24×K4 profile完成61/96 successes、11 mixed、5 all-failure；两轮
  full24 update的grad norm为`.004373/.003961`，ratio有限且clip近零。五个预注册
  trainable block两轮均可达，observer grad0，5/5 all-failure tasks均产生LoRA gradient。
- macro400→profile cycle1逐张量比较：8个policy basis和440个semantic-encoder tensors
  完全不变，恰好只有76个系数侧tensors变化，五block内所有这些张量都变化。
  因此IL dictionary/RL coefficients分界已在真实optimizer step层面成立，不是只有合同标签。
- wall`2033.38s`、peak reserved`19.48GB`、0 OOM/watchdog/action-wall reads；两轮都先
  6/6 CUDA-ready再进NCCL，cycle1全状态checkpoint完整。该证据只说明可训、可封存，
  不说明held closed-loop提升；下一裁决必须来自fresh formal cycle1的strict correct400。

## 2026-08-05 SFT-Anchored macro400进度信号通过机制门

- 24×K4只读diagnostic得到61/96 successes、11 mixed、8 all-success、5 all-failure；
  Spatial/Object/Goal/Long分别`17/17/16/11`，不是单suite成功造成的门通过。
- mixed task的success utility更高`11/11`，success/failure pair AUC`.91429`；all-failure
  utility range `4/5`≥`.05`、中位`.27273`，说明冻结observer同时覆盖binary ordering与
  failure-only credit，并非只复制outcome身份常量。
- correct视频对wrong/shuffled/reversed胜率`1.0/.90164/1.0`，margin中位
  `.55919/.37889/1.53747`；all-failure utility与pixel RMS Spearman`.48421<.8`。这比AS125
  observer提供更强的视频内容/顺序特异性，但只裁决credit可用，不能推断RL更新有效。
- 32-frame推理peak reserved`19.29GB`，0 optimizer/backward/checkpoint/action reads；因此
  打开one-cycle gradient profile以裁决46GB训练显存与冻结basis后的梯度可达性。

## 2026-08-05 参数hybrid因果分解与SFT-Anchored Basis决策

- clean`67b245a`六卡只读分析覆盖24 train tasks、demo0--4、reversed/shuffled、四参数臂
  与8-task fixed action；24/24和8/8完整，wall`333.52s`、peak reserved`19.365GB`、
  0 target-action/validation/test reads。
- AS125→cycle2完整BA变化中位`.02713`。factor-output-only与upstream-only相对完整更新的
  `recovery/cosine/residual`中位为`.614/.692/.727`与`.725/.795/.611`，BA层看似
  upstream更重要；但action变化中位`.00805`时两臂变为`.691/.893/.489`与
  `.494/.795/.668`，共享factor-output的较小BA变化反而更进入policy敏感方向。
- action贡献并不跨task一致：Spatial/Object和Goal-29由factor-output明显主导，Long-39
  upstream残差更低，Goal-20两臂都很差。该suite-dependent leverage与held能力换手一致，
  说明问题是共享decoder与condition composition交互，不是factor raw gradient或位移单独
  过大。
- same-task video variance仅从`.0013992`变`.0013899`；factor-only几乎等于AS125，
  upstream-only几乎等于cycle2，order到BA/action效应也基本不变。RL没有增加视频因果性，
  主要重定向共享policy输出basis。
- 下一单变量方法选择v6-fast macro400=`143`作强IL起点，冻结8个factor-output矩阵为
  policy tangent dictionary，只用RL学习上游/factor-input coefficients。显式多store、
  policy anchor和新loss暂缓；完整设计见
  `docs/action_forecast_writer_sft_anchored_tangent_basis_design.md`。

## 2026-08-05 Progress-Credit cycle2负裁决与参数位移重解释

- 同一formal root从cycle1 exact-resume到2，第二cycle得到49/96 train successes、16
  mixed、5 all-failure semantic、3 all-success与21 active-credit tasks。两epoch ratio/
  gradient finite、observer grad0、完整checkpoint/双ledger、0 watchdog/OOM；训练健康
  不能替代held rollout。
- cycle2 strict correct400=`102`、breadth4，逐task为
  `11/0/0/43/26/22/0/0`。相对cycle1=`104`严格paired gained/lost/retained/both-fail=
  `15/17/87/281`，`p=.8601`；Object-1净丢5、Object-3净增3，其余task净0，没有新task
  coverage。AS125/cycle1/cycle2 union/intersection=`128/79`、single envelope gap24，
  正式证明第二cycle继续换手而不累积；不得续4/8。
- cycle1→2的effective BA relative-L2/cosine/norm-ratio中位为
  `.01493/.999894/1.00214`，stable rank仍约`1.000016`。gained与lost state的变化幅度
  `.014725/.014724`、norm ratio`1.004994/1.004876`几乎相同；Object两task norm都增但
  成功一降一升。LoRA确实改变，却没有沿closed-loop正负结果分离，禁止再诉诸scale/rank。
- 同task更新仍主要是task-mean：cycle1→2的mean-energy fraction为`.9659--.9920`，
  Object-1/3 mean-update cosine`.97996`。这与任务内demo本应共享语义并不矛盾，因此
  不能单独写成“Writer忽略视频”；它与能力换手和outcome不对齐共同说明现有composer
  没把内容credit累积为稳定policy方向。
- raw梯度的factor/visual比约405，但Adam后cycle1→2每参数位移只约2倍：
  delta-L2/sqrtN为semantic`1.13e-5`、visual`6.29e-6`、procedure`9.33e-6`、compiler
  `1.13e-5`、factor-input/output`1.25e-5/1.24e-5`。factor-output相对L2`.00556`主要
  因zero-init后基准norm小；不能仅凭raw grad直接冻结decoder。下一步先做参数hybrid
  固定panel因果分解，再在basis freeze、policy-distance anchor与显式basis/coefficients
  之间选择一个新变量。

## 2026-08-05 Progress-Credit cycle1 strict correct400与LoRA裁决

- AS125 baseline与formal cycle1在同一400-row、无放回correct panel完成，correct=
  `97/104`，gained/lost/retained/both-fail=`22/15/82/281`；全部state、teacher video、
  env seed与共同policy-noise prefix配对。discordant exact two-sided `p=.3240`。
- 逐task（Long-1/2、Goal-3/6、Object-1/3、Spatial-1/3）从
  `10/0/0/43/24/19/1/0`变为`11/0/0/43/31/19/0/0`。breadth从5降4，净增7几乎全由
  Object-1贡献，Spatial-1丢失唯一成功；cycle1产生真实闭环影响，但未解决能力换手。
- 400对相同输入的effective BA relative-L2中位`.01677`、cosine`.999860`、norm ratio
  `.99965`。1,520个target谱样本的top-1 energy中位均`.999983`、stable rank均约
  `1.000017`；B-column cosine`.998846/.998840`。RL没有解除near-rank1结构，主要在
  既有coherent manifold上做小幅task-dependent调节。
- 该低秩结果不能单独否定方法：历史Target-Spectral已证明强制升rank会伤害性能。
  cycle1只有2次full24 update且held aggregate净升，19/24 train tasks有credit，因此
  最小下一证伪是同root exact-resume到cycle2并复评；若仍只有单task净增或breadth不恢复，
  则拒绝本续训轴并把失败接口归回condition-to-policy组合，而不是盲目续4/8。

## 2026-08-05 Progress-Credit formal ready竞态与根修

- 首次AS125-fresh formal0→1完整生成96 rollout和24 task credit，14 mixed、5
  all-success、5 all-failure与profile一致；但旧`FileStore` ready后rank0/1/2/5进入
  NCCL seq18，rank3/4仍停在seq17，600秒watchdog终止。0 update/metrics/checkpoint，
  因此没有科学性能结果，失败root禁止resume/评测。
- 早先profile通过不能证明旧barrier正确：它只表示Python已enqueue本地反向，并用有
  生命周期的临时store文件作一次性barrier；高度错峰下不能可靠证明所有rank CUDA工作
  真正结束。增大timeout只会延迟同一序列错误。
- 根修是每rank先CUDA synchronize，再按torchrun唯一session/cycle/epoch写原子marker，
  观察实际world-size的全部marker后才进入NCCL；marker在run内保留，新launch天然隔离。
  相同输出目录连续两个六卡新session探针都得到6/6 marker与sum21。该证据仍只是最小
  collective门，必须用原96-rollout/two-epoch规模和完整checkpoint/exact-resume裁决。

### 原失败规模正式重放

- clean/pushed`30977b5`从AS125 fresh在全新retry1 root完成cycle1：96 rollout、24,600
  actions、50 successes，14 mixed、5 all-success、5 all-failure。epoch0/1均先形成6/6
  CUDA-complete marker再进入NCCL，2次finite update、0 watchdog/OOM，wall`2125.726s`、
  peak reserved`19.455GB`。
- 两epoch ratio=`.99077--1.02504`/`.77339--1.09274`、grad=`.03635/.05018`、clip0；
  5/5 all-failure task有nonzero LoRA gradient，五下游block可达，observer grad0。
  checkpoint validator确认6 rank各16 rollout/4 progress-credit双ledger和96条全覆盖。
- 相对失败root只复现既有task28成功终止边界微扰：同一成功从76步变83步；其余95条
  rollout字节一致，24 credit文件完全一致。工程根修成立，但Writer性能尚未由closed-loop
  证明；先比较AS125 baseline/cycle1 strict correct400，不按train success续训。

## 2026-08-05 Task-Grounded Semantic Progress Writer profile

- clean`84d856c`从AS125 fresh完成一个full24 K4/Nmc4 two-epoch profile：50/96 successes、
  14 mixed、5 all-success、5 all-failure；5/5全失败task产生nonzero LoRA gradient，五个
  Writer下游block均可达，observer grad0。ratio范围`.99077--1.02504`与
  `.74545--1.09294`，0 clip/OOM/watchdog，peak reserved`19.455GB`。
- 相对只读诊断95/96 rollout完全一致；task28/cursor1同样成功但少一个replan chunk，
  总action少7。all-failure utility排序全部不变，最大/平均绝对差`.01622/.00318`。
  这是终止边界微扰，不构成重跑profile或改变credit的依据。
- profile只通过工程/机制门，不是性能结果。formal已按log-spaced剂量封存为fresh AS125、
  cycle`1/2/4/8`；先裁决cycle1 paired correct400再续。

## 2026-08-05 Task-Grounded Semantic Progress只读机制裁决

- clean`c483497`六卡只读诊断严格复现AS125的50/96 successes、14 mixed、5
  all-success与5 all-failure，96/96 rollout identity/outcome一致；0 optimizer、0 Writer
  backward、0 checkpoint。mixed success utility高于failure为`13/14`，同task pair
  AUC=`.8913`。
- task4/20/36/38/39的utility range=`.1228/.5712/.3338/.2554/.2371`；successful
  rollout上correct优于wrong/shuffled/reversed比例=`1/.88/1`，all-failure组对三反事实
  比例均为1；failure utility与pixel-change Spearman=`.5564`。预注册联合门全部通过。
- 该证据接受冻结AS125 observer作为all-failure相对credit，不接受Writer性能已改善。
  下一步只做一个AS125-fresh、不可续训的two-epoch工程profile；formal仍未授权。

## 2026-08-05 AS125与binary-only Flow-Credit负裁决

- 后续design已封存为
  `docs/action_forecast_writer_task_grounded_progress_credit_design.md`。最小新假设不是增加
  LoRA差异，而是冻结AS125 semantic encoder，把teacher首尾task-grounded内容变化作为
  方向，把rollout自身首尾变化作为位移，从而只给all-failure K4提供bounded相对信用。
  mixed binary与all-success行为保持不变；在只读binary agreement、failure dispersion、
  视频反事实和非pixel捷径门完成前不允许Writer更新。

- 同一fresh v6 AS root exact-resume100→125，累计60,000 queries、3,000 one-video
  conditions、125 finite full24 macros；本段wall`806.928s`，step125 checkpoint完整，
  0 OOM/clip与0 validation/test action reads。第一次resume漏传sealed`--num-workers 0`
  而在step101前fail-close，未污染训练状态；补齐CLI后原root完成。
- step125 K4为50/96 successes、19/24 coverage、14 mixed、5 all-success、5
  all-failure；suite success=`11/20/12/7`、coverage=`5/6/5/3`。相对step100严格配对
  gained/lost/retained/both-fail=`10/12/40/34`，task5/29新获coverage，全失败为
  task4/20/36/38/39。task36/38/39在五个K4点均0/4，不能再假设同一AS自然补齐门。
- 两轮profile credit ratio=`[.98710,1.01237]`/`[.76458,1.10147]`、grad=
  `.03615/.05310`、clip0，peak reserved=`40,338,718,720` bytes，完整cycle1 checkpoint
  与0 watchdog；工程健康不等于binary reward对全任务有credit，profile权重弃用。
- step100/125内部审计48/48 rows，wall`194.743s`、peak reserved`19,306,381,312`
  bytes，0 target-action/validation/test reads。norm中位`99.18→109.11`，stable rank
  `1.000176→1.000262`，但video energy`.1300%→.1154%`、demo1 BA
  `.0475→.0448`、fixed-action demo`.0101→.0086`均未增强；BA/action churn仍
  `.536/.138`。
- 全24任务success变化与video-energy变化Spearman=`-.521,p=.0090`，与BA churn
  `=.416,p=.0430`。新增coverage task5/29的video-energy中位`.1154%`，持续全失败组
  `.2101%`；后者demo1 BA差异`.0681`、churn`.5851`也更大。条件差异存在且更大，却
  不知道朝哪个policy方向有效，是比rank/scale/存储容量更早的失败接口。
- binary-only Task-Relative Flow-Credit正式负裁决，不启动formal RL、不续AS150。
  下一候选必须以teacher-video内容中的语义状态变化给failure trajectory排序，并保持
  binary success最高优先级；不得恢复v4 normalized-video-progress时钟或使用teacher
  action、privileged state、任务特化dense reward。

## 2026-08-04 v6 cold-start四点LoRA与video-to-action审计

- clean`2b775f0`完成step25/50/75/100×24 train tasks正式内部审计，96/96 rows、
  wall`291.333s`、peak reserved`19,308,478,464` bytes；固定demo0--4和按split结构预定
  8-task action面板，0 target-action与0 validation/test reads。root为
  `runs/outputs/pi05_as_writer_v6_relative_flow_coldstart_internal_audit_step025_100_r6_2b775f0_20260805`。
- norm中位随AS从`53.40→80.37→93.17→99.18`，但stable rank仅
  `1.000028→1.000055→1.000153→1.000176`，top singular share仍从`.999972`到
  `.999825`；q/v B-column cosine约`.999`。这是历史v6 macro50--600 rank collapse在
  当前cold-start的复核，不得忽略direct SFT stable rank1.505--1.517和Target-Spectral
  correct34而重新把强制高rank当解法。
- q能量中位从`84.86%`降到`80.97%`，v从`15.10%`升到`19.02%`，action target始终低于
  `.013%`；same-task五video centered/sample energy在`.0813%/.1309%/.1333%/.1300%`，
  step50后平台。demo1 BA差异中位`.0380/.0506/.0499/.0475`，fixed-action差异仅
  `.0081/.0071/.0094/.0101`。
- reversed/shuffled的BA差异到step100为`.1845/.1523`，fixed action为`.0498/.0474`，
  证明v6 Visual Transition→Procedure→LoRA→action顺序链工作；条件入口不是断路。
  但相邻checkpoint BA/action churn中位仍为step50`1.116/.187`、75`.758/.178`、
  100`.608/.142`，能力方向只是在缓慢稳定。
- step100 covered/all-failure tasks的norm中位=`99.13/102.37`、video energy=
  `.1290%/.1311%`、demo1 BA差异=`.0405/.0634`；三者与K4 success Spearman约
  `-.09/-.22/-.22`。失败tasks并不缺幅度或video sensitivity，问题是条件变化落入什么
  policy方向。当前最早有意义实验仍是闭环reward credit，但7个all-fail tasks使LOO
  advantage为零，所以先续同一AS到125获得reward support，不违规提前启动formal RL。

## 2026-08-04 Task-Relative Flow-Credit AS100：aggregate升、breadth回落

- 同一fresh v6 AS cold-start root从step75 exact-resume到100，累计48,000 logical
  queries、2,400 one-video conditions与100个finite full24 updates；最后segment
  wall=`805.085s`，step100 checkpoint完整，0 OOM/clip和0 validation/test action reads。
  24 tasks各2,000 queries、100 video visits并覆盖全部50条视频，A40适配没有缩减原
  logical训练量。
- step100 K4为52/96 successes、17/24 task coverage、11 mixed、6 all-success、7
  all-failure；suite success spatial/object/goal/libero10=`14/19/13/6`，coverage=
  `4/6/4/3`。全失败task为`4/5/20/29/36/38/39`。
- 相对step75的96对task/cursor、env seed、初态hash、policy seed、teacher demo与共同
  policy-noise prefix严格一致；gained/lost/retained/both-fail=`14/9/38/35`。task20
  失去coverage且没有新task获得coverage。success`47→52`与完整轨迹`25/38/47/52`
  不能掩盖coverage`12/14/18/17`，所以AS成熟化并非task breadth单调增长，task换手未解。
- credit两epochratio=`[.98452,1.00771]`/`[.88801,1.06045]`、positive clip均0、grad=
  `.02535/.02563`，max reserved=`45,183,139,840` bytes；完整cycle1 checkpoint、0
  watchdog/OOM/nonfinite。机制健康不等于应启动RL；coverage exit失败，profile更新弃用。
- owner要求把判断重新落到LoRA生成质量与模型内部条件传递。下一证据链固定比较同一AS
  四点真实BA rank/energy、跨video方向与fixed-action传递，并与direct SFT已知stable
  rank约1.52及v5.2较强视频特异性对照；不再用functional loss替代closed-loop结果。

## 2026-08-04 Task-Relative Flow-Credit AS75与breadth积累

- 同一fresh v6 AS cold-start root从step50 exact-resume到75，累计36,000 logical
  queries、1,800 one-video conditions和75个finite full24 updates；step75 checkpoint
  完整，source冻结，validation/test action reads均0，第三段wall=`805.356s`。
- 96条official-random-reset K4 rollout得到47 successes、18/24 task coverage、13 mixed、
  5 all-success、6 all-failure；suite success spatial/object/goal/libero10=`12/17/12/6`，
  coverage=`4/6/5/3`。相对step50严格配对gained/lost/retained/both-fail=
  `21/12/26/37`，success`38→47`、coverage`14→18`。
- 新获得coverage为task`9/16/19/25/37`，task4失去；剩余全失败为
  `4/5/29/36/38/39`。这比25→50有更强breadth净积累，尤其object达到6/6，但仍有12次
  paired能力丢失和task换手，所以继续预注册cold-start轴，不把aggregate上升写成漂移解决。
- 两epoch ratio/grad有限，positive clip仅`0/.000247`；mixed按rank为`3/1/4/1/3/1`，
  FileStore barrier再次在真实不均衡负载下完成两轮update和完整checkpoint，0 watchdog。
  峰值reserved=`40,340,815,872` bytes。
- 首次AS50→75 resume因选卡对应`4+2` NUMA rank拓扑，与root封存的`3+3`不同，被合同在
  训练前正确拦截，无metrics/checkpoint；保持原`3+3`拓扑后原命令完成。这是可信的
  fail-close，不是科学失败或需要放宽resume合同的bug。

## 2026-08-04 Task-Relative Flow-Credit AS50与collective根修

- 同一fresh v6 AS cold-start root从step25 exact-resume到50，累计24,000 logical
  queries、1,200 one-video conditions和50个finite full24 updates；step50 checkpoint
  完整，source仍冻结，validation/test action reads均0。
- step50的96条official-random-reset K4 rollout得到38 successes、14/24 task coverage、
  10 mixed、4 all-success、10 all-failure；suite success spatial/object/goal/libero10=
  `9/12/11/6`，coverage=`4/4/4/2`。相对step25的96个task/cursor，env seed、初态hash、
  policy seed、teacher demo和共同noise prefix全部一致；gained/lost/retained/both-fail=
  `19/6/19/52`，success`25→38`、coverage`12→14`。
- 净增长说明AS25→50仍在获得可闭环能力，不应因step25未过门而提前停止；但task5/16
  失去coverage、任务9/19/25/29/36--39持续全失败，能力轮换与long-horizon breadth仍是
  核心问题。因此继续按预注册cold-start轴到75，而不是用aggregate上升放宽门。
- 首次step50 credit run的96条rollout完整，但rank3恰为0 mixed tasks，较慢rank提前约
  10分钟进入NCCL gradient sum，最终触发480秒watchdog；没有optimizer update或
  checkpoint。这是outcome-skewed rank-local compute与collective入场时序不对称，不是
  reward科学负结果，也不是用timeout可修的transport问题。
- `e5bca71`在每epoch本地反向后加入独立FileStore all-rank-ready，再允许NCCL sum。
  原六卡/96-rollout/两epoch重放中，96/96 JSON与失败run字节级相同，38 successes不变；
  两epoch ratio/grad finite、完整cycle1 checkpoint、0 watchdog/traceback，峰值reserved
  `40,342,913,024` bytes。由此确认修复只约束process topology，不改变采样或credit。

## 2026-08-04 Policy-Target-Owned Factor正式负裁决

- clean`34be4a0`的fresh0→200完整执行200次full24 update、96,000 queries、4,800
  one-video conditions和8个checkpoints，wall`6678.957s`；0 clip/OOM、0
  validation/test action reads，峰值allocated/reserved`33.696/38.729GiB`。
- 严格配对correct400为`99/76/86/68`，breadth=`6/6/7/5`。逐task依次为
  `9/0/1/44/38/6/1/0`、`5/0/4/33/28/2/0/4`、
  `7/0/1/26/39/10/1/2`、`7/0/0/31/27/2/1/0`；Long-2四点全0。相邻
  gained/lost=`15/38,35/25,18/36`，四点union/intersection=`136/37`、envelope
  gap37。winner macro50=99低于Direction Store129与v6-fast143，故不续400。
- 76个完整独立heads按设计真正打破了跨层硬同向：macro50 q/v cross-layer BA cosine
  为`-.00011/-.00030`，而Direction Store是`.9319/.9666`。但correct LoRA norm均值
  只有`19.0257`，layer-energy CV=`1.9607`，q/v top-4能量占比`.7329/.8529`；SFT为
  `.464--.469/.544--.589`。新模型不是没有specialize，而是过度集中到少数晚层，且
  action heads能量仅`.0085%`。
- condition差异从Program到BA的保留明显高于Direction Store，却没有形成等比例action
  变化：same-task的Program/factor/BA/action relative-L2为
  `.90933/.05842/.09119/.03161`，旧模型为
  `.93377/.01935/.03242/.09114`。shuffled为`.86344/.11074/.16155/.09462`，
  reversed为`.96389/.17011/.24736/.12914`。A/E、Core mean、Core-only、Program-only
  与memory reversal仍全可达，故失败不是动态路径断路，而是写出的异质BA大多不在
  policy闭环有效方向上。
- rollout成功也不随video sensitivity增加：Goal-6/Object-1为`44/38`，内部fixed-query
  same/wrong变化很小；Object-3的condition/action变化最大却只有`6/50`。这说明模型
  学会了input-dependent写出，但“变化”与“competence”仍未绑定。
- factor承担单task梯度能量中位数`69.25%`，24-task median cosine`.0040`、负pair
  `.4457`，full24平均后只保留`.0484`能量，近`1/24`随机正交水平。CountSketch中
  task identity仅解释factor方向方差`.0168`（随机基线约`.0048`），相同task+demo
  隔50 macros重现的中位cosine仅`.0046`；控制task后的demo解释率不超过有限样本基线。
  这是per-condition内部credit缺少稳定task/video特征的直接证据，不是用functional
  loss替代rollout。
- 76个head `W_out`的macro50→100 cosine/relative-L2为`.7909/.855`，100→150为
  `.8911/.529`，150→200为`.9422/.364`；参数逐渐稳定时closed-loop仍继续跌落并换手。
  正式拒绝“policy-target参数共享是主要漂移根因”。下一根因是视频条件如何获得
  policy-aware、闭环有用、跨随机action query可累积的credit；不得继续加heads、层
  gate/scale、强制SFT profile或监督专用trick。本轮GPU、rollout和分析结束后暂停。

## 2026-08-04 direct SFT几何复核与Policy-Target-Owned Factor决策

- direct Source-SFT并不要求高effective rank。旧八rank correct122与corrected mixed-task
  correct109的step400 LoRA总norm为`34.4132/35.7362`，38-target mean stable rank为
  `1.5054/1.5169`，energy-weighted top-singular share为`.9229/.9056`。总能量中
  q占`.9390/.9249`，而q mean stable rank只有`1.1420/1.1571`；action heads虽rank较高，
  但只占约`.2%`能量。因此Writer的near-rank1只能作为诊断，不能作为漂移根因。
- SFT与Writer真正不同的是跨policy-target组织。两套SFT的q/v cross-layer effective-BA
  cosine约为0，layer-energy CV=`.705/.937`与`.707/1.052`，top-4层占q/v能量
  `46--59%`；Direction Store/SFB的跨层余弦`.93--.97`、CV`.03--.14`、top-4仅
  `23--27%`，即后者把各层写成几乎同向且均匀的模板。
- SFT层专门化不是随机初始化噪声。old/corrected两套独立recipe的q/v energy-profile
  Pearson为`.99306/.99040`，layer-norm Spearman均`.98349`；q top-4 layers完全一致，
  v重合3/4。对应target BA cosine均值`.84497/.85290`。action-trained闭环参考因此稳定
  表现为“target内低秩、target间异质”。
- 历史所有post-v6 canonical decoders都只有8个factor-family heads。Target-Bound让
  target先读Core/Program，SFB让target选择hidden basis，Direction Store按task拆完整
  heads，但每个head内部仍由18个q或v layers共享`W_in/W_out`。尤其zero-init final
  projection打开时，多layer functional gradients先在同一`W_out`聚合；这是此前从未
  单独解除的parameter-ownership边界。
- 新设计不强迫SFT几何。76个public A/B tensors各自拥有完整
  `1024→256→tensor_width`head；不同heads可自然学成同向，也可专门化。没有谱、正交、
  rank、层能量loss，没有SFT权重/方向初始化，也没有shared-carrier/innovation手工分解，
  因而不重演correct34的Target-Spectral或既往prior/innovation变体。
- 该ownership只依赖base-policy LoRA topology，task/video条件仍全部来自连续
  `Z=[Core,A,E,D]`；没有task-ID、LIBERO语义桶或static LoRA bank。AS functional
  gradient和未来rollout reward都可训练同一参数化。完整head而非只拆`W_out`用于首个
  根因实验，避免negative result仍被shared hidden transform混淆。
- canonical替换后的理论/实测目标参数为`47,857,920`：factor heads`40,517,632`，其余
  Writer`7,340,288`。旧Direction Router、八stores和额外frozen task-anchor forward
  已从active代码退役；Git/frozen config/artifacts保留历史。89项Writer tests及52项
  config/model/eval focused tests通过。首次六卡formal-seed fresh0→1证明B20/B2与显存
  健康，但最长仅82帧，不能作为longest105 profile或closed-loop证据；mode-specific
  profile seed现由runtime自动解析，避免再次污染formal config。
- clean`e03e61b`的真正seed172 longest105 profile随后通过：fresh0→1、exact-resume1→3
  共1,440 queries/72 conditions，峰值reserved`43.936GiB`且无clip/OOM；step2起五个
  主块全部finite/nonzero。该证据只封存mechanics、显存和gradient lifecycle，不对
  closed-loop性能作推断；formal必须从fresh identity开始。

## 2026-08-03 Semantic Direction Store正式负裁决

- clean `91feeef`的fresh0→200严格保持one-shot、logical B20/physical B2、full24 raw
  mean、96,000 queries与4,800 videos。correct400为`129/107/120/129`，breadth
  `7/7/7/5`；macro50和200同为129，但50覆盖7 tasks、200只覆盖5，故single winner
  选50。该结果较SFB macro50的69显著更高，证明稳定语义地址下的独立完整stores能
  加快早期能力获取，但没有超过v6-fast143或严格目标151。
- 相邻success churn为gained/lost `17/39`、`43/30`、`27/18`，四点union174、
  intersection65、single envelope gap45。能力仍集中于Goal-6和两个Object tasks，
  macro200又丢失Goal-3、Long-2、Spatial-1，parameter ownership没有产生稳定共同成熟。
- 固定route本身按设计工作：8 validation tasks的ordered top2数组均不同，但`1,5`与
  `5,1`是同一无序组合；同task五条件的store IDs/weights不变。共享0/1/2 stores的
  task-pair factor-gradient
  cosine均值`-.00043/.00664/.02249`，说明semantic storage确实提供了一定局部化，
  但同store内的条件梯度仍没有形成稳定共同方向。
- 内部纵向把失败定位到compiler：same-task-other使Program memory相对变化`.93377`，
  到factor/effective BA仅`.01935/.03242`；shuffled为`.81049/.04731/.07193`，
  reversed为`.93086/.09808/.15963`。remove-A、remove-E、Core-only、Program-only和
  Core-mean反事实都能改变BA/action，所以视频、Core与Program不是断路，而是其差异
  在写成public LoRA时被强烈压缩。
- nominal rank16并未形成16维功能写入：全部坐标active且能量近均匀，但correct-video
  effective BA的rank90/rank99均为1，stable rank=`1.000043`、entropy effective
  rank=`1.000371`、top singular energy=`.999957`、B-column cosine=`.999971`，无负
  component pairs。SFB macro200同样stable rank`1.000048`、top energy`.999952`；
  Direction Store改变了参数所有权，却延续同一近rank1公共B方向。
- Direction Store相对SFB确实把factor/BA的same/wrong/shuffled/reversed变化进一步
  压小，但fixed-query action变化反而更大；这说明不能仅用LoRA相对差幅度预测行为，
  也不改变核心谱证据：当前缺少的是多个可独立承担闭环功能的public A/B方向，而不是
  再加store、调K/center/top-k或放大scalar。
- 因此“shared factor parameter coexistence是主要漂移根因”只获得早期acquisition的
  部分支持，作为完整解释被正式拒绝。下一架构若获授权，应重构Program语义/时序状态
  如何组合成多维、非共线且仍coherent的A/B子空间；方法核心仍需适用于后续reward
  gradient，不能靠LIBERO桶、AS辅助loss或checkpoint融合修当前panel。
- 六卡内部分析还暴露两处独立于模型的拓扑硬编码：assignment默认4 ranks，seal固定
  接收4 payload/每rank2 tasks。`f82c7cd`和`a115b06`分别把任务ownership与结果
  Cartesian sealing绑定实际`world_size`；8项定向测试和clean六rank正式规模均通过。
  这是可变卡数的根修，不以减少GPU数绕开缺失rank，长期规则已写入`AGENTS.md`。

## 2026-08-03 Semantic Direction Store设计决策

- owner已解除阶段暂停，保持严格one-shot并取消Writer参数量软上限；新容量必须对应
  合理职责。owner同时明确functional loss很早已知不能预测rollout，后续应以模型
  内部表示、方向存储和组合证据解释task漂移，不能把held-loss错位本身冒充新根因。
- SFB不是没续第二小时：完整曲线为`69/91/118/127/117/81/126/120`。八点
  union=`193`、single best=`127`，macro200与350 gained/lost=`31/32`；结合factor
  share约`97%`、task-mean energy约`4.2%`和一阶moment轮换，最直接缺口是共享factor
  参数不能稳定共存不同task生成方向，而不是路由/视频主路径完全没工作。
- 新候选不用task ID、learned video gate或一task一专家。exact language额外经过无
  Meta-LoRA的frozen text-only forward形成checkpoint/video invariant semantic anchor；
  只用24 train languages做8-center spherical k-means，每task固定等权top2。
- 每个direction store拥有八个完整`1024→256→factor_width` heads及独立final
  `W_out`；同task的38 targets×16 ranks共享top2 stores，具体LoRA value仍全部来自
  `Z=[Core,A,E,D]`。预计参数`37,355,776`，新增约26.2M均属于独立方向存储。
- 首跑保持RAW full24/B20/fast-decay400，避免同时混入load-balance、expert重权、
  gradient projection或新loss。BCI只新增B2切片可重建的keyed independent
  Beta/Gaussian sampler，不复用已负裁决的Latin/antithetic VR estimator。
- owner进一步明确方法应从根因出发并尽量不特化于AS统一梯度下降。Direction Store的
  正式主张因此限定为objective-agnostic参数所有权与组合；train24均值、K=8、B20/B2
  只是当前domain/runtime配置。若失败不得继续调route小技巧，须回到条件表示、完整
  decoder参数化或credit assignment。
- canonical实现与train24-only center authority已完成：raw anchor先减train24公共
  均值再归一化，seed7 spherical k-means两轮收敛；primary/top2访问计数为
  `5/7/6/1/2/1/1/1`与`7/11/6/4/4/5/3/8`。61项focused CPU合同通过，实际参数
  `37,355,776`；尚无真实profile或行为结果。证伪重点是固定route下store内部方向是否
  稳定、success churn是否下降及single checkpoint能否严格超过150；若完整独立stores
  仍不超过v6-fast143，不通过增加stores、改K或gate修补。
- 首次clean六卡profile暴露的是process-group生命周期工程缺口，而不是Direction Store
  数值失败：NCCL在rank-local大模型CUDA构造前已启动，480秒后六rank均报告watchdog
  stuck且`only active collectives: 0`。因此正确边界是先用非NCCL ready rendezvous
  确认所有rank完成本地policy/Writer/optimizer构造，再统一创建NCCL；增加heartbeat
  timeout只能掩盖生命周期错误，不能作为canonical修复。
- lifecycle修复重放又把剩余失败精确下沉到真实`SeqNum=1` scalar all-reduce：这台
  BCI A40/NCCL2.28的direct P2P/CUMEM已在迁移验收中证实会hang，而显式SHM路径稳定。
  本次漏传`NCCL_P2P_DISABLE=1`后600秒超时；同六卡补回后sum21与BF16 finite两次
  collective在10.5秒通过。因此需要的是host-specific transport fail-fast，不是扩大
  collective timeout或改变Writer batch。
- 两层多卡修复后的首次step0失败定位为纯应用dispatch重复所有权：config owner已严格
  接受并逐字段验证Direction Store conditioning，但step owner又维护一份历史method
  字符串白名单并拒绝新method。该名单不控制任何算法分支，只是重复防御；正确修复是
  删除它，而不是再同步第六份字符串或增加fallback。
- 删除重复guard的clean pushed`1d0507e`最终越过完整真实vertical path：fresh0→1、
  exact-resume1→3、longest105和三份完整checkpoint均通过，六卡每macro仍严格是24 tasks、
  480 logical B20 queries和240个B2 physical forwards。峰值reserved约`43.893GiB`，证明
  37.36M Writer在46GB A40上无需改变科学batch或拆分模型即可训练。
- zero-output identity使macro1只有factor output梯度是预期生命周期；macro2起semantic
  frontend/Core/Program/compiler/factor五块全部finite/nonzero，macro3继续增长。该证据
  只证明固定route的完整stores可训练和梯度可达，不证明factor coexistence假设或
  closed-loop改善；后者只由fresh正式checkpoint的paired rollout裁决。

## 2026-08-03 BCI VR正式裁决与functional/closed-loop错位

- clean pushed`d9130c9`的有效VR root从fresh identity完成macro0→200：200个finite
  optimizer steps、96,000 logical queries、4,800 single-video conditions、wall
  `6619.670s`、0 clip、validation/test action reads=0。8个checkpoint的64/64 payload
  size/SHA通过，8个held functional panels各512 rows。错误teacher seed的旧10-macro
  root仍是aborted合同事件，不进入任何性能或机制数值。
- A40正式evaluator profile使用3 replicas/GPU与generation batch4；macro50/100/150/200
  各400 unique rows、42/42 shards、9/9 workers return0、每task teacher demos 0--49
  无放回各一次。四点间以及对应ordinary SFB panel的state/video/env/policy RNG均
  400/400严格配对。correct400曲线为`76/88/126/107`，breadth=`7/4/7/5`。
- macro50/100/150/200逐task（Long-1/2、Goal-3/6、Object-1/3、Spatial-1/3）为
  `3/1/0/37/29/4/1/1`、`4/0/0/37/25/22/0/0`、
  `4/2/1/41/42/34/0/2`、`8/0/0/39/33/24/0/3`。macro100的净增长主要来自
  Object-3而breadth掉到4；macro150到200又在Object-1/3分别丢9/10，说明aggregate
  变化仍由task换手驱动，而不是共同成熟。
- 相邻success-set gained/lost为`30/18`、`49/11`、`21/40`，Jaccard为
  `.5472/.5620/.5850`；50→200为`44/13`。四点union/intersection=`158/49`，
  single envelope gap=`32`。VR的四点union低于SFB同期169，但这是少发现能力与更低
  single score共同造成，不能写成已解决漂移。
- VR相对matched SFB同点delta为`+7/-3/+8/-20`，gained/lost依次
  `23/16`、`18/21`、`33/25`、`21/41`；前三点小幅来回，macro200明确更差。
  VR winner macro150相对source base为`83/5`，说明Writer仍提供真实新能力；相对
  v6-fast macro400为`27/44`，净少17。single winner126低于SFB127、v6-fast143和
  strict gate151，不构成新方法上限。
- 与ordinary SFB完全matched的前200步中，VR全段same-task successive all-block/
  factor CountSketch cosine只提高`.002634/.005104`，raw/factor mean-energy retention
  只提高`.001914/.001121`。分段all-block delta为
  `+.00721/-.01307/+.00663/+.00986`；51--100反向，151--200 raw retention还
  `-.000205`。因此早期三步正信号没有扩展成material、持续的梯度稳定化。
- VR held functional loss在50/100/150/200为
  `.130928/.133013/.130568/.129146`；macro200是四点最低，且优于SFB同点
  `.131776`，但closed-loop从126跌至107并比SFB少20。逐task绝对loss与成功的高相关
  主要反映固有任务难度；24个相邻task的loss改善与success变化Spearman仅`.263`。
  这是本轮最直接的surrogate/closed-loop错位，不支持继续靠同一functional MSE的
  Monte Carlo修补解决控制性能。
- 正式拒绝“可约flow time/noise方差是task漂移主要根因”。SFB路由与
  Core/Program→effective LoRA/action路径证据仍保留，但下一最早失效接口升级为
  functional action surrogate与source-policy closed-loop有效流形错位。恢复研究时应
  先整体设计新的training target与fresh证伪合同；不得继续增加basis/router、降低LR、
  续训VR到400、跑五臂或warm-start该checkpoint。owner要求本轮分析后先暂停。

## 2026-08-03 BCI 46GB逻辑B20适配

- 迁移资产、环境、source policy、tokenizer、LIBERO assets和历史formal roots均已在
  BCI项目树中核验；`/data1`个人quota约1TiB、当前个人占用约244GiB，足以容纳下一
  VR formal root，但每次launch仍须重查易变quota和预计峰值。
- 不能把A100 B20直接改成A40 B2：那会把VR estimator、action-query暴露量和训练
  objective同时改变。当前适配只把同一个逻辑B20的frozen-policy forward切成十个B2，
  并从同一keyed full-B20 Latin/antithetic draws切片；一套视频LoRA、task内mean、
  full24等权与一次optimizer update保持不变。
- 六卡把24 tasks机械分为4 tasks/rank。工程profile每macro仍是480 logical queries，
  物理forward增至240；三步约`33.97/31.69/31.24s`，峰值active allocated约34.97GB。
  resume后的reserved达47.11/47.70GB，说明allocator缓存余量很窄，但不是47.11GB
  同时活跃张量；冻结重放仍必须覆盖最长105-frame与exact-resume，不能只据一次
  no-OOM宣告稳定。
- 十个microbatch的LoRA叶梯度改为FP32累加后一次cast，避免BF16逐次求和把新的舍入
  噪声混入本来要验证的variance-reduction假设。该改变不修改随机样本、loss权重或
  optimizer语义。
- clean pushed`391f183`冻结重放确认上述适配：三步峰值active allocated约34.97GB，
  exact-resume后所有五主block可达，step3封存1440 queries/72 videos且没有
  validation/test action读取。dirty与clean step1 loss一致到记录精度；FP32累加只在
  后续非零深层梯度形成轻微预期数值差异。
- 第一次frozen resume在invocation前卡住15分钟；共享内存、P2P-disable和socket正常，
  同六卡object collectives最小探针以及完全相同resume命令随后均通过。证据不足以把
  原因归到代码、checkpoint或NCCL中的某一层，故不添加fallback或修改科学路径；只把
  它保留为一次性runtime风险并要求formal live监控。
- BCI首次formal暴露了一个独立于CUDA适配的科学合同错误：longest105 overlay的实际
  teacher seed仍是profile专用`172`，而sealed字段和ordinary SFB comparator均要求
  `20260722`。task IDs相同但teacher demos逐项不同，故此前按宏步对比的梯度稳定性
  不能称为matched estimator效应。该run在macro10、首checkpoint前停止；修复后loader
  对sealed seed一致性fail-close，后续只从全新identity比较。

## 2026-08-02 Post-seal条件分工假设

- 现有证据证明Writer能区分task/video并生成不同LoRA，但没有证明24项能力能在同一
  参数点稳定共存。full24 task gradients近正交、factor heads约占94%能量，说明问题
  更像共享写出器无法稳定容纳条件创新，而不是普遍负梯度冲突。
- 可泛化的“任务分工”不能使用被禁止的task ID或24个硬experts；语言/Core必须生成
  soft semantic address，让相关任务共享计算、不同任务形成不同activation/write
  paths。视频则提供A/E/D具体教学内容。
- Target-Bound满足这一假设的第一半：Core先决定38 target reads，并进入每个target的
  A/E/D读取地址；它仍共享temporal/reader/factor参数，所以只是可证伪候选，不是已解
  决方案。首小时必须同时看多task共同增长、success-set churn、分块task-gradient与
  video→BA/action传递；若factor端仍吞掉条件差异，下一整体重构应直接让semantic
  carrier决定factor计算路径，而不是增加局部gate或scale。
- longest105/profile只给出可训练性，不给出条件分工成功：macro3五个block梯度均非零，
  但factor仍占早期task-gradient energy约97%，符合zero-init后逐层解冻的早期生命周期。
  必须在成熟checkpoint再判断这种集中是否持续；不能用三步值提前判死，也不能把
  “所有模块可达”误写为task漂移已解决。

## 2026-08-02 A100清理与迁移证据分级

- `/data/ymdai`从约`430,784,090,112`降至EMBER提交前快照`229,312,688,128`
  allocated bytes，净释放`201,471,401,984` bytes。删除集中于可再生Writer/eval
  LoRA caches、profile/
  smoke/resume、退役SmolVLA outputs/numeric数据、rejected source EMA/optimizer
  state、superseded cache、generic base、已验证完毕的EMBER/MemLLM venv与package
  cache和owner明确不迁移的Codex archive；每批外部manifest均保留精确路径、bytes
  和理由。
- 原4.28GB`ember_assets`被初步误判为全退役后，原始contract tests暴露
  site-packages `hf-libero` assets symlink断裂。根因确认后只按
  `lerobot/libero-assets@0b3ea86...`恢复586文件/426.57MB，四个原始失败测试通过；
  其余约3.86GB旧cache/revision仍删除。该回归说明迁移必须单列simulation assets，
  不能把“旧SmolVLA资产”与当前LIBERO runtime资产合并分类。
- frozen source checkpoint的selected raw policy SHA仍为
  `60ea7ee898629321cf34522e5f0e45f4f1c2659c5f5dbc7b02ed9eb46a8cdf36`，formal
  inspector可加载；清理改变的是训练resume能力，不改变下游source inference身份。
- 没有进一步裁剪74.9GB formal checkpoints。现有核心未解问题正是task drift、能力
  轮换和架构×recipe混杂；非winner checkpoint提供不可再生的参数轨迹和内部几何，
  只留winner会破坏用户要求的根因分析。
- 406个complete evaluation roots在LoRA cache删除后合计约1.1GB，仍保留raw rows、
  results、queue、run contract和launcher completion；小容量换来严格paired复核能力。
- canonical feature cache v2约17.99GB虽可重算，但生成成本高且是当前32-task前端唯一
  cache，故归为SSH迁移而非垃圾。generic`pi05_base`已有精确HF revision/weight hash，
  且frozen source policy自包含，故归为BCI按需重下。
- MemLLM此前已完成系统清理；本轮不删除其19GB retained tree，因为模型revision未
  完全锁定，23个results roots又是压缩后的唯一正/负证据。仅venv列为重建。
- A100 Codex不迁移：archived sessions已删除，当前session/auth/cache不进入默认包。
  必要状态已写入Git handoff、formal artifacts和migration manifests。
- 历史docs中26个已不存在output路径均为按计划删除的profile/resume/cache或通配
  前缀，不是formal rows/checkpoints丢失；后续agent不得因此重跑工程smoke。

## 2026-08-02 CV-ADR GROUP4正式行为与阶段暂停根因

- clean frozen `51c0ba5`的normalized randomized-GROUP4自然完成1200 physical
  updates/200 task cycles、96,000 queries、4,800 one-video conditions，wall
  `4944.554s`；每25 cycle checkpoint、scheduler、cursor和信息墙合同完整，all
  finite、全程仅1次clip。与RAW逐`(cycle,task)`核验的4,800个task-visit/demo/frame
  三元组全部相同。
- cycle50/100/150/200 paired correct400为`82/77/73/110`，同topology RAW为
  `76/111/99/117`；GROUP4四点均值`85.5`低于RAW `100.75`，只在首点`+6`，其后
  `-34/-26/-7`。winner cycle200逐task为`10/0/0/41/38/15/2/4`，breadth6、top2
  占`71.82%`；没有达到150、v6-new 143或v5.2-old 132，不做五臂。
- GROUP4相邻success-set Jaccard仅`.445/.429/.500`，四点union150、intersection32、
  single-best envelope gap40；能力轮换未消失。它相对source base保留42/48、gain68、
  lose6，而RAW winner保留34/48、gain83、lose14：GROUP4更保守，却没有同时学到更多
  新能力，也不是“避免遗忘便解决漂移”。
- GROUP4 held functional loss在cycle50/100/150/200为
  `.13013/.13287/.13193/.13125`，末段train loss继续下降；behavior却
  `82→77→73→110`。effective BA norm均值为`69.65/63.56/78.56/72.04`，endpoint
  大于RAW `64.28`但闭环更差，排除简单loss或LoRA gain collapse解释。
- selected4负pair约`.442→.464`但global candidate-negative updates后段近零；factor
  梯度能量约`93.2--96.2%`。GROUP4参数段只有RAW约`.41--.67`长度，匹配段方向余弦
  compiler/Core/factor/Program均仅`.08--.41`，说明normalized六次重线性化仍进入
  不同盆地，却没有恢复稳定的有用conditional update。
- cycle200 exact50完成400 rows、五条件真实frame-order forward、0 rollout与信息墙
  读取0。A+D collective、remove-A、remove-D职责门从RAW `8/1/5 of 8`变成GROUP4
  `0/0/0`；effective BA mean relative L2
  `.06744→.01882/.02050→.00981/.05417→.01533`，fixed-action
  `.03613→.00483/.01091→.00264/.03356→.00446`。memory reversal BA
  `.00607→.00311`，shuffled/reversed BA`.04614/.02653→.02341/.01882`且action
  `.05484/.01539→.00686/.00517`，确认信号最迟在BA→action端再次被压弱。
- same-task BA centered variance/sample energy`.10494%→.09672%`；GROUP4 effective
  norm却`72.06>64.24`、stable rank都约`1.008`、component pair cosine
  `.650→.777`、B-column cosine`.968→.978`、ProgramRead/CoreRead RMS
  `1.021→1.168`。因此normalized GROUP4形成的是更大、更coherent但更static的写入，
  不是Target-Spectral式gain/rank collapse，也不是Program分支幅度不足。
- exact50原始analysis file SHA`f99d7cb1...86f6`；职责audit file/canonical
  `9725f010...b292`/`dc01dd97...5141`；RAW×GROUP4 compare file/canonical
  `a9f1e615...329f`/`2dc9ee29...5f4d`。
- correct-curve producer在JSON integer-key正规化前写入canonical claim
  `e5b00932...ba6`，reload canonical为`16afe12e...b5f`；职责audit以exact file SHA
  `54cd40e5...a985`和两个表示同时fail-closed验证，属于serialization provenance而非
  科学row变化。
- operator裁决因此已经否定“只把24-task更新切成六个normalized四task Adam便可同时
  修复absolute、漂移和视频写出”。该bundle不能拆成单一Adam、grouping或phase因果；
  但结合UCP同方向负结果，normalized GROUP4不再是下一架构默认。old recipe的强视频
  写出更可能依赖未归一六倍optimizer gain及其路径放大，而该成分同时带来task rotation；
  后续必须让架构在RAW full24下自身保持target-bound动态职责，而不能靠旧gain或固定scale。
- owner要求本阶段所有训练、correct400和GPU内部分析结束后暂停。Target-Bound
  Role-Preserving Program只在隔离分支完成CPU实现/结构验证，尚未profile、resume或
  训练；当前阶段不得启动下一GPU工作。

## 2026-08-02 CV-ADR GROUP4 B20/profile/resume seal

- 从clean detached `f6cf775`仅用GPU4--7完成18-update/3-cycle B20 profile；每cycle
  24 tasks各一次，共1,440 queries/72 one-video conditions。最长真实video为
  `task38/demo36=105` sampled frames，wall`73.196s`，峰值allocated/reserved为
  `76,945,014,784/77,370,228,736` bytes，全程finite、0 clip/OOM，信息墙读取0。
- factor从update1可达；identity-zero输出解除后，Semantic Frontend、Core、Program、
  compiler在updates2--18全部finite/nonzero。精确Writer参数`10,241,024`，frozen
  source policy trainable参数为0。
- canonical formal seed另做fresh0→1、exact-resume1→3、exact-resume3→7；step1与
  step3的全部payload SHA/size/mtime未被后续resume改写。7行metrics和3段invocation
  连续，首cycle 24 tasks各一次，scheduler只在update6推进，最终cursor为
  `next_step/data_step=7`、`cycle1/phase1`，validation/test action reads为0。
- profile/resume metrics SHA为`f8afb6ae...d90a`/`53cf0718...9de`；combined log SHA
  `2048cbc0...d4ff`。该vertical path只解除正式运行阻塞，不是科学performance证据；
  正式GROUP4必须从post-seal clean commit与fresh identity运行0→1200。

## 2026-08-02 CV-ADR macro200/400 matched梯度方差根因

- 固定24个train tasks、visit397--399、3 teacher videos、3个独立phase-stratified
  B20 query batches、3个paired flow draws，并在固定video/query上另做3 Gaussian×3
  Beta-time；每checkpoint共792个梯度、15,840个functional samples、72条teacher
  videos，零optimizer update/rollout，validation/test action reads为0。macro200/400
  原始analysis SHA分别为`1727f014...e7656f9`/`61a13978...db40520`，逐样本身份与
  CountSketch严格配对。
- `lora/action`的centered/sample从macro200的`.68581`升到macro400的`.81268`；
  video主效应只占`.001211/.001060`，24 tasks中没有一个以video或其interaction为
  主导，且18/24进一步下降。同期query为`.48588/.49530`、flow为
  `.21611/.21780`、query×flow为`.27171/.26715`。Writer各块结论相同：video
  中位只约`.10--.61%`，query/flow及其交互支配局部更新方向。
- macro200→400的`lora/action` matched单样本梯度余弦仍为`.84331`，exact norm
  ratio为`.92335`；但task-mean方向余弦仅`.61715`、task-mean energy ratio仅
  `.49189`，centered energy ratio为`1.02546`。Program更早发生职责旋转：其
  task-mean余弦仅`.43053`，而video effect余弦`.45218`且能量ratio`.56113`。
  因此末端持续追逐相似的高方差action surrogate，上游教学方向和有用video分量却
  在改变/缩小。
- 完全匹配的33个sample loss在24/24 tasks上都从macro200继续下降；逐task中位
  delta的中位为`-.004112`，与此同时correct400从117降到82。visit397--399对
  macro400是刚曝光的train条件、对macro200尚未曝光，所以下降不能冒充held
  generalization；它与独立held functional loss横盘合起来，证明模型继续拟合train
  surrogate/recency而闭环退化，不能再用更低action loss解释为能力成熟化。
- 根因边界因此调整为：teacher-video抽样不是late梯度旋转的主要随机源，但极小的
  video-conditioned gradient本身证明当前教学路径太弱；query差异包含真实跨episode
  状态/action覆盖，不能当成噪声删除；Gaussian/Beta flow估计方差是可减少但尚未
  证明能改善闭环的训练成分。下一步仍完成同topology GROUP4反事实，再联合重构
  target-bound causal Program、末端compiler职责和无偏低方差functional估计器，
  不做多视频/LoRA平均或固定scale/gate。
- paired analysis SHA为`ad7d6e06...44eb96a`，canonical payload为
  `d21c2cfc...d38b08`；全部向量比较受固定128维CountSketch估计边界约束，exact
  full-gradient仅保留norm而非vector。

## 2026-08-02 CV-ADR full400行为负裁决与训练动力学

- RAW从fresh identity到macro400自然完成：400 cycles、192,000 action queries、
  9,600 one-video conditions、16个every25 checkpoints，all finite、0 clip；末macro
  loss/gradient/LR为`.0979283/.0915952/1.00045e-5`，信息墙action reads为0。
- 候选step100--400的global raw-mean candidate-negative tasks均为0，然而task-pair
  negative fraction仍约`.36--.50`，full24 mean只保留约`4.1--5.6%`单task梯度
  energy。这降低“负冲突投影会解决漂移”的解释：pairwise负余弦并不等于global
  candidate伤害task。
- late factor head占task-gradient energy约`93.6--94.0%`；Program仅约`.15%`、Core
  `.75--.82%`、compiler约`1.45--1.50%`。这不是上游无梯度，但说明functional
  surrogate的局部优化几乎由最终factor owner支配。
- 同task连续one-video条件的task-mean/sample energy在第一/第二小时分别约
  `.92--1.96%`/`.26--.49%`，第二小时centered比例升到`99.51--99.74%`，相邻余弦
  仅`.0236--.0406`。这个量混合teacher video、B20 query和PI05 flow-noise，不能
  直接命名为video噪声；下一受控审计必须分离三者。
- 50-macro参数段虽随LR显著缩短，Core/Program/frontend后半相邻方向仍多次为负或
  近零；16个held functional loss仅在`.13055--.13399`内横盘。故fast decay减小
  位移幅度，却没有证明能力方向稳定；低functional loss也仍不能预测closed loop。
- 250/300/350/400 paired correct400为`77/69/80/82`；完整八点
  `76/111/99/117/77/69/80/82`的single winner保持macro200。200→250发生
  `16 gained/56 lost/Jaccard=.459`；400虽breadth7，仍只有4 tasks达到5 successes。
  第二小时是崩落与继续轮换，不是成熟化，因此不做五臂。
- macro200/250/400的effective BA norm均值为`64.28/69.93/69.29`，行为下降时
  LoRA幅度反而增加；这排除Target-Spectral式gain collapse，却支持functional更新
  离开source policy有效闭环流形。full400 dynamics SHA为`7289eef4...7a4f`；八点
  curve SHA为`fb75464f...ec90a`、canonical payload`bd4f43d4...ce909`。
- 仅凭跨macro梯度草图不能把99.5% centered命名为video噪声。当前matched诊断固定
  visit397--399和checkpoint以外全部条件，分别测video/query/flow及Gaussian/time；
  结果出来前不先验采用time strata、增大B或多视频平均。

## 2026-08-02 v5.2×v6架构—训练—视频因果联合审计

- 四个selected single-checkpoint五臂重新从正式400-row artifacts逐行核验：v5.2-old
  `132/138/74/82/83`、v5.2-task-complete `120/109/107/111/124`、v6-old
  `121/122/111/84/47`、v6-task-complete `143/135/125/128/129`。全部cell内
  state/env/policy seed schedule和video ordinal严格配对；提前终止导致保存的noise
  列表长度不同，但400/400公共前缀一致，不能误判为panel不配对。
- task-complete在两种架构上都压低顺序behavior margin：v5.2 correct-shuffled/
  reversed从`50/49`降到`9/-4`，v6从`37/74`降到`15/14`。但correct absolute的
  winner effect符号相反：v5.2为`-12`，v6为`+22`。在严格matched 150-video-visits
  截面上符号仍相反：v5.2 `132→51=-81`、v6 `95→111=+16`，描述性DiD为97；这
  证明强architecture×recipe-bundle交互，不识别某一个recipe成分。
- 16-row内部forward把共享变化定位到Procedure之后：shuffled/reversed的normalized
  centered Procedure在old/task-complete间基本保持同量级，v5.2为
  `1.234/1.556→1.241/1.541`，v6为`1.194/1.470→1.335/1.493`；但
  Procedure→effective-LoRA transfer降到old的v5.2 `.354/.262`、v6
  `.518/.416`，Procedure→fixed-action降到`.449/.338`与`.503/.354`。因此新recipe
  没有首先删除上游顺序表示，而是在compiler/LoRA/action共适应处系统压弱条件增益。
- old winner的Core、Procedure、slots和AdaLN gamma/beta绝对RMS普遍更大；例如
  v5.2 gamma `.842→.278`、slots `.164→.0431`，v6 gamma `.601→.266`、slots
  `.349→.0460`。这不是“放大就好”：old v6 absolute仅121且仍轮换；matched exposure
  中old每task-cycle执行6次Adam，两个50-visit段与task-complete的参数位移余弦只有
  `.086/.047`，Procedure为`.013/.028`，Adam一阶moment余弦在150 visits仅`.015`。
- 与source base 48个成功state严格配对后，四winner保留`39/30/43/39`，同时获得
  `93/90/78/104`个新成功。v6-task-complete的143主要来自更多新能力且未比v5.2-old
  多遗忘source；v6-old保留最多source却不是absolute best。故不能把差异简化为
  “旧recipe保能力、新recipe遗忘”或相反。
- 根因约束是联合的：full24/task-complete倾向较低的条件写入增益并没有消除task
  churn；old serial/rank-rotating恢复时序写出但以六倍Adam路径、强放大和旋转为代价。
  后续不得一棒子否定v7/v8/Loom等与fast task-complete混杂的思想；只保留其被内部
  反事实单独否定的模块。正式审计analysis SHA为`98371337...2efa`，canonical payload
  `6d9262f8...21cd`。
- matched 150-visits的source分解比winner比较更具解释力：v5.2 task-complete相对old
  同时少保留10个source success、少获得71个新success；v6 task-complete只少保留1个，
  却多获得17个新success，且当时8/8 tasks非零。因此v6 bundle与新recipe的相容性
  是“低动态增益下仍能取得较广语义能力”，不是单纯遗忘率差异；Visual Transition、
  mean-backed Core和256 factor width仍混杂，不能单独归因。
- selected v6 `121→143`的净+22又几乎全部来自Object task3的+24，Goal task6为-5、
  Spatial task1为-2；143最强的Object1/Goal6 correct为`46/36`，shuffled/reversed
  margin仅`1/1`和`-1/2`。这证明后期高aggregate仍是能力轮换和语义性帮助，不是
  breadth+causality共同改善。其Visual Transition与Procedure order response仍大，
  但Object1到BA/action只剩约`.12/.008`与`.13/.005`，最早断点仍在写出端。
- 全曲线state envelope也未显示某recipe普遍稳定：v5.2-old/new、v6-old/new的
  union-minus-single-best分别`93/42/37/77`，相邻mean Jaccard约
  `.453/.454/.527/.586`；网格和终点不同，只作描述性证据。score与source retention
  关系弱/不一致，而与非source新gain近单调，说明functional训练主要移动闭环阈值，
  并未保存固定的task能力集合。逐task/曲线审计SHA为
  `611c9330...c5a1`/`bf5a4609...1770`。

## 2026-08-02 CV-ADR一小时门、内部职责与architecture×recipe因果

- CV-ADR RAW macro50/100/150/200 paired correct400为`76/111/99/117`；winner
  macro200逐task为`23/0/0/34/25/33/1/1`，breadth6、四个tasks>=10、top2
  `57.26%`。相邻gained/lost为`53/18`、`27/39`、`44/26`，Jaccard
  `.450/.522/.511`，所以右端回升没有解决能力轮换。
- 与RNG-v2 UCP RAW做严格同recipe、同task/video/query曝光和同evaluation
  state/video/RNG panel比较，CV四点全部提高`+4/+24/+13/+28`。macro200的净增益
  分解为多保留9个source successes、再多获得19个非source successes；四候选
  envelope也从UCP149升到CV168。这证明mean-backed Core+contextual-value dual
  read作为整体在RAW下有真实架构价值，但不能把增益拆归某一个子模块。
- macro200 exact50中Core-only/Program-only距full effective BA为`.6059/.8119`，
  ProgramRead/CoreRead RMS比均值`1.021`，Effect-only距full BA/action
  `.06744/.03613`；CV修复了AP的Effect-only旁路且没有v10式单路放大。actual
  shuffled/reversed完整forward到BA为`.04614/.02653`，8/8 tasks均超过1%。
- 动态主路仍未过门：删除Action只在1/8 tasks达到5%-BA/2%-action门，删除D为
  5/8；只逆序已经contextualized的Program memory、固定Core/mask/position后BA仅
  `.00607`，0/8 tasks达到2%。这说明真实帧重排会改变上游P并传到BA，但下游reader
  对有序contextual states仍近似set reader，二者不能混写成“完全无顺序信号”。
- same-task 50-video effective-BA centered/sample energy均值仅`.10494%`，固定action
  中位仅`.00856%`；action均值被Object task3单一`.03678`异常值支配。LoRA norm
  `64.24`、stable rank`1.0072`、top energy`99.40%`、16坐标全活跃且负component
  pair为0，排除Target-Spectral式增益/协调坍缩，却不证明视频差异具有闭环作用。
- 同一UCP topology的source-preservation配对审计显示：RAW/GROUP4/SERIAL winner
  分别保留source `25/42/40 of 48`，得分`89/100/121`；SERIAL cycle150相对GROUP4
  cycle150同时多保留13、获得42个新successes，但cycle200回落107。高optimizer
  gain并非只破坏旧能力，normalized GROUP4也并非天然稳定；真正问题是条件路径
  co-adaptation、更新增益/时钟和closed-loop阈值共同造成的旋转。
- RAW每次full24更新只保留平均单task梯度能量约`4.5--5.5%`；同task连续one-video
  梯度约`98--99%`是centered noise，task-mean能量仅`.8--1.9%`。但mean candidate
  很少直接伤害task，所以CP负投影不成立。当前最可信联合解释是：架构必须提供
  稳定semantic carrier与可写出的causal innovation，训练operator又必须让二者
  在单视频高噪声下共同适配，而不能靠RAW稀释动态或SERIAL六倍增益持续旋转。
- 这些结果授权RAW exact-resume第二小时，因为117与UCP/SERIAL低锚同档、右端best、
  多task贡献且架构干预在四点一致为正；它不授权五臂，也不取消同topology GROUP4
  控制。只有macro250--400闭环曲线才能判断成熟度还是继续轮换。

## 2026-08-02 CV-ADR mechanics seal

- teacher-seed172 B20三macro在四rank上真实覆盖最长105-frame视频；稳态约
  `18.74--18.78s/macro`、约`25.6 queries/s`，峰值reserved `83.52GB` decimal，
  没有真实OOM或降到B16的依据。
- functional identity使macro0只有factor head直接有梯度；完成第一次B更新后，
  macro1/2的semantic frontend、Core、Program、compiler、factor全部finite非零。
  这是zero-B identity的预期链式可达，不是上游永久死亡。
- formal seed exact-resume保持step1全部7文件SHA/size/mtime不变，task/video/query、
  scheduler与cursor连续；因此正式run可从fresh identity开始，并在完整macro边界恢复。
- 三个早期macro的task-gradient negative-pair约`.42/.31/.35`，而candidate-negative
  tasks为`4/1/3`。它再次说明负pair值得记录但不能直接等同task漂移根因；闭环判据仍需
  macro50--200 correct400与逐task迁移。

## 2026-08-02 UCP RAW×GROUP4最终裁决与CV-ADR边界

- RNG-v2 normalized GROUP4完成1200 updates/200 cycles，correct400
  `77/76/66/100`；RAW为`72/87/86/89`。GROUP4 endpoint增11但四点均值降3.75，
  winner top2集中到74%，四点累计2 tasks上升、5 tasks下降。它改善部分success-set
  重叠，却没有达到absolute/breadth行为门，故不迁移为CV默认。
- GROUP4累计16次clip/1200（1.33%，最小`.653`），与frame cost几乎无关。
  full24和selected4的negative pair都约42--43%，而mean candidate实际负task很少；
  结果不支持“full24负冲突是漂移根因”，也不能把GROUP4拆成单一Adam或group因果量。
- paired exact50中，RAW→GROUP4的`x_only` BA变化为`.058999→.013291`，action
  `.017173→.005888`；fixed-X shuffled/reversed BA为
  `.028069/.025026→.009288/.009092`，8/8 tasks均压弱。GROUP4 reader把mass从
  X/D/A `.434/.522/.044`改成`.560/.405/.035`，明确转向absolute X。
- GROUP4 effective norm比RAW更大（`63.70` vs `59.42`）、stable rank更接近1
  （`1.0021` vs `1.0066`），故视频写出减弱不是norm/rank collapse。GROUP4 fixed
  action的高均值来自一个closed-loop 0-success Spatial task；median比RAW更小，
  证明“大action变化”可完全off-manifold。
- exact50同一8×50 panel、五个真实frame-order forward、零rollout、信息墙全通过。
  operator/exact analysis SHA为`97c70dd...a6e0`/`7201364a...11fd`；RAW/GROUP4
  原始analysis SHA为`9704b9fd...4067`/`57760475...c52`。
- 架构×recipe根因因此更精确：normalized group updates不会自动解决漂移，还会把
  UCP弱动态路径进一步压向X；旧未归一SERIAL能放大动态但也放大off-manifold方向。
  CV-ADR必须让contextual Program直接成为value，使有用视频动态不依赖六倍optimizer
  gain。若CV RAW失败，仍需同topology GROUP4复核；若BA/action传递成立而闭环弱，
  下一根因转向functional surrogate与source-policy有效流形。

## 2026-08-02 UCP RAW训练噪声实现敏感性

- RNG-v1与RNG-v2 RAW保持Writer topology、optimizer、LR、rank/task顺序、全部
  `4,800`条task/video assignments和`96,000` queries一致；唯一functional变化是
  CPU Beta flow timestep从ambient rank/order stream加入task/query keyed
  fork-seed-restore。四点correct400却从`89/71/82/117`变为`72/87/86/89`，差值
  `-17/+16/+4/-28`，不是统一平移。
- 对全部`4,800`个matched task visits，四块32维CountSketch梯度余弦中位数仅
  `.163--.193`，负余弦比例`.224--.324`；前3--5 updates仍多为`.92--.998`，到
  update10已经明显分叉。每task loss绝对差中位`.01423`。因此一次合法的flow-time
  noise identity变化会快速导向不同optimizer basin，当前训练动力学本身高度
  noise-sensitive。
- 该对照只有一个seed7 realization，不能证明ambient CPU RNG“更好”，也不能把
  v1的高endpoint当作可复现收益；v1缺少跨rank/order identity，仍不能用于
  RAW×GROUP4 operator归因。正式analysis/summary SHA分别为
  `ff6acdf8...b82`/`34988d5f...d80f`，canonical payload SHA为
  `5dba4218...fc4`。

## 2026-08-02 RNG-v2 RAW闭环负结果与训练机制

- RAW macro50/100/150/200 correct400=`72/87/86/89`；winner仅89，breadth6但只有
  4 tasks>=5，top2占60.67%。四点union149相对single-best gap60，相邻Jaccard
  `.359/.504/.549`，task漂移远大于3点aggregate表象。
- train trailing25 loss`.11695→.10027`，held loss约`.130--.1315`，effective BA
  mean norm`45.89→59.36`。因此增加曝光、继续压低functional loss或全局放大LoRA
  都没有证据；功能surrogate与closed-loop有效流形错位仍是主要候选。
- checkpoint处full24 task-gradient energy retention为`6.62/5.23/4.67/4.77%`，
  pairwise negative约`38.0/45.3/48.2/49.3%`，但raw mean candidate-negative tasks
  仅`2/0/0/0`。负pair很多并不表示mean方向即时伤害多数task；主要现象是近正交
  conditional innovation被24-task mean稀释，所以CP式负投影不具备根因依据。
- Writer相邻50-cycle位移方向cosine仅`.232/.122`，Adam一阶moment相邻checkpoint
  cosine为`.081/.027/.059`，与success churn共同证明轨迹持续旋转。正式analysis文件
  SHA256为`0f8545b115c91bde06e20a142079c7dd00c0628bb40845090176f2e0ecab3462`。
- GROUP4仍是必要对照：若四组serial relinearization/Adam adaptivity也不能共同提高
  endpoint/AUC/breadth与动态传递，就应降低“full24一次平均”解释并把重点转回
  UCP职责和surrogate manifold；若改善只随phase/cost，则优先解释为long-first
  optimizer curriculum，不能冒充通用operator收益。

## 2026-08-01 task/query RNG-v2真实重封存

- `dae13bf`把CPU default和指定CUDA generator共同fork/seed/restore，scheme、
  cycle-normalized config、RAW/GROUP4 checkpoint family与shared/trainer/rank state
  schemas均fresh-incompatible升v2；完整CPU回归`241 passed`。
- RAW B20 fresh0→1→exact-resume1→3完成三次full24宏步，共1,440 queries/72
  one-video conditions；GROUP4 fresh0→1→3→7完成首个24-task cycle并跨越边界，
  scheduler只在phase5后推进。两者all finite、主要模块梯度可达、信息墙读取0；
  resume没有改写step1或GROUP4 step3 payload。
- 跨rank/phase操纵把tasks12/14/34/37从GROUP4 ranks `2/3/0/1`换到RAW ranks
  `1/0/3/0`。同demo/frame/action row/query seed下，两臂loss逐位相等为
  `.0845656544/.1445673406/.1301287264/.1523376107`，raw task-gradient norm也
  逐位相等；CountSketch最大绝对差`5.82e-11`。因此v2确实实现跨rank/order的
  task/query随机身份，而非只在相同ambient stream复现。
- RAW run-contract/metrics SHA为`31f2edea...c038`/`a6c41cd2...7cdc`；GROUP4为
  `8b691d48...cee4`/`a8c55ee1...94cd`。旧105-frame profile继续只承担B20容量证据；
  v2真实reseal最大82 frames。两份formal config现可从fresh identity启动，v1
  checkpoint仍严格拒绝。

## 2026-08-01 task/query RNG-v1根因与operator cell失效

- `scoped_policy_randomness`的CUDA分支只seed指定CUDA generator；安装版LeRobot
  PI05的Gaussian flow noise在CUDA采样，但Beta flow timestep经CPU
  `torch.distributions.Beta.sample`生成后才搬到device。所谓v1 stateless合同因此
  实际是`query-keyed CUDA noise + rank/order-keyed ambient CPU time`。
- step0 public LoRA严格identity，Writer topology/operator不可能影响source-policy
  functional loss。离线重建确认RAW/GROUP4共同tasks 12/14/34/37的action row IDs、
  demo/frame IDs、teacher video长度和derived query seed逐项相同；loss仍从
  `.152825/.126055/.099258/.133874`变为
  `.105294/.125041/.114391/.178842`。四task恰好换rank，构成直接因果诊断。
- GROUP4 formal已正常停止于307 updates/51 complete cycles；root、metrics、
  step150/300 checkpoints保留，但没有run summary，且禁止resume/eval。该run不能
  识别update operator，继续消耗到1200只会扩大无效证据。
- RAW autoscaled与true-fast保持相同rank/task/microtask顺序，cycle0 rows/loss/sketch
  完全相同；故scheduler差仍是同一ambient-time stream下的matched comparison。
  但两条absolute只属于RNG-v1实现bundle，不能声称真正query-keyed time，也不能
  与换rank/phase的GROUP4比较。旧ambient recipes不整体失效；被推翻的是新v1合同。
- 最窄职责完整修复是同一fork scope内同时seed/restore CPU default与指定CUDA
  generator，不改数据、信息墙、模型或loss。randomness scheme、cycle-normalized
  config schema、RAW/GROUP4 checkpoint family和shared/trainer/rank schema全部升v2，
  旧checkpoint必须fail-closed。formal在真实manipulation/resume重封存前保持blocked。

## 2026-08-01 UCP true-fast400 scheduler裁决与normalized-group4启动（RNG-v1快照）

- clean frozen `cfc2ad1`的task/query-keyed UCP raw从fresh identity自然完成前
  200/400 logical cycles：96,000 queries、4,800 one-video conditions、wall
  `3884.255s`，200行全部finite，validation/test action和test video读取0。
  runtime保持真实`warmup17 + decay400`，候选50/100/150/200的LR精确为
  `.000289394/.000258333/.000211540/.000156139`。
- 四个严格paired correct400为`89/71/82/117`；single winner macro200逐task为
  Long `9/3`、Goal `1/39`、Object `34/29`、Spatial `0/2`。breadth nonzero为7，
  但只有4 tasks达到至少5次成功，top2为`73/117=62.39%`。150→200虽共同得到
  44、只丢9且Jaccard `.5794`，absolute仍只回到旧UCP raw的117，不达到125强
  五臂门；不做same/wrong/shuffled/reversed。
- 与输入完全一致的autoscaled-decay200消融做同一步严格配对后，scheduler effect
  为`+8/-1/-25/+39`，不是统一性能提升。autoscaled macro150的107与true-fast
  macro200的117最接近，但成功集合仍为81 both、26 autoscaled-only、36 true-only，
  Jaccard仅`.5664`；更慢日程主要把能力峰推迟并重新分配task，而没有提高UCP
  single-checkpoint ceiling或解决漂移。
- 两条run的前200 cycle逐步task/video/query identity和8个checkpoint累计identity
  完全相同；四个evaluation panel的state、video ordinal、env seed和policy RNG
  prefix也逐row相同。cross-run evaluator static contract SHA为
  `6e0b8b2d...be387`。原candidate的intra-run evaluator hash保留训练run/config
  provenance，本来就应不同；scheduler analyzer先fail-close后改为只剔除明确的
  treatment provenance，没有放宽实际row pairing。
- true-fast不是简单把同一方向走远。autoscaled→true的Writer delta norm从step25
  `1.571`增到step200 `4.370`，Adam一阶moment cosine从`.597`降到`.354`；同输入
  25-step更新方向cosine从`.758`降到`.249`。true-fast最后四段参数位移仍为
  `1.576/1.380/1.266/1.136`，相邻段cosine降到`.121/.077/.031`，即持续沿强烈
  旋转路径移动；autoscaled末段则已缩到`.136`。
- 更慢LR确实保留了部分条件梯度稳定性：最后50 cycle同task相邻visit的四块
  CountSketch cosine由autoscaled `.009-.017`提高到true-fast `.083-.114`；但
  full24 mean的平均task-gradient energy retention仍仅`4.72%`，接近正交24方向的
  `1/24=4.17%`。pairwise negative很多，而raw candidate对task为负的均值仅`.02`，
  所以主要问题仍是共同方向很小、conditional innovation近正交，不是CP式均值
  直接伤害多数task。held loss在两条轨迹几乎相同且不追踪correct。
- true-fast candidate analysis为
  `/data/ymdai/outputs/ember/pi05_as_writer_ucp_taskquery_rawfull24_truefast400_candidate_curve_seed7_cfc2ad1_20260801/analysis.json`
  （SHA `7b7d9822780a741b43b8567609dd89bd31489fd2e2c837c4d27baae50e885dd3`）；
  paired scheduler interaction为
  `/data/ymdai/outputs/ember/pi05_as_writer_ucp_taskquery_runtime200_vs_truefast400_scheduler_interaction_seed7_cfc2ad1_20260801/analysis.json`
  （SHA `81eca3ccb88a7c77f3af6f7a12a2d141e9e1580e30ae06c040a7644eaa6bab7e`）。
- 预注册下一格cycle-normalized randomized-group4现已从同一clean frozen
  `cfc2ad1` fresh启动。首个六phase cycle恰好覆盖24 tasks、24 videos和480 queries；
  按task排序后的teacher demo与sampled-frame count和raw cycle0逐项完全相同。
  scheduler只在phase5后推进，每phase LR为logical LR/6，step2起frontend、Program、
  compiler、factor均finite可达；0 clip/OOM，峰值reserved约`83.63GB`。正式结果
  完成前不据train loss预判operator优劣。

## 2026-08-01 UCP formal scheduler自动缩放合同偏差

- `e1299db`已修正formal total并push，但第一次true-fast400 launch在output root
  创建前由runtime拒绝：raw/group4 overlay的`stage_stop_steps`仍残留旧的一小时
  `[200]/[1200]`，而parser正确要求列表以formal total `400/2400`结束。所有ranks在
  video/query读取前退出，没有checkpoint或科学数据；这不是训练不稳定。
- 最窄修复为`[200,400]/[1200,2400]`，并把stage stops非空、严格排序、包含selected
  stop、最终等于total、全部cycle整除加入loader fail-close。新增测试直接走formal
  `resolve_runtime`而不只加载config，定向`25 passed`；新正式run必须换clean commit、
  frozen worktree、root和log。
- task-query raw control从clean frozen `1a09e71`按sealed overlay的
  `total_steps=200, stop=200, decay_steps=400`启动。逐步LR审计发现macro150
  实际为`5.4093e-5`；真正400-step cosine在同点应约`2.1049e-4`。LeRobot
  `CosineDecayWithWarmupSchedulerConfig.build`会在训练总逻辑步小于decay时自动
  把warmup/decay从`17/400`压成`8/200`。因此该root只能称
  `configured-decay400/autoscaled-decay200`，不得作为fast400证据。
- 根因是新control overlays把一小时stop误写成scheduler的formal total：raw
  `200`、group4 `1200 updates=200 cycles`；基础UCP和CV-ADR合同原本正确地使用
  `total=400/2400, stop=200/1200`。既有测试又固定用400逻辑步构造scheduler，
  没走正式config total路径，因而漏检。
- 当前raw不被中止；它自然完成后保留为task/query/video/noise一致的autoscaled-decay200
  scheduler ablation，并正式评测50/100/150/200。错误group4不会启动。最窄修复
  把两份formal total恢复为400/2400，并在config loader新增
  `logical_total >= decay_steps` fail-close；定向24项回归通过。随后从fresh
  identity重跑真正fast400 raw/group4配对，不能把本次发现调试成偏好结果。
- CV-ADR canonical实现已在隔离branch提交为`b2bc70c`：同一contextual Program
  同时作K/V、mean-backed Core与target/rank dual read，精确参数`10,241,024`；
  focused `159 passed`且architecture guard无hard violation。真实B20/profile/
  resume仍pending，且在UCP正确受控格完成前不推送或正式训练。
- autoscaled-decay200 raw现已自然完成200 cycles：96,000 queries、4,800个
  one-video conditions、wall `3892.039s`，200行全部finite，信息墙读取0；唯一clip
  在macro5，后续没有连续非有限或OOM。macro50/100/150/200的train loss为
  `.11826/.11069/.10848/.10726`，held loss为
  `.13509/.13118/.13078/.13154`，后两者仍不能代替closed-loop选择。
- LR缩短主要截断位移，并未使task方向稳定。当前Writer参数段长度
  `50→100/100→150/150→200 = 2.348/1.084/.349`；历史fast400 UCP为
  `2.911/2.252/1.734`，但相邻段方向cosine两者都只有约`.13-.24`。raw mean保留的
  平均task-gradient energy从macro50的`9.05%`降至macro200的`4.22%`，已经贴近
  24个等norm正交方向的`1/24=4.17%`；pairwise negative fraction升至`51.81%`，
  但整体candidate direction对24 tasks仍`0`个负点，所以这不是CP式“均值伤害多数
  task”的证据，而是共同方向极小、task innovation近正交的证据。
- 同一task相邻one-video/query visit的32维梯度CountSketch cosine也从前50步四个
  block均值`.218-.248`降到后50步`.0094-.0173`，晚期几乎由条件噪声主导。旧
  fast400的晚期范围`.073-.127`，但同时混杂ambient policy RNG；只有纠偏后的
  task/query-keyed true-fast400 raw完成后，才能把这部分差异严格归给scheduler。
- 四个正式paired correct400为`81/72/107/78`，observed-best macro150；四个panel
  都是400 rows、36/36 long-first shards、0 failure并严格paired。macro150逐task为
  Long `16/1`、Goal `1/28`、Object `27/32`、Spatial `1/1`，breadth8但只有四个task
  达到至少5 successes，top2为`60/107=56.1%`。它低于旧UCP raw117、SERIAL121和
  v5.2/v6强点，不续训、不做五臂。
- 50→100、100→150、150→200分别gained/lost `28/37`、`54/19`、`14/43`，成功集合
  Jaccard `.4037/.4206/.5289`。effective BA mean norm单调到macro150后仅
  `45.34/50.00/52.94/51.92`，但correct剧烈轮换；macro150→200的参数位移只有
  `.349`且第一moment cosine仍近零量级，微小更新足以跨closed-loop阈值，不支持
  “只看LoRA norm”或“更快衰减即可稳定能力”。
- 正式candidate analysis为
  `/data/ymdai/outputs/ember/pi05_as_writer_ucp_taskquery_rawfull24_configdecay400_runtime200_candidate_curve_seed7_1a09e71_20260801/analysis.json`，
  SHA256 `bfd580d46305d87fdfbdd1f593eecaf49b599fa328584f63dc8d3027dcf30993`。
  analyzer仅兼容已sealed的`selected_task_count=24`与overlay解析，仍完整复用旧UCP的
  panel/cache/shard/checkpoint/Gram/几何校验；不能把本cell与旧ambient-RNG fast400
  直接冒充纯scheduler因果比较。

## 2026-08-01 UCP exact-resume seal与canonical控制恢复

- group4 formal-seed root完成fresh0→1→resume1→3→resume3→7；step1与step3
  checkpoint payload在后续resume后size/mtime/SHA均未变化，cycle0六phase覆盖
  24 tasks恰好一次，scheduler只在step6推进，Adam state逻辑步为1/3/6/7。
  raw formal-seed root完成fresh0→1→resume1→3，三次完整full24更新覆盖72个
  one-video conditions和1,440 queries。两臂cycle0的24个teacher-video assignments
  逐项一致，validation/test action reads均为0。
- group4 run-contract/metrics/summary SHA为`7456739f...795e`、
  `1d2f64c...c08e`、`fbd335d1...a67d`；raw对应为`5b3d4dfc...312d`、
  `f3b8191e...3536`、`eac4eb17...bd34`。两份config的`formal_run.status`均为
  `sealed`；group轴1200 updates=200 cycles，raw轴200 macros，候选分别是
  300/600/900/1200和50/100/150/200。
- group4只消除了固定cost-phase curriculum并按六步复合LR/beta/weight decay、
  cycle-boundary scheduler；它仍有六次参数重线性化、clip/Adam、phase order和
  selected4噪声。因此该格识别的是完整update-operator bundle，不把结果伪装成
  单独Adam、batch size或task平均的因果效应。
- `85a82cb`把封存`b52cb54`的UCP运行面逐blob恢复为唯一canonical，并删除AP、
  endpoint及其仅实现测试；正式负证据仍由Git/artifact保留。聚焦回归`107 passed`，
  compileall、四份JSON和diff check通过。architecture guard相对AP报告UCP既有的
  大函数/测试越界，但active source净减1,061行；正式控制必须保持与已profile/
  resume的UCP位级源码一致，故本轮记录为cohesive scientific-control例外，不在
  launch前重构。CV-ADR必须原位替换同一owner，不能复制runner或保留兼容分支。

## 2026-08-01 Endpoint10负裁决与cycle-normalized受控格

- endpoint10在任何结果生成前封存的18-candidate primary关联门已经原样执行。
  clean `0f92e35` formal root产生9,216 rows，wall `1041.474s`；environment未构造、
  parameter gradient未计算、validation/test action读取0。run contract/rows/summary/
  association SHA为`edb7d3c...583b`、`7087999d...bd0`、`a4a489a3...c2ba`、
  `d54435fe...f707`。
- `quality=-rollout10_executed5_valid_normalized_mse`对correct400的全局Spearman仅
  `.258398`；固定100,000次candidate-label permutation双侧`p=.298447`，所以
  global gate和all gate失败。family-demeaned pooled Pearson/Spearman虽为
  `.40360/.41090`，UCP/v5.2-new/v6-fast family Spearman为`1.0/.4/.45238`，
  两个recipe direction与逐task中位`.16444`、6/8非负也通过，但预注册要求四门
  全过，不能事后用局部门救回。
- 跨架构错位不是小噪声：v6-fast macro200 closed-loop correct133却得到18点中最差
  endpoint quality `-.128863`；v5.2-new macro200 correct91反而得到最好quality
  `-.120544`。v5.2-old correct132的quality `-.120991`也仅略差于v5.2-new macro200。
  ten-step teacher-action误差能追踪部分family内局部拟合，却不能识别共享source
  policy的closed-loop有效流形；它永久只作负诊断，不能选checkpoint、改loss或训练。
- exact UCP受控cell把fresh raw-full24与cycle-normalized randomized-group4绑定到
  同一task/video/query exposure和task/query-keyed stateless policy noise/time。
  group4用六个随机Latin phases各取4 tasks覆盖24 tasks；LR除6，Adam betas取
  `(0.9^(1/6),0.95^(1/6))`，weight decay解六步乘积，scheduler只在cycle末推进。
  这仍是完整update-operator bundle，不冒充单一Adam因素。
- group4 longseed172真实profile完成18 updates/3 cycles：每cycle 24 tasks恰好一次、
  每rank每cycle6 tasks，最大真实video 105 sampled frames，1,440 queries/72 one-video
  conditions，所有数值finite；step0 identity只有factor梯度，step2起frontend、
  Program、compiler、factor全部非零可达。峰值allocated/reserved为
  `76,971,835,904/83,647,004,672` bytes，B20成立但显存余量很窄。
- formal-seed exact-resume必须同时跨midcycle和cycle boundary验证optimizer bias、
  LR/betas/decay、scheduler logical cursor、task/query/video/RNG cursor与旧payload不被
  改写；通过前不启动两臂科学训练。

## 2026-08-01 AP-ADR门失败与key-only/raw-value根因

- AP-ADR fresh macro50/100/150/200 paired correct400为`91/81/94/91`，breadth
  `6/6/5/7`；winner macro150逐task`[18,1,0,37,29,9,0,0]`，top2占
  `66/94=70.2%`。相邻checkpoint gained/lost为`33/43`、`36/23`、`25/28`，
  correct没有右端上升且task能力持续轮换；不resume、不做五臂。held functional
  loss`.13275/.13579/.13096/.13204`和BA norm`55.81/68.85/70.51/65.20`均不追踪
  closed-loop，进一步反驳用低action loss或单一LoRA幅度选点。
- 首轮内部分析暴露约5% `Q_text`重放漂移。最小复现证明Writer重复编码本身bitwise
  稳定，而调用PI05 `sample_actions`即使使用identity LoRA也会永久把language/expert
  `_attn_implementation`从`sdpa`改成`eager`；没有named buffer变化。AS训练和正式
  evaluator分别不调用sampler、或先完成全部Writer cache再rollout，因此训练和正式
  correct400不受污染，只有交错capture/action的内部analyzer受影响。`5d93af3`以
  scoped lifecycle保存/恢复backend，定向`22 passed`，新root 8/8 tasks逐层、BA、
  action严格零误差重放。
- 有效refs1 root为
  `/data/ymdai/outputs/ember/pi05_as_writer_ap_adr_rawfull24_macro0150_internal_refs1_v10_5d93af3_20260801`；
  analysis/summary SHA为`d42fc4eb...bc2b`/`f2c572c5...e682`。same-task的
  `program_raw -> block2 -> program_read -> BA -> fixed action` relative L2为
  `.9188 -> 1.1051 -> .03210 -> .03005 -> .01668`；wrong为上游`1.3069`到
  `.1518/.1454/.02926`，而shuffled/reversed从block2`.09066/.07112`被压到BA
  `.002689/.003903`。顺序动态最早死在compiler read，而非video frontend或axial
  Program。
- 直接反转valid contextual temporal keys，保持Core、raw A/E/D、mask和position
  不变，BA/action只变`.000521/.001944`，8 tasks的BA范围仅
  `.000464-.000566`。ProgramReader entropy约`.904`，top mass约`.0106`；contextual
  stack对输出几乎只有微弱寻址权。
- A/E/D是“只保留所列value”的反事实。Effect-only在8/8 tasks重建full BA，平均
  relative L2`.008208`；Action-only/Change-only则为`.27610/.28320`，固定full
  contextual key后仍为`.27889/.28275`。Effect缩放0.5/2使BA变`.14099/.28910`，
  Action缩放最多`.00802`、Change最多`.00105`。因此raw Effect DC垄断V是直接
  证据，不是从aggregate反推；它重演的是v8已局部否定的Effect dominance，但发生
  在更晚的ProgramReader value接口，不等于否定Action evidence、Core或双reader。
- Core-only相对full BA/action差`.283/.228`，Program不是形式零分支；Program-only
  差`.961/.494`，mean-backed Core仍是共同semantic carrier。删除Core centered
  residual仅改变BA`.0128`，删除Core mean改变`.834`，与Recenter的semantic-basis
  starvation证据一致。下一架构应保留Core mean和separate reads，同时让causal
  contextual Program作为真实value content，而不是再加gate/scale/residual。

## 2026-08-01 同曝光recipe根因与AP-ADR启动

- UCP raw-full24 macro150的exact50已经从clean `b4207d2`自然完成：8 validation
  tasks × 50 references、每row五种视频条件、四rank各100 rows、0 rollout且信息墙
  读取0。analysis/summary SHA为
  `bc6e46209e6ffd9bb50ba0671ff63f5a7c03117c40e2ec6e4ea3c935fa8dba5b`/
  `cbe3760085e179561dd8aec33a89d2a1828d6d2e80b941c119d8479ff9600de4`。
  它与SERIAL step900严格匹配每task 150次video exposure；SERIAL analysis/summary
  SHA为
  `4d7479cb11d0bc6fc364bc02d5681503bb247cd411515f134f1526aed89f0fd7`/
  `68306aff90e973334ca5a972573725f7133a803386719089c5ab048c425a88c7`。
- 严格同曝光汇总已封存在
  `/data/ymdai/outputs/ember/pi05_as_writer_ucp_raw150_vs_serial900_matched_exposure_internal_audit_20260801`；
  analysis/evidence/summary SHA分别为
  `e8cdbc79049079017bb5a38681c746c79e17c60c7ee5a3b85977282efd6dc922`、
  `b44cdec0f4a8de5b0f394cd8336047548864a5928306059b6a67ccd104a84859`、
  `dbc660cd01ece2482feaf12d6f5e6e8a4b6e166c2b811cb84b7a2e9ef3cb688d`。
  correct400为`100->121`，但`50 gained/29 lost`、breadth仍`5->5`且成功集合
  Jaccard仅`.4733`。action centered variance的跨task均值被Object-3的
  `5.7077%`强离群值主导；应以跨task中位数`.00837%->.01900%`和逐task行作
  正式解释，不能把均值的`22.7x`写成普遍增益。
- 同曝光训练更新粒度对视频动态写出有数量级影响。删除A/D、只保留absolute X时，
  raw→SERIAL的coordinates/effective-BA/fixed-action relative L2由
  `.1223/.0653/.01269`升到`.6244/.4184/.12999`；same-task 50-video centered
  variance/sample energy由BA/action `.1096%/.03230%`升到
  `.4865%/.7322%`，且BA视频变化的orthogonal fraction由`89.27%`升到`92.24%`。
  shuffled条件下固定X只换A/D的BA/action变化也由`.0336/.00682`升到
  `.1484/.04165`。因此full24一次平均确实会削弱UCP动态教学，而不是“架构根本
  不看视频”。
- 但SERIAL不是普适解：四个同曝光correct400差值
  `raw→SERIAL = +7/-17/+21/-3`，best只由117升到121，breadth和checkpoint漂移
  都未改善。150-exposure逐task从`[7,0,1,27,32,33,0,0]`变为
  `[25,0,0,36,38,21,1,0]`，四task上升但Object-3下降12，成功task集合仍轮换。
  结论是update-mechanics决定“视频创新能否写到action”，同时也会重排task能力；
  后续必须联合设计topology、scheduler、moment/clip时钟和去除long-first phase
  curriculum的grouped recipe，不能只把架构或full24单独定罪。
- 下一canonical Writer为Amplitude-Preserving Asymmetric Dual Read（AP-ADR），
  authority在`docs/action_forecast_writer_amplitude_preserving_dual_read_design.md`。
  它保留v5.2可证的mean-backed Semantic Core和coherent heads；Program使用outgoing
  `[A_f,G_(f+1),G_(f+1)-G_f]` raw values，38个target-only Core reads与38×16
  target/rank Program reads各自softmax，最后直接concat512生成A/B。没有terminal
  norm、AdaLN/gate/global mixer、谱约束或第二套LoRA。真实参数`10,241,024`。
- AP-ADR最长105-frame B20三macro profile通过：三步
  `20.567/18.717/18.644s`，峰值allocated/reserved
  `77,227,462,656/83,523,272,704` bytes；step2起semantic frontend、Core、
  Program、compiler、factor全部非零可达。formal seed fresh0→1→exact-resume1→3
  也通过，step1七个payload的size/mtime/SHA逐项不变。profile/resume seal已在
  `7dffb6f` push；正式首小时从fresh identity启动macro0→200，不继承smoke。
- AP-ADR fresh首小时随后自然完成macro0→200：200 cycles、96,000 queries、4,800
  one-video conditions、每task 4,000 queries/200 visits，wall `3898.217s`；200行
  metrics全部finite，validation/test action和test video读取均为0。macro50/100/
  150/200 paired correct400已分别在GPU4/5/6/7启动，四个prepared合同均为400
  states、correct videos无放回、36 long-first shards和6 replicas/6 generators。
- 同曝光macro175的模块动力学审计排除了“AP整个Program没训练”的粗解释：最后
  25步Program raw gradient L2仅为UCP的`.856%`，但Adam bias-corrected update RMS
  与累计位移仍为UCP的`71.18%/85.42%`。小梯度主要来自AP只用axial stack生成K、
  raw A/E/D直接走V的职责差异，不是full24额外抵消；AP/UCP Program mean-energy
  retention在末窗仍为`.859/.881`。
- 风险可进一步局部到temporal Q/K：其`sqrt(v)/Adam eps`只有`.135–.171`，累计
  位移仅约UCP的`18–26%`，而temporal output、local Q/K、ProgramReader、target/rank
  identities、Core和factor均有实质位移。因此最早候选接口是
  `temporal Q/K→contextual key→ProgramReader K/softmax`，但这还不是功能失败结论；
  必须用trained-vs-initial/time-permuted keys、attention routing和BA/action sensitivity
  与closed-loop共同判定。审计analysis/summary SHA为
  `1ee02ff2d2daf47dd76f8606d2c7de910cb7e11599848d2ba19faeada585c5a0`/
  `c4c79189e72470803fb1454d9b5388f8893a793d93ad574e7c0446232bac11fc`。
- post-v5×recipe正式只读审计又复核了八个formal commit：v7/v8/v10/Loom/
  Recenter/Core-Program/Prior/Target-Spectral的`as_step.py`、`as_sampling.py`和launcher
  blob逐项相同，全部只跑过full24/B20/fast400，没有matched alternate recipe。
  历史long-first只改变同一次24-task聚合中的rank内累加顺序，不是optimizer
  curriculum；SERIAL才首次把它变成六个真实optimizer phases。因此aggregate只可
  判定“架构×bundle”失败。Prior-Innovation局部因果证据尤其弱；可直接删除的仍只
  是global binder、早期8→1 pooling、terminal amplifier、无锚confidence/gap、硬删
  DC、strict product和强制谱。analysis/summary SHA为
  `a53f528ccf29931de415c1f52d058ed03897c4c6bf3e512c22bbd5066c45b229`/
  `f481f37e60669376e1876cda48d5a6e524303ca2289ce0da7987b2267ae62442`。
- 受控复核优先级因此固定为：先做无rollout的单-cycle update-operator replay，拆分
  raw mean、六次Adam clock与phase order；只有replay稳定支持才跑cycle-normalized
  randomized group4一小时；只有当前mean Action被内部定位为容量瓶颈，才在
  amplitude-preserving canonical路径移植8个Action anchors，不恢复v7/v8 binder。

## 2026-08-01 SPG一小时门、架构×recipe与UCP根因结论

- 严格surrogate审计确认七条训练曲线的512-row held panel manifest完全相同，
  SHA256为`53cbf9e...a3a8`；主20个正式候选的held loss→correct描述性
  Pearson/Spearman为`+.346/+.484`，按架构去均值后为`+.462/+.644`，held
  loss→breadth为`-.501`。16个相邻checkpoint差分中train25/held/norm对
  Δcorrect的Pearson仅`+.031/+.120/-.347`；逐`architecture×task`去均值后
  held→success仅`-.055`。重复checkpoint不满足独立样本假设，但SPG
  `100→150`、UCP `100→150`的held改善/行为下降与v5.2-new `200→400`的held
  恶化/行为上升构成直接反例。held functional loss只能做finite和局部拟合诊断，
  不能选择closed-loop checkpoint、解释task漂移或否定整版架构。正式analysis
  SHA为`91eaabed...12a`，120个输入文件和44个correct400 panel已复验。
- v5.2-new、SPG、UCP和v6-fast四个single-checkpoint winner在同一paired panel
  上的成功集合union为193、intersection为51；各自仍有不被另外三者覆盖的
  `18/6/9/13`个成功state。主20候选union为236；把v6延长到macro600后的扩展24
  候选union为246，而全部checkpoint intersection仅5。post-v5正式审计analysis
  SHA为`406b9098...80e`。
  这不授权checkpoint融合，但证明低aggregate版本仍可能保存真实独有能力；历史
  回顾必须定位局部失效接口和recipe依赖，不能按总分把整版结构一棒子打死。
- serial-4 formal从clean frozen `3db82df`自然完成1,200 updates/200完整cycles，
  wall `4197.076s`，96,000 queries和4,800 videos与raw-full24完全同曝光。1200行
  metrics、8个checkpoint、全部finite与信息墙读取0已核验；raw microtask→serial
  phase的4,800 assignments逐项匹配，replay SHA为`d406f2f1...80cc`。八个held
  loss在`.13035–.13348`间轮换，不据此预判closed-loop。
- architecture×training mechanics正式审计（analysis SHA `c910a933...e521`）确认
  old→full24不是简单的tasks/update开关：每完整exposure旧recipe做六次同LR
  mean4更新，一阶LR integral约为新mean24的`6.0069×`；Adam记忆保留变为
  `.9^6/.95^6=.5314/.7351`，并多做五次重线性化、clip和WD。最干净的v6
  old/new-slow虽然B20、视频序列、query和exposure-phase LR对齐，visits100→150
  参数路径cosine仍仅`.0493`，endpoint exp_avg cosine`.0331`。所以训练更新几何
  是真实根因bundle，但现有数据不能把aggregate、步长、moment、clip或顺序单独定罪。
- 真实触发频率进一步降低了clip/WD解释：匹配150 exposures时raw/serial LR sum为
  `.037808/.226848`；raw 150步没有clip，SERIAL 1,200步仅3次gradient norm超过1，
  且都发生在cycle20前。额外weight-decay累计收缩差约`1.9e-5`。因此后期动态
  写出差异更可信地指向六倍Adam/moment/重线性化时钟和phase-cost curriculum；
  normalized randomized group4必须同时消除LR积分和固定long-first phase年龄。
- UCP最长105-frame B20现场profile连续三macro通过，峰值reserved约77.62GiB；
  每步24 tasks、480 queries、24套单视频LoRA，step2起Program全链梯度可达。
  formal-seed fresh0→1→exact-resume1→3的step1 payload逐文件不变，证明新的raw
  full24/20-strata状态、optimizer、scheduler与每rank cursor可精确恢复。B20因此
  seal；这个smoke只证明工程合同，不预测closed-loop表现。
- UCP真实实现证明删除Core/mixer并不需要用新的硬瓶颈补位：单一raw-value
  Program→38×16 reader→coherent heads可以在`7,683,328`参数内完整生成全部public
  A/B，step0逐tensor严格identity；target/rank/type identities在零Program上不能
  造值，交换identity只交换对应coordinates。
- raw full24实现按全局task ID确定性排序，在world1/world2与rank/task排列变化下
  得到bitwise一致方向；没有projection、weight broadcast或direction allreduce。
  24×24 overall/block Gram、candidate dots和CountSketch不参与优化权重。
- 20-strata B20对每个task visit使用20个不同episodes、完整strata permutation和
  stratum内deterministic jitter；长期episode内row边缘近均匀，exact resume不需
  新cursor。它是估计器方差实验，不是语义phase监督。

- UCP raw-full24 macro50/100/150/200 paired correct400为
  `82/117/100/110`；breadth nonzero `7/7/5/7`，top2任务贡献
  `61.0/66.7/65.0/62.7%`，Spatial两任务始终近零。四checkpoint union为169，
  比single best117高52；三次相邻转移分别gained/lost
  `64/29`、`18/35`、`39/29`。这不是共同单调学习，而是明显能力轮换。
- trailing25 train functional loss从macro50约`.1157`持续降到macro200
  `.1006`，held panel则为`.13090/.13144/.13132/.13244`。m100 raw full24 mean
  只保留平均task-gradient energy的`5.26%`；四候选约
  `5.64/5.26/4.06/4.48%`。functional surrogate改善、closed-loop回落和约95%
  task innovation在平均中消失同时成立，但尚不能把二者写成唯一因果链。
- macro100 refs1证明UCP没有重演SPG compiler同质化：reader entropy约`.9541`，
  target/rank-centered attention energy`.240/.117`，coordinate对应量
  `.158/.0396`。wrong/shuffle/reverse的final Program relative L2
  `.492/.352/.447`传到coordinates`.253/.100/.159`、effective BA
  `.190/.0649/.107`和fixed action`.0674/.0157/.0298`。路由和视频信号均真实
  可达。
- 但显式dynamic教学仍弱：`x_only`相对full只改变effective BA`.0489`、action
  `.0111`；固定X只替换A/D时，wrong/shuffle/reverse BA仅
  `.0208/.0240/.0241`，action`.0062/.0065/.0094`。正确LoRA norm约`59.50`、
  stable rank`1.0031`、top energy`99.72%`、跨层q/v cosine`.917/.923`；不是
  Target-Spectral低norm/正交坍缩，却比v6高增益coherent流形弱。
- 初次refs1诊断把canonical五条件B5切成recompute B1，CUDA BF16不同batch shape
  导致coordinates约`.2%`系统漂移。四rank模式一致且误差在factor/policy前出现；
  同B5 canonical parity为0，排除Writer状态突变和owner遗漏。修复保留B5 carrier、
  只改/抽row0，所有消融也保持同batch；没有放宽`2e-5`阈值。新refs1真实运行的
  Program、coordinates、factor、A/B、BA和action重算误差全部严格0。
- 首次exact50没有形成科学负结果：rank1在本地reference阶段先异常，旧代码捕获后
  进入首个NCCL all-gather，而其余rank仍在local compute；600秒watchdog掩盖了
  原异常。精确schedule确认每condition无放回覆盖50 demos、sampled 17–68 frames，
  排除缺视频、非法长度和正常负载长尾。`874e5f1`改为reference级上下文、failure
  JSON与直接re-raise，并把成功同步交给两小时Gloo控制组。新refs2恢复出原始错误：
  rank1的`libero_spatial task3/reference1`在rank-gauge sanity失败，其他ranks被
  torchrun立即收割。instrumented `e47ffe8`给出raw A/B relative L2
  `.74184/.13602`，effective BA却仅`1.299e-9`、max absolute`7.45e-9`；说明同一
  rank置换的数学函数严格保持。fixed action relative L2为`.002047`、cosine
  `.9999978`，来自两段bf16 LoRA的rank-reduction顺序改变，而不是BA实现错误。
  因此sanity仍对finite和BA `2e-5` fail-close，同时把实际bf16 action drift作为
  诊断记录，不再要求错误的位级函数等价。失败root仍不能冒充UCP科学几何。
- 所以serial-4是有判别力但非预设成功的下一实验。四个近正交等norm梯度的mean
  energy基线本来就是`1/4=25%`，远高于full24的`1/24=4.17%`；因此selected4 Gram
  ratio上升只是干预生效的mechanical check，不能单独支持聚合根因。真正支持必须
  同时看到single-checkpoint breadth/Jaccard改善、envelope gap缩小、A/D→BA/action
  和same-video innovation不下降，并由多个tasks共同提高closed-loop。若只有order
  margin提高而absolute下降，就是重演v6-old；若loss下降但行为不涨，应转向
  surrogate/off-manifold；若A/D贡献仍只有2–5%，应降低聚合解释并重审UCP职责。
- serial-4实现按`cycle,phase=divmod(update,6)`逐项重建raw-full24的task、video、
  action-query和rank内顺序；1,200 updates仍是200 visits/task、4,800 videos和
  96,000 queries，LR只在phase5后推进。因此它隔离的是update-granularity bundle，
  不是纯“梯度抵消”：每cycle同时增加到6次clip/AdamW/weight decay/bias-correction/
  moment update，后续phase也在新参数点求梯度。另因每组按视频长度long-first，
  phase0长期先于phase5更新，计算顺序变成真实optimizer curriculum；若收益按
  phase/长度集中，必须降低聚合解释。4×4 Gram看不到跨phase20个task冲突，约25%
  retention只作mechanical manipulation check。
- raw-full24 200 cycles的4,800个真实video cost重放确认该curriculum很强：visit
  phase与sampled frames Pearson=`-.8331`，24-task mean相关=`-.8734`；phase0..5
  平均sampled frames为`64.62/41.05/32.42/28.88/25.73/20.88`，task38始终在
  phase0。serial结果必须按phase/cost共同审计。
- serial-4 live seal通过而未触发B16 fallback：最长105-frame、B20、18 updates/
  3 cycles全部finite，峰值allocated/reserved为`76,971,835,904/83,647,004,672`
  bytes；formal seed fresh0→1→resume1→3→resume3→7保持step1/3全部文件不变，
  cycle0六phase恰好覆盖24 tasks，step6才推进scheduler、step7使用下一LR。它只
  证明工程和状态合同成立，不预告closed-loop结果。
- clean `c4b85e8` refs2已经以8 tasks×2 references完整通过，16 rows均finite，
  analysis/summary SHA为`e0757f55...cc48`/`c7a42eae...da41`；随后同commit的
  exact50零rollout分析自然完成。8 tasks×50 references共400 rows、四rank各100，
  reference0..49完整且无failure；analysis/summary SHA为
  `a6e40cd6...25a8`/`386a04f5...acaa`。
- exact50 pooled same-task effective-BA/fixed-action centered variance/sample
  energy仅`.09008%/.01656%`。same/wrong/shuffled/reversed的Program→BA→action
  relative L2为`.215/.499/.356/.440 → .043/.187/.063/.105 →
  .0138/.0636/.0153/.0325`；固定X只换A/D时wrong/order BA仅约`2.1–2.5%`、
  action约`.55–.58%`。八task BA centered ratio均仅`.0520–.1568%`，所以dynamic
  教学弱是task-wide结论，不是refs2抽样误差。correct norm/stable rank/top
  singular energy为`59.108/1.00319/99.714%`。

- SPG macro50/100/150/200 paired correct400为`97/115/77/100`。envelope union
  为162，但best single point只有115；macro100→150 lost51/gained13，之后又
  反向轮换。它不续第二小时，也不做正式五臂。
- macro100 refs2证明Program不是断路：same/wrong/shuffled/reversed的Program
  relative L2为`.967/1.186/1.193/1.202`，到Program coordinates为
  `.355/.715/.627/.658`，到effective BA只剩`.066/.221/.116/.116`。固定Core、
  只改Program仍保留order差异。
- 最早失败是compiler同质化：CoreReader entropy`.999992`且target-centered
  attention energy`3.9e-5`；ProgramReader target/rank-centered routing约
  `4–5e-5`；coordinate centered content约`1e-5`。exact50 stable rank约
  `1.000001`、B columns近相同，same-video variance从m50`.419%`降到m200
  `.210%`。这不是Target-Spectral式低norm，而是identity被Core淹没、Core加法
  旁路和global mixer把强Program写成共享方向。
- SPG raw full24 mean保留平均单task gradient energy约`5.74%`，末25 macros
  `4.79%`；CP提高到`9.53%/6.99%`但仍丢失大多数非负近正交innovation。
  projected/raw cosine约`.983`且主要放大norm约`1.25×`；negative-pair投影没有
  解决drift，下一版恢复raw mean。
- 精确重建96,000条query：长期phase均值`.50015`，但4,800个task visits中
  `6.44%`漏至少一个五等分progress bin，单visit TV均值`.1756`、最大`.5`。
  20-strata随机permutation+jitter保持每条episode query边缘uniform，只作方差
  缩减，不把progress当语义阶段。
- v7/v8/v10/Loom及后续正式结果全部来自同一full24/B20/fast400 recipe，没有
  old-recipe反事实。可独立否定的是近均匀binder、早event pooling、无监督
  confidence/gap、DC删除、strict bilinear、高增益gate和强制谱；anchors、causal
  Procedure、双流、Core语义与target-first/rank-last不能整体判死。
- Git blob复核进一步确认：v7、v8、v10、Loom、Recenter、Core-Program、Prior和
  Target-Spectral八个正式训练commit的`as_step.py`与`as_sampling.py`分别完全同
  blob，正式root也共享同一source checkpoint、4-rank、full24等权、B20和fast400
  合同。因而不同aggregate不能反推出这些思想在其他更新粒度下也失败。
- 局部归因按强度分层：v7否定的是Core不进value的近均匀global binder，不是多
  Action anchors；v8否定的是过早`8→1` event pooling，不是Action/Effect双流；
  v10五臂`103/94/75/67/43`证明因果路径可工作，失败集中在tiny Procedure经
  RMSNorm/AdaLN放大13–20倍和breadth崩溃，因此其双流思想最值得保留；Loom的
  confidence/gap缺监督锚点，不会由recipe自动修复；Recenter的DC删除、strict
  Core-Program bilinear和Target-Spectral强制正交已有跨内部量的结构性反证；
  Prior-Innovation只证明手工硬分解失败，稳定semantic prior加软innovation仍未被
  单独否定。
- 匹配每task 150次video visit的正式2×2对照给出v5.2 old/new=`132/51`、v6
  old/new=`95/111`，recipe effect=`-81/+16`、描述性DiD=`97`；paired switch为
  v5.2 old-only/new-only=`90/9`、v6=`19/35`。这是强architecture×training-bundle
  交互，但optimizer count、scheduler phase和AdamW/moment时钟仍未匹配，不能把
  97冒充某个单开关的因果量。
- 更细的内部反事实限定了这些结论：v7 binder entropy`.99963`、有效anchors
  `7.998/8`且Core→BA仅`.001–.002`，所以不能靠serial更新补回缺失的Core value；
  v8 Effect/EventRead entropy`.978/.9967`且固定Effect换Action只产生约
  `.085–.103` event变化，而固定Action换Effect达`1.46–2.95`，否定的是Effect
  dominance和过早pooling。v10在同一full24-fast下仍有五臂
  `103/94/75/67/43`，直接反证“full24必然消灭时序特异性”，但其Procedure RMS
  `.0145`被末端放大`14–20×`，所以absolute失败仍有独立结构根因。
- Prior-Innovation的LoRA norm从m50到m200为`76.1→99.95`、跨层q/v cosine约
  `.97/.98`，coherent高增益本身健康；same-task centered variance/task-mean energy
  却只有`.052–.058%`。它没有隔离prior、innovation reader、final mixer与full24
  聚合，因此是历史上最符合“共享prior被保留、条件innovation被平均掉”的候选，
  而不是已经被整体否定的思想。
- 历史重访优先级必须以serial证据为条件，而不是按旧总分翻案：首先考虑保幅的
  v10双流/interleaved Procedure和Prior的软prior+target-local innovation；其次是
  保留grid、不早池化的Action anchors/局部关系；再后是target-first/rank-last配
  conventional coherent heads。Loom无锚点gap、exact Recenter、strict
  Core-Program和强制正交不能因换recipe自动恢复，除非先出现其失效内部量被修复的
  可识别证据。
- 如果UCP证明Program→coordinate→BA→action传递健康但absolute/breadth仍弱，
  下一步应先冻结拓扑做更新粒度单变量反事实，而不是立刻再改架构。最干净候选是
  `4 tasks/update × 1200 updates`：与full24 macro200同为4,800 videos和96,000
  queries；LR按`LR_serial(u)=LR_full24(floor(u/6))`做六update一档的严格
  exposure staircase。连续warmup102/decay2400会在同cycle引入六个不同LR，
  不是所需反事实。随后才把full24-slow2000作为独立scheduler变量；不得把4-task
  和slow scheduler重新混成一个“old recipe”结论。
- 下一UCP把absolute `X=M+G`、native Action和outgoing patch change放入统一
  causal Program；normalized target/rank单级直接读raw values，删除独立Core
  add、target-Core first hop和跨target mixer。训练用raw full24、stratified B20、
  fast400首段；不同时混入slow2000。

## 2026-08-01 v5.2 task-complete闭环结论

- macro150/200/350/400 paired correct400为`51/91/106/120`；winner macro400
  每task为`33/0/0/30/25/32/0/0`，只有4/8 tasks非零。checkpoint间不是共同
  单调增长：Long-6和Object-1从macro200后回落，Object-3上升，Spatial两task
  回到0，能力轮换仍在。
- winner五臂correct/same/wrong/shuffled/reversed为`120/109/107/111/124`；
  reversed反而高于correct，五臂breadth均未证明正确视频的额外闭环收益。因此
  本轮v5.2没有行为视频特异性，不能因absolute高于source base就宣称教学成立。
- exact50 correct几何的effective norm/stable rank/top energy为
  `113.5185/1.000305/99.9699%`，q/v/action能量为
  `72.701/27.269/.030%`，q/v跨层cosine`.9723/.9837`；16坐标仍建设性参与。
- same-task centered variance/sample energy只有`.6844%`，低于旧v5.2的
  `1.6655%`，但其中约`91.22%`仍是orthogonal-direction变化。五条件内部反事实
  确认Procedure差异传到effective BA和fixed-query action；失败不是数值链路
  完全断开，而是变化方向没有与closed-loop收益对齐。
- functional validation在macro200更优而behavior继续涨到macro400；微小参数
  位移伴随大量gained/lost states。当前证据降低“只需继续训练/放大视频信号”的
  可信度，优先支持functional surrogate与有效policy manifold错位、task轮换和
  单视频条件创新缩弱。

## 2026-08-01 SPG B20与CP-24现场证据

- 最长105-frame、B20、四rank的三完整macro profile在CP通信修复后连续完成：
  `20.5359/18.5778/18.5461s`，1,440 queries、72 videos全finite；峰值
  allocated/reserved为`77.20/83.53GB`。
- 原始profile的macro1完成后，macro2在共卡条件下长时间无进展。逐phase trace
  证明NCCL同步Python调用只把bounded Gram all-gather排入CUDA stream；快rank可
  排入全部13 chunks而慢rank尚未进入首chunk，故“分块”没有形成完成边界。
- canonical修复是在每个CUDA Gram chunk后显式同步当前stream；CPU/Gloo不走
  该路径。修复不改变Gram、projection或optimizer数学，只保证所有rank在同一
  bounded collective上前进。原始最长profile重跑随后稳定通过。
- 三macro negative pair fraction为`.4058/.3514/.4058`，raw/projected cosine
  `.8410/.9426/.9689`；macro2起frontend/Core/Program/compiler/factor梯度均
  finite非零。真实task负冲突存在，但只有正式checkpoint漂移曲线才能判断CP-24
  是否解决科学问题；工程profile不能冒充closed-loop收益。
- clean `f6d4876`上的formal-seed fresh0→1→exact-resume1→3通过；step1全部文件
  哈希不变，三步loss`.152172/.147053/.154108`、gradient norm
  `.031343/.072098/.192859`、chunk gather/sync均`13/13`。恢复合同完整，但同样
  不构成行为证据。
- pushed clean `79fb7ee`的正式fresh0→200已挂起。首macro loss/grad/LR为
  `.152172/.031343/1.6667e-5`，step wall`19.431s`；raw conflict negative pair
  fraction`.3804`，candidate负task由4投影为0。这里只证明正式run与profile合同
  一致，不提前推断macro50行为。

## 2026-07-31 v5.2 task-complete正式训练完成

- clean frozen commit `60f4508`上的exact v5.2 topology完成fresh macro0→400；
  Writer参数`10,237,704`，每macro 24 tasks等权、每task一条video/一套LoRA/B20，
  共`192,000` action queries与`9,600`单视频条件。
- 训练wall `9695.1329s`；macro400 functional train loss `.09633848`、grad norm
  `.10484845`、LR `1.00045e-5`，functional validation loss `.13686878`。全程无
  OOM/NaN，validation/test action读取为0；这些只证明训练合同，不预测行为。
- macro150/200/350/400的paired panel固定8 validation tasks×50 states、teacher
  demos每task0..49无放回，B-scale1且共用state/env/policy/video schedule；
  最终结果与机制结论见上方2026-08-01段。

## 2026-07-31 SPG独立复核与canonical实现

- 逐层复核确认`A_f`与`G_(f+1)-G_f`共享interval `f→f+1`，local轴只在同一
  interval内读Action与task-token change，temporal轴按每个语义列使用严格
  causal mask；prefix/ragged测试锁定无未来帧泄漏。
- Core保留v6式mean backbone与task-selected centered residual，完整task-token
  容量且对frame permutation不变；Program不提前池化，38个真实policy targets
  先于16个rank coordinates，target/rank/order/type identity只进入Q/K。
- compiler用每个target/rank query直接读取完整Program，随后用标准axial
  coordinate mixer允许必要分化，同时没有正交、谱均匀、scale gate、B-only
  residual、terminal norm或第二套LoRA；八个factor heads零初始化保证step0严格
  functional identity。
- CP-24保留24-task等权完整macro。每task gradient先保存，确定性投影只修正负
  pair，最后仍一次clip、一次AdamW；无冲突单元测试严格退化为原始full24 mean，
  四rank用rank0权重广播和最终all-reduce保持同一更新。
- 真实module enumeration为`10,633,216`：frontend `3,453,440`、Core
  `1,836,544`、Program `1,837,568`、compiler `1,326,592`、factor heads
  `2,179,072`。source policy trainable参数为0，public rank16仍是38 targets/
  76 tensors/`1,287,168` scalars。
- 初始实现全仓`201 passed in 26.18s`；shape、mask、Core permutation、Program prefix、
  identity、freeze、gradient、CP world2和checkpoint schema均通过。architecture
  guard为REVIEW但无hard violation。真实B20 profile后续结果见上方2026-08-01段；
  fresh/exact-resume与性能实验在本段时间点仍未执行。

## 2026-07-31 SPG整体架构定稿与持续迭代合同

- Coherent-Procedure/B-only residual已撤回；它只是保守对照计划，不是owner要求
  的整体架构。
- 新SPG用mean-backed Semantic Core、未池化Action×task-token-change axial
  Program Grid、38真实policy targets、rank-last Program read、target/rank axial
  mixer和coherent full-width factor heads重建完整Writer。
- SPG不等待v5.2新recipe结果决定拓扑。接手session先立即挂v5.2正式400-macro
  轨迹，然后充分阅读仓库；无论v5.2好坏都实现SPG。
- 后续每版整体架构先一小时；只有达到同期有效旧架构水平或显示明确续训价值才
  开第二小时与行为五臂，否则做充分内部分析后从根因重构。150不是自动终点。
- 设计authority为`docs/action_forecast_writer_semantic_program_grid_design.md`。

## 2026-07-31 v5.2 LoRA几何

- exact v5.2 step900的400套correct-video LoRA已在零rollout条件下重新生成并
  分析。effective norm `140.441`，stable rank `1.01256`，top singular energy
  `99.0244%`，q/v能量`73.45/26.55%`，跨层BA cosine`.962/.982`。
- q/v固定坐标使用`15.83/15.92 of 16`个能量坐标，最大坐标仅`7.09/6.84%`，
  负component pair为0%。因此近rank1是建设性同向协作，不是能量不均或负相消。
- same-task exact50中心化方差占sample energy `1.6655%`，其中`89.35%`是正交
  方向变化；v5.2确实生成video-specific方向，而非只调共同scale。
- Target-Spectral把rank强制抬高却降到`34/400`，因此正交、均匀奇异值和强制
  使用全部rank均为负方向。Source-SFT与Writer的目标/参数化不同，不能逐项复制
  Source-SFT的谱。
- v5.2×task-complete仍是必须立即完成的重要因果格，但不再决定是否设计SPG；
  它为SPG和后续迭代提供最新强baseline与训练对照。

## 2026-07-21 当前 π0.5 协议与已验证事实

- 活动目标split为四个标准LIBERO suites、每suite 6 train / 2 validation / 2 test，总计24/8/8；final合并为32 source / 8 test。
- generic `lerobot/pi05_base` revision `7de663972b7817d2c4cf2d84c821153dfea772e9` 已完整下载；weights SHA256 `0eb11ca9587678c1d2ef8cf32807c29f8ce53a2bfdfc1aa4a4c96f16fca59b0f`。
- generic base在8 test tasks×50 official fixed states上全部为0/50，总计0/400；400个`(suite, task, init_state)`唯一且全部到suite horizon。该结果只说明原始π0.5没有LIBERO执行能力，不评价EMBER。
- result aggregate SHA256为`8ffa816e...7776`；tracked seal为`configs/libero_24_8_8_v1/pi05_base_feasibility_results.json`（SHA256 `c78e92e9...20c2`）。
- 24-task interface normalization和公开OpenPI tokenizer已完成并核验，但它们只属于generic feasibility合同；新source base将从过滤后的LIBERO-90 source corpus重新计算并冻结自己的source-only stats。
- batch1/8/16 profile分别约27.52、19.76、19.58秒/episode，8→16只快约0.9%。正式静态task/GPU运行中Spatial约1004秒，而horizon-520 task最长约2169秒，证明后续必须优化cost-balanced sharding而不是只加batch。
- EGL rank映射错误已在commit `bf27ebc`修复；正式8卡运行每卡一个CUDA process且全部exit0。

## 2026-07-21 LIBERO-90 source overlap seal

- 完成90 source×40 target共3600对full-task specification audit。只读language、BDDL、objects、roles、initial predicates和ordered composition；没有读取任何numeric action/state、reward、terminal、normalization或policy outcome。
- 排除19个完整semantic/composition等价source IDs：`8,9,10,20,25,27,30,31,44,46,47,48,49,50,51,52,53,54,77`。除已知44/77外，audit捕获Goal与Object中的别名和不同scene复现；过滤规则与逐pair理由已封存。
- 保留71 tasks；primitive containment、不同object multiplicity、source/destination selector或额外ordered subgoal不算full-task overlap。IDs `2,29,12,13,14,15,38`是人工复核并保留的关键near misses。
- 71×50 successful episodes全部存在，共3550 episodes、529,173 frames、52,710,755,898 bytes，aggregate HDF5 SHA256 `81bdb358...a1a50e`；source-only normalization读取这些source numeric rows且validation/test numeric reads为0。
- seal hashes为audit `fe731127...cc003`、manifest `75453a20...2e54`、normalization `e259ee6e...f7c4`和recipe `4c537067...281734`。recipe hash更新只封存pinned OpenPI缺失右腕的zero-image/false-mask相机合同，不改变source IDs或任何outcome；后续outcome不得反向修改source IDs。

## 2026-07-21 π0.5 source-base recipe、profile与resume

- 官方full-SFT anchor来自pinned OpenPI/LeRobot：global batch256、30k steps、AdamW betas `(0.9,0.95)`、eps `1e-8`、weight decay `1e-10`、clip1、peak LR `5e-5`、10k warmup后constant、EMA `0.999`。EMBER source base采用full-SFT，不使用`pi05_libero`，不叠未merge shared adapter。
- 修正后的canonical 8×A100 m32+EMA smoke使用pinned OpenPI的`q99-q01+1e-6` quantile公式，并省略右腕feature key，使LeRobot产生zero padding且`image_mask=false`。3/3 steps的loss/gradient finite，steps2–3平均47.75 global examples/s；峰值allocated/reserved为67,178,351,616/71,179,436,032 bytes，约保留10.7GB稳定余量，故锁定global batch256且不做gradient accumulation。旧m1/m4/m16/m32对比profile把显式zero右腕误标为`mask=true`，只保留工程provenance，不再作活动launch证据。
- 第一次formal启动的CUDA进程拓扑正确，但live `PSR`显示rank未受GPU-local NUMA约束；在step12、首个checkpoint前主动终止，exit130。该root仅保留20KB failure evidence，run contract/metrics/log hashes分别为`997af43a...8b2`、`81dbfcbc...4ca`、`7a169300...118`，不得resume或作科学结果。
- 修复后每rank在CUDA初始化后立即绑定sysfs GPU NUMA cpulist，DataLoader children继承：rank0–3为`0-37,76-113`/NUMA0，rank4–7为`38-75,114-151`/NUMA1。相机mask修正后的3-step m32+EMA smoke exit0，contract/metrics/summary/log SHA256分别为`90fbe1da...0458`、`de2d9889...50d9`、`0a590a29...e1bc`、`26bb5aad...c10`。
- formal attempt2在step316、80,896 global examples处被主动终止：根因是训练与评测都显式传入zero右腕，LeRobot因此把第三相机标为`mask=true`，而pinned OpenPI LIBERO policy要求zero image加`mask=false`。该run无checkpoint、不得resume或作科学结果；failure packet/run-contract/metrics/log SHA256为`2d2a9e40...9b80`、`e79e1c84...e7d8`、`fb0b2edc...f918`、`3f0eb65f...76f7`。修复是省略feature key而不是另建相机路径，训练和评测共用这一唯一合同。
- canonical runner严格加载weights，避免上游异常时静默返回随机模型；每个checkpoint封存policy、EMA、optimizer、scheduler、8个rank RNG/sampler states、metrics cursor、contract和文件hash，并在新checkpoint原子发布后才清理旧状态。
- 两个独立8-rank进程均从同一step1 manifest `0461dee1...5953`恢复；step2 loss、grad norm、LR、cursor和8个rank state文件完全一致。4,143,404,816个policy元素中0.0308%仅有独立NCCL启动末位差，max `1.49e-8`；EMA max `3.73e-9`。这支持state/cursor exact且numerically reproducible的resume合同，不虚假宣称跨新distributed process bitwise identical。
- 三套约32GB probe checkpoint在compact evidence封存后按500GB cap永久清理；保留evidence packet为444KB，comparison SHA256 `16137fa1...b1e`。清理后个人占用379,942,686,720 bytes，atomic双checkpoint峰值估计约447.62GB。
- 2026-07-22 owner将训练预算口径明确为短周期、证据驱动：先profile学习速度/吞吐，按曲线斜率安排廉价固定screen，只给少量候选完整validation，接近饱和即停。约120分钟只是所有适用训练阶段的防失控上限；到上限仍未充分训练则封存曲线与budget-censored判断，不自动追加。task-local RL按每个初始化方法在全部8个test tasks上的合计训练时间计费。

## 2026-07-21 canonical π0.5 target evaluator

- 唯一活动入口为`scripts/evaluate_pi05.py`；旧静态`scripts/evaluate_pi05_base.py`已退役，不从Git历史恢复。`pi05_eval_contract.py`拥有sealed authorities、source final-EMA门与seed schedule，`pi05_eval_queue.py`拥有cost-balanced SQLite队列，`pi05_evaluation.py`拥有persistent policy/env与rollout，`pi05_eval_results.py`拥有worker证据与strict aggregate；这些是单一runner内的故障边界，不是并行实现。
- 40-task screen按`states × suite horizon`切成近等cost shards，8卡使用相同1/2/3 replicas并动态claim/work-steal；每worker持久加载一套policy和当前task env pool，GPU0没有额外CUDA controller。
- formal/screen只接受与当前source config、全部model/tokenizer/recipe authorities完全一致的final step1000 EMA；test `.pruned_init`逐项对sealed protocol hash。worker在load前重算model/tokenizer SHA，raw shard、DB counts与producer/claim均交叉核对。
- launcher在任何queue recovery前独占lock；局部spawn失败只终止本launcher创建的workers并保存PID、logs、failed jobs与hash。吞吐主指标从首worker进程spawn到全体退出，包含model load与首次env/EGL创建；另报shard-only window，避免1/2/3 replicas profile偏置。

下文全部SmolVLA/70-10-10证据仍是真实历史，但只作provenance，不能驱动当前π0.5训练或复用旧checkpoint/normalization/runner。

---

本文件只保留会影响当前科学解释的证据。父提交 `999df28` 保存 2026-07-17 至 2026-07-20 的完整逐次日志、旧配置、runner 和测试；外部 checksummed artifacts 保存原始 rows、metrics、视频和 failure packets。这里不把历史过程重新伪装成活动合同。

## 历史 70/10/10 结论（已退役）

### 新 70/10/10 protocol：已永久封存

- 在读取任何新协议 policy outcome 前，使用 90 条官方 language、scene 和 role factors，通过 `scipy_milp_highs_three_stage_lexicographic_v1`、seed `20260720` 封存 70 train / 10 validation / 10 test。
- validation IDs 为 `[0, 8, 15, 28, 40, 56, 61, 71, 85, 88]`；test IDs 为 `[4, 7, 11, 32, 41, 59, 60, 70, 84, 86]`；其余 70 个为 train。
- validation/test 各自含 10 个不同 scene，并恰好共享相同 scene 分布：5 Kitchen、3 Living Room、2 Study；两者均为 5 个单步/5 个双步、2 actuation / 3 single-place / 4 pick-place / 1 compound。
- exact composition group 不跨 split；所有 held task 的每个精确 role atom 在 train 中至少保留 2 个实例。stacking 因这一严格支持约束全部保留在 train。
- 90 个 pinned HDF5 共 `66,658,085,995` bytes，4500 demonstrations、669,043 frames、每 task 50 demos；90 个 official init-state 文件均为 50 states。controller 为 OSC_POSE/20Hz，camera 为 agentview + eye-in-hand、128×128。
- validation/test HDF5 只读取 metadata/shape/hash；normalization 仅从 70×50 train episodes 读取 state/action 数值。producer `env_args` 有 90 个 legacy suite 注记和 6 个 legacy basename 注记，但 canonical HDF5 BDDL basename/language 均通过。
- canonical hashes：factor table `73828b1b...015`、split `996a3061...77e`、data manifest `b18f1cfa...be7e`、train-only normalization `5141e4b3...2d28`；完整值在 `configs/libero90_70_10_10/checksums.sha256`。

### Source-base 正式训练与 source-only 选择：完成，冻结 step630

- 从 pinned `lerobot/smolvla_base` 严格加载 450,046,176 parameters；98,880,992 个 action-expert/projection parameters 可训练，冻结 VLM trainable 泄漏为零。
- 全部 70×50 episodes 对应 537,946 frames；sampler 在跨 rank global task slots 上做 deterministic no-replacement cycles，并保证每个 checkpoint 边界前 3500 episodes 全覆盖。
- 显式 rank-local device 修复后的 8-A100 profile 使用一张卡一个 rank、batch/rank 352：稳态 2.590s/step、1087.2 global samples/s，每卡峰值 allocated/reserved 65.05/67.35GiB，data wait 0.13ms。
- 首次 formal 启动暴露出无索引 `device="cuda"` 会让非零 ranks 在 GPU0 留额外构造 context；它在首个 checkpoint 前被停止且不复用。改为 `cuda:{local_rank}` 后，满 batch steady-step 进程表为每卡恰好一个 CUDA PID、69,124–69,132MiB，GPU0 不再额外堆积。
- 8-rank continuous/resume 对照中，step-1 policy、optimizer、scheduler 和每 rank RNG 起点一致；启用 DDP static graph 后，step-2 policy 文件 SHA256 位级一致，optimizer/scheduler/RNG 逐值一致。默认 DDP 首轮后 bucket 重建是此前微小漂移的工程原因。
- commit `72eb10d` 上的正式 seed-1 trajectory 在约 28 分钟内完成 630/630 steps、退出码 0；210/420/630 三个 checkpoints 均通过 15-file size/SHA manifest 校验。最终 step loss 为 `0.483089`，吞吐 `1084.95 samples/s`，峰值 allocated/reserved 为 `65.05/67.35GiB`。
- step 630 累计 1,774,080 global examples 和 5,040 global task slots；每个 checkpoint 边界均覆盖全部 70 tasks 和每 task 50 episodes。最终 policy SHA256 为 `eb7e01f2...c1f159f`，checkpoint manifest SHA256 为 `89e9f493...ed22c`，launch contract SHA256 为 `22c4ffb5...2e8`。
- source-only loss 三段均值为 `0.80098 → 0.53083 → 0.49775`。同一 8-task × 50-state source-development panel 的 step 210/420/630 h50 成功数为 `3/400 → 8/400 → 15/400`；420→630 是 11 paired gains、4 paired losses，行为改善不只来自总数拼接。
- step-630 per-task 原始成功数为 `{1:0, 2:3, 6:6, 16:4, 46:1, 63:1, 65:0, 73:0}`，说明它在 5/8 个预声明 source tasks 出现 competence，但 3.75% 绝对成功率仍低；因此只允许一次短续训检验是否仍在改善。
- step 210/420/630 `results.json` SHA256 分别为 `b1445ec2...b3893`、`ba8fdbb5...e8cdb`、`b901f758...5ffa4`；每份均为 400 个唯一 `(task_id, init_state_id)` rows，8 ranks、50 states/task、horizon 400，无跨 checkpoint 拼接。
- 只据上述 train/source evidence，封存一次 315-step continuation：从 step 630 完整恢复 optimizer、sampler/RNG 与 interaction-free data cursor，原 cosine scheduler 保持在 decay LR `2.5e-6`，相对 thirds 为 735/840/945。没有读取 validation/test outcome，也不重启高 LR。
- continuation 在 step735/840/945 均完整覆盖70×50，最终累计2,661,120 examples，8卡仍各恰好一个70,176MiB CUDA rank，exit 0；三个 checkpoint 的15个文件均通过 size/SHA 校验。
- step945 source-development 仍为 `15/400`，per-task `{1:0,2:3,6:6,16:5,46:1,63:0,65:0,73:0}`；相对630为5 paired gains、5 paired losses、10 kept successes，净增益0。按预声明停止规则冻结step630，不再评735/840；该选择及完整 artifact hashes 已封存在 `configs/source_base_selected_v1.json`。

### 新 h50 fresh evaluator：mechanics 通过，未打开 test

- evaluator 只暴露 specification-only 预声明的 8 个 train/source-development tasks 和 10 个 validation tasks；reporting-only test role 在 Phase F 前结构性不可解析。
- 使用官方 LIBERO suite/BDDL/controller/camera/normalization、固定 `.pruned_init` states、dummy settling 10、horizon 400、成功即终止和 SmolVLA h50；固定 states 只服务 fresh evaluation，不会进入 RL update 或 adaptation checkpoint selection。
- step-630 mechanics smoke 在 8 ranks 上各跑 1 个不同固定 state，共 8 条唯一 rows；运行时每卡恰好一个 policy CUDA process、显存一致为 3347MiB，退出后全部归零。`0/8` 只是 smoke 小分母，不作性能证据。
- 完整 source-development/validation 评估按 task 同步、state rank-strided、每 rank 4 个持久 async env workers；这使八卡 policy 进程拓扑完全对称，同时把 MuJoCo rollout 吞吐作为优先优化对象。
- source base 冻结之后才打开 validation reference：step630 在 10 tasks × 50 fixed fresh states 上为 `56/500 = 11.2%`，per-task `{0:28,8:0,15:0,28:14,40:0,56:1,61:0,71:0,85:0,88:13}`。结果集中在三个任务，既提供非零 competence，也要求 Writer 的增益必须跨多个类别而非只追随单一易任务；`results.json` SHA256 为 `3d19f00f...0cac9`。该结果没有参与 source-base 选择，test 仍未打开。

### Frozen Writer feature cache：正式 70×50 cache 完成

- 只读取每条 source teacher episode 的 `obs/agentview_rgb` 和 language；不读取 action、proprio、reward、terminal、task ID/file-name features。每帧按 source-base 相同 OpenGL transform 进入 frozen SmolVLA VLM，64 个 960-d 空间 tokens 经固定 `sqrt(960)` normalization 后确定性均值池化为一个 960-d BF16 frame feature。
- 8-rank smoke 为每 rank 1 个不同 train task/episode，共 1,194 frames；所有视觉/语言 features finite，episode offsets 与原 episode lengths 一致。单 task 108–197 frames 的提取 wall time 为 0.63–0.83 秒，按正式 LPT 调度估计全 537,946 frames 约 5 分钟。
- resume 再运行时 8/8 ranks 均验证既有文件 size/SHA 后 `new=0`；模型加载阶段每卡恰好 1 CUDA PID、414MiB，GPU0 无额外 context，退出后 8 卡全清。
- selected step630 的正式 cache 已完成：70 tasks、3500 episodes、537,946 frames、1,034,531,040 tensor bytes、825 language tokens。70 个 task tensor 均独立通过 size/SHA；cache manifest SHA256 为 `ae5854a6...be127`，run contract 内部 SHA256 为 `7b7fb765...e03ce`。全程每卡一个 3900MiB CUDA rank，GPU0 无额外进程。
- validation action-hidden cache 使用同一冻结 VLM 合同，仅读取预封存 10 tasks 的 RGB/language：500 full episodes、63,544 frames、122,236,320 tensor bytes。10 个 task tensor 均独立通过 size/SHA；manifest SHA256 为 `06087541...05221`，extraction SHA256 为 `65d275d0...4a11`。8 卡仍各一个约 3900MiB CUDA rank。

### Writer functional cold-start：真实 profile/resume 通过，formal 已封存

- Writer 生成的全部74个 LoRA A/B tensors 已通过 `torch.func.functional_call` 接入冻结 SmolVLA 标准 flow/action loss；单元 backward 验证 policy 所有物理参数无梯度、Writer 获得 finite gradient。
- source query sampler 继续使用跨rank 70-task no-replacement cycles，并可对每个 checkpoint 生成精确 `(step, rank, batch offset, task, episode, frame)` identity SHA256；不是只保存不可审计的 step 数。
- feature cache 训练侧使用有界 task LRU，每个 task 首次载入验证 size/SHA，换入时不重复做15MB级哈希；这是吞吐优化，不改变 features。
- canonical runner 保存 Writer、optimizer、scheduler、sampler cursor、consumed identity、每rank RNG和完整 launch contract。
- 真实 8-rank functional profile 选择每 rank batch 384（global 3072）：steps 2–35 平均 3.426s、898.1 queries/s，峰值 allocated/reserved 68.00/70.54GiB；8 卡各恰好一个约 72.6–73.4GiB CUDA rank，GPU0 没有额外 context。更大的 448/512/768/896 batch 均在 8 卡对称 OOM，因此不再为小幅吞吐继续挤压约 10GiB headroom。
- step17 checkpoint 的 Writer、optimizer/scheduler 与 8-rank RNG 共 10 个文件均独立通过 size/SHA；从它恢复至 step35 后 loss、吞吐和显存连续稳定。35 steps 恰好为 4 个完整 70-task cycles，每 task 1536 queries、50/50 episodes 覆盖，consumed identity SHA256 为 `59804c03...6db01`，step35 manifest SHA256 为 `835f9758...ec15`。
- 正式 cold-start 已封存为 1575 steps、525/1050/1575 thirds；按 profile 预计纯训练 89.93 分钟。profile 只证明 mechanics 和资源合同，不能作为 Writer 行为结论。
- fresh evaluator 已能从 checkpoint 与 validation cache 生成、注入同一 37-target LoRA。profile step35 的真实 smoke 中，8 ranks 对 task0 生成的 adapter SHA256 均为 `067780eb...aa421`，8 个固定 state 各一条、设备/EGL 映射完整并 exit 0；小分母 `6/8` 明确不作性能证据。

### Writer functional cold-start：formal seed 1 完成

- commit `69bbdee` 的正式 trajectory 完成 1575/1575 steps、exit 0，约 92.9 分钟；累计 4,838,400 global queries。最终 70 个 train tasks 各精确消费 69,120 queries，并各覆盖全部 50 episodes。
- 525/1050/1575 三个 checkpoint 均在完整 70-task cycle 边界保存；最终 checkpoint 的 10 个 Writer、trainer 与 rank-RNG 文件共 150,436,331 bytes，逐文件 size/SHA 校验通过，manifest 为 `c30c49af...3357`，consumed identity 为 `2029f311...4112`。
- 全程每卡一个 Writer CUDA rank，GPU0 无额外模型/controller context；最终 peak allocated/reserved 为 68.00/70.71GiB，最后一步吞吐 916.1 queries/s。训练 loss 与机械完成本身不作为 Writer 功能价值结论，行为结论只取预封存 validation rows。

### Writer cold-start：首个 validation policy RNG 选择 step1050，但增益仍边缘

- frozen source base 为 `56/500`；Writer step525/1050/1575 分别为 `58/500, 63/500, 60/500`。全部结果各含 500 个唯一 `(task_id, init_state_id)` rows，环境与 policy seed 逐 row 匹配 base，每个 task 的 adapter hash 在 8 ranks 间唯一一致。
- 预封存排名选择 step1050。它的 per-task 原始成功数为 `{0:19,8:0,15:0,28:24,40:0,56:1,61:0,71:0,85:0,88:19}`；相对 base `{0:28,8:0,15:0,28:14,40:0,56:1,61:0,71:0,85:0,88:13}` 是 `31 gains / 24 losses / net +7`。
- 正增益落在 KITCHEN-actuation task28 `+10` 与 STUDY-pick-place task88 `+6`，但 KITCHEN-actuation task0 同时 `-9`，aggregate 只增加 1.4pp。因此这足以确定后续 cold initialization，不足以声称 Writer 已明显跨类别优于 base。
- selection 与所有 artifact hashes 已封存在 `configs/writer_cold_start_selected_v1.json`。根据既有“policy sampling 方差可能左右判断时加第二 RNG”规则，已在查看新 outcome 前封存 `configs/source_base_eval_rng2_v1.json`；它只比较 base 与已选 step1050，不能重新选择 checkpoint，test 仍未打开。

### Writer cold-start：第二 policy RNG 复现两类功能信号，同时确认覆盖有限

- RNG2 的 frozen base/selected Writer 为 `51/500, 57/500`，配对 `30 gains / 24 losses / net +6`；RNG1 对应为 `56/500, 63/500` 与 `31/24/+7`。两组使用同一 50 fixed states 和 env seeds，policy seeds 完全不相交，selected checkpoint 没有重选。
- 两 RNG 合并后 base/Writer 为 `107/1000, 120/1000`。task28 为 `26/100 → 48/100`，配对 `32 gains / 10 losses / net +22`；task88 为 `24/100 → 36/100`，配对 `18/6/+12`。它们分别属于 KITCHEN-actuation 与 STUDY-pick-place，且增益方向在两个 policy RNG 中逐次复现。
- 同时，task0 从 `55/100` 降到 `35/100`，配对净 `-20`；task85 净 `-1`，其余六个任务净零且多数双方均无成功。故可支持“Writer 在两个不同未见类别产生真实即时功能价值”，不能支持“已广泛泛化”。aggregate 只增加 1.3pp，Phase D/后续 matched RL 需要原样报告这项局限。
- 完整 RNG2 合同、result hashes、逐任务配对数与解释封存在 `configs/writer_cold_start_rng2_confirmation_v1.json`；test 从未打开。

### Validation direct-LoRA：formal oracle 完成，LoRA acquisition ceiling 明确非零

- 8 个 validation tasks 各由一个独立 GPU rank 从同一 frozen source base 训练自己的 37-target LoRA；每卡恰好一个 CUDA process，batch384 的 peak allocated/reserved 为 67.09/69.09GiB，实测最慢 step 2.816 秒。
- 每 task 在 step1 保存后由新进程 exact-resume 到 step10；两个边界的 16 个 checkpoint manifests 均逐文件验证 LoRA、trainer 与 RNG state。step1 已覆盖全部 50 teacher episodes，step10 每 task 消费 3,840 queries。
- profile 只验证 mechanics、恢复和资源合同，不看小步性能。formal 每 task 消费 69,120 matched queries，即 batch384 × 180 steps，checkpoints 60/120/180；10 个 validation task 都固定使用 final step180，不按 policy outcome 选择。正式训练约 17.5 分钟，30 个 checkpoint manifests/files 全部通过哈希与 episode coverage 审计。
- 同一 500 条 validation rows 上，frozen base / cold Writer / direct oracle 分别为 `56/500, 63/500, 186/500`。direct per-task 为 `{0:48,8:1,15:17,28:36,40:21,56:11,61:9,71:2,85:11,88:30}`，相对 base 配对 `141 gains / 11 losses / net +130`，相对 cold Writer 为 `138/15/+123`；它在 10 个 task 上都取得正的 raw count gain。
- 因而当前 37-target LoRA 空间并非整体无效，且 target-action acquisition ceiling 充足；cold Writer 与 oracle 的主要差距是跨任务 acquisition/generalization coverage。direct 使用目标 action，只是 oracle/reference，不属于同信息墙主结论，也不驱动 Writer checkpoint 或 test 选择。合同与结果 hashes 封存在 `configs/direct_lora_validation_reference_v1.json`，test 未打开。

### Writer-only RL：formal 完成，但 source reward 使 validation 单调退化

- cold step1050 起点从 update1 checkpoint 由全新 8-rank 进程恢复到 update9，完整覆盖 70 source tasks。每 task 恰好 4 个官方随机 reset rollouts，共 280 interactions、87 successes、90,391 env steps 和 9 个 Writer optimizer updates；生成 LoRA 没有原位更新。
- 72 个 rank/update ledgers 全部声明 `official_random_reset=true`、`fixed_init_state_id=null`；70 个 active task ledgers 合计 280 个唯一 `(task, env_seed, policy_seed)` rows。update1/update9 checkpoints 各含 Writer、trainer 和 8-rank RNG 共 10 个文件，逐文件验证通过，最终 interaction cursor 精确为一个 full cycle。
- max-rank cycle wall 为 405.50 秒，最慢 update 49.73 秒；reward updates 的 peak reserved 5.04GiB。该阶段是 rollout/CPU 受限，增加 dummy 或无科学作用的 batch 只会浪费时间，因此保留一 GPU 一 policy rank、以有效 interactions/秒为准。
- formal 完成 12 个 full cycles：108 declared updates、107 个有成功信号的 optimizer updates、每 task 48 rollouts、总 3,360 source interactions、679 successes、1,176,874 env steps；max-rank wall 4,862.10 秒。864 个 ledgers、3,360 个唯一 seed rows 和 36/72/108 三个 10-file checkpoints 全部通过 cursor/hash 审计。
- 相同 500 条 validation rows 上，cold step1050、update36、update72、update108 依次为 `63,56,36,15` successes；相对 frozen base 的 paired net 依次为 `+7,0,-20,-41`。逐任务原始数已封存在 `configs/writer_only_rl_selected_v1.json`，预封存排序明确选择 cold step1050。
- 因而本轮 Writer-only RL 是真实完成但未带来 held 泛化收益的负结果；source binary-success self-imitation 随交互增加破坏了 cold Writer 的窄 validation 效用。它不被解释成工程失败，也不通过改算法、加 RNG 或重选 checkpoint 来追求正结果；Phase E 使用未经过 Writer-RL 的 cold step1050。
- 首次 validation launcher 因漏传 sealed `--writer-rl-config` 被 canonical evaluator 在 rollout 前拒绝；失败 packet 保留。重试只补齐该 authority 参数，未改 evaluator、checkpoint、rows 或选择规则，三个候选均 exit 0。

### Task-local RL：Writer initialization 赢得 matched fresh evaluation，但覆盖集中

- 4 tasks × 2 arms 恰好映射到 8 卡，每卡一个 CUDA policy process；update1 后由全新进程 exact-resume 到 update3。共 24 ledgers、96 trajectories、16 checkpoints，所有 `fixed_init_state_id=null`，两臂的 task/env/policy seed block 逐项一致。
- 24 个 `task_local_reward_update.step_seconds` 的线性插值 p90 为 `49.8926s`。按读取 reward outcome 前已封存的纯吞吐规则选择 formal `U=18`、每 task/arm `K=72`、checkpoints `6/12/18`；20 单元共 1,440 interactions，含 180 秒开销的投影总 wall 为 2,874.20 秒。
- formal 实际 wall 1,982 秒；360 ledgers、1,440 trajectories、720 个唯一 matched seed rows、60 个 checkpoint manifests/files 全部通过审计。所有 adaptation 与 selection rollouts 均 official random reset、`fixed_init_state_id=null`；fixed 50 states 只在训练结束后的 fresh evaluator 使用。
- adaptation reward identity/Writer 为 `89/720`、`110/720`，AUC 为 `0.1236/0.1528`。成功主要来自 task0/task28/task88，六个任务两臂都没有 reward signal；这已提示 coverage 有限。
- 同一 500 条 fresh validation rows 上，base/cold/identity-RL/Writer-RL/direct 为 `56/63/54/74/186`。identity-RL 对 base 为 `11 gains / 13 losses / net -2`；Writer-RL 对自身 cold J0 为 `37/26/+11`；Writer-RL 对 matched identity-RL 为 `43/23/+20`，exact paired-binomial two-sided `p=0.0187`。
- Writer-RL per-task 为 `{0:30,8:0,15:0,28:10,40:0,56:1,61:0,71:0,85:0,88:33}`；相对 identity-RL 的 raw delta 为 task0 `+3`、task88 `+20`、task28 `-3`，其余为零。相对 cold 则 task0 `+11`、task88 `+14`、task28 `-14`。因此可支持 Writer initialization 在相同 K72 ordinary RL 下带来真实终点优势，但优势主要由 task88 驱动，不能宣称 reward adaptation 已广泛覆盖未见任务。
- 不追加第二 policy RNG：matched +20 的方向同时出现在 reward AUC 与 fresh evaluation，足以支持上述有限结论；第二 RNG 可能改变小的 task0/28 波动，却不会把它升级为广泛覆盖。完整原始 counts、hashes、J0/JK、curve/AUC/time-to-threshold 与 selection 封存在 `configs/task_local_lora_rl_validation_v1.json`，test 未打开。

### Gate -1：通过但带残差

- 初始 action-hidden-video probe 未达到预声明标准。
- 有界 temporal representation recovery 得到 ordered balanced accuracy `19/24 = 0.7917`，same-scene wrong-video 同为 `19/24`，bidirectional paired both-correct `15/24`。
- ordered 明显优于 static/reversed/shuffled controls，说明视频中存在有用时序任务信息。
- 原 `0.80` 内容阈值、paired 不足和 drop-last sensitivity 未被改写。
- owner 接受它作为当前阶段“通过但带残差”，不再烧算力凑 0.80。

完整历史报告在父提交 `999df28:docs/benchmark_validity_report.md`。

### Gate 0：通过但覆盖有限

历史正式 n=32 h16 packet 中：

- task 3：base `22/32`，action-supervised LoRA `28/32`；
- task 4：base `16/32`，action-supervised LoRA `20/32`。

两个任务点估计均为正，分别 +18.75pp 和 +12.5pp，但任务近似、每臂只有 32 episodes，区间较宽。最新 owner 定义下，它足以说明一个成熟 task-local LoRA 空间可以获得有用行为更新；它不再要求 LoRA 在一个已经 source-trained 的 base 上继续跨过人为门槛。

Gate 0 不证明：

- Writer 有效；
- 跨类别普适性；
- ordinary task-local RL 有效；
- 旧 source base 是新协议应使用的起点。

## 旧 Writer 证据

### Source utility 确实出现过

旧 foundation/full-video Writer 在 16 个旧 source tasks、h50、每 task/arm 32 episodes 上：

| 方法 | 成功 |
| --- | ---: |
| generic foundation base | 0/512 |
| frozen Writer LoRA | 55/512 |
| action-supervised direct LoRA | 51/512 |

Writer/base paired gain 为 +10.74pp，说明当前 full-video hypernetwork 结构能够从 source language/video 学到真实闭环行为；它不是只有离线 loss 的空壳。

### Validation transfer 很弱且集中

旧五类 validation comparison：

| 方法 | 成功 |
| --- | ---: |
| foundation base | 0/160 |
| Writer | 1/160 |
| validation-action-supervised direct LoRA | 18/160 |

额外八任务 frozen Writer 为 `4/256`，四次成功全部在 task 22。task 22 的同口径三臂是 base `0/32`、Writer `4/32`、direct LoRA `12/32`。

这说明旧 Writer 有零星未见任务泛化，但远未达到稳健跨类别泛化。

### 不能简单归因于“只是不泛化”

另一组旧 source-trained-base source localization 在五个旧 source tasks、h16 上得到：

| 方法 | 成功 |
| --- | ---: |
| source-trained base | 141/320 |
| Writer | 127/320 |
| direct source LoRA | 137/320 |

旧 Writer 和 direct LoRA 在这个较强 base/recipe 下都未形成稳定 aggregate gain。因此旧实验混合了：

- base 定义不同；
- source split 不同；
- LoRA teacher recipe 不稳定；
- Writer acquisition/normalization 不充分；
- validation generalization 弱。

当时的 70/10/10 计划要求统一 source embodiment base 和全 50 episodes 重训；该要求已被
当前 π0.5 / 24-8-8 方案替代，但“不能只引用有利旧结果”的证据原则仍保留。

## 评估系统发现

- 早期 GPU0 进程/UTL 偏高的主因是 MuJoCo/EGL workers 未绑定各 rank 的物理 GPU，以及某些 direct fit 与 eval 同时被调度到 GPU0。
- 修复后每 GPU 只有一个 policy CUDA model；resource tracker、forkserver 和 env worker 是 CPU/仿真进程，不应创建额外 CUDA context。
- SmolVLA h50 一次 policy inference 后可执行至多 50 个 simulator actions，因此闭环评估本质上常由 CPU/MuJoCo 限速，GPU UTL 低且脉冲化是正常现象。
- persistent env pools、NUMA/EGL binding、只渲染少数视频分别带来小幅但真实提速。下一实现应保留这些原则，不恢复旧 runner。
- 训练阶段可以通过真实 batch 把显存提高到约 70GB/card；评估阶段不应为了“看起来满”分配无用显存。

## Source base 为什么曾经表现差

通用 `smolvla_base` 是预训练 VLA，但没有保证掌握 LIBERO-90 的具体 Panda embodiment、camera/controller、scene/object 和 action normalization。预训练提供视觉语言和部分动作先验，不等于对每个 LIBERO task 有高成功率。

旧对话中“foundation base”与“source-trained base”曾混用：

- foundation base 未在当时 source tasks 上 action-train，很多任务接近 0；
- source-trained base 已在旧 source tasks 上全 action expert/projection 训练，可在相似 source tasks 上很强。

当前方案通过明确的“过滤后 LIBERO-90 × 每 task 50 episodes”π0.5 source base 消除这个
混淆；最终 task 数由新的 specification-only overlap audit 决定。

## Writer architecture 已成立的机械事实

- `VariableEpisodeTaskEncoder` 接受任意有限正数、任意正长度 episode；帧不被固定为三帧。
- full video 通过 chunk temporal attention、episode memory 和 task-level set attention 聚合。
- 语言用完整 token embeddings；视频保留 episode boundaries。
- `CompleteLoRAWriter` 使用 module/factor/rank-aware queries 和 width-typed heads 输出每个 LoRA A/B tensor。
- 当前活动`CompleteLoRAWriter`在temporal encoder之前额外要求恰好一个非空episode，故只接受`offsets=[0,L]`；底层多episode能力不构成活动输入权限。
- 历史SmolVLA/VLM feature cache只证明冻结features可缓存；当前π0.5必须重建pure-language、按单demo切片且与action query独立的新cache，旧cache不可复用。

历史训练 checkpoint 不可续用，因为新协议将改变 split、source base、全 50 episode 输入、normalization 和数据 authority。

## π0.5统一LoRA与Writer接口事实（2026-07-21）

- 真实`pi05_base` safetensors metadata与meta-device模型结构独立确认同一拓扑：18层action expert的`q_proj`为1024→2048、`v_proj`为1024→256，另含`action_in_proj` 32→1024和`action_out_proj` 1024→32。
- rank/alpha 16、dropout 0时共38 targets、76个A/B tensors和1,287,168 trainable parameters；合同文件SHA256为`1dcf58f7...cb07`，canonical payload SHA256为`42d5919e...94dd7`。
- PEFT注入后的36个expert adapter tensors为BF16，而action in/out adapter为FP32。Writer必须逐tensor保留template dtype；全部强转FP32会让functional训练与materialized推理走不同数值路径。
- mixed-dtype toy policy已逐值验证Writer-generated functional state与copy到物理adapter后的loss完全一致；B=0 identity仍为确定性物理恒等。
- LeRobot `PI05Policy.forward`只接受`batch,reduction`，不接受旧Smol functional路径的`noise/time`参数。训练随机flow noise/time由PI05内部RNG产生，exact-resume保存并恢复每rank RNG。
- Writer language不得复用带当前normalized state的PI05 action prompt，否则会泄漏proprio；新feature owner必须单独做pure task-language tokenization。policy functional action loss仍使用完整PI05 observation/action processor，且缺失right-wrist key以保持false mask。
- `lora.py`只拥有跨backbone PEFT注入、identity、state validation/hash与functional-call机械接口；`pi05_lora.py`是唯一活动科学拓扑authority。旧Smol contract/import只作历史模块provenance，不得进入π0.5 runner。

## Target40数据墙与π0.5 AS-Writer机械合同（2026-07-21）

- immutable Hub revision `f13aa24...e35a` 的四个标准suites共40个HDF5已本地齐备：2,000 episodes、338,575 frames、33,784,856,577 bytes。`configs/pi05_target_data_v1/manifest.json` SHA256为`1b28547f...049d`，40个本地文件均与Hub LFS SHA一致，HDF5 identity aggregate为`6342f5d9...78a6`。
- target封存只读取task specification、Hub metadata、HDF5 schema/shape metadata与opaque file bytes用于SHA；没有解码trajectory/video值，action/state/reward/terminal/video value reads均为0。manifest中的24/8/8 global IDs逐项等于既有protocol，policy outcome reads与task-selection changes均为0。
- development feature cache只授权24 train + 8 validation的action-hidden agentview视频；pure-language prompt固定为`Task: {cleaned}\n`，target40实测最长23 tokens，小于sealed max64。PI05投影后每帧256×2048 tokens只做spatial mean，不沿用SmolVLA的`sqrt(dim)`缩放；缓存BF16并保留50条episode边界。
- AS action dataset从同一32-task manifest显式筛到恰好24 train tasks；validation/test actions永不进入dataset。policy functional loss仍使用冻结LIBERO-90 source normalization，Writer路由所需task/demo identity不会作为tensor输入Writer。
- action query与teacher video分别由不同seed的deterministic no-replacement schedule产生；每个rank/step只取同task的一条video并传`offsets=[0,L]`。checkpoint保存两套schedule identity、全部rank RNG、optimizer/scheduler、metrics cursor，并先验证canonical manifest及每文件SHA再读取pickle。
- feature cache formal配置SHA256为`3e3a8ea7...429e`、AS-Writer配置SHA256为`971cac43...f807`；两者分别保持`pending_source_base`/`pending_profile`。这只是机械authority，不是训练或性能结果。
- 当前架构owner为：`pi05_target_data.py`负责held-data seal；`feature_cache.py`负责PI05 cache schema/tensor store；`cache_pi05_writer_features.py`负责唯一8-rank extraction；`as_contract.py`负责24-task action wall与source/cache/hash联锁；`training.py`只负责AS模型与step loop；`checkpoint.py`负责atomic exact-resume。旧`cache_writer_features.py`与`train_writer_cold_start.py`已删除，剩余历史Smol推理/训练入口因新schema fail-close，待对应PI05 owner具备功能对等后删除。

## canonical evaluator中的one-video Writer证据（2026-07-21）

- `writer/inference.py`已原位替换旧Smol/cold-start实现，只接受`ember_pi05_as_writer_launch_v1`、PI05 38-target LoRA、同一final raw source policy及与训练逐字段相同的formal feature cache；checkpoint先核验canonical manifest和全部文件SHA，legacy schema不再有活动分支。
- 每个rollout的视频selection seed为`sha256([namespace, seed, eval suite, eval task, init state])`前63 bits，demo为`seed mod 50`。该纯函数不依赖worker、shard、重试、queue顺序或outcome；correct/wrong不把arm或video task写入seed，因此使用完全相同的demo ordinal。
- wrong map在每个split role内按Spatial→Object→Goal→Long→Spatial循环，并按该role中排序后的task ordinal一一映射；它是跨suite双射且role-preserving。run contract保存显式map及SHA，避免final_source的train/validation混排导致越墙。
- materialized backend在episode开始只运行一次Writer并固定完整LoRA；由于并行env可能使用不同LoRA，每个replan前为对应slot重装其state并逐slot推理。policy noise仍由原有`(task,init,replan)`schedule生成，correct/wrong不变。
- raw row保存Writer checkpoint step/manifest、Writer state、LoRA contract、逐rollout LoRA、video selection seed/suite/task/role/demo、map SHA、teacher-video pairing SHA和generation timing；run contract另保存排除condition/map但覆盖source、tasks/states、env/policy RNG、topology、Writer/cache的`paired_control_sha256`。聚合重新计算schedule/map并报告每task唯一视频数与频数。当前只是机械合同，尚无Writer行为结果。

## π0.5 Source-SFT机械合同（2026-07-21）

- development authority精确选择target manifest中的24个train global task IDs，四suite各6个、共1,200条可用action episodes；validation/test actions与teacher video均不进入训练。当前配置`configs/pi05_source_sft_development_v1.json` SHA256为`32e927c...8a641`，formal仍为`pending_profile`，不能被误当成正式配方。
- 每次fresh stage都从同一final formal raw source policy注入确定性B=0 identity的PI05 38-target LoRA；只有76个LoRA tensors、1,287,168 parameters可训练。八rank各取task-pure batch后由DDP聚合为一套shared multi-task LoRA，不使用functional Writer、per-task adapter或额外shared adapter。
- checkpoint只保存shared `lora.safetensors`，不复制约14.5GB source policy；同时封存optimizer/scheduler、scaler-disabled声明、optimizer/micro cursor、metrics cursor、每rank Python/NumPy/CPU/CUDA RNG、DataLoader seed、deterministic sampler identity与逐task episode coverage。所有文件在读取pickle前先验证bytes/SHA，正式末checkpoint必须覆盖每task全部50 episodes。
- development config只能启动development；validation选择后必须另封final config，32-source formal run从同一确定性identity fresh开始，development checkpoint因stage/config/contract hash不同而无法resume。这避免后续修改配置破坏development provenance。
- evaluator仍只有`evaluate_pi05.py`一条canonical runner。Source-SFT LoRA在每个worker初始化时只安装一次，随后保持原有multi-env batched replan；它不会走Writer的逐rollout生成/重装路径。raw row保存固定`policy_adapter_sha256`，resume重新核验source/run/checkpoint/config/LoRA全部hash；Writer专属`paired_control_sha256`不套用于Source-SFT。
- 新代码按当前故障边界归入`ember.source_sft`子包（contract/data wall、training、checkpoint、inference），`eval_adapters.py`只拥有两类adapter的runtime分派和row证据；这是独立baseline所需owner，不是第二套evaluator。architecture guard无hard violation、无parallel family。旧Smol `direct_lora*`仍只作provenance；待本PI05 owner完成真实8卡finite/exact-resume smoke后，迁移其仍被旧task-local模块使用的通用helper并删除旧direct CLI/module，而非长期保留双活动路径。

## 数据与 benchmark 事实

- LIBERO-90 正好提供 90 个大规模 task 数据文件，每 task 50 条成功 teacher demonstrations。
- LIBERO-10/Spatial/Object/Goal/Long 是另外的标准 benchmark suites；不应和 LIBERO-90 task IDs 当作同分布池混划。
- 旧 60/15/15 specification-only parser 证明 role-aware factorization 可行；其后的
  70/10/10 也已退役。当前目标 split 是已封存的四 suites 24/8/8，并在 final 合并为
  32 source / 8 test。

## 不允许从历史证据推出的结论

- 不能声称 EMBER 已有稳健 validation 泛化。
- 不能声称 Writer-only RL 改善了 validation；本轮已验证的是其完整负结果。也不能把 task-local RL 的 task88 集中增益说成广泛覆盖，不能声称 identity-init ordinary RL 改善 aggregate，或声称 outer learning 已验证。
- 不能声称 direct LoRA 是信息匹配的 held baseline。
- 不能声称 h16 是标准 SmolVLA/LIBERO 主评估。
- 不能把旧 source task 结果、旧 validation 结果或 task 22 用于新 split。
- 不能重新引入 bank/geometry 作为“修复”。

## 历史 SmolVLA RL 实现边界（不约束当前 π0.5 算法）

- Writer-only RL 与 task-local RL 都采用 binary-success on-policy
  success-weighted flow regression；这是为 SmolVLA 没有可直接使用的 exact action
  likelihood 而选的 ordinary reward adaptation，不含 critic、teacher action、外部
  exploration adapter 或伪 PPO。
- task-local 两臂只允许同一 37-target LoRA 可训练；base、Writer、encoders 和所有
  shared state 冻结。其配对 seed schedule 明确排除 arm，官方随机 reset rollout
  ledger 与 worker RNG/interaction cursor 进入 checkpoint；`.pruned_init` 只由独立
  fresh evaluator 使用。
- Writer-only RL 已有完整 source formal 与 4 候选 validation 的真实负结果；task-local RL 已完成 matched formal、预算内选择和独立 fresh validation，支持 Writer-init 相对 identity-init 的有限终点优势。

## 证据定位

外部结果根不写入公开仓库；使用本地 `EMBER_OUTPUT_ROOT` 查找。关键历史目录名：

- `gate_minus1/specification/video_information_recovery1_*`
- `gate_zero/task_local_lora_rl_formal_development/formal_n32_recovery3_*`
- `foundation_source_screen/source_three_arm_eval_v2_*`
- `foundation_source_screen/validation_three_arm_*`
- `foundation_source_screen/additional_val_writer_probe_*`
- `foundation_source_screen/task22_base_direct_*`

精确旧命令/config/code 由 Git commit `999df28` 保存。不要把这些目录复制回仓库。

## π0.5 reward adaptation机械事实（2026-07-21）

- π0.5每次规划产生50步chunk，但活动LIBERO runner只执行前5步后重规划。reward replay不能把未执行的后45步计入成功credit；新`Pi05ExecutedPrefixFlowLoss`直接保留逐action-step flow loss并按每个replan实际执行长度mask，successful episodes先各自平均再等权。仅传旧`action_is_pad`字段不足以建立这一合同。
- reward rollout唯一owner使用raw `OffScreenRenderEnv`的`env.seed(seed) -> reset()`，随后10个dummy settling steps；不调用`set_init_state`，不读取`.pruned_init`。环境seed和逐replan flow-noise seed只依赖task/adaptation/rollout cursor，显式排除arm、rank、worker、queue order与outcome。
- RL-Writer从同seed的fresh随机Writer开始，zero输出头使初始generated LoRA在功能上为identity；source policy物理冻结，只有共享Writer经functional LoRA loss更新。全局没有成功episode时optimizer/scheduler cursor不前进；有成功时8 ranks按全局successful-episode数做等权DDP缩放。
- micro-AS分支当前明确fail-close：只有完整zero-warmup run封存为无信号后才允许从同seed fresh Writer做恰好24个action chunks（每development train task一个）的短warm-up；永不载入完整AS-Writer checkpoint。当前没有运行reward实验，也没有正/负科学结果。
- task-local合同严格绑定8个test global IDs `6,8,10,17,24,27,30,33`。同一`(task, adaptation seed)`的AS/RL Writer臂共享一条hash选定video，初始化LoRA只生成一次并固定；identity臂不读取video。rollout、environment-action、optimizer三类cursor和初始化/ledger hashes均进入checkpoint，fixed-50不能选adaptation checkpoint。
- 活动代码所有权：`ember.reward`只拥有共享random-reset/seed/ledger/executed-prefix mechanics；`ember.rl_writer`拥有fresh shared Writer的authority/runtime/update/checkpoint；`ember.task_local`当前拥有test-only unit/initialization/update/checkpoint mechanics。三个旧Smol可执行入口已删除，避免双canonical runner；剩余历史模块/配置只作provenance，等π0.5 task-local runtime与fresh evaluator功能对等后删除，不加兼容分支。
- seen/source panel已在任何新source/Writer/Source-SFT outcome产生前按specification-only规则封存：每suite只在6个development-train tasks内按`SHA256(tag, seed, suite, task_id, language, BDDL SHA)`排序取前2个，得到global IDs `0,2,15,12,21,28,39,37`。该panel只用于source acquisition诊断，不能替代validation/test泛化。
- frozen RL-Writer不需要第二套evaluator：它与AS-Writer共用task/video mapping、逐rollout materialization、每次replan重装同一LoRA及row validation；adapter明确记录`writer_method`与`reward_update` checkpoint axis。RL checkpoint inspector会重算24-task task/video full-cycle coverage并核验source/cache/config/hash，formal状态未seal时不能进入非smoke评测。

## Fresh 1k source base与evaluator吞吐证据（2026-07-22）

- owner把source acquisition明确限定为从generic base fresh训练1,000 optimizer steps；因此旧30k attempt在step2880无checkpoint停止，不能resume或参与比较。新run在8×A100、global batch256、333-step warmup、EMA下完成256,000 examples，训练loop用时91.58分钟。
- 50-step mean loss由0.26213降到0.08659，后半程下降幅度逐渐缩小但并未数学收敛。科学解释只能是“1k预算下获得轻量LIBERO interface acquisition且末段趋平”，不能写成LIBERO-90已过拟合或已完全收敛；到预算仍缓降按全局规则记为budget-censored。
- step1000 checkpoint的完整manifest/hash验证通过，包含raw policy、EMA和1000条唯一finite metrics；冻结后所有下游方法必须共享同一raw source policy与source-only normalization，不能追加source adapter或因target outcome回改source IDs。
- all-40-task×1-state公平panel显示1/2/3 replicas/GPU的有效吞吐为0.1556/0.1818/0.1897 rollout/s。3 replicas比1 replica高约22.0%、比2 replicas高约4.35%，每卡约31GB且GPU0无额外CUDA角色，因此40-task source screen锁定3 replicas/GPU。
- 上述3个40-episode profile均为吞吐smoke且全部0 success，分母太小，不能用来断言source competence；正式行为判断只来自随后预定的40 tasks×8 states screen及其逐taskraw rows。
- 原正式40-task×8-state screen的0/320只验证了滞后EMA，不能解释为source acquisition负结果。参数比较显示EMA只走完raw更新位移的28.62%（action expert 33.52%，action I/O/time/state 40.13%，VLM 26.87%）；匹配source closed-loop为raw 4/4、EMA 0/4，匹配offline flow loss为raw 0.06165、EMA 0.17775、generic 0.29302。
- raw step1000的正式40-task×8-state screen为`46/320 = 14.375%`，成功覆盖13 tasks和全部四个suites：Long 2/80、Goal 28/80、Object 1/80、Spatial 15/80。它满足“多个tasks有部分真实成功、aggregate不由单task支撑”的source acquisition条件；canonical下游使用raw `policy/`，EMA只保留为训练状态和负诊断。
- 正式raw screen绑定commit `cab2edf72a8b7d5173503735ef33bdd8fc4c2a50`、raw weights SHA256 `60ea7ee8...cdf36`、corrected source summary SHA256 `473ae3dc...f874`。320 rows/task-state唯一、24 workers均exit0；results SHA256为`4e2defaf...db3a`，wall-clock 412.372秒。
- development Writer feature cache在同一raw source policy上完成8卡batch32 smoke：8 tasks各1条video、共1,033 frames全部成功，单task 89.77–113.70 frames/s，按并行critical path为689.47 frames/s；输出仅4.92MB且没有OOM/nonfinite。该证据足以直接锁定batch32，不再做无科学收益的batch sweep。
- 随后的formal cache覆盖development train+validation共32 tasks、每task全部50 videos，总计1,600 episodes/274,523 frames；32个task tensors逐文件SHA全部通过，目录1.127GB，launch-to-manifest 248秒。其information wall明确记录test video与trajectory action/state/reward/terminal读取均为0。

## AS-Writer profile结论（2026-07-22）

- AS-Writer的显存主因是对generated LoRA保留functional policy反传图，而不是Writer本体参数量。batch1只分配约12.9GB；batch16分配63.53GB、reserve 68.17GB，并把稳态吞吐从约32提高到约122 global action queries/s，因此batch16是当前实测有效且保留稳定余量的点。
- 极短训练必须保持预期正式scheduler horizon；把`total_steps`也缩成4会令LeRobot按`4/30000`缩放warmup并取整为0，造成首步直接使用peak LR。这是profile协议伪影，不是Writer科学发散。
- 在1,000-step horizon下执行前128步时，首/末16-step mean functional loss为0.14714/0.11930，后64步线性斜率约`-1.58e-4/step`，gradient norm在warmup后降至约0.2–0.3且无nonfinite。曲线尚未饱和，但实测完整1,000步仅约17.5分钟净训练，故选择1,000步并以四分点做稀疏validation候选，不把120分钟guardrail当目标。

## AS-Writer选择、source-base validation与Source-SFT固定预算（2026-07-22）

- AS-Writer正式训练完成1,000 steps、global batch128；cheap screen选择step250与step500进入完整8-task×50 validation。step250为`119/400`、step500为`99/400`，因此development AS-Writer冻结step250。step250逐任务成功为Spatial 1/3=`0/0`、Object 1/3=`40/36`、Goal 3/6=`0/27`、Long 1/2=`16/0`。
- 同一8-task×50 fixed-state配置下，frozen source base为`48/400`：Spatial 1/3=`0/0`、Object 1/3=`5/0`、Goal 3/6=`0/41`、Long 1/2=`2/0`。AS step250 aggregate增加71，但Goal 6从41降到27，说明增益来自任务重分配而非所有任务一致改善。
- owner选择先做不调step的matched-scale Source-SFT。比较基准是被选中的AS step250所消耗的`250 × 128 = 32,000` action queries，而非完整1000-step曲线的累计消耗。Source-SFT保持profile选定的batch64/rank（global512），固定63 steps=`32,256` queries，与32,000相差0.8%；该匹配不要求batch size或optimizer updates相同，也不声称forward compute、参数量或监督路径相同。
- Source-SFT profile显示batch64/rank稳定finite且峰值allocated/reserved约32.16/42.04GB、约71.88 queries/s；batch128/rank约54.92/67.93GB、约72.05 queries/s，没有有效吞吐增益。因此正式run保留batch64/rank，只保存并验证step63。
- Source-SFT正式run完成63/63 steps、32,256 queries，训练loop wall-clock `450.263s`。首/末8-step mean loss为`0.15475/0.13883`，全程线性斜率约`-3.00e-4/step`；仍有下降迹象，按预先固定的同规模预算记为可能未充分训练，不增加steps。checkpoint的10个exact-resume文件与63条连续finite metrics均通过hash验证。
- 唯一step63完整validation为`61/400 = 15.25%`，5/8 tasks非零；逐任务Spatial 1/3=`5/1`、Object 1/3=`20/0`、Goal 3/6=`0/32`、Long 1/2=`1/2`。相对source base `48/400`增加13，但同时Goal 6从41降到32、Long 1从2降到1；增益主要来自Object 1 `+15`和两个Spatial tasks `+6`，不是全任务一致改善。
- 在400个匹配task/init rows上，Source-SFT与source base/AS都保持相同env seed、policy seed root和共享长度内全部policy-noise seed前缀。AS step250仍明显更高：`119/400`对`61/400`，主要差异为Object 1 `40 vs 20`、Object 3 `36 vs 0`和Long 1 `16 vs 1`；Source-SFT在Goal 6为`32 vs 27`并在两个Spatial tasks合计`6 vs 0`。
- Source-SFT train contract/metrics/checkpoint-manifest SHA256为`4e113268...dd87`/`02ca6611...64bb`/`2dce01f0...fb3e`；validation results SHA256为`92e3e667...3f6d`，24 workers均exit0、38 shards完整、400 raw rows唯一。三方法对照证据文件SHA256为`c376ef9c...a1f`。

## AS-Writer cross-suite wrong-video结果（2026-07-22）

- step250 cross-suite-wrong完整validation为`115/400 = 28.75%`，而correct-video为`119/400 = 29.75%`、source base为`48/400 = 12%`。核心correct−wrong差值仅`+4/400 = +1pp`，因此当前checkpoint虽明显优于base，却没有建立强teacher-video任务内容依赖。
- 逐任务correct/wrong为Spatial 1/3=`0/0` vs `0/1`、Object 1/3=`40/36` vs `37/33`、Goal 3/6=`0/27` vs `0/28`、Long 1/2=`16/0` vs `16/0`。400个paired rows中both-success 102、correct-only 17、wrong-only 13、both-fail 268；不是少数任务的大幅正负效果恰好抵消。
- 两臂400/400 rows的task/init、env seed、policy seed root及共享长度内policy-noise seed前缀完全匹配；noise列表长度仅因成功终止改变replan次数。wrong run 24 workers均exit0、38 shards和400 raw rows完整，results SHA256为`0e6ee518...a9ce`，correct/wrong对照证据SHA256为`d4a4f9f7...eaac`。
- 科学解释应保持克制：结果与Writer主要使用language、或视频编码在该训练下近乎不敏感相一致；在没有进一步干预证据前，不能声称AS-Writer从正确视频恢复了task-specific visual information。

## AS-Writer视频塌缩的根因与Writer-v2决策（2026-07-22）

- `fixed video / changed language`相对有效LoRA差约`4.02e-4`，`fixed language / changed video`约`7.52e-6`；所谓53倍只表示视频残差更弱，两者绝对量都很小，不能作为language conditioning有效的证据。当前v1输出应解释为近乎input-independent shared LoRA。
- 根因首先是目标不可辨识，而非Writer参数不足：functional PI05 policy本身收到正确language和observation；24个train tasks又各有唯一language；teacher video与action episode在同task内独立。因此一套共享domain/control LoRA加policy自身language就能降低loss，目标没有要求Writer使用视频。v1约12.48M参数生成1.287M LoRA参数，增加容量不会消除该捷径。
- 架构进一步放大捷径：每帧256个视觉token被全局平均、整条video只压为4个episode tokens，而parameter-query residual和有bias的共享head可绕过task memory直接产生公共LoRA。1000-step全程warmup可能影响checkpoint退化，但解释不了step250已经视频不敏感。
- owner决定效果优先、消融后补。因此Writer-v2一次性组合修复目标和架构：缓存固定4×4空间网格；language/video分别压到固定memory并使用不含learned-query residual的conditional attention；LoRA decoder只允许parameter query乘性寻址conditional task memory，最终head无bias并保持B=0 identity。
- owner最终口径明确否定零向量language和generic policy prompt。Writer-v2使用经同一PI05 tokenizer/embedding得到的固定中性Writer语言`perform the demonstrated task`（记为`g`）；frozen policy在全部分支始终接收action-query task的正确language A。主循环为`normal → full-language contrast → generic-language contrast`：分别训练`W(l_A,v_A)`、成对比较`W(l_A,v_A/B)`、成对比较`W(g,v_A/B)`。
- 两类contrast都只取半批独立A action queries并复制成correct/wrong两臂，因此每步总policy samples保持与normal相同；两臂从同一CPU/CUDA RNG状态开始，policy observation、language A、query、flow noise/time完全配对。correct臂各自有绝对functional action loss，bounded softplus matching只要求wrong loss高出margin，避免仅靠破坏wrong臂构成训练目标。
- 新cache合同的8卡真实smoke验证了generic不是零向量或运行时占位符：`perform the demonstrated task`经正常PI05纯语言tokenizer得到8个有效tokens，8个task cache中的embedding逐byte完全相同；视频tensor为`frames×16×2048 BF16`。8 tasks/1,033 frames均finite并完成，batch32 critical-path约666.25 frames/s，故只封存cache batch而不从中推断Writer效果。
- 只让Writer做language dropout是不充分的，因为policy language仍可承担任务；仅让wrong分支变差也不构成成功。后续判断必须同时看correct是否超过constant/shared adapter、同query functional loss是否按video-task正确排序、以及多validation tasks上的correct−wrong，而不是只看LoRA hash或相对倍数。
- Writer-v2 formal cache已完成32 development tasks×50 videos（1,600 episodes、274,523 frames），32/32 task tensor hashes、episode/frame counts与information-wall复核通过；generic embedding在全部tasks逐byte一致。该cache约17GB，保留4×4空间token而非旧全局平均。
- 8卡batch16学习profile在500-step scheduler horizon下实际执行30步，三种mode各10步。normal positive首/末3步均值为`0.16161/0.13554`；full-language wrong-minus-correct gap由`-3.44e-6`移至`+9.58e-5`，generic gap由`-1.04e-5`移至`+4.13e-5`。这证明训练方向已开始区分匹配关系，但幅度仍小，不能提前声称视频特异性成立。
- profile中normal/full/generic step中位耗时分别为`1.865/2.127/2.858s`，只作资源记录。owner明确不以约`1.3–1.6×`wall-clock倍率作为设计或启动门槛；四类概念分支、两个correct绝对行为目标、paired RNG与policy恒用正确language的科学合同优先。
- 首轮formal据此封存250 steps、batch16/rank与50-step checkpoints。每个checkpoint先做固定functional matching、generic-correct competence和adapter-specificity诊断；若step250仍明显未充分训练，按统一预算口径封存曲线并标记undertrained，不自动追加。
- Writer-v2首轮在commit `dcfb20689954225aa0cc92ae75f4103a7db6213c`上完成250/250 steps；训练段wall-clock `334.476s`，共产生32,000 policy samples与21,376 global action queries。24 tasks分别覆盖全部50条action episodes与全部50条teacher videos，validation/test action reads均为0；五个checkpoint manifests逐文件验证通过。
- 三种mode的首/末10步positive loss分别为normal `0.14501→0.11478`、full-language `0.14337→0.12463`、generic `0.13617→0.12040`。full/generic的wrong-minus-correct gap从首10步`+9.37e-5/+5.95e-5`增至末10步`+0.00707/+0.00788`，说明matching方向已明显增强且positive competence没有被训练目标主动牺牲；由于各step action query不同，这仍需fixed-query和rollout证据确认。
- canonical evaluator现在可表达`generic_correct`与`generic_cross_suite_wrong`：只有Writer language切换为cache中的中性`perform the demonstrated task` embedding，policy observation/prompt仍沿用evaluation task的正确language。旧`correct/cross_suite_wrong` adapter payload保持不变，已有评测的resume/reaggregation合同不受影响。

## Writer-v2首轮closed-loop结果与充分训练决策（2026-07-22）

- step250固定64-state screen中，full-language correct/wrong=`12/8`，generic correct/wrong=`12/8`；两组paired flips均为correct-only 7、wrong-only 3。generic-correct成功覆盖Goal与Object多个tasks，说明neutral Writer language下视频能生成有绝对competence的adapter，而不是只破坏negative臂。
- full-language完整validation为correct `83/400=20.75%`、cross-suite wrong `63/400=15.75%`，净差`+20/400=+5pp`。paired rows为both-success 43、correct-only 40、wrong-only 20、both-fail 297，exact McNemar `p=0.01349`；5/8 tasks正向、1负向、2持平。共享policy-noise seed前缀逐项一致，列表长度差只来自成功提前终止。
- 逐任务correct/wrong为Long 1/2=`4/2` vs `11/2`、Goal 3/6=`0/31` vs `0/20`、Object 1/3=`33/6` vs `29/0`、Spatial 1/3=`2/5` vs `1/0`。v2已比v1的119/115建立更强视频特异性，但correct由119降至83，当前科学问题从“无视频依赖”转为“特异性与绝对competence的权衡”。
- owner要求分别充分训练Writer-v2与Source-SFT并找各自validation最佳。Writer选择阶段只运行correct-video；只有唯一最强Writer checkpoint选定后才运行一次correct-language + cross-suite-wrong-video完整control，不运行per-checkpoint wrong或generic full arms。RL-Writer与seen继续暂停。
- Writer-v2 ceiling run从identity fresh训练1,500 steps，使用50-step warmup/1,500-step cosine并每250步保存；Source-SFT从identity fresh训练800 steps，使用100-step warmup/800-step cosine并保存100/200/400/600/800。两者独立选择，不匹配steps、queries、compute或参数；到120分钟guardrail仍未饱和则保留曲线并标记budget-censored。

## Development ceiling最终判断（2026-07-22）

- Source-SFT的完整validation在step200/400/600为`74/87/73`，最佳step400已经被前后候选夹住；恢复到计划上限800不会服务于“找最强已观测checkpoint”的当前判断，因此冻结step400且development不再重训。
- Writer-v2原run的step500/750/1000/1500 correct为`99/92/75/72`。独立dense-retention replay的350/400/450/500/550/600/650/700/750 cheap screens为`24/27/20/24/26/31/19/30/33`（各128），只将600/700/750提升为完整validation，得到`90/85/95`。它们均未超过原step500，后段亦持续退化，故没有证据支持再补800–950或更多细粒度训练。
- 选定的原run step500逐task correct为Long `5/0`、Goal `1/38`、Object `37/12`、Spatial `2/4`，合计`99/400`。其唯一cross-suite wrong-video arm为Long `6/2`、Goal `0/27`、Object `20/0`、Spatial `0/0`，合计`55/400`；correct-only/wrong-only=`56/12`，exact McNemar `p=6.21e-8`。
- v1的`119/115`说明高绝对correct主要来自几乎input-independent的公共adapter捷径；v2的condition-only架构与paired contrast使correct下降20、wrong下降60。v2仍有提升绝对competence的空间，但当前`99 > Source-SFT 87 > source base 48`且视频差`+44`跨6个tasks，已同时满足行为收益和视频特异性，当前最有价值的下一证据是RL-Writer而非继续AS消融。

## RL-Writer初始reward可学习性（2026-07-22）

- fresh identity Writer生成的初始LoRA功能上等于source base；首个完整24-task official random-reset cycle取得`7/24=29.17%`成功，成功来自7个不同tasks并覆盖Spatial、Goal、Long。因而zero-AS分支并不缺初始binary reward，当前没有科学理由消耗24条teacher-action warm-up。
- 成功episode需要26–57左右replan chunks时，整轨迹functional反传既逼近80GB，又因成功/失败ranks走不同DDP图而互等。8-chunk proxy-state微批把同一个episode mean loss精确拆分：每chunk权重为`1/(global_successes × episode_chunks)`，生成LoRA梯度汇总后只回传Writer一次；固定顺序all-reduce等价于对全局成功episodes取均值。
- 修复后的三次global updates均完成，successes为`4/1/2`，global gradient norms为`0.0535/0.0729/0.1341`且全部finite，峰值reserved仅40.84GB。这排除了“有reward但更新机械不可执行”的工程解释；下一科学问题是多cycle reward coverage与held validation是否随Writer训练改善。

## RL-Writer development选择与视频对照（2026-07-23）

- canonical zero-AS run从fresh identity出发，经update3→24→36→54三次真实8-rank exact-resume完成；累计`432`条official random-reset source rollouts、`81` successes、`131,354` environment actions，optimizer cursor为44，teacher-action consumption、fixed-pruned-init reads和validation/test reward reads均为0。净训练wall为`2261.716s`，远低于120分钟guardrail。
- 18个完整24-task cycles的后六轮successes为`5/2/4/3/5/3`，没有持续上升。固定同一64-state validation screen在update12/24/36/54依次为`6/11/15/14`；source base同subset为`7/64`，因此选择最早峰值update36并停止到54，不继续72/96/120。
- update36完整correct-video validation为`94/400=23.5%`，逐task为Long 1/2=`1/3`、Goal 3/6=`0/47`、Object 1/3=`40/0`、Spatial 1/3=`3/0`，成功覆盖5/8 tasks。development绝对性能排序为AS-Writer `99`、RL-Writer `94`、Source-SFT `87`、source base `48`。
- 同一checkpoint唯一cross-suite wrong-video arm为`87/400=21.75%`，逐task为Long `0/1`、Goal `0/44`、Object `40/0`、Spatial `2/0`。400对rows的correct-only/wrong-only/both/neither为`10/3/84/303`，exact McNemar `p=0.092285`；方向为正但不足以宣称强视频特异性。
- 科学结论因此分开表述：reward-only Writer确实学到可泛化且优于source base/Source-SFT的held competence，但其增益主要仍可由language/common adapter解释，视频因果控制较弱。已有明确source reward，故不以结果不够正为由启用micro-AS或改协议；后续RL-init task-local arm可以保留，但不得写成已证明强依赖teacher video。

## Sealed seen-panel比较（2026-07-23）

- 在任何这些outcome产生前封存的8-task panel（global IDs `0,2,15,12,21,28,39,37`）上，四方法均完成每task 50个相同official fixed states：source base `137/400`、Source-SFT step400 `182/400`、AS-Writer-v2 step500 correct-video `204/400`、RL-Writer update36 correct-video `164/400`。因此source acquisition在seen任务上成立，绝对排序为AS > SFT > RL > base；seen诊断不替代held validation。
- 按`Long 7/9, Goal 1/8, Object 2/5, Spatial 0/2`顺序，逐task successes为base `2/0,50/34,0/0,22/29`；SFT `0/0,48/41,10/0,41/42`；AS `1/0,49/47,23/1,41/42`；RL `1/0,41/45,1/0,35/41`。AS相对SFT的主要额外收益来自Object-2，而Long-9对四方法均为0，不能把aggregate写成全suite普遍提升。
- 四份results SHA256依次为base `91a9a31f...fb833`、SFT `05c4c0d1...d889b`、AS `3d640e57...d97479`、RL `92a958a3...3f2c8`；对应evaluation wall分别为`504.007/510.400/736.591/815.813s`，每份均保留400条raw rows和逐task aggregation。
- Source-SFT训练因owner在development选择step400后于step600手动停止，原runner未走到terminal summary发布；这不影响已原子发布的step400 checkpoint，但当前fail-closed evaluator要求训练summary。修复只从不可变run contract、600条连续metrics和step600 manifest重建`run_summary.json`，没有GPU forward、optimizer update或权重改写；summary/recovery provenance SHA256为`887ae816...ab2e`/`c7f29ae7...803c`。第一次seen启动在任何rollout前因此失败并保留，成功结果来自新root，未把失败目录续作正式证据。
- 该比较已回答Phase E的source acquisition问题；不再补seen wrong-video、额外checkpoint或generic arms。下一步使用development已选普通配置从规定fresh初态进行final 32-source重训。

## Final 32-source训练合同（2026-07-23）

- final角色只把封存的8 validation global IDs机械并入24 train IDs，四suite各8 source tasks，8 test IDs不变。AS与Source-SFT读取32 tasks各50条source action episodes；RL-Writer只读取同32 tasks的official random-reset reward和action-hidden videos；三者test action/reward/video reads在训练合同中均为0。
- AS复现development最优step500时必须保留原1,500-step cosine horizon；若把formal total直接改成500，LeRobot会把50-step warmup自动缩到16并把decay压到500，已不再是同一配置。因此final合同以`total_steps=1500`封存scheduler，机械`selected_stop_step=500`，只实际训练到并发布step500。
- Source-SFT同理保留800-step horizon与100-step warmup，机械停在development选定的step400；final不是把scheduler重缩到400。RL-Writer development update36在24 tasks上等于每task 12个完整cycles，32-task final据此固定为48 updates=`384` rollouts、每task仍12次，不用test outcome重新选择预算。
- 三份final配置SHA256为AS `ebe269ea...e299e`、Source-SFT `25e99628...d10c2`、RL `32dd979b...2ab30`。AS/RL扩展同一canonical runner的sealed stage和source roles，没有增加第二套入口；现有32-task feature cache完整覆盖train+validation，可直接复用而不生成重复17GB cache。

## Final AS-Writer完成（2026-07-23）

- final AS从同seed fresh identity在32 source tasks上实际完成500/500 selected steps，保持原1,500-step scheduler horizon；训练loop wall为`634.671s`。checkpoint coverage证明32 tasks均使用全部50条action episodes和全部50条teacher videos，累计64,000 policy samples、42,688 unique action queries，test action/video reads均为0。
- normal/full/generic positive loss首20到末20均值分别为`0.13805→0.12183`、`0.14542→0.11595`、`0.13450→0.11863`；末20步full/generic wrong-minus-correct gap为`0.00729/0.00783`。所有500 metrics连续唯一且finite，峰值reserved `68,344,086,528` bytes。
- final step500 checkpoint manifest payload SHA256为`b30b2e1d...c395`；run-contract/metrics/corrected-summary SHA256为`36207182...2de`/`0d208b15...b619`/`a4f76fb2...9de7`。
- 初始summary错误继承了development字段`validation_action_reads=0`，与final source角色冲突；训练contract、checkpoint和coverage均正确。已仅修正summary为400个validation-source action/video episodes available并保存零权重改动的correction provenance，SHA256 `ebc1bed8...414e`；代码同步按stage生成正确字段。

## Final Source-SFT完成（2026-07-23）

- final Source-SFT从fresh identity在32 source tasks上完成development已选的400/400 steps，保留原800-step cosine horizon与100-step warmup；训练loop wall为`2852.793s`，累计204,800 action queries，400条metrics连续且全部finite。
- step400 coverage证明32 tasks均覆盖全部50条action episodes，每task 6,400 examples，共138,952 unique query rows；test action/video reads均为0。首/末20-step平均loss为`0.15139→0.11531`，稳态吞吐约`71.88 queries/s`，峰值reserved `42,037,411,840` bytes。
- step400 checkpoint manifest payload SHA256为`0012ffb6...52bd`，run-contract/metrics/summary/file-manifest SHA256为`bc136964...da31`/`c0d91c9b...6211`/`ff0a33f7...d472`/`cd2f0766...d034`；10个exact-resume文件全部通过size/hash校验。

## Action-Memory Writer设计与资源结论（2026-07-23）

- 冻结PaliGemma的图文prefix可以预计算；但按当前stride-4开发集估算，保存pre-transformer图文prefix约50–60GiB，保存18层KV约250GiB，后者不符合当前收益/存储比。由于直接训练预计低于一小时，本轮不让cache工程阻塞科学结果；若后续确认encoder是主瓶颈，优先缓存language-independent image embeddings而非完整KV。
- Action-Memory Writer将语言理解留给冻结π0.5的PaliGemma，并让16个memory tokens从Action Expert流读取每帧图文prefix；初始化使用16个确定性正交32D action codes经冻结`action_in_proj`投到1024D后detach。Meta-LoRA仅增强teacher encoder对该输入的读取，不成为共享execution adapter。
- 最终profile证明per-rank batch16可执行，10.10M训练参数与rank128 Source-SFT 10.30M基本等量，因此后续AS对SFT的比较不再有约10倍训练参数容量差这一明显混杂。当前尚无closed-loop性能结论；必须由多checkpoint validation和唯一best correct/wrong arm决定。

## bias-free初测、bias恢复与新上限判据（2026-07-23）

- bias-free Action-Memory run的完整correct-video validation在step300/500为`105/400`与`89/400`；step300逐task为Long `15/3`、Goal `1/35`、Object `25/26`、Spatial `0/0`。step500为Long `5/0`、Goal `0/39`、Object `22/23`、Spatial `0/0`。paired step300-only/step500-only=`41/25`，exact binomial约`p=0.064`：300更好，但两个点不足以证明真实峰值或架构上限。
- 同参数口径rank128 Source-SFT先前完整validation曲线为step100/200/300/400/600=`90/105/65/122/111`。step300后又在400大幅恢复，证明一个下降点不能作为饱和证据；旧`122/400`是当前最佳观测值，不是已充分探索的上限。
- `condition-only`只要求完整LoRA经language/video条件路径产生、没有独立公共LoRA输出支路；它不要求所有内部线性层无bias。此前全局`bias=False`会降低约束网络的平移自由度并增加优化难度，属于额外实现限制。owner据此选择保持拓扑不变而恢复conditional path普通bias。
- 恢复bias不会自动创建显式共享adapter：temporal/layer/slot block和factor head仍只处理条件hidden states；factor-head最终bias从零初始化，与最终weight一起保证fresh task LoRA为identity。它仍可能通过共享参数学出近公共输出，这必须由correct/wrong视频行为和生成LoRA差异实证判断，而不是靠`bias=False`宣称排除。
- 新validation functional-loss panel固定512个task-balanced、video/action不配对query，可低成本观察loss斜率、train–val gap和候选checkpoint；由于teacher-forced action loss与closed-loop恢复能力可能错位，它不能单独选best。真实峰值要求完整8-task×50 success曲线，且“峰后持续下降”不能由单一相邻checkpoint判断。
- bias恢复只增加`21,696`个训练参数：Writer从`10,097,601`变为`10,119,297`，仍仅为rank128 Source-SFT容量的`98.27%`。四卡真实profile的显存与旧bias-free八卡profile几乎相同（reserved均约78.87GB），说明恢复bias没有引入隐藏模型副本或新执行支路。
- 每rank batch16/global64的第二个稳态profile step为`1.930s`，对应约`33.16` global queries/s；四卡相对旧八卡global128约`1.93–2.48s/step`的单步时延相近、总吞吐约减半，符合相同单卡工作量和world-size缩减预期。后续step数按实际action-query量解释，不能把四卡step与旧八卡step直接等同。
- rank128 Source-SFT从8 ranks切到4 ranks时无法宣称逐rank RNG与sampler完全相同的bitwise exact-resume，但这不意味着必须从零重训。合理做法是优先延续已有权重与optimizer、封存拓扑切换和重分片后的cursor，并将其标记为`topology-transition continuation`；只有完全相同的world size与合同才称exact-resume。batch size本身不是科学门槛，跨轨迹比较需同时给出optimizer updates、累计action queries和独立task-condition visits。
- 为充分检验`122/400`是否真实上限，新SFT最大horizon可延到2400，但不把cosine decay从800拉长：这样前800步不因扩大上限而获得更高LR，800后只是固定低LR tail。若完整validation已在800前后显示多点持续退化，便无需机械跑满；若仍波动或回升，则同一合同可恢复到1600/2400。
- bias-restored AS的封存512-row validation functional loss在step100–800依次为`0.135237/0.138363/0.134698/0.141123/0.134224/0.138690/0.139285/0.140583`。step400的单点回升随后在500完全恢复，验证了“单点不能早停”；而500后连续三个checkpoint回升、同时train loss继续下降，已经把closed-loop候选区间收缩到step500附近，但不能单独证明closed-loop最优。
- 独立backfill与resident monitor在step300/400/500的1,536条逐query loss完全一致，排除了训练进程内切换eval数据导致数值漂移的实现疑点。validation过程无gradient、无optimizer update，结束后恢复完整RNG与Writer train mode。

## bias-restored AS首轨迹结果与四卡scheduler混淆（2026-07-23）

- decay-6400首轨迹的完整correct-video validation为step300/500/800=`62/77/80`（各400）。逐任务分别为Long `0/0`、Goal `0/36`、Object `16/8`、Spatial `0/2`；Long `2/0`、Goal `0/27`、Object `33/15`、Spatial `0/0`；Long `3/0`、Goal `0/38`、Object `26/12`、Spatial `0/1`。results SHA256依次为`3c2643cf85c1a33a8335fd96636b46e55deef9f1839747c88cd7d62d30fa8334`、`db01087c00b2dd162f6900cead653d553d7c9e2e8ae8c9e20535c5902624fce6`、`f2ef8786ffb536b03483de3900a9fcab3fa3b6e417862c73cc89532174a8af10`。
- paired closed-loop比较中，step500对800为500-only `27`、800-only `30`、exact `p≈0.791`，两点实质持平；step300对800为23/41、`p≈0.0328`。因此functional val loss从step500的`0.134224`连续升至step800的`0.140583`可提示train–val分叉和候选区间，但不能精确排序77与80个闭环成功，最终best仍必须由完整rollout决定。
- 该轨迹存在决定性的scheduler混淆：旧八卡global batch128实验使用warmup50/decay1200；四卡global batch64若按action-query数保持同一学习率轨迹，应机械换算为warmup100/decay2400。现有warmup100正确，但decay6400令step500–800仍接近peak LR，所以`80/400`既不能归因于bias恢复，也不能作为当前架构上限。
- 干净修正只把cosine decay改为2400；冻结prefix、Action Expert memory、Meta-LoRA、temporal/layer/slot架构、全部已恢复conditional bias、数据、sampler和loss均不变。fresh首段到step1200并每100步在驻留进程测封存val-loss panel；若仍未建立闭环峰后持续下降，再exact-resume到1800/2400。

## query-scaled bias-restored AS训练曲线（2026-07-23）

- fresh四卡轨迹在step100–800的task-balanced val functional loss依次为`0.135237/0.141384/0.135191/0.134058/0.134964/0.135579/0.141342/0.139462`。step200上冲在300完全恢复，证明单点不能早停；step400后500/600/700连续上升，800虽较700回落但仍比400高`0.005403`，已形成可执行的候选谷底与峰后区间。
- 同期每100步train-loss mean为`0.138046/0.128935/0.122111/0.117919/0.116702/0.114383/0.110524/0.110282`，持续下降而validation自400后恶化，支持真实train–val分叉。按owner要求用趋势避免无意义长训，run在完整step800 checkpoint和validation后停止，不机械执行原首段1200。
- step800保存了全部24 train tasks的50 action episodes和50 teacher videos覆盖，累计51,200 global action queries；checkpoint manifest payload SHA256为`4198c15cd82c0acc000951462ec6c410273c6d2ea474f5f9673369173fb963a1`，Writer state SHA256为`e680c4f2f45acf4a35ea664ae7078958345bf6b601e0b4ced21880eba880debf`。中断信号到达前额外完成step801–809，但没有覆盖latest checkpoint；这些rows保留作透明运行证据，科学候选只使用原子step800及以前checkpoint。
- 完整correct-video closed-loop在step300/400/500/600/800依次为`57/91/86/87/88`；逐任务step400为Long `2/0`、Goal `0/43`、Object `31/15`、Spatial `0/0`。step400相对step300的paired flips为`49/15`、exact `p=2.44e-5`；相对500/600/800分别为`33/28`、`26/22`、`29/26`，后三个差异均不显著。故step400是该轨迹已观测真实峰值，step300已显著较差，无需再补step200。
- val-loss的用途需要保持克制：本轮最低val-loss和最高closed-loop都在step400，且400后的train–val分叉正确提示停止长训；但五个候选的loss-success Spearman仅`-0.10`，step300的loss接近400而success低34。因此它能判断继续/早停并收缩候选区间，不能独立精确排序checkpoint，最终best仍由完整rollout决定。
- 当前Action-Memory AS首轨迹没有通过相对SFT门槛：step400 AS为`91/400`，旧rank128 Source-SFT step400为`122/400`；paired AS-only/SFT-only=`25/56`、exact `p=7.52e-4`。在充分探索新的Source-SFT ceiling同时，下一轮AS应优先修正训练统计效率，而不是继续当前已经过拟合的轨迹：目前每个task-condition每步只有16个action queries，而rank128 SFT为128个；这会让约10.12M参数Writer的functional梯度方差远高于约10.30M参数SFT。

## 四卡rank128 Source-SFT ceiling profile（2026-07-23）

- 在物理GPU0–3上用4个对称DDP ranks、batch128/rank完成4步真实forward/backward；global batch保持512，与旧八卡rank128轨迹完全相同。四步均finite，step wall为`14.919/14.109/14.152/14.190s`，后3步平均吞吐约`36.18 queries/s`。
- 峰值CUDA allocated/reserved为`54,998,429,696/67,979,182,080` bytes，保留约14GB物理余量；无需降低batch。run contract、metrics、summary SHA256依次为`9a15add8...8479`、`ff033c3f...d2a0`、`c6b3c1f6...70cd`。
- 正式轨迹沿用已建立的rank128 optimizer：warmup100、cosine decay800、每100步checkpoint。首段到step800；之后只在val-loss或闭环候选仍未充分时按300步exact-resume。每个checkpoint在同一驻留policy上原地测封存512-query validation action loss，不卸载模型、不更新参数；完整8×50 closed-loop仍决定最终best。

## AS query-matched训练修正（2026-07-23）

- 首轨迹每rank每个task/video condition只用16个action queries，而容量匹配的rank128 SFT每rank为128；两者虽然训练参数约10.12M/10.30M相当，单步functional梯度统计精度并不相当。这是先于改Writer架构需要消除的混杂。
- 修正保持Writer、16 memory tokens、Meta-LoRA、temporal/layer/slot、rank16输出LoRA和信息墙全部不变：一次生成adapter后，将128 queries拆成8个16-query policy microbatches；每个microbatch独立求adapter梯度，按真实query数加权平均，再只对同一Writer图反传一次。峰值显存保持原batch16量级，计算量约增至8次policy forward。
- 该行为仍由现有`ember.writer.as_step`和同一canonical runner拥有，没有并行入口。normal以及已有contrast模式都复用同一微批机制；paired contrast仍在整组microbatch前后恢复并核验相同policy RNG。正式配置在真实四卡profile前保持pending。
- owner随后明确batch size不应成为公平性门槛。最终待profile方案不再用每condition 64 queries去机械匹配SFT；改为四卡每rank顺序累计2个独立conditions、每condition保留16 queries，使每update合计8 conditions/128 queries，与旧八卡AS逻辑单位一致。上面的128-query/condition方案保留为未启动的设计provenance，不进入正式实验。

## rank128 Source-SFT step100–400训练–validation分叉（2026-07-23）

- 新四卡fresh run的固定512-query validation loss从step100的`0.1330666`连续升到step200 `0.1333360`、step300 `0.1341674`、step400 `0.1371306`；100→400相对恶化约3.05%，最后一段恶化约2.21%。同期train-loss区间均值持续下降，形成了比单点波动更强的早停证据。
- loss曲线只把closed-loop候选收缩到step100–400，不能独立宣布step100最优。旧rank128完整success高度非单调，因此必须比较四个checkpoint的同seed 8×50 rollouts；若真实success仍在最晚点上升，则从step400 exact-resume继续。
- 中间formal checkpoint已由run-contract hash、四rank RNG/optimizer state、LoRA hash和原子manifest封存。正式validation不需要伪造整个run完成；缺失的最终summary保持为null并明确标注checkpoint-before-completion，同时final/test仍禁止打开。

## Source-SFT八卡/四卡训练量口径与step100–800闭环（2026-07-24）

- 旧八卡rank128轨迹每个optimizer update覆盖8个task小批和512个action queries；其`step400=122/400`对应400次更新与204,800 queries。当前四卡轨迹用batch128/rank保持每步512 queries，所以同为step400时主要训练量大体可比。checkpoint实数进一步确认两者均消费`204,800` examples；旧/新每task范围分别为`8,512–8,576`与`8,448–8,576`。每次更新内4个更大task小批与8个较小task小批主要改变梯度方差和顺序，不能据此将四卡step机械乘2或把step800称为旧step400的等价点。
- 当前四卡fresh轨迹step100/200/300/400/500/600/700/800的完整validation success为`81/95/68/78/94/99/108/97`。step700是该轨迹当前best；600→700 paired flips为`19/28`，700→800为`31/20`，均不足以证明显著上升或持续下降。旧八卡step400的`122/400`仍是全部SFT候选的incumbent，而不是可移植的早停step。
- 对应四卡functional val loss为`0.133067/0.133336/0.134167/0.137131/0.134146/0.134832/0.135634/0.135192`，与闭环曲线只呈弱对应，因此后续仅微弱参考。四卡run已从完整step800原地exact-resume到1100；候选判断继续以同seed 8×50 rollout为主，不因batch或卡数变化机械重启。

## Action-Memory时间顺序诊断与最小结构修正（2026-07-24）

- 旧Action-Memory checkpoint在固定语言下换跨suite视频会明显改变有效LoRA，但完整视频倒序/打乱的相对变化仅约`0.036/0.027`，远小于单帧或重复端点帧的`0.237–0.312`；因此问题不是公共LoRA塌缩，而是temporal路径近似将视频作为无序状态集合。
- 当前最小修正不预设functional loss与closed-loop错位，也不增加额外训练目标：只让顺序通过RoPE直接进入temporal Q/K，并用4个不传播query residual的learned memory queries替代单一pool。显式帧差分、手工phase和order auxiliary均暂不采用，以便500-step实验直接回答正常AS监督是否会利用可表达的时间顺序。

## temporal-RoPE Writer 500-step结果（2026-07-24）

- bias-enabled Action-Memory、Meta-LoRA、信息墙和完整rank16 LoRA均未改变；唯一结构变化是temporal self-attention使用原始frame index的1D RoPE，并以4个condition-only temporal memory queries保留多阶段摘要。Writer为`11,252,737`个训练参数，约为rank128 Source-SFT的`1.093×`。
- 四卡native global64 fresh训练到step500，训练body wall为`1188.6s`。封存512-query functional validation loss在step400/500为`0.1364674/0.1369167`，几乎持平且后者略差；它正确提示没有继续改善，但最终选择仍由closed-loop决定。
- 优化后的`per_sample_lora_batched_replan` evaluator在step400/500分别得到`108/400`和`98/400`。paired rows中step400-only/step500-only=`24/14`、both=`84`；逐task step400→500为Long `5→4, 3→0`、Goal `0→0, 37→35`、Object `37→35, 26→24`、Spatial `0→0, 0→0`，故step400是明确observed-best。它仍低于rank128 Source-SFT incumbent `122/400`以及旧v1 AS的`119/400`，本次结构修正没有恢复AS绝对泛化上限。
- 仅在step400进行post-selection、无action/reward/outcome的特异性诊断。保持正确language不变，仅换跨suite错误视频时，temporal feature、LoRA参数和有效LoRA更新的中位相对L2分别为`0.1228/0.1595/0.2267`；同task另一demo对应`0.0255/0.0368/0.0403`。8/8 tasks的跨suite有效更新变化均为`0.1770–0.2715`，说明视频任务内容已经稳定进入adapter，不是公共LoRA塌缩。
- 对同一视频倒放或确定性乱序时，有效LoRA更新的中位相对L2却仅为`0.00937/0.00699`，cosine中位数为`0.999957/0.999976`；而只保留首/中/末帧时为`0.1745/0.1124/0.3339`。因此模型使用了多帧内容和端点状态，但normal functional监督几乎没有让新RoPE路径学习动作顺序。这个结论是representation/adapter诊断，不冒充错误视频的closed-loop performance arm。
- 当前最直接的科学结论不是“视频没用”，而是“视频语义内容有用、时间顺序仍未被当前监督识别”。由于correct性能未超过SFT，按owner的快速子任务合同不启动contrast、更多checkpoint或RL，先停止汇报。

## Action-Forecast Writer最终设计判断（2026-07-24）

- 旧Action-Memory的主要科学缺口不是视频内容完全塌缩，而是它只聚合逐帧
  hidden states：倒序/乱序的effective-LoRA相对变化中位数仅
  `0.00937/0.00699`。仅靠temporal RoPE没有让normal functional loss学会
  “一个连续动作计划怎样随新观察被修正”。
- 新设计因此不再保存18层memory hidden states，而是让每个teacher frame经
  imagined-state + PaliGemma VL Meta-LoRA + Action Meta-LoRA执行完整10-step
  flow，保留最终50×7 action forecast。相同绝对未来时刻的滚动forecast构成
  receding-horizon Plan与有方向的Revision，直接把“最终打算做什么”和“新帧
  如何修正旧计划”暴露给变长temporal encoder。
- 同一视频所有帧必须使用相同fixed flow noise；prefix KV只算一次并供10次flow
  复用。temporal只读取normalized actions/revisions，不再额外读取pseudo-state
  或18×10 hidden states。
- 320个LoRA queries通过显式单向cross-attention读取procedural memory；普通
  conditional bias保留，final factor heads零初始化保证public task LoRA identity。
  实现后Writer为`10,161,217`个训练参数，是rank128 Source-SFT
  `10,297,344`的`98.68%`；public rank16 LoRA仍为76 tensors、
  `1,287,168` scalars。
- 这是2026-07-24的历史设计判断。其实现、profile与结果保留在本ledger和Git；
  该历史版本后来由v4文档`docs/action_forecast_writer_design.md`覆盖；当前
  活动架构为`docs/action_forecast_writer_v5_design.md`。

## Action-Forecast Writer实现与训练profile（2026-07-24）

- canonical路径已原位完成：冻结π0.5/PaliGemma参数但保留到Writer的梯度路径；
  每帧只读`agentview_rgb`与正确task language，构造imagined-state并经VL/Action
  Meta-LoRA执行完整10-step flow，随后用同绝对控制时刻Plan/Revision tokens、
  变长RoPE temporal encoder与单向query decoder生成完整rank16 LoRA。旧
  `action_memory.py`、`conditioning.py`、旧活动config/schema/tests及独立
  specificity runner均已退役。
- 四卡真实forward/backward profile封存训练选择为stride5、
  frame-microbatch32、每rank 16 action queries。覆盖全部24个train tasks的
  17-step长profile共消费1088 queries；稳定step wall中位数`6.1183s`、
  p95 `9.0442s`，global query throughput中位数`10.4611/s`，峰值
  allocated/reserved为`67,078,778,368/70,183,288,832` bytes，未见OOM或
  nonfinite loss/gradient。
- frame-microbatch64不是可用余量：rank1实测占用`80,821/81,920 MiB`后停止
  前进而其余DDP ranks仍在等待，零step完成，故选择32。stride10只保留单步
  吞吐参考；owner指定stride5符合真实控制节奏且无需继续扩测。
- exact-resume已完成step1→2并恢复optimizer、scheduler、sampler、flow-noise
  cursor与四rank RNG；resume contract SHA256为
  `c7a3dc88ae840d386b9d825e6f71f2f9613fccf0f37adf85b29c5a577d0ecd68`。
- 正式AS合同为四卡、每段300 steps、每75 steps一个checkpoint；`12000`只是
  可延续的预声明schedule horizon，不是科学停止点。训练期间只对每段第2/4点
  做correct-video validation并按趋势补点；wrong/shuffled/reversed严格推迟到
  充分训练并选出observed-best以后，不能再把单卡最小诊断前置。

## Action-Forecast Writer两阶段评测拓扑（2026-07-24）

- 原先`replicas_per_gpu`会让每个rollout replica都同时加载source π0.5和Writer，
  因而Writer生成并发、rollout并发和显存占用被错误绑定；早期step150 r3/r4
  profile只能证明耦合路径可执行，不能决定最终rollout吞吐。
- canonical evaluator现先按独立`writer_generators_per_gpu ×
  writer_generation_batch_size`生成固定panel的逐episode rank16 LoRA，并以
  adapter/model/tokenizer/task-state/generation grouping共同hash的原子cache封存。
  rollout拓扑不进入cache identity，所以相同生成recipe可在r3/r4/r5等纯rollout
  profile之间复用；生成batch改变时则不会错误复用数值可能不同的LoRA。
- generator写完后不重载共同模型：只关闭raw-video handles、tokenizer和Writer
  专属模块，保留同一进程内已加载的source π0.5、physical identity LoRA及
  batched-LoRA hooks，释放CUDA cache后直接转为首个rollout worker；launcher
  此后才启动额外rollout-only replicas。生命周期证据显式记录
  `source_policy_reloaded=false`。
- 结果同时保留launcher end-to-end wall与shard rollout-only window，分别报告
  end-to-end和rollout-only吞吐；不能再用模型加载/LoRA生成时间污染replica选择。
  缓存entry采用目录级原子发布，manifest核验每个state/file hash，预计完整
  400-entry panel只新增约1.03GB tensor数据。

## Action-Forecast AS step150–600 validation曲线（2026-07-24）

- 同一固定8-task×50 correct-video validation panel的step150/300/450/600结果
  为`75/99/93/118`。step600逐任务为Long `13/2`、Goal `1/34`、Object
  `46/17`、Spatial `0/5`；它超过四卡rank128 Source-SFT observed-best
  `108/400`，距旧八卡`122/400` stretch目标4个成功。
- step450→600在完全相同的video-level cache/seed协议下，paired flips为
  450-only `29`、600-only `54`，exact McNemar `p≈0.00804`。净提升来自
  Long task1 `0→13`、Goal task6 `23→34`、Object task1 `40→46`和Spatial
  task3 `1→5`等多个tasks；Object task3同时从`24→17`，不能把aggregate
  上升误写成所有任务单调改善。
- step300→450仅`99→93`，且Long2/Object3改善而Long1/Goal6下降；这是幅度小、
  任务方向混合的正常波动，不是owner规定的明显峰后下降。随后step600升到新高，
  进一步证明不得因一个或多个略低checkpoint停止。
- step150/300生成时旧cache按init state重复计算同一可见视频，并使重复视频使用
  不同Writer内部随机流；科学输入墙未泄漏，但协议不再作为最终峰值复测标准。
  step450/600开始按`(language task, video task, demo_id, condition, order
  transform)`去重，并为同一可见视频固定生成随机流，因此后续checkpoint之间
  可严格paired比较。若最终best落回step300附近，须用新协议独立复测该候选。
- 正式panel的400个episodes去重为259个唯一视频LoRA、141个aliases；每卡一个
  generator、batch100只需一个batch，四卡分别生成64/65个。step600最大生成
  wall为`55.50s`、峰值allocated/reserved为`16.55/19.27GB`，释放Writer后
  原进程保留约`9.55GB` source policy并直接转rollout；每卡6 replicas的
  rollout-only吞吐为`0.61045 episode/s`，400 episodes无OOM、无重试。

## Action-Forecast AS step750/900继续探索（2026-07-24）

- step750/900完整correct-video validation为`104/113`，逐任务分别为Long
  `7/1`、Goal `1/34`、Object `42/17`、Spatial `0/2`，以及Long `7/2`、
  Goal `1/38`、Object `42/16`、Spatial `0/7`。两点都低于step600的
  `118/400`，但750后的900已经回升9个成功，故不存在单调或持续下降。
- step600→750 paired为600-only `35`、750-only `21`，exact McNemar
  `p≈0.0814`；step750→900为750-only `20`、900-only `29`，
  `p≈0.2529`；step600→900为600-only `36`、900-only `31`，
  `p≈0.6254`。step900与当前best实质持平，750只是一个未复现的较低点。
- 任务方向同样不满足停止条件：600→750的净下降集中于Long1、Object1、
  Spatial3，Goal6/Object3保持不变；随后750→900又由Goal6与Spatial3回升。
  这不是“幅度非常明显、由多个tasks共同贡献且独立复测成立”的峰后下降，
  所以必须继续训练。
- step750/900的results SHA256分别为
  `584a5c2164b631eb96fc6d60589720ad4ad297626ac750548b78b953c664ea22`
  和`4c1d62d0b3fbc847b776cdbcce0558d502b12a70381fa7f72d0913112d32a1cf`；
  两次均为32/32 shards、400 rows、24 workers exit0、无重试。对应
  rollout-only吞吐为`0.61976/0.62315 episode/s`，继续确认r6稳定。

## Action-Forecast AS step1050/1200新高（2026-07-24）

- step1050/1200完整correct-video validation为`117/125`。逐task按Long
  1/2、Goal 3/6、Object 1/3、Spatial 1/3分别为step1050
  `6/1, 0/40, 44/18, 0/8`，step1200
  `6/3, 1/38, 45/20, 1/11`。step1200在8个tasks中的7个有成功，aggregate
  超过四卡rank128 Source-SFT best `108/400`，也略高于旧八卡stretch
  `122/400`；后者仍不是必须超过的停止门槛。
- step600→1050 paired为600-only `31`、1050-only `30`，exact McNemar
  `p=1.0`，两者实质相同；step600→1200为600-only `31`、1200-only `38`
  （净`+7`，`p≈0.4704`）。step1050→1200为1050-only `15`、1200-only
  `23`（净`+8`，`p≈0.2559`）。此前750的回落不仅未持续，1200还产生新高，
  因此当前不存在可讨论的峰后下降，更不满足强下降停止条件。
- 两次评测均完成32/32 shards、400 rows、24 workers exit0且无错误/重试。
  step1050/1200 results SHA256分别为
  `b88303cbf2a170315a1d5523f58cb1b0b3346d4671a9e37f024a0dda23f339a7`
  和`c575591ba36d949578061aa164f59572fcd59c81952a3f301c4c66b4afd38dd0`。
  两次均只生成259个唯一视频LoRA，batch100、4个generator batches；
  rollout-only吞吐分别为`0.62721/0.60638 episode/s`，r6继续稳定。
- step1200 checkpoint累计76,800 action queries、4,800 video conditions；
  24 tasks各3,200 queries、200次视频访问且50/50 videos全覆盖，Writer、
  optimizer/scheduler、sampler/cursor和四rank RNG文件逐SHA核验通过。正式
  轨迹已从该checkpoint同合同exact-resume到step1500，继续优先评测
  step1350/1500；当前不能进行最终specificity诊断。

## Action-Forecast AS step1350/1500弱回落（2026-07-24）

- step1350/1500完整correct-video validation为`120/119`。逐task按Long
  1/2、Goal 3/6、Object 1/3、Spatial 1/3分别为step1350
  `10/2, 0/32, 43/19, 0/14`，step1500
  `8/1, 1/33, 43/18, 0/15`。相对step1200的`125`只低5/6个成功，且任务
  方向混合；两个后续点彼此几乎完全持平。
- step1200→1350 paired为1200-only `32`、1350-only `27`，净`-5`、
  exact `p≈0.6029`；step1200→1500为`26/20`，净`-6`、`p≈0.4614`；
  step1350→1500为`26/25`，净`-1`、`p=1.0`。这正是owner明确禁止据此停止的
  “多个后续checkpoint只是略低”，远非多个tasks共同贡献的明显强下降。
- 两次评测均完成32/32 shards、400 rows、24 workers exit0且无错误/重试；
  results SHA256分别为
  `edf5b889eb4d6fdc0da9554966f97e8f9e5417cae250597526b2da7336337327`
  和`1a9232906b30d1d2ae679d8b726f332af5279aaff4a8e2ea1d8873981c035cc5`。
  rollout-only吞吐为`0.60530/0.63953 episode/s`；step1350曾瞬时达到约
  `80.6GiB`但未OOM并随worker完成释放，r6仍完成全panel。
- step1500 checkpoint累计96,000 queries、6,000 video conditions；24 tasks
  各4,000 queries、250次视频访问且50/50 videos全覆盖，完整恢复文件逐SHA
  验证通过。现已同合同exact-resume到step1800并将正式评测step1650/1800；
  specificity继续推迟。

## Action-Forecast AS step1650/1800仍未建立强下降（2026-07-24）

- step1650/1800完整correct-video validation为`120/114`。逐task按Long
  1/2、Goal 3/6、Object 1/3、Spatial 1/3分别为step1650
  `6/3, 0/33, 43/21, 0/14`，step1800
  `4/2, 1/34, 45/17, 0/11`。step1800仍有6/8 tasks非零，后段变化不是
  多个tasks一致塌陷。
- step1200→1650 paired为1200-only `29`、1650-only `24`，净`-5`、
  exact `p≈0.5831`；step1200→1800为`31/20`，净`-11`、`p≈0.1608`；
  step1650→1800为`30/24`，净`-6`、`p≈0.4966`。因此
  `125→120→119→120→114`最多是第一处略大的回落，幅度和复现性均不满足
  owner要求的“远超400-rollout正常波动、多个tasks共同贡献、独立复测仍成立”。
- 两次评测均完成32/32 shards、400 rows、24 workers exit0且无错误/重试；
  results SHA256分别为
  `e800361b3bcdf57d57f39f635b20136f043a73d80197560098f0e087b5c35f9a`
  和`5c0de70f6b75c63d332e6e6e35ece5f2f4a57041cf364111123b6d73f61654d3`。
  两次均只生成259个唯一视频LoRA，batch100、4个generator batches并复用
  source-policy进入rollout；rollout-only吞吐为`0.61633/0.61129 episode/s`。
- step1800 checkpoint累计115,200 action queries、7,200 video conditions；
  24 tasks各4,800 queries、300次视频访问且50/50 videos全覆盖，完整Writer、
  trainer、sampler/cursor和四rank恢复状态已封存。轨迹已同合同exact-resume
  到step2100，优先正式评测step1950/2100；仍不能开始最终specificity诊断。

## Action-Forecast AS step1950/2100回落未持续（2026-07-25）

- step1950/2100完整correct-video validation为`110/114`。逐task按Long
  1/2、Goal 3/6、Object 1/3、Spatial 1/3分别为step1950
  `4/0, 1/28, 45/19, 0/13`，step2100
  `4/0, 1/32, 44/14, 0/19`。1950的较低aggregate在2100回升4个成功，
  task间仍有明显相反方向。
- step1200→1950 paired为1200-only `34`、1950-only `19`，净`-15`、
  exact `p≈0.0534`，但净下降中`-10`集中在Goal-6；step1200→2100为
  `36/25`，净`-11`、`p≈0.2000`。step1800→2100恰为`28/28`、净`0`、
  `p=1.0`，step1950→2100则为`22/26`、净`+4`、`p≈0.6655`。
  因此`114→110→114`不是远超rollout噪声且多task共同贡献的持续下降。
- 两次评测均完成32/32 shards、400 rows、24 workers exit0且无错误/重试；
  results SHA256分别为
  `c62e75973b8196e4e6052cecde8e0add00dd948f0536385ac5be44d0a158a576`
  和`934382c211027c3b6407b46898e65f708dfda136a1bc6cbef8a60f18cacf3905`。
  两次均生成259个唯一视频LoRA并复用source-policy；rollout-only吞吐为
  `0.61188/0.61723 episode/s`。
- step2100累计134,400 action queries、8,400 video conditions，24 tasks各
  5,600 queries、350次视频访问且50/50 videos全覆盖；完整checkpoint逐SHA
  与manifest一致。轨迹已同合同exact-resume到step2400，继续正式评测
  step2250/2400；specificity仍只保留给最终observed-best。

## Action-Forecast AS step2250重返峰值平台（2026-07-25）

- step2250/2400完整correct-video validation为`123/111`。逐task按Long
  1/2、Goal 3/6、Object 1/3、Spatial 1/3分别为step2250
  `5/3, 0/34, 45/20, 1/15`，step2400
  `8/0, 0/36, 43/18, 0/6`。2250只比step1200 observed-best低2个成功，
  证明此前`114/110/114`不是已建立的持续下降。
- step1200→2250 paired为1200-only `30`、2250-only `28`，净`-2`、
  exact `p≈0.8957`，两者实质相同；step1950→2250为`21/34`、净`+13`。
  step1200→2400为`32/18`、净`-14`、`p≈0.0649`，step2250→2400为
  `32/20`、净`-12`、`p≈0.1263`。但2400紧随接近峰值的2250，且Long-1、
  Goal-6在2400反向提升，因此只是单点回落，不满足持续且多task共同下降。
- 两次评测均完成32/32 shards、400 rows、24 workers exit0且无错误/重试；
  results SHA256分别为
  `35ff55e3f8c2a5f8ed8885cf2a335862879b255189907a098e63d7ad61525655`
  和`f5c9a77b40048e6826a8b667c887e6d796c14f71be68fe9d9a249329bdc036df`。
  两次均生成259个唯一视频LoRA并复用source-policy；rollout-only吞吐为
  `0.60369/0.61011 episode/s`。
- step2400累计153,600 action queries、9,600 video conditions，24 tasks各
  6,400 queries、400次视频访问且50/50 videos全覆盖。轨迹已同合同
  exact-resume到step2700，继续正式评测step2550/2700；不能开始specificity。

## Action-Forecast AS step2550与observed-best视频特异性（2026-07-25）

- step2550完整correct-video validation为`124/400`，逐task按Long 1/2、
  Goal 3/6、Object 1/3、Spatial 1/3为`8/0, 2/38, 46/20, 0/10`；
  results SHA256为
  `26a9595ca613e8d5beb46444da1275c47f24d76e2cb17e76bb3baa0dca5c5062`。
  它再次回到step1200的`125`附近，确认原正例轨迹到step2550仍未形成持续、
  多task共同贡献且远超rollout噪声的峰后下降。step2700 checkpoint已完整
  保存，但owner指令先检查当前最高AS，因此未把未评测的2700用于选择。
- step1200的correct/cross-suite-wrong为`125/67`；paired
  correct-only/wrong-only为`71/13`，exact McNemar
  `p=7.8639e-11`，6/8 tasks净受益，故视频内容特异性明确成立。wrong逐task为
  `7/1, 1/16, 37/5, 0/0`，results SHA256为
  `e43e5ed054156cad659f074b05a6d17755b0d1685f83bf18a0fda99a5f4a632c`。
- 同一paired panel的shuffled/reversed分别为`121/124`；相对correct的paired
  flips为`17/13`（`p≈0.5847`）和`15/14`（`p=1.0`）。生成LoRA有效
  `B@A`相对L2差的中位数仅为`0.001101/0.001787`，说明失败不仅是400-rollout
  方差：Writer表征本身也几乎不随帧序改变。两份results SHA256分别为
  `636b448d1a829f5ebffa1aa517f94305da1c7dd766d771af15211f24b20dbd3a`
  和`5edba10a3efc1a39dacbdc10484fe21b1258bf8d956c80c5202ce543afa43d59`。
  因此Action-Forecast通过性能与correct/wrong内容门，但没有通过
  shuffled/reversed顺序门，独立RL-Writer的启动条件仍为false。
- 最小修正保持同一Writer架构、信息墙、LoRA schema与唯一AS runner，只在每个
  正例functional update后复用同一action batch和Writer flow noise，对
  shuffled/reversed帧视图施加weight `0.5`、margin `0.01`的stop-gradient
  negative functional gradient。四卡batch16、frame-microbatch32真实两步
  warm-start profile无OOM/非finite；峰值allocated/reserved为
  `67,077,086,720/69,250,056,192` bytes，稳态第二步`11.6213s`。该profile
  只证明mechanics可运行，不作科学性能解释。

## Action-Forecast Writer v2机制修正证据（2026-07-25）

- owner明确不接受用loss强行制造顺序差异，所以上述order-contrast只保留为
  已否决的历史，不再是活动路径。canonical AS现只含normal positive
  functional action loss。
- 对旧step1200在8个validation tasks各2条独立视频做无训练counterfactual：
  raw directed event triplet的normal→reversed/shuffled time-centered相对L2
  中位数为`0.2233/0.2296`；旧Revision token仅`0.0281/0.0316`，说明旧合成
  确实消减了有向差异；移除query residual和additive stability后，新Revision
  content恢复到`0.3554/0.2418`。证据：
  `/data/ymdai/outputs/ember/pi05_action_forecast_step1200_revision_v2_counterfactual_val8x2_20260725/summary.json`。
- v2因此采用三个同构的content-only信息闸：28个state slots、Revision read
  和320个LoRA queries的routing identity都只进入attention Q/K，输出residual
  只携带memory-derived content。Revision稳定性统计只通过
  `[0.75,1.25]`乘法gate调节directed content，不能additive覆盖它。
- v2真实Writer参数为`10,125,376`，是rank128 Source-SFT
  `10,297,344`的`98.33%`；public LoRA仍为76 tensors、
  `1,287,168` scalars。
- GPU0–3、stride5、frame-microbatch32、batch16/rank的真实fresh step1和
  exact-resume step2均finite；resumed step为`6.5025s`，峰值
  allocated/reserved为`67,088,471,040/69,235,376,128` bytes，source policy
  trainable count为0。该结果只封存机械可运行性，科学结论等待step300/600
  closed-loop validation与最终特异性。

## Belief-v3最终架构与raw-RMS强度证据（2026-07-25）

- v3把每个绝对控制时刻压成一个固定布局的256维Belief：
  前128维Plan只编码最新forecast的7维action，后128维Revision读取所有更早
  covering forecasts相对Plan的signed/absolute residual。lead、age、count和
  absolute time只承担寻址，不进入content value或residual。
- Revision显式尺度最终采用
  `Revision_u=stopgrad(m_u)*RMSNorm(z_u)`，其中`m_u`是frozen source
  normalization下的原始7维residual RMS。train-only、optimizer-step0的960个
  非零样本分位数Q00/Q10/Q25/Q50/Q75/Q90/Q100为
  `0.0031/0.0400/0.1909/0.4232/0.6212/0.7630/0.8353`，说明原始无量纲尺度
  可直接使用；这些统计只作诊断，不设置模型超参数。`tau`和分位数校准已从
  模型、配置和authority删除。
- Temporal与LoRA query decoder均满足zero-preserving合同：routing identity、
  lead/count/strength和RoPE只影响Q/K；V与residual只读取content。Belief置零时
  temporal memory、query content和dynamic LoRA严格为零。Writer总参数
  `10,247,872`，是rank128 Source-SFT `10,297,344`的`99.52%`。
- stride5固定后的效率profile选择frame-microbatch32、batch20/rank。相同
  Belief-v3 topology的12-step profile稳态中位`6.49s/step`、全局
  `12.32 queries/s`；frame-microbatch40更慢，48在首步前达到
  `81,153/81,920 MiB`并失去稳定前进。
- 最终raw-RMS代码又完成四卡fresh step1和step1→2 exact-resume；contract
  SHA256为`352f7409d671d97399262b46afe0d415b4b6c145bcca66cbe43725474fa8e234`，
  resumed step为`6.9184s`、`11.5634 queries/s`，峰值allocated/reserved
  `77,090,931,200/83,730,890,752` bytes。梯度finite且非零、source policy
  trainable count为0，四rank RNG/state、Writer、optimizer/scheduler及
  sampler/data/flow-noise cursors均完整可恢复。
- step600后的顺序特异性采用两级门：先在相同帧集合上逐层比较normal/
  shuffled/reversed的forecast residual、Revision、Temporal memory、query
  content和effective LoRA；只有最终差异明确且跨多个tasks/videos稳定，才投入
  实际paired validation rollout。内部路径仍塌缩时直接定位层级，不浪费环境评测。

## Belief-v3 step600内部顺序特异性结论（2026-07-25）

- fresh formal run一次连续完成step0→600，wall `4157.74s`；累计
  `48,000` action queries和`2,400`独立task-video conditions，24 tasks各
  2,000 queries、100次video访问且50/50 demos全覆盖。step600 checkpoint
  schema v3的Writer、trainer与四rank state共6个文件逐SHA校验通过；
  Writer SHA256为
  `a1101c62337c2f365a5891f856360a589d5c976665dafe6eeb476dd6c95695df`。
- 正式validation schedule的8 tasks×2个不同videos内部panel保持task
  language、帧集合、frame indices和Writer flow noise配对，只改变帧顺序；
  actions/reward/outcome读取均为0。normal→reversed/shuffled的相对L2中位数
  依次为：action forecasts `0.0725/0.0678`，Plan-relative residual
  `0.0914/0.0980`，Plan tokens `0.0596/0.0514`，Revision tokens
  `0.0270/0.0266`，Belief `0.0523/0.0464`。
- Revision修正确实有效：其time-centered相对L2达到
  `0.1753/0.1598`，明显高于raw值，说明有向forecast disagreement已进入
  Belief，失败不再位于Revision合成。
- Temporal保留了动态差异的绝对量，但被时间常量淹没。reference RMS从Belief
  `0.834`增长到block1 `3.06`、block2 `9.62`；差异RMS仍约
  `0.044→0.046`，time-centered相对L2也保持
  `0.148/0.133`，但raw相对L2降到`0.00479/0.00425`。
- 单路LoRA query read随后再次塌缩：query block1为
  `0.000555/0.000310`，normalized query output为
  `0.0000719/0.0000448`，最终effective `B@A`只有
  `0.000297/0.000169`。8个tasks全部很小，reversed/shuffled每task中位最大也
  只有`0.000571/0.000618`；该结果不高于旧失败版，内部顺序门明确失败。
- “把RMSNorm加回来”不是充分修正。用同一已训练权重把Temporal/query V改为
  normalized content并把Temporal出口约束到单位RMS，effective `B@A`仅恢复到
  `0.000339/0.000266`，仍是同一失败量级。
- 决定性反事实是只把真实Temporal memory减去其masked时间均值，再交给完全
  相同的现有query/factor heads。memory自身为`0.1427/0.1352`，query output
  恢复到`0.1053/0.0825`，effective `B@A`恢复到`0.0543/0.0401`。
  reversed在8/8 tasks为`0.0098–0.1034`；shuffled除task13的
  `0.00059`外，其余7 tasks为`0.0173–0.4011`。这只作无训练机制归因，
  不是可报告的模型性能。
- 结论：Revision、Temporal动态分量和query表达能力都存在；根因是将
  task-global/time-constant belief与zero-mean temporal innovation混在同一路
  memory里，前者主导Q/K/V并让后者在集合读取前失去信噪比。下一版应显式拆分
  masked global mean与zero-mean temporal innovation，innovation每层重新
  centered，两个content-only reads独立归一化后再合成固定256维query content；
  不引入contrast loss或人工幅度超参数。按owner顺序，未运行昂贵的
  shuffled/reversed environment validation，也未推进RL。
- 三份内部证据summary SHA256依次为
  `161c3c392dae610be0f299810ecb6aec5773bc735f4a1c7d22bd3f9a25890313`、
  `92e48820881fc0c6038a370a2595148ad752bdd33d100ac6f376e3cb1cd7a603`、
  `0af1d094be0f058023fc793d1109aec3865ed06e15d963fc73802ef57094ac4c`。

## 32-token Visual-State新设计结论（2026-07-25）

- 现有证据把最大嫌疑定位到visual-state：旧virtual-state总RMS约`0.652`，
  跨帧变化仅约`0.0057`；逐帧future forecast容易退化为近似task-level action
  chunk。下游Revision虽能恢复部分有向差异，却不能替上游创造缺失的阶段状态。
- 新设计不再冻结随机decoder，也不把video state直接送入LoRA。它用第一帧
  建立绝对锚点，并为每帧同时读取相对首帧和相对前帧的有向变化；每个`h_t`
  都由`h_0+c_t`直接得到，不递归累计，兼顾绝对状态稳定性、局部运动方向和
  无漂移。
- 8个latent state/motion coordinates通过冻结digit-embedding基底渲染到
  PI05原生`" 128"`重复8次的32个state token位置；初始前向等同合法中性原生
  prompt，训练后仍只能在native digit子空间内表达state变化。
- VL与Action Meta-LoRA均保留并可学习、identity初始化。前者适配视觉域与
  image-state融合；后者让机器人把observer-view或未来人类teacher理解成
  “假如我是teacher，此刻接下来会怎么动”的机器人动作forecast。
- Plan、Revision、Belief、两层Temporal和LoRA decoder在该历史v4中的精确定义
  及退役边界见`docs/action_forecast_writer_design.md`。它不再是活动架构；
  后续v5见`docs/action_forecast_writer_v5_design.md`，当前v5.1见
  `docs/action_forecast_writer_v5_1_proposal.md`。旧v1/v2/v3/v4表述只解释
  历史实验，不再约束实现。

## 32-token Visual-State v4实现检查（2026-07-25）

- v4 Writer为`10,299,072`个训练参数，和rank128 Source-SFT的
  `10,297,344`只差`1,728`（`0.017%`）；输出仍严格遵守76个rank-16 tensor、
  `1,287,168` scalars的公共LoRA schema。
- 四卡真实profile确认frame-microbatch32、batch20/rank可运行，连续step2约
  `11.83 queries/s`；峰值allocated/reserved为
  `76,926,757,376/83,703,627,776` bytes。当前显存余量不支持把batch或
  frame-microbatch再安全上调，故不为寻找OOM边界浪费正式训练时间。
- 低成本75-step门控必须沿用正式1200-step scheduler，只在75提前停止；若把
  scheduler horizon本身压到75，会改变warmup/decay并使结果不能预测正式轨迹。
- step1→2恢复的loss与gradient norm逐值等于连续运行，所有数据、视频、
  flow-noise、optimizer、scheduler和rank RNG状态均恢复。CUDA新进程造成6个
  Writer tensors最大`4.28e-8`的数值差异；这是数值等价而非bitwise
  deterministic，不影响checkpoint可恢复性，但已明确记录。

## 32-token Visual-State v4 step75内部门结论（2026-07-25）

- v4在正式1200-step学习率时间轴的前75步已经形成明确视频特异性。
  reversed/shuffled的effective LoRA相对L2中位数为`0.0420/0.0468`，
  相比Belief-v3失败值`0.000297/0.000169`提高约两个数量级；16条reference
  videos全部非零，8个validation tasks都贡献差异。
- 这不是只看到了frame permutation：对同一图像先按相同permutation配对后，
  新visual coordinates仍改变约`0.1610/0.1463`，future forecasts仍改变约
  `0.0231/0.0152`。冻结image embedding在该context-only检查中严格为0，说明
  新增差异确实来自anchor/local visual-state上下文而非诊断错配。
- 正常video的action forecasts跨帧变化RMS约占总RMS的`60.4%`，不再是近似
  重复的task-level chunk。打乱后Revision strength均值中位比为
  `1.119/1.200`（reverse/shuffle），分别13/16和14/16视频对增加；正常顺序
  在统计上具有更一致的future forecasts。
- 下游虽会压缩差异，但不再消灭它：Belief相对L2
  `0.8217/0.7852`，Temporal `0.6902/0.6428`，query output
  `0.0528/0.0593`，effective LoRA `0.0420/0.0468`。因此两层Temporal和现有
  content-conditioned decoder暂时无需再加层或拆分路径。
- 同task另一demo与cross-suite wrong的effective LoRA中位差异分别为
  `0.0250/0.0714`；8/8 tasks均非零。internal specificity gate通过，可以
  从fresh identity进入正式0→1200；最终仍需在具备绝对能力的候选上完成
  correct/wrong/reversed/shuffled paired rollout。

## 32-token Visual-State v4正式训练与observed-best（2026-07-26）

- 正式run
  `/data/ymdai/outputs/ember/pi05_action_forecast_v4_as_development_seed7_ad0db5f_r4_s5_fm32_b20_s1200_20260725`
  已连续推进到step2400后按owner要求终止，不再续训。累计`192,000` policy
  samples；24个train tasks各`8,000` queries、`400`次video访问并覆盖全部
  50 demos。2400条metrics、32个functional validation点和step2400完整
  checkpoint均存在，无OOM、NaN或训练错误。
- run contract、run summary、training metrics和functional validation metrics
  的SHA256依次为
  `7389b8e847784809b1e2d210e2778150d77df952aec875343ce50b279ac784cc`、
  `072aefa438366db1418fc904b3bf50b125bc1c5032a7f8f33e651840688d64e4`、
  `bb3cf6a39808f7d7be3c51d2a526053efdc08a2d24e6467d9a2d199ee5bc7d7c`、
  `835eb52c955af35ef0e747bc945e0cb4cedad2becb1d546665069ea7282b725d`。
- 不再使用波动过大的80-episode快筛。完全相同的固定8-task×50 panel上，
  step675/825/900/1200/1275/1500/1875/2100/2400依次得到
  `100/109/82/96/94/92/90/90/89`。这些点覆盖functional global best
  step825、第二低点step1275、后段局部低点step1875和终点，故现有
  observed-best确定为step825的`109/400`。
- step825比四卡rank-128 Source-SFT `108/400`高1，但低于旧Action-Forecast
  step1200的`125/400`。两个版本使用同一固定panel；新版相对旧版30条
  gain、46条loss。净下降16全部集中在object-1 `45→38`和object-3
  `20→11`，其余六个tasks合计净变化为0。

## v4 step825完整视频特异性结论（2026-07-26）

- 内部panel使用8个validation tasks×2条reference videos，共16个references、
  64个paired comparisons，未读取actions/reward/outcome。证据位于
  `/data/ymdai/outputs/ember/pi05_action_forecast_v4_step0825_internal_specificity_val8x2_ad0db5f_20260725`；
  summary/rows SHA256为
  `618976d558ca58da592c6ab2eadf13e96caff44f971bc76afaf88b5a859682f1` /
  `f5a2735ae31d5a78729a61e359c8b3f2f08dca998f8eb4c2c01cb4d10e017cae`。
- effective LoRA相对L2中位数为：same-task other `0.0955`、shuffled
  `0.2598`、reversed `0.3255`、cross-suite wrong `0.8762`；对应Temporal
  memory为`0.1196/0.9348/0.9085/0.7533`。这说明same-task变化最小，顺序
  与任务语义差异也确实穿过完整链路，而不是再次生成公共LoRA。
- Revision strength均值的candidate/reference中位比为same `1.014`、
  shuffled `1.419`、reversed `1.340`、wrong `1.038`。shuffled/reversed
  确实显著改变forecast一致性；wrong主要改变语义和LoRA方向，而非简单放大
  分歧强度。
- step825五个固定400 arms全部使用同一task、language、init、env seed、
  policy seed schedule与配对Writer随机性。结果为：

| condition | total | Long-1 | Long-2 | Goal-3 | Goal-6 | Object-1 | Object-3 | Spatial-1 | Spatial-3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| correct | 109 | 6 | 2 | 0 | 40 | 38 | 11 | 0 | 12 |
| same-task other | 104 | 6 | 2 | 1 | 37 | 35 | 11 | 0 | 12 |
| cross-suite wrong | 99 | 0 | 3 | 2 | 44 | 39 | 7 | 2 | 2 |
| shuffled | 148 | 8 | 1 | 0 | 45 | 45 | 37 | 2 | 10 |
| reversed | 126 | 6 | 2 | 0 | 41 | 48 | 20 | 3 | 6 |

- same-task other与correct为both-success/correct-only/other-only/both-fail
  `80/29/24/267`，净`-5`、churn `53/400=13.25%`、exact McNemar
  `p=0.583`。它是行为影响最小的条件，支持“部分提取了同任务高层共性”；
  但53条确定性翻转仍不是完全稳健。
- wrong为`69/40/30/261`，净`-10`、churn `17.5%`、`p=0.282`；它并未
  在多个tasks上形成稳定伤害，说明任务语义特异性也尚未通过行为硬门。
- shuffled为`85/24/63/228`，净`+39`、churn `21.75%`、exact McNemar
  `p=3.48e-5`，是显著改善而非退化。两个object tasks合计由`49→82`，
  其中Object-3的correct-only/shuffled-only为`5/31`、`p=1.29e-5`。
- reversed为`79/30/47/244`，净`+17`、churn `19.25%`、`p=0.0675`；
  改善主要来自Object-1 `+10`和Object-3 `+9`，Object-1 paired flips为
  `1/11`、`p=0.00635`。
- shuffled/reversed的effective-LoRA RMS相对correct中位比约
  `0.988/0.964`，并没有把adapter大幅缩回identity。因此其行为改善来自
  LoRA内容/方向变化，不是简单“关掉视频adapter”。
- 综合判断：v4解决了“是否使用视频”的机制问题，也体现了same-task内部与
  行为变化较小的层级；但它没有解决“是否正确使用视频”。正确时序没有获得
  行为优待，反而在旧版损失最集中的object精确任务上被shuffle/reverse改善，
  因而完整视频特异性硬门明确失败。
- 当前最符合证据但尚未因果证明的解释是：v4忠实保留了任务阶段的同时，也把
  不准确forecast、视角/状态估计误差或demo-specific低层变化映射成策略更新；
  coherent normal-order分量在精确接近和抓取任务上可能有害，扰乱它以后稳定
  task/object信息重新占优。same-task只净降5说明问题并非对所有具体轨迹差异
  普遍过敏，更可能集中在“如何解释连贯时序”这一映射。另需注意shuffle/
  reverse会同时改变初始anchor与local transitions，并不是纯粹删除顺序。
- 按owner停止条件，本轮不修改架构、不重训、不推进RL。完整结果留给后续及
  外部专家复核。

## v4 step825固定首帧shuffle归因（2026-07-26）

- 为隔离full-shuffle是否只是通过更换initial anchor获益，新增唯一条件
  `shuffled_keep_first`：先生成与原shuffled逐元素相同的permutation，再把
  原始frame 0移到首位；所有非零frames保持原full-shuffle相对次序。原始
  frame indices不重排，teacher/demo/language/init/env/policy/Writer seeds
  与已有arms严格配对。
- 固定400结果为`136/400`，output位于
  `/data/ymdai/outputs/ember/pi05_action_forecast_v4_as_formal_val8x50_step0825_shuffled_keep_first_6b5923f_g0123_gen1_b100_roll6_20260726`；
  `results.json` SHA256为
  `0ec198d1438bdb85d9eccb41ac5f6796a470903b963576f29260c048b453ac99`。
  32/32 shards、400 rows、24 workers均一次成功，无重试或错误。
- 逐任务success为Long-1/Long-2/Goal-3/Goal-6/Object-1/Object-3/
  Spatial-1/Spatial-3=`9/1/0/45/45/26/1/9`。
- 相对correct `109`，paired both/correct-only/keep-only/both-fail为
  `91/18/45/246`，净`+27`、churn`63/400`、exact McNemar
  `p=8.98e-4`。Object-1 `38→45`、Object-3 `11→26`、Goal-6
  `40→45`共同贡献提升；保留正确首帧并不能消除shuffle收益。
- 相对full-shuffle `148`，paired为`116/32/20/232`，净`-12`、
  churn`52/400`、`p=0.126`。这12次总差异几乎全部来自Object-3
  `37→26`；该task的full-only/keep-only为`14/3`、`p=0.0127`。
- 14个full-shuffle permutation本身以frame 0开头的episodes在两条件间LoRA
  SHA完全相同，且success逐条一致，提供了实现正确性的真实执行校验。
- 结论：随机anchor确实可能额外帮助Object-3，但它不是`109→148`异常改善的
  必要条件或主要解释。两项干预可能非线性交互，不能把39严格做因果加法分解；
  直接观测上，恢复原始anchor只把148降到136，而固定anchor后仍相对correct
  净增27。主嫌疑进一步集中到后续帧顺序、相邻transition和forecast/Temporal
  对正确连贯时序的解释；本实验尚不能在这些机制之间继续细分。

## v4外部专家咨询的科学问题边界（2026-07-26）

- 当前需要解释的不是“Writer是否读取视频”：v4内部same/shuffled/reversed/
  wrong差异已经穿过完整路径到effective LoRA。真正失败的是这些差异的行为
  语义：correct没有被优待，shuffled显著改善，wrong也未形成稳定伤害。
- 外部分析应同时考虑forecast语义未校准、positive task-level AS目标不可辨识、
  coherent temporal分量被错误映射、Revision强度被错误使用，以及静态task
  LoRA是否必须依赖帧序等互相可区分的解释。不能把任一项提前写成既定根因。
- 下一步优先寻求少量、低成本、能区分候选解释的诊断；不以contrast/order
  loss强行制造结果，不读取forbidden teacher action/proprio/reward，也不因
  当前负结果自动增加模型容量或启动RL。
- 自包含的远程咨询材料为
  `docs/action_forecast_writer_expert_consultation.md`。它嵌入理解现状所需的
  架构、逐任务结果、paired统计和内部量，不依赖本地主机output目录。

## v4 forecast-order第一轮因果移植与v5候选（已被下一节覆盖，2026-07-26）

### Forecast之前还是之后

- 在step825的16条validation reference videos上构造四臂：
  `N→N`、`N→S`、`S→N`、`S→S`。`N→S`不重新计算visual-state或Action
  Expert，只把normal-context per-image forecasts按原shuffled permutation放入
  shuffled slots；`S→N`则按image identity把shuffled-context forecasts放回
  normal slots。
- `N→S`与真实`S→S`的effective-LoRA delta几乎相同：cosine/magnitude/
  residual中位数为`0.999978/0.99905/0.00712`；在真实π0.5 action function中
  为`0.999864/1.00020/0.01668`。
- `S→N`相对`N→N`的effective LoRA relative L2中位数只有`0.00337`，
  action chunk relative L2中位数只有`0.00156`。因此按image identity对齐后，
  shuffled context并没有显著改变每张图的forecast；主要异常是同一批forecast
  被放入不同time slots后由downstream构造出来的。
- Object-1/Object-3各50个固定states的新增定向rollout为：

  | condition | Object-1 | Object-3 | total |
  |---|---:|---:|---:|
  | correct / `N→N` | 38 | 11 | 49 |
  | `S→N` | 36 | 11 | 47 |
  | `N→S` | 43 | 29 | 72 |
  | true shuffled / `S→S` | 45 | 37 | 82 |

  `N→S`相对correct净`+23`、`p=4.31e-4`；`S→N`相对correct净`-2`、
  `p=0.625`。真实shuffled context在`N→S`之上仍净`+10`、`p=0.0129`，
  尤其来自Object-3，说明存在较小非线性交互，但上游visual-state不是主因或
  必要条件。

### Plan/Revision中究竟哪一项介导行为

- 固定normal-context forecasts，只交换Plan `P`、Revision direction `D`、
  Revision value strength `V`和Temporal Q/K strength routing `R`。
- strength并非无界OOD：train-normal median/p95/max为
  `0.333/0.614/0.858`，shuffled-slot为`0.454/0.670/<0.858`；没有token超过
  train-normal max。只换routing的action target-delta magnitude约`0.011`，
  可忽略。
- Object rollout得到：

  | arm | Object-1 | Object-3 | total |
  |---|---:|---:|---:|
  | correct | 38 | 11 | 49 |
  | Plan-only | 37 | 24 | 61 |
  | value-strength-only | 39 | 15 | 54 |
  | direction-only | 39 | 28 | 67 |
  | full shuffled Revision | 45 | 30 | 75 |
  | all shuffled-slot downstream | 43 | 29 | 72 |

  direction-only相对correct净`+18`、`p=0.00143`；strength-only仅净`+5`、
  `p=0.359`；完整Revision净`+26`、`p=6.16e-6`。Revision direction是主要
  行为中介，strength只在与direction组合时有次级协同，Plan有次级贡献。

### 它怎样改变policy

- 五条成功Object轨迹上选取approach、pre-grasp、gripper-close、transport和
  terminal共25个current observations；使用相同language、observation和flow
  noise比较12个LoRA反事实。
- `N→S`相对normal的executed-action delta主要位于end-effector translation：
  五阶段translation RMS中位数`0.0197–0.0341`，rotation
  `0.00217–0.00770`，gripper`0.00155–0.00459`。direction-only在
  pre-grasp/close/transport translation上的target-delta cosine为
  `0.925/0.954/0.982`。
- 这解释了Object收益：错误Revision direction成为一条训练分布外、但恰好改善
  approach/transport的action-shaped controller code。它不是“shuffled视频被
  理解得更好”，也不是参数差异落在policy null space。
- 直接把Revision置零会造成目标`N→S` delta的`2.1–5.8×`动作变化，且多阶段
  低相关或反向；现有Plan/Revision已共同适配，不能通过删除一支热修。

### 第一轮根因判断与当时的下一版候选

- v4把第`i`帧的local chunk `A_i[lead]`按`u=t_i+lead`投到共享robot
  absolute-time轴，并假设不同teacher frames在同一`u`覆盖的actions是同一未来
  控制时刻的可比预测。独立teacher/action episodes、observer/human viewpoint、
  不同速度和缺少intermediate supervision都没有识别这条对应关系。
- 所以Revision并不是校准的“多次future forecast一致性”，而是对未经证明的
  lead-position配对求residual。shuffle主要重排这些配对；其direction偶然形成
  更好的Object translation control。
- 第一轮当时提出不优先重做visual-state、不加深两层Temporal、不裁剪strength、不恢复
  decoder静态旁路，也不加contrast/order loss。v5删除absolute-time
  Plan/Revision/Belief，改为：

  \[
  I_i=W_2\,\mathrm{GELU}(W_1\,\mathrm{vec}(A_i))\in\mathbb R^{256},
  \qquad
  \Delta I_i=I_i-I_{i-1}.
  \]

  Temporal顺序为`I_0, ΔI_1, I_1, ..., ΔI_(T-1), I_(T-1)`。它只声称teacher
  视频推进时frame-local execution intent发生有向变化，不再声称不同frames拥有
  共同robot action clock。
- 32-token visual-state、anchor/local reader、VL/Action Meta-LoRA、共同flow
  noise、frozen source base、rank-16 LoRA schema、两层content-only Temporal和
  content-only decoder均保留。参数仍以rank128 Source-SFT的`10,297,344`为
  comparator，差异控制在`0.1%`内。
- 该结构只修复已证明的错误对应关系，不保证positive task-level AS已足以识别
  高层视频语义。若未来75-step仍inversion，下一原则性候选是action-hidden、
  positive-only causal future frozen-visual-feature prediction，而不是继续堆
  downstream补丁。
- 该判断随后被下一节更全面的hidden-semantics、visual-state和translation-only
  证据覆盖；根因复审现已改名为
  `docs/action_forecast_writer_v4_root_cause.md`，活动架构另见
  `docs/action_forecast_writer_v5_design.md`。

## v4全面根因复审：覆盖“absolute-time是唯一主因”的结论（2026-07-26）

- 上一节正确定位了absolute-time Plan/Revision是直接行为放大器，但过早排除了
  visual-state并拍板v5。新增证据证明完整因果链还包括更上游的AS可识别性不足、
  visual-state旁路和Meta forecast语义漂移；Intent+Transition现只保留为局部
  候选。
- 24个development-train tasks×4 demos的post-inference隐藏语义审计显示，
  step75/300/825的latest/earlier forecast MSE ratio为
  `0.966/1.043/1.087`，latest-better pair fraction为
  `0.509/0.419/0.404`，residual到真实误差修正的cosine为
  `0.335/0.254/0.238`。同期trailing-25 AS loss却从
  `0.1339→0.1201→0.1065`持续下降，直接证明functional AS优化与forecast
  语义校准可以反向变化。
- source-base-like forecast的latest ratio/latest-better/correction cosine为
  `0.986/0.543/0.339`。step825关闭VL Meta、Action Meta或两者时，隐藏
  teacher-future MSE比完整trained path低约`5.9%/9.6%/11.0%`。两个Meta-LoRA
  在task-level AS下逐渐改写了原本弱但较合理的source forecast ordering。
- step825 neutral visual-state只使forecast相对变化约`0.855%`、cosine约
  `0.999963`。visual coordinates与同demo teacher-action distance的相关由
  step75 `0.324`降到step825 `0.107`；最终progress probe \(R^2=0.548\)，
  state/action仅`0.046/0.038`。32-token路径没有成为机器人visual state或必要
  信息瓶颈。
- 与此相反，full/frame-main/translation/interaction forecast对同demo真实低层
  动作差异的相关由step75约`0.144/0.327/0.503/0.336`升到step825约
  `0.587/0.652/0.695/0.740`。v4确实看到了视频，但主要通过raw-image/Meta
  旁路提取了具体轨迹、phase和translation，而不是同task多demo共享的高层逻辑。
- 64 references×8独立random permutations的shuffle LoRA delta两两cosine
  中位数为`0.941`，mean-delta保留individual norm的`0.966`；既有Object-1/3
  400-panel delta也有约`0.868`跨video共识。0/31 shuffled adapters向correct
  task consensus靠近，排除“shuffle去除demo噪声、回归任务均值”和
  permutation lottery。
- forecast分量移植把异常进一步定位到frame phase的前三维translation：
  translation-only adapter对完整shuffle delta cosine为`0.99747`；
  rotation/gripper-only影响很小。Object-1/Object-3定向rollout为：

  | arm | Object-1 | Object-3 | total |
  |---|---:|---:|---:|
  | correct | 38 | 11 | 49 |
  | no VL Meta | 40 | 8 | 48 |
  | no Action Meta | 35 | 15 | 50 |
  | lead-only | 31 | 9 | 40 |
  | shuffle frame-main only | 41 | 31 | 72 |
  | shuffle translation only | 44 | 35 | 79 |
  | true shuffled | 45 | 37 | 82 |

  translation-only相对correct净`+30`、`p=5.30e-6`；相对true shuffled只差3、
  `p=0.607`。它几乎闭合了完整异常，不需要better visual context、rotation、
  gripper或真正的frame×lead revision。
- normal/true-shuffle AS loss为`0.133615/0.135925`；Object-1/3 loss只略增
  `0.000414/0.000156`，shuffle LoRA delta与negative local AS gradient的
  cosine约`0.00164`。所以扰动机制是稳定、确定的，但它对closed-loop success
  的正号没有被AS objective识别；精细抓取阈值把近似objective-flat的
  translation controller变化放大成`49→82`。
- 当前完整根因是：
  `positive task-level AS不可识别demo过程 → visual-state非瓶颈 →
  Meta学习低层phase/translation latent → absolute-time Plan/Revision放大`。
  两层Temporal和content-only decoder忠实传递差异，当前没有重写证据。
- 对Object-1/Object-3全部成败翻转做exact replay后，完整400中shuffle收益的
  主要行为来源也已定位。两任务有40条`correct fail→shuffle success`、7条
  反向翻转，净增33；Object-3的31条正向翻转中，correct有23条明确去抓并常常
  抬起深绿色干扰瓶，7条到达红橙色BBQ sauce后抓取/运输失败，1条为多物体
  碰撞。correct/shuffled首次闭合点配对距离中位数为`0.1119 m`，但前60-step
  抬升中位数仍为`0.2165/0.2316 m`，说明主问题是空间目标绑定而非correct不会
  执行动作。
- Object-1的9条正向翻转中，首次闭合step中位数由correct `122`提前到shuffle
  `91`，correct `9/9`在闭合后60 steps抬升不足`0.10 m`；反向翻转则显示
  shuffle同样会毁掉原本有用的抓取控制。Object-3正向翻转跨22个teacher demos，
  demo `14/30/32/43`的同一cached LoRA在不同init上出现相反翻转，排除少数坏
  视频。
- 因此更准确的解释不是shuffle“腾出LoRA容量”，而是correct-order路径把
  demo低层phase/translation写成不可迁移的静态controller bias，压过已有
  language/object semantics；shuffle结构化破坏该bias，使既有高层信号重新
  主导。lead-only更差且shuffled adapter不向task consensus收缩，说明它没有
  新产生更多高层信息。
- 旧v5决定已撤回。下一版必须先解决visual-state必要性、Meta职责、
  train-only forecast语义gate和same-task多demo高层汇聚；不通过
  contrast/order loss强制制造差距。完整证据和SHA见
  `docs/action_forecast_writer_v4_root_cause.md`。

## Semantic Core + Causal Procedure v5最终设计（2026-07-26）

- owner已批准
  `docs/action_forecast_writer_v5_design.md`为唯一活动Writer架构。v4完整代码、
  结果和根因只作provenance；v5现已原位实现并完成机械profile，尚未产生正式
  AS性能或特异性结果。
- v5 teacher侧删除state。每帧图像与正确语言经过frozen PaliGemma和trainable
  rank4 VL Meta-LoRA后，取256个language-conditioned image-position final
  hidden；固定`2×2`空间平均池化为64 tokens，再经bias-free `2048→256`得到
  Semantic Core。Core没有frame ordinal、RoPE、causal mask或adjacency，因此同
  一帧集合的shuffle只置换K/V行，Core compiler输出结构性不变。
- v5保留native fixed 50-token Gaussian suffix、固定`t=1`和rank8 Action
  Meta-LoRA，但只执行一次Action Expert forward并读取action-out之前的
  `[50,1024]` final hidden；不做10-step denoise、不输出7D action、不再构造
  forecast/absolute-time/Plan/Revision/Belief。50个positions均值后用
  bias-free `1024→256`得到每帧robot-semantic interaction token。
- 全部per-frame tokens进入两层、width256、8-head、global causal、ordinal
  RoPE Transformer，保留可变长度`[T,256]` Procedure，不压成固定event数。
  Core-to-LoRA Compiler先从`[64T,256]`集合形成稳定的320-slot动态content；
  Procedure-to-LoRA Refiner再产生独立delta，cross-attention output zero-init，
  最终`Z=Z_C+D`。静态module/layer/rank identities只进Q/K，factor heads不能
  读取。
- 8个bias-free factor heads为`256→420→target_width`且final zero-init，public
  输出保持38 targets、76 tensors、rank16、`1,287,168` scalars。机械设计预算
  为`10,301,440` trainable parameters，比rank128 Source-SFT
  `10,297,344`多`4,096`（`0.0398%`）；真实实现必须重新打印核验。
- 训练初版曾固定`N=4`：每条action query独立抽4条同task不同teacher videos，
  逻辑上生成`B_a×4`个LoRA和functional losses并普通求均值；仅允许按精确视频
  键去重相同Writer forward，不能让整个action batch只共享4个LoRA。推理仍严格
  one-shot。
- 真实构造打印trainable参数恰为`10,301,440`；全套`187 passed`。Core compiler
  对同一memory置换数值不变，causal Procedure具备前缀因果性和顺序差异，
  zero memory不能由routing/ordinal凭空生成content，factor-head zero-init使
  fresh Writer严格输出identity LoRA，固定suffix buffer进入checkpoint。
- GPU4–7实测最终选择`B_a=8`、`N=4`、frame microbatch32。12-step run从step2
  真实exact-resume到step12；稳态11步wall中位/均值/范围为
  `61.39/59.78/38.99–92.08s`，pairs/s中位`2.085`，峰值allocated/reserved
  `60.32/67.47GB`。`B_a=12/20`和`m40/B8`均发生约80GB reserved跳变，只剩
  不足3GB，故即使未OOM也不具备正式稳定余量。
- profile边界明细：`m32/B1`第二步`2.079 pairs/s、64.13GB reserved`；
  `m32/B4`第二步`2.186、64.05GB`；`m32/B8`第二步`2.717、67.24GB`；
  `m32/B12`首步`1.557、81.98GB`；`m32/B20`首步`2.028、83.82GB`；
  `m40/B8`首步中实时已达约`80.7GB`，因稳定余量不足主动停止。只有最终B8
  root作为canonical profile evidence保留，其余可再生边界roots删除。
- v5不继承v4 step等价口径。正式封存为每60 steps约一小时
  exact-resume segment，并在10/20/30/40/50/60均匀保存6个checkpoint。
  focused AS/RL无总wall-clock上限；
  best后只有明显、持续、多task且独立复测成立的下降才可停止。
- 特异性先内后外：Core对same-frame-set order变换应不变，Procedure与delta应
  明显有向变化，same-task other变化应小于wrong，差异须穿过effective LoRA和
  policy function；随后固定400 rollout要求correct明显优于wrong/shuffled/
  reversed且same-task other影响最小。absolute performance最低目标为达到或
  接近`125/400`，目标逼近v4 shuffled `148/400`。不通过则定位最早失败模块后
  fresh迭代，不使用contrast/order loss。

## v5 AS step40→120内部与闭环特异性（2026-07-27）

- canonical formal root为
  `/data/ymdai/outputs/ember/pi05_as_writer_core_causal_v5_dev_seed7_ce0b568_20260726`，
  训练commit为`ce0b568c9ad5ed6ab783924209bbfe02fe601d7b`，contract SHA256为
  `39379e3afd9a512d87b1f0638fc3bdfa2afb09238a8c6c54a8aa0ec94355a981`。
  step10/20/30/40/50/60/70/80/90/100/110/120固定512-row functional
  validation loss依次为
  `0.140365/0.143320/0.137210/0.136874/0.137584/0.140980/0.138384/`
  `0.138306/0.138544/0.137017/0.137679/0.139036`。functional observed-best
  暂为step40，但40→60的回落随后反弹，至120仍没有满足停止条件的明显、持续、
  多task峰后下降。
- step40先完成16-reference内部反事实和五个fixed-400 rollout。内部已有真实
  Procedure顺序路径，但有效LoRA的shuffle/reverse中位相对L2仅
  `0.156%/0.204%`；固定correct Core、只替换Procedure时为
  `0.147%/0.196%`，Core-only BF16排列伪差为`0.085%/0.089%`。闭环
  correct/same-task-other/cross-suite-wrong/shuffled/reversed分别为
  `45/52/52/51/51`，与source base `48/400`实质相同，行为特异性不通过。
- step10→40→60→120的same/wrong有效LoRA中位差异为
  `0.419/2.519% → 0.316/2.101% → 0.391/3.961% → 1.235/15.963%`；
  same/wrong比值由`0.166`降到`0.077`。同期fixed-Core
  Procedure-only的shuffle/reverse差异为
  `0.359/0.634% → 0.147/0.196% → 0.192/0.346% → 0.626/1.087%`，
  而Core-only伪差持续降到`0.073%/0.074%`。因此step120不是再次坍缩成
  task latent；task语义和有向Procedure都在继续学习。
- step120逐层中位相对L2显示顺序信息路径清楚：
  shuffle/reverse的Core set约为`5e-6%/5e-6%`，Core content为
  `0.151%/0.154%`，interaction sequence为`60.99%/82.67%`，
  Procedure sequence为`29.24%/46.00%`，Procedure delta为
  `1.954%/4.391%`，final content为`0.691%/1.285%`，effective LoRA为
  `0.626%/1.087%`。8个validation tasks的effective-LoRA task median范围为
  shuffle `0.371%–1.317%`、reverse `0.427%–2.201%`；不是单task异常。
- step120五个同一fixed-400 panel的闭环结果为：

  | condition | total | per-task（Long1/2, Goal3/6, Object1/3, Spatial1/3） |
  |---|---:|---|
  | correct | 65 | `7/0, 1/29, 22/0, 6/0` |
  | same-task other | 59 | `6/0, 0/29, 20/0, 4/0` |
  | cross-suite wrong | 57 | `3/1, 1/33, 14/0, 5/0` |
  | shuffled | 61 | `10/0, 0/29, 21/0, 1/0` |
  | reversed | 65 | `7/0, 1/33, 20/0, 4/0` |

  相对correct，same/wrong/shuffle/reverse的`correct-only/other-only`为
  `20/14`、`27/19`、`19/15`、`13/13`，双侧exact p分别为
  `0.392/0.302/0.608/1.0`。correct方向上高于same/wrong/shuffle，但没有显著，
  且与reverse持平，故行为特异性硬门仍未通过。
- step120 correct相对step40为`41`条new-only、`21`条old-only，净`+20`，
  exact `p=0.0151`；逐task净变化为`+7/0,+1/+5,+2/0,+5/0`，来自多个
  suites/tasks。这证明继续训练产生了真实闭环提升，而不是只让内部向量差异
  变大。该证据说明旧轨迹仍在学习，不能证明结构已失败。随后旧合同从step120
  尝试续训，但owner在optimizer step128停止；没有step120之后的原子checkpoint，
  因此科学证据止于step120，不能写成已恢复到step180。

## v5共享四视频训练估计器决策（2026-07-27）

- 旧`每action独立4 videos`代码在`B_a=8`时每rank逻辑上请求32套LoRA；虽然
  精确demo碰撞可去重，step126–128每rank仍实际生成24–25套， sampled frames
  为`537–799`。`max 32 frames/call`意味着约17–25次frame encoder调用，再加
  4次functional policy forward；这解释了`61.39s/step`，不是“模型只慢一点”。
- target train videos在stride5并保留末帧后，1,200条的采样帧数
  min/P50/P75/P90/P95/max/mean为
  `16/30/40/57/70/105/35.60`。因此即使只保留4条video，平均也约142帧，
  不能安全硬塞进一次PaliGemma；保留`max_frames_per_encoder_call=32`作为
  纯显存安全分块，但末块不再重复padding，也不把该上限作为科学搜索变量。
- owner拍板的新估计器为：每rank每step一个task；为该task visit抽4条不同
  teacher videos，只生成4套one-shot LoRA；该rank的`B_a`条独立action queries
  均匀分给4套LoRA，每条action只对应一条video，形成`B_a`个等权loss。
  4 ranks每步覆盖4个task，全局schedule每6步覆盖24 tasks，下一visit换新
  video set。
- 在video/action同task且独立采样、round-robin分组时，
  \(\mathbb E[\frac1B\sum_i\ell(A_i,V_{i\bmod4})]\)
  与独立pair采样具有相同population objective。变化的是batch covariance：
  4条video在同一步共同提供梯度；每套video LoRA又被多个
  action初态/轨迹约束，降低直接模仿单条teacher路径的价值。
- 这同时实现了此前两个目标，而不引入对比/order loss：跨4条teacher提取共同
  高层语义，以及让每个one-shot LoRA对同task action distribution通用。推理仍
  是一条video生成一套LoRA。
- 新合同与旧step120采样/optimizer合同不同，必须fresh identity和新root；
  profile只调`B_a`。先以约一小时首轮训练的fixed-400 absolute performance
  判断：至少约`110–120/400`；达到后才做内部与五条件rollout特异性，目标最终
  逼近或超过v4 shuffled `148/400`。
- 新合同GPU4–7真实profile选择`B_a=16`。step2→12 exact-resume后，11个稳态
  steps的wall中位/均值/范围为
  `10.347/10.043/7.072–14.341s`，全局有效pairs/s中位`6.185`。每step严格
  64个全局policy samples、16个Writer video conditions、1次policy forward；
  峰值allocated/reserved为`63,736,767,488/68,415,389,696 bytes`。B20虽跑完
  3步但reserved `83,732,987,904 bytes`、仅余约1.3GiB；B24/B32首步OOM。
  因此正式首段为fresh step0→400，每50步保存，预计约67–69分钟。

## v5单视频完整action-batch估计器（2026-07-27）

- owner最终选择更简单的训练抽样：每rank每step一个task、1条teacher video、
  1套LoRA，完整rank-local action batch都监督该LoRA；task后续被访问时换video。
  同一LoRA必须解释宽action分布，跨video的共同信息则由共享Writer跨step SGD
  累积。共享四视频分组合同因此退役，不能按其step400计划继续。
- canonical实现删除四视频schedule、round-robin action映射、batched
  per-sample LoRA executor和对应兼容逻辑；普通functional LoRA forward只执行
  一次。source路径净删除多于新增，不保留平行runner。
- `max_frames_per_encoder_call`不是optimizer microbatch。历史Action-Memory
  已使用frame microbatch16；当前最长真实视频为task38/demo36，原始517帧、
  stride5后105帧。固定`B_a=1`时，F32一步`5.93s`；F105占到`79,873 MiB`且
  超过90秒未完成，证明帧安全分块必须保留。
- GPU4–7联合profile选择`F32/B20`。最长视频步`6.956s`，随后常规步
  `3.109/3.527s`，峰值allocated/reserved为
  `76,937,901,056/83,630,227,456 bytes`。B24在F32与F24下都OOM；
  F40/B20没有净吞吐收益。owner允许最长视频只保留少量空间，并停止B21搜索。
- profile artifact为
  `/data/ymdai/outputs/ember/pi05_as_writer_v5_jointprofile_f32_b20_long105_20260727`；
  run-contract/metrics/checkpoint-manifest SHA256分别为
  `e83dd24f...ccb1fc`、`0b39de73...9b561`、`993739d4...0cb1b`。
- 正式首段封存为fresh step0→900、每100步保存；按常规约3–4秒/step折算约
  一小时。首段后先做fixed validation绝对性能选择，达到门槛后再做内部和
  rollout特异性。

## v5单视频正式首段启动证据（2026-07-27）

- canonical实现commit
  `0b4e00696113cf6601d6e63b4c73734f3cea1073`已push；formal launch前
  worktree clean且`HEAD==origin/main`。run只见物理GPU4–7，source policy
  trainable参数为0，Writer trainable参数为`10,301,440`。
- start-event contract SHA256为
  `03186c57ac736ac82398400676ff10c33eb46ab3e5f9bcbbe44064305944787c`。
  step1–400全部finite；step100/200/300/400原子checkpoint完整包含Writer、
  trainer、四rank state与manifest。step400累计32,000 unique action queries，
  训练body为`1,534.14s`。
- step400覆盖24/24 tasks；每task有1,320–1,340 action examples、66–67次
  video visits且全部50条unique videos均已覆盖。实际metrics确认每步全局80
  actions、4 videos/LoRAs、4 task conditions和一次policy forward，不存在
  Cartesian重复或optimizer accumulation。
- 常驻模型的预封存512-row validation functional loss在
  step100/200/300/400为
  `0.1360107/0.1349113/0.1332633/0.1324333`，optimizer updates与parameter
  gradients都为0，test action reads为0。它只能安排closed-loop候选顺序，
  不能代替fixed-400 success选择。
- output、tmux、精确launch和跨session接手顺序见
  `docs/active_session_handoff.md`；该step400只是运行中快照，不作性能结论。

## v5单视频首段封存、long-first评估与step900轻量特异性（2026-07-27）

- fresh正式run已正常完成step0→900，训练body wall为`3,485.15s`；900步累计
  `72,000`个action queries与`3,600`个one-video Writer conditions。step100至
  900的9个checkpoint均有Writer、optimizer/scheduler、sampler/data cursor、
  四rank RNG与atomic manifest；step900的24个train tasks各有`3,000` examples、
  `150`次video visits、50条unique videos和50条action episodes。
- resident 512-row validation functional loss在step100/200/300/400/500/600/
  700/800/900依次为
  `0.136011/0.134911/0.133263/0.132433/0.134811/0.135303/0.132294/`
  `0.132574/0.137075`。它和closed-loop不单调，不能代替rollout选择。
- fixed400 correct-video代表点step100/400/700/800/900为
  `62/64/92/76/103`；step900是首段observed-best。step700→900的paired净提升
  为`+11`但exact `p=0.207`，step800→900净提升`+27`、exact `p=0.00155`；
  step900仍低于约`110–120/400`预门，但没有出现可停止的持续峰后下降。
- canonical evaluator commit `3b6d9d1`把最长horizon states按
  `physical_gpu_count × replicas_per_gpu` worker slots切分并赋GPU affinity；
  worker只有在没有unclaimed long shard时才领取普通task。step800四卡、
  每卡6 replicas的真实run先领取全部48个long shards，再动态处理24个普通
  shards，400 rollouts用`921.60s`、`0.4340 rollouts/s`，是首轮单卡约
  `2.66×`吞吐；这一规则不依赖一个checkpoint分到几张卡。
- step900内部16-reference反事实显示顺序机制没有坍缩。shuffle/reverse下
  Core-set相对L2中位仅约`6.2e-8/1.93e-4`，Procedure sequence为
  `63.83%/75.21%`，Procedure delta为`11.16%/16.36%`，effective LoRA为
  `3.68%/5.74%`，policy action为`0.93%/1.29%`；固定Core仅替换Procedure时
  effective LoRA为`3.689%/5.764%`，比旧step120约强`5–6×`。
- 轻量行为screen只取同一8 tasks的init-state `0–9`。correct复用既有step900
  full400的对应80行，另四臂各跑80行；correct/same-task-other/wrong/shuffled/
  reversed为`21/25/14/23/23`，逐task
  （Long1/2, Goal3/6, Object1/3, Spatial1/3）分别为
  `0/0,0/8,8/5,0/0`、
  `0/0,0/10,8/5,1/1`、
  `1/0,0/3,5/3,0/2`、
  `0/0,0/10,7/6,0/0`、
  `1/0,0/8,8/6,0/0`。
- 相对correct，same/wrong/shuffled/reversed的
  `correct-only/other-only`为`3/7、12/5、2/4、2/4`，双侧exact p为
  `0.344/0.143/0.688/0.688`。80/80 episodes的env seed、language、teacher
  selection/order seed与policy-noise共同前缀均配对；完整noise列表只因成功
  即终止而长度不同。该screen给出correct优于wrong的方向，但没有顺序臂优势；
  小样本只用于判断机制仍值得训练，不能替代五臂full400硬门。

## v5单视频step900→1800 exact-resume合同（2026-07-27）

- 续训复用同一formal root与launch contract
  `03186c57ac736ac82398400676ff10c33eb46ab3e5f9bcbbe44064305944787c`；
  resume checkpoint为`step_00000900`，manifest SHA256为
  `157599fc60e565570be7d711362469b7233606635437f6e219125e7e36f7b8e2`，
  canonical payload为
  `c838ccba4b1125753e8f11f1ddfaf25c9d6672fe9e1601d3dbb8d7f816b375e5`。
- 科学合同不变：4-rank DDP只见物理GPU4–7、F32/B20、每step 80 action
  samples/4 one-video conditions、每100步checkpoint、同一source base、
  target HDF5、normalization、optimizer/scheduler、sampler/video schedule和
  RNG。新增900步预计再产生`72,000` samples、`3,600` video conditions、
  9个约121MB checkpoint，峰值新增存储约`1.2GB`。
- 当前main相对训练commit只改了canonical evaluator的long-first调度，训练入口、
  `src/ember/writer/`与v5 config逐文件Git diff均为空。续训使用显式
  `--allow-contract-compatible-code-resume`，由run-contract reconciliation
  要求除recorded Git commit外所有科学字段逐项完全相同；任何其他漂移均
  fail-close。
- resume已从clean、pushed commit
  `db2a6905cc3d7433333d4c95d08345180c9b4fc2`在tmux
  `ember-v5-as-sv1800`启动。start event确认原contract SHA、`resume_step=900`、
  `stop_after_step=1800`与source policy trainable参数0；invocation明确记录
  `contract_compatible_code_resume=true`。step900 resident validation复算与
  原值完全一致，step901起metrics连续追加，初始常规step约`3–4s`。

## v5 observed-best与正式特异性结论（2026-07-27）

- 正式时间轴已正常完成step0→1800。fixed400 correct在step100/400/700/800/
  900/1000/1400/1700/1800依次为
  `62/64/92/76/103/115/115/71/86`。step1000与1400并列observed-best；
  step1400 online functional loss更低且时间更晚，故选它做唯一正式特异性。
  step1700/1800的强下降说明训练已跨过高点，不存在“只要再训一段就可能自然
  解决顺序”的充分依据。
- step1400内部16-reference五条件中位relative L2：
  - same：Core set `4.423%`、Procedure `19.052%`、effective LoRA `7.304%`、
    action `1.394%`；
  - wrong：`44.732%/111.729%/72.638%/16.987%`；
  - shuffled：约`0/64.299%/2.928%/0.486%`；
  - reversed：`0.0157%/72.560%/4.773%/0.752%`。
  fixed-Core Procedure-only的shuffle/reverse effective LoRA仍有
  `2.921%/4.767%`，证明上游Procedure没有忘掉顺序。与step900相比，
  Procedure顺序差基本稳定，而LoRA/action差反而缩小；最早明确失效层是
  Procedure到最终slots/factors的融合/编译。
- fixed400五臂为`correct/same/wrong/shuffled/reversed =
  115/108/74/113/114`，95% Wilson区间分别为
  `24.53–33.37/22.88–31.55/15.00–22.60/24.06–32.85/24.30–33.11%`。
  400个episode的task/state、language/env seed、policy noise共同前缀、
  checkpoint和pairing hash全部严格配对。
- 相对correct：
  - same为both/correct-only/other-only/both-fail=`92/23/16/269`，
    净`+7`、churn `9.75%`、exact McNemar `p=0.337`；
  - wrong为`57/58/17/268`，净`+41`、churn `18.75%`、
    `p=2.18e-6`；
  - shuffled为`101/14/12/273`，净`+2`、churn `6.50%`、
    `p=0.845`；
  - reversed为`103/12/11/274`，净`+1`、churn `5.75%`、`p=1.0`。
- wrong净差主要来自Object-1 `+16`和Object-3 `+21`，但correct-only flips跨
  33条teacher demos，说明不是少数坏视频；它证明视频内容有因果语义性，
  尚不能证明跨全部tasks普适。顺序臂逐task仅`-1..+3`互相抵消；19个
  correct-fail episode至少在一个假顺序臂成功，correct同时优于wrong/shuffle/
  reverse的episode只有3个。结论是same鲁棒性方向可接受、wrong-video门通过，
  order门明确失败，v5 overall AS gate失败。
- 四个counterfactual各独占物理GPU4/5/6/7，每卡6 rollout workers、3 Writer
  generators，36 shards按long-first调度；全部400/400、return code全0、
  无OOM/traceback，wall `2201.9–2255.1s`。results SHA256依次为
  correct `cc0ea739...67c2`、same `e5b9705e...88a0`、wrong
  `2e8b54ab...00c6`、shuffled `514b6647...977`、reversed
  `2f75bc7b...076a`。

## v5.1单一路径的证据依据（2026-07-27）

- v5结果触发owner的条件授权：
  `docs/action_forecast_writer_v5_1_proposal.md`由候选提升为下一唯一focused
  架构。它不是因absolute略低而扩容，而是针对最早失效层重新分配预算：
  factor hidden从420降到240，把容量放到task-token语义表征、language-axis
  Core和slot-normalized fusion。
- v5.1保留已经证明有效的Action Expert causal Procedure，改用text-only
  contextual task queries与multimodal task-token evidence形成
  permutation-invariant Semantic Core；Procedure内容先按时间中心化，再通过
  zero-init AdaLN调制Core slots，最后只过一个post-fusion slot block。
  机械预算`10,244,872`，比rank-128 Source-SFT少`52,472`。
- 首段训练尺度定义为约一小时wall-clock而非固定optimizer step。v5.1必须先
  重新profile显存与吞吐；第二/第三段都要由上一段的早期特异性、absolute与
  曲线证据单独批准，不能把“未看到充分峰后下降”机械翻译成自动续训。

## v5.1实现核验结论（2026-07-27）

- 严格task-token对齐不需要第二次tokenization：完整teacher prompt只做一次
  SentencePiece immutable-proto编码，按piece字符区间提取task span；真实
  PaliGemma tokenizer上两条不同长度LIBERO语言分别得到`L=18/7`，选中IDs
  decode后逐字等于cleaned task。Text分支复用这些IDs并只加BOS。
- 新Core的唯一value路径是每帧最后层final-norm multimodal task-token hidden；
  text-only final hidden只作跨frame query。无frame ordinal的token-aligned
  attention在CPU置换测试中保持数值一致；Procedure的causal prefix测试通过，
  reverse后非同值。
- 参数重新分配严格落在计划表：
  Text/VL/Action Meta-LoRA=`921,600/921,600/626,688`，
  shared language projection/frame attention/Core blocks=
  `524,288/262,664/1,573,888`，interaction projection/Procedure=
  `262,144/1,573,888`，fusion/factor heads=`1,535,232/2,042,880`，
  总计`10,244,872`。
- 梯度阶段与设计一致：fresh时只有factor final layers打开；factor打开后Core、
  post-fusion和zero-init `W_mod`得到梯度，而Procedure仍为零；`W_mod`非零后
  Procedure路径才获得非零梯度。fresh完整public LoRA逐tensor等于identity
  template，routing和position无法从全零value生成LoRA内容。

## v5.1训练与推理profile结论（2026-07-27）

- 新架构没有改变单视频完整action-batch的主要显存斜率。v5.1在真实105帧压力
  条件下F32/B20稳定完成，峰值allocated `76.93GB`、reserved `83.64GB`；
  B20已经是当前A100 80GB上的激进稳定上限。常规步`3.25–3.66s`与最长步
  `7.25s`共同支持将首段定为900 steps约一小时，但这只是新架构实测后得到的
  首段尺度，不意味着下一段必然到1800。
- LoRA生成均摊到全部rollout worker在现场可行。每卡6个worker同时常驻时，
  24个worker对47个unique requests各处理1–2项，最大生成wall仅`5.20s`；
  rollout阶段整卡约63–65GB、利用率`99–100%`，有效吞吐`0.3799 episode/s`。
  主要固定开销是每个完整policy worker约`146–162s`的首次加载，因此同一
  checkpoint的worker必须保留并直接进入rollout，多checkpoint时优先一张卡
  负责一个checkpoint并并发摊销加载。
- long-first必须是全局优先级而非只优先本GPU affinity。旧SQL会在本卡long
  已领完时把ordinary排在他卡long之前；现改为所有`preferred_gpu != NULL`
  的max-horizon shard先于ordinary，随后才在long内部优先本卡。这个规则与
  checkpoint分配几张卡无关。
- profile只回答效率和运行合同，不提供v5.1科学效果证据。首段step900完成后，
  必须先选择observed-best并做内部Core/Procedure/LoRA检查和轻量paired
  correct/same/wrong/shuffled/reversed；若早期顺序特异性没有实质改善，
  不因loss仍下降就机械启动第二段。

## v5.1首段、observed-best与轻量特异性结论（2026-07-27）

- fresh v5.1正式首段完整运行step0→900：4-rank DDP、F32/B20，每step全局
  80条action queries和4个one-video conditions，合计72,000 queries、
  3,600 video conditions；9个每100步checkpoint和exact-resume state均完整。
  wall为`3,622.36s`，最终train loss为`0.10348`，全程平均吞吐
  `25.69 queries/s`。online validation从step100的`0.13212`到step900的
  `0.13314`，中间最低为step200的`0.13061`，整体已平台化且非单调；train
  loss继续下降不能单独支持自动续训。
- 旧有放回video采样脚本的80-rollout checkpoint screen为：
  step100/200/300/400/500/700/800/900 =
  `19/18/15/7/21/17/19/14`。owner指定的四卡并行正式复核把
  step100/500/700/900分别扩到400条，得到`82/96/98/84`。因此step700是
  当前observed-best，但`98/400`仍明显低于v5.1 absolute预门约
  `110–120/400`和旧Action-Forecast最低参照`125/400`；step500与step700的
  2条差距也不足以声称形成清晰性能峰。
- 只对step700做了80条完全配对的五臂轻量检查。最终
  correct/same-task-other/cross-suite-wrong/shuffled/reversed为
  `17/20/7/11/6`。相对correct的discordant pairs和双侧exact McNemar为：

  | control | correct-only | control-only | correct-control | exact p |
  |---|---:|---:|---:|---:|
  | same-task-other | 4 | 7 | -3/80 | 0.54883 |
  | cross-suite-wrong | 12 | 2 | +10/80 | 0.01294 |
  | shuffled | 10 | 4 | +6/80 | 0.17957 |
  | reversed | 13 | 2 | +11/80 | 0.00739 |

  same-task-other没有显著恶化，方向上可视为鲁棒，但小样本不能证明等价。
  wrong和reversed在paired-state层面已有显著信号；然而correct-wrong只有
  3个task正、1个负、4个平，主要由Object-1的`+7/10`贡献，
  correct-reversed也只有3正、0负、5平。shuffled仅2正、1负、5平且不显著。
  因而视频语义与reverse顺序信号存在，但尚未形成跨task稳定的完整order gate。
- 16条reference（8 tasks × 每task 2 videos）的内部检查揭示了清晰的分层机制：

  | condition | Semantic Core | Procedure slots | effective LoRA | policy action |
  |---|---:|---:|---:|---:|
  | same-task-other | 0.0403 | 0.3801 | 0.1163 | 0.0139 |
  | cross-suite-wrong | 0.2041 | 1.1826 | 0.5884 | 0.1414 |
  | shuffled | ~0 | 1.0516 | 0.5285 | 0.1015 |
  | reversed | 0.0030 | 1.6803 | 0.7434 | 0.1668 |

  数值均为相对correct的逐样本relative L2中位数。wrong在所有8个task都改变
  Semantic Core；shuffle/reverse保持set-like Core近似不变而大幅改变causal
  Procedure。固定Core、只替换Procedure时，shuffled/reversed的effective
  LoRA与action差异几乎完整保留；移除Procedure时shuffled严格归零，reversed
  action只剩`0.0021`。这证明v5.1的模块分工和信息路径按设计工作，但下游
  closed-loop成功率没有把这种内部差异稳定转化为足够强的跨task优势。
- cross-suite-wrong第一次运行发生单worker MuJoCo/EGL
  `Offscreen framebuffer is not complete (0x8cdd)`。native resume虽补齐queue，
  aggregation因两次launcher timing窗口不一致而fail-close；该目录没有可信
  `results.json`，其数值不得引用。随后在GPU4–7进行全新单次调用，80/80 rows、
  26/26 shards、24 workers全部成功，正式wrong结果仅使用fresh rerun的`7/80`。
- 当前科学判断：absolute gate失败；same鲁棒性方向可接受；wrong语义与reverse
  在state-pair层面成立但task breadth有限；shuffle顺序证据未成立；overall
  v5.1首段未通过继续训练或进入RL的门槛。owner要求特异性检查后停止，因此
  无放回重测、第二/第三段、full-400五臂和cold-start RL均未启动。

## v5.1 step700无放回全量特异性与新推进判据（2026-07-27）

- owner随后解除旧停止边界并要求持续探索；正式视频合同改为每个validation
  task的50条teacher videos对50个fixed states做一次无放回随机双射。correct、
  shuffled、reversed使用同一state对应的同一demo；same-task-other使用
  `(correct_demo+17) mod 50`；wrong保持demo ordinal并换到cross-suite task。
  五臂的task/state、language、env seed、policy seed和实际消费noise前缀均已
  逐row核验一致。
- step700无放回full400结果：

  | arm | successes | success rate |
  |---|---:|---:|
  | correct | 88 | 22.00% |
  | same-task-other | 97 | 24.25% |
  | cross-suite-wrong | 75 | 18.75% |
  | shuffled | 65 | 16.25% |
  | reversed | 45 | 11.25% |

  correct相对same为`17/26` discordant、净`-9`、`p=.2221`，说明跨同task
  demo鲁棒；相对wrong为`46/33`、净`+13`、`p=.1766`，语义方向存在但未达到
  显著且正向净收益的`73.7%`来自Object-1，breadth不够；相对shuffled为
  `46/23`、净`+23`、`p=.00762`；相对reversed为`60/17`、净`+43`、
  `p=8.91e-7`。order破坏显著伤害行为，且不再复现v4 shuffled优于correct的
  逻辑漏洞，但主要有效task仍集中在Goal-6与Object-1/3，不能把它等同于充分
  absolute competence。
- absolute `88/400`比旧有放回step700的`98/400`更低，且远低于旧
  Action-Forecast约`125/400`与v4 shuffled `148/400`目标区域。内部结构有
  特异性、order OOD会退化，只证明video信息进入函数；它没有证明correct
  Procedure向量对控制有足够有用的方向。因此继续训练后的关键不是只看内部
  distance，而是用统一无放回correct曲线定位observed-best，再检查
  Core-only/full/Procedure强度与LoRA功能尺度。
- 完整逐row分析在
  `/data/ymdai/outputs/ember/pi05_as_writer_v5_1_specificity400_noreplacement_seed7_step0700_paired_analysis_92b1e03_20260727.json`，
  SHA256为
  `c4a62c4c091b1262c3dbcb17382aad757b8865958212ae62b4a9e4f5986231fa`。
  它只承担step700机制证据；不得与旧有放回checkpoint screen混合选best。

## 多GPU评测尾部根因与修正（2026-07-27）

- 24 worker现场已经满足“每个worker先long、全局无long后才ordinary”，但旧
  ordinary分片仅24份，long结束后恰好每worker一份；成功即终止导致单波运行时
  差异，最后阶段出现三张卡等待一个ordinary shard的长尾。
- 新算法不改变long优先级、preferred-GPU affinity或任何state/video配对，只在
  ordinary名义shard数少于两个worker波次时，寻找满足目标的最大state cap。
  四卡×6 worker、8 tasks×50 states时得到48 long + 48 ordinary，ordinary
  每片最多7 states；少卡/少worker按实际slots自适应，已有足够动态工作时不再
  细分。该改动优化有效rollouts/s，不用dummy workload填显存。

## v5.1完整1800-step曲线与当前瓶颈（2026-07-27）

- 同一formal轨迹的无放回correct400闭环曲线为：

  | step | 100 | 500 | 700 | 900 | 1000 | 1100 | 1200 |
  |---:|---:|---:|---:|---:|---:|---:|---:|
  | success | 83 | 98 | 88 | 86 | 114 | 111 | 114 |

  | step | 1300 | 1400 | 1500 | 1600 | 1700 | 1800 |
  |---:|---:|---:|---:|---:|---:|---:|
  | success | 92 | **127** | 95 | 92 | 65 | 126 |

  step1400是observed-best，step1800仅少1个成功；但paired row中
  step1800相对1400新增28、丢失29，双侧McNemar `p=1.0`。这不是所有任务
  同时收敛到稳定平台，而是大幅能力迁移后的aggregate巧合。
- 逐task曲线揭示结构性短板：两个spatial validation tasks在所有checkpoint
  合计都不超过`2/100`，Goal-3不超过`3/50`。step1400的
  Spatial-1/3、Object-1/3、Goal-3/6、Long-1/2依次为
  `0/1, 45/23, 0/42, 15/1`；step1800为
  `1/1, 44/19, 3/39, 16/3`。因此127不是广覆盖ceiling。
- step1000/1200/1400/1800四点的成功集合并集为`180/400`、交集仅`65/400`；
  1400与1800并集也有`155/400`。Writer参数每200步的更新L2约
  `6.39–6.57`，相邻更新余弦仅`.015/-0.030/-0.055`。原scheduler在1400和
  1800仍分别使用`2.904e-4/2.842e-4`，所以继续同一高学习率轨迹更可能继续
  任务迁移，而非稳定累积能力。
- step1400内部机制没有重现v4漏洞。same/wrong/shuffled/reversed相对correct的
  Semantic-Core中位差为`.0509/.2310/~0/.00286`，effective-LoRA为
  `.0960/.6733/.5140/.6734`，policy-action为
  `.0149/.1329/.1035/.1823`。固定Core只换Procedure时order差异几乎完整穿过
  fusion、LoRA与policy；Core-only对order近似不变。逻辑链成立而absolute和
  breadth不足，指向“表示/更新有差异但未形成足够有用的控制修正”。
- frame-set attention的8个learned gates从初始化`0.05`到step1400仍只在
  `0.0497–0.0507`，同时两个spatial task长期近零。这是后续语义容量诊断的
  具体证据，但尚不能仅凭参数静止断言软件bug或直接修改架构；先完成无副作用的
  LoRA函数尺度扫描与低学习率稳定性实验。

## evaluator EGL/resume根因与强度实验合同（2026-07-27）

- 三次相同故障均发生在同一物理GPU的多个worker几乎同时关闭旧LIBERO env并
  创建新env时，报
  `mujoco.FatalError: Offscreen framebuffer is not complete, error0x8cdd`。
  修复只把每张物理GPU的EGL close/create transition置于用户级flock内；正常
  rollout仍并行，GPU间也不串行。
- 旧aggregator只统计最后一次invocation的worker完成shards，使已经完整补齐的
  step500/1600在resume后fail-close。新证据模型逐次校验
  `invocations.jsonl`、failure artifact、累计complete计数和最终worker计数，
  并把active wall按所有attempt求和。正式补聚合得到step500
  `98/400, 2494.758s`和step1600 `92/400, 2578.903s`。
- `writer_lora_b_scale`只在cache load后乘每个public LoRA的B因子，A与原始
  cache hash不变，因而严格等价于整体缩放effective delta；scale进入run
  contract与paired-control hash，但不污染Writer输入或LoRA生成身份。该扫描
  用于检验v5.1 effective-LoRA幅度低于旧125/400 Writer约1.5倍这一可证伪假设，
  不把调大幅度本身视为性能改进。

## v5.1 LoRA函数强度扫描结论（2026-07-27）

- step1400保持同一400-entry Writer cache、A因子、state/video无放回双射和
  policy RNG，只把全部public LoRA-B整体缩放。full400结果为：

  | LoRA-B scale | 1.00 | 1.25 | 1.50 | 1.75 | 2.00 |
  |---:|---:|---:|---:|---:|---:|
  | successes | **127** | 124 | 119 | 99 | 82 |
  | 相对1.00新增 | — | 21 | 26 | 19 | 14 |
  | 相对1.00丢失 | — | 24 | 34 | 47 | 59 |

- `1.25×`虽然把Long-1从`15`推到`20`，却同时把Object-1/3从`45/23`
  降到`40/20`；`1.50×`也只是Long-1和Goal-6上涨，同时Object与全部近零
  task继续恶化。四个scale上的Spatial-1始终`0/50`，Goal-3始终`0/50`。
  因此这不是“LoRA整体幅度太小”，而是不同task的最优控制修正方向/幅度不一致；
  放大只加剧task迁移，不能修复表示和训练稳定性。
- 后续保留`1.00×`。这项负结果排除了最便宜的推理期标量修正，也避免把一个
  易task的局部收益误写成absolute提升；下一实验应处理高学习率迁移和上游
  task-conditioned语义容量，而不是继续扫更大scale。

## v5.1 step1400完整五臂的精确结论（2026-07-27）

- 固定无放回400结果为
  `correct/same/wrong/shuffled/reversed=127/133/94/107/120`。
  same相对correct净`+6,p=.5044`，同任务跨demo鲁棒；correct相对wrong
  净`+33,p=1.12e-4`，视频语义进入行为；correct相对shuffled
  净`+20,p=.0225`，没有复现v4的shuffle获益。
- reversed仍未通过：correct-only/reversed-only=`39/32`，净`+7`，
  `p=.4767`。逐task方向高度异质：reversed在Long-1和Object-3分别
  `+10/+5`，correct主要靠Goal-6的`+19`抵消。因而“内部Procedure差异很大”
  不能替代行为方向门；v5.1是部分修复，不是逻辑闭环。
- wrong净差也并非广泛均匀：Object-1/3贡献`+21/+12`，Goal-6反而`-1`，
  Spatial-1反而`-2`。加上correct在Goal-3/Spatial-1/3仅`0/0/1`，现有
  `127/400`仍是窄task组合，不是满意absolute competence。
- 配对artifact SHA256为
  `51c19b66e2c85501b986e57590deec7726ef19c7a71ae48279d6f551a4ec1579`；
  same/wrong/shuffled/reversed results SHA256依次为
  `3f2078ab...d5b9`、`0ff77de2...37b`、
  `1a73dcfe...ed8`、`0a0d9e32...e73`。

## v5.2 task-queried patch grounding假设（2026-07-27）

- scale扫描排除“public LoRA整体幅度过小”，失败task的effective-LoRA范数
  也不小；最早剩余瓶颈更像是v5.1只保留multimodal task-token hidden、
  过早丢掉patch-level对象/关系/空间细节。
- v5.2让text-only task tokens逐帧cross-attend 256个shared-projected
  image-position contents。Q/K/O可学习，V没有投影且只携带patch content；
  得到的task-aligned patch evidence与原multimodal task-token evidence相加
  后才进入顺序不变Core。Procedure暂不改变，以保持实验可归因。
- 新模块`197,120`参数由factor-head hidden `240→216`释放的`204,288`
  参数支付，总参数从`10,244,872`降至`10,237,704`。因此它检验的是预算上移，
  不是扩容。若它只提高spatial/absolute而reversed仍等价，下一失效owner应是
  Procedure任务条件化/读出，不应重新扩大factor heads或扫LoRA scale。

## v5.1 step1400低LR稳定段训练结论（2026-07-27）

- 从step1400只加载Writer权重，以fresh AdamW/RNG和`1e-4` peak LR运行的首段
  已完整到phase900；消费72,000 action queries和3,600个单视频条件，wall
  `3616.478s`，九个checkpoint与exact-resume状态均完整。
- online functional loss从phase100到900为
  `.135287/.133725/.138699/.133650/.134800/.134450/.133997/.135060/.136773`，
  没有相对原step1400 `.135241`形成持续下降。该loss与rollout相关性弱，故
  只用于否定“训练目标已明显改善”，不用于提前选择或拒绝checkpoint。
- 低LR把每100步Writer update L2从`1.870`逐渐压到`1.176`，但相邻update
  cosine除第一对`.032`外持续为负，范围`-.114`到`-.180`；Core、Procedure、
  encoder projection和Meta-LoRA均出现同向现象。phase900相对初始权重累计
  L2=`4.613`。因此当前证据是步幅减小但task迁移未消失，而不是稳定累积。
- 分模块原始分析SHA256为
  `7564fff2fa68b3d370f344d2b9e2180fe98cea640dbc9c82ed2f65af1b16ddc3`。
  最终科学判定仍等待phase100/300/600/900四个逐state/video/RNG配对的
  correct400；四者已在GPU4–7各占一张卡并发运行。

## v5.1低LR闭环负结果与v5.2 profile（2026-07-28）

- phase100/300/600/900的无放回correct400为`119/115/123/104`，全部低于
  原step1400=`127`。相对原best逐row净差为`-8/-12/-4/-23`；phase900
  exact McNemar `p=.00674`，是明确退化。四个panel都通过同task/state、
  teacher-demo无放回双射、env/policy seed和noise-prefix配对。paired artifact
  SHA256为
  `f52c9b78578cd217fd99dca34f1314421ee347861f215b1504f2e6aa51566543`。
- phase600虽为该阶段best，但逐task只有Long `10/2`、Goal `0/44`、
  Object `35/32`、Spatial `0/0`；没有打开任何缺失能力。结合连续负更新余弦，
  低LR只减小迁移步幅，没有修复任务间梯度冲突或上游空间语义缺口。因此不再
  给v5.1补控制臂，直接切换v5.2。
- v5.2真实patch-grounding hook在B20最长视频profile中测得patch/task evidence
  RMS比`0.404–0.462`、均值`.429`，cosine近零；新路径携带显著且互补的视觉
  内容。artifact SHA256为
  `2e2e69c6b082ac8a07a0681258d3abf1c95feef0748cf4d255d2ff17c1a789eb`。
- F32/B20先完成step1并exact-resume到step3；B21另做连续3步，全部finite，
  global84 queries/step，最大allocated/reserved
  `80,283,666,944/83,892,371,456` bytes。B21 metrics/run-summary SHA256为
  `80685bab...c0bd`/`492ee56a...410f`。
- B22在四rank第一步中对称OOM：每卡只余`54.94–80.94MiB`，仍需分配
  `666MiB`。failure log SHA256为`963e5e7d...3e3ce`。由此实测上界不是保守
  B20，而是可持续B21；正式首段使用F32/B21、global84、step900停止点。

## v5.2首段absolute、五臂与v5.3决策（2026-07-28）

- v5.2 fresh step0→900完整消费75,600 action queries和3,600个单视频条件，
  wall `3674.799s`。无放回correct400为
  `step100/500/700/900 = 72/79/120/132`；online functional loss在九点
  `.1332–.1388`内振荡，不能代替闭环选择。
- step900逐task为Long `11/0`、Goal `1/38`、Object `49/14`、
  Spatial `0/19`。相对v5.1 step1400=`127`，v5.2是new43/lost38、
  `p=.657`；Spatial-3净增18但Object-3净退9，故patch grounding确实打开了
  一个空间task，aggregate `+5`仍不是稳定架构统治。
- step900内部16-reference检查中，same/wrong/shuffle/reverse的
  Semantic Core中位relative L2为`.0397/.1959/~0/.00288`，
  effective-LoRA为`.1345/.6763/.7400/1.0346`，fixed-query policy action为
  `.0253/.1612/.0953/.1902`。固定Core只换Procedure时order差异完整保留，
  Core-only对order近零；v5.2没有v4式Core/order旁路。
- 无放回full400五臂为：

  ```text
  correct / same-task-other / cross-suite-wrong / shuffled / reversed
  132     / 138             / 74                / 82       / 83
  ```

  same的correct-only/control-only=`26/32,p=.512`，属于同档鲁棒；correct相对
  wrong为`75/17,p=7.29e-10`，相对shuffled为`63/13,p=5.04e-9`，相对
  reversed为`69/20,p=1.78e-7`。因此视频语义与正确时序都形成强闭环方向，
  明确排除了v4 shuffled/reversed漏洞。paired artifact SHA256为
  `d8e2f4b827f1aa22e3d778ee15c834f8ffd692c63c8fc46994414c52177a7ae7`。
- step900仍是训练右端而非峰后点。owner决定先沿原版v5.2
  `one task/rank/update` recipe exact-resume测上限；task-complete会改变
  optimizer坐标、冲突和噪声，只保留为独立后续对照。自动task-complete
  profile在用户决策到达时只完成GPU preflight，已在torchrun/输出创建前中止。

## v5.3 Task-Grounded Visual-Transition Procedure（2026-07-28）

- owner指定v5.3为默认下一架构实验，即使v5.2五臂已通过也要做；训练仍沿用
  原版v5.2的每rank每update一个task/视频/B_a合同，不采用task-complete。
- v5.3唯一改动是保留v5.2逐帧task-token patch evidence
  `G_f∈R^(L×256)`，在各arm实际输入顺序内重算
  `D_0=0,D_f=G_f-G_(f-1)`。Action-Expert probe `A_f`以八头Q/K/O、
  raw `D_f` value且无Wv的cross-attention读取transition，形成
  `Z_f=A_f+R_f`后进入原Causal Procedure。Procedure没有absolute patch
  旁路，Semantic Core保持frame-set置换不变。
- transition fusion真实参数`197,120`；factor-head hidden `216→192`释放
  `204,288`，Writer总参数由`10,237,704`降至`10,230,536`，低于
  Source-SFT上限`10,297,344`。step0 public LoRA仍由zero-init factor output
  保持精确functional identity。
- canonical实现与fresh v5.3 schema已在隔离分支
  `codex/v53-visual-transition-procedure@c1e3777` push；全仓回归
  `198 passed`。正式训练仍被真实GPU4–7最长视频profile、B20/B21
  上界、transition非零和step1→3 exact-resume阻塞。

## v6整体架构决策（2026-07-28）

- owner认可Visual-Transition方向，并决定不把它限制为v5.3小修订，而是从
  第一性原理重整整个Writer后命名为v6。完整设计authority为
  `docs/action_forecast_writer_v6_design.md`。
- v5.2的frame aggregation gate在step900–1800约
  `0.0490–0.0504`，几乎停留在0.05初始化附近。v6因此用始终保留静态信息的
  mean backbone，加上由text-only task query选择、以frame-centered evidence
  为value的residual；均匀attention时residual严格为零，且Core仍对frame
  permutation不变。
- Procedure保留v5.3最小可归因设计：`D_0=0,D_f=G_f-G_(f-1)`必须按各arm
  的实际顺序重算，由Action-Expert probe以8头Q/K/O、raw transition value、
  无Wv cross-attention读取；不加入absolute patch旁路、optical flow、
  geometry、long-range matching或order supervision。
- owner明确参数只需同量级，不要求机械等于rank-128 Source-SFT。v6所有主宽度
  统一为256、8 heads×32维、Core/Procedure各2层、post-fusion 1层，并把
  factor hidden从v5.3的192恢复为256。精确手算总参数`10,775,296`，相对
  Source-SFT `10,297,344`多`477,952`（约4.64%），换取更规整的硬件维度和
  不被人为压缩的下游compiler出口。
- 当时曾暂定首个v6比较沿用v5.2 one-task-per-rank训练范式；该暂定项随后被
  owner明确覆盖，最终活动合同见下一节。本时间点仅封存了设计。

## v6 task-complete最终训练合同与CPU实现（2026-07-28）

- owner最终覆盖了上一节最后一条：v6不再沿用v5.2 recipe，而是从fresh step0
  直接采用task-complete宏步。每macro为4 ranks × 6 tasks/rank；每task一条
  video、一套LoRA和B20 queries，task内均值后以`1/6`backward，前5轮
  `no_sync`、第6轮单次DDP同步，随后一次clip/AdamW/scheduler。
- B20时每macro精确覆盖24 tasks、24 video conditions/LoRAs、480 action
  queries和24次functional policy forward。旧900-step的3600 video conditions
  对应新150 macro左右；warmup由100 exposure-equivalent换为17 macro，
  decay由12000换为2000 macro，peak LR保持`3e-4`。
- task assignment按本macro实际选中teacher video的stride-5 frame count做
  四组cost balance；每rank内部long-first，四组随macro轮换物理rank。action
  query继续使用原有episode-balanced no-replacement采样，video/action独立。
- profile只允许B20和唯一B16 fallback；只有最长105帧视频OOM或连续完整macro
  不稳定才退B16，不扫描中间batch。首段约一小时；除非absolute明确下降，
  否则平台、轻微波动或上涨都默认续第二小时。
- v6 canonical实现参数枚举为`10,775,296`，step0 public LoRA仍精确identity。
  config/launch/checkpoint/eval全部提升到不兼容v6 schema；checkpoint记录
  macro cursor与`next_data_step=macro×6`，只允许macro边界exact-resume。
- 全仓CPU回归`200 passed`；architecture guard为REVIEW但无hard violation。
  新增唯一`writer/architecture.py`集中模型拓扑合同，复用现有Writer、training
  entrypoint和evaluator；v5.3 executable config已退役，没有新增平行runner。
- v6确认后，corrected mixed-task rank-128 Source-SFT必须fresh重训：
  physical batch跨tasks，按task→episode→chunk分层均匀采样并做task-balanced
  loss。旧rank-pure SFT只作provenance，不再代表最终strong baseline。

## v6 task-complete B20真实profile与正式seal（2026-07-28）

- GPU4–7在commit `d66e726`完成B20三步真实profile；首步实际包含全局最长
  105帧stride-5 teacher video，全部4 ranks和18个rank-local microtasks正常
  完成，无OOM、nonfinite或拓扑不对称。
- 三个macro的max-rank wall为`20.442/18.585/18.635s`，后两步平均
  `25.793 queries/s`、`193.447 macro/hour`；峰值allocated/reserved为
  `76,985,299,968/83,644,907,520 bytes`。按预声明二选一规则选择B20，
  不运行B16，也不扫描更大或中间batch。
- 独立step1→3 resume smoke确认边界checkpoint文件bitwise相同，后续
  task/video/query、LR、sampler cursor和loss轨迹一致；CUDA后续更新最大参数
  绝对漂移约`9.82e-5`，明确记录为非bitwise数值重现而非状态缺失。
- 真实profile step1→3的visual-transition模块L2位移`0.0111083`，
  `197,067/197,120`参数发生变化，证明真实transition非零且functional
  gradient可达。正式首段为B20、200 macro、每25 macro checkpoint。
- owner明确正式run同样不计算全量HDF5 SHA；保留sealed manifest、精确文件
  size、HDF5 schema和完整训练/checkpoint cursor证据。

## v6 task-complete正式首段训练证据（2026-07-28）

- fresh B20/K6 run 自然完成 macro0→200，wall `3,864.599s`；200 条 metrics
  连续、唯一且 loss/gradient 全 finite。全段平均 max-rank step wall
  `18.668s`、吞吐 `25.720 queries/s`，峰值 allocated/reserved 为
  `76,986,335,232/83,642,810,368` bytes。
- 终点精确消费 `4,800` 个 one-video conditions、`96,000` 条 action queries。
  24 tasks 各 4,000 queries、200 次 video visits；每 task 均覆盖全部
  50 action episodes 与 50 teacher videos。25..200 的 8 个 checkpoint
  schema、contract、cursor、size 与计数一致，macro200 六个状态文件逐文件
  SHA256 重算通过。
- online task-balanced functional loss 在 25..200 为
  `.130744/.133971/.133841/.133092/.132344/.133132/.134178/.137535`。
  后段上升只作诊断；按预封存规则不能替代 closed-loop absolute，也不能单独
  阻止第二段。下一步固定评测 macro50/100/150/200 的 paired
  no-replacement correct400。

## v6 macro0→200 closed-loop correct曲线（2026-07-28）

- macro50/100/150/200 的 fixed no-replacement correct400 全部自然完成，
  successes 为 `114/77/120/129`，成功 task 数为 `6/7/7/5`。四点均为
  400 rows、36 completed shards、6 workers exit 0、零 queue error。
- 四点 task/state、env seed、policy seed、终止前 noise prefix 和 teacher
  video assignment 完全相同；每 task 的 50 teacher videos 都是一一无放回
  使用。paired analysis SHA256 为
  `64fa284511e21230417b9ef27a99a9c050b661670eb90e549974acd8b9672464`。
- suite totals：

  ```text
  macro          50    100    150    200
  libero_10      10     15      9     22
  libero_goal    30     12     43     24
  libero_object  73     44     66     81
  libero_spatial  1      6      2      2
  ```

- macro200 是 aggregate observed-best，但只覆盖 5/8 tasks，并由
  `libero_object:1=50/50` 强烈集中；macro150 为 120/400、覆盖 7/8 tasks。
  macro150→200 的 paired switches 为 30 lost / 39 gained，McNemar
  `p=0.3356`。所以第一段没有出现明确右端下降，满足默认续训条件；同时
  absolute 与 breadth 尚不足以宣称 v6 成立，必须先完成 macro200 五臂和内部
  传递检查。

## v6 macro200正式五臂（2026-07-28）

- 五臂 full400 为
  `correct/same-task-other/cross-suite-wrong/shuffled/reversed =
  129/131/108/111/105`，成功 task 数为 `5/7/7/6/5`。每臂 400 rows、
  36/36 shards、6/6 workers exit 0、零错误；state/env seed/policy
  seed/noise 均 paired，所有 task 的 50 teacher videos 无放回双射。
- same 相对 correct 的 switches 为 `22/24,p=.8830`，属于同档鲁棒。
  correct 相对 wrong 为 `42/21,p=.0111`，相对 shuffled 为
  `36/18,p=.0198`，相对 reversed 为 `37/13,p=.00094`。三项行为门均通过，
  但 aggregate margin `21/18/24` 显著小于 v5.2 step900 的 `58/50/49`。
- 正方向不是全 task 一致：wrong 差异主要来自 Long-1 `+14`、Object-1
  `+7`、Object-3 `+9`，Goal-6/Spatial-1 反而 `-3/-3`；shuffled 主要来自
  Long-1 `+11`、Object-3 `+8`，Goal-6 为 `-2`；reversed 主要来自 Goal-6
  `+15`、Long-1/Object-3 各 `+5`。因此 v6 已排除 v4 式控制臂反超，但尚未
  获得广泛稳定的视频特异性。
- v6 macro200 与 v5.2 step900 在同一 states/videos/RNG 下为 `129 vs 132`，
  v6-only/v5.2-only 为 `45/48,p=.8358`。task delta 为 Long-1 `+11`、
  Object-3 `+17`、Goal-6 `-14`、Spatial-3 `-19`；这是能力重分配，不是整体
  架构统治。由于 macro200 仍是 v6 右端最高点，按 owner 合同继续第二段。

## v6 macro200内部传递（2026-07-28）

- 8 validation tasks × 2 references 的五条件检查完整生成 16 rows。same/wrong/
  shuffled/reversed 的 Procedure median relative-L2 为
  `.0365/.1345/.0888/.1167`，effective LoRA 为
  `.0856/.3233/.2590/.2436`，固定 policy action 为
  `.0139/.0501/.0282/.0392`，说明差异沿完整链路存在。
- shuffled/reversed 在 fixed-Core Procedure-only 反事实下的 effective LoRA
  `.2590/.2437` 与原始结果基本相同，action `.0283/.0395` 也基本相同；
  Core-only 分别接近零。因此顺序信号来自
  visual-transition→Procedure，而不是 Semantic Core 或 frame-set 旁路。
- 相同面板下 v5.2 的 shuffled/reversed Procedure 仅
  `.0393/.0589`，但 effective LoRA 达 `.7400/1.0346`、action 达
  `.0953/.1902`。v6 确实加强了上游时序区分，却在 macro200 仍将其压缩到
  较弱的 LoRA/action 差异；行为 margin 同步变弱，当前候选瓶颈是训练成熟度
  或 Procedure-to-compiler 映射，而不是“Procedure看不到顺序”。
- shuffled 的 visual-transition RMS 中位 `.06446`，correct 为 `.03075`；
  虽有相邻跳变放大，但 transition residual/action-probe RMS 仅
  `.2688`，attention effective tokens 约 `11.0`，未见数值爆炸。

## v6第二小时与任务漂移结论（2026-07-28）

- 同一root从macro200 exact-resume到400，metrics连续`1..400`、每25步完整
  checkpoint。第二段消费额外4,800 video conditions和96,000 action queries，
  wall `3,903.024s`；24 tasks各覆盖全部50 videos与50 action episodes。
- paired无放回correct400完整曲线为：

  ```text
  macro          50   100   150   200   250   300   350   400
  successes     114    77   120   129   117   118   125   125
  tasks > 0       6     7     7     5     7     5     6     7
  ```

  macro200仍是observed-best；它相对250/300/350/400均无显著差异，但成功集合
  大幅交换。Goal-6从24涨到40，同时Long-1从22降到14、Object-3从31降到21。
  因而第二小时改善breadth却没有提高aggregate，能力迁移仍是主要现象。
- 额外200个optimizer updates、下降的train loss和持续增大的Procedure
  modulation norm没有带来多task共同上涨，“只是更新次数不足”明显降权。
  第一嫌疑转为每次full-24平均的优化粒度；v6上游拓扑仍被内部反事实支持，
  若训练粒度修正后仍失败，才把Procedure→compiler增益列为下一架构owner。
- owner最终将corrected Source-SFT提前为紧邻实验，先建立可信baseline再继续
  Writer搜索。focused AS硬门统一为
  `correct400 >= max(150, corrected SFT_best + 30)`，并同时要求same≈correct、
  correct显著优于wrong/shuffled/reversed、多task共同贡献和独立paired复测。

## corrected mixed-task Source-SFT实现与B144 profile（2026-07-28）

- canonical Source-SFT现在每个rank的physical batch直接混合全部24 tasks并
  等量计数；B144时每task每rank6条、全球每task24条。每个batch只做一次普通
  forward/backward/DDP sync/clip/AdamW，不使用gradient accumulation、
  `no_sync`或Writer task-complete micro-round。
- sampler按task→episode→chunk分层均衡，episode/chunk使用确定性无放回周期；
  跨rank row不重复，绝对step决定sample identity，因此resume无需可变sampler
  状态。21个sampler/contract/checkpoint/inference focused tests全部通过。
- GPU4–7 profile root
  `pi05_source_sft_rank128_mixed_profile_r4_b144_55ccbcc_s3_20260728`
  fresh完成step1并从完整checkpoint resume到step3。三步共1,728 unique rows；
  每步24 tasks等量，step3时每task已覆盖全部50 episodes。后两步wall为
  `16.684/15.847s`，吞吐`34.524/36.346 queries/s`，峰值
  allocated/reserved为`60,690,811,904/74,065,117,184 bytes`。
- loss/gradient finite，frozen source policy无梯度，唯一trainable对象为
  `10,297,344`参数rank-128 shared LoRA。B144稳定，因此不触发B120 fallback。
  正式首段封存为fresh step0→225、每25步checkpoint，约61分钟training body；
  若峰值在右端或不稳定，exact-resume到450。

## corrected mixed-task Source-SFT首段与候选筛选（2026-07-28）

- fresh step0→225自然完成，225条optimizer metrics连续且全部finite；累计
  129,600 action queries，24 tasks各5,400 samples，训练body wall
  `3,639.436s`。step25..225共9个checkpoint的LoRA、trainer state、四rank
  RNG/sampler state与manifest hash均完整，首段root仅551MB，不需要裁剪。
- online validation loss在step25/50/75/100/125/150/175/200/225依次为
  `.139748/.134216/.134064/.132966/.133862/.134068/.134527/.135724/.135276`。
  step100最低；后段Object validation tasks仍改善而若干Spatial/Goal/Long
  tasks回退，再次提示shared SFT本身也可能发生task redistribution。但该loss
  与closed-loop success相关性有限，不能据此直接冻结step100。
- 为同时覆盖早期肩部、online best、中后段和训练右端，step50/100/175/225
  分别映射到GPU4/5/6/7完成相同fixed correct400。结果为`60/75/77/56`，
  成功task数为`5/8/6/6`。每点400 unique rows、36/36 shards、6/6 workers
  exit 0、全attempt1、零错误；task/state/env/policy seeds与noise prefix
  完全paired。
- 逐task step50/100/175/225依次为：

  ```text
  Long-1       0 /  4 / 14 /  7
  Long-2       2 /  4 /  4 /  0
  Goal-3       0 /  3 /  1 /  1
  Goal-6      30 / 29 / 25 / 15
  Object-1    25 / 26 / 27 / 26
  Object-3     0 /  2 /  6 /  6
  Spatial-1    2 /  1 /  0 /  0
  Spatial-3    1 /  6 /  0 /  1
  ```

  step100→175为37 lost/39 gained、`p=.9088`，aggregate同档但能力明显迁移；
  step175→225为40 lost/19 gained、`p=.00864`，右端下降可信。online
  per-task loss也呈相同方向：100→225时Object-1/3改善，而Spatial-1、
  Goal-6和Long-2恶化，说明aggregate loss掩盖任务冲突。
- step175虽比source base `77 vs 48`显著（paired `50/21,p=.00077`），却显著
  低于旧四卡rank-128 SFT `108`（`18/49,p=.00019`）、旧八卡`122`
  （`18/63,p=5.20e-7`）和v6 macro200 `129`（`24/76,p=1.81e-7`）。
  这首先反映当前只训练225 updates/129,600 queries，不能把corrected
  mixed-task recipe判为低上限。
- step100/175近乎平局、task集合持续交换且单个右端下降不足以排除后续恢复，
  满足预封存的“不稳定峰值”条件。下一段从step225 exact-resume到450，
  不改recipe；已有checkpoint全部保留。

## corrected full-24 Source-SFT上限与global-8候选（2026-07-28）

- full-24同一root从step225 exact-resume到450；450条metrics连续finite，
  step25..450共18个checkpoint全部保留。dense paired correct400在
  step50/100/175/225/275/300/325/350/375/400/425/450为
  `60/75/77/56/77/57/87/71/98/109/107/74`。
- step400/425为`24/22` switches、`p=.883`；425→450为
  `45 lost/12 gained,p=1.31e-5`，400→450为
  `50/15,p=1.57e-5`。因此observed-best封存为step400=`109/400`，有充分
  post-best下降，不再续训full-24 recipe。
- step400只与旧四卡rank-pure best `108`同档，仍低于v6 macro200=`129`
  （paired `32/52,p=.0375`）。full-24普通SFT也表现为任务能力反复迁移，
  说明漂移不只来自Writer异构LoRA或Procedure，而与共享多任务优化粒度有关。
- 新canonical sampler将每个24-task平均update改为global-8：
  4 ranks×2 tasks，连续3 updates通过cycle permutation完整覆盖24 tasks；
  B144时global576 queries/update不变，每task跨cycle的平均sample clock与
  full-24精确一致。没有gradient accumulation或第二套训练入口。
- GPU4–7 B144 profile fresh0→3并exact-resume到6；两轮cycle均24 tasks
  各一次、3,456 query identities唯一。稳态`36.27–36.38 queries/s`，
  峰值allocated/reserved `60.69/74.07GB`，loss/grad finite且无OOM。
  正式首段封存fresh0→240、每30步checkpoint；默认再续到480，除非首段出现
  可信的多task绝对下降。

## global-8 Source-SFT负结果与漂移归因更新（2026-07-29）

- global-8保持与full-24完全相同的frozen source base、rank-128 LoRA、
  global576 queries/update、LR/scheduler及平均task/sample clock，只把一次
  24-task梯度平均拆成三个8-task AdamW updates。正式0→480完整消费276,480
  queries；16个checkpoint、optimizer/scheduler、四rank RNG/cursor和终点
  manifest均完整；两段进程启动至封存合计约`11.32 GPU-hours`。
- paired correct400曲线为
  `63/83/85/98/90/62/90/105`。step480虽然是该recipe右端best，但420→480
  为`21 lost/36 gained,p=.0627`，240→480为`30/37,p=.464`；此前360又跌至
  62。这不是稳定单调上升。八点逐task envelope为126、best为105，仍有21个
  success无法共存于同一checkpoint。
- global-8 step480与full-24 observed-best step400为`105 vs109`，paired
  full24-only/global8-only=`32/28,p=.699`。global-8把Long-1/Goal-6提高
  `+5/+4`，同时Long-2/Object-1/Object-3下降`-4/-7/-2`，两个Spatial task
  均为0。故8-task更新只改变能力分配，没有提高aggregate或breadth上限。
- 这条SFT直接对照显著削弱“full-24平均过强是任务漂移第一根因”的判断。
  漂移更可能来自共享参数上的多目标不一致、action-MSE与闭环success阈值错位，
  以及仍偏大的持续更新步幅；Writer的per-video条件噪声和较弱
  Procedure→compiler传递是附加问题，但不是SFT也漂移的共同解释。
- v6八点逐task envelope为156、observed-best为129，存在27-success可组合
  空间。因此下一项最低成本判别不是直接运行同构cyclic-8 Writer，而是把同一
  v6轨迹的若干高分/互补checkpoint在参数空间合成单套Writer权重并跑paired
  correct400。该derived权重必须显式provenance、只允许inference且部署仍为
  一次Writer前向；若线性合成失败，再fresh验证更快LR衰减/稳定化优化，最后
  才修改Procedure→compiler。

## v6参数平均的可证伪合同（2026-07-29）

- 该screen检验的是“同一训练轨迹上互补能力是否位于可线性合成的局部参数
  basin”，不是ensemble：每个候选最终只有一套Writer权重、每条视频仍只有
  一次Writer forward，没有额外policy或LoRA投票。
- 四候选在看outcome前固定，分别覆盖raw best局部邻域、macro200/400互补端点、
  两个高分邻域和宽窗口late-SWA。若四者均不能可信超过raw macro200=`129`
  且改善多task breadth，就能直接降低“只需轨迹平均”的解释概率，下一fresh
  实验转向更快LR衰减/较小持续步幅，而不是事后挑选更多平均组合。
- derived schema与raw training schema分离；manifest封存全部source
  manifest/Writer hash和均匀权重，inference会复核source小manifest与正式run
  authority，但训练初始化和exact-resume在目录/schema层均拒绝derived权重。
- 真实state包含523个训练参数tensors与77个固定buffers。四个候选的独立重算
  均为0 mismatch、最大绝对误差0、全部finite；因此后续closed-loop差异可归因
  于参数平均本身，而不是漏tensor、dtype漂移或错误加载。

## v6参数平均结果、视频因果与下一稳定化实验（2026-07-29）

- 四个预封存候选的paired correct400依次为
  `129/140/144/145`，成功task数为`7/6/7/7`。六点late average
  `{150,200,250,300,350,400}`相对raw macro200=`129`为
  `37 gained/21 lost,p=.04794`，净增16且把nonzero task从5/8扩到7/8；
  相对corrected Source-SFT best=`109`高36，但仍比absolute硬门150少5。
  候选screen paired artifact SHA256为
  `09d4399662de821a1de0d6f38903eeba60a571fee2594c02fe6a445013dfb8ac`。
- winner的正式五臂为
  `correct/same-task-other/wrong/shuffled/reversed =
  145/134/128/119/122`。correct相对wrong为
  `38/21,p=.03634`，相对shuffled为`44/18,p=.001299`，相对reversed为
  `45/22,p=.006741`；三项均显著。正向贡献task数分别为5/6/5，最大单task
  占正贡献`.391/.615/.560`，没有由一个task独占。wrong在Goal-6上仍反向
  `-5`，但Long-1/Object-1/Object-3/两个Spatial均正向，语义门整体成立。
- same相对correct为`30/19,p=.1524`，统计上没有显著差异；aggregate少11。
  它只比分析前固定的保守`|delta|<=10`边界多1，因此当前记为borderline
  same-tier、要求最终winner独立复测，而不是把阈值事后放宽。
- 五臂均400 unique rows、36 shards、6 workers exit0、全attempt1且无adopt；
  task/state/env/policy/noise完全paired。每task 50 videos均无放回双射，
  same固定`+17 mod 50`，wrong/shuffled/reversed复用correct demo；
  四个queue均在领取任何其它task前先领取全部12个horizon-520 shards。
  五臂paired artifact SHA256为
  `9244b8db004f4155f9ee254bbddbaf013ee033640b6d9974c2b98cd283579d8b`。
- 16-reference内部传递中，same/wrong/shuffled/reversed的effective LoRA
  median relative-L2为`.0801/.3591/.2689/.2923`，policy action为
  `.0116/.0568/.0576/.0434`。shuffled/reversed的fixed-Core
  Procedure-only分别复现`.2689/.2925` LoRA与`.0576/.0431` action；
  Core-only为`0/.00130`与`0/.00192`。因此参数平均没有抹掉视频语义或时序，
  也没有形成Semantic Core顺序旁路。内部summary SHA256为
  `7596fbd5cd03232d99667b5eb5b500995e5b1cbf6d1c01b97bb2c8a8628d169d`。
- 相对raw macro200，average的shuffled/reversed Procedure sequence由
  `.0888/.1167`降到`.0631/.0784`，但Procedure slots保持
  `1.055/1.393`，effective LoRA/action没有坍缩，shuffled action反而从
  `.0282`升到`.0576`。这进一步把当前瓶颈从v6拓扑降权为训练轨迹稳定性：
  原recipe到macro400的LR仍为`2.724e-4`，50-step更新方向cosine长期很低，
  而宽窗口平均能合成互补task能力。
- 下一fresh单变量实验保留v6、task-complete B20、24-task等权、AdamW全部
  不变，只把cosine `decay_steps 2000→400`；peak `3e-4`、warmup17和floor
  `1e-5`不变。实际LR将于macro200降到`1.55e-4`、macro400降到`1e-5`。
  首段0→200筛50/100/150/200，除可信absolute下降外默认续到400并筛
  250/300/350/400。若仍失败，下一步直接量化per-task gradient冲突，再决定
  update粒度或Procedure→compiler，而不凭直觉同时改多项。

## v6 fast-decay400结果与第二次参数平均screen（2026-07-29）

- fresh fast-decay run从step0完整训练到macro200，再以同commit/config和完整
  optimizer/scheduler/sampler/RNG从macro200 exact-resume到400。
  `metrics.jsonl`连续1..400、全部finite；累计192,000 action queries和
  9,600 video conditions。每25 macro的16个checkpoint及全部评测cache均
  保留。run summary/final manifest SHA256为
  `ceb03e39...bc84/c970026f...b91c`。
- fixed paired correct400八点为
  `106/64/111/133/132/117/138/143`，成功task数为
  `6/7/7/6/7/7/7/6`。macro400是raw右端observed-best，相对原v6同点125为
  `46 gained/28 lost,p=.04739`，相对corrected Source-SFT 109高34；
  因此已过`+30`底线但仍比absolute150少7。完整曲线artifact SHA256为
  `99b04bf1cf72ad2385119638ca8020c5caf24e2c33075d758ee7f38dcc253d03`。
- fast decay显著缩小位移但没有让梯度方向一致。原/新225→250 update L2为
  `2.292/.940`，325→350为`2.123/.283`；最后375→400仅`.1265`，
  cosine`.1189`。完整dynamics SHA256为
  `804689cac6e108357e6977fb1f263cdc7a13611be46eb6bd3e477d6cae805f32`。
- 能力漂移幅度随低LR缩小但没有消失：250→300为
  `29 gained/44 lost,p=.1006`，300→350为`49/28,p=.02203`，
  350→400为`25/20,p=.5515`。macro400逐task为
  `Long1/Long2/Goal3/Goal6/Object1/Object3/Spatial1/Spatial3 =
  20/1/0/36/46/37/0/3`。最后50 macro仅净+5，且参数已近冻结；所以不能把
  raw右端best机械解释为还应继续第三训练段。
- outcome前封存四份fast-decay uniform checkpoint-average候选：
  `{350,400}`、`{200,350,400}`、`{200,250,350,400}`和
  `{150,200,250,300,350,400}`。前三者从局部平滑逐步加入互补高分模式，
  第四者与原v6 late-six SWA口径匹配。screen config SHA256为
  `07d115811cf6042d5d0246e9f91c304aed3e5289b53d898d17af0330526951f5`。
  若它们不能可信超过raw143，则下一判别是per-task gradient conflict，而非
  继续同recipe或同时修改多个架构/训练变量。

## v6 fast-decay checkpoint-average结果（2026-07-29）

- 四份derived checkpoint均通过manifest/source provenance/inference-only
  authority检查；独立逐tensor复算覆盖600个tensors、12,064,064个state
  elements，四组`max_abs_error=0`。原16个every-25 checkpoint与全部派生
  checkpoint均保留。
- outcome前固定的四组correct400按
  `{350,400}`、`{200,350,400}`、`{200,250,350,400}`、
  `{150,200,250,300,350,400}`依次得到`139/135/129/130`，成功task数
  为`6/6/6/7`；均未超过raw macro400=`143`，也均未达到absolute150。
  只有局部两点平均`139`恰好满足corrected SFT 109的`+30`底线。
- 相对raw的paired`gained/lost,p`依次为
  `18/22,.6358`、`21/29,.3222`、`13/27,.03848`、
  `18/31,.08543`。最佳average `{350,400}`逐task为
  `Long1/Long2/Goal3/Goal6/Object1/Object3/Spatial1/Spatial3 =
  18/0/0/39/47/33/1/1`；它换来Goal6/Object1的小幅提高，但失去Long1、
  Object3和Spatial3，仍是能力重分配。
- 四组均为400 unique rows、36/36 attempt1 shards、6/6 workers return0、
  无adopt；teacher demo对每task严格`0..49`双射，且与raw的state、
  env/policy seed、noise prefix和video assignment完全paired。队列前12个
  shards全为horizon520，之后无long；因此负结果不是调度、采样或缺失造成。
- 四个averages的episode union为174，加入raw后为180：不同权重确实掌握
  大量互补成功，但fast-decay轨迹的均匀参数平均不能把它们压入单套Writer。
  这否定了“只靠post-hoc轨迹平均即可越过150”的当前解释，但不能单凭本screen
  证明具体梯度冲突机制；下一科学判别应先直接量化per-task gradient conflict。
- 完整artifact file/canonical SHA256为
  `ac6e1545...1d30/a9ffd347...9fdb`。owner要求本步完成后暂停讨论，所以没有
  对非winner补五臂/内部分析，也没有启动第三训练段或下一fresh实验。

## fast-decay续训封顶与macro400机制（2026-07-29）

- owner后续要求先把当前v6上限与best机制测清，因此同一fast-decay root从
  macro400 exact-resume到600。新增450/500/550/600的paired correct400为
  `131/130/132/126`；macro400=`143`继续是12点single-checkpoint best。
  400→600为`31 lost/14 gained,p=.01609`，右端已显著下降，排除了“只是还需
  继续同recipe”的解释。
- macro400五臂为
  `correct/same/wrong/shuffled/reversed=143/135/125/128/129`。same同档；
  correct对wrong为`38/20,p=.02475`且4个tasks正向；对shuffled为
  `37/22,p=.06744`，对reversed为`37/23,p=.09246`，后两项方向正确但未显著。
  所以task-complete v6的absolute有34点SFT优势，却把大量增益保留给无序视频。
- 内部检查没有发现上游时序失效：shuffled/reversed的Transition为
  `2.149/1.375`、Procedure slots为`1.137/1.230`，fixed-Core
  Procedure-only几乎复现effective-LoRA/action差异；Core-only近零。问题是
  正确与无序视频在下游仍只相差`.237/.042`和`.214/.047`量级，信号没有被
  行为决策充分放大。

## v6旧训练范式对照的判别结论（2026-07-29）

- 对照严格保持v6拓扑、数据、信息墙、B20、policy和public LoRA空间，只把
  update改为旧rank-rotating口径：一task/rank/update、global4 tasks、
  连续6 updates覆盖24 tasks、每步一次DDP sync/AdamW。正式900 updates约
  `3,626.7s`，消费72,000 action queries和3,600视频条件。
- correct400在step100/500/700/900为`98/121/76/95`。step500→700出现
  `50 lost/5 gained,p=2.14e-10`，到900仍只有95。500→700与700→900的整体
  update direction cosine仅`.190/.081`；factor heads每段相对位移仍约
  `.148/.144`。能力漂移对应真实、近正交的大参数运动，不是评测噪声。
- step500五臂为`121/122/111/84/47`。同样的v6拓扑在旧训练下把shuffled和
  reversed分别压低37和74，exact p为`3.76e-5/3.92e-16`；因此v6的
  Visual-Transition Procedure绝对能够形成理想的顺序因果行为，task-complete
  控制臂高分不能归因于架构完全读不懂顺序。
- 旧范式仍不是解：absolute仅121，比task-complete best143低22、只比SFT高12；
  仅5/8 tasks成功。wrong仍为111，correct优势10且`p=.237`，只有2个tasks
  正向，最大单task占正贡献`.867`。它恢复了order specificity，却没有恢复
  v5.2式wrong-video semantic specificity。
- 内部传递解释了行为差异。old recipe下shuffled/reversed的effective-LoRA与
  action median为`.363/.066`和`.606/.148`，显著高于task-complete的
  `.237/.042`和`.214/.047`；fixed-Core Procedure-only完整复现，Core-only
  近零。训练粒度确实在调节Procedure→compiler的有效增益。
- 综合结论不是二选一：task-complete倾向学习跨视频都有效的通用帮助，从而
  absolute较高但顺序margin弱；old recipe让单task连续更新产生更强、更不稳定
  的Procedure依赖，从而顺序margin大但absolute/breadth下降。两者共同指出
  当前瓶颈位于多任务优化与Core/Procedure融合增益的接口；现有证据尚不足以
  单独判定应只改训练还是只改融合，需先与owner讨论下一项单变量实验。

## v7第一性原理需求与设计决策（2026-07-29）

- owner把设计顺序明确为“需要什么→已经有什么→只补最少缺口”，同时强调
  最少拓扑不能牺牲必要模块的表达能力。Core负责高层对象/角色/关系、环境、
  目标与语言指定的子目标顺序；Procedure负责高层动作和随后语义效果的因果链；
  融合必须先借Core理解任务，再从Procedure读取怎样完成。
- 现有multimodal prefix的image/task hidden已经双向交互，足够建立唯一
  task-aligned semantic trajectory，不再需要text-only Gemma。逐帧
  task-to-patch grounding沿用v5.2的已验证机制；Core直接对token-aligned
  trajectory做frame mean，再用两层task-token Transformer组合高层语义。
- Action Expert teacher suffix的核心改动是单次forward中从1个probe扩为8个
  稀疏原生positions`[0,7,14,21,28,35,42,49]`；不是8次forward，也不改变
  execution policy的50-action chunk。8个probe分别读取
  `frame f→f+1`的forward semantic change；最终收敛为在全部`8×L`
  action–effect pairs上joint softmax，以语义变化为value、Action作逐通道
  调制，直接汇成一个高层event。不做提前mean，也不把8个anchors当作低层
  时间轴；Core不参与，双流直到compiler才首次相遇。
- Procedure使用三层256-wide causal blocks。compiler删除v5.1/v6的
  Core-primary AdaLN：Core只形成task-conditioned query，Procedure是进入
  dynamic LoRA slots和factor heads的唯一value/content。由此Core-only在结构
  上保持identity，避免模型重新学成language/scene驱动的通用adapter。
- 首版明确不加入null token、Action-only residual、额外视觉encoder、flow/
  geometry、7D forecast、Core到factor旁路或order loss。机械设计预算为
  `10,312,192`，比rank-128 SFT多0.144%、比v6少463,104；所有主宽度仍为256。
- 完整authority已写入
  `docs/action_forecast_writer_v7_design.md`。本段只封存设计事实；写入时尚未
  修改Writer源码/config、创建output或启动GPU进程。

## v7真实profile与首轮recipe封存（2026-07-29）

- canonical实现的真实参数为`10,312,192`。全仓192 tests通过；Core
  permutation、forward transition、`D=0→event=0`、Procedure causality、
  Core-only identity、完整rank16 LoRA、freeze与gradient合同均成立。
- GPU4–7最长视频profile中，B32与B24分别在首个functional policy forward
  仍需968MiB/726MiB时OOM；B20连续三步finite且首步确含105-frame视频。后两步
  均值为`27.477 queries/s`、`206.075 macros/hour`，峰值
  allocated/reserved为`77.020/83.647GB`（十进制）。
- 真实fresh0→1再resume1→3完整通过；checkpoint1恢复后未改写，task/video/
  query/LR/cursor身份与连续run相同，独立连续run的最大mean-loss差仅
  `2.33e-5`。joint Action–Effect binder全部`262,656`参数发生更新，step1→3
  L2位移`0.08944`，因此新路径并非断梯度。
- 第一轮只同时采用两个已有证据最强的固定选择：v6上absolute较高的
  task-complete topology，以及把v6 single-checkpoint推至143的fast cosine
  decay400。v7从identity fresh运行B20 macro0→200、每25 checkpoint；架构是
  本轮唯一新科学变量，不额外修改loss或task采样。

## v7完整结果与两个结构瓶颈（2026-07-29）

- 正式root完成fresh macro0→200与exact-resume200→400，loss总体下降且全程
  finite。macro50/100/150/200/250/300/350/400的correct400为
  `82/106/114/120/101/114/115/106`；best macro200仍只有120。
- macro200五臂为`120/112/91/100/69`。v7确实比v6更依赖时间方向，但absolute
  退步，不能把更低的无序臂本身当作成功。
- macro200的pair-logit std约`.0579`、attention entropy为理论均匀值的
  `.99963`、effective Action probes为`7.9976/8`；macro400仍为
  `.0581/.99964/7.9976`。joint `8×L` softmax没有学会Action对effect的选择性
  绑定。
- fixed Procedure只变化Core时，macro200各arm的effective-LoRA relative L2
  约`.001–.002`；fixed Core只变化Procedure则约`.145–.682`，几乎复现完整
  差异。Core query conditioning近乎失效，v7实际是Procedure-only Writer。
- 两项瓶颈到macro400未改善且absolute从120降到106；“只是不够训练”被排除为
  当前首要解释，v7停止。该结论不否定Task-Aligned Semantic Trajectory、
  Semantic Core、Causal Procedure或task-complete recipe。

## v8最小结构修正（2026-07-29）

- 每个Action anchor独立在`L`个task-token effects上softmax；raw transition
  经learned value projection与Action gate形成8个bound action–effect tokens，
  再由Procedure-only EventRead汇成每interval一个event。
- transition只在K路径新增post-difference RMSNorm。EventRead不使用Core、
  learned pooling token、null token或Action-only residual，因此
  `D=0→event=0`。
- compiler保留Core-conditioned Procedure query，并新增
  `tanh(W RMSNorm(CoreSlots))`对Procedure slots的乘性调制。没有additive
  Core residual，所以`Procedure=0→public LoRA identity`。
- 真实参数为binder`590,848`、compiler`1,469,696`、Writer
  `10,706,176`，相对rank-128 Source-SFT多约3.97%。新增参数全部对应两个
  实证缺失接口。
- 完整authority为`docs/action_forecast_writer_v8_design.md`。首轮保持
  task-complete与fast decay400不变；B20必须重新live profile，OOM或重复不稳
  才直接降B16。

## v8 B20真实profile与resume（2026-07-30）

- GPU4–7均空闲、个人空间352GB后启动。B20连续3个完整macro finite；首步
  task38/demo36明确为105 sampled frames。step wall为
  `19.243/17.506/17.450s`，后两步均值`27.463 queries/s`、
  `205.974 macros/hour`。
- 峰值allocated/reserved为`77,035,771,904/83,655,393,280 bytes`。新结构
  没有触发OOM或重复不稳，所以按预声明规则不测试B16。
- 独立fresh0→1再resume1→3通过。step1所有文件大小、mtime和SHA未变化；
  三步task/video/query/LR/cursor与uninterrupted profile相同，最大mean-loss
  差`4.7951e-5`。
- step1→3中hierarchical binder全部`590,848`参数变化，L2`.14898`；
  binding/EventRead分别`328,448/262,400`参数全变化；Core modulation全部
  `65,792`参数变化，L2`.05172`。Semantic Core、Procedure、compiler和factor
  heads也全部可达。

## v8正式结果与strict binding根因（2026-07-30）

- macro50/100/150/200/250/300/350/400的paired correct400为
  `90/110/82/110/90/125/98/115`；macro300 observed-best没有超过v6的143。
- macro300五臂为`125/121/110/110/117`。absolute和视频margin均不够，不能把
  shuffled/reversed略低本身当成成功。
- 内部反事实显示fixed Effect时改变Action只带来约`8–10%` event relative-L2，
  fixed Action时改变Effect带来约`147–300%`；Effect attention和EventRead
  entropy ratio约`97.79%/99.67%`，有效Action anchors约`7.95/8`。
- Action Expert probe是冻结policy的action hypothesis，不是teacher实际action；
  相邻视觉变化由未知teacher action产生。信息墙内没有逐interval真实配对身份，
  因此strict multiplication与8→1 EventRead把Action证据压缩成Effect-dominant
  event。v8停止。

## v10 Evidence-Preserving Dual-Stream实现与profile（2026-07-30）

- v10恢复text-only task axis与v6 Semantic Set Core；8 sparse probes形成
  Action stream，task-queried patch forward difference形成Visual-Effect
  stream，二者按`A0,V0,A1,V1,...`进入两层causal Procedure。
- compiler以Core-conditioned query读取按stream分别中心化的Procedure；
  Procedure通过bias-free `256→512→gamma,beta`提供content并门控full-rank
  Core。`D=0→Effect=0`但Action保留，`Procedure=0→LoRA identity`。
- 真实参数`11,627,520`，全仓192 tests通过。GPU4–7 B20三macro包含105-frame
  最长视频并全部finite；后两步`26.446/26.313 queries/s`、
  `198.346/197.345 macros/hour`，峰值allocated/reserved
  `77,008,402,432/83,653,296,128 bytes`。
- Core/Procedure时序建模与LoRA compilation按职责拆为唯一canonical
  `temporal.py`/`compiler.py`模块；architecture guard无hard violation、无
  parallel version/function family，数值路径与参数命名不变。
- fresh0→1→exact-resume1→3中step1未改写，task/video/query/LR/cursor一致，
  最大mean-loss差`2.6332e-6`。Text/VL/Action Meta-LoRA、Core、Action phase、
  Visual Effect、Procedure、compiler modulation/Core content和factor heads
  全部参数发生更新。

## v10正式训练、五臂与内部瓶颈（2026-07-30）

- 正式identity-fresh task-complete fast-decay run完成macro0→400，共
  `9,600`个one-video LoRA conditions、`192,000`个action queries、
  `7,832.833s`。400行metrics全部finite，24 tasks每macro等权、rank内
  long-first与teacher-video无放回cycle均通过。
- 12点paired无放回correct400为
  `25/50/75/100/150/200/225/250/300/325/350/400 =
  95/103/84/89/82/90/96/96/89/96/97/91`。macro50 observed-best只有
  `103/400`，低于corrected Source-SFT `109`，距absolute门150为47。
  Goal-6/Object-1合计贡献`86/103`；Long-2与Goal-3为0，说明不是八个tasks
  共同增强。
- 训练loss的25-step均值从`.13424`持续降到`.09651`，但online validation
  functional loss在macro50达到全程最低`.131935`，closed-loop右端也没有
  上升。继续同recipe已不符合“尚未训练成熟”的证据门。
- macro50五臂为`correct/same/wrong/shuffled/reversed=103/94/75/67/43`。
  same的paired switches为`26/17,p=.2221`；correct相对后三臂为
  `52/24,p=.001762`、`51/15,p=1.01e-5`、`68/8,p=5.63e-13`，且各有6个
  正向tasks。v10通过视频语义和顺序行为门，absolute失败不能归因于v4式
  shuffled旁路或Writer完全忽略视频。
- 内部中位relative-L2从Core/Procedure/Procedure-slots/effective-LoRA/
  policy-action依次为：

  ```text
  same      .0437 / .4050 / .5665 / .2523 / .0970
  wrong     .2029 / .4087 / 1.292 / .8832 / .1674
  shuffled  .0000 / .0873 / 1.092 / .7391 / .3481
  reversed  .0042 / .0836 / 1.346 / .8718 / .1922
  ```

  same-frame-set顺序臂的Core保持不变；fixed-Core/vary-Procedure完整复现
  LoRA/action差异，fixed-Procedure/vary-Core的LoRA差异最多仅wrong
  `.0116`。Procedure精确置零时public LoRA严格identity，所以结构硬合同成立。
- 在完整上下文中，fixed Effect/vary Action对shuffled/reversed产生
  `.6299/.8659` effective-LoRA差异；fixed Action/vary Effect只有
  `.0808/.1004`。Effect attention熵仍为理论均匀熵约`99.86%`。因此v10的
  顺序特异性主要由frozen-policy Action-hypothesis变化驱动，而不是稳定的
  Action+observed-Effect教学关系。
- correct的Procedure slots RMS仅`.0145`，经RMSNorm调制后gated-Core RMS
  达`.1781`，平均比值`14.39`；shuffled比值`20.53`。与v5.2相比，
  same-task换正确video时Procedure/effective-LoRA/action差异从
  `.0126/.1345/.0253`扩大到`.4050/.2523/.0970`。v10把正确顺序信号与
  同task示范方差一起高增益放大，造成过度video-conditional、低breadth的
  adapter；这比“Procedure信号仍太弱”更符合全部证据。
- v10与v6使用同一task-complete fast-decay合同，却由v6 single-checkpoint
  `143`降到`103`。这使架构差异成为40点退步的第一解释，并证伪“完整保留
  Action再增强Procedure→Core compiler会自然兼得absolute与特异性”。它不
  排除共享多任务优化本身仍有漂移问题，也不自动选定下一架构。
- artifacts及SHA256：

  ```text
  training audit  6701ec353433203ef89490f0fe6b179eefddaf9e304fd60c9800e204e70ff97f
  correct curve   6e9d97dcf31afdd7d867e4b3f66646db3efa68df552b625f5db2b3ba05012dfd
  five-arm        a2dbcacdfcfbe4ba2a3a9010c4c28664b2ff8ce4530c532560a24e680474be6b
  internal        df5b0271991b6ff95360b138dfe72dd7ab5daf34cc54383b92688acab539ec9f
  ```

- owner要求v10完成后暂停。Loom、其它架构、one-shot和RL均未启动；下一步是
  先共同讨论v10负结果，而不是自动实施候选。

## Loom负结果与Recenter根因级重设计（2026-07-30）

- Loom首段macro50/100/150/200 paired correct400为
  `79/106/105/112`，没有达到v5.2/v6同期水平。内部分析中raw-patch
  correspondence、teacher confidence与Teacher–Policy gap没有形成可靠、
  可解释的教学锚点；这与v7/v8已经证明的“Action Expert hypothesis不是
  teacher真实相邻动作”一致。继续调confidence、gap scale或correspondence
  权重属于围绕不可辨识变量打补丁，因此Loom停止。
- Recenter从信息需求重新推导：Core保留v6 frame-set语义结构；Procedure以
  已经验证过的原生50-token suffix mean Action为主干；task-grounded visual
  transition只能以Action RMS的25%为上限作残差，不能形成effect-only旁路。
- 新compiler只让Core提供slot地址和`[0.75,1.25]`乘性调制；LoRA主value来自
  raw time-centered Procedure。slot mixer只混合单位方向，随后恢复输入slot
  RMS，删除v10把微小Procedure提升到固定尺度的terminal normalization。
- 结构硬合同为：任意Core下constant/zero Procedure均产生零compiler content；
  Core不能单独生成adapter；Procedure按常数缩放时compiler content同尺度变化；
  step0 public LoRA仍精确identity。
- Recenter真实参数枚举为`10,709,248`，只比corrected rank-128 Source-SFT
  `10,297,344`高约4.0%。删除Loom-only correspondence/confidence/gap后，
  参数减少不是目标本身，而是不可辨识模块退出的结果。
- 实现审查发现直接对zero mean-square开平方会在`D_0=0`和zero Procedure
  mixer处产生非有限反向。最终径向分母直接使用mean-square；RMS只作detach
  诊断。slot mixer使用`torch.linalg.vector_norm`定义的物理RMS（PyTorch在
  零点采用零subgradient），只在归一化分母使用`1e-6`下限；zero-input输出和
  梯度均精确零，near-zero梯度有界，常规尺度保持齐次。

## Loom实现与启动前工程证据（2026-07-30）

- 最终Loom不是在v10上继续放大同一Procedure，而是把teacher-visible
  action-free事件与source-policy Action hypothesis分开编码，只让两者的
  confidence-bounded gap成为LoRA主内容；Core只能提供受gap约束的小幅assist。
  这直接针对v10“顺序门强但Action主导、示范方差被高增益放大”的证据。
- Writer真实参数为`12,855,552`。最长105-frame视频下B20连续3 macro finite，
  稳态`195.843 macro/hour`且峰值reserved约83.73GB，已经符合显存利用和
  long-first task-complete合同，无需扫描B21或其它无意义中间档。
- fresh0→1→resume1→3与uninterrupted profile的task/video/query/LR完全一致，
  最大mean-loss差仅`1.5891e-6`；step1 checkpoint逐文件hash不变。由此可把
  正式macro0→200的中断恢复视为已验证工程能力。
- evaluator无需新增执行路径即可并行跑四个single checkpoints和五臂；
  episode cache的遗留v4 provenance标签已改为Loom schema。

## Loom一小时结果与内部根因（2026-07-30）

- 正式Loom fresh macro0→200自然完成：200个task-complete updates、
  `4,800`个one-video Writer条件、`96,000`个action queries，wall
  `3,855.28s`，loss全程finite且未OOM。macro50/100/150/200的paired、
  无放回correct400为`79/106/105/112`；macro200是右端observed-best，但
  只比macro100/150净高`6/7`且paired不显著，主要成功集中在Goal-6、
  Object-1、Object-3。
- 相同task-complete fast-decay合同、相同macro200坐标下，Loom=`112`，
  v6=`133`，差21；相近一小时的v5.2=`132`，Loom相对v5.2的paired
  gained/lost为`32/52,p=.0375`。因此Loom没有通过一小时absolute门，不续训
  第二小时，也不做行为级same/wrong/shuffled/reversed rollout。
- macro200的8-task内部五条件检查全部finite、compiler replay误差0、
  zero-Teacher-Events时memory/confidence/scale/LoRA严格为0。Core保持
  same-frame-set顺序不变，Teacher Events的差异也能强烈传到LoRA/action；
  所以失败不是Writer完全忽略视频或数值/工程故障。
- Loom的核心新语义没有成立：
  - raw-patch matcher entropy `.991755`，mutual consistency
    `.0039218≈1/255`，visual confidence约`1.2e-6`，对应路径功能上近零；
  - correct/shuffled semantic relation RMS为`.0625/.1694`，teacher
    confidence约`.316/.455`，adaptation scale约`.232/.333`，即乱序大跳变
    获得更高授权；
  - Teacher–Policy aligned cosine在各条件仅约`.03–.10`，gap RMS约
    `1.34–1.39`、gap strength约`.729–.736`，没有形成source competence gap；
  - 固定Core/Policy只换Teacher时effective-LoRA差异可达`.208–1.097`，固定
    Core/Teacher只换Policy多数order差异仅约`.020–.024`；Teacher支配
    compiler；
  - same-task换正确视频的effective-LoRA/action中位差为`.189/.071`，
    高于v5.2的`.132/.039`和v6 fast-decay的`.069/.012`。
- 根因不是某个confidence阈值，而是架构要求positive-only AS从无共同锚点的
  visual Teacher latent与Action-Expert latent显式计算“能力差”，并学习一个
  没有负例监督的教学可信度。修matcher、反转scale或增加gate仍保留不可识别
  中央变量，因此Loom整体退役，不做局部补丁。

## Recenter第一性原理重构决策（2026-07-30）

- owner建立新的session-local Goal：持续执行
  `根因重构→一小时训练→四checkpoint correct400→未恢复旧架构则只做内部
  分析`循环；single-checkpoint达到`150`或稳定接近且多task共同贡献后，才做
  行为级视频特异性rollout。视频特异性用于证明能力来自输入视频，不以牺牲
  absolute为目标。
- v7/v8的`120/125`相对v5.2 Action-only `132`和v6
  Action+transition `143`证明：不能因Loom gap失败就删除Action value。
  下一版的根原则是`Action-anchored, teacher-corrected Procedure`：
  policy-native Action hidden是动态主干，task-grounded transition只能提供
  zero-preserving、范数受控的修正。
- Recenter保留v5.2/v6的稳定`Q_text→patch values`、frame-set Semantic Core、
  causal Procedure、320 routing slots、rank16 public LoRA和full-width factor
  heads；恢复已验证的native 50-suffix mean Action probe。删除Loom raw
  matcher、3+5 Events、teacher confidence、Policy stream与latent gap。
- compiler权力改为：Core先寻址，time-centered Procedure提供主要value；
  Core只能对非零Procedure slot做identity-init的有界乘性调制；slot
  coordination不得用terminal RMSNorm抹掉幅度，且
  `Procedure read=0→public LoRA identity`。第一轮保持v6 fast-decay
  task-complete B20不变，以隔离模型架构贡献。

## Recenter B20 profile与exact-resume证据（2026-07-30）

- Recenter没有继承Loom的硬件或恢复证据。commit `93c7e32`在GPU4–7用
  profile-only teacher seed `172`独立完成3个task-complete B20 macros；
  首步实际覆盖task38/demo36的105个stride-5帧，3/3 finite。三步wall为
  `20.5124/18.5473/18.6504s`，后两步均值为`25.8083 queries/s`和
  `193.5619 macro/hour`；峰值allocated/reserved为
  `76,989,294,080/83,644,907,520 bytes`，因此选择B20且不触发B16。
- profile checkpoint step1→3的523个Writer参数名、全部`10,709,248`个
  trainable scalars均变化；text/VL/Action Meta-LoRA、language projection、
  patch grounding、interaction projection、Semantic Core、visual transition、
  Procedure、compiler和factor heads均有真实functional-loss更新。
- 正式teacher seed `20260722`的独立root先fresh到step1，再从完整macro边界
  exact-resume到step3；metrics连续`1,2,3`，累计queries为
  `480/960/1440`、video conditions为`24/48/72`，LR与cursor连续。resume后
  step1的manifest、4个rank state、trainer和writer文件逐项SHA256不变，
  validation/test action reads均为0。
- 配置据此封存task-complete B20、fresh macro0→200、每25 macro checkpoint。
  一小时规模为`4,800`个one-video LoRA conditions、`96,000`个action queries
  和8个checkpoint；结束后只评测macro50/100/150/200的paired correct400。

## Recenter正式负结果与Core-Program根因（2026-07-30）

- Recenter macro50/100/150/200 paired correct400为`55/84/79/85`；所有
  validation tasks均低于v6 best，Object-3明显坍塌，远未恢复v5.2/v6同期。
- 内部检查显示各模块与factor持续更新、输出幅度不小，排除“未训练到”或
  “decoder没有增益”的简单解释。根因是time-centering删除Procedure DC，
  Core又只剩slot address和窄幅标量调制，导致模型只能从很小的AC残差同时重建
  task semantic basis与video program coefficient。
- constant nonzero Procedure被强制identity也删除了可用的公共高层程序。
  因此不再调整cap/gate/scale，而重构为Core semantic license × full raw
  Procedure：Core提供slot basis，Core-keyed query读取含DC/AC的raw Procedure，
  width512 bilinear严格要求二者共同产生content。
- 新架构保留`Q_text`、`M+G`、v6 Semantic Core、native 50-suffix mean
  Action、uncapped transition和2-layer causal Procedure；Core-only、
  Procedure-only与zero Procedure都严格identity。精确参数`10,905,856`。

## Core-Program真实硬件profile（2026-07-30）

- 新架构没有继承Recenter硬件证据。main `4769b36`在GPU4–7、真实最长105帧
  条件下完成B20三macro，全部finite且无OOM；后两步吞吐为
  `25.8712 queries/s`、`194.0340 macro/hour`，峰值allocated/reserved
  `76.99/83.64GB`，故B16 fallback不触发。
- step1→3间全部523个trainable tensor均有数值变化且保持finite；不是冻结、
  梯度断路或新增compiler未被functional loss触达。配置恢复formal seed后才做
  exact-resume smoke，profile checkpoint不会进入正式训练。
- 正式seed的独立fresh0→1→resume1→3验证了完整checkpoint恢复：metrics/LR、
  task-video-query cursor、optimizer/scheduler和四rank RNG连续，旧macro1文件
  没有被重写，且validation/test action reads均为0。可以从identity启动正式
  macro0→200，不使用任何profile/smoke Writer权重。
- 首段fresh macro0→200机械合同完整成立：200步finite、4,800视频条件、
  96,000 queries、8个checkpoint，训练体约64.3分钟。online validation loss
  在macro25/50/75/100/125/150/175/200为
  `.13259/.13214/.13943/.13905/.13634/.13453/.13659/.13481`；该曲线不单调，
  不能替代同一fixed correct400 panel选择。

## Core-Program correct400与内部根因（2026-07-30）

- 四个paired、无放回fixed correct400均完成并通过严格审计：
  macro50/100/150/200为`84/75/60/76`；逐task成功数依次为
  `6/2/0/39/27/8/2/0`、`4/0/0/36/27/7/0/1`、
  `2/1/0/43/10/4/0/0`、`8/0/0/40/15/12/1/0`。四点逐task
  envelope仅`95`、episode union仅`127`，并非只需选择不同single checkpoint。
- macro50相对v5.2 step900 `132`为gained/lost=`21/69`、
  `p=3.88e-7`；相对v6同recipe macro200 `133`为`23/72`、
  `p=4.76e-7`。这是明确absolute科学非通过，不续第二小时、不做行为级
  same/wrong/shuffled/reversed rollout。
- macro50的16条件非rollout检查显示，shuffled/reversed在Procedure的
  relative-L2为`.571/.775`，但到effective LoRA仅`.0288/.0446`、policy
  action仅`.00669/.00995`。Procedure DC/AC RMS为`.573/.284`；AC包含更强
  顺序变化，但raw Procedure读取让DC主导。
- constant Procedure LoRA范数约`59.3`，真实视频约`59.7`，说明raw DC几乎
  决定全部输出幅度。bilinear、fused与effective-LoRA目标的Procedure/Core
  每坐标gradient RMS约`.36`；实际Procedure方向的JVP cosine约`.94`且局部
  ratio约1，排除简单数值断路，指向大DC基线和moving Core basis的结构压缩。
- 结合Recenter约`85`与Core-Program约`84`，证据同时否定
  `Core不得直接生成任何有用LoRA`和`Core×Procedure严格双必要`。v5.2能够同时
  达到`132`及五臂`132/138/74/82/83`，直接证明Core prior与强视频因果可以
  共存。

## Prior–Innovation第一性原理重构与CPU验证（2026-07-30）

- 新authority为
  `docs/action_forecast_writer_prior_innovation_design.md`。LoRA生成被重新定义
  为`stable semantic prior + ordered video innovation`，不是对
  Core-Program增加gate/scale。
- 保留v6成功上游；compiler以routing Q/K/V/O读取Core并RMSNorm成`B`，仅用
  `B`查询保留DC语境与RoPE的raw Procedure keys，learned value只读取FP32
  time-centered Procedure；`U`与`B`直接相加，再经routing只进Q/K的完整
  Q/K/V/O residual slot block和final RMSNorm。
- `Core=0`时query为零，均匀权重对centered value严格为零；
  constant Procedure的innovation为零但Core prior保留。所有attention
  projection正常非零初始化，只有factor-head final projection zero-init，
  避免人为形成Core先学、Procedure后追。
- 活动Core-Program config/schema/class全部原位退役；唯一fresh config为
  `configs/pi05_as_writer_prior_innovation.json`，profile/formal/resume/
  gradient evidence全部清零pending，不能继承旧checkpoint。
- 精确参数为Writer`10,643,968`、compiler`1,403,904`。fresh独立验证：
  全仓`195 passed in 16.13s`、compileall和diff check通过；architecture guard
  为REVIEW但无hard violation，active source相对`a53c432`净增36行。GPU
  B20、真实BF16、gradient reachability和exact-resume尚待当前schema独立完成。

## Prior–Innovation B20与exact-resume证据（2026-07-31）

- GPU4–7以profile-only seed`172`完成B20三个task-complete macros；首步实际
  覆盖task38/demo36的105个stride-5帧。三步wall为
  `20.2923/18.5736/18.6098s`，后两步均值`25.8180 queries/s`、
  `193.6350 macro/hour`，峰值allocated/reserved为
  `76,987,188,224/83,644,907,520 bytes`，全finite且不触发B16。
- 正式seed`20260722`的独立root先fresh0→1，再从step1 exact-resume到3；
  metrics严格连续`1/2/3`，累计queries`480/960/1440`、video conditions
  `24/48/72`，LR和task/video/query cursor连续。resume前后step1 manifest、
  Writer、trainer与四rank state逐文件SHA完全相同；validation/test action
  reads与test video value reads均为0。
- step1→3的523个trainable tensors中521个数值变化，compiler、factor heads、
  Semantic Core、transition、Procedure和全部主要Meta-LoRA组都finite且变化。
  唯二未产生float32可见变化的是Action Meta-LoRA layer5 K/V的A；其配对
  zero-init B在step3已非零且finite，说明路径可达但三步内A的二阶更新低于
  float32分辨率。该事实透明封存，不冒充523/523。

## Prior负结果、rank塌缩与Target-Spectral决策（2026-07-31）

- Prior–Innovation fresh macro50/100/150/200 paired correct400为
  `100/61/89/88`；observed-best仅100，显著低于v5.2 `132`与v6 `143`。
  按预定门未续训、未补same/wrong/shuffled/reversed rollout。
- CPU跨架构复核产物为
  `/data/ymdai/outputs/ember/pi05_rank_layer_collapse_stability_cpu_20260731/analysis.json`
  （SHA256
  `d9d781bdaa8302e6dc12453ae666fd784bbac97d380db742f05a2a37117fec11`）。
  v6从macro50到600的effective BA stable rank始终约`1.0001–1.0003`；
  B列余弦约`.997–.999`，跨层q/v方向约`.969/.983+`，Core-Program和Prior
  同样塌缩。
- A行尚有明显差异，直接塌缩主因是B的16列近乎同向。same-task多视频的差异
  约90.6%位于task-mean正交方向，不是纯scale；但全部视频方差只占mean-LoRA
  能量约0.30%，说明上游视频创新被巨大的task/common写入主干淹没。
- 新canonical Target-Spectral只修复该直接瓶颈：38个真实policy targets先
  完成Core/Procedure融合，再展开16个rank coordinates；A/U分别保持row/column
  正交，16个spectral scales决定实际有效rank。模型可以选择rank1，但不能
  复制16个相同方向伪装rank16。
- 首轮不同时改optimizer。full24漂移是否来自任务冲突，待decoder实验后以固定
  macro的24-task Gradient Gram、cancellation和Adam候选伤害任务数判定。

## Target-Spectral负结果与rank病灶纠正（2026-07-31）

- fresh macro50/100/150/200 paired correct400为`30/12/18/34`。每份均为
  8 tasks×50唯一states、每task demos 0–49无放回各一次，四点state/video/RNG
  完全配对；36/36 shards、400唯一LoRA caches、worker和结果hash完整，没有
  OOM、NaN、traceback或数据错配。best macro200低于source base `48`、
  corrected Source-SFT `109`、v5.2 `132`、v6 `143`和门`150`，是真实严重
  scientific non-pass，不续训也不做行为级控制。
- CPU产物
  `/data/ymdai/outputs/ember/pi05_as_writer_target_spectral_rank_layer_cpu_aa9d89a_20260731/analysis.json`
  的SHA256为`4d7dfc68efa84b9863b8a6d9b7d4ab717f529018992b6c316c06320631d10a89`。
  Target m200把effective stable rank提高到`3.3245`，q/v分别`1.964/4.655`；
  约`15.62/16`个rank超过本模块最大scale的1%，A行和active B列coherence约
  `6e-5/1e-4`，证明伪rank16确实被移除。
- 但Target m200 effective LoRA norm仅`25.87`，v6 m200/m400为
  `94.71/108.91`；q/v能量从v6 m200的`74.5/25.5%`翻为`39.0/60.9%`，
  q/v layer-energy CV从`.047/.043`恶化到`1.294/.805`，跨层方向余弦从
  `.968/.988`降到`.032/.066`。16个同向component按16建设性相加，正交
  component只按sqrt(16)合成；理论4倍与实测范数比`3.66×`高度吻合。
- same-task视频相对中心化方差从v6 m200 `.4425%`升至Target `.6465%`，其中
  `98.2%`是task-mean正交方向；但按真实mean energy还原的视频变化RMS约
  `2.02`，低于v6约`5.96`。所谓相对视频占比改善主要来自分母缩小，不是更强
  的绝对视频写入。
- Target与v6在macro200的train functional loss为`.10023/.10043`，online
  validation loss也只差`.14020/.13751`，远不足以解释closed-loop
  `34/133`。新decoder能拟合action chunks，却产生闭环off-manifold更新；
  functional loss无法约束所需的跨层协调policy geometry。
- 无rollout内部产物
  `/data/ymdai/outputs/ember/pi05_as_writer_target_spectral_single_checkpoint_macro0200_internal_specificity_refs2_aa9d89a_20260731/summary.json`
  的SHA256为`7ddd91577f01972dc243c0871cd772847207db1fccd29306ecf8b8142824446c`，
  验证上游没有失效：shuffle/reverse在Procedure-centered为`1.349/1.525`，
  到effective LoRA仍为`.302/.352`、fixed-query action为`.0365/.0733`。
  fixed-correct-Core只改变Procedure复现这些差异，Core-only对顺序近乎不变；
  wrong-video更产生`.427` LoRA和`.091` action差异。模型对视频很敏感，但差异
  不一定有助于任务。
- 因此“rank≈1就是v6性能瓶颈”被直接否定。v6的rank冗余同时承担了有用的
  coherent gain、q-dominant分配、跨层协调和强共享归纳偏置。下一版不能强制
  full-rank正交替换它；应保留该公共主写入manifold，只把额外rank作为可选、
  zero-init的视频innovation容量。Target-Spectral只作负结果，不在其scale或
  gate上继续打补丁。

## v5.2 × task-complete控制与LoRA几何纠偏（2026-07-31）

- direct Source-SFT rank128的effective BA stable rank为`1.517`，第一奇异
  方向能量约`76.47%`；v6为`1.0003/99.97%`。低有效rank不是Writer独有。
- 以`||b_i a_i^T||_F^2`计，SFT q/v有效坐标为
  `121.7/114.2 of128`，v6为`15.96/15.97 of16`；v6坐标能量反而更均匀。
  v6 q/v rank-one pair cosine为`.716/.861`且负pair为0%，不是rank相消，
  而是等强分量和跨层方向高度建设性同向。
- A/B分量能量和符号受LoRA gauge影响，不能作为硬优化目标。可靠对照应使用
  effective BA谱/范数、q/v与target/layer分配、跨层组织、视频中心化BA变化和
  fixed-query/closed-loop action。
- 当前最重要未识别单元是`v5.2 + task-complete fast-decay400`。活动实现
  精确恢复v5.2参数`10,237,704`，不混入v6 transition或Target正交decoder；
  训练仍为full24、B20、long-first、一次AdamW、every25、fresh到macro400。
- 旧v5.2 step900与v6 macro400现存的相同16条件内部panel可严格配对：v5.2/
  v6 correct-video effective BA norm均值为`142.338/109.311`；same-task
  other-video BA absolute delta为`20.188/7.496`，其中v5.2在15/16条件更大；
  fixed-query action RMS delta为`.04842/.00646`，v5.2在14/16更大。按两独立
  视频估计的BA中心化能量约`1.251%/.2645%`。因此v5.2较强视频特异性已经
  出现在gauge-invariant BA与policy action，不只是rollout噪声。
- 旧v5.2正式LoRA cache已按仓库清理合同删除，故不能诚实补写其stable rank、
  q/v layer分布或跨层方向；如未来必须补齐，只需从step900重新生成LoRA，
  不需要rollout。新task-complete winner评测时应保留LoRA cache，直接完成
  BA谱/norm、q/v/action、layer/target、50-video task-centered innovation和
  fixed-query action对照；A/B coordinate只作同gauge次级诊断。

## Post-seal Target-Bound裁决与Semantic Factor-Basis（2026-08-02）

- Target-Bound fresh macro0→200的paired correct400曲线为`75/120/90/110`。
  macro50→100 gained/lost=`56/11`，100→150=`11/41`，150→200=`31/11`；
  winner macro100 breadth7但top2占67.5%，没有解决single-checkpoint能力轮换，故
  不续到400、不做昂贵行为五臂。
- winner macro100的8-task refs1不是“视频没传下去”：remove-A、remove-D和causal
  program-memory reversal均8/8过门，mean effective-BA relative L2分别
  `.38865/.12374/.06850`；Core-only与Program-only距full BA为`.83840/.58622`。
  correct→wrong/shuffled/reversed的BA差异为`.35971/.12228/.18813`，fixed action
  为`.27665/.15314/.24187`。最早剩余结构失败因此位于已工作Program之后的shared
  factor conditional coexistence，而不是evidence/Core/Procedure断路。
- Semantic Factor-Basis保持Target-Bound的38-target-first、private A/E/D causal
  channels和rank-last读取，只把八个共享factor MLP改为同一个Core Q/K router软选择
  四个独立value bases；route均值固定为1以不缩小hidden amplitude，keys不进入value。
  没有task ID、gate、global scale、entropy/load-balance loss或谱约束。精确参数
  `11,159,296`，source trainable为0。
- `e87363f`完成55项聚焦回归；最长105-frame、DDP4、B20三macro wall60.15秒，峰值
  reserved 83,508,592,640 bytes。macro2起frontend/Core/Program/compiler/factor五块
  梯度均finite/nonzero。formal seed fresh0→1再exact-resume1→3保持合同
  `0495a071...`，累计1,440 queries/72 videos且validation/test action reads为0。
  这些只授权`f5ddfe3` fresh0→200，不构成性能结论。

## Semantic Factor-Basis首小时与续训门（2026-08-03）

- fresh macro50/100/150/200的paired correct400为`69/91/118/127`，breadth为
  `7/7/6/8`。macro200逐task为Long-1/2=`13/2`、Goal-3/6=`1/44`、
  Object-1/3=`31/32`、Spatial-1/3=`3/1`；不是单一task独占，但主要能力仍集中在
  Goal-6和两个Object tasks。
- success-set相邻换手为50→100 `37/15`、100→150 `46/19`、150→200
  `38/29`；50→200为`68/10`。因此SFB相对Target-Bound确实形成更可信的共同累积，
  但没有消除短期能力轮换。
- macro200相对Target-Bound macro200 paired gained/lost=`40/23`，净增17；相对
  Target-Bound observed-best macro100为`34/27`，只净增7。SFB是有作用的条件分工，
  不是架构统治。
- macro200内部route task-centered/sample energy为`.2171`，task均值路由pair
  relative-L2中位`.6049`；A/E/D移除、memory reversal、Core-only和Program-only
  均证实主路径工作。与此同时晚期factor gradient share升至`.95045`、同task相邻
  CountSketch cosine降至`.06595`、task-mean/sample降至`.01292`。行为改善与该
  sketch稳定性没有一一对应，不能仅凭内部指标否定续训。
- 由于absolute接近v6同期、右端持续上涨、breadth到8且内部路径成立，按预注册门
  exact-resume 200→400；不能因未到150机械停止。下一候选variance-reduced estimator
  只检验flow Monte Carlo噪声，代码commit`1d04ae5`尚无GPU结论。

## Semantic Factor-Basis第二小时、漂移根因与VR垂直路径（2026-08-03）

- SFB 400-macro完整paired correct400曲线为
  `69/91/118/127/117/81/126/120`，single winner仍是macro200。第二小时相邻
  gained/lost为`19/29`、`16/52`、`60/15`、`20/26`；八点success union/intersection
  为`193/39`，single-best envelope gap=`66`。继续成熟化没有提高单点性能，反而给出
  本轮最强的checkpoint能力轮换证据。
- macro200相对source base gained/lost=`84/5`，所以SFB不是无效模型；相对v5.2-old
  `49/54`、v6-fast macro200 `33/39`、v6-fast macro400 `27/43`，没有形成新的上限。
  macro350达到126但与macro200 gained/lost=`31/32`、Jaccard`.6013`，相同aggregate
  不能冒充相同能力。
- 后半段201--250/251--300/301--350/351--400的raw full24 mean energy retention为
  `.04443/.04285/.04219/.04203`，candidate-negative tasks全部为0，factor share为
  `.9586/.9660/.9685/.9691`；同task successive CountSketch cosine从`.0676`降到
  `.0461/.0106/-.0099`。这更接近24个近正交条件方向加高方差sample，而不是负冲突。
- 相邻50-macro Adam一阶moment余弦在第二小时为`.0114/.0237/-.0014/-.0334`，二阶
  moment余弦却为`.9145/.9237/.9334/.9448`。优化器的尺度归纳稳定，方向持续轮换；
  降LR只缩短参数位移，未建立共同task方向。SFB router参数自身改变量很小且多段反向，
  不能再靠增加basis、entropy、gate或scale解释。
- 因absolute未达到strong门且winner仍为已有macro200，不新增1600个五臂rollout；
  macro200 refs1已证明route和A/E/D→BA→action工作，足以把最早剩余接口转到训练
  estimator/surrogate，而不是重复内部GPU分析。
- VR estimator首个真实launch暴露config method未接入`as_step`合法mode的一行工程
  缺口；它发生在首macro前，无训练结果。`50662a8`修复并增加参数化回归，4个focused
  tests通过且代码已push。
- 修复后的longest105 DDP4 B20三macro wall=`60.83s`、peak reserved
  `83,508,592,640` bytes，all finite、0 clip；formal seed fresh0→1/exact-resume1→3
  完成1,440 queries/72 videos，validation/test action reads为0。与普通SFB完全相同
  task/video assignments的前三步相比，mean gradient-energy retention
  `.11346→.13255`，same-task cosine`.26439→.29206`，factor-only
  `.27354→.29758`。这是小幅、方向正确的机制证据，分母仅三步，不支持性能宣称。
- 当前最可信下一证伪是VR fresh0→200后paired correct400。若梯度稳定性显著提高但
  absolute/breadth仍不提高，则主要根因应升级为functional action surrogate与
  source-policy closed-loop有效流形错位；若稳定性也不提高，则拒绝该estimator，
  回到完整training target设计，而不是继续修补SFB。

## Task-Relative Flow-Credit设计与BCI AS profile（2026-08-04）

- 历史PI05 Writer-RL不是policy gradient：它只保留成功rollout，再把自身成功的
  executed action prefix当监督目标做flow regression；没有failure、advantage、
  old/current ratio或trust region。其correct/wrong=`94/87`，不能作为“RL已经试过”
  拒绝真实reward credit，也不能恢复为活动路径。
- direct SFT的effective BA也接近低秩，因此“Writer LoRA near-rank1”不是漂移根因。
  Target-Owned已解除跨layer硬同向却只得99；最早剩余接口是condition如何获得对真实
  closed-loop occupancy有符号、且跨随机query可累积的policy-aware credit。
- 当前实验用同task K4 binary reward构造leave-one-out advantage。全成/全败task严格
  零梯度，避免无相对证据时制造更新；成功和失败executed prefixes均进入逐CFM sample
  old/current ratio，正项PPO clip、负项SPO pullback。functional loss只承担ratio估计，
  其数值仍不用于checkpoint选择。
- v6 fresh AS在A40六卡保持A100逻辑数据量：每macro 24 tasks、每task B20，共480
  unique queries和24 one-shot videos；B2只是物理切片。16-frame chunk覆盖105帧视频，
  峰值reserved`44.816GB`，距离46GB卡仍有有限但可运行余量，不需要减少B20或改科学
  目标。独立exact-resume证明该结论不只成立于单段profile。
- step1只有factor梯度不是上游断路，而是template-A/zero-B的预期staging；step2开始
  semantic/Core/compiler可达，step3 Program也非零。后续若reward梯度近零，必须结合
  nonzero-advantage task数和ratio证据判断，不能把AS step1现象误诊为结构断路。

## Task-Relative Flow-Credit step25 coverage与真实runtime根因（2026-08-04）

- 独立v6 AS fresh0→25完整通过，12,000 queries/600 videos、wall`810.991s`、0 clip/OOM。
  functional loss下降只证明优化健康，不用于预测reward或closed-loop。
- canonical pre-update K4得到96条完整official random-reset ledger、28,085环境动作、25
  successes。只有12/24 tasks至少一次成功；9 mixed task、3 all-success、12 all-failure。
  suite success/24 rollouts为spatial7、object7、goal10、libero10仅1。step25的主要门失败
  是exploration/cold-start breadth，而不是ratio梯度整体消失。
- 9个mixed task给出7,012个ratio samples/epoch。epoch0/1 ratio范围分别
  `[.9860,1.0174]`与`[.8902,1.0629]`，clip fraction`0/.001781`，grad norm
  `.04016/.03035`；task-relative credit可微、非零且仍在窄trust region内。这是机制通过，
  不是性能有效性结论。
- 真实environment vertical path发现两处此前静态测试未覆盖的根因：`hf-libero`
  `get_assets_path()`不读取config.yaml，必须每rank绑定runtime asset cache；torchrun
  local rank不能直接作为physical EGL ID。修复后task37与其他long-horizon task同样在
  520步落盘，run contract记录正确物理卡`1,2,3,4,5,7`。
- max CUDA reserved=`45,183,139,840` bytes，K4/Nmc4/two-epoch可在A40运行但余量有限；
  不扩大K、MC或replay batch。下一证伪严格是同一AS root 25→50后的新K4 coverage。
# Phase-Aligned K4 formal启动前工程裁决（2026-08-07）

- `fresh formal AS-Writer launch must be pushed`并非未push：当前分支HEAD与其configured upstream
  `origin/codex/bci-continuation`一致，只是旧guard错误地把`origin/main`作为所有分支的唯一比较对象。
- 拒绝发生在output root创建、模型构造和GPU工作前，没有可解释的训练或科研结果。修复只把
  pushed判断绑定到当前分支upstream，保留clean worktree与exact commit约束，不能作为跳过formal
  provenance检查的先例。

## v6-Prior B10 resume profile与verifier根因（2026-08-09）

- clean pushed`5fbcb27`在`gpu02:0--5`以4+2 NUMA完成fresh0→1+exact-resume1→3和独立
  contiguous0→3；两个正式root各3 metrics、macro1/3 checkpoints和completion，0 OOM/nonfinite/clip。
  contiguous/resumed三步step wall=`61.368/64.450s`，input wait=`.203/1.153s`；restart冷启动解释主要
  差值，steady-state input wait约`.0006s`。峰值allocated/reserved=`43.266/47.119GB`，B10容量稳定。
- macro1 row是更新前warm-start panel。到macro3 row，generated norm mean
  `140.973→138.738→136.066`、expert loss`1.79433→1.79036→1.78536`，说明norm纠偏方向正确；cosine仍
  约`.022`，三步没有解决方向对齐，ranking margin因negative schedule轮换不能作单调趋势解释。
- 两轨cursor/contract/6-rank RNG/scheduler/AMP、559 frozen Writer tensors精确相等，全部trainable Writer
  与Adam tensors通过`2e-4/2e-3`逐tensor科学门，metrics最大tolerance ratio仅`.2338`。macro3 Writer
  maxabs/relL2=`4.6033e-5/1.06393e-5`，轨迹差异是两步更新L2的`1.023%`；Adam最大绝对差仅
  `2.6865e-6`，其`.007719` relative L2由近零moment分母放大。
- 原比较器把同一`7.5e-6/1e-5` aggregate门同时用于Writer和Adam，是维度/尺度不匹配的工程
  false negative，不是resume语义或科学方法失败。离线v2门改为Writer global relative L2`≤.002`；
  Adam每个moment要求symmetric norm ratio`≥.99`、cosine`≥.999`，raw maxabs/relative-L2只诊断。
  macro3两moment的ratio/cosine=`.999632/.999970`和`.999820/.999986`；其余exact/逐tensor fail-closed
  门不变。没有为逐元素复现降低并行度、改变kernel或重跑GPU。
- retained roots经修正后的assembler完整通过并写回canonical config，formal现在合法解锁。该结果只
  证明吞吐、容量和断点语义，不证明closed-loop；最早科学未知仍是expert/ranking纠偏能否在held任务上
  把正确视频推向policy-effective方向并超过143/150。

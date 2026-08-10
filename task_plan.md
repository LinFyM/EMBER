# EMBER Task Plan

更新时间：2026-08-11。本文只保留当前可执行计划；完整历史结论见
`docs/active_session_handoff.md`实验谱系，旧命令和流水由design、Git、`findings.md`、`progress.md`及
formal artifacts保存。

## Goal and fixed boundaries

- [ ] 同一shared method/single checkpoint strict paired correct严格`>150/400`并继续提高。
- [ ] correct实质优于wrong/shuffled/reversed/no-video，same-task-other鲁棒，breadth高、checkpoint换手低。
- [x] one-shot：exact language + exactly one action-hidden video；video-only dynamic value；一套完整
  38-target rank16 LoRA；无language bypass、多video/LoRA/checkpoint融合。
- [x] fixed 24/8/8 split、frozen source/normalization、validation/test action隔离和official paired evaluator。
- [x] task experts只作历史privileged policy-effective参考；当前native compiler与deployment均不读取其输出。
- [x] GPU launch前live比较`gpu01/gpu02`，选择一个节点并使用该节点所有真正空闲、健康且能提高有效吞吐的
  A40；没有固定6卡上限，不等待凑卡、不dummy占位、不为跨节点碎片改launcher。训练多卡遵守
  `NCCL_P2P_DISABLE=1`、NUMA映射和deferred-NCCL；独立评测按所选单节点live空卡动态扩展queue。
- [x] 删除canonical evaluator/preflight的旧owner-six-GPU硬上限；受单节点config（当前8卡）、unique index和
  live ownership约束，7/8卡均可用。定向环境回归`51 passed`；训练world-size泛化留到fresh训练获行为授权后，
  exact-resume仍锁原topology。
- [x] 吞吐优先：接受普通BF16低位差异，不为逐元素复现固定batch1、重复forward或扩宽LoRA cache。

## Phase 0 — authority and throughput correction

- [x] 撤回batch1/`1e-5` direct reproduction gate；Writer generation默认至少batch8，最终取profile最优值。
- [x] LoRA cache保持72 BF16 + 4 F32原生dtype，batched D2H单次同步。
- [x] 历史source-action functional图的B20/B16/B10只作旧图容量provenance；active Reward-Credit live B2
  实测仅`16.34/19.42GB` allocated/reserved，故重profile直接上调Nmc4 replay physical B8，不为数值微差缩
  K/Nmc/dtype；短replay允许B8自然退化为一次完整batch。
- [x] 合并correct effective alignment、task metric和gradient norm host transfer，移除重复finite scans/sync。
- [x] action DataLoader改为2 spawn persistent workers + prefetch2，并验证serial/prefetch/resume rows一致。
- [x] 最终相关定向`68 passed`、全仓`227 passed`、compileall和`git diff --check`全部通过。
- [x] 完成当前authority、README、design、findings/progress一致性，吸收并行只读审计结果。
- [x] 真实validation8×4-state asset inspector和CLI prepare：32 requests、600 Writer tensors、native
  72 BF16 + 4 F32 cache、deployment expert-bank reads=0；临时root已清理且未初始化CUDA。
- [x] clean commit/push；清理已被新提交覆盖且无唯一改动的旧smoke worktree/local branch。

## Phase 1 — single A40 throughput and vertical smoke

- [x] 使用`gpu-preflight`实时检查两节点和`/data1` quota，选择一张完全空闲A40。
- [x] 在同一loaded historical macro400 Writer上，用同一个32-request/同一总帧数的longest-first panel
  profile真实异长video batches `8/16/32/...`；只改变forward分批，记录LoRAs/s、repeat wall、actual
  forward batches、peak allocated/reserved和headroom。
- [x] 三候选repeat稳定且同panel wall近似；没有为了未出现的瓶颈增加同步型instrumentation。
- [x] batch8实测吞吐最高；batch16/32没有上升趋势，未因显存空余选择更慢配置，也未按低位数值选择。
- [x] 用选定batch从fresh root完成validation8×state0 correct smoke：8 videos/LoRAs/cache/rows，Writer
  release、source reuse、0 forbidden reads/retry/failure/OOM/nonfinite，GPU自然释放。
- [x] 把精确device/root/commit/batch/wall/peak/cache dtype/release evidence写回config，将六卡profile解锁。
- [x] artifact-backed assembler从profile与vertical roots生成evaluation seal，没有人工拼evidence/status。

## Phase 2 — six A40 gradient/resume/throughput profile

- [x] 实现并CPU验证gradient artifact assembler：从retained contract/profile/completion/invocation重算
  weights，精确核对canonical config、clean pushed Git、六卡NUMA拓扑、480 unique queries、24-task
  deterministic correct/negative/video schedule、frozen manifest/HDF5 frame metadata、显存/等待时间和
  零OOM/nonfinite；status本身不能解锁下一阶段。
- [x] 实现并CPU验证fresh/resume/contiguous assembler与只读checkpoint inspector：核对Git phase ancestry、
  contract/cursor、600 Writer tensors、41 trainable tensors、6-rank RNG、Adam moments、scheduler/AMP和
  三宏步scientific metrics；接受预注册parallel roundoff而非逐bit一致。
- [x] 首次live比较两节点并选择当时空闲`gpu01:0,1,2,4,5,7`的3+3 NUMA拓扑；两次physical B20均在
  PI05 policy MLP容量OOM后自然释放，未触碰他人GPU3/6。后续每次launch仍必须重新live选择。
- [x] 用默认allocator和唯一一次`expandable_segments:True`重试排除碎片主因；失败roots仅有contract/
  invocation，无gradient/completion，禁止resume、合并或冒充科学结果。
- [x] 实现完整有序logical-panel keyed的logical-B20 microbatch：20条独立time/noise、每task mean、
  480 unique queries和objective分布不变，FP32 leaf-gradient按真实slice权重累积；轻量整数seed mix不使用
  SHA/MD5，policy checkpointing保持关闭。
- [x] clean pushed `eddba96`并创建frozen worktree；live比较两节点后只用空闲
  `gpu01:0,1,2,4,5,7`，实际run contract封存3+3 NUMA和local/physical映射。
- [x] physical B16+4完整启动后在第一条functional attention统一OOM：allocated=`42.49GiB`、
  reserved-unallocated=`1.25GiB`、free=`235.31MiB`，尚需`254MiB`；因此没有B16吞吐点，也不再做
  allocator retry/A-B-A。
- [x] 从只改两处microbatch `16→10`的clean pushed `9c814ff` frozen worktree运行balanced B10+10完整
  macro49；wall=`21.095s`、input wait=`.076s`、peak allocated/reserved=`40.332/43.859GiB`、0异常。
- [x] 固定macro49覆盖train24×B20=480 unique queries和最长105-frame video；记录positive/expert/ranking
  对compiler/factor的未加权gradient norms。
- [x] 一次性封存`lambda_expert/lambda_rank=.0083551721/.2857046689`，两个blocks上各auxiliary均不超过
  positive的`.25`；不按held
  outcome sweep或在线自适应。
- [x] 对正式gradient artifact运行assembler，再把其原样证据写回gradient+aux并置为profile-ready；
  不人工拼weight/evidence，不从外部复制config绕过canonical tracked config。
- [x] B10 input-wait share仅`.36%`，不测试workers4；最长panel完整通过且active余量约`4.09GiB`，保持B10，
  不预防性降低microbatch或开启policy checkpointing。
- [x] clean commit/push gradient seal并创建严格后继frozen worktree；同一worktree、同一六卡拓扑已完成
  resumed root fresh0→1 + exact-resume1→3，以及独立contiguous0→3。gpu01首段后设备ownership变化，
  因此正式retry1在重新live确认的`gpu02:0--5`以同一4+2 NUMA拓扑重建完整两链，未混用partial root。
- [x] 依据profile retained `input_wait/step wall/peak VRAM`判断后续瓶颈；data wait在steady state约`.0006s`，
  B10峰值allocated/reserved=`43.266/47.119GB`，当前不再实测workers/prefetch或policy checkpointing。
  只有未来新证据要求时才重新profile这些变量。所有
  候选保持logical B20/full24 scientific batch，不为“多记录阶段”在热路径增加CUDA同步。
- [x] 丢弃型权重完成fresh0→1、same-root exact-resume1→3、independent contiguous0→3；验证cursor、RNG、
  optimizer/scheduler和scientific metrics语义，接受正常parallel roundoff。
- [x] 对三条正式roots运行assembler并原样写回证据，封存profile/formal-ready；原aggregate tolerance把
  近零Adam moments误用Writer门，离线v2比较现按Writer relative state与Adam moment norm/direction
  分层判断，没有改训练路径或重跑GPU。
- [x] clean commit/push当前profile seal并创建formal frozen worktree；profile checkpoint永久禁止进入formal。

## Phase 3 — formal v6-prior continuation（已裁决）

- [x] clean pushed frozen worktree、fresh formal root，从historical macro400 load-only创建全新optimizer等状态。
- [x] 用当前paired schedule跑method macro0 baseline；旧`143`只作历史参照，不能替代同schedule baseline。
- [x] 训练0→50，保存10/25/50；持续记录per-task三项loss/gradient、effective BA、margins、full24 retention、
  data/GPU wait、step wall、clip/nonfinite和完整resume state。
- [x] 从同一correct400 root派生correct80；macro0/10/25/50全部完成paired correct400，防止functional loss
  与真实性能错位。
- [x] 每点和macro0、历史143、v5.2 old、v6 old/task-complete做per-task/per-suite/breadth/gained-lost/
  churn和内部transfer对比。
- [x] 曲线`134/127/105/123`未形成共同上升，按门停止，不续100/200、不扫aux权重。
- [x] single winner未超过macro0，故不为loser补完整六臂；保留macro0历史五臂参照。

## Phase 4 — v6 Expert-Component Projection

- [x] 在唯一canonical objective中用`a=<G,E>/||E||²`替换whole-LoRA direction/norm；correct SmoothL1到1，
  negative只做bounded projection margin；不加shadow macro0 branch、residual retraction或新forward。
- [x] CPU锁定低秩有效BA恒等式、finite/broadcast、projection gradient不直接压缩expert-orthogonal分量，
  同时保持macro0 load、no-video identity、信息墙和functional path。
- [x] 新config/schema/metrics明确记录`a_correct/a_negative/a_margin`、generated norm和per-target贡献；旧v1
  由Git/formal artifact保存，不保留并行可执行objective。
- [x] clean push/frozen worktree，live双节点/存储preflight后只做一次B10六卡gradient profile，按同`.25`
  compiler/factor budget封存两个aux weights；不扫权重。
- [x] strict后继clean frozen worktree完成fresh0→1、same-root exact-resume1→3和独立contiguous0→3；
  assembler封存contract/cursor/RNG/optimizer/scheduler/Writer语义、吞吐和显存证据，正式训练已解锁且profile
  checkpoint永久弃用。
- [x] 在canonical analysis owner新增严格cross-family historical-baseline transition：legacy/current分别
  native验证、400 rows精确配对、family标签不变、不能冒充checkpoint curve；全仓`262 passed`。
- [x] fresh formal只训练0→10并停止，直接跑macro10 correct400；历史macro0=`134`用native-family严格
  cross-family transition比较，先不为schema身份重复400条rollout。
- [x] fresh短训保存10/25。macro10 strict若`≤129`且多task净损失则停止；130--133只有内部方向和右斜率
  健康才允许到25；macro25若仍不超过134或只是换手则停止。
- [x] macro10/25=`133/120`；macro25对macro0严格paired gained/lost=`13/27`、net=`-14`、
  `p=.038477`，按门退役ECP，不继50/100、不扫权重、不补六臂。

## Phase 5 — targeted iteration loop

- [x] 完成condition-local frozen-v6 dynamic baseline的anchor/tangent历史去重与数学design；首轮只隔离
  expert completion和expert-orthogonal drift，不同时改encoder、topology、functional estimator或video schedule。
- [x] 在唯一canonical path原位实现两臂condition mean：不物化dense BA、不在部署时增加第二LoRA/
  专家库、只复用已有memories增加correct/negative各一次小型frozen decoder
  forward。exact-D三状态oracle、独立gauge、macro0/parallel/orthogonal、双臂mean、chain rule、same-memory、
  trainable-only resume/deployment和三family分析均已封存；全仓`276 passed`、compileall与diff-check通过。
- [x] clean pushed/frozen`2616773`在live空闲`gpu01:0,1,2|4,5,7`完成唯一六卡gradient/whole-macro
  profile；B10+10 wall=`21.531s`、0 OOM/nonfinite，assembler写回唯一权重/evidence，未因BF16低位
  一致降低batch或并行度，也未启用无收益cache。
- [x] strict后继clean pushed/frozen`c1bdcae`已完成fresh0→1、same-root exact-resume1→3和independent
  contiguous0→3；assembler复核cursor/RNG/checkpoint/Writer/Adam和普通reduction误差后已原样写回evidence，
  formal解锁且profile checkpoints永久弃用。两轨step wall=`62.341/61.959s`，0 OOM/nonfinite。
- [x] 保留profile科学预警而不提前换路线：macro3 correct/negative tube median约`.0316/.0317`、directional
  ratio约`61×`且发生clip，尚未过mechanism门；这使macro10的tube recovery成为硬续训条件，不是工程
  resume blocker，也不授权weight/LR sweep。
- [x] 对config、tests和8份authority完成CPU seal；clean pushed`b308941`严格后继于`c1bdcae`，并建立
  独立tangent formal frozen worktree。live比较双节点GPU和`/data1` quota后，fresh root与launch contract
  通过正式门；旧v6-prior worktree未被复用。
- [x] formal fresh0→10与strict correct400完成：训练exit0、0 OOM/nonfinite；macro10 tube半径两臂
  `24/24`过`.03`，但directional两臂`0/24`且completion`0/24`；strict=`131/400`、breadth5，
  相对macro0 gained/lost=`16/19`。按门不续25、不补六臂、不扫weight/LR/WD。
- [x] 退役当时的Tangent recipe且保持解释边界：它控制了总半径但没有完成expert方向写入；由于
  `|a_correct-1|≤.05`门未到，不把负结果扩大成expert-component假设整体无效。config/runtime已formal
  non-pass后fail-closed。
- [x] 实现并CPU封存第36节matched no-update Expert-Flow Teacher Viability Audit：唯一CLI mode保持train24、
  logical B20/physical B10+10和相同action query/noise/time；每task 6次PI05 forward，真实7维四类loss转FP32，
  full24等权gradient与Gram pinv固定`rtol=1e-5`。0 optimizer/scheduler/update/rollout、8/8/8 negatives和
  near-collinear span均有oracle；当时全仓`284 passed`。
- [x] audit从clean frozen`e8e4728`完成：480/480 queries、144 forwards、0 update/rollout/OOM/nonfinite；
  expert/macro0/tangent loss=`.098631/.091802/.091843`，teacher仅`2/24` tasks、`0/4` suites过门。gradient
  residual=`.6864/.8387`虽非冗余，但teacher更差，因此CEFD正式否决，不做weight profile/训练/换step。
- [x] 按第36节触发删除一次性teacher-audit mode、flow-teacher/audit owners及feature tests；保留canonical
  run-contract owner、Git和formal artifacts，旧config formal non-pass/fail-closed。
- [x] 实现第37节Frozen-v6 Counterfactual-Null Condition-Kernel Program Residual：冻结historical v6全部
  600 tensors，P256×320×256 FP32 memory从零开始；fixed zero-preserving temporal feature、24 correct
  functional cotangent+24 counterfactual zero-motion rows、FP64小Gram/FP32大write，无Adam/expert/ranking。
- [x] CPU门覆盖step0 macro0 identity、base frozen、真实order feature、full48 predicted/observed、negative-null、
  A/B、checkpoint/resume、moving-authority exact-resume、deployment family seal和0 forbidden reads；聚焦回归、
  artifact gate现从raw macro重算profile并绑定完整科学run，formal result必须绑定completion/50-row metrics/
  macro10/25/50 manifests，deployment checkpoint必须位于active authority lineage；compileall、diff-check与
  全仓`280 passed`完成，architecture guard无hard violation。
- [x] clean pushed/frozen`6903ee6`完成第37节v1唯一macro49 profile：10/13门通过，correct retention=
  `.807966`且24/24、A/B/action/closure均成立；但condition=`1315.33`、negative/correct=`.264351`、
  null仅15/24，shuffled feature cosine mean `.98552`且只2/8过门。production ratio=`1.115458`也未过
  1.10。按门不训练、不降lambda、不扫seed/P/threshold，v1由Git和无checkpoint artifact封存。
- [x] 依据同profile和历史phase16 causal-prefix证据，只把canonical key原位升级为第38节v2：video-DC
  static与centered sqrt-causal-prefix dynamic分别fixed-JL到128、各自zero-L2后拼成P256；frozen v6、memory、
  full48、`.01` damping、B20/B10+10和0 negative forward全部不变。v1 config/code不留活动并行路径。
- [x] v2 CPU门新增同static/反dynamic两帧反例，natural/reversed unit key内积0；聚焦`52 passed`、带
  LIBERO assets最终全仓`281 passed in 21.34s`。projection降到`[2,128,256]`，并移除per-condition GPU
  sort/mask同步、把profile-only bookkeeping/zero allocation移出production timer，不降精度或batch。
- [x] clean pushed/frozen`5d93434`经live双节点/quota preflight后，在空闲`gpu01:0,1,2|4,5,7`完成唯一v2
  macro49 profile：13/13门通过，rank48、condition=`106.114`、correct/cotangent=`.968254`、negative/
  correct=`.0218514`、24/24 correct与24/24 null、4/4 fixed-action、closure=0。production=
  `20.021842s`、ratio=`.949122`，峰值allocated/reserved=`43.261/46.917GB`；0 OOM/nonfinite/negative
  forward，profile权重未保存且六卡释放。artifact由raw结果重算并写回config seal。
- [x] clean frozen`2af82aa`在实时空闲`gpu02:0`完成新residual deployment graph双root seal。同一32-request/
  1093-frame panel的batch8/16/32=`.911238/.901898/.906482 LoRA/s`，均稳定且有约32.4GiB headroom，选batch8；
  validation8×state0 correct真实闭环8/8 rows、`4/8` success、单次launcher、8套native LoRA、0禁止读取，
  assembler通过且GPU释放；写回后全仓`284 passed`。`4/8`只作执行证据。
- [x] `d228d0d` frozen CPU-only formal prepare在0 CUDA/0 row时发现`runs`软链接artifact被误判越界；
  `af7b101`以canonical `runs/outputs` containment窄修复并保留nested-symlink fail-close。全仓`285 passed`，
  clean frozen prepare exit0并封存8×50、correct/no-replacement、zero residual macro0、18 rollout workers +
  18 Writer generators、batch8合同；临时root已清理，正式评测必须使用新root。
- [x] clean frozen`6b5f7a6`完成zero-memory macro0 strict400：`134/400`、breadth6、per-task=
  `0/5/48/34/0/35/11/1`；72/72 shards、400 fresh LoRAs、18 rollout workers + 18 Writer generators、
  0 retry/error。与历史native
  macro0 400-row identity和success逐行完全相同，gained/lost=`0/0`；CPU重聚合通过且GPU释放。
- [x] clean frozen`abd8e08`完成v2 formal fresh0→25与macro10/25 strict correct400：曲线
  `134/140/139`、breadth均6；macro0→10 gained/lost=`19/13`，macro10→25=`12/13`。没有超过历史
  `143`或目标`>150`，按门退役v2，不续50、不补多臂、不扫P/lambda/eta、不解冻base。macro25内部
  72/72 jobs、400 rows、18/18 worker return0完整，但外层wrapper exit只能记为unobserved/missing。
- [x] LoRA/Program与same-task 50-video诊断完成：effective delta/base median=`1.69498e-4`、stable rank
  `1.000022`、top1 energy=`.999978`；不同正确视频raw correction consistency=`.141539--.142175`，
  effective pair cosine近0。视频路径非零，但最早瓶颈更符合跨macro retention/reconciliation；不把该结果
  写成视频因果性证明，也不据此直接切few-shot。

## Phase 6 — Exact Anchored Reconciliation（已裁决）

- [x] 完成第39节最小设计：部署图、one-shot信息墙、balanced v2 `phi256`、frozen v6 decoder、
  `M[256,320,256]`和完整rank16 LoRA均不变；只在训练/checkpoint加入FP64 `Lambda[256,256]`与
  `assimilated_rows`，用anchored RLS累计旧约束，fresh与v2 checkpoint不兼容。
- [x] canonical v3实现、checkpoint Program/precision隔离、首步blind ridge与累计direct-solve CPU oracle、
  zero-cotangent assimilation、old/current/blind diagnostics和focused regression已通过；纯FP64 oracle误差
  `6.1e-16/3.5e-14`不冒充runtime逐bit误差，也不扩dtype或降低batch。
- [x] 封死正式状态机：fresh0→10必须预注册唯一macro0/macro10 strict roots；formal evaluator只接受
  predeclared macro10/25；10→25会从immutable shards重聚合两份400-row paired panel并核对commit、checkpoint、
  state/RNG/language/video identity，只有`correct>=140`、`lost<=6`、breadth`>=6`才允许exact-resume。
- [x] focused=`75 passed`、加载`.env.local`后全仓=`300 passed`、compileall/JSON/diff-check通过；
  architecture guard为review且0 hard violation，authority已同步v2终局、RLS合同与条件macro25。
- [x] implementation/contract已clean commit并push为`f0c3f51`，从该提交创建独立detached profile frozen
  worktree；主分支、远端和frozen checkout均精确一致。
- [x] live比较`gpu01/gpu02`与`/data1` quota后，在空闲`gpu02:0--5`完成首次fresh0→3 disposable profile；
  exit0、0 checkpoint/OOM/nonfinite/negative forward。原18门16/18通过，RLS old-row/current-motion核心门
  全过；首步`1.97e-5`低位差和单步fresh-vs-warm wall门失败，旧artifact诚实保留non-pass。
- [x] 按第39.4.1节只修正证据合同：首步ppm ratio改为diagnostic，精确数学由CPU oracle负责；吞吐改为
  三宏步production arithmetic mean对原baseline仍`<=1.10`。架构、数学、batch、dtype、worker和forward
  数均不变，旧artifact不得重解释。
- [x] 新合同focused=`82 passed`、全仓=`300 passed`，compileall/26 JSON/diff-check通过；旧artifact在新
  config下仍明确拒绝，证明不能被post-hoc重解释。
- [x] 测量合同修正已clean commit/push为`f28fc8b`并创建独立profile frozen worktree；live空闲
  `gpu02:0--5`上的全新fresh0→3 natural exit0，17/17通过，0 checkpoint/OOM/nonfinite/negative forward。
  三步production mean ratio=`.952297`，raw artifact=`100452B`且completion passed；旧`f0c3f51`仍保持non-pass。
- [x] config已登记f28 immutable evidence并转为`active_deployment_sealed_formal_ready`；通用historical
  transition登记RLS macro10/25并拒绝50，evaluator在创建目录前硬绑定training run contract预注册的macro10
  root。相关定向回归`70 passed`。
- [x] profile seal/root-binding后定向=`70 passed`、全仓fresh=`304 passed in 92.19s`，compileall、26 JSON、
  real config/artifact load和diff-check通过。
- [x] clean commit/push并创建新的formal frozen worktree；formal identity fresh0→10后立即跑预注册macro10
  strict400，不过支持门即停止。
- [x] macro10 strict=`140/400`、breadth6，但相对macro0 gained/lost=`21/15`；`lost<=6`失败，故未
  exact-resume10→25、未补六臂。correct80虽为`5/0`，已由full400证明不能用于选点。
- [x] 和blind-v2 macro10逐row比较：两者同为140，v2→RLS换手`17/17`；RLS相对macro0的lost从v2的13
  增至15。由此正式否定feature-row anchoring足以解决held closed-loop retention，config/runtime fail closed。
- [x] Phase6没有超过历史143，按证据回到最早失效接口；当前统一control规则由Phase8定义为首次≥144补
  同checkpoint controls，严格>150必须完整六臂，不能从本历史phase恢复旧“达到goal才测controls”表述。

## Phase 7 — Reward-Credit Program Cotangent（formal cycle1与strict已裁决）

- [x] 保持Balanced-v2 one-shot部署图、exact language + exactly one action-hidden video、frozen v6 decoder、
  P256、single Program和完整rank16 LoRA不变；只把offline source-action cotangent替换为train24真实闭环
  binary-reward credit，禁止language bypass、few-shot、progress reward、SPSA和第二LoRA。
- [x] 从历史Task-Relative Flow-Credit中只复用已验证的K4 task-relative binary LOO、executed-prefix
  Nmc4 functional estimator、failure样本、runtime asset/EGL/NCCL修复；删除old/current第二forward、第二epoch、
  shared Adam和success-only replay，不恢复已退役RL路径。
- [x] clean frozen`c4507e9`完成首次full24×K4×Nmc4 B2 profile；24/96、11 mixed/13 homogeneous、rank48、
  closure0与runtime健康，旧门唯一因固定`0/7/14/21`中三个homogeneous action-null而失败。旧root保持
  immutable/non-pass，不事后seal。
- [x] 把profile v2改为all-mixed穷举：K4真实首query+原始noise、before/after batch4、raw per-task
  Program→LoRA A/B→action全非零与四suite覆盖；B8 physical replay和静态cost-balanced one-suite-per-rank map
  只优化执行，不改full24等权样本或数学。
- [x] 新合同全仓`338 passed in 37.69s`，compileall、27 JSON、Black、diff-check和architecture guard通过；
  0 hard violation、无parallel family，旧artifact在v2 raw gate下仍fail。
- [x] clean frozen`e6024cf`与live双节点/quota preflight后，在全新root完成唯一
  full24×K4×Nmc4 B8 fresh0→1 discarded profile；raw v2 gates全部通过，physical batch唯一反推为8，
  0 runtime fault且无checkpoint。
- [x] 将profile raw/run/completion/invocation、唯一B8公式、A40 topology和无checkpoint纳入fail-closed seal；
  active config置为formal ready，未复用discarded state。
- [x] 收口formal evaluator启动门：selected Writer batch精确为B8；Reward-Credit绑定validation 8×50、
  without-replacement、同一clean pushed/frozen commit、checkpoint及预注册root；删除historical bypass；correct
  达到阈值后才开放controls且与support门解耦；prepare使用NFS原子准备锁、私有staging和一次目录rename，
  失败不占用canonical root；正式Reward只接受预注册六臂。全仓`358 passed in 23.72s`，architecture guard
  0 hard violation、无parallel family。
- [x] 从clean pushed`e3857f7` frozen worktree完成formal fresh cycle0→1与strict correct400；训练natural
  exit0、B8、0 OOM/nonfinite，strict仍为`134/400`、breadth6、相对macro0 gained/lost=`14/14`。支持门失败，
  不续cycle2、不补controls、不扫reward scale/K/Nmc/RLS参数。
- [x] 分层诊断定位到最早失效接口：Program/video与continuous FactorHead/effective-BA tangent均健康；q/v
  factor delta约`1e-8 RMS`，加到非零BF16 A/B后低于约`1e-4` ULP。FP16、dither、local-CD、gauge/scale、
  absolute refactor均失败，不把closed-loop non-pass误写为Reward方向已被证伪。

## Phase 8 — Q/V Rank-Reserved Native Reward Compiler（active）

- [x] 写入独立design authority；保持exact language + exactly one action-hidden video、P256、frozen v6、single
  Program、38 targets/public rank16和action full-rank16 FP32，只把36个q/v targets改为pivot-preserving
  rank14 base + condition-local rank2 physical zero-B residual。
- [x] full80 generation-only门通过：q/v base error约`.0007523`、task max≤`.001302`，rank2 capture
  `.9997088`，dynamic cosine`.9975247`、video-centered`.950556`，action exact；0 policy forward/rollout，
  不冒充性能。
- [x] 在一个canonical owner中实现load-only native compiler、新family/config/schema与derived Program reference；
  旧Reward cycle1 checkpoint只读取原84MB Program，不resume optimizer/RNG/precision，也不复制大tensor。
- [x] CPU门覆盖deterministic pivot、kept-B bit exact、rank14 solve、compact top2、stable BF16 tangent、
  zero-Program后2个B exact zero、action exact、76 tensors/native dtype、old schema/cache fail-close和信息墙。
- [x] 旧Reward fresh/profile/resume/cycle2在CLI/training/runtime三层初始化前机械fail closed；active config仍为
  `awaiting_live_a40_rank_reserved_deployment_profile`且`online_smoke_evidence=null`，不冒充formal ready。
- [ ] live单卡做同32-request B8/16/32吞吐profile并选samples/s最高点；同一cycle1 vertical smoke比较五臂
  （含q/v-only hybrid）的四suite batched fixed-action。profile更大候选OOM只作ineligible；vertical明确区分
  configured winner与8-entry actual cache batch，full/q/v-only使用cache-loaded q/v，paired base从同一state
  清零last2 slots。要求真实adapter/cache、Program q/v/action response、Writer release与0 nonfinite/
  forbidden read，不以逐元素微差门禁。
  registered roots分别是`runs/outputs/pi05_v6_qv_rank_reserved_native_reward_profile_b8_b16_b32_20260811`和
  `runs/outputs/pi05_v6_qv_rank_reserved_native_reward_vertical_four_suite_20260811`。
- [x] 实现纯CPU `rank-reserved-seal` assembler，只从注册profile/vertical raw artifacts写回status/evidence；
  tracked Program asset resolver与`runs/outputs` evidence resolver分离，Gate C不再机械阻断。seal只允许在
  tracked canonical branch head执行；Gate A后必须seal commit/push并新建frozen worktree，才进入Gate B/C。
- [ ] 先跑新rank14 zero-Program strict400；若correct<130、breadth<6或相对旧134 lost>10则reject，不浪费
  第二个400。旧macro0 rows、source policy、split、video和RNG schedule直接复用，不重跑旧baseline。
- [ ] 只有新macro0过门，才跑现有cycle1 Program的rank14+2 load-only strict400；只有correct≥144、
  breadth≥6、相对新macro0 lost≤6且gained>lost才算通过并补同checkpoint controls。140--143为诊断性
  non-pass且不授权新训练；>150完成六臂。
- [ ] 两个行为门前不实现或启动fresh训练。若load-only通过，再单独设计native-forward + continuous VJP/STE；
  不对zero tangent做SVD backward，不扩dtype、不降B8。

## Ongoing evidence-driven iteration rules

- [ ] 若absolute升但video margin弱，只改counterfactual credit/Procedure temporal objective。
- [ ] 若margin升但absolute降，诊断ranking伤害policy；不写成训练不足。
- [ ] 若expert alignment升而held下降，重构/减弱train-expert流形监督；不恢复online expert bank。
- [ ] 若Procedure信息存在但BA传递弱，只改compiler ownership/topology。
- [ ] 若BA/action传递健康但task换手，只改task aggregation/credit coexistence，不靠scale/rank。
- [ ] 若same-task跨video方差被定位为最早瓶颈，才设计固定K few-shot invariant aggregation。
- [ ] 只有下游目标、梯度和传递均成立而上游没有正确时序语义，才解冻或重构Procedure/encoder。
- [ ] 每轮先写最小可证伪design，profile、充分训练、及时strict测试、和历史谱系逐项对比，再决定下一轮。

## Completion

- [ ] 至少一个single checkpoint strict paired correct`>150/400`。
- [ ] 完整paired五臂/六臂证明真实video/task/temporal causality，不是text bypass或低层伪特征。
- [ ] breadth、checkpoint churn、repeatability和per-task得失达到可接受水平；达标后继续探索更高absolute。
- [ ] formal artifacts、机制分析、当前authority和Git clean pushed；仅在Goal真实完成时标记complete。

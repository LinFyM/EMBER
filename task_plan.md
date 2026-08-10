# EMBER Task Plan

更新时间：2026-08-10。本文只保留当前可执行计划；完整历史结论见
`docs/active_session_handoff.md`实验谱系，旧命令和流水由design、Git、`findings.md`、`progress.md`及
formal artifacts保存。

## Goal and fixed boundaries

- [ ] 同一shared method/single checkpoint strict paired correct严格`>150/400`并继续提高。
- [ ] correct实质优于wrong/shuffled/reversed/no-video，same-task-other鲁棒，breadth高、checkpoint换手低。
- [x] one-shot：exact language + exactly one action-hidden video；video-only dynamic value；一套完整
  38-target rank16 LoRA；无language bypass、多video/LoRA/checkpoint融合。
- [x] fixed 24/8/8 split、frozen source/normalization、validation/test action隔离和official paired evaluator。
- [x] task experts只作train24 policy-effective监督，不进入deployment。
- [x] GPU launch前live比较`gpu01/gpu02`，只用最多6张空闲A40；多卡遵守
  `NCCL_P2P_DISABLE=1`、NUMA映射和deferred-NCCL。
- [x] 吞吐优先：接受普通BF16低位差异，不为逐元素复现固定batch1、重复forward或扩宽LoRA cache。

## Phase 0 — authority and throughput correction

- [x] 撤回batch1/`1e-5` direct reproduction gate；Writer generation默认至少batch8，最终取profile最优值。
- [x] LoRA cache保持72 BF16 + 4 F32原生dtype，batched D2H单次同步。
- [x] functional单物理batch直接梯度与真实multi-microbatch FP32 accumulation均有唯一实现；A40当前图已
  依次证伪physical B20和B16，active执行保持logical B20并使用balanced physical B10+10。
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
- [ ] 两类profile均seal后先评测zero-memory macro0 strict400，再fresh0→10并立即strict correct400；按第37.6/38.3门决定
  停止、续25或补六臂，不用correct80 screen、不扫P/lambda/eta、不解冻base。
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

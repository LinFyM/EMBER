# EMBER Progress Ledger

## Reward-Credit cycle1终局与rank-reserved native后继（2026-08-11）

- clean pushed/frozen`e3857f7`完成formal Reward cycle0→1；root=
  `runs/outputs/pi05_v6_reward_credit_program_cotangent_formal_cycle0to2_r6_k4_nmc4_b8_balanced_20260810`，
  completion macro1、checkpoint/manifest完整、natural exit0。预注册correct root=
  `runs/outputs/pi05_v6_reward_credit_program_cotangent_correct400_cycle0001_20260810`，400 rows、success134、
  breadth6、相对macro0 gained/lost=`14/14`。按门终止cycle2。
- 完成stable analytic tangent、native factor representability、FP16、dither、local-CD、balanced rank reservation
  和q/v pivot-preserving full80诊断。前两次pivot脚本只在JSON aggregation阶段失败、没有科学artifact，空root/
  log/exit已删除；成功artifact和日志保留，所有诊断GPU均自然释放。
- generation-only winner为q/v rank14+2、action exact；artifact=
  `runs/outputs/pi05_reward_qv_pivot_rank14_plus2_transport_v1_e3857f7_20260811/analysis.json`，`passed=true`但
  evidence scope明确0 action forward/rollout/update。新design authority已写入
  `docs/action_forecast_writer_qv_rank_reserved_native_reward_design.md`并加入mandatory reading。
- canonical rank-reserved load-only路径已原位实现：q/v rank14 pivot base + 两个physical residual slots、
  action full-rank16 FP32、v9 commit-bound native cache、Program-only reference、no-video source identity和
  old-base/old-reward/rank14-base/q/v-only/full-reward五臂vertical都由同一active owner执行；旧Reward训练入口
  在distributed/runtime初始化前fail closed。vertical的full/q/v-only action已改为使用cache重新加载的q/v
  state并绑定相同video identity；profile OOM候选只作ineligible，tracked Program与output-root resolver已分离，
  且新增纯CPU `rank-reserved-seal` assembler。初始带真实LIBERO assets的全仓CPU回归为
  `386 passed in 28.76s`。
  这只封住实现合同，尚无新图A40行为或strict rollout证据。
- clean implementation commit`82c18cc`已push并建Gate-A detached worktree。实时比较两节点后选择空闲
  `gpu02:3`；launcher和worker双重preflight通过，但首个profile候选warmup在compact rank2 SVD前自然
  fail closed：CUDA autocast把32×32 core matmul降为BF16，而A40不支持BF16 batched SVD。没有profile
  result/cache/rollout/分数，失败root已删除。修复对compact QR/SVD/rank2 lift局部关闭autocast并保持最终
  q/v native BF16；新增模拟CUDA autocast回归，正确assets环境全仓=`387 passed in 28.53s`。
- 修复提交`c5638a9`在live空闲`gpu02:3`完成32-request/1093-frame profile；B8/16/32=
  `.906874/.903246/.904735 LoRA/s`，三点stable、0 OOM、reserved约12.90GB/headroom约34.8GB，选B8，
  相对旧rank16 B8只慢`.479%`。随后vertical在共享五臂生成后、cache/rollout前因diagnostic row缺少
  `suite/task_id/init_state_id`而fail closed；queue为8 pending/0 complete，无cache manifest、vertical/results。
  修复显式合并request identity并新增方法级回归；全仓真实assets=`388 passed in 23.56s`。因Gate-A两份
  artifact必须同commit，成功profile和失败vertical两个partial roots都删除，下一commit完整重跑两步。
- Owner进一步澄清设备策略：不存在固定6卡要求。authority现统一为每次live比较`gpu01/gpu02`，选择一个
  节点并使用该节点所有真正空闲、健康且提高有效吞吐的A40，不等待凑卡、不dummy占位；独立evaluator使用
  所选单节点当时全部有效空卡，训练
  exact-resume才锁原world size/NUMA/rank topology。
- canonical evaluator的可执行owner-six-GPU门已删除：run contract由配置的8-card node topology约束，
  preflight只拒绝空/重复/负index并继续live检查进程；7/8-card选择不再被软件截断。加载`.env.local`后的
  `tests/test_pi05_eval_contract.py tests/test_pi05_eval_launcher.py`为`51 passed`。
- 下一步是将identity修复clean commit/push并重建frozen worktree；随后单张空闲A40完整重跑新graph吞吐profile与五臂
  vertical。两份artifact通过后必须回tracked主分支seal、commit/push，再从sealed commit冻结新worktree，
  然后按有序Gate B/C条件运行两个strict400。
  q/v一阶tangent固定为`B0 dA+dB A0`，action保留实际二阶cross term；no-video source identity与correct-video
  zero-Program rank14 base分开处理。只有cycle1 correct≥144、breadth≥6、lost≤6且gained>lost才算通过；
  140--143为诊断性non-pass。两个行为门前不实现或启动fresh训练。

## Reward-Credit首次profile non-pass与all-mixed/B8修正（2026-08-10）

- clean frozen`c4507e9`在live preflight后使用`gpu02:0--5`完成首次full24×K4×Nmc4 B2 discarded profile；
  root=`runs/outputs/pi05_v6_reward_credit_program_cotangent_profile_full24_k4_nmc4_r6_b2_20260810`，exit/pipeline均0、
  completion passed=false、无checkpoint，六卡随后释放且未触碰他人设备。
- 24 tasks/96 rollouts/24 videos、11 mixed/13 homogeneous、60/36 success/failure、4452 replay chunks和22124
  action steps完整；rank48、condition105.66、negative/correct `.017081`、closure0、LoRA A/B与aggregate action
  response非零，0 OOM/nonfinite/watchdog。唯一失败为固定action probes `1/4`。
- 只读复核定位固定probes`0/7/14/21`中只有0有mixed reward RHS，另外三个homogeneous按合同zero credit；
  旧门与zero-credit保护冲突。旧artifact保持immutable/non-pass，不重写、不续用。
- canonical已改为profile schema v2：all-mixed K4真实首query/原始noise、raw per-task LoRA A/B/action gate；
  physical replay B8基于live`16.34/19.42GB` headroom，静态rank map基于旧profile wall并保持每rank一task/suite。
  全仓`338 passed in 37.69s`，compileall、27 JSON、Black、diff-check和architecture guard均过，0 hard violation/
  parallel family；旧artifact在v2 raw gate下仍fail。本节封存时下一步是新root profile且formal blocked；
  该profile现已完成并通过，当前边界见文件末尾。

## Reward-Credit Program Cotangent实现与CPU封口（2026-08-10，profile前状态）

- 第39.5当时的唯一后继已在canonical `v6_prior` path原位实现；当时active config=
  `configs/pi05_v6_reward_credit_program_cotangent_v1.json`。部署仍为exact language + exactly one action-hidden
  video、Balanced P256、frozen v6-fast、single Program和一套完整rank16 LoRA；fresh state禁止继承RLS10。
- runtime完成K4四persistent-lane batch4 rollout、success+failure executed-prefix、binary LOO、homogeneous zero
  fast path、mixed Nmc4 direct LoRA gradient→Program VJP、full48 single write、interaction cursor与deferred-NCCL
  readiness。删除旧ledger/single-lane/success-only、RLS-specific executable owners、old/current第二forward、
  ratio、第二epoch和shared Adam；历史由Git/docs/artifacts保存。
- 吞吐保持六rank×4 tasks、BF16和当时最大已证实并行；B2的旧容量类比已由上节live证据覆盖。homogeneous
  不拼replay，热路径无SHA/MD5/逐tensor扫描；profile另报shared solve对zero-credit task的
  motion以定位漂移。
- 当时全仓CPU回归`336 passed in 59.12s`。后续首次profile与当前执行边界见上节。

## RLS formal0→10、strict400与正式退役（2026-08-10）

- profile/root-binding seal已clean commit/push为`25bbd52c16cc0f0fd48f478f0fa8b554fcb28dc6`，对应独立
  detached frozen worktree完成fresh0→10。formal root=
  `runs/outputs/pi05_v6_exact_anchored_reconciliation_formal_fresh0to10_r6_lb20_mb10_25bbd52_20260810`；
  launcher、run contract、completion、macro10 checkpoint和exit完整/natural0。10步wall=`199.425195s`、
  input wait=`.278241s`，peak allocated/reserved=`43,247,554,048/46,919,581,696B`；六卡`gpu02:0--5`
  结束后回到0MiB，未触碰他人GPU。
- 预注册strict root=
  `runs/outputs/pi05_v6_exact_anchored_reconciliation_correct400_noreplacement_seed7_method_macro0010_25bbd52_20260810`；
  72/72 shards、400 rows、18 workers return0，wall/shard window=`844.837/592.924s`，overall/rollout-only
  throughput=`.473464/.674623 episodes/s`，400 LoRAs、batch8、18 generators/rollout workers合同不变。
- strict correct=`140/400`、breadth6、per-task=`2/3/47/35/0/34/19/0`。macro0→RLS gained/lost=
  `21/15`，门为true/false/true，overall `passed=false`；因此未启动10→25或六臂。RLS相对blind-v2 macro10
  同分且`17/17`换手，证实feature-space anchoring没有转化成closed-loop retention。
- 通用transition首次因legacy-baseline硬编码在CPU后处理阶段fail closed，未创建半成品。根因定位后只增加
  candidate→唯一allowed baseline family映射；focused 28/28、正确环境全仓304/304、compileall/diff-check
  通过，clean commit/push=`866cca9`。从该clean HEAD生成immutable 33,091B transition artifact；没有改动或
  重跑训练/rollout。
- config现由completion、10-row metrics、macro10 manifest、1,875,159B strict results和33,091B transition
  共同验证`retired_after_macro10_strict_closed_loop_nonpass`。formal runtime拒绝fresh/restart/resume；下一步只
  做第39.5 reward-credit后继的设计与CPU seal。

## RLS f28 live seal与formal-ready封口（2026-08-10）

- 测量合同commit`f28fc8b`已clean pushed并创建独立detached worktree。live检查`gpu01/gpu02`、选定
  `gpu02:0--5`逐卡UUID/serial/NUMA/process及`/data1` quota后，保持六rank、B20/B10+10、BF16、workers2、
  Ring/Simple/deferred NCCL启动全新fresh0→3；六卡计算期约44.8GB（43.7GiB）且100% utilization。
- run natural exit0，0 checkpoint/OOM/nonfinite/negative forward；mechanism profile=`100452B`、completion
  passed且17/17 hard checks全过。production三步=`19.9974/20.7508/19.5182s`、mean ratio=`.952297`，peak
  allocated/reserved=`43,261,790,208/46,919,581,696B`。old drift/blind=`.248611/.213872`、old rows
  improved=`1/1`、current/blind=`.999980/.784334/.640650`；六卡随后回到0MiB。
- 独立subagent与主进程均从raw macros重算通过，核对run/launch commit、completion、exit和无checkpoint。
  config已登记f28 immutable artifact并切为`active_deployment_sealed_formal_ready`；旧f0c artifact未修改，
  继续为16/18 non-pass，discarded profile权重均不进入formal。
- formal前只读审计发现并原位修复两个窄缝隙：通用historical transition登记RLS 10/25并拒绝50；evaluator
  在创建formal correct macro10 output前重读training run contract与checkpoint manifest，要求路径精确等于
  预注册root且formal/schema/commit/next_macro/metrics rows一致，完整cursor由既有adapter验证。没有新增CLI
  或并行evaluator path；定向回归`70 passed`、
  全仓fresh=`304 passed in 92.19s`，compileall、26 JSON、real config/artifact load和diff-check通过。
- 下一步是clean commit/push并创建新formal frozen worktree；live preflight
  后只运行identity fresh0→10，再立即在预注册root做strict correct400。不复用profile state，不先跑80-row
  screen，不在结果前授权10→25。

## RLS fresh0→3首次live证据与profile合同修正（2026-08-10）

- implementation已由clean pushed`f0c3f51`封存，并从独立detached worktree在`gpu02:0--5`完成首次
  fresh0→3。启动前逐卡UUID/serial/NUMA/process与`/data1` quota通过；运行保持六rank、B20/B10+10、
  workers2、deferred NCCL和`NCCL_P2P_DISABLE=1`，自然exit0，未保存checkpoint，卡已释放。
- 原18项检查16项通过。RLS机制证据为old drift/blind=`.248611/.213872`、old rows improved=`1/1`、
  current/blind=`.999980/.784334/.640650`；其余feature/null/LoRA/action/closure/state/OOM门均成立。失败项
  仅是首步`1e-5`低位ratio门与逐macro fresh-vs-warm wall门；旧root继续保留`passed=false`。
- 第39.4.1已预注册修正：删除ppm级GPU等价hard check但保留raw diagnostic，吞吐使用三宏步production
  arithmetic mean对原baseline`<=1.10`。当前实测mean ratio=`1.029799`，同schedule v2总wall只差`.277%`。
  Writer/RLS/dtype/batch/forward完全不改；新合同focused=`82 passed`、全仓=`300 passed`，compileall、
  26 JSON和diff-check通过，旧artifact matcher仍返回false。下一步是clean push、新frozen worktree和一次
  新的fresh0→3，不能复用此次artifact或profile权重。

## Exact Anchored Reconciliation实现与正式状态机封口（2026-08-10，首次live profile前的CPU seal）

- canonical v2 executable config已由Git历史保存，active config改为
  `configs/pi05_v6_exact_anchored_reconciliation_program_residual_v3.json`；部署计算图不变，fresh checkpoint
  schema升级为Program+FP64 precision联合状态，禁止从v2 macro10/25伪resume。
- `condition_update`已实现anchored RLS、old/current/blind diagnostics和joint apply；首步非正交blind ridge、
  streaming/direct累计解、zero-cotangent assimilation、reference/current/blind motion均有CPU oracle。
  checkpoint codec分离Program与precision，并新增wrong key/dtype/shape/nonfinite/non-PD回归；save只在主rank、
  预声明checkpoint边界做256×256 Cholesky，不进入macro热路径。
- 只读审计发现并已修复三个fail-open缺口：formal evaluator曾接受任意macro；runtime曾无条件允许10→25；
  paired analysis曾不认识RLS v3。fresh0→10现在必须预注册macro0/macro10 strict roots到原run contract，resume
  会重聚合400-row结果并自动执行`correct>=140/lost<=6/breadth>=6`支持门，摘要只追加invocation而不污染
  checkpoint/deployment state。
- architecture guard经拆分checkpoint payload、contract spec与paired decision gate后为0 hard violation；
  focused=`75 passed`，加载`.env.local`/LIBERO assets后的全仓=`300 passed in 61.72s`，compileall、JSON与
  diff-check通过。该CPU seal当时尚需authority终检、commit/push、frozen worktree与fresh0→3；这些步骤现已
  由本文件顶部f28 live seal完成，当时仍为0 RLS GPU/strict分数。

## Balanced DC--Causal v2 formal与strict终局（2026-08-10）

- clean frozen commit=`abd8e0826e52758eda53b1963f8b12db92bf3748`；formal root=
  `runs/outputs/pi05_v6_balanced_causal_condition_residual_formal_r6_lb20_mb10_abd8e08_20260810`完成0→25、
  25 metrics与macro10/25 checkpoints。累计step wall/input wait=`535.464796/2.208183s`，mean step=
  `21.418592s`，peak allocated/reserved=`43,247,029,760/46,917,484,544B`，0 OOM/nonfinite/negative forward。
- macro10 strict root=
  `runs/outputs/pi05_v6_balanced_causal_condition_residual_correct400_noreplacement_seed7_method_macro0010_abd8e08_20260810`：
  72/72 jobs、400 rows、18/18 workers return0，`140/400`、breadth6、per-task=`1/2/48/31/0/38/20/0`、
  per-suite=`3/79/38/20`。相对macro0 gained/lost=`19/13`、union/intersection=`153/121`。
- macro25 strict root=
  `runs/outputs/pi05_v6_balanced_causal_condition_residual_correct400_noreplacement_seed7_method_macro0025_abd8e08_20260810`：
  `139/400`、breadth6、per-task=`2/4/48/30/0/38/17/0`、per-suite=`6/78/38/17`；相对macro10
  gained/lost=`12/13`。内部72/72 jobs、400 rows、18/18 workers return0完整；外层wrapper exit没有记录，
  后验audit明确保留unobserved/missing而没有伪造exit0。
- v2曲线最终为`134/140/139`，没有超过历史`143`；按预注册门退役，不续50、不补多臂、不扫超参。所有
  GPU已在运行结束后自然释放，本实现阶段没有重新启动GPU。

## v2 LoRA与same-task视频诊断完成（2026-08-10）

- macro0→10 effective delta/base median=`1.69498e-4`、stable rank=`1.000022`、top1 energy=`.999978`；
  Program residual L2/RMS/max=`.0174616/3.818e-6/4.997e-5`。这些是机制参考，不单独选择checkpoint。
- 50视频/task raw correction consistency=`.141539--.142175`，接近`1/sqrt(50)=.141421`；fixed macro10
  all-target pair cosine=`-.001371--.003280`、action target=`-.009579--.014302`。结论是视频condition确实
  产生非零变化，但同task demo correction几乎不共享方向；不能写成顺序因果已学会，也不能直接用few-shot
  平均掩盖根因。
- 与macro10→25 success换手一起，证据把下一单变量定位为跨macro retention/reconciliation。第39节RLS因此
  保留全部部署和吞吐路径，只替换training-time update kernel；few-shot、reward-credit与Procedure重构均
  继续作为RLS被实证否决后的有序后备，而不是180度切换。

## Balanced DC--Causal v2 zero-memory macro0 strict400完成（2026-08-10）

- clean pushed/frozen commit=`6b5f7a6ad6ef1a778205071f38faec9f936cf54e`；启动前live比较两节点并选择
  0MiB/0%/P8、无compute process的`gpu02:0--5`，没有触碰忙碌的`gpu01:3`和`gpu02:6--7`。`/data1`
  used/quota=`573,456,828/1,073,741,824KiB`，正式root实际`1,085,108,227B`。
- root=`runs/outputs/pi05_v6_balanced_causal_condition_residual_correct400_noreplacement_seed7_method_macro0000_6b5f7a6_20260810`；
  72/72 shards、400 rows、18 workers均attempt1/exit0，strict correct=`134/400`、correct80=`26/80`、
  breadth6，per-task=`0/5/48/34/0/35/11/1`、per-suite=`5/82/35/12`。wall/rollout window=
  `867.152/616.138s`，CPU aggregate重建完全一致。
- 400套LoRA由18 generators以54 batches全部fresh生成，configured/max batch8、max frames69、0 reuse/
  redundant forward；Writer全释放、source policy全复用且未reload。max per-generator allocated/reserved=
  `11,745,421,312/12,895,387,648B`，0 retry/OOM/nonfinite/forbidden reads；结束后六卡0MiB/P8。
- 与历史native v6 macro0的400 rows严格配对：state/language/env seed/policy noise、teacher demo/order/selection
  seed和video mapping均0差异，success逐行同一，gained/lost=`0/0`、共同成功/失败=`134/266`；每task demos
  0--49各一次。新旧400 cache entries逐tensor直比30,400 tensors、514,867,200 values全部bit-exact；仅一条
  共同成功episode终止step `106→107`，其余399 rows steps一致。该结果只封存exact zero-memory baseline，
  不是v2改进。下一步从新clean pushed/frozen
  authority formal fresh0→10并立即strict correct400。

## Frozen-worktree formal prepare阻塞修复并通过（2026-08-10）

- deployment写回由clean pushed`d228d0d`封存。对应detached frozen worktree首次CPU-only formal prepare在
  任何CUDA初始化、cache或scientific row产生前fail closed；精确错误为residual Writer sealed contract
  violation。只读分层定位证明raw seal/profile/results/manifest均匹配，根因是`runs`软链接resolved到canonical
  仓库后被旧verifier误判为越出frozen worktree。
- clean pushed`af7b101`以单一path owner窄修复：仅允许词法`runs/outputs/...`映射到resolved canonical
  outputs root，绝对/`..`/伪前缀/nested symlink和vertical manifest越界继续拒绝。新增symlink round-trip与
  escape回归；全仓`285 passed in 21.38s`，compileall/Black/JSON/diff-check通过。
- clean frozen`af7b101`重跑同一CPU-only prepare自然exit0：formal validation 8 tasks×50 states、correct/
  without-replacement、seed7、18 rollout workers + 18 Writer generators、batch8；v8 adapter确认historical v6 macro400 load-only、
  method macro0、`[256,320,256]` FP32 fresh elementwise-zero residual且residual checkpoint bytes=0；
  `estimated_peak_new_bytes=1,064,370,176`。临时prepare root已清理；未启动GPU，也没有新增strict成绩。
  下一步仍是新clean pushed/frozen authority上的zero-memory macro0 strict correct400。

## Balanced DC--Causal v2 deployment双root seal（2026-08-10）

- GPU前contract/fail-close修复由clean pushed commit=`2af82aa6769570786c64d3c026374150d259360c`
  封存，detached frozen worktree=
  `/data1/user/ymdai/worktrees/EMBER-balanced-key-deployment-2af82aa-20260810`。启动前实时比较`gpu01/gpu02`；
  选择当时0MiB、0%且无compute process的`gpu02:0` A40。`/data1` quota live used/quota=
  `573,433,076/1,073,741,824KiB`，新增约25MiB只占余量`.00512%`。
- throughput root=
  `runs/outputs/pi05_v6_balanced_causal_condition_residual_writer_throughput_profile_val8x4_correct_gpu02g0_2af82aa_20260810`。
  validation8×4固定32 requests、1093 sampled frames、最长67 frames；batch8/16/32完全共享entry/frame panel，
  1 warmup+2 measured，LoRAs/s=`.911238/.901898/.906482`，repeat wall=
  `[34.9668,35.2673]/[35.6341,35.3274]/[35.2987,35.3039]s`。三者stable，peak reserved约
  `12.91/12.93/12.93GB`、headroom约`34.77GB`，按最高实测吞吐选择batch8。
- vertical root=
  `runs/outputs/pi05_v6_balanced_causal_condition_residual_writer_vertical_smoke_val8x1_correct_b8_gpu02g0_2af82aa_20260810`。
  validation8×state0 correct真实执行8 videos→8完整rank16 LoRAs→native cache→释放Writer→复用source
  policy→8条LIBERO闭环；8/8 rows、`4/8` success、总wall=`336.0559s`、rollout window=`199.7986s`，
  单次launcher return0、0 retry/runtime failure。`4/8`只作执行smoke，不是formal性能结论。
- deployment assembler共同重读profile `10225B`、results `92811B`、cache manifest `52153B`并通过：同一
  clean commit/v8 adapter、batch选择、8新entries、76 tensors/entry、`2,641,920B`、72 BF16+4 F32、
  Writer release/source reuse/no reload和0 teacher-action/state/reward/terminal、0 expert-bank读。未做SHA/MD5
  或逐tensor防御性扫描；结束后`gpu02:0`回到0MiB/P8。
- config已切为`active_deployment_sealed_formal_ready`，formal=
  `ready_after_live_mechanism_and_deployment_seals`。下一步只从新的clean pushed/frozen seal先跑zero-memory
  macro0 strict correct400，再决定fresh0→10；当前仍无v2训练或formal strict成绩。写回后全仓
  `284 passed in 26.86s`，compileall/Black/JSON/diff-check、raw seal重建、formal-ready与pre-deployment
  fail-close负回归全部通过。

## Balanced DC--Causal v2 mechanism profile全门通过（2026-08-10）

- CPU seal clean pushed commit=`5d93434`；旧v1 frozen worktree在确认无进程、无未跟踪改动后移除，新建
  detached frozen worktree=`/data1/user/ymdai/worktrees/EMBER-balanced-key-profile-5d93434-20260810`。
  launch前实时比较`gpu01/gpu02`并重查`/data1` quota=`573,432,200/1,073,741,824KiB`，选择与sealed
  baseline相同且当时空闲的`gpu01:0,1,2|4,5,7` 3+3 NUMA panel；GPU3他人VLLM、GPU6和gpu02均未触碰。
- retained root=
  `runs/outputs/pi05_v6_balanced_causal_condition_residual_mechanism_profile_macro49_r6_lb20_mb10_5d93434_20260810`；
  24 tasks×B20、8/8/8 shuffled/reversed/wrong、0 negative policy forward、无checkpoint，tmux launcher自然
  exit0。root只有contract/invocation/profile/completion四个预期文件、总计60,502 bytes；六张卡结束后回到
  14MiB，0 OOM/nonfinite。
- 原13项机制/吞吐门**13/13全部通过**：full48 rank48、regularized condition=`106.114`、aggregate correct
  motion/cotangent=`.968254`、negative/correct=`.0218514`；24/24 correct retention且最小`.942261`，24/24
  paired negative null且最大leakage`.048462`。application predicted/observed relative RMS=`0`。
- 分臂cosine mean/max与leakage mean/max：shuffled=`.479565/.851083`、`.024184/.048462`；reversed=
  `.013732/.023307`、`.018664/.032562`；wrong=`.507178/.762135`、`.025999/.033571`。这与v1 shuffled/
  reversed/wrong cosine mean `.985525/.956451/.906269`形成直接、同schedule修复；没有aggregate掩盖某类负臂。
- A/B response RMS=`1.37744e-5/1.38187e-5`，4/4 suite fixed-action response RMS=`.001210888`；Program
  memory→完整LoRA→policy action路径未因key修复减弱。value-delta RMS从v1`2.12559e-6`降到
  `1.16318e-6`，但correct motion retention反而从`.807966`升到`.968254`，符合病态逆放大被移除而非能量塌缩。
- production task/kernel=`19.585536/.436306s`，合计`20.021842s`，相对同host/panel sealed baseline
  `21.095110s` ratio=`.949122`；input wait=`.069295s`与baseline`.076318s`接近，故吞吐pass不是跨host
  wait假象。verification另计`.735397s`，step总wall=`20.760785s`；峰值allocated/reserved=
  `43,261,166,592/46,917,484,544` bytes。
- artifact evidence已写入v2 config并由contract从raw profile/run/completion完整重算通过；profile状态切为
  sealed；复核发现当时formal状态仅凭mechanism已提前ready，与文档顺序冲突，现已改为硬阻塞到deployment
  双root seal。当前仍没有v2 rollout、训练或strict成绩，且独立block L2可能放大很小但
  非零dynamic的same-task视频噪声仍待closed-loop实证。下一动作只做新clean pushed seal上的单卡
  residual deployment batch8/16/32 profile与correct smoke，不直接启动formal训练。
- deployment evidence也从旧的profile-only verifier修正为唯一双root owner：同commit的profile
  `writer_generation_profile.json`、vertical `results.json`和cache manifest缺一不可，并重算固定panel选型、
  validation8×state0、单次launcher、8 rows/entries、native76-tensor LoRA、release/reuse与零禁止读取。
  该修复只在一次性seal路径执行，不进入Writer生成、policy推理或训练热路径。
- 双root/fail-close修复后的无GPU最终回归为全仓`283 passed in 26.10s`、compileall、Black、26份JSON、
  真实config artifact重载与diff-check通过。architecture guard相对`5d93434`为`+968/-318`、净增650行、
  0 hard violation；原1243行contract缩到1101行，新增624行deployment-seal单owner和其238行聚焦测试，
  无parallel family。review项是明确的一次性artifact owner、既有长contract/tests和目录密度，不进入热路径。

## v1机制profile裁决与Balanced DC--Causal v2 CPU seal（2026-08-10）

- clean pushed/frozen`6903ee6`只在live空闲`gpu02:0--5`完成一次第37节v1 macro49 mechanism profile；
  root=`runs/outputs/pi05_v6_condition_residual_mechanism_profile_macro49_r6_lb20_mb10_6903ee6_20260810`，
  launcher自然exit0且按设计不保存checkpoint。0 OOM/nonfinite/negative policy forward，六卡退出后均回到
  0MiB，GPU6/7上的他人进程未触碰。
- 13项门中10项通过：feature rank48、correct motion/cotangent=`.807966`、24/24 correct retention、
  application closure relative RMS=`0`、A/B response RMS=`1.27385e-5/1.26956e-5`、四suite fixed-action
  response=`4/4`。这证明显式kernel、frozen-v6 decoder和Program→完整LoRA→action路径工作，未发现
  full48 gather/order/sign/solve的工程错误。
- 正式non-pass来自旧key几何：regularized Gram condition=`1315.33`、negative/correct=`.264351>.25`、
  task-local null=`15/24<18/24`；shuffled/reversed/wrong feature cosine mean=`.98552/.95645/.90627`，
  各自null过门=`2/8,6/8,7/8`。paired ridge解析leakage与实测相关`.99021`，最难shuffled距离`.07777`
  要求至少约`12.86x`差分放大，因此不训练v1、不扫lambda/seed/P/threshold。
- production=`23.530704s`、相对sealed baseline ratio=`1.115458>1.10`按预注册保留non-pass；但超门仅
  `.326083s`，小于跨host input-wait差`.633711s`，不能扩大成稳定结构慢化，也不为它单独重跑或降低
  batch/dtype/并行度。
- 历史phase16证据显示DC能量占`.98057`，而centered sqrt-causal-prefix对correct/reversed/shuffled
  template cosine=`.96263/-.94287/-.04463`。据此只把canonical key原位升级为第38节v2：video-DC
  static与centered causal dynamic分别fixed-JL到128、各自zero-L2后拼成P256；historical v6的600 tensors、
  `[256,320,256]` memory、full48、`.01` damping、step1、B20/B10+10和0 negative forward均不变。
- v2同时移除per-condition GPU sort/mask scalar同步，并把profile-only bookkeeping和约15MiB zero allocation
  移出production timer；没有牺牲科学batch、dtype或底层精度。v1 executable config/code从active tree退役，
  Git和无checkpoint profile artifact保留；v2 schema/checkpoint fresh-incompatible。
- CPU回归新增同static、反dynamic两帧反例，natural/reversed unit keys内积为0；聚焦`52 passed`，加载
  `.env.local`和LIBERO assets后的最终全仓为`281 passed in 21.34s`。compileall、26份JSON和diff-check通过；
  architecture guard相对`6903ee6`为`+144/-126`、净增18行且0 hard violation，1243行legacy contract未增长。当前仍
  没有v2 GPU、训练、rollout或strict成绩；下一步是clean commit/push并从新frozen worktree做一次
  live-preflight后的v2 mechanism profile。
- 提交前独立只读审查未发现数学、shape、zero-preserving或hot-path阻断；补充锁定了训练时`frame_order`
  重排与部署时物理重排evidence的key等价，并把正式analysis family从已退役的`v1`标签改成`v2`。AGENTS、
  README、design顶部和三份长期概念文档的旧“当前”表述同步纠正，历史正文仍保留为证据而不能恢复执行。

## 第37节canonical实现与CPU seal完成（2026-08-10，尚未启动GPU）

- 已在唯一active path完成Frozen-v6 Counterfactual-Null Condition-Kernel Program Residual：frozen v6
  macro400 base + fixed temporal JL feature + single FP32 Program memory + full48 explicit update。旧
  teacher-audit/effective-objective/flow-teacher执行路径及一次性tests删除，没有保留并行旧trainer。
- run/config/checkpoint/evaluator全部换成fresh-incompatible residual schema。checkpoint只含约80MiB memory、
  cursor和6-rank RNG；resume保持原commit且要求authority ancestry。deployment worker只安装一次finite
  residual memory，base600/projection/template继续冻结；v8 adapter和strict row evidence已接通。
- mechanism profile增加真实task-local 24+24 motion证据、4-task/4-suite fixed-action传递和production wall
  gate。8次fixed-action inference属于verification、与生产计时分离且不读target action；production ratio
  对历史sealed `21.095109596s`必须`≤1.10`，不为底层低位精度降低B10+10或六卡并行。
- config仍是`awaiting_live_a40_macro49_profile`，formal与新deployment evaluation都fail closed。下一步必须
  clean commit/push、frozen worktree、live双节点/quota preflight后才运行一次macro49 profile；profile weight
  不保存。通过后另做单卡batch8/16/32新图profile/correct smoke，再评测macro0和formal0→10。
- 启动前最后合同复核又封住三条证据漏洞：mechanism seal从raw macro重算全部门并匹配完整科学run；
  `formal_result_sealed`必须有completion、50-row metrics和macro10/25/50 checkpoint manifests；trained
  deployment checkpoint的training commit必须属于当前authority lineage。通用evaluator同时允许clean
  detached frozen authority ancestor，避免用额外branch/worktree换取`@{upstream}`。
- CPU final：全仓`280 passed in 21.02s`、compileall、JSON、`git diff --check`通过；architecture guard
  `hard_violations=[]`。当前没有EMBER GPU进程，也没有新strict rollout成绩。

## Expert-Flow正式audit完成、CEFD否决与第37节选择（2026-08-10）

- 修复后clean pushed/frozen`e8e4728`在live双节点GPU/quota preflight后，只使用空闲
  `gpu01:0,1,2|4,5,7`；GPU3他人VLLM和gpu02忙卡均未触碰。formal root=
  `runs/outputs/pi05_v6_expert_flow_teacher_audit_r6_lb20_mb10_e8e4728_20260810`自然exit0，结束后六卡回到
  14MiB。
- 结果完整覆盖24 tasks、suite 6×4、480/480 queries、8/8/8 negatives和144 policy forwards；0 update/
  rollout/checkpoint/OOM/nonfinite。wall/input wait=`39.698123/.684060s`，peak allocated/reserved=
  `43,418,974,720/47,133,491,200` bytes；retained output只有run contract、invocations、teacher audit和
  completion四个预期文件。
- expert/macro0/tangent10 flow loss=`.098631330/.091801740/.091843160`；teacher gate仅`2/24` tasks、
  `0/4` suites通过而要求`18/24+3/4`。gradient nonredundancy以compiler/factor residual=
  `.686410/.838727`通过，但来自整体更差teacher；正式decision=`authorize_cefd=false`，不做CEFD weight
  profile、训练或其它expert step搜索。
- config已记录formal non-pass并fail-closed。按第36节，下一代码阶段删除一次性audit/flow-teacher路径，
  保留从runtime抽出的canonical run-contract owner、Git和formal evidence。旧run-contract runtime字段的
  `2 forwards/task`实际指单臂两个B10 microbatches；科学合同/result的6和总数144才是真实三臂forward，
  结果不受影响且不重跑GPU追逐退役schema。
- 三条独立只读审计共同选择第37节Frozen-v6 Counterfactual-Null Condition-Kernel Program Residual：冻结
  v6全部600 tensors，在320×256 fused Program后加P256 zero memory；correct只写真实functional cotangent，
  24个轮换counterfactual作zero-motion rows。它复用历史Condition-Kernel已证明的显式update隔离并替换其
  唯一cold-decoder失败，保留v6高增益起点；本句记录当时的pre-implementation状态，现已由本文件顶部
  “canonical实现与CPU seal完成”覆盖，仍未启动新GPU工作。

## Expert-Flow Teacher Audit实现、首启wiring失败与修复（历史，2026-08-10）

- clean pushed/frozen`7be51b1`的首次正式启动通过live双节点GPU/quota、24个step2000 experts、comparison
  macro10和CUDA前二次UUID/serial/NUMA门；只使用空闲`gpu01:0,1,2|4,5,7`。六rank完成大部分资产装载后，
  在写run contract前一致触发`UnboundLocalError`：公开`checkpoint_contract()` builder被同名局部变量遮蔽。
  这是确定性Python wiring错误，不是OOM/NCCL、audit或科学non-pass；output root未创建，0 forward/update/
  rollout/checkpoint，六卡自然回到14MiB。失败launch contract/log/exit保留，修复只把局部结果重命名为
  `checkpoint_contract_value`；任何retry必须来自新clean pushed commit和全新root。
- 唯一canonical `scripts/train_v6_prior_writer.py`已接入fresh-only `--mode teacher-audit`：world6、4 tasks/rank、
  schedule macro49、logical B20/physical B10+10、workers2、0 optimizer/scheduler/update/checkpoint/rollout。
  Tangent formal runtime继续fail-closed，comparison只读sealed`b308941` macro10的41个compiler/factor tensors。
- matched flow primitive严格捕获action projection的真实7维velocity。每个B10 slice复用同一keyed noise/time/
  offset运行expert、tangent10和macro0三臂；只有student可微，四个小型real-action loss统一FP32。full24四类
  gradient由rank内4-task mean后一次stacked all-reduce生成，不增加热路径逐tensor同步或长期cache。
- run-contract构造已从runtime抽成唯一公开owner；删除旧runtime wrappers/re-export。audit通过显式frozen
  bindings复用canonical task objective、gather、layout/norm和runtime metrics，消除training↔audit import环；
  architecture guard无hard violation或第二CLI/runner/deployment family。
- CPU oracle覆盖B20/B10+10 matched randomness、正式6 forwards/task、real7 width/FP32 loss、same-memory、
  8/8/8 negatives、480 unique queries、Gram pinv `rtol=1e-5`、近共线effective rank与0 update。加载
  `.env.local`的修复后全仓seal为`284 passed in 32.66s`，compileall/JSON/diff-check通过；尚未产生有效audit root，
  也没有GPU或CEFD结果。
- 该段记录的是正式结果前的实现/首启状态；修复后audit已完成且teacher门失败，最新裁决只取上一节。

## Condition-Local Tangent Tube formal与strict裁决完成（2026-08-10）

- current seal的clean pushed commit=`b308941`；独立formal branch/worktree绑定该commit。launch前live比较
  `gpu01/gpu02`和`/data1` quota，只选择当时空闲`gpu01:0,1,2|4,5,7`，未触碰GPU3及gpu02他人进程。
- formal root=`runs/outputs/pi05_v6_tangent_tube_formal_r6_lb20_mb10_b308941_20260810`完成fresh0→10，
  10 metrics、macro10 checkpoint、completion齐全。总step wall/input wait=`207.4436/.2655s`，peak
  allocated/reserved=`43,316,440,064/47,112,519,680` bytes，0 OOM/nonfinite；结束后六卡自然释放。
- macro10机制：correct/negative tube中位=`.013900/.014079`、两臂`24/24`过`.03`；direction ratio=
  `108.926/126.883`、两臂`0/24`过`≤1`；completion error中位`.252295`、`0/24`过`.05`。半径约束
  生效，但expert方向写入失败。
- strict root=`runs/outputs/pi05_v6_tangent_tube_correct400_noreplacement_seed7_method_macro0010_b308941_20260810`
  自然exit0，72/72 shards、400 rows，得分`131`、correct80=`27`、breadth5、per-task=
  `0/3/46/31/0/40/11/0`。wall=`858.578s`、吞吐`.465887 rollout/s`，六卡再次自然释放。
- exact historical transition root=
  `runs/outputs/pi05_v6_tangent_tube_macro0010_historical_baseline_transition_b308941_20260810`：相对macro0
  `134`为gained/lost=`16/19`、churn35、net`-3`、`p=.735879`。按预注册门不续25、不补六臂、不扫
  weight/LR/WD；Tangent runtime现formal non-pass后fail-closed。
- 并行历史/结构审计共同收敛到第36节matched no-update Expert-Flow Teacher Viability Audit。下一步先在
  同一train24 B20/noise/time上验证step2000 expert是否真是更好且非冗余的policy-flow teacher；两门同时
  通过才实现CEFD，否则转structured update parameterization。formal non-pass config/runtime的focused=
  `25 passed`、全仓=`277 passed in 25.36s`，compileall与diff-check通过；当前无EMBER GPU进程。

## Condition-Local Tangent Tube exact-resume seal（2026-08-10）

- clean pushed/frozen`c1bdcae`、新v3 config在live比较`gpu01/gpu02`与`/data1` quota后，只使用空闲
  `gpu01:0,1,2|4,5,7`。fresh0→1、同root exact-resume1→3和independent contiguous0→3均按
  B10+10、logical B20、workers2、3+3 NUMA、Ring/Simple、`NCCL_P2P_DISABLE=1`和deferred-NCCL运行。
- fresh后原自动chain的inter-phase selected-GPU preflight发现设备不再满足expected-idle合同，安全
  fail-close且没有创建第二次scientific invocation。重新live检查通过后，resume与contiguous分别由新
  tmux完成；三份phase exit均为0，原chain exit1只保留为安全门证据。六卡最终自然回到14MiB。
- 两个retained roots各3 metrics、macro1/3 checkpoints和completion；step wall=
  `62.34061/61.95860s`、input wait=`.09366/.13220s`、peak allocated/reserved=
  `43,316,387,840/47,137,685,504` bytes，0 OOM/nonfinite。没有改batch、workers、allocator、kernel或
  objective，也没有建立anchor cache。
- roots=`runs/outputs/pi05_v6_tangent_tube_profile_resume_r6_lb20_mb10_c1bdcae_20260809`与
  `runs/outputs/pi05_v6_tangent_tube_profile_contiguous_r6_lb20_mb10_c1bdcae_20260809`；canonical v4
  evidence已嵌入tracked config，`.codex/tmp`输出只作本次assembler的临时副本。
- 恢复原gradient worktree路径后，canonical assembler从三个distinct roots完整通过：run contracts、
  cursor、6-rank RNG、scheduler/AMP和checkpoint semantics等价；macro3 Writer relative L2=
  `1.14428e-6`，scientific metrics最大tolerance ratio=`.67790`。evidence已原样嵌入v3 config，profile与
  formal同步sealed，formal runtime=`(50,(10,25,50))`，profile runtime按预期关闭。
- profile早期机制尚未过门：macro3两臂tube median约`.0316/.0317`，directional ratio约`61×`，并发生
  gradient clip。该证据已写入design/findings，不冒充性能或机制通过，也不据三步提前转向。旧状态单测
  已更新并新增formal strict-descendant guard；focused=`25 passed`、全仓=`277 passed in 27.80s`，
  compileall与diff-check通过。下一操作是clean commit/push与formal frozen worktree；随后fresh0→10并
  立即跑strict correct400。

## 2026-08-09 Tangent Tube六卡gradient profile完成并封存权重

- canonical CPU seal commit`2616773`已push，独立frozen worktree/branch绑定同一upstream commit。live
  双节点preflight选择空闲`gpu01:0,1,2|4,5,7`，GPU3与gpu02他人进程未触碰；`/data1` quota=
  `571,993,132/1,073,741,824KiB`，root最终仅160KiB。
- tmux launcher自然exit0；root=
  `runs/outputs/pi05_v6_tangent_tube_gradient_profile_macro49_r6_lb20_mb10_2616773_20260809`。24 tasks、
  480/480 unique queries、8/8/8 negatives、最长105帧，wall/input wait=`21.53076/.60603s`，peak
  allocated/reserved=`43,353,948,672/47,112,519,680` bytes，0 OOM/nonfinite，六卡回到14MiB。
- 24/24 correct/negative student-anchor delta与tube均exact zero；assembler给出projection/ranking=
  `.00686480847114155/.010514453175708578`，compiler应用比例均`.25`、factor=
  `.108659/.026876`。证据已原样写回v3 config，gradient/aux sealed、profile ready、formal仍blocked。
- 下一步先CPU回归并clean commit/push严格后继，再从新frozen worktree运行fresh0→1、same-root
  exact-resume1→3及independent contiguous0→3；gradient checkpoint不进入后续训练。evidence写回后的
  config/runtime状态回归为全仓`276 passed in 25.49s`。

## 2026-08-09 Tangent Tube canonical实现与CPU门完成（尚未启动GPU）

- 旧ECP v2 executable config已由唯一
  `configs/pi05_v6_condition_local_tangent_tube_writer_v3.json`原位替换；历史ECP代码身份只在generic
  read-only analysis family中保留。新run/checkpoint/trainer/RNG/adapter/episode使用独立v3/v4/v7 schema，
  ECP checkpoint不得resume成新方法。
- canonical effective objective已加入correct/negative condition-local frozen-v6 tangent tube；Writer
  decoder支持显式成对compiler/factor-head override，runtime在deferred NCCL同步后、resume前构造
  training-only anchor。metrics以每task一次packed D2H记录35个机制量，不增加逐tensor host sync。
- checkpoint与deployment回归证明只恢复41个trainable tensors；即使checkpoint内frozen upstream或
  template被篡改，也保留historical warm-start值。dynamic anchor的optimizer/checkpoint/deployment
  ownership均为false，formal evaluator继承未变部署图的A40 batch8 throughput seal并继续fail-close。
- CPU exact-D/gauge/gradient/same-memory oracle、contract/checkpoint/adapter和三family strict分析全部通过；
  旧smoke assembler/runtime状态退役后，`source .env.local && .venv/bin/python -m pytest -q`为
  `276 passed in 28.74s`，compileall与diff-check通过。
  未启动训练、rollout、GPU profile或长期实验。下一步是clean commit/push、frozen worktree、实时双节点/
  quota预检后的一次六卡gradient/throughput/exact-resume profile。

## 2026-08-09 ECP formal0→25与strict负裁决完成

- clean pushed/frozen`450e688`的formal fresh0→10与exact-resume10→25均由tmux自然exit0；root=
  `runs/outputs/pi05_v6_ecp_formal_r6_lb20_mb10_450e688_20260809`。25 metrics、macro10/25
  checkpoints、completion和完整resume state齐全，0 OOM/nonfinite/clip。
- macro10/macro25的strict correct400均以`gpu01:0,1,2|4,5,7`、18 workers、batch8完成，分别
  `133/400`与`120/400`；两root都是exit0、72/72 shards、400 rows、400 LoRAs、54 batches、
  0 retry/reuse/redundant forward。macro25 wall=`859.138s`，overall/rollout-only吞吐=
  `.4656/.6516 rollout/s`，随后六卡回到14MiB且无compute进程。
- cross-family native validation和400-row exact pairing通过。macro10对macro0=`134`为`22/23`、
  net`-1`；macro25为`13/27`、net`-14`、McNemar `p=.038477`。内部`a_correct`与expert
  component持续上升，但闭环下降且正交norm漂移主导，按门退役ECP；不继50/100、
  不扫权重、不补六臂。
- 当时canonical historical transition入口只接受ECP macro10/25/50；现已由上节generic v2入口覆盖，
  继续保留旧ECP只读结果并加入tangent family，不允许任意mixed family。
- 第35节已封印v6 Condition-Local Dynamic Expert Tangent Tube：对correct和当前negative分别复用
  same-input frozen-v6 baseline，只惩罚增量的expert-orthogonal分量，两臂取mean、ranking不变、
  无新权重。它与历史global basis freeze、低LR/WD、Recenter/Prior-Innovation和尚未实施的
  behavior distillation均已去重；其CPU实现与合同已由上节完成。

## 2026-08-09 ECP resume profile完成并解锁formal

- gradient seal strict后继`fea3f40`已clean/pushed并有独立frozen worktree；live重查后只使用空闲
  `gpu01:0,1,2|4,5,7`。resumed root fresh0→1后exact-resume1→3，独立contiguous root fresh0→3；三个
  tmux launcher自然exit0，六卡结束后均释放。
- roots=`runs/outputs/pi05_v6_ecp_profile_resume_r6_lb20_mb10_fea3f40_20260809`与
  `runs/outputs/pi05_v6_ecp_profile_contiguous_r6_lb20_mb10_fea3f40_20260809`。两者各3 metrics、macro1/3
  checkpoints和completion；0 OOM/nonfinite，step wall resumed/contiguous=`62.369/61.017s`，peak
  allocated/reserved=`43,275,957,248/47,118,811,136` bytes。
- artifact assembler完整通过：contracts、cursor、rank RNG、scheduler/AMP和checkpoint语义一致，Writer
  relative L2=`4.845e-6`，metric tolerance ratio=`.4290`。evidence已原样写入v2 config，profile/formal
  statuses均为`sealed_from_live_a40_resume_profile_evidence`；profile checkpoint永久不进入formal。
- 三步`a_correct`在23/24 tasks向1移动、expert component在23/24上升且norm不系统塌缩，只授权fresh
  formal0→10。canonical evaluator已增加具名`historical-baseline-transition`，保留native family隔离并
  省去无增量的v2 macro0重跑；定向`28 passed`、全仓`262 passed in 24.92s`、compileall和diff check通过。
  下一步是clean commit/push、formal frozen worktree、live preflight、fresh0→10和macro10 correct400；
  不自动训练到50。

## 2026-08-09 ECP六卡gradient profile完成并封存权重

- CPU seal commit`de28157`已push，独立frozen worktree及分支均绑定同一commit/upstream；historical
  `runs/outputs`只通过ignored symlink提供load-only资产。live双节点preflight选择空闲
  `gpu01:0,1,2|4,5,7`，GPU3的`nlge`进程和`gpu02`他人进程均未触碰；`/data1`quota余量约481GiB，
  本阶段预计峰值不足`.5GiB`。
- tmux单次run自然exit0，root=
  `runs/outputs/pi05_v6_ecp_gradient_profile_macro49_r6_lb20_mb10_de28157_20260809`。24 tasks、480/480
  unique queries、8/8/8 negatives、最长105帧，wall/input wait=`20.42496/.17998s`，peak
  allocated/reserved=`43,316,129,280/47,093,645,312` bytes，0 OOM/nonfinite；结束后六卡均14MiB。
- artifact assembler完整复核Git/config/HDF5、3+3 NUMA、physical/local rank、deferred-NCCL、ownership和
  panel，给出projection/ranking weights=`.006883349605446485/.010514451404229894`；compiler应用比例
  均`.25`，factor=`.10873/.02688`。证据已原样写入v2 config，只把profile解锁为`(3,(1,3))`，formal
  仍fail-closed。状态更新后全仓`259 passed in 30.88s`、compileall和`git diff --check`通过；下一步
  clean commit/push并从严格后继frozen worktree完成resume门。

## 2026-08-09 ECP objective与v2执行合同完成（尚未启动GPU）

- 唯一canonical objective已从whole-LoRA cosine+log-norm原位替换为Expert-Component Projection：
  correct用全38-target gauge-invariant coefficient SmoothL1到1，negative沿同一language-task expert做
  softplus margin；没有新增Writer/policy forward、dense BA、shadow/residual或部署scale gate。
- 新config=`configs/pi05_v6_ecp_policy_effective_writer_v2.json`。run、gradient/resume、checkpoint/trainer/
  RNG、adapter/episode均使用独立ECP v2 identity；旧v1 gradient weights、optimizer和resume evidence全部
  失效，live路径拒绝旧checkpoint。历史results只由analysis显式legacy family只读解析，不能恢复模型。
- metrics一次bulk CPU transfer记录projection loss、`a_correct/a_negative/a_margin`、expert component、
  generated/expert norm及38-target signed component/absolute numerator fraction；训练热路径仍是FP32低秩
  contraction，不materialize BA，不增加逐target同步。
- 小矩阵dense oracle、gauge invariance、orthogonal-energy invariance、SmoothL1解析gradient、ranking符号、
  batch broadcast、output→factor chain rule及v2 config/checkpoint/evaluator/analysis均纳入全仓CPU封印：
  `259 passed in 30.82s`，`git diff --check`通过。当前没有ECP GPU profile、训练或strict结果；
  下一步是clean push/frozen worktree后的一次六卡profile。
- architecture guard没有新文件、parallel version或parallel function family；其hard项来自既有超大
  checkpoint/contract/test owner被schema替换触碰。按design第34.5节记录cohesive exception：checkpoint
  仅+4行schema接线、contract +56行fail-closed协议、测试+26行，拆分不会减少责任；canonical
  `effective_objective.py`净减5行。

## 2026-08-09 v6-prior formal 0→50与四点strict分析完成

- formal root=
  `runs/outputs/pi05_v6_prior_formal_r6_lb20_mb10_eff15db_20260809`完整训练0→50，checkpoint10/25/50、
  optimizer/scheduler/sampler/6-rank RNG和completion齐全；50 macros约`1080.75s`，peak allocated/reserved
  约`43.266/47.094GB`，0 OOM/nonfinite/clip。
- 6卡18-worker correct400 roots的macro0/10/25/50均exit0、72/72 shards、400 rows，结果=
  `134/127/105/123`。macro50 wall=`871.42s`、overall/rollout-only吞吐=`.459/.632 rollout/s`；退出后
  gpu01:0/1/2/4/5/7回到14MiB，GPU3他人进程未触碰。
- strict分析root=
  `runs/outputs/pi05_v6_prior_checkpoint_curve_strict_paired_eff15db_20260809`；逐项锁定sealed validation
  tasks/languages、state/RNG/video identity，执行GPU/worker/batch拓扑只保留provenance。分析器定向
  `9 passed`、全仓带LIBERO assets`256 passed`，提交`24e7aae`已push。
- mechanism诊断确认whole-LoRA objective主要做径向收缩且绝对expert投影下降；按预注册门停止该路线，
  不续100/200、不扫权重、不为loser补六臂。下一步按design第34节实现objective-only ECP。

## 2026-08-09 v6-prior B10 gradient artifact完成并写回seal

- clean pushed/frozen `9c814ff6c880b77f109bf02445ff6364bb1c024d`在live空闲
  `gpu01:0,1,2,4,5,7`完成root=
  `runs/outputs/pi05_v6_prior_gradient_profile_macro49_r6_lb20_mb10_9c814ff_20260809`；tmux自然exit0，六卡
  回到14MiB，忙碌GPU3/6未触碰。
- root恰含run contract、单次fresh invocation、gradient profile和completion；24 tasks、480/480 unique
  queries、20/task、最长105帧、8/8/8 negatives、0 OOM/nonfinite。wall=`21.095109596s`、input wait=
  `.076318255s`、peak allocated/reserved=`43,305,942,016/47,093,645,312` bytes。
- assembler在frozen与canonical实现均通过，推荐expert/ranking weights=
  `.008355172068998324/.28570466890490887`；原样evidence已写入config，gradient/aux状态sealed、profile
  状态ready，formal仍blocked。全24 task records finite，唯一明显outlier是global39 reversed的高norm/
  负margin，属于后续逐task科学跟踪而非artifact异常。
- 下一步先commit/push该seal，从严格后继clean frozen worktree依次运行同一resumed root的fresh0→1和
  exact-resume1→3，再运行独立contiguous0→3；两条profile必须保持同commit/config绝对路径、同卡/
  NUMA、workers2和default allocator。

## 2026-08-09 v6-prior physical B16容量裁决与B10切换

- logical-B20 microbatch实现以clean pushed `eddba96d38d71fd89d80f9a23cc91881171bae84`封存；frozen
  worktree=`/data1/user/ymdai/worktrees/EMBER-v6-prior-gradient-b16-eddba96-20260809`。启动前live双节点
  preflight选择`gpu01:0,1,2,4,5,7`，六卡均14MiB且无compute进程，精确形成3+3 NUMA；`/data1`
  quota=`564,273,448/1,073,741,824 KiB`。忙碌`gpu01:3`、`gpu02:6/7`均未触碰。
- 首个root=`runs/outputs/pi05_v6_prior_gradient_profile_macro49_r6_lb20_mb16_eddba96_20260809`使用非持久SSH
  后台launcher，只写contract/invocation后exit0，缺少start/gradient/completion，判为无效托管尝试并
  原样保留。后续长期GPU任务统一使用tmux或仓库launcher，不再依赖该nohup方式。
- tmux retry root=`runs/outputs/pi05_v6_prior_gradient_profile_macro49_r6_lb20_mb16_eddba96_retry1_20260809`
  完整打印start，并在六rank第一条functional eager-attention一致OOM：申请`254MiB`，allocated=
  `42.49GiB`、reserved-unallocated=`1.25GiB`、free=`235.31MiB`。exit1、无gradient/completion；tmux
  自然结束，六卡释放。
- 按预声明规则不再做allocator retry或宽batch sweep。canonical config只把optimization/profile两处
  microbatch从16改为10；下一步clean commit/push、frozen worktree后用同logical B20和同六卡panel运行
  balanced B10+10。

## 2026-08-09 v6-prior六卡physical B20容量失败与logical-B20修复

- clean frozen/pushed`a17805c`的首次macro49 root=
  `runs/outputs/pi05_v6_prior_gradient_profile_macro49_r6_b20_a17805c_20260809`；当时live比较两节点后只用
  空闲`gpu01:0,1,2,4,5,7`，保持3+3 NUMA、physical/local rank、`NCCL_P2P_DISABLE=1`和deferred-NCCL。
  第一条PI05 policy functional B20在Gemma MLP OOM：申请`606MiB`，allocated=`42.29GiB`、
  reserved-unallocated=`1.29GiB`、free=`395.31MiB`。root只有`run_contract.json`和`invocations.jsonl`。
- 唯一allocator retry root=
  `runs/outputs/pi05_v6_prior_gradient_profile_macro49_r6_b20_a17805c_allocseg_retry1_20260809`，只增加
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。reserved-unallocated降到约`157MiB`，但allocated=
  `43.43GiB`且仍在申请`606MiB`时OOM，排除fragmentation主因。该root同样没有`gradient_profile.json`或
  `completion.json`；两次均exit1、禁止resume/合并，所选GPU随后自然释放且未干扰当时忙碌GPU3/6。
- canonical worktree已实现logical B20/physical B16+4：配置显式记录logical batch、microbatch和每task
  两次forward；完整有序logical-panel keyed independent Beta/Gaussian sampling让slices复现同一20个draw，
  functional leaf gradients以FP32按`16/20`和`4/20`累积。logical query count、full24 task mean和objective
  分布不变；seed使用固定SplitMix64整数mix而非SHA/MD5，policy checkpointing保持关闭。失败allocator
  retry不固化为runtime合同，B16/B10都从默认allocator开始。
- 第二轮审阅后聚焦合同/functional tests为`34 passed`，全仓带LIBERO assets回归`246 passed`，Black、
  compileall和`git diff --check`通过。config loader当时得到`20/16/2`，且只允许microbatch`{16,10}`；
  clean commit/push/frozen worktree随后完成，B16 live容量结果由上一节覆盖。
- 新`v6_prior_policy_batch.py`是logical/physical batch、RNG和runtime selection的唯一窄owner，被training、
  runtime和artifact contract共同使用；不是第二套训练路径。它使既有超大contract/runtime文件相对HEAD分别
  净减`2/1`行。architecture guard无hard violation、无parallel family；review只来自既有大文件与本次新增
  单owner/测试的规模，owner与退出条件已由本节和design 33.10明确。

## 2026-08-09 v6-prior六卡artifact seal实现完成（尚未启动GPU）

- gradient artifact assembler现从retained run contract、invocation、gradient profile与completion重建
  evidence；它精确核对canonical tracked config、clean pushed Git、6-rank A40/NUMA/affinity拓扑、
  macro49、train24×B20=`480/480` unique queries、24-task deterministic teacher demo与
  reversed/shuffled/cross-suite-wrong schedule，并回查frozen target manifest、HDF5 path/bytes及对应demo的
  真实frame metadata；compiler/factor gradient norm、推荐weights、input wait、peak VRAM及0 OOM/nonfinite
  同时封存。status字符串、stale tracked config或外部复制config均不能单独跳门。
- fresh/resume/contiguous assembler接入只读checkpoint inspector，验证manifest/file sizes、contract/
  cursor、600 Writer tensors、41 trainable tensors、6-rank RNG、AdamW两个moment fields、scheduler/AMP，
  并以预注册tolerance比较三宏步scientific metrics及macro1/3 checkpoint；gradient→profile还要求strict
  Git ancestry。
- retained训练metrics只在已有宏步边界同步中附带whole-step wall、input wait和peak allocated/reserved，
  没有为阶段计时增加热路径CUDA同步。先用真实证据决定是否需要workers/checkpointing候选，避免profile
  本身降低正式吞吐。
- 聚焦合同测试`27 passed`，全仓带LIBERO assets回归`238 passed`，compileall、Black check和
  `git diff --check`通过。本阶段没有启动CUDA、训练、rollout或长期实验；下一步clean push后从frozen
  worktree做live双节点/存储preflight和正式macro49六卡gradient profile。

## 2026-08-09 v6-prior单卡profile、vertical smoke与evaluation seal完成

- canonical纠偏在clean pushed `ded0c80`封存，并创建
  `/data1/user/ymdai/worktrees/EMBER-v6-prior-throughput-profile-ded0c80-20260809`。旧失败smoke
  worktree/local branch已在确认clean、远端与artifact保留后删除。
- live双节点preflight选择`gpu02:0`（UUID `GPU-2f8ac922-88b6-5c94-af12-70b67ddbdde8`），
  `/data1`个人quota约538GiB/1TiB；忙碌`gpu01:3`、`gpu02:6/7`均未触碰。
- profile root=
  `runs/outputs/pi05_v6_prior_writer_throughput_profile_val8x4_correct_gpu02g0_ded0c80_20260809`。
  同一32-request/1093-frame panel上batch8/16/32吞吐为`.911427/.905107/.906432 LoRA/s`，
  peak reserved=`12,824,084,480/12,847,153,152/12,847,153,152` bytes；按sealed rule选择batch8。
- vertical root=
  `runs/outputs/pi05_v6_prior_writer_vertical_smoke_val8x1_correct_b8_gpu02g0_ded0c80_20260809`。
  model load=`111.469s`，8-LoRA generation=`10.597s`，8 rows/4 successes、single attempt、
  0 retry/failure/OOM/nonfinite/forbidden reads；总wall=`325.540s`、rollout window=`196.816s`。
  Writer release、source reuse/no-reload和native cache全部成立；success只作execution evidence。
- artifact assembler从两个roots自然通过，config evaluation=`sealed`且gradient-profile=`ready`。
  定向seal tests随当前状态更新；当前无EMBER GPU进程，`gpu02:0`已回到0MiB。下一步是实现/验证
  Phase2 artifact verifier，clean push后做六卡macro49 gradient profile。

## 2026-08-09 throughput-first correction与authority收敛

- owner撤回为`.001953125`级BF16 batch-shape roundoff固定batch1的决定，要求吞吐和有效显存利用优先。
  canonical worktree现移除8次冗余direct Writer forward、`1e-5`逐tensor复现门、forced-FP32 LoRA cache、
  单physical batch梯度accumulation和多处热路径host sync；Writer batch改为A40 profile后选择。
- native LoRA storage经checkpoint metadata→run contract→cache descriptor/write/load接线：72 BF16 + 4 F32，
  `2,641,920` tensor bytes/entry。action DataLoader改为2 workers/spawn/persistent/prefetch2，serial、prefetch
  和prefix+resume rows测试一致。
- PI05 loss-only fast path删除formal每个policy forward的日志型host sync；Writer offsets、frame ordinal/order、
  language span/condition ownership和token packing也已清理重复barrier。新增单卡真实batch profile、artifact-
  backed evaluation seal、无循环状态图和task-expert窄authority loader。
- GPU preflight现记录device name并在任何模型load/worker spawn前拒绝忙卡、非A40；当时普通evaluator还
  错误附加了六卡cap（随后经owner澄清并删除）；
  profile入口还要求clean pushed、validation/correct/without-replacement、单卡单replica/generator及真实
  `8/16/32`候选，并在独立单卡worker中再次live preflight与核对checkout。
- batch吞吐使用同一个32-request longest-first panel和同一总sampled frames；8/16/32只改变物理forward
  分批，避免大batch因额外加入短视频而获得混杂优势。每个候选记录实际forward分组、native D2H wall、
  peak显存和headroom，artifact assembler要求三行完整panel严格一致。
- 真实validation8×4-state CPU prepare已通过：32 requests、historical Writer 600 tensors/12,064,064
  values、deployment expert-bank reads=0、cache 72 BF16 + 4 F32；临时root已清理，未初始化CUDA。
- authority、README、design、findings和task plan已收敛；相关定向回归`68 passed`、全仓`227 passed`，
  compileall与`git diff --check`通过。当时的下一执行门是clean commit/push与单卡profile/vertical smoke；
  该门现已由本文件顶部的新证据完成。

## 2026-08-09 v6-prior batch8复现失败、根因定位与batch1修复（batch1已撤回）

- clean frozen`30b2ccf`在live空闲`gpu02:0`完成首次historical warm-start启动；model load=`117.190s`、
  NUMA0绑定正确，但batch8输出在cache前超过`1e-5`复现门，launcher按合同退出。失败root=
  `runs/outputs/pi05_v6_prior_warmstart_reproduction_smoke_validation8_correct_gpu02g0_30b2ccf_20260809`；
  0 cache/rollout，GPU释放。
- 同卡最小诊断得到direct-repeat max-abs=`0`；duplicate batch8与heterogeneous batch8对direct均为
  `.001953125`，mean约`4.70e-5`，peak allocated=`11,700,880,384` bytes。结果定位为BF16 batch-shape
  数值路径，不是串样、padding或随机性；诊断artifact保存在失败root的`diagnostics/batch_equivalence.json`。
- 当时没有放宽复现阈值，并把canonical Writer错误固定为batch1；全仓`211 passed`只证明了该历史合同。
  此决定已由上节throughput-first裁决撤回，失败root仍不resume且不作为当前下一步依据。

## 2026-08-09 v6-prior canonical evaluator封存

- clean pushed`bca3f6d`把Expert-Manifold部署从rejected hard-route原位替换为v6-prior raw-video完整
  Writer。删除hard-route config/class/topology专属测试；CLI改为config+Writer checkpoint+raw video+
  condition，旧expert-bank/feature-cache部署参数fail closed。
- adapter/episode升为v5；historical macro400显式登记为method macro0，本方法checkpoint登记真实method
  macro。correct/same/wrong/shuffled/reversed各读恰好一条video，no-video不读frames并返回identity；
  LoRA cache后释放Writer并复用同一source policy。
- 全仓`210 passed`、compile/diff check通过。真实只读asset gate确认600 state tensors、12,064,064 values、
  validation8映射8条视频；真实CLI prepare生成8-entry cache合同，deployment expert-bank reads为0。
- historical correct smoke runtime会在cache写入前把每个batched output与同输入逐episode direct forward
  比较；state names/shapes/nonfinite或max-abs`>1e-5`立即fail closed，并把entry/tensor counts、wall和最大
  差异写入generator evidence。
- 本阶段没有启动CUDA、训练、rollout或长期实验。下一步从包含本提交及authority更新的clean pushed
  frozen worktree，在live双节点GPU/quota preflight后用一张空闲A40完成batch-vs-direct warm-start
  reproduction和8-task cache/release/rollout smoke；通过前六卡gradient profile保持blocked。

## 2026-08-09 v6-Prior训练runtime封存

- clean pushed`dd57edc`完成六卡train24 runtime、policy-effective三项output-gradient组合、one-shot
  counterfactual schedule、single-flat-allreduce和hashless exact-resume checkpoint；profile/formal均由sealed
  config fail-close。
- 全仓`215 passed`、compile与diff check通过。真实CPU gate为24 tasks、206,346 rows；profile macro49
  为480 unique B20 queries并包含最长105 sampled-frame视频。冻结/训练参数精确为
  `7,060,992/3,714,304`。
- 当前没有GPU工作。下一步原位替换rejected hard-route evaluator/runtime，clean push后才做单卡历史
  warm-start输出复现smoke；该门通过前六卡gradient profile无法启动。

## 2026-08-09 v6-Prior Policy-Effective Temporal-Ranking设计封存

- hard/soft/sparse 24-expert部署字典已由strict结果关闭。新design authority写入
  `docs/action_forecast_writer_video_expert_manifold_design.md`第33节：部署恢复历史v6-fast完整动态Writer，
  experts只作train24 policy-effective监督，不进入在线路由。
- CPU只读核对macro400 checkpoint为600 tensors；encoder/Core/transition/Procedure与compiler/heads的
  ownership分别为483/41 tensors和`7,060,992/3,714,304` parameters。新候选只训练后41 tensors。
- 目标固定为correct positive functional + exact effective-BA expert direction/norm + bounded correct-over-
  reversed/shuffled/wrong ranking；不最大化negative action loss，不加scale/gate/bypass/few-shot。
- 当前仍未启动新GPU工作。下一步原位替换rejected evaluator/runtime；只有replacement clean push和
  CPU门通过后才做单卡warm-start reproduction smoke及六卡formal profile。

## 2026-08-09 Hard-routed correct80完成、严格淘汰并释放GPU

- clean frozen launch=`99c4506`、scientific seal=`1d58781`在live空闲`gpu02:0,1,2`完成固定
  validation8×states0--9 correct/without-replacement screen。root=
  `runs/outputs/pi05_expert_manifold_hard_routed_correct80_screen_noreplacement_seed7_1d58781_20260809`；
  36/36 jobs、80 unique rows/LoRAs/cache、9 workers attempt1/exit0，0异常/forbidden reads，三卡已回到
  0MiB/P8。score=`3/80`、breadth2，触发预注册淘汰门；没有启动160/400或五臂。
- CPU posthoc写入同root的
  `hard_route_strict_screen_and_policy_effective_route_audit_v1.json`。相对strict same-video soft15为
  retained/gained/lost/both-fail=`1/2/14/63`，exact `p=.0041809`；identity、RNG和video/order全配对。
- 80 online LoRA对24 raw experts的exact effective route审计得到nearest cosine中位/最小=
  `.998544/.997096`、second gap最小`.35133`、11 experts被选择；79/80匹配旧soft argmax。结果否定
  24-expert hard/soft部署字典，不做mixture内部调参；下一阶段转向v6-prior transferable
  policy-effective Writer的CPU设计与唯一canonical替换。

## 2026-08-09 Hard-routed online smoke通过并seal

- clean pushed launch=`12c8d1e`在live空闲`gpu02:0`完成validation8×1-state correct smoke；root=
  `runs/outputs/pi05_expert_manifold_hard_routed_online_smoke_gpu02_14495d9_20260809`。8 rows/generated/cache、
  3 workers exit0、attempt1、0 error/retry/OOM/nonfinite/forbidden reads；GPU已回到0MiB/P8。
- model load=`117.475s`，generation 2×batch4=`10.497s`，peak allocated/reserved=
  `10,576,896,000/11,238,637,568` bytes；Writer release、source-policy reuse/no-reload成立。总wall=
  `315.902s`，`0/8`只作execution smoke。
- posthoc artifact=`hard_route_online_smoke_route_audit_v1.json`：8/8 LoRA精确匹配one-hot expert，effective
  cosine最小`.999999799`，覆盖7 experts；7/8与旧soft argmax一致，Long-2 state0在旧margin`.000664`处
  从ordinal12翻到13。精确smoke/audit evidence写回后formal=`sealed`；下一步为固定correct80。

## 2026-08-09 Hard-routed canonical实现与真实资产CPU门

- 原soft policy-effective config/runtime已由Git保存并从canonical删除；新config=
  `configs/pi05_video_expert_manifold_hard_routed_policy_effective_v2.json`，实现提交`1619631`已push。
  runtime固定signed-argmax one-hot support1，adapter/episode升v4；全仓182项CPU tests与compile通过。
- 真实24-expert/train24×50 feature cache只读分析写入
  `runs/outputs/pi05_expert_manifold_hard_routed_cpu_real_assets_20260809/analysis.json`。24/24 centroids和
  1,200/1,200 videos self-route；ordered/reversed、ordered/fixed-shuffle selection changes=
  `1200/1200`、`699/1200`，24 experts全部覆盖。
- zero identity、one-hot support/sum、finite state均成立；24×38 effective target cosine中位/最小=
  `.998982/.961962`。旧correct80系数的hard argmax覆盖11 experts。formal只推进到
  `blocked_until_live_a40_online_smoke`，尚未启动GPU或新闭环实验。

## 2026-08-09 Policy-Effective correct80完成、审计与GPU释放

- frozen`ffed252`的validation8×states0--9 correct/without-replacement screen已自然完成：root=
  `runs/outputs/pi05_expert_manifold_policy_effective_correct80_screen_noreplacement_seed7_ffed252_20260809`，
  score=`15/80`、breadth=`5/8`；36 jobs、80 rows/cache LoRAs和9 workers全部attempt1/exit0，0异常/
  forbidden reads，三张A40均已释放。按预注册门不扩跑160/400或五臂。
- CPU只读写入`strict_screen_and_paired_audit_v1.json`：相对source/addressless/address-binding/raw/v6-fast
  的gained/lost依次=`13/7,12/7,9/7,6/3,5/18`；前四个identity/RNG严格配对，其中后三个还exact
  same-video，v6只作different-video same-state参考。
- CPU exact有效空间写入`policy_effective_geometry_correct80_v1.json`：覆盖全部3,160 generated pair、
  1,920 generated-expert pair和80 matched raw pair，无sketch近似。norm/stable/top=
  `4.148/1.234/.847`，current/raw cosine=`.958`，same/cross/task-mean=`.989/.703/.712`，nearest expert=
  `.641`。下一步只设计video-routed hard-one-hot expert单变量screen，先判别soft mixture与basis support。

## 2026-08-09 Policy-Effective online smoke通过并seal formal

- attempt0因detached worktree缺少`@{upstream}`在GPU加载前被合同拒绝，物理0始终0 MiB；partial root与
  2,006-byte日志保留为工程失败证据。retry改用clean pushed`321bded`、有upstream的冻结run branch。
- live只使用空闲`gpu02:0`完成validation8×1-state correct/without-replacement纵向链路；有效root=
  `runs/outputs/pi05_expert_manifold_policy_effective_online_smoke_retry1_gpu02_fb5b367_20260809`。8 rows、
  8 unique LoRAs/cache entries、2个batch4、3 workers exit0，0 retry/failure/OOM/nonfinite，四类forbidden
  reads均0；Writer释放后source policy原位复用且未reload。
- generation wall=`11.070s`、peak allocated/reserved=`10,576,896,000/11,238,637,568` bytes；整panel
  wall=`310.715s`。`1/8` success只登记为execution smoke，不作性能推断。GPU已自然回到0 MiB且无进程。
- 专属evidence已写回policy-effective config并将formal切为`sealed`。下一步是clean push该seal后，
  预注册小规模strict correct panel；只有absolute/breadth支持才扩到400和五臂。

## 2026-08-09 Policy-Effective canonical实现与CPU真实资产封存

- raw-factor Writer/config已原位替换为唯一Policy-Effective Barycentric runtime；旧实现只由Git与formal
  artifacts保留。新config=
  `configs/pi05_video_expert_manifold_policy_effective_barycentric_v1.json`；adapter/episode schema升为v3，
  旧cache不能冒充。实现提交=`469e033`且已push；其后专属A40 smoke已在上节通过并seal。
- 全仓`182/182` tests、compile和真实fixed-asset inspector通过。CPU artifact=
  `runs/outputs/pi05_expert_manifold_policy_effective_cpu_real_assets_20260809/analysis.json`：0 parameters、
  68,863,192 buffer bytes、build`2.33s`、batch24 compile`.85s`、zero identity exact。
- one-hot expert/demo0 intended effective cosine中位=`.99838/.99836`；ordered/reversed coefficient L2最小
  `1.268`。demo0 generated cross-task cosine中位`.203`，norm/stable/top=`4.179/1.125/.910`，A/B RMS=
  `.01891/.00846`，q/v/action B-column cosine=`.815/.813/.455`，16 rank coordinates全active。
- architecture gate无parallel Writer family：保留的`TopologicalLoRAChunkLayout`仅供现有expert/cache
  分析脚本做exact layout inspection，不参与deployment；active source/config总行数没有扩张。

## 2026-08-09 Causal Barycentric correct400完成、负裁决与compiler诊断

- clean frozen eval自然完成，root=
  `runs/outputs/pi05_expert_manifold_causal_barycentric_correct400_noreplacement_seed7_0397be6_20260809`。
  72/72 jobs、400 unique rows/LoRAs、18 workers attempt1/exit0、0 retry/error/OOM/nonfinite，四类
  forbidden reads为0；GPU自然释放。score=`63/400`、breadth=`5/8`，未过absolute门，未启动五臂。
- `strict_panel_and_paired_audit_v1.json`封存same-video对照：相对source/addressless gained/lost=
  `46/31`、`p=.1100`，相对address-binding=`27/39`、`p=.1753`。能力从Goal-6迁到Object-1，仍是
  turnover而非共同积累。
- `generated_lora_geometry_and_coefficients_full400_v1.json`覆盖400 LoRAs并用48个exact pair校准
  effective-BA sketch；inversion output-token relative-L2中位`.000884`。LoRA norm/stable/top=
  `3.958/1.155/.894`，same/cross/task-mean cosine=`.988/.685/.697`，coefficients abs support中位`13.75`。
- CPU反事实新增`contrastive_coefficient_reader_loo_v1.json`和
  `rectified_prototype_reader_loo_v1.json`。它们证明order discrimination可改善，但held policy-update
  direction/amplitude仍不足，故不在下一轮同时改reader。
- 当前只做Policy-Effective Barycentric CPU feasibility：比较shared joint rank-16 projection和
  per-query exact mixture compression。没有启动GPU、训练或新长期实验。

## 2026-08-09 Policy-Effective compiler CPU门通过

- CPU只读分析覆盖400组真实recovered coefficients、24个step2000 experts和全部38 targets；输出=
  `runs/outputs/pi05_expert_manifold_causal_barycentric_correct400_noreplacement_seed7_0397be6_20260809/`
  `policy_effective_compiler_feasibility_full400_rank128_v2.json`。没有启动GPU或修改实验状态。
- pure affine effective combination的norm中位`2.220`、expert ratio`.527`，因幅度稀释被拒绝。
  per-target unit-effective-direction + expert-envelope log norm的norm=`4.155`、ratio=`.986`。
- shared rank96 + public best-rank16对400 queries的captured-energy/cosine中位=
  `.99365/.99682`、最小=`.99065/.99532`；24 experts captured-energy中位/最小=`.99677/.99331`。
  full-span rank16的8-task样本中位`.99523`，表明public rank16和rank96 basis均不是首要瓶颈。
- 已更新design authority，下一步在唯一canonical runtime原位替换raw-factor compiler并重做CPU合同；
  video representation、ridge coefficients、one-shot与信息墙不变。

## 2026-08-09 Causal Barycentric strict correct400预注册

- 05:10 CST live比较两节点后预选`gpu01:0,1,2|4,5,7`六张14MiB/0%空闲A40，保持NUMA0/1各三张；
  物理3他人VLLM、物理6和`gpu02:6/7`他人进程均不触碰。gpu01 host available memory=`479GiB`，
  `/data1`个人quota=`561,350,572/1,073,741,824 KiB`。
- 400套FP32完整LoRA加结果/log/queue保守新增低于3GiB；frozen branch/worktree、fresh root/log/tmux、
  exact correct/without-replacement 400-row命令和72-job/18-worker信息墙门已登记在`task_plan.md`。相关
  目标登记时全部不存在；先clean commit/push launch record，真正启动前再次live检查六张目标卡。

## 2026-08-09 Causal Barycentric online smoke通过并seal formal

- 在live比较`gpu01/gpu02`后只使用空闲`gpu02:0`完成validation8×1-state correct/
  without-replacement纵向smoke；忙碌的`gpu02:6/7`及`gpu01:3`均未触碰。root=
  `runs/outputs/pi05_expert_manifold_causal_barycentric_online_smoke_gpu02_3c8ce25_20260809`，8 rows、
  8 unique LoRAs/cache entries、2个batch4、3 workers exit0，0 retry/failure/OOM/nonfinite/forbidden
  reads；Writer模块释放后source policy原位复用且没有reload，GPU自然释放。
- generation wall=`9.895s`，peak allocated/reserved=`10,645,668,864/11,305,746,432` bytes；8套LoRA
  norm/stable-rank/top-energy中位=`3.9802/1.1555/.89243`，cross-task/nearest-expert cosine中位=
  `.69277/.65624`，16/16 coordinates active。`1/8` success只登记为execution smoke。
- 精确evidence已写回canonical config并切为`sealed`；真实24-basis/validation panel的
  `require_formal=True` inspector和全仓180/180测试通过。下一步是clean commit/push、建立新的frozen
  worktree并预注册strict correct400；启动前再次live检查设备、quota和fresh root。

## 2026-08-09 Causal Barycentric canonical实现与CPU封存

- clean pushed`1d9d030`新增
  `configs/pi05_video_expert_manifold_causal_barycentric_v1.json`并把evaluation改为固定step2000
  expert bank + train24×50 feature cache + 在线一条teacher video；learned Writer checkpoint参数退役。
- 旧`train_expert_manifold_writer.py`、writer training/checkpoint和learned decoder路径及专属测试已删除；
  `task_plan.md`同时收敛为当前执行计划，旧命令只由Git、ledger与formal artifacts保存。
- 全仓180/180、py_compile、diff check通过；architecture guard无hard violation、无parallel family，
  active diff净删941行。真实24-basis CPU只读：0 parameters、168 chunks、1,287,168 valid values、
  one-hot最大误差`2.235e-8`、zero identity exact、affine sum误差`1.192e-7`、24/24 ordered/reversed不同。
- config formal状态保持`blocked_until_live_a40_online_smoke`。本阶段未启动GPU；下一步live比较两节点与
  quota后，只用一张空闲A40完成validation8×1-state online generation/cache/release/rollout smoke。

## 2026-08-09 Address-binding strict负裁决与barycentric CPU设计门

- clean eval commit`033db91`的address-binding macro50 correct400自然完成并释放六卡：正式root=
  `runs/outputs/pi05_expert_manifold_writer_addressbind_correct400_noreplacement_seed7_macro0050_925e7b1_20260809`，
  72/72 jobs、400 rows、400 unique LoRAs、18 workers全exit0，0 retry/error/OOM/nonfinite；score=
  `75/400`、breadth=`4/8`。严格审计artifact为`strict_panel_and_paired_audit_v1.json`。
- 相对旧addressless macro50，task/state/env/policy-noise公共前缀和400条teacher video schedule全相同，
  gained/lost=`31/4`；相对source base也是`31/4`。相对v6-fast为`18/86`，但v6历史teacher schedule
  不同，已明确标成same-state reference而非same-video严格配对。
- full400 exact effective-BA artifact=`generated_lora_geometry_full400_v1.json`：norm/stable/top=
  `3.201/1.318/.778`，same-video/cross-task/task-mean cosine=`.99791/.94197/.94270`，nearest expert=
  `.12734`。据此正式拒绝同root resume100和五臂。
- 新CPU-only root=
  `runs/outputs/pi05_expert_manifold_causal_barycentric_loo_step2000_cpu_20260809`完成24 folds×50 videos×
  correct/reversed/phase-shuffled×两种重构，共7,200套完整LoRA精确审计。选择ridge `.3`和topological
  direction/log-scale重构；correct/reversed/shuffled effective target cosine=
  `.38302/.09900/.18539`，correct norm/stable/top=`3.84385/1.15056/.89540`。全过程CPU只读，未启动
  GPU、未改实验状态；下一步是authority后实现canonical runtime与CPU合同。

## 2026-08-09 Address-binding formal0→50与strict correct400预注册

- clean pushed launch-record`925e7b1`从identity fresh完成50/50 finite macros、1,200 train24 one-shot
  conditions及完整macro50 checkpoint；body=`10.204s`，peak allocated/reserved=
  `761,802,752/836,763,648` bytes，0 OOM/nonfinite，六卡自然释放。
- 在live空闲`gpu02:0`完成train24 demo0内部诊断后卡已释放。cross/axial的chunk-rank energy仍约
  `1e-6`，addressed恢复到`.493/.477`，output为`.467/.616`；raw/own-effective target cosine中位=
  `.1177/.1342`。LoRA norm/stable-rank/top-energy=`3.360/1.349/.757`且16 coordinates active。
- generated task-pair cosine仍`.8686`且只有8/24 own-nearest，故不凭内部信号resume。macro50 strict
  correct400的新frozen eval branch/worktree/root、6卡r3/generator3/batch4命令与400-row/72-job门已写入
  `task_plan.md`；启动前重新live复核。

## 2026-08-09 Address-binding formal0→50预注册

- 新图profile/smoke evidence seal已由clean`448f760`通过193项全仓CPU回归并push；config正式状态为
  sealed，profile权重永久弃用。formal branch/worktree/root/log/tmux均使用全新名字且现场不存在。
- 03:19 CST live比较选择`gpu01:0,1,2|4,5,7`六张空闲A40的3+3 NUMA；物理3他人VLLM、物理6和
  `gpu02:6/7`他人进程均不触碰。个人quota533.2GiB/1TiB，首段预计新增低于300MiB。
- exact identity-fresh 0→50命令、world6/24-task/scheduler800合同、checkpoint完整性门与macro50后
  strict correct400/internal-first裁决顺序已写入`task_plan.md`。真正启动前仍需再次live复核，不传
  resume，不加载任何profile或旧decoder权重。

## 2026-08-09 Address-binding online smoke通过并重新seal formal

- clean pushed`eb32f3f`在live空闲`gpu02:0`完成8-task×1-state correct/without-replacement smoke；
  8/8 unique rows、8 unique LoRA references、8 cache entries和3 workers均首次完成，0 retry/failure/
  OOM/nonfinite。GPU自然释放，物理6/7他人进程及gpu01物理3他人VLLM均未触碰。
- 一个generator以2个batch4生成8套FP32 LoRA，generation wall=`9.731s`；peak allocated/reserved=
  `10,576,056,320/11,182,014,464` bytes，释放Writer后为
  `9,391,467,520/9,651,093,504` bytes。source policy原位复用、无reload，四类forbidden reads为0。
- CPU只读检查8套LoRA共608 tensors全finite；norm/stable-rank/top-energy中位=
  `.70069/1.98260/.51202`，16个rank coordinates全active。`1/8` success只作execution smoke。
- 新profile/smoke evidence已写入`configs/pi05_video_expert_manifold_v1.json`并共同绑定address-binding
  identity，meta formal重新seal。下一步在clean pushed launch-record/frozen worktree从identity fresh
  只跑0→50，不加载profile权重，随后先做strict correct400与内部几何/target传递裁决。

## 2026-08-09 Address-binding core profile通过与online smoke预注册

- clean pushed`a3666ba`的fresh0→1/resume1→3与独立contiguous0→3均自然完成并释放六卡。
  三步科学指标逐值一致；macro1全部checkpoint files、macro3 Writer和六份rank RNG逐字节一致，
  trainer optimizer/scheduler语义精确相同。仅trainer容器raw serialization不同。
- `address_norm`在macro3的Adam一阶/二阶矩最大值=`8.213e-7/3.737e-14`，macro1→3权重最大变化
  `1.621e-5`且finite。resume/contiguous峰值reserved=`897,581,056/836,763,648` bytes，全部
  physical/local/NUMA/deferred-NCCL合同正确，0 OOM/nonfinite；profile权重弃用。
- 03:01 CST重新live选择`gpu02:0`做8-row online generation/cache/release/rollout smoke；物理0空闲，
  6/7他人进程不触碰。fresh root/log不存在，exact命令与门已登记；先clean push并在启动前复核。

## 2026-08-09 Address-binding A40 reprofile预注册

- 02:52 CST live比较选择`gpu01:0,1,2|4,5,7`六张空闲A40，保持3+3 NUMA；物理3他人VLLM、
  `gpu02:6/7`他人进程均不触碰。gpu02空闲0--5为4+2 NUMA，故不用。
- `/data1`个人用量532.4GiB/1TiB，新fresh/resume与contiguous roots均不存在，预计合计低于1GiB。
  exact三条命令和byte/semantic/NUMA门已登记到`task_plan.md`顶部；先提交推送并创建frozen worktree，
  真正启动前再次live看卡。尚未启动GPU。

## 2026-08-09 Zero-preserving topology-address binding实现与CPU封存

- clean pushed`cd95281`已在唯一`VideoConditionedTopologicalWriter`内完成单变量修订：
  `topology_address=chunk_query+rank_query`，cross-attention与四个axial blocks产生的动态video latent
  先经RMSNorm，再与独立RMSNorm后的静态address逐元素相乘，随后才进入共享output projection。
  address没有独立输出支路；scale head仍只读动态latent。
- zero/phase-constant video的动态值为零，乘积与完整LoRA增量精确为零；ordered/reversed仍产生不同
  动态值。合成回归同时证明即使cross-attention输出对所有坐标相同，绑定后chunk/rank centered
  energy均大于`.1`，且zero-output首步打开head后address norm获得非零梯度。
- 新结构不兼容旧macro50 checkpoint，旧A40 profile/online-smoke证据已从meta formal seal移除；
  config现为`blocked_until_live_a40_profile_and_online_generation_smoke`，且重新seal时两份evidence都
  必须显式记录本address-binding identity，防止复用旧decoder证据。
- 同步修正Writer rollout cache资源估算：生成compute为BF16，但落盘LoRA state实际为FP32；400 entries
  tensor预算应为`2,064,364,800` bytes。聚焦47/47、正式assets环境全仓192/192、compileall和
  diff check通过；architecture guard无hard violation或parallel family。尚未启动任何新GPU工作。

## 2026-08-09 Macro50 correct400完成、负裁决与地址塌缩定位

- clean pushed`9406802`的r2评测自然完成并释放GPU：72/72 jobs、400 rows、18 workers全exit0且
  attempt1，0 retry/error/OOM/nonfinite；400个one-shot correct LoRA、teacher frame used与四类
  forbidden reads零计数全部闭合。正式score=`48/400`，只有3/8 tasks非零。
- 与source-base旧formal结果严格按task/state/env/policy RNG配对，aggregate同为48，gained/lost=
  `5/5`；因此当前checkpoint等价于没有共同闭环增益。实际cache为FP32 `2,064,364,800` tensor bytes，
  已更正此前低于1.5GiB的估算。
- CPU内部诊断覆盖400 generated LoRA与24个step2000 experts：norm中位`4.549`，但stable rank=
  `1.0000014`、nearest-expert cosine中位`.00797`。train24 demo0 effective target cosine同样只有
  `.01081`；不是held-task泛化单点失败。
- 纵向probe把根因定位到cross-attention：rank/chunk centered energy从query约`.48/.49`降到
  `~1e-6`，axial输出再降到`~1e-8/~1e-10`，而expert target约`.936/.994`。原轨迹不resume100，
  下一动作是在唯一Expert-Manifold model原位设计zero-preserving video×topology address binding；
  先走architecture gate与CPU合同，不立即启动新GPU实验。

## 2026-08-09 Macro50 correct400 CPU prepare失败与worktree-path根修

- 首次formal correct400在adapter inspection阶段停止：0 CUDA worker、0 LoRA cache、0 scientific row，
  六张目标GPU始终空闲；root仅有LIBERO临时配置并已写`ABORTED.md`，永久不得resume。
- 根因是training/evaluation各自使用合法clean frozen worktree，而inspector把同一config的绝对路径前缀
  当作科学身份。canonical改为仓库相对路径相等，并新增bytes相等；schema、method、information wall、
  topology、meta、source、checkpoint manifest等既有门全部保留。
- 同worktree、跨worktree、错relative path、错bytes回归闭合；聚焦36/36、真实macro50 formal inspector、
  全仓189/189与compileall/diff check通过。下一步clean push后使用全新replacement root。
- 修复已clean push为`d59841e`；r2 frozen branch/worktree、fresh output/log/tmux和exact command已登记，
  保持原400-row科学合同与6卡r3/batch4资源合同，等待重新live preflight。

## 2026-08-09 Expert-Manifold formal0→50完成并预注册correct400

- clean pushed`446cd42`在`gpu01:0,1,2|4,5,7`从identity fresh自然完成50/50 finite macros；
  macro50 checkpoint含Writer、trainer、manifest和六份rank RNG，日志无异常，GPU自然释放。
- 训练body=`10.239s`，peak allocated/reserved=`737,273,344/815,792,128` bytes；run contract的
  branch/commit/upstream、3+3 NUMA、physical/local rank、deferred NCCL、P2P-disable、Ring/Simple和
  single-flat mean逐项通过。last loss/raw/direction/log-scale=`.099576/7.8499e-5/.97642/.018554`，
  只作surrogate证据。
- macro50 correct400已固定fresh root、frozen eval branch/worktree、6卡×3 replicas、每卡3 generators、
  batch4和400-row验收门。先闭环和内部分析，不因loss直接resume到100。

## 2026-08-09 Expert-Manifold formal0→50预注册

- clean pushed`fcaf733`完成canonical退役后，预注册唯一identity-fresh formal root、frozen branch/
  worktree、exact world6命令和macro50门；不加载profile权重，不改变sealed模型、target或optimizer。
- 01:47 CST live比较选择`gpu01:0,1,2|4,5,7`六张空闲A40，3+3 NUMA；GPU3他人VLLM和gpu02:6/7
  他人进程均不触碰。启动前仍须再次复核。
- `/data1` quota=`556,052,656/1,073,741,824 KiB`，首段预计新增低于300MiB。训练只先到macro50，
  随后做strict correct400和内部传递分析，不以reconstruction loss代替闭环裁决。

## 2026-08-09 Expert-Manifold成为唯一canonical Writer

- 在formal启动前完成design第12节的原位退役：旧K4/AS/RL executable、入口、配置和专属测试已删除，
  历史只由Git、文档与formal artifacts保存；未启动GPU、未改实验root或checkpoint。
- 通用data、functional LoRA、topology、evaluation cache/runtime继续保留并收紧为Expert-Manifold
  one-shot schema；动态rollout不再有旧adapter分派或全局LoRA-B scale入口。
- CPU-only全仓`186 passed`，compileall与diff check通过；architecture guard为review但hard violations和
  parallel families均为空，active source净删约13k行。下一阶段是从clean pushed identity fresh formal
  分段训练，不使用任何profile权重。

## 2026-08-09 Online smoke通过、formal seal与K4退役触发

- replacement root
  `runs/outputs/pi05_expert_manifold_writer_macro0003_online_smoke_r2_gpu02_c33a16b_20260809`已自然完成：
  8/8 unique rows、8个唯一LoRA references、8 entries/2个batch4，3 workers均attempt1/exit0，0
  retry/failure/OOM/nonfinite；总wall=`318.488s`。
- generation wall=`12.634s`，peak allocated/reserved=`10,576,054,272/11,182,014,464` bytes；
  Writer/encoder释放后source policy原位复用，cache/evidence有效，forbidden reads全0。GPU自然释放。
- `1/8` success只登记为execution smoke，不进入任何性能比较。profile与online smoke证据已写入
  `configs/pi05_video_expert_manifold_v1.json`，meta `formal_run.status=sealed`；新增配置回归拒绝缺失或
  被篡改的seal evidence。
- design第12节的旧K4 executable移除触发现已满足。下一步不启动GPU，先原位删除旧model/training/
  checkpoint/live-generation路径并保留通用data/topology/functional/evaluation组件，完成CPU回归和push。

## 2026-08-09 Online smoke CPU-only首次失败与scoped repair

- 首次macro3 smoke在prepare时被authority拒绝，尚未产生run contract、CUDA worker或scientific row；
  root只含LIBERO config和`ABORTED.md`，不得resume。GPU0始终0MiB。
- 根因是同一final source checkpoint在formal training descriptor含`source_run_summary`、smoke inspection
  将其写为`null`，其余字段完全相同。canonical仅为这一模式差异做归一化，并现场验证training summary
  path/bytes/schema；没有放宽checkpoint/model identity。
- 新regression同时覆盖合法缺省、不同checkpoint拒绝、summary bytes变化拒绝；聚焦58/58、正式assets
  全仓224/224，真实macro3 authority smoke inspection通过。已登记replacement fresh root，等待clean push
  后重新live preflight。

## 2026-08-09 Flat-reduction core profile通过并封存online smoke

- clean pushed`b00024b`的flat-reduction fresh0→1/resume1→3与独立contiguous0→3自然完成；三步科学
  metrics逐值一致，macro1/macro3 Writer及六份macro3 RNG逐字节一致，0 OOM/nonfinite。macro3
  trainer仅raw serialization不同，反序列化optimizer/scheduler逐项0差异。
- 两root峰值allocated/reserved分别为`736,117,760/876,609,536`和
  `735,831,552/815,792,128` bytes；run contract的3+3 NUMA、physical/local rank、无DDP wrapper、
  single-flat reduction、P2P disable、Ring/Simple全部通过。旧DDP hidden-state问题闭合，profile权重弃用。
- 00:48 CST重新比较两节点，选择空闲`gpu02:0`做唯一online smoke；忙碌`gpu02:6/7`与`gpu01:3`
  不触碰。固定8-task×1-state、1 generator、batch4、3 replicas和macro3 profile checkpoint；只验证
  generation/cache/release/rollout纵向合同，不把success当科研分数。exact command与验收门已登记。

## 2026-08-09 Meta profile exact-resume失败与DDP static-graph候选修复

- clean`ac56ab8`首轮六卡fresh/resume/contiguous均自然完成、NUMA与资源合同健康，但macro3 Writer
  byte comparison失败；macro1完整checkpoint跨roots逐字节一致，macro3最大tensor差约`1.30e-5`。
  按门否决该profile，未启动online smoke/formal。
- 两组临时诊断依次启用deterministic algorithms+cuBLAS workspace、再强制math-SDPA，仍稳定复现
  resume/contiguous双轨，排除checkpoint state和随机kernel。诊断weights永久弃用。
- canonical meta DDP现对齐仓库既有训练器：`static_graph=True`、`broadcast_buffers=False`、
  `find_unused_parameters=False`，run contract显式封存static/buffer语义。下一步CPU回归、clean/push后
  从全新roots重做fresh/resume/contiguous，不沿用失败root。
- 候选修复已作为clean pushed`12727b8`通过聚焦46/46；新static-graph reprofile roots、commands与
  exact byte gate已登记，等待live六卡复核后启动。
- 该reprofile在macro1的多次`no_sync` backward触发PyTorch reducer内部断言，0 optimizer step、无
  checkpoint，root不得resume。随后临时dynamic/no-buffer probe完整跑通，但resume/contiguous仍按
  原A/B分叉，否决buffer和static-graph两个候选。
- canonical正在替换为显式、无历史的单flat-gradient Ring/Simple all-reduce mean；local microbatch1、
  每rank4 tasks、24-task等权、loss/optimizer/RNG均不变。需CPU合同和新clean GPU profile确认。
- retained实现与config合同已闭合：聚焦49/49、全仓223/223通过；architecture guard无hard violation或
  parallel family，新collective tests独立放入聚焦文件，既有>800行测试文件只缩不增。尚无GPU通过。
- clean pushed`c33a16b`现为stateless flat-reduction seal；新roots、固定Ring/Simple commands与
  fresh/resume/contiguous exact门已登记，旧DDP/static/probe roots全部禁止复用。

## 2026-08-09 Task-expert五点封存、step2000 target与meta profile preflight

- step1500/2000两组正式评测在23:59 CST自然完成：各1200 unique rows、126/126 jobs、9/9 workers
  exit0、attempt1、0 retry/failure；跨五点task/state/RNG公共前缀pairing mismatch=0。旧合同中的108
  是6-worker roots的分片数，本轮9-worker动态队列正确值为126，已更正。
- step1500/2000=`638/658`；1500→2000四suite全净增，gained/lost=`77/57`、tasks升/降/平=
  `17/5/2`。配置已选择统一step2000，不按task混点；meta formal仍blocked。
- 00:01 CST live比较后预选空闲`gpu01:0,1,2|4,5,7`做六rank 3+3 NUMA profile；GPU3的`nlge`
  VLLM不触碰。gpu02物理6/7均有他人进程而空闲卡不满足3+3，故不选。gpu01 available memory约
  516.5GB，`/data1` quota=`552,249,764/1,073,741,824 KiB`，profile预计新增低于2GiB。
- clean pushed`d96f0fb`已封存统一target/config；两个profile roots、fresh/resume/contiguous三条
  exact command及逐rank NUMA、科学metric、step3 writer byte-equality验收门已写入`task_plan.md`。
- launch前发现frozen safety worktree按预期没有相对`.venv`，故三条命令在任何CUDA初始化前被审计
  出不可执行；root/log仍不存在。已将launcher原位修正为共享项目venv的绝对`torchrun`路径，不改
  科学config、设备、输入、输出或随机数。

## 2026-08-08 Task-expert 2000完成、CPU分析与1500/2000 launch seal

- 22:39 CST，clean`81101fe`原root的6个workers全部自然完成：24/24 completion、6/6 summaries、
  24个step1500、24个step2000、0 error/OOM/nonfinite；六张gpu01卡自然释放。canonical checkout
  随后从临时frozen状态恢复并fast-forward/push到`codex/bci-continuation@b19409f`。
- 后台CPU watcher完成五点full-bank geometry及1000/1500/2000 feature-target dynamics，随后自然退出；
  临时watcher脚本已删除。norm/stable-rank/task cosine从1000后平台，1500→2000 update近零；causal
  B proxy由`.38820`微降到`.38685/.38678`，没有晚期可预测性收益。
- `gpu-preflight` live比较选择空闲且host memory更充足的gpu02:0--5；GPU6他人进程、空闲GPU7和
  gpu01:3他人进程均不使用。独立clean pushed`1362d15` worktree已确认两个formal roots/log不存在，
  `/data1` quota充足；launch合同已登记，下一步并发评1500/2000各1200 rows。

## 2026-08-08 Expert-Manifold cached-rollout schema根修

- profile前只读追踪完整`online generation → release Writer → cached rollout`路径，定位统一adapter
  wrapper漏传`evidence_schema`；原状态会在LoRA cache成功生成后、第一条scientific row前报
  unexpected-keyword `TypeError`。
- 隔离分支已为old/Expert-Manifold adapter共同保留schema合同，并让Expert-Manifold mismatch
  fail-close；新增正确/错误schema regression。`PYTHONPATH=src`下聚焦62/62、全仓220/220以及
  changed-file `py_compile`通过。
- 未启动额外GPU工作，未改变正在运行的expert2000 root、Writer数值、实验配置或artifact。该修复
  并入后仍按原顺序等待expert完成，再做完整online-generation/cached-rollout A40 profile smoke。

## 2026-08-08 causal-prefix Expert-Manifold实现与expert2000续训启动

- 对sealed train24×50 cache和full24 step1000 experts完成可复现CPU dynamics审计，确认静态phase-DC
  能量主导但ordered temporal residual稳定且能迁移到expert B方向；3/5-shot proxy无足够增益，保持
  one-shot。分析入口为`scripts/analyze_expert_manifold_feature_dynamics.py`。
- 初版phase-centered value仍存在“忽略learned phase key后退化为unordered frame set”的精确结构
  风险。CPU反例确认后，在隔离分支把唯一Writer value改成sqrt-normalized causal-prefix integral；
  full projected video仍用于key/routing。增加严格paired且不读frame的no-video counterfactual；没有
  第二套Writer、language-only bypass或scalar gate。
- profile前复核发现meta入口虽实际调用GPU-local NUMA binder，却允许绑定失败且run contract不记录
  逐rank topology；已改为fail-fast，并记录local/physical GPU、NUMA node和CPU affinity。该修复不改
  模型、task batch、DDP平均、优化器或feature/expert输入。profile step wall和累计peak allocated/
  reserved显存均跨全部rank取`MAX`，防止rank0局部低估。architecture gate无hard violation，聚焦
  28/28和全仓220/220 CPU测试通过。
- evaluator原先虽允许`smoke`读取profile run，却无条件用formal checkpoint集合验收cursor，合法
  macro1/3必然失败。现已按training mode分别绑定profile/formal集合；formal仍要求sealed且选定统一
  expert step。meta profile后可先做online generation显存smoke，再决定formal topology。
- 2026-08-08 17:38 CST从clean`81101fe`沿原正式root启动24 experts统一exact-resume1000→2000：
  六个独立tmux workers固定`gpu01:0,1,2,4,5,7`及对应NUMA，每worker续原4 tasks，显式
  `NCCL_P2P_DISABLE=1`。GPU3他人进程和GPU6均未使用；1500 partial checkpoints不作科研结论。
- 下一步等待24/24 completion，复算1500/2000 full-bank geometry，并按已验证r3拓扑做两个
  development-train 1200-row official closed loop；只选择一个统一expert target后才profile meta-Writer。

## 2026-08-08 Expert-Manifold formal feature cache完成

- clean pushed`222d3ac`的6个独立workers在`gpu02:0,1,2,3,4,7`自然完成train24×50
  action-hidden feature extraction；GPU6上他人进程未被触碰，全部会话已自然退出。root=
  `runs/outputs/pi05_expert_manifold_feature_cache_train24x50_r6_222d3ac_20260808`。
- 24 records、6 summaries和24个`[50,16,3072]` BF16 feature tensors齐全，task/demo ordinal
  分别完整覆盖`0--23`/`0--49`，cache约113MiB。peak allocated/reserved=
  `10,504,039,936/19,232,980,992` bytes，teacher action/state/reward/terminal reads合计0。
- canonical CPU seal自然完成并生成`cache_manifest.json`。当前不再阻塞于feature cache；
  下一步按原训练合同从clean`81101fe`沿同一root统一resume全24 experts到2000，再评
  1500/2000并选唯一meta target。

## 2026-08-08 Task-expert闭环完成并触发统一2000 continuation

- clean pushed`1362d15`的三组有效r3 evaluation自然完成：step250/500/1000=
  `432/557/624` of 1200，全部108 shards attempt1、6 workers exit0、0 failure/retry且配对通过。
  四suite依次为Spatial=`123/147/170`、Object=`125/191/208`、Goal=`142/163/164`、
  Long=`42/56/82`；500→1000为143 gains/76 losses、18/4/2 tasks升/降/平、breadth=`23→24`。
- 初始总36 workers在0 rows前触及主机内存边界，第二次总24 workers因每卡4 replicas的37.7GB
  静态占用在首个inference OOM；均停止且标记ABORTED。最终每卡3 replicas、总18 workers在
  `gpu02:0,1|2,3|4,7`稳定运行，每卡约30.3GB。没有覆盖或resume无效root。
- full result审计通过schema v2、1200 unique keys、task/state/RNG公共前缀、24×50 coverage、
  108 shard manifests与worker exit。三点union/intersection=`731/332`，per-task oracle=`636`，
  只比step1000多12；正式决定从`81101fe`沿原root把全部24 experts统一resume到2000并评1500/2000。
- 同时在已释放的`gpu02:4`完成phase16×3072 feature cache profile：task0×4 videos wall=`4.372s`，
  peak reserved=`19.23GB`、0 forbidden reads/OOM/nonfinite；profile root=
  `runs/outputs/pi05_expert_manifold_feature_profile_task00_1362d15_20260808`，formal cache已seal。

## 2026-08-08 Expert-Manifold持续执行恢复

- owner在完成当前状况与方法讨论后明确恢复持续自主执行，并设定长期Goal：同一single checkpoint
  strict paired correct严格超过`150/400`，同时要求correct真实依赖有序teacher video、
  same-task跨video鲁棒、较高task breadth和低checkpoint漂移；只有实质性阻塞才回报。
- 唯一工作checkout已从滞后的local `main@15014eb`切换到clean、已推送且与upstream一致的
  `codex/bci-continuation@3592b41`；没有创建额外branch/worktree，也没有删除迁移后的runtime、
  cache或历史artifact。
- 当前证据顺序不变：先做full24 expert geometry与development-train expert closed-loop统一比较
  step250/500/1000；只在step1000仍有明确closed-loop上升证据时，从frozen`81101fe`沿原root
  统一resume2000；随后再做feature cache与meta-Writer的A40 profile/formal和strict五臂。

## 2026-08-08 Task-expert full24 geometry完成

- CPU canonical analysis已读取24 tasks×step250/500/1000共72个formal adapters并自然完成；artifact=
  `runs/outputs/pi05_task_expert_bank_geometry_full24_steps0250_0500_1000_05d4868_20260808/analysis.json`。
  没有GPU、rollout、环境交互或held action/video reads。
- effective-LoRA norm中位=`2.792/3.652/4.170`，stable rank中位=
  `1.126/1.129/1.129`，top singular energy=`.903/.907/.909`；跨task effective cosine中位=
  `.108/.095/.100`。16 coordinates均active且top4 energy约`.26`，但q/v B-column仍高度同向。
- geometry确认bank具有task-specific policy directions，却不能选择统一checkpoint或支持resume2000；
  下一动作仍是step250/500/1000 development-train official closed-loop严格比较。

## 2026-08-08 Expert-Manifold实现整合与task-expert bank封存

- clean`81101fe`的6个independent A40 workers已自然完成24/24 task experts统一step1000；root=
  `runs/outputs/pi05_task_expert_bank_formal_step1000_r6_81101fe_20260807`，包含6份worker summary、
  24份completion、72个step250/500/1000 checkpoints，约562MiB。GPU已释放。
- 三个统一checkpoint最后50步24-task等权mean action loss=
  `.115355/.107207/.105372`，只记作拟合趋势，不用它选择closed-loop checkpoint。
- 新增完整train24 task-expert bank evaluator和统一step LoRA几何分析；evaluation只开放
  `development_train`且按实际task安装对应expert，不引入task-ID部署输入或held expert。
- 新增action-hidden frozen PI0.5 video innovation cache：exact language query每帧的2048维joint
  task-span hidden与1024维Action-Expert suffix hidden，拼成3072维后统一减matched no-image
  baseline；每条video保留phase16，cache不含action/state/reward/terminal。
- canonical 168-chunk rank16 decoder与六rank task-complete DDP meta trainer已闭合；direct output不经
  窄factor head，direction/scale分离，zero video恒为identity。checkpoint保留Writer/optimizer/
  scheduler/每rank RNG和macro cursor，BCI deferred-NCCL及P2P-disable合同沿用。
- one-shot evaluator为每个episode只选一条action-hidden video，50 states无放回覆盖50 demos；五臂
  共享paired schedule，online生成LoRA cache后释放Writer并复用source policy rollout。held split只读
  video，validation/test expert与action reads保持0。
- 隔离worktree的14个retained commits已fast-forward并入`codex/bci-continuation`，不再保留第二个
  活动实现分支。该实现尚无meta GPU或性能结论；feature cache与meta formal仍blocked。
- owner要求当前session完成仓库交接后停止执行。新session先讨论，再决定full24 geometry/
  development-train expert rollout、统一resume2000或meta-Writer profile；当前不自动launch。

## 2026-08-07 Video Expert-Manifold task-expert builder与profile seal

- 建立hashless task-expert formal config、task-local deterministic sampler、adapter+optimizer+
  scheduler+RNG checkpoint、all-task stage resume与单GPU多task串行worker；formal固定6 workers×4 tasks，
  不建DDP/NCCL，不共享task参数或optimizer。
- profile先后暴露并修复三步诊断误缩formal scheduler horizon、trainer反序列化把CPU RNG搬到CUDA
  两个真实合同问题；失败run均未越过对应边界，也未作为科研证据。
- clean`174d292`最终在`gpu01:0`以B16通过fresh0→1、same-root resume1→3和独立contiguous0→3。
  三步finite，loss=`.221725/.283785/.259915`，峰值allocated/reserved=
  `15,082,000,384/21,313,355,776` bytes，0 OOM/nonfinite；resume和contiguous的科学metrics、
  step3 adapter逐字节一致。config已seal，下一步正式训练24 experts到统一step1000。

## 2026-08-07 K4 Phase-Aligned全部封存并切换Expert-Manifold

- clean`2356d33`的K4 Phase-Aligned formal root完成200 finite macros、96,000 queries、
  19,200 K4 videos和8 checkpoints；wall=`16,228.904s`，peak reserved=
  `39,187,382,272` bytes，source trainable=0且held action reads=0。
- strict correct50/100/150/200=`88/108/80/99`，breadth>=5=`4/4/3/4`，相邻
  gained/lost=`40/20,27/55,47/28`，union/intersection=`157/36`。macro100为winner，不resume400。
- macro100五臂=`108/115/94/101/121`，全部paired mismatch=0。correct相对wrong
  gained/lost=`28/14,p=.04356`；correct相对same/shuffled/reversed分别为`13/20,23/16,20/33`。
- 8-task refs1 root完成Core→Procedure→Program→BA→fixed-action五臂分析。correct
  LoRA norm/stable-rank/top-energy中位=`91.12/1.00021/.99979`；wrong的BA/action中位
  relative-L2=`.330/.0765`。视频路径material，但LoRA近单方向且task更新仍抵消。
- 新authority为`docs/action_forecast_writer_video_expert_manifold_design.md`。已完成真实
  38-target topological round-trip原型：168个`[16,512]`chunks、1,287,168 valid values、
  两个axial blocks约7.70M参数、zero-output identity。下一工程步骤为task-expert builder。

## 2026-08-07 Grounded-Video负裁决并切换K4 Phase-Aligned v6

- Grounded formal四点strict correct=`76/88/77/82`、breadth>=5=`3/4/3/3`，相邻
  gained/lost=`27/15,17/28,25/20`，union/intersection=`125/40`；winner macro100五臂=
  `88/87/82/86/86`，direct pairing均0 mismatch。
- 8-task refs1显示wrong到Reader/BA/action中位relative-L2约`.293/.433/.099`，shuffled/reversed
  到BA约`.426/.435`，LoRA stable-rank/top-energy=`1.463/.773`。视频、时序和高层route都能改变
  一套真实LoRA，但correct没有行为优势；完整expert隔离与约`.5`局部retention也未解决漂移。
- 当前authority切到`docs/action_forecast_writer_k4_phase_aligned_v6_design.md`。canonical实现已恢复
  v6 trainable high-level encoder/Core/Procedure/compiler，并加入K4逐video phase16对齐与公共程序组合；
  旧`fewshot_m2p.py`已退休，fresh config/checkpoint family建立。
- 全仓CPU回归`190 passed`，Writer参数`10,775,296`，step0 identity与K4 set permutation成立。
  clean`e1d0b62`随后在gpu01六张空闲A40完成fresh0→1及same-root exact-resume1→3；三步
  `86.20/87.52/87.47s`，0 clip/OOM/nonfinite，peak reserved`47,016,050,688` bytes，step3五个
  owner全可达。K4/B20/B2/full24/phase16保持，profile权重弃用，formal现已seal。

## 2026-08-07 Grounded-Video formal0→200与hashless evaluator完成

- clean`a758bba`在`gpu01:0,1,2|4,5,7`、world6、3+3 NUMA、显式
  `NCCL_P2P_DISABLE=1`下自然完成identity-fresh0→200；root为
  `runs/outputs/pi05_as_writer_k4_grounded_video_expert_trace_m2p_formal_fresh0_200_r6_a758bba_20260807`。
  共200 finite macros、96,000 action queries、19,200 K4 video conditions、8个every25完整
  checkpoints，0 clip/OOM/nonfinite，wall=`8828.911s`，peak allocated/reserved=
  `36,708,964,864/42,727,374,848` bytes；source trainable=0且held action/video value reads=0。
- 八个grounded-route experts全程获得其冻结owner tasks的梯度。按expert真实task数消除24-task
  zero-padding后，Reader/axis的25步窗口retention中位从首窗`.580/.518`到末窗`.453/.486`，
  明显保持在约半能量而非shared 1/24抵消区；global padded retention仍从`.0679`降到`.0455`。
  这只说明parameter coexistence合同工作，不能替代closed-loop选点。
- evaluator canonical launch schema已切到`ember_pi05_target_eval_launch_v2`：不再计算或比较source
  checkpoint、authority、normalization、tokenizer、task asset、raw shard、aggregate或completion的
  内容hash；以path/schema/size、真实解析/加载、direct pairing字段和显式UUID run reference替代。
  policy-noise RNG及deterministic job ID仍保持原科学配对算法，不属于artifact完整性校验。
- evaluator聚焦`55 passed`、全部改动模块py_compile、diff check通过；真实validation Writer
  `prepare` vertical path生成hashless v2 contract和8-task queue，递归检查无SHA/MD5字段。下一步
  clean commit/push后只按预注册四点correct400合同启动rollout。

## 2026-08-07 Grounded-Video Route input-only gate完成

- clean`563089a`在live空闲`gpu01:0,1,2|4,5,7`上以六个独立进程完成train24×50 frozen
  multimodal task-token video address提取；root为
  `runs/outputs/pi05_grounded_video_expert_route_train24x50_563089a_20260807`。只读action-hidden
  train videos，route生成器记录teacher action/state/reward/terminal和validation/test video reads=0。
- 初始top2 input-only预门的随机K4 primary/exact/overlap=`1.0/.984833/.992417`，usage无空置；
  但task35的secondary owner在batch4/singleton间翻转，严格门为`23/24`。没有用rollout调center
  或放宽门，而是依据全部6,000个随机K4与24个batch/singleton都完全稳定的primary，收敛为top1
  one-hot完整expert route。
- 最终artifact `configs/pi05_grounded_video_expert_route_v1.json`通过：route stability=`1.0`、
  batch4/singleton=`24/24`、8-expert usage=`2/6/7/3/1/1/2/2`。fresh config/schema/checkpoint与唯一
  canonical runtime已同步，聚焦`30 passed`及py_compile通过。
- clean`0be3627`随后在`gpu01:0,1,2|4,5,7`完成fresh0→1和same-root exact-resume1→3 profile。
  三步loss=`.150377/.152826/.148513`、step time=`42.63/41.72/41.20s`，0 clip/OOM/nonfinite，
  peak allocated/reserved=`36,709,136,896/45,237,665,792` bytes；step1八Reader可达、step2起
  16 Reader/axis blocks全可达，train24真实route逐expert与artifact一致。累计1,440 queries/288
  videos，source trainable=0；profile权重弃用，formal config已seal。

## 2026-08-07 Sparse正式裁决并开启Grounded-Video Route

- clean`3820f27`的routefix formal已自然完成0→200：200 finite macros、96,000 queries、
  19,200 K4 videos、8 checkpoints、0 clip/OOM/nonfinite，peak reserved42.86GB。四点=
  `74/74/78/75`、breadth=`6/5/5/5`，winner macro150=78。
- winner same/wrong/shuffled/reversed四臂全部完成并释放GPU；五臂=
  `78/85/90/83/92`，correct最低。随后clean`507ae6e`用六张空闲gpu01 A40按production cache
  batch完成8-task trace→expert Reader→program→BA→fixed-action分析，0 target-action reads；视频
  路径、LoRA gain/rank均成立，language-only route被定位为最早失败接口。
- 当前唯一authority切换为
  `docs/action_forecast_writer_grounded_video_expert_route_design.md`。下一步原位增加冻结multimodal
  task-token video address并流式生成train24×50 route artifact；先过input-only stability/usage门，
  再fresh schema/config、A40 profile与identity formal。旧sparse checkpoint不resume/warm-start。

## 2026-08-07 Sparse Semantic-Expert route稳定性根修

- 新canonical已接通冻结task anchor、fixed top2 router、8套完整独立Trace Reader+axis M2P、
  memory-level等权组合和single-LoRA decode；fresh config/launch/checkpoint family与live evaluator
  同步切换，旧checkpoint严格不兼容。
- 首次formal在macro28主动停止。训练gradient ownership显示task9同时进入experts1/2，而旧
  route artifact声明2/7；根因是route生成以24-language BF16 batch取anchor，runtime逐task
  forward，secondary近邻发生batch-shape数值翻转。该root与旧profile都不得resume或作为formal
  证据。
- task anchor现逐exact language独立forward，generator同时比较co-batch与singleton路径；最大
  anchor差`1.49e-8`且top2完全一致。新primary/top2 usage=`5/7/6/1/1/2/1/1`和
  `7/11/6/5/4/4/3/8`。真实参数仍为`487,415,808`、272 tensors，video value路径未改。
- 聚焦route/dense equivalence/zero identity/unselected-gradient/config/checkpoint/model合同通过；
  clean`bf1aae6`六卡longest105 profile随后完成fresh0→1、exact-resume1→3。三步约48.1--48.7s，
  loss/gradient finite、0 clip/OOM，step2起全部16 expert-local blocks可达，peak reserved45.59GB；
  累计1,440 queries/288 videos且source trainable=0。由于它绑定旧route buffer，profile现已作废；
  clean`bbe5cf2`随后以新root重做fresh0→1/exact-resume1→3：三步`42.299/43.074/42.275s`，
  0 clip/OOM/nonfinite，peak reserved`45,592,084,480` bytes，step2起16 blocks全部可达，step1
  train24 route逐expert与authority完全一致。新profile已seal；下一步另起identity-fresh formal。

## 2026-08-07 Evidence-Factorized完整裁决并开启Sparse Semantic Experts

- macro200四个追加control和六卡8-task内部probe已自然完成，五臂=
  `84/85/66/83/78`；internal root为
  `runs/outputs/pi05_as_writer_k4_evidence_factorized_layer_trace_m2p_macro0200_internal_refs1_r6_8c8b502_20260807`。
  6份ownership、8 tasks、11 conditions/counterfactuals、K4/zero/set、trace branches、BA、
  fixed action、LoRA geometry及训练gradient完整，held action reads=0，GPU自然释放。
- 视频task identity、direction/physical/evidence与LoRA leverage均闭合；最后50步shared Reader/
  axis仍近1/24抵消。Evidence-Factorized正式负裁决，不续训、不warm-start。
- 新authority为
  `docs/action_forecast_writer_sparse_semantic_expert_trace_design.md`：固定train24 language top2
  route，两个完整独立Reader+axis experts等权生成一套video-owned LoRA；下一步生成route
  authority、原位实现、聚焦验证和A40 profile。

## 2026-08-06 Evidence-Factorized Trace formal0→200完成

- clean/pushed launch commit`7e3559f`在`gpu01:0,1,2|4,5,7`自然完成identity-fresh
  formal0→200；root为
  `runs/outputs/pi05_as_writer_k4_evidence_factorized_layer_trace_m2p_formal_fresh0_200_r6_692ab5e_20260806`。
- 共200 finite macros、96,000 queries、19,200 K4 action-hidden videos、8 checkpoints、0 clip，
  wall`7272.774s`、peak reserved20.30GB；source trainable=0、validation/test action reads=0，
  六张A40已自然释放。下一步按预注册只评50/100/150/200 strict correct400。
- 四点strict correct400已完成为`74/59/65/84`、breadth=`6/6/5/5`；macro200固定single
  winner。下一步只做macro200其余四个video arms与全部内部分析，不续训练。

最后更新：2026-08-06。

阅读规则：本文按时间顺序保留真实执行状态。早期段落中的“当前”“下一步”、
GPU范围和训练步长是当时快照；活动状态只取
`docs/a100_to_bci_migration_handoff.md`、`docs/active_session_handoff.md`和本文顶部
最新段落，
不能用旧快照覆盖后续owner决定。

## 2026-08-06 Evidence-Factorized Trace设计开启

- Energy-Preserving的四点、五臂与全部内部分析已封存并释放GPU；method行为负裁决，但与
  unit-direction前版共同明确了direction/content和physical reliability不能用一次破坏性
  normalization绑定。
- 新authority为`docs/action_forecast_writer_evidence_factorized_trace_design.md`：保留K4视频、
  20-group DCT16和axis M2P，从raw token同时形成normalized direction、physical value及
  energy/K4-consensus key evidence，用shared attention和bias-free vector fusion组合。
- canonical Reader、fresh schemas/config/checkpoint family已原位替换完成；Writer参数精确
  `60,926,976`，全仓`192 passed`、compileall、real config load和diff check闭合。
- live`gpu01:0,1,2|4,5,7`六卡fresh0→1、exact-resume1→3 profile已通过并自然释放GPU：
  三步loss=`.150377/.152820/.148508`，0 clip/OOM/nonfinite，step2起全部新Reader与axis可达，
  peak reserved20.47GB。profile权重弃用，formal已seal；下一步identity-fresh0→200。
  不加载旧Writer、不用subagent、不做hash检查或scalar/band sweep。

## 2026-08-06 K4 Layer-Trace五臂/内部裁决完成，开启Energy-Preserving Trace

- macro100四个追加control全部自然完成并释放GPU；五臂为
  `99/92/57/94/105`。correct相对wrong明显更好，证明视频任务语义有效；顺序
  controls不降，说明时序差异未对齐任务程序。
- 六卡8-task内部probe已完成，持久root为
  `runs/outputs/pi05_as_writer_k4_layer_trace_m2p_macro0100_internal_refs1_r6_cd78d47_20260806`；
  6份ownership、8 tasks、5 conditions/refs、trace→LoRA→action与频谱证据完整，无held action read。
- 最早结构故障定位到每`group × frequency`单独L2 normalize把原始仅`.359%`
  的high8能量相对放大约140倍。已封存
  `docs/action_forecast_writer_energy_preserving_layer_trace_design.md`，旧layer-trace不续训、不
  warm-start。
- 下一步原位实现每视频全局能量匹配、fresh identity/config/checkpoint family；聚焦
  CPU合同与全仓回归后，当时计划live选择单节点可用空闲A40重新profile与formal；后续实际使用了当时空闲的
  六张，不代表owner设有6卡上限。
- clean`22234c4`已完成上述原位替换并push branch/main；旧config退休，新schema/family拒载
  旧checkpoint。全仓`191 passed`，compileall、real config load和diff check通过，formal仍blocked。
- 新profile的root、scale、storage、fresh/resume分段和GPU边界已预注册在`task_plan.md`顶部；
  `gpu01:0,1,2|4,5,7`已严格fresh0→1再exact-resume1→3完成。三步loss稳定、0 clip/
  OOM/nonfinite，step2起reader/axis均可达，peak reserved20.38GB，GPU自然释放。
- config seal=`3b7eb4a`已回写profile证据并push branch/main，profile权重弃用。launch
  commit`d833961`随后从identity完成formal0→200：200 finite macros、96,000 queries、19,200
  K4 videos、8 checkpoints、wall`7373.955s`、0 clip，peak reserved20.48GB，GPU自然释放。
- 训练期四个50步gradient retention为`.12497/.08564/.08050/.05079`；频谱修复明显改善
  前150步共存，末段仍回落。两波四点strict correct400随后自然完成并释放GPU：
  `67/83/74/85`、breadth=`5/6/7/7`，macro200固定single winner。五臂/内部分析已预注册。
- macro200四个追加control与六卡8-task内部probe随后全部自然完成并释放GPU。五臂=
  `85/85/80/74/87`，correct-wrong不显著；trace/Reader/BA的video差异和effective groups均被
  raw幅度显著压缩，而LoRA gain/rank充足。当前方法负裁决，下一步转向direction/energy/
  K4-consistency因子化读取，不续训、不先开experts。

## 2026-08-06 K4 Policy-Layer Trace四点strict correct400完成

- 两波macro50/100与150/200均自然完成；每root 400 rows、42 shards、9 workers exit0，GPU
  自然释放。完整correct=`69/99/88/94`、breadth=`5/6/6/6`，相邻gained/lost=
  `42/12,28/39,28/22`，union/intersection=`145/37`，全部paired K4/state/RNG字段闭合。
- macro100固定为single winner=99；150/200回落，layer-aligned新方法只在macro100比旧K4高5，
  没有超过旧K4 winner108或v6-fast143。当前不续训、不warm-start、不按loss另挑checkpoint。
- `task_plan.md`已预注册macro100其余四个video-control arms与hashless内部分析。当时下一步是live选择
  单节点可用空闲A40分两波运行same-task-other/wrong/shuffled/reversed；后续实际使用六张不代表owner cap。再按最早失败接口裁决
  sparse condition-specific parameter sharing，而不是直接堆参数。

## 2026-08-06 K4 Policy-Layer Trace M2P设计封存

- 复核K4内部机制后，没有把视频旁路掉，也没有直接堆多experts；定位到旧final-layer固定
  随机128维压缩与public policy topology不对齐这一更早接口。
- 新authority为`docs/action_forecast_writer_k4_layer_trace_m2p_design.md`：冻结PI05的20组
  action-expert layer traces、K4×16 temporal tokens、`20×68×1024`双轴M2P直接reshape完整
  rank16 LoRA。clean`a2c6d94`已原位实现并退休旧K4 config/family；全仓BCI assets下
  `190 passed`、compileall和config/schema real load通过。尚未启动GPU，下一步live profile。
- `89f5384`首个三步diagnostic没有按step1分段，且在step2暴露axis FFN pre-LN幅度爆炸；root
  禁止resume。clean`ed4f46e`已根修为raw-value FFN，全仓`191 passed`。下一步从新root严格
  fresh0→1再exact-resume1→3。
- clean`44e248b`新root严格完成fresh0→1→exact-resume1→3：三步loss稳定、0 clip/OOM/
  nonfinite，step2起reader/axis均finite update，peak reserved20.38GB、每步约34.6秒。profile
  权重弃用，GPU自然释放；formal从fresh identity启动条件成立。
- clean/pushed`d3f568d`已seal正式config和profile evidence；fresh0→200的唯一root、scale、
  storage预算、六卡3+3 NUMA边界、exact command及50/100/150/200严格裁决已写入
  `task_plan.md`。启动不加载任何profile或历史Writer权重。
- launch commit`1b868ed`的fresh formal0→200自然完成：200 finite macros、96,000 action
  queries、19,200 videos、8个checkpoints、wall`7350.114s`、peak reserved20.48GB、0 clip/
  OOM/nonfinite，GPU自然释放。最后50步task-gradient retention中位`.04573`，早期层对齐优势
  到晚期已明显衰减；下一步按预注册合同完成四点strict correct400，不用loss挑点。

## 2026-08-06 K4四点strict correct400与内部分析完成

- macro150/200各自完成400 rows、42 shards、9 worker exit0；与50/100共同曲线为
  `70/94/99/108`、breadth=`6/6/6/7`，adjacent gained/lost=`42/18,30/25,25/16`，
  union/intersection=`150/42`。全部K4 sets/state/env/policy RNG严格配对，GPU自然释放。
- 六张空闲`gpu01`卡运行8-task refs1内部probe，无NCCL、无rollout、无target/held action read；
  durable root为
  `runs/outputs/pi05_as_writer_k4_invariant_m2p_macro0200_internal_refs1_r6_4951d4e_20260806`。
  K4置换、zero-video identity、alternate set、LOO、wrong/shuffle/reverse、LoRA geometry与
  fixed-action路径均封存。
- 结果确认video-common representation和约28 norm的高增益LoRA成立；共享Writer最后50步
  task-gradient retention仍仅`.04326`且约一半pair为负。K4行为未过门并正式停止同一配置
  续训；下一设计只从condition-specific shared-parameter coexistence继续。

## 2026-08-06 K4 fresh formal与首批strict correct400完成

- clean`500294c`在`gpu01:0,1,2|4,5,7`从functional identity完成macro0→200：200 rows、
  96,000 action queries、19,200 K4 teacher videos、wall`6879.816s`、peak reserved
  `19,690,160,128` bytes，0 clip/OOM/nonfinite，source trainable=0，validation/test action
  reads=0；25--200共8个checkpoints完整，tmux自然结束并释放GPU。
- live双节点检查后用`gpu01:0,1,2`与`3,4,5`并行完成macro50/100 strict correct400；
  每root 400 rows、42 shards、9 workers和400个K4 caches完整。结果=`70/94`、breadth=`6/6`，
  50→100 gained/lost=`42/18`、union/intersection=`112/52`。
- 两个launcher只在最终CPU aggregate触发旧K1字段`teacher_demo_index` KeyError；已在唯一
  `pi05_eval_results` owner改为汇总K4 `teacher_demo_indices`与video-set counts，聚焦复现和
  全仓`189 passed`通过，并直接从既有shards生成results，未重复GPU rollout。
- 下一步在clean提交上评macro150/200；四点完成前不按loss或当前94分改架构。

## 2026-08-06 K4 Invariant-Program M2P设计与CPU实现封存

- owner恢复持续推进并允许few-shot，同时明确EMBER不得绕开video。封存新authority
  `docs/action_forecast_writer_fewshot_invariant_m2p_design.md`：每个task condition联合
  K4 action-hidden videos，以video-value-only invariant slots和policy-wide M2P生成一套LoRA。
- canonical实现已原位替换Condition-Kernel：新增`fewshot_m2p.py`，更新frozen temporal
  descriptor、CompleteLoRAWriter、full24 end-to-end AS、K4无重合schedule、fresh-only
  checkpoint family与live/cached rollout。旧kernel、online validation和method-specific
  cold-start analysis executable paths删除，历史仍由Git/artifact保留。
- Writer-specific authority/checkpoint/evaluation不再生成或比较文件内容hash；只保留路径、
  schema、size、shape、real load和真实runtime证据。新架构显式拒绝任何历史Writer
  warm-start，profile与formal都必须从functional identity fresh开始。
- 聚焦合同与完整项目测试现为`188 passed in 19.68s`；compileall和`git diff --check`通过。
  真实A40容量、三步梯度开放、NCCL、finite与exact-resume尚待live profile，当前没有据此
  宣称闭环性能改善。
- 首个clean profile用`total_steps=3`压缩了scheduler：显存与resume虽通过，但step1直接使用
  peak LR、step3触发clip，不代表formal前3步，因此不予seal。根修把profile总轴固定为正式
  200步并只early-stop到3；新clean`8807ae0` root完成fresh0→1与exact-resume1→3，三步
  `34.055/33.955/33.831s`，peak allocated/reserved=`17,142,612,480/19,690,160,128` bytes，
  formal LR=`1.154e-5/2.308e-5/3.462e-5`，0 OOM/clip/nonfinite，step2起四block全可达。
- sealed profile累计1,440 action queries、288 teacher videos，source trainable=0，六rank
  optimizer/scheduler/RNG/sampler/checkpoint连续；GPU自然释放。profile权重弃用，formal config
  已开放独立functional-identity fresh0→200。

## 2026-08-06 Condition-Kernel formal、四点rollout与内部分析全部完成

- fresh formal AS0→200自然完成：200 metrics、96,000 queries、4,800 videos、四个完整
  checkpoints，wall=`3951.928s`、peak reserved=`19,344,130,048` bytes，0 OOM/clip和0
  validation/test action reads。50/100/150/200 correct=`46/46/45/49`、breadth全3，
  reward gate失败。
- 四点各400 rows、42 shards、9 worker exit0、每task 50个无放回videos；paired
  gained/lost=`5/5,4/5,6/2`、union/intersection=`55/40`，共同controls和policy-noise prefix
  0 mismatch。成功几乎全部集中Goal-6，不能把低换手写成task drift已解。
- clean`2972f8f`六卡内部分析完成96/96 rows、6/6 payload，wall=`273.968s`、peak reserved=
  `19,277,021,184` bytes、0 target-action/validation/test reads。feature→Program→BA视频/
  order差异保持，但LoRA norm仅`.176→.178`，fixed action效应只有`.19--.24%`。
- 结合200步rank24 kernel、predicted/observed equality和macro50后严格freeze，正式定位最早
  失败为fresh zero-B decoder未在固定50步内建立足够增益、policy-effective的Program→LoRA
  basis；condition credit隔离成立但AS substrate严重失败。精确汇总已写入internal root的
  `experiment_analysis.json`，design第11节及handoff/brief/findings同步。
- 预注册direct reward禁止；所有训练、rollout和analysis进程自然结束，GPU释放。当时的
  讨论暂停已由owner解除；长期single-checkpoint `>150/400`仍未完成。

## 2026-08-05 Condition-Kernel实现与profile封存

- 完成train24×50、validation8×50 action-hidden address audit：50组schedule Gram全rank24，
  最坏condition7.547；same-task video与reversed/shuffled feature均非零，所有action/reward
  reads为0。固定authority SHA=`7a49226e...0f86`，不做feature/seed sweep。
- canonical AS路径原位替换为固定condition feature、83.9M Program Value Memory和fresh
  FactorHeads；旧Program-Credit一次性analysis runtime及v6 AS condition path已退休。23项
  condition/model/training/checkpoint聚焦合同通过，compileall与diff check闭合。
- `gpu01:0,1,2|4,5,6`完成fresh0→1与exact-resume1→3：三步约
  `20.71/19.84/19.45s`，峰值reserved19.34GB，0 OOM/clip。step2/3 Program cotangent和
  value update有限、cap未触发，六rank checkpoint/scheduler/RNG连续；进程自然结束、GPU
  已释放。profile权重弃用，下一步是clean pushed commit后的fresh AS0→200。

## 2026-08-05 Program-Credit内部分析完成并封存Condition-Kernel Memory设计

- six-rank只读内部分析root
  `runs/outputs/pi05_antithetic_program_credit_internal_as125_cycle1_r6_129cab6_20260805`
  已完整结束：48 rows、24 tasks、2 checkpoints、6/6 payload，wall=`272.876s`、peak
  reserved=`19,304,284,160` bytes，0 target-action/validation/test reads，GPU全部释放。
- exact task cotangent由近正交`.000107/0`、retention`.041874`经过共享Writer更新变成
  task-mean program delta`.5801/.6128`、retention`.55537`且无负pair；same-task更新
  program/BA task-mean energy fraction=`.82990/.91623`。held gained/lost LoRA变化不可区分，
  视频/顺序传递与LoRA spectrum基本不变。Program-Credit正式负裁决，不续cycle2/4/8。
- 新authority
  `docs/action_forecast_writer_factorized_condition_kernel_memory_design.md`已封存：固定
  task×video RFF condition feature、P1024完整Program Value Memory、24×24显式kernel
  correction、fresh decoder bootstrap0→50、memory-only AS50→200与同一memory direct reward。
  尚未修改canonical实现或启动GPU；下一步是退休旧analysis runtime并原位实现新Writer。

## 2026-08-05 Program-Credit environment CRN根修

- AS125-fresh formal cycle0→1已完成96 rollout、48 valid CRN pairs、54 successes、6个
  binary-discordant pairs、一次finite update和完整cycle1 checkpoint；wall`418.692s`、
  peak reserved`19,308,478,464` bytes，0错误，GPU已释放。
- 随后的strict correct400完整结束：cycle1=`106`、breadth5，相对AS125=`97/5`的严格
  gained/lost=`18/9`，union/intersection=`115/88`；三suite改善但Spatial task1→task3换手，
  净增9未过预注册净增10门，故禁止resume cycle2。
- canonical只读内部分析入口已实现并拆为authority/runtime/metrics三个owner；聚焦12项及带
  BCI assets全仓223项通过，architecture guard无hard/parallel family。下一步只在clean
  pushed代码上运行AS125→cycle1的train24 program/BA/action/credit分析，不启动新训练。

- clean`318b6f6`首次原六卡profile完成68/96 rollout后，在0 update/0 checkpoint处由pair
  初态hash合同终止；task38 env/policy seeds一致而首帧不同，失败root仅保留诊断用途。
- 单GPU真实复现确认LIBERO默认hard reset的environment-local placement history不由再次
  `env.seed`清空；同env同seed可得到不同XML和3--4cm物体位移。两个独立persistent env保持
  相同reset index/seed时，即使中间动作不同，连续三轮XML、47维state和双相机像素均逐字节
  相同。
- canonical runtime已改为每task两条lockstep lanes，plus固定lane0、minus固定lane1；rollout
  与program-credit schema升为v2并绑定lane，仍使用official random reset且从不调用
  `set_init_state`。聚焦20项及全仓221项合同通过，compileall/diff check通过；待clean
  commit/push后用全新root原规模重放。
- clean`f3f6b15`全新root随后完成六卡cycle0→1及exact-resume1→2：累计192 rollout、48
  task-credit rows、96 CRN pairs、2次finite full24 update和完整cycle1/2 checkpoints；两轮
  各54 successes，四上游block可达、冻结梯度0，wall约431秒/轮、峰值19.33GB，0错误，GPU
  自然释放。formal config已seal，profile权重禁止使用；下一步为AS125-fresh formal cycle1。

## 2026-08-05 Antithetic Program-Credit实现与CPU seal

- canonical Writer已原位恢复v6，并把确定性函数显式拆为`encode_program(320×256)`和
  `decode_program(complete LoRA)`；Policy-Lane module/config/checkpoint family已删除，旧
  Flow-Credit、PPO/SPO、executed replay与progress diagnostic不再有活动实现。
- canonical RL runtime现只保留两对共享随机性的program扰动、binary-first pair credit、
  direct program cotangent与每cycle一次full24 update。semantic encoder、8个FactorHeads、
  source policy全部冻结；pair ledger绑定方向seed、±cursor、randomness、LoRA hash与outcome。
- 69项聚焦合同和项目正式activation下全仓220项均通过；py_compile和`git diff --check`
  通过。AS125 cold-start manifest定向确认是fresh identity训练125 full24 macros、60,000
  queries、3,000 one-video conditions的v6 family。profile config已seal，formal仍blocked；
  当前尚未启动GPU，下一步只做clean/pushed代码上的六卡cycle0→1/resume1→2 profile。

## 2026-08-05 Antithetic Program-Credit设计封存

- Policy-Lane内部分析后完成下一单变量设计：恢复v6 compiler的`320×256` policy program
  作为episode-level高层动作，K4组成两组严格同randomness的`+/-`扰动，以binary-first
  pair差直接估计program cotangent并反传条件网络。
- 唯一cold start为fresh AS125阶段边界；semantic encoder、完整FactorHeads、source policy
  与normalization冻结。旧action CFM ratio、PPO/SPO、Nmc4与two-epoch replay不再进入新
  方法，部署仍是一条video、一套确定性rank16 LoRA。
- authority已写入`docs/action_forecast_writer_antithetic_program_credit_design.md`并同步
  handoff/execution brief/task plan/AGENTS。尚未修改代码或启动GPU；下一步是canonical原位
  实现、聚焦合同和独立六卡A40 one-cycle profile。

## 2026-08-05 Policy-Lane四点strict rollout完成

- 50/100/150/200四个formal correct400 roots全部自然结束并释放GPU：每点400 rows、
  42 shards、一次launcher、9/9 worker exit0、400 LoRA cache、每task 50 unique无放回视频，
  0 error/retry。
- correct=`70/63/37/61`、breadth=`6/4/6/6`；相邻gained/lost=
  `17/24,14/40,40/16`，union/intersection=`117/14`。macro50是single winner但远低于
  PWAD80、v6-fast143和严格门，禁止resume400。
- clean`3869d20`的50/100/150/200六卡内部分析随后完成：96/96 cells、6/6 rank
  payload、wall`318.446s`、peak reserved`19,295,895,552` bytes，0信息墙读取，GPU自然释放。
- 分析证明约10个输出lanes、stable rank1.34--1.54与SFT量级跨layer专门化均真实成立，
  但same-task video hidden/BA能量仅约`.05%/.02%`，结构改善与闭环严重错位。下一步仅
  封存直接closed-loop Writer/LoRA credit的新design，不续Policy-Lane或加容量。

## 2026-08-05 Policy-Lane Coupled Hyperdecoder profile/resume封存

- 同一方法的clean/pushed`244b677`正式fresh0→200已自然完成：200 finite macros、96,000
  logical queries、4,800 one-video conditions、every25共8个完整checkpoint，wall
  `6651.965s`，最终峰值allocated/reserved=`36,174,262,272/42,150,658,048` bytes。
- 200步累计0 OOM/clip/nonfinite/collective stall，source policy trainable=0，validation/test
  action reads=0；contract=`a8ce75f2...00f6`。GPU已释放，下一步固定四点strict correct400。

- clean pushed`2aeb22a`在`gpu01`空闲六卡完成longest105、logical B20、full24三步profile：
  step max wall=`33.457/31.024/31.007s`，峰值allocated/reserved=`36,168,858,624/
  47,053,799,424` bytes，0 OOM/clip/nonfinite，累计1,440 queries/72 one-video conditions。
- step1只有Policy-Lane梯度符合zero-B阶段；step2起Semantic Frontend、Core、Program、
  Composer、Policy-Lane五个主块全部可达。独立fresh0→1→exact-resume1→3通过，合同
  `f0f3ec32...55261`，optimizer/scheduler/RNG/sampler/task-cycle与六rank state完整连续。
- fresh段后物理GPU0被他人占用，resume自主改用`gpu01:1,2,3,4,5,7`，仍保持sealed
  `3+3 NUMA`且未共享他人GPU。profile/smoke权重永久弃用；formal config现已seal，下一步
  clean commit/push并从全新functional-identity root训练0→200。

## 2026-08-05 PWAD训练/评测/内部分析完成并切换Policy-Lane设计

- clean`69563a0`独立fresh0→200完成200 macros、96,000 queries、4,800 videos、8个
  checkpoint，0 OOM/clip/nonfinite。四个strict correct400 roots均400 rows/42 shards/
  0 error，结果=`77/71/80/80`、breadth=`5/6/5/5`；全部评测GPU已释放，不续400。
- clean`941c5e3`六卡内部分析完成24 tasks×4 checkpoints全部96 cells与6份rank payload。
  首次聚合只因non-action rows缺少可选`reversed_0`失败；clean`c08d985`从sealed payload
  CPU恢复results/completion，0 GPU重跑、`rank_payloads_reused=true`。GPU均已释放。
- 分析证明64 atoms广泛使用，但condition mixing的16行及public B列近乎完全同向；视频
  差异弱且随训练没有增强，action层能量继续下降。PWAD正式负裁决，旧training/eval root
  禁止resume、warm-start或扩大K。
- 新design authority为`docs/action_forecast_writer_policy_lane_hyperdecoder_design.md`：
  每个public lane用同一个32维condition hidden生成全policy A/B，取消PWAD独立mixing。
  canonical原位实现已完成：旧module/config family删除，新Writer=`49,041,664`参数，聚焦
  Writer合同`84 passed`且architecture无hard/parallel family。尚未启动GPU；formal保持
  blocked，下一步live profile与独立exact-resume。

## 2026-08-05 Policy-Wide Atom Dictionary profile/resume封存

- live比较双节点后选择`gpu01:1,2,3,4,5,7`的3+3 NUMA六卡；clean`60e45f8`完成longest105、
  logical B20/B2三步profile，step seconds=`32.860/30.418/30.404`，峰值allocated/reserved=
  `35,024,829,440/44,883,247,104` bytes，0 OOM/clip/nonfinite。1,440 queries、72 one-shot
  videos，step2起五个声明block均可达，source policy trainable=0。
- 独立fresh0→1 checkpoint完整。首次resume在进入step2前由optimizer restore family
  validator fail-fast，0新增metric/checkpoint；新增family的save/schema已闭合但restore
  allowlist漏接。最小修复与聚焦回归通过后，原六卡step1→3重放成功，六rank状态、
  optimizer/scheduler/RNG/data cursor与累计1,440 queries/72 videos闭合，GPU释放。
- profile evidence与formal config已seal；下一步是修复/authority clean commit/push后的live
  preflight及全新identity root 0→200，不复用任何profile/smoke Writer权重。

## 2026-08-05 Policy-Wide Atom Dictionary实现与CPU合同完成

- 封存fresh architecture authority：16个condition policy coordinates共同组合K64个
  跨38 targets对齐的A/B atoms；旧320-slot compiler与8 factor heads已从canonical Writer
  原位删除，不加载v6或任何历史Writer checkpoint。
- 新实现、config、launch/checkpoint family、五block gradient ownership与inference/
  validation schema兼容已完成。参数量13,033,728；formal config保持
  `blocked_until_live_profile`。
- 41项聚焦回归通过，覆盖config/schema、38-target shape、step0 exact identity、conditioned
  写出、BA functional三阶段梯度、checkpoint family与既有训练合同；py_compile和diff
  check通过。尚未启动GPU；下一步是live双节点preflight与六卡longest105 profile。

## 2026-08-05 SFT-Anchored Tangent-Basis profile完成

- clean`2f934bd`在`gpu01:1,2,3,4,5,7`完成独立fresh one-cycle profile：96 rollout、
  61 successes、11 mixed、5 all-failure、two finite updates，wall`2033.38s`，peak reserved
  `19,478,347,776` bytes，0 OOM/watchdog/action-wall reads。
- 两轮都完成6/6 CUDA-ready marker后再做NCCL gradient sum；五个可训练block全可达，
  5/5 failure-only tasks有LoRA gradient，observer grad0。cycle1 Writer/trainer/6-rank state与完整
  consumed schedule已原子封存，tmux自然退出，六卡全部释放。
- CPU逐张量比较证明8 basis + 440 semantic tensors完全不变，恰好76个预注册
  coefficients tensors全部改变。profile权重永久弃用；当前下一步是clean/pushed
  formal0→1后strict paired correct400。

## 2026-08-05 SFT-Anchored macro400 diagnostic通过

- clean`303e714`在`gpu01:1,2,3,4,5,7`完成96/96 read-only rollouts：61 successes、
  11 mixed、8 all-success、5 all-failure，六个预注册机制门全部通过。wrong/shuffled/
  reversed counterfactual、binary AUC、failure dispersion和non-pixel门精确数值取design第8节。
- wall max`388.797s`、peak reserved`19,289,604,096` bytes；0 optimizer update、Writer
  backward、checkpoint、teacher/validation/test action read。tmux自然退出，六卡回到14MiB。
- 首两次启动分别在旧raw config schema和旧manifest schema fail-fast，均0 rollout；根修与
  load-only边界已提交`314948c`/`303e714`。当前下一步为clean/pushed commit上的独立
  one-cycle profile，不能把诊断61/96当作训练提升。

## 2026-08-05 参数hybrid完成与SFT-Anchored Basis实现

- live空闲`gpu01:1,2,3,4,5,7`完成正式参数hybrid：24 tasks×7 conditions×4 arms、
  8-task fixed action全部封存，wall`333.523s`、peak reserved`19,365,101,568` bytes；
  6 rank payload、results/completion完整，0 target-action/validation/test reads。结束后六卡
  均回到14MiB，tmux自然退出。
- 结果显示BA层upstream residual`.611<.727`，action层却factor-output
  `.489<.668`，且Long-39反向；据此选择SFT-Anchored Tangent-Basis，不做全factor冻结、
  scale/rank或多store修补。
- canonical RL config/schema/contract已原位升级，以v6-fast macro400 cold start并冻结
  semantic encoder与8个factor-output basis。progress diagnostic改为checkpoint无关的
  机制门；旧A100 source路径通过已有source identity跨host匹配，不改历史contract。
  一次性hybrid分析入口已删除。聚焦回归`24 passed`，pycompile、JSON与diff check通过；
  architecture guard无hard violation且净删除563行。尚未启动macro400 diagnostic。
- clean`a2d3f8d`首次macro400 diagnostic在所有rank加载AS config时立即被退役历史v6 schema
  拒绝，未构造模型、未做rollout/checkpoint，六卡回到idle。没有恢复旧schema；canonical
  config改回当前受支持v6 owner，并在RL contract中显式封存macro400原始32-frame
  non-parameter runtime override，runtime/evaluator统一解析。effective authorities/Writer
  与旧run contract逐项相等，聚焦回归`21 passed`；等待clean push后重新live preflight。
- 第二次diagnostic完成六份模型构造后在真实macro400 manifest处fail-fast：旧checkpoint
  schema不在现行exact-resume注册表，仍为0 rollout/checkpoint且GPU全部释放。兼容逻辑已
  归入`checkpoint_schema.py`，仅由IL→RL `initialize_writer_phase`显式开启；default验证、
  exact-resume与AS evaluator均保持拒绝。真实artifact的manifest canonical payload、owning
  contract、逐文件size/hash、launch schema、authorities、Writer、cursor与writer SHA已一次
  性通过；聚焦回归`22 passed`、architecture guard无hard violation。

## 2026-08-05 Progress-Credit cycle2训练、rollout与续训轴停止

- 在live空闲`gpu01:1,2,3,4,5,7`、原sealed 3+3 NUMA topology上从同一formal
  cycle1 exact-resume到2。第二cycle完成96 rollout、24,501 actions、49 successes、
  16 mixed、5 all-failure semantic、3 all-success与21 active-credit tasks；two-epoch
  wall`2056.376s`、peak reserved`19,457,376,256` bytes，完整checkpoint/双ledger、
  0 watchdog/OOM。训练结束后六卡释放。
- live复查后用`gpu01:4,5,7`完成cycle2 strict correct400：400 rows、42 shards、
  success102、wall`1486.017s`，0 error/retry；Writer cache400/400且rollout结束后三卡均
  回到14MiB。结果root为
  `runs/outputs/pi05_task_grounded_progress_credit_cycle002_bci_correct400_noreplacement_seed7_56a167d_20260805`。
- 相对cycle1严格paired gained/lost=`15/17`、breadth`4→4`，逐task从
  `11/0/0/43/31/19/0/0`变为`11/0/0/43/26/22/0/0`。无新coverage且Object能力换手，
  按预注册门停止同root cycle4/8。
- 400对LoRA与Writer权重分析封存在cycle2 root的
  `paired_to_cycle1_and_as125_analysis.json`：gained/lost更新幅度与norm增长不可区分，
  near-rank1不变；Adam后factor每参数位移仅约为visual的2倍，故下一步先做固定panel
  参数hybrid分解，不按raw gradient直接冻结decoder，也不加scale/rank修补。

## 2026-08-05 Progress-Credit cycle1 paired correct400完成

- live检查后在`gpu01`并行使用两组互斥卡：1/2/3评AS125，4/5/7评formal cycle1；
  每组3 replicas/GPU、3 Writer generators/GPU、batch4。两面板各400 rows、42 shards，
  wall`1470.979/1487.315s`，无OOM/失败/重试，完成后六卡自然释放。
- AS125/cycle1 correct=`97/104`，严格paired gained/lost=`22/15`、breadth=`5/4`。
  Object-1从24升31贡献全部净增，Spatial-1从1降0；其余task净变化为0或+1，不能写成
  broad improvement。
- CPU只读分析覆盖400对LoRA和1,520个target谱样本：effective BA变化中位`.01677`、
  cosine`.999860`；top-1 energy与stable rank几乎不变，仍为near-rank1。分析已封存在
  cycle1 eval root的`paired_to_as125_analysis.json`。
- 裁决为仅续同一formal root cycle1→2，再用同一strict panel复评；不改config、K/Nmc、
  两epoch、task/video schedule或3+3 topology，不按train reward或+7直接续4/8。

## 2026-08-05 Progress-Credit formal首次失败与ready根修

- clean`bc4ff60`在`gpu01:1,2,3,4,5,7`启动AS125-fresh formal0→1。96 rollout和24
  progress-credit均完整，但第一轮gradient sum只有rank0/1/2/5进入seq18，rank3/4停在
  seq17；600秒watchdog终止，0 update/metrics/checkpoint，失败root禁止resume/评测。
- 根因收敛为旧`FileStore` ready没有显式等待本rank CUDA完成，且临时store不能在高度
  错峰下可靠提供一次性all-rank barrier；不是OOM、transport、task ownership或科学负结果。
- canonical代码已改成CUDA synchronize→本次torchrun唯一session/cycle/epoch原子rank
  markers→NCCL。相同输出目录连续两次真实六卡新session探针均6/6 markers、sum21，
  旧session marker没有污染重启。
- clean/pushed`30977b5`随后在`gpu01:1,2,3,4,5,7`用全新retry1 root完成原96-rollout/
  two-epoch formal重放。两轮均严格经过6/6 markers再NCCL，2 finite updates、完整cycle1
  checkpoint、0 watchdog/OOM；wall`2125.726s`、peak reserved`19,455,279,104` bytes。
- 5/5 all-failure task LoRA梯度非零，五block可达、observer grad0；六rank双ledger通过
  validator。GPU自然释放。下一动作是AS125 baseline/cycle1 strict paired correct400，
  不在评测前resume cycle2。

## 2026-08-05 Task-Grounded Semantic Progress Writer profile通过

- clean`84d856c`在`gpu01:1,2,3,4,5,7`完成AS125-fresh profile：96 rollout、24,593
  actions、50 successes，two-epoch wall`2129.187s`、peak reserved`19.455GB`。两轮
  deferred FileStore→NCCL均按实际负载完成，0 watchdog/OOM/clip，GPU自然释放。
- 5/5 all-failure task有nonzero generated-LoRA gradient，五下游block两轮均非零，
  observer gradient tensors=0；19 active tasks严格等于14 mixed+5 semantic failure。
- formal配置seal为fresh AS125、6 ranks、two epochs、total8、checkpoint1/2/4/8；首段
  0→1后先比较AS125 baseline与cycle1 strict correct400。profile checkpoint禁止续训。

## 2026-08-05 Task-Grounded Semantic Progress只读门通过

- clean`c483497`在live空闲`gpu01:1,2,3,4,5,7`完成AS125严格配对只读诊断；96
  rollouts、24,600 actions、50 successes，14/5/5 outcome分组及96/96身份与旧profile
  一致。0 optimizer/backward/checkpoint，wall`401.874s`，peak reserved
  `19,289,604,096` bytes，GPU自然释放。
- mixed agreement=`13/14`、pair AUC=`.8913`；五个all-failure utility range均`>.12`；
  correct对wrong/shuffled/reversed胜率=`1/.88/1`，failure utility与pixel Spearman
  `=.5564`。全部预注册门通过，profile gate已开放，formal继续fail-close。
- canonical Writer-update路径已接入binary-first显式advantages和冻结observer：mixed
  精确保持binary LOO、all-success零、all-failure semantic LOO。下一步是从AS125 fresh
  运行一个不可续训的full24 K4/Nmc4 two-epoch profile并裁决梯度/NCCL/A40合同。

## 2026-08-05 AS125、K4、两点内部审计与binary-only负裁决

- 已封存下一阶段design
  `docs/action_forecast_writer_task_grounded_progress_credit_design.md`。它永久冻结AS125
  semantic encoder作为action-free progress observer，用task-grounded patch和固定
  Action-Expert interaction的teacher/rollout首尾内容delta构造bounded potential；先跑
  预注册只读五门，未过门不得启动Writer更新。mixed binary信用不变、all-success为0，
  仅all-failure允许semantic LOO；profile仍必须从AS125 fresh且不得续权重。
- canonical实现已升为fresh progress-credit schema并接入同一launcher的只读
  `diagnostic`模式：rollout只额外保留旋转后agentview起点/terminal RGB，observer永久
  冻结且不接收action/proprio；六rank payload联合计算paired K4、content、binary
  agreement、all-failure dispersion、三种视频反事实与pixel nuisance门。diagnostic固定
  0 optimizer/backward/checkpoint，profile/formal当前fail-close。23项reward/RL与53项
  evaluation/model聚焦回归通过，compileall通过，architecture guard无hard violation。

- 同一AS root从step100 exact-resume到125：60,000累计queries、3,000 videos、125
  finite macros，segment wall`806.928s`；step125 checkpoint、metrics125 rows与summary
  完整，0 OOM/clip和0 validation/test action reads。首次命令漏传`--num-workers 0`
  被resume contract在step101前拒绝；补齐完整命令后成功，未产生污染checkpoint。
- live比较两节点后在`gpu01:1,2,3,4,5,7`完成step125 K4 profile，root为
  `runs/outputs/pi05_rl_writer_relative_flow_profile_from_v6_macro125_r6_bci_6fe4e52_20260805`。
  96 rollouts、24,600 actions、50 successes、19/24 coverage、14 mixed；相对step100
  gained/lost/retained/both-fail=`10/12/40/34`，静态身份与共同noise prefix严格一致。
- 两epoch ratio/grad finite、clip0；实际mixed负载不均下两轮FileStore ready→NCCL顺序
  正常，完整cycle1 checkpoint、0 watchdog。profile GPU自然释放，权重禁止续训。
- 同一clean commit完成step100/125两点内部审计：48 rows、6 rank payloads、wall
  `194.743s`、peak reserved`19,306,381,312` bytes、0 target-action/validation/test reads。
  norm增大但video/action条件差异不增；success变化与video-energy变化显著负相关，持续
  全失败tasks反而变化更大。binary-only Flow-Credit据此负裁决，不续AS150或formal RL。
- 下一步先设计并profile teacher-video语义状态变化credit；必须对全失败trajectory给出
  content-grounded相对次序，同时证明score不由轨迹长度或teacher帧ordinal时钟解释。
  只有机制门通过才做Writer update与paired closed-loop评测，长期`>150`目标不变。

## 2026-08-04 Policy-Target-Owned Factor训练、rollout、内部分析与暂停

- profile与authority封存commit为`34be4a0`。正式launch live比较两节点后只使用
  `gpu02:1,2,3,4,5,7`，在detached clean worktree从fresh identity完成macro0→200；
  root为
  `runs/outputs/pi05_as_writer_target_owned_factor_bci_rawfull24_decay400_formal_r6_b20_micro2_seed7_formalvideo20260722_34be4a0_20260804T051244Z`。
- 200 steps/200 task cycles、96,000 logical queries、4,800 one-video conditions、
  8 every25 checkpoints均完整；wall`6678.957s`、0 clip/OOM、峰值
  allocated/reserved`33.696/38.729GiB`，0 validation/test action reads。runtime
  contract为`6af3b4fe...904b`，profile或其他Writer权重均未进入。
- 50/100/150/200四个strict paired correct400全部完成：每root 400 unique rows、
  42 shards、9 workers exit0、每task 50个teacher demos无放回且无retry/adoption。
  曲线`99/76/86/68`，breadth=`6/6/7/5`；逐task为
  `9/0/1/44/38/6/1/0`、`5/0/4/33/28/2/0/4`、
  `7/0/1/26/39/10/1/2`、`7/0/0/31/27/2/1/0`。相邻gained/lost=
  `15/38,35/25,18/36`，union/intersection=`136/37`，envelope gap37。winner50=99
  明显低于Direction Store129和v6-fast143，按门不续400。
- macro50 refs1五条件内部分析在live空闲的同六卡完成，wall`100.864s`；6份rank rows、
  8 tasks、5 conditions、strict replay、rank gauge、checkpoint unchanged和信息墙全通过，
  0 rollout。root为
  `runs/outputs/pi05_as_writer_target_owned_factor_bci_macro050_internal_refs1_seed7_34be4a0_20260804`，
  contract`7a45a2c2...e77868c`；GPU随后自然释放。
- q/v跨层BA余弦降为`-.00011/-.00030`，证明76 heads的policy-target ownership真实
  生效；但LoRA norm仅`19.0257`、layer CV`1.9607`、q/v top-4能量
  `.7329/.8529`，比Source-SFT更过度集中。same Program→factor→BA→action为
  `.90933→.05842→.09119→.03161`：差异写出增强但policy action利用下降。
- 完整200步梯度显示factor占单task能量`69.25%`，task pair cosine`.0040`、负pair
  `.4457`、full24能量保留`.0484`；相同task+demo的CountSketch重现余弦仅`.0046`。
  由此把最早失败接口更新为condition-to-policy credit缺少可重复task/video方向，正式
  否定policy-target sharing作为主要根因。按owner要求，rollout与全部分析结束后暂停，
  不启动下一架构、训练或评测，等待讨论；长期`>150` Goal仍未完成。

## 2026-08-04 owner恢复推进与Policy-Target-Owned Factor实现

- owner明确授权恢复长期目标，后续科学问题自行深入分析并继续，不为中间判断再次
  请求确认；仍不使用subagent。当前先完成设计/实现，尚未启动GPU或新formal run。
- CPU直接读取两套现有rank-128 Source-SFT step400，按gauge-invariant `BA`复核谱与
  target/layer能量。两套SFT均由q-dominant低秩更新主导，但层能量profile高度稳定且
  异质：q/v profile Pearson`.9931/.9904`、rank correlation均`.9835`、对应target BA
  cosine`.8450/.8529`。该证据把问题从“增加effective rank”改写为“解除factor
  decoder跨policy-target硬共享”。
- 新authority为
  `docs/action_forecast_writer_target_owned_factor_design.md`。保留Target-Bound的
  mean-backed Core、private A/E/D causal Program、38-target/16-rank reader；每个公开
  A/B tensor使用独立完整`1024→256→width`head。没有task store、frozen route、谱/正交
  loss、SFT teacher或static bypass，未来reward gradient可复用同一架构。
- 唯一canonical路径已原位替换：Direction Router/Store Head、frozen task-anchor
  forward及其专用internal字段/测试退役；training/eval/checkpoint/internal analysis
  继续复用原owner并切到fresh schema/config。活动源码相对HEAD净减少334行，不保留
  parallel runner或compatibility resume。
- 预期并由模型合同锁定Writer总参数`47,857,920`，其中76个target-owned factor heads
  `40,517,632`。compileall、config loader/fresh checkpoint family、89项Writer tests和
  显式BCI assets环境下52项聚焦model/config/eval tests全部通过。
- 实现与authority已由`20479d3`clean commit/push到branch/main。live比较后选择
  `gpu02:1,2,3,4,5,7`六张空闲A40；formal-seed fresh0→1工程smoke finite，loss
  `.150377`、单步`32.668s`、峰值allocated/reserved`33.325/38.729GiB`，480 queries、
  24 conditions均完整。但实际最长只有82帧，故只保留工程证据，不冒充longest105。
- 根因是profile mode没有消费配置中已声明的profile seed172。canonical runtime现按
  mode在内存中解析有效teacher seed并写入run contract，磁盘formal seed始终保持
  `20260722`；24项受影响聚焦测试通过并以`e03e61b`push。
- clean`e03e61b`在`gpu02:1,2,3,4,5,7`完成真实longest105 fresh0→1与同contract
  exact-resume1→3。三步耗时`34.249/32.273/31.187s`、loss
  `.150377/.152275/.140054`，峰值allocated/reserved`33.695/43.936GiB`；1,440
  queries、720个B2 forwards、72 conditions完整，0 clip/OOM/CUDA/NCCL错误、0
  validation/test action reads。step1只factor非零，step2起五主块finite/nonzero。
- config已选择B20/B2并seal formal seed`20260722`与fresh0→200；profile/smoke权重不
  进入正式轨迹。下一步按正式launch gate重查clean code、两节点空卡与storage后启动。

## 2026-08-03 Semantic Direction Store正式裁决、六卡分析与暂停

- clean pushed`91feeef`在`gpu02:0--5`从fresh identity自然完成macro0→200：200个
  finite rows、96,000 logical queries、4,800 one-video conditions、8 checkpoints，
  wall`6619.255s`；0 clip/OOM、0 validation/test action reads，峰值reserved
  `39,806,042,112` bytes。logical B20/physical B2、full24 raw mean和formal seed均未改。
- 四个严格配对correct400均为400 unique rows、无失败，曲线
  `129/107/120/129`，breadth=`7/7/7/5`。逐task依次为
  `7/2/0/42/45/31/1/1`、`5/1/1/37/37/22/0/4`、
  `9/2/0/40/40/26/2/1`、`10/0/0/38/41/36/0/4`。
  macro50与200同分，按breadth选择macro50；仍低于v6-fast143和严格门151。
- 相邻gained/lost=`17/39,43/30,27/18`，四点union/intersection=`174/65`、single
  envelope gap45。Direction Store相对SFB macro50提高60，证明独立store显著改善早期
  acquisition；但后续跌落与恢复、breadth晚期收缩说明task漂移未解，按门不续400。
- step133按task-pair共享0/1/2 stores分层的factor-gradient cosine均值为
  `-.00043/.00664/.02249`：store ownership局部隔离梯度，但store内部仍近正交。
- macro50 refs1五条件内部分析显示route跨video固定；8 validation tasks的ordered
  top2数组均不同，但`1,5`与`5,1`是同一无序组合。same-task-other的
  Program/factor/effective-BA relative-L2为
  `.93377/.01935/.03242`；A/E与Core mean carrier均能到达BA/action，动态路径未断。
- 16个rank坐标全部active，但effective LoRA stable rank=`1.000043`、entropy rank
  `1.000371`、top singular energy=`.999957`、B-column cosine=`.999971`；完整独立
  stores仍共同写入近同一B方向。正式负裁决parameter-store-only假设，下一根因位于
  Program到public A/B的多维功能方向形成与组合。
- 内部分析最初暴露两处历史4-rank假设：LPT assignment缺省4 ranks，以及final seal
  强制4 payload/每rank2 tasks。`f82c7cd`与`a115b06`分别改为实际`world_size`驱动
  ownership和Cartesian sealing；8项定向测试及clean六卡真实8-task×5-condition运行
  均通过，成功root为
  `runs/outputs/pi05_as_writer_direction_store_bci_macro0050_internal_refs1_seed7_retry2_a115b06_20260803`。
  原失败roots保留工程诊断，不混入科学结果。
- 多卡根修规则已写入`AGENTS.md`并push到branch/main。正式训练、四点rollout和全部
  当前假设所需内部分析均已结束；本方GPU进程已退出。按owner要求在此暂停，不启动
  下一架构、训练目标或rollout，等待讨论。

## 2026-08-03 owner恢复推进与Semantic Direction Store设计

- owner确认继续严格one-shot，取消Writer参数量软上限，要求优先重构条件生成方向
  如何存储/组合，必要时配套修改训练；仍禁止subagent，并要求减少无关hash/防御性
  扫描，以聚焦验证和真实实验效率优先。
- 重新核验clean branch`codex/bci-continuation`、HEAD/origin-main=`02ce673`及canonical
  SFB/VR代码。未启动GPU、训练或rollout。
- 完成新authority
  `docs/action_forecast_writer_semantic_direction_store_design.md`：frozen task-language
  anchor固定top2八个full-capacity direction stores，完整Core/A/E/D只进value，预计
  Writer参数37,355,776；首跑保持RAW full24/B20和现有optimizer。
- owner补充长期可迁移性要求：架构机制需能在reward gradient下成立，AS sampler与
  LIBERO固定常数不得冒充方法；已写入design的objective-agnostic边界和失败后禁调项。
- 已原位替换SFB router/head，不保留parallel model/runner；Writer实际参数
  `37,355,776`。frozen text-only anchor移出activation-checkpoint重算，完整Core/A/E/D
  继续作为value；B20 keyed independent Beta/Gaussian可由B2×10逐slice精确重建。
- 仅用24 train languages建立center authority：减去raw-anchor train24均值后，seed7
  spherical k-means两轮收敛，primary/top2计数为`5/7/6/1/2/1/1/1`与
  `7/11/6/4/4/5/3/8`。当前center固定，不再调K/seed/route权重。
- shape、route、identity/freeze、selected-store gradient、信息墙、B20/B2 loss与LoRA
  leaf-gradient parity、六rank topology和fresh checkpoint family共61项focused CPU
  测试通过。architecture guard无hard violation；下一步只做clean pushed最长105-frame
  fresh/exact-resume GPU profile，不从VR checkpoint warm-start。
- clean `7b13b6c`首次六卡profile在任何active collective或训练step前触发NCCL 480秒
  heartbeat；六rank均报告`only active collectives: 0`。现场显存/进程证明耗时段是
  rank-local source CUDA构造，失败root已停止且不resume。按owner要求不使用timeout/
  watchdog覆盖，改为各rank本地policy/Writer/optimizer构造完成并经非NCCL all-rank
  ready rendezvous后才初始化NCCL，再以原六卡B20/longest105规模重放根因修复。
- `78d8b4f`重放不再出现无active-collective heartbeat，而是在六rank对称
  `SeqNum=1/ALLREDUCE/Numel=1`处暴露explicit launch漏传BCI已知transport合同：A40+
  NCCL2.28 direct P2P/CUMEM 600秒hang。相同`gpu02:0--5`六卡显式
  `NCCL_P2P_DISABLE=1`后，sum21、BF16 finite与两次all-reduce 10.5秒通过。代码和
  `AGENTS.md`现都要求BCI多卡SHM fail-fast，不依赖`.env.local`偶然source。
- clean `eaa8bce`在新空闲六卡`1,2,3,4,5,7`再次以9.2秒通过sum21/BF16 collective，
  随后完整越过ready rendezvous、NCCL和run-contract发布。step0唯一失败是
  `as_step.py`退役重复method白名单未列入Direction Store；`as_config.py`已经拥有完整
  canonical conditioning验证，故删除第二名单并把新method加入真实step聚焦测试。
- 重复guard修复经16项step聚焦测试后提交/push为`1d0507e`。live重查选择
  `gpu02:0--5`六张空闲A40，显式SHM transport；fresh0→1再exact-resume1→3完整通过，
  contract`749773d8...8fd6`，三步`33.451/31.823/31.025s`、loss
  `.150377/.152492/.142434`，峰值allocated/reserved
  `35,827,363,840/47,129,296,896` bytes，无OOM/clip。
- profile覆盖105-frame真实视频、1,440 logical queries、720个B2 physical forwards和
  72个one-video conditions；validation/test action reads均为0。step1按identity只打开
  factor output，step2起五个主块全部finite/nonzero。配置已恢复formal seed`20260722`、
  选择B20/B2并seal fresh0→200；profile权重不进入正式轨迹。

## 2026-08-03 BCI VR正式训练、rollout与阶段暂停

- formal teacher seed fail-close修复提交/push为`d9130c9`，并重新核验clean
  `HEAD=origin/main`、gpu01/gpu02进程与ownership、`/data1`独立quota、source/data/
  tokenizer/assets和全新retry1 root。gpu01八卡均有他人任务；gpu02只选空闲
  0/1/2/3/4/7，5/6从未触碰。
- retry1从fresh identity完成macro0→200，wall`6619.670s`；200 finite rows、
  96,000 queries、4,800 videos、8 checkpoints、0 clip、0 validation/test action
  reads。64/64 checkpoint payload与8个held panel完整性复验通过，训练自然停在预注册
  stage stop200，没有resume到400。
- A40 evaluator先以8-rollout smoke确认r3 fresh-cache路径；复用不同generator-count
  cache的尝试被manifest fail-close，未启动worker、没有科学row。正式采用3 GPUs/panel、
  3 replicas/GPU、3 generators/GPU、generation batch4；live显存保持在46GB卡内。
- 两批并行完成macro50/100与150/200正式correct400，只用gpu02空闲六卡；四个root均
  400 rows、42 shards、9 workers return0、无放回teacher demo与hash审计通过。曲线为
  `76/88/126/107`，breadth=`7/4/7/5`，single winner macro150=126。
- 完成全部预注册分析：success union/intersection=`158/49`、150→200 gained/lost=
  `21/40`；相对SFB同点delta=`+7/-3/+8/-20`。全200步matched gradient稳定性只
  微幅、分阶段改善；macro200 held loss最好却closed-loop下降，正式把根因转向
  functional surrogate与closed-loop有效流形错位。
- 正式root、逐task、paired gained/lost、mechanism与hash写入active handoff、execution
  brief、VR design、findings和task plan。owner要求rollout与全部分析后先暂停；当前
  无EMBER tmux/worker或本方GPU占用，不续训、不做五臂、不启动下一方法，等待owner
  继续指示。长期single-checkpoint `>150` Goal未完成。

## 2026-08-03 BCI接管与当时六卡VR工程profile

- 旧ledger曾把owner授权误记为“跨两节点最多6张”；owner于2026-08-11明确纠正：从未设6卡硬上限，应比较
  `gpu01`/`gpu02`后选择一个节点并使用该节点所有真正空闲且提高吞吐的卡。当时目标是缓解task漂移并让
  同一single checkpoint correct严格超过150，且当时禁止subagent。
- 完整读取34项authority与Target-Bound/SFB设计，核验项目树、迁移hash、环境、quota、
  source checkpoint、tokenizer、LIBERO assets和历史结果。现场`gpu01`全忙，`gpu02`
  0/1/2/3/4/7空闲；只用这六卡完成NCCL/BF16 smoke。
- 当前分支`codex/bci-continuation`实现logical B20/policy microbatch2和6 ranks×4 tasks；
  full-B20 keyed Latin/antithetic draws、task mean、full24 raw mean与一次AdamW保持不变。
  23项focused回归通过。
- 未提交源码上的工程profile完成fresh0→1、exact-resume1→3；每步24 tasks、480
  logical queries、240 physical forwards，峰值allocated/reserved为
  `34,970,270,208/47,108,325,376` bytes，五主block从macro2起finite/nonzero，
  validation/test action reads为0。由于不是frozen commit，下一步是提交/push后重放，
  不从该checkpoint继续formal。
- 23项focused、226项全仓回归与结构守卫通过；实现提交/push为`391f183`。clean frozen
  profile重新完成fresh0→1与exact-resume1→3，contract`31ea4bc9...55de0`、峰值
  allocated/reserved`34,970,270,720/47,108,325,376` bytes，正式seal条件满足。
- 首次frozen resume在第二invocation前出现一次15分钟stall并只终止本方进程；同六卡
  object-collective探针和同一原命令重试随后通过。没有足够证据改代码，记录为未归因
  transient；formal仍fresh启动并保留live timeout/进程监控。
- BCI fresh formal 0→200 launch contract已预注册到active handoff：固定项目内
  source/tokenizer/data/output/log、6-rank full24 logical-B20、96,000 queries、8个
  checkpoints和1.5GiB峰值预算；启动只等待最终clean `HEAD=origin/main`及live GPU
  preflight，不从profile warm-start。
- commit`6f18499`首次formal通过Git/GPU/storage/start门并完成10个finite宏步，但现场
  matched审计发现longest105 A40 overlay把profile专用teacher seed`172`误带进sealed
  formal；正式基线与配置自述要求`20260722`。在首个checkpoint前只终止本方tmux，
  GPU0/1/2/3/4/7均释放，5/6他人任务未受影响；该partial root禁止resume/评测。
  10-row/0-checkpoint incident marker SHA256为`9d5d03b8...cf9907`。
- actual teacher seed已修回`20260722`，config loader新增sealed seed一致性fail-close，
  23项training/functional focused回归通过。retry1使用全新root/log/tmux并需重新走
  clean pushed commit、双节点live GPU preflight与fresh identity门。

## 2026-08-02 19:18 UTC Post-seal研究窗口重新开放

- owner在迁移由另一session运行期间，重新授权约十小时A100研究；GPU边界仍严格为
  物理4--7，要求关键代码先上云、所有新增artifact形成二次迁移delta。
- 以`f9a144c`为封存基线建立
  `/data/ymdai/migration_manifests/ember_postseal_20260802/`，并创建独立
  `codex/postseal-target-bound` worktree。
- 主进程完整复核CV-ADR、historical Coherent-Procedure与Target-Bound authority。
  第一实验选择Target-Bound：task/Core语义先绑定38个真实targets，A/E/D保留private
  causal value到rank read；它尚不能被宣称解决shared factor-head互扰，该项由正式
  task breadth/churn与分块梯度证伪。
- `b260a57`已移植到post-seal基线为`fbbb784`；环境从lock恢复中，尚未查询GPU或启动
  profile、训练、评测。
- 环境按`uv.lock`恢复；缺失系统C++编译器只用便携CMake/Zig完成`hf-egl-probe`构建，
  不改变依赖。显式`EMBER_LIBERO_ASSETS_ROOT`后48项聚焦回归全过。
- 首个普通formal-seed三macro虽健康但max95 frames，未冒充105合同；新增并push只改
  teacher schedule seed的profile overlay`e8fb96c`。真正seed172 profile首macro
  max105，三macrowall59.07s、峰值reserved83,506,495,488 bytes、五主block梯度可达。
- 正式seed fresh0→1→exact-resume1→3通过；三步loss`.15404/.15141/.14509`、cursor/
  scheduler/RNG连续。`e8fb96c` formal调用于19:53:46 UTC在任何模型加载/macro前被
  `formal_run.status=pending_profile`正确拒绝；失败root无metrics/checkpoint。已把live
  evidence写入base config、更新两个overlay SHA并通过27项定向回归，下一调用使用新
  commit与全新root。
- sealed live evidence提交`cfd26df`已push并成为`main=origin/main`。从clean detached
  frozen worktree于20:01:34 UTC启动正式fresh macro0→200；tmux
  `ember_tb_formal_cfd26df`，只用GPU4--7、4 ranks、B20、every25。
- start contract为11,092,224 Writer参数、source trainable=0。前两个macro各完整覆盖
  24 tasks、24条single videos和480 independent queries，rank内long-first且跨rank
  cost-balanced；loss`.15404/.15159`、reserved约77.77GiB、无clip/OOM/nonfinite。
  第二macroSemantic Frontend/Core/Program/compiler/factor五block均finite/nonzero。

## 2026-08-02 A100清理与BCI迁移准备

- 核验EMBER pre-cleanup `main=origin/main=f0b123f`、MemLLM
  `main=origin/main=edc549d4`，两repo工作区clean；无训练、评测、torchrun或tmux。
- 创建EMBER 138-ref recovery bundle并验证，SHA
  `c78fb94d...44ec7`；MemLLM 186-ref bundle复验SHA`feef0e90...a7656`。
- 分批清理并留下精确外部ledger：52个Writer LoRA caches 55.39GB、138个operation
  roots 27.98GB、退役SmolVLA outputs/numeric及旧asset cache约50.90GB、source
  rejected EMA/optimizer state约
  24.48GB、superseded/endpoint caches约2.83GB、reseal/capacity roots约1.00GB、
  generic base14.47GB和Codex archive5.65GB。
- `test_pi05_eval_contract`随后发现`hf-libero`的active assets symlink被上述asset清理
  误伤。只恢复`lerobot/libero-assets`精确revision`0b3ea86...`的586文件/426.57MB；
  file-list SHA`721aa248...96b9`，原4个失败测试复跑全过，旧额外cache不恢复。
- source step1000保留9.35GB selected raw policy、trainer state和原manifest；formal
  inspector验证policy SHA`60ea7ee8...df36`。该资产改为inference-only，不再可
  exact-resume source训练。
- outputs由约214.53GB降至102.85GB；writer LoRA cache和endpoint LoRA tensor均为0。
  60个formal checkpoint roots约74.9GB、406个complete eval roots约1.1GB、CV exact50
  和endpoint10正式负证据全部保留。
- 删除55个clean辅助worktree和全部本地实验branches/stash；A100只留main checkout。
  Target-Bound commit`b260a57`仍在GitHub远端，所有旧refs另由bundle保存。
- 用`uv pip freeze`封存EMBER 171行、MemLLM 73行环境；分别SHA
  `ee072580...0956`/`9fda882d...24d3`。全仓验证完成后删除EMBER 9.74GB venv和
  可重建uv/Hugging Face caches；owner随后关闭MemLLM venv消费者，复核只有另一用户
  `/data/pcpan`环境的`nvitop`仍在运行且未触碰，于是也删除7.60GB MemLLM venv及其
  ignored workspace link。两个环境均不进入迁移。
- `src/ember/pi05_eval/launcher.py`增加`EMBER_STORAGE_ROOT`，使BCI容量preflight不
  依赖`/data/ymdai`；targeted pytest通过。
- 新增`docs/a100_to_bci_migration_handoff.md`和机器可读资产表，重写README、AGENTS、
  active handoff、execution brief；明确GitHub/SSH/重下载分流、MemLLM反向路径映射、
  no-Codex migration和新agent接手顺序。
- simulation-assets原始4个回归、相关20个launcher/contract tests、EMBER全仓pytest和
  compileall均通过后才删除A100专用EMBER venv；cleanup `SHA256SUMS`全项复验通过，
  其自身SHA为`338f8a0b...7173a`。

## 2026-08-02 CV-ADR GROUP4正式控制完成与阶段交接

- frozen `51c0ba5` GROUP4自然完成1200 updates/200 cycles、96,000 queries、4,800
  one-video conditions，wall`4944.554s`，all finite、1 clip、validation/test action
  reads为0；final checkpoint为step1200。
- 四个paired correct400已完成且0 error：cycle50/100/150/200=`82/77/73/110`。
  observed-best single checkpoint为cycle200/step1200；逐task
  `10/0/0/41/38/15/2/4`，breadth6、top2`71.82%`。同topology RAW为
  `76/111/99/117`，GROUP4四点均值低15.25、winner低7；按门不做五臂。
- 封存correct400 curve artifact，analysis/canonical SHA为
  `54cd40e5...a985`/`e5b00932...ba6`。GROUP4四点union150、intersection32、
  envelope gap40，漂移继续；相对source保留42、gain68、lose6，表现为更保守而非
  共同增长。
- 完成training dynamics与RAW参数交互审计；analysis/canonical SHA分别为
  `56b206f6...4563`/`92470478...219b`和
  `81c64e7b...119`/`c6ce55c4...015c`。held loss与closed loop错位、factor持续占
  约94%梯度能量、GROUP4参数段较短却与RAW低余弦。
- 完成cycle200 exact50：400 rows、五条件真实frame-order forward、0 rollout、四rank
  无failure。A+D/remove-A/remove-D职责门从RAW `8/1/5`降到GROUP4 `0/0/0`，same-task
  BA variance略降，norm却`64.24→72.06`；封存职责audit与RAW×GROUP4 compare，file/
  canonical SHA分别`9725f010...b292`/`dc01dd97...5141`和
  `a9f1e615...329f`/`2dc9ee29...5f4d`。
- 最后一项GPU工作自然退出；GPU4--7为6--8MiB且无compute process，本任务无活动
  训练、评测或tmux。Target-Bound未做profile/resume/training。
- Target-Bound现有CPU实现已作为隔离feature commit
  `b260a57a94dc21bd3446b212bfa42f71b037ce13` push；49个受影响测试与compileall通过，
  main仍保持CV-ADR canonical，没有合并或运行该分支。
- owner暂停边界：上述winner exact50完成并封存后，不启动Target-Bound/SPG或任何
  下一架构GPU profile/训练；只完成CPU对照、文档、Git和无活动进程核验后汇报。

## 2026-08-02 CV-ADR full400八点负裁决与matched方差诊断

- frozen `254ade4`的RAW从step200 exact-resume到400自然结束；累计400 cycles、
  192,000 queries、9,600 one-video conditions，metrics连续、every25 checkpoints
  完整、all finite、0 clip，训练tmux自然退出。
- 生成full400 training-dynamics artifact；analysis SHA256为
  `7289eef422a4021c9cd57504d90cb83ef27a34f132a61336c9007d4d000d7a4f`，canonical
  payload为`82ed1026a6a82657bbb43bb4a45b1c06f6a146b5ad3cf07e7f13d4e8e1990f40`。
  它确认late global mean无candidate-negative task，但条件梯度约99.5% centered、
  参数段方向仍不稳定，held functional loss横盘。
- live preflight确认main/origin、frozen worktree、assets、四checkpoint、新root、存储
  和GPU4--7；`/data/ymdai`约418.77GB，预计峰值低于500GB。未查询GPU0--3。
- macro250/300/350/400的4个正式correct400均自然完成、0 error，为
  `77/69/80/82`。完整八点winner仍是macro200=`117`；formal analysis SHA256
  `fb75464fca28ef01d764579b32eba98836b6dbe53288188fe2a44424d57ec90a`，canonical
  payload`bd4f43d417db52326ec81d4f45820d4920da5d3a9b02175f628425082ccce909`。
- 200→250 lost56/gained16，后段能力未恢复，且effective BA norm没有坍缩；RAW
  不做五臂，正式进入同topology GROUP4因果格。
- 创建只读matched gradient diagnostic：macro200/400共用visit397--399，按24 tasks
  分解3 video×3 B20 query×3 paired flow及3 Gaussian×3 Beta time，零rollout/update。
  两端已自然完成；analysis SHA为`1727f014...e7656f9`/`61a13978...db40520`。
- 生成严格matched pair artifact：video梯度主效应约`.1%`且0/24 tasks主导，query/
  flow主导；macro400 task-mean能量下降而随机能量不降。visit397--399对macro400
  是刚曝光的train条件，对macro200尚未曝光；24/24 loss改善只说明train surrogate/
  recency拟合加强，不能冒充held泛化，而correct400崩落。pair SHA为
  `ad7d6e06...44eb96a`、canonical`d21c2cfc...d38b08`。
- 从clean detached `f6cf775`仅用GPU4--7完成GROUP4 longseed172 18-update/3-cycle
  B20 profile：每cycle 24 tasks各一次，包含`task38/demo36=105` frames，峰值
  allocated/reserved `76,945,014,784/77,370,228,736` bytes，all finite、0 clip/OOM。
- formal-seed fresh0→1→exact-resume1→3→7通过；step1/3逐文件SHA/size/mtime未改写，
  首cycle与scheduler/cursor连续。profile/resume metrics SHA为
  `f8afb6ae...d90a`/`53cf0718...9de`，config已seal，当前无活动进程。
- post-seal `51c0ba5`已push；live preflight只查GPU4--7，四卡各8MiB无进程，个人
  用量`424,594,886,656` bytes，root/log/tmux均全新。现已从fresh identity启动
  GROUP4 0→1200，tmux `ember-cvadr-group4-m1200-51c0ba5`。首cycle恰好24 tasks/
  24 videos/480 queries；updates1--41 finite、0 clip，scheduler/cursor和信息墙正确。

## 2026-08-02 v5.2×v6 recipe/video-causality联合审计

- 新建只读正式审计root
  `pi05_as_writer_v52_v6_recipe_video_causality_audit_seed7_20260802`，逐行重验四个
  architecture×recipe cell的五臂、source base配对、16-row内部传递、matched
  exposure和v6参数/Adam动力学。analysis SHA256为
  `98371337e2cf1f7cec09d04e81445b419fc21c654fe173cb081a4b5e63092efa`，canonical
  payload为`6d9262f8595bfcd3eeb93c3a4808091386cd29c29ef37c8abf0fc374d8b421cd`。
- 初版验证器错误地要求实际消耗的policy-noise列表全等；成功早停会截短该列表。
  已删除并重建本轮刚生成、完全可再生的错误两文件root；修正版验证同一seed schedule
  的公共前缀，四cell、source和全部五臂均为400/400一致。
- 审计确认task-complete在v5.2/v6上都压弱Procedure→LoRA/action和顺序behavior
  margin，但correct absolute分别下降/上升；old六倍Adam路径也同时放大视频写出与
  task旋转。因此后续整体设计必须联合处理semantic carrier、causal write gain、
  single-video噪声和optimizer clock，不能单边归咎架构或full24平均。
- 追加逐task transfer审计和完整checkpoint-curve/source-retention审计；analysis
  SHA为`611c9330...c5a1`与`bf5a4609...1770`。matched150显示v5.2 recipe effect由
  source retention -10与new gains -71构成，v6则为-1/+17；selected v6 +22又被
  Object task3单项+24主导。由此将“v6与task-complete相容”限定为早期较广语义
  acquisition，不误写成漂移或视频因果已经解决。

## 2026-08-02 CV-ADR一小时结果、exact50与第二小时启动

- RAW macro0→200自然完成：200 macros、96,000 queries、4,800 teacher videos、
  wall`3916.79s`，全部finite、0 clip、信息墙读取0。paired correct400
  50/100/150/200=`76/111/99/117`；macro200为右端best但仍有明显task churn。
- candidate curve analysis SHA256为`71cbd02d...33dc3`（canonical payload
  `b6684ccd...41a34`）。同RAW architecture audit SHA256为
  `a4b1b03f...8c47`；UCP三recipe source-preservation audit为
  `b43fe53f...01ff`。两个审计都严格核验曝光与evaluation pairing。
- exact50完成8 tasks×50 videos×五种真实frame order、400 Cartesian rows、0
  rollout/0 failure，全部信息墙和checkpoint不改写合同通过。raw analysis、summary
  SHA为`b60f6ed4...0407`/`5f1df44e...4a56`；compact responsibility audit为
  `2aec024f...9cb9`。结果同时确认Core/Program必要和Action/order reader仍弱。
- 根据预封存一小时门，已从同一frozen `254ade4` step200 exact-resume到400。
  活动tmux为`ember-cvadr-raw-formal-resume200to400-254ade4`，沿用正式root；log为
  `/data/ymdai/logs/ember/pi05_as_writer_cvadr_rawfull24_taskcomplete_decay400_formal_dev_r4_b20_seed7_254ade4_20260802_retry1_resume200to400.log`。
  首两个恢复宏步201/202各覆盖24 tasks/24 videos/480 queries，loss约`.1010/.1014`、
  gradient`.1040/.1253`、0 clip；四rank只在GPU4--7、cost-balanced long-first，
  五个参数块Adam step均精确续到201/202。完成后评测250/300/350/400。

## 2026-08-02 CV-ADR B20/profile/resume seal

- 从clean detached `ff57a9f`只用GPU4--7完成teacher-seed172三macro profile；
  首macro含`task38/demo36=105` sampled frames，每宏步24 tasks/24 videos/480 queries，
  三步wall `20.698/18.777/18.737s`，峰值allocated/reserved
  `77,227,462,656/83,523,272,704` bytes，无OOM、非有限或信息墙读取。
- 另用canonical formal seed fresh0→1并exact-resume1→3。step1七文件bitwise、size和
  纳秒mtime均未改变；3行metrics、2次invocation、scheduler/cursor连续，step1起全部
  五个主块梯度可达。profile/resume metrics SHA为
  `9a3b490c...f0b11`/`55366fc4...94028`。
- RAW config已seal为ready。下一步从本次post-seal clean detached commit fresh启动
  macro0→200、every25 checkpoint；profile/smoke权重不进入科学轨迹。
- `50ac8ee`首次formal启动在output root创建前由fail-close拒绝：描述性status不等于
  runtime要求的canonical `sealed`。无metrics/checkpoint/科学数据；失败log保留，
  config与测试原位修正后使用全新retry root。
- 修复commit `254ade4`已push；clean detached retry从fresh identity成功进入formal
  0→200。首macro 24 unique tasks/24 videos/480 queries，rank cost
  `207/216/206/204`且strict long-first，formal LR/loss/gradient finite、0 clip。
- step25 checkpoint与512-row held functional panel已完成；CV held`.13332`，同recipe
  RNG-v2 UCP held`.13181`。训练继续运行，correct400等macro200释放GPU后启动。

## 2026-08-02 UCP operator格闭环并集成CV-ADR

- normalized GROUP4自然完成1200 updates/200 cycles、96,000 queries、4,800
  videos、wall`4906.16s`，16次clip、all finite。四个paired correct400为
  `77/76/66/100`；与RAW `72/87/86/89`联合裁决后，行为门false，不做五臂。
- 封存operator audit：GROUP4 drift envelope gap34低于RAW60，但AUC、task support
  和winner集中度更差，不能迁移为默认。analysis SHA256 `97c70dd...a6e0`。
- 顺序完成GROUP4与RAW exact50，每臂8 tasks×50 videos、五conditions、400 rows、
  zero rollout。RAW/GROUP4 analysis SHA为`9704b9fd...4067`/
  `57760475...c52`，paired audit SHA为`7201364a...11fd`；panel SHA
  `222f4e7d...9394`。结果确认GROUP4把A/D→BA压到RAW约22.5%，且唯一action异常
  是0-success task。
- merge `b97960f`把CV-ADR设为唯一canonical实现，旧UCP可执行config和专用analyzer
  退役。参数`10,241,024`；全仓`226 passed in 30.51s`，compileall、diff check和
  四config loader通过。当前无活动训练/评测/tmux；下一步为CV最长105-frame B20
  profile和formal-seed exact resume，后续不使用subagent。

## 2026-08-02 RAW随机训练噪声审计与GROUP4中段健康

- 完成UCP RAW RNG-v1/v2只读paired审计：除CPU Beta flow timestep identity外，
  topology、optimizer、schedule、task/video/query和eval panel均匹配；correct400
  曲线差`-17/+16/+4/-28`，matched per-task梯度草图余弦中位仅`.163--.193`。
  结果证明optimizer basin对训练噪声实现敏感，但不是seed-general估计，也不恢复
  v1跨operator效力。analysis SHA256为`ff6acdf8...b82`。
- GROUP4在一份live snapshot已到physical update271/45 complete cycles；step150
  checkpoint和512-row held validation完整，loss/gradient/LR finite。累计13次clip
  （4.8%，最小coefficient`.653`），故最终裁决会把clip、task、phase与cost共同纳入
  完整operator bundle，不把GROUP4冒充单一Adam/relinearization因果量。

## 2026-08-02 RNG-v2 RAW正式负结果与GROUP4启动

- RAW自然完成200 cycles、96,000 queries、4,800 videos、wall`3878.963s`；全部
  finite、0 clip、信息墙读取正确。四个formal correct400均400 rows/36 shards/0
  failures并通过严格pairing/cache/checkpoint重验，50/100/150/200=`72/87/86/89`。
- winner macro200 breadth6、>=5 breadth4、top2占60.67%；相邻churn
  `45/30`、`28/29`、`27/24`，Jaccard`.359/.504/.549`，四点union149相对best89
  的gap为60。train loss下降、held不动、BA norm上升，故不resume、不做五臂。
  candidate analysis SHA256为`0f8545b1...3462`。
- 旧`55faeeb`GROUP4 fresh尝试被exact-origin guard在初始化前拒绝，因为RAW launch
  后main只增加了两个文档commit；失败log保留、无checkpoint/科学数据。验证
  `55faeeb→8dfe6ed`的`src/scripts/configs` tree逐项相同后，建立clean detached
  `8dfe6ed` frozen worktree并从新root启动GROUP4 update0→1200。
- 活动tmux为`ember-ucp-rngv2-g4-tf400-8dfe6ed`，只用物理GPU4--7一卡一rank；
  首5个完整cycles逐次覆盖24 unique tasks、24 videos、480 queries，scheduler只在
  phase5推进，all finite、0 clip、信息墙正确。完成后固定评测physical300/600/900/
  1200并执行预注册operator裁决。CV-ADR继续隔离，后续不使用subagent。

## 2026-08-02 RNG-v2 RAW fresh正式启动

- seal authority `55faeeb`已push，并建立clean detached frozen worktree。一次live
  preflight确认GPU4--7空闲健康、个人占用`399,218,013,812` bytes、全部资产存在、
  output/tmux无冲突；未查询GPU0--3。
- RNG-v2 RAW已从fresh identity启动macro0→200；tmux
  `ember-ucp-rngv2-raw-tf400-55faeeb`，root/log与精确命令记录在active handoff。
  四rank各占物理GPU4--7一张，没有额外CUDA role。
- 首macro完整覆盖24 train tasks一次，24 teacher videos/480 queries；四rank sampled-
  frame总和`207/216/206/204`且rank内严格long-first。loss/gradient/LR finite，
  step2起Semantic Frontend、Program、compiler、factor全部非零可达；信息墙与source
  freeze正确，0 OOM/NaN/contract mismatch。
- 让RAW自然完成后固定评测50/100/150/200，再运行GROUP4；当前不并发第二个GPU
  workload。CV-ADR继续隔离，后续不使用subagent。
- 在任何RNG-v2 correct400 outcome出现前，已把RAW×GROUP4裁决冻结为cycle200
  endpoint、四点cycle-AUC、single-best breadth、相邻success churn/envelope gap、
  逐task phase/cost混杂和A/D→BA→action传递的联合证据；含混时不把6倍AdamW时钟与
  length curriculum迁入CV。functional loss或机械25% selected4 energy不能改写裁决。

## 2026-08-01 task/query RNG-v2重封存完成

- `dae13bf`已实现并push CPU+CUDA共同按task/query seed的随机scope；完整CPU回归
  `241 passed`，JSON和compileall通过。两份v2 config/checkpoint family与state
  schemas均fresh incompatible，旧v1 checkpoint继续fail-closed。
- detached frozen `dae13bf`上仅用物理GPU4--7完成RAW B20 fresh0→1→resume1→3和
  GROUP4 B20 fresh0→1→3→7。all finite、主要模块梯度可达、信息墙读取0，完整cycle/
  scheduler/cursor与checkpoint不改写合同均通过。
- tasks12/14/34/37在两operator间跨rank重排后，四个functional losses和raw task
  gradient norms逐位相等；CountSketch最大绝对差`5.82e-11`。两份formal config已
  重新seal为fresh RAW 0→200与GROUP4 0→1200。
- 当前无活动训练、评测或tmux。下一动作是提交/push reseal authority，建立新clean
  formal frozen worktree，live preflight后从全新root启动RNG-v2 RAW；随后GROUP4与
  paired correct400。CV-ADR继续隔离；后续不使用subagent。

## 2026-08-01 task/query RNG-v2纠偏与旧GROUP4停止

- 审计policy forward随机源后确认RNG-v1遗漏CPU Beta flow timestep：CUDA Gaussian
  noise按query固定，time仍跟随rank-local ambient CPU stream。step0 identity的四个
  跨rank重叠task在rows/video/seed完全一致时loss显著不同，根因已由实现与观测双证。
- 对唯一属于本任务的tmux发送Ctrl-C，GROUP4正常停止在physical step307、51 complete
  cycles；本训练进程/tmux均退出。root、307行metrics、step150/300 checkpoints保留
  为invalid-contract provenance，无summary，不resume、不评测。
- main正在实施CPU+CUDA RNG-v2：同一scope共同fork/seed/restore，升级randomness
  scheme、cycle-normalized config与task-query checkpoint schemas。两份formal config
  已fail-closed为pending reseal；旧v1 checkpoint不能被新family接受。
- 定向CPU回归当前`19 passed`，新增无需物理GPU的CUDA-branch测试同时验证CPU time、
  CUDA noise可重复且外层两个generator均恢复。下一步是全回归、commit/push，再用
  新frozen worktree在GPU4--7完成真实manipulation/resume并fresh重跑两臂。
- 既有RAW autoscaled-vs-true scheduler pair因rank/task/microtask顺序相同仍可作为
  同一ambient stream下的匹配scheduler比较；其absolute只保留为RNG-v1 bundle结果。
  CV-ADR继续隔离等待有效operator裁决；后续不使用subagent。

## 2026-08-01 UCP true-fast400裁决与normalized-group4正式运行（RNG-v1快照）

- clean frozen `cfc2ad1`的task/query-keyed UCP raw已按真实
  `warmup17 + decay400`从fresh identity完成前200/400 cycles：96,000 queries、
  4,800 one-video conditions、wall `3884.255s`，所有训练量finite且信息墙读取0。
  macro50/100/150/200 paired correct400为`89/71/82/117`；winner macro200
  breadth7、仅4 tasks达到至少5次成功、top2占`62.39%`，未达到125强五臂门，
  因此不做same/wrong/shuffled/reversed。
- 同一输入的autoscaled-decay200与true-fast400在相同步的严格paired差为
  `+8/-1/-25/+39`。最接近峰值的autoscaled macro150=107与true macro200=117
  仍有26/36个各自独有成功状态，Jaccard仅`.5664`；慢日程主要把能力峰推迟并
  重新分配task，没有提高single-checkpoint ceiling或解决漂移。
- 参数与optimizer审计证明两条scheduler不是沿同一直线走不同距离：跨日程Writer
  delta norm由step25的`1.571`增到step200的`4.370`，Adam一阶moment cosine由
  `.597`降到`.354`；true-fast相邻25-cycle位移方向cosine末段降到
  `.121/.077/.031`。full24 mean晚期只保留约`4.72%` task-gradient energy，
  但raw candidate对task平均为负的比例很低，主要矛盾是近正交conditional
  innovation没有稳定共同方向，而非CP式投影可修复的多数task直接受害。
- candidate与scheduler交互analysis SHA分别为`7b7d9822...dd3`与
  `81eca3cc...ab7e`。四条评测逐row state/video/env/policy RNG prefix严格匹配；
  static evaluator contract SHA为`6e0b8b2d...be387`。
- 预注册的cycle-normalized randomized-group4已经从同一clean frozen `cfc2ad1`
  fresh正式启动。root为
  `/data/ymdai/outputs/ember/pi05_as_writer_ucp_taskquery_cycle_normalized_group4_truefast400_formal_dev_r4_b20_seed7_cfc2ad1_20260801`，
  tmux为`ember-ucp-tq-g4-tf400-cfc2ad1`。首个六phase cycle与raw cycle0逐task使用
  完全相同的teacher demo和sampled-frame count，恰好24 tasks/24 videos/480
  queries；scheduler只在cycle边界推进，step2起全部主模块梯度finite，0 OOM/clip。
  训练完成后固定评测cycle50/100/150/200，不根据train/held loss提前选点。
- CV-ADR保持隔离在clean branch/worktree，RNG-v2 rebase/verification snapshot为
  `ed21244`；在group4裁决前不集成、不启动GPU。后续全部由主进程执行，暂停
  subagent使用。

## 2026-08-01 UCP scheduler合同纠偏与CV-ADR实现冻结

- scheduler-total修复`e1299db`已push。其首次true-fast400 launch在创建output root
  前被formal runtime fail-closed：stage stops仍是旧`[200]/[1200]`而没有覆盖
  total400/2400；0 video/query、0 checkpoint。现已修正为
  `[200,400]/[1200,2400]`并把完整stage合同移入loader，直接formal runtime回归后
  `25 passed`。下一步从新commit和全新root fresh启动。
- frozen `1a09e71`的task-query raw control已fresh启动并持续finite；root为
  `/data/ymdai/outputs/ember/pi05_as_writer_ucp_taskquery_rawfull24_decay400_control_formal_dev_r4_b20_seed7_1a09e71_20260801`。
- live LR审计发现formal total200触发LeRobot把配置的decay400自动压到200；
  macro150实测LR`5.4093e-5`而未压缩合同应约`2.1049e-4`。该run自然完成后只作
  autoscaled200 scheduler ablation，原group4不启动。
- main已在最窄config边界把raw/group4 formal total改为400/2400，并加入loader
  fail-close，禁止formal逻辑总步少于decay；`tests/test_writer_training.py`与
  `tests/test_writer_serial4.py`共`24 passed`。待当前run和其paired correct400完成
  后提交/push，再用新frozen authority fresh跑真正fast400两臂。
- 独立CV-ADR worktree已提交：真实参数`10,241,024`，四config保持
  profile pending，focused `159 passed in 26.32s`，compileall/diff/config loader
  与architecture guard通过；实现随后rebase到当前`8dfe6ed` authority，并在
  `ed21244` snapshot上完成全仓`226 passed in 35.56s`、compileall、四config
  loader及diff check。没有推送，也没有启动GPU
  profile/formal。
- autoscaled-decay200 raw已自然完成macro200：wall `3892.039s`、96,000 queries、
  4,800 videos、200行finite、信息墙读取0，四候选checkpoint完整。四条correct400
  已在GPU4/5/6/7全部自然完成；每条400 states、36/36 long-first shards、0 failure。
  macro50/100/150/200为`81/72/107/78`，macro150 breadth8但随后lost43/gained14，
  所以不续训、不做五臂。
- 训练内部量确认LR缩短压低位移但没有消除方向轮换：三段参数长度为
  `2.348/1.084/.349`，macro200 raw-mean task-gradient energy retention为`4.22%`，
  同task相邻条件梯度的晚期CountSketch cosine仅`.009-.017`。这些量将在真正
  task/query-keyed fast400 raw完成后做严格scheduler配对，当前不与旧ambient-RNG
  UCP混作单因结论。
- candidate analyzer先后对新增`selected_task_count=24`和overlay config解析
  fail-closed，兼容层逐项验证后完整通过；analysis SHA为`bfd580d4...0993`。三次
  checkpoint转移gained/lost=`28/37,54/19,14/43`，Jaccard=
  `.4037/.4206/.5289`，BA mean norm=`45.34/50.00/52.94/51.92`。

## 2026-08-01 UCP控制seal与canonical恢复

- group4 formal-seed exact-resume root
  `/data/ymdai/outputs/ember/pi05_as_writer_ucp_group4_formalseed_resume_smoke_r4_wip_20260801_retry1`
  已完成0→1→3→7；raw root
  `/data/ymdai/outputs/ember/pi05_as_writer_ucp_rawfull24_formalseed_resume_smoke_r4_wip_20260801`
  已完成0→1→3。两者checkpoint均未被resume改写，cycle0 teacher-video mapping逐项
  相同，所有loss/gradient/LR/cursor finite且validation/test reads为0。
- group4与raw config已分别seal为fresh update0→1200和macro0→200、every150/25。
  UCP精确参数`7,683,328`。封存实现commit为`b52cb54`，canonical restoration
  `85a82cb`逐blob复原同一运行面并退役AP/endpoint runner。
- 聚焦13个测试文件共`107 passed in 17.30s`；compileall、JSON、diff check通过。
  architecture guard的hard项均来自封存UCP既有owner相对AP树重新出现，active source
  实际净减1,061行。为保持正式控制与profile/resume精确一致，本轮不做launch前
  重构；该例外及CV-ADR原位替换/不复制路径的移除合同已写入authority。
- 当前没有需要继承的训练、评测或tmux进程。下一动作是把本authority clean push，
  建立detached frozen worktree，只做一次GPU4–7与storage live preflight后依次启动
  fresh raw与group4正式训练；后续所有工作由主进程亲自完成，不使用subagent。

## 2026-08-01 Endpoint10负裁决与UCP normalized-group4 live profile

- endpoint10 portable historical overlay在首个formal尝试暴露了v6-old历史config
  provenance变化；该次在任何数值row生成前fail-close、未创建科学root。`0f92e35`
  将overlay绑定portable provenance并完成profile，随后用全新retry1 root正式运行。
- 四rank formal自然完成18 candidates×512 held rows=9,216 rows，wall
  `1041.474s`，environment未构造、parameter gradient未计算、validation/test action
  读取0。root为
  `/data/ymdai/outputs/ember/pi05_endpoint10_formal_18candidate_seed7_0f92e35_20260801_retry1`；
  run contract/rows/summary SHA为`edb7d3c...583b`、`7087999d...bd0`、
  `a4a489a3...c2ba`。
- 预注册关联analysis SHA为`d54435fe...f707`。global Spearman `.258398`、固定
  100,000次permutation `p=.298447`，all gate失败；family、recipe direction和
  per-task门通过但不得覆盖global失败。endpoint10封存为no-gradient负诊断，不进入
  checkpoint选择、loss或训练。
- 在隔离UCP worktree完成fresh raw-full24/cycle-normalized randomized-group4受控
  实现：task/query-keyed stateless policy RNG、随机Latin六phase、LR/beta/decay
  exposure composition、cycle-boundary scheduler、fresh checkpoint family、midcycle
  cursor和per-block optimizer diagnostics；最新聚焦测试`31 passed`。
- group4 longseed172四卡B20 profile自然完成18 updates/3完整cycles，wall
  `72.455s`；每cycle 24 tasks恰好一次，最大真实teacher video 105 sampled frames，
  1,440 queries/72 video conditions，step2起所有主块梯度finite可达。峰值
  allocated/reserved `76,971,835,904/83,647,004,672` bytes；formal teacher seed已
  恢复为`20260722`，正在核验0→1→3→7 exact-resume。

## 2026-08-01 AP-ADR正式裁决与endpoint10启动

- AP-ADR四个paired correct400已完成：macro50/100/150/200为`91/81/94/91`，
  breadth`6/6/5/7`，winner macro150逐task`[18,1,0,37,29,9,0,0]`。四点无可信
  上升趋势且持续gained/lost，一小时门失败；未resume到400，未启动五臂。
- 定位并修复PI05 recursive sampler永久切换attention backend造成的内部分析污染。
  `5d93af3`已push，定向`22 passed`；修复后macro150 refs1在8 tasks上逐层、
  effective BA、fixed action严格零误差重放，checkpoint文件均未变化。
- 有效AP内部root的analysis/summary SHA为`d42fc4eb...bc2b`/
  `f2c572c5...e682`。same-task上游Program变化`.919-1.105`到reader只剩`.0321`；
  temporal key反转对BA仅`.000521`。Effect-only距full BA`.00821`，A-only/D-only
  约`.276/.283`，固定full key仍相同，故中央失败是key-only contextual Program
  配raw Effect-dominant value，不是whole Program starvation。
- endpoint10三组历史portable cache已从clean frozen extension commits用四rank自然
  完成：v5.2-old 64、v6-fast 8×64、v6-old 64，共640套public LoRA；wall分别
  `9.874/77.036/9.949s`，environment/action/test-video读取全0，tmux均已退出。
  v5.2-old/v6-old manifest file SHA为`ab158969...9de1`/`988ef3ee...4398`；v6-fast
  八点为`14086ba7...fbee`、`488989b2...7436`、`5dfb854d...b1fb`、
  `1d86b51f...492b`、`44367a8a...26f8`、`a53057ed...e989`、
  `ea47d859...564b`、`db47ab99...fd0a`。
- 在任何endpoint数值生成前修正候选表的文字错误：v5.2-new正式、paired
  correct400候选是macro150/200/350/400，不是50/100/150/200；候选总数仍为18，
  recipe方向仍比较macro150，因此不构成outcome selection。下一步是真实CUDA
  profile/parity与18-candidate formal no-gradient诊断。

## 2026-08-01 AP-ADR live seal与正式首小时

- UCP raw macro150 exact50已补齐，与SERIAL step900构成严格150-exposure内部
  对照。raw→SERIAL的`x_only`相对full effective-BA/action变化从
  `.0653/.01269`升到`.4184/.12999`；same-video centered variance/sample
  energy从`.1096%/.03230%`升到`.4865%/.7322%`。训练更新粒度因此被证明会
  决定视频动态信号能否穿过compiler到policy action；但四点同曝光correct差值
  `+7/-17/+21/-3`且漂移仍在，SERIAL不升级为默认recipe。
- matched audit root已生成并核验，analysis/evidence/summary SHA为
  `e8cdbc79...c922`/`b44cdec0...859`/`dbc660cd...88d`。严格配对行为为
  `100->121`、`50 gained/29 lost`、breadth `5->5`、Jaccard `.4733`；action
  方差均值由单一Object-3离群值主导，后续归因使用逐task和中位数。
- AP-ADR canonical实现commit `8306549`已完成CPU回归与真实GPU vertical path。
  exact module count为`10,241,024`；105-frame B20三macro profile覆盖72套视频、
  1,440 queries，三步finite且所有五个主块在identity step后可达。峰值
  allocated/reserved为`77,227,462,656/83,523,272,704` bytes。
- formal seed root先fresh0→1，再exact-resume1→3；三行loss/grad/LR/cursor连续，
  step1 manifest、Writer、trainer和四rank state的size/mtime/SHA均未改变。
  seal commit `7dffb6f`已push。
- clean detached `7dffb6f`已从fresh identity启动AP-ADR raw-full24/fast首小时
  macro0→200；frozen worktree、root、log和tmux分别为
  `/data/ymdai/.codex/worktrees/EMBER-ap-adr-formal-7dffb6f-20260801`、
  `/data/ymdai/outputs/ember/pi05_as_writer_ap_adr_rawfull24_decay400_formal_dev_r4_b20_seed7_7dffb6f_20260801`、
  `/data/ymdai/logs/ember/pi05_as_writer_ap_adr_rawfull24_decay400_formal_dev_r4_b20_seed7_7dffb6f_20260801.log`、
  `ember-ap-adr-formal-7dffb6f`。正式run不从profile/resume warm-start。
- 该fresh run已自然完成macro200并退出：96,000 queries、4,800 one-video
  conditions、200 cycles、wall `3898.217s`，所有metrics/checkpoints/信息墙计数
  完整。macro50/100/150/200 paired correct400已分别在GPU4/5/6/7启动，tmux
  `ember-ap-adr-correct400-7dffb6f`；四卡各有6个Writer generator并已生成prepared
  contract，teacher action reads=0。
- AP macro175与UCP raw macro175同曝光module-dynamics只读审计完成。AP Program
  raw gradient虽仅为UCP约`.856%`，Adam update/cumulative displacement仍达
  `71.18%/85.42%`，whole-block starvation被排除；剩余最窄风险是eps-limited
  temporal Q/K routing。analysis/summary SHA为`1ee02ff2...5a0`/`c4c79189...1fc`。
- post-v5×recipe审计完成并复核30项primary evidence SHA。八个formal架构共享完全
  相同的full24/B20/fast400 step/sampler/launcher blobs，无一拥有matched alternate
  recipe；历史long-first不是optimizer phase。封存analysis/summary SHA为
  `a53f528c...b229`/`f481f37e...2442`，后续只做update-operator replay、证据门后的
  normalized randomized group4和条件化8-Action-anchor transplant，不整套重跑旧架构。
- endpoint10 no-gradient held诊断已以`544c0ef`/`2055a82`集成main；职责拆为
  candidate/pairing、historical provenance、result aggregation和唯一runtime，
  未分叉Writer/evaluator。定向`52 passed`、隐藏全部CUDA的全仓`222 passed`；
  真实CUDA sampler parity、portable-v2 cache生成和四rank formal run仍待AP首小时
  训练与correct400评测释放GPU4--7后执行。
- 在看到任何endpoint结果前已预注册唯一primary为executed-first5 valid normalized
  action `quality=-MSE`；18-candidate全局/within-family/task级关联门和v5.2、v6
  两个等曝光recipe方向均已写入AP authority。secondary metrics不得救回失败主门。

## 2026-08-01 SPG门失败、内部根因与UCP设计

- 完成跨v5.2/SPG/UCP/v6新旧recipe的strict functional-surrogate审计：七条曲线
  共用同一512-row held manifest（SHA256 `53cbf9e...a3a8`）。主20点held→correct
  在按架构去均值后反而为Pearson/Spearman `+.462/+.644`，相邻checkpoint的
  Δheld→Δcorrect仅`+.120`；逐task去均值后held→success仅`-.055`。因此held
  loss降级为finite/局部拟合诊断，candidate选择继续只认paired correct400、
  breadth、gained/lost/Jaccard和effective BA/action传递。
- surrogate审计正式root为
  `/data/ymdai/outputs/ember/pi05_as_writer_functional_surrogate_closedloop_audit_seed7_20260801`；
  analysis SHA `91eaabed...12a`，120个输入SHA及44个strict paired panels复验通过。
  post-v5能力审计analysis SHA为`406b9098...80e`：扩展24 candidates的四架构
  envelope union/intersection为`246/110`，全部checkpoint intersection仅5；低分
  架构仍保存独有能力，不能整体判死。
- serial-4 formal已自然完成step1200：1,200行finite metrics、8个checkpoint、
  96,000 queries、4,800 videos、200 task cycles，wall `4197.076s`，信息墙读取0。
  held loss八点为
  `.132407/.131304/.133484/.132973/.130352/.132508/.132237/.132918`，不作行为
  预选。4,800条raw→serial exposure逐项匹配，replay SHA为`d406f2f1...80cc`。
- step300/600/900/1200 paired correct400已分别启动在GPU4/5/6/7；四个launcher和
  24个Writer generation workers存活，prepared contract均为400 states、36个
  long-first shards、6 replicas/6 generators且teacher action reads=0。tmux为
  `ember-ucp-serial4-correct400-3db82df`。
- 完成architecture×training mechanics只读审计并封存167个输入SHA；analysis
  SHA为`c910a933...e521`。v6 old/new-slow对齐B20、video/query与exposure LR后，
  一阶LR integral仍差`6.0069×`，visits100→150参数路径cosine仅`.0493`、Adam
  exp_avg cosine`.0331`；更新粒度同时改变累计步长、重线性化、moment、clip与WD，
  不能再缩写成“full24梯度抵消”。
- UCP exact50在clean frozen `c4b85e8`自然完成：8 tasks×50 references共400
  rows、四rank各100、reference0..49完整、0 rollouts、无failure。pooled
  same-task effective-BA/fixed-action centered variance/sample energy为
  `.09008%/.01656%`；Program的same/wrong/shuffled/reversed差异仍明显，到BA/action
  后大幅压弱。analysis/summary SHA为`a6e40cd6...25a8`/`386a04f5...acaa`。
- observed-best与匹配150次video exposure的v5.2×v6审计均确认强
  architecture×recipe交互；匹配点recipe effect为`-81/+16`、描述性DiD=`97`。
  结论只覆盖训练bundle，不把v7以后与fast recipe混杂的思想整体判死。
- serial-4的long-first optimizer curriculum已量化：4,800 visits中phase与sampled
  frames Pearson=`-.8331`，task mean相关=`-.8734`，task38始终phase0。
- clean detached main `10a71a1`的最长seed172 B20 profile已自然完成：18 updates、
  3个完整cycles、每cycle 24 unique tasks、1,440 queries/72 videos，真实105-frame
  视频进入首update；峰值allocated/reserved为`76,971,835,904/83,647,004,672`
  bytes，全部finite，step2起四个主模块梯度可达。
- formal seed fresh0→1→resume1→3→resume3→7通过；step1与step3全部checkpoint
  文件在后续resume后SHA不变，cycle/phase为`0:0..5,1:0`，step6才推进scheduler，
  step7 LR从`.0003`变为`.0002275`。canonical serial config已seal，formal必须从
  新commit fresh identity启动1,200 updates，不能续接smoke。
- seal commit `3db82df`已push，clean frozen worktree已fresh启动serial-4 formal
  update0→1200；tmux `ember-ucp-serial4-3db82df`，root为
  `/data/ymdai/outputs/ember/pi05_as_writer_ucp_serial4_exposurematched_decay400_formal_dev_r4_b20_seed7_3db82df_20260801`。
  首cycle六phase覆盖24 unique tasks；rank内frames单调long-first，四rank总frames
  `207/216/206/204`；step2起四主块梯度可达、全部finite、信息墙读取计数0。
- UCP raw-full24 frozen `c94f1c6` fresh macro0→200自然完成，96,000 queries、
  4,800 videos和全部25-step checkpoints完整；四候选paired correct400为
  `82/117/100/110`。union169远高于single best117，breadth与成功集合持续轮换；
  一小时门失败，不resume到400、不跑五臂。
- candidate curve和drift analyzer均fail-close通过。train loss持续下降而held
  `.131–.132`不改善；四候选raw mean只保留约`4.06–5.64%`平均task-gradient
  energy，支持但尚不独证“full24抹掉task innovation”。
- UCP内部analyzer在main提交`7385ff3/4837673/0ab3212/2d4b03c/a4b06f5`完成并push。
  初次CUDA refs1揭示B5 canonical与B1 recompute的shape-dependent BF16数值漂移；
  修复后所有counterfactual保留B5 carrier、只改/抽row0，原阈值不放宽，真实refs1
  的Program→coordinates→factor→A/B→BA→action重算误差全部0。
- refs1确认reader target/rank routing健康，wrong/order变化可传到BA/action；但固定
  X只换A/D时BA约`1.4–2.4%`、action约`.5–.9%`，dynamic教学弱。
- 首次exact50在rank1本地异常后进入旧NCCL gather并被600秒watchdog掩盖；其余
  ranks仍在同一local compute阶段，schedule/长度审计排除正常长尾和collective
  错序。失败root只含run contract，没有科学rows，四个本任务进程均已退出。
- `874e5f1`实现reference级上下文与failure JSON、直接re-raise fail-fast，以及
  analysis-only两小时Gloo控制组；训练/provenance protected owners未改。21个定向
  测试通过，architecture guard无hard violation。
- 新refs2在`libero_spatial task3/reference1`精确复现rank-gauge sanity失败，
  failure JSON保留原trace且torchrun立即终止其余ranks，验证fail-fast合同。旧错误
  经`e47ffe8`判别为raw A/B确实改变`.74184/.13602`且effective BA保持到
  `1.299e-9`；唯一超阈的是bf16 factorized policy action `.002047`，其rank求和
  顺序随置换改变。修复保留finite和BA `2e-5`硬门、记录action execution drift，
  没有放宽数学函数合同；20项相关CPU测试通过，architecture guard无hard violation。
- clean `c4b85e8`新refs2已自然完成：8 validation tasks×2 references共16 rows、
  四rank各4 rows，无failure artifact；analysis/summary SHA为
  `e0757f55...cc48`/`c7a42eae...da41`。同一frozen commit随后启动exact50，
  run contract封存400 references、4 ranks、0 rollouts及protected-path provenance。
- serial-4已由独立write worktree集成到main `ccdf21f/92548ed`：六phase精确重建
  同一full24 exposure，LR按cycle重复六次，fresh config/checkpoint/rank schemas和
  midcycle cursor完整；formal边界强制整除6，profile/formal video seed分别明确为
  `172/20260722`。全仓`233 passed`、compile/diff通过，architecture guard无hard
  violation；尚未GPU profile、resume seal或formal launch。
- UCP live seal完成：detached `0d4c271`上最长105-frame、B20、四rank三macro
  连续通过，step wall `20.394/18.494/18.504s`，峰值allocated/reserved
  `77,127,082,496/83,345,014,784` bytes；72个视频条件、1,440 queries和全部
  checkpoint finite，step2起frontend/Program/reader/factor梯度均可达。
- canonical formal seed fresh0→1→exact-resume1→3通过；step1七个文件SHA不变，
  metrics/LR/task-video-query/RNG cursor连续，10个gradient chunks逐项
  gather/completion/CUDA sync。config已seal B20；下一步从新clean commit fresh
  启动macro0→200，不从smoke续训。
- UCP canonical CPU实现完成：`[X_f,A_f,G_(f+1)-G_f]`两层causal axial Program，
  单级normalized target/rank raw-value reader，无独立Core add和global mixer。
  semantic frontend/Program/reader/factor参数分别为
  `3,453,440/1,838,592/212,224/2,179,072`，总计`7,683,328`。
- CP active path已删除；每macro仍逐task求梯度并严格组成raw full24等权mean，
  overall和semantic_frontend/program/compiler/factor Gram仅作诊断。B20每visit覆盖
  全20个normalized-progress strata，episode边缘仍由permutation+jitter保持均匀。
- fresh config/checkpoint/evaluation schema均不兼容旧SPG；step0 identity、causal
  prefix、outgoing alignment、target/rank置换、padding、raw gradient world1/2和
  sampler exact resume通过；`CUDA_VISIBLE_DEVICES=''`全仓`203 passed`。
- 当前没有活动训练、评测或tmux；下一步提交现场seal并从新frozen commit只在
  GPU4–7启动formal macro0→200。

- SPG fresh macro0→200完成，macro50/100/150/200 correct400为
  `97/115/77/100`；按一小时门停止，不resume、不跑五臂。
- 四候选exact50几何、candidate gained/lost、Gradient Gram/energy和macro100
  refs2内部反事实完成。Program对order和wrong-video有强差异，但CoreReader与
  ProgramReader几乎不区分target/rank，最终LoRA近严格rank1/B-column相同；最早
  失效接口锁定为compiler，而非evidence extraction。
- 审计v5.2/v6 old/new及v6 slow/fast后，降低“full24或scheduler单独解释一切”
  的置信度；CP能消负pair但不能恢复近正交task innovations，functional loss与
  closed-loop仍错位。
- 完成现有B20前200 macros的96,000-query phase audit；长期无偏但单task visit
  覆盖方差可观，支持无偏20-strata estimator作为训练联合设计，而非新监督。
- 新write worktree
  `/data/ymdai/.codex/worktrees/EMBER-unified-program-534064a-20260801`已从clean
  `534064a`创建；设计authority为
  `docs/action_forecast_writer_unified_causal_program_design.md`。

## 2026-08-01 v5.2正式评测、五臂与内部分析封存

- paired correct400候选macro150/200/350/400完成，为`51/91/106/120`；选择
  single-checkpoint macro400，不做checkpoint融合。
- macro400正式五臂完成：`120/109/107/111/124`。逐task、逐suite、gained/lost
  state和严格pairing审计封存；四个控制臂都没有证明correct的行为优势。
- exact50 LoRA几何与五条件Core/Procedure/BA/fixed-action反事实完成；数值顺序
  信号可下传，但same-task视频方差缩至sample energy的`.6844%`且方向未与行为
  收益对齐。v5.2新recipe cell因此完成并停止，不再训练或评测。

## 2026-08-01 SPG最长profile与CP通信修复

- 初始最长profile macro1完成后，macro2在CP Gram交换处stall；只终止本任务
  tmux，未触碰任何外部进程。最小NCCL/Gram probe健康，逐phase trace把故障定位
  为分块all-gather仅入队、缺少逐chunk CUDA completion boundary。
- canonical CP实现加入每CUDA Gram chunk的stream completion，并记录
  all-gather/sync计数；CPU/Gloo路径保持0 sync。修复后原始105-frame/B20 profile
  三macro连续通过，step wall `20.536/18.578/18.546s`，峰值reserved
  `83,529,556,160` bytes。
- 72个单视频条件、1,440 queries全finite；每步24 tasks唯一且long-first，
  macro2起所有五个主块梯度可达。
- clean `f6d4876`上的formal-seed fresh0→1→exact-resume1→3已完成；step1六个
  状态文件与manifest在resume后哈希不变，metrics三行连续，72 videos/1,440
  queries、LR、task/video/RNG cursor和信息墙均核验。下一步提交seal并从最终
  frozen commit fresh启动macro0→200，不从profile/smoke warm-start。
- resume seal已push至`79fb7ee`；detached frozen worktree的正式fresh0→200已在
  tmux `ember-spg-cp24-79fb7ee`启动。首macro `19.431s`，24 tasks/480 queries/
  24 videos、B20、long-first、4 CUDA ranks和CP `13 gather=13 sync`全部通过。

## 2026-07-31 v5.2 task-complete macro400与候选启动

- frozen worktree commit `60f4508`上的正式root自然完成macro400；run contract
  SHA `152c0818...6088e`，run summary SHA `857f0111...ee66`，未提前截断或融合。
- `400` macros消费`192,000` action queries、`9,600` teacher-video conditions；
  wall `9695.1329s`，最后train/validation functional loss为
  `.09633848/.13686878`，全程finite，validation/test action reads为0。
- 一次live preflight确认main/origin/frozen均为clean `60f4508`、个人占用
  `350,451,040,256` bytes；只查询GPU4–7，未触碰0–3。随后tmux
  `ember-v52-candidates-60f4508`用GPU4/5/6/7分别启动macro150/200/350/400
  correct400，四个launcher存活且命令显式B-scale1、without-replacement、
  6 replicas/6 generators/batch16。

## 2026-07-31 SPG canonical CPU实现

- 独立写worktree `EMBER-spg-60f4508-20260731`已把canonical Writer切换为
  Semantic Program Grid，并删除活动`temporal.py`/v5.2 320-slot执行路径；历史
  与正在运行的v5.2由Git及独立frozen worktree保存。
- 新owner边界为`video_program.py`证据前端、`semantic_program.py` Core/Program、
  `program_compiler.py` target/rank compiler、`conflict_projection.py` CP-24，
  `model.py`只负责编排与public LoRA合同。
- 精确参数`10,633,216`；fresh incompatible schema/config为
  `configs/pi05_as_writer_semantic_program_grid_cp24_decay400_v1.json`。
- 全仓回归`201 passed in 26.18s`，`git diff --check`通过，architecture guard无
  hard violation。当前配置临时使用teacher-video seed172，只为确保首个三macro
  profile覆盖task38/demo36的最长105-frame视频；profile后必须改回formal seed
  `20260722`，再在同一干净commit完成fresh/exact-resume seal。

## 2026-07-31 当前交接

- main `799aa66`已恢复exact v5.2 topology并封存task-complete fast-decay400
  config；最长视频B20 profile和fresh0→1→exact-resume1→3通过。
- v5.2 step900的400套correct-video LoRA已完成生成和内部几何分析，未启动env
  或rollout；永久analysis SHA256为
  `9d816baadace851153415a06334efad6f9927bf334f014d5e8ae760be357e1af`。
- 结论：v5.2 q/v 16坐标能量均匀、建设性同向，effective近rank1不是负相消；
  其same-task视频创新明显高于v6既有估计。
- Coherent-Procedure/B-only residual已撤回；下一整体架构SPG已封存在
  `docs/action_forecast_writer_semantic_program_grid_design.md`。
- 新session先立即启动v5.2 task-complete macro0→200→400，然后充分审计仓库；
  无论v5.2结果好坏都实现SPG并进入每版一小时的持续根因迭代。
- 当前没有正式训练、rollout或tmux。v5.2 task-complete macro0→200→400属于
  新session第一实验，本session未启动。

## 当前状态

- 当前session-local Goal以工具实时状态为准；不会因source base、Writer或任一局部阶段提前完成长期主线。
- 活动目标split仍为四个标准LIBERO suites、每suite 6 train / 2 validation / 2 test，总计24/8/8；seal位于 `configs/libero_24_8_8_v1/`。
- generic `lerobot/pi05_base` revision `7de663972b7817d2c4cf2d84c821153dfea772e9` 已下载，weights SHA256 `0eb11ca9587678c1d2ef8cf32807c29f8ce53a2bfdfc1aa4a4c96f16fca59b0f`。
- generic base在8 test tasks×50 fixed states上为 `0/400`。400 rows唯一、全部到suite horizon，result seal SHA256 `c78e92e9...20c2`；该结果不评价EMBER。
- Phase A source audit、71-task manifest、source-only normalization、pinned official recipe与hash seal已完成；cost-balanced dynamic evaluator及fail-closed contracts已完成，真实1/2/3 replicas吞吐profile选择3 replicas/GPU。
- canonical π0.5 source-base full-SFT runner、atomic checkpoint与exact-resume机制已完成；fresh 1,000-step、333-step warmup、global batch256正式训练及step1000 raw/EMA checkpoint验证已完成，目标是轻量interface acquisition而非LIBERO-90收敛。
- formal attempt1因NUMA affinity缺失在step12终止；attempt2因显式zero右腕被LeRobot误标`mask=true`而在step316终止。两者均无checkpoint、failure packet已封存且永不resume；修正后的训练/评测都通过missing feature key得到OpenPI规定的zero image + `mask=false`。
- source/evaluator阶段的112-test里程碑是历史证据；v5初版曾完成`187 passed`，
  当前单视频合同切换后只运行防止无效正式实验所需的focused 25 tests并全部
  通过。没有用重复全仓仪式性校验延迟GPU训练。
- 第一轮完整流程只跑一个training seed；不提前扩多seed或direct action-budget curve。
- v5已完成step0→1800与正式五臂并因顺序行为门失败退役。当前focused Writer
  为v5.1 Language-Axial Semantic Core + Causal Action Procedure +
  Slot-Normalized Fusion；实时接手信息见`docs/active_session_handoff.md`。
- 2026-07-27交接审计时`/data/ymdai`占用约337.34GB，低于500GB cap；该值是
  live快照，新launch仍须重查。只清理已核验可再生的smoke/profile产物，不删除
  正式checkpoint、raw rows或来源不明文件。

## Generic feasibility已验证的实现事实

- 使用LeRobot official π0.5 conversion：model chunk50、`n_action_steps=10`、每次执行前5 actions后重规划、10 flow inference steps。
- official evaluator：render256、model224、两相机旋转180°、seed7、50 fixed init states、dummy settling10；horizons Spatial/Object/Goal/Long=`220/280/300/520`。
- OpenPI公开PaliGemma tokenizer已逐token核验；模型/tokenizer manifests与24-train interface stats均封存在当前config目录。
- 首次8卡formal因固定`MUJOCO_EGL_DEVICE_ID=0`使GPU1–7在rollout前失败；修复为每进程物理`CUDA_VISIBLE_DEVICES`后formal全部exit0。失败root与正式root隔离。
- 单卡profile：1 env约27.52秒/episode，8 env约19.76秒/episode，16 env约19.58秒/episode；8→16仅约0.9%提升，峰值显存约20.1→23.2GB。
- 静态一task/GPU使两个horizon-520任务成为最后拖尾：正式最长task rollout约2169秒，而Spatial约1004秒。下一evaluator必须做cost-balanced state sharding/dynamic queue，不得复用静态映射作为效率上限。

## Target split

| suite | train | validation | test |
| --- | --- | --- | --- |
| `libero_spatial` | 0,2,4,5,7,9 | 1,3 | 6,8 |
| `libero_object` | 2,4,5,6,8,9 | 1,3 | 0,7 |
| `libero_goal` | 0,1,2,5,8,9 | 3,6 | 4,7 |
| `libero_10` | 4,5,6,7,8,9 | 1,2 | 0,3 |

算法seed `20260721`；key为 `seed\0suite\0task_name\0language\0bddl_file` 的SHA256排序，前6/中2/后2。test IDs不得按outcome替换。

## Source-base corpus audit：完成并封存

- 只读task language、BDDL、objects、roles、initial predicates与完整ordered composition，逐一审计90×40=3600对；算法为`manual_role_bddl_full_task_equivalence_v1`，不读取action/reward/proprio/terminal/normalization/policy outcome。
- 规则只排除完整有序任务等价，保留primitive/subtask containment与不同multiplicity/source/destination selector；scene、coordinates、distractors和BDDL实例编号不参与等价判断。
- 排除LIBERO-90 IDs `8,9,10,20,25,27,30,31,44,46,47,48,49,50,51,52,53,54,77`，active为其71-task补集。Goal3/4/7/8/9、Object0/1/4/5/6/7/9及Long5的完整映射记录在audit rows中。
- 明确保留near-miss IDs `2,29,12,13,14,15,38`，因为其完整composition或role selector与目标不同。
- active corpus为71×50=3550 successful episodes、529,173 frames、52,710,755,898 bytes；HDF5 aggregate SHA256 `81bdb358...a1a50e`。
- canonical hashes：overlap audit `fe731127...cc003`、manifest `75453a20...2e54`、source-only normalization `e259ee6e...f7c4`、recipe `4c537067...281734`；`sha256sum -c`通过。recipe hash变化只加入pinned OpenPI相机mask authority，不改变source corpus。

不得根据后续source-base/Writer outcome修改这些source IDs。

## π0.5 source-base recipe与工程合同

- 官方anchor固定为OpenPI `15a9616...ccac`、LeRobot `30da8e6...76ce`：full action-SFT、AdamW `(0.9,0.95)`、eps `1e-8`、weight decay `1e-10`、clip1、peak LR `5e-5`、10k linear warmup后constant、EMA `0.999`、30k steps、global batch256。
- source base采用full-SFT并最终直接冻结policy/EMA，不叠shared source adapter；下游统一LoRA合同为18层action expert q/v加action_in/out共38 targets、rank/alpha16、dropout0、B=0 identity init。
- 当前有效8卡profile只取相机mask修正后的m32+EMA smoke：3/3 steps finite，steps2–3平均47.75 examples/s，每卡peak allocated/reserved为67.18/71.18GB，八卡角色与NUMA绑定对称。contract/metrics/summary/log SHA256为`90fbe1da...0458`、`de2d9889...50d9`、`0a590a29...e1bc`、`26bb5aad...c10`；旧batch对比profile只作工程provenance。
- checkpoint包含policy、EMA、optimizer、scheduler、per-rank Python/NumPy/CPU/CUDA RNG、sampler/data/metrics cursor与完整hash manifest。两次从同一step1恢复到step2得到相同loss `0.3457298893481493`、grad norm `4.03354549407959`与逐rank state hashes；独立NCCL启动后policy/EMA最大末位差为`1.49e-8/3.73e-9`。compact evidence SHA256 `16137fa1...b1e`。
- formal fresh launch要求clean且HEAD已推到`origin/main`；resume由保存的commit/contract约束，不因后续`origin/main`前进失效。每次invocation另存当时完整Git观测。

## Canonical code ownership与生命周期

- `pi05_source_corpus.py`只拥有specification filter、data seal和source normalization；`pi05_source_setup.py`只拥有model/data/distributed setup；`pi05_source_contract.py`只拥有launch/resume contract；`pi05_source_checkpoint.py`只拥有atomic state；`pi05_source_training.py`只拥有训练编排与step loop。
- `scripts/train_source_base.py`是唯一活动π0.5 source-base入口；拆分这些owner是因为严格加载、52.7GB一次性校验、DDP训练与33.8GB atomic checkpoint是独立故障边界。
- 旧`source_base.py`/`source_base_checkpoint.py`仍被历史SmolVLA模块import，只作provenance；不得作为活动入口。π0.5 source-base训练与40-task evaluator达到功能对等后，先迁移剩余通用依赖，再删除旧可执行路径，不保留双canonical runner。
- target/Writer新增代码按故障边界分属六个owner：`pi05_target_data.py`负责零数值读取seal，`feature_cache.py`负责PI05 cache schema/store，`cache_pi05_writer_features.py`负责8-rank extraction，`writer/as_contract.py`负责24-task action墙及authority联锁，`writer/training.py`原位替代旧cold-start训练owner，`writer/checkpoint.py`负责atomic exact-resume。新增体量来自target authority、PI05 2048-d cache和完整AS state合同，不是平行算法实现；architecture guard无parallel family、无hard violation。
- 已删除旧`cache_writer_features.py`和`train_writer_cold_start.py`两个可执行入口。历史Smol `writer/inference.py`、RL与direct modules仍为provenance且会被新AS schema/PI05-only store fail-close；当PI05 Writer evaluation、RL-Writer、Source-SFT与task-local owner逐个具备功能对等时，删除对应旧CLI/import，而不是再加兼容分支。Phase F若需test video cache，将扩展同一PI05 cache authority并新增sealed role，不创建第二个cache runner。

## Formal source-base attempt 1：工程失败并关闭（2026-07-21）

- commit/worktree：`236202ed4371f301ef94bf8984aab423eef98db1`，`/data/ymdai/worktrees/EMBER-pi05-source-formal`；model/data/tokenizer全部hash门禁通过，8卡各一个69,130MiB CUDA rank。
- root/log：`/data/ymdai/outputs/ember/pi05_source_base_v1_seed7_236202e_20260721`与同名`.log`；live CPU `PSR`发现rank可跨GPU的本地NUMA node漂移，违反launch contract。
- 在step12、3,072 global examples、首个checkpoint前发送SIGINT，pane exit130；failure packet明确禁止resume或科学使用。run contract/metrics/log SHA256分别为`997af43a...8b2`、`81dbfcbc...4ca`、`7a169300...118`。
- 修复：初始化CUDA device后从PCI sysfs解析NUMA node，把rank及其DataLoader children限制到该node cpulist；formal缺少binding时fail closed。修复smoke记录8个rank的完整affinity，metrics/contract SHA256为`bc01e6ce...6ca`/`42727d8c...18d`。

## Formal source-base attempt 2：工程失败并关闭（2026-07-21）

- canonical workspace：`/data/ymdai/worktrees/EMBER-pi05-source-formal-77ff1ef`；task-owned branch `codex/pi05-source-formal-77ff1ef`；commit `77ff1ef21567e1d5290921ca308cd8792813a504`已在`origin/main`且worktree clean。
- command：`PYTHONPATH=/data/ymdai/worktrees/EMBER-pi05-source-formal-77ff1ef/src CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=8 /data/ymdai/projects/EMBER/.venv/bin/torchrun --standalone --nproc-per-node=8 scripts/train_source_base.py --config configs/pi05_source_base_v1.json --data-root /data/ymdai/ember_data/LIBERO-datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a/libero_90 --foundation-path /data/ymdai/ember_data/lerobot_pi05_base --tokenizer-path /data/ymdai/ember_data/openpi/paligemma_tokenizer.model --output-dir /data/ymdai/outputs/ember/pi05_source_base_v1_seed7_77ff1ef_20260721 --mode formal --num-workers 4`。
- scale/topology：71×50 successful source episodes；30,000 steps×global batch256=`7,680,000` examples；8-way DDP、每rank microbatch32、EMA0.999、无accumulation。formal代码要求rank0–3绑定NUMA0、rank4–7绑定NUMA1，并把8行affinity写入launch contract；每卡仍恰好一个同角色CUDA process。
- provenance/evidence：model、71 HDF5、tokenizer、audit、normalization和recipe全部重算/核验SHA；只读source action/state，不读取target action/reward/outcome。训练metrics只作source acquisition；完成后另行运行40-task fixed-state screen。
- output/log/storage：全新root为上述output，stdout/stderr为同名`/data/ymdai/logs/ember/*.log`；checkpoint每5,000 steps原子发布并只保留最新。与launch同一preflight shell测得个人占用379,947,614,208 bytes，双checkpoint峰值447,622,427,872 bytes，低于500GB cap且不下载/复制模型数据。
- resume/failure：仅从该root最新完整manifest、相同commit/config/contract恢复；超前metrics隔离到failure packet。attempt1 root永不resume；任何NUMA、hash、设备数、非空output或checkpoint漂移均fail closed。
- live acceptance：8个rank分别为PID 1131909–1131916，逐卡仅一个同角色CUDA进程且NUMA正确；运行至step316、80,896 examples时发现相机合同违反pinned OpenPI recipe，主动SIGINT停止。无checkpoint，不得resume或作科学结果。
- 根因与修复：显式传入zero右腕使LeRobot自动生成`image_mask=true`；官方OpenPI LIBERO transform要求third-camera zero image但mask为false。canonical processor和evaluator现在省略右腕feature key，由LeRobot创建zero padding + false mask；source config对此fail closed。
- evidence：failure packet/run contract/metrics/log SHA256分别为`2d2a9e40...9b80`、`e79e1c84...e7d8`、`fb0b2edc...f918`、`3f0eb65f...76f7`。旧tmux dead shell已在确认无GPU进程后清理。

## Canonical cost-balanced evaluator（实现与source profile完成）

- `scripts/evaluate_pi05.py`取代并删除旧静态`evaluate_pi05_base.py`，是唯一活动π0.5目标评测入口；不保留双runner。
- `pi05_eval_contract.py`拥有authority/final-policy/test-state门，`pi05_eval_queue.py`拥有cost-balanced SQLite WAL队列，`pi05_evaluation.py`拥有persistent policy/env与official rollout，`pi05_eval_results.py`单独拥有worker拓扑证据和strict aggregation；拆分是为隔离调度、runtime与不可变结果故障边界，不是平行runner。
- state shards按`count × horizon`估算cost并动态work-steal；8 GPUs上统一1/2/3 replicas，launcher CPU-only，GPU0无额外CUDA角色。policy noise按`(seed,suite,task,state,replan)`确定性独立，不受batch或worker顺序影响。
- launcher lock覆盖active-worker audit、queue recovery、preflight与spawn；partial spawn/failure只回收本launcher PIDs并封存logs/jobs/hashes。正式吞吐包含worker spawn、model load和首次env/EGL，另报raw shard window。
- formal/screen拒绝非当前完整source config、非final step1000 selected raw policy、相机interface漂移、test init hash漂移及同大小model/tokenizer篡改；aggregate交叉核对raw rows、DB counts、producer、8×replica topology、GPU UUID和NUMA。

- 在同一all-40-task×1-state panel上，1/2/3 replicas/GPU分别得到有效`0.155553/0.181832/0.189738` rollout/s，shard window分别`122.879/79.176/65.454`秒；3 replicas稳定、每卡约31GB，正式source screen据此使用3 replicas/GPU。

## Formal source-base attempt 3：已由owner停止并封存（2026-07-21）

- canonical workspace/commit：`/data/ymdai/worktrees/EMBER-pi05-source-formal-aa8b055`，branch `codex/pi05-source-formal-aa8b055`，commit `aa8b0556619889480d8d9c129ea2f54af26c9d06`；启动时clean且等于`origin/main`。
- exact command：`PYTHONPATH=/data/ymdai/worktrees/EMBER-pi05-source-formal-aa8b055/src CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=8 /data/ymdai/projects/EMBER/.venv/bin/torchrun --standalone --nproc-per-node=8 scripts/train_source_base.py --config configs/pi05_source_base_v1.json --data-root /data/ymdai/ember_data/LIBERO-datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a/libero_90 --foundation-path /data/ymdai/ember_data/lerobot_pi05_base --tokenizer-path /data/ymdai/ember_data/openpi/paligemma_tokenizer.model --output-dir /data/ymdai/outputs/ember/pi05_source_base_v1_seed7_aa8b055_20260721 --mode formal --num-workers 4`。
- output/log/tmux：`/data/ymdai/outputs/ember/pi05_source_base_v1_seed7_aa8b055_20260721`、同名`/data/ymdai/logs/ember/*.log`、session `ember_pi05_source_aa8b055`。这是从generic base fresh启动的新root，绝不resume attempt1/2。
- launch authorities：source config SHA256 `366a5e12...ee8`、recipe `4c537067...1734`、run contract file `6db50495...1bec`；foundation weights和71个source HDF5均在formal启动时完整重算SHA并通过，right wrist合同为missing feature key→zero padding + `image_mask=false`。
- scale/topology：71×50 episodes，30,000 steps×global batch256；8个同角色rank PID 1264369–1264376，rank0–3绑定NUMA0、rank4–7绑定NUMA1。step7时每卡约69.0–69.2GB、GPU UTL 100%，loss/gradient finite，稳态约47.5 examples/s。
- live preflight：8卡启动前均0MiB且无compute apps；driver570.158.01、CUDA12.8、torch2.11.0+cu128。`/data/ymdai`为379,033,156,799 bytes，按实测33,837,406,832-byte checkpoint的atomic双态峰值估计446,707,970,463 bytes，低于500GB cap；`/data`可用约3.059TB。
- checkpoint每5,000 steps原子发布、只保留最新；完成前不作source competence结论。首个checkpoint预计约7.4小时，完整30k按当前吞吐约44.7小时；等待期间只在另一worktree推进不改其import/config/output的后续代码。
- step48 fresh live check仍为8卡各一个同角色PID、约69.0–69.2GB、100% GPU；loss/gradient finite，稳态约47.2 examples/s。该检查只证明运行健康，不作行为结论。

- owner随后把source预算明确改为fresh 1,000 steps；本attempt在step2880停止且没有checkpoint，`superseded_run.json` SHA256为`7aa78458d9f79885206b850f3c61738a5828019a14ee75ee06be91f1f9ff40a4`。不得resume或用于后续方法。

## π0.5 LoRA / one-video Writer core里程碑（2026-07-21）

- 新活动合同`configs/pi05_lora_v1.json`绑定generic `lerobot/pi05_base`完整revision/weights/config、当前source config与recipe hashes；文件SHA256 `1dcf58f7...cb07`，canonical contract SHA256 `42d5919e...94dd7`。
- 真实foundation safetensors metadata和meta-device `PI05Pytorch`结构均核验38个精确Linear targets；rank16得到76 tensors、1,287,168 parameters。没有加入state/time projections，也没有沿用旧37-target Smol合同。
- 通用`lora.py`只保留PEFT mechanics协议；活动科学拓扑由`pi05_lora.py`单独fail-close加载。这样旧历史imports不进入π0.5 runner，也没有第二个活动训练/评测入口。
- `CompleteLoRAWriter`现在在活动边界只接受一个非空video (`offsets=[0,L]`)；LoRA template和输出逐tensor保留真实BF16/FP32 dtype。functional action loss调用真实PI05 `forward(batch)`，不再传旧接口`noise/time`。
- fresh验证：真实checkpoint target metadata通过；mixed-dtype functional/copy parity精确通过；全套`91 passed`、compileall、diff/checksum通过。architecture guard为`REVIEW`、无hard violation；review仅来自既有Writer构造函数长度和目录密度，owner/lifecycle如上。

## Target40 seal、PI05 feature cache与AS-Writer owner（2026-07-21）

- target40下载已完成且未复制现有cache：四suite各10个HDF5，总计33,784,856,577 bytes。metadata-only seal为40 tasks/2,000 episodes/338,575 frames，24/8/8 IDs逐项一致；manifest/checksum均通过，manifest SHA256 `1b28547f...049d`。
- `configs/pi05_writer_feature_cache_v1.json` SHA256 `2165a2d9...e3ce`，只授权development train+validation视频，禁止test video与任意trajectory action/state/reward/terminal读取；raw source policy上的8卡batch32 smoke完成8 tasks/8 episodes/1,033 frames，critical-path `689.47 frames/s`且无OOM/nonfinite，因此profile已封存。
- smoke从launch到manifest约122秒，其中实际每task抽取1.03–1.50秒；run-contract/cache-manifest/log SHA256为`f166e86b...cda7`/`906e75b3...a178`/`f845a53c...65d3`。development formal cache规模为32 tasks/1,600 episodes/274,523 frames，BF16视觉主体约1.124GB。
- development formal cache已完成32 tasks/1,600 episodes/274,523 frames，实际目录1,126,794,227 bytes，launch到manifest 248秒；32/32 task safetensors逐文件hash复核通过，`test_video_values_read=0`。run-contract file/contract、manifest file/payload、extraction、log SHA256分别为`f803806a...1c7b`/`edcd59da...9410`、`72edc286...e92a`/`4b8f064c...a6a0`、`4ee82d04...2757`、`ce481b0d...5e94`。
- `configs/pi05_as_writer_v1.json`随feature authority更新后SHA256为`eb54c748...ea09`，明确24 train actions、one-video input、independent sampler/video seeds、frozen source normalization及≤120分钟正式wall-clock；formal状态`pending_profile`，当前4-step/batch1仅为未来mechanics profile默认值。
- AS训练已原位替换旧Smol cold-start owner：每rank每step由task-balanced action sampler给出task/visit，再由独立teacher-video schedule选一条demo；`WriterFeatureStore.load_one_video`只暴露pure language、该episode的video features和`[0,L]` offsets。policy、base与encoder冻结，只有shared Writer DDP更新。
- checkpoint先验证canonical manifest和全部file SHA再读取optimizer/RNG pickle，交叉核对manifest/trainer/rank cursor，metrics JSONL按checkpoint cursor隔离orphan rows；formal最终coverage由launch total-step自动推导，调用者不能关闭。rank写盘/发布失败会跨8 ranks一致传播，避免barrier死锁。
- 删除`cache_writer_features.py`和`train_writer_cold_start.py`；相关活动测试改为PI05 schema。fresh全仓`107 passed`，compileall、config SHA和target checksum通过；architecture guard结果`REVIEW`且hard violations为空。review增长理由与retirement trigger见ownership段。
- formal source-base attempt3在上述工作之外的隔离worktree继续健康运行；最近只读观测step1100，8卡各约69GB且97–100%利用率，loss/gradient finite、约46.5–47.4 examples/s。此状态不构成source competence或行为结果。

## PI05 Writer evaluation与wrong-video机械证据（2026-07-21）

- 没有新增平行runner：`evaluate_pi05.py`、`pi05_eval_contract.py`、`pi05_evaluation.py`和`pi05_eval_results.py`原位支持source-base或AS-Writer arm，共用dynamic queue、persistent env、fixed-state rows、resume和aggregate。
- AS evaluator逐字段联锁raw source policy、AS config/run/checkpoint、PI05 LoRA和feature cache；正式screen/formal只接受formal AS run，development cache可用于train/validation且会对test显式fail-close。未来test-open cache必须另行封存，不能把当前`test_video_values_read=0` cache冒充final cache。
- 每个rollout由顺序无关的哈希独立抽一条teacher video；correct与wrong arm共用selection seed/demo ordinal。wrong video按同split role跨suite双射，完整map、map SHA与condition进入run-contract hash。
- materialized backend每episode生成一次完整LoRA，并在该episode每次replan前安装同一state；不同活动env不会被错误合成普通同adapter batch。functional batched backend仍待final source产生后做真实rollouts/s profile。
- raw rows和aggregate保留checkpoint/cache/map/video/LoRA/timing证据，row validator会重算video seed、demo和map。fresh全仓`112 passed`，compileall、diff check通过，architecture guard为`REVIEW`且无hard violation；尚未因source未完成而运行GPU Writer smoke或产生科学结果。
- formal source-base attempt3最近只读观测step1100：8卡约69GB、97–100%利用率，loss/gradient finite、约46.5–47.4 examples/s；仍未到首个checkpoint，不作competence结论。

## PI05 shared Source-SFT owner与静态评测接入（2026-07-21）

- 新增development-only配置`configs/pi05_source_sft_development_v1.json`及checksum（SHA256 `32e927c...8a641`）。它只授权24 train tasks×50 action episodes，四suite各6个；final stage在该config内fail-close，validation选择后必须创建独立final authority，不能续接development LoRA。
- `ember.source_sft`成为单一owner：`contract.py`联锁target manifest、final raw source policy、tokenizer/source normalization和PI05 LoRA；`training.py`在8个对称DDP ranks上只训练一套shared LoRA；`checkpoint.py`原子保存adapter-only exact-resume state；`inference.py`核验formal artifact并一次性安装静态adapter。薄入口为`scripts/train_source_sft.py`，没有复用旧Smol per-task direct runner。
- exact-resume state包含optimizer/scheduler、optimizer=micro cursor、metrics cursor、每rank RNG、sampler/data identity、DataLoader-derived worker seed与50-episode coverage；manifest在pickle前逐文件验SHA。development/final、source checkpoint、config或LoRA合同任一变化均拒绝resume/evaluation。
- canonical `evaluate_pi05.py`新增与AS-Writer互斥的Source-SFT参数；共享LoRA每worker只安装一次，随后继续普通multi-env batch和dynamic queue。raw rows记录固定LoRA state SHA；Source-SFT不生成Writer row，也不生成Writer correct/wrong pairing hash。
- fresh验证：Source-SFT与evaluator聚焦测试`23 passed`，全仓`119 passed`，compileall、config checksum、diff check通过；architecture guard为`REVIEW`且hard violations为空。review增长来自新增baseline的独立data-wall/training/checkpoint故障边界；旧`direct_lora*`的删除触发为本PI05路径完成真实8卡finite loss/grad与exact-resume smoke。
- formal source-base attempt3保持隔离且未被修改；最近只读观测step1450、371,200 global examples，8卡各约69GB、100%利用率，loss/gradient finite、约47.39 examples/s。仍未到step5000 checkpoint，不作source competence结论。

## PI05 reward core、zero-AS RL-Writer与test-only task-local合同（2026-07-21）

- Writer-v2替换活动authority后，`configs/pi05_rl_writer_development_v1.json`与`configs/pi05_task_local_rl_test_v1.json`只做机械rebind；随Writer-v2 formal seal更新后当前SHA256为`6eac8449...b954`/`97a4ce86...b509`。前者formal状态仍为`pending_source_screen_and_real_profile`；后者所有正式budget保持0并写明`blocked_until_zero_interaction_test_and_test_open`，因此当前代码完成不会越过阶段信息墙，也没有启动RL。
- `ember.reward`统一实现official random BDDL reset、10-step settling、suite horizon、显式逐replan PI05 flow-noise、成功即停、immutable raw ledger和三类cursor。成功trajectory只保留真正执行的每个replan前缀；reward loss不会监督未执行的45/50 actions。
- `ember.rl_writer`拆分为contract、runtime、loop和checkpoint owner。fresh Writer在8 ranks上使用共同确定性seed且generated LoRA功能恒等；只有rank0原子发布run contract并跨rank校验digest。Writer-only DDP更新、task/video full-cycle coverage、完整per-rank RNG、optimizer/scheduler、metrics与ledger-bound exact-resume均已接入；micro-AS分支在zero完整负证据前硬拒绝。
- `ember.task_local`已封存8 test tasks、三臂/cohort video/匹配seed、一次性初始化bundle、physical task-LoRA-only executed-prefix update、随机reset reward checkpoint选择和hash-bound resume mechanics；尚未实现或启动test formal runtime，不把机械合同写成结果。
- 为保持一条活动路径，删除旧Smol `evaluate_source_base.py`、`train_writer_only_rl.py`与`train_task_local_lora_rl.py`三个可执行入口。历史模块/配置暂时只供provenance；新π0.5 task-local runtime和canonical fixed-50接入达到功能对等后删除其余旧实现与测试。
- fresh验证：reward/RL-Writer/task-local/evaluator定向`37 passed`，全仓`141 passed`，三份新config checksum、compile与diff check通过；architecture guard为`REVIEW`且无hard violation。约4k新增活动行按上述三个故障边界分属共享reward、shared Writer与task-local mechanics，不存在parallel version/function family；retirement trigger如上一条。
- formal source-base attempt3仍在隔离worktree不受修改；最近只读观测step2050、524,800 global examples，8卡各约69GB且100%利用率，loss/gradient finite、约47.37 examples/s。该状态仍不是source competence或行为结论。
- `configs/pi05_seen_panel_v1.json`在上述outcome之前封存四suite各2个development-train tasks，文件SHA256 `6f96b28e...0aee`；选择只使用task language/identity/BDDL SHA与固定seed，global IDs为`0,2,15,12,21,28,39,37`，policy outcome和trajectory value reads均为0。

## Frozen RL-Writer canonical evaluator接入（2026-07-21）

- `src/ember/rl_writer/inference.py`核验RL run contract、8-rank checkpoint全部file hashes、source/cache/config联锁，并从checkpoint cursor重算task-balanced schedule和teacher-video coverage；development config未seal时formal evaluation继续fail-close。
- `writer/inference.py`把AS/RL两种Writer统一到同一video mapping、pairing hash、per-rollout LoRA生成和raw evidence；row evidence现在显式记录`writer_method`、checkpoint axis/cursor。AS axis为optimizer step，RL axis为reward update。
- `evaluate_pi05.py`新增互斥RL-Writer资产参数，但仍只使用原dynamic queue/persistent worker runner；prepare与runtime adapter ownership集中在既有`eval_adapters.py`，脚本由797行降至755行，没有第二套评测入口。
- fresh验证：相关`38 passed`，全仓`143 passed`，compile/diff check通过；architecture guard为`REVIEW`且hard violations为空、无parallel family。formal source-base attempt3最近只读观测step2250、576,000 global examples，loss/gradient finite、约47.17–47.42 examples/s；仍不作行为结论。

## Fresh 1k source base完成（2026-07-22）

- canonical workspace/commit为`/data/ymdai/worktrees/EMBER-pi05-source-formal-aa8b055`、`e2cc238b6423d3c41c681e3764fca96d64203a16`，启动时clean且等于`origin/main`。全新root为`/data/ymdai/outputs/ember/pi05_source_base_v1_seed7_1k_e2cc238_20260722`，没有续接任何旧权重。
- 8×A100、global batch256、333-step warmup、EMA0.999，恰好完成1000/1000 optimizer steps与256,000 global examples；训练loop wall-clock `5494.806s`（91.58分钟），启动至结束约95分钟，低于120分钟guardrail。
- loss的50-step means从steps1–50的`0.26213`降至951–1000的`0.08659`；末段仍缓慢下降但已经明显趋平。按owner指定budget停止并记录budget-censored，不自动追加训练。
- final checkpoint为`checkpoints/step_00001000`，完整file/manifest验证通过；原始contract/summary/metrics/checkpoint-manifest/log SHA256分别为`ae05c077...e374`、`921ea45a...2032`、`b77c8498...6d3c`、`c236cb2d...d6bf`、`d2f17289...2d0c`。checkpoint contract canonical hash为`b6090341...e867`；原summary选择EMA已被后续诊断纠正并保留原hash作provenance。
- checkpoint apparent bytes为`33,837,823,088`；trainer state确认step1000、raw与EMA都存在、1000条metrics step唯一且无invalid rows。下游共同起点现明确为raw `policy/`与同一source-only normalization；不重训source base。
- 先前40-task×8-state EMA screen为0/320，机械执行虽完整，但该模型选择无效：EMA0.999从generic初始化，在1k短训后仅走完raw参数更新位移的28.62%。同一4个active LIBERO-90 tasks、相同init/noise下raw 4/4、EMA 0/4；同一32 source samples flow loss为raw `0.06165`、EMA `0.17775`、generic `0.29302`。
- raw 40-task×8-state正式screen得到`46/320 = 14.375%`，成功覆盖13/40 tasks及全部4 suites；逐suite为Long `2/80`（1 task）、Goal `28/80`（5 tasks）、Object `1/80`（1 task）、Spatial `15/80`（6 tasks）。因此跨多task partial competence门槛正式通过，不是0 competence或单易task支撑。
- screen共有320个唯一task/state rows、24/24 workers exit0、40 shards完整、无错误；wall-clock `412.372s`、有效`0.775999 rollout/s`。results/run-contract/run-summary/log SHA256分别为`4e2defaf...db3a`、`e496dc6a...1d7b`、`7befa655...ed71`、`ab553c0e...4e9d`；旧EMA 0/320仅保留为superseded engineering evidence。

## 已对齐的后续方法

- frozen source base：过滤后LIBERO-90×50 action-SFT，必要source LoRA merge，source-only normalization冻结；快速screen全部目标40 tasks，需开始在多个tasks有部分真实成功，不能只靠一个易task aggregate。
- 全部适用训练阶段：先短profile学习速度/吞吐，用固定廉价screen与曲线斜率淘汰候选，仅对少量候选做完整validation；约2小时只是guardrail而非目标，到上限仍未充分训练则保存证据后停止自动追加。
- AS-Writer：24 train/8 val开发，one video，video/action episode同task独立采样；loss驱动稀疏val与早停。
- RL-Writer：随机Writer、零AS warm-up起步；无reward再极少warm-up，仍失败则关闭。
- Source-SFT：24/32 source tasks联合一套shared LoRA；独立val选最佳，不匹配AS steps/data。
- seen comparison：specification-only预声明覆盖四suites的source panel。
- wrong-video：直接另一suite，正确language/task/state/RNG不变。
- final：合并为32 source，单seed分别重训后先seen、再zero-interaction test。
- test-only RL：不碰validation；test task上训练identity/AS/RL Writer三臂到接近最佳，官方random resets，fixed50只fresh eval。
- task-local RL预算按每种初始化方法跨全部8个test tasks的合计训练wall-clock约束，不是每task各给2小时；逐task记录是否因总预算截断。
- direct oracle：最后使用8 test tasks×50 actions联合一套shared LoRA，不是per-task LoRA。
- optional：核心后有时间再做ViVLA；outer learning不阻塞。

## 当前后续动作

1. 进入final 32-source单seed fresh重训；development Source-SFT、AS-Writer和RL-Writer均不再追加训练或额外消融。
2. final模型先完成sealed seen comparison，再打开8个test tasks做zero-interaction correct/wrong对照。
3. RL-Writer后续只保留既定task-local RL-init与final correct/wrong用途；其development视频因果证据较弱，不补generic arm或micro-AS追正结果。

## AS-Writer短周期profile与正式seal（2026-07-22）

- 最初4-step/batch1 mechanics probe使用了4-step scheduler horizon，LeRobot的自动缩放把1,000-step warmup取整为0，首步直接施加`3e-4`并产生无代表性的loss/gradient上冲；该run仅作mechanics evidence，不参与正式步数选择。
- 修正后的profile保持1,000-step schedule horizon，只执行前128 steps；8卡每rank batch16、global action queries/step为128。稳态约1.05秒/步、约122 queries/s，每rank峰值allocated/reserved为63,534,307,840/68,167,925,760 bytes，符合约10GB稳定余量目标且无需继续batch sweep。
- 16-step mean loss依次从`0.14714`、`0.13880`、`0.12935`降到末段`0.11930`；后64步线性斜率为`-1.576e-4/step`，仍在学习且无nonfinite/冻结越界。由此选择完整1,000-step schedule，而不是按120分钟上限倒推；预计净训练约17.5分钟。
- v1正式AS配置曾封存batch16、1,000 steps、checkpoints 250/500/750/1000；这些profile与训练结果只作v1 provenance。v2仍处于真实profile前，不能继承v1 batch/step结论或冒充已封存配方。

## 历史边界

旧SmolVLA 70/10/10曾完成到旧Phase F并留下真实结果，但与当前π0.5、split、one-video和source-base合同不兼容。只能用作经验/provenance，不能复用checkpoint、normalization或runner。

## AS-Writer validation与matched-scale Source-SFT seal（2026-07-22）

- AS cheap screens为step250/500/750/1000=`24/18/15/15`（各64 rollouts），由预封存规则只将250和500送入完整validation。完整8×50结果为step250 `119/400`、step500 `99/400`，development AS-Writer据此冻结step250。
- owner要求补测的同配置source-base validation完成`48/400`，逐任务为Spatial 1/3=`0/0`、Object 1/3=`5/0`、Goal 3/6=`0/41`、Long 1/2=`2/0`；AS step250逐任务为`0/0, 40/36, 0/27, 16/0`。
- 在RL-Writer前先运行Source-SFT。首版不调step，只匹配已选AS checkpoint约32,000 action-query总训练量；保持实测高效的8卡每rank batch64，固定63 steps=`32,256` queries，只保留step63。此前batch64/128 profile仅作稳定性与吞吐证据，不作为科学结果。
- 下一动作：封存并推送Source-SFT配置后完成该固定训练；仅对step63运行一次完整8-task validation，与source base `48/400`和AS `119/400`比较。
- Source-SFT固定run已完成：63 steps、32,256 queries、训练wall `450.263s`，唯一checkpoint完整exact-resume核验通过；首/末8-step mean loss `0.15475/0.13883`，仍下降但按固定预算停止，不追加或精调step。
- step63完整8×50 validation得到`61/400`，逐任务Spatial 1/3=`5/1`、Object 1/3=`20/0`、Goal 3/6=`0/32`、Long 1/2=`1/2`。相对source base为`+13`，相对AS step250为`-58`；400 rows的env/policy seed与noise-seed共享前缀全部配对。
- 训练root为`/data/ymdai/outputs/ember/pi05_source_sft_development_equalq32k_99b6020_b64_s63_20260722`，validation root为`/data/ymdai/outputs/ember/pi05_source_sft_val8x50_equalq32k_step0063_99b6020_r3_20260722`。results/comparison SHA256为`92e3e667...3f6d`/`c376ef9c...a1f`。
- AS step250 cross-suite wrong-video正式对照完成`115/400`，correct-video为`119/400`，source base为`48/400`。paired flips为both-success 102、correct-only 17、wrong-only 13、both-fail 268；correct−wrong仅+4，当前development AS checkpoint未显示强视频内容依赖。
- wrong-video root为`/data/ymdai/outputs/ember/pi05_as_writer_val8x50_cross_suite_wrong_step0250_fa635cc_r3_20260722`；24 workers exit0、38 shards和400 rows完整，results/comparison SHA256为`0e6ee518...a9ce`/`d4a4f9f7...eaac`。
- canonical `evaluate_pi05.py`现直接支持已封存的8-task `seen_panel` derived role：入口核验`pi05_seen_panel_v1.json`及checksum、target-data authority、四suite各2个train task和零outcome/value reads；Source-SFT同一静态adapter路径仅接受该精确subset。没有新增runner或改写旧checkpoint/config authority，相关聚焦43 tests及全仓144 tests通过。

## Writer-v2组合修订已实现、待真实profile（2026-07-22）

- 旧v1 correct/wrong为119/115，且400/400视频虽不同、生成LoRA hash虽不同，有效`B@A`的correct/wrong相对差中位数只有`7.52e-6`；当前状态判定为科学shortcut negative而非cache/evaluator故障。seen与RL-Writer继续暂停。
- 新活动authority为`configs/pi05_writer_feature_cache_v2.json`和`configs/pi05_as_writer_v2.json`；真实cache smoke封存后当前SHA256为`3dc3557d...396c`/`5ffd85d4...b93c`。旧v1配置由schema fail-close，只保留artifact/Git provenance。v2仍使用同一raw source policy、24 train actions、development train/validation videos、38-target complete LoRA和同一canonical训练/评测入口。
- cache从每帧全局mean改为固定4×4 spatial grid，tensor为`frames×16×2048 BF16`；预计完整274,523-frame cache主体约17.99GB。当前`/data/ymdai`实占约232GB，峰值远低于500GB cap，无需删除既有证据。
- Writer-v2为14,403,200 trainable parameters；language/video使用独立固定token memory，所有层级attention去除query-only residual，decoder只以parameter query乘性调制conditional memory，output heads无bias且identity init不变。CPU full-shape direct check生成76个三condition tensors、全部finite，人工开启head后condition mean diff非零。
- owner最终口径已在任何artifact产生前覆盖初版三分支：固定中性language `perform the demonstrated task`经正常tokenizer/embedding进入Writer，policy在所有分支始终收到正确task language。训练按`normal → full-language contrast → generic-language contrast`三步循环；contrast用半批query复制为correct/wrong两臂并共享policy RNG，总policy samples/step不变，两个correct臂都有绝对functional action loss。action query仍来自同task独立episode；paired negative只来自预封存的另一个development-train task并采用对称配对。
- v2 cache smoke完成8 tasks/8 episodes/1,033 frames，8 ranks全部exit0；最慢rank task wall为1.550秒，对应critical-path约666.25 frames/s。每个task tensor包含普通task language、同一个8-token generic language embedding和`frames×16×2048 BF16`视频；8份generic embedding逐byte SHA256均为`62172105...7d27`。run-contract/cache-manifest/log SHA256为`b4313579...d2d7`/`bcadf191...17b`/`b0f8124c...7045`，batch32据此封存。
- v2 formal cache现覆盖32个development tasks、每task50条视频、共1,600 episodes/274,523 frames；32/32 task tensors、episode/frame counts与SHA256复核通过，generic embedding在全部tasks逐byte一致，test-video/action/state/reward/terminal读取均为0。cache root为`/data/ymdai/outputs/ember/pi05_writer_feature_cache_v2_development32_raw_e4c19f9_b32_20260722`；run-contract/manifest/log SHA256为`219920cc...1f3a`/`b98a934c...ade2`/`06cb5f6d...51a6`。
- 真实8卡batch16学习profile执行30步且三种mode各10步；normal positive首/末3步均值从`0.16161`降至`0.13554`，full-language gap从`-3.44e-6`移至`+9.58e-5`，generic gap从`-1.04e-5`移至`+4.13e-5`。最大allocated/reserved为`63,748,333,032/68,543,315,968` bytes，全部finite、policy prompt始终为正确language。profile只说明机制开始向正确方向移动，尚不构成视频特异性结论。
- 据此封存首轮250 steps、batch16/rank、checkpoints 50/100/150/200/250；每个checkpoint先做固定functional specificity诊断，step250仍未饱和则按owner规则记录undertrained而不自动追加。`configs/pi05_as_writer_v2.json`当前SHA256为`65383ab8...40ab`；下游RL-Writer/task-local仅机械rebind为`6eac8449...b954`/`97a4ce86...b509`，均未启动。不同mode的wall-clock和吞吐只作资源记录，不作为删减分支、缩小对照或拒绝启动的门槛。
- 正式首轮root为`/data/ymdai/outputs/ember/pi05_as_writer_v2_development_seed7_dcfb206_b16_s250_20260722`，完成250/250 steps与五个完整checkpoint；训练段wall `334.476s`，24 tasks各覆盖全部50 action episodes与50 teacher videos。run-contract/metrics/summary/log SHA256为`08e58005...af4c`/`592212b6...289`/`8b3e456f...37b`/`dfd866b1...a7d`，250 rows连续、唯一、全部finite。
- 首/末10步positive loss为normal `0.14501→0.11478`、full `0.14337→0.12463`、generic `0.13617→0.12040`；full/generic末10步wrong-minus-correct均值为`+0.00707/+0.00788`。这是比30-step profile更强的训练内信号，但不同step query不相同，尚不能替代fixed-query或closed-loop验证。
- 同一canonical evaluator已增加`generic_correct/generic_cross_suite_wrong`条件；generic只替换Writer language为正常cache得到的中性embedding，policy始终接收正确task language。聚焦测试`33 passed`；下一步在clean/pushed commit上对step200/250做四分支cheap validation screen。

## Writer-v2首轮validation与双方法ceiling任务（2026-07-22）

- Writer-v2 step250的64-state screen为full-language correct/wrong=`12/8`、generic correct/wrong=`12/8`；完整full-language correct/wrong=`83/63`（各400），paired correct-only/wrong-only=`40/20`、exact McNemar `p=0.01349`。视频特异性已跨5个正向tasks成立，但correct绝对性能低于v1的119/400。
- owner将当前执行范围收敛为：充分训练并独立选择Writer-v2与Source-SFT的validation最强checkpoint，然后比较并暂停。Writer候选选择只看correct-video；唯一最佳checkpoint确定后才补一次correct-language + cross-suite-wrong-video完整control，不再运行generic full或候选wrong arms。
- formal seal已通过全仓`150 passed`：Writer-v2 fresh 1,500 steps、batch16/rank、50 warmup/1,500 decay、250-step checkpoints，配置SHA256 `34c5a1f8...34b7a`；Source-SFT fresh 800 steps、batch64/rank、100 warmup/800 decay、checkpoints 100/200/400/600/800，配置SHA256 `3f5a2c93...15818`。下游未启动的RL-Writer/task-local仅机械rebind authority hash；两者若到120分钟guardrail仍未饱和，保存曲线并明确标记budget-censored，不自动追加。

## AS-Writer与Source-SFT development选择完成（2026-07-22）

- Source-SFT ceiling run完整validation为step200/400/600=`74/87/73`（各400）；step400是明确峰值，故停止在最近完整checkpoint并冻结step400，不恢复到800。development Source-SFT至此完成，后续不重复。
- Writer-v2原1,500-step run的correct-video完整曲线为step500/750/1000/1500=`99/92/75/72`。为排除未保存细峰，另从identity fresh运行相同训练合同、只加密350–750 checkpoint retention；16-state screens将600/700/750送入完整validation，结果为`90/85/95`。没有候选超过原run step500，且原run后段持续下降，因此冻结原run step500=`99/400`。
- 唯一最终wrong-video arm在同一step500上得到`55/400`；paired both-success/correct-only/wrong-only/both-fail=`43/56/12/289`，exact McNemar `p=6.21e-8`。逐task correct−wrong为Long `-1/-2`、Goal `+1/+11`、Object `+17/+12`、Spatial `+2/+4`，视频正效应来自6/8 tasks而非单任务。
- development主比较为source base `48/400`、Source-SFT `87/400`、Writer-v2 correct `99/400`。v1虽有`119/400` correct，但wrong仍为`115/400`；它更像公共LoRA。v2牺牲部分绝对correct性能，却把wrong降到55并建立强视频因果差异，按owner口径暂时通过。
- peak-scan run summary/metrics SHA256为`31e36942...f29`/`171acd12...9c1c`；最终correct/wrong results SHA256为`e55cdf8e...66e4`/`3f9d6f1e...8b6f`。下一阶段直接进入zero-AS RL-Writer，不补不影响当前判断的AS消融。

## RL-Writer真实profile与formal seal（2026-07-22）

- 第一次profile在reset前发现64-bit hash seed不满足LIBERO内部`np.random.seed`的uint32范围；0 rollouts/0 actions。seed owner现只对environment seed做uint32映射，policy/update seeds与跨arm配对不变，相关22 tests通过。
- 第二次profile完成首批8 rollouts（4 successes）后暴露成功/失败ranks的DDP分支互等；成功轨迹26–44 replans被整批反传，峰值约80GB。修复为8-chunk policy replay microbatch：先累积生成LoRA叶梯度，再一次回传Writer，并按固定顺序手工all-reduce；逐成功episode全局等权不变，没有新增算法分支。
- 第三次profile完整覆盖24/24 train tasks各1条official random-reset rollout，得到7 successes、7167 environment actions、3/3 optimizer updates；成功覆盖Spatial 0/2/5/7、Goal 1/8和Long 5。单cycle max-rank wall `133.471s`，三update分别`46.668/42.297/44.510s`，峰值reserved `40,842,035,200` bytes。
- profile checkpoint update3全文件hash、24-task no-replacement coverage和cursor复核通过；run-contract/metrics/summary/checkpoint-manifest SHA256为`487fee45...4d28`/`d7e22bc7...cdea`/`3bb5b550...008a`/`2a27f673...c95`。零warm-up已有明确reward signal，不进入micro-AS。
- formal最大合同为120 updates=40 full cycles，按profile约89分钟净循环，并为分段模型加载、checkpoint与判断保留约31分钟；checkpoint为3/6/12/24/36/54/72/96/120。runner现允许只在这些sealed checkpoints暂停并保持同一contract exact-resume；首段预声明只跑到update12。

## RL-Writer development完成（2026-07-23）

- canonical root为`/data/ymdai/outputs/ember/pi05_rl_writer_zero_formal_seed7_r2_376ac0f_20260722`。此前resume启动前的metrics reconciliation错误已定位为AS轴`optimizer_step`与RL轴`next_update`混用；修复后rank0异常会广播到全部ranks，fresh全仓`154 passed`，旧失败run只保留failure evidence，未跨代码commit恢复。
- 新run先到update3，再从update3真实8-rank exact-resume到24，随后从24到36、36到54；最终432 rollouts、81 successes、131,354 environment actions、44个有效optimizer updates，净训练wall `2261.716s`。run-contract/invocations/metrics/summary SHA256分别为`d637959f...073`/`637f4878...946`/`d5a064c7...22d`/`ed5aee66...4d4`。
- 固定64-state screens为update12/24/36/54=`6/11/15/14`，results SHA256分别为`97c78986...e78`、`59dacc72...17e`、`d7e73252...3eb`、`ae641ec8...2a1`；据此冻结update36，不继续训练。selected checkpoint manifest/writer SHA256为`85577446...596`/`57f9b12c...2af`。
- selected update36完整correct-video validation为`94/400`，root为`/data/ymdai/outputs/ember/pi05_rl_writer_validation8x50_r2_update00000036_correct_376ac0f_r1_20260722`，results SHA256 `d1d4b1cf...aa5`。唯一cross-suite wrong arm为`87/400`，root suffix为`...update00000036_wrong_376ac0f_r1_20260722`，results SHA256 `6601221a...325`；两者`paired_control_sha256=57e3985c...321`。
- paired correct-only/wrong-only为`10/3`，exact McNemar `p=0.092285`。因此RL-Writer held competence成立但视频特异性较弱；已有zero-AS reward signal，不启用micro-AS，不补generic或更多wrong arms。下一阶段是预封存seen panel比较。

## Phase E sealed seen-panel完成（2026-07-23）

- 预封存8 train-task panel的四个必要arms均完成8×50 official fixed-state评测：source base `137/400`、Source-SFT step400 `182/400`、AS-Writer-v2 step500 `204/400`、RL-Writer update36 `164/400`。逐task原始counts与runtime已写入各自`results.json`；AS最强，SFT次之，RL仍高于base。
- results roots依次为`pi05_source_base_seen8x50_raw_e92b482_r3_20260723`、`pi05_source_sft_seen8x50_step0400_e92b482_r3_r2_20260723`、`pi05_as_writer_v2_seen8x50_step0500_e92b482_r3_20260723`、`pi05_rl_writer_seen8x50_update00000036_e92b482_r3_20260723`；SHA256依次为`91a9a31f...fb833`、`05c4c0d1...d889b`、`3d640e57...d97479`、`92a958a3...3f2c8`。
- Source-SFT原训练被owner在选择step400后于step600停止，未发布terminal summary；当前evaluator的fail-closed检查在0 rollout处暴露该缺口。只依据不可变contract、600条metrics和step600 manifest确定性重建summary并另存recovery provenance，SHA256为`887ae816...ab2e`/`c7f29ae7...803c`，未运行训练或改写checkpoint。
- Phase E的必要seen比较至此封存；不补seen wrong-video或额外checkpoint。下一步为final 32-source fresh AS/SFT/RL重训合同与运行。

## Final 32-source合同已封存（2026-07-23）

- 单一AS与RL canonical runner现按immutable config支持development/final stage；Source-SFT已有final owner。32-source角色严格为`train+validation`，每suite8 tasks，test不进入训练。
- final AS保持1,500-step scheduler horizon并机械停在选定step500；final Source-SFT保持800-step horizon并停在选定step400，避免LeRobot因缩短`total_steps`自动改变warmup/decay。final RL固定12 full source-task cycles，即48 updates/384 rollouts。
- final AS/SFT/RL配置SHA256依次为`ebe269ea...e299e`、`25e99628...d10c2`、`32dd979b...2ab30`；32-task Writer cache可直接复用。聚焦测试与全仓fresh验证为`28 passed`/`159 passed`，下一步在clean pushed commit上依次启动三套fresh单seed训练。

## Final AS-Writer训练完成（2026-07-23）

- root为`/data/ymdai/outputs/ember/pi05_as_writer_v2_final32_seed7_a5173a1_b16_stop0500_20260723`；fresh训练实际500 steps、原scheduler horizon 1,500、wall `634.671s`，32 tasks各覆盖50 action episodes与50 videos，500 rows全部finite。
- checkpoint/run-contract/metrics SHA256为`b30b2e1d...c395`/`36207182...2de`/`0d208b15...b619`；末20 full/generic matching gap为`0.00729/0.00783`，训练机械与信息墙通过。
- 修正了run summary沿用development `validation_action_reads=0`的报告错误；权重和optimizer改动为0，corrected summary/correction provenance SHA256为`a4f76fb2...9de7`/`ebc1bed8...414e`。全仓fresh仍为`159 passed`；下一步启动final Source-SFT。

## Final Source-SFT训练完成（2026-07-23）

- root为`/data/ymdai/outputs/ember/pi05_source_sft_final32_seed7_5922c61_b64_stop0400_20260723`；fresh shared LoRA实际完成400 steps、保留原800-step scheduler horizon，wall `2857.608s`（metrics训练loop `2852.793s`），共204,800 queries。
- 32 tasks各用满50条action episodes，400 metrics连续finite；首/末20-step mean loss `0.15139→0.11531`。checkpoint/run-contract/metrics/summary SHA256为`0012ffb6...52bd`/`bc136964...da31`/`c0d91c9b...6211`/`ff0a33f7...d472`，exact-resume文件全部校验通过。
- final AS与Source-SFT现均完成；下一步从fresh zero-AS Writer初态启动48-update final RL-Writer，不启用micro-AS或额外分支。

## Action-Memory AS-Writer实现与formal profile（2026-07-23）

- owner将当前执行范围重置为新Action-Memory AS-Writer闭环，暂不推进RL。新canonical路径直接对每个stride-4采样帧运行冻结PaliGemma图文prefix，再用16个可学习memory tokens读取Action Expert全部18层hidden states；encoder-only rank8 Meta-LoRA只在teacher-video编码时安装，执行policy不携带它。
- 变长视频按padding mask批处理；`B×18×16`条时间序列在同一temporal Transformer中并行但互不注意，随后批量layer/slot mixing并直接解码完整76-tensor、1,287,168-scalar task LoRA。Writer共10,097,601个训练参数，是rank128 Source-SFT 10,297,344的98.1%。
- 最终源码8卡profile root为`/data/ymdai/outputs/ember/pi05_as_writer_action_memory_v1_profile_refactor_20260723`：per-rank batch16、global128，两步耗时3.851/2.477秒，loss与gradient全部finite；峰值allocated/reserved为75,991,897,600/78,873,886,720 bytes。
- profile run-contract/metrics/summary SHA256为`21f15cab...24c`/`1979da53...7e9`/`6175b068...061`。全仓fresh `166 passed`，architecture guard无hard violation。
- 正式训练合同为fresh normal-positive-only，scheduler horizon 1,200；先到step500并每100步保存。若loss/validation未饱和，避免每100步重载模型，改为每次约300步exact-resume至800、1100，累计训练wall约60分钟后无论是否充分均封存曲线并停止追加。

## 四卡bias-restored Action-Memory新Goal启动（2026-07-23）

- owner将本轮设备权限明确收窄为2026-07-28前只用物理GPU `0,1,2,3`；GPU4–7即使空闲也不进入训练、评测或额外controller角色。此前8卡profile与bias-free checkpoint只作provenance，不作为新trajectory的resume初态。
- 当前模型拓扑保持冻结PaliGemma prefix、16个Action-Expert memory tokens、encoder-only Meta-LoRA、变长temporal/layer/slot聚合和完整task-specific rank16 LoRA不变。仅把conditional path内部Linear/MHA/factor-head普通bias恢复；final factor-head bias和weight都从零初始化，fresh adapter仍为functional identity，且不存在独立公共LoRA支路。
- Writer构造字段已收敛到单一代码合同，AS训练、checkpoint评测和RL runtime共同使用；同时修复RL新Action-Memory Writer初始化漏传冻结`action_in_proj`的问题。当前只影响未来fresh RL启动，不修改任何旧artifact。
- 在读取validation action值前封存`pi05_validation_functional_loss_panel_v1`：8个validation tasks各8个teacher-video groups、每组8个不同episode action queries，共512 rows/checkpoint；query seed、task-equal aggregation和“closed-loop覆盖loss”的冲突规则均固定。新增评测只读取validation label算loss，不向Writer或optimizer暴露，不读取test action/video值。
- 四卡AS预合同使用per-rank batch16/global64、每100步checkpoint；首个候选阶段到step3200，对应204,800 action queries，和rank128 Source-SFT step400的query量相同，但只是首个判断点。可exact-resume到4800/6400，最终是否继续由validation loss与完整closed-loop峰后持续下降证据决定，不设wall-clock上限。
- fresh全仓验证为`169 passed`；配置/面板加载、CLI、compile和diff check通过。architecture guard为`REVIEW`但无hard violation，新增validation owner位于`ember.writer`且没有第二套policy evaluator。下一动作是live四卡/storage preflight与真实profile，profile后再封存并启动fresh formal AS。
- live preflight确认物理GPU0–3均为0 MiB/0%且无compute process；GPU4–7属于其他用户的活动作业，未读取其任务状态以外的数据也未干扰。个人占用`275,195,838,464` bytes，profile与后续checkpoint峰值预算低于500GB cap。
- bias-restored四卡profile root为`/data/ymdai/outputs/ember/pi05_as_writer_action_memory_bias_profile_r4_b16_580793a_20260723`：per-rank batch16/global64，两步max-rank wall为`3.8103/1.9300s`，loss与gradient全部finite；peak allocated/reserved为`75,992,039,424/78,871,789,568` bytes。
- 恢复bias后的Writer为`10,119,297` trainable parameters，是rank128 Source-SFT `10,297,344`的`98.27%`。run-contract/metrics/summary SHA256为`819bbbb6...45a1`/`19b8fd85...de28`/`950ff995...3ef`；四rank均绑定NUMA0对应GPU-local CPU集合，GPU0额外CUDA角色为0。正式四卡AS合同据此封存。

## bias-restored四卡AS正式首阶段launch contract（2026-07-23）

- canonical workspace为`/data/ymdai/worktrees/EMBER-as-valdiag-r6`，代码/config必须clean且等于launch时`origin/main`。fresh run不继承任何bias-free、v1/v2/v3或profile权重；冻结source仍为`pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000` raw policy，数据为sealed 24 train tasks，tokenizer与normalization沿用source authority。
- 规模为4-way DDP、物理GPU `0,1,2,3`、每卡一个同角色policy CUDA process、per-rank batch16/global64、scheduler horizon6400；首段fresh运行到step3200，即204,800 action queries，每100步原子保存。若需继续，只允许从该root的完整step3200 checkpoint在同一合同下exact-resume到4800/6400。
- output为`/data/ymdai/outputs/ember/pi05_as_writer_action_memory_bias_dev_r4_seed7_s3200_20260723`，log为`/data/ymdai/logs/ember/pi05_as_writer_action_memory_bias_dev_r4_seed7_s3200_20260723.log`，tmux session为`ember_as_bias_r4_s3200`。按profile每checkpoint约124.4MB，32个checkpoint连同临时原子副本预计低于4.5GB；preflight个人占用275.2GB，因此预计峰值低于280GB和500GB cap。
- exact command：`PYTHONPATH=/data/ymdai/worktrees/EMBER-as-valdiag-r6/src CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false /data/ymdai/projects/EMBER/.venv/bin/torchrun --standalone --nproc-per-node=4 scripts/train_as_writer.py --config configs/pi05_as_writer_action_memory_v1.json --mode formal --source-run /data/ymdai/outputs/ember/pi05_source_base_v1_seed7_1k_e2cc238_20260722 --checkpoint /data/ymdai/outputs/ember/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000 --tokenizer-path /data/ymdai/ember_data/openpi/paligemma_tokenizer.model --data-root /data/ymdai/ember_data/LIBERO-datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a --output-dir /data/ymdai/outputs/ember/pi05_as_writer_action_memory_bias_dev_r4_seed7_s3200_20260723 --stop-after-step 3200 --num-workers 2 --log-every 10`。
- acceptance/monitoring：首个optimizer step必须loss/gradient finite；四rank GPU/NUMA对称且GPU0无额外CUDA角色；metrics cursor连续、每100-step checkpoint完整；不得使用GPU4–7或读取validation/test action用于更新。checkpoint筛选另用预封存512-row validation-loss panel，最终以完整8×50 closed-loop success为authority。

## rank128 Source-SFT四卡上限探索准备（2026-07-23）

- 在AS正式进程运行期间，只在隔离worktree`/data/ymdai/worktrees/EMBER-sft-ceiling-r4`修改尚未启动的Source-SFT owner；没有改动AS已import的代码、config或output。
- Source-SFT runtime不再硬编码8 ranks，而是从profile/formal sealed contract读取world size。当前profile待物理GPU0–3可用后以4 ranks×batch128运行，从而保持旧比较的global batch512；此前8×batch64 profile与训练只作provenance，不能跨world-size exact-resume。
- rank128 fresh formal预合同保留原100-step warmup和800-step cosine decay，使前800步LR轨迹定义不变；最大horizon扩至2400，只在800后增加`1e-5`低LR tail。每100步保留checkpoint，首段到800，只有validation仍未建立真实峰值与持续峰后下降时才exact-resume到1600/2400；owner明确不设wall-clock上限。
- 当前只封存“pending four-rank profile”，不得直接formal启动。相关Source-SFT/LoRA聚焦测试`16 passed`，architecture guard无hard violation；AS释放四卡后先跑真实batch128 profile，再写入吞吐/显存/hash并封存正式合同。
- Source-SFT现复用同一封存512-row panel并支持两种互补生命周期：active formal训练在每个checkpoint后用常驻policy原地测loss、恢复完整RNG后继续；独立只读入口仅用于历史checkpoint backfill或训练进程结束后的复核，不执行训练、也不构成第二套policy evaluator。两者共享panel manifest、task-equal summary和Source-SFT adapter owner。

## checkpoint内联validation-loss监控（2026-07-23）

- owner澄清validation functional loss应在训练进程保持π0.5、Writer和显存常驻时原地计算，而不是每个checkpoint暂停并重载模型。bias-restored AS正式run已在完整step300 checkpoint后做一次必要代码升级；step100/200/300均已原子保存，训练轨迹未回滚。
- canonical AS runner现在只在development formal checkpoint后切换`eval/inference_mode`，换入封存512-row panel，四rank各处理16个`task×video-group`，随后恢复Writer train mode和进入monitor前的Python/NumPy/CPU/CUDA RNG。validation actions不进入Writer、optimizer或反向传播，policy/Writer不复制、不重载，训练更新合同不变。
- 每个step目录保存512 raw losses、逐task等权mean/std、相邻checkpoint loss delta、wall和checkpoint hash；连续下降用于继续训练，平台或连续多点回升用于收缩候选区间。单点波动不触发早停，完整8×50 closed-loop success仍覆盖loss诊断选择最终best。
- 独立历史backfill一次模型加载同时复核step100/200/300/400/500，task-balanced loss依次为`0.135237/0.138363/0.134698/0.141123/0.134224`；其中step300/400/500共1,536条逐query loss与训练进程内结果逐值完全相同。backfill只作旧checkpoint补齐，后续checkpoint全部由常驻训练进程原地测量。
- step500 exact-resume后，step600/700/800 loss为`0.138690/0.139285/0.140583`，相对step500形成连续三点上升；同期100-step train-loss mean从step401–500的`0.117381`降至step601–700的`0.111555`，显示train/validation开始分叉。run已在完整step800 checkpoint和在线validation后干净暂停，没有未checkpointed metrics；step500是当前loss谷底，但最终best仍待closed-loop确认。

## bias-restored AS closed-loop候选launch contract（2026-07-23）

- canonical workspace为`/data/ymdai/worktrees/EMBER-as-valdiag-r6`，branch `codex/as-valdiag-r6`，launch前必须clean且等于`origin/main`。上游训练root为`pi05_as_writer_action_memory_bias_dev_r4_seed7_s3200_20260723`，source checkpoint、tokenizer、sealed validation split、fixed-50 states、policy/video seeds与旧正式AS评测完全相同；本轮只改变Writer checkpoint。
- val-loss用于收缩区间而不替代closed-loop：首批完整8-task×50 correct-video候选为谷底前step300与谷底step500，并行使用GPU `0,1`和`2,3`；每张物理卡统一5个persistent policy replicas，两个CPU-only launcher不在任何GPU上增加模型。完成后再测峰后肩部step800，确认loss趋势与success是否一致。
- 输出分别为`/data/ymdai/outputs/ember/pi05_action_memory_as_bias_val8x50_step0300_correct_18c9e9a_g01_r5_20260723`与`...step0500_correct_18c9e9a_g23_r5_20260723`，均为不存在的新root，禁止覆盖或混入bias-free结果。每个正式结果预计约3MiB，连同queue/log远低于当前258GiB个人占用和500GB cap。
- step300 exact command：`PYTHONPATH=/data/ymdai/worktrees/EMBER-as-valdiag-r6/src /data/ymdai/projects/EMBER/.venv/bin/python scripts/evaluate_pi05.py run --config configs/pi05_target_evaluation_v1.json --source-run /data/ymdai/outputs/ember/pi05_source_base_v1_seed7_1k_e2cc238_20260722 --checkpoint /data/ymdai/outputs/ember/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000 --tokenizer-path /data/ymdai/ember_data/openpi/paligemma_tokenizer.model --output-dir /data/ymdai/outputs/ember/pi05_action_memory_as_bias_val8x50_step0300_correct_18c9e9a_g01_r5_20260723 --role validation --mode formal --state-count 50 --replicas-per-gpu 5 --gpu-indices 0,1 --as-writer-config configs/pi05_as_writer_action_memory_v1.json --as-writer-checkpoint /data/ymdai/outputs/ember/pi05_as_writer_action_memory_bias_dev_r4_seed7_s3200_20260723/checkpoints/step_00000300 --writer-video-data-root /data/ymdai/ember_data/LIBERO-datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a --writer-video-condition correct`。
- step500 command仅将output、`--gpu-indices 2,3`与Writer checkpoint改为对应step500路径。两臂各400 rollouts，raw rows必须唯一、38个cost-balanced shards必须完整、10 workers均exit0；结果按相同task/init/policy/video seeds配对，任何partial queue只能按canonical resume恢复。

## bias-restored AS首轨迹closed-loop与scheduler修正（2026-07-23）

- decay-6400轨迹的step300/500/800完整8×50 correct-video结果已完成，分别为`62/400`、`77/400`、`80/400`，raw rows均为400且无worker error。step800用GPU0–3、每卡5 replicas和32个cost-balanced shards，wall `1282.51s`、吞吐`0.31189 rollouts/s`；动态队列先按实际四卡拆Long shards，再由全部workers接普通任务。
- step300/500/800 results SHA256依次为`3c2643cf...fa8334`、`db01087c...4fce6`、`f2ef8786...8af10`。step500与800 paired flips为27/30，说明两者闭环持平；functional val loss的连续回升适合收缩候选区间，但不能替代完整closed-loop选best。
- 审计发现四卡global batch64沿用了warmup100，却把旧八卡global batch128的decay1200错误拉长到6400。正确query-equivalent日程是warmup100/decay2400；现有step500–800长期停在peak LR附近，因此该首轨迹保留为scheduler-confounded provenance，不用于否定bias或Action-Memory架构。
- canonical配置现只修正scheduler/阶段合同：total/decay为2400，fresh首段step1200，必要时exact-resume到1800/2400；每100步仍由同一驻留模型原地运行封存512-row validation-loss panel。配置SHA256为`cf031d79b6273ba71f3b5969a491cd6cf5d13d9c17a85e989cfc1c2d8f6ac69f`；聚焦验证`15 passed`、全仓fresh验证`172 passed`，下一步commit/push与live preflight后在GPU0–3启动不存在的新root。

## query-equivalent四卡AS fresh launch contract（2026-07-23）

- scheduler-only修正已在commit `43608a7e8d48e0f03a425d32a6b9152c67dfff0f`验证并推送；训练架构、bias、source checkpoint、24-task数据、sampler、optimizer、loss和validation panel均未改变。新run必须fresh identity，不能从decay-6400轨迹resume。
- output为`/data/ymdai/outputs/ember/pi05_as_writer_action_memory_bias_qscaled_r4_seed7_s1200_20260723`，log为`/data/ymdai/logs/ember/pi05_as_writer_action_memory_bias_qscaled_r4_seed7_s1200_20260723.log`，tmux session为`ember_as_bias_qscaled_r4_s1200`；三者启动前必须不存在。4 ranks只绑定物理GPU0–3，每卡一个同角色训练进程；GPU4–7不进入visible set。
- 每100步保存checkpoint并在同一驻留进程运行512-row task-balanced validation loss；首段stop为1200。checkpoint按既有实测约124.4MB估算，12份加原子临时副本、metrics和validation rows低于2GB，远低于500GB个人cap。
- exact command：`PYTHONPATH=/data/ymdai/worktrees/EMBER-as-valdiag-r6/src CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false /data/ymdai/projects/EMBER/.venv/bin/torchrun --standalone --nproc-per-node=4 scripts/train_as_writer.py --config configs/pi05_as_writer_action_memory_v1.json --mode formal --source-run /data/ymdai/outputs/ember/pi05_source_base_v1_seed7_1k_e2cc238_20260722 --checkpoint /data/ymdai/outputs/ember/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000 --tokenizer-path /data/ymdai/ember_data/openpi/paligemma_tokenizer.model --data-root /data/ymdai/ember_data/LIBERO-datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a --output-dir /data/ymdai/outputs/ember/pi05_as_writer_action_memory_bias_qscaled_r4_seed7_s1200_20260723 --stop-after-step 1200 --num-workers 2 --log-every 10`。

## query-scaled四卡AS早停与closed-loop候选（2026-07-23）

- fresh run root为`/data/ymdai/outputs/ember/pi05_as_writer_action_memory_bias_qscaled_r4_seed7_s1200_20260723`。step100–800 validation loss为`0.135237/0.141384/0.135191/0.134058/0.134964/0.135579/0.141342/0.139462`；step400后形成持续上升区间，故在step800完整checkpoint/val后早停。
- step800训练elapsed为`1951.68s`，累计51,200 queries；最大allocated/reserved为`79,540,023,808/82,288,050,176` bytes，loss/gradient均finite。run contract、当前完整metrics和validation metrics SHA256依次为`b5053300...e978347`、`2279f960...18461e`、`4d5936b0...722079`；latest checkpoint仍严格指向step800。
- 中断前step801–809已完成并保留在metrics中，但没有checkpoint也不进入候选；不删除或改写该证据。step300/400/500/600各用两张物理GPU、每卡5个persistent replicas，step800用四卡同样每卡5 replicas；所有run均有400条唯一rows、全部shards完成、workers exit0。
- step300/400/500/600/800完整correct-video success依次为`57/91/86/87/88`，results SHA256依次为`57ea1be8...e8179`、`07847743...99197`、`60b41c56...1b35d`、`6cb2d5af...4768`、`70b33252...74745`。对应wall为`2190.35/2135.77/2197.09/2146.22/1231.29s`；四卡step800有效吞吐`0.32486 rollouts/s`，两卡runs约`0.182–0.187`。
- step400相对300/500/600/800的paired `400-only/other-only`为`49/15`、`33/28`、`26/22`、`29/26`；只有300显著较差。val-loss最低点与真实峰值同在400，但五点总体相关性很弱，因此在线loss只负责趋势早停和候选筛选。该轨迹best `91/400`显著低于旧rank128 SFT的`122/400`（paired `25/56`，`p=7.52e-4`），下一步先完成SFT真实ceiling，再对AS做训练统计效率修正。

## rank128 Source-SFT四卡profile与正式首段合同（2026-07-23）

- profile root为`/data/ymdai/outputs/ember/pi05_source_sft_rank128_profile_r4_b128_22fae58_s4_20260723`；仅使用GPU0–3，一卡一rank，batch128/rank、global512。4步全部finite，稳定吞吐约`36.18 queries/s`，峰值allocated/reserved为`54.998/67.979GB`。
- profile artifact约61MB；正式run到step800保留8个约61MB checkpoint、在线validation rows及原子临时副本，峰值新增低于1GB。当前个人占用约259GB，远低于500GB cap。
- 配置已封存首段step800、每100步checkpoint与驻留512-query val-loss monitor；允许的后续stop为`1100/1400/1700/2000/2300/2400`。val-loss连续下降则续训，连续上升且train loss仍降则停止；单点不决策，完整8×50 closed-loop覆盖loss选择最终best。

## AS condition-balanced累计实现待profile（2026-07-23）

- canonical `as_step.py`现支持同一generated adapter跨多个policy microbatches复用；128 queries按`16×8`顺序执行、按chunk实际样本数加权loss/adapter-gradient，最后只反传一次Writer。新增聚焦测试覆盖尾部不等长切片和加权梯度等价性；既有normal/contrast owner、checkpoint和sampler cursor未分叉。
- owner澄清batch size不应成为方法门槛后，候选profile收敛为四卡每rank顺序2个独立conditions、每condition 16 queries，即8 conditions/128 global queries；这恢复旧八卡AS的逻辑训练单位，不再机械匹配rank128 SFT的global512。配置为`pending_condition_balanced_profile`，formal仍不能启动；SFT正式进程期间只做代码和CPU测试，不抢占或改写其输出。

## 四卡rank128 Source-SFT在线早停与候选评测修复（2026-07-23）

- fresh formal root为`/data/ymdai/outputs/ember/pi05_source_sft_rank128_ceiling_r4_b128_af658c4_s2400_20260723`。step100/200/300/400的task-balanced validation functional loss依次为`0.133067/0.133336/0.134167/0.137131`，同期100-step train-loss mean为`0.138862/0.117804/0.109806/0.106153`；validation连续且加速回升而train继续下降，故在完整step400 checkpoint/validation后暂停，后续可从step400 exact-resume。
- 首次候选评测在任何rollout前被旧机械合同拒绝：adapter检查仍硬编码formal world-size为8，并要求整个训练run已有最终`run_summary.json`，与当前四卡协议及“暂停→评中间checkpoint→必要时续训”冲突。
- 最小修复改为读取sealed formal config中的`expected_world_size`；仅`development + validation`允许用已完整发布、manifest校验通过的formal checkpoint在run summary尚不存在时评测，并显式记录`published_checkpoint_before_run_completion`。seen/final/test仍要求完成run summary；相关Source-SFT与evaluator回归`51 passed`。

## Source-SFT四卡step100–800结果与继续训练（2026-07-24）

- 四卡fresh rank128 SFT的完整8-task×50 validation曲线为step100/200/300/400/500/600/700/800=`81/95/68/78/94/99/108/97`。step700是该轨迹当前best，但600/700/800之间的paired差异均未形成明确峰后持续下降；旧八卡step400的`122/400`仍是全局SFT incumbent。
- 旧八卡8×64与当前四卡4×128都为global batch512，所以同一step的optimizer updates与总queries相同；两个step400 checkpoint也都实际记录`204,800` examples，每task覆盖范围仅相差一个128-example小批，确认训练量大体可比。每次更新内task小批数量不同只作为次要梯度方差信息。四卡step800的updates和queries已经是旧step400的两倍，不能因condition visits相同就称为等价点。后续仍记录拓扑与条件覆盖，但不再因GPU数量或batch变化机械缩放step或从零训练。
- 当前四卡run已从完整step800 checkpoint在相同四卡合同下exact-resume到step1100，首个新finite metric为step802、loss `0.0867007`、gradient norm `0.0237412`、吞吐`36.26 queries/s`。step900/1000/1100保存后仍以完整closed-loop决定是否继续；functional val loss只作微弱参考。
- AS累计实现最终采用每rank每optimizer update处理2个独立task/video conditions、每condition 16 queries；四卡每update合计8 conditions/128 queries，与旧八卡AS一致。此前每condition64/global512仅是未启动的SFT-batch-matched提案，已在owner澄清后退役。该设置用于恢复条件覆盖与梯度稳定性而非把batch设为科学门槛，真实profile后优先warm-start现有best。
- canonical AS runner新增显式`--initialize-writer-checkpoint`阶段初始化：可从已封存且source/authority/Writer/LoRA完全兼容的最佳Writer权重启动新优化阶段，同时重新初始化optimizer、scheduler、sampler和RNG；run contract记录源checkpoint、源step及三类hash，并明确标为warm-start而非exact-resume。这样GPU数或训练统计合同变化时无需重跑0→best，也不伪造跨合同exact-resume。旧qscaled step400 checkpoint已通过只读manifest/architecture/authority兼容检查；是否采用仍由真实低方差profile后决定。
- evaluator新增受控`6 replicas/GPU, OMP=1`运行profile；它只扩展同一dynamic-queue/persistent-policy owner，不改变task、seed、rows或结果聚合。按既有3 replicas约31GB估算6 replicas仍低于80GB，但必须由下一轮900/1000 checkpoint正式评测实测显存稳定性与有效rollout/s；若OOM或吞吐不优于5，立即退回5并保留failure/profile证据。

## Action-Memory temporal-RoPE快速实验合同（2026-07-24）

- owner要求快速验证新的时间聚合架构：保留有bias的冻结PaliGemma/Action Expert memory/Meta-LoRA/完整rank16 LoRA owner，只将乘性time gate和单一对称pool替换为实际采样位置上的1D RoPE、4个condition-only learned temporal memory queries；不加入transition token、order auxiliary loss、contrast或shared adapter。
- 真实GPU0–3 profile使用一rank一condition、16 queries/rank、global64且无梯度累计。两步loss/gradient均finite，step wall为`3.8037/1.9552s`，峰值allocated/reserved为`76,119,387,136/78,928,412,672` bytes；Writer参数`11,252,737`，为rank128 Source-SFT参数的`1.092781×`。
- fresh formal只训练500 optimizer steps，保存step400/500；两者使用同一paired 8-task×50 validation panel，随后只对observed-best做视频、单帧、倒序与打乱诊断。output为`/data/ymdai/outputs/ember/pi05_as_writer_action_memory_rope_mem4_native_r4_seed7_s0500_20260724`；两份checkpoint和原子临时副本预计新增低于0.5GB，启动前个人占用`278,289,612,800` bytes，峰值远低于500GB cap。
- exact command：`PYTHONPATH=/data/ymdai/worktrees/EMBER-as-valdiag-r6/src CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false /data/ymdai/projects/EMBER/.venv/bin/torchrun --standalone --nproc-per-node=4 scripts/train_as_writer.py --config configs/pi05_as_writer_action_memory_v1.json --mode formal --source-run /data/ymdai/outputs/ember/pi05_source_base_v1_seed7_1k_e2cc238_20260722 --checkpoint /data/ymdai/outputs/ember/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000 --tokenizer-path /data/ymdai/ember_data/openpi/paligemma_tokenizer.model --data-root /data/ymdai/ember_data/LIBERO-datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a --output-dir /data/ymdai/outputs/ember/pi05_as_writer_action_memory_rope_mem4_native_r4_seed7_s0500_20260724 --num-workers 2 --log-every 10`。

## temporal-RoPE Writer快速子任务完成（2026-07-24）

- 实现commit `182a038`：canonical bias-enabled Action-Memory temporal owner原位加入frame-index RoPE与4个condition-only memory queries；聚焦Writer tests为`41 passed`。四卡native-global64 profile通过后，fresh formal root `pi05_as_writer_action_memory_rope_mem4_native_r4_seed7_s0500_20260724`完成500 steps，body wall `1188.6s`，step400/500 checkpoint均保存完整四rank RNG/optimizer/scheduler/sampler state。
- step400/500在线512-query validation functional loss为`0.1364674/0.1369167`。正式8×50 correct-video validation使用commit `d2997ed`的`per_sample_lora_batched_replan`，不是旧`materialized_per_rollout_sequential_replan`。6 replicas/GPU在Writer视频编码阶段OOM且0 rows，作为failure evidence保留；稳定拓扑为4 replicas/GPU、8 envs/replica。
- step400 root `pi05_as_writer_rope_mem4_batched_val8x50_step0400_correct_d2997ed_g01_r4_20260724`：`108/400`，wall `1712.444s`，`0.233584 rollout/s`，results SHA256 `163b0df72a14523ada233a8f846c1eaac9fac2cc6db9df822af47f8a5cff6d81`。
- step500 root `pi05_as_writer_rope_mem4_batched_val8x50_step0500_correct_d2997ed_g23_r4_20260724`：`98/400`，wall `1764.352s`，`0.226712 rollout/s`，results SHA256 `f8e0e1b299ac2b468cd0f46945f27c39daa25b7da1e0908b3922be7c8a4087c0`。paired step400-only/step500-only为`24/14`，故冻结step400。
- evaluator优化将此前AS约`36.6min/0.182 rollout/s`改善为最佳run的`28.5min/0.234 rollout/s`；旧sequential失败目录无有效rows，不参与科学结果。优化commit `d2997ed`及诊断commit `85962bf`均已推送`origin/main`。
- step400视频/帧特异性artifact为`pi05_as_writer_rope_mem4_step0400_video_frame_specificity_85962bf_20260724/diagnostic.json`，SHA256 `4f8110c1bf719d2ff07b220b5965af8d98d818ad7ae5e85c95c3216dc03a9316`，wall `173.316s`。跨suite错误视频、同task另一demo、倒序、乱序的有效LoRA相对变化中位数分别为`0.2267/0.0403/0.00937/0.00699`；单首/中/末帧为`0.1745/0.1124/0.3339`。
- 快速子任务到此按owner要求停止：新Writer对视频任务内容有明确特异性，但对时间顺序仍近似不敏感；correct峰值`108/400`未超过rank128 Source-SFT incumbent `122/400`。本轮不启动contrast、额外AS训练、SFT或RL。

## Action-Forecast Writer v1设计封存（历史，2026-07-24）

- owner决定将实现和实验交给新的独立session；当前session停止改代码和GPU
  launch，没有产生半成品实现，也没有修改正在运行进程的import/config/output。
- 当时的v1设计、参数预算、退役边界、profile矩阵、AS分段训练和RL cold-start
  口径已执行并由后续结果封存；相关旧辅助文档现已被2026-07-25 canonical
  design覆盖并删除，不再作为活动入口。
- 交接只读快照：main/`b78584ab05e7f639cf1c022fdf457b3a971d64e6`
  当时clean且等于origin/main；GPU0–3空闲，GPU4–7为其他用户进程；
  `/data/ymdai`占用`278,857,052,160` bytes。新session必须重新核验所有live
  状态，不能把该快照当launch许可。
- 最终交接审计确认旧tmux `ember_as_bias_r4_s3200`只剩空bash且没有训练/eval
  子进程，随后已删除该空session；main外15个历史worktree均clean且无活跃写
  进程，因此全部保留provenance。新session在main clean且无并发writer时直接
  使用main，不因历史worktree数量另造平行canonical路径。
- owner要求确认新session无需读取旧对话即可实现后，又对交接文档做逐层完整性
  审计：补齐了端到端tensor shapes、single-agentview信息墙、state-width128
  coordinate-query head、连续state token在PaliGemma文本state位置的插入方式、
  冻结backbone仍需保留的梯度路径、per-condition flow noise、Plan/Revision
  精确MLP/attention聚合、query identities、8个factor heads及真实输出宽度、
  长Long-shard动态调度、近似候选复测规则和已封存`122/400` SFT artifact。
- owner进一步纠正比较口径：`122/400`来自旧八卡rank128 SFT，不是四卡成绩。
  四卡step100–1100为`81/95/68/78/94/99/108/97/95/104/94`，best为step700
  `108/400`。当前AS硬比较是“不明显落后于108”，超过122为stretch。
- owner随后明确禁止把“多个峰后点略低”的判断套给Writer：AS/RL都必须在
  validation找到best，并在其后看到幅度非常明显、明显超过rollout噪声、由
  多个tasks贡献且独立panel复测后仍成立的下降趋势；否则继续训练。
- 当时还发现旧启动提示错误要求8卡、重做Phase A/source base、RL零warm-up和
  AS约2小时上限；该提示随后又被后续架构演进反复覆盖，现已删除。当前口径只
  由根authority和`docs/action_forecast_writer_design.md`定义。

## Action-Forecast Writer实现与formal训练收口（2026-07-24）

- 新session已将旧Action-Memory活动owner原位替换为唯一Action-Forecast路径：
  imagined-state/PaliGemma融合、Writer内部VL和Action Meta-LoRA、每帧完整
  10-step flow action plans、同绝对时刻Plan/Revision tokens、变长temporal
  Transformer及单向LoRA query decoder均已接入AS训练、checkpoint、inference和
  canonical evaluator。旧source/config/schema/tests及独立specificity runner
  已删除；历史文档和artifact只保留provenance，不再是可执行入口。
- Writer真实训练参数为`10,161,217`，相当于rank128 Source-SFT
  `10,297,344`的`98.68%`；生成的public task adapter保持完整38-target
  rank16合同，共76 tensors、`1,287,168` scalars，初始化为严格functional
  identity，source policy trainable parameter count为0。
- GPU0–3真实profile已封存stride5、frame-microbatch32、每rank batch16。
  17-step长profile覆盖全部24个train tasks与1088 action queries，steady
  step中位数/p95为`6.1183/9.0442s`，吞吐中位数`10.4611 queries/s`，最大
  allocated/reserved为`67.08/70.18GB`。frame-microbatch64令rank1达到
  `80,821/81,920 MiB`并失去前进，已拒绝；owner决定不再扩测stride10。
- profile checkpoint完成step1→2 exact-resume；flow-noise cursor、sampler、
  optimizer/scheduler及各rank RNG均恢复，contract SHA256为
  `c7a3dc88ae840d386b9d825e6f71f2f9613fccf0f37adf85b29c5a577d0ecd68`。
  两组提交前focused tests分别为`30 passed`和`51 passed`，相关Python模块
  `py_compile`及`git diff --check`通过。
- formal AS配置现为四卡、每rank batch16、每75 steps checkpoint、每300 steps
  exact-resume segment。首段之后只先评测step150/300 correct-video；owner明确
  撤销前置的单卡最小顺序诊断，wrong/shuffled/reversed只能在充分训练、找到
  validation observed-best且验证明显稳健峰后下降之后，对最佳checkpoint执行。

## Action-Forecast AS首段与两阶段evaluator（2026-07-24）

- fresh formal root
  `/data/ymdai/outputs/ember/pi05_action_forecast_as_development_seed7_49cef59_r4_s5_fm32_b16_20260724`
  已完成step0→300：300条metrics、19,200 action queries、1,200独立video
  conditions，训练body wall `2022.169s`；step wall中位数/p95为
  `6.1408/9.1705s`，loss范围`0.089996–0.186392`、最大grad norm
  `1.845997`，峰值allocated/reserved为`67,092,966,912/
  70,185,385,984` bytes，全部finite。
- step75/150/225/300四个约124.8MB checkpoint均通过完整manifest与file SHA
  核验。512-query在线functional monitor依次为
  `0.137364/0.133570/0.137465/0.134575`，只作弱候选信号，不作为closed-loop
  selection或停止依据。
- step150耦合式correct screen仅作实现provenance：GPU0–1 r3和GPU2–3 r4各跑
  16 episodes，分别为`1/16`，execution window约`94.3/100.5s`，显存约
  `40/53GB`每卡；小分母不作科学解释，也不能用于确定rollout replicas。
- owner指出LoRA生成和rollout并发必须解耦、共同模型应尽量常驻复用。现已在
  唯一`evaluate_pi05.py`内实现两阶段cache/handoff：generator数量和batch独立，
  cache完成后原generator进程只释放Writer并保留source π0.5直接rollout，随后
  再启动额外rollout-only workers；相同cache可跨rollout replica profile复用。
- cache身份包含完整adapter/model/task-state与生成batch/grouping但排除rollout
  replicas；逐entry safetensors+evidence以目录原子发布，final manifest核验
  coverage、LoRA SHA及file SHA，完整400-entry panel估计约1.03GB。结果新增
  rollout-only吞吐，同时保留end-to-end wall。
- 结构门无hard violation（`REVIEW`仅保留既有大函数与可接受的新模块审阅项）；
  新逻辑收口在`src/ember/pi05_eval/`和`src/ember/writer/evaluation_*`，没有
  平行runner。全仓fresh验证为`177 passed`，`py_compile`、`git diff --check`
  通过。下一步commit/push后在GPU0–3分别profile生成batch与纯rollout replicas。

## Action-Forecast AS step300→600与正式validation（2026-07-24）

- formal AS从完整step300 checkpoint同合同exact-resume到step600；本段300
  optimizer steps消费19,200 queries和1,200 video conditions，body wall
  `1978.45s`，step wall中位数`6.0985s`。step375/450/525/600四个checkpoint
  均含Writer、optimizer/scheduler、sampler/data cursor与四rank RNG；累计到
  step600为38,400 queries和2,400 video conditions，loss/gradient均finite。
- step150/300/450/600的完整correct-video validation依次为
  `75/99/93/118`。step600逐任务为Long `13/2`、Goal `1/34`、Object
  `46/17`、Spatial `0/5`；32/32 shards、400 rows、118 successes、所有24
  workers exit0且无重试。results SHA256为
  `bf3c98dc9a9df0e067b6589d7627b02863197528555ac6c11964799dfd7733be`。
- step450→600 paired为450-only `29`、600-only `54`（exact
  `p≈0.00804`），多个tasks共同贡献净提升。step600成为新observed-best并超过
  四卡Source-SFT的`108/400`；尚未出现任何峰后下降，故不做特异性诊断、不停训。
- commit `493917e`将cache identity原位改为可见视频级去重，并让aliases复用同一
  LoRA tensor与Writer生成随机流；全仓fresh测试`182 passed`。正式400-episode
  panel现为259个唯一LoRA和141个aliases，生成约53–56秒；每卡6 replicas稳定，
  step600 end-to-end/rollout-only吞吐分别为`0.45836/0.61045 episode/s`。
- 启动下一段前GPU0–3全空闲、GPU4–7仍为其他用户进程且未进入visible set；
  main/`493917e` clean并等于origin/main，个人占用约285.6GB。现已从step600
  启动四卡exact-resume到step900，仍按75步保存并优先正式评测step750/900。

## Action-Forecast AS step600→900与继续训练（2026-07-24）

- 四卡从step600 exact-resume到900，300步body wall `1994.75s`，step wall
  中位数/p95为`6.1495/9.0615s`；本段mean functional loss `0.114818`、
  最大grad norm `0.360586`，峰值allocated/reserved为
  `67,084,895,744/70,176,997,376` bytes，全部finite。
- step675/750/825/900在线functional monitor为
  `0.135046/0.134811/0.134446/0.134562`，仅显示弱平台。step900 checkpoint
  逐文件SHA核验通过，累计57,600 queries、3,600 video conditions；24 tasks
  各150次视频访问且均覆盖全部50 videos，完整optimizer、sampler与四rank RNG
  可恢复。
- step750/900正式validation为`104/113`，均为32/32 shards、400 rows、24
  workers exit0。600→750下降14但paired `p≈0.0814`；900随后回升9，且600与
  900 paired `p≈0.6254`。因此step600仍是observed-best，但没有满足owner停止
  条件的峰后下降，不能做特异性诊断。
- step750/900 results SHA256分别为
  `584a5c2164b631eb96fc6d60589720ad4ad297626ac750548b78b953c664ea22`
  和`4c1d62d0b3fbc847b776cdbcce0558d502b12a70381fa7f72d0913112d32a1cf`。
  在GPU0–3实时空闲、GPU4–7仍隔离、个人占用约287.4GB时，已从step900启动
  下一段exact-resume到step1200，继续优先评测step1050/1200。

## Action-Forecast AS step900→1200与新observed-best（2026-07-24）

- 四卡从step900同合同exact-resume到1200；step975/1050/1125/1200在线
  functional monitor为`0.134745/0.134612/0.134946/0.134434`，只显示微弱
  摆动。step1200累计76,800 queries、4,800 video conditions，24 tasks各
  3,200 queries与200次视频访问且50/50 videos全覆盖；checkpoint全部文件SHA
  校验通过。
- step1050/1200正式8×50 correct-video validation为`117/125`，均为32/32
  shards、400 rows、24 workers exit0、无错误/重试。step1200逐task为Long
  `6/3`、Goal `1/38`、Object `45/20`、Spatial `1/11`，7/8 tasks非零。
- step600→1050 paired flips为`31/30`、exact `p=1.0`；step600→1200为
  `31/38`、净`+7`；step1050→1200为`15/23`、净`+8`。step1200成为新的
  AS observed-best `125/400`，因此不存在持续峰后下降，不能停止或提前做
  wrong/shuffled/reversed诊断。
- step1050/1200 results SHA256为
  `b88303cbf2a170315a1d5523f58cb1b0b3346d4671a9e37f024a0dda23f339a7`
  和`c575591ba36d949578061aa164f59572fcd59c81952a3f301c4c66b4afd38dd0`；
  rollout-only吞吐为`0.62721/0.60638 episode/s`。两次均按视频级cache只
  生成259个唯一LoRA，generator batch100与每卡6 rollout replicas解耦。
- GPU0–3再次实时空闲、GPU4–7仍为其他用户进程且未触碰、个人占用约289.8GB
  时，已从step1200 exact-resume到1500；仍按75步保存并正式评测
  step1350/1500。若没有出现明显、多个tasks共同贡献且独立复测成立的峰后
  下降，继续下一段而不设总wall-clock上限。

## Action-Forecast AS step1200→1500与继续探索（2026-07-24）

- 四卡从step1200同合同exact-resume到1500，本段wall `2019.83s`；
  step1275/1350/1425/1500在线functional monitor为
  `0.135102/0.134932/0.134596/0.135012`，仍只是弱平台。step1500累计
  96,000 queries、6,000 video conditions，24 tasks各4,000 queries与250次
  视频访问且50/50 videos全覆盖；最终checkpoint逐文件SHA验证通过。
- step1350/1500正式8×50 correct-video validation为`120/119`，均为32/32
  shards、400 rows、24 workers exit0、无错误/重试。逐task分别为Long
  `10/2, 8/1`、Goal `0/32, 1/33`、Object `43/19, 43/18`、Spatial
  `0/14, 0/15`。
- 相对step1200，1350/1500的paired净差仅`-5/-6`，exact
  `p≈0.6029/0.4614`；1350→1500净差`-1`、`p=1.0`。因此
  `125→120→119`只是两个略低且彼此持平的点，不满足强峰后下降停止条件。
- step1350/1500 results SHA256为
  `edf5b889eb4d6fdc0da9554966f97e8f9e5417cae250597526b2da7336337327`
  和`1a9232906b30d1d2ae679d8b726f332af5279aaff4a8e2ea1d8873981c035cc5`；
  rollout-only吞吐为`0.60530/0.63953 episode/s`。r6完成两次全panel，
  step1350瞬时显存约80.6GiB但未OOM。
- GPU0–3实时空闲、GPU4–7仍隔离、个人占用约291.6GB时，已从step1500
  exact-resume到1800；继续正式评测step1650/1800，仍不提前做specificity。

## Action-Forecast AS step1500→1800与继续充分探索（2026-07-24）

- 四卡从step1500同合同exact-resume到1800，本段wall `2051.35s`；
  step1575/1650/1725/1800在线functional monitor为
  `0.135525/0.134902/0.134651/0.134745`，仍是弱平台。step1800累计
  115,200 queries、7,200 video conditions，24 tasks各4,800 queries与300次
  视频访问且50/50 videos全覆盖；完整checkpoint和四rank恢复状态已核验。
- step1650/1800正式8×50 correct-video validation为`120/114`，均为32/32
  shards、400 rows、24 workers exit0、无错误/重试。逐task分别为Long
  `6/3, 4/2`、Goal `0/33, 1/34`、Object `43/21, 45/17`、Spatial
  `0/14, 0/11`。
- 相对step1200，1650/1800 paired为`29/24`与`31/20`，净`-5/-11`，
  exact `p≈0.5831/0.1608`；1650→1800为`30/24`，净`-6`、
  `p≈0.4966`。该幅度未远超400-rollout噪声，任务方向也混合，不能把
  `125→120→119→120→114`解释成充分确认的峰后下降。
- step1650/1800 results SHA256为
  `e800361b3bcdf57d57f39f635b20136f043a73d80197560098f0e087b5c35f9a`
  和`5c0de70f6b75c63d332e6e6e35ece5f2f4a57041cf364111123b6d73f61654d3`；
  rollout-only吞吐为`0.61633/0.61129 episode/s`。两次均以batch100生成
  259个唯一视频LoRA，随后每卡6 replicas稳定完成。
- GPU0–3再次实时空闲、GPU4–7仍为其他用户进程且未触碰、个人占用约274GB
  时，已从step1800 exact-resume到2100；下一步正式评测step1950/2100，
  specificity继续推迟。

## Action-Forecast AS step1800→2100与再次续训（2026-07-25）

- 四卡从step1800同合同exact-resume到2100，本段wall `2033.67s`；
  step1875/1950/2025/2100在线functional monitor为
  `0.134469/0.134724/0.135176/0.134929`，仍只有弱摆动。step2100累计
  134,400 queries、8,400 video conditions，24 tasks各5,600 queries与350次
  视频访问且50/50 videos全覆盖；最终checkpoint文件逐SHA与manifest一致。
- step1950/2100正式8×50 correct-video validation为`110/114`，均为32/32
  shards、400 rows、24 workers exit0、无错误/重试。逐task分别为Long
  `4/0, 4/0`、Goal `1/28, 1/32`、Object `45/19, 44/14`、Spatial
  `0/13, 0/19`。
- step1200→1950 paired为`34/19`，净`-15`、exact `p≈0.0534`，但主要由
  Goal-6贡献；step1200→2100为`36/25`，净`-11`、`p≈0.2000`。
  step1800→2100恰为`28/28`、净`0`，1950→2100反而净`+4`。所以
  `125→…→114→110→114`仍未建立明显、持续、多task共同贡献的峰后下降。
- step1950/2100 results SHA256为
  `c62e75973b8196e4e6052cecde8e0add00dd948f0536385ac5be44d0a158a576`
  和`934382c211027c3b6407b46898e65f708dfda136a1bc6cbef8a60f18cacf3905`；
  rollout-only吞吐为`0.61188/0.61723 episode/s`。两次均以batch100生成
  259个唯一视频LoRA，随后每卡6 replicas稳定完成。
- GPU0–3再次实时空闲、GPU4–7持续隔离、个人占用约276GB时，已从step2100
  exact-resume到2400；继续正式评测step2250/2400，仍不提前做specificity。

## Action-Forecast AS step2100→2400与峰值平台回访（2026-07-25）

- 四卡从step2100同合同exact-resume到2400，本段wall `2037.47s`；
  step2175/2250/2325/2400在线functional monitor为
  `0.135174/0.134400/0.134857/0.134759`，仍是弱平台。step2400累计
  153,600 queries、9,600 video conditions，24 tasks各6,400 queries与400次
  视频访问且50/50 videos全覆盖；最终checkpoint完整保存。
- step2250/2400正式8×50 correct-video validation为`123/111`，均为32/32
  shards、400 rows、24 workers exit0、无错误/重试。逐task分别为Long
  `5/3, 8/0`、Goal `0/34, 0/36`、Object `45/20, 43/18`、Spatial
  `1/15, 0/6`。
- step1200→2250 paired为`30/28`，净`-2`、exact `p≈0.8957`，2250实质
  追平observed-best；step1950→2250反而净`+13`。step1200→2400为
  `32/18`、净`-14`、`p≈0.0649`，2250→2400为`32/20`、净`-12`、
  `p≈0.1263`，但这是接近峰值后的一次单点回落，且task方向混合，不能停止。
- step2250/2400 results SHA256为
  `35ff55e3f8c2a5f8ed8885cf2a335862879b255189907a098e63d7ad61525655`
  和`f5c9a77b40048e6826a8b667c887e6d796c14f71be68fe9d9a249329bdc036df`；
  rollout-only吞吐为`0.60369/0.61011 episode/s`。两次均以batch100生成
  259个唯一视频LoRA，随后每卡6 replicas稳定完成。
- GPU0–3再次实时空闲、GPU4–7持续隔离、个人占用约277GB时，已从step2400
  exact-resume到2700；继续正式评测step2550/2700，specificity继续推迟。

## Action-Forecast AS observed-best特异性门与顺序修正profile（2026-07-25）

- step2550正式correct-video完成`124/400`，再次回到step1200的`125/400`
  峰值平台；step2700 checkpoint完整存在但尚未评测。owner随后要求先对当前
  最高AS做视频特异性，若通过再推进RL。
- step1200四个同seed、同state、同policy-RNG的8×50 arms全部完成32/32 shards、
  400 rows、24 workers exit0且无重试：correct/cross-suite-wrong/shuffled/
  reversed=`125/67/121/124`。correct-vs-wrong paired为`71/13`、
  `p=7.8639e-11`，内容特异性通过；correct-vs-shuffled/reversed分别为
  `17/13`、`15/14`，顺序特异性失败。故没有越过RL硬门槛。
- canonical `src/ember/writer/as_step.py`现原位支持一个最小order-contrast训练
  mode：正例与shuffle/reverse负例共享物理action batch、policy language与
  Writer flow noise，两个functional forward串行执行以保持峰值显存；负例只在
  loss低于`correct+0.01`时施加`-0.5`梯度。source policy仍为0 trainable，
  Writer输入与禁入信息不变，没有新增runner或恢复Action-Memory。
- 从step1200 Writer权重、fresh optimizer/scheduler/RNG启动的四卡2-step真实
  profile已完成；batch16/rank、frame-microbatch32，双forward全局
  128 policy samples/step，峰值allocated/reserved为
  `67,077,086,720/69,250,056,192` bytes，首步/第二步
  `19.4614/11.6213s`，无OOM或非finite。focused tests为`13 passed`，
  config解析、`py_compile`和`git diff --check`通过。配置已封为正式首段
  300 steps；训练后只对新轨迹的validation候选选best，再在best上复测完整
  四arm特异性，通过前不启动RL。

## Action-Forecast Writer v2实现与正式前检查（2026-07-25）

- 按owner最新口径原位完成28-state-token、directed Revision bounded-gate、
  content-only LoRA decoder；删除order-contrast活动配置与`as_step.py`分支，
  schema/checkpoint/config统一升级到v2。focused tests `26 passed`，
  `py_compile`和`git diff --check`通过。
- Revision反事实诊断完成8 tasks×2 videos，未读actions/reward/outcome：
  新合成time-centered reversed/shuffled相对L2中位数
  `0.3554/0.2418`，旧Revision为`0.0281/0.0316`。
- GPU0–3实时空闲、个人占用`301,090,004,992` bytes、总盘可用约3.06TB时，
  运行fresh step1后从
  `/data/ymdai/outputs/ember/pi05_action_forecast_v2_profile_resume_r4_s5_fm32_b16_20260725/checkpoints/step_00000001`
  exact-resume到step2。contract SHA256为
  `5afbb65786f70ab67c131a78ca59959fde3284dd9bbbbb4932f35eec1ddc83a6`；
  四rank state、Writer、optimizer/scheduler与trainer state均在checkpoint
  manifest中逐文件封存。
- profile保持`stride=5`、`frame_microbatch_size=32`、
  batch16/rank；step2为`6.5025s`、全局`9.8424 queries/s`，峰值
  allocated/reserved为`67,088,471,040/69,235,376,128` bytes，无OOM或
  nonfinite。下一步提交并push唯一canonical代码/配置，再从fresh identity
  启动正式step0→600，checkpoint间隔75，完整评测step300/600。

## Belief-v3实现、效率选择与正式启动前封存（2026-07-25）

- 唯一canonical Writer已升级为Belief-v3：一个absolute-time token内concat
  Plan128/Revision128；Revision比较所有更早covering forecasts与最新Plan；
  Temporal和LoRA query decoder均为content-only、zero-preserving路径。
- owner最终取消所有人工Revision强度尺度。活动公式为
  `Revision=stopgrad(raw source-normalized residual RMS)*RMSNorm(direction)`；
  routing strength也detach，`tau`与分位数只作诊断、不参与前向。
- 固定GPU0–3和stride5的效率profile选择frame-microbatch32、batch20/rank。
  12-step参考output为
  `/data/ymdai/outputs/ember/pi05_action_forecast_belief_v3_profile_r4_s5_fm32_b20_20260725`；
  稳态中位`6.4942s`、`12.3188 global queries/s`。fm40较慢，fm48在首步前
  达到`81,153/81,920 MiB`且失去稳定前进。
- 最终raw-RMS实现的fresh+resume output为
  `/data/ymdai/outputs/ember/pi05_action_forecast_belief_v3_rawrms_resume_r4_s5_fm32_b20_20260725`。
  fresh step1后从`checkpoints/step_00000001` exact-resume到step2；contract
  `352f7409d671d97399262b46afe0d415b4b6c145bcca66cbe43725474fa8e234`，
  resumed step `6.9184s`、`11.5634 queries/s`，峰值allocated/reserved
  `77,090,931,200/83,730,890,752` bytes，无OOM/nonfinite。
- checkpoint schema v3逐文件封存Writer、trainer/optimizer/scheduler及四rank
  RNG/sampler状态；flow-noise cursor从global visit 4准确推进到8。Writer
  `10,247,872` trainable parameters，source policy trainable count为0。
- focused测试、JSON解析、compile和diff检查通过后，下一步不再重做profile：
  提交/push当前唯一路径，实时复核GPU/storage，然后用同一配置从fresh identity
  一次连续训练0→600，每75步保存且不中途评测。
- step600顺序特异性先跑低成本内部数值诊断；只有normal/shuffled/reversed在
  effective LoRA等最终输出上已有明确、跨多个tasks/videos的稳定差异，才启动
  昂贵的paired validation arms。正常correct-video多checkpoint评测仍保留。

## Belief-v3正式step600与内部特异性failure packet（2026-07-25）

- commit `3363345`上的formal run
  `/data/ymdai/outputs/ember/pi05_action_forecast_belief_v3_as_development_seed7_3363345_r4_s5_fm32_b20_s600_20260725`
  已一次连续完成0→600；run contract SHA256为
  `afbdea64b3b660baaa7576bc544c37f44575b9e001715ebea5191726a65a5071`，
  run-summary file SHA256为
  `110b45d521d61a4b35e933906d733e6a749a86379ad48a5fa2945d01bef2fc50`，
  wall `4157.74s`。75/150/225/300/375/450/525/600八个checkpoint均存在；
  step600完整manifest校验通过。
- 8 tasks×2 videos正式schedule子集的paired内部诊断位于
  `/data/ymdai/outputs/ember/pi05_action_forecast_belief_v3_step0600_internal_order_val8x2_3363345_20260725`。
  Revision和time-centered Temporal均有明显顺序差异，但normalized query与
  effective LoRA分别只剩reversed/shuffled
  `0.0000719/0.0000448`和`0.000297/0.000169`相对L2，故内部gate失败。
- 两个8 tasks×1 video无训练反事实分别位于
  `pi05_action_forecast_belief_v3_step0600_bounded_counterfactual_val8x1_3363345_20260725`
  与
  `pi05_action_forecast_belief_v3_step0600_centered_memory_counterfactual_val8x1_3363345_20260725`。
  normalized-V/bounded-output不能解决；仅去除Temporal masked时间均值即可把
  query/effective LoRA差异恢复到`0.1053/0.0825`与`0.0543/0.0401`，
  定位为global constant遮蔽temporal innovation，而非Revision或query容量不足。
- owner要求完成特异性检查和归因后停下汇报。按先前两级门，未启动
  shuffled/reversed environment rollout；也未启动多checkpoint correct-video
  validation、后续AS续训、架构改写或RL。GPU0–3已释放，4–7始终未触碰。

## 32-token Visual-State canonical design已记录（2026-07-25）

- owner最终对齐的完整设计已集中记录在
  `docs/action_forecast_writer_design.md`：32-token native state anchor、
  初始帧锚点加非递归anchor/local有向变化、可学习identity-init双Meta-LoRA、
  future-action forecasts、Plan/Revision、单-token Belief、两层Temporal、
  content-conditioned query decoder和完整rank-16 LoRA。
- 旧Action-Forecast辅助提示和handoff文档已删除；根`AGENTS.md`、
  `README.md`、`docs/execution_brief.md`、
  `docs/decisions_and_open_questions.md`与`task_plan.md`的活动引用统一指向
  canonical design。旧v1/v2/v3结果继续留在findings/progress作为历史证据，
  但相关段落已明确标为历史，不再形成平行活动口径。
- 当前下一步是原位实现并做必要mechanical checks，然后固定stride5用GPU0–3
  fresh训练75 step，先完成低成本内部顺序与直接换视频特异性闭环。通过后才
  启动fresh 0→1200正式AS；未通过则按最早失效层级快速迭代，不使用contrast
  loss。

## 32-token Visual-State v4实现与profile（2026-07-25）

- canonical v4已原位实现：32个原生anchor tokens、8坐标的initial/anchor/local
  visual-state reader、可学习VL/Action Meta-LoRA、Plan/Revision单-token
  Belief、两层identity-safe Temporal、routing/content分离query decoder及完整
  rank-16 LoRA。旧v3 config/schema已退役。
- Writer实测`10,299,072`个训练参数，和rank128 Source-SFT
  `10,297,344`相差`1,728`（`0.017%`）；public LoRA仍为76 tensors、
  `1,287,168` scalars。focused CPU checks为20 passed。
- GPU0–3真实profile选择stride5、frame-microbatch32、batch20/rank。连续step2
  吞吐约`11.83 queries/s`，峰值allocated/reserved为
  `76,926,757,376/83,703,627,776` bytes，无OOM或nonfinite；现有reserved
  已无batch22或frame-microbatch40的安全余量，因此不做故意OOM试验。
- 75-step specificity训练保留正式1200-step scheduler时间轴，只把本次
  `selected_stop_step`设为75；不得把scheduler总步数压缩成75后冒充正式轨迹
  的前75步。
- step1 checkpoint恢复到step2后，loss、gradient norm、数据/视频/flow-noise
  游标与四rank RNG均匹配连续运行；rank-state文件bitwise一致。CUDA进程重启后
  Writer仅6个tensor出现最大`4.28e-8`的浮点差异，因此checkpoint完整可恢复，
  但不把跨进程CUDA计算误称为bitwise deterministic。

## 32-token Visual-State v4 step75特异性门（2026-07-25）

- 有效fresh轨迹位于
  `/data/ymdai/outputs/ember/pi05_action_forecast_v4_gate75_fb280b3_r4_s5_fm32_b20_20260725`；
  它保留正式1200-step scheduler时间轴，连续完成0→75，消费6000 action
  queries和300个task-video conditions，24 tasks各覆盖12–13条teacher
  videos。step50/75 checkpoint均完整发布，训练wall为`542.04s`。
- 8 validation tasks×2 reference videos×4反事实的内部诊断位于
  `/data/ymdai/outputs/ember/pi05_action_forecast_v4_step0075_internal_specificity_val8x2_fb280b3_20260725`。
  正确language、flow noise和order反事实的frame indices均固定；实际重算
  reversed/shuffled forecasts，action/reward/outcome reads均为0。
- reversed/shuffled在effective LoRA上的相对L2中位数为
  `0.0420/0.0468`，16/16 comparisons均非零，8/8 tasks均有贡献；旧Belief-v3
  failure只有约`0.000297/0.000169`。同task换demo为`0.0250`，cross-suite
  wrong为`0.0714`，直接换视频特异性同样成立。
- 差异没有在下游再次坍缩：reversed/shuffled从Belief
  `0.8217/0.7852`到Temporal `0.6902/0.6428`，query output仍有
  `0.0528/0.0593`，最终effective LoRA为`0.0420/0.0468`。Revision strength
  中位数分别增加约`11.9%/20.0%`，并由13/16与14/16视频对同向贡献。
- 该低成本门判定通过。step75尚不要求绝对correct success，环境paired
  rollout推迟到已有绝对能力的正式候选，避免低成功率地板把机制检查变成无效
  证据。下一步从fresh identity连续训练到1200。

## v4正式轨迹终止与现有checkpoint选择完成（2026-07-26）

- 正式v4 run
  `/data/ymdai/outputs/ember/pi05_action_forecast_v4_as_development_seed7_ad0db5f_r4_s5_fm32_b20_s1200_20260725`
  已完成step2400并停止；不再续训。2400-step run无OOM/nonfinite/error，
  step2400 checkpoint完整，训练过程共消费`192,000` policy samples。
- owner取消80-episode快筛。固定400 panel评测
  step675/825/900/1200/1275/1500/1875/2100/2400分别为
  `100/109/82/96/94/92/90/90/89`；现有observed-best为step825。
- step825 correct结果位于
  `/data/ymdai/outputs/ember/pi05_action_forecast_v4_as_formal_val8x50_step0825_correct_ad0db5f_g0123_gen1_b100_roll6_20260725`，
  `results.json` SHA256为
  `92434e9df8e25fdd85f4b09b8102c7410cce32c758e196df196ff6a025222a82`。

## v4 step825完整特异性评测完成并停止（2026-07-26）

- canonical evaluator新增`same_task_other`条件：实际teacher demo固定为paired
  correct demo的`+17 mod 50`，task/language/init/env/policy seeds及Writer
  flow/order随机性保持配对。400/400 rows均核验为同task、不同demo。
  当前fresh复核eval contract/runtime/cache tests为`34 passed in 4.98s`；
  实现已在commit `64af8b0` push到`origin/main`。
- step825内部16-reference特异性证据位于
  `/data/ymdai/outputs/ember/pi05_action_forecast_v4_step0825_internal_specificity_val8x2_ad0db5f_20260725`。
  effective LoRA相对L2中位数same/wrong/shuffled/reversed为
  `0.0955/0.8762/0.2598/0.3255`。
- 五个固定400结果为correct/same-task-other/cross-suite-wrong/shuffled/
  reversed=`109/104/99/148/126`。新增四臂output及`results.json` SHA256：
  - same：
    `/data/ymdai/outputs/ember/pi05_action_forecast_v4_as_formal_val8x50_step0825_same_task_other_64af8b0_g0123_gen1_b100_roll6_20260726`，
    `36be0c368f278ae1f36a863c672bf890566366f7c25e2b966f27fcc96aeb38f1`；
  - wrong：
    `/data/ymdai/outputs/ember/pi05_action_forecast_v4_as_formal_val8x50_step0825_cross_suite_wrong_64af8b0_g0123_gen1_b100_roll6_20260726`，
    `a5f302da57a8a6d19d102f6ac05e7f21249838221f10f251e94658cfcabf501e`；
  - shuffled：
    `/data/ymdai/outputs/ember/pi05_action_forecast_v4_as_formal_val8x50_step0825_shuffled_64af8b0_g0123_gen1_b100_roll6_20260726`，
    `d466374207e32adfdb33ccedee093bfc7bf3f8ff167bcb1f551d53ae710057db`；
  - reversed：
    `/data/ymdai/outputs/ember/pi05_action_forecast_v4_as_formal_val8x50_step0825_reversed_64af8b0_g0123_gen1_b100_roll6_20260726`，
    `d17c9d66aab8f4f46163e914ef64ffcb1b409d93fa151d2642b4ae8ab66bb101`。
- same-task other只净降5且行为churn最小；但shuffled显著净增39
  (`p=3.48e-5`)，reversed净增17，收益集中在object tasks。实际行为
  特异性硬门失败，当前不进入cold-start RL，也不修改架构或继续训练。
- 所有训练/评测进程已结束。GPU0–3实时均为`0 MiB`且无
  `train_pi05/evaluate_pi05`进程；4–7未触碰。`/data/ymdai`当前占用约
  `321.61 GB`，低于500GB cap。本轮按owner要求在记录、验证、commit、push后
  停止，等待后续讨论或外部专家意见。

## step825固定首帧shuffle快速归因完成（2026-07-26）

- owner授权一个scoped anchor ablation。commit `6b5923f`新增
  `shuffled_keep_first` canonical evaluator条件：复用原full-shuffle
  permutation，只把原始frame 0移回首位。eval contract/runtime/cache focused
  tests为`35 passed in 4.90s`，compile和diff检查通过，commit已push。
- GPU0–3预检均为`0 MiB`、0% utilization；个人占用约`321.61 GB`、预计新增
  `672 MB`，未触碰GPU4–7。固定400 run以4 generators、batch100、24 rollout
  workers一次完成，wall `864.49s`、有效`0.4627 rollouts/s`。
- 结果为`136/400`，逐task
  `9/1/0/45/45/26/1/9`。相对correct `109`为`18`条correct-only与
  `45`条keep-only，`p=8.98e-4`；相对full-shuffle `148`为`32`条
  full-only与`20`条keep-only，`p=0.126`。
- full-shuffle相对fixed-anchor直接净高12，且主要集中在Object-3；两项干预
  可能非线性交互，不能严格做因果加法分解。固定anchor后仍相对correct显著
  净增27，因此当前不再把随机anchor视为必要条件或主要根因；后续专家分析应
  优先审查非首帧order/local-transition/forecast-Temporal映射。
- run output：
  `/data/ymdai/outputs/ember/pi05_action_forecast_v4_as_formal_val8x50_step0825_shuffled_keep_first_6b5923f_g0123_gen1_b100_roll6_20260726`；
  results SHA256
  `0ec198d1438bdb85d9eccb41ac5f6796a470903b963576f29260c048b453ac99`。
  完成后GPU0–3均释放为`0 MiB`。

## 外部专家咨询材料已收敛（2026-07-26）

- 新增`docs/action_forecast_writer_expert_consultation.md`作为只能访问远程
  GitHub的专家唯一自包含入口，按“EMBER思想→全部关键架构演进→当前v4模块与
  完整结果→未解问题”组织，并附远程代码/配置/证据阅读路径。
- 文档嵌入source-base、各历史Writer、v4参数预算、step75内部量、step825
  fixed400逐任务/paired结果及fixed-anchor归因；不要求专家访问历史聊天或
  `/data/...`本地输出。
- README、`docs/execution_brief.md`和
  `docs/decisions_and_open_questions.md`已从旧“75→1200→600续训”未来式更新为
  当前事实：v4停止于2400、observed-best为825、行为特异性失败、RL暂停。
- 本次只整理远程可见的科学上下文，没有启动训练、rollout或新架构修改。

## 外部复核后的v4第一轮因果诊断（后被全面复审覆盖，2026-07-26）

- owner授权自主推进到“决定下一版架构”并明确后续只使用物理GPU4–7；0–3上
  他人进程未被停止、重置或干扰。本轮所有新增GPU launch均只把4–7放入
  `CUDA_VISIBLE_DEVICES`，最终阶段探针峰值reserved为
  `12,530,483,200` bytes/GPU。
- 新增本地一次性forecast-order transplant诊断，固定step825与16条validation
  references，完成`N→N/N→S/S→N/S→S`逐层和policy-function检查。summary：
  `/data/ymdai/outputs/ember/pi05_action_forecast_v4_step0825_forecast_order_transplant_val8x2_2fa1a1d_20260726`
  与
  `/data/ymdai/outputs/ember/pi05_action_forecast_v4_step0825_forecast_order_policy_function_val8x2x2_2fa1a1d_20260726`。
- 只对Object-1/Object-3各50 states运行新增四臂rollout，不做full400。结果
  `correct/S→N/N→S/S→S=49/47/72/82`，output为
  `/data/ymdai/outputs/ember/pi05_action_forecast_v4_step0825_forecast_order_cross_rollout_object13_2fa1a1d_20260726`。
- 完成Plan、Revision direction、value strength和Q/K routing四因子内部/
  policy-function交换，以及两个Object定向rollout。Plan-only/
  strength-only/direction-only/full-Revision=`61/54/67/75`；主要行为中介为
  Revision direction，strength单独与routing均非主因。
- 完成五条Object轨迹、25个经图像与gripper qpos核验阶段、12个LoRA反事实的
  action probe。异常主要改写pre-grasp/close/transport的end-effector
  translation；Revision=0会产生更大且常反向的动作变化，不能直接删除。
- 当时根据仍不充分的证据，过早把v4根因判定为未经识别的shared robot
  absolute-time forecast alignment及其Revision direction，并排除了
  visual-state；下一节的全面复审已撤回“唯一根因”和visual-state排除结论。
- 当时新增的文档（现已改名为
  `docs/action_forecast_writer_v4_root_cause.md`）曾拍板原位删除absolute-time
  Plan/Revision/Belief，改为256D frame-local Intent和adjacent ordered
  Transition；保留32-token visual-state、两个Meta-LoRA、两层content-only
  Temporal及decoder。该架构决定已被下一节撤回为局部候选；从未实现或训练。
- 当时的诊断summary SHA256仍作provenance；当前根因和未决合同以重写后的
  v5 decision文档及下一节为准。

## v4根因全面复审完成，旧v5决定撤回（2026-07-26）

- owner指出上一轮分析过早结束后，继续固定v4 step825并只使用物理GPU4–7完成
  更细粒度诊断；0–3上的他人进程未停止、重置或干扰。个人存储峰前占用约
  `305 GB`，低于500GB cap。
- 完成24 train tasks×4 demos的step75/300/825 hidden forecast semantics审计。
  teacher action/proprio只在inference完成后作measurement target，不进入Writer、
  optimizer或validation/test。summary SHA256依次为
  `99f341c2...b2baa`、`de5a4529...b763c`、`a1633aa5...edb4bf`。
- 完成三个checkpoint的same-task demo geometry及Writer参数演化。证据显示
  visual-state由弱demo信号退化为主要progress code，而raw-image/Meta forecasts
  越来越贴近低层demo translation；AS loss下降时latest-is-best和
  residual-correction语义持续恶化。
- 完成既有400 LoRA consensus、64×8 random permutations、endpoint/time-warp、
  AS loss/gradient和forecast component分解。summary SHA256为
  `390fcad1...f9a6`、`edbb86c8...916e`、`2bd6ae54...7186`。
- 生成Object-1/Object-3共100 episodes的五种root-cause LoRA cache，并运行
  official fixed-state rollout。no-VL/no-Action/lead-only/frame-main-only/
  translation-only为`48/50/40/72/79`；translation-only几乎复现true shuffled
  `82`。LoRA geometry/rollout summary SHA256为
  `3d0b6679...65c1`/`d384219c...662d`。
- 全面结论不再是“absolute-time唯一主因”。当前因果链为AS可识别性不足、
  visual-state旁路、Meta低层phase/translation化及absolute-time Revision放大。
  此前Intent+Transition v5只能修最后一层，已撤回为局部候选。
- 原位重写该根因文档（现名
  `docs/action_forecast_writer_v4_root_cause.md`），并同步README、
  execution brief、task plan、findings、decisions和v4 provenance。当前没有
  v5代码或训练；不继续AS、不进入RL，停在下一版重新设计前。

## correct/shuffled成败翻转行为复放完成（2026-07-26）

- 只使用物理GPU4–7、sealed step825 correct/shuffled LoRA cache和原固定
  Object-1/Object-3各50 states，完成四个condition/task的exact replay；
  success与termination step均`50/50`复现。0–3上的他人进程未触碰，结束后
  4–7均释放。
- 只为47条成败翻转保存每5 steps的agentview/wrist和每步EEF/gripper/action；
  未读取object pose、teacher action/state或隐藏目标。输出为
  `/data/ymdai/outputs/ember/pi05_action_forecast_v4_step0825_correct_shuffle_flip_replay_object13_g4567_20260726`。
- Object-1/Object-3的`shuffle-only/correct-only`分别为`9/2`和`31/5`。
  Object-3的31条shuffle-only中，correct有23条明确选择深绿色干扰瓶；两臂
  首次闭合点配对距离中位`0.1119 m`。Object-1收益主要来自更早到达和更可靠
  抓取/抬升；反向翻转证明shuffle也会破坏有用控制。
- Object-3收益跨22个teacher demos；四个相同cached LoRA在不同init geometry
  上出现相反翻转。结论从“正号只是无法解释的偶然补偿”细化为：正序视频的
  低层translation controller bias会压过物体语义，shuffle破坏该bias后让已有
  高层任务信息重新主导；不是shuffle生成更多语义或释放参数容量。

## Semantic Core + Causal Procedure v5设计封存（2026-07-26）

- owner批准新focused Goal并要求持续推进到AS特异性/性能与独立cold-start RL
  全部完成。Goal无token budget；完成focused AS/RL后停止，不自动进入
  final-32、task-local RL、joint oracle或ViVLA。
- `docs/action_forecast_writer_v5_design.md`现为唯一活动设计authority，完整记录
  teacher无state prompt、language-conditioned image-position Core、fixed
  native suffix、双Meta-LoRA、per-frame robot-semantic hidden、两层global
  causal Procedure、Core compiler、zero-init Procedure refiner、320
  routing-only identities、factor heads、公开LoRA schema和退役边界。
- 原`action_forecast_writer_v5_decision.md`已改名为
  `action_forecast_writer_v4_root_cause.md`，明确只保存v4根因证据；v4完整设计
  仍为历史provenance，不再定义当前代码。README、AGENTS、execution brief、
  decisions、concept、task plan、findings和progress已开始统一到v5。
- v5已原位实现并删除活动`visual_state.py`、`action_forecast.py`和v4 config；
  AS training、checkpoint、online validation、inference、evaluation cache及
  canonical evaluator均切换到v5 schema，不保留运行时兼容分支。
- 真实构造打印v5 trainable参数为`10,301,440`，比rank128 Source-SFT多
  `4,096`；公开LoRA保持rank16、76 tensors、`1,287,168` scalars。全套
  `187 passed`，Core permutation invariance、causal prefix、zero-content、
  identity、视频条件梯度和固定suffix buffer均通过。
- AS初版曾固定每action独立`N=4`条同task videos、`B_a×4`个逻辑LoRA/loss；
  推理严格one-shot。后续只使用物理GPU4–7，frame stride5固定，重新profile
  `B_a`与frame microbatch后按约一小时segment训练，每段均匀保存6个checkpoint。
- GPU4–7真实profile完成`B_a=1/4/8/12/20`及`m40/B8`边界。最终选择
  `B_a=8`、`N=4`、frame microbatch32；step2→12真实exact-resume通过，
  11个稳态steps中位/均值/范围为`61.39/59.78/38.99–92.08s`，峰值
  allocated/reserved为`60,319,360,000/67,471,671,296 bytes`。`B_a=12/20`
  及`m40/B8`均因reserved跳到约80GB、余量不足3GB而淘汰。
- 正式AS因此封存为每约一小时60 steps、每10 steps一个checkpoint，每段6个；
  下一步是fresh identity第一段和resident validation functional-loss选择。当前
  profile checkpoint只作mechanics/吞吐证据，不作科学性能结论。

## v5 AS首轮训练、step40/120特异性与续训（2026-07-27）

- fresh formal训练已完成step0→60并exact-resume到120；每10步均保留完整
  Writer/optimizer/scheduler/data cursor/4-rank RNG checkpoint。functional
  validation observed-best暂为step40 `0.136874`，step100曾反弹至
  `0.137017`，step120为`0.139036`；尚无足够峰后下降证据。
- step40内部顺序路径存在但很弱，五条件fixed-400为
  correct/same/wrong/shuffle/reverse=`45/52/52/51/51`，行为硬门失败。
  checkpoint未被误判为最终架构上限；step10/40/60内部纵向比较表明task语义
  分离和Procedure顺序差异仍在演化，因此保持架构不变继续训练。
- step120内部反事实通过结构门：Core set对shuffle/reverse数值不变；
  fixed-Core Procedure-only有效LoRA差异为`0.626%/1.087%`，Core-only伪差仅
  `0.073%/0.074%`；8/8 tasks均贡献。same-task-other与wrong有效LoRA差异为
  `1.235%/15.963%`。
- step120完整fixed-400为`65/59/57/61/65`。correct相对step40净增20，
  `41`条new-only、`21`条old-only、exact `p=0.0151`，且跨多个task提升；
  但correct相对same/wrong/shuffle/reverse的净差仅`+6/+8/+4/0`，均未显著，
  所以行为特异性和`125/400`性能门仍未通过。
- 五条件采用每卡5个模型副本并发：每条件仍是完整400 panel、4个持久
  policy/env workers；GPU4–7实测约`64GB`峰值、约`60GB` rollout稳态，五个
  panel约48分钟同时完成。没有触碰GPU0–3。
- 曾从step120按原合同续训，但owner在step128停止；没有生成step120之后的原子
  checkpoint，旧合同科学证据止于step120。

## v5共享四视频训练合同启动（2026-07-27）

- 复核确认旧`B_a=8,N=4`并非每rank只生成4套LoRA：每条action独立采视频，
  实际每step/rank在demo碰撞去重后仍生成约24–32套；step126–128 sampled
  frames为`537–799`，这是约一分钟一步的主要根因。
- owner现拍板：每rank每step一个task，确定性抽4条不同teacher videos，只生成
  4套one-shot LoRA；`B_a`条独立action queries均匀分给4套LoRA，每条action
  只对应一条video，形成`B_a`个普通均值functional losses。4 ranks全局
  task-balanced轮转。
- canonical sampler、AS step、checkpoint schedule identity、run metrics和
  config已开始原位切换到共享set；不新增runner或兼容分支。旧step120不可按
  新合同resume，后续使用fresh root。
- frame stride5保持不变。单video sampled frames为P50/mean/max
  `30/35.6/105`，所以保留`max_frames_per_encoder_call=32`显存安全块；末块
  改为真实长度、不再padding。profile只搜索`B_a`，不做optimizer accumulation。
- 该段记录的是共享合同刚切换时的待办；其focused tests、exact-resume与
  GPU4–7 `B_a` profile已在下一节完成。首轮结束仍先做absolute fixed-400
  validation，达到约`110–120/400`后再做特异性。

## v5共享四视频一对一分组profile封存（2026-07-27）

- canonical映射已改为每rank每step一个task、4条不同teacher videos生成4套
  one-shot LoRA，`B_a`条action按`i mod 4`均分；每条action只对应一条video，
  总functional losses为`B_a`而不是`4B_a`。focused tests直接锁定
  `[0,1,2,3,...]`映射、等分计数和不可整除fail-close。
- GPU4–7真实选择`B_a=16`。canonical 12-step profile先fresh到step2，再从完整
  checkpoint exact-resume到step12；合同SHA256
  `8dd6dfe6...263fb2`，metrics SHA256 `a570c916...09cea`，step12 manifest
  SHA256 `1ef4bde3...61bdd`。24 tasks两轮均覆盖，每task恰好32条action和
  8次video visits。
- 11个稳态steps的wall中位/均值/范围为
  `10.347/10.043/7.072–14.341s`，全局有效pairs/s中位`6.185`；每step始终
  64个全局policy samples、16个Writer video conditions、1次policy forward。
  峰值allocated/reserved为`63,736,767,488/68,415,389,696 bytes`，观察到的
  rank0四视频sampled frames范围`82–240`。
- B20虽完成3步，但reserved跳到`83,732,987,904 bytes`，只余约1.3GiB；
  B24/B32均在首个policy forward OOM，故拒绝。按
  `3600/10.347≈348 steps`取整，正式segment封存为400 steps、每50步一个
  checkpoint，预计约67–69分钟。下一步从fresh identity启动step0→400；
  首段后先做fixed-400绝对性能选择，不先跑特异性。

## v5单视频完整action-batch切换与profile（2026-07-27）

- owner终止共享四视频分组合同，活动训练改为每rank每step一个task、1条video、
  1套LoRA，完整action batch监督这套LoRA；后续task visit轮换video。
- canonical data/as-step/checkpoint/functional路径已原位简化，删除四视频
  schedule、round-robin映射和batched per-sample LoRA执行器；无新runner。
- 最长真实stride5视频为105帧。F32/B1完整一步`5.93s`；F105/B1占
  `79,873 MiB`且超过90秒不完成，因此保留F32显存安全分块。
- GPU4–7联合profile：F32/B20三步为`6.956/3.109/3.527s`，峰值
  allocated/reserved `76,937,901,056/83,630,227,456 bytes`；F32/B24和
  F24/B24 OOM；F40/B20无收益。owner接受最长视频少量余量，选择F32/B20并
  停止B21。
- 正式配置改为fresh step0→900、每100步checkpoint，使用物理GPU4–7；
  fixed validation和后续特异性均等待首段完成。

## v5单视频正式首段启动与跨session交接（2026-07-27）

- canonical单视频实现、focused tests和F32/B20 profile已在commit
  `0b4e00696113cf6601d6e63b4c73734f3cea1073`封存并push；正式launch前
  `HEAD==origin/main`且worktree clean。
- fresh formal已在tmux `ember-v5-as-sv900`启动，只见物理GPU4–7；output为
  `/data/ymdai/outputs/ember/pi05_as_writer_v5_single_video_dev_r4_seed7_s12000_0b4e006_20260727`。
  start-event contract SHA256为
  `03186c57ac736ac82398400676ff10c33eb46ab3e5f9bcbbe44064305944787c`。
- 首步确认每rank B20、1 task/1 video/1 LoRA、全局80 unique actions与4个
  video conditions、一次policy forward、无optimizer accumulation；source
  policy trainable params为0。首步`6.209s`，随后常规步约3–4秒，四卡物理
  显存约77.9GB。
- step100/200/300/400均已原子发布完整checkpoint；step1–400全部finite。
  step400训练body为`1,534.14s`，累计32,000 action queries；24/24 tasks每task
  覆盖1,320–1,340 examples、66–67次video visits和全部50条unique videos。
  常驻模型写出的512-row validation functional loss为step100/200/300/400
  `0.1360107/0.1349113/0.1332633/0.1324333`，无gradient、无optimizer update
  且test action reads为0；正式run仍继续到900。
- 跨session当前状态、精确launch、tmux/log/output、实时复核命令、step900后
  absolute-first评测顺序和禁止事项集中记录在
  `docs/active_session_handoff.md`。该文件是临时live-state ledger，不覆盖
  v5设计authority；新session不得据快照重复启动run。

## v5单视频首段封存、评估加速与轻量特异性（2026-07-27）

- 正式fresh run正常完成step0→900，训练body `3,485.15s`，累计72,000 action
  samples与3,600 one-video conditions；9个每100步checkpoint全部atomic且
  exact-resume state完整。step900每个train task恰有3,000 examples、150次
  video visits并覆盖全部50 videos与50 action episodes。
- correct-video fixed400代表点step100/400/700/800/900为
  `62/64/92/76/103`，首段observed-best为step900。虽然尚低于absolute预门，
  step800→900 paired净提升`+27`、exact `p=0.00155`，没有持续峰后下降证据。
- owner要求任何GPU/checkpoint分配下都先处理long。canonical evaluator已在
  commit `3b6d9d1`实现worker-slot级long-first；step800四卡24 workers先取完
  48个long shards后才取24个普通shards，400 rollouts用`921.60s`，
  `0.4340 rollouts/s`，约为首轮单卡吞吐`2.66×`。focused tests `27 passed`，
  commit已push。
- step900内部16-reference检查显示Core顺序不变性保持，fixed-Core
  Procedure-only effective-LoRA shuffle/reverse差异为`3.689%/5.764%`，
  policy action差异为`0.921%/1.406%`；顺序通路比step120明显增强并到达policy。
- 四个80-rollout反事实臂分别独占GPU4/5/6/7并行，correct直接复用full400的
  init-state 0–9。五臂correct/same/wrong/shuffled/reversed为
  `21/25/14/23/23`；配对净差correct-other为`-4/+7/-2/-2`，exact p为
  `0.344/0.143/0.688/0.688`。这只支持wrong-video方向性，尚无顺序优势；
  按owner定义不把80样本screen冒充full400特异性结论。
- 下一正式动作已封存为同一root从step900 exact-resume到step1800；新增900步、
  72,000 samples、3,600 video conditions与9个checkpoint，预计约一小时和
  `1.2GB`新增存储。GPU仍只用4–7，F32/B20与全部scientific contract不变；
  训练代码/config相对原run commit无diff，当前main仅多了评估调度改动，故使用
  fail-close的`--allow-contract-compatible-code-resume`。

## v5单视频step900→1800正式续训启动（2026-07-27）

- 续训launch前`HEAD==origin/main==db2a690`、worktree clean；step900 checkpoint
  的Writer、trainer与四rank state逐文件SHA256全部通过。个人存储
  `342.21GB`，预计峰值约`343.42GB`；物理GPU4–7均空闲，0–3未进入查询或
  visible set。
- tmux `ember-v5-as-sv1800`已用同一formal root exact-resume。start event为
  原contract `03186c57...94787c`、`resume_step=900`、
  `stop_after_step=1800`、24 tasks、source policy trainable参数0；
  invocation记录runtime commit `db2a690`与
  `contract_compatible_code_resume=true`。
- resume resident validation重算step900仍为`0.1370745508`，optimizer updates
  为0、无parameter gradient、test action reads为0。step901起metrics连续，
  初始核验至step917全部finite；常规step约`3–4s`，每步全局80 actions、
  4个one-video conditions与1次policy forward。GPU4–7约`77.9GB`且UTL接近
  100%。

## v5训练封存与step1400正式特异性（2026-07-27）

- step900→1800 exact-resume正常结束；metrics连续finite、每100步atomic
  checkpoint完整，旧训练tmux已退出。fixed400 correct在step1000与1400并列
  `115/400`，step1700/1800降至`71/86`，选择step1400 observed-best完成唯一
  正式机制检查，不再补1100/1200/1300/1500。
- step1400内部16-reference检查已在GPU4–7完成，16/16 rows、最大wall
  `27.12s`、peak reserved `19.316GB`。Core对同帧集合保持不变，
  Procedure shuffle/reverse中位差`64.30%/72.56%`，但effective LoRA仅
  `2.93%/4.77%`、policy action仅`0.49%/0.75%`；下游融合衰减是最早失效层。
- correct复用既有115/400；same/wrong/shuffled/reversed分别独占GPU4/5/6/7
  同时正式运行，每卡3 Writer generators + 6 persistent rollout workers。
  259个unique LoRA cache/臂只各生成一次；前12个long shards在普通task之前
  全部领取，随后动态分配。四臂均36/36 shards、400/400 rows、六worker
  return code全0、无错误，GPU已释放。
- 最终五臂为`115/108/74/113/114`。相对correct的
  correct-only/arm-only和exact p：
  same `23/16, p=.337`；wrong `58/17, p=2.18e-6`；
  shuffled `14/12, p=.845`；reversed `12/11, p=1.0`。
  视频内容门通过，same方向可接受，顺序门明确失败；v5停止且不进入RL。

## v5.1 authority切换（2026-07-27）

- 已完整保存side-chat收敛方案到
  `docs/action_forecast_writer_v5_1_proposal.md`。v5正式失败触发owner的直接
  推进授权；`AGENTS.md`、`docs/execution_brief.md`和
  `docs/active_session_handoff.md`已切换为v5.1唯一下一架构。
- profile前不预设900/1800 steps：实现与必要smoke后，先在GPU4–7用真实105帧
  视频重新profile显存、action batch和step吞吐，再换算约一小时fresh formal
  stop。实测后来得到首段900；它不规定下一段到1800。首段后先查内部五条件与
  轻量paired行为；第二/第三段只有在特异性、absolute和曲线共同支持时才单独
  启动，绝不自动续训。

## v5.1 canonical实现与CPU合同验证（2026-07-27）

- 已在既有`CompleteLoRAWriter`、AS training/evaluation/checkpoint入口内原位
  替换v5，没有新增runner或并行Writer。活动配置改为
  `configs/pi05_as_writer_language_axial_v5_1.json`；v5 config、constructor
  key和checkpoint/eval/generation schema均已从活动代码删除，v5结果只由Git
  与文档保存。
- tokenizer从完整权威prompt的SentencePiece immutable piece offsets提取task
  span；Text路只输入`BOS + 同一组task-span IDs`，不重新分词也不含模板。
  `video_program`现有Text/VL/Action三套独立rank4 Meta-LoRA，共享
  `2048→256`语言投影；Core value只来自multimodal task-token hidden，raw
  image-position hidden不再进入下游。
- `temporal`现实现token-aligned、跨frame无序的mean-anchored attention，
  两层language-axis Core、两层causal Procedure、centered Procedure reader、
  zero-init AdaLN和一个post-fusion slot block；factor head hidden为240。
  逐模块真实计数与设计表完全一致，总计`10,244,872`。
- CPU验证覆盖真实tokenizer round-trip、可变L/T shape、Core frame permutation
  invariance、Procedure prefix causality、routing/value隔离、fresh identity、
  三阶段gradient opening、固定suffix与不兼容schema。全仓
  `PYTHONPATH=src .venv/bin/pytest -q`为`189 passed`；architecture guard无
  hard violation，既有大owner仅保留review flag。

## v5.1 GPU合同、训练/推理上限与首段seal（2026-07-27）

- GPU4–7真实policy smoke完成step1并从完整checkpoint exact-resume到step2；
  两个cursor、四rank state、Writer/optimizer/scheduler/sampler/video schedule
  与RNG均通过原生checkpoint校验，source policy trainable参数为0。step1/2
  分别约`4.49/2.67s`，没有nonfinite或OOM。
- F32/B20重新在v5.1上实测，不是继承v5：105帧真实最长video步为`7.248s`、
  `11.04` global queries/s；三步profile为`7.322/3.249/3.664s`，常规吞吐
  `24.63/21.84` queries/s。峰值allocated/reserved为
  `76,926,205,440/83,638,616,064` bytes；B20保留约`8.36GiB` allocated
  headroom，按实测batch斜率不再冒险启动B21。
- 推理profile在GPU4–7每卡一次性启动6个worker，24个worker共同按确定性分片
  生成47个LoRA并保留source policy直接rollout。最大单worker generation wall
  `5.203s`，峰值allocated/reserved约`11.63/12.81GB`；48 episodes的
  rollout-only吞吐`0.3799/s`（`1367.6/h`），现场整卡约63–65GB且GPU利用率
  `99–100%`。首次24-policy并发load耗时约`146–162s`，是主要固定成本。
- evaluator queue已进一步修正为全局long-first：任何GPU的未领取max-horizon
  shard都压过ordinary；GPU affinity只决定long内部先取本卡还是偷取他卡。
  新回归覆盖“本卡long耗尽但他卡仍有long且preferred task为ordinary”的情况。
- v5.1首段按新吞吐换算为step900约一小时：4-rank DDP、F32/B20、每step
  80 action queries/4 one-video conditions、每100步checkpoint并做512-query
  online validation。`total_steps=12000`只保留scheduler/最大探索包络；
  `selected_stop_step=900`为唯一当前launch边界。第二/第三段的停止点未预定，
  step900后必须先看早期特异性、absolute和train/validation曲线，不能自动resume。

## v5.1首段训练、step700选择与特异性封存（2026-07-27）

- 正式训练根目录：
  `/data/ymdai/outputs/ember/pi05_as_writer_v5_1_language_axial_dev_r4_seed7_s12000_c199ad3_20260727`。
  fresh step0→900已正常结束，900条metrics连续、checkpoint100..900完整，
  总wall `3622.358s`；contract payload/run-summary file/metrics file
  SHA256分别为
  `acc57fd96cace6d3a9d38a7dbfe6d8593cd29bdce1a0ff10e1f2b4239de46227`、
  `327ba70c9fc9854441a1ce75bb8b6bba103299ae4b49add8dd8c3aa361e96cb0`、
  `0fe5d2490d2d692b98b9c3e8f70177f7839ad0a4e6cdcd5cb943f179d74d4a86`。
- 有放回80-rollout screen全部通过aggregate验证：
  step100/200/300/400/500/700/800/900=`19/18/15/7/21/17/19/14`。
  随后按一张物理卡负责一个checkpoint，同时在GPU4/5/6/7完成
  step100/500/700/900的正式correct400，结果为`82/96/98/84`。四个root：
  `pi05_as_writer_v5_1_correct400_withreplacement_step{0100,0500,0700,0900}_c199ad3_20260727`；
  results SHA256依次为
  `023a9c5fb98fe4b937a1c760a2fa74bb9bb5ba944098af48d593b4cb4ac98577`、
  `23f5032f32d0e95b301ee4b11146efe06a8c955b9e56ad86c7bf735aab9defd5`、
  `cb42f0e7802463cb2e4a26efffc0ce5e41abdb72dad44b750ff2764bb2f9049b`、
  `1b0e28b1afedf133dd43585e9a3b4e6e2a9711e2b436ba5d2ee65c1eaef26ab2`。
  每个root均400 rows、8 tasks×50 states、36 shards、6 workers、return code全0。
- step700轻量五臂复用既有correct80=`17/80`；same/shuffled/reversed分别为
  `20/11/6`，正式root为
  `pi05_as_writer_v5_1_specificity80_withreplacement_step0700_{same_task_other,shuffled,reversed}_c199ad3_20260727`，
  results SHA256分别为
  `ecee24fd84d15d23bf512da8e60316f0224d7c47e3c959d1c7b841ad8bc3fd9b`、
  `a12dda7d65cad76dc2f808bde2f0883969b0fe04e45ec6e5e47500d2ff409324`、
  `52553b14073e8dcca16301bd0e5b0f0ac537e016c9656dd991620d4fd34703a5`。
- 初次wrong root
  `pi05_as_writer_v5_1_specificity80_withreplacement_step0700_cross_suite_wrong_c199ad3_20260727`
  遭遇单worker EGL 0x8cdd；resume后aggregation按launcher timing证据
  fail-close，未产生可信`results.json`，只作失败provenance。正式fresh root
  `pi05_as_writer_v5_1_specificity80_withreplacement_step0700_cross_suite_wrong_fresh2_c199ad3_20260727`
  使用GPU4–7一次调用、24 workers、26 shards完成80/80，wrong=`7/80`，
  results SHA256为
  `e11c0daa1994420dd24b7d52bff5e153a2f1628396527468f1cccda0b5406b75`。
- 五臂逐row exact-pair分析保存在
  `/data/ymdai/outputs/ember/pi05_as_writer_v5_1_specificity80_withreplacement_step0700_paired_analysis_c199ad3_20260727.json`，
  SHA256为
  `6fecb53d051104b72698b5f776eb588240ee5931520bf233985e3b72e2984316`。
  correct-only/control-only为same `4/7`、wrong `12/2`、shuffled `10/4`、
  reversed `13/2`。
- 内部16-reference检查保存在
  `/data/ymdai/outputs/ember/pi05_as_writer_v5_1_internal_specificity_step0700_refs2_c199ad3_20260727`，
  summary SHA256为
  `7a0ced20700b38cd8500396453c7958d94dedde04bd53d5a9c562dda032ec0fe`；
  4 ranks、16/16 rows、8 tasks×2 reference videos全部通过，无validation action
  target或teacher state value读取。
- 所有训练、rollout和内部probe进程完成后均已释放GPU4–7；没有启动第二段、
  第三段、无放回重测、full-400五臂或cold-start RL。当前按owner要求停在
  v5.1首段特异性结论，等待讨论。封存前fresh重读全部上述JSON与SHA、验证逐row
  paired统计和内部counterfactual；全仓`PYTHONPATH=src .venv/bin/pytest -q`
  为`190 passed`，物理GPU4–7均为`0 MiB`且没有活动EMBER tmux/process。

## v5.1无放回五臂与持续探索恢复（2026-07-27）

- owner明确解除上述停止点并创建开放式AS性能Goal；只要absolute没有提高到
  可信满意水平，或提升存在v4-shuffled式逻辑漏洞，就继续探索，不需逐项审核。
- step700新的无放回paired full400已全部完成：
  correct/same/wrong/shuffled/reversed=`88/97/75/65/45`。五个root均400 rows、
  8 tasks×50 states；每task teacher demo0..49恰好各用一次，所有worker
  exit0。结果SHA256依次为
  `d3391e3a...ae1b`、`2369d50c...f388`、`1e295154...a12c`、
  `11a98c83...37ca`、`622b0bca...d598`。
- 逐row配对分析封存在
  `pi05_as_writer_v5_1_specificity400_noreplacement_seed7_step0700_paired_analysis_92b1e03_20260727.json`
  （SHA256 `c4a62c4c...31fa`）。same净`-9,p=.2221`；correct相对wrong
  净`+13,p=.1766`；相对shuffled净`+23,p=.00762`；相对reversed净
  `+43,p=8.91e-7`。新结果消除了v4式shuffle获益，但absolute与wrong breadth
  均未达到停止标准。
- 根据四卡24-worker reversed现场尾部，canonical queue现在保持long全局优先
  的同时，把ordinary工作至少保留两个worker波次。实际标准panel从
  48 long + 24 ordinary变为48 + 48；覆盖仍为400/400且long领取顺序不变。
  focused evaluator测试`48 passed`、全仓`194 passed`，commit
  `73f171a`已push。
- step900 checkpoint重新核验：manifest/canonical payload/writer/trainer
  SHA256分别为`6958498b...b828`、`2971d3a4...8fe`、
  `17da429d...7ac`、`a7057a84...cda`；原训练合同SHA256
  `acc57fd9...227`。下一动作是只在GPU4–7上同root exact-resume至step1800，
  随后按一张卡一个checkpoint并发建立无放回correct400密集曲线。

## v5.1 step900→1800正式resume启动（2026-07-27）

- preflight时HEAD/origin均为`a92850f`且tree clean；个人占用
  `361,804,259,328 bytes`，预计新增约1.2GB；物理GPU4–7均0MiB、无计算
  进程，GPU0–3未查询。正式命令与`task_plan.md`记录一致，在tmux
  `ember-v51-as-sv1800`启动，runtime只见GPU4–7。
- start event为`resume_step=900`、`stop_after_step=1800`、4-rank DDP、
  Writer `10,244,872`参数、source policy trainable count0；
  `contract_compatible_code_resume=true`，原formal合同SHA256保持
  `acc57fd96...227`。resident step900 validation精确复现
  `0.13314267079249476`。
- step1000已生成第一份新增完整checkpoint并继续训练：manifest/canonical
  payload/writer/trainer SHA256为`61e7e66a...6cfc`、
  `b1c7f209...9d2e`、`ea249c56...f065`、`ff391fcc...e9b3`；累计80,000
  action queries，24 tasks均已读全50 action episodes与50 unique videos。
  step1000 online functional validation为`0.1373837591`，比step900高
  `0.0042411`；它只作诊断，不用于替代无放回rollout选择。

## v5.1 1800-step封存、稠密曲线与scale扫描启动（2026-07-27）

- 同一formal root已经完整到step1800并正常退出；新增900步耗时约一小时，
  checkpoints1000..1800每100步完整保留。最后一步applied LR仍为
  `2.84213e-4`，所以没有把“训练结束”误写成“学习率已充分退火”。
- 无放回correct400曲线全部完成：
  `100/500/700/900/1000/1100/1200/1300/1400/1500/1600/1700/1800`
  对应
  `83/98/88/86/114/111/114/92/127/95/92/65/126`。step1400为全局
  observed-best。step500与1600原先因EGL失败后resume聚合规则不完整而没有
  `results.json`，现已从不可变raw shards正式补聚合为`98`和`92`。
- step1400内部16-reference root为
  `pi05_as_writer_v5_1_internal_specificity_step1400_refs2_42a9707_20260727`；
  run-contract/summary/rows SHA256依次为
  `39cb5206...9d3`、`1749a354...d78`、`56b3314d...342`。16/16 references
  表明Core、Procedure、fusion、effective LoRA和policy function的信息路径
  均按v5.1合同工作。
- commit `082090f`完成三项canonical evaluator改进：同GPU EGL transition
  flock、跨resume累计launcher证据、LoRA-B rollout scale。targeted
  `37 passed`、全仓`196 passed`、architecture guard无hard violation；
  feature branch与main均已push，HEAD=origin/main=`082090f`。
- tmux `ember-v51-scale`已启动四个step1400 full400：
  GPU5=`1.25×`、GPU6=`1.50×`、GPU4=`1.75×`、GPU7=`2.00×`。四者均使用
  6 policy workers、long-first queue、无放回state/video双射并复用原
  400-entry LoRA cache；preflight只查询GPU4–7。启动前个人占用
  `375,770,816,512 bytes`，低于500GB cap且scale roots不复制1GB cache。

## v5.1 scale封存与step1400全量控制启动（2026-07-27）

- 四个scale full400均完整退出：
  `1.25/1.50/1.75/2.00 = 124/119/99/82`，均低于原`1.00=127`。
  相对1.00的逐row`new/lost`依次为`21/24, 26/34, 19/47, 14/59`；
  不是一致增益。results SHA256依次为
  `b22e7854...6c48`、`88d84a3d...1964b`、
  `d8e025ac...2c378`、`075f9d3f...0f0b`。选择保留scale 1.00。
- 为避免卡间等待，每个scale所在GPU一释放就自动接入一个step1400控制臂。
  tmux `ember-v51-step1400-specificity`当前在GPU5/6/4/7分别运行
  `same_task_other/cross_suite_wrong/shuffled/reversed`；每臂full400、
  每卡6 generators后原进程切换为6 rollout workers、无放回配对、
  long-first。四臂预计新增约`4.27GB`，个人占用峰值仍显著低于500GB。
- 新配置
  `configs/pi05_as_writer_language_axial_v5_1_stabilization.json`
  已在commit `52503e1`封存并push。它不改Writer拓扑、F32/B20、数据或信息墙，
  只从原step1400加载Writer权重，开启fresh AdamW和
  `peak_lr=1e-4,warmup=50,decay_steps=1800,decay_lr=1e-5`的新阶段；
  首段只运行phase step0→900。完整控制结束后做live GPU/storage preflight，
  再用GPU4–7正式启动。

## step1400五臂完成、低LR运行、v5.2实现（2026-07-27）

- step1400四个控制root均400/400正常退出；五臂总分
  `127/133/94/107/120`，逐row pairing、video bijection、env/policy RNG及
  noise prefix全部通过。统一paired分析artifact已原子写入outputs，SHA256
  `51c19b66...1579`。
- 低LR preflight确认main/origin=`756bdaa`、tree clean、个人占用
  `379,485,047,888 bytes`，GPU4–7均0MiB且无compute process；GPU0–3未进入
  查询或visible set。
- tmux `ember-v51-stabilize1400`已正式启动4-rank F32/B20 phase0→900。
  run contract canonical SHA256为`b19937ce...c95a`，初始化精确记录原
  step1400 manifest `a503eaac...26cb`与Writer
  `22da8417...5d1a`，optimizer/scheduler/RNG均fresh。step100完整checkpoint
  manifest SHA256为`5387b2cf...0a9`，训练继续。
- 隔离worktree `EMBER-v52-20260727`完成canonical v5.2实现、schema/config、
  参数预算与设计文档；focused Writer测试`61 passed`、`git diff --check`
  通过。当前commit `4011966`已push至`origin/codex/v52-patch-grounding`。
- 训练结束后的固定动作是用GPU4/5/6/7各负责phase100/300/600/900一个
  checkpoint，四点同时做无放回correct400；每卡6个Writer generators完成
  cache后转6个persistent rollout workers，queue保持全局long-first。

## v5.1低LR首段完成与并发correct400启动（2026-07-27）

- phase0→900已正常结束：72,000 action queries、3,600 one-video conditions、
  wall `3616.478s`；run summary SHA256 `238853ad...e7b`，最终checkpoint
  manifest SHA256 `c0bf283d...e8d`。
- 九个online validation loss没有持续改善；权重漂移artifact
  `writer_drift_analysis.json`显示100-step update逐渐变小，但后续相邻方向
  持续负余弦，SHA256 `7564fff2...ddc3`。这把低LR描述为待rollout判定的
  稳定化尝试，而不是已成功的新best。
- tmux `ember-v51-stabilize-correct400`已将phase100/300/600/900分别分配到
  GPU4/5/6/7。每个checkpoint只加载一次，每卡6 generators→6 persistent
  rollout workers；全部固定validation 8×50、无放回、全局long-first。
- tmux `ember-v51-stabilize-analysis`等待四个结果并自动生成相对原step1400
  的逐row paired artifact。评测期间main保持`756bdaa` clean，不合并v5.2。

## v5.1低LR封存与v5.2正式合同（2026-07-28）

- 低LR phase100/300/600/900四个无放回correct400全部完成，结果
  `119/115/123/104`，没有超过原step1400=`127`；paired artifact SHA256
  `f52c9b78...543`。phase600仍由Goal-6与两个object task构成，Spatial两task
  均0，故v5.1停止。
- v5.2 branch已推进至`849e622`并完成真实GPU4–7 profile。B20三步含一次
  exact-resume；Task-Queried Patch evidence相对task evidence RMS均值`.429`。
  B21连续三步finite，最大allocated/reserved
  `80,283,666,944/83,892,371,456` bytes；B22四rank对称OOM。
- 配置现封存F32/B21、4-rank、global84、scheduler探索包络12000 steps；
  只授权fresh首段stop=900、每100步checkpoint/512-query online validation。
  900之后必须先做无放回correct400 checkpoint选择与机制检查，不自动进入
  第二或第三段。

## v5.2首段封存、原recipe续训与v5.3实现（2026-07-28）

- v5.2 step0→900、四点correct400、step900内部检查和五臂full400全部完成。
  correct曲线`72/79/120/132`；五臂`132/138/74/82/83`。same无显著差异，
  correct相对wrong/shuffled/reversed均为极显著优势；paired artifact
  SHA256 `d8e2f4b...7ae7`。
- owner明确先测原版v5.2训练上限，并将v5.3设为默认下一fresh架构；v5.3仍用
  原版one-task-per-rank update，不采用task-complete。main
  `529da6b`已从step900 exact-resume到本次segment边界1800，tmux
  `ember-v52-resume-1800`；每100步checkpoint。训练结束后自动在GPU4–7并行
  评测step1200/1400/1600/1800的无放回correct400。
- v5.3设计封存在`docs/action_forecast_writer_v5_3_design.md`。隔离分支
  `c1e3777`已实现task-grounded adjacent visual transition、fresh schema和
  参数预算搬移，全仓回归`198 passed`并push。它不影响当前v5.2训练
  commit；待v5.2上限封存后再做真实GPU profile。

## v6设计封存（2026-07-28）

- owner把默认下一fresh架构提升为EMBER Writer v6，并批准
  Task-Grounded Semantic Set + Visual-Transition Procedure整体方案。
- 新authority已写入`docs/action_forecast_writer_v6_design.md`：Core采用
  mean backbone + centered residual，Procedure采用按actual arm order重算的
  adjacent task-grounded transition，compiler保持v5.2已验证的传递路径，
  factor hidden恢复为256。手算总参数`10,775,296`。
- `AGENTS.md`、active handoff、task plan与findings已同步版本定位。当前只完成
  文档封存；没有修改code/config，没有启动profile、训练或评测。

## v6 canonical实现与task-complete CPU封存（2026-07-28）

- owner提供最终训练合同并覆盖旧v6 recipe：K6 task-complete、首选B20、
  OOM/连续不稳定才退B16；首段后除非absolute明确下降，默认续第二小时。
- 隔离worktree
  `/data/ymdai/.codex/worktrees/EMBER-v53-20260728`已完成唯一canonical v6：
  Semantic Set mean backbone + centered residual、Visual Transition、
  hidden256 factor heads、总参数`10,775,296`。
- 训练入口原位改为每rank六个task-pure micro-round：前5轮DDP`no_sync`，
  第6轮同步；每个task loss乘`1/6`立即backward；一个macro一次zero_grad、
  clip、AdamW和scheduler。B20计数为24 video conditions、480 queries、
  24 functional forwards。
- sampler根据本次video长度做四组cost balance、rank内long-first并跨macro
  轮换物理rank；checkpoint和resume只在macro边界，run contract与metrics封存
  全部计数及24个task/video assignment。
- `configs/pi05_as_writer_language_axial_v5_3.json`已由唯一v6 config替换；
  v5.2/v5.3 checkpoint/eval artifact fail closed。全仓
  `PYTHONPATH=src .venv/bin/pytest -q`为`200 passed`，architecture guard无
  hard violation，`git diff --check`通过。
- corrected mixed-task Source-SFT合同已写回authority，待v6完成后fresh实现/
  重训并寻找validation best。

## v6 B20 profile、resume smoke与正式配置封存（2026-07-28）

- GPU4–7只读preflight均为空闲后，在commit `d66e726`完成B20三步真实
  task-complete profile。root为
  `/data/ymdai/outputs/ember/pi05_as_writer_v6_taskcomplete_profile_b20_d66e726_r2_20260728`；
  3 macro共1,440 queries和72 video conditions，wall `58.730s`。
- 三步max-rank wall=`20.442/18.585/18.635s`，后两步平均
  `25.793 queries/s`与`193.447 macro/hour`；峰值allocated/reserved
  `76,985,299,968/83,644,907,520 bytes`。最长105帧条件成功，loss/grad
  全finite，故选择B20且不运行B16。
- run-contract/metrics/summary/final manifest SHA256依次为
  `5f9b66fc...161e0e`、`e13f250d...16df6`、
  `30bb3798...401a`、`282825c4...733b`。
- 独立resume root从bitwise相同step1边界继续到step3；任务、视频、query、
  LR和cursor一致。GPU kernel非确定性使两步后Writer最大参数漂移约
  `9.82e-5`，不影响exact-state resume合同。visual-transition step1→3
  L2更新`0.0111083`，真实梯度路径成立。
- 正式config封存B20、首段200 macro、每25 macro checkpoint、默认第二段到
  400（除非absolute明确下降）。owner取消正式run全量HDF5 SHA；启动仍核对
  manifest、文件size和HDF5 schema。下一动作是验证、commit/push、集成main，
  再在GPU4–7启动fresh macro0→200。

## v6 task-complete正式首段完成（2026-07-28）

- 正式root
  `/data/ymdai/outputs/ember/pi05_as_writer_v6_taskcomplete_dev_r4_b20_seed7_s2400_149badc_20260728`
  已自然停在 macro200；tmux/训练进程退出，run summary 为 200 metrics、
  4,800 video conditions、96,000 queries、wall `3,864.599s`。
- checkpoints `25/50/75/100/125/150/175/200` 全部存在；终点24 tasks各
  4,000 queries、200次video visits、50/50 action episodes和50/50 teacher
  videos。macro200 Writer/trainer/四rank state逐文件SHA256与manifest一致；
  正式log无OOM、nonfinite、CUDA/NCCL error或traceback。
- 全段平均 `18.668s/macro`、`25.720 queries/s`；峰值
  allocated/reserved `76,986,335,232/83,642,810,368` bytes。训练后GPU4–7
  均回到0MiB。
- 训练等待期的仓库清理在隔离分支完成：101 files changed，
  431 insertions/18,853 deletions；tracked tree降至约3.17MB，退役临时
  `.codex/tmp` 另清除108 files/1,820,301 bytes。正确worktree
  `PYTHONPATH`下全仓`177 passed`、Markdown link audit与`git diff --check`
  通过；正式run退出后可安全fast-forward合入。
- 下一动作是合入清理提交，再用GPU4/5/6/7分别运行macro50/100/150/200
  no-replacement correct400；每卡6 Writer generators和6 persistent policy
  workers、全局long-first。

## v6 四点 correct400 启动（2026-07-28）

- 清理/evidence commits `24bdc5d/aecb100` 已 fast-forward 到 main 并push；
  main现场全仓 `177 passed`，status clean。
- live preflight 时GPU4–7均0MiB、个人占用`402,806,314,951` bytes。
  tmux `ember-v6-correct400` 已把macro50/100/150/200依次映射到GPU4/5/6/7；
  每点一个checkpoint、400 episodes、6 generators、generation batch16、
  6 persistent workers、无放回video。
- 四点各自400-entry LoRA cache均已完成；同一进程保留source policy切换
  rollout，避免第二次约150秒模型加载。首批每点6个claimed shards全部来自
  两个horizon-520 `libero_10` tasks，global long-first现场核验通过。

## v6 correct曲线完成与macro200特异性启动（2026-07-28）

- macro50/100/150/200 的 no-replacement correct400 全部自然完成：
  `114/77/120/129`；对应成功 task 数为 `6/7/7/5`。所有 launcher workers
  exit 0、queue 36/36 complete、400 rows、零错误。
- paired artifact
  `/data/ymdai/outputs/ember/pi05_as_writer_v6_correct_curve_paired_aecb100_20260728.json`
  核验四点 state/env seed/policy seed/noise/video assignment 完全一致，
  每 task 50 teacher videos 无放回双射。macro200 是 aggregate best，但与
  macro150 的 9-success 差异不显著，且 breadth 从 7 降到 5。
- GPU4–7 清空后，tmux `ember-v6-specificity400` 已将 macro200 的
  same-task-other/cross-suite-wrong/shuffled/reversed full400 映射到
  GPU4/5/6/7；每臂仍使用 6 generators、batch16、6 persistent workers、
  无放回视频与 global long-first。
- 等待期额外清除约 3.8 MiB Python/pytest/editable-install 可再生缓存。
  Git 仍 clean；9.1 GiB 唯一活动 `.venv` 和实验证据未动。
- 进一步审计发现 117 组 Writer LoRA cache 共 91.74 GB。只删除其中 113 组
  已有 matching results/launcher completion 且所有 worker exit 0 的历史
  cache，共 `87,487,144,566` bytes；当前四个 control cache 被硬性排除。
  rollout rows/results、queue、日志、contract、checkpoint 全部保留，个人占用
  从 `411,326,994,567` 降到 `323,840,205,468` bytes。

## v6 macro200五臂完成与内部检查启动（2026-07-28）

- 五臂结果为 `129/131/108/111/105`；same相对correct switches
  `22/24,p=.8830`，wrong为`42/21,p=.0111`，shuffled为
  `36/18,p=.0198`，reversed为`37/13,p=.00094`。行为方向通过，但后三臂
  margin明显弱于v5.2，且correct只覆盖5/8 tasks。
- 四个control均400 rows、36/36 shards、workers exit 0、零错误；视频无放回
  双射和global long-first复核通过。paired artifact位于
  `pi05_as_writer_v6_specificity400_noreplacement_seed7_macro0200_paired_analysis_faf6e33_20260728.json`。
- control结果封存后删除其4组可重建LoRA cache，额外释放
  `4,254,855,093` bytes；结果/rows/queue/log/contract/checkpoint均保留，个人
  占用降至`319,598,037,816` bytes。
- tmux `ember-v6-internal-m200` 已在GPU4–7启动16-reference内部传递检查；
  输出root为
  `pi05_as_writer_v6_internal_specificity_macro0200_refs2_aecb100_20260728`。

## v6 macro200内部检查完成（2026-07-28）

- 16/16 rows和四个rank输出正常完成，五条件、fixed-Core Procedure-only与
  Core-only反事实均齐。新增visual-transition让shuffled/reversed的Procedure
  median relative-L2达到`.0888/.1167`，并传到effective LoRA
  `.2590/.2436`和policy action`.0282/.0392`。
- fixed-Core结果几乎复现全部顺序差异，而Core-only接近零，排除了Semantic
  Core顺序旁路。相对v5.2，上游Procedure差异增强但下游LoRA/action差异减弱；
  结合macro200仍是absolute右端最高点，按合同exact-resume到macro400。

## 第二轮可重建cache清理（2026-07-28）

- 审计并删除17个旧v5.1 standalone LoRA cache：16个已有完整results/
  launcher completion，另1个结果缺失wrong-video run已被保留的fresh2重跑
  替代，共释放`5,282,177,024` bytes。
- 删除清单为
  `/data/ymdai/outputs/ember/cache_cleanup_legacy_v51_standalone_lora_20260728.json`；
  结果、rows、queue、日志、合同及全部Writer/source checkpoint未删。至此
  outputs中不再残留`writer_lora_cache`或顶层`*_cache`目录。

## v6第二段exact-resume启动（2026-07-28）

- tmux `ember-v6-formal-400` 从`step_00000200`续到macro400；invocation记录
  `contract_compatible_code_resume=true`、`monotonic_stage_extension=true`，
  canonical contract SHA仍为
  `e0d0cf703b596e73552f4150f5abed9f9726a80e5af214095baca33719bdd6a3`。
- GPU4–7各一DDP rank，稳态约78.0GB/卡、约25.25queries/s；resume后metrics
  从201连续追加，没有重放首段。

## v6第二段、四点correct400与focused判断封存（2026-07-28）

- macro200→400 exact-resume自然完成；metrics连续1..400，225..400每25步
  checkpoint和四rank state完整。第二段wall `3,903.024s`，累计9,600 video
  conditions与192,000 action queries；训练/评测进程均退出，GPU4–7释放。
- macro250/300/350/400的paired无放回correct400为
  `117/118/125/125`，没有超过macro200=`129`。完整八点curve artifact SHA256
  `7789350d...72e1`；所有点task/state、env/policy/noise和50-video双射一致。
- 第二小时没有显著aggregate下降，但成功能力在tasks间大幅迁移；因此停止
  full-24 v6 recipe，不补every-25 rollout。macro200仍为observed-best，已有
  `129/131/108/111/105`五臂和16-reference内部证据继续有效。
- owner把corrected mixed-task Source-SFT提前为下一实验，并统一focused AS门：
  `correct400 >= max(150, corrected SFT_best+30)`，同时保留全部视频因果、
  same-task、多task breadth和独立paired复测条件。

## corrected mixed-task Source-SFT集成与profile（2026-07-28）

- 隔离分支`codex/source-sft-mixed`在commits
  `4c527dd/55ccbcc/effbd4b`实现并封存hierarchical mixed sampler、checkpoint
  v2与B144正式合同，随后fast-forward合入main。
- 每rank B144 physical batch固定为24 tasks×6 samples；每个batch一次普通
  同步optimizer update，无gradient accumulation。task→episode→chunk分层
  均衡、跨rank row disjoint、absolute-step sample identity和exact resume由
  focused tests锁定；profile seal后`21 passed`。
- GPU4–7 fresh step1→resume step3完成；root为
  `/data/ymdai/outputs/ember/pi05_source_sft_rank128_mixed_profile_r4_b144_55ccbcc_s3_20260728`。
  后两步wall `16.684/15.847s`、吞吐`34.524/36.346 queries/s`，峰值
  allocated/reserved `60.69/74.07GB`；三步共1,728 unique rows且24 tasks
  每步等量，step3已覆盖每task全部50 episodes。B120 fallback未触发。
- config已封存formal fresh step0→225、每25步checkpoint，约61分钟训练body；
  冷加载单独报告。首段峰值在右端或不稳定时exact-resume到450，之后不做机械
  续段。正式launch前仍需main/origin clean、GPU4–7 live preflight和存储复核。

## corrected mixed-task Source-SFT首段完成与四点correct400启动（2026-07-28）

- main/origin均为`64622795314ab2223b7948f526e7e32767c468df`且launch时
  worktree clean。正式root
  `/data/ymdai/outputs/ember/pi05_source_sft_rank128_mixed_dev_r4_b144_seed7_s2400_20260728`
  已自然完成step225；225条metrics连续finite，累计129,600 queries、每task
  5,400 samples，训练body wall `3,639.436s`。
- step25..225每25步的9个checkpoint均保留；step225的LoRA、trainer state、
  四rank state与manifest SHA256复算一致。run root为551MB，个人占用约295GB，
  因而按owner最终澄清不做checkpoint删除。
- online validation step25..225为
  `.139748/.134216/.134064/.132966/.133862/.134068/.134527/.135724/.135276`；
  step100暂为online best，但closed-loop排序尚未得出。
- tmux `ember-source-sft-mixed-val4`将step50/100/175/225映射到
  GPU4/5/6/7，每点fixed validation 8×50、6 persistent workers、一个
  checkpoint只在一张卡加载。四卡冷加载约149–154秒，rollout时约
  72–73GB/卡、接近100% UTL；四个queue首批6个shard均为horizon-520 long
  tasks。
- 四点自然完成为`60/75/77/56`，每点400 unique rows、36/36 shards、
  6/6 workers exit0、全attempt1、零错误；paired state/env/policy/noise
  合同完整。results SHA256依次为
  `760bca21...7976/346100c7...e8a/a3f95801...b546/76687676...a863`。
- step175与100仅差2，paired为`39/37,p=.9088`；175相对225为
  `40/19,p=.00864`。能力在Long/Object上涨的同时从Goal/Spatial迁出，峰值
  不稳定且训练量仅129,600 queries。按sealed规则从完整step225
  exact-resume到450；不改recipe、不补当前密集rollout、不删checkpoint。

## full-24 Source-SFT封顶与global-8 profile（2026-07-28）

- full-24 formal从step225 exact-resume到450自然完成；metrics连续1..450，
  step25..450的18个checkpoint全部保留，root约1.1GB。12点paired
  correct400为
  `60/75/77/56/77/57/87/71/98/109/107/74`；step400为observed-best，
  step450相对400显著下降`50 lost/15 gained,p=1.57e-5`，故停止该recipe。
- dense analysis artifact为
  `paired_correct400_step0050_0450_dense.json`，SHA256
  `5a781a50344b72085ac154b1602a6842cb9bcb6b44a0a957f3da544e5e8791c4`；
  12个面板均400 rows、36 shards、6 workers exit0并满足paired seeds/noise和
  global long-first。
- 隔离分支`codex/source-sft-mixed8`提交`c25cd5d`实现唯一global-8 cyclic
  sampler：4 ranks×2 tasks/update、3 updates覆盖24 tasks，保持
  B144/global576、rank-128 LoRA、LR/scheduler及平均task/sample clock。
- GPU4–7 profile root
  `pi05_source_sft_rank128_mixed8_profile_r4_b144_c25cd5d_s6_20260728`
  完成fresh0→3与exact-resume3→6。两轮cycle均精确覆盖24 tasks，3,456
  samples全唯一；稳态wall `15.833–15.883s`、吞吐
  `36.27–36.38 queries/s`，峰值allocated/reserved `60.69/74.07GB`，
  loss/gradient finite，无OOM/NCCL/CUDA错误。B128 fallback未触发。
- config封存formal fresh0→240、每30步checkpoint、closed-loop
  60/120/180/240；除非可信下降，否则exact-resume至480并评测
  300/360/420/480。owner最终要求保留原checkpoint，不做删除；当前个人占用
  约296GB，远低于500GB cap。

## global-8 Source-SFT正式上限与下一AS判别实验（2026-07-29）

- global-8正式root
  `/data/ymdai/outputs/ember/pi05_source_sft_rank128_mixed8_dev_r4_b144_seed7_s2400_85bfe8e_20260728`
  已从identity fresh完成0→240并从完整step240 exact-resume到480。
  `metrics.jsonl`连续1..480，loss/gradient全部finite；累计276,480 action
  queries，24 tasks各11,520 samples、160次task visits，并覆盖全部50 action
  episodes。step30..480共16个checkpoint全部保留，终点LoRA、trainer和四rank
  state逐文件SHA256复算与manifest一致。两段从进程启动到终点封存合计约
  `11.32 GPU-hours`（4×A100），唯一trainable对象为`10,297,344`参数shared
  rank-128 LoRA。
- step60/120/180/240/300/360/420/480的fixed paired correct400为
  `63/83/85/98/90/62/90/105`，成功task数为
  `4/8/6/6/8/7/4/5`。八点均400 rows、36/36 shards、6/6 workers exit0，
  task/state/env/policy/noise完全paired且global long-first通过；analysis
  artifact SHA256为
  `9446b471016dfb99abb18f107de047163f3245cc9d009456673fe42115c8d2be`。
- step480相对420为`36 gained/21 lost,p=.0627`，只是一次边缘显著的反弹；
  相对step240为`37/30,p=.464`。逐task envelope为126，而任一checkpoint
  最好仅105，仍有21-success能力错位。它相对full-24 step400=`109`为
  `28 gained/32 lost,p=.699`，两个Spatial tasks同为0；因此global-8没有
  提高SFT上限，也没有消除能力漂移，不续到step600。
- corrected Source-SFT development observed-best最终封存为full-24
  step400=`109/400`。focused AS absolute门仍为
  `max(150,109+30)=150`。
- 隔离分支`codex/v6-cyclic8-training@eb7943b`已实现与SFT同构的Writer
  cyclic-8候选并通过正确worktree下全仓190 tests，但尚未合并或启动。
  SFT的直接负对照使“8-task update本身解决漂移”降权，因此不因沉没成本运行。
  下一步先做现有v6 checkpoint的单权重参数平均screen：显式记录源checkpoint/
  权重/hash，derived checkpoint只允许inference、禁止resume/warm-start；
  若不能同时改善absolute和breadth，再fresh改LR/优化器，最后才动
  Procedure→compiler。

## v6 checkpoint参数平均实现与screen封存（2026-07-29）

- 新增唯一的inference-only derived checkpoint owner与薄CLI。派生目录只含
  `writer.safetensors`和canonical manifest，记录source checkpoint路径、
  cursor、manifest/Writer SHA、均匀有理权重与tensor合同；训练resume和
  warm-start仍只接受原始`checkpoints/step_*`。
- outcome前封存四组候选与固定GPU映射：
  `{150,200}→GPU4`、`{200,400}→GPU5`、
  `{150,200,350,400}→GPU6`、
  `{150,200,250,300,350,400}→GPU7`。评测固定为correct400、seed7、
  50-video无放回、6 generators、batch16、6 persistent workers和global
  long-first；合同位于
  `configs/pi05_as_writer_v6_checkpoint_average_screen_v1.json`。
- 四份真实派生权重均已生成，原始及派生checkpoint全部保留。每份包含600个
  state tensors、12,064,064个元素；独立重算确认523个可训练浮点tensor按
  float32均值后回写原dtype，77个固定buffer保持一致，逐元素mismatch为0且
  全部finite。四份formal evaluation adapter authority均通过。
- focused Writer tests为`65 passed`，全仓为`190 passed`；
  `git diff --check`与screen JSON语法通过。下一动作是集成clean main并在
  GPU4–7启动四点paired correct400。

## v6 checkpoint-average评测、五臂和内部传递完成（2026-07-29）

- commit `ea99f65`已合入并push到clean main。四个derived候选分别在
  GPU4/5/6/7完成correct400，结果为`129/140/144/145`。全部输出均400 rows、
  36/36 shards、6/6 workers exit0、无重试/OOM；每个queue前12个claim均为
  horizon-520。winner六点late average相对raw macro200净增16，
  `37 gained/21 lost,p=.04794`，screen paired artifact SHA256为
  `09d4399662de821a1de0d6f38903eeba60a571fee2594c02fe6a445013dfb8ac`。
- winner的same/wrong/shuffled/reversed在GPU4/5/6/7各自完成full400，
  与已有correct合成`145/134/128/119/122`。四个run wall为
  `2279.47/2295.95/2315.81/2338.50s`；所有cache、results、rows、queue、
  logs、contract以及原始/派生checkpoint均保留，不做删除。
- paired checks全部通过：五臂同400个state keys、env/policy/noise，
  teacher video每task无放回双射；same为`+17` demo offset，另三臂复用
  correct demo；所有worker全attempt1且无adopt。correct相对后三臂精确
  p为`.03634/.001299/.006741`，各有5/6/5个正向tasks。same为
  `30/19,p=.1524`，aggregate差11，按预先写入的`<=10`保守边界记borderline。
  artifact SHA256为
  `9244b8db004f4155f9ee254bbddbaf013ee033640b6d9974c2b98cd283579d8b`。
- tmux `ember-v6-avg-late6-internal`在GPU4–7完成16-reference五条件内部检查，
  自然exit0；max allocated/reserved为`11.69/19.33GB`，probe wall
  `26.92s`。fixed-Core Procedure-only保留shuffled/reversed的
  effective-LoRA/action，Core-only近零；summary/rows/run-contract SHA256为
  `7596fbd...169d/b678403f...25d/4ed4aa43...639`。

## v6 fast-decay400 fresh正式合同准备（2026-07-29）

- 新sealed config
  `configs/pi05_as_writer_language_axial_v6_decay400_v1.json`只改变scheduler
  `decay_steps 2000→400`。authorities、information wall、完整v6 Writer、
  data、task-complete conditioning、B20、AdamW和seed逐对象核对完全相同；
  config loader通过。实际LR核验为macro
  `50/100/150/200/250/300/350/400 =
  2.8896e-4/2.5753e-4/2.1049e-4/1.55e-4/9.951e-5/
  5.247e-5/2.104e-5/1e-5`。
- 首段仍fresh0→200、每25 checkpoint且全部保留；评测50/100/150/200。
  除非首段出现可信多taskabsolute下降，否则exact-resume到400并评测后四点。
  该run不从raw或derived checkpoint warm-start。本文记录的是提交前合同；
  正式launch只使用包含这些变更的clean/pushed main，并紧邻执行live GPU4–7、
  storage和新output root核验，最终事实以run contract为准。

## v6 fast-decay400正式训练与八点评测完成（2026-07-29）

- commit `4efa737`的fresh run在GPU4–7完成0→200，随后从完整macro200
  exact-resume到400。两段各约一小时，metrics恰好1..400且全部finite；
  16个25-step checkpoint、optimizer/scheduler、4-rank sampler/RNG和
  trainer state全部保留。累计192,000 action queries、9,600 one-video
  conditions，信息墙记录test/validation action reads均为0。
- macro50/100/150/200和250/300/350/400分别在GPU4/5/6/7完成correct400；
  每点400 rows、36/36 shards、6 workers return0、全attempt1且无adopt。
  每task 50 teacher videos为无放回双射；每个queue前12个shards全部为
  horizon520，清空后才领取普通任务。八点结果为
  `106/64/111/133/132/117/138/143`。
- macro400相对原v6同点125为`46 gained/28 lost,p=.04739`；相对SFT109
  高34但仍低于absolute150。350→400仅`25/20,p=.5515`。fullcurve artifact
  SHA256为`99b04bf1...53d03`，完整checkpoint dynamics SHA256为
  `804689ca...05f32`。
- 训练和八点评测结束后GPU4–7均释放。个人空间约332GB；全部原checkpoint、
  derived checkpoint、LoRA cache、raw rows、queue、logs和results均保留，
  没有执行删除。

## v6 fast-decay checkpoint-average screen封存（2026-07-29）

- outcome前新增
  `configs/pi05_as_writer_v6_decay400_checkpoint_average_screen_v1.json`，
  固定四候选/GPU为
  `{350,400}→4`、`{200,350,400}→5`、
  `{200,250,350,400}→6`、
  `{150,200,250,300,350,400}→7`。
- 所有派生权重继续使用已验证的float32均匀平均、原dtype回写和
  inference-only manifest；不得用于resume/warm-start。评测固定correct400、
  无放回video、6 generators、batch16、6 persistent workers与global
  long-first。config SHA256为
  `07d115811cf6042d5d0246e9f91c304aed3e5289b53d898d17af0330526951f5`。
- screen只从包含本文与config的clean/pushed main执行；先生成四份derived
  checkpoint并完成CPU逐tensor复算/authority检查，随后GPU4–7并行评测。
  所有源checkpoint、派生checkpoint及评测cache/rows/results继续保留。

## v6 fast-decay checkpoint-average screen完成并暂停（2026-07-29）

- commit `7c3879c`的sealed screen在GPU4–7完成。四份derived checkpoint均
  独立复算为`max_abs_error=0`，formal authority通过；四卡各负责一份
  checkpoint，每卡6 generators、batch16和6 persistent workers。
- `{350,400}`、`{200,350,400}`、`{200,250,350,400}`、
  `{150,200,250,300,350,400}`的correct400为
  `139/135/129/130`，均低于raw macro400=`143`。最佳两点average相对raw
  为`18 gained/22 lost,p=.6358`；没有candidate达到absolute150。
- 所有run均400 rows、36/36 attempt1 shards、6 workers return0、无adopt；
  video无放回双射、paired seeds/noise与global long-first全部通过。
  artifact file/canonical SHA256为
  `ac6e1545...1d30/a9ffd347...9fdb`。GPU4–7已释放。
- owner明确要求本步后稍停讨论；未启动五臂、内部传递、gradient-conflict
  分析、第三训练段或下一fresh实验。全部checkpoint和评测证据继续保留。

## fast-decay续训、五臂与内部检查完成（2026-07-29）

- 同一fast-decay正式root从完整macro400 exact-resume到600；metrics连续
  1..600且finite，累计288,000 action queries和14,400 one-video conditions。
  新增450/500/550/600 correct400=`131/130/132/126`，每点400 rows、
  36/36 attempt1 shards、6 workers return0、无adopt；long-first与无放回
  video双射均通过。macro400仍为best，400→600显著下降`p=.01609`。
- macro400五臂在GPU4–7完成：
  `correct/same/wrong/shuffled/reversed=143/135/125/128/129`。五臂pairing、
  state/env/policy/noise、video和global long-first审计全部通过。
- 16-reference内部检查自然exit0；fixed-Core Procedure-only复现
  shuffled/reversed的effective-LoRA/action，Core-only近零。全部原checkpoint、
  cache、rows、queue、logs和结果保留。

## v6旧训练范式实现、profile与正式run（2026-07-29）

- commits `36f1cf4/86438ab/42ac8c0/a937e52/bad9a96`实现并封存唯一的
  rank-rotating v6 control。canonical config为
  `configs/pi05_as_writer_language_axial_v6_old_recipe_v1.json`；固定B20，
  对更大action-query batch fail closed。`B21`只曾作为未运行的候选名出现，
  没有profile、训练或评测。
- 最长105-frame真实视频完成fresh step0→1和exact-resume1→3；后两步稳态
  `20.091 queries/s`、约904 updates/hour，峰值allocated/reserved
  `76.94/83.72GB`。全仓192 tests通过后才启动正式run。
- 第一次formal launcher因额外CLI checkpoint列表终点900与sealed
  `total_steps=12000`不一致而在创建有效step/output前fail closed。失败log
  保留；移除该错误覆盖后以不变科学合同重启。
- 正式root
  `/data/ymdai/outputs/ember/pi05_as_writer_v6_oldrecipe_rankrotating_dev_r4_b20_seed7_s12000_bad9a96_20260729`
  完成900 updates；`metrics.jsonl`连续1..900，run summary记录72,000 action
  queries、3,600 video conditions和`3,626.731s`训练body。每100步checkpoint
  全部保留；每6 updates完整覆盖24 tasks。

## v6旧训练范式评测与内部传递完成（2026-07-29）

- step100/500/700/900分别映射GPU4/5/6/7并行correct400，结果
  `98/121/76/95`。每卡一checkpoint、6分布式Writer generators、
  batch16、6 persistent rollout workers；每个queue前12 shards均为
  horizon520，之后才领取普通task。四点均400 rows、36 shards、全attempt1。
- step500 single-checkpoint best完成五臂：
  `121/122/111/84/47`。same同档；wrong不显著且贡献集中，语义门失败；
  shuffled/reversed强显著通过。
- checkpoint dynamics与16-reference内部检查完成。old recipe显著放大
  Procedure→effective-LoRA/action的顺序差异，fixed-Core反事实完整复现，
  Core-only近零。训练和全部评测自然退出，GPU4–7释放。
- 当前按owner要求停下讨论；没有修改v6架构，没有启动后续fresh训练、
  one-shot或RL。

## v7第一性原理设计封存（2026-07-29）

- owner解除上一轮暂停边界，要求先记录需求/设计，再创建session-local Goal
  自主推进absolute performance。完整v7 authority已新增为
  `docs/action_forecast_writer_v7_design.md`。
- v7定义唯一的Task-Aligned Semantic Trajectory、frame-mean Core、8-token
  sparse Action Expert probes、forward Action–Effect binding、三层causal
  Procedure与Procedure-content-only compiler。
- owner进一步把8→1聚合收敛为单步joint action–effect pooling：全部`8×L`
  pairs直接归一化并形成每区间一个event，删除独立EventRead；Core直到
  compiler才首次与Procedure相遇。真实模块枚举为`10,312,192`，与更新后设计
  预算逐项吻合。
- root `AGENTS.md`、README、execution brief、active handoff、task plan和
  findings已同步下一fresh架构定位。architecture guard修改前baseline为pass，
  无hard violation、parallel version family或活动source diff。
- 设计落盘前现场只读核验：HEAD与origin/main均为
  `f920f4a0e13366864fee3334eb60beb56c4edf6d`，原worktree clean；GPU4–7为
  0MiB，GPU0–3存在其他用户进程且未触碰；个人空间约338GB；无EMBER训练/
  评测进程。
- session-local Goal已经建立。canonical Writer源码/config已原位切换到v7，
  v6 schema/checkpoint不兼容且没有并行可执行分支；全仓192 tests通过，
  architecture guard无hard violation或parallel family。

## v7 B20真实profile、resume与正式合同（2026-07-29）

- GPU4–7上B32、B24均在首个functional policy forward明确OOM；不再扫描中间
  batch。B20连续3个完整macro finite，首步含105-frame最长视频，三步wall为
  `19.234/17.492/17.447s`，后两步均值`27.477 queries/s`、
  `206.075 macros/hour`；峰值allocated/reserved为
  `77,020,274,176/83,647,004,672 bytes`。
- B20 root为
  `/data/ymdai/outputs/ember/pi05_as_writer_v7_profile_b20_jointae_r1_20260729`；
  run-contract/metrics/summary SHA256为
  `c0f1becf...e0ee3/fc1f361d...9dc8/6da42ada...fc25`。
- 独立resume root fresh0→1后exact-resume 1→3；checkpoint1未改写，task、
  video、query、LR与cursor身份和连续run一致，最大mean-loss绝对差
  `2.33e-5`。joint binder的`262,656/262,656`参数在真实step1→3全部变化，
  L2位移`0.08944`。
- 正式配置封存task-complete B20、teacher seed`20260722`和fast cosine
  decay400；首段fresh0→200、每25 checkpoint，共96,000 queries与4,800
  one-video conditions。实现/profile seal commit为
  `ca7db57d0c2d1ec2e7032a44b58238b6de35b1f4`，已push至`origin/main`。
  正式root预声明为
  `/data/ymdai/outputs/ember/pi05_as_writer_v7_jointae_taskcomplete_decay400_dev_r4_b20_seed7_s2400_ca7db57_20260729`；
  尚未启动正式训练或评测。

## v7正式训练、评测与内部根因完成（2026-07-29）

- 正式root完成fresh macro0→200及exact-resume200→400，metrics连续1..400、
  loss finite；16个every-25 checkpoint及完整resume状态保留。
- macro50/100/150/200/250/300/350/400 correct400为
  `82/106/114/120/101/114/115/106`。macro200五臂为
  `120/112/91/100/69`。
- refs1内部检查覆盖8个validation tasks。Action–Effect pair attention熵约为
  理论均匀熵的99.96%，有效8 probes约7.998；fixed-Procedure/vary-Core的
  effective-LoRA差异只有约0.1–0.2%，fixed-Core/vary-Procedure几乎复现全部
  差异。v7停止且GPU4–7释放。

## v8设计与canonical CPU实现（2026-07-29）

- 新authority`docs/action_forecast_writer_v8_design.md`记录hierarchical
  Action–Effect binding、Procedure-only EventRead与Core multiplicative
  gate。v7 source/config已原位替换，不保留并行可执行路径或checkpoint兼容。
- 每个Action anchor独立读取task-token effects，得到8个bound tokens后再
  聚合成一个event；Core gate只乘性调制Procedure slots，不增加Core-only
  value path。
- 真实枚举：binder`590,848`、compiler`1,469,696`、Writer
  `10,706,176`。聚焦38 tests和全仓192 tests通过；Markdown link audit零
  缺失、`git diff --check`通过，architecture guard无hard violation或
  parallel version/function family。shape/mask、Action/effect gradient、
  identity、`D=0`与`Procedure=0`硬约束均成立。
- 活动config为`configs/pi05_as_writer_language_axial_v8.json`，profile和
  formal状态均为pending。下一步是全仓回归/clean push，然后只在GPU4–7做
  B20三macro最长视频profile；失败才直接B16。

## v8 B20 profile、resume与formal seal（2026-07-30）

- live preflight：GPU4–7均0MiB/0%，无进程；个人`/data/ymdai`用量352GB，
  总盘余量2744GB。GPU0–3未进入visible set。
- B20 root完成3/3 macros，含105-frame最长视频；三步
  `19.243/17.506/17.450s`，稳态`27.463 queries/s`、
  `205.974 macros/hour`，峰值allocated/reserved
  `77,035,771,904/83,655,393,280 bytes`。B16未触发。
- 独立resume root完成fresh0→1→resume3；step1未改写，scientific cursor
  全同，最大loss差`4.7951e-5`。全部binder和Core modulation参数以及所有主
  模块step1→3均变化。
- config已恢复正式teacher seed`20260722`并封存B20、fresh0→200、
  every25 checkpoint；下一步是CPU复验、clean commit/push和正式launch。

## v8完成并切换v10（2026-07-30）

- v8 macro0→400正式训练与八点correct400完成；曲线
  `90/110/82/110/90/125/98/115`，best macro300五臂
  `125/121/110/110/117`。内部检查确认event被Effect主导，v8停止。
- owner批准Evidence-Preserving Dual-Stream v10并创建session-local Goal。
  `docs/action_forecast_writer_v10_design.md`、唯一canonical源码和不兼容v10
  config已完成；尚未封存的v9草案与v8 executable config原位退役。
- v10真实参数`11,627,520`；全仓192 tests与`git diff --check`通过。
- GPU4–7 B20 profile三步finite且包含105-frame视频，后两步约
  `26.38 queries/s`、`197.85 macros/hour`，峰值约`77.01/83.65GB`。
  fresh0→1→resume3通过，最大loss差`2.63e-6`，所有新增路径梯度可达。
- config已恢复正式teacher seed`20260722`，封存task-complete B20、
  fast-decay400、fresh0→400、every25 checkpoint。紧邻动作是clean
  commit/push和正式两小时训练。

## v10正式序列完成并按owner要求暂停（2026-07-30）

- main `5fd0a25`上的v10正式run在GPU4–7从identity fresh自然完成
  macro0→400；400行metrics finite，累计`9,600`个Writer视频条件、
  `192,000`个action queries，wall约`7,832.8s`。训练与全部rollout进程均已
  退出，GPU4–7释放。
- 12个single checkpoints的paired、每task 50 teacher videos无放回
  correct400为`95/103/84/89/82/90/96/96/89/96/97/91`；
  observed-best是macro50=`103/400`，未使用checkpoint融合。
- macro50五臂完整完成：`103/94/75/67/43`。same同档；
  wrong/shuffled/reversed相对correct的exact p依次为
  `.001762/1.01e-5/5.63e-13`，视频行为门通过，但absolute低于
  corrected Source-SFT `109`且未达150。
- refs1内部检查覆盖8/8 validation tasks。Core顺序不变、Procedure差异可完整
  传到LoRA/action、Procedure=0严格identity；同时Action变化远强于Effect变化，
  Effect attention近均匀，compiler将很小的Procedure slots通过RMSNorm调制为
  高增益Core content。同task换正确video的Procedure/action方差也显著高于
  v5.2，解释了强特异性与低absolute并存。
- 中间因main存在owner写入、未跟踪的Loom文档，评测clean-worktree guard按
  设计fail closed；失败调用未创建queue/run contract，输出和log已原样移入
  `.codex/tmp/v10_dense_failed_clean_guard_20260730/`。随后从同一commit的
  detached clean eval worktree完成全部正式评测，科学合同与结果未污染。
- owner最新指令为“v10做完就先停下”。没有启动Loom、one-shot或RL；Loom
  相关未跟踪文档和隔离worktree中的未提交草案均保持原样，未接入main。

## Loom canonical实现与正式启动门（2026-07-30）

- owner后续明确授权Loom，因此v10后的暂停边界已解除。Loom原位替换唯一
  canonical Writer/config；真实枚举参数为`12,855,552`，不从旧Writer
  checkpoint resume。
- 全仓191项CPU测试、compileall、diff check通过；architecture guard无
  hard violation或parallel version/function family。GPU4–7的B20三macro最长视频profile
  含105-frame条件，三步`20.463/18.397/18.367s`，稳态约
  `26.112 queries/s`与`195.843 macro/hour`；峰值allocated/reserved为
  `77,566,232,064/83,732,987,904 bytes`，B16未触发。
- 正式seed`20260722`下fresh0→1→exact-resume1→3通过。step1 checkpoint
  全文件未改写，task/video/query和LR逐步等于uninterrupted profile，最大
  mean-loss差`1.5891e-6`。首次尝试因两次进程加载之间修改config文件而被
  SHA合同正确拒绝，未污染checkpoint。
- 正式首段封存为GPU4–7、task-complete B20、fresh macro0→200、每25 macro
  checkpoint；结束后只比较single checkpoint 50/100/150/200，不做融合。

## Loom正式首段、correct曲线与内部停止判定（2026-07-30）

- main `1e5870f`上的Loom fresh macro0→200自然完成，wall`3,855.28s`、
  `4,800`视频条件和`96,000`action queries，训练机械、task-complete覆盖、
  checkpoint和online validation均完整。
- macro50/100/150/200的paired无放回correct400为
  `79/106/105/112`。每点400 rows、36/36 shards、无failure；macro200为
  observed-best，但比同macro、同recipe v6的`133`低21，也比同期v5.2的
  `132`低20，未触发第二小时。
- owner要求先做内部数值分析且暂停rollout。已停止自动启动的四个特异性臂；
  停止时cache=0、results不存在，未运行任何same/wrong/shuffled/reversed环境
  rollout。
- macro200内部五条件检查完成且没有环境交互：Core顺序合同、差异传递、
  compiler replay和zero-Teacher identity均通过；同时matcher近uniform、
  visual confidence近零、shuffled confidence/scale高于correct、
  Teacher–Policy gap近常数、Teacher支配LoRA且same-video方差偏高。Loom据此
  作为科学non-pass停止，不继续修补或续训。

## Recenter canonical实现与CPU合同（2026-07-30）

- owner明确授权在同一session持续自主推进，已创建session-local Goal；目标为
  single-checkpoint correct400至少150或稳定接近且显著高于旧架构，达到
  absolute门后才做行为级视频特异性。
- Loom首段四点correct400为`79/106/105/112`，内部gap/confidence/
  correspondence缺少可靠锚点。按owner“不得打补丁、必须从根因重设计”的
  要求，新增
  `docs/action_forecast_writer_recenter_design.md`并把Loom降为provenance。
- 唯一canonical源码已原位切换：恢复原生50-token Action mean，保留v6
  Semantic Core，新增25%径向上限的task-grounded transition residual与单路
  causal Procedure；compiler改为Core-keyed、time-centered raw Procedure
  values和amplitude-preserving slot mixer。Loom-only `relations.py`、双
  Procedure和gap compiler已退役，无平行可执行路径。
- 活动fresh config为`configs/pi05_as_writer_recenter.json`，schema、checkpoint、
  launch、eval adapter和episode evidence均切换为Recenter。Loom的profile/
  resume/gradient evidence没有复制；profile与formal状态保持pending。
- 精确参数枚举为`10,709,248`。确定性测试覆盖Core permutation、transition
  cap、Action-zero无旁路、causality、constant Procedure identity、Core gate、
  Procedure scale、step0 identity、staged gradient及零点finite backward。
- 修复审查发现的zero-RMS反向NaN：transition分母直接使用mean-square，
  diagnostic RMS detach；slot mixer用`torch.linalg.vector_norm`物理RMS及
  零subgradient处理零输入。
  targeted tests和全仓`196 passed`；额外覆盖bf16非2幂长度constant Procedure
  精确零与near-zero mixer有界梯度。compileall和diff check通过，architecture
  guard只有既有大文件review提示，无hard violation，active source净删约
  1,100行。紧邻动作是clean commit/push；之后只在GPU4–7重新做Recenter
  B20/B16 profile和exact-resume，不继承Loom seal。

## Recenter B20 profile、resume与formal seal（2026-07-30）

- main已在`93c7e32`封存canonical Recenter实现。GPU4–7独立完成B20三macro
  最长视频profile：包含真实105-frame条件，3/3 finite；后两步均值
  `25.808 queries/s`、`193.562 macro/hour`，峰值allocated/reserved
  `76.99/83.64GB`，B16未触发。
- 正式seed `20260722`下fresh0→1→exact-resume1→3通过；metrics、LR、
  task/video/query cursor连续，step1 checkpoint各文件hash在resume前后
  完全不变，validation/test action reads为0。
- profile step1→3间全部`10,709,248`个Writer参数变化，覆盖11个主模块组。
  config现已恢复正式teacher seed并seal为B20、fresh macro0→200、每25 macro
  checkpoint；紧邻动作是clean push后在GPU4–7启动约一小时正式段。

## Core-Program canonical CPU实现（2026-07-30）

- 新authority为
  `docs/action_forecast_writer_core_program_design.md`。canonical源码/config/
  launch/checkpoint/eval schema已原位切换；删除
  `configs/pi05_as_writer_recenter.json`，不保留兼容执行路径。
- compiler现为raw Core value reader、Core-keyed full raw Procedure reader、
  width512 bias-free strict bilinear和zero-preserving slot block；transition
  恢复v6 uncapped `A+R`。精确参数枚举为Writer `10,905,856`、compiler
  `1,665,792`。
- 模型合同tests `14 passed`；config/checkpoint/evaluation相关affected tests
  `34 passed`；最终全仓`194 passed`、compileall与diff check通过。
  architecture guard为REVIEW、无hard violation，active source净删643行。
- 按single-checkpoint合同删除checkpoint-average module/CLI，evaluation明确
  拒绝`derived_checkpoints`；RL-Writer在raw-video Core-Program接口完成重建和
  fresh retrain前于任何GPU/data加载前fail closed。GPU profile、resume smoke
  与formal训练尚未开始，不得继承Recenter evidence。

## Core-Program B20最长视频profile（2026-07-30）

- main `4769b36`在GPU4–7以profile-only teacher seed `172`独立完成3个
  task-complete B20 macros；首步覆盖task38/demo36的真实105个stride-5帧，
  三步loss/gradient均finite。
- 三步wall为`20.4094/18.5197/18.5874s`；后两步均值
  `25.8712 queries/s`、`194.0340 macro/hour`。峰值allocated/reserved为
  `76,993,247,232/83,644,907,520 bytes`，因此选择B20且不触发B16。
- profile step1→3的523个trainable tensor全部发生变化且finite，覆盖
  Meta-LoRA、Semantic Core、transition、Procedure、strict bilinear compiler
  和factor heads。配置已恢复正式teacher seed`20260722`；下一步为独立
  fresh0→1→exact-resume1→3 smoke。

## Core-Program exact-resume与正式seal（2026-07-30）

- 当前schema/commit的独立root先fresh到macro1，再从完整macro边界resume到3；
  metrics严格为`1,2,3`，累计queries为`480/960/1440`、video conditions为
  `24/48/72`，LR、task/video/query cursor和四rank RNG均连续。
- resume前后macro1的manifest、Writer、trainer和四个rank state逐文件hash
  不变；三步均finite，validation/test action reads保持0。配置现已封存B20、
  formal teacher seed`20260722`、fresh macro0→200和every25 checkpoint。

## Core-Program首段正式launch合同（2026-07-30）

- sealed config commit为`d67d9f5`；fresh identity、GPU4–7四rank、NUMA1、
  B20、2 workers/rank，macro0→200，每25保存。首段精确消费4,800个
  one-video LoRA conditions和96,000 action queries，不继承profile/smoke权重。
- output为
  `/data/ymdai/outputs/ember/pi05_as_writer_core_program_taskcomplete_decay400_dev_r4_b20_seed7_s2400_d67d9f5_20260730`；
  启动前个人目录`433.62GB`，formal与四点correct400预计新增低于7GB。

## Core-Program首段完成与correct400启动（2026-07-30）

- fresh macro0→200自然完成：200行连续finite metrics、8个every25 checkpoint、
  4,800个single-video LoRA conditions和96,000 action queries；training body
  `3,858.26s`，last-50 steady约`25.742 queries/s`，终点loss`0.10119`。
- 24 tasks每macro恰好各出现一次，DDP每macro一次同步；validation/test action
  reads保持0。8个checkpoint manifest、trainer和四rank state均通过校验。
- macro50/100/150/200现分别在GPU4/5/6/7并行做paired、无放回correct400；
  每卡6 workers、4 Writer generators、generation batch8，全局long-first。
  四个run均已完成prepare，绑定各自唯一raw checkpoint。
- macro150首次launcher的source-run参数误填，合同校验在模型/GPU加载前拒绝；
  删除508B空壳后已按正确source-base合同fresh重启，不影响其它三点。

## Core-Program评测完成与停止（2026-07-30）

- macro50/100/150/200 fixed correct400全部完成，结果为`84/75/60/76`；四个
  roots均为400 rows、8 tasks×50，全部50 teacher videos无放回恰好一次，
  mapping和seed schedule完全相同，teacher action reads为0。
- observed-best macro50相对v5.2 step900与v6 macro200分别净低48/49，
  paired p值为`3.88e-7/4.76e-7`。按预定门停止Core-Program，不续训、不做
  行为级视频控制。
- GPU4–7完成macro50 refs2内部检查：16条件、无rollout/reward、validation
  actions0、teacher states0。结果定位为compiler压缩强AC顺序信号和bilinear
  梯度耦合，不是上游完全忽略视频。

## Prior–Innovation canonical CPU实现（2026-07-30）

- 新authority为
  `docs/action_forecast_writer_prior_innovation_design.md`。唯一canonical
  compiler已整体替换为Core semantic prior + centered Procedure innovation；
  Core-Program config/schema/class退役，不保留并行执行路径。
- 新fresh config为`configs/pi05_as_writer_prior_innovation.json`，所有profile、
  resume、gradient和formal证据重置pending；不能继承旧硬件证据或checkpoint。
- 精确Writer/compiler参数为`10,643,968/1,403,904`。focused不变量、全仓
  `195 passed in 16.13s`、compileall、JSON、diff check全部通过；
  architecture guard无hard violation。尚未使用GPU。
- `/data/ymdai`当前占用约`438.61GB`，本轮profile、训练和四点correct400预计
  新增`6–8GB`，低于500GB hard cap。紧邻动作是clean commit/push，然后只在
  GPU4–7做最长105-frame B20三macro profile和exact-resume。

## Prior–Innovation B20 profile与formal seal（2026-07-31）

- main `7b7abf1`在GPU4–7完成独立最长105-frame B20三macro profile，全部
  finite；稳态约`25.818 queries/s`、`193.635 macro/hour`，峰值
  allocated/reserved约`76.99/83.64GB`，B16未触发。
- 正式teacher seed下fresh0→1→resume1→3通过；step1逐文件SHA未改写，
  metrics/LR/cursor/RNG连续，累计72 videos和1,440 queries，validation/test
  信息墙读数均为0。所有主模块finite且可达。
- config已恢复正式seed并seal为B20、fresh macro0→200、every25；下一步为
  clean push sealed evidence后，从identity启动约一小时正式段。

## Prior–Innovation首段正式launch（2026-07-31）

- sealed config commit为`807266b`；live preflight确认Git clean/pushed、
  GPU4–7各约81.1GB free且无compute process，个人目录`439.39GB`，正式段及
  四点correct400仍低于500GB hard cap。
- tmux `ember-prior-innovation-m200`已从fresh identity启动：GPU4–7四rank、
  NUMA1、B20、2 workers/rank、macro0→200、every25，精确预算4,800个
  one-video conditions和96,000 action queries。
- output为
  `/data/ymdai/outputs/ember/pi05_as_writer_prior_innovation_taskcomplete_decay400_dev_r4_b20_seed7_s2400_807266b_20260731`；
  首段完成后只做macro50/100/150/200 paired correct400，不融合checkpoint。

## Prior完成与Target-Spectral CPU实现（2026-07-31）

- Prior formal macro0→200已自然完成；macro50/100/150/200 paired correct400
  为`100/61/89/88`。没有启动第二小时或行为级视频控制。
- 新authority为
  `docs/action_forecast_writer_target_spectral_design.md`。唯一canonical
  compiler已从320个rank-level semantic slots替换为38-target-first、
  rank-last spectral compiler；Prior config/schema退役。
- Target-Spectral Writer精确参数`14,495,744`。A/U采用FP32 reduced-QR并
  固定R对角符号；已补强共同方向压力测试、effective-LoRA视频条件测试、
  38-target拓扑guard、BF16输入的FP32 Procedure centering以及不手工开权重的
  三步gradient staging。
- 当前训练合同没有变化：一条video生成一套LoRA，action query跨episode；
  full24等权、B20、一次AdamW。profile/resume/formal evidence全部重新置为
  pending，不能继承Prior。
- 下一步是完成全仓验证、clean commit/push，只在GPU4–7做最长视频B20 profile
  和exact-resume；通过后fresh macro0→200并评测四个single checkpoints。

## Target-Spectral B20最长视频profile（2026-07-31）

- main `f8bbce6`在GPU4–7以profile teacher seed172完成三个task-complete
  B20 macros；首步包含task38/demo36真实105-frame条件，三步loss和gradient
  均finite。
- 后两步均值`25.488 queries/s`、`191.159 macro/hour`；峰值allocated/
  reserved为`77,074,980,864/83,649,101,824 bytes`，因此B16不触发。
- step1→3的530个trainable tensor中458个变化，所有主模块finite且变化。
  唯一整组暂未变化的是72个Action Meta-LoRA A；这是spectral scale、
  Procedure AdaLN与Meta-LoRA B连续zero-init造成的预期四步staging，配对B已
  全部变化。配置已恢复formal seed，下一步做独立fresh0→1→exact-resume1→3。

## Target-Spectral exact-resume与formal seal（2026-07-31）

- formal seed `20260722`的独立root先fresh0→1，再从完整macro1恢复到3；
  metrics严格为1/2/3，LR、task/video/query cursor、累计queries
  `480/960/1440`和video conditions`24/48/72`连续且finite。
- resume前后macro1的manifest、Writer、trainer和四rank state共七个文件SHA
  完全不变；validation/test action reads和test video value reads均为0。
- formal-seed step1→3同样为530个trainable tensors中458个变化，所有主模块
  finite且可达；72个Action Meta-LoRA A按分级zero-init延迟，配对B已变化。
  config现已seal为B20、fresh macro0→200、every25。

## Target-Spectral训练、correct400与内部分析完成（2026-07-31）

- sealed commit `aa9d89a`的fresh run自然完成macro0→200：200行finite metrics、
  4,800个single-video conditions、96,000 queries、every25的8个完整checkpoint；
  training body `3920.15s`，终点loss/grad为`.10023/.06443`，峰值allocated/
  reserved约`77.08/83.65GB`。全部checkpoint manifest通过校验。
- macro50/100/150/200在相同8×50 fixed states、每task teacher video无放回
  0–49和同一RNG配对下得到`30/12/18/34`。macro200逐task为
  `12/0/0/6/13/1/1/1`；31/34成功集中在三个tasks。独立审计确认四份结果、
  36/36 shards、400 LoRA caches、worker return codes和hash链完整。
- 按门停止行为评测与续训。CPU rank/layer/video分析完成，产物SHA256
  `4d7dfc68efa84b9863b8a6d9b7d4ab717f529018992b6c316c06320631d10a89`；
  Target m200 stable rank/norm为`3.3245/25.87`，v6 m200为
  `1.00017/94.71`。Target q/v跨层余弦仅`.032/.066`、layer-energy CV高达
  `1.294/.805`，确认强制正交拆散了v6高增益协调方向。
- GPU4–6现场有他人进程，按owner要求没有挤占；只在空闲GPU7单卡完成16条件
  non-rollout内部探针。首轮被旧probe字段`transition_norm`拦截，核对当前模型
  后只把capture更新为`transition_key_norm`，原始复现随后成功。该故障仅影响
  disposable instrumentation，不影响训练或正式correct400。
- 内部结果证明Core/Procedure工作且order差异传到LoRA/action；失败集中在
  compiler写入几何和functional-loss→closed-loop错位。当前暂不正式训练，
  只允许使用GPU4–7中现场空闲卡分析；下一步由owner讨论后封存保留v6主方向、
  仅增加可选视频innovation rank的架构。

## v5.2 task-complete控制实现（2026-07-31）

- owner解除临时训练暂停，要求先完成原版v5.2拓扑与成熟full24 fast-decay400
  训练的两小时对照，再基于全部证据设计下一架构。
- 当前main原位恢复v5.2 Core/Procedure/320-slot AdaLN compiler，删除
  Target-Spectral-only compiler源码；保留现有cost-balanced long-first
  task-complete训练、raw-video信息墙、checkpoint/resume与evaluator。
- 新config固定B20、4 ranks×6 tasks、24 tasks/macro、480 queries/macro、
  LR`3e-4`、warmup17、decay400到`1e-5`、every25、0→200→400。
- 精确参数预算`10,237,704`；聚焦模型/训练/checkpoint/evaluator合同
  `41 passed`，全仓首轮`188 passed/1 message-only failure`，该消息测试已
  同步修正。architecture guard为REVIEW、无hard violation，active source
  净删约609行。

## v5.2 task-complete B20 profile与formal seal（2026-07-31）

- main `62598d3`全仓fresh回归为`189 passed`；核心model/temporal/
  video-program逐文件SHA与正式v5.2 commit `529da6b`完全一致，随后已push。
- profile seed172在GPU4–7完成三个full24 macros；首步包含task38/demo36的
  105 sampled frames。三步loss/gradient finite，峰值allocated/reserved为
  `76,967,302,656/83,638,616,064 bytes`；B20通过，不触发B16。
- 现场GPU4和6有未干扰的他人轻量进程；后两步均值`18.943 queries/s`、
  `142.074 macro/hour`。因此相同400-macro科学预算在当前共卡吞吐下约需
  169分钟body，formal wall上限如实放宽到190分钟，而不是偷偷减少updates。
- 独立formal teacher seed `20260722`先fresh到macro1，再从完整checkpoint
  exact-resume到3。metrics、LR、task/video/query cursor连续为`1/2/3`、
  `480/960/1440 queries`和`24/48/72 videos`，validation/test action与test
  video读取均为0。
- formal-seed step1→3的519个trainable tensors中447个变化，Core、Procedure、
  compiler、factor heads和三条semantic projection/Meta-LoRA主路径均可达；
  72个Action Meta-LoRA A因zero-B与BF16短profile分级暂未量化变化，配对B全部
  变化，且旧v5.2 step100→900证实72/72 A随后均变化。配置已seal为fresh
  `0→200→400`、B20、every25。
- 清理108个已核验无进程引用、可由评测重建的`writer_lora_cache`，删除约
  `105.77GB`；正式result rows、contracts和checkpoints均保留。个人占用从
  `453.12GB`降到`347.35GB`，这些缓存未进入回收站、只能重新生成。

## Post-seal Target-Bound完成与Semantic Factor-Basis launch（2026-08-02）

- Target-Bound formal macro0→200、四个paired correct400=`75/120/90/110`和winner
  macro100 refs1内部分析均已完成；按一小时门停止，不resume、不做五臂。
- 基于最早失效接口完成Semantic Factor-Basis canonical替换，commit`e87363f`已push；
  参数11,159,296，55项聚焦回归通过。longest105 B20三macro及formal-seed
  fresh0→1/exact-resume1→3通过后，seal commit`f5ddfe3`已push至实验分支和main。
- 22:37:45 UTC从clean detached`f5ddfe3`和fresh identity启动formal macro0→200：
  tmux`ember_sfb_formal_f5ddfe3`，GPU4--7 DDP4、NUMA1、B20、every25。前3 macro
  loss为`.15404/.15159/.14764`，macro2起五主block梯度可达，无clip/OOM/nonfinite。
  output/log及后续评测root逐项登记在
  `/data/ymdai/migration_manifests/ember_postseal_20260802/assets.tsv`。

## Semantic Factor-Basis首小时评测与第二小时启动（2026-08-03）

- fresh0→200自然完成：200 finite macros、96,000 queries、4,800 single-video
  conditions、8个every25 checkpoints；无OOM/nonfinite/clip，validation/test action
  reads保持0。
- GPU4--7各负责一个checkpoint，6 persistent workers/card完成相同paired、无放回
  correct400；macro50/100/150/200为`69/91/118/127`，全部36/36 shards和400 rows。
- macro200 breadth8、50→200 gained/lost=`68/10`，但150→200仍有`38/29`换手；结合
  内部task routing及A/E/D→BA/action路径证据，通过第二小时门。
- 00:32:58 UTC从frozen`f5ddfe3`和原root step200 exact-resume到400；tmux
  `ember_sfb_resume400_f5ddfe3`，log为
  `/data/ymdai/logs/ember/pi05_as_writer_semfactor_postseal_resume200to400_r4_b20_seed7_f5ddfe3_20260803.log`。
- 等待评测期间完成variance-reduced estimator实现、18项focused测试和design，
  commit`1d04ae5`已push到branch与main；尚未使用GPU，不构成效果证据。

## Semantic Factor-Basis第二小时与A100 GPU封存（2026-08-03）

- frozen`f5ddfe3`从macro200 exact-resume到400正常结束；完整run为400 macros、
  192,000 queries、9,600 single-video conditions、16个every25 checkpoints，all
  finite、0 clip、validation/test action reads为0。
- GPU4/5/6/7并行完成macro250/300/350/400的400-row paired correct；结果为
  `117/81/126/120`，全部worker return code0。结合首小时，single winner仍是
  macro200=`127`；未过strong absolute门，不做正式五臂。
- evaluator第一次prepare因清理后的LIBERO site-packages assets symlink不存在而在GPU
  前退出；四个root没有rollout。补入已封存
  `EMBER_LIBERO_ASSETS_ROOT=/data/ymdai/ember_assets/datasets/libero-assets/0b3ea86...`
  后从空root重启，四卡各6 persistent workers正常完成。该问题不影响任何科学row。
- 训练metrics与checkpoint状态完成CPU drift审计：后半段task mean能量约`4.2%`、
  candidate-negative tasks为0、factor约占`97%`，Adam一阶moment跨50-macro近正交、
  二阶moment高度稳定；结果写入`findings.md`与SFB design。
- VR首个profile在macro前发现mode字符串未接入`as_step`；`50662a8`用一行runtime接线
  和参数化回归修复，4个focused tests通过并push远端分支。失败root/log已删除。
- 修复后longest105 B20三macro通过；formal seed另行fresh0→1再exact-resume1→3。
  retained roots：

```text
/data/ymdai/outputs/ember/pi05_as_writer_semfactor_vr_postseal_long105_profile_r4_b20_seed172_50662a8_20260803
/data/ymdai/outputs/ember/pi05_as_writer_semfactor_vr_postseal_formalseed_resume_r4_b20_seed7_50662a8_20260803
```

- 02:42 UTC后停止全部A100 GPU工作。剩余动作仅为config evidence seal、Git/doc push、
  post-seal migration ledger更新与无进程只读核验；不启动VR fresh0→200。

## Post-seal迁移增量封存（2026-08-03）

- 将`docs/a100_to_bci_migration_handoff.md`提升为最终双阶段迁移authority：原封存集
  不重传，第二次同步只取post-seal ledger的`must-transfer`行。
- `/data/ymdai/migration_manifests/ember_postseal_20260802/assets.tsv`已补齐所有正式
  root的实时bytes；34个必迁对象合计`16,483,938,529` bytes，逐项存在且尺寸一致。
- 12个新增formal correct400 root均重新核验为400 rows；VR profile/resume明确列为
  do-not-transfer/debug-only，代码和配置只经Git迁移。本窗口没有MemLLM新增资产。
- 聚焦回归为3 passed；两份VR JSON可解析，Git diff无空白错误。live tmux为空，
  没有EMBER训练、评测或GPU分析进程。

## Task-Relative Flow-Credit实现与AS profile seal（2026-08-04）

- owner恢复持续推进并要求中间问题自行深入分析。复核历史RL后建立
  `docs/action_forecast_writer_relative_flow_credit_design.md`：恢复唯一v6 Writer做
  独立AS cold start，随后关闭teacher action，以K4 LOO advantage、Nmc4逐sample ratio、
  PPO正项和SPO负项做Writer-only reward训练。
- 原位替换`src/ember/rl_writer/*`、reward replay与evaluator接线；删除旧final/task-local
  config和success-only活动路径。新增`flow_credit.py`为唯一数学owner；Writer live
  rollout生成从大而混杂的inference authority文件拆到`live_adapter.py`。architecture
  guard无hard violation。
- 聚焦RL/reward/evaluator 43项通过；全仓为避免同进程累计内存峰值拆为135项和75项，
  合计210项全部通过。v6 identity/freeze/schema、K4 advantage、正负梯度符号、failure
  replay、实际world-size assignment与checkpoint resume均覆盖。
- live检查：`gpu01`八卡均有cyzhao进程；`gpu02`物理0和6由yfwang占用，1/2/3/4/5/7
  空闲。仅使用后六卡完成
  `runs/outputs/pi05_as_writer_v6_relative_flow_coldstart_profile_r6_bci_20260804`：三步
  `33.464/30.886/30.977s`，峰值allocated/reserved
  `34,948,858,880/44,816,138,240` bytes，最长105帧、0 OOM/clip。
- 独立root
  `runs/outputs/pi05_as_writer_v6_relative_flow_coldstart_resume_smoke_r6_bci_20260804`
  fresh0→1再exact-resume1→3；合同`1d2290ea...d457a87`，metrics 1/2/3、累计1,440
  queries/72 videos、0 validation/test action reads。六卡随后自然释放。
- config已seal为fresh0→400、every25、首段stop25；profile checkpoint禁止warm-start。
  下一步是clean commit/push后正式fresh0→25，再用canonical reward cycle的pre-update
  K4 ledger做24-task coverage与最长failure/two-epoch profile。

## Task-Relative Flow-Credit AS25与reward profile（2026-08-04）

- clean pushed commit上的正式AS fresh0→25完成：25个full24宏步、12,000 queries、600
  videos、wall`810.991s`、0 OOM/clip，step25完整checkpoint原子发布。
- 首次reward初始化发现RL环境池没有绑定sealed LIBERO runtime asset cache，停止在rollout
  前并由`c1029a5`根修；retry1又发现非连续CUDA选卡时EGL错误使用local rank，15条诊断
  ledger后在参数更新前停止，由`6ff7599`根修并写入`AGENTS.md`。两个root均不作科研结果。
- retry2完成96/96 K4 rollouts和两epoch Nmc4 PPO/SPO：25 successes、12/24 tasks有
  success、9 mixed，coverage失败；28,085环境动作，wall`1066.374s`，峰值reserved
  `45,183,139,840` bytes。ratio/clip/grad机制健康，完整cycle1 checkpoint仅作profile。
- 六卡自然释放。下一步从同一AS root和step25 checkpoint exact-resume到step50，再用
  新的pre-update K4 ledger裁决coverage。

## Task-Relative Flow-Credit AS50与credit collective根修（2026-08-04）

- 同一正式AS root以合同兼容代码resume从step25续到50；累计50 rows、24,000 queries、
  1,200 videos，segment wall`816.191s`，step50 checkpoint完整，0 OOM/clip。
- step50首次K4 run完成96条pre-update ledger：38 successes、14/24 coverage、10 mixed、
  4 all-success、10 all-failure。相对step25严格配对gained/lost/retained=`19/6/19`，
  coverage`12→14`；总体上升但task5/16失去coverage，门仍未过。
- 首次run在credit阶段因0-mixed rank提前进入NCCL sum触发480秒watchdog；96条ledger
  保留，但无update/checkpoint。`e5bca71`加入每epoch独立FileStore all-rank-ready，
  聚焦回归25项通过并push到branch/main。
- 原六卡、96 rollout、两epoch规模retry完成；96/96 rollout JSON与失败run字节级一致，
  两epoch finite、完整cycle1 checkpoint、0 watchdog/traceback。wall`1465.939s`、
  peak reserved`40,342,913,024` bytes。六卡自然释放；下一步只续AS step50→75。

## Task-Relative Flow-Credit AS75与reward profile（2026-08-04）

- 同一正式AS root从step50 exact-resume到75，累计75 rows、36,000 queries、1,800
  videos；segment wall=`805.356s`，step75 checkpoint完整，0 OOM/clip。首次resume因live
  选卡形成`4+2` NUMA rank分布而与root sealed`3+3`不符，在训练前被合同正确拒绝；
  改用同节点`1,2,3,4,5,7`保持原topology后完成。
- step75 K4完成96条pre-update ledger：47 successes、18/24 coverage、13 mixed、5
  all-success、6 all-failure。相对step50严格配对gained/lost/retained/both-fail=
  `21/12/26/37`，coverage`14→18`；task`9/16/19/25/37`获得、task4失去coverage。
- 两epoch ratio为`[.97777,1.02659]`/`[.91171,1.08119]`，positive clip=`0/.000247`、
  grad=`.03184/.02709`；rank mixed=`3/1/4/1/3/1`，0 watchdog并保存完整cycle1
  checkpoint，peak reserved=`40,340,815,872` bytes。下一步只续AS step75→100。

## Task-Relative Flow-Credit AS100与内部审计切换（2026-08-04）

- 同一正式AS root exact-resume75→100完成：100 metrics rows、48,000 queries、2,400
  videos、step100完整checkpoint，segment wall=`805.085s`，0 OOM/clip/nonfinite，六卡
  自然释放。24 train tasks各2,000 queries、100 visits、50 unique videos。
- step100 K4 pre-update为52/96 success、17/24 coverage、11 mixed、6 all-success、7
  all-failure；24,275 environment actions。相对step75严格配对gained/lost/retained/
  both-fail=`14/9/38/35`，共同noise prefix无差异；task20失去coverage且没有新task进入。
- 两epochratio=`[.98452,1.00771]`/`[.88801,1.06045]`、clip均0、grad=
  `.02535/.02563`，peak reserved=`45,183,139,840` bytes；两轮finite、完整cycle1、0
  watchdog。coverage失败，profile checkpoint不续、正式RL不启动。
- owner允许IL/RL自主选择但要求先理解LoRA质量。新增聚焦内部审计入口
  `scripts/analyze_relative_flow_coldstart.py`：24 train tasks×demo0--4×AS四点测真BA谱、
  rank-coordinate energy与video variance；按split结构固定8-task面板测固定action传递，
  0 target-action/validation/test reads。先做真实profile，封存后再据结果裁决下一方法。
- clean`2b775f0`正式四点审计完成96 rows，wall291.333s、peak reserved19.308GB。
  norm持续增大但stable rank仍约1；same-video energy从step50起约`.13%`平台，失败tasks
  的video BA差异反而更大；reversed/shuffled到fixed action路径有效。正式拒绝把rank、
  scale或更多store作为下一小改，下一段按coverage合同只续AS100→125。

## SFT-Anchored Tangent-Basis formal与strict裁决（2026-08-05）

- clean`059d40f`在`gpu01:1,2,3|4,5,7`完成fresh formal cycle0→1：96 rollout、61
  successes、11 mixed、5 all-failure、two finite updates，wall`2046.03s`、peak
  reserved`19,478,347,776` bytes、0 OOM/watchdog/action-wall reads；完整cycle1与6 rank
  状态原子封存，设备自然释放。
- 同checkpoint strict correct400完成400 rows、0 error/retry，结果142；历史v6-fast
  macro400同panel为143。严格配对gained/lost/retained/both-fail=`20/21/122/237`、
  breadth`6→7`、union/intersection=`163/122`、`p=1.0`。配对artifact写入eval root的
  `paired_to_macro400_analysis.json`。
- 预注册cycle2门失败，当前root禁止resume。该实验只作为固定factor-output basis的消融：
  aggregate未崩但能力仍换手，不能把142写成fresh架构成绩或LoRA/video问题已解决。
  下一阶段回到fresh identity的policy-coordinated LoRA generator设计。
# Phase-Aligned K4 formal launch provenance guard（2026-08-07）

- sealed profile后的第一次formal launch在任何output root、checkpoint、metrics或GPU计算产生前停止。
  原因是fresh formal guard硬编码比较`origin/main`，而本项目当前clean pushed write authority明确为
  `origin/codex/bci-continuation`；HEAD实际上已与branch upstream一致。
- canonical `git_state`现在显式记录configured upstream及其commit，fresh formal继续要求clean
  worktree且HEAD必须等于该upstream commit。聚焦contract回归18项通过；该修复只校正launch
  provenance，不改变K4视频输入、Writer图、优化器、数据或checkpoint语义。

## v6-Prior B10 profile完成并解锁formal（2026-08-09）

- gradient seal后继commit`5fbcb27`已clean/pushed并有独立frozen worktree。gpu01 fresh0→1完成后原卡被
  他人取得，保留该partial chain且未混入比较；重新live检查后仅使用空闲`gpu02:0--5`，未触碰6/7。
- retry1 resumed链fresh0→1+exact-resume1→3与contiguous0→3均在tmux自然完成，所有invocation exit0，
  两root各3 metrics、macro1/3 checkpoints、completion，结束后六卡释放。
- 只读assembler先在原门下精确定位macro3 aggregate tolerance false negative；量化Writer、Adam、实际
  更新量及metrics后，只修离线比较器的state-specific aggregate tolerance，没有修改或重跑训练。
- v2 checkpoint/contract定向回归`11 passed`、正确加载`.env.local`后的全仓回归`247 passed`；同一
  retained roots随后完整assemble通过。canonical config已
  原样嵌入v2 artifact evidence，并把profile/formal同时置为
  `sealed_from_live_a40_resume_profile_evidence`；`runtime_for_mode(..., formal)`返回`(50,(10,25,50))`。
- 当前没有EMBER GPU进程，也没有v6-prior新strict成绩。authority同步和全仓回归已完成；下一操作是
  clean commit/push和formal frozen worktree，然后live选卡运行macro0及fresh0→50。

## Reward-Credit B8/all-mixed profile完成并解锁formal（2026-08-10）

- `e6024cf`从clean pushed frozen worktree在live空闲`gpu02:0--5`完成full24×K4×Nmc4 B8 discarded
  profile；torchrun/tee exit0，completion passed，未保留checkpoint/metrics，结束后六卡均0MiB/P8。
- raw v2复算与stored evidence完全相等：11 mixed/13 homogeneous、60/36 success/failure、4452 chunks，
  all-mixed 11/11 LoRA A/B/action非零，rank48、negative/correct`.01704835`、closure0，0 runtime fault。
  functional forward总数928且physical batch唯一候选为8。
- 主分支随后修复start/run-contract中的旧字面量，明确collectives字段只描述parameter update，并扩充
  profile seal：核对sealed optimization/runtime、raw唯一B8公式、completion、单次fresh invocation、A40
  topology和无checkpoint。active config已置`formal_ready`，discarded profile state不进入训练。
- formal evaluator启动门随后完成CPU收口：exact B8、validation 8×50、without-replacement、同一clean
  pushed/frozen commit、checkpoint/root和correct-before-controls均fail-close；historical bypass删除，control
  trigger与cycle2 support解耦，prepare以NFS原子准备锁、私有staging和一次目录rename发布，正式Reward的额外
  parser condition不可绕过六臂登记。真实`/data1` NFS smoke通过；结构重归属后architecture guard 0 hard、
  无parallel family；全仓`358 passed in 23.72s`。
- 下一步是完成全仓验证、clean commit/push，创建新formal frozen worktree；重新live检查双节点、A40和
  `/data1` quota后从`M0=0,Lambda0=I`跑cycle0→1并立即strict correct400，不跑80-row screen。

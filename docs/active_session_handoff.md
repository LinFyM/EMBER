# EMBER Active Session Handoff

## 1. Current truth

这是当前跨session科研入口，覆盖所有历史文档里的旧“当前”“下一步”和阶段性暂停。

- Goal仍未完成：同一shared method、同一single checkpoint的strict paired correct必须严格超过
  `150/400`并尽可能继续提高，同时要求真实视频时序因果性、same-task鲁棒、breadth、低checkpoint
  漂移和可重复累积。
- Owner已授权持续自主推进，只有实质阻塞才沟通，并已允许subagent承担独立、只读或隔离写入的
  加速工作；主进程仍负责统一科研判断和最终验证。
- canonical仓库是`/data1/user/ymdai/projects/EMBER`，主写分支是`codex/bci-continuation`。
  retained formal GPU工作必须来自clean pushed commit的frozen worktree。
- 当前没有EMBER GPU进程。任何新GPU工作前必须实时比较`gpu01/gpu02`；没有固定6卡上限，使用当时
  所有真正空闲、健康且能提高有效吞吐的A40，不等待凑卡、不dummy占位、不触碰他人compute进程。
  训练多卡固定`NCCL_P2P_DISABLE=1`、NUMA physical/local rank映射和deferred-NCCL；独立评测按live空卡
  动态扩展cost-balanced queue。

### 1.1 Active Q/V Rank-Reserved Native Reward Compiler（2026-08-11）

- Reward-Credit已从clean frozen`e3857f73ce92fa7f790a7e49f8166d7e5ef5b9e5`完成formal cycle0→1。
  root=`runs/outputs/pi05_v6_reward_credit_program_cotangent_formal_cycle0to2_r6_k4_nmc4_b8_balanced_20260810`；
  natural exit0、24 tasks/96 rollouts、11 mixed/13 homogeneous、B8、0 OOM/nonfinite，macro1 checkpoint完整。
- 预注册correct root=
  `runs/outputs/pi05_v6_reward_credit_program_cotangent_correct400_cycle0001_20260810`；400 rows、strict=
  `134/400`、breadth6、per-task按Spatial1/3、Object1/3、Goal3/6、Long1/2为
  `1/4/46/31/0/38/14/0`。相对zero-Program exact macro0仍为134且严格`14 gained / 14 lost`，因此不续
  cycle2、不补controls、不扫reward scale/K/Nmc/RLS参数。
- 这个non-pass后的分层诊断定位出新信息：correct Program、analytic FactorHead tangent和continuous
  effective-BA均保留task-common与same-video结构，wrong/shuffled/reversed Program量级约correct的`1.15%`。
  首个失效接口是36个q/v target的原生BF16 factor materialization：约`1e-8 RMS`的factor delta远小于
  非零A/B约`1e-4`的局部ULP，own-target native cosine约`.037`。action四个factor为FP32，路径仍健康。
- FP16 direct、fixed dither、local-CD/ULP搜索、gauge/global scale、balanced rank14/15和absolute rank16
  refactor都已有直接non-pass。唯一active候选由
  `docs/action_forecast_writer_qv_rank_reserved_native_reward_design.md`定义：q/v保留14个pivot-selected
  原生B columns并重解A，两个physical zero-B slots承载condition-local stable-Reward rank2 residual；action
  保持原full-rank16 FP32。
- full80 generation-only artifact=
  `runs/outputs/pi05_reward_qv_pivot_rank14_plus2_transport_v1_e3857f7_20260811/analysis.json`正式过其几何门：
  q/v base error约`.0007523`、每task max≤`.001302`，rank2 capture`.9997088`，dynamic energy-weighted
  cosine`.9975247`、task-common`.998448`、video-centered`.950556`，action exact；但0 policy action forward、
  0 rollout、0 update，不能冒充性能。
- 当前顺序只允许：canonical单一路径实现；单卡同32-request panel的B8/16/32吞吐profile和cycle1 vertical
  fixed-action smoke；先跑新rank14 zero-Program strict400。若correct<130、breadth<6或相对旧134 lost>10，
  立即reject且不跑第二个400。只有base过门，才用小型load-only manifest读取原84MB Program跑rank14+2
  cycle1 strict400；只有correct≥144、breadth≥6、相对新macro0 lost≤6且gained>lost才算load-only通过并补
  同checkpoint controls。140--143只作诊断性non-pass，不授权新训练；>150仍须完整六臂。两项行为门前不授权
  新训练。
- 新旧两个400均复用同一official state/video/env/policy RNG schedule，但必须新生成LoRA cache；旧macro0、
  旧Reward cycle1结果和Program tensor直接作为immutable baseline，不重跑、不复制84MB Program。evaluator是
  单节点、无NCCL；每次优先关注`gpu02`并选单节点当时最多的空闲卡，不为跨节点碎片改launcher或等待凑卡。

### 1.2 Historical Reward-Credit implementation and retained evidence（2026-08-10--11，formal cycle1已裁决）

- 第39.5节当时的唯一后继曾在canonical path原位实现；retained config为
  `configs/pi05_v6_reward_credit_program_cotangent_v1.json`，profile artifact与当时的`formal_ready`状态均已封存。
  formal cycle1及strict134完成后该训练路线关闭，不得fresh、resume cycle2或作为当前active config。
  部署严格不变：historical v6-fast macro400的600 tensors冻结，
  exact language加恰好一条action-hidden video经Balanced P256 key和single FP32 Program residual生成一套
  完整38-target rank16 LoRA；没有language bypass、few-shot、expert-bank deployment或第二套LoRA。
- 训练从`M_0=0, Lambda_0=I` fresh开始，不能继承RLS10。每个train task的一条video只生成一次LoRA，随后
  在四个persistent env lanes做K4 official random-reset batched rollout；成功与失败都保留真实executed prefix。
  `A_e=(4R_e-sum R)/3`，全成/全败task严格零credit且不拼replay；mixed task以episode等权、Nmc4 keyed CFM
  直接对完整LoRA求signed gradient，再VJP到Program cotangent。现有full48 exact reconciliation只接收正号
  `G`，其内部构造负梯度写入；negative 24 rows仍为exact-zero incremental motion且无额外policy forward。
- 当时runtime保持六rank×4 tasks、K4 policy batch4、BF16 policy、四persistent env lanes、deferred NCCL和
  CUDA-complete FileStore readiness后两次fixed gather。已移除旧ledger、single-lane collector、
  success-only collator、RLS-specific gate/deployment/nonpass owner、old/current第二forward、ratio、第二epoch、
  shared Adam和热路径hash；历史仍由Git、docs与formal artifacts保存。
- canonical owners为`reward/rollout.py`（K4 batched environment path）、`expert_manifold/v6_reward_credit.py`
  （唯一reward数学）、`v6_prior_training.py`（task-immediate graph→rollout→replay→cotangent→release）、
  `v6_prior_profile.py`（机制/profile门）、`pi05_eval/reward_credit_gate.py`（registered strict root与cycle2门）。
  `pi05_eval/preparation.py`是唯一prepare/staging发布owner；`writer_family_registry.py`和
  `pi05_eval_analysis_fixture.py`只是从超大owner拆出的单一登记/fixture，不是并行方法。
- CPU已覆盖16种K4 outcomes、LOO符号/零和、ASPO首epoch一阶等价、Nmc4 time/noise/physical-batch不变性、
  BF16 decoder→FP32 Program grad、homogeneous zero fast path、full48 transport、checkpoint/cursor、profile→formal
  fail-close及strict registered-root/six-arm trigger gate。后续formal cycle1与strict134已由本文件1.1封存。
- clean frozen`c4507e9f4872a88cccca37ca7956371bd8a18bd4`的首次profile已在`gpu02:0--5`自然exit0：root=
  `runs/outputs/pi05_v6_reward_credit_program_cotangent_profile_full24_k4_nmc4_r6_b2_20260810`。24 tasks、96
  rollouts、24 videos、11 mixed/13 homogeneous、60/36 success/failure、4452 replay chunks、22124 executed
  steps完整；wall=`554.268s`，peak allocated/reserved=`16,336,873,984/19,417,530,368B`，0 OOM/nonfinite/
  watchdog，profile不留checkpoint且六卡已释放。
- 旧profile按预注册gate正式`passed=false`，唯一false为`program_to_action`：fixed probes为
  `0/7/14/21`，只有ordinal0是mixed，7/14/21均homogeneous并按合同exact-zero cotangent。11个mixed的
  cotangent均finite/nonzero，13个homogeneous均zero且0 functional forward；full48 rank48、condition=
  `105.66`、negative/correct=`.017081`、closure=`0`、LoRA A/B response=`5.736e-6/6.001e-6`，aggregate action
  RMS=`.0006390`。因此该non-pass暴露的是probe集合与zero-credit合同冲突，不能写成Reward-Credit action链失败，
  也不能事后把旧artifact seal为通过。
- 修正后的profile v2删除固定ordinal，穷举运行时全部mixed tasks；每task使用K4 rollout的四条真实首query和
  原始首个policy-noise seed，before/after各一次batch4 policy forward，并从raw per-task rows核对mixed
  ordinals精确覆盖、K4数量、四suite覆盖及每task Program→LoRA A/B→BF16 action均非零。homogeneous不要求
  action change，仍保持exact-zero direct credit并报告shared-solve motion（旧profile为mixed的`1.182%`）。
- 吞吐按live直接证据上调为replay physical B8；它只改变keyed Nmc4 panel的物理切片，短batch自然一次forward，
  不改变objective。旧profile的`rollout_seconds+credit_seconds`同时冻结静态rank map
  `[[3,8,13,23],[5,9,16,22],[1,11,15,20],[0,7,14,18],[2,6,12,21],[4,10,17,19]]`：每rank仍一task/suite、
  mixed counts=`2/2/1/2/2/2`，旧B2 critical local wall约`547.9→410.1s`；RNG不含rank且gather仍按ordinal排序。
- clean frozen`e6024cf97200721b13834c6ad81de85ce6588ffb`的新root
  `runs/outputs/pi05_v6_reward_credit_program_cotangent_profile_full24_k4_nmc4_r6_b8_allmixedk4_20260810`
  已在`gpu02:0--5`自然exit0并正式`passed=true`。完成24 tasks/96 rollouts/24 videos、11 mixed/13 homogeneous、
  60/36 success/failure、4452 replay chunks；full48 rank48、condition=`105.65998`、negative/correct=
  `.01704835`、closure=`0`，LoRA A/B aggregate RMS=`5.611e-6/5.885e-6`。
- all-mixed raw rows覆盖ordinals`[0,2,3,4,5,8,9,10,12,15,18]`和四suite；11/11 task各有4 queries、
  2次batch4 before/after forward，LoRA A/B/action全部finite/nonzero，action RMS范围
  `.00106586--.00364465`。13个homogeneous task仍exact-zero cotangent与0 functional forward；shared motion
  仅为mixed的`1.1794%`。0 OOM/nonfinite/watchdog/old/negative forward，且无checkpoint或training metrics。
- raw `(replay_chunks,Nmc,functional_forwards)`逐task唯一反推出physical batch=`8`，总forward=`928`；旧B2同
  面板为3648。B8使mixed credit seconds sum降低`15.01%`，balanced map使本地critical proxy从旧
  `547.928s`降至`383.947s`；尽管新profile额外做11-task action panel，end-to-end仍从`554.268s`降至
  `507.305s`。peak allocated/reserved=`36,575,930,368/40,928,018,432B`，结束后六卡均0MiB/P8。
- e6024cf的start行仍打印旧硬编码`reward_replay_microbatch_size=2`，但实际调用链、config/run contract和raw
  forward唯一解均为B8；后继seal commit已把start/run-contract字段改为读取sealed config。profile→formal
  matcher同时新增optimization/runtime、唯一B8公式、completion、单次fresh invocation、A40 topology及无
  checkpoint核对，故无需为展示字段重复GPU profile。
- 正式evaluator入口已经fail-close：Reward-Credit sealed deployment只能用selected B8；evaluation contract
  必须是validation 8×50、without-replacement、同一clean pushed/frozen training/evaluation commit和预注册
  checkpoint/root，historical checkpoint不再绕过cycle登记。correct400先于controls；macro1首次`>=144`开放
  五臂，macro2仅在首次跨144或自身`>=151`时再开放，避免无信息重复；这一触发与cycle2 support门独立，故
  `>=144`但support失败仍先完成同checkpoint因果面板再停止。prepare用NFS同目录原子`mkdir`跨进程锁，在
  私有staging完成全部校验后以一次目录rename发布；失败时canonical root保持不存在，正式Reward的额外parser
  条件不能绕过预注册六臂。加载`.env.local`后的当时全仓回归为`358 passed in 23.72s`，architecture guard
  为0 hard violation且无parallel family。
- 当时的下一动作formal cycle0→1与strict correct400现已完成并由本文件1.1裁决；旧profile与新discarded
  profile的state/weight（均未保留）永久不进入formal，Reward cycle2不再授权。

### 1.3 Previous completed experiment and architecture decision（2026-08-10）

- 第39节RLS已从clean frozen`25bbd52c16cc0f0fd48f478f0fa8b554fcb28dc6`完成formal fresh0→10和
  唯一预注册的macro10 strict correct400。formal natural exit0、10 rows、macro10 checkpoint完整；step
  sum/mean=`199.425195/19.942519s`、input wait=`.278241s`，peak allocated/reserved=
  `43,247,554,048/46,919,581,696B`，0 OOM/nonfinite/negative policy forward。六张`gpu02:0--5`结束后释放。
- strict root=
  `runs/outputs/pi05_v6_exact_anchored_reconciliation_correct400_noreplacement_seed7_method_macro0010_25bbd52_20260810`；
  72/72 shards、400 rows、18/18 workers return0，correct=`140/400`、breadth6、per-task=
  `2/3/47/35/0/34/19/0`、per-suite=`5/82/34/19`。相对预注册balanced macro0=`134`严格paired
  retained/gained/lost=`119/21/15`、net`+6`、churn36；`lost<=6`门失败。相对blind-v2 macro10同为140却
  gained/lost=`17/17`、churn34；RLS没有改善closed-loop旧成功保留。
- immutable transition=
  `runs/outputs/pi05_v6_exact_anchored_reconciliation_macro0010_historical_baseline_transition_866cca9_20260810/analysis.json`。
  correct80子集会给出`31 vs 26`、gained/lost=`5/0`，与full400的`21/15`方向相反；因此不续25、不补六臂、
  不用80-row screen选点、不扫RLS damping/step/window/forgetting。config/runtime现封为可验证的
  `retired_after_macro10_strict_closed_loop_nonpass`，fresh/restart/resume均fail closed。
- RLS内部current/blind从`.999980`降到`.230340`、negative/correct升到`.291493`，logged/final precision
  condition约`7510.5/8325.5`；formal热路径没有保存reference rows，`reference_correct_rows=0`，所以
  `reference_rows_improved_fraction=1`是空集合值。最早失效接口已转向offline source-action functional
  cotangent与真实closed-loop occupancy/reward credit错位。该结论当时触发了第39.5 Reward-Credit实现：保持
  one-shot部署图，只替换为train24 binary-reward on-policy Program cotangent；其后formal cycle1与strict134
  已完成并由本文件1.1裁决，不再是当前active实现。
- 第38节v2已从clean frozen`abd8e0826e52758eda53b1963f8b12db92bf3748`完整训练0→25。formal root=
  `runs/outputs/pi05_v6_balanced_causal_condition_residual_formal_r6_lb20_mb10_abd8e08_20260810`，25 rows、
  completion25，step sum/mean=`535.464796/21.418592s`，peak allocated/reserved=
  `43,247,029,760/46,917,484,544B`，feature rank全48，0 OOM/nonfinite/negative policy forward。
- v2 macro0/10/25 strict correct400=`134/140/139`、breadth均6，per-task按Spatial1/Spatial3/Object1/
  Object3/Goal3/Goal6/Long1/Long2=
  `0/5/48/34/0/35/11/1`、`1/2/48/31/0/38/20/0`、`2/4/48/30/0/38/17/0`。macro10 root的72/72
  shards、400 rows、18 workers均return0；macro25内部72/72与400 rows科学完成，但外层wrapper exit未观测，
  必须保持这一provenance，不得补写exit0。
- 严格配对0→10 gained/lost=`19/13`、0→25=`18/13`、10→25=`12/13`；macro0∪macro10=`153`。这证明
  blind-add短期可换入新成功，但single checkpoint仍低于历史143且10→25净退化，不能靠续训或checkpoint
  union达标。v2不续50、不补五臂/六臂、不扫lambda/seed/P。
- macro10相对macro0的effective-BA delta/base中位=`1.69498e-4`，stable rank中位约`1.000022`、top-1
  energy约`.999978`；LoRA健康形态几乎未动。同task 50-video raw correction consistency=
  `.141539--.142175`，等于随机正交基准`.141421`；fixed10 effective-BA pair cosine跨8 tasks在
  `[-.001371,.003280]`。最早失效接口因此是blind-add没有把连续video-keyed写入reconcile成可保留的共同
  function，而不是video key、frozen decoder或单纯全局scale。
- 第39节Exact Anchored Reconciliation当时作为唯一active implementation，部署端完全沿用v2：exact language +
  exactly one action-hidden video、Balanced P256 key、frozen v6 decoder、完整38-target rank16 LoRA；训练端
  维护FP64`Lambda_0=I_256`，对`F=[correct;negative]`和`E=[-G;0]`使用RLS，把每批目标锚在
  `F M_before+E`。checkpoint新增training-only precision与`assimilated_rows`，deployment仍只加载FP32 Program。
- RLS首macro在数学上精确退化为v2 blind solve，之后才产生保留效应；family fresh-incompatible，禁止加载
  v2 macro10/25。旧`f0c3f51`继续按原合同保持16/18 non-pass；第39.4.1未改任何训练热路径。clean
  pushed/frozen`f28fc8b`随后完成独立fresh0→3，natural exit0、0 checkpoint/OOM/nonfinite/negative forward，
  raw artifact与completion均`passed=true`且17/17门通过。old drift/blind=`.248611/.213872`、old rows
  improved=`1/1`、current/blind=`.999980/.784334/.640650`；production三步=`19.9974/20.7508/19.5182s`，
  mean ratio=`.952297`，peak allocated/reserved=`43,261,790,208/46,919,581,696B`。config已登记
  `100452B` immutable evidence并formal-ready；profile state未保留。fresh启动前固定macro0与唯一macro10
  strict roots写入run contract；任何
  10→25续训都从immutable shards重聚合并核对checkpoint与paired identity，且必须由macro10 correct≥140、
  lost≤6、breadth≥6的严格结果授权。formal evaluator只接受predeclared macro10/25；这些profile与启动合同
  继续作为有效历史机制/provenance证据，但已由上面的full400否决，不再授权GPU动作。

### 1.4 Chronological evidence retained below

- **2026-08-10最新裁决**：第35节Tangent Tube已从clean frozen`b308941`完成formal fresh0→10和
  同一one-shot correct400。训练10 macros总step wall=`207.444s`、input wait=`.265s`、peak
  allocated/reserved=`43,316,440,064/47,112,519,680` bytes，0 OOM/nonfinite。macro10两臂
  relative-anchor tube中位=`.01390/.01408`，但directional ratio中位=`108.93/126.88`、两臂
  `0/24` tasks过`≤1`，completion error task median=`.25229`且`0/24` tasks过`.05`。
- strict correct400=`131`、correct80=`27`、breadth5、per-task=`0/3/46/31/0/40/11/0`、per-suite=
  `3/77/40/11`。相对同schedule macro0=`134`严格paired gained/lost=`16/19`、churn35、net`-3`、
  `p=.735879`；所以不续25、不补六臂、不扫weight/LR/WD。当前config/runtime已封为formal non-pass并
  fail-closed。该证据淘汰当时的tangent recipe/window，但completion从未成立，不能宣称expert-component
  假设已被干净证伪。
- **2026-08-10最新audit裁决**：第36节matched Expert-Flow Teacher Viability Audit已从clean frozen
  `e8e4728`自然exit0。formal root=`runs/outputs/pi05_v6_expert_flow_teacher_audit_r6_lb20_mb10_e8e4728_20260810`；
  480/480 queries、24 tasks、suite 6×4、reversed/shuffled/wrong 8/8/8、144 policy forwards、0 update/
  rollout/OOM/nonfinite。wall/input wait=`39.698/.684s`，peak allocated/reserved=
  `43,418,974,720/47,133,491,200` bytes，所选六张A40结束后自然释放。
- expert/macro0/tangent10 matched真实7维flow loss=`.098631330/.091801740/.091843160`。expert只在
  `2/24` tasks且`0/4` suite means同时优于两baseline，远未达到预注册`18/24+3/4`；四suite相对macro0
  都差约`4.89%--11.90%`，剔除最差task后仍差约`6.07%`。teacher-quality是方向性non-pass，不是边缘
  fail或单一outlier。
- CEFD gradient在compiler/factor相对existing span的residual=`.6864/.8387`，finite且非冗余；但
  “不同方向”不能把整体更差teacher变成有用teacher，且distillation loss最大的位置反而偏向direct expert
  较弱tasks。因此`authorize_cefd=false`：不做CEFD weight profile、训练或事后扫描其它expert steps。
  一次性audit config已formal non-pass并fail-closed，runtime按第36节触发退役。
- **2026-08-10第37节v1 profile正式non-pass**：clean frozen`6903ee6`在`gpu02:0--5`自然完成root=
  `runs/outputs/pi05_v6_condition_residual_mechanism_profile_macro49_r6_lb20_mb10_6903ee6_20260810`；没有
  checkpoint/retained weight，0 OOM/nonfinite/negative policy forward，六卡退出后均回到0MiB。
  13项机制门中10项通过：feature rank48、correct motion/cotangent=`.807966`且`24/24` task retention、
  application closure relative RMS=`0`、A/B response均非零、四suite fixed-action=`4/4`。因此显式kernel、
  frozen-v6 decoder和Program→完整LoRA→action传递成立，未发现gather/order/sign/solver工程bug。
- non-pass是同一个key-geometry根因：regularized Gram condition=`1315.33`，aggregate negative/correct=
  `.264351>.25`，task-local null=`15/24<18/24`。shuffled/reversed/wrong的paired feature cosine mean=
  `.98552/.95645/.90627`，null过门=`2/8,6/8,7/8`；全部9个失败row cosine均`>=.97099`。单位pair ridge
  leakage解析式对实测相关`.99021`，证明v1未平衡DC主导了顺序key。v1不训练、不扫lambda/seed/P/阈值。
- v1 production wall=`23.530704s`，相对sealed baseline ratio=`1.115458>1.10`，按原门保留non-pass；只
  超上限`.326083s`，而跨host input-wait差`.633711s`，所以不扩大解释为稳定计算退化，也不为它单独重跑。
- **第38节v2在当时成为active的Balanced DC--Causal Condition Key**：historical v6的600 tensors和
  `[256,320,256]` memory/full48/`.01` damping/step1/B20/B10+10全部不变，只把fixed key改为video-DC
  static与phase-centered sqrt-causal-prefix dynamic两个独立JL128 blocks，各自zero-L2后拼成P256。
  no-video仍精确zero；same-frame-set reverse/shuffle共享static但RHS为`g/0`，所以static不能单独拟合。
- **第38节v2 mechanism profile已正式13/13通过**：clean frozen`5d93434`在与sealed baseline相同的
  `gpu01:0,1,2|4,5,7`完成root=
  `runs/outputs/pi05_v6_balanced_causal_condition_residual_mechanism_profile_macro49_r6_lb20_mb10_5d93434_20260810`。
  rank48、condition=`106.114`、correct/cotangent=`.968254`、negative/correct=`.0218514`；24/24 correct
  retention且最小`.942261`，24/24 null且最大leakage`.048462`，A/B、4/4 fixed-action和closure全通过。
- shuffled/reversed/wrong cosine mean=`.479565/.013732/.507178`，各臂最大`.851083/.023307/.762135`；
  leakage mean=`.024184/.018664/.025999`且三类均8/8过门。production=`20.021842s`、对sealed baseline
  ratio=`.949122`，input wait=`.069295s`与baseline`.076318s`匹配；0 checkpoint/OOM/nonfinite/negative
  forward，selected GPUs结束后回到14MiB。
- 一次性teacher-audit/effective-objective/flow-teacher owners及tests已删除；checkpoint只拥有单个Program
  memory、cursor和六rank RNG，base600和fixed projection不被保存或覆盖。v8 residual deployment adapter、
  strict paired evaluator和analysis family已经联锁，错误Writer family不能借用本候选profile seal。
- 最后合同复核已封住artifact自报通过、formal状态空壳和任意checkpoint lineage三条缺口：profile从raw
  macro重算13项门并匹配完整scientific run；formal result必须绑定completion、50-row metrics和10/25/50
  manifests；deployment training commit必须属于active remote authority lineage。clean detached frozen
  authority ancestor现可直接用于v8 evaluator，不需制造第二主分支。
- v2聚焦`52 passed`、带LIBERO assets最终全仓`281 passed in 21.34s`；compileall、26份JSON、diff-check与
  architecture guard通过。profile artifact已从raw macro/run/completion重算并写入config；mechanism状态
  sealed。此处记录的是当时尚无v2训练成绩的阶段，已由1.1的formal/strict结果覆盖。
- 当前deployment verifier已恢复并收敛为一个双root owner：必须同时重读同commit的32-request batch8/16/32
  profile、validation8×state0 correct results和native LoRA cache manifest，核对单卡A40、selected batch、
  8 rows/entries、单次launcher、0 retry/runtime failure/forbidden reads及Writer release/source reuse。旧的profile-only
  evidence不能seal，formal runtime也同时要求该evaluation artifact，不再靠文档顺序防止误启动。
- 该GPU前修复的最终CPU门为全仓`283 passed in 26.10s`、compileall、Black、26份JSON、真实config load与
  diff-check；architecture guard相对`5d93434`为`+968/-318`且0 hard violation，原contract缩到1101行，
  没有parallel family或训练/推理热路径变化。
- clean frozen`2af82aa`已在实时空闲`gpu02:0`完成deployment双root。固定32-request/1093-frame panel的
  batch8/16/32吞吐=`.911238/.901898/.906482 LoRA/s`，repeat分别约`34.97/35.27`、`35.63/35.33`、
  `35.30/35.30s`；三者稳定且reserved约12.9GB（约12.0GiB），按最高实测吞吐选择batch8。
- validation8×state0 correct vertical root真实执行8 videos→8完整LoRAs→native cache→释放Writer→复用
  source policy→8条LIBERO闭环；`4/8` success、总wall=`336.056s`、rollout window=`199.799s`，8/8 rows、
  单次launcher、0 retry/runtime failure/forbidden reads。双root assembler已通过且GPU释放；`4/8`不是正式性能分数。
- v2 config当时为`active_deployment_sealed_formal_ready`。下一GPU动作当时是从新clean pushed/frozen seal评测
  zero-memory macro0 strict correct400；只有该真实基线封存后才fresh0→10并立即strict correct400。
- deployment写回由clean pushed`d228d0d`封存。其frozen worktree的首次CPU-only formal prepare在0 CUDA
  worker/0 scientific row时暴露一个工程合同错误：`runs`软链接经`.resolve()`落到canonical仓库后被旧
  evaluation verifier误判为越出worktree。`af7b101`只修artifact路径owner，允许词法
  `runs/outputs/...`且resolved target仍在canonical outputs root；nested symlink和manifest越界继续拒绝。
  全仓`285 passed in 21.38s`，clean frozen`af7b101`的同一formal prepare已exit0，精确登记8 tasks×50
  states、correct/without-replacement、method macro0 historical-v6 load-only + fresh elementwise-zero residual、
  18 rollout workers + 18 Writer generators和batch8；临时prepare root已清理。该prepare没有启动GPU或
  形成性能证据。
- **zero-memory macro0 strict400已正式封存**：clean frozen`6b5f7a6`在实时空闲`gpu02:0--5`自然exit0，
  root=`runs/outputs/pi05_v6_balanced_causal_condition_residual_correct400_noreplacement_seed7_method_macro0000_6b5f7a6_20260810`。
  strict correct=`134/400`、correct80=`26/80`、breadth6；per-task按Spatial/Object/Goal/Long=
  `0/5/48/34/0/35/11/1`，per-suite=`5/82/35/12`。wall/rollout window=`867.152/616.138s`，72/72 shards、
  400 rows、18 workers均attempt1/exit0。
- 400套native LoRA由18 generators以54 batches全部fresh生成，configured/max batch均8，0 reuse/redundant
  forward；Writer全释放、source policy全原进程复用且未reload。max per-generator allocated/reserved=
  `11,745,421,312/12,895,387,648B`，0 retry/OOM/nonfinite/forbidden reads；六卡结束后0MiB/P8。
- 与历史native v6 macro0 root的400-row严格identity检查中，state/language/env seed/policy-noise、teacher demo/
  order/selection seed和video mapping均0差异；success也逐行完全相同，gained/lost=`0/0`、共同成功/失败=
  `134/266`。新旧400 cache entry IDs相同，逐tensor CPU直比30,400 tensors、514,867,200 values全部
  bit-exact。唯一微差为一条共同成功episode终止step `106→107`，其余399 rows steps相同；不改变formal
  结论，也不为它降低吞吐。每task demos0--49各一次。由此v2 zero-memory部署图基线成立，不把`134`写成改进；
  随后的formal与strict证据已在1.1封存。
- 历史最好single checkpoint仍是v6-fast macro400的`143/400`。v6-prior已完成formal 0→50；同一
  schedule macro0/10/25/50 strict correct400=`134/127/105/123`，correct80=`26/26/24/27`。小panel在
  macro50看似上升而full400仍下降，进一步证明不能用screen替代正式裁决。
- 四点严格逐row配对分析：0→10 gained/lost=`19/26`；0→25=`19/48`、McNemar
  `p=.000522`；0→50=`20/31`。四点success union=`172`、intersection=`77`、逐task envelope=`147`，
  但union/envelope均不能作checkpoint融合。macro0仍是当前schedule单点winner。
- 内部根因不是“expert loss没动”：generated norm约`140.97→107.00`，cosine仅
  `.02194→.02630`，expert loss下降的约`94.2%`来自log-norm径向项；绝对expert投影系数
  `a=<G,E>/||E||²`反而`.736→.662`且23/24 tasks下降。held state0 macro50相对macro0的norm ratio/
  cosine/radial coefficient/orthogonal residual/base/delta/base均值=`.7180/.9755/.7007/.1551/.3373`。
  因而当前训练主要缩小已有v6 LoRA，而不是补足有用expert方向。
- 当前代码已完成对`0894856`后batch1/逐元素复现策略的撤回。A40已证明batch8只产生普通BF16
  batch-shape roundoff（max`.001953125`、mean约`4.70e-5`，direct-repeat为零）；owner明确要求不为这种
  微差牺牲吞吐。新路径在同一个固定request/总帧panel上从稳定且有显存余量的候选中选择实测LoRAs/s
  最高的batch，使用原生BF16/F32
  LoRA cache、零重复Writer forward、更少host sync和2-worker action prefetch。clean pushed
  `ded0c80`的A40 fixed-panel profile选择batch8；随后8-task纵向smoke完整通过并由retained artifacts
  组装evaluation seal。gradient/resume结构化artifact verifier与只读checkpoint comparator也已完成
  CPU验证。clean frozen`a17805c`随后两次启动六卡macro49 gradient profile：默认allocator和唯一一次
  `expandable_segments:True`重试都在第一个PI05 policy functional B20的Gemma MLP前向发生容量OOM；后者把
  reserved-unallocated从约`1.29GiB`降到约`157MiB`仍无法分配`606MiB`，因此碎片不是主因。两个root均只有
  run contract/invocation、没有gradient/completion，不能seal或resume，也没有产生方法性能结论。
- 当前修复保持logical B20、每task mean、train24×20=`480/480` unique queries和objective分布不变；
  完整有序logical-B20 panel keyed的physical slicing使用FP32 leaf-gradient加权累积，seed为轻量固定64-bit
  整数mix，不调用SHA/MD5。clean frozen `eddba96`的B16+4在六rank第一条functional attention一致OOM：
  allocated=`42.49GiB`、reserved-unallocated=`1.25GiB`、free=`235.31MiB`，尚需`254MiB`。所以当前只把
  physical microbatch改成balanced B10+10；这仍是A40容量实现变量，不是减小scientific batch。
  policy activation checkpointing目前不启用，因为
  OOM在frozen PI05 policy而现有checkpoint flag只覆盖Writer，启用policy重算会是更侵入且可能更慢的变量。
- clean frozen `9c814ff`的balanced B10+10已完整通过macro49：wall=`21.095s`、input wait=`.076s`
  （`.36%`）、peak allocated/reserved=`40.332/43.859GiB`、0 OOM/nonfinite；完整assembler通过。唯一权重为
  expert=`.008355172068998324`、ranking=`.28570466890490887`，两者在compiler各等于positive梯度的`.25`，
  在factor仅`.05254/.03993`。config已原样写入gradient evidence。
- strict后继`5fbcb27`已在`gpu02:0--5`完成fresh0→1+same-root exact-resume1→3和独立contiguous0→3；
  两root合同相同、各3 metrics、macro1/3 checkpoints和completion，0 OOM/nonfinite/clip。contiguous/
  resumed总step wall=`61.368/64.450s`，峰值allocated/reserved=`43.266/47.119GB`，steady-state input wait
  约`.0006s`。所有cursor/RNG/scheduler/AMP/frozen tensors精确相等，trainable Writer与Adam逐tensor科学
  门通过。原比较器误把近零Adam moments套入Writer aggregate relative门，已只修离线state-specific
  tolerance而未改训练或重跑GPU；retained artifacts重新assemble通过，当时的v2 profile/formal已正式sealed。
  该历史状态不解锁后续第38节或当前第39节config。

第33节whole-LoRA direction/norm与第34节Expert-Component Projection（ECP）均已由formal
closed-loop证据退役。ECP保留原v6架构和上游冻结边界，只把auxiliary换为
`a=<G,E>/(||E||²+epsilon)→1`与bounded negative ranking；因此这是对expert component假设的
干净单变量检验，不是新video encoder或LoRA topology实验。

clean pushed/frozen`450e688`的formal root=
`runs/outputs/pi05_v6_ecp_formal_r6_lb20_mb10_450e688_20260809`，fresh0→10后又按预注册门
exact-resume10→25；25 metrics、macro10/25 checkpoints、optimizer/scheduler/sampler/六rank RNG与
completion完整，0 OOM/nonfinite/clip。macro1→10的`a_correct=.736184→.828442`、expert
component=`3.06189→3.44394`、generated norm=`140.973→151.343`，24/24 tasks的`a`向1移动且
component上升。macro10 strict correct=`133/400`、breadth6，对同schedule macro0=`134`的精确
gained/lost=`22/23`、net=`-1`。这证明ECP修复了部分旧objective的径向伤害，但没有建立
held共同改善。

10→25的内部机制继续按目标运作：`a_correct=.828442→.884127`、component=
`3.44394→3.67225`、generated norm=`151.343→159.817`；23/24 tasks的`a`向1移动，24/24
component和norm上升。但expert-orthogonal norm约`151.303→159.774`，增量`8.471`远大于
component增量`.228`。macro25 strict correct反而降到`120/400`、breadth6、per-task=
`0/1/43/27/0/33/15/1`；相对macro0的严格配对gained/lost=`13/27`、net=`-14`、
McNemar `p=.038477`，suite net=`-4/-12/-2/+4`。macro10→25也是`18/31`、net=`-13`，
四个suite全部净下降。

裁决：ECP不续50/100、不扫权重、不为loser补六臂。直接增大expert component权重已被
证据禁止。第35节已完成与历史anchor/tangent/distillation去重，并在同一canonical vertical path
原位实现Condition-Local Dynamic Expert Tangent Tube：historical v6对correct和当前negative的同一
language/video/order输出分别作局部baseline，只惩罚student增量的expert-orthogonal分量。新v3 config、
training-only decoder anchor、trainable-only resume/deployment load、双臂metrics及独立评测family已通过
exact-D/gauge/gradient oracle、seal后formal-lineage guard和全仓`277 passed`、compileall与diff-check。
clean pushed/frozen
`2616773`随后在live空闲`gpu01:0,1,2|4,5,7`完成唯一六卡gradient/whole-macro profile：24 tasks、
480/480 unique queries、8/8/8 negatives、最长105帧，wall/input wait=`21.53076/.60603s`，peak
allocated/reserved=`43,353,948,672/47,112,519,680` bytes，0 OOM/nonfinite，六卡自然回到14MiB。
correct/negative的student与same-input anchor在24/24 tasks上完全一致，全部tube/delta指标exact zero；
projection/ranking唯一权重=`.00686480847114155/.010514453175708578`，assembler完整通过并写回config，
只解锁严格后继的resume profile。

strict后继clean pushed/frozen`c1bdcae`随后在live空闲`gpu01:0,1,2|4,5,7`完成resumed root的
fresh0→1与same-root exact-resume1→3，以及独立contiguous0→3。首段后的inter-phase selected-GPU
preflight发现设备不再满足expected-idle合同并安全停止；重新live检查六卡满足合同后分别启动剩余两段，
3个scientific invocations均exit0，没有重跑fresh或混用root。两轨各3 metrics、macro1/3 checkpoints和completion；
step wall=
`62.34061/61.95860s`、input wait=`.09366/.13220s`、macros/s=`.048123/.048419`，peak
allocated/reserved=`43,316,387,840/47,137,685,504` bytes，0 OOM/nonfinite，结束后六卡自然释放。

retained roots为：

```text
runs/outputs/pi05_v6_tangent_tube_profile_resume_r6_lb20_mb10_c1bdcae_20260809
runs/outputs/pi05_v6_tangent_tube_profile_contiguous_r6_lb20_mb10_c1bdcae_20260809
```

artifact assembler证明两份run contract完全相等，scientific metrics最大tolerance ratio=`.67790`；
macro1/3的cursor、checkpoint contract、6-rank RNG、scheduler/AMP语义相等，559个frozen tensors exact，
41个trainable Writer tensors的macro3 maxabs/relative-L2=`8.5067e-6/1.14428e-6`。82个Adam moments的
最低cosine与symmetric norm ratio均远高于`.999/.99`门。evidence已原样写入v3 config，profile和formal
同时置为`sealed_from_live_a40_resume_profile_evidence`，`runtime_for_mode(..., formal)`返回
`(50,(10,25,50))`；profile checkpoint永久不得进入formal。

以下是formal前profile阶段的历史判断，不再覆盖上面的最新裁决。三步pre-update轨迹中，macro1→2有21/24 tasks把`a_correct`推向1，
但macro1→3为0/24；macro3 correct/negative的orthogonal-relative-anchor task median约
`.03158/.03173`，仅`10/24`和`6/24`低于`.03`，orthogonal-to-direction中位约`60.98/61.2`。
这符合“quadratic tube在anchor处一阶梯度为0、首步可能先发生正交漂移”的结构风险，也说明不能把
resume seal写成mechanism pass。随后formal0→10与paired correct400已经按该门完成，结果如本节顶部；
当前recipe已停止，不能从这段历史表述恢复macro25。

## 2. EMBER problem and information wall

EMBER不是video imitation replay。它要求：

```text
exact task language + one action-hidden teacher video
                    │
                    ▼
               shared Writer
                    │  one pre-rollout forward
                    ▼
       one complete 38-target rank-16 LoRA
                    │
                    ▼
       frozen π0.5 source policy + live observation/state
                    │
                    ▼
             closed-loop task execution
```

Writer只能从语言与视频抽取任务的高层语义、对象关系、阶段和动作顺序；teacher action、proprio、reward、
terminal、task ID、filename、object pose和hidden normalization均在墙外。视频是唯一dynamic value；语言
可以定位“要解决什么”，不能单独形成LoRA旁路。video和functional action query必须错开episode，避免把
教学轨迹的低层运动与监督动作机械对齐。最终要求同一生成LoRA泛化到该任务不同初始化，而不是复现视频
轨迹。

one-shot是当前目标合同。few-shot的合理作用是从多个同任务视频中提取共同程序并消除单视频偶然细节，
但历史K4已证明“看多个视频”本身不解决condition-to-policy credit、共享梯度抵消或正确时序辨识。
因此只有当当前one-shot的同任务跨video方差被closed-loop证据定位为最早瓶颈后，才恢复固定K聚合；不能
通过平均多个无效LoRA或expert route伪造提升。

## 3. Current deployment architecture and closed historical training path

部署图恢复历史v6-fast完整video-to-LoRA生成器：

1. frozen π0.5 multimodal hidden对exact language和raw video形成task-grounded per-frame evidence；
2. Semantic Core聚合跨帧稳定语义；visual transition显式计算相邻变化；Causal Procedure保留动作阶段
   和顺序；
3. 320-slot compiler将Core/Procedure写入public LoRA topology；
4. 8个factor heads直接生成38个policy targets的完整rank-16 A/B；
5. LoRA只生成一次，Writer释放，原source policy原位加载该LoRA做闭环rollout。

当前唯一部署图仍由历史v6-fast macro400 checkpoint初始化：

`runs/outputs/pi05_as_writer_v6_decay400_taskcomplete_dev_r4_b20_seed7_s2400_4efa737_20260729/checkpoints/step_00000400`

其600个Writer tensors只作load-only初始化并全部冻结。Balanced P256 video key在fused Program后添加single
FP32 residual memory；当前compiler把36个q/v targets编译为pivot-preserving rank14 base加两个condition-local
rank2 physical zero-B slots，两个action targets仍走full-rank16 FP32 factors。部署checkpoint只读取该Program，
不读取training reward、observation/action replay、RLS precision或task experts。correct-video且Program为零时得到
rank14 v6 base而不是identity；no-video由显式fast path跳过pivot/solve/SVD并返回template-A/zero-B source identity。

第39.5已完成且关闭的Reward-Credit训练路径曾只在development-train信息域内增加真实闭环credit：

- 每task exact language加一条correct action-hidden video只生成一次LoRA，同一LoRA在四条persistent env lanes
  执行K4 official random-reset batched rollouts；
- success与failure都保留executed prefix，binary LOO只在mixed task产生正负credit；全成/全败精确zero且在
  replay拼接与functional policy forward前退出；
- mixed task以episode等权、Nmc4 keyed time/noise直接对完整LoRA求signed CFM gradient，再VJP到Program；
- 当时六rank各完成4 tasks后，经CUDA-complete rendezvous固定gather 24 cotangents与24+24 condition rows；现有
  full48 solve进行single identical manual Program/precision write，无optimizer/scheduler/scaler或memory all-reduce；
- reward/observation/action仅属于train24 credit，Writer输入和deployment checkpoint仍只有language+video；
  validation/test action/reward、teacher action、expert output与negative policy forward均为0。

当前没有active训练路径：只允许先完成新compiler的CPU/vertical/throughput门、rank14 macro0 strict400，再在
base过门后对既有cycle1 Program做一次load-only strict400。两个行为门裁决前不得fresh或resume任何Writer训练。

## 4. What task experts solve and do not solve

正式expert root：

`runs/outputs/pi05_task_expert_bank_formal_step1000_r6_81101fe_20260807`

24/24 train tasks均统一续到step2000，共120 checkpoints。统一step250/500/1000/1500/2000的
development-train direct-expert closed-loop为`432/557/624/638/658` of 1200；step2000有23/24 tasks非零、
task9仍为0，因此统一选择aggregate最强的step2000但不把它写成完美oracle，也不按task混点。

Experts解决：

- 在同一frozen source policy和public rank-16 topology上，给出“这个train task的什么参数更新确实能
  闭环工作”的policy-effective target；
- 提供正常task-local SFT LoRA的能量、rank坐标、跨target分配和有效方向参考；
- 避免meta-Writer只被高方差functional query推动、完全不知道有效参数流形在哪里。

Experts不解决：

- held task如何从video生成可迁移LoRA；
- 同一task的多个video之间应关注哪些共同程序；同task expert target对所有video恒定；
- correct、shuffled和reversed为什么不同，或时间顺序是否被真正理解；
- train24 expert是否可直接充当validation expert字典。soft/hard部署的`15/80`和`3/80`已明确否定这一点。

因此expert reconstruction loss和健康几何只能作训练辅助；真正裁决仍是paired closed-loop五臂/六臂。

## 5. Cumulative experiment lineage

下表是设计决策的连续证据链，不是候选排行榜。精确roots和完整分析仍保留在对应design、`findings.md`、
`progress.md`、Git和formal artifacts中。

| 方法/干预 | 最强strict证据 | 实际证明 | 失败或未证明 | 当前保留结论 |
| --- | ---: | --- | --- | --- |
| frozen source base | `48/400` | generic-source policy无目标适配也有非零能力 | 不读video，未检验video adaptation | 所有Writer共享的frozen起点 |
| mixed-task Source-SFT rank128 | `109/400` | direct target action可形成共享适配 | privileged且仍低于目标 | 参数预算/闭环参照，不是同信息墙baseline |
| v5.2 old recipe | `132/138/74/82/83` | 当前最强correct-vs-negative视频特异性 | absolute未过150 | 动态写出与顺序margin可实现，不能遗忘 |
| v5.2 task-complete | `120/109/107/111/124` | recipe会改变video传递 | absolute和margin均退化 | task-complete并非普遍改进 |
| v6 old recipe | `121/122/111/84/47` | 强时序差异可传到闭环 | absolute低、task旋转 | old recipe增强动态也增强不稳定 |
| v6-fast task-complete | `143/135/125/128/129` | 历史最佳eligible raw single-checkpoint absolute；Procedure差异可传到LoRA/action | 原objective续到450/500/550/600降为`131/130/132/126`；冻结上游迁移性仍是假设 | 当前representation prior与macro0基线，不续旧objective |
| CV-ADR RAW / GROUP4 | best`117` / `110` | 更大coherent更新不等于更好闭环 | 曲线漂移；video梯度主效应约`.1%` | query/flow variation主导；仅flow MC noise可约且不是主因，credit仍错位 |
| Target-Bound | best`120` | remove-A/D和memory reversal 8/8达门，动态路径工作 | correct漂移、共享factor共存失败 | 不再把首因归为video完全未使用 |
| Semantic Factor-Basis | best`127`，union`193` | common accumulation一度改善 | envelope gap66、严重换手 | shared credit仍未稳定积累 |
| variance-reduced estimator | best`126` | 精确Beta/antithetic MC只小幅改善gradient consistency | held loss改善但closed-loop退化 | flow Monte Carlo方差不是主因 |
| Semantic Direction Store | best`129` | 独立store改善早期acquisition | 同分checkpoint breadth不同；Program→factor压缩 | parameter coexistence是局部因素，不是根因 |
| Policy-Target-Owned Factor | best`99` | 解除38-target共享显著改善跨层异质性 | action效果和性能仍差 | 健康跨target几何不是充分条件 |
| Policy-Lane | best`70` | 形成约10条有效output lanes和SFT量级专门化 | video BA能量约`.02%` | 容量/形态健康不能替代动态credit |
| Policy-Wide Atom Dictionary | best`80` | 64 atoms广泛使用 | mixing/effective LoRA近rank1 | 不用增atom/rank/正交loss救活 |
| Factorized Condition-Kernel | best`49` | kernel full-rank、stable且跨video差异大 | LoRA约比direct SFT小200×，identity-like | 低增益decoder曾是明确瓶颈，但非唯一根因 |
| Few-Shot Invariant-Program K4 | best`108` | K4置换、same/LOO/wrong/order路径都工作 | full24 gradient retention约`.043` | few-shot可去偶然性，不能自动解决共享credit |
| K4 Policy-Layer Trace | best`99` | all-layer trace带来correct>wrong | reversal仍高、逐频单位化把低能量DCT高频放大约`140×` | 被放大的高频不能替代有物理意义的时序程序 |
| Energy-Preserving Trace | best`85` | 修复原始频率能量比例 | correct/wrong从`99/57`收缩到`85/80`；effective groups`13.97→10.63` | 能量保真本身不等于语义保真 |
| Evidence-Factorized Trace | best`84` | correct>wrong且trace→BA→action闭合 | shared Reader retention约`.05` | 参数隔离值得检验，但不是直接答案 |
| Sparse Semantic-Expert | best`78` | expert-local retention提高 | language route固定owners，wrong/order更成功 | language-only ownership不够；video须参与credit |
| Grounded-Video Expert | best`88` | video route、Reader、BA、action和rank均material | correct无margin、task轮换 | video sensitivity与parameter isolation仍不充分 |
| K4 Phase-Aligned v6 | best`108`; reversed`121` | video未被忽略 | 近rank1、高能量、程序retention约`.04` | phase alignment不能独立解决语义/credit |
| AS125 + semantic progress RL | `97→104→102` | failure轨迹可获得非零semantic credit | breadth下降、继续训练换手 | reward信号存在，但共享更新不稳定 |
| Program-Credit RL | `106` | lockstep CRN和program gradient可达 | task cotangent几乎正交却被压成common update | shared condition map会吞掉task-specific credit |
| SFT-Anchored Tangent-Basis | `143→142` | 在强warm-start上小幅reward更新可运行 | gained/lost`20/21`，无净提升 | 不能把warm-start保持分数冒充生成器改进 |
| task experts step2000 | train`658/1200` | aggregate最高且`23/24` tasks非零的privileged train target | task9仍为0、存在state turnover；不证明video或held泛化 | 保留为监督流形，不作部署字典 |
| addressless Expert-Manifold | `48/400` | raw-expert reconstruction能训练出norm约`4.55` | 与source exact同分、paired`5/5`；topology identity在decoder后坍缩，nearest expert cosine约`.008` | 无显式topology address的decoder已证伪 |
| topology-address binding | `75/400` | 静态chunk/rank坐标可乘性调制video dynamic value并进入闭环 | 输出仍高度task-common，held绝对性能低 | 地址辨识修复有效；不能单独调address解决迁移 |
| Causal Barycentric | `63/400` | temporal coefficients和raw-factor组合可运行 | `k≠j` cross terms使raw A/B组合不保持effective update；未单独隔离held support | policy-effective compiler必须先于组合几何 |
| policy-effective soft / hard bank | `15/80` / `3/80` | hard compiler近精确复现所选expert | 当前causal reader + 24个step2000 experts的soft/hard held support均失败 | 关闭当前24-expert online部署字典，不外推所有未来流形方法 |
| v6-prior whole-LoRA objective | `134→127→105→123` | 冻结上游、只训写出端可高吞吐稳定运行；晚段可部分回升/breadth7 | 整体方向+norm吸引主要径向收缩，macro0仍最佳，绝对expert投影下降 | 退役该objective，不外推v6表示无效 |
| v6 Expert-Component Projection | `134→133→120` | `a_correct`与component按构造上升，修复旧径向收缩 | 正交漂移继续增大，macro25 paired net`-14`、`p=.038477` | 退役；不续、不扫权重 |
| Condition-Local Tangent Tube | `134→131` | relative-anchor tube中位`.01390/.01408`，证明局部半径约束工作且吞吐可接受 | direction ratio=`108.93/126.88`、completion`0/24`、breadth`6→5`；只压小更新而未旋进expert方向 | 已退役；不续25、不扫权重、不补六臂 |
| Expert-Flow Teacher Audit | 无rollout | gradient residual`.6864/.8387`，teacher方向非冗余 | expert flow loss只在`2/24` tasks、`0/4` suites优于两baseline | CEFD否决；一次性runtime已删除 |
| Frozen-v6 Counterfactual-Null Program Residual v1 | 无rollout | correct retention `.807966`、A/B/action/closure成立 | DC key导致condition`1315.33`、null仅15/24，吞吐门亦non-pass | v1退役；不训练、不调lambda/seed/P |
| Balanced DC--Causal Program Residual v2 | macro0/10/25=`134/140/139`、breadth6，m0∪m10=`153` | 13/13机制门、24/24 null、A/B/action/吞吐和部署全通过；短窗net+6 | 10→25 gained/lost=`12/13`；50-video correction近随机正交，旧能力不保留 | blind-add已退役，不续50/不补五臂 |
| Exact Anchored Reconciliation v3 | `134→140` | CPU oracle/profile成立；full400 gained21 | lost15、相对v2没有改善retention、correct80误导 | 已退役；不续25、不补六臂、不扫RLS超参 |
| Reward-Credit Program Cotangent | cycle1 strict`134`、相对macro0`14/14`换手 | Program/video与continuous tangent结构健康 | native BF16 q/v factor写出丢失tiny motion | 不续cycle2；Program仅作rank14+2 load-only evidence |
| Q/V Rank-Reserved Native Reward | generation-only门通过，尚无行为分数 | pivot rank14 base error`.000752`、rank2 capture`.999709`、dynamic cosine`.997525` | base-drop仍约Reward的`1727x`，0 action forward/rollout | 先新macro0 strict400，过门后cycle1 load-only strict400 |

任何需要精确数字的决策必须回到对应design/artifact，而不是从本表反推未列指标。

## 6. Stable cross-experiment cognition

1. **视频被使用不等于视频被正确使用。** 多条路线都证明wrong/shuffle/reverse能显著改变hidden、BA和
   action，但correct仍可能更差。下一分析必须问改变是否沿policy-effective方向，而非只看差异大小。
2. **LoRA健康度是约束，不是目标。** 低能量、过度rank1和高列相似度曾解释部分失败；但形成SFT量级
   能量、多个lanes、跨target异质或正确expert cosine也未自动提高closed loop。不得单指标优化。
3. **functional surrogate长期错位。** held functional loss下降、gradient更稳定或full-rank kernel均曾与
   closed-loop退化共存；关键checkpoint必须及时rollout，不能用loss挑点。
4. **task drift不是一种表象对应一个单因。** query/flow variation（其中只有flow MC noise可约）、full24正交抵消、shared parameter
   coexistence、Program→factor压缩和condition-map common update都被逐步检验；其中每个只解释局部。
5. **正常时序必须有因果意义。** shuffled/reversed真正破坏frame展示顺序；模型不能依靠原时间戳恢复。
   correct必须同时接近有效policy update并超过negative，不能仅把negative推向坏LoRA。
6. **架构与recipe耦合。** v5.2/v6交叉结果证明不能按某一architecture aggregate整体判死，也不能直接
   恢复old recipe；需要对比最早传递接口和任务换手。
7. **当前假设是局部且可证伪的。** Reward-Credit已经证明correct Program、analytic FactorHead tangent与
   continuous effective BA保留task-common和same-video结构，首个明确断点是q/v非零BF16 factors无法表达
   sub-ULP motion。因此当前只改变q/v physical slot compiler，不换video encoder、Procedure、Reward objective、
   Program或public rank预算。若rank14 base保留而rank14+2仍不提高closed-loop，才上移到Reward归因或更早的
   Balanced key/Procedure；不能因一个分数对整条路线180度转向。

## 7. Engineering provenance and current implementation boundary

第39.5 Reward-Credit的工程实现与formal artifact均作为历史provenance保留：

- 当时唯一CLI是`scripts/train_v6_prior_writer.py`，mode为`mechanism-profile/formal`；一次性
  `teacher-audit`、flow teacher和旧effective objective owner/tests已经删除，没有第二runner或部署路径；
- `reward/rollout.py`唯一拥有K4 batched env path，`v6_reward_credit.py`唯一拥有LOO/Nmc4/Program VJP数学，
  `v6_prior_training.py`拥有task-immediate graph/replay/cotangent与两次gather，`v6_prior_profile.py`只拥有
  一次性verification和gates；`reward_credit_gate.py`拥有registered strict root与cycle2 decision；
- 600-tensor historical v6 strict-load后全部冻结，fixed projection nonpersistent。Program memory是该路线唯一
  deployment mutable state；training checkpoint另存FP64 precision、assimilated rows、cursor和六rank RNG，
  不存在optimizer/scheduler/scaler或memory all-reduce；
- 当时config/runtime/contract/checkpoint/adapter/evaluator/analysis状态机联锁；fresh必须等于当时remote authority，
  exact-resume保持原frozen commit且要求authority ancestry；错误family和stale artifact fail closed；profile只
  允许fresh0→1且不留state，formal只允许fresh0→1或通过raw400 support gate后的exact-resume1→2；
- retained config为`configs/pi05_v6_reward_credit_program_cotangent_v1.json`；B8/all-mixed profile、cycle1
  checkpoint与strict134现均已封存，cycle2关闭。旧c4507e9 root是immutable non-pass，e6024cf profile是
  immutable pass且不留state；该config不再是当前训练入口。

当前Q/V Rank-Reserved Native Reward Compiler尚待在canonical owner中实现。它只从历史v6 base与一个小型
derived manifest读取既有84MB Program tensor，不加载Reward optimizer/RNG/precision，不复制Program，也不
启动训练。实现后必须先过CPU合同、同panel B8/16/32吞吐和三臂vertical smoke，再进入两个有序strict400门。

以下是仍被当前候选继承或用作比较的historical throughput/runtime provenance：

- evaluator取消historical smoke中的8次冗余direct Writer forward和`1e-5`逐tensor门；batch默认8并要求
  profile至少实测`8/16/32`。三者处理同一32-request longest-first panel和同一总帧数，只改变实际
  forward分批，最终从稳定且有显存余量的候选中取LoRAs/s最高值；
- 76-tensor LoRA保持template原生dtype：72个BF16、4个F32，单entry tensor bytes从强制FP32的
  `5,148,672`降到`2,641,920`；batch GPU→CPU staging只同步一次；
- 旧source-action functional objective的logical B20/B10+10只作容量参照；historical Reward rollout是K4 policy
  batch4、四persistent lanes，mixed replay为Nmc4 physical B8。它由当时B2 profile的`19.42GB reserved`
  headroom上调；短batch自然一次forward，不做隐藏fallback，也不为低位数值缩K/Nmc/dtype/并行度；
- PI05 formal functional路径不再调用只供日志使用的`.cpu().numpy().tolist()`/`.item()`；loss-only实现与
  原forward的loss及LoRA leaf gradients由固定noise/time测试验证一致，通用details接口保持不变；
- correct effective alignment只计算一次，task metrics和gradient norms合并成少量host transfer；
- action DataLoader默认2 workers、spawn、persistent workers和prefetch2；确定性sampler的serial、
  prefetched和prefix+resume rows已精确一致；
- Writer offsets、frame ordinal/order、language span/condition ownership的重复CUDA→host门已合并或
  hoist到CPU，vectorized token packing不再逐row产生动态CUDA selection；必要的D2H handoff和宏步wall
  synchronization保留；
- 新`profile-writer-generation`在同一loaded source policy/Writer上做真实video→LoRA→native D2H sweep，
  记录fixed-panel actual forward batches、repeat wall、longest video、peak allocated/reserved和headroom；
  launcher及独立单卡worker均在模型load前live检查空闲NVIDIA A40，worker还核对clean pushed checkout；
  普通evaluator在spawn前拒绝忙卡、非A40、重复或越界device；可用数量只受所选单节点config topology与live
  ownership约束，没有额外6卡cap；
- evaluation seal只能由profile root与vertical smoke root的retained artifacts组装，校验三候选完整request/
  sampled-frame panel严格相同、warmup/repeats、最长视频、selected throughput、native cache、release/reuse
  和单次成功launcher；
- 当前状态图为`canonical compiler → CPU/throughput/vertical seal → rank14 macro0 strict400 → conditional
  rank14+2 cycle1 load-only strict400 → only-if >=144 controls/future design`；task experts不进入当前部署；
- clean frozen `ded0c80`在live空闲`gpu02:0`完成32-request、1093 sampled-frame fixed panel。
  batch8/16/32吞吐分别为`.911427/.905107/.906432 LoRA/s`，三者均稳定且峰值reserved约
  `12.82--12.85GB`；batch8按封存规则实测最快。大batch没有吞吐收益，剩余显存本身不是选慢配置的理由；
- 同提交fresh vertical root完成8 videos→8 LoRAs/cache→8 rollouts，单次attempt、`0` retry/failure/
  OOM/nonfinite/forbidden reads。Writer生成`10.597s`，peak allocated/reserved=
  `11,651,564,544/12,811,501,568` bytes；release后source policy原位复用且未reload。总wall=
  `325.540s`、rollout window=`196.816s`，进程结束后GPU回到0MiB；
- artifact assembler已从两个单卡retained roots重建evaluation evidence；gradient assembler又从macro49
  retained root重建权重和完整provenance；
  gradient assembler现会精确重建macro49的24-task teacher-demo/counterfactual schedule、480 unique
  queries、canonical config、clean pushed Git、frozen target manifest/HDF5 frame metadata与六卡拓扑；
  resume assembler会比较fresh/resume/
  contiguous的contract、cursor、6-rank RNG、600 Writer tensors、41 trainable tensors、Adam moments、
  scheduler/AMP和scientific tolerance，并要求gradient→profile的strict Git ancestry；
- frozen`a17805c`在当时live空闲`gpu01:0,1,2,4,5,7`的3+3 NUMA拓扑完成了两次有效工程诊断。默认allocator
  OOM时PyTorch allocated=`42.29GiB`、reserved-unallocated=`1.29GiB`、free=`395.31MiB`；唯一allocator
  retry为allocated=`43.43GiB`、reserved-unallocated约`157MiB`、free=`389.31MiB`，仍请求`606MiB`失败。
  这关闭“只调allocator即可保留physical B20”，但不关闭logical B20、当前Writer或任何科研假设。两次
  launcher退出后所选六卡均释放，未触碰当时由他人占用的GPU3/6。
- clean frozen`eddba96`在新一次live preflight后复用当时仍为空闲的同一3+3拓扑。首个非持久SSH后台
  launcher只写contract/invocation便exit0，没有start/gradient/completion，作为无效进程托管证据保留；
  改用tmux的fresh retry完整进入start，六rank均在第一条functional eager-attention申请`254MiB`时OOM，
  allocated=`42.49GiB`、reserved-unallocated=`1.25GiB`、free=`235.31MiB`。因此B16不存在可比较的吞吐点，
  当前canonical config已转为B10+10；上述root都不能seal、resume或选择auxiliary weight。
- clean frozen`9c814ff`随后用同一拓扑完成B10+10。24-task/480-query/105-frame panel、Git/config/
  HDF5、NUMA/NCCL、default allocator和single invocation全部经assembler闭合；input wait仅`.36%`，
  不再测试workers4。macro0 generated/expert effective norm mean=`140.52/4.182`、cosine=`.02196`，而
  reversed/shuffled margin仅`.000832/.000634`；这是当前要由受控expert/ranking更新纠正并由closed-loop
  证伪的核心机制矛盾，不是新增性能成绩。
- clean frozen`5fbcb27`的正式retry1比较root位于
  `runs/outputs/pi05_v6_prior_profile_resume_r6_lb20_mb10_5fbcb27_retry1_20260809`和
  `runs/outputs/pi05_v6_prior_profile_contiguous_r6_lb20_mb10_5fbcb27_retry1_20260809`。macro3 Writer
  maxabs/relative-L2=`4.6033e-5/1.06393e-5`，只占两步update L2的`1.023%`；Adam maxabs=`2.6865e-6`，
  `.007719` relative值来自近零moment分母。离线v2门改为Writer global relative L2`≤.002`，Adam每个
  moment的symmetric norm ratio`≥.99`且cosine`≥.999`；raw maxabs/relative-L2只诊断，并保留逐tensor
  `2e-4/2e-3`、全部语义exact和frozen exact门。config现为profile/formal
  `sealed_from_live_a40_resume_profile_evidence`；该seal只证明工程连续性，不是性能成绩。
- v2 retained assembler、config load和状态机均重新通过；聚焦checkpoint/contract为`11 passed`，加载
  `.env.local`后的全仓CPU回归为`247 passed`。未为本次seal启动任何额外GPU工作。

被撤回的失败root仍保留科学诊断：

`runs/outputs/pi05_v6_prior_warmstart_reproduction_smoke_validation8_correct_gpu02g0_30b2ccf_20260809`

其中`diagnostics/batch_equivalence.json`只证明BF16 batch-shape差异，不是performance evidence。

## 8. Key uncertainties and structural risks

- **frozen prior风险**：v6-fast提供143高增益起点，也可能把task-complete造成的弱时序margin一起冻结。当前
  residual只能在fused Program后修正，若真实order feature健康而正确时序知识根本没进入Program，显式kernel
  也无法创造上游语义。
- **feature sufficiency风险**：固定256维JL key只保留四个时序矩和frame-evidence/text-query innovation。
  它原理上区分content/order且无language-only value，但可能对细粒度阶段、接触事件或same-task视频变化
  不充分。首先看真实full48 rank、task-local retained/null及五臂，不能按漂亮Gram宣告成功。
- **稀疏binary credit风险**：K4只有mixed task产生相对credit；全成/全败严格零是无偏且防止臆造方向，但也
  无法直接扩展当前完全失败tasks的breadth。profile至少6 mixed只是可训练性门，不代表验证任务能共同提高。
- **失败prefix归因风险**：负advantage把失败episode全部executed prefixes作为anti-target；真正失败原因可能只
  位于后段，均匀episode/chunk credit会把责任扩散到此前正确动作。若Program/LoRA/action传递健康而strict下降，
  首先检查这一reward-credit assignment，不用scale、rank或延长训练掩盖。
- **shared solve漂移风险**：homogeneous task的直接cotangent精确为零，但full48共享condition solve仍可能因
  mixed rows使其correct condition发生Program motion。discarded profile额外报告zero-credit task motion比例；
  它是定位task drift的诊断，不另设事后门，最终只由paired gained/lost与breadth裁决。
- **counterfactual覆盖风险**：每个task/visit只取wrong/shuffled/reversed之一，full48只把当前24个negative
  motion压近零；它可能学习局部异常特征而非可迁移程序。必须由same-task-other、cross-suite wrong、
  shuffled、reversed和no-video同checkpoint六臂裁决。
- **验证指标边界**：all-mixed K4 fixed-action raw rows只证明每个有reward RHS的Program motion穿过frozen
  decoder和policy，不是性能、breadth或视频因果结果；verification forward不计入production throughput。最终目标仍只有strict
  closed-loop及其真实配对控制。
- 吞吐优化允许BF16低位、batch shape和parallel reduction普通差异，但不能改变K4/Nmc4 logical panel、信息墙、完整
  LoRA topology或paired evaluation；不得为逐元素复现牺牲batch/显存利用。

## 9. Evidence order and next action

证据关系固定为：

1. full-bank geometry、expert closed-loop和统一step2000续训已经完成；它们只证明train expert target的
   policy effectiveness，不进入当前cotangent/deployment，也不能证明held video causality。
2. whole-LoRA/ECP/Tangent/audit连续证据把首个失效接口定位到shared condition update transport，并否决
   expert-flow teacher；这是选择显式condition kernel的依据，不是新方法成绩。
3. v1/v2已经依次证明full48 algebra、zero identity、freeze、balanced key、LoRA/action传递、mechanism与
   deployment吞吐；v2 strict=`134/140/139`又证明blind-add短期换入能力但不能共同保留。它们是当前部署图的
   正证据和exact macro0对照，不再是待执行动作。
4. RLS fresh0→3内部17/17通过，但formal macro10 strict=`140`、相对macro0 lost15，否定offline feature-row
   anchoring足以保留held closed-loop能力；RLS全部runtime已fail closed，不能从其precision或Program续训。
5. Reward-Credit已经完成fresh cycle0→1与strict134；K4 binary LOO确实生成了Program，但旧native compiler
   使q/v tiny tangent落不到非零BF16 factors。故cycle2关闭，Program只保留为load-only causal probe。
6. full80已经证明pivot-preserving rank14+2可在generation层保存base和Reward结构；但base-drop仍约Reward的
   `1727x`，必须先以新macro0 closed-loop裁决，不能靠几何宣告成功。
7. 当前先做canonical compiler、单卡吞吐/vertical、新rank14 macro0 strict400；correct<130、breadth<6或
   相对旧134 lost>10即停止，不跑第二个400。
8. 只有macro0过门才跑cycle1 Program load-only strict400；仅当correct≥144、breadth≥6、lost≤6且
   gained>lost才算通过、补同checkpoint controls并授权后续训练design。140--143为诊断性non-pass，不授权
   新训练；严格>150必须完整六臂，再继续提高。

当前具体顺序以本文1.1和新design为准。实时比较`gpu01/gpu02`，独立evaluator选单节点当时所有真正空闲且
提高吞吐的A40；没有额外6-card cap，不等待凑卡。当前行为门无训练collective；未来fresh训练若获授权才
泛化world size，exact-resume锁定原NCCL/NUMA/rank topology。设备不空闲、拓扑不符或storage不足都fail close，
不触碰他人进程。

## 10. Canonical assets

- source policy：由当前config/CLI显式传入的frozen generic-source step1000 asset；不是
  `pi05_libero`，也不支持source-SFT exact resume。
- task experts：上述formal root的统一step2000 checkpoints。
- historical Writer prior：上述v6-fast macro400 checkpoint。
- retired config：`configs/pi05_v6_condition_local_tangent_tube_writer_v3.json`；Tangent和teacher audit均
  formal non-pass/fail-closed。
- audit/tangent comparison assets只作retained provenance，不进入第37节runtime。
- retired RLS config：`configs/pi05_v6_exact_anchored_reconciliation_program_residual_v3.json`；由f28 profile、
  25bbd52 formal/strict与866cca9 transition共同封为closed-loop non-pass。
- retained Reward config：`configs/pi05_v6_reward_credit_program_cotangent_v1.json`；B8/all-mixed e6024cf
  profile、cycle1 checkpoint与strict134均已封存，cycle2关闭。当前native compiler config尚待canonical实现，
  不得用旧Reward config fresh/resume或冒充当前入口；旧c4507e9 profile保持immutable non-pass。
- training/evaluation entries：`scripts/train_v6_prior_writer.py`与`scripts/evaluate_pi05.py`。
- target split：`configs/libero_24_8_8_v1/`。
- current source policy、tokenizer、data、video和simulation asset的BCI绝对路径均由CLI或`.env.local`
  提供；历史A100路径只作provenance，不原位改写。

旧formal outputs、diagnostics和设计文档保留。旧活动worktree、临时cache或重复本地branch只有在确认无进程、
无未合并唯一改动且远端/Git/artifact已保存证据后才清理。

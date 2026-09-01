# EMBER progress

更新时间：2026-09-02。

## 当前快照

- canonical集成目标为`main`。锁定科学提交为`a185fe223d1ef77635d83696c3e164a48520edbf`；第八次全局专家原文已逐字归档并在
  `3101232204265f379ad2282ecf9a1a9ee30bad8c`推送。
- owner已在2026-09-02正式采纳专家主选A并建立active goal：以PNBTT替换已停止的Program--bank接口，持续推进E0--E4和最终
  matched whole-Writer joint，直至满足正式闭环合同或出现真实阻塞。
- 当前active design为`docs/program_conditioned_native_bank_tangent_transport_design.md`；旧
  `summary -> family-scalar gate -> shared event-additive anchor`不再是active实现。
- PNBTT实现与E0已在clean pushed `2664e0d3705da3cdfb4bde2e7633317e0b102b4a`完成。首个single-key-chart E1在
  macro70/110相邻checkpoint均为`non_pass`。step110 task1 correct fit0/fit1/held为`.641984/.660311/.622909`、wrong为
  `.122637/.186146`；task93 correct为`.713247/.737497/.685649`、wrong为`.006121/.269427`。all-pairs、near-bound与
  信息墙通过，主要缺口是correct/held和`.50` margin；70到110的改善仅`.013--.037`，已形成稳定裁决。

## 最新科学结论

### 仍成立的正证据

- frozen source validation8为`48/400`，validation8 task-local rank16 oracle为`250/400`。
- held5 source/carrier/independent successful members为`21/43/113`；mobile-rank4解析容量覆盖held5五个task。
- G1 action-in native-block free-code strict250为`114/250`，breadth5/5、Goal2、Long1，正式通过。
- G2 boundary-anchored Natural Program的held full相对endpoints改善`22.2047%`，probe`38/40`、median active events`4`，
  same-task/K1/K4均通过。
- P0/P1、R5等正控证明真实native bank、current-bank operator和task-local功能方向具有容量；它们不证明shared mapping。

### 当前G3停止点

- Program-through-bank topology-matched free-summary S0双task正式通过：task1 correct/held约`.974--.989`、wrong约`-.565`；
  task93 correct/held约`.917--.947`、wrong约`-.342--.394`。
- fresh real Program-through-bank S1正式non-pass：task1 correct fit0/fit1/held为`.826825/.855228/.797545`，task93为
  `.776511/.792673/.719798`；wrong、margin、all-pairs、信息墙、Action Meta 0和唯一rank16均通过。按预注册条件没有运行shared S2。
- §7.1 bank-conditioned-primal恢复correct，但wrong specificity不足：原双tasktask1 wrong为`.428/.477`，task93为`.627/.654`。
- calibrated Q_free把wrong从`.815/.832`降到`.526/.534`，同时把correct降到`.808/.826/.795`，确认capacity--specificity权衡。
- base-LR A_free虽然233个anchors全部更新，但RMS仅`.0094`、约为candidate的`3.7%`，因此只淘汰under-travel版本。
- 最终calibrated A_free把free-anchor RMS提高到`.17664`，已与candidate anchor`.188--.192`同量级。task93 correct
  fit0/fit1/held为`.853296/.858892/.818467`，wrong为`.611592/.668511`；all-pairs通过，wrong和margin正式non-pass。
- 同checkpoint精确F=0后correct升至`.879708/.883433/.849663`、wrong升至`.750229/.756445`。F确实更强抑制wrong，
  但也伤害correct；candidate delta的correct/wrong cosine约`.718--.772`，占主导的free delta约`.993--.995`。
- 最早缺口因此是高相似summary经family-scalar gate调制共享event-additive anchor时只能近同向移动correct/wrong，无法把bank内容差异
  放大为所需功能分离。停止边界只覆盖这一具体parameterization。

完整历史及每个旧架构的结果在`docs/research_history.md`；长期跨轮结论在`findings.md`；八份专家原文均位于`docs/`。

## 当前active路线

- PNBTT保留G2 Natural Program、真实38-target X/Y及四类output bank、frame quadrature、exact signed replay、small-core
  canonicalization与首版carrier12+residual4。
- Program只产生低维query；当前bank的真实candidate产生key并继续作为唯一native value。B0只做可微key-space whitening，B1在同一bank
  上执行一次联合measure的antithetic signed transport；没有base primal、bounded correction、family scalar gate或free anchor。
- 首个E1 single-key-chart已稳定non-pass；它只淘汰当前per-target/side线性key chart + free query + whitening signed transport函数类，
  不裁决冻结的Natural Program。当前严格按专家§5.10先检查`T=Cov(v,k)`功能梯度投影谱，决定是`m`截谱还是需要
  family-shared trunk + target-specific低秩key projection，完成针对性修订并fresh重跑E1；E1通过后才进入E2。
- 专家远程artifact缺口中大部分本地已存在；当前实质缺少G2逐condition Program tensors，因此E2从frozen G2 checkpoint按condition重算，
  不误用fixed-token S1语义或cache。

## 最新formal evidence

- PNBTT E1 free-query transport：
  `runs/outputs/pi05_ecp_pnbtt_e1_free_query_s110_2664e0d_gpu01p12_20260902/`；110步、macro70/110 Panel-B与
  `evaluations/qualification.json`均完成，最终为相邻一致`non_pass`。

- Program-through-bank S0：
  `runs/outputs/pi05_ecp_program_through_bank_bottleneck_s0_gate_s110_b11dc3e_gpu01p23_20260901/`
- Program-through-bank S1：
  `runs/outputs/pi05_ecp_program_through_bank_bottleneck_s1_gate_s110_9047230_gpu01p23_20260901/`
- bank-conditioned-primal双task：
  `runs/outputs/pi05_ecp_bank_conditioned_primal_gate_s110_eb9f295_gpu01p12_20260901/`
- calibrated Q_free：
  `runs/outputs/pi05_ecp_bank_conditioned_primal_qfree_calibrated_task93_s110_fdc669f_gpu01p0_20260901/`
- base-LR A_free：
  `runs/outputs/pi05_ecp_bank_conditioned_primal_afree_task93_s110_b0d81bb_gpu01p0_20260901/`
- calibrated A_free：
  `runs/outputs/pi05_ecp_bank_conditioned_primal_afree_calibrated_task93_s110_e02f4ca_gpu02p4_20260901/`
- A_free逐层与F=0因果审计：
  `runs/analysis/pi05_ecp_bank_conditioned_primal_afree_causal_audit_144d59b_gpu02p46_20260901/`

以上formal evidence、唯一checkpoints、raw rows、aggregate、run contracts与completion均保留；没有因交接清理删除。

## 仓库与workspace整理

- 交接前58个累积worktree已清理；当前只保留canonical main、唯一`codex/pnbtt`实现worktree和正在运行E1的
  detached formal worktree。formal结束并集成后再清理任务所有worktree。
- 删除8个local `codex/*` branch：已合并分支由`main`保存；两个未合并EBSRI S2草案因S1预注册non-pass而失去执行资格；历史
  `g3-vector-interaction@2295f48`仍由`origin/codex/g3-vector-interaction`保存。
- 两个旧dirty worktree分别是已被clean S0/S1链和后续G3历史取代的实现草案；确认无运行进程、无formal authority引用后随worktree清理，
  未提交内容不可恢复。
- `.codex/tmp`中约`5.1GB`旧smoke/profile/script/cross-language临时cache已删除；其中影响决策的结论均已进入`findings.md`或
  `docs/research_history.md`。`.codex/tmp`当前为空。
- 未删除或移动dataset、models、formal runs、checkpoints、raw rows、aggregate、source policy、task experts、condition caches或
  ownership不清资产。
- tracked科学代码、测试和历史configs暂不在专家裁决前退役，避免提前删除新路线可能需要审计或复用的实现；active计算面仍以当前main为唯一
  canonical source，旧结果不得因文件仍存在而恢复为路线。

## 当前执行状态

- 第八次专家原文、代码/config/authority冲突和formal evidence已完成逐项复核；未发现推翻主路线判断的结果错误。
- PNBTT canonical compiler已接通：Natural Program只供query，real bank同时供key/value；包含可微batched key whitening、joint-K等video
  质量、exact chunked antithetic signed replay、四类output scope、38-target rank4 materialization及唯一carrier12+residual4 rank16。
- E1 task-local free-query训练与Panel-B evaluator已接通；policy、Program、carrier/scale、native values与Action Meta均冻结，correct fit0/fit1和
  wrong fit0产生梯度，held/wrong fit1/Panel-B为零梯度。task8/94只提供unrelated Panel-A states，preservation用同一keyed flow
  time/noise比较generated与carrier真实action velocity；wrong-video仍有单侧carrier上界。run contract从真实policy/Program模块审计
  Action Meta，而非声明式写零。
- E0 synthetic hard checks通过：zero native value给出zero residual；candidate/video排列与chunked replay误差仅为FP32低位；K2 video质量各
  `.5`；bank swap改变方向；forward/gradient finite；真实policy消费唯一38-target rank16。
- 首个真实双卡profile因一次性保留38-target covariance/Cholesky autograd图在A40约44GiB OOM；按target即时链式回传后不改变梯度
  （synthetic leaf-gradient最大误差`0`）。接入真实`D_policy`后的最新两步profile在task1/task93 microbatch 8/4下稳定完成，分别为
  `25.000/24.665s`；rank0/1峰值allocated为`39.773/36.154GB`、reserved为`46.376/44.109GB`。step1 free-query梯度非零且shared key按
  非对称LoRA零初始化预期为0；step2 shared-key梯度为`.293542`，task1/93 paired policy distance为`.003844/.002297`，correct/wrong已分离。
- 上述profile只验证工程图与吞吐，不参与E1科学Gate。E1 macro70/110均完成五臂各16次Panel-B；两枚checkpoint的task gate均为
  `non_pass`，总体与逐task结论一致。step110相对step70的correct/held改善仅`.013--.037`；near-bound最大值从未超过`.022005`，
  因此当前失败不是softmax饱和、训练过短或Natural Program。下一动作是专家指定的`T=Cov(v,k)`功能梯度投影谱，不进入E2。
- `c992b3f0d1fc5954f55ad939368881aa7a78a52e`已删除430行仅绑定退役primal/gate/anchor拓扑的stale tests，保留active cache、
  set不变性、信息墙和member-effect合同；25项focused tests通过。该清理提交已fast-forward至`main`，不改变正在运行的
  detached scientific authority。
- `50f876cb0e5e2e3623a4b77e768d67658960fccc`修正detached formal评测把会正常前进的`origin/main` tip误当训练身份的问题；
  现在仍锁定实际commit、clean/detached拓扑与全部科学合同，只允许包含该commit的authority tip前进。26项focused tests通过。
- `HANDOFF.md`已消费并删除；长期信息全部由authority、active design、本文件与Git保存。

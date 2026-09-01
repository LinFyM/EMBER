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
- 专家§5.10要求的train-only tangent spectrum诊断已在`8306a4cb43ee612671955354fbe0c508de996344`完成：task1/93各16个
  Panel-A visits、correct fit0/fit1与wrong fit0三条gradient arms，未读取held或Panel-B。`m=128`对应每个operator的1024列，
  99%谱能量rank与末端10%能量均表明没有有效谱被key width截断；q/v各side的功能梯度保留率和跨family差异则支持专家规定的
  family-shared nonlinear trunk + target-specific低秩key projection。该结构修订的fresh E1已从clean detached `75db5f84`
  完成，macro70/110相邻一致`non_pass`。同一step110的v2 train-only spectrum也已完成：key width仍未截谱，family chart只分离了
  部分output-side correct/wrong几何，没有恢复q/v correct功能梯度可达性。当前不进入E2，也不再修改key chart；下一步只做专家允许的
  单次同构task-local full-rank16 oracle。该oracle实现与两步真实profile已在`57969a68`完成，当前准备clean detached formal。

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
- 首个E1 single-key-chart与family-key v2均稳定non-pass。v2 spectrum相对首版的q correct-preserve-wrong中位仅从input
  `.555`到`.566`，四类output约为`.174/.235/.220/.224`；v input从`.463`到`.476`，abs改善到`.643`，但adj/init/goal反而从
  `.808/.727/.734`降到`.769/.685/.693`。尾端10%谱能量仍近零；family chart主要把action-out adj/goal operator cosine从
  `.839/.748`降到`.712/.627`，与formal wrong改善一致，却未补足correct容量。因此不增加`m`或继续改chart；只用一次同构
  full-rank16 oracle判别carrier12/task4是否是剩余瓶颈，E1通过前不进入E2。
- 唯一full-rank16 oracle只比较rank分配端点：保持相同family-key PNBTT、free query、数据、loss、Gate和110步cadence，将
  `carrier12+task4`改为`carrier0+task16`；最终仍是单一38-target rank16，不形成rank28或第二adapter。task16冻结幅度先验由与`s_ref`
  一致的fit19、非held task-local rank16 Action Experts做exact small-core singular component RMS后task-equal median得到；不读取
  validation/test，也没有task/video lookup。
- 专家远程artifact缺口中大部分本地已存在；当前实质缺少G2逐condition Program tensors，因此E2从frozen G2 checkpoint按condition重算，
  不误用fixed-token S1语义或cache。

## 最新formal evidence

- PNBTT E1 free-query transport：
  `runs/outputs/pi05_ecp_pnbtt_e1_free_query_s110_2664e0d_gpu01p12_20260902/`；110步、macro70/110 Panel-B与
  `evaluations/qualification.json`均完成，最终为相邻一致`non_pass`。
- PNBTT E1 tangent spectrum：
  `runs/analysis/pi05_ecp_pnbtt_e1_tangent_spectrum_m128_step110_8306a4c_gpu01p12_20260902/`；task1/93共380个
  target-side spectra、16个Panel-A visits、三条gradient arms，`completion.json`完整，耗时`376.97s`。
- PNBTT family-key E1：
  `runs/outputs/pi05_ecp_pnbtt_e1_family_key_s110_02633a39_gpu01p12_20260902/`；训练authority固定为clean detached
  `75db5f84`，gpu01物理1/2双rank；110步、macro70/110五臂各16次Panel-B、两个checkpoint与
  `evaluations/qualification.json`完整，最终为相邻一致`non_pass`。
- PNBTT family-key tangent spectrum：
  `runs/analysis/pi05_ecp_pnbtt_e1_family_key_tangent_spectrum_m128_step110_75db5f84_gpu01p12_20260902/`；同一v2 macro110、
  task1/93各16个Panel-A visits、共380个target-side spectra，held/Panel-B/validation/test均未使用，`completion.json`完整，
  耗时`381.48s`。

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

- 交接前58个累积worktree已清理；首个E1、family-key E1及两次spectrum结束后对应detached evidence worktree均已删除。当前只保留
  canonical main与唯一`codex/pnbtt`实现worktree；v2训练/spectrum日志已移入各自formal root，`.codex/tmp`只保留当前可删除profile。
- 删除8个local `codex/*` branch：已合并分支由`main`保存；两个未合并EBSRI S2草案因S1预注册non-pass而失去执行资格；历史
  `g3-vector-interaction@2295f48`仍由`origin/codex/g3-vector-interaction`保存。
- 已删除完整并入`main`的远程`codex/g3-bank-set-relative-interaction`与`codex/g3-v4-evaluator-authority`；未合并的
  `origin/codex/g3-vector-interaction@2295f48`明确保留。
- 两个旧dirty worktree分别是已被clean S0/S1链和后续G3历史取代的实现草案；确认无运行进程、无formal authority引用后随worktree清理，
  未提交内容不可恢复。
- `.codex/tmp`中约`5.1GB`旧smoke/profile/script/cross-language临时cache已删除；其中影响决策的结论均已进入`findings.md`或
  `docs/research_history.md`。后续profile只作可删除工程证据，不与formal roots混存。
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
- `T=Cov(v,k)`诊断已自然完成：380个谱均来自train-side Panel-A，operator列数固定1024；除结构性零bank外，99%谱能量rank远低于
  1024且末端10%能量通常不超过`1e-6`量级，因此不增加`m`。q/v的功能梯度保留与correct/wrong operator重合暴露的是chart
  表达问题。family-shared nonlinear trunk + target-specific rank16 low-rank projection已经接入；`m=128`、rank4、query、loss、
  数据与Gate未改，35项PNBTT/shared-compiler/joint-primal focused tests通过。
- v2 implementation `02633a3964ecfd9d40f9827ba98456c87c07552b`已在clean pushed main完成双A40两步真实profile。step2
  family-key aggregate gradient为`.155687`，task1/93 free-query gradient为`13.945/9.212`，correct/wrong已分离；单步
  `25.266s`，两rank峰值allocated为`39.789/37.260GB`、reserved为`46.272/44.082GB`，无OOM或non-finite。
- `0f052cccc9ddb96fbcaaa2a036fdc61ee190d945`在不改变当前K1 E1的前提下补齐E2前置硬合同：每条视频在每个有效
  event/scope先归一为等质量再按`1/K`混合，并缓存授权内容排序键以稳定集合归约；K2每半event mass精确为`.5`，相同Program
  context下的native内容换序测试通过。`a2c3fe9e`同时把canonical runner默认配置从退役J3收敛到当前PNBTT v2；两提交均已
  fast-forward并推送至`main`，正在运行的E1仍固定在其祖先`75db5f84`。
- fresh E1 formal launch：从`02633a39`之后只增加本记录的clean pushed detached `75db5f84`运行；配置为
  `configs/pi05_ecp_pnbtt_e1_family_key_v2.json`，task1/93双rank DDP、110 optimizer steps、macro70/110 checkpoints，数据、
  Panel-A/B、loss与Gate完全复用首个E1。使用gpu01物理1/2，launch瞬间两卡均空闲，固定`NCCL_P2P_DISABLE=1`和NUMA0；输出
  `runs/outputs/pi05_ecp_pnbtt_e1_family_key_s110_02633a39_gpu01p12_20260902/`在launch时为fresh空目录。`/data1`当前user用量
  `772469868/1073741824 KiB`，参考上一E1的`257MB`，本轮含两个checkpoint峰值估计小于`1GB`。只允许同commit、同world-size2、
  同config exact resume；不覆盖无效root。科学裁决仍只认macro70/110五臂各16次Panel-B及相邻一致E1 Gate。
- family-key E1已经自然完成。macro70 task1 correct fit0/fit1/held为`.598648/.599961/.581859`、wrong为
  `.028320/.041884`；task93 correct为`.693744/.706930/.650097`、wrong为`.036270/.224452`。macro110 task1 correct为
  `.616630/.620958/.601512`、wrong为`.027332/.051458`；task93 correct为`.707775/.725727/.655429`、wrong为
  `.047247/.223365`。wrong、all-pairs与near-bound均通过，task1 margin也通过；两task correct/held和task93 margin稳定不足，
  70到110的correct/held改善只有`.0053--.0210`。因此family-key提高了specificity但没有恢复absolute capacity，不追加训练且不进入E2。
- v2 tangent spectrum也已自然完成：仍为380个train-side Panel-A spectra、每task 16 visits，耗时`381.48s`。相对首版，q/v input的
  correct-preserve-wrong中位只小幅变化为`.566/.476`；q四类output为`.174/.235/.220/.224`，v为
  `.643/.769/.685/.693`，没有形成correct容量所需的新可达方向。action-out adj/goal correct--wrong operator cosine降至
  `.712/.627`，解释了wrong specificity改善；但q/v input仍约`.958`，abs仍约`.927/.963`。全部非结构性operator的尾端谱能量仍远低于
  width上限，因此停止增加`m`或继续改key chart。该诊断本身不证明rank4 ceiling；只准入专家限定的一次同构full-rank16 oracle。
- full-rank16 oracle实现`57969a6895adfe2e336e5d83a30d1a80c12d47d2`保持一个参数化运行面：PNBTT residual rank由配置取4或16，
  rank4仍走原12+4拼接，唯一oracle直接物化task16；overcomplete action-out canonicalization以small-core SVD后零填充保持合法rank16
  shape。16项native/PNBTT与22项shared-compiler/functional focused tests通过。gpu01物理1/2两步真实profile自然完成，step1/2为
  `29.469/28.910s`；step2 task1/93 free-query梯度`9.883/11.488`、shared-key梯度`.205305`，全部finite。两rank峰值
  allocated为`39.841/38.584GB`、reserved为`45.722/44.080GB`，没有OOM；相对rank4约`25.3s`只增加约17%步时。
- 计划formal root为
  `runs/outputs/pi05_ecp_pnbtt_e1_fullrank16_oracle_s110_57969a68_gpu01p12_20260902/`。配置固定
  `configs/pi05_ecp_pnbtt_e1_fullrank16_oracle_v1.json`，task1/93双rank、110步、macro70/110、五臂各16次Panel-B；除rank分配和对应
  fit19冻结task16 scale prior外，E1数据、三项loss、LR、seed与Gate均不变。launch前重新live检查两节点、配额与目标root；只从最新
  clean pushed detached commit运行，不覆盖任何旧root。
- formal launch preflight已同时检查两节点：gpu01物理1/2均仅`15MiB`、util `0%`，物理3/4也空闲；gpu02物理5空闲、4/6可共驻，
  0--3与7为他人高负载任务。训练选择gpu01物理1/2与NUMA0，因为两task一rank一卡已是有效拓扑且复用该节点23GB condition cache；
  不跨节点拼卡、不干扰他人。`/data1` user blocks为`772567180/1073741824 KiB`，参考上一formal仅`94684 KiB`且本轮两个更大
  Writer checkpoints仍预估小于`1GB`；目标root确认不存在。固定`NCCL_P2P_DISABLE=1`、world-size2、相同commit exact resume，
  macro70出现后可在仍空闲的物理3/4并行Panel-B以隐藏评测时间。
- `c992b3f0d1fc5954f55ad939368881aa7a78a52e`已删除430行仅绑定退役primal/gate/anchor拓扑的stale tests，保留active cache、
  set不变性、信息墙和member-effect合同；25项focused tests通过。该清理提交已fast-forward至`main`，不改变正在运行的
  detached scientific authority。
- `50f876cb0e5e2e3623a4b77e768d67658960fccc`修正detached formal评测把会正常前进的`origin/main` tip误当训练身份的问题；
  现在仍锁定实际commit、clean/detached拓扑与全部科学合同，只允许包含该commit的authority tip前进。26项focused tests通过。
- `HANDOFF.md`已消费并删除；长期信息全部由authority、active design、本文件与Git保存。

# EMBER Persistent Plan

更新时间：2026-08-15。本文只记录长期Goal与当前迭代阶段；实时run状态见
`docs/active_session_handoff.md`，历史实验见`docs/research_history.md`。

## Goal

在owner已经对齐的科学目标、信息墙与工程原则下持续迭代EMBER，使同一shared Writer的单一checkpoint真正
利用正确教学视频获得跨初始化的高层任务知识，在strict paired closed-loop中稳定超过`150/400`并继续提高，
同时保持视频内容/顺序因果性、多task能力共存和实验效率。

Goal不绑定Dynamic-K、memory token数量、LoRA rank、mapper形式或optimizer；这些是当前可证伪方法变量。

## Success evidence

- 性能继续追求single-checkpoint strict paired correct `>150/400`；约145若要成立，必须由相邻checkpoints共同保持；
- 高breadth、低checkpoint churn、高相邻success-set重合、per-suite不过度集中；
- correct materially优于wrong/shuffled/reversed/no-video，same-task-other接近correct；
- Program、LoRA、effective BA与action证据能解释closed-loop，而非替代它；
- 方法不依赖teacher actions、task ID、held expert bank、挑video、多LoRA/checkpoint融合或language-only shortcut；
- 从fresh训练可复现，不长期依赖旧checkpoint换手。

## Completed iteration: Dynamic-K through Visual-Value

- [x] 综合owner昨晚要求、SHINE/Doc2LoRA类Hypernetwork原则和EMBER历史实验，形成Dynamic-K、真实backbone
  memory、per-video causal Program、cross-video set、policy group/rank M2P与完整rank8 LoRA数据流；
- [x] Dynamic-K backbone-memory macro50 strict=`100/400`，定位absolute Semantic Core被删除；
- [x] Query-only semantic-address恢复absolute address，但macro50 strict=`101/400`，终局non-pass；
- [x] validation8x4 five-arm逐接口probe把首个新增common-direction定位到旧family hidden/GELU；
- [x] 原位实现Direct-Family-B，删除hidden/GELU与inactive dynamic-A，保留全部上游和训练recipe；
- [x] 完整CPU`372 passed`、world5 full24 B20 profile、B8/B16/B32 deployment profile通过并锁B8；
- [x] 封存clean pushed `c5353f3`；
- [x] 记录owner中止的world6 macro16 run；无checkpoint，不resume、不冒充成绩；
- [x] 双节点live资源选择、quota检查和fresh world5 formal launch；
- [x] 完成当前fresh formal macro0->50与macro25/50 checkpoint；
- [x] 完成macro50 single-checkpoint K1 strict paired correct400：`102/400`、breadth5；
- [x] 完成absolute、per-task/per-suite、breadth、paired churn、能力集中和effective-BA接口分析；
- [x] 按`<120`或breadth<6预注册门终止；不resume到100、不做小扫、不补K1五臂controls；
- [x] 补齐canonical K2--K4 evaluator并用同一checkpoint完成K4 strict correct400：`98/400`、breadth5；
- [x] 完成nested K1→K4与effective-BA分析：set将same-task方差约降`6.3x`，但没有修正task mean或解锁task；
- [x] 固定下一fresh design：保留Dynamic-K/memory/set/M2P/rank8，单独恢复task-grounded视觉Value与有向transition；
- [x] 原位实现Task-Grounded Visual-Value并切换fresh-incompatible config/checkpoint/evaluator schema；
- [x] 完整CPU`378 passed`与机制门通过；首次profile `1.2603x`超门后只做数学等价效率修复；
- [x] matched world4 profile优化为`49.0775/46.2242=1.061727x`，K平衡、finite、无OOM并seal config；
- [x] 从clean pushed `caa2e30`创建detached frozen worktree，在gpu01物理`4,5,6`以world3 fresh启动正式0→50；
- [x] 完成macro25完整checkpoint，并以相同生成图完成K1 B8/B16/B32 live deployment profile，锁B8；
- [x] 从sealed clean commit fresh macro0→50，完整封存50条metrics、macro25/50 checkpoints与completion；
- [x] 同world3 exact-resume 50→100，并从clean frozen evaluator并行启动macro50 K1 strict paired correct400；
- [x] 完成macro50 K1 strict=`88/400`、breadth5及逐task/churn/effective-BA分析；
- [x] 定位当前最早断点：视觉Value确实改变并分化BA，但当前functional credit没有把它对准held on-policy方向；
- [x] 完成同world3 exact-resume 50→100与完整macro100 checkpoint，启动100→150及macro100 K1 strict400；
- [x] 完成macro100 K1 strict=`86/400`、breadth6以及macro50→100的严格churn与effective-BA方向分析；
- [x] 完成macro100→150 exact-resume与完整world3 macro150 checkpoint，并启动macro150 K1 strict400；
- [x] 完成fixed-A reachable-subspace分析，区分rank8容量与固定随机A可达行空间，并否定单一train24静态A作为held解；
- [x] 完成macro150 strict=`86/400`、100→150 churn48、三点union/intersection与effective-BA方向分析；
- [x] 完成old134同task 3-video-fit/1-video-held-out A行空间分析：overall保留`.9997255`，确认候选A应跨
  same-task视频稳定并由完整Program生成；
- [x] 完成150→200 exact-resume与macro200 strict=`96/400`；完整曲线`88/86/86/96`终局non-pass；
- [x] 完成macro150→200=`71/25/15` retained/gained/lost、相对old134=`68/28/66`与compiler138=`73/23/65`
  的严格裁决，确认晚期净增仍集中在少数task且没有共同积累；
- [x] 写Dynamic-K Task-Grounded Full-Factor单变量authority：输入、前端、Program、rank8与B20全部不变，只让
  同一Program生成`A_template + dynamic residual A`和dynamic B；
- [x] 原位替换唯一canonical mapper并切换fresh-incompatible config/checkpoint/eval schema；
- [x] 完成step0 identity、A/B梯度staging、顺序/set/动态K与freeze边界机制验证；全量CPU=`383 passed`。

## Completed iteration: Task-Grounded Full-Factor Rank-8

- [x] 双节点live检查后用gpu01物理1/4/5/6 world4完成full24 B20 profile：`47.4409s/macro`、
  peak reserved `45.563GB`、K1--K4各6、无OOM/nonfinite；
- [x] 将live profile seal进formal config；
- [x] 从clean pushed commit fresh训练macro0→50，checkpoint every25；
- [x] 用macro25 checkpoint完成K1部署定标：B8/B16/B32=`.979553/.975323/.972106 LoRA/s`，锁B8；
- [x] 完成macro50 K1 strict paired correct400：`91/400`、breadth5、per-task=`4/1/38/0/0/37/11/0`；
- [x] 相对matched fixed-A=`70/21/18` retained/gained/lost，仅净增3且低于fixed-A best96与Direct-B102；
- [x] 定位tiny-B/weak-near-orthogonal-BA重参数化，未达到125门，不resume、不做rank/scale/mapper小修；
- [x] 转以v6-fast为性能骨架写受控Dynamic Slot-Set bridge authority。

## Completed iteration: V6 Dynamic Slot-Set Bridge

- [x] 原位切换canonical Writer到冻结v6-fast + 197120参数Slot-Set层，并删除退役rank8 executable路径；
- [x] 完成全量CPU=`370 passed`与真实GPU机制门：K1逐tensor恒等、K轴换位仅BF16低位差异、video内倒序敏感、
  base无梯度；
- [x] 完成gpu01 world5 full24 B20真实profile：`30.7422s/macro`、peak reserved=`40.75GB`、K1--K4各6、
  最长323帧、0 OOM/nonfinite，并seal formal合同；
- [x] 从clean detached `26ebc43` fresh训练到macro25，25/25 metrics、completion和checkpoint完整；
- [x] 完成macro25 K4 B8/B16/B32 deployment profile，按最高吞吐锁B8；
- [x] 完成macro25 K4 strict paired correct400：`130/400`、breadth6、per-task=`1/2/48/32/0/34/13/0`；
- [x] 相对old134=`117/13/17` retained/gained/lost，net`-4`；未过134/breadth7门，终局停止；
- [x] 完成nested-dose分析：same-task方差降`9.26x`但task mean K1→K4 cosine`.999832`，定位post-compiler
  aggregation只能稳定nuisance、不能修正高层task mean。

## Completed iteration: V6 Shared-Core Procedure-Set Bridge

- [x] 写单变量authority：保留v6全部强路径，只把set从完整compiler后前移到shared Core与最终fusion之间；
- [x] 原位替换canonical实现/schema/config，删除旧post-compiler runtime path；
- [x] 完成64项定向CPU门：compiler阶段化严格等价、K1恒等、K-set换位不变、video内顺序敏感、梯度/freeze正确；
- [x] 完成全量CPU=`371 passed`；
- [x] 完成真实GPU机制门：K1 zero/nonzero set均逐tensor恒等，倒序敏感，base无梯度；
- [x] 完成gpu01 world6 full24 B20 profile：steady`24.249s`、reserved`40.758GB`、K各6、最长323帧；
- [x] macro1→2 q/k参数均非零更新并seal formal合同；profile checkpoint不进入formal；
- [x] clean detached `502618b`、gpu01 world6 fresh macro0→25：25/25 metrics、completion与完整checkpoint，
  总耗时`662.730s`，0 OOM/nonfinite；
- [x] macro25 K4 deployment profile：B8/B16/B32=`.223358/.223313/.223323 LoRA/s`，三者stable，
  按最高吞吐锁B8；
- [x] 从profile-sealed clean commit完成K4 strict paired correct400：`139/400`、breadth6、per-task=
  `1/4/46/34/0/36/18/0`；相对old134净`+5`、相对post-compiler130净`+9`；
- [x] 完成nested K1/K4、matched K4和trained-output归零分析：same-task方差降`9.69x`，但训练set只贡献
  `.000918` effective-BA改写，139主要来自无参数shared Core数据流；
- [x] breadth仍6且Goal3/Long2仍0，按门终局non-pass；不resume、不补controls、不扫K/LR/seed/temperature。

## Completed iteration: Semantic-Core Set Bridge

- [x] 写单变量authority：将同预算可学习集合共识从Procedure readout前移到语言对齐Semantic Core tokens；
- [x] 后端per-video Procedure只作无参数mean，保留v6底座、rank16、B20、动态K与K1严格恒等；
- [x] 原位替换唯一canonical实现/schema/config，完成step0/K1/置换/顺序/梯度门和full CPU=`372 passed`；
- [x] 完成真实GPU视频机制smoke与gpu01 world6 full24 B20 profile并seal：macro1/2=`27.214/24.277s`、
  reserved`40.758GB`、K各6、最长323帧、q/k第二步非零更新；
- [x] 从clean detached `884e55e`完成fresh macro0→25：25/25 metrics、完整world6 checkpoint、completion与
  exit0，loss first/last=`.101182/.095644`，0 OOM/nonfinite；
- [x] 完成macro25 K4 B8/B16/B32 deployment profile：`.223147/.223184/.223287 LoRA/s`，三者stable、
  0 OOM/nonfinite，按最高吞吐锁B32；
- [x] 从profile-sealed clean commit完成K4 strict paired correct400：`135/400`、breadth7、per-task=
  `1/2/46/30/0/35/20/1`；
- [x] 相对matched139=`120/15/19`净`-4`；按`<140`门终局，不resume、不扫参；
- [x] 完成trained-output归零与Core尺度分析：BA改写`.001763`，原始Core correction仅`1.8275e-5`，attention
  entropy/log4=`.999885`，把最早失败定位到centered Value相消而非compiler衰减。

## Completed iteration: Semantic-Core Common-Value Set Bridge

- [x] 写单变量authority：只把set Value从weighted centered residual改为weighted raw common Core；
- [x] 原位更新canonical schema/config/runtime；K1、step0、raw-mean Value、set、顺序、gradient门与full CPU=
  `374 passed`；
- [x] 修正显式K1旁路的零导数训练语义；world6 full24 B20 profile通过：macro1/2=`25.930/22.530s`、
  K各6、最长323帧无截断、0 OOM/nonfinite，gradient约比centered路径打开三阶，formal config已seal；
- [x] 从clean sealed commit完成fresh macro0→25：25/25 metrics、完整checkpoint/completion/exit0，总耗时
  `614.636s`，gradient全程`.00250--.00325`；
- [x] 实际K4 B32 longest-panel一次确认通过：`.225360 LoRA/s`、reserved`13.181GB`、0 OOM/nonfinite；
- [x] macro25 K4 strict paired400完整：`133/400`、breadth6、per-task=`2/3/48/31/0/35/14/0`；按双门终局；
- [x] 完成配对与first4机制分析：相对135=`118/15/17`、相对139=`119/14/20`；Core/BA改写已打开到
  `.065856/.053648`但Long能力换手，最早剩余接口定位为offline credit与held occupancy对齐；
- [x] 补做train-seen output-zero严格反事实：trained/zero=`63/59`，paired net`+4`但held net`-2`，把结论收敛为
  task-local credit存在而held可组合程序失败；
- [x] owner授权继续推进；冻结下一单变量authority，不从失败checkpoint续训。

## Completed iteration: Shared-Core Ordered-Procedure Common-Value Bridge

- [x] 写单变量authority：恢复matched139的shared-Core边界，只把Procedure-Set Value从centered residual改为raw
  common ordered Procedure；
- [x] 原位替换canonical schema/config/runtime并完成K1、step0、set、顺序、raw-Value与gradient CPU门；full CPU=
  `374 passed`；
- [x] 真实source-policy机制smoke与full24 B20 profile通过：macro1/2=`26.112/22.543s`，q/k展开、K各6、最长
  323帧、0 OOM/nonfinite，formal config已seal；
- [x] clean detached `d316623`、gpu01 world5 fresh macro0->25完整：25/25 metrics、checkpoint/completion/exit0，
  总elapsed=`745.622s`、K各6、最长359帧无截断、0 OOM/nonfinite；
- [x] first4 trained-output归零机制分析：Procedure correction=`.09601`，effective-BA mean/task-mean=
  `.01397/.01392`；credit已打开但compiler后的policy改写仍较小；
- [x] 完成macro25 live K4 deployment profile：B8/B16/B32=`.225016/.224729/.224704 LoRA/s`，三者stable、
  0 OOM/nonfinite，锁B8并seal evaluator；
- [x] 完成同一macro25 K4 strict paired400：`139/400`、breadth6、per-task=`1/2/46/32/0/36/22/0`；
- [x] 相对matched139严格配对=`120/19/19`，Long1净+4由Spatial/Object各净-2支付，双门终局non-pass；
- [x] 完成train-seen output-zero严格反事实：trained/zero=`64/64`、paired=`60/4/4`、net0，确认B20在train和
  held on-policy都只造成换手；
- [x] 将最早接口推进到训练credit本身，不续训、不改Value/rank/compiler、不扫参。

## Completed iteration: Ordered-Procedure On-Policy Preference Writer

- [x] 写单变量authority：保留macro25 K4完整架构与部署图，只把短AS cold start后的credit从B20 source-action
  functional loss改为train24真实success/failure executed-prefix preference；
- [x] 原位接通detached readout recompile、K4 replay、full24 task-equal reward gradient、fresh reward checkpoint与
  evaluator contract；
- [x] 完成CPU机制门与一个task真实GPU smoke：full CPU=`395 passed`；task4=`1/4` mixed，BA/action response=
  `.00018146/.00557193`，0 OOM/nonfinite；
- [x] 首个world5 formal先因合法长task超过600秒collective timeout失败，PG失败后最慢rank才报告OOM；收窄历史
  已知的不必要compiler graph生命周期并设reward专用30分钟timeout，不降B8、不改Nmc4/objective；
- [x] `fa53ce4`同一task4 B8复测exit0，科学量逐位不变，full CPU仍=`395 passed`，formal evidence重新seal；
- [x] 从clean frozen `9c26386`完成first full24 reward cycle：24 tasks/96 rollouts/64 successes、14 mixed、
  wall=`674.031s`，q/k/output与BA/action response非零，0 OOM/nonfinite/watchdog/forbidden read；
- [x] 完成cycle1 K4 strict paired400：`138/400`、breadth7、per-task=`2/5/46/33/0/36/15/1`；相对AS139=
  `120 retained / 18 gained / 19 lost`，按门终局，不运行cycle2。

## Completed iteration: V6 Actual-Delta Success-Support Projection

- [x] 结合owner最新澄清，明确memory token是重要候选机制而非强制形式；比较后确认V6当前最早断点是
  shared reward update的support preservation，不是LoRA生成健康度；
- [x] 写单变量authority：保留同一K4/rank16/V6 Writer、raw LOO reward proposal、optimizer与部署图，只约束
  task汇合后的actual AdamW parameter delta不增加train24成功executed-prefix task-mean loss；
- [x] 原位替换唯一reward runtime，完成success-support cotangent、small-dual projection、fresh schema与必要CPU门；
- [x] 完整CPU=`401 passed`；task4真实mixed smoke保持raw proposal逐位一致，support路径非零、identity fallback、
  BA/action与显存/吞吐门通过；all-success/all-failure语义由CPU直接覆盖，config已seal；
- [x] 首次world6 formal在metric/checkpoint前定位旧raw replay的all-success summary-only工程边界；收窄为仅
  all-failure summary-only并加入完整collation回归，失败root不resume；
- [x] 从clean pushed `ad2e1be`和fresh AS139完成world4 full24 cycle：22条support rows中raw违反6条，投影
  激活并得到0 violation，descent/energy保留`.963787/.980958`，wall=`1033.501s`、exit0；
- [x] 完成同schedule K4 strict paired400：`138/400`、breadth7、per-task=`3/2/45/30/0/36/21/1`；相对
  AS139=`116/22/23`、相对raw138=`117/21/21`，逐task/suite与first4 geometry均已封存；
- [x] projection active但仍`<144`、lost23且gained不超过lost，本V6 constraint方向终局；不做cycle2、
  LR/scale/rank/constraint小扫。下一轮先与owner讨论架构级接口，layer-aligned memory只是候选而非既定答案。

## Active iteration: V6 Layerwise Action-Probe Conditioned Procedure Reader

- [x] 对齐owner最新边界：memory token是可能改善layer-aligned LoRA生成的机制，不是强制形式；V6与memory不是
  二选一标签；
- [x] 对比V6、Dynamic-K memory、ADSP以及SHINE/Doc-to-LoRA，定位未被强V6检验的接口：18层真实Action-probe
  evidence尚未与对应policy layer/rank直接对齐；
- [x] 写单变量authority：冻结AS139全部强路径与rank16，从同一次joint forward旁读layer probes，经rank-query、
  video内causal delta形成zero-init Procedure-query conditioner；literal memory只作有证据触发的后继carrier；
- [x] 原位实现唯一canonical Writer、fresh schema/config与最小机制测试；同一次forward tap、step0 AS139、
  K-set换位不变、video倒序敏感、constant-frame zero、两步梯度展开及base/K-set冻结均通过；全量CPU=`402 passed`；
- [x] 吞吐前审计并移除逐288-slot重型temporal展开：改为每video一次shared causal controller，再轻量汇聚
  layer/rank deltas；科学变量与injection接口不变；
- [x] 完成真实one-forward/order/static carrier smoke和full24 longest-video吞吐profile：world3 B20 macro wall=
  `66.134/61.544s`、K各6、最长323帧完整、peak reserved=`41.385GB`；79帧joint forward=`4/4`，reverse
  query-delta/Program relative-L2=`2.0572/.40414`，constant query-delta max-abs=`3.38e-8`；
- [x] 从clean detached `515f91e` world6 fresh训练到macro25：25/25 metrics、完整checkpoint/completion/exit0，
  macro mean=`26.462s`、最长359帧完整、0 OOM/nonfinite；K4 generation B8/B16/B32=
  `.221225/.221402/.221500 LoRA/s`，锁B32；
- [x] 完成同一macro25 single-checkpoint K4 strict paired400：`143/400`、breadth7、per-task=
  `1/4/48/35/0/38/16/1`；相对AS139=`120 retained / 23 gained / 19 lost`、churn42、net`+4`；
- [x] 按预注册门终局non-pass：`<144`且lost19>10，不续到50、不补controls或小扫；全400 effective-BA只改
  `.002653` relative-L2，first4 same-task correction coherence median`.568`，最早缺口不是carrier或LoRA健康，
  而是Procedure-to-LoRA commitment与blind functional credit对held occupancy的方向选择。

## Continuous loop after this result

1. 从strict closed-loop和逐task成功集合确定最早失效接口；
2. 保留未被否定且已接通的机制，只选择一个主要因果变量；
3. 写简洁、可证伪design authority，说明为什么能改善高层视频知识、正确顺序、policy-effective写出和共同积累；
4. 原位修改唯一canonical Writer path，不保留退役parallel implementation；
5. 完成最小必要CPU/机制验证和真实吞吐profile；
6. fresh训练到预注册节点，尽快做single-checkpoint strict paired400；
7. 深入分析后进入下一轮，直到长期Goal真正达成。

## Active iteration: V6-LPCP Paired Causal Success Distillation

- [x] 纠正口径：LPCP143只追平历史v6-fast，对AS139净`+4`且`p=.643969`，不是突然变强；
- [x] 审计SHINE/Doc-to-LoRA原则、V6已有layer/module/rank slots，以及Target-Owned/Policy-Lane/Target-Spectral/
  Dynamic-K memory负结果；确认新增token、capacity或漂亮LoRA geometry不是当前最早变量；
- [x] 审计旧LOO reward、ADSP与candidate guards；确认它们没有检验“严格reference/candidate两臂中只蒸馏唯一
  成功trajectory”的continuous credit；
- [x] 写PCSD单变量authority：K4 V6-LPCP部署不变，只训练65,536参数query commitment map；
- [x] 原位替换退役reward runtime，完成shared conditioning cache、K2 paired replay与selected-success CFM；
- [x] 完成CPU/机制门：全量`387 passed`；architecture guard无hard violation；
- [x] review task-scoped diff并以`efc17be`commit/push clean authority；
- [x] 双节点live GPU/quota preflight后在gpu01物理`5/6/7`完成world3 full24 cycle1：24 tasks/48 pairs/
  96 rollouts，candidate/reference=`34/33`、gains=`5/4`、9 active tasks/3 suites，机制门通过；
- [ ] 机制过门后立即single-checkpoint K4 strict paired400并做逐task/union-retention分析；
- [ ] 首次约145且retention过门即补视频因果controls并运行相邻checkpoint；单点>150也不能跳过稳定性裁决。

## Non-negotiable boundaries

- exact language与正确action-hidden video共同构成任务知识；不能去掉任何一方或允许language独立写LoRA；
- 每条video内部保序，多video之间集合聚合；不平均frames/features/final LoRAs，不挑video；
- 当前先解决初始Writer性能，生成LoRA后的task-local RL留作后续独立实验；
- 一次尽量只改一个主要变量；局部建议不触发无证据的整套推翻；
- closed-loop absolute优先，内部健康度只解释；
- GPU至多6张但不要求6张，有多少真正合适就用多少；允许安全低util共驻；
- 吞吐优先，不加无意义防御性代码、重复forward、batch1、扩dtype、逐tensor scan或内容hash；
- 当前暂不使用subagents；
- 中止、历史或不完整run不得恢复成当前状态，exact-resume必须来自兼容完整checkpoint。

## Current blockers

无权限或资产阻塞。Ordered-Procedure AS139、raw reward138、ADSP138与V6-LPCP AS阶段均已按门终局。当前active
PCSD选择policy-aligned shared credit：利用AS139/LPCP union162的retrospective证据训练一个single checkpoint，
绝不在validation部署oracle selector。canonical实现与CPU门已完成，当前没有EMBER GPU进程；下一步按formal
preflight封存并发射。literal
memory保留为以后有触发证据的commitment/carrier变量，不与本轮credit混合。生成LoRA后的task-local RL仍是初始
Writer达到强zero-interaction起点之后的独立实验。

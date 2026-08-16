# EMBER Persistent Plan

更新时间：2026-08-17。本文只记录长期Goal与当前迭代阶段；实时run状态见
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

## Completed iteration: V6-LPCP Paired Causal Success Distillation

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
- [x] 完成cycle1 K4 strict paired400：`135/400`、breadth6、per-task=`0/4/48/32/0/35/15/1`；相对
  LPCP143=`121 retained / 14 gained / 22 lost`、churn36、net`-8`，相对AS139=`115/20/24`、churn44；
- [x] 完成全400 effective-BA与FP64跨video分析：PCSD/LPCP relative-L2 mean=`.0006834`，gained/lost幅度
  不可分；同task四个K4 video-set增量pairwise cosine平均`-.00187`、mean/sample energy=`.24860`；
- [x] 按四项门终局PCSD：correct、breadth、lost与net全部失败，不续cycle2、不补controls、不参数小扫；
- [x] 从FP64近正交证据选择下一单变量：只改变成功credit如何跨same-task不同video sets合并。

## Completed iteration: V6-LPCP Cross-Video Causal Success Distillation

- [x] 写CV-CSD可证伪authority：anchor K4只负责产生唯一成功trajectory；同一trajectory在四个互不重叠的
  same-task correct K4 conditions下分别形成exact functional gradient，task内等权汇合；
- [x] 原位替换PCSD runtime/config/schema，完成schedule、shared replay/time/noise、view permutation、duplicate-view
  degeneration、freeze与step0合同；
- [x] 完成CPU与task4真实active smoke：4 rollouts/4 credit views/16 unique demos，query/BA/action均非零，
  `145.526s`、peak reserved`40.752GB`、0 OOM/nonfinite/禁读；
- [x] 完成fresh full24 cycle1兼吞吐profile：24 tasks/48 pairs/96 rollouts，四view全梯度非零，wall=
  `863.432s`=`1.0307x` PCSD；三rank各8 tasks/3 active tasks，负载max/min=`1.0828x`；
- [x] 从同一LPCP macro25、fresh optimizer完成cycle1与K4 strict paired correct400：`134/400`、breadth7、
  per-task=`1/2/47/32/0/36/15/1`、per-suite=`3/79/36/16`；
- [x] 完成逐task/per-suite/breadth与严格success-set比较：相对LPCP=`122/12/21`、churn33、net`-9`；相对
  AS139=`121/13/18`；相对PCSD=`115/19/20`；四suite相对LPCP全部下降；
- [x] 完成全400 effective-BA及FP64跨video correction分析：CV-CSD/LPCP BA relative-L2 mean=
  `.000683702`，gained/lost不可分；四correct K4增量cosine=`.000205`、energy ratio=`.250155`；
- [x] cycle1五项门只有breadth通过，按预注册合同不做cycle2或same/wrong/shuffled/reversed/no-video；
- [x] 终局定位最早失效接口：cross-video exact credit已成立，但shared query-only commitment仍经video-specific
  Jacobian写成近正交局部BA方向；不靠view数、LR、rank、scale或seed小扫。

## Completed iteration: V6-LPCP Semantic Factor-Memory Commitment

- [x] 检查canonical V6/LPCP的Procedure query、K-set、fusion、320 policy slots、八factor families与38-target
  FactorHeads真实owner；确认q/v及主要action frozen output bases保留其允许的完整hidden span；
- [x] 对齐SHINE/Doc-to-LoRA的layer/module/rank结构化生成原则，并排除literal-memory旧低分路线、generic Program
  residual、Target-Owned/Policy-Lane、raw A/B与SFB原样恢复；
- [x] 写单变量authority：从同一cached condition计算LPCP/AS139 K-set Procedure差作为innovation memory，exact
  language只作四basis语义address，在factor-family hidden owner内zero-init residual，step0 exact LPCP143；
- [x] 预注册cycle1 strict门、cycle1/2约145稳定门、相邻churn/Jaccard和same/wrong/shuffled/reversed/no-video六臂；
- [x] 原位实现唯一canonical Writer/reward/evaluator schema，旧CV-CSD可执行状态只由Git/artifacts保存；
- [x] 完成聚焦CPU机制门与architecture guard：真实参数2,164,224、zero-init exact LPCP、memory/ownership/gradient/
  checkpoint合同通过，聚焦72 passed；
- [x] 完成真实GPU one-task机制/显存门：task4四view、8/8 maps、BA/action响应通过，wall为CV-CSD`.958x`；
- [x] 完成full24 cycle1兼吞吐profile：24 tasks/48 pairs/96 rollouts，8/8 family maps更新，cycle=
  `920.555s`=`1.0662x` CV-CSD，三rank任务=`8/9/7`、负载max/min=`1.0653x`；
- [x] 完成single-checkpoint strict paired400：`144/400`、breadth7、per-task=`1/3/47/36/0/38/18/1`；
  相对LPCP=`128/16/15`、churn31、Jaccard`.805031`，因lost>10终局不续cycle2或六臂；
- [x] 完成稳定FP64全400与跨video分析：effective-BA relative-L2 mean/median=`2.899e-7/1.066e-9`，
  q/v/action非零样本=`249/16/1`；first4 cosine=`-8.10e-6`、energy ratio=`.249995`；
- [x] 终局定位最早失效接口：continuous SFMC hidden residual在冻结W2与native public factor处被量化成稀疏
  q-family ULP crossing，cycle1 semantic router尚未形成；不把单点144误报成稳定145或视频因果资格。

## Completed iteration: V6-LPCP Gradient-Open Semantic Commitment

- [x] 根据SFMC终局定位单一变量：zero-init family maps在首个backward关闭semantic router，随后hidden residual
  又低于native factor局部量化边界；不续失败checkpoint、不改carrier/rank/LR/dtype/view数；
- [x] 写active authority：family maps作为zero-init delta，semantic query zero-init，冻结V6-W1提供balanced
  policy-aligned anchors；step0严格LPCP identity，但maps与query首步同时有梯度；
- [x] 原位替换唯一canonical commitment与fresh-incompatible schema/config，删除SFMC active runtime语义；
- [x] 完成step0 identity、首步maps/query梯度、q/v/action native写出、顺序/set/freeze与checkpoint机制门；
- [x] task4真实smoke通过：query delta较SFMC提高约6.8万倍，q/v/action BA均非零，wall=`.9501x`；
- [x] fresh world5 full24 cycle1完整exit0：24 tasks、10 active、四suite、cycle=`581.924s`，
  5 rank recorded wall max/min=`1.2121x`，完整checkpoint/completion；
- [x] 完成cycle1 K4 strict paired correct400：`141/400`、breadth7、per-task=
  `1/3/48/29/0/36/23/1`、per-suite=`4/77/36/24`；
- [x] 完成严格换手分析：相对LPCP=`128 retained / 13 gained / 15 lost`、churn28、Jaccard`.82051`，
  suite净变化=`-1/-6/-2/+7`；相对SFMC=`124/17/20`、churn37；
- [x] 完成稳定FP64全400与跨video分析：BA relative-L2 mean=`9.6632e-6`、为SFMC约`33.3x`，
  q/v/action非零样本=`400/399/368`，但first4 cosine=`.0001442`、energy ratio=`.250124`；
- [x] 按预注册门终局：correct、lost、net和suite四项失败，不续cycle2或六臂；因此没有same-task-other或
  negative/no-video资格，不能声明稳定145或视频特异性；
- [x] 最早失效接口后移到shared semantic address/cross-video reward credit经video-conditioned Jacobian写成
  近正交effective-BA方向；下一变量必须直接形成跨video可复现的causal task Program。

### Pre-registered fresh cycle1 launch contract

- frozen workspace=`/data1/user/ymdai/worktrees/EMBER-gradient-open-formal-eb543d3`，detached clean commit=
  `eb543d3fbc1148501fc67299c6af53047fd474f3`，且由`origin/codex/bci-continuation`包含；
- fresh output=`runs/outputs/pi05_v6_lpcp_gradient_open_semantic_commitment_formal_cycle0to1_r5_k4_views4_nmc4_b8_eb543d3_gpu01_20260815`；
  不覆盖、不resume任何SFMC或smoke root；
- gpu01 physical=`2/4/5/6/7`，world5 DDP，UUID尾分别=`ef13/f288/9218/34f/0915`；自动GPU-local
  NUMA，`NCCL_P2P_DISABLE=1`、deferred NCCL；选择world5是因为该节点只有这5张满足约40.762GB/rank峰值，
  其余三张分别高util或只余约4GB，不跨节点也不冒险共驻；
- command=`torchrun --standalone --nproc-per-node=5 scripts/train_reward_writer.py`，使用sealed gradient-open
  config、formal mode、canonical source step1000、LPCP macro25、canonical tokenizer/target HDF5，
  `--stop-after-cycle 1`；精确argv由run root的`invocations.jsonl`保存；
- scale=`24 tasks / 48 paired states / 96 rollouts`，每个active task四个互斥correct K4 credit views，B8 replay，
  完成条件是cycle1 metric、world5完整checkpoint与completion均存在且进程exit0；
- `/data1` quota snapshot=`537805228/1073741824 KiB`，matched formal root约79MB，本run加临时log估计
  `<0.2GiB`；后续strict400 matched root约1.1GB，均远低于独立余量；
- cycle1 checkpoint按预注册K4 correct400评测，并与LPCP143、SFMC144、v6-fast143及old134/compiler138/
  online128逐task比较；只有correct≥144、breadth≥7、相对LPCP lost≤10、gained>lost且至少3 suites不降才续cycle2。

## Completed iteration: V6-LPCP Causal Coefficient Transport

- [x] 从GOSC终局确定单一变量：梯度/native写出已经打开，但256维video memory仍作为condition-local Value
  direction，导致four-view LoRA增量约`0/.25`且suite换手；
- [x] 完成历史去重与design authority：保留LPCP、dynamic K、有序Procedure、four-view selected-success、
  V6 compiler和rank16，只把video memory改为320 slots各两个causal coefficients；exact language经冻结V6
  W1/GELU提供共享policy directions；
- [x] 原位替换canonical commitment，建立fresh-incompatible config/checkpoint；聚焦68项及完整CPU
  `397 passed`；
- [x] 完成task4 four-view真实机制/吞吐门：q/v/action和fixed-action非零；aggregate CCT-only BA
  corrected cosine/energy=`.575776/.681821`，倒序cosine=`.014842`，静态首帧coefficient norm=`2.74e-5x`，wall=
  `.9870x` GOSC；corrected action family `.081102/.310853`记录为formal风险；
- [x] gpu01物理`2/4/5/6/7` world5从sealed LPCP fresh完成full24 cycle1：24 tasks/48 pairs/96 rollouts，
  candidate/reference=`33/32`、9 active tasks覆盖四suite、cycle=`577.729s`，checkpoint/completion完整；
- [x] 完成K4 strict paired correct400：`142/400`、breadth6、per-task=`1/2/48/31/0/37/23/0`；相对
  LPCP143=`125 retained / 17 gained / 18 lost`、churn35、Jaccard`.78125`；
- [x] 修正task4机制分析的counterfactual标签：纯CCT train-seen four-view cosine/energy仍为
  `.575776/.681821`，但held first4约为`0/.25`；旧v1只保留provenance；
- [x] exact live loader排除checkpoint遗漏；train→held transported coefficient、pre-W2 hidden与BA L2分别缩小
  `1.63x/1.70x/249.92x`，最早断点定位为held residual经native BF16 factor/compiler未形成policy-effective BA；
- [x] breadth、retention与held共同方向门失败，按预注册合同终局；不做cycle2、六臂或axis/scale/rank/LR/seed小扫。

## Active iteration: V6-LPCP Native Probe-Value Commitment

- [x] 从CCT终局与完整历史选择最早接口：train/held pre-W2 hidden只差`1.70x`而BA差`249.92x`，tiny
  LPCP-AS139差分不是可靠held Value；
- [x] 比较literal-memory rank8、rank18 residual lane与direct native probe Value；选择只替换factor Value来源，
  避免同时改变carrier/decoder/rank；
- [x] 写单变量、可证伪design authority：保留LPCP143、rank16、FactorHeads、CCT transport与objective，复用
  Procedure-set attention聚合已有320-slot ordered probe deltas；
- [x] 原位实现fresh-incompatible NPVC：复用Procedure-set attention聚合native probe Value，移除tiny
  LPCP-AS139 Value差分；定向CPU`43 passed`、canonical assets下完整CPU`398 passed`；
- [x] 完成task4 selected-success与validation8×4 K4只读视频held gate：train=`.5929/.6792`，held8=
  `.4494/.5715`、6/8过门，held/train BA L2=`.7525x`；reverse/constant/wall均通过，task31/32登记风险；
- [x] fresh full24 cycle1完整结束：24 tasks/48 pairs/96 rollouts，candidate/reference=`33/32`、9 active tasks
  覆盖四suite，cycle=`584.053s`；checkpoint/completion、禁读、数值与显存合同完整；
- [x] K4 strict paired correct400=`136/400`、breadth6、per-task=`1/2/48/33/0/34/18/0`；相对LPCP143=
  `120 retained / 16 gained / 23 lost`、churn39、Jaccard`.754717`；correct、breadth与lost三门失败；
- [x] 完成all400 stable-FP64与post-train held分析：NPVC/LPCP BA relative-L2 mean=`.0004683`，held8
  four-view cosine/energy=`.40870/.54227`，但lost改写大于gained、retained-failure最大；full24后train task4
  从`.5929/.6792`漂到`.0569/.2951`；
- [x] 按预注册合同终局，不续cycle2、不做六臂或参数小扫；稳定约145资格未获得。完整终局artifact为
  `npvc_cycle1_terminal_analysis.json`、`npvc_cycle1_posttrain_mechanism_analysis.json`与
  `npvc_cycle1_runtime_balance.json`；
- [x] 审计reward-direction、support-preservation与Dynamic-K memory历史；选择PAFS-NV单变量：保留NPVC native
  Value和LPCP143，把common zero-init router换成fixed language pre-address + factor-owned zero-init selectors；
- [x] 原位实现fresh-incompatible PAFS-NV、16,384参数合同、per-task gradient coexistence evidence与CPU门；
  NPVC executable path已退役，完整CPU=`399 passed`，架构门无block；
- [x] 完成task4→validation8机制/吞吐裁决：smoke八family/q-v-action/wall均健康，task4 cosine/energy=
  `.4352/.5703`；但address effective rank=`2.1575<4`，held8=`.1681/.3729`且仅3/8过门，显著低于NPVC
  `.4494/.5715`与6/8，故终局不启动full24、不扫basis/scale/rank/LR/seed；
- [ ] 后继若达到约145，必须继续用相邻checkpoint低churn/high-overlap、same-task-other鲁棒及correct相对
  wrong/shuffled/reversed/no-video的明确paired优势认证。

## Completed iteration: V6-LPCP Shared Joint Native-Value Gate

- [x] 从NPVC与PAFS联合证据锁定单变量：保留NPVC shared native Value与冻结V6 geometry，去掉低有效维fixed
  address和八套factor selectors；
- [x] 写SJNV-Gate authority：以共享zero-init `2x256` joint `M elementwise_mul L` gate产生所有factor共用的
  direct/signed coefficients，总trainable=`512`；
- [x] 原位替换唯一commitment/runtime/schema并退役PAFS executable semantics；
- [x] 完成step0、video-required、K-set、shared gate gradient、checkpoint与CPU机制门：定向`76 passed`、完整
  `399 passed`、compileall通过，architecture guard无hard violation；
- [x] clean `913d3d3`完成matched task4 selected-success：cycle=`135.757s=.99775x` NPVC，train four-view
  cosine/energy=`.47227/.59781`，gate/q-v-action/BA/action response均非零；
- [x] validation8机制门终局：aggregate=`.20190/.39645`、仅2/8 tasks过门，action cosine=`.04299`，held/train
  BA L2=`.45251x`；未启动full24/strict；
- [x] stage localization确认gate/continuous hidden cosine约`.94`，但frozen W2后的raw factor cosine/energy=
  `.02135/.26592`、action factor cosine=`.00267`；最早断点为continuous hidden到native public A/B。

## Completed iteration: V6-LPCP Direct Joint Native-Factor Residual

- [x] 以SJNV最早断点选择单变量：保留LPCP/NPVC carrier、K-set、rank16与reward，只绕过冻结W1/W2；
- [x] 写DJNFR authority：`X=(M elementwise_mul RMSNorm(L))/sqrt(256)`经八个zero-init factor-shape heads直接
  写public A/B residual，总trainable=`1,654,784`，step0/no-video exact LPCP；
- [x] 原位替换canonical commitment/runtime/checkpoint/evaluator schema，定向CPU合同`60 passed`；
- [x] 完整CPU=`399 passed`、compileall通过，architecture guard无hard violation；
- [x] clean pushed `e756fa1`完成matched task4：八head/q-v-action均更新，cycle=`139.069s=1.02439x` SJNV；
- [x] validation8机制强通过：BA=`.77670/.76899`、8/8 tasks，raw factor=`.64461/.69769`、action cosine=
  `.55765`、held/train=`.46980x`、reverse=`1.22287`、constant=`1.76e-6`；
- [x] clean `49a4129` world4 full24 cycle1完成：24 tasks/96 rollouts、9 active tasks、八head均更新；
- [x] strict=`136/400`、breadth7，相对LPCP=`120 retained / 16 gained / 23 lost`，按门终局；
- [x] post-train held8 BA=`.79024/.78583`且8/8过门；生成端已通，最早失败接口后移到reward-useful direction。

## Completed iteration: V6-LPCP Direct-Factor Paired Common-State Preference

- [x] 写DF-PCSP authority：完整保留DJNFR输入、carrier、K-set、rank16与direct heads，只改变reward credit；
- [x] 原位把selected-success整轨迹正蒸馏替换为discordant pair同一初态winner-vs-loser首段flow preference；
- [x] 更新fresh-incompatible config/checkpoint/evaluator schema；两次clean task4工程smoke定位同seed hard reset以及
  flattened-state后再次hard reset都不能保持exact图像；现每lane只hard reset一次，每臂deterministic soft reset
  controller/observables后恢复同一qpos/qvel。修正后定向CPU=`26 passed`、完整CPU=`399 passed`、compileall与
  architecture hard gate通过；
- [x] exact task4/task7均为tie；task9/15/18分别产生1/2/1个discordant pairs，三者margin均下降且八head、
  q/v/action、reverse/static、wall均健康；
- [x] 完成三个有效anchor的train→held机制分析：task9 held/train仅`.105x`，task18 train跨video仅
  `.290/.428`，只有task15全门通过；按shared-method门终局，不full24、strict或cycle2；
- [x] 下一单变量设计须把final success credit放到成功occupancy上的多个exact shared query states，不能继续把
  数百步后的胜负全部归因给第一prefix。

## Completed iterations: successful-occupancy credit through MB-SOP

- [x] 写DF-SOCP authority：保留LPCP/DJNFR、rank16、K4、八direct heads与exact paired rollouts；
- [x] 原位把first-prefix batch替换为winner全部replan observations上的loser-policy counterfactual action batch；
- [x] 保持pair内replans、pairs、四views、tasks逐级等权，并移除DF-PCSP executable/schema语义；
- [x] 完成实现门：counterfactual actions按loser arm B8查询一次并跨四views复用；不等长trajectory等权、完整replay、
  microbatch语义与margin descent定向合同通过；全量CPU=`401 passed`、compileall与architecture hard gate通过；
- [x] 完成真实task9/15/18三anchor机制与吞吐门：outcomes/replay、跨video、八head、q/v/action与顺序链基本通过；
  但task9/15的B2/B1 stored action与B8 counterfactual差异超过名义策略contrast，三项wall=`3.083--5.335x`且
  task9 held/train仅`.118x`，按门终局，不full24/strict/cycle2；
- [x] 建立MB-SOP单变量authority：相同B8 observation/noise panel重查两臂，每条成功轨迹8个等进度strata各选
  matched action分歧最大的一项；保留LPCP/DJNFR、rank16、K4、Nmc4、四views与optimizer；
- [x] 原位实现MB-SOP action panel、fresh schemas与定向合同，预定task9/15/18 credit pairs=`8/16/8`；旧DF-SOCP
  executable family已移除；
- [x] 完成全量CPU=`402 passed`、compileall与architecture guard 0 hard violation；
- [x] 从clean `ad65347`完成固定task9/15/18 GPU机制与吞吐门：outcomes/counts、匹配action panel、吞吐、跨video
  BA均健康，但task15/18 post-margin上升且task9 held/train仅`.1096x`，按门终局，不full24/strict400；
- [x] 补四view flat-gradient几何：三anchor raw等权均值对4/4 views均为下降，最小view-to-mean cosine仍
  `.695/.629/.601`；将最早接口定位到AdamW实际delta而非视频梯度汇合。

## Active iteration: V6-LPCP Direct-Factor Adam-Radius Euclidean Commitment

- [x] 建立AR-EC authority：保持MB-SOP全部输入、credit、四view、direct heads与rank16，只改变parameter
  commitment；
- [x] 实现same-radius commitment：AdamW候选照常更新moments，最终delta严格为负raw shared gradient方向且全局
  L2半径逐步相等；
- [x] 增加已有四view flat gradients的Gram诊断和smoke-only四view同panel/noise post-margin，不增加训练forward；
- [x] 完成fresh schemas、定向CPU、全量CPU=`404 passed`、compileall与architecture guard 0 hard violation；
- [x] 冻结clean pushed `b578d56`并完成task9/15/18三anchor与完整post分析；raw/final方向门全过，但三任务各仅
  `1/4` views真实下降，故终局，不full24/strict400。

## Active iteration: V6-LPCP Direct-Factor All-View Monotone Backtracking Commitment

- [x] 建立AV-MBC authority：保留MB-SOP/AR-EC全部输入、credit、direction、optimizer state、八heads与rank16；
  唯一把full Adam radius改成first-all-view-monotone backtracking；
- [x] 原位实现fresh schema与`j=0..10`确定性回退；全量CPU=`404 passed`、compileall与architecture guard 0 hard；
- [x] 冻结clean pushed `aa819f2`并完成首次三anchor；三者完整exit0但暴露gradient/inference margin baseline与
  rollout/batch1 action baseline混比，故只作工程证据、不作科学裁决；
- [x] 修正为同一个inference evaluator的step0/candidate比较，并让fixed-action前后走同一batch1路径；全量CPU
  仍为`404 passed`、architecture guard 0 hard；
- [x] 修正版已冻结为clean pushed `202a64d`；
- [x] 完成task9/15/18三anchor及train/held/reverse/constant/吞吐分析：task18全门通过，task9 held两门失败，
  task15无accepted candidate；AV-MBC终局，不full24/strict；

## Terminal iteration: V6-LPCP Direct-Factor Maximum-Margin Common-Descent Commitment

- [x] 从AV-MBC的有效/near-identity/空集三种radius结果定位下一接口：raw equal mean虽4/4连续下降，但没有足够
  worst-view余量穿过不同task的native BF16有限步；
- [x] 建立MMCD authority：保留全部science graph、Adam upper radius与backtracking，只从已有四view gradients
  确定性求per-task maximum-margin direction，再保持task mean norm与跨task等权；
- [x] 原位替换AV-MBC executable/config/checkpoint/eval schema，实现4x4 active-set solver、几何证据与CPU测试；
  完整CPU=`405 passed`、compileall/diff check通过、architecture guard 0 hard；
- [x] clean pushed `fc3bdd7`后完成task9/15/18训练与完整机制分析；固定outcomes/counts、0禁读、geometry、native
  BA/action、held8、reverse/constant与wall均已保留；
- [x] 按预注册门终局：task9只失败held/train`.160558x`，task15仍exact no-op，task18全过，合计1/3；不实现
  distributed formal acceptance、不full24/strict/resume或参数小扫；
- [x] 从native finite-step metric/held amplitude这一最早接口建立下一份单变量authority后再改实现或启动GPU。

## Terminal iteration: V6-LPCP Direct-Factor Preconditioned All-View Backtracking Commitment

- [x] 核对历史边界：MB-SOP只测Adam full step，AV-MBC只backtrack raw equal-mean ray，实际Adam candidate ray加
  同路径all-view backtracking尚未检验；
- [x] 建立PAV-BC authority：保留全部science graph，唯一沿`d_adam`固定减半；不新增metric、forward、参数或
  部署分支；
- [x] 原位替换MMCD solver/schema/evidence；完整CPU=`404 passed`、compileall/diff check通过、architecture guard
  0 hard，active source净减少156行；
- [x] clean pushed `581140c`后完成task9/15/18训练、train/held/native/temporal/throughput分析与terminal artifact；
- [x] 按固定门终局：task9 j5但held/train`.109466x`，task15/18 exact no-op，0/3过门；不full24/strict/resume；
- [x] 终止parameter-space trust-ray路线；不混合raw/MMCD/Adam rays或扫scale；
- [x] 从LoRA输出/effective-BA参数化的native-safe线性Value接口建立下一份单变量authority。

## Terminal iteration: V6-LPCP Anchored Linear-B Native Value Commitment

- [x] 审计LPCP correct400真实A/B geometry及三anchor gradients；以全局固定规则选择B side，不按task/family/held
  切换：BA灵敏度比q/v/action-in/action-out=`1.049/1.411/2.594/8.258x`；
- [x] 建立ALB-NV authority：只删除A residual heads，固定`A=A0`、`B=B0+delta-B(X)`；LPCP、MB-SOP、PAV
  acceptance、rank16与信息墙不变；
- [x] 原位完成fresh schema、四B-head 860,160参数实现；定向CPU=`58 passed`、完整CPU=`404 passed`、compileall与
  architecture guard 0 hard；
- [x] clean pushed `0899166`后在gpu02物理1/2/3并行完成固定task9/15/18及held8、temporal、A不变、wall分析；
- [x] task9 exact no-op、task15 j5但held 5/8/raw-B失败、task18 j0全门通过，实际1/3；按门终局，不full24/strict、
  A-side completion、side mix或参数小扫；
- [x] 封存terminal adjudication并定位最早缺口：fixed-A去除了gauge/cross term且能救活部分PAV no-op，但小的共同
  B-only方向仍不能跨task稳定进入native finite-step/held coverage；
- [x] 建立下一份单变量authority：残差必须从native-zero坐标写入effective BA，同时不压缩、不重生成LPCP
  rank16 carrier；先用同一task9/15/18 panel快速否决。

## Terminal iteration: V6-LPCP Native-Zero Residual Bank Commitment

- [x] 固定严格反事实：连续新增BA仍为`delta-B A0`，public state唯一改为rank32
  `A=[A0;A0], B=[B0,delta-B]`，alpha/rank=1；
- [x] 原位实现rank32训练/机制路径、fresh schema/config与CPU合同；第一rank16 bank逐元素保留LPCP，second-B
  step0精确zero；定向CPU=`50 passed`、完整CPU=`405 passed`、architecture guard 0 hard；
- [x] clean `d4fc92e`并行完成固定task9/15/18、held8、temporal、稳定FP64 bank合同、wall与显存分析；task15/18
  纠正后过门，task9 outcome漂移且11步no-op；总wall/ALB=`1.16565x`；
- [x] 按2/3与吞吐门失败终局，不实现distributed evaluator/full24，不改变reserved bank宽度、side、ray、scale或
  dtype。

## Terminal iteration: V6-LPCP Native Endpoint Action-Preference Credit

- [x] 从NZRB与完整ray历史定位单一变量：accepted residual的native/held写出已通，但CFM surrogate不能保证真实
  10步action endpoint存在跨video finite step；不再换ray solver、bank、rank或scale；
- [x] 写NEAP-C authority：LPCP/NZRB/MB-SOP、四correct K4 views、optimizer和native backtracking全部保留，
  只把CFM preference换成同noise完整10步endpoint action距离偏好；
- [x] 原位替换canonical objective/schema，实现physical B8共享prefix、10步可微action endpoint与同metric
  inference acceptance；
- [x] 完成定向CPU=`50 passed`、完整CPU=`405 passed`、compileall与architecture guard 0 hard；active source
  `+616/-632`、净`-16`，没有新增平行实现；
- [x] 记录task9唯一launch contract：clean pushed code commit=`33f69fd4591ee8534004cbf569724575055c3e9c`，
  detached frozen worktree=`/data1/user/ymdai/worktrees/EMBER-neap-c-task9-33f69fd`；fresh root=
  `runs/outputs/pi05_v6_lpcp_native_endpoint_action_preference_task9_mechanism_b8_33f69fd_gpu02p1_20260816`；
  gpu02物理1=`GPU-19cb2d30-d2ca-77b5-54a5-6b23fd2eede4`，single process/physical B8。输入复用canonical
  source step1000、tokenizer、target HDF5与LIBERO assets；规模为task9两initial states、四arms、预期25 complete
  chunks、8 selected pairs和四个disjoint K4 credit views。`/data1` quota已用542,421,408 KiB/1 TiB，预计新增
  `<1 GiB`且不建cache；fresh-only，不resume/覆盖非空root。唯一首门是completion、outcome=`1/0`、25/8、0 OOM/
  nonfinite/禁读、physical B8、wall<=633.810s、4/4 endpoint descent、finite j<=10及全部held/temporal门；任一失败
  即终局，不跑task15/18；
  exact command=`CUDA_VISIBLE_DEVICES=1 PYTHONPATH=<frozen>/src MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
  NCCL_P2P_DISABLE=1 <canonical>/.venv/bin/python scripts/train_reward_writer.py --config
  configs/pi05_writer_v6_lpcp_native_endpoint_action_preference_v1.json --mode smoke --source-run
  <canonical>/runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722 --checkpoint
  <source-run>/checkpoints/step_00001000 --tokenizer-path <canonical>/models/tokenizers/openpi/paligemma_tokenizer.model
  --data-root <canonical>/data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a --output-dir <fresh-root>
  --smoke-task-id 9`；`<frozen>`、`<canonical>`、`<source-run>`和`<fresh-root>`分别严格展开为本条已记录路径；
- [x] clean `33f69fd`完成task9：outcome/count=`1/0,25/8`、physical B8、cycle=`97.107s`、reserved=
  `19.367GB`；gradient cosine/energy=`.846/.865`，j0一次接受，q/v/action、reverse/constant、rank-bank全通；
- [x] held8为8/8且BA/raw-B/action cosine=`.953/.955/.485`，但held/train BA L2=`.234<.30`；26/27门通过仍按
  authority终局，不跑task15/18、full24/strict或小扫；stage localization为probe/joint=`.671/.665x`到direct
  rows=`.223x`，最早缺口是one-task joint condition写入shared direct-B head的跨task幅度；
- [x] 从该终局证据建立下一份单变量authority；不得放宽NEAP门或把task15/18补跑冒充通过。

## Terminal iteration: V6-LPCP Task-Complete Endpoint Coexistence

- [x] 定位单一变量：NEAP endpoint/same-task/native链已通，held幅度在joint Value`.665x`到one-task direct rows
  `.223x`之间丢失；下一步先检验多个task credits能否在同一shared head update中覆盖condition manifold；
- [x] 写TCEC authority：保留NEAP/NZRB/LPCP全部图与equal-task mean，不加solver/memory/rank/scale；所有active
  task×4 endpoint margins全局共同下降才接受一个rank-synchronized candidate；
- [x] 原位实现probe list、global preference-row collective、统一scale/参数delta及world3固定anchor mode；
- [x] 完成定向`41 passed`、完整CPU`405 passed`、compileall、diff check与architecture guard 0 hard；
- [x] 记录唯一world3 launch contract：clean pushed code commit=`9ed6a082af20cd1d3668713f592753da6c2825a7`，
  detached frozen worktree=`/data1/user/ymdai/worktrees/EMBER-tcec-shared3-9ed6a08`，fresh root=
  `runs/outputs/pi05_v6_lpcp_task_complete_endpoint_coexistence_shared3_task9_15_18_b8_9ed6a08_gpu02p123_20260816`；
  2026-08-16 live双节点检查后选择gpu02物理`1/2/3`，UUID分别为
  `GPU-19cb2d30-d2ca-77b5-54a5-6b23fd2eede4`、`GPU-944f32c9-2377-7d51-5b03-c62053fc3ecb`、
  `GPU-c0c11da8-07ae-b68b-00dc-c3bda27d49ae`，三卡均0 MiB/0%且健康。gpu02另有两张空卡，但本门固定
  task9/15/18一task一rank的world3；使用6卡会改变科学拓扑或产生dummy rank，不是资源不足。`/data1` quota实时为
  542,436,820 KiB used / 1,073,741,824 KiB quota / 1,084,227,584 KiB limit，预计训练、日志与analysis峰值新增
  `<1 GiB`。复用canonical source step1000、tokenizer、target HDF5与LIBERO assets；fresh-only，不resume、不覆盖。
  exact command=`CUDA_VISIBLE_DEVICES=1,2,3 PYTHONPATH=<frozen>/src MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
  NCCL_P2P_DISABLE=1 <canonical>/.venv/bin/torchrun --standalone --nproc-per-node=3 scripts/train_reward_writer.py
  --config configs/pi05_writer_v6_lpcp_task_complete_endpoint_coexistence_v1.json --mode smoke --source-run
  <canonical>/runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722 --checkpoint
  <source-run>/checkpoints/step_00001000 --tokenizer-path <canonical>/models/tokenizers/openpi/paligemma_tokenizer.model
  --data-root <canonical>/data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a --output-dir <fresh-root>
  --smoke-task-ids 9,15,18`；三卡共同产生唯一delta，首门严格按authority的outcome/count、12/12、3/3、held8、
  temporal、rank-bank、wall与0禁读/OOM/nonfinite裁决；任一失败即终局，不补task、拆rank winner或参数小扫；
- [x] 两次pre-rollout启动工程失败均独立保留且不构成科学run：首个detached one-liner未把`--smoke-task-ids`传给
  workers，在创建root前CLI exit1；第二次已正确启动world3并写run contract，但launcher未source canonical
  `.env.local`，模型加载后在环境创建前因LIBERO assets路径缺失exit1，0 rollout/update。该非空root与两份log均不
  resume、不清空或冒充fresh。唯一修正版仍用同一code/scientific contract/GPU topology，只在launcher中source
  既有`.env.local`并把fresh root更新为
  `runs/outputs/pi05_v6_lpcp_task_complete_endpoint_coexistence_shared3_task9_15_18_b8_9ed6a08_gpu02p123_retry1_20260816`；
  exact training command其余不变，环境前缀增加`source <canonical>/.env.local`；
- [x] clean frozen `9ed6a08` world3共同运行task9/15/18：预定outcome/count与三rank ownership全部复现，
  cycle=`182.142s`、max reserved=`19.367GB`、0禁读/OOM/nonfinite；
- [x] 完整分析跨task gradient与12-view native panel：三task内部四video coherence均健康，但task15 gradient norm
  为task9/task18的`41.45x/10.43x`，task间cosine mean=`-.14513`，global raw mean仅1/3 task下降；11个scale
  最多8/12 margins下降，最终exact no-op；
- [x] 验证保存的四B heads共860,160参数逐元素全零，因此held8、时序与deployment correction必然为零，不重复
  GPU forward；终局artifact已封存；
- [x] 按预注册门终局，不full24、strict paired400、held controls、cycle2、task权重/normalization/solver或scale小扫。

## Terminal iteration: V6-LPCP Capacity-Matched Action-Probe Grid

- [x] 从TCEC最早断点选择单一架构变量：删除shared wide B heads，保持LPCP carrier、NEAP endpoint、K4、rank32、
  equal-task mean与native acceptance不变；
- [x] 完成CAPG design authority：同一forward保留18层raw Action-probe states，37个post-backbone parameter latents
  由rank16 B-only payload精确推导；逐video causal、K-set、layer/token M2P后direct reshape native-zero B；
- [x] 原位实现CAPG与fresh-incompatible config/checkpoint/eval schema，退休TCEC active runtime；
- [x] 完成step0 exact、payload-gate首步gradient、gate-open全链gradient、constant-zero、K-set、direct reshape、
  信息墙和one-forward CPU/机制验证；
- [x] live检查双节点GPU与`/data1`quota，建立clean pushed frozen world3 launch contract；空卡不足时可在实时余量
  充足且低util的`ycliu`设备上安全共驻；
- [x] clean frozen`878b5e4`固定task9/15/18完整运行preformal：outcomes/counts、wall、信息墙通过；same-task
  cosine/energy提升到`.983/.985,.898/.870,.982/.949`，raw shared coverage从1/3到2/3，native best从8/12
  到10/12；task15仍主导且task18反向，11 scales无12/12，最终exact no-op；
- [x] 封存`capg_shared3_terminal_adjudication.json`并按门终局，不运行held/full24/strict/cycle2或小扫。

## Active iteration: V6-LPCP Capacity-Matched Backbone-Memory Grid

- [x] 从CAPG最早断点选择预注册的单变量：保留其已通过的temporal/K-set/M2P/direct-B、NEAP、K4、rank32与
  global gate，只把post-backbone 37-query latents换成真实prefix内逐层更新的37个one-way memory tokens；
- [x] 冻结CMBG authority：37由rank16 B-only payload精确推导，Action不能看memory，memory能看真实
  image/language/Action/memory；不恢复旧8-token Dynamic-K整条路线；
- [x] 首版joint-matrix实现完成CPU/CUDA机制，但clean`38f7fc7` world3使task15固定outcome/count从`2/0,65,16`
  漂为`1/2,47,8`；已封存engineering adjudication，明确该run不能作memory科学裁决；
- [x] 在不改科学变量的前提下修正carrier接口：原生context forward一次，37 memory tokens作为one-way layer observer
  走真实Action Expert update；真实task15 K4的text/frame/grounded/interactions与18层Action states相对LPCP逐元素
  exact，task9 K4/112-frame zero staging、全链gradient、policy零gradient、wall/显存均通过；
- [x] 完整CPU与architecture gate通过，并在clean`2aecece` fixed world3完成task9/15/18：成功关系=
  `1/0,2/0,1/2`、selected=`8/16/8`，cross-task cosine mean=`+.09842`，raw/final=`3/3`、native=`12/12`，
  j0写出非零delta；历史chunk数只作诊断；
- [x] 完成validation8 four-view、reverse、constant、held/train幅度与throughput门：8/8 tasks，BA cosine/energy=
  `.983541/.985926`、raw/action=`.982788/.986590`、held/train=`.960650x`，全门通过；
- [x] 定位clean`a62348e`首次full24的工程失败：formal只保留view0，与global four-view commitment冲突，
  无checkpoint/completion或科学结果；最小修正为保留已计算的4 views，focused/related=`42 passed`、
  full CPU=`411 passed`；
- [x] 从clean`b4dbf84` fresh重跑full24 cycle1：24 tasks/48 pairs/96 rollouts、6 active tasks，cycle=
  `527.605s`，checkpoint/completion完整；11 scales最好仅`17/24` task-view margins下降，final delta与
  q/v/action/action-response全为0；
- [x] 按`restore_step0_parameters_and_terminal_non_pass`终局CMBG；strict400只会重测LPCP carrier而跳过，
  不cycle2、controls或小扫。
- [x] 从CMBG证明的最早缺口选择单变量CFMG：保留已通过的carrier/video-coherent memory，只把zero payload
  gate从temporal/K-set/M2P之前移到完整content grid之后，使首步gate gradient读取真实内容依赖程序；
- [x] 原位完成唯一canonical实现与fresh-incompatible config/checkpoint/eval schema；focused与完整CPU=
  `5/411 passed`，compileall/diff check通过，architecture guard无hard violation；
- [x] 完成fixed task9/15/18 world3：成功关系与selected pairs保持，三task跨video均过门，cross-task gradient
  cosine全正、raw/final=`3/3`、native=`12/12`、j0接受，cycle=`114.216s`；
- [x] 完成validation8 held四view：8/8 tasks、BA cosine/energy=`.982412/.985173`、raw/action=
  `.981593/.987546`、held/train=`.960548x`、reverse约2x；保留原始constant门单项false，并以exact constant-hidden
  为0和最坏public residual仅`.514%`作透明numerical adjudication；
- [x] 从clean`bb5341e` fresh完成CFMG full24 cycle1：24 tasks/48 pairs/96 rollouts、6 active tasks、cycle=
  `467.783s`，但11 candidates最好仅`14/24` margins下降，final delta与q/v/action response全为0；
- [x] 按exact no-op终局CFMG并跳过只会重测LPCP的strict400；不cycle2、controls、resume或小扫；
- [x] 从CFMG full24行级证据定位raw endpoint action secant尺度：六active tasks selected pair数相同，但task38
  action RMS为其它tasks的`4.5--5.9x`且Writer gradient支配达`58.73x`；
- [x] 建立USEP单变量authority：保留CFMG全部模型/数据/commitment，只按每个matched state的winner/loser RMS
  把endpoint cotangent变成unit secant；不读task ID/gradient norm，不改变task权重，不用MSE反比放大小pair；
- [x] 原位实现唯一loss owner与fresh config/checkpoint/eval identity；相关CPU=`143 passed`、完整CPU=`413 passed`、
  compileall/diff check与architecture guard 0 hard，且未增加forward、参数、module或并行runtime；
- [x] fixed task4/34/38 world3完整裁决：outcome/count复现，dominance=`6.1538x`、raw shared=`3/3`、native finite
  commitment=`12/12`，但task34 raw views仅`2/4`，按USEP预注册门终局；
- [x] 完成task34 action-hidden画面与stage localization：未见相反子任务顺序，Program/content/residual BA约
  `.98--.99`跨K4一致，最早冲突后移到condition-local policy/action Jacobian；
- [x] 建立USFC fresh successor：模型/目标/训练逐项不变，只让raw coverage降为diagnostic并保留actual finite
  all-view deployed descent硬门；切换fresh config/checkpoint/eval identity；
- [x] 完成USFC CPU合同、clean`db7ab24` push与gpu02 world6 fresh full24 cycle1：24 tasks/48 states/96 rollouts、
  5 active tasks、cycle=`480.284s`；exact Adam `j0`使`17/20` margins下降，但20/20硬门拒绝全部11 scales，
  saved delta0并跳过strict；完成逐task、rank负载与CFMG对比artifact，USFC终局；
- [x] 根据owner裁决建立USDC单变量authority：不把20/20换成另一个人为阈值，只提交一次未经缩放的exact Adam
  `j0`，20/20降为诊断，删除重复step0 baseline forward；保留unit-secant、四view与等task raw gradient；
- [x] 完成USDC canonical identity与单次j0路径；完整CPU=`413 passed`、compileall/diff check通过、architecture
  guard 0 hard violation，active source净减88行且没有新module/parallel path；
- [ ] 完成clean push与frozen full24 cycle1，随后立即strict400；
- [ ] 若cycle1有closed-loop正信号，锁定world topology exact-resume cycle2，必要时cycle3；每个checkpoint分别
  strict400并分析相邻churn/Jaccard，不能以偶然winner替代稳定共同积累。

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

无权限或资产阻塞。Ordered-Procedure AS139、raw reward138、ADSP138、V6-LPCP、PCSD、CV-CSD、SFMC、
Gradient-Open、CCT、NPVC、PAFS-NV、SJNV-Gate、DJNFR、DF-PCSP、DF-SOCP、MB-SOP与AR-EC均已按门终局；它们都
不得resume或参数小扫。AV-MBC、MMCD、PAV-BC、ALB-NV、NZRB-C与NEAP-C均已终局且无可resume checkpoint。
NEAP已证明真实10步endpoint能把task9从NZRB no-op变成高coherence j0 update，但held/train幅度`.234<.30`。
最新USFC full24已终局：exact `j0`达到17/20但被原20/20硬门恢复step0。当前active USDC只检验这个唯一自然
optimizer candidate的真实strict与连续训练稳定性；不得resume USFC/USEP、回退CFM、扫parameter ray/scale/
normalization/PCGrad，也不得把旧100分Dynamic-K整条恢复。
约145只有在相邻checkpoint低换手、same-task-other鲁棒且correct相对
wrong/shuffled/reversed/no-video明确占优时才算成立。生成LoRA后的task-local RL仍是初始Writer达到强
zero-interaction起点之后的独立实验。

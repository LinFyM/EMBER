# EMBER Persistent Plan

更新时间：2026-08-14。本文只记录长期Goal与当前迭代阶段；实时run状态见
`docs/active_session_handoff.md`，历史实验见`docs/research_history.md`。

## Goal

在owner已经对齐的科学目标、信息墙与工程原则下持续迭代EMBER，使同一shared Writer的单一checkpoint真正
利用正确教学视频获得跨初始化的高层任务知识，在strict paired closed-loop中稳定超过`150/400`并继续提高，
同时保持视频内容/顺序因果性、多task能力共存和实验效率。

Goal不绑定Dynamic-K、memory token数量、LoRA rank、mapper形式或optimizer；这些是当前可证伪方法变量。

## Success evidence

- single-checkpoint strict paired correct `>150/400`；
- 高breadth、低checkpoint churn、per-suite不过度集中；
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

## Active iteration: Semantic-Core Common-Value Set Bridge

- [x] 写单变量authority：只把set Value从weighted centered residual改为weighted raw common Core；
- [x] 原位更新canonical schema/config/runtime；K1、step0、raw-mean Value、set、顺序、gradient门与full CPU=
  `374 passed`；
- [x] 修正显式K1旁路的零导数训练语义；world6 full24 B20 profile通过：macro1/2=`25.930/22.530s`、
  K各6、最长323帧无截断、0 OOM/nonfinite，gradient约比centered路径打开三阶，formal config已seal；
- [x] 从clean sealed commit完成fresh macro0→25：25/25 metrics、完整checkpoint/completion/exit0，总耗时
  `614.636s`，gradient全程`.00250--.00325`；
- [x] 实际K4 B32 longest-panel一次确认通过：`.225360 LoRA/s`、reserved`13.181GB`、0 OOM/nonfinite；
- [ ] 从profile-sealed clean descendant立即做macro25 K4 strict paired400。

## Continuous loop after this result

1. 从strict closed-loop和逐task成功集合确定最早失效接口；
2. 保留未被否定且已接通的机制，只选择一个主要因果变量；
3. 写简洁、可证伪design authority，说明为什么能改善高层视频知识、正确顺序、policy-effective写出和共同积累；
4. 原位修改唯一canonical Writer path，不保留退役parallel implementation；
5. 完成最小必要CPU/机制验证和真实吞吐profile；
6. fresh训练到预注册节点，尽快做single-checkpoint strict paired400；
7. 深入分析后进入下一轮，直到长期Goal真正达成。

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

无权限或资产阻塞。Semantic-Core Set已在135/breadth7按门终局；Common-Value实现、profile、fresh macro25与
K4 B32确认均已通过，下一步strict400。

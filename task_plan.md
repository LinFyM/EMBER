# EMBER Task Plan

状态：2026-08-18 **completed，EMBER-LMMPC预注册训练轨迹与漂移归因已封存**。Core-Addressed Reader已按同一
world6/topology exact-resume到macro100，完成macro25/50/75/100四个同口径strict paired400及Program、FactorHeads、
B20 credit和shared retention联合归因。本轮没有建立或实现successor；下一session从封存结论重新建立goal。

## Goal

在EMBER稳定科学合同、owner原则与当前LMMPC主架构思想内，把该方法从明确设计推进到唯一canonical实现、机制和吞吐
验证、充分fresh训练、strict全面评测、逐接口问题定位与有证据的局部迭代。不得在尚未观察到性能峰值或训练仍有
有效共同上升趋势时过早判定无效，也不得因单次反馈或负结果大幅改换架构路线。

## Done when

满足下列之一：

1. 同一shared single-checkpoint方法稳定达到约145+或更高，同时具备合理breadth/retention、相邻checkpoint低churn、
   same-task不同teacher videos鲁棒，以及correct明显优于wrong/shuffled/reversed/no-video；或
2. LMMPC主链经过多轮充分训练、全面stage/closed-loop分析和针对最早断点的局部改进，仍重复无法达到资格，并形成
   接口明确、足以终局否决当前路线的证据。

## Fixed boundaries

- 不使用subagents。
- 保持exact language + action-hidden ordered videos → one shared Writer → one complete LoRA → frozen source policy。
- 正式架构保持`V6 Core/Procedure → Procedure reads layer/rank memory → address-preserving T/K aggregation → dynamic
  Core fusion → axial M2P → one native rank16 A/B LoRA`主链。
- 允许针对已定位断点局部调整temporal readout、K-set、Core gate、M2P/factor commitment和shared credit；不无证据
  切换到完全不同hypernetwork、expert dictionary、第二LoRA、checkpoint融合或生成后task-local RL。
- incompatible正式训练必须fresh。旧V6/LPCP activations只可做短机制接线诊断，不可成为最终方法或成绩。
- 好结果训练到相邻checkpoint有稳定性信息；坏结果不靠rank/scale/seed/LR/dtype小扫或无限续训挽救。
- closed-loop absolute首先选择方法，内部几何只作定位。

## Evidence plan

- 机制：one-forward、one-way memory、source zero-gradient、T/K不混layer/rank、identity路径、K permutation、
  reverse/shuffle重算、八factor family梯度、native BA/action、longest-video吞吐。
- 表示链：Core → Procedure → per-video memory → K-set memory → Core-fused grid → M2P → factor → BA → action。
- 正式：single-checkpoint strict paired400；per-task/per-suite/breadth/retained/gained/lost/churn/Jaccard。
- 强结果：correct/same-task-other/wrong/shuffled/reversed/no-video六臂与相邻checkpoint稳定性。
- 每轮先定位最早失效接口，再决定一个主要局部变量。

## Work plan

- [x] 根据owner讨论修正数据流：Action先形成V6 Procedure，Procedure再读取layer/rank memory。
- [x] 删除独立320-slot重寻址；聚合memory tensor直接作为20×16 axial M2P grid。
- [x] 将修正后的完整流水线、fresh边界、充分训练和局部迭代合同写入active design authority。
- [x] 核对现有canonical runtime owner、删除/复用边界并形成最小实现diff。
- [x] 实现LMMPC唯一运行面、fresh config/checkpoint schema和必要CPU合同。
- [x] 完成全量CPU验证、clean-commit真实full24动态K吞吐profile、最长K4机制证据与正式recipe封存。
- [x] micro5最长schedule条件与clean profile已经通过；重新fresh train24到首个有信息量节点macro25并完整封存。
- [x] 完成K4部署generation profile并封存batch32真实吞吐证据；macro25与macro50使用同一部署合同。
- [x] 同一run exact-resume完成macro26--50，保留完整macro25和macro50相邻checkpoint。
- [x] 完成macro25/50 strict paired400、逐task/suite、retention/churn及逐stage分析；定位到Procedure reader endpoint
  bypass，而非训练量或K-set首先失效。
- [x] 冻结v1负结果边界，明确v2仅修改Procedure→layer/rank memory接口和与同一shortcut绑定的训练合同。
- [x] 完成LMMPC-v2唯一canonical实现、fresh schema、全CPU合同和architecture gate。
- [x] 在真实K4上验证完整Procedure阶段被使用、constant identity、非硬反号、八family梯度、native BA及最长
  video吞吐；worktree结构和资源门通过。
- [x] 从clean pushed detached commit复现两macro吞吐、真实K4机制与371-frame门，并封存fresh formal recipe。
- [x] v2 fresh train24到macro25/50，执行strict paired400和完整stage/task分析；结果`71→73`、churn60，定位到
  unbounded M2P覆盖Core-fused task/order Program。
- [x] 用v2 macro50 hidden完成逐block和bounded-counterfactual：当前两层把order`.2573→.0938`、between-task
  `.3381→.6560`；bounded initial/max保留到`.2479/.2308`与`.3608/.4056`。
- [x] 完成v3唯一canonical实现、fresh schema、VL Meta-LoRA恒等路径清理、CPU/architecture合同。
- [x] 从clean pushed detached commit完成v3真实K4机制、world3两macro吞吐、371-frame门并封存formal recipe。
- [x] v3 fresh train24到macro25/50并完成strict paired400、逐task/suite/stage与retention/churn分析；结果
  `102→60`、churn70，定位到unbounded K-set覆盖per-video mean。
- [x] 用v3 macro25/50 hidden完成raw/mean/bounded K-set counterfactual，确认raw branch同时降低between-task
  separability和same-task coherence，而bounded branch保留部分order成分。
- [x] 完成v4唯一canonical实现、fresh schema、CPU/architecture合同、真实K4机制、吞吐、最长视频与formal seal。
- [x] 完成v4 K4 deployment generation profile并只写回evaluation throughput authority。
- [x] v4 fresh train24到macro25并完成strict paired400；同一run exact-resume到macro50并完成第二次strict、
  逐task/suite/stage、retention/churn和effective-BA分析。
- [x] 将v4与历史V6做同口径Procedure/reader诊断，确认原始Procedure趋同不是首因：V6从更趋同的Procedure中
  放大有向差异，而v4读取器继续衰减。
- [x] 完成v5唯一canonical实现、fresh schema、CPU/architecture合同，以及Core条件化、完整有序Procedure读取、
  constant identity、K置换、八family梯度、native BA和最长371-frame机制门。
- [x] 完成validation8 matched v4/v5 reader诊断：raw Procedure差异不变而reader/raw由`.718x→1.819x`，H_set
  within-task仍`.970`且between-task降至`.245`，通过formal前置接口门。
- [x] 从sealed clean pushed commit完成v5 fresh train24到macro25，封存world6完整checkpoint、generation profile和
  validation8 pre-strict stage证据。
- [x] 执行v5 macro25 strict paired400和全量逐接口分析：`123/400`、breadth8，相对v4 macro25严格净增19，但
  相对LPCP143净丢20；reader改进保留，最早缺口后移到functional cotangent到native factor commitment。
- [x] 保持原world6/topology exact-resume v5 macro26--50，完成第二次strict paired400和相邻checkpoint稳定性分析：
  strict由`123→84`，`71 retained / 13 gained / 52 lost`，确认当前recipe未稳定共同积累。
- [x] 完成macro25/50逐stage、all400 effective-BA、first4 factor结构及同task四K4 update coherence联合分析；排除
  Procedure/reader失效、video-local update分裂和LoRA未写出为本轮首因。
- [x] 撤回macro50对整个recipe的过早终局判断，按既有预注册合同锁定macro50→100 exact-resume与macro75/100
  两次同口径K4 strict paired400。
- [x] 保持原world6和gpu01物理`1/2/4/5/6/7` topology，从macro50 exact-resume到macro100并保留完整macro75/100。
- [x] 完成macro75/100 strict paired400；联合macro25/50/75/100报告逐task/suite、breadth、retained/gained/lost、
  churn、Jaccard与same-row恢复，区分共同积累、单纯回升和循环换手。
- [x] 用四checkpoint Program/BA轨迹与同架构Program×FactorHeads交叉解码诊断，拆分Program drift、FactorHeads
  coordinate drift、B20 functional credit和shared retention责任。
- [x] 更新active design、findings与research history，形成新session可直接接续的一个最早失效接口和下一单变量结论；
  本goal不实施该successor，完成后停止。

## Current decision

EMBER-LMMPC Core-Addressed Reader的同一formal run已完整训练到macro100。四个K4 strict paired400为
`123→84→89→87`，breadth为`8→5→6→4`；macro50之后只有小幅回升，未恢复macro25，更没有形成多task共同积累。
400个固定rows中仅49个始终成功、150个曾成功；macro25→50丢失的52行只有22行在macro75或100任一点恢复，至
macro100只恢复15行。macro25→50新增的13行到macro100只保留6行。该轨迹证明先降后升确实存在，但属于循环换手，
不是被macro50过早截断的共同性能峰值。没有checkpoint达到145，故不做六臂controls。

固定K4+B20 train24 loss在macro25/50/75/100为`.112124/.099353/.098427/.101337`：25→50有19/24 tasks改善，
strict却净丢39；50→75几乎平台；75→100已有13/24 tasks变差。交叉解码同时显示每个相邻区间的compiled Program
relative-L2仍为`.770/.730/.710`，而只换FactorHeads或只换Program都会造成material BA变化。FactorHeads主导
25→50的norm扩张，后期则与Program漂移责任相当；它是错误credit的放大与承载者，不是可单独冻结便能修复的根因。

终局最早失效接口是**静态cross-episode offline B20 functional credit没有约束held closed-loop occupancy上的
shared support retention**。Core-Addressed Reader、Dynamic-K、bounded K-set/M2P、memory carrier和native rank16
FactorHeads均保留为已接通机制；Procedure趋同仍应监控，但不是本轨迹最早抹掉task/order信息的接口。下一session
最直接的单变量候选是保持整套架构与decoder不变，只把functional query distribution替换为train24 on-policy state
replay，并用冻结task-local experts提供action targets，以测试occupancy-matched credit能否保留breadth和相邻
strict success set。它不是本轮active successor，也尚未获得实施授权；本goal至此完成并停止。

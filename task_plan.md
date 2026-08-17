# EMBER Task Plan

状态：2026-08-18 **active**。本计划对应当前持续研究goal；当前phase为LMMPC-v3终局定位后，原位实现和验证
LMMPC-v4的mean-anchored bounded K-set commitment。

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
- [ ] 完成v4唯一canonical实现、fresh schema、CPU/architecture合同、真实K4机制、吞吐、最长视频与formal seal。
- [ ] v4 fresh train24到macro25并完成strict paired400、逐task/suite/stage与retention/churn分析；有真实共同上升
  证据时继续同一run到macro50。
- [ ] 对有希望checkpoint继续相邻训练；首次约145补六臂和same-task视频鲁棒性。
- [ ] 若存在问题，定位最早接口并在LMMPC主链内做单变量局部改进，重复充分训练和全面评测。
- [ ] 达成性能资格或形成多轮终局证据后，更新历史与findings并完成goal。

## Current decision

当前active design为`docs/layer_matched_memory_program_compiler_design.md`。v3的bounded M2P确实把
Core-fused→compiled的改写限制在约`.25x`，但macro25/50 strict为`102→60`；25→50=`46 retained / 14 gained /
56 lost`、churn70、net`-42`。最早断点前移到K-set：raw nonlinear consensus相对per-video mean改写
`10.188x/5.831x`，把between-task cosine由`.654/.767`破坏到`.903/.922`，并降低same-task coherence。下游Core fusion
反而恢复task分离，bounded M2P只小幅改写，所以不能把回落继续归咎于M2P或训练量。

v4唯一增加与M2P同形的mean-anchored逐cell K-set commitment：fresh gate初始`.25`、最大`.5`；不删除learned
cross-video correction，也不改变Core/Procedure、memory、M2P、rank16、loss或recipe。当前CPU实现接通，下一步从clean
pushed detached commit验证真实K4机制/资源并封存formal recipe；这些机制证据尚不是closed-loop成绩。

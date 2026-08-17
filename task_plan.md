# EMBER Task Plan

状态：2026-08-17 **active**。本计划对应当前持续研究goal；当前phase为修正LMMPC最长条件显存切片并重新封存profile。

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
- [ ] micro5最长schedule条件与clean profile通过后，重新fresh train24到首个有信息量节点且不在未达峰值时过早终止。
- [ ] 执行strict paired400并完成逐task、逐suite、retention/churn及逐stage分析。
- [ ] 对有希望checkpoint继续相邻训练；首次约145补六臂和same-task视频鲁棒性。
- [ ] 若存在问题，定位最早接口并在LMMPC主链内做单变量局部改进，重复充分训练和全面评测。
- [ ] 达成性能资格或形成多轮终局证据后，更新历史与findings并完成goal。

## Current decision

当前active design为`docs/layer_matched_memory_program_compiler_design.md`。canonical实现、fresh schema与CPU合同已经
落地。clean `de0b298`的首个world5 formal在macro1--16保持有效下降：functional `.15609→.11888`、Program matching
`.36194→.17621`；macro17的rank2没有进入唯一gradient collective，最终定位为task38/K4/359帧在functional
microbatch6下真实OOM，而非NCCL、架构负结果或训练峰值。microbatch5在B20下仍为4次policy forward，并已完整通过
原故障五任务序列及100-macro封存schedule的真实最大task38/K4/371帧条件。当前只等待clean micro5 full24 profile
重新封存recipe；失败run没有checkpoint，不可resume，随后必须fresh重启。

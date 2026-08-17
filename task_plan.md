# EMBER Task Plan

状态：2026-08-17完成。本文记录本轮仓库整理后认知重建与新架构设计goal；实现与GPU实验属于后续独立工作。

## Goal

基于已经完成的仓库整理和统一历史证据，系统重建EMBER认知，逐步定位最早未解决接口，并形成一套符合owner原则、
继承有效历史机制、具有明确数据流水线与可证伪实验合同的新架构设计。设计讨论完成前不启动GPU实验。

## Done when

- 不以“问题清单”冒充结论：对Program、compiler、reward direction与shared retention逐层作出证据裁决。
- 明确哪些问题已解决、哪些仅局部接通、哪一个是下一架构应改变的最早接口。
- 新设计给出从language/videos到一套完整LoRA的端到端数据流水线，并解释每个模块的必要性。
- 动态K、每video内部时序、跨video共同信息、language/video分工和single-LoRA部署合同全部闭合。
- 说明memory、V6/LPCP carrier、native rank16 compiler和历史reward机制哪些保留、哪些替换及原因。
- 训练目标、fresh/warm-start边界、single-variable合同、快速否决门、strict paired400与六臂时点明确。
- 设计不靠额外target-task数据、task ID、expert route、checkpoint union或生成后task-local RL提高当前分数。
- owner看完后能判断完整pipeline和关键取舍，而不是继续追问各token/分支“到底在做什么”。

## Constraints

- 不使用subagents。
- 设计讨论完成前不实现新架构、不启动GPU训练或评测。
- memory token、rank、V6 compiler、MCPS和RL都只是候选方法，不预写进最终结论。
- 优先复用现有formal artifacts做只读分析，不为分析增加大cache、防御性hash或复杂框架。
- 一次只选择一个主要因果变量；负结果只淘汰实际检验的组合。
- closed-loop absolute首先选择方法，稳定性、same-task video鲁棒和视频因果性决定方法资格。

## Work plan

- [x] 完成仓库、文档、代码与历史证据的统一整理；clean commit `120eeec`已推送。
- [x] 明确当前真正未决的架构分叉：Program形成、native compiler还是shared credit/retention。
- [x] 建立V6/LPCP、Dynamic-K、GOMQ及关键reward路线的stage-wise证据矩阵。
- [x] 判断GOMQ 151→135→131中，能力丢失首先发生在representation、compiler还是policy credit。
- [x] 判断memory-derived Program与V6 native rank16 topology能否形成有原理依据的统一接口。
- [x] 比较GOMQ Direct-B、memory-only LPCP Query和Layer-Matched Memory Program Compiler，给出决定性取舍。
- [x] 写完整新design讨论稿：输入、表示、memory/Program、聚合、compiler、LoRA、训练和评测。
- [x] 将完整设计、三项核心取舍与实现边界交付owner讨论；本轮保持无active GPU run。

本轮最终设计提案为`docs/layer_matched_memory_program_compiler_design.md`。它完成了本goal要求的设计推理与实验合同；
是否升级为active implementation authority、是否启动实现/GPU，属于下一决策，不由本goal自动授权。

## Evidence adjudication

1. Dynamic-K的between-task结构在M2P/final Program仍约`.49/.53`，到family hidden/B才升为约`.63/.78`同向；最早
   明显collapse在nonlinear compiler，不是raw carrier。
2. GOMQ的`.993`说明相邻shared BA update没有被four-K4 video-set相消，但isolated memory-only coherence仅`.127`；
   gained/lost改写幅度不可分，151回落首先支持shared reward/held retention失败，不能反推Program已经完善。
3. LPCP证明layerwise carrier有效；其BA相对AS139仅改`.002653`又说明143大量来自旧baseline support。V6 native
   rank16 topology可保留，但冻结Procedure Query→W2路径不可原样继续。
4. 历史尚未把“可检测顺序”变成稳定有用方向。LMMPC用反对称Program与language matching阻断旁路，再由correct
   functional loss和六臂闭环裁决，不把latent margin冒充答案。
5. shared failure不是容量单因：GOMQ memory/downstream跨task gradient低coherence、持续失败样本改写大且held
   gained/lost不可分。先修Program→native commitment；reward/optimizer另轮处理。
6. sealed LPCP143加zero-forward、同rank16的LMMPC A/B residual branch提供step0 support；最终同拓扑必须fresh训练，
   bridge不冒充成品。

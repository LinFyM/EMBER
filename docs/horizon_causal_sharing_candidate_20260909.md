# 当前Horizon输出共享：候选对照准备，不启动训练

本项在两组语言路径学习期间准备，服务原因分析goal。旧v6重训已暂停；不恢复它，也不枚举历史架构。此文只确定一个当前接口的可检验差异，CPU实现验证不等于值得启动或行为有效。是否投入学习须结合正在运行的语言对照及剩余解释决定，不自动launch。

## 具体问题与历史反证

当前完整LoRA的末层为`D[target,rank,side,native,256]`，各rank独立。不同rank的条件代码仍可共享上游，但它们不能直接共同更新同一个native输出映射。待检验的问题是：这种独立性是否使当前协议下的训练获取更容易，而可迁移、可保留的修正更难共同学成。329,515,008个D参数占比高只是结构事实，不是过拟合诊断。

历史Target-Owned已经使用同target内跨rank共享D，末投影20,594,688参数，correct为99/76/86/68。完整历史设计在`3a6f801d:docs/action_forecast_writer_target_owned_factor_design.md`§1–8，索引见research_history§2。它同时采用每target独立1024→256前投影、Core/private A/E/D前端、full24×20、decay400，并替换Direction Stores。因此它反对“rank共享普遍有效”，不能给当前仅改变末层的效应定量；也降低了无新证据就投入重训的优先级。

旧v6的family共享还跨layer，且有不同前端/Meta；本候选不恢复该架构包，不测试跨target共享。历史梯度方向低重合与层几何不能单独给当前问题定责，旧最终视频controls不用于本候选设计。

## 唯一改动与共同计算

候选将末层改为`D[target,1,side,native,256]`，对同target各rank广播使用。A/B不共享，target之间不共享，native通道自由度不改；完整38 targets、rank16、全部76 tensors保持。共享参数由各rank梯度自然求和后进入同一AdamW，不另加手工归一化、学习率补偿或scale。

保留当前逐帧contextual语言、完整H、真实视觉/图文prefix、四组过程读取/回写、Compiler、target/rank queries、全局U_A/U_B、A0模板和零D初始化。没有平均视频、C或最终LoRA。不同rank仍由不同代码生成不同A/B；共享输出映射不强制rank输出相同。

新旧共同模块按同顺序初始化。D都是零分配，不消耗随机数，所以改变其shape不改变共同随机初始化。两者fresh都产生原identity LoRA；incompatible D/optimizer shape必须fresh，不允许从当前checkpoint切换后称同一学习实验。

这个干预检验的是输出参数绑定及其内在梯度聚合效应，不能将结果进一步拆称纯参数量、纯正则化或纯优化器因果。若后续需要区分这些因素，必须另有必要性与设计，不能预先展开网格。

## CPU准备与行为裁决

隔离分支从当前main建立，复用唯一trainer/materializer/evaluator；显式探索schema与architecture字段阻止误读旧checkpoint。main正式实现不改。只准备以下直接验证：

1. 完整LoRA identity与native shape正确；共享参数数量是独立D的1/16。
2. 同一个D广播到rank时，forward与显式绑定的独立表达式一致；反向共享D梯度是各rank贡献之和，跨target/side不串线。
3. 非零D后原生视频、过程与Compiler仍能获得梯度；不同rank代码仍能产生不同输出。

CPU检查只证明改动真实且局限在该接口，不证明泛化收益。若决定启动，沿用语言对照的train24合法池、4×64、相同实际采样、seed7、LR3e-5/warmup8及200/400固定节点，比较对应的当前contextual基线；先真实profile，再fresh学习，不能用CPU通过直接launch。

两个固定节点均需validation400及held-video train96的逐task/suite、breadth和R/G/L/churn。train提高但validation不改善，不能称迁移修复；validation收益若伴随大规模丢失或仅单点出现，不能称保持改善。两侧均更差只否定这个具体绑定，不证明所有共享有害。稳定且有广度的收益才支持继续用独立教师/状态复核，不自动正式采纳。

目前没有该候选的训练、checkpoint、闭环结果或启动安排。当前运行仍只有两组语言路径对照。

## 已完成的准备与未验证范围

隔离实现位于`codex/horizon-causal-sharing`，从main建立，沿用已检验的探索训练/物化身份处理，只在当前`native_factor.py`增加同target跨rank绑定；没有新增model模块、runner或架构包。探索schema与`decoder_rank_sharing`字段拒绝误读旧checkpoint，formal入口不可用。候选不合入main；最终不采用时由Git保留原件，清理此task-owned工作树，不留下canonical fallback。

模型/物化/FM三组CPU检查共86项通过（首次85pass，1项测试对无梯度的None处理修正后定向pass）。共享和独立两种参数化均验证了identity后的完整图功能反向，显式验证共享梯度为rank贡献之和、跨target/side隔离、不同rank输出仍可不同。配置解析通过。精确参数计数：Writer368,675,520→59,755,200，D329,515,008→20,594,688；只报告数量，不把减少解释为疗效。

结构检查无hard violation；review项来自既有大文件/长函数，实际方法改动集中于两个已有owner，Git authority仅作隔离分支替换。没有GPU profile、真实学习、闭环结果或launch脚本；86项CPU通过不能代替这些证据。

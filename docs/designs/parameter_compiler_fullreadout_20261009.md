# 冻结χ360的完整补充读出：区分晚期学习与局部能力交换

2026-10-09；主讨论负责科学取舍，实验session负责实现、执行与Git闭环。
本合同依据Owner持续自主推进及自主预算授权登记；原批次的预注册停止、全部阴性和费用保持原样。

## 1. 实际问题与改变判断的新证据

原[参数条件编译合同](parameter_conditioned_compiler_20261009.md)在χ180的127/400及train MT/end/null27/25/25后
按合取规则科学停止。这个判断及execution退出记录不撤销，不改选χ180。
合法并行已经完成χ360全部训练，而long-first队列实际完成的32个Long1条件为20成功，同32行χ180为6、MT16。
χ360对χ180 R5/G15/L1、对MT R13/G7/L3，是值得解释的局部正证据；样本集中于同一task的state0–31，不能代表8-task整体。
两个train48总分均25且有相邻能力交换；当前不能区分“晚期学习形成了更广能力”与“Long1改善被其它任务退步抵消”。

χ180之前720学习事件全部是MT incoming；χ360后半加入真实刷新，但全程只有81/1440事件为非MT incoming。
180读出既未覆盖完整后半学习，也未测出所有后继递推行为。另一方面J1也丢MT能力，不能仅用递推曝光解释全部不足。
因此主讨论修正的是对早期停止证据的外推范围：不据180否定已实际出现的晚期学习，也不据32行宣布完整方法成立。

最小直接证据是完整读出已经冻结的唯一原终点χ360。它比再开训练、改架构或局部调参更直接回答上述取舍。
这是看到局部结果后明确登记的开发补充评估，透明保留适应性决策来源；不倒写成原批预注册通过或独立确认集。

## 2. 固定对象、执行范围及配对

- 唯一checkpoint：`/data1/user/ymdai/ember_runs/parameter_conditioned_compiler_20261009/training/checkpoints/macro_00000360`。
- 原32行全部复用，新增只执行原formal360中尚未完成的368个condition；合并后恰好原定400行。
- 每task全部50个state与50条teacher各一次，固定原task/state/video ordinal、exact language、condition seed、scene和env/policy RNG。
- 每condition仍固定MT起步，最多1024实际环境steps（含settling），成功/截断后先编辑，最后固定唯一完整LoRA；最终state不参加该condition实践。
- 最终仍为原官方render/rotate/state/action、10-flow、前5动作重规划和suite horizon；只correct arm，无Test、视频controls、其它checkpoint或新训练。
- 不因中途分数停止剩余条件或选换checkpoint；工程故障保留有效负例和已完成原件后接续。科学输出是单一χ360完整配对400。

原ROOT和原队列/停止标记不改：
`/data1/user/ymdai/ember_runs/parameter_conditioned_compiler_20261009`。
新增root：`/data1/user/ymdai/ember_runs/parameter_compiler_fullreadout_20261009`。
原32行用明确来源引用或小metadata导入接入新的完整队列，保留其原代码身份和原始路径，不重跑或复制大经验/参数载荷。
新368行使用新root，完整聚合消费者按相同配对语义读取32旧行与368新行；导入、复用及新增范围在completion中分列。
不得删除父root的early_stop标记、篡改scientific_stopped或令旧Supervisor重新启动训练/刷新。

## 3. 解释与后继分支

报告完整χ360对MT153、T161、旧experience135和χ180的逐task/suite、breadth、R/G/L、churn及success-set重合。
将原32和新增368分列，Long1的原32/剩余18分列；分列只解释采集来源，不用其局部分数代替完整400。
所有自身成功/预算失败、实践steps、真实J、读取次数、失败尾部和最终新初态结果一并保留。J是事后分组，不作轮数因果结论。

若完整χ360仍没有总体收益，且train无净修订，降低当前学习组织作为近期修复的优先级；不追加原样续训或用局部高点保护它。
若出现超过强参照的整体收益，说明180的早期阴性不足以排除晚期学习；再由main结合能力保持、覆盖、训练曝光与成本选择后继。
无论何种结果，这次补充本身不证明视频/经验的独立贡献或正式相邻稳定性，也不自动选点、开controls或新一轮训练。
结果须回到“哪些实际经验/参数修订获得了什么可迁移能力、哪些能力仍被丢失”的原问题，而不是只排列分数。

## 4. 实际投入与保存

上批formal180完整400实践/最终共367796环境steps、两worker计费约3.1001GPUh，提供同一消费者的估计依据。
预计新增计算45–120分钟、2–4GPUh；准备/物理修复与科学消费预计另30–60分钟。
本批成本观察线8GPUh、科学elapsed4小时；这些由main自主负责调整，不是Owner审批门槛，显著超期须说明原因及可执行取舍。
368×1024=376832实践steps，原各suite最终horizon及settling合计最多119040步；新增环境总上界495872。
原32行已发生的22362实践/13279最终steps和原批费用仍归原批，不因复用重复计费或扣除。

新增存储计划峰220GiB，包含原始经历、完整中间/最终参数、派生cache、临时写入及代码；不再沿用未兑现的旧145–200GiB整批预测。
运行前在strg01重新核对data1独立quota、当前个人实际用量及共享容量，测新增root并根据实际每condition增长修订预测。
优先复用合法资产和已有有用缓存，host cache总量有界；生命周期已结束的可重建cache按Owner授权自主退役并记录释放量。
当前依赖、关键权重、原始科学证据和完整180/360恢复资产保持；不为掩盖超计划删除原件，也不切到data0写入。

主讨论在实际存储消费中发现一个明确的物理保存问题：`Runner._infer`将批量H的单slot view保存到chain，
`torch.save(chain.to_record())`会连带保存底层整批storage。Spatial0/teacher43的train180原件有151个H，
逻辑29.492MiB而不同底层storage合计200.781MiB；train360为38.086对121.680MiB。
这不改变实际读取的H值，也不是当前科学阴性的解释；它会放大存储和I/O，须由实验session在canonical保存边界做窄修复，
使新condition只保留其自己的逻辑tensor载荷。只做实际序列化消费者的针对性检查，保持模型/数值/标签/实践/配对语义，
由新clean pushed detached版本承接，训练代码身份与补充读取代码身份分开。原32有效行和旧原件不重写。

GPU按当前仓库准入和有用吞吐分配，可跨两节点安排独立persistent评测；总物理卡及单节点上限保持，不人为锁死两卡。
复用实际B16/native64/reader128/经验64/decoder131072消费者与已有profile；有明显可用余量时按实际吞吐作必要物理调整，
保持科学矩阵及逻辑随机流，不用重复大profile或dummy占卡制造工作。

## 5. 执行、所有权与回报

main登记并推送后，已有实验session独占必要工程/执行/Git窗口；main继续只读科学分析，不重复工程验收。
复用canonical evaluator/queue/runtime；必要的小范围导入/保存修复由执行者自主实现、检查、推送并建立clean detached运行版本，
保持一个active运行面，不恢复旧frozen路径或平行fallback。完整旧科学产物及故障费用保留。
只保留一个可存活整批完成/异常回报，等待实际退出事件；不轮询正常进度、不叠加自Queue或心跳。
整批原件消费、资源退出和Git闭环后一次回报当前main `01a11b05-3460-7eb0-aca8-177d6d86ef48`并交回所有权。
main根据完整结果主动裁决和接续。此补充读出解决证据缺口，不被当作已经改进了方法。

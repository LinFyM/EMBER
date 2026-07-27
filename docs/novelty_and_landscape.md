# Novelty and Baseline Landscape

## EMBER 的核心问题

直接target action-SFT很强，但要求目标机器人轨迹。EMBER研究的是：一个在独立source corpus上获得基本embodiment能力的VLA，能否在held task只看一条action-hidden teaching video，就生成比无视频source adaptation更有用的task LoRA；如果环境practice继续放大差异，再增加reward-adaptation claim。

当前证据链：

1. generic π0.5在目标8 tasks为0/400，说明必须先建立公平共享的LIBERO source foundation；
2. 过滤后的LIBERO-90 action-SFT产生共享frozen π0.5-LIBERO base；
3. Source-SFT只使用目标source actions，held不看视频；
4. AS/RL Writer在held看一条正确视频并生成task LoRA；
5. cross-suite wrong-video控制视频内容是否真正有因果价值；
6. test-task identity/AS/RL Writer三臂RL检验初始化是否改善practice；
7. 8-test联合action-SFT shared LoRA给出privileged ceiling。

当前唯一活动Writer是Semantic Core + Causal Procedure v5。它不再把
action-hidden teacher video强行解释成与独立机器人episode逐时刻对齐的7D
future-action trajectory，而是把单条视频分解为两种互补证据：

- 对帧集合置换严格不变的Semantic Core，保存对象、场景、目标关系和整段视频
  共同支持的高层任务内容；
- 对真实顺序敏感的可变长causal Procedure，保存这些交互状态如何有向演进。

Core先生成稳定的LoRA内容，Procedure只作zero-init的有向修正。这一设计直接吸收
v4 `shuffled=148/400 > correct=109/400`的教训：shuffle没有创造更多任务知识，
而是破坏了压过高层语义的低层translation controller bias；新架构必须先保住
这部分高层任务内容，再让正确顺序提供额外增益。

训练时每rank每step只处理一个task：1条teacher video生成1套one-shot LoRA，
整批`B_a`条独立action queries全部监督该LoRA；每条action只计算一次，推理
仍严格one-shot。每套video LoRA由完整action batch约束，要求其跨初态有效；
后续task visits轮换video，共享Writer的跨step梯度应强化跨示范稳定的高层
语义；demo-specific速度、路径和抓取角度则因与宽action分布不一致而难以稳定
解释监督。这仍需same-task一致性、
wrong/shuffled/reversed控制和rollout共同验证，不能仅凭内部LoRA不同宣称成立。

## 为什么共享LIBERO-90 base不破坏故事

所有方法都需要一般视觉、语言、机器人控制和action-space能力。用与目标40 exact task去重后的LIBERO-90 actions训练共同base，等价于一般机器人foundation adaptation；它不向任何方法泄露目标task actions。真正受比较的信息差仍然是held task video、held reward或held actions。

若base过弱，Writer必须同时解决embodiment与task acquisition；若base已有基本能力，correct-video与wrong-video、Source-SFT之间的差异更能回答视频是否提供目标知识。base不应被故意训弱，也不应先追求把目标40做满。

## Source-SFT

Source-SFT从同一frozen base出发，在24/32目标source tasks上联合训练一套shared LoRA，held只靠language/current observation。它与Writer各自按validation选最佳，不强制相同步数；通过完整报告action chunks、GPU-hours和参数量解释计算差异。

## Wrong-video是核心机制对照

错误视频来自另一suite，正确language与执行task保持不变。若Writer只是生成通用adapter或主要依赖language，wrong-video可能同样提升；`correct - wrong` 才是视频内容价值的直接证据。这不是独立训练Language-only/Video-only Writer arm，不违反精简baseline原则。

对当前v5还必须同时报告same-task other teacher、shuffle与reverse。期望关系是：
same-task other与correct的policy-function差异最小且表现接近；correct稳定优于
wrong、shuffle和reverse。大多数当前操作任务都包含有价值的阶段顺序，因此
`correct > shuffled/reversed`仍是硬门，而不是允许相等的装饰性指标。

## ViVLA 是最直接的可选 matched baseline

ViVLA-style方法在执行时直接以expert video、当前观察和language condition policy；EMBER则把视频一次编译成LoRA。公平matched版本使用同一个frozen π0.5-LIBERO source base、24/32 source tasks、one-video held输入、相同target-action wall和evaluator。是否每步重看视频是方法差异，不是不公平。

优先完成EMBER核心闭环；有时间再实现ViVLA-style matched reproduction，并报告success、video preprocessing、policy latency、显存和rollout throughput。原生ViVLA的大规模外部数据结果只作相关工作，不与matched因果表混写。

## Test-task reward adaptation

task-local RL在最终test task本身训练，与一般RL benchmark在同一任务训练/测试的口径一致。它不需要validation预冻结；但三种初始化仍应使用相同环境/策略seed schedules、RL代码和可比资源，报告完整learning curves、interactions-to-best和fresh fixed-state结果。

## Joint direct-action oracle

target-action oracle在8个test tasks上联合训练一套shared LoRA并使用每task全部50条actions。它回答“目标actions可见时，同一共享LoRA空间可达到什么水平”，不是同信息墙baseline，也不是8套task-local adapters。

## Seen-task evidence

seen panel用于区分“方法根本没在source distribution学会”与“source acquisition成立但held transfer失败”。它必须覆盖四suites并在outcome前选定；不能替代validation/test，也不能单独证明泛化。

## 相关工作边界

R+X、SeeTraceAct、RAD、RoboCasa等继续作为视频数据场景与相关工作参考；当前实证收敛在LIBERO-90 source foundation + LIBERO-40 target benchmark。sim-to-real与真实机器人暂不进入核心执行。

## 当前不允许的结论

- generic π0.5的0/400不是EMBER失败，只说明原始base不适合直接执行LIBERO。
- 新source base在40-task快速screen有少量成功不等于EMBER成立。
- correct-video只高于base但不高于wrong-video，不能充分证明视频内容被利用。
- Source-SFT或joint direct oracle强不能否定视频setting，因为它们使用额外action labels。
- task-local RL不是EMBER必要组成；RL-Writer失败也不能抹掉AS-Writer结果。
- 旧SmolVLA/70-10-10结果不能冒充新π0.5协议证据。

# EMBER Novelty and Baseline Landscape

状态：2026-08-13稳定研究定位。本文不授权当前run或绑定具体架构。

## Research position

EMBER研究video-conditioned parameter generation for robot control：shared Writer在rollout前读取一次exact task
language和一条或多条action-hidden正确教学视频，生成一套完整task adaptation；frozen source policy随后从未见
初始化闭环执行。

它不是：

- 直接target action-SFT，因为部署task没有action labels；
- 每步video-conditioned policy，因为示范不会在每个control step反复输入；
- trajectory imitation，因为video与action supervision跨episode、不逐帧对齐；
- retrieval/expert routing，因为held部署不能读task ID或expert bank；
- language-only adapter，因为video必须是唯一dynamic value；
- 只追求LoRA reconstruction的通用Hypernetwork，因为价值由robot closed-loop决定。

潜在贡献不是“首次生成LoRA”，而是在严格信息墙下同时解决：action-hidden视频高层理解、正确顺序、一次性
policy参数编译、跨初始化泛化、动态视频数量和single-checkpoint多任务共同积累。

## Relationship to mature Hypernetworks

SHINE、Doc2LoRA等工作说明：在原生backbone context中放置少量memory、保留逐层状态、再通过共享结构化mapper
生成LoRA，是比每个参数独立预测更可扩展的设计原则。EMBER学习这一原则，但不机械复制它们的文本输入、token
数量、rank或flat payload。

EMBER新增的困难是视频内部有向过程、多个videos的集合结构、异构robot-policy target topology、action-hidden
跨episode监督、闭环状态分布和视频因果controls。外部研究只提供架构参照，不增加target-task训练数据，也不能
替代EMBER从v4到v6、K4和task-drift实验形成的负结果边界。

## Empirical landscape

| reference | deployment information | role / best evidence |
| --- | --- | --- |
| generic π0.5 | language + observation | 未适配foundation下界；目标8曾为0/400 |
| frozen source base | language + observation | 过滤source actions建立的共同起点；48/400 |
| mixed-task Source-SFT | language + observation；训练读target actions | privileged shared-LoRA reference；109/400 |
| EMBER AS-Writer | language + action-hidden video(s) | 核心zero-interaction方法；历史最好143/400 |
| optional RL-Writer | 同上；训练期读source reward | 检验action-free practice能否改善Writer |
| matched direct video policy | 每步language + video + observation | 可选比较一次编译与持续conditioning |
| direct-action oracle | language + observation；训练读test actions | privileged ceiling，不是同信息墙baseline |

共同比较应共享source policy、normalization、split、task states、policy interface和official evaluator。参数量、
训练actions、视频数、rollout interactions、wall time、policy latency和显存分别诚实报告；不要求机械相同步数或
FLOPs来掩盖方法本身的取舍。

## Historical constraints on novelty

- action-hidden视频含可解码task与时序信号，但视频敏感不等于正确理解；v4曾有`shuffled=148>correct=109`；
- Semantic Core/Procedure分解能增强wrong-video辨识，但compiler可把顺序差压到行为无效；
- v6-fast证明共同source base上的Writer可达143，但晚期训练和后续路线持续task换手；
- 更高rank、更均匀谱、更多atoms/lanes/experts、SFT量级norm和更低functional loss均非充分条件；
- task experts是有效task-level policy targets，却不含same-task video specificity或时序；
- K4能改善部分集合稳定性，但旧实现没有解决strict performance或full24 credit retention；
- rank14去混杂证明compression与online regeneration可独立破坏closed-loop support；
- Dynamic-K 100和semantic-address 101说明真实memory、动态K与Query semantic address仍未自动形成正确policy方向。

因此下一贡献不能只是换一个decoder名字。它必须说明有向视频证据如何落到有用policy direction，以及不同
task/video能力为何能在同一checkpoint共存。

## Required causality controls

wrong控制language-only或generic-adapter旁路；same-task-other测demo nuisance；shuffled/reversed测帧序因果；
no-video测动态通路必要性。所有arms严格配对state、RNG和video identity，并在真实frames重排后完整forward。

理想证据是correct高且广、same接近correct、wrong/shuffled/reversed/no-video因正确原因实质更低。absolute仍是
第一目标；一个`correct=80, wrong=20`的漂亮margin不能取代`correct=143`的强方法。

## One-shot, few-shot and scaling claims

one-shot要求从一条demo区分任务本质与偶然性；few-shot把跨demo共同信息变成可学习对象。动态K方法应逐video
保序、跨video置换不变，不平均生成后的LoRAs，也不挑video。

论文claim由实际最强可靠设定决定，而非预先强制K1对K4的人工公平：

- K1最好：报告one-shot；
- 固定K>1最好：报告few-shot；
- 同一模型在多个K上稳定且随K提升：报告dynamic-cardinality/scaling。

每种claim都要报告实际K、帧数、计算和训练曝光，并用single-checkpoint closed-loop与视频controls验证。内部
一致性提高而absolute/retention不提高，不构成核心进步。

## Claim boundaries

- zero-interaction claim：correct-video single checkpoint严格超过matched baselines并有因果controls；
- stability claim：breadth高、相邻checkpoint churn低、重复训练呈共同积累；
- few-shot/scaling claim：更多videos稳定增加真实能力，不来自挑video、平均LoRA或计算隐瞒；
- adaptation claim：相同reward interactions下，视频生成LoRA提高后续learning curve或终点。

generic base的0/400、训练loss、small panel、checkpoint union、expert reconstruction或漂亮LoRA谱都不能单独
推出上述claim。精确实验谱系见`docs/research_history.md`。

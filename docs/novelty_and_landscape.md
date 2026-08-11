# EMBER Novelty and Baseline Landscape

状态：2026-08-11稳定研究定位。当前没有active successor；本文不授权实现或实验。

## Research position

EMBER研究的是video-conditioned parameter generation for robot control：shared Writer只在rollout前读取一次任务
语言和action-hidden教学视频，生成完整task LoRA；frozen source policy随后从新初始化闭环执行。

它与几类相邻问题不同：

- 不是直接target action-SFT：部署task不提供action labels；
- 不是每步video-conditioned policy：示范不会在每个control step反复进入policy；
- 不是trajectory imitation：video与action supervision跨episode，不要求逐帧对齐；
- 不是retrieval/expert routing：held部署不能读取task-expert bank或task ID；
- 不是通用hypernetwork几何任务：生成LoRA的价值由robot closed-loop而非reconstruction决定；
- 不是language-only task adapter：video必须是唯一dynamic value。

潜在贡献不在“首次生成LoRA”这一单点，而在一个严格信息墙下同时解决：action-hidden视频理解、一次性参数
编译、跨初始化泛化、跨task共同积累，以及视频内容/顺序的闭环因果验证。

## Matched empirical landscape

| reference | information at deployment | role |
| --- | --- | --- |
| generic π0.5 | language + observation | 未适配foundation下界；目标8曾为0/400 |
| frozen source base | language + observation | 过滤source actions建立的共同embodiment起点；48/400 |
| mixed-task Source-SFT | language + observation；训练读target actions | privileged shared-LoRA reference；109/400 |
| EMBER AS-Writer | language + one action-hidden video | 核心zero-interaction方法；历史最好143/400 |
| optional RL-Writer | 同上；训练期读source reward | 检验action-free practice能否改善Writer |
| matched direct video policy | 每步language + video + observation | 可选baseline，比较一次编译与持续condition |
| direct-action oracle | language + observation；训练读test actions | privileged ceiling，不是同信息墙baseline |

所有matched比较应共享source policy、normalization、split、task states、policy interface与official evaluator。
参数量、训练action chunks、rollout interactions、wall time、policy latency和显存应分别报告，不用机械相同步数
掩盖方法差异。

## What the historical evidence already establishes

- action-hidden视频确实包含可解码的task与时序信号；旧Gate -1与后续hidden/LoRA/action反事实均支持这一点。
- 视频敏感不等于正确视频理解：v4曾出现`shuffled=148 > correct=109`。
- Semantic Core/Procedure分解能增强wrong-video区分，但compiler可把顺序差压到行为无效。
- v6-fast证明one-shot Writer在共同source base上可达`143/400`，但晚期训练与大量后续路线持续换手。
- 更高rank、更均匀谱、更多atoms/lanes/experts、SFT量级norm与更低functional loss均不是性能充分条件。
- task experts是有效task-level policy targets，却不携带same-task video specificity或时间顺序。
- few-shot K4能改善部分集合/leave-one-out稳定性，但尚未解决full24 credit retention和strict performance。
- 最新rank14去混杂证明，即便parameter reconstruction很小，compression与online regeneration也可独立破坏
  closed-loop support。

因此下一贡献不能只是“再换一个hypernetwork decoder”。它必须以已有失效链为约束，说明为何视频的有向
证据能落到有用policy direction，且不同task/video更新能在同一checkpoint共存。

## Required causality controls

wrong-video控制language-only或generic-adapter旁路；same-task-other控制demo nuisance；shuffled/reversed控制帧序
因果；no-video控制动态路径必要性。所有arms需严格配对state、RNG和video identity，且重排真实frames后完整
forward。

理想关系不是只让negative变差，而是：

```text
correct high and broad
same-task-other close to correct
wrong, shuffled, reversed, no-video materially lower for the right reason
```

absolute仍是第一目标。一个`correct=80, wrong=20`的漂亮margin不能优于`correct=143`的强但视频margin弱方法；
健康度和因果controls是解释与约束，不是替代性能的主指标。

## One-shot versus few-shot novelty

one-shot要求从单条video区分高层任务与低层偶然性，是最强设定。few-shot可以把“跨demo共同信息”显式变成
统计对象，但会带来新的公平性问题：video数量、总帧数、Writer FLOPs、挑选策略、每条内部时序与集合聚合。

值得研究的few-shot方向应同时具备：

- 每条video内部使用有序encoder；
- videos之间用置换不变集合聚合；
- 能处理固定`k`，之后再讨论variable cardinality与mask；
- 不平均生成后的LoRA，不做多checkpoint融合；
- 与one-shot按video/frame/FLOP预算清楚对照；
- 仍通过same/wrong/shuffled/reversed/no-video和single-checkpoint paired400裁决。

few-shot若只提高内部一致性而不提高absolute/retention，不构成核心进步。

## Claim boundaries

可以主张的最终结果取决于证据：

- zero-interaction claim：correct-video single checkpoint严格超过matched baselines并有因果controls；
- stability claim：多task breadth高、相邻checkpoint churn低、重复训练呈共同积累；
- few-shot claim：多video在匹配口径下改善one-shot且不是挑video/平均带来的假象；
- adaptation claim：相同reward interaction下Writer初始化提高learning curve或终点。

不能从generic base的0/400、source base的少量成功、训练loss、small panel、checkpoint union、expert
reconstruction或漂亮LoRA谱单独推出任何上述claim。精确实验谱系见`docs/research_history.md`。

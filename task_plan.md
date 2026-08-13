# EMBER Persistent Plan

更新时间：2026-08-13。本文只记录长期Goal与当前迭代阶段；实时run状态见
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

## Current iteration: Direct-Family-B

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
- [ ] 完成macro50 single-checkpoint strict paired correct400；
- [ ] 分析absolute、per-task/per-suite、breadth、retained/gained/lost、能力集中和最早失效接口；
- [ ] 按预注册门决定终止、exact-resume到100或补完整视频controls。

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

无权限或资产阻塞。当前工作由active formal run和随后strict400决定。

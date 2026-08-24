# EMBER owner requirements

本文只记录长期有效的目标、边界与协作要求。动态进度见`progress.md`，执行计划见`task_plan.md`，历史事实见
`docs/research_history.md`。

## 1. 最终目标

EMBER必须从generic `lerobot/pi05_base`建立的冻结PI0.5-LIBERO source policy出发，只接收：

- 目标task的exact language；
- 一条或多条同task、action-hidden、内部有序的正确教学视频。

Writer在rollout前运行一次，直接生成一套覆盖Action Expert全部38个目标层的完整task-conditioned LoRA。冻结policy加载这
一套LoRA后，应从未见初始化闭环完成相同或相近场景中的任务。部署时不得再次观看视频，也不得进行环境交互、task-local
优化或第二阶段适配。

语言负责说明目标与关注对象，视频必须提供不可被语言或静态端点取代的动态证据。模型应理解“在什么条件下、按什么过程
完成任务”，而不是记住task ID、文件名、场景模板或少量训练任务特征。

## 2. 不可改变的部署合同

- 输入：exact language + `K`条action-hidden ordered videos。
- 输出：唯一一套完整rank16 LoRA；rank和内部参数化若有强证据可重新讨论，但不得因历史惯性固定成rank12+rank4。
- source PI0.5完全冻结；默认只修改Action Expert，不让Writer改变Gemma权重。
- 每条视频独立保序编码，跨视频只做置换不变聚合；不得平均frames、raw features或最终LoRA。
- 每个condition只生成一套LoRA；不得挑video、融合checkpoint、部署第二adapter或并行expert。
- Writer只在rollout前运行一次；zero-interaction分数不混入生成后的task-local RL。
- deployment Writer不得读取teacher action、state/proprio、reward、terminal、task ID、filename、pose或policy outcome。

训练期可在授权的non-held tasks上使用actions、privileged task experts、simulator reward和occupancy学习共享机制，但这些信息
不得成为deployment输入、held dictionary或task-ID route。

## 3. 数据边界

- 固定benchmark为LIBERO Spatial/Object/Goal/Long 40 tasks。
- 固定development split为`configs/libero_24_8_8_v1/`的24 train / 8 validation / 8 test。
- source corpus为LIBERO-90排除与目标40重合的19项后剩余71 tasks，每task 50条成功episode。
- 后继meta-training可以使用train24和经精确语义审计、排除validation/test及重复项的其它现成LIBERO tasks。
- 当前不制作人工任务、人工process数据集、人工controller trajectory或额外仿真场景；推进必须直指ECP核心。
- 不得使用读过目标40 actions的`pi05_libero`。
- video和action query同task但跨episode采样，阻断逐帧轨迹复制。
- validation/test actions或reward不得产生梯度；Test默认留到最终方法冻结。

held5只是train24内部的leave-task-out机制门，用于在不消费validation设计信号的前提下快速检查共享映射。正式开发选择必须回到
固定validation8；方法确定后才按合同从fresh使用32 source tasks训练并评测test8。

## 4. 当前方法方向

经过专家和owner反复讨论，当前方向称为ECP（Event-Conditioned Policy Compiler）。它不是GOMQ、PECS或历史v24的改名，
也不要求继承这些实现。ECP的核心是：

1. 用PI0.5原生语言、视觉和Action Expert内部时序结构形成有序、layer/target对应的Video Program；
2. 用跨任务共享且部署兼容的机制把Program编译为一套完整LoRA；
3. 训练期privileged posterior和部署期video posterior必须输出同构Program，避免任意latent与decoder共同旋转；
4. 最终端到端Writer必须在冻结backbone的前提下进行联合训练，而不能永远停在互相割裂的阶段模块。

Program当前候选schema使用固定最大`E=8`个event slots；slot激活数量、视频段落到slot的分配由内容学习，不由人工规定。固定
容量并不意味着每个任务使用相同数量的事件。schema和后续阶段将在待回专家意见中再次确认，未确认的实现细节不得冒充最终
架构。

首版可以不启用Action Meta-LoRA，但后续必须做一次与canonical匹配的尝试；只要没有负面影响并能改善probe稳定性、视频
因果性或闭环表现，就启用并冻结。历史中性结果不能被忘记，也不能被包装成永久禁止。

shuffled/reversed不进入训练、loss或checkpoint选择。它们只在最终冻结checkpoint上作为严格配对的时序特异性测试；正确
视频应稳定优于打乱与倒序输入。full video还必须优于language/no-video、scene/first+final和wrong-video controls。

## 5. 成功标准

- 继续追求validation8 `>150/400`，越高越好。
- 约145也可构成有价值结果，但必须由相邻single checkpoints、低churn、高breadth和same-task不同视频鲁棒性共同证明。
- Goal与Long必须有真实贡献，不能依靠Spatial/Object掩盖失败。
- full video必须有必要条件增量，并在多数任务上形成收益；same-task其它视频应保持高retention。
- shuffled/reversed最终表现应揭示真实时序特异性，而不是仅让内部latent距离变大。
- closed-loop absolute表现优先；loss、reconstruction、LoRA norm/cosine、hidden margin和surrogate只用于定位。

一次实验必须报告per-task、per-suite、breadth、retained/gained/lost、churn和相邻checkpoint success-set重合。明确坏结果不靠
小幅seed/LR/rank/scale或超长续训挽救；负结果只淘汰实际检验的组合，不外推为整个EMBER目标失败。

## 6. 推进方式

- 先理解因果链和最早失效接口，再实现；不得用连续版本号替代思考。
- 每个阶段都要说明：输入数据、训练模块、冻结模块、输出、验证问题、通过条件和失败后的分支。
- 优先做能改变路线判断的实验；不钻无关紧要的shape、低位浮点误差、防御性代码或冗余测试。
- 不新增MD5/SHA-256 sidecar和大规模逐tensor校验；只保留信息墙、shape、finite、OOM、pairing、asset、checkpoint与resume所需
  的直接检查。
- 复用已经训练出的可用资产，避免重复长训练；profile和smoke只做最小必要验证。
- 遇到困难先回看专家原始意见与修正，检查执行是否偏移，再决定是否实验或咨询。
- 专家意见是设计约束与启发：不能为了速度随意丢弃，也不能不经理解机械照搬。
- 只有性能显著跃升、路线存在实质歧义或需要新增权限时，在关键节点暂停和owner讨论；不频繁汇报。

## 7. GPU、仓库与文档

- 每次GPU launch前同时live检查gpu01/gpu02；单个job只用一个节点，最多6张真正提高吞吐的A40。
- EMBER并发总量通常不超过6张；只有大量空闲时最多8张。可与低显存、低util进程安全共驻，但不得抢占、kill或reset。
- gpu01物理0若仍标记prohibited则不得使用；GPU身份和状态每次都要重新确认。
- 正式训练遵守storage quota、clean pushed commit和frozen worktree合同；探索实验不做冗余流程。
- canonical集成目标是`main`。只有需要隔离或并发写入时创建`codex/*`分支和worktree，验证后尽快合并、推送并清理。
- 不在活动树保留退役实现、平行fallback、过时配置、重复文档或临时结果；历史由Git、formal artifacts和一份精简历史记录保存。
- 代码、文档、branch、worktree和运行产物应在每个阶段及时整理，不等到几十版后集中失控。

## 8. 沟通与交接

- 未经owner当次明确允许，绝不能直接向外部专家发送消息；只能提供可复制prompt给owner。
- 给专家的prompt只补充他未知的新事实、结果与问题，不重复整段既有对话，也不人为限制专家的核心判断。
- owner主要语音输入；明显同音词或断句错误要按EMBER上下文理解。
- 跨session前，稳定要求、设计、历史、当前进度与下一步必须写入仓库；新session不应要求owner重新解释项目或GPU约束。

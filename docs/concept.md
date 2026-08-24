# EMBER concept

## 问题定义

人看过一段没有动作标注的教学视频，通常会先理解目标，再把视频中的条件、过程和结果迁移到自己的身体与当前场景。EMBER
研究PI0.5能否做同一件事：只看task language和`K`条action-hidden正确视频，在rollout前把观察到的知识编译成Action
Expert的一套LoRA，随后零交互完成任务。

这不是视频检索、task-ID分类、行为克隆或运行时视频条件策略。部署时没有teacher action、state、reward和第二个expert；
Writer只运行一次，输出的参数必须直接成为闭环策略的一部分。

## 为什么问题困难

原生PI0.5中，Gemma处理当前language和静态图像prefix，Action Expert把50个未来horizon位置上的noise tokens通过flow
matching推进为动作chunk。教学视频则是一串跨时间的静态帧，而且没有teacher actions。EMBER必须同时解决三个接口：

1. 从帧级PI0.5表示中提取与动作过程相关、而非只识别物体或task模板的动态证据；
2. 把可变长度、可变`K`的视频压缩成保留event顺序和Action Expert层对应关系的固定结构；
3. 用跨任务共享的编译机制把这个结构转成闭环有效的完整LoRA，而不是只重建某个expert的因子或内部hidden。

训练task数量有限还会造成欠识别：language、video和task identity可能高度相关，模型即使完全忽略过程也能降低训练loss。因此
方法必须靠task-disjoint评测、视频controls、多个独立策略lineages和真实closed-loop结果证明因果路径。

## ECP假设

当前方法方向称为ECP（Event-Conditioned Policy Compiler）。核心假设是：教学视频中的可迁移知识可以表示为一个有序、
event-conditioned、与Action Expert target/layer对齐的Program；同一个Program schema既能由训练期privileged policy evidence
推断，也能由部署期language+video推断；一个任务共享的realizer再把Program编译为唯一一套rank16 LoRA。

```text
deployment: exact language + ordered action-hidden videos
              -> q_V(Program)
              -> shared Program-to-LoRA realizer
              -> one complete rank16 LoRA
              -> frozen PI0.5 closed loop

training only: successful policies + actions/occupancies/reward
              -> q_pi(Program)
              -> teach/calibrate the same Program and realizer
```

`q_pi`不是手工标签或外部专家。它是一个只在授权meta tasks上训练的共享网络，用privileged policy evidence推断Program
posterior；其价值必须由task-disjoint、冻结realizer的闭环结果证明。`q_V`是部署Writer的video posterior。两者输出同构，
因此privileged信息教的是可由视频预测的中间结构，而不是隐藏的task-local code。

## Program候选结构

每个视频帧使用原生PI0.5 prefix和一组固定Gaussian action probes。flow时刻`t=1`表示denoising的噪声端点：输入仍是50个
按未来horizon排列的noise tokens；它们的中间hidden是当前language/image条件下的时间索引policy response，不是已经预测好的
50步动作，也不包含teacher action。

当前Stage 0候选保留38个LoRA target owners、50个horizon位置和各层hidden，再将帧序列分段为最多`E=8`个有序event slots。
`E=8`是固定最大容量；每个任务实际激活多少slot、哪个视频段落写入哪个slot均由模型学习。跨视频聚合只在event对齐后进行。

当前候选Program为：

```text
P_lang    [38, 128]
P_scene   [38, 128]
P_process [8, 38, 128]
rho       [8]            # event presence
sigma     [8, 38, 128]   # cross-video uncertainty
```

这是已讨论的schema，不是专家最终回复前可随意扩写的架构。slot数、坐标、posterior形式和realizer具体网络可被新证据修正，
但必须保持：事件顺序、target/layer对应、Dynamic-K真实性、部署信息墙和单LoRA输出。

## 训练原则

- 只使用现成且授权的LIBERO tasks，不制作人工process数据集。
- train24与审计后的non-held LIBERO-90 meta tasks产生梯度；validation/test不产生梯度。
- video与action query跨episode；多个successful policies用独立优化lineages构成分布，不把同一轨迹的checkpoint当独立任务知识。
- 先证明Program-to-LoRA共享映射在held tasks闭环成立，再训练video posterior；否则`q_V`会学习一个没有政策意义的latent。
- staged gates用于定位接口，最终必须有冻结backbone、全Writer联合训练阶段。
- shuffled/reversed只在最终冻结checkpoint评测时序特异性，不进入训练或选模。

## 目前知道与不知道的

已经知道：task-local rank16 LoRA有足够闭环容量；Action Expert内部能捕获任务相关动态结构；共享carrier可提供有限支持；过去
失败主要集中在把结构稳定地编译为跨任务有效LoRA，而非证明输入输出目标不可能。

尚不知道：现有自然LIBERO任务是否足以识别强过程Program；`q_pi`如何在不变成task dictionary的情况下形成可迁移posterior；
realizer应直接预测LoRA、预测policy-effect distribution还是使用可微inner solver；以及Stage 0的native probes是否比纯视觉
时序编码提供稳定净增量。这些是待专家回复后必须明确的核心问题。

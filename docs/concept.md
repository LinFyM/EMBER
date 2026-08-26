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
3. 让这个结构直接从PI0.5各LoRA目标的原生input/output空间选择低秩因子，而不是从低维latent凭空生成高维参数或把held更新
   投影回fit-task固定span。

训练task数量有限还会造成欠识别：language、video和task identity可能高度相关，模型即使完全忽略过程也能降低训练loss。因此
方法必须靠task-disjoint评测、视频controls、多个独立策略lineages和真实closed-loop结果证明因果路径。

## ECP假设

当前方法称为ECP Native-Factor Compiler。核心假设是：教学视频中的可迁移知识可以表示为一个有序、event-conditioned、与
Action Expert target对齐的Program；同一condition在冻结PI0.5各目标层中产生的原生input/output activations提供task-specific
参数基底，Program只学习选择、组合与缩放这些向量。

```text
exact language + ordered action-hidden videos
              -> Pass A: q_V(owner-specific Program)
              -> Pass B0: Program-conditioned native anchors + current-bank statistics
              -> regularized bank-conditioned query solve
              -> Pass B1: replay the same bank and exact signed pooling of real X/Y
              -> current first implementation: rank4 task residual + frozen rank12 carrier
              -> one complete rank16 LoRA
              -> frozen PI0.5 closed loop

training only: successful policies + verified occupancies/actions/effects/reward
              -> nonparametric set-valued functional critic
              -> supervise generated policy response, never produce Program
```

canonical不再包含神经`q_pi`或privileged Program teacher。privileged evidence仍重要，但只在训练时以global-member set loss和
functional/effect critic约束最终生成的policy，不进入deployment forward，也不要求video预测不可观察的recovery信息。

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
tau       [8, 2]         # event center and duration
sigma     [8, 38, 128]   # cross-video uncertainty
```

这是专家复核后固定的schema。Pass B另外读取每个q/v/action-in/action-out目标的真实input/output以及output的adjacent、init、goal
differences。当前bank-conditioned实现先按每条视频单位质量流式累计均值、协方差和Program-conditioned native anchors，求解
regularized query，再重放同一bank，由正负softmax之差对真实X/Y做exact signed pooling并产生rank4 outer products；这些量不是
Program字段，也不在内存中整段物化。内部两阶段读取仍属于rollout前一次Writer调用。

## 训练原则

- 只使用现成且授权的LIBERO tasks，不制作人工process数据集。
- train24与审计后的non-held LIBERO-90 meta tasks产生梯度；validation/test不产生梯度。
- video与action query跨episode；多个successful policies用独立优化lineages构成分布，不把同一轨迹的checkpoint当独立任务知识。
- 当前先用task-local free-code证明native factor bank与pooling具有闭环容量；通过后再训练Natural Program和冻结Program的shared
  compiler，避免在核心参数基底无效时训练更大的Writer。
- staged gates用于定位接口，不是Final必须重演的训练课程。Final既保留从已验证组件初始化的fresh joint run，也保留整套Writer
  完全随机初始化并直接端到端fresh训练的正式选项，由同一closed-loop合同选择。
- shuffled/reversed只在最终selected checkpoint已选定并冻结后评测时序特异性，不进入训练、loss、
  checkpoint选择、G1--G5 Gate或架构修正依据。

## 目前知道与不知道的

已经知道：task-local rank16 LoRA有闭环容量；Action Expert内部能捕获任务相关动态结构；rank12 carrier有有限支持；mobile rank4
解析投影在held5具有5/5容量；policy-effect objective对known-success paths有用；fit-span realizer会丢失held低能量创新。12+4因此
是首版最合理的参数分配，但不是不可由capacity evidence推翻的永久结论。

G1已经证明自然视频产生的target-native banks与exact signed pooling可形成强task-local rank4 residual；G2已经证明Natural Program
保留了可用的视频动态。当前未知集中在G3：共享compiler能否用Program-conditioned内容兼容性与每个当前bank的统计量，学习跨task、
跨video稳定的bank-conditioned selection。旧candidate-local one-pass实现对随bank covariance旋转的解析dual/score不可辨识；下一步
先验证新operator能否精确恢复native-factor容量，再验证shared anchor mapping，而不是继续回归逐video dual标签。

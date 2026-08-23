# EMBER Concept

## 一句话定义

EMBER让一个shared Writer把“任务语言 + action-hidden正确教学视频”一次性编译为一套task-conditioned policy
adaptation，使已有通用机器人能力的冻结source policy能从未见初始化完成新任务。

## 人类学习类比

一个会基本打乒乓球的人，看别人正确示范一次逆旋转发球后，通常不会逐帧复制对方的关节轨迹。他会提取更抽象的
知识：球拍怎样接触球、运动方向怎样产生旋转、动作阶段如何衔接。这个理解先给他一个明显好于盲试的起点；之后
根据真实击球结果继续练习，才逐渐熟练。

EMBER要复制的是这种“从正确视觉示范获得可迁移技能起点”的能力，而不是把视频伪装成动作标注。

## 问题合同

```text
exact task language
    + one or more internally ordered, action-hidden teaching videos
    -> shared Writer runs once
    -> one complete task-conditioned LoRA
    -> frozen π0.5-LIBERO source policy
    -> closed-loop execution from unseen initialization
```

语言告诉模型任务关注什么、目标是什么；视频告诉模型正确过程如何演化。两者联合应形成一个跨初始化成立的高层
task Program：对象与关系、目标状态、必要阶段、阶段间因果顺序，以及可忽略的demo-specific nuisance。

## 为什么视频不能等同于轨迹监督

teacher video可以来自不同视角、速度、路径和抓取姿态，部署环境的初始状态也不同。合理模型不应复刻原demo的
低层轨迹，而应使用冻结policy已有的视觉、语言和动作先验，在当前观测下重新实现同一目标。

因此训练可让video episode与action query episode同task但错开。这阻断了逐帧动作复制捷径，却也使监督target对
同task不同video可能恒定。架构和objective必须额外解决这个不可识别性：task identity正确不等于视频过程理解。

## One-shot、few-shot与动态K

一条视频足以定义one-shot问题；多条视频则允许比较同task示范，过滤单demo的偶然细节。若采用多视频：

- 每条video内部必须有序编码；
- videos之间必须置换不变聚合；
- 聚合对象应是高层程序证据，不是raw frame、feature或最终LoRA的简单平均；
- 训练必须覆盖声称支持的K；
- K由真实性能选择，不由形式偏好决定。

多视频只是提供可识别性的机会，不会自动产生正确task Program。历史上K4确实降低过same-task LoRA方差，也曾只是
更稳定地保留错误方向。

## 正确时序的因果意义

correct视频展示物理可行的初态→目标态过程；shuffled破坏阶段连续性；reversed颠倒有向因果关系。模型需要利用
这种结构判断“先做什么、后做什么、为什么”，而不是仅对时间戳、动作phase或negative标签敏感。

真正的证据是correct视频相对same-task-other、wrong、shuffled、reversed和no-video，沿有用policy方向提高闭环
成功率。hidden或LoRA不同、negative变坏、内部margin变大都不是充分证明。

现有LIBERO source/target reward只约束最终状态，没有已证明的same-endpoint/different-required-procedure任务对。
因此时序control可以证明模型对视频顺序敏感并且这种敏感性有用，却不能单独把结果升级为一般“过程理解”；
该claim还需要训练数据中真正排除endpoint、language或task-identity捷径的process-identifying mappings。

## 输出为什么仍是一套LoRA

LoRA提供一次性、可缓存、可挂载到冻结policy的task adaptation，并能自然成为未来task-local RL的起点。Writer
生成LoRA不意味着必须直接回归每个A/B元素；当前更有依据的实现是先形成与policy topology对齐的Program，再预测
event/layer/family policy-effect distribution，最后由固定、受约束的realization solver在PI0.5自己的参数坐标中生成完整LoRA。

rank、memory token、parameter grid、FactorHead和decoder只是实现选择。LoRA是否“健康”最终看它能否产生合理
effective BA、action response和closed-loop improvement，而不是强求高rank、正交或均匀能量。

## 学习系统的四个接口

EMBER可以分成四个必须同时成立、但应分别诊断的接口：

1. **Evidence extraction**：语言确定语义query，视频提供有向动态Value；
2. **Program formation**：同task不同video形成可复现的高层表示，多task保持可分；
3. **Policy compilation**：Program先约束策略响应等价类，再被固定实现器写成native、policy-effective的一套LoRA；
4. **Shared credit and retention**：训练更新在同一checkpoint中积累多个tasks，不轮流换手。

“视频被读到”“LoRA非零”“reward gradient存在”“single checkpoint分数高”分别只关闭其中一个局部问题。

## 当前成功定义

正式方法由single-checkpoint strict paired400选择。目标不仅是高correct，还包括：

- 高task breadth；
- 相邻checkpoint稳定、低churn；
- same-task不同视频鲁棒；
- correct明显优于wrong/shuffled/reversed/no-video；
- 能从Program追踪到LoRA、effective BA、action和闭环收益；
- 不依赖language-only shortcut、挑video、expert dictionary或checkpoint union。

zero-interaction Writer是当前研究对象。生成LoRA后的环境交互和task-local RL是合理的第二阶段，但必须单独评价，
不能替代初始adaptation本身的能力。

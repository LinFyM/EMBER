# EMBER Current Owner Requirements

本文只记录owner的稳定目标、共识原则和协作边界，不记录逐轮实验状态或下一条架构。当前进度见仓库根目录
`progress.md`，历史证据见`research_history.md`。

## 1. 项目的现实出发点

EMBER源于人通过正确示范快速学习新技能的经验：一个已有基础运动能力的人，看一条第三人称教学视频后，虽然不
会逐帧复刻示范者，也未得到手把手动作轨迹，仍能理解“这件事本质上怎么做”，立刻获得比盲试更好的起点；之后再
根据环境反馈练习，才逐渐精通。

对应机器人任务，教学介质首先是视频而不是动作序列。模型需要从正确视频提取跨初始化仍成立的任务知识：对象与
关系、目标状态、必要阶段、阶段之间的有向因果顺序，以及哪些细节只是某条demo的速度、路径、视角、抓取角度或
偶然扰动。部署目标不是复刻teacher原始低层轨迹，而是在自己的观测和初始化下完成同一任务。

因此EMBER当前核心问题是：

```text
exact task language + one or more action-hidden correct teaching videos
    -> one shared Writer
    -> one complete task-conditioned LoRA
    -> one frozen source policy
    -> closed-loop success from unseen initialization
```

## 2. 输入必须同时包含语言和视频

语言不能删除：只看视频时，模型未必知道任务要求关注哪个对象、关系或结果。视频也不能删除：只看语言时，模型
只知道“要做什么”，不知道一条正确示范所提供的“怎样完成”。合理分工是：

- language提供task query、关注点、目标与语义地址；
- video提供动态过程、状态变化、阶段顺序和正确做法的Value；
- 两者联合形成task Program，任何一方都不能成为可删除装饰。

language-only路径不得独立写出有效LoRA。反过来，也不能为了阻断language shortcut而制造不自然的zero-image、
空prefix或与π0.5原生计算无关的fake action query。

## 3. One-shot、few-shot和动态视频数量

one-shot仍是清晰且有价值的canonical问题，但方法不必先验锁死K=1。多条同task视频有合理优势：它们可帮助过滤
单条demo的低层nuisance，提取跨video共同的高层程序。

若声称支持动态K，则训练时必须真实覆盖不同cardinality，不能只训练K1/K4后口头外推。数据流应满足：

1. 每条视频独立保留内部顺序；
2. 不同视频在集合维做置换不变聚合；
3. 提取共同语义的同时保留每条视频的时序证据；
4. 不平均frames、raw features或生成后的LoRAs；
5. 不挑“最好视频”；
6. 所有视频保持action-hidden。

不要求为了论文形式做人为削弱的K1/K4公平对照。最终若K1最强就做one-shot claim，K4/K8最强就做few-shot，动态
K都强才做video-scaling claim。选择依据是真实闭环性能与因果证据。

同一video生成的LoRA应对同task多种未见初始化和action queries有效。训练时video与action episode同task但
跨episode错开，是阻断逐帧动作复制的重要手段；但它也造成“同task监督target对所有video恒定”的不可识别性，
所以仅输入正确顺序或重建task expert并不能证明模型理解了视频过程。

## 4. 正确顺序必须有结构作用

correct、shuffled和reversed不是人为negative：

- correct展示从初态到目标态的物理可行演化；
- reversed产生违反正常因果方向的过程；
- shuffled破坏阶段连续性和先后依赖。

架构应在每条视频内部显式保留有向过程，而不是只把shuffle/reverse推远。最终需要证明correct沿有用policy direction
提高成功率；仅让negative LoRA变坏不够。

## 5. 输出是一个task adaptation；具体参数化是方法

当前部署输出是一套完整38-target task-conditioned LoRA。Writer只在rollout前运行一次，policy闭环时不反复看
teacher video。一个condition只能得到一套LoRA，不能按video分别生成再平均，也不能用checkpoint union、expert
route或第二套adapter。

LoRA rank、memory token数量、FactorHeads、A/B生成方式和decoder都属于可修改的方法变量，不应写成goal。
owner当前认为rank16完全可以保留；此前建议rank8只是降低生成维度的一种思路，不是硬要求。

LoRA健康度应参考正常task-local SFT的policy geometry：不能小到近identity，也不应无理由所有targets完全共线；
但低stable rank、q-dominant和跨列coherent本身可能是正常expert结构。高rank、均匀能量、正交、更多atoms/lanes/
experts均不是独立优化目标。

优先分析effective `BA`、fixed-action response和closed-loop，而不是raw A/B gauge符号。

## 6. Memory token和成熟Hypernetwork工作的准确启发

SHINE、Doc2LoRA等工作值得学习的不是表面“加token”，而是第一性原理对应关系：原backbone本来能处理输入内容，
为目标参数层设置少量可学习memory tokens，让内容处理过程中形成与层/参数位置有对应的state，再把这些state解码为
各层LoRA。

对EMBER，这带来三个有价值候选：

- memory可让视频/语言内容在policy topology中形成layer-matched坐标；
- layer correspondence可能比一个扁平latent直接吐出百万参数更可扩展；
- Writer规模、基础模型规模和LoRA target数量可通过同构token/grid扩展。

但memory token不是必须形式，也不能为了使用Action Expert而运行缺失其原生图像/文字prefix的无意义forward。
π0.5的Gemma/VLM与Action Expert各自承担什么原生计算，必须先从真实backbone接口推导，再决定memory放置、读写方向
和是否需要纵向/横向交互。

历史GOMQ已给出重要但有限的正证据：learned input memory query曾把strict correct从matched fixed-query的135
提高到151，证明memory reader可能有用；之后135/131又证明“memory + 当前shared direct-B tail”没有稳定积累。
因此不能宣称memory失败，也不能原样重复这条完整架构。

## 7. 训练和RL边界

最终方法应有可从零复现的训练方式；允许开发阶段从强checkpoint做受控单变量实验，但成品不能依赖task轮换、挑
checkpoint或能力换手。

当前目标是生成LoRA后、环境交互前的性能。Writer可在train24使用监督、functional credit、privileged task expert
或on-policy reward，只要部署信息墙不被破坏；但生成LoRA后的task-local RL属于后续独立实验，当前不得把它混入
初始分数。

长期设想仍成立：生成LoRA应成为后续快速RL的良好起点，RL直接优化这套adaptation；但先把zero-interaction Writer
做强、做稳、证明视频因果性，再单独评价后续交互收益。

若纯监督长期无法提供policy-aligned credit，可以使用强化学习训练Writer。不能因为最初选择了监督就把RL排除，
也不能因为一次reward实验失败就否定reward credit一般。

## 8. 性能与科学有效性

长期目标继续追求strict paired correct严格超过`150/400`并越高越好。owner进一步明确：稳定的约145也可以是好
结果，前提是：

- 不是高波动训练中的单个winner checkpoint；
- 相邻single checkpoints保持接近性能、低churn和高success-set重合；
- 高breadth，多tasks在同一checkpoint共同积累；
- same-task不同teacher videos性能稳定；
- correct明显优于wrong、shuffled、reversed和no-video；
- 高分不是language-only shortcut、挑video、专家字典或checkpoint融合。

absolute closed-loop性能优先，但视频因果性和训练稳定性是方法资格，不是可替代absolute的另一个排行榜。

每个架构训练后必须充分分析per-task、per-suite、breadth、retained/gained/lost、churn、相邻checkpoint和最早失效
接口。好结果应适当多训练确认稳定性；明显坏结果不靠无限训练或小参数sweep挽救。

## 9. 如何使用过去的大量实验

过去实验必须形成连续因果认知，而不是几十个互不相关版本：

- v4证明视频与顺序能影响LoRA/action，但也暴露absolute-time/action-phase shortcut；
- v5/v5.2证明Semantic Core与Procedure分离能增强视频内容辨识，但Procedure可在fusion/compiler衰减；
- v6-fast达到143，说明特定architecture×recipe有效，同时后续checkpoint回落暴露能力不稳定；
- v7/v8/v10、Loom/Core/Prior等证明更漂亮的内部结构不自动提高closed-loop；
- rank/atom/lane/ownership实验否定把高rank、均匀能量或容量本身当目标；
- checkpoint union与大量相邻评测证明task drift首先是共同积累问题；
- K4/Dynamic-K证明多视频能降低same-task nuisance，却可能稳定错误task mean；
- task experts证明task-local LoRA有效，但不提供视频特异性、顺序或held shared support；
- reward/commitment路线逐步关闭carrier、gradient、native写出等局部断点，仍未解决held policy-useful shared direction；
- GOMQ证明memory有真实正贡献，但当前tail连续更新不保留支持。

任何负结果只淘汰实际测试的组合。rank14失败不否定所有rank reservation，Expert-Manifold失败不否定所有task-level
manifold监督，K4失败不否定few-shot，GOMQ不稳定也不否定memory token。

owner的局部建议是启发和约束，不应导致整套方案每次大改。新判断必须说明继承哪些已验证机制、针对哪个最早失败
接口，以及什么证据能快速否决。

## 10. 效率、GPU和协作

- 训练/评测最多使用单节点6张A40，但不是必须6张；有多少合适卡就用多少。
- 所有同时运行的EMBER作业合计最多占用8张GPU；并行多组训练、评测和诊断时也不得突破该全局上限。
- 低util、只占少量显存的他人GPU可在峰值余量足够时共驻；owner已与ycliu沟通过可共驻。
- 训练按K、帧数和历史cost平衡rank负载，不要求每rank进程数表面完全一致；第一个rank额外小控制进程应在不影响
  实际推进时顺手简化。
- 接受设备、batch和kernel导致的正常低位浮点差异；不为逐元素一致牺牲吞吐。
- 不增加防御性hash、重复forward、逐tensor扫描和无必要校验。
- 暂时不使用subagents。
- owner提出疑问时应独立判断、给出证据和完整pipeline，不机械顺从，也不因一点反馈推翻全部设计。
- goal应表达最终目标和原则，不把memory token、rank等细枝末节方法写成目标。

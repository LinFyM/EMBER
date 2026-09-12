# 过程获取：监督关系与下一项干预边界

2026-09-12。本文是当前候选关闭后的机制判断与可审阅提案，**不是active design或launch授权**。
目标仍是正确教学过程在唯一完整LoRA中的可重复闭环增量；当前状态见[progress](../progress.md)。

## 1. 改变哪项判断

Video Functional主方案相对匹配frame_set，在训练与validation两侧均未建立可重复有序增量。
两种额外读出、去蒸馏、teacher VL适配与去辅助比较也未兑现原预测。因此停止在该候选上继续
调整读出容量、辅助系数或冻结层位；下一解释首先面对过程获取，而不能默认已有好过程只待保持。
完整数字与外推边界见[findings§62–65](../findings.md)。

一个尚需区分的问题是：表示缺少可迁移的观察知识，还是现有本地监督没有把观察到的变化与其操作意义直接联系？
这两者可以共存，但一次同时换大编码器和动作监督，正负结果都难以改变选择。

## 2. 跨episode功能监督没有增加局部变化的标签

原主FM与辅助FM均按任务抽取独立教学视频V及执行样本(X,Y)。若exact language L在训练任务中识别任务，
理想化采样下有Y与V在给定X,L后条件独立。辅助执行query能改变优化路径，但没有新增
“这一段观察变化实际对应什么操作”的配对标签。更短梯度路径不等于更强的过程监督。

这不能推出普通FM不可能使用视频：有限数据、参数化和初始化会影响选择的解；v5.2正例说明
既有图文／动作先验与普通监督有时足以形成视频依赖。新监督若有效，支持的是它改善当前学习问题，
不能倒推它是所有视频学习的理论必要条件，也不能解释全部旧recipe差异。

## 3. 最近等价历史的实际差别

| 历史 | 实际观察与动作关系 | 对下一项的约束 |
| --- | --- | --- |
| Action-Forecast v4 | 当前／首帧／前帧产生visual-state，10步flow预测未来50步动作，再通过另一episode的functional loss学习 | 有7D预测不代表已根据发生的视觉变化反演实际动作；不能恢复其forecast作为现成强教师 |
| v5.2／v6 | 固定Gaussian probe、flow time=1的原生响应，经视频模块生成LoRA；没有真实7D动作预测监督接口 | 普通FM正证据保留；不把fixed-probe响应称作已校准的动作反演 |
| ECP Stage0／G2 | 真实动作标签来自另一episode，按归一化进度映射未来50步，再形成10×7 phase targets | 确有动作grounding与动态正证据；但没有同一观察变化与造成该变化的动作配对 |

最后一项已直接核对：`c1493a1:src/ember/privileged_actions.py:61–110`拒绝video/action demos交集，
并按video进度映射action episode；`src/ember/ecp/natural_program_data.py:215–226`分开取两组episodes。
Stage0 `e675b87:src/ember/ecp/stage0_data.py:126–127`同样分池。
v4真实积分见`f2163eea:src/ember/writer/action_forecast.py:302–323`；完整历史索引仍在research_history。
在所查索引、相关设计与源码范围内未发现下文局部反演链的执行证据，不宣称全部历史从未尝试。

## 4. 推荐继续推导：动作训练池中的局部观察—动作配对

所提新增监督只用于学习共享读取器，完整LoRA主FM仍按原跨episode合同：

```text
teacher pool 0–15：完整action-hidden RGB视频 + exact language
  → 共享native observer（完整H）及有序表示
  → Compiler/native D → 唯一38-target完整LoRA
  → action pool 16–41中的独立执行query → 主FM

action pool 16–41：同一动作episode的真实stride5 RGB片段
  → 同一个observer及有序表示 → 训练专用局部反演读出
  → 监督该片段内已经执行的动作；这些标签不进入表示／Compiler
```

部署只执行上行；不读取teacher动作、state、reward或terminal，不将反演头部署为另一policy，
不对教学视频生成的伪动作做task-local BC。共享参数从fresh开始与Compiler联合学习，不先冻结一个新教师再训练第二阶段。
这里借用的是观察到的转移与动作关联这一思路；经典[BCO](https://arxiv.org/abs/1805.01954)
另有探索与任务模仿训练合同，不能直接作为EMBER一次编译成功的外部证据。

若采样帧是t−5与t，目标应对应其间已执行动作，不能继续按另一episode的归一化进度配标签，
也不能把source从t预测的未来动作当作已经发生的真实动作。具体数据集中obs/action先后索引须在实现前核对。
stride5看不见间隔内全部运动，两端RGB也可能对应多种动作序列；因此不能承诺精确确定性反演。
候选采用局部动作分布的FM监督而非强迫唯一动作重建，动作仅在训练读出中加噪，不能流入共享视觉表示。
不把这段局部动作沿教师时间轴直接用于机器人执行；独立主FM仍承担跨初始化的策略编译。

预期新增作用是给有序表示提供“看见的转移如何与控制相关”的局部信用。它仍可能只学到机械臂位移／夹爪变化，
而不学目标物、接触结果或组合顺序；完整视觉语义与完整H均不能因此删除。
本提案也不证明跨具身能力：以LIBERO动作坐标监督存在具身特异性。

## 5. 为什么它与重新调辅助FM不同，以及怎样证伪

设C1为有序观察片段，C0为同一片段全部帧的无序集合，A为片段内实际动作；给定相同语言。
对平方风险，若C0是C1的确定性函数，则Bayes风险差为
`R*(C0) − R*(C1) = E || E[A|C1,L] − E[A|C0,L] ||² ≥ 0`。
这个恒等式只说明如何定义有序信息的潜在预测价值，不证明本数据严格大于零，也不证明有限模型能获得它。
若场景内容和语言足以恢复先后，局部有序增量仍可能为零。FM中的相同噪声／时间条件应纳入两边条件集合。

旧跨episode目标对当前片段并没有这种实际转移配对。新增的是标签对应关系，不能只比较loss数值后声称过程成立。
下一项若实施，必须区分下面三个结论，且在看分数前固定节点：

1. **局部获取仍弱：**在train24留出action episodes上，局部反演不能超越保留相同RGB内容的无序参照。
   停止把该配对监督当作有效过程教师，不扫反演头容量，不追加完整Writer轮次解释失败。
2. **局部获取较强但LoRA无收益：**反演的有序价值成立，主train闭环或validation不改善。
   停止把局部动作可解码性当作足够的策略表示；只有此时才有依据将该具体链的消费／编译列为下一竞争解释。
3. **LoRA形成收益：**主correct与same-task-other跨相邻节点优于匹配训练的全帧无序参照，并向validation迁移。
   才进入既定qualification及选点冻结后的sealed controls；辅助头、错误条件下降或单点峰值不能代替这些条件。

需有同图无局部目标的归因参照，并保持主FM任务／条件／query／RNG及曝光相同；新增局部样本和计算单独登记。
这里没有给旧pureFM结果冒充完整匹配参照，也未承诺先跑三条完整长训练。
先固定最小能裁决局部获取的证据范围与真实计算成本，不能以额外probe成绩降低闭环门槛。
不使用shuffled/reversed作训练、局部资格或架构选择；无序参照是单独训练的置换不变模型，保留全部真实帧。

## 6. 另一解释：额外冻结视频先验

官方[V-JEPA实现](https://github.com/facebookresearch/vjepa2/blob/main/src/models/vision_transformer.py)
的普通视频forward没有施加时间因果mask，并用3D tubelet先混合帧。
因此先编码整段视频、再给下游加causal mask，不能满足EMBER逐时点前缀依赖；
将其输出tokens作无序集合也没有删掉上游已经编码的顺序。
若采用，需限定每次读取的真实前缀，并重新定义能保留全部静态视觉信息的参照。

[V-JEPA 2.1](https://arxiv.org/html/2603.14482v1)提供图像／视频两种tokenizer与共享encoder，
有助于构造逐帧静态参照，但两路还改变tokenizer与模态条件，不能把差额全归于顺序。
这是另一种知识来源，不能与局部动作配对同时引入后再笼统解释涨跌。暂不下载模型、添加cache或启动该路线。

## 7. 实施前的实际合同问题

当前AGENTS要求“video与action query同task但跨episode采样，阻断逐帧轨迹复制”；
现有`WriterTrainingData`也将四类episode角色严格分开。上文主LoRA路径遵守它，
但新增辅助分支有意使用**同一action episode的RGB与其动作标签**。
这并非teacher部署输入泄漏，却改变了本项目此前实际采用的观察—动作监督关系，不能仅以train24已获授权静默引入。

需明确允许的例外范围：仅action pool 16–41中的局部共享表示监督，标签永不成为Writer条件；
teacher0–15及held46–49仍action-hidden，diagnostic42–45无梯度，validation/test无动作训练，
主LoRA功能查询继续跨episode。允许该范围后再登记active design、确定最小配对节点和profile；
不用本分析文件恢复已关闭的候选，也不以本提案声称goal已有根本性能进展。

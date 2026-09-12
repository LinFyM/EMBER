# 过程获取：监督关系与下一项干预边界

2026-09-12。本文保存候选关闭后的机制判断与提案形成过程；Owner随后明确授权核心科学精神内的修正。
已登记的唯一执行合同为[Local Action Grounded Writer](local_action_grounded_writer_design.md)，下文待确认措辞保留其形成时点。
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

部署只执行上行至LoRA生成的部分，再由冻结policy闭环；不读取teacher动作、state、reward或terminal，不将反演头部署为另一policy，
不对教学视频生成的伪动作做task-local BC。共享参数从fresh开始与Compiler联合学习，不先冻结一个新教师再训练第二阶段。
这里借用的是观察到的转移与动作关联这一思路；经典[BCO](https://arxiv.org/abs/1805.01954)
另有探索与任务模仿训练合同，不能直接作为EMBER一次编译成功的外部证据。

若采样帧是t−5与t，目标应对应其间已执行动作，不能继续按另一episode的归一化进度配标签，
也不能把source从t预测的未来动作当作已经发生的真实动作。§9已核实当前数据的post-action时序：
该区间取`actions[t−4:t+1]`，不取通常pre-action约定下的`actions[t−5:t]`。
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

## 8. 待确认期间的只读可行性核对

### 局部输入与参照不能泄漏先后

现有`src/ember/writer/video.py`提供同一个`encode()`入口，返回每视频`E[T,L,256]`；
`memory()`在ordered模式提供时间路由，在frame_set模式置零时间路由并将全部帧展平。
因此局部训练读出可以复用现有表示边界，不必增加第二套视频编码器或把执行query送回Compiler。

一个具体的最小候选是每个task额外取action池中的四个连续stride5采样帧，对应三个已发生的5步间隔；
读出预测共15×7维的局部动作分布。动作token的序列位置与flow time属于训练读出，
真实动作／加噪动作不能成为`encode()`输入。标签起止按§9的生产时序确定，不能根据数组同长度猜测对齐。

有序与无序两臂均让局部读出查询全部四帧的表示，拥有相同的15个动作输出位置。
**无序臂不得选取`E[-1]`、按目标间隔切出有身份的两端、传入原帧号或单独标记终点图像**；
否则即使上游attention取消RoPE，下游仍知道哪帧先后，不能作为无序资格参照。
无序臂的frame tokens允许保留单帧内容与空间位置，通过集合读取输出动作序列；
有序臂使用原有时间路由。输出动作位置本身不是帧的时间标签，不能通过实现细节将两者绑定。
这项约束针对参照有效性，不预先保证有序臂会更好。

### 有界工作量与尚未验证的范围

只读取现有manifest的train24长度元数据：teacher0–15共384条件，stride5帧数均值35.484375，
最短17、最长93；action16–41中没有不足16个原始采样点的episode。
每task增加四个真实观察帧，相当于当前平均teacher观察帧数的11.27%；
这只是native观察计算的输入量比例，**不是训练墙钟、GPU利用率或新增总成本的实测值**。
局部读出、反传、数据读取及额外共享表示计算仍需实际profile；这里没有据此指定GPU或启动段长。

只取完整局部间隔即可避免末端重复action填充，不需要修改主FM既有horizon/padding合同。
完整teacher视频仍逐stride5读到末帧；局部监督的四帧预算不能偷换为部署teacher截断。
本节未修改数据、sampler、源代码或训练配置，尚未进行局部模型拟合。

## 9. 当前数据的实际时间对应：post-action RGB

生产证据来自[LIBERO固定版本的create_dataset.py](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/6a71fae1724c1b84b62cfb4eeb96398c60fdc095/scripts/create_dataset.py)：
循环先执行`env.step(action)`，再保存返回的RGB、关节与末端观察；末尾对原states/actions使用相同valid_index。
它只略过开头五步，不按action值删除中间步，不创建next_obs。
因此保留行的关系是`states[i]=s_i, actions[i]=a_i, obs[i]=o_(i+1)`。

本地artifact由目标manifest固定为`yifengzhu-hf/LIBERO-datasets@f13aa24a3da8c43c7225569f28c562979fa0e35a`。
只读四suite的train tasks 0/12/20/34各demo0：前11行`obs/joint_states[i]`均与`states[i+1,1:8]`完全对应；
与同一行states的最大差分别为.03444218/.00660394/.03280896/.00503797。
四个demo均无next_obs，完整state时间轴每步约.05秒、未见内部缺口。
这项检查未读action数值或held数据，未重放模拟器，也不是全数据逐帧像素验证。
当前数值证据与生产代码一致，支持用于本提案的时间对应。

一般区间`obs[p] → obs[q]`应取`actions[p+1:q+1]`，共q−p步。
四帧`p,p+5,p+10,p+15`对应15个动作`actions[p+1:p+16]`。
obs[0]已经是actions[0]之后的图像，当前RGB序列不提供该首动作的pre-action端点。
Writer会追加真实末帧；例如N=98的95→97只有两步，不能视作五个动作或用padding伪造转移。
局部候选只选完整15步区间，完整teacher视频的末帧保留规则不变。

既有`src/ember/writer/data.py:317`采用obs[i]与actions[i:]同索引；这是当前主FM实际合同。
本核对未测一位时间差的闭环影响，不能把它认定为历史视频收益缺失的根因，也没有据此改主FM或source。
局部反演必须遵循已发生转移的真实标签关系；改变既有主FM监督是另一项科学变量，不能同时静默引入。
若以后改用会删除已执行控制步的数据，须保留原控制索引；retained行号相邻不再自动等于原时间相邻。

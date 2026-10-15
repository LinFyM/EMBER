# Source/r8参考之后：控制空间学习的具体准备

2026-10-15，主讨论在参考库补齐期间完成。本稿落实下一段的计算对象、学习信号和判读，
不是D已实现/启动的记录。当前唯一计算仍是[source/r8 design§8](../designs/source_policy_space_compiler_20261015.md#8-首窗科学消费后的参考库补齐窗口)。
完整参考库回报后，主讨论消费其质量与未收束项，再将实际采用的范围、节点和资源写入active design并派发；
无Owner例行许可停点，也不把本稿自动转成背景训练。

## 1. 首窗究竟改变了哪项判断

同source、完整rank8、合法50示范的独立任务学习，在7个已冻结任务的独立初态上得到103/126，source为26/126。
这提供了同容量参考函数确能学成的正例，减少了把高rank教师压入r8时的容量混淆。
task20/34仍较弱、task13/34后期能力回落、task32同分但成功集合互换，均不能由平均FM loss解释掉。
下一段应问“共享解码器是否保存这些真实控制函数”，而不是继续证明参考权重文件存在。

它仍没有证明低维参数空间。36个端点可以被64维代码轻易编号；对任何可逆坐标变化T，
`E'=T∘E, D'=D∘T⁻¹`可以保持端点功能，却改变欧氏距离及各向同性探索的意义。
所以端点重构、代码看起来分簇或latent维数小，都不等于学到了可迁移的控制方向。
这个不可辨认性不是新增模块能自动消除的问题；后续实际提案与新任务闭环必须约束结论。
另一个具体退化来自执行policy正常读取exact L：若每task只有一个参考函数，
一套几乎不依赖z的通用adapter可能靠policy内的语言/观测区分任务。
因此§4让同task的不同真实参数在完全相同query/noise上给出功能监督，不把跨task输出不同误当作代码被使用。

## 2. 两类参考资产承担不同学习职责

| 资产 | 用途 | 不能据此声称 |
|---|---|---|
| 每task按selection冻结的唯一最好参考 | 强控制端点、质量参照；其真实成功RGB可作为辅助正确教学 | 最好点代表所有初态最优，或一task多视频就是多task |
| 同一轨迹中其余完整checkpoint | 标签侧控制函数覆盖；学习不同完整LoRA实际怎样改变动作响应 | 它们都是合格教师，或训练进度天然等于某个技能坐标 |
| source identity | 精确的无改动起点与闭环参照 | 所有task都弱或source成功就是新expert成功 |
| 原始action-hidden教学 | 后续视频编译的主要真实分布 | 其实际producer是上述某个新参考 |

只取每task一个best会再次把学习数据压成固定共同E的端点表。
现有训练已经产生一条真实参数轨迹，可在不新增参考训练/环境运行的情况下使用其完整checkpoint作为函数标签。
首窗62个非零更新点，task0接续再加1，余28各至多10，故至多343个非零更新点；它们仍只来自36个任务。
较差checkpoint可以教D表示失败或未完成的控制，但其视频不能因此成为正确教学。
本稿不新增中间点的视频采集，不补造“多种正确操作方式”的对应；当前每task仍只有一个被选定的正确教学producer。

参考库的selected与report冻结后不再依据D或视频训练结果回调。
首窗独立report的结果不产生参考训练梯度；后续辅助教学只消费已登记成功RGB及真实producer身份，
控制器权重/任务ID/路径/score全部留在标签和调度侧，不进入Compiler条件。

## 3. E/D的实际算子

复用现有rank8 FactorLayout：643,584个实际因子标量，10,064个64维块，另有512个padding位置。
块地址是target、rank、A/B侧和真实坐标，padding只作mask；不输入任务名称、编号、视频或language。
所有E/D参数fresh；不加载旧G/MT、额外高rankadapter或旧Reader值。

首个实现保留现有结构的可复用部分：width256、4 heads、E的两层block→rank→target通信、
D的六层同类通信，代码z为64维。这些是具体初值，不是已测得的内在维数或新的架构贡献。
E将38个target表征汇聚后映射成一个64维向量；D的条件只有z，不能把完整参考因子旁路送给D。
D的块查询由固定source identity的布局/地址构成，z经共享投影成为条件memory。
标签侧E可读参考完整因子；视频编译及最终部署时E退出，不能把它变成held expert检索器。

令r为参考因子相对identity的归一化块，采用明确的恒等锚：

    z = E(r) = e(r) - e(0)
    D(z) = Lambda_id + S * [h(z) - h(0)]

`S`只由合法训练bank的因子变化统计得到，逐因子固定RMS单位，task等权；选定点与其它点的权重和§4一致。
每项设数值下限1e-6，仅防除零；它不是held normalization、每video私有尺度或动作单位重算。
原始动作/proprio normalization仍只用冻结source统计。
减去`e(0)`/`h(0)`的两个分支均参与本次真实梯度；不能把它们detach后仍宣称同一导数。
实际输出仍是全部38个rank8 A/B，部署为`W_source + B(z)A(z)`，没有第二adapter。

D输出头零初始化时，首个输出是identity，第一步主要先训练输出头，E不一定立即得到非零梯度。
这是可解释的初始化性质；不能要求每个张量第一步都非零，也不能在后续长期无有效更新时继续用它免责。
反过来，产生了非零参数也不证明形成有用控制。

## 4. 功能监督的完整消费者

每次先选task，再同时取它的选定参考与另一个完整非零点；后者在其余完整点中均匀轮转。
二者在完全相同的query和noise上分别提供目标，每项权重1/2，task等权。
若不存在其它非零点，则该task两部分都指向选定点且复用同一次计算；不因其它任务checkpoint多就增加其权重。
source选定点也是合法端点；由于恒等锚已保证其重构，零损失不能被改成额外任务权重。

query来自该task已有合法原始50 episode的真实RGB/proprio/exact L，episode/frame有确定的均匀流。
每个逻辑更新四个task、每task七个query，两个参数点共用这些query；覆盖顺序按36任务permutation循环，保持任务权重。
每query共享同一个原生50×32 Gaussian noise，两个reference及各自student各完整执行10步flow，取实际前5×7动作：

    a_ref = stopgrad F_source+Lambda_ref(x, epsilon)
    a_hat = F_source+D(E(Lambda_ref))(x, epsilon)
    L_control = mean_task mean_query mean_two_points mean_5x7 (a_hat - a_ref)^2

动作在现有固定source归一化坐标中比较。本段是纯函数保留目标，不用BA或因子MSE作额外主项，
也不把较差reference的动作称为正确操作标签。损失梯度经过student的真实10-flow、D和E；reference、source权重不反传。
它不是成功率梯度，更不需要对MuJoCo求导。
同一x/L/noise下，若两个参数的动作确实不同，不依赖z的D就不能同时精确拟合两项。
这是实际目标的约束，不另外添加“代码必须分开”的margin；若两点在这些query上功能相同，允许它们合并。
它减少一个可明确写出的退化，但仍允许记忆有限参数轨迹，也不证明这种差异就是跨task可迁移的操作知识。

这里有一个已核实的效率条件：当前38个execution targets全部在action suffix，prefix不能看到action tokens。
因此同一真实query的冻结prefix KV可在reference/student之间复用，不会漏掉本合同内的LoRA梯度；
依据为`writer/function_credit.py:NativeFlowPrediction.prepare`和`writer/practice/execution.py:NativeVelocity/native_actions`。
不能把这个结论扩大到以后修改prefix权重的架构，也不能复用含旧MT merged base的Runtime。
激活checkpoint重放须重新绑定这次完整生成因子；已有native_actions消费者支持显式因子重绑定，不另造flow近似。

本段query是离线示范的观测分布，不是假称已经覆盖所有失败状态。
当前参考报告保存了真实RGB及case信息，没有完整逐步proprio/hidden训练集；不能臆造这些配套标签。
离线分布上的小动作误差仍可能在闭环中累积；因此下面的真实功能保留读出不可由loss代替。

## 5. 一个有边界的首段学习与读出

准备方案为同一fresh E/D轨迹至1152个逻辑更新，576和1152保存完整状态；每task共128次访问，
共32,256个真实观测/noise query、64,512个reference/student功能配对。AdamW沿既有共享学习的明确起点：lr1e-4、betas(.9,.95)、
eps1e-8、weight_decay1e-4、global clip1、constant LR；这不是已验证最佳超参，不安排rank/LR/seed菜单。
物理任务/微批分配按实际显存与吞吐选择，不改变四task×七query×两个参数点的逻辑平均。
具体训练拓扑、首消费者wall/显存实测与存储上界在launch合同补齐，不能从参考FM单步吞吐直接推算10-flow反向。

两个节点都对每task的选定参考做一次`D(E(Lambda_selected))`，每task部署一套完整固定LoRA，
在原init0..15/root202610151上跑36×16个真实闭环；与已有选定参考的严格配对行比较，不新跑reference。
共最多1152个新环境episode；这是标签侧重构能力检查，teacher选择面板已有选择偏差，绝非EMBER或held资格。
本段不把teacher report18换成训练数据，不打开held/Test，也不额外采集新的正确教学。
source/原参考/重构的逐task与suite、保持/得失/churn和相邻success-set一起报告，不能只用整体平均数遮住能力丢失。

两个节点给出同一有界学习轨迹，而不是逐模块通过后自动晋级的许可证：

- 误差和真实控制都未学到：不能把当前D交给RL并期待更多环境次数自动补出控制空间。
- 功能误差下降而闭环明显损失：优先检查监督观测分布、关键决策误差与实际目标，不将其默认归工程bug。
- 大体保存已有控制：只说明已知函数在这些初态被保存，仍不证明代码邻域有用或新任务能迁移。

本段之后必须尽快接上原始视频→代码→完整LoRA的真实闭环学习；不能继续追加一串latent探针代替完整方法。
最终Writer的选点仍只认预注册strict paired400与相邻稳定性；本段内部576不能获得正式模型资格。
若某个节点大范围丢失参考能力，主讨论消费实际分布后改动假设；不以“再补参考任务”保护同一失败D。

## 6. 与过去失败的关系及尚未解决处

本段没有重新发明功能监督：旧LocalField/Functional/G-C等已经有真实native梯度，且G的BA-only正例曾优于添加native损失。
本次实际差别是raw source/r8同容量参考、标签侧E/D的函数空间学习、完整训练轨迹而非每task单一E端点。
这些差别提供可检验条件，不构成“功能loss这次必胜”的推导。
只使用轨迹节点仍可能记忆每task的一条优化曲线；它们不自动形成跨物体、场景、操作顺序可组合的基元。

初始视频映射仍要用原始跨episode合法动作FM；真实producer视频才允许使用其控制器的标签侧代码。
当前一个selected producer/task依然不足以证明同语言下不同操作方式被区分，不能把中间checkpoint数量偷换成这项证据。
共享dense阶段可以让视频映射与D共同学习并保留上述函数约束；D只在后续完整score/RL版本中按其推导冻结，
不是获得一次重构正例后永久保护D。
完整proposal/stop概率信用、实际H与独立新初态query的合同沿候选说明，尚需对应实现；本段没有替它们提供成功证据。

归根结底，参考函数提供“存在什么可执行控制”的标签；它们没有直接给出“新视频需要哪个函数、失败后如何修订”的答案。
下一段必须保持这两件事的区分，又在真实视频学习中接通；不能把第一件事完成当成EMBER的核心问题已解决。

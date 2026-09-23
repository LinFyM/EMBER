# 语义状态路径到条件控制参数

> 文档角色：封存设计合同。原文中的“当前／active／下一步”仅指当时阶段；当前授权与暂停状态以[progress](../../progress.md)为准。保留原科学定义和结果，不据此恢复执行。

2026-09-14。两臂fresh100与全部八面板已完成，按原资格关闭，完整裁决见§8。
§1–7保留原登记与实施合同；当前执行状态只看progress，本文件不恢复已关闭Writer。

## 1. 主假设与历史取舍

[工作理论](../analyses/temporal_control_compilation_theory.md)要求同时推断适用条件与操作作用。新的具体假设是：
把视频表达为**语言条件下的状态路径**，以显式的有向变化交互连接语义内容与完整LoRA，
会比只在时序attention后让全局Compiler自行发现顺序关系，更容易从跨episode FM中学出可迁移的过程作用。
这是学习与泛化假设，不是由数学信息量直接推出的行为结论。

本方案保留原生图文与动作知识、全部帧、full50、跨episode主FM、共同学习和完整自由A/B。
替换当前全局RX／B生成和参数标签目标，也替换旧C／Video Functional的时间token直接读出。
不增加局部动作头、参数重建、空间KL、梯度校准或新的privileged训练数据。

历史约束如下：

- v5.2普通FM的正例说明最终动作信用可以学会视频条件参数；同时配方交互与后段下降禁止将共享head或调制单独称为根因。
  本方案借用标准条件调制与小型共享head，实质新假设是**可学习语义状态上的路径运算**，不恢复coarse／H均值前端或旧初始化。
- C与Video Functional说明分开语义和过程、共同E、辅助FM、改VL或读出都不充分。本方案不再以模块分工或梯度可达作为进展；
  唯一学习目标直达实际LoRA，顺序作用具有明确的低阶运算，并用独立训练的全帧集合模型直接否证。
- Native Correction的A空间内误差与高系数代价不证明B是唯一根因；本方案不继续以固定坐标／单位或G拟合为学习目标。
  G诊断的实际闭环价值会记录，但精确参数拟合不是这里的必要中间任务。
- 双相机source动作响应有实际可读性，空间辅助的若干闭环收益也保留；它们不足以证明局部接触标签或V-JEPA先验必须保留。
  本方案以已有source图文／动作知识共同解释状态，去掉额外视频模型与38目标裸X重放，缩短真实学习路径和计算。
- 共享prior、DJNFR、Target-Owned和D绑定已有负例。自由家族head只是紧凑实现，不承担保持保证；保持由相邻行为集合裁决。

## 2. 全部输入怎样形成可用状态

每条视频固定stride5并保留最后一帧，使用官方双相机预处理。exact task token及两幅真实图像通过冻结source的原生prefix；
读取侧VL Meta与Action Meta均rank4、fresh、只在observer中安装。图文语义Z、prefix KV和full50 Action响应R属于同一参数版本。
Action输入使用既有公开固定noise probe与flow t=1，不读teacher state／action，也不伪造缺失图像prefix。
冻结vision／token embeddings可缓存；Meta后的Z/KV/R不能跨optimizer版本缓存。

共有width256、8 heads。保留每个有效语言token的轴L：

1. 每帧的上下文task token查询同帧全部有效图像patch，得到`z[t,l]`。它表示任务文字所关注的视觉状态，
   并不假称已经是对象标签、接触状态或正确分割。
2. exact language只作为查询，从所有`z[t,l]`读完整视频语义上下文`c[l]`；Value来自真实视频，允许双向解释。
3. 用同帧`z[t,l]`与全局`c[l]`形成查询，读取**全部T×50原生Action响应**。H位置编码只标识相对动作horizon，
   此处没有teacher时间位置编码，也不将H平均。K/V一次投影、查询按帧块执行，避免复制完整memory。
4. 将图像状态、语言上下文与上述Action read融合为`e[t,l]`。两层语言attention保留文字顺序；
   两层完整视频的无位置双向attention提供共同上下文。它们在有序／无序两臂完全相同且置换等变。

Action响应给出source对这些视觉状态的动作计算表征；跨帧读取允许比较不同状态的原生响应。
高层知识不是R本身，而是下节共同学习的状态坐标、语义上下文与变化先后关系。
没有额外冻结视频模型，没有裸X bank或第二次38目标输入采集。

## 3. 顺序如何成为实际消费的内容

共享映射`psi[t,l]=tanh(Linear(LayerNorm(e[t,l])))`产生32维有界状态坐标。
这些坐标没有预先指定成抓取／移动标签；FM只奖励对新episode行为有用的坐标。
对每个语言角色l，令`d_t=psi[t+1,l]-psi[t,l]`，`s_t=sum_{u<t}d_u`，形成

```text
delta = sum_t d_t                                  # 32
area  = 1/2 sum_t (s_t d_t^T - d_t s_t^T)           # 32 x 32, antisymmetric
p[l]  = concat(delta, upper_triangle(area))         # 528
```

这是分段线性状态路径的二阶log-signature。第一项表示状态净变化；第二项区分不同状态坐标的变化先后。
如果某些坐标学会表示“夹持已建立”和“目标已移动”，它可消费两种变化的顺序关联。
它不把相关性变成物理因果证明，也不将某个坐标／rank强行命名为技能。

这个有限运算对**同一已给定状态路径**的平移、停顿和共线细分有不变性；不保证神经状态在换初态、遮挡、
不同采样或不同视频下仍形成同一路径。二阶截断会丢失某些高阶顺序和回溯信息，32维也可能不够；
这些是预先接受的可失败边界，不能在结果不佳后自动加阶数／宽度扫描。
采用FP32累计这些小张量，避免长序列有符号和的无谓误差；其它计算保留正常BF16／TF32。

路径变换及可学习前置映射的背景分别见[Chevyrev与Kormilitzin](https://arxiv.org/abs/1603.03788)与
[Kidger等，NeurIPS 2019](https://papers.nips.cc/paper_files/paper/2019/file/d2cdf047a6674cef251d56544a3cf029-Paper.pdf)。
这里只采用一个二阶路径层；论文不提供EMBER迁移或闭环成功的证据。

匹配的`frame_set`模型具有独立fresh参数、相同初始化种子／全部帧／数据曝光／优化／编码器／heads。
它不接收时间位置、端点身份、相邻差分或有序局部视频先验。它用同一`psi`全帧经验二阶矩的上三角
`upper_triangle(sum_t psi_t psi_t^T / T, diagonal_included)`提供同样528维，包含32对角和496交互项；
它的语义attention还读取全部e，而非只读二阶矩。两臂只有这一个无参数聚合运算不同，不给无序臂零向量或少帧输入。
这是对学习后的语义坐标进行集合统计，不平均raw frames／native features／最终LoRA。

## 4. 从知识到唯一完整LoRA

先以语言查询完整e得到`c_final[l]`，把p经LayerNorm和528→256投影得到过程内容`q[l]`。
通过标准有界条件调制

```text
u[l] = c_final[l] * (1 + tanh(W_g q[l])) + W_v q[l]
```

形成同时含对象／状态条件与过程作用的知识。38个target×16个rank的learned queries只作attention查询，
从`u[l]`读出各自参数内容；查询本身不作为Value残差直接送入head。后接一个标准残差MLP。
每个形状家族的A／B head使用256→256→对应native维度MLP，同家族跨层／rank／task共享，target/rank身份来自查询。
这不是固定A的RX空间，也没有目标纠正幅度校准；两侧都可生成完整自由因子。

A采用一次固定合法Kaiming identity基础行加fresh零末层的生成残差；B生成head末层为零。
初始化更新为零，第一步主要学习B、随后真实FM传到A及视频读取器；不会要求所有张量或第一步每个模块梯度非零。
共享参数只提供跨条件复用机会；同task／跨task保持不由head共享、低秩或小范数保证。

最终固定`B_l A_l x_l(o)`消费机器人自己的实时观测诱导激活，表达条件行为而非教师时间播放。
语言中对象身份决定关注哪种状态，路径变化提供操作关联，FM训练它们怎样改变当前输入到动作的函数。
未见任务的可迁移性来自共享状态／关系坐标与控制译码；如果训练仅学成语言记忆或公共adapter，资格比较应拒绝该假设。
本轮只声称K1；同一调用返回一套38-target rank16 LoRA，闭环不再运行Writer。

## 5. 学习与有界行为预测

两臂均全部模块fresh、fresh AdamW和scheduler，唯一loss为真实完整LoRA的跨episode主FM。
source执行权重始终冻结，两个Meta只影响observer。每update四suite各一个task、每task一个K1条件及64 queries，
总256 queries，task等权；teacher从16–41采样，主动作episode也从16–41但排除该teacher。
42–45仅训练侧无梯度动作诊断；46–49仅训练侧跨视频诊断。固定train24以外无新梯度，validation/Test信息墙不变。
初始优化复用已验证数值配方：lr3e-5、AdamW(0.9,0.95)、eps1e-8、weight_decay1e-4、clip1、warmup8。
不存在“改优化器就会保持”的假设，不做LR、seed、rank、scale或loss扫描。

原有分块FM VJP与observer同版本重放复用，确保实际动作信用依次到完整A/B、语义调制、路径坐标、R/Z、两组Meta。
路径运算让有向状态交互具有短而显式的梯度通路；它不会补充FM没有提供的监督，也不保证梯度选择过程而非静态内容。
学习的两个可失败预测是：train held-video行为随曝光获得能力；且有序的能力增量在validation及相邻节点保持。

先做一次最长真实条件的完整forward、FM backward／replay与一次唯一LoRA推理profile，检查第二次更新关键梯度、
source冻结、所有帧/fullH、76因子finite、实际峰值和吞吐。CPU运算性质与现有checkpoint／信息墙回归按实际改动验证。
数学玩具路径只用于检查公式；不对真实教学视频提前运行shuffle/reverse等controls。

在看到闭环分数前，依据实际profile登记约一小时连续学习的两个等间隔节点，采用50或100的倍数。
两臂更新数与每task曝光完全匹配；节点配置确定前不启动正式训练。每节点均完成train96及validation400 correct，
训练动作诊断仅定位，不选点；正式候选只由两轮完整paired400与相邻结果选择。

初段资格要求：两个相邻节点的validation ordered−frame_set task-cluster95%CI下界均严格>0，
每节点至少两个suite有序净正；有序每节点相对固定配对source参照的净增CI下界也严格>0；
相邻有序成功数不下降、breadth不下降且非零suite不减少，不能以更差无序臂的退化冒充成功。
CI使用固定seed20260914、20,000次按8task配对bootstrap。预先登记并启动一次validation source400，
固定8task×50init与新候选相同的执行／RNG合同，供两臂与所有节点复用；旧source47只作历史参照。
参照须在资格裁决前完整完成，其独立计算可与已登记的学习并行，不因尚未收齐source rows阻塞无依赖的训练。
train source96复用刚完成且合同匹配的oracle诊断source17/96，若执行合同有实质差异则先解决配对。
报告全部task／suite、breadth、R/G/L、churn及Jaccard。
该初段资格不要求145，但也不等同完整goal完成。

达到初段资格后，补同task-other的完整matched400及内容更强的参照所需验证，再冻结single checkpoint；
最后才做预登记独立映射的wrong／no-video／shuffle／reverse内容和时序controls。不得提前用这些controls改方法或选点。
如果有学习但尚无资格，继续投入须有相邻行为与学习证据支持且先登记有界节点；
若正确行为没有获取、顺序优势不保持或仅无序退化，关闭这套“二阶语义路径＋此FM译码”的主假设组合，
不把失败自动变成调整路径维度、阶数、头部或辅助loss的下一轮。

## 6. 工程与生命周期

替换canonical `writer/video.py`与`native_factor.py`，后者由紧凑自由因子head取代。
复用attention、Meta安装、完整source读取、任务采样、分块主FM、checkpoint、dynamic queue与原视频调度。
删除活动的额外video_prior、native_inputs、spatial_supervision、correction_supervision及只服务这些已关闭路径的配置／测试，
不删除其原始label／checkpoint／正式证据。旧运行从clean pushed frozen Git及research_history复现。
native/runtime/supervised/training/materialization只更新必要接口与新schema，旧checkpoint明确不兼容、不能加载到新图。
不增加另一训练器、评测器、source copy或兼容fallback。

主写在独立`codex/semantic-path-writer`工作树，验证后集成main并推送；正式训练／物化使用clean pushed detached frozen树。
新大输出放data0，先按strg01的两个独立quota登记峰值，复用canonical模型／数据／环境。
每次GPU launch检查两个节点与EMBER总占卡边界；按实际可用设备和峰值选择，不等待凑卡或占位。

## 7. 实施核验与profile准入

核心Writer为9,464,704参数，其中encoder6,204,960、完整因子head2,187,584，两组Meta另计。
12项路径公式／帧集合／完整H及梯度检查、128项接线／重放／checkpoint／采样／物化回归通过；
checkpoint字段核验改为同一函数内的声明表后，70项相关物化回归再通过。
本次活动源码与测试合计净减558行，新增唯一59行factor owner、删除五个旧模块，没有新trainer或fallback。
结构检查无新增／增长的hard问题；原selection／物化编排保持同一职责，输入验证是短shape／mask合同，未为计数拆成零碎模块。
这些仅证明实现合同。f46e3303 clean pushed detached完成最长task38/demo36的105帧完整主FM／observer重放：
首尝试microbatch8通过、16在policy FM显存不足；保留原件后fresh以8／8重做，两次分别26.452／26.539秒，
峰值allocated34.274GiB、reserved35.455GiB。第二次A/B、路径、native投影及两组Meta均有非零梯度，source始终冻结。
无梯度一次Writer推理4.322秒返回38个target／76个finite因子，全部真实帧与full50保留，exit0。
profile初始化、optimizer与任何权重均不供正式训练复用；不因16的执行OOM改变科学图或数值配方。

看新闭环分数前，正式登记**50／100 updates**。三张A40、物理batch8、四suite各一K1条件和64 queries，
逻辑256 queries/update不变；每臂末尾累计400教学条件、25,600主queries，两臂独立fresh且曝光完全匹配。
用上述实测固定query代价与帧代价、既有同seed同采样合同的帧数及canonical三rank调度估算，100步约2687秒；
加初始化与0／50／100三次固定train24独立动作诊断，约一小时。该估计未使用旧闭环分数或结果选择节点。
50／100每个节点均按§5完成train96和validation400；资格、相邻保持与后续controls边界不变。
仅100步不能称充分训练或平台；后续投入按预登记的学习／行为证据分支裁决，不从loss或配置占位数字恢复200。

完整profile、吞吐估计及首尝试原件保存在`runs/analysis/semantic_path_writer_20260914/profile/`。
同一冻结代码的新source400在另外两张卡运行，初始launch、双节点现场与存储预算见同研究根；
训练须用包含本登记的clean pushed detached commit，并在启动前更新实际设备与独立quota。

## 8. 完整50／100结果与关闭裁决（2026-09-14）

两臂各fresh100／400教学条件／25,600主FM queries完成，所有曝光及诊断字段实际配对。四checkpoint、
八sealed bank和1,984闭环rows完整，source／环境／state-video／RNG、实际stride5帧及完整38-target身份核验通过。
运行代码5116deb0 clean pushed frozen；39个最终worker全部exit0，累计有效评测10,202.87秒。

| 节点 | train96 ordered / frame_set | validation400 ordered / frame_set |
| --- | --- | --- |
| 50 | 16 / 22 | 77 / 75 |
| 100 | 29 / 26 | 48 / 44 |

新source400=47，train配对source96=17。有序−frame_set的validation净率95%区间为[−1.25,+2.50]pp、
[−1.75,+5.00]pp，净正suite数2／1；有序−source区间[−2.25,+23.50]pp、[−19.25,+17.25]pp。
两个节点资格及相邻保持资格全部失败，没有selected checkpoint。

有序50／100的S/O/G/L为0/38/36/3→1/32/14/1，breadth5→6；无序0/35/38/2→1/34/7/2，breadth3→6。
有序对无序的R/G/L为64/13/11→27/21/17，churn24→38，J=.72727→.41538。
有序相邻R/G/L=30/18/47、churn65、J=.31579；无序23/21/52、churn73、J=.23958。
Object仍有相对source增量，Goal回退显著；100的有序优势只有Goal净+7，其余suite为0/−2/−1，不构成广泛过程收益。

训练有序16→29相邻净率CI[+5.21,+21.88]pp、breadth8→12；100相对source净+12，CI[+2.08,+22.92]pp。
但同节点有序−无序−6／+3，区间[−11.46,−1.04]／[−6.25,+13.54]pp，没有相邻训练过程优势。
固定动作FM有序.153285→.130330→.121139、无序.153285→.130311→.121899；50→100两组全部24task均改善。
这些是实在的获取诊断，不能抵消held行为回退或证明过程表示已充分；100也不被称为充分训练／平台。

有序50 validation的首次renderer在252行后报告EGL离屏framebuffer错误；原始失败、日志及已完成行保留，
原contract的canonical resume重建worker并完成其余148行。400行配对通过，累计1307.46秒包含两次有效运行；
最后九worker exit0，根因未定。该恢复没有改模型、frame／state／RNG映射或择优使用重复行。

**按§5关闭“二阶语义路径＋此联合FM译码”的主假设组合。** 不续训150/200，不扫路径阶数／宽度、heads、
rank、LR、seed、scale或辅助loss；没有资格，不补other、最终wrong／no-video／shuffle／reverse、Test或RL。
保留有限训练获取和Object作用，但不把更弱无序终点或50步峰值当成成功。负证据限制实际检验组合，
不否定所有过程表示、普通FM或一次LoRA编译；它同样不能作为继续保留本主假设投入的理由。

全量task／suite、source／相邻配对、学习及审计见
[READOUT](../../runs/analysis/semantic_path_writer_20260914/READOUT.md)、同根`paired_readout.json`与`bounded100_decision.json`。
整体goal未完成；后继先更新机制判断并区分当前实际失败，不自动登记另一完整Writer。

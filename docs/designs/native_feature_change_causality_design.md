# 冻结C0：逐帧视觉内容与原生动作计算特征的变化是否贡献执行能力

2026-09-26登记。状态与实际派发只看progress；规格为
`configs/native_feature_change_causality_v1/experiment_spec.json`。
这是特征层冻结诊断，不是训练、候选修复、视频必要性资格或恢复旧static实验。

## 1. 当前证据与需要区分的问题

固定C0补齐400后，Source/B/C0 correct/C0 other为57/121/120/132。
C0 correct−B为−1/400，task-cluster区间[-3.5,+2.75]pp；other−B为+11/400，[-.25,+6]pp。
Goal21的旧巨大缺口没有在当前C0复现；不能继续用“视频结构必然损害语言能力”推动改架构。
也不能把other的较高分数当selected，或把相对Source的收益归给视频，因为语言B也有同量级收益。
尚需识别的核心缺口是：未建立正确教学内容相对充分参照的有益增量。

先问一个更窄、可干预的问题：当前完整C0的执行是否受益于教学过程中变化的特征？
若有，来自任务条件视觉内容E、原生动作计算响应H，还是两者的配合？这能决定后续是否应保护已有的有效视频读取，
还是先验证/学习超出语言与初始场景的操作信息；不能先根据模块名称指定缺失了“语义”或“过程”。
不改变flow采样、动作通道、训练目标、数据池或架构参数；不再研究与EMBER条件映射无差异化联系的通用VLA性质。

历史约束必须同时保留：

- findings§128已有旧C600/旧O的首帧E/H重复诊断；静态末帧边界会留下P信号，但同预算训练没有证明修边界有效。
  因而本批不把首帧参照的P非零当bug，不修边界，也不把重复首帧解释成关闭所有时序计算。
- 旧A900的正确/错误Core-P交叉有局部正确P收益；这不证明当前C0也有，更不区分E与H的内容来源。
- 旧learned frame-set、V-JEPA重复图窗口对照仍读取全部真实帧；它们不是只见初始场景的对照。
  旧冻结首帧对照也不是充分训练的static baseline，不能将性能下降直接解释为正确过程优势。
- `video_functional_writer_design`的冻结表示读头能改善跨episode动作代理，但E同时包含语言/静态因素，且没有新LoRA闭环。
  `native_corrective_closed_loop_audit`的privileged纠正90/384对Source68/384证明受测LoRA接口能承载有益纠正，
  不证明合法RGB已经提取这些特征；相应RGB学生失败不能唯一归因提取或写入。

本批的新辨识是：固定当前C0、语言、原始视频和完整Compiler，直接交叉两个真实特征流中的逐帧变化，
在全八诊断任务上连接中间表示、实际生成参数、共同首轮动作和闭环。它不重复旧动作替换或新架构排名。

## 2. 实际特征与干预算子

原生读取给出q[L,256]、E[T,L,256]、H[T,50,1024]。
q是text-only task-span特征；E是看过图像的task-span投影加任务查询的patch内容；
H是真实图文prefix下、固定public noise/tau=1的Action Expert隐藏响应，不是真实teacher action或待执行动作。
Source及三组Meta和全部Writer权重固定为C0@630；只读取原条件合法RGB、语言、frame indices。

设e,h∈{0,1}，对每条视频独立构造：

```
E_t^(e) = E_0 + e (E_t - E_0)
H_t^(h) = H_0 + h (H_t - H_0)
I_t^(h) = interaction_projection(mean_over_native_horizon H_t^(h))
```

公式表达干预；实现e/h=1直接使用原tensor，不能用减再加引入无意义舍入差。
e/h=0只将该视频真实首帧特征广播到该视频原有有效位置；不跨视频、不对frames/features/LoRAs取平均。
H0仍保留全部50×1024，不能平均H后再广播horizon。尤其h=0时必须同步重算I，不能留下原逐帧H的信息旁路。
q、语言mask、T、valid_frames、真实原frame indices、末帧边界和public noise均不变；不改teacher RGB做假native forward。

随后完整复用既有计算：

```
Core = language_blocks(FrameRead(q,E^(e)))
P = Procedure(I^(h), H^(h), E^(e), original_positions)
slots = Compiler(Core,P)
Theta_eh = existing FactorHeads + original identity templates
```

每格始终是一套完整38-target rank16 LoRA，部署时一次加载，teacher不进入rollout。
00是“首帧内容＋原视频时钟”的冻结参照：仍有语言、初始图像/原生知识、视频长度及模型时序结构，**不是无视频或learned static baseline**。
11是原完整C0。10保留E的逐帧变化而固定H；01反之。
两个混合格可能偏离训练中的E/H联合分布，故交互或混合格退化不能单独宣布某特征流有害。
11与00的差也含视觉覆盖、状态变化和结果画面，不单独识别动作语义、时间顺序或操作因果知识。

## 3. 特征到函数的证据与竞争预测

对四格记录同一计算链的q、E/H来源、Core、P、centered P、Core/Procedure slots、gamma/beta、融合及输出slots。
所有“语义/操作信息”的解释须与实测数值分开，不用feature norm或可解码性冒充能力。
既有FrameRead逐头是均匀与选择读的混合；已读取最终C0的lambda约.04994–.05062。
由保存的E、原weights和实际frame attention在CPU拆出均值项与选择性修正，正常数值余项如实报告；
这不是95%信息被丢掉的证明，也不自动授权改gate。Procedure的centered Value均匀读取为零，同样不证明实际读取均匀。

四格实际首轮执行动作记A_eh(x)，从正式rollout原件读取，不另加function query。固定同一初态/观测与policy noise，报告：

```
Delta_E_at_H0 = A10 - A00       Delta_E_at_H1 = A11 - A01
Delta_H_at_E0 = A01 - A00       Delta_H_at_E1 = A11 - A10
Interaction  = A11 - A10 - A01 + A00
```

主动作口径为真实10-flow最终动作的前5步、归一化7D及实际scaled OSC；full50保存但不以未执行后缀代替主要作用。
相同分解用于闭环成功率；不把线性加法恒等式当根因。首轮后的环境状态已分叉，不将后续动作直接当同state函数比较。

- **初始内容已足够解释当前功能：** 00与11的函数/闭环差很小，与“当前策略没有兑现逐帧变化的额外价值”相容；
  不能凭小样本或区间跨0证明等效，也不能推出原视频不包含信息或所有架构都不需要视频。
- **E变化有可用贡献：** 10−00或11−01有正确闭环净收益，并有对应特征/动作作用；
  只说明该干预背景下E中的逐帧内容有用，不将E直接命名为操作语义。
- **H变化有可用贡献：** 01−00或11−10有净收益；原生动作计算响应参与有用编译获得当前证据，
  仍不能把H当真值动作或宣称超过learned语言/static参照。
- **联合匹配重要：** 11相对00及混合格有绝对收益、并有正交互，与E/H联合消费相容；
  混合格的分布偏移也是竞争解释，不能仅靠它选择新的融合结构。
- **变化进入函数但没有净收益：** 差异与敏感性存在而有益利用未识别；不能继续靠增大变化、拉低对照或调尺度求漂亮间隔。

只有完整链证据才决定下一项数据/训练/架构干预。正向冻结消融也不是方法资格；若要主张额外教学价值，
后续仍需充分学习的匹配语言/static参照及正确视频绝对增益。没有这种区分就不能从本批直接再训一个补丁。

## 4. 固定小面板与执行

实际执行者Sol `01a0cd90-ebb7-77a1-a20b-a858825d2f66`；主讨论 `01a0cd94-65da-7b22-8ca9-7ba35f454632`。
唯一checkpoint为C0@630，来源F=dca1b550；路径见规格。held8仍为0,1,14,15,20,21,36,38，state0..9。
按原seed20260911/50池正确teacher排列，每任务10条不同视频，四格逐state使用同一条；明确是有限80条件，非全400资格。
选择按固定task/state规则，不按刚看到的成功/失败挑case。所有条件均已作为诊断输入使用，不读取新的expert action/state/reward。

80次完整native条件编码，复用当次冻结特征生成四格共320套完整LoRA；不跨checkpoint复用激活。
320条新闭环，四格各80；11也重新闭环作本批主对照，旧C0/Source/B只作带来源的历史参照，不拼入主比较。
pilot为task0/state0和task36/state0的四格共8条，计入320；只验收接口/trace/配对/动作合同，不按成功分数决定放行。
正式采集全部compact、T+1连续对象/EEF/夹爪、实际动作、BDDL谓词；full固定四格×task0/14/21/38×state0，共16双相机。
不新增独立policy function query、训练、优化器、其它checkpoint、wrong/shuffle/reverse、官方Validation/Test或新teacher标签。

若需要工程native等价检查，最多两次额外完整Writer调用，仅用上述已登记条件且无env.step，计入GPU总账，
不能用额外闭环smoke或重跑已有效pilot来满足单提交形式。CPU合成验证覆盖逐视频独立广播、padding、h=0的I同步、
全H50、11复用原值，以及四格复用同一canonical Compiler/head；不要为诊断新建第二套Writer。

复用semantic_encoder、_pack_video_program、semantic_core、procedure、compile_encoded_task等owner接口。
优先只在本批薄执行面拼接既有模块与读取trace；若需保留源码改动，保持一个canonical计算面并按code-architecture-gate限制增长。
为当前诊断添加的范围准入/采集只允许本spec，不扩大旧实验范围。Source清除及完整38-target加载复用既有检查。
实现从最新main隔离；CPU/接口验收通过后合main并push，全部新bank/闭环来自唯一clean detached E。
当前F和旧原件只读；不改训练配置、不热改E。仅CPU调度/分析修正可另记来源，不改实际模型/rollout计算且保留有效行。

## 5. 统计、资源与停止

主要比较11−00；E/H的四条简单效应及一条交互为预定机制比较，全部报告，不挑高分格作为候选。
统一task-cluster bootstrap20000/seed20260926，共同draws；每task/state配对，单训练seed、小面板区间为描述性。
列per-task/suite、breadth、R/G/L/churn/Jaccard。相对同批11列得失；旧Source成功集合仅作历史保持分层，
不得冒称新Source配对部署收益。旧11的逐state差异单列，不隐去正常复评变异。

先核对退出、行数/唯一性/信息墙/配对和原件，再读成功分数。保留每condition输入特征索引、四格各接口特征、
320bank manifest、raw rows、trace/case索引、首轮函数分解、六条预定比较、总completion及实际完整GPU账。
无需新的长报告；执行者机械汇总后主动Queue主讨论并停止新增实验，科学解释由主讨论核对。

预算最多2.5 GPU-hours，覆盖加载、80(+至多2)次native读取、全部派生编译、320闭环与任何工程GPU过程。
依据刚完成fixed400吞吐估计正式约1.6 GPU-hours，实际ETA由接手后的命令/资源计划确认。
研究根`/data0/user/ymdai/ember_runs/native_feature_change_causality_20260926`新增峰值≤6GiB；
80份全H/输入特征约数百MiB、320bank约1.54GiB，其余接口特征/trace/full与临时产物计入；开发+formal代码≤768MiB。
建root及启动前由执行者查strg01独立quota/共享容量，估计实际最长视频与产物峰值；复用Source/data/checkpoint。
每launch双节点live准入，本批最多6物理GPU、单节点最多6，不等待凑卡，不占位，不启动未计时GPU步骤。
正常运行一次等待退出事件，非故障不轮询日志/cache。预算将越界、非finite/OOM、来源/信息墙或实际干预错误时停受影响步骤回报。
固定320完成后不自动追加400、节点/seed、学习或新干预；既有长期授权覆盖此明确范围，无需Owner再次批准。

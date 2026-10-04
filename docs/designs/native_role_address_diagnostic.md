# 同一对象换位下的原生角色方向读回

本项是冻结source特征的分析实验，不是新Writer、训练、策略干预或模型选择。
科学依据见[主分析§126](../analyses/feature_to_operator_mechanism_20260926.md#126-在同一换位反事实中区分原生内容对应与实际attention方向的可迁移性2026-10-04)；状态只看[progress](../../progress.md)。

## 1. 要改变的判断

当前900在task16原场景搬错orange；同一物理XY交换既有转搬butter，也有跨位置仍搬orange，不能统一归为固定位置或错身份。
两域实际A坐标辅助没有取得额外角色控制，且教学/自身端风险下降主要来自均值和槽位/flow波动，不能先假定角色关系已学准。
下一项仍研究这个对应问题：合法教学里的对象方向，进入机器人自身图像的实际读取算子后，是否保持指向同一实体。

§25曾推导原生prefix K构成Q参数梯度的方向，但因缺少跨情境度量、使用系数与功能utility证据而没有实施，也没有派发泛化方向探针。
本次重新考虑**一处必要前提**，依据是§122/125新增的同机器人状态、同语言、对象XY交换的完整反事实输入和角色响应失败；
这个配对能区分同实体对应与旧位置对应，比原同条件注意力差异更具体。§110已证明实际图像读取能改变部分选物，同时保留完整控制未转移的阴性。
不是因为坐标辅助失败就选Q架构，也不以一个更容易得到的特征阳性代替跨情境控制。

结果决定是否还值得把“直接利用原生视觉方向建立角色地址”保留为有实证前提的候选原理。
它不决定新架构或formal fresh：方向可迁移仍未回答视频怎样选出该方向、动态Value怎样指定操作、自身hidden何时调用及完整动作后果。
若该前提不成立，不追加alignment head、ROI、层、相机或方向构造扫描来保护它。

## 2. 唯一固定输入及信息墙

仅task16的原八init0..7，teacher依次47/33/28/46/32/1/24/43；与已完成两轮换位诊断相同。
读取每条teacher的原始第一帧双RGB，共8份；自身读取原布局及交换布局的首个、任何新动作前双RGB和机器人自身state8，共16份。
不选帧、不增加video/scene、重新render或step；没有训练、动作标签、held teacher state/pose/reward或Test。
八个base scenes重复形成两布局，24份prefix不是24个独立成功试验。

教学原件由原conditional_read_write_continuation900 spec的data_protocol解析task16 HDF5，只读
`data/demo_{id}/obs/{agentview_rgb,eye_in_hand_rgb}[0]`，按原流程180度rotate；teacher原像128，不能假定为render256。
自身原件使用`role_coordinate_credit_20261004/F/cases/F_task016_teacher{teacher}_init{init}_{layout}/initial_states/state_{init:03d}.npz`，
只取已rotate的`after_rgb`及`after_eef_pos/after_eef_quat/after_gripper_qpos`构造原生state8；不把body pose或其它sim状态输入模型。
读取current900原exact language，teacher state-free tokenization与自身官方state tokenization分开复用；normalization为原冻结source口径。

main在任何模型特征读回前，已查看全部八组真实双RGB并固定agentview的butter/orange矩形ROI。
registration及全部source/ROI核对图在canonical `.codex/tmp/prefix_role_read_20261004/`；执行者准入新root后整体移入其`analysis/visible_input/`，保留同一份。
ROI只来自可见RGB，没有从held几何或分割恢复；它提供分析中的对象名称，不是Writer、训练标签或部署输入。
只在agentview上定义读出，因为两物体在全部24张agentview均可见且能直接对应；模型仍读取原有双相机完整prefix。
不按特征/结果修ROI。矩形包含少量背景、物体约一至数个patch；阴性不能外推为source所有特征不含身份。

ROI采用rotate后归一到256×256的半开XYXY坐标，teacher实际像素坐标相应除2。原生图像resize224、16×16 patch网格。
每个patch权重等于它与矩形相交面积/矩形面积；总和1，不膨胀、挑中心token、重采样或学习投影。
固定框：teacher butter[182,120,200,138]、orange[104,94,120,132]；自身原butter[185,122,201,139]、orange[103,93,122,133]；
自身交换butter[107,118,120,133]、orange[181,96,204,138]。main已逐图核它们覆盖对应可见包装，不使用world坐标生成框。

## 3. 实际读取与算子

使用aligned1000冻结source及现有canonical加载/预处理，不加载新模型或下载权重。38处LoRA均在动作侧，不能改动图文prefix。
分别复现原teacher state-free双RGB+language prefix，以及自身官方双RGB+language/state prefix（第三相机的原生false mask保持）。
仅运行真实PaliGemma prefix；不造action query、不运行suffix/Writer/10步采样或环境。
prefix不能读suffix的原mask使这正是实际attention使用的K来源；按实际代码确认，而非用zero-image或缺prefix替代。

一次捕获全部18层、两个真实相机的K投影输出及实际cache中的RoPE后K，保留实际mask、有效position_ids和prefix有效长度。
自身第三masked图像不纳入分析。原生GQA的K只有一个KV head、由8个query heads共用，不把重复head计为独立证据。
teacher与自身state文本、有效长度及相机padding可能不同，不能沿用teacher query位置到自身。

以下每层分别计算，t为配对teacher，s为原/交换自身输入，r为两对象，j为图像patch。
先用固定ROI面积权重计算`kbar_r=sum_j w_rj k_j`及`kbar_tilde_r=sum_j w_rj ktilde_j`。
`k`来自原生k_proj、尚未RoPE；`ktilde`是实际cache key。后层图像位置混有语言/其它图像/自身state上下文，不能称纯局部物体特征。

**主读出是实际Q方向，不是脱离位置编码的余弦。** 令`Delta_tilde_k=kbar_tilde_butter-kbar_tilde_orange`，
`R_i`为该输入第i个真实50-slot action query位置使用的原生RoPE矩阵，d=256：

```
d_ti = R_ti^T Delta_tilde_k_t / sqrt(d)
b_t = mean_(i=0..49) d_ti
u_t = b_t / ||b_t||
m_si = u_t^T R_si^T Delta_tilde_k_s / sqrt(d)
```

位置只由官方prefix mask和suffix50规则计算；不为取得R而执行fake suffix。若b恰为零，如实记录退化，不以epsilon或新方向替换。
对任一head的pre-RoPE Q输出施加`delta q_i=c_i u_t`时，在当前输入处精确有
`delta(mean_ROI logit_butter - mean_ROI logit_orange)=c_i m_si`。
这是区域平均logit对比（亦为区域加权平均log-prob之差），不是区域总attention质量的log-odds，不涉及实际softmax后的Value或动作变化。
正使用系数只是这个几何量的条件；实际LoRA的`A h`和下游utility没有由本分析测定。

保存全部8×2×18×50的m、teacher自身参照`mean_i d_ti^T u_t=||b_t||`、query间方向变化，
以及m对应单位方向余弦（分母非零才定义）。按每个配对/层报告原与交换的正负和量级，固定first5/full50均值均保留。
汇总18层全分布和逐层结果，不挑最好层/slot/head、设合格阈值、调正负号或按结果选方法。

**唯一辅助分解**是同一捕获K在RoPE前的内容对应：
`cos(kbar_butter_t-kbar_orange_t, kbar_butter_s-kbar_orange_s)`。
它区分已存在的key内容对应与实际位置编码后的Q方向；没有第二模型、学习映射或额外forward。
它不是可部署替代算子，更不能把其阳性移作主读出的阳性。全部原始K和ROI权重保留供main核原件。

## 4. 结果分支与停止

- 原布局正而换位转负，支持本具体教学方向依赖旧位置/上下文，降低直接复制原生角色方向作为修复的支持。
- RoPE前两布局均正，但实际Q方向在交换后转负/不稳，说明这个简单参数方向没有保留内容对应；不能宣称图像完全缺少身份，也不自动改RoPE。
- 两种布局实际方向均稳定指向同实体，只支持这个首帧、相机和两对象对比下存在可复用方向；不能宣称原LoRA使用了它、视频理解已解决或完整控制会提升。
- 两域/各层混合或微弱时保留完整分布，结论为条件性/未区分，不用少数层的高点拼成资格。

同一对象方向可取与不可取都须回到§122/125的实际错对象、旧位置及第三对象反例解释；本项没有重新测量策略行为。
只完成这一批24个prefix，停止；不追加其它frame/对象/相机/ROI/层/seed或训练、LoRA构造、环境干预、400、controls/Test/RL。
尤其不以本局部阳性恢复§25旧完整配方或已经关闭的grounding/辅助蒸馏；后继由main结合实际控制关系独立判断。

## 5. 执行与交付

main提交后交既有实验session独占canonical tracked/Git。它在隔离分支实现一个任务专用捕获/分析入口，复用原生prefix owner，
不复制policy/evaluator、添加常驻fallback或通用探针框架。整批来自clean pushed detached冻结树；完毕后退役入口，Git/frozen保留。
原root统一`/data1/user/ymdai/ember_runs/native_role_address_20261004/`。预计含工程45–90分钟，硬限完整.5GPUh/新增峰4GiB；
24份K预计不足.3GiB，另计frozen工程树、PNG/统计、失败和I/O，准入时以实际估计更新但不静默超限。
所有新增data1；launch前live核strg01 data1独立quota、相关实占/共享空间及双节点GPU，遵守当前8/6卡规则。
单个source常驻一次、teacher8和自身16按实际prefix形状批处理，用有收益的batch；不为填显存新增例或重复profiling。
一次退出事件等待；全部加载/失败/捕获/写盘/退出时间计入成本，不日志轮询或阶段selfQueue。

必要检查仅为：输入身份与字段墙、真实双RGB/尺寸/rotate及第三mask、state/tokenizer来源、实际18层K形状/RoPE位置、
固定ROI权重和metric公式、finite与捕获完整。普通BF16/FP32差异不追逐逐bit，不添加hash或测试框架。
CPU main准备曾因未设PYTHONPATH在读取数据前import失败，已修正后完成；保留于ROI registration，不计为模型或科学失败。
交付source/代码/精确命令、24输入provenance与ROI原件、全部raw K/位置、完整正反统计、completion、GPU/存储/退出账本。
一次整批有来源回报后交回canonical/Git并停止；不自动选架构或后继，main负责科学消费。

## 6. 固定执行交付事实（2026-10-04，科学裁决仍由main消费）

实际冻结3221801e、source训练b8ea00e9/原aligned1000，24/24真实prefix、288行及全部8×2×18×50齐备。
teacher有效长度530/query530–579；官方自身有效561/query561–610，第三相机false mask保留。全部18层、双真实相机pre/post K、真实mask/position/input/ROI均存原件；没有suffix/Writer/flow/学习/环境或新行为。
原first5/full50平均m为1.054743/.991081，换位−.252052/−.249631；配对层均值原全144正，换位first5为44正/100负、full50为45正/99负。
pre-RoPE cosine原均值.932228/144正，换位−.194058/47正97负，105条不利行及全部层/槽/正例保留。
它们只刻画固定首帧、框和两实体的区域平均logit方向；没有改变§4适用边界或测量新控制结果。
原件根`/data1/user/ymdai/ember_runs/native_role_address_20261004/`：`raw/`、`readback.json`、`analysis/all_readout.npz`、
`analysis/aggregate.json`、`analysis/pair_layer_rows.csv`、`analysis/README.md`及事前`analysis/visible_input/`。

第一次486fb2b6 frozen因任务禁用canonical BF16 autocast，在首层SDPA bias dtype退出1/无完整K；
恢复既有默认context后新push/frozen读取成功，未改source/mask/数学/样本。失败源码/log/费用/contract保留，非科学阴性。
整批含两次source加载/失败/I-O/CPU读回/退出50.569134秒/.014046981723GPUh，硬.5；实际阶段新增1.024082GiB，估计1.5/硬4。
合法teacher8/self16全组打包1.631518/2.654550秒，成功reserved峰12.109375GiB；无额外profile/样本，失败显存峰未观测，不声称全批VRAM精确峰或相对提速。
两PID退出、GPU归零证据及完整账本保留；两临时文件交付时退役，仅保留Git/frozen/原件。停止线全部保持。

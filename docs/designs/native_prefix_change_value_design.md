# 用同一原生动作查询读取教学变化：Value来源的有界共同学习

**状态：已撤回，不再授权执行。** 2026-10-02 Owner纠正后，主讨论确认本文的计算/信用推导
未建立具体task失败、方法缺陷证据与干预预测之间的联系，block17选取也仅有启发，因而撤回剩余计算。
正文保留原选择及合同的历史；实际收束见progress，决策错误见findings§276，长期禁止事项见Owner要求§3。
本批未完成原定科学检验，没有性能阴性结论，不能沿此合同恢复或用后来分数补写选型理由。

2026-10-02。Owner授权固定数据内自主推进，并要求先推导再决定。本文登记一个fresh学习批次，
不是对S有害、原H没有语义或新方法必然有效的确认。只有progress登记及实际派发后的范围生效。

## 1. 本次要改变的完整判断

问题仍是在相对稳定的情况下大幅超过强MT。T的有效证据支持保留实际A同时用于教学地址和执行读取的约束；
它不是不可撤换的架构。原T450/900/2340为122/148/161，成熟收益主要集中在三项强任务，困难操作覆盖仍不足。
Context900151、条件读写900140及支持扩展630的156对154，均没有形成广泛的完整优势。
这些结果降低只在原H上增加解释容量、条件读取自由度或相同功能监督曝光的优先级，
但不证明H必然丢失了某类语义，也不能把未见任务失败全部归罪于Value。

本轮冻结重表达保留13/32并新增1条，说明所测当前控制不必依赖最终S读取；主要失败没有修复，S仍参与编译。
它不授权直接去S训练。新的学习对象是：在已产生有效控制的T读写联系中，
**补充同一个动作查询对真实相邻图文prefix变化的原生attention响应，能否使写入内容更容易从现有监督中学成。**
不用参数冗余或小重构误差替代这个学习检验。

这也是对先前额外图文Value候选的具体收窄。固定调用核只限制何处调用，不能推出改进写入内容必然无用；
反过来，多一个输入也不自动产生更好的内容。本文明确实际读取算子、接入位置、信用及停止线，
不启动通用raw-Z解释器、辅助动作头、native动作监督或条件S。

## 2. 已核对的原生计算与新特征

源码依据：当前`src/ember/operator_writer/native.py`、`model.py`，原T实现`e2afbfd7`；
安装的`lerobot/policies/pi05/modeling_pi05.py:compute_layer_complete`先分别产生prefix/suffix的Q/K/V，
拼接后使用实际position_ids做RoPE，执行8个Q heads、1个KV head、head_dim256的GQA attention，
再由各自o_proj投回prefix2048/suffix1024。新特征只取最后一个Action Expert block（索引17）的这次真实读取。
这里选择最后一层是为了直接使用最深的现有图文表示和1024维原生动作输出坐标，不是层位扫描的结果。

保持原T输入和公共参数beta：exact language、完整双相机RGB、stride5及真实末帧；
每帧相同公开probe1729、50×32、tau=1，teacher无state。source所有基础权重冻结，beta仍由完整FM共同学习。
一次完整公共native取得原38处X、最终H，并在上述真实attention处保留以下必要小字段：

- 已按原生position_ids旋转的suffix Q_t，全部50位置/8 heads；
- 同层已旋转的prefix K^P_t、原生prefix V^P_t，包含两路真实图像和有效语言/原padding；
- 同层suffix K^S_t、V^S_t；
- 原生suffix attention经Action Expert o_proj、在residual/MLP之前的1024维输出u_t。

不保存各层巨大attention概率图，不重新运行policy获得这些字段，不改原native前向。
prefix没有任何source参数梯度，也不依赖action侧beta；suffix Q/K/V及原H/X对beta的真实梯度须保留。
字段只在本次编译内流式使用/重放，跨帧chunk边界仍须使用真正的相邻帧，不能丢掉边界transition。

对相邻帧t、t+1，用同一实际Q_t和当前suffix K/V，只把整个图文prefix替换为下一真实帧的prefix：

```
u_t = O_attn concat_heads Attn(Q_t, [K^P_t, K^S_t], [V^P_t, V^S_t]),
u_cf_t = O_attn concat_heads Attn(Q_t, [K^P_(t+1), K^S_t], [V^P_(t+1), V^S_t]),
dP_t = RMS(u_cf_t) - RMS(u_t).
```

Attn使用原生scaling、GQA、mask、position_ids与dropout0；O_attn是同一冻结Action Expert o_proj，包含其原有bias。
下一帧prefix来自同一exact language/双相机/有效长度；语言字符串不变，但其contextual K/V可随图像变化，不能错误冻结这部分。
Q和prefix的RoPE坐标使用原生token位置，不把视频帧编号当RoPE位置。当前suffix始终不换成下一帧的suffix。
RMS采用原T的无参数1024维RMS、eps1e-6；运算保留50×1024，不先平均horizon、heads、frames或raw features。

这是Writer内部对已读取真实native字段作的一次固定attention计算。没有teacher action/state、额外图像模型、
环境交互、优化器或部署循环；也不是另跑一个缺失真实prefix的伪policy。完整策略推理和官方十步采样不改。
u_cf不是新帧的完整动作网络输出，更不是教师动作或真实物理反事实。

## 3. 它比原H差多给了什么，又没有给什么

对一个head，令当前attention输出r=sum_j alpha_j v_j，固定Q、当前suffix和mask，只改变prefix。
有限差可精确分为

`r_cf-r = sum_j alpha_cf_j (v_cf_j-v_j) + sum_j (alpha_cf_j-alpha_j) v_j`。

第一项读取内容变化，第二项读取权重变化；suffix的第一项为零，但其权重可因prefix竞争变化而改变。
相应局部微分为

`dr = sum_(j in prefix) alpha_j dv_j
    + sum_(j in prefix) alpha_j (v_j-r) (q^T dk_j)/sqrt(256)`。

原生RoPE已包含在q/k中。跨heads及o_proj后才得到1024维u，不能把单head式直接当最终动作。
这里没有dq项；原完整H_(t+1)-H_t同时包含各层query、suffix、门控、MLP及最后归一化的改变。
原H差的全部变化同样来自合法RGB变化，因为probe固定；不能将其query改变贬为新增随机噪声。

新增的是**在一个固定原生动作读取状态下，实际图文变化产生的输出响应**，不是新数据。
它可能保留最终H压缩/共同响应中不易被旧Value head提取的视觉变化，也可能突出与控制无关的背景变化。
若原生query从未关注被操作物体，或prefix已经丢失所需区分，本算子没有保证补回它。
不得由向量维数、attention位置、非零差值或图像变化，宣称它已经识别物体角色、接触或操作知识。

与直接复制原生key不同，本读数同时使用当前query、竞争keys及native values/o_proj，
得到的是该查询的真实读取作用；但它仍没有监督给出“这个变化应产生哪种LoRA控制”。后者仍由下一节的真实FM学习。

## 4. 唯一完整Writer及实际信用

保留原T所有公共A/B0、P/C/D/O、原H差、矩阵递推和输出。每个target只增加一个无bias线性E_m：1024→256：

```
K_tm = normalize(A_m X_tm),
g_tm = GELU(P_m K_tm + C_m hbar_t),
V_tm = O_m [g_tm * (D_m (hbar_(t+1)-hbar_t) + E_m dP_t)],
M_(t+1)m = M_tm + (V_tm - M_tm K_tm) K_tm^T / 50,
A_final_m = A_m,                 B_final_m = B0_m + M_Tm.
```

公式按列向量排列，hbar、dP及K均保留全部50列。E为零严格包含原T的图；没有删除原动态Value来削弱参照。
先完成原T全部P/C/D/O及公共参数初始化，再在独立RNG作用域添加E并精确置零，避免改变原T初始化序列。
新增参数为38×1024×256=9,961,472；不因少参数声称有效，也不声称与T是等参数比较。
所有参数fresh、fresh optimizer/scheduler，source冻结；无T/MT/条件900权重继承、分段冻结或辅助loss。
O和B0原合法零初始化保持。最初上游信用延迟按实际复合图理解，不要求第一步所有模块梯度非零。

训练仍只用原同task跨episode完整50×7 FM，单套LoRA对query自己的RGB/state/noise/time预测。
若G_tm为经后续递推传回M_(t+1)m的余切，U_tm=O_m^T G_tm K_tm/50，则

`dL/dE_m = sum_t (g_tm * U_tm) dP_t^T`。

这给E的目标是实际query误差所需要的写入方向，标签不进入教学表示。dP的信用继续经过两次attention之差
回到当前Q_t及suffix K/V，再到公共beta；原H、X、A、B0及旧Value路径的所有信用继续存在。
prefix与O_attn是冻结source知识，不为给它们制造梯度而增设Meta副本。E/Writer不在闭环中运行。

还可精确指出表达与学习的区别。固定公共读取、keys和旧P/C/O，仅改变E时，

`Delta M A h = sum_t O_m [g_tm * (E_m dP_t)] omega_tm(h)`，

其中omega就是原T由K、后续覆盖矩阵和实际A h组成的调用核。新增内容没有自动修复这个核，
也不扩大同一A行空间或O列空间；本假设关注视频到有效系数的获取，不宣称解决输出rank容量。
在固定执行Jacobian的局部平方风险中，将这项作用写成F_q e，残差为r_q，
`R(e)-R(0)=e^T H e-2 b^T e`，H=E[F_q^T F_q]、b=E[F_q^T r_q]。
理想局部可改善量为b^T H^+ b；这要求新读取与有用功能残差实际相关，当前没有测定b，更不保证共同学习或held闭环收益。
公式说明必须由什么信用学成，不以可达性充当已获得控制知识。

## 5. 最近似完整历史与本轮承担的风险

| 近邻 | 保留的正负证据 | 本次实际差别及边界 |
| --- | --- | --- |
| 原T与Context | T有完整闭环及相邻稳定性正例；Context900151未强于MT153 | 保留T全部路径，新增原生图文读取响应；不再只重组最终H，不声称T地址已经充分 |
| 条件读写/S与冻结重表达 | seen获取增加、held未保持；重表达有限控制保留 | 本轮不继承S或深c/d，不由冻结保留推断去S共同学习；以独立T为有效参照 |
| SemanticPath | 原生图文/Action共同学习、路径统计、自由A/B；50→100为77→48 | 不恢复语义路径/自由参数头；新字段进入实际A地址下的Value，同query原生attention替换语言patch pooling；不能把该旧阴性抹去 |
| Task-Grounded Visual-Value、VisibleObject | 真实patch差分或实际注意力监督均未形成所需完整能力 | 原生动作query/竞争/Value/o_proj共同定义响应，并直接接T；无新空间标签。输入更具体仍可能失败 |
| LocalActionGrounded、NativeCorrection/LocalField | 局部动作、真cotangent及实际参数消费者已实施，未广泛修复 | 无局部动作目标、真纠正标签或部署VJP，不把旧方法说成信用未接通 |
| 原生key提案（机制§25） | 数学上是Q梯度方向，但合法可迁移utility未建立，未实施 | 不复制key或预测attention cotangent；实际响应差只作可学习Value内容，仍须完整功能裁决 |
| SEOD/GOMQ、flow路径提案 | 已有真实十步反传；路径提案撤回，非实验阴性 | 不重开端点/latent监督，以单步/多步差异为新根因的路线不采纳 |

本轮比较同时增加9.96M参数和一条具体输入/信用路径，不把与T的分数差唯一归因于原H信息丢失。
关键主张仅是这个完整、受限的Value获取机制是否值得继续。成熟T、MT及当前条件图高点均保留，不只胜一个失败版本。
若只更容易拟合训练、没有迁移，不能接着改层位、归一化、E强度或添加inverse loss保护同一假设。

## 6. 固定fresh学习与读出合同

唯一新mode为`native_prefix_change`。原`learning_spec.json`及`context_value_fresh_spec.json`提供原T数值/事件参考，
历史配置的world2、峰2卡、旧输出根或暂停文字不成为本批约束。执行者在同一canonical owner内形成唯一机器spec。

- 原source aligned1000、冻结normalization/tokenizer、38-target rank128/alpha128、identity_seed20260721、module_seed7保持。
- 原coverage36显式24+12 task集合、全部每task50视频、跨episode28query、offset1、每步4task/112query与等权保持。
  沿原seed20260928的task/teacher/query事件及seed7 flow流；不引入旧D71额外任务、新episode、privileged标签或新数据源。
- AdamW lr3e-4、betas(.9,.95)、eps1e-8、weight_decay1e-4、clip1、warmup150、decay_updates1200、floor1e-5保持。
  不因本批只到450压缩LR时钟，不扫rank/seed/层位/步长或dtype。
- **fresh450**，累计1,800个条件、50,400完整query、每task50次视频访问。完整ECP在90/180/270/360/450。
  **仅270和450各一次correct400，450另作原seen144**。270结果不改变剩余训练/评测范围；360仅恢复，不选点。
- formal K1每task整轮50视频各一次，沿canonical video_schedule seed7、同一固定state-video映射及既存scene。
  validation为[3,6,11,16,23,26,31,39]×states0–49；seen复用原36×states32–35，
  其teacher按`seen_task_scope.json`的video_schedule_seed20260928从原0–49池固定映射、每task四条不同视频；
  不擅自换成demo46–49，不称新teacher留出或整轮50无放回。
  validation只读复用`/data1/user/ymdai/ember_runs/demonstration_transfer_learning_20260927/scenes`；
  seen只读复用`/data1/user/ymdai/ember_runs/operator_seen_task_diagnosis_20260929/attempts/scene_canonical144/scenes`，
  不使用已被canonicalization替代的旧seen场景，不重新创建场景矩阵。
- 三面板共**944条新环境原行**；validation各8 full/392 compact，seen36 full/108 compact，共52 full。
  全部保存实际动作、goal、continuous EEF/object/gripper和compact；完整比较使用现有原T270/T450、T900/2340、Context450/900、条件450/900及MT153原行，不重跑有效参照。
- 复用官方render256/model224、双相机rotate、own8state/7action、10flow、执行前5/replan、dummy10、suite horizon和成功即停。
  dynamic queue、long-first、persistent workers保持。无held action梯度、Test、其它视频controls、RL或checkpoint融合。

不同年龄的T900/2340只作强能力参照，不能混称同龄因果比较。主要同龄比较为新270/450与原T对应点；
旧Context/条件图提供已知替代路径及高点，不能省略。报告per-task/per-suite/breadth、R/G/L/churn/Jaccard和相邻success-set。
差额区间沿原8任务成簇bootstrap、seed2026092810/10,000重采样；seen另分target24/support12，不能用support的易例掩盖target。

不新开A28/attention/投影诊断矩阵。工程profile只检查真实机制、完整梯度/identity/finite/最长视频吞吐和ECP消费者；
第一轮对比不需要逐tensor相等、source hash、held动作标签或额外闭环smoke。
可随已有物化被动保存预登记8条train条件（原seen面板中task0/12/20/32、各states32/33对应两视频）的紧凑统计：
dP及原H差的逐transition/horizon范数、E dP与D dH范数、最终两项M的线性分解范数。
它们不增加forward、不保存巨大activation、不运行E-off policy，也不参与选点或充当视频因果证据。

## 7. 裁决与停止线

这批先检验共同学习后的完整能力，不设必须先通过某个特征/梯度Gate的课程。

1. 若450不优于同龄T且缺乏跨任务获取，或270→450出现广泛能力坍缩，则关闭本次“用原生prefix变化补Value”的实现假设；
   不以训练loss、feature差或局部任务正例自动续训，不改一项归一化/层位/强度接着追。
2. 若seen上升而held无实际收益，降低它促进可迁移内容获取的支持；不把它说成原H已被证明缺信息、只差更久学习。
3. 若相对同龄T有分布较广的held净收益、相邻没有大范围能力丢失，则提高本机制的优先级，
   但必须同报MT153及成熟T161的差距、普通能力交换和不确定性；是否有必要再投入成熟阶段由完整结果另行裁决。
4. 单个高点或只胜T450不构成selected。当前目标仍是大幅超过强MT并具有相对保持；
   本批不自动选270/450、不授权900/更长训练或视频controls。选择及必要视频增量需要后继明确登记。

有界阴性不证明所有原生图文信息、条件A或视频学习不可能；它约束的是这里继承T地址、当前query原生读取和full-FM学习的组合。
冻前模型损伤、未训练模块、数学可达和实际fresh结果分别解释，不把“没有保证”当作实验阴性。

## 8. 实施、预算与交付

唯一实验session `01a0fabb-f7a0-7100-93d8-6a0f66055553`负责代码、检查、运行、资源、Git和冻结交付。
主讨论负责本文的科学选择与结果消费。派发时移交canonical tracked/Git窗口，直到整批完成/实质阻碍释放；不并发写同一树。
复用native/model/credit/data/run/bank/evaluator及完整ECP；新增字段/Value接入由现有owner承担，
不复制policy、trainer、逐层Composer或第二运行面。源码结构与生命周期由执行者按实际改动把关。
保留原T等sealed checkpoint读取以复现历史；本候选是唯一新活动训练mode，关闭后退役其专属训练入口，原件与Git保留。

允许在本合同内自主完成实现、实际消费者的针对性检查和一次最长视频两步profile，然后fresh正式运行；
profile权重/optimizer不继承。合理工程修复按AGENTS闭环，冻结代码不原地改；普通阶段无需主讨论重复工程验收。
若必须改变上述数学、数据/标签、损失、选择或突破预算，报告具体科学边界，不能以接口修复改方法。

新root：`/data1/user/ymdai/ember_runs/native_prefix_change_value_20261002`。
预计**9–11GPUh、含实现3–5小时wall**；依据原T270实测3.6656GPUh、450线性外推约6.1，
两次bank/400及seen预留约3GPUh，新计算只是末层两帧attention及38个E投影，不是第二次全policy读取。
一次profile据实际吞吐修订ETA。**硬限14GPUh、48GiB新增峰值**，
包含所有加载、profile、失败/恢复、等待占用、临时文件、ECP、banks、捕获和代码；达到12GPUh且剩余工作预计可能触及硬限时报告具体缺口。
新增存储估计32–38GiB，来自约944个B载荷/共享A、5个完整ECP及52 full/892 compact，现场按实际大小复核。

全部新增在data1，开root/大缓存前由执行者查strg01独立quota、个人实占与共享容量。
每次launch live查两节点，按AGENTS总8/空闲≤10时总6及单节点≤6，选有实际吞吐收益的rank/worker；无额外峰2卡限制。
不同阶段在依赖允许时并行，不等待凑卡、dummy占卡或干扰他人。正常长任务一次持续等待退出，不固定轮询日志/cache。
正式消费者来自clean pushed detached提交；新旧训练、物化/读取代码身份分别保留，不重算有效历史结果凑同提交。

交付完整run contract、训练事件/metrics、五个ECP、两个400及seen144原行/bank/capture、所有配对统计、
紧凑被动字段、费用/峰值/退出回执和completion/readback。完整944到齐即一次回报主讨论并停止新增计算；
不安排阶段自通知或自动后继。主讨论收到后直接分析与接续；科研目标不会因本批完成而被写成已经达成。

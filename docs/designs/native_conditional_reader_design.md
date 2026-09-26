# 原生执行查询读取教学内容：候选学习比较

2026-09-26准备稿。**尚未激活/派发，不是新GPU启动许可。** 当前唯一执行合同仍为S0，见progress。
本稿把[特征到算子的机制分析](../analyses/feature_to_operator_mechanism_20260926.md)落实到可审阅的下一比较；
S0结果可能修订其优先级。不得从本文件自行恢复/启动实验，阶段转换由主讨论完成科学裁决，不增加Owner审批。

## 1. 要裁决的一个问题

给定相同合法教学输入和动作监督，**让执行特征直接查询教学内容、并共同学习读取表示**，能否获得当前视频Writer
未取得的有益闭环能力？若能，再检验其作用能否交给完整LoRA；若不能，不投资后续蒸馏。

R_V是视频条件的原生读出；R_L使用相同执行读出接口但只读exact language。它们是训练/诊断模型，
执行中会消费预计算的教学memory，**不符合最终EMBER只加载一次完整LoRA的部署合同**。
其成绩不能被报告成EMBER方法成绩或替代强MT-BC比较。

R_V优于C0但不优于R_L，只说明执行条件化的参数化可能更有效，未说明视频价值。
R_V取得正确及另一正确视频收益，才给后续编译提供值得传递的功能对象。
此比较仍同时改变了原生读出的计算位置和学习参数化，不把系统级干预夸成单个attention模块的唯一因果归属。

## 2. 固定的特征和原生接口

### 教学侧

复用C0的真实text-only q、双相机图文E、完整50位置H、LanguageSemanticCore和RecurrentProcedureEncoder。
R_V memory为`C[L,256]`及`P[T,256]`，保留各自mask及P的真实frame positions；不新增外部视觉模型、对象标签、
动作中间量、低阶路径统计或新的联合token编码器。H只在真实learned read后收缩。
R_L memory为同一个LanguageSemanticCore的language-only `C=T(q)`，不输入假图像、fake action或零视频。
两臂共有组件的初始化固定种子，未使用组件不装作活动参数；读取模块、query网络及输出head均fresh。
编译器、FactorHeads和旧checkpoint不参与该Reader的学习或初始化。

### 执行侧

固定原生第10层（index9）的`self_attn.q_proj/v_proj`两个真实LoRA target输出，加上相应读出残差。
它们的输入都是该层AdaRMS后的1024维hidden；q输出2048，v输出256。保留原gate、K通道、其它层、mask与flow算法。
旧native reader在同层的input_layernorm之后注入，不能称为完全相同接口；本次将作用点对齐到实际LoRA target，
避免将1024维hidden纠正冒充2048/256维投影纠正。选择此深度为复用已检查的执行位置，不声称最佳、不搜索其它层。
执行完整H50，每次实际denoise在上述两个target各消费一次reader。

令该处原生hidden为`h[B,50,1024]`；新增模块width256、8 heads：

```text
q = W_q LayerNorm(h)
c = CrossAttention(q, RMS(C), C)
p = CrossAttention(q + RMS(c), RMS(P), P - masked_mean(P), rope_keys=frame_positions)
u = W_c c + W_p p
delta_y_m = W_om [GELU(W_h LayerNorm(h)) * u], m in {q,v}
y_m_out = W_m h + delta_y_m
```

R_L不执行P读取，取p=0；其余共有运算相同。两个target共享读出trunk，输出投影`W_oq:256→2048`、
`W_ov:256→256`均零初始化，其余投影bias=false并使用固定fresh初始化。
R_V的P读取沿用当前Compiler的Value中心化和Key位置规则，以免在同一比较中另试取消中心化。
Core值和P值来自教学；q只决定执行时读取与作用。原source投影残差负责保留初始函数，输出零初始化不要求首步所有上游梯度非零。
本Reader只修改这两个target，不伪称已修改全部38个。后续若编译，Writer仍须生成全部38-target完整LoRA，
主FM全链训练；局部作用匹配只在这两个有对应R输出的target定义，不能给其它target虚构教师标签。

这个作用点有直接的特征解释：每个实际attention head中，`P_attn=softmax(QK^T/sqrt(d_head))`。
q纠正改变当前位置怎样读取原生prefix/suffix Key，v纠正改变attention传出的内容。局部一阶式为
`delta(P_attn V) = J_softmax[deltaQ K^T/sqrt(d_head)] V + P_attn deltaV`；
RoPE和GQA重复仍按真实native实现，不将这个局部式当作整网线性化精确值。
它把“教学内容帮助关注什么/传递什么”落实到实际算子，但不预先给某个head赋予物体或阶段含义。

旧native reader使用固定E/Meta、仅训读头并只验离线FM；本候选让上述教学读取器和reader在真实原生续算下共同学习，
并以闭环和匹配language读出裁决。旧失败仍是风险证据，不能因名称不同忽略它。

## 3. 学习合同准备

两臂均fresh、source权重始终冻结。唯一目标是真实全H50随机tau的动作FM，不新增KD、RL、参数重建或保持loss。
拟沿用C0/S0的fit28 allowlist、demo0..45、同teacher跨episode规则、4tasks/update、21主+7额外queries、
原两组权重及同一事件流前630宏步（70560queries、2520教学事件）。教师行动只作为query输出后的损失标签。
teacher46..49保留为训练侧诊断，held/official Validation/Test不产生梯度。
Optimizer、LR时钟、任务权重及world2恢复合同按C0原630定义，不能因reader吞吐不同暗改科学预算。
拟只在固定630评测，210/420/630保留完整恢复点；不预铺六节点曲线，不把此预算称充分收敛或方法上限。

可按现有VJP设计分块：先无梯度生成memory；每组真实FM对memory叶子与reader参数求梯度；
累积同condition的memory cotangent后，同参数版本重放教学encoder并反传。
这与直接保留整个图应有相同更新语义；不能detach教学encoder、跨更新复用Meta后特征或丢失原生续算梯度。
Reader始终在实际query前向及其activation-checkpoint重算期间安装，避免hook生命周期造成漏读。

## 4. 先完成的工程准备范围

本节是下一项可派任务的具体内容，仍未派发。工程不读科学分数择优；先准备两种模式的共享接口，
科学训练分阶段，只先投入R_V，不默认把第二臂及蒸馏全部排满：

1. 独立`codex/` worktree从最新main实现上述唯一读出，复用现有特征、sampler、真实FM和canonical evaluator。
   实验owner集中管理reader/memory；不建立第二套source、loader或完整评测器。诊断路径有明确退役点：本候选否决后由Git保存。
2. CPU检查shape/mask、source identity初值、query依赖、完整H、memory棱边与梯度消费者；不做与实现同义的庞大测试矩阵。
3. 两臂各4宏步及2→4完整world2恢复；最长合法训练video一次真实forward/backward/profile；不把smoke权重用于formal。
   验有效梯度、Source冻结、完整恢复和配对流即可，接受正常BF16/TF32及reduction差异。
4. 每臂在train任务global2/12/22、state0、teacher46各1条canonical闭环接口smoke，共6条。
   R_L没有teacher内容输入，保留配对metadata不建立task-ID条件通路。六条仅验实现，不按成功数决定是否开跑。
5. 计入全部加载/失败/profile时长，报告真实每宏步吞吐、最长峰值、推算完整630与拟定评测代价。
   工程准备硬上限0.75 GPU-hours、data0新增2GiB；预算预计不足先回报，不改层数/参数/矩阵硬凑。

实施前按code-architecture-gate检查所有权与临时诊断路径；formal前满足clean pushed detached。
允许纯调度/CPU验收修正有明确版本来源，不因此重跑已有效的科学计算；模型、memory或rollout计算改变须冻结新来源。

## 5. 拟定的最小科学面板与投资上限

只有阶段4的真实接口/成本证据及主讨论裁决后，才登记formal合同和实际命令：

- 第一阶段只训R_V：held correct80、held other80、seen64，共224条。若没有值得传递的绝对能力，不自动训练R_L。
- R_L是必要的后继视频价值参照，但只在R_V取得投资信号后启动；同held80及seen64，共144条，
  同task换video对它无输入作用，不重复运行other。R_L定义、数据、630预算在看R_V前即冻结。
- 两阶段最多368新闭环。held8、seen16、state/video映射、policy/env RNG全部复用S0/C0的固定合同。
  C0优先固定引用S0批次的完整新配对面板和原F银行，不按新旧C0分数择优；B仅历史参照。
  版本/历史self差异限制显式报告，不拼接新旧行选较好成绩。
- 全行T+1对象/EEF/夹爪/真实动作及BDDL谓词；固定full为R_V held-correct8、other4（global0/14/21/38）、seen4，
  R_L held8、seen4，共28；seen取登记seen列表前4task/state0。pilot计入总数。
- 从真实rollout保留首轮full10动作，不额外生成函数query；本批不开展rank、局部回归或teacher特征蒸馏实验。
  有用功能教师尚未成立时，不为§5的数学分解先铺大量hidden capture。
- 第一阶段含工程准备上限9完整GPU-hours、data0峰值8GiB、data1代码768MiB。
  第二阶段若值得执行另限5 GPU-hours，两阶段总峰值data0上限10GiB。
  这是投资上限，不是已测ETA；必须由阶段4真实profile核定能完成，不能先跑满再报告预算不足。
- 只用合格共享GPU，按项目双节点live准入和全局卡数；不占卡等待、不改他人进程。

第一阶段的默认投资信号预先定义为：R_V correct相对固定C0至少净增8/80、other不低于固定C0 other，
正确面板至少两个suite净增；完整报告能力丢失与seen，不把达到数值线当显著性或最后方法接受标准。
若明显不满足，不自动启动R_L/蒸馏；若有矛盾且追加会实质改变判断，主讨论须明确最小缺口和新预算，
不能通过换层、加步数或换参照追逐过线。这一线仅约束是否值得投入后继对照，不用于正式选模型。

两阶段齐全后的主要读数为R_V−R_L的正确/另一正确及R_V相对当前C0的能力、Source相对R/G/L、任务/suite/breadth、seen获取。
固定8task联合bootstrap用于描述有限不确定性，不能用是否跨零自动断言等效。完整行、失败与计量缺口均公开。
小面板正信号后仅针对会改变结论的缺口补证据，不自动补400、延训、加seed、加层或开始蒸馏。

## 6. 完成后的科学停止点

Sol收到未来正式派发后只完成该批，主动向主讨论Queue原件/退出/缺项，停止新增实验。
主讨论独立核验后依机制分析§7作判断：Reader无益则下调该假设；只改善语言能力不作为视频教师；
有正确教学闭环增量才准备传给唯一LoRA。Teacher成功仍不是最终方法成功，更不取消强MT-BC、保持和完整配对评测。

本稿不授权利用Test、held actions、wrong处罚或最终shuffle/reverse来修正架构。
它也不锁死未来只能采用功能教师路线；新反例应实际改变方法优先级，不能把“尚未普遍证伪”当作继续投入的理由。

# 关系450的动态信号与实际学习信用：一次有界原件核验

2026-10-07。Owner持续自主授权下，main消费关系共同学习整批后登记。
本批是train-only冻结机制诊断，不是新的学习候选或闭环评测；不恢复关系450训练。
依据及已完成CPU原件见findings§368和`docs/analyses/relation450_main_consumption_20261007.json`。

## 1. 为什么这一项会改变完整方法判断

G450=126/400，G360=131，强MT153/T2340=161；同龄T450122、U450143、C450137、Context450126。
G未显示广泛同龄优势，且特权F seen66弱于G84；“已学成强物理反馈，只待视频编译”的前提不成立。
不靠延长或修改KD系数延续该前提。旧Video Functional去蒸馏、原生读出和C900的训练/迁移分离负例仍约束判断。

新的具体线索是：G的Phi参数组占几乎全部post-clip范数，450步中438步发生clip1；
末90窗pre-clip中位3.124。它不是单独的辅助梯度读数，不能由日志直接断言辅助项唯一挤占了功能信用。
在原450 Adam状态的固定分层样本中，G编译器约89.94%的sqrt(vhat)<eps1e-8，
中位衰减sqrt(vhat)/(sqrt(vhat)+eps)=.02070；同龄C的相同10类矩阵显著较轻。
例如a_out/b_out该中位量G=.267/.233、C=.989/.976；a_dynamic为.0299/.910。
这是固定moments下相对eps=0的代数量，不是实际无eps训练或性能因果，也不是修改eps的理由。

实际算子有可核对的差别：C从native H相邻差出发，各InterpretationLayer用rms_zero动态输入；
G则从预测物理字段经Omega形成w，再以H查询得到w[t]-w[t-1]的加权和及bias-free投影d，
未作这项幅度归一化。两者ConditionalTarget都用动态线性项乘context门，再由零初始化输出头写S/M。
小动态幅度可能同时压低写入和上游学习；也可能只是预测运动失真、软presence稀释或整体未学成的结果。
3–6倍全局裁剪不能单独解释全部差异。必须把原物理运动、预测轨迹、d和真实FM信用连起来检查。

本批回答这项有限问题，决定是否有依据保留关系表示并修订其学习接口；
无论内部数值如何，都不能证明能超过MT、不能修复F弱的事实，也不自动取得下一轮训练资格。

## 2. 固定对象、数据与信息墙

- 冻结G450/F450：`relation_grounded_writer_20261006/train/attempts/fresh/checkpoints/macro_00000450`。
  G/F训练代码85614d9c，标签及数学实现读取原frozen；后续工程读取版本与训练身份分开记录。
- 冻结C450：`conditional_read_write_fresh_20261001/conditional_read_write/train/attempts/resume360_native_packing/checkpoints/macro_00000450`。
  不使用更晚C900或T2340替代同龄动态参照；source1000/prefix不变。
- 表示面板为原G450 seen144中每个36-task的init32所用teacher，共36条件，各模型保持其原公共beta/native读取。
  不把一个模型的H硬塞给另一个。GT只作离线匹配/误差参照，绝不进入G条件或presence mask。
- 功能信用面板固定为关系训练原metrics/event的宏步1、225、450，每步原4个条件及各28个query，
  合计12条件/336 query。使用原teacher/query/frame/flow_seed；模型均冻结在450，非训练轨迹重放或checkpoint选择。
  全部为已有36任务、已有标签请求；没有新任务、视频、state采集、held teacher特权或Test。
- K1、stride5及真实末帧、完整50×7 FM、跨episode、同z/tau/y继续沿原合同。
  RGB末帧保留，GT缺失仅在误差项mask。实体匹配为整视频同一Hungarian，沿原loss定义。

## 3. 只做以下实际消费者

**A. 36条件的表示链。** 逐条件流式读取，不另建native/图像大缓存。
G记录各真实实体/hand的p、R、q和presence误差；报告p的米制绝对误差及相对首帧位移误差、
GT与预测的相邻运动幅度，静止/无标签分母单列，不把body原点称抓取点或空槽称对象。
保存固定匹配、有效帧/槽数和原teacher来源。报告真实匹配槽与空槽的软presence总质量；
GT分组只作统计，不能用于重新加权实际forward。
在原forward中采集H、w/context、实际d及写入前线性动态项的RMS/分布，记录38处S/M的聚合量。
C在完全相同36视频上记录其原H、c、d及相同编译接口量。单位不同的隐藏量不能当物理精度，
范数也不能直接当控制作用；不得以一个整体平均掩盖任务/实体/时段差异。

**B. 12条件的真实功能信用。** 对冻结G原完整图，在原同一FM sample上分别读出
L_FM、.25 L_KD、.1 L_relation的梯度；F只为提供原stopgrad target，不训练F。
保留原四task等权/28query均值，分别汇总public/Phi/Omega/read/各类Compiler的范数，
以及共享组内功能项和辅助项的内积/夹角、合并后的clip因子。
使用原optimizer状态，只计算真实梯度对应的Adam分母/eps衰减统计，不执行optimizer.step，
不进行反事实LR/eps/clip系数扫描；在报告中区分梯度、moments、候选更新与实际学习效果。
C在相同12条件/query/noise下仅计算其原完整FM和对应组统计，原式与模型身份保持。

**C. 唯一固定算子干预。** 仅在上述12条件，将G交给ConditionalTarget的d替为
`d * rsqrt(mean(d**2, -1, keepdim=True) + 1e-6)`，即已有rms_zero公式；其它输入、参数、标签和F不变。
只比较同样真实完整FM及其梯度、S/M与动作输出的差异，保留全部不利变化/非finite或OOM。
这是冻结条件下的单位幅度干预，首帧零仍为零；不是部署候选、闭环分数、归一化有效或视频理解证明。
每个condition只做这一个固定变换，不改常数、插入其它层、调输出增益、看结果后选择视频或追加变换。

不得新增环境/rollout、action采集、训练更新、checkpoint选择、公共/错误视频/乱序/倒序controls、
参数或dtype扫描、teacher forcing、动作/phase/goal标签。保留正常BF16/TF32，不做逐bit/全tensor防御比较。
此处针对已观察学习尺度的统计不是一般精度审计；若实际消费者暴露科学语义/原件有效性未知则报告边界。

## 4. 解释与停止线

若物理轨迹本身严重失真，单独放大d没有得到可信动态证据的依据，不能据梯度变大再开归一化长训。
若轨迹保留了任务相关运动而d/乘法写入显著衰减，且固定变换确实解除真实功能信用衰减，
只加强这项有限接口解释；F弱、同龄迁移阴性与动作变差的反例仍然有效，下一完整方法由main重新裁决。
若d和功能信用并无相应缺口，则关闭幅度/eps解释，不接epsilon、rank、scale、seed或更多层位扫描。
不把pose可读、梯度非零或内部FM下降当最终成功，也不要求本诊断证明唯一根因才允许结束。
这批完成后必须回到完整方法选择；不自动把局部阳性接成新的诊断链。

## 5. 资源、复现与交付

唯一root：`/data1/user/ymdai/ember_runs/relation_signal_readback_20261007`。
预计含消费者实现45–90分钟，依据已完成G每条件约13–20秒与本次最多48条视频/336个独立query；
反传拆项和C成本尚未实测，因此硬上限实际承接起3wall-hours、2完整GPU-hours、4GiB新增峰。
只用一张适合的A40；现场全项目卡数仍遵AGENTS，所有加载/失败/物理profile计入新批预算。
旧检查点/源模型/标签/数据直接引用，不能复制；launch前双节点、data1独立quota/个人用量/共享容量核验。
复用已测frame/microbatch；在固定case内按实际余量放大物理执行，不额外造profile科学case或反复读共享缓存。

实验session独占tracked/Git窗口并闭环工程，使用原clean pushed frozen运行面和有明确来源的诊断消费者；
不热改旧frozen、不复活已退役关系trainer，不为一次诊断增设大型长期模块或评测器。
源码/实际命令、样本manifest、原始紧凑数值、分组统计、解释限制、失败与完整成本/退出记录保留。
main只消费科学原件，不重复工程测试。整批完成或真正边界一次回main并交回写窗口，
以既有可靠Queue/resume路径接续；无自通知/轮询/阶段消息、无自动训练或后继。

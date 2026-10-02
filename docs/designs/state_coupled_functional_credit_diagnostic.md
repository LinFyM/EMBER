# 自身执行特征的功能信用：两臂有界学习辨识

2026-10-02。依据机制分析§102登记；仅在既有四个train任务上辨识一种学习联系。
不是新canonical Writer、fresh完整训练、held选点或部署第二控制器。实施与真实状态只看progress。

## 1. 问题、已有证据与竞争解释

旧T2340的64步学习已经产生实际控制：parent/S/P/D分别17/20/20/23成功（各32行）。
task32的D17补齐搬壶，D43没有同样收益，S/P又有D没有的成功；不能把D当作唯一正确参数标签。
同B20上task32的S/P/D修正差异能量仅.346%/.479%/.789%，但它们的闭环收益不同。
Q/R与视频来源干预又同时出现有益和有害联合作用，未给出可直接删去的坏组或独立阶段。

这支持研究“共享教学特征如何学成在自身状态有用的完整控制函数”，没有定位一个坏神经模块。
本次具体假设是：在原真实LoRA FM之外，让同一教学表示接受**当前生成策略自身hidden的动作残差信用**，
并让该信用返回产生hidden的LoRA/Writer，可能改善原S在搬壶和其它任务上的获取，而非仅改善旁路预测。
竞争解释是：额外监督只训练出一个新头、仅对视频表示有用，或反而让执行hidden适合辅助读出而损害原动作出口。
真正检验的是下述live/stop梯度差，不以“多了一条梯度边”宣称原因已经找到。

最近似完整历史为`video_functional_writer_design.md`§3–11：旧Reader用未挂生成LoRA的冻结source query，
曾共同训练视频表示；额外拟合及中层读出仍弱于学生，去蒸馏、VL学习与移除辅助均未建立完整迁移优势。
所以不能把联合训练或动作辅助当作新原理。本次只检验**当前执行hidden的可学习消费联系**，保留负证据，
不恢复旧表示、扩大头/层位/拟合预算或蒸馏。两臂都读取当前hidden，仅一臂允许辅助信用经它返回Writer。

## 2. 固定父点、数据与唯一变化

- 父为旧T2340，原训练`e2afbfd7`；不是条件读写900或S64权重。
  `/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928/continuation2340/T/train/attempts/continuation/checkpoints/macro_00002340`。
- 固定source、公共A/B0、probe、normalization和native；只学习原P/C/D/O及新共享辅助头gamma。
  所有condition共享一套Writer和一套gamma，没有task/teacher私有参数。
- 任务/teacher：0:(40,11)、12:(25,14)、20:(38,42)、32:(17,43)。
  精确复用`operator_learning_limit_diagnosis_20260930/query_manifest.json`的64步、A28、frame、flow seed与offset1；
  每condition每步28 query，八condition各1/8，主FM完整50×7及原末动作补齐。两臂使用完全相同的逻辑流。
  每臂14,336次condition-query使用，并非等量独立样本。B20仍只独立于这64步，父点已见原训练池。
- 两臂均从同一父Writer、新gamma及fresh AdamW开始，64更新，lr1e-4、betas(.9,.95)、eps1e-8、wd1e-4、
  统一clip1，无scheduler。参数初始化seed7；gamma在独立RNG作用域初始化，两臂初值相同。
- `live`：辅助头的query使用当前完整LoRA执行hidden并保留梯度。
  `stop`：同一hidden数值仅在进入辅助头时detach；主FM路径、视频表示梯度、gamma梯度均保留。
  这是本次唯一比较因素；不添加蒸馏、公共loss、native动作辅助、时序/一致性正则或参数目标。

## 3. 实际特征、辅助函数与信用

复用旧T的真实计算。每个target l、stride5转移t：

```
K_t = normalize(X_t A_l^T)^T
U_t = [GELU(P_l K_t^T + C_l Hbar_t) * D_l(Hbar_(t+1)-Hbar_t)]^T
Z_(t+1) = Z_t + (U_t - Z_t K_t) K_t^T / 50,  Z_0 = 0
M_l = O_l Z_final;  A_l固定，B_l = B0_l + M_l
```

保持原M递推/浮点语义；Z是同一递推的O前256×128内容，可同时累计，不用参数拟合得到。
Hbar沿原RMS eps1e-6，K由固定公共A生成；没有动作标签进入K/H/Z。正常浮点下M=OZ只作聚合数值读回，
不为逐元素一致重跑native、固定batch或换dtype。部署bank仍由原完整T递推生成，不用历史bank残差查表锚定。

辅助memory为全部38个Z的128列，共4,864个256维token。token的target/rank身份是公共参数坐标，
不是task、episode、filename或结果身份。一次标准四头attention，宽256/每头64，无dropout/FFN：

```
q = Wq RMS(h_current)                         # 原action_out_proj前50×1024 hidden
k_(l,r) = Wk RMS(Z_l[:,r]) + e_target[l] + p_rank[r]
v_(l,r) = Wv RMS(Z_l[:,r])
R = Wo concat_heads(softmax(q k^T / sqrt(64)) v)     # 50×7
```

Wq/Wk/Wv/Wo无bias；Wo零初始化，其他线性沿PyTorch正常初始化，e_target用std.02正态，
p_rank为固定标准sin/cos位置编码（128位置、256维、base10000）。固定RMS无可学仿射、eps1e-6。
身份只入key不入Value；所有Z为0时R为0。此结构仍可学成task识别或共同修正，不能由此声称视频必要性。

在**同一真实FM forward**取得学生速度v和h_current，y为原noise−action；无第二次source query计算：

```
L_F = mean((v - y)^2)
L_R = mean((stopgrad(v) + R(h_current, Z) - y)^2)
L = L_F + L_R
```

两项均只对完整50×7求均值、按condition各1/8；L_R权重固定1。v的停止梯度避免额外直接复制主FM梯度，
辅助query/动作标签只进入生成后的训练消费者，不能反向成为Writer条件输入。
允许分块VJP，但必须同时回传LoRA叶子的余切和Z的余切，且来自同一参数版本；不能只更新gamma。

以均值归一化后的e_F/e_R记残差，裁剪前的实际信用为

`g_phi = J_v^T e_F + J_Z^T R_Z^T e_R + nu J_h^T R_h^T e_R`，nu=1/0对应live/stop。

J_h包含当前完整生成LoRA的上游执行计算；最终action_out的O不在h上游，不能声称这条新项直接覆盖所有38出口。
P/C/D可经Z收到辅助信用；所有P/C/D/O仍接真实完整FM。gamma只接辅助真实标签。
该附加项可能推动原动作出口的无用方向；它是需要行为证据辨识的风险，不是自动更好的梯度。

## 4. 只在64读回：完整学生与明确标记的辅助诊断控制器

原parent/S/P/D的32行及A28/B20预测复用，不重训S或重跑参照环境。旧native曾在RAM中；新浮点读取可能使
重建起点与历史bank稍有偏差，报告一次聚合误差。live/stop同批特征、同起点的比较才是主要因果对照；
与历史S的能力比较保留这个边界，不以微小净差证明辅助相对纯FM的严格收益。

每个学习臂固定64后生成八套学生LoRA。每臂各有两种消费者，均执行原32行：

1. **student**：仅完整LoRA，Writer/辅助头/Z在rollout前退出；这是唯一符合EMBER部署形式的结果。
2. **reader diagnostic**：同一LoRA加R速度残差，R读取预先编译的Z及自身每步真实hidden；
   不重读视频、不优化、不使用teacher标签。它是训练期功能机会的诊断控制器，绝不计入EMBER部署分数。

合计固定**128新增环境行**（两学习臂×两消费者×八condition×四init）；只有16个不同task/init。
每组32行固定六full：task0/12/20各第一teacher init0，task32 teacher17 init0及teacher43 init2/3，
共24 full、104 compact；全部保留continuous/goal/实际actions。其余无新case或逐结果补图。
每task两teacher各复用四次，是已登记有限诊断，不能称整轮无放回或paired400。

两消费者均用原scene/root7/绝对policy噪声时钟、官方双256→224/rotate/state8/action7、10 flow、前5、
settling10及suite horizon，成功即停。R仅修正7维velocity，pad维保留原值。source加载与持久queue按吞吐共享。
在原固定A28/B20上分别读学生/reader的FM及原十步B20预测；保存原full50，首5/全50及motion/gripper分别报告。
不拟合新探针、不取held标签，不把辅助头离线MSE当实际功能教师证据。

## 5. 结果分支与停止线

首先报告每臂student绝对成功、逐task/teacher、R/G/L/churn与原S/parent的行为变化，再比较live与stop。
task32保留抬壶/放置与开炉得失，3cm不作抓持判据；task0的原全成功及task20的已知损失不得隐去。

- live学生有跨condition的实际新增控制、优于stop且值得与原S比较，才提高“自身hidden的辅助信用有用”的支持。
  单例改善、主要由参照退化产生的差额、或伴大范围能力损失，均不能作为后继完整训练依据。
- 两学生近似或stop更好：本窗口不支持所检验的新增query信用；即使都改善，也不能归因于该项。
- reader控制器确实更强、学生没有相应获得：存在该额外函数类的局部机会，却未完成固定LoRA能力传递；
  不自动加蒸馏、延长、扩头或把reader作为部署补丁。
- 连reader的实际闭环也未改善：降低此功能读取/学习构造的优先级；不能只看拟合下降继续追加。

任何分支都在64及上述读取后停止。没有相邻选点、Test、RL、shuffled/reversed、扩数据、λ/层位/seed/rank扫描，
不自动触发fresh、另一个诊断或新架构。主讨论结合全部原件及完整历史作下一判断，正例也不直接证明跨任务迁移。

## 6. 资源、工程所有权与生命周期

唯一新root：`/data1/user/ymdai/ember_runs/state_coupled_functional_credit_20261002/`。
硬限**3完整GPUh、8GiB新增峰值**，预计峰6GiB；预计含工程/训练/物化/读取**75–135分钟**。
依据旧S64单卡1537s/.427GPUh，两臂新增头及VJP预计约1–2GPUh，128行与功能读回约.2–.5GPUh；
工程约35–65分钟，适当并行下训练约20–45分钟、物化/读取10–20分钟。实测profile更新预计，不静默超预算。
达到2.5GPUh而预计剩余无法在3内完成时报告具体缺口并收束；不以低loss延长或悄然削减正式面板。

复用canonical source/datasets/assets及旧spec。teacher17/43已有同父H/K可只读复用，其余六条native每条只读一次；
固定公共beta/A使H/K缓存合法。只保留必要H/38 K/indices、完整checkpoint/manifest、bank、原始rows、汇总与失败费用，
不复制完整raw X、大资产或历史checkpoint。所有自动cache/tmp/字节码等新增写入也必须在data1。

执行者独占工程分支/worktree，完成检查、main集成/push及clean detached冻结后运行；不得原地改历史冻结树。
沿用原T、canonical FM及evaluator owner，新增内容只属本诊断，不新建长期Writer模式或并行trainer。
必要检查覆盖live/stop真实梯度边界、同逻辑样本权重、完整LoRA/辅助值、学生部署移除头、配对与完整恢复。
真实profile仅用本批已登记输入，最多每臂两次可丢弃更新后恢复初值/RNG，不新增环境smoke或科学case。
允许有依据的I/O/调度修复，不改变标签、目标、模型计算、对照或选择合同；科学语义/预算变化由主讨论裁决。

启动前现场检查双节点GPU及data1独立quota/个人实占/共享容量。按实际吞吐选rank、microbatch和frame chunk，
有余量须验证更大物理batch/适当并行；不把旧micro14/frame32或单臂world size设成上限，不为填显存增样本。
两臂可独立并行；多rank共享更新须守原有效batch/各condition1/8、NCCL与拓扑合同。所有失败/加载/退出计费。
整批一条完成或实质边界消息，执行者结束后移除诊断专用入口/hooks，保留Git/frozen/原件并交回canonical窗口；
主讨论不重复工程验收，也不以其验收作为执行停点。禁止自通知链、额外agent、自动后继和固定间隔空轮询。

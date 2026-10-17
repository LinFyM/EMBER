# 真实自身观测上的原生算子编译：完整原理与首写学习窗口

2026-10-17。科学owner：01a13dd9-837d-7f33-b15a-4eb170bcf04b；
唯一实验执行：01a13dbf-dbc3-7621-82be-87be44a5bca6。
Owner已授权边界内自主实施、分析与科学接续，无例行Owner或main工程ACK。
本文件登记新候选和唯一有限窗口；登记不代表已经实现、启动或获得性能。

## 1. 决策及当前范围

最终目标仍为raw source上的一套完整38-target/rank8教学LoRA，固定后在新初态明显超过强MT，
并说明教学贡献和能力保持。source/normalization冻结，0MT初始化/合并/回退/第二adapter。
§21/22回归家族已依相邻290→306、121→128的集中增量与强参照缺口停止；
完整消费、最近似源码和竞争解释见
[控制空间分析§32–33](../analyses/source_control_space_learning_20261015.md#32-相邻原件触发回归家族停止而不是抹去已学到的能力)。

当前唯一授权是：实现下述原生完整算子及真实共享信用，在原36个non-held任务上fresh学习180更新，
用已保存的真实source经历做初次完整写出，末点36套LoRA执行原144新初态query。
一次受限临时消费者/物理profile归同批；无新实践采集、reference训练、held400、controls、Test或自动后段。
这是真实学习加闭环的有界窗口，不以非零梯度、内部loss或算子重构作为科学通过。

本批检验首写，尚不训练完整反馈循环。完整方法的后续适应可用实际当前LoRA实践所得H再次整体编译，
按登记的成功/资源规则停止；每次替换一套完整LoRA，不平均多套输出或回放动作。
不预设观看/实践/重读轮数，不把本次首写接口推广成永久单轮限制。
后续循环的自身状态分布、预算和学习组织须依据本批原件另登记，不在本批隐含运行。

## 2. 固定资产、输入与信息墙

沿用coverage_v1的24/8/8协议及原36-task显式allowlist：
0,1,2,4,5,7,12,13,14,15,17,19,20,21,22,25,28,29,32,34,35,36,37,38,
42,43,51,55,56,62,64,73,95,96,97,101。
36任务等权；target24/source-seen support12、五identity参考任务分开报告。
不新增source71其它meta tasks，不由已有大目录扩大标签范围。

沿用source_policy_space_compiler_20261015.md §2的canonical source step1000、冻结normalization、
tokenizer、LIBERO assets和38-target形状；实际rank=alpha=8、dropout=0，76因子/643584标量。
source始终无trainable parameter。允许对其真实计算求输入/中间响应导数，不优化source权重。
执行和合法FM query正常读取exact language/真实8维proprio；Compiler不直接读任务名、L tokens或task ID。
自身真实hidden中的间接语言信息合法，不额外去信息。

教学为原始同task双相机RGB、stride5加真实末帧，teacher动作/state/proprio/reward/terminal/pose等不进入条件。
只读复用Reader900产生的完整有序M，身份、来源与原费用均继承；原1800条视频/59114采样帧池不扩展。
M保留每帧8个slots和原时序位置，不平均frames，不读545 producer条目，不跑缺state的teacher action probe。
Reader900/Gemma/source图像塔冻结；本批不复制原高维prefix，不读取held M。
这个Reader曾由原FM/producer监督学习，参考库的历史来源与成本不因当前不回归参考因子而消失。

真实H唯一来源：

    /data1/user/ymdai/ember_runs/source_policy_space_compiler_20261015/feedback_proposals_seed7
    conditions.json
    episodes/task_TTTT_v0_practice_000/state_III.json
    上述row明确指向的history_path

main已核36任务/144有效source episode、全部behavior_y=0、MT_loaded=false及144个H文件存在；
合计38937 control＋1440 settling＝40377步、52次success、7808个真实replan。
执行者用原schema/identity消费者确认其source零代码语义和raw H结构，不能仅凭文件名推断行为。
不采用practice001以后H、Reader的83成功初始实践、旧MT H或后续query结果。
源历史只读引用，不解压落盘成第二套原始H，不改旧SQLite、manifest或receipt。

每task的四个source实践初态均在16..49，报告query为0/4/8/12；源环境与noise root分别202610161/202610151。
这四初态是原已发生的经验面板，当前不重采或把它称作四轮学习。
同task源行为不依赖teacher，因此共享训练可让不同合法V读取同一真实H；披露固定H可以识别task，
不以更多video×同H组合冒充更多独立task或反馈轨迹。
报告36条件各自消费这四段H，继承40377步交互依赖；本批0新实践不等于方法zero-interaction。

## 3. 实际特征与局部功能压力

将四个episode按conditions.json的practice_states顺序串接，仅保留真实decision顺序。
在全部N个decision中取M=min(16,N)个不同索引：
floor-linspace(0,N−1,M)，保留原episode/step/实际noise seed。
这是首批函数支撑预算，不是视频帧抽样，不按成功、误差或人工操作阶段选点。
完整raw H作为原件保留，当前网络明确只消费这M个选定decision及其真实终止事实，不声称读取了全部7808事件。

每个选定decision使用：

- pre/post真实双RGB的冻结source视觉patch，各512×2048；
- 原同次完整10-flow的第0/9调用、action_out_proj之前hidden，各50×1024；
- pre/post自身proprio 2×8、实际执行5×7 actions、5D mask；
- 自身reward/success/terminal/truncation/failure五事实、实际episode/step两个计数。

复用已存在的图像/hidden/事实编码原理：image 2048→256、hidden 1024→256各带输入LayerNorm，
空间512/pre-post2/flow-call2/action-position50地址，facts63→256加入learned event query，
4-head attention读该事件的真实tokens；不输入旧64维behavior code、参考参数或task标签。
proprio/actions按source固定单位，计数沿既有8192单位，字段/实际执行mask语义不改变。
每事件再读完整M_V；M个事件经两层pre-LN Transformer（256宽、4 heads、FF1024、dropout0）形成c_i。
编译时这些H均已发生，采用全已见H的双向attention；无未来query或后续实践输入。

以完整raw source、选定真实RGB/proprio与原noise重算完整50×32 latent/10-flow，
得到F0_i前5×7归一化动作；不能以旧执行动作、历史hidden、time1 velocity或假state替代。
原生prefix必须经过真实Pi05LiberoProcessor含自身state的prompt，不能调用teacher的无state路径。

共享pressure头：

    p_i = Linear(256,35)(SiLU(Linear(291,256)(concat(c_i, flatten(F0_i)))))

最后Linear权重均值0、标准差1e−3的Gaussian、bias0；其余模块fresh seed7及既有默认初始化。
非零小头用于避免rank投影在全零矩阵处的不可微起点，不是用参考或MT初始化任务LoRA。
输入任务LoRA仍为source恒等；未训练编译器首次读后可输出小随机编辑，不将其写成identity保证。
不另加幅度head、动作decoder、critic、阶段标签或正确动作压力监督。

p是当前原生响应上的局部余切，不是一条恢复出的正确动作轨迹。
等价的局部线性准则为C_phi(z)=mean(p_phi(V,H,F0)·z)，只对z求导；
其方向由共享训练的实际完整LoRA功能损失教会，部署时没有正确动作标签或task-local拟合optimizer。
预测余切与原生回写在PPW中已存在；本候选的区别仅在下面真实状态、完整十步、写入空间和功能校准。

## 4. 从真实原生导数直接形成完整rank8算子

### 4.1 在权重算子坐标求导

把38个原生linear的附加矩阵记作ΔW，F(ΔW)为上述M个真实观测上的完整十步前5×7。
当前首写在ΔW=0处，J0=∂F/∂ΔW；不是在固定A0/B0因子切空间求导。
令n=35M，用FP32聚合：

    G_l = [J0^T p]_l / n

实现可在真实各flow调用捕获该linear的输入X及输出余切C，按全部M、50 latent位置、十次调用求ΣCᵀX/n。
各次调用的C必须来自完整十步最终F的真实反传，不能把十个独立time1余切相加。
source权重冻结；使用既有中间响应VJP/伴随原理或等价显式零算子叶，不加载/优化source参数。
完整38处同时进入同一个J0，物理chunk只改变执行安排、不重复除M/n或改变层权重。

### 4.2 梯度决定的rank8范围

每个target固定公开Ω_l∈R^(d_in×8)，CPU generator seed=2026101700+target ordinal，
均值0、标准差d_in^(−1/2)的Gaussian；Ω不训练、不依赖task/video/reference/结果，不作seed/幂次数菜单。
对非零G_l令Z_l=G_l/||G_l||_F，在FP32计算：

    Q0 = qr_reduced(Z_l Ω_l)
    Q1 = qr_reduced(Z_l (Z_l^T Q0))
    Q2 = qr_reduced(Z_l (Z_l^T Q1))
    C_l = Q2^T G_l
    D_l = Q2 C_l

所有Q均为d_out×8；两个幂迭代是数值范围构造，不是观看/实践轮数。
这是明确的rank8近似，不能称为最优truncated SVD；两侧空间随真实G改变，
不是全局参考bank U、固定输入PCA或固定A0，仅Ω为公共计算探针。
QR及归一化对G的导数必须进入共享信用；不能detach Q来换成另一个科学算法。
不为QR加标签、随机jitter、截断秩菜单或悄悄用固定旧空间fallback。
确切零G的该target输出零算子并保留完整r8结构，按实现定义零分支；一般QR秩退化/非finite若实际出现，
作为具体计算边界交main，不把替换算法伪称数值修复。

### 4.3 用实际功能方向校准幅度

再用同一J0计算r=J0 D，所有38个D同时作用。定义：

    alpha = <p,r> / <r,r>
    ΔW_l = -alpha D_l

内积对同M×5×7坐标，sum或mean须分子分母一致。
精确算术中<p,r>/n=Σ_l||D_l||_F²≥0。非退化分支alpha>0；
零方向输出identity并明确记录，不以epsilon/clip/abs或事后scale改变公式。
消费者要核实际伴随关系及有限alpha；真实精度误差可按已有容差解释，不追逐逐bit。

该alpha最小化沿当前方向的||p−alpha r||²，故一阶ΔF≈−alpha r；
其幅度不再额外乘因子RMS平方。它只归一化这一方向，不是完整核逆/natural gradient，
也不保证有限非线性步或新状态的有益改变。没有任务内动作目标、轨迹数据库或BC微调。
q方向、低秩投影及自身状态覆盖仍必须由实际学习/控制判读。

直接输出A_l=C_l/s_l、B_l=−alpha s_l Q2，使B_l A_l=−alpha D_l。
为避免无意义的大/小因子，可取s_l²=||C_l||_F/(alpha||Q2||_F)并stop-gradient s_l；
任何正s下乘积恒等，故停止这项纯gauge缩放的梯度不改变算子或其完整功能导数。
完整38×2因子FP32，正常native BF16/TF32；source仅执行一次W_source+B_l A_l。
不SVD截断已有参考、平均LoRA、加入旧W生成器或部署第二执行分支。

## 5. 完整共享信用与实际科学限制

相同task、不同teacher episode的真实原始动作query，沿原native FM50×32/前7维、
offset1/source normalization/flow_sample/NativeFlowPrediction监督最终完整A/B。
主loss只有原生FM，0参考BA/pair/factor MSE/新reward shaping/自身动作BC。
query标签只进入外层loss；p、M、H和source J0条件输入不读这些标签。

若外层因子余切为v_A/v_B，真实反传须经过：

    FM -> 完整A/B -> alpha、C、Q -> D、G、p -> pressure/context/自身读取

Q/QR、范围归一化及alpha的所有有效导数均保留；不能把alpha当常数或仅训练最后head。
J0是固定source、真实H上的一阶线性算子：
G分支的伴随为J0/n，r分支的伴随为J0^T。
因此需要真实十步VJP/JVP或等价的固定pullback伴随，不需要对source权重求Hessian。
重算必须绑定正确observations/noise/完整算子；禁止有限差分、单flow、假梯度或factor-only切空间替代。

本批M_V冻结，但读取M的attention、H encoder/context和pressure全fresh共同学习。
H重复且间接含task信息，当前p可能不使用教学；即使train控制好，也未证明视频操作知识或held迁移。
局部J0、rank8近似和跨初态函数外推各有失败可能。它们没有因“方向归一化”而消失。
原LocalField/PPW、Experience、功能修订143和当前回归负证据保持，不由本次公式清零。

## 6. 唯一180更新学习与144读出

180个有效共享更新、logical4条件×28 FM query=112/更新，总720事件/20160 FM query。
每9更新覆盖36任务一次，20个周期每task20次；各task每次从seed7的49条非报告教学permutation取新video。
报告teacher为feedback_proposals_seed7/conditions.json中v0那条，Writer本窗不以它产生训练更新；
Reader900以前已读过该视频，因此不能称它是全方法未见教学。
每事件query为7个非teacher episode×4个时段；query cursor/noise/time root随逻辑行保存，
允许排到报告teacher所在episode作为其它condition的合法non-held动作query，不混淆报告视频输入与训练标签范围。

AdamW fresh：lr1e−4、betas(.9,.95)、eps1e−8、weight_decay1e−4、global clip1、constant scheduler。
source/Reader900冻结，无旧optimizer恢复；不扫描LR/seed/rank/pressure幅度/幂次数/损失。
保存90和180两个完整checkpoint及一个原子rolling恢复点；90仅恢复/学习轨迹，不读出或选模型。
保存θ、Adam、scheduler/scaler、task/video/query cursor、Python/NumPy/Torch/CUDA RNG、
world/topology/schema与固定Ω身份；正常物理迁移按完整边界合同，不称bitwise exact。

唯一终点180对36个原v0条件，各消费原practice000真实H和完整报告V，生成一套完整LoRA；
每套在init0/4/8/12执行4个新episode，共36新LoRA/144新query。
完全沿原query noise root202610151、scene/env/policy RNG、official preprocessing/10-flow/first5、
dummy10和suite horizon；support horizon400。query反馈不进入编译或梯度。
这是注册有限池训练诊断：同一task-video复用四个新初态，不称400不同教学条件、未见task或正式模型选择。

只读强参照source58/144、selected reference127/144、同教学Reader90089/144，
以及旧first73/full79、随机搜索86、功能修订93等原完整读出；由实际raw rows核分数与配对身份。
selected reference是标签侧能力参照，不是当前deployment输入。含实际搜索/不同H者标明成本和比较限制。
当前source-W 1090/1180在此面板仅同视频init0的36锚可严格同condition比较，其余108不能伪配对；
其完整576/400成绩保留为背景，不拼成新控制器或重跑旧臂。强MT153/历史G/D held身份保持。

报告所有task/suite/target-support/identity、breadth、reference-only/source成功保持、R/G/L/churn/Jaccard/完整成功键，
训练流覆盖与真实FM曲线、q/alpha/实际参数变化、信息墙、数据重用、实际费用和失败。
只在受限消费者读取必要同支撑实际F_out以核一阶校准作用，不铺180更新/全部报告条件的额外动作探针。
train正效应也不自动证明视频作用、反馈修订或held能力；后继相邻/视频证据/strict400由main消费后登记。

若有明确广泛崩溃/数值或数据边界，携完整当前证据提前停止当前批并整批回报，不为180而盲跑。
到180必须结束本有限批，无自动360/新loss/尺度/QR幂次数/seed/rank/P微扫。
若与source58相比无有意义的多任务获取，或只换成功集合而大幅落后Reader89，降低并停止本pressure/原生范围组织的继续投入；
若较强且合理保持，才依据真实学习轨迹、代价与未解决缺口决定下一步，不由内部信号自动晋级。
首批不能以所有弱任务恢复作为先验成功保证，也不能用只胜旧失败版来降低最终强MT目标。

## 7. 实施、真实消费者与资源预算

唯一canonical运行面继续使用control_space，替换当前source-W生成职责；
旧model/weights/只读1090专用入口消费完后退役到Git、frozen和原件，不保留可自动恢复的训练fallback。
复用reference Runtime/NativeVelocity、真实H加载、FM、BatchedLoRA、动态队列/ledger/runner.close，
原生pullback/伴随复用PPW和功能修订已验证原理，不能硬载旧MT或teacher probe路径。
按code-architecture-gate审查职责、尺寸和生命周期；不复制第二Writer/训练栈或长期兼容开关。

有具体新数学消费者风险，故授权一次至多4个可丢弃logical4更新/448 FM query、0环境episode：
用既有task0/13/20/32的合法V和source H验证真实十步完整G、双伴随、QR/alpha/gauge完整信用、
首步及后续到context/视频读取attention的真实更新、完整保存恢复与LoRA实际消费者。
比较必要的小矩阵oracle和真实有限方向/伴随关系，不能只有随机toy或非零grad；
真实非线性F_out检查限这些四个条件及本临时窗口。临时权重不进入正式180。
如接口或数学不符，工程在授权内修复；不得用detach Q/alpha、改原生计算或新尺度掩盖科学变化。

profile同时服务这些有用消费者，比较support microbatch4/8/16、FM7/14/28及可行多condition pack；
有余量则在这4更新额度内验证更大的有用pack/无重算，按实际完整compiler/s、updates/s和峰显存选择。
逻辑4和任务权重保持，rank数由真实吞吐决定，不把单条件/旧world数当永久上限。
有条件的query分片可以增加物理rank，但每condition的G/QR/alpha只逻辑计算一次，不能按rank重复加权。
不增加假event、裁视频或dummy占卡填显存。

唯一ROOT：

    /data1/user/ymdai/ember_runs/source_native_operator_compiler_20261017/first_write_seed7

新增峰12GiB，含完整CP/原子临时/36 LoRA/必要原件/小cache。
原始H、旧M、source、参考、数据、tokenizer只读引用；旧M整体仍统一≤1GiB，不写旧SQLite。
新冻结H特征如确需落盘，所有worker合计≤4GiB，兼容旧特征优先引用；source native KV/全Jacobian/G矩阵不建全池磁盘缓存。
训练不保存720份dense G或逐tensor快照；只留必要数值/来源和少量消费者证据。
不复制原148GiB视频prefix池、MT或参考bank；新增全部data1。

本批新环境预算：

| 项目 | 正常范围 | 加16工程extra后的累计硬上限 |
| --- | ---: | ---: |
| 提交/episode | 144 | 160 |
| control steps | 50880 | 59200 |
| dummy steps | 1440 | 1600 |
| 联合total steps | 52320 | 60800 |

各上限同时有效，不相加。16只用于真实工程失败/未知执行，不给正常失败重跑或额外科学smoke；
有效行不重跑，未知按保守完整额度收费。40377历史source实践单列依赖，不混成当前新步数或从总方法成本抹掉。

建新root/cache、每launch/resume前由执行核strg01 data1独立quota/相关个人du/共享容量和预计峰；
双node live GPU/index/显存/util/process、AGENTS总6/8/node6及CPU-PSS/环境进程准入保持。
按实际可共驻余量安排，不触碰foreign进程；训练同node、P2P禁用/NUMA/deferred NCCL，独立eval无NCCL。
动态long-first persistent queue，空闲环境完整close，按CPU和真实吞吐选择slots。
没有累计GPUh硬门槛；所有失败/加载/临时消费者/重算及历史已付依赖均准确记账。

预计工程/实际消费者1.5–3小时，正式学习/读出约1–2.5小时，整批2.5–5.5小时；
依据旧真实十步四事件FM消费者39.91秒、本批多一条功能JVP和较小rank/H，以及144 query规模。
新dense算子伴随/QR的实际成本未知，首有用工作后须按实测修订，不伪称已profile；
显著超期或明确无效携具体证据提前回main，不机械训练五六小时以履行版本计划。

## 8. 所有权、Git与一次整批交付

main先完成科研记录与推送，再向既有执行者实际派发；接手后执行者独占canonical工程/科研记录/Git窗口，
main只读分析，不并发写。实现从最新main的独立codex worktree完成，针对实际消费者验证、及时合并推送；
正式训练/eval均用新的clean pushed detached frozen版本。原§22及所有历史source/CP/结果保持只读。
工程恢复保留原失败/有效更新/episode/时钟，科学数学/标签/选择/信息墙变化才回main裁决，无例行ACK。

一个真实owner持续直接等待整批退出；0阶段Queue、自通知、心跳或第二waiter。
完成后统一核有效更新、144唯一新行、数据墙/配对/账本、资源退出和Git/开发树生命周期，
一次将原件/完整正负分布/费用/吞吐/限制交main并归还窗口。到本有限目标即停止，无自动后继。
main承担科学消费和接续，不将此停止解释为等待Owner再次许可。

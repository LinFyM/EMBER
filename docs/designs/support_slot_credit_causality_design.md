# 一项任务替换为何改变另一任务：有害新增、保护丢失与共同学习的有限干预

2026-09-25。是否active只看progress。机器合同`configs/support_slot_credit_causality_v1/experiment_spec.json`。
本批已完成并关闭，独立原件复核与事前预测裁决见findings§145；下文保留原登记，不构成自动续训许可。
主讨论01a0cd94-65da-7b22-8ca9-7ba35f454632负责判断；现有Sol 01a0cd90-ebb7-77a1-a20b-a858825d2f66执行并主动Queue完成。

## 1. 要改变哪项判断

关系支持第一阶段没有兑现绝对修复（findings§143）。后继N/W交叉400条已独立核验（§144）：
Goal N0W0/N1W0/N0W1/N1W1=39/38/19/28；换W在两种N下均有害，N1在W1下反而补偿。
后验真实OSC首轮分解中，W变化在50/50初态都沿黑碗→盘子的方向，N大多反向；非目标物先移动也主要是盘子。
目标任务是把碗放到炉面，不能把盘子称为正确目标。几何不等于抓取/内部语义归因，混合未共同训练仍有交互。
W包括全部外部读取/映射/输出，不能从该结果挑某层冻结。

S00→S10唯一task替换为77（白碗放盘子右侧）→76（白碗放盘子上），场景/初始分布定义相同但真实episode/actions不成对。
两个完整训练端点同时改变了“删77”和“加76”。至少三种解释仍能产生现有结果：

- H_add：76当前监督通过共享生成器，把不适合Goal21的动作偏置写进条件映射。
- H_remove：77监督原本抵消其它任务的偏置；换掉它后失去保护，76不一定主动损害。
- H_path：主要是更早的表示/Adam共同适应或其它共同任务的演化；短窗口的这个slot未必复现。

本批不只是再证明参数共享可能干扰。旧§44已在另一模型/单实例证明过有限跨任务伤害，§127更表明删除其它任务梯度并非一般修复。
新的判别对象是这个受控77→76替换的**方向和来源**，其结果决定下一步该研究有选择的功能保持、保护性支持，还是否定晚期局部信用解释。
§129 freeAB/privateWriter及H54反例、§118–119有用视频过程路径均保留；不由本批授权缩head、扩大rank、删N或笼统独立训练。

## 2. 计算机制与可反驳预测

记完整Writer参数为phi，生成器G_phi(l,V)，真实动作函数为
`F_phi(q,l,V)=Flow10(q; theta_source+G_phi(l,V))`。Source及normalization冻结。
第一个被干预更新前，三臂拥有同一模型、m/v、scheduler与相同的先前4次更新。
令g_c为该macro其余3任务按原1/4权重求和的当前梯度，g_77/g_76为slot的1/4主FM+1/12辅FM梯度。
用U表示真实AdamW、clip、decay与scheduler更新，三种更新是

`phi'_drop=U(phi,m,v,g_c)`，
`phi'_keep=U(phi,m,v,g_c+g_77)`，
`phi'_swap=U(phi,m,v,g_c+g_76)`。

共同m/v内仍含历史77信息；drop不是遗忘77、重置momentum或不更新模型。所有臂继续全部optimizer steps，不给其余任务重归一化。
一阶说明为`F(phi'_r)-F(phi'_drop) ≈ D_phi F (phi'_r-phi'_drop)`；
在固定预条件SGD近似下为`-eta D_phi F P grad_phi L_r`，而`grad_phi L_r=J_G(r)^T grad_Delta L_r`。
这说明训练条件r的函数信用可能经共享Jacobian作用到另一个条件；符号不由任务图连通性或参数norm决定。
真实Adam的分母/clip也随当前梯度变化，不能把梯度cosine、线性近似或上述式子直接当验证。
本批用真实有限更新后的10-flow与闭环检验方向，不求虚假的精确一阶归因。

在同一q/teacher/noise下定义新增作用`A=F_swap-F_drop`、保留作用`R=F_keep-F_drop`；
因此替换作用`F_swap-F_keep=A-R`是有限函数的精确恒等式，无需线性假设。
首轮xy按canonical OSC裁剪/缩放后，取初态黑碗→plate的单位向量u，报告前5命令平均投影`b_r=mean_k <F_r,xy(k),u>`。
这是动作命令坐标中的描述量，不是body运动或抓持标签；位姿只在冻结诊断分析使用，不进Writer或梯度。

预定裁决：

- H_add：首个slot更新后`b_swap-b_drop>0`，末点Goal的swap相对drop有更多plate先动、更少成功；keep-drop可小或反向。
- H_remove：`b_keep-b_drop<0`，keep相对drop保住闭环，而swap-drop接近零/方向不稳定；此时不能叫76主动有害。
- 两者可并存。区间含零不是等价证据；所有方向、幅度及不确定性完整报告，不按任意显著性选一种故事。
- 三学习臂相近而都对父变坏，支持优先追查共同学习/历史状态，不能归因该slot。短窗没有效果只削弱这个起点和4次slot曝光的局部解释，不能推翻1260步端点事实。
- 有功能偏移但无闭环损害，只说明传递了动作变化，不能声称找到失败根因；只有闭环差而方向不符，则本plate偏置预测没有验证。
- Object14是另一类获取/保持参照，不要求出现同一plate方向，也不能由Goal修复牺牲Object来宣布方法通过。

本批为机制实验，三臂均不是候选部署方法；没有以混合checkpoint、task ID或特殊语言条件部署。未来改法必须另验有益视频与能力保持。

## 3. 共同起点与唯一学习变量

父模型只用C_S00@1155的**完整**checkpoint：
`/data0/user/ymdai/ember_runs/relational_support_causality_20260924/training/C_S00/checkpoints/macro_00001155`。
训练实现7dc95edb；2026-09-25已核对model、trainer、两rank RNG载荷存在，manifest next_macro1155/world2。
选择1155因其是末端1260前最后完整保存节点，尚有未执行于本分支的原登记事件；没有看该节点闭环后挑点。
父模型、optimizer、scheduler、两rank RNG全部复用；全Writer与三Meta继续共同学习，不冻结W或N。

- 三臂均从1155执行原全局更新1156..1183，固定28步=4个完整task轮；不得延长或换节点。
- KEEP77：完全复用原C_S00这28步的事件及权重。
- SWAP76：仅以原C_S10中相同occurrence的76事件替换对应77；其它27任务逐条原事件不变。
- DROP77：仍以原C_S00事件运行；仅77这个event的主、辅**当前梯度权重**归零，保留其它event已经积累的梯度。
  不跳过整macro、不清空所有grads、不重置Adam，不改变其它任务1/4主项及1/12辅项。可照常执行该event前向/反向再令其贡献零；避免由随机流变化间接改变其它事件。

主讨论已从两原training_events.json逐条CPU核对：28macro/112event/3136query slots每臂，27共同task各4次，共108相同事件。
四个slot出现在宏步1160/1165/1175/1178，occurrence165/166/167/168；77的teacher28/15/0/34，76为18/3/43/36。
两任务实际视频、action episode/frame与noise使用各自原事件，不强造逐动作配对。每个macro其余事件保持同一顺序和完整RNG定义。
KEEP/SWAP各3136非零权重query；DROP有3024非零权重query、112个slot query权重0（若照常计算则总执行slots3136）。
总84学习macro，9296个非零信用queries，上限9408个执行query slots；这些计数不是新独立meta-task数。

这是从完整状态派生的**受控fork**，不是把修改后的采样器冒称原训练exact-resume。
特别是SWAP的76 occurrence165是预注册供体事件索引，不表示父模型已见过165次76；不得伪造曝光历史。
新branch sampler保存自己的0..28 cursor、原macro与event引用、gate权重；本branch中断恢复保持world2及完整状态。
旧run与checkpoint只读，不改原config/max_updates/allowlist或旧JSON。
训练梯度仅来自S00∪{76}、且每臂按上述slot有效白名单；diagnostic-held8及官方Val/Test无梯度。
部署Writer输入仍只有exact language和action-hidden RGB，stride5/full H50/一套38-target rank16 LoRA均保持。

## 4. 最小评测与功能证据

四模型：未更新父P1155与三臂固定1183，无checkpoint选择。四模型在同一新实现下各生成100个correct条件，
评测global14/Object local4与global21/Goal local1各50state，共400新闭环。
state0..49、teacher映射seed20260911、policy/env seed7、每task每臂50条视频各一次；canonical50×7/10flow/前5执行等不变。
16固定full病例=四模型×两任务×state0/25；全部400保存compact actions和T+1对象/EEF/夹爪/BDDL谓词，复用已验收passive入口。
正式评测pilot8=每模型两任务state0，计入400；通过工程验收后执行剩392，绝不按pilot成功率改规模。

训练第一被干预更新1160后，每臂对两任务state[0,10,20,30,40]做一次冻结真实10-flow，共30次预测。
使用crossed_video study已保存的这10个t0 query（按query_index定位，真实图像/8D state及stateless noise），
teacher取原state映射，30条件均由该臂1160 Writer重新完整生成；禁止用父bank或新的伪造图像/query替代。
三臂都未在这两任务上用expert标签；这些预测不产生梯度、不挑节点、不启动rollout。
原1260端点动作不充当该首步比较的内部参照。报告全部30个full50×7、前5真实OSC及A/R/替换恒等式；五state只支持有限局部读数。
最终400条首轮动作直接供最终函数比较，不另加预测面板，后续不同轨迹时点不冒充共同query。

监督是否实际作用：四冻结模型各对合法non-held76/77、固定teacher46、16个同task不同episode查询（只从47..49），
做128条固定随机tau/fullH FM读出（**不做额外10-flow**），两任务使用canonical diagnostic_batch与seed20260925。
每模型每task只生成一套相应LoRA；同task四模型复用query/noise。此处允许额外读取76/77诊断actions，但无梯度，且从不读14/21 expert actions。
各学习event已记录真实非零/零信用、FM及optimizer/clip。FM仅检查局部监督响应，不代替有益视频、任务掌握或闭环成功。

每task报告三学习臂两两及三臂对父的六个配对成功差（共12项）、R/G/L/churn/Jaccard；
50-state联合bootstrap20000、seed20260925，单训练seed、未校正多重比较。不要用非显著当等价。
保存连续几何的原有1/2/3cm持续5步口径，以及Goal的plate先于目标移动计数（其它非目标物单列）、25步目标距离/高度。
主结论是新批内部比较，原1260分数独立历史栏，不把旧行拼成新模型分数。
不补other/wrong/seen/held400、其它节点、官方Validation/Test或原3300预测；没有给数据或方法下全局定论。

## 5. 工程、资源、停止和交付

Sol从最新main隔离开发，复用当前SupervisedEngine、query/FM/Writer、checkpoint loader和canonical evaluator。
允许一个窄的registered fork/gate接口和科学provenance适配，不复制整个trainer/evaluator，不放宽旧exact-resume/bank guard。
结构变更应用code-architecture-gate；旧冻结树/原件均不热改。所有新正式训练、物化、预测、闭环统一新clean pushed detached提交。
父训练来源单列7dc；无新增必须同commit的旧资产假要求。工程诊断错误先修复，不把失败包装为科学non-pass。

正式前重点验证：原两事件流的108公共event相等；DROP只抹掉该event的贡献且不伤其它已累积梯度；
world2求和/权重、restore的模型/m-v/step/LR与双rank RNG、Source冻结、所有活动模块梯度和原生视频信息墙；
真实完整LoRA物化→这两个目标与被动采集接口可用。CPU测试按新稳定合同需要，不为过程凑数量。
最多4条工程闭环common non-held58 state0（可每模型一次），原训练构建/最长full-H profile可复用同架构证据，实际改动再作必要smoke。

三臂正式训练pilot至1160（每臂5步，计入28）；保存完整状态，验证第一slot已生效及分支恢复后继续至1183。
所有完整保存点只需1160与1183；1160兼作30函数预测，不做中途闭环。可先完成短训练再统一物化/分析，不能因pilot函数读数改训练。
工程smoke权重不能替代父1155初始化；中断从本branch保存点恢复，不重跑已经完成的正式评测行。

新root `/data0/user/ymdai/ember_runs/support_slot_credit_causality_20260925`，新data0≤8GiB、新data1代码≤768MiB。
父资产只读复用；400+30+8套bank约2.2GiB、六份完整branch checkpoint约0.85GiB，加trace及临时件预计<6GiB，Sol须实测核对。
launch前在strg01核对两filesystem独立quota及共享容量；每次launch同时live核对gpu01/gpu02，项目合计≤6物理卡。
训练每臂world2、单节点，不跨节点拼；有六张有效卡可三臂并行，否则串行复用，不等凑卡、不抢占他人任务。
总正式预算≤8 GPU-hours（训练/物化/诊断/闭环合计）；按前批吞吐粗估2–3 GPU-hours，工程开发时间另算，真实profile更新一次ETA。
保持原NCCL_P2P_DISABLE/NUMA配置；正常任务一次持续等待退出，不轮询日志/cache。记录一份实际命令/env/设备/预算launch合同。

达到1183及400闭环、30初次slot预测、128FM读出后主动Queue主讨论，停止新增实验。
若来源/梯度/RNG/finite/restore/passive trace或信息墙有缺项，暂停受影响部分保留原件并说明；其它独立合约工作可继续。
资源逼近上限、明显不能在预算内完成时回报主讨论，不自行加样本/步数/seed/LR或删对照。
输出analysis/completion.json、逐branch事件与更新记录、原bank/预测/rollout/trace、12对比/成功集合/功能与几何及资源缺项。
主讨论再判读：本批可验证局部学习信用的有害/保护方向；仍不自动授予统一根因、修复或新架构成功结论。

# 强MT起点的原生视频条件控制：一次训练型分析

2026-10-07。Owner最新要求“我暂时没空跟进，你仔细分析清楚现状，分析完后先自主推进”。
本稿经progress登记后是唯一active批次。它检验一个训练期直接控制参照，不是新canonical Writer，
不改变最终RGB视频＋exact language一次生成单套完整LoRA的目标，不自启后继编译或完整重训。

## 1. 已有失败、具体假说与可改变的判断

当前T的合法视频路径有实质作用，强T/MT的原strict400为161/153，但尚无广泛稳定优势。
最近同scene seen144中T109、MT93、关系G84/F66；真实动作NN91及T50接力91有部分控制，仍不能作为普遍可靠教师。
这些分数来自不同计算干预，不当作单一变量的梯度尺。固定NN、校准Gamma及旧关系编译修补均不恢复。

具体行为不足包括task12的取瓶入篮、29的瓶到架、32的开炉后搬壶，以及38的双壶放置与保持。
main直接归约原144行，MT在这四项分别2/0/2/1、T为3/2/1/0（各4行），合计5/16、6/16；
29/init32的NN接力能形成On而原T失败，38/init32接力只保持第二壶On、第一壶未完成。
这些证据说明任务内容、当前状态上的控制和完整多目标保持尚未接通；不将所有失败归因选物体或固定LoRA。

最近似源码的实际边界很关键：Video Functional虽联训教学Meta/encoder/Reader，Reader的执行query仍来自
未适配、no_grad的source；关系F同样绕过全部PEFT adapter，用裸source H0/v0接7维残差，公共A/B属于独立G。
9/26层9 q/v Reader只做工程更新，正式学习撤回；10/2 live/stop从T2340出发，只训P/C/D/O与末端gamma，公共A/B固定。
因此“直接Reader都已证明无效”不成立；反过来，联合视频表示、进入原生中层、增加自身信用也绝不是新原理。

本次有限假说：**从已有完整控制函数出发，让教学memory在各执行层直接参与状态相关的内容选择，
并让完整执行A/B与该读取共同学习，可能比同数据继续训练MT取得额外控制，且部分迁移到新模块未训练的任务。**
它针对旧F仍需在冻结裸source特征后重新学控制的问题，撤去一次参数编译这一中间限制后检验所需条件函数。
这同时改变了初始化后的计算类和学习参数化，不唯一隔离“哪一层”“预训练”或“softmax”为根因。
源码未覆盖这个组合只是历史边界；投入依据还包括旧F本身弱、T已有有益条件作用，以及上述真实控制缺口。

竞争解释是新增模块只学一般控制修正、只适合训练映射、原生视频内容仍不足，或新增更新损害原共同能力。
以匹配MT继续训练、额外训练留出的四task、同task另一视频和完整得失来区分；不凭loss或非零梯度宣布修复。

## 2. 两臂与固定数据

- **M**：从强MT300的唯一完整38-target rank128 A/B继续学习，无教学输入。
- **V**：相同MT300 A/B初值，加下节唯一视频encoder/全层Reader；A/B、encoder和Reader共同学习。
- 两者共享冻结aligned Source1000、冻结图文prefix、source normalization和官方执行预处理。
  MT权重为`/data0/user/ymdai/ember_runs/coverage_retraining_20260920/training/mtbc/checkpoints/step_00000300/lora.safetensors`；
  只读其原run_contract/checkpoint_manifest与`configs/pi05_lora_rank128_aligned.json`，不叠第二adapter或复制大资产。
  source/数据协议复用`configs/pi05_source_aligned.json`与`configs/libero_24_8_8_coverage_v1/manifest.json`。
- 原授权36任务不扩张；本次新增训练只用显式fit32：
  `0,1,2,4,5,7,13,14,15,17,19,20,21,22,25,28,34,35,36,37,42,43,51,55,56,62,64,73,95,96,97,101`。
- `12,29,32,38`只作本批新增训练留出读回，不进入两臂新query/teacher训练。
  **MT300已用过这四task的原示范；这不是从所有训练阶段留出的新任务，不是official validation/test。**
  它只检查新条件模块和本次参数更新能否跨越其新增训练任务；不能据此宣称最终未见任务泛化。
- fit32每task16个teacher事件，共512事件。每8个宏步遍历32task一次；每宏步4task、每task28个不同episode query，
  均不等于该teacher，共112 query/update、固定128更新、每臂14,336 query使用。任务等权，不挑失败或成功样本。
  复用FormalData的分层确定性构造：root seed20261007，visit0..15；task permutation种子`[root,0,visit]`后分四task组，
  每task 50-video permutation种子`[root,1,task]`取第visit条；query RNG种子`[root,2,task,visit]`，
  query demo从余下49条无放回取28，frame从`0..length-2`均匀取。
  flow seed继续用`task_logical_batch_policy_rng_seed`（optimization_seed7/task/visit/demo/frame）；两臂逐事件相同。
- HDF offset1、真实50-action chunk、原末动作补齐与冻normalization保持。唯一loss是完整50×7原生随机tau FM，
  全部28条用现有`flow_sample(noise_endpoint=False)`；不加tau1项、前5加权、KD、动作/物理辅助或RL。
  教学仍只读双RGB＋exact language，stride5含原末帧，K1；无teacher action/state、实体、ID或结果输入。

## 3. 唯一实际计算与梯度

令β为完整MT A/B，基础权重冻结；V所有新参数fresh seed7，M/V的β初值相同。
教学先用现有`operator_writer.native.read_native_video`的真实双图文prefix与原生动作读取取得
`Hβ[V] : T×50×1024`。probe沿现有CPU独立Generator(seed1729)的标准高斯`[50,32]`、tau1、无teacher state合同；
绝不用零图像、缺prefix或teacher动作query。教学读取期间不安装下方执行Reader。
每个条件保留全部T与50槽，不做帧/horizon平均、选帧或缓存跨参数版本的H。

```text
U[t,j] = Linear_1024→256(RMS(Hβ[t,j])) + PE_time(frame_index[t]/5) ⊕ PE_horizon(j)
C = LayerNorm(TransformerEncoder_2layers(U.flatten(time,horizon)))
h_l^+ = 原生第l层(自身prefix、flow输入、β、h_(l-1))
R_l = Wo_l MultiHeadAttention(Wq_l RMS(h_l^+), Wk_l RMS(C), Wv_l RMS(C))
h_l = h_l^+ + R_l,  l=0..17
v = 原生final norm及action_out_β(h_17)
```

PE为固定标准sin/cos（base10000），time/horizon各128维拼成256，使用真实frame位置和0..49动作槽；
编码器宽256、8 heads、FFN1024、GELU、pre-LayerNorm、dropout0，两层独立参数、末LayerNorm。
全教学可见，self-attention双向；有序性通过真实二维位置进入，不把具备位置就当已学操作顺序。
Reader每层独立8 heads、总宽256；Q:1024→256，K/V:256→256，O:256→1024，均无bias，
RMS固定eps1e-6且无仿射；18个O均零初始化，其余使用PyTorch固定seed7常规初始化。
每次真实denoise在每个完整Gemma expert block输出后、下一原生层前加一次残差；
不是在frozen prefix加token，不修改source基础权重、MLP或图文prefix的参数。
若当前消费者并不调用该block边界，须在同一数学输出边界接入并记录，不能改成任意层位或仅末端头。

初始函数在R=0时等于MT；这是初值性质，不是训练保持保证。β完整A/B均可学习，rank不扫。
教学H与自身h都使用同一当步β；H没有自身state，h有官方自身观测，两者不会因共享β自动语义对齐。
V真实梯度为查询侧`∂L/∂β|C`加教学侧`J_Hβ^T J_encoder^T ∂L/∂C`，并更新encoder/Reader；
M只含原生query侧。允许按condition先计算C叶子、累积28query的C余切与β/Reader梯度，
再按同一参数版本重放H/encoder；不得detach教学链、重复计权或在更新后使用旧C。
同condition的各层K/V可预计算并缓存；训练若在K/V处分块，必须累积其余切并向K/V投影、encoder、H/β完整重放，
不能只保留Q/O梯度。推理时同一固定C/Reader版本的K/V可跨replan复用；这是同一函数的执行优化。
各condition权重1/4，物理microbatch按真实样本数/28加权；全任务/更新权重不随设备数改变。

两臂各fresh AdamW：lr1e-4、betas(.9,.95)、eps1e-8、wd1e-4、统一clip1，无scheduler；
学习参数及optimizer状态用FP32、source原生BF16/TF32计算按现有支持执行，不为低位一致牺牲吞吐。
β从MT复制而非fresh重置，故本批不称fresh Writer或原MT exact-resume。
完整保存0来源、64/128的β/新模块、optimizer、无scheduler声明、sampler/cursor、rank RNG、拓扑及schema。
64仅恢复证据，不物化/闭环/选点；唯一科学终点128。无多个seed、rank、LR、层位或步数扫描。

训练语义检查针对真实消费者：零Reader初值恢复MT的同一FM输出；第二步后真实教学/Reader信用、source冻结；
activation-checkpoint重算期间Reader必须仍生效，query与teacher作用域不能串；物理分块保持同一逻辑流。
小CPU合成图可验证memory余切分解，不做全参数bitwise比对或另造测试框架。
工程最多额外4个实际宏更新/448 query（独立V smoke状态，fresh2后从完整ECP续到4，不重复已执行更新，不能混入128）；
M初值/信用利用同FM消费者和首正式步核验，不另开M小训练。另以最长fit训练video的同一个28-query condition，
最多三次完整forward/backward（84 query使用，0更新）比较物理microbatch/frame chunk吞吐与峰值，不增科学样本。
不新增接口环境episode；以正式面板最先完成的合法行验证capture，保留其科学行、不重复运行。

## 4. 一次固定读回：304个新episode

M128与V128各跑原seen144：原36task各init32..35、原scene/teacher映射/root20260928，
复用`operator_seen_task_diagnosis_20260929/attempts/scene_canonical144/MT/evaluation/correct144/results.json`
与T原144的场景、env/policy RNG和绝对noise时钟，不重建一套方便的初态。
V另跑新增训练留出四task的same-task-other各4条，共16，合计304新行；M无video输入，原对应16行直接配对复用。
other使用canonical `expert_manifold/video_schedule.py`的同task +17 ordinal规则，逐行与correct不同；
分别从完整50无放回映射切出这四state，每task每臂4个不同video。此为有限诊断，不称paired400或整轮50-video覆盖。

V在rollout前一次编码教学得到C；执行每个flow步只查询固定C，不重读视频、不优化、不读teacher标签。
**C在执行中被消费，V是非EMBER部署的直接条件控制参照，304行没有最终方法选择资格。**
接入显式memory-controller消费者，保存共享β/Reader及每condition的C来源；不能用假zero-LoRA bank伪装成Writer评测。
M仍是单LoRA官方policy；两臂均用render256/model224、双相机rotate180、state8/action7、10 flow、执行前5、
settling10、原suite horizon、成功即终止。无GT在线输入、时钟阶段路由、动作记忆NN或第二expert。

各臂correct所有36task/init32保存full，其余compact；other四task/init32 full，其余compact，
共76 full/228 compact。全部保存T+1 EEF/quat/gripper/body/官方谓词和真实动作、full50提案及实际执行前缀。
事前固定查看12/29/32/38的M/V correct/init32共8clip，以及四个V other/init32共4clip；
每clip八均匀已存时点×双RGB，共192图。只陈述实际可见内容，不把抬高、原点位移或单谓词叫完整技能/抓持。

报告逐task/suite、target24/support12、fit32/本批留出4、breadth、R/G/L/churn/Jaccard及成功集合；
完整比较M↔V、各自↔原MT93/原T109，并展示四留出任务的correct/other，所有失败保留。
T来源仍为`denoising_return_writer_20261006/readouts/parent/seen/ODE/evaluation/results.json`的132行
加`query_conditioned_transition_read_20261005/parent/evaluation/seen12/results.json`的12行，不伪称独立新144。
强T161/MT153的official400只作既有完整背景，不新跑official validation/Test、wrong、shuffled、reversed或no-video。

## 5. 结果如何改变投入，不能自动接成编译方案

主要比较是V相对匹配M及未更新MT的完整能力。只有V取得有覆盖的实际新增控制，且留出任务/另一视频
没有与之矛盾的严重退化，才提高“值得进一步建立直接条件控制参照”的支持；只胜退化M、少数偶然行、
训练loss下降或只在fit任务变强都不够。原T109是额外强能力参照，不因更容易胜MT93而隐藏。
M和V同幅改善主要支持继续训练共有部分；V只在fit变强会削弱新读取自然迁移的预测；两臂退化则记录保持代价。
任何结果都在这批终点停止并由main消费，不按分数自动扩训练/层数/width/数据或启动蒸馏、Writer、400。

V优于M仍混合新增条件计算/容量与视频信息，尚无learned language/static匹配对照，不能声称video动态必要性已证。
本次四task只对新增训练留出，16/16的面板亦不足以认证总体泛化；即使有正例，也不能把fit和留出分开拼成资格。
若无有用优势，停止这套强MT全层Reader有限假说的默认延长，不将“未收敛”作为无限保护。

固定LoRA在单个投影处是`ΔWh`；此Reader的softmax权重随h改变，其局部输出一般非线性，
不能将C直接改名为LoRA、声称精确可编译或证明全网LoRA不可表示。原生LoRA策略本身仍有深层非线性。
本批若成功，只建立一个具体可用函数及后续传递的目标；最终一次参数生成的学习、可表示性、
held能力与保持仍须另行裁决，不能以更自由的诊断控制器替代EMBER目标。

## 6. 工程、资源、所有权与一次回报

唯一root：`/data1/user/ymdai/ember_runs/native_video_control_diagnostic_20261007`。
预计4–7小时（工程2–4h、两臂训练/编译/304读回及消费约2–3h）；硬上限从实际承接起10 wall小时、
12完整GPU-hours、data1新增峰48GiB，包含所有失败/加载/profile/训练/编码/EGL/分析/临时与冻结代码。
依据为既有450关系批50,400query＋1088读出约12.99GPUh；本批28,672query＋304读出规模更小，
但18层memory读取成本尚未测定，预算是上限而非虚构profile结果。先由上节限定profile核实际吞吐/显存及分项ETA；
若预计不能在总界完成则停止于已有完整边界并回报，不缩科学图/改loss/超界续跑。8wall或9GPUh时仅在有超界迹象时回报。

建大root前实查strg01 data1独立quota、个人/相关目录实际du和共享容量，估计checkpoint、memory、capture及代码峰值；
全部新增只在data1，历史data0只读复用，不新增hash/大资产复制。
每launch前两节点live准入，执行AGENTS的项目总8卡/空闲≤10时总6卡/单节点6卡，含训练、编码和EGL。
按真实吞吐选rank与物理batch/frame chunk，利用显存余量；M与V就绪后可独立并行，不因一个world size压低整批并发。
不跨节点拼训练、不占卡等待；NCCL_P2P_DISABLE=1、NUMA/deferred NCCL、拓扑迁移按既有合同。

实验session独占工程/Git窗口；从最新main建隔离分支/worktree，实现复用native、数据、FM、LoRA、canonical evaluator/queue/capture。
新临时模型/训练消费集中于一个内聚owner包，薄入口；不复制整套policy/evaluator、不让operator_writer大文件继续膨胀。
按code-architecture-gate记录实际所有权/增量，实际消费者检查通过后自行集成push并用clean detached冻结版本运行，
不等待main做工程验收。已定位、不改科学语义的修复可在原预算闭环，旧代码/失败/费用保留，不热改冻结树。
本批停止后关闭临时入口、专用hooks和可恢复执行面，模型来源/Git/frozen/checkpoint/原始行与报告保留；
main若支持后继再登记新的唯一active合同，不保留默认并行fallback。

正常长任务直接等待退出事件，不轮询共享log/cache、不逐阶段Queue或自通知。只有整批完成或实际边界
一次回main，交回canonical tracked/Git，给出代码/训练/读取身份、完整原件/成本/资源退出及未知。
Owner总体目标尚未完成；main接回后独立分析并决定下一步，不把这份计划当成已经运行。

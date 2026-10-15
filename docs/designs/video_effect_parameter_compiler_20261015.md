# 视频关系目标与真实动作干预驱动的参数编译

2026-10-15。Owner已明确恢复自主科学推进及必要有界实验，并要求持续推进。本稿由主讨论定案；登记到progress后成为唯一active design。它替代whole_method_reassessment_20261015.md §9尚未实施的终局Q-only议案，不恢复旧G/ImageCompiler、关系G/F或局部J加权FM训练。

当前执行状态：§14/§14.1 N16有限恢复已于2026-10-15T06:15:43.573Z完整exit0并通过原件验收，当前无active计算；结果交新主讨论科学消费，无自动后继。§13及此前合同、阴性和成本保持历史，不恢复旧批。

监督—原生算子—梯度—新初态迁移的完整科学说明见[学习链分析](../analyses/video_effect_learning_chain_20261015.md)。这是本合同解释，非另一active design或附加执行门槛。

## 1. 所解释的不足与本次实质修改

强G的427/576、165/400及BA端点175/168的正证据保留；多实践没有稳定胜过首G、纯V/VL的seen拟合增长没有带来held增长也保留。不能再用更多固定E曝光、去L或一次局部视频依赖来代替反馈怎样形成有用参数修订。

上一稿的关键缺口是：给定充分状态、任务目标和继续策略，终局成功Q无需视频。把V作为Q的输入，没有指定教学中的操作内容接受什么监督。新方法将两个学习对象明确分开：

1. 共享视觉模型把合法RGB/L读为有序的手、对象及关节关系；其语义由合法non-held物理标签直接监督。
2. 对新condition，视频关系序列定义一个可计算的观测模仿目标。真实自身动作分支给出这个目标及最终成功的实际差值，原生policy的真实导数把差值写到完整LoRA。

不新增需要凭稀疏终局标签自行学出斜率的critic，也不先恢复/搜索正确动作轨迹再作FM/BC。动作干预是取得真实后果的实验输入；没有拟合动作目标的后端。

竞争解释至少包括：旧方法仍受视频关系获取限制；已有内容可以获取，但共同E与候选排序没有教会当前状态下的净改进；当前策略附近的短程改进即使真实，也不足以形成跨初态的完整能力。本批的已有消费者输出须能区分这些范围，不预先认定唯一根因。

## 2. 完整运行图与所学内容

    action-hidden双RGB视频V + exact language L
      → 冻结source的真实图像patch特征、合法文字embedding
      → 共享视觉关系模型Phi
      → 一条完整有序的预测关系序列和对象相关性
    强MT合并基底 + 当前完整38-target/r8 LoRA
      → 自身真实实践，保存合法自身观测、动作、状态和结果
      → 同一自身状态的成对动作干预，再由同一当前LoRA继续
      → 真实终局成功与视频关系匹配的分差
      → 原生10-flow前5动作对全部LoRA的VJP
      → 一次完整LoRA提案，实际再试、接受或拒绝
      → 按实际成功、无改进证据与资源预算停止
      → 固定一套完整LoRA；所有读取、评分、干预与优化退出

视频提供的内容被限定为可见的关系变化及其演示顺序，不称为已恢复因果必要程序。相同语言下，教学中间关系不同，会改变同一真实自身转移的评分；这里的标签本身依赖视频，而不只是网络输入多了V。自身试做回答的是当前身体、状态和参数怎样实现或未实现这些关系。

例如当前策略后续始终没能完成放物，全部分支的终局成功都可为0；若其中一支确实带动抽屉到视频展示的关系，另一支只有手移动，视频目标可以给出非零差值。它避免了终局Q的这一平坦标签，但不证明短期关系改善必然导致最终成功。

最终LoRA仍通过W_l h_l + B_l A_l h_l作用于自己的RGB/proprio/L和flow hidden。固定参数可产生依状态而变的控制；没有将教学时钟交给最终policy。当前状态上的修改向新状态转移取决于J_new delta_Lambda，须由未参与适应的初态检验，不能从VJP存在推出。

## 3. 继承历史与选择依据

- 关系G/F已用真实关系监督、跨episode FM和真值关系控制函数，G360/450为131/126，seen G84/F66对T109；其后准确关系替换也没有建立总体修复。故本批不声称首次使用物理关系。新的下游消费者是实际自身反事实结果和直接完整参数更新，撤掉关系到共同专家/动作残差的共享映射；感知输入也改为保留空间位置的真实图像patch，而不是旧50个action hidden。
- LocalField、joint action-effect及functional revision已有实际动作信用或原生J。它们不能支持“只要接通梯度就会学会”；本批的余切来自实际环境分支的后果，而非FM训练能量或另一个未验证的动作目标。
- 2026-09-13物理干预的局部J显著优于零预测，但整体误差比.29736未满足原<.25合同，Goal/Long残余大。本批不拟合/复用该六维、五步共同偏移的线性物理模型；连续分支覆盖30维时变动作，直接使用完整续跑结果，夹爪另以真实离散分支处理。
- 10月10日低维Q讨论中的控制族限制、未识别导数、全失败平台和遗漏元梯度反例保持。此处没有额外z-decoder或共享适应几何训练主张；直接优化完整因子，仍承担其Jacobian、双线性和跨状态冲突。
- 旧Product、P/I及局部内容移植不为本批背书。本批不构造跨task的正确动作标签，也不把去L的阴性当成换成跨场景监督的依据。

不选择新learned Q/世界模型作为首批消费者，是为了避免把尚未识别的动作斜率和教学贡献同时交给一个新网络。真实分支成本较高，但每个分差有可核对的实际来源。若后续要摊销这些成本，须依据本批实际数据另作判断。

## 4. 唯一数据范围与内部迁移划分

固定当前coverage split及原36个train24+support12 allowlist，不新增独立meta tasks，不改变validation/test划分。

本批内部probe按每个target suite的训练global ID升序取前两项，预先固定为[0,1,12,13,20,21,32,34]。它们全部是既有train24，禁止称为正式validation/test。

这里的任务留出只针对新Phi的共享监督；source及强MT过去对这些任务的暴露照实保留，不称为完整系统从未见过这些任务。

Phi共享监督只用其余28项：target [2,4,5,7,14,15,17,19,22,25,28,29,35,36,37,38]，support [42,43,51,55,56,62,64,73,95,96,97,101]。每task固定demo16..23共8条，合计224条；同task等权、同task内episode等权，不按帧长改变任务权重。

复用relation_grounded_writer_20261006/labels中对应的既存位置、旋转、关节、语义、mask与registry；按原对齐合同读取，不从RGB伪造sim state，不重建整库。先核实际schema和这224条覆盖；若缺失则回报具体对象，不静默改demo或任务。OOI相关性是新加的训练标签，只从这28项原公开对象关注列表映射到既存实体registry；region owner只作为相关实体，不假装其为区域内/成功谓词。

内部8项的demo0/1仅作为action-hidden教学条件。其离线标签只允许在所有适应、最终读出及模型冻结后，供§11预注册只读诊断使用；不进入Phi共享训练、任务内评分、LoRA更新或checkpoint选择。固定validation/test的任何离线标签均不读，新Test使用为0。

### 4.1 实际schema修订（2026-10-15，主讨论裁决）

224条真实几何标签及stride5/末帧身份齐全。公开OOI region若真实owner存在但被原registry table/floor规则排除，则该任务8条的整组OOI监督mask；实际仅task25的main_table_stove_front_region→main_table。保留224输入和所有其它几何/变化标签，不丢任务、不映flat_stove_1、不造区域/table实体、不读success谓词。实现按已核实arena owner排除事实判定，不设置task25路由；未知owner或解析错误仍raise。

OOI有效条件216/224，各组原1/8系数保持，missing组可导地贡献0，另7组不重归1/7；mask不记0误差或负标签。label metadata保存region/owner/exclusion/groupmask，相关性只作为可表示关注对象proxy，不称所有因果相关实体有完整标签。probe标签仍在最终冻结后sealed读取，同类缺项届时如实报告，不静默替换。

## 5. 视觉关系模型的具体监督

复用canonical视频加载、stride5、双相机方向及render256/model224处理。直接取source真实embed_image的512×2048投影patch；不运行假图像、假action query或缺原生prefix的suffix。语言用冻结source合法token embedding，padding不进注意力。source和MT均不接受梯度。

Phi：patch线性投影到256维，加固定两相机/16×16位置编码；一个hand及32个匿名entity query，经两层cross-attention/FFN读取图像和合法语言token，再对每个slot沿完整真实帧索引作两层双向时间attention/FFN。width256、8heads、FFN1024、pre-LN、dropout0；时间位置只进Q/K，使用frame_index/5的RoPE。实体槽跨视频共享，不输入task ID、实例编号、GT数量、registry或OOI。

hand预测p3、R6、双指q2；匿名槽预测p3、R6、semantic64、presence、none/slide/hinge、两类joint q和OOI relevance。R6按现有Gram-Schmidt形成旋转。语义标签沿原seed20261006冻结词向量投影；不重做语义词典或引入held标注。

整条视频作一次Hungarian匹配，用有效帧均值的位置、旋转和语义代价；它只决定loss索引，不能成为实际教学输入。presence表示该物理实体存在，不能把它叫可见性或可靠接触。32槽沿既存36场景审计容量，不能截断更多实体。

损失按以下八组等权，各组先对有效实体/帧均值，再对condition均值：位置p/.25m的SmoothL1；旋转.25||Rhat-R||_F²；1-cos语义；平衡presence BCE；joint类型CE及有效q误差组内等权（slide/.25m、hinge/pi）；hand双指q/.04m；平衡OOI relevance BCE；相邻有效帧的平移/旋转/joint/hand-q变化误差组内等权。变化沿相同物理单位归一化，不按任务重新估计normalization，不把静态预测的低总误差当动态准确。

原末帧无完整state标签，保留其真实RGB进入时间模型，仅mask不可用标签；不补0或复制上一真值。对称、遮挡与不可唯一读出的姿态保留为模型限制，不把输出当完美感知。

fresh seed20261015；AdamW lr3e-4、betas(.9,.95)、eps1e-8、wd1e-4、clip1。固定20个完整epoch，224条/epoch，逻辑B4，共1120次更新；每epoch固定seed派生的task-balanced排列，每task恰8次，各epoch顺序预先生成。无额外损失、动作标签、FM、教师LoRA或RL。

唯一用于编译的共享点为1120；560只作恢复与学习轨迹记录，不做闭环选点。完整保存Phi、optimizer/scheduler（常数LR亦显式）、sampler/cursor、rank RNG、topology/schema。物理卡数只影响吞吐，完整边界可按仓库合同迁移。

## 6. 从视频到自身关系匹配评分

Phi固定后，完整读取每个合法教学condition，得到按真实顺序的预测关系。新condition不读取其teacher labels。

自身实践侧的实体表只来自自身真实环境的物理body/joint registry、当前pose及已执行后状态，不读取teacher registry或teacher episode状态。自身语义采用同一个公开类别/通用部件名称投影；不用实例数字、task ID、OOI/goal程序作匹配输入。

每次规范初态按teacher首帧与自身初态，对预测presence≥.5的匿名槽和自身实体作Hungarian匹配，代价为1-cos语义加位置/.25m的SmoothL1。位置仅帮助区分同类实例；不是不变性保证。该次rollout的映射固定，不能随结果改对象。若没有可用实体，不伪造视频分数，记录该condition无法形成目标并交付MT；这属于本方法的科学失败，不换阈值补救。

相关性权重w_i为该预测slot全视频presence×OOI-relevance的均值，固定用于整次评分，不由自身成败或分支结果修改。hand固定存在。使用以下关系组：

- hand到实体、实体到实体的世界坐标相对位移/.25m，分别按w_i与w_i w_j加权；
- 相应相对旋转的chordal距离；
- 相关实体的世界竖直高度/.25m（保留抬升），slide/hinge q按既定尺度；
- hand双指q/.04m及相对世界重力的姿态。

精确分六组后对有效组等权：①hand→entity和i<j的entity→entity位移，按w_i或w_i w_j加权，除以权重和，每条edge的xyz先均值；②同一edge集合的相对旋转，按同权重取.25||Q_own-Q_video||_F²的均值；③各相关实体高度的加权SmoothL1均值；④teacher预测joint类别argmax为slide/hinge且自身匹配实体确有同类joint的q误差，加权后按.25m/pi归一化；⑤hand两指q误差均值；⑥hand的R^T e_z单位向量误差(1-dot)/2。所有SmoothL1采用beta1。没有joint的组mask，不补真值。未匹配的预测实体单列记录，不以其建立虚构edge；若没有任何匹配实体，按上文直接记录目标形成失败。

所有常数固定，无任务-specific程序、手工阶段、抓稳/接触伪标签或按结果挑关系。这个共同物理坐标/关系表示只减少部分初态差异，并不保证任意场景、对称物体或同类实例对应正确。

令c(i,j)为自身第i个实际观测与teacher第j帧的关系距离。以标准三邻接单调DTW从(0,0)对齐到(n-1,m-1)，D_ij=c_ij+min(D_i-1,j,D_i,j-1,D_i-1,j-1)，边界只允许合法路径；得到d=D_end/max(n,m)，视频评分S_V=exp(-d)。不截教学、不按自身成功删尾帧；所有真实replan观测及实际terminal post观测参与。没有将teacher索引传给最终policy。

完整实践评分为R=2*success+S_V；无视频评分参照为R=2*success。success来自自身实际环境；预算截断不伪造terminal/success。S_V只是有偏观测模仿目标，不是原任务的potential-invariant shaping，不宣称最优策略不变。它可能奖励不必要动作、错误感知或某一示范路径，必须由实际成功、能力保持和另一正确视频裁决。

## 7. 真实分差如何产生完整LoRA更新

固定merged-MT基底，初始为现有非零A/零B的完整r8 identity。每次更新的数据来自当前已接受LoRA在四个适应初态的完整真实episode；不把旧MT/G的无snapshot记录叫当前可恢复反事实。

每个episode从其真实决策点均匀抽1个，记录采样概率和实际replan数T_e。恢复该点完整v4 snapshot及相同前缀，所有分支保持原absolute control step、相同后续policy noise和同一当前LoRA。分支只改首个最多5步动作块，随后立即由当前LoRA原生执行至官方结束；不得使用专家继续结果。

唯一联合分布（主讨论2026-10-15实施前修订）：raw native前5×7均值a；连续30维u~N(0,I)、夹爪五维v~Uniform(0,1)，二者独立，p_i=sigmoid(a_grip_i)。z^A_i=1[v_i<p_i]，z^B_i=1[1-v_i<p_i]。四个实际分支为Cartesian product(s∈{+1,-1},k∈{A,B})：continuous=clip(a_cont+s*.1*u)，gripper=2*z^k-1。之后均由同一当前完整Λ及同未来native noise继续。每格实际executed mask m_{s,k}独立记录。

以与新u/v独立的父完整episode R_parent为baseline：g_cont=(1/(4*.1))*sum_{s,k}[(R_{s,k}-R_parent)*s*u*m_{s,k}]；g_grip_i=(1/4)*sum_{s,k}[(R_{s,k}-R_parent)*(z^k_i-p_i)*m_{s,k,i}]。未执行45步永远不是后果标签。每格边际来自同一个Gaussian连续动作×Bernoulli夹爪首块联合探索分布，四格反相关只改变方差；不把相关另一支当独立baseline，不用环境普通导数或flow density。

若z^A=z^B，同s的严格相同物理请求可复用一次实际结果，但统计权重仍四格等权，unique费用另记。不通过近似参数或结果合并请求，不新建hash机制。

同一父episode的所有分支使用相同固定前缀参与DTW，不能只比较后半段而丢失已完成关系。完整评分的共享teacher/映射固定。分支全部实跑和计费，不拟合局部物理J或用critic预测替代。

对四个抽中点重算真实完整10-flow、50×32 noise、前5×7输出，经过冻结source unnormalization取得raw a。梯度为四个episode的mean[T_e * (d a/d Lambda)^T g]，保持每episode等权的决策和，而非按总帧数重新加权。Phi、评分、环境和分差在actor更新时均停止梯度。

实际优化器最小化负的mean[T_e * dot(a_raw, stopgrad(g))]，方向为上述分数上升；没有a_star、MSE动作目标、成功动作回放拟合或teacher action恢复。连续和夹爪两项为同一首块联合探索目标的score梯度估计；继续策略与当前访态在此局部估计中固定，最终部署不加探索噪声，故不是整个未扰动部署策略的精确梯度，仍以候选完整再试与新初态裁决。

采用现有PolicyContexts.prox给出的每因子固定单位S，在完整坐标u=(Lambda-Lambda0)/S上作Adam（lr.1、betas(.9,.95)、eps1e-8、无weight decay、global clip1），更新全部76个因子叶；不新增低维控制码。首步A梯度为0、B有梯度是正确双线性行为，不要求76个tensor首步都非零。

一次Adam提案作函数变化限制：每个父episode预取四个均匀分位决策点，加该次梯度点去重，最多20点；仅用同点原native noise比较提案与父前5×7实际raw动作。每点RMS变化≤.1且finite才执行提案；否则将本次参数增量依次乘.5，最多4次。仍不满足则拒绝该次提案。此为预注册优化步限制，不是寻找最好成功率的尺度扫描；不做额外环境试分挑scale。Adam状态在被拒绝时一并回退，保留拒绝记录。

候选在同四个适应初态、同environment/policy RNG实际完整再试。先按四条真实成功数比较；成功数更少拒绝，更多接受；相同则视频臂仅在mean S_V严格增大时接受，无视频臂拒绝相同成功数。只用确定的数值容差1e-6避免浮点相等误判。该规则不保证新初态单调；最终评测不参与接受/拒绝。

接受后使用候选真实H与完整LoRA进入下一次；拒绝保留父及其H。连续两次没有可接受提案（包括真实零分差或函数限制拒绝）、当前四个适应初态全成功、或资源不足以完成下一原子比较时停止。次数由实际证据及预算产生，不预设观看/实践轮数菜单。

## 8. 首批完整循环、参照与固定读出

8个内部probe，各两个合法视频demo0/1分别独立编译，另有一个无视频评分参照，共24个适应condition。每个condition独立LoRA/optimizer/RNG/经历，绝不融合两套video LoRA。

所有适应使用init16..19，所有最终读出使用init0..15，严格分离。编译起点、动作干预分布、停止规则、最大预算与正式控制接口相同；无视频臂保留exact L、强MT和全部自身反馈，只把S_V置0。它是本方法的匹配去视频评分参照，不冒充充分优化的强language-only RL。

实践noise root固定202610150，最终root202610151；各自按现有task/state/absolute-replan派生。不同臂初始种子配对，后续因实际经历不同而按已发生事件推进，不按结果重新抽seed。两个视频可共用完全相同的无视频参照，但统计上不把它当两份独立证据。

同task的初始MT四条实践与snapshot只物理执行一次，供两个视频和无视频condition各自读取/评分；分别记录共享输入成本与全批unique实际成本，不假装发生了三次。其它严格相同parent身份、snapshot、实际干预、noise的请求也可由一次物理执行供多个评分消费者复用，但不按近似参数或结果复用，不新增hash machinery；无法直接证明身份相同的请求独立执行。以下资源总量仍是未扣共享复用的安全上界。

最终每condition只部署已确定的一套完整38-target/r8 LoRA。24×16=384条最终episode，另MT同8×16=128条强参照，共512条；若存在精确相同case和noise的已保存MT可复用并明示，否则本批真实执行。不得拿其它seed/面板MT冒充配对。

完整报告demo0、demo1、无视频与MT各128，per-task/suite、breadth、R/G/L、churn/Jaccard、两视频的成功集合重合；再报告每condition实际预算、接受/拒绝、最终所选参数身份。两个视频各固定一条用于16新初态，明确这是有限训练任务内部诊断，不是400个不同condition、formal checkpoint选择或正式held能力。

不取两视频最好结果、不挑checkpoint、不使用shuffled/reversed、不追加held/Test。第一完整批结束后由main消费结果并继续科学决策；没有等待Owner批准的停点，也没有自动开展更大矩阵的许可。

## 9. 资源上限与时间预期

每condition上限96次环境start提交（恢复分支也算）、64×该suite官方horizon个实际control steps；正常dummy settling另计并单列。原子更新须先预留16个分支及4个候选episode的剩余最坏步数/提交数，不因预算不够改成挑部分成功分支。

24项最多2304次适应提交、506880适应control steps；最终512次提交、168960 control steps；加最多16次工程失败提交及按520计的8320最坏control steps，总提交上限2832、control steps上限684160。dummy settling最多28320步另列，总动作步上界712480。正常提前成功和拒绝按真实数计，提交上界不叫精确reset数。

全批最多16次工程失败，不自动扩充；未知未回执按完整horizon保守记账。科学阴性、全失败和小收益不是工程重试理由。预算若因实现/真实资源不能容纳，main按具体缺口裁决，不缩面板冒充完成。

新ROOT=/data1/user/ymdai/ember_runs/video_effect_parameter_compiler_20261015。新增峰值估96GiB，包含224条真实patch缓存、16条probe教学、Phi/optimizer、全部必要自身H、物理分支trace、完整LoRA、冻结树和临时输出。原大资产/label cache只读引用，不复制。正式准备和launch前核strg01 data1独立quota、相关个人用量及共享容量；不能借旧配额数字开跑。

总卡数遵live两节点8/6、单节点6规则；按实测吞吐选共享学习rank/特征worker/动态环境slot，不将训练world固定成整批上限。首次有效消费者验证更大物理batch/frame chunk/worker的收益与峰值，保持逻辑B4、相同任务权重与同一科学矩阵，不填充无用计算。最多两次丢弃的首合法训练批profile，之后复位全部权重/optimizer/RNG；所有开销计入。

首次派发起暂估准备和完整计算3–6小时。依据旧256条/69603步在六卡16slots的872秒量级及新约71万步最坏上界，并为感知学习、真实native反向、工程和保存预留；没有新消费者实测前不把它当保证。首次有效训练和完整适应单元后只修正物理时间预期，不按分数改合同。GPUh没有累计硬限，仍准确登记全部费用。

## 10. 实现归属、验证与生命周期

实验session独占代码/Git窗口：从当前main创建独立codex worktree，完成真实消费者检查、main集成/push，再从新的clean pushed commit建立detached冻结版正式运行。主讨论只作科学分析，不重复工程测试；冻结代码不得热改。

复用现有Runtime、PolicyContexts、native_actions、EpisodeRunner、EnvironmentSlots、snapshot v4、official evaluator及持久long-first动态队列。新增职责为共享关系读取/学习、视频关系评分、完整参数优化的编排；不得另造Actor、环境engine或并行Writer fallback。首块动作override、恢复后t0及terminal物理读出属于canonical执行器的窄扩展，真实native proposal、外部干预和clip后executed须区分。

数据侧区分四类：28项共享训练标签；probe教学RGB/L；自身合法状态/动作/结果；冻结后特权只读诊断标签。实际模型调用必须看得见这条信息墙。所有最终feedback只报告，不进入共享/任务内梯度或参数选择。

验证聚焦真实端到端消费者：一个共享训练更新实际接收geometry/motion/relevance信用；一个当前LoRA的真实恢复分支确实只改变首块并由同Λ继续；真实分差经官方10-flow到完整因子；候选接受/拒绝和完整恢复；最终只用一套LoRA。完成必要shape/finite/asset/配对/预算检查，不新增hash、逐tensor低位一致、全树扫描或镜像实现的测试。

结构所有权由code-architecture-gate约束。可把本批必要新职责放入一个内聚的scientific owner并保留薄入口，但不得堆进已有大型run.py或复制旧关系G/F运行面。旧ImageCompiler专用训练/分析入口已无本轮用途，须在确认无当前依赖后退休；共用Reader/runtime/历史frozen及formal原件保留。若实际新增超过1000 active行或5个source文件，交付明确所有权及退休理由，不机械拆文件或请求Owner例行审批。

执行者对已定位的非科学接口违约自主修复、针对性核验、新push/freeze接续，保留失败和成本。若改变目标、标签、评分、更新/选择或矩阵，先报main具体科学边界，不擅自修改，也不把main裁决误作Owner许可门槛。

一位owner持续等待实际子进程退出，只在整批完成或真实边界时可靠唤醒实验session；实验验收/Git后一次回报main。无固定轮询、阶段ACK、重复自Queue、第二等待者。正在独立推进的主讨论不陪跑空等。

## 11. 与完整结果一起消费的预注册分析

只使用本批已经产生的学习/实践/分支/最终原件，不另开局部probe：

1. 共享训练各物理字段及相邻变化误差、OOI/presence，避免把总体loss下降称为操作理解。
2. 每次实际分差的终局项和视频项、当前真实分支对、native更新幅度、信赖限制及候选实际得失。不能用非零梯度替代候选能力。
3. 所有编译与最终读出冻结之后，内部8项demo0/1的旧合法几何标签只作一次sealed只读诊断：在共同有标签的教学前缀，分别以预测关系和真值关系重算同一已发生分支trace的评分，使用同一匹配与DTW规则；末帧缺失范围、GT presence/相关性等特权区别明示，不把该值反哺本批。
4. 报告真实分支的评分差方向一致率、效应量与不利例；预测评分与真值评分矛盾说明当前视觉目标可能误导。两者都改善而success不改善，支持模仿目标/短程控制与完整任务仍有缺口；适应初态改善但最终初态不改善，限定为未取得迁移。不得由这些分支唯一证明一项总体根因。
5. 保留每task完整成功/失败与能力交换；选有代表性的已保存真实RGB/物理过程解释至少一个不利个例。行为标签限于观察到的事实，不把接触、共动或靠近自动叫抓稳和理解。

首批没有资格只交一张分数表，也不要求所有中间指标先过门才运行完整循环。若完整收益弱，应下调本次“视频关系目标＋自身分差＋直接完整参数优化”的主假设，结合最早失效接口决定后继；不得默认沿rank/LR/seed、DTW权重、感知辅助或局部探针无限续补。

## 12. 原批settling协议违约及128固定MT修订

原批1556提交/1556返回，原128 MT实际dummy0而报告10；384编译final实际dummy10且保留。832 snapshot分支恢复前多出8320 dummy为真实成本；恢复完整状态和结果保留，不称原协议完全合规。原MT64及相对MT−4/−6/+10撤出科学结论。

只新增原8 internal probe×init0..15共128固定MT，原source/MT、env/policy root202610151、official dummy10、suite horizon、双相机预处理、native10与执行前5动作、原pairing均保持。针对性CPU消费者后new clean pushed detached；0新profile。新提交最多128，不自动重试或扩大；保留有效行，超出报告具体余量。原16工程失败额度不扩大，也不挪用原适应余量。

不重跑384有效final、24套完整LoRA/Adam/RNG、Phi1120、cache、32bootstrap、832 snapshot分支或任何sealed诊断。原控制303472，实际dummy修正14280，总317752；新增control最多42240、dummy1280。累计提交上界1684、control345712、dummy15560、总动作步361272。原4.793367211752468 GPUh完整保留，再加修订真实费用；该范围是单独授权的协议替换读出，不追认原MT为合规。

参数和选择21:01:01.939804Z冻结；21:21:15.600653Z已经完成只读GT诊断；修订MT在其后。明确sealed顺序例外，不能再声称GT是在修正有效512全部完成后才读。GT不得影响参数、选择、case或MT配置，不重开或重算sealed。

原results/final_freeze/complete/sealed/128invalid MT原行及费用保持原件；修订产物单独保存，唯一有效结果由384原有效编译final+128修正MT构成512有效配对。实际final执行640而非512成本。训练8801e701、原评测8801e701/fe034f78、新修复读取代码身份分列。

原2026-10-14T18:16:48.556Z/3–6h不重置；修订从本裁决起25–45min估计，live双节点6/8-node6与data1独立quota/个人用量继续。一个owner只在整批或实际边界唤醒实验，验收/Git后一次交主讨论；无普通ACK、阶段Queue、第二waiter、自通知陪跑或自动后继/Test。

主讨论补充的已消费身份事实（不新增复算）：no_video14次更新/0接受，8个final均MT300；74/128不是无视频学习增益。video0/1分别20/18更新、8/7接受；实际改参数仅task0/1/12/21/32，其80行各36成功，对同case no_video45。task13/20/34三臂同MT300但48行成功24/22/29，首noise/proprio一致，保留正常执行不确定性，不能归因视频LoRA损害或以逐tensor/dtype检查/补测消除。128修订MT保持独立正式参照，不选择或平均多MT高点。

## 2026-10-15 自身关节schema接口违约；仅未来入口窄修

主讨论从原sealed与源码发现own geometry将MuJoCo2 slide编码成2、3 hinge编码成1，并按kind−1写q列；原teacher labels/scoring合同为none0/slide1/hinge2、q[0]米/q[1]弧度。未来canonical bind映射修正为{2:1,3:2}；实际MuJoCo enum/既有label owner及bind→observe CPU消费者通过，0新模型/环境/GT评分。
该字段沿physical_trace/mapping→joint可用mask/有效组分母→DTW/S_V/R→四格credit/native全因子VJP/Adam和接受/后续parent进入原video条件，旧评分、梯度及编译不能称完全执行原合同。原physical joint_type/q和全部H/LoRA/分差/sealed不改写、不重算；不把未来修复当作旧批已经正确。Phi1120/正确teacher schema/patches/sourceMT、真实snapshot/动作、无视频score及已冻结参数final仍可复用；当前128MT补测physical=False不受该字段影响，沿a986e7b2继续。解释性旧类型1↔2和q两列对换有精确schema来源、无值猜测，但未实际应用；是否重编译待main独立有限合同。原snapshot多8320dummy及全部费用另账保持。影响原件ROOT/analysis/engineering_completion_20261015/joint_schema_impact.json；不称此错误已解释低分唯一原因。

读取原保存的Phi预测与固定mapping（0模型/GT/DTW重算）确认影响还包括task0/1，合计task0/1/20/21/32十个video条件、27更新/432分支原score、12接受。原这些mapping所有teacher帧joint项不可用；纠正类型后对应全部帧可以参与。task12/13/34自身与预测均none，不受这项mask错误影响。该事实说明实际评分监督范围少于合同，不能据此推定修正后的R/梯度/接受或性能；完整影响文件已一次交main。

## 13. 受影响10条件一次有限joint schema纠正

2026-10-15，main 01a13399-cba2-7da2-9dc6-df430ca80280明确裁决。Owner持续自主推进授权下接续原video_effect_parameter_compiler_20261015，不等待Owner新许可。现有128固定MT补测继续；本合同授权准备和运行受关节字段错误影响的10个视频condition。不得据此扩展模型、超参数、任务、视频或评测矩阵。

### 科学理由与唯一变化

原teacher监督与评分合同明确none0/slide1/hinge2、q列分别为米/弧度。own bind错误把MuJoCo2/3映射为2/1，并按错误kind写q列。这使task0/1/20/21/32的两个视频条件（10项）原保存预测在全部teacher帧均失去joint评分项，改变有效组分母、DTW、S_V、真实四格credit、完整LoRA更新和后续接受。原27更新/432分支/12接受是实际错误实现证据，不是完整原设计的有效检验；不声称该错误是低性能的唯一原因或修正后必然提高。

唯一干预是恢复已预注册schema。Phi1120、原28/224监督、240 RGB patch缓存、16原predicted_teaching、所有评分形式/归一化/权重/联合动作探索/native10 VJP/Adam/信赖限制/接受/停止规则均冻结。不得新增loss、重训Phi、调DTW或权重、增加教学、换init/seed、做新profile或读取新held/Test资料。

任务集合仅由原保存预测与own字段的实际消费者依赖决定，与原分数高低及sealed GT差方向无关。task12/13/34的6个视频condition，原own和预测均none，评分计算不受这次修正影响；连同8个no_video condition、对应224有效final行原样复用，不重编译。原10个受影响LoRA及其160 final仍保留为原实现证据，不能按新旧谁高挑结果。

### 一次性输入解释与重新编译

新子根为 `/data1/user/ymdai/ember_runs/video_effect_parameter_compiler_20261015/joint_schema_corrected`；原ROOT及原frozen/H/CP/评分/sealed均只读。只对明确来自旧错误producer的自身physical字段作有来源的一次性导入：articulated type1与2互换，其q两列互换，type0保持；保留原row/路径、旧新schema、body/joint/qpos地址来源及转换说明。不得猜值、改teacher预测、改RGB/proprio/action/success/完整snapshot/controller/RNG或覆盖旧H。新canonical观察已经用正确schema，不能再转换一次。用必要的真实CPU消费者核对转换后trace与固定mapping一致，不做逐tensor/hash扫描。

五task各init16..19的20个原MT bootstrap物理episode有效且有完整v4 snapshot，只读导入；每视频condition各自消费四个原parent，重新在原初始场景绑定并计算正确基线S_V。32个原bootstrap总资产不重跑。10个condition全部从原同一MT300完整因子中心、u=0、空Adam、原抽样RNG开始；不从旧最终CP继续，不保留旧信用、接受或计数器。

复用这10项原首次更新的160个已实跑分支：原MT parent、snapshot、实际首块override、后续policy和noise身份可直接由原request/row/draw证明；仅其自身physical视图转换schema。保留原uniform point、u/v、四格实际动作及执行mask；重新计算正确parent基线和完整prefix+分支的S_V/R/credit。不能复制旧draw中的baselines或旧prepared proposal/Adam/trust/outcome。原首次分支合计20,336 control、原逻辑四parent合计9,458 control，均占每condition的实践预算，但不新增物理成本。旧分支恢复前额外dummy仍只计原实际费用，不因复用再计，也不从历史费用抹去。

后续分支、native提案和候选全部按当前正确编译过程实际发生，不导入旧非初始branch/candidate或旧已接受参数。新proposal使用独立完整producer身份，不能因文字ref与旧root同名误判为同一LoRA。原第一次物理分支不能安全复用时报告具体身份缺口，不静默重复160次或扩大本合同。

每condition仍96逻辑starts与64×suite horizon control上限，复用的4 parent和16首次分支在计数器中照常占用；保留现有每更新16分支、最多4候选以及完整原子预算预留。共享/导入不购买更多迭代。无改进、四成功、预算停止规则全部不变；旧低分不是重试理由。

### 最终读出与信息边界

每新condition冻结一套完整38-target/r8 LoRA，再读原init0..15、final root202610151、official settling10及所有原控制合同，共最多160条新final。若新condition与其同condition的原最终身份均明确为MT300，可直接复用该condition原有效16行：这是预先规定的同参数原件复用，不按结果选MT高点。不从新旧不同proposal同名、近似张量或分数相似推出同一参数；不新增哈希或逐tensor比对。其余受影响condition一律使用新冻结LoRA的读出，即使较差。

有效整批报告仍为video0、video1、no_video和修正settling的正式MT各128，共512逻辑行。构成为10纠正condition最多160新final + 6未受影响视频96原行 + no_video128原行 + 当前单独补测的128 MT。明确每行是新执行还是复用、原/新编译/评测代码身份，报告per-task/suite、breadth、R/G/L、churn/Jaccard、两个视频的重合、每condition接受/拒绝与预算。无视频14更新0接受、最终全MT300的事实必须保留；同参数重复执行差异不得叫学习或视频损害，不追加采样/逐位复算消除波动。

原sealed诊断已在旧批冻结后读取，且自身joint错误也影响该诊断。纠正发生在这之后，明确记为一次由既定schema合同与源码证实的协议修复，不能声称整个新批仍满足首次未见诊断的原时序。不得重开、重算或追加GT评分，不把原GT的分数/方向用于新梯度、选择、输入、分支或新参数。原内部8项不是formal held；正式Val/Test保持封存。原修正MT的sealed时序例外也保留。

保留旧results、旧complete和当前settling修订的corrected_results；本子批完整后以子根results/aggregate作为恢复原schema后的唯一方法读出入口，不能混写旧方法结果或抹成本。若纠正后仍弱，结束这次完整假设的投入并交main作完整判断，禁止自动续Phi、rank/LR/seed、权重或额外局部probe。

### 资源、工程与回报

五task horizon之和1560，10条件之和3120。新适应control最多64×3120−9458−20336=169886；新final最多16×3120=49920。有效新增最多10×(96−4−16)+160=920环境提交、219806 control。原编译计数每更新固定16分支，96额度最多容纳16完整候选episode/condition，因此新增正常dummy最多3200（160候选+160final）；snapshot恢复分支dummy0。

工程失败沿用全程原16上限而不扩充；执行者据实际事件确认剩余e≤16，未知保守按horizon520。连同这些失败，本修订新增绝对上界936提交、228126 control、3360 dummy，总231486动作步；少于16剩余时相应扣减。与已授权原批+128MT补测上界合并为最多2620提交、573838 control、18920 dummy，总592758，仍不挪用未登记的科学矩阵，也不扩大原16工程失败上限。真实费用按unique执行记，不能把共享逻辑预算当新费用或只报纠正后低成本。

新增峰值暂估24GiB，原30GiB资产只读复用、不复制cache/source/模型；launch前按strg01 data1独立quota、个人实际目录用量、共享容量更新预测。所有新增产物data1。每launch live核双节点6/8总量、单节点6与实测显存，不抢占外人任务；沿用已实测slots16/8及动态持久队列，不另开discarded profile，不固定task/GPU。MT补测仍有活跃GPU时合并计数，CPU准备可并行，真正就绪按吞吐安排。

本修订从本派发起暂估45–90min，包括窄导入/冻结/完整计算/验收。依据原受影响27 native更新合计3406.64秒、均126.17秒，最多六持久消费者及160 final，预留加载和工程时间；不是成功保证。原18:16:48.556Z/3–6h首次时钟不重置，所有旧错误和计算成本保留。

执行者独占工程/Git窗口，将本合同纳入当前design与状态，再new clean pushed detached执行；复用唯一canonical compiler与执行器，导入是一次性已知schema迁移，不维护第二学习路径。必要检查聚焦实际导入/评分/预算消费者，不让main重复工程测试。现有MT整批回报照常完成；本新纠正批只有一个实际owner等待子进程退出，只在整批或真实边界可靠唤醒实验后一次交main。无普通ACK、阶段Queue、第二等待者、自动后继或新Test。main收到完整原件后自主科学消费和接续。

§13工程落实：沿canonical ActualEpisodes仅由180个明确旧producer路径登记的schema视图导入，原数组/registry不写回；bootstrap和160首次分支自带MT300/RNG/override/snapshot实际来源。原prediction只读link，所有正式first draw保持baseline待重算，无旧proposal/CP/Adam。source只改compiler/run两个文件（+73/−12，净+61；409/273行、无新source文件/入口）；import load为旧schema一次桥，本实验持有，科学消费后退休其manifest迁移/旧baseline分支，历史formal frozen保留。canonical prediction缓存、独立producer namespace、arm集合与final同参数复用保持单一实现；final复用拆为独立职责，未扩大既有复杂worker/step。code-architecture-gate无hard、仅原复杂函数及必要draw迁移review信号，40–52行函数均≤60，未为形式拆Actor/engine。
队列以10个独立condition为adapt项，而非5个task串行两视频；20既有bootstrap已登记complete，0重跑或并写共享H；六持久consumer long-first承接10适应+10final，参数/预算/逻辑task权重未改。就绪原件可及时评测，不静态task/GPU，也不提前空占等待不存在的额外阶段。CPU真导入/scorer/credit用真实4parent+16cached分支，shadow draw不进入formal；native完整梯度与fresh Adam由本批首有效更新验证，0额外profile/GT。

§13实际完成：joint_schema_corrected_01于22:44:51.958Z exit0，执行afe6acc5 clean pushed detached；10完整38-target/r8 LoRA/Adam/RNG、26native更新18接受（5四成功/2两次无接受/3预算停止），CPU原件验收通过。唯一子根results有效MT67/V0 64/V1 70/no_video74；V0对MT R55/G9/L12，V1 R58/G12/L9，no_video全MT300/0接受，不将+7称学习。352旧行+144新final+16同condition MT复用组成512，累计实际final784；旧results/corrected_results/GT/预测均保留、不重算。新增500完整/98748control+2440dummy/1.873172839GPUh，累计2184完整/433706control+18000dummy=451706步/7.005042365GPUh，0失败partial/未知，仍在新旧联合边界内。六worker各slots16、19.002min，0profile，已退出；原派发、全部旧成本及sealed在修订前读取的例外保持。完整分布/配对/停止/资源/代码原件见子根analysis/engineering_completion_20261015/report.md及acceptance/paired_metrics，科学解释由main消费；无自动续块/GT/held/Test。


## 14. 16初态实践支撑：一次有界完整方法检验

2026-10-15。主讨论01a13399-cba2-7da2-9dc6-df430ca80280在完整消费joint_schema_corrected后作出本裁决；Owner持续自主推进授权有效。将本合同纳入现有design新§14并登记progress后实施，不等待Owner或main例行ACK。本批是科学后继，不是工程恢复，也不是正式held资格。

### 判断与主要干预

纠正后的完整结果V0 64/V1 70/MT67/no_video74，各128。16个视频condition共37更新21接受，实践成功34/64→45/64；11个真正修改参数的condition最终97/176，对MT92、no_video100。另5个仍MT条件37/80对MT42，其净损失不能称学习遗忘；no_video14更新0接受，+7也不是学习收益。完整数据而非该事后子集裁决方法。

纠正26更新中12次video-only、14次mixed，前者10接受且5次提高实践成功。视频提供了终局差分为零时的实际更新依据。但task12/video0实践1→3、最终仍8/16，task1/video0实践0→1、最终2对MT4；局部接受尚未形成可靠新初态能力。mixed的视频/terminal动作信用范数比中位.00732，不能将损失简单归因视频压过终局；该数不等于参数梯度分量。

本批只改变实践分布支撑的宽度：同一教学下，形成梯度、限制函数变化及接受提案都从原4初态扩大到16初态。实际梯度由(1/4)Σ_e T_e J_e^T g_e改为(1/16)Σ_e T_e J_e^T g_e；同一参数变化须同时面对更多自身访态。它降低只适合四个初态的提案通过机会，并改变经验梯度的状态覆盖与抽样方差；不保证新初态单调，也不解决感知/目标偏差。逻辑batch改变是科学干预，不伪称纯物理batch优化。

这是一项有限的分布支撑检验，不把16升级成永久方法限制，不形成4/8/16菜单。单condition预算只增为原两倍，较宽原子更新可能减少实际修订次数，真实成本与优化机会差异须报告；不把新旧差额称严格隔离的纯初态数量因果效应。

保留原弱证据并下调该方法的优先级。若本次仍只有实践收益、同参数波动或少数任务交换而无完整新初态收益，收束当前“固定Phi关系评分＋真实局部分支＋直接native/Adam参数更新”家族的继续投入，不自动再扩初态、续Phi、改DTW、调权重/LR/rank/seed或换optimizer保护它。旧Gaussian/RB、SDE、LocalField/functional revision及关系G/F的负证据继续有效；本批不复活这些路径。研究整体继续由main裁决。

### 唯一范围与复用

仍为原8内部任务[0,1,12,13,20,21,32,34]，各video0/demo0、video1/demo1及no_video评分参照，共24个独立condition。最终仍为各16初态，四臂512逻辑行；不减少不利任务、不挑视频。只使用既有Phi1120、16份原预测及同一source/merged MT；0共享训练、新教学、新标签、新held/Test或新GT/sealed读出。

每condition fresh MT300中心、空Adam、完整38-target/r8因子，不承接旧适应LoRA/optimizer。实践固定init16..31，最终固定init0..15，严格分开。实践noise root202610150、final202610151；原draw seed20261015/task规则保持，按init升序组织16 parents。N改变使随机流后续消费不同，保存实际流，不声称新旧整个draw逐项相同。三臂在初始MT时配对，之后随真实经历推进。

原32份MT bootstrap(init16..19)具备相同真实起点、policy/noise及snapshot，可按已有明确旧schema来源只读解释后复用；init20..31共96份新MT bootstrap按task物理共享给三臂。旧bootstrap在每condition仍占16项中的4项及原真实control预算，不以复用多买迭代。不要复制大H、patch或模型。旧分支/proposal不导入；不把相同字符串当参数身份。新原件使用独立namespace和subroot。

本批复用32份旧bootstrap所需的窄schema解释桥保留到此次唯一有限批科学消费后退休，原纠正批科学消费已完成。只登记这32个明确producer路径，canonical新H永远使用none0/slide1/hinge2且不二次转换；原180路径中其它旧first-branch导入与纠正专用调度无后继用途，应退休。历史frozen/原件不动，不建通用兼容fallback。

旧sealed已在原参数冻结后、纠正及本批之前读过，此时序明确保存。本次根据真实实践/新初态差距改变支撑分布，不用GT作为评分、输入、标签或选点；0新GT/DTW诊断重算。Phi的内部8任务留出与整个系统已经见过这些训练任务的范围分别说明，不称formal unseen tasks。

### 公式、接受和停止

Phi、mapping/六组距离/DTW/S_V、R=2*success+S_V及no_video的2*success均不变。每个当前parent episode均匀抽一个真实decision，共16点；每点同Gaussian连续30维×Bernoulli五夹爪Cartesian四格，64逻辑分支，均由同一当前LoRA继续。原u/v/p/z、各格executed mask、完整prefix、absolute clock、官方horizon及native noise合同保持。

四格score_credit内部的1/4不变；外层episode平均改为1/16，仍乘各自T_e。全部76叶经真实完整native10/前5×7 raw动作VJP更新，Adam lr.1/betas(.9,.95)/eps1e-8、固定因子单位、clip1均不变。无动作target、BC/FM、Q、世界模型、额外训练头或第二部署adapter。

每parent仍取原四个分位点加本次梯度点，去重后全16 parents最多80点；同noise比较当前和candidate，逐点raw动作RMS≤.1及原最多4次halve规则不变。不新增环境试分选scale。

candidate在相同16实践初态/RNG完整再试。成功数增加接受，减少拒绝；相同则video仅在平均S_V提高>1e-6时接受，no_video拒绝。被拒参数及Adam回退，accepted真实H进入下一次。当前16条全成功、连续两次无可接受提案、或剩余预算不足完成下一完整原子单元即停；不写固定修订次数循环，不因本批结果修改停止规则。

一次原子更新预留64分支+16候选的80逻辑请求及相应最坏control（沿原保守full-horizon预留）。同一物理请求严格可复用时四格仍等权，condition逻辑占额不减少；物理唯一费用单列，不能因去重/旧数据复用扩大更新机会。

### 固定最终读出与解释

唯一算法终点的24套最终参数冻结后，以原官方双相机/8D state/native10/前5/settling10/horizon及原mapping/noise评测。MT参照唯一复用已修订的正式128行（MT67），不重新挑选或平均MT/no_video高点。

若某新condition最终结构性仍为MT300（0已接受更新，或显式恢复原中心），预先规定直接引用同task/init的这份正式MT参照，不重跑相同政策。不能因某次MT旧执行更高而改来源；旧no_video74和旧V所有同参数差异完整保留为历史，不能改写为67。本批逻辑行与新物理执行数分别报告。修改过的参数不能仅凭ref名字或小norm当MT复用，不做逐tensor低位扫描。

主比较V0、V1各128对同一MT及新no_video；per-task/suite、breadth、R/G/L/churn/Jaccard、两视频重合、实际参数身份、实践前后16成功、每更新信用和成本完整报告。最终初态不产生梯度或接受反馈，0checkpoint union/择视频/controls/Test。

主要结果分支：两正确视频均有明显净增、合理保持与多suite覆盖，才提高完整方法支持度；只一条/少任务改善保留局部证据；只增加实践成功继续判迁移不足；no_video也改善则分清通用实践收益与视频增量。内部128不产生strict held400选择资格，也不要求这一批补齐全部视频因果controls。无自动更大批次；整批回main消费后自主接续。

### 资源、吞吐、工程与交付

每condition最多192逻辑环境请求、128×本suite horizon实际control steps；共享输入照占逻辑预算，正常dummy另计。全24适应逻辑请求≤4608、control≤1,013,760。新增final最多384请求/126,720control，MT128只读依赖；工程失败最多16请求，最坏8,320control。新批保守全局上限5008提交、1,148,800control；dummy保守≤50,080（实际snapshot为0、其它正常10），总动作步≤1,198,880。严格共享/复用只减实际费用，不扩大科学矩阵。全失败/无增益不是工程重试。旧2184提交、451706动作步、7.005042365GPUh作为依赖历史单列，不重计为新成本、不覆盖旧invalid原件。

新ROOT=/data1/user/ymdai/ember_runs/video_effect_parameter_compiler_20261015/practice_support16，先由执行者核strg01 data1独立quota/个人用量/共享容量再建立。新增峰值暂估80GiB：依据上批500新病例/101188动作步/5.767GiB，计本批最大步数、24完整参数/Adam/H/冻结树及临时量；无新大特征缓存。若实际元数据推算超出，在launch前报告具体差额，由main裁决，不写data0。

live双节点总8/6、单节点6上限保持。按就绪工作与实测安排六或其它有益数量的持久worker；64分支wave允许验证更大slots（如32），不能把旧最大wave16当本批永久上限。原16slots约15.54GiB只是旧实际消费者，新的native逻辑B16及最长输入峰值须从本批真实首单元核验；按现场余量、CPU/renderer和实测选择physical microbatch/slots，保留吞吐及不继续放大理由。可用本批原登记首wave逐段增加有效物理并发，所有请求都是正式有用工作，不增加环境profile或dummy。GPU计算profile仅在必要时用原16点最多两次可丢弃重放，正式参数/Adam/RNG不承接，实际费用另计。

从首次本合同派发起暂估完整准备/运行/验收2–4小时。依据纠正批26个N4更新、500新病例、19分钟六卡；本批更宽native及分支成本明显增加，按注册请求和最大control外推，实际提前成功/拒绝/共享会缩短。不是deadline或正结果保证；首真实N16完整单元后据吞吐作一次必要ETA修订，不读分数改量。

执行者独占代码/Git：在现有canonical compiler/run所有者内把实践state集合、外层分母、atomic预留、成功停止和发布计数明确化；四格分母仍4，三臂数量仍3，不能混同。保持一个active运行面，专用旧corrected调度退休；不复制trainer/evaluator。完成真实consumer的N16权重、完整parent/candidate、预算/恢复与发布针对性检查，clean push、新detached冻结后执行。无需main重复工程验收。

完整checkpoint保存全部因子/Adam/流/RNG/N16 schema/参数及H来源，不能把旧N4 CP当新resume。狭窄工程违约由执行者按原科学合同修复并新push/freeze；科学目标/公式/信息墙/预算改变报main具体边界。

一个owner持续等待实际子进程退出，只安排一次整批完成或真实异常的可靠唤醒；不逐阶段Queue/心跳/第二waiter。实验完整验收、五份现有记录串行Git集成后一次回main；无自动后继或Owner再批准门槛。

§14科学消费入口：父ROOT/joint_schema_corrected/analysis/main_scientific_readout/corrected_method_scientific_readout_20261015.md及credit_analysis/{report.md,analysis.json}；主讨论已完整消费，执行不重复CPU统计/GT。首次派发2026-10-14T23:06:33.739733Z与2–4h预计保持。

§14工程落实：同一compiler/run两文件替换N4假设，N16 checkpoint/proposal schema与16个practice IDs严格恢复，四格仍4、16个VJP物理分块统一除16再clip/Adam；最大80信赖点按16物理batch读取。共享bootstrap独立8项只物理执行一次，24独立condition与24final组成56项持久long-first队列，全部24参数端点就绪后才开启final。只剩32旧bootstrap路径解释桥；旧first-branch/baseline导入、纠正同condition旧final复用及mt-final专属入口退休，历史冻结不改。固定MT只有预注册正式128参照，不用旧noV高点。
注册64分支内的有效wave比较slots16/32，并据实测steps/s与live余量决定下一64；native有效16点由4/8/16物理chunk逐段验证，不新增环境profile。CPU实际canonical代数/队列/MT引用/N4拒绝及真实旧MT snapshot视图通过；不把CPU标量oracle称实际native10通过，真实完整N16将在首正式单元核验。source增长179/删除88、净91行，仅2source/无新模块或Actor；结构无hard，原ActualEpisodes.run18、事务step24与worker20的review信号由清晰事务/执行职责持有，均未过增长>25边界；worker已抽出condition编译职责并缩小。派发23:06:33.739733Z/2–4h、全部旧费和时钟保持。ROOT/disk_projection推算74.35GiB<80GiB预估，source/cache只读link；每launch现场准入，GPU当前尚未启动。

### 14.1 中断后的唯一有限恢复修订（2026-10-15）

Owner最新指定实验执行01a13dbf-dbc3-7621-82be-87be44a5bca6，主讨论及唯一科学边界/整批接收方
为01a13dd9-837d-7f33-b15a-4eb170bcf04b（同BCI-GPU02），不再使用旧主讨论/实验回报路由。
新主讨论明确仅完成本§14，不开其它科学批；旧N4、全部正负结果与原派发/超期事实保持。
16 bootstrap和每次64分支+16 candidate的80请求，在192逻辑上限内实际至多两个更新机会；
这是有限支撑检验，不是充分收敛或纯N隔离，也不能由阴性否定全部自身交互学习。

原attempt814提交/743完成，71无回执且无保存row，保守未知control上界18485、dummy0。
一次性允许这71个明确原科学请求各一个恢复尝试；保留原16正常工程失败额度，后续失败消耗该额度，
不滚动扩充。累计物理上限修订为5079提交、1167285 control、50080 dummy、1217365总步。
71是潜在额外失败上界，不称已证实71失败；旧未知18485步单列，不用恢复结果回填原费用。
有效保存行不得重跑，不增加更新/数据/面板，不改变Phi、奖励、探索、接受、停止或最终512合同。
新GT/sealed/held/Test和共享训练仍为0。其它未登记未知、不可恢复科学状态或新边界直接报告新主讨论。

原六job无exit，按各launch到最后可证活动的时长给GPUh下界、到本次开机时刻03:30:32Z给上界；
保留各原件时间及假设，不伪造退出码或精确GPUh。旧依赖2184次/451706步/7.005042365GPUh独立不动。
恢复前保存原contract/队列及owner源码证据，核完整N16因子/Adam/流/RNG与draw/proposal事务，
只恢复死亡worker的claim；现存H/模型/预测原地复用，不复制大资产、不修改旧冻结树。
live双节点6/8-node6、实测峰值与data1独立quota/个人目录用量/80GiB峰值保持。

执行者独占记录/工程/Git，新clean pushed detached接续，主讨论此阶段只读共享仓库。
仅一个owner，实验直接等待子进程，删除本次新owner的阶段/自Queue回报路径；整批验收/Git后一次向新主讨论交付，
报告clean commit与记录写锁释放。原首次23:06:33.739733Z/2–4h及超期不重置；
本次恢复准备、余计算及验收暂估90–150min，依据已存首单元约6–11min、六持久worker与剩余条件量，
实际事务/资源检查若改变依据则据实修正。完整24condition/512逻辑行与固定MT128/67不变；
无自动扩此家族或新架构，科学接续由新主讨论消费后裁决。

§14.1实际恢复：05:00:51.161Z的practice_support16_resume_02/PID1360022采用clean pushed detached
78d4f73e，src与原7a146fc0相同；六job分别live双节点/总6-node6/data1准入，实际gpu02:1/7/2/0/4/6。
CPU真因子/Adam恢复、71 current draw/743保存行及四类有限额度consumer通过，GPU完整单元由正式消费者承接。
旧owner源码/contract/队列/71身份与费用区间保留于inputs/interruption_recovery_20261015；
原冻结树不改，canonical操作owner补修订额度/真实区间且取消自通知，实验直接等待退出后验收/Git/一次新main交付。


§14/§14.1完整交付（非新授权）：practice_support16_resume_02与六job全exit0；24完整38-target/r8终点、
因子/Adam/流/RNG、56项和512配对齐。MT67/V0 61/V1 71/noV62；48更新21接受，实践66→77/72/80（/128），
7仍MT条件按固定MT128/67引用，272新final+240引用。各条件仅2次更新，非充分收敛或纯N因果结论；
[完整报告](/data1/user/ymdai/ember_runs/video_effect_parameter_compiler_20261015/practice_support16/analysis/engineering_completion_resume_20261015/report.md)保存各task/suite、保持得失、两视频、全部条件及旧N4边界。
原743有效请求0重跑、旧71各一次恢复，新3305全部返回，累计4119/4048、已知666747+旧未知≤18485步，
有限上限均满足；原派发/已超期及全部旧费用保持。新7.459967182GPUh、旧无exit[2.346457453,21.469853089]，
含独立旧依赖总区间[16.811467000,35.934862636]。Owner内存排查发现空闲池高驻留，退出后used412→17GiB，
任务进程/CUDA已释放；CPU总驻留/idle释放遗漏如实记录，未热改冻结代码或新增profile。0新GT/sealed/held/Test/Phi或自动后继，
串行Git交付后释放记录写锁，完整科学消费/接续交新主讨论01a13dd9。

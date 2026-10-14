# 视频关系目标与真实动作干预驱动的参数编译

2026-10-15。Owner已明确恢复自主科学推进及必要有界实验，并要求持续推进。本稿由主讨论定案；登记到progress后成为唯一active design。它替代whole_method_reassessment_20261015.md §9尚未实施的终局Q-only议案，不恢复旧G/ImageCompiler、关系G/F或局部J加权FM训练。

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

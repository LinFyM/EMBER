# 36任务共同教师、评估经历复用与不同预算编译：main执行合同

Owner 2026-10-11约14:30北京时间明确要求立即自主推进；本稿由main定科学内容，实验session在独占工程窗口整合进唯一active设计/状态文件后，用clean pushed detached实施。旧四task G480仅自然收口，不恢复旧π矩阵。新批不再以四task试训筛方法。

## 1. 目标、范围与时间

- 数据/G/π覆盖既有审计train24+support12全部36任务，按task等权；固定Val8/Test8离线标签不读取，不新增任务或分割。Task36名单复用现有ALLOWLIST，libero_90支持任务official horizon400必须进入环境合同。
- 今天数据准备原目标4–5小时、最迟约5–6小时；实际基准从本次约14:30授权计，原训练数据目标北京时间20:30前。Owner随后明确质疑4/8初态精度，main消费gpu01直接实测后将选点/独立报告各扩大到16。原precision预测20:40–21:05；正式教师bank已于北京时间15:21:08从clean pushed detached dab2f725启动，工程/冻结实际耗时后最新预测20:55–21:20，比原20:30目标晚25–50分钟。这是明确披露的精度/时间取舍，不把时钟重置为formal启动，也不承诺已按原期限完成。独立教师报告在E锁定后与G并行，完成前不宣称已验证教师泛化。工程、清理、profile及数据等待均计入wall。
- UTC08:09资源/ETA修订：gpu01:5准入时free由21262降至3275MiB，task37原attempt未启动、GPUh=0；UTC08:07:46在gpu02:4接续原800×28链，其MT读出仍由原协调器负责。23个已完成200-step段实测gpu01四卡约2.363/2.869/2.415/2.797秒每更新，gpu02:6约4.482秒；取代此前快卡代理估算后，bank保守预测北京时间22:04。main已读取coordinator/task37_recovery_01/forecast.json；全36 capture尚未完成，早成功或共驻变化仍会改变ETA。该预测不含G训练或独立教师报告。原有效更新与失败记录保留，代码不热改，调度承接不能重复教师计算。
- 第一轮G/π用约1小时量级的学习窗口取得全任务初步证据；这是投入规划，不是声明1小时充分或变成逐小时批准。按实际吞吐冻结整数更新数/完整task轮次，结束前后预估偏差及时回main，不把4task或无限缩短教师作为偷偷赶时限的办法。
- GPUh只记录，无硬cap；原物理合计6/8、单节点6、live显存非干扰与data1独立quota保持。新增峰值按实际H压缩样本/CP/模型估计，不能复用旧64GiB而不核算。

## 2. 模型与参数消费者

保留source+已合并MT冻结基底，一套38-target/rank8任务A/B，共同A0非零/B0=0；两组fresh rank32读取meta、现有有序教学Reader、真实参数/经历编码、完整参数CFM G及独立三head π。当前参数的摘要与局部原始因子均保留，不新增7B权重压缩器/世界模型/辅助动作网络。

G输出更新因子ΔA/ΔB，替换为A+ΔA,B+ΔB；目标E与全部parent同基底同坐标。纯参数CFM为本批G目标，不新增rec/keep标签构造或直接动作辅助loss，不为产生多模态把弱教师混回目标。共同目标允许噪声样本收敛，不能把随机性当有用探索证据。

新condition所有共享权重冻结；其执行LoRA只由G生成/修订与π选择，不从自身或准备的动作轨迹做policy微调。当前完整策略的实际hidden/动作/后果进入H；教学meta不叠入执行。首次G前强制MT真实实践，计入同condition预算，无空H生成路径。

## 3. 一任务一教师轨迹，checkpoint评估就是H

- 每task一条持续任务LoRA训练轨迹，共同强目标为本批已登记读出候选中selection闭环最优E；包含MT节点，tie选更早节点。未评估的保留checkpoint不参与当前选点，不把当前E声称为全部保存节点中的已知最优。目标是整体任务能力，不再逐parent/H训练Q。
- main已消费完整更新/capture实测，保持每task800更新、每100保存完整checkpoint，当前仅MT/200/400/600/800进入读出与选点。100/300/500/700保留真实参数、optimizer、query cursor/RNG及manifest供以后直接补评估/经历或恢复；本批尚无对应H，不预先纳入G parent池、不按当前结果追加选点、不作为旧架构冗余清理。这是本批学习与时间取舍，不是Owner硬数字。有效query batch28、AdamW既定lr1e-4/betas(.9,.95)/eps1e-8/weight_decay1e-4/clip1、无scheduler。36条单轨迹共28,800更新/806,400 query，不能把物理microbatch变小写成逻辑batch仍112。teacher pure demo actionFM沿原native/mask/offset/normalization，保留对共同初始MT的既定1e-3坐标prox，不加parent-H rec/keep分支。
- Owner随后澄清：任务内验证资料的关键是未进入当前这次任务LoRA的参数训练，不是之前从未读取。新教师默认action demos0–45训练，46–49只作该教师无梯度任务内验证/教学；固定MT过去已见这些数据如实登记，不妨碍研究残差任务学习。共享G使用这些合法训练task的RGB视频与教师参数标签；不把离线示范变成自身经历，也不把Val8/Test8混入。历史旧教师训练池49/MT50保持原身份，不能倒写新留出。
- 保存节点后立即进入同task、同初态、同env/policy RNG的真实闭环读出；reader/teacher评估代码保存现成EpisodeRunner输出History，不重复collect。旧只有score/first_native的JSON不能重建H。
- 每节点16个真实selection/H初态，完整保留成功与失败；36×(MT+4CP)×16=2880 episodes。每个任务固定的16个初态及RNG在所有checkpoint共用；预先分成两个8-state折。E按全部16行成功数择优，tie取最早节点；有限面板排序噪声与不利节点全报，不宣称实测E是全局最优。无模型/训练效果后追加易初态。相较此前为时间提出的4行，本次增加决定全部G目标的配对证据；16仍不证明精确峰值、过拟合或充分收敛。
- 这些行明确uses=[teacher_selection,G_history,selector_supervision]、independent_report=false。π监督若复用两折，输入H只取其中一折，候选回报标签只取另一折；交换折可复用数据，但同一训练样本不把query回填H。
- E锁定后另取16个预留初态读E与MT，共1152条逻辑报告行，作为教师能力报告；不用于挑E或本批G/π/H，teacher报告阴性不重标成bug。训练bank与独立报告合计4032条逻辑行（实际新增episode扣除合法复用），角色分开、费用完整记录。报告在E锁定后就绪即运行，可与首段G并行；bank齐备不等于报告已经完成，不因报告未完成静默改变E或共享训练数据。
- 上项1152/4032是逻辑报告行数，不再默认全为新环境episode。MT固定参照先索引既有逐行原件，只补实际缺口；每行登记reuse/new及原始来源、真实新增成本。复用须保持同MT身份、task/init、官方执行与真实scene/env/policy RNG；让新E对应行沿用现有配对条件，无必要不另造随机根。当前旧canonical seen144覆盖全部36×init32–35，按未变的新16报告面板有52个初态交集；这些是候选复用行，真实scene/RNG消费者匹配后才可正式引用。其它既有MT原件继续按元数据合并，不按success挑来源。validation原paired400继续复用其原强MT逐行参照，不拿不同任务面板的总分替代本批36配对。没有相应完整H的旧score不能冒充G bootstrap经历；完整且合法的已有H也应复用。
- 评估参数引用必须真实且唯一，例task/checkpoint/base/rank的稳定ref；禁止跨事件复用parent/160/480导致History.states覆盖。原始文件名/ID只做索引，不进入学习特征。无需为G输入保存sim snapshot；只保存已有必要RGB/proprio/action/hidden/feedback/noise/结束原因/产生者，source raw-image embedding可以运行时重算/有界缓存。
- 出现晚期退化，后段P→E天然提供回退标签；若仍单调增长或持平，如实记录，没有必要也不允许人为弄坏模型制造过拟合。达到最大节点仍上升的任务不能称已充分收敛；可在同一条轨迹、当日时间余量内安排已登记的一段延长来判平台，不开LR/rank/seed扫描。

## 4. G数据、覆盖与固定尺度

每个样本为(L_t,V_tj,P_ti,H_tik ; E_t−P_ti)。同task所有parent/H共享E；P=E的零编辑也进入数据。H来自该P的真实节点读出，非空。多producer经历仅由真实episode及其实际producer组成，并准确标为监督重放，不冒称on-policy采集。首先用single-episode真实H，不为数据量合成假动作或假hidden。

全部36任务都到齐后才固定S_G并启动共享G；可用较少video/H样本训练，但task权重和覆盖不缩小。Owner最新明确按任务、再按实际parent等权采样：MT与各教师checkpoint平等，含E本身，不额外提高MT权重；每个parent内部均匀抽真实H，不能因其H数量不同改变parent权重。早期可用与更多预算的收益主要落实到下节π的训练，不用G采样偏置替代。S_G按同一task/parent等权分布对E−P的每因子RMS计算，floor沿既定1e-3 S_prox；不按H条数、audit或held调尺度。其后数据刷新保持S_G与坐标身份。K=1，教学RGB可使用本批登记的全部合法训练task视频；曾作内部留出的episode一旦进入G即不再称其G未见条件。

G从fresh meta/G初始化学习新目标（旧G240/480不是此合同exact resume）。按预排task-balanced轮次取样，样本量逐步增加，checkpoint保存全部optimizer/sampler/RNG/topology。输出target为相对当前原始因子的差，目标标签/teacher性能不输入条件。

Owner进一步要求真实的“修订—实践发现偏差—再次修订”学习，并提出无额外教师与G/π共同训练的可能性。main据现有控制教师正证据及旧经验/功能监督负证据，保留本批共同教师；教师不是理论必要条件，也不以正在运行作为保留理由。原“G监督初始化、一次学生补充、之后整个π阶段永久固定ψ”的训练组织退役。改为密集G初始化后，持续按当前编译器实际访问的参数/经历聚合监督，并交替更新共享G与π。保留现有CFM/完整参数出口和π三head，本次主要改变训练组织，不另造恢复教师、逐H目标、额外动作轨迹或并行Writer。

首个有信息的G节点后覆盖全部36任务采实际生成parent及其实际H，每task至少一次有效实际G候选实践。初始学生覆盖沿既有合同，随后复用下节π采集的真实长主干继续刷新G，不能只补一批后永久停止。新鲜路径始终用同一ψ_k、θ_k走完该逻辑batch；第一次G前仍先MT真实实践。每轮实际实践的P_i及截至当时完整H_i加入G监督候选，保留混合经历的真实producer。标签直接是同一E_t−P_i；E是标签侧，绝不输入G，也不按H重训E。较早节点只包含当时信息，失败和晚期回退均保留，不用最终成功筛掉困难状态。

G监督更新仍是当前已登记参数CFM：在真实(P_i,H_i)上以(E_t−P_i)/S_G为端点，真实CFM梯度进入全部ψ，包括双meta、Reader、参数编码和速度场；P_i/H_i/目标作为记录停止梯度，不假称反传MuJoCo，也不声称这是完整成功率策略梯度。固定目标不要求人为指定多步路径；若一次可正确生成E应鼓励一次达到，不插值出固定五段“正确修订”。共享梯度训练的是从自己实际错误状态重新修订的规则。新增H是否被有益利用仍须实证，数据聚合不构成收敛或经验利用定理。

每个共同学习循环的次序固定为：当前ψ/θ采集完整batch与独立query→在尚未改变ψ的前提下完成§5的一次θ更新→把真实新P/H并入监督bank，完成登记的ψ更新块→失效变化meta/context缓存→用新ψ/θ重新采下一batch。π的旧score路径不在ψ更新后重用；G可复用有完整参数/经历来源的旧监督记录，因为同一E−P标签仍有效，须使用当前ψ重新编码。整个训练称交替共同训练：π用真实query回报，G用共同强E的密集参数监督，不把π梯度穿过变动生成器而漏掉生成/实践分布导数，也不冒称G已经直接获得完整RL信用。

任务等权、task内全部登记的实际parent等权、parent内部H等权保持；学生、MT和教师节点不设人为类别比例，S_G/坐标不随刷新变化。新P/H只保留一份可追溯原件；π消费后的参数/H退休须检查G bank仍在引用的依赖，不能先删除再重采。可合法复用已有MT bootstrap H/查询记录，唯一环境成本与每condition逻辑投入分别记。首个约1h G初始化窗口及随后完整36-task共同学习窗口由真实消费者吞吐冻结更新数/采集数，不以“联合”默许无限交互或把未满一遍36覆盖称学习完成。

读出必须保存并固定同一时点的完整(ψ,θ)与两个optimizer/sampler/RNG/数据版本，不拼不同阶段最优G和π。训练已有各节点query用于观察修订前后能力和保持，不用参数距离下降认证控制进步。首轮新方法的实际新增H作用可以在冻结模型、同V/P/噪声的有限train条件上对照是否提供新经历；这是机制诊断，不是独立训练的无H算法或正式资格，任何额外行须在执行前明确面板及成本。若只有反复重采样收益而无新增H贡献，不能声称已实现反馈纠错；不会因此恢复逐H教师。正式结论仍按§6。

## 5. π的五轮训练展开、终止分支与真实停止

Owner进一步提出训练可固定迭代深度（以5轮为例），并追问停止头怎样训练。main据此采用首批R=5训练展开；它是当前共享训练采集深度，不是部署固定轮数或未来预算输入。退休上一稿用环境步区间加权出口的F目标和Cmax驱动训练终点。π及baseline不读取未来总/剩余预算；保留parent/STOP、practice、final三head、独立Reader/static；ψ在每次行为采集和对应θ更新内完整冻结，跨批次按§4更新。外部资源限制/实际步数/GPUh仍登记，实际训练首段更新次数按完整消费者吞吐冻结。

一轮定义为：在当时真实H/P上选择parent，冻结G产生现有最多4个提案，practice头选择一套实际实践，完成一条官方horizon/success终止的episode，再在已实践集合上输出当前可交付LoRA。MT真实bootstrap是第0阶段，不计为一次G修订轮；第一次G仍非空H。首批每条件训练主干实采5轮，最多20个G提案、16 Euler，最多MT+5条实践episode；不按旧4*(horizon+10)提前截断这5轮。每轮官方episode horizon继续生效，资源故障/外部中断不能冒称正常完成5轮。部署可在任一完成节点交付一套固定LoRA，也允许π主动STOP，实际上限与成本在评测前登记。

阶段A：教师真实prefix的离线final监督，随后必须加入当前G实际生成/实践候选；不强求有限旧教师重放恰有5个不同节点，不造H补长度。prefix只含当时真实H/P，教师H折与query折分离。q_i(θ)=sum_{λ∈P_i} π_final,θ(λ|s_i)*r_query(λ)，以条件内prefix均值/task均值训练final；不把此阶段称完整停止/继续决策RL。

完整共享RL采用“当前π条件继续采样，停止选择解析求和”。在节点i，先用完整parent+STOP候选集合计算原Actor logits；其softmax中STOP概率为p_i。仅为采集长主干，parent按同一组完整logits在非STOP项上重新归一化后的μ_parent抽样，然后按当前π_practice实践，真实执行5轮。不能在Actor forward时移除STOP候选再算logits：现有candidate set attention会改变其它logits，那不是原π的条件分布。也不能沿用原无条件parent log-prob；数据明确登记conditional_continue行为，不冒称部署策略完整on-policy路径。每批保持当前θ/ψ固定。

每个节点i=0..5都有“此刻停止并由final head选择”的真实闭环价值q_i。输入只用当时L/V/H/P/原生响应；候选query来自该condition实践之外的同一4-state面板，固定scene/env-policy RNG，每个已实践λ只算一次并跨节点复用；未实践提案不额外query，query及未来H不能进入早期节点输入。第0节点只有MT。final分支RNG与继续行为RNG分离；部署从同一final分布采样，不换argmax。

为同时训练早期和较晚输出，对每个训练截断深度d=1..5计算一个停止分支期望，F=(1/5)*sum_d V_0^(d)。递推为V_d^(d)=q_d；对i=d-1..0，V_i^(d)=p_i*q_i+(1-p_i)*(V_(i+1)^(d)-c)。这等价于对所有可能的首次STOP位置按实际π概率求和，未停止概率自然相乘，不把各短分支当独立经历。d只供训练目标展开，不喂π、不作为部署轮数菜单。第d轮外部截断只规定终值q_d，不标为“π应该STOP”的正标签。

c为一次继续修订/实践的轻微成本，首批固定c=0.001（成功奖励0/1，5轮最大扣0.005），不做成本系数扫描。若完全不计继续成本且旧LoRA始终可选，停止和无改善的继续可能价值相同，成功率目标无法唯一教会优先早停。这里成本明确表达同等能力时少实践的偏好，也承认它引入很小的质量/成本取舍；正式性能与能力得失仍是方法裁决依据，不能用省成本接受明显更差的闭环模型。实际训练花费仍按全部5轮采集和query计算，不能用预测的部署STOP减少真实账单。

固定一条已采的继续主干，停止倾向的直接梯度由q_i-(V_(i+1)^(d)-c)决定，并乘到达该节点的未停止概率等因子：继续的预期收益超过成本时降低STOP，当前输出更好或继续无益时提高STOP。未来收益是当前π后续选择的加权价值，不是拿未来最好checkpoint/候选作oracle标签。

整个估计的梯度须同时包含采样分布和解析停止/最终选择两部分。对每个d，使用stopgrad(V_0^(d))*sum_{i<d} grad log[μ_parent(a_i|s_i)*π_practice(b_i|...)]，另加在固定已采主干上grad V_0^(d)，再对d和task求均值。第二项训练STOP概率、final概率及共享θ表征；第一项给实际parent/practice选择正确信用。不能mask STOP后仍套旧“单个终值乘全路径无条件score”，也不只用未来收益硬标签拟合停止头。独立baseline仅看当时状态、当批前冻结；每批一次actor更新后重采。teacher replay仅用于阶段A，G更新后重新采完整条件继续主干，按§4继续G/π交替共同学习。无需额外展开树分支：这些V均可在同一条5轮真实主干上反向递推。

按上述采集，部署实际STOP后立即停止；训练为知道继续分支的价值，仍会采集停止节点之后的真实反事实继续经历，并用正确概率加权。这是显式变更的采样/梯度合同，不冒称每个后续状态都在部署实际发生。π训练来自全部36任务等权；未来截止预算仍不输入模型，超出训练深度的表现不承诺自动泛化或单调提升。

成本上界：每条件最多6套已实践参数（含MT）各4条query，加6条实践，最多30条episode；36条件一遍最多1080条，合法MT复用/重复候选可减少真实新增量。首π消费者须包含G生成、原生响应、真实5轮、候选query、反传与写盘，按测得吞吐冻结约1h量级的完整36-task更新窗口；不缩成4task或假称分叉零成本。只做比例适当的实际消费者/短有限分支梯度检查，不建新测试框架或并行算法路径。

数据池维持当前已冻布局：teacher独立报告16初态不进G/π/H；π实践20个，统一query为init_order[36:46]的10个、每condition均衡4个并全部排除实践；init_order[46:50]有限train诊断。旧MT逐行配对复用按§3，不改当前teacherbank。正式结果用实际STOP/外部截止下最终选定的唯一LoRA做strict400及能力保持；训练F和所有未来候选的oracle最大值都不用于正式模型选择。

## 6. 最小完整读出、继续与停止

全36学习覆盖不可省。先小的每task样本/更新窗口，再根据真实学习扩大；不先批量铺36×多Q。teacher/G/π学习阶段有不同产物身份，不把teacher强或CFM下降记成视频能力。

第一轮完整模型至少做一次固定共享checkpoint的strict validation paired400（correct K1、50视频/task各一次、固定state-video映射、适应不使用该condition最终初态），对当前强MT核R/G/L/suite/breadth；无Test。运行前预登记这一阶段的资格checkpoint和预算，不能用36任务小面板选formal模型。预算敏感性可先在36训练任务的固定小面板读出，明确它不是正式400资格。

出现初步优势继续相邻完整checkpoint判断稳定性，并在selected冻结后补规定same-task-other及视频controls；shuffled/reversed最后，不参与选点或架构修正。无优势先消费最早缺口：教师是否真强、G是否传递、π是否错选、视频/H是否提供效用。不靠全图乱改/参数小扫或无限续训保护弱假说。

无足够教师/数值稳定性、数据墙或资源违约时按真实原因停止相关投入并报main，仍继续独立有效工作；这不是Owner新许可门槛。阶段数值与deadline如不相容，必须在启动前给出实际预测和具体取舍，不能偷偷取消task覆盖或冒称准备充分。

## 7. 效率、清理与交付

同task单轨迹保存小LoRA节点，不为每节点复制source；checkpoint训练与已就绪节点读出按独立队列并行。持续复用frozen policy/evaluator，按真实长短任务cost调度；不要为固定world数空rank或因低util排除可共驻设备。G训练原生图像raw可有界复用，变化meta的contextual特征不得缓存跨更新；冻结G/π阶段合法复用教学特征/不变候选响应，实际θ编码仍重算。

实际profile比较利用显存的batch/worker；先使用现有batched LoRA与原生算子，不为赶时间加第二运行面。必要批量teacher模型可复用共同基底，须保持每task独立参数/optimizer/均值与stream；吞吐要测，不声称线性提速。

Owner允许退休旧架构无恢复/当前依赖的checkpoint和训练状态：保留source/MT/原始数据/tokenizer/assets、必要强参照权重、当前新数据候选和准确历史小证据；记录路径、释放量、metadata-only/weights-only/full-resume身份。当前G运行依赖不删除。旧rank8 seed可复用部分须先确定数据角色，不能将49-query池历史写成新0–45留出合同；不为单轨迹叙事重跑已可合法复用的模型。

实验session可在上述已经定清的scope内自主完成工程、清理、完整检查、Git集成/push、freeze与launch；main负责及时消费实际吞吐确定尚未冻结的数字，不把重复人工ACK变成流水线等待。实现修改旧canonical路径，历史代码由冻结worktree/Git/原件保留，active tree一个运行面。

## 8. 保留的实际计算与历史边界

当前canonical实现为 `src/ember/proposal_writer/`。双meta各18层q/v rank32/alpha32/dropout0，seed202610101/102；教学读取原source双RGB/exact L、stride5、50×32固定probe1729/time1/state-free。Gemma finalnorm contextual图像位与合法语言位、action finalnorm H均有G消费者，source冻结；实际任务执行关闭meta并用MT合并基底+唯一38/r8。
G保留完整ξ/time、64数值块、block/rank/target通信、6层width256/FF1024与16Euler；meta及全部G编码/归一化/速度网络归ψ，π独立下游Reader/trunk/heads归θ。π学习冻结全部ψ，不用θ可训练embedding改变G核。源raw量可有界缓存，变化meta输出跨更新失效。
实际H复用EpisodeRunner同一次10步原生ODE的头尾hidden、RGB/proprio/实际动作/反馈/噪声/producer，不由离线demo或旧score补造。
旧四task/逐H-Q/空H/1024与rank8 bank/G240/G480只作为原Git/冻结代码/正式原件的历史身份；新版不用旧G optimizer恢复、不执行旧π/报告，不重写旧成本或数据留出。


## 9. 共同目标的已知限制与等价计算复用

共同E让监督E−P确定；CFM可拟合单目标，旧多模态/避免均值/随机性自然带来有益探索论证不再适用。
不造多个弱E，不新增回归/采样数臂。当前有界生成的控制传递及完整循环证据裁决后续，参数差异不等于探索收益。
不同H共享E意味着G可能不用H，学生parent/H覆盖不自动证明经验增量；每次θ更新时ψ固定，π梯度只能学习选择；跨批ψ更新从实际学生P/H监督学习，不能因此宣称已修复生成能力。
当前同节点G context仅算一次供最多4个ξ/16Euler共享。native绝对响应cache由单Compilation拥有，保留一个实际支持集合，
身份含冻结MT基底/任务初始坐标、不可变candidate ref、exact L、实际pre观测key与噪声；跨head仅复用相同绝对预测，parent差值重新计算。
变化meta/contextual图不跨G更新缓存，θ Reader每次梯度消费者fresh；实际端到端收益由完整π消费者计时，尚未实测不声称已提速。


实现所有权：path保持唯一状态机与实际参数/H生产；learning保持参数CFM、final监督及条件继续梯度；pipeline只调度阶段，不复制原生策略。
supervision拥有local/RL共同使用的lazy真实重放及once-per-λ query标签，替换旧每prefix重新query；evaluation拥有held400无梯度适应/固定MT参照，
并给教师报告复用同一模型/normalization/执行口径兼容检查。新增两文件是训练监督与held报告各自已有消费者责任，不是新runner或fallback。
旧固定local128/RL16/48/U和step-weighted F消费者已删除；历史仅由旧commit/冻结树/原件保存。局部复杂函数按bootstrap、CFM更新和共同训练职责收窄；新CFMTrainer复用同一参数更新，联合原件同时保存ψ/θ优化器与bank版本。旧selector消费后删除参数的路径及专用测试退役，学生参数/H仍是G依赖。

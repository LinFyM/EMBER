> 2026-10-12 authority：本文件仍是唯一登记设计；§14恢复后48行leg已实际退出0，全36/576读出完整。
> MT340/G864341/G1296=348/E455；原截止内496/逾时32/恢复后48分列，旧exit1/15:55保护失败不追认成功。
> 本有界实例到此结束，后继由main自主消费决定；不自动训练/新生成/practice或其它阶段，实际来源见progress。
> §13 v7 G864/query16已完成并由main消费，原factor-MSE组合停止延长；旧§11–12与旧full36均已收口。
> 同图从完整v7 CP864显式迁移目标，新增432到1296；1080仅恢复，1296唯一科学出口。
> 保持MT-only H bank0、同36/logicalB4/SG/RNG；未被§14覆盖的信息墙与静态/Actor合同保持。

# 36任务目标教师、自身实践与交替共同学习：main执行合同

Owner 2026-10-11约14:30北京时间明确要求立即自主推进；本稿由main定科学内容，实验session在独占工程窗口整合进唯一active设计/状态文件后，用clean pushed detached实施。旧四task G480仅自然收口，不恢复旧π矩阵。新批不再以四task试训筛方法。

## 1. 目标、范围与时间

- 数据/G/π覆盖既有审计train24+support12全部36任务，按task等权；固定Val8/Test8离线标签不读取，不新增任务或分割。Task36名单复用现有ALLOWLIST，libero_90支持任务official horizon400必须进入环境合同。
- 今天数据准备原目标4–5小时、最迟约5–6小时；实际基准从本次约14:30授权计，原训练数据目标北京时间20:30前。Owner随后明确质疑4/8初态精度，main消费gpu01直接实测后将选点/独立报告各扩大到16。原precision预测20:40–21:05；正式教师bank已于北京时间15:21:08从clean pushed detached dab2f725启动，工程/冻结实际耗时后最新预测20:55–21:20，比原20:30目标晚25–50分钟。这是明确披露的精度/时间取舍，不把时钟重置为formal启动，也不承诺已按原期限完成。独立教师报告在E锁定后与G并行，完成前不宣称已验证教师泛化。工程、清理、profile及数据等待均计入wall。
- UTC08:09资源/ETA修订：gpu01:5准入时free由21262降至3275MiB，task37原attempt未启动、GPUh=0；UTC08:07:46在gpu02:4接续原800×28链，其MT读出仍由原协调器负责。23个已完成200-step段实测gpu01四卡约2.363/2.869/2.415/2.797秒每更新，gpu02:6约4.482秒；取代此前快卡代理估算后，bank保守预测北京时间22:04。main已读取coordinator/task37_recovery_01/forecast.json；全36 capture尚未完成，早成功或共驻变化仍会改变ETA。该预测不含G训练或独立教师报告。原有效更新与失败记录保留，代码不热改，调度承接不能重复教师计算。
- UTC08:22必要事实更正：六卡调度交接时执行者误向外层launcher发送SIGINT，原协调器UTC08:11:21退出130。task0/15/17/19自然完成200点，经退出事件和完整manifest接管，但原OS wait状态不可读；local14在100点前的未保存更新丢失，保守失败成本0.077613906132 GPUh，已从初始MT重启，不能继续称全部有效任务无中断。teacher_bank_resume_02于UTC08:13:47从同一dab2 clean frozen接续六卡依赖调度，有效checkpoint不重训。main已读取ROOT/run_contract.execution_revisions及两个协调器回执。22:04为此前五卡队列模型，六卡接续后的预测仍待真实capture修正；不重置14:30起算时间，不提前承诺提前完成。
- 第一轮G/π用约1小时量级的学习窗口取得全任务初步证据；这是投入规划，不是声明1小时充分或变成逐小时批准。按实际吞吐冻结整数更新数/完整task轮次，结束前后预估偏差及时回main，不把4task或无限缩短教师作为偷偷赶时限的办法。
- GPUh只记录，无硬cap；原物理合计6/8、单节点6、live显存非干扰与data1独立quota保持。新增峰值按实际H压缩样本/CP/模型估计，不能复用旧64GiB而不核算。

## 2. 模型与参数消费者

保留source+已合并MT冻结基底，一套38-target/rank8任务A/B，共同A0非零/B0=0；两组fresh rank32读取meta、现有有序教学Reader、真实参数/经历编码、完整参数CFM G及独立三head π。当前参数的摘要与局部原始因子均保留，不新增7B权重压缩器/世界模型/辅助动作网络。

G输出更新因子ΔA/ΔB，替换为A+ΔA,B+ΔB；目标E与全部parent同基底同坐标。纯参数CFM为本批G目标，不新增rec/keep标签构造或直接动作辅助loss，不为产生多模态把弱教师混回目标。共同目标允许噪声样本收敛，不能把随机性当有用探索证据。

新condition所有共享权重冻结；其执行LoRA只由G生成/修订与π选择，不从自身或准备的动作轨迹做policy微调。当前完整策略的实际hidden/动作/后果进入H；教学meta不叠入执行。首次G前强制MT真实实践，计入同condition预算，无空H生成路径。

## 3. 每任务一个目标教师；选点分数与学生自身经历分开

Owner本轮再次指出教师只需提供目标LoRA。main纠正此前继承的teacher-H bank和teacher-prefix warm：共同E足以为实际学生P/H提供E−P监督，教师中间点经历不是当前方法的必要准备。此前全节点16H及两折热身条款退役，下面保留选E所需真实闭环精度，不用少存H冒充省掉全部评估计算。

- 每task一条持续任务LoRA训练轨迹，共同目标E为登记候选的selection闭环最优；包含MT，tie选更早节点。未评估的保留checkpoint不参与选点，不声称E为全部保存节点的已知最优。不得逐parent/H重训Q。
- 每task最多800更新、每100保存完整checkpoint，候选仅MT/200/400/600/800。28,800更新/806,400 action query是未裁剪上界；科学省略不得算作已完成训练。有效query batch28、AdamW既定lr1e-4/betas(.9,.95)/eps1e-8/weight_decay1e-4/clip1、无scheduler；native纯demoFM+对初始MT的1e-3坐标prox保持，无parent-H rec/keep。全部已完成CP/optimizer/query cursor/RNG和费用保留，不重训。odd100节点不自动补评估或进入共享训练。
- 教师action demos0–45训练，46–49只作该教师无梯度任务内验证/教学。MT50/旧Q49历史暴露如实保留；该任务内留出表示未参与这次教师参数拟合，不是从未读取。Val8/Test8离线标签仍禁止，合法train36的RGB可用于共享G。
- 每个实际候选仍读完整16个固定selection初态，跨节点共用scene/env/policy RNG，全部成功/失败计分。36×5×16=2880是未裁剪选点episode上界，不再是H样本数。E按完整16行success count择优、earliest tie；不改易初态、不用loss或小样本普通回退选点。selection默认只保存score/必要first-native证据，不采集保存全程RGB/action/hidden H，不参与G/π监督。
- 任务内先MT16，再顺序消费已有CP或继续原轨迹到下一200节点后读16；不同task无阶段屏障，canonical cost-balanced队列及实际slots16承接。某节点16/16且更早候选完整时，该节点在原max-success/earliest-tie规则下已锁定E，后继尚未启动训练/readout以selection_saturated省略。无16/16继续到原800上界，不采用部分行竞赛。已经在途的有效工作自然保存结束，不为本次变更丢未保存更新。
- 教师端交付唯一选定E、候选分数、完整出处与必要恢复资产；中间CP只是选点/恢复证据，不再是G的parent池。只因过去保存过多parent代码，不能继续制造其H或teacher replay。旧已保存完整H保留历史身份，不接回active共享监督。
- E锁定后用另外16预留初态读E与MT，合计1152逻辑报告行，不参与E选择、G/π/H或尺度；selection加报告合计4032逻辑行上界。报告可与G并行，未完成不能声称已验证教师独立能力。E=MT时同一行只执行一次；兼容历史MT按实际参数/task/init/scene/env-policy RNG逐行引用并标明reuse/new，52交集等旧索引须保留实际消费者核验身份。报告阴性不当工程故障。
- 唯一选择经历复用例外是下节事前指定的MT起步行：每task原teacher_selection_states[0]不按score挑选，允许同一次MT rollout同时产生selection分数和一条自身bootstrap H。其余selection行不采H。面板task/init/RNG顺序不改，只显式修订用途；8+8折退为历史元数据，不再是学习接口。旧只有score/first_native的行不能重建H，必要时补一条真实MT实践并如实增加成本。
- 没有必要人为制造退化、回退或多教师标签。未出现的后期parent不宣称已覆盖；学生真实错误和回退由后续共同学习自然取得。所有参数ref/实际producer必须真实唯一；不需要sim snapshot，不新增校验侧车或并行数据管线。

## 4. 学生自身P/H、共同目标与固定尺度

实际共享样本为(L_t,V_tj,P_ti,H_tik ; E_t−P_ti)。E只作训练标签，绝不进入G的条件；H非空、来自P及此前实际产生者的真实实践，不用teacher中间节点构造。

初始化每task仅一条事前指定MT自身H：优先引用原teacher_selection_states[0]同一次MT rollout的完整合法经历，无则显式补采并计费；36任务均到齐。新鲜G首先学习(L,V,MT,H_MT;E−MT)，明确是MT-only初始化，撤回main此前从首更新混合教师parents的课程，不声称已覆盖任意P。多个教学视频可以复用同一真实MT H，不能称其为多个不同实践。

全部36个E齐备后，用task等权的E−MT每因子RMS和既定1e-3 S_prox floor一次固定S_G，之后学生刷新不重算。该尺度来自新鲜初始实际parent分布，教师参数拟合、坐标和基底不变。K=1，教学只来自登记合法train36视频；用过的训练task资料不声称G未见。

G从fresh meta/G初始化学习新目标（旧G240/480不是此合同exact resume）。按预排task-balanced轮次取样，样本量逐步增加，checkpoint保存全部optimizer/sampler/RNG/topology。输出target为相对当前原始因子的差，目标标签/teacher性能不输入条件。

Owner进一步要求真实的“修订—实践发现偏差—再次修订”学习，并提出无额外教师与G/π共同训练的可能性。main据现有控制教师正证据及旧经验/功能监督负证据，保留本批共同教师；教师不是理论必要条件，也不以正在运行作为保留理由。原“G监督初始化、一次学生补充、之后整个π阶段永久固定ψ”的训练组织退役。改为密集G初始化后，持续按当前编译器实际访问的参数/经历聚合监督，并交替更新共享G与π。保留现有CFM/完整参数出口和π三head，本次主要改变训练组织，不另造恢复教师、逐H目标、额外动作轨迹或并行Writer。

首个有信息的G节点后覆盖全部36任务采实际生成parent及其实际H，每task至少一次有效实际G候选实践。该节点就是唯一G初始化出口，不再为旧整阶段固定G课程另设学生刷新后的G-only段。每task沿原有seed/顺序最多尝试4个候选，生成到首个valid即实践，后续未消费候选不计算；不按成功率挑候选或补采。初始学生覆盖沿既有合同，随后复用下节π采集的真实长主干继续刷新G，不能只补一批后永久停止。新鲜路径始终用同一ψ_k、θ_k走完该逻辑batch；第一次G前仍先MT真实实践。每轮实际实践的P_i及截至当时完整H_i加入G监督候选，保留混合经历的真实producer。标签直接是同一E_t−P_i；E是标签侧，绝不输入G，也不按H重训E。较早节点只包含当时信息，失败和晚期回退均保留，不用最终成功筛掉困难状态。

G监督更新仍是当前已登记参数CFM：在真实(P_i,H_i)上以(E_t−P_i)/S_G为端点，真实CFM梯度进入全部ψ，包括双meta、Reader、参数编码和速度场；P_i/H_i/目标作为记录停止梯度，不假称反传MuJoCo，也不声称这是完整成功率策略梯度。固定目标不要求人为指定多步路径；若一次可正确生成E应鼓励一次达到，不插值出固定五段“正确修订”。共享梯度训练的是从自己实际错误状态重新修订的规则。新增H是否被有益利用仍须实证，数据聚合不构成收敛或经验利用定理。

每个共同学习循环的次序固定为：当前ψ/θ采集完整batch与独立query→在尚未改变ψ的前提下完成§5的一次θ更新→把真实新P/H并入监督bank，完成登记的ψ更新块→失效变化meta/context缓存→用新ψ/θ重新采下一batch。π的旧score路径不在ψ更新后重用；G可复用有完整参数/经历来源的旧监督记录，因为同一E−P标签仍有效，须使用当前ψ重新编码。整个训练称交替共同训练：π用真实query回报，G用共同强E的密集参数监督，不把π梯度穿过变动生成器而漏掉生成/实践分布导数，也不冒称G已经直接获得完整RL信用。

任务等权、task内全部登记的实际学生parent等权、parent内部H等权；MT与实际生成parent不设人为类别比例，没有教师中间点池，S_G/坐标不随刷新变化。新P/H只保留一份可追溯原件；π消费后的参数/H退休须检查G bank仍在引用的依赖，不能先删除再重采。可合法复用已有MT bootstrap H/查询记录，唯一环境成本与每condition逻辑投入分别记。首个约1h G初始化窗口及随后完整36-task共同学习窗口由真实消费者吞吐冻结更新数/采集数，不以“联合”默许无限交互或把未满一遍36覆盖称学习完成。

读出必须保存并固定同一时点的完整(ψ,θ)与两个optimizer/sampler/RNG/数据版本，不拼不同阶段最优G和π。训练已有各节点query用于观察修订前后能力和保持，不用参数距离下降认证控制进步。首轮新方法的实际新增H作用可以在冻结模型、同V/P/噪声的有限train条件上对照是否提供新经历；这是机制诊断，不是独立训练的无H算法或正式资格，任何额外行须在执行前明确面板及成本。若只有反复重采样收益而无新增H贡献，不能声称已实现反馈纠错；不会因此恢复逐H教师。正式结论仍按§6。

## 5. π的五轮训练展开、终止分支与真实停止

Owner进一步提出训练可固定迭代深度（以5轮为例），并追问停止头怎样训练。main据此采用首批R=5训练展开；它是当前共享训练采集深度，不是部署固定轮数或未来预算输入。退休上一稿用环境步区间加权出口的F目标和Cmax驱动训练终点。π及baseline不读取未来总/剩余预算；保留parent/STOP、practice、final三head、独立Reader/static；ψ在每次行为采集和对应θ更新内完整冻结，跨批次按§4更新。外部资源限制/实际步数/GPUh仍登记，实际训练首段更新次数按完整消费者吞吐冻结。

一轮定义为：在当时真实H/P上选择parent，冻结G产生现有最多4个提案，practice头选择一套实际实践，完成一条官方horizon/success终止的episode，再在已实践集合上输出当前可交付LoRA。MT真实bootstrap是第0阶段，不计为一次G修订轮；第一次G仍非空H。首批每条件训练主干实采5轮，最多20个G提案、16 Euler，最多MT+5条实践episode；不按旧4*(horizon+10)提前截断这5轮。每轮官方episode horizon继续生效，资源故障/外部中断不能冒称正常完成5轮。部署可在任一完成节点交付一套固定LoRA，也允许π主动STOP，实际上限与成本在评测前登记。

阶段A只做最小final热身，完全复用§4同一个初始学生refresh的MT→G真实路径。退休teacher-prefix/两折warm及其数据依赖，也退休为热身另采uniform五轮local_data。每task的学生路径权重/H只保存一份；只给其中已实践MT/G参数补统一query池的4-state配对标签，每参数/初态/RNG一次，已有兼容行引用。query不进H，未实践候选不查询。不为多候选比较另造教师轨迹；这段只有MT/G pair的初始覆盖，随后真实R5训练全部head并持续更新G。

唯一热身段覆盖全部36 task，更新数按实际消费者冻结；学生refresh时的ψ也是首RL初始化ψ，没有第二段G-only或选优点。q_i(θ)=sum_{λ∈P_i}π_final,θ(λ|s_i)*r_query(λ)，以真实prefix的condition均值/task均值训练final，不将其称完整STOP/继续决策RL。热身结束前将原始MT/G真实P/H登记同一G监督bank，原标签有效，下一次ψ更新使用当前编码；不因ψ变更重采原件。统一顺序为G的MT-only初始化→36学生首实践/必要query→同一学生pair final热身→持续R5交替共同训练。

R5首次visit可以复用已登记MT bootstrap，后续按原practice_states及condition seed真实采MT，沿现有compiler的bootstrap_history=None能力执行；已有严格同输入/RNG完整H可引用并准确记唯一环境成本。不得从退休teacher16H池索引起步，不把一个固定H永久冒充20-state分布；每个condition仍将MT真实起步计入逻辑适应预算。

初始student query与首轮RL的每task visit=0共享query_panel(panel,0)、seed(70,task,0)和固定MT；path identity不参与环境/policy RNG。首RL的MT四条查询直接引用初始student/query兼容原行（全部36最多144条），只补真实缺口，保留来源/逻辑condition与实际episode身份、旧费用不重计。G查询仍逐实际参数运行。后续visit的RNG不同不得为缓存改seed；无需通用全局cache或逐tensor校验。

完整共享RL采用“当前π条件继续采样，停止选择解析求和”。在节点i，先用完整parent+STOP候选集合计算原Actor logits；其softmax中STOP概率为p_i。仅为采集长主干，parent按同一组完整logits在非STOP项上重新归一化后的μ_parent抽样，然后按当前π_practice实践，真实执行5轮。不能在Actor forward时移除STOP候选再算logits：现有candidate set attention会改变其它logits，那不是原π的条件分布。也不能沿用原无条件parent log-prob；数据明确登记conditional_continue行为，不冒称部署策略完整on-policy路径。每批保持当前θ/ψ固定。

每个节点i=0..5都有“此刻停止并由final head选择”的真实闭环价值q_i。输入只用当时L/V/H/P/原生响应；候选query来自该condition实践之外的同一4-state面板，固定scene/env-policy RNG，每个已实践λ只算一次并跨节点复用；未实践提案不额外query，query及未来H不能进入早期节点输入。第0节点只有MT。final分支RNG与继续行为RNG分离；部署从同一final分布采样，不换argmax。

为同时训练早期和较晚输出，对每个训练截断深度d=1..5计算一个停止分支期望，F=(1/5)*sum_d V_0^(d)。递推为V_d^(d)=q_d；对i=d-1..0，V_i^(d)=p_i*q_i+(1-p_i)*(V_(i+1)^(d)-c)。这等价于对所有可能的首次STOP位置按实际π概率求和，未停止概率自然相乘，不把各短分支当独立经历。d只供训练目标展开，不喂π、不作为部署轮数菜单。第d轮外部截断只规定终值q_d，不标为“π应该STOP”的正标签。

c为一次继续修订/实践的轻微成本，首批固定c=0.001（成功奖励0/1，5轮最大扣0.005），不做成本系数扫描。若完全不计继续成本且旧LoRA始终可选，停止和无改善的继续可能价值相同，成功率目标无法唯一教会优先早停。这里成本明确表达同等能力时少实践的偏好，也承认它引入很小的质量/成本取舍；正式性能与能力得失仍是方法裁决依据，不能用省成本接受明显更差的闭环模型。实际训练花费仍按全部5轮采集和query计算，不能用预测的部署STOP减少真实账单。

固定一条已采的继续主干，停止倾向的直接梯度由q_i-(V_(i+1)^(d)-c)决定，并乘到达该节点的未停止概率等因子：继续的预期收益超过成本时降低STOP，当前输出更好或继续无益时提高STOP。未来收益是当前π后续选择的加权价值，不是拿未来最好checkpoint/候选作oracle标签。

整个估计的梯度须同时包含采样分布和解析停止/最终选择两部分。对每个d，使用stopgrad(V_0^(d))*sum_{i<d} grad log[μ_parent(a_i|s_i)*π_practice(b_i|...)]，另加在固定已采主干上grad V_0^(d)，再对d和task求均值。第二项训练STOP概率、final概率及共享θ表征；第一项给实际parent/practice选择正确信用。不能mask STOP后仍套旧“单个终值乘全路径无条件score”，也不只用未来收益硬标签拟合停止头。独立baseline仅看当时状态、当批前冻结；每批一次actor更新后重采。G更新后重新采完整条件继续主干，按§4继续G/π交替共同学习；没有teacher replay阶段。无需额外展开树分支：这些V均可在同一条5轮真实主干上反向递推。

按上述采集，部署实际STOP后立即停止；训练为知道继续分支的价值，仍会采集停止节点之后的真实反事实继续经历，并用正确概率加权。这是显式变更的采样/梯度合同，不冒称每个后续状态都在部署实际发生。π训练来自全部36任务等权；未来截止预算仍不输入模型，超出训练深度的表现不承诺自动泛化或单调提升。

成本上界：每条件最多6套已实践参数（含MT）各4条query，加6条实践，最多30条episode；36条件一遍最多1080条，合法MT复用/重复候选可减少真实新增量。首π消费者须包含G生成、原生响应、真实5轮、候选query、反传与写盘，按测得吞吐冻结约1h量级的完整36-task更新窗口；不缩成4task或假称分叉零成本。只做比例适当的实际消费者/短有限分支梯度检查，不建新测试框架或并行算法路径。

数据池维持当前已冻布局：teacher独立报告16初态不进G/π/H；π实践20个，统一query为init_order[36:46]的10个、每condition均衡4个并全部排除实践；init_order[46:50]有限train诊断。旧MT逐行配对复用按§3，teacher selection只选E、独立报告只报告；唯一登记MT bootstrap有双用途，其它行不回填H。正式结果用实际STOP/外部截止下最终选定的唯一LoRA做strict400及能力保持；训练F和所有未来候选的oracle最大值都不用于正式模型选择。

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

## 8. 首个完整共享窗口与正式读出定案（2026-10-11 UTC11:33）

本节冻结此前待实测的整数，适用于当前全36共同E/自身H合同；不恢复历史G480/local128/RL16。main已直接消费3b042855的profile_physical_cost_04脚本、consumer及exit：实际UTC11:01:50–11:18:59退出0、1029.814秒/0.286059436 GPUh，分析例外设备已释放。先前因回报延后推断超时不成立，不作为故障或额外成本记录。

### 8.1 实测与物理安排

- 同一Long任务、104-record MT H、71-frame视频，部分B4的CFM热步：chunk8为27.083秒/18.061GiB reserved，chunk16为26.085秒/26.057GiB。只支持这台共享卡/该输入的约3.7%热步差；不是完整B4或全36实测。现场28,798MiB余量限制继续放大，不能宣称chunk32已测或最优。
- 当前624-record H大小约2.823GiB，超过旧2GiB cache。借用同一path/冻结φ后，最长H的部分B4 ψ步27.491秒、image_observations=0；不再解压或重编码原始图像。它与03的60.261秒同时存在chunk/现场负载差异，不把全部差额归因一个改动。变化meta/θ/ψ表示仍fresh。
- 同8-query面板、相同task/参数/初态/RNG，ready合批0→0.1秒，wall356.780→348.286秒；实际环境步3973→4200，需同时报告吞吐，不能当成相同执行步数。0.1秒安排实际多数native batch8，选择此物理安排，不依据这两次闭环分数选模型。native批量4/8/16的早期测量明显非平稳，不宣称4倍收益。
- G优先使用已实测能缩短完整B4 wall的同节点4卡、logical B4不变；首launch现场没有4卡时用可立即运行且有效的拓扑，不等凑卡。frame chunk按真实显存选16或8，必要的更大物理安排比较由实验在当前授权内自决，不能因此再铺科学矩阵或阻止已就绪工作。每次启动保留实测峰值/余量和选择原因。
- 独立环境worker在高余量设备沿已测slots16，低余量沿slots8；采用已验证的ready coalescing。持久模型/队列、多condition采集、frozen raw/context合法复用保持。共享optimizer常驻下的峰值不可用本次释放optimizer后的native峰值替代；按实际启动准入承接。

### 8.2 冻结整数和数据消费者

1. **G初始化216次完整logical B4更新**，即24遍全36任务、每task24个采样机会。全部E与指定MT H到齐、S_G一次固定后fresh；只在216保存当前阶段出口/完整恢复点，不用中间loss选点。旧合法G仅用于成本，不作为此训练初始化。
2. **初始学生一次全36覆盖**：每task原顺序至首个valid G提案后实践，至多4个提案，沿既定失败处理，不按成功率补采。复用原MT H，同一个MT/G pair及query4供final warm与G bank；无额外教师H或uniform R5段。
3. **final warm36次logical B4更新**，即4遍全36 task，只消费上述现成pair/查询。共享参数、optimizer及查询墙沿已实现合同，不新增损失/LR/seed搜索。
4. **第一共同窗口9次logical B4更新**，即全部36 task各一条当前策略condition。每condition固定实采5轮；每批固定ψ/θ采集与独立query→θ一次→登记真实P/H→**ψ做9次CFM更新**→下一批重新采集。ψ块9是一次完整36-task均衡覆盖，避免只更新刚采到的4个task；task→parent→H原等权不变。整个窗口合计9次θ更新、81次ψ更新，G sampler由216接到297；不将全任务覆盖或一次采样当作收敛证明。
5. **RL9是首个完整(ψ,θ)资格checkpoint**，保存两个optimizer、baseline、G/θ sampler、bank版本、全部rank RNG及拓扑。只用这一完整点进行首轮formal；不拼阶段最优G/π。此前学习路径上的query仅诊断既定机制，不挑共享checkpoint。

工程恢复补充：每个完整θ→append→ψ块结束保存同点完整恢复状态；RL1–8只用于恢复，资格节点仍唯一RL9。未持久化的整batch从前一完整点接回，已保存且同核的真实路径/query可复用，不对旧路径做额外actor训练。θ消费结束释放该路径native KV和已消费梯度，保留G bank仍引用的参数/H及frozen raw特征；联合RL的chunk16现场门槛为free≥40000MiB，G初始化仍为28000MiB，低余量按既有8/4物理chunk承接，不改逻辑B4/任务权重/训练步数。

EGL工程边界：渲染设备按实际CUDA可见身份解析，不把EGL枚举位置直接当物理卡号；非CUDA DRM设备不作替代。完整查询结束后释放闲置renderer，CPU worker保留，下批仍按原state/scene/RNG重置；进入环境阶段前释放未使用的PyTorch allocator缓存。不得释放未完成episode的环境或改动真实H/P/query，物理资源阶段交接不改变固定R5和θ→append→ψ顺序。

首学生最多新增36次G实践及288条query；首共同窗口在兼容MT bootstrap/144条MT query引用后最多新增900个episode（180实践+720查询），别名/提前成功按实际计账。36条件仍各有完整五轮，引用不伪造新经历。没有新的训练诊断矩阵或独立无H/语言/多预算模型；消费已有实际query解释最早缺口。

### 8.3 首个strict paired400

RL9保存后运行唯一correct K1 validation400，与现有兼容MT300/153参照逐行配对；video_seed=7，8任务每task全部50合法视频一次、同state-video映射贯穿后续checkpoint。测试集保持关闭，适应排除该condition最终初态，最终反馈不进入H或更新。

本次正式读出在已训练深度内预登记资源截止：MT起步最多一条官方episode，最多5次修订实践、外部环境步上界6×(该suite horizon+10)。π按真实parent/STOP分布决定是否继续，可在任何已完成节点提前交付一套LoRA，绝不强制部署跑满5轮；这个首批上界不成为永久方法轮数，也不新增轮数菜单。未来预算仍不输入π/baseline，外部资源终止与π主动STOP分开记录。

第一400后main直接消费per-task/suite、R/G/L、breadth及原件。若有初步实质优势，再在同一科学合同下登记相邻完整窗口/CP检验稳定性；没有优势先定位教师供给、G传递、π选择或视频/H效用的最早缺口，不自动延长训练、调小参或重开架构。selected冻结后补必要same-task-other及视频controls；shuffle/reverse仍最后，不进模型选择。

### 8.4 时间预期与接续

216步G按当前共享卡26秒及历史实际四快卡约15.5秒/global B4，连同首次编码/启动暂估约1–1.75小时；新全36真实全局wall仍待首正式消费者校正。final warm按部分B4 7.23秒/现成3-episode prefix外推约5–10分钟，该prefix并非初始学生精确2-episode成本。

首学生全36实践/查询暂估0.5–1.25小时；首共同9批按当前Long R5范围、20-query成本、36任务官方horizon及81次ψ更新外推约1.5–3.25小时。它明确可能超过早先“约1h共同窗口”的规划，不缩水任务或五轮。首400若都走到外部截止暂按约3–7小时安排；实际STOP、长短任务与多condition吞吐会改变wall，首完成shard再修订预测，不靠预先假定成功/早停压低预算。这些是有出处的范围预测，不是全36或formal400已测结果；数据bank/独立教师报告的现行ETA另列，不重置14:30数据准备时钟。

实验session据本节直接完成必要合同/配置消费者、集成push、clean detached及真实依赖接续，无需main逐项ACK。既有有效教师工作/原件不重跑，独立报告与就绪共享阶段并行遵守总卡数。完成方法/效率goal并不声称EMBER能力已证明；正式整批结果或真实边界仍按既定可靠消息回main，main承担科学消费与后继选择。

## 9. 保留的实际计算与历史边界

当前canonical实现为 `src/ember/proposal_writer/`。双meta各18层q/v rank32/alpha32/dropout0，seed202610101/102；教学读取原source双RGB/exact L、stride5、50×32固定probe1729/time1/state-free。Gemma finalnorm contextual前512图像位、raw frozen合法语言embedding及action finalnorm H均有G消费者，source冻结；实际任务执行关闭meta并用MT合并基底+唯一38/r8。
G保留完整ξ/time、64数值块、block/rank/target通信、6层width256/FF1024与16Euler；meta及全部G编码/归一化/速度网络归ψ，π独立下游Reader/trunk/heads归θ。π采集及对应θ更新冻结全部ψ，跨批由登记CFM块更新ψ；不用θ可训练embedding改变G核。源raw量可有界缓存，变化meta输出跨更新失效。
实际H复用EpisodeRunner同一次10步原生ODE的头尾hidden、RGB/proprio/实际动作/反馈/噪声/producer，不由离线demo或旧score补造。
旧四task/逐H-Q/空H/1024与rank8 bank/G240/G480只作为原Git/冻结代码/正式原件的历史身份；新版不用旧G optimizer恢复、不执行旧π/报告，不重写旧成本或数据留出。


## 10. 共同目标的已知限制与等价计算复用

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


## 11. 当前有界批：G.field发送端归一化与query16（2026-10-12）

目的为检验旧G216实际生成学习失败是否受跨层raw-state循环放大影响，并精确读取G传递E控制优势的情况。
主讨论完整推导、竞争解释及最近似历史见[生成传播与query16决定](../analyses/proposal_transport_and_query16_20261012.md)。
本节是新ROOT唯一执行合同；内部统计不替代完整闭环、held迁移或视频/H增量结论。

### 11.1 计算与学习

只改G.field六层BlockLayer的跨层发送量：原局部FF残差后，用local_norm(blocks)同时计算原group内softmax的score和value；
rank self-attention不改，再用rank_norm(ranks)同时作原target_query的score和value。target self/cross attention及真实memory不改；
下行rank接收target_norm(targets)，block广播concat(rank_norm(ranks),target_norm(targets))的原Linear消息。
最终field.norm、Linear256→64、valid mask、完整643584坐标、16步Euler、xi/time原输入和CFM标签保持。
static两层、Actor独立static/Reader/三head、ContextReader、双meta的实际算子/参数结构保持；flow/static在同一模块内按职责组织，
不保留旧flow fallback。旧代码由1b44903d/Git保留，不原地修改冻结树。

G/meta/Actor全部seed7 fresh、参数形状/注册/RNG顺序保持可比、两个meta原seed不变，optimizer fresh；新Writer checkpoint schema v6
明确拒绝旧同shape恢复。复用原MT merged frozen基底、初始非零A0/B0、全部36共同E、一次固定S_G和每task原MT bootstrap真实H。
E只在标签侧，不进入条件；教学为exact L与action-hidden双RGB/stride5，H仍只来自自身真实交互。源权重冻结。

全36等权、logical B4、216次CFM/24轮，原任务/parent/H及L/V采样、AdamW1e−4/(.9,.95)/1e−8/wd1e−4/clip1保持，
不加时间网络、endpoint loss、功能损失、解析去噪或rank/dtype/LR扫描。这个G学习阶段不消费query16，传播学习比较仍单一主要变量。
只在216保存完整模型/optimizer/sampler/RNG/topology出口，不用loss选点；按live同节点物理执行安排实际rank/chunk，保持逻辑更新。
EMBER_META_FRAME_CHUNKS由run的实际Runtime消费者按LOCAL_RANK承接，不能只在launch env登记而忽略。旧在途运行不重启。

### 11.2 面板、首学生及E参照

沿原init_order/video_order/seed，practice=init_order[:14]；query=init_order[36:46]+init_order[14:16]+init_order[32:36]，
teacher independent report=init_order[16:32]、finite train report=init_order[46:50]保持。14/16/16/4各池互斥覆盖50。
MT bootstrap仍init_order[0]；首visit原6实践前缀不变。query中2初态参与过E selection，仅作train36机制/信用读出，
不称独立teacher泛化或formal资格；教师独立report16原身份不变。

216后全部36各一个原教学/MT-H/noise规则的首valid G实际实践，最多4提案尝试、不按成功补采。对固定已实践MT/G同16初态、
official scene/env/policy RNG各一次查询，跨prefix复用；query不入H、G条件或Teacher选点。分母/循环/cache/complete/成本均实际16行，
不能重复旧4行充数。旧MT只在实际参数/task/state/scene/RNG与原MT/base/initial都兼容时引用，保留原episode/成本与reuse/new；
旧G标签不复用。新查询只作只读评价，任何反馈不改变本批G216或候选选择。

固定E另外在同16初态/RNG只读运行；E=MT9task引用同一MT行。E不进Actor候选、path.states、H或信用rewards，
参照只用于解释传递。保存MT/G/E全576配对条件、per-task/suite、R/G/L/churn/breadth、success-set和E-only/MT-only/both-success/both-fail
各类中的G结果，空类如实为0。每task原件及唯一新增episode/环境步/GPUh均保存。

本批仅G216→首学生36→MT/G/E query16→原件分析，停于此，不启动新warm/RL/held400；主讨论消费后决定接续。
只在216对task13/17原相同真实V/L/MT-H/P/xi重复已有简洁输运/传播统计，不追加probe系列或训练修复臂。

### 11.3 输出、资源、证据与回报

ROOT=/data1/user/ymdai/ember_runs/video_guided_proposal_writer_20261012/normalized_flow_query16。
原E/MT/base/S_G/真实H用绝对引用或小metadata，禁止复制大资产/旧教师重训。保存完整216恢复CP、fresh initial模型、全部候选参数/真实H、
固定查询raw rows、completion与实际计费。新增峰值估16GiB（完整CP/模型<2GiB、真实36条G H及原生raw/cache约12GiB、余量），
launch前核strg01 data1独立quota和在途旧root峰值，共享df另查。所有新增位于data1。

旧G216定位的稳定原件为ROOT/analysis/previous_G216/native_diagnostic/{diagnose.py,statistics.json,analysis.md,launch.json,exit.json,accounting.json}；
主讨论CPU复算为ROOT/analysis/previous_G216/initial_generation_analysis.json，推导transport_reasoning.md亦保留。
它们明确是previous_G216，不是新方法结果；原旧ROOT、CP/冻结代码继续保留，不把tmp作为唯一持久证据，不新做hash。

工程准入暂20–40min；新学习+首学生读出暂2–5h，G约50–100min、控制读出1–3h，由当前拓扑/完整B4/峰值更新预测。
最多1728逻辑query行+36条G实践，E=MT及合法旧MT引用减少真实新增；计费不重算历史复用。
现场双节点总6或8/node6、显存/非干扰合同保持，允许有收益的共驻，不等待world4、不dummy占卡。
新CPU协调器仅这一有限依赖图及单次整批完成/异常回报；旧RL9/strict400原冻结合同和唯一退出回报者并行继续。
实验闭环实现/验证/Git/push/clean detached/运行恢复；整批完成或科学边界只向01a12bb0主讨论一次可靠回报，不逐阶段心跳、自Queue或重复等待。


## 12. 当前有界批：确定MT中心完整端点与同面板query16（2026-10-12）

目的为检验去掉两轮均未学成的独立参数噪声输运后，现有真实条件能否传递共同E控制优势。
完整数学关系、旧expert-manifold§24–29对照、公共均值风险与可失败预测见
[端点参数化科学决定](../analyses/proposal_endpoint_parameterization_20261012.md)；本节是唯一执行合同。

### 12.1 参数、算子与学习消费者

固定P0为MT合并基底上的初始完整38-target/rank8 LoRA（A0非零/B0=0），S_G不重估。
当前P仍输入p=(P−P0)/S_G；唯一六层decoder每块读取[p,valid]（Linear128→256）及原地址、实际ContextReader memory。
保留六层发送端归一化和末LayerNorm，输出Linear256→64的weight/bias零初始化，预测e_hat；
部署P_new=P0+S_G*e_hat。目标固定e=(E−P0)/S_G，不用E−P作端点目标，也不把输出误加回当前P。
移除生成参数的xi、x_t、t及16步Euler，不计算1/(1−t)。原生policy十步action flow及教学/实践/query全部RNG保留。
exact L/action-hidden有序双RGB/stride5、双rank32 meta、真实非空MT bootstrap H及producer、static两层与Actor三head实际算子不改。
E/teacher动作/state/reward及任务元信息不进入condition；本批读取P/H不构成反馈利用或视频因果已证实。

G/meta/Actor/optimizer全部fresh seed7，双meta原seed不变；v7 schema明确拒绝旧v6/旧参数状态。
唯一G更新owner由CFM替为masked endpoint MSE，初始及未来joint共用同一owner，不保留旧flow fallback或另建runner。
全36既有allowlist、每task24次、216更新/864条件/logical B4、原任务/parent/H/video采样与AdamW
lr1e−4、betas(.9,.95)、eps1e−8、wd1e−4、clip1、数值精度不改。删除xi/time不改变独立video采样。
梯度给G/Reader/static/双meta，source/Actor无梯度；零head首步上游零梯度符合初始化定义，随后由真实监督打开。
只在216保存完整共享模型/optimizer/sampler/rank RNG/topology；记录endpoint loss及同标签MT基线，不与旧CFM raw loss比较。
公共均值MSE .813884（MT1）仅诊断，不作teacher均值adapter/检索/额外loss或选点。

### 12.2 唯一候选、固定条件与配对对照

每个实际P/H/V/L决策一次forward只产一个完整新候选；不计算/复制四个确定重复提案、不伪造noise_seed。
非finite/非法候选按真实失败，不重复同输入采样恢复。以后真实H或parent改变仍可重新生成，非固定适应轮数菜单。
本批G216后36首学生各沿normalized-flow原path的teacher_demo/path_ordinal/practice noise_root，以及同一MT bootstrap H/P。
prepare冻结完整36映射并核实践初态/RNG，首次唯一G实际实践一次，不按成绩补采；不以旧G产生H作新首学生条件。

保持practice14/query16/teacher independent report16/train report4全部state池与scene/RNG；query root仍seed(70,task,0)。
新G每task16行，共576；精确兼容的normalized MT/E576及旧G576原行引用，不重跑或改变seed求缓存。
E=MT九task仍引用同MT行，E及旧G不进G/Actor候选、H、信用或梯度；旧v6路径仅读取原JSON控制原件，不能由v7恢复生成。
query内每task两行（72）曾teacher选点，明确只作train机制读出，并分列余504；不能称held/独立teacher泛化/正式资格。
报告MT/旧G/新G/E四臂per-task/suite、覆盖、R/G/L/churn/成功集重合及新G在E-only/MT-only/both-success/both-fail结果。
从36已保存端点作一次CPU MSE/目标方向/task差异/公共均值算术；已有query首次动作可直接汇总，不开旧输运或新probe。

### 12.3 有界停止、物理安排与可靠交付

ROOT=/data1/user/ymdai/ember_runs/video_guided_proposal_writer_20261012/deterministic_endpoint_query16。
只执行fresh G216→36首学生真实实践→新G576 query与兼容MT/E/旧G引用→完整科学交付；兼容时新增612 episodes。
真实不兼容才补必要对照并保留原因/成本。完成停止，不自动warm/RL/held400/Teacher/视频controls/相邻点/科学配置扫描。
初估工程30–60min、训练+首学生1.5–4h（基于旧world2 G1.491h外推），依实际消费者/吞吐/live设备修订，不称已测新图。
新峰值估16GiB，复用初始参数/MT merged/E/S_G/H/大模型，旧原件/冻结消费者/关键恢复点保留；所有新增data1。
现场双节点quota/总量6或8/node6/非干扰；按真实完整B4和最长输入峰值选择实际world/chunk/worker，不能固化历史world2或忽略显存余量。
formal使用clean pushed detached代码，精确命令、冻结身份、费用（含profile/失败/历史引用）与退出收据记录在ROOT。
唯一有限coordinator等待实际子进程退出，只一条整批完成/真实异常接续；无子阶段心跳、自Queue或重复等待者。
实验持canonical/Git独占窗口，闭环工程/恢复/集成push及分支worktree管理；实际启动允许一次实质身份/预测回报main，随后只整批回报。
主讨论接收方01a12bb0-5273-7462-b5e2-72a18aba7b8d，科学阴性由main消费取舍，不当工程bug挽救。

### 12.4 实际消费者与首次物理安排

54项聚焦检查通过；真实上一批MT/E/旧G各576行兼容，不需控制补跑。实际36教学/实践映射由prepare读原path冻结。
同node两完全空闲卡gpu01:0/6的world2最长合法92帧/logical B4实测：chunk16热20.50495s/峰26.322GiB，
chunk32热20.50381s/峰42.186GiB。无吞吐收益，选择16；更大chunk按实际增长约超45GiB容量，不作进一步尝试。
source始终无梯度、首步等真实MT端点标签基线、后续双meta真梯度已打开；profile全部更新/optimizer丢弃。
原件ROOT/profile_physical_chunks/selection.json与attempt_001；116.440s/.064689GPUh记账，不当科学训练节点。
正式有限批已从clean pushed detached 48fa1247db1845831ec9bac5afd865ad86cbd4eb启动；唯一37-job owner及
实际双rank chunk16/logicalB4身份保存在ROOT/coordinator/deterministic_G216_students_query16_01。
两节点live检查后的分配为gpu01:0/6，正式缓存也在ROOT/cache，data1独立quota覆盖16GiB峰值预测。
工程开发树/分支与未运行候选冻结树已退休，正式冻结树、科学原件、profile与复用验证保留；准确清单见该owner的development_tree_retirement.json。
启动身份与当前状态以progress顶部为准；这次启动和真实传播/梯度验证仍不构成端点控制有效性结论。
当前只两同node完全空闲卡、其它有他人任务，先world2正式，不等world4或跨node碎片；首学生同两卡动态long-first执行。
G预测3600–6600s、读出1800–5400s、整批1.5–3.5h，参考旧G1.491h及本次只612新增episode，实际每次launch再次live准入。
默认Runtime16/launch每rank实际chunk消费者已验证，正式chunk ceiling16；不保留无当前吞吐收益的更大chunk活动运行面。

### 12.5 原216有限批执行完成

48fa1247/v7实际G216+36首学生/query16整批退出0；严格576条件MT340/旧G339/新G341/E455，
MT→新G R314/G27/L26、net+1、churn53，未形成明显超过MT信号。
端点MSE .961878，输出97.677%编辑能量来自公共均值，条件差异只改善.001424；首5原生动作仍近MT。
实际完整CP/optimizer/sampler/RNG、每task24训练事件、映射/池隔离/native配对通过；不把内部改善称控制或视频资格。
612新增episode/143672环境步，整批2.078h/4.121798GPUh、无GPU失败，旧控制引用未重计。
本节有界运行已经按原合同停止，没有自动warm/RL/held/Teacher/controls/probe；独立新授权见§13；
完整原件/得失/公共方向与竞争解释见[端点整批分析](../analyses/proposal_deterministic_endpoint_query16_20261012.md)，
当前消费交付状态看progress，后继科学取舍由main依据完整结果自主登记。

## 13. 同一v7图有限学习时长检验：完整216→864（2026-10-12）

本节有限批已完成，执行结果见[整批分析](../analyses/proposal_endpoint_learning864_query16_20261012.md)：
同576为MT340/G216341/G864341/E455，时长增强参数条件拟合但控制net0对216、net+1对MT。
以下保留事前合同，不再表示在途运行或自动续训授权；后继只认progress另行登记。

完整事实、算子解释、最近似历史反证与结果分支见
[学习时长决定](../analyses/proposal_endpoint_learning_horizon_20261012.md)。该文是本节的完整推导，
本节只登记唯一执行合同；不是第二份active design。原216ROOT/contract/结果不改。

### 13.1 为什么修订216出口判断

216只有24次/task，完整36-task loss持续1→.962570、原生meta信用主要后期增大，未有充分收敛证据。
原输出97.677%公共能量、条件增益.001424和闭环341对MT340仍是阴性。
显式修正上一端点决定§6的“阴性后默认改机制”：216是早期行为出口，不充分代表训练后的架构裁决。
零head初期仅末层得到梯度，之后Reader/meta经W_out转置信用打开；共同特征/标签可先支配末层外积。
这给出有限时长竞争解释，不保证条件协方差学成。旧直接因子回归公共方向负证据仍有效，不无限续训。

### 13.2 恢复、监督与唯一更新变量

- 从完整48fa1247/v7 CP216继续至总864，新增648更新/2592条件，每task累计96。
  保留ψ/双meta/Reader/static/未训练θ/optimizer/sampler-cursor/rank RNG/psi_version216及监督version0；非fresh重跑。
- 同源冻结policy+MT、P0/S_G/共同E/36任务/语言/真实action-hidden视频/MT parent/原MT-H、原采样、
  masked endpoint MSE/AdamW/clip/logical B4/native10步flow及RNG；唯一科学变量为学习时长，不改图/损失。
- 新ROOT的实际refresh_bank严格MT_only，supervision_version0；不读取216学生P/H或G_student_bank，
  query永远不入H/梯度。不能称本批已学习任意parent反馈修订。
- 可保存432/648完整恢复CP，不生成学生、不读出、不选点；updates.G只登记864科学出口。
  初次明确--resume原CP216，故障恢复选新root中最高完整恢复点，保留原有效更新和失败账目。

### 13.3 固定全36读出与停止

864后沿原36同teacher_demo/MT-H/P/实践初态/env及policy RNG映射，各一次确定G生成与实际实践；
同query16新G576，与兼容MT/G216/E原行组成四臂576严格配对。E/G216只读，不入条件/候选/H/信用/梯度；
E=MT引用同一原行。预计612新增episode，未改变14/16/16/4池和72选点重叠/504其余分列。
报告任务/suite/train24-support12/breadth/RGL/churn/success overlap、E/MT成功类别以及216→864能力得失；
保存端点只做一次已有CPU分解并复用首5动作记录，无新forward/probe。

到864+首学生/query16完成即停，不自动Actor/warm/RL/held400/Teacher/controls或>864。
若仍公共拟合主导且无控制净增，停止再延长当前条件编码+全坐标回归组合，即使loss仍下降；
不以LR/rank/seed/scale/归一化小扫保护它。若参数条件增强却无控制增益，优先重新论证功能传递；
若控制净增/保持合理，再由main另登记相邻/held/视频证据，train576不替代正式资格。

### 13.4 资源、冻结与交付

新ROOT=/data1/user/ymdai/ember_runs/video_guided_proposal_writer_20261012/deterministic_endpoint_learning864_query16。
复用大资产，新增峰值16GiB；现场双节点GPU与strg01 data1独立quota，共享容量分别核验。
原两rank216训练1.113h/读出.955h，新增训练外推3.34h；整批暂2.5–5.5h，工程20–40min。
物理rank不是科学变量；完整CP边界可正确迁移、记录optimizer/sampler/已有与新增RNG来源，不能绕过校验/称bitwise。
相同布局旧最长视频chunk16/32无wall差别、reserved26.322/42.186GiB的profile可复用，保留更新丢弃证据。
本次现场两张完全空闲gpu01:0/6适用原world2；读出按余量可共驻gpu01:2低占用DA3服务，不把world2变成整批上限。
常规合计6/8、单node6、非干扰与准确GPUh记账不变，实际选择及预测只由真实准入/吞吐修订。

实验独占canonical/Git，完成工程消费者验证、push/clean detached/实际启动及分支worktree生命周期；
main接收01a12bb0-5273-7462-b5e2-72a18aba7b8d，无工程ACK停点。
启动实质身份/ETA一次投递，随后只整批完成/真实异常可靠接续；一个CPU整批owner等真实退出，
不增心跳、自Queue、子阶段投递或第二等待者。

### 13.5 实际出口与停止

37正式job实际退出0，完整v7 CP864、485 optimizer step864/cursor864/RNG2、2592新增条件/每task72均核验。
MT→G864 R309/G32/L31、G216→G864 R313/G28/L28；train24较MT净−1、support12净+2。
一次保存参数CPU MSE .843335，输出公共均值占69.057%，相对自身均值的条件MSE增益 .054377；
这较216有实际条件学习，不恢复“完全恒定输出”的解释，仍未传递E455的控制优势。
新增612episode/145036步、4.305975h/8.503693GPUh；原件/得失/边界与账目由整批分析引用。
本批到此停止，不自动延长或后段；432/648仍仅恢复，完整CP保留不产生新科学授权。后继由main自主取舍。

## 14. 完整部署BA算子监督：v7 G864→新v8 G1296有限接续（2026-10-12）

完整原件、推导、最近似历史正反边界、结果分支与成本见
[算子监督科学决定](../analyses/proposal_operator_supervision_20261012.md)，本节是唯一活动执行合同。
原864分解总因子改善.156665中A贡献.156504、B仅.000161；实际BA的G/E norm ratio .021085、cos .020679。
这解释因子loss为何未代表执行写入，不证明唯一根因或修复必有效。

唯一主变量把G更新目标换为38层mean(||s*B_G@A_G−s*B_E@A_E||F²/D_l)，
D_l=全36原E算子能量等权均值，全部正且有限；九零E仍等权，不逐样本归一化或加floor。
使用完整A=A0+生成变化、完整B；精确rank Gram在正常FP32中计算，原模型BF16/数值政策保持。
原factor-MSE只允许无梯度廉价诊断，不混入训练；不叠nativeFM/action、B权重、冻结A或新decoder。
E及D始终在监督侧，不进入教学、H、候选、信用或部署条件。

新ROOT=/data1/user/ymdai/ember_runs/video_guided_proposal_writer_20261012/operator_effect_learning1296_query16。
唯一G路径显式导入原完整v7 CP864的psi/meta/Reader/static/decoder/theta/Adam/cursor/rank RNG；
目标变更后schema为ember_full36_operator_effect_writer_v8，不称旧合同exact-resume。
新v8同目标完整恢复有效；不能普遍接受旧v7或重置零head/Adam。
同P0/SG/MT-only真实H bank0、36任务等权/logicalB4、原视频采样与原生policy RNG。
新增432/1728条件、每task新增48次/累计144次，保存1080仅恢复、1296唯一出口；不得导入学生bank。

1296同原36实际teacher_demo/MT-H/P/实践映射各一次确定G和真实实践，同query16新576行。
兼容MT/G864/E576逐行只读引用，E=MT同源别名；预计612新episode，实际引用来源和历史费用分开。
报告任务/套件/train24support12/breadth/得失/churn/重合/E分类与72选点重合、504其余行；仍是train机制读出。
终点仅一次已保存端点CPU A/B+BA及公共/条件分解，首次原生动作复用已有记录。
本批不启动Teacher/Actor/warm/RL/held400/controls/probe或>1296；整批结束交main科学裁决。
算子弱且控制弱将降低当前Reader+完整因子decoder共同E学习假设，不能靠延长/小扫/重启噪声保护组合。

新增峰值16GiB、大资产复用data1；工程预计30–60min，正式2.5–4.5h（旧两rank外推中心3.2h）。
live双节点/独立quota和真实新目标梯度、恢复消费者由实验验证，按实际吞吐选物理安排；不固化world2。
旧最长chunk16/32读取profile可复用，但不当成新loss消费者验收；实际验证更新全部discard。
工程/Git/clean pushed detached/运行恢复/分支生命周期由实验闭环，无main ACK或Owner新许可门槛。
一次真实启动身份/ETA及唯一整批完成或真实异常回报给main 01a12bb0-5273-7462-b5e2-72a18aba7b8d；无心跳/子阶段Queue/第二等待者。

§14绝对时间修订：Owner最新要求2026-10-12 UTC+8 16:00（UTC08:00）前结束，优先15:50完成消费，
最迟15:55本批GPU全部退出且不再派发；原2.5–4.5h预测保留为历史，不允许移到gpu02延长。
独立双节点ops guard身份/命令/CPU信号验证见上述ROOT/coordinator/deadline_ops_20261012及run_contract.absolute_deadline_revision。
15:50只SIGINT登记CPU coordinator停止新派发并让在途job收尾；15:54 TERM、15:54:30 KILL本批残留，15:54:55最终核验。
进程按同UID、精确root/consumer argv和start_ticks验证，仅本批owned范围；该时间措施不增加完成等待者或通知。
保持1080恢复only/1296唯一出口；不足则保全最近登记完整checkpoint和真实已完成原件，报告未保存更新及缺task/rows，
不把部分结果冒充576整批、不自动截止后或重启续跑。物理安排仍可在科学合同、live余量/非干扰和6/8卡上限内改善。

§14物理读出修订：继续原训练来源2a1fb813及G进程，从完整1296的真实退出原件交接到独占学生condition动态队列。
复用现有student-data入口、Runtime/CP/EpisodeRunner、compile_conditions/query_conditions和SQLite队列；每worker只加载一次，
普通单task也走同一refresh_group主体，不保留平行科学实现。完成的真实practice立即保存，各query逐行保存并按原映射/RNG去重。
36原condition、576新G query及36 practice共612新episode不变，MT/G864/E仍兼容只读引用；不消费held或把query反馈写入H。
G退出事件live核双节点并选择至多常规6/8、node6个distinct GPU，允许满足余量及非干扰要求的高util共驻；
physical候选为free≥40000MiB的slots32/group3/chunk16、≥28000的16/2/8、≥18000的8/1/4。
这是物理初始安排，不是新科学超参或已验证峰值；首原定组记录实际GPU/EGL余量、batch histogram与吞吐，没有额外episode。
G~15:30后六卡若接近旧服务速度，读出预估15–25min；竞争、native服务及端点改变的episode长度使完整截止尚未保证。
原CPU coordinator仅PID级SIGINT停止旧学生派发，ThreadPoolExecutor继续保留G真实退出；新owner持原通知非阻塞lock，
原helper在发送前被排除，由新owner唯一整批交付。初始与V2 guard历史、精确registry/交接/实际新来源均留run ROOT。
15:50不新派GPU job或领取condition group；在途组由原绝对guard收尾/停止，保存完整practice与已完成query，不降低1296/576合同。

§14实际停止：G于15:29:43退出0，432新增更新完整保存到v8 CP1296，1080恢复点及原864保留，未保存G更新0。
36实际端点/实践齐全，只有33条件/528 query；截止前31/496为MT302/G864308/G1296=314/E394，MT→G net+12。
task5/7的32行在15:59:27整组完成，MT13/G86411/G7/E26；全部保存528为315/319/321/420，MT→G net+6。
task32/34/35的48行因gpu01:6 EGL framebuffer错误缺失；不以旧G/参照/重复行补全。
gpu02 guard15:50 CPU coordinator身份校验异常取消GPU终止时刻，最后gpu02:4在15:59:52退出；15:55硬边界没有兑现。
退出现场双节点已核无本批GPU，原guard/错误/逾时原件保留；定向候选只做CPU验证、没有重新armed。
唯一整批owner15:59:53退出1，实际v8恢复/信息墙/配对/费用消费通过不代表完整576或科学资格通过。
BA端点loss .961445/MT1、聚合norm ratio .092718/cos .099524；已保存有限控制局部收益不能替代缺失全36读出。
完整范围、任务得失、一次算术、费用及工程责任见对应分析§11/findings§443；总6.505926GPUh含失败与profile。
本批到此停止，不自动补query或任何后段/>1296；原恢复点及关键原件可用，不授权截止后/重启续跑。
后继科学由main自主消费实际部分证据另登记，不以Owner新许可或工程ACK使研究停滞。

§14恢复后新leg（Owner2026-10-12 16:37 UTC+8明确确认gpu01已恢复）：仅补task32/34/35各原query16。
沿保存唯一G1296端点/原teacher_demo/MT-H/P/query初态及env/native RNG，禁止重新生成、重复实践或续训。
实际新增成功只48条G query，MT/G864/E兼容原48行只读引用；其余528行（含不利逾时32）不重采/择行。
原截止内496/逾时32/新恢复48分别记录，完成后新analysis合成全36/576；旧partial/exit1/guard失败不改写。
旧绝对时刻仅属已结束旧批，本次没有新截止，不复活旧队列或旧guard；新且单一整批owner只消费三项缺失读出。
优先用c7e76b56既有冻结restore→query消费者，实际CPU准入已核保存权重/48请求/兼容控制，代码未变不另建工程分支。
按现场双节点GPU/非干扰/6或8卡、node6和data1独立quota选择有效worker；新增估<1GiB，大资产复用。
暂估工程/启动10–20min、Long读出10–25min、全消费5–10min，总25–55min；启动身份/真实准入后修订，不承诺未测提速。
算子空间mean(sBA)分解表明真实条件增益.02594551、公共能量44.2479%，不是纯公共坍缩；完整证据/边界见对应分析§12。
固定MT-H/共同E可以学B，缺多种失败H不能解释当前B弱；H覆盖只限定后续反馈修订资格。
本次无新算术/GPU分析/Teacher/Actor/warm/RL/held/controls/probe/>1296；一次实际启动及唯一整批完成回报main。

§14恢复后leg实际完成：16:56:31启动、17:03:03退出0，c7e76b56冻结/三个gpu01独立worker slots16/chunk8。
三个原保存端点及实践/H未重做，只48新G query/18457步/.297693609GPUh；实际CPU消费和双节点GPU退出核验通过。
全36/576为MT340/G864341/G348/E455，MT→G R310/G38/L30/net+8/churn68；旧G→G R305/G43/L36/net+7。
时间集合496/32/48分别302/308/314/394、13/11/7/26、25/22/27/35，原不利逾时行保留、旧exit1/guard违约不改写。
target24净+5、support12净+3，Spatial−7/Object+6/Goal+1/Long+5/support+3；E-only134传22，非明显超强MT资格。
72选点重合与504其余分别43/43/47/55和297/298/301/400；没有held/视频因果/相邻稳定性资格或新控制。
完整结果/机制边界/36task原件/累计费用见既有分析§13、findings§445，ROOT/analysis/query48_recovery_completion_20261012。
累计formal6.727676593+profile .075942840=6.803619433GPUh，612完成新episode/步下界142758，旧失败步未知。
完整CP和关键权重/H/raw保留；到此不自动任何后段，由main直接消费后续科学裁决，不新增Owner/工程ACK门槛。

## 15. 已结束：fresh匹配参数块直接memory读取（2026-10-12）

### 15.2 最新Owner停止裁决覆盖原864义务（2026-10-12 23:00 UTC+8）

main接受Owner“没必要非得864步”的纠正，撤回必须等待两864完整匹配终点的决定。
半程BA约.98及G432337/MT340/E455已足以判断当前G未有效拟合教师，不继续支付本批训练/未完成读出。
22:57:53先冻结唯一scoped派发，23:00:19全部本批GPU进程退出，23:01:51双节点确认本批/ymdai GPU进程均0；待派发依赖取消，不恢复旧队列。
control完整CP864、candidate完整CP432保留；candidate两rank日志到641，已记录未保存433–641共209更新，未记录642状态未知。
早期432的36/576完整结果保持；control864仅351有效完成query/24task（16task有query组完整收据），candidate864无query。
原1728期望与864匹配比较已取消，partial不得称完整匹配；所有有效原行、旧失败、费用和冻结来源保留，实际收口见对应分析§14。
当前没有执行中的active实验合同或后继架构/实验授权。main继续学习原理分析；新投入先取得能改变判断的有效学习证据，
不机械承诺每个架构五六小时或固定终点，也不形成永久长训练禁令/新步数菜单。未拟合的是G而非无能力的E。
下方§15.1及原864/资源/ETA均为历史合同；最新停止优先，无Actor/warm/RL/held/第三臂或自动恢复。

### 15.1 Owner授权的候选432提前诊断与优先派发（2026-10-12晚）

最新阶段纠正：先通过MT起点一次G修订的完整闭环与保持，再训练/评估完整Writer；11:00不是必须完成Writer/400的合同。
22:15有界运行核查及worker03真实EGL故障后，派发由scoped_readout_recovery_02独占恢复；原34早期条件/544行保留，仅补两缺项。
后续读出组2→4、slots16/chunk8/coalesce0.1s，端点后释放未用CUDA缓存；模型/数据/标签/目标/schema/CP及每行RNG不变，未声称wall收益。
控制读出使用62e615c9/v8旧图同等缓存修复，候选读出eac6ebf4/v9；原训练冻结身份继续。原旧CPU派发停止/实际退出收据保留，迟到回报按handoff判重。
独立就绪工作不因局部worker故障停派，失败有界恢复且完整计费；当前六卡有效工作/早期与两864的唯一科学回报保持，无新后段。
物理证据、未采用slots32的余量约束及未测边界见现有分析§12和progress；不为运行优化修改科学矩阵。

§15.1提前诊断实际于22:34:58完成：v9完整CP432、36真实条件/576新G query、0practice/共享更新，G337/MT340/E455。
原544有效行保留，仅从原保存端点补task95/96的32query；失败/恢复收费和原配对均核，完整结果见现有分析§13。
R311/G26/L29、72选点重合净+4/504其余净−7，未取得当前G阶段净增；没有held/完整Writer/正式选择或结构归因资格。
两864既定匹配训练/读出继续，由main消费早期完整原件裁决，不因半程阴性自动停掉原训练或追加科学矩阵。

Owner要求四卡先完成后立即做有用工作，并允许提前查看训练结果；main在未读取任何新闭环分数前登记：
增加唯一block_memory_candidate完整CP432的全36条件、每condition同原query16，共576条提前诊断。
同exact L/原teacher_demo/真实MT-H/P0/初态/env/native RNG，唯一完整LoRA，0新实践/共享更新/额外native探针；
MT340/E455/G864341/G1296=348原兼容参照只读引用，不补control432、不选择容易task或缩为query4。
432原为恢复only，现明确改为一次提前科学诊断；864两臂训练与匹配比较保持，禁止回退重训或丢未保存更新。
候选432与control864曝光不同，不能用于memory结构因果比较；432不是held、完整反馈Writer或正式checkpoint选择资格。
完整576先交main及时裁决，阴性不自动推出864不可能改善，也不自动按最高分选择CP或添加其它中间点。

唯一新增576query，总期望1728；输出独立ROOT/early_candidate432_query16，CP/大资产只读引用，预计新增<1GiB。
控制实际退出后四卡优先早期432，完成立即接control864；candidate训练继续两卡，864就绪后按有效空余slot读出。
在途GPU工作及实际收据保留，派发权串行交接，只有一个实际dispatcher；旧冻结模型代码不原地修改。
所有准入仍须live双节点/6或8总量/node6/非干扰及data1独立quota；沿已实测slots16/chunk8/conditionbatch2，不新profile。
当前11:00为交付目标，不设强杀截止。早期按共驻成本预计1–2h，实际启动与新整批ETA以progress/ops收据为准。
早期完整结果是Owner要求的独立科学决策事件，允许一次可靠提前交main；原两864完整交付仍保留，分别判重。
无心跳/每task状态/第二整批等待者，无Teacher/Actor/warm/RL/held/Test/第三臂/>864自动授权。
下面原始注册中的“432恢复only/864唯一出口/1152episode”保留历史，涉及早期读出的当前执行语义由本修订覆盖。

本节替代已完成§14的执行状态，保留其全部失败/逾时/恢复完整576及348小幅收益记录。
完整数据—算子—学习—部署推导、925e7b1最近似负证据、结果分支与停止线仅由[匹配结构分析](../analyses/proposal_coordinate_memory_read_20261012.md)承载；本节引用为完整合同，不复制第二active design。
ROOT=/data1/user/ymdai/ember_runs/video_guided_proposal_writer_20261012/coordinate_memory_read864_query16；target_memory_control/block_memory_candidate两臂均fresh全部共享状态/未训练θ/Adam/sampler/RNG，seed7/meta原seed，同真实MT-only H/bank0、同36/P0/E/S_G/D38、BA目标从首更新、logicalB4/原优化器。
唯一结构变量：六层MHA从38target入口移至10064block的local FF后/pool前；旧target入口只在clean pushed frozen控制源，canonical仅候选，不保留双路径。
两臂864更新=3456条件=96/task，432仅完整恢复、不读出；864唯一出口，不迁旧权重/学生parents/bank，不自动延长。
各36确定完整38-target端点直接同query16=576，无额外G-practice；总1152新G query，MT/E/G864/G1296真实兼容参照只读复用，72选点重合/504其余分别解释。
新schema拒旧图；物理执行按真实最长完整L/V/H可丢弃profile及现场双节点吞吐选择，保持完整softmax/梯度，不截断memory。
常规live6/8、node6、data1独立quota/16GiB预估，无GPUh硬门槛；Owner恢复授权后没有新绝对截止，不复活旧guard/队列。
初估工程30–60min、并行训练/读出3–5h，以实测修订。一个整批owner及时运行就绪臂/读出，一次真实启动身份/ETA和一次整批完成/异常交main01a12bb0，无心跳/阶段Queue/第二等待者。
无Teacher/Actor/warm/RL/held/video controls/probe/第三臂/>864授权；本批结果须更新机制支持度与main自主后继，不设工程ACK。

§15实际执行补充：17:46控制c7e76b56/world4/gpu01:0–3已启动，18:12候选c35c34b5/v9/world2/gpu01:4/5已启动；控制直接读出c4f53841保持旧v8计算图。三树均clean pushed detached，canonical仅block路径，开发分支/树已核实集成后清理。
62项针对性检查与真实fresh prepare/直接query CPU消费者通过；完整最长L/V/H候选profile退出0、5更新全部discard/0episode/.075387129GPUh。chunk16实测20.602秒/26.10GiB reserved，chunk32无收益且42.79GiB，采用16。完整原件/成本/物理依据只在对应分析§9及ROOT，未获得CP864/控制结论。
普通cap6下控制4+候选2全部有效使用；后续动态读出不受训练rank数限制。候选训练4.4–5h加读出.5–1h，18:12启动后暂估23:10至次日00:15；原3–5h预测保留为初估。唯一owner matched_coordinate_read864_01已独立存活，不加整批等待者/阶段Queue/心跳或新截止。

## 16. 原144真实context冻结与endpoint短拟合（本轮已完成）

2026-10-13实际完成状态：cb27adfc冻结来源，23:42:42启动/00:00:17退出0；144完整m缓存102.997秒，四rank仅原endpoint从432至8883。
全144起点.97882444、5min .83078116、15min .57489586；第一次值允许继续，15min完整全36边界900.845秒保存终点并退出。
新增8451更新/33804条件/939次每task，166 Adam均step8883/完整sampler-RNG-topology及冻结CP432引用、144端点和原36固定LoRA齐备。
未达<=.50实质拟合调查信号，保留部分进步；本轮执行范围已结束，不自动加时、原query16或后段，main消费原件后自主登记后继。
实际分布、条件分解、吞吐比较/未测边界、费用与记录欠项在现有coordinate分析§15.4和progress；下方为本轮预登记合同，旧§15不恢复。

本节是Owner要求研究继续后main登记的新有限授权，不恢复已停止§15，不等待Owner/main工程ACK。
唯一完整科学解释引用[现有coordinate分析§15](../analyses/proposal_coordinate_memory_read_20261012.md#15-冻结全144真实context现有endpoint的有界共享拟合)。
ROOT=/data1/user/ymdai/ember_runs/video_guided_proposal_writer_20261012/frozen_context_decoder_fit。
源是旧block_memory_candidate/checkpoints/G/update_00000432完整v9；433–641未保存更新不使用，控制351partial不补、不作匹配结论。

全36=train24+support12，原policy_videos每task4条，共144完整合法context；exact L/action-hidden有序双RGB/stride5、原非空MT-H真实producer/P0/S_G/D38/E路径保持。
m=冻结CP432的Generator.context(real features,real H,actual states)，保留全部memory，不平均/压缩frames或只取decision，不读offline teacher动作、held、E/ID条件。
144只是36独立E目标的示范变化；各task固定L/H/E，允许仅L识别train task的统计解，故拟合不能识别video/H贡献。固定MT-H本来可学共同E，不以缺多H解释当前弱拟合。
原meta/Reader/static/layout/theta/source/MT冻结；唯一梯度owner是原generator.endpoint的input/地址/全部六层/output，不加head、自由code或任务私有LoRA拟合。
保持完整A=A0+S_A*y_A、B=S_B*y_B与部署s*BA目标；正常BF16、BA loss FP32、rank8/alpha8/38targets/原AdamW lr1e-4、betas(.9,.95)、eps1e-8、wd1e-4、clip1。
endpoint权重与166个逐参数Adam state从CP432按保存名称与实际构造顺序承接，不按shape猜、不reset动量；其余330参数不step。
显式梯度owner/物理topology迁移，非exact-resume；cursor432，原logical_tasks与随机demo调用顺序、等权logicalB4/bank0/MT-only H/P保持。
旧rank0/1 RNG恢复，新rank来源明确登记；无scheduler/LR/rank/scale/seed矩阵或学生bank。

先缓存144并读一次全池CP432 baseline；训练wall从首个有效optimizer.step计。
首次超过5分钟后的完整9更新/全36轮边界读全144：L>=.90或L<=.10立即保存终点退出；.10<L<.90允许到超过15分钟后的首个全36边界全池终点。
只读预登记节点、不挑历史最好点，报告真实更新/曝光/吞吐/峰值/读出时间。L<=.50只是实质教师拟合的后续调查信号，非控制/完整Writer资格；.5<L<.9也不自动加时。
5/15分钟和阈值是本次有限投入规则，不是数学必然/性能标准或未来固定课程。弱结果结合因子864、BA1296、coordinate432与历史负证据关闭当前真实Reader+共享坐标E回归组合及入口/宽深补丁投入，
不宣称所有Reader/decoder/教师监督不可能；positive也不自动重启几小时native共同训练。

保存全部144起终点loss、task/train24/support12/suite均值，原36固定teacher_demo最终各一套完整LoRA，大小合理时保存全部144端点。
终点完整endpoint optimizer/sampler/rank RNG及冻结组件CP引用明确；不能当旧全psi exact-resume。小Gram条件分解在同一读出中完成，既有teacher均值原件直接引用，不重跑旧CPU分析。
其部署链仍是L/V/自身H→冻结真实C→同共享F→38处完整BA→当前自身hidden/十步原生动作→固定LoRA闭环；缓存不是新部署特权。
无0成本泛化或video/H因果宣称，无新environment episode/Actor/fullWriter/held/Test/control/probe/free-code/shuffle/reverse或自动后继。

缓存可独立任务分片，结束即释放大source/meta；decoder物理安排按实际有效B4吞吐/峰值选，不固定旧world2或为填显存用多于有效条件的rank。
各launch双节点live、合计6/8/node6、非干扰，data1独立quota/实际用量/共享容量；峰值预计<8GiB，资产原位引用，无GPUh硬cap。
最小诊断consumer复用FactorEncoder/OperatorObjective/原Runtime，仅当前诊断生命周期使用；不是第二canonical Writer/fallback，结束后按main家族裁决决定退休入口，Git/冻结原件保留。
工程预计30–60min、缓存5–20min、训练5或15min加两次全池读出均为待实测估计；不设新绝对截止，不承诺正结果或精确95%提速。
一个实际调度owner、唯一持续退出等待；一次真实启动/ETA与一次完整结果/实际异常可靠交main01a12bb0，无心跳、阶段Queue或第二等待者。


## 17. 已完成：固定terminal8883保存端点的原36/query16闭环（2026-10-13）

main已直接消费§16全部144原件、实际目标与历史，保留原<=.50调查信号未通过：全池.574896，原36面板.579024。
27非零E逐task相对自身MT基线的终点比例min .489257/median .594187/max .695858，27/27改善，36/36四视频均较CP432改善；九零E更近0。
这属于部分但广泛的实际共享学习进展，main明确授权一次完整控制读出，不追认旧阈值通过或追加学习。
主讨论两份原样小原件在原ROOT/analysis/main_scientific_readout/{short_fit_terminal_scientific_consumption.json,short_fit_interpretation_bound.txt}，不重算。
同D38下任何condition不变的矩阵输出loss下界.940825（教师均值可能超rank8，非部署臂）；原36条件/公共增益.389582/.031394，不把其分解混为全144。
拟合未识别video/H各自贡献，成功率不能由BA降低推算，固定MT-H可以学习共同E的因果边界保持。

唯一输入为ROOT/complete.json.original36_endpoints指向的terminal_15min/update8883原指定teacher_demo完整38-target/rank8 LoRA。
原ROOT=/data1/user/ymdai/ember_runs/video_guided_proposal_writer_20261012/frozen_context_decoder_fit；训练来源cb27adfc、冻结前端完整CP432与endpoint8883分列。
新唯一读出根ROOT/query16_terminal8883，saved-endpoint来源schema明确；实际读取权重/真实MT-H与pairing校验，不装载或运行Writer/Reader，也不生成新的G-H/学生bank资格。
每task同原query16/task/init/env/native RNG及官方10flow/前5动作重计划口径，36×16=576新增G episode；0新实践/共享更新/教学forward/G生成。
MT/E/G432/G864/G1296原完整576只读兼容引用，兼容不成立须实际修复或交具体边界，不静默新采参照。
不按四视频loss择video，不改用5min，不融合/重新生成端点，不运行Actor/完整Writer/held/Test/其它controls/probe。

全576完成后分train24/support12、suite/task、coverage、retained/gained/lost/churn及旧G集合重叠，列E-only134、both-success321、MT-only19、both-fail102；72选点重合与504其余分列。
若仍是337/341/348附近低量级且新能力被损失抵消，形成较大参数拟合未传成有用控制的新证据；明显超MT且保持/覆盖合理则由main决定后继。
单次train36面板不是held、完整Writer或video/H因果资格；到此退出，无自动追加时间或后段。

直接复用保存端点restore→query运行面，必要窄适配独占分支验证/集成push后clean detached执行，不热改原冻结树。
long-first/cost-balanced动态resident队列，独立就绪工作不因局部失败停派；一个实际owner等待worker退出并一次可靠整批交main01a12bb0。
每launch同时live双节点/6或8合计/node6/非干扰，data1独立quota、实际用量/共享容量；新增峰值预计<1GiB，大资产/完整CP只读引用。
真实显存余量允许一个原定组验证slots32，其余就绪worker同时使用已实测slots16/group4；按实际吞吐/peak选后续未启动worker，不为profile等待或增加episode。
准备15–30min、六卡读出40–80min、消费5–10min为已有早期576/恢复组依据的初估，启动后按实际消费者修订；无硬deadline/GPUh门槛。
只允许一次真实启动/ETA与一次完整结果/真实异常回报，无心跳、子阶段Queue、自Queue或第二等待者；旧864批永久关闭。

### 17.1 实际终点与本有限批停止

10月13日00:44:28实际完整退出0，7job均0，无恢复；36唯一保存端点与原576G-only query/全部参照严格配对，0生成/教学forward/实践/共享更新。
MT340/G432337/G864341/G1296=348/G8883=362/E455；R305/G57/L35、train24+22/support12净0、Long+17，完整任务/能力得失见同一分析§16.2。
E-only134传44、both-success321保持296、MT-only19保持9、both-fail102新增13，72/504净+2/+20；保留缺失能力/不利任务与原阈值未通过。
原GPU代码61ea7f86，CPU读出消费者f3e878e1仅修复历史未实践G的权重来源字段；不改模型/查询/原始行，不造bank资格，实际576旧行/RNG检查通过。
本次GPU读出wall768.782s、1.055261149GPUh/131251步；此前短fit含profile1.166691458，合计2.221952607，所有GPU已退出。
到此停止：main消费完整原件后自主决定下一科学干预，当前没有新的训练/完整Writer/held/Test/controls/probe授权；原§17有限合同与启动估计保留历史。

## 18. 已完成：完整8883端点状态的有界优化与唯一终点query16（2026-10-13）

main消费六臂3456实际逻辑行后授权：G362/MT340，R305/G57/L35，E-only44/134支持部分教师控制传递，但全部+22在train24，task32/36贡献14；去二者仅+8/544。
G1296→8883净+14由E-only多22、both-success少2、MT-only少3、both-fail少3组成；保持缺口不允许自动完整Writer。
§16/17完成、原substantial_teacher_fit=false/.50信号未过、旧coordinate864停止与不利原件全部保持；本节是独立新授权，不恢复native全梯度长训或同族入口/宽深/超参小扫。

唯一问题是同一冻结合法读取/共同E/BA目标下额外共享优化是否继续获取E能力并改善保持；唯一变量额外优化量，图、目标、输入/标签、task权重与超参不变。
新根 /data1/user/ymdai/ember_runs/video_guided_proposal_writer_20261012/frozen_context_decoder_fit/refinement_from8883_20261013；原根不改。
从原完整update_00008883承接166参数张量/8,761,152参数、全部Adam、sampler/cursor、四rank RNG；原冻结CP432、144完整真实m、P0/S_G/D38/E/4-video池只读复用，不重建native缓存。
owner仍原endpoint六层/地址/输入/输出；330前端参数不step，不载大source/meta作GPU计算。logicalB4、全36等权、lr1e-4/原AdamW/clip1、BF16等运算保持。
完整endpoint恢复不等于load_version或bitwise exact；优先原四rank逻辑拓扑/RNG，真实物理迁移须正确承接和登记，不绕过校验。

计时从恢复后首个有效optimizer更新起：首次越过600秒的完整9update/全36边界读全144 loss。
若L10 >= .95×L8883（L8883=.5748958587646484），保存该实际终点并停止；否则仅继续到首次越过1800秒的完整全36边界并保存停止。
两节点记录精确update、elapsed、新增更新/条件与实际task/demo曝光，不再漏节点update。10分钟规则是有限投入保护，不是性能合格线、数学必然或未来固定课程。
不选10/30最好点、不按.50追阈值、不训练到指定loss；finite异常如实处理。中间节点没有checkpoint选择资格，只有实际停止终点参与控制。
终点从原cache物化原36指定teacher_demo各一套完整38-target/rank8 LoRA；不择视频/平均LoRA。原144等权loss不能用末batch或36面板均值替代。

训练退出立即释放，原36保存端点做同task/MT-H/P0/query16/init/env/native RNG/官方口径的576新G query。
MT/E/G432/G864/G1296/G8883各576兼容只读引用，0新增教学/native context读取、MT/G实践、Actor、student bank、held/Test或完整Writer。
未实践端点bank=false，query不进入H/梯度/训练选择或中途干预。保存端点restore→query仍唯一运行面，v2登记实际terminal update/node，旧8883 v1由原冻结树保留，无canonical fallback。
六卡slots16/group4，long-first/cost-balanced动态resident队列，不重测32/静态task绑卡；局部失败隔离且至多一次有界缺项恢复，不重做有效行。

交付完整576、train24/support12/task/suite、RGL/churn/相邻8883重叠、E-only134/both-success321/MT-only19/both-fail102、零/非零E及72/504。
若获取与净收益停滞或新增损失抵消新增取得，结束原cache/endpoint堆更新；loss下降不豁免，本批后不直接追加第三同缓存时间块。
若实质进一步改善，再由main按真实缺口登记完整方法接续；无自动Actor/R5/held，不把BA接近E当控制相同、video/H贡献或未知task资格。

每launch同时双节点live、普通6/8合计/node6/非干扰；原四rank约.10s/update对单卡仅前反传.309s下界支持优先安排，非永久上限或绝对最优，不加profile矩阵。
data1独立quota/实用量/共享容量和<8GiB峰值核实，大资产/缓存只读引用，GPUh计费无硬cap。
初估准备20–40min、fit最多30min、query15–35min、消费5–10min，总1–2h；12.8min最近只读query实测不是完整Writer成本，按共驻/消费者实际修订，无强杀截止。
一个owner接训练退出→唯一terminal→保存端点→query，直接等退出；真实启动一次回main身份/准入/ETA，整批完成或真实异常一次可靠回报，无心跳/阶段Queue/第二等待者。
两份main CPU原件归档已完成8883 root/analysis/main_scientific_readout，不重算；科学解释并入既有coordinate分析§17和findings453。


### 18.1 实际停止、完整读出与本批收口

首节点update14166/600.74585秒，全144 .47569141相对8883 .57489586降低17.25%，按5%规则继续；首30min全36边界update24849/1800.71863秒，全144 .23293421后保存停止。
新增15966更新/63864条件/1774次每task，完整166 endpoint Adam均24849、sampler/四rank RNG与冻结CP432引用齐备；原144未重建，原36指定teacher_demo的terminal30min LoRA是唯一被测策略，不选10min或video。
576G-only query完整，六参照兼容引用：MT340/G432337/G864341/G1296=348/G8883=362/G24849=425/E455各576；MT对新G R313/G112/L27/net+85，8883对新G R336/G89/L26/net+63。
E-only134传97、both-success321保持305、MT-only19保持8、both-fail102取得15；train24+83/support12+2，各suite净正、21task净正/10零/5负，72/504净+10/+75。所有不利行保留，不称held/完整Writer/video-H因果。
一个fit/六query均0，02:06:41整批退出，wall42.661min；新fit2.063189811+query .962372164=3.025561975GPUh，576episode/121030步，含旧冻结阶段累计5.247514581，参照/CP432不重复计费。
本批已结束，原8883调查信号false与旧864停止不变，新参数信号true不成为方法资格。无第三同缓存块/Actor/完整Writer/held/Test自动授权，main完整消费后自主登记新的科学决定。
完整分布与实际链路解释见现有[coordinate分析§18](../analyses/proposal_coordinate_memory_read_20261012.md)；稳定ROOT/analysis/completion_20261013及coordinator/refinement8883_query16_01/exit.json。

## 19. 已完成：固定24849真实学生经历与早期教学内容证据（2026-10-13）

main已直接消费4032原行并裁决：G24849=425/MT340/E455，R313/G112/L27、21task净正、仅5task各−1、全部suite净正，去task32/36仍+65/544，满足当前G从MT一次修订明显增益与合理保持的先行要求。仍不是held、完整Writer、视频/H因果或不同经历修订资格。§18已完成；原8883 .50未过/全部负结果和§15旧864停止保留，不追加原144缓存同状态第三时间块，也不立即恢复native前端全梯度或改架构。

### 19.1 固定组件、数据消费者与36次真实实践

唯一推理组合为原完整v9 CP432的meta/Reader/static/layout/Actor及共享endpoint24849；前端与endpoint来源、schema和参数名称明确绑定，后者不冒称完整Writer checkpoint。沿现有Generator/Runtime实际计算，全部ψ/θ/source/MT冻结，0共享/局部更新或optimizer恢复。P0/S_G/D38、MT合并基底、完整38-target/rank8和原官方十步native policy保持。
SOURCE=/data1/user/ymdai/ember_runs/video_guided_proposal_writer_20261012/frozen_context_decoder_fit/refinement_from8883_20261013，端点为complete.json.original36_endpoints逐项引用的terminal_30min/update24849。
新ROOT=/data1/user/ymdai/ember_runs/video_guided_proposal_writer_20261012/terminal24849_student_video_evidence_20261013。

原全部36 task各一次正确G24849真实完整实践，直接使用已保存的原指定teacher_demo LoRA，0正确G再生成、0新MT实践、0正确G/MT/E参照query。原非空MT-H只读复用，保持真实producer/参数/RNG。新状态按既有practice消费者：n=len(MT-H.episodes)=1，state=practice_states[(path_ordinal+n)%pool]，env/native policy root=seed(32,condition.noise_root,n)。launch前practice_mapping_manifest冻结全36实际初态与seed，均不同bootstrap且与原query16分离，不用query补H。

保存实际RGB/proprio/两次flow hidden/执行action/反馈、真实参数ref、完整History和实际终止范围；新G段和组合MT/G前缀明确producer/records/episodes。只有实际完整实践（包括正常horizon失败）的新P/H可进真实学生bank；model_failure/截断/未完成不得伪造完整或资格。原MT/G24849各16 query信用按真实权重/task/state/scene/env/native RNG兼容只读引用，与H分开。共同E保持，只作标签/参照，不入可部署候选或特权条件。明确后继消费者为G对真实non-MT P/H的共同E监督及π真实候选/经历的选择与on-policy学习；本批没有授权它们的共享训练，不整理自身轨迹再微调最终task LoRA。

### 19.2 固定24849的两个教学控制

每arm全36条件各生成唯一完整LoRA并做原query16共576新G-only episode。P0、原MT-H、目标exact L、固定完整推理组合保持；不使用本批新G-H。same-task-other demo=(原correct demo+17)%50，即46–49→13–16，未入Writer原144缓存，只称Writer输入未见video，不称未参与E/meta的独立数据。wrong保留原correct demo ordinal，语言始终目标task。两arm均真实双RGB/stride5重新完整读取与生成，0平均frames/features/LoRAs、0择video、0零图/fake query。

main在任何wrong生成/读出前修正原派发遗漏：原task_video_mapping只遍历四suite，36输入实际仅24输出，遗漏12个LIBERO-90，证据ROOT/analysis/wrong_schedule_actual_boundary.json。当前有限面板按实际(SUITE_ORDER四suite,libero_90)及各suite local task_id升序排36已审计non-held keys，目标i取来源(i+18)%36。大小[6,6,6,6,12]，目标/来源各一次、全部cross-suite；spatial↔long、object↔support前6、goal↔support后6。显式global/local/source/demo映射在video_mapping_manifest.json固定，不能把global ID当local或按性能改映射。它改变旧24下一suite轮换，不与旧wrong成绩作匹配因果比较、不代表全部wrong来源平均效应；旧正式四suite/held调度不变。

两臂所有task/init/scene/env/native policy RNG与原correct24849、MT/E/旧G逐行配对，参照只读兼容引用，兼容失败不得暗中新采参照。两臂无实践/适应，最终各condition仅一套固定完整LoRA。总新增1188环境episode=36实践+1152控制query，72教学条件生成，0正确G/MT/E新query/梯度、0其它controls/held/Test/checkpoint选择。other检验换教学保持，wrong与other共同检验内容敏感性及原condition记忆/L-H任务识别的解释；wrong下降不单独证明操作知识，弱视频效应不自动推翻已有G能力或触发架构更换。这是早期有限证据，不是补齐全部controls才能开完整Writer的新Gate。

### 19.3 执行、投入与交付

36保存端点实践与两个教学/查询队列独立就绪并行，一个总owner以cost-balanced/long-first/persistent worker调度，空卡及时接就绪工作，不锁原world4或静态task绑卡。沿现有saved endpoint/History/query底层，窄消费者承接由实验session独占分支、实际消费者验证、main集成push后clean detached执行；冻结树不热改、不增加平行Writer/fallback。局部工程失败隔离、有效行不重做、不停无依赖队列；实际科学计算/标签/选择/信息墙变更由main裁决，常规工程无额外ACK。

每launch同时live gpu01/02，合计6或8/node6、非干扰；所有新增data1、独立strg01 quota和shared capacity分别核对。旧36 bootstrap约602MiB/旧36学生1.4GiB，大资产原位引用，含新H/cache/临时峰值估<32GiB。已测query slots16/group4为起点，实践可用原定36episode测H/吞吐/峰值并保留有效行，不新增profile episode或铺slots32矩阵。
准备30–60min，GPU/IO20–50min、消费10min、总1–2h为初估；旧纯576 query11.2min/六GPU/.96GPUh只支持两个query臂25–40min量级，72 native生成/新H IO需真实修订。11:00是希望效果/判断目标，非强杀或正结果保证，GPUh准确记账无硬cap。

一个实际owner等待整批退出，启动给一次真实身份/准入/ETA，完成或真实异常一次可靠交main01a12bb0；无心跳、子阶段Queue、自Queue、第二等待者或自动共享训练。全交付包括36真实H/兼容信用/producer清单，correct/other/wrong全576的task/suite/train-support/RGL/churn/重叠/E类别/72与504及完整映射，教学读取/实践/query/steps/失败费用/存储/恢复能力。

实际02:53:42.723启动fixed24849_student_video_01/PID3426524，clean pushed detached来源c60bd494。gpu01:0–5六worker：两卡保存端点实践condition_batch16/slots16，四卡控制group4/slots16/frame-chunk16；独立队列动态承接，首个实践workerexit0后原卡立即转other。普通live cap6/node6/独立quota通过，全部参数冻结。启动时整批消费暂估03:25–04:00，真实native生成/H IO与完整物理吞吐在退出后结算，不据启动瞬时显存/util声称充分batch；原预计与科学范围保持。

### 19.4 有限批完成与授权收口

03:16:24实际exit0，8job均0/无恢复，全部36实践及other/wrong各36生成/576查询齐备；0更新/正确G再生成/参照新episode。correct425/other414/wrong395、MT340/E455；完整配对/全task-suite/得失/72与504和解释见[coordinate分析§19](../analyses/proposal_coordinate_memory_read_20261012.md)。新增1188episode/257842步、2.085421662GPUh，22.688min；GPU已退出。原预计、原阈值及旧停止不改写。
原启动run_contract的零生成/零教学forward及顶层旧query-only字段属于继承的保存正确端点子合同，已在ROOT/analysis/completion_20261013/contract_clarification.json分列当前arm实际范围和澄清出处；原合同不修改，当前other/wrong实际各36真实读取/生成。原保存端点、完整推理组合与本批读取代码身份分别保留。
36实践原件完整审计：25成功/11正常horizon失败，全36bank资格真实，新G1477 records/7686步、组合3039 records/72episodes，实际producer与信用清单保存practice_audit.json。原MT-H/正确G权重/1152信用只读引用，query不进H，other/wrong不使用新G-H。
本节到此完成，没有自动后继共享学习/Actor/fullWriter/held/Test授权。main直接消费完整原件后自主登记后继；新真实P/H给现有G和π提供明确消费者，不构成已学反馈资格，也不要求Owner逐项许可。


## 20. 当前授权：强G24849后的完整共享Writer有限窗口与唯一validation400（2026-10-13）

main已消费§19全36真实G-H及5184逻辑行，授权本完整窗口；§19已结束且不得重复。correct425/other414/wrong395对MT340，other→wrong少19，其中E-only少18；train24为252/240/221对169，support12为173/174/174对171，Long正确→other少13/→wrong少22，wrong局部正例保留。换未入Writer四视频池的同task教学保持大部分能力，内容变化影响部分能力取得，但wrong仍高于MT、跨suite视觉风格与L/V冲突未分离，不能把+85全部归因教学理解。原36真实G实践25成功/11正常horizon失败、query未入H，允许用于本窗口。

### 20.1 机制、单一图与梯度owner

source+合并MT执行基底冻结，同P0/S_G/D38/E、38-target/rank8。完整G由v9完整CP432双读取meta/Reader/static及其它前端与endpoint24849组合，F为166张量/8,761,152参数；只F可训练，前端/source冻结。F承接24849原Adam/cursor/四rank RNG，同BA算子目标/AdamW/LR/clip1/logicalB4/任务等权；不是全ψ exact-resume，不把仅F旧checkpoint称完整Writer。唯一EndpointTrainer须实际保持这一梯度owner，禁止generator.requires_grad_(True)重新开启全前端。
θ从CP432随存且未做local/RL的Actor初始化，自己的Reader/static/三head可训练，θ optimizer及独立baseline fresh；不得引入旧RL9。G每parent只生成一个完整确定提案；joint/deployment的practice保留[parent,newG]两选及真实score。真正单候选消费者的forced/零score一般语义保持。φ可固定缓存，θ学习表示每次fresh；E仅F目标/只读参照，不入条件、候选、H或π信用。query只供θ/baseline信用，不入H/F目标/held梯度；自身动作H仅Writer条件，禁止准备轨迹再微调最终task LoRA。
真实链为m=C(T(L,V),H,{P_producer})，P_new=P0+S_G F((P−P0)/S_G,m)，完整sBA作用于自身hidden及原生动作。原P0训练未教过F数值parent域，真实新P/H补该覆盖；共同E仍允许忽略P/H，不能把覆盖称为反馈利用证实。原两候选condition oracle430仅高于425五条，选择器不能替新提案/泛化。旧RL9候选弱及Context900 held阴性不被当前train收益清零。

### 20.2 两项固定初始学习与组件组装

唯一全36=train24+经审计support12、原panel/allowlist、教学46–49/action-hidden/stride5不变。§19原MT及G24849完整权重/H/producer/1152 MT-G query信用引用，无新refresh/实践/参照query。
A固定5400次logicalB4 F更新（600遍全36），cursor24849→30249。每task只有真实MT/H_MT与G24849/H_MT+G，task等权→实际parent等权→该parent实际H等权→原四video随机采样，不按成功筛parent。约每parent300/每video75为预期，记录实际曝光而非保证收敛。固定φ的精确输入可一次编码复用；原144 MT cache须身份一致，新G/H×4的144真实m必须重新读取编码，使用真实G的p，不以旧MT m替代。此为明确non-MT覆盖，不是第三个原144/P0同状态时块。
B固定90次logicalB4 final warm（10遍全36），只消费同实际MT/G pair及16-state信用，condition内实际prefix均值/task等权，F/φ不更新。原forced保存实践不冒称on-policy，无新增STOP监督；相同参数/观测/L/noise的冻结raw/native绝对响应可复用，θ训练表示fresh。只取90终点；允许一次0新环境的全36期望选择读回，对照uniform382.5/alwaysG425/事后430，不用于选点或放行。
A/B梯度owner独立可真实并行，若串行无明显wall损失也可；禁止第二Writer运行面。两个预登记终点组装同一个初始完整Writer，保留各optimizer/sampler/RNG来源，不拼最佳点；仅初始化/恢复用途，无额外576资格读出。

### 20.3 九个B4实际共同批与完整恢复点

R=5仅本窗口有限展开，不是永久轮数/未来预算输入。共9次logicalB4 θ更新，全36各一条当前ψ/θ的五轮conditional-continue路径。每批严格冻结当前核采路径+独立query16→θ/baseline各一次有效更新→登记所有实际P/H→F900次logicalB4（100遍全36）→下一批重采。joint共8100F，初始加joint共13500，最终Fcursor38349；θ local90后RL9。F每块保持全36等权、实际parent/H分层等权，最后加入者曝光不等如实记录，不只教新四task。
原解析所有d=1..5均值价值、实际STOP递推、c=.001、条件继续parent分布与score梯度保留。parent只有一个非STOP候选时条件继续score为零，但完整STOP直接梯度仍保留；practice正常两选不得forced或清零。当前核新路径只消费一次θ更新，不在新ψ下重放旧path冒称on-policy；baseline不读query/未来H，外部截断不是STOP正标签。首visit引用实际MT bootstrap及兼容576 MT query；每新G完整实践后在独立原16初态query，失败候选保留，0新E/query作H。
固定180轮新完整practice、180新G提案；新G实际被实践数量至多180，重新实践parent也执行本轮新初态/原RNG的完整episode。query只覆盖outputs实际已实践参数，同condition同参数16初态只读一次，新增G query至多2880，3060为新增episode上界而非采满要求。未实践提案无query信用、不入真实parent bank；重复parent的新H在该parent内等权，权重不复制。相同真实候选仅按现有语义复用，不据分数省略。每批θ→bank→900F后保存完整G/θ/F optimizer/θ optimizer/baseline及optimizer/两个sampler/bank版本/全部rank RNG/topology/冻结身份。RL1–8仅恢复，RL9同一时刻F38349与θ为唯一资格点；禁止把不同核训练query拼成完整模型分数。

### 20.4 明确自动接唯一correct K1 strict validation400

RL9后直接执行首轮fixed validation8×50不同task-video conditions=400，video_seed7/canonical video_schedule/跨点state-video映射不变；Test关闭，不加其它checkpoint/controls/多预算/held探针。MT300强参照按实际兼容原行引用，不挑弱参照。
每condition共享Writer冻结，不使用held离线标签或最终评测反馈学习；合法教学+自身实践编译/选择唯一完整38-target LoRA，Writer退出后新初态固定评测。adaptation排除该condition最终初态。外部界限MT bootstrap最多1官方episode+最多5轮实践（可选择旧parent或新G），最多6*(suite horizon+10)步；π可在任意完成节点STOP并按原final分布multinomial采唯一LoRA，非argmax/强制五轮。未来预算不入π/baseline，主动STOP/资源截断/异常/读取/生成/成本均记录。
报告strict400单完整checkpoint的task/suite、RGL/churn/breadth与逐condition最终LoRA来源；原147/MT153只在实际兼容行并列，不称单因素归因。joint按实际采集核/轮解释首G保持、后续改进/重复/退化及π保护/STOP，不为解释新增环境/H消融或拼资格分数。结果提高/降低完整过程跨task取得能力、已见task映射/有限教学泛化或θ保持等竞争解释，不预归因Reader/数据量/训练不足。
本窗口到唯一400整批完成立即停止新增计算并交main，main据原件自主接续；不自动延长/换rank/LR/seed/架构，阴性不当bug，单次小正值不称最终目标完成。相邻稳定性、视频必要性与Test尚未完成如实保留。

### 20.5 工程、资源与一次可靠整批接续

ROOT=/data1/user/ymdai/ember_runs/video_guided_proposal_writer_20261012/full_writer_after24849_20261013。全部新增data1，大模型/数据/旧H/144cache/信用原位引用；按新context、180完整H、9全状态CP、held400 H及临时峰值预算<96GiB，正式前核strg01独立quota/个人用量/shared，不套旧query-only体量。只退休生命周期已核实temporary，原formal/raw/当前依赖保留。
每launch双节点live/cap6或8/node6/非干扰，GPUh准确计账无硬限。rank按完整消费者吞吐选择：原F缓存4rank约.1s/step可参考，joint F-only不套旧native全G吞吐。collect_batch每rank一condition时slots16不是16实践；沿compile_many/query_conditions合批当前同一逻辑B4，保持任务/loss、更新次序及采样流，禁止预采未来会被新ψθ改变的batch。只改变物理执行，不为profile加科学episode/臂；θ表示fresh，完整CP边界迁拓扑显式记录optimizer/RNG，不绕校验/称bitwise。
初估准备45–90min、A/B含IO20–50min、joint1.5–3h、400约2–4h、全批4.5–8h。依据旧query4 joint1.65h含故障/native全梯度，当前新增3060 episode及F约23–30min计算下界；旧400实际2.081h含1133adapt/119新final/281复用，只是有边界参考。11点是希望有用结果而非强杀/正结果保证；超期说明具体原因、更新ETA，不缩水16query/400或重置时钟。
独占codex branch/worktree→真实CPU/小消费者验证及必要结构检查→main集成push→clean detached执行；不热改旧冻结代码，不要求main工程ACK。已定位且科学语义不变的工程问题自主隔离恢复，有效行不重做，真正计算/数据/标签/选择/信息墙变化交main具体裁决。一个owner接A/B→组装→joint→RL9→400；启动一次真实身份/资源/ETA，整批完成/实质异常一次可靠回main，无心跳/子阶段Queue/第二等待者。


### 20.6 主讨论实践说明纠错与工程owner（2026-10-13）

主讨论直接核源码后确认原派发将确定一套提案误读为practice单候选：原说明是科学误读，非实现bug。最新明确保留[parent,newG]真实π两选，训练与validation一致；180是全practice轮数，G query/3060只为上界。原派发消息历史保留，本节及上述当前合同记录修正来源，不能静默强制新G或student_only。
保留单一Generator/Runtime/EndpointTrainer与现有compile/query/formal消费者。新增context缓存模块只持冻结输入函数，组件模块只做受控CP432+24849导入/固定A/B组装，完整checkpoint模块统一保存/恢复独立owner；旧fixed原144诊断CLI退休，历史以原冻结源码/原件保留。run旧fresh864注册不再是canonical默认；同一运行面prepare现在登记§20。必要新增文件分别对应缓存、组件迁移、完整状态三个凝聚职责，不建立并行Writer/版本fallback。
本轮结构自审：三新增模块分别179/294/99行；active源净增约493行，新增约754行，删去旧注册与重复checkpoint owner。`full_writer._students`的34行/complexity28来自同一受控迁移的完整科学谓词核对，作为凝聚的声明式合同保留，不拆成散落的小验证器；它只在准备时消费真实原件，不产生模型选择或额外实验。旧长函数保持局部改动。CPU验证覆盖原36/1152资产与完整166参数Adam承接；实际CUDA读取、完整状态保存与吞吐仍由正式消费者验证，不能以CPU检查声称GPU已运行。

§20物理执行登记：A四rank与B两rank独立并行，总六卡；完整A/B边界组装到四rank后采joint逻辑B4，每rank一条真实condition。选择依据是900F块已有四rank实测，以及query16阶段最多四套slot16推理并发，不能拿slots16称实践batch16。θ/B两rank迁移后的行为RNG rank0/1承接local90，新增rank2/3显式取完整A30249对应rank RNG；G和θ各自完整source RNG/optimizer/sampler均保存，不称bitwise或同拓扑exact resume。正式400六persistent worker，slots16起点；允许首个已登记真实group用condition8检验物理收益，保留所有有效行且0额外episode，其余group4就绪立即执行。后续设置以相同suite/horizon的实测wall、实际batch histogram和峰值决定，无可比收益证据继续已测4，不复测slots32，不等待profile让就绪卡空闲。
CPU实际检查：完整496 G/212 θ状态与166 F Adam24849/独立baseline及optimizer完成一次state save/load；真实task0保存H34条，warm仅构建两个实际final prefix，不写Actor outputs或on-policy decisions。检查是丢弃的CPU状态运输/真实H消费者，无新forward/episode，不验证CUDA RNG或完整Runtime。原检查脚本与JSON随启动归档。
存储测量：旧完整400结果根约31.7GiB，当前36 MT+G H与学生原件约1.16GiB，新完整state约221MiB；结合180实践、400最多6段H、九完整CP、冻结输入cache/临时输出，96GiB峰值预测保留并在每launch核独立quota。大source、原144缓存与既有H/信用仅引用。

在当前约44GiB空闲显存下，冻结native候选响应的observation microbatch登记为32，沿真实现有responses消费者承接每行已保存noise与P；这是当批真实warm/joint/formal中的物理执行验证，无新增episode/训练臂，不重测环境slots32。记录实际响应wall/observation与process peak，尚未声称吞吐提升；已定位OOM可在原范围新job改回已测16，保留失败费用与有效行。环境slots保持16，logicalB4/概率/更新顺序不变。

### 20.7 异步采集的CPU就绪屏障与完整RL1原范围恢复（2026-10-13）

原1bcb2601六卡A/B已完成，RL1全状态/F31149已保存；随后RL2快rank先进入Actor梯度同步，慢rank仍执行合法query，默认600秒NCCL窗口耗尽，原owner05:05:41退出1。原失败/代码/费用保留，RL9/400尚不存在。实际task38到05:05:29仍保存有效query，足以区分“长采集与集体超时混用”与已证实collective顺序错误；不诊断GPU故障。
原科学计算完全保持，仅在collect_batch返回、renderer正常释放后，按本torchrun run-id/port/update隔离CPU FileStore就绪键；等待全部rank完成合法path/query后进入原on_policy_update，等待最多30分钟。不同环境耗时不再占首NCCL同步默认窗口；真实θ/F梯度、B4/任务权重、顺序及采样/RNG合同不变。CPU消费者已复现原故障并通过实际等待函数+原synchronize_gradients，0GPU/模型forward/episode。
从完整RL1恢复同四rank，保留F/θ/baseline各自optimizer/G及θsampler/bank1/四rank RNG；按原kernel恢复第二批已保存path/真实H、只读有效query行，只补其实际缺项，禁止重复40实践或240有效G query。A5400/local90/组装不重做，无中间科学读出或更换终点。新恢复owner替代已经退出的旧owner，所有失败/成功费用统一结算；新代码须独占隔离检查、main集成push、clean detached执行，不热改1bcb冻结树。
完整RL9/F38349→唯一strict validation400自动接续授权仍有效，Test及其它后继关闭；旧08:45–12:15总预测保留历史，恢复后的剩余时间据真实完整RL1 571.8秒和最长合法query修订，11点不是硬截止。详细工程证据仅ROOT/analysis/collective_timeout_recovery_20261013，不新增active design。

§20.7实际恢复登记：6f6a8aa6已push、clean detached执行；05:16:15唯一新owner/PID4112034，05:16:31四rank实际RLjob。live gpu01:0–3/cap6/node6，完整RL1同world4承接，仅消费原当前核缺行。F31149/θ1/bank1→最终F38349/θRL9→400范围保持；原失败/有效产物/费用/启动预测不覆盖。完整评测到来后重新live选所有有益可用slot，不以world4收紧整批并发。按首周期571.8秒和最长query实测，暂估恢复后剩余3.5–7h，旧11点软目标保持；已完成CPU验证不冒称剩余真实更新或400完成。

### 20.8 完成状态：唯一完整RL9与correct K1 validation400（2026-10-13 08:17）

§20科学合同已实际完成，恢复owner exit0；旧原owner exit1保持失败历史。A5400/F30249、local90、九批真实[parent,newG]R5/每批θ→bank→900F、F38349完整v10 RL9及唯一400全部齐备。166 F Adam均38349、全G/θ/baseline及独立optimizer/两个sampler/bank9/四rank RNG真实保存；φ/source冻结，未读取Test或held离线监督。

strict400 Writer165/强MT153，R129/G36/L24/net12，收益集中task16、Long净−3，不称最终明显稳定Val/Test目标取得。共同180实践和77被实践G的1232 query信用是九不同采集核定位，不拼完整CP资格；103未实践提案无信用/bank。formal400 MTbootstrap+1465实践+289新G final共2154新episode，主动STOP150、五轮外部退出250，final111 MT原行兼容引用/289唯一G，adapt与最终初态分离。

全过程wall4.044h/18.857689823GPUh含旧失败，3566新episode/910556步；原启动/ETA/失败原件保留。实际资源/状态/完整分布见progress、既有coordinate分析§20.2–20.4与ROOT/analysis/completion_20261013；不再重复活动合同。完整RL9和恢复资产/正式H/raw保留，当前无运行中的科学批；本合同到此停止新增计算，Git及一次可靠整批交main后由其自主登记后继，无自动相邻点/延长/controls/Test，也不以Owner新许可或工程ACK阻塞main取舍。

## 21. 同一完整RL9的首G配对validation400机制诊断（2026-10-13）

### 21.1 原件、问题与唯一科学范围

§20正式Writer165/MT153、R129/G36/L24，小幅且集中，task16净+10，去掉后145/143。main直接复算800行/400path；最终MT111为47成功、最终G00069为30/MT21、laterG220为88/MT85。这些子集由旧策略形成，不能把其净+9/+3直接称后来修订退化。训练425及共同信用不是RL9的held能力；固定E监督与non-MT P/H覆盖仍允许忽略反馈，旧Context900/旧RL9147负证据保留。

唯一完整v10 checkpoint=full_writer_after24849_20261013/checkpoints/RL/update_00000009，F38349/原完整冻结读取前端/source。以原formal_panel和validation/RL_00000009/correct全部400条件的exact L、50不同video/task、各自实际完整MT bootstrap为输入，arm=first_G_from_same_MT_bootstrap。原348保存G000全部直接引用（含未被实践提案），首attempt parentMT300/episodes1/valid/retained_U；52在首前STOP无G000，只读自己完整1episode前缀生成一次。原最终恰G000的69同scene/RNG行只读复用，其余331新最终episode。实际CPU400出处核对已符合348/52/69/331。

0新MT/G实践、0梯度/Actor决策、0checkpoint选择/其它视频臂/全部候选oracle/相邻/Test。52不用后来H/query反馈/其它condition H，不偷换MT起点；348不为形式重放H/教学。每condition最终唯一完整38-target rank8 LoRA固定、Writer退出，原官方flow/场景/env/native noise前缀严格配对。依赖既有400次MT实践，不称zero-interaction；旧依赖成本与新增成本分开。原正式165/失败/完整状态不改写，本posthoc诊断不提供方法qualification。

### 21.2 输出、解释与停止

firstG↔MT、firstG↔full各400逐行R/G/L/churn/集合重叠、task/suite/覆盖；原STOP52、finalMT111、finalfirstG69、finallaterG220及复用69/新增331预先分列。子集都是原策略选出的上下文，不提升为随机化单部件因果；不能逐初态取三臂最好形成方法分数。
若首G有明显较广收益而full损失，main转向完整适应如何保持利用，不唯一归因θ/STOP/parent/H/laterG；若首G也弱或交换大，重估当前G跨task映射/读取/目标，不靠更多选择器/同缓存续训保护假说；若差小或仍task16集中，保留广泛迁移不足。唯一终点400完整行，不中途按task结果停或增臂，完成即停止新增计算交main自主接续。

### 21.3 唯一消费者、执行与成本

ROOT=原full_writer_after24849_20261013/diagnostics/first_G_validation400_20261013。finite first_g模块只持出处/52直接生成/固定参数读出与配对汇总，使用现有load_version、history_prefix、generate_candidates、官方EpisodeRunner和EvaluationQueue；没有第二Writer/模型fallback。任务结束且main消费后退休诊断入口，冻结代码/raw保留历史。run仅接一个有界命令，旧formal消费者语义保持。

独占codex分支→真实完整CP加载/400出处/69兼容行与队列CPU检查→集成push→clean detached；旧冻结树不热改。源H/348 LoRa/数据/模型只读引用，data1峰值8GiB，独立strg01 quota和相关用量/shared capacity核实，每launch两节点live总6/8/node6。已有279需新query的saved参数队列及时接卡；missing52共享同一owner动态就绪，不阻塞独立ready。persistent long-first/cost-balanced，未绑task/card；本331真实组允许16/32 slots/group有界物理比较，所有有效行保留，0额外episode/profile矩阵，按真实steps/wall/实际推理batch/peak选择，不称配置最大等于充分batch。局部失败隔离，至多一次原范围有界恢复有效缺项，不重跑完成行。

初估准备20–40min，52生成/331query15–40min，共35–80min；依据§19实测1152query+36practice约23min但加载/native长度不确定，实际资源与吞吐后修订。11点软目标，无强杀或正结果保证。一个实际owner，启动一次+整批完成/异常可靠交main一次，无阶段Queue/心跳/第二等待者。

§20 main已完成CPU原件原样归档到旧ROOT/analysis/main_scientific_readout/{full_writer38349_main_scientific_consumption.json,full_writer20_readout_interpretation_20261013.txt,warm90_fixed_pair_scientific_consumption.json}。不重跑main算术；新事前合同、400出处、存储/CPU消费者/命令/身份/费用由新ROOT保留。

### 21.4 实际完成与当前授权状态（2026-10-13 09:06）

唯一owner first_G400_01于09:06:25.632退出0，46306c07 clean pushed detached来源、八job均0；348保存/52 own-MT完整1episode生成/69同G000配对复用/331新final符合合同，400行及每task50不同教学齐备。完整checkpoint仍是原RL9/F38349，0新实践/Actor决策/梯度/Test，原正式full165不改。

firstG165/MT153/full165；firstG对MT R122/G43/L31，对full R143/G22/L22。+12中task16占11，去掉后144/143/350，Long18/24；没有支持广泛强首G主要被完整后段损失的分支，仍需main按完整分布判断初始迁移与保持缺口，不能唯一归因组件。预定义策略形成子集及全部task/suite/集合原件见现有coordinate分析§21.1–21.3、ROOT/analysis/completion_20261013。

wall638.866秒/.931672850GPUh、六卡峰值、失败0；实际batch7.675、peak allocated13.782/reserved32.037GiB。有效32-slot首组与无剩余组的23.754秒continuation开销均结算，不能称已证整轮最优。依赖旧400 MT bootstrap114908步/不可拆GPUh分列不重计，消费前新根143.62MiB/8GiB准入，双节点退出快照无本项目GPU进程。

本合同已完成，停止新增计算，Git和一次可靠整批交main后由其自主登记后继；当前无自动训练/其它诊断/Test。finite first-g入口在main科学消费后退休，冻结源/raw/关键权重继续保留；生命周期清理不是重新执行诊断或新增科学工作。

## 22. 固定表示上的完整端点原生FM校准与固定读出（2026-10-13）

§20、§21均已完成。main消费首G/full同165、MT153且收益集中task16后，明确授权本节；不恢复任何已停止批。完整事前派发原样保存在 `/data1/user/ymdai/ember_runs/video_guided_proposal_writer_20261013/endpoint_native_FM_after38349/scientific_dispatch.md`，实际run_contract/三份fixed_g_panel登记全部输入与整数。本节是当时唯一active后继，现已完整结束（§22.6），当前无执行中的科学批；讨论候选不成为新active design。机制、竞争解释与最近似历史见现有分析§22。

### 22.1 完整源、唯一梯度owner与目标迁移

起点 `/data1/user/ymdai/ember_runs/video_guided_proposal_writer_20261012/full_writer_after24849_20261013/checkpoints/RL/update_00000009`：完整v10 RL9/F38349、bank9、四rank RNG、全G/θ、166个F Adam及独立θ/baseline状态。新schema `ember_full_writer_endpoint_native_FM_v11`、目标 `task_equal_cross_episode_native_FM50x7_v1`。v10只允许受控导入这一明确RL9/F38349来源，不是通用shape兼容或原科学合同exact-resume。
只更新原endpoint166张量/8,761,152参数；双meta/ContextReader/static/source/实际MT合并基底/θ/baseline冻结，图、P0/S_G、38target/rank8不变。承接原F Adam lr1e-4/betas(.9,.95)/eps1e-8/wd1e-4/clip1。完整新checkpoint保存未更新θ及其来源，不能称新版G/θ共同训练。
唯一监督从operator BA替换为普通native FM。合法L/V/实际P/H先形成完整LoRA，随后28个跨episode query加载真实执行RGB/state/action，按实际flow_sample的 `t*noise+(1-t)*a`/`noise-a`、完整50×7速度MSE回传；32维padding、offset1、冻结source normalization及优化seed7不变。每query等权、B4条件各/4。
允许76完整A/B因子叶子累积全部query余切，再对同次F作一次精确VJP；`Runtime.task_parameters`供应实际MT合并基底。查询hidden/action不得返送G条件，E仅历史/只读参照，无BA混合/public/prox/额外动作项。

### 22.2 固定bank与训练曝光

原真实bank9只读：149 actual parents、252 P/H前缀、1008合法四视频context。按task→actual parent→该parent真实H各层等权；四教学46..49按原sampler，绝对F cursor38349延续。完整缓存必须身份相同，不补新的教学/native前端读取、不吸入新学生/实践/query。
QueryStream显式pool0..45，排除全部四教学而非仅当前video；event ordinal20261013、每task独立visit0..47、每visit28 query，与绝对F cursor分开保存。432 B4更新/1728条件/48,384 query，36 task各1344 query，F终点38781。38565（新增216）只恢复；禁止据loss缩短、延长或选点，无新实践、bank更新或θ/reward学习。

### 22.3 已冻结的三项控制读出

1. 父F38349原36指定教学/原MT-H/P0，各query16，共576新增最终episode，可与训练并行；旧F24849=425不能代替当前父分数。
2. 唯一末F38781同seen576，原MT340/E455/历史24849及本父点只读配对。
3. 唯一末F38781从原held400各自完整MT bootstrap一episode前缀、P0、exact L和原50不同教学/task生成首G；400新生成/400新final，无Actor或后来H，原firstG165/full165/MT153只读参照。

总新增1552 environment episode、472完整端点；0新MT/G practice、0held离线标签/梯度/选点/Test。最终单套LoRA冻结、Writer退出。旧bootstrap及完整bank的交互费用作为依赖保留，不能称zero-interaction。全部query禁止进H、训练或中途干预。
报告全部task/suite、train24/support12、R/G/L/churn/sets/coverage、held去task16与MT保持。seen与held同时有清楚合理收益才加强固定表示下功能校准的支持；seen改善而held弱/集中则进一步反对输出修正作为主要迁移解法。正常阴性不是bug。旧full165不改，也不冒称已完成新版多轮Writer/θ学习。
所有分支在432和三读出完整后停止计算交main；无第二同表示块、BA-FM混合、超参小扫、相邻/controls/Test自动授权。停止线约束固定phi仅校准F即可广泛迁移这一假设。

### 22.4 执行、成本与完成接续

ROOT=`/data1/user/ymdai/ember_runs/video_guided_proposal_writer_20261013/endpoint_native_FM_after38349`；全data1、源资产/H/cache仅引用，新增峰值32GiB。准备核strg01独立quota used1530127520KiB/soft2147483648KiB及shared81TiB，约589GiB余量；正式每launch再现场核两节点/cap6或8/node6/非干扰。初始现场gpu01八卡空闲，普通cap6；优先同node四训练卡+两父seen卡，末点ready后动态long-first/persistent使用实际空余卡，world4不是整批上限。
最多两个实际初始B4可丢弃micro14/28 profile，完整源/F Adam/cursor/rank RNG恢复，按实测完整B4最大rank wall和显存/finite选择；不新增episode/科学曝光。最终query复用slots16/32经验，在真实请求选择物理batch，不把曾无严格配对的32当最优。
工程30–60min、训练25–60min、读出20–35min/重叠总1.25–2.5h均为main待实测估计；按首真实update修订，11点软目标非强杀/正结果保证。必要接口/调度修复独占分支→针对消费者检查→main集成push→clean detached；冻结树不热改。有限first-g入口已消费，由一个fixed-g原生Runtime读出面替代，不留旧新fallback。
一实际owner接profile/训练、独立父seen、末seen/held及汇总；实际启动一次/整批完成或真实异常一次可靠交main01a12bb0，无心跳/阶段Queue/第二等待者/例行工程ACK。

seen两点严格复用原MT-H/P0/指定教学的完整冻结m，经同一Generator.generate物化新F端点，避免重读同一φ输入；缓存路径、frontend/P0/SG/MTbase/source、demo及MT完整前缀均核验。held400仍只使用各自真实MT前缀与原教学，生成新端点；不借用seen m或旧G结果。

### 22.5 准入退出及不改科学语义的共驻恢复（10:02）

原native_FM38349_fixed_readouts_01在09:56:01的CPU选择阶段exit1，0GPU job/费用/episode；准备时空闲卡已被他人任务使用，idle-only/40GiB的未实测准入并不代表本新F-only消费者必需。原快照/exit/交付保留，completion consumed指向唯一recovery_02。
恢复owner PID1100235使用同一clean pushed frozen47edd274。原消费者实际17.82GiB allocated/18.64GiB reserved只是准入参考，新native目标仍只通过已登记两实际B4 profile14/28量测，不称旧peak证明新目标安全。现场低util及24GiB以上free可按项目共驻授权使用；完整源/四rank RNG/B4/dataflow不变，别人进程不修改。actual profile gpu01:4,5,6,1，父seen gpu02:6/4，共6卡；每job再次双节点/quota/实际余量准入。profile正常结束后恢复起点再运行432更新，其GPUh/所有失败单列，正式学习与科学阴性按原合同处理。
精确原件 ROOT/coordinator/native_FM38349_fixed_readouts_recovery_02/{launch.json,physical_admission_decision.json}、ROOT/jobs各actual launch。ETA待profile最大rank实际wall修订，11点不变软目标；此资源恢复不新增梯度/episode/科学臂或第二调度者。

两真实B4 profile14/28均正常完成；micro28最大rank2.873867秒对14的5.544698秒，峰值allocated17.003966/reserved18.142578GiB，完整源/F Adam/RNG恢复38349。正式432更新按28实际启动，暂估训练23.7min及后读出20–35min，source47edd274冻结不改。Owner随后取消11:00希望时点，不缩减/中断当前432/三面板合同；原预测与软目标记录只作历史。
14先测含首批query/IO/kernel准备，28后测；上述整体wall用于本批物理选择，不称严格稳态配对1.93倍因果提速或绝对最优。28已容纳每condition完整logical query batch，峰值与实际consumer记录保留，不为继续放大扩大逻辑batch/矩阵。

### 22.6 实际终点、缺项恢复与科学停止（11:28）

正式432更新exit0，唯一完整v11 F38781/166 F Adam/四rank RNG/bank9已经实际核验，48,384 query及每task1344均齐全。两个discarded B4/224 query单列，38565未读出或选点。父seen576、末seen576、末held400全部new-only，472唯一完整端点，0新实践/Actor/held梯度/Test，旧正式full165保持。
recovery_02原exit1来自两个renderer初始化error0x8cdd，保留1518有效行/470生成；不能确诊唯一OOM或GPU故障。CPU03错误把seen布局16当32，被既有冻结队列校验拒绝，0GPU job；04在同冻结47edd274下按实际16/32布局仅补task34的16和heldtask31的18，共34新final/2生成，于11:28:50.675退出0。没有恢复训练/父seen、没有改动旧有效行或冻结源码。全部失败与有效工作计入6.753025254GPUh，GPU owner至完整wall86.762min；11:36双节点本项目GPU进程0。

父seen427→末431但target24 249→247、support12 178→184；E-only134传递94→90、共同成功321保持309→299。末held158/MT153、旧firstG165→158（33得40失），去task16仅145/143，成功覆盖仍6/8。固定φ仅校准F未解决主要迁移问题，当前投入结束，无第二同表示块/混合/小扫/其它点；不推出所有FM/Reader/视频编译不可能。全部task/suite/能力得失及物理/费用依赖见ROOT/analysis/completion_20261013和现有分析§22.1–22.5。
Owner取消原11:00目标，最新希望今晚新加坡23:00有进展、科学判断先不考虑时间；无新强杀或缩水协议。main与Owner现在讨论下一步，尚无解冻Reader、扩任务、选择器训练或Test派发。候选不进入本设计的执行合同，本节完成交付后仅保留恢复资产与历史，不自动新计算。

## 23. 固定原G38349的选择学习迁移与唯一完整400（2026-10-13）

本节是当前唯一active科学合同。Owner11:50原话“我先去吃饭，你分析完后自主推进，我吃完饭再找你讨论”，main据已完成分析明确授权整链；§22已结束，原讨论未派发仅为旧时点。完整事前派发和补充原文及冻结选择JSON保存在/data1/user/ymdai/ember_runs/video_guided_proposal_writer_20261013/selector_transfer72_fixedG38349/analysis/main_contract，本节没有另开平行design。当前工程准备，未launch。

### 23.1 主要问题、固定来源与梯度owner

旧G38349+θRL9完整165/MT153、36得24失，旧warm36 G普遍更强且后续只有9个B4 on-policy，θ缺少真实G迁移情况下的比较曝光。§22 nativeFM target24未净增/held158，不用38781替原较强G。检验固定已有G能力在G未见任务上能否被θ学会使用，不预设θ全部根因或追事后189。相对旧joint同时改变分布/曝光/冻结组织，不称同预算单变量因果。
唯一源v10 /data1/user/ymdai/ember_runs/video_guided_proposal_writer_20261012/full_writer_after24849_20261013/checkpoints/RL/update_00000009：完整G38349、θ及Adam、baseline/Adam、samplers与四rank RNG承接。新schema ember_selector_transfer72_fixedG38349_v12为显式任务/组织迁移，非旧合同exact-resume。全部G/source/MT/尺度冻结，F Adam/cursor38349、bank9只读，不append学生bank/teacher/F缓存训练；θ不fresh/reset/拼接。warm baseline冻结，on-policy才更新。新CP保存同一时刻全G+θ/baseline、各optimizer、源cursor与当前阶段/任务调度/RNG/物理拓扑；迁移可承接2/4等实际物理rank，记录未分配源rank状态，不绕校验或称bitwise。
Actor三head、解析STOP/所有截断深度平均value/conditional-continue score、.001代价、单确定G提案、rank8/38target、原生flow和最终multinomial不变。无价值head/MT bias/阈值/熵/CE或Q→LoRa动作梯度；无G解冻/轨迹准备后拟合最终LoRa。

### 23.2 72任务、run-scoped权限与等权调度

保留原G seen36，加main冻结的source71中G-novel36；显式名单见原selection JSON：40,41,52,53,54,57,59,61,63,66,68,69,72,74,78,80,81,82,99,102,103,104,105,107,108,111,112,114,115,116,119,121,122,124,126,127。global=40+source90 local。source71扣原support12剩59，按KITCHEN29取18/LIVING_ROOM14取9/STUDY16取9及SeedSequence([20261013,23,stratum_index])事前不放回；无按结果换名单。新36只33 exact语言，69与原seen42同语言，source policy已见/G未训，不称36新操作。
原全90 specification/19排除、真实Bddl/data/init metadata和原36身份审计；完整run-scoped authority在analysis/task_audit/task_authorities.json，只供train72，global52/fixed24/8/8不改，validation/test仍canonical固定authority。新增36仅θ/baseline non-held meta梯度，H为自身真实经历，离线教学只exact L/有序action-hidden双RGB/stride5；任务/路径/场景规范不入模型。
每B4 seen2+novel2、层内36等权，独立seeded permutation每18更新全72一次。实际实现SeedSequence([20261013,23,0或1,epoch])，与GPU/任务难度/已知结果无关；A10周期、B1周期。source90 horizon400，其他官方协议不变。

### 23.3 固定MT→首G pair数据与final学习

seen36复用§22 parent seenF38349的正确36保存LoRa、兼容576G+576MT query及原MT一episode H，只新增每task保存G一次完整实践，不能用24849参数/G-H替代38349。novel36各真实MT bootstrap→第一合法四视频46..49教学→一次完整G→一次G实践→MT/G各独立query16；无E/offline动作标签。所有72 init_order沿seed(1,task)，query为[36:46]+[14:16]+[32:36]，practice前14，MT0/G1；具体seed/source/producers在run manifest固定，seen原query/MT bootstrap不变。
G实践forced仅供final数据、无on-policy行为score。query不入H/G/teacher条件，只θ信用；H保留真实RGB/proprio/two-native-hidden/action/feedback与producer。最大1260新增episode/36生成，seen1152 query/旧权重只读；不为profile重复有效行或按成功省略。
A180 B4/720条件/每task10次，原final expected-query及真实prefix condition内均值保留，θ/Adam承接、G/baseline冻结；90只恢复/180末点。共享Reader/static表示获梯度，不能称三项决策已校准。实际已有forward记录query credit/pMT与gradient，取消旧36 hardcoded425/430读回；不额外GPU读回或挑出口。

### 23.4 B18当前核完整选择学习

由A180完整状态接18次B4，每task一条新condition，四视频下一条（ordinal1），自己的真实MT bootstrap前缀开始。实际五轮conditional-continue，parent/STOP、[parent,newG] practice与final均保留；最多360提案/360真实实践，尝试G数不保证360。未实践G无query或credit，重复parent仍采新初态完整实践；不补强制G。
每当前固定G/θ批完整路径+独立query→θ/baseline各一次→新θ重新采下一批。原所有截断深度平均value/解析STOP/final/行为score和.001代价不改，无未来预算输入，无旧路径反复on-policy/未来batch预采。query visit0的原16common-noise，path ordinal1与阶段身份区分；MT兼容引用优先，不同noise不冒称同一行。
G不更新/bank9不追加；CP1–17仅恢复，完整B18唯一出口，未按内部loss选点/延长。最大360practice+5760Gquery=6120新episode，MT不兼容可另1152必须计费；实际重复parent会减少query，不凑上界。

### 23.5 唯一完整correct validation400与停止

B18自动接同一完整G38349+新θ的K1 fixed validation8×50=400，video_seed7/50不同教学task、原scene/state/env/native RNG和MT153/full165参照不变。完整Compiler最多1真实MT bootstrap+5实际practice，可任何已完成节点STOP，final multinomial选唯一完整LoRa；最终新初态Writer退出/不继续适应，0held离线标签或梯度/Test。
旧400 own MT完整first prefix允许精确兼容引用（原114908步及费用作为依赖），不引用后来G-H/旧Actor路径/G最终结果；若不兼容最多补MT400并计费。新G final实际执行，MT final兼容引用，不离线组合旧400冒充新方法。
报告task/suite/coverage/RGL/churn/Jaccard、MT回退/候选能力与选择信用、STOP/实际轮数/成本，以及train72 seen/novel/语言重复分列。训练不同核query不拼资格；更多MT回退/回153/仍集中165或内部值涨均不够。一次400不证明相邻稳定/视频必要性/Test。阳性后main另登记必要证据，阴性降低当前仅改选择学习可迁移利用已有G的优先级，不自动第二同数据块/Q补丁，仍继续研究。

### 23.6 工程、资源、预期与可靠交付

一套Runtime/Generator/Actor/compile/query/shared checkpoint，旧§22 fixed-g及FM训练入口按已消费生命周期退休，历史冻结/tree/CP/raw/失败保存。独占分支隔离→真实消费者检查→main集成push→clean detached正式运行。metadata audit/实际完整Runtime与optimizer CPU加载0forward/episode已完成；真实GPU执行尚待launch，不称模型已训。
全新增data1，峰值160GiB计入72pair/360practiceH、恢复CP/400H/LoRa/临时cache；独立strg01 quota/个人用量/shared现场核、大资产原位引用。每launch双节点live、总6或8/node6/非干扰/GPUh无hardcap，动态数据/ready query/formal不锁world4；logicalB4及权重不变，physical rank/group/slots/response micro按真实吞吐/显存选择。有余量在既有请求验证扩大，不追加科学episode或重做有效行。
事前工程45–90min、pair30–60min、warm20–45min、on-policy2–3h、4001.5–2.5h，总5–8h（旧warm90 rank0 1225s及query16/400依据），实测/共享负载修订。原今晚新加坡23点希望已被Owner“我回来了，晚上十一点的截止时间不用管了，又有多的空余时间推进了”取消，11点也取消；无deadline/强杀/缩水或自动加时扩科学矩阵。
一真实owner接data→A180→B18→400→完整消费/Git，launch一次真实身份/资源/ETA、整批完成或实质异常一次可靠交main；正常持续退出等待，无心跳/阶段Queue/第二等待者/例行ACK，有效行/所有失败与费用保留。工程语义不变可自行恢复缺项，科学边界具体交main。

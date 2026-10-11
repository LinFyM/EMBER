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

正式评测的执行责任复用既有pi05_eval_queue（long-first、atomic claim及raw-row completion），每物理worker保留同一Runtime/共享checkpoint，compile_conditions与EpisodeRunner合批合法独立condition。queue面板/scene/RNG与最终初态排除不随物理卡数/condition chunk变化；MT选中结果引用既有逐行参照，查询仍不进H。实际batch按完整消费者吞吐/峰值选择，尚未测得的收益不声称已实现；没有第二套独立适应runner。

## 8. 保留的实际计算与历史边界

当前canonical实现为 `src/ember/proposal_writer/`。双meta各18层q/v rank32/alpha32/dropout0，seed202610101/102；教学读取原source双RGB/exact L、stride5、50×32固定probe1729/time1/state-free。Gemma finalnorm contextual前512图像位、raw frozen合法语言embedding及action finalnorm H均有G消费者，source冻结；实际任务执行关闭meta并用MT合并基底+唯一38/r8。
G保留完整ξ/time、64数值块、block/rank/target通信、6层width256/FF1024与16Euler；meta及全部G编码/归一化/速度网络归ψ，π独立下游Reader/trunk/heads归θ。π采集及对应θ更新冻结全部ψ，跨批由登记CFM块更新ψ；不用θ可训练embedding改变G核。源raw量可有界缓存，变化meta输出跨更新失效。
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

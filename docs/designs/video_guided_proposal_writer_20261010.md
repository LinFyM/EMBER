# 36任务共同教师、评估经历复用与不同预算编译：main执行合同

Owner 2026-10-11约14:30北京时间明确要求立即自主推进；本稿由main定科学内容，实验session在独占工程窗口整合进唯一active设计/状态文件后，用clean pushed detached实施。旧四task G480仅自然收口，不恢复旧π矩阵。新批不再以四task试训筛方法。

## 1. 目标、范围与时间

- 数据/G/π覆盖既有审计train24+support12全部36任务，按task等权；固定Val8/Test8离线标签不读取，不新增任务或分割。Task36名单复用现有ALLOWLIST，libero_90支持任务official horizon400必须进入环境合同。
- 今天数据准备原目标4–5小时、最迟约5–6小时；实际基准从本次约14:30授权计，原训练数据目标北京时间20:30前。Owner随后明确质疑4/8初态精度，main消费gpu01直接实测后将选点/独立报告各扩大到16。当前36任务E/H训练数据预计20:40–21:05齐备，比原目标晚10–35分钟；这是明确披露的精度/时间取舍，不把时钟重置为formal启动，也不承诺已按原期限完成。独立教师报告在E锁定后与G并行，完成前不宣称已验证教师泛化。工程、清理、profile及数据等待均计入wall。
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
- E锁定后另取16个预留初态读E与MT，共1152 episodes，作为教师能力报告；不用于挑E或本批G/π/H，teacher报告阴性不重标成bug。训练bank与独立报告合计4032次真实episode，角色分开、费用完整记录。报告在E锁定后就绪即运行，可与首段G并行；bank齐备不等于报告已经完成，不因报告未完成静默改变E或共享训练数据。
- 评估参数引用必须真实且唯一，例task/checkpoint/base/rank的稳定ref；禁止跨事件复用parent/160/480导致History.states覆盖。原始文件名/ID只做索引，不进入学习特征。无需为G输入保存sim snapshot；只保存已有必要RGB/proprio/action/hidden/feedback/noise/结束原因/产生者，source raw-image embedding可以运行时重算/有界缓存。
- 出现晚期退化，后段P→E天然提供回退标签；若仍单调增长或持平，如实记录，没有必要也不允许人为弄坏模型制造过拟合。达到最大节点仍上升的任务不能称已充分收敛；可在同一条轨迹、当日时间余量内安排已登记的一段延长来判平台，不开LR/rank/seed扫描。

## 4. G数据、覆盖与固定尺度

每个样本为(L_t,V_tj,P_ti,H_tik ; E_t−P_ti)。同task所有parent/H共享E；P=E的零编辑也进入数据。H来自该P的真实节点读出，非空。多producer经历仅由真实episode及其实际producer组成，并准确标为监督重放，不冒称on-policy采集。首先用single-episode真实H，不为数据量合成假动作或假hidden。

全部36任务都到齐后才固定S_G并启动共享G；可用较少video/H样本训练，但task权重和覆盖不缩小。Owner最新明确按任务、再按实际parent等权采样：MT与各教师checkpoint平等，含E本身，不额外提高MT权重；每个parent内部均匀抽真实H，不能因其H数量不同改变parent权重。早期可用与更多预算的收益主要落实到下节π的训练，不用G采样偏置替代。S_G按同一task/parent等权分布对E−P的每因子RMS计算，floor沿既定1e-3 S_prox；不按H条数、audit或held调尺度。其后数据刷新保持S_G与坐标身份。K=1，教学RGB可使用本批登记的全部合法训练task视频；曾作内部留出的episode一旦进入G即不再称其G未见条件。

G从fresh meta/G初始化学习新目标（旧G240/480不是此合同exact resume）。按预排task-balanced轮次取样，样本量逐步增加，checkpoint保存全部optimizer/sampler/RNG/topology。输出target为相对当前原始因子的差，目标标签/teacher性能不输入条件。

首个有信息的G节点后覆盖全部36任务采实际生成parent及其实际H，直接以同一E−P_student补监督；不重新优化E/不重新准备教师。每task至少一次有效实际G候选实践，采集行为不依赖final report；可复用已有真实MT bootstrap H，不重复采MT，只计准确唯一环境成本并保留每condition的逻辑预算消耗。补训仍按task等权、task内全部已登记parent等权，学生起点与MT/教师节点平等，不预设MT/教师/学生类别比例；保持S_G。新增数据角色/版本/更新窗口冻结后复用，旧真实样本保留来源。不把teacher轨迹parent等同真实G分布，不把此监督采集写作π on-policy。

## 5. π的不同预算学习与中途可用性

目标是早期已有可用输出，给更多实际预算时能利用追加实践改善并保持既有能力；不宣称已证明有限网络单调收敛。

采用budget-conditioned策略，不用1/2/3/4固定轮数菜单。每条non-held训练condition事前独立抽实际环境步预算B，并把总量/剩余量作为输入；训练分布覆盖短、长预算。拟定B在[2(H_t+10),4(H_t+10)]的整数步区间均匀取样，其中H_t为official horizon，包含MT bootstrap；32提案尝试/每次至多4/16 Euler作为本批计算边界保留，不成为固定实践轮数。成功可提前结束episode，后续实际轮数由π与剩余预算决定。四维budget语义明确为total_env/(H_t+10)、remaining_env/(H_t+10)、remaining_proposals/32、completed_episodes/33；不要只给初始均为1的remaining/total而省略总预算，schema同时更新。

阶段A：冻结完整ψ（G、两组meta及全部G读取/归一化）。π的独立Reader/static/trunk和final head使用真实候选/历史prefix的独立新初态回报学习，parent/STOP与practice头不把这个warmstart冒称已经获得完整决策信用。对每个合法prefix，令P为截至此时实际实践过的候选集合、r_q(λ)为候选在独立query面板上的平均success，使用L_final=-sum_{λ∈P} π_final(λ|L,V,H,P,B) r_q(λ)，按task等权平均。覆盖浅H和较深H，始终保留原样MT及已实践候选。可先复用教师checkpoint读出的两折监督作离线warmstart；随后必须加入当前G的真实候选后果，不能把教师候选排序称当前G全流程能力。

教师离线warmstart的prefix必须预算可行：先取一段实际MT H，再取已登记候选各自的一段真实H，按预先固定的、与回报无关的顺序组成逐步增长的候选/经历集合；每个prefix按这些episode实际environment_steps累计成本。H折中未进入该prefix的其它episode不读，标签仍来自另一折；不可把所有checkpoint的整折H全部塞给一个声称短预算的样本。单个prefix只列实际已实践候选，MT始终保留。旧轨迹重放属于离线final监督，不能称当前π生成的on-policy行为；完整路径RL另用当前固定θ/G真正采集。早期/晚期前缀的选择训练共用同一个θ，不为不同预算训练独立π或选择checkpoint。

同一固定候选λ在相同query初态/env-policy RNG上的最终闭环不依赖prefix H。故同一路径的每个候选只评估一次固定query面板，并把成绩复用于所有含该候选的prefix；不在每个prefix重新评估旧候选或另换noise重复付费。查询只产生训练标签，不进该样本H。θ的所有编码仍随更新重新计算；这种query复用不等于缓存θ的表示或把未实践候选放进final集合。

阶段B：同一固定G/meta下，训练完整parent/STOP、practice、final的路径score，目标J=E_{task,B,path}[R_query(λ_final)]，task等权、预算先抽、query独立且不进H；完整condition累加所有真实决策score，batch平均，不除可变决策数。一批θ冻结采样→一次actor更新→重采。R_query初步用4个实际新初态的平均success，MT控制严格配对；跨更新均衡轮换登记query池，完整路径实践初态必须排除本condition的全部query初态。4个随机query在持续重采的梯度估计中与4行一次性选E的身份不同，不能用前者为后者的排序可靠性辩护。训练/适应同一实际采样分布，不最后偷换argmax。baseline独立、当批前冻结，θ不向G提供trainable输入。

在每task固定50初态内，teacher独立报告初态继续不进G/π/H。合法teacher-selection初态本来属于训练资料，可同时作为π实践池来源；不必因其先前用于教师selection而永久排除，避免扩大教师面板后把π实践机械压到仅4个初态。π训练query统一登记为原init_order[36:46]这10个初态（含local与RL来源），每condition均衡取4个，实践池排除全部10个及teacher独立报告；init_order[46:50]仍作有限train诊断报告，不作为正式选点。这个池划分须在launch前冻结；同condition query不进入H，不把已用于训练的初态重称未见。

B必须在on-policy路径开始前抽取；既有长预算路径不能改填短预算输入、简单截断后冒称短预算下的on-policy样本，因为π的较早决策也依赖总预算。teacher重放与prefix监督明确为离线数据，完整路径信用只来自当批实际θ/G与实际B。若G更新，旧路径只可保留真实数据身份，不作为新G核下的on-policy样本。

这使短预算的最终输出也直接得到训练信用，prefix final监督学习保留较好的已试参数；更长预算允许选回旧参数/STOP，所以不强迫最后新生成者覆盖旧成功。此结构仅让保持策略可表示，不保证学到；不能用max/union替代实际选择。

同初态/视频/RNG下分别报告较短和较长实际预算的success-set retained/gained/lost及任务分布；额外报告长路径中实际发生的各prefix能力，不把观察到的轮次设为运行上限。增加预算若只增加成本或引入净损失，须据真实候选供给、选择与生成覆盖判断，不继续给同弱假说加轮数。

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

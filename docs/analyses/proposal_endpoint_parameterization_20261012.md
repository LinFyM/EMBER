# 共同端点学习：取消独立参数噪声输运的有界接续

2026-10-12，主讨论科学决定。Owner授权在硬边界内自主推进。旧v5和normalized-flow有限批均已完成；
本文保存主讨论完整推导；唯一执行合同为既有active design §12，不另建并行active design。

## 1. 结果更新了什么判断

normalized-flow新批MT/G/E为340/339/455（576个配对train/non-held条件），G相对MT R304/G35/L36。
广播约0.6、末端局部差异约0.49，说明旧巨量广播问题已真实缓解；但控制没有净增益。
task13的E=P=MT仍未学会消去xi；task17在真实t=.5样本约3% loss改善保留，但不是完整输运。
因此不再把巨量广播当作当前充分解释，不追加本版Actor/联合训练或时间/scale/LR小扫。

主讨论直接读取模型/训练/诊断源码、原统计、保存LoRA及噪声，另完成全36 CPU算术：
原件 `/data1/user/ymdai/ember_runs/video_guided_proposal_writer_20261012/normalized_flow_query16/analysis/main_scientific_readout/normalized_endpoint_arithmetic.py` 及同名json。平均G与xi余弦0.986249；目标MSE为2.038007，MT为1.000000。
纯算术上扣掉已知xi后的累计field，目标MSE仍1.015798，RMS0.169761；不是一套被噪声遮住的好编辑。
该累计field没有被物化部署或闭环评测，不从这项参数算术推断其闭环成绩。
公共教师均值的参数MSE为0.813884、均值能量0.186116；它只作诊断，不部署均值adapter/teacher dictionary。

剩余可能包括时间与局部坐标的利用、条件到编辑的映射、有限训练组织；本次没有唯一识别根因。
选择新参数化的理由是当前共同E监督已把所需目标变为确定端点，同时两轮实际学习仍主要保留独立xi。
这让继续为随机输运补模块缺少正面依据。取消这项可解析的副任务，直接检验现有合法条件能否学到有用端点。

## 2. 同一目标与不同学习问题的数学关系

P0为冻结source+已合并强MT基底上的初始38-target/rank8任务LoRA（A0非零、B0=0）。
P为当前真实完整任务LoRA；E为授权non-held任务的固定共同教师；S_G与现有坐标/valid mask不变。
记p=(P-P0)/S_G，e=(E-P0)/S_G，d=e-p。原CFM在x_t=(1-t)xi+t*d上学习v*=d-xi。

若条件端点网络输出e_hat(V,L,H,P)，不读取xi/x_t/t，则可形式上构造

    v_hat = (e_hat - p - x_t) / (1-t)
    (1-t)^2 * ||v_hat - (d-xi)||^2 = ||e_hat - e||^2

等式只对t<1的数学分析使用。新实现直接优化右侧，不计算分母、不采xi/t、不执行参数Euler。
没有把1/(1-t)的奇异损失留给optimizer，也不把含teacher目标的x_t交给预测器形成复制捷径。
给定本批共同E，准确端点对应原CFM相同的理想参数目标；这不保证有限网络可学到、闭环等价或held迁移。
新损失是端点MSE，不再声称原未加权CFM目标、同一数值loss或bitwise轨迹；两阶段的raw loss不能直接比较。

实际输出为

    P_new = P0 + S_G * e_hat
          = P  + S_G * (e_hat - p)

已知的当前参数抵消由代数完成，不要求网络再学习减P。模型仍可读取P及真实H来理解当前实践，
但共同E本身不要求每种H产生不同终点；不得将形式读取P/H称为反馈利用已证实。
部署条件没有E、teacher动作/state/reward或任何held离线标签；P_new由共享网络一次forward生成，
没有新condition的动作轨迹准备、BC/FM微调、教师检索、参数搜索或平均多套video LoRA。

## 3. 唯一生成运行面与实际算子

保留当前真实exact L、action-hidden保序V、stride5、非空自身MT bootstrap H、当前P及H的实际producer。
双rank32读取meta、ContextReader有序video/history、静态参数摘要及所有现有合法事实输入不改。
原生Z/H来自现有真实native读取，不增加zero-image/fake-action forward或新特征源。

唯一G decoder输入为每64数值块的[p_block, valid_block]，Linear128→256；加现有target/rank/side/block地址。
保留六层发送端归一化的block→rank→target→rank→block通信和对同一真实context memory的attention，
保留末LayerNorm、Linear256→64、valid mask及完整643584坐标。输出语义改为e_hat。
彻底移除G decoder中的x/t输入、参数噪声采样和16步参数ODE；视频/经历本身的位置编码照常保留。
这里只移除参数生成的xi；原生policy十步action flow、教学native读取的既定随机域、自身H和query的policy noise/RNG全部保留。
static两层、Actor三个head及其独立Reader的实际算子保持，不能顺手归一化或重构Actor。

输出Linear的weight/bias为零初始化，保证fresh G输出正是P0的MT行为；其它共享模块按原fresh规则/seed初始化。
这是本次“MT为初值的确定端点参数化”的明确组成部分；不是保持旧随机head后只改一个loss，也不作单因素归因。
零head仅首步阻断上游梯度，后续由真实监督打开；不预加静态teacher均值、teacher basis、identity以外的先验adapter。

G/全部meta/Actor/optimizers均fresh，新schema与语义必须拒绝旧v6 CP；不迁移旧216权重。
沿现有module/Runtime/path/learning消费者修改，复用通信与布局owner；不新增并行Writer、旧flow fallback或第二runner。
旧v5/v6科学历史由Git、旧formal frozen和原件保存，旧数据只用其冻结消费者解释。

确定性意味着同一P/H/V/L调用得到同一提案，四个noise提案将是重复计算。新协议每个决策只生成一个完整新候选；
不能为保持旧4候选接口反复计算/复制相同参数。对应proposal元数据登记deterministic endpoint，不伪造有用noise seed。
后续H或父P改变可再次生成，parent/STOP/practice/final仍由既有Actor处理实际集合；不把一次提案写成固定适应轮数。
当前首学生阶段仍只需一次首次生成与一次真实实践，Actor并不学习或参与本次候选筛选。
同输入确定性的非finite输出不得重复4次伪装采样恢复；按真实失败处理。

## 4. 共享训练与数据边界

本批仍为既有train24+经审计non-held90 support12共36 tasks；目标E、P0、S_G、原MT H均只读复用。
不扩大Teacher训练，不增加held或动作标签，不重估scale，不引入endpoint标签到condition或task-ID route。
每任务既定24次visit，共216次optimizer更新、logical B4、864个训练条件；每9更新覆盖完整36任务。
原task/parent/H/video采样及seed规则保持，初始仅P=MT/实际MT H。移除xi/t随机变量不改视频选择序列。
原AdamW lr1e-4、betas(.9,.95)、eps1e-8、wd1e-4、clip1，原数值精度，全部不扫描。

    loss = mean_valid_coordinates (e_hat - (E-P0)/S_G)^2
    每个逻辑task条件权重1/4，rank同步及任务等权仍按现有合同。

标签反传给G decoder、Reader、static及双meta；source policy和Actor无训练梯度。
将当前唯一G更新owner替换为端点学习，未来联合阶段亦使用同一owner/共同E监督；不要留下旧CFM更新在joint路径悄悄执行。
本批不启动未来联合阶段。checkpoint保存新语义、完整共享模型/optimizer/sampler/RNG/topology与版本216。
训练指标明确叫endpoint loss，直接报告初始MT基线1、公共均值诊断0.813884及实际学习值；不复用CFM标签名冒称改善。
不增加cosine、log-scale、功能loss、蒸馏权重或读写新辅助头。

## 5. 固定全36配对读出与复用

新ROOT建议 `/data1/user/ymdai/ember_runs/video_guided_proposal_writer_20261012/deterministic_endpoint_query16`。
同一训练出口216之后，全部36沿normalized-flow保存的首学生teacher_demo、实际MT bootstrap H/P生成唯一G，
各进行一次同原practice初态/RNG的真实G实践。教学/条件不按结果挑选，不复用旧G产生的H来构造新首学生输入。
如果实现原student seed会随版本名改变教学选择，显式固定旧36条教学/实践条件映射，不能默默换面板。

随后读取相同query16，依旧14/16/16/4四池；正式效用读出为新G对MT及只读E的576配对行。
精确复用normalized-flow已有MT/E行，要求相同参数/初态/scene/env与policy RNG/官方口径及真实native噪声前缀；
不改变seed追求缓存。如消费者发现真实身份不兼容，由实验在不改变科学配对的前提下修正或补齐必要对照并如实计费。
旧normalized-flow G的576行也直接作为同面板历史paired comparator，不重跑旧G。
E只读，不进G条件、Actor候选、信用rewards或H；9个E=MT仍引用MT原行。
每task两个teacher-selection重合query（共72）仍仅是train机制读出，保留分列，不据结果挑面板或新增资格arm。

报告MT/旧G/新G/E完整per-task/per-suite、覆盖、R/G/L、churn、success overlap，保留所有不利个例。
沿E-only/MT-only/both-success/both-fail给出新G实际控制，检查教师净优势是否被传递以及MT保留代价。
从36套已保存参数做一次CPU端点算术：对自身E与P0/公共均值的MSE、目标方向、task差异，解释是否只有公共方向。
这些是定位证据，不能选checkpoint或替代实际控制。原native query已保存的首次动作差异可以直接汇总，不另开probe。
不运行旧xi/time输运诊断，因为新模型已无该过程；无需通过一组不再适用的解析消噪门。

## 6. 最近似历史及本次可失败预测

已完整复读`3a6f801:docs/action_forecast_writer_video_expert_manifold_design.md`§24–29，
以及findings§63和原rethink§9.5–9.9。直接专家因子回归不是未试过的新想法：
旧raw-MSE+cos/log-scale在rank16/source起点，地址修复后48→75/400，仍向公共方向收缩；
train目标平均方向占约.414能量，generated跨task cosine约.942、最近expert约.127。
地址保留修复是真改善，仍没有建立完整控制与迁移；这些负证据不因本次改名或更强base失效。

| 实际区别 | 本次安排 | 尚未得到的保证 |
|---|---|---|
| 旧16-phase冻结causal innovation且零/phase-constant必须identity | 现有可学习双meta的真实Z/H/L、有序Reader与真实自身实践 | 原生特征必然包含可迁移操作知识 |
| 旧cross/axial近共模后再乘性地址绑定 | 保留地址进入局部value/residual的完整64坐标块与归一化双向通信 | 新decoder不会学习公共均值 |
| 旧source起点/rank16/train24 step2000参数目标 | 强MT冻结基底/rank8/36个真实有控制增量的共同E及固定S_G | 强MT上的教师优势必然被学生保留 |
| 当前v6仍学xi/time速度并让teacher混入x_t | 新端点只读合法条件，全部误差直接监督完整目标编辑 | 去掉噪声就能解决高维条件映射或held泛化 |

本次没有恢复旧raw-factor配方，也没有证明其原理错误。它检验当前native条件/通信/教师下，
取消已表现出学习失败的随机输运，是否足以学成有用端点。可以支持参数化取舍，不能唯一归因旧时间编码。

- 若端点误差实质降低且同面板控制有可信增益，支持继续完成共享/局部学习与held固定LoRA评估；
  此时仍须解决选择/保持并验证视频增量，train576不能冒充当前目标已完成。
- 若只学公共均值或loss下降而G仍近MT/更差，降低“现有条件编码+全坐标共同E回归可直接传递能力”的支持，
  不自动追加端点续训、LR/seed/rank或再补一层归一化。后继须改变确有依据的映射/功能学习机制。
- 若参数已显著接近E但控制不随之改善，更新对参数距离与实际函数保持的判断，不用内部漂亮指标掩盖负结果。
- 此批没有Actor更新，不能据读出断言选择信用改善；V/H因果与held迁移仍未解决。

## 7. 有界投入、执行与交付

只执行fresh G216→36个首学生实践→新G query576及兼容MT/E/旧G引用→整批科学交付。
没有自动warm/RL/held400、Teacher扩展、相邻点、视频controls或额外诊断。
预计训练在2–4张有效A40约0.8–1.6h（上轮两卡实测1.491h）；首学生只新增G，预计0.5–1.5h，
合计正式约1.3–3.1h，考虑实际物理准入可按1.5–4h登记；工程实施/必要验证另预计30–60min。
这些是基于旧实测的外推，实验按实际消费者修订，不能声称已测新图吞吐。
逻辑新增下限为36实践+576query=612 episodes；MT/E兼容复用时不新增，必要不兼容补齐须明确原因与实际成本。
新ROOT增量暂按16GiB上限估计、实际可能约2GiB；旧资源/恢复点保留，禁止复制大模型/数据。

实验自主完成架构/实际消费者检查、提交集成push、clean detached、双节点live GPU与data1独立quota准入。
按真实吞吐/余量选rank与frame chunk，不因历史两卡固定物理安排，不扩logical B4或任务权重。
项目整体遵循live常规6/8卡与单节点6卡上限，训练和读出按实际吞吐安排，不额外固定更小整批并发上限；
不等凑卡、不为填显存占卡，profile不扫描科学变量。
实际wall/GPUh/失败成本完整保留，不以GPUh预测阻止有收益的正常启动。

canonical/Git窗口继续由实验独占。请把本决定和归一化批新认识整合进现有active design/current state/findings/history，
详细推导可进入现有analyses体系并由唯一active引用；不改AGENTS动态时间线。
归一化批的主讨论CPU脚本/json请保存到其ROOT/analysis/main_scientific_readout，临时目录不作唯一证据。
一条可存活的整批结束/异常回报接续本主讨论；不增加子阶段心跳、重复Queue或轮询等待者。

## 8. 本次有界检验的实际结果

本决定已按原范围完成：新G341/576对MT340（旧G339/E455），MT→新G R314/G27/L26，没有明显控制增益。
端点MSE .961878，但输出公共方向占编辑能量97.677%，相对自身常量均值的条件增益仅.001424。
实际更新/传播/梯度事实保留，结果未支持只取消独立噪声输运便足以传递共同E控制优势。
完整原件与解释见[确定端点整批分析](proposal_deterministic_endpoint_query16_20261012.md)；本决定不自动授权后继训练。

# 当前 source 上的困难任务教师与学生访态恢复

2026-10-08。经progress登记后为唯一active批次，接续Owner持续自主推进的授权。
本批建立或否定一个训练期功能教师的具体资格；不把任务专家作为EMBER部署方法，不训练Writer，
不由教师正例自动启动蒸馏、RL或完整新架构。main负责整批后的科学消费与后继，有限阴性不终止项目。

## 1. 当前不足、已有约束与本批要改变的判断

跨情境配对两臂完成216更新和1762行：Within147/148、Product152/151，强T161/150，强MT153。
Product对Within净+5/+3，但对原T correct的净−9全部来自task3的−2和task31的−7；
source54中Parent21→Within36/Product35，Product没有独有新增。有限结果没有支持继续把情境配对作为优先修复。
它保留了真实学习能力，也保留教学条件间的交换；不是视频无用、普遍不变性不可学或冻结source不能控制的证明。

当前主FM的联合分布可写成`p(V|task) d_demo(s,a|task)`；部署是`p(V|task) d_{pi_G(V)}(s|task)`。
这里s包括机器人自身图像/state，V是action-hidden教学。更换V会改变完整LoRA，也可能改变实际访问的状态。
在目标投影中，视频写出的M固定，但执行修正为`M(V) A x_pi(s,z_tau)`，其中x来自这套LoRA自己的逐层计算。
训练在示范状态上的梯度会经过该x，却不自动覆盖这套条件策略实际走到的其它状态。
本轮Product改变教学scene与示范query的配对，没有改变query仍来自示范状态这一点。
这给出一个值得区分的学习分布问题，不证明它是目前唯一根因，也不证明off-policy FM原理上不能解决它。

所需的额外监督不能只叫“专家动作”。固定NN已经在自身状态反馈检索示范，但Value是原示范的五动作，
不修正当前状态与示范状态间的差；T50接力109→91，新增11同时丢29。原T/NN都成功的71行仅保留59，
说明初态成功不认证中途接手。也不能把旧专家bank直接挂到新source：旧bank基于8月source e2cc238、旧协议/rank16。
旧SEOD/GOMQ确实让合法视频学生经过自身完整10步去噪端点学习，曾达143/151但未稳定保持；
旧phase aggregation也确实在学生访态上查询expert velocity，却只学习privileged-code decoder，未接视频Writer。
这些负证据降低简单复用蒸馏故事的支持，不能改名清零；具体接口和完整正反结果见findings§392。

下一种可用标签需要来自在当前source上、对当前状态确有可靠控制的函数`a_E(s,L,noise)`，
而非附近示范的固定动作。相同rank128单LoRA的任务专家可直接改变自身执行hidden中的完整A/B，
用同一source的既有感知与动作计算拟合任务反馈；这比检索Value多了可学习的状态相关修正。
这里相同是38-target/rank128的执行家族；每个专家独立A/B，而T的A跨教学共享，
不能由专家成功证明当前G_T的条件输出族已经足以表示全部专家，更不能省略后续功能迁移检验。
但原始FM本身仍可能学不会恢复，因此本批直接检验控制，不先把它命名为强教师。
这不是最终视频方法的可表示/可学/可迁移证明：即便E可靠，G仍须仅从合法V/L生成参数，
并将监督传回真实自身执行与教学路径；本批不实施或预授权这一后段。

## 2. 唯一模型、四项数据与学习预算

只训练四个独立任务专家，任务在最近native_video_control诊断前已经登记为困难面板，未按本批结果选取：

| global task | 精确语言 | 原T/4 | 原MT/4 |
| --- | --- | ---: | ---: |
| 12 | pick up the salad dressing and place it in the basket | 3 | 2 |
| 29 | put the wine bottle on the rack | 2 | 0 |
| 32 | turn on the stove and put the moka pot on it | 1 | 2 |
| 38 | put both moka pots on the stove | 0 | 1 |

这四项均为当前coverage train24内的授权任务，各50条canonical成功示范；MT300此前已见过它们。
它们不再沿用上一批fit32的临时留出角色；没有改变official 24/8/8，更不是未见任务泛化。
task38完整官方目标还包含TurnOn，不能只报告两个On；全部任务按原BDDL目标AND裁决。

- 每个专家由**同一强MT300完整38-target rank128 A/B**初始化，独立fresh optimizer，互不共享更新。
  权重只读`/data0/user/ymdai/ember_runs/coverage_retraining_20260920/training/mtbc/checkpoints/step_00000300/lora.safetensors`。
  冻结aligned source1000、图文prefix、normalization、tokenizer及`configs/pi05_lora_rank128_aligned.json`；
  不叠第二adapter、不改原MT、不训练source或新增Reader/视频Encoder。
- 模型输入仅当前自身双RGB、8维state与exact L；只有本task的动作作为FM标签。
  task metadata用于训练数据/专家选择并明确属于训练期特权；不把四专家路由、平均或组合成部署方案。
- 复用近期M的实际数值与优化口径：A/B与Adam状态FP32，source计算BF16/正常TF32，
  AdamW lr1e−4、betas(.9,.95)、eps1e−8、weight_decay1e−4、clip1；无scheduler，记录为显式null。
  这不是修复原T混合dtype或进行dtype/LR对照，不重设旧T optimizer。
- 每task固定480更新、112 query/update，即53,760 query；四项共215,040。
  相比近期M128的全任务14,336 query，此处给每个困难任务约54k直接监督，避免只给少量访问便下容量结论。
  数量是有界学习机会，不预先宣称收敛；不追加seed/LR/rank/长度扫描。
- 采样seed20261008；每task固定逻辑query序列n=0..53759。令c=n//50，
  episode为`permutation(50, SeedSequence([root,task,0,c]))[n%50]`。
  该episode在第c次出现时，frame从它的`0..length-2`无放回循环取：
  长度m=length−1，`permutation(m, SeedSequence([root,task,1,episode,c//m]))[c%m]`。
  每更新消费连续112个query，物理切分不改顺序、loss权重与实际随机量。
  每更新分四个逻辑28-query块，以现有`task_logical_batch_policy_rng_seed`生成种子，
  optimization_seed7、task、task_visit=4*update_zero_based+block、该块完整demo/frame列表；
  `flow_sample(noise_endpoint=False)`保留当前原生tau/noise分布并固定random_batch28/offset。
- 唯一loss沿近期M实际消费者，是完整50×7原始随机tau FM；沿用offset1、末动作补齐与mean_velocity_loss，
  不另加padding mask、tau1、first5加权、蒸馏、奖励、成功筛选或aux。无teacher-video输入，
  不读取validation/test action、state、reward，也不生成新训练rollout标签。
- 每task保留0/160/320/480完整checkpoint；160只作恢复点，320/480是固定相邻读回节点。
  两节点都报告，唯一末点480，不按task各挑不同checkpoint或把二者success union当模型。
  checkpoint保存A/B、optimizer、显式null scheduler/scaler、sampler/query cursor、metrics cursor、rank RNG与实际拓扑。

## 3. 80个固定闭环与T50接手定义

固定原36-task seen144中上述四task、init32/33/34/35，共16起点；
沿用原scene、env seed7、policy seed root7、各原T条件及原生horizon。原T6/16、原MT5/16直接复用，不重跑。
所有起点的原T均执行超过50步，最早122步成功；不以“没有可接手起点”择例。

新读回恰为：

1. 专家320和480各从官方初态执行16行，共32。
2. 同两节点各接续固定T前50真实动作后的16行，共32。
3. **原MT300**接续同T50起点16行，作为当前共享控制的接手参照。

合计80新完整episode，不增额外接口smoke/MT初态/T全程；16条MT接手可与训练独立并行。
有限训练侧诊断不称strict paired400，不用于正式Writer checkpoint选择。
T50只代表这些固定早期访态，不能认证更晚失败状态、完整学生occupancy或未见状态上的广泛恢复。

T50严格复用`teacher_state_handoff_20261007/memory/cohort.json`中的四task16行及其原件引用，
原T来自`denoising_return_writer_20261006/readouts/parent/seen/ODE/evaluation/results.json`的132行
和`query_conditioned_transition_read_20261005/parent/evaluation/seen12/results.json`的12行。
优先复用已核实的T50原动作重放机制（36f2a044及其实际source），不在线运行T、不重新生成视频LoRA。
从原official初态/settling后执行原capture的前50条真实raw action，保持原环境历史；第50步后的观测交给新控制器。
不能只设置qpos/qvel便称相同接手状态，也不能重置控制器/物体或补一段完整horizon。
比较原记录的前51个EEF/quat/gripper/body/谓词，按既有physics容限与正常单档RGB量化政策核对；
需要且现有原件不足以恢复的科学边界回main，不能猜控制命令或换起点。

接手时专家/MT从**绝对replan序号10**起沿原policy noise时钟执行10 flow steps，完整生成50×7、只执行前5并replan。
不从seed0重新计时；初态臂仍从replan0开始。T前缀为确定性原动作重放，0在线T/source/Writer计算。
接手后三种控制器接收完全相同物理起点，单个静态完整LoRA；精确语言仍是本task。
总步数含原T50前缀，suite horizon按原合同，成功即终止。该接手是训练侧诊断，明确不是零交互部署分数。

每行保存raw结果、完整50提案/噪声种子与实际末prefix、执行动作、T+1 EEF/quat/双指/body位置和完整goal谓词；
接手段明确前50命令来自何处，原提案未执行tail不得伪造为执行。沿canonical cost-balanced queue、long-first、persistent workers。
80行中每panel四task的init32为full，共20full/60compact。固定审看480初态4clip、480接手4clip、MT接手4clip，
共12clip×8实际保存时点×双RGB=192图；首个接手后观测应在时点列表中，若未保存terminal RGB必须明确缺失。
不由RGB/物体运动推定抓稳、接触、唯一失效原因；不额外搭接触仪器或读取held状态。

## 4. 如何形成方法判断

逐task报告初态与接手两节点、原T/MT/旧NN接手的对应16行：success sets、R/G/L、churn/Jaccard，
并按原T成功6/失败10分层，保留“救回失败却破坏原成功”的交换。
旧NN接手16行从其已完成144原件切出复用，不新运行。对每个控制器区分初态与T50，不能把两者择优union。
并列320→480能力与loss/query覆盖，观察相邻保持；初态更强不自动认证恢复，单谓词改善不算完整目标。

下一阶段有用的**局部教师候选**事前标准是：两个固定节点的初态及T50各至少12/16，
每task每panel至少2/4，T50对原T的6个成功每节点至少保留5个。
这描述所需的显著局部控制裕量与保持，避免因1–2条波动便投入完整蒸馏；不是统计显著性或总体泛化认证。
同样展示原MT在T50的能力：若共享MT已经足够好，就不能把可恢复性全部归给专家独立学习。
即使门槛通过也必须由main判断完整结果是否支持下一步，不自动开始Writer或扩大专家bank。

- 初态和T50均明显强且相邻保持：加强“当前source的可学习控制函数存在，值得有界检验给合法G提供此函数监督”的支持。
  仍未证明视频内容能预测它、来自实际访态的监督比原FM更好、未见任务迁移或最终超MT。
- 初态强、T50弱：直接否定本候选的广泛纠正教师资格，不把初态成功蒸馏成学生状态上的假标签。
- 两者弱或严重交换：本配置没有建立所需函数；结合真实查询覆盖/曲线判断学习充分性，
  不外推rank128/冻结source/专家监督不可能，也不以欠收敛为由自动续训或扫描。
- 480明显仍在改变而相邻不稳：完整记录证据，main负责是否有理由做新的有限投入；执行者不自行延长。

这些分支都结束本批新增计算。它们改变后继标签与学习路线的优先级，不结束EMBER；
不将此四项训练任务成绩与held400分数相加或冒充最终视频增量。

## 5. 执行、预算与交付

唯一root：`/data1/user/ymdai/ember_runs/aligned_teacher_recovery_20261008`。
预计4–7 wall小时、10–12完整GPUh；依据近期M128/14,336query实测.5326GPUh，
本批215,040query约15倍，对应约8GPUh训练，加工程/profile/加载/80读回及消费余量。
单任务数据局部性与物理并行可能改变吞吐，不能把线性估计称已实测。
硬上限实际承接起10wall小时、16完整GPUh、data1新增峰32GiB，包含所有失败/加载/临时产物。
若profile后预计无法完整承接，或实际有超预算/科学语义/原件有效性边界，保存有效进展并回main，不静默缩科学矩阵。

沿原FM实际消费者做最多一个独立4更新/448query的2→完整恢复→4 smoke，选task38，不混入正式四专家。
最多三次同一个112-query正式批的无更新forward/backward profile，比较物理micro28/56/112，
每次都包含完整112逻辑样本及真实梯度；0额外环境smoke。按吞吐/峰值主动利用显存，物理实现可合并逻辑28块，
但须保留其已登记实际tau/noise与总均值。四个独立教师可并行；单专家物理rank数由实际吞吐定，不锁死为一张卡。

launch前现场核strg01 data1独立quota、个人/相关du、共享空间与checkpoint/capture/临时峰；旧data0只读。
每次launch同时核两个节点，执行项目8卡/空闲≤10时6卡/单节点6卡上限，训练/评测/EGL共同计数；
不跨节点拼一个训练、dummy占卡或干扰他人。保持NUMA/NCCL_P2P_DISABLE=1/deferred NCCL及完整checkpoint边界迁移合同。

实验session独占工程/Git，从最新main建隔离树；复用当前data/FM/DirectLoRAParameters/checkpoint/evaluator与既有重放机制。
旧rank16 expert入口的固定source/合同不能直接绕过；不为本任务复制整个旧trainer或常驻第二套pipeline。
按code-architecture-gate保持单一内聚临时owner和必要公共接口；实际消费者针对性验证后自行集成push，clean detached冻结运行。
已定位、保持科学语义的工程修复在原预算内闭环，完整保留来源与失败费用；main不重复工程审查/测试。
停止后退役专用入口/hooks，保留全部checkpoint/原始行/报告/冻结Git与复现命令，清理task-owned工程树。

长任务等真实退出事件，不轮询log/cache、不发阶段/心跳/selfQueue。只在整批结束或真实边界一次可靠回main，
交回canonical tracked/Git，报告来源、实际成本、完整证据与未知。main核实际承接；计划/入队不等于训练已经启动。

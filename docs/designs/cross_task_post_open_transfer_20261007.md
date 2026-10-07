# 强T已学放碗控制在task23实际post-open状态上的冻结迁移

2026-10-07。Owner最新明确继续自主推进；本批是有限机制诊断，0训练，不是新部署方法或checkpoint选择。
main负责科学合同及整批裁决；既有实验session实际承接后独占工程、canonical tracked/Git并闭环。

## 1. 为什么做，以及最近似历史

strong T2340的完整条件参数在task23原9个已打开起点上仍0/9，删除M的Common也是0/9；
固定完整参数、改为明确目的地或仅剩余目标的执行文本也未取得完成。此前没有证明T在这些状态上已有可调用后续控制。
与此同时，同一个T2340在task42“put the black bowl in the top drawer of the cabinet”的既有seen四行全部成功，
init32/33/34/35分别对应demo30/14/43/09，完成步101/117/113/127。四套已编译完整LoRA全部保留，不能挑最高视频。
task42是LIBERO-90 task2、KITCHEN_SCENE10，初态Open；task23是Goal task3，原先关闭。
二者同为黑碗In上层抽屉，但桌面、杂物、柜体位置与机器人实际状态不同。原4/4不是跨场景或接手资格。

具体问题：固定同一T公共参数，task42条件所实现的控制，能否从task23已到达的状态完成同一个剩余物理目标？
这区分“这个已学控制确可迁移、但原task23编译没有产生它”与“这个控制仍依赖原场景/指令条件”。
只裁决这项具体可复用性，不能从一个task的成功反推通用primitive表示、全部知识已学到或新架构必然有效。

旧ECP确实曾中途安装另一个task的完整LoRA并改phase语言：55/56单任务各50/50，组合两方向0/50与44/50；
65/68单任务43/50与47/50，组合28/50与9/50。它同时提供接手正例与强上下文依赖反例。
后续recovery的14/100为禁止第一目标再掉落的严格口径，普通environment success为31/100，不混作当前官方In口径。
旧学生访态expert velocity蒸馏没有证明expert能从这些状态闭环完成；另一批2773/3998 query的蒸馏被取消、未执行。
本批直接运行真实continuation，继承这些边界，不恢复人工process数据、expert蒸馏或多策略部署。
P/I教学对应结果及已结束NN接力109→91也限制乐观预期，不能用另一任务初态成功升级为可靠纠正教师。

## 2. 唯一变量和完整计算

同一T2340在每个target线性层使用 `W0 + (B0 + M_c) A0`。冻结source、A0/B0、全部38-target与官方执行口径。
`M23_i = Writer(V23_i,L23)`，`M42_d = Writer(V42_d,L42)`均直接读旧已编译bank；不重新生成/native读取。
这里更换的是完整教学condition所生成的控制参数，包括教学语言和视频的共同影响，不能称纯视频变量。
每条续行开始前安装一套完整因子，后续保持；没有融合、平均、拆层、缩放、第二adapter或rollout内切换。

执行 `pi_(B0+M,A0)(own_RGB, own_state, L, flow_noise)`，完整十步flow、执行前5再观测。
在固定L的比较中，M改变各层对实际自身hidden的作用及后续反馈；它不是固定动作模板或单层输出差。
再交叉L用于区分参数迁移与执行文本依赖，不改变Teacher编译、不把语言替换包装成canonical部署。

| 因素 | 固定取值 |
| --- | --- |
| 完整因子 | 每起点自己的M23；task42已存demo30、14、43、09的四套完整M42 |
| 执行语言L23 | `open the top drawer and put the bowl inside` |
| 执行语言L42 | `put the black bowl in the top drawer of the cabinet` |

5套参数×2语言×9起点=90条新续行。M23两文本各9作同期对照；四donor每种文本各9，不能把重复起点当独立36初始化。
L42与旧remaining_goal字符串不同，必须有M23+L42对照；不直接借用旧文本阴性。
原environment/official goal/global target仍task23；donor task/condition metadata另存，不把task42 bank映射误当新场景。
修改执行语言、跨任务参数安装和使用既有Open选择起点均是显式诊断例外，不能进入EMBER部署或报告初态成绩。

## 3. 固定来源、起点和配对

唯一起点来源 `/data1/user/ymdai/ember_runs/task23_post_open_operator_20261006/cohort.json` 的全部T2340九行：

| init | branch控制步 | 剩余步 | 原task23 demo |
| --- | --- | --- | --- |
| 3 | 120 | 180 | 38 |
| 7 | 260 | 40 | 36 |
| 8 | 60 | 240 | 20 |
| 14 | 60 | 240 | 12 |
| 16 | 65 | 235 | 39 |
| 19 | 65 | 235 | 7 |
| 28 | 285 | 15 | 10 |
| 32 | 290 | 10 | 25 |
| 34 | 185 | 115 | 37 |

branch仍为原首次Open向上取整至五步边界；不另挑更方便的状态，三条极短剩时完整保留并单列解释。
每套参数/语言九行最多1310剩余控制步，90行最多13100；总horizon始终300，In即成功退出，不延长追回成功。
原task23 banks来自 `operator_read_write_learning_20260928/continuation2340/T/banks/2340/`。
四donor banks来自 `denoising_return_writer_20261006/readouts/parent/seen/bank/task_42_demos_{30,14,43,09}.safetensors`。
以上相对run路径均位于 `/data1/user/ymdai/ember_runs/`；source/公共shared均引用原T2340合同及实际metadata。
donor的原四成功行与编译身份在 `denoising_return_writer_20261006/readouts/parent/seen/ODE/evaluation/{results,run_contract}.json`。
只核必要来源、shape和实际安装，不新增tensor逐项比对或hash。

复用原scene和自身实际actions[:branch]构造每条起点，零额外settling、零物理改动；不得用teacher动作造起点。
优先复用已验证的动作前缀重放消费者，不把仅qpos/body pose恢复冒充完整中途状态克隆。
与原Full捕获branch核EEF、对象、三层关节与必要sim/ctrl/warmstart/OSC/gripper状态，沿用原配对准入，
含1e-8运行状态容限、原前缀EEF/碗≤1cm、In序列一致且branch仍Open；这是现存配对合同，不推广为低位数值要求。
若不符，保留实际有效范围/原因，不放宽容限、换起点或用旧失败补齐新对照。
噪声使用原policy_noise_seeds，从branch/5索引续取，原动作重放不消耗policy RNG。
官方render256/model224、双相机180度、8维state/7维action、normalization和source完全沿用。
新M23+L23自行闭环，不强制原后缀；正常数值差不要求每步相等，旧Full0/9仅作历史参照。

## 4. 保留什么，结果改变什么判断

所有90行保留每步真实命令、EEF/夹爪、碗/原对象、三层qpos/qvel/Open/In、policy seeds、退出和配对证据。
沿用现有真实柜体contact采样及其substep缺项；位移、开度、中心距离不冒称抓稳或最终成功。
首次实际规划保留normalized50×7和physical前5，证明参数/文本进入真实消费者，不另增功能query矩阵。
T/init8与19的全部十臂共20 full捕获实际replan双RGB，其余70 compact；全部数值行为仍保留。
固定20 full每条展示8个按实际后缀时间等距索引的观测，共160对双RGB；无终止图就明确缺项，不插值造帧。
优先直接比较同起点/文本的五组行为；任何改善与反向损失均保留，不只呈现成功donor。

逐donor、逐语言、逐起点报告In，分别对同期M23作retained/gained/lost；语言效应在每套参数内比较。
报告跨四donor共同成功/分歧与全部行，不挑一个最佳donor给单一成绩，不把90行当paired400或伪独立样本做显著性。
三条剩余≤40步与其余六条分列，但主面板保持全部九条；局部搬碗改善只能用于描述尚缺哪段控制。

- M42在原L23下有成片新增且跨donor保留，支持同一执行文本下已有条件控制可在这些状态复用；
  原M23没有实现该控制成为更强的具体编译缺口证据。仍不证明视频缺哪项语义、为何训练没学会或单LoRA组合已成立。
- 仅M42+L42改善，支持参数与执行文本联合可迁移，不能单归因Writer没调用现有技能；M23+L42结果必须同时解释。
- M23+L42本身改善，先保留表达/条件分布效应，不把该收益记给donor或新增视频控制。
- 全部没有完整改善，关闭这四donor在这九状态/两文本下的迁移假说，降低直接复用这套已学控制解释；
  不能推成所有primitive不可组合、状态不可救、所有LoRA没有能力或需要更大Reader。
- 混合或单例正结果保留效应大小、起点和剩时，最多说明局部可行；不自动升级为强teacher或新训练依据。

任何结果整批结束回main。没有自动换donor任务/视频、语言/切点扫描、参数拼接、新数据、梯度、distillation、RL、400或Test。
不改原T161、task23初态0/50；本批是validation已有轨迹的冻结诊断，不使用held actions/reward产生梯度或选择checkpoint。

## 5. 工程、资源和交付

唯一新root `/data1/user/ymdai/ember_runs/cross_task_post_open_transfer_20261007/`；大资产只读复用，全部新增在data1。
预计实际承接后1–2小时，硬3小时wall、2完整GPUh、16完整CPUh、8GiB新增峰，包含工程/失败/加载/EGL/分析/冻结与临时。
依据是原64条同类续行12175控制步含工程52.39分钟、.381711635 GPUh；本次13100步及更多参数/文本分组，工程占主要不确定性。
新root/大输出前核strg01 data1独立quota、个人及相关du、共享容量和峰值；每launch同时live核两GPU节点与AGENTS总卡上限。
按吞吐、峰值与实际模型共驻选择persistent workers/物理batch，cost-balanced queue、long-first；
首init8与19的M23和demo30各两文本共8条计入90，兼作实际消费者与batch/并发检查，其余就绪独立调度，不另增科学行。
有实测显存余量需在固定90条内验证更大物理吞吐，不能套保守默认、等待凑卡或制造dummy/profile rollout。

工程从最新main独占分支/worktree实现，经实际消费者检查并集成push后使用clean detached frozen；旧冻结树不改。
已定位且不改模型/数据/科学口径的接口修复在预算内自主闭环，只续未完成行，保留全部失败/费用，不等main工程审批。
科学有效性原因不明、合同或预算变化才回main；不能以科学阴性修bug或增加面板。
交付contract/冻结90-row映射、原始行/全部配对/差额/行为/RGB、实际来源和代码身份、launch/exit/完整费用/释放、completion及报告。
整批后退役本批专用入口和hooks，保留Git/frozen/raw/失败；tracked/Git与main串行交回。
只在整批完成或真实边界通过既有可靠入口回main一次，正常过程等退出事件，不发阶段/心跳或自Queue。

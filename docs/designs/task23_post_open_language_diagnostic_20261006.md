# task23已打开状态的执行语言诊断

2026-10-06。固定已有完整LoRA，只改执行policy的文本；一次机制诊断，不是新Writer或正式性能评测。
main负责科学解释与合同，既有实验session独占工程、运行及Git窗口；以progress的实际承接为准。

## 1. 具体问题与辨识价值

findings§354–355的32起点Full/Common比较已经削弱“删除条件作用即可恢复已有完整后续能力”：
T为0/9对0/9，C为1/23对0/23。T19与C16中Full能离柜搬碗，Common不能；C16 Full实际完成。
两个init8的Common虽然末态开度较大仍不搬碗。不能把条件参数统一视为开柜干扰，也不能把Open当足够控制状态。

仍未区分的是：在这些具体自身状态，后续控制本身不可由现有完整参数有效调用，还是完整执行文本持续提及
已经做过的开柜目标，妨碍了后续目标的调用。第三个解释是原句中“inside”的指代/表达影响，而非先后目标冲突。
本次直接干预这三个解释的不同预测，不假定注意力指向、语言词向量或某个神经模块已经确诊。
若固定完整参数在仅给剩余目标时能完成，可说明同一参数系统在改变文本条件后具有该状态的后续能力；
若保留开柜但明确目的地也同样改善，则不能归因于移除已完成目标。两种结果均不自动证明视频编译理解了阶段。

最近似历史是§354冻结条件删减、§352单开51与初态Open的42不可拼接、机制§98/115同谓词不同完整状态的交叉续行。
它们没有改执行文本，也不能作为本次结果。此前角色crop/选对象、动作辅助、L/R读取及局部控制负证据仍约束方法投入。
本批不以小诊断替代EMBER绝对性能目标；它只决定是否有依据优先研究“教学过程怎样改变自身当前目标的调用”。
没有改善则关闭这项执行语言解释，不继续找同义句、改更多任务或以该假说直接正式训练。

## 2. 唯一变量与数学对象

对每条原condition，完整因子固定为W=G(V,L0)，直接读原已编译38-target bank；不重新运行Writer或native读取。
T的W为(A0,B0+M)，C为(A0+S,B0+M)，仍是原T2340与conditional_read_write900，非Gamma模型。
执行函数记为pi_W(o,s,L,epsilon)，包括固定source、完整LoRA、真实双RGB/state及十步flow。
比较只改变L，因此前缀token/KV以及后续action hidden会自然改变；source图文prefix权重与全部LoRA参数均不变。
不把这种有限干预写成单层线性语言方向或纯attention效应，也不改教师语言或把两条语言分别编译成新LoRA。

| 臂 | 执行时完整字符串 | 来源 |
| --- | --- | --- |
| exact_full | `open the top drawer and put the bowl inside` | 复用§354的32条实际Full续行，包含C16成功 |
| explicit_full | `open the top drawer and put the bowl in the top drawer` | 本次32条 |
| remaining_goal | `put the bowl in the top drawer` | 本次32条 |

explicit_full与remaining_goal共用同一明确目的地，后者只删除前一开柜分句；exact_full与explicit_full保留完整目标。
这仍是表达和条件分布的有限干预，不宣称完美分离全部语义变量。三臂官方goal始终是原In谓词。
Writer的合法输入及原任务定义仍为L0；本次修改执行文本明确登记为诊断例外，不计初态EMBER成绩或推荐部署改写。
每条新臂从branch到退出只使用一套固定完整LoRA和一条固定文本，没有在线阶段检测、语言切换、二次看视频或第二adapter。

## 3. 固定起点与旧对照复用

唯一旧root：`/data1/user/ymdai/ember_runs/task23_post_open_operator_20261006/`。
复用其中封存cohort的全部32起点（T9/C23）、原scene/命令前缀、branch、teacher、seed序列及剩余时限。
原32条Full的canonical row.json、实际trace/prefix路径和完整branch运行状态就是固定exact_full对照；
部分NPZ在attempt子目录，只按row.json实际路径读，不猜路径。C16实际成功不可换成更早原400失败，也不重跑择优。
旧Common不新增计算，只保留为已完成背景。旧28条失败后缀缺失仍明确登记，不用缺项推断行为。

本次每起点两个新文本，共64条新续行；不增加起点、视频、init、seed、checkpoint或重复exact_full。
新两臂均从原post-settling scene执行原自身actions[:branch]到同一边界，不改物理、接触或对象；零额外settling。
优先复用已验证的前缀恢复与连续采样，分别原动作重放，不引入不完整中途克隆。
核每条新臂与旧Full已捕获branch的EEF、物体、三层关节、sim/ctrl/warmstart/OSC/gripper运行状态，
沿用上批的1e-8运行状态容限；原前缀EEF/碗各≤1cm、完整In序列一致且branch仍Open的准入保持。
这仅检查实际配对；不新增全树hash、因子逐tensor扫描或要求浮点后缀相同。若不符，不放宽、不换起点或用旧失败补齐。
明确报告可配对范围；科学有效性未知或预算越界才回main，已定位工程修复在原合同内自行闭环。

复用原policy_noise_seeds，从branch/5索引开始；未来噪声逐规划固定，绝不从0重启。
render256/model224、两相机180度旋转、真实8维state/7维action、十步flow/前5执行与原source/normalization保持。
所有原轨迹总horizon300；64新臂剩余步上界12210，达到原In即止。5个短余时起点及全部反例保留，不增加时间追回成功。
原32个branch由上批事前cohort决定，不因Full/Common结果选择；C/init38的早前重放边界继续不纳入。

## 4. 原件、主要比较和停止条件

全部新64行保存真实命令、EEF/夹爪、原物体、三层qpos/qvel/Open/In、步末真实柜体接触/力、seed、退出与配对结果。
首次规划保留normalized50×7及physical前5，用于确认真实文本干预进入同一消费者；无额外功能query矩阵。
固定T8、T19、C8、C16的两个新臂共8 full，保存每次实际规划输入双RGB；其余56 compact。
旧四条对应exact_full画面只读引用；区分旧/新来源和绝对控制步，不重建终止帧、逐控制步图像或缺失失败后缀。
所有新失败尽可能即时落盘实际partial前缀/后缀/图像/计时；已完成行不得重复，修复仅续未完成行，全部失败计费。

逐模型、逐起点列出三臂In、保留/新增/丢失；主要比较为remaining_goal对explicit_full，另外两者各对exact_full。
三臂共同可配对才进入主要比较，保留所有排除/不符及各自结果。列出1cm位移/3cm抬高、接近、开度、接触与剩余时限，
这些是描述，不作为成功gate、抓稳真值或替代最终In。T/C年龄及结构不同，不作两模型间单因素归因。

- remaining_goal带来完整成功且explicit_full未同样带来：支持该状态下持续开柜分句影响后续能力调用；
  同时报告反向损失和有限覆盖。它不证明原视频内容丢失、某层错误或足以支持新架构/阶段切换。
- explicit_full与remaining_goal都改善：优先保留指代/表达效应；不把它包装成时间理解或视频动态修复。
- 两者无完整增益、或仅局部运动不同：降低该执行文本解释与目标调用修补的优先级，停止本语言诊断。
- exact_full更好或结果混合：保留原完整指令与条件共适配的正例，不按局部改善筛语言或初态。

单个成功只提供该状态的能力存在性；不能外推32起点总体、未见初态或其它任务。主要判断同时看效应大小与所有行。
任一分支均整批结束回main；无自动同义句/子句/切点/层扫描、新学习、任务扩展、正式400、RL或Test。
原T161/C140及task23初态0/1不改，本诊断不选择checkpoint、teacher或部署语言。

## 5. 预算、执行与交付

唯一新root：`/data1/user/ymdai/ember_runs/task23_post_open_language_20261006/`，旧大资产/LoRA/轨迹只读复用。
预计含工程45–75分钟；硬限实际承接起2小时wall、2完整GPUh、16完整CPUh、4GiB新增峰，包含失败/加载/profile/冻结树/临时。
依据是上批64条相同长度续行含工程52.39分钟、.381711635完整GPUh；本次物理/推理消费者可复用，文本长度是新增变量。
大型输出前核strg01独立data1 quota、个人实占及共享容量；每次GPU launch同时核两节点、项目总数与单节点上限。
按实测吞吐选persistent workers、cost-balanced dynamic queue/long-first和物理batch；上批batch16仅为已有证据，
实际新增不同文本长度的padding/处理须正确。有显存余量时在固定授权行内验证更大并发，不额外生成profile科学行。
首T8/C8两个起点各两个新臂计入64，兼作真实文本/pairing消费者检查，不要求先把整个panel串行跑完才扩可用吞吐。

只从clean pushed detached frozen运行；工程修复自行隔离、针对实际消费者核验、集成push及新冻结后续未完成行。
不原地修改旧冻结树、不以科学阴性当bug。整批后退役本批专用入口，保留Git/frozen/raw/全部失败与费用。
交付contract/cohort/三臂精确文本、32旧Full引用、64新原件、逐行/三方配对/完整差额/全部反例、资源/退出/completion。
仅一次整批完成或真实边界回main，交回tracked/Git窗口；不发阶段、心跳或重复自通知，无自动后继。

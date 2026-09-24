# 接近阶段动作通道的冻结因果干预

> 2026-09-24登记；执行状态与实际接收方见[progress](../../progress.md)。本批是机制定位，不是训练修复或部署方法。

Spec：`configs/approach_channel_causality_v1/experiment_spec.json`。
Study：`approach_channel_causality_20260924`，研究根`/data0/user/ymdai/ember_runs/approach_channel_causality_20260924`。

## 1. 证据与待检验机制

完整1600前段交换已完成，见findings§136及原study的`coordination/completed_recheck.json`。
Goal21固定接手策略时，B前段比C前段普遍更有利；Object14无同向普遍优势。
第25步两锚下所有50初态的物体body position相同，EEF位置差的中位数为Goal5.38cm、Object7.89cm。
Goal中C相较B距目标的平均水平距离大1.77cm、平均高度低3.61cm；Object中C水平距离小4.17cm、高度低5.27cm。
按原controller的限幅与缩放审计前25步命令，两个任务各50/50初态的C累计z增量命令均比B更负；累计命令不等于实际位移。
这是后验几何描述；body中心不等于抓取表面，距离不等于语义选择，位置相同也不证明完全无接触或姿态变化。

Goal在C前段后出现更多干扰物先持续位移，Object多数失败仍曾移动正确ketchup。
9月14日完整行为回放已有异质失败；9月13日空间Q/K监督提升一般能力却未恢复Goal。
因此不能直接由本轮差异宣布“只缺grounding”，也不能把延缓所有动作当通用修复。
本批将水平命令与下降命令拆开，检验更具体的接近协调机制，随后再追查生成LoRA为什么产生相应动作偏置。

## 2. 数学定义、竞争解释与预测

取原四臂已保存的实际环境动作前25步`a_t^B,a_t^C ∈ R^7`，t=0..24，不含settling。
当前官方OSC使用fixed impedance、delta pose；执行者启动前核对实际配置和动作顺序。
位置通道是0/1/2，旋转为3/4/5，夹爪为6；使用反归一化后的环境输入，不再反归一化。

在保持C的旋转与夹爪命令时，定义2×2干预：

`u_t(i,j) = (i*a_B[0:2]+(1-i)*a_C[0:2], j*a_B[2]+(1-j)*a_C[2], a_C[3:7])`。

`Y_ij(p)`是同一初态执行这25步后由冻结p接手、原总horizon内成功的指示量。
另保留完整B前段`Y_B(p)`作参照。比较是对**动作命令通道**的干预；非线性机器人动力学会使状态通道耦合，
不得把替换x/y命令等同于只改变最终x/y位置，或把替换z等同于纯时间变换。

接近几何的竞争解释：令`e_xy(t)`为EEF与目标的水平偏差，`h(t)`为相对高度。
首次接触时的可抓取状态还依赖姿态、夹爪、速度与物体几何；若下降使接触早于水平对齐，
则单个时点较小的动作MSE仍可能进入错误接触/恢复区域。这是待检验解释，不是已验证训练根因。

- **H_xy，水平接近偏置主导。** 换B的x/y应改变实际接近几何，并在C的z仍保留时改善Goal与目标/干扰物运动顺序；
  若只换z同样有效而x/y无作用，或x/y确已改变却无对应行为变化，则该强版本被削弱。
- **H_z，下降与对齐的时序主导。** 保留C的x/y、换B的z应改变实际下降/接触时点并改善Goal；
  若下降确已改变而成功和对象行为未改善，或只能换x/y改善，则该强版本被削弱。
- **H_coord，二者需共同配合。** 单独干预不足，联合干预有额外收益；预定义交互为
  `I(p)=E[Y_11-Y_10-Y_01+Y_00]`。交互及几何/接触读数共同判断，不由一格最高分命名机制。
- **H_other，其它通道或后续控制仍重要。** 联合xyz替换后仍明显不同于完整B前段，需保留旋转/夹爪及其物理交互的解释；
  该比较不单独分离旋转与夹爪。Object失败若仍在目标已经移动之后，不用Goal解释覆盖它。

上述机制可并存。若替换命令未产生预期几何改变，应记为未有效操纵相应状态机制；负分不能直接否定该机制。
任何成功都只证明本批冻结干预范围内的作用，不能称已恢复视频理解或得到部署修复。

## 3. 冻结矩阵与数据墙

- 原任务global IDs `[14,21]`，各init states `0..49`；两任务均为已暴露的Train诊断任务，不扩官方Validation/Test。
- 原四臂study `conditional_compilation_diagnostics_20260923`，固定B_language@630、C_video_fm@420 correct；
  donor使用同task/state原`executed_action_prefixes`，接手复用原correct banks与teacher映射。
- 五个前段：`C_all`、`B_xy_C_rest`、`B_z_C_rest`、`B_xyz_C_rest`、`B_all`。
  五者都在同一新执行合同下运行；已有同策略重放成功集合变化，所以旧分数只作参照，不能替代本批两个纯前段对照。
- 接手者`[B,C_correct]`；共`2 tasks × 50 states × 5 prefixes × 2 followers = 1000`分支。
  主读出为C_correct，B用于检验改变前段的效应是否依赖接手策略；不把两个接手者当独立重复。
- 所有分支只改前25步的登记通道。无平滑、缩放、外推、手工waypoint、额外限幅或延长horizon；保留官方controller原有变换。
- 无训练、梯度、teacher标签读取、checkpoint选择、LoRA融合或新物化条件。B是冻结的干预供体，不指定公共底座训练课程。
  simulator特权记录仅作诊断输出，不成为任何模型输入；分支中的双策略使用不是合法EMBER部署候选。

## 4. 实现、验收与记录

从最新main隔离开发，复用canonical evaluator、前段重放、RNG、trace/capture和动态队列；不另造平行evaluator。
这是已有动作干预接口的有界扩展；源码结构检查按code-architecture-gate，旧正式树与原件只读。
验证后集成推送；1000分支来自同一个clean pushed detached commit。不得运行中热修。

1. 同原seed/init、settling10、suite horizon、10-flow与执行前5动作；接手global replan index固定5，沿原stateless噪声继续。
2. 先核对每个task/state donor与source/bank身份及7维命令含义。每个混合prefix逐步只有登记通道来自B，其余来自C。
   保存实际发送环境的25×7命令、两donor路径、每通道来源和两套原命令；不能把合成命令记录成真实网络预测。
3. 纯B/C前段仍按原8维state重放阈值验证。混合前段按设计会改变state，不能强行匹配原state。
   同一混合prefix的两followers须有相同完整环境/controller历史；保存initial/cut sim state与controller记录供核对。
4. 延用每控制步对象body position、EEF姿态/夹爪、实际动作和完整BDDL谓词；新增实际可用的机器人-物体接触对，
   依据geom/body审计登记名字。接触在控制步采样，不能据采样空白声称整个物理子步无接触；遮挡或缺失不能编造标签。
   保留原持续位移读数。计算EEF水平偏差、相对高度、首个观测接触/目标移动时点，以检查通道干预是否生效。
5. 全部compact，固定states`0,25`的40分支保留双相机full capture。prefix画面和实际动作必须真实记录；
   混合prefix没有新的flow预测，不能伪造`action_chunks`。可在canonical capture中明确区分外部动作段与模型预测段，
   改动限于当前干预所需，不建立另一套通用轨迹系统。
6. 先完成两任务、states0/25、纯B/C前段、两followers的16分支，计入1000，核对纯前段回放与接手合同。
   合成通道机制检查仅用固定Goal21/state0，验证来源/实际施加/记录，不按分数选择实现。普通浮点差异不追逐逐bit复现。
7. raw rows、命令/trace、fixed cases、worker exits、run commands与完整结果保留。两个纯前段相对上一批的差异单列。
   主要比较来自同一新合同；prefix如已成功则立即终止并报告，不能强行采尾段。

每task、每follower独立报告五格分数及paired retained/gained/lost；事前比较为10−00、01−00、11−00、11−10、11−01、
B_all−11、B_all−00，以及交互I。按同一task的50个init state联合重采样20000次、seed20260924，保留描述性95%区间，
不把多前段/接手者扩成独立N，不用两task cluster bootstrap。按预定义几何/接触和成功联合裁决，不事后挑窗口或新阈值。

## 5. 资源与停止

新data0峰值上限8GiB，data1≤1GiB；40个双相机full cases的上界约4GiB，加trace、物化只读引用和余量。
Sol在创建根/正式launch前核对strg01独立quota、共享容量、实际峰值和两节点live所有权；最多4张物理GPU，遵守项目合计上限。
复用大模型/banks，无DDP；按已验证persistent workers吞吐执行。正常长任务等待退出事件，不周期读日志/cache。

缺资产、信息墙、命令来源、RNG、同prefix历史配对、纯前段重放或数值/资源合同失败时停止受影响执行并回报。
完成1000分支后主动Queue主讨论`01a0cd94-65da-7b22-8ca9-7ba35f454632`，给完整原件与机械汇总，然后停止新增实验。
科学解释及后继修复由主讨论在核对原件后决定；不自动加新任务、视频条件、通道、训练或参数扫描。

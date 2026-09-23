# Process Pullback Writer：共享可学习出口

> 文档角色：封存设计合同。原文中的“当前／active／下一步”仅指当时阶段；当前授权与暂停状态以[progress](../../progress.md)为准。保留原科学定义和结果，不据此恢复执行。

## 1. 当前决定与证据边界

Owner于2026-09-15要求完成原因诊断后，自主决定修改、实施、正式训练和测试，无需再次审查。本设计替代固定出口v1，
旧实现、900窗口和诊断的原合同保存在Git、冻结运行树及[研究历史](../research_history.md)。当前状态只看[progress](../../progress.md)。
唯一主要变化是裸source导出的LoRA增加共享的两侧线性变换；过程读取、7维q、数据、纯FM、rank16和学习曝光保持。

依据为完整train24有界对照：共同从固定900／teacher16生成的A/B起步，source与Writer冻结，只在跨episode支持动作上
用同一LBFGS预算分别优化q或完整A/B。独立初态32–35的闭环为source17、原输出20、free_q18、free_AB46（均/96）。
free_AB覆盖19/24任务，S/O/G/L为11/17/13/5；相对free_q净增28，task-bootstrap区间[+13.54,+44.79]pp。
独立动作full10 MSE为.171922→.150169／.140512；因此q的动作误差改善没有自动转成闭环能力。
这支持优先修正固定出口的参数化／优化约束，不能单独把责任归给PCA、证明数学容量上界，或证明共享RGB已可学。
诊断产物不得作初始化；所有拟合、模型、noise、raw rows与具体预算在原研究的`causal_diagnostics/`保留。

训练池teacher16–19在300/600/900为18/20/22，独立teacher46–49为24/22/26（均/96），未出现一致的训练视频优势。
故“主要只是换视频过拟合”不足以解释整体缺口。相邻回放显示对象／实例选择、获取、放置及组合推进均会失败；
不能将全部失败写成后半程遗忘或由画面直接定位内部模块。固定900的视频controls只描述旧模型，不参与本设计选择。

历史固定A投影损失、原生纠正的空间内获取缺口和共享mapping负例同时保留。旧calibrated A_free使用event-additive anchor、
family gate和局部特权目标；本次是乘法共享变换与合法过程q经完整执行FM共同学习。此区别说明实际检验对象不同，
不保证本次修正有效，也不授权失败后继续堆校准／gate／rank或超参扫描。

## 2. 完整数据流水线与信息墙

```text
exact language + 一条同步双相机教学视频（K=1、action-hidden）
 → 冻结vision/Gemma基础 + 共享VL Meta与Action Meta
 → 逐帧图文Z、完整50-horizon动作响应H
 → language引导状态角色对齐；真实相邻变化及双向上下文形成过程Value
 → 每个真实frame × 50 horizon的7维动作余切q
 → 同视频裸冻结source的导数与全视频native X的PCA16，得到A0/B0
 → 共享identity起步的L/R变换，得到A=A0 R、B=L B0
 → 唯一38-target rank16 LoRA；source按自身当前观测闭环执行
```

两路为同步agentview和eye_in_hand，属于同一条演示。stride固定5并保留真实末帧；两路均进入读取器与裸source编译器。
Teacher action/state/proprio/reward/terminal/task ID/filename/pose/hidden normalization或policy outcome不进入Writer。
身份字段只供加载、调度和审计。执行policy仍可读自身观测和state。部署一次调用、内部固定只读重放，无loss、optimizer、
环境试错、候选选择、第二adapter或闭环中重新看视频。Source基础权重始终冻结。

固定train24/validation8/test8；只有train24梯度，无额外meta tasks、q辅助、RL或负视频训练。Teaching与主action池均为
16–41，按同task跨episode采样且排除当前teacher。观测对应post-action future control。独立动作42–45、teacher46–49
只用于训练侧固定诊断。所有task等权，更多episode不称更多独立meta tasks。Test仅在本轮方法冻结后用于登记的最终读出。

## 3. 语义、动作知识和过程Value

共享Meta属于Writer，与过程网络共同学习。真实prefix给出Z，原生Action Expert完整50位置H直到实际learned read才聚合；
H不是教师动作标签。Teacher时间、action horizon、flow time和layer depth保持不同语义。

Language为每帧语义角色与H读取提供查询，形成e_t。过程内容来自d_t=e_t-e_(t-1)，递推m_t=a_t m_(t-1)+b_t d_t，
m_0=0；a/b可由当前语义及已有记忆条件化。另有反方向上下文，但正／反保持不同角色。静态语义仅进入查询和门，
不直接加到Value或q；无绝对时钟Value或bias旁路。重复相同合法画面的理想零变化条件仍必须给出q=0。
这只是结构性质，不证明真实视频已获得有益的方向理解。

q覆盖完整T×50×7；其余原生32输出维度补零。裸source公开probe固定[50,32]、seed1729、flow time=1。
末端q投影零初始化，其余读取模块fresh。裸source的probe-F0动作估计只作读取条件。

## 4. 裸source编译与可学习出口

给定同一合法视频，裸F0在所有Meta与执行LoRA作用域之外读取。对全部38目标层：

```text
G_l  = (1/T) sum_t J_(W_l) F0(V_t)^T q_t
A0_l = full-video bare X_l的top16右奇异向量（正交行）
B0_l = G_l A0_l^T
L_l  = I + U_L,l V_L,l          R_l = I + U_R,l V_R,l
B_l  = B0_l + U_L,l(V_L,l B0_l)
A_l  = A0_l + (A0_l U_R,l)V_R,l
DeltaW_l = B_l A_l = L_l G_l P_l R_l
```

每侧变换秩直接使用LoRA合同rank，真实为16，不新增待扫描的rank超参。L/R分别作用于该层实际native输出／输入坐标，
参数跨全部task和视频共享，不读取task身份或单独language。每侧U为零、V按相应native宽度正常随机初始化，故L/R初始严格identity。
不显式构造dense I，不添加独立参数更新。无论共享参数怎样变化，q=0仍有B=0；A/B联合输出rank16、alpha=rank的唯一LoRA。
PCA基内符号或正交旋转同时作用于A0/B0时，在最终BA中相消；变换作用于native坐标，不利用任意PCA列号作为任务标签。

原始A0与q→B0保持冻结；学习可以改变最终参数作用方向，故最终出口已不等于纯G P，不能继续宣称固定q→LoRA映射。
L/R是本轮待检验的共享归纳偏置：完整A/B局部可达并不证明这种共享变换足够，参数共享也可能引入新的跨任务漂移。
F0导数只描述局部关系，最终行为仍由完整非线性policy与新初始化闭环裁决。

## 5. 联合纯FM与精确信用

Writer（含L/R）和两组Meta、optimizer、scheduler、sampler及RNG全部fresh，不加载旧900或任何诊断拟合。
初始q=0、L/R=I给出合法identity；无需每个可训练tensor首步均有非零梯度。
唯一loss为同task跨episode执行FM，完整50 horizon、7个真实动作维度；不混辅助损失、特权拟合标签、RL或trust回滚。

每logical update四task、每suite一个，每task一个K1条件和64个queries，共256 queries、task权重1/4。
物理chunk／成本分工不改变权重。优化沿用AdamW lr3e-5、betas(.9,.95)、eps1e-8、weight decay1e-4、warmup8、clip1。

每条件先保留无梯度裸A0/B0，通过L/R生成完整state，执行FM得到76因子的cotangent。
再在同一参数版本重放小型出口图，得到L/R参数梯度和B0余切；既有裸source精确伴随把B0信用传到q，随后重放过程网络及
两Meta的R/Z信用，全部完成后才统一optimizer更新。不得额外重做第二次昂贵source编译，也不得只训练L/R而detach视频路径。
仅冻结pre-Gemma输入与裸source量可跨更新缓存，已适配Z/KV/H不能跨参数版本缓存。
正常BF16/TF32及高效kernel差异按实际数值语义接受；小型LoRA坐标运算及累积保持FP32。

## 6. 工程核验、规模与资源

video.py拥有过程与q，factor.py拥有冻结编译、精确q伴随包装和共享出口，supervised.py拥有分生命期联合信用。
复用既有runtime、observer、data、checkpoint、materialization及唯一官方evaluator；旧v8只由冻结运行树执行，不加兼容模式。
诊断专用自由拟合入口与static bank准入在完成使命后从main退役；历史由`adc31a15`及冻结诊断树保存。

先用直接autograd oracle检查非identity L/R的完整参数／R/Z重放、无第二compile、identity、零过程作用与基旋转不变。
随后对真实最长task38/demo36（105个stride5帧）做两次完整联合更新和一次部署编译，检查source冻结、第二步各模块组的finite
有效信用、完整38-target／50-horizon／全视频及实际吞吐和峰值。Profile初始化不复用。
沿用frame_chunk16、FM microbatch16；只有实际资源或数值合同失败才作必要工程调整，不把低位差异当故障。
实际profile已通过：两步29.866／24.283s，allocated峰值37.049GiB、reserved37.770GiB，部署7.850s。
各共享模块组均获得有效finite信用，source无可训练参数；原件见新研究`profile/actual/results.json`，初始化不复用。

新研究根为`runs/analysis/process_pullback_learned_outlet_20260915/`，峰值额外预算32GiB，复用全部source、数据和环境。
每次launch同时live检查gpu01/gpu02，单节点最多6张实际提高吞吐的卡；DDP固定P2P禁用、NUMA绑定和deferred NCCL。
大root及正式训练前检查strg01独立data1 quota、实际个人用量和共享容量；正式train/eval来自clean pushed detached commit。
实际命令、拓扑、来源、quota与成本只记本研究launch contract，不在AGENTS追加动态信息。

## 7. 正式学习窗口与预登记读出

固定fresh900个logical updates，3,600教学条件／230,400queries；每100保存完整checkpoint，300/600/900各做
validation correct400和train96，0/300/600/900做既定独立动作诊断。与v1完整窗口匹配曝光，改变的主要变量是共享出口。
三段在完成相应闭环后exact-resume，保留同一world topology、optimizer、scheduler、sampler、rank RNG与schema。
实际最长视频profile通过后更新墙钟估计，预计四卡约4–5小时训练，评测另计。Profile和旧局部正例不能替代新模型学习。

Validation K1每task每轮50初态对应全部50合法视频各一次。沿用seed20260911及canonical video_schedule，在三个checkpoint
复用固定state/video/RNG映射。Train96为24task×states32–35，teacher46–49明确登记有限池。
Source train17/96和validation47/400只有完整执行及逐行RNG合同一致才复用，复用与新增rollout分别报告。

所有正式性能判断只用single-checkpoint完整400，报告per-task／suite、breadth、retained/gained/lost、churn、相邻重合，
以及task-cluster配对bootstrap（20,000次、seed20260915）。内部loss、norm或screen不选择模型；不使用checkpoint union或融合。
当前优先级仍为真实能力、迁移、保持及视频条件增量；>145/400为长期参照，本轮不另作硬门槛。

为完成本次授权的完整训练和测试，最终读出点在任何新分数之前固定为窗口末尾900，绝不在三个节点间挑峰值。
报告能力／相邻资格与固定终点的描述性评估为两个判断：未合格的终点不得称为selected合格模型。
训练及方法决定全部冻结后，补same-task-other400、wrong400、no-video/source400及最后shuffled/reversed各400；
真实双路frames先变换再完整重编译。所有臂保持同一逐行配对；它们不产生梯度、不改变训练／架构／选点。

最终Test只做固定test8的source400与terminal900 correct400。执行前写明具体checkpoint、method metadata及停止后续训练／
架构更改的freeze record，materializer与evaluator共同核验。Test全50视频／初态、零梯度、不选点、不反哺本轮设计。
禁止无freeze的Test读取、Test子集、未登记臂或根据Test追加实验。若有工程合同错误，只修复该合同并明确受影响证据。

## 8. 裁决和结束

有真实广泛获取、相邻保持、same-task换视频鲁棒性且冻结后的视频对照支持必要增量，才支持本轮完整方法假设。
若只改善train而held仍弱，说明共享获取／迁移未解决；若L/R后仍不提高闭环，降低该具体共享出口修正的支持，
不能拿free_AB46作为新Writer成功或继续追加预算的理由。固定终点视频对照或Test不得反过来驱动再设计。

本goal完成指本轮必要修改、验证、正式900窗口、全部登记评测与对话内原因分析完成，不要求制造正结果。
不自动回退v5.2、不无限续训或小扫。所有正负证据保留，结果直接在对话中说明，不新增用户报告。

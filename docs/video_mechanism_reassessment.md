# 正确视频机制复核：形成过程、证据缺口与迭代依据

日期：2026-09-11。本文服务Owner最新三步goal，当前完成架构形成过程审查，失效定位尚未完成。授权与执行只看[progress](../progress.md)，计划只看[task_plan](../task_plan.md)。95-task扩展不恢复。

## 1. 要解决的问题

正确视频指同任务、内部顺序正确的教学视频；错任务或乱序都是错误条件。同一候选必须在正常训练后满足correct高于冻结source，并分别高于wrong-task与shuffled，且差额不能由人为惩罚错误视频、削弱参照或选择偶然峰值制造。最终EMBER的>145/400、相邻保持、任务广度和独立验证要求保留。

这次最新授权允许开发wrong/shuffled诊断指导原因分析和架构修改，覆盖旧禁止反馈条款；开发结果不冒充最终独立验证。validation/test不产生梯度，Test继续封存。本文不把当前模型整体称为“完全不看视频”：已测模型有输入敏感性，缺失的是可信、有益的正确教学增量。

## 2. 科学精神与数学对象

科学动机是从他人演示的状态变化、接触后果与操作先后，理解一个可迁移的控制规律。教师动作隐藏，新执行初态不同，因此输出不能是按教师时钟播放的动作脚本。原件为`3a6f801d:docs/action_forecast_writer_expert_consultation.md`第24–58行，以及[第二轮推导§3](review_materials/20260908/expert_round2_evidence_review.md)。

给定语言与视频条件C=(L,V)，Writer生成每个执行目标的A/B：

\[
\Delta W_m(C)=B_m(C)A_m(C),\qquad
y_m=W_mx_m+\sum_r b_{mr}(C)\,a_{mr}(C)^\top x_m.
\]

rollout中A/B固定，执行激活x随机器人自身观测变化，所以固定参数可以改变“什么时候接近、抓取、搬运、释放”的状态条件规律。它没有证明rank16能表达任意程序，也没有补齐source缺失的可观测状态。task experts只支持部分实际容量。

语言确定目标和关注对象；视频提供实际操作及其后果；执行观测确定机器人目前该做什么。视频帮助有限模型推断未知控制规律，不要求它在理想无限模型中是唯一可能的信息来源。跨具身是动机，当前LIBERO本身不证明跨身体泛化。

## 3. 如何一步步得到目前的图

| 讨论阶段 | 当时的问题与推导 | 形成的选择及边界 |
| --- | --- | --- |
| 早期pooled/Action-Memory/RoPE | 公共adapter可有分数；内容敏感、顺序敏感和正确过程有益不是同一性质 | pooled119/wrong115；仅加时间位置不能证明过程学习。见旧咨询原件141–230行 |
| Forecast到v4 | 绝对未来时间对齐暗含“最新forecast更准、revision是有效纠错” | v4该前提随训练恶化，乱序反而更强；下游传递差异不能证明上游语义正确。见`3a6f801d:docs/action_forecast_writer_v4_root_cause.md`80–139、194–230行 |
| v5/v5.1/v5.2 | 把可用对象/关系语义与有向过程分开，让语义确定过程读取关注点 | Core不含视频时钟；Procedure有序、中心化后被Core-conditioned槽读取，再调制Core并生成因子；v5.2补逐task-token patch grounding。不是“中心化必有效”的定理 |
| privileged、G1/G2、G3 | 从局部有用更新、native容量、可读动态走向共享视频编译 | G1局部容量和G2动态各有正证据；它们没有共同证明共享Writer学会。反复的shared失败不能由oracle通过被排除 |
| 9月6–7日分层关系 | 保留动作知识参与视频理解，并区分video time、horizon和计算层 | 单probe；18×50响应；局部50×50软对应；内容与位移分布非线性解释；保留H直到learned read。依赖接通和完整轴不等于物理对应 |
| 9月7日第一轮审查 | 训练熟悉视频也弱，不能只归为held迁移；几何症状不能唯一定位decoder | 要求原件与行为对应，保留旧强结果；不把P/Q clone或末读出局部改善当成上游免检。见[原文](review_materials/20260907/expert_review_round1.md) |
| 9月8日第二轮证据修订 | 原件纠正batch/曝光混杂、rank指标、训练预算与收敛的混淆 | 提出同帧S直接图文访问及native D交叉候选；没有认定这两个接口是唯一根因。见[原文](review_materials/20260908/expert_round2_evidence_review.md) |
| 过程条件化视觉回读 | 同帧S并没有实现“先形成过程，再用过程核实画面” | 跨帧响应关系→过程query→分帧读取真实Z→修正完整H；新增长程组织，提出实际闭环信用。见[原文§6–8](review_materials/20260908/expert_process_visual_loop.md) |
| Owner选择末层、逐对核实 | 不保留18层轴；聚合前应知道哪一条关系由哪两端画面支持 | 读取action_out_proj实际输入；每帧对完整H query先读Z_u/Z_t，再聚合。输入层与输出target不必一一对应。见[原文](review_materials/20260908/expert_fm_rl_protocol.md) |
| Owner选择过去窗口、完整H query与交替 | 新的整条对应模式须共同决定视觉查询；重叠历史证据须彼此影响 | 过去四帧；H-query全H attention；短GRU按历史u排序；四组局部—长程，前三组逐H非线性回写。见[最终专家稿](review_materials/20260908/expert_past_interleaved.md) |
| Owner最终覆盖长程方向 | 最终选择过去单向 | 四组past+self；早帧不能利用后段证据回看早期画面。专家早期“未来证据修正过去”的论证不属于最终方法；不是实现遗漏。见[Owner裁决1–7](review_materials/20260908/README.md) |
| Owner覆盖训练阶段 | 联合训练范围与FM/RL同一步是两回事 | 先fresh端到端FM，平台后再独立共享RL。未做RL不能称为实现了原联合学习机制；也不能据此认定缺RL是视频失效原因 |
| 9月9–10日Horizon修改 | 首Compiler共同语言位移压缩槽间查询；之后出现获取与保持交互 | 去裸语言内容残差、逐帧contextual语言、最后关闭Compiler额外语言query。off有真实净收益，但正确视频必要性未随之成立。D绑定未修复迁移 |
| 9月11日首轮意见与Owner追问 | 专家先把功能保持与过程aux放前面，未充分解释v5.2正例 | Owner要求正面解释普通FM曾自然学出视频依赖；专家撤回前置保持/aux，转为消费接口与条件分配竞争。见[追问](review_materials/20260911/FOLLOWUP_PROMPT.md)、[修订](review_materials/20260911/expert_review_round2.md) |
| R/C/S实际实验 | R重分配条件；C恢复直接语义→条件化过程变化消费；S提供充分无序集合参照 | C/S仅100/200两个节点；C43→50，不能据此宣告平台或整个机制已否定。后续R400和off600只回答其各自曝光问题，不能替代C的观察 |

这些不是应按顺序重新训练的路线。专家文稿中的推荐默认、Owner裁决、实际实现、科学证据分别解释。

## 4. 最终Horizon的完整因果链及其数学承诺

### 4.1 动作响应只是待核实假设

\[
Z_t,KV_t=\mathrm{FrozenPrefix}(L,I_t),\quad
R_t=\mathrm{PreActionOut}\big(\mathrm{AE}_{\theta_0,\mu}(KV_t,\xi_0,s=1)\big),
\quad U^0_{t,h}=W_RR_{t,h}+e_h.
\]

R为最终归一化后50×1024 hidden，source完整计算仍运行；共享probe为50×32固定Gaussian。没有teacher state/action，也没有在teacher路径执行完整10步采样。因此h只有动作生成位置角色，不是已校准的教师未来动作。共用probe减少独立噪声，也可能产生公共horizon结构；高对角对应不能自动解释成实际过程。

代码核对：[native.py](../src/ember/writer/native.py)90–167行。真实Z包括patch与exact task tokens，所以“来自视频分支”仍可能主要承载语言/场景共同语义。

### 4.2 软对应新增了什么

当前帧t只读取过去四个实际u，真实gap为Δ=n_t−n_u。当前h对应过去g的名义关系为g−h−Δ≈0：

\[
C_{tu}^{a}(h,g)=F_a(LN U_{t,h})^\top F_a(LN U_{u,g})/\sqrt{32}
 +B_a(\Delta/5,(g-h-\Delta)/50).
\]

softmax包括空匹配；对应内容m、相对位移分布ρ与非空质量进入关系token。joint-gap只是可学习先验，不强迫斜带，不是物理标定；horizon time、video time与flow time不能混用。

### 4.3 “核实”是计算依赖，尚非正确语义

先让全部50条新关系通过一层完整H attention，共同形成Q，再分别读取Z_u/Z_t，联合当前状态、匹配内容、位移、gap与两端视觉差形成M。这样另一行新对应能影响本行看什么；先逐对核实保留证据归属。

代码确有这条依赖：[relation.py](../src/ember/writer/relation.py)63–104行。但Q/M还直接包含当前状态、语言和两端绝对内容；这些可用内容没有被要求必须由真实变化解释。视觉读取是一次learned attention，没有外部“接触已成立”真值认证。“核实”描述设计职责，不能直接作为实验结论。

### 4.4 三条有序轴各有用途

H-query沿h双向；GRU沿历史起点u从早到晚读取，消息区间[u,t]重叠，非相接动作片段；长程沿视频t过去单向。GRU可提供非交换的消息条件依赖，但不保证消除重复或理解事件。

每组临时H-read得到E_t，一层past+self temporal得到P_t；前三组用当前U与P共同决定非线性回写，下一组重读原始Z。它避免单Key attention或简单相同广播的退化；没有恢复未来证据。第四组P直接送Compiler，不做无用途回写。

代码对应[relation.py](../src/ember/writer/relation.py)106–146行、[horizon.py](../src/ember/writer/horizon.py)61–96、164–186行。完整U保留、组间无detach，但数学依赖不证明有益依赖。

### 4.5 参数生成与共享学习

off使用608个target/rank身份槽、两层Compiler读取P4。额外静态语言query已关闭，语言仍通过逐帧Z、R和local/H-read进入。Compiler读出含残差、bias和共同内容。native D为：

\[
a_{mr}=a^0_{mr}+D^A_{mr}\,GELU(U_Ac_{mr}),\quad
b_{mr}=D^B_{mr}\,GELU(U_Bc_{mr}).
\]

D按target/rank/side独立，跨task共享，末映射329,515,008参数。其动机是让native梯度有直接学习通道：固定h、单B列b=Dh时，SGD一阶δb=−η‖h‖²g；旧坐标MLP相应切片为−ηΦΦᵀg。它改变可学习通道和函数类，不能推出更强泛化。

对其它条件c′，同一D更新仍有δb(c′)=−ηg(c)〈h(c),h(c′)〉。独立rank不隔离task；相似代码不一定无用，也可能造成共享更新耦合。该式不是完整AdamW训练的精确因果归因。

代码：[attention.py](../src/ember/writer/attention.py)85–103行、[native_factor.py](../src/ember/writer/native_factor.py)。任何h相似或D大小都不能单独命名为根因。

### 4.6 正常训练的信用怎样到达

同task视频与独立episode action queries；源policy读取自己的观测、state、语言及noisy action，普通FM误差先对生成A/B求梯度，重放Writer，再重放Meta。训练只用正确顺序的视频。冻结Z/KV可缓存，更新中的R不可跨step缓存；R-leaf与上游Meta没有永久断梯度。

实际源码：[supervised.py](../src/ember/writer/supervised.py)27–72、[learning_data.py](../src/ember/writer/learning_data.py)93–164。首步B=0导致上游短暂零梯度是初始化代数，不能重演为SEOD式永久断梯度解释。

设X含执行观测、语言、noisy action和flow time。理想平方损失下视频的Bayes风险收益为E‖E[Y|X,V]−E[Y|X]‖²；若task内独立且语言确定task，该量可能为0。但实际有限共享生成器的未见任务规律尚未确定，视频仍可帮助推断。该推导说明没有自动保证，不说明优化为何选择某条路，也不能解释旧v5.2成功与当前失败的差别。

## 5. C相对off到底改变什么

C保留上述完整R→P4过程图，另从真实Z保留task-token轴读取patch、沿帧无时间集合读取为S。每槽先读S得到s，以s条件化读取P4，Value用P4减该视频时间均值，融合s/p再用原D生成完整LoRA。代码：[semantic.py](../src/ember/writer/semantic.py)、[horizon.py](../src/ember/writer/horizon.py)230–237行。

它检验的是“直接语义内容＋语义条件化变化消费”是否更容易由普通FM学习，不是强制动态唯一入口。S可独立支持任务；融合可以忽略p。

还有一个与旧v5.2的重要非等价：旧Procedure在逐帧interaction恒定时保持恒定，中心化后为零；当前Horizon的边界、gap、GRU时间输入、horizon身份和回写允许重复静态帧仍形成时变P4。因此C中心化只删除共同分量，不认证剩余是实际运动。不能把C叫作已经恢复了旧版全部变化语义。

S参照保留全部真实帧，独立逐帧完整H/Z处理，再无时钟集合读取；它能看到无序状态变化，不能称为纯单帧模型。C/S的encoder函数类与计算量仍不同，差异不唯一归于顺序。

## 6. 历史与当前证据的正确使用

| 证据 | 能支持什么 | 不能支持什么 |
| --- | --- | --- |
| 旧v5.2 step900：132/138/74/82/83，各400 | 普通正样本FM曾自然学出有益的正确任务视频与顺序依赖；correct对wrong/shuffle仅correct成功75/63，对方仅成功17/13 | 已满足当前全部资格；仅由一个模块造成；当前严格分池与旧采样完全相同 |
| 同结构v5.2 task-complete：120/109/107/111/124 | 完全相同Writer配置/模型源码也可学出明显不同依赖；优化和条件组织需进入解释 | 更多条件/全task batch必有效或必有害；batch是唯一原因 |
| v6-fast：143/135/125/128/129 | 语义聚合、视觉transition及heads的耦合变化有能力 | 每项独立贡献已识别；高能力与最强视频差额可以跨模型拼接 |
| off400固定train24面板：correct53、other60、same-suite wrong59、cross-suite wrong59、shuffled55，各96 | 当前最强候选未建立稳定正确动态净收益；输入仍改变成功集合 | 完全不看视频、纯语言解或纯task记忆已被证实；这是validation400率 |
| off首帧58、跨suite wrong丢4增10 | 对完整视频的替代在这个面板上很强 | 所有状态等效；训练过的static baseline；可部署选择或union |
| 真实P4/V有时间变化；首层off读取差异增大，闭环有局部净收益 | 特定查询组织有作用，P4不完全常量 | 这些变化表达操作意义；更大rank/query差异就能修复 |
| P/C/D端点与四次真实更新 | 共享映射变化可造成真实目标选择损害，且多个模块有贡献 | 唯一根因、整个D有害、冻结某块必修复；局部all实例可直接外推off |
| C100/200 correct43→50，S59→45，各400 | 该观察范围未取得absolute优势 | C已经平台、过程结构不可能、C已测得与off同样的视频失效 |

旧v5.2原始15臂中的本次复核范围为v5.2 old/task-complete、v6-fast三组共6000条rows；总数、task/state、env/policy根及共同执行噪声前缀吻合。来源为[20260907原件索引](review_materials/20260907/index.json)。v6 old这一行只沿用历史记录，不标成此次raw重算。

同结构v5.2比较同时改变：4→24 task/update，21→20 queries/condition，900→400 updates，75,600→192,000 queries，3,600→9,600条件，warmup/decay100/12000→17/400；旧每六更新轮转24task，不是仅训练四task。旧峰值LR3e−4，当前3e−5也是差异，但不能据此直接扫LR。旧teacher/action独立调度于同一50episode池，未严格排除碰撞；当前角色互斥。

当前off证据与配对见[findings§39–49](../findings.md)、[原始视频变换脚本](review_materials/20260911/methods/video_controls__panel.py.txt)。wrong保留目标语言，shuffled重排真实RGB后完整重算；原始索引仅作provenance，呈现位置单调，模型不能按原索引把视频还原。

## 7. 到这里真正知道什么

1. 最终源码实现了过去关系、完整H-query、视觉回读、GRU和回写依赖；没有发现能直接解释全部现象的永久断梯度或把视频暗中排序回去的问题。
2. 这套图同时保留了静态/共同语义的有效路径。仅有结构依赖不决定普通训练将哪种信息学成有益控制；历史v5.2证明“不强制”也可自然学出依赖。
3. 现有最强off有输入效应而无稳定正确视频净收益。几何分析和保持实验没有识别变化是否包含可用操作信息、这种信息是否被最终LoRA消费、以及是否被控制误差抵消。
4. C修改针对消费，但缺少对应视频对照且学习只观察到200。需要先补其机制状态，不能从off复制失效结论，也不能继续无根据扩大task覆盖。

**当前尚未找到唯一、已证实的失效原因。**下一项工作必须让不同解释产生不同可观察结果；不得把本页的候选数学解释写成根因已经找到。

## 8. 第二步首先需要补齐的区分

首先在训练任务的固定独立初态与视频上，用相同正常执行合同检查已有C100/C200、匹配无序S及source；off400既有九臂直接复用。比较correct、另一正确视频、错任务、真实乱序及预定首帧参照，结合相同独立动作queries，区分：

- C是否已出现视频增量但absolute/学习进度不足；
- C是否仍主要产生与顺序无关的一般条件收益；
- 正确视频在监督动作函数中已有收益，是否在闭环中被抵消。

这些是冻结开发诊断，无新模型更新，不以差额挑checkpoint；C后续是否需要延长，要结合以上机制事实和预登记完整学习轨迹决定。具体面板、命令、资源和作用路径干预另在结果前登记；本节不预判原因或选择新架构。

## 9. 冻结诊断中的首项结构事实

C200在沿用的八任务面板（0、7、14、16、20、25、34、35）上，重复真实首帧并保留原时间轴后，中心化P4的RMS仍为正常视频的0.692–0.935倍，中位数0.817。实际像素没有运动，编码器的时间、边界和历史状态仍能产生相当幅度的时变输出。因此C的中心化Value不能自动解释为真实视觉变化；它没有保留旧Procedure的静态零响应性质。

来源：`runs/analysis/video_mechanism_20260911/paths/step200/manifest.json`的`representation_trace`。这是表示性质，不是“81.7%的执行来自假动态”的归因，也没有单独证明失败根因。冻结S/P4替换和匹配动作/闭环正在检验这些表示是否被消费及其实际贡献；在结果完整前不据此启动新架构训练。

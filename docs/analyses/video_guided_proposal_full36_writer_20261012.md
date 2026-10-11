# 全36共同教师 Writer 首个完整闭环：RL9未超过强MT

旧 v5 完整链已实际退出0。唯一预注册 RL9 联合 checkpoint 的 correct strict paired400 为147/400，
同场景、环境与policy RNG的强MT为153/400；R145/G2/L8、churn10，breadth同为6/8。
本结果没有取得当前Owner要求的完整Writer明显超过强MT信号。它是一轮有效科学负结果，不是工程失败，
也不因超过历史145门槛而取得新的资格；没有相邻checkpoint或视频因果证据。
主讨论已据完整结果收口旧版本，不追加旧续训、相邻checkpoint或controls；该裁决不表示整条路线不可能。

本报告只记录旧链的已执行事实，不是第二份active design。当前已派发的normalized-flow/query16新批
仍以[现有设计§11](../designs/video_guided_proposal_writer_20261010.md#11-当前有界批gfield发送端归一化与query162026-10-12)
及其实际run contract为准，复用旧MT/E/S_G/MT H，并独立fresh；旧结果不改变新批范围或自动启动其它阶段。

## 原件与执行身份

ROOT：`/data1/user/ymdai/ember_runs/video_guided_proposal_writer_20261011/full36_common_teacher`。
正式原件为`validation/RL_00000009/correct/results.json`、`formal_panel.json`、106份动态queue完成记录，
以及各condition的`path.json/query.json/complete.json`。
CPU复算保存于`ROOT/analysis/whole_completion_20261012/{consume.py,readout.json,paired_rows.jsonl}`；
全36训练消费脚本与统计保存于同目录`training_readout/{read_originals.py,statistics.json}`。
主讨论独立科学消费的五份原件保存于
`/data1/user/ymdai/ember_runs/video_guided_proposal_writer_20261011/full36_common_teacher/analysis/whole_completion_20261012/main_scientific_readout`：
`whole_result_decision.md`、`whole_result_reasoning.py/json`及`choice_conditioning_analysis.py/json`。
这些是旧v5实际标签与已保存概率的CPU复算，未修改formal raw rows；原脚本和JSON内的临时来源路径保留原样，
上述稳定副本作为持久证据，不需要再次运行。
没有加载模型大张量、raw H或held离线标签，没有新增GPU、episode、训练或probe。

资格权重为`ROOT/checkpoints/RL/update_00000009`，schema为
`ember_full36_student_joint_conditional_continue_writer_v5`。G216、首学生、warm36的来源为db72ee91；
联合训练因EGL工程恢复跨db72ee91、5c6bf6bb至1b44903d，原已完成更新保留，末次训练与全部formal读取为
1b44903d775876d2a944497984738077f7c60cbb的clean detached。原失败与全部成本保留。
当前canonical v6/query16实现没有用于解释或重读旧训练schema。

所有81个登记窗口job、358个当前最新job收据均退出0；362个job尝试中4个失败已恢复。
唯一旧退出事件/投递已经消费，不重做G216/学生/warm/联合更新，不设置新的旧链等待者。

## 完整配对结果与保持范围

| task | suite | MT/50 | Writer/50 | retained | gained | lost |
|---|---|---:|---:|---:|---:|---:|
| 3 | Spatial | 41 | 38 | 38 | 0 | 3 |
| 6 | Spatial | 7 | 5 | 5 | 0 | 2 |
| 11 | Object | 36 | 36 | 36 | 0 | 0 |
| 16 | Object | 10 | 10 | 9 | 1 | 1 |
| 23 | Goal | 0 | 0 | 0 | 0 | 0 |
| 26 | Goal | 35 | 35 | 34 | 1 | 1 |
| 31 | Long | 24 | 23 | 23 | 0 | 1 |
| 39 | Long | 0 | 0 | 0 | 0 | 0 |

suite依S/O/G/L为MT48/46/35/24，Writer43/46/35/23；没有suite净增。
MT成功集合保留145/153，Jaccard145/155=0.935484；净少6、仅10行交换，不能称大范围能力丢失。
两个原零成功任务仍为零；新增2行分别伴随同task一行丢失，没有task净增。
只有一个checkpoint与8个validation任务；小样本得失不证明统计等价或普遍不可能。

400个task-state-video条件均唯一，每task50条视频各一次，沿预注册映射；没有选择高点、融合或union。
逐行环境/policy RNG、实际policy noise共同前缀及sealed scene一致。最终初态不在登记实践池或实际修订行；
执行consumer另已直接检查实际H中没有最终初态。最终G执行不读教学，固定一套LoRA，无评估梯度。
281个最终MT condition引用同参数旧MT原行，其余119个实际评估G；历史MT400成本没有重计。

## 生成与选择的最早缺口

共同教师控制有真实能力增量：独立报告E453/576对MT350，R327/G126/L23，27/36任务选非MT。
这是train/non-held目标供给证据，不是held教师泛化或Writer证据；9个E=MT仍只提供回到MT的端点。
不能因此认定教师已经完善，但现有目标不是只有无增益的微弱参照。

G216从MT及真实非空bootstrap H开始，每task24次，共864次CFM事件，source无梯度。
按已保存S_G/编辑均方和实际有效坐标维数，task等权零场期望loss为2；最后三个全36块为
1.999106、2.000087、1.999566。首个实际G学生的配对query4为MT83/G83/144、R74/G9/L9；
24个condition奖励相同，6个G较好、6个较差，按每condition固定一套的经验oracle为89。
因此最早控制传递不足出现在Actor共同训练前，不能只归因最终选错。

既有G216真实输运原件进一步显示输出几乎保留xi、field修正小，以及六层消息累计后局部信号受压；
其稳定原件位于新ROOT
`/data1/user/ymdai/ember_runs/video_guided_proposal_writer_20261012/normalized_flow_query16/analysis/previous_G216`。
这些与当前学习/闭环证据相互支持，但没有确认唯一根因、视频或H的因果作用，也没有确认发送端归一化修复有效。

warm36真实消费同一36个首学生的两个prefix各4次，没有额外R5。
只有12个condition有MT/G奖励对比，24个精确同分；单MT prefix不能给final排序梯度。
全局gradient norm均值0.000293、最大0.000577，更新12/19/27/29/33为零；四轮loss近−0.57639。
这说明真实更新与可区分信用规模有限，不把非零梯度直接称为已学会。

联合9批覆盖全36各一次真实R5：180次实践（G162/MT18），9次theta及81次psi更新，末G cursor297、
bank version9、216个实际prefix。query有712个逻辑独立行，其中144条兼容MT引用、568条直接行；
没有把query回填H。末次行为采样86/144对MT83、R79/G7/L4来自跨不同behavior checkpoint，不能作RL9资格。
最后36个训练final分布相对均匀TV均值0.000439，真实query上的加权期望86.582与均匀86.583基本相同；
parent/practice全分布TV均值0.0640/0.0774，故不能把三个head一并称为完全均匀。
联合CFM均值2.94475，而MT/首学生/真实RL父均值1.96481/2.96267/3.66018；父参数使速度标签尺度改变，
不能与初始2直接比较后宣称训练退步。

主讨论按全部324条实际P/E/xi重建速度标签`delta-xi`，给出同样本的零field MSE基线：
实际loss均值2.9447506456，配对零场2.9443830616，相对改善为−0.012484%。
九个完整36-condition块均未优于各自的零场均值，按MT/首学生/真实RL父分组也都几乎零场；
MT父组相对改善仅0.006560%，其余两组为−0.018364%/−0.015492%。95/324条单事件优于零场，
不能把局部改善当成总体学习收益。复算的是每次已记录更新前的实际训练样本，没有新增G297 forward。
因此联合阶段仍未打破近零场的判断有配对依据，不再仅依赖跨阶段raw loss比较。
它与初始G的xi保留、消息累计及首学生无净增相符，加强了生成学习失效的解释；
仍不确认唯一根因、held迁移或视频/H因果作用。

主讨论进一步将Actor的角色偏好与具体候选排序分开。下表是保存概率相对均匀分布的平均TV；
parent条件统计去掉STOP后重新归一化，practice条件统计去掉已实践父后只比较四个新G。
parent条件行只纳入至少两个候选，final只统计最后实际选择且支持数大于1的分布。

| 已保存路径 | parent全分布 | parent给定continue | practice全分布 | practice给定新G | 最后final |
|---|---:|---:|---:|---:|---:|
| 训练36 | 0.064005（180） | 0.000290（133） | 0.077364（180） | 0.000361（180） | 0.000439（36） |
| 正式400 | 0.114745（1,040） | 0.000269（567） | 0.075934（733） | 0.000321（733） | 0.000528（213） |

括号内为对应分布数。正式practice给已实践父的平均概率0.275934，均匀参照为0.2；
显著偏移主要体现在STOP/继续、已实践父/新G等角色间，同角色候选的排序仍很弱。
角色偏好不证明已学会能力排序，小TV也不证明全部context/native响应未被使用；
候选本身弱、query信用粗、表示或训练量不足仍未唯一分解。训练路径跨行为checkpoint，不是新资格臂。

正式400最终281个选MT、119个选G；119这个由模型选择的子集为MT43→Writer37，R35/G2/L8，
全部净差来自此处。最终实际支持有187个singleton MT，其余213个最后final输出的TV均值0.000528，
最大0.001343。较高比例的保留不能单独证明有效选择策略；基底保留与弱final排序共同解释了结果接近MT，
但不是上游生成无效与下游信用不足之间的唯一因果分解。

实际final通过softmax后的multinomial采样，不能把微小logit差当成argmax选择收益。
对固定最终候选集及各候选成功概率`q∈[0,1]`，相对均匀分布u的重加权满足
`|Σ(p-u)q|≤TV(p,u)`。213个非singleton分布的TV求和约0.11238，另外187个singleton贡献为0；
故只替换最后一次重加权，期望成功数相对均匀最多变化约0.11238/400。
这不是一次实际采样成绩的上界，也不约束改变上游STOP、实践、候选生成或H之后的整个适应过程。

本轮强化两个层面的负证据：G尚未建立明显的目标控制传递，final排序也未从有限信用中获得明显收益。
完整共同循环已经执行，其接通、更多P/H及非零梯度不能替代闭环效果。
继承已知生成失败解释后，已登记新批先检验单一发送端归一化；query16另修正廉价且重要的信用精度，
不同时扫描时间网络/endpoint loss或替换Teacher/Actor。主讨论消费新证据后决定后续共同训练。
新批G216学习没有消费query16，生成架构仍为单变量对照；MT/G/E均在新16-state配对面板读取，
不能把旧query4与新query16的raw分数差归因于归一化。后续判断须看到目标方向的实际控制传递及闭环得失，
仅有去噪、输入敏感性或E=P解析子问题改善不足以证明视频操作知识编译；不由局部改善自动接续共同训练。

## 真实投入与恢复能力

formal适应有400个实际MT bootstrap与733次实际修订实践，共1,133个episode、334,511环境步；
2,932次实际候选生成。修订实践G613/MT120；round0/1/2/3/4/5分别152/86/35/24/10/93。
307个路径实际STOP，93个到轮数/外部边界；不是强制全跑R5，也不是zero-interaction。
最终新增119个query episode、35,339步，总新增369,850步；281个MT final引用及400条对照不重计旧费用。

本ROOT登记allocation GPUh为57.533346，其中teacher train21.153673、selection8.448224、独立report2.982325、
G2163.167964、首学生1.766303、warm0.510231、联合含失败6.585964、strict40012.153929、profile/恢复小分析0.764733。
六个失败尝试（4个job、2个profile）共1.278765 GPUh，保留在总数内。
继承历史carry62.121703，总登记119.655049；独立G216诊断0.053786另列，不在本ROOT ledger中，不双计其内段计时。
这些是launch-inclusive的job分配device-hours；与新批共驻时不能把两root相加称为去重物理GPU小时。
本ROOT首个job UTC07:21:24至末job19:31:17约12.165h，最后恢复coordinator为3.486h；不同范围不能混称总wall。

全部关键权重、formal raw rows、CP与优化器/RNG/sampler、指定bootstrap/当前依赖仍保留；没有删除旧训练状态。
当前新批复用旧资产，旧v5冻结源码也是重读消费者，故不退休该formal tree。此前已集成开发树的清理不损失这些恢复能力。
无Teacher扩展、额外旧probe、adjacent checkpoint或视频controls随本结果自动启动。

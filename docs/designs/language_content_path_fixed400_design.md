# 内容路径阴性后：固定C0的完整条件覆盖

2026-09-26。原736条已结束，独立复核见findings§153及原study的`coordination/main_recheck.json`。
这是新登记的有限补评，不是恢复Cplus、训练新架构或按80条筛选checkpoint。执行状态只看progress。
机器合同：`configs/language_content_path_fixed400_v1/experiment_spec.json`。

## 1. 为什么先补这个，而不再改架构

原假设预测加入直接语言内容通路后减少Source能力损失并保住C0新增。本批Cplus correct/other为16/19，
C0为24/30（各80），Source成功丢失从C0的5增加到Cplus的10，C0新增13仅保住8；预测未兑现。
关闭该零初始化共享标量的学习分支，不扫描初值、尺度、层位、seed或延长训练。
梯度非零只说明路径参加学习；实际`||a q||/||A||`中位约0.20%，不将本结果推成所有直接内容路径无效。

同时，新C0相对B630的额外保持损失没有在这80条中复现：二者均丢Source成功5条，新增13/10。
旧同80子集B/C为24/22，本批B/C0为21/24；B是冻结policy复评，C0是world2重新训练，二者不是同类变化。
旧全400的C−B为−19，主要由Goal21的−32抵消其余七任务的+13；新Goal21为C0 7、B8（各10）。
不能把旧一次训练的模块缺口视作已稳定复现，也不能凭本批+3/80宣布视频已超过语言。

固定任务等权，记成功差为d_ts=Y_C(t,s)−Y_B(t,s)。当前测量
`Delta_80=3/80`，完整固定集合为`Delta_400=(3+sum_(s=10..49,t) d_ts)/400`。
另一正确视频当前分子为9，两个描述性task-cluster区间均包含0。
未测320状态可以改变净差方向以及保持/新增分解；这些闭环不存在于任何当前C0原件。
对同批Source有恒等式`S_C−S_B=(G_C−G_B)−(L_C−L_B)`，须同时报告两项，不能只用总分判断修复。
这些是明确测量对象及不确定性，不是性能根因定理。

补评会改变下一项科学决策：

- C0在两种正确视频下均取得相对B的完整集合优势且保持未更差：旧结构存在可用候选证据，先区分视频内容贡献与训练差异，
  不再把“原结构必然输给语言”当架构修改的理由；仍非视频因果或稳定性证明。
- 完整集合重现C0相对B的缺口：将实际缺口按新增、保持和任务列明，再结合旧条件交换/私有拟合正例选择可区分的干预。
- 差异小、方向随正确视频改变或区间宽：登记未识别，不按分数选视频/选点，不自动补seed或更多节点。

旧B/Source剩余320行可形成便宜的历史参照，但新旧B在已测80行的净差−3恰与本批C0−B的+3同量级。
因此本批重新执行这两个冻结参照，保留跨运行差异，避免用旧参照冒充新配对。新评测仍有正常数值/环境变异。
相比新架构训练，这里不新增任何优化；相比只补Goal21，这里保留全部八任务，避免重新陷入单任务诊断。

## 2. 冻结输入与1280条新增闭环

实际执行者Sol：`01a0cd90-ebb7-77a1-a20b-a858825d2f66`；主讨论：`01a0cd94-65da-7b22-8ca9-7ba35f454632`。
只有C0@630、原B630和Source1000；Cplus所有结果保留，不新增Cplus计算，不训练/恢复任何模型。
C0训练提交F=`dca1b5500ac0f912d56cc1c76382004457e807a4`，路径见机器合同；全部权重冻结。
B只读复用43d801b1的原对应bank，Source不使用adapter。无新held expert action、官方Validation/Test或梯度。

四个面板：Source correct、B630 correct、C0 correct、C0 same-task-other；每格held8×states10..49=320，共1280。
held8固定`0,1,14,15,20,21,36,38`；与上一批states0..9严格不重叠。
新C0 correct/other各320条件完整Writer生成，共640新bank；B复用320既有条件，不重物化。
原736条中的这四格共320行原样引用，构成四格各400、合计1600条完整矩阵；原Cplus/seen等结果不删除、不混入此矩阵。
不另设GPU smoke或pilot矩阵；相同科学计算的真实接口已在上一批通过。
使用正常动态队列执行登记行，每面板退出后签收接口/配对/trace，不按成功分数决定是否继续。

teacher映射完整复用seed20260911、50视频池、原canonical不放回排列与other offset17。
补齐后C0 correct和other每任务各覆盖全部50视频一次，逐state不同；不挑视频，不平均LoRA。
环境/策略root seed7；noise按原公式逐条推导，比较共同replan前缀；不要求成功即停后的seed列表等长。
原canonical render256/model224、双相机旋转、state8/action7、10-flow/执行前5、settling10及horizon不变。

全部新增1280条保存compact、实际动作、T+1对象/fixture/arena-region/EEF/夹爪及BDDL谓词。
新增full固定为每面板global0/14/21/38、state10，共16条双相机；不按成绩换case。
复用原被动采集及arena-region修复；不用增加function probe、视频控制或环境smoke。

## 3. 实现边界与来源

从最新main隔离开发。复用现有trainer checkpoint loader、materialization、canonical evaluator及passive capture。
允许的实现工作仅是为上述额外状态/输出根注册范围，精确绑定冻结C0/B/Source身份，并提供有界调度和只读分析。
不复制Writer或evaluator；不得改原训练spec、原736行或F冻结树，不因新Git提交重跑已有有效行。

新增物化及闭环来自统一clean pushed detached提交E。E相对F的科学GPU计算必须不变：
只修改配置/范围准入/来源验证/调度/只读分析。C0训练F、原320行F、新640bank和1280行E、B原bank43d须逐行登记。
这是事前明确的阶段来源，不能伪称全部同提交，也不要求无意义的重复训练/物化来制造单提交。
若发现必须改变模型输入、数值、训练或rollout行为，保存证据并交主讨论裁决；不要在本补评中夹带修复。
同一既定阶段内仅CPU调度/验收窄修可记录新clean提交和diff接续，保留有效行；禁止热改冻结树。

CPU验收应覆盖原80范围不变、新320范围不重叠、50视频完整覆盖、C0冻结来源与B复用身份、全行采集。
既有真实模型接口已在原批通过；不为新selector增加GPU smoke或模型forward。
同一批工程异常保留原件，不重复有效episode；不为验收方便增加未登记的状态或模型条件。

## 4. 分析与停止条件

先验收退出/唯一行/映射/trace，再读取成绩。分别保存新增1280和合并1600的原始索引。
明确本次范围是在看过原80后登记；新增320/task集合的结果单列，合并400不冒称全部前瞻盲测。
未使用screen挑checkpoint，C0@630始终固定；本diagnostic-held8也不冒称官方Validation/Test。

主要比较为C0 correct−B和C0 other−B；报告两个集合差，不选更高视频作为模型分数。
其余为C0 correct/other−Source、other−correct，及两C条件相对Source的R/G/L。
列全任务/suite/breadth、R/G/L/churn/Jaccard，Goal21和其余七任务补充分解，同时保留完整八任务主比较。
所有四格共享task-cluster bootstrap draws，20000次、seed20260926；8个task等权，
分别给新增states10..49和完整0..49的点值与95%描述性区间。它不包含training-seed不确定性，
不能把包含0称等效，不能用重复补测追逐显著性。保留原80及旧四臂全400，仅作带身份的历史比较。
完整400也没有建立相邻稳定性、强MT-BC最终优势或视频动态的因果必要性，不选canonical方法或宣称修复。

固定矩阵完成即主动Queue主讨论并停止新增计算；不自动追加节点、seed、wrong/shuffle/reverse、训练或新架构。
main独立核验后再按第1节决定下一项。OOM/nonfinite、信息墙/来源/配对错误停止受影响面板并保留原件。

## 5. 成本与接续

研究根：`/data0/user/ymdai/ember_runs/language_content_path_fixed400_20260926`。
依据刚完成的真实进程回执：Source/B各约15.5/16.2 GPU秒/行，C0 correct/other约19.9/19.55，
1280条合计约6.33 GPU-hours，640条件物化约0.3 GPU-hours，启动和正常波动另计。
新预算为**8 GPU-hours**，包括加载、物化、评测、失败及所有GPU工程进程；不是借用原批18小时余量。
原批已计时15.35 GPU-hours，工程未端到端计时，总量未知；1小时规划预留不能当实测上界。
新批每个GPU进程必须从启动至退出按物理卡数记账，不用kernel计时或预留值补未知时长。
预计4卡时约1.7–2小时墙钟，工程配置/资源等待另计；不是完成时限保证。

data0新增峰值≤6GiB（约3.1GiB LoRA，其余trace/图片/索引留余量），新开发＋formal代码data1≤768MiB。
复用全部source/data/B bank/C0 checkpoint和原320行，不复制大资产。
建root前由Sol在strg01查独立quota与共享容量；每launch查双节点live身份、余量和项目总占用。
本批最多6物理卡、单节点最多6；按真正吞吐选择1–6 persistent workers，不等待凑卡、不跨节点DDP。
若预计剩余矩阵将越过预算，先停止新增panel并向主讨论报告实际账目；不静默截断登记行或自行扩预算。
正常运行只等退出事件；完成后统一验收。原授权覆盖该明确补评，工程/资源检查通过后直接接续，不要求Owner再次批准。

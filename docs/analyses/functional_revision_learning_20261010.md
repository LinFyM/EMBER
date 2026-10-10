# 功能修订首批真实学习：360更新、72刷新、两个train48与固定400完整消费

固定φ360的correct结果为**143/400**，对强MT153保留127、获得16、丢失26；对T161保留109、获得34、丢失52。
原固定train面板从φ180的29/48降至φ360的25/48，MT为27/48。当前FM+Dkeep组织尚未形成稳定净修订。
能量末层确有更新，教学Reader图像投影更新很弱，实际完整LoRA修订极小；已有前5函数记录支持少量变化，
不能把接近MT的分数、非零梯度或非零参数变化写成理解了教学与实践。相对历史130的净增13保留，但不是超过强参照。

本报告只消费已经完成的原件，不增加GPU forward、梯度、环境steps或checkpoint选择。
科学解释与后继预算由新主讨论01a12385-4cfc-7ec2-9017-ef2b242243cf承担；
实验接任者01a12384-17a9-7382-aeec-cee89479428f完成工程、配对、费用、资源、Git和一次整批交付。
前任实验01a11a0a-fd39-74b1-83ba-001ede5330df的已完成计算全部继承；旧主讨论01a11b05-3460-7eb0-aca8-177d6d86ef48退役，不再唤醒。

## 1. 合同、执行身份与恢复范围

唯一合同为[功能修订设计§6](../designs/functional_revision_compiler_20261010.md#6-首批正式学习fresh-fm360真实分布刷新与一个固定400节点)，
事前判断见[实施报告§7](functional_revision_compiler_20261010.md#7-main判断信用接通以后首先检验真实学习而非扩大梯度数字)。
ROOT=`/data1/user/ymdai/ember_runs/functional_revision_learning_20261010`，下文原件路径均相对此ROOT。

共享Compiler及其AdamW fresh，lr3e−5、eps1e−8、betas(.9,.999)、WD0、clip1、constant；逻辑batch4、36任务等权。
固定MT因子RMS²的P、M16、原生十步、完整38-target/rank128、正常BF16与FM+Dkeep保持原合同。
teacher只有exact language及内部有序的双相机RGB，stride5；自身合法实践与反馈可读，φ在condition内冻结，
condition内不准备动作轨迹再运行拟合optimizer，最终每个condition只部署一套固定完整LoRA。
实践最多1024环境steps（含settling），success或预算决定停止与实际修订次数；评估状态不参与该condition实践。
没有PG更新、Test、held视频controls、额外400或超过360的更新；旧135停止线未恢复。

零更新物理profile来源clean pushed detached `a8212333`，共享学习、φ180刷新和两个train读出来源`6b2b4ffd`。
正式读出首次也使用`6b2b4ffd`；其磁盘统计碰到cache原子重命名的ENOENT后退出1，非模型故障或科学停止。
窄修复后的读取/恢复来源clean pushed detached `8d7cab05`，模型计算、信息墙、loss、选择和评测口径未改变。
首次formal已发布13行、80个claimed条件及307个pending条件保持；恢复只补未发布的387行，
保存的适应/有效final可复用，没有重跑360学习、72刷新或两个48。原失败、SIGTERM取消、费用与代码身份全部保留。
最终`batch_exit_code.json`退出0、`whole_compute_complete=true`；它最初的`scientific_complete=false`表示当时尚未消费，
接任消费、集成和交接完成状态另见`scientific_completion.json`及`main_delivery_receipt.json`。

## 2. 学习与实际曝光

有效更新为360，全部360行已发布，重复/废弃更新行0；共1440个编译事件、40320个跨episode FM query。
每个task每阶段20次事件呈现，两阶段共40次呈现；这不是40条独立新经验。

| 阶段 | 真实条件池 | 事件 | 实际非MT incoming | 非MT覆盖 |
| --- | --- | ---: | ---: | ---: |
| 1–180 | 原216条件、362 endpoint（314 MT/48非MT） | 720 | 160 | 16/36任务 |
| 181–360 | φ180实际新采72条件，每task两条新teacher | 720 | 180 | 18/36任务 |

task等权后仅在真实可用层内分配；缺非MT层的task全用MT，没有虚构递推或重权其它task。
新teacher排除了原六条及原train48教学，train实践排除32–34；显式候选和原面板冻结在执行前。
完整task/层/condition/endpoint/query曝光见`analysis/learning_events.json`及`analysis/learning_summary.json`，
曲线见`analysis/learning_curve.csv`；sampler与各rank cursor/RNG由完整checkpoint保留。
有限递推覆盖依然是限制，但后续train未获净增，不能把不足仅归因于held迁移。

CPU只读取四个已用于前轮判断的投影，与实际固定seed20261010的fresh模型比较，未加载source或做新forward：

| 投影 | φ360相对fresh的权重RMS变化 | 该权重RMS |
| --- | ---: | ---: |
| `energy.4.weight`（初始零头） | 1.27958e−3 | 1.27958e−3 |
| `reader.image.weight` | 8.96101e−8 | 1.27735e−2 |
| `encoder.image.weight` | 9.05252e−7 | 1.27382e−2 |
| `action_projection.weight` | 9.82693e−6 | 9.66137e−2 |

能量末层RMS在90/180/270/360为3.02677e−4/5.44726e−4/9.08145e−4/1.27958e−3，确有持续数值学习。
Reader图像投影相对其权重RMS只改变约7.02e−6；这支持上游学习很弱，
不等于整个Reader完全未更新，更不证明视频语义已被学会。完整四点变化见`analysis/selected_projection_learning.json`。

## 3. 能量、真实写入及前5函数：接通以后仍有多小

教学/语言经有序Reader，自身观测、实际动作和后果经经验encoder形成条件上下文；当前LoRA在自身支撑上的
完整十步响应F进入共享能量C，产生q=∂F C，实际写入`δΛ=−P J_Eᵀq/M`。
外层跨episode FM及成功自身函数keep把参数信用v经`−J_E(Pv)/M`传回准则、视频与经验路径。
P只定因子数值单位，不把真实功能响应归一化；新状态效应仍依赖`δF_new≈−J_new P J_Eᵀq/M`的方向与幅度。
这些是实际算子关系，不是已经学会或迁移的证据。

| 已有9-update窗口 | q RMS均值 | 实际δLoRA/MT RMS均值 | paired FM均值（out−in） | Reader梯度norm均值 |
| --- | ---: | ---: | ---: | ---: |
| 1–9 | 3.56946e−6 | 4.92363e−11 | 0 | 3.61330e−11 |
| 172–180 | 7.04987e−5 | 1.85381e−9 | 约0 | 1.56730e−9 |
| 352–360 | 1.47627e−4 | 7.50844e−9 | +3.98310e−7 | 3.42610e−9 |

paired FM是相同实际query的out−in，正值为更差；不能比较不同采样窗口的raw FM后宣称学习带来下降。
头部增长与q增大没有产生有量级的参数修订，后期paired FM也未给出明确净改善。

φ180刷新132次实际编辑的单次δ/MT RMS均值2.61136e−9、最大1.92885e−8；
72个最终LoRA相对MT均值4.44085e−9、最大5.24814e−8。
正式400条件共881次实际编辑，单次均值7.97514e−9、最大2.48049e−7；
最终相对MT均值1.60177e−8、最大2.47964e−7。
实际FP32因子归约没有零编辑或完全identity的最终condition；这是落到真实张量的变化，
但“非MT”标签不能替代有用修订。统计见`analysis/formal_subgroups_and_parameter_effects.json`。

已有Dkeep消费者对同一自身成功观测/flow noise的完整十步前5×7归一化动作算MSE：
1–9的24个有success支撑事件全部0；172–180有25个支撑事件，2个非零，均值3.58552e−9；
352–360有26个支撑事件，21个非零，均值2.04257e−7、中位1.85729e−7，均方根约4.52e−4。
这是有限train成功支撑上的实际函数变化，数值没有包含Dkeep梯度权重，未获益也不等于保持问题已解决。
见`analysis/task_coverage_uncertainty_and_functions.json`。

没有held同观测incoming/outgoing前5函数对照，也没有本轮视频/经验反事实。
BF16、batch及执行核带来的正常微小差异按合同接受；不以成功翻转倒推出极小修订已经有效，
也不追加低位或GPU探针。143与MT的42条翻转目前不能唯一归因于教学、经验、修订方向或普通执行差异。

## 4. 原train48：后半学习没有稳定保持前半小幅收益

| 节点/参照 | 成功 | retained | gained | lost | churn | 成功集合Jaccard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| φ180 对 MT27 | 29/48 | 25 | 4 | 2 | 6 | .80645 |
| φ360 对 MT27 | 25/48 | 24 | 1 | 3 | 4 | .85714 |
| φ360 对 φ18029 | 25/48 | 24 | 1 | 5 | 6 | .80000 |

breadth从φ180的7/8到φ360的6/8。Long2为4→2、Goal0为2→0、Object3为5→4，Object2为1→2；
Spatial0=6、Spatial1=0、Goal1=6、Long4=5不变（每task6行）。
suite按Spatial/Object/Goal/Long为180的6/6/8/9及360的6/6/6/7（各12行）。
损失存在于学习任务面板，不能由held域差异包办解释；180和360的实践condition独立，
这是按同teacher/最终state/scene/RNG配对的完整算法读出，不是固定同一E的参数反事实。
没有由29/48选择180或补它的formal400，两个train读出只作预注册机制与相邻诊断。

## 5. 固定φ360 strict paired400与强参照、完整历史

400个不同task-video条件，8个task各50条合法teacher恰好各一次；原state-video/scene/env-policy RNG逐行复用。
raw shards、queue完成状态、聚合成功键和所有reference比较经当前消费者重新核对；没有union、fusion或重择checkpoint。
逐行配对成功键、gained/lost及任务/suite统计完整保留在`evaluation/formal360/results.json`与`shards/`。
`analysis/formal360_paired_rows.jsonl`另给400条逐行配对索引、原始路径和各臂success/steps；
完整原始scene/RNG/action执行metadata仍由各reference的canonical results及本轮shards保存，不另复制。

| 每task50行 | MT153 | T161 | 旧135 | 旧φ180127 | 旧φ360130 | 本轮143 | 对MT R/G/L |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Spatial3 | 41 | 45 | 38 | 39 | 36 | 39 | 36/3/5 |
| Spatial6 | 7 | 6 | 2 | 0 | 1 | 5 | 4/1/3 |
| Object1 | 36 | 45 | 29 | 29 | 26 | 34 | 31/3/5 |
| Object6 | 10 | 5 | 14 | 10 | 7 | 10 | 8/2/2 |
| Goal3 | 0 | 0 | 1 | 0 | 0 | 1 | 0/1/0 |
| Goal6 | 35 | 36 | 32 | 34 | 33 | 35 | 33/2/2 |
| Long1 | 24 | 24 | 19 | 15 | 27 | 19 | 15/4/9 |
| Long9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0/0/0 |

suite为Spatial44/Object44/Goal36/Long19（各100），breadth7/8；Goal3一条成功保留，Long9仍0。
相对历史130的改善主要回到更接近MT的Spatial/Object水平，旧Long1=27的正例未保持。

| paired参照 | R/G/L | churn | 成功集合Jaccard |
| --- | --- | ---: | ---: |
| MT153 | 127/16/26 | 42（10.5%） | .75148 |
| T161 | 109/34/52 | 86（21.5%） | .55897 |
| experience135 | 106/37/29 | 66 | .61628 |
| parameter180127 | 105/38/22 | 60 | .63636 |
| parameter360130 | 103/40/27 | 67 | .60588 |

配对开发面板的描述性不确定性：对MT净−2.5pp，普通配对95%区间约[−5.67,+.67]pp，exact McNemar p=.164；
对历史130净+3.25pp，区间约[−.75,+7.25]pp、p=.142。未校正反复开发使用，不能称显著、等价或独立确认。
只有一个held400节点，没有相邻held稳定、selected checkpoint资格或视频因果结论；测试集仍封存。

正式实践分组为：success242条件，MT138→本轮131（R117/G14/L21）；budget-stop158，MT15→12（R10/G2/L5）。
J1=169条件，MT106→101（R90/G11/L16）；J≥2=231，MT47→42（R37/G5/L10）。
损失同时出现于成功/耗尽、单次/多次修订，不能仅由“成功后还改了”或递推层缺失解释全部现象。
分组由本轮实际实践结果产生，只作关联描述，不是随机因果干预；原件见`analysis/formal_subgroups_and_parameter_effects.json`。

## 6. 真实实践、教学读取与完整费用

| 完成范围 | 实践成功/预算停止 | 实际编辑/重置 | 初读/重读/全片读取 | teacher frames | 实践steps | 最终steps |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| φ180刷新72 | 58/14 | 132 | 72/60/132 | 4326 | 30932 | 无最终评测 |
| train180的16条件 | 12/4 | 31 | 16/15/31 | 976 | 7392 | 11046 |
| train360的16条件 | 13/3 | 31 | 16/15/31 | 978 | 7594 | 12011 |
| formal360的400条件 | 242/158 | 881 | 400/481/881 | 32333 | 236183 | 117125 |

完整已保存condition合计1075次编译全片读取、504初读/571重读、38613个teacher frames、422283环境steps。
formal已保存条件共353308实践+final steps；失败实践190732、占实践80.75%，tail39962已包含其中，不再相加。
formal成功实践组74391steps，预算组161792；J1组31417，J≥2组204766。
881次编辑累计7006.47秒；condition wall之和是并行工作总量，不作为科学batch elapsed。

神经读取receipt覆盖包括共享学习重放和物理profile：已知最低3961次全片读取、131168frames；
teacher native encoded34103frames/1917.35秒，自身native encoded45941，缓存命中57620，record等待2374.90秒。
这些不是deployment教学读取次数，不能与上表全片读数相加。
首次SIGTERM的5个worker缺读取receipt和未保存环境步数，未知额外量明确保留，不能填零。
formal聚合的`attempted_confirmed_environment_steps=329699`只计成功恢复worker的receipt；
它与完整已保存353308的范围不同且重叠，不是整轮总量，也不相加。422283仍是已确认最低总steps。
空`cost.failures`数组不能证明没有工程失败：首次batch失败/取消由父级记录保存。

完整费用**14.262670 GPUh**，科学elapsed **5.133958h**（包括首次失败、恢复及profile），
在10–16GPUh/3–6h预期及20GPUh/8h观察线内。父allocation与child consumer是两层账，不能相加：
parent14.262658、child14.188625、未覆盖consumer .00001264，以parent+未覆盖计预算。

| 父allocation阶段 | GPUh |
| --- | ---: |
| 共享学习 | 6.381753 |
| φ180刷新 | .745363 |
| train180 | .201895 |
| train360 | .245238 |
| formal首次+恢复 | 6.433548 |
| 零更新物理profile | .254861 |

首次失败的9.089735GPUh已包含，所有费用窗闭合；接任者仅CPU消费，没有新增GPU、环境或学习费用。
物理profile同fresh逻辑batch零更新，warm world1/2/3/4为37.63/20.68/21.85/18.24秒；
3卡与2/4卡节点/共驻不同，不能宣称硬件无关最优。4卡对2卡wall约快13%，GPU时间高约76%，
选2卡训练，让独立刷新/读出利用其它卡；peak allocated26.86/reserved28.08GiB。
frame32/experience64/native32、FM56/support64/adjoint64的实际收益与未放大依据保留于`analysis/physical_profile.json`。

## 7. 存储、资源、Git、原件与科学边界

收尾strg01独立data1 user用量1340.375GiB/2048GiB，ROOT实占136.887GiB；共享可用约79.91TiB另核验。
cache两节点登记payload63.870GiB，小于64GiB；实际磁盘分配64.096GiB包含SQLite/文件系统开销，计入ROOT。
收尾与已保存准入快照未越224GiB，但没有连续测量历史峰值，不能把136.887写成最大值。
现场gpu01/gpu02本批Python/torchrun与GPU进程均0；此次owner其它GPU进程也0，未操作他人任务。
见`analysis/final_storage_snapshot.json`及`analysis/resource_exit.json`。

当前实际消费者80项CPU检查通过（71.11秒）；当前结构检查无hard violation，四个新增职责明确的模块复用同一运行面。
48/48/400原行等于聚合，400状态/teacher无放回及scene/RNG、practice与final分离通过实际消费者核对。
90/180/270/360均保留完整model/optimizer/sampler/cursor/rank RNG/schema/topology状态，world2、逻辑4，非weights-only资产。
检查原件为`analysis/raw_pairing_verification.json`、`analysis/architecture_delivery.json`及`analysis/verification_completion.json`。
CPU汇总辅助脚本只修正评测wrapper中嵌套condition_id的读取，不改冻结代码、原科学数据或模型。
代码与消费记录串行集成main并推送；已整合的task开发工作树退役，frozen源码、原失败、完整checkpoint、raw rows与必要分析保留。
具体Git/lifecycle/交接身份以`scientific_completion.json`、`analysis/lifecycle_cleanup.json`及`main_delivery_receipt.json`为准。

本轮降低“当前固定P及FM+Dkeep在360更新内足以教会有用功能修订”的支持度；
能量头增长但Reader学习及实际写入都弱，使首轮两步仅属初始化暂态的乐观解释受到更多限制。
仍未区分幅度、方向、上游信息利用、有限支撑迁移或监督信用之间的独立因果作用；没有PG训练结果。
旧T161说明合法FM可以获得强闭环，旧PPW/local-field说明导数可执行不保证有效，ADSP/SKNC及shared-RL阴性继续约束保持/回报充分性。
旧Goal/Long正例和本轮Goal3局部成功均保留，但不能代替总体净增或视频必要性。
本批完成不关闭EMBER；新主讨论消费实际输入—算子—梯度—参数—执行链路后自主登记后继，
不由此报告自动启动PG、改变P/rank/LR/seed或追加GPU探针。

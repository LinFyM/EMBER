# 当前要求、状态与不可混同的历史选择

## 1. 当前真正要求解决的问题

给定一个新教学condition：exact task language，以及一条或多条同task、action-hidden、内部有序的正确教学视频。系统从既有冻结π0.5-LIBERO source的恒等任务LoRA起点出发，允许有预算的真实实践、任务内学习和参数修订，在最终评估前形成一套完整的task-conditioned LoRA。之后这套LoRA冻结，在未参与该condition适应的新初态闭环完成任务。

目标是完整固定LoRA明显、稳定地超过强MT，并理解教学内容怎样产生能力，尤其怎样帮助本来失败的任务。局部功能损失、LoRA范数、参数拟合、语义读出、非零梯度、工程完成以及大量实验本身都不能替代这一结果。第一阶段不要求先赢过所有充分语言适应baseline或先做齐全部视频controls；它也没有取消实际视频作用和完整能力要求。

当前明确使用**raw source为唯一执行基底，全部38 targets、每处rank8**。不以MT初始化任务LoRA，不将MT合并进source，不增加隐藏的高rank共享执行adapter或第二专家。源模型物理权重始终冻结。若恒等LoRA采用非零A、零B，其初始有效修改仍为零；这不等于全部参数必须为零。

## 2. 合法信息与方法边界

| 阶段/来源 | 允许 | 不允许 |
| --- | --- | --- |
| 新condition的教学 | 指定正确视频的RGB、内部时序及exact language | teacher episode动作、proprio/state、reward/terminal、task ID、文件名、物体pose、隐藏normalization或特权outcome作为条件 |
| 当前真实实践 | 自己实际RGB、proprio、执行动作、环境状态、接触、reward/success/terminal等 | 把这些与teacher标签混同；用最终评测反馈继续适应 |
| 合法共享训练 | train24及明确审计non-held meta任务的动作、专家、功能和真实回报监督，标签消费者须明确 | held离线标签产生共享梯度，未登记的held经验混入共享训练，按task ID查专家字典 |
| 参数编译 | 用教学与实践生成/修订当前完整LoRA；可含学习、搜索或多轮内部过程 | 先恢复、合成、搜索、采集出动作轨迹，再以其微调最终LoRA；交错执行也不豁免 |
| 最终部署 | 同一condition一套完整固定LoRA，原生policy读取自身观测与exact language | 继续读取teacher视频、Writer在线运行、外部阶段选择、继续适应、多video LoRA平均、第二执行adapter |

禁止的是“准备动作轨迹后拟合最终LoRA”这条方法路线，不是所有梯度、优化或真实反馈。共享训练用合法动作教Writer怎样生成LoRA仍然允许。新condition直接依据实际反馈修改参数，必须说明这仍是教学编译机制，不能将伪专家动作拟合重新命名为编译。

Owner已明确放弃**没有实践信息的直接生成路线**，原因是数月真实性能不足。不得仅换decoder、A/B参数化或名称便重新恢复。此选择也不意味着必须追求人类式轮数、某种特定RL或每个episode结束才更新。

## 3. 最新对实践与迁移的要求

- 不预设观看/实践/修订的固定轮数，不把“两轮”改成别的固定轮数菜单；具体组织与预算须由完整原理解释。
- 使用现有合法初态和真实实践。不得凭空增加任意中间状态重置、从教学目标状态开始练、专家代走前段，或把场景重建当已经解决的接口。
- 对自己已访问状态的分支/重放，须说明实际已有接口、数据身份和预算；不能把某个历史诊断接口当成无限任意重置的授权。
- 全失败可以有局部操作信息；不能把成功当所有更新的先决条件，也不能由局部关系变化推断完整任务已学会。
- 新初态泛化须由同一固定LoRA承担。实践成绩变好、选择到一个较好练习分支或拟合当前H，不等于跨初态能力。
- 保留真实自身H中的间接语言/任务信息。Owner要求抑制Compiler以任务名/完整指令作索引绕开视频，不要求输入在信息论上与语言独立；删除直接L不是自动理解视频的证明。
- 不能以“source有能力”“rank8参考很强”“理论可达”代替视频识别、实际学习或迁移的论证。

## 4. 数据与正式评测

目标为LIBERO Spatial/Object/Goal/Long共40 tasks。发展协议使用唯一显式24 train/8 validation/8 test划分，具体名单与同义/spec重合审计以当前仓库合同为准，不能按结果更改。历史协议、训练任务数和来源要保留，不混排。

source来自generic `lerobot/pi05_base`，使用LIBERO-90中排除与目标40重合的19项后留下的71任务、每task50成功episode训练。不能使用已经读过目标40动作的`pi05_libero`。normalization只来自过滤source并冻结。后继共享训练可增加经精确审计的non-held LIBERO-90任务，但须明确allowlist和provenance；更多同task episodes不算更多独立任务映射。

正式结论使用single-checkpoint、严格paired400，报告per-task、per-suite、breadth、retained/gained/lost、churn及相邻success-set重合。80行screen、checkpoint union或融合不能选模型。K=1的每task50初态必须使用50条合法teacher各一次；跨点复用同一state-video映射，固定调度不随GPU/worker改变。实践、选点和最终评测初态分开登记。

官方执行为render256/model224、双相机180°旋转、8维state/7维action、10 flow steps、每次执行前5动作后replan、10步dummy settling、成功即停、suite horizon 220/280/300/520。Teacher frame stride固定5。

正确/同task另一视频/错误视频/no-video等比较须同task、state、env/policy RNG、video ordinal配对。shuffled/reversed只在冻结选点后用于最后时序特异性确认，不进入训练、选点或架构修正。Test保留至最终方法冻结；既有封存历史不反哺本次设计。

## 5. 科学判断与协作要求

Owner要完整、连贯的机制判断。每个提议必须说明：要解释的实际不足；最近似旧方案；原特征和算子已做了什么；新改变到底增加了什么；真实标签教哪个参数；为什么可能可学、可迁移；什么结果会推翻该解释。一般链式求导、模块名称或可表示性证明不够。

不要求在有限实验前证明必胜，也不要求定位唯一根因才允许研究；但不能把已知逻辑缺口留给后面的未知模块，再逐项让Owner指出。科学阴性不自动是bug；负结果只淘汰实际检验范围，也不能靠版本改名、无限续训或rank/LR/seed小扫逃避停止线。

架构基础可以撤换。memory token、FactorHeads、shared A、固定latent、Reader、E、U、G、π、具体decoder都不是不可改变的目标；**当前source/rank8与信息墙是Owner明确要求**。不为保留历史模块或“用上backbone”制造无意义forward。

请保留有利与不利证据。训练域G明显强于MT的正例不能被完整Writer不足抹去；native少量净增不能替代广泛获取；旧FM/真实反馈早已存在，不能重新接入接口便宣称机制突破。因果结论、推导、假设与尚未测过必须分开。

## 6. 后续规划的实际约束

当前只向Pro征求科学判断与可执行规划，不授权其启动实验。没有active design或在途训练。不要把旧设计未完成清单自动恢复，也不要发送消息联系执行者。

未来若讨论实验，需要从小而完整、能改变方法判断的一批开始，给wall-clock预期、证据依据、停止/扩大分支及实际资源成本。没有累计GPU小时硬上限，但必须准确记账，不扩大无效矩阵。物理GPU两节点总上限8、空闲总数≤10时降为6、单节点最多6；只用真正改善吞吐的安排，不挤掉别人任务。新产物在data1并查独立user quota；大资产复用。正式运行来自clean pushed的冻结代码，不能原地改冻结树。

这些是实施边界，不是要求Pro检查实时GPU或在回答前增加工程审批。理论回答应先解决完整学习问题，具体执行由后续明确合同承接。

## 7. 权威文件

本摘要服务本次阅读，不替代[Owner要求](../../current_owner_requirements.md)、[项目AGENTS](../../../AGENTS.md)、[concept](../../concept.md)与[当前progress](../../../progress.md)。这些文件包含很多历史段落，只有最新明确要求与顶部状态有当前效力。不要机械采用旧MT基底、固定轮数、rank128或旧145/400门槛。

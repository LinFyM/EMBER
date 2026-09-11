# 普通FM下扩大独立非held任务映射：95-task有限对照

2026-09-11登记。唯一active design由progress确认；Owner休息期间自主修正、验证和再次修正的授权继续有效。前轮R/C/S及两项续训已完整结束，原合同保存在[消费接口设计](video_consumption_writer_design.md)，原件与结论见findings§51–53。本设计不恢复其待办。

## 1. 问题、假设与历史边界

目标保持validation8 single-checkpoint strict paired correct>145/400，以及相邻稳定、低churn、breadth、四suite/Goal/Long、换正确视频和独立最终视频因果资格。Test封存，初次生成唯一完整LoRA即闭环执行。

R/C/S200为83/50/45（各400），低于同target动作曝光旧off200的108；R扩到400仍只有85，train96=56，而旧off400为126/56。旧off继续500/600为73/54，train600却达到64/96。条件数量和后续普通监督曝光均未稳定改善迁移；不能把它们视作已成立的修复。

本次假设：24个独立任务映射限制共享video→LoRA函数的迁移学习，扩大正确任务/视频/动作关系的组合覆盖可能改善它。R400虽覆盖383种task-video组合，仍只有24个任务；增加同task视频不同于增加独立meta-task。该假设尚未被当前强off图检验，不由训练分数或loss自动推出。

历史近等价边界已经核查：meta73/target18使用较弱component图、128updates、每target1024queries和两条fit视频；混合55meta+18target的train180为42，target18为55，两者四点screen80均弱。该组合曾显示加入meta的取舍，不能忽略；但其target模型能力、参数化和预算不同于当前off126/train56/96，不能等同于本次。详见research_history“meta73、target18与fully-random”。不恢复原component、专家字典或旧teacher初始化。

只改变训练任务覆盖和与之相应的显式分层采样。保留较强off图，恢复其每task一条K1视频/64个跨episode queries。不是沿用R双条件配方再叠加数据。新模型fresh，历史off200/400只作同target动作曝光的行为参照，不能被重标为新formal模型。

## 2. 不变的部署与参数生成

exact task language + 一条action-hidden正确有序视频，stride5、agentview
→ 冻结vision/Gemma真实prefix的Z/KV和task-token mask
→ 单固定probe、flow time1、Action Expert及共享读取Meta
→ 最后50个post-norm H
→ 原四组过去局部对应、完整H查询、两端Z视觉核实、有序GRU、过去长程与前三组回写
→ local_h_read集合Compiler（额外Compiler语言query关闭）
→ 原独立native-factor D
→ 唯一38-target rank16完整A/B LoRA。

逐帧上下文language、本来的视觉核实和读取Meta均保留。Writer、Action Meta和probe按原合法identity规则fresh联合训练，source无trainable parameters；不打开VL Meta，不换D/rank/视角，不加辅助loss、保持约束或RL。不继承126或R400的Writer/optimizer/RNG。

保留off构造次序及随机初始化语义。退役本轮未获收益的C/S分支、专属semantic/frame-set模块、配置及仅验证这些路径的测试；历史由7fedbe85冻结runtime、run contracts和Git保存。退役不得改变保留off路径的计算、参数构造或采样随机流。仍为一个canonical训练/物化/eval入口。

## 3. 95个训练任务的权限与身份

- 固定target train24保持原ID和protocol，不移动validation8/test8。
- 额外使用`configs/pi05_source_corpus_v1/source_manifest.json`中71个source tasks。其`overlap_audit.json`已按完整任务语义/BDDL审查3600对，排除与目标40重合的19项；不是按当前结果挑task。
- allowlist完整记录在新配置的`data.extra_meta_tasks`，采用`40 + source task_index`作为仅供编排的global ID；`meta_task_id_offset=40`、`meta_source_manifest`和`meta_overlap_audit`明确指向上述已有authority。训练loader核对显式71项、manifest/audit一致、排除项和固定target分界，不能按legacy split恢复权限。
- 旧LIBERO-90内部split仅为历史metadata；本次梯度权限来自排除固定target validation/test及全部target40重复项后的71项审计集合。原source的3550条成功episode及已有资产直接复用。
- target和meta各自均采用video demos0–15、action demos16–41、独立动作诊断42–45、held teacher46–49。每条监督query与本条件video跨episode。诊断只在train24进行，以保持前轮口径。
- source normalization保持冻结，不重算。部署仍只读取RGB、真实帧索引和exact language；task/global ID、文件路径、action/state/reward和source/target分层均不进入Writer。

71项数据已经存在于canonical数据目录，合计52.71GB，无新下载或复制。meta最长视频344原始帧，未超过已知target最长457帧；仍须用真实混合更新验证完整计算与资源峰值。

## 4. 普通FM、分层权重与随机性

每个macro更新固定8个不同task：从四个target suite各取1个train task，再从71个meta tasks均匀无放回取4个。每task独立K1教学条件、64个action queries；合计8条件/512queries。target组和meta组各占loss的1/2，每个条件权重1/8；组内task等权。各target suite保持原等权，不以source任务数量淹没目标域。

配置明确`tasks_per_update=8`、`meta_tasks_per_update=4`、`conditions_per_task=1`、`queries_per_task=64`。只有上述混合合同或原target24合同可通过配置与job检查，不引入任意权重/多策略框架。functional仍先按每task完整64个query生成随机批，再按执行微批分块；所有条件按SUM归并，不改变loss定义。

target task/video/query三个随机流保持旧off seed7和调用顺序。meta使用独立`meta_seed=20260911`派生task/video/query流，不消耗target流。完整checkpoint保存所有流、target/meta task occurrence、cursor、Writer/Meta/probe、optimizer/scheduler和rank RNG；实际恢复下一批须包含两类任务且与未中断采样一致。增加meta不允许碰held梯度。

AdamW、恒定LR3e-5、原warmup/weight decay/clip与BF16/TF32执行语义保持旧off配方。每个target仍在同一macro获得原64queries，但在总目标中的权重为1/8；加入meta梯度和target/meta各半的目标是本次数据分布改动的一部分，不宣称与target-only更新逐bit或梯度等价。

## 5. 有限学习、比较与裁决

fresh seed7，登记200/400两个checkpoint；各执行同一canonical correct strict400，400补train24×states32–35×teacher46–49的96行。独立train24动作验证0/200/400，保持原seed20260908+task、每task128queries、teacher46+task%4和actions42–45，无梯度。

200累计target51200 + meta51200 = 102400queries；400累计target102400 + meta102400 = 204800queries。target动作/视频曝光分别与旧off200/400匹配，新增meta的动作、条件、独立task覆盖和资源成本单独报告，不把额外数据隐藏为同总预算比较。监督condition累计1600/3200，各半来自target/meta。

报告每个节点per-task/per-suite/breadth，以及相对source、旧off同target预算、相邻节点的retained/gained/lost/churn/Jaccard。400 train96与旧off400及新200/400动作诊断共同区分训练获取和未见任务迁移。不得以loss下降、更多task或较高训练分数选择方法。

若200/400仍显著低于旧off同target曝光、没有广泛未见任务改善，则结束本次扩展，不再原样加步数或扫比例/seed/LR。只有实质、广泛并接近或超过目标线的改善，才另行登记相邻稳定节点。当前不预许400以后续训，也不因一个峰值完成goal。

没有新增wrong/static/no-video/shuffle/reverse臂参与学习、选点或架构修正。达成correct相邻资格后才登记same-task-other和独立最终因果controls；已看过的历史视频诊断不冒充新最终证据。更多任务可能仍产生语言或静态捷径，只有完整行为资格能够排除该解释。

## 6. 实现、资源与交付

`learning_data.py`拥有固定target loader及审计meta loader/采样和恢复；`training.py`拥有配置、实际job与曝光合同；`supervised.py`按同一逻辑batch权重传递FM梯度。复用RawTeacherVideoStore、FunctionalQueryDataset、prefix cache、Observer、完整checkpoint和native eval，不新增trainer、入口或持久化cache。

数据/权重实现与C/S退役在两个独立worktree并行，写作用域不重叠；主agent整合、检查off路径及正式配置、运行针对性合同测试和真实混合GPU更新/最长视频/恢复。新meta task的input wall、分层权重、target流不被meta消耗及两类任务的exact-resume为必须验证的合同；不做全模型哈希或防御性逐tensor扫描。

formal来自clean pushed detached commit。两节点live检查后选择单节点最多6张真正提升吞吐的A40；用实际8条件完整梯度吞吐与显存选world/microbatch，不等待固定卡数。NCCL_P2P_DISABLE=1、GPU-local NUMA、deferred NCCL保持；不占空卡或干预他人进程。

预计新增峰值40GiB，包含临时profile及resume checkpoint、两份formal checkpoint、896个物化LoRA、指标和raw rows；不复制source/数据/环境。最近/data1 quota观察used802241268KiB、soft1073741824KiB、shared83TiB，仅供预算准备，实际launch前刷新。明确已消费的profile checkpoint可删除，unique formal证据全部保留。

本设计登记只确定这一项有限修正。实现/profile失败按真实工程合同修复；科学非通过按上述行为边界裁决。全过程继续维护task_plan/progress/findings/history，目标仍未完成。

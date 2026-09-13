# Visible-Object Grounded Writer

2026-09-13登记。Owner要求继续推导、实施并依结果调整；本项检验新的训练信用关系，
不恢复已关闭的双相机纯FM实验，也不改变整体视频特异性目标或现行资格口径。

## 1. 依据、竞争解释与历史边界

旧v4冻结回放出现具体错误绑定：Object3的31条shuffled-only成功中，23条correct运行接近／抓起绿色干扰瓶。
Horizon也有错误对象／实例及Object13后期转抓干扰物。能力不足不只是未能移动机器人，但这些回放不能证明
当前所有失败都来自注意力。原件为`3a6f801d:docs/action_forecast_writer_v4_root_cause.md`§2/4/7.1及
[Horizon冻结诊断](horizon_k1_frozen_diagnostics_20260909.md)。旧共享更新也可改变目标选择；完整LoRA存在这项行为作用。

限定近邻审计：v4诊断没有pose读取、optimizer更新；v5只用normal-order AS；Horizon两端真实视觉cross-attention
只由最终LoRA FM间接学习。它们的task/padding/causal mask没有物体标签，H×H对应也不是图像patch权重。
没有找到这些近邻已用真实可见物体／部件位置直接监督实际视觉cross-attention Q/K的训练。
这不宣称全部研究历史或文献从未使用物体监督。

最近标签工作确认物体／关节部件可恢复，但双指接触和region owner不是通用操作语义。
局部物理Jacobian也没有通过精度资格，继续关闭；本项不使用它、不监督语义rank或增加世界模型。
新增信用落点是实际读取原始视频证据的注意力分布。相对旧辅助动作头、参数几何和隐藏状态蒸馏，
它约束被读取的空间位置；最终能否编译成有益参数仍完全由真实跨episode主FM及闭环裁决。

[GAP作者材料](https://lambdavi.github.io/gap/)提供mask监督空间pooler后用于策略学习的外部先例，
但使用代理任务warm-up和关键点几何，本项不复制其新仿真任务、阶段初始化或部署接口，也不继承其分数。

竞争解释及本项可区分范围：

- 若缺少物体绑定信用是一个实际限制，直接指导真实视频读取应在相同曝光下改善正确视频闭环，并帮助有序收益迁移。
- 若空间监督学会但闭环／有序收益未保持，则此项物体与运动落点信用不足以修复当前共享编译链；不据此追加mask头或堆模块。
- 若监督本身未学会，不由此断言Compiler是根因；同一有界预算结束后报告实际失败接口，禁止loss/层位/权重扫描。
- 更好的物体定位可能只改善静态任务识别。本项保留充分训练的全帧集合参照，合法选定后还须最终内容／顺序controls。
  空间监督或有序对集合的系统差额，都不能单独证明部署视频因果性。

## 2. 完整信息路径与唯一主要变化

```text
exact language + 一条有序、同步双相机教学RGB，stride5、完整末帧
 → 原冻结vision/Gemma + teacher VL Meta → 图文Z/task-span/KV
 → 原冻结Action Expert + Action Meta，固定public probe/t=1 → 完整R[T,50,1024]
agentview同一组真实帧 → 原冻结V-JEPA过去四帧 → M[T,576,1024]
task tokens实际读取native图像patch → 实际读取M → 两端完整H条件化读取
 → 原过去单向T×L表示 → 原Compiler/native A/B → 唯一38-target rank16 LoRA
 → 冻结source读取自己的当前观测和state执行

仅训练：官方OOI/XML/存储state → 可见物体与部件运动patch分布
                       → 上述两处实际注意力的辅助损失
```

只增加训练侧监督关系，不新增可训练参数、输入分支、部署mask、对象检测器、字典或第二adapter。
所有原有模块fresh共同训练，source/V-JEPA冻结；完整H和过去依赖保持。
部署Writer的forward/输入不接收标签；标签只由训练调度加载并在loss中与实际Q/K分布比较。
不把教师state或OOI身份交给特征、queries、Value或参数生成。teacher主FM仍与action query跨episode。

## 3. 可核实的空间标签

只构建train24的教学池demo0–15，共384条；不构建validation/Test标签，不读action、reward或terminal数值。
CPU MuJoCo只恢复运动学及渲染segmentation，无环境step、GPU或新模型。沿已验证合同恢复obs[i]对应
states[i+1]，回退一个2ms积分子步后forward。当前末帧缺少下一存储state时只跳过该帧监督，Writer仍读取真实末帧。

OOI采用官方BDDL原列表；物体保留其匹配body及后代，site-only条目保留owning body及后代，排除world。
region owner只提供相关实体表面，不被标为区域内部或放置谓词。机器人、背景、其它干扰物不是OOI。
仅可见visual geom参与mask，遮挡由真实相机渲染处理，不把不可见完整轮廓当作像素标签。
MuJoCo Renderer输出已垂直翻转；随后水平翻转，与原RGB的180度旋转合同对齐。

native标签：对每个可见OOI，将双相机可见像素的总质量归一为1，再对可见OOI等权。
按原native224空间布局投到两路16×16 patch，合并后归一；无OOI可见帧没有该项监督。
这样柜体的面积不直接压倒小物体，但不假设同一语言token对应哪个OOI。

motion标签：每个OOI中的实际rigid body单独比较当前与上一采样帧，保持抽屉／门的关节运动。
以`d_b=||p_now-p_prev||+2*r_b*sin(theta_b/2)`度量表面移动的上界；r_b由该body实际visual geom
的局部偏移及MuJoCo包围半径决定。该量单位为米、不是接触／意图／成功标签，也不是像素光流。
每个body在agentview的当前可见mask先按面积归一，再乘d_b；不重复计入重叠OOI中的同一body。
按原V-JEPA确定性resize438/center384布局，投到24×24 patch。首帧／无法恢复末帧的motion质量为零。
运动质量在整条视频内归一后形成时间权重与每帧空间分布；静止帧不被强行放大成运动标签，
整条无可见运动时该项为零，不用静态标签替代它。不扫描运动阈值、身体子集、尺度或camera。

四suite样例已用真实RGB叠图检查native位置；构建后检查全部384条shape、finite、采样索引及空间映射，
保留每episode覆盖和运动质量，不以覆盖率阈值冒充学习／行为通过。数据不对齐属于工程失败，修复前不学习。

## 4. 实际注意力监督与反例

两处读取已有`A_hl=softmax(Q_hl K_h^T/sqrt(d))`。监督平均后的真实读取质量
`a_p=mean_(h,l) A_hlp`；heads与语言tokens仍可分工，不把每一head都压成同一分布。
直接复用同一次投影的Q/K；SDPA仍生成原Value读出，训练额外计算稳定log-mean概率用于损失。
不使用额外预测头，不用标签重加权实际attention或替换Value。

native损失为有效帧等权的`KL(q_object || a_native)`；motion损失为上述视频内运动质量加权的
`KL(q_motion,t || a_prior,t)`。每个condition目标为`L_FM + .1*(L_object+L_motion)/2`，
再乘既有task/condition权重；λ=.1是固定实现默认，不是推导出的最优值，不扫描。
主FM数值、256queries/update、noise/t和所有原梯度保持。两辅助损失经实际读取器及teacher VL Meta反传；
Action Meta、Compiler和D继续通过真实主FM学习，不因本次标签另设冻结课程。

反例仍在：Value可能在patch间相同，或下游可能忽略读出，使定位正确却无行为效果；B=0也未被这项loss排除。
因此只把它称为位置信用干预，不把“训练实际注意力”推成有益视频必要性证明。
source当前观测、语言和静态图像仍可能足够；本项不人为删除这些合法信息以制造差额。

## 5. 比较、资格与停止条件

两臂均采用相同空间标签监督、fresh200、seed7、既有优化器及逻辑曝光：ordered保留过去四帧和原时序读取；
frame_set沿原合同对每帧独立重复四帧后全帧无序集合。frame_set获得相同训练侧标签，部署仍不读运动／时钟标签。
这检验有序输入能否更好消费同一训练信用，不假装是固定模型输入干预。

固定100/200两个checkpoint，每臂800条件／51,200queries；两节点各train96及validation400 correct，
共8个面板／1,984rows，完整执行后裁决。主动作留出0/100/200仍只定位。旧无监督dual两臂的相同8面板
作为预先固定的训练信用参照；先验证曝光、query/noise/t与eval映射，报告同模式新−旧及交互，不能拼接分数。

qualification完整继承[双相机设计§4](native_dual_video_writer_design.md#4-有界学习与停止条件)：
validation correct−匹配frame_set的task-cluster95%下界>0、至少两个suite净正，另一相邻节点增量同向；
对source有实际收益、正确条件后段不能明显退化。报告per-task/suite、breadth、R/G/L、churn和相邻Jaccard。
采用相同development seed20260911、env/policy seed7及20k双侧bootstrap；现有frame_set口径问题未回答，故不放宽。
视频映射沿原canonical schedule，validation整轮50video各一次；train96明确复用46–49/states32–35有限池。

基础资格通过才补same-task-other、登记的frame_set_image参照及跨初始化保持，再选定并冻结single checkpoint。
最后才运行sealed内容／shuffled/reversed controls；Test、held梯度、RL、checkpoint union/融合全部不使用。
任一资格未通过则关闭当前监督组合，不续训、增节点、换seed/LR/λ/rank/层位或运动标签定义挽救。
若只比旧无监督模型好而无有序／换视频保持，只支持一般定位或正则收益，整体goal仍未完成。

## 6. 工程、资源与运行登记

复用现有`video.py`、`attention.py`、`supervised.py`与trainer/evaluator；训练标签loader/loss由一个小模块拥有。
标签生成是单个有界数据入口，数据建成且核验后退役入口，源保留Git。旧纯FM入口没有独立副本；
旧配置为明确关闭的历史参照，不作为当前默认训练。模型计算图/初始化不变，实验身份由新config与run contract标识。

构建前strg01报告/data1为991.9GiB/1TiB、共享83TiB空闲，相关历史标签2.7MiB、workspace tmp13GiB。
标签构建预算512MiB（含短期217MiB frozen worktree）；未来训练/四checkpoint/LoRA banks另预留不超过12GiB，
正式启动前再测独立quota、相关目录并确认总峰值。复用source、V-JEPA和数据，不复制大模型。
所有正式数据构建和train/eval来自clean pushed detached frozen commit。训练前用实际最长视频完成一次信用/峰值profile，
检查标签只进loss、两辅助项到Q/K的梯度、主FM/Meta共同更新、identity、finite与原采样合同。
不因新增loss写镜像单元测试；用真实图及比例适当的检查。profile不选择λ或模型，不继承其参数/optimizer。

GPU启动前同时live检查两节点，遵守Owner总量与每节点限制、NUMA、NCCL_P2P_DISABLE=1和deferred NCCL。
精确命令、frozen commit、环境、设备身份、输出和恢复合同写一次launch记录；当前尚未启动学习。

实施检查：完整LoRA／视频墙／主FM重放17项、sampling/resume/qualification/evaluation合同111项已有测试通过；
另用真实task0/demo0标签与小尺寸合成特征做一次checkpointed联合信用smoke，全部梯度finite，两处Q/K均非零。
它只验证训练图，尚不是原生模型profile或定位学习证据。活动源约新增400行，两个新源文件分别拥有标签构建与训练loss；
标签构建的单episode函数保持一处模型／renderer生命周期，76行/复杂度22，构建完成即退役，避免拆出永久工具层。
既有materialization复杂函数只更新运行身份，未扩展其行为；新训练身份拒绝旧checkpoint混入。

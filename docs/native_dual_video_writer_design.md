# Native Dual-View Video Writer

2026-09-13登记。当前状态由progress决定；整体goal仍是正确action-hidden教学过程经唯一完整LoRA产生可重复闭环收益、
跨同task视频／初始化／相邻checkpoint保持并迁移到固定validation8。暂不强制145/400。

## 1. 为什么这次有新的机制依据

时间对齐两臂及其8个闭环面板已关闭；旧52/96对40/96的冻结复核也以359/768对361/768结束，不能恢复为稳定正例。
局部动作反演、共享功能读出、输出共享、覆盖、VL适配和旧v4十步forecast的负结果及适用边界继续保留。
本设计不由局部loss差额或早期闭环峰值启动，而由新完成的[原生端点×相机诊断](source_endpoint_readout_audit.md)支持。

相同384个train位置、语言、state-free输入、八噪声与真实未来动作下，单视角full10/t1 MSE=.36548/.34789，
双视角=.13673/.13217，task均值=.25060；固定public probe同样成立。相同读出下，双视角改善24/24任务，
配对task-cluster区间严格为正；增加flow深度没有补偿单视角限制。
这说明当前native输入范围限制了冻结source可直接读出的动作知识，尚未分开腕部内容与预训练输入分布匹配。
它不证明视频动态已经获取，更不能保证共享Writer保留这些知识或变成跨初态行为。

旧99ee2d03只实现同步双视角输入，未登记性能结论；库存审计的38份可判定observer均为agentview，
其余无内嵌observer合同不能用于断言全部历史从未尝试。旧双视频K实验不是双相机比较。
旧v4同时使用不同视觉state提示、完整forecast与时间绑定，其失败约束仍在；本项不恢复它的预测动作编译链。

当前竞争解释与可失败预测：

- 输入范围改善若可经现有过程图消费，应在同一学习预算下改善正确条件能力，并使有序方法相对充分训练全帧集合
  在相邻validation节点产生净收益；只有后者及换视频保持成立才支持当前过程机制。
- 若ordered和frame_set仅共同改善，说明输入帮助一般适应，仍不支持本组合的额外有序收益；关闭本候选。
- 若原生读出收益没有转成闭环能力，说明该读出充分性不能转移到当前共享Meta／表示／编译链；
  不把这次结果唯一归因某个下游模块，不再围绕此组合扫描相机融合、冻结层位、flow time、LR或rank。

## 2. 唯一变化与完整流水线

```text
exact language + 同一episode内部有序的同步agentview/wrist RGB，stride5，含真实末帧
 ├─ 两相机进入各自原生槽位 → frozen vision/Gemma + teacher VL Meta → 共同Z/task-span/KV
 │  → frozen Action Expert + Action Meta，原public probe、t=1 → 完整R[T,50,1024]
 └─ 只取同一组帧的agentview → 原冻结V-JEPA2.1过去四帧dense M[T,576,1024]
 Z任务token读取全部真实native image patches；查询M的Value，再读取两端完整H
 → 原过去单向T×L表示 → 原两层target/rank Compiler/native A/B
 → 唯一38-target rank16完整LoRA
 → 冻结source依据自身当前双相机／state闭环执行
```

相机轴不是视频K，不分别编码成多套LoRA再平均。两路RGB有同一帧索引，分别180度旋转、官方native预处理；
腕部仅新增到native图文prefix。V-JEPA权重、agentview像素、窗口／模态、预处理和dense Value保持原合同。
显式登记`observer.camera_view=dual`与`video_prior.camera_view=agentview`，禁止把双相机tensor整体送进单视角先验。
没有新增可训练参数、辅助头、状态估计器、伪动作输入、teacher action/proprio、expert或第二执行adapter。
任务名称／视频文件名／task ID只在调度元数据出现，不成为deployment条件。

Writer、两组Meta、Compiler与D共同fresh，合法identity LoRA、fresh optimizer／scheduler／sampler；
只用主FM，source和V-JEPA保持冻结。参数形状和初始化顺序保持；不以旧checkpoint或profile状态初始化。
现有joint学习可能改变诊断所见的原生动作知识，这是待检验的消费问题，不通过预先冻结Meta规避。

## 3. 数据与匹配比较

固定train24及teacher0–15、action16–41、诊断42–45、held teacher46–49；K1、stride5、完整末帧保持。
执行观测为post-action，监督从下一动作开始，主FM仍跨episode。四suite各一task、每条件64queries，
每update256queries、task权重1/4；相同seed7、优化器、学习率、初始化、逻辑采样／noise／flow-time及总曝光。
不扩大meta tasks、改split、读held动作梯度或混入RL。

新dual ordered与dual frame_set都从fresh学习；frame_set对两路native单帧内容做相同无序集合处理，
V-JEPA仍对每个真实agentview重复四帧，无端点身份、gap或时间路由，保留整条视频全部采样画面。
沿用[原完整参照合同§4](pretrained_video_grounded_writer_design.md#4-匹配静态参照与强度检验)，不削弱无序臂。

已完成的execution-aligned agentview两臂100/200作为预先固定的输入范围参照，原始train96/validation400不重跑。
核验新旧及新两臂的逻辑曝光、teacher、query/noise/time与评测映射后报告同模式dual−agentview和相机×顺序交互。
它们是固定seed下训练系统比较，不能当作固定模型的输入因果干预，也不能证明跨训练seed稳定。
硬件／物理分片引起的正常微小数值差异接受；不为逐bit复现浪费计算。

## 4. 有界学习与停止条件

预登记100/200两个checkpoint，每臂200updates、800条件、51,200queries；不由profile或早期分数改变学习节点。
两臂先完成两个节点train96及validation400 correct，共8个新面板／1,984rows。
主动作留出沿用0/100/200、train24每task128queries，只作定位，不以loss选择模型。

qualification保持原严格口径，不借尚未回答的全帧无序全局门槛疑问放宽本候选：
validation400 correct相对匹配dual frame_set的task-cluster bootstrap95%下界>0，至少两个suite净正，
另一相邻节点增量同向；correct对冻结source有实际收益，报告四suite、全部task、breadth、R/G/L、churn与相邻Jaccard。
两个节点若有共同正向候选，才补两臂同节点same-task-other及原登记条件性的frame_set_image强静态参照；
correct/other及静态参照资格均通过后，按原后节点优先规则选择一份single checkpoint，不用union或融合。
source的历史47/400只作描述；资格时须有同task/state/RNG合同的source原始rows，缺失部分补齐后方可声称配对收益。

train96沿用held46–49/states32–35有限池，明示复用；validation每task50视频整轮无放回，跨checkpoint固定映射。
development seed20260911、env/policy seed7、20,000次task-cluster paired bootstrap和双侧95%口径保持。
不得将源输入诊断的task/actionMSE区间替代闭环资格。

两节点若没有validation有序净收益、correct后段明显退化或其它资格未通过，按固定窗口关闭；
不追加节点、相机组合、seed、learning rate、rank、Meta冻结或先验窗口扫描来挽救。
若通过，先完成跨同task视频和新初始化保持核验，再按既有规则冻结方法及single checkpoint；
只在该合法冻结后运行sealed seed20260912 correct/other/wrong/shuffled/reversed。
顺序干预重排真实RGB后完整forward所有分支，不提前用于训练、选点或架构修改。Test继续封存。
完整评测、相机收益、局部能力改善及本候选关闭均不等于整体goal完成。

## 5. 实现、吞吐与资源

复用现有NativeVideoObserver同步双相机prefix及单一trainer/materializer/evaluator；
只在native输入准备明确分配agentview给先验，validator与config/run/training身份登记新合同。
旧执行对齐入口由冻结Git树保留，active tree只有本候选运行合同；跨相机配置禁止exact-resume。

验证相机路由时让两路像素不同：native应收到两路，先验仅收到原agentview，K仍为1；
校验完整frame索引、cache只保存frozen embedding/prior、科学配置拒绝错误view与旧身份。
学习／materialization使用同一prior view合同；不建立平行运行路径或新大模型副本。

双相机增加native有效prefix，旧单相机峰值不能直接沿用。先在最长train24 task38 demo0、93帧完成真实两次完整更新profile，
初始frame_chunk4、policy_microbatch8、prior_window_batch4；以真实吞吐和显存余量确定物理batch，不改逻辑曝光。
profile若OOM只调整物理batch／chunk并如实保留失败，不改变科学模型；profile初始化不用作训练。
formal前刷新两节点GPU、strg01独立quota、相关用量与全轮峰值预算，从clean pushed detached commit冻结运行。
最多单节点6卡，训练与全部评测共享Owner两节点8卡／空闲总数≤10时6卡的额度。实际分配依live可用性。

## 6. 实施与正式入口登记

126项针对性检查通过；额外错误prior view拒绝回归通过。初次测试因未设置PYTHONPATH而未收集，
补上仓库既有src路径后通过，未为环境问题修改科研实现。唯一trainer默认配置已随重命名更新。

profile源代码4591cc09，task38/demo0的全部93帧，两次完整条件更新exit0；热条件26.4771秒，
allocated峰值39.1695GiB、reserved42.1699GiB。第二次Writer／Action Meta／VL Meta及prior读取投影均有有效梯度，
source／V-JEPA参数无梯度，profile不保存正式checkpoint。采用frame_chunk4、policy_microbatch8、prior_window_batch4。
该事实只证明计算可运行，不证明有益过程或闭环资格。

初始两臂100/200加8面板参考已有同规模输出约26GiB，额外峰值登记32GiB；
data1现场966.9/1024GiB、预计998.9GiB，含checkpoint、临时写入、LoRA banks、代码树、日志与余量。
正式启动仍刷新live quota／GPU。条件性other／image参照／最终controls触发后另核增长与独立quota，不预先启动。
精确formal command、节点设备、环境、来源commit及资源写入`runs/analysis/native_dual_video_20260913/launch_contract.json`。

## 7. 有界实验裁决（2026-09-13）

两个checkpoint、8个correct面板／1,984rows全部完成，worker均exit0；本候选按§4资格关闭。
validation100 ordered64／frame_set56，差额95%CI[0,+4.5]pp；200为53／39、CI[−.5,+9.25]pp，
两节点下界均未严格>0，200仅Object净正且ordered的Spatial／Goal归零。未触发other／image或最终controls。
相机×顺序交互的正事实与同模式200绝对退化同时保留，详见findings§79与
`runs/analysis/native_dual_video_20260913/bounded_200_decision.json`、`camera_comparison.json`。
不追加训练／扫描，不选checkpoint；整体goal仍未完成，当前执行状态以progress为准。

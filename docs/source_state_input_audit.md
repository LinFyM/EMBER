# 冻结source状态输入诊断（2026-09-13）

本项只定位原生动作知识的输入条件，不训练Writer、不生成LoRA、不运行环境，也不改变teacher部署信息墙。
是否处于执行阶段以progress登记为准。所有结果产生前固定以下合同。

## 1. 问题与竞争预测

当前观察器向冻结source提供真实RGB与exact task language，使用state-free prompt；实际执行policy则按原生
格式读取自己的8维state。此区别在旧v5.2/v6中也存在，是设计选择，不是本轮新bug，更不能单独解释版本间差异。
尚未验证的是：在当前数据和冻结source上，省略state究竟损失了多少可直接使用的动作生成知识。

本项在train24合法action诊断池上，把同一查询当作普通执行输入，比较完整、均值状态和state-free输入。
它与已关闭的局部动作读出不同：没有可训练读出头，直接使用source原生动作生成；不重新拟合旧头或重开旧实验。

- 若真实state显著优于两种参照且优于任务动作均值，则支持“原生知识的有效调用依赖当前状态内容”。
  这只支持随后研究能否从合法RGB得到足够状态信息，不自动采用状态估计模块或新Writer。
- 若真实state不优于均值state，则没有证据把缺少当前状态内容视为这个接口的主要障碍。
- 若只有完整／均值格式优于state-free，而真实state不优于均值，优先保留格式／分布效应解释，不能归因状态知识。
- 若三种source读出都不如动作均值，则本项不支持借状态补全获得强动作教师；停止这一具体修正依据。

预测涉及有限协议下的动作知识，不把离线MSE等同闭环、整个hidden表示的信息量或唯一科学根因。

## 2. 输入、样本与信息墙

固定冻结source，使用现有source checkpoint、tokenizer和source normalization，不载入Writer、Meta或task expert。
同一模型权重用于三臂，无梯度。采用原local-action诊断的train24×16片段，共384个位置：
`runs/outputs/local_action_grounded_20260912/ordered/local_diagnostics.jsonl`中step200的task、demo与start。
episode只含42–45；每task各4片段／episode，索引不因本项结果改变。

每个位置p只读取obs[p]的官方双相机RGB、exact language，以及该臂允许的状态信息；目标为真实
actions[p+1:p+16]，15步均有效。source仍生成完整50-horizon，再对前15×7维评分。
这与旧局部逆动作头看完四帧再估计中间动作不同，因此不作同输入模型竞赛。

三臂：

1. `state_free`：真实RGB＋原Pi05TeacherPrefixTokenizer，完全不提供state。
2. `state_mean`：真实RGB＋官方执行prompt，state替换为该train task在action16–41中、episode等权的平均8维状态。
3. `state_true`：真实RGB＋官方执行prompt与obs[p]真实8维状态，即source本来的执行输入。

后两臂只存在于有标签的train-side诊断。均值使用task身份做离线参照，不是合法部署字典；真实状态不送入Writer。
部署仍只允许teacher RGB与exact language。本项没有validation/test动作、训练梯度、最终视频controls或checkpoint选择。

## 3. 生成与评分

每个位置固定8个独立Gaussian初始噪声，形状50×32；三臂逐样本复用相同噪声。
种子由固定基数20260914与task、clip确定，显式记录；不得依赖设备或运行顺序。
每次10步原生flow生成，三臂相同BF16/TF32语义。8次生成的前15×7均值作为点估计；这不涉及视频或LoRA平均。
真实动作只在生成函数返回后归一化并评分。另报告单次样本平均MSE，但它不替代主要指标。

动作均值参照复用先前保存的`inverse_action_task_means.npz`，其训练池同为16–41且标签同为后15步；
本项逐位置确认原先记录的task/demo/start与标签合同一致，不重新选择均值估计或噪声数。

预登记主差额为state_free−state_true、state_mean−state_true、task_action_mean−state_true的MSE。
先在每task内对16片段等权，再对24task等权。固定种子20260914做20,000次task-cluster paired bootstrap；
三项差额各自95%CI下界均严格>0，且state_true相对两种source参照在至少两个suite有正差额，才通过本项前提。
这是三个同时成立的诊断要求；不能挑一个正数宣布通过。所有task／suite及原始预测完整保留。

## 4. 资源、停止与后续

一次有界384位置×3臂×8采样；不扫state缩放、噪声数、采样步数、checkpoint、dtype或额外层。
没有模型拟合；预计输出小于32MiB、单个源码冻结树约220MiB，新增峰值按1GiB预算。
使用一张合适A40顺序完成三臂，或同节点最多三张逐臂独立运行；共享设备总额遵守Owner合同。
启动前核对两节点GPU、独立data1 quota、相关目录用量与共享容量；命令、设备与完成状态保存于analysis根。
代码来自clean pushed detached冻结树。运行前只做必要的索引、shape与真实入口检查，不创建额外训练设施。

完成后按第1节分支裁决并退役一次性入口。通过不自动触发新Writer：仍需证明合法RGB能恢复足够的内容，
以及这些内容对视频到唯一LoRA的功能有用；未通过就停止本具体状态补全依据，不换头或扩训练挽救。
总体目标和原有闭环判据不变，也不依赖待明确的全帧frame_set是否为长期硬门槛。

# Writer 训练稳定性诊断 E0–E3

状态：2026-09-21 Owner 已授权执行；这是当前 active design。它只解释旧、新 Writer 训练差异，
不产生新的模型资格、checkpoint 选择或 Test 结论。

## 1. 固定边界

- 不 fresh 训练，不运行新 Test，不改变已冻结的 Writer1000 与 MT-BC300 选择，不启动 FT、RL 或外部比较。
- 六个只读父资产为旧 Writer1200（O1200）、旧 Writer1500（O1500）、新 Writer1000（N1000）、
  新 Writer1800（N1800）、新 MT-BC300（M300）和 Source-71 step1000（S1000）。
- O1200 与 N1800是按每任务200次访问及相同学习率对齐的主要优化诊断父节点；O1500和N1000仅为
  冻结行为参照。所有 Writer 诊断恢复父节点完整参数；涉及更新时也恢复各自AdamW moments与step。
- 数据只使用已授权训练36任务及固定诊断demo46..49。Validation闭环只读状态、RGB、reward/terminal用于
  评测，不产生梯度。没有Test动作、reward或状态读取。
- 诊断节点、短更新和虚拟更新均不是候选模型，不得用于重选、调参、延长训练或进入后继实验。

## 2. E0：执行完整性

在O1200和N1800上使用当前36任务事件1801的前两个condition，分别恢复其原生frame chunk 16/8。
独立重放跨episode主cotangent和同视频教学cotangent，检查完整Writer、三组Meta、AdamW参数名、shape、
finite及optimizer state相互独立。输出`assets.json`与`e0_integrity.json`。确定的接口错误、非有限数值或
资产不完整立即停止；数值大小本身不触发停止。

## 3. E1：冻结函数与小闭环

### E1-A 固定动作探针

在当前36个训练任务上，对六个模型使用固定teacher/query对46→47、48→49；query episode分别取25%和
75%位置；每条使用两个固定噪声seed，共8条/任务/模型、1728条。保存：

- 全50步及前5步的主FM误差；
- `tau=1`前5步误差；
- 真实10步flow采样的前5步动作误差；
- 相对S1000的动作变化，并保留动作维与时间维分解。

结果写`probe_manifest.jsonl`与`frozen_probe_rows.csv`。

### E1-B 固定闭环

- 共同held面板：Spatial3、Object1、Goal3、Goal6、Long1，state0..3，六模型，共120条。
- 训练面板：Spatial5、Spatial7、Object2、Long7、LIBERO-90的3、56、11、33，state0..3；
  O1500、N1000、N1800、M300、S1000，共160条。四个LIBERO-90任务对旧Writer是训练外任务，明确标记。

每条保存成功、终止、控制步数、每次replan的执行观察/状态/动作、teacher身份，以及模拟器可提供的目标谓词。
视频映射固定为每个task的state0..3分别使用teacher demo0..3；状态、环境与policy RNG预先固定。轨迹保存
evaluator实际送入policy的图像/归一化state和policy生成的归一化action chunk，并由正式postprocessor执行；
总计280条，只作诊断。

## 4. E2：真实任务梯度与独立单步更新

O1200与N1800使用同一套当前36任务事件1801..1809，每个任务一次，动作来自train demo0..45。
逐任务计算经完整LoRA、冻结policy与整个Writer的主梯度`gQ`、未乘1/3的教学梯度`gS`及
`gJ=gQ+(1/3)gS`。记录Writer main、TextMeta、VLMeta、ActionMeta的norm、Q/S内积与36×36 task
gradient Gram矩阵；保存矩阵，不保存高维梯度向量。

每个父节点均从同一恢复状态独立执行21个单步：9个四任务batch的joint、相同9个batch的main-only、
全36 joint、全36 main-only、零当前梯度但保留Adam历史与weight decay。固定LR
`1.6279650115e-4`，全局clip一次、AdamW一步，每次后恢复父状态。探针比较父节点与更新节点的主FM、
真实10步flow前5步误差及动作变化；非零更新另做norm-matched纯参数位移测量副本。
每个单步版本使用E1-A完整固定面板（36任务×8条），父节点动作作为action-change基准。

## 5. E3：父状态×学习率2×2短窗

从O1200与N1800各恢复完整Writer和AdamW，分别以高LR `1.6279650115e-4`和低LR
`2.959936e-5`形成O-H、O-L、N-H、N-L。四臂使用相同当前36任务事件1801..1872，连续72个更新；
每更新仍为4任务、每任务main21＋teaching7、教学权重1/3、一次全局clip、一次AdamW，保留父step与moments，
不调用原scheduler覆盖固定LR。

局部0、18、36、72保存固定Train36动作探针；18、36、72对共同held五任务×state0..3做闭环，新增240条。
保存每步LR、loss、更新norm和task事件，以及per-task success、gained/lost和动作变化。每次模型更新后重新
编译teacher条件，只允许跨版本缓存原始RGB和frame index。72步后无论趋势如何都结束，不自动延长。
Train36动作探针复用E1-A完整固定面板；闭环同样使用state0..3→teacher demo0..3映射。

## 6. 执行、产物与裁决

严格按E0→E1→E2→E3执行。阶段结果不改变后续阶段内容；只有关键资产缺失、非法数据访问、非有限值、
明确实现错误或资源故障会中止。正式根为
`/data0/user/ymdai/ember_runs/writer_stability_diagnostics_20260921`，大轨迹与权重仅本地保留；轻量结果、
固定视频、索引及复现脚本进入Git。

最终轻量文件至少包括：`registration.json`、`assets.json`、`probe_manifest.jsonl`、
`frozen_probe_rows.csv`、`frozen_rollout_rows.csv`、`task_gradient_metrics.csv`、
`task_gradient_gram.csv`、`virtual_update_rows.csv`、`shadow_training_steps.csv`、
`shadow_task_exposures.csv`、`shadow_probe_rows.csv`、`shadow_rollout_rows.csv`、
`trajectory_manifest.csv`和`completion.json`。README只陈述完成范围、缺失项和事实结果，不提前写因果归因。

## 7. 实现职责与生命周期

`writer/stability_diagnostics.py`只负责checkpoint/Adam恢复、梯度、固定动作探针和参数更新；
`writer/stability_rollouts.py`只负责把固定condition编译为LoRA并调用仓库唯一official `rollout_shard`；
`scripts/run_writer_stability_diagnostics.py`只登记资产和编排阶段。三者不进入训练器、materializer或evaluator的
默认入口，也不替代这些canonical实现。新增表面约1300行是E0–E3可复算合同所需，复用已有runtime、FM credit、
采样器、LoRA batch执行、LIBERO环境池和rollout实现，没有复制第二套训练/评测器。

本诊断结束后删除detached runtime、临时高维梯度和可再生成bank；保留这三个显式诊断入口、轻量结果与设计，
用于报告复算。若E0–E3结论已进入最终研究历史且不再要求代码级复算，则由后续明确清理任务删除这些显式入口；
它们在此之前没有默认调用者，不构成偶然active fallback。

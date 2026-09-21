# Writer 输出空间与 code 投影诊断合同（2026-09-21）

## 1. 问题与范围

本轮只检验当前 Writer 的 q/v-B 共享末层输出空间是否损害旧 Writer 已知有效更新，以及这种影响能否仅由
有效更新幅度缩小解释。它不把 216 维小于 native width 当作容量不足的结论，也不把投影结果直接转换为新架构。

只执行：

- A0：CPU 读取 O1200、N1000、N1800 三个 checkpoint；
- A1：O1200/N1000 × global task 5/7/12/37 × demo46/48，共 16 次无梯度视频编译；
- A2：固定 held global task 3/11/26/31、state0..3，对 O1200 LoRA 做 SELF/NEWSPACE/SHRINK 三臂，
  每臂 16 条、共 48 条新增冻结闭环；
- 复用既有 O1200 原始 16 条作为未修改参照，不重跑。

不执行反向传播、optimizer update、fresh/continuation 训练、Test、FT、RL、完整 E2 或拆 head 学习实验。
本轮结果完成后停止，由 Owner/专家决定是否另行授权后继学习实验。

## 2. 冻结资产与信息边界

| 名称 | 正式资产 | 用途 |
| --- | --- | --- |
| O1200 | 旧同视频教学 Writer macro1200 | 已知有效更新与 A2 parent |
| N1000 | 覆盖重训 Writer macro1000 | 当前选中 Writer 的新空间 |
| N1800 | 覆盖重训 Writer macro1800 | A0 时间参照 |

A1 仅使用当前训练任务的 action-hidden RGB、exact language 与诊断 demo46/48。A2 使用固定 Validation 条件的视频、
环境与 policy RNG，但不读取 held action、reward 作为模型输入，不产生梯度，也不参与 checkpoint 选择。
所有模型、Source、Meta 与 LoRA A 固定。

正式 study 根：
`/data0/user/ymdai/ember_runs/writer_output_space_diagnostics_20260921`。
正式执行来自 clean pushed detached runtime。大权重、code tensor 和轨迹留在 study；Git 仅保留轻量统计、图表、
报告与复算源码。

## 3. A0：输出空间只读检查

读取每个 checkpoint 的：

- `writer.factor_heads.q_b.network.2.weight`，形状 2048×216；
- `writer.factor_heads.v_b.network.2.weight`，形状 256×216；
- 所有 LoRA-B template，要求为物理零。

末层权重先量化到正式 forward 使用的 BF16，再转 FP64 做分解。数值秩使用线性代数库默认的
`max(m,n) * eps * sigma_max` 容差，不按结果挑 top-k。投影基使用该完整数值列空间。

保存每个 family/checkpoint 的完整奇异值、数值秩、条件数、能量，以及同 family 三个 checkpoint 两两的主角余弦、
chordal 距离和投影重合度。N1800只作空间随训练变化的参照，不进入 A2。

## 4. A1：16 次无梯度视频编译

在 q-B/v-B `FactorHead` 最后线性层的 forward-pre hook 读取真实 216 维输入 code；每次编译每个 family 必须得到
18 个 layer × 16 个 rank code。hook 不改变 forward，退出后必须移除。

每个条件保存：

- rank、layer 与 demo46/48 之间的 normalized Gram；
- rank-centered、layer-centered energy ratio；
- 实际 B 的奇异值、Frobenius norm 与能量；
- 实际有效更新 `DeltaW = (alpha/rank) BA` 的非零奇异值、Frobenius norm 与能量；
- `C @ code` 重建 B 的绝对和相对误差。

完整 code 以 safetensors 留在 study，Git 只导出汇总表。高相关仅作描述，不自动加入白化、正交或对比损失。

## 5. A2：三臂 48 条冻结闭环

对每个 O1200 条件和每个 q/v layer，记旧更新 `M = sBA`；`P_O`、`P_N` 分别为 O1200 与 N1000
对应 family 的完整列空间投影。

| 臂 | q/v-B 变换 | 目的 |
| --- | --- | --- |
| SELF | `B' = P_O B` | 投影数值与实现对照 |
| NEWSPACE | `B' = P_N B` | 检查 N1000 空间是否删除旧有效方向 |
| SHRINK | `B' = alpha B` | 保留旧方向、匹配 NEWSPACE 的逐层有效更新范数 |

其中 `alpha = ||s(P_N B)A||_F / ||sBA||_F`。action-in/out、所有 A、LoRA scale、视频、language、
Source 与执行合同不变。三臂分别运行固定 16 条，严格复用 O1200 参照的 task/state/teacher/env RNG/policy RNG。

逐条件逐层记录：

- `rho = ||s(B'-B)A||_F / ||sBA||_F`；
- 原/新有效更新范数与比例；
- 投影 residual、alpha 与 finite 状态。

闭环保存 success、真实控制步数、最终/曾经满足谓词、谓词转移与 policy noise seeds。轨迹采用 compact capture，
不保存不必要图像。

SELF 另用既有 O1200 轨迹的首个 processed observation、noise seed 与 action chunk，重放 ORIGINAL 与 SELF：
记录相对 O1200 原始 chunk 的 max-abs/relative-L2，以及 SELF 相对 ORIGINAL 的误差。接口必须 finite，原始重放应在
正常 BF16 误差内；异常时停止 A2 结论，不把 NEWSPACE 损失归因输出空间。

## 6. 完成与解释边界

完成要求：A0 三 checkpoint、A1 16 编译、A2 三臂各 16 条全部有 completion/exit0；既有 O1200 16 条参照通过
task/state 完整性检查；所有统计可由原始轻量表和本地 code/trajectory 复算。

报告配对 SELF/NEWSPACE/SHRINK 与 O1200 的 retained/gained/lost/churn、每 task 成功、步骤与谓词变化，并按以下边界解释：

- NEWSPACE 明显损害而 SHRINK 保留较好，只支持 N1000 空间删去该旧见证的重要方向；
- NEWSPACE 与 SHRINK 都下降，不能区分方向与适配减弱等效应；
- NEWSPACE 保留功能，会削弱“新 B 空间排除旧能力”的解释；
- SELF 异常则本轮投影干预不足以裁决；
- code 几何差异只定位系数映射候选，不直接授权正交约束或拆 head。

16 个条件是小型冻结诊断。无论结果是否清楚，本轮都不自动增加组合、训练新架构或恢复其它历史路线。

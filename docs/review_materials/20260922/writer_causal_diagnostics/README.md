# Writer 因果路径与辅助梯度诊断：远程专家审计包

本目录是正式 study `writer_causal_diagnostics_20260922` 的可推送审计副本。它让只能访问远程仓库的审计者能够读取
完整合同、实现修正、逐条件结构化结果、完成性和成本，而无需访问 `/data0`。先阅读[结果报告](report.md)，再按下表复核。

专家随后要求用既有真实执行轨迹定位最早行为差异。相应的四个固定 `O1200 success / C600 failure` 对照，连同已验证的
保存动作重放画面、逐控制步动作/状态、完整谓词时间线和同一 teacher 轻量视频，位于
[behavior_pairs/README.md](behavior_pairs/README.md)。这是事后失败定位材料，不是新成功率样本、评测、训练或模型推理。

本包不包含 checkpoint、源数据、视频、完整 rollout trajectory tensor 或环境资产等大二进制；这些不适合提交 Git，仍保留在
canonical study 根。所有用于报告计算的**结构化原始行**均在本目录 `raw/`，所以不需要这些二进制即可重算本报告的均值、
success、breadth、R/G/L、Jaccard 与成本总计。

## 审计入口

| 问题 | 远程材料 |
| --- | --- |
| 预登记目标、D1/D2/D3 边界、允许数据和解释限制 | [设计合同](../../../designs/writer_causal_diagnostics_20260922.md)、`registration.json`、`launch/launch_contract.json` |
| 最终完成性、行数、是否使用 Validation/Test/选点 | `completion.json`、`raw/trajectory_manifest.csv` |
| 全部 D1 路径 probe、路径效应与112条闭环 | `raw/path_probe_rows.csv`、`raw/path_effects.csv`、`raw/path_rollout_rows.csv` |
| D2 native 联合梯度、七个互斥组、batch Gram、J/Q/M/Z 虚拟更新 | `raw/gradient_rows.csv`、`raw/virtual_update_rows.csv`、`raw/virtual_probe_rows_C600.csv`、`raw/virtual_probe_rows_O1200.csv` |
| D3 三臂18步的实际 loss/gradient/displacement、probe和48条闭环 | `raw/microtrain_steps.csv`、`raw/microtrain_probe_rows.csv`、`raw/microtrain_rollout_rows.csv` |
| 轨迹和条件键的本地留存索引 | `raw/trajectory_manifest.csv` |
| stage-worker 成本 | `costs.json` |
| 四个固定旧成功/新失败对照的执行画面、动作/状态、谓词和 teacher 视频 | [behavior_pairs/README.md](behavior_pairs/README.md) |
| 初始 launch 到 retry6 的全部文本 launch/exit/stdout/stderr、contract 与 preflight | `launch/` |
| 结果的保守解释与不可成立主张 | [结果报告](report.md) |

## 行数与复算范围

canonical CSV 的数据行（不含表头）为：D1 probe 544、D1 rollout 112、D2 gradient 190、D2 virtual update 16、
D2 virtual probe 512（每个 asset 256）、D3 step 54、D3 probe 192、D3 rollout 48、路径效应 1,159、轨迹索引160。
`completion.json` 只把 D1/D2/D3 的预登记主交付计为 544/112/16/54/192/48；D2 virtual probe 属于16个候选各32条的
支持性逐行输出，故在 `parts/` 独立保存并在此包完整导出。

各 CSV 都保留 task、suite、demo/cohort、seed、arm/branch、MSE、LoRA 或梯度统计、success、steps和运行元数据。动作
数组、trajectory tensor 和视频帧本身没有放入 Git；这不会影响本包的汇总复算，但意味着审计者不能从远程重新执行 policy。
为通过 Git 的文本检查，CSV 副本仅将 canonical CRLF 行尾规范化为 LF；字段、行序、数值和表头均未改变。

## 工程修正与结果身份

初始 detached asset-root、encoder 调用和旧 VJP 比较问题在相应结果行被采纳前即停止；这些 retry 的全部文本证据已镜像到
本目录 `launch/`，canonical study 也完整保留。最终运行采用以下已推送实现：

- `f9fa880b`：生产路径先合并 LoRA cotangent 再作一次 native BF16 Writer VJP；测量直接保存 `g_J` 与 `g_Q`，并在参数
  空间定义 `g_A=g_J-g_Q`。一事件分解残差为 `7.276e-12`。
- `107806f5`：修正 D2 分组表把 draw-level `joint_grad_norm` 覆盖 group-level 同名字段的报告错误。retry6 仅重放两个
  D2 资产并重新 finalization；D1、D3 和任何 rollout 没有重跑。

`launch/retry6_artifact_relocation.json` 记录这次重放的精确理由和被隔离的 retry5 D2 载荷。retry5 的 D3 结果在结构化
表中保留，且 retry5 J/Q/M controller/worker exit 均为0；retry6 controller/D2/finalize/runtime exit 也均为0。

## 结论边界

这是训练侧冻结诊断，不是正式 Validation/Test，也不选择 checkpoint。D1 只能说明 Core/Procedure replacement 都有下游
功能影响；D2 只能说明当前端点前缀辅助项是非零、非纯比例的梯度项；D3 的18步小面板不能在 J/Q/M 间选出正式候选。
正确视频内容、顺序特异性、辅助项整体价值以及 `tau=1`、前5步、权重1/3各自必要性，均没有由本包证明。

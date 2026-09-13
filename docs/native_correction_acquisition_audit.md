# 原生纠正获取：固定A的误差分解

2026-09-13登记。Owner授权继续仔细推导、实施并按结果调整；整体有益视频特异性goal未完成。
本项是已关闭Native Correction Writer的冻结训练侧诊断，不恢复训练，不选择checkpoint，也不新增部署方法。

**本诊断已完整完成并关闭，触发“空间内误差占主导”分支，见§6；以下登记保留为历史合同。**

## 1. 需要区分的竞争解释

完整有界结果见[原设计§10](native_correction_writer_design.md#10-完整有界结果与关闭裁决)与findings§91。
实际参数误差约.91、独立动作FM变化有限、validation有序50/400且未取得相邻有序增量；
真实纠正oracle的跨episode功能前提和标签共同结构仍成立。尚不能唯一归责某个读取或输出模块。

当前模型输出`W_hat = B A`、`A = normalize(R X)`。完整X包含oracle右空间是任意R的存在性；
实际生成的16行A可能没有覆盖所需方向。即使这些方向已被覆盖，B也可能没有预测出对应纠正。
先检验这两个**参数拟合层次**，比直接换大读取器、追加训练或改变监督权重更能排除无依据的投入。
它不把参数误差等同于行为误差，也不能在B侧进一步唯一分离表示、信息、尺度、优化或共享干扰。

历史近邻已核对`fcdb6e43:docs/research_history.md`§7、§10–12及§21–24：

- fixed-A和窄native Y投影曾实际丢失闭环行为，G1分组后又有局部容量正证据；空间约束与共享获取必须分开。
- 旧G3已有fit-native-span、free-query／解析dual及跨task压缩诊断；较高投影恢复与功能正数没有自动解决共享学习。
- 本项不重做原生完整bank的存在性、不求新dual或扫描维度；只分解这四份已生成模型在其**实际A**下的现存拟合误差。
  旧结果不替代当前输入、目标、生成器与误差分解，也不恢复旧阶段冻结、held数据或task-local solver。

## 2. 精确分解及其限制

对同一视频、同一目标层，令已封存训练标签为`G = B* A*`，当前生成因子为A/B。
P为A行空间的正交投影。任意B的最优乘积为GP，因此：

```text
||BA − G||_F² = ||G(I−P)||_F² + ||BA − GP||_F²
                固定A无法表达的部分     A空间内尚未拟合的部分
```

两项正交，所以第一项是固定当前A、允许任意B时的严格下界；第二项是实际输出距离该下界的差额。
不优化任何模型、不构造可部署oracle，不把最优B写入文件或安装policy。原始度量保持
`Σ_layer error / Σ_layer ||G||²`，再按video和task等权；不重加权层或挑action-out。
若完整标签为零，分母沿原loss取1并单列该条件。

由A的小型SVD取得正交基，float64及默认矩阵秩阈值`max(shape) × eps64 × sigma_max`仅服务此统计的线性代数。
保留全部奇异值和计算秩；不扫描阈值、不修改模型dtype、不做逐元素推理一致性要求。
这个下界允许任意大B和任意条件专属B，可能利用很弱的A方向；下界低**不证明**有限共享B可学或稳定可达。
误差用低秩Gram精确计算，避免物化38个dense更新；先用固定小矩阵和一个实际action-out层核对正交恒等式。

## 3. 固定输入与运行范围

- 使用原ordered/frame_set的100/200全部四checkpoint，无筛选、融合或新训练。
- 固定train24×demo16–19的96个同视频条件；这是已存在oracle的完整96条件，包含所有负条件。
  这些视频属于本模型授权教学池16–41，不能一概称为每个checkpoint已实际见过；从原曝光记录标注seen/unseen，描述性分组不改变总体权重。
- 为四checkpoint分别从完整双相机RGB、exact language与stride5真实末帧生成96套合法LoRA，共384次Writer调用。
  复用现有materializer，`development_train/K1/correct`、video_pool16–19、schedule seed20260911；
  states32–35仅供既有调度给四video分配ordinal，本诊断不创建环境、不读取或执行这些初始化状态。
- 标签只引用`native_correction_writer_20260913/correction_labels/registration.json`中的同task同demo原件，
  不读新动作、不构建新label、不读取42–49标签或validation/Test，不产生梯度、优化器更新、动作query或rollout。
- 正式物化来自clean pushed detached frozen树；冻结原checkpoint、source、Meta、prior及全部生成语义。
  复用数据和模型，不改变当前唯一Writer实现。大输出使用原data0 run根下`acquisition_audit/`，不复制checkpoint或label。

## 4. 预登记读出与停止条件

每个checkpoint报告96条件的实际相对误差E、固定A下界F、空间内差额D，并核对`E=F+D`。
保留38层、24task、4video、4suite与seen/unseen完整分解、原始能量／奇异值，主要汇总保持原共同参数尺度。
对每task四video等权的`F−D`做固定seed20260913、20,000次task-cluster bootstrap，报告95%区间。
这只比较当前参数误差哪部分更大；不评价有序资格，不在四模型间选优。

1. 若四checkpoint的`F−D`区间均严格正，当前实际A空间解释参数误差的主要部分：停止“仅改善B便足够恢复该目标”的投入依据。
   后续首先需要可迁移且合法的A获取／职责论证；本结果本身不授权增rank、换key、求dual或重训Writer。
2. 若四checkpoint的区间均严格负，当前误差主要在A空间内：停止“必须先扩A空间”的投入依据。
   转而审查已有可表达方向为何未被取得；不由此宣称RGB充分、B是唯一根因或只需更多训练。
3. 若区间触零或模型间方向不同，记录混合／未分离结论，停止这项诊断；不追加视频、checkpoint、阈值或变体追求单一归因。

无论哪一分支，只运行上述384个合法前向与一次CPU分解。几何结论不证明视频必要性、闭环修复或完整goal达成。
下一项如需模型／行为干预，必须以本结果和全部历史另行推导、登记；不自动续训本组合或新增完整Writer。

## 5. 资源、核验与生命周期

物化沿既有GPU-local NUMA、正常BF16/TF32及resident runtime；四个独立worker各负责一份checkpoint，无NCCL。
按两节点live可用设备决定是否并行，用于加速真实生成而非占卡；累计所有自有GPU遵守Owner总量规则。
384套同shape完整因子约1.85GiB，连同manifest、原始分解及临时输出，data0新增峰值预算3GiB；
data1新增文档与frozen worktree，按实际tracked文档体积修正预算为512MiB。启动前查strg01两额度、相关个人目录和共享容量。

核验同task／video／checkpoint、完整76因子、有限值、单次合法Writer调用、真实帧与匹配标签、
四份sealed manifest／worker退出和误差恒等式；不做额外hash或全树完整性扫描。
保留registration、launch、requests、日志／exit、原始分解和结论。一次性CPU分析只保留在该run根作证据，
不增加活动CLI；本项结束后移除干净task-owned frozen worktree，原checkpoint／label／formal证据均保留。

## 6. 完整结果与关闭裁决

8c714e91 clean pushed detached完成四份冻结checkpoint×96个同视频条件，共384套合法LoRA。
四worker均exit0、墙钟501–505秒；CPU分解26.286秒exit0，四个sealed bank、全部同视频配对与E=F+D通过。
固定满秩／零A／rank1及一个实际action-out层的dense/Gram／正交核对通过；无新动作、模型更新或环境步。

| 模型 | 实际误差E | 固定A下界F | 空间内差额D | F−D task-cluster95%CI |
| --- | --- | --- | --- | --- |
| ordered100 | .951379 | .313762 | .637617 | [−.371229,−.277388] |
| frame_set100 | .950511 | .314918 | .635593 | [−.366038,−.275823] |
| ordered200 | .900546 | .270922 | .629623 | [−.404789,−.314671] |
| frame_set200 | .900279 | .265043 | .635235 | [−.411855,−.330517] |

四模型的区间均严格负，分别24/24个task均值D>F，各suite与四个teacher序号的汇总也均为D>F。
实际误差中66.9%–70.6%在当前A空间内。按§4第二分支关闭本诊断，停止把“必须先扩展A空间”作为下一修正依据。
下界仍约.27–.31，不能把A称为完整充分；允许任意条件专属B的乐观下界也不能证明有限共享B稳定可达。

原96条件在100/200分别有49/67条实际见过、47/29条未见过；两臂完全相同。
ordered200已见／未见的E/F/D为.894723/.265958/.628765及.913999/.282393/.631606；
frame_set200为.895587/.259538/.636050及.911117/.277763/.633353。
已见教学条件仍有较大空间内差额，不能把当前缺口只归为未见视频；这不是seen/unseen随机因果比较。

相邻有序E下降约.05083，F下降.04284、D仅下降.00799；无序E下降.05023，F下降.04987、D下降.00036。
这些固定面板描述保留A覆盖有所改变，同时空间内获取缺口大体仍在；不据其自动延长学习或调尺度。
本96标签的总能量87.6726%在action-out，不能与全624标签的87.4149%混用或把参数尺度当行为权重。
全部奇异值保留：存在很弱的A方向及数值rank15的情况，乐观投影不保证这些方向易被有限共享模型使用；不扫描阈值或修补数值。

完整24task／4suite／4video、seen/unseen、38层原始能量与奇异值在各`decomposition.json`、`summary.json`和
`ACQUISITION_READOUT.md`；`decision.json`保存固定分支。已核对76因子、真实帧／末帧、checkpoint及单次Writer身份。
当前无active design、在途GPU／CPU诊断或selected checkpoint，整体goal未完成；原Writer闭环non-pass与oracle功能正事实均保持。
下一步先解释有限共享纠正为何未取得当前可表达作用，区分表示／信息、可用坐标及学习信用；
本诊断不唯一定位B或读取端，也不授权新Writer、续训、rank／scale／LR扫描或最终controls。

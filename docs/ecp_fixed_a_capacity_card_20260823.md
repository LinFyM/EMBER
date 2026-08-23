# ECP Fixed-A Closed-Loop Capacity Diagnostic

状态：2026-08-23 **在formal adapter materialization与GPU rollout前预注册；现已完成并裁决为capacity-binding。**

## Scientific question

occupancy-complete Stage 1B在Goal/Long把内部effect objective显著降低，却得到`0/50 + 0/50`。本诊断只区分一个
紧邻接口：当前stable carrier的固定`A_c`行空间，是否本身无法保留已经成功的task-local policy行为；还是成功行为在该
参数面内仍可表达，而现有owner/flow/action objective与12-step求解没有找到它。

这不是ECP新版本、Writer训练、solver sweep或deployment方法。结果只决定下一步应先修正parameterization，还是先修正
effect identification/realization。

## Existing evidence before the diagnostic

对held5每task的latest、independent、earliest三个成功member做了无训练的低秩几何审计。最佳fixed-A投影保留所需
`expert-carrier` correction energy的`83.3%--96.7%`，但只保留expert绝对effective update energy的
`41.5%--62.7%`。Goal最低，Long反而最高；因此几何量本身不能解释二者共同的closed-loop 0，必须直接rollout。

## Fixed analytic projection

每个LoRA target独立求唯一的最小Frobenius误差解：

```text
W_expert = B_e A_e
B_star = B_e A_e A_c^T (A_c A_c^T)^+
W_projected = B_star A_c
```

产物保留carrier的每个`A_c`，用解析得到的`B_star`替换完整B；仍是一套完整38-target rank16 LoRA。它不叠加第二
adapter、不平均LoRA、不使用optimizer、不做interpolation、不选择checkpoint，也不依赖task ID之外的任何部署输入。
这里的task route只属于train-fold privileged mechanism diagnostic，不构成deployment Writer。

## Fixed member and rollout panel

固定使用occupancy-complete bank中已经登记的三个member，不增删、不按投影结果选择：

- `latest`：原始fixed250为`108/250`，逐global `0/9/18/25/36 = 27/30/40/8/3`；
- `independent`：`113/250`，逐task `26/32/37/13/5`；
- `earliest`：`74/250`，逐task `17/16/35/3/3`，Long沿用已登记step500，其余step250。

每个projected member都在完全相同的held5 fixed50 init states、environment seed、policy seed root、language和policy-noise
前缀上运行，共`3 x 250 = 750` rows。三个arms彼此独立报告，不取union、不挑member、不作为模型选择。

## Required report

每个member及其五个task必须报告：

- projected absolute success；
- 相对matched direct member的retained/gained/lost、retention与Jaccard；
- 相对stable carrier的retained/gained/lost；
- per-suite breadth；
- expert/correction energy coverage与closed-loop retention的对应关系；
- pairing、single-LoRA与information-wall检查。

## Preregistered interpretation

以下阈值只裁决fixed-A参数面，不选择EMBER最终方法：

- **capacity-supported**：三个arms合计保留至少`70%`的matched direct successes；Goal与Long分别合计保留至少一半的
  direct successes；且Goal、Long各至少两个member仍非零。此时fixed-A不是当前最早失败接口，下一步应只检验
  effect objective/solver是否能识别已知可表达解，不能扩rank或换Writer。
- **capacity-binding**：三个arms合计保留不超过`50%`，或Goal/Long任一suite合计保留不超过`25%`且至少两个member归零。
  此时fixed-A对成功行为是实质约束；当前fixed-A Stage 1B路线停止，下一步只设计一个carrier-preserving但允许A/B共同变化、
  最终仍合并成单rank16 LoRA的oracle，不先训练video predictor。
- **mixed**：介于两者之间。不得据此扫rank、插值系数或solver；先用per-target energy与paired loss定位是特定target family、
  hard-task row space还是objective/calibration，再预注册一个单目的诊断。

无论结果为何，本卡都不授权Stage 1C video-to-effect predictor，也不授权自动迭代新架构。

## Execution boundary

formal materialization与rollout必须来自clean pushed commit的detached frozen worktree。launch前live检查两节点GPU；只在一个
节点使用最多六张真正增加吞吐的非prohibited GPU。复用现有policy、carrier、expert、normalization和fixed evaluation
assets，不进行训练或大资产复制。

## Completed adjudication

三个解析投影arms的strict250结果为：

- latest：`49`，逐global `0/9/18/25/36 = 26/4/19/0/0`；相对direct-latest的
  retained/gained/lost为`31/18/77`，只保留`31/108=28.70%`；
- independent：`41`，逐task `22/4/15/0/0`；相对matched direct为`22/19/91`，保留
  `22/113=19.47%`；
- earliest：`35`，逐task `23/2/10/0/0`；相对matched direct为`14/21/60`，保留
  `14/74=18.92%`。

三个固定arms合计只保留`67/295=22.71%`的matched direct successes，projected absolute合计`125`，比三次carrier
panel的`129`还少4。Goal原有`24`个direct successes、Long原有`11`个，投影后两者分别为`0/150`与`0/150`，三个member
全部归零。全部750行的episode key、environment seed、policy seed root、language与policy-noise common prefix零mismatch；
18个workers均返回0。

因此同时触发overall、Goal和Long三条capacity-binding判据。当前stable carrier的固定A行空间作为主线参数化停止；这不证明
数学上不存在任何fixed-A功能等价解，也不否定stable carrier、effective-update加法、single-rank16部署或ECP核心。下一oracle
必须允许task-conditioned row-space motion，同时保持zero correction精确返回carrier、不产生A/B交叉项，并最终仍物化为一套
rank16 LoRA；本卡不授权直接启动该oracle或Stage 1C。正式证据：
`docs/evidence/ecp_20260823/ecp_fixed_a_capacity_gate_20260823.json`。

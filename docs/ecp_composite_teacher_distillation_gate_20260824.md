# ECP composite teacher 一轮on-policy phase-expert distillation合同

> **状态：2026-08-24专家复核后取消执行。** 本文保留为历史上已冻结但未启动的实验合同；其`347/500`步训练不得启动。
> 专家确认当前primitive phase expert没有在student composite occupancy上建立oracle authority，且完整50步标签可能跨越
> phase边界，与真实中途切换expert并重新规划的controller语义不一致。已采集`2773/3998` queries只作为student occupancy
> 与weak-teacher response资产归档。当前裁决与替代步骤见
> `docs/ecp_recovery_teacher_expert_ruling_20260824.md`。

## 1. 当前裁决

两个order-specific composite LoRA已经按冻结合同从Gate A3成功轨迹分别训练到step1000。训练工程合同完整、loss正常下降，
但Gate A4的state0预正式资格检查双向均失败：red→yellow-white在step114先满足错误目标而永久invalid；
yellow-white→red在step114完成第一事件，却直到strict400仍未完成第二事件。因此不启动原定100行formal Gate A4。

这不是adapter路由、训练数据错配或LoRA退化：两个adapter的有效更新明显不同，训练HDF5与原teacher ledger的动作逐项一致，
原primitive phase experts在同一observation/noise上可近乎精确复现动作标签。离线paired forward进一步显示，composite expert能降低
部分完整chunk或flow loss，也能在原成功轨迹的phase transition处改善执行前5步，但在自身闭环访问状态上仍会偏离并累积错误。

最早失效接口因此定位为：仅在成功teacher occupancy上做50-token行为克隆，不能覆盖composite policy自身执行时访问的状态与恢复动作。
继续原数据step2000、扫LR/rank/seed或直接跑100行都不回答该问题。

## 2. 唯一修正变量

只进行一轮on-policy phase-expert distillation：

1. behavior policy固定为各variant现有step1000 composite LoRA，全episode始终读取统一composite language并安装同一套LoRA；
2. 在原固定50个initial states上执行behavior policy；每个replan state都按环境predicate确定当前phase；
3. 在完全相同的当前双相机observation、8维state与policy-noise下，临时装载相应task65/68 primitive expert，生成完整50-step
   teacher action chunk作为privileged训练标签；随后恢复composite LoRA并执行它自己预测的前5步；
4. phase、primitive task ID、teacher action、state、predicate和outcome只用于non-held训练数据构造，不进入composite policy forward，
   不成为deployment Writer输入；target40 action/reward读取保持0；
5. 每个variant固定收集50条behavior episodes，不按成功与否筛选，不删除失败或恢复状态。

这样只改变训练状态分布与标签authority：模型结构、source checkpoint、rank16、38 targets、统一language、环境、initial states、
noise、strict400和最终Gate均不变。它不是第二个部署adapter，也不是闭环中动态切换expert；phase expert只在离线数据采集时产生标签。

## 3. 固定再训练合同

- 初始化：各variant唯一的step1000 composite LoRA权重；只继承权重，不继承optimizer、scheduler或cursor；
- 数据：本轮固定50条on-policy episode中的全部replan queries；不与bootstrap轨迹按结果加权，不做成功筛选；
- objective：同一PI0.5 flow-matching action objective，直接监督phase expert生成的50-step action chunk；
- batch：16；训练恰好完整遍历数据两遍；formal optimizer steps固定为`ceil(2 * query_count / 16)`；
- optimizer：AdamW、BF16、gradient clip1、peak LR `1e-5`、25-step warmup与cosine decay；两个variant使用完全相同合同；
- selection：每个variant只保存并使用最终step，不依据loss、state0或variant分别挑checkpoint；不做第二轮DAgger，
  不扫step/LR/rank/seed/scale。

正式采集完成并得到精确query count后，训练config才物化其派生的固定step数；这不是数据后验选择。

## 4. Gate A4与停止规则

最终adapter先各跑一个预注册state0工程资格检查。双向均success、无invalid、adapter/variant及noise匹配后，才运行原50×2 formal面板。
formal门保持不变：

- red→yellow-white至少`20/50`；
- yellow-white→red至少`20/50`；
- 合计至少`50/100`；
- wrong-first invalid、adapter/variant错配、paired state/noise mismatch和information-wall violation均为0。

同时报告A3 bootstrap-success states的retained/lost、A3 failure states的gained、两事件completion、breadth与success-set overlap。
通过才授权Gate B。若一轮distillation后的state0或formal仍不通过，就把结果作为当前order-specific composite teacher路线的
scientific non-pass提交复盘；不自动增加第二轮、延长训练、改pair或恢复phase-switched部署。

## 5. 生命周期与代码边界

`composite_distillation`只负责privileged采集及固定数据manifest；`composite_distillation_data`只负责训练时lazy读取。
最终Gate仍由现有process collector执行，deployment运行面没有新增路径。A3 replay bootstrap和step1000 SFT被冻结为本轮初始化证据，
不与distillation checkpoint并行部署；一轮裁决完成后，临时shard与task-owned formal worktree按证据保留规则清理。

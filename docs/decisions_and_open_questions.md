# Decisions and Open Questions

## 已拍板

### 科研范围

- 当前 Goal 只包含 SmolVLA + LIBERO-90，不包含 OpenVLA-OFT。
- split 改为同分布 70/10/10，旧 60/15/15 退役。
- 70 train tasks 的全部 50 success episodes 用于 source base 和 Writer。
- frozen source embodiment base 统一指：通用预训练 `lerobot/smolvla_base` → 在 70 个 train tasks、每任务全部 50 条成功 teacher episodes 上联合训练 → 得到一个共享、多任务、语言条件的 source embodiment base → 训练完成后冻结。它只能按 train/source evidence 选择，是 EMBER、target direct LoRA oracle 和 ordinary task-local LoRA RL 的共同起点。
- Writer 是一个共享 hypernetwork，但输出 task-specific LoRA。
- Writer 输入是完整 language + 任意数量/长度 action-hidden videos。
- Writer source training 时，同一 50 episodes 的 actions 可作 functional labels。
- validation Writer 冻结且不可看 actions；用于选择 Writer 和冻结 task-local RL 合同。
- 全部方法、checkpoint、预算、selection rule 和 baseline 冻结前不打开 test policy/action/reward surface。
- task-local LoRA RL 先在 validation 冻结合同，最终 test 只按该合同更新 task-local LoRA。
- direct validation LoRA 和最终统一 test 中的 direct LoRA 保留为 target-action oracle/reference。
- 主评估采用标准 LIBERO max horizon 400、SmolVLA execution horizon 50。
- 训练最多用 8 张 A100，每卡平均保留约 10GB。

### 训练对象

- 只用 LoRA，不加其他 adapter。
- 首轮 LoRA 使用成熟 37-target、rank32、alpha16、dropout0 完整空间。
- Writer/direct/RL matched arms 同 LoRA 空间。
- cold start 更新 Writer，base 冻结。
- Writer-only RL 只更新 Writer，不原位更新 LoRA。
- task-local RL 只更新 LoRA，base/Writer 冻结。
- source-only reward/meta outer learning 只可在 Phase F 完成后执行；它只用 source reward 更新 Writer、不更新 shared base，是不阻塞 Goal complete 的可选增强，也不改写已报告的核心 test。

### 不再做

- canonical bank；
- shared task-update subspace；
- soft geometry；
- residual escape；
- Gate 1；
- learning-rate/mask/metric/radius generation；
- extra shared trainable adapter；
- standalone Language-only Writer / Video-only Writer；
- 旧 custom Gate0 RL recovery tree；
- source direct-LoRA localization；
- 所有方法无条件从头全量重训；完全符合最终合同的开发 trajectory 可作 seed 1，只重训受合同变化影响的 arms；
- 当前 OpenVLA-OFT scale confirmation；
- 把旧 h16 当标准主结果。

## 已解释清楚的公平性

- source methods 匹配 70 tasks、50 episodes/task、可见字段和对应训练 steps/data budget。
- LoRA methods 额外匹配结构和参数量。
- RL methods 匹配 interaction、optimizer updates 和 checkpoint selection。
- direct val/test LoRA 因使用目标 actions，不进入信息匹配主结论。
- native zero-step/video-conditioned method 没有 target optimizer steps，不人为补 steps。
- wall-clock/GPU-hours 是报告量，不替代科学预算匹配。

## 尚需在 validation 冻结、但不需要现在询问 owner

### 70/10/10 的精确 IDs：已冻结

deterministic specification-only MILP、seed `20260720`、精确 IDs、factor table、data manifest、train-only normalization 和 hashes 已封存在 `configs/libero90_70_10_10/`。不得根据任何 policy outcome 重新搜索或替换。

### Source base 精确训练 steps

由 8-GPU 吞吐、完整 70-task cycles 和约 30 分钟初始预算决定；checkpoint 在 thirds。

### Writer 精确 global batch/learning rate/steps

从真实显存与吞吐确定；不缩 LoRA/视频。global batch 改变时合理缩放 LR。

### Task-local RL algorithm

在 validation 上选择成熟 ordinary LoRA RL。必须比历史 custom chunk-level flow-loss PPO 更忠实，且先固定 interaction/update budget、checkpoint rule，再开 test。

### 评估是否使用第二个 policy RNG

默认每 task 全部 50 官方 init states。若 variance analysis 显示一个 RNG 不足，再扩到 100 episodes/task。

### 最终 baseline 的具体实现优先级

机制成立后按信息匹配程度与实现成熟度，在 HyPoGen/DISC-style、ViVLA/DAML-style、direct conditioned、retrieval/average、WTL/RIPT-style 中冻结完整矩阵。

## 必须回到 owner 的情形

- 新权限或不可逆数据损失；
- reporting-only test/held 泄漏风险；
- 需要改变 language+video→Writer→完整 LoRA 核心；
- 需要训练 shared base/shared adapter 于 Writer/RL 阶段；
- 多个不可比较选择会实质改变论文 claim；
- 充分、有界 recovery 后核心假设可能被证伪。

机械问题、证据充分且基本唯一的可逆修复不需要形式确认。

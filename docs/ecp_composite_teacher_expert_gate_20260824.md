# ECP order-specific composite privileged experts：训练与Gate A4合同

## 1. 要回答的问题

Gate A3的37条成功轨迹已经通过确定性replay转成两个标准policy-SFT HDF5。本轮只回答：分别训练一个读取统一composite
language与当前执行observation的完整task policy，能否把这37个成功basins扩展成双向均可用的privileged process teacher？

这两套expert仅在non-held process数据阶段存在。它们不部署、不进入Writer、不成为target task-ID route，也不改变EMBER最终
输入输出目标。

## 2. 固定训练合同

两个variant分别使用：

- configs：
  `configs/pi05_ecp_composite_teacher_experts_v1/red_then_yellow.json`与
  `configs/pi05_ecp_composite_teacher_experts_v1/yellow_then_red.json`；
- data：`data/datasets/ecp_composite_teacher_bootstrap_a3_v1/`中的28条与9条成功episode；
- 相同冻结source：`pi05_source_base_v1_seed7`的step1000；
- exact policy condition固定为
  `put the red mug on the left plate and the yellow and white mug on the right plate`；variant、required order、phase language、
  predicate、state ID、reward和success均不进入policy forward；
- source PI0.5全部冻结，只训练38 targets、rank16的单套task LoRA；每个variant从同一deterministic A / exact-zero B
  初始化独立训练，不共享参数；
- batch16、BF16、AdamW、peak LR `5e-5`、25-step warmup、cosine decay、gradient clip1；
- 固定训练1000 optimizer steps，只保存和使用step1000；两个variant不按loss、rollout或各自状态选择不同step，不做
  LR/seed/rank/step小扫。

sampler task ID只用于单数据集内的确定性row ownership，不进入PI0.5输入。训练数据中的8维state与actions是privileged；
target40 actions/reward读取为0。

batch16在同一真实PI0.5 forward上的一次3-step profile已通过：峰值allocated/reserved为
`15,099,039,744/21,313,355,776` bytes，steady steps为`4.096/4.074s`，data time仅`0.045/0.034s`，OOM与
nonfinite均为0。因此正式训练保持batch16，不为显存最小化降batch，也不重复profile第二个同形variant。

## 3. Gate A4执行

训练完成后创建一个新的composite-teacher process manifest。每个variant在整条episode开始前安装其唯一step1000 adapter，
之后不phase switch、不更换LoRA，全程使用同一统一composite language。环境、BDDL、50个init states、dummy settling、
strict400、replan5、10 flow steps、render256、policy noise、public information wall均与Gate A3不变。

正式面板仍是两variant各50行并严格配对state与noise。只运行single checkpoint，不先用部分rollout筛选。

## 4. 预注册裁决

沿用process teacher的原始资格门：

- red→yellow-white至少`20/50`；
- yellow-white→red至少`20/50`；
- 合计至少`50/100`；
- wrong-first invalid、adapter/variant错配、paired state/noise mismatch和public information-wall violation均为0。

同时必须报告但不用于事后选模：

- A3 bootstrap-success states上的retained/lost；
- A3原failure states上的gained；
- first/second event completion、最终conjunction保持、完成步数；
- 两variant success-set overlap与新获得状态分布。

通过只授权Gate B observer discrimination；不授权旧shared realizer、`q_pi/q_V`或Writer。若不通过，先判断是少量bootstrap
coverage、SFT闭环分布偏移还是统一language下的order policy不可分；不返回primitive phase composition，也不自动续训或换pair。

## 5. 运行与停止

正式训练和Gate A4都必须来自clean pushed commit的detached frozen worktree。两套训练可各占同一节点一张A40并发；模型
forward相同且不存在NCCL。训练输出必须保留run contract、1000行metrics、step1000 checkpoint和worker summary。

任一训练出现nonfinite、OOM、数据authority变化或非LoRA梯度则只作为工程失败修复后按原合同重跑；正常loss但闭环门失败是
scientific non-pass，不用训练曲线替代Gate。

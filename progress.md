# EMBER progress

更新时间：2026-09-06 19:10 CST。

## 当前目标与授权

- 最终科学goal持续active；owner授权权限内自主推进。Active design为`docs/joint_process_policy_writer_design.md`。
- 最终资格仍为validation8 strict single-checkpoint >145/400及预登记相邻/跨视频稳定、breadth、四suite与Goal/Long贡献；
  同图fully-random候选和冻结后视频因果controls仍待完成。方法冻结后才做32/8 fresh及Test，没有selected checkpoint。
- Canonical为共同P/Q完整38-target rank16、4750208 Writer参数、source/observer冻结，无独立carrier或部署expert字典。
  最新评测改动仅允许显式outcome-informed training fitting登记，保留真实metadata，禁止其checkpoint选择用途；8项针对性回归已通过。

## 已封存的主要证据

- complete short4 m64 fit/held64/62（各150）主要改善Goal；mixed meta73 m32/64/96/128 validation screens15/19/19/19，未扩strict400。
- 相同18targets/128步的target-only对照实际采样匹配，validation17/17/20/16；Long新增未保留，没有充分支持扩strict400或续同run。
  原三task fit/held行为由meta73的32/36恢复为53/53，恢复仍不全面。历史§173–176与target18 analysis保留全部边界。
- 原18task terminal128 held-video training breadth完成：meta73为42/180、breadth9/18，target18为55/180、breadth13/18；
  按Spatial/Object/Goal/Long14/5/21/2与15/5/25/10，分母40/40/50/50；R/G/L32/23/10，churn33、Jaccard.4923。
  同prefix独立专家111/180，仅为不同预算训练容量参考。Object仍5/40，说明未见task迁移不是唯一缺口。
- 动作轨迹与teacher-state接续显示接触/抓取/最终放置等异质失败，不能统一归因occupancy。Long93专家本身弱且后半阶段并未漏采，
  故该任务不能代表所有Long学习。历史§177/178与`runs/analysis/pi05_ecp_prw_complete_target18_20260906/`保留完整证据。

## 同图单任务学习对照已完成

- Clean pushed detached6efdd2e031dfd42484fa89ca5e472a36a1e9a96a，frozen tree为
  `/data1/user/ymdai/projects/EMBER-worktrees/prw-complete-single-task-20260906`。
- task75（Spatial7）和77（Object2）各一个whole-Writer clone，不添加task query或冻结evidence；component、128updates、每步8queries、
  原两fit视频分别1/2与7/10，每video64exposures；两run的实际128task executions/1024queries/video/policy RNG/normalizer与target18匹配。
  只改变单task目标相对18task均值的共享梯度统计，不能独立命名容量/梯度冲突/优化根因。
- gpu01p2/NUMA0与p3/NUMA1各world1。Train177.21/177.99秒、Panel-B257.20/254.85秒，另计启动加载；peak reserved34.38/34.40GiB。
  source/observer参数0，全部held/Panel-B backward0。训练与物化、两个评测launcher及workers均自然exit0。
- 固定terminal128、原held48视频及states0–9：task75为6/10，对shared18的2/10 R/G/L2/4/0；task77为8/10，对1/10 R/G/L1/7/0。
  合计14/20对3/20，原3个成功全部保留。任务是基于已读训练侧缺口选择，明确非独立held、非checkpoint选择，不能部署clone集合。
- Canonical analysis为`runs/analysis/pi05_ecp_prw_complete_single_task_20260906/`，training_comparison.*与behavior_comparison.*含实际样本、
  功能曲线、20个新raw rows、paired IDs和source引用，精确launch/资源合同在launch目录；历史§179/findings§177已封存。
- 只读核验old v6实际step50/400每task1000/8000queries，历史106/143；其完整train24、50episode/video池、width256、可训练observer与优化均不同，
  因此不能把当前差距简单归为训练量。原件引用在historical_v6_recipe_context.json，不恢复旧coarse/native读法。

## 当前运行：同图 fully-random target18

- 原shared18四点训练侧行为已封存：32/64/96/128为2/3/4/3（各20），Spatial2/3/2/2、Object0/0/2/1；
  相邻R/G/L1/2/1、1/3/2、2/1/2，churn3/5/3。没有接近clones14/20的强阶段；历史§180/findings§178保留边界。
- 当前候选只跳过G2参数复制，保留完整图、4750208个Writer参数、原18tasks/两fit视频/128updates/每task1024queries及AdamW/LR。
  Source与observer冻结，全部训练模块随机初始化，公开非零A模板/零B身份起点保留；不会恢复旧A2 random。
- CPU真实config/loader通过，从clean pushed f3717836 detached tree运行；frozen tree为
  `/data1/user/ymdai/projects/EMBER-worktrees/prw-complete-target18-random-20260906`。
- 2026-09-06 18:48:55 CST启动gpu02p0–5、world6，NUMA0/0/0/0/1/1，launcher806281、torchrun806302。
  0–3/5原有<=210MiB低负载，p4原有5754MiB/2%，peer作业不动。该训练六卡已全部释放。
  训练128updates已完成，34.613GiB peak reserved；实际2304task executions/18432queries/video/policy RNG/权重与component逐项匹配。
  训练及Panel-B均自然exit0：614.19/404.46秒，另计启动。完整audit通过，held/Panel-B backward0、source/observer/tasklocal参数0。
  Random后段五个target功能诊断有4项正收益，component为5项；多数数值接近，内部loss不裁决闭环或初始化根因。
- Quota使用726502144KiB/1073741824soft，shared84TiB，预计新增峰值<16GiB。精确命令、资源与配置合同在
  `runs/analysis/pi05_ecp_prw_complete_target18_random_20260906/launch/`；新run为
  `pi05_ecp_prw_complete_target18_random_s128_f3717836_gpu02p012345_20260906`。
- 已刷新两节点live资源，四个primary validation banks在gpu01p2/NUMA0物化，launcher4123394；固定terminal128训练诊断
  在gpu01p3/NUMA1另一次驻留，launcher4123742。本任务当前只用这两卡；两项完成后接闭环评测。
  四点screen80只分配后续strict400投入；固定task75/77 held48/48 states0–9共20新rows只定位训练侧学习。
  评测采用dynamic long-first persistent workers，启动前两节点live准入、总量<=6；全部negative/Test仍未启用，没有selected checkpoint。

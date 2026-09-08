# EMBER task plan

## 当前目标与授权

继续已创建的全过程goal：完整架构、正式学习、结果分析与方法迭代，直到validation8 strict paired correct>145/400、
相邻与跨视频稳定、breadth/四suite/Goal/Long和最终因果要求达标，再完成冻结方法32/8 fresh与Test。
不设token预算、总工期或总尝试次数，代码/profile/loss/单峰不是完成。

2026-09-08最新Owner安排覆盖从首轮FM/RL同一步更新：**先纯监督FM到有证据平台，再独立共享Writer RL**。
完整[active design](docs/horizon_relation_video_writer_design.md)与信息墙保持；不重新审查全仓库，不继续RL profile。

## 执行计划

1. **已完成：** 全面阅读、历史证据审计、完整Horizon架构及真实FM/Writer/Meta梯度验证。旧384永久停止、旧native草稿保护。
   联合profile全部正常结束，保留代码Git与checkpoint/evidence；不作正式监督起点或科学分数。
2. **已完成：纯监督训练接线并正式启动。** Writer/Meta fresh共同训练，source冻结；同task跨episode FM。
   监督入口不计算RL loss、采集RL更新rollout或做trust回滚。保留原生执行精度/LoRA布局及累加修复。
   复用完整图、批量FM与冻结prefix缓存，检查采样/完整梯度/Adam更新/checkpoint恢复和物化阶段合同，集成push。
   正式从clean pushed detached worktree启动，现场双节点GPU/独立quota/真实用量与新增峰值按合同检查。
3. **当前：监督学习和闭环。** 首段24updates；checkpoint24/64/128/192。每轮4conditions/256FM queries，真实K1/2/4与task曝光记录。
   独立训练侧held-action FM在0及每checkpoint；J0 train120在每checkpoint；validation strict paired400 correct/other在64/128/192。
   24只是早期获取，64以后必须对照source47/SFT109/107；结合per-task/suite/breadth/RGL/churn/相邻及跨视频保持。
4. **判断平台与改进。** 至少连续3个有信息量资格节点、覆盖≥128updates，联合监督曝光、held FM、train闭环和validation。
   有实质进步则预登记后续节点并继续；多信号不再改善才结束监督。明确弱平台先定位支持/表示/生成/执行并依据证据改进，
   不无限续训，也不宣布饱和后交给RL救场。操作化口径见design §8.2，不事后放宽科学资格。
5. **独立RL。** 选定并保留单个监督checkpoint初始化Writer/Meta；独立optimizer/scheduler和stage，默认仅RL目标。
   以监督后真实行为重新设计探索/信用/约束，不机械恢复已停滞设置；报告对监督起点的收益/遗忘/breadth/稳定/J0。
6. **选点与最终流程。** 相邻single checkpoints符合资格后冻结selected，再执行必要视频controls；shuffled/reversed最后且不反哺设计。
   完成既定32/8 fresh和Test。保留唯一canonical实现、完整可复核证据及单个监督回退基线。

## 边界与工作方式

合同内实现、正式训练、评测与证据支持的实质方法修订已授权，不重新等待批准。
GPU/存储/Git只在依赖它们时刷新；不复用旧空闲卡或quota预算。历史由Git、formal artifacts与research_history保留。
validation/test无梯度；无teacher action/state/reward/task-ID部署输入；完整视频/horizon/梯度与单完整LoRA合同保持。
当前状态看[progress](progress.md)，历史结论看[findings](findings.md)与[research_history](docs/research_history.md)。

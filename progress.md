# EMBER progress

更新时间：2026-09-06。

## 当前快照（2026-09-06 14:20 CST）

- 最终科学goal持续active。完整LoRA四任务32 fit/held为50/54、64为64/62（各150），旧P/Q38/37、41/39，A2 43/41、44/45。
  四组新600 rows及全部train/materialize/eval launcher均exit0。主要收益为Goal21/22→31/27；Spatial64为33/31，比A2均少7，Long0/4。
- 新32→64 R/G/L为fit43/21/7、held43/19/11，Jaccard.606/.589；64跨视频51/11/13、J.680。不是最终稳定资格，不能按loss或净分声称完成。
  task6/79功能迁移仍负；真实训练侧收益支持继续同图、更广任务和更充分训练。完整comparison及历史§173已封存。
- 当前主候选：完全相同Writer、55meta+18target、fresh component/optimizer、128updates，每步73任务各8fresh action-query、
  K1/两fit视频、权重1/73及原normalizer；checkpoint32/64/96/128，74752总action rows。延长decay horizon，明确不是纯task-count消融。
  先用固定primary screen80分配评测投入，有希望再逐步做strict400相邻/跨视频。Final同图random候选仍保留。
- Full73四卡profile exit0：micro2每步34.27/32.73秒，146video缓存25.43GiB，最长87帧已覆盖。额外最长样本micro8 profile exit0，
  peak allocated/reserved34.36/34.67GiB，VJP1.20秒（micro2约1.35），相同首步query/noise的loss仅差4.05e-5。
  主训练采用实测micro8，保持8query/task和权重；预估四卡train60–75分钟，另计初始化/capture/Panel-B10–15分钟，实际以formal记录为准。
- 主训练已从clean pushed detached041aff55启动，gpu01物理0/3/4/5/6、world5、NUMA0/1/1/1/1；第五张peer job在preflight前自然结束。
  精确命令见`.codex/tmp/prw_complete_meta73/train.sh`，launcher PID3147740；run为`runs/outputs/pi05_ecp_prw_complete_meta73_s128_041aff55_gpu01p03456_20260906`，
  analysis为`runs/analysis/pi05_ecp_prw_complete_meta73_20260906`，其`launch/contract.json`、`preflight.json`及profile证据已保存。
  当前65/128，32/64 checkpoint已完整保存；最近10步均值22.66秒，peak reserved34.67GiB。
  实际4745 task executions/37960 action rows；四任务256项及旧A2 4672项重叠query/video/policy RNG与normalizer一致，
  跨episode与1/73权重通过，除以既有functional normalizer后的每task梯度系数为.069735–.279459；world5 exact resume锁此拓扑。
  实际启动13:48 CST，预计剩余train约24分钟，之后另计Panel-B；四个primary物化/screen80 launcher及配对统计已备好。
  同query的33–64步原四task loss均高于短四任务（task1/72/83/93差+.000328/+.008831/+.011236/+.008192）；
  扩任务均值与延长decay耦合，尚须新视频及validation闭环判断，不据此更改训练或宣称根因。
  /data1启动前使用824532892KiB/1073741824KiB，shared84TiB可用，主阶段新增磁盘峰值<36GiB；两个profile缓存已正常回收。
- SFT历史109与当前source400逐条task/state/language/env/policy RNG配对，实际合同都是state8/action7/replan5，normalizer在Git未变。
  旧AGENTS“7维state/action”为文档错误，已按官方及冻结runtime纠正，无运行变更；109为历史复用，不冒充新后端重跑。相同80-state前缀SFT24、source9。
- 等待期间确认两节点无旧run进程/mmap引用，回收中断A2 random的97.5857GiB临时cache（manifest明确非checkpoint/非formal evidence），
  并移除clean inactive的7f6b1611与b2bb03ce detached trees；Git、checkpoint及formal结果保留，当前仅041aff55 frozen tree活动。
- 唯一运行图为完整38-target LoRA，无carrier；source源码与b2bb03ce相同，后续只有配置/文档变更。此前Writer12+static bank10测试、真实梯度/
  物化/两种microbatch/full73 profile均已通过。旧A2 random checkpoint与日志保留，不恢复；Test及negative controls仍未使用。

## 历史与证据入口

- 当前active design为`docs/joint_process_policy_writer_design.md`，原始专家意见为
  `docs/expert_review_20260905_full_history_joint_process_policy_writer.md`；后续判断按相关原文及实际触发条件。
- `docs/research_history.md`§170–173保留P/Q短对照、A2 meta73、owner完整输出重对齐及四任务新结果。
- 更早设计、运行记录和旧执行清单已从本即时状态页移除，由`docs/research_history.md`索引、`findings.md`及Git历史保存，不能自动恢复。
- 本轮formal精确命令、资源合同与预飞证据以当前analysis的`launch/`和运行root为准；四任务全部证据在
  `runs/analysis/pi05_ecp_prw_complete_shared4_20260906/`。

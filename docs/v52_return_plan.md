# v5.2恢复与保持对照（2026-09-15）

## 授权与问题

Owner在完整历史分析后要求“你计划下接下来怎么做，然后就启动吧”。本计划据此启动，实际状态由
[progress](../progress.md)登记。Process Pullback窗口已关闭，不从其controls或冻结Test产生新设计。

旧v5.2存在两类缺口：部分任务始终未获取；已出现的条件能力在继续FM后大量交换。
700／900／1200为120／132／92，900→1200保留65、新增27、丢失67。900→1800的source旧成功保留反而39→44，
因此不能只把问题叫作source遗忘。共享读出、完整A/B与视频内容Core曾共同产生有效能力，应先保留这个工作基线。
后续增加容量、压缩到固定出口、局部保持约束及更长训练均没有建立通用修复；没有高置信度的新架构捷径。

## A：固定旧checkpoint的历史复核——现在启动

- 固定模型：`529da6bbe290f7393422937aa7cc278cee732107`的
  `configs/pi05_as_writer_language_axial_v5_2.json`及原训练run的`step_00000900`；不加载optimizer、不更新参数、不选新checkpoint。
- 原训练run：`runs/outputs/pi05_as_writer_v5_2_patch_grounded_dev_r4_seed7_s12000_529da6b_20260728`。
  原参考面板：`runs/outputs/pi05_as_writer_v5_2_correct400_noreplacement_seed7_step0900_529da6b_20260728`。
- 保留原计算图：单agentview、stride5与末帧、Text/VL/Action Meta、50-horizon旧mean、视频Core、Procedure调制、38-target完整rank16 A/B。
  此项是已授权的历史checkpoint复核，旧mean与单视角不恢复为新方法合同。
- 固定validation8 correct400；seed7、states0–49，每task全50条teacher无放回，复用原逐行视频映射与env／policy RNG。
  双相机官方执行、render256/model224、10 flow steps、前5 action重规划、settling10、suite horizon均保持。
- 复用canonical source1000、source normalization、tokenizer、dataset与LIBERO assets。原绝对路径的迁移只改变资产定位，
  不修改原manifest、checkpoint、模型或数据；旧路径兼容及当前运行环境须登记。
- 先做实际装载／真实长视频生成和吞吐检查，再做完整400。A100→A40、正常BF16/kernel与物理batch差异如实记录，
  不要求132或逐行bit一致，不因分数差异重跑挑选。报告总分、task／suite、breadth及相对旧132的R/G/L、churn、Jaccard。
- 本阶段不新增视频controls或Test，不把复核判作方法资格。若结果分叉，先区分运行合同差异与正常闭环波动；
  只修已定位工程问题，不用改模型来追132。
- 追加存储预算4GiB，覆盖约1.03GB完整LoRA缓存、原始rows、logs及小型profile；没有大资产复制。
  每次GPU launch按双节点live快照选单节点至多6张合适A40，动态long-first队列与persistent workers。

运行职责：历史529da6b detached树拥有原模型、generation／LoRA cache、视频调度、官方rollout与统计。
当前`replay_v52_checkpoint.py`只桥接三项明确旧路径比较、现行GPU／quota准入、canonical LIBERO assets和现有strict权重加载。
原run/checkpoint不重写；source内容、Writer schema与全部76个A/B校验不放宽。旧源文件也不改写，operator对精确函数位点作显式替换，
每个worker经同一入口重装；原算法commit和operator commit同时封存。该临时入口由主agent负责，A完成后从活动树退役，
其冻结运行树与Git保留复核原件；不复制第二套科学实现，也不恢复旧训练launcher。

## B：现行合同下的fresh基线——A后推进

保留v5.2的主要能力路径：exact language查询真实视频内容Core，Procedure提供变化条件，Core内容进入Value，
完整A/B由共享family heads生成，两类内容在同一Writer内结合。继续纯同task跨episode FM，source始终冻结。
不默认加入RL、辅助loss、task expert、固定原生span、第二adapter或多套LoRA融合。

新训练必须满足同步双相机、全部50-horizon保留到实际learned read、严格排除同episode action query、fresh Writer/Meta与优化状态。
这些属于明确登记的协议／读取迁移，不能把新模型称为旧132的无变化复现，也不能把迁移收益或损失归因任务分组。
完整H-read接口与训练入口将在A的运行兼容确认后写入本设计，再做真实profile；未封存精确接口前不启动B。
已有授权允许agent完成这些实施选择，无需再向owner逐项询问。

基线拟保留旧曝光尺度：每更新4个等权task，每task一个video条件、21个跨episode action queries；
物理microbatch与GPU数只影响实现。预设观察节点300／600／900／1200，保存每100，最大4,800条件／100,800queries，
其中1200覆盖旧900后明显保持失效的尺度。各节点完整correct400，加固定train24独立视频96条有限面板；
train面板的4条video／task复用范围须登记，不称其为50视频无放回正式验证。
具体初始化、LR/scheduler、teacher/query分池与任务事件表在首个训练更新前封存，保持两臂一致。
如实际profile表明此尺度不合理，先用测量依据更新预算，不从验证分数反推学习预算。

## C：只改变任务共现的一个有界对照

仅在B形成可辨认的新能力后执行。B若已在协议迁移中丧失获取能力，先定位迁移影响，不把失败转嫁给分组。
候选只重排同一批train24训练事件的四任务共现，使天然同类操作／对象条件在同更新中比较；分组只使用train24规范，
不使用validation失败名单、Test或最终视频controls。

保持每task总曝光、每条件视频／query事件、4task／update、query数、总更新、初始化、FM噪声与LR时间轴一致。
旧task-complete实验同时将900更新压为150更新，不能充当这项对照。
在首次运行前封存具体分组及事件表，验证两臂事件多重集合相同；物理负载均衡不改变等权梯度。
沿用B的节点与预算，比较获取和保持：只有新增能力、已获成功保留和breadth共同改善，才支持继续；
不能只以lost减少、总分单峰或loss下降宣称修复。没有这些信号则停止该分组假设，不展开seed/LR/rank小扫。

## 裁决与后续边界

本次顺序为A历史复核→B现行基线→条件满足时C单变量对照，不同时启动多个架构。
正式资格仍看strict single-checkpoint correct>145/400及相邻稳定、四suite贡献、breadth和same-task换视频鲁棒性；
恢复参考和有效局部学习可以记录为正证据，不等于通过资格。
候选能力与相邻资格成立后才按现行合同冻结单checkpoint，补最终视频controls。Test保持封闭，任何未来使用先登记方法冻结。
结果在对话中交付；原始证据放run目录，状态／跨轮发现／历史索引各归其主文件，不另写用户报告。

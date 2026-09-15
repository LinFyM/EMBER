# v5.2恢复与保持对照（2026-09-15）

## 授权与问题

Owner在完整历史分析后要求“你计划下接下来怎么做，然后就启动吧”。本计划据此启动，实际状态由
[progress](../progress.md)登记。Process Pullback窗口已关闭，不从其controls或冻结Test产生新设计。

旧v5.2存在两类缺口：部分任务始终未获取；已出现的条件能力在继续FM后大量交换。
700／900／1200为120／132／92，900→1200保留65、新增27、丢失67。900→1800的source旧成功保留反而39→44，
因此不能只把问题叫作source遗忘。共享读出、完整A/B与视频内容Core曾共同产生有效能力，应先保留这个工作基线。
后续增加容量、压缩到固定出口、局部保持约束及更长训练均没有建立通用修复；没有高置信度的新架构捷径。

## A：固定旧checkpoint的历史复核——已完成

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
当时的`replay_v52_checkpoint.py`只桥接三项明确旧路径比较、现行GPU／quota准入、canonical LIBERO assets和现有strict权重加载。
原run/checkpoint不重写；source内容、Writer schema与全部76个A/B校验不放宽。旧源文件也不改写，operator对精确函数位点作显式替换，
每个worker经同一入口重装；原算法commit和operator commit同时封存。该临时入口由主agent负责，A完成后已从活动树退役，
其冻结运行树与Git保留复核原件；不复制第二套科学实现，也不恢复旧训练launcher。

实际复核为125/400（原132），breadth均6，S/O/G/L为15/58/41/11；相对原件R/G/L为111/14/21，
churn35、Jaccard .7603，任务配对差额95%CI为[-4.25,+.75]pp。12个workers均正常结束，原始400行及无放回映射完整。
原汇总器沿用旧GPU index→NUMA假设，外层收尾exit1；依据实际PCI拓扑修正这一处汇总判断后，原400行成功汇总，
未重跑模型或闭环。原exit1、全部worker exit0及修正脚本分别保留。证据根为`runs/analysis/v52_return_20260915/`。

## B：现行合同下的fresh基线——已完成实现与profile

保留v5.2的主要能力路径：exact language查询真实视频内容Core，Procedure提供变化条件，Core内容进入Value，
完整A/B由共享family heads生成，两类内容在同一Writer内结合。继续纯同task跨episode FM，source始终冻结。
不默认加入RL、辅助loss、task expert、固定原生span、第二adapter或多套LoRA融合。

新训练必须满足同步双相机、全部50-horizon保留到实际learned read、严格排除同episode action query、fresh Writer/Meta与优化状态。
这些属于明确登记的协议／读取迁移，不能把新模型称为旧132的无变化复现，也不能把迁移收益或损失归因任务分组。
以下接口在B首个更新前登记，再做真实profile；未完成profile与资源检查前不启动formal B。
已有授权允许agent完成这些实施选择，无需再向owner逐项询问。

### B的精确接口与学习合同

1. 原Text/VL/Action三组rank4 Meta、2048→256共享语言投影、视频patch内容Core、两层Core、两层因果Procedure、
   320个rank slots、zero-init AdaLN和八类共享完整A/B heads均沿原v5.2。source及vision/token embedding冻结。
   一条同步双视角视频每时点连接agentview256与eye_in_hand256个真实patch tokens，再接原native task prefix；
   Core按exact text queries读取全部512个patch的真实内容，不把两视角平均或当作两条video。
2. Action Expert仍用公共固定Gaussian probe `[50,32]`、seed `7+0x5A17`、flow t=1；完整原生prefix和18层均保留。
   对每帧最终完整 `H[50,1024]`，learned horizon read使用
   `w_h=softmax(q·RMSNorm_no_affine(H_h)/sqrt(1024)+b_h)`，输出 `sum_h w_h H_h` 后接原1024→256投影。
   共享q[1024]及相对horizon bias b[50]从零开始；初始化为原mean函数，更新后权重可读取所有50个内容及其位置。
   H在这次实际learned read之前不平均／截断；没有新增attention block、动作标签或原生span输出约束。
3. Raw双路RGB、原frame indices、exact token/span是模型接口；frame stride5及真实末帧保持。
   新模型只声称K1；video time采用原v5.2实际执行的原frame indices RoPE（0,5,…及真实末帧），horizon bias只描述动作相对位置。
   训练采用同版本完整LoRA叶节点FM余切→Writer重放，frame chunk内checkpoint重算同时安装三组对应Meta，
   保持全部原生信用。不会跨更新缓存适配Z/KV/H；物理chunk只限峰值内存。
4. Train24每6次更新完整访问全部24task各一次，每更新4个不同task各一条K1／21个queries，等权1/4。
   基线按原sampler seed20260721的逐轮task permutation分为四任务更新，task访问不绑定GPU。
   每task的video seed20260722无放回循环来自demo0–45；action queries同池排除当次teacher episode，
   在独立的确定性episode循环中取21个queries，每个query在对应episode内均匀选合法观测时点。
   demo46–49保留为固定train96独立教学视频；其动作只用于冻结模型下train-only诊断，不产生梯度。
   相比历史全50训练池，这是为独立视频读出明确登记的46/4迁移，不归因架构或共现。
5. 新query遵循已核验的post-action observation→future control：obs[t]监督actions[t+1:t+51]，排除无未来动作的末观测。
   历史v5.2为offset0；此项作为额外协议迁移单独登记，旧125／132不能作为这一个变量的配对反事实。
6. Fresh seed7；完整基础LoRA采用原rank16 A、物理B=0，heads末层=0、Meta B=0；不加载旧Writer或任何诊断拟合。
   AdamW betas(.9,.95)、eps1e-8、weight_decay1e-4、clip1；沿原warmup100、peak3e-4、cosine decay12000至1e-5。
   最大1200只覆盖原学习时间轴的前段，不把cosine压缩到1200。Optimizer/scheduler/RNG/sampler均fresh。
   Native BF16/TF32、完整LoRA float32与SUM梯度；不固定低效batch来追求逐位一致。
7. 在首个formal更新前保存全部4800个(task,visit,teacher,query episode/frame,flow RNG seed)训练事件和baseline更新分组。
   事件由train24规范与固定seed决定；C若准入只重排同轮事件共现，事件多重集合、总更新及全局LR时间轴保持。
   No-held-gradient、严格跨episode、等权、全50读取及完整38-target均由实际profile／入口检查。

### 运行面与退役

现行`train_writer.py`／`materialize_writer.py`和官方evaluator继续拥有训练、完整checkpoint、bank及闭环调度。
`writer/model.py`／`temporal.py`／`video_program.py`恢复原图并承载上述窄迁移；不是新增并行版本。
通用数据、functional LoRA信用、Meta hooks、task负载均衡及checkpoint协议复用。
Process Pullback专属video/correction/factor/native路径和专属测试、配置、检查入口从活动树退役，历史由原Git与冻结运行树保留；
临时A replay入口一并退役。主agent负责runtime／训练／物化集成；隔离agent分别负责模型移植和确定性事件采样，互不写同一文件。
预期新增三份模型职责文件并替代原专属模块，不复制旧训练/evaluation orchestrator；最终检查净增长、依赖与单一路径。

基线拟保留旧曝光尺度：每更新4个等权task，每task一个video条件、21个跨episode action queries；
物理microbatch与GPU数只影响实现。预设观察节点300／600／900／1200，保存每100，最大4,800条件／100,800queries，
其中1200覆盖旧900后明显保持失效的尺度。各节点完整correct400，加固定train24独立视频96条有限面板；
train面板的4条video／task复用范围须登记，不称其为50视频无放回正式验证。
上述初始化、LR/scheduler、teacher/query分池及实际任务事件表在首个训练更新前封存，保持两臂一致。
如实际profile表明此尺度不合理，先用测量依据更新预算，不从验证分数反推学习预算。

实际最长train视频为task38/demo36、105帧。clean pushed `4e26d224`代码上，chunk4与8各完成三次
完整单条件联合FM更新（每次21 queries、权重1/4）；第三次Writer、Text/VL/Action Meta信用全部非零，source冻结，
完整76张量LoRA finite。chunk8后两次平均25.381秒，reserved峰值29.004GiB，chunk4为27.939秒／19.275GiB。
采用frame chunk8、policy microbatch8；四task正式更新在单节点四张合适GPU上分别计算条件后SUM，
不复用profile参数。profile原件见`runs/analysis/v52_return_20260915/baseline_profile/chunk{4,8}/results.json`。
仍保留上述1200更新、四个闭环节点和全部曝光预算；profile不构成能力或正式完整四task更新证据。

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

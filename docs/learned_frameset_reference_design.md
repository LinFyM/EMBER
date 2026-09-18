# A的learned frame-set参照：配对训练诊断

2026-09-18。Owner授权按[专家意见](review_materials/20260918/expert_review.md)设置goal，完成所要求的实验与分析并推送结果。
本轮替代此前暂停状态；这是一个匹配A的训练诊断，不同时启动新架构路线，不重训原A或原始v5.2。
当前登记及执行状态见[progress](../progress.md)。本文件先于新训练和分数登记。

## 1. 问题、唯一干预与完成标准

要回答的是：已有约130–140能力的A，在相同学习条件下，使用给定的视频顺序是否带来额外闭环能力。
已有冻结P消融和Core/P交叉说明该路径可消费有用内容，不等于给定播放顺序必需。
本轮fresh训练一个全部真实帧的集合Writer，与A既有同节点比较；无序模型仍可从状态和物理关系推断操作过程。
本轮无论正、负或不确定结果，完成预定窗口、配对分析、可复核交付及推送即完成goal；不是要求取得特定分数。

唯一科学干预是移除**video-frame order**：

1. Procedure的两层pre-norm Transformer去掉frame-index RoPE和causal三角mask，保留相同Q/K/V/O、FFN、norm及残差。
2. Procedure→slot读取去掉frame-index RoPE，保留原Core条件化query、P中心化、AdaLN、postfusion及八组共享完整A/B头。
3. Core的逐帧真实patch/task-span读取及集合聚合保持A原样；语言token的位置处理、native空间位置、action horizon和flow time全部保留。
4. stride5及真实末帧采样不变；加载器可记录原frame indices用于provenance，但模型数值路径不能使用视频位置或时序mask。
   不通过随机乱序训练构造另一个任务，不删帧、不改图像内容，不把K1变成多视频。

原A参数构造顺序和fresh seed保持；上述删除不增删trainable参数，也不增加随机初始化消费。
输出仍是一套38-target rank16完整LoRA，A模板非零/B为零；三Meta和Writer共同获得纯FM信用，source冻结。
这项有针对性的历史匹配授权使用A的agentview与fixed-mean H50，不将其恢复为未来新方法的永久输入规定。

## 2. 匹配原件与学习合同

原A训练运行：`runs/analysis/source_alignment_20260915/A/training`，authority commit `575c189a743be121bc9801092b0d94b3a53285e6`。
直接复用其`run_contract.json`、`training_events.json`和八份既有`evaluation/*/results.json`；不重跑A。

| 更新 | 既有A correct /400 | 既有A train /96 |
| ---: | ---: | ---: |
| 300 | 99 | 36 |
| 600 | 88 | 47 |
| 900 | 140 | 54 |
| 1200 | 135 | 62 |

- 同一aligned raw1000 source及冻结normalizer；执行offset1、同task跨episode，不恢复旧错位。
- train24、teacher/action0–45、独立诊断46–49；每更新4task×1video×21queries，全局84，task权重各1/4。
- seed7、sampler seed20260721、teacher seed20260722，baseline grouping，事件顺序及每task访问相同。
- AdamW lr3e-4、betas(.9,.95)、eps1e-8、weight decay1e-4、clip1；warmup100、decay12000、floor1e-5。
  scheduler按12000构造，不能因窗口1200而压缩优化时钟；Writer/三Meta/optimizer/scheduler/sampler/RNG全部fresh。
- 1200更新共4800视频条件、100800动作queries，约每task200条件；每100保存完整checkpoint。
- native BF16、Writer/FM LoRA FP32。原A使用四卡、native chunk8、policy microbatch8。
  物理分片或microbatch可由真实profile选取，但不得改变逻辑batch、flow随机样本、task权重、更新次数或优化时钟。
  六卡若使用，只能沿现有两组三卡的真实帧/query分片；不靠重复计算或dummy占卡。
- flow随机性按task、visit、完整21query序列派生，切片保留logical batch和offset；设备分配不参与任务采样。

训练正式窗口固定到1200，节点300/600/900/1200；不按新分数临时挑点、换seed或延长窗口。
工程错误、资源安全问题应修复或精确恢复，不将无效运行计入科学结果；没有结果驱动的第二轮架构实验。

## 3. 配对评测与裁决

各预定节点完成single-checkpoint correct validation400及独立train96。使用A原requests中的selection：
validation8为task1/3/11/13/23/26/31/32、init0–49、teacher0–49每task各一次、seed20260911、per_init_ordinal；
train24为init32–35、teacher46–49的同一固定有限面板。train96不同于9月18日teacher46复用面板。
执行source、language、state、env/policy RNG、official preprocessing、10 flow steps及前5 actions均匹配。
物化和评测复用当前动态cost-balanced queue、long-first及persistent workers；新root不能覆盖A原件。

主解释预定900和1200相邻节点；300和600给出获取轨迹，不从四点挑最好的一点证明结论。
每点报告两模型per-task/per-suite、breadth、Goal/Long、A→frame-set的retained/gained/lost/churn/Jaccard，
以及每个模型内部相邻节点的保持。差值区间采用task-cluster bootstrap、20,000次、seed20260915、95%CI。
同时保留逐task结果，明确只有8个validation任务的推断限制；总分接近或区间含零不证明等价。

- A在900和1200持续有广泛净优势且与覆盖/保持一致：支持此匹配配方中时间处理的行为价值，不等于已理解全部语义顺序。
- frame-set达到相近能力且没有一致的A优势：支持现有高分不需依赖给定播放顺序的可能性；不能说视频无用或两模型等价。
- 节点/任务分化或大范围成功交换：保留不确定性，说明差异发生在哪些任务和学习阶段，不强行二分。

本轮不训练learned language-only、不重跑SFT、不新增wrong/shuffled/reversed/no-video闭环，不以这些controls选点或修改架构。
结构测试中的合成帧置换用于检查集合不变性，不能冒充视频因果性能结果。
不使用Test、validation/test梯度、RL或额外meta tasks。最终交付只给本轮能支持的下一步含义，不自动启动后续方案。

## 4. 实现、验证与生命周期

Canonical owner仍为`writer/model.py`、`temporal.py`、`video_program.py`和现有training/materialization入口。
恢复A已验证内容路径，替换已结束的统一图；统一图源码和证据由Git `1b9c38de`及原formal artifacts保留，不保留平行fallback。
新schema明确标识frame-set诊断，避免把同shape的旧A checkpoint当作新模型加载或继续训练。

验证范围：非零内容/已打开调制的Procedure等变、完整编译不变、frame index变化无效、语言位置仍有效，
以及真实原生读取/三Meta信用、全38-target身份初始化、功能梯度、完整恢复和必要物化合同。
匹配事件计划按原件逐字段核对；不做全树hash或逐tensor精度扫描。
保留原生帧分片时须确认不等长/空分片及汇总梯度语义，profile使用真实最长视频和真实跨episode FM。

正式train/eval来自clean pushed detached runtime。profile状态不用于formal初始化；fresh正式训练完整保存恢复状态。
训练段结束再完成该节点评测；后续段原拓扑exact-resume。正常运行后台执行，不逐分钟播报或按分数改变合同。
launch前用gpu-preflight同时检查两节点并遵守总GPU上限，实际UUID/NUMA、命令和日志记入本地launch contract。

## 5. 存储与远程交付

新study计划放`/data0/user/ymdai/ember_runs/a_learned_frameset_20260918`，仓库`runs/analysis/a_learned_frameset_20260918`只作symlink入口。
复用source、数据、tokenizer及环境；完整窗口预估新增峰值不超过20GiB，包含12个恢复点、四节点LoRA物化、raw rows及profile。
2026-09-18规划检查：strg01报告data0 user已用122283960KiB、data1为922233368KiB，soft各1073741824KiB，
两者独立；共享data0仍有约1.3TiB，data1约83TiB。正式launch前刷新实际余量并封存估计。

远程交付包含专家原文、本合同、唯一实现与相关验证、逐task/相邻配对表、结构化汇总及足够复核计算的精简raw成功行。
不上传模型权重、数据集、巨大二进制或host-private配置。真实完整原件留本地study，公开结果保留原件路径与代码provenance。
更新README、findings、research_history、progress/task_plan，提交并确认远程main同步后完成goal。

## 6. 实现审查记录

CPU验证覆盖集合不变性、非零内容梯度、完整76张量identity及非identity编译、原生帧分片、
事件采样、optimizer时钟、恢复及物化合同。原有两条camera负例由agentview改为dual，匹配本轮的合法输入口径。
结构检查相对1b9c38de为净增534行、无新源码文件；主要是原A内容路径回归替换统一结构。
保留原A构造器和模块顺序以维持初始化RNG，故不为行数机械拆分；原生读取、时序模块、完整LoRA输出仍由原owner负责。
复杂度提示中的物化配对与恢复校验没有新增职责；统一图已退役到Git，没有第二条可运行fallback。

Profile后物理执行登记：三卡、每卡FM chunk8、native chunk8；每更新仍4条件/84queries。
105帧最长条件与三卡6更新通过，含第4步完整恢复；详见[profile摘要](review_materials/20260918/profile_summary.json)。
原A和本轮4800训练事件完全相同。当前只有三张低利用率设备适合共驻，因此不等待六卡、不改采样或更新时钟。

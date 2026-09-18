# 经最终LoRA的同视频教学：模型与训练合同

2026-09-19。Owner明确授权设置goal、完成[专家最终修订](review_materials/20260919/expert_proposal.md)、推送并汇报。
本文先于新候选训练与分数登记。最后一轮修订覆盖初始提案中的P-only教学梯度；整个Writer与三个Meta共同学习。
Owner随后明确主实验先用单相机agentview；单相机结束后才据具体证据判断双相机是否值得补做，不自动增加实验。

## 1. 问题、依据与完成标准

工作假设：现有跨episode FM能学出有效任务条件参数，却未稳定要求这些参数消费所看视频的具体操作证据。
用同一套生成LoRA解释教学episode实际发生的短程动作，同时保留完整跨episode迁移监督，可形成更直接的功能信用。
这不是已定位的唯一根因，也不保证视频顺序必要、未见初始化迁移或保持。

保留A的完整自由A/B、Core条件化的P读取、中心化/AdaLN、归一化与八组共享head，以及四任务逻辑更新。
Core是当前计算接口、内容起点和过程查询，不再宣称它是已经证实不可替代的正确视频内容主干。
旧P的冻结贡献不能推出新增教学信用只应给P。完整H与反复视觉读取在Horizon/v6已有先例，不当作本轮独有突破。
与Local Action的实质区别是教学误差穿过实际部署LoRA/策略；与Native Correction的区别是无参数标签回归或受限native出口。
历史细节由证据审计、research_history与findings§117–120约束，不把局部负结果升级成所有相似方法不可能。

完成标准：完成候选实现、真实profile和完整1500更新窗口、预定闭环与配对分析；有保留价值时完成唯一监督消融与冻结后controls；
保存可复核结果并推送main。科学结果允许正、负或不确定，不要求以特定分数结束goal，不无限续训或扫描。

## 2. 唯一部署图与模块职责

```text
exact language + action-hidden完整agentview教学视频，K1，stride5+真实末帧
 → 冻结aligned source + fresh Text/VL/Action读取Meta
 → 逐帧图文E[t,language-token]与完整原生H[t,1:50]
 ├→ 原有Semantic Core
 └→ H均值投影初始化P[t]
      → 重复过程块×2：完整H50内容/位置读取 → 当前/下一帧真实E读取 → 有序causal RoPE交互/FFN
 → Core条件化读取居中P，AdaLN融合、槽间处理和归一化
 → 八组共享完整A/B，唯一38-target rank16 LoRA
 → frozen source根据机器人当前观察执行，rollout期间不重读teacher或优化
```

新增两项读取是零输出初始化的残差；完整H先保留全部50个位置至实际可学习K/V，不以mean代替该路径。
初始P允许沿用旧H均值投影。相邻视觉保留两端完整内容及先后角色，不限制Value为差分；末帧无下一帧时明确mask。
视频帧位置、语言位置、图像空间位置、动作horizon位置、flow time各司其职，不混淆。
默认宽度256、两块、八头，复制同类过程块可加深；本轮不做深度/宽度/rank扫描。
新增残差可表达旧P函数，不保证训练保持旧能力。仍保留三组Meta和原头的合法identity初始化。

## 3. 数据与两项功能损失

source复用`pi05_source_aligned_seed7_1k_20260915/checkpoints/step_00001000`，不重训A或原v5.2。
训练仅固定train24、episodes0–45；46–49仅冻结诊断，validation/test无梯度。source normalization冻结。
每次仍采四个不同task，每task一条teacher，21个严格跨episode主query；原A的主采样/RNG事件前缀保留。

每条件另采7个同教学episode query。合法观测索引p满足p为stride5采样位置且p+5<N；
监督真实actions[p+1:p+6]，不重复尾动作冒充五步。短于五步的尾间隔不作为教学query，最后RGB帧仍供Writer读取。
从合法p中均匀无放回取7个；若合法p不足7则登记有放回抽样，不删帧或改动作。采样使用独立固定seed流。
query正常使用两执行相机与obs/ee_states、obs/gripper_states；这些正常执行输入不进入Writer。

先由RGB与exact language生成一次完整Delta，再计算：

L_step = (1/4) sum_task [ mean_21(main_FM_cross_episode) + (1/3) mean_7(same_video_prefix_loss) ]。

主FM使用官方50-step horizon与原Beta flow-time采样。教学项固定tau=1、完整50×32独立Gaussian noise；
仅前五步、真实7维动作上平均MSE，预测a_hat=noise−velocity，等价于该前缀velocity对noise−action的MSE。
不能把两组112个query混成一个均值；1/3为首版固定权重，不等于由查询数证明梯度已平衡。
两项LoRA余切相加，再重放Writer一次，将信用传入全部Writer和三个Meta；无P-only mask、辅助动作head或第二adapter。
teacher动作/state仅作训练标签/执行query；在Writer生成前不读取标签，不按query位置重编译，不让身份、文件名或结果进入部署条件。

## 4. 优化与硬件

全部fresh，seed7，AdamW lr3e-4、betas(.9,.95)、eps1e-8、wd1e-4、clip1。
前900更新沿用A的warmup100/decay12000/floor1e-5时间函数；随后600更新按cosine衰减至900节点学习率的0.1。
1500硬终点；共6000条件、126000主query、42000教学query。LR尾段是预注册选择，不宣称已修复遗忘。

四task均在同一phi_t上计算，最后global SUM、一次clip/Adam/scheduler。1–4卡逐条件放置；5卡用3+2组、6卡用3+3组。
组内可分原生帧与主/教学query，完整H/E可微gather后才做全视频过程计算；按各自21和7分母加权，禁止额外除world size。
物理microbatch/frame chunk按真实吞吐与最长视频profile确定，支持每rank不同microbatch；不改任务权重或科学batch。
不承诺任意小显存可放下source，不为本轮引入参数分片。exact-resume锁profile后原拓扑，不添加跨拓扑迁移。
launch前同时检查两节点，按现行总卡数/单节点限制选真正有吞吐收益的资源；正式代码来自clean pushed detached worktree。
资源与storage实测在launch记录登记，profile权重丢弃。

## 5. 预定观察与判读

完整checkpoint每100更新；formal主要节点900、1200、1500。各节点strict validation correct400与train correct96。
冻结train24 held-action FM在0/900/1200/1500记录，不选点。预留旧A同点140/135/112（validation）与54/62/56（train）作历史配对参照。
state-video映射使用旧A seed20260911，validation每task50条合法teacher各一次；train沿用states32–35、teacher46–49有限面板。
A与本候选同时改变过程读取、同视频目标和尾段LR，因此不是单变量归因实验；唯一额外监督来源消融才隔离教学配对。

报告逐task/suite、breadth、R/G/L、churn、相邻success-set交集与task-cluster bootstrap95%CI（20,000次，seed20260915）。
不恢复历史>145门槛、不用union/融合/偶然峰值，也不以局部loss好看接受明显较差闭环。
若没有两个相邻节点均达到130/400，则本轮未证明可保留的相邻能力，不增加消融或新架构；仍完成分析交付。
该工作线用于触发进一步证据，不把它称作科学等价界限。

满足上述工作线时，补这些候选相邻节点的same-task-other400（逐行另一视频、整轮各50条无放回）。
以每个相邻对两点correct/other四个分数的最小值排序，平局取更晚相邻对；再以该对内min(correct,other)较高者冻结，平局取更晚点。
同时报告覆盖及R/G/L，若收益明显只来自轨迹绑定或大范围能力替换，不以排序掩盖该失败。
selected固定后才做paired wrong/shuffled/reversed/no-video；真实重排frames后完整生成，不用于loss、选点或改架构。
正确与换视频能力保持/提高是主要进展，controls下降本身不算教学收益。Test保持关闭，无RL。

## 6. 条件性唯一消融与双相机裁决

主候选有上述相邻correct能力且换视频不出现明显崩溃时，最多增加一个匹配fresh消融：
相同图、相机、seed、21主query、7额外query、tau1、前缀5、lambda1/3、全部联合梯度和LR；
只将7额外query换成同task另一episode（排除teacher），从该episode相同规则合法stride5位置采样。
固定同一1500窗口和节点，与主候选逐行比较。这区分具体教学对应关系与额外动作监督，失败时不追扫权重/seed。

Owner授权单相机结束后自行判断是否补双相机；默认不补。只有具体证据指向视角遮挡/接触信息缺失，
且增加相机有独立于失败候选整体重构的合理收益预期时，才在启动前追加唯一双相机合同与资源预算；不得因低分自动加一臂。
不同时开其它新架构、不重训历史A、不以相机变化冒充同视频目标的因果证据。

## 7. 代码与证据归属

沿用`writer/{model,temporal,video_program}`唯一模型入口和`training/supervised/function_credit/learning_data/task_execution`训练链。
frame-set实现退役至Git b9bd90ac、封存合同和formal artifacts，不保留平行fallback。
模型子任务在独立worktree；主代理负责训练、数据、配置、文档、最终集成验证与运行。保留完整checkpoint和raw eval资产在本地，
远程推送源码、合同、专家修订、结果分析及精简配对证据；不会把本地权重或数据集加入Git。

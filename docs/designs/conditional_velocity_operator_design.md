# 条件速度算子：共同状态基与单LoRA编译候选

2026-09-27。科学依据与最近完整历史比较见[机制分析§29](../analyses/feature_to_operator_mechanism_20260926.md#29-共同学习状态反馈基直接编译条件速度场一个有明确代价的完整候选)。
这是候选及其有界工程合同，不是已验证修复。是否实际启动只看progress；本文没有正式学习或held评测许可。

## 1. 假设、输出与信息墙

完整假设：可从合法教学学得任务相关系数R(C)，与共同学习的自身状态函数U h_β(x)组合，形成有用的条件FM速度场。
β提供完整共享LoRA，基础source冻结；β、视频encoder、R读出和U全部fresh、同一真实跨episode FM共同学习。
不是先训/冻结MT-BC，不加载其权重，不建立Reader教师，也没有语义/纠正/几何辅助loss。
“共同反馈基+条件系数”是受限的完整方法，不能由恒等式保证视频被理解、held泛化、绝对能力或保持。

输入仍为exact language和一条完整agentview RGB教学，stride5及真实末帧；K1，不声称Dynamic-K。
Writer不读teacher动作/state/reward/terminal/pose/task ID/文件名或query。动作和query自身state只用于训练policy。
source、normalization、tokenizer和预处理复用现有canonical资产，不复制。部署自身双相机/8维state/7维动作合同不变。

输出一套完整38-target LoRA；共同部分β的rank128，视频部分仅位于action_out、有效rank≤7。
统一部署rank135/alpha135，其余37目标以零算子补齐额外7rank。它们是公共参数，不能声称逐层都由视频改变。
rollout前生成一次，无第二adapter、运行时教师memory、额外控制器或task-local优化。

## 2. 固定的完整图

```text
L + 完整RGB → 原生Text/VL/Action读取及各rank4 Meta → q / E / 完整50×1024 H
           → 现有LanguageSemanticCore C与RecurrentProcedure P（width256、各2 blocks）
           → 七个动作维度query读取C和P的真实Value → R(C) [7,256]

自身真实query + noise/time → source + 自由共同rank128 LoRA β → h_β [50,1024]
U [256,1024]与R(C)相乘得到M(C) [7,1024]
共同action_out + E7 M(C) → 原生32维FM速度（真实前7维监督）
```

教学native读取不装公共β，只沿现有fresh三Meta读取合法prefix/fullH，避免在本候选又增加参数回读因果变量。
复用既有q/E/H、完整帧打包、Core与Procedure的所有实际读取规则；不恢复初帧广播、状态补全、时序重排或额外先验。
Core不启用`language_content_path`。P保留其原始内容/顺序，不使用旧Compiler的中心化调制。
不实例化旧FactorHeads/Compiler后丢弃其输出来取得memory。

七个learned query仅为动作行的检索身份，shape[7,256]、正常非零初始化；不是task身份。
用现有ContentCrossAttention（width256/8heads）先读C得到c，再以query+RMSNorm(c)读P得到p。
C读取无时间RoPE，P使用真实frame positions的既有RoPE；mask沿用真实有效token/frame。
两个Value都来自真实C/P。令d=GELU(Wc c+Wp p)，Wc/Wp为无bias正常初始化的256→256线性层。
R=Wo d，Wo为无bias、零初始化256→256线性层，各行动作行共享这些映射；query本身不旁路进入d的Value。
不增加softmax层位/深度/中心化对照，不由profile分数选择这套读取。

U为无bias自由线性投影，非零fan-in初始化；它不是fixed source/PCA span，也不接受独立标签。
β为自由完整LoRA因子、A非零/B零identity；全部参数fresh，seed7，公共identity seed沿canonical contract。
U、R及β的初始化顺序和RNG需在机器规格中固定，完整恢复保存实际RNG；不要求复现旧MT-BC的逐元素随机轨迹。

设E7=[I7;0]∈R^(32×7)，实际合成：

```text
out: A=[Aβ; R U], B=[Bβ,E7]
others: A=[Aβ;0], B=[Bβ,0]
```

公共和条件算子相加，无同rank因子交叉项；生成state只含76个canonical因子。
初始R=0，虽然out额外B=E7非零，BA仍为零。source物理参数requires_grad数始终0。
训练按现有完整LoRA cotangent/VJP接回β/U/R/encoder；共享β在各condition下的梯度正常求和，不重复计权。
真实7维、完整50-horizon、随机noise/time普通FM；不启用端点/前5辅助、KD、RL或回滚。

## 3. 最接近的失败与候选边界

旧Video Functional的末端query来自未挂LoRA的冻结source，reader与encoder联训，但query特征不学；有另一个LoRA学生。
本候选共同训练执行特征且从第一步就是可精确编译的单policy，没有强教师前置条件。
旧LocalField已让同一U/r受真实纠正标签并组成LoRA；本项不以“直接消费者”作为新理由，改变的是实际条件速度场和共同状态基。
旧公共prior/Unified/DJNFR反对“加公共路径自动保持”；β自由且fresh共同学习并不能独自消除这条反例。
当前完整Writer/private正例反对“生成全层本来就不可学”。这里主动收窄条件函数，可能牺牲所需早期条件读取。

可失败预测：若该共同状态基能承接可迁移教学关系，完整模型应取得强共享参照以外的正确/换正确视频能力，
而不靠大范围丢失和单个峰值；若主要获得公共能力、只降低FM或R有变化而没有正面视频收益，假设不通过。
训练条件不同导致的总分差不能唯一归因于本参数化；正式比较要锁同一数据/评测协议，并如实保留强MT-BC。
若完整有信息量窗口失败，关闭本组合，不逐层加入条件出口、不扫rank/辅助/冻结公共分支继续维护它。
未测功能基的可用性，不用“未试过”或常量可达性称其已获科学资格。

## 4. 当前仅可派发的有界工程

唯一执行者：`01a0dd6c-f2e5-7971-821a-56766e1c0f22`。
主讨论负责科学裁决、机器规格的实质变更和后继学习合同；执行者不得自动进入正式训练或held。

**时间预期**：CPU实现/核对45–60分钟，GPU部分约12–25分钟wall；本批总计约60–90分钟，120分钟为工程判断上限。
这是待实测的工程估计，不是训练速度证据。明显超时或资源不足时回报具体完成面与阻碍，不继续堆叠实现或扫描。
GPU硬限0.75完整GPU-hour，含加载、失败、恢复、profile及case；上限优先于任何预定步骤。最多同节点2张物理卡。
旧Reader的15.53秒宏步不是本候选profile，不能直接据它启动后继长训。

**存储**：所有新增在data1。study root为
`/data1/user/ymdai/ember_runs/conditional_velocity_operator_engineering_20260927`。
原件峰值上限4GiB，开发+冻结代码峰值768MiB；创建前由执行者在strg01核data1独立quota及共享容量，并按实际参数/保留checkpoint估算。
不得新增data0写入或复制canonical大资产。未通过容量核算不得创建大run root。

**当前工程梯度只使用固定train条件**：global2/12/22/32，各teacher demo0；query从同task demo1–49取，严格跨episode。
四task每宏步各28query、等权，总112；固定sampler seed20260927，沿既有offset1与50步chunk规则。
这四task均为coverage train24，且不涉及当前diagnostic-held8；profile还允许coverage train的global29/demo0。
不读任何official validation/test动作/state标签，不把工程case列入科学比较。
工程优化仅用于图/恢复核验：AdamW lr3e-5、betas(.9,.95)、eps1e-8、wd1e-4、clip1、8步warmup后固定；所有参数同一loss。
该短跑配方不预设正式学习LR或曝光窗，profile loss不参与后继超参选择。

CPU核对：完整因子合成/identity、实际7维嵌入、参数梯度消费者、checkpoint字段、官方接口与信息墙。
需要的是少量有真实失败含义的检查，不构建另套测试框架或逐tensor/逐bit验收。
实现必须先clean pushed，GPU使用detached frozen树；启动前按项目规则同时live检查两个GPU节点、NUMA/NCCL和全项目用卡数。

GPU允许范围固定为：

1. 一个fresh world2的4宏步，保存2/4；从完整2恢复到4，额外2宏步。共6实际更新/672训练query，不重跑fresh。
   核验真实梯度、source冻结、共享β计权、恢复sampler/optimizer/RNG/topology和日志前缀；允许正常浮点差异。
2. global29/demo0的完整合法长视频，一次28-query反传profile，无optimizer更新。原记录347帧/stride5含末帧71帧，
   实际资产若不符则先核身份/配置，不按长短或loss改选视频。
3. global2与global32各state0/teacher0一次train-only canonical episode，共2条；2全RGB、32 compact，保存真实T+1动作/body/EEF/gripper/BDDL接口。
   成功与否只作原始smoke行，不能称科学收益、不能据失败补episode或改模型。

默认真实query microbatch28；仅为OOM/已证吞吐问题调整物理microbatch，不改变112逻辑query、task权重或精度合同。
不增加语言臂、冻结β臂、局部oracle、policy probes或最终顺序controls。不得以工程throughput差恢复旧Reader。
结束后核全部退出状态、原件、实际资源计费和自身进程清理；只回报一次完整完成/实质阻碍。

## 5. 工程所有权与后继裁决

优先复用执行者已空闲的适合隔离树；从交付的main commit开始，分支`codex/conditional-velocity-operator`。
主讨论只改main文档，执行者不改main。架构技能由执行者按实际结构应用，不把其review signal变成人工批准关卡。
视频/native输入与C/P拥有一个共享owner；条件R/U/β合成拥有一个owner；训练/恢复与canonical case尽量复用现有接口。
不能创建旧Writer/Reader副本或新通用runner框架。允许工程分支暂有一个明确命名的有界入口，执行者是owner。
科学否决时由Git保留并退役其活动入口；若进入正式学习，须先完成canonical替换/旧输出路径退役及main集成，不留下平行fallback。
源码超过500新增行或3新文件须按skill完成代理自审，报告主要所有权/增长与退役触发；不机械拆文件。

正式学习尚未授权：工程回报提供真实秒/宏步、query吞吐、最长峰值、物化和两类case墙钟。
主讨论据此在学习前冻结数据/采样/学习时标、首段最多两个节点及完整比较预算；不从4步loss选checkpoint或决定配方。
正式选择仍须single-checkpoint strict paired400、相邻保持及same-task-other；小screen只作有界早期读出，不能选模型。
当前coverage强MT-BC与C0 diagnostic-held属于不同合同，不能拿155−120解释本候选，也不重跑旧实验补排列组合。
测试集和最终shuffled/reversed保持关闭；本合同不允许自动蒸馏、第二架构、长训或其它实验。

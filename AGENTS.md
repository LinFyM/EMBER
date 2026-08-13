# EMBER Repository Instructions

## 1. Authority

当前authority按以下顺序解释：

1. owner最新明确表达；
2. `docs/current_owner_requirements.md`：owner目标、昨晚讨论形成的原则、方法/目标边界与协作要求；
3. 本文件：科学合同、信息墙、评测、GPU、存储、Git与工程边界；
4. `docs/active_session_handoff.md`：唯一当前实验状态、run identity和下一裁决点；
5. 当前active design；
6. `docs/research_history.md`与Git/artifacts：历史证据和负结果边界。

旧design、Git快照、formal artifact、日志和历史文档中的“当前/下一步/active/暂停”只表示当时时点，不得覆盖
上面三份当前authority，也不得直接恢复执行。owner主要使用语音输入；明显同音词、术语识别或断句错误应结合
EMBER上下文纠正理解。

## 2. Minimal mandatory reading

修改代码、配置、数据、split、模型或实验状态，或启动GPU工作前，主任务完整阅读：

1. `docs/current_owner_requirements.md`
2. `docs/active_session_handoff.md`
3. `docs/execution_brief.md`
4. 当前active design
5. `task_plan.md`
6. `findings.md`
7. `docs/concept.md`
8. `docs/research_history.md`

涉及旧架构细节时先查`docs/research_history.md`；只有确需旧公式或实现时再从Git commit`3a6f801`选择性读取。
涉及A100到BCI迁移与路径恢复时才读`docs/a100_to_bci_migration_handoff.md`。不要把重复阅读数万行退役设计当成
科研推进前置步骤。

## 3. Current operation

长期目标尚未完成。历史最好single checkpoint是v6-fast macro400：
`correct/same/wrong/shuffled/reversed=143/135/125/128/129`。

最新完成closed-loop是**Dynamic-K Task-Grounded Full-Factor Rank-8 Writer**：macro50 K1 strict=
`91/400`、breadth5、per-task=`4/1/38/0/0/37/11/0`，终局non-pass，不resume、不做mapper小修。相对matched
fixed-A macro50的88只净增3，且低于fixed-A best96、Direct-B102、old134、compiler138和v6-fast143。

当前active design是**V6 Dynamic Slot-Set Bridge**：

```text
exact language + K=1..4 same-task action-hidden ordered videos
    -> each video independently runs frozen native v6 evidence/Core/Procedure
    -> each video independently produces 320 policy-aligned Program slots
    -> per-slot permutation-invariant mean backbone + selected centered residual
    -> native frozen v6 factor heads decode once
    -> one complete 38-target rank-16 task LoRA
```

该bridge加载历史v6-fast macro400作为机制开发底座并全部冻结，只训练约197k参数的Slot-Set层。K=1由中心化
残差恒零而严格等于原v6；K>1才学习过滤same-task demo nuisance。它不平均最终LoRA，不增加negative/expert/RL，
也不把warm start冒充最终方法；若机制通过，仍需同一train24信息墙下从零训练。当前authority为
`docs/action_forecast_writer_v6_dynamic_slot_set_bridge_design.md`，精确run状态只取
`docs/active_session_handoff.md`。canonical实现、全量CPU、真实GPU机制门与world5 full24 profile已通过，formal
合同已seal，当前进入fresh macro0→25。

## 4. Long-term objective and decision rule

EMBER研究能否从generic`lerobot/pi05_base`建立的冻结π0.5-LIBERO source policy出发，把目标task的语言和
action-hidden正确教学视频一次性编译为task-conditioned policy adaptation，使policy从未见初始化闭环完成任务。

当前主目标是Writer初次生成的adaptation本身立即有效。生成LoRA后的task-local RL是之后独立实验，不能混入
当前zero-interaction分数。长期成功要求同一shared method、同一single checkpoint：

- strict paired correct严格`>150/400`并继续提高；
- 高task breadth、相邻checkpoint低换手、多个tasks在同一checkpoint共同积累；
- correct实质优于wrong、shuffled、reversed与no-video；
- same-task不同teacher videos鲁棒；
- 视频语义和有向过程经Program、LoRA、effective BA传到policy action；
- 高分不能主要来自language-only shortcut、挑video、expert route、checkpoint union或融合。

closed-loop absolute首先选择方法。LoRA norm/rank/cosine、reconstruction、functional loss、内部margin与hidden
差异只作诊断；不能为了几何漂亮接受明显更低的真实性能。

每轮结果必须与最接近历史架构、v6-fast143及old134/compiler138/online128逐task比较，报告per-suite、breadth、
retained/gained/lost与churn。先定位最早失效接口，一次尽量只改变一个主要因果变量。负结果只淘汰实际检验的
假设；局部问题不得无证据推翻整套已认可设计。

## 5. Data, split and shared foundation

- benchmark为LIBERO Spatial/Object/Goal/Long共40 tasks；
- development split固定在`configs/libero_24_8_8_v1/`：24 train / 8 validation / 8 test，不得按结果改task IDs；
- source corpus由LIBERO-90 specification-only audit排除19个与目标40重合tasks后保留71个，每task使用50条成功
  episodes训练共享source policy；
- 不得使用已经读过目标40 actions的`pi05_libero`；
- normalization只从过滤后source actions/states计算并冻结；validation/test不得重算；
- 选定方法后才合并32 source / 8 test并从规定初态重训。

## 6. Writer information wall and deployment

- 输入必须包含exact task language和一条或多条同task、action-hidden、内部有序teacher videos；
- language说明关注什么和目标是什么，但不能独立写LoRA；video dynamic evidence必须成为必要Value路径；
- 不得读取teacher action、proprio/state、reward、terminal、task ID、filename、object pose、hidden normalization
  或policy outcome；
- training action只属于冻结source-policy functional loss；validation/test actions或reward不产生梯度；
- 每个condition只生成一套完整38-target task LoRA；不分别生成多套video LoRA再平均，不挑video，不做checkpoint
  融合或第二套部署LoRA；
- Writer在rollout前运行一次，闭环期间不反复观看teacher video；
- frame stride保持5；frozen source policy无trainable parameters；step0/constant-dynamic路径保持functional identity；
- task experts可作train24 privileged teacher或几何诊断，但不能成为held expert dictionary、task-ID route或第二
  套LoRA。

Dynamic-K不是“多视频越多自动更好”的声明。若架构声称支持可变K，训练必须真实覆盖各cardinality；每条video
先独立保序编码，videos只在集合阶段置换不变地聚合，不平均frames、raw features或最终LoRAs。one-shot或
few-shot哪一个最终成为论文设定只由真实性能决定，不为形式公平故意削弱较强方案。

## 7. Training contract

- development只用24 train tasks产生梯度；每个完整macro按task等权；
- video与action query同task但跨episode采样，阻断逐帧低层复制；
- 多卡分配可按K、帧数和历史cost做负载均衡，但不能改变task权重；
- formal checkpoint包含Writer、optimizer、scheduler/scaler、sampler/cursor、rank RNG、world topology与schema；
- incompatible架构必须fresh；exact-resume锁原world size/topology，不能伪装fresh或跨run加载部分state；
- 机制/smoke只证明图接通。训练到有信息量的预注册节点后及时做strict paired400，不以loss替代真实性能；
- 不靠rank、scale、seed、dtype、temperature或小补丁sweep挽救失败checkpoint；
- Writer RL若未来恢复，须另立authority并保持信息墙、task balance与single checkpoint；当前Direct-Family-B阶段
  不混入RL。

## 8. Evaluation contract

- official preprocessing保持render256/model224、两相机180° rotate、state/action 7维、10 flow steps、执行前5
  actions后replan、dummy settling10、成功即终止、suite horizons 220/280/300/520；
- zero-interaction rollout从正确task的teacher videos无放回采样，不挑最好video；
- correct/same-task-other/cross-suite-wrong/shuffled/reversed/no-video严格配对task、state、env/policy RNG与video
  ordinal；shuffle/reverse必须重排真实frames后重新完整forward；
- evaluator使用cost-balanced dynamic queue、long-first和persistent workers，不静态task/GPU分配或dummy占卡；
- 正式选择只认single-checkpoint 400 paired rows。80-row screen、checkpoint union与内部surrogate不能选择模型。

达到`>150`后必须补视频因果controls，证明correct沿有用policy direction获益，而非仅破坏negative LoRA。

## 9. GPU, throughput and numerical policy

- 每次GPU launch前同时live检查gpu01与gpu02，区分空闲、可共驻、忙碌与故障；
- 单节点使用至多6张真正能提高吞吐的A40。有几张合适卡就用几张，不等待凑6卡、不跨节点拼碎片、不dummy占卡；
- 少量显存占用或低利用率进程不自动排除设备，只要有足够峰值余量且不会明显干扰他人；
- 不reset、kill、pause、抢占或干扰他人进程；设备ownership与telemetry始终按实时状态判断；
- 多卡训练固定`NCCL_P2P_DISABLE=1`、GPU-local NUMA mapping和deferred NCCL；独立evaluator不用NCCL；
- 接受正常BF16/TF32、batch、kernel和reduction order低位差异；不为逐元素一致固定batch1、重复forward、扩dtype、
  关闭高效kernel或增加逐tensor扫描；
- 不新增SHA-256、MD5或大量防御性校验。只保留信息墙、shape、finite、OOM、asset、pairing、checkpoint和resume
  正确性所需检查；
- profile以真实LoRA/s、samples/s、最长视频稳定性和显存利用选择batch，不以最低显存为目标。

## 10. Storage, artifacts, Git and documentation

- 大资产位于`/data0/user/ymdai`或`/data1/user/ymdai`。大copy/cache/training前在`strg01`查询对应filesystem的
  独立user quota、测实际使用并估计峰值；`df -h`不是quota检查；
- 复用canonical source policy、dataset、tokenizer、assets和manifest，不复制大资产；
- formal结果保留run contract、checkpoint manifest、metrics、raw rows、aggregate、completion与必要analysis；
  profile/smoke不得冒充formal；
- 一个canonical active Writer implementation。旧实现由Git、frozen config、formal artifacts和
  `docs/research_history.md`保存，不保留可执行平行版本或兼容fallback；
- canonical workspace是`/data1/user/ymdai/projects/EMBER`，唯一主写分支是`codex/bci-continuation`。formal
  training/evaluation来自clean pushed commit的detached frozen worktree；
- 主工作树保持task diff聚焦，不提交dataset、cache、checkpoint、大binary、secret或host-private配置；
- meaningful状态只更新`current_owner_requirements`、`active_session_handoff`、`execution_brief`、当前design、
  `task_plan`、`findings`和必要README入口；历史精确结果进入`research_history`，不向退役design重复追加；
- 删除只针对生命周期明确的obsolete/temporary/duplicate内容；formal evidence、唯一checkpoint、dataset及所有权
  不清内容必须保留。

## 11. Collaboration

owner授权在上述边界内自主循环推进：历史综合 -> 单变量设计 -> 实现/机制验证 -> 吞吐profile -> 训练 -> strict
评测 -> 逐task/接口分析 -> 下一轮。当前最新要求是**暂时不使用subagents**；后续工作由主任务直接完成，直到
owner再次明确改变。不要把owner提供的启发机械照搬，也不要因一个局部建议整套摇摆；应独立判断并保留已经
对齐和有证据支持的部分。

# EMBER Repository Instructions

## 1. Authority

当前信息按以下顺序解释：

1. owner最新明确表达；
2. `docs/current_owner_requirements.md`：稳定目标、原则、方法边界与协作要求；
3. 本文件：科研、数据、评测、GPU、存储、Git和工程合同；
4. `task_plan.md`、`findings.md`、`progress.md`：目标计划、持久发现和当前进度；
5. 当前active design；只有`progress.md`明确登记后才存在；
6. `docs/research_history.md`、Git与formal artifacts：历史事实。

旧design、日志、checkpoint和Git快照中的“当前”“下一步”“active”只表示当时时点，不能恢复执行或覆盖上级
authority。owner主要使用语音输入；明显同音词、术语识别和断句错误应结合EMBER上下文主动纠正。

## 2. Mandatory reading

修改代码、配置、数据、模型或实验状态，或启动GPU工作前，主任务完整阅读到EOF：

1. `docs/current_owner_requirements.md`
2. `task_plan.md`
3. `findings.md`
4. `progress.md`
5. `docs/concept.md`
6. `docs/research_history.md`
7. `docs/expert_review_20260824_native_factor.md`
8. `docs/expert_review_20260826_bank_conditioned_native_factor.md`
9. `docs/expert_review_20260828_g3_functional_sketch.md`
10. `docs/expert_review_20260829_joint_program_primal.md`
11. `docs/expert_review_20260830_program_bank_interaction.md`
12. `docs/expert_review_20260831_event_conditioned_bank_set_relative_interaction.md`
13. `docs/expert_review_20260901_program_through_bank_bottleneck.md`
14. `docs/expert_review_20260902_global_route_reassessment.md`
15. 当前active design，仅当`progress.md`明确登记时读取；当前为
    `docs/program_conditioned_native_bank_tangent_transport_design.md`。

旧架构先查`research_history`；只有需要精确公式、实现或命令时，才从该文档登记的Git
快照和formal artifact选择性读取。不得把重复阅读几十份退役设计当成推进前置步骤。

## 3. Repository role

`AGENTS.md`只记录稳定的项目总览、科学合同和执行原则。它不记录active run、最新checkpoint、实验分数、动态下一步
或临时协作状态。仓库级工作状态只使用`task_plan.md`、`findings.md`和`progress.md`；历史结果只进入
`research_history.md`。临时handoff只在真实跨session交接时创建，消费后删除。

## 4. Scientific objective

EMBER研究能否从generic `lerobot/pi05_base`建立的冻结π0.5-LIBERO source policy出发，把目标task的exact language
和一条或多条action-hidden正确教学视频，在rollout前一次性编译为一套完整task-conditioned LoRA，使policy从
未见初始化闭环完成任务。

当前目标是Writer初次生成的LoRA立即有效。生成LoRA后的task-local RL是后续独立实验，不能混入当前
zero-interaction分数。

唯一正式性能目标线为validation8 strict paired correct严格`>145/400`。通过还必须由相邻single
checkpoints、低churn、高breadth、四个suite均非零、Goal/Long贡献、same-task不同视频鲁棒性和最终视频
因果controls共同证明，不能是训练波动中的偶然峰值。

closed-loop absolute性能首先选择方法。LoRA norm/rank/cosine、reconstruction、functional loss、内部时序margin、
hidden差异和surrogate只作定位证据，不能为了数值漂亮接受明显更差的闭环性能。

## 5. Input, output and information wall

- 输入必须包含exact task language和一条或多条同task、action-hidden、内部有序teacher videos。
- language说明关注什么和目标是什么，但不能独立写出有效LoRA；video dynamic evidence必须是必要Value路径。
- deployment Writer不得读取teacher action、proprio/state、reward、terminal、task ID、filename、object pose、
  hidden normalization或policy outcome。授权的non-held meta tasks可在训练时使用action、privileged expert或
  on-policy reward学习共享Writer/functional decoder，但这些信息不得成为deployment输入或task-ID route。
- validation/test actions或reward不得产生梯度。允许模型冻结、无checkpoint选择、预注册的一次性sealed post-hoc
  held诊断；Test默认保留到最终方法冻结后，提前使用必须明确登记且不得反哺设计。
- 每个condition只生成一套完整38-target task LoRA；不生成多套video LoRA后平均，不挑video，不融合checkpoint，
  不部署第二套expert adapter。
- Writer在rollout前运行一次；一次调用内部可以对同一组授权视频/native activations做固定、只读的多阶段流式读取与重放，
  但闭环中不反复观看teacher video，也不进行task-local优化或环境交互。
- frame stride固定为5；frozen source policy无trainable parameters。允许learned language-only诊断baseline，以及
  rollout前合并为一套LoRA的principled shared prior/base adapter + video-conditioned residual；canonical仍必须证明
  video相对language/static prior有必要条件增量，且不得部署并行carrier、expert或第二adapter。
- task experts可作为train24及经审计的non-held LIBERO-90 meta tasks的privileged teacher或几何诊断；不得成为held
  dictionary、task-ID route或第二套LoRA。

Dynamic-K若被方法声称支持，训练必须真实覆盖各cardinality。每条video先独立保序编码，videos只在集合阶段做
置换不变聚合；不得平均frames、raw features或最终LoRAs。one-shot、few-shot或动态K最终采用哪种论文设定，只由
真实性能决定，不为形式公平故意削弱更强方案。

memory token、LoRA rank、FactorHeads、layer correspondence和具体decoder都是候选方法，不是目标或硬约束。
不得为了“用上backbone”构造无意义的zero-image、fake action query或缺失原生prefix的forward。

## 6. Fixed data and benchmark contract

- LIBERO Spatial/Object/Goal/Long共40 tasks；
- development split固定为`configs/libero_24_8_8_v1/`的24 train / 8 validation / 8 test，不得按结果改ID；
- source corpus由LIBERO-90 specification audit排除与目标40重合的19 tasks后保留71 tasks，每task 50条成功episode；
- successor Writer/meta-training可使用train24，以及LIBERO-90中经过精确语义/specification审计、明确排除固定
  validation/test tasks及其重复项的其它任务；必须保存显式allowlist与provenance，不得以更多同task episodes冒充
  更多独立meta-task mappings；
- 不得使用读过目标40 actions的`pi05_libero`；
- normalization只从过滤后的source actions/states计算并冻结；validation/test不得重算；
- 方法选定后才允许按规定合并32 source / 8 test并从fresh重训。

## 7. Training and decision contract

- target development gradients只来自24 train tasks；active design可登记额外、经审计的non-held LIBERO-90 meta-task
  gradients。所有授权meta tasks按预注册口径等权或显式分层，validation/test不得产生梯度。
- video与action query同task但跨episode采样，阻断逐帧轨迹复制。
- 多卡可按K、帧数和历史cost平衡负载，但不得改变task权重。
- formal checkpoint保存Writer、optimizer、scheduler/scaler、sampler/cursor、rank RNG、world topology和schema。
- incompatible架构必须fresh；exact-resume锁原world size/topology。
- G1--G3的分段冻结只用于机制验证，不构成Final必须照搬的训练课程。Final允许把通过Gate的组件作为初始化，也必须保留
  整套Writer完全随机初始化后直接端到端fresh联合训练的正式候选；两者都使用fresh optimizer/scheduler并由同一closed-loop
  合同裁决，不得用内部loss替代。
- 机制smoke只证明图接通。到有信息量的预注册节点后及时做strict paired400；loss不能代替闭环。
- 好结果应训练到足以判断相邻稳定性；明确坏结果不得靠无限续训或rank/scale/seed/LR/dtype小扫挽救。
- 每轮必须报告per-task、per-suite、breadth、retained/gained/lost、churn及相邻success-set重合，并定位最早失效接口。
- 一次尽量只改变一个主要因果变量。负结果只淘汰实际检验的假设，不能因局部失败整套180度转向。
- 正式checkpoint选择只使用active design预注册的qualification arms与相邻稳定性；selected checkpoint选定并
  冻结后再补视频因果controls。shuffled/reversed最后测试，只确认时序特异性；它们不进入训练、loss、
  checkpoint选择、G1--G5 Gate或架构修正依据。

## 8. Evaluation contract

- official preprocessing：render256/model224、双相机180度rotate、7维state/action、10 flow steps、执行前5 actions后
  replan、dummy settling10、成功即终止、suite horizon 220/280/300/520。
- zero-interaction rollout从正确task的teacher videos无放回采样，不挑最好video。
- correct/same-task-other/cross-suite-wrong/shuffled/reversed/no-video严格配对task、state、env/policy RNG和video
  ordinal；shuffle/reverse必须重排真实frames后重新完整forward。
- evaluator采用cost-balanced dynamic queue、long-first和persistent workers，不做静态task/GPU分配或dummy占卡。
- 正式选择只认single-checkpoint 400 paired rows；80-row screen、checkpoint union、融合和内部surrogate不能选模型。

## 9. GPU, throughput and numerical policy

- 每次launch前同时live检查gpu01与gpu02，按节点、GPU index、显存、utilization和process判断空闲、可共驻、忙或故障。
- 单节点至多使用6张真正提高吞吐的A40；有几张合适就用几张，不等待凑6、不跨节点拼碎片、不dummy占卡。
- 少量显存占用或低util进程不自动排除GPU；只要峰值余量足够且不会显著干扰即可共驻，但不得kill、pause、
  reset或抢占他人任务。
- 多卡训练固定`NCCL_P2P_DISABLE=1`、GPU-local NUMA映射和deferred NCCL；独立evaluator不用NCCL。
- 接受正常BF16/TF32、batch、kernel和reduction order低位差异；不为逐元素一致固定batch1、重复forward、扩dtype、
  关闭高效kernel或逐tensor扫描。
- 不新增SHA-256、MD5或大量防御性校验。只保留信息墙、shape、finite、OOM、asset、pairing、checkpoint和resume
  正确性所需检查。
- profile以真实LoRA/s、samples/s、最长视频稳定性、GPU利用率和显存峰值选择batch，不以最低显存为目标。

## 10. Storage, artifacts, Git and documentation

- 大资产只放`/data0/user/ymdai`或`/data1/user/ymdai`。大copy/cache/training前在`strg01`检查对应filesystem的独立
  user quota、测实际用量并估计峰值；`df -h`不是quota检查。
- 复用canonical policy、dataset、tokenizer、assets和manifest，不复制大资产。
- formal结果保留run contract、checkpoint manifest、metrics、raw rows、aggregate、completion和必要analysis；
  profile/smoke不得冒充formal。
- active tree只保留一个canonical Writer运行面。退役实现由Git、sealed configs、formal artifacts和
  `research_history.md`保存，不保留平行fallback。
- canonical workspace是本仓库，主写与集成目标为`main`。需要隔离、并发写入或独立实现时，从最新`main`创建
  `codex/<topic>`分支与独立worktree；验证后及时合并回`main`并推送远端，确认集成完整后清理task-owned worktree。
  formal train/eval必须来自clean pushed commit的detached frozen worktree。
- 不提交dataset、cache、checkpoint、大binary、secret或host-private配置。
- 稳定目标写入`current_owner_requirements`和`concept`；当前goal与计划写入`task_plan.md`，即时进度写入
  `progress.md`；历史结果写入`research_history.md`，跨轮结论写入`findings.md`。不得向`AGENTS.md`追加动态
  实验年表。
- 只删除明确obsolete、temporary或duplicate内容；formal evidence、唯一checkpoint、dataset和所有权不清内容保留。

## 11. Collaboration

owner授权在上述边界内自主推进。具体协作模式以owner最新要求和`progress.md`为准。反馈是设计约束或启发，不是要求
机械照搬；不得因owner提出一个局部问题就推翻所有已对齐内容。需要解释架构时，先给出完整数据流水线和每个模块
的因果作用，区分目标、原则、诊断与具体方法。

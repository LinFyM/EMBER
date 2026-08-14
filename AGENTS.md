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
4. 当前active design：
   `docs/action_forecast_writer_v6_lpcp_semantic_factor_memory_commitment_design.md`
5. `task_plan.md`
6. `findings.md`
7. `docs/concept.md`
8. `docs/research_history.md`

涉及旧架构细节时先查`docs/research_history.md`；只有确需旧公式或实现时再从Git commit`3a6f801`选择性读取。
涉及A100到BCI迁移与路径恢复时才读`docs/a100_to_bci_migration_handoff.md`。不要把重复阅读数万行退役设计当成
科研推进前置步骤。

## 3. Current operation

长期目标尚未完成。v6-fast与最新V6-LPCP的correct single checkpoint同为143；v6-fast仍是有完整五臂的历史最好：
`correct/same/wrong/shuffled/reversed=143/135/125/128/129`，LPCP只完成correct且breadth7。

当前最强zero-interaction carrier baseline是**V6 Layerwise Action-Probe Conditioned Procedure Reader**（V6-LPCP）：macro25 K4
strict=`143/400`、breadth7、per-task=`1/4/48/35/0/38/16/1`、per-suite=`5/83/38/17`。相对同schedule
AS139严格=`120 retained / 23 gained / 19 lost / 238 both-fail`、churn42、net`+4`、p=`.643969`；它
count-only追平不同teacher schedule的历史v6-fast143并把breadth从6增到7，但按`<144`和lost>10两项门终局
non-pass，不resume50、不补controls或扫memory/rank/LR/scale/seed。

该轮否决的不是layerwise视频carrier：同一次真实context forward的18层probe旁读满足one-forward，倒序使
query-delta/Program relative-L2=`2.0572/.40414`，constant query-delta近零；reader/controller也获得持续credit。
全400 effective-BA相对AS139只改`.002653` relative-L2、cosine`.99999479`、norm ratio`.99997391`，而first4
same-task correction coherence median`.56804`。Goal3虽有`.004224`改写和`.88373` coherence仍为0，Long1只改
`.001324`却净丢6。最早缺口是conditioned Procedure经冻结fusion/compiler承诺成AS139邻域小方向，以及blind
B20 functional credit不能选择held on-policy有用方向；只替换已通过的carrier为literal memory不直接针对该缺口。

封存authority=`docs/action_forecast_writer_v6_layerwise_probe_conditioned_procedure_design.md`。正式root、AS139严格
配对和effective-BA artifact均已保留。owner随后明确授权继续，并再次澄清memory token是候选机制而非必须形式。
V6-LPCP Paired Causal Success Distillation（PCSD）保留LPCP/AS139/rank16部署图，以同初态
reference/candidate两臂的唯一成功轨迹只校准65,536参数`query_delta`。精确authority=
`docs/action_forecast_writer_v6_lpcp_paired_causal_success_distillation_design.md`。canonical实现已经原位替换旧ADSP
runtime：同一K4 context只编码一次，K2两臂返回完整executed-prefix replay，ties为零，只训练
`query_delta.weight`；全量CPU=`387 passed`、architecture guard无hard violation。clean frozen `efc17be`在
gpu01物理`5/6/7`以world3完成full24 cycle1：24 tasks、48 pairs、96 rollouts，candidate/reference=
`34/33` successes、`5/4` gains、9 discordant/active tasks覆盖3 suites，gradient、parameter delta、BA/action
response均非零，0 forbidden read/OOM/nonfinite，wall=`837.694s`。

PCSD cycle1 K4 strict paired400已终局为`135/400`、breadth6、per-task=
`0/4/48/32/0/35/15/1`、per-suite=`4/80/35/16`。相对LPCP143严格=`121 retained / 14 gained /
22 lost / 243 both-fail`、churn36、net`-8`；相对AS139严格=`115/20/24/241`、churn44、net`-4`。
全400 PCSD相对LPCP effective-BA relative-L2 mean/median仅`.0006834/.0006767`，gained/lost幅度不可分；
FP64 first4显示同task四个不同K4 video sets的增量pairwise cosine平均`-.00187`、mean/sample energy ratio=
`.24860`，即几乎正交且平均后只剩约四分之一能量。reward credit有内容且LoRA→action链路工作，但稀疏
paired success经单一shared query commitment没有形成跨task/video可保留方向。PCSD按四项门终局non-pass，
不得cycle2、补controls或参数小扫。

最新完成successor是**V6-LPCP Cross-Video Causal Success Distillation**（CV-CSD），精确authority=
`docs/action_forecast_writer_v6_lpcp_cross_video_causal_success_distillation_design.md`。clean `c1d8952` full24 cycle1完整：
24 tasks/48 paired states/96 rollouts，33/34两臂成功、5/4单臂成功、9 active tasks；四view全部LoRA/query gradient
非零，wall=`863.432s`=`1.0307x` PCSD，3 ranks各8 tasks/3 active tasks、负载max/min=`1.0828x`。

CV-CSD K4 strict paired400终局=`134/400`、breadth7、per-task=`1/2/47/32/0/36/15/1`、per-suite=
`3/79/36/16`。相对LPCP143严格=`122 retained / 12 gained / 21 lost / 245 both-fail`、churn33、net`-9`，
四suite全降；相对AS139=`121/13/18/248`、相对PCSD135=`115/19/20/246`。correct、lost、net与suite四项门失败，
不续cycle2、controls或小扫。

全400 CV-CSD/LPCP effective-BA relative-L2 mean=`.00068370`，gained/lost约相同；FP64同task四K4 correct
conditions的增量pairwise cosine=`.000205`、mean/sample energy=`.250155`，相对PCSD也为`-.001908/.248578`。
所以四个正确视频下的exact成功credit经过shared `query_delta`均值后仍落成近正交局部方向。

当前active successor是**V6-LPCP Semantic Factor-Memory Commitment**，authority=
`docs/action_forecast_writer_v6_lpcp_semantic_factor_memory_commitment_design.md`。它保留LPCP143、cross-video
selected-success与rank16强图，冻结query path；从同一cached condition计算K-set LPCP/AS139 Procedure差形成
layer/rank innovation memory，以exact-language作Q/K-only四basis语义地址，在八个真实factor-family的冻结V6
output basis前写zero-init hidden residual。它不增加literal memory token、不重复backbone forward、不直接写raw
A/B。当前尚未实现、profile或训练；不得把design当成结果，也不得同时改rank/view/LR/scale。

## 4. Long-term objective and decision rule

EMBER研究能否从generic`lerobot/pi05_base`建立的冻结π0.5-LIBERO source policy出发，把目标task的语言和
action-hidden正确教学视频一次性编译为task-conditioned policy adaptation，使policy从未见初始化闭环完成任务。

当前主目标是Writer初次生成的adaptation本身立即有效。生成LoRA后的task-local RL是之后独立实验，不能混入
当前zero-interaction分数。性能继续追求`>150/400`并越高越好；owner最新明确补充，约`145`也可成为科学上
有价值的稳定方法，前提是它不是训练波动中的单点winner。方法资格要求同一shared method的相邻single
checkpoints同时具备：

- strict paired correct保持约`145+`或更高，而非只在一个checkpoint偶然出现；
- 高task breadth、相邻checkpoint低换手/高success-set重合、多个tasks共同积累；
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

任何首次达到约`145`且retention过门的checkpoint就必须补视频因果controls，并继续评测相邻checkpoint稳定性；
不能等到`>150`才检查correct是否沿有用policy direction获益。

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

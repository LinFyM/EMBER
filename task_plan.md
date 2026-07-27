# EMBER Task Plan

最后更新：2026-07-27。长期 EMBER Goal 是完成共享 π0.5-LIBERO source base、
one-video Writer、source baselines、seen/wrong-video机制证据、final
single-seed test、test-only三臂RL和联合target-action oracle；当前focused
Semantic Core + Causal Procedure Writer由本文末尾执行段及
`docs/action_forecast_writer_v5_design.md`定义。

## 完成定义

只有以下核心项全部完成，长期 Goal 才可完成：

1. specification-only 封存过滤后的 LIBERO-90 source-base corpus、数据/normalization/model hashes；
2. 训练并冻结共享 π0.5-LIBERO source base，快速screen全部目标40 tasks并确认已出现部分真实成功；
3. 在24 train / 8 validation上完成并选择AS-Writer、RL-Writer（若能启动）与Source-SFT；
4. 完成source/seen panel及correct-video vs cross-suite wrong-video证据；
5. 合并validation后，在32 source tasks上以单seed从规定初态重训已选方法；
6. 完成final seen comparison与8-task zero-interaction test；
7. test打开后在8个test tasks上将identity/AS-Writer/RL-Writer三臂task-local RL训练到各自接近最佳，并做训练分离的fresh evaluation；
8. 使用8 test tasks × 50 action episodes联合训练一套shared target-action LoRA oracle并评估；
9. 原始rows、per-task counts、learning curves、seeds、interaction/data counts、runtime与hashes齐全，验证、commit、push。

ViVLA-style matched baseline和source-only outer learning为时间允许时的后续项，不阻塞核心Goal complete。当前focused RL-Writer合同已改为独立、短且task-balanced的AS cold start，直到24个train tasks逐task至少一次official random-reset success后才转pure reward；不得再按旧“零warm-up优先”口径启动。

## Phase 0：generic π0.5 feasibility

状态：已完成。

- [x] 封存24/8/8目标split。
- [x] official-compatible π0.5 evaluator与24-train interface stats mechanics。
- [x] 8 test tasks × 50 fixed states：generic `pi05_base` 为 `0/400`。
- [x] 结果seal、49项测试、commit/push完成。

该结果只作原始模型校准。Phase A/B及Source-SFT comparator现已完成；当前活动
路径是本文后方的Action-Forecast Writer执行段。

## Phase A：source corpus、成熟recipe与高吞吐评测

状态：source corpus与recipe已封存；高吞吐evaluator实现、机械门禁和1/2/3 replicas真实rollout/s profile完成。

- [x] 调研官方/成熟π0.5 action fine-tuning与LoRA项目；锁定OpenPI/LeRobot revision、full-SFT recipe、下游LoRA targets/rank/alpha/dropout与identity init。
- [x] 对LIBERO-90与目标40 tasks完成3600对language/BDDL/semantic/composition specification audit。
- [x] 排除19个完整任务等价项并封存71个active source IDs、可执行规则与hashes；包含已知task44/task77及其余semantic aliases。
- [x] 核验71 tasks×50 successful episodes、529,173 frames、52,710,755,898 bytes；封存source-only q01/q99 normalization，validation/test numeric reads为0。
- [x] 下载并按opaque SHA/HDF5 schema-only核验目标40 tasks×50 episodes；封存24/8/8 target-data manifest，未解码target action/state/reward/terminal/video值且未改变task IDs。
- [x] 已将当前“一task/一GPU”评测改成按 `episodes × horizon` cost-balanced state shards、动态队列、持久model/env，并删除旧静态活动入口。
- [x] 在同一canonical evaluator内加入PI05 AS-Writer逐rollout materialized LoRA、one-video哈希schedule、role-preserving cross-suite wrong map及逐row可重算证据；没有新增第二套runner。
- [x] final source checkpoint后在同一40-task×1-state panel完成每卡统一1/2/3 replicas真实profile；有效rollout/s为0.1556/0.1818/0.1897，选择3 replicas/GPU。
- [x] 确认所有卡CUDA进程数相同、GPU0无额外角色；3-replica每卡约31GB且24 workers全部对称。未来Writer异LoRA functional batching仍需在Writer checkpoint可用后另测。

## Phase B：共享 π0.5-LIBERO source base

- [x] 建立唯一canonical π0.5 full-SFT runner，严格模型加载、task-balanced no-replacement sampler、EMA、atomic checkpoint、metrics reconciliation与完整per-rank RNG/cursor恢复。
- [x] 完成相机mask修正后的8卡真实profile；选定8×microbatch32、global batch256、EMA与GPU-local NUMA binding，稳态47.75 examples/s，单卡reserved 71.18GB。
- [x] 从同一个不可变step-1 checkpoint做两次8-rank恢复；loss/grad/RNG/cursor完全一致，policy/EMA最大独立NCCL末位差分别为1.49e-8/3.73e-9。
- [x] 旧30k attempt3按owner决定在step2880停止；无checkpoint、不得resume，也不作source competence结果。其loss仍下降，只用于估计训练速度和否定“已饱和”的说法。
- [x] 从generic `pi05_base`在过滤后的71-task LIBERO-90 corpus上完成共享full action-SFT；没有未merge source LoRA。
- [x] 从generic base fresh完成1,000 optimizer steps；333-step warmup、8卡一卡一rank、global batch256，最终step1000保存并完整验证唯一checkpoint。
- [x] 按用户指定在step1000停止；末50-step mean loss为0.08659，曲线仍缓慢下降但明显趋平，记为budget-censored而不自动延长。
- [x] 诊断并推翻错误的EMA screen：`decay=0.999`的step1000 EMA只走完raw更新位移的28.62%；匹配LIBERO-90 source tasks上raw为4/4、EMA为0/4，故EMA 0/320不是科学负结果。
- [x] 用step1000 raw policy完成全部40 targets×8 fixed states正式screen：46/320成功，覆盖13 tasks与全部4 suites；逐suite为Long 2、Goal 28、Object 1、Spatial 15 successes，满足多task partial competence门槛。
- [x] 冻结step1000 raw policy、source-only normalization及model/data/tokenizer hashes，作为全部后续方法共同起点；EMA仅保留为训练状态与负诊断证据。

## Phase C：AS-Writer development

- [x] 将Writer core、target feature-cache与exact-resume训练owner适配到π0.5成熟LoRA空间；同task video/action episode使用独立确定性schedule，Writer只见language+one action-hidden video。
  - 已封存38 targets / 76 tensors / 1,287,168 parameters的rank16合同；真实checkpoint metadata与meta模型结构一致。
  - 已修复PI05 forward签名、BF16/FP32 LoRA dtype保真，并在Writer入口对`offsets=[0,L]`单视频合同fail-close；functional/materialized parity已测试。
- [x] 封存target40 metadata/hash authority与development 32-task视频cache配置；AS formal由真实训练profile封存，不将候选steps冒充正式配方。
- [x] feature-cache完成raw source policy上的真实8卡batch32 smoke：8 tasks/8 episodes/1,033 frames，critical-path 689.47 frames/s且无OOM/nonfinite；batch32已封存。
- [x] 生成development 32 tasks×50 videos、274,523 frames的formal feature cache；32/32 task tensor hashes与manifest/contract复核通过，test video reads为0。
- [x] 24 train tasks均衡混合；source base冻结，actions只进functional loss；正式训练完成1,000 steps / 128,000 action queries并保留四个候选checkpoint。
- [x] 完成真实8卡短profile并将wall-clock换算为steps：batch16每rank峰值allocated/reserved为63.53/68.17GB，稳态约1.05秒/步、约122 global action queries/s；128-step曲线的首/末16-step mean loss为0.14714/0.11930，后64步斜率仍为负。
- [x] cheap screen从250/500/750/1000中选择250与500做完整validation；step250为119/400，step500为99/400，因此冻结step250为development AS-Writer。
- [x] 报告8 validation tasks逐任务raw success与视频采样seed；同配置source-base control为48/400，AS step250为119/400。
- [x] correct/wrong control确认旧AS-Writer未建立视频因果性：119/400对115/400，有效LoRA的correct/wrong相对差中位数仅`7.52e-6`；旧v1不再作为活动Writer配方。
- [x] 在同一canonical runner内完成Writer-v2组合修订：每帧保留4×4空间网格、condition-only层级memory和LoRA decoder；按`normal → full-language contrast → generic-language contrast`训练，generic只进入Writer且policy始终看正确language；没有新增adapter、subspace或第二套runner。
- [x] 生成并封存v2 development feature cache：32 tasks×50 videos、274,523 frames、32/32 tensor hashes与information-wall复核通过；完成8卡batch16三分支短profile，positive loss下降且两类matching gap由微负向微正移动，据此封存250-step首轮与50-step checkpoint间隔。wall-clock倍率只记录，不作为科学接受门槛。
- [x] 完成Writer-v2首轮250-step正式训练并保存50/100/150/200/250 checkpoints；250条metrics全部finite，24 tasks各覆盖50 action episodes与50 teacher videos，三种positive loss下降且末10个contrast steps的wrong-minus-correct均值为full `+0.00707`、generic `+0.00788`。
- [x] Writer-v2首轮step250完成四分支64-state screen；full-language与generic两组均为correct `12/64`、wrong `8/64`，显示初步视频特异性与generic-correct绝对competence。
- [x] Writer-v2 step250完成full-language完整validation：correct `83/400`、cross-suite wrong `63/400`，paired correct-only/wrong-only=`40/20`、exact McNemar `p=0.01349`；5 tasks正向、1负向、2持平。相较v1的119/115，视频特异性增强但correct绝对性能下降。
- [x] 从identity fresh完成Writer-v2 1,500-step完整cosine ceiling run，并补一条仅改变checkpoint retention的350–750细粒度replay；correct-video完整validation确认活动run step500为最强已观测checkpoint（99/400），随后只对它运行一次cross-suite wrong-video（55/400）。不再补generic full或其他wrong arms。

## Phase D：RL-Writer development

- [x] 建立共享π0.5 reward mechanics：official random BDDL reset、10-step settling、suite horizon、显式逐replan flow-noise seed、只执行/监督前5 actions、successful-episode等权与不可变interaction ledger。
- [x] 建立fresh zero-AS RL-Writer唯一活动owner、8-rank task/video full-cycle schedule、Writer-only functional update、完整checkpoint/RNG/cursor/ledger exact-resume；真实24-task profile得到7/24 successes与3/3有效Writer updates，峰值reserved 40.84GB，formal已按实测封存为可在稀疏checkpoint暂停/续训的120-update guardrail、首段只到update12。
- [x] 从随机Writer、零AS warm-up直接跨24 source tasks用官方random-reset reward联合训练；到update54累计432 rollouts、81 successes、131,354 environment actions，全部teacher-action consumption为0。
- [x] fresh identity首个cycle即有7/24正信号，故按预声明规则不进入micro-AS分支；这不是缺失证据，也不从完整AS-Writer继续。
- [x] 通过update3→24→36→54的真实8-rank exact-resume保存worker RNG、seed schedule、interaction cursor和完整checkpoint；source reward后段cycles已平台。
- [x] 固定64-state validation screens在update12/24/36/54为`6/11/15/14`，据此冻结最早峰值update36；完整correct-video validation为`94/400`，不再追加72/96/120。

## Phase E：Source-SFT、seen与视频因果证据

- [x] 建立development-only Source-SFT authority与唯一PI05训练owner：24 train tasks共享一套38-target LoRA，raw source policy冻结，checkpoint保存adapter/optimizer/scheduler/per-rank RNG/确定性sampler与metrics cursor；现有evaluator按静态adapter一次安装并保留普通batch。
- [x] Source-SFT matched-scale pilot使用batch64/rank训练63 steps、32,256 queries，validation为61/400；该run仍处于继承的1,000-step warmup且loss继续下降，只作undertrained provenance，不代表上限。
- [x] 从fresh identity完成独立Source-SFT ceiling run；完整validation为step200/400/600=`74/87/73`（各400），step400明确最强并冻结。曲线在600前后已出现validation回落，因此不恢复到800，也不重复训练。
- [x] 在outcome前按specification-only SHA256规则封存四suites各2个、共8个seen tasks（global IDs 0,2,15,12,21,28,39,37）；policy outcome与trajectory value reads均为0。
- [x] 在预封存8-task×50 fixed-state seen panel上比较source base、Source-SFT、AS-Writer与RL-Writer：分别为`137/400`、`182/400`、`204/400`、`164/400`；AS最强，四方法的Long task 9均为0，结论保留逐task raw rows而不追加无决策价值的seen消融。
- [x] 封存同split role、按suite循环和排序ordinal构造的cross-suite wrong-video机械map；correct/wrong两臂保持同一language、task、init state、policy RNG、video seed与demo ordinal。
- [x] canonical evaluator在同一dynamic queue、persistent worker和raw-row schema内支持frozen AS-Writer或RL-Writer；两者共享correct/wrong mapping与video seed，artifact kind/checkpoint axis分别fail-close。
- [x] 用正式AS checkpoint运行并报告correct/wrong/base逐任务结果：119/115/48（各400）；correct−wrong仅+4，当前AS结果未显示强teacher-video内容依赖。RL-Writer若成立后另补其对照。
- [x] 完成development比较：source base `48/400`、Source-SFT step400 `87/400`、Writer-v2 step500 correct `99/400`、同Writer唯一wrong-video arm `55/400`。correct-only/wrong-only=`56/12`、exact McNemar `p=6.21e-8`，6/8 tasks为正向视频效应；AS-Writer暂时通过并解除RL-Writer暂停。
- [x] 对唯一冻结的RL-Writer update36完成correct/cross-suite-wrong完整validation：`94/400` vs `87/400`；paired correct-only/wrong-only=`10/3`、exact McNemar `p=0.09229`。reward-only held competence成立，但视频因果证据较弱；不补generic或其他checkpoint wrong arms。

## Phase F：32-source final retraining与zero-interaction test

- [x] 将8 validation tasks机械合入形成32 source / 8 test；final AS/Source-SFT/RL配置只选择`train+validation`，test IDs与test action/reward reads保持封闭。
- [x] 第一轮只用一个training seed，从规定初态完成final AS-Writer与Source-SFT重训。
- [ ] 从fresh zero-AS Writer规定初态完成final RL-Writer重训。
- [ ] 先完成final seen-task comparison和wrong-video control。
- [ ] 打开test，评估新frozen source base、Source-SFT、AS-Writer、RL-Writer及correct/cross-suite-wrong video。
- [ ] held Writer每rollout随机采一条正确task video，不挑最好video；全部方法用相同fresh evaluator和paired seeds。

## Phase G：test-only three-arm task-local RL

- [x] 封存test-only机械合同：恰好8 test tasks、identity/AS/RL Writer三臂（RL失败时合规缺席）、同cohort video/seed schedule、一次生成并固定初始化LoRA、三类cursor和hash-bound exact-resume；formal预算保持0且test打开前无法启动。
- [ ] 不在validation上预先训练或冻结此RL；test打开后直接针对每个test task调优并训练到曲线接近最佳。
- [ ] 三臂：identity-init、AS-Writer-init、RL-Writer-init；均基于同一frozen source base和同一LoRA空间。
- [ ] 每task/adaptation seed随机一条teacher video，AS/RL Writer共用并固定初始化LoRA。
- [ ] 三臂匹配task、env/policy seed schedule、official random BDDL init sequence、RL代码和可比资源上限；保存全部learning curves与time/interactions-to-best。
- [ ] adaptation与checkpoint选择不用fixed `.pruned_init`；最终固定50 states只作fresh evaluation。

## Phase H：8-test联合target-action oracle

- [ ] 前述无action方法与RL结果封存后，才读取8 test tasks的actions。
- [ ] 从同一frozen source base出发，用8×50完整action episodes联合训练一套shared multi-task LoRA；不是8套task-local LoRA。
- [ ] 第一轮只做50/task，不做1/5/10 action-budget curve。
- [ ] 使用同一LoRA targets/rank/alpha/dropout，报告逐任务和aggregate；明确标记privileged oracle。

## Optional after core

- [ ] 时间允许时做同source base、split、one-video输入墙的ViVLA-style matched reproduction与test。
- [ ] 核心成立后才考虑source-only outer learning；不阻塞Goal complete。
- [ ] 有足够核心性能差异后再补独立training seeds；第一轮不提前扩大。

## 每次运行前

- [ ] workspace/branch/commit/status明确；无未识别并发writer。
- [ ] live GPU owner、进程、driver/CUDA与storage audit；预计峰值低于500GB个人cap。
- [ ] exact command、model/data/config hashes、output root、process topology与停止条件记录。
- [ ] 一卡一训练rank为默认；若评估每卡多replica，所有已授权卡的replica数必须一致且GPU0无额外角色。
- [ ] checkpoint/output不得覆盖；resume必须校验完整state与合同兼容性。
- [ ] 所有适用阶段先短profile并由loss/reward/behavior曲线决定廉价screen间隔，只给必要候选完整validation；若owner为某阶段给出时间上限，到上限仍未充分训练则记录后停止。当前focused v5.1没有机械总时间上限，但任何第二/第三segment都必须重新过特异性、absolute和曲线证据门；Source-SFT comparator已经封存，不在本子任务重训。

## 历史 Goal：四卡Action-Memory / Source-SFT / cold-start RL-Writer上限（2026-07-23，已被下方Action-Forecast执行段覆盖）

本节未勾选项随该Goal退役，表示“当时未执行”，不是当前待办；不得据此恢复
GPU0–3权限、Action-Memory、condition-balanced旧估计器或旧训练步长。

- [x] 将冻结PaliGemma逐帧图文prefix、16个Action-Expert memory tokens、encoder-only Meta-LoRA、变长temporal/layer/slot聚合与完整rank16 LoRA解码实现为唯一canonical Writer路径。
- [x] owner确认此前全局`bias=False`是额外优化限制而非condition-only必要条件；已只恢复conditional temporal/layer/slot与factor-head内部普通bias，不增加公共LoRA支路、层、token、宽度或输出。
- [x] 在读取新validation action值前封存8-task task-balanced functional-loss panel：每task 8条video × 8个独立action query，共512 rows/checkpoint；只作候选筛选，400-rollout closed-loop success仍是最终authority。
- [x] 2026-07-28前所有训练和评测只能使用物理GPU `0,1,2,3`；四卡真实forward/backward、batch ceiling、显存与吞吐profile及fresh AS合同已封存，GPU4–7始终未触碰。
- [x] 从fresh identity充分训练首条bias-restored AS-Writer轨迹；旧decay-6400轨迹因global batch减半后的scheduler混淆只作provenance。query-scaled warmup100/decay2400轨迹的val-loss在step400到达`0.134058`，随后整体恶化，故在完整step800后早停。step300/400/500/600/800的完整8×50 correct-video结果为`57/91/86/87/88`，确认step400为该轨迹真实峰值且无需补step200；但它仍显著低于旧rank128 Source-SFT step400的`122/400`，所以只完成了首轨迹上限诊断，尚未通过本Goal的AS>SFT门槛。
- [ ] 对AS做不改架构的condition-balanced训练修正：四卡时每rank每个optimizer update顺序处理2个独立task/video conditions，每个condition仍使用16个独立action queries；因此每次更新覆盖8个独立conditions和128个queries，恢复旧八卡AS的逻辑训练单位，而不是机械匹配Source-SFT的global512。该设置只用于恢复条件覆盖与梯度稳定性，不把batch size当作方法门槛；已有checkpoint能合规续训时不因GPU数量变化机械重训。现有`as_step.py`仍是唯一训练owner，不新增runner；profile通过后优先从现有best warm-start，新优化阶段先跑300 steps并每100步保存，未充分再按300步续训，fresh只作必要归因对照。
- [ ] 充分探索rank128 Source-SFT。四卡batch128 profile已完成，稳定吞吐约`36.18 queries/s`、峰值reserved `67.98GB`；旧八卡`step400=122/400`仍是全局incumbent。旧8×64与当前4×128在同一step拥有相同optimizer updates和global 512 queries/update，因此训练量大体可比；每次更新内8个task小批与4个task大批的差异只作次要梯度方差诊断，不能据此把四卡step机械乘2。
  - 当前四卡fresh run的step100–800 closed-loop依次为`81/95/68/78/94/99/108/97`；step700是该轨迹当前best，但相邻候选尚无显著峰后下降。已从step800原地exact-resume到1100。selection只微弱参考functional val loss，最终依赖同seed完整8×50 rollouts。
  - GPU数量变化不再触发从零重训：优先延续已有checkpoint，并同时报告optimizer updates、累计action queries、独立task-condition visits与拓扑切换点。跨world-size续训若不能保持逐rank RNG的bitwise exact-resume，则明确封存为同权重/optimizer的topology-transition continuation，而不是错误宣称exact resume。
- [ ] 要求AS在validation上明显超过Source-SFT；若未达到，保留逐task证据并诊断/迭代，但不改变source base、split、信息墙或核心科学问题。
- [ ] 对唯一最强AS checkpoint运行必要的correct/cross-suite-wrong video对照，确认correct优势来自多个tasks；若特异性不足，在保持绝对competence优先的前提下研究并解决。
- [ ] 只有AS同时通过相对Source-SFT与视频特异性门槛后，才启动fresh cold-start RL-Writer；cold-start阶段先取得24个source tasks逐task成功信号，再切到纯reward训练，并充分探索validation峰值，不设时间上限。
- [ ] 保存raw rows、loss/reward curves、seeds、actions/interactions、runtime、config/hashes与exact-resume证据，更新文档、验证、commit、push后完成Goal；本Goal不推进task-local RL或test阶段。

### 历史快速子任务：temporal-RoPE Writer（2026-07-24，已覆盖）

- [x] 原位替换canonical temporal owner：1D RoPE + 4个condition-only learned memory queries；保留bias、Action Memory、Meta-LoRA、信息墙与完整LoRA decoder。
- [x] GPU0–3原生global64两步profile通过，不做梯度累计或8卡逻辑batch模拟。
- [x] fresh训练到step500并封存step400/500；500-step训练body wall为`1188.6s`，两个checkpoint及完整四rank resume state均通过manifest校验。
- [x] 用同一paired 8×50 validation panel评测两个checkpoint；step400/500分别为`108/400`与`98/400`，逐task上step400处处不差，故冻结step400为本轨迹observed-best。
- [x] 只对step400运行视频/单帧/倒序/打乱特异性诊断并停止：固定语言时跨suite错误视频令有效LoRA相对变化中位数为`0.2267`，同task另一demo为`0.0403`，但倒序/打乱仅`0.00937/0.00699`。当前Writer已利用视频任务内容，却基本未利用动作时序；不继续本轮训练、contrast或RL。

## 历史执行：Action-Forecast Writer v1（2026-07-24，已覆盖）

下列任务是Action-Forecast v1已完成历史，不是活动设计，不恢复旧checkpoint。
该历史版本设计见`docs/action_forecast_writer_design.md`；当前活动设计见
`docs/action_forecast_writer_v5_design.md`。

- [x] 原位替换旧 Action-Memory owner，完成 imagined-state、VL/Action
  Meta-LoRA、完整10-flow action plans、absolute-time Plan/Revision、
  variable-time temporal encoder和one-way LoRA query decoder。
- [x] 退役旧 source/config/schema/tests，确认活动树只有一个 AS runner和一个
  Writer architecture；真实Writer参数量为`10,161,217`，是rank128
  Source-SFT `10,297,344`的`98.68%`。
- [x] 在GPU0–3实测并封存训练效率参数：stride=`5`、
  `frame_microbatch_size=32`、per-rank action-query batch=`16`。stride10仅有
  单步参考且owner决定不再扩测；frame-microbatch64在一rank达到
  `80,821/81,920 MiB`并失去前进，已明确拒绝。
- [x] 将Writer LoRA生成与rollout并发原位解耦：每卡generator数量、生成batch
  和rollout replicas各自封存；generator生成完整fixed panel cache后只释放
  Writer专属模块，同一进程保留source π0.5并转为首个rollout worker，其余
  replicas随后扩容。缓存可跨rollout拓扑复用，canonical入口仍只有
  `scripts/evaluate_pi05.py`。
- [x] 用正式Action-Forecast checkpoint分别实测生成batch/generator数与纯
  rollout replicas；早期耦合r3/r4只作provenance，不用于最终拓扑选择。不得用
  早期checkpoint运行wrong/shuffled/reversed，机制诊断只对最终observed-best执行。
  正式step450/600均确认每卡1个generator、batch100、随后每卡6个rollout
  replicas稳定；生成约53–56秒，rollout-only约`0.61–0.62 episode/s`。
- [x] fresh formal AS首段完成step0→300，按75/150/225/300保存四个完整
  exact-resume checkpoints；19,200 queries与全部loss/gradient均finite。
- [x] 旧版曾按约30分钟segment推进；该执行口径已退役，以下曲线只保留为
  Action-Forecast v1的历史性能证据，不约束当前75-step门控或fresh 1200训练。
  - 当前step150/300/450/600/750/900/1050/1200/1350/1500/1650/1800/
    1950/2100/2250/2400/2550 correct-video为
    `75/99/93/118/104/113/117/125/120/119/120/114/110/114/123/111/124`；
    step1200仍是observed-best，但2250/2550=`123/124`均回到同一峰值平台；
    2250与1200 paired仅净`-2`、
    exact `p≈0.896`，实质追平。1200→2400净`-14`、`p≈0.065`，但2400
    只是紧随123的单点回落且task方向混合，随后2550又恢复到124，仍未形成
    持续多task强下降；step2700完整checkpoint已保存，按owner指令暂停原轨迹
    候选评测并先检查当前observed-best的视频特异性。
- [x] 旧版observed-best的correct/wrong/shuffled/reversed证据已完成；其失败
  结论只作为新架构的设计动机。
  - step1200的correct/cross-suite-wrong/shuffled/reversed为
    `125/67/121/124`；内容特异性强且跨多个tasks，但shuffle与reverse均和
    correct实质相同，故顺序特异性门未通过，RL仍关闭。
  - 曾短暂profile过stop-gradient order-contrast warm-start，但该分支随后
    已删除且被owner明确否决；不得恢复。
- [x] 以下充分探索停止标准已迁移到当前执行段：AS和RL都不能以
  train/val-loss平台、单个较差点或多个仅略低的validation
  点停止；必须找到validation observed-best，并在其后观察到幅度非常明显、
  明显超过rollout噪声、由多个tasks共同贡献且独立panel复测后仍成立的下降
  趋势，才能确认饱和与最佳checkpoint。
- [x] 以下RL入口条件已迁移到当前执行段：仅在AS同时通过性能与特异性后，
  推进独立short-AS cold start，直到24个
  train tasks各至少一次random-reset success，再切换pure-reward RL-Writer并
  完成train平台与validation选择。
- [x] 旧执行段已由当前执行段覆盖；authority、验证和交付要求以当前段为准。

### 历史执行：Action-Forecast Writer v2（2026-07-25，已覆盖）

本节覆盖上方仍把order-contrast写成active next step的历史条目；不得恢复
contrast loss。

- [x] 将8-scalar/Fourier state bottleneck原位替换为28个content-only
  virtual-state tokens；routing identity只进入attention Q/K。
- [x] 将Revision原位替换为directed-event content read、Plan/Revision独立
  RMSNorm和bounded multiplicative stability gate；旧step1200反事实诊断中，
  normal→reversed/shuffled的time-centered相对L2从旧`0.0281/0.0316`
  恢复到`0.3554/0.2418`。
- [x] 将LoRA query decoder改成routing/content分离且factor heads只读取
  memory-derived content；factor heads bias-free，fresh public LoRA保持
  functional identity。
- [x] 删除canonical order-contrast配置和训练分支；AS只保留positive
  functional action loss。
- [x] GPU0–3真实fresh step1→exact-resume step2通过：
  `frame_stride=5`、`frame_microbatch_size=32`、batch16/rank，Writer
  `10,125,376`参数，source policy 0 trainable；峰值allocated/reserved
  `67,088,471,040/69,235,376,128` bytes，无OOM/nonfinite，完整四rank
  checkpoint和flow-noise/data cursor可恢复。
- [x] 本版本在这些待办启动前被Belief-v3覆盖，未把v2 mechanics checkpoint
  冒充正式性能轨迹；后续性能与特异性证据由v3/v4活动段负责。

### 历史执行：Belief-v3 Writer（2026-07-25，已覆盖）

本节当时覆盖上方v2的活动实现与待办；现在v2/v3结果都只保留为provenance。

- [x] 原位实现单token
  `Belief_u=[Plan_u(128)|Revision_u(128)]`：Plan只来自最新7维action；
  Revision使用所有更早covering forecasts相对Plan的signed/absolute residual，
  不再包含绝对action或adjacent-only比较。
- [x] Revision显式强度固定为
  `stopgrad(raw source-normalized residual RMS)`，方向为content-only
  `RMSNorm(z_u)`；删除`tau`、训练集分位数尺度与其他人工强度超参数。
- [x] Temporal和LoRA query decoder均改为routing只进Q/K、raw content只进V
  和residual的zero-preserving路径；静态identity、position、lead/count/
  strength均不能凭空生成dynamic LoRA。
- [x] 唯一活动schema/config/checkpoint升级到v3，v2活动配置退役；focused
  shape/gradient/identity/freeze测试通过。
- [x] 固定stride5后完成效率选择：frame-microbatch32、batch20/rank。
  最终raw-RMS实现fresh step1→exact-resume step2通过，Writer
  `10,247,872`参数、source policy 0 trainable、无OOM/nonfinite，完整
  optimizer/scheduler/sampler/RNG/cursor checkpoint可恢复。
- [x] 从fresh identity一次连续训练0→600，每75步保存，不中途主动评测；
  48,000 action queries、2,400 task-video conditions，step600完整
  schema-v3 exact-resume checkpoint已逐文件校验。
- [x] step600完成8 validation tasks×2 videos的低成本内部顺序诊断。
  Revision顺序差异已经恢复，但Temporal的时间常量与单路query read把它重新
  压到effective LoRA的`0.000297/0.000169`（reversed/shuffled）相对L2；
  内部门失败，按owner顺序不运行昂贵的shuffled/reversed paired rollout。
- [x] Belief-v3在内部失败后按门控停止，没有运行多checkpoint rollout；
  后续由32-token visual-state v4从fresh identity重新建立证据。
- [x] v3没有通过双门，未推进cold-start RL，也未使用contrast loss。

## 历史执行：32-token Visual-State Action-Forecast Writer（2026-07-25）

该历史v4的完整设计见`docs/action_forecast_writer_design.md`。当前唯一活动
设计见`docs/action_forecast_writer_v5_design.md`。上方
Action-Memory、temporal-RoPE、Action-Forecast v1/v2和28-slot Belief-v3均为
历史证据，不得恢复其schema、配置、checkpoint或已否决的contrast路径。

- [x] 记录owner最终对齐的端到端设计：32个原生中性state tokens；初始锚点
  `h0`；同时读取`X_t-X_0`与`X_t-X_{t-1}`但不递归累计的visual-state；
  trainable identity-init VL/Action Meta-LoRA；共享flow noise的future-action
  forecasts；绝对时间Plan/Revision；单-token Belief；两层Temporal；
  routing/content分离的LoRA query decoder；完整rank-16 public LoRA。
- [x] 明确保留任务、语言、场景和动作方案的稳定信息；不得建立直接
  `visual-state→LoRA`旁路，也不得用减去时间均值粗暴删除稳定内容。
- [x] 明确Revision比较所有更早covering forecasts与Plan，使用
  `stopgrad(raw residual RMS) × RMSNorm(direction)`，不使用`tau`、分位数
  校准、count/stability加性分支、absolute-action捷径或contrast loss。
- [x] 原位实现新架构并删除旧活动实现；核验32-token原生prompt、shape、
  gradient、identity、冻结对象、rank-16 schema、参数预算、microbatch尾块和
  exact-resume，不做无关全仓仪式性校验。
- [x] 固定stride5，在GPU0–3上重新profile训练batch与frame-microbatch；4–7
  绝不触碰。
- [x] 从fresh identity训练到75 step；先做normal/reversed/shuffled、
  same-task-other-demo、cross-suite wrong-video的逐层内部数值检查，要求差异
  经过forecast、Revision、Belief、Temporal、query直至effective LoRA仍明确且
  由多个tasks/videos共同贡献。内部通过后才做必要paired rollout；此阶段不以
  correct arm绝对成功率为门槛。
- [x] 75-step内部门已通过，无需启动失败修正循环；没有使用对比损失。
- [x] 最终v4从fresh identity连续训练到step2400，每75 step保存；owner随后
  明确终止本轨迹，不再续训，也不再使用方差过大的80-episode快筛。
- [x] 在同一固定8 tasks×50 panel上完整评测现有代表性checkpoint：
  step675/825/900/1200/1275/1500/1875/2100/2400分别为
  `100/109/82/96/94/92/90/90/89`；现有observed-best为step825的
  `109/400`。
- [x] 对step825完成16-reference内部检查及五个固定400 arms：
  correct/same-task-other/wrong/shuffled/reversed分别为
  `109/104/99/148/126`。same-task影响最小，但shuffled显著优于correct，
  reversed也沿改善方向，故“正确时序应被行为上优待”的核心特异性门失败。
- [x] v4性能只略高于四卡rank128 Source-SFT `108/400`，低于旧
  Action-Forecast约`125/400`；证据指向coherent temporal信号在object精确
  抓取任务上可能被转成有害策略变化，而非简单“没有看视频”。
- [x] 按owner最新停止条件，本轮不修改架构、不重训、不推进cold-start RL，
  停下汇报并保留证据供后续外部专家分析；也不自动继续final-32、task-local
  RL、joint oracle或ViVLA。
- [x] owner随后只重开固定首帧shuffle归因：step825固定frame 0、仅保留
  full-shuffle其余帧相对次序得到`136/400`。相对correct仍显著净增27，
  因而随机anchor不是主要原因；不扩展为新的训练或架构修改。
- [x] 面向只能访问远程仓库的外部专家，新增自包含咨询入口
  `docs/action_forecast_writer_expert_consultation.md`；完整记录EMBER思想、
  架构演进、v4模块、内部/rollout/固定anchor证据和待分析问题，并同步修正
  README与活动authority中的旧未来式。本阶段不启动新GPU实验。

## 历史执行：外部复核后的v4根因复审与v5重开（2026-07-26）

本节覆盖上方“等待外部专家分析”的停止点。目标只到能够用直接证据决定下一版
架构；不实现/训练v5，不继续AS，不运行新增full400，不进入RL、final-32、
task-local RL、joint oracle或ViVLA。

- [x] 固定step825、同一validation references和flow noise，实现normal/shuffled
  forecast-order四臂交叉移植：`N→N`、normal forecast换shuffled slots的
  `N→S`、shuffled-context forecast按image identity放回normal slots的`S→N`
  和真实`S→S`。
- [x] 完成16-reference逐层与policy-function诊断。`N→S`几乎完整复现`S→S`
  effective-LoRA/action delta，而`S→N`几乎恢复`N→N`；主效应定位到
  per-image forecast之后，而不是shuffled context首先改变了forecast。
- [x] 只在已有差异最集中的Object-1/Object-3各50个固定states运行四臂定向
  rollout，得到`N→N/S→N/N→S/S→S=49/47/72/82`；不扩展到full400。
- [x] 在同一normal forecasts上拆分Plan、Revision direction、Revision value
  strength和Q/K strength routing。内部与policy-function均证明routing可忽略，
  strength处于train-normal支持范围内且单独行为效应弱。
- [x] 完成Object因子rollout：
  Plan-only/value-strength-only/direction-only/full-Revision分别为
  `61/54/67/75`；direction为主要行为中介，Plan与strength有次级非线性交互。
- [x] 在五条成功Object trajectories、25个经画面与gripper qpos核验的阶段上
  比较12个LoRA反事实。shuffled Revision direction主要改写pre-grasp、close和
  transport的end-effector translation；不是只在参数空间存在的无效差异。
- [x] 验证直接置零Revision不是修复：zero-Revision action相对目标shuffle
  delta放大`2.1–5.8×`且多阶段低相关或反向，Plan/Revision已共同适配。
- [x] 第一轮曾把absolute-time alignment过早写成唯一主因，并拍板保留
  visual-state/Meta的Intent+Transition v5；后续证据已正式覆盖该结论。
- [x] 对step75/300/825在24个train tasks×4 demos上完成post-inference隐藏
  forecast语义审计。AS loss下降时latest-is-best和residual-correction语义持续
  变差；Meta-LoRA关闭反而改善step825隐藏teacher-future误差。
- [x] 证明32-token visual-state不是必要瓶颈：step825 neutralization几乎不改
  forecast；visual coordinates最终主要可预测video progress，几乎不预测
  robot state/action，而raw-image/Meta forecast越来越贴近demo低层translation。
- [x] 完成64 references×8 permutations、same-task demo几何、AS loss/gradient
  与forecast分量拆解；排除permutation lottery、task-consensus回归、endpoint、
  flow noise和“只删除demo detail”。
- [x] 完成Object-1/Object-3五臂root-cause rollout：
  no-VL/no-Action/lead-only/frame-main-only/translation-only为
  `48/50/40/72/79`；translation-only相对correct `49`净增30，且与true
  shuffled `82`无显著差异。
- [x] 将根因修正为“AS可识别性不足→visual-state旁路→Meta低层phase/
  translation latent→absolute-time Plan/Revision放大”；Temporal和
  content-only decoder暂不重写。
- [x] 原样复放Object-1/Object-3的47对correct/shuffled成败翻转并逐帧审计：
  Object-3的31条shuffled-only中，correct有23条明确抓取深绿色干扰瓶；
  两臂首次闭合点中位相差`0.1119 m`。Object-1主要是correct到达更晚、空夹或
  抓后掉落。31条Object-3收益跨22个teacher demos，且同一cached LoRA会随
  init geometry产生相反翻转，证明主效应是不可迁移的低层空间控制绑定，而非
  少数坏视频或shuffle“释放模型容量”。
- [x] 撤回已拍板v5。Intent+Transition保留为删除absolute-time放大器的局部
  候选；下一版必须先解决visual-state必要性、Meta职责、forecast语义gate和
  same-task多demo抽象。完整证据写入
  `docs/action_forecast_writer_v4_root_cause.md`。
- [x] 在该历史停止点还没有v5代码、checkpoint或训练结果；当时不继续AS或
  进入RL，后续GPU工作仍只使用物理4–7。该状态已被下方owner批准并启动的
  Semantic Core + Causal Procedure v5覆盖。

## 当前执行：Semantic Core + Causal Procedure Writer v5（2026-07-26）

唯一活动设计为
`docs/action_forecast_writer_v5_design.md`。上方v4与根因复审是已经封存的
provenance，不得恢复visual-state、7D future-action forecast、absolute-time
Plan/Revision/Belief或旧活动config/schema。

- [x] 与owner完成第一性原理设计对齐：
  - teacher侧无state；
  - language-conditioned PaliGemma image-position hidden形成
    permutation-invariant Semantic Core；
  - fixed native 50-token suffix、VL Meta-LoRA rank4和Action Meta-LoRA
    rank8只产生每帧robot-semantic interaction hidden；
  - 两层global causal Transformer形成可变长Procedure；
  - Core先编译稳定LoRA content，Procedure通过zero-init refiner产生有向修正；
  - 320 routing identities只进Q/K；
  - factor heads保持完整rank-16 public LoRA；
  - 机械设计预算`10,301,440`，比rank128 Source-SFT只多`4,096`。
- [x] 初版训练科学合同曾为每条action独立抽`N=4`条video；该合同已完成
  step0→120与机制检查，但因每rank每step实际生成约24–32套LoRA、每步中位
  `61.39s`而由owner退役，不再续训。
- [x] 固定新训练合同：每rank每step一个task，抽1条teacher video并只生成
  1套one-shot LoRA；完整`B_a`条独立同task action queries全部监督该LoRA，
  形成`B_a`个等权functional losses。4 ranks全局均衡换task，后续task visit
  轮换video；推理严格one-shot；
  不使用contrast/order/margin loss。
- [x] 固定新执行合同：frame stride5；只使用物理GPU4–7；
  `max_frames_per_encoder_call=32`只作长视频显存安全分块且末块不padding；
  联合profile frame/action batch，不用optimizer gradient accumulation。
- [x] 原位替换v4代码/config/checkpoint schema，删除visual-state与误导性的
  action-forecast活动owner；适配AS training、online validation、inference和
  canonical evaluator。
- [x] 完成最小shape/causal/Core-invariance/gradient/identity/freeze/LoRA
  schema/parameter-count/OOM/exact-resume检查；不做无关全仓仪式性校验。
- [x] 封存旧独立N=4估计器的GPU4–7 profile：
  选择`B_a=8`、`N=4`、frame microbatch32；12-step profile从step2真实
  exact-resume到step12，稳态11步中位`61.39s/step`，故正式每段60 steps、
  每10 steps保存一次。`B_a=12/20`和`m40/B8`均因reserved显存只剩不足3GB
  被淘汰。
- [x] 从fresh identity完成第一段AS训练；用固定400 panel选择
  validation observed-best，不使用80-episode快筛。functional loss暂选step40，
  并因内部/闭环均仍在学习而exact-resume到step120。
- [x] 对step40与step120做逐层内部correct/same/wrong/shuffled/reversed检查：
  Core对同帧集合顺序不变，Procedure/refinement有明确有向差异，same-task影响
  小于wrong，差异穿过effective LoRA与policy function。step120的fixed-Core
  Procedure-only有效LoRA shuffle/reverse差异为`0.626%/1.087%`，8/8 tasks
  均贡献，内部结构门通过。
- [x] 内部门通过后完成step40与step120五个固定400 paired rollout。
  step40为`45/52/52/51/51`；step120为`65/59/57/61/65`。step120 correct相对
  step40显著净增20，但相对same/wrong/shuffle/reverse仅`+6/+8/+4/0`且均未
  显著，故行为硬门仍未通过。
- [x] 共享4-video估计器曾完成mechanics/profile，随后由owner退役；旧step120
  与共享4-video checkpoint都不得resume到当前合同。
- [x] 实现并验证单视频完整action-batch估计器；GPU4–7选择F32/B20，最长105帧
  步`6.956s`、常规步`3.109–3.527s`、峰值reserved
  `83,630,227,456 bytes`；B24 OOM、F40无收益。正式封存900-step segment、
  每100步留ckpt。
- [x] 新合同首个约一小时segment已从fresh identity完成：GPU4–7、F32/B20、
  step0→900、每100步checkpoint，全部finite并完整exact-resume state。固定400
  correct-video代表点step100/400/700/800/900依次为
  `62/64/92/76/103`，首段observed-best为step900；尚低于absolute预门但
  700→900仍有净提升，不能判为峰后持续下降。
- [x] evaluator已改为每个worker先取long shard，所有long均被领取后空闲worker
  才取普通task；该规则与一个checkpoint分配几张卡无关。step800四卡24 worker
  实测先取完48个long shards，再动态取24个普通shards；400 rollouts约
  `15.36 min`，相对首轮单卡吞吐提高`2.66×`。实现commit `3b6d9d1`已push。
- [x] 按owner要求先对step900做轻量而非五臂full400的特异性诊断。内部16-reference
  显示Core set对shuffle/reverse近乎不变，fixed-Core Procedure-only有效LoRA
  差异已增至`3.689%/5.764%`并穿过policy action；五臂固定8 tasks×10 states
  为correct/same/wrong/shuffled/reversed=`21/25/14/23/23`。轻量结果只支持
  wrong-video方向性，不足以宣告顺序特异性通过或失败。
- [x] 同一formal root已从step900 exact-resume并正常完成至step1800；
  optimizer/scheduler/sampler/video schedule/RNG、F32/B20与每100步checkpoint
  全部保持不变。fixed400 correct在step1000/1400并列`115/400`，step1700/
  1800降至`71/86`，构成明确峰后下降；选择更低online functional loss且较晚的
  step1400作为唯一正式observed-best特异性checkpoint。
- [x] step1400内部16-reference检查确认Core顺序不变、Procedure有强顺序差，
  但Procedure→effective LoRA/action持续衰减；固定400五臂为
  `115/108/74/113/114`。correct相对wrong净`+41`、`p=2.18e-6`，相对
  shuffle/reverse仅`+2/+1`、`p=0.845/1.0`。v5只通过wrong-video语义性，
  顺序行为硬门失败，停止v5且不进入cold-start RL。
- [x] 按owner条件授权，将side-chat收敛设计完整记录到
  `docs/action_forecast_writer_v5_1_proposal.md`；v5失败条件触发后把它提升为
  唯一下一架构authority。设计为Language-Axial Semantic Core + Causal Action
  Procedure + Slot-Normalized Fusion，机械预算`10,244,872`，不加辅助loss。

## 当前执行：Writer v5.1（2026-07-27）

- [x] 建立新的session-local v5.1 Goal；按`code-architecture-gate`检查owner、
  生命周期和单一canonical替换边界。
- [x] 原位实现v5.1：text-only task-token queries、multimodal task-token
  evidence、token-aligned frame-set attention、language-axis Core、rank4
  Action Meta-LoRA causal Procedure、centered Procedure zero-init AdaLN与
  post-fusion slot block；factor hidden改240，旧v5 checkpoint/schema不兼容。
- [x] 完成最短shape/token alignment/Core invariance/Procedure causality/
  gradient staging/identity/freeze/parameter/schema和真实policy exact-resume smoke；
  GPU4–7上step1→step2完整恢复，source policy冻结且cursor/RNG/checkpoint合同一致。
- [x] 用105帧真实最长视频联合profile frame/action batch与step吞吐；v5.1重新实测
  选择F32/B20，峰值allocated/reserved为`76.93/83.64GB`，最长步`7.25s`、
  常规步`3.25–3.66s`。推理实测选择每卡6个worker全部分摊LoRA生成并原进程
  rollout；全局queue现保证所有未领取long均先于ordinary。
- [x] 依据v5.1吞吐而非继承v5步数，将首个fresh formal segment换算为step900
  约一小时；每100步保存并做512-query online validation。900只是本首段停止点，
  不是未来第二/第三段的固定间隔。
- [x] 从fresh identity完成sealed v5.1首段step0→900；只使用GPU4–7，
  F32/B20、4-rank DDP、每步80 action queries/4 one-video conditions，
  900步共72,000 queries和3,600 video conditions。9个每100步checkpoint与
  exact-resume state均完整，wall `3,622.36s`；没有自动resume。
- [x] 完成旧有放回采样下的checkpoint absolute screen与正式复核。80-rollout
  screen为step100/200/300/400/500/700/800/900 =
  `19/18/15/7/21/17/19/14`；随后按owner指定四卡同时完成
  step100/500/700/900各400条，结果为`82/96/98/84`。当前observed-best为
  step700，但`98/400`明显低于约`110–120/400`预门和旧Action-Forecast
  `125/400`最低参照。
- [x] 只对当前最佳step700完成轻量五臂paired rollout与16-reference内部检查。
  correct/same/wrong/shuffled/reversed为`17/20/7/11/6`；相对correct，
  wrong为`12/2` discordant、`p=.01294`，shuffled为`10/4`、`p=.1796`，
  reversed为`13/2`、`p=.00739`。wrong与reverse有state-pair信号，但贡献仍
  集中少数task，shuffle未显著。
- [x] 内部机制检查确认v5.1结构按设计工作：wrong相对correct的Semantic Core /
  effective LoRA / policy-action中位relative L2为
  `0.204/0.588/0.141`，same仅`0.040/0.116/0.0139`；shuffle/reverse的Core
  近零，但Procedure slots为`1.052/1.680`、effective LoRA为
  `0.528/0.743`、action为`0.102/0.167`。固定Core时保留这些顺序差，去掉
  Procedure后shuffle归零、reverse action仅`0.0021`。
- [x] owner要求特异性检查完成后停止。第二段、第三段、无放回重测、full400
  五臂和cold-start RL均未启动；当前结论是内部机制成立，但absolute不足且
  rollout顺序证据不够跨task稳定，等待owner讨论。
- [ ] AS全部通过后独立启动v5.1 cold-start RL：short AS直到24个train tasks逐task
  至少一次official random-reset success，随后永久关闭action入口转pure reward；
  同样探索validation observed-best与强峰后下降。
- [ ] focused AS/RL通过后更新证据、验证、commit/push并停下向owner汇报；
  不自动推进final-32、task-local RL、joint oracle或ViVLA。

## 当前执行：开放式AS绝对性能探索（2026-07-27夜间）

owner已解除此前“特异性后停止”和“AS全部过门后才可探索RL”的临时执行边界，
授权在EMBER核心逻辑与硬约束内自主训练、诊断、修改架构并可提前做RL-Writer
可行性实验，不需要逐项审核。新的session-local Goal以提高AS Writer
correct-video绝对性能为主，同时保留`task language + exactly one
action-hidden teacher video -> one task LoRA`的核心映射。

硬边界：

- 固定source base、normalization与24/8/8 split；不得让Writer读取action、
  state/proprio、reward、outcome、task ID、filename或simulator privileged state；
- validation只作development选择，不打开test；只使用物理GPU4–7，GPU0–3不查询、
  不进入visible set；
- one-shot public输出仍是一套完整task-specific LoRA；不以额外shared execution
  adapter、bank或第二条并行Writer路径替代核心问题；
- 所有正式比较使用相同state、policy RNG和teacher-video pairing；新canonical
  full400使用每task 50条teacher videos无放回一一匹配50 states；
- 保留可恢复checkpoint、命令、合同、hash与负结果；任何架构替换只保留一条
  canonical active implementation。

夜间证据路径：

- [x] 对v5.1 step700完成无放回paired
  correct/same/wrong/shuffled/reversed各400；现有有放回correct不能混作paired
  baseline。五臂为`88/97/75/65/45`；相对correct的
  correct-only/control-only及双侧exact McNemar分别为same
  `17/26,p=.2221`、wrong `46/33,p=.1766`、shuffled
  `46/23,p=.00762`、reversed `60/17,p=8.91e-7`。所有arm均400 rows、
  每task demo0..49与state0..49一一无放回，逐row seed/noise前缀配对通过。
  step700 absolute仅`88/400`，wrong优势还不显著且主要来自Object-1；
  shuffle/reverse门已有真实方向，但不能据此停止探索。
- [x] 修正四卡评测的ordinary尾部：保持所有未领取long shard全局优先，
  同时按实际`physical_gpu_count × replicas_per_gpu`把ordinary确定性细分到
  至少两个worker波次。标准四卡×6 worker面板由48 long + 24 ordinary改为
  48 long + 48 ordinary，state覆盖不变；focused 48 tests和全仓194 tests通过，
  实现commit `73f171a`已push。
- [x] 同一formal root从step900 exact-resume到step1800；合同不变，每100步保存，
  不因online functional loss单独选best。新增900步、9个checkpoint和完整
  exact-resume state均已封存，训练进程正常退出。
- [x] 用无放回correct400建立足够密的step500–1800闭环曲线，选择同一合同下的
  observed-best。全局best为step1400=`127/400`；step1800=`126/400`但
  paired churn为new28/lost29，不能误判成稳定收敛。
- [ ] 对最终observed-best复查内部Core/Procedure/LoRA/action与完整五条件行为；
  分开判断依赖性、方向性、对source/Core-only的有用性和absolute ceiling。
- [ ] 若特异性成立但absolute长期低于约125，按最早失效层比较Core-only、
  full、去Procedure、跨demo梯度一致性与LoRA到action的功能对齐；只修改被证据
  定位的owner后fresh训练。
- [ ] 可提前进行独立、低成本RL-Writer机制探索，但不得把validation reward写回
  AS输入、把完整AS best冒充cold start、污染test或用RL结果掩盖AS绝对性能。
- [ ] owner醒来前封存当前最好结果、失败归因、资源消耗、代码/Git状态与下一步；
  不自动推进final-32、test task-local RL或joint oracle。

继续/停止判据：

- 绝对分数只要仍低于可信满意区间或曲线未形成充分峰后下降，就继续训练、诊断
  或fresh架构实验；不能因单点略涨而结束；
- 任何absolute提升若依赖wrong video、shuffle/reverse、validation泄漏或其他
  与EMBER核心映射冲突的捷径，一律判为机制失败，不算进展；
- 只有correct absolute达到或实质超过旧Action-Forecast约`125/400`并向
  `148/400`区域推进，同时same/wrong/order五臂与内部因果链均无逻辑漏洞，
  才能考虑封存AS结论；否则继续定位最早失效层。

step900→1800正式resume合同：

```text
workspace/branch: /data/ymdai/projects/EMBER @ main
code commit:      73f171a（评测尾部修正；训练代码相对原formal合同兼容）
source policy:    pi05_source_base_v1_seed7_1k_e2cc238_20260722/step_00001000
resume:           v5.1 formal root/step_00000900
devices:          physical GPU4,5,6,7; 4-rank DDP
scale:            +900 optimizer steps, +72,000 action queries,
                  +3,600 one-video conditions
training:         F32/B20, bf16, exact optimizer/scheduler/sampler/RNG resume
retention:        checkpoints1000..1800 every100; projected ~1.2GB
selection:        validation split only; no-replacement correct400 curve,
                  then full paired five-arm on observed-best
resume policy:    same output root, no overwrite; reject any contract mismatch
```

精确resume命令（在tmux `ember-v51-as-sv1800`中执行）：

```bash
env PYTHONPATH=/data/ymdai/projects/EMBER/src \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,5,6,7 \
OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
/data/ymdai/projects/EMBER/.venv/bin/torchrun --standalone --nproc-per-node=4 \
scripts/train_as_writer.py \
--config configs/pi05_as_writer_language_axial_v5_1.json \
--mode formal \
--source-run /data/ymdai/outputs/ember/pi05_source_base_v1_seed7_1k_e2cc238_20260722 \
--checkpoint /data/ymdai/outputs/ember/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000 \
--tokenizer-path /data/ymdai/ember_data/openpi/paligemma_tokenizer.model \
--data-root /data/ymdai/ember_data/LIBERO-datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a \
--output-dir /data/ymdai/outputs/ember/pi05_as_writer_v5_1_language_axial_dev_r4_seed7_s12000_c199ad3_20260727 \
--resume /data/ymdai/outputs/ember/pi05_as_writer_v5_1_language_axial_dev_r4_seed7_s12000_c199ad3_20260727/checkpoints/step_00000900 \
--stop-after-step 1800 \
--num-workers 2 --log-every 10 \
--allow-contract-compatible-code-resume
```

### 2026-07-27 21:00 UTC状态更新

- step900→1800 exact-resume已经完整结束；checkpoints1000..1800、optimizer、
  scheduler、sampler/data cursor和四rank RNG state均落盘，训练现场已退出。
- canonical无放回correct400曲线为：
  `100:83, 500:98, 700:88, 900:86, 1000:114, 1100:111,
  1200:114, 1300:92, 1400:127, 1500:95, 1600:92, 1700:65,
  1800:126`。全局observed-best为step1400；step1800近似复现aggregate，
  但1400→1800有`28`个new successes和`29`个lost successes，不是稳定平台。
- step1400内部16-reference机制检查已完成。same/wrong/shuffled/reversed相对
  correct的effective-LoRA中位relative L2为
  `.096/.673/.514/.673`，policy-action为`.0149/.1329/.1035/.1823`；
  Core对order近似不变，固定Core只替换Procedure仍保留完整order差异。
  因此没有v4-shuffled式信息旁路；当前最早科学问题是有用控制能力不足。
- 全曲线两个spatial validation tasks在任一checkpoint合计最多`2/100`，
  Goal-3也最多`3/50`；高分主要来自Object-1/3、Goal-6和Long-1。
  相邻200-step Writer更新余弦接近零，且任务能力大幅迁移。机械沿原
  `~3e-4`高学习率续训不再是优先动作。
- evaluator在commit `082090f`加入同物理GPU EGL close/create串行化、
  跨失败/resume累计wall与shard证据，以及显式封存的
  `writer_lora_b_scale`；196项全仓测试通过并已push main。历史step500/1600
  resume roots已正式聚合为`98/400`和`92/400`，累计attempt分别为2次。
- 当前tmux `ember-v51-scale`在GPU5/6/4/7分别运行step1400的
  `1.25/1.50/1.75/2.00×` LoRA-B full400；每个run严格复用同一个
  identity SHA256
  `40e53b52caac343275fdb63f98bb224fe48f88b32037ad50071505cfc3873043`
  的400-entry无放回cache，A因子、state/video双射和所有paired RNG不变。
  完成后先按多task paired收益选scale，再在选定scale跑完整五臂；若不能把
  absolute和task breadth实质抬高，下一优先级是从step1400开启fresh optimizer
  的一小时低学习率稳定阶段，而不是继续原高学习率轨迹。

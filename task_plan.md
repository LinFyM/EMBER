# EMBER Task Plan

最后更新：2026-07-25。长期 EMBER Goal 是完成共享 π0.5-LIBERO source base、one-video Writer、source baselines、seen/wrong-video机制证据、final single-seed test、test-only三臂RL和联合target-action oracle；当前 active 子任务由本文末尾“32-token Visual-State Action-Forecast Writer”段及 `docs/action_forecast_writer_design.md` 定义。

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
- [ ] 所有适用阶段先短profile并由loss/reward/behavior曲线决定廉价screen间隔，只给必要候选完整validation；若owner为某阶段给出时间上限，到上限仍未充分训练则记录后停止。当前focused Action-Forecast AS/RL Writer不设总时间上限；Source-SFT comparator已经封存，不在本子任务重训。

## 历史 Goal：四卡Action-Memory / Source-SFT / cold-start RL-Writer上限（2026-07-23，已被下方Action-Forecast执行段覆盖）

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

### 当前快速子任务：temporal-RoPE Writer（2026-07-24）

- [x] 原位替换canonical temporal owner：1D RoPE + 4个condition-only learned memory queries；保留bias、Action Memory、Meta-LoRA、信息墙与完整LoRA decoder。
- [x] GPU0–3原生global64两步profile通过，不做梯度累计或8卡逻辑batch模拟。
- [x] fresh训练到step500并封存step400/500；500-step训练body wall为`1188.6s`，两个checkpoint及完整四rank resume state均通过manifest校验。
- [x] 用同一paired 8×50 validation panel评测两个checkpoint；step400/500分别为`108/400`与`98/400`，逐task上step400处处不差，故冻结step400为本轨迹observed-best。
- [x] 只对step400运行视频/单帧/倒序/打乱特异性诊断并停止：固定语言时跨suite错误视频令有效LoRA相对变化中位数为`0.2267`，同task另一demo为`0.0403`，但倒序/打乱仅`0.00937/0.00699`。当前Writer已利用视频任务内容，却基本未利用动作时序；不继续本轮训练、contrast或RL。

## 历史执行：Action-Forecast Writer v1（2026-07-24，已覆盖）

下列任务是Action-Forecast v1已完成历史，不是活动设计，不恢复旧checkpoint。
当前设计见`docs/action_forecast_writer_design.md`。

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
- [ ] 从fresh identity正式训练到step600，每75步保存；完整评测step300/600。
  若没有明显且稳健的峰后下降，以600-step大段继续到1200及之后。
- [ ] 对最佳或暂时最佳checkpoint比较旧AS `125/400`，并完成
  correct/wrong/shuffled/reversed。双门通过才推进独立cold-start RL；否则只
  排查可纠正的明显错误，不加contrast，若无明显错误则停止向owner汇报。

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
- [ ] 多checkpoint correct-video validation暂未启动。owner要求本轮把特异性
  诊断与归因闭合后停下汇报；后续应先把global/time-centered memory拆成独立
  路径并重新训练，再恢复性能曲线评测。
- [ ] 只有AS性能与视频时序特异性双门通过，才推进独立cold-start RL；
  不使用contrast loss，无法以第一性原理架构修正通过时停下向owner汇报。

## 当前执行：32-token Visual-State Action-Forecast Writer（2026-07-25）

完整且唯一的活动设计见`docs/action_forecast_writer_design.md`。上方
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
- [ ] 最终通过的架构从fresh identity直接训练0→1200，checkpoint可每75 step
  保存但不中断训练；随后挑选8×50 validation候选。若best后未出现明显、远超
  rollout噪声、跨多个tasks且独立panel复测成立的下降，每次exact-resume增加
  600 step继续探索。
- [ ] 最终AS best应至少接近并力争超过旧Action-Forecast约`125/400`；四卡
  rank128 Source-SFT `108/400`是已知基线，旧八卡`122/400`仅为stretch参考。
  被选best必须同时保持correct/wrong-video及normal/reversed/shuffled特异性。
- [ ] 双门通过后推进独立cold-start RL Writer：从fresh identity做短AS，
  24个train tasks逐task至少一次official random-reset success后永久关闭
  action入口，转pure reward，并同样充分寻找validation best与明显峰后下降。
- [ ] focused AS/RL完成后停止并向owner汇报，不自动继续final-32、task-local
  RL、joint oracle或ViVLA。

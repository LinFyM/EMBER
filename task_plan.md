# EMBER Task Plan

最后更新：2026-07-22。当前长期 Goal 是完成共享 π0.5-LIBERO source base、one-video Writer、source baselines、seen/wrong-video机制证据、final single-seed test、test-only三臂RL和联合target-action oracle；不是停在generic base feasibility或任一局部阶段。

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

ViVLA-style matched baseline和source-only outer learning为时间允许时的后续项，不阻塞核心Goal complete。RL-Writer如在零warm-up和微量AS warm-up后均无法获得训练信号，可用完整失败证据关闭该路线，不伪造第三臂。

## Phase 0：generic π0.5 feasibility

状态：已完成。

- [x] 封存24/8/8目标split。
- [x] official-compatible π0.5 evaluator与24-train interface stats mechanics。
- [x] 8 test tasks × 50 fixed states：generic `pi05_base` 为 `0/400`。
- [x] 结果seal、49项测试、commit/push完成。

该结果只作原始模型校准。当前活动路径从Phase A继续。

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
- [ ] 一卡一训练rank为默认；若评估每卡多replica，8卡replica数必须一致且GPU0无额外角色。
- [ ] checkpoint/output不得覆盖；resume必须校验完整state与合同兼容性。
- [ ] 所有适用阶段先短profile并由loss/reward/behavior曲线决定廉价screen间隔，只给少量候选完整validation；约120分钟是guardrail而非目标，到上限仍未充分训练则记录后停止。task-local按每个方法覆盖8 tasks的合计时间计费，不按单task计费。

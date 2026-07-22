# EMBER Task Plan

最后更新：2026-07-21。当前长期 Goal 是完成共享 π0.5-LIBERO source base、one-video Writer、source baselines、seen/wrong-video机制证据、final single-seed test、test-only三臂RL和联合target-action oracle；不是停在generic base feasibility或任一局部阶段。

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

状态：source corpus与recipe已封存；高吞吐evaluator实现与机械门禁完成，待final source checkpoint做1/2/3 replicas真实rollout/s profile。

- [x] 调研官方/成熟π0.5 action fine-tuning与LoRA项目；锁定OpenPI/LeRobot revision、full-SFT recipe、下游LoRA targets/rank/alpha/dropout与identity init。
- [x] 对LIBERO-90与目标40 tasks完成3600对language/BDDL/semantic/composition specification audit。
- [x] 排除19个完整任务等价项并封存71个active source IDs、可执行规则与hashes；包含已知task44/task77及其余semantic aliases。
- [x] 核验71 tasks×50 successful episodes、529,173 frames、52,710,755,898 bytes；封存source-only q01/q99 normalization，validation/test numeric reads为0。
- [x] 下载并按opaque SHA/HDF5 schema-only核验目标40 tasks×50 episodes；封存24/8/8 target-data manifest，未解码target action/state/reward/terminal/video值且未改变task IDs。
- [x] 已将当前“一task/一GPU”评测改成按 `episodes × horizon` cost-balanced state shards、动态队列、持久model/env，并删除旧静态活动入口。
- [x] 在同一canonical evaluator内加入PI05 AS-Writer逐rollout materialized LoRA、one-video哈希schedule、role-preserving cross-suite wrong map及逐row可重算证据；没有新增第二套runner。
- [ ] final source checkpoint后完成每卡统一1/2/3 replicas与未来Writer异LoRA functional batching的真实profile。
- [ ] 确认所有卡CUDA进程数相同、GPU0无额外角色；只按真实rollout/s选实现。

## Phase B：共享 π0.5-LIBERO source base

- [x] 建立唯一canonical π0.5 full-SFT runner，严格模型加载、task-balanced no-replacement sampler、EMA、atomic checkpoint、metrics reconciliation与完整per-rank RNG/cursor恢复。
- [x] 完成相机mask修正后的8卡真实profile；选定8×microbatch32、global batch256、EMA与GPU-local NUMA binding，稳态47.75 examples/s，单卡reserved 71.18GB。
- [x] 从同一个不可变step-1 checkpoint做两次8-rank恢复；loss/grad/RNG/cursor完全一致，policy/EMA最大独立NCCL末位差分别为1.49e-8/3.73e-9。
- [ ] 正式attempt3正在隔离worktree从generic base fresh训练；最近只读观测step2050，8卡各约69GB/100%利用率，loss/gradient finite、约47.37 examples/s，首个step5000 checkpoint前不作行为结论。
- [ ] 从generic `pi05_base`按成熟recipe在过滤后的LIBERO-90 source corpus上联合action-SFT；若用LoRA则merge成base。
- [ ] 以已锁定8卡配置完成30,000 optimizer steps；一卡一rank，真实显存平均预留约10GB，checkpoint每5,000 steps且只保留最新完整状态。
- [ ] 根据loss与快速行为screen避免过训，不追求高ceiling。
- [ ] 在全部目标40 tasks上做小型快速screen，确认source base已经开始在多个tasks产生部分真实成功，不能只靠单个易task aggregate；保存每task原始counts。
- [ ] 冻结base、normalization、model/data hashes，作为全部后续方法共同起点。

## Phase C：AS-Writer development

- [x] 将Writer core、target feature-cache与exact-resume训练owner适配到π0.5成熟LoRA空间；同task video/action episode使用独立确定性schedule，Writer只见language+one action-hidden video。
  - 已封存38 targets / 76 tensors / 1,287,168 parameters的rank16合同；真实checkpoint metadata与meta模型结构一致。
  - 已修复PI05 forward签名、BF16/FP32 LoRA dtype保真，并在Writer入口对`offsets=[0,L]`单视频合同fail-close；functional/materialized parity已测试。
- [x] 封存target40 metadata/hash authority与development 32-task视频cache配置；cache和AS formal均保持pending real profile，不将候选batch/steps冒充正式配方。
- [ ] 24 train tasks均衡混合；source base冻结，actions只进functional loss。
- [ ] 先profile约短时loss/吞吐，将wall-clock换算为steps；checkpoint频繁exact-resume。
- [ ] 单次训练不超过约2小时。用loss斜率决定何时值得运行便宜val screen，完整val只评少量候选，尽快选择接近饱和checkpoint。
- [ ] 报告8 validation tasks逐任务raw success与视频采样seed。

## Phase D：RL-Writer development

- [x] 建立共享π0.5 reward mechanics：official random BDDL reset、10-step settling、suite horizon、显式逐replan flow-noise seed、只执行/监督前5 actions、successful-episode等权与不可变interaction ledger。
- [x] 建立fresh zero-AS RL-Writer唯一活动owner、8-rank task/video full-cycle schedule、Writer-only functional update、完整checkpoint/RNG/cursor/ledger exact-resume；formal仍被source screen与真实profile硬门禁阻止，尚无科学结果。
- [ ] 从随机Writer、零AS warm-up直接跨24 source tasks用官方random-reset reward联合训练。
- [ ] 若无正信号，加入极少量AS warm-up并明确记录teacher-action consumption；不得从完整AS-Writer继续。
- [ ] 仍无法启动时保存完整reward coverage/interaction/failure evidence并暂停路线。
- [ ] 若成立，根据validation快速寻找接近饱和checkpoint；保存worker RNG、seed schedule、interaction cursor和exact-resume state。

## Phase E：Source-SFT、seen与视频因果证据

- [x] 建立development-only Source-SFT authority与唯一PI05训练owner：24 train tasks共享一套38-target LoRA，source EMA冻结，checkpoint保存adapter/optimizer/scheduler/per-rank RNG/确定性sampler与metrics cursor；现有evaluator按静态adapter一次安装并保留普通batch。
- [ ] 从同一source base在24 train tasks上联合训练一套shared Source-SFT LoRA；按validation独立选择最佳，不强制匹配AS-Writer steps/data。
- [x] 在outcome前按specification-only SHA256规则封存四suites各2个、共8个seen tasks（global IDs 0,2,15,12,21,28,39,37）；policy outcome与trajectory value reads均为0。
- [ ] 比较source base、Source-SFT、AS-Writer、可用RL-Writer的seen performance。
- [x] 封存同split role、按suite循环和排序ordinal构造的cross-suite wrong-video机械map；correct/wrong两臂保持同一language、task、init state、policy RNG、video seed与demo ordinal。
- [x] canonical evaluator在同一dynamic queue、persistent worker和raw-row schema内支持frozen AS-Writer或RL-Writer；两者共享correct/wrong mapping与video seed，artifact kind/checkpoint axis分别fail-close。
- [ ] 用正式checkpoint运行并报告AS/可用RL Writer的correct/wrong/base逐任务结果。
- [ ] 冻结AS-Writer、RL-Writer、Source-SFT的architecture、LoRA空间、optimizer与最终训练steps。

## Phase F：32-source final retraining与zero-interaction test

- [ ] 将8 validation tasks合入source，形成32 source / 8 test；不改test IDs。
- [ ] 第一轮只用一个training seed，从规定初态分别重训AS-Writer、RL-Writer（若成立）和Source-SFT。
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

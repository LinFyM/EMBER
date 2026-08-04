# EMBER Task Plan

最后更新：2026-08-04 UTC。

本文件只保存尚未完成的长期闭环与当前执行顺序。历史实验过程见
`findings.md`、`progress.md` 和 Git；实时进程见
`docs/active_session_handoff.md`；迁移状态见
`docs/a100_to_bci_migration_handoff.md`。不得从历史 ledger 中恢复已退役 recipe、
runner、split、路径或 GPU 权限。

## 当前Task-Relative Flow-Credit推进（2026-08-04）

- [x] 复核历史RL：确认旧PI05 Writer-RL只是success-filtered executed-prefix BC，没有
  failure、advantage、old/current ratio或trust region；不得恢复或改名复用。
- [x] 封存task-relative flow-credit design：fresh v6 AS cold start，随后teacher action
  永久关闭；full24、K4 official random resets、task-local LOO advantage、Nmc4、PPO正项
  与SPO负项，Writer-only update。
- [x] 原位恢复唯一v6 Writer并替换canonical RL/checkpoint/evaluator；删除Target-Owned、
  flat task-local与旧RL并行活动路径。210项全仓分组回归与architecture hard gate通过。
- [x] 在live空闲`gpu02:1,2,3,4,5,7`完成longest105 AS profile：logical B20/B2，峰值
  allocated/reserved`34,948,858,880/44,816,138,240` bytes，三步finite、0 OOM/clip。
- [x] 独立fresh0→1/exact-resume1→3通过；1,440 queries、72 videos、五主block到step3
  可达、source policy trainable=0；AS config已seal。
- [ ] clean commit/push至origin main，从fresh identity运行AS 0→25。
- [ ] 用canonical reward cycle的pre-update K4 ledger做24 train task coverage；不过则只
  exact-resume同一AS root到下一25-step边界，通过则冻结coldstart并永久关闭action data。
- [ ] 完成最长failure、Nmc4、两epoch RL profile；机制通过后做两个outer cycles与paired
  screen/correct400，持续以single checkpoint strict`>150/400`为最低成功门。

## 历史BCI Policy-Target-Owned Factor推进（2026-08-04）

- [x] owner恢复长期`>150`目标，要求科学问题自行深入分析后继续推进，不再为中间判断
  请求确认；继续禁用subagent、保持one-shot与效率优先。
- [x] 重新分析两套direct rank-128 Source-SFT step400的effective LoRA：两者同样以
  q-dominant低秩更新为主，排除“Writer near-rank1本身导致漂移”。
- [x] 确认SFT的policy-target specialization跨配方稳定：q/v layer-energy profile
  Pearson=`.9931/.9904`、rank correlation均`.9835`，对应target BA cosine均值
  `.8450/.8529`；而Direction Store的q/v跨层余弦`.93--.97`、能量近uniform。
- [x] 封存
  [`docs/action_forecast_writer_target_owned_factor_design.md`](docs/action_forecast_writer_target_owned_factor_design.md)：
  保留Target-Bound Core/private A-E-D/rank-read，删除task Direction Store；76个公开
  A/B tensors各自拥有完整bias-free`1024→256→width`head，不加正交/rank/energy约束。
  机制与AS objective解耦，后续可直接由rollout reward训练。
- [x] 原位替换唯一canonical Writer、config/schema/checkpoint/inference/internal
  decode路径；删除只服务Direction Store的额外frozen text-anchor forward和专用测试，
  不保留并行模型/runner。精确参数预算预期`47,857,920`。
- [x] compileall、config/fresh checkpoint family和89项Writer focused tests通过；
  architecture diff净减少334行，无新增hard signal。环境完整评测合同在显式加载
  `.env.local`资产路径后52/52通过。
- [x] authority与canonical替换以`20479d3`clean commit/push；live六卡formal-seed
  fresh0→1工程smoke finite且显存通过，但最长仅82帧，已明确不冒充longest105。
- [x] 根修profile/formal seed切换：磁盘config固定formal seed20260722，profile mode
  自动解析声明的seed172并写入run contract，不再手工改seed；24项聚焦测试通过。
- [x] 实时重查后在`gpu02:1,2,3,4,5,7`完成clean seed172 longest105、logical B20/B2、
  fresh0→1/exact-resume1→3；峰值reserved`43.936GiB`，step2起五主块finite/nonzero，
  config已恢复formal seed20260722并seal，profile权重不进入正式轨迹。
- [x] profile封存后从fresh identity完成0→200、每25保存；50/100/150/200严格配对
  correct400=`99/76/86/68`，breadth=`6/6/7/5`，union/intersection=`136/37`，
  envelope gap37。winner macro50=99，低于Direction Store129、v6-fast143和门151，
  按预注册判据不续400。
- [x] winner完成六卡refs1五条件内部分析。76 heads已把q/v跨层BA余弦降到约0，证明
  policy-target ownership生效；但层能量过度集中、LoRA norm下降，Program差异扩大的
  BA没有转成等比例action差异。完整训练梯度又显示factor task directions近随机正交，
  同task+demo没有稳定重现。正式拒绝policy-target sharing为主要根因。
- [x] 按owner此前要求，在本轮rollout与全部分析封存后暂停；不启动下一架构、训练或
  评测。下一讨论边界是重构condition-to-policy credit，使video条件获得policy-aware、
  闭环有用且可由AS或未来reward共同训练的累积方向，而不是继续加head/gate/scale或
  强制SFT几何。

## 当前BCI Semantic Direction Store推进（2026-08-03）

- [x] owner解除VR结果后的阶段暂停：继续one-shot，取消Writer参数量软上限，优先
  重构条件生成方向的存储/组合，允许配套修改训练；继续禁用subagent并保持效率优先。
- [x] 重新按内部证据区分task drift与functional-loss不可预测性：后者不能单独解释
  漂移；SFB的核心缺口是已学会activation routing，但约97%梯度仍进入共享factor且
  方向持续轮换。
- [x] 完成`Semantic Direction Store Writer`设计authority：frozen text-only semantic
  anchor在24 train languages上建立8个固定centers，每task稳定top2；每store拥有完整
  1024→256→factor-width input/output参数，预计Writer为37,355,776参数。保持完整
  Core/A/E/D value、one-shot、single LoRA与信息墙。
- [x] 原位替换canonical SFB factor path，不保留并行架构；增加BCI B2切片可重建的
  keyed independent Beta/Gaussian sampler，退役VR estimator的活动配置。
- [x] 建立仅基于24 train language的center authority；完成route、独立W_out、sparse
  gradient、identity、freeze、B20/B2 parity与fresh/resume聚焦验证。
- [x] live比较`gpu01`/`gpu02`后用`gpu02:0--5`六张空闲卡完成longest105真实profile；
  fresh0→1/exact-resume1→3通过，峰值reserved`43.893GiB`，无需改变logical B20、
  full24 raw mean或一次AdamW。根治rank-local构造与NCCL生命周期错位，并封存BCI
  A40/NCCL2.28显式SHM transport fail-fast到代码和`AGENTS.md`。
- [x] clean pushed`91feeef`从fresh identity完成0→200：200 finite macros、96,000
  queries、4,800 videos、8 checkpoints；四点paired correct400为
  `129/107/120/129`，breadth=`7/7/7/5`。macro50以同分更高breadth成为single
  winner129，未超过v6-fast143或严格门151，不续到400、不做五臂。
- [x] 完成macro50 refs1五条件内部分析：固定route和A/E/Core→BA/action路径成立，
  但same-task Program relative-L2 `.93377`到factor/BA只剩`.01935/.03242`；16个
  active rank坐标的stable rank仅`1.000043`、首奇异值能量`.999957`。正式拒绝“只靠
  独立完整factor stores解决漂移”，定位到Program→public A/B的多维方向形成失败。
- [x] 根修复六卡内部分析的历史4-rank假设：任务LPT分配与最终Cartesian sealing均绑定
  实际`world_size`，clean`a115b06`六rank/8-task/5-condition真实分析完整封存；规则写入
  `AGENTS.md`。当前无GPU工作，按owner要求暂停讨论。

- [x] 完整阅读authority、迁移handoff、架构/recipe设计、代码与历史证据；核清
  data/model/tokenizer/source checkpoint、formal outputs、环境与simulation assets。
- [x] 核验A100→BCI迁移清单、hash、223项旧环境CPU回归和四卡训练/评测验收；新资产
  统一位于项目`data/`、`models/`、`runs/`和`evidence/`。
- [x] 实时比较`gpu01`/`gpu02`；首轮仅`gpu02`的0/1/2/3/4/7空闲，六卡collective
  通过，未触碰有他人任务的卡。
- [x] 在不改变逻辑B20、full24 raw mean或优化器合同的前提下，实现policy B2物理
  microbatch与6 ranks×4 tasks拓扑；23项focused CPU测试通过。
- [x] 未冻结工程profile完成fresh0→1/exact-resume1→3；峰值allocated/reserved约
  `34.97/47.11GB`，三步finite，五主block从macro2起可达。
- [x] 实现提交/push为`391f183`；从clean pushed commit重放同一最长路径与
  exact-resume并seal。一次setup stall未复现，最小collective和原命令重试均通过；
  profile checkpoint不得warm-start。
- [x] 预注册BCI VR formal launch contract：fresh 0→200、六卡、logical B20/B2、
  96,000 queries、every25、1.5GiB峰值预算、fresh root、tmux/log、启动与resume门。
- [x] 首次formal在checkpoint前发现A40 overlay误保留profile teacher seed`172`并安全
  停止；修回正式`20260722`、新增sealed seed fail-close，partial root禁止resume/评测。
- [x] 从fresh identity完成VR 0→200、every25；200步/96,000 queries/4,800 videos、
  8 checkpoints和全部hash/信息墙门通过。严格配对correct400为
  `76/88/126/107`，breadth=`7/4/7/5`，single winner126；四点漂移与matched
  VR→SFB机制/held-loss分析完成，正式负裁决，不续到400、不做五臂。
- [ ] 长期Goal仍是同一single checkpoint correct严格超过`150/400`并尽可能提高；
  不复用VR checkpoint、不使用subagent、checkpoint融合、挑video或信息捷径。

## Post-seal A100研究窗口（2026-08-02 19:18 UTC起）

- [x] owner重新授权约十小时GPU4--7研究窗口；以`f9a144c`为迁移封存基线，创建外部
  delta ledger与独立`codex/postseal-target-bound`写worktree。
- [x] 完整复核CV-ADR、historical Coherent-Procedure与Target-Bound设计；判定首项
  真实证伪应保持mean-backed Core，并让task/Core语义在38 targets和A/E/D私有时序
  读取之前进入，而不是再加gate/scale或硬task-ID experts。
- [x] 将远端Target-Bound实现`b260a57`无冲突移植到`f9a144c`，得到`fbbb784`。
- [x] 恢复frozen Python/CUDA环境，完成48项focused CPU vertical path并立即push。
- [x] 只对GPU4--7做一次live preflight；完成longest105 B20三cycle、fresh/exact resume。
- [x] Target-Bound clean frozen`cfd26df`完成fresh0→200并并行评测
  50/100/150/200 correct400=`75/120/90/110`；不续到400、不做昂贵五臂。
- [x] winner macro100内部反事实证明A、D、causal memory、Core与Program均传到
  effective BA/action；最早剩余失败接口定位为shared factor conditional coexistence，
  而不是视频路径断路。
- [x] 完整实现Semantic Factor-Basis并push`e87363f`；11,159,296参数、55项聚焦
  回归、longest105 B20三macro及fresh0→1/exact-resume1→3均通过，seal为`f5ddfe3`。
- [x] clean frozen`f5ddfe3`完成fresh0→200；paired correct400为
  `69/91/118/127`，macro200 breadth8，50→200 gained/lost=`68/10`，但
  150→200仍为`38/29`。
- [x] 同一root exact-resume 200→400并并发评测250/300/350/400；完整曲线
  `69/91/118/127/117/81/126/120`，single winner仍为macro200，第二小时明确轮换。
- [x] 在不替换SFB架构的前提下实现variance-reduced functional estimator；mode接线
  修复与回归commit为`50662a8`，只改变exact-marginal flow time/noise批内依赖。
- [x] 完成VR longest105 B20三macro与formal-seed fresh0→1/exact-resume1→3；matched
  早期梯度稳定性小幅改善，但尚无fresh0→200或closed-loop证据。
- [x] 02:42 UTC停止全部GPU工作；不在剩余窗口启动无法完成paired评测的新训练。
- [x] 封存并交付最终代码/文档及34行、16,483,938,529 bytes post-seal
  `must-transfer`增量清单；迁移后由owner重新授权才运行VR fresh0→200与
  50/100/150/200 paired correct400。

## A100清理与BCI迁移准备（2026-08-02）

- [x] 核验EMBER/MemLLM Git、工作区、tmux和训练/评测进程；没有活动实验需要继承。
- [x] 创建并验证EMBER 138-ref全量bundle；复验MemLLM 186-ref历史bundle与SHA。
- [x] 清理52个Writer LoRA caches、138个profile/smoke/resume/WIP roots、退役
  SmolVLA outputs/numeric数据、旧feature cache、endpoint LoRA tensors、reseal/
  capacity roots和Codex archive；每批有外部manifest与SHA。
- [x] 定向测试发现active `hf-libero`的simulation-assets symlink依赖原
  `ember_assets`；按精确HF revision只恢复426.57MB必需snapshot，原4.28GB多余缓存/
  revision不恢复，4个原始contract失败测试重新通过。
- [x] 将source step1000精简为selected raw policy inference asset；保留policy、
  trainer state和manifest，formal inspector通过，明确不再支持source exact resume。
- [x] 删除可按精确revision重下的generic `lerobot/pi05_base`，封存revision、bytes和
  SHA；LIBERO exact dataset、tokenizer、formal outputs与feature cache v2保留。
- [x] 保守保留原封存60个正式checkpoint roots和406个complete evaluation roots，
  并登记post-seal新增2个正式训练root和12个formal correct400 roots；它们是task
  漂移、checkpoint轮换和架构×recipe混杂的唯一证据，不只留winner。
- [x] 封存EMBER/MemLLM完整dependency freeze；验证后删除EMBER venv/package cache；
  owner关闭MemLLM venv消费者后也删除其7.60GB环境，两者都列为BCI重建项。
- [x] 删除55个clean辅助worktree、36个本地实验branch和obsolete stash；历史由
  bundle保存，Target-Bound仍在GitHub远端分支。
- [x] 评测preflight支持`EMBER_STORAGE_ROOT`，不再写死`/data/ymdai`；定向测试通过。
- [x] 重写README、AGENTS、active handoff和execution brief，新增迁移handoff与机器
  可读资产表；A100 Codex不迁移，新Codex从Git authority接手。
- [x] 完成最终全量验证、cleanup manifest总SHA、Git commit/push与两repo状态核验。

迁移由后续智能体执行。本计划不授权跨机写入或迁移后GPU实验。

## 当前交接顺序（2026-08-02）

- [x] 恢复exact v5.2 topology到mature task-complete/B20/long-first/
  fast-decay400 config，并完成最长视频profile与exact-resume smoke。
- [x] 对v5.2 step900重新生成400套correct-video LoRA并完成零rollout几何分析；
  证明近rank1来自建设性coherence，不是坐标能量失衡或负向相消。
- [x] 撤回Coherent-Procedure/B-only residual，封存完整SPG模型与CP-24训练设计。
- [x] 新session完成全部authority、代码、Git历史和正式artifact审计，并从独立
  frozen `60f4508` worktree启动v5.2 task-complete fresh macro0→400。
- [x] v5.2 macro150/200/350/400 paired correct400完成，为
  `51/91/106/120`；winner macro400五臂`120/109/107/111/124`，exact50几何和
  Core→Procedure→LoRA→action内部传递完成，不融合checkpoint。
- [x] v5.2 run挂起后充分阅读全部文档、代码、Git历史与正式outputs，形成完整
  v1→SPG证据模型。
- [x] 独立复核并实现canonical SPG+CP-24；精确参数`10,633,216`，全仓
  `201 passed`，architecture guard无hard violation。
- [x] SPG最长105-frame B20四卡三macro profile通过；定位并修复共卡NCCL
  chunk只入队导致的CP-24 starvation，修复后step为
  `20.536/18.578/18.546s`且全部主模块梯度可达。
- [x] 同一clean `f6d4876`和formal seed完成fresh0→1→exact-resume1→3；step1
  文件bitwise不变，三步metrics/LR/cursor连续，CP chunk gather/sync严格对应。
- [x] resume seal已提交并push；从最终clean `79fb7ee` frozen worktree fresh启动
  macro0→200，首macro的24-task/B20/long-first/CP同步合同通过。
- [x] SPG macro0→200与四个paired correct400完成：`97/115/77/100`；一小时门
  失败，不续到400、不做五臂。
- [x] 完成SPG exact50四候选几何、macro100 refs2分层反事实、24-task gradient
  coherence和checkpoint drift分析；最早失败定位为compiler routing同质化，
  CP-24无法恢复近正交task innovation。
- [x] 按组件×recipe重审v7/v8/v10/Loom/Recenter/Core-Program/Prior/Target-
  Spectral；只封存局部强负机制，不整体判死与fast recipe混杂的架构思想。
- [x] 封存Unified Causal Program设计authority和现有B20 phase-variance审计。
- [x] 实现UCP canonical路径、raw full24 Gram诊断和无偏20-strata B20；删除旧SPG
  Core add/global mixer与CP投影active path。真实参数`7,683,328`，全仓
  `203 passed`；fresh formal config先保持pending，由下一项live evidence解封。
- [x] 完成shape/mask/identity/freeze/gradient/resume和最长105-frame B20 profile；
  三macro峰值reserved约77.62GiB，formal-seed fresh0→1→exact-resume1→3逐文件
  不变，选择B20。
- [x] 提交并push UCP live seal `c94f1c6`，从新的clean detached commit建立formal
  frozen worktree并fresh启动macro0→200；首macro合同健康，未从smoke warm-start。
- [x] 完成clean frozen `c94f1c6` fresh macro0→200和50/100/150/200 paired
  correct400：`82/117/100/110`；union169、single best117，门失败，不续到400、
  不做五臂。
- [x] 完成macro100 refs1内部纵向和CUDA batch-shape诊断；保持B5 carrier后
  canonical重算各层严格一致，reader路由健康，但dynamic A/D写出仅约2–5%。
- [x] 定位首次`a4b06f5` exact50失败为rank1本地异常被NCCL gather掩盖；失败root
  只有run contract。实现reference上下文、rank failure artifact、torchrun
  fail-fast和analysis-only Gloo控制组，不修改训练protected owners。
- [x] 新refs2精确暴露rank1异常为`libero_spatial task3/reference1`的rank-gauge
  sanity失败，并验证failure artifact与torchrun立即收割；加入BA/action/raw A/B
  判别量；确认BA误差仅`1.299e-9`、bf16 action drift为`.002047`，修正错误的
  位级动作硬门而保持BA `2e-5` fail-close。
- [x] 用新clean `c4b85e8` root验证refs2通过；四rank共16 rows、无failure，随后
  用另一root启动exact50并封存clean provenance。
- [x] 完成exact50：8 tasks×50 references共400 rows、四rank各100、0 rollout、
  无failure；封存逐task same-video variance、Program→BA→action、消融和有效LoRA
  几何，确认pooled BA/action条件方差仅`.09008%/.01656%`。
- [x] 在独立write worktree完成并集成UCP exposure-matched serial-4单路径：六phase
  重建同一full24 cycle，LR按cycle阶梯重复；全仓`233 passed`，fresh incompatible
  config/checkpoint及midcycle cursor通过，architecture guard无hard violation。
- [x] 完成serial-4最长105-frame B20 profile：18 updates/3完整cycles、B20 finite、
  task38/demo36真实105 sampled frames；formal seed fresh0→1→resume1→3→跨cycle
  boundary到7，step1/3文件不变、scheduler/cursor连续，canonical config已seal。
- [x] 完成跨v5.2/SPG/UCP/v6新旧recipe的strict surrogate审计；确认同一held panel
  无法选择closed-loop checkpoint或追踪逐task漂移，并把历史结论分成局部机制否定、
  recipe混杂和现有证据不可识别三层。
- [x] 完成architecture×training mechanics审计；量化old/full24的6×LR integral、
  Adam记忆、clip/WD/重线性化差异和v6 matched参数路径，明确它仍是多因素bundle。
- [x] serial-4从clean frozen `3db82df` fresh identity完成1,200 optimizer updates；
  96,000 queries/4,800 videos/200 cycles与raw-full24逐项同曝光，8个checkpoint和
  信息墙合同完整。
- [x] serial step300/600/900/1200 paired correct400完成，为
  `89/100/121/107`；同曝光raw为`82/117/100/110`，差值
  `+7/-17/+21/-3`，best仅+4且漂移未解。
- [x] 补齐raw macro150与SERIAL step900 exact50同曝光内部对照；SERIAL将
  x-only→full BA/action差异从`.0653/.01269`提高到`.4184/.12999`，证明
  update granularity强烈控制视频动态写出，但task能力仍轮换。
- [x] 独立复核、实现并集成AP-ADR；精确参数`10,241,024`，保留mean-backed
  Core、outgoing A/E/D raw Program、独立Core/Program reads和coherent heads，
  删除terminal amplifier、global mixer、谱约束与并行旧路径。
- [x] AP-ADR最长105-frame B20三macro、formal-seed fresh0→1→exact-resume1→3
  全部通过；step1七个payload逐项不变，seal `7dffb6f`已push。
- [x] clean frozen `7dffb6f`的AP-ADR formal fresh macro0→200自然完成：200 cycles、
  96,000 queries、4,800 one-video conditions、wall `3898.217s`，信息墙读取0。
- [x] macro50/100/150/200 paired correct400完成，为`91/81/94/91`，breadth
  `6/6/5/7`；winner macro150仅94且四点持续能力轮换，一小时门失败，不resume、
  不做五臂。
- [x] 修复PI05 action sampler永久切换attention backend造成的内部分析重放污染；
  `5d93af3`后8-task refs1逐层/BA/action严格零误差重放。AP根因定位为contextual
  Program只作高熵K、raw Effect DC主导V：Effect-only距full BA仅`.82%`，反转
  temporal keys仅改变BA `.052%`。
- [x] 封存下一整体CV-ADR设计：保留mean-backed Core与separate dual reads，删除
  Program key/raw-value二轨，让同一causal contextual Program直接作为K/V；无新
  gate/scale/loss，预期参数不变，fresh schema。实现前先完成UCP normalized
  randomized-group4训练因果格。
- [x] endpoint10实现已合入`544c0ef`/`2055a82`：exact ten-step sampler无
  autocast/ACTION输入，sealed512 pairing、finite和historical provenance均
  fail-close；CPU全仓`222 passed`。
- [x] 在生成任何endpoint数值前预注册executed-first5主指标、18-candidate
  global/within-family/per-task关联门及两个matched-recipe方向；secondary不能
  覆盖主门。
- [x] 从三个clean frozen历史extension commits生成并核验portable-v2 LoRA cache：
  v5.2-old 64、v6-fast 512、v6-old 64，信息墙全0；所有tmux自然退出。
- [x] 完成真实CUDA profile/parity与18-candidate四rank formal endpoint诊断：
  9,216 rows，global Spearman `.258398`、permutation `p=.298447`，预注册all gate
  失败；endpoint10永久只作负诊断，不进入训练、loss或checkpoint选择。
- [x] 在独立UCP worktree实现fresh raw-full24 vs cycle-normalized randomized-group4
  受控cell：task/query-keyed stateless policy noise/time、随机Latin group4、LR/beta/
  decay/scheduler exposure composition、fresh checkpoint family与midcycle cursor，
  聚焦测试`31 passed`。
- [x] 完成group4 longseed172真实105-frame B20四卡18-update/3-cycle profile；每cycle
  24 tasks恰好一次、step2起主块梯度可达、峰值reserved `83,647,004,672` bytes。
- [x] 完成group4正式seed 0→1→3→7与raw 0→1→3 exact-resume；cycle0 teacher-video
  assignments逐项一致，step1/3未改写，scheduler/cursor连续，两份fresh配置已seal。
- [x] 将封存`b52cb54` UCP运行面逐blob恢复为唯一canonical；退役AP/endpoint可执行
  路径，聚焦`107 passed`、compileall/JSON/diff check通过。
- [x] task-query raw `configured-decay400/autoscaled200`及50/100/150/200 paired
  correct400完成：`81/72/107/78`；best macro150后lost43/gained14，不续训、不做
  五臂，只封存为scheduler ablation（analysis SHA `bfd580d4...0993`）。
- [x] 提交并push formal scheduler fail-close与corrected total=400/2400；从clean
  `cfc2ad1` fresh完成真正fast400 raw及50/100/150/200 paired correct400：
  `89/71/82/117`。winner macro200只有117、breadth7且仅4 tasks达到5次成功，
  不做五臂；candidate/scheduler-interaction SHA分别为
  `7b7d9822...dd3`/`81eca3cc...ab7e`。
- [x] 审计task/query RNG-v1并定位真实合同偏差：CUDA Gaussian noise按query锁定，
  但LeRobot PI05 Beta flow timestep仍从ambient CPU generator采样。step0 identity的
  四个跨rank重叠task在action rows/video/seed完全一致时loss仍改变，直接证伪跨
  rank/phase stateless合同。
- [x] 正常停止`cfc2ad1` GROUP4 formal于physical step307/51 complete cycles；保留
  metrics与step150/300 checkpoint为invalid-contract provenance，禁止resume/eval。
- [x] 完成CPU+CUDA task/query RNG-v2修复：同时fork/seed/restore两个generator，
  升级config、checkpoint family及三类state schema；`dae13bf`已push，CPU全回归
  `241 passed`。
- [x] 从`dae13bf` frozen authority仅在GPU4--7完成RAW 0→1→3与GROUP4
  0→1→3→7 fresh/exact-resume及跨rank manipulation；四task loss/gradient逐位相等，
  CountSketch最大差`5.82e-11`，两份formal config重新seal。
- [x] 从全新root和fresh identity完成RNG-v2 RAW与cycle-normalized GROUP4，固定
  cycle50/100/150/200 paired correct400为`72/87/86/89`与`77/76/66/100`；
  行为门false，GROUP4不迁移为CV默认。v1 RAW只作observed bundle，v1 GROUP4不进入
  裁决。
- [x] 完成RAW/GROUP4 matched cycle200 exact50：GROUP4将A/D→BA从`.058999`压到
  `.013291`且8/8 tasks一致，唯一fixed-action反向异常来自0-success task；paired
  analysis SHA为`7201364a...11fd`。
- [x] 在RNG-v2 closed-loop outcome前冻结operator裁决层次：cycle200 endpoint与四点
  cycle-AUC、single-best breadth、success-set churn/envelope gap、逐task/phase-cost
  方向及A/D→BA→action传递共同决定；含混时CV首跑保留更简单RAW。
- [x] 完成RAW RNG-v1/v2单变量训练噪声审计：仅CPU Beta timestep identity变化即使
  四点曲线差`-17/+16/+4/-28`、matched梯度草图余弦中位降至`.163--.193`；将其
  定位为optimizer-basin敏感性而非v1优越性或seed-general估计。
- [x] 在独立worktree实现并以merge `b97960f`集成CV-ADR canonical路径；参数
  `10,241,024`、完整CPU回归`226 passed`、结构门无hard violation；旧UCP executable
  path已退役，历史由Git/artifact保存。
- [x] 完成CV-ADR teacher-seed172最长105-frame B20三macro profile与formal-seed
  fresh0→1→exact-resume1→3；真实105帧、B20、五主块可达、step1七文件不改写，
  RAW config已解除profile blocker。
- [x] 从post-fix clean frozen `254ade4`与fresh identity启动CV-ADR RAW macro0→200；
  tmux健康、首macro合同通过，每25保存，未从profile/smoke warm-start。
- [x] CV-ADR RAW macro0→200及paired correct400完成：50/100/150/200为
  `76/111/99/117`，macro200为右端best、breadth6、top2占`57.26%`；与完全相同
  RAW recipe/exposure/panel的UCP相比四点均为正增益`+4/+24/+13/+28`。
- [x] 完成CV-ADR macro200 exact50和34项结构反事实。Core-only/Program-only距
  full BA为`.606/.812`，Effect-only距full`.0674`，证明双路与新contextual value
  都真实工作；但remove-A只在1/8 tasks达门、contextual-memory order在0/8达门，
  same-task BA variance仅`.1049%`，固定action中位仅`.00856%`。
- [x] 完成UCP RAW/GROUP4/SERIAL source-capability配对审计及UCP→CV同RAW架构审计：
  SERIAL cycle150同时改善source retention与新能力但到200回落；CV macro200的
  `+28`由多保留9个source successes和多获得19个新successes共同构成。由此拒绝
  “optimizer gain只会破坏旧能力”和“CV增益只是遗忘更多”两个简单解释。
- [x] 完成v5.2/v6 old×task-complete五臂、source retention、内部传递与matched
  optimizer dynamics联合审计：新recipe在两架构上都压弱Procedure→BA/action与
  顺序margin，却对absolute产生`-12/+22`的相反winner effect；因此后续按
  architecture×training整体根因裁决，不整体否定与recipe混杂的post-v5思想。
- [x] 将联合审计展开到逐task内部transfer与全部历史checkpoint：v6 matched150的
  recipe收益为source retention -1/new gain +17，但selected +22被Object task3
  单项+24主导；保留v6语义/transition bundle证据，明确重构下游reader/compiler。
- [x] CV-ADR RAW已从step200 exact-resume到400：400 cycles、192,000 queries、
  9,600 videos、every25 checkpoints、all finite、0 clip；full400动力学审计降低
  CP负投影与低LR自然止漂移解释。
- [x] 完成250/300/350/400 paired correct400=`77/69/80/82`；八点winner保持
  macro200=`117`，200→250 lost56/gained16，LoRA norm不坍缩。RAW不做五臂。
- [x] 完成macro200/400固定visit397--399的24-task video/query/flow梯度方差分解：
  video主效应约`.1%`且0/24主导，query/flow支配；macro400 task-mean SNR继续下降，
  24/24刚曝光train条件的matched functional loss改善但held loss横盘、闭环崩落。
- [x] 完成CV GROUP4最长105-frame B20 profile与formal-seed fresh0→1→3→7 exact
  resume；B20 finite、cursor/scheduler连续、step1/3未改写，canonical config已seal。
- [x] 从post-seal clean frozen commit与fresh identity完成GROUP4 0→1200正式控制；
  cycle50/100/150/200 paired correct400=`82/77/73/110`，低于RAW四点均值与winner，
  漂移未解且不做五臂。formal root
  `pi05_as_writer_cvadr_group4_taskcomplete_decay400_formal_dev_r4_b20_seed7_51c0ba5_20260802`；
  config/launcher SHA为`a8dd6c83...da79`/`bd7d3210...4082`。
- [x] 完成GROUP4 cycle200 exact50并封存RAW×GROUP4职责对照：A+D/remove-A/remove-D
  门从`8/1/5→0/0/0`，norm上升而视频职责下降；GPU4--7和本任务tmux/process均已
  清空。这是owner本阶段暂停前最后一项GPU工作。
- [ ] Target-Bound Role-Preserving Program已在隔离feature branch完成CPU实现和
  architecture gate，并以`b260a57a94dc21bd3446b212bfa42f71b037ce13` push；按owner
  暂停边界不做最长视频profile、resume或正式训练，下一session现场复核后才决定启动。
- [ ] 后续每版整体架构只有达到同期有效旧架构水平或显示明确续训价值才开第二
  小时和行为五臂。
- [ ] 持续定位task漂移、视频学习和closed-loop off-manifold根因，禁止补丁式
  gate/scale/bypass；150不是自动完成线。

当前UCP受控格实现与实时状态只认
`docs/active_session_handoff.md`。

## 长期完成定义

只有以下核心项全部完成，长期 Goal 才可完成：

1. 使用过滤后的 LIBERO-90 corpus 训练并冻结共享 π0.5-LIBERO source base；
2. 在固定 24 train / 8 validation 上完成并选择 AS-Writer、RL-Writer（若成立）
   与 corrected mixed-task Source-SFT；
3. 完成 source/seen、correct/same-task-other/wrong/shuffled/reversed 机制证据；
4. 合并 validation 后在 32 source tasks 上从规定初态重训已选方法；
5. 完成 final seen comparison 与 8-task zero-interaction test；
6. 在 8 test tasks 上完成 identity/AS/RL Writer 三臂 task-local RL；
7. 用 8 test tasks × 50 action episodes 联合训练一套 shared target-action
   privileged oracle；
8. 原始 rows、逐 task counts、learning curves、seeds、interaction/data counts、
   GPU-hours、参数量、runtime 与关键 hash 齐全，代码验证、commit、push。

ViVLA-style matched reproduction 和 source-only outer learning 是核心闭环后的
可选项，不阻塞 Goal complete。

## 已封存基础

- [x] 固定 LIBERO-Spatial/Object/Goal/Long 40-task benchmark 和 24/8/8 split。
- [x] 完成 LIBERO-90 × target40 的 3,600-pair specification-only audit：
  排除 19 个 exact semantic/composition 重合 tasks，封存 71 active source
  tasks × 50 successful episodes。
- [x] 从 generic `lerobot/pi05_base` fresh 训练 1,000-step shared source base；
  raw step1000 在 target40×8 screen 为 `46/320`，覆盖 13 tasks 与四 suites。
- [x] 冻结 source-only action/state normalization、source policy、tokenizer、
  model/data manifests 和 canonical π0.5 evaluator。
- [x] evaluator 支持 cost-balanced dynamic queue、persistent model/env、
  Writer per-rollout LoRA、无放回 video schedule 与逐 row paired RNG evidence。
- [x] v4/v5/v5.1 失败根因和 v5.2 五臂成功证据已封存；旧可执行路径已退役。

## 当前执行：EMBER Core-Program Writer

- [x] 完成Loom首段和内部负证据：macro50/100/150/200 correct400为
  `79/106/105/112`；correspondence/confidence/Teacher–Policy gap缺少可靠
  锚点，因此停止且不围绕其scale打补丁。
- [x] Recenter fresh macro50/100/150/200 correct400仅
  `55/84/79/85`；所有tasks低于v6 best且Object-3坍塌。内部更新/幅度证据把
  根因定位为time-centering和弱Core造成的semantic-basis starvation，而非
  简单训练不足。
- [x] 从根因重新封存Core-Program设计：v6 Semantic Core提供slot semantic
  basis，uncapped transition+native Action形成full raw causal Procedure，
  width512 bilinear严格要求两分支共同产生content。
- [x] 原位替换canonical Writer/config/schema，退役Recenter可执行配置；
  fresh不兼容，精确参数`10,905,856`。
- [x] 建立Core permutation、uncapped transition、causality、Core/Procedure
  双必要性、constant Procedure DC、zero-preserving slot block与step0 identity
  的确定性模型合同。
- [x] 全仓`194 passed`、compileall与diff check通过；architecture guard仅有
  既有大文件review提示、无hard violation，active source净删643行。
- [x] 集成canonical commit并push。
- [x] GPU4–7最长105-frame B20三macro独立profile；真实覆盖105帧，
  后两步约`25.871 queries/s`、`194.034 macro/hour`，选择B20。
- [x] fresh0→1→exact-resume1→3通过；metrics/LR/task-video-query/RNG cursor
  连续，step1文件不变，全部523个trainable tensor可达；formal config已seal。
- [x] fresh task-complete macro0→200完成：200行finite metrics、4,800 videos、
  96,000 queries、8个every25 checkpoint，未从profile/smoke warm-start。
- [ ] paired、无放回correct400正在GPU4–7并行评测macro50/100/150/200，
  每卡一个single checkpoint。
- [ ] 一小时best若未达v5.2/v6同期`132–133`同档，不做行为级特异性rollout，
  只做Action/transition/Core/Procedure/compiler/LoRA/action反事实和per-task
  gradient conflict分析后重构下一版；达到同档则默认续训第二小时。
- [ ] 第二小时达到`150`，或至少两个相邻checkpoint稳定`145+`且多task共同
  贡献，才对single-checkpoint winner补same/wrong/shuffled/reversed full400。
- [ ] 在上述absolute与视频因果证据完成前不启动one-shot或RL。

## Phase C：v6 AS-Writer development

- [x] 封存
  [`docs/action_forecast_writer_v6_design.md`](docs/action_forecast_writer_v6_design.md)：
  Task-Grounded Semantic Set + Visual-Transition Procedure。
- [x] 在唯一 canonical Writer/runner 内实现 v6，参数 `10,775,296`，step0 public
  LoRA 精确 identity；不保留 v5.3 平行 executable path。
- [x] 实现 task-complete macro：4 ranks × 6 tasks/rank、每 task 一视频一 LoRA
  和 B20 queries、24 tasks 等权、前 5 次 `no_sync`、一次 DDP sync/AdamW。
- [x] 实现 selected-video cost balancing、rank 内 long-first、跨 macro rank
  rotation、macro-boundary checkpoint/resume 和 every-25 retention。
- [x] GPU4–7 最长 105-frame B20 profile 连续 3 macros finite；稳态约
  `25.793 queries/s`、`193.447 macros/hour`，选择 B20，不触发 B16。
- [x] step1→3 resume smoke 恢复 task/video/query/LR/cursor；真实
  visual-transition gradient 可达。
- [x] fresh macro0→200 正式段完成：200 条连续 finite metrics、8 个 every-25
  checkpoint、24-task 等权消费与终点全文件 SHA 均已核验。
- [x] 在 GPU4–7 对 macro50/100/150/200 做并行 fixed correct400；每卡一个
  checkpoint，6 Writer generators + 6 persistent workers，视频 50 条无放回，
  全局 long-first。结果为 `114/77/120/129`，paired 输入合同全部通过。
- [x] 选择 macro200 为 absolute observed-best；其 129/400 仅覆盖 5/8 tasks，
  与覆盖 7/8 tasks 的 macro150（120/400）差异不显著，保留 breadth 风险。
- [x] 对 macro200 做 correct/same-task-other/wrong/shuffled/reversed
  full400：`129/131/108/111/105`；same同档，correct对后三臂 paired
  `p=.011/.0198/.00094`，方向门通过但 margin 弱于 v5.2。
- [x] 完成 macro200 的 16-reference 内部传递分析：顺序差异由新增
  visual-transition 路径进入 Procedure，并在 fixed-Core 反事实下传到
  effective LoRA/action；无 Semantic Core 顺序旁路。相对 v5.2，Procedure
  差异更强但下游 LoRA/action 差异更弱，需由续训判断是成熟度还是新瓶颈。
- [x] exact-resume macro200→400 与 macro250/300/350/400 correct400 已封存；
  后四点为 `117/118/125/125`，均未超过 macro200=`129`。第二小时提升部分
  breadth但aggregate不涨，能力继续在tasks间迁移；不继续同一full-24 recipe。
- [x] v6拓扑与机制证据封存：Semantic Set、Visual Transition、Causal
  Procedure职责成立，macro200五臂通过方向门；但absolute、margin和跨task
  稳定性未达最终满意门，后续训练粒度/下游compiler仍需改进。
- [x] 实现显式provenance、inference-only的derived Writer checkpoint：
  导出单套平均权重并保持一次Writer前向；原始checkpoint全部保留，training
  resume/warm-start对derived路径fail closed。真实四候选逐tensor独立重算
  完全一致，formal evaluation authority检查通过。
- [x] 按outcome前封存的四候选
  `{150,200}`、`{200,400}`、`{150,200,350,400}`、
  `{150,200,250,300,350,400}`在GPU4–7各跑paired correct400；结果为
  `129/140/144/145`，最后一组相对raw macro200净增16、
  `37 gained/21 lost,p=.04794`，覆盖从5/8增至7/8 tasks。
- [x] 对六点平均winner完成full400五臂与16-reference内部传递：
  `correct/same/wrong/shuffled/reversed=145/134/128/119/122`；
  correct对后三臂均显著且各由至少5个tasks正向贡献，fixed-Core
  Procedure-only保留到effective LoRA/action，Core-only近零。same差11且
  `p=.152`，只比预封存的保守差值阈值多1；absolute仍比150硬门少5。
- [x] fresh运行唯一的v6 fast-decay400稳定化对照：只把cosine
  `decay_steps 2000→400`，其余架构、task-complete B20、AdamW、数据与seed
  全部保持；先0→200并评测50/100/150/200，除可信absolute下降外默认
  exact-resume至400并评测250/300/350/400。八点结果为
  `106/64/111/133/132/117/138/143`；macro400比corrected SFT高34但仍比
  absolute150少7，末段参数位移已很小且350→400净增不显著，不机械续第三段。
- [x] 按outcome前sealed合同筛选四个fast-decay checkpoint-average：
  `{350,400}`、`{200,350,400}`、`{200,250,350,400}`和
  `{150,200,250,300,350,400}`；GPU4–7各负责一组，跑paired correct400。
  结果为`139/135/129/130`，均未超过raw macro400=`143`；只有局部两点平均
  恰好达到SFT+30，四者均未达absolute150。所有源checkpoint、派生checkpoint、
  评测cache/rows/results均保留；完整paired与long-first审计通过。
- [x] 按owner后续决定，把fast-decay从macro400 exact-resume到600并评测
  450/500/550/600；结果`131/130/132/126`均低于macro400=`143`，
  400→600为`31 lost/14 gained,p=.01609`，形成可信post-best下降。
- [x] 对fast-decay单checkpoint best macro400完成正式五臂与内部传递：
  `143/135/125/128/129`；wrong显著，shuffled/reversed方向正确但不显著。
  顺序信号存在于Procedure并能传到LoRA/action，只是task-complete下游增益弱。
- [x] 在不改v6拓扑的前提下实现并封存旧rank-rotating训练范式；最长视频只做
  fixed-B20 profile，`B21`从未运行且正式入口拒绝更大batch。
- [x] v6旧范式fresh训练900 updates并评测step100/500/700/900：
  `98/121/76/95`，step500为single-checkpoint observed-best，后续有显著下降。
- [x] 对旧范式step500完成五臂与16-reference内部分析：
  `121/122/111/84/47`；顺序门强通过、wrong语义门失败，Procedure-only几乎
  完整复现shuffled/reversed的LoRA/action差异。
- [x] 按owner要求在上述证据完成后停下讨论；owner随后已批准v7第一性原理
  设计与自主迭代，因此该临时停止边界结束。

## Phase D：corrected mixed-task Source-SFT

- [x] 从 frozen source base 实现唯一canonical corrected rank-128 Source-SFT；
  每个physical batch一次普通同步forward/backward/clip/AdamW，无gradient
  accumulation或Writer式micro-round。
- [x] 每rank physical batch包含全部24 tasks等量样本；按
  task→episode→chunk分层无放回周期采样，跨rank row不重复、exact resume，
  task-balanced普通batch mean。
- [x] GPU4–7 B144真实fresh step1→resume step3通过；每步全球576 queries、
  峰值allocated/reserved `60.69/74.07GB`，稳态`34.52–36.35 queries/s`。
  B144稳定，未触发B120 fallback。
- [x] 用sealed config从identity fresh训练step0→225（约一小时训练body，
  冷加载另计），每25步checkpoint；225条metrics连续finite，9个checkpoint
  和完整resume state均已核验。
- [x] GPU4/5/6/7各加载step50/100/175/225之一，并行完成四个fixed
  validation correct400；结果为`60/75/77/56`，每点400 rows、36 shards、
  6 workers、零错误，paired seeds和noise prefix完全一致。
- [x] 从step225 exact-resume到450并完成12点dense correct400；
  step400/425为`109/107`同档，step450降到`74`且paired显著，封存full-24
  observed-best step400=`109/400`，不再续训该recipe。
- [x] 实现global-8 cyclic mixed替代sampler：4 ranks×2 tasks、每update
  8个disjoint tasks、连续3 updates完整覆盖24 tasks；保持B144/global576、
  rank-128 LoRA、LR/scheduler与平均task/sample clock不变。
- [x] GPU4–7完成global-8 B144 fresh0→3→resume6 profile；两轮完整cycle、
  3,456 query identities唯一，稳态`36.27–36.38 queries/s`，峰值
  allocated/reserved `60.69/74.07GB`，无OOM或nonfinite。
- [x] global-8从identity fresh训练step0→240并exact-resume到480；16个每30步
  checkpoint全部保留，累计276,480 queries、每task 11,520 samples。
- [x] global-8八点paired correct400为
  `63/83/85/98/90/62/90/105`；step480=`105`为该recipe observed-best，
  但相对step420仅`+15,p=.0627`，相对full-24 step400=`109`为
  `28 gained/32 lost,p=.699`。它没有解决task漂移或提高SFT上限，故不续到
  600；最终corrected Source-SFT development best仍为full-24 step400
  `109/400`。
- [x] 与 v6 使用同一 frozen source base、normalization、policy interface 和
  validation rows；不机械匹配 optimizer steps。

## Phase C2：v7第一性原理Writer

- [x] 封存
  [`docs/action_forecast_writer_v7_design.md`](docs/action_forecast_writer_v7_design.md)：
  明确Core、Action–Effect Procedure与Procedure-content-only compiler的需求、
  已有prefix/suffix信号、最少结构、参数预算和可证伪判据。
- [x] 原位替换唯一canonical Writer：删除Text-only分支与Core-primary AdaLN；
  一次Action Expert forward使用8个原生稀疏suffix anchors；不保留v6/v7
  parallel executable path或checkpoint兼容分支。
- [x] 完成task-span、shape/mask、Core permutation invariance、forward
  transition、D=0 binder、causality、Core-only identity、freeze/gradient、
  public-LoRA schema和checkpoint-resume最短验证；全仓192 tests与真实
  step1→3 exact-resume均通过。
- [x] 只在物理GPU4–7完成最长真实视频profile：B32/B24 OOM，B20三步finite，
  含105-frame视频；稳态约27.48 queries/s、206.08 macros/hour。
- [x] task-complete B20、fast-decay400从identity fresh完成macro0→200并
  exact-resume到400；每25 checkpoint，metrics连续且finite。
- 正式首段launch contract：

  ```text
  workspace  /data/ymdai/projects/EMBER
  branch     main
  commit     ca7db57d0c2d1ec2e7032a44b58238b6de35b1f4
  devices    physical GPU4,5,6,7; 4-rank DDP; NUMA node1
  input      frozen source-base raw step1000 + sealed 24 train tasks
  scale      200 macros = 96,000 queries = 4,800 one-video conditions
  output     /data/ymdai/outputs/ember/
             pi05_as_writer_v7_jointae_taskcomplete_decay400_dev_r4_b20_seed7_s2400_ca7db57_20260729
  retained   8 every-25 checkpoints; projected peak additional storage <1.3GB
  selection  paired fixed correct400; full five-arm only for current best
  resume     only exact same-contract complete macro checkpoint
  ```

  Exact command:

  ```bash
  numactl --cpunodebind=1 --membind=1 env \
    PYTHONPATH=/data/ymdai/projects/EMBER/src \
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,5,6,7 \
    OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
    /data/ymdai/projects/EMBER/.venv/bin/torchrun \
    --standalone --nproc-per-node=4 scripts/train_as_writer.py \
    --config configs/pi05_as_writer_language_axial_v7.json --mode formal \
    --source-run /data/ymdai/outputs/ember/pi05_source_base_v1_seed7_1k_e2cc238_20260722 \
    --checkpoint /data/ymdai/outputs/ember/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000 \
    --tokenizer-path /data/ymdai/ember_data/openpi/paligemma_tokenizer.model \
    --data-root /data/ymdai/ember_data/LIBERO-datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a \
    --output-dir /data/ymdai/outputs/ember/pi05_as_writer_v7_jointae_taskcomplete_decay400_dev_r4_b20_seed7_s2400_ca7db57_20260729 \
    --stop-after-step 200 --num-workers 2 --log-every 10 --skip-data-sha
  ```
- [x] paired fixed correct400八点为
  `82/106/114/120/101/114/115/106`；macro200五臂
  `120/112/91/100/69`。内部检查定位joint `8×L` attention近均匀以及
  Core→LoRA影响近零，v7停止。
- [x] 未达门后按“表示→传递→多task优化→闭环目标错位”的最早瓶颈做
  fresh单变量迭代；不使用checkpoint融合、ensemble、contrast/order loss或
  信息墙捷径。
- [ ] 最低成功门仍为single-checkpoint correct400≥150、same≈correct、
  correct显著优于wrong/shuffled/reversed、多task共同贡献，并在独立
  RNG/video permutation下复测成立。150不是自动停止点。

## Phase C3：v8 Hierarchical Action–Effect + Core-Gated Procedure

- [x] 封存
  [`docs/action_forecast_writer_v8_design.md`](docs/action_forecast_writer_v8_design.md)：
  v7 joint attention和Core弱影响的定量根因、hierarchical 8→1 binder、
  bounded multiplicative Core gate、参数预算及可证伪判据。
- [x] 原位切换唯一canonical源码/config到不兼容v8 schema；没有v7并行
  executable或checkpoint兼容分支。
- [x] 聚焦CPU合同通过：Writer总参数`10,706,176`、binder`590,848`、
  compiler`1,469,696`；`D=0→event=0`、`Procedure=0→LoRA identity`、
  Action/effect梯度和Core有效调制均成立。
- [x] 全仓`192 passed`、Markdown link audit零缺失、`git diff --check`通过；
  architecture guard无hard violation、无parallel version/function family。
- [ ] clean commit/push。
- [x] live核验GPU4–7和存储后，B20完成最长105-frame真实视频连续3个
  task-complete macros；全finite且稳态约205.97 macros/hour，不触发B16。
- [x] B20完成fresh0→1、exact-resume1→3；step1未改写，任务/视频/query/LR/
  cursor相同，binder/EventRead/Core gate和所有主模块梯度可达。
- [x] clean commit/push后保持task-complete、fast decay400，从identity fresh
  完成macro0→400；八点correct400的observed-best为macro300=`125/400`。
- [x] macro300五臂为`125/121/110/110/117`；内部检查显示Action变化仅贡献
  约`8–10%` event差异，Effect变化贡献约`147–300%`，EventRead近均匀。
  strict local binding缺少teacher-action身份，v8停止。

## Phase C4：v10 Evidence-Preserving Dual-Stream Writer

- [x] 封存
  [`docs/action_forecast_writer_v10_design.md`](docs/action_forecast_writer_v10_design.md)：
  恢复text-only task axis；保留独立Action hypothesis与Visual-Effect streams；
  interleaved causal Procedure；Procedure提供content并门控full-rank Core。
- [x] 原位替换唯一canonical源码/config到不兼容v10 schema；删除尚未封存的
  v9草案和v8 executable config，不保留strict binding/EventRead并行路径。
- [x] 真实参数枚举`11,627,520`；全仓192 tests通过，覆盖Core置换不变、
  dual-stream shape/mask/order、`D=0→Effect=0`、Action保留、
  `Procedure=0→LoRA identity`、完整rank-16 target与freeze/gradient staging。
- [x] GPU4–7最长105-frame B20连续3个macro finite；后两步约
  `26.38 queries/s`、`197.85 macros/hour`，峰值allocated/reserved约
  `77.01/83.65GB`，B16未触发。
- [x] 完成fresh0→1→exact-resume1→3；step1未改写，cursor/采样/LR完全一致，
  与独立连续run最大mean-loss差`2.63e-6`；正式teacher seed与profile已封存。
- [x] clean commit/push后从identity fresh按task-complete fast-decay400训练
  至macro400，共`9,600`个one-video LoRA conditions、`192,000`个action
  queries，约`7,832.8s`；每25保存。
- [x] 对12个single checkpoints完成paired fixed correct400；曲线
  `95/103/84/89/82/90/96/96/89/96/97/91`，observed-best为macro50
  `103/400`，不做checkpoint融合。
- [x] 对macro50完成五臂`103/94/75/67/43`和内部
  Text/Core→Action/Effect→Procedure→slots→effective-LoRA→policy检查。
  same同档且wrong/shuffled/reversed行为门均通过，但absolute低于Source-SFT
  `109`并距硬门150为47；Action主导、Effect近均匀读取和高增益compiler使
  same-task视频方差被放大。v10判为absolute负结果。
- [x] 按owner要求完成v10后暂停：不续训、不改canonical架构、不启动Loom、
  one-shot或RL，等待共同讨论。

## Phase C5：Core-Program负结果与Prior–Innovation重构

- [x] Core-Program fresh macro0→200训练合同完整成立；fixed correct400
  `84/75/60/76`，四点逐task envelope仅`95`，不续训、不做行为级特异性。
- [x] 对macro50完成无rollout内部数值分析：Procedure已有强顺序差异，但到
  effective LoRA/action压缩两个数量级；raw DC主导readout、AC被压弱，
  bilinear形成moving basis，Procedure/Core梯度比约`.36`。
- [x] 从根因撤销strict double-necessity，封存
  [`docs/action_forecast_writer_prior_innovation_design.md`](docs/action_forecast_writer_prior_innovation_design.md)：
  Core提供稳定semantic prior，Core-only query读取time-centered Procedure
  innovation，二者在固定slot坐标直接相加。
- [x] 原位替换唯一canonical源码/config/schema；退役Core-Program活动config，
  不保留兼容resume或并行compiler。
- [x] 精确参数枚举Writer`10,643,968`、compiler`1,403,904`；全仓
  `195 passed`、compileall、diff check通过，architecture guard无hard
  violation。
- [x] canonical实现clean commit `7b7abf1`并push。
- [x] 只在GPU4–7完成最长105-frame B20三macro profile、全参数
  reachability与formal-seed exact-resume；不继承旧证据。
- [x] seal后fresh macro0→200、every25；固定评测50/100/150/200
  correct400为`100/61/89/88`，不融合checkpoint。
- [x] 未恢复同期旧架构，未续第二小时、未做行为级视频特异性；跨架构内部分析
  将最稳定瓶颈定位为B列、rank和跨层effective update塌缩。

## Phase C6：Target-Spectral Writer

- [x] 封存
  [`docs/action_forecast_writer_target_spectral_design.md`](docs/action_forecast_writer_target_spectral_design.md)：
  保留v6 Core/Procedure上游；把320个rank伪语义slots改为38个真实policy
  targets，target-first融合、rank-last展开，并固定A/U spectral gauge。
- [x] 唯一canonical源码/config/schema原位切换；Prior executable config退役，
  不兼容旧Writer checkpoint。
- [x] 精确参数`14,495,744`；step0 effective identity、38-target拓扑、
  target/rank坐标、FP32 Procedure centering、强共同方向QR稳定性和真实三步
  gradient staging均有CPU合同。
- [x] 全仓`196 passed`、compileall、JSON、diff和architecture guard复核后
  clean commit/push。
- [x] 只在GPU4–7完成最长105-frame B20三macro profile；三步finite，稳态约
  `25.488 queries/s`、`191.159 macro/hour`，峰值约`77.07/83.65GB`，
  B16未触发。
- [x] 正式teacher seed下fresh0→1→exact-resume1→3；steps/LR/query/video
  cursor连续、全部finite、validation/test reads为0，step1七个文件逐项SHA
  未改写。
- [x] fresh macro0→200、every25自然完成；固定评测50/100/150/200
  correct400为`30/12/18/34`。四点完整审计通过，best低于source base、SFT、
  v5.2和v6；未续训、未做行为级控制。
- [x] macro200完成无rollout rank/layer/video与五条件内部分析。强制spectral
  gauge把stable rank从约1提高到3.32，却把LoRA范数缩小3.66倍、打散跨层方向、
  翻转q/v能量并造成极端layer不均；Core/Procedure和order传递保持工作。
- [ ] 基于负结果重新设计：保留v6高增益、q-dominant、跨层协调公共主方向，
  把额外rank作为可选zero-init视频innovation；不得在Target-Spectral的
  orthogonal scale/gate上打补丁或resume。

## Phase C7：v5.2 × task-complete fast-decay因果格

- [x] owner授权补齐此前缺失的`v5.2 + 新训练`单元；不得把v6的143直接归因
  于模型拓扑。
- [x] 在成熟long-first task-complete训练框架中原位恢复正式结果对应的v5.2
  拓扑；参数`10,237,704`、step0 identity、信息墙和public rank16不变。
- [x] fresh config固定B20、4 ranks×6 tasks、full24等权、LR
  `3e-4`、warmup17、cosine decay400到`1e-5`、every25。
- [x] GPU4–7完成最长105-frame三macro profile和formal-seed
  fresh0→1→resume1→3；B20 finite、所有主模块可达，配置已seal。
- [ ] clean push后fresh macro0→200，再默认exact-resume到400。
- [ ] 并行评测macro150/200/350/400 correct400；winner若在内部点，只补±25。
- [ ] 对winner与旧v5.2/direct Source-SFT做有效BA谱、范数、q/v、layer/target、
  视频中心化变化和policy action对照；达到absolute门后才补行为控制臂。

## Phase E：matched π0.5 action one-shot baseline

- [ ] 在看 outcome 前，每个 validation task 用固定 seed 从 50 episodes 中抽
  1 条 action episode。
- [ ] 对每 task 只训练一次 one-shot LoRA，不做 50 次 one-shot。
- [ ] EMBER 比较臂使用与该 episode 对应的 action-hidden video；保持 task、
  state、env/policy RNG 和评测预算 paired。
- [ ] 比较 absolute performance、训练/适配 wall、GPU-hours、action supervision、
  trainable parameters 和 deployment-time forward 成本。
- [ ] EMBER 只看 video 且一次 Writer forward，因此不把“必须绝对超过 action
  one-shot”设成唯一成立条件；若能超过则作为更强结果。

## Phase F：RL-Writer development

- [ ] 从 v6 架构规定初态做独立、短且 task-balanced AS cold start；不得从完整
  AS observed-best 继续。
- [ ] 直到 24 个 development-train tasks 各在 official random-reset rollout
  中至少成功一次，才关闭 action 入口并进入 pure-reward。
- [ ] reward 阶段只用官方 binary reward/success；不读 object pose、
  privileged shaping、validation/test reward 或 `.pruned_init`。
- [ ] 保存 Writer/optimizer/scheduler、worker RNG、env/policy seed schedule、
  interaction cursor、video schedule、完整 reward ledger 与 exact-resume state。
- [ ] 用 correct/wrong-video、source/seen 和 absolute validation 选择；
  RL 不能用来掩盖 AS 的绝对性能或逻辑漏洞。

## Phase G：32-source final 与 zero-interaction test

当前 focused v6/SFT/one-shot/RL 完成并向 owner 汇报后才进入；不得自动启动。

- [ ] 将 8 validation tasks 机械合入形成 32 source / 8 test。
- [ ] AS-Writer、Source-SFT、RL-Writer（若成立）各自从规定初态单 seed 重训。
- [ ] 打开 test 前完成 final seen comparison。
- [ ] zero-interaction test 比较 source base、Source-SFT、AS-Writer、RL-Writer
  及 correct/wrong-video；每 rollout 随机抽正确 task 的一条 teacher video。

## Phase H：test-only task-local RL 与 oracle

- [ ] test 打开后，identity/AS/RL Writer 三臂在每个 test task 上使用相同
  official random-reset sequence、同一 cohort video 和可比预算训练到各自最佳。
- [ ] fixed 50 `.pruned_init` states 只作训练分离的 fresh evaluation。
- [ ] 三臂结果封存后，才读取 8 test tasks × 50 action episodes，联合训练一套
  shared multi-task target-action LoRA oracle；不是 8 套 task-local LoRA。

## 每次 GPU 运行前

- [ ] 只读核验 workspace/branch/HEAD/origin/status、现有进程和输出根。
- [ ] 实时比较`gpu01`与`gpu02`，只用空闲卡且合计最多6张；记录owner/进程/显存/
  利用率，不reset、kill、pause或干扰他人。
- [ ] 检查目标`/data1`个人quota、项目占用、峰值新增量和共享filesystem余量。
- [ ] 封存 exact command、config/model/data paths、output root、process topology、
  checkpoint cadence、停止与继续判据。
- [ ] 正式昂贵 run 前做 live GPU preflight；不杀、暂停、reset 或干扰他人进程。
- [ ] output 不覆盖；resume 核验完整 state。stage stop 只可在 sealed total axis
  内单调延长，其它 scientific contract 变化必须 fresh。
- [ ] 评测按 `episodes × horizon` 动态调度；所有 worker 先处理 long task，
  long 耗尽后再取其它 task；任何 checkpoint/GPU 分配都遵守。
- [x] 等训练/rollout 时推进不污染运行的代码、分析和仓库清理；已退役旧路径
  18,853 行、约 3.8 MiB 仓库缓存和 87.49 GB 已完成评测 LoRA 中间 cache；
  活动运行环境、checkpoint、rollout rows/results 和 contract 证据完整保留。

## 当前继续/停止判据

- absolute 低于可信满意区间或尚未形成充分峰后下降时，继续训练、诊断或 fresh
  架构实验；不能因单点略涨结束。
- correct 提升若依赖 wrong video、shuffle/reverse、validation 泄漏或其它违反
  EMBER 映射的捷径，一律判为机制失败。
- focused AS-Writer的absolute硬门统一为
  `correct400 >= max(150, corrected Source-SFT observed-best + 30)`；两个条件
  必须同时满足。还必须same≈correct、correct显著优于wrong/shuffled/reversed、
  多tasks共同贡献并通过独立RNG/video permutation复测。
- `122/400`旧八卡Source-SFT只是背景，不是独立硬门；`+30`是最低研究里程碑，
  不是达到后强制停止。新corrected Source-SFT必须重新训练和选峰后再比较。

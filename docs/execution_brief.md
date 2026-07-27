# EMBER Current Execution Brief

状态：2026-07-27。共享 π0.5-LIBERO source base 与 Source-SFT comparator 已
封存。Action-Forecast Writer v4 已训练至step2400并停止；observed-best
step825在固定400 panel上为`correct=109`、`same=104`、`wrong=99`、
`shuffled=148`、`reversed=126`，行为特异性失败。

外部专家复核后的第一轮诊断证明，shuffle的直接行为放大器位于per-image
forecast之后的absolute-time Plan/Revision；但后续全面复审又确认它不是唯一
根因。24个train tasks的隐藏语义演化显示，AS loss持续下降时，latest forecast
更准的比例从step75约`0.509`降到step825约`0.404`，residual与真实误差修正的
cosine从`0.335`降到`0.238`。step825 neutral visual-state几乎不改变forecast，
而raw-image/Meta路径越来越贴近同demo低层translation。Object-1/Object-3上，
只重排forecast前三维translation即得到`79/100`，与true shuffled的`82/100`
统计上无差异，correct仅`49/100`。

完整根因是：同task独立video/action的positive AS目标不可识别demo高层过程语义；
32-token visual-state没有成为必要瓶颈；Meta-LoRA学习了低层phase/translation
action-shaped latent；absolute-time Plan/Revision再将其放大。

owner随后批准Semantic Core + Causal Procedure v5作为唯一活动架构：

- language-conditioned image-position hidden形成对帧顺序严格不变的Core；
- native fixed suffix与两个Meta-LoRA保留，但Action Expert只输出每帧
  robot-semantic hidden，不再产生7D forecast；
- 两层global causal Transformer形成可变长Procedure；
- Core先编译稳定LoRA content，Procedure以zero-init refiner作有向修正；
- 每rank每step一个task、1条teacher video、1套one-shot LoRA；该rank完整
  action batch的所有独立queries都监督这套LoRA。下一task visit轮换video，
  推理仍严格one-shot。

完整活动合同见
[`docs/action_forecast_writer_v5_design.md`](action_forecast_writer_v5_design.md)；
完整v4根因证据见
[`docs/action_forecast_writer_v4_root_cause.md`](action_forecast_writer_v4_root_cause.md)；
原咨询材料见
[`docs/action_forecast_writer_expert_consultation.md`](action_forecast_writer_expert_consultation.md)。

## 1. 研究问题

机器人 action trajectories 对目标任务稀缺时，一条 action-hidden teaching video 是否能让共享 Writer 为一个已有通用控制能力的 VLA 生成更好的完整 task-specific LoRA？直接 target-action SFT 是“教练拉手”的 privileged oracle；Writer zero-interaction 是“看过教程后的第一次尝试”；test-task reward adaptation 是后续自行练习，不是 EMBER 必须存在的尾巴。

## 2. 目标 benchmark 与 split

- 目标任务来自 `libero_spatial`、`libero_object`、`libero_goal`、`libero_10` 四 suites，共 40 tasks。
- development split 为每 suite 6 train / 2 validation / 2 test，总计 24/8/8，封存在 `configs/libero_24_8_8_v1/`。
- split 只使用 task identity、language 和 BDDL/specification；不得按新 outcome 改 IDs。
- development 完成后将 8 validation tasks 合入 source，最终为 32 source / 8 test；第一轮完整流程只跑一个 training seed。

## 3. 共享 π0.5-LIBERO source base

所有主方法的共同起点统一定义为：

```text
generic lerobot/pi05_base
→ specification-only 过滤 LIBERO-90 与目标40 exact semantic/composition overlap
→ 在剩余 source tasks、每 task 全部50条成功action episodes上联合SFT
→ 若使用source LoRA则merge进policy
→ 冻结共享、多任务、语言条件的 π0.5-LIBERO source base
```

完整3600-pair specification audit已经在看新policy outcome前封存：排除19个
exact semantic/composition重合tasks，保留71个source tasks。task44对目标
`libero_goal` task7、task77对目标`libero_10` task5是其中两条已知同语言
重合，不是尚待完成的audit。禁止使用已经在目标40 actions上训练的
`pi05_libero`。

source action/state normalization 只由过滤后的 LIBERO-90 source corpus计算，并随 base 冻结供所有下游方法共用。source-base LoRA recipe、targets与训练参数必须先参考官方或成熟 π0.5 fine-tuning 项目，不能猜。

base 不追求高 ceiling。fresh step1000 raw source policy已经冻结；全部目标40
tasks×8 states screen为`46/320`，覆盖13 tasks和四个suites，满足跨task基本
interface competence。generic π0.5 的正式结果`0/400`仅为原始校准，不能替代
新source base结果。

owner于2026-07-22将source-base正式训练改为从generic base fresh运行1,000 optimizer steps，333-step线性warmup后到官方peak LR；旧30k attempt在step2880停止且无checkpoint，不得resume。这里的目标只是轻量获得LIBERO embodiment/control interface，而不是在LIBERO-90上收敛或过拟合。

## 4. Development methods

### AS-Writer

- 输入恰好一条 action-hidden teacher video + 正确 task language；输出完整 task-specific LoRA。
- 在24 train tasks上做均衡混合。每rank每step一个task，抽1条teacher video并
  只生成1套LoRA；`B_a`条独立action queries全部在该LoRA下各计算一次，
  `B_a`个functional losses求均值。video与action episode/chunk不要求配对，
  action只进functional behavior loss。
- source base冻结，只有Writer更新。Writer不得看到action、proprio、reward、terminal、task ID、filename或隐藏stats。
- v5原位替换v4的visual-state/forecast/Plan/Revision/Belief路径，不保留兼容
  分支。初版机械预算`10,301,440`，与rank-128 Source-SFT只差`4,096`
  parameters；公开rank-16 LoRA仍为76 tensors、`1,287,168` scalars。
- 单视频合同在GPU4–7联合profile后选择frame batch32、`B_a=20`：105帧
  最长真实视频步`6.96s`，常规步`3.11–3.53s`，峰值reserved
  `83,630,227,456 bytes`；B24 OOM，frame40没有净吞吐收益。owner允许最长
  视频只保留少量显存余量。正式约一小时一个900-step exact-resume segment，
  每100步保存checkpoint；不继承v4的
  600/1200-step等价口径。
- 第一段先用functional panel安排顺序，再以fixed-400 correct-video寻找
  absolute observed-best；不使用80-episode快筛。若全部候选仍明显低于约
  `110–120/400`，先定位训练/架构问题，不提前花费五臂rollout。
- 达到absolute预门后，对暂时best先做内部Core/Procedure/LoRA gate，再跑固定
  400 correct/same/wrong/shuffled/reversed；要求same影响最小且correct明显
  优于wrong、shuffle、reverse。不通过则定位最早失效层后fresh迭代，不用
  contrast/order loss。
- 最终correct至少达到或接近`125/400`，目标逼近v4 shuffled `148/400`。
  没有明显、持续、多task且独立复测成立的峰后下降就继续下一段；focused v5
  AS不设总wall-clock上限。

### RL-Writer

- 与完整AS-Writer best分开，从新架构规定初态做短、task-balanced AS cold
  start；持续用官方random-reset reward screen，直到24个train tasks每个至少
  一次真实success，再关闭action数据入口并跨source tasks做纯reward训练。
- rollout初态来自官方随机reset/BDDL机制，不来自fixed `.pruned_init`。
- cold-start必须报告teacher-action queries、每task first-success step和wall；
  不能偷换成完整AS-Writer continuation。
- 使用官方reward/success，不额外读取object pose构造privileged shaping。

### Source-SFT

- 从同一frozen source base开始，在24 train tasks上联合训练一套shared multi-task LoRA。
- test不看held video/action；它控制“只加强source policy”与“额外读取held video”的差异。
- 它与AS-Writer各自在validation上选最佳，不要求相同步数或数据量；必须报告各自steps、action chunks、参数量、GPU-hours和搜索上限。

所有方法共享同一frozen source base、normalization和policy接口，但不机械要求
相同LoRA rank。Writer生成sealed rank-16 task LoRA；capacity-matched
Source-SFT可使用rank128，并以其`10,297,344`个trainable参数约束Writer本体
参数预算。各自targets/rank/alpha/dropout、identity initialization和参数量均
需显式报告；不能把LoRA两因子同时全零造成无梯度。

## 5. Seen-task 与 wrong-video 机制证据

- 在看outcome前，按specification从source tasks预声明覆盖四suites的seen panel。
- 比较frozen source base、Source-SFT、AS-Writer，以及可用的RL-Writer。
- Writer同时跑correct-video与cross-suite wrong-video。wrong-video实验保持正确language、evaluation task、init state和policy RNG，只替换为另一suite的视频。
- 主要报告 `correct-video - source base`、`wrong-video - source base` 和 `correct-video - wrong-video`，防止把通用或language-only adapter误写成视频利用。
- held zero-interaction每个rollout随机抽一条正确task teacher video，不挑最好视频。

## 6. Final retraining and test

development在24/8上选定AS-Writer、RL-Writer（若成立）和Source-SFT配置后，把validation合入形成32 source。三个方法从规定初态各自重训一次；先做final seen comparison，再打开test。

zero-interaction test比较：

- 新frozen source base；
- Source-SFT shared LoRA；
- AS-Writer one-video task LoRA；
- RL-Writer one-video task LoRA（若成立）；
- correct-video与cross-suite wrong-video Writer controls。

旧generic base已打开过并为0/400，owner明确不要求继续争论test purity；但新source base是不同模型，必须另行测量。

## 7. Test-only three-arm task-local RL

task-local RL不在validation上预先冻结算法，也不在test打开前运行。test阶段开始后，每个test task本身就是adaptation training domain，可直接根据官方随机-reset reward rollouts调优并训练到曲线接近最佳。

三臂：

1. frozen source base + functionally identity LoRA；
2. frozen source base + AS-Writer LoRA；
3. frozen source base + RL-Writer LoRA。

每个task/adaptation seed随机选一条该task视频；两个Writer臂用同一条，并固定由它生成的初始化LoRA。三臂匹配task、env/policy seed schedule、随机初态序列、RL实现与可比资源上限，保存完整interaction cursor和exact-resume状态。adaptation、调参和checkpoint选择使用该test task随机reset rollouts；固定50 states只做训练分离的fresh evaluation。

## 8. Joint target-action oracle

无action主结果和三臂RL封存后，从同一source base出发，使用8个test tasks × 每task全部50条teacher action episodes，联合训练一套shared multi-task LoRA。它不是8套task-local LoRA。第一轮只做完整50/task，不做action-budget curve；它是privileged oracle，不能反向修改其他方法。

## 9. Evaluation efficiency

- 保持官方π0.5/LIBERO preprocessing：render256/model224、两相机旋转180°、7维state/action、10 flow steps、执行5步后replan、dummy settling10、成功终止、horizon 220/280/300/520。
- 旧“一task/一GPU”使两个horizon-520进程严重拖尾。下一实现先调研成熟项目，再按 `episodes × horizon` cost-balanced切state shards，使用动态队列、持久model/env和work stealing。
- Writer每rollout LoRA不同，profile batched functional LoRA与每卡统一1/2/3个policy replicas。所有卡CUDA process数相同，GPU0不得额外放controller/server/model。
- batch8到16只有约0.9% per-episode提升；不靠盲目加batch伪装效率。
- 训练最多8张A100，一卡一DDP rank为默认。当前focused v5只使用物理GPU4–7；
  0–3不进入visible set。当前AS/RL按约一小时segment持续exact-resume且不设总
  wall-clock上限；历史task-local RL预算合同不覆盖本focused阶段。

## 10. Optional work and hard boundaries

- 核心闭环完成后有时间再做同base/split/one-video信息墙的ViVLA-style matched reproduction。
- source-only reward/meta outer learning只作更晚的可选增强，不阻塞Goal complete。
- 不使用bank、geometry、shared update subspace、residual escape、额外shared adapter、旧SmolVLA checkpoint/runner或MemLLM。
- meaningful状态更新 `task_plan.md`、`findings.md`、`progress.md`，验证、commit、push；等待长任务时继续推进不污染运行的后续工作。

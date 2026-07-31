# EMBER Current Execution Brief

状态：2026-07-31。共享 π0.5-LIBERO source base与corrected mixed-task
rank-128 Source-SFT均已封存，后者development observed-best为`109/400`。
当前可信架构标杆是v5.2与v6：v5.2 step900 single-checkpoint correct400为
`132`，五臂`132/138/74/82/83`；v6 task-complete single-checkpoint best为
`143`，五臂`143/135/125/128/129`。前者视频语义和顺序margin强，后者
absolute更高但margin较弱；两者都未达到focused absolute门150。

v7/v8/v10/Loom/Recenter/Core-Program/Prior–Innovation均已完成并作为负结果
provenance。其关键correct400 best依次为`120/125/103/112/85/84/100`；
Prior四点为`100/61/89/88`。这些结果共同说明：围绕Action–Effect binding、
strict Core×Procedure双必要、DC/AC手工重分配或不同reader反复重构，均没有
恢复v5.2/v6的absolute。

Target-Spectral已完成对“近rank1是否为直接瓶颈”的干净反证。fresh
macro50/100/150/200 correct400为`30/12/18/34`；四点配对、无放回和输出完整性
审计通过。best `34`甚至低于source base `48`，因此不续训、不做行为级控制。

它保留v6完全相同的`Q_text + M_f + G_f`、Semantic Core、native 50-suffix
mean Action、teacher-video transition和两层causal Procedure，只把compiler改为：

```text
Core/Procedure → 38个真实policy targets
→ target-specific value coordinates
→ 最后展开16个rank coordinates
→ row-orthogonal A、column-orthogonal U、16个learned spectral scales
→ rank16 public LoRA
```

Target-Spectral确实把effective stable rank从v6 m200的`1.00017`提高到
`3.3245`，把q/v跨层方向余弦从约`.968/.988`降到`.032/.066`；但LoRA范数
从`94.71`降到`25.87`，q/v能量从`74.5/25.5%`翻转到`39.0/60.9%`，层间能量
CV从`.047/.043`恶化为`1.294/.805`。16个同向component的建设性增益被16个
正交component的平方和合成取代，理论4倍幅度损失与实测`3.66×`一致。

Target m200与v6 m200的train functional loss仍几乎相同（`.10023/.10043`）；
内部Core/Procedure和order signal也相同，shuffle/reverse差异可传到effective
LoRA与policy action。失败因此不是上游没读teacher video，而是形式上更健康的
谱把更新写出source policy的高增益、q-dominant、跨层协调adaptation manifold。
同task视频相对方差略升但绝对创新约低3倍；不能把分母缩小当作视频能力增强。

当前活动实验不是继续Target-Spectral，而是补齐唯一缺失的2×2因果格：
`v5.2 topology + v6 task-complete fast-decay400 recipe`。模型精确恢复
v5.2的patch-grounded Core、native Action mean、causal Procedure和320-slot
AdaLN compiler，参数`10,237,704`；训练沿用4 ranks×6 tasks、full24等权、
真实视频长度cost balance、rank内long-first、B20和每25 checkpoint。fresh
macro0→200后默认exact-resume到400，候选150/200/350/400做paired correct400。
main `62598d3`的105-frame B20 profile和formal-seed resume smoke均已通过，
配置已seal；当前共卡吞吐折算400 macros约169分钟body。只使用GPU4–7；
GPU0–3不得查询或使用。Target-Spectral不得resume。

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

owner随后批准Semantic Core + Causal Procedure v5并完成了完整实现和训练：

- language-conditioned image-position hidden形成对帧顺序严格不变的Core；
- native fixed suffix与两个Meta-LoRA保留，但Action Expert只输出每帧
  robot-semantic hidden，不再产生7D forecast；
- 两层global causal Transformer形成可变长Procedure；
- Core先编译稳定LoRA content，Procedure以zero-init refiner作有向修正；
- 每rank每step一个task、1条teacher video、1套one-shot LoRA；该rank完整
  action batch的所有独立queries都监督这套LoRA。下一task visit轮换video，
  推理仍严格one-shot。

v5 fresh训练至step1800。fixed400 correct-video在step100/400/700/800/900/
1000/1400/1700/1800为`62/64/92/76/103/115/115/71/86`，step1000与1400并列
observed-best，按更低online functional loss和较晚时间点选择step1400做机制
检查。step1400固定400五臂为：

```text
correct / same-task-other / cross-suite-wrong / shuffled / reversed
115     / 108             / 74                / 113      / 114
```

correct相对wrong的paired净差为`+41`、exact McNemar `p=2.18e-6`；相对
shuffle/reverse仅`+2/+1`、`p=0.845/1.0`。内部Procedure sequence对
shuffle/reverse仍有`64.30%/72.56%`中位差，但到effective LoRA只剩
`2.93%/4.77%`，到fixed-query policy action只剩`0.49%/0.75%`。因此v5学习了
视频语义和上游顺序表征，却没有把顺序可靠编译成行为；继续同架构训练或进入
RL没有依据。

owner据此批准v5.1。它把预算从factor decoder移到上游语义表征：text-only
Gemma产生video-independent task-token queries，multimodal task-token hidden
提供逐帧视觉证据，token-aligned frame-set attention与language-axis
Transformer形成Core；Action Expert fixed suffix形成causal Procedure；中心化
Procedure通过zero-init AdaLN调制Core slots，再经一个post-fusion slot block
生成LoRA。v5.1训练与低学习率稳定段最终都未超过无放回
correct-video `127/400`，且best的reversed为`120/400`、与correct无显著差异；
它因此只作provenance。

Core-Program历史设计见
[`docs/action_forecast_writer_core_program_design.md`](action_forecast_writer_core_program_design.md)。
Core-Program使用text-only task axis、multimodal evidence与task-queried patch
evidence；Semantic Core对frame set置换不变，native Action mean与uncapped
视觉变化形成full raw causal Procedure。compiler以Core semantic basis和raw
Procedure program做width512严格双线性融合；公共宽度256、8 heads×32、
factor hidden256，真实预算`10,905,856`。Recenter/Loom/v10负结果见对应设计
和handoff；v8负结果与旧设计见
[`docs/action_forecast_writer_v8_design.md`](action_forecast_writer_v8_design.md)；
v7负结果与旧设计见
[`docs/action_forecast_writer_v7_design.md`](action_forecast_writer_v7_design.md)；
旧v6设计见
[`docs/action_forecast_writer_v6_design.md`](action_forecast_writer_v6_design.md)；
旧v5设计见
[`docs/action_forecast_writer_v5_design.md`](action_forecast_writer_v5_design.md)；
完整v4根因证据见
[`docs/action_forecast_writer_v4_root_cause.md`](action_forecast_writer_v4_root_cause.md)；
原咨询材料见
[`docs/action_forecast_writer_expert_consultation.md`](action_forecast_writer_expert_consultation.md)。

v5.2 step900 fixed correct400为`132/400`，五臂
`correct/same/wrong/shuffled/reversed=132/138/74/82/83`；same鲁棒且correct
相对wrong与两种order破坏均显著，因此没有v4 shuffled漏洞；它是v6的可信背景
而非当前resume入口。

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
- 每个macro update全局等权覆盖24 train tasks；4 ranks各顺序处理6 tasks，
  每task抽1条teacher video、生成1套LoRA，`B_a`条独立action queries先task内
  求均值，再以`task_loss/6`backward。前5轮DDP`no_sync`，第6轮唯一同步，
  然后只做一次clip/AdamW/scheduler。video与action episode/chunk不要求配对，
  action只进functional behavior loss。
- source base冻结，只有Writer更新。Writer不得看到action、proprio、reward、terminal、task ID、filename或隐藏stats。
- v10原位替换v8活动schema/config，不保留平行runner。公开rank-16 LoRA仍为
  76 tensors、`1,287,168` scalars；Writer参数`11,627,520`。
- GPU4–7真实最长视频profile选择B20。B20时每macro为24
  videos/LoRAs、480 queries、24次functional forward、1次同步和1次AdamW；
  三步含105-frame视频且全部finite，step1→3 exact-resume通过。
- 正式fast cosine decay400轨迹已从identity fresh完成400 macro、每25保存；
  12个paired correct400候选与macro50 best五臂/内部传递均已封存，不做
  checkpoint融合或同recipe续训。
- absolute达到预门后，对暂时best跑固定400
  correct/same/wrong/shuffled/reversed；要求same影响最小且correct明显优于
  wrong、shuffle、reverse。不通过则定位最早失效层后fresh迭代，不用
  contrast/order loss。
- focused absolute硬门为single-checkpoint correct400至少150且至少比
  corrected Source-SFT best109高30。150不是自动停止点；还需满足五臂、
  multi-task breadth和内部传递合同。

### RL-Writer

- 与完整AS-Writer best分开，从新架构规定初态做短、task-balanced AS cold
  start；持续用官方random-reset reward screen，直到24个train tasks每个至少
  一次真实success，再关闭action数据入口并跨source tasks做纯reward训练。
- rollout初态来自官方随机reset/BDDL机制，不来自fixed `.pruned_init`。
- cold-start必须报告teacher-action queries、每task first-success step和wall；
  不能偷换成完整AS-Writer continuation。
- 使用官方reward/success，不额外读取object pose构造privileged shaping。

### Source-SFT

- v6确认后必须从同一frozen source base fresh重训一套shared rank-128 LoRA。
- physical batch混合多个tasks，按`task→episode→chunk`分层均匀采样并做
  task-balanced loss；旧rank-pure SFT只作provenance。
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
- 训练最多8张A100，一卡一DDP rank为默认。当前focused v5.1只使用物理GPU4–7；
  0–3不进入visible set。首个AS segment约一小时；后续segment须重新过特异性、
  absolute和曲线证据门。历史task-local RL预算合同不覆盖本focused阶段。

## 10. Optional work and hard boundaries

- 核心闭环完成后有时间再做同base/split/one-video信息墙的ViVLA-style matched reproduction。
- source-only reward/meta outer learning只作更晚的可选增强，不阻塞Goal complete。
- 不使用bank、geometry、shared update subspace、residual escape、额外shared adapter、旧SmolVLA checkpoint/runner或MemLLM。
- meaningful状态更新 `task_plan.md`、`findings.md`、`progress.md`，验证、commit、push；等待长任务时继续推进不污染运行的后续工作。

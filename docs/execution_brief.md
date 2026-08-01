# EMBER Current Execution Brief

## 2026-08-01 AP-ADR current override

当前canonical AS-Writer和唯一formal训练由
[`action_forecast_writer_amplitude_preserving_dual_read_design.md`](action_forecast_writer_amplitude_preserving_dual_read_design.md)
负责。精确参数`10,241,024`；mean-backed permutation-invariant Core提供稳定语义
carrier，outgoing `[A_f,G_(f+1),G_(f+1)-G_f]`保留raw Program amplitude，38个
target-only Core reads与38×16 target/rank Program reads独立归一化后直接concat
生成coherent A/B。没有terminal RMSNorm/AdaLN/gate、global mixer、谱约束或第二
套LoRA。

105-frame、B20、4-rank三macro profile和formal-seed fresh0→1→exact-resume1→3
已通过；step1全部payload逐项不变，profile/resume seal `7dffb6f`已push。当前从
clean detached `7dffb6f` fresh identity运行raw-full24/fast macro0→200：

```text
tmux   ember-ap-adr-formal-7dffb6f
root   /data/ymdai/outputs/ember/pi05_as_writer_ap_adr_rawfull24_decay400_formal_dev_r4_b20_seed7_7dffb6f_20260801
log    /data/ymdai/logs/ember/pi05_as_writer_ap_adr_rawfull24_decay400_formal_dev_r4_b20_seed7_7dffb6f_20260801.log
frozen /data/ymdai/.codex/worktrees/EMBER-ap-adr-formal-7dffb6f-20260801
```

不得重复启动或从smoke续接。完成后评测macro50/100/150/200 paired correct400；
只有absolute/breadth/趋势或内部主路证据达到一小时门才resume到400。

新的同曝光证据把架构和recipe责任进一步拆开：UCP raw macro150→SERIAL step900
将x-only相对full的BA/action变化从`.0653/.01269`提高到`.4184/.12999`，但四个
checkpoint correct差值为`+7/-17/+21/-3`且漂移继续。故update mechanics确实
控制视频创新写出，却不是单独的性能解；后续必须联合分析scheduler、Adam/clip
时钟、long-first optimizer curriculum和topology，不得把v7/v8/v10/Loom等整版
思想按fast/full24 aggregate一棒子打死。

## 2026-08-01 UCP historical override

UCP raw-full24正式macro0→200及四个paired correct400已经完成；不得重复启动。
macro50/100/150/200为`82/117/100/110`，union为169，single best只有117；
train loss下降而held loss约`.131–.132`不改善，checkpoint间持续大幅gained/lost。
因此不resume到400、不做五臂。

macro100 refs1内部纵向已在`a4b06f5`通过：保持五条件carrier batch后canonical
recompute各层误差严格0；reader target/rank routing健康，但dynamic A/D到effective
BA/action的影响远弱于absolute X。首次零rollout exact50 root
`/data/ymdai/outputs/ember/pi05_as_writer_ucp_rawfull24_macro0100_internal_exact50_seed7_a4b06f5_20260801`
在rank1本地异常后被旧NCCL错误传播路径掩盖，只保留run contract，没有科学rows。
`874e5f1`已改为reference级fail-fast artifact和analysis-only两小时Gloo控制组。
新refs2随后精确复现`libero_spatial task3/reference1`的rank-gauge sanity失败，
torchrun立即收割其他ranks。instrumented `e47ffe8`证明raw A/B改变
`.74184/.13602`，effective BA relative L2仅`1.299e-9`，而bf16 fixed action因
rank-reduction顺序改变产生`.002047` execution drift。sanity现只对finite与BA
`2e-5` fail-close，action drift保留为实际诊断，不伪称bf16位级gauge不变。clean
`c4b85e8` refs2已经完整通过：8 tasks×2 references共16 rows、四rank各4，无
failure artifact。随后同一frozen commit的exact50也已自然完成，root为
`/data/ymdai/outputs/ember/pi05_as_writer_ucp_rawfull24_macro0100_internal_exact50_v2_seed7_c4b85e8_20260801`，
400 rows严格覆盖8 tasks×reference0..49、四rank各100 rows、0 rollouts且无failure。
pooled same-task effective-BA/fixed-action centered variance占sample energy仅
`.09008%/.01656%`；same/wrong/shuffled/reversed的Program→BA→action为
`.215/.499/.356/.440 → .043/.187/.063/.105 → .0138/.0636/.0153/.0325`。
correct norm/stable rank/top energy为`59.108/1.00319/99.714%`。analysis/summary
SHA为`a6e40cd6...25a8`/`386a04f5...acaa`。旧tmux已自然退出；不得复用失败root
或修改frozen worktree。

训练反事实冻结UCP拓扑，只改update granularity：四rank每update各1 task，
六phase重建同一full24 cost-balanced cycle；1,200 updates等于200 task visits、
4,800 videos和96,000 queries。LR严格满足
`LR_serial(u)=LR_full24(floor(u/6))`；不是连续warmup102/decay2400。实现已以
`ccdf21f/92548ed`集成main；fresh serial config/checkpoint/rank schemas、midcycle
cursor、cycle-boundary scheduler及formal `%6` fail-close均通过，全仓
`233 passed`。clean detached `10a71a1`的最长视频seed172 18-update B20 profile
已自然完成：3个cycle各覆盖24 tasks，首update真实读入105 sampled frames，峰值
allocated/reserved为`76,971,835,904/83,647,004,672` bytes。formal seed又完成
fresh0→1、resume1→3、resume3→7；step1/3全部文件SHA不变，前6 phase覆盖24
unique tasks，scheduler只在step6推进且step7使用下一LR。config已seal；随后已从
clean frozen `3db82df` fresh identity完成1,200 updates，不从smoke续接。正式root为
`/data/ymdai/outputs/ember/pi05_as_writer_ucp_serial4_exposurematched_decay400_formal_dev_r4_b20_seed7_3db82df_20260801`。
训练wall `4197.076s`，1,200行metrics、8个150-step checkpoint、96,000 queries、
4,800 videos、200 cycles全部完整；validation/test action读取0。训练tmux已自然退出，
不得重复启动。

held loss在step150..1200为
`.132407/.131304/.133484/.132973/.130352/.132508/.132237/.132918`；不用于提前
选择candidate。step300/600/900/1200四个paired correct400现已分别在GPU4/5/6/7
启动，tmux为`ember-ucp-serial4-correct400-3db82df`；四个prepared contract均为
400 states、36个long-first dynamic shards、6 replicas/6 Writer generators，
teacher action读取0。当前只等待自然完成，不得重复启动。

严格跨曲线审计确认七条正式run共用同一512-row held panel manifest
`53cbf9e...a3a8`；主20点按架构去均值后的held→correct相关反而为
Pearson/Spearman `+.462/+.644`，相邻Δheld→Δcorrect仅`+.120`，逐task去均值后
held→success仅`-.055`。因此functional loss只作finite和teacher-state局部拟合
诊断；closed-loop裁决必须使用paired correct400、breadth、gained/lost/Jaccard及
effective BA/action方向传递。

该审计已封存到
`/data/ymdai/outputs/ember/pi05_as_writer_functional_surrogate_closedloop_audit_seed7_20260801/analysis.json`
（SHA256 `91eaabed...12a`）；120个输入文件和44个correct400 panels均经脚本复验。
post-v5能力审计另封存在
`/data/ymdai/outputs/ember/pi05_as_writer_postv5_recipe_confounded_capability_audit_seed7_20260801/analysis.json`
（SHA256 `406b9098...80e`）：扩展24 checkpoints的四架构envelope union为246，而全部
checkpoint共同成功仅5 states，进一步禁止按aggregate整体处决历史结构。

训练mechanics审计封存在
`/data/ymdai/outputs/ember/pi05_as_writer_architecture_training_mechanics_audit_seed7_20260801/analysis.json`
（SHA256 `c910a933...e521`）。旧recipe每完整24-task exposure做六次mean4同LR更新，
新full24只做一次mean24；实测累计一阶LR系数约差`6.0069×`，并同步改变Adam
记忆、clip、WD和groups2–6的参数重线性化。v6 old/new-slow在B20、video/query与
exposure LR对齐后，visits100→150参数路径cosine仍仅`.0493`。故serial结果只能
解释完整update-mechanics bundle，不能单因归为梯度抵消或task batch size。

当前下一整体架构设计为
[`action_forecast_writer_unified_causal_program_design.md`](action_forecast_writer_unified_causal_program_design.md)。
exact v5.2 task-complete已以候选`51/91/106/120`和winner五臂
`120/109/107/111/124`封存，不resume。SPG canonical macro0→200也已完成；
macro50/100/150/200 paired correct400为`97/115/77/100`，一小时门失败，
不续到400、不跑五臂。

SPG内部证明Program不是断路：same/wrong/shuffled/reversed的Program relative L2
为`.967/1.186/1.193/1.202`，但到effective BA压到
`.066/.221/.116/.116`。CoreReader entropy `.999992`，ProgramReader的target/rank
centered routing只有约`4–5e-5`；exact50 LoRA几乎严格rank1且B columns相同。
最早失败是`std=.02` identity被normalized Core淹没、独立Core加法旁路和global
coordinate mixer造成target/rank同质化，不是Program没看视频。

训练端同样是根因的一部分。CP projected/raw cosine约`.983`且主要把norm放大
约`1.25×`；raw full24 mean末段只保留平均单task gradient energy的`4.79%`，
投影不能恢复非负但近正交的task innovations。B20长期无偏，但4,800个task
visits中`6.44%`漏掉至少一个五等分进度区间，单visit phase TV均值`.1756`。

Unified Causal Program canonical实现把absolute `X_f=M_f+G_f`、native `A_f`和正确
outgoing `G_(f+1)-G_f`放在同一个causal axial Program中；删除独立Core旁路、
target-Core first hop和跨target/rank mixer；normalized target/rank identities
单级直接读取raw Program。训练恢复raw full24 mean并保留只读Gram，B20使用
边缘仍uniform-row的20-strata随机jitter降低过程覆盖方差；首版保持fast400，
不同时混入slow2000。真实参数`7,683,328`，全仓无GPU回归`203 passed`；fresh
config/checkpoint/eval schemas已经替换旧SPG active path。真实105-frame B20三
macro已在`0d4c271` frozen code上完成，step wall
`20.394/18.494/18.504s`，峰值reserved约77.62GiB；formal-seed
fresh0→1→exact-resume1→3逐文件不变且cursor连续，config现已seal。

v7/v8/v10/Loom及后续低分只证明“架构×当时fast task-complete recipe”失败。
只有全局binder、早event pooling、无监督confidence/gap、DC删除、strict bilinear、
高增益gate和强制谱等被内部反事实独立否定；anchors、causal Procedure、双流、
Core语义和target-first/rank-last仍可复用。当前下一分析和训练反事实就是本节
开头记录的refs2→exact50与serial-4；不得复用失败root。150只是里程碑，不是focused
自动终点。以下在`## 1`之前的旧状态叙述
只作历史背景，不得覆盖本节。

匹配每task 150次video exposure的正式2×2审计进一步支持该边界：v5.2 old/new
为`132/51`，v6 old/new为`95/111`，recipe effect=`-81/+16`、描述性
difference-in-differences=`97`。它仍混杂optimizer/scheduler/AdamW时钟，不能
归因给单一recipe开关；但足以禁止把任何fast-task-complete低分整版架构判死。

状态：2026-07-31。共享 π0.5-LIBERO source base与corrected mixed-task
rank-128 Source-SFT均已封存，后者development observed-best为`109/400`。
当前可信架构标杆是v5.2与v6：v5.2 step900 single-checkpoint correct400为
`132`，五臂`132/138/74/82/83`；v6 task-complete single-checkpoint best为
`143`，五臂`143/135/125/128/129`。前者视频语义和顺序margin强，后者
absolute更高但margin较弱；两者都未达到focused absolute门150。

v7/v8/v10/Loom/Recenter/Core-Program/Prior–Innovation均已完成，并作为各自
“架构×当时fast task-complete recipe”的负结果provenance，而不是对整版思想
的独立否定。其关键correct400 best依次为`120/125/103/112/85/84/100`；
Prior四点为`100/61/89/88`。这些结果说明上述具体组合没有恢复v5.2/v6的
absolute；其中只有近均匀global binder、早event pooling、无监督confidence/
gap、联合删除DC、strict bilinear、高增益末端放大和强制谱等被内部证据局部
否定。Action anchors、双流、局部关系、Core语义与target-first/rank-last仍需
结合训练合同判断。

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

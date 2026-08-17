# V6-LPCP CFMG Gradient-Open Memory Query

状态：2026-08-17 **cycle4稳定性裁决完成，终局non-pass**。简称 **GOMQ**。本轮从sealed V6-LPCP143 fresh
开始，不resume SEOD cycle4。部署前向、K4输入、rank32输出、successful-expert occupancy credit、four-view
task gradient、median upper cap和natural Adam全部保持；唯一主要变量是让37个真实backbone input memory tokens
在payload gate打开后真正接收reward gradient。

## 1. 最早失效接口与新证据

SEOD四个相邻strict score/breadth=`129/6 -> 135/6 -> 143/5 -> 136/5`，cycle3→4=
`119 retained / 17 gained / 24 lost`、churn41、net`-7`。cycle3→4 all400 effective-BA relative-L2=
`.002809`，first4同task不同K4更新cosine/energy=`.994106/.992830`；cycle4 train same-task view gradient
cosine=`.945774`，但cross-task mean/min仅`.053370/-.457658`。所以训练量、跨video相消和native LoRA写出都不是
当前最早解释；缺口是task-local有用方向不能在一个shared checkpoint中共同保留held support。

终局审计新增了一个架构级反事实。正式runtime先在`torch.no_grad()`中缓存`WriterConditioningState`，reward
backward只对缓存的`layer_memory_states`重跑下游grid。37个输入`memory_tokens`因此四轮的task/shared gradient、
Adam `exp_avg`和`exp_avg_sq`严格为0，checkpoint间只有weight decay。SEOD实际训练的是fixed random memory
queries之后的temporal/set/M2P/gate，而不是原设计声明的learned memory reader。

该事实不自动说明“把梯度打开就会变好”。它只提供一个此前没有被正式检验、且可能直接影响shared representation
的最小架构变量。

## 2. 方向取舍

### 2.1 选择GOMQ

memory input位于18层Action Expert policy topology之前。它的初始值决定每层query怎样从真实image/language/
Action context读取Value；同一组learned queries在不同task context中产生不同状态，但参数本身跨task共享。
如果reward能把它学成一组可复用的task-procedure读取基，异质task就不必只靠backbone之后的2.79M参数各自拟合，
有可能在进入temporal/set/M2P前先形成更共同的policy-aligned坐标。这正面针对当前cross-task coexistence缺口。

### 2.2 本轮不继续改credit/optimizer

PCSD、CV-CSD、support projection、unit secant、finite/backtracking/common-descent、median cap和SEOD已经广泛改变
credit或shared commitment。SEOD又证明successful-expert occupancy稠密、same-task跨video一致并能写到action，
但没有提供新的task-level solver证据。继续换loss、cap、PCGrad式solver或scalar ray会重复已关闭的接口。

### 2.3 本轮不同时改rank或full A/B

owner关于rank8、输出维度和可扩展LoRA生成的判断仍有效；历史uniform rank14也没有否定fresh rank8。但把
rank32 B-only residual同时换成rank8/full-factor会改变carrier support、可达子空间、token数量和mapper，无法判断
收益来自哪里。本轮保留rank16 LPCP carrier + rank16 native-zero B residual，只裁决learned memory query。

## 3. 完整输入到输出

```text
exact task language + four same-task action-hidden ordered videos
    -> each stride-5 frame: native image/language prefix + 50 native Action probes
    -> one frozen native context forward
    -> 37 shared learned memory queries read same-layer prefix/Action K/V
       and update through all 18 Action-Expert layers
    -> per-frame Z[18 layers, 37 memory tokens, 1024]
    -> per-video directed temporal reduction (adjacent delta + end-minus-start)
    -> permutation-invariant K4 video-set aggregation
    -> layer-axis and token-axis topology mixing
    -> coordinate payload gate
    -> direct native B rows for q/v/action-in/action-out
    -> one rank32 38-target LoRA
       = untouched rank16 LPCP carrier + rank16 native-zero B residual
    -> frozen source policy closed-loop rollout
```

37来自当前rank16 B-only residual的`680,448`个B值与`18x37x1024` grid ownership，不是任意“多加token”。每帧
共享同一组37个input tokens；`F x 18 x 37`是运行状态，不是为每个video创建独立参数或最终LoRA再平均。

## 4. 视频、语言与顺序为什么仍不可绕过

- exact language与真实image tokens共同构成native prefix，language说明任务对象/目标并为memory attention提供地址；
- memory只能单向读取真实prefix与50 Action states，Action/prefix不能读取memory，因而不改变sealed LPCP carrier；
- 新增B Value来自每video的相邻变化和goal residual，再经过有因果方向的temporal reducer；constant dynamic路径仍为0；
- K轴只做置换不变的集合聚合，不平均frames、raw features或最终LoRAs；
- 同一个language下reverse/shuffle改变逐帧memory states的有向组合，不能由静态language独立写residual。

GOMQ不声称已经理解高层程序；它只提供让共享query学习“在policy层内看什么”的能力。最终仍必须由correct相对
wrong/shuffled/reversed/no-video和same-task-other closed-loop controls证明视频知识真正有用。

## 5. 唯一训练图变化

SEOD前向数值与GOMQ step0逐元素相同。实现只把memory observer从no-grad cached-state边界中打开：

1. native image/language/Action context仍在`torch.no_grad()`中运行一次并完全冻结；
2. 37 memory inputs和其18层one-way observer在autograd中运行，frozen policy weights只传input gradient；
3. candidate LoRA用于expert/student matched action query时detach，不让policy query反向建立第二条图；
4. functional LoRA cotangent仍只通过一次cached-condition recompile回传；
5. payload gate保持zero-init，因此cycle1 memory gradient按结构仍为0；cycle1 gate更新后，cycle2才是第一个真实
   memory-token update；
6. 所有source-policy、LPCP/query carrier、Meta-LoRA、expert adapters和held assets仍零trainable。

formal必须fresh。若从SEOD cycle4中途打开memory，会把四步Adam moments、已经漂移的下游grid与新上游参数混在
一起，既不是exact resume也不能裁决单变量。

## 6. 效率实现边界

- 不增加第二次native context backbone forward；继续使用activation checkpointing只重算memory observer；
- expert rollout先完成，只有出现successful trajectory后才保留differentiable condition graph，避免图跨环境
  rollout长期占显存；inactive task仍保持零credit和task权重语义；
- 每个active task的四个correct K4 conditions各自只编码一次并各自backward一次；不重复single forward；
- actual Adam后的四view objective继续明确标成cached-layer-memory downstream diagnostic，不能冒充包含新
  memory value的完整finite counterfactual；
- 每个active task只额外完整re-encode anchor K4一次，用于记录memory update真实穿过Writer→LoRA→q/v/action
  的post-update response；不为四个views全部重复context；
- 不新增batch1、FP64训练、逐tensor扫描、内容hash、重复inference或防御性校验。

## 7. Pre-formal两周期机制门

固定四suite train anchors=`2/12/21/35`，只用于图和吞吐裁决，不用于选择formal checkpoint。fresh smoke连续两
cycles：cycle1打开gate，cycle2第一次更新memory。必须同时满足：

1. cycle1 input-memory gradient和moments为0，payload gate非零；cycle2 memory gradient、Adam moments和parameter
   delta均finite且非零；
2. policy、expert、carrier、Meta-LoRA仍零gradient，LPCP first bank逐tensor不变；
3. cycle2四个task各自four-view gradient非零，same-task coherence不因开图崩溃；
4. 单独报告memory-token slice的跨task pairwise cosine，以及去掉该slice后的downstream cosine。若memory-only
   mean≤0且full shared descent不优于fixed-query部分，说明它没有形成预期共同读取基，本轮在formal前终局；
5. post-update anchor重新完整forward时，memory-token更新对native q/v/action BA和fixed action有material response；
   只在cached old layer-memory中变化不算；
6. train anchors与validation8 action-hidden four-view的memory-only LoRA contribution保持同task共同方向，不破坏
   reverse/constant/K-set；held不读action/reward；
7. longest-video、真实wall、samples/s和peak显存可在A40上稳定运行。相对fixed-query SEOD显著变慢而没有上述
   新证据则终止，不用小batch或低效dtype救。

## 8. Formal训练量与strict裁决

若机制门通过，从sealed LPCP fresh启动full24：

- cycle1与cycle2作为预先计划的一段连续训练；cycle1只有gate update，不单独花400 rollouts重测已知低信息节点；
- cycle2 checkpoint立即做K4 strict paired400，并与fixed-query SEOD cycle2=`135/400`、LPCP143、v6-fast143及
  old134/compiler138/online128逐task比较；
- cycle2若低于130、breadth不高于4且retention没有任何改善，说明第一次memory update已是明确坏方向，终局；
- 其它非灾难结果继续到cycle3，因为cycle2只有一次memory update；cycle3再做strict400并报告cycle2→3 churn；
- cycle3若correct、breadth、retention和churn相对matched fixed-query SEOD均没有一项实质改善，或absolute更低且
  lost更多，终局；不得靠cycle4反复抽峰；
- 若cycle3达到约143以上且breadth/retention/churn明显优于SEOD，完成cycle4相邻稳定性节点；若达到约145且
  retention合理，立即补correct/same-task-other/wrong/shuffled/reversed/no-video，同时仍需cycle4证明稳定；
- 若cycle4稳定保持约145或更高，即使没有超过150也可按owner标准成为有价值结果；若再次峰值回落则如实终局。

全部选择只认single checkpoint strict rows；内部cosine、memory gradient和BA只解释。不得扫LR、token数、rank、
scale、seed、gate init或训练长度。

## 9. 快速否决与负结果边界

以下任一足以否决当前GOMQ：

- cycle2 memory gradient仍为0或只剩weight decay；
- memory update只改变train cache，完整re-forward的LoRA/action无响应；
- memory-only task gradients冲突且没有改善shared descent；
- held video contribution相消、失去有向时序或产生static/language-only bypass；
- strict曲线继续复现SEOD的高churn峰值换手，没有breadth/retention改善；
- 吞吐/显存代价显著而没有新的policy-effective signal。

负结果只淘汰`LPCP + 37 learned one-way memory queries + CFMG B-only grid + SEOD credit`这一组合。它不否定
rank8/full-factor、其它memory容量、dynamic-K/few-shot、生成LoRA或未来生成LoRA后的task-local RL。

## 10. 预期实现影响面

- `src/ember/writer/reward_cycle.py`：拆分pack与live memory encoding，打开memory observer gradient并避免图跨rollout；
- `src/ember/writer/reward_gradient_update.py`：anchor post-update full re-forward和memory-slice coexistence诊断；
- `src/ember/writer/reward_config.py`及唯一active config：fresh GOMQ schema/identity、两周期smoke与formal节点；
- focused tests：no-grad旧路径必须失败，gate-open cached recompile必须到达memory token，cycle1/2 staging、
  detached candidate、policy freeze和post-update full re-forward；
- 旧SEOD executable/config由Git和formal artifacts保存，不保留并行兼容fallback。

实现验证：以上改动已在唯一canonical runtime原位完成；候选LoRA保持detached、frozen policy无梯度、live
`layer_memory_states`可把gate-open cotangent传到input `memory_tokens`，post-update anchor执行完整Writer re-forward。
focused回归通过，完整CPU=`418 passed`，`compileall`与`diff --check`通过，architecture guard无hard violation。

## 11. Pre-formal实测裁决

clean`51e3d73`在gpu02物理`1/2/3/4`完成world4 two-cycle smoke并exit0。cycle1/2 wall=
`96.0967/102.5570s`，expert successes=`8/6`、failures=`0/2`，peak reserved=`20.818GB`。cycle1 memory gradient按
zero gate为0；cycle2四task均非零，shared gradient RMS=`8.0168e-8`、parameter delta RMS=`.0001537793`，由
Adam更新方程可知一、二阶moment同时非零。memory-only/downstream task cosine mean/min=
`.155845/-.280301`与`.412169/.296898`；四task same-task four-view cosine=`.974869--.993822`。四个post-update
anchor完整重编码均产生非零q/v/action BA与fixed-action response。

随后固定全部cycle2 downstream参数，只把37个memory queries恢复初始值，使用train anchors与validation8的四组
action-hidden K4视频做反事实。validation8的memory-only contribution 8/8 pairwise cosine为正，aggregate
cosine/energy=`.126548/.343180`、held/train L2=`1.11159x`。full residual的natural→reversed relative-L2=
`1.95298`，constant/natural L2=`.002048`。K4视频集合换序的full residual cosine最低`.99994085`；其差异相对
LPCP BA估算最大`3.687e-5`，符合允许的BF16 reduction-order低位差异。memory-only是两个近同BF16 forwards之差，
其换序relative值被数值底噪放大，不能覆盖部署full output证据。

因此第7节机制门通过，formal从sealed LPCP fresh cycle0→2；不得resume smoke parameter grid。该通过只说明
learned query值得full24/strict检验，不是closed-loop性能结论。

## 12. Formal cycle2/cycle3结果与训练量裁决

clean pushed/frozen `8553b61`在gpu02物理`1/2/3/4/6/7`、world6从sealed LPCP fresh完成cycle1--3；cycle1
只打开payload gate，cycle2和cycle3才是input memory及其content链真正更新的前两步。三个cycle均为24 tasks、
48 expert rollouts和一次shared Adam update；cycle2/3 active tasks=`17/16`，memory query gradient、Adam moments、
parameter delta及q/v/action response均非零。

cycle2 K4 strict paired400首次得到`151/400`、breadth6、per-task=`0/3/47/34/0/40/26/1`、per-suite=
`3/81/40/27`。相对LPCP143严格=`126 retained / 25 gained / 17 lost`、churn42、net`+8`；相对完全matched的
fixed-query SEOD cycle2=`135`严格为`122 retained / 29 gained / 13 lost`、net`+16`、McNemar
`p=.0195205`。因此打开reward到input memory queries确实产生了closed-loop增益，不能把GOMQ写成只改善内部
geometry。

但cycle3 strict降到`135/400`、breadth6、per-task=`0/1/48/31/0/37/17/1`、per-suite=`1/79/37/18`。
cycle2→3严格=`122 retained / 13 gained / 29 lost / 236 both-fail`、churn42、net`-16`、Jaccard
`.743902`、`p=.0195205`；Long1净丢9，四个suite均不增。相对LPCP143为`119/16/24`，相对matched SEOD
cycle3=`143`为`120/15/23`。所以151目前只是一个早期高点，不具稳定资格。

FP64相邻checkpoint分析排除了“第二步又把同task不同视频写散”这一解释：cycle2→3 all400 effective-BA
relative-L2 mean/median=`.00235810/.00202229`，first4同task四组K4增量cosine/energy=
`.993160/.992606`。gained/lost的改写幅度=`.001999/.002154`，不能选择有用方向。cycle3训练侧仍有
`54/64` cached-view objectives下降，memory/downstream跨taskgradient cosine mean=`.134870/.054144`，但
held closed-loop显著下降。最早缺口是**连续shared reward commitment没有保留held on-policy support**，不是
memory未获梯度、视频集合相消、native写出或训练图未工作。

本节结果落盘后，owner进一步明确：当前每cycle训练量很小，出现好结果后应多训练再判断。该最新authority允许
在不改代码、数据、optimizer、world topology或科学变量的前提下exact-resume cycle4，目的仅是刻画训练剂量和
相邻稳定性，不是从曲线挑峰。若cycle4继续恶化则停止；若恢复，也不能把恢复点选成答案，必须继续至少一个相邻
checkpoint并同时完成cycle2的same-task-other/wrong/shuffled/reversed/no-video因果controls。原第8节的
“不得靠cycle4反复抽峰”仍成立；这次continuation只防止用两次真实memory update过早外推终局。

## 13. Cycle4终局稳定性裁决

相同clean frozen `8553b61`、world6与物理拓扑从cycle3 exact-resume到cycle4。训练完整完成24 tasks、48
expert rollouts、`29/19` success/failure，18 active tasks覆盖四suite，72个four-view credit conditions使用288条
不同视频；cycle=`412.952s`。memory-token shared gradient RMS=`4.031e-7`，memory/downstream跨task cosine mean=
`.125322/.056180`，17/18 task final descent、65/72 cached-view objectives下降，q/v/action与完整Writer
re-encode response均非零。训练图、memory gradient、native写出、吞吐和信息墙都没有工程失效。

cycle4 K4 strict paired400完整72/72 shards、400 rows、18/18 workers exit0，结果为`131/400`、breadth6，
per-task=`0/1/45/32/0/36/16/1`、per-suite=`1/77/36/17`。cycle3→4严格=
`116 retained / 15 gained / 19 lost / 250 both-fail`、churn34、net`-4`、Jaccard`.773333`；相对cycle2高点为
`121/10/30`、net`-20`、`p=.002221`，四suite全部不增。相对LPCP143为`117/14/26`、churn40、net`-12`。
三个真实memory/content update的score因此为`151 -> 135 -> 131`，breadth始终6；151是早期峰值，不是稳定平台。

FP64 cycle3→4 all400 effective-BA relative-L2 mean/median=`.00284989/.00255794`，比cycle2→3的`.00235810`
更大；first4同task四K4增量cosine/energy仍为`.993372/.992705`，q/v/action分别都约`.993--.996`。
gained/lost改写均值=`.002596/.002791`且大幅重叠。Spatial3得到全task最大改写`.005171`却仍只有`1/50`，
Goal3改写`.002115`仍为0；训练局部credit广泛下降也没有转化为held成功。因此最早失败接口最终确定为：
**video-coherent learned memory representation经独立direct-B共享尾部连续提交时，没有形成可保留的held
policy support**。不是memory没有读取视频、同task不同视频再次正交、native LoRA不可见或训练量不足。

GOMQ终局，不cycle5、不补cycle2六臂、不扫rank/LR/scale/seed或继续训练长度。六臂未运行，所以本轮不能宣称
same-task-other鲁棒或correct相对wrong/shuffled/reversed/no-video的因果优势；这些资格是未知，不是失败。
本轮只否定当前`LPCP + 37 learned one-way memory queries + temporal/K-set/grid + rank32 native-zero direct-B +
successful-expert occupancy shared Adam`的连续稳定性，不否定memory token、V6 compiler、rank8、few-shot或生成
LoRA。后继若保留memory，应改变memory Program到LoRA的commitment接口，而不是继续调整同一direct-B tail。

正式root：
`runs/outputs/pi05_v6_lpcp_cfmg_gomq_cycle4_k4_correct400_noreplacement_seed7_trainr6_evalr6_8553b61_gpu02p123467_b16_20260817`。
同root封存`gomq_cycle2_cycle3_cycle4_strict_adjudication.json`与
`gomq_cycle3_to_cycle4_effective_ba.json`。

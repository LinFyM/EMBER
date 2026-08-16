# V6-LPCP Capacity-Matched Backbone-Memory Grid

状态：2026-08-16 full24 cycle1已因global commitment exact no-op终局，不strict400/cycle2/controls或小扫。
简称**CMBG**。CAPG已经
终局，不得resume、放宽其门或用normalization/PCGrad/task权重/scale小扫补救。历史8-memory Dynamic-K只选择性
复用了其已验证的layer-matched memory语义，没有整条恢复。

## 1. 决策

CMBG保留V6-LPCP、NEAP endpoint preference、K4 four-view credit、rank32 native-zero residual bank、task9/15/18
world3共同裁决，以及CAPG已经证明有效的video内有向聚合、K-set、layer/token M2P和direct B reshape。唯一主要
架构变量是把CAPG的

```text
已有50个Action states -> backbone之后的37个query latents
```

替换为

```text
真实image/language prefix + 50个Action probes
    -> 一次逐元素保持LPCP的native PI0.5 joint 18-layer forward
    -> 37个one-way memory tokens逐层读取同层prefix/Action K/V并走Action Expert update
    -> 每层37个contextual memory states
```

这37个逐层使用真实Action Expert的AdaRMS、q/k/v/o、attention、MLP与residual，不是视频结束后的latent slots；
但它们是严格单向observer，不扩张原carrier的attention矩阵，因而不能通过数值shape副作用改变LPCP。每个condition
仍只生成一套完整38-target LoRA；Writer仍在rollout前运行一次。

## 2. CAPG实际证明了什么

CAPG clean`878b5e4`的world3固定outcome/count、48条correct videos、信息墙和wall全部通过。相对TCEC，它把
task9/15/18的same-task four-view pairwise gradient cosine从`.846/.596/.448`提高到
`.983/.898/.982`，energy ratio从`.865/.645/.557`提高到`.985/.870/.949`；raw equal-task mean由只帮助
`1/3`提高到`2/3`，native最佳覆盖由`8/12`提高到`10/12`。所以capacity-matched condition coordinates不是无用
美化，它已经显著修复同task不同video credit近正交。

CAPG仍失败于更窄接口：step0的共同写出只经过一个shared coordinate gate。task15梯度范数仍是task9/task18的
`36.29x/5.99x`，task18到shared mean cosine=`-.1570`，11个native scales无一达到12/12，最终exact no-op。
因此本轮不改endpoint、optimizer、rank、video数量或后处理；只检验**让parameter-aligned values在真实policy
backbone内先完成task-conditioned读取，能否在到达同一个gate以前形成跨task可分流坐标**。

## 3. 完整数据流

```text
exact task language + K=4 action-hidden ordered teacher videos
  -> 每帧真实224图像、exact language、固定t=1的50个Action probes
  -> 原生18层PaliGemma/Action-Expert context forward并只读每层context
  -> 37个parameter-aligned memory tokens逐层读取同层context并走Action Expert
  -> 每帧18x37x1024 layer-memory grid
  -> exact-zero coordinate payload gate
  -> 每video adjacent transition + terminal goal causal reducer
  -> K维permutation-invariant set attention
  -> layer轴和parameter-token轴M2P
  -> deterministic direct reshape为q/v/action-in/action-out B residual
  -> 与LPCP rank16 first bank组成一套rank32 LoRA
  -> frozen source policy
```

CAPG的compact LPCP probes、Semantic Core、conditioned Procedure和first-bank LoRA全部保留；memory grid只拥有
新增second-B value。当前实验只claim K4 few-shot，不因模块可接受K1--K4就宣称已经完成Dynamic-K训练。

## 4. 为什么是37个memory tokens

当前单变量继续保留rank16 LPCP carrier加rank16 B-only residual。每个Action Expert layer的native B payload为：

```text
q_B: 16 x 2048 = 32 x 1024
v_B: 16 x  256 =  4 x 1024
```

所以q/v需要36个1024-wide tokens。第37个token沿layer轴承载endpoint：layers0--15写`action_in_B[16,1024]`，
layer16的前512值写`action_out_B[16,32]`。总grid为`18x37x1024=681,984`值，消费680,448，padding1,536，
只占`.225%`。这是SHINE式capacity-matched direct payload的真实shape推导，不是随意提出约70个tokens，也不是把
token数等同视频阶段数。

本轮不同时降到rank8：fresh rank8仍是开放且重要的独立变量，但它会同时改变carrier support、A/B topology、
memory count和closed-loop compute。CAPG最早失败在跨task first commitment，先保持rank和LoRA图才能解释memory
变量。CMBG失败不得写成rank8失败。

## 5. memory位于哪里、看见什么

对每个真实frame，carrier的正常prefix和suffix保持封存LPCP原样：

```text
prefix = [256 real image tokens, exact language tokens]
suffix = [50 native fixed Action probes]
```

每层另有37个learned memory queries。其可见关系等价于三个attention blocks，但实现为carrier与observer分离：

1. prefix queries只看有效prefix；
2. Action queries看prefix和完整50-token Action block，执行图与LPCP逐元素相同；
3. memory queries看该层prefix、Action和memory K/V，memory自身经过对应Action Expert layer更新。

所以memory处于有意义的真实图文/Action context中，不是blank image、memory-only forward或凭空Action query；同时
memory不能反向改变原50个Action probes的可见集合。memory使用与固定`t=1` probes相同的AdaRMS condition，经过
每层attention residual、post-attention norm和MLP residual后保留状态：

```text
Z_frame in R[18,37,1024]
```

实现只运行一次原生context backbone forward；不得另算no-video baseline、重复vision/language/Action context或
修改site-package。observer保存每层原生context输入，重算该层冻结K/V供37个memory queries读取，不重算context
queries、attention、MLP或最终输出。首版把memory直接追加到joint矩阵，虽有three-block mask，仍因attention矩阵
shape改变使task15固定outcome从`2/0`漂成`1/2`，故属于工程违约。修正版已经用原生shape与fixed world3关闭这个
结构问题。跨运行正常BF16/TF32、batch shape、kernel和reduction低位差异不作为门；不会为逐元素一致退化到
batch1、FP64或第二次context forward。

## 6. 视频顺序和多视频共识

CMBG不把memory本身误当作视频程序。对每条video，37个memory tokens先逐帧独立得到`Z_0...Z_T`，再沿真实
stride-5 ordinal构造：

```text
D_0 = 0
D_t = Z_t - Z_(t-1)
G   = Z_T - Z_0
```

沿时间只有一个shared causal controller；它以signed adjacent deltas为Value并以terminal goal residual收尾。这样：

- reversed改变D符号和G方向；
- shuffled破坏相邻阶段连续关系；
- constant video令D和G逐元素严格为零；
- static language、learned memory identity或单帧外观不能单独写出LoRA Value。

每条video先独立完成有向聚合。之后只在固定`layer x token` cell的K轴做无shot-position set attention并对equivariant
输出对称归约，得到一套shared grid。不得平均frames、raw per-frame features、分别生成的LoRAs，也不得选择“最好
video”。CMBG沿用CAPG已通过的K置换不变合同。

## 7. layer/token M2P与LoRA写出

K-set后grid依次做最少的两种结构通信：

1. 固定parameter token，在18个policy layers之间通信；
2. 固定layer，在37个parameter tokens之间通信。

时间轴只在video causal reducer，video轴只在set，policy/parameter轴只在M2P；不把四个轴在每一阶段全部交替
attention。最终按固定shape直接reshape：

```text
grid[:, :, 0:32]       -> q_B[18,16,2048]
grid[:, :, 32:36]      -> v_B[18,16,256]
grid[:, 0:16, 36]      -> action_in_B[16,1024]
grid[:, 16, 36, 0:512] -> action_out_B[16,32]
```

没有`320x256 -> wide heads`、family W2或分别为38 targets建立独立hyper-head。扩展到更多policy layers或LoRA
targets时增加对应layer/token grid，而不是平方增长一组互不共享的heads。

## 8. exact-zero staging和训练参数

memory states先乘一个`18x37x1024` coordinate payload gate。该gate逐元素zero-init，因此step0 second-B严格为零，
public LoRA为LPCP first bank加native-zero second bank。constant video经过D/G仍严格为零。

本轮明确保留CAPG的staging：首个backward只有payload gate获得梯度；accepted gate update以后，memory tokens、
video causal、K-set和M2P才获得梯度。这样world3首门直接回答“真实backbone memory values能否让三个task在同一
shared gate中找到共同finite step”，而不是先让一个task私下打开通道再要求其它task追赶。若该首步仍失败，不能
事后加per-task warmup或放宽12/12。

预计trainable参数=`2,828,928`：CAPG删除217,344参数的post-backbone context Q/K/query readout，保留681,984
coordinate gates，新增37,888个learned input memory values，其余temporal/set/M2P不变。实现后以实际枚举为准，
若不等必须在GPU前修正authority或实现，不能默默改变图。

source policy、LPCP/AS139 carrier、现有Text/VL/Action Meta-LoRA和query delta继续冻结。当前只训练CMBG branch；
不增加external target data、task ID route、expert bank、language-only LoRA、第二套adapter或生成后task-local RL。

## 9. reward与global commitment合同

训练完全复用CAPG/NEAP：task9/15/18各自同一固定paired states产生唯一成功trajectory，四个互不重叠correct K4
conditions在同一endpoint action panel上求梯度；view内等权、task间等权。三个ranks先all-gather完整task gradient，
再形成同一个AdamW candidate，并在`1,1/2,...,1/1024`上找第一个让12/12 inference-path endpoint margins严格下降
的scale。所有ranks接受同一scale和parameter delta；否则逐元素恢复step0。

不改变LR、betas、weight decay、clip、B8 batch、trajectory、views、task weights、seed、dtype或acceptance。结果
仍是zero-interaction Writer LoRA；生成后的task-local RL不进入本轮分数。

## 10. GPU前实现门

CPU/synthetic和一次真实CUDA mechanism必须证明：

1. 原生image/language/50 Action context只forward一次，37 memory tokens逐层读取同层真实context；
2. Action/prefix执行图完全不含memory，memory能看prefix/Action/memory；改变memory值不能改变carrier；
3. layer-memory shape=`[F,18,37,1024]`，每层状态真实变化；
4. step0 second-B、parameter grid和constant-video output逐元素exact zero；
5. payload gate首步gradient非零；人工打开gate后memory/temporal/set/M2P全部获得非零gradient；
6. K1--K4走同一ragged图，K置换不变，video内reverse/shuffle改变grid；
7. source policy零trainable，teacher/target/held信息墙不变；
8. 原50 Action probes、18层Action states、text/frame/grounded/interactions及LPCP first bank在同一原生执行图内
   不受memory反向影响；固定task成功关系与selected credit保持功能锚点，occupancy chunk数只作诊断；
9. 不发生第二次context backbone forward、重复inference、dtype扩展或逐tensor扫描。

### 10.1 首版engineering-invalid world3

clean`38f7fc7`首版完整exit0，但固定task15从预注册`candidate/reference=2/0, chunks=65, pairs=16`漂为
`1/2,47,8`；task9/task18仍为`1/0,25,8`与`1/2,44,8`。因此它不能裁决memory科学假设。仅作诊断时，其same-task
cosine/energy=`.983/.984,.930/.937,.857/.498`，cross-task gradient cosine mean=`+.03865`，raw仍`2/3`，native
best=`11/12`，11 scales无12/12并exact no-op。该轮说明memory values可能改善CAPG的跨task几何，但不能把它写成
CMBG通过或失败。精确artifact为同root的`cmbg_shared3_engineering_adjudication.json`。

### 10.2 Carrier-exact canonical correction and real CUDA evidence

修正版保留同一科学变量与全部下游图，只把扩张joint attention矩阵改为原生carrier加one-way layer observer。
完整CPU=`410 passed`、compileall/diff check通过，architecture guard无hard violation；review项仅为该cohesive
owner既有的大文件/函数长度，没有新模块、并行版本或hard structural问题。
真实task15、demos`49/5/23/45`共130帧的LPCP与CMBG独立编码逐元素比较：text queries、frame evidence、grounded
evidence、interactions和`[130,18,50,1024]`Action states全部max-abs=`0`、relative-L2=`0`。这直接关闭旧检查只证明
“second bank为零”却未比较carrier的漏洞。

真实task9 K4 112帧机制再次通过：layer-memory=`[112,18,37,1024]`，step0 grid exact zero、payload-gate gradient
RMS=`2.22420e-7`且其余branch为0；打开gate后grid L2=`.126103`，memory/temporal/K-set/layer-M2P/token-M2P
gradient RMS=`5.90365e-10/2.68898e-10/1.86177e-10/1.48842e-9/1.58698e-9`，policy gradient tensor数0。
两次forward/backward wall=`121.251s`，peak allocated/reserved=`17,854.6/18,848.0 MiB`，优于首版且0
OOM/nonfinite。

### 10.3 Carrier-exact world3 functional pass

clean`2aecece`在gpu02物理`1/2/3`完成fixed world3：3 tasks、6 paired states、12 rollouts，candidate/reference=
`4/2`，task9/15/18分别=`1/0,2/0,1/2`，selected pairs=`8/16/8`；cycle=`114.732s`。task15实际完整occupancy为
70而历史同条件为65，这属于批形/低位轨迹差异，只作诊断，不改变成功关系或credit内容。

三task同task four-view gradient cosine/energy分别=`.96838/.74782,.85666/.86361,.97881/.97660`；跨task
gradient cosine mean由CAPG的`-.13938`变为`+.09842`，equal-task shared mean与最终Adam delta均覆盖`3/3`，
12/12 inference-path margins严格下降并接受原始Adam candidate`j0`。最终parameter delta L2=`.168481`，q/v/action
effective-BA和fixed-action response三task均非零，0禁读/OOM/nonfinite。精确artifact为run root下
`cmbg_world3_functional_adjudication.json`。因此world3功能门通过，最早待决接口已后移到held task/video泛化。

## 11. world3快速否决门

首个GPU实验固定task9/15/18、world3、一task一rank且已经完成。裁决结果为：

- outcomes：task9=`1/0`、task15=`2/0`、task18=`1/2`；
- occupancy chunks=`25/70/44`仅作诊断；selected pairs=`8/16/8`，共12 rollouts、48 correct videos；
- 三task各自four-view raw descent=`4/4`，cross-video cosine/energy至少`.40/.55`；
- global raw shared-mean descent coverage=`3/3`；
- 11个scale中至少一个真实达到native `12/12`并被所有ranks一致接受；
- accepted delta与预期candidate scale relative error<=`1e-6`，direction cosine>=`.999999`；
- q/v/action native BA与fixed-action response非零；
- payload gate以外的memory/temporal/set/M2P在gate-open follow-up backward中非零；
- 0禁读、OOM、nonfinite，cycle wall<=`318.749s`。

以上功能项已全部通过。validation8 held same-task four-view也已完成：8/8 tasks过门，effective-BA cosine/energy=
`.983541/.985926`，raw-factor/action cosine=`.982788/.986590`，held/train effective-BA L2=`.960650x`。三个train
anchors的reverse相对L2=`1.99829/1.94271/1.97400`，constant/natural norm ratio=`.002687/.003419/.004703`，
信息墙和吞吐均通过。精确artifact=`cmbg_held_video_gate.json`；因此fresh full24 cycle1已获授权。

首次full24从clean`a62348e`启动，24 tasks均已完成rollout/credit，但在global commitment的baseline
重评前暴露旧runtime合同缺口：`_differentiate_credit_views`只在smoke保留4 views，formal只保留
view0。rank0/4因此报`backtracking commitment lost four video views`，其余ranks在同all-gather等待后触发
30分钟NCCL watchdog。run以exit1结束，没有metrics、checkpoint或completion，所以这不是CMBG科学结果。

canonical修正只去掉该mode条件，使formal active task也保留credit backward已算出的4个
`RewardPreferenceView`。它不重算video/environment、不新增forward，不改变data、reward、optimizer、
commitment或CMBG架构。新focused regression在旧代码下会稳定得到`len(views)=1`，修正后为4；
相关测试`42 passed`、全量CPU=`411 passed`。修正后必须fresh重跑，不存在可resume checkpoint。

### 11.1 full24 terminal result

clean`b4dbf84`在gpu02物理`1/2/3/4/7`、world5完成fresh cycle0->1：24 tasks、48 paired states、96
rollouts，candidate/reference=`32/32`、gains=`3/3`，6 active tasks=`4/19/20/25/34/38`覆盖四suite，
cycle=`527.605s`。world5 checkpoint/completion完整，信息墙、OOM和nonfinite全过。

task4/19/20/25/38的same-task four-view cosine/energy约`.988--.992/.956--.994`，证明literal memory对多数
active tasks仍能形成video-coherent local credit；但task34=`-.105/.339`且仅`2/4` views对本task mean下降。
跨task gradient cosine mean/min=`.006807/-.210223`，shared raw descent仅`5/6`。task38 gradient norm=`.010322`，
是次大task4的`54.45x`，与shared mean cosine=`.999781`；task4与shared方向cosine=`-.160805`。Adam
candidate与`-gradient` cosine也只有`.269384`。

全11 scales中最好的`j2, scale=.25`仅让24/24中17个deployed endpoint margins下降；缩小scale没有
形成单调共同下降区。search因此rejected，按authority恢复step0。最终所有parameter delta RMS、q/v/action
effective-BA response和fixed-action response均为0。精确artifact=`cmbg_full24_terminal_adjudication.json`。

因部署policy是exact step0 LPCP carrier，strict400只会重测carrier或捕获设备/批形低位差异，不能表示
CMBG学习，故按`failure_action=restore_step0_parameters_and_terminal_non_pass`跳过。CMBG不resume cycle2、
不补六臂或做参数小扫。

## 12. full24与真实性能裁决

preformal已全过，但full24 global commitment exact no-op使本轮在strict前终局。原继续cycle2的门为
correct>=142、breadth>=7、相对LPCP lost<=15且gains>=losses，本轮无部署改写而不具备进入该门的候选。任何约145 checkpoint立即补same-task-other、
wrong、shuffled、reversed和no-video controls，并评相邻checkpoint稳定性。最终资格仍要求约145+的相邻single
checkpoints、低churn/high Jaccard、高breadth和correct对各negative的明确paired优势；单点高分不够。

每轮报告per-task/per-suite、retained/gained/lost/churn，并与CMBG的直接前身CAPG/TCEC、LPCP143、v6-fast143、
old134/compiler138/online128比较。loss、memory cosine、LoRA norm/rank和内部margin只用于解释。

## 13. 快速否决解释边界

- same-task coherence退回TCEC水平：literal memory没有保住CAPG已验证的跨video坐标作用；
- same-task健康但global raw仍不是3/3：memory values没有形成跨task parameter ownership；
- raw 3/3而native非12/12：共同continuous memory direction仍未形成finite policy-effective step；
- preformal通过但held幅度/方向失败：memory在train conditions内分流，未外推到held language/video；
- held通过而strict换手：最早接口后移到endpoint reward方向与full24多task长期共存。

CMBG负结果只淘汰“37个one-way in-backbone memory values + CAPG zero-gated B-only direct grid + NEAP一轮global
commitment”组合。它不否定memory token一般、旧8-token memory的其它子机制、rank8、完整A/B、few-shot、生成
LoRA或后续task-local RL。

## 14. 工程所有权

- 新cohesive owner只负责native carrier capture、one-way layer memory和layer states；不得把旧Dynamic-K训练器、rank8 mapper或旧design
  恢复成平行runtime；
- `parameter_grid.py`保留temporal/set/M2P/direct reshape，但用layer memory替换context readout；旧CAPG
  post-backbone query owner应删除而非保留fallback；
- `CompleteLoRAWriter`外部ragged input和76-tensor public LoRA API不变；
- frozen implementation必须来自clean pushed commit的detached worktree；正式产物保留contract、metrics、completion
  和terminal/continuation adjudication。

## 15. 历史继承而非重复

CMBG继承：V6/AS139的高absolute Semantic Core与Procedure、LPCP的18层视频carrier、NZRB的native-zero rank bank、
NEAP的deployed endpoint credit、CAPG的capacity-matched direct grid和其新证明的same-task cross-video coherence。
它只从旧Dynamic-K恢复已经单独验证的真实prefix、one-way memory mask和逐层状态机制。

它不是旧100分Dynamic-K重跑：不删除V6 absolute Core，不用8-token shared family mapper，不改fresh rank8，不改
dynamic-K supervised recipe，也不丢LPCP first bank。旧Dynamic-K低分不能预判本轮；反过来，本轮失败也不能把
旧路线所有结论改写成“memory无效”。

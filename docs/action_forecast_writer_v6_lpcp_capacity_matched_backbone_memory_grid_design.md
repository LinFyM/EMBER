# V6-LPCP Capacity-Matched Backbone-Memory Grid

状态：2026-08-16 active single-variable successor authority。简称**CMBG**。本轮只能实现和检验本文这一条canonical
路径；CAPG已经终局，不得resume、放宽其门或用normalization/PCGrad/task权重/scale小扫补救。历史8-memory
Dynamic-K只可选择性复用其已验证的three-block mask和joint-layer语义，不得整条恢复。

## 1. 决策

CMBG保留V6-LPCP、NEAP endpoint preference、K4 four-view credit、rank32 native-zero residual bank、task9/15/18
world3共同裁决，以及CAPG已经证明有效的video内有向聚合、K-set、layer/token M2P和direct B reshape。唯一主要
架构变量是把CAPG的

```text
已有50个Action states -> backbone之后的37个query latents
```

替换为

```text
真实image/language prefix + 50个Action probes + 37个one-way memory tokens
    -> 同一次PI0.5 joint 18-layer forward
    -> 每层37个contextual memory states
```

这37个是真正进入Action Expert计算并逐层更新的memory tokens，不是视频结束后的latent slots。每个condition仍只
生成一套完整38-target LoRA；Writer仍在rollout前运行一次。

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
  -> 追加37个parameter-aligned memory tokens
  -> 同一次18层PaliGemma/Action-Expert joint forward
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

对每个真实frame，正常prefix和suffix为：

```text
prefix = [256 real image tokens, exact language tokens]
suffix = [50 native fixed Action probes, 37 learned memory tokens]
```

使用三个attention blocks：

1. prefix queries只看有效prefix；
2. Action queries看prefix和完整50-token Action block，但不能看后置memory；
3. memory queries看prefix、Action block和完整memory block。

所以memory处于有意义的真实图文/Action context中，不是blank image、memory-only forward或凭空Action query；同时
memory不能反向改变原50个Action probes的可见集合。memory使用与固定`t=1` probes相同的AdaRMS condition，经过
每层attention residual、post-attention norm和MLP residual后保留状态：

```text
Z_frame in R[18,37,1024]
```

实现只能运行一次joint layer loop；不得另算no-video baseline、重复原backbone或修改site-package。优先复用旧
`100095d/f1e2fcd`中经过测试的three-block mask与joint-loop公式，但只恢复这个窄机制，并嵌入当前V6 encoder。原
Action/prefix输出对memory值应结构不变；正常BF16 kernel低位差异用public LPCP effective-BA、fixed outcome和
action-response实测裁决，不为逐元素相等退化到batch1、FP64或第二次forward。

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

1. 37 memory tokens与真实image/language/50 Action probes处于同一次joint forward；
2. three-block mask中Action不能看memory，memory能看prefix/Action/memory；改变memory值不改变Action/prefix的
   可见集合；
3. layer-memory shape=`[F,18,37,1024]`，每层状态真实变化；
4. step0 second-B、parameter grid和constant-video output逐元素exact zero；
5. payload gate首步gradient非零；人工打开gate后memory/temporal/set/M2P全部获得非零gradient；
6. K1--K4走同一ragged图，K置换不变，video内reverse/shuffle改变grid；
7. source policy零trainable，teacher/target/held信息墙不变；
8. 原50 Action probes、LPCP first bank和正常prefix不因memory内容改变；public step0 effective-BA与fixed action保持
   正常BF16容差且固定task outcomes不漂；
9. 不发生第二次video backbone forward、重复inference、dtype扩展或逐tensor扫描。

## 11. world3快速否决门

首个GPU实验仍固定task9/15/18、world3、一task一rank，不先跑单task挑好结果。必须复现：

- outcomes：task9=`1/0`、task15=`2/0`、task18=`1/2`；
- occupancy chunks=`25/65/44`，selected pairs=`8/16/8`，共12 rollouts、48 correct videos；
- 三task各自four-view raw descent=`4/4`，cross-video cosine/energy至少`.40/.55`；
- global raw shared-mean descent coverage=`3/3`；
- 11个scale中至少一个真实达到native `12/12`并被所有ranks一致接受；
- accepted delta与预期candidate scale relative error<=`1e-6`，direction cosine>=`.999999`；
- q/v/action native BA与fixed-action response非零；
- payload gate以外的memory/temporal/set/M2P在gate-open follow-up backward中非零；
- 0禁读、OOM、nonfinite，cycle wall<=`318.749s`。

任一项失败即终局，不跑validation8、full24、strict、rank/token/LR/scale/seed sweep。通过后才做validation8 held
same-task four-view、held/train amplitude、reverse/constant门；仍沿CAPG阈值要求至少6/8 tasks通过和held/train
effective-BA L2>=`.30`。全部通过才允许fresh full24 cycle1。

## 12. full24与真实性能裁决

若preformal全过，full24仍只训练一个shared checkpoint。cycle1后立即strict paired correct400；继续cycle2的门
保持correct>=142、breadth>=7、相对LPCP lost<=15且gains>=losses。任何约145 checkpoint立即补same-task-other、
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

- 新cohesive owner只负责three-block joint memory和layer states；不得把旧Dynamic-K训练器、rank8 mapper或旧design
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

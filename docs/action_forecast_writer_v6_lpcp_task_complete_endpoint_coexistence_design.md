# V6-LPCP Task-Complete Endpoint Coexistence

状态：2026-08-16 implementation-ready / preformal，简称`TCEC`。本轮从sealed LPCP fresh启动，完整保留NEAP-C已经验证的
10步deployment endpoint credit与NZRB native-zero rank32 LoRA。唯一变量是把reward update的承诺单位从一个
task-local probe改成**同一shared Writer update覆盖的全部active tasks**；task gradients仍等权，不增加
parameter-ray solver、task router、memory、rank、scale或第二套adapter。

## 1. 新证据与最早接口

NEAP-C task9已经关闭此前最早断点：相对NZRB的随机CFM，10步endpoint使四view gradient cosine/energy从
`.286/.448`升至`.846/.865`，并让原始Adam candidate从11点no-op变成`j0`一次接受；cycle与显存也显著下降。
held8方向、raw-B、action、reverse/constant和rank-bank全部健康。

唯一失败是held/train BA L2=`.234<.30`。stage localization为：

```text
held/train probe Value       = .671
held/train joint Value       = .665
held/train direct B rows     = .223
held/train effective BA      = .234
```

因此视频、语言、顺序、K-set与native compiler没有先丢幅度；一个zero-init linear B-head只接受task9的一次outer-product
式reward update，held conditions只能获得其在task9 condition方向上的投影。真正目标本来就是task-complete shared
Writer，而不是让一条task-local update独自覆盖整个task manifold。下一最小反事实应让多个task endpoint credits在
同一次shared commitment中共同写head，并直接要求它们全部受益。

## 2. 保持不变的输入、Writer与LoRA

```text
exact language + four ordered action-hidden correct videos
 -> one LPCP joint context forward
 -> 18-layer Action probes
 -> causal per-video Procedure
 -> permutation-invariant K-set aggregation
 -> video-required joint Value
 -> four shared B-only heads
 -> one rank32 native-zero residual LoRA
 -> frozen pi0.5-LIBERO policy
```

每个condition仍只生成一套完整38-target adapter：`A=[A0;A0]`、`B=[B0,delta-B]`，原LPCP rank16 carrier逐元素
保留，second-B从zero写入。frame stride5、K4、四个disjoint same-task correct views、MB-SOP同B8 action panel、
10步endpoint距离、AdamW、`j=0..10` native backtracking与信息墙全部不变。

本轮不加memory token：NEAP已经证明carrier、顺序与joint Value在held仍有约`.665x`幅度，最早断点在shared head
如何累积task coverage。若task-complete endpoint commitment仍不能共存，capacity-matched memory/M2P才成为针对
condition-to-LoRA共享映射的下一候选；现在同时更换会失去可归因性。

## 3. Task-complete endpoint update

对每个有discordant reward pair的active task `t`，沿用NEAP的四个correct K4 views：

```text
g_t,v = gradient_theta softplus(D_w(t,v) - D_l(t,v))
g_t   = (1/4) sum_v g_t,v
g     = (1/|A|) sum_{t in active tasks A} g_t
```

不按trajectory数、suite、成功臂或gradient norm重权；没有PCGrad、MMCD、simplex、ray mixture或task-specific LR。
AdamW由`g`产生唯一candidate `delta_adam`。

对`j=0..10`，所有ranks写入同一个`2^-j delta_adam`，并用各自保留的全部active-task probes在相同observation、
noise与physical B8下重新计算四view endpoint margins。小型结果行通过collective聚合成全局有序集合：

```text
accept j iff every active task t and every correct view v has
margin_t,v(2^-j delta_adam) < margin_t,v(step0)
```

所有ranks必须选择同一个首个accepted `j`，否则exact恢复step0。parameter tensors不all-gather，不平均checkpoint，
只collect small scalar evidence；最终仍是一套shared Writer state。该规则既防止各rank因只看本地最后一个task而写出
不同checkpoint，也把“多task共同积累”变成结构合同而非训练后描述。

## 4. 固定三anchor shared gate

formal前不再做三次独立task-local更新，而是在同一节点world3 fresh运行一次共享step：

- rank0/1/2分别固定task9/15/18；task权重各`1/3`；
- paired outcomes必须复现`1/0`、`2/0`、`1/2`；
- complete chunks=`25/65/44`，selected pairs=`8/16/8`；合计6 paired states、12 rollouts、134 complete chunks、
  32 selected pairs、12 credit conditions、48 unique correct videos；
- task9/18每view一个physical B8 endpoint batch，task15每view两个B8 batches；不得降batch、offload或扩dtype；
- 三个task gradients对等权mean的continuous descent coverage必须3/3；四个B heads均有finite nonzero credit；
- global backtracking必须在`j<=10`找到12/12 margins严格下降的同一candidate，三个ranks的accepted scale与参数
  delta必须一致；否则终局，不换solver或task weights；
- accepted update后，三个train anchors各自four-view BA cosine/energy至少`.40/.55`；validation8至少6/8过
  `.15/.40`，aggregate至少`.30/.48`，raw-B cosine至少`.30`，action cosine至少`.15`；
- held mean BA L2 / 三个train anchors mean BA L2至少`.30x`，直接检验多task coverage是否修复NEAP的`.234x`；
- 三anchor natural/reversed BA relative-L2均至少`.50`，constant/natural均不超过`.005`；
- world3 cycle wall不超过NZRB三anchor最大单task wall`528.274s`的`1.25x=660.343s`，每rank峰值低于A40物理
  容量，0 OOM/nonfinite/禁读。

该gate不是用三个task挑checkpoint：三者在历史路线开始前已固定，且本轮只产生一个共同参数delta。任一项失败即
终局，不拆成task-local winners、不保留部分rank state、不补其它tasks。

## 5. Full24与稳定裁决

三anchor全过才允许fresh full24 cycle1。每rank保留其所有active task probes；每个candidate scale由全局所有
active task×4 views共同裁决，最终checkpoint各rank逐元素同一。cycle1后立即K4 strict paired400；至少correct
`>=142`、breadth`>=7`、相对LPCP lost`<=15`且gained不少于lost才允许cycle2。

最终资格仍要求相邻single checkpoints correct均至少142、均值约145或更高、churn`<=20`、Jaccard`>=.85`、
final lost`<=10`。首次达到约145且retention过门，立即补same-task-other、wrong、shuffled、reversed、no-video；
correct必须沿有用policy direction获益。

## 6. 快速解释边界

- 三anchor equal-mean在continuous处已有task下降冲突：endpoint metric解决same-task video credit，但task gradients
  本身不能共同累积；下一步才考虑显式task-space representation/coordination，不做ray solver；
- continuous 3/3但12/12没有finite native step：shared direct-head的finite task coexistence失败；memory/M2P或
  task-structured output成为合理候选；
- 12/12通过但held/train仍低于`.30`：增加task diversity仍不能覆盖unseen condition manifold，下一接口是
  condition-to-LoRA共享映射，届时capacity-matched memory token有直接动机；
- 三anchor与held全过但strict换手：最早缺口后移到train24 endpoint panels对真实held rollout occupancy的覆盖；
- 高correct但视频controls失败：学到reward/language shortcut，不构成有效视频教学。

负结果只淘汰`LPCP + NZRB + NEAP endpoint credit + equal-task-mean + globally synchronized task-complete native
commitment`。不否定endpoint preference、memory token、rank8、Dynamic-K、few-shot、LoRA生成或未来task-local RL。

## 7. Canonical implementation evidence

canonical runtime已经原位替换NEAP active schema，没有保留平行runner：

- config=`configs/pi05_writer_v6_lpcp_task_complete_endpoint_coexistence_v1.json`；smoke只接受world3与
  `--smoke-task-ids 9,15,18`，local rank按固定顺序各拥有一个anchor；
- 每个active task均保留自己的四view probe；gradient tensor仍只走原有all-reduce，backtracking每次只
  `all_gather_object`小型task/view scalar rows；全局按`(task_id, view_index)`排序并验证完整4-view ownership；
- step0 baseline与每个candidate都在同一inference evaluator中全局重算。只要一个task的一个view不下降，所有ranks
  同时拒绝该scale；CPU合同专门覆盖了“task0下降、task1上升时全局拒绝，下一scale才共同接受”；
- accepted rows按task还原到各自deployment response probe；smoke和formal task records都按固定task集合全局汇总；
- 定向回归=`41 passed`，完整CPU=`405 passed`，compileall与diff check通过；architecture guard=`0 hard
  violations`，没有新增module/entrypoint，active source净`+141`行。以上只关闭实现门，不提供GPU机制结果。

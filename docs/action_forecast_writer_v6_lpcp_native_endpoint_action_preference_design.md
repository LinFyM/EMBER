# V6-LPCP Native Endpoint Action-Preference Credit

状态：2026-08-16 terminal preformal non-pass，canonical task9已完成并按held/train幅度门终局，
简称`NEAP-C`。本轮从sealed LPCP fresh启动，完整保留
LPCP视频carrier、MB-SOP同状态matched action panel、四个disjoint correct K4 views、NZRB-C的一套rank32
native-zero residual LoRA与PAV first-all-view native acceptance。唯一变量是：reward credit不再优化随机flow
time上的CFM surrogate，而是直接优化冻结source policy经过完整10步部署求解器后产生的action endpoint。

## 1. 为什么下一变量在reward functional metric

历史已经逐层关闭了更早接口：

- LPCP143证明exact language、ordered videos、Action probes、causal Procedure与K-set carrier能得到当前最强底座；
- MB-SOP在同一B8 observation/noise panel重查两臂，去除了stored action与counterfactual batch-shape混杂；
- raw mean、MMCD与Adam-preconditioned三类parameter rays均不能跨task稳定形成native finite step，不能再换solver；
- ALB-NV删除A/B gauge与cross term后救活部分任务；NZRB-C进一步令accepted residual从native zero写入，task15/18
  的held BA/raw-B均约`.93--.95`且8/8通过，说明LoRA写出已对可接受方向有效；
- task9仍在11个native points上没有all-view step。继续改bank、rank、factor origin、scale或混合旧rays不针对现存断点。

当前CFM preference只在随机单一flow time比较velocity regression loss。最终部署却从同一noise连续运行10个
denoise steps，再执行endpoint action。一个方向可以降低CFM surrogate，却不必在真实action endpoint上把policy
移向成功arm；这正好可以解释“continuous gradient存在，但finite native acceptance为空”。本轮只检验这个解释。

## 2. 保持不变的数据流与部署图

输入仍为exact task language与四条同task、action-hidden、内部有序teacher videos，frame stride5：

```text
language + four ordered correct videos
 -> one LPCP joint context forward
 -> 18-layer Action probes
 -> causal per-video Procedure
 -> permutation-invariant K-set aggregation
 -> video-required joint Value
 -> four B-only residual heads
 -> one rank32 native-zero residual LoRA
 -> frozen pi0.5-LIBERO policy
```

public LoRA仍为NZRB-C：

```text
A_public(c) = concat_rows(A0(c), A0(c))
B_public(c) = concat_columns(B0(c), delta-B(c))
B_public A_public = B0 A0 + delta-B A0
```

原rank16 LPCP carrier逐元素保留，second-B从zero写入；每个condition仍只有一套完整38-target adapter。rank32只是
保留强carrier并无损承载residual的当前表示，不是本轮变量。memory token、rank8与Dynamic-K仍开放，但当前证据没有
把最早断点指向carrier或capacity，因此本轮不同时改变。

## 3. Native endpoint action preference

paired AS139-reference/LPCP-candidate rollouts、成功/失败label、成功occupancy、8个等进度strata及每stratum最大
matched action disagreement全部保持。对每个被选共同observation `o_s`，两臂仍以相同physical B8、相同policy
noise `epsilon_s`重查，得到reward标记的winner/loser action chunks `a_w,a_l`。

对一个video condition `c`，Writer生成native LoRA后，直接通过冻结policy完整10步部署求解器得到：

```text
a_theta(c,o_s,epsilon_s) = PI05_10_step_endpoint(o_s, LoRA_theta(c), epsilon_s)
D_w = mean_valid ||a_theta - a_w||^2
D_l = mean_valid ||a_theta - a_l||^2
J_s = softplus(D_w - D_l)
```

只比较真实会执行的前1--5步。每条成功trajectory内部strata等权，1--2条discordant trajectories等权，四个correct
K4 views等权，active tasks等权。梯度穿过完整10步Euler求解和native public LoRA到四个residual B heads；source
policy、LPCP与LoRA carrier全部冻结。

这不是teacher-action supervision：target action来自train24中同初态、同噪声的两条on-policy arms，只有最终成功
结果决定哪一臂是winner；teacher videos始终action-hidden。部署时不需要reward、arm或action target。

## 4. 为什么它可能形成跨video共同Value

四个video views面对的是同一组observation、noise和reward-labeled action contrasts。旧CFM对每个view通过局部
velocity regression间接指向policy；NEAP直接要求四个condition生成的LoRA在同一部署action空间完成同向移动。
因此共同对象不是raw parameter direction或漂亮LoRA geometry，而是：

```text
same successful action contrast
 <- actual 10-step policy endpoint
 <- condition-specific native LoRA
 <- shared Writer parameters
```

如果同task视频确实编码同一高层程序，不同video condition的梯度应能在endpoint action上共同下降；如果它们仍只
编码video-local nuisance，task9会继续没有all-view finite step，设计可被快速否决。

## 5. 视频与语言边界

- exact language继续通过LPCP说明目标对象和关系；
- ordered video dynamics继续是Procedure与residual Value的必要路径；constant/no-video不能由language单独写residual；
- 每条video内部的Action-probe sequence保持因果顺序，reverse/shuffle必须重新完整forward；
- 四个videos只在既有K-set阶段置换不变聚合，不平均frames、raw features或LoRAs；
- 四个disjoint K4 conditions只在Writer gradient处等权，不能挑最容易下降的video；
- reward objective变化不构成部署时交互，Writer仍在rollout前运行一次。

## 6. 实现与吞吐合同

- target arms仍只做现有一次matched B8 query；不重复environment rollout或teacher-video backbone forward；
- endpoint prediction对每个selected observation只生成一条action，再同时与winner/loser targets比较；不分别运行两次；
- 10个denoise steps共享一次视觉/语言prefix cache；suffix按需activation checkpoint，不保存十份完整图；
- logical与physical preference batch保持B8。若B8 endpoint backward在A40上不能稳定运行，则本设计按效率门终局，
  不降到batch1、扩dtype、CPU offload或关闭高效kernel挽救；
- 正常BF16/TF32、kernel与reduction低位差异接受；不增加hash、逐tensor扫描或重复forward；
- trainable仍为四个B heads共860,160参数，optimizer、LR、betas、eps、weight decay与clip不变。

## 7. Native acceptance

AdamW仍由四view等权、task等权的endpoint gradient产生候选。candidate写入native second-B后，使用完全相同的
10步endpoint action preference evaluator先测step0 baseline，再按`1,1/2,...,1/1024`检验。只接受四个correct
video views的`D_w-D_l`全部严格下降的第一个candidate；否则exact恢复step0。

不使用CFM margin选step，不混合raw/MMCD/Adam rays，不求finite response simplex，不按task挑方向。endpoint objective
同时拥有gradient与acceptance，避免再次用一个surrogate产生方向、另一个部署量裁决。

## 8. 固定anchor与快速否决

按效率顺序先只跑NZRB唯一未解的task9；它通过后才跑task15/18。三项不是挑task，而是固定的失败/成功覆盖：

1. task9 pre-update必须复现rank32 NZRB outcome=`candidate1/reference0`、complete chunks=`25`、selected pairs=`8`；
2. physical B8完整10步endpoint forward/backward finite，peak reserved小于A40物理容量，0 OOM/禁读；
3. 四view endpoint gradients对等权mean的descent coverage必须4/4；四个residual B heads均有非零finite gradient；
4. task9必须在`j<=10`找到all-view endpoint-margin下降candidate，q/v/action BA与fixed-action response均非零；
5. task9 train four-view BA cosine/energy至少`.40/.55`；validation8 aggregate至少`.30/.48`、至少6/8过
   `.15/.40`、action cosine至少`.15`、held/train BA L2至少`.30x`；
6. natural/reversed relative-L2至少`.50`、constant/natural不超过`.005`；
7. task9 cycle wall不超过NZRB task9 `362.177s`的`1.75x`。超过即视为不能扩展到full24，不用batch1或精度技巧救；
8. task9全部通过后，task15/18必须复现NZRB pre-update outcomes/counts并通过同一native、held、temporal门；三项总
   wall不超过NZRB `1111.763s`的`1.75x`。

task9任一项失败即终局，不运行task15/18，不换distance、temperature、MC noise、backtrack、batch、LR或rank。
三anchor全过才实现distributed formal acceptance与evaluation schema。

## 9. Full24与strict裁决

若三anchor全过，fresh full24 cycle1后立即做K4 strict paired400。cycle1至少需要correct`>=142`、breadth`>=7`、
相对LPCP lost`<=15`且gained不少于lost，才允许cycle2。最终稳定资格仍要求相邻single checkpoints都至少142、均值
约145或更高、churn`<=20`、Jaccard`>=.85`、final lost`<=10`。

首次达到约145且retention过门，立即补same-task-other、wrong、shuffled、reversed、no-video；correct必须沿有用
policy direction受益，不能只把negative LoRA破坏。最终仍只认single-checkpoint strict paired400，endpoint margin、
gradient cosine、BA coherence和action distance都只是诊断。

## 10. 结果解释边界

- task9 endpoint gradient仍无finite all-view step：CFM surrogate不是主要断点，应转向真实环境分支credit或更直接的
  Writer-level policy optimization，不再做parameter-ray solver；
- task9通过但task15/18失败：endpoint metric对单task有效但未形成shared task-stable Value；
- 三anchor过而full24换手：最早缺口后移到多task shared update coexistence；
- correct高但video controls失败：endpoint reward学到了shortcut，不构成视频教学；
- B8 OOM或wall超门：完整endpoint backprop不可扩展，科学假设未被性能验证，但当前实现路线因效率终止。

负结果只淘汰`LPCP + NZRB rank32 + MB-SOP panel + full 10-step endpoint action preference + one reward cycle`。不否定
memory token、rank8、few-shot、LoRA生成或未来task-local RL。

## 11. Canonical implementation state

NEAP-C已原位替换NZRB-C的active objective/schema/config，未保留平行runner或兼容fallback。部署图、rank32
native-zero bank与860,160个B-head参数不变；新增实现只负责一次functional prefix、10个checkpointed denoise
steps、executed-prefix endpoint距离及同metric inference acceptance。CPU compileall通过，定向机制合同
`50 passed`、完整回归`405 passed`，architecture guard为0 hard violation且active source净`-16`行。

这些在launch前只证明实现与信息墙合同接通，不能用CPU结果预判显存、吞吐、finite candidate、held或temporal
证据；随后完成的task9真实裁决见下一节。

## 12. Task9 result and terminal boundary

clean pushed `33f69fd`在gpu02物理1完成task9：candidate/reference=`1/0`、complete/selected=`25/8`，physical
B8完整10步endpoint的四view forward/backward全部finite；cycle=`97.107s`，max reserved=`19.367GB`。相对同
outcome NZRB task9，gradient cosine/energy从`.286028/.448222`提高到`.846183/.865229`，并从11个candidate
全部拒绝变为原始Adam candidate `j0`一次接受；q/v/action BA与fixed-action均非零。故endpoint metric确实关闭了
CFM到真实部署action的主要错位，而且更快、更省显存。

held/temporal稳定FP64分析中26/27门通过：validation8全部过`.15/.40`，BA cosine/energy=`.952537/.932938`，
raw-B cosine=`.954789`，action cosine=`.485498`；reverse effective-BA relative-L2=`1.218588`，constant/natural=
`.00015964`；rank-bank五项误差均精确0。唯一失败是held/train BA L2=`.234042<.30`，因此按第8节不得运行
task15/18、full24或strict，也不放宽门、扫scale或补normalization。

stage localization显示held/train probe与joint Value幅度仍为`.671059/.664719`，但direct factor rows骤降到
`.222730`，BA随后为`.234042`。最早失败接口是**task9 joint condition -> one-task shared direct-B-head update
的跨task幅度**；不是carrier、视频顺序、endpoint differentiation、native-zero bank或compiler。精确artifact为
`neap_c_mechanism_gate.json`与`neap_c_terminal_adjudication.json`。

# V6-LPCP Direct-Factor Paired Common-State Preference

状态：2026-08-16 active mechanism-ready authority。简称 `DF-PCSP`。本轮从sealed LPCP fresh启动，完整保留
DJNFR的视频carrier与direct native-factor生成图，只替换reward credit；canonical实现与CPU门已通过，但尚未
授权full24、strict或cycle2。

## 1. 决策与最早失效接口

DJNFR已经关闭了前一轮最早断点。task4到validation8的joint/direct/raw-factor/effective-BA共同方向完整，full24
后held8 BA cosine/energy仍为`.790242/.785834`且8/8过门，说明有序视频与语言联合Value能稳定写成native
public LoRA。可是cycle1 strict只有`136/400`；相对LPCP143为`120 retained / 16 gained / 23 lost`，四suite中
三项下降。all400中retained-failure获得最大改写，lost改写也大于gained。最早缺口已后移到：

```text
sparse paired binary success
 -> selected-success-only trajectory imitation
 -> one shared direct-factor update
 -> held on-policy reward-useful direction
```

当前credit只降低成功轨迹的CFM loss，没有告诉Writer同一初态下哪种失败动作应被排斥；而成功轨迹后续
observations又已经与失败臂分叉，不能作无混杂的逐步对照。

本轮唯一主要因果变量是：**对每个candidate/reference discordant pair，只在两臂严格共享的初始观测处，用最终
成功臂与失败臂的第一段已执行动作构造有正有负的paired flow preference。**

## 2. 完整保留项

- exact task language + dynamic `K=1..4` action-hidden ordered videos；formal固定K4；
- stride5、每video内部因果编码、videos间置换不变set aggregation；
- AS139/LPCP底座、同一次真实图像+语言+50 Action probes forward、18层LPCP carrier；
- 320个layer/rank slots、`X=M*RMSNorm(L)/sqrt(256)`与DJNFR八个direct factor heads；
- 完整38-target rank16 public LoRA，step0/no-video/constant路径exact LPCP；
- 两个paired states、AS139 reference与LPCP+direct-head candidate两臂、四个disjoint correct K4 credit views；
- train24 task等权、AdamW、Nmc4、BF16/TF32、source policy、split、rollout数与信息墙；
- rollout前一次生成一套LoRA，部署时无reward、reference、selector、expert、LoRA平均或checkpoint融合。

本轮不加memory token，不改rank、scale、head width、carrier、K、optimizer、rollout数或source-action loss。DJNFR
cycle1 checkpoint不续训；从sealed LPCP重新fresh，使step0和optimizer state精确可解释。

## 3. Paired common-state objective

对task的两个paired states分别以完全相同的env seed、reset、dummy settling与policy-noise seed运行reference和
candidate。若最终binary success相同，该pair credit严格为0。若只有一臂成功，记成功臂为`+`、失败臂为`-`。

两臂只有第一次policy query的观测尚未受各自动作影响，因此只取：

```text
o_i^0                       # 两臂逐tensor相同的初始policy observation
a_i^+, a_i^-                # 成功/失败臂第一次生成的normalized action chunk
h_i = min(h_i^+, h_i^-)     # 两臂共同实际执行的1--5步前缀长度
```

不使用第二次及以后replan：那些observations属于不同occupancy，不能把失败轨迹的后来状态当作成功轨迹的负例。
对每个MC样本，winner和loser严格共享同一flow time与Gaussian noise。对任一correct K4 view生成的candidate LoRA
`lambda_v`：

```text
ell_i^+(v,m) = CFM(a_i^+[:h_i] | o_i^0, lambda_v, t_im, epsilon_im)
ell_i^-(v,m) = CFM(a_i^-[:h_i] | o_i^0, lambda_v, t_im, epsilon_im)

J_i(v) = mean_m softplus(ell_i^+(v,m) - ell_i^-(v,m))
```

`softplus`提供单调、bounded-slope的pairwise update：下降同时要求成功动作相对失败动作更受当前LoRA支持，而不靠
无限线性负loss。一个task有1或2个discordant pairs时先等权平均pair；四个disjoint K4 views分别求exact Writer
gradient后等权平均；最后只对active tasks等权。不同轨迹长度、task horizon或GPU ownership都不能改变权重。

## 4. 为什么它比selected-success-only更有辨识性

- winner与loser来自同task、同初始化、同reset、同policy RNG，只差AS139/LPCP条件造成的动作与后续结果；
- 比较发生在动作之前的共同观测，不混入两臂后来不同的state occupancy；
- correct videos仍是四个candidate LoRA的唯一dynamic Value，winner/loser动作本身不能按task ID路由参数；
- 同一pair在四个互斥same-task K4 conditions中给出相同偏好，直接要求不同正确视频形成可复用的高层更新；
- reference只在train24产生一个受控反事实，部署时不存在第二臂或在线交互。

它不同于历史OPPP。OPPP在同task四个不同随机rollouts上用LOO advantage比较整条executed prefixes；success与
failure的observations并不相同，strict138且lost19。本轮使用同一初态的paired AS139/LPCP反事实，只监督分叉前
唯一共同观测的首段动作，因此只检验此前没有被OPPP检验的“common-state causal preference”。

## 5. 视频、语言与LoRA生成仍不可绕过

数据流完全保持：

```text
exact language + ordered action-hidden videos
 -> one native context forward
 -> layerwise Action-probe temporal deltas
 -> per-video ordered Programs
 -> permutation-invariant K-set Value M
 -> joint payload X=M*RMSNorm(L)/sqrt(256)
 -> eight direct factor-shape heads
 -> one complete rank16 LoRA
```

language决定对象、关系和目标，但`M=0 -> X=0`，不能独立写新增LoRA；reverse/shuffle改变有向transition后再改变
LoRA。preference只决定八个shared heads应沿什么policy方向更新，不增加language-only、reward-only或task-ID路径。

## 6. 多task共存假设

DJNFR active-task parameter gradients pairwise cosine mean约`.00247`，说明positive imitation为不同tasks提供近正交
cotangents；尽管shared mean对每个train row都是局部下降方向，held closed-loop仍换手。common-state preference
删除了每条长轨迹中task-specific occupancy的大量cotangent，只保留“在相同状态成功动作应优于失败动作”的局部
判别方向。假设是这种因果对照比全成功轨迹模仿更容易在同一shared map中共存，而不是参数量本身解决冲突。

该假设必须由full24 task-gradient、retained/gained/lost与相邻checkpoint裁决；内部gradient cosine不能选择方法。

## 7. 实现边界

- `reward_cycle.py`保留同一paired rollout owner；只把discordant pairs整理成共同初态winner/loser首段batch；
- 每pair运行时确认两臂初始observation、pair identity与policy seed相同，`h_i`取共同执行长度；
- `reward_preference.py`原位替换selected-success cotangent，winner/loser按pair相邻排列并共享flow panel；
- 四view仍只重解小Writer/compiler，不重复video backbone，不增加rollout；
- checkpoint/config/evaluator schema fresh-incompatible，旧DJNFR artifacts由Git与formal roots保存；
- canonical源代码只保留一种active reward objective，不增加strategy switch或兼容fallback。

实现已原位完成：paired rollout只保留winner/loser共同初态首段，flow panel按pair共享且physical microbatch不拆
pair；smoke-only probe会在真实Adam step后复算同一panel margin。config/checkpoint/evaluator均fresh-incompatible。
定向CPU=`44 passed`、完整CPU=`398 passed`、compileall通过；architecture guard无hard violation，reward cycle
owner由790行缩至698行且没有新增active并行模块。以上只关闭工程门，不替代GPU机制或closed-loop证据。

## 8. formal前机制门

先在历史确定会产生一个discordant pair的train task4做B8真实smoke：

1. reference/candidate两臂pair identity、初始observation与policy RNG严格相同，winner/loser首段动作确实不同；
2. tie不进入CFM；active pair只使用首个replan与共同执行长度，四view使用同一pair/time/noise；
3. pairwise objective、LoRA cotangent和Writer gradient finite/nonzero，真实optimizer step后同panel
   `ell_winner-ell_loser`下降；
4. 八个direct heads全部更新，q/v/action raw factor、effective BA和fixed-action response非零；
5. task4 four-view BA cosine/energy至少`.40/.55`；validation8 aggregate至少`.30/.48`且至少6/8 tasks过
   `.15/.40`；held raw factor cosine至少`.30`、action cosine至少`.15`、held/train BA L2至少`.30x`；
6. natural/reversed BA relative-L2至少`.50`，constant/natural不超过`.005`；
7. 0 forbidden read/OOM/nonfinite，不增加rollout或backbone forward；cycle wall不超过DJNFR`1.10x`。

任一共同观测、preference descent、q/v/action、held或效率门失败即终局，不full24，不扫margin、temperature、LR、
Nmc、batch、rank或scale。

## 9. full24与closed-loop裁决

机制门全过后，才从sealed LPCP fresh做一次full24 cycle1并立即K4 strict paired400。只有同时满足：

- correct至少`142/400`、breadth至少7；
- 相对LPCP143 lost不超过15、gained不少于lost、无suite清空；
- post-train task4/held跨视频共同方向不坍塌；
- common-state margin与deployment q/v/action均保持非零；

才允许exact-resume cycle2。稳定资格仍要求两个相邻single checkpoints均至少142、均值至少145、churn不超过20、
Jaccard至少`.85`、final相对LPCP lost不超过10且gained不少于lost。首次约145且retention过门立即补
same-task-other/wrong/shuffled/reversed/no-video。

## 10. 快速否决与负结果边界

- preference margin不下降：pairwise LoRA cotangent到Writer接口失败；
- margin下降但held BA不共同：新credit破坏已通过的生成端；
- margin、held BA均健康但strict仍换手：common-state CFM preference仍不是held reward-useful surrogate；
- train task4健康但full24后旋转：进入显式shared-update/coexistence接口，不回头调memory/rank/head scale；
- absolute过门但correct不优于视频controls：教学视频claim仍失败。

负结果只淘汰“LPCP/DJNFR生成图 + paired AS139/LPCP最终success标签 + 分叉前共同初态首段flow preference +
four-view one-cycle shared update”组合；不否定memory token、rank8、few-shot、生成LoRA、其它reward credit或未来
生成LoRA后的task-local RL。

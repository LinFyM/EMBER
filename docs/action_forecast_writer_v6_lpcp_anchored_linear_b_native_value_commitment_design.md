# V6-LPCP Anchored Linear-B Native Value Commitment

状态：2026-08-16 active single-variable design authority，简称`ALB-NV`。本轮从sealed LPCP fresh启动，完整保留
MB-SOP matched successful-occupancy credit、四个disjoint correct K4 views、PAV同路径native acceptance和rank16
部署图，唯一把八个joint A/B residual heads改成四个fixed-A、B-only residual heads。

## 1. 决策与最早失败接口

PAV-BC在固定task9/15/18上0/3过门。raw equal-mean、raw maximum-margin与Adam-preconditioned三类parameter ray
都已测试，继续换gradient权重或scale没有新信息。共同断点是：

```text
shared video-conditioned Value
 -> simultaneous delta-A and delta-B
 -> native factor quantization plus bilinear BA cross term
 -> held policy-effective amplitude
```

当前direct residual同时写两侧：

```text
A = A0 + delta-A
B = B0 + delta-B
BA = B0 A0 + B0 delta-A + delta-B A0 + delta-B delta-A
```

同一个reward cotangent因此被拆到两种不同gauge、两组不同native resolution及一个二阶cross term中。连续
parameter-space common descent不保证四项相加后仍是native functional descent，也不能保证train与held条件具有一致幅度。
本轮只删除这个双边写出自由度，不改已经通过的视频carrier、credit panel或任务合同。

## 2. 为什么固定A并只写B

选择不是按task、held结果或一次sweep决定。对LPCP143正式correct400缓存的400套真实LoRA统计，单位RMS的B侧
增量通过固定A形成effective-BA的灵敏度，相对单位A侧增量通过固定B的比值为：

| family | `RMS(A0)/RMS(B0)` | native `ULP(A0)/ULP(B0)` |
|---|---:|---:|
| q | `1.0488` | `1.0521` |
| v | `1.4112` | `1.4065` |
| action-in | `2.5935` | FP32 factors，约`2.6284` |
| action-out | `8.2577` | FP32 factors，约`8.7112` |

所以B侧在四类target上都不比A侧弱，在v与action尤其明显；q/v的B绝对值和BF16步长也不更大。三anchor原始
gradient进一步显示v-B与action-out-B持续占主导，q两侧相近。固定A、只写B是由sealed LPCP gauge和policy
topology共同给出的唯一全局规则，不按family或task切换side。

历史Dynamic-K Direct-Family-B 102不能否定本轮：它从fresh弱rank8 Writer出发，固定随机template A；ALB-NV
保留完整LPCP143的condition-specific rank16 A0/B0，只学习一个附加B residual。

## 3. 唯一输出参数化变量

LPCP继续生成完整baseline `{A0(c), B0(c)}`。已有joint Value保持：

```text
X(c) = M_video(c) * RMSNorm(L_language(c)) / sqrt(256)
```

四个shared、bias-free、zero-init heads只输出：

```text
delta-B-q          : 256 -> 2048
delta-B-v          : 256 -> 256
delta-B-action-in  : 256 -> 1024
delta-B-action-out : 256 -> 32
```

总trainable为：

```text
256 * (2048 + 256 + 1024 + 32) = 860,160
```

部署仍是一套完整38-target rank16 LoRA：

```text
A(c) = A0(c)
B(c) = B0(c) + delta-B(X(c))
BA(c) = B0(c) A0(c) + delta-B(X(c)) A0(c)
```

对固定condition，新增effective-BA对head参数和video Value严格线性；没有`B0 delta-A`、没有
`delta-B delta-A`、没有A/B gauge cancellation，也不生成第二套LoRA。step0四head为零，76个public tensors逐tensor
exact LPCP。

## 4. 视频、语言与有向过程

ALB-NV不把LoRA mapper当成视频理解器。新增B Value仍只能来自：

```text
exact language + real ordered action-hidden frames
 -> one joint context forward and 18-layer Action probes
 -> adjacent transition plus causal Procedure
 -> per-video ordered representation
 -> permutation-invariant K-set aggregation
 -> joint video-required X
 -> anchored delta-B
```

语言提供对象、关系和目标address；video动态M提供不可替代的Value。没有video或constant video时`M=0`，所以
`delta-B=0`，语言不能独立写新增LoRA。reverse/shuffle改变真实frame顺序后必须重新走完整carrier，不能只破坏
negative LoRA。

## 5. 多task共存与support保留

所有tasks共享同四个B heads，condition-specific `X`决定写入内容；没有task ID、expert bank或held route。与双边
residual不同，ALB-NV代数上完整保留`B0A0`，新增reward update不能通过修改A侧、gauge rotation或二阶cross term
重构LPCP底座。它仍可能在action函数上产生破坏性干涉，所以只有full24 retained/gained/lost与strict结果能证明
共存；“baseline项存在”不是成功结论。

本轮保留rank16，不做rank8、rank32 reserved lane或compression。历史rank14说明压缩本身会损伤support；若本轮
仍因向非零B0相加而卡native plateau，下一设计才允许用**保留全部rank16并追加zero-B reserved lane**消除局部ULP，
不能在本轮同时改变rank。

## 6. 完整保留的训练合同

- sealed LPCP cold start、AS139 reference、K4、stride5、同task跨episode teacher/action错开；
- 两个paired states、四rollouts、8-strata matched-action successful occupancy、Nmc4；
- 四个互斥correct K4 views、view/task等权、AdamW与PAV actual-candidate backtracking `j=0..10`；
- 同一inference baseline、first-all-view strict descent acceptance、失败exact no-op；
- source policy、split、normalization、BF16/TF32、信息墙、single-checkpoint与一次性Writer部署；
- 不新增memory token、loss、negative、task route、第二adapter、dtype、forward或逐tensor扫描。

保留backtracking只是相同的native acceptance测量，不把本轮重新变成ray研究；若ALB-NV失败，不调ray、LR或scale。

## 7. 固定三anchor快速否决

仍只运行task9/15/18，三项必须全部满足：

1. outcomes=`2/1,2/0,1/2`、complete=`26/65/44`、selected=`8/16/8`、0禁读；
2. step0逐tensorexact LPCP，A tensors训练前后逐tensor不变；只有四个B heads trainable且首步均finite/nonzero；
3. 三任务都在`j<=10`找到first all-view-monotone native candidate；task15不得再卡view3 plateau；
4. accepted delta与Adam candidate同ray，q/v/action effective-BA及fixed-action response非零；
5. train four-view BA cosine/energy至少`.40/.55`；
6. validation8 aggregate至少`.30/.48`、至少6/8过`.15/.40`、raw B至少`.30`、action至少`.15`；
7. held/train effective-BA L2至少`.30x`，task9不得复现`.109x`幅度断裂；
8. natural/reversed至少`.50`、constant/natural不超过`.005`、0 OOM/nonfinite；
9. cycle wall不超过对应PAV的`1.10x`；记录真实search trial但不增加backtrack。

任一失败即终局，不补另一side、不按family混A/B、不扫optimizer/rank/scale/seed/dtype。三项全过后才实现formal
distributed acceptance并启动full24。

## 8. Full24与真实性能裁决

cycle1后立即K4 strict paired400；只有correct至少142、breadth至少7、相对LPCP lost不超过15且gained不少于lost，
才允许exact-resume cycle2。稳定资格保持相邻两个single checkpoints均至少142、均值至少145、churn不超过20、
Jaccard至少`.85`、final lost不超过10。首次约145且retention过门立即补same-task-other、wrong、shuffled、reversed、
no-video，不等到150。

若三anchor机制通过但strict仍下降，最早接口后移到reward Value的on-policy usefulness或shared multi-task mapping，
不能以更线性的LoRA几何冒充性能。

## 9. 快速否决边界

- task15仍无native candidate：非零B0的BF16 residual写入仍是断点，下一步才测试无压缩zero-B reserved lane；
- task9 held/train仍低：断点在joint Value跨condition幅度，不靠side或scale修复；
- anchors通过、full24换手：输出线性化成立但shared reward update不共存；
- correct高而六臂无因果margin：方法仍不能形成有效教学视频claim。

负结果只淘汰`LPCP + MB-SOP + four-view credit + fixed-A B-only rank16 residual + one cycle`。不否定memory token、
rank8、few-shot、无压缩reserved lane、生成LoRA或未来task-local RL。

## 10. Canonical实现状态

canonical实现已原位替换PAV的双边residual topology与fresh schema：active `factor_commitment`只含
`q_b/v_b/action_in_b/action_out_b`四个head；decode只向38个public B tensors加residual，38个A tensors直接返回
LPCP值。旧PAV executable config、checkpoint/completion/evaluator identity已从active runtime移除，历史由Git、design
与formal artifacts保留。定向CPU=`58 passed`，完整CPU在`.env.local` LIBERO assets合同下=`404 passed`，compileall
与diff check通过；architecture guard无hard violation，active source净增长26行且无新module/entrypoint。以上只证明
输出图与fresh边界正确，尚无GPU机制结果。

# V6-LPCP Direct-Factor Adam-Radius Euclidean Commitment

状态：2026-08-16 active single-variable design authority；canonical实现、fresh schema、CPU回归和结构门已完成，
尚未运行GPU。简称`AR-EC`。本轮从sealed LPCP
fresh启动，完整保留MB-SOP的video-language carrier、matched successful-occupancy panel、四个correct K4 views、
八个direct native-factor heads与rank16部署图，只替换shared functional gradient到实际参数delta的commitment。

## 1. 选择依据与最早失败接口

MB-SOP已修复DF-SOCP的stored B2/B1 winner对B8 loser批形混杂，并把三anchor wall从DF-PCSP的
`3.083/5.335/3.887x`降到`1.655/2.119/1.542x`。task9/15/18全部形成健康train/held四video BA；然而真实AdamW
step后，task15 margin从`-.0003242`升到`+.0000303`，task18从`-.0022745`升到`-.0018652`。

同一次确定性复跑直接测得四条flat view gradients：三任务每个view与等权均值的dot全部为正，view-to-mean cosine
minimum为`.695/.629/.601`，共同下降覆盖率均为`4/4`。因此最早失败接口是：

```text
same matched occupancy + four correct-video LoRA cotangents
 -> equal raw parameter gradient is first-order descent for every view
 -> per-coordinate AdamW preconditioning and finite candidate step
 -> retained view0 functional margin ascends on task15/task18
```

本轮唯一主要因果变量是：**保留AdamW候选更新的全局L2半径，但把最终parameter delta严格放回负的raw shared
gradient方向。** 不改credit、video aggregation、carrier、memory、rank、FactorHeads、LR、Nmc、batch或rollout。

## 2. 为什么不是旧projection、PCGrad或LR sweep

- 不做PCGrad/MGDA：原始等权均值已经对4/4 views共同下降，改变view权重会解决不存在的问题；
- 不做旧ADSP式support projection：ADSP约束的是train success-prefix support，strict138证明其不能代表held
  closed-loop occupancy；本轮只处理同一matched functional panel上已直接观测到的optimizer失真；
- 不把Adam LR调小：那会把方向与幅度同时改变并形成小超参sweep；
- 不直接用普通SGD：raw gradient RMS约`1e-8`，固定`3e-4` SGD步会低于native factor有效尺度；
- 不做line search：首轮先用Adam候选本身提供无新增超参的可信半径，检验仅改变方向是否足够。若仍上升，才说明
  最早接口进一步缩到finite-step curvature/trust radius。

## 3. 完整保留项

- exact language + internally ordered action-hidden videos；formal K4，stride5；
- 同一次真实image/language/50 Action-probe context forward与18层LPCP carrier；
- ordered Procedure、K-set聚合、DJNFR `X=M*RMSNorm(L)/sqrt(256)`和八factor-shape heads；
- 每condition一次生成完整38-target rank16 LoRA，rollout期间不再观看视频；
- exact paired AS139/reference与LPCP+heads/candidate、两个states、四个disjoint correct K4 credit views；
- identical B8 reference/candidate occupancy action queries、每成功trajectory八个等进度strata各取最大action RMS；
- trajectory/view/active-task等权、Nmc4、BF16/TF32、0 target/teacher/validation/test action泄漏；
- source policy、normalization、split与部署图冻结；step0/no-video/constant仍严格退化LPCP。

## 4. Adam-radius Euclidean commitment

设task内四video gradient等权，active tasks再等权后、clip后的shared raw gradient为`g`。完全按旧合同对同一
FP32 heads和optimizer state运行一次AdamW，得到候选delta：

```text
d_adam = theta_adam - theta_0
r      = ||d_adam||_2
```

Adam moments、step counter和未来checkpoint state保留该真实raw gradient所产生的更新；但本cycle最终公开参数写成：

```text
d_final = -r * g / ||g||_2
theta_1 = theta_0 + d_final
```

因此`||d_final||_2 == ||d_adam||_2`，没有新增scale、temperature、threshold或solver。只要某view gradient `g_v`满足
`<g_v,g> > 0`，则`<g_v,d_final> < 0`，而Adam逐坐标预条件不能再旋转该一阶方向。若`g`或`d_adam`为零、非finite，
直接机制失败，不做fallback。cycle2仍以保留的Adam moments产生半径，但每个cycle最终方向始终由当次task-equal raw
gradient决定。

这不是把四套LoRA平均：四video仅在训练gradient层约束同一个shared Writer；部署仍对输入K4一次生成一套LoRA。

## 5. 机制诊断

训练时已经存在四条flat gradients，只额外计算一个`4x4` Gram，记录：

- pairwise cosine mean/min/max；
- 每view到task mean的dot/cosine与共同下降覆盖率；
- shared mean energy / view energy。

optimizer step额外记录`d_adam`与`d_final`的L2/RMS、cosine、radius relative error，以及每个active task mean到
`d_final`的descent dot。该诊断不增加policy/Writer forward，也不逐tensor扫描。

smoke-only保留四个view conditioning states和同一selected panel，在step后逐view复算相同flow noise下的margin；
formal不做这四次额外probe forward，避免长期训练吞吐受机制审计拖累。first-view BA/action response仍按旧合同报告。

## 6. Canonical实现边界

- 原位替换MB-SOP config/checkpoint/completion/evaluator schema与optimizer commitment；不保留runtime strategy switch；
- `reward_preference.py`拥有四view raw-gradient geometry；
- `reward_gradient_update.py`唯一拥有Adam candidate和same-radius Euclidean final delta；
- `reward_cycle.py`只传递一个active smoke task的四view probe，不复制policy或LoRA owner；
- fresh incompatible；不得加载MB-SOP smoke heads或optimizer；
- MB-SOP由Git、terminal design、三anchor roots和terminal artifact保存。

## 7. 固定三anchor机制门

仍固定task9/15/18，全部满足才允许full24：

1. outcomes=`2/1,2/0,1/2`，complete chunks=`26/65/44`，selected pairs=`8/16/8`；
2. matched B8两臂batch sequence相同，0 stored rollout action进入loss，0 forbidden read；
3. 四view raw mean共同下降覆盖率=`4/4`，每view dot finite/positive；
4. Adam candidate与final delta finite/nonzero，L2 relative error`<=1e-6`，final到`-g` cosine`>=.999999`；
5. 同一panel、同一flow noise下四个view的post-step margin全部严格下降；
6. 八head delta、q/v/action BA与fixed-action response全部非零；
7. train four-view BA cosine/energy至少`.40/.55`；
8. validation8 aggregate至少`.30/.48`、6/8 tasks过`.15/.40`，raw factor cosine至少`.30`、action cosine至少
   `.15`、held/train L2至少`.30x`；
9. reversed BA relative-L2至少`.50`，constant/natural不超过`.005`；
10. 除smoke-only四view post probe外，core cycle wall不高于MB-SOP对应anchor的`1.10x`；0 OOM/nonfinite。

任一失败即终局，不改radius、LR、eps、optimizer、batch、rank、scale或anchor。若raw coverage不是4/4，说明本轮前提
不成立；若coverage为4/4而same-radius raw step仍上升，说明需要另立finite-step trust design，不能小扫救本轮。

## 8. Full24与真实性能裁决

三anchor全过后从sealed LPCP fresh运行full24 cycle1，立即K4 strict paired400。cycle1须correct至少`142/400`、
breadth至少7、相对LPCP143 lost不超过15、gained不少于lost、无suite清空，且post-train机制不坍塌，才允许exact
resume cycle2。稳定资格仍要求两个相邻single checkpoints均至少142、均值至少145、churn不超过20、Jaccard至少
`.85`、final lost不超过10且gained不少于lost。首次约145且retention过门立即补same-task-other、wrong、shuffled、
reversed与no-video。

full24还必须报告active-task raw gradient pairwise、每task到shared mean/descent delta的coverage。若anchor全过但
task-level coverage或strict仍换手，最早接口后移到跨task共同下降/credit可识别性，而不是再次修改视频carrier。

## 9. 负结果边界

本轮只检验“MB-SOP matched functional credit + equal raw gradient direction + Adam candidate global radius + one fresh
cycle”。负结果不否定memory token、rank8、few-shot、生成LoRA、matched successful occupancy、其它可信半径或未来
生成LoRA后的task-local RL。

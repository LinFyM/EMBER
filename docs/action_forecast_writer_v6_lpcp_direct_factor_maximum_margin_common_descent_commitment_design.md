# V6-LPCP Direct-Factor Maximum-Margin Common-Descent Commitment

状态：2026-08-16 active single-variable design authority，简称`MMCD`；canonical实现、fresh schema与CPU合同已完成，
尚未运行GPU。本轮从sealed LPCP
fresh启动，完整保留MB-SOP的matched successful-occupancy credit、AR-EC的Adam upper radius、AV-MBC的同路径
all-view backtracking、八个direct native FactorHeads与rank16部署图，唯一改变finite-step方向怎样由已经计算出的
四个correct-video gradients确定。

## 1. 选择依据与最早失败接口

AV-MBC clean `202a64d`三anchor复现固定outcomes/counts和raw `4/4` first-order descent，但scalar radius结果分裂：

```text
task9  : first accepted j=10, radius=1/1024, final L2=.0001183;
         held4/8, held/train BA=.18446x，机制门失败
task15 : j=0..10无四view共同下降candidate，恢复step0且BA/action精确为0
task18 : first accepted j=5, radius=1/32, final L2=.005163;
         held6/8, held/train BA=.55527x，全部机制门通过
```

task9与task18可用半径相差约`43.6x`，task15在注册范围内为空；task15到`j=10`仍有一个view增加
`+6.377e-6`，继续缩小时只会进入native BF16零/ULP平台。故full24若仍沿同一raw-mean ray，只可能被最难task压成
near-identity或完全拒绝。AV-MBC否决的是**固定raw equal-mean方向上的单一scalar trust radius**，不是LPCP
carrier、matched reward、direct FactorHeads、rank16或all-view acceptance。

当前最早接口是：**四个video gradients各自有信息且equal mean在连续一阶上4/4下降，但该mean没有为native
BF16 factor/compiler的有限步留下足够的worst-view下降余量。** 下一变量必须改变共同方向，而不是继续缩半径、
扩dtype、扫LR/rank/scale，或退回重做已通过的视频前端。

## 2. 完整保留项与唯一变量

完整保留：

- exact language + internally ordered action-hidden K4 videos，stride5；
- 同一次真实image/language/50 Action-probe context forward、18层LPCP和DJNFR direct FactorHeads；
- 同B8双臂matched actions、每成功trajectory八个等进度max-disagreement states、Nmc4、四个disjoint correct K4
  views及所有information-wall约束；
- 每个task单位权重、source policy/split/normalization、38 targets、rank16与step0 exact LPCP；
- AdamW原始equal-mean gradient的moments、clip和candidate global L2 upper radius；
- AV-MBC同一个inference evaluator的step0 baseline、power-of-two radius schedule和first-all-view acceptance。

唯一变量：final commitment direction不再直接等于四view raw gradients的等权均值，而是它们的
**maximum-margin common-descent direction**。不新增参数、loss、video、rollout、checkpoint、LoRA或部署分支。

## 3. 每task确定性maximum-margin方向

一个active task已经产生四个同单位CFM preference gradients `g_1,...,g_4`。原方向为
`m=(g_1+...+g_4)/4`。MMCD解四变量凸对偶：

```text
lambda* = argmin || sum_i lambda_i g_i ||_2
subject to lambda_i >= 0 and sum_i lambda_i = 1
z       = sum_i lambda*_i g_i
d_task  = z / ||z||_2 * ||m||_2
```

`-d_task`等价于在单位半径内最大化`min_i g_i dot d`的共同下降方向；四个views作为对称约束，没有按video身份、
结果或held性能挑选。`||d_task||=||m||`保持原task梯度尺度。实现以原生FP32 gradient dot形成`4x4` Gram，只将
这个小矩阵的active-set求解转成FP64；模型forward、Writer参数、policy dtype与LoRA仍为原生BF16/FP32合同，
不会为低位一致性扩展百万维gradient dtype。

formal时每个active task独立得到一个`d_task`，再像原合同一样按task等权求mean；Adam moments和upper radius仍由
原始每task equal-view means的task-equal mean产生。这样只改变commit方向，不改变optimizer状态、task权重或
半径来源。所有per-view gradients本来就已计算，MMCD不增加CFM/video/environment forward。

若四gradient convex hull包含原点，则不存在严格共同一阶方向，立即恢复step0并终局；不做gradient weight sweep。

## 4. Native finite-step commitment

令Adam候选upper radius为`r_0`，global MMCD direction为`d`。仍依次检验：

```text
Delta_j = -2^(-j) * r_0 * d / ||d||_2,  j=0,...,10
```

step0和candidate复用同一inference CFM evaluator、occupancy、flow times/noises与conditioning states；接受第一个
所有active panels中每个correct-video margin都严格下降的candidate。world1锚点即四个views；formal必须覆盖所有
active task×4 views，不能只检查每rank首个probe。若无candidate，恢复step0。fixed-action前后继续使用同一
batch1 query/noise路径。

## 5. 为什么仍符合EMBER目标

- 高层任务知识仍由LPCP对exact language与有序视频的joint context提取；MMCD只处理该知识产生的reward cotangent；
- reversed/shuffled会改变LPCP native probes和最终LoRA，constant/no-video不能写出同等dynamic Value；
- 四个same-task correct videos不再只被平均，而是共同约束一个task direction，直接要求跨video可共存；
- Writer仍在rollout前一次生成一套完整LoRA，部署不带reward、expert bank、第二adapter或多checkpoint；
- task等权、single shared heads与single checkpoint不变，不能按task保存不同direction或LoRA。

## 6. 固定三anchor快速否决门

task9/15/18全部满足才允许formal：

1. outcomes=`2/1,2/0,1/2`、complete=`26/65/44`、selected=`8/16/8`、0禁读不变；
2. 原equal mean与MMCD direction都对4/4 views有positive finite dot；MMCD的worst-view directional derivative不得
   低于原mean，权重非负、和为1、结果对view permutation不变；
3. 三任务都在`j<=10`内找到first all-view-monotone candidate；task15必须从AV-MBC no-op恢复，task9不得再失败
   held4/8或held/train`.184x`门；
4. accepted radius、direction、optimizer moments与search顺序精确符合合同，八heads和q/v/action/action response非零；
5. train BA cosine/energy至少`.40/.55`；validation8至少`.30/.48`、6/8过`.15/.40`、held/train至少`.30x`，
   raw factor/action cosine至少`.30/.15`；
6. reverse BA relative-L2至少`.50`，constant/natural不超过`.005`，0 OOM/nonfinite；
7. 记录MMCD Gram/weights、mean与MMCD worst margin、每个search trial和真实wall；不以内部最优替代native结果。

任一失败即终局，不继续改solver tolerance、backtrack数、LR、rank、scale、seed或dtype。

## 7. Full24与strict裁决

三anchor全过后才实现/解锁distributed active-task×4 acceptance，从sealed LPCP fresh完成cycle1并立即K4 strict
paired400。继续门不变：correct至少142、breadth至少7、相对LPCP lost不超过15且gained不少于lost。稳定资格仍需
两个相邻single checkpoints均至少142、均值至少145、churn不超过20、Jaccard至少`.85`、final lost不超过10；
首次约145且retention过门立即补same-task-other/wrong/shuffled/reversed/no-video。

若MMCD三anchor通过而strict仍换手，最早接口才后移到跨task shared mean或matched preference与held closed-loop
reward的对齐；不能再归因于video mean direction、scalar radius或native writeout。

## 8. 负结果边界

本轮只检验“MB-SOP credit + per-task maximum-margin four-view direction + original Adam upper radius + AV-MBC
backtracking + one fresh cycle”。负结果不否定memory token、rank8、few-shot、生成LoRA、direct native heads、
其它policy-space commitment或未来生成LoRA后的task-local RL。

## 9. 实现与启动状态

canonical executable、config/checkpoint/eval schema已原位从AV-MBC替换为MMCD；`4x4` active-set solver、
optimizer-gradient与commitment-direction分离、simplex/permutation/worst-margin证据均有定向合同。完整CPU=
`405 passed`，compileall与diff check通过，architecture guard=`0 hard violations`；未新增active module、并行runtime
或额外forward。当前formal仍锁定，只允许从clean pushed commit分别运行task9/15/18三个world1锚点。

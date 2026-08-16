# V6-LPCP Direct-Factor Preconditioned All-View Backtracking Commitment

状态：2026-08-16 terminal non-pass single-variable authority，简称`PAV-BC`；clean `581140c`三anchor、完整机制分析
与terminal adjudication均已完成，未进入full24/strict。本轮从sealed LPCP
fresh启动，完整保留MB-SOP matched successful-occupancy credit、AdamW optimizer、AV-MBC同路径native acceptance、
八个direct FactorHeads与rank16部署图，唯一把final commitment ray换成实际AdamW candidate delta。

## 1. 选择依据与最早失败接口

MMCD clean `fc3bdd7`已证明三anchor的raw four-view gradients存在更好的continuous common direction：worst-view
directional derivative分别提高`1.216/1.334/1.356x`。但native结果仍是：task9从j10跳到j0却held/train仅
`.160558x`，task18从j5变成j6，task15到j10仍保留同一个view3 `+6.376e-6` plateau并恢复exact no-op。

所以当前断点不是“怎样给四个raw gradients重新加权”，而是：

```text
continuous shared gradient
 -> highly anisotropic FactorHead parameter coordinates
 -> native BF16 generated factors / effective BA
 -> finite-step held policy response
```

已有唯一不增加参数、forward或可调metric的坐标尺度是同一次AdamW产生的真实candidate delta。MB-SOP只测过该
direction的full step，task15/18 view0上升；AV-MBC只对raw equal-mean ray做过power-of-two backtracking；
**实际Adam candidate ray与同路径all-view backtracking从未组合检验。**

## 2. 完整保留项与唯一变量

完整保留MMCD的：

- exact language + internally ordered action-hidden K4 videos，stride5与LPCP one-forward carrier；
- 同B8双臂matched actions、8-strata max-disagreement successful occupancy、Nmc4、四个disjoint correct K4 views；
- trajectory/view/task等权、source policy/split/normalization、38 targets、rank16与step0 exact LPCP；
- AdamW的raw equal-view/equal-task gradient、moments、clip、LR、betas、eps与weight decay；
- 同一个inference CFM evaluator的step0 baseline、`j=0..10`固定搜索顺序、first-all-view acceptance；
- 信息墙、native BF16/TF32、八head、q/v/action/fixed-action、held8与temporal诊断。

唯一变量：删除MMCD raw-gradient solver；final ray直接使用同一次optimizer step得到的`d_adam`，而不是把其L2当
半径再换到raw/MMCD direction。没有新参数、loss、video、rollout、metric、LoRA、checkpoint或部署分支。

## 3. Preconditioned native candidate ray

设四view raw gradients在task内等权、active tasks再等权并clip后为`g`。照原合同运行一次AdamW：

```text
d_adam = theta_adam - theta_0
```

然后先恢复`theta_0`，在同一个retained panel/noise/inference evaluator上依次检验：

```text
Delta_j = 2^(-j) * d_adam,  j=0,...,10
```

接受第一个所有active panels、所有四correct-video views的preference margin都严格低于step0的candidate；没有则
恢复step0。Adam moments与step counter保留真实`g`产生的state。world1锚点覆盖四views；formal仍须在解锁后实现
所有active task×4 panels的共同acceptance。

PAV-BC不声称Adam就是正确policy metric。它检验更窄的命题：**已有per-coordinate preconditioning能否提供跨过
native factor resolution的可用ray，而同路径backtracking能否控制其full-step曲率。** 若失败，后继应改变
LoRA输出/BA参数化或显式native response metric，不能继续在parameter-space rays之间排列组合。

## 4. 为什么不现在改memory、rank或dtype

- LPCP carrier的video/order读取、MB-SOP matched credit与direct q/v/action写出已分别通过，当前证据指向最后的
  finite commitment，不支持同时替换前端；
- memory token仍可能改善layer-aligned LoRA生成，但它不是Adam/raw/MMCD三条ray都在task15 native plateau的直接
  对照变量；
- rank8与reserved/one-sided lanes仍开放，若PAV-BC失败，它们可作为输出参数化设计的一部分另立authority；
- 不用FP32 public LoRA、逐coordinate ULP扫描或更深backtrack来掩盖BF16合同；不扫LR/eps/scale。

## 5. 固定三anchor快速否决门

task9/15/18全部满足才允许formal：

1. outcomes=`2/1,2/0,1/2`、complete=`26/65/44`、selected=`8/16/8`、0禁读不变；
2. `d_adam` finite/nonzero；j0精确等于optimizer candidate，accepted delta精确为`2^-j d_adam`，cosine至少
   `.999999`、L2 relative error不超过`1e-6`；
3. 三任务都在`j<=10`找到first all-view-monotone candidate；task15必须从两轮no-op恢复，task9不得再失败
   held/train幅度门；
4. 四view raw mean仍为4/4 continuous common descent，search前项全部拒绝、接受项四delta全负；
5. 八head与q/v/action/fixed-action response非零；train BA cosine/energy至少`.40/.55`；
6. validation8至少`.30/.48`、6/8过`.15/.40`、held/train至少`.30x`，raw factor/action cosine至少`.30/.15`；
7. reverse BA relative-L2至少`.50`、constant/natural不超过`.005`、0 OOM/nonfinite；
8. cycle wall不高于对应AV-MBC的`1.10x`；记录每个trial与真实wall，不以内部门替代native结果。

任一失败即终局，不增加backtrack、调optimizer、混合raw/MMCD/Adam rays或做rank/LR/scale/seed/dtype sweep。

## 6. Full24与strict裁决

三anchor全过后才实现distributed active-task×4 acceptance，从sealed LPCP fresh完成cycle1并立即K4 strict
paired400。继续门仍为correct至少142、breadth至少7、相对LPCP lost不超过15且gained不少于lost。稳定资格仍需
两个相邻single checkpoints均至少142、均值至少145、churn不超过20、Jaccard至少`.85`、final lost不超过10；
首次约145且retention过门立即补same-task-other/wrong/shuffled/reversed/no-video。

## 7. 快速否决后的架构边界

若task15仍无native共同步，或task9 held幅度仍失败，则`raw equal mean / raw maximum margin / Adam candidate`三类
parameter-space rays连同同路径backtracking都已被固定anchors否决。下一步必须转向**输出参数化级**变量，例如让
LoRA effective BA对共享memory/value呈线性、native-safe的one-sided anchored/reserved lane，而不是继续改梯度权重。

本轮负结果只否定`MB-SOP credit + actual Adam candidate ray + AV-MBC backtracking + one fresh cycle`；不否定LPCP、
memory token、rank8、few-shot、生成LoRA、其它显式native BA参数化或未来task-local RL。

## 8. 实现与启动状态

canonical executable/config/checkpoint/eval schema已原位从MMCD替换：删除maximum-margin solver、per-task
commitment direction与第二个165万维distributed vector；AdamW candidate生成、同路径baseline/backtracking和所有
science graph不变。定向合同确认j0等于optimizer candidate、accepted delta只按`2^-j`缩放；完整CPU=`404 passed`，
compileall/diff check通过，architecture guard=`0 hard violations`，active source相对MMCD净减少156行。formal仍
blocked；当前只允许从clean pushed commit分别运行task9/15/18三个world1锚点。

## 9. 三anchor终局结果

clean pushed `581140c`在gpu02物理`1/2/3`完整复现固定outcomes=`2/1,2/0,1/2`、complete=`26/65/44`、
selected=`8/16/8`与0禁读。实际Adam candidate到负raw gradient cosine分别为`.733415/.798882/.697401`，native
结果为：

```text
task9 : j=5接受，final L2=.00378558；四margin delta=
        [-2.228e-4,-2.858e-6,-5.328e-5,-9.625e-6]；
        train BA cosine/energy/L2=.580922/.667924/7.984e-5；
        held8=.425213/.549139/8.740e-6、6/8，但held/train=.109466x，失败。

task15: j=0..10全拒绝；j10=
        [+3.49e-10,-2.21e-9,-2.572e-6,+6.380e-6]；
        view3与raw/MMCD两轮相同plateau，最终BA/action exact zero，失败。

task18: j=0..10全拒绝；j10=
        [-3.26e-9,-2.794e-6,-9.31e-10,+1.578e-6]；
        最终BA/action exact zero；而raw/MMCD rays此前均能通过该task，失败。
```

cycle wall=`247.970/651.760/367.789s`，相对AV-MBC=`.6953/.9875/1.3922x`；task18还因11次搜索失败吞吐门。
三项机制门为`0/3`，formal未解锁；不full24/strict/resume，不混合raw/MMCD/Adam rays或扫参数。canonical artifact=
`runs/outputs/pi05_v6_lpcp_direct_factor_preconditioned_all_view_backtracking_commitment_task9_mechanism_b8_581140c_gpu02p1_20260816/pavbc_terminal_adjudication.json`。

## 10. 最早失败接口与路线裁决

task9的held/train `.109466x`几乎精确复现MB-SOP full Adam的`.109639x`，说明沿同一Adam ray缩放只改变native ULP
crossing与局部coherence，没有修复跨task幅度传递。task15仍卡同一个BF16 view3 plateau；task18则显示Adam
preconditioning会把raw/MMCD可解task变成空集。因此：

**raw equal-mean、raw maximum-margin与Adam-preconditioned三类parameter-space rays都不能形成跨固定tasks稳定的
native commitment rule。**

最早接口是`shared parameter gradient ray -> native BF16 factor/compiler finite step -> held policy-effective
amplitude`。后继必须改变LoRA输出/effective-BA参数化，使task value通过显式native-safe线性路径写出；不得继续
设计gradient ray、ray mixture或trust scale。

本轮只否定`MB-SOP credit + actual Adam candidate ray + AV-MBC backtracking + one fresh cycle`，不否定LPCP、
memory token、rank8、few-shot、生成LoRA、one-sided anchored/reserved lanes、其它显式BA参数化或未来task-local RL。

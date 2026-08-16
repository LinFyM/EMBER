# V6-LPCP Capacity-Matched Action-Probe Grid

状态：2026-08-16 active design，简称`CAPG`。本轮从sealed LPCP fresh启动，完整保留其correct=`143/400`的
carrier、K4输入、一次真实图文joint forward、NEAP 10-step endpoint credit与native-zero residual bank。
唯一主要变量是把TCEC中所有tasks共用的`320x256 -> four wide B heads`替换为**由逐层Action-probe context直接形成、
容量与部署B payload匹配的参数网格**；不改reward、rank、scale、视频采样、optimizer或部署policy。

CAPG不是literal memory-token架构。它采用Doc-to-LoRA式的per-layer activation queries和SHINE式的
capacity-matched layer/token M2P/direct reshape，但不改变PI0.5 joint backbone的token序列。若本轮证明现有Action-probe
activations不足以形成跨task可共存网格，下一反事实才是在同一真实prefix中加入真正的backbone memory tokens。

## 1. 为什么现在改这个接口

TCEC在同一个world3 shared update中给出以下边界：

- task9/15/18各自四个correct-video endpoint gradients均为`4/4`下降，same-task cosine/energy=
  `.846/.865,.596/.645,.448/.557`，所以视频carrier与endpoint credit不是最早断点；
- 三个task mean gradient norm相差至`41.45x`，task-gradient cosine mean/min=`-.145/-.337`；未经norm重权的shared
  arithmetic mean只对task15下降；
- 11个native scales最多覆盖`8/12` margins，最终四个wide B heads的860,160参数与BA/action响应精确为零。

当前宽head对每个policy slot都执行同一`W h(c)`；一个task的cotangent与condition feature形成全局outer-product
update，异质tasks在同一`W`坐标里直接竞争。梯度归一化、PCGrad、task权重或再换parameter ray只能在训练器外部
调和冲突，不能证明一个shared condition-to-LoRA map自然容纳多task，因此不作为TCEC补丁。

CAPG检验更结构性的反事实：每个condition先产生与最终LoRA B坐标一一对应的contextual payload；共享模块只负责
读context和在layer/token轴通信，不再由四个wide heads把所有tasks压进同一低维feature-to-output矩阵。

## 2. 保持不变的科学图

```text
exact language + four ordered action-hidden correct videos
 -> sealed V6/LPCP real image+language+50 Action-probe joint forward, once
 -> frozen LPCP carrier LoRA
 -> CAPG residual branch
 -> one native rank32 LoRA: A=[A0;A0], B=[B0,delta-B]
 -> frozen PI0.5-LIBERO source policy
```

frame stride5、K4四个disjoint same-task correct views、MB-SOP同B8 panel、NEAP 10-step deployed-action endpoint
preference、AS139/LPCP paired reward outcomes、equal-task arithmetic mean、AdamW与`j=0..10`全局all-view
backtracking全部不变。CAPG不读teacher action/proprio/reward/task ID/filename，不引入expert bank、第二adapter、
checkpoint union或生成LoRA后的task-local RL。

本轮继续保留rank16 carrier加rank16 residual，而不是同时降到rank8。原因不是认定rank16长期最优：TCEC最早失败在
shared mapping，历史fresh rank8 Dynamic-K的`100--102`与uniform rank14 support损伤又说明rank变化会同时改变
reachable policy support。先固定当前已验证的A0 row space和public rank32，才能把结果归因给parameter-grid mapping；
rank8仍是后续独立变量。

## 3. 一次forward中的逐层context readout

现有LPCP hook已经从同一次真实joint forward旁读18层Action Expert hidden。CAPG在不改变token序列的前提下额外保留
每层50个Action-probe states：

```text
H[f,l,a,:] in R^(frames x 18 x 50 x 1024).
```

这些states已经在有效image+exact-language prefix下形成，不是blank image、language-only或memory-only空跑。
对每层使用37个有位置归属的latent queries，通过低维Q/K路由读取50个states，Value保持native 1024维：

```text
Z[f,l,m,:] = Attention(Q[l,m], K(H[f,l,:,:]), V=H[f,l,:,:])
Z in R^(frames x 18 x 37 x 1024).
```

这些37个是backbone之后的**parameter latents**，不是memory tokens；它们不进入Action Expert层内attention。
Q/K route width固定为128以控制吞吐，payload不经`1024->256`压缩，也不通过family hidden/GELU wide head。

## 4. 37来自输出合同，不是任意token数

CAPG只生成rank16 native-zero B residual，A使用LPCP已经policy-effective的`A0`。每个Action Expert layer需要：

```text
q_B: 16 x 2048 = 32768
v_B: 16 x  256 =  4096
total per layer = 36864 = 36 x 1024.
```

所以每层前36个payload slots可无损直写q/v B。第37个slot沿layer轴承载endpoint targets：

- layers0--15的16个完整1024 payload直写`action_in_B: 16x1024`；
- layer16前512值直写`action_out_B: 16x32`；
- layer16后512与layer17的1024共1536值为固定unused padding。

全B payload=`680,448`，网格容量=`18x37x1024=681,984`，unused仅`1,536`=`0.225%`。因此37是当前
rank16-B-only deployment topology的精确结果；若未来改rank或生成完整A/B，payload cardinality必须重新推导。

## 5. 视频内有向过程与跨视频共同程序

每条video独立处理。对逐帧Z构造：

```text
D[0]=0; D[f]=Z[f]-Z[f-1]
G=Z[last]-Z[first].
```

一个共享的轻量causal controller只在frame轴运行一次；它从所有layer/payload的D摘要获得有RoPE真实frame ordinal的
final context，再为每个`(l,m)`计算frame weights。完整1024维D按这些weights汇聚并加terminal goal G，得到每video
`18x37x1024` Program。它继承LPCP已经通过的“一条视频一个causal controller，再轻量汇聚所有policy cells”原则，
不为666个cells分别跑长序列Transformer。

正确顺序是结构必要量：reverse翻转G并改变全部adjacent D，shuffle改变D与causal context；constant video使D和G
严格为零。exact language只通过真实joint context和Q/K address决定“关注什么”，不能独立成为B payload Value。

同condition的K条video Programs只在video集合轴进入无video-position的set attention，再做对称pool，保证置换不变；
不平均raw frames/features，不分别生成LoRA后平均，也不挑video。实现接口允许K1--K4，但本轮训练与裁决保持K4以匹配
LPCP/TCEC；在训练真实覆盖其它cardinality前不宣称Dynamic-K性能。

## 6. Layer/token M2P与direct reshape

集合Program随后只做两次必要的拓扑通信：

1. 固定payload coordinate下沿18层通信；
2. 固定policy layer下沿37个payload coordinates通信。

两轴都使用128维Q/K routing、native 1024维Value的single axis mixer，带layer/token positional route但没有video
ordinal。它保留payload宽度并避免1024宽FFN或四个独立wide factor heads。最终网格按第4节确定性slice/reshape成
`q_b/v_b/action_in_b/action_out_b`，不再学习从256 hidden到2,048等宽度的输出矩阵。

这采用SHINE中“layer/token交替通信后按容量直接映射LoRA”的可扩展原则：policy层数或LoRA payload增长时，状态网格
线性增长，而共享Q/K controller参数不为每个target另建巨大head。

## 7. Step0 exact carrier与gradient-open anchor

CAPG保留一份初始化相同、永久冻结的轻量readout/temporal/set/M2P anchor `F0`，trainable branch为`Ftheta`；两者
读取同一份detached Action-probe H，不重复backbone forward：

```text
delta_grid(c) = (Ftheta(H(c)) - F0(H(c))) / sqrt(1024)
delta_B(c)    = deterministic_reshape(delta_grid(c)).
```

所有模块无dropout。初始化时两分支逐元素相同，故任意condition的delta-B为exact zero，deployment逐元素退化LPCP；
同时gradient只经过Ftheta，latent queries、temporal、set和M2P首步均可获得非零credit。冻结anchor不是第二套部署
LoRA或第二backbone，它只是一个轻量zero-origin parameterization；部署仍输出一套adapter。

constant video在任意训练阶段均因D=G=0而产生零residual。若实现不能同时满足step0 exact identity、首步全链gradient
非零与constant zero，工程门直接失败。

## 8. 为什么它可能让多task共存

CAPG不要求不同task reward gradients同向，也不以正交为目标。它提供三层条件分流：

- 不同language/video context在每层50个Action states上产生不同attention；
- 每个LoRA payload coordinate有独立query/position ownership；
- layer/token M2P只在明确policy topology内共享通信，最后无wide output matrix竞争。

因此不同tasks可以通过不同context-dependent Jacobians写同一个Writer的不同有效坐标；same-task不同videos则由
causal Program和K-set提取共同部分。是否真的成立不看representation漂亮程度，而直接用TCEC相同三task endpoint
gradient与closed-loop裁决。

## 9. 预正式三task可证伪门

实现先用固定task9/15/18、world3、每rank一个task复现TCEC的6 paired states/12 rollouts/48 correct videos。
除CAPG mapping外所有数据、noise、B8 panel与endpoint metric相同。

必须同时满足：

- step0 carrier tensor exact、factor/BA/action counterfactual为零；anchor全冻结，只有Ftheta可训练；
- 每task four-view gradients均4/4 continuous descent，same-task pairwise cosine/energy不低于`.35/.50`；
- 不做norm重权的三task arithmetic mean对三个task均为descent，即TCEC的`1/3`提升为`3/3`；记录task-gradient
  norm ratios与pairwise cosines，但不以几何本身选方法；
- 同一个global Adam candidate沿`j=0..10`必须使12/12 endpoint margins严格下降，否则exact no-op并终局；
- accepted后train三anchor与validation8的four-view BA、raw-B、action、held/train、reverse/constant门沿用TCEC阈值：
  train至少`.40/.55`，validation aggregate至少`.30/.48`且6/8过`.15/.40`，raw-B cosine`.30+`、action
  cosine`.15+`、held/train BA L2`.30x+`、reverse`.50+`、constant/natural不超过`.005`；
- world3 wall不超过TCEC`182.142s`的`1.75x=318.749s`，峰值低于A40容量，0 OOM/nonfinite/禁读。anchor只重复
  轻量post-backbone path，不能造成第二次joint forward。

连续shared mean若仍非3/3，说明per-layer activation parameter grid没有解决task coexistence，立即终局；不加
normalization/PCGrad/task router。continuous通过而native失败，则direct grid仍没有共同finite policy step；不扫scale。
held/temporal失败则分别定位condition generalization或有向Value失败。只有全门通过才启动full24。

## 10. Full24、strict与稳定性

三task门通过后从同一clean architecture fresh运行full24 cycle1，task等权，仍只保留一个shared checkpoint；随后
立即K4 strict paired400。至少correct`>=142`、breadth`>=7`、相对LPCP lost`<=15`且gained不少于lost才允许相邻
cycle。首次达到约145且retention过门立即补same-task-other/wrong/shuffled/reversed/no-video，不等到150。

最终资格要求相邻single checkpoints约145或更高、churn`<=20`、Jaccard`>=.85`、final lost`<=10`，并且correct
在严格配对闭环中实质优于所有视频controls。若CAPG只提高内部coherence或LoRA幅度而strict下降，仍为non-pass。

## 11. 负结果边界

本轮只淘汰实际检验的组合：`sealed LPCP Action-probe activations + capacity-matched post-backbone parameter latents +
causal/set/axis M2P + anchored direct native-B reshape + NEAP endpoint credit`。

它不否定literal memory token、rank8、Dynamic-K/few-shot、生成完整A/B、LoRA本身或未来task-local RL。特别是，CAPG
中的37个post-backbone latents若失败，不能被描述为“memory token失败”；真正让memory在Action Expert层内逐层读取
context的反事实仍未执行。

# V6-LPCP CFMG Median-Capped Task-Tangent Commitment

状态：2026-08-17 cycle1完成、strict400 pending的active formal authority。简称 **MCTC**。本轮从sealed LPCP fresh开始，不resume USDC；完整保留
USDC已经形成跨视频共同LoRA坐标的memory/content-grid/rank32图、K4 unit-secant reward和一次自然Adam commitment，
唯一改变active-task gradients进入shared mean之前的幅度聚合。

## 1. Earliest failed interface

USDC cycle1 strict=`138/400`、breadth6，相对LPCP143为`120 retained / 18 gained / 23 lost`，Long suite净`-4`，
因此坏cycle1不允许用训练量小为由resume。但它不是memory或LoRA写出失败：all400 BA relative-L2 mean=
`.003236`，8个held tasks的同task correction cosine/energy=`.555--.953/.663--.965`。

unit-secant已经把旧task38/次大gradient dominance从`58.73x`降到`6.274x`，却没有消除它。USDC full24的六个
active-task norms为`.147608/.022184/.066100/.047038/.124794/.926067`；task38到shared mean cosine=
`.977239`，raw shared descent只覆盖4/6 tasks。它在train task38上获得成功credit，却对应held Long suite最大净损失。
所以当前最窄反事实不是再改视频carrier、M2P、rank或reward occupancy，而是：**同一套video-coherent task
tangents若不再被单个幅度outlier控制，能否形成更可共存的shared Writer update。**

## 2. Unchanged deployment and training graph

```text
exact language + K=4 ordered action-hidden same-task videos
 -> one carrier-exact native PI0.5 context forward
 -> 37 layer-matched one-way memory tokens
 -> per-video temporal program + permutation-invariant K-set
 -> layer/token M2P content grid -> native rank16 B residual
 -> concatenate frozen LPCP rank16 carrier
 -> one complete rank32 38-target LoRA -> frozen policy rollout
```

frame stride5、两paired states、每成功轨迹8个等进度max-disagreement states、四个互斥correct K4 views、
per-state unit action secant、BF16/TF32、batch8、LR/betas/clip、AdamW、RNG和信息墙逐项不变。仍只运行96个
train rollouts；不增加expert、validation gradient、second adapter、language-only Value、rank/scale search或生成后RL。

## 3. Median-capped task tangent

每个active task先与USDC完全相同地把四个video-view gradients等权平均为`g_t`。令

```text
n_t = ||g_t||_2
m   = median({n_t | t is active})
c_t = min(1, m / max(n_t, eps))
g_shared = mean_t(c_t * g_t)
```

`eps`只防止除零；严格零gradient task不进入active panel。对偶数task数量使用框架确定性的中间两值平均定义。
这个规则没有task-specific常数或可扫阈值：所有task使用同一中位数cap。它只抑制上半部的大幅度outliers，绝不
放大小梯度，不改变任一task tangent方向，也不做PCGrad、orthogonalization、maximum-margin solver或per-suite权重。
task采样、每task states/views和macro权重仍相等；task ID只属于训练时既有的distributed aggregation owner，不进入
Writer输入或held部署。

对`g_shared`执行与USDC完全相同的一次fresh AdamW step并保存唯一`j0`。不backtrack、不扫cap multiplier、不拿
20/20 margin挑checkpoint。pre/post task×view margins、uncapped/capped norms、task-to-shared cosine和q/v/action
response全部记录为诊断。

## 4. Why this is not a surrogate-only patch

历史variance-reduced estimator改变的是flow Monte Carlo噪声，K-set variance reduction又只稳定了错误task mean；
本轮二者都不重复。当前证据是一个具体的跨task幅度outlier在unit action-secant之后仍控制public update，并与held
Long净损失同suite对应。median cap不声称局部margin必然代表成功；它只生成一个预先唯一确定的candidate，仍立即
由single-checkpoint strict400裁决。

## 5. Formal plan and training volume

canonical实现与CPU合同通过后，从sealed LPCP、fresh optimizer做一次full24 cycle1。已有USDC已经关闭carrier、
cache、rank32 evaluator和batch16 longest-video工程风险，因此不重复held surrogate screen或参数profile。

cycle1训练量仍只有24 tasks/48 states/96 rollouts，故判断分两层：

- 若candidate nonzero，立即K4 correct strict paired400；
- 若correct至少142、breadth至少7、相对LPCP lost不超过15且gained不少于lost，则锁原world topology
  exact-resume cycle2并再次strict400；
- 约145只有在cycle1/2相邻success sets低churn、高Jaccard并共同积累时才有资格补六臂；
- 若cycle1像USDC一样同时丢correct、breadth和retention，则终局，不以“下一cycle才打开temporal模块”为由盲续。

## 6. Fast falsification and negative boundary

实现门必须证明uncapped task gradients与USDC定义一致、cap从不放大小task、最大capped norm不超过median、所有ranks
使用同一ordered active panel且j0 finite/nonzero。以上只证明算法接通。

若strict没有相对LPCP/USDC减少lost并恢复breadth，淘汰“当前unit-secant CFMG tangents的median upper-cap能形成
held-useful shared commitment”。下一接口应转向sparse selected-success credit本身如何学习task-general Value，
而不是继续扫cap quantile、clip multiplier、optimizer、rank或memory数量。

无论结果如何，本轮不否定literal memory、dynamic K/few-shot、rank8、生成完整LoRA或未来独立task-local RL。

## 7. Canonical implementation status

唯一active config、checkpoint/evaluator identity与reward update已经原位切换。实现只增加task-panel median upper cap，
并复用原coexistence所需的global task matrix作为optimizer输入，删除旧shared-gradient额外all-reduce；没有新增forward、
module、entrypoint或并行runtime。集成测试直接验证不等幅task gradients的cap、无小task放大以及Adam一阶矩；定向/
完整CPU=`47/416 passed`，compileall与diff check通过，architecture guard 0 hard。当前尚无GPU机制或closed-loop结果。

## 8. Cycle1 execution state

clean`1a0700f`在gpu01 physical`0/2/4/5/6/7` world6完成fresh full24：24 tasks/48 states/96 rollouts，
candidate/reference=`33/32`、gains=`3/2`，active tasks=`4/19/25/34/38`覆盖四suite，cycle=`388.239s`。
五个raw norms=`.147608/.022184/.047038/.124794/1.133739`，median=`.124794`；只对task4/38应用
`.845440/.110073` scale，其余三个严格为1。shared与final descent coverage均为5/5，保存j0 L2=`.236963`；
5/5 probes的q/v/action BA与fixed-action response全部非零，0禁读/OOM/nonfinite。14/20 local margins下降只作
诊断，不选择checkpoint。checkpoint现已按标准post-training合同sealed；下一裁决是同一single checkpoint的K4
strict paired400，尚无closed-loop结果。

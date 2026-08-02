# Contextual-Value Asymmetric Dual-Read Writer 设计

**状态：2026-08-02 已集成为canonical，参数10,241,024。RAW八点correct400为
`76/111/99/117/77/69/80/82`，single winner保持macro200；第二小时没有成熟化，
不做五臂。macro200/400 matched梯度方差已完成，同topology GROUP4 B20/105-frame
profile与formal-seed exact-resume已经seal，待fresh 0→1200正式recipe反事实。**

## 0. Formal evidence override

CV-ADR与UCP RNG-v2 RAW在训练recipe、task/video/query exposure和paired evaluation
panel上严格相同。CV四个候选相对UCP为`+4/+24/+13/+28`；macro200的+28同时包含
多保留9个source successes和多获得19个新successes。因此本设计作为整体通过
“架构路径有真实价值”判别，但117 absolute与明显task churn不构成成功。

macro200内部结果同时给出保留/否定边界：Core-only和Program-only距full BA分别
`.6059/.8119`，ProgramRead/CoreRead RMS比均值`1.021`，Effect-only距full`.0674`；
保留mean-backed Core、contextual Program value和dual read。删除Action只在1/8 tasks
达到预注册门、删除D为5/8，contextual-memory order为0/8；same-task BA variance
`.1049%`、fixed-action中位`.00856%`。所以不能宣称已解决视频教学，也不能退回AP
raw Effect value。第二小时只判别成熟度；若仍失败，最早未满足职责是有序Program
内容到target/rank read及高噪声训练operator的共同接口，不得用scalar gate/scale修补。

macro200→400成熟度段已完整结束。step100--400的global full24 raw-mean direction
没有candidate-negative task，但约`.36--.50` task-pair仍为负；late factor head占
task-gradient energy约`93.6--94.0%`。第二小时同task one-video梯度仅
`.26--.49%`属于task mean，约`99.5--99.7%`为centered变化，相邻余弦
`.024--.041`；即使LR降到`1.0e-5`，相邻50-macro参数段方向仍不稳定。16个held
functional losses横盘于`.13055--.13399`。因此full24 candidate并未直接伤害任一
task不足以说明优化稳定，CP投影也不会修复高方差条件估计或functional/closed-loop
错位。正式250/300/350/400为`77/69/80/82`，macro200→250立即lost56/gained16；
后段LoRA norm不降反升，故是闭环能力轮换/off-manifold，不是训练不足或增益坍缩。

固定visit397--399的matched方差分解进一步识别了上述centered变化的来源。
`lora/action`在macro200/400的video主效应仅占centered energy
`.1211%/.1060%`，0/24 tasks由video或其interaction主导；query占
`48.59%/49.53%`，flow与query×flow合计`48.78%/48.50%`。macro400 task-mean
energy降为macro200的`.492`，centered energy保持`1.025`，Program task-mean方向
余弦只有`.431`。同时24/24 tasks的匹配functional loss继续下降，中位delta
`-.00411`。但visit397--399对macro400是刚曝光的train条件、对macro200尚未曝光，
这个下降不能冒充held generalization；结合独立held functional loss横盘，它只证明
train surrogate/recency拟合继续加强而closed loop退化。所以单条teacher video的
随机选择不是晚期参数旋转主因，但video对局部训练方向几乎没有控制力本身就是架构
失败；真正的训练问题是query/flow主导的低SNR与closed-loop错位。后续训练估计器
可以研究保持B20边缘分布的无偏flow方差降低，
但不得删除真实query多样性、固定query过拟合或平均多video/LoRA。正式pair SHA为
`ad7d6e06...44eb96a`。

跨v5.2/v6的2×2正式审计进一步收紧本文的recipe解释。task-complete在两种架构上
都保留约相同的normalized Procedure顺序差异，却把shuffled/reversed的
Procedure→effective-LoRA transfer压到old的`.26--.58`、Procedure→action压到
`.34--.50`左右；selected behavior margin也在两种架构上一致收缩。但correct
absolute分别`132→120`和`121→143`，matched 150-video-visits effect仍为
`-81/+16`。所以CV若失败，不能只怪RAW/full24，也不能直接恢复old recipe：old每
cycle六次Adam形成近正交、更大幅的参数路径，同时恢复条件写出和能力轮换。下一结构
必须把causal Program内容与一个有界但不被full24稀释的conventional写入接口共同
设计；这是一项整体职责重构，不授权事后添加固定scale。联合审计SHA为
`98371337...2efa`。

逐task审计又排除了“v6高分证明当前compiler已正确”的解释。matched150时v6-new
相对v6-old的+16由source retention -1和new gain +17构成、8/8 tasks非零，说明v6
语义/transition bundle值得保留；但selected +22几乎全由Object task3的+24主导，
Object1/Goal6的顺序margin接近零。它们的Visual Transition和Procedure order response
仍大而BA/action很小，故下一版若需要重构，应把mean-backed Core和显式visual change
作为可复用证据，把当前set-like target/rank reader与写出接口作为待替换owner；不能
把Visual Transition本身判死。逐taskanalysis SHA为`611c9330...c5a1`。

本文负责AP-ADR一小时门失败后的下一整体AS-Writer结构。它同时服从两条已经独立
成立的事实：

1. 架构职责会决定视频动态是否有一条可训练的写出路径；
2. 同一架构下，update granularity、Adam时钟、参数重线性化和task顺序又会显著
   改变这条路径最终写到effective BA/action的程度。

因此本文只解决已被AP内部反事实直接定位的架构接口；训练方法另由exact UCP上的
cycle-normalized randomized-group4受控格先识别，再决定是否迁移。不得把两者重新
混成一个无法归因的首跑。

## 1. 最终目标与非目标

最终模型必须让一条action-hidden teacher video同时提供：

- 稳定、跨视频可共享的task semantic carrier；
- Action hypothesis之后出现的endpoint Effect与change；
- 真实有序、因果的Program content，而不只是attention寻址扰动；
- 38个真实policy targets共享但不完全相同的写入；
- 16个可保持coherent高增益流形、同时允许必要分化的rank coordinates；
- 从视频到effective BA再到fixed-query policy action可量化的函数变化。

非目标是制造高rank、低cosine、漂亮margin或更大的LoRA norm。首版不新增gate、
scale、confidence、null token、第二套LoRA、谱约束、辅助loss、static bypass或
teacher action/state/reward输入，也不改变split、frame stride、source policy、
normalization、38-target public LoRA contract或一视频一LoRA信息墙。

## 2. 直接证据：AP最早死在什么地方

AP macro50/100/150/200 correct400为`91/81/94/91`，不进入第二小时。修复PI05
sampler attention-backend生命周期后，macro150 refs1在8/8 validation tasks上逐层、
effective BA和fixed action严格零误差重放。有效analysis/summary SHA为
`d42fc4eb...bc2b`/`f2c572c5...e682`。

跨task均值为：

| 条件 | raw Program | block2 Program | Program read | BA | action |
|---|---:|---:|---:|---:|---:|
| same-task-other | .9188 | 1.1051 | .03210 | .03005 | .01668 |
| wrong | 1.0651 | 1.3069 | .15185 | .14540 | .02926 |
| shuffled | .2808 | .09066 | .02790 | .002689 | .002200 |
| reversed | .2414 | .07112 | .03787 | .003903 | .002160 |

反转valid contextual temporal keys、保持Core/raw A/E/D/mask/position固定时，
BA/action仅变`.000521/.001944`。ProgramReader entropy约`.904`、top mass约
`.0106`。Contextual Program有内容和顺序变化，却只拥有近均匀softmax的K职责。

A/E/D反事实把V职责进一步定位：Effect-only距full BA仅`.00821`，Action-only和
Change-only距full为`.2761/.2832`；固定full contextual key后结论不变。Effect
缩放0.5/2改变BA `.141/.289`，Action或Change缩放最多`.008/.001`。所以AP的真实
写入是raw Effect DC，causal Program只造成微弱寻址扰动。这是v8式Effect dominance
在更晚接口的复现，不是“所有Action tokens无用”。

同时，Core-only距full BA/action`.283/.228`，Program-only距full`.961/.494`；两路
都不是形式分支。删除Core mean改变BA`.834`，删除centered residual只变`.0128`，
故Recenter已经揭示的mean semantic basis必须保留。

## 3. 实现前唯一实质选择

合理候选有三种：

1. **选择：同一contextual Program同时作K和V。** 两层axial后的状态是唯一Program
   memory；ProgramReader用`RMSNorm(P)`形成K、用未归一化`P`形成V。
2. 把`P`重新按raw RMS缩放后作V。它能机械限制幅度，但重新引入手工scale和
   terminal normalization；D本来幅度小，按raw幅度还会先验保持其弱势，缺乏证据。
3. 删除residual/FFN，只让一次attention convex average作V。它同时改掉Program
   capacity和优化路径，且可能重演Recenter式basis starvation，无法隔离AP已定位的
   K/V职责错误。

选择1改变最少且职责完整：AP现有causal computation不再只是路由器，而是实际被
写入的内容。它不是给失败checkpoint加一个residual；fresh模型中不再存在并行raw
Program value接口。选择2、3只有在选择1被内部幅度或可训练性证据证伪后才可重开。

## 4. Canonical计算图

新结构命名为 **Contextual-Value Asymmetric Dual Read（CV-ADR）**：

```text
task language + one action-hidden teacher video
  -> Q_text, M_f, G_f, native mean Action A_f
  -> mean-backed permutation-invariant Semantic Core C
  -> raw outgoing Program R_i = [A_i, E_i=G_(i+1), D_i=G_(i+1)-G_i]
  -> two causal axial blocks: P = Axial(A/E/D over columns and time)
  -> target-only CoreRead(C): [38,256], broadcast over rank
  -> target/rank ProgramRead(K=RMSNorm(P)+QK identities, V=P): [38,16,256]
  -> raw concat [CoreRead ; ProgramRead]: [38,16,512]
  -> eight bias-free coherent factor heads
  -> one complete public rank-16 LoRA over all 38 targets
```

`P`是唯一Program输出。model/compiler接口不得再同时暴露`program_key`和
`program_value`，避免raw Effect旁路被以后无意恢复。frame/type/target/rank/order
identity只进入Q/K；任何identity在零内容时都不能制造V。Core和Program使用独立
softmax，最后只concat，不相加、不相乘、不归一化、不调制、不经过global mixer。

## 5. 与SPG、UCP和AP的精确区别

- 相对AP，只改变已被直接否定的职责：`V=raw A/E/D`改为`V=contextual P`，并删除
  key/value二轨接口；frontend、Core、A/E/D对齐、两层axial、双reader和heads保留。
- 相对UCP，恢复mean-backed Core和独立Core read；Program用`A/E/D`而不是把
  absolute `X`与A/D塞入单一softmax；factor head继续读512维双路concat。
- 相对SPG，没有Core加法旁路、target-Core first-hop加法或跨target/rank global
  mixer；Program content按target/rank直接读取。
- 相对v10，没有terminal RMSNorm/AdaLN/gate把微小Procedure强制放大14--20倍。
  `P`的物理幅度直接受functional path约束，不在V端强制单位化。

## 6. 幅度风险与证伪

AP macro150中raw/block1/block2 Program RMS约`.20/.61/1.88`。这个增长在AP里
对函数几乎不可识别，因为block2只进RMS-normalized K；不能把该checkpoint幅度直接
当成CV-ADR初始化或训练后的幅度。CV-ADR让P直接进V后，functional gradient会首次
识别其scale，但仍可能产生两种失败：

1. ProgramRead远大于CoreRead，重演v10式动态支配；
2. optimizer把P压回近零，Core独占写入。

因此训练热路径每次optimizer update按owner记录梯度、Adam
`sqrt(v)/eps`和相对fresh identity的累计参数位移；raw P、每block P、CoreRead、
ProgramRead、concat与factor的RMS则在候选checkpoint上由canonical内部分析按task、
video condition完整重建。不得为了逐step激活日志再做额外Writer前向或保留大tensor，
因为那会改变正式B20的显存与吞吐合同。首小时内部门要求多个task上
ProgramRead/CoreRead RMS处在有限、非单边坍缩的宽区间，并由反事实证明两路都
必要；不通过时重构block职责，不得事后加固定scale或gate。

## 7. 参数和代码边界

CV-ADR已在独立写worktree实现为fresh incompatible schema。真实module
enumeration为`10,241,024`，与设计预算精确一致：

| owner | enumerated parameters |
|---|---:|
| Meta-LoRA frontend | 2,469,888 |
| evidence projections | 983,552 |
| Semantic Core | 1,836,544 |
| causal contextual Program | 1,838,592 |
| asymmetric readers | 409,088 |
| coherent factor heads | 2,703,360 |
| **total** | **10,241,024** |

代码owner保持：

- `semantic_core.py`：mean-backed、task-selected centered Core；
- `semantic_program.py`：raw A/E/D、causal axial P；
- `program_compiler.py`：单一P的Core/Program dual read；
- `model.py`：只编排，不保存旧AP可选分支；
- `architecture.py`/config/checkpoint schemas：fresh incompatible authority；
- `internal_path.py`：canonical路径重建、K/V parity和结构反事实；
- `internal_analysis.py`：task/probe/runtime编排；
- `internal_metrics.py`/`internal_results.py`：纯指标与分布式结果封存。

历史UCP/AP由Git与frozen control worktree/artifact保存；CV canonical tree不保留
runtime switch、version enum、UCP专用分析器或并行runner。结构门复核结果为
`hard_violations=[]`；新路径owner分别为635/494行，整个active-source diff相对
`1a09e71`净减少779行。

## 8. 最短vertical path

正式训练前至少通过：

1. A/E/D outgoing alignment与全部ragged masks；
2. Core frame permutation invariance；
3. P causal prefix invariance和未来帧不泄漏；
4. 单一Program tensor同时拥有K/V职责，源码与模型接口均无raw-value旁路；
5. type/order/target/rank identities不进入V，零Program不能造值；
6. CoreRead `[B,38,256]`、ProgramRead `[B,38,16,256]`、concat512；
7. 76个public tensors/38 targets/rank16/transpose全部完整；
8. step0逐tensor严格template-A/zero-B identity；
9. source policy trainable参数0；
10. frontend/Core/Program/readers/factor在identity lifecycle后梯度finite可达；
11. checkpoint与exact resume；
12. 最长105-frame真实video、B20、4 ranks三完整macro。

B20只在真实OOM或连续非有限时降B16，不扫描B17--B19/B21。

截至本次实现seal，shape、mask、identity、freeze、causal/permutation、主要模块
gradient、checkpoint/update/evaluator schema及内部分析parity的focused回归共
`159`项通过。随后实现rebase到`8dfe6ed`的RNG-v2与fail-closed schedule authority，
全仓CPU回归`226 passed in 35.56s`；`compileall`、四个CV config loader与diff
check通过。task-query RAW/GROUP4使用fresh-incompatible v2 family，旧CV RNG-v1
family fail-closed。这里不把CPU/合成验证冒充第12项真实profile，第12项仍明确pending。

## 9. 架构与recipe的执行顺序

训练根因已经在exact UCP上完成预注册受控格。RNG-v2 RAW与cycle-normalized
randomized-GROUP4共用task/query-keyed stateless CPU time/CUDA noise、相同200
task visits和paired panel；correct400分别为`72/87/86/89`与`77/76/66/100`。
GROUP4 endpoint增11但四点均值下降3.75、winner top2集中到74%，四点累计只有2/8
tasks上升、5/8下降。它把success envelope gap从60缩到34，却没有达到absolute/
breadth行为门，也没有消除task轮换。

matched exact50进一步证明normalized GROUP4不是中性operator：相对RAW，删除A/D的
effective BA变化从`.058999`降到`.013291`，fixed-X shuffled/reversed BA从
`.028069/.025026`降到`.009288/.009092`，8/8 tasks一致。reader X/D/A mass
从`.434/.522/.044`转到`.560/.405/.035`。LoRA norm`59.42→63.70`、stable rank
`1.0066→1.0021`，即更coherent但更static，不是增益坍缩。唯一GROUP4 action异常
来自一个closed-loop 0-success task。operator/exact SHA为
`97c70dd...a6e0`/`7201364a...11fd`。

所以CV-ADR首跑仍用RAW：这使AP→CV只改变已经被直接否定的K/V职责，优先识别
topology。GROUP4保留为同topology fallback，而不是默认；如果CV RAW弱或含混，
必须再跑CV GROUP4后才可拒绝架构。旧未归一SERIAL仍只说明六倍optimizer gain可
放大动态和off-manifold方向，不能直接迁移。

endpoint10已经以全局Spearman`.258398`、100,000次permutation `p=.298447`失败
预注册global门；尤其v6-fast macro200的correct133对应18点最差quality。故CV-ADR
不能使用endpoint10选点或训练，candidate裁决仍只认paired closed-loop、breadth、
gained/lost/Jaccard及Core→Program→BA→action传递。

CV-ADR四个fresh schema config已经随merge `b97960f`进入main；旧UCP config与专用
analyzer退役。真实B20最长视频profile、formal-seed exact resume和RAW fresh
macro0→400均已完成；八点paired correct400全部封存，single winner保持macro200。
RAW行为门失败，不做五臂；matched方差诊断后执行同topology GROUP4。

raw第二小时门为：best至少与UCP/SERIAL的`117/121`同档且由多个task贡献，并且右端
趋势或内部Program主路明确；默认强续训仍要求`>=125`、breadth`>=6`且top2不过度
集中。该门已授权并完成exact-resume到400；强single checkpoint才跑五臂。

## 10. 内部成功与直接失败条件

对correct/same/wrong/shuffled/reversed逐task报告完整Core/Program/compiler/BA/action
链。至少包含以下fresh反事实：

- contextual P作为K/V的full；
- Core-only、Program-only；
- A-only、E-only、D-only、A+E、A+D、E+D，均从raw列开始重算全部axial P；
- fixed Core/vary Program、fixed Program/vary Core；
- temporal order/key/value实际重排后完整前向；
- target/rank identity permutation；
- effective BA geometry和fixed-query action。

主路径工作的最低方向量：

- Effect-only不能在8/8 tasks以`<2%` BA误差复现full；
- A/D移除或只保留应在至少6/8 tasks造成`>=5%` BA或`>=2%` action变化；
- shuffled/reversed的Program变化不能再次在reader处压成`<1%` BA；
- contextual order intervention应在至少6/8 tasks造成`>=2%` BA变化；
- Core-only与Program-only都不能近似full；
- 没有v10式Program/Core RMS单边放大，也没有Target-Spectral式norm/coherence坍缩。

若上游P强、上述量仍在ProgramRead处消失，则reader职责仍错误；若P本身不再保留
order，则问题在axial evidence computation；若BA变化充分而correct持续弱，则转向
functional surrogate/source-policy有效流形与训练update，而不是继续改router。

这些内部门不能替代closed-loop absolute、breadth和漂移。即使视频margin漂亮，
absolute低仍不是成功；即使correct高，Effect-only或order-independent旁路仍需继续。

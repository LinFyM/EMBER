# Contextual-Value Asymmetric Dual-Read Writer 设计

**状态：2026-08-01 evidence-sealed design authority；尚未实现、profile或训练。**

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

因此每步记录raw P、每block P、CoreRead、ProgramRead、concat和factor的RMS、梯度、
Adam `sqrt(v)/eps`与累计位移。首小时内部门要求多个task上ProgramRead/CoreRead RMS
处在有限、非单边坍缩的宽区间，并由反事实证明两路都必要；不通过时重构block职责，
不得事后加固定scale或gate。

## 7. 参数和代码边界

CV-ADR不新增parameterized module，预期真实参数仍为AP的`10,241,024`：

| owner | expected parameters |
|---|---:|
| Meta-LoRA frontend | 2,469,888 |
| evidence projections | 983,552 |
| Semantic Core | 1,836,544 |
| causal contextual Program | 1,838,592 |
| asymmetric readers | 409,088 |
| coherent factor heads | 2,703,360 |
| **total** | **10,241,024** |

真实实现后必须重新enumerate，不能把预期值冒充实测。代码owner保持：

- `semantic_program.py`：raw A/E/D、causal axial P；
- `program_compiler.py`：单一P的Core/Program dual read；
- `model.py`：只编排，不保存旧AP可选分支；
- `architecture.py`/config/checkpoint schemas：fresh incompatible authority；
- `internal_analysis.py`：只保留与新canonical路径严格parity的诊断。

历史AP由Git/frozen artifact保存；main不保留runtime switch、version enum或并行runner。

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

## 9. 架构与recipe的执行顺序

训练根因先在exact UCP上完成已经预注册的受控格：fresh raw与cycle-normalized
randomized-group4共用task/query-keyed stateless policy noise/time；这保留UCP已有
raw、未归一SERIAL和normalized group4三角，不拿CV-ADR替训练假设背书。

截至2026-08-01，group4最长105-frame B20 profile、formal-seed
fresh0→1→resume1→3→7和raw fresh0→1→resume1→3均已通过，两臂cycle0的24个
teacher-video assignments逐项一致，配置已seal。canonical UCP恢复commit为
`85a82cb`；正式两臂尚未启动。只有完成相同paired correct400后，才能按预注册证据
决定CV-ADR是否仍先用raw或迁移normalized group4，不能根据smoke/profile选择recipe。

endpoint10已经以全局Spearman`.258398`、100,000次permutation `p=.298447`失败
预注册global门；尤其v6-fast macro200的correct133对应18点最差quality。故CV-ADR
不能使用endpoint10选点或训练，candidate裁决仍只认paired closed-loop、breadth、
gained/lost/Jaccard及Core→Program→BA→action传递。

CV-ADR实现后首跑仍用raw-full24/B20/fast400、fresh macro0->200，评测
50/100/150/200 paired correct400；它与AP raw只差K/V职责，优先识别topology。
只有UCP normalized group4通过预注册absolute/breadth/drift/dynamic联合门，才把同一
recipe迁移到CV-ADR；否则CV-ADR不因UCP训练负结果被否定，继续按raw结果判断架构。

raw第二小时门为：best至少与UCP/SERIAL的`117/121`同档且由多个task贡献，并且右端
趋势或内部Program主路明确；默认强续训仍要求`>=125`、breadth`>=6`且top2不过度
集中。达到门才exact-resume到400；强single checkpoint才跑五臂。

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

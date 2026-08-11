# Policy-Innovation Goal–Causal Key Writer

状态：2026-08-11唯一active successor authority。PICK-GC canonical owner、fresh-incompatible config、
implementation阶段`345 passed` CPU回归；exact raw full48、world6 discarded mechanism、B8/16/32吞吐与
zero-memory deployment vertical均已通过并封存，seal后完整回归为`346 passed`；当前已formal-ready，但尚未
训练或做strict paired400。

## 1. Decision

PICK-GC只改变PICK已经失败的static descriptor，并原样保留已通过raw order门的causal descriptor：

```text
PICK:
  [whole-video static mean,
   centered causal prefix]

PICK-GC:
  [terminal-quartile goal residual,
   centered causal prefix]
```

frozen source-policy frame innovation、zero-image subtraction、phase16、fixed 2×128 JL、
historical v6-fast 600个frozen tensors、单一FP32 Program memory、原生38-target rank16 compiler、
blind full48 correct/negative-zero solve、B20跨episode source-action credit和one-shot部署全部不变。

因此本设计检验的唯一主要假设是：PICK的全视频static common mode造成full48 condition collision；只把它换成
“最终目标相对整段过程的残差”，能否在保留PICK完整有向阶段证据的同时，让24 task与24 counterfactual在
同一个可解condition basis中共存。

## 2. Why PICK stopped

PICK通过authority-matched raw-frame probe：same causal`.95567`、cross`.15582`、reverse`-.98045`、
八shuffle aggregate`-.06703`、complete same`.97450`，raw重排与hidden重排完全等价，所有信息墙与freeze门通过。

随后clean pushed commit `c7e58c9`上的world6、train24×B20 discarded full48 profile只失败一项：

- feature rank=`48`；
- regularized Gram condition=`483.61515`，超过预注册上限`200`；
- correct motion/cotangent=`.95151`；negative/correct leakage=`.03815`；
- correct retained=`24/24`，reversed/shuffled/wrong null均=`8/8`；
- Program application closure=`0`，LoRA A/B与fixed action均响应，四suite breadth=`4/4`；
- production wall ratio=`1.16910`，0 OOM/nonfinite/negative policy forward。

所以最早失效接口是full48 key conditioning，而不是policy evidence读取、functional cotangent、blind solver、
Program→LoRA→action传递或吞吐。不得放宽condition门、增大damping或扫seed/width/scale来挽救PICK。

## 3. Exact goal–causal key

对exact language `T`和每个sampled frame，仍由frozen source policy产生zero-image-subtracted innovation，
并按真实顺序线性resample为：

```text
x_0 ... x_15 in R^3072
```

定义：

```text
m = Mean(x_0 ... x_15)
g = Mean(x_12 ... x_15)

r_goal   = g - m
z_p      = x_p - m
r_causal = Mean_p (Sum_{u<=p} z_u / sqrt(p+1))

u_goal   = ZeroL2(P_goal r_goal)
u_causal = ZeroL2(P_causal r_causal)
phi      = ZeroL2(concat(u_goal, u_causal)) in R^256
```

`P_goal/P_causal`继续使用PICK已封存的fixed seed `20260810`、无bias、逐row normalized FP32
`[128,3072]` projection。首版不改变quartile、phase数、projection、block权重、feature width或normalization，
也不做descriptor sweep。

## 4. Why this represents transferable task knowledge

`r_goal`不是绝对场景均值。它表示终段相对同一视频整体过程新增的policy-grounded对象/关系状态；同task不同
初始化只要到达同一目标关系，就应保留共同方向。`r_causal`是PICK已经验证的全阶段有向prefix contrast，
保留“先接近、再抓取、再移动、最后放置”等内部阶段连续关系，而不要求复刻速度、路径或抓取角度。

video与action query继续同task跨episode错开，因此Writer不能用逐帧动作对应复制teacher轨迹。PICK-GC也不读
task ID或expert bank；held部署仍只获得exact language与一条action-hidden video。

## 5. Why correct order is structurally necessary

reverse会把终段换成原初段，并近似翻转PICK causal prefix：

```text
correct  -> [g - m,  r_causal(x_0 ... x_15)]
reverse  -> [i - m,  r_causal(x_15 ... x_0)]
```

第二块读取整条有序过程，reverse近似翻转且任何中间阶段重排都会改变它；第一块从目标残差变成初态残差。
shuffle通常也改变terminal quartile，因此同时破坏目标与阶段过程。即使某个shuffle恰好保留终段，causal块
仍使内部阶段顺序具有结构作用。
这不是在negative LoRA上施加破坏性margin：blind solve的negative RHS仍严格为零，negative保留同一个frozen-v6
base。若方法有效，只能因为correct ordered key获得由correct source-action cotangent定义的正向Program increment。

静态视频满足`g=m`且全部`z_p=0`，两块与完整key精确为零。zero-image subtraction后language本身没有condition value，
因此不存在language-only或static-scene residual bypass。

## 6. Train-only selection evidence

只读sealed train24×50×phase16 action-hidden cache，无action/reward/terminal/validation/test reads：

| key | same mean/median | LOO mean/median | cross mean/median | correct24 cond max |
| --- | ---: | ---: | ---: | ---: |
| PICK static/causal | `.95762/.96281` | `.97810/.98083` | `.49866/.49105` | `108.56` |
| PICK-GC goal/causal | `.90260/.91604` | `.94875/.95606` | `.13455/.11375` | `21.62` |

PICK-GC另有complete correct/reverse cosine mean/median=`-.80305/-.80877`、八shuffle raw complete
mean/median=`.00412/-.00421`，50个correct24 panels全部rank24。它牺牲一部分仍高于门槛的same-task tightness，
换来约3.7倍cross-task降低和约5倍correct24 conditioning改善。goal块same mean=`.88910`，保留的causal块
same mean=`.91610`；后者对只打乱中间阶段、保留首尾的shuffle仍会改变。

一个曾考虑的reversal-symmetric endpoint key虽把reverse cosine精确压到0、correct24 cond max降到`16.95`，
但same-task mean/median只有`.86177/.88508`，低于预先沿用的complete same`>=.90`门，已拒绝且不实现。
一个仅用terminal-minus-initial effect的版本虽same mean=`.90054`，却对保留首尾quartile、只打乱中间阶段的
shuffle完全不敏感；聚焦CPU测试暴露该结构漏洞后也已拒绝，不能靠削弱测试进入runtime。

cache中的wrong video使用其自身language，不能精确模拟target-language wrong；full48近似rank仅40，condition
`821.06`。当前PICK同一近似为`2244.40`而raw exact为`483.62`，这个比例只能说明PICK-GC值得做一次exact
raw gate，不能预测它必过`200`。

## 7. Coexistence and support preservation

PICK-GC去掉跨task高度相关的whole-video static mean；两个zero-sum temporal contrasts让不同task主要按目标关系
与有向过程分开。fixed key加linear Program memory继续让异质task cotangent写入各自condition neighborhood，
避免shared neural condition map把它们压成common update。

已有support仍由以下不变量保护：

- historical v6 base逐tensor冻结，fresh Program memory全零时LoRA逐tensor等于immutable native macro0；
- 不压缩、SVD、refactor、regenerate或改变rank16 factor topology；
- residual仍在frozen-v6 fused slots之后、FactorHeads之前进入原生高增益compiler；
- checkpoint只拥有一个Program memory与完整cursor/RNG/topology，不拥有第二套base或condition bank。

这不能保证闭环共存；它只移除PICK已经实测的condition common-mode bottleneck。真正retention仍由paired400
的gained/lost/churn裁决。

## 8. Implementation lifecycle

1. 原位把`PolicyInnovationCausalConditionFeature`替换为单一
   `PolicyInnovationGoalCausalConditionFeature`；不保留runtime strategy flag或import shim。
2. 删除PICK active config，建立fresh-incompatible PICK-GC config/checkpoint schema；历史PICK由Git、本文和
   retained raw/full48 artifacts保存。
3. `FrozenV6ConditionResidualWriter`、training graph、solver、compiler、evaluator与launch topology不复制。
4. CPU只新增goal/causal公式、reverse/shuffle、zero、freeze、fresh/resume与config fail-close回归。
5. 来自clean pushed commit的frozen worktree先运行exact raw-key full48 feature gate，再运行discarded full48
   mechanism profile；不过门不创建formal root。

## 9. Falsification gates

### 9.1 Exact raw-key feature gate

使用PICK profile封存的同一24 correct与8 reversed/8 shuffled/8 target-language wrong panel：

- feature rank必须=`48`；
- regularized Gram condition必须`<=200`，沿用相同`.01 × mean diagonal`定义；
- 4-suite raw same-task complete mean必须`>=.90`；
- raw correct/shuffle aggregate absolute mean必须`<=.10`；
- raw-frame重排与一次encode后hidden重排max error必须`<=1e-5`；
- zero key与所有trainable counts必须为0。

任一失败即拒绝PICK-GC，不改threshold、damping、seed、quartile、scale或projection。

### 9.2 Discarded full48 mechanism and throughput gate

沿用PICK全部门：rank48、condition`<=200`、correct retained至少21/24、三类negative各至少6/8且合计至少18、
negative/correct motion`<=.15`、Program application/LoRA A/B/四suite fixed-action闭合、0 OOM/nonfinite/
forbidden reads/negative action forward、production wall不超过sealed world6 baseline的`1.75x`。

随后只做一次longest-video B8/16/32 generation profile并选择最高stable LoRAs/s；不为低位一致性固定batch1、
重复forward或扩dtype。

### 9.3 Real behavior gate

全部机制与deployment门通过后：

1. 证明zero-memory对immutable macro0 vertical identity；
2. fresh Program memory blind full24 B20训练`0→10`；
3. 立即评测同一schedule的strict paired correct400。

macro10必须correct`>=144`、breadth`>=6`、相对macro0 lost`<=8`且gained>lost。失败即退役当前
PICK-GC+blind-credit组合，不切RLS/Reward/rank/few-shot，不做小参数sweep。若144--150且全门通过，仅允许一次
exact-resume`10→25`后再paired400；首次`>150`或达到144并通过稳定门，立即补同checkpoint
same/wrong/shuffled/reversed/no-video严格controls。

### 9.4 Sealed pre-formal result

clean pushed commit `717b561`上的全部前序门已经完成：

- exact raw full48：rank=`48`、condition=`152.45803`、4-suite same-task mean=`.94501`、
  reversed=`-.81318`、shuffled=`-.09783`、target-language wrong=`.12235`，raw/hidden reorder误差和zero key均为0；
  shuffle aggregate距`.10`上限很近，因此这里只解释为过预注册门，不写成强shuffle分离。
- world6 discarded mechanism：condition=`152.61008`、correct motion/cotangent=`.96457`、
  negative/correct=`.03901`、correct retained=`24/24`，reversed/shuffled/wrong各null=`8/8`；Program、LoRA A/B、
  四suite fixed action均闭合，production wall ratio=`1.13558`，0 OOM/nonfinite/negative policy forward。
- longest-first同面板吞吐：B8/B16/B32分别`.47119/.47244/.47299` LoRAs/s，三者均stable、最长67帧、
  约34.8GiB余量；按预注册最高吞吐规则选B32，但其相对B16优势仅约`.12%`，不作方法收益解释。
- zero-memory vertical：四suite每个`76/76` LoRA tensors与immutable native v6 bit-exact，四个同噪声action也
  bit-exact，Program memory非零数0；canonical evaluator随后生成8-entry cache、释放Writer并复用同一source
  policy完成8/8 rollouts，观测`4/8` success仅是smoke结果，不参与方法选择。

首次vertical把prepare与identity放在同一Python进程，LIBERO缓存了原子rename前的staging config而失败；retry只
恢复标准进程边界，没有改变模型、面板或门。canonical rollout全部成功后，第一次CPU finalizer又误读manifest
字段；修正为`len(entries)`后只重跑CPU验证。两项均是保留的工程失败，不是科学non-pass，也没有重跑GPU结果。

这些证据把最早未决接口推进到blind offline full24 credit能否在held on-policy occupancy上积累support；只有
fresh`0→10`和随后的strict paired correct400能裁决，前序机制数值不能替代closed-loop。

## 10. What this result can and cannot conclude

exact raw与world6 full48 condition已经分别以`152.45803`和`152.61008`通过，因此key数值可解性、
Program→LoRA→action闭合和production吞吐均不再是formal前的未决接口。这些结果仍不能说明blind offline
credit能覆盖held on-policy occupancy，也不能预言closed-loop分数。若macro10闭环失败，最早剩余接口就是
blind offline credit或key与held occupancy错位；不得回头把失败归因于已通过的full48 condition或
Program→LoRA增益，也不因此否定condition-local memory、few-shot、task-level manifold supervision或新的
on-policy credit。

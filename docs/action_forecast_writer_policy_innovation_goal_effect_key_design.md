# Policy-Innovation Goal–Effect Key Writer

状态：2026-08-11唯一active successor authority；PICK原始static/causal key已在exact full48
mechanism profile因条件数non-pass，PICK-GE尚未实现、profile、训练或rollout。

## 1. Decision

PICK-GE只改变PICK的两块condition descriptor：

```text
PICK:
  [whole-video static mean,
   centered causal prefix]

PICK-GE:
  [terminal-quartile goal residual,
   terminal-minus-initial directed effect]
```

frozen source-policy frame innovation、zero-image subtraction、phase16、fixed 2×128 JL、
historical v6-fast 600个frozen tensors、单一FP32 Program memory、原生38-target rank16 compiler、
blind full48 correct/negative-zero solve、B20跨episode source-action credit和one-shot部署全部不变。

因此本设计检验的唯一主要假设是：PICK的全视频static common mode造成full48 condition collision；
以“最终目标相对整段过程的残差”和“初态到终态的有向效果”作为key，能否保留same-task共同任务知识，
同时让24 task与24 counterfactual在同一个可解condition basis中共存。

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

## 3. Exact goal–effect key

对exact language `T`和每个sampled frame，仍由frozen source policy产生zero-image-subtracted innovation，
并按真实顺序线性resample为：

```text
x_0 ... x_15 in R^3072
```

定义：

```text
m = Mean(x_0 ... x_15)
i = Mean(x_0 ... x_3)
g = Mean(x_12 ... x_15)

r_goal   = g - m
r_effect = g - i

u_goal   = ZeroL2(P_goal r_goal)
u_effect = ZeroL2(P_effect r_effect)
phi      = ZeroL2(concat(u_goal, u_effect)) in R^256
```

`P_goal/P_effect`继续使用PICK已封存的fixed seed `20260810`、无bias、逐row normalized FP32
`[128,3072]` projection。首版不改变quartile、phase数、projection、block权重、feature width或normalization，
也不做descriptor sweep。

## 4. Why this represents transferable task knowledge

`r_goal`不是绝对场景均值。它表示终段相对同一视频整体过程新增的policy-grounded对象/关系状态；同task不同
初始化只要到达同一目标关系，就应保留共同方向。`r_effect`表示从初段到终段发生了什么有向改变，区分
“把对象放入容器”和相反过程。两块都由跨初始化仍成立的目标与效果构成，不要求复刻路径、速度或抓取角度。

video与action query继续同task跨episode错开，因此Writer不能用逐帧动作对应复制teacher轨迹。PICK-GE也不读
task ID或expert bank；held部署仍只获得exact language与一条action-hidden video。

## 5. Why correct order is structurally necessary

reverse会把`g`与`i`交换：

```text
correct  -> [g - m,  g - i]
reverse  -> [i - m,  i - g]
```

第二块严格翻转，第一块从目标残差变成初态残差。shuffle会随机改变首尾quartile，因而同时破坏目标与效果。
这不是在negative LoRA上施加破坏性margin：blind solve的negative RHS仍严格为零，negative保留同一个frozen-v6
base。若方法有效，只能因为correct ordered key获得由correct source-action cotangent定义的正向Program increment。

静态视频满足`g=m=i`，两块与完整key精确为零。zero-image subtraction后language本身没有condition value，
因此不存在language-only或static-scene residual bypass。

## 6. Train-only selection evidence

只读sealed train24×50×phase16 action-hidden cache，无action/reward/terminal/validation/test reads：

| key | same mean/median | LOO mean/median | cross mean/median | correct24 cond max |
| --- | ---: | ---: | ---: | ---: |
| PICK static/causal | `.95762/.96281` | `.97810/.98083` | `.49866/.49105` | `108.56` |
| PICK-GE goal/effect | `.90054/.91575` | `.94760/.95587` | `.13525/.11310` | `22.04` |

PICK-GE另有complete correct/reverse cosine mean/median=`-.81563/-.81973`、八shuffle raw complete
mean/median=`.00460/-.01332`，50个correct24 panels全部rank24。它牺牲一部分仍高于门槛的same-task tightness，
换来约3.7倍cross-task降低和约4.9倍correct24 conditioning改善。

一个曾考虑的reversal-symmetric endpoint key虽把reverse cosine精确压到0、correct24 cond max降到`16.95`，
但same-task mean/median只有`.86177/.88508`，低于预先沿用的complete same`>=.90`门，已拒绝且不实现。

cache中的wrong video使用其自身language，不能精确模拟target-language wrong；full48近似rank仅40，condition
`866.05`。当前PICK同一近似为`2244.40`而raw exact为`483.62`，这个比例只能说明PICK-GE值得做一次exact
raw gate，不能预测它必过`200`。

## 7. Coexistence and support preservation

PICK-GE去掉跨task高度相关的whole-video static mean；两个zero-sum temporal contrasts让不同task主要按目标关系
与有向效果分开。fixed key加linear Program memory继续让异质task cotangent写入各自condition neighborhood，
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
   `PolicyInnovationGoalEffectConditionFeature`；不保留runtime strategy flag或import shim。
2. 删除PICK active config，建立fresh-incompatible PICK-GE config/checkpoint schema；历史PICK由Git、本文和
   retained raw/full48 artifacts保存。
3. `FrozenV6ConditionResidualWriter`、training graph、solver、compiler、evaluator与launch topology不复制。
4. CPU只新增goal/effect公式、reverse/shuffle、zero、freeze、fresh/resume与config fail-close回归。
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

任一失败即拒绝PICK-GE，不改threshold、damping、seed、quartile、scale或projection。

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
PICK-GE+blind-credit组合，不切RLS/Reward/rank/few-shot，不做小参数sweep。若144--150且全门通过，仅允许一次
exact-resume`10→25`后再paired400；首次`>150`或达到144并通过稳定门，立即补同checkpoint
same/wrong/shuffled/reversed/no-video严格controls。

## 10. What this result can and cannot conclude

若exact full48仍因condition失败，只淘汰这组goal/effect linear descriptor，不否定frozen policy innovation、
condition-local memory、few-shot、task-level manifold supervision或新的on-policy credit。若机制通过而macro10
闭环失败，则最早剩余接口转为blind offline credit或key与held occupancy错位；不能再把失败归因于
Program→LoRA增益或full48数值可解性。

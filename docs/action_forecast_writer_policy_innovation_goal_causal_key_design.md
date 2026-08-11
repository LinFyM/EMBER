# Policy-Innovation Goal–Causal Key Writer

状态：2026-08-11 formal non-pass、已退役。PICK-GC canonical owner、fresh-incompatible config、exact raw
full48、world6/world4 discarded mechanism、B8/16/32吞吐与zero-memory deployment vertical均通过；随后
formal fresh`0→10`完整训练，但single-checkpoint strict paired correct仅`138/400`、breadth6，相对immutable
macro0 retained/gained/lost=`118/20/16`。它同时未过`correct>=144`和`lost<=8`预注册门，因此不得resume到25、
不得补controls或做参数sweep。本文现在是PICK-GC+blind offline source-action credit的封存结果authority，
不再授权GPU执行。

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

沿用PICK全部科学门：rank48、condition`<=200`、correct retained至少21/24、三类negative各至少6/8且合计至少18、
negative/correct motion`<=.15`、Program application/LoRA A/B/四suite fixed-action闭合、0 OOM/nonfinite/
forbidden reads/negative action forward。world4 topology reprofile的吞吐基准只按每rank task数从sealed PICK-GC
world6实测`25.351229s × 6/4 = 38.026844s`线性归一，预注册上限为该值的`1.25x`；不得拿减少GPU导致的自然
wall增长冒充架构退化，也不得在结果后改比例。

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

### 9.5 Resource-driven world4 topology amendment

`5200bee`已经把world6与deployment证据封存并push。2026-08-11 18:47+08:00再次同时live检查两节点时，
`gpu02`只有physical`1--5`五张空闲，`gpu01`只有`2/4/6/7`四张空闲；没有任何单节点world6。不能跨节点拼卡、
触碰他人进程或等待凑卡，而train24 task-complete要求world size整除24，因此选world4；第五张空卡不能在保持
每task等权、每rank固定local task数的合同下提高吞吐。

这不是新的科学方法：每个task仍独立产生相同video/query/RNG cotangent，collective仍只all-gather完整24个
correct、24个negative与24个cotangent，随后按task ordinal固定排序`0...23`再做同一FP32 full48 solve；没有
memory all-reduce。唯一变化是world6/local4变为world4/local6及其低位kernel/reduction order。正式资格和config
artifact seal已经主动重置；world4必须在fresh macro0 discarded profile上复现全部数值/闭合门及上述归一吞吐
门，失败就不训练。既有exact raw、B32 generation和zero-memory vertical不依赖训练world size，保留为immutable
evidence，但只有world4 profile过门后才原样重新挂回formal seal。

实际`09bbed3` world4 profile使用`gpu02:2,3,4,5`，physical/local/NUMA为`2/0/0, 3/1/0, 4/2/1, 5/3/1`；
launch前四卡均4MiB/0%、ECC与row remap全0，Git clean且origin一致。结果中functional loss、24 task records、
full48 update、application与LoRA/action response相对world6 artifact逐字段exact：rank48、condition=`152.61008`、
correct motion/cotangent=`.96457`、negative/correct=`.03901`、retained/null=`24/24`，三类negative各`8/8`。
step=`34.94275s`，相对归一基准`38.02684s`为`.89871<1.25`；14项检查全true、exit0、无checkpoint，GPU已释放。
因此纯拓扑复现通过，既有B32 deployment evidence原样重新挂回，world4 formal fresh`0→10`获准。

### 9.6 Sealed formal result

formal训练root为
`runs/outputs/pi05_pick_gc_goal_causal_formal_fresh0to10_r4_b20_c2e1ff8_20260811`，训练commit
`c2e1ff878b6b68cb5bc45bb5443cdbd54ab8e62a`。world4从全零Program memory完成10个task-complete macro，
checkpoint、4-rank RNG、cursor和10行metrics完整；每一步feature rank均48，condition范围`83.61--152.88`，
functional loss约`.093--.100`且没有改善趋势。10次value-delta RMS的RSS=`3.4596e-6`，最终FP32 Program
memory RMS=`3.5493e-6`、20,971,520个值全部非零，说明更新主要在连续memory中积累而非简单相消。

strict evaluation先暴露一个纯工程错误：evaluator用过时的默认world6校验真实world4 checkpoint，未进入GPU；
`398425ee018097ba4c446f91bfe04ea65f6c7c5f`把expected world size改为读取sealed config并加回归，canonical
环境完整测试`346 passed`。随后从该clean pushed commit的detached worktree完成48/48 shards、400/400 rows、
12/12 persistent rollout workers，正式root为
`runs/outputs/pi05_pick_gc_goal_causal_correct400_noreplacement_seed7_macro0010_retry1_398425e_20260811`。

按Spatial1/3、Object1/3、Goal3/6、Long1/2顺序：

| checkpoint | total | breadth | per-task |
| --- | ---: | ---: | --- |
| v6-fast historical best | 143 | 6 | `0/3/46/37/0/36/20/1` |
| immutable macro0 | 134 | 6 | `0/5/48/34/0/35/11/1` |
| rank14 compiler-only | 138 | 7 | `1/1/46/32/0/35/22/1` |
| rank14 online | 128 | 7 | `1/1/47/29/0/36/13/1` |
| PICK-GC macro10 | 138 | 6 | `1/3/48/33/0/39/14/0` |

macro0→PICK-GC在相同400个task/state、env seed、policy seed root和policy-noise common prefix下为
retained/gained/lost/both-fail=`118/20/16/246`、churn36、net`+4`。79个noise-list长度差异全部来自不同的
提前终止；teacher video identity也逐row一致。PICK-GC虽与compiler-only同为138，却少一个breadth，仍低于
v6-fast 5分；Long1和Object3的丢失由Goal6等增益抵消，属于target-heterogeneous capability rotation。

对同一400个task/state/video的macro0 B8与macro10 B32缓存，以低秩trace恒等式计算effective BA而不展开矩阵：
整体范数中位比=`1.000016`，cosine=`.99999724`，相对L2=`.002397`；q/v/action相对L2中位分别
`.002406/.002385/.003968`。batch shape会贡献普通BF16低位差异，因此不解释逐元素细节；但结合非零Program、
fixed-action闭合和真实`+20/-16`，可以确认写出既非identity也非能量塌缩，而是很小、很高杠杆且未对准held
occupancy的切向更新。这个结果不支持增大scale、norm或rank来补救。

完整决策、逐task/逐suite转移、pairing和effective-BA证据封存在上述eval root的
`pick_gc_formal_decision_evidence.json`。正式裁决是：只退役PICK-GC key与blind train24 B20 source-action
functional cotangent的组合；保留goal-causal ordered evidence、condition-local FP32 Program memory和原生
rank16 compiler作为未被否定的子机制。下一设计必须只改变credit/occupancy接口，不能恢复本run、做小sweep或
把negative LoRA人为破坏。

## 10. What this result can and cannot conclude

PICK-GC已经实测修复PICK的full48 condition collision，并证明ordered key、full48 solve、FP32 Program写入、
native compiler和fixed action全链闭合；formal closed-loop却只有138且lost16。故当前最早失效接口是blind
train24 offline source-action functional cotangent到held on-policy有用support与共存的映射，而不是key数值
可解性或单纯Program→LoRA传递。该负结果不否定condition-local memory、few-shot、task-level manifold
supervision或新的on-policy reward credit；也不能反向证明任何这些候选有效。

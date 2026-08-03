# Semantic Direction Store Writer

状态：2026-08-03 BCI fresh候选设计authority；canonical实现、train24 center
authority与61项聚焦CPU合同已完成，尚未做真实profile、训练或rollout，不得写成
有效方法。本文在ordinary Semantic Factor-Basis完整0→400和
variance-reduced estimator 0→200均完成负裁决后建立。实现时原位替换canonical
factor router/head；历史SFB只由Git、frozen config和正式artifact保存，不保留可执行
并行架构。

## 1. 要回答的问题

当前证据已经把“视频没有被理解”与“动态路径没有到达policy”降为次要解释：

- Target-Bound的remove-A、remove-D和causal-memory reversal均为`8/8` tasks过门；
- SFB macro200的task-conditioned route确实分化，task-centered/sample energy为
  `.2171`，task均值route pair relative-L2中位为`.6049`；
- Core-only、Program-only、A/E/D反事实均证明完整路径到达effective BA和fixed
  policy action；
- SFB macro200相对source base paired gained/lost=`84/5`，Writer具有真实新增能力。

但ordinary SFB的完整paired correct400为：

```text
69 / 91 / 118 / 127 / 117 / 81 / 126 / 120
```

八点union=`193`而single winner只有`127`，macro200与350近乎同分却
gained/lost=`31/32`。第二小时factor gradient share由`.9586`升到`.9691`，
24-task raw mean只保留约`.0444→.0420`的单task gradient energy，同task successive
factor CountSketch cosine由`.0676`降到`-.0099`。Adam二阶moment相邻段余弦稳定在
`.9145--.9448`，一阶moment却接近零并转负。VR只给出极小、非持续的机制改善，
closed-loop仍为`76/88/126/107`。

因此本版检验的不是“再增加一个router”或“把hidden机械加宽”，而是：

> 不同task所需的完整LoRA生成方向是否需要真正独立的参数存储，并由稳定的任务语义
> 地址组合，而不是继续共享同一个factor output matrix。

## 2. 保持不变的任务与信息墙

仍为严格one-shot：

```text
exact task language + exactly one action-hidden teacher video
→ one complete task-specific rank-16 LoRA
→ frozen PI05-LIBERO source policy rollout
```

保持：

- 24 train / 8 validation / 8 test development split；
- 一条video生成一套完整LoRA，不做few-shot、多video平均、多LoRA平均或checkpoint融合；
- Writer不读取teacher action、proprio/state、reward、terminal、task ID、filename、
  normalization或policy outcome；
- teacher action只进入同task独立query上的frozen-policy functional loss；
- frame stride5、38个真实policy targets、rank16、76个public A/B tensors；
- Target-Bound的mean-backed Core、target-bound private A/E/D causal Program和
  rank-last compiler；
- template A / zero B与factor final zero-init保证step0 functional identity；
- source policy保持0 trainable parameters。

取消旧的Writer参数量软上限。参数增加必须对应当前已观测的条件方向存储职责，
不能通过无主的dense层、全局scale、第二套adapter或静态language LoRA旁路堆容量。

### 2.1 方法核心必须与监督目标解耦

本候选的科学主张只允许是：**稳定任务语义地址组合多个完整、独立的条件生成参数
存储，从参数所有权上降低task间生成方向干扰**。这条机制只要求最终学习信号能从
generated LoRA回传，因此应同时适用于action-supervised functional gradient和后续
rollout reward gradient；不得把当前AS estimator、完整24-task宏步或LIBERO固定常数
包装成方法贡献。

明确区分：

- objective-agnostic架构：冻结语义地址、完整Direction Store、value只来自
  Core/Program、同一任务整套LoRA共享组合；
- 当前domain配置：train24均值、`K=8`、PaliGemma 2048维anchor、PI05的38 targets与
  rank16；换任务集合或base policy时允许按训练语言和真实adapter topology重新实例化；
- AS执行配方：full24 raw mean、logical B20和B2切片随机数重建；RL不继承这些常数，
  只继承同一Writer输入输出和Direction Store参数化。

首轮若失败，不继续搜索K、聚类seed、top-k、route temperature、load-balance loss或
LIBERO手写语义标签。那类调整只能优化当前任务分桶，不能回答根因。下一步必须回到
条件表示如何形成可组合生成方向、完整decoder权重如何参数化，或reward/functional
credit如何到达这些方向。

## 3. 为什么SFB没有形成真正的方向存储

SFB把每个factor head的`1024→256`切成四个`1024→64` value bases，用Core route
缩放四段hidden，再经过一套共享`256→output_width`：

```text
h_b = GELU(W_b Z), width64
y   = W_out concat(alpha_b h_b)
```

它的总factor参数量与普通hidden256 head相同。不同task虽能选择不同activation，
但仍共享：

1. 每个basis只有64维；
2. 最终完整factor output matrix `W_out`；
3. video-dependent target Core产生的route位置；
4. 所有task的factor optimizer state。

所以它解决的是activation routing，不是完整生成方向的parameter coexistence。
继续增加basis数只会进一步切窄每个basis，不能检验当前根因。

## 4. 选择：冻结语义地址上的八个完整Direction Stores

### 4.1 稳定、video-independent的任务地址

同一exact task language额外经过一次**无Meta-LoRA、无梯度**的frozen PaliGemma
text-only forward。只取authoritative task span的末层hidden并做token mean。train24
language的归一化anchor均值作为固定公共方向`mu_train`，路由时先减去该方向再L2
normalization：

```text
r(T) = L2Norm(Mean_token(H_frozen_text(T)))
a(T) = L2Norm(r(T) - Mean_train24(r(T))) ∈ R^2048
```

该地址和冻结的`mu_train`：

- 只来自部署时已有的task language；
- 不读task ID、suite、video、action或outcome；
- 对同一task跨teacher video严格相同；
- 不随Writer checkpoint变化；
- 使用foundation contextual language geometry，而不是手写verb/object标签；
- `mu_train`只由预封存的24个train language建立，不读取validation/test语言或outcome，
  去除的是LIBERO指令共同句式方向，避免全部task挤向同一个store。

现有带Text Meta-LoRA的`Q_text`仍完整服务Semantic Core和视频grounding；冻结地址
只负责选择参数存储，不能作为factor value或LoRA residual。

### 4.2 只用24个train language建立固定中心

在任何训练或validation outcome产生前，对24个train task的`a(T)`做seed7确定性
spherical k-means，`K=8`：seed7只抽取第一个anchor，后续七个初值逐次取与既有
中心最大cosine最小的train anchor（相同值取最低ordinal），再做Lloyd更新直到assignment
不变、上限100轮。中心和route audit写入小型config authority；不读取
validation/test语言来拟合中心。runtime对任意language取cosine最近的两个不同中心：

```text
I(T) = Top2_k cosine(a(T), center_k)
route weights = [0.5, 0.5]
```

八个stores和top2的理由：24个train tasks产生48个store assignments，平均每个store
约6个task，既给每个store多task训练和validation组合泛化，又显著低于“一task一专家”。
等权top2避免learned gate、temperature、route amplitude和load-balance loss成为新的
漂移来源。实现前的text-only audit只检查每个store至少有primary train owner、top2
usage不过度塌缩以及相似language具有可解释共享；不根据rollout结果改中心或K。

### 4.3 每个store拥有完整factor生成参数

保留每个target/rank的完整value：

```text
Z[t,r] = concat(Core_t, Read_A[t,r], Read_E[t,r], Read_D[t,r]) ∈ R^1024
```

对八个public factor families，每个direction store拥有完整独立的：

```text
F_e(Z) = W_out[e] GELU(W_in[e] Z)
W_in[e]:  1024 → 256
W_out[e]: 256  → factor_width
```

最终只计算并组合该task选中的两个stores：

```text
factor(Z,T) = 0.5 F_i(Z) + 0.5 F_j(Z),  {i,j}=I(T)
```

关键区别：

- 每个store都有完整256 hidden和独立`W_out`，不再共享最终生成方向；
- 同一个task的全部38 targets、16 ranks和8 factor families共用同一top2 store集合，
  保持整套LoRA coherent；
- target/rank/video差异仍只通过完整`Z`进入value，语言地址本身不能生成LoRA；
- 相似任务可共享一个或两个stores，未见任务可组合两个最近语义中心；
- 训练只触及当前task的两个stores，其余store的该task梯度严格为零。

使用batched gathered expert weights，只执行top2计算，不在forward中顺序运行八个完整
heads。各store的`W_out`均exact-zero；step0输出仍为template A / zero B。第一macro
按既有identity lifecycle只打开所选store outputs，随后梯度才能到其input、Core、
Program与semantic frontend。

## 5. 参数、显存与BCI执行

当前Target-Bound/SFB非factor参数为`7,340,288`；一个完整1024→256 factor family
集合为`3,751,936`。八个完整stores的预计trainable总量为：

```text
7,340,288 + 8 × 3,751,936 = 37,355,776
```

实现后必须用真实module enumeration确认。新增约26.2M参数是方向存储本身，不扩大
Core/Program、public LoRA rank或source policy。top2 activations远小于八路dense计算；
主要新增显存来自Writer参数、gradient和Adam state。

BCI保持logical B20，但继续把frozen-policy forward物理切成B2。新增一个
task-query-keyed **independent** Beta(1.5,1) time与independent Gaussian noise sampler，
只为在B2切片中精确重建同一logical-B20独立样本；不得复用已负裁决的Latin/antithetic
VR estimator。6 ranks×4 tasks、full24 raw equal mean、一次clip/AdamW/scheduler和
96,000-query首段均不变。若最长105-frame真实profile OOM，先降低Writer frame encoder
microbatch；policy B2、logical B20和科学样本量不变。

## 6. 训练选择与被拒绝的替代

首个formal保持SFB的RAW full24、B20、fast-decay400 optimizer schedule，从fresh
identity训练。新架构本身已改变parameter ownership；首跑不再同时加入expert-wise
loss重权、route load-balance、task-ID supervision、gradient projection、serial Adam、
contrast、reward或辅助一致性loss。

这里保持旧AS recipe只是为了隔离架构变量，不表示Direction Store依赖统一full24
监督下降。若后续进入RL，固定anchor mean/centers/top2，关闭action supervision，采用
task-balanced rollout、per-task advantage/return尺度控制和per-store访问/梯度/更新审计；
reward稀疏导致的store starvation应先由采样与credit assignment解决，而不是修改router
来追逐当前reward。

不选择：

- **dense hidden加宽**：增加容量但所有task仍共享同一个输出方向矩阵；
- **learned video/Core MoE gate**：route会随video和checkpoint变化，重新制造待解释漂移；
- **task-ID hard experts**：违反信息墙且不能泛化validation/test；
- **一task一store**：等价记忆训练split，不能证明组合泛化；
- **静态LoRA dictionary**：language系数可绕过视频value，重演static bypass；
- **输出rank正交/谱约束**：Target-Spectral已证明破坏有效coherent高增益流形；
- **few-shot或LoRA平均**：owner已固定继续one-shot。

一个可信相邻方案是learned top2 full-capacity MoE。它参数存储充分，但gate本身可漂移，
会把“完整方向存储是否有效”和“router是否稳定”混成一次实验。固定foundation-language
centers更直接检验当前假设，故作为第一候选。

## 7. 最短实现与证据路径

保留一个`CompleteLoRAWriter`、一个训练入口和一个evaluator。原位替换：

- `video_program.py`：text-only branch同时返回trainable`Q_text`与无梯度frozen anchor；
- `program_compiler.py`：删除SFB router/head，拥有fixed-center top2 route和gathered
  full-capacity direction-store factor head；
- `model.py`：传递task anchor并用一套top2 route生成全部76 tensors；
- `architecture.py` / `as_config.py`：fresh schema、37,355,776参数和route authority；
- `functional.py`：只增加切片可重建的keyed independent randomness；
- 现有checkpoint、task-gradient、training、validation和evaluation owner继续复用。

不新增第二套model、runner、optimizer或evaluator。旧SFB runtime classes/config schema
随canonical替换退役；历史checkpoint只由frozen commit执行。

聚焦CPU/小张量验证：

1. frozen anchor不随video、Writer参数或train/eval mode变化；
2. centers只由24 train languages建立，top2 distinct且route可重复；
3. language相同的视频共享完全相同store IDs；
4. 只有所选两个stores获得该task factor gradient；
5. 每个store拥有独立input/output参数，没有共享`W_out`；
6. zero coordinate不能由route凭空生成factor；
7. step0 76 tensors与identity template逐tensor一致；
8. source policy trainable=0，validation/test actions不进optimizer；
9. keyed independent B20一次forward与B2×10在samples/loss/LoRA-leaf gradient上相符；
10. fresh0→1、exact-resume1→3与最长105-frame真实六卡macro通过。

前九项及fresh checkpoint family的CPU合同已通过；第十项必须由clean pushed commit
在BCI空闲卡上完成。center authority由frozen source policy对24条train language做
text-only forward后建立；减去train24 raw-anchor均值再归一化，seed7 spherical
k-means两轮收敛。primary/top2访问计数分别为`5/7/6/1/2/1/1/1`与
`7/11/6/4/4/5/3/8`，当前不再调K、seed或centers。

只运行与上述新职责直接相关的focused测试、compile和真实vertical path；不重复全量
artifact hash、历史root扫描或无关旧回归。

## 8. 首小时裁决与内部预测

fresh0→200保存every25，并对50/100/150/200做同一paired correct400。只有
absolute、breadth、右端趋势或明确的共同能力累积支持时才exact-resume到400。

主行为判据：

- single checkpoint严格超过`150/400`是最低目标，不是自动终点；
- breadth、逐task分布和top-task集中度不能靠Goal-6/Object两项独占；
- 相邻checkpoint gained/lost、Jaccard和union-to-single envelope必须较SFB改善；
- 只选择一套single-checkpoint LoRA，不融合checkpoint或挑video。

预注册内部预测：

1. route IDs对同task跨video/checkpoint严格不变；
2. selected stores的factor gradient能量分散，不再由一套共享`W_out`吸收全部task；
3. 共享0/1/2 stores的task pair应呈现可解释的梯度相似性层级；
4. 每个store内部同task successive gradient cosine与mean/sample energy retention应高于
   ordinary SFB factor block；
5. 两个selected store contribution都非零，不能长期由一个store包办或互相抵消；
6. Core/Program/A/E/D反事实继续传到effective BA/action，same-task video变化不能归零；
7. effective LoRA norm、near-rank1 coherence和q/v跨层共同方向不能发生
   Target-Spectral式坍缩。

若固定route和完整独立stores均按合同工作，而correct仍未超过v6-fast `143`、
success集合继续大幅轮换，或store内部梯度仍近正交，则“factor parameter coexistence是
主要缺口”被否定；下一步应改变训练目标或更上游的条件表示，而不是增加stores、改K、
加入load-balance或在失败checkpoint上加gate/scale。若absolute提高但视频路径退化，
则这是language-conditioned static adaptation，不是合格EMBER结果。

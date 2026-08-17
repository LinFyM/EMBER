# Layer-Matched Memory Program Compiler

状态：2026-08-17 **本轮设计goal最终提案，不自动授权实现或GPU实验**。简称暂定 **LMMPC**。本文是在仓库统一
整理和stage-wise evidence裁决之后形成的完整架构判断；它不是已经取得结果的active method。

## 1. 一句话决策

LMMPC用exact language形成任务Query，用每条action-hidden有序video在真实π0.5 image/language/Action context中
读出的逐层memory states形成唯一动态Value；每video先通过forward-minus-reverse编成反转反对称Program，多个videos
再在Program层做置换不变且保符号聚合；最终把共同Program放进V6已经验证的`policy layer × LoRA rank`坐标，由
一套共同训练的native rank16 factor compiler生成完整38-target A/B LoRA。训练同时用correct跨episodefunctional
credit和language-directed Program matching，分别解决policy方向与positive-only不可识别性。

它不是：

- 在GOMQ后继续追加rank32 B-only residual；
- 把memory仅作为LPCP旧Procedure Query的小修；
- 复制SHINE的flat payload大小或四轴全注意力；
- 让language/static Core直接提供LoRA Value；
- 把生成LoRA后的task-local RL混进当前zero-interaction分数。

## 2. 为什么选择这一接口

现有证据把两个极端都测过：

1. LPCP保留V6强support，但新layerwise Procedure经过冻结tail后相对AS139只改`.002653`，属于邻域小修；
2. GOMQ绕过tail直接写rank32 B，第一次learned-memory update到151，却连续回落到135/131；
3. Dynamic-K stage probe显示M2P/final Program跨task结构仍在，family hidden和B端首先明显变同向；
4. SJNV显示同task hidden residual约`.99`一致，过冻结factor W2后raw-factor可跌到约`.05`；
5. V6/v6-fast和task experts已经证明native rank16本身可以policy-effective，不能把输出维度或低stable rank当首因。

因此下一架构要解决的不是“再读一次视频”，而是：

```text
ordered video evidence
    -> layer/rank-matched Program
    -> jointly learnable native A/B commitment
```

同时，GOMQ的reward梯度在不同task间低coherence且不能选择gained/lost；第一轮必须回到dense functional credit，
避免把新compiler和新reward混成两个变量。

## 3. 完整部署数据流

```text
exact task language
+ K same-task action-hidden ordered videos, stride 5
  (first claim: K in 1..4, training covers every K)

exact language -> Language_slot
per-video order-invariant semantic summaries
  -> permutation-invariant K-set readout
  -> shared task-grounded Core (Query only)

for each video independently:
  for each sampled frame:
    native image + exact language prefix
    + 50 fixed π0.5 Action-probe states
    + 16 one-way learned rank-memory queries
      -> one native context encoding
      -> memory states from all 18 Action-Expert layers
         H[k,t,l,r] in R^1024
  ordered adjacent deltas + endpoint change
      -> shared causal reduction on forward and reversed cached states
      -> 0.5 * (forward - reverse)
      -> P_video[k,l,r] in R^256

K videos:
  language/Core-routed, permutation-invariant common-Value aggregation
      -> P_set[l,r] in R^256

policy topology:
  18 x 16 expert slots
  + 16 action-in slots from the layer-0 boundary
  + 16 action-out slots from the layer-17 boundary
      -> 320 video-Value policy slots
  task-grounded Semantic Core supplies Query/gates only
      -> one policy-slot mixer
      -> eight jointly trained native factor-family heads
      -> one complete 38-target rank16 LoRA

frozen π0.5 source policy:
  unseen initialization closed-loop rollout
```

Writer在rollout前只运行一次。teacher actions、proprio/state、reward、terminal、task ID、filename、pose和hidden
normalization均不进入部署Writer。

## 4. Native context与memory放置

### 4.1 有意义的prefix

每帧仍使用π0.5原生语义：真实image、exact language和50个固定noise/time Action probes。Action probes不是teacher
action；它们只问冻结policy“在这幅画面和任务语言下，动作层内部如何理解当前交互”。

memory observer复用这一次prefix/Action计算的逐层K/V，不运行zero-image、blank-language或memory-only Action
Expert。native prefix和Action states看不到memory，因而source carrier数值不被新token改变；memory block可以读取
完整native context和同block的rank memories。

### 4.2 为什么是16个memory queries

16不是目标约束，也不是视频阶段数。当前public LoRA为rank16，V6 compiler已有每个Action-Expert layer的16个rank
slots；因此使用16个共享rank queries，让第`l`层第`r`个memory state自然对应第`l,r`个policy slot。

若未来rank改变，query数随rank改变。它不同于GOMQ为flat B payload容量匹配得到的37个tokens，也不同于为38个
targets各建一套独立token。参数跨层共享，layer identity和rank identity只作address。

### 4.3 一次context，不重复内容forward

实现可沿用GOMQ已经通过的one-way observer：native prefix/Action context完整计算一次，逐层memory stream读取已捕获
K/V。activation checkpoint只重算memory stream，不重复image/language/Action content forward，不让policy参数获得
gradient。

### 4.4 Backbone与Meta-LoRA权限

冻结π0.5 source backbone始终没有gradient。bridge直接复用sealed LPCP的context计算并保持其VL/Action Meta状态
冻结，避免把carrier变化混进Program→compiler裁决。

最终fresh recipe不新增trainable VL Meta-LoRA：真实image/language prefix直接使用冻结VLM表示，避免重开v4曾暴露的
raw-image/VL-Meta旁路。保留一套zero-init Writer-local Action Meta-LoRA，因为memory需要在原生Action Expert交互
坐标中学习怎样提问；它只改变Writer内部Action/memory context，不直接输出LoRA，也不挂载到部署policy。若Action
Meta被消融后无功能贡献，应删除而不是因形式保留。

## 5. 每条video内部的反转反对称有向Program

只让网络“有能力看到顺序”仍不够。若训练输入始终是correct、同task functional target又对所有videos相同，模型仍
可忽略causal block，只把video当task identity。LMMPC因此不把普通causal Transformer输出直接交给compiler，而先
构造一个对时间反转严格反对称的Program。

先把memory state共享投影到256维，并在当前输入顺序内重算变化：

```text
R[t,l,r] = W_memory(H[t,l,r])
Delta[0,l,r] = 0
Delta[t,l,r] = R[t,l,r] - R[t-1,l,r]
Endpoint[l,r] = R[T,l,r] - R[0,l,r]
```

定义一个共享的方向读取器`F(sequence, Core)`：

```text
Core/route Query在(l,r)轴汇聚每个t的Delta Value -> c[t]
c[0:t] -> causal temporal block -> z[t]
每个(l,r) Core/route Query从{z[t], Delta[t,l,r]}读取自己的有向变化
+ zero-bias endpoint projection
-> F(sequence, Core)[l,r]
```

causal位置只使用**重排后重新编号的序列位置**`0..T-1`；不输入原视频timestamp、原frame ordinal或normalized
absolute phase。然后在已经缓存的同一组`R`上运行两次轻量方向读取器：

```text
P_video(V) = 0.5 * (F(V, Core) - F(reverse(V), Core))
```

这不重复image/language/Action backbone forward，只重复远小于backbone的temporal reducer，并得到结构恒等式：

```text
P_video(reverse(V)) = -P_video(V)
```

因此任何对frame集合、静态外观或video presence不变的分量都会被消去；constant或no-video严格给出零Value。shuffle
必须先真实重排frames、重新计算Delta并重新编号，再计算自己的forward-minus-reverse Program，无法继承correct的
transition。该结构不能保证correct闭环一定更好，但它阻断“完全忽略方向仍得到同一Program”的旁路；正确符号最终
仍由functional loss和严格controls裁决。

## 6. 多video集合聚合

每条video先得到`P_video[k,18,16,256]`，之后才允许K轴通信。对每个layer/rank slot：

```text
mu = Mean_k(P_video[k])
centered[k] = P_video[k] - mu
dispersion = Mean_k(centered[k] ** 2)
even_gate = 1 + tanh(W_dispersion(dispersion)) * tanh(W_core(Core_slot))
odd_correction = W_out Mean_k(tanh(W_centered(centered[k])))
P_set = even_gate * mu + odd_correction
```

性质：

- K顺序严格置换不变；
- 所有映射zero-bias；K=1时`dispersion`和`odd_correction`为0，因此`P_set=P_video`；
- mean只发生在已经保序编译的高层Program，不平均frames、raw backbone features或最终LoRA；
- dispersion只调节共同Program的可信度，centered odd correction处理跨demo偏差；所有videos等权进入统计，不挑一条；
- 若所有videos同时reverse，`dispersion`不变而`mu/correction`反号，因此`P_set`也严格反号；
- Core说明关注对象和目标，但所有被聚合的additive Value仍来自video。

首个训练合同真实覆盖K1/K2/K3/K4；架构数学上可以接受更大K，但在训练覆盖前不声称对未见cardinality泛化。

聚合后建立320个policy slots：18×16 expert slots直接使用`P_set[l,r]`；16个action-in和16个action-out slots分别由
layer0和layer17边界经zero-bias projection得到。这里不为action边界另外复制一套video encoder。

## 7. Language/Core与video Value的融合

保留V6 task-grounded Semantic Core，因为其对象、关系和目标语义是143架构的重要组成；删除它会重复Dynamic-K
100附近的absolute损失。每条video先独立产生order-invariant semantic summary，K轴只在这些高层summary上作
permutation-invariant set readout，再与exact-language summary形成shared Core；不跨video平均raw patches。因为该
Core只看frame set，所以correct/reverse/shuffle共享同一Core。

改变的是Core权限：它只生成320个slot Query、route和乘性gate，不提供additive factor content。另保留一份只由
exact language形成的`Language_slot`，既控制language必要gate，也作为训练时language—Program matching的anchor，
避免matching两端共同读取同一个video static identity。

一种同时保持信息墙和方向性的实现是：

```text
QK_slot = Norm(Core_slot + policy_route)
language_gate = tanh(W_language(Language_slot))
S0 = W_value(P_set_320) * language_gate
A = Softmax(Q(QK_slot) K(QK_slot)^T)
S1 = S0 + A V(S0)
S = S1 + W2 tanh(W1 RMSNorm(S1))
```

全部Value/FFN映射zero-bias。若video dynamic Value为0，或language被移除并显式得到零`Language_slot`，则`S=0`；
language、route、静态Core或first frame不能独立写LoRA。Q/K只由对frame集合置换不变的Core与policy route决定，video
只进入V；所以所有videos reverse时attention routing不变而`S`严格反号。

policy轴只在这里通信一次。第一版复用V6已经验证的320-slot ownership和full-slot通信职责，但把旧standard block
改为上述zero-preserving Query/Value分权形式；不同时引入SHINE式row/column M2P。
若将来层数/rank扩大到full attention成本不可接受，再把同一职责机械替换为layer/rank axial block。

## 8. Rank16 LoRA生成

slot布局沿用V6：

```text
expert:    18 layers x 16 ranks
action-in: 16 ranks
action-out:16 ranks
```

八个共享、zero-bias native factor heads分别生成：

```text
q_A, q_B, v_A, v_B,
action_in_A, action_in_B,
action_out_A, action_out_B
```

每个slot生成对应rank row/column，按38 targets的native ownership组装一套完整rank16 LoRA：

```text
deltaA(S) = h_A(S)
deltaB(S) = h_B(S)
A(S) = A0 + deltaA(S)
B(S) = deltaB(S)                                # B0=0
```

`h_A/h_B`沿用V6各family的input/output ownership和bias-free MLP形状，但从新LMMPC分支共同训练，不加载冻结W2。
无video时仍为正常随机`A0`和exact-zero `B0`，effective BA为identity；A由video/language共同生成而不是fixed-A，避免
重走rank8 fixed-subspace路线。

这里**不**强制`LoRA(reverse)=-LoRA(correct)`。反转反对称只约束Program，保证direction是compiler可见且不可被
static旁路替代；correct LoRA由functional loss训练成有用方向，reverse/wrong/shuffle/no-video是否更差必须由真实
paired controls证明，不能靠输出端硬编码破坏negative adapter。stage diagnostics会检查Program符号在factor/BA/action
何处被抹掉，若joint head仍把`S`和`-S`映成近同一方向，则该compiler终局失败。

factor heads在最终fresh recipe中与memory/compiler共同训练，不把新Program塞进冻结W2，也不绕过为flat B rows。
它们的作用是共享“同一类policy factor怎样由slot生成”，而不是让所有tasks共享同一LoRA Value。

## 9. 为什么它不是失败的类SHINE复刻

LMMPC继承成熟Hypernetwork的层对应和结构化生成原则，但没有照搬：

- 旧Dynamic-K Backbone-Memory去掉V6 absolute Core、使用rank8/flat mapper，strict约100；LMMPC保留V6 Core作Query、
  native rank16 ownership和task-complete recipe；
- GOMQ的37-token grid直接拥有680,448个B values，再追加rank16 bank；LMMPC的16 queries对应rank slots，最终只有
  一套rank16 A/B；
- [SHINE](https://arxiv.org/html/2602.06358v3)的M2P需要在layer/token轴重排足够大的flat payload；LMMPC已有V6的320 policy slots，只在一个明确policy
  stage通信；
- [Doc-to-LoRA](https://arxiv.org/html/2602.15902v1)逐层用rank latent queries生成A/B的原则更接近本接口，但EMBER额外加入video内有向过程和K-set聚合。

## 10. 训练合同

### 10.1 两个不可互相替代的训练信号

- development gradient只来自固定train24；
- 每macro覆盖24 tasks且task等权；
- K1--K4各6 tasks，连续四个macro内每task轮到每个K；
- 每条video独立保序，video与action episode同task但跨episode；
- source policy、normalization和split冻结；
- 不加reward、expert reconstruction、LoRA geometry、subset consistency、gradient surgery或额外数据。

第一个信号是现有dense functional objective。每task用冻结source policy处理B20个同task跨episode action queries，
只在correct Program生成的LoRA上下降functional loss。它负责回答“这套LoRA是否沿policy-useful方向”，不能被
representation loss替代。

第二个信号是**语言—有向视频匹配**，专门处理positive-only错开监督的不可识别性。对exact-language anchor `j`和
当前task `i`的Program定义slot-wise匹配分数：

```text
score(j | P_i) = Mean_slot cosine(Wq(Language_j[slot]), Wk(P_i[slot]))
```

`Wk`线性且无bias，所以reverse Program反号时匹配分数也反号。24条固定train language在每个rank一次性形成很小的
anchor table；每个task/video仍可按现有task-complete顺序forward/backward，不保留其它video backbone计算图。loss由
两部分组成：

- `L_task_match`：每条same-task correct `P_video`及`P_set`都应在24条language anchors中分类到自己的exact language，
  其它23条language自然成为cross-task negatives；
- `L_order_match`：对同一exact language，correct Program应高于从同一真实frame set重算的shuffle和reverse Program；
- 不读取shuffle/reverse或wrong condition的action、reward、outcome，也不规定negative LoRA必须失败。

这与历史“把negative adapter人为推坏”的margin不同：matching只要求exact language不能与任意video过程互换；真正
有用的policy方向仍只由correct跨episodefunctional loss提供。因为Program已先去掉reversal-even/static分量，matching
不能只靠first frame、原timestamp或video-presence token完成。

总loss为`L_functional + lambda_match * L_match`。`lambda_match`不做sweep：在首个task-complete calibration macro用
初始loss值把matching标量贡献固定为functional的10%，之后冻结并写入run contract。same-task Program/BA consistency
仍只作诊断，不直接加variance loss；历史已经证明强推variance可能降低closed-loop。

### 10.2 开发桥接与最终fresh的边界

先做一个受控bridge，避免直接用大规模fresh run掩盖接口错误：

```text
sealed LPCP143 rank16 factors (A_base, B_base)
+ zero-forward LMMPC factor residuals (deltaA, deltaB)
-> A = A_base + deltaA, B = B_base + deltaB
-> one rank16 LoRA
```

- 新分支完整包含rank-matched memory、temporal/K-set、policy-slot mixer和八个native rank16 residual factor heads，
  不再经过LPCP旧Procedure Query或冻结W2；
- residual heads前的单一zero-output gate保证第0步逐tensor等于LPCP143；constant/no-video新增residual恒为0；
- Program matching从首步训练memory/temporal/K-set；functional gate首步打开，随后slot mixer与residual heads也获得
  gradient；
- LPCP底座完整冻结，只训练新增LMMPC分支，用相同B20 functional加上述matching；不追加rank32 bank；
- bridge只能回答“新的Program→native rank16 compiler能否material且不破坏support”，不能作为最终
  language-only/no-video或fresh recipe claim。

bridge通过后，才建立fresh formal：相同LMMPC拓扑从random Writer initialization训练，Core、memory、slot compiler和
factor heads共同学习；不依赖LPCP checkpoint、optimizer或task能力换手。bridge失败时先按stage定位，不用fresh为它
辩护。

### 10.3 Reward边界

若fresh joint训练达到强且相邻稳定的平台但仍明显低于目标，可另立Writer-RL authority。它仍输出同一套LoRA，
保持信息墙和task balance。当前设计不包含reward，也不包含生成LoRA后的task-local RL。

## 11. 机制门与stage diagnostics

硬合同只覆盖真实工程/信息墙：

1. native prefix/Action只编码一次；source policy参数0 gradient；
2. step0 bridge与LPCP exact identity；final fresh的constant/no-video为identity；
3. 去掉language或video任一方时新增fresh路径为零；不能只用language加generic video-presence gate；
4. memory gate打开后，query、temporal、K-set、slot residual和八factor heads均有finite nonzero gradient；
5. 在同一缓存memory上直接验证`P(reverse)=-P(correct)`、set/slot同样反号，并确认factor/BA/action没有把符号
   完全抹成同一输出；不预设LoRA必须严格反号；
6. shuffle从真实frames重排、Delta重算和位置重编号后material改变Program；
7. K置换只允许正常BF16 reduction低位差异；
8. q/v/action-in/action-out均有native BA和fixed-action response；
9. longest-video与额外temporal shuffle arm无OOM/nonfinite，batch按真实samples/s选择。

下面是科学定位，不设成僵硬20/20门：

- per-video Program、K-set Program、fused slot、factor、BA、fixed action逐stage报告same-task alignment；
- 同时报告validation8 between-task mean cosine/effective rank，不能只追求within-task高cosine；
- 报告train24与validation8的language→correct Program retrieval，以及相对wrong/shuffle/reverse的匹配差；若只记住
  train24而held塌缩，identifiability路径不成立；
- 若Program健康而factor/BA再次出现数量级coherence或幅度断裂，立即否决当前compiler；
- 若BA/action material而strict gained/lost仍不可分，问题后移到functional credit/held occupancy；
- 若correct/reverse差异只在latent、不沿正向functional/closed-loop方向，不能称为视频因果学习。

## 12. 训练量、strict与稳定性裁决

bridge在完整full24机制通过后训练到预注册macro25，立即做同一single checkpoint K4 strict paired400。结果按证据
解释，不用一个绝对硬门机械代替判断：

- 明显低于强基线且Program→BA链路已坏：终止，不扫token/rank/LR/seed；
- absolute一般但stage链健康：只允许一次有因果理由的后继，不用小补丁堆救；
- 约140以上且retention/breadth有改善：继续一个相邻训练节点判断积累，不把早期峰值当答案；
- 首次约145或更高且retention合理：立即封存并补correct/same-task-other/wrong/shuffled/reversed/no-video六臂，同时
  继续相邻checkpoint；
- 约145稳定、same-task鲁棒且correct controls合格，即使未超过150也可成为有价值结果。

正式比较报告per-task、per-suite、breadth、retained/gained/lost、churn和success-set Jaccard，并对照V6/LPCP143、
v6-fast五臂、old134/compiler138/online128。checkpoint union、挑task checkpoint或多checkpoint平均均无效。

## 13. 快速否决与负结果边界

当前LMMPC应在下列情形终止：

- matching可被static identity/video presence解决，或constant/no-video不为零；
- 反对称Program在held videos近零，说明去除even分量也去掉了policy需要的信息；
- K聚合提高same-task一致却进一步压平between-task结构；
- slot Program过factor head再次出现SJNV式断裂；
- rank16 BA material但bridge显著破坏LPCP support且无breadth/gain；
- joint task-complete训练仍复现151式单点峰值和相邻持续回落；
- correct不优于wrong/shuffle/reverse/no-video，或same-task-other明显失稳。

负结果只淘汰：

```text
16 rank-matched one-way memories
+ forward-minus-reverse directed per-video Program
+ permutation-invariant odd K-set aggregation
+ Core-as-Query / video-as-Value slot fusion
+ jointly trained native rank16 A/B factor compiler
+ dense functional + language-directed Program matching
```

它不否定memory token、其它rank、few-shot、LoRA生成或未来Writer-RL一般。

## 14. 可扩展性

LMMPC不为每个LoRA标量创建token。若基础policy有`L`个被观察层、公共rank为`R`、factor families为`F`：

- 每帧输入仍只有`R`个memory queries，逐层收集后得到`L×R` states；
- policy slots随`L×R`线性增长，action边界只增加`O(R)`；
- temporal与K-set模块跨layer/rank共享参数；
- factor head按family共享，output width随对应原生module维度增长；
- K只增加video集合轴，不生成K套LoRA，也不增加deployment adapter数量。

因此Writer可随backbone layers、target families和rank扩展，而不是像flat payload grid一样让token数直接追赶全部参数
元素。若`L×R`使320-slot full attention成为瓶颈，只机械替换成layer/rank axial attention，信息职责和训练合同不变。

## 15. 实现与效率边界

若owner确认后进入实现：

- 原位替换terminal GOMQ runtime，保持一个canonical Writer，不保留第二套可执行版本；
- `backbone_memory.py`现有observer已经真实返回prefix/action states与`[frame,18,37,1024]` layer memory；保留其
  one-way mask、native context capture和deferred memory stream，但把hard-coded 37容量tokens原位替换为16个rank
  queries，不保留37-token compatibility分支；
- 复用V6 semantic encoder的per-video memories与320-slot ownership；新增的language-only slot anchor必须与
  task-grounded visual Core分开，不能从同一个video summary复制；
- `legacy_v6_model.py`已有38-target ownership、A0/B0 template和八种factor widths；复用schema与head形状，但新heads
  属于LMMPC并共同训练，不加载LPCP冻结factor W2；
- 删除37-token capacity grid、rank32 public bank和reward-only接线，不叠加compatibility fallback；
- internal reverse与training shuffle只在缓存layer-memory上重复轻量temporal reducer，不重复image/language/Action
  forward；八factor heads不再为`S/-S`重复运行；
- 24条exact-language anchors每rank只编码一次；每个current Program直接做24-way匹配，不保留其它video计算图、不做
  cross-rank Program all-gather，也不产生额外policy rollout；
- 不增加重复forward、batch1、FP64训练、逐tensorhash或大规模防御性扫描；
- 多卡按K、总frames和历史wall动态平衡，task权重不变；
- GPU launch前另按live双节点、quota和formal launch合同执行。

当前没有active successor、没有GPU授权，也没有需要resume的checkpoint。完成本文不等于授权下一阶段执行。

## 16. 下一阶段需要裁决的三个核心点

这些不是token数或小超参，而是会改变科学含义的选择：

1. **Directed Program**：是否接受在缓存memory上显式计算`0.5*(forward-reverse)`，用结构消去static/order-even
   bypass；我的判断是接受，但不把反号继续硬编码到LoRA输出。
2. **Identifiability supervision**：是否接受language—Program matching只约束表示相符、negative arms不读
   action/reward且不直接训练negative LoRA；我的判断是这是比“只输入correct”更完整、又不靠破坏controls的必要
   训练信号。
3. **Bridge到fresh**：是否先用sealed LPCP143+zero-forward rank16 residual branch裁决support，再以相同LMMPC拓扑
   从fresh Writer训练最终方法；我的判断是需要bridge隔离接口，但论文方法与最终分数只认fresh recipe。

这三点确认后，本文才从最终设计提案升级为active implementation authority；其它module width、head数和精确LR应在实现前按
现有V6/GOMQ规模机械确定，不再作为架构讨论反复摇摆。

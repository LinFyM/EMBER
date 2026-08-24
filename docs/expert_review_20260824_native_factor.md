# 核心裁决

我确认远程 `main` 正好锁定在 `7ab5a04d920b0ec63843b89afb37b4ad1d137623`。仓库已经把人工 process 分支正式关闭：最后一次 recovery teacher Gate 为 `9/50 + 5/50 = 14/100`，相对 A3 的 37 个成功只保留 14、没有新增、丢失 23；工程、route、pairing、seed 和公开视频信息墙均正常，因此这是明确的科学 non-pass。owner 也已经把“不再制作 custom task、temporal wrapper 或人工 process 数据”写入稳定边界。

我的总体判断是：

> **ECP 仍值得做，但当前最深卡点已经不是缺少一个更好的 `q_pi`、solver 或低秩坐标，而是：如何让自然视频产生的、task-specific policy-native信息，直接决定那些能在未见occupancy上起作用的低能量LoRA方向。**

当前架构还没有真正解决这一问题：

1. Stage 0 保留了 layer 和 horizon，但当前 q/v owner 实际仍来自同一层的 input state 与 residual，只用 family embedding 和 gate 区分；它没有读取真实 q_proj/v_proj 的输出空间。因此“owner correspondence”仍主要是索引对应，而不是完整的 target-native parameter correspondence。
2. 当前 language 只是跨帧平均的全局 summary，scene 只是 first/final/difference 的全局 384 维量；尚未成为贯穿编译链的 owner-specific `P_lang[38,128]` 和 `P_scene[38,128]`。
3. 两种原则化 shared realizer 已经表明，**将 held policy innovation 投影回 fit-task 共享坐标**会丢失大量闭环支持：

   * balanced-SVD learned realizer 只有 `33/37`，低于 carrier 43，breadth 2/5，Goal/Long 为 0；
   * centered two-sided coordinate 虽保持 `.877–.960` 的整体 update cosine，仍只有 80/250、breadth 3/5，并把 Goal/Long 全部清零。
4. 另一方面，15/15 known-success paths 在现有 owner/flow/action effect objective 上都严格单调改善，说明 policy-effect space 仍然是有用的训练坐标，只是不适合作为一个必须先预测、再由固定共享逆映射还原LoRA的部署中间层。

因此，我会对仓库目前保留的未来设计做一个重要修改：

> **取消 canonical `q_pi(P) → fixed effect-code realizer → LoRA` 前置链。**
> 最终主线只保留一个视频Program encoder `q_V`；Program通过一个直接借用π0.5各目标层原生输入/输出激活的 **native-factor compiler** 生成LoRA。
> privileged expert/effect evidence只作为训练时的set-valued critic，不再生成另一个任意latent Program，也不进入部署forward。

这不是恢复旧LMMPC或FactorHeads。新的核心区别是：

* 不从128维latent凭空生成1024/2048维参数向量；
* 不把held更新投影进fit-task固定span；
* 不给LoRA rank赋技能语义；
* 每个LoRA因子的行、列空间直接来自该condition下π0.5目标模块自己的原生激活；
* Program决定的是“从这些policy-native向量中选什么、如何组合和缩放”。

我把这条主线称为：

# **ECP Native-Factor Compiler**

---

# 一、最终建议架构

## 1. 总体forward

```text
exact language + K条action-hidden ordered videos
        │
        ├── Pass A：frozen PI0.5 native observation
        │       └── owner-specific language / scene / ordered-event Program
        │
        └── Pass B：同一批frames上的target-native factor readout
                └── Program-conditioned signed pooling
                        └── task residual rank4
                                └── 与frozen shared carrier rank12拼接
                                        └── 唯一一套38-target rank16 LoRA
```

这是两次读取同一个冻结backbone，但不是两个独立语义分支：

* Pass A 决定“视频表示了什么任务、场景和过程”；
* Pass B 使用同一个Program去π0.5各LoRA目标的原生输入/输出空间中读取参数因子；
* 两者共享owner、event和video assignment，不存在一个独立video model和一个无关hypernetwork最后拼接。

Writer仍只在rollout前运行一次。两次backbone读取的计算成本可以接受，并且可以复用image/language prefix cache。

---

## 2. Language编码

exact language只经过冻结PaliGemma/Gemma原生语言路径。

设原生语言tokens为：

[
H_L\in\mathbb R^{N_L\times 2048}.
]

建立38个固定owner queries：

[
Q_L\in\mathbb R^{38\times128}.
]

每个owner独立读取语言：

[
P_{\text{lang}}[j]
==================

\operatorname{CrossAttn}
\left(
Q_L[j],\Pi_L(H_L)
\right),
]

得到：

[
\boxed{
P_{\text{lang}}\in\mathbb R^{38\times128}
}
]

其中38个owner严格对应：

* 18层 q_proj；
* 18层 v_proj；
* action_in_proj；
* action_out_proj。

当前canonical LoRA确实就是这38个目标，q层输入/输出为1024→2048，v层为1024→256，action-in为32→1024，action-out为1024→32。

不使用Text Meta-LoRA。

---

## 3. 每帧的静态与policy-native表示

对视频 (k) 的第 (t) 帧 (I_{k,t})，在两个固定、task-independent的antithetic Action probes下运行冻结π0.5：

[
p\in{+,-}.
]

不能像旧effect bank那样在落盘前把probe直接平均；probe轴用于表示不确定性。

### 视觉部分

冻结VLM产生：

[
Z_{\text{patch}}[k,t,p_i]\in\mathbb R^{256\times128}.
]

language-conditioned patch grounding形成task-relevant patch tokens。

### Action Expert紧凑表示

保留当前Stage 0的：

[
Z_{\text{owner}}
\in
\mathbb R^{K\times T_k\times2\times38\times50\times128}.
]

这部分可复用当前全层observer、transition matcher、event-horizon binding和有序segmenter。当前实现已经真实捕获19个层边界、50个action positions，并产生38-owner紧凑lattice。

但必须新增第二类hook，供后面的factor compiler使用：

### 每个LoRA target的原生输入/输出bank

对每个target (j)，捕获其线性层真实输入和真实输出：

[
X_j[k,t,p,h]\in\mathbb R^{d^{in}_j},
]

[
Y_j[k,t,p,h]\in\mathbb R^{d^{out}_j}.
]

具体为：

* q_proj：input `[50,1024]`，output `[50,2048]`；
* v_proj：input `[50,1024]`，output `[50,256]`；
* action_in：input `[50,32]`，output `[50,1024]`；
* action_out：input `[50,1024]`，output `[50,32]`。

同时构造同维度的自然动态bank：

[
Y^{\Delta}_j(t)=Y_j(t)-Y_j(t-1),
]

[
Y^{init}_j(t)=Y_j(t)-Y_j(1),
]

[
Y^{goal}_j(t)=Y_j(T)-Y_j(t).
]

这些不是额外学习出来的低维basis，而是该task video在π0.5自身target坐标中的真实向量。

为避免显存爆炸，不需要一次保存整段视频的全部宽向量：

* Pass A只保留128维紧凑Program；
* Pass B按frame chunk重新运行冻结backbone；
* 使用已经确定的Program query在线累计factor sufficient statistics；
* 不在内存中物化完整 (K\times T\times38\times50\times2048) tensor。

---

## 4. 每条视频如何形成有序Program

当前四类transition candidates可以保留：

1. adjacent transition；
2. short-window transition；
3. initial-to-current；
4. current-to-final。

它们与owner×horizon lattice做双向cross-attention，然后进入有序semi-Markov segmenter。当前代码已经实现了event candidate与owner/horizon的双向绑定，以及只能停留或向前跳转的有序event posterior。

### Event slot数量

固定最大容量：

[
\boxed{E=8}
]

但active event数量动态。

每条视频得到：

[
\alpha_k[e,t]\quad
\text{或更细的 }\alpha_k[e,t,m],
]

满足：

* 沿视频时间单调；
* 可以stay；
* 可以跳过slot；
* 不强制使用全部8个slot；
* 不按手工phase固定边界。

presence：

[
\rho_k[e]\in[0,1]
]

由soft occupancy产生。active/inactive只用于报告；forward始终使用soft presence，不做离散剪枝。

每条视频的process Program：

[
P_{\text{process},k}
\in
\mathbb R^{8\times38\times128}.
]

同时保留：

[
\tau_k\in\mathbb R^{8\times2},
]

表示每个slot的归一化中心与duration，以及：

[
\sigma_k
\in
\mathbb R^{8\times38\times128},
]

表示frame、candidate和probe不确定性。

---

## 5. Scene表示

scene不能继续只是一个全局 first/final/difference 384维向量。

为每个owner分别读取：

* first-frame task-grounded patches；
* final-frame task-grounded patches；
* first/final relation；
* final−initial relation。

得到：

[
\boxed{
P_{\text{scene},k}\in\mathbb R^{38\times128}
}
]

它表示：

* 哪些对象和关系是任务相关的；
* 视频初态；
* 视频目标态；
* 与每个policy target对应的scene evidence。

---

## 6. K条视频如何聚合

每条视频必须先独立完成：

* frame encoding；
* ordered segmentation；
* scene extraction；
* event Program。

不平均frames、raw features或LoRAs。

### Event alignment

每个视频都有最多8个有序slots。使用固定的8个canonical event queries，对每条视频的8个slots做带顺序惩罚的soft monotonic alignment：

[
M_k[e,e'].
]

要求：

* K=1严格identity；
* 只允许保序映射；
* inactive slot可跳过；
* 不能由一个视频完全覆盖其它视频。

### Set aggregation

对每个canonical event和owner：

[
\mu[e,j]
========

\frac{\sum_k w_{k,e,j}M_kP_{k}}
{\sum_kw_{k,e,j}},
]

并计算variance。

允许一个初始化为零、受anchor RMS约束的DeepSets修正，但修正幅度不能超过anchor RMS的25%，防止历史K-set overwrite问题。

最终Program为：

[
\boxed{
P=
\left{
\begin{aligned}
&P_{\text{lang}} &&[38,128]\
&P_{\text{scene}} &&[38,128]\
&P_{\text{process}} &&[8,38,128]\
&\rho &&[8]\
&\tau &&[8,2]\
&\sigma &&[8,38,128]
\end{aligned}
\right.
}
]

这就是唯一Program schema。

其event轴和owner轴有固定含义；128维channel允许内部旋转，不需要人为声称每个channel是某个可解释技能。

---

# 二、是否需要 `q_pi`

## 我的裁决：canonical第一版不需要神经网络 `q_pi`

我建议从active设计中删除：

[
q_\pi(P\mid\text{successful policies})
]

作为必需模块。

原因不是privileged policy evidence无用，而是：

1. 没有真实Program标签；
2. 一个policy encoder与一个video encoder同时映射到同一latent，并不能自动消除latent gauge；
3. 若`q_pi`和realizer联合训练，仍可做任意可逆旋转；
4. 当前最具信息量的policy-effect code都无法经共享realizer稳定恢复held update，再加一个`q_pi`只会把问题上移一层；
5. 现有95 task/118 member数据更适合直接监督最终policy function，而不是再学习一个未验证latent teacher。

当前design仍把distributional `q_pi`和fixed realizer写为未来主链，但仓库本身也承认二者都未实现；在两种shared mapping已经闭环失败后，继续把它们当强制前置没有足够依据。

## 替代：非参数 privileged policy evidence critic

保留：

[
\mathcal E_\pi(\tau)
====================

{
\text{successful members},
\text{verified states},
\text{policy effects},
\text{actions},
\text{reliability}
}.
]

它不是一个网络，也不输出Program。

生成LoRA后，在训练时计算其policy response，并使用global-member set loss：

[
\mathcal L_{\text{equiv}}
=========================

-\eta
\log
\sum_m
w_m
\exp
\left[
-\frac{
D(R_{\Delta},R_m)
}{\eta}
\right].
]

关键约束：

* 对一个logical example，全trajectory只选择一个global member；
* 不允许不同event自由拼接不同member；
* 只有经过continuation验证的member-state pair才是valid target；
* multiple successful policies在loss层形成等价类；
* 不要求generated LoRA接近某个raw A/B。

policy effects仍然不可替代，因为它们解决：

* LoRA gauge非唯一；
* 不同successful adapters参数不相似；
* raw Frobenius受q_proj大能量支配；
* factor loss与closed-loop function不一致。

但它们只作为**训练critic**，不成为部署forward中的“Program→effect→fixed solver”链。

---

# 三、Program如何生成LoRA

## 1. 不再使用固定fit-span realizer

不使用：

* balanced-SVD effect-code decoder；
* centered two-sided fit span；
* global 16D/32D code；
* 21M target/rank query hyperdecoder；
* full-width FactorHeads从128维直接吐2048维向量。

当前shared-realizer family已经被充分关闭。

## 2. Native-factor compiler

对每个target (j) 和最终residual rank slot (r=1,\ldots,4)，由对应owner Program产生query：

[
q_{j,r}
=======

F_q
\left(
P_{\text{lang}}[j],
P_{\text{scene}}[j],
P_{\text{process}}[:,j],
\rho,\tau,\sigma,
e_r
\right).
]

query先给event分配权重：

[
\eta_{j,r,e}
============

\operatorname{softmax}*e
g(q*{j,r},P_{\text{process}}[e,j]).
]

随后在Pass B中，对所有视频、frame、probe、horizon及native feature type做**signed pooling**。

为避免softmax只能正加权，使用两个分支：

[
w_n
===

## \operatorname{softmax}(l_n^+)

\operatorname{softmax}(l_n^-).
]

### 输入侧因子

[
a_{j,r}
=======

\operatorname{Normalize}
\left[
\sum_n
w^A_{j,r,n}
X_{j,n}
\right]
\in\mathbb R^{d^{in}_j}.
]

### 输出侧因子

输出bank包含：

* absolute target output (Y_j)；
* adjacent difference；
* initial difference；
* goal difference。

[
b^{unit}_{j,r}
==============

\operatorname{Normalize}
\left[
\sum_n
w^B_{j,r,n}
Y^{type}_{j,n}
\right]
\in\mathbb R^{d^{out}_j}.
]

Program再产生signed scale：

[
s_{j,r}
=======

s_j^{ref}\tanh(\hat s_{j,r}),
]

其中 (s_j^{ref}) 由fit task expert correction的target-wise median RMS预先确定，之后冻结；不能沿held结果调scale。

[
b_{j,r}=s_{j,r}b^{unit}_{j,r}.
]

最终task residual：

[
\Delta W_j^{task}
=================

\sum_{r=1}^{4}
b_{j,r}a_{j,r}^{T}.
]

通过small-core balanced SVD做确定性canonicalization，保持effective update不变、rank≤4。

这个compiler没有固定跨任务parameter span：

* (a)来自当前task video下的真实target input；
* (b)来自当前task video下的真实target output及其动态变化；
* learned部分主要决定选址、符号和尺度；
* held task的新方向可以来自held video自身的native activation，而不是必须存在于fit90 PCA basis中。

这正面针对two-sided coordinate暴露的“低能量held innovation不在fit span内”问题。

---

# 四、最终rank16还是rank12+rank4

## 我的重新判断：仍选择rank12 carrier + mobile rank4 residual

这不是历史惯性，而是目前最强的闭环证据所支持的选择。

mobile-rank4解析投影已经在held5三个member arms得到：

* 110；
* 120；
* 76；

并且三个arms均5/5 task非零，整体接近或超过matched direct experts。当前失败来自共享映射和solver，而不是rank4 residual容量。

相比之下，直接生成完整rank16意味着：

* task-specific输出自由度扩大约四倍；
* 更容易覆盖或破坏carrier支持；
* 重回历史fresh full-LoRA moving decoder问题；
* 在只有24个source-unseen mappings时显著增加统计难度。

最终参数严格为：

[
A_j^{final}
===========

\begin{bmatrix}
A_{j,c}^{12}\
A_{j,r}^{4}
\end{bmatrix},
\qquad
B_j^{final}
===========

\begin{bmatrix}
B_{j,c}^{12}&B_{j,r}^{4}
\end{bmatrix},
]

所以：

[
B_j^{final}A_j^{final}
======================

B_{j,c}A_{j,c}
+
B_{j,r}A_{j,r}.
]

没有raw-factor交叉项，也没有第二adapter；最终就是一套76 tensor、38-target、rank16 LoRA。

### 什么时候才重开完整rank16

只有新native-factor free-code oracle满足以下组合时，才允许一个full-rank16诊断：

* native factor bank明确能表示task update方向；
* rank4 task-local free-code在优化上已收敛；
* 但解析/response分析证明剩余误差确实来自rank ceiling；
* full-rank16同构compiler的单次oracle显著通过。

当前已有解析rank4正证据，不满足这个重开条件，因此不应现在增加full-rank16 arm。

---

# 五、Action Meta-LoRA

默认关闭。

当前matched Action Meta结果中性，既没有解决probe invariance，也没有闭环正收益。它不进入canonical checkpoint。

仅在完整base Writer已经出现明确闭环增量后，允许一次matched control：

* 只训练共享Action Meta；
* Stage 0和compiler全部冻结；
* 检查probe稳定、same-task video和closed-loop；
* 若有明确净收益且无breadth/retention损害，则永久冻结；
* 否则删除。

不能让Action Meta与Program或LoRA输出坐标共同旋转。

---

# 六、模块冻结与训练关系

## 永久冻结

* source PI0.5 backbone；
* PaliGemma/VLM；
* native Action Expert；
* q/v/action-in/action-out原始权重；
* 最终rank12 carrier；
* Action Meta默认不存在。

## Writer可训练模块

* patch/language local projections；
* 38个owner language queries；
* owner-specific scene reader；
* transition matcher；
* event-horizon binding；
* ordered event segmenter；
* Dynamic-K alignment/aggregation；
* rank4 factor queries；
* native input/output bank key/query projections；
* signed pooling logits；
* per-target scale predictor。

## 分阶段冻结

1. 先训练Program encoder；
2. 冻结Program，训练native-factor compiler；
3. 两者分别通过后再联合解冻；
4. carrier始终冻结；
5. final outer credit先只更新Program与factor attention/scale，不直接扰动A/B tensor。

---

# 七、从当前状态到完整ECP的必要阶段

## 阶段0：自然数据authority与角色冻结

### 数据

使用现成数据：

* 71个审计后LIBERO-90 non-held tasks；
* target train24；
* validation8；
* Test8。

角色必须明确：

* 71 tasks已经被source见过，主要用于：

  * native observer预训练；
  * primitive/scene/语言覆盖；
  * shared carrier；
  * preservation；
* train24是主要source-unseen adaptation mappings；
* validation8只做后期deployment development；
* Test8保持sealed。

开发阶段采用现有五折：

* non-held 71：56 fit / 15 held；
* target train24：19 fit / 5 held。

每个macro对两类任务role-balanced，而不是让71 tasks在数量上压倒19个adaptation tasks：

* 19个target-fit全部访问一次；
* 从56个meta-fit中轮换采样19个；
* 两个role各占50% task weight。

### 允许读取

fit tasks可读取：

* exact language；
* teacher videos；
* demonstrations/actions；
* task experts；
* reward/BDDL progress；
* successful trajectories。

held fold只做预注册mechanism evaluation；validation/test不产生共享梯度。

### 产出

* 固定fold manifests；
* baseline rows；
* carrier/task expert/effect authority；
* 无模型选择。

### 成本

CPU/资产审计为半天级；不需要新大规模GPU。

---

## 阶段1：Natural Stage 0-V Program

这是不可省略阶段。

### 输入

* exact language；
* K∈{1,2,4}，uniform采样；
* action-hidden自然teacher videos；
* 每条视频内部stride5，始终保留first/final；
* video demos与functional/action query demos严格错开。

### 训练数据

使用meta56 + target-fit19。

允许使用existing demonstrations的派生标签：

* 同一episode action chunk；
* gripper/contact变化；
* BDDL predicate satisfaction/progress；
* initial/final object relations；
* demo速度扰动和temporal crop。

这些都来自现成LIBERO任务和demonstrations，不创建新任务、成功语义或人工trajectory。

### 模型

训练：

* owner-specific `P_lang`；
* owner-specific `P_scene`；
* `P_process/rho/tau/sigma`；
* K聚合。

backbone冻结。

### Loss

[
L_{\rm obs}
===========

L_{\rm local-action}
+
L_{\rm progress}
+
L_{\rm event}
+
L_{\rm relation}
+
L_{\rm cross-video}
+
L_{\rm probe}.
]

具体：

1. **local action grounding**
   从frame/event owner tokens预测当前frame后10个phase的7D action summary。
2. **natural progress/event**
   预测BDDL predicate progress、predicate rising、gripper/contact变化。
3. **scene/goal relation**
   从`P_scene`预测现成任务的initial/final relations。
4. **cross-video consistency**
   同task不同episode按soft event alignment后一致，但不要求逐帧一致。
5. **speed/crop robustness**
   同一video不同采样速度保持event顺序和task representation。
6. **probe stability**
   两个固定probe的Program均值接近，同时保留variance，不提前平均。

不使用shuffled/reversed作为训练negative。

### Stage 0资格门

这是representation qualification，不是方法成功。

在meta-held15 + target-held5上要求：

* 至少90%的same-task pairs在owner/event-aware距离上近于nearest cross-task；
* 至少75%的rows中，替代probe带来的变化小于nearest cross-task margin的一半；
* active events不坍缩：单event占比不超过25%，median active slots在2–6；
* full video相对first+final，在action/progress held loss上至少有10%的相对改善；
* K=1 identity、K permutation invariance通过。

失败说明：

* 当前video Program仍主要是task/endpoint code；
* 不能进入LoRA训练。

允许一次架构修正仅限：

* target-native q/v capture；
* event grounding位置；
* owner-specific language/scene。

不允许再改slot数、width、seed做版本链。

### 成本

按现有Stage 0 v3吞吐，单次formal为小时级；连同held机制审计，正常一个完整迭代约1–2个工作日。

---

## 阶段2：Native-factor task-local capacity oracle

这是第一道不可替代的closed-loop门，也是现在唯一最合理的下一执行步骤。

### 目的

先不训练共享video→LoRA映射，只回答：

> 在自然视频产生的target-native input/output banks中，是否存在一组小型task-local选址权重，可以生成强闭环rank4 residual？

### 输入

* 冻结Stage 0；
* fold0 held5的自然teacher videos；
* known successful task experts/mobile projections；
* existing carrier43；
* 不读取validation/test。

### 可训练参数

每个held task单独优化：

* 4个rank queries；
* event权重；
* signed pooling权重；
* per-target scales。

不更新：

* Stage 0；
* compiler共享参数；
* source；
* carrier；
* task expert。

这是free-code upper bound，不是部署方法。

### 目标

* global-member effect loss；
* sensitivity-normalized effective-update loss；
* independent action-query functional loss；
* carrier support preservation。

### 闭环比较

同一strict250：

* carrier；
* direct latest；
* known mobile-rank4 projection；
* native-factor free-code candidate。

### 通过条件

同时满足：

1. relative oracle recovery：

[
\frac{
S_{\rm free}-S_{\rm carrier}
}{
S_{\rm mobile}-S_{\rm carrier}
}
\ge 0.70;
]

以当前43与110参考，即总分至少约90；
2. breadth 5/5；
3. Goal与Long均非零；
4. 至少4/5 tasks严格高于carrier；
5. carrier success保留至少33/43；
6. single rank16、严格paired、无second adapter。

### 失败含义

若解析mobile rank4仍强，而native factor free-code失败，则问题在：

* target-native token bank不够；
* signed pooling形式不够；
* q/v output capture不正确。

允许一次只读span分析和一次实现修正。修正后仍失败，停止Native-Factor架构，不训练q_V/compiler，也不把失败归因于数据量。

### 成本

5个task-local小优化 + strict250，一次完整周期约1个工作日。

---

## 阶段3：冻结Program的shared compiler预训练

只有阶段2通过才进行。

### 数据

* meta56 + target-fit19；
* 每task K=1/2/4自然videos；
* 95-task/118-member现有expert evidence可复用；
* video demos与action/query episodes错开；
* task role 50/50平衡。

### 训练模块

冻结Stage 0，只训练：

* Program→rank query；
* native-factor signed pooling；
* per-target scales；
* bounded K correction。

### Loss

#### 1. 全trajectory single-member effect等价损失

不能event-wise拼接不同members。

#### 2. Target-family-balanced functional loss

q、v、action-in、action-out四个family总权重相等，再在family内均分，防止q_proj 91%以上能量掩盖低能量关键方向。

#### 3. Cross-episode action flow loss

generated LoRA在独立action episodes上降低PI0.5 flow loss。

#### 4. Effective-update辅助

对known mobile projections做有效更新距离，但按policy sensitivity归一，不用raw factor MSE或aggregate cosine。

#### 5. Carrier preservation

在carrier/source已成功或member disagreement高的states上限制伤害。

#### 6. Same-task video functional consistency

同task不同video生成的LoRA不要求参数相同，但在固定query states上的policy response应接近。

### 第一轮held5闭环门

比较：

* carrier；
* learned language-only baseline；
* full video；
* first+final；
* same-task-other。

继续条件：

* full至少60/250；
* breadth至少4/5；
* carrier保留至少33/43；
* Goal或Long至少一个非零；
* full相对language-only及first+final各净增至少5；
* same-task retention至少80%。

这不是最终门，只判断shared mapping是否有基本信号。

若free-code很强而shared compiler仍低于carrier或breadth≤2，说明现有24个source-unseen mappings不足以学习该映射，不能用联合训练掩盖。

### 成本

一次完整训练和held5评测约2–3个工作日。最多一个完整结构版本。

---

## 阶段4：联合解冻Writer

这是最终方法必需阶段，但只有阶段3有真实信号才进行。

### 解冻

解冻所有Writer参数：

* semantic projections；
* scene reader；
* transition matcher；
* event binding；
* segmenter；
* K aggregator；
* factor compiler。

继续冻结：

* π0.5 backbone；
* carrier；
* task experts。

### 训练顺序

#### 4A. Functional joint warmup

继续使用阶段3全部loss，Program checkpoint作为anchor，compiler较小学习率。

#### 4B. Natural-task verified on-policy support

在fit tasks上用generated LoRA rollout。

从student真实visited states分支查询多个task experts，但不能直接把off-policy expert输出当oracle。一个member-state pair只有在固定短continuation中满足以下之一才可作为label：

* 最终成功；
* BDDL progress严格提高且不撤销已完成predicate；
* 明显优于carrier/source continuation。

有效member才进入set loss；无有效member的state只使用reward/progress，不生成伪动作label。

这直接吸收人工process分支暴露的“task expert在改变后的occupancy上未必可靠”教训，同时完全使用现成自然LIBERO tasks、experts和reward。

### held门

至少在两个train24 folds上分别要求：

* oracle-normalized recovery至少0.40；
* breadth 5/5；
* Goal/Long均非零；
* carrier retention≥75%；
* same-task video retention≥85%；
* 两个相邻checkpoint分数差≤10且success-set Jaccard≥0.75；
* full相对first+final和language-only均有正增量。

若阶段3有信号、联合训练却崩落，只允许一次：

* 冻结compiler；
* Program proximal/EMA；
* 更低Program学习率。

不能再改架构、slot、rank或seed。

### 成本

一次joint round约2–4个工作日；最多两轮，包括on-policy数据采集和closed-loop。

---

## 阶段5：structured outer credit

这是条件阶段，不是前置。

只有阶段4已经证明：

* full video高于carrier；
* full高于language/endpoints；
* breadth基本成立；

才开始。

### 更新对象

只扰动和更新：

* event posterior；
* Program；
* rank attention；
* residual scales。

不直接在百万A/B tensor上做无结构ES。

### 数据

仅fit/meta自然LIBERO tasks，使用task-equal CRN rollouts：

* success；
* BDDL progress；
* efficiency；
* carrier/full paired advantage；
* retained-success barrier。

### Gate

两个预注册outer节点中的最终节点必须相对阶段4：

* 净增至少10个held successes；
* breadth不下降；
* Goal/Long不下降；
* same-task retention不下降；
* churn不以压低gains为代价。

两节点均无净改善则停止该estimator，不做seed、sigma或step sweep。

### 成本

约2–3个工作日；只允许一个estimator family。

---

# 八、最终ECP如何构建

## 1. 进入最终模型的checkpoint与schema

进入final run的只有：

1. 固定Program schema；
2. Stage 0初始化权重；
3. Native-factor compiler结构；
4. rank12 carrier训练recipe；
5. 经过两个train folds验证的joint training recipe；
6. outer credit仅在其通过时进入。

不进入最终模型：

* 人工process data；
* recovery/composite experts；
* GOMQ checkpoint；
  -历史q_pi；
* PECS；
* fixed-effect realizer；
* two-sided coordinate；
* Action Meta，除非后续matched通过。

## 2. Final fresh训练数据

使用：

* 全部71个审计后non-held LIBERO-90任务；
* 全部target train24；
* 现成demonstrations/videos/actions；
* task experts及成功轨迹；
  -自然task reward/progress。

任务权重仍role-balanced：

* train24承担50% adaptation weight；
* 71 meta tasks承担50% observer/prior/preservation weight。

不得让71个source-seen tasks因数量优势淹没24个source-unseen mappings。

## 3. Final训练顺序

1. fresh训练rank12 shared carrier；
2. fresh训练Stage 0 Program；
3. 冻结Stage 0训练compiler；
4. 联合解冻全部Writer；
5. 运行至由train24 cross-validation预先确定的训练horizon；
6. 只有此前outer credit通过，才接固定数量outer updates。

π0.5 backbone全程冻结。

---

# 九、最终部署forward

对一个新task：

1. 输入exact language；
2. 输入K条same-task、action-hidden、内部有序视频；
3. Pass A：

   * frozen PaliGemma读取language和frames；
   * frozen Action Expert在两个fixed probes下运行；
   * q_V生成`P_lang/P_scene/P_process/rho/tau/sigma`；
   * K视频做event-aligned aggregation；
4. Pass B：

   * 使用Program query重新读取每个frame中38个LoRA target的native input/output banks；
   * signed pooling生成每target 4个outer-product factors；
   * balanced SVD canonicalize为rank4 residual；
5. 与frozen rank12 carrier严格拼接；
6. 输出唯一一套38-target rank16 LoRA；
7. rollout前安装；
8. 闭环中不再观看teacher video。

---

# 十、validation8选择与Test8

## 1. 不应再用validation8做开放式选模

validation8已经经历大量历史评测。架构、训练horizon、loss和checkpoint schedule应先由train24多fold固定。

Final run只保留三个预注册相邻checkpoint，例如：

[
H-1,\ H,\ H+1
]

或三个固定macro，而不是训练后挑winner。

## 2. Validation8有限选择规则

对三个checkpoint先只运行：

* correct full；
* same-task-other；
* learned language-only；
* first+final。

一个checkpoint先满足资格：

* correct至少135，最终目标仍为>145；
* breadth@1=8/8；
* breadth@5至少6/8；
* 四个suite均非零；
* same-task success retention≥90%；
* 相邻checkpoint均不低于它10分以上；
* 相邻success-set Jaccard≥0.80；
* top3 task share≤70%。

若多个通过，选择correct最高者；若并列，选择更早checkpoint。若没有通过，不进入完整controls，也不打开Test。

## 3. 冻结后完整视频因果面板

selected checkpoint冻结后才跑：

* correct；
* same-task-other；
* cross-suite wrong；
* video-only；
* learned language-only；
* no-video/carrier；
* static-first-repeated；
* first-only；
* final-only；
* first+final。

随后才跑：

* shuffled；
* reversed。

shuffled/reversed绝不进入训练、loss或选模。

## 4. 最终方法资格

必须同时满足：

### Absolute

[
\boxed{\text{correct}>145/400}
]

### 稳定性

* 两个相邻checkpoint均在selected的10分内；
* Jaccard≥0.80；
* breadth@5不下降超过1个task；
* 没有suite突然归零。

### 跨视频

* same-task-other总分在correct ±10内；
* correct成功rows保留≥90%。

### 视频必要增量

correct相对以下每个arm：

* language-only；
* no-video；
* static；
* first+final；
* wrong；

至少满足：

* paired净增≥10；
* exact McNemar (p<0.05)；
* 至少3/4 suites不为负；
* 新增成功不能几乎全部来自单一task。

### 时序特异性

correct相对shuffled和reversed分别：

* paired净增≥15；
* exact McNemar (p<0.05)；
* Goal/Long不得系统性反向；
* correct绝对分数必须保持，而不是仅把controls推坏。

全部通过后，方法、checkpoint、K和controls永久冻结，才打开Test8一次。Test结果不再反哺设计。

---

# 十一、哪些历史步骤可以删除

## 从active路线删除

* 所有custom process tasks、wrappers和teacher acquisition；
* GOMQ训练或压缩救援；
* deterministic/mean q_pi；
* Program-to-A/B full hyperdecoder；
* 20×16 rank grid与M2P；
* PECS selected-frame solver；
* fixed-A solver；
* raw mobile-rank4 solver；
* matrix-free solver；
* balanced-SVD shared realizer；
* centered two-sided fit-span realizer；
* Action Meta默认路径；
* open-loop geometry gate。

## 继续复用

* native Stage 0 v3初始化；
* full-layer/horizon capture；
* transition matcher；
* event binding与segmenter；
* strict evaluator与controls；
* task expert bank；
* successful policy members；
* effect path calibration；
* probe-particle captures；
* stable carrier和mobile-rank4解析容量证据；
* natural occupancy、action和reward基础设施。

---

# 十二、什么结果才足以判定ECP在现有数据下存在根本问题

必须先完成以下所有条件：

1. 自然Stage 0 Program通过资格门；
2. native-factor free-code oracle通过；
3. frozen Program→LoRA shared compiler在至少一个fold产生基本闭环信号；
4. joint Writer在至少两个fold复现；
5. 使用了verified on-policy natural task evidence，而不是盲信off-policy expert；
6. 尝试过一次结构化outer credit；
7. final validation使用全部自然授权数据fresh训练；
8. full video与language/endpoints/wrong controls完整比较。

若在此之后仍出现以下组合：

* validation correct低于130–135；
* breadth@5≤4/8；
* Goal或Long为0；
* full不显著优于language-only或first+final；
* same-task retention低于80%；
  -相邻checkpoint持续大幅换手；

则应判断：

> 在现有LIBERO任务数、自然demonstrations和zero-interaction static-LoRA合同下，action-hidden video不足以支持稳定的跨任务amortized policy compilation，或现有任务没有提供足够条件增量让模型必须使用视频过程。

此时应停止当前ECP主线，而不是再修改slot、rank、width或decoder。

---

# 当前唯一最合理的下一执行步骤

> **实现target-native q/v/action-in/action-out输入输出hook，以及Program-conditioned signed outer-product rank4 compiler；先做fold0 held5的task-local free-code closed-loop capacity oracle。**

现在不应：

* 训练q_pi；
* 重建shared realizer；
* 训练fresh Stage 0；
* 启动joint Writer；
* 开outer credit；
* 运行新的人工数据或GOMQ实验。

这一步必须先回答最基础的问题：

> **自然视频在π0.5各目标模块中产生的native input/output向量，本身是否构成足以表达强task LoRA的条件化参数基底？**

通过条件固定为：

* 相对carrier→known-mobile gap恢复至少70%；
* 约90/250或以上；
* breadth 5/5；
* Goal/Long均非零；
* 4/5 tasks高于carrier；
* carrier保留至少33/43；
* single complete rank16；
* strict pairing有效。

这一步通过，才值得训练Program和共享compiler；这一步失败，则新的核心架构本身不可达，应立即停止，而不是进入下一轮版本化。

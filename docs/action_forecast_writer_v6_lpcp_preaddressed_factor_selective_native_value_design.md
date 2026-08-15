# V6-LPCP Pre-Addressed Factor-Selective Native Value

状态：2026-08-15 mechanism implementation ready，formal仍由第8节机制门阻断。简称`PAFS-NV`。本设计建立在NPVC cycle1完整终局之后，只替换
`native probe Value -> factor hidden direction`的选择接口；LPCP143 carrier、K4 ordered set、38-target rank16
public LoRA、paired selected-success reward、frozen source policy、split与评测合同全部保持不变。

## 1. Decision

NPVC同时给出一个正结论和一个否决结论：

- 正结论：validation8的不同correct K4能够形成cosine/energy=`.40870/.54227`的共同native-scale写入，reverse
  使probe/BA relative-L2=`1.84084/1.60518`；CCT的held compiler消失已经解决；
- 否决结论：strict只有`136/400`，相对LPCP143为`120 retained / 16 gained / 23 lost`、churn39；gained/lost
  BA改写不可分，retained-failure最大；full24后train task4共同方向从`.5929/.6792`漂到`.0569/.2951`。

因此当前不能再修改已通过的video carrier、放大LoRA或只追求coherence。最早缺口是：同一个zero-init
semantic query先把所有task和八个factor families压入四个共同route，再由一次full24 mean决定组件和符号。
本轮把任务分流前移到第一次reward update之前，并让q/v/action的Value选择彼此拥有。

```text
exact language + K4 ordered action-hidden videos
  -> frozen LPCP Core / causal Procedure / Procedure-set
  -> same ordered native Action-probe Value M[320,256]
  -> frozen language pre-address a[320,4]
  -> zero-init factor-owned selectors over M x language
  -> two coefficients per factor family and policy slot
  -> frozen V6-W1 policy axes + frozen W2
  -> one complete 38-target rank16 LoRA
```

## 2. The single causal variable

NPVC对所有八个factor families共享：

```text
q = Wq L
p = softmax(q K^T)
c0,c1 = fixed dot products between M and L / signed-L
R_f = (p0-p1)c0 A_f,0(L) + (p2-p3)c1 A_f,1(L)
```

`Wq=0`使所有task在第一次update前都是相同uniform route；同一个route同时决定q-A/q-B/v-A/v-B/action-in/out
A/B。PAFS-NV替换为：

```text
Lhat_s       = RMSNorm_no_affine(L_s)
a_s,b        = softmax(Lhat_s dot RMSNorm_no_affine(K_fixed,b) / sqrt(256))
x_s          = M_s elementwise_mul Lhat_s
z_f,s,b,o    = W_f,b,o dot x_s                       # o in {direct,signed}
c_f,s,o      = sum_b a_s,b z_f,s,b,o
A_f,0(L_s)   = GELU(W1_f Lhat_s)
A_f,1(L_s)   = GELU(W1_f (sign elementwise_mul Lhat_s))
R_f,s        = c_f,s,0 A_f,0(L_s) + c_f,s,1 A_f,1(L_s)
```

`K_fixed`沿用NPVC确定性四basis seed，但改为persistent buffer，不训练、不随checkpoint漂移。每个factor family
拥有`W_f in R^(4 x 2 x 256)`，全部exact-zero初始化；总trainable=`8*4*2*256=16,384`。四basis数沿用
NPVC，不做capacity sweep。所有输出仍经对应冻结V6 FactorHead的真实W2写回同一rank16 topology。

这一个接口同时完成两件不可拆的语义：固定pre-address在第一次shared update前区分task/slot，factor-owned
selector再让q/v/action从同一有向video evidence中选择不同组件与符号。它不是给每个task建expert，也不是增加
LoRA容量；部署时held task仍只靠exact language和video计算同一个连续共享函数。

## 3. Exact invariants

- `W_f=0`时所有`R_f=0`，step0 public LoRA逐tensor exact LPCP；不靠正负大数抵消；
- `M=0`时任何language/address都不能写新增LoRA，不存在language-only bypass；
- same-task不同videos共享完全相同的language pre-address，但`M`仍来自各自有序过程；
- shuffled/reversed必须重排真实frames并完整重算LPCP、set与`M`；
- correct K-set仍置换不变，K1--K4走同一LPCP图；
- fixed address只决定共享selector的混合权重，不携带task ID、filename、expert LoRA或deployment bank；
- source policy、LPCP query delta、Procedure-set、compiler、W1/W2和全部非selector参数冻结。

## 4. Why this is not a repeated historical arm

- **不是CCT**：CCT用tiny LPCP-AS139 Procedure差分作为Value，held BA缩小约250倍；本轮保留NPVC已证明能在
  held穿过compiler的native probe Value。
- **不是SFMC/GOSC**：它们让所有families共享一个learned semantic route，且route从uniform开始；本轮address
  在reward update前已经由language确定，实际Value组件由factor-owned selectors学习。
- **不是Policy-Lane/Atom/target-owned heads**：不生成新policy axes、不增加public rank或target heads；固定
  V6-W1/W2 geometry仍来自LPCP143，只学习16,384个条件化系数权重。
- **不是task expert bank**：四basis是共享连续address，没有task ID、nearest route、held dictionary或第二LoRA。
- **不是support guard/PCGrad**：不投影或屏蔽task gradient；先检验当前representation能否让raw full24 mean
  自然形成共同descent。OSG-PC/SKNC/RLS已说明只保护旧support不能产生正确acquisition direction。
- **不是Dynamic-K Backbone-Memory rank8重跑**：不删除V6 Semantic Core、不换rank8/fixed-A mapper、不重新
  训练视频前端；历史100/101/102只说明那套fresh memory Program丢失absolute与task geometry。
- **不是memory token否决**：literal memory仍开放；只有PAFS-NV证明现有native Value缺少可学习内容时，才有
  证据把carrier替换成真实context memory。

## 5. High-level video knowledge and order

PAFS-NV不重新声称selector理解视频。高层对象/目标/阶段知识继续来自历史最强LPCP链：exact language形成
task-grounded Core与query，stride5真实frames形成有向Procedure，K-set在每video内部完成因果编码后做置换不变
聚合。native Action probes旁读真实图文与Action Expert的逐层交互，已经由constant/reverse/held证据证明有内容。

selector只回答更晚的问题：对于同一个task-language和有向video Value，哪些通道应驱动q、v、action-in/out的
A/B factors，以及符号是什么。wrong/shuffled/reversed/no-video是否沿有用方向变差仍只能由final paired
closed-loop controls证明；训练不加入negative-video margin或人为破坏negative LoRA。

## 6. Multi-task coexistence hypothesis

NPVC task4 local update强、full24后坍塌，说明“先uniform共享、再靠一个Wq分流”太晚。PAFS-NV的地址在任何
reward gradient前固定存在：不同language/slot产生不同四basis mixtures；同一task跨video地址完全相同。每个
active task的reward gradient因此先落到自己的address-weighted selector组合，q/v/action又不再争用同一route。

这只是可证伪假设，不保证无冲突。formal cycle必须保留每个active task的16,384维mean-four-view gradient，
在一次global mean应用前报告：pairwise cosine、每task与global mean的dot/cosine、正descent覆盖率和每family
gradient energy。该矩阵每cycle至多约`24 x 16,384` FP32，远小于LoRA/backbone，不增加forward、逐tensor扫描或
防御性校验。它只解释task4是否仍被其它task覆盖，不能替代strict结果，也不能据此做gradient surgery。

## 7. Implementation ownership and fresh boundary

1. `src/ember/writer/factor_commitment.py`原位替换NPVC commitment，不新增parallel Writer；
2. `model.py`仍传入同一个`shared_probe_value_slots`与`language_slots`，decode和public topology不变；
3. reward trainer只训练16,384 selectors；LPCP与fixed address无gradient；
4. reward cycle在已有per-task four-view mean处暂存小gradient row，global mean后只汇总必要scalar/matrix evidence；
5. 新config/checkpoint/evaluator schema fresh-incompatible；NPVC checkpoint即使shape局部相似也不得加载；
6. evaluator和LoRA cache仍使用同一个canonical deployment path，不新增baseline forward或第二套adapter。

截至`36e30fd`之后的实现基线，NPVC executable schema/config已原位退役，PAFS-NV fresh-incompatible路径已接通；
selector与fixed address合同、reward checkpoint/eval schema、per-task gradient coexistence evidence均已实现。
canonical LIBERO assets环境下完整CPU为`399 passed`，架构门无block；这些只证明实现合同，不是机制或性能结果。

## 8. Fast mechanism and efficiency falsifiers

CPU/synthetic必须证明：

1. step0 exact LPCP、constant/no-video zero、K-set permutation与natural/reverse material；
2. fixed address同task跨video exact、不同task/slot非退化，train24 flattened address有效rank至少4且mean
   off-diagonal cosine低于`.98`；
3. selector公式逐family/slot与实际输出一致，只有16,384 trainable parameters；
4. 第一次synthetic与真实selected-success update中八family selector均finite/nonzero，fixed keys与所有base参数0
   gradient；
5. post-update q/v/action effective BA与fixed-action response均nonzero；
6. task4四个disjoint correct K4 pure-PAFS correction cosine至少`.15`、energy至少`.40`；
7. validation8只读视频至少6 tasks过`.10/.35`，aggregate至少`.15/.40`，held/train BA L2至少`.10`；
8. reverse material、constant/natural BA L2不超过`.005`；
9. 无第二backbone forward、禁读/OOM/nonfinite，task4 wall不超过NPVC的`1.10x`。

任一结构/held门失败即终局，不启动full24，不扫basis数、selector scale、rank、LR、temperature或seed。

## 9. Formal training and stability adjudication

机制门通过后，从sealed LPCP macro25 fresh做与NPVC相同的full24 cycle1：24 tasks、两组paired states、
AS139/LPCP两臂，唯一成功trajectory在四个disjoint correct K4 conditions上CFM；active task等权，AdamW/LR/
Nmc4/B8与rollout seeds不变。只有factor commitment形式变化。

cycle1后立即strict paired correct400。只有同时满足才exact-resume cycle2：

- correct至少`142/400`、breadth至少7；
- 相对LPCP143 lost不超过15，且没有suite清空；
- post-train held geometry仍过门；
- active-task shared-mean descent覆盖至少75%，task4方向不再次从preformal共同方向坍到近`0/.25`。

owner认可的稳定约145资格：

- cycle1与cycle2都至少142、两点均值至少145、breadth都至少7；
- 相邻checkpoint churn不超过20、success-set Jaccard至少`.85`；
- final相对LPCP lost不超过10且gained不少于lost，增益覆盖多个tasks/suites；
- checkpoint按预注册末端选择，不用union、per-task winner、LoRA平均或checkpoint融合。

稳定资格通过后，对同一final checkpoint做strict paired correct/same-task-other/wrong/shuffled/reversed/no-video。
same相对correct下降不超过8；correct相对每个negative/no-video至少净高10，并在至少3 suites中paired gained>
lost。单点145或151不跳过稳定性与视频因果资格。

## 10. Interpretation boundary

- selector写出仍sub-ULP：当前zero-init selector parameterization不足，只否定该写出形式；
- q/v/action均material但held geometry失败：factor-owned selection破坏了NPVC的跨video共同Value；
- held geometry健康、task gradient明显冲突且strict高lost：pre-address不足以实现task共存，下一接口才是显式
  shared-update coordination；
- task gradients共存但strict仍低：selected-success trajectory本身不是held occupancy有效target，下一接口是
  reward credit而非架构容量；
- cycle1好、cycle2高churn：当前固定address没有形成稳定累积；
- 稳定约145且six-arm健康：即使未到151，也达到owner认可的有价值结果。

负结果只淘汰“LPCP native probe Value + fixed four-way language pre-address + factor-owned diagonal selectors +
matched selected-success reward”组合；不否定memory token、dynamic K、rank8、reward学习或生成LoRA。

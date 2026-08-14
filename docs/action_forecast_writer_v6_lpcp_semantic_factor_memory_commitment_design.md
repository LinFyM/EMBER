# V6-LPCP Semantic Factor-Memory Commitment

状态：2026-08-15 active design authority；canonical实现、CPU门与task4 GPU smoke已完成并seal，尚未full24训练。本文在CV-CSD cycle1完整终局之后建立，
只改变跨视频成功信用最终写入policy topology的位置与参数化。V6/LPCP视频载体、AS139强底座、K4部署、
selected-success replay、rank16 public LoRA和frozen source policy保持不变。

## 1. Decision

下一轮不增加credit view、不替换已经通过的Action-probe carrier，也不恢复历史91--102分的literal-memory
Writer。它保留LPCP macro25的完整143分候选，在其上增加一个zero-init的**Semantic Factor-Memory
Commitment**（SFMC）：

```text
exact task language + K ordered action-hidden videos
  -> frozen LPCP evidence / Core / Procedure / K-set
  -> exact K-set LPCP innovation memory
  -> language-only semantic address
  -> layer/rank/factor-family-owned hidden residual
  -> frozen trained V6 factor output bases
  -> one complete 38-target rank16 LoRA
```

唯一主要变量是：

```text
CV-CSD:
  trainable shared query_delta
  -> video-specific Procedure-attention Jacobian
  -> frozen set/fusion/factor heads

SFMC:
  frozen LPCP query_delta and Procedure path
  -> K-set innovation memory
  -> trainable semantic factor commitment
  -> frozen factor output bases
```

SFMC中的memory是同一次真实language+video forward产生的layer/rank-aligned动态状态，不是追加到backbone的
literal memory token。当前证据已通过carrier门，memory在本轮只服务最早失败的commitment接口；“使用memory”
本身不是目标。

## 2. Evidence fixing this interface

LPCP macro25 K4 strict=`143/400`、breadth7，相对AS139为`120 retained / 23 gained / 19 lost`。它证明：

- 同一次真实image/language/50 Action-probe forward的18层旁读可以高效提供有向视频证据；
- reverse使query-delta/Program relative-L2=`2.0572/.40414`，constant query-delta近零；
- same-task LPCP correction coherence mean/median=`.61786/.56804`，carrier本身并未再次退化为随机正交。

但LPCP相对AS139的全400 effective-BA只改`.002653`。PCSD把真实唯一成功trajectory回传到query map后降到
`135`；CV-CSD进一步让同一成功trajectory在四个互不重叠的correct K4 conditions下完整反传，36/36 view
gradients非零且wall只增加`1.0307x`，strict仍为`134`。CV-CSD相对LPCP的四view deployed correction
pairwise cosine=`.000205`、mean/sample energy=`.250155`，几乎正好是四个正交方向平均后的四分之一。

因此下一未检验接口不是“视频有没有被读到”或“credit是否覆盖多个视频”，而是query参数变化必须先穿过每个
视频不同的Procedure-attention Jacobian，随后又经过冻结fusion/compiler。共享参数梯度相同，不代表部署到四个
视频条件后的policy方向相同。

## 3. Retained input and strong base

本轮完整冻结并保留：

- exact task language与`K=4` same-task ordered action-hidden teacher videos；Writer仍支持已有K1--K4输入图，
  但本轮canonical reward和validation先固定K4；
- frame stride5、每video内部真实顺序与video间置换不变集合边界；
- V6 Semantic Core、task-grounded transition、causal Procedure、trained common-Value K-set；
- LPCP的18-layer×16-rank Action-probe reader、video内causal conditioner和macro25 query_delta；
- AS139 fusion/post-fusion、八个factor families、38 targets和完整rank16 public topology；
- source policy、normalization、24/8/8 split、official LIBERO rollout与全部信息墙。

初始化读取clean sealed LPCP macro25 Writer weights。所有上述参数都frozen；step0 candidate逐tensor等于LPCP143，
reference arm仍是同一cached condition关闭query_delta后的exact AS139。

## 4. Exact K-set innovation memory

对一个condition，令`R`为320个policy layer/rank routing slots，`C`为共享Core slots，`P`为每条video的ordered
Procedure memory，`U_k`为已训练LPCP layer/rank conditioner。使用同一冻结Procedure reader分别计算：

```text
P0_k = ProcedureRead(R, C, P_k, query_condition = 0)
P1_k = ProcedureRead(R, C, P_k, query_condition = query_delta(U_k))
```

再用同一个冻结、已训练、置换不变K-set聚合器：

```text
S0 = FrozenSet({P0_k}_{k=1..K})
S1 = FrozenSet({P1_k}_{k=1..K})
M  = S1 - S0                    # [condition, 320, 256]
```

`M`是本轮唯一新增Value：它精确表示LPCP分层有序视频证据在K-set之后带来的Procedure innovation。

- 它不是raw frames/features或多个LoRA的平均；每个video先独立经过真实顺序编码与Procedure读取，再由已有
  learned K-set共同处理；
- K轴没有shot-position embedding、hard top-k或best-video selector；
- query_delta关闭、constant layerwise innovation或K-set两臂相同时，`M=0`；
- language、routing和learned basis keys都不能单独提供新增Value；
- 计算`S0/S1`复用同一次backbone和缓存，只多做小型Procedure reader与set forward，不重复图像/语言/
  Action Expert backbone。

candidate base Program仍按原LPCP精确计算：

```text
Z1 = FrozenFusion(C, S1)
```

reference Program为`Z0 = FrozenFusion(C, S0)`。SFMC不是以`Z0`替代`Z1`，也不把reference带到部署；`S0`只用于
从同一cached condition分离视频innovation memory。

## 5. Stable semantic address

同一task不同correct videos应共享任务语义地址，而不能让video nuisance决定参数owner。复用同一次text-only
task-token states，经冻结native Core slot reader得到：

```text
L = NormalizeCoreSlots(CoreRead(R, exact-language task tokens))  # [B,320,256]
```

`L`与frames、video ordinal和K-set无关；exact language相同则地址相同。它只选择四个shared semantic factor
bases，不进入factor Value：

```text
q_s       = W_q RMSNorm(L_s)
alpha_s,b = softmax(q_s dot RMSNorm(K_b) / sqrt(256)), b=1..4
```

`K_b`是Q/K-only learned keys。`alpha`和为1，不附加global scale，不按task ID、suite、filename或held outcome
路由。相关tasks可共享basis mixture，未见task可连续组合，而同task不同videos使用相同semantic address。

四个bases不是新的sweep值：历史Semantic Factor-Basis已经证明4-way soft semantic routing能形成task-conditioned
分工并把八checkpoint union提高到193，但其fresh blind-B20 factor generator只得single127且持续换手。本轮只
继承该机制中没有被否定的“连续语义子空间共享”，用cross-video on-policy success credit与143强底座正面处理
当时未解决的方向信用和retention。

## 6. Factor-family hidden commitment

V6每个factor family的冻结head为：

```text
h_f = GELU(W1_f Z1_s)       # hidden width 256
row_f = W2_f h_f
```

SFMC为八个真实factor families各提供四个bias-free `256x256` memory maps，跨18 layers与16 rank coordinates
共享：

```text
Delta-h_f,s = sum_b alpha_s,b * V_f,b M_s
h'_f,s      = h_f,s + Delta-h_f,s
row'_f,s    = W2_f h'_f,s
```

全部`V_f,b` exact-zero初始化，所以step0与LPCP逐tensor相同。最终`W2_f`继续使用已经训练到143区域的policy
output basis；不直接生成raw A/B residual，不引入第二套LoRA，也不要求高rank、正交或均匀能量。离线审计显示
LPCP的q/v、action-in-B与action-out-A `W2`为256维full span，action-in-A为真实32维full span；低能量的
action-out-B当前只有约7个有效方向，必须如实保留而不能宣称全头满秩。总体上这些是已经共同产生143分的
policy-effective output bases，比重新训练一个未经验证的wide decoder更可信。

八个family分别是q-A/q-B/v-A/v-B/action-in-A/action-in-B/action-out-A/action-out-B。一个family map在对应
layers/ranks间共享；不是38或76个target-specific heads。layer、endpoint与rank ownership来自320 slots，factor
ownership来自真实family，language route在A/B与各family间共享，因此既有policy对应，又不退化成task hard expert。

本模块的trainable参数预计为：

```text
8 families * 4 bases * 256 * 256 = 2,097,152
semantic query                         65,536
four basis keys                         1,024
two RMSNorms                              512
total                                2,164,224
```

实现后由真实enumeration裁决。query_delta、reader、conditioner、set、fusion和原factor heads全部frozen。

## 7. Why this is not a repeated failed architecture

- **不是literal-memory rank8旧路线**：不替换V6 Core/Procedure/decoder，不增加backbone token；旧路线低absolute的
  前端与fresh mapper问题不会混入本轮。
- **不是generic Program residual**：新增Value是`Set(P1)-Set(P0)`的layer/rank innovation，并在factor-family
  hidden owner内提交；不是condition级320-slot自由向量blind-add。
- **不是Target-Owned/Policy-Lane**：不拆成38/76 heads，不重新学习完整wide factor输出；冻结强V6 `W2`。
- **不是SFB原样恢复**：保留其semantic routing原则，但不fresh替换完整factor generator；本轮只生成zero-init
  dynamic residual，并使用cross-video selected-success而非blind B20。
- **不是raw A/B tangent或rank compression**：不在gauge-dependent factors上直接加bank/LoRA，不改变public rank。
- **不是language-only shortcut**：language只有Q/K address；`M=0`时所有新增factor residual严格为0。

## 8. Rank decision

本轮保留rank16不是把它写成长期目标，而是为了让commitment成为唯一主要变量：LPCP143、AS139 tail、320 slots、
factor output bases和paired baseline都属于rank16。此时同时降rank会混入compression/regeneration与fresh decoder
影响；历史uniform rank14已独立lost15，fresh rank8完整替换路线只有91--102。

这不否定最终rank8。若SFMC在rank16上形成稳定absolute与视频因果性，后续可写独立authority，在相同semantic
factor-memory接口下fresh训练rank8，检验更小payload能否保留性能；不能用本轮的受控选择宣称rank16永远最优。

## 9. Training credit and optimizer

训练保留CV-CSD的完整科学样本：train24每task两组严格配对states，AS139 reference与当前candidate共96 rollouts；
只有单臂成功pair提供selected-success executed-prefix CFM。同一selected trajectory在四个互不重叠的correct K4
conditions下复用相同Beta times/Gaussian noise，分别做完整Writer-to-LoRA gradient，再task内等权、active tasks
等权、full24一次shared AdamW update。

唯一优化对象从65,536参数query map改为2,164,224参数SFMC。query_delta被冻结在LPCP macro25。AdamW、LR
`3e-4`、betas、eps、weight decay、clip1、Nmc4、physical B8、rollout数和task queue保持不变，不做LR/rank/
scale/basis数/seed sweep。

zero-init使cycle1首先打开family memory maps；semantic router在maps非零后获得梯度。cycle1必须立即strict400，
只有closed-loop、retention与breadth过门才允许exact cycle2；不能用“router还会继续成熟”给失败cycle1续命。

## 10. Mechanism, efficiency and engineering gates

GPU formal前至少证明：

1. old LPCP macro25 state只缺新增SFMC tensors，并被显式zero-init；其它tensor逐项严格加载；
2. candidate step0逐tensor、effective BA与fixed action在正常BF16范围内等于LPCP；reference仍等于AS139；
3. `M`对K permutation不变，constant/query-disabled为zero，natural与reversed不同；
4. language变化可改变`alpha`，但`M=0`时language不能产生任何SFMC LoRA residual；
5. cycle1所有八family maps获得finite/nonzero gradient；source policy、LPCP、set/fusion/factor heads保持frozen；
6. Program innovation、factor hidden residual、effective BA与fixed-action response均非zero且无raw A/B gauge解释；
7. 0 teacher/target/validation/test forbidden reads、OOM、nonfinite或collective watchdog；
8. 不重复backbone forward；full24 cycle wall不超过CV-CSD matched wall的`1.25x`，多卡负载仍由动态队列平衡。

实现只保留一个canonical active Writer/reward/evaluator path。新cohesive module放在独立小文件，避免继续膨胀
`model.py`、`temporal.py`或`reward_cycle.py`；CV-CSD旧schema/config/runtime由Git与formal artifacts保存，不保留
可执行parallel version。

## 11. Closed-loop, stability and video-causality gates

cycle1完成后立即做同一K4 schedule single-checkpoint strict paired correct400，并与LPCP143、AS139、PCSD135、
CV-CSD134、v6-fast143、old134/compiler138/online128逐task与success set比较。

cycle1只有同时满足以下条件才允许exact cycle2：

- correct至少144、breadth至少7；
- 相对LPCP lost不超过10且gained大于lost；
- 至少3个suite不下降。

cycle2后“稳定约145”资格必须同时满足：

- cycle1与cycle2各至少144，两点均值至少145，breadth各至少7；
- 两相邻checkpoint churn不超过20、success-set Jaccard至少`.85`；
- 不是一个task或一个suite的峰值掩盖其它能力丢失。

达到稳定资格后，必须对同一final single checkpoint做strict paired六臂：correct、same-task-other、cross-suite
wrong、shuffled、reversed、no-video。same/correct至少`.9`；correct相对每个negative至少`+8/400`，同时报告
paired correct-only/control-only、McNemar、per-task与per-suite。correct没有明确优于shuffled/reversed/no-video时，
即使absolute超过150也不是有效视频教学方法。

四view correction coherence与mean/sample energy必须相对CV-CSD的`.000205/.250155`完整报告；它们用于解释
SFMC是否改变了目标接口，不能替代strict或独立选择checkpoint。

## 12. Fast falsifiers and interpretation boundary

- `M`在constant/query-disabled时非zero：新增路径允许absolute-time/static bypass，架构合同失败；
- factor hidden residual非zero但BA/action仍闭合：冻结`W2`不是足够的policy-effective basis；
- 四viewSFMC correction仍近正交且strict下降：semantic factor commitment没有解决跨video共同方向；
- cross-video coherence提高但correct/retention不增：方向一致性不是held occupancy有效性的充分条件；
- cycle1 absolute提高但cycle2高churn：获得新能力但没有稳定共同积累；
- correct与negative同步提高：language/static task prior主导，视频因果资格失败；
- correct稳定约145、低churn、same鲁棒且negative/no-video显著更低：即使尚未151，也构成owner认可的有价值结果。

负结果只淘汰“frozen LPCP innovation memory + 4-way language semantic route + family-hidden residual + CV selected-
success credit”这一组合；不否定生成LoRA、rank8、literal memory token、few-shot或未来生成LoRA后的task-local RL。

## 13. Completed result and adjudication

本设计已由clean frozen `8994180`完成full24 cycle1与同一single checkpoint strict paired correct400。

训练为24 tasks/48 pairs/96 rollouts，reference/candidate success=`34/34`，8个active tasks、32个四view credit
conditions与128个unique videos；8/8 family maps获得finite/nonzero更新。cycle=`920.555s`，为CV-CSD的
`1.0662x`；三rank任务=`8/9/7`、记录时长max/min=`1.0653x`，没有吞吐、负载、禁读、OOM或nonfinite问题。
但semantic query与basis-key delta仅约`1.7e-9`，zero-init staging使cycle1主要只打开family maps。

strict=`144/400`、breadth7、per-task=`1/3/47/36/0/38/18/1`、per-suite=`4/83/38/19`。相对LPCP143严格=
`128 retained / 16 gained / 15 lost / 241 both-fail`、churn31、net`+1`、Jaccard`.805031`、p=`1`；只有
预注册的lost≤10门失败。该单点恢复了CV-CSD的absolute，但没有证明相邻checkpoint稳定积累。

由于普通trace identity对该微小改写发生消去，终局采用FP64稳定低秩展开
`Δ(BA)=B_candidate·ΔA+ΔB·A_reference`。相对LPCP的all400 effective-BA relative-L2 mean/median=
`2.899e-7/1.066e-9`；q/v/action非零样本=`249/16/1`。first4同task修正pairwise cosine=
`-8.10e-6`、mean/sample energy=`.249995`，没有形成跨video共同方向；candidate相对CV-CSD的几何又几乎
复现CV-CSD到LPCP的历史距离。

因此触发§12的“hidden residual存在但BA/action闭合”与“跨video correction仍近正交”否决分支。最早失败接口是
**continuous SFMC hidden residual经冻结W2写成native public LoRA**：learned semantic router尚未形成，绝大多数
factor residual又被压到原生BF16局部ULP以下，只留下稀疏q-family crossing。144是LPCP邻域的高churn阈值重排，
不是稳定145。按§11终局不续cycle2，不做six-arm controls或参数扫；没有same/wrong/shuffled/reversed/no-video
证据，故不得宣称视频鲁棒性或特异性。

该负结果只否定本设计的完整受检验组合，不否定memory token、rank8、few-shot、SHINE式layer correspondence或
生成LoRA本身。正式训练、strict与终局artifact root记录于`docs/active_session_handoff.md`§14。

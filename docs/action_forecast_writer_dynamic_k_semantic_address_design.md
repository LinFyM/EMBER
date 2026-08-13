# Dynamic-K Semantic-Address Backbone-Memory Rank-8 Writer

日期：2026-08-13。状态：fresh-incompatible successor authority；实现、profile、formal和closed-loop均未完成。

## 1. 要解决的最早失效接口

前一版Dynamic-K backbone-memory rank-8 Writer已经证明三件事：真实图像、exact language与50个Action probes
确实在同一次18层backbone forward中更新8个memory tokens；K1--K4 ragged训练和有向/集合图可以稳定优化；Writer
也确实随video改变完整LoRA。它没有证明这些变化是有用的。

formal fresh`0→50`的functional loss稳定下降，但macro50 strict paired correct400仅`100/400`、breadth4，
相对old134为`82 retained/18 gained/52 lost`。effective BA并不小，却相对old134方向近乎正交，stable rank约1，
八个held task的mean BA方向高度相似。成功集中在三类通用pick/place，依赖精确空间关系、drawer阶段和stove阶段的
任务几乎完全失败。

代码给出比“shared mapper不好”更早且可直接定位的失配：

```text
joint backbone memory Z[f,l,r]
    -> adjacent D[f] = Z[f] - Z[f-1]
    -> terminal G = Z[T] - Z[0]
    -> temporal/set/M2P/mapper
```

absolute `task_hidden`、absolute `probe_hidden`没有消费者，`Z`本身也不进入Program。因而现有图在Procedure之前
主动删除了“哪些对象、关系和目标状态”的Semantic Core，只留下“发生了什么变化”。language只能间接影响这些
差分，不能明确决定应从同一段运动里关注哪个对象/关系。这与用户要求的language回答“任务是什么”、video回答
“正确过程怎样做”不一致。

本successor只修改这一个接口。shared mapper的低stable-rank与跨task共线是已观察到的下游症状，但本轮不同时
修改；否则无法判断低分来自Semantic Core缺失还是参数写出拓扑。

## 2. 保留不变的合同

- 输入仍为exact task language与动态K=`1..4`条same-task action-hidden ordered teacher videos；部署时给几条就
  用几条，不挑video、不平均LoRA。
- stride5后每condition最多64个真实frames；每video内部保序，跨video置换不变，K1走同一网络。
- 真实图像、语言、50个native Action probes、8个memory tokens在同一次π0.5 joint backbone forward中交互；
  原Action probes不能看后置memory。
- 每条video的signed adjacent transition `D`与terminal goal residual `G`仍是唯一dynamic value/content。
- 两层causal temporal、两层set、20×8 M2P、现有shared family mapper、fixed template A/dynamic B均不变。
- 输出仍是一套完整38-target fresh rank8 LoRA；source policy frozen，step0/constant-video functional identity。
- train24、B20 cross-episode functional objective、K-balanced full24 macro、optimizer、scheduler和BF16/TF32均不变。
- 不新增language-only/static LoRA head、scalar gate、confidence、expert bank、第二套LoRA或held outcome入口。

## 3. 唯一架构变量：Semantic Core只作为temporal Query address

对第`k`条video、Action Expert层`l`、rank memory token`r`，已有backbone memory为
`Z_k[f,l,r] in R^1024`。新增：

```text
C_k[l,r] = W_sem( RMS( mean_f Z_k[f,l,r] ) ) in R^256
R[l,r]   = E_layer[l] + E_rank[r]
```

其中`RMS`无可学习偏置，`W_sem`是唯一新增的bias-free `1024→256`投影。temporal self-attention改为：

```text
Q <- W_q( RMS(dynamic content) + R + C_k )
K <- W_k( RMS(dynamic content) + R )
V <- W_v( dynamic content from D/G only )
```

`C_k`不进入K、V、residual、FFN、set value、M2P value或LoRA head。它只回答“在这条任务与video的语义背景下，
哪些有向变化应被选择和组合”。这恢复v5/v6有效的Semantic Core×Procedure原则，但不恢复其旧前端、固定K4、
DCT/phase mean或rank16 compiler。

只进入Q而不对称地同时进入Q/K，是为了避免经RoPE产生由absolute task semantics独立决定的时间kernel。若同一个
`C_k`同时进入Q/K，旋转后的semantic×semantic项可以形成几乎不读D/G的task-conditioned phase template，重演v4
的absolute-time/action-phase shortcut。query-only address仍能让Semantic Core选择D/G key/value，却不给静态语义
自己构造一套时序轨迹的通道。

选择`mean_f Z`而不是单独再建language encoder或visual encoder有三个原因：

1. 它就是SHINE式memory在真实native context中逐层形成的absolute状态，保持layer/rank与未来LoRA topology对应；
2. 它同时受exact language、图像和native Action hypothesis影响，不是空prefix或另造post-encoder token；
3. 每video分别形成address后才做有向编码，跨video set聚合仍能比较多个demo的共同Procedure，而不会先平均frames。

## 4. 为什么不会退化成static或language-only bypass

如果video所有frames相同，则`D=0`、`G=0`。无论`C_k`多强，attention的V、所有residual content与bias-free FFN
都为零，最终Program、LoRA增量与policy action增量必须精确为零。只改变language、首帧或static background可以
改变Q/K，但没有有向video value时不能写LoRA。因此Semantic Core只能调制真实video Procedure，不能自己成为
task-ID route或language-only adapter。

normal video下，correct/shuffled/reversed共享近似absolute Core但具有不同D/G与RoPE顺序；address不能抹去它们
的因果差异。若训练把value忽略掉，输出仍会退化到identity而无法从functional loss获得有效梯度。

## 5. 动态K与高层共同知识

每条video独立计算`C_k`并编码`P_k`，因此每条内部顺序始终完整。跨video set block继续只聚合`P_k`，不平均
raw frames、absolute memory或生成后的LoRA。same-task多video可通过共同的对象/目标address选择各自D/G中相同的
阶段证据，再由置换不变set提取共同Program；K1则把唯一`P_1`走完全相同的set路径。

singleton→stop-gradient(full-set) consistency仍保留，但它不是“让K1模仿K4的最终LoRA”，只约束同网络产生的
Program。它的weight、schedule与当前实现不变，避免把本轮混成新的训练方法。

## 6. LoRA写出判断

rank8、fixed-A/dynamic-B和shared family heads本轮全部保留。这不是因为其健康已被证明，而是因为：

- rank8总BA norm已经足够大，首要故障不是identity；
- 当前没有保存mapper输入级held representation，无法证明信息一直健康到M2P后才丢失；
- 先恢复代码中确定缺失的Semantic Core，能用一次真实closed-loop裁决最早接口。

如果semantic-address训练后，task/order差异直到M2P仍清晰但family-head BA继续stable-rank≈1、action-target能量
低且task mean方向共线，则下一单变量才是layer-direct topology readout；不得在本轮一起加入。

## 7. 快速机制否决

CPU与单GPU机制必须同时满足：

1. constant-frame K1/K4的Program与完整LoRA增量精确为零；只换language也不能越过该identity；
2. video permutation不改变K>1输出；每video shuffle/reverse产生非零Program差异；
3. `C_k`按video、layer、rank有正确shape，且只出现在temporal Query route，Key/Value不读取它；
4. mapper B打开后functional与consistency gradient均能到达`W_sem`、Action Meta-LoRA、temporal、set、M2P、B；
5. step0 template A/zero B仍完整38-target rank8 identity；source policy零trainable parameter；
6. longest-video full24 B20 profile无OOM/nonfinite，K1--K4各6 tasks，macro wall不劣于前版`32.81s`的1.20倍，
   不靠降batch、扩dtype、重复forward或防御性扫描；
7. fresh checkpoint不能误载旧Dynamic-K v1，exact resume仍锁world/topology。

任一失败先修明确工程合同；若修复需要改变scientific variable，则本设计直接退役。

## 8. Formal训练与真实性能门

机制与吞吐通过后，从fresh随机初始化训练`0→50`，每25 macro保存；旧macro50 checkpoint不是初始化。macro50
立即做同一个single checkpoint、correct-video、without-replacement seed7 strict paired400，并逐episode与
old134/compiler138/online128比较；与v6-fast143按aggregate/per-task比较。

macro50快速裁决：

- `<120/400`或breadth`<6`：直接否决Semantic Core address，不resume；
- `120..133`：仍低于old134，除非出现预注册前无法解释的合同违约，否则不resume；
- `134..143`：只有相对old134 gained>lost、至少3 suites不下降且task mean BA共线明显减弱，才允许到100；
- `>=144`：允许exact-resume到100，但目标仍是同一single checkpoint严格`>150/400`；
- 达到`>150`后立即补same-task-other、wrong、shuffled、reversed、no-video严格配对controls；correct必须沿有用
  policy direction胜出，不能只把negative破坏。

所有区间都报告per-task/per-suite、breadth、retained/gained/lost、top3 share、BA/action energy和相邻checkpoint
churn。functional loss、consistency、rank、norm或cosine均不能单独选择checkpoint。

## 9. 工程边界

- 原位修改`writer/memory_program.py`的canonical owner；不新增第二个Program类、mode flag或兼容fallback。
- 使用fresh config、launch/checkpoint/eval schema；历史v1只由Git、frozen worktree和formal artifacts执行。
- 不新增大型诊断框架。机制中间态复用现有`MemoryProgramDiagnostics`，只增加semantic address这一项。
- GPU发射前live检查gpu01/gpu02，使用同一节点最多6张真正有可用余量且健康的A40；部分占用但低util的卡可在
  不干扰他人的前提下使用。有几张用几张，不等待凑6卡。
- 正式run来自clean pushed commit的detached frozen worktree，使用`NCCL_P2P_DISABLE=1`与NUMA/local-rank映射。

## 10. 科学解释边界

成功只支持“absolute Semantic Core作为有向Procedure的address，是Dynamic-K backbone-memory中缺失的必要接口”。
失败只淘汰这一具体Q/K-address formulation；不淘汰dynamic K、few-shot、true backbone memory、rank8或所有
Semantic Core/Procedure fusion。尤其不能因本轮失败直接跳回language-only、平均LoRA或专家路由。

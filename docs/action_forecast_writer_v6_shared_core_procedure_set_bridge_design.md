# V6 Shared-Core Procedure-Set Bridge

状态：2026-08-14 macro25 K4 strict paired correct400=`139/400`、breadth6，按预注册门终局non-pass。canonical实现、
训练、profile、raw rows和机制分析均已封存；不得resume或通过K/LR/seed/temperature sweep救援。上一轮V6 Dynamic Slot-Set Bridge已在macro25 K4 strict paired
correct400得到`130/400`、breadth6，按预注册门终局non-pass；其实现和checkpoint只由Git与formal artifacts保存，
不得继续训练或通过K/LR/seed/temperature sweep救援。

## 1. 当前证据与最早失效接口

上一轮以历史v6-fast macro400为冻结底座，K1对原生v6严格恒等，只在每条video已经独立完成
`evidence -> Core/Procedure -> compiler -> 320 fused policy slots`之后做K轴集合聚合。它完整保住了强底座，但没有
形成有用的few-shot修正：

- K4 strict=`130/400`、breadth6、per-task按Spatial1/3、Object1/3、Goal3/6、Long1/2为
  `1/2/48/32/0/34/13/0`；
- 相对结构恒等的old134为`117 retained / 13 gained / 17 lost`，net`-4`、churn30；四个suite净变化
  `-2/-2/-1/+1`；
- K4对old134的effective-BA cosine mean/median=`.998690/.999275`，norm ratio mean=`.998592`，relative-L2
  mean/median=`.046910/.040562`；
- validation每task前4 states的same-task centered variance/sample从K1`.002281`降到K4`.000246`，约`9.26x`，
  但task mean K1->K4 cosine仍为`.999832`，跨task mean off-diagonal cosine只从`.49935`到`.49749`。

所以Slot-Set确实读取并稳定了多条视频，但它只能在每条video已经被compiler分别压成最终policy program后做小幅
邻域修正。此时对象/目标Core与有向Procedure已经各自经过一次非线性融合和跨slot协调，集合层没有机会让多条
视频先形成共享高层任务理解。最早可检验接口因此从“是否有set”前移到“set位于compiler之前还是之后”。这不
否定动态K、逐video保序、v6 Core/Procedure、rank16 factor heads或one-LoRA部署。

## 2. 单一架构变量

冻结v6-fast的language-axial evidence、Semantic Core、Causal Procedure、原生compiler和八个FactorHeads全部保留。
唯一变化是把同一个约`197k`、bias-free、zero-output集合层从**最终fused slots之后**前移到
**共享Core读出与Procedure读出之间**：

```text
exact language + K=1..4 same-task ordered action-hidden videos
    -> each video independently runs frozen v6 evidence
    -> each video independently forms Semantic Core C_k
       and causally ordered Procedure P_k
    -> frozen native Core reader jointly reads the unordered union {C_k}
       into one shared 320-slot semantic state S_core
    -> for every video k, frozen native Procedure reader uses the same S_core
       to read its own ordered P_k into S_proc,k
    -> one permutation-invariant set layer aggregates {S_proc,k}
       per aligned policy slot into S_proc
    -> frozen native AdaLN Core/Procedure fusion and post-slot block run once
    -> frozen native FactorHeads decode once
    -> one complete 38-target rank-16 LoRA
```

它不是frame拼接：Procedure绝不跨video boundary，frame ordinal在每条video内从零开始并经过原生causal encoder。
它也不是feature或LoRA简单平均：共享Core由原生policy-routing cross-attention从全部video Core tokens中联合读取；
Procedure集合层保留稳定mean backbone，并由task/policy-slot query选择centered跨video residual；最终非线性
Core/Procedure fusion和FactorHeads只运行一次。

## 3. 精确计算图

对同一condition的K条video，原生v6先独立得到：

```text
C_k in R[L,256]              semantic tokens aligned by the exact language
P_k in R[T_k,256]            ordered causal Procedure with native positions
```

原生compiler的320个module/layer/rank routing queries记为`R`。所有video使用同一语言token mask，因此每条video
对Core union贡献相同数量的有效task tokens：

```text
C_set  = concat_set(C_1, ..., C_K)       # no shot embedding or shot order
S_core = NativeCoreReader(R, C_set)
```

随后每条Procedure都由同一个共享语义状态解释，而不是先各自形成完整LoRA program：

```text
S_proc,k = NativeProcedureReader(R + norm(S_core), P_k, native_position_k)
mu        = mean_k S_proc,k
center_k  = S_proc,k - mu
q         = Wq(norm(mu))
k_k       = Wk(norm(S_proc,k))
a_k       = softmax_k(q dot k_k / sqrt(256))
S_proc    = mu + Wo(sum_k a_k * center_k)
```

`Wq/Wk/Wo`沿320个policy slots共享，全部无bias，`Wo` zero-init。最后使用原生冻结的modulation、post-fusion和
FactorHeads：

```text
gamma,beta = NativeAdaLN(norm(S_proc))
S_final    = NativePostFusion((1+gamma)*norm(S_core)+beta, R)
LoRA       = NativeFactorHeads(S_final)
```

K=1时Core union只有原始`C_1`，`center_1=0`，所以无论集合参数如何训练，`S_core`、`S_proc`、`S_final`和76个
LoRA tensors都与原生v6逐步相同。视频集合换位只重排Core union和`S_proc,k`，不改变任何输出；视频内部倒序则
必须重新生成`P_k`，因此仍改变有向Procedure。

## 4. 为什么选择这个边界

不选择frame-level union或重新phase alignment：前者会把不同demo边界误作物理连续过程，后者已经在历史K4
Phase-Aligned v6中得到best108且reversed121，不能换名重跑。不选择继续修改最终Slot-Set：130结果已证明它只
稳定v6原task mean。也不恢复rank8/memory-token mapper：Visual-Value和Full-Factor已分别终局96和91。

当前边界保留三项未被否定的优势：

1. 每条video由历史最强v6前端先提取language-grounded Core和有向Procedure；
2. 不做速度/长度对齐，原生causal Procedure完整保留每条demo内部过程；
3. 多video在最终compiler承诺policy方向之前共享语义并比较过程，仍由已证明policy-effective的原生compiler与
   FactorHeads生成完整LoRA。

它的可证伪假设是：v6的高层memories含有可互补的same-task证据，而上一轮把集合操作放在完整compiler之后过晚。
若本轮仍只产生old134邻域的小幅无净增修正，就应否定这个memory/compiler边界，而不是再前后移动同一set。

## 5. 训练与信息墙

- 加载历史v6-fast macro400并冻结；只训练一个`197120`参数Procedure-Set层；
- train24、full24 task-equal、B20跨episode functional supervision、AdamW与scheduler不变；
- 每macro K1/K2/K3/K4各6，K条teacher videos同task、action-hidden、互不重复并与B20 action episodes错开；
- 每条video继续stride5并使用全部可用帧，不读取teacher action/state/reward/terminal、task ID、filename、pose或
  held actions；
- 不增加singleton imitation、negative、expert、reconstruction、rank/norm/orthogonality或reward loss；
- K1提供结构恒等门但集合梯度恒零，K2--K4训练跨video Procedure共识；部署仍只生成一套LoRA。

warm start仍只检验机制。若它过门，最终方法必须在相同train24信息墙下建立从零可复现recipe；不得把历史v6
checkpoint能力包装成论文最终方法。

## 6. 快速机制门与正式裁决

实现后只做必要检查：

1. K1逐阶段及76个LoRA tensors与native v6完全相同，训练一次后仍相同；
2. K2--K4 video集合换位只允许正常BF16 batched sample-order低位差异，video内倒序显著改变Procedure/LoRA；
3. K>1的shared Core、per-video Procedure slots、set residual与Program->LoRA->functional梯度finite非零；
4. source policy和v6 base无trainable parameter，optimizer只含197120个set参数；
5. full24一macro保持K1--K4各6、B20、所有视频完整且无OOM/nonfinite；
6. profile后fresh macro0->25，立即运行K4 strict paired correct400。

裁决保持简单：

- K4若没有明确超过old134=`134`或breadth低于7，终止，不续训、不扫K/LR/温度/seed；
- `135..150`只有相对old134 gained>lost、至少3 suites不下降且出现旧0-task解锁，才允许一次exact-resume到50；
- K4若严格超过150，先封存single-checkpoint结果，再补K1--K4 scaling以及correct/same/wrong/shuffled/reversed/
  no-video因果controls；
- 无论分数如何，报告per-task/per-suite、breadth、retained/gained/lost、top-task concentration和effective BA；
  不用K1/K4 union或内部健康度替代closed-loop。

## 7. Ownership与生命周期

`src/ember/writer/model.py`继续是唯一active Writer；`legacy_v6_model.py`和`temporal.py`只增加原生compiler内部
读出/融合的窄复用接口，不复制compiler。`slot_set.py`继续拥有唯一跨video集合算子。上一轮post-compiler路径、
schema、config和专属tests在本次替换时删除，由commit`34c0431`及formal artifacts保存；不保留runtime flag或
第二canonical实现。

## 8. 实现与live profile证据

canonical实现保持一个Writer路径，只给native compiler增加Core read、normalized-Procedure read和最终fusion的窄
阶段接口；`fused_slots`由这些接口原样组合，CPU逐tensor等价测试锁住旧图。K1在Procedure-Set output为zero或
nonzero时，76个LoRA tensors均与native v6逐元素相等；K>1 set换位只有正常BF16 sample-order低位差异，真实
video倒序使Program mean abs变化`.217034`。source/v6 base无梯度，唯一trainable为5 tensors / `197120`参数。

full24 B20 live profile root：

`runs/outputs/pi05_v6_shared_core_procedure_set_bridge_profile_r6_b20_97c0de2_gpu01_20260814`

- clean detached `97c0de2`，gpu01物理`0,1,2,4,5,6` world6；macro1/2均24/24 tasks、K1--K4各6；
- macro1/2 wall=`26.01095/24.24948s`，functional=`.10118184/.09570904`，gradient norm=
  `4.3242e-6/6.2963e-6`；
- 全部selected videos未截断，跨两步最长condition=`323`帧；peak allocated/reserved=
  `36,495,013,888/40,758,149,120` bytes；0 OOM/nonfinite；
- macro1→2的Procedure-Set query/key delta norm=`4.1365e-5/3.9905e-5`，证明zero-output首步后真实functional
  credit已展开到完整set，不是永久只训练output head；
- completion、两组完整checkpoint、6 rank RNG state和exit0齐全。profile只seal吞吐/机制，正式训练必须fresh
  macro0，不加载profile checkpoint。

clean detached `502618b`随后在相同gpu01物理`0,1,2,4,5,6` world6完成fresh macro0→25：25条metrics、
macro25 checkpoint、completion与exit0齐全，总耗时`662.7296s`，每macro K1--K4严格各6，loss first/last=
`.10118184/.09565479`、梯度范数范围`3.5721e-6..1.3584e-5`、0 OOM/nonfinite。macro25 checkpoint在
gpu01物理4的K4 correct val8×4固定最长优先面板上完成deployment profile：B8/B16/B32=
`.2233579/.2233132/.2233235 LoRA/s`，全部stable、0 OOM，峰值reserved约`12.95/12.97/13.01GB`；按预注册
最高吞吐规则锁B8。该profile只封存正式evaluator batch，不作方法有效性证据。

## 9. Strict结果与终局裁决

profile-sealed clean detached `64c91a4`在gpu01物理`0,1,2,4,5,6`完成K4 correct strict paired400：

`runs/outputs/pi05_v6_shared_core_procedure_set_bridge_k4_correct400_noreplacement_seed7_macro0025_trainr6_evalr6_64c91a4_gpu01_20260814`

- 400/400 LoRA、72/72 queue shards、400 raw rows、18/18 persistent workers均完整exit0；总wall=`1165.937s`，
  rollout-only=`674.917s`；
- strict=`139/400`、breadth6；per-task按Spatial1/3、Object1/3、Goal3/6、Long1/2为
  `1/4/46/34/0/36/18/0`，per-suite=`5/80/36/18`，top3=`116/139=83.45%`；
- 相对K1严格恒等的old134为`118 retained / 21 gained / 16 lost / 245 both-fail`，net`+5`、churn37、
  McNemar p=`.511376`；四suite净变化=`0/-2/+1/+6`，其中Long1单task净增7，Goal3与Long2仍为0；
- 相对matched K4 post-compiler130为`118/21/12/249`，net`+9`、churn33、p=`.162756`；相对v6-fast143、
  compiler138和online128的aggregate分别为`-4/+1/+11`。

matched K1→K4 effective-BA全400 cosine mean=`.998564`、relative-L2 mean=`.047749`、norm ratio=`1.002122`；
first4/task的same-task centered variance从`.00228114`降至`.00023539`，约`9.69x`，但task-mean cosine仍为
`.9998256`。因此边界前移确实比post-compiler set多获得9个closed-loop净分，却没有形成新breadth或稳定的多task
共同提升。

进一步把同一checkpoint的Procedure-Set `output`归零，在canonical validation8×first4 K4条件重生成LoRA：训练
残差对完整effective BA的relative-L2 mean仅`.0009181`，task-mean仅`.0005745`；相反，保留的无参数
`shared-Core union + per-video Procedure mean`相对K1分别改变`.0396742/.0169820`。所以139几乎全部来自更早的
shared-Core数据流，而不是可学习的后端Procedure-Set。它否定继续训练或放大该后端set，并把下一最早可检验接口
定位到语言对齐的per-video Semantic Core本身：学习跨video共同高层语义必须发生在native Core reader之前。

由于breadth=`6<7`，本设计按门终止；不resume50、不补五臂、不扫K/LR/seed/temperature。负结果只否定当前
`parameter-free Core union + trainable post-readout Procedure set`组合，不否定动态K、逐video保序、Semantic
Core/Procedure分解或在更早高层语义token处学习集合共识。

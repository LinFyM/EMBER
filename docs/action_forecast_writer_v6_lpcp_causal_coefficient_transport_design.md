# V6-LPCP Causal Coefficient Transport

状态：2026-08-15 canonical实现、CPU合同与真实GPU机制/profile均通过，已封fresh full24 cycle1；尚无formal
closed-loop结果。前一轮
Gradient-Open已经按门终局，不得resume或用本设计解释其`141/400`。本轮从sealed LPCP checkpoint fresh建立
不兼容commitment，只改变**video-conditioned factor memory怎样成为policy hidden residual**。

## 1. 最早失效接口

Gradient-Open真实修复了SFMC的首步梯度和native factor量化断点：相对LPCP的all400 effective-BA
relative-L2 mean从`2.899e-7`升到`9.6632e-6`，q/v/action非零覆盖从`249/16/1`升到
`400/399/368`。因此不能再把失败归因于没有梯度、LoRA近identity或Action family没有写出。

但同task四个disjoint correct K4 conditions的训练后增量pairwise cosine仍只有`.0001442`，四个增量取均值
只保留`.250124`能量。strict为`141/400`；相对LPCP143=`128 retained / 13 gained / 15 lost`，suite净变化
`-1/-6/-2/+7`。这说明共享language address虽然给所有视频相同routing，GOSC仍把每个条件的256维
`factor_memory`经full matrix/W1直接当作hidden **Value direction**；不同视频的局部Jacobian仍可把同一成功credit
写成几乎任意的高维方向。

当前最早接口因此是：

```text
same-task ordered-video Program
    -> video-conditioned factor Value direction
    -> common policy-effective task direction
```

本轮不再增加view、训练cycle、LR、rank、memory token或一致性loss。它检验一个结构性假设：**视频应决定“沿共同
task/policy轴写多少”，而不应独自决定“在256维hidden空间朝哪里写”。**

## 2. 保留的数据流

```text
exact task language + K=1..4 same-task ordered action-hidden videos
    -> frozen V6/LPCP one-forward image-language-Action-probe carrier
    -> per-video ordered Procedure
    -> frozen permutation-invariant K-set fusion
    -> LPCP Program and exact AS139 reference Program
    -> video causal memory M = Procedure_LPCP - Procedure_AS139
    -> Causal Coefficient Transport
    -> frozen V6 factor W2 and complete 38-target rank16 LoRA
    -> frozen source policy closed loop
```

以下全部不变：source policy、normalization、train24/validation8 split、stride5、dynamic K=1--4、每video内部
有序编码、K轴集合语义、LPCP query conditioner、AS139强底座、320个policy/rank slots、八factor families、
38 targets、rank16 public LoRA、四view selected-success replay、optimizer、rollout数、dtype和信息墙。

canonical实验仍使用K4；动态K支持是部署图性质，不把K1/K4公平对比写成目标。

## 3. 唯一新模块：video是系数，language是方向

对每个condition与每个policy/rank slot，已有：

- `M ∈ R^256`：LPCP query-conditioned ordered Procedure相对AS139的video causal memory；
- `L ∈ R^256`：exact task language经冻结text-only V6 Core得到的slot-aligned language state；
- `W1_f ∈ R^(256×256)`：冻结V6 factor family `f`已经使用的第一层权重。

令：

```text
u0 = RMSNorm(L)
u1 = S ⊙ u0                       # S为固定交替±1符号
c0 = <M, u0> / sqrt(256)
c1 = <M, u1> / sqrt(256)
h_f0 = GELU(W1_f u0)
h_f1 = GELU(W1_f u1)
```

`c0,c1`是video-derived causal coefficients；`h_f0,h_f1`是同一exact language下固定、family-specific、
policy-slot-aligned的hidden directions。GOSC原有language router保留：

```text
q = Wq RMSNorm(L)
p = softmax(q RMSNorm(K)^T / sqrt(256))       # p0...p3
```

最终只向冻结factor hidden加入：

```text
R_f = (p0 - p1) c0 h_f0 + (p2 - p3) c1 h_f1
```

同task不同视频具有相同`L、p、h_f0、h_f1`，只能通过`c0,c1`改变每个slot的写入。于是每个slot/family的
condition-specific residual被结构性限制在同一个二维task/policy子空间，而不能像GOSC那样由256维`M`任意旋转
Value。全局仍有320 slots × 2 video coefficients，不是一个global scalar gate，也没有把完整LoRA降成rank1。

## 4. identity、gradient与视频不可绕过性

`Wq`严格zero-init，因此step0 `p0=p1=p2=p3=1/4`，所有`R_f`精确为零，输出逐tensor等于sealed LPCP。
第一轮selected-success backward中，`p0-p1`和`p2-p3`对`Wq`的导数非零；只要真实`M`和policy loss有内容，
semantic query首步即可获得gradient。四个basis keys、两个RMSNorm和`Wq`共`67,072` trainable parameters；八个
`256×256×4` family maps被删除，不再保留可绕开二维transport的高维Value支路。

这没有新增language-only LoRA路径：新residual同时乘video-derived `c0/c1`；`M=0`时新写入严格为零。
exact language只决定共享方向，不能单独提供Value。整个LPCP base仍按原架构处理language+video，因此实际
no-video能力必须由paired control实测，不能从局部公式外推。

正确顺序的必要性来自已经通过机制门的LPCP有向路径：`M`由layerwise Action probes、相邻视觉变化和causal
Procedure在真实frame order下产生。reversed/shuffled改变`M`，进而改变`c0/c1`；CCT不另造absolute-time标签，
也不把顺序仅作为训练negative。formal前必须直接验证natural/reversed coefficient与LoRA response不同。

## 5. 为什么不是重复历史失败

- **不是Semantic Factor-Basis**：历史SFB用Core routing选择四个64维wide factor value bases，并训练完整Writer；
  它没有把同task不同视频的部署增量限制到共享的低维output span，最终best127、union193。本轮冻结强LPCP和
  V6 compiler，只改变`M`作为direction还是coefficient。
- **不是PCSD/CV-CSD query-only重跑**：它们的shared matrix作用在每video不同的Procedure query conditioner上，
  所以仍产生高维video-local方向。本轮route读取相同language address，condition变化只能进入两个系数。
- **不是post-compiler平均或cosine loss**：不平均frames/features/LoRAs，不优化漂亮coherence；结构先限定可写
  子空间，selected-success policy credit仍决定写入符号和强度。
- **不是literal memory-token或rank8回退**：memory token与fresh rank8仍是开放方法，但当前证据已把最早缺口
  定位到commitment方向；同时改carrier或payload会混入第二变量。若本轮成立，再独立检验可扩展rank8/memory实现。
- **不是language-only shortcut**：language提供方向，video提供每个slot的有向Value系数；缺任一者均不能产生
  CCT residual。

## 6. 训练合同

从sealed LPCP checkpoint fresh加载base writer、procedure set、layer probe conditioner和`query_delta`；CCT参数
fresh，checkpoint schema不兼容GOSC/SFMC。reference仍是同cached conditioning下关闭LPCP query delta的exact
AS139，candidate为LPCP+CCT。

train24每task两组paired initial states；只对candidate/reference唯一成功trajectory做CFM replay。同一成功
trajectory在四个互不重叠same-task correct K4 conditions下各自完整forward/backward，先task内四view等权，再
active tasks等权。ties、both-success和both-fail为零credit。训练不加入contrastive、coherence、expert、
reconstruction、negative-margin、B20 blind functional或task ID loss。

该选择很关键：若显式强迫四个hidden向量cosine变大，language可以在不理解video时满足surrogate；CCT改的是
部署函数类，训练仍只奖励真实policy success direction。

## 7. formal前快速否决

实现后先做CPU合同与一个真实task的四view机制/profile，不因“架构合理”直接启动full24。

必须同时满足：

1. step0生成LoRA逐tensor exact LPCP；source/base/procedure/LPCP参数0 gradient；
2. 第一次selected-success update后semantic query finite/nonzero，q/v/action effective-BA与fixed-action response
   均非零；
3. 同task四个disjoint correct K4条件的CCT-only effective-BA增量pairwise cosine mean至少`.15`，mean/sample
   energy至少`.40`，显著离开GOSC的约`0/.25`；
4. natural→reversed的`c0/c1`与CCT LoRA均material变化，constant/orderless输入不能伪造相同有向响应；
5. longest-video真实吞吐不慢于GOSC`1.10x`，无OOM/nonfinite，动态队列按wall而非task数量判断均衡。

第3项只授权formal，不选择方法；若结构连共同低维方向都没有形成，就不烧full24。

canonical实现已原位替换旧GOSC commitment：没有保留并行class或runtime schema；trainable从`2,164,224`
降为`67,072`，旧GOSC config只作历史artifact且active loader拒绝。聚焦68项与设置真实LIBERO assets后的完整
CPU suite=`397 passed`；step0 identity、参数边界、video-zero、set invariance、二维shared span、checkpoint
fresh incompatibility和reward/evaluator合同均通过。

clean pushed `3b55feb`在gpu02物理1完成task4 B8真实机制门。四个互斥K4条件共16条正确视频、64次CFM
forward/backward；candidate/reference为`2/1` successes，semantic query delta=`1.4603e-4`，q/v/action native
effective-BA response=`4.4562e-7/8.7685e-7/2.0105e-8`，fixed-action=`.00267335`。cycle=`130.737s`，为
matched GOSC的`.9870x`，peak reserved=`40,751,857,664` bytes，无禁读、OOM或nonfinite。

同一post-update state的稳定FP64低秩分析给出CCT-only effective-BA four-view aggregate cosine=`.563803`、
mean/sample energy=`.672852`，显著超过`.15/.40`并离开GOSC的`.000144/.250124`。q与v分别为
`.580886/.685664`和`.518110/.638505`；action仅`.079285/.309455`，虽因能量很小没有阻断aggregate门，仍是
formal必须追踪的预注册风险。natural→reversed使CCT修正cosine=`.014842`、relative-L2=`1.15358`；把每条视频
全部帧替换为首帧时，factor memory与transported coefficient norm降到natural的`2.42e-5/2.74e-5`，不能伪造
相同有向过程。机制门因此只授权fresh full24 cycle1，不预告closed-loop收益。

## 8. cycle1、稳定性与视频因果裁决

fresh full24 cycle1后立即做single-checkpoint K4 strict paired correct400，并逐task对LPCP143、GOSC141、
v6-fast143、old134、compiler138和online128比较。

只有同时满足以下探索门才exact-resume cycle2：

- cycle1 correct至少`140/400`、breadth至少7；
- 相对LPCP lost不超过15；
- four-view CCT-only correction cosine至少`.15`、mean/sample energy至少`.40`；
- 没有单一suite的灾难性能力清空。

稳定约145的最终资格按owner最新标准：

- cycle1/cycle2都至少142，两点均值至少145，breadth都至少7；
- 相邻checkpoint churn不超过20、Jaccard至少`.85`；
- final相对LPCP lost不超过10且gained不少于lost，多task而非单一suite净积累；
- 不用checkpoint union、per-task挑点或平均LoRA。

只有稳定资格通过，才对同一final checkpoint做strict paired六臂：correct、same-task-other、wrong、shuffled、
reversed、no-video。same-task-other应不低于correct 8分以上；correct相对每个negative/no-video至少净高10分，且
paired gained>lost在至少三个suites成立。最终报告paired transitions与McNemar，不用aggregate差单独冒充因果。

单点150以上但高churn或没有video specificity仍不合格；稳定约145且跨video/因果门完整，可以成为有价值成立
结果并进入后续从零recipe复验。

## 9. 负结果边界

- 机制门不提高cross-video span：否定当前两轴`<M,L>` coefficient transport，不否定其它causal bottleneck；
- coherence提高但closed-loop下降：说明共同子空间仍未对齐held occupancy，不能靠加轴数、scale或coherence loss救；
- absolute尚可但cycle1→2大换手：说明低维commitment没有解决多task共同积累，应重审shared optimizer/credit；
- stable correct通过但六臂无margin：说明方法仍主要依赖base/language shortcut，不具EMBER视频学习资格；
- 任何失败都不外推否定memory token、rank8、few-shot、V6 carrier或生成LoRA本身。

本轮完成前不命名下一个successor，不做参数sweep。

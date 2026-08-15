# V6-LPCP Gradient-Open Semantic Commitment

状态：2026-08-15 terminal non-pass authority。本文建立在SFMC cycle1终局之后，只改变semantic factor
commitment的零输出参数化。LPCP视频载体、AS139强底座、K4 cross-video selected-success credit、rank16
public LoRA、八factor families、优化器和全部信息墙保持不变。

## 1. Decision

SFMC的设计意图没有被closed-loop结果整体否定，但其具体初始化存在一阶梯度断路：

```text
R_f(M,L) = sum_b alpha_b(L) V_f,b M
V_f,b = 0 at initialization
```

这保证step0严格等于LPCP，却同时使`dR/dalpha=0`。因此cycle1只更新family maps，semantic query与basis
keys几乎不动；随后很小的continuous hidden residual又大多没有跨过native public LoRA的局部量化边界。

本轮把它替换为一个**严格零输出、但首步梯度打开**的anchored commitment：

```text
exact language + K4 ordered action-hidden videos
  -> frozen LPCP evidence / Core / Procedure / K-set
  -> K-set innovation memory M
  -> zero-init semantic query + nonzero semantic keys
  -> trainable zero-init family delta maps
     + frozen V6-W1 policy-aligned address anchors
  -> frozen V6 factor W2
  -> one complete 38-target rank16 LoRA
```

唯一主要变量是commitment的初始化与等价函数形式。不是续训失败SFMC checkpoint，也不改变credit views、
rank、LR、dtype、basis数或视频数量。

## 2. Evidence selecting this interface

SFMC full24 cycle1完成24 tasks、32 credit conditions与128 unique videos，八family maps全部更新，wall仅为
CV-CSD的`1.0662x`；因此GPU负载、reward credit、video carrier和训练图都不是失败源。strict=`144/400`，
但相对LPCP143为`128 retained / 16 gained / 15 lost`、churn31，只是高换手邻域重排。

稳定FP64差分进一步给出：

- all400 effective-BA relative-L2 mean/median=`2.899e-7/1.066e-9`；
- q/v/action非零样本=`249/16/1`；
- semantic query/basis-key参数delta约`1.7e-9`；
- first4同task correction cosine约0，mean/sample energy约`.25`。

所以最早失败接口是`continuous hidden residual -> frozen W2 -> native public LoRA`，并且该接口前还有明确的
zero-init router staging。当前证据不支持重做已通过的LPCP carrier，也不支持增加view、继续cycle2或做
LR/scale/dtype小扫。

## 3. Retained graph and information wall

本轮完整保留：

- exact task language与K4 same-task、action-hidden、内部有序teacher videos；
- frame stride5、逐video causal encoding、video间置换不变K-set；
- V6 Semantic Core、有向Procedure、LPCP 18-layer Action-probe conditioner及其macro25权重；
- `M = FrozenSet(P_LPCP) - FrozenSet(P_AS139)`的layer/rank aligned innovation memory；
- AS139/V6 fusion、38 targets、八factor families与完整rank16 public topology；
- train24两组paired states、AS139/LPCP两臂、四个disjoint correct K4 credit views、selected-success replay；
- AdamW、LR `3e-4`、Nmc4、B8、task equal aggregation、source policy和official rollout。

language只提供semantic address；所有新增factor Value仍乘以video-derived `M`。`M=0`时新增LoRA residual严格
为0，不存在language-only输出。reference只在训练中产生paired success trajectory，不进入部署。

## 4. Exact gradient-open parameterization

对每个condition的320个policy slots，semantic address为：

```text
q_s       = W_q RMSNorm(L_s)
alpha_s,b = softmax(q_s dot RMSNorm(K_b) / sqrt(256)), b=1..4
```

本轮将`W_q` exact-zero初始化，`K_b`沿用确定性nonzero初始化。因此step0的`alpha`逐slot严格为`1/4`。

每个factor family保留四个trainable delta maps `D_f,b`，全部exact-zero初始化。另从对应冻结V6 factor head的
第一层`W1_f`构造四个不训练、无额外scale的policy-aligned anchor operators：

```text
A_f,0(M) =  W1_f M
A_f,1(M) = -W1_f M
A_f,2(M) =  W1_f (S M)
A_f,3(M) = -W1_f (S M)
```

`S`是固定的balanced signed diagonal，operator norm为1；它只提供与identity分支不同的第二个坐标保持方向，
不增加可调幅度、随机seed或新expert bank。anchors使用已经共同产生LPCP143的冻结V6 `W1_f`，而不是任意wide
decoder或raw A/B residual。

最终hidden residual为：

```text
Delta-h_f = sum_b alpha_b D_f,b M
          + sum_b (alpha_b - 1/4) A_f,b(M)
```

step0时两项分别严格为0，不依赖正负大数的浮点抵消：第一项因`D=0`为0，第二项因
`alpha-1/4=0`为0。因此candidate逐tensor等于LPCP。

但首步导数已经打开：

```text
d Delta-h_f / d D_f,b = (1/4) M                    != 0
d Delta-h_f / d W_q   = d alpha/dW_q * A_f,b(M)   != 0
```

basis keys与两个RMSNorm可在query打开后的下一cycle继续学习；本轮关键是cycle1结束时semantic query本身已经
material更新，而不是像SFMC一样等到失败cycle之后才开始获得有效梯度。

## 5. Why this is the narrow successor

- **不是续训SFMC**：从sealed LPCP macro25重新fresh初始化commitment与optimizer，不加载SFMC cycle1 state。
- **不是量化补丁**：不改BF16/TF32、不扩dtype、不dither、不重复forward；修复的是训练函数的一阶可学习性。
- **不是scale sweep**：anchor幅度由冻结V6 `W1_f`和norm-1 signed transform确定，没有新增scalar gate。
- **不是新视频前端**：LPCP carrier、Core、Procedure、K-set和four-view credit全部冻结。
- **不是raw factor bank**：residual仍进入真实factor hidden owner，再经冻结W2生成唯一public LoRA。
- **不是language bypass**：anchors、delta maps和semantic address没有`M`都不能输出任何新增Value。
- **不是漂亮几何目标**：coherence、rank和norm只验证接口，方法仍由strict closed-loop裁决。

该设计保留SFMC正确的layer/rank/family对应和连续semantic routing，只修复其已由真实结果证实的zero-init
staging与sub-ULP writeout接口。

## 6. Expected coexistence and video behavior

首步family delta提供shared、video-dependent更新；同一步semantic query利用exact language选择四个冻结
policy-aligned anchors。因同task不同correct videos共享language address，而Value仍来自各自有序`M`，模型可学习
共同task direction，同时保留每条视频的过程证据。不同tasks通过连续address组合共享bases，不需要task ID或
hard expert route。

正确顺序仍在结构中必要：`M`来自每video causal Procedure和K-set后的LPCP/AS139差；shuffle/reverse必须重排
真实frames并重新生成`M`。本设计不把negative LoRA人为推坏，correct是否沿有用方向获益只由最终六臂闭环判断。

## 7. Implementation ownership and lifecycle

现有`src/ember/writer/factor_commitment.py`继续拥有唯一commitment实现；不新增第二个Writer模块。当前SFMC
schema/config/runtime由Git commit`8994180`和formal artifacts保存，active schema原位替换为本设计，避免并行
可执行路径。

实现只需要：

1. 将semantic query改为zero-init；
2. 把现有family maps解释为zero-init delta maps；
3. 在同一`hidden_residuals`中加入冻结V6-W1 anchored address term；
4. 更新fresh-incompatible config/checkpoint/evaluator schema和聚焦机制测试。

不增加backbone forward。anchor只增加小型factor hidden projection，full24 reward cycle必须保持在SFMC matched
wall的`1.25x`以内。

## 8. Mechanism and efficiency gates

进入formal前必须证明：

1. sealed LPCP state完整加载；新commitment state全部fresh，SFMC cycle1不能被误载；
2. step0 public LoRA、effective BA与fixed action在正常native dtype下严格等于LPCP；
3. `M=0`、query-disabled/constant输入不能产生新增residual；K-set仍置换不变，natural/reversed仍material不同；
4. 第一次真实selected-success update中八family delta maps和semantic query均finite/nonzero更新；
5. post-update q、v、action public factors都发生非稀疏变化，effective BA与fixed-action response均非零；
6. source policy、LPCP、set/fusion、W1/W2和全部非commitment参数保持frozen；
7. 不重复backbone forward，0 forbidden read/OOM/nonfinite/watchdog；
8. task4 wall和full24 wall均不超过matched SFMC的`1.25x`，多卡仍按真实task cost动态平衡。

跨video correction cosine与mean/sample energy必须报告，并与SFMC的约`0/.25`比较；它们只判断新接口是否真的
打开，不选择checkpoint。若query仍近零或action family仍只有孤立ULP crossing，则本设计在formal前即失败。

### 8.1 Sealed implementation and live mechanism evidence

唯一canonical commitment已原位替换，fresh-incompatible config/checkpoint/deployment schema与实现由
`5b14c89`封存；正确LIBERO assets下full CPU=`396 passed`。gpu02物理1的task4 B8真实smoke完整exit0：
4 rollouts、1个active task、4个互斥K4 credit views、16 unique videos、64次functional forward/backward，
8/8 family maps更新。semantic query parameter-delta RMS=`1.1978575e-4`，而matched SFMC只有
`1.7564392e-9`，所以首步router不再关闭。

同一次post-update decode给出native effective-BA response：q=`6.616899e-7`、v=`9.151695e-7`、
action=`4.890831e-8`，三组全部跨过public BF16 factor边界；总BA=`6.939059e-7`，为SFMC
`3.521205e-8`的`19.7x`。fixed-action response=`.00270327`，与SFMC `.00281061`同量级，没有以破坏
action制造内部增益。cycle=`132.458s`，为SFMC `139.420s`的`.950062x`，peak reserved相同为
`40.762GB`。因此第8节机制与效率门通过，可以启动fresh full24 cycle1；这些证据不提供absolute、retention、
checkpoint稳定性或视频因果性能结论。

### 8.2 Fresh full24 cycle1 evidence

clean detached `eb543d3`在gpu01物理`2/4/5/6/7`以world5 fresh完成：24 tasks/48 paired states/96
rollouts，candidate/reference successes=`33/31`、candidate/reference-only=`6/4`、10 active tasks覆盖四suite，
40 credit views、160 unique correct videos、387 replay chunks、1,918 executed action steps。8/8 family maps
更新，semantic query delta=`6.9499e-5`，仍为SFMC约3.96万倍。

5个post-update deployment probes的effective-BA全部非零；q/v为`5/5`，action为`3/5`。BA RMS mean=
`1.4991e-7`。这证明gradient-open route与v写出在full24仍成立，但action写出尚不均匀，不能由机制指标推断
closed-loop。cycle=`581.924s`；rank任务数=`3/5/2/5/9`是cost queue结果，其recorded wall=
`485.314/560.082/462.083/482.851/538.237s`，max/min=`1.2121x`。相对SFMC world3 wall=`.6321x`，约
95%理想扩展效率；峰值reserved=`40.764GB`，禁读/OOM/nonfinite/watchdog为0，world5 checkpoint与completion
完整。由于step0相同但world size/GPU reduction会造成正常低位轨迹分叉，`33/31`与SFMC `34/34`不是严格性能
比较；下一步必须做同checkpoint strict400。

## 9. Closed-loop and stability adjudication

full24 cycle1完成后立即做同一K4 single-checkpoint strict paired correct400。cycle1只有同时满足以下条件才允许
exact cycle2：

- correct至少144、breadth至少7；
- 相对LPCP143 lost不超过10且gained大于lost；
- 至少3个suite不下降。

稳定约145资格继续使用owner最新标准：

- cycle1与cycle2各至少144，两点均值至少145，breadth各至少7；
- 相邻checkpoint churn不超过20、success-set Jaccard至少`.85`；
- 末端checkpoint按预注册规则使用，不按最高分挑选；
- 增益不能由单一task或suite掩盖其它能力丢失。

达到稳定资格后，对同一final checkpoint做strict paired六臂：correct、same-task-other、cross-suite wrong、
shuffled、reversed、no-video。same/correct至少`.9`；correct相对每个control至少`+8/400`，并报告paired
correct-only/control-only、McNemar、per-task和per-suite。

稳定145且视频因果资格完整，可视为有价值成立结果；单点150以上但高churn或没有video specificity仍不合格。

### 9.1 Cycle1 strict result and terminal interpretation

同一clean `eb543d3` cycle1 checkpoint的K4 strict paired correct400完整完成：400套LoRA、60/60 queue jobs、
400 raw rows与15/15 worker exit0；wall=`1405.667s`、effective=`.28456 rollout/s`。结果为
`141/400`、breadth7、per-task=`1/3/48/29/0/36/23/1`、per-suite=`4/77/36/24`、top3 share=
`.80142`。

相对直接初始化邻居LPCP143严格为`128 retained / 13 gained / 15 lost / 244 both-fail`，churn28、net`-2`、
Jaccard=`.82051`、McNemar `p=.85055`。suite净变化=`-1/-6/-2/+7`：Long1以`12 gains / 5 losses`
净增7，但Object3以`1/7`净丢6、Goal6净丢2、Spatial3净丢1；breadth没有增加，Goal3仍为0。相对SFMC144
则为`124/17/20`、churn37、Jaccard=`.77019`。因此这不是多task共同积累，而是更明确的Object/Goal/Spatial
能力向Long1换手。cycle1的absolute、lost、net与suite四项续训门均失败，不运行cycle2或六臂controls。

稳定FP64低秩差分同时证明本轮确实修复了SFMC的写出断点，而非再次回到identity：相对LPCP的all400
effective-BA relative-L2 mean=`9.6632e-6`，约为SFMC的`33.3x`；effective-BA RMS mean=`1.6261e-7`，
q/v/action非零样本=`400/399/368`。但gained/lost relative-L2 mean仅=`8.7461e-6/9.2809e-6`，幅度仍不能
区分有用与有害改写。8 tasks各取前4个disjoint correct K4 sets后，增量pairwise cosine mean=
`.0001442`、mean/sample energy=`.250124`，仍等同四个近正交video-local方向。

所以第6节“共享language address足以把不同正确视频的Value提交为共同task direction”的核心假设被真实证据
否定。最早失败接口现已从`hidden residual -> native factor ULP`后移到：

```text
shared semantic address + cross-video selected-success credit
  -> video-conditioned factor Value / effective BA direction
```

gradient、router、q/v/action public写出都已打开，但每个video condition的Jacobian仍把同一成功credit编译为
近正交局部修正；shared checkpoint因此只在tasks之间重新分配边界成功。下一设计必须直接形成跨video可复现的
causal task Program，再进入policy-aligned compiler；不能继续放大anchor、增加cycle、扫LR/rank/scale，或只用
loss/coherence选择方法。

## 10. Fast falsifiers and negative-result boundary

- step0不再exact LPCP：anchored参数化实现错误；
- family maps有梯度但semantic query仍为零：gradient-open核心假设失败；
- continuous residual非零但v/action public factor仍不动：W1-anchor到W2/native factor接口仍不足；
- cycle1 correct低于144、breadth低于7或lost大于10：不续cycle2、不改scale/LR/rank/anchor seed；
- cycle1/2仍高churn：梯度打开没有解决shared checkpoint共存；
- correct与wrong/shuffle/reverse/no-video同步：video Value没有沿正确过程产生有用方向。

负结果只淘汰“LPCP innovation memory + W1-aligned anchored gradient-open commitment + one/two CV selected-success
cycles”这一组合；不否定memory token、few-shot、rank8、生成LoRA或未来从零训练recipe。

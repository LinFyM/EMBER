# V6-LPCP Direct Joint Native-Factor Residual

状态：2026-08-15 active formal-cycle1-ready authority。简称 `DJNFR`。task4 -> validation8机制门已强通过，
仅授权从sealed LPCP macro25 fresh启动full24 cycle1；不续SJNV、PAFS或NPVC checkpoint，也尚未授权cycle2。

## 1. 决策与最早失效接口

SJNV在validation8中并不是没有形成共同表示：shared gate cosine=`.940337`，continuous factor-hidden residual
cosine/energy=`.941165/.923978`。真正断裂发生在：

```text
coherent continuous hidden residual
    -> frozen V6 W2
    -> native BF16 public A/B delta
```

经过冻结W2后，raw factor delta cosine/energy骤降到`.021353/.265925`，action factor cosine仅`.002672`；
effective BA只部分恢复到`.201903/.396448`。因此继续放大SJNV gate、增加两个hidden axes、扫LR/scale或只改
optimizer，都没有针对最早断点。

本轮唯一主要因果变量是：**不再把新增视频知识压成少量系数后交给冻结W1/GELU/W2，而让slot-aligned
joint video-language Value经factor-shape-matched trainable heads直接写public A/B residual。**

## 2. 保留项

以下逐项冻结：

- exact task language + dynamic `K=1..4` same-task action-hidden ordered videos，formal仍以K4裁决；
- stride5、每video内部有序编码、跨video permutation-invariant set aggregation；
- AS139/LPCP强底座、同一次真实image+language+50 Action probes context forward、18层probe carrier；
- sealed LPCP `query_delta`、Procedure-set attention和已有native probe Value；
- 320个`policy-layer/rank` slots、38 targets、完整public rank16 LoRA；
- source policy、normalization、split、four-view selected-success reward、optimizer、rollout数、dtype和信息墙；
- Writer rollout前运行一次，部署只有一套LoRA，无expert route、checkpoint union或LoRA平均。

本轮不加literal memory tokens、不改rank8/rank14、不加第二adapter lane、不做SVD/refactorization、不加额外loss、
expert target、更多rollouts、support projection或gradient surgery。

## 3. 输入到direct factor的完整数据流

LPCP对每条video在真实context中读取18层Action-probe hidden，先做相邻frame delta，再做causal temporal
controller，得到每video的：

```text
H_k in R^(320 x 256)
```

同一condition继续用sealed Procedure-set attention逐slot聚合：

```text
M[s] = sum_k alpha[k,s] H_k[s]       # [320,256]
```

exact language经V6 Core得到同slot对齐状态`L`。唯一direct payload为：

```text
X = (M elementwise_mul RMSNorm(L)) / sqrt(256)  # [320,256]
```

`M`提供不可替代的video动态Value；`L`说明任务目标与应关注的关系。没有video时`M=0 -> X=0`，language不能
独立写新增LoRA。constant video的probe delta为0，因此也不能产生新增residual。

每个expert layer的16个rank slots分别进入四个共享shape heads：

```text
q_A: 256 -> 1024       q_B: 256 -> 2048
v_A: 256 -> 1024       v_B: 256 -> 256
```

action-in与action-out的各16 slots分别进入对应四个heads：

```text
action_in_A:  256 -> 32       action_in_B:  256 -> 1024
action_out_A: 256 -> 1024     action_out_B: 256 -> 32
```

八个head均无bias、exact-zero初始化，总trainable：

```text
256 * (1024+2048+1024+256+32+1024+1024+32) = 1,654,784
```

head直接输出对应rank row/column并加到LPCP生成的同一A/B tensor；随后只有一次native BF16 public cast。没有
冻结W1/GELU/W2，也不生成第二套LoRA再融合。step0所有direct heads为0，所以public 76 tensors逐tensor exact
LPCP。

## 4. 正确顺序为何仍是必要输入

direct head没有绕开有向视频前端。`M`来自：

```text
real ordered frames
 -> per-layer Action-probe states
 -> adjacent delta
 -> causal temporal controller with real ordinals
 -> per-video Program
 -> K-set aggregation
```

reverse/shuffle先改变相邻转移和causal state，再改变`H_k`、set attention、`M`、`X`和最终A/B。direct readout
只移除末端compiler瓶颈，不把frames平均成静态特征。机制门仍要求natural相对reversed产生material BA变化；最终
资格只认correct相对wrong/shuffled/reversed/no-video的closed-loop paired margin。

## 5. 为什么本轮没有直接加literal memory token

memory token仍是重要候选，不是被否定。SHINE的关键不是token名称，而是让每层memory payload覆盖该层LoRA
参数量，再经layer/token M2P直接reshape为A/B。对EMBER当前rank16 Action Expert，一层q/v A/B共有`69,632`
个参数，hidden width为1024，因此SHINE式容量匹配需要`ceil(69632/1024)=68`个memory tokens；rank8则是34。

历史Dynamic-K只使用8个backbone memory tokens，把它们压成`20 x 8 x 256` Program，再经共享mapper生成约64万
rank8参数。它验证了真实prefix、memory、temporal、K-set和M2P能运行，但不是容量匹配的SHINE直写，因此其
100--102分不能否定未来的68-token direct reshape。

当前不立即采用68 tokens有三个证据原因：

1. LPCP carrier已经在one-forward、reverse/static和143 absolute上通过，SJNV断点位于末端public factor写出；
2. literal memory会同时改变joint forward、carrier、temporal payload、M2P和输出decoder，无法判断收益来自哪里；
3. sealed LPCP提供当前唯一强而可精确保留的step0，direct residual能先单独检验“绕过W2”是否足够。

若DJNFR的`X`在held跨video共同、但direct A/B仍不能形成共同方向，才说明`320 x 256` slot payload或共享linear
readout本身不足，下一架构才升级为capacity-matched Action-memory grid + row/column M2P + direct reshape。那时68
不是随意token数，而是rank16 payload公式的结果。

## 6. 与Doc-to-LoRA、SHINE和历史Direct-Family-B的关系

- Doc-to-LoRA让每层rank latents以A/B output heads写参数。DJNFR采用相同的factor-shape-matched direct emission
  原则，但输入是video-required policy slots，且只写LPCP residual以保留强底座。
- SHINE避免wide heads，以容量匹配memory grid直接reshape。DJNFR不是一比一复制SHINE；它先检验EMBER当前最早
  W2断点，SHINE-style grid是预注册的下一升级条件。
- Dynamic-K Direct-Family-B从fresh弱rank8图出发、A基本固定，strict102。DJNFR保留LPCP143、rank16和完整A/B，
  所以不是恢复该低分架构。
- 参数增加不是独立目标。1.65M参数恰好来自八种public factor shape，放在已定位的失效边界；不增加hidden
  atoms、rank或无关encoder capacity。

参考：

- SHINE: https://arxiv.org/abs/2602.06358
- Doc-to-LoRA: https://arxiv.org/abs/2602.15902

## 7. 多task共存假设

NPVC把task/video压成少量shared coefficients，再经冻结axes输出；full24后task4 coherence从`.5929/.6792`坍塌到
`.0569/.2951`。DJNFR不为task建parameter bank，而让所有tasks共享同八个heads，同时以每个condition真实
`X[s]`决定其更新作用位置：

```text
delta W_f = reward cotangent_f outer X
delta factor_f(condition) = delta W_f X(condition)
```

因此不同task可通过不同joint payload在同一head中形成不同映射，而不是task ID route或独立expert。这个假设必须
由full24 active-task gradient coexistence、post-train task4/held geometry和strict retained/gained/lost共同裁决；
较高参数量本身不构成共存证据。

## 8. 实现与fresh合同

canonical实现原位替换SJNV runtime：

- `factor_commitment`只含八个zero-init direct heads；
- `decode_program`先生成exact LPCP，再将direct rows按既有layer/rank/factor ownership加到同一76 tensors；
- LPCP cold start只允许缺失全部新head；任何partial/new checkpoint都拒载；
- reward optimizer只训练`factor_commitment.heads.*.weight`的1,654,784参数；
- checkpoint、evaluator family与deployment kind fresh-incompatible；
- 不重复backbone forward，不保存第二套部署LoRA。

定向CPU已覆盖step0、video-required、K-set、八family梯度、76-tensor shape/ownership、cold-start拒载和reward
gradient aggregation。完整CPU=`399 passed`、compileall通过，architecture guard无hard violation；active source
diff净增长8行，没有建立并行实现。真实GPU机制尚未完成，不能据此声称有效。

## 9. formal前真实机制门

先在train task4做一次与NPVC/SJNV matched的B8 selected-success update，再用同一state只读视频生成task4和
validation8每task四个disjoint correct K4 conditions；held不读actions/reward/outcome。

必须同时满足：

1. step0 public LoRA逐tensor exact LPCP，base/source/LPCP 0 gradient；八heads首步均finite/nonzero；
2. post-update q/v/action raw factors、effective BA与fixed-action response均非零；
3. train task4 four-view BA cosine至少`.40`、mean/sample energy至少`.55`；
4. validation8至少6 tasks达到cosine`.15`、energy`.40`；aggregate至少`.30/.48`；
5. validation8 raw factor delta aggregate cosine至少`.30`，action BA cosine至少`.15`；
6. held/train pure-DJNFR BA L2至少`.30x`，不能以train-only巨大写出过门；
7. natural/reversed BA relative-L2至少`.50`，constant/natural BA不超过`.005`；
8. cycle wall不超过SJNV`1.10x`，无OOM/nonfinite/禁读，仍只有一次context forward。

任一held、action、raw-factor或效率门失败即终局，不full24、不strict、不扫LR/scale/head width/rank。

### 9.1 真实结果

clean pushed `e756fa1`在gpu01物理5完成task4 B8 smoke：candidate/reference=`2/1`，1个discordant pair；八head
梯度与参数delta、q/v/action native BA和fixed-action response全部非零，cycle=`139.069s=1.02439x` SJNV，峰值
allocated/reserved=`36.488/40.756GB`，0禁读/OOM/nonfinite。

同一checkpoint只读task4与validation8的四个互斥correct K4 sets：

- task4 effective-BA cosine/energy=`.813895/.794975`；
- validation8 effective-BA=`.776695/.768990`，8/8 tasks过门；
- joint Value=`.803616/.831027`，continuous direct rows=`.933698/.918759`；
- native public raw factor=`.644605/.697686`，action BA cosine=`.557652`；
- held/train BA L2=`.469796x`；reversed BA relative-L2=`1.222871`；constant/natural=`1.762e-6`。

所有预注册门同时通过。SJNV的最早断点确实被绕过：共同方向从joint payload连续传到direct rows、native public
factors与effective BA，而不是只在hidden中变漂亮。因此formal cycle1已解锁；这些内部证据仍不替代strict400。

## 10. full24与closed-loop裁决

机制门通过后，才从sealed LPCP fresh进行matched full24 cycle1。cycle1后立即K4 strict paired400。只有同时满足：

- correct至少`142/400`、breadth至少7；
- 相对LPCP143 lost不超过15，无suite清空；
- post-train task4与held共同方向不坍塌；
- gained/lost改写开始可分，active-task共存没有单task支配；

才允许exact-resume cycle2。稳定资格保持两点均至少142、均值至少145、churn不超过20、Jaccard至少`.85`、
final相对LPCP lost不超过10且gained不少于lost。首次达到约145且retention过门，立即补same-task-other、wrong、
shuffled、reversed、no-video六臂，不能等到150。

## 11. 快速否决与负结果边界

- `X`跨video不共同：断点仍在slot payload，direct heads不应靠coherence loss修；升级carrier/memory才有依据；
- `X`共同但raw A/B不共同：共享256-to-factor readout不足，支持capacity-matched memory direct reshape；
- raw A/B与BA共同但strict下降：方向不符合held on-policy occupancy，不以LoRA健康度继续；
- task4机制健康但full24后坍塌：进入显式shared-update/coexistence接口，不回头调head scale；
- correct高但六臂无视频margin：方法仍不具有效教学视频claim。

负结果只淘汰“LPCP/NPVC carrier + joint `M*L/sqrt(256)` + 八个shared zero-init direct factor heads + matched
one-cycle selected-success credit”组合；不否定memory token、capacity-matched M2P、rank8、few-shot、reward credit
或生成LoRA。

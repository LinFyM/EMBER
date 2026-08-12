# Causal–Goal Interaction Key Joint Credit

## 1. Decision

本设计是PVJFC唯一一次macro0机制profile以`regularized condition=597.861>200`终止后的active
successor authority。它只替换condition key的最后固定组合：

```text
PVJFC / PICK-GC:
  psi = Normalize([goal, causal])

CGIK-JC:
  psi = Normalize([causal, Normalize(goal * causal)])
```

其中`*`是两个独立固定JL投影后、逐坐标的Hadamard interaction。其余全部保持PVJFC合同：两条不同、
原序、action-hidden训练视频，各自完整生成one-shot Program cotangent；共享同一个跨episode B20与policy RNG；
48 correct和48 matched negative以`1/2+1/2`权重进入同一个full96 continuous solve；部署仍只输入一条视频并
生成一套完整38-target rank-16 LoRA。

本方法简称 **CGIK-JC**：Causal–Goal Interaction Key Joint Credit。当前只授权实现、CPU门和一次discarded
full96机制profile。profile通过前不授权formal训练、deployment结果或closed-loop评测。

## 2. The earliest failed interface

PVJFC profile已经闭合：

- correct feature rank=`48`，full rank=`96`；
- 24/24 tasks的primary和companion都沿各自cotangent下降，四个suite总方向也都下降；
- negative/correct Program motion ratio=`.072657`；
- Program到LoRA A/B、effective BA和fixed action全部非零；
- total wall=`51.543s`，无OOM/nonfinite/forbidden read；
- 唯一失败是`.01` damping下full96 regularized condition=`597.8607>200`。

因此最早失败不在视频读取、paired cotangent是否有内容、Program memory、v6 compiler、rank、action传递或吞吐，
而在：

```text
unit-normalized PICK-GC condition rows
  -> highly repeated same-task and correct/reverse directions
  -> full96 Euclidean row Gram
```

retained train24×50 action-hidden cache可以精确重建本次macro0选中的48条correct key。它不能精确重建raw-frame
shuffled、target-language wrong或完整full96谱，因此下面只把correct-side结果当作design-selection evidence，
full96仍必须由真实raw-frame profile裁决。

| correct rows | rank | min eigenvalue | max eigenvalue | `.01` regularized condition |
| --- | ---: | ---: | ---: | ---: |
| primary 24 | 24 | `.13385` | `2.50369` | `17.47` |
| companion 24 | 24 | `.14101` | `2.41422` | `16.05` |
| primary+companion 48 | 48 | `.009582` | `4.88759` | `250.11` |

两条same-task selected key的cosine mean/median=`.89895/.91010`；pair-symmetric能量约占`94.95%`，
pair-difference只有`5.05%`。把两组原样当作48条独立等能rows，会以很小的private eigenvalues换来形式rank48。
negative又加剧相同模态：live profile中reversed correct-negative cosine均值约`-.8174`；Gram只看外积，反号不会
抵消能量。这解释了correct-only已经`250.11`，full96进一步到`597.86`。

这个诊断也说明不能选以下修补：提高damping、放宽condition门、全局去均值、PCA whitening或强制两view相同。
它们分别改变数值容忍、只美化度量，或重演CVEG的hard E；都没有让condition本身更依赖正确的有向过程。

## 3. Single structural variable

对phase-aligned frozen-policy innovation `x_0...x_15 in R^3072`，沿用PICK-GC的两个descriptor：

```text
whole  = Mean_p x_p
goal   = Mean_{p in terminal quartile} x_p - whole
causal = Mean_p (Sum_{u<=p}(x_u - whole) / sqrt(p+1))
```

沿用现有seed `20260810`生成的两个独立、固定、无bias、row-normalized FP32 JL block：

```text
u_g = ZeroL2(P_g goal)       in R^128
u_c = ZeroL2(P_c causal)     in R^128
u_x = ZeroL2(u_g * u_c)      in R^128
psi = ZeroL2([u_c, u_x])     in R^256
```

没有新参数、optimizer、loss、bank、task route或部署输入。两个JL block仍不进入checkpoint。`u_g*u_c`是固定
随机bilinear sketch：它不假设两个raw innovation坐标语义对齐，而是用独立随机投影的乘积近似goal similarity与
causal similarity的乘积kernel。

唯一改变是：旧key给goal一条独立等能value path；新key要求每个非零condition坐标都依赖`causal`，goal只能
调制有序过程，不能单独形成地址或Program write。

## 4. Why this should use the teaching video

### 4.1 Cross-initialization task knowledge

`goal`表示视频中终态相对整段的目标关系变化，`causal`表示有序阶段累积过程。两者都从同language下真实图像相对
zero-image的frozen-policy innovation得到；B20 action queries仍与两条teacher episodes错开。故监督不能依靠逐帧
轨迹复刻，condition必须支持同task多种source states/actions。

interaction保留“什么目标变化”与“怎样有序发生”的共同证据：相同目标但因视角、速度或抓取细节不同的视频仍可
共享causal-goal kernel；只有静态目标相似、过程不同的视频不会仅靠goal占据半个key。

### 4.2 Correct order is structurally necessary

两个输出block都含`u_c`。shuffle破坏prefix continuity；reverse改变prefix方向和terminal-relative goal。若理想化
reverse同时近似翻转`u_g,u_c`，旧key整体落到correct的反向直线，而新key变为：

```text
correct: [ u_c,  u_g * u_c]
reverse: [-u_c,  u_g * u_c]
```

两块等能时二者近正交，而不是Gram中的同一个高能方向。retained phase16 cache近似审计中，correct-reverse cosine
由历史强负相关变为mean/median约`-.198/-.191`；这不是raw-frame正式结论，真实reverse/shuffle/wrong仍由一次
完整profile验证。

### 4.3 Language/static bypass is blocked

exact language只作为frozen policy的grounding context。innovation始终是actual image减matched zero image；
zero/no-video innovation使`goal=causal=u_x=psi=0`。没有bias、video-present constant、language-only residual或
Core到Program memory的旁路。即使goal非零，`causal=0`也使完整key为零。

## 5. Why this is not metric cosmetics

CGIK-JC不是对旧256维key做可逆whitening。Hadamard interaction改变单视频kernel邻域，而且在部署时可由一条
video独立计算。它也不做pair-dependent common/secant normalization；后者会把只有约5%能量的video-private
secant强制单位化，在duplicate views处不连续，并重演历史“放大低能、高rank更健康”的错误。

同一cache的预注册selection evidence为：

- selected correct48 rank仍为`48`；condition `250.11 -> 163.88`；
- selected same-task pair cosine mean/median=`.85312/.87463`，未把同task打散；
- 全1200 correct videos same-task cosine mean/median=`.85745/.87427`；
- 50个same-demo train24 panels condition min/median/max=`8.10/9.65/11.33`；
- interaction归一化前norm min/mean/median/max=`.06940/.08479/.08523/.10286`，没有近零块被强行放大；
- selected correct key的全体off-diagonal absolute mean从`.16721`降到`.12198`。

这些数只授权一次真实profile，不选择模型。历史PICK-GC曾把condition降到约`152.61`却只得strict138，因此
condition好看不是成功标准；只有closed-loop single checkpoint能裁决shared能力是否真正共存。

## 6. Inherited mechanisms and output health

CGIK-JC完整保留：

- v6-fast 143的Semantic Core、ordered Procedure、高增益slot compiler和native 38-target rank16 FactorHeads；
- frozen source policy、zero-image-subtracted phase innovation与condition-local FP32 Program memory；
- PVJFC两条video各自独立完整Program cotangent、same B20/RNG、continuous half-weight joint credit；
- matched wrong/shuffled/reversed zero-RHS，不把negative LoRA人为推坏；
- dynamic task queue、world1--6、native BF16/TF32和memory-only checkpoint；
- one-shot部署、完整public LoRA、zero Program/no-video identity。

LoRA健康度继续只作链路诊断。profile必须复用Program→A/B→effective BA→fixed-action门，但不能因rank、norm或
condition漂亮替代strict400。

## 7. Canonical implementation boundary

1. 原位把`PolicyInnovationGoalCausalConditionFeature`替换为causal/interation组合；不新增parallel feature class。
2. 复用同一两个fixed JL blocks、feature width256与policy-innovation owner。
3. 新config、run/checkpoint/profile/eval schema fresh-incompatible；旧PVJFC config保持results-only sealed non-pass。
4. `v6_prior_step.py`、paired graph、B20 cotangent、full96 gather/solver、profile和evaluator算法不改。
5. active runtime仍只有一个condition→Program owner；历史公式由Git、旧config、design和artifact保存。
6. 不新增hash、逐tensor扫描、防御性fallback、重复forward、dtype扩展或新worker系统。

## 8. Fast falsification

### 8.1 CPU contracts

- exact formula、shape、FP32 output、fixed seed和无learned state；
- zero innovation、goal-only或causal-zero输入都产生exact zero key；
- synthetic joint sign flip使correct/reverse从共线变为两块分离；
- 现有view-swap、duplicate-view degeneration、task weight、negative zero RHS、full96 gather与resume测试不变；
- zero Program identity、source freeze、one-shot information wall和完整38-target rank16不变；
- fresh schema拒绝PVJFC memory exact-resume。

完成focused tests后运行完整CPU suite、compileall、CLI config check与`git diff --check`。不为探索性统计新增通用
分析框架。

### 8.2 One discarded full24 macro0

从zero Program fresh运行同一个24 tasks × 2 videos × B20 full96 profile。除feature kind/schema外，seed、videos、
action queries、RNG、view weights、`.01` damping、solver、gates和profile probes全部与PVJFC相同：

- positive rank至少24，full96 rank至少48，regularized condition不超过200；
- total、primary、companion与四suite directional derivative严格小于0；
- 至少12/24 tasks两个view都下降；
- negative/correct motion ratio不超过`.15`，wrong/shuffled/reversed各至少12/16 task-local rows通过`.15`；
- Program delta、两view四suitemotion、LoRA A/B、effective BA和fixed action非零；
- 0 OOM/nonfinite/forbidden reads/outcome rollout；
- total wall不超过`292.325s`。

任一项失败即退役当前interaction key，不扫damping、condition阈值、block weight、projection seed/width、view数、
rank、scale或dtype。只有可复现工程合同错误允许一次不改变数学定义的窄修。

## 9. Formal and real-performance decision

profile全过后才从zero Program fresh训练`0→5`，保存每个macro并立即做macro5 single-checkpoint strict paired
correct400。比较v6-fast143、immutable old134、PICK-GC138、SKNC137、NPCG135、CVEG131及PVJFC无checkpoint
边界，逐task报告retained/gained/lost、breadth、suite和churn。

macro5任一项不满足即终局non-pass，不补controls或小扫：

- correct `<140/400`；
- breadth `<6`；
- 相对old134 lost `>10`或gained不大于lost；
- 少于3个suites不下降，或总net gain主要由一个task换手贡献。

通过才exact-resume到macro10并再次strict400。只有single checkpoint达到至少144且retention保持，才补同checkpoint
same/wrong/shuffled/reversed/no-video；首次严格`>150/400`时立即运行完整controls。最终成功要求correct沿有用
policy direction实质优于所有controls、same-task-other鲁棒，并在同一checkpoint积累多task能力。内部condition、
loss、rank或checkpoint union均不能授权继续。

## 10. Efficiency and launch contract

科学计算量与PVJFC相同，Hadamard只增加可忽略的128维逐元素操作；预计profile wall约52秒、每卡peak约
43.3GB allocated，profile新增存储小于1GiB，formal五个Program checkpoints加日志小于1GiB。

每次GPU launch前同时live检查gpu01/gpu02，选择一个节点，用1--6张真正有益且峰值显存余量充分的健康A40；
不要求6卡、不等待凑卡。低利用率小占用设备在真实余量足够且不明显干扰时可以共享，但本B20 profile的A40峰值
接近满卡，不能仅因利用率低就使用余量不足的卡。不抢占、kill、reset或跨节点拼卡。

正式run来自clean pushed commit的detached frozen worktree，使用`NCCL_P2P_DISABLE=1`、NUMA local-rank映射、
deferred NCCL、native BF16/TF32和dynamic queue。launch前查询`strg01 /data1` user quota并确认fresh output root；
不做内容hash、逐tensor扫描、重复single forward或低位一致性降速。

## 11. Rejected alternatives

- **调ridge或condition门**：不改变row语义，只让旧失败过数值门。
- **全局centering/PCA/whitening**：correct去均值后condition仍约255；可逆变换是度量美容，截断又会压缩support。
- **paired common/secant单位化**：依赖训练pair、放大低能private方向、duplicate view处不连续，部署没有独立
  per-video feature语义。
- **hard same-video equality**：CVEG strict131已经否决`E DeltaM=0`当前组合。
- **learned cotangent metric**：可能有价值，但同时引入condition learner、optimizer和新监督接口；当前固定
  interaction有直接cache证据且可在一次52秒profile中证伪，先隔离更小变量。
- **更多rank/basis/experts/few-shot**：历史已说明更大capacity或多video输入不会自动解决shared credit，且会改变
  本轮唯一变量或canonical one-shot合同。

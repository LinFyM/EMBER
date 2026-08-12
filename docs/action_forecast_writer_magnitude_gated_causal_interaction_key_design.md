# Magnitude-Gated Causal Interaction Joint Credit

## 1. Decision

CGIK-JC的唯一raw-frame full96 profile把PVJFC condition从`597.861`降到`270.188`，并保持rank`48/96`、
24/24双view descent、三类negative各16/16、完整Program→LoRA→BA→action与`50.032s` wall；但仍未过
`regularized_condition<=200`。因此本设计只替换CGIK condition key的第一块：

```text
CGIK-JC:
  u_g    = ZeroL2(P_g goal)
  u_c    = ZeroL2(P_c causal)
  u_even = ZeroL2(u_g * u_c)
  psi    = ZeroL2([u_c, u_even])

MGCI-JC:
  u_g    = ZeroL2(P_g goal)
  u_c    = ZeroL2(P_c causal)
  u_odd  = ZeroL2(abs(u_g) * u_c)
  u_even = ZeroL2(u_g * u_c)
  psi    = ZeroL2([u_odd, u_even])
```

`*`为逐坐标Hadamard product；`odd/even`只描述理想reverse下的符号奇偶，不是phase奇偶池化。其余全部保持
CGIK/PVJFC合同：两条不同、原序、action-hidden训练视频各自生成完整Program cotangent；same B20与policy RNG；
48 correct与48 matched zero-RHS negative以半权进入同一个full96 solve；部署仍只有一条video、一套完整38-target
rank-16 LoRA。

本方法简称 **MGCI-JC**。当前只授权实现、CPU门和一次discarded raw-frame full96 profile。profile通过前不授权
formal、deployment结果或closed-loop评测。

实现状态：fresh-incompatible canonical owner、checkpoint/RNG/inspection三套identity和eval v11 family已经完成，
完整CPU回归`339 passed`。clean`eb1e53b`上的唯一raw full96 profile以condition=`174.813`、rank`48/96`、24/24
双view与四suite descent、negative ratio=`.02088`、三类各16/16、全链路和`49.841s`通过12/12门，无checkpoint。
因此当前按第9节开放fresh formal`0→5`；不得重跑profile或绕过macro5 strict400门。

## 2. Earliest failed interface

CGIK profile的12项门只有condition失败，且相对PVJFC已经同时改善：

| live full96 | PVJFC | CGIK-JC |
| --- | ---: | ---: |
| regularized condition | `597.861` | `270.188` |
| negative/correct Program motion | `.07266` | `.02457` |
| reversed/shuffled/wrong null rows | `14/15/16` | `16/16/16` |
| both-view descent tasks | `24/24` | `24/24` |

retained phase16 cache可以精确重建selected correct48，并可靠近似本次16条reverse rows；它不能精确重建
target-language wrong或完整raw-frame shuffle，因此只作design selection：

| CGIK rows | condition | lambda min | lambda max |
| --- | ---: | ---: | ---: |
| correct48 | `163.882` | `.011456` | `3.506323` |
| correct48 + reverse16 | `213.681` | `.009094` | `4.070007` |

reverse重建与live correct/reverse cosine的MAE只有`.00439`。只加入reverse16就越过200，说明最早剩余失败不是
wrong、solver、Program、cotangent或compiler。

进一步把CGIK两块分开：

| block | correct48 condition | correct48 + reverse16 |
| --- | ---: | ---: |
| independent raw causal `u_c` | `322.87` | `679.39` |
| signed interaction `u_g*u_c` | `106.24` | `201.58` |

reverse rows约98%能量落在correct raw-causal span；第一主模态由Spatial/Object共同占据，不是一条异常video。
因此最早失败接口是：

```text
independent raw causal block
  -> cross-task process-common cluster
  -> reverse rows reinforce the same correct causal span
  -> full96 high-energy shared mode
```

## 3. Single structural variable

沿用CGIK的phase-aligned frozen-policy innovation、goal/causal descriptor、两个fixed JL block、seed`20260810`、
width`128+128`与全部归一化。唯一变化是第一输出块从`u_c`变成`ZeroL2(abs(u_g)*u_c)`。

这不是block weight：两块最终仍各自单位化并等权concat。它不是whitening：变换对每条部署video独立、非线性，
直接改变single-video kernel邻域。它不增加parameter、optimizer、bank、task route、outcome或部署输入。

`abs(u_g)*u_c`与`u_g*u_c`逐坐标绝对值完全相同，因此pre-normalization norm也完全相同；CGIK已经验证该norm
min/mean约`.0694/.0848`，本设计不会新增“把近零第一块强行单位化”的风险。

## 4. Why video and correct order remain necessary

### 4.1 Cross-initialization task knowledge

`goal`是terminal quartile相对整段的任务目标变化，`causal`是有序phase prefix的累积过程。MGCI两块都要求二者
同时非零：第一块用goal magnitude选择哪些causal coordinates属于这个目标，第二块保留signed goal-causal关系。
B20 action queries与两条teacher episodes错开，不能逐帧复制低层轨迹。

### 4.2 Directed order

理想reverse近似`u_g→-u_g, u_c→-u_c`时：

```text
correct: [ u_odd,  u_even]
reverse: [-u_odd,  u_even]
```

两块等能使correct/reverse趋于正交。shuffle破坏prefix continuity，wrong改变目标与过程的共同坐标。该结构没有
引入新的时序阶数；它只保留CGIK已经接通的reverse parity，同时移除独立raw-causal地址。若真实full96仍失败，
就说明剩余mode不是该独立causal block，当前假设直接被否决。

### 4.3 Language/static bypass

exact language仍只作为frozen policy grounding context；innovation是actual image减matched zero image。zero/no-video、
goal-zero或causal-zero都使完整key exact zero。没有bias、video-present constant、language-only residual或静态Core旁路。

## 5. Design-selection evidence and limitation

同一selected macro0 correct48 cache预验：

- condition `163.882→108.812`，rank仍48；
- `lambda_min/max=.016110/2.831079`；
- primary/companion各自condition约`7.42/7.18`；
- selected same-task pair cosine mean/median/min=`.806/.848/.600`；
- 全1200 same-task cosine mean/median约`.813/.835`；
- 50个same-demo train24 panels condition约`6.0--8.4`；
- correct48+reverse16 condition=`130.78`；
- correct48+shuffle16近似=`121.63`，correct+reverse+shuffle近似=`146.95`。

same-task邻域仍高，但比CGIK的`.853/.874`下降；这是需要真实closed-loop观察的nuisance风险。最大科学局限是两块
不是独立证据：`u_even`只是`u_odd`按goal sign逐坐标翻转。它没有新的credit语义，也不能因condition变漂亮就
宣称更理解视频。wrong与完整raw shuffle仍是live未知量。

## 6. Inherited policy-effective mechanisms

MGCI-JC完整保留：

- v6-fast 143的Semantic Core、ordered Procedure、slot compiler与native 38-target rank16 FactorHeads；
- frozen source policy、zero-image policy innovation、condition-local FP32 Program memory；
- PVJFC两条video的独立complete cotangent、same B20/RNG和continuous half-weight joint credit；
- wrong/shuffled/reversed zero RHS，不把negative LoRA人为推坏；
- one-shot部署、zero Program identity、dynamic queue、world1--6、native BF16/TF32和memory-only checkpoint。

本设计唯一目的，是让这些已接通机制首次获得通过full96数值可识别门并进入真实paired400的机会；它不把内部门
当成性能目标。

## 7. Canonical implementation boundary

1. 原位把condition第一块`u_c`改成`ZeroL2(abs(u_g)*u_c)`；公开Python owner/interface不改。
2. 第二块`ZeroL2(u_g*u_c)`、projection、width、Program memory与paired solve完全不改。
3. 新config/run/profile/completion/checkpoint/RNG/inspection/eval schema fresh-incompatible；CGIK config保持sealed non-pass。
4. `v6_prior_step.py`、B20 cotangent、full96 gather/solver、profile/evaluator算法不改。
5. 一个canonical active path；历史公式只由Git、sealed config、design和artifact保存。
6. 不新增fallback、hash、逐tensor扫描、重复forward、dtype扩展或新worker系统。

## 8. Fast falsification

### 8.1 CPU gates

- exact formula、shape、FP32、fixed seed和无learned/persistent key state；
- zero innovation、goal-zero与causal-zero均exact zero；
- ideal joint sign reversal产生一块翻号、一块不翻号；
- 第一/第二块pre-normalization norm exact equal；
- existing swap、duplicate-view、negative zero RHS、full96 gather与resume tests不变；
- zero Program identity、source freeze、one-shot wall、完整38-target rank16不变；
- fresh schemas拒绝CGIK checkpoint/RNG/inspection与evaluation identity。

完成focused tests后运行完整CPU suite、compileall、CLI config check与`git diff --check`。

### 8.2 One discarded full96 profile

从zero Program fresh运行与CGIK完全相同的24 tasks × 2 videos × B20 profile。除第一condition block和fresh identity
外，seed、videos、action queries、RNG、negative schedule、half weights、`.01` damping、solver与12项门不变：

- positive rank至少24，full rank至少48，regularized condition不超过200；
- total、primary、companion、四suite derivative均负，至少12/24 tasks两个view都下降；
- negative/correct motion不超过`.15`，三类各至少12/16；
- Program、A/B、effective BA、fixed action非零；
- 0 OOM/nonfinite/forbidden reads/outcome rollout；
- total wall不超过`292.325s`。

任一失败即退役，不扫damping、门、block weight、projection seed/width、view数、rank、scale或dtype。

## 9. Formal and real performance

profile全过后才从zero Program fresh训练`0→5`，保存每macro并立即做single-checkpoint strict paired correct400。
macro5任一项不满足即终局non-pass：

- correct `<140/400`或breadth `<6`；
- 相对immutable old134 lost `>10`或gained不大于lost；
- 少于3 suites不下降；
- 总net gain主要由一个task贡献。

通过才exact-resume`5→10`。只有single checkpoint至少144且retention保持，才补same/wrong/shuffled/reversed/no-video；
首次严格`>150/400`立即跑完整controls。最终成功必须是correct沿有用policy direction优于controls、same-task鲁棒且
多task能力在同一checkpoint积累；condition、loss、rank和union都不能选模型。

## 10. Efficiency and launch contract

计算图与CGIK相同，只增加一个128维`abs`和Hadamard，预计profile约50秒、A40峰值约43.3GiB、artifact<1GiB。
每次launch前同时live检查gpu01/gpu02，使用同节点1--6张真正有益且有峰值余量的健康A40，不等待凑6卡；低利用
部分占用卡仅在真实余量足够时共享。本B20峰值接近满卡，不能使用只剩约41GiB的卡。不抢占、kill、reset或跨节点。

正式run必须来自clean pushed commit的detached frozen worktree，使用`NCCL_P2P_DISABLE=1`、NUMA local-rank映射、
deferred NCCL、native BF16/TF32和dynamic queue。launch前查`strg01 /data1` quota与fresh root，不做内容hash、
逐tensor扫描或重复single forward。

## 11. Rejected alternatives

- **提高ridge或放condition门**：只改变容忍度，不改变raw causal shared mode。
- **whitening/PCA/centering**：可逆变换只美化度量，截断会重演compression support损伤。
- **hard E或pair common/secant**：CVEG131与低能private放大风险已给出反证。
- **phase-adjacent path-area key**：cache condition更低，但same-task cosine约`.71--.76`，更易重演v4/trace低层路径
  nuisance；只有MGCI被真实profile否决后才可重新立独立authority。
- **更多rank、experts、few-shot或learned metric**：同时改变capacity、部署或监督接口，不能与当前最早失败隔离。

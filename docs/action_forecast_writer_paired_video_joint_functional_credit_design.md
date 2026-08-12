# Paired-Video Joint Functional Credit

## 1. Decision

PVJFC是CVEG终局non-pass后的active successor design。它只替换最早已经被证据定位的
`single-video blind functional cotangent -> hard cross-video/outcome constraints` shared-credit接口；deployment、
frozen v6 Writer、PICK-GC ordered condition feature、FP32 Program memory、完整38-target rank16 compiler、B20
source-action监督与negative zero-RHS均保持不变。

CVEG directional strict只有`131/400`、breadth6；old134→CVEG为`113 retained/18 gained/21 lost`，NPCG135→
CVEG为`114/17/21`。canonical与诊断formal又因一条K2 outcome翻转而走出不同hard-guard轨迹。因此后继不再：

- 用两条binary rollout决定高维exact equality；
- 强制same-task两条video产生完全相同的Program motion；
- 给已经错位的single-video proposal继续叠加point guard、E、threshold或scale。

PVJFC改为：每个train task同时从两条不同、原序、action-hidden正确视频取得各自完整one-shot Program cotangent，
两条视频共享同一个跨episode B20 action-query batch与同一个flow/time/noise RNG；随后用交换不变、task权重不变的
联合正规方程一次写入shared Program。没有rollout outcome分支、success bank、硬E、第二套部署LoRA或video平均。

## 2. First-principles hypothesis

### 2.1 What must transfer across initialization

teacher video的可迁移内容是对象/关系、目标状态、必要阶段与有向顺序，而不是示范轨迹。B20 queries继续与两条
teacher videos跨episode错开，因此任何一条video产生的LoRA都必须服务同task多种source states/actions。两个
视频又共享完全相同的B20与policy RNG，所以两条Program cotangent的差异不再由query、Beta time或flow noise
混杂，主要反映两条video condition经完整Writer和policy后需要的不同修正。

历史Balanced residual发现same-task不同video correction近正交；这正是否决hard `E DeltaM=0`的理由，而不是
把第二条video丢掉的理由。PVJFC允许两条video在shared map中拥有不同的有用motion，同时要求同一个FP32 Program
memory能在一次联合solve中让两者都形成下降方向。

### 2.2 Why order is necessary

对primary和companion各自构造同类型counterfactual：reversed、shuffled或cross-suite wrong。两条正确video是
positive functional rows，各自counterfactual是exact-zero incremental rows。target language始终相同；因此
language-only/static map无法同时满足correct positive credit与same-language shuffled/reversed/wrong zero response。
正确顺序不是被事后推远的LoRA，而是唯一获得positive policy credit的有向过程条件。

### 2.3 Why this is not feature, LoRA or gradient averaging

每条video都独立形成：

```text
ordered RGB video
  -> historical v6 base Program for that video
  -> PICK-GC condition feature
  -> current shared Program residual at that feature
  -> complete 38-target rank16 LoRA
  -> same B20 / same policy RNG functional loss
  -> its own complete Program cotangent
```

两条features、Programs、LoRAs和cotangents都保留为独立rows。没有先平均frames/features、没有挑video、没有平均
生成后的LoRA，也没有把两个Program cotangent压成一个task vector。联合solve只在shared condition→Program map
的正规方程中对称汇合。

## 3. Sealed mathematics

对task `i`的primary/companion view记为`v in {a,b}`：

- correct condition feature `phi_i^v in R^256`；
- matched ordered counterfactual feature `n_i^v in R^256`；
- 在各自完整current Program上、相同B20和相同policy RNG下的cotangent
  `g_i^v in R^(320x256)`。

每个view权重固定`w=1/2`，使两个views合计仍是一个task权重。构造：

```text
X = [sqrt(w) Phi_a; sqrt(w) Phi_b; sqrt(w) N_a; sqrt(w) N_b]
G = [sqrt(w) G_a;   sqrt(w) G_b;   0;           0]
```

relative damping保持`.01`，但绝对damping按未丢失task质量的weighted row energy定义：

```text
lambda = .01 * sum_j w_j ||x_j||^2 / sum_j w_j
DeltaM = - X^T (X X^T + lambda I)^-1 G
M <- M + DeltaM
```

这里positive和negative各有48 rows，合计96；task仍先在两views内以`1/2 + 1/2`汇合，再对24 tasks等权。大
`256 x 81920` RHS与Program write保持FP32，只有`96 x 96`小solve使用FP64；正常BF16/TF32 policy/Writer路径
不变。

两条结构退化是hard contract：

1. **view-swap invariance**：交换任一或全部task的a/b顺序，`DeltaM`不变；
2. **duplicate-view degeneration**：若每个task的两条correct、negative与cotangent都相同，PVJFC严格退化为原
   single-view B20 weighted solve；不会把task权重、step size或regularization翻倍。

不保存历史row、normal equation、video feature或cotangent；checkpoint只保存一个FP32 Program memory及完整
cursor/RNG。该首版检验“当前paired continuous credit”本身，不混入RLS replay、momentum或第二个稳定化变量。

## 4. Information wall and deployment

- canonical deployment仍是exact task language + exactly one action-hidden teacher video；
- Writer rollout前运行一次，生成一套完整38-target rank16 LoRA，policy随后不再看video；
- companion只存在于train24监督，部署数量为0；
- 两条video都不读teacher action、proprio、state、reward、terminal、task ID、filename、pose或hidden normalization；
- source actions只进入两条正确video各自的functional loss，并与两条videos跨episode；
- validation/test action/reward仍为0 reads；
- 不使用task expert bank、nearest route、language-only residual、checkpoint/LoRA/video平均或held outcome。

companion schedule沿用CVEG已封存的deterministic no-replacement pair：primary保持历史schedule，companion是同task
cycle中跳过primary与B20 action episodes后的下一合法demo。wrong时为wrong task同样取两个distinct demos；
reverse/shuffle分别对每条真实输入video完整重排。任一task的a/b交换不得改变write。

## 5. Multi-task coexistence and inherited advantages

PVJFC保留：

- v6-fast 143的Semantic Core、Procedure、高增益compiler与task-complete起点；
- PICK-GC已经闭合的ordered goal-causal policy-innovation feature；
- zero Program时与当前online immutable old134相同的deployment路径；
- B20跨episodesource functional幅度、negative zero RHS与task-complete full24等权；
- Work-Queue的completion-driven long-first task execution；
- single FP32 Program及native BF16/F32 rank16 public LoRA topology。

它删除CVEG/NPCG的success bank、K2 base/candidate rollouts、binary classification、hard E、final affine guard与其
对应checkpoint state。Program→LoRA→effective BA→fixed action仍必须在profile中闭合，但几何指标不选择方法。

能力共存的可证伪机制是：同一solve必须同时对48个独立correct video addresses形成总体下降，且primary、companion
和各suite都不能只靠另一view抵消。它未声称两view足以覆盖held occupancy；只有macro5 strict paired400能裁决。

## 6. Canonical implementation boundary

1. 原位把one-task graph改成两个完整view graphs；companion不再只是feature，而拥有自己的base Program、LoRA与
   functional cotangent；
2. 两view重用同一个prepared B20 batch及同一个keyed policy RNG，顺序串行执行并及时释放graph，不同时保留两套
   policy autograd graph；
3. full24 gather改为48 correct、48 negative、48 cotangent rows及固定`1/2` weights；
4. condition update owner改为weighted joint solve，封存swap/duplicate degeneration；
5. 删除candidate guard、success-key bank和outcome rollout active path；旧结果由Git/config/artifacts保存；
6. config、run/checkpoint/eval schemas fresh-incompatible，不能读取CVEG/NPCG Program或bank checkpoint；
7. deployment model graph不变，复用已封存B32吞吐证据；新formal checkpoint仍走同一online evaluator；
8. 不新增worker pool、fallback、hash、逐tensor扫描、重复single forward或dtype扩展。

## 7. Fast falsification

### 7.1 CPU contracts

- a/b schedule distinct、action-query-disjoint、exact-resume cursor一致；
- same B20 batch与policy seed进入两个view；
- weighted solve满足view-swap invariance与duplicate-view single-view degeneration；
- zero negative cotangent、task总权重1、no-outcome/no-bank checkpoint schema；
- zero Program保持immutable online old134 vertical identity；
- counterfactual构造对两条view都保持target language且完整读取/重排真实frames。

### 7.2 One clean live macro0 mechanism profile

从historical v6-fast、zero Program fresh运行24 tasks x 2 videos x B20：

- 48 correct、48 negative、960 logical action queries，teacher action/reward reads为0；
- positive feature rank至少24，full96 rank至少48，regularized condition不超过200；
- total、primary-only、companion-only weighted functional directional derivative均严格小于0；
- 四suite joint derivative全部小于0，至少12/24 tasks的两个view derivative都小于0；
- negative/correct Program-motion RMS ratio不超过`.15`，wrong/shuffled/reversed各至少12/16 rows通过task-local
  `.15`；
- Program delta finite/nonzero，primary与companion各四suite都有nonzero motion；
- LoRA A/B、effective BA和四suite fixed-action response非零；0 OOM/nonfinite/outcome rollout；
- total wall不超过sealed CVEG `584.649s`的`.5x`，即`292.325s`。

任一机制项失败即淘汰当前PVJFC，不扫view count、weight、damping、scale、rank、seed、dtype、condition threshold
或microbatch。只有可复现工程合同违约允许一次不改变数学定义的窄修。

## 8. Formal and real-performance decision

profile全过后从zero Program fresh训练`0→5`，保存每个macro以保留相邻checkpoint证据，但不根据loss、rank或
small panel挑checkpoint。macro5立即运行single-checkpoint strict paired correct400，并逐task比较CVEG131、
NPCG135、SKNC137、PICK-GC138、online old134与v6-fast143。

macro5出现任一情况即终局non-pass，不补controls或小扫：

- correct `<140/400`；
- breadth `<6`；
- 相对old134 lost `>10`或gained不超过lost；
- 少于3个suites不下降，或总net gain主要由一个task换手贡献。

若macro5 `>=140`且retention门通过，才exact-resume到macro10并再次strict400；只有单checkpoint达到至少144且
保持低换手，才运行same/wrong/shuffled/reversed/no-video controls。最终成功仍要求correct严格`>150/400`，
correct沿有用policy direction优于所有controls并具有same-task-other鲁棒性。PVJFC内部common descent不能替代
真实closed-loop结果。

## 9. Efficiency and launch boundary

profile预计只增加第二条video的base/condition与B20 functional path，同时删除每macro 96条base/candidate
rollouts，wall应显著低于CVEG。每次GPU launch前同时live检查gpu01/gpu02，选择一个节点，使用1--6张真正能提高
吞吐且有峰值显存余量的健康A40；不等待凑6卡。已有低利用率进程或少量显存占用不自动排除该卡，只要预计峰值
余量足够且不会明显干扰；不抢占、kill、reset或跨节点拼卡。

正式run必须来自clean pushed commit的detached frozen worktree，遵守`NCCL_P2P_DISABLE=1`、NUMA local-rank
映射、deferred NCCL与动态work queue。正常BF16/TF32、batch和reduction低位差异被接受，不固定batch1、不重复
forward、不扩dtype、不加防御性扫描或内容hash。预计profile小于1GiB、formal五个约84MiB Program checkpoints
加日志小于1GiB、strict400约1.1GiB；launch前仍按实时quota确认。

## 10. Rejected alternatives

- **保留hard E再只删binary guard**：CVEG131已显示hard E组合没有absolute价值；
- **把两个cotangent直接平均**：会丢失历史已观察到的same-task正交有用修正；
- **平均features或LoRAs**：改变one-shot条件几何并掩盖单video鲁棒性；
- **K4/few-shot deployment**：这是另一主变量，历史best108且不回答当前credit接口；
- **RLS/history replay**：历史140/lost15已经否定offline row保留足以保护held occupancy，首版不混入；
- **reward/outcome guard**：binary K2不稳定，full replay/landmark图又超A40显存或wall；
- **task experts或bank routing**：不提供held support、same-video差异或时序因果；
- **rank/scale/dtype sweep**：连续强结果已经排除identity、rank与Program/LoRA传递是最早断点。

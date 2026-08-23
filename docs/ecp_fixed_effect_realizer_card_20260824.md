# ECP Phase 2B/2C：fixed effect-code coordinate and deployment realizer

状态：2026-08-24 **预注册合同不变；effect-particle、fit-only coordinate与formal realizer训练已完成，held materialization与closed-loop Gate待执行。**

## 唯一科学问题

在Phase 2A已证明15/15 known-success balanced-SVD rank4 paths都沿固定policy-effect坐标严格单调改善后，
本轮只回答：一个held-free、target-local的小型amortized realizer，能否把不含task ID或LoRA factors的
结构化privileged effect code转换为在source-unseen held tasks上真实有效的mobile-rank4 residual。

这不是`q_pi`、`q_V`、Stage 0或video inference训练，不再建task-local solver，不用held occupancy作为realizer
inference输入。held的privileged evidence只允许经冻结坐标变换为代码，然后一次realizer forward生成唯一
LoRA。未来deployment由`q_pi/q_V + Program-to-code`生成同构code，`D_eff`本身不读取执行state。

## 固定资产与folds

- task/member authority固定为95 tasks、118 successful members的
  `runs/analysis/pi05_ecp_stage1_mdco_task_evidence_20260823/manifest.json`；
- 71个审计后LIBERO-90 non-held tasks在所有fold均作fit；target train24按既有结果无关的ordered-position
  modulo-5规则轮换19/5；validation8/Test8 action、reward与expert读取为0；
- 首先执行fold0：held global IDs固定为`0/9/18/25/36`。只有fold0 final checkpoint通过强门后，
  才用完全不变的架构、损失、step和checkpoint规则fresh训练fold1作第二个方向复现；
- 训练按task等权，同task多个members在task内等权，不把member数量冒充独立mapping。

## Phase 2B：固定输入与输出坐标

### 输出坐标

每个successful member的task expert相对stable carrier形成exact effective correction，每个38-target correction独立做
compact SVD并使用deterministic sign gauge，只保留前4个singular directions：

```text
W_j = W_carrier,j(rank12) + B_j(rank4) @ A_j(rank4)
```

carrier12不变，不生成full rank16，不为rank slot赋予skill/event语义。训练target是固定canonical
`A_j/B_j`，同时用低秩inner-product精确计算effective-update Frobenius loss，不依赖raw-factor cosine选模。

### privileged effect particles

每个member只在自己已证明successful的on-policy trajectory上取8个已登记有序phase anchors。对每个anchor使用
记录的PI0.5 suffix noise `z`与`-z`各做一次原生10-step denoising，保留：

```text
owner_delta[trajectory, probe_sign, event=8, owner=38, horizon_mode=4, hidden=128]
```

`owner_delta = successful_member - stable_carrier`，两者共用完全相同的language/image prefix、noise与Action Expert
forward。正负probe在落盘前不平均；两条trajectory也不提前平均。每个training example保持一个global member
identity，不做event-wise member拼接。本轮只使用有真实success保证的on-policy states；candidate/recovery不冒充
valid target，后继`q_pi`前另补continuation/recovery validity。

### 冻结effect code

对每个owner，将`4 x 128`的horizon-mode response展平为512维；只用当前fold的fit tasks、按task/member/
event/particle等权拟合一个owner-local PCA-whitening `512 -> 128`，然后永久冻结。held只做transform，不拟合均值、
basis、scale或free code。得到：

```text
C_effect[particle, event=8, owner=38, 128]
```

PCA不读LoRA target、task ID或reward，只固定native effect表示的数值坐标。落盘资产保留raw particles、particle mask、
member reliability、trajectory/probe provenance和信息墙metadata。

## Phase 2C：target-local amortized realizer

`D_eff`只读`C_effect`、particle mask和reliability，不读task ordinal/global ID、language string、filename、raw adapter或
occupancy tensor。网络固定为：

1. shared `128 -> 128` token MLP，加固定8-event、38-owner/family/layer embeddings；
2. event attention先在每个particle内保序聚合，DeepSets attention再在particles间做置换不变聚合，不在输入处取均值；
3. 每个owner得到256维state，只允许一个共享global owner summary作有限跨层通信；
4. 38个target-local bottleneck-32 heads分别输出该target的canonical rank4 `A/B`；output basis属于realizer并在
   `q_pi`前训练、通过Gate后整体冻结。

损失按task等权，为每target归一化canonical factor MSE、gauge-invariant exact effective-update loss和小权重
null-code-to-zero-residual约束。不用held loss、held checkpoint selection、LoRA cosine gate或simulator reward。Action Meta关闭。

## 执行节点与closed-loop Gate

1. 先完成CPU asset/shape audit与一个GPU capture profile；profile只固定microbatch与显存，不选方法；
2. 用单节点最多6张有效GPU一次捕获118 members，按member shard并行，不重跑task experts或大规模训练；
3. fold0 formal realizer固定1000 optimizer steps，保留step800和1000两个相邻节点；先用不超过50 rows的
   invalidity screen排除serialization、all-zero或明显低于carrier，不用screen选checkpoint；
4. screen有基本信号后，step800和1000均运行预登记held5 strict paired250。每task统一使用`latest`
   successful member的privileged code，不按task挑member、video、code或checkpoint；只生成一套完整LoRA。

fold0 final step1000通过必须同时满足：

- total `>=69/250`，即相对carrier `43`恢复至少40%的direct-latest `108` gap；
- breadth `5/5`，Goal和Long均非零；
- 保留carrier至少`33/43` successes；
- step800同时达到breadth `5/5`、Goal/Long非零、carrier retention `>=33/43`，且total不低于final 5个
  successes以上；
- episode key、environment seed、policy-noise common prefix与reference rows严格配对。

fold0强门通过后才fresh复现fold1；同样要求held breadth全覆盖、Goal/Long类别非零、明显高于fold-specific
carrier且保留率达门。两fold通过后冻结`D_eff`，才能进入fresh Stage 0和后续`q_pi`。

## 失败解释边界

- known-success exact balanced-SVD projection自身仍能闭环，但fit-only effect code在held上无法重建residual，则失败在
  cross-task effect-code-to-update mapping，不回去做alpha/trust/solver sweep；
- 若只是`512 -> 128`坐标严重丢失known correction，允许依专家合同只试一次fixed two-sided sketch +
  deterministic reconstruction；
- fold0闭环不高于carrier、breadth `<=3/5`或Goal/Long均为0，不靠更长训练、width/LR/seed/head小扫救援；
- 本轮通过也只证明realizer接口成立，不证明language/video能生成正确code。

## Phase 2B capture milestone

clean pushed detached `565c055ee7187546c017f253646d70c25a330b7e`在gpu01 physical`1,2,3,4,5,7`以6个独立
shards完成唯一formal capture，physical0未使用。118/118 members、95/95 tasks与188条既有successful trajectories
全部覆盖，形成376个保留`trajectory x probe-sign`轴的particles；每particle形状为`[8,38,4,128]`。

六个shards均返回0，capture本体耗时为`83.79--87.40s/shard`，每路峰值allocated为18.72 GB；118个
member tensors合计195,744,144 bytes。一条两轨迹profile上正负probe response RMS差为`.05092`，所以新轴
不是重复副本。本次只采集successful member自身的on-policy states，candidate/recovery没有被冒充valid target；
validation/Test读取、held optimizer steps和task-ID model input均为0。formal authority：
`runs/analysis/ecp_fixed_effect_particles_565c055_gpu01p123457_20260824/manifest.json`。

## Phase 2B/2C coordinate and training milestone

fold0的fit-only coordinate由clean pushed detached `e05ffca`生成：90个fit tasks/108 members用于拟合，5个held
tasks/10 members只经冻结transform；owner-local `512 -> 128`解释方差比例最低`.90695`、平均`.94106`，118份code
均保留particle/event/owner轴且finite。authority：
`runs/analysis/ecp_fixed_effect_code_fold0_e05ffca_gpu01p1_20260824/manifest.json`。

同一authority下的formal realizer只加载fit members，1000步耗时`136.99s`、峰值allocated 1.90 GB。step800的
factor/effective/total loss为`.23058/.08608/.31668`，step1000为`.19417/.07186/.26605`；两节点均已保存完整
model/optimizer checkpoint。该下降只证明fit mapping可优化，不替代held closed-loop。training authority：
`runs/outputs/pi05_ecp_fixed_effect_realizer_fold0_e05ffca_gpu01p1_20260824/`。

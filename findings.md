# EMBER findings

只记录跨session仍影响决策的结论。专家原文见`docs/expert_review_20260824_native_factor.md`，精确分数、提交和历史脉络见
`docs/research_history.md`；当前唯一架构合同见`docs/event_conditioned_policy_compiler_design.md`。本文件是结论索引，不替代原文。

## 科学结论

### 1. 输出形式可行，amortized Writer仍未解决

validation8 task-local rank16 oracle为250/400，四suite均有收益；source只有48/400。因此“冻结PI0.5、只给Action Expert安装唯一
完整LoRA”不是根本错误，核心瓶颈是如何由source-unseen task的language+video生成正确更新。

内部hidden、LoRA cosine/reconstruction、retrieval或低training loss均不能替代closed-loop证据。

### 2. Action Expert有可利用的时序结构，但旧owner并非完整target-native对应

成功task experts的跨层、跨horizon response能形成task geometry；Stage 0 v3的owner/layer/horizon observer通过基本非退化门。
固定`t_flow=1`probe的50个noise tokens按未来horizon排列，其hidden是当前language/image条件下的time-indexed policy response，不是
teacher action或已经预测好的动作。

当前代码的q/v owners主要来自同层input state与residual，再用family embedding/gate区分；它没有捕获真实`q_proj/v_proj`输出
空间。原生target input/output hooks是新架构第一项必需实现，不能把现有`Z_owner`误称为LoRA factor bank。

### 3. 视频因果性尚未建立

多个历史Writer的full-video接近language-only、video-only或first+final，Goal/Long为0。不能声称EMBER已经理解视频过程。最终
correct必须稳定优于language/no-video/static/endpoints/wrong，same-task其它视频保持高retention；shuffled/reversed只在冻结
checkpoint上测试时序特异性，不进入训练或选模。

### 4. 自然task数量是共享映射的识别边界

train24中的language、scene、video和task identity高度耦合，可用审计后的non-held LIBERO-90扩展observer/prior/preservation，但71个
任务已被source见过，不能冒充71个source-unseen adaptation mappings，也不能在task weight上淹没train24。开发macro固定由19个
target-fit与轮换19个meta-fit各占50%。

owner明确不制作人工process数据。若free-code容量强而shared compiler低于carrier或breadth不超过2，应诚实判断现有source-unseen
mappings不足，不能靠joint training或更多同task episodes掩盖。

### 5. Policy effects适合做critic，不适合做部署中间code

15/15 known-success paths在owner/flow/action effect objective上严格单调改善，说明effect space能处理LoRA gauge、successful policies
参数不相似、q-family能量支配和factor loss与policy function错配。

但balanced-SVD realizer只有33/37且低于carrier43；centered two-sided fit span即使aggregate update cosine为.877--.960，仍只有
80/250、breadth3/5、Goal/Long0。将held innovation压回fit-task固定坐标会丢失低能量但闭环关键的方向。

因此effect evidence只作nonparametric set-valued functional critic；它不再生成Program、不进入deployment，也不形成
`Program -> effect code -> fixed inverse -> LoRA`。

### 6. canonical删除神经`q_pi`

没有真实Program标签；同时训练policy encoder、video encoder和realizer仍允许latent任意旋转。现有95-task/118-member evidence更适合
直接监督generated policy function，而不是再训练一个未经验证的privileged Program teacher。

canonical只保留video Program encoder`q_V`。一个logical trajectory只能由一个global successful member解释，不能按event混合
members；只有short-continuation verified member-state pairs可作训练target。

### 7. Native-factor compiler直接针对最早失效接口

新主线用同一视频在冻结PI0.5各目标层产生的真实native inputs`X_j`和outputs/differences`Y_j`作为task-specific参数基底。Program
只学习对video/frame/probe/horizon/feature-type的signed selection和target scale，再形成rank4 outer products。

这既不从128维直接吐出2048维参数，也不要求held方向存在于fit-task PCA/span中。它是否具有足够容量尚未被实验验证，当前唯一
合理下一步是fold0 held5 task-local free-code strict250，而不是先训练fresh Program或shared compiler。

### 8. rank12 carrier + mobile rank4是当前有证据的首版选择，不是封死结论

shared carrier为43/250；mobile-rank4解析投影在三个member arms为110/120/76，且均5/5 task非零。当前失败是shared mapping/solver，
不是rank4容量。因此首版canonical用frozen rank12 carrier + native-factor mobile rank4，严格拼成一套rank16 LoRA。专家没有把
12+4说成全局最优或不可改变；它只是现有证据下统计难度更低的起点。

这不恢复fixed-A或raw-factor短solver。只有native bank可表达、rank4 free-code已收敛、response分析证明rank ceiling且一次同构full-rank16
oracle显著通过，才重开task full-rank16并按结果调整carrier/task rank；总输出仍是唯一rank16 adapter。

### 9. Program结构已经明确

唯一schema为`P_lang[38,128]`、`P_scene[38,128]`、`P_process[8,38,128]`、`rho[8]`、`tau[8,2]`、
`sigma[8,38,128]`。38个owner固定对应18 q、18 v、action-in/out；`E=8`是最大容量，presence和frame-to-slot assignment动态学习。

每条视频独立保序编码，K-set只在soft monotonic event alignment后聚合。language与scene必须owner-specific，不能退回全局均值或
first/final/difference summary。

### 10. staged Gate是因果诊断，最终必须联合训练

当前执行顺序固定为：authority -> native-factor free-code capacity -> Natural Program -> frozen-Program shared compiler -> all-Writer
joint training -> conditional structured outer credit -> fresh final。每门失败只定位对应接口；不能跳过free-code，用joint training掩盖
参数基底无效，也不能在没有视频闭环增量时启动outer credit。

只有Natural Program、capacity、shared compiler、两fold joint、verified natural on-policy evidence、outer、fresh validation和完整
controls都完成后仍系统失败，才足以判定现有数据/zero-interaction static-LoRA合同存在根本问题。

### 11. Action Meta是后期matched control

当前结果中性，canonical默认关闭。base Writer有明确闭环增量后做一次matched attempt，Stage 0/compiler冻结；只有明确净收益且
无breadth/retention损害才启用，否则保持关闭。

## 已关闭路线

- 旧action-memory、LOOM、CVADR、LMMPC/LPCP及其gradient/credit小变体；
- ECP Stage 1 v1--v24、MDCO和deterministic privileged codes；
- neural `q_pi`、fixed effect-code/balanced-SVD realizer和centered two-sided fit span；
- PECS、fixed-A、raw mobile-rank4短solver、matrix-free solver和full-width factor hyperdecoder；
- 人工opposite-order tasks、primitive/recovery expert acquisition与distillation；
- 把GOMQ重跑或归入ECP阶段；
- Action Meta默认路径和open-loop geometry gate。

这些历史路线只作证据与启发，不恢复活动代码或并行fallback。

## 工程结论与复用面

- 继续复用source/corpus/SFT、rank16 LoRA materialization、task experts、Stage 0 v3、transition/event modules、policy effects、functional
  flow loss、reward/occupancy和strict dynamic evaluator。
- 下一实现缺口是38-target native input/output hooks、chunked online bank accumulator、signed rank4 compiler、task-local free-code optimizer
  和strict250 wiring；当前仓库尚无这些模块。
- 旧Writer/realizer/ECP Stage 1已从活动树删除；后续只允许一个canonical Native-Factor implementation surface。
- formal checkpoints/raw rows保留在ignored `runs/`；精确旧代码用Git恢复。人工process datasets及约12GB产物已删除且当前路线不需要。
- 不新增checksum sidecar、重复证据JSON或一实验一文档；跨轮结论只更新本文件、`progress.md`和`research_history.md`。

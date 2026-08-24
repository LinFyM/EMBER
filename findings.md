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

多个历史Writer的full-video接近language-only、video-only或first+final，Goal/Long为0。不能声称EMBER已经理解视频过程。唯一
正式性能目标线是validation8 strict paired correct严格`>145/400`；同时必须稳定优于language/no-video/static/endpoints/wrong，
same-task其它视频保持高retention，并满足稳定性、breadth、四suite非零和Goal/Long贡献。shuffled/reversed只在
最终selected checkpoint已选定并冻结后测试时序特异性，不进入训练、loss、checkpoint选择、G1--G5 Gate或
架构修正依据。

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
在最终shared compiler中学习content-derived signed selection与target scale：输入`X`候选只索引video/frame/probe/horizon，输出`Y`
候选才额外索引abs/adj/init/goal type，再形成rank4 outer products。G1则允许直接优化task-local selection logits/weights，只作为
native-bank容量upper bound；共享Program-query到candidate-key的映射由G3单独学习和验证。

这既不从128维直接吐出2048维参数，也不要求held方向存在于fit-task PCA/span中。它是否具有足够容量尚未被实验验证，当前唯一
合理下一步是fold0 held5 task-local free-code strict250，而不是先训练fresh Program或shared compiler。

### 8. rank12 carrier + mobile rank4是当前有证据的首版选择，不是封死结论

shared carrier为43/250；mobile-rank4解析投影在三个member arms为110/120/76，且均5/5 task非零。当前失败是shared mapping/solver，
不是rank4容量。因此首版canonical用frozen rank12 carrier + native-factor mobile rank4，严格拼成一套rank16 LoRA。专家没有把
12+4说成全局最优或不可改变；它只是现有证据下统计难度更低的起点。

这不恢复fixed-A或raw-factor短solver。只有native bank可表达、rank4 free-code已收敛、response分析证明rank ceiling且同构full-rank16
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

当前结果中性，canonical默认关闭。base Writer有明确闭环增量后做matched controls，Stage 0/compiler冻结；只有明确净收益且
无breadth/retention损害才启用，否则保持关闭。

### 12. Gate不等于时间或修正次数上限

专家原文给出的工期和“只允许一次/最多两轮”等是其当时的效率建议，owner后续明确不采用为硬约束。当前路线不设阶段工期、
修正次数、结构版本或训练轮数上限；只要求每次修正有新的机制证据并重新通过同一Gate。无信息的超参小扫不算有效修正，充分证据
持续否定接口时才停止。整体实现与关键Gate在保质前提下尽可能快推进，顺利时力争数天内形成完整架构。

### 13. Final fresh数据顺序待Final前裁决

`docs/current_owner_requirements.md`记录了方法选定后的32-task fresh refit，active design当前记录的则是71 meta+train24
fresh development recipe。两者的精确顺序、validation8是否并入32-task refit以及如何保持Test8 sealed尚未裁决。该问题
延迟到Final前由owner确认，不阻塞G1--G5，也不得在此前为任一种解释启动数据合并或训练。

### 14. G1首轮最早失效接口是scalar native-Y输出空间

首轮free-code strict250为`88/250`，逐task`33/18/37/0/0`，Gate non-pass。该结果不能用loss解释为通过，但也不是
Native-Factor根本失败：Object/Spatial已有强闭环信号。read-only解析证明，对冻结linear target有`Y=W X+b`，而positive/negative
两个softmax各自质量为1；无bias的q/v outputs位于`column_space(W)`。action-in带bias，且abs与difference type可跨类型相减，
所以其精确结构上限是`span(column_space(W), bias)`而不是此前简写的纯列空间。因此18个q target的scalar pooling至多覆盖
`1024/2048`输出维，action-in至多覆盖`33/1024`；15个known-success mobile-rank4 reference整体仅保留
约55--56% update energy。

闭环response诊断进一步把同一independent mobile member从`120/250`、Goal/Long=`11/8`投影为`109/250`、Goal/Long=`0/0`，
而三个Spatial/Object task仍为`34/30/45`。这说明被scalar q measure排除的方向对process-sensitive task是必要的。当前只改变这一
最早接口：候选索引仍为`(k,t,p,h,u)`，真实q value按模型原生八个query heads恢复为`[8,256]`，各head独立做signed measure后
拼回2048维；不增加fake type、task route或非native value。action-in仍有独立结构上限，但当前不同时改第二个主要变量。

### 15. q-head复评把最早失效接口推进到free-logit优化

q-head修正后的formal strict250为`84/250`，逐task`28/21/35/0/0`，比scalar首轮`88/250`更低，Gate仍non-pass。其step500
generated residual与三个known-success references的整体effective-update cosine仅约`0.06`，Goal task对latest/independent的
sensitivity-normalized update loss仍为`1.18/1.17`；因此“增加q输出自由度”没有被随机近均匀、千万级dense softmax logits的优化
实际利用。

随后对真实K=1视频bank做稳定中心子空间投影：以action-in已知结构秩校准的relative singular threshold `1e-3`，将latest mobile
rank4的每个input factor和q-head-grouped output factor投影后再按冻结`s_ref`截断。该唯一rank12+4 rank16诊断在paired strict250达到
`94/250`，逐task`24/24/44/1/1`，relative recovery、breadth5/5、四task高于carrier以及Goal/Long非零均成立；carrier retention只有
`22/43`，所以它不是G1 Gate pass。它仍直接证明：稳定native bank内存在具有process-sensitive闭环能力的signed-pooling方向，当前最早
问题是free logits从随机稠密softmax无法到达这些方向，而不是bank本身完全不可达。

将known-success latest member的稳定投影系数分解为positive/negative simplex并写入实际free logits后，精确step0 strict250达到
`100/250`、逐task`24/28/45/3/0`，relative recovery`0.851`；但breadth4/5、Long 0、仅3/5 task高于carrier且retention仍为
`22/43`。step0 residual与解析projection cosine为`0.952--0.964`，第一次Adam更新后即降为`0.039--0.070`；五task 500-step
formal的最终effective-update loss也全部差于step0。故解析点必须以step0保留，不能用step1冒充，也不能用被扰动路径的内部loss代替闭环。

paired evidence又把失败精确定位到set-valued reference选择：task90 carrier为38/50，强于三个mobile members的`27/26/17`，因此
该task的最强合法free-code解是zero residual；task91--94最强member依次为independent/latest/independent/independent，成功数
`32/40/13/5`，而latest-only正好在Long只有3且step0闭环归零。当前修正用同一fixed50 success-count规则在四个verified members中
选择，tie优先carrier；不改bank、rank、pooling或loss。它仍是privileged task-local capacity solve，不是G3共享Program-to-attention
映射。

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
- G1 scalar、q-head与latest-only exact step0分别为`88/250`、`84/250`、`100/250`；最后者恢复Goal但Long为0、retention仅
  `22/43`，仍不能冒充Gate pass。set-valued reference修正已通过真实profile，尚待clean pushed formal复评。
- G1 free logits是held-task capacity upper bound；最终shared Program query到content key的attention仍只属于G3，不得从G1代码或结果
  推断deployment Writer已经成立。
- 旧Writer/realizer/ECP Stage 1已从活动树删除；后续只允许一个canonical Native-Factor implementation surface。
- formal checkpoints/raw rows保留在ignored `runs/`；精确旧代码用Git恢复。人工process路线与约11.6GB可重建主要产物已
  删除，recovery Gate A残留作为历史formal evidence保留，不恢复为当前数据或训练路线。
- 不新增checksum sidecar、重复证据JSON或一实验一文档；跨轮结论只更新本文件、`progress.md`和`research_history.md`。

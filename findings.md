# EMBER findings

只记录跨session仍影响决策的结论。四次专家原文见`docs/expert_review_20260824_native_factor.md`、
`docs/expert_review_20260826_bank_conditioned_native_factor.md`、`docs/expert_review_20260828_g3_functional_sketch.md`与
`docs/expert_review_20260829_joint_program_primal.md`，精确分数、提交和历史脉络见
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

owner进一步提醒：G1--G3的冻结与分段训练只服务于逐接口可证伪，不应被机械复制为最终模型的训练范式。组件Gate通过后，G4/Final
默认应从已验证机制出发直接联合优化完整Writer，并优先采用能支撑闭环因果问题的最小loss集合；只有新证据表明联合训练不稳定或某个
接口需要单独预热时，才保留有明确退出条件的staged warmup。最终取舍继续以single-checkpoint closed loop为准，而不是以loss数量或
分段形式本身为目标。

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
拼回2048维；不增加fake type、task route或非native value。action-in仍有独立结构上限，但该轮没有同时改第二个主要变量。

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

paired evidence把下一接口定位到set-valued reference选择：task90 carrier为38/50，强于三个mobile members的`27/26/17`；task91--94
最强member依次为independent/latest/independent/independent，成功数`32/40/13/5`。该规则的formal strict250为`111/250`、逐task
`35/29/45/2/0`，relative recovery`1.015`、retention`34/43`，但breadth4/5、Long 0且仅3/5 task高于carrier，仍non-pass。

task94的真实初始化报告揭示了更早的数值接口：`1e-3` singular threshold允许scatter inverse condition number约`1e6`，FP32最小
input/output direction cosine只有`0.978/0.883`，没有实际实现解析span点。将仅在初始化时运行的小型eigenspace/inverse-scatter
solve改为FP64后，同一真实forward/materialization的两侧minimum cosine均为`>=0.99999988`，38 hooks、Action Meta 0和唯一rank16
不变。这是由闭环Long失败和方向误差共同支持的数值机制修正，不是seed/LR/threshold扫；仍不证明G3共享映射。

### 16. FP64复评把最早接口推进到action-in whole-vector输出上限

FP64 clean formal把解析点完整实现后，strict250达到`116/250`，逐task`35/34/44/3/0`，relative recovery`1.090`、carrier
retention`35/43`；总分、Goal和retention均通过，但breadth4/5、Long0且仅3/5 task高于carrier，故G1仍non-pass。task94初始化
两侧minimum direction cosine已为`>=0.99999988`，所以Long0不再能归因于FP32 solve。

剩余四个output family中，v和action-out的base Linear output row space可覆盖完整输出，q已按真实八个query heads分组；只有
action-in把`32 -> 1024`线性层的完整Y向量共享一个scalar signed measure，必然受限于`span(column_space(W),bias)`、至多
`33/1024`。paired response只把task94完整rank16中的action-in target恢复为known-success independent mobile，其它37 targets保持
当前native candidate不变，Long由`0/50`变为`1/50`；完整counterfactual为`118/250`、逐task`35/35/44/3/1`、breadth5/5、
4/5高于carrier、retention`35/43`，即数值上满足全部G1门。它仍不是G1 candidate，因为action-in来自privileged reference；其作用
是证明action-in被排除方向本身具有闭环因果作用，而不是根据内部cosine猜测。

当前修正不改变候选索引、不复制X、不增加fake type：每个action-in Y candidate仍只出现一次，只把真实1024D Y按native input
width切成32个连续32D blocks，各block独立做positive/negative softmax后再拼回1024D。32组是由`1024/32`线性shape推出、解除
已证明上限所需的最小full-width partition，不是group-count小扫；G1 logits仍是task-local free code，G3以后必须以共享Program query
和content keys生成这些group measures。

### 17. action-in native-block修正使G1 capacity Gate正式通过

clean pushed `main@31f0053`的task-local formal bank使用真实38-target X/Y、四类output banks、signed positive/negative pooling与
唯一rank12+4 rank16 adapter。paired strict250达到`114/250`，逐task`35/31/45/2/1`；relative recovery`71/67=1.060`、
breadth5/5、Goal2、Long1、4/5 task高于carrier、carrier retention`35/43`，全部G1 Gate checks通过。54 shards、250 rows和18 workers
完整，Action Meta module/parameter为0，未用shuffled/reversed。

该闭环结果与此前task94 action-in-only response形成一致因果链：解除已证明的whole-vector output ceiling后，真实native pooling本身
恢复Long并满足全部容量门。因此G1问题已经回答为“存在”；它仍不证明task-unseen的共享Program query-key selection能够学习，下一最早
接口是G2 Natural Program，随后才是G3 shared attention。

### 18. G2必须在native forward层面逐video独立，不能只在padding后声称集合不变

G2首轮真实held检查中，K1 aggregation已是bitwise identity，但把K条video的frames先扁平、再按全局frame chunk送入native policy时，
同一K4视频集合仅改变video顺序就使Program最大绝对差异达到`0.132`。这不是集合mean公式的问题：不同video长度改变了各帧所在的
native microbatch/chunk，实际输出没有满足部署合同要求的逐video独立性。

把每条video的positive/negative probe native forward完全独立，只在每video event形成并经canonical alignment后用FP32
`beta_k=1/K`聚合，同一真实检查的最大差异降为`2.38e-7`，K1仍完全相等。后续G2/G3的集合不变性必须覆盖完整native forward，
不能只对预先构造的local tensors测试mean交换律；G2仍不学习video reliability。

G2派生标签的真实时间合同同时得到验证：HDF5 `obs[i]`对应`states[i+1]`（terminal post-action state缺失），因此末帧成功predicate
可由successful-demo合同置真，但contact必须mask；稀疏query的rising target要对相邻query区间取any。LIBERO-90 scene4的四个任务还存在
`salad_dressing_1 -> new_salad_dressing_1`模型identifier历史改名，只能在内存XML恢复时按当前BDD model显式对齐，不能改写原始HDF5。

formal前复核还定位到两个数据权重接口：辅助robustness/contrast若按rank-local执行顺序抽样，会让task接受不同数量或不同规模的loss，
即使最终梯度再按全局task数归一化也不等于task-equal；因此G2现对每task计算一次robustness，并为每task选固定8个、两种fit role各半、
与rank/world-size无关的language-content negatives。跨episode action与progress/rising/contact/predicate必须共享同一action-episode query
index；不能先经video长度取整再二次映射。label v2同时明确`rising[0]`比较`states[0] -> states[1]`；全量4750 demos中该边界恰无正例，
所以数值总量仍为7344，但schema必须显式区分，防止未来数据静默改变语义。

### 19. G2首轮non-pass是decoder静态旁路，不是native动态捕获失败

clean pushed `main@141a110`的G2 macro10 held20 Gate中，same-task separation、probe margin、event non-collapse、K1 identity与K4集合
置换全部通过，但full相对endpoints的action/progress loss只改善`0.0226%`，所以正确结论是G2 non-pass，不能进入G3。

同一checkpoint的无梯度消融提供了最早接口证据：full与endpoints的`P_process/rho/tau`差异相对same-task不同video分别约为
`2.20x/13.77x/60.00x`，native process确实保留了中间帧信息；但decoder action/progress输出几乎不随query time变化，action时序
标准差为`0.00060`，而training target为`0.33789`。清零`P_process`后静态路径combined loss由`0.39574`改善到`0.39088`，说明
`P_lang/P_scene`被重复加到每个event以及直接进入process fusion，使模型能够用task/endpoint code拟合跨episode priors并忽略动态。

因此当前修正只切断这条已证实的静态旁路：`P_process`由native process与native uncertainty形成，时序heads只读
`P_process/rho/tau`；`P_lang/P_scene`仍按固定schema输出，并只供独立scene relation head读取。它不改变Stage 0、K aggregation、slot/width、
seed、训练数据或Gate，也不使用shuffled/reversed。若fresh复评仍失败，下一定位应检查event-token内部时序分离和query-to-event读出，
不能恢复静态旁路或用无信息超参扫掩盖。

### 20. 静态旁路移除后，最早接口是G2梯度侵蚀已验证的Stage 0 event grounding

clean pushed `main@30b98ef`的static-free fresh macro10仍未通过held20 Gate：full相对endpoints改善`-0.0570%`，one-event
fraction `0.30`，probe margin `0.65`；same-task、K1、K4和active-event median仍通过。无梯度readout消融显示tau产生的event weights
已有明显时变，但owner pooling近乎均匀、event tokens彼此接近，最终action预测的temporal std只有`0.00093`，而target为`0.32725`；
hard-nearest、uniform event measure与mean-repeated process都不能显著改变loss。因此不能把失败归因于某个query核或再调tau。

target-held5的前后对照把最早接口进一步前移：初始Stage 0 v3的event/owner relative RMS为`0.06069/0.36992`，同一observer经G2
macro10训练后raw值降至`0.02601/0.22824`，fusion后owner仅`0.14837`。也就是说，在新的Program readout尚未学会使用动态前，联合梯度
先抹平了已有的event/owner结构。首个有证据修正是保留Stage 0 v3为frozen observer，只训练新增Program层；若该隔离仍失败，才用
owner entropy证据处理owner-structured readout。它不改变数据、slot/width/rank、优化超参、K权重或Gate，也不使用shuffled/reversed。

### 21. 冻结observer后最早接口是对固定38-owner轴置换不变的temporal readout

clean pushed `main@db84a50`的frozen-observer formal从fresh macro10按原world5 topology exact-resume到macro20；full相对endpoints的
held action/progress改善分别只有`+0.0051%/-0.0207%`，而fit total继续下降。无梯度诊断确认Stage 0 raw full event/owner relative RMS
保持`0.06252/0.36771`，full/endpoints的fused Program RMS差异也仍为`0.00618`，所以失败不再来自observer侵蚀或视频动态缺失。

training-only decoder原先用同一个`Linear(128,1)`给38个固定LoRA owners打分；同时置换owner content与score后加权和严格不变，因而
把有固定target语义的owner轴当成无身份集合。对应实证是owner entropy `0.99898`、action prediction temporal std `0.00173`，而target
为`0.32725`，继续训练到macro20没有修复。当前最小修正是38个固定owner各自持有一个跨task共享的linear query，只读取
`P_process` content；38条query从旧共享Linear完全相同的向量初始化，保持其余head的旧RNG序列，之后只由owner-specific梯度分化。
它不是task-ID route，也不改变deployment Program schema、Stage 0、probe、数据、loss、seed/LR、slot/width或Gate。
raw antithetic branch margin仍是独立接口，不能用不改变canonical Program或action/progress utility的residual缩放去美化Gate。

### 22. owner-specific scalar queries未解决时间均值坍缩，下一接口是query-time residual监督

clean pushed `main@407340b`的owner-specific scalar-query formal从fresh macro10 exact-resume到macro20；held full相对endpoints改善
分别仅`+0.0158%/-0.0340%`，probe均为`0/40`。query rows的分化从自身RMS的`1.58%`增至`2.94%`，但actual与强制shared-query的
macro20 held combined loss只差约`4.9e-5`，hard-owner也不改善，action prediction temporal std仍为`0.00171`而target为
`0.33589`。因此失败不是query没有更新或softmax温度不足，继续训练该scalar selection没有新机制依据。

raw Stage0 process配回其已训练action head可把held absolute action loss从`0.25511`降至`0.20767`，但full相对endpoints仍只有
`0.2467%`且prediction temporal std仅`0.00298`。这证明旧坐标/head值得复用，却否定“只转移旧head”会自然解决10%动态门；它并未
提供直接增加owner value map的充分证据。当前absolute cross-episode action/progress MSE主要奖励trajectory mean；有证据的下一修正是保留absolute项，
再等权加入query-centered action/progress residual MSE，使常数预测无法满足local temporal grounding。该修正不使用held梯度或
shuffled/reversed，不改变Program schema、模型容量、数据、K、seed/LR或Gate。

### 23. temporal residual未失败于表示容量，最早接口是optimizer cadence

clean pushed `main@68f8705`的temporal-residual fresh macro10仍为G2 non-pass：held20 full相对endpoints只改善`0.0381%`，probe
margin为`0/40`，而same-task、K1/K4与event范围继续通过。该结果先被冻结，没有立即再改Program架构。

后续可证伪诊断把问题分开：固定现有Program后，full-owner temporal readout相对endpoints可产生`15.17%`改善，说明已有动态bank
可被读出；tied-query与independent-query初始化的学习曲线近乎相同，排除对称初始化；cross-episode target也可识别。真正异常是旧
trainer把每macro的38个task全部累积后只做一次Adam更新，所以macro10只有10次更新。同一frozen readout的temporal loss从
`0.311873`开始，10/60步仅到`0.311827/0.311164`，200/500步才降到`0.294034/0.257824`。因此当前最早失效接口是优化时间尺度，
不是需要新增slot、width或第三种readout架构。

有证据的单一修正是保持Program、数据、loss、K与Gate不变，把一个macro拆成10个role-balanced optimizer steps：常规每step
2个target-fit+2个meta-fit，最后1+1并随macro轮换；scheduler和resume cursor按真实optimizer step计数。这个案例同时固化为后续
G2/G3/G4的诊断纪律：显著non-pass先冻结证据、定位最早接口并做可证伪probe，只有新机制证据才允许修改对应结构。

### 24. cadence恢复了宽泛动态信号，但近常数readout仍造成temporal gradient starvation

clean pushed `main@49e7769`的cadence fresh macro10实际完成100次optimizer update。held20 full相对endpoints改善由旧`0.0381%`
升到`0.3080%`，probe由`0/40`升到`13/40`，17/20 held task方向改善；same-task、K1/K4、event范围与tau仍通过。因此cadence
确实修正了一个真实问题，但幅度仍远低于`10%` Gate，不能把约`8.1x`相对提升冒充G2 pass。

冻结checkpoint后的fit-only梯度几何进一步定位接口：full/endpoints `P_process` delta RMS为`0.07296`，动态bank没有消失；full
action/progress prediction temporal std仅`0.00379/0.00160`，target为`0.35248/0.32500`。temporal与non-temporal梯度cosine在
Program process/decoder上只有`-0.065/-0.071`，不存在足以解释坍缩的强反向抵消；真正异常是temporal norm仅为non-temporal的
约`1/10`和`1/21`。也就是说，共用近常数readout时，query-centered loss虽然数值不小，却因时变state极小而形成自我维持的
弱梯度通道。

既有frozen-readout曲线显示100步后才开始展开、200--500步继续增长，所以同一formal exact-resume到macro20是有明确预测的时标
检验，不是盲目续训。若held增量和prediction temporal std不随之实质增长，学习时标解释即被证伪，后续应直接修改
Program-to-temporal-readout的残差/owner-value保留结构；这类结构修改是允许的，但必须由该证据驱动并fresh复评同一Gate。

### 25. macro20验证了readout学习时标，同时暴露K>1 canonical alignment坍缩

同一clean detached `49e7769` exact-resume到macro20/200 updates后，held20 full相对endpoints改善从macro10的`0.3080%`
跃升到`8.6878%`，probe margin从`13/40`升到`36/40`；18/20 tasks方向为正，8/20已超过`10%`。fit-only同一12-task
panel中，full action/progress prediction temporal std从macro10的`0.00379/0.00160`升到`0.03393/0.04789`，full相对
endpoints改善为`15.82%`。因此“100步后readout才开始展开”的时标预测得到验证，不能再把最早接口留在近常数readout，也没有
依据此轮直接换成full-owner value head。

Gate仍明确non-pass：median active events为`1`、one-event fraction为`1.0`，动态增量也尚未严格超过`10%`。分K证据把根因
精确定位到跨视频alignment：macro20训练条件中K=1仍为平均`6.42`个active events、one-event为0；全部K=2/K=4条件却都只有
1个active event。原始每video local presence仍有约7--8个有效槽，但learned DP把约`6/8` alignment mass集中到同一个
canonical slot，故不是Stage 0、阈值或总presence mass假象。

无梯度fit-only反事实只改变alignment measure：identity把K>1推到5--8个active events而过强；unit-step prior也改变了中间路径
偏好；仅把现有forward-only DP的首/末local slot分别锚到canonical 0/7，保留全部中间stay/skip、content/time emission与原transition，
就把K>1恢复为稳定3个active events，并将同一frozen decoder的full增量从`15.82%`略升到`16.47%`。因此当前最小结构修正是
boundary-anchored monotonic alignment；它不固定事件数，不改变K权重、loss、readout、数据或Gate，必须fresh复评。

### 26. boundary-anchored G2正式通过，冻结Program进入G3

clean pushed `main@c1493a1`只把K>1 monotonic DP的首/末local slot锚到canonical 0/7，保留全部中间stay/skip、content/time score、
uniform `beta_k=1/K`以及原readout/loss/data/LR/seed/Gate。fresh macro10已将held event指标从旧one-event坍缩修复为median 2、
one-event 0，但动态增量仅`0.8268%`；同一world4 exact-resume到macro20/200 updates后，full相对endpoints改善达到`22.2047%`，
probe `38/40`，median active events 4、one-event 0，same-task/K1/K4均通过，tau violation仅`0.00357`，所以G2 Gate正式pass。

这组因果对照同时验证两点：一是旧readout确有可用学习时标，不能因macro10弱信号误判为结构无容量；二是K>1单事件坍缩确由
未约束首尾的alignment path造成，边界锚定在不固定事件数量的前提下恢复动态资格。G3必须冻结该macro20 Program，只学习共享
Program-query到native content-key的signed selection；G1 task-local free logits不能进入部署路径。

### 27. G3 formal前的最早工程接口是实际action-flow监督与长视频反向显存

首个G3 runtime把expert member的flow response拟合误记为`cross_episode_flow`，而schedule保留的`action_demos`没有实际进入loss。
这不满足active design的独立action episode PI0.5 flow合同。修正后只为meta56+target-fit19建立授权query store；每step从与video demos
不相交的action episodes确定性取4个query，使用matched policy RNG计算generated唯一rank16 adapter的真实PI0.5 flow loss，再以
detached LoRA gradient bridge回传compiler。member flow仍保留在whole-trajectory single-member effect中，但不再冒充独立flow监督；
meta-held、target-held、validation与test actions均不读取。

真实两步profile随后暴露第二个工程接口：same-task primary与other两套完整Writer图同时驻留会在A40达到约`44.39 GiB`并OOM。
当前修正先完成primary全部主loss并backward，再以primary detached response作为轮换teacher，只让other response承担consistency梯度；
不同macro持续轮换primary，不改变deployment forward。每条video的chunked signed pooling同时使用activation checkpoint，反向逐video
重算同一online accumulator，避免candidate数线性保留激活。修正后三项真实profile全部通过：普通K1+K4峰值`16.68 GB`、包含
same-task consistency的K2+K4峰值`17.39 GB`、target93共332个采样帧的长K4峰值`29.28 GB`；三项关键gradient probes均finite/nonzero，
Action Meta module/parameter、source/Program trainable均为0。该结果只证明formal运行面接通和显存合同成立，不是G3闭环Gate结果。

### 28. G3 macro5 non-pass先发生在shared selection方向，不是多视频鲁棒性

clean detached `5140362`的首个G3 formal checkpoint完成macro5/95 optimizer updates；paired strict250的
carrier/language/full/first+final/same-task分别为`43/42/35/40/44`。full逐task为`27/4/4/0/0`，breadth`3/5`、carrier
retention`28/43`、Goal/Long均0、相对language/endpoints为`-7/-5`，只有same-task retention `33/35=94.3%`通过；三个video bank、
single compiler checkpoint、唯一完整rank16和Action Meta 0均通过authority检查。因此这是shared compiler的科学non-pass，不能用
内部loss或same-task稳定性冒充G3通过。

read-only几何把最早接口定位在Program到native signed selection：对四个G1非零held residual，macro5 full residual相对G1可行方向的
整体update cosine仅约`0.001--0.005`，而learned language residual约为`0.557--0.699`；full相对same-task residual cosine为
`0.992--0.999`，但full相对endpoints已有约`38--47%`的相对update差异。即模型对中间帧有反应、换同任务视频也稳定，反应方向却尚未
成为有用LoRA。训练侧global-member/effective-update到macro5仍约`0.936/0.892`，checkpoint的logit scales、uncertainty与scale bias
几乎停在初始化；95步中前50步又属于warmup，84%的target-fit tasks在已有五次访问内仍显示member/effective-update改善。

因此下一步不是LR/seed/width小扫，也不能直接宣判架构无容量：原formal schedule的macro10是对“有效训练时标不足”的一次明确
证伪节点。当前实现同时存在可检验的结构风险：正负pooling的raw差异无条件经`rms_normalize`变为完整方向，未学会的attention不能自然
退回carrier。若macro10仍近正交、full不优于language/endpoints，就应在任何macro20续训前修正signed-factor置信度/初始化，或以fit-task
可行native selection提供更直接的shared mapping supervision；不能靠继续同一路径或调loss数字掩盖。

### 29. G3 macro10证伪欠训练解释，最早接口是shared native selection supervision

同一clean detached `5140362`从fresh训练至macro10/190 updates；五臂paired strict250的
carrier/language/full/first+final/same-task=`43/42/38/39/40`。full逐task为Spatial0 `32`、Spatial9 `2`、Object8 `4`、
Goal5 `0`、Long6 `0`；breadth`3/5`、carrier retention `32/43`、相对language/endpoints `-4/-1`，只有same-task
retention `32/38=84.2%`和全部authority检查通过。相对macro5的`35`只增加3，仍无Goal/Long且不优于任何主要control，故预注册
“有效更新时标不足”解释已被closed-loop证伪，不能续训或用内部loss替代Gate。

训练记录进一步定位了原因：total/global-member/effective-update从约`2.381/1.015/0.929`降到`2.135/0.926/0.894`，但全部
190 steps都触发同一个global clip；macro10 pre-clip norm median约`10.87`，scale gradient均值约`13.88`，input/output query
分别约`0.754/1.057`。macro5到10的input/output query-key相对变化仅`2.14%/1.70%`，而scale组约`14.4%`。所以v1把间接
mobile functional target、selection与scale混在一个梯度预算中，scale path持续吞噬shared content selection的有效更新。

独立fit-only K1 functional-span证据排除了更早的容量失败：6 tasks/9 verified members的full-to-mobile update cosine median
`0.9978`，mobile named-effect retention median`0.9892`；投影到对应真实K1 native bank后，update cosine median仍为`0.7029`，
named/global functional retention median为`0.7855/0.7981`，positive action benefit为`9/9`。这不能证明held shared mapping，
但证明真实native X/Y span能保留强rank4 member功能；最早失效接口是Program-query到native-key的共享选择映射缺少直接可达监督。

因此首个有机制依据的修正不改Program、sampler、K、LR、rank或bounded beta，也不先加confidence：离线对formal40实际出现的K1
fit-task/video/member组合做稳定native投影，只封存pre-scale directions与scales。K1训练用detached set-valued functional
responsibilities选择member，再以gauge-invariant input/output subspace、paired update direction及独立small-core spectrum监督共享
query/key和scale；selection与scale/video分开clip。K2/K4严格不读teacher，继续只承担functional、flow、carrier、same-task与
multi-video职责。该teacher是fit-only training label，不是task/frame参数表，不进入deployment或checkpoint model state；其通过也仍须
由held5五臂closed loop证明shared mapping。

clean pushed `main@93dffc7`的实际封存与三步真实profile证明该修正按上述边界成立：formal40 K1 union为50 tasks、451 videos、
662 teacher states，held/Action Meta/deployment reads均0；K1精确lookup而K2/K4 tensor reads为0。profile中selection与scale/video
分别clip，K1 input/output query梯度均显著非零，长K4峰值29.32GB且唯一rank16被policy实际消费。该结果只消除了loader、梯度墙、
显存和materialization工程风险；是否学会shared mapping仍必须看fresh macro5的fit-teacher曲线与held closed loop，不能用初始
teacher loss或gradient大小提前宣称通过。

### 30. G3 v2 direct teacher仍被旧credit覆盖，不能靠续训或调权重修复

clean detached `2a7f760`的v2 fresh macro5/95 updates完成后，五臂strict250的carrier/language/full/first+final/same-task为
`43/42/41/38/37`。full breadth`3/5`、carrier retention`33/43`、Goal/Long均0、相对language/endpoints`-1/+3`、same-task
retention`73.2%`；单checkpoint、三条video banks、唯一rank16、配对与Action Meta 0均有效，shuffled/reversed未使用。相对v1虽有
小幅改善，但仍没有G3所需的跨suite、视频必要增量或same-task稳定性。

固定同一fit K1真实bank比较deterministic step0和macro5，input/output subspace从`0.9298/0.9292`降到`0.9070/0.9083`，
paired update cosine反而从`0.00409`降至`0.00299`，spectrum loss从`3.7536`恶化到`4.2118`。梯度墙本身严格为0泄漏，但teacher
selection梯度范数仅`0.3235`，同一步其它functional/flow/carrier selection梯度为`21.8015`；teacher spectrum与其它scale梯度cosine
为`-0.989657`。因此“分组clip即可让direct teacher起作用”的v2假设被反证：问题不是scale再度消费selection clip，而是两个目标仍在
各自参数组内直接争夺同一次更新。

teacher-only反事实能让同一条件的selection、paired update和spectrum共同下降，说明teacher loader、真实bank、shared forward和
autograd链路有效；但paired-update梯度又明显小于两个subspace分量，等权scalar objective没有把最终LoRA更新方向作为首要credit。
下一修正必须隔离fit-K1 mapping acquisition与旧functional职责，并以paired update为首要可证伪量；K2/K4仍不能读取teacher，最终
checkpoint仍需恢复多视频职责并由held closed loop判定。该证据不支持task/frame lookup、改变rank/K或直接进入G4。

owner进一步明确：上述LoRA teacher只承担G3中间接口监督，不能被机械延伸为Final数据合同。G4/Final不得假设每个任务都有目标LoRA；
在授权fit/meta tasks上直接以teacher actions、functional/on-policy闭环信号训练完整Writer是正式候选路径，具体loss删留仍由实际
closed-loop效果与最早失效接口决定。deployment信息墙与zero-interaction输入合同不变。

正式训练的world size是吞吐分片选择，不得成为科学batch定义。当前G3以固定3个target-fit加3个meta-fit的全局optimizer group
保持task/role权重与update cadence不变，再按launch时1--6张有效GPU做cost-balanced分片；最后1+1尾step允许其余rank自然无本地task，
但参加同一全局梯度归并。exact-resume仍锁定run最初topology。

### 31. G3最早selection接口是高条件数native dual的获取，不是signed pooling表达力

在固定fit K1 `meta9/video40/member=meta_step1000`、同一真实target20 q bank上，直接优化free full-native query即使到500步，
canonical factor cosine也只有`0.4313`、effective-update cosine只有`0.1624`；uniform frame measure、raw key、LayerNorm、独立或
paired antithetic branches均没有结构性改善。相反，按冻结G2 `rho`与canonical assignment形成的base measure，对同一X/Y bank做
FP64 stable inverse-covariance dual解析解，input/output factor cosine达到`0.99628/0.99997`，完整update cosine为`0.99750`，而
各bank retained scatter condition约`9.6e5--9.9e5`。把该dual缩到最大absolute logit `0.1`后放回现有online-compatible
`softmax(+s)-softmax(-s)`，update cosine仍为`0.99749`，所以不需要以第二次bank读取或新pooling公式解释失败。

这组反事实把最早接口进一步定位为：旧factor/update loss要求共享Program query通过普通梯度隐式学会每个bank约`1e6`条件数的
inverse-covariance preconditioning；bank、candidate索引、两路softmax、chunked accumulator、rank4和materialization本身均能表达
teacher方向。下一步先在fit-only analytic dual集合上做按task留出的低维target-native key-basis oracle，以一次解析谱及held-out真实bank
回放判定低维共享basis是否保留full dual；不通过多个width训练选分数。只有该oracle有信号，才允许把compact dual supervision显式写入
active design/schema并实现shared Program-to-coefficient mapping；内部cosine仍不能替代最终held5 closed loop。

### 32. compact dual-basis失败同时暴露了functional geometry与query-span两层问题

clean pushed detached `main@e7d86b0`的50-task、98-condition、四family task-LOTO formal中，完整解析dual回放仍稳定成立：task-mean
update cosine median/p10/min为`0.996949/0.995468/0.993884`，worst-video为`0.996487/0.994944/0.991649`。但同一真实bank把
target-specific raw dual压到最大128维后，overall median/p10只有`0.288444/0.249615`，`0/50` tasks达到`0.95`；action-in
单独接近1不能掩盖action-out `0.146885`、q `0.000490`和v `-0.000586`。因此该Gate淘汰的是`<=128`固定raw-dual code，
不是native bank、signed pooling或Native-Factor。

对最差q/v/action-out条件做bank-conditioned functional least-squares后，128维update cosine分别约为`0.684/0.507/0.805`，远高于
错误的欧氏dual投影但仍不通过；同一LOTO span要到约384--512维才接近强回放，selected-task effect basis也呈相同宽度需求。这说明不能
用换一种投影方式恢复fixed basis/effect realizer。现有compiler还把key按native width共享，且`input_query/output_query`最终只是从
128维context做一次线性映射；即使只把key width调大，query仍落在一个固定至多约128维的线性像中，会重新引入formal已否定的
compact-span风险。后继必须直接验证content-derived key的functional image、exact有界softmax与共享Program mapping，并允许
owner-specific、非线性高容量query生成；这仍不得成为task/frame lookup或直接factor hyperdecoder。

小型fit-only screen进一步把两项职责分开：当前width-shared 64维近线性key即使拟合两条同task视频，第三条video的三family update cosine
均值也只到约`0.52`；owner-specific复制改善训练拟合但不解决未见video。按前述解析容量选择512维owner-specific key时，自由解析
functional span在三条未见video上约`0.99`，证明高容量动态key具有候选表达力；但最小/最大奇异值比约`1e-8--1e-6`，尚未证明
Program query能稳定取得所需scores。所以下一检查是固定bank的截断谱与exact bounded signed-softmax可用性，不是继续width、seed或LR扫。

exact可用性检查给出了更强的架构选择证据。随机512维key只有使用约`1e7`条件数的tail时才使q/v/action-out达到约
`0.993/0.984/0.994`，在`1e6`内只有约`0.956/0.936/0.966`；同task三条video间的query cosine也很低，v甚至为负。
直接使用真实native X/Y作为content key则在`1e6`截断时已足够：用固定、deployment-compatible的`0.01`全局small-logit scale而非
读取完整bank做逐条件校准，三family跨三video update cosine均值为`0.99886/0.99551/0.99788`，minimum为
`0.99810/0.99447/0.99703`。这证明不需要高成本512维candidate projection，最早接口转为Program对owner-native dual/score的获取。
q的八个output groups还暴露独立幅度职责：逐group单位化把update降到约`0.967--0.985`，而将解析query norm化为`[0,1]` bounded
relative gains并只保留一个公共score scale可恢复约`0.999`。首个mapping修正因此必须同时包含非线性高容量query生成和显式group gain；
它仍只通过真实X/Y的signed pooling生成factor，不是full FactorHead或fixed effect realizer。

### 33. 同task功能稳定但解析dual旋转，逐video score不是shared mapping标签

对q/v/action-out各选一个有三条K1 video的真实fit condition复核后，teacher effective update的same-task cross-video cosine均值约为
`0.873/0.866/0.884`，冻结G2 Program flattened cosine均值为`0.9971`；因此任务功能和Program都没有随video随机漂移。漂移的是将该
功能从当前bank取出的minimum-norm inverse-covariance dual：直接跨video replay时q/v update接近零、action-out约`0.086`，一条raw
query同时适配三条video的解析upper bound也只有`0.736/0.381/0.823`。保留8个canonical event query可拟合两条video，却在第三条
video降至`-0.004/0.012/0.049`；每event单帧anchor更低。这排除了“只需保留event轴”或“只需稀疏稳定锚点”的解释。

candidate-local nonlinear 512D key配合一个跨video共享free query的factor训练也未建立该映射：q/v/action-out的train update为
`0.177/0.244/0.593`，held-video为`0.105/0.175/0.487`。为区分factor loss难优化与标签不可泛化，又直接监督解析score并固定训练
2000步；训练score cosine已到q `0.887/0.699`、v `0.897/0.722`、action-out `0.912/0.979`，但held q/v分别只有
`0.133/0.111`和`-0.246/-0.232`，最终held update为`-0.001/-0.003`。action-out output score能迁移到`0.961`，仍因input只有
`0.491`而使update仅`0.114`。所以不是简单续训问题：约`1e6`条件数使不精确score无法保留factor，而逐video解析score本身又随全bank
measure/covariance改变。

50-task/98-condition的frozen-Program dual decoder给出一致证据：除近恒等action-in外，q/v/action-out在task holdout只有约
`0.07/0.05/0.03`，held-video约`0.293/0.224/0.292`；模型可记住train却不能把稳定Program解码成新的video dual。结论只淘汰
“Program或单candidate内容直接回归逐video analytic dual/score”及当前已测raw/event/anchor实现，不证明所有content attention失败。
剩余可证伪方向是跨大量task/video用paired factor功能监督学习bank-independent canonicalizer；若它仍失败，则需要认真考虑先累计
bank-global sufficient statistics再condition query/key，或等价的额外pass。后者会修改当前Pass B流式合同，不能在专家复核前假装成
普通实现细节，也不能恢复neural FactorHead、fixed effect realizer或task/video lookup。

### 34. 第二次专家复核把G3修正为current-bank-conditioned operator；Final保留整套Writer随机fresh选项

全新专家锁定远程`main@ed2883b`及其可达历史后，确认G1 bank capacity和G2 Natural Program结论仍成立；G3的结构问题是稳定的
task/video功能被表达成随当前candidate measure/covariance旋转、条件数约`1e6`的minimum-norm dual/score coordinates。旧
candidate-local one-pass compiler在query形成前看不到这个bank-global gauge，因此即使train score cosine很高也不能稳定保留q/v factor。
这只淘汰已测pointwise/raw/event/anchor/direct-score实现，不淘汰真实native banks、signed pooling或Native-Factor主线。

active G3改为两阶段流式、set-equivariant的bank-conditioned Pass B：每条video的B0按单位质量累计native mean/covariance与
Program-conditioned native anchors，regularized solve形成query；B1重放同一bank，精确重建X与abs/adj/init/goal Y banks，以正负
softmax之差pool真实native values。内部多次只读同一授权bank仍是rollout前一次Writer调用；不存在task-local适配、禁用信息或第二
adapter。`global_statistics_off`只作一次预注册candidate-local消融，若off失败/on通过即删除并正式淘汰严格one-pass假设。

owner接受上述G3裁决，但明确不同于专家的Final默认偏好：完全随机初始化整套Writer并从头端到端fresh联合训练必须保留为Final正式
可选项。G1--G3的分段冻结是因果验证，不是Final必须照搬的课程；若整体梯度下降能形成Program/anchor/selection/scale内部功能分化，
不应人为分段。通过组件初始化与全随机初始化都必须使用fresh optimizer/scheduler、同一信息墙与closed-loop合同；Final不预设存在
目标LoRA，最终选择不能由内部loss代替。

### 35. F1证明bank-conditioned operator能恢复解析上限，但不替代shared mapping Gate

clean pushed detached `main@435cb4a`在既有50-task/98-condition analytic-dual authority上完成B0/B1 operator formal。代表38-target
四类native拓扑的q20、v21、action-in36、action-out37均使用真实X与abs/adj/init/goal Y、固定G2 measure、Program-conditioned
bounded analytic anchors、FP64 current-bank covariance截断谱solve及exact antithetic signed replay；streaming严格跨frame chunks保留
adjacent/init/goal视频边界状态。analytic teacher只用于隔离operator capacity，不进入deployment模块或checkpoint。

四family的operator-to-analytic task-mean median为`0.999871/0.999824/0.999960/0.999884`，minimum为
`0.999757/0.999544/0.999951/0.999743`；536 rows的streaming-to-materialized minimum为`0.99999988`，故预注册F1 Gate明确
通过。Action Meta module/parameter为0、held action/reward reads为0。该结果排除了显式covariance、截断谱solve、output-group
relative gain及chunked replay本身是G3当前瓶颈，也不需要转向matrix-free block-CG/Lanczos；它没有训练shared anchor scorer，不能
被解释为Program-to-attention mapping或closed-loop成功。下一最早接口仍是F0 canonical forward与F2/F3 shared mapping acquisition。

吞吐合同也用真实profile固定：单worker约`19.3GB`reserved，gpu01 p1--p6每卡双worker时六卡均约`37.5--37.8GB`且稳态UTL
`100%`，12个cost-balanced workers在最长`228.44s`内完成。第三worker没有安全显存余量；后续仍按任务图和live状态选择card/process
数，而不是把“多卡”或“显存占满”本身当科学结果。

### 36. F2的off边界、F3训练权重与吞吐实现已被精确定义

`global_statistics_off`采用专家允许的`C=I`消融：仍以每video单位measure累计centered first-moment native anchor，再由B1对真实X/Y做
两branch exact replay；它只关闭current-bank covariance/preconditioning，不等于固定query、普通平均或完全不读bank的字面单pass。
因此F2只检验candidate-local compatibility加first-moment anchor能否泛化；若F2失败而F3通过，应删除off模式并淘汰该假设，不能把
F2 non-pass解释为两阶段bank-conditioned Writer失败。

预注册451条件解析为40 fit tasks（25 meta、15 target）、40 held-video和10 task-holdout/82 conditions。为保持两种role每步各50%，
每macro固定5个六任务updates，完整覆盖15个target并按seed从25个meta中轮换15个；不能沿用旧“19+19尾step”的历史表述。
同一科学batch在单卡与gpu01物理1/2/4/5/6五卡的真实profile中分别为`181.21s`和`44.96s`，五卡约`4.03x`，各卡计算段大多
`100%` UTL。后续GPU效率同时看world-size scaling与单卡SM/UTL、显存峰值、step time/LoRA吞吐；不以dummy显存占用或48GB填满率
替代有效计算。

### 37. F0证明canonical B0/B1图接通，chunk Gate必须比较有效更新而非rank槽位

首次同bank chunk4/one-chunk复核的raw A/B槽位最大差为`.00311`，但solve metrics最大差仅`3.71e-14`。small-core SVD的rank槽位
允许符号、顺序和子空间内旋转，因此逐槽坐标不是LoRA功能等价量。clean detached `19b5b3f`改用不物化大矩阵的
`B.T @ A` Frobenius内积后，38 targets最终更新cosine最低`.99999976`、相对误差最高`.00066443`；raw槽位差仍完整保留为诊断，
没有放宽其阈值冒充通过。

同一formal F0同时证明真实K1梯度、B0/B1 chunk边界、K4均匀集合聚合/置换不变、teacher零读取、Action Meta实际未加载及唯一
rank16 policy consumption均成立，故工程Gate通过并解封F2。它不测mapping泛化或closed-loop，不能被解释成F2/F3或G3通过。

### 38. F2正式淘汰`C=I` first-moment容量假设，但不反证current-bank operator

clean pushed detached `2199a76`的一次性F2消融令`C=I`、保留B0单位measure centered first-moment anchor与B1 exact replay，且只训练
约101万参数的shared anchor scorer。world6 fresh macro1到macro5的mean recovery从`.000639`单调升到`.019690`，说明优化图有响应；
但451条件task-equal aggregate的fit/held-video/task-holdout median仍只有`.022243/.022858/.018919`。held-video四family median为
action-in `.039958`、action-out `.022185`、q `.004722`、v `.023158`，相对F2要求的overall `.75`、family `.65`、task-holdout
`.60`不是边缘不达标。

关键区分是fit本身也接近零，held与fit数量级相同；因此最早失效接口不是跨video/task泛化，而是candidate-local scorer加first-moment
anchor无法形成teacher功能。451条件全覆盖、六worker completion、0 held gradients、0 Action Meta及0 shuffled/reversed use排除了
评测缺行或信息墙污染。F1已经独立证明同一真实bank在current-bank covariance solve下可恢复analytic factors，所以F2只否定off模式，
不授权放弃两阶段bank-conditioned Pass B。下一项有信息量的实验是fresh F3，不是继续F2或扫描LR、seed、width。

### 39. F3 current-bank solve修复了泛化与部分幅度，但shared anchor acquisition仍有family结构瓶颈

clean detached `c1e26ce`的F3从fresh macro5以同一world6 optimizer/scheduler/topology exact-resume到macro10。训练mean recovery从
`.002204`单调升到`.087444`，macro5与macro10的451-condition held median分别为`.048433/.089704`，证明current-bank solve相对
F2 `.022858`有真实增量；但仍不是接近`.75`门槛的边缘失败。macro10 fit/held/task-holdout median为
`.089915/.089704/.096849`，held/fit `.997650`，所以训练分布、held video和held task处于同一量级，不能把失败归因于过拟合、
task split或video泛化。

最关键的不对称在family：macro10 held action-out/action-in已到`.177230/.125947`，v只有`.052761`，q仅`.013288`。F1已用相同真实
banks证明四family operator-to-analytic recovery均约`.9998`，所以q/v弱不是covariance solve、B1 replay或native bank容量不足。
macro5到10 held median虽增加`.041271`，最后单macro训练增量已降至`.005105`；继续macro20无法解释family差异，只会把结构问题
伪装成训练时长。下一最早接口是shared anchor scorer如何从Program/native content生成q/v functional anchors及其梯度尺度；应先做
同条件、同checkpoint的family幅度/方向/gradient分解，再决定单一机制修正。

### 40. F3最早失败是两侧subspace credit starvation，不是rank pairing或shared泛化

在不修改checkpoint的前提下，对`c1e26ce` macro5/macro10同一task93/video31真实bank逐family计算student rank4与teacher两侧
row-subspace的最大可达update ceiling，并分别反传旧update-only objective。macro10的实际update recovery与input/output one-sided
ceiling依次为：q `.012892/.192547/.094122`，v `.046297/.260870/.282059`，action-in
`.125741/.775570/.145645`，action-out `.094190/.237240/.657518`。两侧ceiling的乘积已近似解释family层级，说明最终
pairing并不是最早接口；q从macro5到10的ceiling只由`.184690/.089680`增至`.192547/.094122`，held task2/video4 q也只有
`.013565/.204308/.088427`，所以这不是fit或单condition特例。

旧loss虽把四family scalar等权，但双线性完整update对已经错误的一侧只能通过另一侧传梯度；q input/output key gradient norm只有
`.0643/.0313`，而action-out为`2.6973/.6383`，差约一个到两个数量级。F1的四family operator recovery约`.9998`、solve residual
约`1e-12`、retained trace约`.99996`，进一步排除了B0/solve/B1数值失败。故当前有证据的单变量修正是保留由完整38-target
update选出的一个global member posterior，并用该detached posterior对input subspace、output subspace和paired update direction固定
等权；不改Program、banks、query/key容量、rank、group gain、data、LR或seed，也不恢复per-video dual/score。

六卡真实5-macro qualification使上述三项loss从`.939056/.922342/.999256`连续降至
`.923254/.902963/.997695`，Action Meta 0、source/Program/scale冻结且梯度有限。这只证明修正后的credit graph能直接改善最早接口，
不证明shared mapping或G3 Gate；下一步仍须从clean pushed detached commit fresh训练并完整评估451 conditions。

### 41. 等权subspace credit仍被共享parameter ownership覆盖，family/fixed-owner分解获得修正资格

clean pushed detached `84903aa`把input subspace、output subspace和paired update固定等权后，从fresh训练到macro5并exact-resume到
macro10。三项训练loss持续下降，macro10为`.771307/.825056/.930808`；但完整451-condition macro5/macro10 held median仅
`.025418/.073029`，macro10 p10 `.057174`，远低于`.75/.50` Gate。held/fit `.998320`、task-holdout `.087636`说明泛化没有先坏；
macro10 action-in/action-out/q/v held median仅`.098990/.146806/.008482/.040693`，也没有超过旧update-only F3。因此该结果淘汰
“只要给两侧直接等权credit就足够”的假设，不授权续到macro20或调loss权重。

同一真实bank的family backward显示更早的共享参数竞争：q/v/action-in/action-out output-key norm为
`.022128/.056223/.014089/.251221`，output-query为`.018642/.055363/.016834/.305811`，跨family梯度大多近正交且action-out
支配幅度。q的18个固定层目标pairwise median cosine约0，aggregate gradient只有per-target norm和的约`.29--.32`，层间norm差最高
约`20.5x`；独立task94/video11把aggregate ratio复现为`.282/.276`。macro10 q input/output span ceiling均值
`.235654/.093766`而实际update cosine仅`.008407`；结合F1约`.9998`的同bank operator recovery，说明native bank容量存在，
但当前width-shared scorer尚未把它同时暴露给不同family/层owner。

这与第二位专家“family共享trunk、固定owner/group用FiLM或embedding”的明确建议一致。当前有证据的单变量修正是在唯一canonical
scorer内将Program/rank/query/event/gain/native-candidate模块按四family共享，并以38-target固定LoRA拓扑的zero-init bounded FiLM
调制candidate hidden direction。它不是task或frame查表，不改变Program、bank、width、rank、scale、loss、data、optimizer或Gate。
真实一步profile使222/222 trainable tensors进入optimizer state，Action Meta/source/Program/scale trainable均为0；这只解封fresh
formal F3，仍须以同一451 Gate判断。

### 42. family/fixed-owner分解仍未取得绝对mapping，direct-native并不是有效的bank-stable修正

clean pushed detached `c3fc8e3`把四family trunk及38个fixed owner的bounded modulation从fresh训练到macro5/macro10，完整451-condition
macro10 fit/held-video/task-holdout median只有`.074715/.074620/.081644`，held p10 `.058381`、held/fit `.998724`；held q/v/
action-in/action-out为`.027938/.066509/.044464/.164942`。它和`84903aa`一样是fit、held-video、task-holdout同量级但绝对能力很低，
所以parameter sharing竞争不是唯一根因，也不应继续macro20。

冻结同一candidate map、允许task-local free event query后，稳定`1e-3`谱下q/v/action-in/action-out joint update ceiling约为
`.226/.315/.975/.629`；raw-native key约`.250/.336/.960/.600`，FiLM tangent约`.280/.381/.973/.645`。降到`1e-6`虽可恢复
大部分方向，却要求使用三到六个数量级更弱的奇异尾。action-in的`.975`容量与训练held `.044`并存，明确区分了两个问题：q/v/
action-out key conditioning不足，以及即便方向可表达，shared stable code仍没有选择到它。

`4117117`的direct-native scorer完成了真实F0工程合同，但后续代数检查发现`a=Cq`后再做`C^-1a`在可逆子空间近似返回原始`q`，
等价于把Program raw query直接跨video transfer；它没有消除随bank变化的识别问题。故没有启动formal F3，及时回退该活动实现。
这只淘汰direct-native query/FiLM tangent这一具体修正，不淘汰真实X/Y、signed pooling或bank-conditioned operator。

### 43. same-task feature chart存在强共同code，失败来自minimum-norm两video解的巨大nullspace

在task93同一teacher members、train videos 31/32与held video46上，先冻结`c3fc8e3` candidate features并只读比较共同code。full
feature inverse的两video inductive held q/v/action-out约为`0/-0.001/0`，action-in约`.097`；改用symmetric inverse-square-root后
约为`-.003/.001/.035/.593`。但把held video只加入共同code估计的transductive正控制时，q/v/action-out立即达到约
`.896/.902/.926`，action-in为`1.0`，说明三个bank确实共享强task-level feature code，只是两video minimum-norm解落入未被第三bank
观测的train-only nullspace。

按8个canonical events分别做128维inverse-square-root后，q/v/action-out两video inductive仍近零，action-in升到`.986`；
transductive q/v/action-out为`.905/.912/.929`。所有q/v/action-out event blocks均满rank128，故继续调全局/分event谱floor或再加
covariance并不能识别稳定code。这是analytic task-local interface oracle，不训练shared mapping，也不把held视频或teacher信息送入
deployment；其因果作用是把下一修正锁定为same-task稳定anchor，而不是另一次width/LR/loss sweep。

### 44. 当前F3以`P_lang`固定task anchor、动态Program event measure和per-event feature gauge分离稳定与视频职责

冻结G2 Natural Program中，`P_lang`只由exact language产生，同task不同video确定性相同；`P_scene/P_process/rho/tau/sigma`则承载
video scene/process/alignment。当前唯一canonical修正据此把两种职责分开：family-shared query以`P_lang`加固定owner/event/rank topology
形成task-stable anchor code；动态Program字段和canonical assignment只控制每video的event/frame measure；真实native direction/
log-magnitude与frame/probe/horizon/type metadata经family-shared、fixed-owner bounded candidate encoder后，按每video、每event统计
detached symmetric inverse-square-root，再与stable query做content compatibility。

这仍不是language-only Writer：`P_lang`只提供“寻找什么”的稳定坐标，最终native anchor、B0 solve和B1 positive-minus-negative softmax
都必须读取当前video真实X/Y，LoRA factor始终是这些values的有符号加权和。三次流式读取保持每video adj/init/goal边界并只发生在
rollout前一次Writer调用。world6真实profile已完成一组3+3 optimizer update，六张A40计算段基本满载，step `77.806s`、峰值
`25.59/25.99GB`，Action Meta/source/Program/scale trainable及held gradient均为0；这只获得clean detached F0/formal F3资格，不能
冒充mapping Gate。

### 45. stable anchor修复了迁移，当前最早接口是q target在shared query head中的梯度相消

clean detached `main@20acc33`的stable-anchor F3从fresh训练并exact-resume到macro10后，完整451-condition fit/held-video/
task-holdout median达到`.141080/.142120/.145828`，held/fit为`1.00737`，且40/40 held tasks相对macro5改善。这比此前
family/fixed-owner macro10的held `.074620`有实质增量，说明task-stable `P_lang`加per-event feature gauge确实修复了跨video与
跨task迁移；但held p10仅`.116653`且q/v/action-in/action-out为`.030186/.110266/.180031/.253562`，仍不满足F3 Gate。

六个meta/target task各自只用一条fit video继续优化20步时，另一fit与held video始终跟随train，overall达到约`.20--.25`；q却只到
`.0197--.0277`，而action-in/out通常到`.31--.49`。所以最早问题不是held泛化、视频不稳定或仅仅50-task语义竞争。task93的18个q
targets对family-shared input/output query heads产生近正交且大量相消的梯度：aggregate-to-norm-sum仅`.272/.268`，153对中有
`76/74`对负向；candidate key trunk的相应比例为`.364/.602`，但它已经拥有fixed-owner FiLM，query侧没有对称的owner梯度路径。

冻结G2 `P_lang`的owner-baseline-free task variation仍有约`3.2--3.6` effective rank并严格same-task跨video不变；这不足以证明最终
语义容量已充分，却也不支持在更早query-ownership接口未修复时重训G2。当前最小因果修正只给family-shared query trunks加入
zero-init bounded fixed-owner input FiLM与fixed-owner/output-group FiLM；它们表示38-target及真实output-group固定拓扑，不包含task/
video/member/frame ID，task-dependent query仍由共享trunk读取`P_lang`。若fresh F3仍失败，再按owner/group与task-content分解决定是否
重开language content接口，不能把该probe或内部recovery冒充shared mapping Gate。

### 46. fixed-owner query路径已通过F0，生命周期错误不改变科学判断

首个clean pushed `7e232b0` F0在GPU计算前因内部`_apply` helper覆盖`torch.nn.Module._apply`而失败；将其唯一改名为
`_modulate`并补`.to(device)`回归后，clean pushed detached `d64f7ad`通过完整F0。新input/output owner-query gradients分别为
`.015828/.000958`，证明38-target固定owner/group路径实际进入训练图；Action Meta与teacher reads仍为0，K4均匀measure与置换
不变、chunk有效更新一致性及唯一完整rank16 materialization全部保持。该证据排除了“新路径未接图”的工程问题，但不回答它能否
提高shared mapping；后者仍只由fresh F3的451-condition primary及相邻checkpoint Gate判断。

### 47. fixed-owner query只实质帮助action-in，q/v瓶颈继续下沉到candidate compatibility image

clean detached `3e4e9a0`的fresh F3 macro10把held median从stable-anchor的`.142120`提高到`.163128`，40/40 tasks同向改善，但
q/v仍只有`.032001/.111951`；action-in由`.180031`提高到`.256629`，几乎解释了全部新增收益。四臂ablation进一步显示input
owner-query路径的overall因果效应只有`.000193`，output路径的主要效应也是action-in `.056749`，所以“query owner梯度已接通”与
“q/v functional selection已学会”必须严格区分。

仅优化fixed FiLM的task-local probes即使把query移动约半个base RMS，也不能改善完整q/v update。更强的free-query正控制直接移除
FiLM、family trunk和Program-to-query表达约束，却在六个fit tasks上只把q/v update median由`.02983/.10730`提高到
`.06519/.14487`。这不是严格收敛上界，但足以阻止下一步盲目扩展owner query head：当前candidate encoder、whitened compatibility与
bounded anchor形成的可达image至少同样可疑。F1已证明真实X/Y、covariance solve和signed replay在analytic anchors下约`.9998`，故
下一项高信息量工作是把“free score可达、current-key free query可达、shared Program query可达”三层容量分开，而不是再改loss或续训。

### 48. 深层target解析见证证明线性query-key image本身失容，根因不只是loss或共享梯度

在`3e4e9a0/macro10`同一task85/video34真实bank上，六task free-query延长到100步后q/v update仍仅约`.1455/.2197`；进一步把
credit缩到单个target，分别使用update-only、先subspace后update、small-core SVD balanced pair和条件最小二乘pair，也只得到约
`.06--.21`。这些对照均为fit-only、Action Meta 0、冻结Program/candidate/operator，并且没有held/validation梯度；它们排除了
“只因18个target平均稀释”以及“换一个gauge-aware pair loss就会恢复”的解释。

更决定性的exact B0/solve/B1解析见证将teacher native dual投影回**当前冻结candidate key产生的compatibility image**。浅层
target0 q与target1 v在放宽到`1e-6`奇异尾时仍可达到约`.994/.997`，但layer9的target18/19只能达到
`.5186/.5583`，layer17的target34/35也只有`.6537/.6079`；稳定`1e-3`谱下四个深层target更只有
`.0861/.0892/.1824/.2076`。相同bank直接native factor reference仍为`.995--.997`，而深层失败主要来自input侧：`1e-6`下input
rank均值仅约`.51--.65`，output侧已约`.995--1.0`，且所需input key-image coefficient RMS最高约`5.7e4`。因此真实X/Y、rank4、
native covariance solve和B1 signed pooling仍有容量；当前受限的`query dot whitened_key`函数族却无法稳定表示深层input选择。

这一结果为单一结构修正提供了直接资格：保留旧点积作为已验证浅层残差，同时按专家原式允许的
`tanh f_j(c, X_hat, metadata, assignment)`加入family-shared additive joint compatibility。它仍只输出每个
query/candidate的一个bounded scalar，不输出高维factor，不含task/video/frame/member表；Program、candidate encoder、真实banks、
B0/solve/B1、rank、data、loss和F3 Gate均不变。该修正只有在真实F0及fresh 451-condition F3通过后才算shared mapping成立。

### 49. joint scorer首轮F0失败来自signed初始化抵消，而不是chunk边界或新函数族本身

clean pushed detached `a2a56a7`的首轮真实F0在训练前被chunk Gate拦截：同一cached task93 K1 bank的chunk4/one-chunk
feature metric误差为0、solve metric误差约`7.2e-13`，但38-target有效更新minimum cosine为`.9999365`、maximum relative
error为`.01127`，未达到`.99999/.005`合同。K4置换误差仍为`1.91e-6`且Action Meta为0；因此这是signed factor数值接口，
不是video boundary、bank capture、solve或scientific mapping non-pass。

固定同一真实bank的机制对照进一步隔离了原因。仅把additive scalar缩到`.1/.03/0`时maximum relative error仍为
`.00863/.00859/.00859`，排除“只因joint幅度过大”；保持旧随机初始化流并以`.03`启动可降到`.00184`。更直接地，把positive/
negative query rows初始化为严格antithetic且joint以`.03`非零残差启动时，在不依赖旧随机状态下达到minimum cosine
`.9999965`、maximum relative error`.00264`；同一antithetic初始化若让joint满幅启动仍为`.00678`。故当前唯一修正同时采用
可立即解绑训练的antithetic signed初始化和small-nonzero joint residual。它不改Program、bank、operator、rank、loss、data或Gate，
也不放宽F0阈值；下一步必须从新clean pushed detached commit重跑完整K1/K4 F0。

该唯一修正随后由clean pushed detached `e784eb9`通过完整formal F0。task93 chunk4/one-chunk的minimum update cosine为
`.999996512`、maximum relative error`.00264108`、median relative error`5.34e-5`，K4置换误差`1.91e-6`；input/output
joint query/key/scalar、candidate、anchor和owner-query梯度均finite/nonzero。Action Meta module/parameter、source/Program trainable与
K4 teacher reads均为0，38-target/76-tensor唯一rank16被policy实际消费。六卡真实mapping profile再完成固定3 meta+3 target的一个
optimizer step：`89.83s`，单卡峰值allocated/reserved约`25.60/26.00GB`，4,102,024个compiler参数为唯一trainable owner，
joint gradient probes全部非零且held/validation/test gradients为0。这只解封从fresh运行同一451-condition F3，不证明shared mapping。

### 50. joint compatibility formal non-pass揭示Program、interaction与conditioning三层连续失效

clean pushed detached `main@55710bb`的additive joint compatibility从fresh运行到macro5并按原world6 topology exact-resume至
macro10，两个checkpoint均完整评估451 conditions。macro10 fit/held-video/task-holdout task median为
`.126205/.128720/.129465`，held p10 `.103610`、held/fit `1.019925`；held q/v/action-in/action-out为
`.025341/.095210/.121352/.276225`。所有451 conditions相对macro5改善且fit/held/task-holdout一致，说明没有评测缺失或单独的
held泛化崩溃，但绝对值低于前一owner-query FiLM的`.163128`并远低于`.75/.50`，所以明确non-pass且不续macro20。

四个fit-only tasks的checkpoint因果消融显示：关闭joint后16个family update cosine median/minimum为`.999752/.999450`且recovery
平均只变`-.000175`；只保留joint时recovery仅`.0018--.0116`。joint/dot score RMS median约`.0735`，macro5到10 joint
query/key参数相对变化通常只有约`1--2%`，small scalar residual使其虽然有梯度却没有成为实际selection路径。该结果淘汰当前
small-residual additive启动/credit实现，不淘汰所有nonlinear compatibility。

同一因果panel还揭示任务条件坍缩。task85与wrong task93的`P_lang` cosine为`.99704`，当前P_lang-only query trunk输出的各family
query cosine为`.9947--1.0`；wrong-language替换后最终update family cosine minimum仍为`.98020`。相反，G2完整full
`rank_event`在same-task不同video间为`.99750`，wrong-task dynamic为`.92811`，说明稳定且可区分的动态Program信息实际存在，但当前
`stable_rank_event`没有读取它。wrong-dynamic替换后的最终update minimum仍为`.99211`，与代码中dynamic fields只控制measure、
不进入content query完全一致。

最后，task85两条fit video训练、第三条video零梯度的task-local final-factor对照把candidate/solve接口分开。200步后current-key
train/held为target0 q `.4235/.4237`、target1 v `.4849/.4852`、target18 q `.2920/.2425`、target19 v
`.2905/.2212`、target34 q `.2741/.1345`，action-in target36则为`.9963/.9961`。直接读取raw native并学习owner-local
projection没有消除深层held缺口，故不能把“把key换回raw X”恢复为canonical修正。score-supervised对照还显示约`.983` score
cosine对action-in可给`.991` factor，却对浅层q/v仅给`.125/.075`，证明高条件数native solve会把很小的anchor误差放大为大方向误差。

由此当前最早失效链按顺序为：P_lang-only task code近坍缩；small-residual joint interaction未实际工作；高维q/v的pointwise
candidate chart加高条件数solve无法产生跨video稳定functional anchor。F1 analytic operator、G1 free-code、G2 full Program、真实
X/Y banks、chunked replay、rank4和唯一rank16均不受该局部non-pass影响。下一版只能由同时针对这三个接口的机制证据驱动，不能再用
loss下降、非零gradient、score cosine或seed/LR/width小扫冒充修复。

### 51. 稳定functional target存在，旧G3同时丢失target-native方向与主selection路径

对40个mapping-fit tasks逐task/member只使用fit videos平均真实rank4 update，并在small core中截断回rank4；预注册held video从构造中
完全排除，只作只读验证。该consensus在held video上的overall update recovery median/p10/min为
`.945820/.935622/.908717`，q/v/action-in/action-out family均远高于F3 `.75`门；set-valued member upper bound与固定fit-only member
selection几乎相同。稳定的跨video functional target因此广泛存在，旧shared compiler约`.13`不能再解释为teacher本身互相矛盾。

同一fit authority的native input谱显示，family-shared q/v basis top128只能保留约`.7966`能量、达到90%分别需要约306/309维；按38个
固定LoRA target各自取basis时，target0/1 top128接近1，target18/19约`.966`，target34约`.976`、target37约`.959`。这证明压缩发生在
owner调制之前会系统性丢失深层方向；38-target basis后再family-share是有机制证据的parameter ownership，不是扫宽度。action-in完整
32D为1，继续作为正控制。

由此当前唯一replacement同时更改三个相互依赖接口：完整G2 `rank_event`进入selection query；真实native direction/log-magnitude先经
target-owned 128D basis再进入family trunk；旧dominant dot加tiny additive residual被单一normalized bilinear Q/K兼容度替换。训练只用
fit-video rank4 consensus的set-valued four-family paired-update direction；逐video input/output subspace不再等权牵引不同bank gauge。
真实K1/K4 smoke和world6一步profile证明full Program、target candidates、Q/K、owner query和group gain均获得finite/nonzero梯度，
Action Meta/source/Program/scale仍冻结，输出仍是唯一38-target rank16。该证据只证明新机制可优化，不等于F3 mapping已通过。

### 52. functional-anchor F3揭示common-residual shortcut与pointwise acquisition双重结构失效

clean pushed detached `main@3062de8`的full-Program/target-native primary bilinear replacement从fresh完成macro5/25 optimizer steps，
随后六个只读workers完整覆盖451 conditions。fit、held-video、task-holdout task median为
`.084298/.082754/.093856`，held p10 `.072027`、held/fit `.981684`；held q/v/action-in/action-out为
`.020707/.065711/.084290/.171636`。训练、同task新video和task holdout同样低，排除了泛化崩溃；F1 solve residual、feature retained
trace、Action Meta 0、唯一rank16和信息墙均正常。训练mean recovery从macro1 `.001057`升至macro5 `.077663`，但绝对F3 Gate仍差
近一个数量级，因此不续macro10。

fit-only update geometry暴露了此前Gate与teacher decomposition共同遗漏的shortcut。由40个fit tasks形成的一套task-independent
universal rank4，在held-video/task-holdout上的overall median/p10分别为`.825054/.682588`与`.835443/.658992`；meta-meta、
meta-target、target-target pairwise median分别为`.780820/.621810/.553756`。这意味着原F3 `.75/.50`口径本身可能被公共LoRA方向
通过，而无需Program/video因果。现有`carrier12 + universal4`可被rank12以`.998741` update cosine重拟合；再加task-difference rank4
对held原完整adapter的median/p10为`.956049/.946049`。但这只证明LoRA几何：直接从原native teacher减去universal后，task85 q/v
在真实bank中的analytic input recovery仅约`.828/.765`，current key task-local held recovery也只有`.199/.135`。因此新的carrier
假设必须从完整expert-minus-new-carrier重新做native projection/free-code capacity，不能把代数差分factor直接当新teacher或闭环成功。

四task frozen-checkpoint因果干预给出一致的ownership证据。same-task dynamic替换后的四family update cosine均约`.999`；完全
wrong-task Program后的q/v/action-in/action-out均值仍为`.973/.981/.992/.948`，recovery没有实质改变。相反，wrong-task native bank
使q/v/action-out降到`.863/.834/.569`，但action-in仍为`.99990`。所以full Program信号确实连到query，却在context/scorer中被压平，
当前输出主要跟随bank-common content而非task semantics。单task两fit-video factor正对照进一步显示，current frozen keys对held q/v
input subspace只到`.188/.177`，新学target-native pointwise projection也只有`.171/.130`，而原teacher direct native reference为
`.997/.997`；去掉common项没有修复该image。当前pointwise Program-query/candidate-key anchor即使有bank solve，也没有形成稳定的
functional canonicalizer。

最后，一次fit-only、零optimizer-step backward中global preclip norm为`21.51`而clip为1；约`99.88%`原始gradient energy位于
input/output candidate encoders与family trunks，Program context、queries、compatibility和topology参数只占极小部分。Adam的逐坐标
归一化意味着该比例不能单独证明某组实际step必然小，但它与checkpoint输出近task-independent、task-local capacity低值共同证明，
当前图把绝大部分学习负担放在从头建立38-target native坐标，而没有获得Program-conditioned selection。下一修正必须先分别通过
new-carrier native feasibility与set-conditioned canonicalizer/Program causality正对照，再fresh接受相同451 Gate；不再通过续训、
LR/seed/width扫或universal prototype制造表面pass。

关键artifacts：

- `runs/outputs/pi05_ecp_shared_compiler_g3_f3_functional_anchor_fold0_m5_gpu01p012345_r6_20260827/`；
- `runs/analysis/pi05_ecp_shared_compiler_g3_f3_functional_anchor_macro5_mapping_eval_3062de8_gpu01p012345_w6_20260827/`；
- `runs/analysis/pi05_ecp_g3_fit_consensus_geometry_v3_20260828.json`；
- `runs/analysis/pi05_ecp_g3_functional_anchor_causality_t{16,72,85,93}_3062de8_macro5_20260828.json`；
- `runs/analysis/pi05_ecp_g3_functional_anchor_input_content*_t85_j{18,19}_s200_factor_3062de8_macro5_20260828.json`；
- `runs/analysis/pi05_ecp_g3_functional_anchor_gradient_groups_t85_3062de8_macro5_20260828.json`。

### 53. F1与canonical B1之间隐藏着TF32数值断层，旧checkpoint不能post-hoc修复

继续改carrier或canonicalizer前，对同一真实native bank逐层拆分query storage、score matmul与pool reduction的精度。F1解析operator
一直用FP64，而source runtime会全局开启TF32；既有F0只比较chunked与one-chunk两条相同runtime路径，因而不能发现共同数值偏差。
对深层q target34、v target19和action-out target37的learned/teacher native anchors做全组合复核后，IEEE FP32相对FP64的
update-cosine最大绝对误差分别只有`2.7e-5/7.4e-5/1.9e-5`；TF32相对FP64的误差median却为
`.528/.675/.524`、maximum为`.817/.840/.704`。ridge `1e-6`的held learned-anchor recovery在q/v/action-out上由TF32
`.256/.178/.262`恢复到IEEE `.705/.798/.673`，且IEEE与FP64一致。根因是约`1e6`条件数的native dual在TF32约10-bit
mantissa下发生灾难性score cancellation，不是query存成FP32、softmax公式或FP32 reduction本身。

把已训练的`3062de8/macro5`直接切到IEEE或全FP64重放，12个代表条件均值仍为`.081656/.081651`，与原`.081656`不变；旧权重已在
错误forward/backward下没有取得强anchor，不能靠post-hoc高精度推理救回。这项证据因此既否定“继续用TF32”，也不允许宣称G3已经
通过：唯一有信息量的复评是保持架构、data、loss、rank、optimizer与451 Gate不变，从fresh用IEEE FP32训练。canonical
`main@78b7e58`在compiler forward开始关闭TF32并保持到backward，run contract与F0新增明确精度资格；4卡一步真实profile已证明
全部主路径gradient finite/nonzero、Action Meta 0，耗时`123.62s`、峰值约`25.65GB`。

随后clean pushed detached `78b7e58`的真实F0全部通过：`native_dual_uses_ieee_fp32=true`且compiler forward后
`allow_tf32=false`；chunk4相对one-chunk的38-target有效更新minimum cosine为`.99999955`、maximum relative error为
`.0009452`，K4置换误差`1.43e-6`且四条video权重严格均匀。全部Program/candidate/query/gain gradient finite/nonzero，
Action Meta 0、source/Program冻结、K4 teacher reads 0，最终76 tensors/38 targets的唯一rank16被policy实际消费。因此工程Gate
已经解封fresh IEEE F3，但这仍不是shared mapping科学Gate。

关键artifacts：

- `runs/analysis/pi05_ecp_g3_native_dual_precision_audit_abff0a7_20260828/`；
- `runs/analysis/pi05_ecp_g3_ieee_profile_step1_78b7e58_gpu01p1235_w4_20260828/`；
- `runs/analysis/pi05_ecp_shared_compiler_g3_f0_ieee_precision_78b7e58_gpu01p1_20260828.json`。

### 54. fresh IEEE未修复G3；actual-operator functional metric才是最早错配

clean pushed detached `main@78b7e58`从fresh完成macro5/25 steps及全部451-condition评估。fit/held-video/p10/task-holdout为
`.086508/.083131/.072629/.096191`，held/fit `.960958`；q/v/action-in/action-out held median为
`.021698/.065269/.085933/.173804`。因此IEEE修复了此前共同数值偏差，却没有让shared mapping获得绝对功能；fit、held-video和
task-holdout继续同量级，失败仍不是泛化、Action Meta、信息墙、chunk或评测实现。

真正的功能算子不是candidate anchor/key的普通Euclidean geometry，而是`J_r=C_r C_0^+ H`：`C_0`为B0 base native covariance，
`H`为event-normalized centered native/post-`Wk` key image，`C_r`为第`r`个rank实际B1 frame measure下的native covariance。旧路径先
单位化Program query，再在Euclidean key chart中优化；它没有控制经过高条件数B0 solve和rank-specific B1 replay后的方向与幅度。

同一真实task93/video2 bank的task-local解析见证显示，直接在actual functional image中选系数时，深层q target34、v target19、
action-in target36、action-out target37的update cosine约为`.996/.999/1.000/.998`。这证明现有真实X/Y、candidate content和post-`Wk`
keys有足够容量，旧`.02--.17`不是rank/video/G2或key内容的上限。进一步消融把实现选择收窄：per-event rank-specific polar对v/action
families接近1，但q约`.947`，必须跨event global；跨rank共享global polar的q约`.994`，v/action-out却降至`.915/.831`；不用per-event
feature whitening时q约`.911`。因此首版选择rank-specific global cross-event polar并继续保留feature whitening，不是扫width/LR/seed。

active v4将raw full-Program query经detached current-bank polar gauge变换，再用一个target/rank公共scale把B0 coefficient norm限制到
`1e-3`；native solve后又用一个target/rank公共gain把实际`C_r` score RMS定为`.02`，最后才执行两个softmax分支和真实X/Y加权。
bank statistics、polar和gain是当前deployment video在线产生的sufficient statistics，不是task/video table，不进入checkpoint；Q/K、
Program和candidate parameters仍经实际anchor路径学习。逻辑上仍是一个Pass B0规划加一个B1 replay，但B0内部需feature、functional、
anchor三次流式读取，总计四次native read；全部发生在rollout前唯一一次Writer调用中。

该task-local witness只证明修正有机制容量，不证明shared Program mapping已通过。下一资格仍是真实K1/K4 F0，随后fresh完整451-condition
F3；internal polar residual、solve cosine或loss下降不能代替`.75/.50/.8` Gate。旧`C=I`与Euclidean normalized-bilinear active开关删除，
其formal artifacts和Git历史保留。

### 55. functional-polar机制有解析依据，但当前per-condition执行形态被真实吞吐否决

在不启动formal的前提下，对task93/video的真实K1 condition逐段profile。首个可运行版本即使已经只capture一次frozen X/Y，仍需
`82.114s`：Pass A `5.283s`、native capture `5.622s`、compiler forward `59.595s`、backward `11.615s`；其中233个bank的polar
约`38.15s`。`da3fd3e`将同shape polar批处理、统计/solve改为IEEE FP32、矩形functional image改为thin-QR small SVD，并按解析见证
让q input及v/action使用per-event、只保留q output global polar。最终condition为`58.332s`、compiler forward `35.753s`、polar
约`14.15s`，peak allocated/reserved约`29.34/30.70GB`。全量cache coalescing与移除activation checkpoint分别在约
`43.36/44.37GB`真实OOM，已回退。

这次约`29%`端到端和`63%` polar降本仍不足以改变结论：即使假设polar与feature whitening免费，现有B0 anchor/solve、B1 replay和
其余stream编排仍约`14.4s` forward；再加Pass A/native capture/backward，不能靠小型kernel、dtype、batch或更多GPU压到与25-step
macro5相称的墙钟。正式schedule固定300个K1 condition，六卡理想训练下限约`49min`，完整451-condition Gate还需额外几十分钟。
卡数也不能改变每个rank两条video的重算临界路径。因此吞吐失败属于结构接口，不是GPU调度失败。

这项负结果只否决“把full 233-bank covariance/eigh/functional SVD/B0 solve放在每次deployment/shared forward中”的当前执行形态；
它不推翻task-local `.996/.999/1.000/.998` functional witness、真实X/Y、signed pooling、G1/G2、rank4或bank-global context的必要性。
当前没有formal F0/F3新结果，也没有新checkpoint。可能的后继包括把昂贵full operator降为fit-only teacher/诊断，再让轻量共享student
基于Program和当前bank低维summary生成dense signed measure并exact pool真实X/Y；但该方向尚未经过专家审查和容量/因果正控，不是
active design，更不能恢复旧fixed realizer、FactorHead或task/video lookup。

### 56. 当前源码保留的是可审查reference，不是另一个已获资格的Writer

`da3fd3e`没有添加新runner、策略flag或并行Writer，只在现有`functional_polar.py`数学owner、`shared_compiler.py`编排owner和
mapping condition loader中保存上述最佳执行形态；同一condition frozen X/Y cache是ephemeral、只用于mapping acquisition/evaluation，
不进checkpoint且不改变部署信息墙。deterministic leading blocks、batched global/per-event polar与矩形QR-SVD均有定向合同；全仓
`189 passed`。代码被保留是为了让远程专家能从完整源文件复核复杂度和降本边界，不能据此声称真实K1/K4 F0、吞吐Gate或F3已通过。

### 57. 第三次专家复核把根因收敛到task-specific credit ownership，并将low-dimensional sketch设为唯一下一接口

第三位专家锁定`main@9b52e59`及其可达历史后确认：F1与task-local functional-polar共同证明native bank、signed pooling和actual
functional image有容量；连续F3的fit/held/task-holdout一起处于`.08--.16`量级，说明失败不是先拟合再泛化，而是shared mapping在
训练条件自身也没有取得task-specific selection。fit-only universal rank4在held约`.825/.835`又证明旧absolute Gate可被公共修正
绕过。因此当前最早科学接口同时包含Program对selection的因果ownership、teacher/common residual分解与functional coordinate，不能
再把坐标修正单独当充分解。

专家建议full functional-polar永久只作fit-only teacher/reference；deployment先用current-bank native/key cross-image与一次sealed
nested projection构造`r_s<=64`的bank-adaptive native basis，只在小空间累计covariance/operator，再lift query并对真实X/Y做exact
signed pooling。先做50-task/98-condition无训练rank curve与真实吞吐Gate，再做12-task free-query正控和shared student，最后才恢复
451-condition。owner接受该方向，并补充：fit task universal必须leave-one-task-out；12 tasks中保留meta/target各一个true task-holdout；
首个student的projection/basis/statistics冻结；causal数值门在看shared结果前以固定正负controls一次校准。以上只改变G3 shared
acquisition路线，不推翻G1/G2、唯一rank16、信息墙或Final端到端fresh候选。

### 58. S1证明失败的是`r_s<=64` native-query bottleneck，不是full functional image或chunk实现

clean pushed detached `main@27bde62`在预先沿用的task93/q20 actual-operator witness上完成formal early disqualifier。两条sealed K1
videos、两个verified members的rank64 sketch-to-teacher effective-update只有`.156687--.157438`，input/output full-native
`C_rQ` free-query最低为`.413974/.253733`；同task/video/member/target的sealed F1 analytic operator仍为`.995560--.997907`，其
materialized replay相对analytic为`.999935--.999948`。streaming/materialized最低`.9999769`，source/G2/compiler trainable均为0，
Action Meta 0，故不能把差距解释为teacher、native bank、chunk边界、TF32或训练不足。

最初只解`Q^T C_r Q`会把小空间投影可解误报为full-native可解；改为仅需`O(d r_s)`的流式`C_rQ`并对真实factor做最优least-squares
后，上述低值仍存在。进一步dirty exploratory把fixed random `HR`替换为完整cross-image `H`的最佳top64，input/output各rank recovery
仍仅约`.415--.497/.264--.501`。这排除“随机projection seed抽坏”，把最早失效接口锁定为固定64维native query basis无法同时保留
真实B1 functional response。由于S1 Gate要求任一row至少`.95`，该正式反例足以否决合取条件而无需浪费资源完成其余96 conditions；
但不能用它估计50-task分布。

因此full polar与该native-Q sketch都永久降为fit-only teacher/diagnostic，不再往deployment增加polar、SVD、projection rank或seed
变体。专家预先规定的后继是pure low-dimensional set-summary student：Program或task-local free code与current-bank低维集合摘要、
candidate content/metadata直接产生正负candidate logits，最终仍exact pool真实X/Y。它必须先在12-task、两fit video到第三held video的
task-local free-query正控中证明overall/q/v/action容量，再允许shared Program training；该修正不推翻G1/G2或唯一rank16合同。

关键artifact：

- `runs/analysis/pi05_ecp_functional_sketch_s1_early_q20_27bde62_gpu01p2_20260828/`。

### 59. S2首轮fit失败来自随机candidate chart；free logits仍精确可达

clean pushed detached `main@4d84dee`在固定task93/q20、fit videos 18/48、zero-gradient held video0上完成1000-step set-summary
witness。fit effective-update median为`.328188`，held为`.175318`、held/fit`.5342`，held input/output subspace仅
`.100649/.042760`，全部Gate checks失败。训练loss从`.868`持续降至`.673`、梯度有限非零，total wall`296.52s`且训练
`7.72 step/s`；因此这是fit-bank selection acquisition的科学non-pass，不是运行、吞吐、Action Meta或held泛化先失败。

同一video18 bank的nested oracle把算子与score image分开：解析global free logits达到`.9999996`，按首轮eventwise归一化构造的free
logits也达到`.9999861`，证明真实X/Y、teacher、signed pooling及两种归一化都具有精确容量。相反，冻结formal candidate basis并允许
每条fit video独立直接优化低维coefficients、移除summary/code映射且加入强factor/subspace credit，update仍只有`.359/.353`；改成
canonical global reduction更只有`.048`。从fresh训练同一candidate basis时，eventwise bound8、由解析range推出的bound14及global
bound14分别只有`.233/.241/.203`，且最终logits均未接近bound。故bound8与eventwise reduction虽有合同差异，却不是当前数量级首因。

代码审计随后找到更早的authority错误：`prepare_frozen_native_bank_runtime`调用`_load_training_assets`后只按config seed新建compiler，
没有加载任何G3 checkpoint；`materialize_condition_banks`因此把fresh随机`NativeCandidateEncoder -> family trunk -> key projection`输出
当成“frozen existing candidate encoder”。q20 input stable rank为490，而随机chart只有128维；首轮正控实际检验的是随机降维组合，不能
淘汰fit-trained existing encoder、set summary本身或Program。clean detached `78b7e58/macro5`的formal F3 checkpoint含完整fit-trained
candidate encoder authority；下一最小修正只加载并冻结其中candidate encoder/trunks/metadata/key projections，其余函数类与Gate不变。

关键artifact：

- `runs/analysis/pi05_ecp_g3_set_summary_s2_witness_task93_q20_v1_gpu01p0_20260828/`。

### 60. fit-trained frozen candidate chart只带来小幅fit增益，最早接口转为chart acquisition

clean pushed detached `main@6b97100`显式加载`78b7e58/macro5`的609个candidate encoder/trunk/metadata/key-projection tensors
（8,006,400 parameters），排除旧query/Program路径，并保持task93/q20、videos 18/48/0、1000 steps、summary、score、loss与Gate
不变。formal fit median为`.349191`，仅比随机chart v1的`.328188`高约`.021`；held从`.175318`降至`.131624`，held/fit
`.37694`，held input/output`.112037/.038104`，五项Gate均失败。Action Meta、source/G2/compiler trainable、held outcome/gradient及
shuffled/reversed use均为0，训练`7.55 step/s`且梯度/loss正常，故这是科学non-pass。

结合相同bank的global/eventwise free logits都约1.0，当前可排除的是**冻结**`78b7e58`的128D score image足以支持该学生这一假设；
不能据此声称native bank、signed pooling、set summary或所有candidate encoder失败。该authority本身只在旧失败F3中训练25步，且本轮没有
让native-value chart接受当前task的factor/subspace credit。下一最早接口因此是candidate chart acquisition：先从同一权重初始化学生自有
value encoder/trunk/key projection做fit-only、无task/video lookup的可训练容量诊断。只有它显著恢复fit，才有依据正式解冻；若仍停留
在`.3`量级，则应替换candidate score函数类，不把bound、normalization、LR或额外summary宽度当作修正。

关键artifact：

- `runs/analysis/pi05_ecp_g3_set_summary_s2_witness_task93_q20_v2_gpu01p0_20260828/`。

### 61. 正确event measure后首版set-summary仍失容，简单近似逆也不能补救

解析dual artifact使用的真实固定测度是`rho_e * assignment_te * temporal_quadrature_t`。现有S2 runtime先把每个event的
`assignment * quadrature`各自归一成unit mass，再直接以uniform或raw `rho`混合，因而丢失了event的原始时间体积。task93/q20、
videos18/48、两个verified members的确定性重放显示：global固定测度仍为`.99556--.99791`，错误的uniform/raw-rho event混合只有
约`.043--.055`；把event权重改为`rho_e * pre-normalization event volume_e`并归一后恢复到`.9757--.9876`。这是真实执行合同错误；
以后所有eventwise B1都必须保留该体积，不能把unit-mass event bank再用raw `rho`直接平均。

修正该测度后，当前函数类并未恢复。冻结fit-trained chart、原bound8的1000-step matched run只有fit/held
`.31884/.04363`；bound0.1的independent/antithetic branches分别只有`.20855/.04639`和`.22339/.05198`；完整native H加逐channel
diagonal preconditioner只有`.05033/.03371`。最后把q20自身candidate chart也从`78b7e58`初始化后解冻，让所选target的
`363,520`个chart参数与scorer/free code共同接受factor/subspace credit；总trainable为`2,648,100`，1000步fit仍只有`.30286`，
zero-gradient held仅`.03527`。因此失败不再能归因于frozen/random chart、score bound、正负branch gauge、对角尺度或训练未接通；
首版128D mean/variance separable scalar-energy函数类按预注册Gate正式淘汰，不进入12-task/shared训练。

同一真实bank上的不训练operator curve解释了为什么小修补无效。q20 input在两条video的稳定秩为`483--487/1024`，八个output group为
`243--256/256`。isotropic-plus-low-rank correction在rank64/128时完整update仅约`.274--.280/.509--.511`，rank256也只有
`.832--.846`；input需要约384个bank-local方向才使output-exact update达到约`.983`，output需要约192--224个方向才接近
`.979--.996`。diagonal-plus-low-rank同样直到接近全维才恢复。普通和diagonal-preconditioned PCG在256次后input residual仍大于1，
完整update只有约`.48--.64`，且曲线明显失去共轭稳定性；它们不能作为廉价迭代替代。以上均为fit-only disposable诊断，不是S2 Gate，
但共同排除了“对角/少量rank/朴素PCG即可近似full inverse”这一实现方向。

当前最早接口因此是**query-specific cross-candidate geometry**：单个固定mean/variance summary与separable candidate basis无法表达随当前bank
旋转的高秩selection。下一函数类只能做一次有明确机制差异的task93/q20正控：让每个rank/event query直接、多步读取当前candidate set，
形成query-conditioned set state后再回到每个candidate产生signed logits；仍只pool真实X/Y，不含task/video/frame lookup，也不把full
covariance/eig/SVD或teacher tensor放进deployment。若这一更强的permutation-equivariant set operator仍不能先恢复fit与held，才根据
结果重开teacher/credit目标，而不是继续扫width、token、LR或bound。

### 62. query-conditioned多次读取仍未取得task-local方向，S2轻量score函数类关闭

按上述唯一分支完成了不保留的query-conditioned set operator诊断。每个rank/event/positive-negative query直接读取当前candidate set
三次，再用query-conditioned key逐candidate产生signed logits；input使用四个128维native heads，output使用两个128维native heads，
保留真实X/Y exact pooling、正确`rho * event-volume`、跨videos18/48共享free code、video0零梯度和Action Meta 0。该图共
`2,648,100`个trainable parameters，真实第一步约`.41s`、peak reserved约`22.9GiB`，所以失败不是运行未接通或吞吐不足。

强factor/subspace+update credit到step400的fit/held仅`.14678/.09718`；随后保持同一结构、数据、optimizer与scale，只删除专家明确不建议
长期使用的equal-subspace约束，改为set-valued paired effective-update加跨视频dispersion，step500也只有fit`.15209`、held`.09229`。
两种credit都没有数量级跃升，held也未随fit稳定提高。由此淘汰的是当前query-conditioned多次读取、candidate-local rescore这一明确
函数类；不能把失败归因于旧mean/variance summary、错误event measure、frozen chart或A/B子空间过约束，也不继续扫read次数、head、
width、LR、bound或seed。

由实测operator秩直接指定而非搜索的补充反事实也关闭了“只把S1 sketch做宽”这一解释。沿同一frozen native/key cross-image把nested
rank从64直接提高到224/384后，task93/q20两条videos、两个members的effective-update仍仅`.1593--.1630`；input/output linear recovery
约`.419--.427/.273--.293`，384相对224没有实质增益。单condition约`12.5s`，其中rank curve/replay约`2.0s`，peak reserved
`19.34GB`，Action Meta及所有authority trainable均为0。这说明该cross-image的有效方向本身不含teacher response；不是projection rank
不够，也不给恢复full polar deployment或继续projection版本链提供依据。

现有证据尚不能把稳定LoRA teacher直接宣布为错误：fit-only consensus在held video仍约`.946`，同bank exact functional operator仍约
`.996--.998`。但S2已证明“为了复现这一特定parameter-space teacher而设计的轻量score路径”没有容量或可优化性。下一步只做一个
更上游的behavior-aligned identifiability诊断：在授权fit task上，以cross-episode teacher action/flow或等价冻结functional gradient
作为credit，分别检查direct free logits和同一轻量selector能否形成跨视频稳定、实际降低policy loss的native rank4方向。若free logits
有效而轻量selector仍失败，最早接口仍是selection函数类；若轻量selector对behavior明显强于对teacher LoRA，则G3应把selection主监督
改为行为/functional equivalence，而不是继续追逐唯一参数分解；若两者都失败，则先处理behavior credit/optimization，不能直接发射shared
或完整451-condition F3。该诊断不使用validation/test action/reward，也不恢复旧Writer、realizer或full-polar deployment。

### 63. behavior credit确认根因是bank-specific dual坐标，global primal-to-dual恢复跨视频方向

授权fit task93/q20的定向诊断用与video demos完全跨episode的action demos 1--4，在唯一rank12+4 adapter下读取真实PI0.5 flow
gradient。carrier baseline loss为`.0991105`；dense gradient的canonical rank4 descent降到`.0880155`，反向则升到`.1148061`，top4
捕获gradient能量`.80575`。因此行为credit、符号、rank4与materialization均有效，不是parameter-teacher独有的伪方向。两条fit videos
18/48及全程零梯度held video0的optimistic真实bank signed update recovery分别为`.91037/.90435/.90243`，方向存在于每条bank。

同一behavior target下，旧2.648M query-conditioned三次读取selector训练500步后fit仅`.288/.298`且held坍塌到`.0229`；跨videos共享的
bank-independent full-native dual训练1000步后fit约`.231--.236`、held`.0745`。这共同证明Program或task code不能直接预测一个跨bank固定
dual：candidate covariance随video旋转，fit score会过拟合自己的dual坐标。

把稳定task intent改写为native primal `d`，再对每条当前bank的全局单位质量covariance做`q=C^+d`，并在完全相同的全局measure下以
`softmax(+q^Tv)-softmax(-q^Tv)`重放，未训练即在videos 18/48/0取得`.91122/.90437/.90055`，一步后仍为
`.91122/.90430/.90053`；input约`.824--.834`、output约`.982--.989`。三个q20 bank的全局inverse总计`.734s`，input retained rank
`471--487`，八个output groups约`239--256`。必须禁用TF32；允许TF32的同构run因近`1e6`条件数发生灾难性抵消，不构成科学反例。

最早接口因此明确为“共享Program预测跨视频稳定primal，当前video bank确定性提供dual坐标”，不是再造candidate scorer。global solve后
按event分别replay会施加`C_e C_global^+`并重新旋转，已经失败；event assignment改为只在Program/rank-event聚合中形成primal，native
solve/replay使用同一个全局时间quadrature measure。该结果是one-task/one-target nonformal机制证据，只选择下一P0实现，不证明38-target、
multi-family、shared Program mapping或G3 Gate。validation/test action/reward、shuffled/reversed与Action Meta均未使用。

### 64. v5真实P0通过并把下一失效接口收敛到multi-task primal容量

clean pushed detached `e2f9d33`的真实task93、38-target K1/K4 P0全部通过：Action Meta module/parameter 0、source/G2冻结、76 tensors
唯一rank16、真实policy consumption、uniform K4和全部primal/Program/event/scale梯度均成立。K4 permutation最大误差仅
`2.384185791015625e-07`；最终chunk4与one-chunk的raw/solve/conditioning error均为0，minimum update cosine为
`0.9999999999999998`。因此v5当前没有未解释的hook、bank边界、两pass、chunk、materialization或信息墙工程故障。

P0也给出一个可复用的数值边界：近`1e6`条件数下，B0 covariance与B1 signed replay都必须按固定candidate microblock归并，不能让
外部frame chunk改变浮点归并树；input/output microblock分别为400/1600 candidates，流式状态只缓存不足一个block的remainder。
这不是扩大模型或增加防御校验，而是让chunked算法与其reference拥有同一个数值定义。下一资格只在六个预注册fit tasks、多family/
深度targets上优化跨video稳定primal，held video保持零梯度；scale固定到fit-only authority并推迟到F4。若P1失败，首先按family、
input/output group和retained spectrum解释，不能修改shared Program scorer或用task93单点外推。

clean pushed detached `c9e8198`的P1正式结果把上述多task外推成立：六task fit/held median为`.971731/.954539`，held/fit
`.982308`，held相对各自optimistic projection `.992193`；q/v/action-in/action-out held medians均在`.9398--.9954`，最弱task
held仍为`.9350`。初始step0已经在`.9573--.9709`，500步只稳定提升约`.0047--.0082`，说明结果主要来自正确的task primal→
current-bank dual接口而非长时间拟合或偶然optimizer峰值。meta与target两role、浅中深与action均成立，held完全零梯度。

因此native bank、global covariance/solve、signed pooling、rank4、固定fit scale和跨video operator已经不再是G3当前首因；P2若fit低，
最早接口就是frozen full Program到task-specific primal的shared可识别性，不能再修改bank operator掩盖。P2若fit高而video/task held低，
才处理shared generalization；absolute高而correct-vs-wrong Program margin低则是universal shortcut。P1并未验证Program或deployment
Writer，不能把task-local code带入P2 forward/checkpoint。

### 65. P2必须隔离scale ceiling；compact frozen-bank replay保持同一数学

P2只裁决共享Program能否预测task-specific primal，不能让一个已知过弱的冻结scale把正确方向判成失败。原统一`.1*s_ref`即使直接使用
teacher方向，fit task-equal update cosine也只有median`.767177`、p10`.751008`、minimum`.742298`，几乎要求方向完美才可能跨过
held median`.75` Gate。由40个mapping-fit tasks、排除各自预注册held video的fit-consensus scales先按member、再按task等权做
coordinatewise median，得到唯一共享`[38,4]` ratio template后，fit解析ceiling提高到median`.997017`、p10`.974083`、minimum
`.964334`；held-video与task-holdout medians也为`.996952/.997577`。这不是task/video lookup或scale sweep：所有condition使用同一个
fit-only模板，scale head保持冻结，F4仍单独裁决scale学习。

重复冻结policy capture不是P2的科学变量。run-local cache只封存Program、raw X/Y、final Y和同一B0谱operator；输入候选仍没有output
type，四类output bank在replay时由每条视频自己的first/previous/final状态在线生成。真实task1的相同bank在原frame-chunk streaming与
compact replay之间最大误差`2.384185791015625e-07`，cache round-trip factor误差0，且IEEE scope调用后恢复进程原TF32状态。因此cache
只消除重复冻结计算，不改变signed attention measure、视频边界、梯度owner或deployment合同。独立重跑冻结BF16 policy时深层activation
可发生一个量化步差，不能用随机近零primal下的跨capture cosine冒充cache不一致；同bank reference和P1跨video有意义方向才是正确裁决。

### 66. frozen G2只通过动态Gate，未形成可迁移的policy-behavior坐标

按第三次专家复核预先规定的fallback，在不修改G3 operator的前提下完成95-task behavior sufficiency诊断。每个task用两组互不重叠、
与video跨episode的256-row flow-gradient panel，在q/v浅中深及action-in/out共8个native targets上形成rank4行为方向；75个fit tasks按
meta56/target19各占50%建立normalized behavior kernel PCA，meta-held15+target-held5全程零梯度且只在固定final probe读取。

该行为流形本身对未见task有充分容量：held20两panel repeatability role-equal median为`.6666`，fit75 rank16 span对独立panel-B与
consensus分别达到`.7160/.8006`，q/v/action-in/action-out对panel-B为`.7571/.8009/.5482/.7779`；task-independent universal只有
`.1908`，rank32相对rank16仅增加约`.0526`。因此rank16是有证据的首版容量，native行为目标、held噪声或rank ceiling不是当前首因。

随后从通过旧G2 Gate的`c1493a1/macro20`为全部95 tasks提取两个disjoint K4与一个K1 frozen Program view。四个匹配读出器固定训练
1000步，其中full/process/dynamic在fit75的coordinate cosine均达到约`.97--.98`，排除了读出器没有拟合训练条件；但held20的exact
rank4对panel-B仅为full`.2695`、process`.2470`、dynamic`.2368`，与learned language-only`.2687`同量级，远低于`.7160` span oracle。
full的q/v/action-in/action-out仅`.4335/.2873/.1583/.1886`，meta-held/target-held分别`.3684/.1707`。wrong-Program margin虽为
`.2054`，说明Program含task identity，却没有把未见task组织到正确policy-behavior方向。K1/K4与两套K4 view均同样低，故不是某一视频
cardinality或单次抽样。

最早失效接口由此从G3 scorer/operator上移到**G2跨task行为可识别性**：旧G2的`22.2047%` full-vs-endpoints动态Gate仍是有效正证据，
但只证明事件、进度与视频动态存在，不能再被解释为Program已足够支持shared policy adaptation。当前最小修正保持Program固定schema、
Stage0 v3、boundary-anchored alignment、uniform K、全部原动态loss与唯一执行路径；从`c1493a1/macro20`只加载model参数，以fresh optimizer
新增一个训练期process-only shared behavior decoder。它只读取`P_process/rho/tau/sigma`，以75 fit的role-balanced rank16坐标产生一项
behavior alignment loss；20 held坐标只进入预注册exact-rank4 Gate，不进入梯度或普通checkpoint选择。首版Gate要求overall/panel-B
`.50`、consensus`.60`、q/v各`.50`、action-in/out`.35/.50`、wrong margin`.10`、两种held role各`.45`，并保持K1/K4与跨video
robustness以及原G2全部动态Gate。

真实task74 K4 profile已验证该新增loss为`1.3104`，行为decoder与既有`process_fusion`梯度范数分别`.6286/4.4444`，旧temporal owner
gradient`.0805`，source/Native Stage0 trainable均为0、Action Meta module/parameter为0；一步墙钟`13.30s`、peak allocated约
`9.28GiB`。同一新decoder在旧frozen Program上的独立Gate smoke仍只有exact`.2121`、oracle`.7160`并正确non-pass，证明新Gate不是
被实现错误自动放行。当前因此先formal验证behavior-aligned G2，未恢复旧Writer/realizer，也不继续叠加G3 polar/SVD数学。

关键artifacts：

- `runs/analysis/pi05_ecp_g2_behavior_manifold95_rolebalanced_5781694_gpu01p6_20260829.json`；
- `runs/analysis/pi05_ecp_g2_behavior_sufficiency_probe75_20_5781694_gpu01p6_20260829/`；
- `runs/analysis/pi05_ecp_g2_behavior_sufficiency_program95_combined_5781694_20260829/manifest.json`。

### 67. pointwise behavior decoder改善fit loss却未改变Program拓扑，credit必须由部署表示直接拥有

clean detached `5cbe76e`的G2-B pointwise decoder已训练到macro60，不存在“一次non-pass就停止”或训练不足混杂。behavior loss从
`1.2723`单调降到`.7080`，旧动态full-vs-endpoints同时从`31.85%`升到`39.40%`；但是panel-B exact rank4只在
`.1837/.2622/.2938/.2828`间变化，最终consensus`.3027`、meta/target held`.3803/.1853`。这表明优化图与原动态职责都在工作，
只是目标没有转化成跨task的部署能力。

三组冻结反事实把原因收紧到credit ownership。macro60 Program上重新fresh拟合的full reader仍只有约`.262` task-holdout；fit-only
kernel/linear CV在fit task内几乎完美、到held仍约`.30`；而Program自身full behavior pairwise correlation只从旧`.1610`到`.1694`，
official held约0。decoder可以为每个训练task建立一套可读code并自己学习坐标变换，loss下降不要求Program中“行为相近的task彼此接近”。
因此继续增加reader容量、训练步数或普通超参只会强化同一shortcut；该负结果淘汰的是pointwise decoder supervision，不淘汰完整Program
schema、Stage0、rank16 behavior manifold或G3 current-bank operator。

原fit75内固定的role-stratified train60/internal-held15提供了不反复使用official held20的资格面。train60 rank16 basis在internal15对
panel-B/consensus为`.6184/.7158`，四family为`.6556/.7373/.4550/.6676`，universal overall仅`.0543`；所以内部holdout既有
足够上限，也不能由公共方向轻易通过。旧Program的固定block-equal完整feature基线中，train topology约`.186/.205`；internal meta只有
`.172/.183`，而四个internal target偶然高到`.763/.754`。这说明只看role-equal平均会被小target组误导，Gate必须同时要求两个role各自
达到`.25`。

当前有机制依据的单一修正是decoder-free behavior kernel。固定抽取`P_lang`、`P_scene`、`sqrt(rho)P_process`、
`sqrt(rho)sigma`、`rho`、`tau`六个等质量blocks，保留owner/event顺序；两组disjoint same-K views分别形成Program cosine Gram，
直接对齐train60的panel-A与consensus factor-cosine Gram，并约束跨view Gram一致。它没有新reader、task lookup或held target；梯度必须
改变部署Program本身的task topology。fixed kernel-ridge只负责internal Gate把Program邻域译回rank16 behavior coordinate，不进入训练图、
checkpoint或Writer。Stage0首轮继续冻结；只有fit topology明显上升而internal meta/target仍不升，才有证据重开窄grounding tail。

该修正的三卡真实一步已经证明distributed all-gather autograd、role pairing、两组video与language/scene/process梯度均接通；Action Meta、
source与Stage0 trainable为0。它仍只获得一次internal Gate资格，不能用训练kernel loss或旧动态分数冒充通过，official held20也不能用于
当前架构修正。

### 68. v3只学到batch-local behavior排序，需要连通的joint-role credit graph

clean detached `60fb18b`的v3 macro5训练和Gate从运行层面全部正常：15 updates的最后local A/B correlation为
`.7036/.7037`，direct behavior-kernel梯度持续非零，旧动态Gate仍以`13.945%` full-vs-endpoints改善通过。但Gate上
train60 A/B只`.2315/.2358`，相对旧block-equal基线`.186/.205`只小幅改善；internal meta仅`.2152/.2332`，
target的`.7842/.7930`与旧`.763/.754`同量级。exact panel-B/consensus role-equal仅`.1207/.1253`，所以不能用局部
train metric、偶然target小组或旧动态代替behavior Gate。official held20在整个formal中仍为0 reads。

根因是objective图而不是optimizer超参。v3每step在meta和target内部各自对5个tasks做centered kernel；预注册15个batches对
meta45只产生126/990条edges，且是5个不连通components。即使无限续训，该loss也没有直接约束components之间的相对
几何。这解释了为什么小批几何迅速改善而全量几何基本不动，也排除直接解冻Stage0、加reader或做LR/seed小扫。

behavior authority的meta-target cross-role panel-A与consensus关系相关`.8629`，是可用而非域噪声。在原5+5 batch中加入一个
等质量joint kernel后，不增加任何forward就把15个batches的总监督图变为483/1770 edges、minimum degree 9、唯一1个
60-task component。v4因此使用`.5 joint + .25 meta + .25 target`，保留两role等质量与原资源成本。三卡真实一步
已证明joint关系真正改变loss且梯度直接进Program，step `18.33s`、peak `9.98GB`、Action Meta/Stage0/source均保持冻结。

### 69. 连通pair graph仍不足；batch-local affine gauge允许Program近坍缩

v4 formal证明joint edges只是必要条件而非充分条件。虽然每批关系图已经跨role连通，macro5全量train60 topology仍只有
`.2360/.2362`，internal meta为`.2064/.2257`，exact panel-B/consensus为`.1129/.1177`，与v3没有实质改善；旧动态
Gate则以`14.6553%`增量继续通过。固定checkpoint的full Program跨task cosine集中在均值约`.965`、标准差约`.020`，而
teacher behavior cosine为`.145/.316`，说明部署Program几乎把所有task放在一个小球帽内。

根因在监督坐标而不是graph、LR或训练长度。逐batch双中心化会删除kernel常量偏移，Frobenius单位化又删除幅度；因此同一真实
behavior geometry可在不同batch中对应不同的`aK+b`，且接近公共向量的低方差Program在中心化后仍可取得较高局部相关。
这也解释了为何local batch correlation很高，却不能组成全局可读坐标。v5采用PSD且可实现的固定lift
`K_target=(1+K_behavior)/2`，直接拟合raw off-diagonal Gram，并只用完整train scope预先确定的teacher dispersion统一量纲。
它保留公共轴来表示正负behavior cosine，却不再允许当前batch自行选择均值或尺度；没有新增参数、decoder或信息路径。

三卡真实一步显示这个修正给出的behavior/Program梯度为`1.7323/2.7450`，并把初始Program std `.0141`与固定teacher
std `.1478`作为显式误差，而不是再次标准化掉。该结果只证明机制与优化面接通；若同一macro5 formal仍不能显著抬升全量
train60和internal meta，就应停止继续构造v6，向专家报告“现有固定Program字段/observer无法通过这种直接全局behavior credit形成
可迁移几何”的证据，而不是做seed、LR、width或训练长度小扫。

### 70. v5扩大了Program任务差异，却没有把差异对齐policy behavior

clean pushed detached `main@7f4df1b`的v5 macro5/15 updates运行、checkpoint和Gate全部正常，旧动态full-vs-endpoints改善
`20.8602%`并继续通过；source/Stage0冻结、Action Meta 0、official held20未读。但train60 topology A/B只有
`.2160/.2208`，internal meta`.2022/.2169`，exact panel-B/consensus role-equal`.1054/.1289`，wrong Program
margin`-.0466`。这些关键量均没有相对v3/v4产生数量级改善，且错误Program在多数family上不比正确Program差。

冻结block审计说明v5的global calibration不是完全无效：full Program cosine均值/标准差由v4约`.965/.020`变为`.926/.046`，
process由`.898/.086`变为`.750/.220`，所以raw Gram的绝对均值/幅度已经获得梯度；两套video的full/process cross-view仍约
`.970/.994`，排除了视频抽样不稳定。可是full/process对teacher consensus的相关从约`.150/.135`变为`.142/.131`，表明
新增spread主要沿错误方向增长。macro4--5 behavior alignment平均值约`12.6190/12.6196`，所有梯度有限且非零；这不是NaN、loader、
Action Meta或冻结错误，而是当前固定block-equal Program feature与直接pairwise geometry credit没有形成正确的task-specific
policy-behavior ownership。

因此v3--v5形成一条完整排除链：v3的role-local graph不连通；v4连通graph但batch-local affine gauge仍允许near-collapse；v5固定
绝对gauge后能拉开任务，却没有对齐behavior方向。该证据淘汰这三种objective及pointwise decoder，阻止继续G3 P2或用训练时长、
seed/LR/width/rank小扫掩盖；它不证明native bank、G1、current-bank primal-to-dual operator、Stage0或固定Program schema根本无解。
最早未解决接口仍是部署Program与shared primal scorer之间的联合behavior identifiability；不能只再增加一种Program几何归一化。

### 71. 第四次专家复核把独立behavior-Gram Gate改为joint functional credit

第四位专家锁定`main@910fb20`及`9b52e59..910fb20`历史后，确认P0/P1已经把native bank、rank4、current-bank dual和exact replay从
当前首因中排除；V5只有15次updates、前9次warmup，而且监督固定block-equal、单位化且等权的Program Gram。`rho/tau`还跨selected
owners重复，使不同target native kernels被迫共享大量几何。因此V5是有效protocol non-pass，却既不是公平的联合functional测试，也不
证明Program schema或Stage0结构性不可能。

当前最早未解决接口精确改为Natural Program与`ProgramNativePrimalScorer`之间的联合可识别性和functional credit。唯一下一实验联合
这两个相邻模块，用generated完整rank16 LoRA在cross-episode action/flow panel上的真实功能损失反传；source、Native Stage0、已通过P1的
bank operator、carrier、scale与Action Meta全部冻结。这样既不恢复frozen-Program P2，也不是提前解冻全Writer的黑箱训练。

12-task Gate必须先有同loss的task-local、跨fit-video共享free-primal正控，再同时报告train、held-video、true task-held、四family、
language/endpoints、wrong Program、wrong bank、interaction、same-task和相邻checkpoint。只有train与held-video强而task-held弱时，才做
matched raw frozen Stage0 sufficiency probe；raw相对Program task-held增加至少`.15`且达到`.40`才归因Program压缩，raw也低于`.25`
才允许把最早瓶颈上移到Stage0。原文逐字保存在`docs/expert_review_20260829_joint_program_primal.md`。

### 72. task-local functional正控排除了panel/scale不可达，shared Program仍待J2验证

10个预注册gradient tasks各自用同一task-local primal共享两条fit K1 videos、100次真实functional updates，第三video和disjoint
panel B严格零梯度。held/fit functional benefit retention median/min为`1.0144/.8896`，10/10 held panel-B均显著优于carrier；
held factor recovery median`.8078`，q/v/action-in/action-out medians为`.7973/.7722/.8436/.8481`。因此当前action/flow panel、
frozen scale、current-bank dual/replay与free-primal之间确有跨video稳定的功能下降方向；以后joint若fit低，不能再归因“teacher action
监督本身不工作”或“native operator没有可达行为方向”。

这不证明shared Program→primal成立。task8/75的factor recovery只有`.742/.749`，但functional held benefit分别为`.00335/.02005`
且显著为正，进一步说明factor cosine只能作几何诊断，不能覆盖functional primary。系统上physical microbatch4在长task93达到
`37.07GiB`；同logical16改physical2后为`32.41GiB`且step1 loss仅`.060%`相对差，所以这是规约/激活内存问题，不是科学结构问题。

### 73. J2充分训练仍学成公共残差；最早接口是task-specific functional routing

clean detached J2以10 warmup+100 effective updates联合训练Natural Program与shared primal scorer，step70/110 train recovery仅
`.1596/.1708`、held-video`.1487/.1646`；step110两个task-held为`.1228/-.1092`，四family均约零，wrong Program/bank margin
`.0080/.0071`、interaction`.0024`。held/train、same-task、event/K1、信息墙和相邻数值稳定成立，故这是结构性科学non-pass，不是
video overfit、Action Meta污染、工程断图或偶然checkpoint。

checkpoint110零optimizer-step审计显示Program/primal task-pair cosine median约`.93--.95`，generated update median`.678`且
action-in`.997`；但十task成功free-primal input/output code median只有`.203/.149`。同一panel的真实task gradient cosine median
`-.023`，`62.2%` task pairs为负，六task更新组的cancellation ratio为`.421--.536`；global gradient norm始终低于clip阈值且所有
有效Program/scorer组均有gradient。也就是说，相近条件表示被要求产生彼此接近正交的功能方向，而纯correct-pair objective允许一个
对所有task略有帮助的common residual。下一修正必须直接建立Program/bank与task functional effect的配对ownership，不能靠续训、
LR/seed/width/rank小扫或raw Stage0探针替代。

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
- G1 scalar、q-head、latest-only、set-valued、FP64 exact和action-in native-block step0依次为`88/250`、`84/250`、`100/250`、
  `111/250`、`116/250`、`114/250`；最后一项以breadth5/5、Goal/Long非零、4/5高于carrier和retention`35/43`正式通过G1。
  task94 action-in-only privileged response`118/250`只用于定位机制，最终pass来自真实native pooling而非该counterfactual。
- G1 free logits是held-task capacity upper bound；最终shared Program query到content key的attention仍只属于G3，不得从G1代码或结果
  推断deployment Writer已经成立。
- G2 boundary-anchored `c1493a1/macro20`以`22.2047%` held动态增量、median events 4、one-event 0和完整K/probe/same-task合同通过；
  它现在是G3唯一frozen Program authority。
- 旧Writer/realizer/ECP Stage 1已从活动树删除；后续只允许一个canonical Native-Factor implementation surface。
- formal checkpoints/raw rows保留在ignored `runs/`；精确旧代码用Git恢复。人工process路线与约11.6GB可重建主要产物已
  删除，recovery Gate A残留作为历史formal evidence保留，不恢复为当前数据或训练路线。
- 不新增checksum sidecar、重复证据JSON或一实验一文档；跨轮结论只更新本文件、`progress.md`和`research_history.md`。

### 74. J3只学会破坏负例，未形成正确task route

J3 step70/110的correct train/held-video recovery仅`.1369/.1316`与`.1486/.1477`，均没有超过J2；四family仍约零。
counterfactual训练并非断图：训练normalized gap持续增大，formal Gate中wrong Program/bank controls在8/10与7/10 tasks改善，interaction也从
J2的`.0024`升至`.0054`。但correct fit只有3/10改善，少数task8/32的错误臂大幅变坏抬高mean，median margin仍只有约`.01`。因此该机制
提供的是“这不是正确task”的稀疏排斥信号，scorer可以靠制造不对称坏negative降低hinge，却没有获得“正确task应选哪条native primal”的
稠密正routing坐标。最早接口仍是Natural Program表示与scorer函数类之间，不能再用margin/weight/LR/seed/width/rank小扫修饰。

下一对照固定真实bank/operator/functional loss，只用训练期正交routing token替代Natural Program内容并仅训练现有scorer。若train/held-video
达到`.60/.50`及四family门，说明scorer能利用清晰route，根因是Natural Program没有形成可分离task representation；若仍失败，则根因下移到
当前additive family-shared scorer无法把清晰route变成target-native primals，需要更强multiplicative/nonlinear head。该token表不是参数、
不是deployment input、不能通过G3，并在Gate解释后退出活动树。

### 75. 清晰route有效但不充分；functional-only没有从随机scorer发现四族强方向

R1用10个固定正交task token替代Natural Program内容后，step110 train/held-video由J2的`.1708/.1646`提高到
`.2678/.2798`，10 tasks中9个fit改善，wrong-token margin达到`.2384`且same-task retention`.9910`。因此Natural Program原来的近公共
表示确实是一个真实瓶颈，不能再声称route完全未被使用。但q/v/action-in/action-out仍只有`.0037/.0078/.0011/.0333`，所以清晰
route远不足以让现有functional-only scorer达到`.60/.50` Gate。

checkpoint几何把“route”与“方向发现”分开：不同token进入scorer后，四family hidden cross-task cosine只有`.18--.27`；q/v预测对10个
task-local成功code的正确task检索为8--9/10，说明任务身份没有再次坍缩。然而预测与自身成功code的coupled primal cosine仅
q`.0015`、v`.0051`、action-in`.0085`、action-out`.0675`。各family step70到110参数更新量相近，排除简单断图或某个head完全不动。
固定hidden的最优last-head线性解可将全部input以及v/action-out output拟合到`1.0`，但q八组和action-in三十二组output在当前共享128D
group feature下上限约`.658/.363`：函数类对这两族确有结构限制，但它也没有用上v/action-out已经具备的表示容量。

更关键的可比性修正来自J2正控初始化。`TaskLocalPrimalCode`并非随机初始化，而是先从fit-only teacher consensus构造方向；在第一个
functional optimizer step时，它已拥有最终收益的中位约`.431`。因此J2正控证明“好方向可跨video并被functional loss精修”，不证明
“functional loss能从随机shared scorer发现好方向”。R1从随机scorer只学到task-specific action-out shortcut，符合高维双因子发现credit
稀疏而不是route/gradient缺失。下一最小对照应在同一fixed-route/scorer图上加入已有set-valued paired-update critic；若四family恢复，
瓶颈是direction discovery credit，若仍失败才根据q/action-in grouped-output证据替换primal decoder。该critic只在fit训练期使用，不能
成为deployment输入或G3通过捷径。

### 76. R2证明critic有效，但group-shared output decoder只能恢复半套解

R2 step110的q/v/action-in/action-out family recovery为`.2205/.4076/.1668/.6635`，相对R1近零是数量级提升；训练日志中的critic
recovery也由0升至约`.322`并在最后20步平台。因此fit-only set-valued critic不是no-op，继续增加weight、LR或步数不再是有机制的新实验。
但真实train/held recovery只有`.2058/.1936`，低于R1的`.2678/.2798`，wrong-token margin也由`.2384`降至`.0906`。部分teacher几何
恢复会挤掉R1的action-out功能捷径，却没有形成完整强LoRA，内部指标不能替代functional primary。

最早剩余接口由解析capacity确定。保持R2的task hidden与10个成功free-primal code不变，当前一个owner所有output groups共享head时，
q/action-in最优median只有`.691/.392`；让每个native group拥有独立head后，四family median和minimum都为`1.0`，所有group的hidden
rank均为`40/40`。这不是width扫，而是移除已证实的错误参数共享：q仍8组、action-in仍32个native-width blocks，candidate measure、
真实X/Y、signed pooling和唯一rank16完全不变。下一R3只检验这一变量；只有四family高而functional仍低，才把根因进一步判为
set-valued teacher-to-utility不充分。

### 77. R3排除decoder后，首因收敛为从随机shared scorer发现强方向

R3用owner×native-group独立output heads清除了R2的解析容量限制。step110 action-in/out recovery达到`.6562/.6689`，证明decoder修正
有效；但q/v只有`.2853/.2773`，train/held-video只有`.3053/.2875`。六task真实gradient分解显示，旧fit-only critic相对functional
gradient的全局cosine median为`-.1489`，q为`-.2696`；成功task-local code的直接监督梯度也仅与functional有`.0324`中位cosine。
所以提高q/v权重、增加critic、把code loss长期并入或继续普通超参调整都没有依据。

冻结scale并非根因。把十task中六个成功code原方向经同一current bank回放、只换成R3 shared scale后，真实policy functional recovery
中位`.9398`，范围`.7534--1.0232`且6/6超过`.60`。这与P0/P1、owner×group解析容量一起把最早接口锁到direction discovery：强
双因子方向在现有图中存在，scale足够，但从随机shared scorer出发的functional或teacher credit都没有把优化带入该basin。

R4据此只做一次确定性边界对照：十个fit-only、functionally validated task-local primal通过现有fixed routing token hidden插值到同一套
shared owner×group heads；不使用task-local scale，初始化后删除critic，只保留真实functional loss。真实step0六task相对原code recovery
为`.980/1.155/1.012/.944/.754/.815`，中位约`.962`，证明FP32最大`.00399`的head插值误差没有破坏policy功能。R4若在相邻
checkpoints保持强解，才允许把utility-aligned初始化/短warmup机制接回Natural Program；R4本身含training-only task route与privileged
初始化，永远不是deployment或G3通过。

### 78. R4把强functional解保留到`.82/.84`，唯一剩余action-in失败来自moving feature chart

R4 step110的train/held-video recovery达到`.819437/.839139`，q/v/action-out为`.439578/.388131/.400750`，wrong-token margin
`.913637`、same-task retention`1.002751`；11项primary checks只有action-in`.249310<.30`。因此functional-code初始化确实把shared
scorer送入强功能basin，R4 non-pass不能解释为bank、operator、rank4、scale、route、video overfit或functional loss整体失效。

action-in的最早接口由三组相互支持的graft证据锁定。第一，step0/70/110 raw outer recovery约`.999988/.298330/.301140`，但固定
各checkpoint当前hidden、只FP64重拟合33个action-in heads即可恢复到`1.0`；40个hidden rows始终full rank，minimum singular ratio约
`.0025`。第二，initial到step110的action-in/all-head relative drift只有`.000784/.000810`，feature chart为`.008930`；checkpoint chart
加initial heads仍是`.301099`，initial chart加checkpoint heads则为`.998320`。第三，program context、rank context、input trunk或output
trunk任一从checkpoint单独移入initial graph都把outer降到约`.22--.32`，而只移动embedding/event模块几乎不影响。故根因是
minimum-norm heads所依赖的高条件数feature坐标在整条链中distributed co-adapt，不是某个head更新过大或单层bug。

R5据此只冻结functional-code初始化后的完整feature chart，并让38 input加195 output native heads继续接受相同functional loss；不改
数据、rank、scale、bank/operator、seed、LR、预算或Gate。首个真实step确认trainable仅`10,297,344` head参数，全部233 heads有finite
nonzero gradient，frozen chart无gradient，Action Meta/source/Stage0/scale均0，native teacher reads 0且仍生成唯一完整rank16。R5若
通过也仍只是training-only fixed-route边界；它证明的是utility-aligned head初始化必须绑定稳定chart，不证明Natural Program shared route。

### 79. R5正式闭合moving-chart根因；最小G3接回是Natural Program加native heads、固定功能chart

R5 step70/110的train recovery为`.933583/.940336`，held-video为`.957202/.963277`；step110 q/v/action-in/action-out为
`.815834/.839439/.820583/.837113`，wrong-token margin`.895772`、same-task retention`1.006591`。两个checkpoint全部primary
checks通过，step110相邻稳定性也通过。与R4仅action-in`.249310`失败相比，唯一改变就是冻结初始化后的feature chart，因此R4缺口的
因果解释已得到正式Gate支持：不是heads、bank、scale、rank或functional loss失效，而是高条件数minimum-norm heads依赖的内部坐标移动。

R5不允许直接冒充G3，因为它仍由training-only authority ID选择fixed orthogonal token。接回真实Writer的最小机制不是恢复fresh scorer、
counterfactual或更多几何loss，而是加载R5已通过的**共享**scorer权重，删除fixed-token forward，恢复G2 Natural Program并保持chart冻结。
R5已经证明233 native heads在固定chart上接受functional gradient不会破坏强坐标，因此R6让Natural Program与这些heads共同训练；这保留
专家要求的Program--scorer真实functional credit，同时消除J2/R4实证的moving-coordinate gauge。loss只保留generated rank16的跨episode
PI0.5 flow，source/Stage0/operator/carrier/scale/Action Meta继续冻结。R5 checkpoint不含task lookup参数，fixed token既不是参数也不会
被R6加载；tasks2/74与全部held controls仍零梯度。R6只有通过完整12-task train/held/task-held/family/causal/stability Gate才是G3资格。

### 80. R6证明passed fixed-token chart没有Natural Program内容几何；下一接口是固定heads下的chart acquisition

R6 step110 train/held-video只有`.165181/.143114`，task-held mean为`-.034333`，除action-out`.319160`外q/v/action-in均近零；
wrong Program/bank与interaction也只有`.077886/.001131/.000018`。这不是同task video不稳：在共同R5/R6 mapping下，同task不同video
输出cosine约`.9994`。决定性坐标对照是，R5 fixed token通过R5 heads的functional-code cosine为`.998514`，G2 Natural Program通过
同一heads只有`.010736`，R6最终Program通过R5或R6 heads也仅`.020074/.020914`。因此R5证明的是稳定、utility-aligned的固定token
codebook，并没有证明deployment-visible Program已位于该chart中；R6的functional gradient不足以从错误坐标直接旋转到该codebook。

对实际G2 Program hidden用两fit-video做minimum-norm head solve可把80/80训练行精确插值，但第三held video code cosine仅`.353777`，
task2/74仍约零；简单“重新拟合heads”会利用高条件数nullspace记住fit views，不是共享映射解。下一R7因而冻结R5已通过的233 heads，
只让Natural Program及其共享feature chart用10个gradient task的validated task-level positive-control outer-update direction获取内容坐标。
该label仅训练期使用、同task两video共享且held零梯度；它是隔离最早接口的mapping acquisition，不是deployment输入，也不能靠内部loss
通过G3。R7之后仍须用真实native banks、current-bank dual、exact signed replay和唯一rank16执行完整12-task functional Gate。

### 81. R7排除“冻结功能chart后只训练Program侧即可取得内容坐标”

R7在同一10个gradient tasks上使用validated task-local positive-control outer-update direction作dense、四family等权训练标签，冻结
R5已通过的38 input与195 output native heads，只训练Natural Program readers/fusion/aligner及共享feature chart。clean detached
`024fc55`的110步训练将内部family direction提高到约`.64--.74`，但loss在约`.337`平台；完整Gate的step70/110 train recovery仅
`-.183186/-.133386`、held-video仅`-.177017/-.129792`，全部target-role gradient tasks仍为负。task2保持约`.55`而task74约
`-.58`，wrong-bank与interaction仍约零，因此不能以少数meta task改善或内部cosine宣称成功。

该结果只淘汰“Natural Program必须迁就冻结R5任意chart”的具体函数类。R5 fixed route经同一heads仍有约`.9985` code cosine和强闭环，
R7 fit/held video同步，历史shared-scale transfer又已排除scale ceiling，所以当前没有依据修改bank、rank4、scale、视频采样或延长训练。
下一最小机制检验是在**同一绝对functional-code label持续锚定输出**时联合训练Program与完整primal scorer。它不同于R4/R6只靠弱
functional loss造成的moving-coordinate：当heads移动时，outer-update target仍固定，不能通过共同坐标漂移逃避。若该联合函数类连
gradient-task fit都不能显著越过R7平台，才把问题上移；若fit/held-video强而task-held低，才触发matched raw Stage0 probe。

### 82. R9稳定功能chart初始化同时改善fit与零梯度task-held，取得完整functional Gate资格

fresh Program+fresh scorer的R8直接outer-code joint在110步只到约`.449`方向；绕过learned process fusion的raw-process版本约`.456`，
加入contextual language、scene transition和raw process的完整raw Stage0版本约`.452`。contextual language/scene在12 task三video上本身
可稳定区分task，三种matched结果却几乎相同，因此当前不能把首因归给Program字段遗漏、Stage0压缩或普通视频噪声。

只把scorer初始化改为R5已通过的共享functional chart、其后仍让Program与完整scorer联合训练，R9在同预算把训练任务直接outer
direction提高到约`.712`，最终loss`.334220`；相同实现的retained world6 step1逐值复现，全部Program/scorer路径有finite nonzero
gradient。该变化证明随机Program/scorer双线性坐标的共同发现是R8/R7的一项真实瓶颈，而不是数学形式本身没有容量。

决定性的任务外只读检查没有重拟合head：最终R9 Writer对零梯度task2/task74三条video的正确outer recovery分别约`.731/.550`，
总中位`.640`，same-task跨video约`.9998`。这说明稳定chart引入的是可迁移共享先验，不只是10-task codebook记忆；同时task74对task2
label反而更近，correct-vs-wrong总margin`-.0139`，表明task-specific ownership尚未由内部几何完全证明。故R9已经越过“不要formal”的
机制门槛，但不能由内部方向宣布G3通过；下一唯一裁决是原12-task cross-episode panel-B functional Gate，包括train、held-video、
true task-held、四family、language/endpoints、wrong Program/bank、interaction、same-task与相邻checkpoint。

### 83. R9把随机chart瓶颈与code-to-utility失败正式分离

R9 clean formal完整复现了诊断的outer-code获取：step70/110 acquisition loss为`.354164/.334220`，step110内部q/v/action-in/
action-out medians为`.728694/.744085/.745741/.642526`，四family均过内部门且same-task视频方向稳定。这确认R5稳定共享chart是有效
初始化先验，随机Program/scorer双线性坐标共同发现确实是R7/R8的一个真实优化瓶颈。

真实policy却给出相反的充分性结论。step70/110 train recovery仅`-.181514/-.131825`，held-video仅`-.175532/-.129718`，
task-held mean仅`-.009468/-.011724`；五个target-role gradient tasks在两个checkpoint全部为负，task74也约`-.59`。step110
full-over-endpoints、wrong-bank margin与interaction分别为`-.082736/-.003829/.001946`。因此较高outer cosine不是功能等价LoRA，
也不能替代cross-episode action/flow credit；继续延长outer-code训练或雕琢相似度没有机制依据。

下一最小修正不是回到fresh functional优化。R4/R5已正式证明functional loss会破坏高条件数feature chart，而固定chart时native heads可
稳定保持强解；R9又首次取得与Natural Program共同形成的内容坐标。因此R10加载R9 step110 Program/scorer model tensors、冻结feature
chart、移除outer-code loss，只让Program与native heads接受真实functional flow。它检验“稳定内容初始化能否把弱functional gradient
送入正确basin”；若train仍低，才说明当前Program/heads函数类或functional credit本身不足；若train/held高而task-held低，才触发
active design的matched raw Stage0 sufficiency分支。

### 84. R10证明稳定内容初始化可转化为真实效用，并把剩余瓶颈收窄到target-task泛化与Program--bank交互

R10从R9 step110完整Writer tensors初始化，冻结feature chart，移除outer-code loss，只让Natural Program与233 native heads接受真实
cross-episode panel-A PI0.5 flow。clean detached formal连续完成110步；step70/110 train recovery为`.532227/.559896`，held-video为
`.500728/.544189`，相对R9的`-.181514/-.131825`与`-.175532/-.129718`发生数量级翻转。四family、same-task retention、
full-over-language和wrong-Program margin均通过，证明R9 content basin加真实functional credit确实抓住了主根因，不能再把当前问题描述为
“Program/scorer无法优化”或“内部方向完全不能转化为policy utility”。

R10仍不是G3 pass。step110 task-held mean仅`.151475`，meta2为`.375386`而target74为`-.072436`；五个target gradient tasks也只有
`.185780--.499314`，显著弱于meta tasks。wrong-bank margin`.007864`、interaction`-.002683`、full-over-endpoints`.061382`，说明
模型主要由Program决定一套功能residual，却几乎不要求该Program与正确当前video bank配对。R5在相同target tasks上的fixed-route正控均约
`.84--1.01`，且P0/P1/operator/family Gate都已通过，所以该role差距不能归因于native bank、rank4、scale或replay容量；续训和小型
超参扫描也没有机制依据。

最新专家预先规定，只有“fit与same-task held明显可学而true task-held低”之后才能做matched raw Stage0 probe。R10恰好提供该证据，
尽管train median`.560`未越过`.60`硬线：held已过`.50`、所有family已过且target tasks由全负转为多数显著正，足以把probe作为诊断而非
救分版本。下一实验保持相同scorer容量、functional loss、数据、预算、bank/operator/rank/scale和Gate，只交换Natural Program压缩与
部署可见raw frozen Stage0 evidence。raw task-held只有比R10高至少`.15`且达到`.40`，才能把首因判给Program schema；raw fit高但held低
指向task diversity/shared decoder归纳偏置，raw task-held低于`.25`才支持frozen Stage0为上游瓶颈。

### 85. R11排除Program压缩首因，并暴露target-role与v-family的选择性失效

R11保持R10的12-task split、R9初始化、functional loss、110步预算、scorer容量、bank/operator/rank/scale与Gate，只把Natural
Program压缩替换为部署可见raw frozen Stage0 evidence。clean detached step70/110 train recovery为`.218691/.292321`，held-video为
`.232166/.288053`，true task-held为`-.139011/-.092369`。step110 task2/74分别为`.116054/-.300793`；相对R10 task-held
`.151475`下降`.243844`，没有达到“提高`.15`且绝对`.40`”的schema判据。因此不能停止当前Program schema，也不能重新设计schema
来解释R10的剩余失败。

R10与R11的110步task、action demo/frame、panel visit完全逐值配对。R11最后40步对五个target tasks的functional loss全部更高，task72/73
在全部66次出现中`0/66`优于R10；target gradient train median仅`.110012`，而meta为`.470816`。family分解更有选择性：q、action-in、
action-out仍为`.550257/.494693/.474379`，只有v降到`.101550`；R10对应四项为`.645745/.717575/.548006/.614858`。
这既不是所有raw字段都不可读，也不满足“q/v/action均缺少信息”的frozen-Stage0停止条件。R11 language gradient约为R10的`3.99x`，
native input/output head gradient约为`1.13x/.92x`，无断图或clip；更多同预算优化没有机制依据。

R10/R11另有共同结构证据：wrong-bank margin分别仅`.007864/-.003253`，interaction均约零，而wrong-Program明显有影响。当前
primal-to-dual solve会让每条高覆盖bank近似重建同一Program primal，跨task bank可能因此退化为通用实现基底。下一步先用R5/task-local
成功primal做cross-task wrong-bank functional upper-bound；若成功primal换错bank仍保留效用，则Gate失败首先属于operator的bank交互
可识别性，不能继续归咎Program；若正控有强bank margin，才保留shared decoder/target generalization为首因。

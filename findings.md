# EMBER findings

只记录跨session仍影响决策的结论。专家原文见`docs/expert_review_20260824_native_factor.md`与
`docs/expert_review_20260826_bank_conditioned_native_factor.md`，精确分数、提交和历史脉络见
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

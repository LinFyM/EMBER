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

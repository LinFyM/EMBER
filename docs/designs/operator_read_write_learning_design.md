# 同一实际LoRA算子的教学写入与执行读取

2026-09-28。候选，未验证；科学论证见机制分析§42。只有progress登记的阶段有执行许可。
本文件§1–5定义完整方法，§6工程已完成、§7为独立验收；当前§8仅授权CPU正式运行面转换，**不授权正式学习或held评测**。
它替换旧Core/Procedure/FactorHeads生成图，不在已关闭P/I、条件速度或LocalField上追加补丁。

## 1. 为什么投入，以及不凭什么投入

假设：由执行动作信用共同学习的实际LoRA读取坐标，可以同时作为教学中状态条件的写入地址；
观察变化提供待学习的作用Value，按地址逐步写成矩阵，使同一个条件—作用联系直接参加自身状态下的FM。
关键干预是**实际执行A、教学key投影、教学读取所用公共策略共用学习参数**，不是给encoder再接独立几何头。

历史约束：LocalField已有逐位置`U r X^T`和同一参数的真cotangent/FM，但X为裸source冻结坐标；
ProcessPullback已有差分Value、递归记忆、source Jacobian、PCA及共享L/R；条件速度已有共同状态基和普通FM，
151→102未保持；P/I已有自由公共128和全层条件16，124→108未保持。上述功能不冒称本次首创。
本候选改变的是完整读写约束及其信用传递，且采用原有跨episode数据；不继续用人工迁移A替代原B来保护旧解释。

这是一项有限学习偏置假设：它没有新增任务信息，也不证明原生特征已经有对象角色/接触语义。
公共常量rank128可达、递归的非扩张性及同一A都不保证学成、跨任务泛化或相邻保持。
未知允许有界检验；目前不声称已找到576下降根因。

## 2. 唯一完整前向

### 2.1 输入、公共策略与原生读取

K=1，exact language和完整同步双RGB，stride5含真实末帧；不声称Dynamic-K。
source固定为既有aligned step1000，normalization/tokenizer/资产/官方执行规则不变。
训练覆盖未来仍用已审coverage36允许集合；本工程仅用§6四个明确train任务。
Writer不读teacher action/state/reward/terminal/pose/ID/文件名、query或policy outcome；不建新数据。

每个38-target m具有fresh公共参数`A_m[128,d_in]`、`B0_m[d_out,128]`，alpha=rank=128。
A沿canonical identity初始化，B0=0；没有预训练MT底座/分段冻结。source物理参数trainable数始终0。
公共策略β在每次教学读取中实际以`W_source+B0 A`执行；不是读回本次已生成M，β不在一次编译内更新。
公共β也进入最终policy，接受同一普通FM。它兼任本候选Action读取适配，无另一个Text/VL/Action Meta副本。
图文prefix本身冻结：真实双相机patch+exact语言；禁止zero-image、伪prefix或teacher state补全。

同一public `N(0,1)` probe，CPU seed1729、shape50×32，所有帧/条件共用；flow time=1。
通过现有原生prefix/KV和完整denoise suffix取得每个target的真实输入`X_tm[d_in,50]`，
以及最终action_out之前完整`H_t[1024,50]`。动作内50位置保留到下述逐位置learned read，不能先平均。
prefix只读embedding可无梯度；β→X/H必须保留梯度，禁止detach为裸source字段或跨更新缓存。
这仍是state-free教学；执行查询含自己的state/noise/time，两者的hidden不被宣称相同。

### 2.2 同一A读取地址、变化Value、矩阵递归

令`hbar_t`为H_t每列的无可训练参数RMS归一化（epsilon1e-6），`dh_t=hbar_(t+1)-hbar_t`。
每个实际target独有四个无bias线性映射：P:128→256，C:1024→256，D:1024→256，O:256→d_out。
P/C/D采用标准fan-in初始化，O精确零；构造顺序按canonical target顺序，module seed7。
这些是训练共享参数，不是task字典；用统一owner处理38个shape，不能另建每层trainer或重复模块。

对t=0…T−2，全部50列同时计算：

```
K_tm[:,h] = A_m X_tm[:,h] / max(norm(A_m X_tm[:,h]), 1e-6)
V_tm = O_m( GELU(P_m K_tm + C_m hbar_t) * (D_m dh_t) )
M_0m = 0                         # [d_out,128], FP32
M_(t+1)m = M_tm + (V_tm - M_tm K_tm) K_tm^T / 50
```

`*`为逐元素乘；没有Value bias/static shortcut。O零使全部M初始精确0，且B0=0，整套初值为source identity。
只对真实相邻帧产生write，最后真实帧参加dh；不按成功/时长删帧、重排或构造动作标签。
key列单位化只用于写入；实际执行读取是未单位化的A h，不能称为对称cosine检索。
递归步长固定1，50为真实horizon数，非额外调参。M是一次Writer前向的隐藏状态，不是optimizer参数。
没有部署loss、autograd、动作标签、policy梯度、适应步数选择或临时task adapter试运行；
此递推虽代数上等同一个线性关联误差的delta update，但不把它冒称真实任务优化器/学习到的真动作更新。
一次编译只流式读授权视频并输出，执行中M/A不再更新、不再读取教学。

每个target最终只返回`A_final=A_m, B_final=B0_m+M_(T−1)m`，共76因子，唯一rank128 LoRA。
不是两个同rank因子相加后引入交叉项，也不部署两adapter。M=0退化为完整自由公共rank128。
conditional更新被约束在共同A行空间；Value输出还受共享O的列空间约束，这是真实表达代价，不以“完整38”掩盖。
action_in/浅层X可能不含图像，不能假称其key已经识别角色；V仍来自完整原生H变化。

### 2.3 有益控制解释及其条件

只有当自身hidden与教学hidden中的相应角色/进度能投到相容地址时，`M(V) A h_query`才会给自身状态正确作用。
例如“杯仍在桌面”和“杯已放下”必须由可见对象进度在自身hidden中区分；没有观察区分的历史状态，静态LoRA不能补造记忆。
视频H变化可能携带物体随夹爪移动、目标区域占据等关系；它也包含视角、教师走法和噪声，当前均非已识别语义。
FM通过实际Q/V/输入输出投影教会V的符号与幅度，V不是直接把像素位移当机械臂动作。
相对MT的预期来自新任务上教学提供的条件作用关系，而不是复制某条参考路线；若只得到共同能力，则方法目标未达。

## 3. 真实学习与信用

用原授权同task跨episode RGB/action query，HDF5既有offset1，query自身双RGB/8state和真实50×7动作。
普通FM：`z_tau=(1-tau)a+tau epsilon, y=epsilon-a`，完整50位置，既有独立Gaussian/Beta(1.5,1)时钟；
无端点前5、teacher自身功能aux、真cotangent标签、几何aux、KD、RL、回滚或迁移构造query。
本候选全部参数fresh，以同一FM共同更新，source冻结。沿用完整LoRA cotangent/VJP回放，禁止只训一个独立读头。

若G_m=dL/dM_T，递归的实际反传（把K/V看成图上节点）为：

```
R_t = I - K_t K_t^T / 50
G_(t-1) = G_t R_t
dL/dV_t = G_t K_t / 50
dL/dK_t = [G_t^T V_t - (G_t^T M_(t-1) + M_(t-1)^T G_t) K_t] / 50
```

K/V网络、单位化及β→X/H继续链式回传。实际weight信用`G_W=sum(delta h_query^T)`给A直接读取项
`(B0+M)^T G_W`，又通过教学key和公共β的native读取给同一个A信用；B0也同时接受执行与教学路径。
不能遗漏“公共参数改变teacher表示”的项，也不能拿矩阵公式代替真实非线性policy验证。
单位key保证R谱范数≤1，仅限制一条视频中已有矩阵的传播；新Value可增大输出，Adam也会改所有坐标，
所以它不证明动作稳定或跨checkpoint保持。重复相近key可抹去前段有用作用，这是必须接受的失败模式。

## 4. 最接近完整历史与单个科学干预

| 历史 | 已有机制 | 本候选的实质变化/保留反例 |
| --- | --- | --- |
| 50dafb05 LocalField | 同一U/r为真C标签与LoRA/FM消费者，BA=mean(UrX^T)，X裸source冻结 | 本次实际A同时决定key/执行read，β共同改变教学X；不是首次同消费者/逐位置绑定。旧失败降低单凭绑定理由的信心 |
| ProcessPullback与共享L/R版 | 差分Value、递归memory、裸source Jacobian/PCA16、共享出口、真FM | 本次隐藏矩阵就是最终B的一部分，无物理Jacobian/PCA转译；不声称第一次动态Value或可微写出 |
| PNBTT | Cholesky query whitening、真实native Value transport；停在free-query，natural未启动 | 不复活expert bank/任务token/局部Gate；现为原生RGB、公共实际A与端到端FM。未试natural不是收益依据 |
| 条件速度270/450 | fresh共同rank128、自身反馈U、视频R、普通FM | 不在末层添更多出口挽救旧法；替换整个条件生成图，共用实际A并写全层输出；151→102的保持反例继续约束 |
| P/I288/576 | 原v5.2读取/Compiler，全层自由公共128+条件16，跨初态对应 | 不改原Writer配对权重；不用迁移数据，独立读取Meta/自由因子头被完整读写图替代；124→108不被淡化 |

未来完整比较候选T与U：T按§2共用A；U只把教学key中的A替换为独立S_m[128,d_in]，初始化逐值clone A，
其余公共β教学读取、V/M递归、最终A、数据/query/噪声/optimizer/评测不变。S只在U接受自己的梯度；
其它参数初始化不受多一次随机抽样干扰。U参数更多，这是一项共享约束干预，非严格等参数消融。
两臂首个非零更新前输出与初始函数相同；没有detach教师特征或削弱对照。工程只验证实际计算，不能选T为赢家。

解释分支：T相对U稳定有益且达到强参照，再检验正确视频必要增量；T≈U且都较强只支持整体经验收益，
不证明共享关键；T只胜U但仍弱于强MT不算成功；两者低或邻点大面积丢失，关闭本完整假设窗口，
不补rank/η/门控/语义标签/保持正则/冻结公共项来维护它。不得在结果后改变归因。
真正视频证据仍需冻结单checkpoint后的same-task-other、合法语言/static参照及最终controls；不在工程读取controls。

正式样本量、节点和成本须由本工程实测后另行冻结。每个选择节点仍是严格single-checkpoint paired400，
保留per-task/suite/breadth/RGL/churn/相邻重合；历史MT155仅标量参照，不能伪称同物理scene严格配对。
后继须安排公平强MT参考和能力保持，不故意弱化baseline；当前没有400、M长训或新scene许可。

## 5. 工程所有权与实现边界

复用canonical source/processor、RawTeacherVideoStore/FunctionalQueryDataset、真实FM/VJP、ECP、official捕获与评测。
单个私有`src/ember/operator_writer/`包，至多native读取、model递归、run调度三个实际owner及必要__init__；
一个config、一个定向测试文件。只保留一个入口，T/U只是同一算子的参数共享开关，不复制trainer。
公共813行旧trainer不扩写；不导入退役专用CLI，不恢复P/I或LocalField模块。原生读取以函数式β替换和scoped hooks实现，
可复用现有prefix与完整suffix；不得把新正确性寄托在复制一套source/model或永久monkey patch上。
临时绑定在前向/回放范围正确生效，公共β与执行policy物理source隔离；不能遗留上一个condition的生成adapter。
新源预期约800–1100行，若超出先作owner/lifecycle自审；允许内聚例外但不能靠拆碎文件躲guard。
候选关闭时删除私有训练/物化运行面，Git/冻结树/原件保存科学证据；仅有真实official历史消费者才保留sealed读取。

## 6. 当前有界工程合同

唯一执行者01a0dd6c-f2e5-7971-821a-56766e1c0f22，任务`operator_read_write_engineering_20260928`。
独占既有dev，基于派发的已推送main，分支可沿既有codex/demonstration-transfer；主讨论独占main/docs。
完成源码/CPU检查后从clean pushed commit建独立detached冻结树，既有冻结树不得改。
这不是formal性能实验；在以下固定范围内允许GPU工程，无须等另一轮确认。

**时间/资源**：CPU实现和检查预计60–90分钟，GPU预计15–30分钟wall，全批预计75–120分钟；
150分钟为重审/实质阻碍回报线。估计依据最近完整FM/ECP工程0.21–0.51GPUh；新矩阵读写尚未profile，不能当实测速度。
完整GPU硬限1.0GPUh，含加载/保存/失败/重算/等待，峰值同节点2卡；不能为了凑卡等待或占卡。
先按实际trainable参数和ECP数核算，原件峰值≤6GiB；新冻结代码≤300MiB、开发及相关冻结树合计≤3GiB。
root=`/data1/user/ymdai/ember_runs/operator_read_write_engineering_20260928`。所有新增data1；
创建前查strg01独立quota+共享容量，每次GPU前live查双节点，UUID/NUMA/world2/NCCL_P2P_DISABLE=1写launch contract。

**固定实现/事件**：工程每步4task×28query=112，tasks [2,29,38,97]逐步等权；必须由coverage authority核实均为train。
4步各task teacher依次demo0/1/2/3，query从其余demo0–49均匀固定seed采样，28条均跨episode、实际HDF5 offset1。
事件seed20260928、module/optimization seed7，两臂query/noise/time与task分配相同。
优化AdamW lr3e-4、betas(.9,.95)、eps1e-8、wd1e-4、clip1、150 warmup/1200 decay/floor1e-5；
沿用现有绝对时钟，不因只有4步缩短warmup。数值、model、source和完整事件写一份机器规格，不自动生成长训计划。

**GPU仅以下完整序列**：T fresh4、U fresh4，各保存ECP2/4；仅T从其ECP2→4验证一次完整恢复。
共10实际宏步/1120真实query，工程权重不能作未来正式初值。world2、物理拓扑同T恢复保持；
micro28/framechunk8，OOM时仅按14/7、4缩小且记账，不减查询/视频或改数学。
最长合法已知train视频global38/demo36（517原帧/105采样帧）在T4作一次28-query完整FM/VJP，0更新；
若authority元数据不符先回报，不改挑更短视频。记录compile/FM/replay、samples/s与allocated/reserved峰值，
用4步真实曝光与完整video长度元数据给后继成本范围，不启动后继。
最后一次顺序接口：先compile global2/demo46而不跑环境，再compile global38/demo46；
核第二次确实从公共β读取、不受前一次最终M污染，只对global38/state0跑一条官方train-only canonical full双RGB，horizon520。
可用通用单adapter/official入口；不为1条case建设bank400平台，不增加Source/MT病例。成败原样保留，只作接口证据。

**CPU/原件验收**：模型公式的轻量线性反例/梯度检查覆盖共用A、M读写、单BA无交叉项和无bias零变化Value，
不是只测试shape。验证source trainable0、首步O与B0学习/后续A/P/C/D/native读取信用，U的S独立消费者；
递归重计算不得detach跨帧信用。恢复保存Writer/common/optimizer/scheduler/sampler/rank RNG/拓扑/schema，
日志前缀与事件/LR/游标连续，不追逐bitwise。唯一case动作T、body/EEF/gripper/谓词T+1、双RGB实际时点；
工程总成本/失败尝试/退出码/资源终值保留。没有held动作/像素、环境新样本采集、长训、正式400、RL或controls。
工程严重违约可以在同一预算修正可复核实现并留失败原件；不得据smoke闭环好坏调模型或增加例子。

完成或实质阻塞仅一次Queue回主讨论，附源码commit/原件绝对路径、实际计费与未验证边界；不自启后继。

## 7. 工程独立验收（2026-09-28）

执行实现`9766cb8bc624239abcc9a53ce11b6d5341dfc49f`，主讨论已审native/model/run完整相关图、两个失败修订、
原始metrics/ECP/唯一病例和计费。六文件+1012行，其中源码839；三个内聚owner符合§5，guard REVIEW无hard。
run649行和case94行为本次工程例外；正式转换须删除已完成工程CLI，不长期保留两套运行面。
主讨论独立CPU 3/3通过；因成功冻结树为稀疏checkout，测试从同commit完整dev读取、模块从冻结src导入。
一次初始测试路径不存在未执行测试；读回脚本对metrics_rows整数与state批维的两处假设已纠正，均非工程缺陷。

T/U fresh4和T2→4合计10实际更新/1120query，五个完整ECP的优化器、scheduler、sampler和两rank RNG游标成立；
T恢复原两行原文前缀、后两步事件/LR成立。两臂同任务/teacher/query/flow，28个query demo各异且均非teacher。
首步O/B0活动、其它预定零；第二步起A/P/C/D及native X/H cotangent活动，U的独立S活动，source物理trainable0。
**这里只证明实际图接通，aggregate梯度没有分离公共β教学路径的贡献，更未测得地址或Value的控制语义。**

最长517→105帧、28query/0update整condition31.065秒（事件内30.984秒），compile7.439/FM2.472/replay21.073；
峰allocated27.58/reserved32.54GiB。四步宏时长T33.33–35.73/U30.72–34.17秒/112query。
顺序compile先安装2/demo46的非零LoRA（delta .02804），第二次38/demo46教学读取前恢复source identity，偏离精确0；
公共β仍由函数式图装入，不能把物理identity误读为教学忽略公共β。唯一38/init0病例520步失败、104次重规划，
NPZ 520动作/521状态及PT真实动作前缀/双RGB已核；只作接口事实，不触发性能补丁。

d566cf83/c9887a2a两次失败均零已应用更新，只修retain_grad条件与梯度统计generator；约56/112次失败FM尝试单列。
七次完整占卡合计**0.3334568577GPUh**，峰2卡；所有成功计算9766cb8b，五成功进程exit0，失败账保留。
run root约2.30GiB，data1 quota终值525.6GiB/1TiB、共享83TiB，按执行时原件解释，不冒充未来live状态。
原件root：`/data1/user/ymdai/ember_runs/operator_read_write_engineering_20260928`。
主讨论CPU读回：`.codex/tmp/operator_read_write_engineering_20260928/main_acceptance.py`及同名JSON。
尚无正式训练或held性能；本验收没有运行policy/环境/GPU或重跑工程。

## 8. 完整比较的预注册与当前CPU转换合同

### 8.1 科学干预、节点和投入裁决

本节固定首批的完整比较；**当前仅授权CPU转换，不授权GPU或新formal root**。CPU实现独立验收后，主讨论再登记
精确main冻结commit及整批GPU合同；不要求再做一轮smoke/接口病例。§2模型、§3普通FM和优化数值全部不变，
T/U唯一区别仍是教学key共享A或独立S；两臂fresh，工程权重只作历史原件。

首节点每臂270更新/30,240 query，每步4task×28，36task各30次访问。只在270各correct400一次；
90/180/270保存ECP，90/180不物化、不评测、不选点。工程四任务刻意包含长视频，不能直接当完整任务池的平均成本。
270越过150步warmup，已有条件速度同曝光节点曾有完整能力，是一个有信息量的有限窗口，非收敛或上限证明。
唯一预留邻点450（新增各180步、累计50,400 query、各task50次访问）；本次CPU不开放450训练，不能自动续训。

强参照B采用既有coverage36 MT-BC step300的原权重，**不重训、不换弱点、不改变因子/精度**，只新增一次同scene400：
`/data0/user/ymdai/ember_runs/coverage_retraining_20260920/training/mtbc/checkpoints/step_00000300/lora.safetensors`。
来源run_contract、checkpoint_manifest和既有official contract只读；source是同一aligned step1000、唯一共享38-target rank128，
无teacher输入。旧155/400保留为历史标量，新同scene分数单独记；不把旧行伪称配对。此次改变的只有新比较的共同物理场景，
不是恢复旧训练或重跑旧实验合同。MT原300步实际172,800 query，不称与T/U样本预算相同；它是强能力参照，因果干预只认T/U。

在看本批结果前固定以下**本轮投入/选择规则**，不冒称统计定理或移植旧145硬门槛：

- 记`B_ref=max(155,本次同scene MT成功数)`。若T/U270最佳达到`B_ref−15`且该臂breadth≥6/8，允许主讨论登记
  唯一450节点并保持两臂共同续训；否则关闭本次完整学习窗口，不因T胜弱U、loss好看或工程已投入而续训。
  15/400=3.75pp是为“接近强参照才值得检验相邻节点”预留的资源容差，非无效/等效置信界。
- 450后可保留候选须：450严格超过B_ref；同臂从270到450净下降不超过8/400；保留至少80%的270成功集合；
  breadth不下降且≥6/8。三项保持指标分别限制总量回退、能力交换和任务坍缩，不以总分增长遮蔽大面积丢失。
  这些是本轮可证伪的能力保持选择，不代表所有正常随机交换的普遍上限。
- 不摘270峰值、不接630、不扫seed/rank/η/LR、不加门控/aux/保持正则或冻结公共分支挽救本假设。
  若两臂均合格，选450正确成功较高者，平手选T；只有一臂合格就选该臂。所有选择只用登记correct节点。
- 两臂都强不证明共享A必要；U胜或T/U接近削弱绑定优势。T在两节点均较好也只支持本seed的共享约束收益，
  要同时报告成对任务bootstrap及不确定性，不将其唯一归于语义对齐。若只U稳定强，应保留整体读写经验、否定强绑定解释。
- 选定并冻结单checkpoint之后，才另登合法语言/static参照、same-task-other及最终视频controls，验证实际教学增量。
  本批T/U都用视频，不能从T−U推断视频必要性；shuffled/reversed、Test、RL均不进入当前选择或设计。

首批报告每臂400原行、逐task/suite/breadth、T−U/T−MT/U−MT的R/G/L/churn/Jaccard和完整成功集合；
以8-task cluster bootstrap10000、seed2026092810报告差额区间，明确不含重新训练的不确定性。
450若获得授权，用同一映射/scene计算所有同点和相邻比较；不择优重跑物理失败或不完整episode。

### 8.2 实际训练/评测数据流

训练唯一任务集合固定为
`[0,1,2,4,5,7,12,13,14,15,17,19,20,21,22,25,28,29,32,34,35,36,37,38,42,43,51,55,56,62,64,73,95,96,97,101]`。
全部由coverage_v1 authority核为train；沿既有24目标train+12明确meta授权，不引入迁移query/新标签。
每9个宏步完整遍历36task：访问轮v的任务排列使用`SeedSequence([20260928,0,v])`；
每task的50个teacher排列使用`SeedSequence([20260928,1,task])`，访问v取第v个；270只用前30，不能按结果换teacher。
query使用`SeedSequence([20260928,2,task,v])`，从其它49demo无放回取28个，再各取合法frame `[0,length−2]`；
HDF5 action offset1、own双RGB/8state/50动作，flow_seed仍用现有task_logical_batch_policy_rng_seed(seed7,task,v,queries)。
完整事件T/U相同；按真实帧数平衡两rank只改变执行位置，不改逻辑权重、query和noise/time。一个condition权重1/4。

唯一formal规格由engineering_spec改名为learning_spec；model/native的数值及源码计算保持，source/normalization/tokenizer不换。
同节点world2、micro28/framechunk8、NCCL_P2P_DISABLE=1/GPU-local NUMA。正式初次物理GPU现场选择，无须沿用工程3/7；
同臂故障恢复锁该臂原world2/UUID/NUMA，完整optimizer/scheduler/sampler/rank RNG与连续metrics前缀恢复到最新登记ECP。
初始及同臂恢复都校验固定Git/source/model/数据/优化数值；之后main的文档前进不能使冻结身份失效。
只有OOM可降framechunk8→4、micro28→14→7；必须保留每query实际FM随机流和完整112query/update语义，记录packing变更。
若既有FM/恢复owner不能满足这些条件，应回报具体阻碍，不静默豁免校验或重置Adam。零剩余游标只完成ECP/pointer发布，不再更新。

validation固定 `[3,6,11,16,23,26,31,39]×state0–49`，seed7/K1/correct，沿video_schedule的400个task-video条件，
每task50条video无放回、T/U完全相同；held只读取authority和合法双RGB，绝不构造held FunctionalQueryDataset。
使用已封存`/data1/user/ymdai/ember_runs/demonstration_transfer_learning_20260927/scenes`的400共同起点，
不新建/修改scene，不恢复任何P/I模型。三臂首动作前均通过同一official full-scene restore和双RGB核验。
每臂8 full（每task state0）、392 compact，保存真实T动作/T+1 body/EEF/gripper/BDDL、实际重规划双RGB；MT使用同等证据口径。
复用canonical evaluator、long-first动态队列、persistent workers及BatchedLoRAInference，每卡2 replicas，协议不变。

T/U bank每臂只存一次38个公共A、每condition保存最终38个B0+M，FP32不另降dtype，重建唯一rank128 LoRA。
不存多套LoRA平均、不部署公共+条件两个adapter。MT引用旧单一完整rank128文件一次，无视频条件或重新物化其权重。
bank/capture/episode准入在一个内聚owner；MT是固定外部参照的只读分支，不开放MT训练/任意外部checkpoint路由。
MT来源须核旧run/step300/source/目标shape/normalization及任务数据边界，不靠修改旧authority或删除不匹配字段通过。
既有hash身份合同按原owner保留，但不额外建立hash/sidecar或全模型扫描。

### 8.3 成本与渐进限额

成本脚本/结果：`.codex/tmp/operator_read_write_engineering_20260928/formal_cost_plan.*`，仅读authority长度和工程原timing。
36task首270视频采样帧均值32.86、p95=58、max105；完整1800源池max105，held400长度均值36.32/max91。
以每臂16个实测condition的compile/replay每帧中位数、28query FM中位数和真实双rank分配外推，T/U270训练分别
3.62/3.35 GPUh；不是完整分布profile。训练保守规划7–9 GPUh，bank约0.53的线性估计规划0.6–1.0；
三臂1200官方episode由最近P/I完整成本约1.13 GPUh/400外推3.4–4.5。首批完整预期**12–15 GPUh、6–8小时wall**，
峰2物理卡（训练/物化/评测统一计数），加载/保存/失败/恢复计入；正式硬限拟为18 GPUh（train11、bank1.5、eval5.5）。
若实际资源/吞吐不能在此范围完成，按完整批次有意义阻碍回报，不按时间心跳，不先加卡/加预算或抽少评测。

每condition最终B为21,774,336 bytes、共享A为19,415,040 bytes，两臂800条件bank约16.26GiB；
三点ECP两臂约2.7GiB、1200行捕获按既有产物规划8GiB及余量，首批formal root峰值32GiB。
拟root=`/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928/stage1`；当前不得创建。
新冻结采用已验证稀疏代码树≤32MiB，开发及相关冻结代码总额≤3GiB；全部新写data1。
formal实际launch前由执行者重新核strg01 data1独立quota、共享容量及现场双节点/GPU/UUID/NUMA，不能把§7终值当live准入。
450若达到投入线，额外训练仅已有外推T/U2.41/2.22 GPUh，届时按首批实测重新预算；本节不预支执行权。

### 8.4 当前唯一可执行任务：CPU正式运行面转换

任务`operator_read_write_formal_transition_20260928`；唯一执行者01a0dd6c-f2e5-7971-821a-56766e1c0f22。
基于派发时已推送main，在原独占dev/codex分支完成；主讨论独占main/docs。
预计60–90分钟，120分钟为实质阻碍回报线；**GPU预算0，不创建formal root/新冻结树、不运行policy或环境**。

工作：把run转换为上述唯一270训练入口、完整36task事件与ECP恢复，去掉工程四步/profile/case活动CLI；
新增一个内聚bank/official接入owner，支持唯一T/U完整bank和严格封存MT只读引用、400 scene复用及配对捕获；
仅在独立职责已形成时提取data/事件owner，不复制trainer/scene平台或通过碎片拆分躲guard。
工程model/native文件原则上不改；若调用整理确需改动，逐项说明为什么不改变§2/3的计算，禁止混入科学修改。
共享evaluator只作必要路由；旧P/I/条件速度sealed读取、Source/MT-BC既有能力不得破坏。

CPU交付须核270完整事件36task各30访问/1080条件/30,240query，T/U同teacher/query/flow，合法offset与teacher排除；
source/operator/optimization数值与9766cb8逐项相同。测试覆盖实际新合同的fresh/恢复/错臂或工程来源拒绝、唯一LoRA组合、
held不构造query、MT来源和scene登记；复用工程通过的模型测试，不复制只测实现形状的新测试矩阵。
允许读取小型旧MT因子/headers和ECP元数据核复用，禁止载入完整policy或扫描原400权重/held动作/state。
对新增来源/边界不够确定就精确回报；不为避免一次CPU回报绕过原合同。

按code-architecture-gate自审本次净增长、owners和生命周期；预计保留源码净增加≤500行，超过需解释实际职责，不机械拆碎。
提交并推送隔离分支，完成或实质阻碍只给主讨论一条Queue；附精确commit、差异、CPU证据、预算边界及后继命令模板。
这条完成消息触发主讨论审阅/集成和精确GPU合同；不要自启270/450、bank、held episode或另一次smoke。

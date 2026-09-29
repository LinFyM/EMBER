# 同一实际LoRA算子的教学写入与执行读取

2026-09-28。候选，未验证；科学论证见机制分析§42。只有progress登记的阶段有执行许可。
本文件§1–5定义完整方法，§6–18的工程、训练及诊断均已完成；当前冻结训练任务闭环诊断见§19。
§11的节点/预算不自动延续；模型、信息墙与评测约束继承，精确冻结与派发以progress为准。
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

2026-09-28 Owner澄清：当前性能优先、不限制Writer参数量；MT历史rank128不是本候选输出rank的要求。
读写公式可用一般r成立，128并非数学必需值。本候选A跨task共享，故所有条件的更新行空间均包含于row(A)；
旧逐条件A(C)/B(C)的rank16不要求所有任务共用同一个16维行空间。这个结构差别提供较大r的容量考虑，
但没有证据证明实际任务需要128，也没有证明16不足。M=0可表示同rank MT常量解是所选模型类的性质，
不能独自成为必须选128的理由。T/U固定同rank检验共享约束，不能证明128优于其它rank；冻结运行数值保持。

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

## 8. 完整比较的预注册与已完成CPU转换合同

### 8.1 科学干预、节点和投入裁决

本节固定首批的完整比较；登记当时仅授权CPU转换。该阶段现已由§9验收，正式执行以§10和progress为准。主讨论登记
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

Owner后续强调的判断边界见§10.6：以下首点数值线保留为原预注册资源依据，不再将未过线自动解释为放弃整个架构。
本批270/三臂400与预算不变；任何新增训练仍须有明确科学依据和另行范围，不能自行越过当前执行合同。

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
初始估计采用峰2物理卡，现由§10.4撤销该额外并发上限；加载/保存/失败/恢复计入，完整硬限18 GPUh（train11、bank1.5、eval5.5）保持。
按§10.4及仓库规则选择有吞吐收益的物理资源；若预计仍无法在预算内完成，按完整批次有意义阻碍回报，不抽少评测。

每condition最终B为21,774,336 bytes、共享A为19,415,040 bytes，两臂800条件bank约16.26GiB；
三点ECP两臂约2.7GiB、1200行捕获按既有产物规划8GiB及余量，首批formal root峰值32GiB。
拟root=`/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928/stage1`；当前不得创建。
新冻结采用已验证稀疏代码树≤32MiB，开发及相关冻结代码总额≤3GiB；全部新写data1。
formal实际launch前由执行者重新核strg01 data1独立quota、共享容量及现场双节点/GPU/UUID/NUMA，不能把§7终值当live准入。
450若达到投入线，额外训练仅已有外推T/U2.41/2.22 GPUh，届时按首批实测重新预算；本节不预支执行权。

### 8.4 CPU正式运行面转换合同（已完成）

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


## 9. CPU转换独立验收（2026-09-28）

01242f6c5a4cf5c95fd1edca7cc3853f4b98c0a9已独立审阅并集成。model/native零改动，source/operator/optimization三个完整字典
与9766工程逐项相同；旧engineering_spec及四步/profile/case活动入口删除，没有第二trainer或GPU重验。
完整事件270宏步/1080条件/30,240 query/臂、36task各30个不同teacher，实际teacher来自0–49的固定排列。
`teacher_pool`元数据中的0–29表示访问序号，不能解释成只用原demo0–29；实际event和审计保存真实demo。
恢复只接本臂latest完整90/180/270及固定Git/source/数值/world2 UUID/NUMA，完整前缀与ECP owner承接；零剩余节点可正确发布。

主讨论独立新/原operator CPU 8/8（14.16秒）、封存P/I bank/scene回归4/4（11.28秒），diff检查通过。
400 scene/teacher调度的metadata与官方实际任务顺序一致；held实例不创建query dataset。
旧MT step300源路径、source1000、36train/offset1、76因子shape及原BF16/F32分布成立。
旧临时normalization路径已不存在；主讨论按原训练3ebb979b与原评测bee2d5e8的Git快照读回JSON，与当前5759-byte文件完全相同，
没有重建旧runtime、改旧authority或读取held动作。旧source-base-config SHA的历史差异仍明示保留，不称文件身份完全相同。

实际src净+592（预算参考500以上92），来源为data事件96行、bank553行、run649→516及共享路由增量。
guard原始结果为4个hard信号，**不是全部无hard**：episode adapter复杂度32、passive capture分派31、preparation800→807、
_prepared_payload120→123。主讨论接受这四个窄增量的内聚例外：它们为已有集中分派接入当前bank/scene，拆出另一evaluator反而形成平行owner；
新data/bank无hard，run缩小。关闭候选时移除训练/物化私有路径，只按实际消费者保留sealed读取；不把例外扩成通用框架。

验收/guard原件：`.codex/tmp/operator_read_write_engineering_20260928/formal_transition_main_acceptance.*`与
`formal_transition_main_guard.json`。主讨论未执行policy forward/backward、环境/GPU、400物化或formal root创建。
CPU完成回报已消费；后继科学假设、唯一因果干预和§8.1停止线均不变。

## 10. 首批正式执行合同

任务`operator_read_write_learning_stage1_20260928`，唯一执行者01a0dd6c-f2e5-7971-821a-56766e1c0f22。
本节是Owner持续授权下主讨论作出的整批执行裁决，**允许在下列固定范围内完成GPU工作**；不需要再次等待许可。
主讨论先提交并推送本合同与机器预算，在progress和派发消息写精确40位commit；执行者只从该commit的clean pushed detached
稀疏冻结树运行，建议路径`/data1/user/ymdai/projects/EMBER-operator-stage1-formal`，不改旧树/main/原件。
后续main文档推进不改变这批冻结身份，origin/main只须包含该commit。CPU开发树不作计算源。

### 10.1 完整范围与准入

资源调度以2026-09-28 Owner效率优先修订§10.4为准；本节原world2描述保留已经实际执行的训练身份，不构成后继物理rank上限。

- T、U分别fresh270，各30,240真实FM query，90/180/270完整ECP；模型、普通跨episode FM、任务/视频/query随机流与优化数值按§2/3/8。
  不读工程权重或MT作初值；两臂按顺序world2训练，不并行开四卡。只在最终270物化/评测，不看90/180选点。
- T/U各一个400条件bank，从本臂完成的270生成。MT只在本批CPU登记旧step300原权重引用；在训练前先完成其只读来源/scene核验，
  提前暴露接口阻碍，但不提前用MT结果改变训练合同。不得重算normalization、teacher调度或重新冻结scene。
- 三臂T/U/MT各一次validation/formal/correct400，共1200行，共用已封存的400 scene、同env/policy RNG；
  T/U共用K1 seed7每task50视频无放回。每臂8full+392compact，官方10flow/执行前5/重规划/dummy10/horizon均不变。
  MT的行保留相同调度索引以配对，但其policy不读教学视频；不得把这个索引记为MT视频条件。
- 完成统一机械验收和§8.1规定的配对统计后停止新增计算，一次完整Queue回主讨论；不自动进入450、其它video controls、Test、RL、新数据或额外episode。

创建新root或任何实质输出前，执行者在strg01核data1独立user quota、共享容量，测旧study/代码占用并按新增32GiB峰值估算。
新root唯一`/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928/stage1`；旧MT data0与旧scene/data1均只读。
新冻结≤32MiB、开发及相关代码合计≤3GiB；复用source/dataset/tokenizer/资产，不复制模型或形成第二数据缓存。
现存规范缓存继续复用，新增缓存/临时产物全部data1且计入32GiB；冻结前按实际稀疏checkout内容核代码上限。

每次GPU launch/resume先同时live核gpu01/gpu02，占卡总量包含训练/物化/评测，**现按AGENTS的8/6总卡、单节点6卡规则；原本批两卡上限撤销**。
训练必须同一节点world2，按现场UUID/NUMA选择并保存两rank绑定；不从历史3/7推断可用，不等待凑卡或干扰其他用户。
`NCCL_P2P_DISABLE=1`、deferred NCCL及GPU-local NUMA保持；物化可两臂各一单卡并行；官方每卡2 persistent replicas，long-first动态队列。
实际设备、拓扑、命令、工作目录、完整环境及来源由执行者在首次计算前写一份launch_contract；不得填占位GPU冒称已冻结。

**预期12–15完整GPUh、6–8小时wall；硬限18GPUh，train11/bank1.5/official5.5，含全部加载/保存/失败/重算/等待。**
依据仍为§8.3实测外推，不是已完成耗时。超过8小时若仍正常但预计无法在总/分阶段硬限完成，作有意义阻碍回报；
硬限前必须停止新增进程并保留能安全保存的状态，不以删评测行、改任务权重、额外调参或追加卡数凑结果。
资源与成本累计由launcher/退出事件计算；主讨论不做固定间隔查询，执行者复用持续退出等待，不分阶段反复Queue。

### 10.2 精确入口、恢复与失败处理

实际命令必须以冻结树为cwd及PYTHONPATH，ASSET固定`/data1/user/ymdai/projects/EMBER`；
`EMBER_STORAGE_ROOT=/data1/user/ymdai`、`EMBER_LIBERO_ASSETS_ROOT`沿已验收0b3ea86be5fe169d0fd036ae63d1070ec09e90f6资产、`MUJOCO_GL=egl`。
使用ASSET/.venv/bin/python，不下载/更换依赖。命令模板由§9对应实现给出，以下是完整语义，不是现场launch记录：

```
python -m ember.operator_writer.bank register-mt --asset-root ASSET
python -m torch.distributed.run --standalone --nproc_per_node=2 -m ember.operator_writer.run train --asset-root ASSET --mode T --attempt fresh
python -m torch.distributed.run --standalone --nproc_per_node=2 -m ember.operator_writer.run train --asset-root ASSET --mode U --attempt fresh
python -m ember.operator_writer.bank materialize --asset-root ASSET --mode T --checkpoint T_COMPLETED_MACRO270 --device cuda:0
python -m ember.operator_writer.bank materialize --asset-root ASSET --mode U --checkpoint U_COMPLETED_MACRO270 --device cuda:0
```

world2启动须实际传CUDA_VISIBLE_DEVICES和已有rank-local NUMA绑定机制；源路径/资产变量展开后原文保存。
官方入口为冻结树`scripts/evaluate_pi05.py run`：coverage_v1/evaluation.json、source aligned step1000、既有tokenizer；
`--role validation --mode formal --state-count 50 --replicas-per-gpu 2`，三臂分别传本root的bank manifest、
冻结`configs/operator_read_write_v1/official_capture.json`及`<T|U|MT>/evaluation/correct400`；设备列表现场核定。

训练恢复只接同臂latest完整90/180/270的ECP，给新attempt名，复制至该checkpoint为止的metrics原文前缀，不覆盖父日志或重置Adam/LR/RNG。
初始micro28/frame8；仅有OOM证据才按§8.2缩packing，并保持逻辑FM随机流。非OOM故障不得顺便调整数值。
若首个90前故障且无完整ECP，只能在原seed/原事件/原冻结代码下重启fresh；保留失败attempt、明确其中已应用和未应用query、
计入全部成本，不能将其记作exact-resume或只报最终270的计算账。有完整ECP时禁止fresh重来。
若需保留失败fresh路径才能重启，先将这一明确task-owned未完成attempt登记为failed并记录原/新路径，禁止覆盖或删除；正式root之外不动。

bank仅允许同一materialization_contract续作未完成条件，已发布manifest不重算；若写入中断留下不能读取的单文件，
原文件和错误保留，先回报，不把来源校验改松或静默挑不同checkpoint。官方故障用现有队列resume，保留已完成shard及失败/partial原件；
场景初态失败是工程合同问题，不能改scene后混合结果；普通episode失败则保留，不重试择优。
冻结后的源码/科学字段不得边跑边改；已定位且不改科学语义的工程违约可由执行者按§10.5在隔离分支修复、验证、新冻结后继续。
涉及科学计算、原因未定或预算越界时，带原件/成本一次回报，由主讨论裁决。

### 10.3 完整交付和科学边界

保存原launch/attempt/环境/退出码/完整GPU秒，三套bank来源、两臂metrics及完整ECP、1200 official JSON/NPZ/PT和共同scene引用。
逐行核task/init/env-policy seeds、T/U condition/video ordinal、T动作/T+1 body/EEF/gripper/predicate；每臂400完整且8full/392compact。
训练报告source trainable0、实际梯度消费者、任务曝光、恢复前缀和游标；不要求逐bit一致或新增tensor/hash扫查。
报告actual applied updates/queries、失败尝试和重算分别计数；不把计划270当所有实际计算总量。

按§8.1给T/U/MT三臂的任务/suite/breadth与全部成对R/G/L/churn/Jaccard、成功集合、bootstrap区间；
旧MT155单列历史标量，不混成新严格配对行。任意正确episode得分仅说明本节点能力；当前无视频必要性、语义理解或相邻保持结论。
正式T/U可能比强MT差，也可能U优于T；这些是合法科学结果，不触发工程修补或自动后继。
主讨论完成原件验收后据预注册规则裁决；执行者在一次完成/实质阻碍Queue后停止新增GPU，主讨论负责接续。

### 10.4 Owner效率优先的现场调度修订（2026-09-28 06:32 UTC起）

Owner明确要求结合已经完成/正在执行的状态重新指导执行者，加快推进；本节覆盖§8.3/10.1及原派发、冻结spec中的额外峰2卡和阶段串行调度限制。科学模型、数据、270节点、三臂correct400、配对/捕获、停止线及18 GPUh/32GiB总预算均不改变；正式计算仍使用784febbff32d991e53b9e5c6ba9f74683890425e，不改已有冻结树、权重或完成原件。

主讨论06:32 UTC只读核见：T fresh270完成、exit0/3.665571 GPUh；U最新完整日志为269，尚未见270完成记录；MT bank已登记，T/U bank及三臂评测未发布。执行者同轮记录Owner已澄清“不是停进程”，且未发送终止信号。此为一次决策快照，实际完成状态由执行者启动前更新，不据此重跑。

1. 已完成T直接复用。U若仍在收尾就完成270/ECP/正常退出，不为扩卡中断最后更新；若已经完成则直接使用。只有真实退出失败才按现有完整ECP恢复，不重启fresh或覆盖失败账。当前没有为剩余一步开发训练拓扑迁移的收益。
2. 立即解除无关阶段之间的等待：MT已有bank，可先行official；T bank可与MT并行，U完成后其bank也可同时生成。每臂bank完整准入后即可评测，不等待其它臂bank全部完成。沿现有official动态队列、long-first、每卡2 replicas，多卡分摊固定400；三臂可以并行，不共享写同一输出/分片。
3. 每次launch按AGENTS实时核双节点：项目总计最多8物理卡，双节点空闲卡总数≤10时最多6，单节点最多6；共驻统一计数，允许余量充足且不显著干扰的低负载卡。选择能缩短剩余墙钟的组合，不盲目占满、不等凑卡。单臂分布式训练不跨节点；独立臂/评测可使用不同节点。
4. 保留原run来源/数值身份；只在现有launch记录增加本次Owner修订、实际阶段重叠/设备/预算及依据。冻结spec的旧peak_physical_gpus=2是历史调度字段，按本节覆盖，不因此要求新模型冻结或重做工程；不得删除其它来源与科学字段校验。
5. 原train11/bank1.5/official5.5及总18 GPUh、root32GiB仍计全部加载/失败/重算。训练接近结束，优先把可用卡用于bank与official；按既有bank0.6–1.0、official3.4–4.5 GPUh估计，若有4–6张适用卡并正确重叠，剩余规划约1–2小时含启动/长尾/验收，非实时速度保证。执行者从首个真实阶段成本修订预计，不另开profile或增加episode。
6. 后续若科学裁决允许450，须按Owner要求让物理rank/分片服务于吞吐，正确承接完整ECP、逻辑112查询/4task权重、全局归约/裁剪/单次Adam更新及LR时钟；不能继续拿world2硬编码当科学限制。当前不预先启动450或为尚无资格的续训建通用平台。

本次为对当前活跃执行的Steer，不另开重复批次，不自Queue。执行者沿现有可靠退出等待承接已启动进程；整批完成或实质阻碍一次回报。主讨论另存消息回执，不能把接口接受冒称已启动全部并行阶段。

### 10.5 评测接口修复与原产物复用（2026-09-28）

首批训练/物化已经完成：T/U各270更新、30,240query及三个完整ECP，各400条件bank；MT旧step300只读登记。
三臂official尚无episode。MT首次prepare在`build_run_contract`读取adapter缺失的`arm`时退出1；
这是可复现接口违约，不构成科学阴性。原训练/物化代码仍是784febbff32d991e53b9e5c6ba9f74683890425e，
不得重训、重新物化、修改旧manifest、共同scene或冻结树。

任务`operator_read_write_official_repair_20260928`由原指定执行者在独占开发树修复并续行，主讨论并行验收原件、审阅与集成。
Owner指出接口违约应能自主修复；本节据此撤销本次CPU交付后再次等待主讨论许可的停点，允许以下固定范围一气完成。
预计15–25分钟，范围明显超过40分钟或涉及科学计算改变时说明实质阻碍。源码限bank到official的适配边界及针对性回归：
返回经固定调度核定的`arm=correct`，区分T/U/MT模型模式；同时解除旧bank provenance对**当前评测checkout**的误绑定，
按原封存spec/run/ECP/Git核验训练身份，新评测代码另记Git。不得将旧训练重新归属新commit，或放松工程/错臂/错source拒绝。
现有训练与materialize的准入不扩大；模型、native、FM、数据、优化器和评测策略计算均不变。

CPU验证必须让三份实际旧bank经过真实adapter→run_contract/official prepare边界，保持信息墙和场景/捕获登记；
仅输出轻量临时准备证据，不创建正式`evaluation/correct400`，不加载policy/运行环境/GPU，不用一次额外smoke代替接口验证。
CPU检查通过后，执行者提交、推送原隔离分支，登记唯一clean pushed detached **evaluation-only**冻结commit和来源差异，
直接继续剩余三臂correct400，无须等待另一条Queue或main集成。该窄修可从已推送隔离commit冻结；主讨论并行审阅、及时集成main。
不改变训练/materialize原有main来源准入，旧bank仍归属784febb；新评测Git与旧训练Git分别保留。
这不是新科学批次；旧失败命令、日志、退出和成本保留，新尝试用独立名称，不覆盖失败原件。

已付完整成本train7.062870483、bank0.623652598、失败official0.060123485，合计7.746646567 GPUh。
原总18/official5.5硬限不变，剩余分别10.253353433和5.439876515 GPUh；原root峰32GiB保持。
§10.4并行安排已经实际应用，旧批峰3卡，不能继续记作未承接；修复后按现场可用资源并行三臂，
不恢复额外两卡上限，不等凑卡。剩余official按原3.4–4.5 GPUh规划，4–6张适用卡时约1–1.5小时含启动/长尾，
另计本次CPU修复；实际GPU分配、quota和预计完成时点由正式续行launch更新。仅运行原三个correct400，复用原输出位置、
bank和场景，现均无正式输出；保留首次失败，续行attempt采用新名。科学范围/数据/预算问题或原因不确定才回报阻碍；
整批修复及评测完成一次完整Queue，不在CPU完成处分批停等。没有450、controls、Test或RL许可。

执行与独立CPU验收补充：83946ae1仅bank/针对性测试2文件+53/−7；旧来源严格指784febb/spec、保留数值与scene/condition检查，
materialize默认仍核本次冻结来源，official adapter从已核selection导出correct。预算调度字段不参与科学数值相等判断。
三套实际旧bank经`_prepared_payload`产生400条件/28shards/8full且CPU阶段未发布正式输出；证据位于执行者dev的
`.codex/tmp/operator_read_write_official_repair_20260928/cpu_preparation.json`及`check_prepare.py`。主讨论审阅并独立6项CPU通过，
以3b24f9ea集成main；新eval83946ae1已启动三臂official，原训练784febb保留，不因集成再启动或重写已记录Git。

### 10.6 Owner要求认真检验架构，避免早期分数触发整版切换（2026-09-28）

Owner本次明确：当前架构有机制上的合理性，应看实际性能，不随便放弃后直接换架构；主讨论尚未验收到本批三臂正式分数。
本节覆盖§4/8.1中“有限窗口关闭”可能被机械执行为“架构立即退役”的含义，不修改本批训练、样本、评测、选点或预算。

270是首个完整行为节点，不是充分训练/收敛证明；两臂30,240query，150步warmup后只有120更新，
末步LR约峰值97%，末两个45步训练FM均值仍下降。不同样本的在线均值不能证明仍有闭环收益，也不能证明平台期。
主讨论须把三臂实际闭环、任务/成功集合、真实曝光与学习状态放在一起裁决，不能只按一个早期总分换架构。
未达原投入线时不自动启动450，也不自动退役；若已有证据支持一项有界后继能区分训练不足与方法问题，
须先登记理由、竞争预测、最小规模/成本与停止条件，保留原节点及原预注册判断，不把后续探索混称原资格通过。

T/U只检验教学地址是否绑定实际A。U若更强，首先更新这项共享约束的判断，不等于应抛弃整个原生读取/矩阵写入路线；
两臂共同不足也不唯一定位读取、表示、更新或容量。明确负证据仍须降低对应假设的支持度，不能用“还可能学会”无限续投。
应保留有用机制与正例，只有更完整证据或有实质理由的替代原则才支持基础重构；不以局部工程故障或单次早期非通过代替。

## 11. Owner授权的270→900学习趋势检验（2026-09-28）

### 11.1 问题、依据与执行边界

Owner在与新主讨论讨论后明确允许开始推进，要求新Sol提高真实吞吐，希望次日早晨看到尽可能好的完整性能。
主讨论已说明并采用T/U从既有270继续至总900、450/900作正式行为节点的计划；本节据此覆盖先交流暂停及旧§8.1的自动后继限制。
首批T116/U128/同scene MT153、breadth5/6/6未达原140投入线，不能改写为原资格通过。
这是明确追加的有限学习检验：270每task仅30访问、warmup后120更新、在线FM继续下降；同时历史V/L和P/I已经证明
loss下降不能保证闭环提高或能力保持。因此允许观察更长学习，但不保证最终收敛，不用追加替共享绑定假设免除负证据。

唯一任务`operator_read_write_continuation_900_20260928`，唯一执行者新Sol `01a0e70e-48bc-7470-a725-c7420f96ee81`。
旧执行者已交付并释放dev，不再派任何后继。主讨论独占main/docs/科学裁决；Sol独占
`/data1/user/ymdai/projects/EMBER-native-conditional-reader-dev`的`codex/demonstration-transfer`源码/测试。
先安全并入已push最新main；不得覆盖他人工作或改旧冻结树。当前模型/native、38-target/rank128、source、FM及优化数值不变。
不增加loss、aux、门控、保持项、rank/LR/seed扫描、MT训练/评测、数据、scene、Test、RL或其它video controls。

### 11.2 完整学习程序、恢复与曝光

两臂分别从旧root `.../operator_read_write_learning_20260928/stage1/<T|U>/train/attempts/fresh/checkpoints/macro_00000270`
的完整ECP继续，各新增630实际更新/70,560 query，累计900/100,800 query；不是fresh重来、权重热启或重置Adam。
旧root绝对前缀为`/data1/user/ymdai/ember_runs`，旧训练Git为784febbff32d991e53b9e5c6ba9f74683890425e。
新唯一root=`/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928/continuation900`，父产物只读引用。

- 每宏步仍4个不同task、各28 query，单condition损失权重1/4；36 task每9步完整访问一次。450/900分别每task50/100访问。
- 第0–49次task访问继续原`SeedSequence([20260928,1,task])`的50视频排列；第50–99次使用
  `SeedSequence([20260928,1,task,1])`的新50视频排列。两轮各自无放回；不把重复50视频称为新增独立meta-task。
  原前270的teacher/query/frame/flow事件保持；task排列与query RNG继续以绝对visit索引，不复位。
  每condition从其它49 episodes无放回取28，frame范围及action offset1保持，flow seed继续原函数。
  旧sampler的`teacher_pool=range(30)`是历史字段，不可解释成实际demo ID仅0–29；显式版本转换记录访问数/完整50池与循环规则，不能静默放宽。
- 绝对LR时钟继续warmup150/decay1200/floor1e-5及原AdamW。保存360/450/540/630/720/810/900完整ECP，只有450/900物化/评测。
  保存Writer/optimizer/scheduler/scaler语义/sampler/各rank RNG/拓扑，metrics继承270行原文前缀，新增行及实际重算单列。
- 明确兼容旧训练schema/source/spec/事件前缀和本次新程序；运行记录父ECP、训练Git、续训Git和迁移原因。
  同拓扑可恢复；完整ECP处允许改变设备及world2/3/4，不称bitwise exact。复用现有ECP的显式world迁移支持；
  旧rank恢复其RNG，新rank保留并登记seed来源，逻辑flow噪声继续由task事件决定。不能删除来源检查来获得恢复。
- rank按真实帧数/历史condition成本负载平衡。world3的任务数2/1/1不改变各任务1/4权重；
  对全部rank的加权梯度作SUM，不作rank均值或按本地任务数重新归一化；全局clip一次、Adam一次、scheduler一次。
  world4为1/1/1/1。不得跨节点拼单臂训练；多卡NCCL/NUMA约束继承。

实现须用针对性的CPU消费者检查证明900事件/前缀、非均匀分配的全局更新、真实旧ECP来源/游标、迁移/拒绝不相容输入、
以及450 ECP在训练继续期间的bank来源识别。已验收模型图与全部1200 capture不重复测试；不做逐tensor/逐bit或全树hash扫描。
如需要实际吞吐核验，必须是本任务当前瓶颈且计入总账；不要另造强制smoke、无用GPUcase或反复2/3/4扫表。

### 11.3 节点并行、正式评测和科学裁决

450完整ECP原子发布后即可只读物化/评测，训练可继续；不能让bank消费者要求900整段退出才能读取已完整450。
每臂每节点一个400条件bank；每个bank完整后接correct400，合计四组/1600新行。复用旧400 scene、seed7全50无放回teacher、
task/init/env/policy RNG和官方执行/capture合同，每组8full+392compact；不新增MT或重跑270。
旧MT153/400原行是同scene强参照，旧MT155是历史标量。bank/training/evaluation Git与显式兼容分别记录。
旧270的sealed bank消费者继续可读；当前唯一运行面扩展节点，不复制一套trainer/evaluator或保留过时CLI。

450是趋势观察点，不是分数低于140便机械放弃整个架构的开关。正常数值/合法训练且预算成立时完成预定900，
不读取中途小样本改样本、挑视频/seed或改学习程序。900后停止新增GPU；没有1200/controls/RL自动许可。
工程失败即时定位，恢复只等待同队列旧worker退出；其它独立臂/节点继续。科学语义不变的窄修按§10.5新push clean freeze续行，
所有有效行、失败/partial、实际成本保留；原因不明、计算/标签/选择语义变化或预计越界交主讨论。

报告完整每task/suite/breadth、全部成功集合、同点T−U与对MT的R/G/L/churn/Jaccard及8-task bootstrap；
每臂270→450、450→900、270→900相邻/跨段保持，说明绝对数和占前节点成功集的比例，不混跨臂差异与时间遗忘。
两臂都涨只支持更长学习改善此范围；U持续较强削弱绑定优势，不能抹掉整体读写；T较强只支持该干预，不识别视频或β必要性。
若总分上升但丢失大量旧成功，仍无稳定能力结论。没有只摘最高点、checkpoint union或以loss弥补闭环缺口。
本批目标为学习曲线和有限方法判断，不承诺选出最终模型；相邻稳定、视频因果与最终Test仍需各自证据/后继范围。

### 11.4 吞吐、预算与时间预期

两节点每次launch/resume同时live准入，仓库合计8/6张及单节点6张上限照常，包含所有训练/物化/评测共驻。
Owner随后明确短时分析例外：已满卡上限时，已授权的小分析确需GPU可临时超限，结束立即释放；
仍现场准入、不干扰他人任务、计完整成本，不扩大本批科学范围或30GPUh预算，不借此扩常规长训/整轮评测。
优先T/U并行；每臂world2–4由当前设备和实际吞吐选，不继承旧两卡整批上限或固定每卡2 evaluators。
4逻辑condition意味着超过4训练rank无本任务收益；6卡可按3+3或4+2等成本依据分配，8卡可4+4，
也可留适用资源重叠bank/official，目标是整批最早有效完成，不为占卡空转或为扩卡中断未保存更新。
评测replicas按实时显存、CPU、实际episode吞吐选择；旧3 workers约38GiB仅内存估计，不能代替吞吐依据。
采用已有动态long-first队列/persistent workers；允许真实packing/CPU预取优化，但不改变逻辑batch、随机流或任务权重。

成本依据：旧T/U训练7.0628704835GPUh×630/270≈16.48；两节点bank约1.25；四组official按旧T/U含失败成本外推约4.39。
**新增预期约22.1GPUh；完整新增硬限30GPUh**，其中一切加载、保存、必要吞吐核验、失败、恢复和占用等待计入，
旧stage1的11.1582785066GPUh另列累计，不能冲掉旧失败。30不是应花满的额度；一次实质吞吐核验最多0.5GPUh且并非必做。
以旧world2宏步T中位23.41s/U21.58s估计、并行训练及节点重叠，GPU阶段约5–7小时；CPU实现/验证约45–75分钟，
整批初估6–9小时。相对本次17:40左右北京时间启动准备，目标次日早晨08:00前形成完整可审阅结果；这不是性能或机器可用保证。
若真实关键路径明显无法满足时间/30GPUh，尽早报告实质缺口和可保存节点，不等额度耗尽，不静默删行/改单臂或续超预算。

旧两bank各8.732GB、两arm训练合计2.782GB、两组评测约1.634GB；四新bank约34.93GB、14个完整ECP约6.5GB、
新capture约3.3GB，预期新增约42GiB。**新root及新增缓存/冻结代码/临时失败产物合计峰值≤56GiB**；
旧root约21.1GiB保留，不复制source/data/tokenizer/父ECP。创建实质输出前在strg01核data1独立quota、共享容量、实际用量和峰值余量，
不足先回报具体缺口，不改写data0或删除科研原件。新冻结稀疏代码沿现有方式，目标≤32MiB，不重复拷贝环境。

新Sol完成上述同一任务工程后，可以从其已push的独占分支clean detached冻结直接执行，精确Git/ref/命令/环境/输入/
parent/输出/拓扑写一次launch_contract；允许`origin/codex/demonstration-transfer`为此次明确来源，不能伪称origin/main。
主讨论并行审阅和及时集成，不因科学不变的工程改动再要求Owner许可；主讨论发现实质契约问题可即时纠正。
源码就绪时仅一次有内容的交付消息便于主讨论集成；整批完成或实质阻碍再回本主讨论，不能为每个checkpoint/bank自排通知。
使用退出/完整产物事件衔接，不固定间隔反复扫日志/缓存；没有持续等待者才安排唯一可靠整批唤醒。

主讨论同时推进实际特征—矩阵写入—自身动作—FM更新的可失败推导，继承既有原件验收并补真正缺证；
不把已完成的消费者审阅再包装成研究，不以无限纸面推导阻挡有界学习，也不凭代数反例直接改架构。

## 12. 与续训并行的原270训练侧功能读取诊断（2026-09-28）

Owner指出主讨论不应在完成一项输入层局部分析后就等待续训。主讨论继续完整机制推导，并登记这一项最小实际证据：
当前视频条件残差在自己的query上几乎无功能作用，还是已有实质作用却跨视频/状态得失相抵？
§43的A/S/probe不能回答这个问题；普通hidden差异/gradnorm也不能代替。此项使用原270，不等待或选择续训节点。

- 固定train四suite各首个登记代表task0/12/20/32；每task取原teacher排列第0、1条，T/U共8条件/臂。
  每task为两teacher共用28个其它episode query，排除两teacher，`SeedSequence([20260928,44,task])`无放回选demo及合法frame；
  flow seed沿现有函数、optimization seed7、独立diagnostic visit1000。两臂/两视频/参照共用实际queries、time/noise。
- 只读784febb冻结数值和原T/U270完整Writer；不更新参数、不改训练采样、无环境episode，无validation/Test标签或选点。
  Writer仅读合法teacher RGB/L；query state/action只进入原普通FM监督消费者。
- 取得真实教学H和第8层Q/V输入、完整LoRA下自身query输入及FM完整cotangent；从实际K/V递归计算各帧写入对
  `⟨G_B,M⟩`的有符号贡献，并核其求和等于该target的值缩放偏导。这是当前完整FM的局部信用，不是成功率归因。
  给出native变化、实际写入/读取和功能信用的对应，不能由cosine/幅度直接命名物体或操作语义。
- 同query读取完整生成LoRA的FM和当前公共β的FM，以及全38个M统一缩放的当前导数。
  β是共同训练后去掉条件残差的冻结参照，**不是独立训练的language baseline，不识别视频必要性或泛化收益**。
  该差额仅回答当前训练query上残差的功能作用；不据小面板挑视频/模型或改变450/900合同。V随key、其它层hidden与真实query梯度一起解释。
- 每个condition保留可复算的小型feature/credit原件、样本/权重/源码来源，报告全部16条件；不能只挑较好视频或只报有利target。
  stage1及续训正式结果不动；输出在`continuation900/analysis/teacher_to_query_270`，新增≤2GiB并计入原56GiB峰值。
- 唯一新Sol执行；主讨论提供task-owned临时脚本并独立分析。优先完成续训实现/启动，诊断在独立适用卡与长任务并行，
  不阻塞已就绪训练；现有长任务已满上限时按Owner短时例外准入。只需1张A40，预计5–12分钟、完整硬限0.5GPUh/30分钟，
  全部加载/失败/占用计入续训总30GPUh。按现场显存/CPU选择打包，可窄修接口但不得扩任务/节点/干预。
  诊断完立即释放；以一份有内容的分析交付回主讨论，后续formal整批回报保持原唯一渠道，不为各条件发通知。

结果分支：若完整M几乎不改变实际FM prediction，降低“有强功能但只因保持失败”的解释；若作用显著但跨query/视频混合，
不能称视频未被用，也不能将所有负信用当错误写入；若训练FM一致受益而formal仍不足，优先区分迁移与闭环分布，
不以train面板通过当泛化资格。任一分支都不自动触发局部loss/保持/寻址补丁或新架构；与900的完整曲线共同裁决。

## 13. 450已就绪后的固定面板功能与地址绑定诊断（2026-09-28）

Owner要求现在深入450的实际机制，尤其T/U差别，不将分析推迟到900。正式续训/评测保持§11；
本节仅增加一次与其并行的短训练侧诊断，不增加训练、环境episode、held标签、正式controls或选点。

要区分的两个解释是：U的当前优势主要伴随共同β能力变化，还是独立地址已经参与产生有益的条件修正。
仅测S/A距离、key集中度或270某层信用衰减不足以区分；也不能把T450双物入篮失利直接归因于编译时覆盖。

- 原样复用§12的task0/12/20/32、每task两teacher、28个跨episode query与flow种子，完整报告16条件；
  不按450表现挑task/video/query。读取各自完整450 ECP，正式数值代码使用81846ed3冻结树。
  同一模型比较完整LoRA与共同训练β，并与已经保存的270同面板结果比较，不重跑270。
- 在U450做一个明确的冻结干预：只将全部38处教学地址投影S替换为该U自己的A；
  β/native X/H、A/B0、P/C/D/O、query及flow均固定，重新计算K、依赖K的Value及完整递归M，再做普通FM前向。
  不交换T/U的参数、不只改key后沿用旧Value/M、不优化、不部署或物化这个诊断LoRA。
  记录实际FM变化、prediction差以及完整LoRA处cotangent预测的一阶变化；保留全部8个U条件。
- 继承§12的Q8/V8/action_out实际native/query特征和逐帧Value信用原件，比较450与270的功能变化，
  顺手保留同一次FM已有的G_A（不增加GPU反传），供CPU区分直接执行与固定native下key的局部信用；
  第三条β→native路径未由此拆出，不能把两路内积当总梯度冲突或Adam实际更新。地址谱/尾部信用也不单独成为性能根因。
- 若U正常残差有益而强行绑定使FM变差，支持“当前U学得的独立地址具有功能价值”；
  不等于训练T一定失败，冻结干预还包含破坏共同适配的效应，不能唯一证明梯度冲突。
  本干预同时改变Value的K输入与递归地址，不能把功能差额全部解释为地址几何，更不称独立投影已学到语义。
  若绑定变化很小，则降低“450优势主要由当前S的独特读取作用造成”的解释，保留学习路径与β/Value差异；
  若变好，登记独立自由度在该面板上的代价，不据此回绑、选点或重训。
  无论哪一分支，train功能不是held闭环中介或视频必要性证据，后继须结合完整450/900和历史反例。

唯一Sol执行，脚本为同名task-owned tmp中的`teacher_to_query_450.py`，运行时保存实际脚本/合同/16份PT/全部行。
输出`continuation900/analysis/teacher_to_query_450`，预计约0.8GiB、硬限2GiB；复用已准入根目录，
现场确认可容纳于原56GiB峰值及data1 quota。单张A40，原270实测179秒，据此预计5–10分钟含新干预与启动；
所有加载/失败/占用累计硬限0.5GPUh/30分钟，计入原续训30GPUh，不加总预算。
按实时资源并行，必要时使用Owner已授权短时超卡例外，结束即释放，不抢占他人或阻塞正式训练/评测。
若预计剩余预算/存储不成立，回报具体缺口；不静默放大、重复失败矩阵或中断正常formal来凑诊断。
完成或实质阻碍一次有内容回报主讨论，formal原整批回报不变；不按条件通知或向自己Queue。

执行与分析已完成：133.4325秒/.037064589GPUh、exit0、16条件/8回绑及小型原件完整；
实际功能、局部路径信用和非线性反例见findings§209/机制§49。此处不追加后继实验许可。

## 14. Owner授权的完整链路机制研究（2026-09-29）

Owner明确允许选择分析/实验，以建立自己的视频→动作过程指导→LoRA→自身闭环理论；
最终选择继续学习或已有合理修正后的正式重训时，停下解释证据与推理，不直接开训。
本节不改已完成§11–13；正式checkpoint、旧分数和所有原件封存。诊断结果可支持当前方法判断，
但训练面板不是held资格，不能挑teacher/拼checkpoint；Test与held action/reward梯度墙保持。
尚无当前T/U视频特异性差的直接实验，先区分真实匹配贡献与此前其它架构的负证据。

### 14A. 固定900的训练侧视频来源交叉与功能读取

问题：已测完整LoRA优于公共β，究竟主要是视频通路提供通用有益修正，还是正确教学内容提供匹配目标的功能？
二者对继续学习的理由不同。若已有匹配功能而迁移/保持不足，不能再把问题归为视频图没学到；
若不同任务视频仍给出相近收益，则一般动态输入引起修正不等于获得了教学过程知识。
这一比较保持target exact language，不能把换语言的task变化归给视频；所有干预是已冻结模型的诊断，非新训练样本。

- 固定T/U900两个完整ECP，数值source沿81846ed3 clean detached树，模型/native/loss不改。
  权重路径为`continuation900/{T,U}/train/attempts/continuation/checkpoints/macro_00000900/ecp.safetensors`。
- 目标与视频来源均为原train0/12/20/32四task；每个来源仍用§12原排列前两teacher。
  4 target × 4 donor task × 2 donor teacher × 2 mode = 64完整编译/功能条件。
  条件使用donor真实同步双RGB/stride5/末帧/原顺序，tokens/mask明确来自target语言；不传donor语言、动作或state给Writer。
  对角线16条件与原270/450面板同口径；其它48为固定语言下的视频来源干预。
  每个target的28个query/demo/frame、flow seed与§12/13严格复用，排除该task两个正确teacher；不按结果选样本。
- 每target另读本臂共同训练公共β（共8）及同源identity/source和原强MT300（各4，仅需跨臂复用，合计8）。
  MT读取旧已审`stage1/MT/banks/300/manifest.json`引用的实际完整因子与原precision，不重训MT或创建新bank。
  source/MT/β/完整条件的query与flow完全相同；不是匹配训练算法的因果消融，也不将β命名为独立语言baseline。
- 使用现有`paired_functional_credit`/`flow_sample`实际普通FM消费者，保持50×7、offset1和原随机time/noise。
  保存每query每horizon实际prediction与target（仅授权train标签）、flow time、全部标量FM；由此CPU可读前5/其余45但不改主loss。
  只对16对角线正确条件求已有完整LoRA cotangent，保存76项G_A/G_B及3target Q8/V8/out的自身query输入。
  64条件保存实际H、这3处native X、全部38最终M；每臂A/B0/共享Value权重只保存一次或引用900 ECP，避免重复大资产。
  保存真实frame indices/teacher/target language来源/源码与权重路径，保留全部条件和数值，不以norm做资格。
- 0 optimizer更新、0环境episode、无validation/Test action或label，无900后训练、无shuffle/reverse、无最终checkpoint选择。
  不做无意义首帧重复矩阵：在本图完全静态H令ΔH=0，M必为0，精确退化公共β；这不等于独立训练的静态参照。
  原生批处理的正常低位差异按既有数值规则处理，不能为它强制batch1或扩大dtype。

初批输出`/data1/user/ymdai/ember_runs/operator_chain_diagnosis_20260929/train_functional_900`，
预计新增4–6GiB、峰值硬限8GiB；创建实质输出前Sol核strg01/data1 quota及共享容量，source/data/env只读复用。
从原16条件133秒估算新增80个功能前向（其中16需反传）加编译，GPU约5–15分钟量级；CPU实现/读回预计30–60分钟，
初批外部全部占用/加载/失败合计硬限1GPUh，任一设备进程硬限30分钟，超出前报告具体阻碍，不自行扩量。
T/U可各占一张合适A40并行，source/MT只算一次；microbatch按真实显存/吞吐，不能机械继承两workers或等凑卡。
每次launch live查两节点，遵循常规总卡规则和Owner短时例外；完成立即释放。

唯一Sol在已独占dev/任务临时目录实现诊断脚本，不改main；脚本随run原件保存，调用原clean frozen模型消费者，
不为本诊断新增canonical trainer/模型分支。接口/记录/调度窄修可在本scope自主完成；科学变量、标签配对、预算变化向主讨论报告。
完成或实质阻碍一次有内容回报主讨论，主讨论并行做理论与CPU分析；不按条件通知、自Queue或索取重复回执。
结果分支由主讨论独立核原件后解释：正确是否优于错来源、两正确是否保持、相对β/MT是否有益、效果在什么query/horizon出现，
以及它们与270/450已有学习趋势能否相容。错误视频损伤本身也可能来自分布偏移，不能单独称正确内容必要，
更不能将4个训练task的差额冒充8个held任务闭环根因。下一因果实验依据实际缺口另登记，不自动续做矩阵。

### 14B. 只读既存810 ECP的相邻能力观察

继续学习的一个实际未知是：450→900的长区间掩盖了什么近期趋势？T净进步、U净下降不说明最后90更新仍同向。
现已有两臂完整810 ECP，无须更新模型即可缩小这个未知。此项与14A独立；不等待其loss决定选择哪个checkpoint，
事先固定只看810，不按此分数再寻找720/630等峰值，不将追加观察自动纳入最终checkpoint选择。

- T/U都读取原`continuation900/{T,U}/train/attempts/continuation/checkpoints/macro_00000810`，0 optimizer更新。
  来源仍81846ed3、原teacher/query/flow/绝对LR，完整ECP身份及游标810如实保留；不得伪装成900、修改旧规格或重训。
- 每臂物化完整400个correct条件并各跑一次official400，沿现有50视频无放回映射、原400 scene、env/policy RNG和官方执行口径。
  8 full+392 compact PT、400 passive NPZ、失败/续行与全部成本按原消费者保留；不新算MT、270/450/900或任何controls/Test/RL。
  与已有450/900/MT配对，报告per-task/suite/breadth、R/G/L/churn/Jaccard、固定8-task bootstrap及原件完整性。
  原有已核证据继承，不重扫旧1200/1600 PT/NPZ；如发现具体配对问题则仅针对相关边界核验。
- 工程复用`operator_writer/bank.py`及现有official owner，显式只增加登记的810 evaluation-only读取/物化准入。
  不扩大train入口、不改变事件/优化/模型/native/loss/source，不新建平行bank或evaluator。
  原818训练来源与新物化/评测Git分别记录，显式兼容旧clean frozen规格/来源；禁止放宽所有checkpoint或静默绕过检查。
  旧450/900/MT消费者保持。针对真实810 ECP元数据、错mode/来源拒绝、capture与既有读取做必要CPU检查即可。
  在独占dev完成窄扩展、push新clean detached冻结后执行，主讨论并行审阅集成；不等待main集成或额外许可。
- 新bank与official可复用原canonical路径的空白810子目录：`continuation900/{T,U}/banks/810`、
  `continuation900/{T,U}/evaluation/810/correct400`，不修改任何已发布270/450/900结果。
  新合同、launch、成本与汇总放`/data1/user/ymdai/ember_runs/operator_chain_diagnosis_20260929/adjacent810`，
  不将新成本回填成旧900批原成本，不复制旧权重、旧bank或训练状态。

依据已完成900两bank .58805GPUh、两official1.93951GPUh，预计新增约2.6–3GPUh，硬限4GPUh，
全部加载/失败/共驻外部占用计入新账；不因无失败而省略cost。新产物预计18–20GiB，含代码/失败/临时峰值硬限24GiB，
须live strg01/data1 quota与14A峰值合计可容纳后才启动。CPU窄实现/验证预计30–60分钟；实际适用卡并行时
bank/eval及CPU原件验收预计再45–90分钟，整体约1.5–2.5小时，非性能保证，明显超期/预算变化先说明具体原因。
T/U物化独立、各bank发布后eval立即可开始，不等另一臂；现场按吞吐选择适用卡/replicas，遵守两节点总量与单节点规则。
既有900用3 workers/卡不成为硬上限；不为追求形式扩卡重复计算或干扰他人。完成立即释放。

唯一Sol保持14A原件/完整成本，不因本追加中断它；可将无需占GPU的14B窄实现与14A实际运行重叠。
14A完成时原约定实质回报可正常送出供主讨论分析，14B最后一份完整回报；不按分片/节点自通知或反复发状态。
两项均不授权900后训练、新架构训练、更多观察点或最终选点。若810与900仍无法判断更长学习，保留不确定性，
不能自动以“再多看一个”无限补曲线；最终选择须连同机制证据向Owner解释后停下。

Owner后续明确回报路由：Sol的源码交付、完整结果和实质阻碍直接Steer到主讨论当前活跃轮，不使用Queue。
发送前一次只读核目标active turnId并用expectedTurnId准确投递；旧已送回报不重发，同task/root继续判重。
这覆盖本批早先派发正文中的Queue方式，不改变科学范围、预算或独立工作分工。

### 14C. 对14A匹配优势的近邻教学来源检验

14A实际原件已显示四个train task中匹配视频一致优于跨suite来源，但这种干预同时大幅改变场景/对象集合。
因此只补一个具体问题：所学功能的匹配粒度是场景共性、操作共性，还是目标对象/关系？
这不是已选checkpoint的正式controls，仍为train-only冻结功能诊断；不测试shuffle/reverse或改变架构/训练。

- 固定原T/U900、14A的target0/12/20语言及原28个query/flow/correct两teacher；原correct/β/source/MT结果直接复用。
  新donor：target0仅用train1（ramekin旁黑碗）、train2（桌面中心黑碗）；target12仅用train13（BBQ sauce）、
  train14（ketchup）；另target20用train21（碗放炉灶）、train25（盘子推到炉前），区分同类厨房中的不同操作。
  六donor均在已授权训练集，每donor仍取`SeedSequence([20260928,1,task])`原排列前两video。
  3 target × 2 donor × 2 teacher × 2 mode = 24新增完整条件，全部固定后一次完成，不挑视频或再铺其它任务。
  在看到任何14C结果之前加入20组并澄清解释：0/12近邻也可能提供恰当的共用抓取/运输/释放过程，
  对它们不敏感不能直接判视频失效；20组用于进一步限制“只是场景相近”的解释。并非同一物理初态的纯动作干预。
- donor只供真实双RGB/stride5/末帧/原顺序；target exact language、query、flow及81846ed3数值消费者保持。
  无需反传；保存prediction/实际train target/time、H/三处native X/38个M、真实video帧索引和全部scalar。
  24项实际FM/配对CPU读回，与已有正确和远来源作功能分解；不重复14A正确/参照或读取held动作。
- 若近邻也明显损伤，扩大对细粒度对象/关系匹配的支持，但仍不能排除静态目标/结果画面；
  若同操作近邻接近正确，则保留操作共性有效与粗场景线索两种解释，结合不同操作组再收窄；
  不宣称过程知识已建立或细目标失败，同suite视频本身仍含不同初态/手法。
  两种结果均不直接支持改损失或重训，需与完整原件/历史和14B近期能力合并作选择。

唯一Sol在任务临时脚本中复用14A读取，不改模型/native/正式训练或新建运行面；脚本和合同随原件保存。
新root=`/data1/user/ymdai/ember_runs/operator_chain_diagnosis_20260929/train_nearfoil_900`。
从14A实际80条件/.099074GPUh估算此项GPU约3–8分钟，CPU复用/读取约15–30分钟；含全部加载/失败硬限0.5GPUh，
任一设备外部进程硬限15分钟。预计1–1.5GiB、峰值硬限2GiB，live quota与14A/B总新增量合计准入。
可在14B bank/eval独立运行时使用适用卡并行，按现场显存/CPU/卡总量不干扰已就绪工作，结束立即释放。
完成或具体阻碍直接Steer主讨论，不Queue、不为普通状态通知；不因这项变更打断正常14B。
14A/B/C是当前明确诊断范围，全部0更新；无正式续训/重训/选点/其它旧点或自动追加对照。

## 15. 夜间有界继续学习：T优先、U先观察恢复（2026-09-29）

Owner已恢复主讨论自主裁决/执行，不再要求分析后等其同意；此处先授权唯一Sol在14B独立运行期间完成CPU续训准备。
初始只做CPU准备，主讨论消费810整批结果后在§15.4释放最终GPU范围；该节覆盖早先的CPU等待边界，
不另向Owner设审批。不得为准备工作打断14B或读取中途分数，A/C不追加矩阵。

### 15.1 判断问题与预备规模

14A/C已证明当前训练分布上有内容匹配、方向相关、涉及上游hidden的有益功能；270→450→900固定query继续改善。
T闭环后段改善，U完整train FM更好却held退化；没有经验证的架构/训练修正。故准备一次保留当前方法的有限学习窗口，
检验“有用的条件函数继续学习能否形成超过强MT的闭环能力”，而非以图接通或未证明失败作为投入理由。
810的最近90步方向用于决定是否两臂都投入；本节预备上界T/U各900→1350，不实施模型/loss/LR/seed/rank小扫。
1350对应每task第三轮50视频刚好完成，是有限曝光单位，不是最优步数断言；不能据此保证收敛。

Owner随后明确90步保存间隔可以，原先未意识到旧ECP已经每90步保存。因此沿用每90宏步完整ECP
（990/1080/1170/1260/1350），撤回更密保存的准备；预备正式评测仍为1080、1350两个完整correct400，不是逐点全评。
新窗口450更新/50,400 query每臂，累计151,200 query/150访问每task；四task等权、每task28 query不变。
后继无论留一臂或两臂，必须保存全部已发布成绩，不以旧点搜索、挑视频、union或checkpoint融合获取胜MT表述。
1080/1350用于预定趋势及绝对比较；不自动形成最终论文checkpoint选择或controls/Test/RL授权。
新结果若仍不超MT且所学功能不能扩大有效覆盖/保持，不自动续1800或把LR回升、seed/rank更换包装成同一次学习。
主讨论届时结合实际结果重新裁决，Owner当前授权允许这种自主裁决但不免除新的依据和有界合同。

### 15.2 唯一运行面与900完整恢复

在现有`operator_writer/data.py/run.py/bank.py`所有权内替换active continuation入口；不复制trainer/bank/evaluator。
保留旧270/450/810/900 sealed读取/audit/official消费者与历史冻结树，不让旧训练spec重新启动。
正式新spec建议`configs/operator_read_write_v1/continuation1350_spec.json`，输出
`/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928/continuation1350`；旧root只读。
新的唯一train默认/入口只接受本spec、指定臂的原81846ed3完整900 ECP，或本root最新完整ECP；不从810/270重放。
明确区分新Git/spec与原训练818/spec身份，检查同臂source/operator/lora/optimizer/信息墙/完整cursor，
继承全部900行metrics前缀、Adam状态、scheduler绝对时钟、rank RNG和拓扑，不声称bitwise exact。

事件schema显式由v3迁移v4。visit仍取绝对macro//9；teacher round0的种子保持`[20260928,1,task]`，
round r≥1保持/扩展为`[20260928,1,task,r]`，r=visit//50；query/flow和task排列的绝对visit规则不变。
旧0–899事件严格继承，第三轮使用r=2新排列，仍50视频各一次；不可让现有v3的第二轮分支被无限重复。
当前真实LR公式保持warmup150/decay1200/floor1e−5，不重新warmup、不重置Adam或重新拉长衰减。
跨过1200后继续原floor是本窗口含义的一部分；1350是观察边界，不把低LR导致变化小称为已证明架构上限。
主讨论按实际`clamped_lr_multiplier`复算：900完成后的下一更新LR为6.4594e−5，1080后1.9246e−5，
1200后1e−5；900→1350的LR系数和约.010138，仅为450→900的.069840的14.5%。
这不是Adam实际位移/有效梯度量，更不能线性外推成绩；本窗口检验已有解附近的继续拟合与保持，不能冒称充分大步搜索。
不为获得更大变化擅自重启LR或改scheduler；若窗口无实质进步，需要重新论证投入，不能直接解释成“总步数还不够”。

每90步保存Writer/optimizer/scheduler/scaler/sampler/RNG/topology完整ECP。一次保存应原子发布后继续训练，
可并行物化预定节点；不用同步等bank/eval。可在完整ECP边界实现窄范围受控结束能力，
使预算/主讨论停止或更合适拓扑迁移不必丢未保存更新；不加热更新科学参数/通用调度平台。
world2/3/4沿既有四条件各.25、SUM梯度后一次全局clip/Adam/绝对LR；按真实负载与可用卡选，不机械固定world2。
T/U、bank/eval独立并行，资源调度以整批墙钟吞吐为依据，不等凑卡或为扩卡重复更新。

### 15.3 证据、成本与交付边界

沿旧strict paired400的50视频无放回、scene/state/video/env与policy RNG、official口径，MT153与旧各点原件直接复用。
各新组保留8full+392compact PT、400 NPZ、原行/配对/成功集合/逐task/suite/breadth/退出及完整成本。
主讨论独立复算新结果及关键来源，完整捕获继承经审实际消费者，避免为可能的假想变化防御性重扫旧原件。

旧630更新两臂合计训练16.969787GPUh，按450/630线性外推12.121276；原900两bank/两official约2.52756GPUh，
两个新节点约5.05512，总预计17.2GPUh；含加载/失败/所有外部占用硬限24GPUh，不用剩余预算自动增项。
world2两臂并行旧实测外推训练约3小时，CPU窄实现/检查约30–60分钟，末节点bank/eval/读回约45–90分钟；
整体约4–6小时，实际拓扑能改善时取更快方案，争取早晨08:00前完成；显著超期/预算风险及时给具体说明。
原完整ECP实测T428MiB/U457MiB，新每臂5份约4.4GiB；一组新bank+official按原900实际约9GiB，四组约36GiB。
加代码、原子写入、临时及失败余量，新增峰值硬限52GiB，须由Sol在strg01 live quota并合计14A/B/C原件后确认准入。
所有新增data1，不复制源模型/dataset/旧bank，不因预算富余做profile、smoke GPU或重扫旧数据。

Sol在独占dev实现、按`code-architecture-gate`检查本次旧大文件增长/唯一路径，不把共同逻辑复制为新版本分支；
冻结旧入口与兼容读取是有实际原件消费者的边界，不是保留可随意开训的fallback。必要时内聚提取共享合同读取，
不为行数切碎模块。做实际900 ECP迁移、第三轮排列/旧prefix、90步完整ECP/新bank准入、错来源/mode拒绝及旧capture消费的
定向CPU验证（其中新ECP按Owner纠正后的90步口径）；push clean commit及新detached冻结，向主讨论Steer源码/CPU证据。主讨论并行审阅和集成。
本节初始只允许CPU准备；14B后的实际释放见§15.4，无需Owner再次确认。
纯工程窄修仍由Sol按原scope自主处置；模型/标签/更新语义/评测/预算变化交主讨论，不能将科学阴性修成正结果。

### 15.4 810验收后的正式执行释放（2026-09-29 01:30）

主讨论已直接读新800结果行、配对和来源/退出账，独立复算7组统计，与§14B报告一致；
完整捕获继承已审实际CPU消费者。T810144→T900148，R/G/L129/19/15，保持89.6%、Jaccard.791；
U132→132，108/24/24、保持81.8%、Jaccard.692。T最近收益包含task31+6，task3反而−1，
而U的task31仍18→14。两臂固定train功能均改善但U未转化为更好held能力，故投入优先级应区分，不能只比较train loss。
这增加T有限继续学习的依据、降低U直接完整追加的优先级；不证明绑定机制普遍胜出或U永不能恢复。

**现正式释放：T从自身900完整ECP续至1350，U从自身900先续至1080并受控停止。**
T分别在1080/1350物化和correct400；U在1080物化和correct400。旧MT153、270/450/810/900原件全部复用。
本次初始释放3组1200新行，T450/U180新增宏步；U启动用`--stop-after-macro 1080`，不能把停止点写成已完成1350。
U1080完整结果直接Steer主讨论；主讨论核验后，只有恢复到超过旧143观测点、且全量得失/覆盖未显示新增大范围退化时，
才考虑在原24GPUh/52GiB上界内释放U剩余270宏步及1350一次correct400。144不是资格线、MT仍153，
这个投入条件用于避免弱趋势得到自动完整追加；没有新的明确主讨论释放，U到1080就结束计算。
不得为跨过该投入条件搜索其它旧点、重选teacher或重启optimizer/LR。T正常完成既定有限窗口，不自动续1800。

初始释放的实测外推：训练按旧630窗口为T约6.061/U约2.424GPUh，三组bank+official按旧900共约3.791，
合计约12.3GPUh；若后续释放U剩余部分，则仍用上文完整约17.2/硬限24。72GiB旧准备数字已被52覆盖。
物理拓扑交Sol现场依据吞吐决定，T作为性能主臂优先；不将T4/U2固定为新限制。
若适用卡允许，可T world4与U world2并行，并在U1080受控退出后立即用释放卡做就绪bank/eval；
其它更快且满足全局权重/卡数/预算的安排也可。不得让已就绪评测等待与其无关的训练完成。

CPU源`14bac4cdd6c27eee06f5574317da8257834e3884`已push，新冻结树
`/data1/user/ymdai/projects/EMBER-operator-continuation1350-formal`为同提交clean detached；
main集成`b43eef90`，主讨论独立14项CPU于16.05秒通过，直接审过恢复、第三轮、受控停止和新旧bank来源。
新模型/native/loss未改。源码增长限于原run/data/bank（742/143/679行），保留一个active train入口；
source净增长158行，旧大文件的共同合同/只读来源仍内聚，不为行数拆碎；architecture REVIEW无新增hard的范围接受。
旧900恢复原件及与新bank读出的测试已覆盖；没有重跑GPU smoke或全树/逐tensor测试。

实际命令沿原launcher：新freeze下`python -m ember.operator_writer.run train --spec <新freeze>/configs/operator_read_write_v1/continuation1350_spec.json`
附同臂900完整`--resume`、唯一`--attempt`、`--asset-root /data1/user/ymdai/projects/EMBER`及现场world/microbatch/NUMA；
bank/official复用现有owner与官方config，所有数值PYTHONPATH/cwd均为新freeze。Sol在launch记录写实际完整argv、环境、
拓扑、准入和原件路径；训练/物化/评测均须clean pushed冻结，live双节点卡与strg01 quota成立后直接启动，不等main另行确认。
实质故障/预算风险按原窄修边界处理，源码/关键完整节点/整批结果直接Steer主讨论活跃轮，不Queue或发心跳。

### 15.5 U1080完成后的投入裁决（2026-09-29 03:43）

主讨论独立核验U1080=128/400、breadth6，U900→1080保留110/得18/失22，未恢复到§15.4预先要求的旧143以上。
据此本轮U在1080完整ECP结束，不释放剩余270步或U1350 bank/official。原件、完整配对与解释边界见findings§213；
不是提前故障中止，不是架构不可能结论，不为跨过投入条件搜索旧点/更换视频/重启LR。
T1080和1350继续既有释放，预算上限不增加；最终外部账完整保留，不把U未执行部分记为实际成本。

## 16. T单臂1350→1800的最后一个本夜原样学习窗口（2026-09-29 04:20）

### 16.1 证据、竞争预测与停止边界

Owner已恢复持续自主研究/执行，要求争取实际超过强MT；无需再等其意见。§15整批已由主讨论独立验收，
T900/1080/1350为148/147/154，U1080=128并结束追加。T1350相对同scene MT153只多1，不能作稳健优势结论。
T1080→1350净+7、保留87.8%，收益分布于四task，覆盖未扩大；完整得失/不确定性见findings§215。
冻结900的实际视频→M→自身执行功能已有内容匹配证据，但不能冒称1350已证明同样特异性；尚无经过验证的重训修正。

继续学习假设：现有K/V/递归M与自身A的共同映射仍能通过同task跨episode功能梯度获得有益的有限调整；
目前并非只能改善单一目标碗。竞争解释：低LR下训练函数的小幅变化主要交换闭环成功边界，平均能力已到平台，
重复相同50视频不会补足欠缺的迁移关系。新增一轮保持模型/数据/损失/时钟，直接检验前者的剩余实际价值，
不把非零梯度、在线FM小幅改善或“可能尚未收敛”单独当继续依据。

本批只T从完整1350续到1800，新增450宏步/50,400 query；U不继续。1710与1800固定两轮correct400，
相隔90更新，1350为已完成固定参照；不搜索1440/1530/1620等旧点，不根据中途分数改节点或选teacher。
1800是本夜原样监督学习的终点，**没有自动2250、LR重启、其它seed或临时补评授权**。
若1800未高于154或新两点主要表现为当前水平的交换，降低/结束原样续训的投入；不能以任何单点偶然高值续开窗口。
若两点表现出实质收益和相邻保持，则记录候选能力及剩余视频因果缺口，也不自动宣称资格或自动延长。
1710是相邻观察，不是1800失败后挑峰的兜底；最终方法/controls仍须另作基于全量证据的明确裁决。

### 16.2 唯一运行面、恢复及信息墙

任务`operator_read_write_continuation_1800_20260929`；唯一Sol在既有独占dev分支实现，主讨论独占main/设计/集成。
从最新已push main安全合入，保持同一run/data/bank owner；当前1350和更早数值冻结不改。
唯一新root=`/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928/continuation1800`。
父源严格为T的`continuation1350/T/train/attempts/continuation/checkpoints/macro_00001350`，训练Git14bac4cd。
只新增`continuation1800_spec.json`当前训练配置，mode仅T；旧270/900/1350训练入口关闭，旧spec/ECP/bank继续合法只读。
parent游标/臂/来源/冻结spec须显式兼容；不因新评测Git静默放宽旧来源，不重复物化/评测MT或任何旧点。

原0–1349事件及1350行metrics、Adam状态、绝对scheduler、原rank RNG/拓扑承接；v4→v5显式迁移。
第四轮teacher排列为`SeedSequence([20260928,1,task,3])`，每task50视频各一次；query/task/flow继续绝对visit，
保持每步4task各28跨episodequery、各权重.25、全局SUM后一次clip/Adam。累计1800/201600query/各task200访问。
source/38-target/model/native/β/读写算子/LoRA/FM均不改；LR全程沿原floor1e−5，不重置warmup或任何moments。
完整ECP仅1440/1530/1620/1710/1800，1710/1800可只读bank；完整发布后可并行物化，训练不必停下等待评测。
保持已有完整ECP受控停止、最新同臂ECP拒绝旧父重放、world2/3/4迁移与真实RNG来源；不称bitwise exact。

### 16.3 原件、资源、时限与执行自主

两组各400，复用sealed scene/video/RNG与50不同teacher/task、10flow/前5动作/官方horizon/8full+392compact。
训练与数值评测来自新clean pushed detached冻结；main并行审阅，不以等待main cherry-pick阻挡科学不变的已验实现。
直接复用已验1350统计/旧捕获；新800行/PT/NPZ由实际消费者完整核，成本含加载、CPU/GPU失败和所有外部占用。
明确原数据1目录原件与新来源，只保留当前canonical路径，不另造并行trainer/adapter。

新增预计约10.2GPUh，**硬限14GPUh、新增峰32GiB**；依据刚完成同T450更新7.789GPUh，两个bank/official约2.43。
已测T world4训练7010秒，macro约15–16秒；CPU窄扩展/验证预计20–35分钟，训练约2小时，最终物化/评测/验收约45–60分钟，
可并行阶段及时重叠，预计04:20起约3–3.5小时完成。非完成时刻保证，显著超期/预算风险直接回报。
存储依据上批两臂7个ECP/三组bank及捕获终值29.55GiB，单T完整ECP约.418GiB，本批5个新ECP/两组正式产物，32GiB预留临时峰。
Sol启动前须重新live核strg01 data1 quota、两节点GPU与实际峰值估计；沿AGENTS卡数/NUMA/NCCL合同选择真实吞吐，
有适用卡优先world4，完整边界可迁移但不丢弃/重放未保存更新；bank/eval独立就绪即运行，不等待无关阶段。

真实1350父ECP迁移、第四轮/旧前缀、错mode/Git/游标拒绝、1710/1800 bank与旧1350消费者做针对性CPU检查。
现有重复continuation验证已出现实际复用需求，应内聚共享恢复/来源读取，避免再复制一套大validator；
按code-architecture-gate审本次结构增长，保留一个active执行路径，旧证据读取不是任意旧配置的开训fallback。
不重跑GPU smoke、全树检查或旧1200捕获。检查通过、push并冻结后可在本合同直接准入启动，不需第二条科学释放。
源码/CPU、实际launch、1710完整结果、最后整批或实质阻碍直接Steer主讨论活跃轮，禁止Queue积压回报；
纯工程窄修按既有§10.5隔离验证/push/newfreeze续行，任何模型/数据/标签/损失/更新/评测改变交主讨论。

执行前源码验收：Sol提交fcc23cd15cc475530c385e354670efee6bacfa12，主讨论集成275a70a7，独立新1800/旧1350/formal
共17项CPU通过（20.51秒）。新freeze核为该commit clean detached；原run/data/bank承接，没有模型/native/update差异。
run788/bank698保持现存owner，新增config300行/目标测试167行占主要增长；后段resume、packing与sampler轮次内聚复用，
architecture REVIEW无新增hard的范围接受，不为行数拆散。实际launch仍依live准入与原件，不由本段预报。

### 16.4 完成与裁决

T1350/1710/1800均154/400；1710→1800为135/19/19，1350→1800为133/21/21，MT153复用。
主讨论已独立验收新800行配对/来源与五份外部账，新增9.833677124GPUh；完整捕获/训练事件消费者继承，见findings§217–218。
本窗口未取得净收益，按§16.1结束本夜原样续训，不自动2250、重启LR或搜索中间点；U仍在1080。
1800原件完整保留，不因此称selected/final方法或视频因果资格。Owner自主授权继续覆盖主讨论的机制与修正研究，
但当前无已验证修正或新执行合同，不把旧未执行清单作为新授权恢复。

## 17. 固定T1800公共β的闭环功能分解（2026-09-29）

### 17.1 完整问题、单个干预与解释边界

Owner持续自主授权有效。T1350/1710/1800均154，原样续训停止；新行为证据仍有目标物未动/其它物体运动与子目标获取缺口。
模型已是rank128，自由公共β包含同类MT常量解；训练36任务也包含butter入抽屉/托盘、多个碗定位和搬运任务。
因此不能从旧rank16或对象完全未见直接推出扩容/加数据。冻结900训练面板已证明内容相关功能，但未回答当前held闭环中M的贡献。
本项只区分：条件修正在部署中净损害公共策略的现有能力，还是其已有必要的功能贡献而完整共同学习仍未超过MT。
共同适配允许β在单独执行时较弱，公共参数还参与教学表示；任何结果都不是独立训练责任或唯一H/M/Aq根因的证明。

唯一干预：固定已预注册末点T1800的全部参数，38个target从`(B0+M(V,L))A`变为`B0 A`。
从真实完整T1800 ECP取其自身76个公共因子，原alpha/rank/source/normalization/语言与自身观测执行完全保持；
不换A、不加载强MT权重、不重新学习β、不改M缩放参数、不计算假视频或fake native forward。
β LoRA对全部条件共用一个文件，policy仍读各自exact language/自身RGB/state；teacher ID仅作为沿用配对的元数据，0 teacher video values读取。
完整T1800 correct400与MT153直接复用。仅新增公共β的400行，0更新；不计算U/其它checkpoint、same-task-other、wrong、shuffled/reversed或Test/RL。

这是AGENTS§5允许的模型冻结、无checkpoint选择、事前登记的一次性validation事后诊断。
固定1800由原§16终点决定，不在1350/1710中追峰；本项不是正式视频因果资格controls或最终方法选点，不能据结果回头改选checkpoint。
β是共同训练后的条件分支删减，不是独立language/static训练参照。full优于β只识别该冻结M的实际作用，
还不能排除通用条件修正、证明匹配视频/时序必要性，或自动授权“先训公共底座”的旧课程。

解释在完整400与任务/行为上一起判断，不另造显著性门槛：
- β实质较强且消除部分目标选择错误：降低当前M在这些条件可迁移的假设，优先约束条件功能的迁移；不能直接归因递归覆盖或开门控/scale sweep。
- full实质较强：反对“删掉有害M即可解决”的解释，保留条件通路功能；不能因β弱就自动预训练/冻结公共项，旧条件速度/P-I等反例继续约束。
- 总数接近但交换明显，或不同task方向相反：保留局部混合贡献，不能用单分支统一根因或据此堆多臂探针。
任何结果均需主讨论结合原件形成下一具体判断，不自动重训或追加controls。本项的价值是改变修正优先级，不是凑视频收益证明。

### 17.2 来源、唯一消费者与验收

任务`operator_public_beta_diagnosis_20260929`，新root `/data1/user/ymdai/ember_runs/operator_public_beta_diagnosis_20260929`。
唯一权重来源是fcc23cd15下的`continuation1800/T/train/attempts/continuation/checkpoints/macro_00001800`，schema完整ECP保持只读。
唯一期望能力是复用现存bank/evaluator的共享LoRA读取，增加显式公共β诊断身份；不得冒名MT/T完整视频或静默放宽来源。
不创建平行policy/trainer/官方评测实现，优先复用共享因子加载及现有动态queue/trajectory capture。
保存诊断源ECP/训练Git、新evaluation-only Git、精确导出公共参数来源与本条干预；旧full/MT来源原样保留。
源码在唯一Sol独占dev实现，main并行审阅集成；新clean pushed detached freeze运行，旧冻结树与有效结果不改。
CPU应直接核真实1800公共因子导出、同一实际adapter消费者/源模式拒绝以及旧T1800/MT读取；不重复旧800捕获或逐tensor泛扫。
仅与导出映射正确性有关的公共因子核对属于实际干预验证，不扩为全模型参数检查。

新400严格复用旧scene/state/video ordinal/env及policy RNG，官方render/rotate/state/action/10flow/前5/horizon均不改。
各task50行，8full+392compact；配对元数据中的50 teacher不意味着β实际消费了视频。
新JSON/NPZ内容与PT动作前缀/full RGB由现存实际消费者验收，旧capture沿已验结果继承。
报告per-task/suite/breadth、full−β和β−MT的R/G/L/churn/Jaccard/固定8-task bootstrap，并保留所有成功集合/失败/孤儿/尝试成本。
总分、动作差或几何统计不能单独标成视频语义；主讨论按需要读6/16/31等已明确问题的被动行为原件。

### 17.3 渐进投入、资源和停止

只一组400、无需GPU物化/训练。CPU实现/真实导出/验证预计20–35分钟，official参考刚完成3卡9workers的1093秒约20–30分钟，
CPU验收约5–10分钟；整体预计45–75分钟，依据实际公共adapter复用复杂度修订，不为时限并行开额外矩阵。
预计约1GPUh，硬限2GPUh；新增峰12GiB，依据正式每组捕获与仅一份公共LoRA，不复制400份参数或原ECP/资产。
启动前由Sol live核两节点显存/CPU与strg01 data1独立quota；按实际吞吐选择卡数/replicas，常规仓库上限保持，无固定2workers模板。
所有失败与外部占用计入预算。CPU语义检查、push新freeze与live准入完成后本条即允许唯一official，无第二条许可或等待main集成。
纯工程窄修按§10.5自主处理；科学语义/来源不明或资源越界回主讨论。源码、完整400或实质阻碍直接Steer，不按分片/步数回报。
完成后停止新增计算，等待主讨论具体后继合同；没有从本项自动恢复1800后监督学习。

## 18. 同一父状态下的公共执行风险干预（2026-09-29）

### 18.1 问题、依据与唯一变量

§17已独立核得full154/β100/MT153。条件M修复大量真实目标获取，也在task26/31损害部分已有成功；
当前不能靠删除M解决不足，也不能把共同训练β当独立语言baseline。机制§56给出两风险的实际梯度区别和可失败的学习分工假设。
研究问题是：在保持全部视频功能监督时，直接约束同一公共参数的执行风险，能否使完整条件策略获得更有益的迁移？
公共β还参与teacher H/X，附加风险可能损害这项作用；这是一项待验证修正，不是β分数低便已定位根因。
旧prior/Unified/DJNFR、条件速度V/L和P/I的完整负例，以及D54小试到fresh不迁移的反例，全部继承。
新变量是显式公共执行风险；不增加模块、重训一个独立MT、冻结β、变更视频地址/Value/递归或引入task门控。

两臂均从唯一fcc23cd15同一完整T1800 ECP恢复，Writer mode均为T：
- `control`：原`L_full`，作为相同更新/事件的完整参照。
- `public_aux`：`L_full + L_beta`，额外项固定系数1，不扫权重、不将原完整项减半。

两项都在同一task、同28个跨episode query、同真实action/随机tau/noise上算完整50×7普通FM，
每task权1/4，宏步仍4task×28个独立query。public_aux多一次112-query功能读出，不冒称224个独立标签或更多meta任务。
`L_beta`仅将自身公共76因子装入同一冻结source；不运行假视频/native读取、不读取teacher标签。
其完整cotangent直接累加到公共A/B0，不能送进完整LoRA replay从而误传到M/teacher。
原`L_full`仍以原完整cotangent对生成LoRA回放，保留A执行/key/native三路中实际存在的路径及全部P/C/D/O信用。
全部source基础参数冻结。两个风险求和后做一次全局SUM、一次clip、一次AdamW；没有分步公共更新或交替课程。

### 18.2 固定学习窗口、事件与完整恢复

每臂仅90新增宏步，绝对1800→1890；原完整Adam、scheduler及floor LR=1e−5保持，不重启LR、不增fresh optimizer。
原1800行metrics与事件前缀逐行保留，新目标/分支身份另列；不得改写旧T学习历史或将public_aux称原T无变化续训。
延用每九步36task无放回覆盖一次、每task10次新条件；第五轮teacher排列固定`SeedSequence([20260928,1,task,4])`，
各task取该轮前10项，query使用原绝对visit200..209与既有跨episode规则/flow RNG，不重放1800以前事件。
教学仍原合法36任务/50视频池，明确是新的事件而非新的独立任务支持。
每臂只保存终点1890完整ECP及必要故障恢复状态；90步间隔保持。完整ECP记录损失身份、旧/新拓扑、RNG与旧指标前缀。
允许在完整父边界按既有world2/3/4逻辑权重实现选择设备，记录迁移不称bitwise exact；任务权重与Adam/LR时钟不随rank数变化。
架构/初值/帧stride/双RGB/native probe/绑定/源资产/归一化均不改。旧T/U/MT原件只读复用。

### 18.3 唯一读出和事前结果分支

两个终点各新bank400及official correct400，复用sealed T1800/MT的完整scene/state/video/env/policy RNG，50视频/task。
只比较1890 `public_aux−control`、各自相对父full1800及强MT；不搜中间点、不加U或新的公共β400。
官方policy、capture和dynamic queue保持，各组8full+392compact；完整JSON/NPZ/PT消费者及配对、失败/孤儿/全部成本按现有合同。
每task/suite/breadth、R/G/L/churn/Jaccard和固定8-task bootstrap完整报告；这是固定有界修正检验，尚无相邻节点资格或最终checkpoint选择。

另在原§12–14固定train0/12/20/32×两teacher、同28query/flow面板上，读取父T1800及两个1890的full和自身β：
同task两teacher沿原面板共用query/flow，公共β与teacher无关，因此只做24次full与12次β，共36独立功能前向；
48项full/β比较引用这些原件，不重复算相同β。0更新/0环境/无held动作标签；保存实际prediction/target/time、条件/来源、标量FM及完整−β差额。
复用原面板身份，既有900的数值不能代替未测父1800；不保存新H/X/梯度矩阵，不扩wrong/nearfoil或其它探针。
该面板用于确认功能目标实际改变了什么，不替代两组完整闭环。

- 若公共FM改善而完整闭环对匹配参照无实质收益、或收益依赖较大能力丢失，则本修正未通过，不以公共分数救场。
- 若完整策略取得有覆盖的实际增益并保留原有能力，才支持下一次有界确认；仍不能由一次继承Adam的小试声称fresh或视频必要性已成立。
- 若作用太小/任务方向混合，按未识别或有限局部收益记录；不自动延长、换父状态、改lambda/clip/LR或冻结β。
无任何结果自动授权长训练、fresh重训、最终video controls/Test/RL。主讨论独立裁决，Sol本批完即停新增计算。

### 18.4 实现、成本、所有权与实际释放

task=`operator_public_function_pilot_20260929`，唯一root `/data1/user/ymdai/ember_runs/operator_public_function_pilot_20260929`，
下设`control`/`public_aux`身份。唯一Sol01a0e70e独占dev，main由主讨论独占审阅/集成；不创建第二训练器/策略/评测器。
复用既有data/run/bank的完整父恢复、FM cotangent、native replay和共享梯度；新增目标分派应内聚，旧运行面不复制。
run.py/bank.py已有大文件，按结构guard抽取内聚的实验合同或来源检查，不能机械拆文件或留下平行fallback。
原§17公共bank消费者仍只读T1800，不为新任务静默放宽；新训练/物化来源、loss variant和parent lineage须显式验证。
CPU核真实1800 ECP/Adam/scheduler、第五轮前10视频与旧事件前缀、分支损失身份、额外cotangent只进公共参数、旧sealed读取。
只做有真实失败含义的针对性检查；首次正式宏步同时承接运行finite/真实梯度检查，不另开GPU smoke或profile。
实现push、新clean detached freeze及CPU通过后，本条即释放上述完整范围；main并行审阅，不等待额外Owner许可或集成。

实际依据：T1800末90步world4均14.62秒/macro，其中每条件FM均2.23秒；额外公共FM预计使public_aux约17秒/macro。
两臂训练预计约3.2GPUh、两bank约.6、两official约2、36功能读出<.2；总预计约6GPUh，硬限8GPUh含加载/失败/恢复。
CPU准备预计30–60分钟；训练、就绪bank/eval及读出按依赖重叠，GPU与读回预计约60–90分钟，整批约1.5–2.5小时。
新增峰24GiB：两份400条件B约16.6GiB、两个完整ECP/临时写入、捕获与源码余量；父/source不复制，诊断预测远小于1GiB。
Sol在新root/launch前live查strg01 data1独立quota及两节点GPU/CPU，按真实吞吐选择world/replicas；常规合计6/8、单节点6上限保持。
两臂与就绪物化/评测独立衔接，不为并行形式等待凑卡；不丢未保存更新。记录完整argv/env/精确Git/设备/外部限时与账。
只在科学语义/原因未明/预算越界时回报边界；纯工程窄修仍按§10.5隔离验证/push/newfreeze自主继续。
源码/完整节点或整批/实质阻碍直接Steer主讨论，禁用Queue回报/逐step心跳；主讨论不轮询分数、进程或日志。

### 18.5 完成与裁决

本批已完成并独立验收，control158、public_aux148、父T1800 154、强MT153，各400且breadth6。
aux−control为136/12/22、净−10，任务簇区间[−6.75,.5]pp；control−父141/17/13、净+4。
公共功能面板改善未带来匹配闭环收益，按§18.3结束该修正；不继续aux、扫系数、重启LR或直接fresh。
control158保留为真实单点，不自动恢复原样长训练或称selected/稳健优势。完整理论更新及原件范围见findings§221–223/机制§56.4–57。
9次外部尝试含两失败/恢复共5.630391058GPUh；control原300完成行保留、41孤儿NPZ不入正式统计，aux失败0行，均留原件。
当前训练许可全部结束。后继§19只读冻结参数，不产生optimizer更新；本条不恢复其它旧run入口。

## 19. 固定模型的已见任务闭环获取诊断（2026-09-29）

### 19.1 要改变的问题与比较边界

机制§57给出当前真实缺口：四个train任务的正确视频FM优势与八个held任务闭环平台之间，
尚无本读写架构的已见任务闭环读出。不能先认定只差迁移，也不能仅看FM便认定已有足够任务内能力。
旧版本train96的不同正负例不能替代当前T；本项不重跑旧Writer，不测试shuffled/reversed或把视频controls作前置门槛。

唯一两个冻结模型为原预登记末点T1800、原强MT300。T1800锚在§16已决定，不因本轮control158改点；
MT自身数据/时钟/权重如实复用，不弱化参照。0更新、0新FM、无held标签、无validation/Test新episode，
不运行β、aux1890、control1890、U或额外checkpoint，不选择最终模型。

覆盖当前实际全部36训练task、每task四个固定init32/33/34/35，两臂各144行、共288新episode。
四task功能面板0/12/20/32只作为预先指明的子集解释，不另外挑视频/初态或占用独立评测配额。
若T在多数已见任务获取相当/更好能力而held缺口仍在，降低全局未学会解释，优先研究新关系迁移；
若T已见任务也明显更弱，不能把主要缺口直接归为新组合，先解释条件策略在自身状态的能力获取。
方向混合或样本不足时保留未识别范围，不自动扩为每task50状态、补中间点或下一探针矩阵。
无论哪种结果，都不单独定位H/M/Aq、证明增加任务有效或授权fresh/长训练；后继须主讨论具体理论裁决。

### 19.2 冻结来源、合法条件与场景

task=`operator_seen_task_diagnosis_20260929`，唯一新root `/data1/user/ymdai/ember_runs/operator_seen_task_diagnosis_20260929`。
T唯一完整ECP为fcc23cd15的`operator_read_write_learning_20260928/continuation1800/T/train/attempts/continuation/checkpoints/macro_00001800`；
MT为`/data0/user/ymdai/ember_runs/coverage_retraining_20260920/training/mtbc/checkpoints/step_00000300/lora.safetensors`，
训练Git3ebb979b，沿stage1/MT/banks/300原manifest核来源。历史data0仅只读，不复制source/model/dataset。
新的evaluation-only源码须显式兼容这两个旧来源，不能伪造新训练Git或静默开放其它checkpoint。

任务列表固定为当前coverage36协议及原训练事件中的
`[0,1,2,4,5,7,12,13,14,15,17,19,20,21,22,25,28,29,32,34,35,36,37,38,42,43,51,55,56,62,64,73,95,96,97,101]`。
前24来自coverage24训练划分，后12是已审计并实际使用的LIBERO90支持任务，global ID=40+其原task_id。
只为这个显式注册的诊断范围使用完整36task authority；不得普遍放宽development_train/nonheld_meta角色或混入val/test/重复任务。
Teacher仍原每task50条RGB/L池、stride5/真实末帧/完整双相机，按既有`video_schedule`、seed20260928为init32..35取配对4条件；
每task四teacher不同且逐臂相同。视频已参与训练，不称held-video、不称整轮50无放回评测；MT仅保留teacher元数据、0视频值读取。
一条teacher一次生成完整38-target LoRA，无选视频/平均/重新适配。source/normalization/语言/tokenizer/完整控制规则不变。

复用`pi05_eval/scene.py`实际scene snapshot/restore消费者，新建唯一144场景注册，两模型复用同一完整model body pose、sim/controller与双RGB初始状态。
可在独立无policy预生成中保存post-dummy10 scene；若随第一个模型捕获，第二个必须恢复同一封存scene，不能只复用init ID冒充完整配对。
T/MT评测都使用官方render256/model224/rotate180、8state/7action、10flow、前5replan、成功即止；
原4suite horizon220/280/300/520、LIBERO90为当前evaluation.json的400。policy/env RNG仍seed7的原固定调度。
允许按原24与支持12拆成实际role子面板实现，全部144仍为一个预登记比较；报告不得把不同source或未配对scene拼在一起。

### 19.3 唯一运行面、来源验收与有限工程

仅新增这个明确evaluation scope，复用现有Writer物化、公共MT文件、FrozenOperatorAdapter、dynamic queue和被动采集；
不写第二trainer/evaluator，不在原冻树写入、不改变模型/native/loss/参数。bank/scene/capture从历史400到新144的二次使用，
应把task/scene/role/case注册内聚到明确scope owner，再由原消费者使用；不要再向大函数逐个堆study/窗口分支。
现有400/公共β/1890消费者保留其原界限；§18已完成pilot训练入口退役，main不再开放任何新/恢复训练，读取和物化仍可复用。
历史源码由Git/旧freeze保留，不删除ECP、已完成bank、scene或raw evidence。

CPU直接核实际T1800/MT300来源、36task确属当前已训练allowlist、4state/4video映射、144scene注册、同一adapter/capture路由；
拒绝val/test、错checkpoint/任务/teacher或未注册scope。旧400/public_beta/1890读取做有意义定向回归，
不再扫描旧1200/800捕获、不做逐tensor全模型比较、GPU smoke/profile或新model前向诊断。
新源码在Sol独占dev实现、针对检查通过、push clean detached freeze后即可按本合同live准入运行，main并行审阅集成。
若实际接口需要改变科学信息、放宽完整scene一致性或超过已登记规模/预算，报告具体边界，不静默改口径。

每新臂144正式诊断行，保存完整逐行condition/scene/RNG、success/步骤、全步被动NPZ及stage predicates；
每task init32一条full，其余compact，即每臂36full+108compact。全部新引用NPZ/PT由实际消费者读回，
报告两臂同task/state/video的R/G/L/churn/Jaccard、任务簇区间、全部per-task及原24/支持12/suite分项。
不能把144训练任务诊断当strict400资格，不与153/400直接相减。保留所有失败、未引用trace/步骤/成本，不择优恢复。
已有训练状态/旧结果的可靠验收直接继承，只新增本项相关身份与消费者证据。

### 19.4 时间、资源与停止

此项为冻结闭环诊断，预计新增约1–1.5GPUh，**硬限3GPUh、新增峰10GiB**，含scene渲染、模型加载、物化、失败和恢复。
依据400条件物化约.27GPUh、400official约.94GPUh，144×2按执行步粗估约.8–1.1GPUh，另留meta horizon/加载与失败余量；
不将该线性估计冒称实测。144份B约3GiB，full/compact捕获与新freeze/场景预留10GiB，不复制父/ECP/source资产。
CPU扩scope和真实来源检查预计30–50分钟；就绪物化/两个模型评测及时并行、读回约30–50分钟，总约1–1.75小时，超期给具体原因。
GPU前Sol live核双节点及strg01 data1独立quota/共享空间，沿当前常规6/8上限，按真正吞吐安排，不等待凑卡。
最近4workers/card已有OOM，3workers有效；按本轮真实内存/CPU余量准入，不继承固定2workers，也不无依据重试4以制造效率。
完成整组288或出现实质边界直接Steer主讨论，停止新增计算；无新训练、更多初态、视频controls、Test或RL自动许可。
Owner已有“常规卡满时短期小分析可临时超限、结束立即退出”的授权。12:25调度复核发现§19场景/两bank已就绪但正等待§20；
本次允许仅这组固定288在常规6卡已满时，live准入合格后临时加1张、总量至多7张且单节点仍≤6，按实测余量顺次完成两臂即释放。
不扩大原3GPUh/10GiB或§20总预算，不占用忙卡/干扰他人，不把例外扩到长训练、五个400或后段controls/Test；现场没有适用卡则按原依赖调度。

### 19.5 首次评测前的场景观测一致性修复（2026-09-29约11:40）

两臂尚无有效正式行。真实libero_10/task2/init32复现中，旧post-dummy观测与从其保存sim_state恢复的观测不同，
而sim_state/model_body/controller保持一致；eef位置最大差6.757e-6，RGB有4084个通道元素不同、最大差58。
同一环境池重新prime后的两次恢复及canonical快照重放在该案例一致。主讨论直接核两份JSON及实际probe脚本，
据此认定旧场景的观测/保存状态一致性合同未满足；不将具体缓存或物理积分原因、跨硬件bitwise一致或模型失效称为已证实。
证据为本root的`analysis/scene_restore_first_failure.json`、`scene_canonicalization_probe.json`及对应launch脚本/退出记录。

主讨论批准在原3GPUh/10GiB、并计§20总30GPUh/128GiB内修复首次诊断：从旧封存144场景的相同post-dummy
sim_state/model_body/controller来源重生一致观测，隔离写`scene_canonical`；不额外推进physics或更换seed/task/init/video/horizon。
144场景统一处理，逐个实际restore并通过原严格scene断言后发布；T/MT共用新scene与显式绑定它的新评测attempt。
新资产记录旧来源和实际evaluation-only Git，旧scene/bank/失败队列/费用全部保留，不覆盖原件或放宽RGB/状态检查。
旧已物化LoRA不依赖评测初始scene，应以显式lineage复用权重及必要metadata，不因场景路径变化重跑Writer或复制整套权重。
若物理状态/controller发生实质变动或新恢复仍不一致，须报告具体边界；不借修复追加episode、选模型或改科学计算。
实现仍沿唯一scene/scope/bank消费者、定向CPU验证和push后的新clean freeze，main并行审阅；该许可不涉及旧sealed
validation400或§20节点，不重封这些资产、不重训。失败、两次短诊断和新生成均计入原预算。

## 20. Owner授权1890后续训观察、冻结视频比较与有条件Test（2026-09-29约10:30）

### 20.1 最新问题、授权与时间窗口

Owner醒来交流后明确恢复执行，希望下午约15:00得到尽量完整结果。T1890原目标参照158高于1800的154，
Owner要求继续观察可能的小幅增长与连续下降，不能将既往平台误当已确认最高能力。U450/810/900/1080为143/132/132/128，
是整体回落而非每点严格下降；本轮U保持停止。公共辅助项148的阴性裁决保持，不续aux或改loss。
本次是新的有界追加授权，覆盖§16/18停止线和10:20暂停；旧记录仍是当时有效判断，不改写历史。
只检验原目标在更多更新下的实际学习轨迹，不重置LR/Adam、不换数据/seed/rank/模型、不据controls改变训练。
已见任务诊断§19恢复原固定范围，与本项并行；不可将本次最高点替换其事前固定T1800。

### 20.2 真实1890父、事件、卡数与唯一训练入口

task=`operator_read_write_continuation_2340_20260929`，root=`/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928/continuation2340`。
唯一父是`/data1/user/ymdai/ember_runs/operator_public_function_pilot_20260929/control/train/attempts/continuation/checkpoints/macro_00001890`，
训练Git9801641d0967e163d91474ff92e6fb6520be1084，mode=T/pilot_arm=control/loss_variant=full。
实际路径按原已核manifest消费，若attempt名与本段文字不同，以该唯一control1890完整原件纠正文档，绝不可接public_aux或静默放宽来源。
承接Writer/Adam/scheduler、1890行metrics与全部旧事件、完整rank RNG；累计1890→2340最多450新宏步，仍四task各28query、权重各.25。
第五teacher轮从每task第10访问继续至50，第六轮用`SeedSequence([20260928,1,task,5])`，到2340每task累计260访问。
新schema显式承接v6，旧事件逐个不变；不把第五轮重新从0开始。全局SUM后一次clip/Adam，绝对floor LR1e−5保持。
各90步1980/2070/2160/2250/2340保存完整ECP，恢复`stop_at_next_ecp.request`和预设完整边界stop；完成信息记录实际退出游标而非预定终点。
同窗口只从最新完整同臂ECP恢复，源/损失/游标/旧前缀/optimizer时钟均须真实检查。只开放本轮T训练入口，旧pilot/旧续训不恢复为可随意开训的fallback。

Owner特别要求卡数不锁死。优先GPU02四张有吞吐价值的可用A40，完整ECP边界允许world2/3/4迁移，旧rank恢复RNG、新增rank来源显式记录，
不称bitwise exact；无四卡时不等待凑卡，可用2/3卡启动。当前四条件整项分片最多四rank有工作，额外卡用于就绪bank/eval，
不让空rank占卡；本轮不为扩至六rank改逻辑batch或另造query分片器。不得为换卡丢弃/重放未保存更新。
沿两节点总6/8及单节点6的实时资源合同，短诊断可用Owner已有临时超限授权，但不借此把长训练/大official永久超限。

### 20.3 固定观察节点、停止与选择

1890的sealed400直接继承；每个新完整节点各物化400并strict paired400，原固定scene/video/env/policy RNG、每task50不同teacher不改。
节点就绪即物化/评测，与后段训练重叠；只报完整400，不读取/据分片分数改变节点。保留per-task/suite/breadth、R/G/L/churn/Jaccard与任务簇区间。
全部预注册节点上限2340；若两个相邻完整节点连续下降且期间没有新高，主讨论可要求下一个完整ECP收束，已经生成/启动的预注册评测仍完整计入，
不丢掉不利节点。若分数平台，不冒称已证持续下降；若到2340仍涨，也只说观测窗口内最高，不自动续2790。
约13:30检查一次实际完成范围与剩余wall-clock，用完整节点确定可交付范围；不为凑15:00隐去未完节点或把不足400当正式结果。
若新节点排期显著推迟，应在下一完整边界停训并保留所有已发布节点，给后段controls/Test留真实时间；不做中途分数轮询。

候选范围只有原目标T1890及本轮实际完成的预注册correct400。按最高完整成功数选单checkpoint，同分选较早节点；
同时如实报告相邻保持、覆盖和多次选点的乐观性，不声称单点独立确认。Owner本轮希望观察高点，不另强加旧资格线或要求先证明全局收敛。
选点固定在任何新视频controls/Test结果读取之前，由main登记具体ECP/Git/选择表；后者不能反向改选。
Test为条件授权：若候选相对MT只有既有1–5个成功量级优势，仍不算明显超越，不自动打开Test。
更大改善须main结合差额、任务覆盖/得失与区间明确判断并发布固定候选；不从统计显著性不足推出两者等效，也不临时降低强MT。
无明显优势时，完成视频诊断与已见任务比较，如实回报Test触发条件未满足，不以未运行冒充阴性结果。

### 20.4 冻结候选的视频比较与Test范围

新后段root=`/data1/user/ymdai/ember_runs/operator_selected_validation_20260929`；选点前可做CPU实现/实源检查，不启动依赖候选的模型工作。
冻结后复用该候选correct400，新增same_task_other、cross_suite_wrong、shuffled各400；同task另一正确视频沿既有offset17，
wrong沿原平衡cross-suite task mapping，均使用`expert_manifold/video_schedule.py`与原seed7，整轮视频不重复且逐行匹配scene/env/policy RNG。
保持目标exact language，仅视频来源改变；shuffled用既有固定permutation对stride5选中的真实双RGB帧同序重排，重新完整native→Writer，不能只扰hidden或复用correct LoRA。
本次先做shuffle，不为凑表自动补reverse。再导出该同checkpoint的公共β单份LoRA跑400，以区分内容敏感与相对共同静态分支的有益增量；
β仍不是独立训练语言baseline。时序结果只作冻结方法的事后检验，不进入训练、选点或本轮架构修正。
预算/排期不足时先完成wrong/shuffled整组，再完成other/β；未完成范围明示，不能只挑有利条件回报。

若Test条件成立，另在同root/test生成唯一完整400 scene注册，固定coverage协议Test全局ID `[8,9,10,18,24,27,30,33]`、init0..49，
使用同一选定T、原强MT300（旧训练Git3ebb979b）和本项目canonical frozen Source（三者同一基础source与norm）各400。
Source是当前过滤Source71训练后的冻结源policy，不以generic未适配pi05_base替代；MT不重训、不换弱checkpoint。
T每task50合法action-hidden teacher各一次、seed7；Source/MT保留配对video元数据但不读其值。完整scene/环境及policy RNG共享，官方动作口径/horizon保持。
保存每task/suite/breadth及三组配对成功集/区间；Test不产生梯度、不挑teacher/后改checkpoint，不据Test继续本轮训练或架构反馈。
协议本身标注historical_test_exposure=true，结果须保留这一历史限制，不称从未见过的全新Test。禁止读取Test action监督或运行RL。

### 20.5 交付、成本与协作

唯一Sol沿现有run/data/bank/scope/evaluator实现；不并行写main，不新增第二trainer/evaluator。e499f94c的§19准备先复用，
main并行审阅；新控制范围应内聚进scope/来源规则，不继续逐window复制大validator。纯工程窄修仍按§10.5自主验证/push/newfreeze续行。
CPU应核真实control1890与aux拒绝、旧metrics/第五轮余项/第六轮、world2/3/4任务权重与完整恢复、受控停止、旧sealed消费者和新scope/视频重排的实际调用方。
训练代码、评测扩展可以分别push/freeze：CPU合格的训练先启动，后段实现不阻塞GPU；新evaluation-only Git与旧训练来源显式兼容。
每次GPU前由Sol live核双节点及strg01 data1 quota/共享容量；所有新产物data1，旧资产复用，不复制大数据或擅删原件。

本次新增总硬限30GPUh/128GiB，包含恢复§19的3GPUh/10GiB、最多五correct节点、四视频功能比较及条件成立时三Test组和全部失败；不与旧已结账批次混算。
依据已测450更新world4约1.84–1.95小时/7.4–7.8GPUh、每400 bank约.27–.30及official约.9–1GPUh；
预计训练/五节点约13–14GPUh，§19约1–1.5、四controls约4.5–5、三Test约3–3.5，加加载/失败余量，预计总22–25GPUh。
新增LoRA banks约80–95GiB，ECP/capture/临时写入保留128GiB峰值上限，启动前须以实际quota核准。
目标15:00是交付预期不是性能/完工保证；10:30至15:00约4.5小时，CPU准备尽快，训练优先，已就绪诊断/物化/评测独立重叠。
明显超期风险及时说明实际缺口与优先顺序，不默默增加预算/缩正式行；选点若到后段才完成，Test可能无法在15:00前完整结束。
源码/实际launch、每完整观察节点、§19整组与最终交付及时回报。main活跃时直接Steer，若main已结束回合则只投一条可唤醒的新消息，
不得为了等待Steer保持main在线；不发重复Queue/心跳、自通知或半批分数。main收到完整交付后验收、裁决、接续。

## 21. Owner明确指定的四卡2340→2790继续学习（2026-09-29下午）

### 21.1 授权、问题与边界

Owner在看过161最高点、要求先讨论后，又明确“先让sol继续四卡续训，之后我们再讨论别的。这是我的判断”。
因此先执行同一T的下一有限学习窗口，检验末段小幅新高后能否继续获得净收益；不以主讨论此前不急追加的判断阻挡本次授权。
§20曲线158/156/152/155/156/161全部保留，既不能写成持续单调下降，也不把末点新高当成无限续训依据。
本次不改结构/目标/数据，不开展U、public_aux、视频controls、Test或RL；后段其它选择留待与Owner讨论。

唯一task=`operator_read_write_continuation_2790_20260929`，唯一新root=
`/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928/continuation2790`。
唯一父为`continuation2340/T/train/attempts/continuation/checkpoints/macro_00002340`完整ECP，
训练来源`e2afbfd7c997e3f792921600608efa2fa3c1b25a`、T/full、事件v7；不能用后续cb535c1e读取Git冒充训练来源或回接旧1890/aux。

### 21.2 学习、恢复与观察

从2340到2790共450新宏步/50400 query，36task等权、每宏步四条件各28跨episode query、每条件权重.25；
保持原SUM全局梯度后一次clip/Adam、原optimizer/scheduler绝对时钟及floor1e−5，不重置LR、不加入辅助目标。
完整2340行metrics字节前缀与旧采样事件保留；同world4恢复四rank RNG，物理设备变动如实记录，不称bitwise exact。
事件v7→v8明确只追加第七teacher轮：绝对visit260起继续原第六轮`[20260928,1,task,5]`剩余排列，
visit300起采用`SeedSequence([20260928,1,task,6])`。原task/query/flow及teacher第0–5轮事件不变。
来源验证区别首接2340父与本窗口最新同臂完整ECP；有更新完整边界时拒绝旧父重开，旧冻结spec/原件保持只读可消费。

完整ECP与correct400节点固定为2430/2520/2610/2700/2790。ECP包含Writer、Adam/scheduler、sampler/cursor、rank RNG、拓扑和schema；
保留预设`stop_after_macro`及`stop_at_next_ecp.request`受控边界。除Owner新指令、数值/恢复故障或资源硬限，
不因单个小幅下降自行缩掉已释放窗口；所有已启动节点完整计入。到2790收束，不自动再顺延450步。
评测严格复用sealed validation400的scene/video/env/policy RNG，50不同teacher/task、8full+392compact和被动trace；
旧2340与强MT153只读复用。报告每task/suite/breadth、相邻及对父/MT的R/G/L/churn/Jaccard/任务簇区间，
不读中途分数、不融合checkpoint、不因本轮分数自动发布视频controls/Test选择文件。

### 21.3 实现、资源与交付

Sol独占现有dev，从最新main沿唯一run/data/bank运行面实现新窗口，旧CLI退役为只读来源，禁止平行trainer/evaluator。
对本次实际触及的窗口/父来源声明作内聚复用，不能再逐window复制大校验分支；不把全面重构变成GPU启动前置。
若仍有有界结构例外，保留真实guard/diff与后处理范围供main并行审阅，不能冒称全过或放宽科学/恢复合同。
Owner随后纠正主讨论职责：常规工程实现、测试及恢复检查由Sol负责；main核科学语义和结果，保留Git管理但不重复工程审核或另做结构重构。
针对性CPU核真实2340父/时钟/前缀、v7→v8、同窗口恢复和旧父重放拒绝、新旧bank/捕获消费者；不为既有可靠验收重扫旧1200/2000捕获。
训练端检查通过并push新clean detached freeze后直接启动，评测读取接入与main合入可以并行，不需第二次执行许可。

Owner本次明确四卡：优先GPU02同节点四张有实际吞吐价值的A40；launch前同时live核两节点、GPU UUID/显存/进程及NUMA。
Owner后补充：后续暂不使用GPU01，除非GPU02没有满足真实作业需求和原并发上限的可用卡；训练、物化、评测及恢复均适用。
四rank各一条件，`NCCL_P2P_DISABLE=1`、既有GPU-local NUMA/deferred NCCL不变。确实无适用四卡时报告具体限制，
不抢占他人、不dummy占卡；未来Owner或现场必要迁移仍须完整ECP与原逻辑权重，不丢未保存更新。
总量遵循AGENTS常规跨节点上限；四卡训练不等于整批仅四卡，独立物化/评测利用余量及时启动。
后段就绪official按实时吞吐使用多卡/适当replicas，不再无依据沿用末点单卡；不为扩已近完成的队列丢弃有效行。

本批增量硬限18GPUh/64GiB，所有失败、加载、训练、物化、五组official均计入；旧§19/20已完成15.621703396GPUh单列不清零。
依据上批同450更新四卡7137秒/7.930GPUh、五bank合计约1.61GPUh、official含恢复约5.01GPUh，预计新增约14.5GPUh，留有限失败余量。
训练预计约2小时，重叠bank/评测后整批约2.5–3小时，源码准备和实际资源另计；不是原15:00交付保证。
五套新bank约40–45GiB，五ECP与捕获/临时写入按新增64GiB上界准入；Sol在strg01重查data1独立quota/实际用量与共享容量，
父ECP/source/dataset只读复用，不复制大资产。实际argv/env/cwd/source/GPU/时限登记在本root唯一launch合同中。
报告限必要源码交付、实际正式启动、完整可裁决节点及需主讨论决定的实质阻碍；日常调度与已解决故障并入完整结果，
不发确认、定时心跳或重复回执。main完成独立工作后结束回合，实际交付再接续；本项进行期间其它事项与Owner讨论。

## 22. 续训期间的冻结原生读取诊断（2026-09-29，Owner已授权并行分析）

### 22.1 问题、干预与可失败解释

Owner明确“可以派发分析实验……现在开始趁正在续训，你自由进行并行的分析”。§21四卡续训保持优先；
本项解释已学视频→LoRA函数，不改变正式学习、不提前选点、不开最终wrong/shuffled/other400或Test。
最近条件速度方案的教学端不读公共β，且条件作用只在action_out；当前T/U在教学端用公共β产生X/H，
在全部38target生成M，并由同一A（T）或独立S（U）写入地址。多个结构变量同时改变，不从跨方案分差唯一归因。

先检验尚未被现有β执行消融回答的一项：已共同学习的公共β用于教学原生读取，是否给当前生成的LoRA带来功能收益？
对每个冻结模型，正常Mβ=Fφ(A或S;Xβ,Hβ)；干预M0=Fφ(A或S;Xsource,Hsource)。
只在read_native_video的公共因子参数替换为实际source identity，视频/语言/probe/帧序保持；
writer的公共A/B0、独立S（若有）、P/C/D/O与最终执行公共β都保持原模型，最终仍是(A,B0+M0)。
不能将整套writer.common清零、把执行β也去掉、重绑U、注入query信息或使用假视频。
这识别冻结函数对适配后教学特征的依赖，不等价于“从头冻结读取”的训练反事实，也不单独识别T绑定的收益。

事前分支：若source-native显著损伤多数条件，支持共同适配后的原生响应确有功能作用；
若差额小或部分系统改善，降低“公共β教学读取是当前优势的主要机制”的支持度，并定位条件差异。
T/U900的差额可作同学习阶段的交互观察，T2340反映当前较长学习状态；不能仅由交互差指定独立因果根因。
即使正常读取胜出，也不证明关系/时序语义或held收益；不据此立即追加冻结训练、gate或新架构。

### 22.2 固定面板、原件与分析

唯一task=`operator_native_reader_diagnosis_20260929`；唯一新root=
`/data1/user/ymdai/ember_runs/operator_chain_diagnosis_20260929/native_reader_transport`。
冻结T900、U900来源81846ed3；T2340来源e2afbfd7。三者实际完整ECP读取，权重不更新、不重选点。
沿§14A固定train0/12/20/32，每task原两teacher、同28跨episode query和flow；全部数据来自已授权训练任务。
新执行24个source-native功能条件，另T2340正常full8条件及task公共β4条件，共36个新FM前向。
T/U900正常full16和各自β共8沿§14A原件引用；旧source/MT参照也沿原件引用，不重算。
source-native的同8个视频响应与冻结source/probe相同，可在三模型间复用，保持明确来源，不制造重复缓存资产。
正常T2340与干预都沿已有native/Writer/paired_functional_credit消费，完整50×7 FM及前5另读；0 optimizer更新、0环境episode、0held标签。

保存每条件prediction/target/flow time、实际视频/查询身份、全部38个M；保存正常T2340与source-native的H及Q8/V8/out三处X，
source-native特征允许一份原件多处明确引用。完整teacher特征不必为三模型重复写三份。
action_in的X为同一固定probe，读取干预不应改变它；action_out的公共投影输出被丢弃，不将38个公共target都称为有效原生作用路径。
CPU读回实际新36个预测原件、对应特征/M及范围；不逐tensor复扫旧模型/80PT或旧official捕获。
输出逐task/teacher和三模型差额、前5与其余45；对d=p_source-native−p_normal，核精确恒等式
ΔFM=2 mean((p_normal−y)d)+mean(d²)，保留反向或接近零结果，不只报均值。
特征/参数变化大小仅供定位，不把它称为语义或闭环性能。已见任务/旧held表现只作为已有背景，不产生新资格。

### 22.3 执行边界

Sol在原独占执行范围内准备一次性诊断脚本，复用clean frozen数值代码与canonical资产，保存实际脚本/命令/来源/退出/成本；
不扩canonical trainer/evaluator，不为临时分析搭新测试体系，主讨论负责科学解释而不重复工程审核。
新增独立上限1GPUh/4GiB，包含所有加载、失败、临时写入；与§21的18GPUh/64GiB分别计费并合计存储准入。
依据§14A的80个功能前向/部分反向合计.099GPUh，此项GPU预期5–12分钟，连CPU准备预计20–35分钟；资源/文件系统等待另计。
优先GPU02的一张适用空闲卡，只有GPU02无适用卡才考虑GPU01；不挤占四卡训练、不停已有任务。
现场live两节点及strg01 data1独立quota检查由Sol负责；就绪正式bank/official不因无依据串行等待此诊断。
完成后只直接Steer整批结果或实质阻碍，无确认/心跳；停止新增诊断，主讨论结合已有历史和实际数据决定是否还有必要分析。

## 23. 条件作用从训练读出到自身动作生成的传递（2026-09-29，续训并行分析）

### 23.1 要改变的判断

Owner指出仍有关键未知，不能把一项诊断收束等同于没有可并行研究。§22结果已消费，不重做读取替换。
当前具体缺口是：T/U900的训练FM排序与held闭环相反，当前视频条件有训练功能收益，但该收益是否保留到
同一真实观测下的官方10步动作生成尚未测量。先固定观测与任务，区分条件函数在生成过程中的调用，
不把通用FM/采样差距本身当EMBER根因；必须比较完整视频策略、各自公共β与强MT的差异及T/U交互。
本项独立于2790学习结果，不改变§21，不使用held动作、环境反馈或最终wrong/shuffled controls。

唯一task=`operator_sampler_transport_diagnosis_20260929`；唯一新root为
`/data1/user/ymdai/ember_runs/operator_chain_diagnosis_20260929/sampler_transport`。
冻结T900/U900（81846ed3）及T2340（e2afbfd7），面板沿§14A/22的train0/12/20/32、两teacher、各28跨episode query。
正确视频生成的38个M沿已存原件复用，并与各自真实ECP A/B0合成；不重读视频、不重新编译、不重选query。
T/U900原件见train_functional_900，T2340正常M见native_reader_transport；若原件不含所需字段，先报具体缺口。

### 23.2 有限实验与精确分解

复用现有policy.predict_action_chunk/sample_actions的官方10步路径、query图像/state/语言预处理与冻结source normalization。
每task复用原query的flow_seed产生一套28×50×32纯Gaussian noise；三模型、β与MT使用同一实际noise，
沿现有logical-batch RNG消费者保证microbatch不改配对。部署采样从纯噪声开始，真实action只用于事后评分，
不得作为采样状态、Writer条件或生成过程的插值输入。保留真实32维suffix，最终评分仅真实7维。
MT沿原300 mixed-precision LoRA，共四task，只计算一次并供三模型引用；不新增source参照以填矩阵。

24组full、12组公共β及4组MT，共40条批量生成路径，每条28query、10个真实flow step；不是40个环境episode。
对24组full另在对应公共β的每步实际suffix状态上计算full速度，共240个额外速度批次；全部上限640个速度批次。
不另开自定义solver，不改time/step/precision，不用预测均值替代每条实际生成路径。
记录本模型full路径zF、公共路径zB、各自速度vF(zF)/vB(zB)，以及vF(zB)。对于dt=-0.1，精确有：

```text
delta_z_(k+1) = delta_z_k + dt * (direct_k + feedback_k)
direct_k = vF(zB_k) - vB(zB_k)
feedback_k = vF(zF_k) - vF(zB_k)
delta_z_0 = 0
```

这是以公共轨迹为参照的有限函数分解，非对称、不是参数Taylor展开、梯度冲突或独立因果中介百分比。
direct包括同一suffix输入下整套条件LoRA对hidden与速度的作用；feedback包括模型前面生成不同suffix后自身响应的变化。
不把它叫环境闭环反馈。统计最终前5动作、完整50及真实未padding future的归一化动作MSE，按task/teacher/query报告；
公共β与MT按task共用，不当作两个独立样本。single noise/有限训练面板不证明多模态动作的正确性或闭环排名。
主讨论另从既有FM原件读flow_time分层、前5/后45差额，作为相关性背景，不把不同tau下不同query当同query因果干预。

保存实际脚本、合同、LoRA/M来源引用、query身份、noise、真实action/有效future长度（仅训练评分）、40条生成路径的
z/velocity/最终normalized动作，以及24条cross-velocity；full/β路径另保存Q8/V8/out的Aq，用于和已有教学key/M读出联系。
无需复制ECP、所有M或原图像；不扫描旧official原件。CPU逐个实际读回新增预测、配对、步数及上述递推，正常浮点容差即可。
报告完整−β、完整−MT及T/U的前缀/全长差额与反向条件，并给direct/feedback的合成量和误差方向，不靠范数宣布有益。

### 23.3 结果分支与执行

若完整视频的功能优势到真实生成前5仍保留，降低“视频功能仅在训练插值上有用”的解释，后继优先区分关系迁移与
环境自身访问状态；若优势在生成前缀消失或反转，且T/U呈与既有功能不同的交互，优先解释条件算子沿自身生成状态
的调用与训练信用错位。单纯全模型都同样改变、仅均值微小差或不一致结果，不作为EMBER特异根因或改loss理由。
无论哪种结果，都不自动恢复旧endpoint/mean5辅助、扫采样器或开展fresh训练；必须结合历史反例和已有closed-loop证据裁决。

Sol负责一次性脚本与执行，复用clean frozen数值代码/唯一policy消费者，不新增canonical trainer/evaluator或机械测试套件。
主讨论负责科学解释，不承担重复工程审计。独立硬限1GPUh/4GiB含加载/失败/临时输出；不占§21预算但须合计quota准入。
依据§14A 80组混合功能前后向约.099GPUh，本项更多速度前向但复用教学编译，GPU粗估15–40分钟、准备/读回共30–60分钟，
资源等待另计；外部时限不得超过剩余GPU预算。若无法在限额内完成，保留已完成原件并报告，不删条件或追加预算。
GPU02优先，只有无适用卡时考虑GPU01；沿Owner短分析临时超卡例外，不挤占在跑四卡训练或中断official。
launch前由Sol现场核两节点资源和strg01 data1 quota。完成或实质阻碍直接Steer，禁止心跳/确认消息；完成后不自动开下一项。

## 24. Owner直接授权最佳验证checkpoint的Test，复用已有MT/Source（2026-09-29）

### 24.1 最新授权与唯一选择

Owner判断短期可能难有更大突破，明确“可以直接看看最佳ckpt在test上的性能了”，随后纠正MT和Source已有结果。
因此本条覆盖§20的“明显超过MT才触发Test”和§21暂不Test边界：只新增一个T checkpoint的400 Test，
旧MT/Source Test直接复用，不新跑基线、视频controls、其它checkpoint、训练或RL。

唯一task=`operator_selected_test_20260929`，沿已登记后段root
`/data1/user/ymdai/ember_runs/operator_selected_validation_20260929/test`；selection仍是其父目录的`selection.json`。
不得另建平行evaluator或为Test重新编译validation400。已有冻结ECP、源模型、数据与旧原件只读引用。

选点只用T/full同一学习链的完整预登记validation400，成功数最高、同分取较早macro。
已完成1890/1980/2070/2160/2250/2340/2430/2520/2610/2700/2790分别158/156/152/155/156/161/147/154/160/154/159；
最后2610已收束且main直接核400行/配对/完成，所有预登记点均有完整结果。更早已验T点均低于161，U/public_aux不入候选。
**main最终选择T2340（161/400）**，训练来源`e2afbfd7c997e3f792921600608efa2fa3c1b25a`，唯一完整ECP为
`/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928/continuation2340/T/train/attempts/continuation/checkpoints/macro_00002340`。
对应correct bank为上述continuation2340根目录下`T/banks/2340/manifest.json`、完整结果为`T/evaluation/2340/correct400/results.json`，主讨论已核源及exit0。
精确selection载荷保存于既有tmp `operator_chain_diagnosis_20260929/selected_test_selection.json`；Sol在本批存储准入后按该内容发布到
上述唯一selection路径，直接执行，无需第二次GPU许可。selected_by=science_main，实际落盘者可在run contract说明。
选点在任何新Test模型推理/成绩之前冻结，读Test后不得更换checkpoint或据此改本轮方法。

### 24.2 复用基线与Test比较口径

主讨论已直接读取旧两组各400原行和合同：
`/data0/user/ymdai/ember_runs/coverage_baseline_test_20260923/evaluation/source_1000_test`为75/400，
同根`mtbc_300_test`为121/400；Source1000是项目过滤Source71训练后的冻结源policy，MT为原强MT300。
保持这些历史分数/合同，不重测、不重训、不换较弱基线。旧数据0新增写入，所有新产物data1。

Test沿coverage固定8全局ID `[8,9,10,18,24,27,30,33]`、每task init0..49、inference seed7、exact language，
原render256/model224/双RGB rotate180、8state/7action、10flow、前5replan、dummy10、成功即止、horizon220/280/300/520。
与旧Test使用同一canonical source/normalization/tokenizer及同一policy-noise调度；不用generic pi05_base替代Source1000。
T每task50个合法action-hidden teacher各用一次，沿既有video_schedule的固定seed7映射；不挑视频或改变query/teacher来源。
仅读取Test教学双RGB/语言作一次LoRA编译，禁止Test action/proprio/reward/terminal进入Writer或任何梯度；执行自身state合法。
模型/Writer/native/损失/权重不改，只增加有明确role/source的Test读取与评测范围。

旧Test合同没有注册完整scene快照，也没有NPZ/PT capture，无法事后升级为严格完整RGB/sim-state配对。
新T沿旧合同的`seed/reset/set_init_state/dummy10`初始化，不为补齐当前scene合同重跑两baseline或构造它们不存在的初态证据。
若当前operator adapter强制registered-scene，沿唯一owner增加**显式legacy Test初始化scope**，复用已有普通Test episode消费者；
这只适用于本条指定旧Test比较，不静默放宽validation400的strict-scene断言、不改变其恢复规则。
报告可核的task/init/语言/env seed/policy noise共同前缀配对，另明确完整scene/RGB无法与旧基线核对；
配对得失与任务簇区间仍可报告，但不能称当前validation式的strict full-scene paired400或bitwise一致。
`historical_test_exposure=true`沿旧协议保留；Test仅评价这次固定方法，不参与梯度、视频选择或后续设计反馈。

### 24.3 执行、证据、资源与收尾

Sol独占dev，沿唯一bank/scope/adapter/evaluator增加必要Test读取；保留旧数值来源与新evaluation-only Git的显式对应。
CPU检查实际选中ECP/role、400 task-video条件、旧Test源/norm/预处理/RNG、禁止错误checkpoint/标签来源；
检查直接消费者，不添加整树/逐tensor扫描或重验旧训练。push clean detached freeze后按本条直接运行，不等main工程审核或集成。
formal launcher登记精确argv/env/cwd/设备、输入与选择、输出、并发和预算；纯工程窄修按§10.5保留原件/失败账、自主验证新freeze续行。

唯一新增T bank400条件及T Test400行，沿已有dynamic long-first persistent queue多卡并行。
每task一条full、其余compact，共8full+392compact；保存新400 NPZ/PT及stage predicates，CPU实际读回必要动作/初态/谓词/RGB。
旧baseline原行/可靠验收直接继承，无不存在的capture补扫。汇报三者per-task/suite/breadth、T对两baseline的R/G/L/churn/Jaccard与
固定8task bootstrap区间、选择表、来源、退出、失败及全部成本；注明验证选峰乐观性与历史Test曝光。
失效仅重领原队列缺失分片，不择优重跑；如果真实原因涉及模型数值/数据标签/评测改变或预算，先报告科学边界。

新增独立**硬限3GPUh、峰值16GiB**，与§21的18GPUh/64GiB分别计账，全部失败/加载/物化/评测计入。
依据最近400 bank约.27–.30GPUh、400 official约.9–1.2GPUh，预计约1.3–1.7GPUh；新增一套400权重约8–9GiB，
其余场景记录/capture/冻结代码/原子临时余量纳入16GiB。launch前由Sol核strg01 data1独立quota、实占及共享容量，不能仅看df。
GPU02优先，只有该节点无满足作业需求的适用卡才可GPU01；live核双节点，沿既有常规总量/单节点上限，不借短诊断例外扩本Test。
§21已全部收束，Test准备直接开始；候选已固定，及时利用空闲卡，bank就绪即official，不沿用无依据单卡/固定2worker限制。
按既有bank/eval实测，从源码准备到完整读回预计约1–2小时，现场资源/必要修复另计；重大延期或硬限风险直接报告，不发定时心跳。
完整Test或实质阻碍直接Steer；当前本条不包含新controls、更多Test seed、追加训练或基线GPU计算。

## 25. Owner恢复自主后继：固定T2340的视频证据与Test后的研究分支（2026-09-29）

### 25.1 权限、分支和固定对象

Owner最新明确离开期间由主讨论自主推进：Test若T明显胜MT，补视频特异性后推进以前规划的RL与one-shot比较；
若接近或不如MT，仍做视频分析，再以深入理论、实际特征/算子干预寻找改进并重训，不停在一个貌似合理的原因。
本条覆盖§24的后继暂停与“Test不能影响任何设计反馈”旧句：Test整体结果允许决定上述研究分支。
不改变已冻结T2340、不给本次Test换checkpoint/挑teacher，不让Test标签进入共享训练；后续研究保留历史Test曝光，不能再称全新盲测。
独立task-local适应若进入执行，目标标签只作用于隔离的task-local参数，另列适应成绩，不能回流共享Writer/source。

“明显胜MT”由完整差额、各task/suite得失与覆盖、区间及选择历史共同判断，不恢复旧+40/400硬线，
也不把多一两条成功或一个suite的集中增长写成普遍优势。区间跨零不等于两者等效；尚不清楚时保持这一结论并优先机制研究，
不为了打开下游而降低参照。当前尚无T Test结果，本条不预判分支。

§24唯一T Test继续、基线MT121/Source75复用，原3GPUh/16GiB范围不变。两分支都需要以下视频证据，
故当前即可准备；Test全部GPU退出后直接衔接本条GPU任务，无需再等main/Owner许可。必要工程修复由Sol按§10.5自主处理。
主讨论后续科研裁决无需再次询问Owner，但每个新增执行仍须有明确的科学变量、规模、预算、停止条件和实际承接。

### 25.2 本次唯一新增执行：冻结候选validation视频四对照

task=`operator_selected_video_controls_20260929`，复用
`/data1/user/ymdai/ember_runs/operator_selected_validation_20260929`及已经固定的selection；
各臂沿现有`selected_scope`的canonical bank/evaluation路径，不另建平行运行面。训练源仍为§24的e2afbfd7完整T2340 ECP。
原correct161/400及MT153/400只读复用；新增加以下四组各400，不再训练、选点或增加Test模型：

- `same_task_other`：同task视频ordinal固定offset17，全50视频各一次；只重绑定现有正确视频因子，不能挑有利视频或重编相同LoRA。
- `cross_suite_wrong`：目标exact language不变，沿§20已固定的平衡跨suite donor映射和seed7；给真实双RGB，完整native→Writer编译。
- `shuffled`：同一正确视频stride5抽中的真实双RGB共同重排；使用既有固定frame_order_seed/permutation，完整重新前向，保留帧索引/变换原件。
- `public_beta`：同一2340 ECP的公共A/B0，单份LoRA，无teacher RGB读取；video ID只作配对元数据，不复制400份公共权重。

只有wrong/shuffled各新增一套400因子。正确视频50条/任务、原validation8/init0..49、exact language、scene、env/policy RNG、
官方预处理及动作消费者保持；other逐行视频不同且全轮无放回。这里使用原validation strict full-scene，
不能把§24的legacy Test初始化例外带入，也不能因为旧scene个别恢复失败而放宽断言或重封这一组scene。
0 optimizer更新，validation/Test动作或reward不进入生成器；不补reverse、更多seed或其它checkpoint。

解释对象事先分开：correct/other给同task视频更换的能力交换；correct/wrong给对目标内容来源的功能敏感性，
但跨suite还改变视觉分布；correct/shuffled只给冻结方法的顺序依赖，分布外扰动损伤不等于正确动作过程理解；
correct/β给共同训练静态分支之外的当前条件功能，不把β叫独立语言baseline。
总分相同仍报告R/G/L，wrong更差也不替代correct相对β/强MT的有益增量；其它checkpoint旧诊断只作有边界的背景。
本组是validation证据，不将其移植成Test视频必要性；若后续需要对Test优势作同分布归因，由main明确最小补证范围与成本。
shuffled/reversed不进入训练、loss、checkpoint选择或架构修正；无论有无顺序差额，都不能直接反向增设时序模块。

### 25.3 执行、预算与交付

Sol独占dev完成现有selected接口的必要实现/针对性消费者验证、push clean detached evaluation-only freeze，然后执行。
main不重复做工程测试或要求等待源码集成。旧源码、有效原件、失败与真实来源保留；不改变旧Test正在运行的冻结树。
现在可CPU准备；Test全部GPU退出后，按当时适用卡及时并行两新bank与已就绪other/β评测，不等待无关臂。
GPU02优先，仅其没有适用卡时考虑GPU01；实际worker/卡数由live显存、CPU、吞吐决定，沿正式并发上限，不占位或为占卡中断有效作业。

新增独立硬限**6GPUh、峰值32GiB**，所有加载/失败/恢复计入；不借§20或§21未花完的预算隐形扩量。
依据近期每bank .27–.30GPUh、每official .9–1.2GPUh，预计总4.2–5.4GPUh；两bank约16–18GiB、四组捕获/日志与原子临时余量纳入32GiB。
Test新增资产继续存在，launch前须将两批尚需空间合并核strg01 data1独立quota及共享容量，不能只核本条孤立预算。
从Test资源释放起，利用多卡和依赖重叠预计约1.5–2.5小时取得整组读回；源码修复或实际资源约束会改变预期，重大越界及时报告。
硬限确有风险时先交具体剩余范围/成本，不缩400、不丢失败账、不只保有利臂。

四组完成400行/退出后报告各task/suite、breadth及correct对四臂的R/G/L/churn/Jaccard、固定8task簇bootstrap和不利个例。
新1600 JSON/引用NPZ由实际消费者验收；PT核动作前缀与每臂8full的完整初态RGB，区分compact/full与实际核验范围。
旧correct/MT消费者沿可靠验收继承，不复扫；孤儿/失败单列，不计正式分母。保留selection、bank、来源、变换、run_contract、raw、completion和全部外部成本。
Test完成仍及时直接Steer主讨论；本四臂一次整组完成或实质阻碍直接Steer，不发定时状态/半批分数/自通知。
本次Sol具体执行到四对照交付止；RL/FT或修正训练由主讨论依结果另给具体合同，无需Owner重新逐项许可。

### 25.4 已恢复的后继目的与不能机械继承的旧细节

主讨论已完整复读封存`paper_experiments_design.md`§3–6及`coverage_retraining_design.md`，
确认one-shot指单support示范适应比较，RL指生成后task-local A/B学习，不是把旧共享Writer FM+RL联合课程自动恢复。
单support FT允许对照读取同一episode的action/state，而EMBER只读RGB/语言；旧方案另有独立选点episode动作，
因此若保留它必须明确额外信息成本，不能声称FT总共只接触一条示范。不能用Test闭环调FT超参或选适应checkpoint。
RL保持Writer/source冻结、fresh局部optimizer、无FM混合，主横轴真实环境控制步；训练与评测初态隔离、零步/适应成绩分列。
旧rank16与当前T生成rank128不同，须采用当前真实38-target/缩放与可比局部自由度，不能静默压缩或重分解生成因子。
旧task-local方案尚未执行，历史共享RL实现已退役，不能把它或可读文档当作本方案已有运行面；
在train范围验证实际动作概率、完整10flow梯度、终止mask和恢复，再按实测成本定正式规模。
这些是后继具体合同需要承接的科学工作，不新增一套机械逐模块审批，也不因历史未执行而认为路线必定有效。

弱优势分支的研究要区分：视频证据本身不足、已编码内容经M与自身Ah调用不合适、以及跨episode功能信用没有教会关键选择。
已有事实已排除“所有条件分支都没有作用”“所有视频都只是同样通用修正”等强说法；β较弱、平均FM改善、固定query夹爪缺口
均不能单独区分上述解释。后继先找能改变完整方法选择的最小干预，保留原公共辅助FM阴性、旧LocalField/ProcessPullback功能消费者等近邻反例。
修正须同时给出特征/算子如何改变、梯度怎样教会、部署怎样使用的可失败预测，以及同预算原方法对照的实际能力验证。
中间指标改善而闭环无收益应收缩该假设；经验涨分而机制未区分就如实记录经验收益。不无限追求唯一根因，也不以尚有未知为由停止研究。

### 25.5 Test后裁决与Owner再次确认的重训前提

§24现已完成并由主讨论核验：T115/MT121/Source75，具体任务交换与比较边界见findings§238。
进入机制定位和修正验证分支，不启动旧one-shot/task-local RL，不改选T或重评基线；§25四对照沿原授权继续。
Owner再次明确正式重训之前要找到机理、机制、特征层原因并尽可能验证修正有效。
因此候选修正须先把实际特征的缺口、算子怎样造成该缺口、真实梯度能否教会所需作用连成可失败预测，
在non-held范围用有限干预检验，并与原方法的匹配短学习/能力对照区分机制改变、一般再训练收益与能力损失。
这允许有界修正验证中的学习，不等于先启动整条fresh训练或用一次loss下降为其补理由。
Test行为用于如实解释本次完整结果；不读取其专家动作、不按Test逐个失败实例调模型或选择新点，修正证据须在合法训练/开发范围建立。
若只看到内部变化而无预期功能效应，或短学习只改善代理而损害能力，应修订/关闭该假设；
不通过不断换rank/scale/LR、扩大面板或改架构名称保护相同解释。机制未全部唯一识别时保留竞争解释，不将猜测写成根因。

## 26. 完整功能信用经实际Writer更新后怎样传到独立query（2026-09-29）

### 26.1 问题、历史与范围

§24/25已完成，Test115不胜MT121，validation correct/other/wrong/shuffle/β=161/150/65/1/103。
本条依Owner自主机制分析与“验证修正后才正式重训”的授权，只做训练侧信用与极小临时更新实验。
**不是新正式续训、fresh模型、checkpoint选择或已经选定前5 loss。** shuffled不参与本项推理或数据选择。
main用实际H/X/权重取消末层擦除的有限反事实未得到一致功能改善，见机制§63；不因此开启memory/scale搜索。

待区分的完整问题是：训练的完整FM信用已偏离另一episode的实际短动作风险，还是实际Compiler的参数联系/更新
破坏了原本相容的功能信用；若均未支持，则不得把FM horizon或Compiler梯度几何作为修正根据。
旧D2/D3已测t1前5辅助、虚拟/短学习，SEOD/GOMQ也有真实端点学习及保持阴性；本次不把这些概念当首次提出。
不同点仅是当前T2340真实J与Adam、随机time的完整FM/前5分解、独立episode及两条正确视频的实际有限传递。
任何局部正例仍须后续匹配短学习和闭环收益才能放行正式修正重训，不能靠余弦或MSE越过此限制。

唯一task=`operator_functional_credit_transport_20260929`，输出
`/data1/user/ymdai/ember_runs/operator_chain_diagnosis_20260929/functional_credit_transport`。
原e2afbfd7完整T2340及原Adam/scheduler为唯一父，数值读取可复用当前clean pushed f207a12e冻结树；
旧权重、优化器、原件均只读，不重新打开原正式训练CLI或旧输出路径。
只用固定train0/12/20/32，不加载任何Validation/Test标签或专家；Source冻结。

### 26.2 固定输入与两个episode集合

A沿§14/22/23的每task原两teacher、28个跨episode query/flow，不换难例，不重复旧正常预测作为新结果。
需新反传时允许相同query前向，原正常FM/生成预测作为实际消费者校准依据。
B为每task50条中排除原两teacher和A的28条query episode后，**全部剩余20个episode**，每episode一个query。
按demo升序，独立`SeedSequence([20260929,26,task])`依次在`[0,episode_length−2]`选frame；
flow/noise用既有seed函数，visit=2600、optimization_seed=7、实际B20的demo/frame列表，逻辑batch=20。
B在任何新分数前写入合同；同task各模型/两个teacher/有限干预共用对应noise/time，不能复用A episode冒充独立query。
保留真实valid-future/repeat-padding元数据，不修改原训练50×7损失。Teacher始终只进真实有序双RGB/语言。

两组j=0/1分别取四task各自第j条teacher，task各.25，一组逻辑112个A query；
它们各从同一父独立开始，不是两个依次训练宏步。B同时检查原teacher及同task另一teacher，明确同视频/跨视频信用。

### 26.3 实际余切、完整重放与有限更新

对八个父condition分别取得：

- A的完整50 FM LoRA余切`a50`和前5 FM余切`a5`，保持原随机time、noise、真实七维与物理microbatch offset；
  用`a45=(a50−.1 a5)/.9`表示同一次风险分解，不再为尾部独立跑一套相同模型。
- B的**真实10步生成后前5×7归一化动作MSE**余切`b`。这只是训练动作风险，无环境/闭环资格含义。
  使用真实完整50×32 latent、原noise、10步Euler和执行LoRA载荷精度；首5有目标，其它位置可通过真实采样图间接影响。
  优先复用`writer/flow.py`等已有消费者；在固定小批与实际`predict_action_chunk`核数值语义，
  若旧辅助函数的dtype/载荷处理不匹配，沿实际消费者作有界修正或报告，不能静默换成另一精度/采样器。
- 每个余切经当前`runtime.compile(...retain_native=True)`的完整VJP传到真实θ，包含公共A/B0及P/C/D/O。
  不detach β→native；同一A各路正确相加，不为action_in/out本来不存在的路径补依赖；Source参数无梯度。

保存必要LoRA余切、实际θ分组/总梯度与科学读回；不要保存全Jacobian、全参数谱或重复全树审计。
每组四task等权汇总后，分别报告原生LoRA余切联系、经过真实J的联系、公共/Value参数组及同/另一teacher的传递。
Adam二阶量得到的正对角度量只能作为局部解释，与实际momentum/clip/weight decay区分；
实际有限位移的`g_B·Δθ`预测与真实B风险差必须同时呈现，不能把正预条件SGD公式冒称原Adam精确更新。

有限干预固定为每组j的两臂，**共四次相互独立的临时单宏步**：
`F`使用原完整FM；`P`仅将本次风险改为前5 FM的独立均值。每一臂都从同一真实T2340完整Adam/scheduler、
旧RNG/绝对2341时钟起，原四task×28query等权、global SUM后一次原clip/AdamW与绝对LR floor；
不重置momentum、不扫LR、不调范数、不跨组接续。临时更新可在隔离内存/本诊断root中执行，绝不覆盖父ECP。
生成因子、native、Key和Value均按更新后的真实参数重编，不能仅改末层M或把自由LoRA优化冒充Writer学习。

对四个临时状态各计算B的八condition（四task×两teacher），及其各自A组的训练FM，核对实际变化。
父B八condition、父自身β四task、强MT300四task只各计算一次，作为新B面板的共享参照；
因此B共**48条实际生成路径**（父8、临时32、β4、MT4），不是48个独立task或环境episode。
同时报告B full50/first5/valid-future及运动6维/夹爪维误差、实际符号与条件间得失；不把轨迹单一标签等同唯一正确控制。
所有条件保存prediction/target/noise、query来源及临时参数位移/必要优化器统计，可复算实际风险和一阶剩余项。
这些临时权重仅为诊断原件，不发布正式训练checkpoint，不作为resume/选择/official/Test候选。

**随本轮有限更新核实一项实际精度线索，不另增GPU矩阵。** main定向读取2340→2790的Q8/V8公共A，
分别有93.98%/93.89%元素在450更新后相同，实际B0仍有变化，C/O为FP32且继续改变；
真实2340 Adam slots中Q8/V8的m/v也为BF16，LR=1e−5、betas=(.9,.95)。这不是已证明梯度被舍入或根因。
在本来就要执行的四次实际单步中，记录这些组的非零梯度、真实Δθ和dtype，并用**同一已clip梯度、同一原Adam状态数值**
在CPU作FP32算术影子一步：量化前位移、回转原dtype后的位移、真实Adam位移分开，报告被存储舍入抹去的比例/能量及`g_B·Δθ`。
只是一项实际更新的算术对照，不把影子权重用于新模型推理、额外训练或dtype/LR搜索；原两臂的数值合同不改。
不把action_out中无直接七维监督的25/32行不变当精度故障。若非零拟议位移未被明显抹去，关闭这一解释；
若有具体有效学习损失，再由main决定同前向精度、仅高精度累积的匹配有界学习验证，不能直接重训。

### 26.4 结果分支与边界

1. 若功能信用在LoRA端已与独立B的实际生成风险冲突，且前5临时更新在B及另一teacher上有一致有限改善，
   才提高“功能监督分配值得修正”的支持度；需保留全50/运动/其它task代价，后续仍须同预算原方法短学习与闭环对照。
2. 若裸功能余切相容、实际Writer/Adam后的作用变坏，且有限干预支持，优先定位J中实际参数组与教学—执行特征联系；
   不自动投影梯度、冻结β、解绑A或换optimizer，旧public_aux/回绑与共享学习反例继续有效。
3. 若临时更新只改变A、不改善B/另一teacher，预测不稳定，或实际步幅导致差额无法可靠区分，明确记为不支持当前修正，
   不自动扩大步数/提高LR/扫描参数/再做同类余弦图。下一选择由main结合本结果与整体能力作出。

三个分支都不能凭一次局部梯度或MSE宣布held根因。测试只缩小目标分配与实际编译学习联系的竞争解释，
没有新增环境、formal correct400、视频controls、Test、RL、长期续训或fresh重训许可。

### 26.5 执行与资源

Sol负责一次性脚本、完整实际消费者/临时更新正确性、clean数值来源与运行，main负责推理和科学验收，不重复常规工程测试。
已有数值代码可直接复用；如确需保留窄代码改动，按原独占dev→针对性验证→push新clean freeze，不改旧冻结树。
不新增平行trainer/sampler、永久损失入口或大框架；该task-owned脚本和必要原件在唯一诊断root保留。

独立硬限**2GPUh、峰12GiB新增存储**，全部加载/失败/临时优化器/权重及反传计入；0环境episode，四次临时单步计数不能写成0更新。
依据§23四十条生成与cross速度合计.1395GPUh及既有完整macro约一分钟GPU总占用，
本次多次LoRA/native VJP粗估.4–1.2GPUh，含实现/CPU读回预计1–2小时，实际资源与反传成本另计。
先用实际小批完成消费者核对并计费，按实际显存提高microbatch；超出上限风险时交具体剩余范围，不降低B独立性或静默扩规模。
launch前Sol实查strg01 data1 user quota及双节点；GPU02优先，仅无适用卡才GPU01。当前旧批GPU已全部释放，
按实际吞吐使用1–2张适用卡即可，不为满卡占位；临时两个j组可独立，但同一父/分组权重和四次单步含义不能变。
完整一批或实质阻碍直接Steer，不发源码完成/准入/心跳常规回报；结束后停止新增计算，由main主动解释与决定下一步。

### 26.6 完成与取舍

两独立teacher组均exit0，共四次临时更新、48条B生成路径，外部.2341999299GPUh、10.397GiB；
真实原件和完整推理见findings§240/机制§64。前5目标无一致独立B/另一teacher收益，BF16舍入未证明丢失有益方向，
不追加精度、loss、步数或正式训练。本项完整收束；后继优先研究架构，不从本条旧分支自动派发。

## 27. 已生成LoRA参与同一次教学回读的冻结结构检验（2026-09-29，已收束）

原32路径及§27.6的8父匹配参照已完成，累计.1406927840GPUh；科学解释见机制§65.5。
Owner最新明确不再派新实验，本节没有未执行后段，不自动恢复学习/official。

### 27.1 科学问题、历史和唯一范围

Owner要求以架构改进为主线，先验证特征/算子机制再重训。§26已收束；main检查共享输出上界后，
没有直接扩宽出口或自由生成A的充分理由。本条按机制§64.3–64.4，检验公共读取与条件参数读取的具体计算差别，
不把它先命名为错位bug，也不把旧§20回读、Unified、Reader反例遗忘。

唯一task=`operator_self_conditioned_readout_20260929`，新root：
`/data1/user/ymdai/ember_runs/operator_chain_diagnosis_20260929/self_conditioned_readout`。
仍使用真实e2afbfd7完整T2340的同一套冻结参数；可复用其clean pushed detached数值树。
四train task0/12/20/32，原两teacher、§26原B20 query/noise/target/valid，绝不取held/Test标签。
0参数更新、0环境episode；没有正式checkpoint、选点、训练CLI或新架构长期运行面。

### 27.2 同一套参数的四种有限重编译

对每个原正确teacher v，先计算原模型的`(X0,H0)=N_β(L,v)`、`M0=F_θ(A,X0,H0)`。
同task两teacher的M0只各编译一次并共享给本task后续读取，不复制已存在的训练/数据大资产。
原完整执行状态`w0=(A,B0+M0)`仍为单LoRA、同一真实source。
然后在相同RGB/语言/probe/tau1下做`(X1,H1)=N_w0(L,v)`；另以同task另一teacher的w0读取**原视频v**，得到`(Xo,Ho)`。
不得把另一teacher的RGB直接换进来，也不读取teacher action/state。

固定四个新臂，不扫系数或选择迭代次数：

| 臂 | Writer消费 | 要区分的计算 |
| --- | --- | --- |
| joint_self | `F_θ(A,X1,H1)` | 本视频完整条件参数参与读取之后的整体重编译 |
| address_input_self | `F_θ(A,X1,H0)` | 实际X链变化，含key、Value中的key依赖及递归，不叫纯地址效应 |
| value_context_self | `F_θ(A,X0,H1)` | 最终H/变化Value链；保留原X寻址 |
| joint_other | `F_θ(A,Xo,Ho)` | 同task另一正确视频参数的读取，区分特定自反馈与一般任务适配 |

所有臂最终只执行`(A,B0+M_new)`，**不加上M0**，不平均LoRA，不再回读第二轮。
调用现有`read_native_video`与`OperatorReadWrite`；只替换native读取时实际传入的完整LoRA。
F中的common A/B0/P/C/D/O始终为原T2340，不能因native传入w0而把Writer的公共B0也覆盖成B0+M0。
source基础权重冻结；安装/functional_call必须沿现有实际消费者，任务结束不修改父文件或旧冻结树。
action_out本身不反馈进返回H/X的原边界继续保持，不补假路径。

这是冻结结构干预，混合X/H两臂只用于拆分依赖，不能直接称为可部署新方法或已验证修正。
即使joint_self改善，也不由本项唯一归因于语义对齐、梯度相容或共同训练后的可学性。

### 27.3 实际功能与必要原件

八个teacher条件×四臂=**32条新B生成路径**，每条仍为20 query、完整50×32 latent、10步flow，
仅最终前7维对真实动作计风险。沿§26消费者及同microbatch/噪声语义，不改solver/normalization/目标。
原八个parent、四公共β、四MT300的B路径直接引用§26，不重复参照矩阵。
新任务首个父条件用固定前两个B query与§26保存预测作一次必要消费者核对，成本计入；
不是逐bit验收，也不因低位正常差异重跑整组。若差异与待解释效果同量级，应如实降低结论强度。

报告每task、每teacher的前5/全50/valid-future、motion6/gripper1、逐query差额及同/另一teacher的交互；
不只报总均值，不把单一专家动作MSE当唯一合法动作或闭环分数。
保存实际32预测、原query/noise/target引用及必要38个新M；H/X仅保留既定Q8/V8/action_out与最终H、
原始帧索引和教学读取参数来源，避免为所有层/帧复制全部中间量。
重读r0/r1/ro需要的native前向如实计费；两种混合臂复用已有X/H，不重复native读。
CPU按实际预测复算全部32风险、按实际script检查w0/公共参数边界、配对与finite；沿用已可靠的§26面板/旧预测验收。
结果中X/H/M改变与功能变化分开，不用更大hidden距离认定收益。

### 27.4 决策分支

- 若joint_self或一个真实输入链在独立B及两teacher上有清晰、跨task的功能改善，同时没有被其它horizon/运动/任务损害抵消，
  main才考虑同数据/预算原方法的有界学习及闭环验证；本项本身不放行fresh或正式续训。
- 若joint_other同样或更好，应降低“必须是本视频自身参数一致”的解释；只支持更一般条件化读取，不用自反馈名称夸大。
- 若收益只在一个task/teacher、量级很小、呈无方向交换或整体更差，不据此投入回读新架构，
  不加轮数、blend/gate、LR/初始化/辅助目标延长该假设。没有唯一根因时保留未知而不拼成整体资格。

### 27.5 执行、资源与停止

唯一Sol负责脚本/实际消费者/准入/执行；main负责上述假设及独立解释，不重复工程测试。
这是task-owned冻结诊断，无永久模块、平行trainer或evaluation400。
新增硬限**1GPUh、6GiB峰值**，包括加载/所有失败/父两query核对/native重读/临时写入。
依据§26两组完整反传和48路径仅.2342GPUh，以及§22原生读取成本，本项粗估.2–.5GPUh，
实现、执行和CPU读回约45–90分钟；先按真实占用修订，不将估计写成已运行。
launch前Sol核strg01 data1独立quota与双节点；GPU02优先，只有无适用卡才GPU01。
可按task分两独立组(0,12)/(20,32)用1–2张适用卡，避免按teacher分组后重复编译父视频；
不等待凑卡、不dummy占用，不因当前正式训练为空就填满卡数。
整组完成或实质阻碍直接Steer，常规准备/心跳并入整批；完成后停止新增计算，无自动下一轮或新训练。

### 27.6 原批完成后的实际消费者配对补项（2026-09-29）

Sol已完成32条新B路径、0更新/环境/held，两成功与一次父核验失败总.120205GPUh。
main直接读实际脚本发现：§26原父B用microbatch5，本批32干预用10；失败记录中父两query用2时
前5风险差.00078012。重试只将父两query核验改为5并得到0差，并未检验新臂所用10的父读出。
这不是模型bug或要求低位一致，而是已有差异可能接近小干预效应，须限定原差额的因果解释。

只补8个原父条件在本批相同生成消费者、microbatch10下的B20路径；同噪声/target/valid、同T2340和原M0。
优先复用已存38个父M0和父公共A/B0，不重新做视频读取；核对重装使用原单LoRA含义，不叠加回读M。
原32干预不重跑，β/MT不重跑，不改seed/solver/dtype，不添模型/学习/环境；新8条记为数值配对参照，不冒充新的架构臂。
报告parent10−旧parent5及candidate10−parent10的逐条件/任务/前5/全50/valid/motion/gripper差额，
旧readback和失败保留，输出独立matched_parent_readback并显式说明原比较范围。CPU逐条读新8预测，原32继承已有验收。
沿原1GPUh/6GiB硬限，GPU02优先，预计.02–.08GPUh及15–30分钟实现/运行/读回；实际准入与费用由Sol登记。
不把这项补齐扩成多batch精度扫描；完成直接Steer并停止，仍未释放回读学习、official或正式重训。

§27.6完成记录：两组exit0、8父预测逐条CPU读回，原32/β/MT未重算。
匹配后joint/X/H/other前5均值差+.00521058/+.00238276/+.00201408/+.00531835，
改善条件1/1/1/2，各8；数值差异影响微小个例而未改变主要判断，不采纳该冻结回读修改。
原件matched_parent10/matched_parent_readback.json；新增.0204880834GPUh、累计.1406927840GPUh/约1.3GiB。
完整回报已由main消费，未派新的实验。

## 28. 完整自身生成中的Q/V/动作投影条件作用（2026-09-29）

Owner已恢复自主机制与架构研究；本节为有界冻结诊断，不是正式重训。主讨论新假设及完整历史约束见机制§66。
固定T2340/e2afbfd7、train0/12/20/32、各原两teacher、§26独立B20查询/噪声/target/valid。
主要问题是条件寻址与条件内容是否存在跨自身状态的功能冲突；不能用局部余切或末层固定hidden替代完整生成。

**仅三个干预臂。** 从既有完整38个M与父公共A/B0重装唯一LoRA，不重新读取视频：

- `no_q_write`：18个expert q_proj的M置零；其它20个M保持。
- `no_v_write`：18个expert v_proj的M置零；其它20个M保持。
- `no_io_write`：action_in/out两个M置零；36个Q/V的M保持。

三臂都保留完整38-target公共β；不是移除整个LoRA、把某组B归零或恢复source权重。
不重标定范数、不缩放、不换teacher、不增加层/头选择。0参数更新、0环境、0新held分析。
每臂八个条件，共24个新B20十步生成路径；使用官方50×32噪声、10 flow steps、原消费者microbatch10。
父参照优先直接复用§27.6 `matched_parent10`的八个原父预测；只在真实消费者不兼容时保留理由并补同批父，最多8条。
公共β/MT沿§26现存参照，说明其旧batch5数值范围，不重算；干预主要裁决相对同batch10父。

**交付与裁决。** 保存唯一LoRA装载的目标集合、各预测/target/query索引、逐条件和逐task的前5/全50/valid风险，
分别报告运动6维、夹爪及开闭符号，不合并掩盖不利任务；主分数为实际十步生成，不以FM或一阶信用排序。
同teacher/跨teacher比较保留；原件中的两teacher不是独立task重复。若V删除仅帮task0而损伤其它任务，
不采纳全局删V；若全组没有稳定实际收益，关闭本次全局删减假设，不扩层扫/幅度扫或自动训练。
若有跨teacher、跨task的实质收益，主讨论再裁决下一项真实闭环验证；本合同不自动释放official或新学习。

**执行与预算。** 新执行者01a0ed67-9144-72d0-9309-07fdfbf5f211独占dev；main独占主树/科学记录。
复用已审生成消费者，若需保留源码按现有工程合同使用独占分支/pushed冻结，不在原冻结树原地改写。
新增上限.75GPUh/3GiB，预计实现、运行、CPU读回45–70分钟，90分钟未完成须说明实质原因与剩余工作。
按原40路径约.14GPUh的吞吐估算24路径本体约.09GPUh；预算含加载、必要父补齐及失败，不用于凑满实验。
launch前实际核双节点/GPU总量与strg01 data1独立quota、当前用量；GPU02优先，1–2张有实际吞吐收益的A40。
完成或实质阻碍整批Steer主讨论，不发送普通心跳、不自续实验；费用、失败和原产物全部保留。

§28完成：e0ef50e4冻结实现，24新路径、两exit0、复用八个matched_parent10、无父补跑，
.032783506974940084GPUh/约3.7MiB，0更新/环境/新held。删Q/V/IO前5均值差为+.01706043/+.00306638/+.000218632。
Q删除显著损伤task12；V删除在task0的局部收益由少数query主导，其全50/有效future及其它任务变坏。
原件为`operator_chain_diagnosis_20260929/operator_branch_intervention_20260929/{README.md,readback.json,group0,group1}`。
本节不支持直接删减修复；不能将冻结共适配破坏外推为删减架构共同重学后的上限，见机制§67。
两个专用入口通过Git e0ef50e4及冻结树保留，完成后不留在main活动运行面；无自动后继学习。

## 29. 覆盖按原生特征变化推进：有界fresh学习比较（2026-09-29）

Owner允许有界机制学习，要求在原因与有效修正验证后、正式重训之前停止。Owner最新指出冻结改动破坏共适配，
本项据机制§67–68改为真正共同学习比较，不将T2340冻结改好作为前置。它是270宏步原型检验，
不是新方法正式重训或部署资格；无自动后继900/2340、RL或Test。尚未证明覆盖是绝对性能主因。

### 29.1 唯一模型干预与完整学习

候选身份`T_change_clock`，保留T完整38-target rank128/alpha128、公共β、实际A地址、原P/C/D/O、stride5双RGB和单probe。
仅将TargetWrite的`memory @ key`替换为`(memory @ key) * g[None,:]`，其中
`d=vector_norm(hbar[t+1]-hbar[t],dim=-1)/sqrt(1024)`、`g=-expm1(-d)`，每个horizon位置一值。
V不再乘g；不得使用detach、学得gate、手工系数、clamp阈值、范数匹配或新loss。其余递归和输出B0+M不变。
全图真实FM梯度包括g→H→公共β，完全静态变化时M保持、零初值保持identity；norm零点反传应finite。

候选fresh Writer/公共A/B0/P/C/D/O、fresh Adam/scheduler；合法初值/seed与旧stage1 T完全同源，
不加载T2340、旧T270、MT或工程更新作初值。原始source冻结。所有公共与Writer模块共同学习，不设冻结课程。
首270严格复用stage1 `learning_spec`的36任务、teacher/query/flow事件、112query/宏步、任务权重、普通full50 FM、
normalization、优化参数及150步warmup；没有teacher同episode辅助或新数据。新增图不要求新参数形状，
但checkpoint/run/bank必须明确记`T_change_clock`与其精确公式，禁止因shape相同被当成原T exact-resume。

### 29.2 参照、评测及诊断原件

主要学习参照为已完成stage1 T270（116/400），原冻结784febb及原事件/初值/更新语义；
执行者先核旧真实合同和当前可复用owner，确认除覆盖外的模型/数据/目标/优化一致，物理拓扑与正常数值差异如实记录。
可以复用历史共同学习参照，不为形式重训T。若实际发现影响比较的语义差别，先报告具体项，不能静默换对照或自动加一臂。
强参照沿同scene MT300=153，成熟T2340=161仅作不同训练预算的完整能力背景；不把候选胜116写成完成最终目标。

只在候选270完成后物化一个correct400 bank，并沿stage1完整scene、state/video/RNG映射和官方消费者跑一次strict paired400。
每task全部50合法teacher各一次；8 full/392 compact及物体/EEF/谓词证据与原口径相同。
复用原T270及MT原行，不重评基线。若原件无法按同scene/video/噪声配对，先报告，不以新随机面板补成“匹配”。
90/180仅作完整恢复点，不物化/评测/挑点；270是本批唯一预定读出点。没有other/wrong/shuffle/reverse/Test。
报告逐task/suite/breadth、对T270与MT的R/G/L/churn/success-set、task-cluster区间；候选单节点没有相邻稳定性证据。

仅另用既有§12固定train0/12/20/32×两teacher/A28面板，对候选270做8完整FM及4公共β FM，
沿原query/action/noise/time；保存prediction/target及Q8/V8/out的真实X/H/M/G_B、帧索引供main学习机制读回。
不更新参数，不开启B20/新teacher/新环境诊断矩阵。旧T270同面板结果直接引用，消费者差异限制微小效应。
训练日志保留原loss/clip/活动梯度与耗时；不把信用更均匀、M更大或梯度接通当成能力证据。

### 29.3 工程、资源与停止

唯一指定执行者负责独占分支实现/实际消费者/现场准入，main独占科学记录并并行审阅。
从本节已push main创建或复用空闲dev，验证后用clean pushed detached冻结代码；旧冻结树和原件不改。
复用现有model/native、训练更新/恢复、bank和官方队列owner，不复制一套长期trainer，不向超大run/bank追加一串历史版本常量。
候选身份与合同明确、必要行为检查通过即可在本合同内继续，不另等main逐步批准。main及时集成经过验证的代码。
CPU检查应覆盖旧T行为不变、候选静态中性/非零变化/真实梯度、模型身份和事件匹配；必要GPU首个真实消费者检查计费，
不得把工程步权重当fresh初值或把通过它写成科学通过。保留失败和实际更新总数。

新增root仅`/data1/user/ymdai/ember_runs/operator_change_clock_pilot_20260929`。
硬限**8GPUh、24GiB新增峰值**，包括实现所需真实consumer、全部训练/物化/评测/诊断/加载/失败/恢复；不为花满追加计算。
依据旧T270训练3.6656GPUh、单bank约.31、单400约1.1及固定train面板小于.1，预计总5–6GPUh。
实现/CPU约45–75分钟，按实际2–4训练卡和后续并行队列预计整批2.5–4小时；超过4小时需说明实质原因与剩余预算。
launch前核strg01 data1独立quota/实际用量/峰值及双节点GPU；GPU02优先，资源上限沿仓库，
训练world2–4按有效吞吐，保持4task/112query与单Adam语义，不额外固定整批两卡。正常长任务等退出事件，不重复读进度。

非finite/源权重变化/数据或LoRA身份失配须停止该计算并保留事实；工程修复沿既有权限，不能改科学规则补救。
若预算不足以完成完整270/400，先报告缺口，不缩成80行选点或恢复不完整权重。完成整批后主动Steer主讨论并停止新增计算。
无自动续训、gate/rate/LR/seed扫描、公共aux或其它架构。阴性只约束该有限学习窗口，不能宣判所有共同学习不可能；
阳性须保留弱覆盖/幅度/优化的竞争解释，主讨论另作实际机制与相邻能力裁决，不能直接释放正式重训。

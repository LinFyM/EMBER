# 同一实际LoRA算子的教学写入与执行读取

2026-09-28。候选，未验证；科学论证见机制分析§42。只有progress登记的阶段有执行许可。
本文件§1–5定义完整方法，§6–16的工程、训练及诊断均已完成；当前一次冻结公共分支诊断授权见§17。
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

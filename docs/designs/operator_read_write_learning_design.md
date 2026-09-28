# 同一实际LoRA算子的教学写入与执行读取

2026-09-28。候选，未验证；科学论证见机制分析§42。只有progress登记的阶段有执行许可。
本文件§1–5定义完整方法与可失败判断，§6仅授权指定执行者完成有界工程，**不授权正式学习或held评测**。
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

# Factorized Condition-Kernel Program Memory Writer

状态：2026-08-06（BCI local）设计、实现、address audit、A40 profile、fresh AS0→200、
四点strict correct400与全部内部分析均已完成，并作正式负裁决；AS200未过预注册门，
reward阶段禁止实现或启动，当前暂停新实验等待owner讨论。本文建立在
Antithetic Program-Credit cycle1正式训练、strict correct400、train24内部机制分析和
400个held LoRA逐条复核全部完成之后；它不是对旧v6 checkpoint续训，也不恢复Direction
Store、Policy-Lane或任何历史并行Writer。

## 1. 最新证据把最早故障定位到哪里

Program-Credit从fresh v6 AS125阶段直接对`320×256`program施加closed-loop pair credit。
formal cycle1相对AS125由`97→106`，严格配对gained/lost=`18/9`，说明真实binary return
可以改善Writer；但净增9未过预注册净增10门，禁止resume cycle2。

更关键的是内部分析给出了此前架构都没有的同一接口两侧证据：

- 24个task的exact program cotangent pair cosine mean/median=`.000107/0`，负pair比例
  `.2464`，full24 energy retention=`.041874`，几乎就是互不相干的24个方向；
- 同一次共享Writer参数更新后，24个task的condition-mean program delta pair cosine却变成
  `.5801/.6128`，负pair=`0`，full24 retention=`.55537`；
- same-task五video的program/BA更新中task-mean energy fraction中位分别`.82990/.91623`，
  更新不是video-specific；
- AS125→cycle1的same-task video centered/sample energy在program仅
  `.002153→.002149`，在BA仅`.001154→.001178`，wrong/reversed/shuffled到program、BA、
  fixed action的相对差异也几乎不变；
- 400个held LoRA的BA relative change中位`.005519`，gained与lost分别
  `.004726/.004742`，无法区分；stable rank、top-1 energy和B-column cosine均几乎不变；
- binary cotangent energy为`.00261635`，semantic tie-break仅`.00003600`，前者约为后者
  `72.7×`。因此结果不是由LIBERO专属dense shaping或functional loss制造。

冻结decoder并没有完全阻断传递：program→BA→fixed action的更新relative-L2中位为
`.006782/.004713/.002279`。最早失效发生在**不同condition的closed-loop cotangent写入
共享condition map时**，早于decoder完全失活，也早于LoRA外观。

对共享网络`H_theta(c)`，一次小步参数更新对另一condition的影响近似为：

```text
Delta H(c_i) = -eta sum_j J_i J_j^T g_j
```

实测结果说明EMBER当前隐式`J_i J_j^T`把本来不同的`g_j`压成公共方向。神经网络训练可在
函数空间表示为切向核更新的理论边界见
[Neural Tangent Kernel](https://arxiv.org/abs/1806.07572)；该工作只支持这个数学解释，
不预先证明下面方法有效。

## 2. 本方法要回答的问题

> 如果condition之间的credit共享关系不再由可漂移的深层Writer Jacobian隐式决定，而由
> 固定、包含task language与单条视频时序innovation、可逐步审计的显式核决定，能否让不同
> task/video的policy program共同积累，并降低checkpoint能力轮换？

目标：

1. 从generic frozen source policy和functional identity**全新训练**一个完整Writer；
2. 同一架构接受AS functional cotangent或closed-loop return cotangent；
3. deployment仍是task language + exactly one action-hidden teacher video→一套完整rank16
   LoRA；
4. 每个更新对任意condition的影响可由sealed key和小型Gram矩阵精确预测；
5. decoder bootstrap结束后，不再存在能把task credit重新压成公共方向的可训练condition
   encoder、router或decoder。

非目标：

- 不强制LoRA高rank、正交、SFT能量profile或更大scalar；
- 不用task ID、suite、filename、outcome、teacher action或object pose形成地址；
- 不做one-task-one-expert、top-k outcome search、多video/LoRA平均或checkpoint融合；
- 不从AS125、v6-fast143、Direction Store129、Policy-Lane或任一reward checkpoint warm-start；
- 不把当前LIBERO的24 tasks、K4、binary reward或FactorHead宽度包装成通用方法贡献。

## 3. 信息墙与部署合同

输入保持：

```text
exact task language + exactly one raw action-hidden teacher video
  -> one deterministic condition feature
  -> one policy program H[320,256]
  -> one complete public rank-16 LoRA
  -> frozen PI05 source policy
```

Writer不得读取teacher action、proprio/state、reward、terminal、task ID、filename、object
pose或hidden normalization。固定policy probe只使用source policy、task language、teacher
RGB、固定synthetic suffix noise与固定flow time；它不是teacher action。train视频可用于
建立地址尺度，validation/test语言或视频不得参与拟合地址统计。

frame stride仍为5；38 targets、320 program slots、rank16、template A / zero B、source
policy和normalization均不变。reversed/shuffled必须在真实输入帧重排后重新计算condition
feature，不能只重排下游缓存标签。

## 4. 固定的task与video address

### 4.1 冻结foundation task descriptor

在任何Writer-owned adapter之外，用冻结PI05 PaliGemma执行text-only task forward。只取
authoritative task span的最后层hidden并做token mean：

```text
s_raw(T) = Mean_task_tokens(H_frozen_text(T)) in R^2048
s(T) = L2Norm(s_raw(T) - Mean_train24(s_raw))
```

`Mean_train24`只由24个train language建立，不读validation/test或outcome。它和Direction
Store使用过的稳定text address同源，但本方法不据此选择store；它只是显式条件核的一半。

### 4.2 冻结、policy-aware的逐帧video innovation

对每个采样帧，在无Writer Meta-LoRA、无梯度的source policy上执行与当前Writer相同的
image+language prefix和固定synthetic Action-Expert suffix probe。构造：

```text
z_t = concat(
    Mean_task_tokens(H_frozen_vl(frame_t,T) - H_frozen_text(T)),
    Mean_time(H_frozen_action_expert(frame_t,T,fixed_suffix))
)
```

第一项是task-grounded visual innovation，第二项把地址绑定到source policy实际可见的
action-expert latent；两者都不读teacher action。用固定seed的Johnson-Lindenstrauss投影把
`z_t`映到128维。投影是persistent buffer、永不训练，也不根据rollout结果选择。

令采样帧ordinal归一化为`tau_t∈[-1,1]`，使用通用四项temporal basis：

```text
b(tau) = [1, tau, cos(pi*tau), sin(pi*tau)]
v_q(V,T) = sum_t b_q(tau_t) z_t / sqrt(sum_t b_q(tau_t)^2)
v(V,T) = L2Norm(concat_q v_q) in R^512
```

常数项保留内容，奇偶time basis使reversal改变有符号坐标，shuffle改变frame-content与真实
ordinal的对应。这里没有LIBERO verb/object规则、首尾成功启发式、learned attention或
order auxiliary loss；其它视频任务可用同一固定序列核。

### 4.3 factorized random Fourier feature

分别对`s`和`v`建立32维fixed random Fourier features（16个frequency的cos/sin pair）：

```text
phi_T(T)   in R^32
phi_V(V,T) in R^32
phi(c) = L2Norm(vec(phi_T outer phi_V)) in R^1024
```

language bandwidth取24个不同train language距离的中位数；video bandwidth取train split内
同一exact language跨video距离的中位数。两者只读action-hidden train inputs，seed与数值写入
config。validation/test只应用已封存映射。随机特征用低维内积近似固定核的依据见
[Random Features for Large-Scale Kernel Machines](https://papers.nips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html)；这里仍须由真实Gram审计裁决有限1024维近似，
不能把论文当EMBER性能证据。

乘积不是把Core和Procedure当对称LoRA value；它只定义**credit可共享的条件邻域**：task或
video任一侧不相似都会减小共享。`phi`没有trainable参数、没有constant bypass，也不能随
checkpoint追逐当前loss。

## 5. Program Value Memory与fresh decoder

唯一condition-to-program路径为：

```text
M in R^[1024,320,256]
H(c) = sum_m phi_m(c) M_m
```

`M`有`83,886,080`个trainable values，按fixed seed从`N(0,.02^2)`初始化。`phi`的L2 norm
为1，因此fresh program尺度与历史随机compiler同量级。它是一张完整program value memory，
不是1024个scalar gate；每个feature coordinate拥有全部320×256 policy program方向。
外积key/value写入与线性读取可解释为fast-weight memory，相关形式边界见
[Linear Transformers Are Secretly Fast Weight Programmers](https://proceedings.mlr.press/v139/schlag21a.html)；本方法的memory是跨训练持久参数而非序列内临时state。

只保留PI05已验证的320 program-slot public topology和八个fresh hidden256 FactorHeads，
不加载任何v6权重。真实enumeration为FactorHeads `2,179,072`参数、Program Memory
`83,886,080`个显式更新values，完整Writer共`86,065,152`参数。原v6的Meta-LoRA、Semantic Core、Visual
Transition、Procedure和compiler全部退出trainable condition map；冻结foundation probe
没有optimizer owner。

FactorHead final Linear保持exact-zero，因此M虽为nonzero，step0 public LoRA仍逐tensor等于
template A / zero B。首个functional backward只能打开B-side output；B打开后A-side、M和
decoder input才可达。不得用手工nonzero B、static LoRA residual或第二套decoder绕过identity。

## 6. 显式kernel-corrected value update

一个full24宏步对每个当前condition得到program cotangent`g_i=dL_i/dH_i`。堆叠
`Phi∈R^[24,1024]`，定义：

```text
K = Phi Phi^T
A = solve(K + lambda I, G)
Delta M = -eta Phi^T A
```

其中`G`是24个`g_i`，`lambda=.01`乘以K的mean diagonal（phi归一后即`.01`）。由此当前
conditions上的精确更新为：

```text
Delta H = -eta K (K + lambda I)^-1 G
```

若K在当前24条件上健康，更新接近各task自己的`g_i`，不会再被隐式共享Jacobian压成公共
方向；对未见condition则由`phi(c)Phi^T`按task×video相似性泛化。只解24×24 FP64 Cholesky，
随后在CUDA上形成value outer product。所有rank先收齐actual world-size的24个`phi/g`并以
相同顺序计算；collective前仍遵守all-rank CUDA-ready和`NCCL_P2P_DISABLE=1`合同。

M不使用Adam、逐坐标preconditioner、weight decay或momentum，因为它们会把一个显式scalar
kernel重新变成output-coordinate-specific隐式kernel。`eta`只控制函数空间步长；profile按
finite、predicted-vs-observed update和不触发全局program trust cap选择一个数值，不能按held
rollout搜索。若需要安全边界，只允许一个全局induced-program RMS上限，且仅在真实超限时
缩放整次`Delta M`，不得逐task重权或改方向。

这个update不依赖credit来源：AS functional gradient、离线policy-aware gradient、直接
reward ES cotangent都产生相同shape的`G`。因此kernel correction是架构/优化接口，不是
监督微调专属auxiliary loss。

## 7. 从零的统一AS→reward训练

### 7.1 fixed decoder bootstrap：macro0→50

从generic source和functional identity全新开始，不加载任何Writer checkpoint：

- 24 train tasks、每task一条video、B20同task独立action queries、full24等权；
- M从第一步起按第6节kernel-corrected update；
- fresh FactorHeads在0→50由现有full24 AdamW functional gradient训练；
- foundation address、source policy和normalization始终冻结；
- macro50是预注册decoder freeze边界，不由functional loss或rollout选择。

这一阶段只负责建立program→complete LoRA的可用坐标系。它仍可能有decoder共享更新，因此
必须在固定50处永久结束，不能因held结果延长。

### 7.2 kernel-memory AS：macro50→200

macro50后永久冻结全部FactorHeads，只更新M；teacher action仍只进入合法functional
cotangent。此后condition map、kernel与decoder都固定，AS本身也在同一显式函数空间更新。
checkpoints固定`50/100/150/200`并全部做strict paired correct400，观察absolute、breadth、
相邻gained/lost、union/intersection和曲线是否共同累积；functional loss不选点。

AS200是预注册reward阶段边界，不从四点挑best。若AS200低于`120`或breadth低于6，说明固定
condition feature/decoder没有建立足够exploration substrate，方法直接负裁决，不用旧best
救援。若达到边界，则从同一fresh pipeline的macro200进入reward，不构成历史warm-start拼接。

### 7.3 direct program reward：AS200之后

永久关闭teacher action入口，沿用已经实证健康的双persistent-lane K4 CRN和binary-first
pair credit，直接估计每task`g_H`。FactorHeads、address、source policy和normalization都
冻结，只按第6节更新M。semantic progress仍只是双失败tie-break；binary energy必须单独报告，
方法主张不依赖该observer。

先只跑cycle1并相对AS200 strict correct400：严格`>150`则继续；否则必须净增至少10、
breadth不降且至少两suite改善才允许cycle2。不得按train reward、kernel condition number、
LoRA norm或“只差一条”改门。

## 8. 最短实现与验证路径

canonical原位替换，不保留v6并行model/loader：

- `video_program.py`：改为无梯度frozen text/VL/Action-Expert descriptor；
- 新建一个cohesive `condition_kernel.py`：temporal feature、RFF、Program Value Memory和
  24×24 kernel solve的唯一数学owner；
- `model.py`：删除trainable Core/Procedure/compiler，只保留memory read和fresh FactorHeads；
- `as_step.py` / training：显式收集per-task `phi/g_H`，M做kernel update，FactorHeads只在
  0→50进入optimizer；
- `rl_writer`：复用双lane rollout/credit，只把`g_H`交给同一memory update owner；
- architecture/config/checkpoint/evaluator建立fresh incompatible family，拒载v6与所有历史
  Writer checkpoint。

Program-Credit一次性analysis入口在本design与结果artifact封存后删除；有第二用途的纯
gauge-invariant metrics可迁入已有analysis owner，不能让退役方法保留active runtime。

聚焦CPU/小张量合同：

1. frozen address在train/eval、参数改变与resume前后逐tensor相同，所有buffer无grad；
2. reversal/shuffle重新计算并改变video feature，静态重复帧只有常数temporal坐标；
3. `phi=phi_T⊗phi_V`、norm、seed与bandwidth可复现，validation/test不参与拟合；
4. M非零而step0 LoRA严格identity，B打开后M/A/decoder-input按阶段可达；
5. direct read与batched read等价；
6. 小矩阵上observed `Delta H`逐元素等于
   `-eta K(K+lambda I)^-1G`；无冲突/正交K时退化为各task独立更新；
7. macro50后FactorHeads参数、optimizer owner和grad严格为0；
8. full24 assignment、all-rank gather、checkpoint/resume和actual world size闭合；
9. source policy trainable=0，validation/test action reads=0。

### 8.1 action-hidden address audit

正式AS前先在train24×50 videos上封存tiny descriptor/Gram artifact，不存原RGB或大feature
cache。只看：

- task与video bandwidth、Phi stable rank、K eigenvalues和`K+.01I` condition number；
- same-language跨video、cross-language、reversed、shuffled feature距离；
- 24-task每次no-replacement video schedule上的Gram rank与最大off-diagonal；
- validation 8 tasks只做apply-only coverage，不回写任何统计。

若phi重复、Gram系统性低于24-task有效rank或order反事实完全不变，属于地址机械失败，修复
feature定义后重新seal；不得根据rollout结果调bandwidth、RFF seed或维度。1024维若机械健康
即固定，不做K/seed sweep。

### 8.2 A40 profile

先单/六卡最长105帧、B20/B2三步，随后独立fresh0→1→exact-resume1→3。必须证明：

- frozen descriptor峰值和吞吐；
- FactorHead identity阶段、M kernel update、predicted/observed program delta finite；
- 24×24 solve、all-rank ownership和checkpoint完整；
- 46GB内0 OOM；若OOM先降frame/policy physical microbatch，不改变B20、full24、P1024或
  scientific update；
- profile权重永久弃用。

## 9. 可证伪结论与退役边界

支持本假设至少需要：

1. AS50→200的task cotangent与observed program delta关系符合显式kernel预测，post-update
   task delta不再从近正交变成全正高cosine；
2. same-task video更新不再由task mean占`.83/.92`，且video/order差异能传到BA/action；
3. single-checkpoint correct、breadth和相邻gained/lost比Program-Credit/v6形成更一致累积；
4. 若进入reward，cycle1相对同pipeline AS200过预注册门。

若地址Gram健康但AS绝对性能低，失败在固定foundation feature或program decoder覆盖；若AS
健康而reward后仍同向，先核对显式predicted/observed equality，不能回到rank/scale；若
program差异存在但BA/action无响应，失败在fresh decoder；若视频核有差异而训练values对所有
video仍相同，说明当前credit本身没有分辨这些variation。

首轮失败后不允许：调RFF seed、增加/减少feature数、改bandwidth追held、恢复learned router、
加入contrast/order auxiliary、one-task experts、multi-video、LoRA平均或旧checkpoint。下一步
只能根据最早失败接口重做foundation condition representation、program decoder或interaction
credit；历史实现和artifact由Git与design保留，canonical runtime原位退休。

## 10. 实现、地址审计与profile封存结果

- canonical AS路径已按第8节原位替换。Writer实际参数为Program Value Memory
  `83,886,080`加8个FactorHeads `2,179,072`，总计`86,065,152`；M的`requires_grad=false`
  且不进入Adam，唯一更新owner是full24 condition Gram correction。FactorHeads在macro50
  之后的optimizer、scheduler、checkpoint与resume cursor均冻结。旧v6 trainable condition
  path、独立temporal module和Program-Credit一次性analysis runtime已删除。
- address authority为`configs/pi05_condition_kernel_address_v1.safetensors`，SHA256=
  `7a49226e89529321f6764170247ce3b926ea6773aa9920cc5a3633f0e8cf0f86`。六卡只读audit覆盖
  train24×50与validation8×50 apply-only；50个no-replacement schedule的Gram全部rank24，
  最坏`K+.01I` condition=`7.547092`、最大off-diagonal=`.426992`。same-task video与
  cross-task demo0 feature距离中位=`.871805/1.405841`，reversed最小/中位=
  `1.156730/1.406382`；全部action/reward/outcome reads为0。机械门通过后按规则固定
  bandwidth、seed、P1024，不做held sweep。root为
  `runs/outputs/condition_kernel_address_audit_r6_seed2026080501_20260805`。
- `gpu01:0,1,2|4,5,6`的fresh0→1再同root exact-resume1→3通过。三步wall=
  `20.7128/19.8416/19.4484s`，峰值allocated/reserved=
  `16,556,672,000/19,344,130,048` bytes，longest105、logical B20/B2、0 OOM/clip。
  step1因zero FactorHead final layer使Program cotangent/update严格0；step2/3 cotangent RMS=
  `1.99464e-7/3.57174e-7`，predicted update RMS=`1.96840e-7/3.52395e-7`，global cap
  scale均1。三步Gram rank24、condition=`7.52338/6.63205/6.02257`；累计1,440 queries、
  72 videos，六rank state、sampler/RNG/scheduler exact-resume闭合，validation/test action
  reads为0。root为
  `runs/outputs/pi05_as_writer_condition_kernel_memory_profile_r6_b20_seed7_20260805`，profile
  权重永久弃用。

## 11. Formal AS、四点rollout与机制负裁决

### 11.1 训练与显式kernel合同成立

clean implementation seal `4038960`从generic source与functional identity完成fresh
AS0→200：200 macros、96,000 logical queries、4,800 one-video conditions、四个完整
checkpoint，wall=`3951.928s`、峰值reserved=`19,344,130,048` bytes，0 clip/OOM、0
validation/test action reads。root为
`runs/outputs/pi05_as_writer_condition_kernel_memory_formal_fresh0_200_r6_4038960_20260805`。

显式memory机制没有工程或数值失效：200步condition Gram均rank24，condition number范围
`5.139--7.750`，global cap scale始终1；predicted/observed Program update relative RMS在
macro50/100/150/200为`.002184/.001731/.001718/.001304`。raw cotangent与observed task
delta的pair cosine、负pair和full24 energy retention逐步对应；macro200 observed median
pair cosine=`.08298`、negative fraction=`.22826`、mean/average task energy=`.09350`。
macro51--200的FactorHeads freeze violation严格为0。因此后续失败不能归因于kernel solve、
共享condition Jacobian重新同向化、freeze/resume或多卡实现错误。

### 11.2 四点closed-loop失败且reward门关闭

50/100/150/200的strict paired correct400为：

| macro | correct | breadth |
| ---: | ---: | ---: |
| 50 | 46 | 3 |
| 100 | 46 | 3 |
| 150 | 45 | 3 |
| 200 | 49 | 3 |

四点共同state、video ordinal、env/policy seed与实际执行长度的policy-noise common prefix均
0 mismatch。相邻gained/lost=`5/5,4/5,6/2`，四点success union/intersection=`55/40`。
40个四点共同成功中Goal-6占37、Object-1占3；macro200的49个成功中Goal-6占42，另外仅
Long-1为2、Object-1为5，其余5个validation tasks全为0。曲线看似比历史Writer稳定，但这是低增益、
接近source identity的能力平台，不是多task共同累积或task drift得到解决。

AS200=`49`且breadth3，远低于预注册`correct≥120 && breadth≥6`。因此第7.3节direct reward
阶段按原合同禁止实现/启动；不得从macro50--200挑best、延长bootstrap或用reward掩盖AS
substrate失败。四点root统一为
`runs/outputs/pi05_as_writer_condition_kernel_memory_bci_correct400_noreplacement_seed7_macro{0050,0100,0150,0200}_4b04c90_20260806`。

### 11.3 地址与LoRA方向成立，绝对policy leverage失效

六卡内部分析完整覆盖24 train tasks×4 checkpoints=96 rows、6/6 rank payload，wall=
`273.968s`、峰值reserved=`19,277,021,184` bytes，0 target-action/validation/test reads；
root为
`runs/outputs/pi05_as_writer_condition_kernel_memory_internal_all4_r6_2972f8f_20260806`。

条件差异没有在上游被抹平。四点same-task demo0→1 relative-L2中位稳定为：fixed feature
`.78616`，Program约`.7839→.7833`，effective BA约`.7753→.7673`；reversed/shuffled BA
约`1.396→1.391/1.374→1.363`。same-task Program centered/sample energy约`.347`，所以
fixed task×video address、memory read和Program→BA方向传递都真实工作。

LoRA也不再是历史v6的伪rank1：macro50→200 effective norm中位仅`.17614→.17791`，但
stable rank=`3.794→3.724`、top singular energy=`.280→.284`、q/v B-column cosine约
`.186--.205`，16个coordinates全活跃。这个norm比corrected direct SFT的`35.7362`小约
`200×`，而高stable rank没有转化为增益。same/reversed/shuffled条件虽然产生相对完全不同的
tiny BA，fixed-policy action relative-L2始终只有约`.19--.24%`；identity→demo0也只有
`.20--.22%`。相邻checkpoint BA churn中位仅`.0169/.0189/.0225`，到action仍约`.20%`。

同task五video的checkpoint update中，task-mean energy fraction在Program为
`.7304/.7181/.6716`，BA为`.7839/.7807/.7268`。它确实低于旧Program-Credit的
`.82990/.91623`，说明显式condition kernel部分修复了credit同向化；但更新仍以task-common
为主，更重要的是全部更新被锁在极弱的policy tangent中。

### 11.4 最早失效接口与退役决定

fresh template-A/zero-B使macro1 Program cotangent严格为0；FactorHeads只在0→50用
functional AS bootstrap，到freeze时public A/B RMS中位约`.01829/.000369`，形成的LoRA
norm仅`.176`。macro50后kernel memory可以准确、彼此独立地移动Program，却只能通过已经
冻结的低增益decoder Jacobian影响policy。也就是说，本方法解决了condition之间“如何存”
和“如何避免共享参数重新混合credit”，却没有先建立一个具有足够绝对增益和policy-effective
方向的Program→LoRA写出基底。

因此正式拒绝“只要把condition credit换成显式独立kernel，fresh decoder会自行形成健康
policy program”这一联合假设。该结论不等同于fixed kernel本身无效，也不支持调global scale、
强制rank、延长同一bootstrap、增加RFF维度或改seed。下一方法若恢复推进，必须从低增益
decoder cold-start这一最早接口重做policy-grounded写出基底/训练耦合；在owner讨论前不实现、
不profile、不launch。配对、训练与内部汇总封存于internal root的
`experiment_analysis.json`。

# V6-LPCP Content-First Memory Grid

状态：2026-08-16 active preformal-GPU authority。简称 **CFMG**。本设计从sealed LPCP与fresh
CMBG branch初始化，不续训CMBG exact-noop checkpoint；只改变native-zero payload gate在memory grid中的位置。

## 1. Decision

CMBG full24证明了真实backbone memory不是无效carrier：5/6 active tasks的同task four-view gradient
cosine约`.988--.992`，held8方向、幅度和有向时序门也全部通过。但它在full24仍由task38以次大task的
`54.45x`梯度范数主导，24个deployed margins最多只下降17个，最终按合同恢复step0。

代码图揭示了比“再加容量或改optimizer”更早的接口：CMBG把zero-init payload gate放在temporal、K-set和
layer/token M2P之前。step0时这些模块的输入逐元素为0，所以首个backward只经过它们在零点的固定Jacobian：

- temporal attention在零输入处退化为均匀权重，主要只保留terminal endpoint差；
- video-set attention在零输入处退化为均匀集合归约；
- layer/token attention只使用初始化route，不能根据当前任务和视频内容选择通信；
- memory、temporal、set和M2P参数本身首步都没有梯度。

因此CMBG虽然在backbone中读到了真实内容，第一次shared commitment并没有先运行其声称的完整
`ordered video -> K-set consensus -> parameter topology`程序。本轮把同一个gate移到完整grid之后：

```text
CMBG: H -> zero gate -> temporal -> K-set -> layer/token M2P -> B rows
CFMG: H -> temporal -> K-set -> layer/token M2P -> zero gate -> B rows
```

唯一主要变量是gate placement。memory token数、rank、参数量、初始化seed、optimizer、reward、video数量、
source policy与全部信息墙保持不变。

## 2. Complete deployment graph

```text
exact task language + K=4 ordered action-hidden teacher videos
  -> per-frame real image/language prefix + 50 fixed Action probes
  -> one carrier-exact native PI0.5 joint context forward
  -> 37 one-way memory tokens read same-layer prefix/Action K/V
  -> H: per-frame [18 layers, 37 parameter tokens, 1024]
  -> per-video signed adjacent transitions + terminal goal residual
  -> permutation-invariant K-set attention and symmetric reduction
  -> layer-axis then parameter-token-axis M2P
  -> P(x): one content-conditioned [18,37,1024] parameter grid
  -> P(x) elementwise multiplied by one zero-init shared payload gate
  -> direct reshape to q/v/action-in/action-out rank16 B residual
  -> concatenate with frozen LPCP rank16 carrier as one rank32 LoRA
  -> frozen source policy closed-loop rollout
```

Writer仍在rollout前只运行一次，每个condition只生成一套完整38-target LoRA。没有per-video LoRA平均、
task ID、expert bank、第二套adapter或生成后RL。

## 3. Exact functional and gradient contract

令`F_theta(H)`表示现有per-video causal reducer、K-set和两轴M2P，令`G`是现有
`18x37x1024` payload gate：

```text
R_CFMG(x) = F_theta(H(x)) * G
G(step0)  = 0
```

所以step0仍逐元素满足`R=0`，public policy严格为LPCP carrier；constant video使每video adjacent delta与
terminal residual为0，因此不论`G`为何值都不能产生动态Value。首个backward满足：

```text
dR/dG = F_theta(H(x))                         != 0
dR/dtheta = G * dF_theta(H(x))/dtheta         = 0 at step0
```

这仍是清晰的两阶段学习，但与CMBG不同，首步gate gradient已包含内容依赖的有向时间、K-set与参数拓扑特征，
而不是`F`在零输入处的固定线性化。若gate update被共同接受，下一cycle的memory tokens、temporal、set与M2P
自然获得梯度；不需要straight-through estimator、冻结副本、第二次backbone forward或正负大数抵消。

## 4. Why this targets the earliest failed interface

- CMBG已经通过carrier parity、held same-task video、reverse/constant与native q/v/action链，故不重做视频前端。
- full24失败发生在task-local coherent memory credit进入第一个shared gate时；gate之前的content-conditioned
  sequence/set/M2P在实际首步没有执行其非线性选择，因此先修这个接口。
- gate仍共享，task权重仍等权；若content-first features不能自然支持共同更新，结果会直接否决本假设，不能用
  task-gradient normalization、PCGrad、task weights或scale sweep救回。
- 不新增language-only route。exact language仍在原生context与V6 Core中说明关注对象和目标，新增B Value必须
  经过有序video-derived `H`；无动态视频不能写出新增LoRA。
- 不降rank、不改A/B topology。rank8仍是开放的独立变量，但本轮必须先隔离gate placement。

## 5. Relation to historical evidence

CFMG保留V6/AS139的高absolute底座、LPCP的18层Action-probe conditioner、CMBG的37个真实context memory、
K4 four-view reward、capacity-matched direct B reshape与native-zero residual bank。

它吸收Gradient-Open的有效教训——exact-zero参数化不能让真正需要学习的condition route在第一次更新前失效——
但不复制旧W1 anchor或semantic basis。旧Gradient-Open解决了native写出却仍产生跨video近正交方向；CMBG已经在
多数active tasks上解决了该跨video接口，所以本轮只让这份video-coherent evidence在首步经过完整程序处理。

它也不重复CAPG/CMBG：两者均在程序处理前gate输入，因此首步看到的是零点Jacobian。CFMG明确检验完整
content-conditioned processing是否能把同task视频共同特征变成跨task可共存的parameter coordinates。

## 6. Implementation ownership

`src/ember/writer/parameter_grid.py`继续拥有唯一canonical grid；原位把`payload_gate`乘法移到两轴M2P输出。
不新增第二个Writer、兼容fallback或平行runtime。旧CMBG由Git commit`b4dbf84`、formal artifacts与历史文档保存。

fresh-incompatible config、checkpoint和launch schema必须改名为CFMG；CMBG checkpoint不得resume。外部ragged
input与76-tensor public LoRA API保持不变。实现不能增加backbone forward、模型副本、dtype扩展或逐tensor扫描。

## 7. Preformal mechanism gates

进入full24前依次证明：

1. 完整CPU与architecture guard通过；step0 public LoRA/effective BA/fixed action保持LPCP identity；
2. 真实K4中pre-gate `F_theta(H)`非零，final grid为0，首步只有payload gate gradient非零；
3. 人工打开gate后memory、temporal、K-set、layer/token M2P均获得finite nonzero gradient；
4. natural/reversed material不同，constant严格或数值近零，K-set置换不变；
5. carrier text/frame/grounded/action states不被memory或gate placement反向改变；
6. 固定task9/15/18 world3保持预注册成功关系，并要求每task four-view raw descent=`4/4`、
   cosine/energy至少`.40/.55`、shared raw coverage=`3/3`、至少一个原有11-scale trial达到native `12/12`；
7. world3通过后，validation8 held four-view至少6/8 tasks过门、aggregate cosine/energy至少`.40/.50`、
   held/train BA L2在`.25--4.0x`；reverse与constant继续通过；
8. 0 forbidden read/OOM/nonfinite/watchdog，world3 wall不超过carrier-exact CMBG matched wall的`1.25x`。

任一门失败即终局，不通过增加scale、seed、LR、rank、dtype或修改task权重补救。

### 7.1 Canonical implementation and CPU evidence

唯一canonical `parameter_grid.py`已原位把gate乘法移到token-axis输出；旧config/schema同时被fresh CFMG
config、checkpoint与evaluator identity替换，没有保留可执行fallback。聚焦测试直接hook最终token-axis输出，
证明step0 gate前content grid非零而public grid逐元素为0；K-set、constant video、gate-open全链gradient与
LPCP identity仍通过。focused=`5 passed`、完整CPU=`411 passed`，compileall与diff check通过；architecture
guard无hard violation、active source净减少2行且没有新module或并行version。以上只关闭CPU实现门，不提供
真实backbone、reward、throughput或closed-loop证据。

### 7.2 Real world3 and held-video adjudication

clean`010487b` fixed task9/15/18 world3完整exit0：成功关系=`1/0,2/0,1/2`、selected pairs=`8/16/8`，
三task four-view cosine/energy=`.966885/.747588,.875539/.871250,.971962/.969798`；cross-task gradient
cosine mean/min=`.092145/.021066`，raw/final coverage=`3/3`、native=`12/12`，j0 actual Adam被接受且
cycle=`114.216s`。0 forbidden read/OOM/nonfinite，故功能、共存与吞吐门通过。

同一checkpoint的validation8 held四view得到8/8 tasks，effective-BA cosine/energy=
`.9824118/.9851732`、raw/action cosine=`.9815933/.9875456`、held/train L2=`.9605484x`；三个train anchors
的reverse effective-BA relative-L2=`1.9549--2.0118`。原始12门中`constant_zero`因task18
constant/natural public BA=`.0051436`略超预注册`.005`而保持false；task9/15分别仅`.002580/.003696`。
结构测试证明exact constant layer memory严格输出0，实际counterfactual把重复图像经过32-frame BF16
microbatches，故该`.514%`最坏残差按设计中已预先声明的batch/kernel/reduction低位差异政策判为数值近零。

不得修改原始`cfmg_held_video_gate.json`或声称其12/12通过；独立
`cfmg_held_video_numerical_adjudication.json`透明记录例外。该裁决只授权一个从sealed LPCP fresh开始的full24
cycle1，不提供closed-loop成绩，也不放宽global all-view commitment、strict400或后续稳定/六臂资格门。

## 8. Full24 and closed-loop adjudication

preformal全过后，从sealed LPCP和fresh CFMG branch运行一个完整train24 cycle1。保留CMBG的两paired states、
四个disjoint correct K4 views、equal task mean、AdamW candidate及`1...1/1024`全局monotone commitment；
所有active task的全部views必须沿deployed endpoint preference下降，否则恢复step0并终局。

cycle1若产生非零accepted checkpoint，立即做single-checkpoint strict paired correct400，并报告per-task、
per-suite、breadth、retained/gained/lost/churn，与LPCP143、v6-fast143、SFMC144及最近direct baselines比较。
只有correct至少142、breadth至少7、相对LPCP lost不超过15且gained不少于lost，才允许exact cycle2。

稳定资格沿用owner标准：相邻single checkpoints约145或更高、breadth高、churn低/Jaccard高、同task不同teacher
稳定；首次达到约145且retention过门立即补correct/same-task-other/wrong/shuffled/reversed/no-video六臂。
correct必须沿有用policy direction显著优于controls，不能只让内部geometry更漂亮。

### 8.1 Formal cycle1 terminal result

clean`bb5341e` world5 full24 cycle1完整exit0：24 tasks/48 paired states/96 rollouts，candidate/reference=
`32/32`、gains=`3/3`，active task IDs=`4/19/20/25/34/38`覆盖四suite，cycle=`467.783s`，0 forbidden
read/OOM/nonfinite。相对matched CMBG，六task gradient norms只统一放大`1.78--1.92x`；pairwise task cosine
mean/min=`.00923/-.20641`，task34 four-view cosine/energy=`-.09228/.33697`，task38 gradient norm为次大的
`58.73x`。因此content-first并未形成task-selective coordinates或幅度平衡。

11个global candidates最多只让`14/24` deployed task-view margins下降，低于CMBG的`17/24`；search rejected，
exact step0 restored，final parameter delta、q/v/action effective BA与fixed-action response全为0。strict400只会
重测LPCP143而跳过。精确终局artifact=
`runs/outputs/pi05_v6_lpcp_cfmg_formal_cycle0to1_r5_k4_views4_nmc4_b8_bb5341e_gpu02p12347_20260816/cfmg_full24_terminal_adjudication.json`。

最早失败接口是`content-conditioned random Program -> balanced multitask first public commitment`：首cycle只有
payload gate有gradient；global gate update被拒绝后，temporal/K-set/layer-token M2P从未获得后续学习。CFMG
终局，不cycle2、strict、controls、resume或参数小扫。

## 9. Fast falsifiers and negative boundary

- content-first gate gradient仍由单task以数量级主导，shared raw不能覆盖全部active tasks：本接口没有形成跨task
  ownership；
- same-task coherence下降：非线性content processing破坏了CMBG已验证的共同video方向；
- raw共同而native trial失败：content-first continuous方向仍未形成finite policy-effective commitment；
- full24接受但strict高换手或明显降分：endpoint preference仍不能选择held on-policy有用方向；
- correct与shuffle/reverse/no-video同步：有序memory Value没有提供真实任务知识。

负结果只淘汰“CMBG完整content processing置于同一zero payload gate之前 + 一轮NEAP global commitment”组合；
不否定memory token、rank8、完整A/B、dynamic K/few-shot、生成LoRA或后续task-local RL。

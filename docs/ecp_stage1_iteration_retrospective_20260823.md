# EMBER编译式Writer跨版本迭代复盘

状态：2026-08-23扩展至MDCO、PECS与GOMQ因果补审。本文是方法与执行复盘，不是下一版设计，
也不授权新的GPU训练。

范围边界：v1--v24与MDCO属于ECP Stage 1；MDCO之后的PECS删除了ECP `q_pi + learned compiler`，只是一项独立
privileged upper-bound诊断，不是ECP实现进度；GOMQ是更早退役架构的历史补审，与当前Stage 1没有直接关系。把PECS写成
`ECP/PECS`、以及在没有当前决策价值时补跑GOMQ controls，都是本次需要接受外部审阅的执行偏移。

## 1. 总体结论

v24是一个有效负结果：layer-resolved target-local heads、连续static/process融合、fit-only prior校准和完整policy梯度都真实接通，
但24-task输出重新收缩为近共享方向，own-policy retrieval只有`1/24`；full/prior两臂都没有保住stable-shared support，且full
相对prior在22/24 tasks上略差。因此不能把失败归因于dead graph、family-shared head、simple fusion、输出幅度、GPU数值或训练
没有开始。

更重要的复盘结论是：**Stage 1的24个版本名并不代表24个彼此独立的科学问题。** 它们大致属于13组因果问题；其余版本是
同一问题的窄变体、实现修正、尺度校准、schedule复验或信息量不足的中间节点。Stage 0用三版native observer和一个matched
Action Meta arm解决了一个明确接口并通过Gate 1；版本膨胀主要发生在Stage 1。

Stage 1从`8562334`到`631aab7`不足22小时内形成65个提交，tracked tree净增12,163行、涉及61个文件。研究账本登记的formal
节点合计约4,997 task visits，尚不包括profiles、support-bank构建、materialization和308-panel audits。首版v1完成过held5
closed-loop并明确输给shared prior；此后v2--v24连续23个版本没有再产生held5 closed-loop rows，而主要由LoRA几何与冻结
open-loop policy-response support决定去留。

这套早门避免了反复评测明显坍缩的checkpoint，科学上不是毫无道理；但在连续多个版本中，它从“节省昂贵rollout的screen”
逐渐变成未经closed-loop校准的主要研究目标。项目因此在同一个最早断点附近不断重写compiler，而没有及时重新审查数据可识别性、
专家要求的fixed-compiler oracle顺序和代理门本身。

## 2. v24最终裁决

v24正式训练来自clean pushed `main@631aab7`，world-size6完成114 visits/19 updates。21,348,608个compiler参数可训练；
visible Program、privileged `q_pi`、observer和source policy全程冻结。compiler、FactorHead、process fusion与Action LoRA leaf的
gradient均持续非零，两个冻结模块gradient为0。fit19-only prior校准把factor residual从`1.21412`降到`.02583`。

最终证据为：

- candidate pair cosine mean `.97092`，门为不高于`.95`；
- own/nearest-direct cosine `.04779/.06674`，own retrieval `1/24`；
- effective norm ratio `1.32925`，所以失败不是输出近零；
- full fit/held相对shared response为`1.19865/1.05384x`，breadth `1/19、2/5`；
- prior fit/held相对shared为`1.18717/1.04658x`，breadth同为`1/19、2/5`；
- response越低越好，full相对prior在fit/held反而高`.00465/.00453`，只在2/24 tasks更好。

因此v23的fixed-hidden ridge/SVD定位只证明“给定hidden存在target-local readout容量”，没有证明当前19-task目标能从训练中识别该
readout。v24把最早失败接口从简单head capacity前移到：**当前数据与policy objectives没有识别process-conditioned own-policy
方向；校准后的shared-like解是一个更容易的吸引子。**

完整remote-safe证据见
`docs/evidence/ecp_20260823/stage1_layer_resolved_single_surface_compiler_v24_gate.json`。

## 3. 24个版本实际检验了什么

| 版本 | 主要变量 | 实际结果 | 复盘分类 |
| --- | --- | --- | --- |
| v1 | scalar-bounded `q_pi` + stable-prior residual compiler | held5 `27`，低于shared `43`；输出pair cosine `.9968` | 首个有效baseline |
| v2 | direct absolute A/B surface | 输出仍`.9942`同质，retrieval `2/24`；前228 visits没有functional gradient | 主要output-image检验，但训练合同不完整 |
| v3 | 同时切断address旁路并从首步加functional response | pair cosine降到`.9392`，own cosine仅`.0128` | 两个变量耦合，证明content进入但未识别方向 |
| v4 | BA/canonical coordinate bootstrap | 监督loss下降，retrieval退到`1/24` | 窄训练顺序变体 |
| v5 | query对已读Program content做乘性调制 | own cosine升到`.0821`、retrieval `3/24`，仍低于nearest | 窄readout变体 |
| v6 | multi-policy successful/learner/source/shared support bank | norm恢复、24/24优于source，但0/24优于shared | 第一个完整support-teacher检验 |
| v7 | stable prior与residual做SVD rank16 union | 大幅恢复shared，fit仅`9/19`胜过shared | prior-preservation output-image检验 |
| v8 | 删除raw parameter gradients；随后修复task-balance prefix并fresh复验 | 两次formal均失败，balanced复验更同质 | 同一方法的一次科学复验加一次schedule修复，不应扩成新路线 |
| v9 | bounded exact-prior rank selector | aggregate接近shared，但correction近全局、retrieval `1/24` | 新output-image检验 |
| v10 | selector只读process Values | fit support继续改善，held仍失败，compiler再次抹平Program差异 | 窄Value-path消融 |
| v11 | structured outcome credit首次接入 | macro1微小改善；macro2有16倍rank-sum尺度错误 | 一次有效profile加一次无效formal节点 |
| v12 | 修复rank-mean credit并matched重做 | graph有效但geometry/support不改善 | v11的真正科学复验 |
| v13 | baseline-relative support barrier | 8项support门过7项，held aggregate只差`.1675%`，geometry仍坍缩 | 有价值的objective修正；未校准代理门与held success的关系 |
| v14 | owner-resolved full-layer response | 只做1个optimizer update，未移动最早接口 | 应属于wiring/profile节点，不足以成为独立负版本 |
| v15 | 把v14完整训练到38 updates | hidden response可优化，own mapping不变且successful support退化 | v14假设的正式裁决 |
| v16 | target-local `B(Ax_ref)` activation effect | 局部retrieval到`11/24`，完整policy support严重破坏 | 新监督对象的有效负结果 |
| v17 | exact cross-episode action flow loss | 输出更task-specific，仍不指向own policy并破坏shared | 完整组合action监督检验 |
| v18 | action proposal + paired closed-loop factor credit | reward梯度真实，Program坐标几乎不动，compiler继续旋转 | 首个factor-space outer-credit检验 |
| v19 | 冻结compiler，只在Program切空间做structured reward | 切向可达但两次updates后仍坍缩 | fixed-compiler Program-credit检验，但信息剂量很小 |
| v20 | 固定Program和`q_pi`，只识别compiler | candidate更分散却不own，matched support退化 | 强因果锁，证明“task-diverse”不等于“task-correct” |
| v21 | bounded compiler下task-local free Program oracle | Program已明显分离，compiler image仍不可达 | 有价值reachability oracle |
| v22 | 复用direct-absolute compiler做同一free Program oracle | retrieval升到`10/24`，support变为`0/19、0/5` | 证明absolute image必要但不充分 |
| v23 | prior/full使用同一个direct absolute surface | prior和full都丢shared；定位family-shared head/fusion问题 | 同一surface结构检验 |
| v24 | target-local heads + separate reads + continuous fusion | 训练有效但重新收缩，full略差于prior | 排除“只修layer readout/fusion即可识别compiler” |

按科学问题归并，24个名字只有约13组主要因果问题：首版baseline；absolute/content/query readout；multi-policy support；
prior-preserving output image；outcome/barrier；policy-response标签；exact action；factor-space closed-loop；fixed-compiler Program credit；
Program-locked compiler identification；free-Program reachability；single prior/full surface；layer-resolved continuous surface。版本号增长速度
明显快于独立信息增长速度。

## 4. 为什么长期没有转化为闭环进展

### 4.1 代理门从screen变成了事实上的目标

own-direct cosine/retrieval最初用于快速排除24套几乎相同的LoRA。但专家已经明确指出successful LoRA不是唯一解，raw/effective
update接近某个direct expert不是成功策略的必要条件。项目仍把`.30 own cosine`和`13/24 retrieval`作为进入held closed-loop的
硬门，等于让一个非必要几何条件决定是否观察真正目标。

308-panel support比raw LoRA几何更接近policy function，但它仍是固定occupancy上的open-loop response，不是generated-policy
occupancy上的success。v13已经接近完整support门，却没有用一次bounded held5 rollout校准“差`.1675%`是否真的意味着闭环无效”。
此后11个版本继续围绕同一代理面优化。早门节省了单轮rollout，却增加了总迭代和错误归因成本。

### 4.2 数据独立单元远少于参数和输出自由度

v24训练21.35M参数，最终写1.287M LoRA values，但shared mapping只有19个fit tasks。188个successful panels、120个learner
panels和更多episodes提供同一19个task-to-policy mappings的重复状态，不会自动变成更多独立meta mappings。专家第二轮已指出
“独立统计单元主要是任务”；Stage 1仍未把已授权、经审计的non-held meta tasks及其successful policy assets纳入compiler
identification，也没有建立真正same-endpoint/different-procedure的process-identifying tasks。

这种欠识别下，stable shared prior是一个覆盖广、loss容易下降的共同解。v24 target-local heads增加了自由度，却没有增加独立
约束，因此优化器更容易复制一个shared-like surface，再让process路径产生很小或破坏性的修正。

### 4.3 fixed compiler原则执行得太晚

专家的关键顺序是先用privileged policy evidence建立并证明一个fixed functional compiler，再冻结它学习video-to-Program。
v1--v18长期让Program teacher与compiler共同或相继改变；moving gauge虽然被反复诊断，却没有在Stage 1一开始通过固定、广泛
policy-functional坐标彻底消除。v19才第一次永久冻结compiler，v20才做互补的Program-locked identification；到v21/v22才用
free Program直接裁决compiler image。

这些后期因果锁有价值，但也说明前18个版本中的很多“Program不对”与“compiler不对”仍然互相混杂。

### 4.4 专家给出的direct layer/family-local surface被替换成了长时间的bounded变体

专家建议的是target-local numeric rank queries、layer bias与family-specific direct A/B heads。项目在v7--v21花了大量版本于
SVD union、bounded selector、QR normalization、固定factor能量和angle retraction；这些路径主要为保住shared而设计，却把
own-policy image压窄。v22的oracle才直接证明这种偏差重要，v23才修正prior/full surface，v24才恢复38个target-local heads。

这不是完全忽略专家意见，而是每次只落实一小部分，并让历史实现的局部便利持续支配下一步。最核心结构到最后两版才真正同时
出现，而当它出现时，数据与objective identification问题已经成为主因。

### 4.5 privileged teacher仍未成为可识别的Program分布

专家要求的是多个successful policies与video views共同识别`q_pi(P)`和`q_V(P)`，允许功能等价、不可见recovery与uncertainty。
Stage 1虽然使用multiple members、reliability、occupancy和uncertainty，但很多后期实验把每个task在固定video visit上的一个
`q_pi` Program捕获并锁定，再识别compiler。这对因果诊断合理，却没有证明该Program是视频可预测、跨task稳定的分布对象。

因此Stage 1一直在回答“这个privileged坐标能否写出own policy”，还没有建立“视频可观察部分与successful policy公共结构是否
一致”。在只有19 mappings时，继续扩大Program或head自由度不会自动解决这一点。

### 4.6 每个负结果过快地产生一个新版本

65个提交集中在不足22小时，意味着实现、profile、formal、materialization、audit、定位、文档和下一设计几乎连续发生，没有
设置跨多个结果的强制概念复盘点。v11的尺度bug、v14的一次optimizer update和v8的schedule偏差都被计入版本叙事；v3同时改变
address path与functional start；v24同时引入target-local heads、双read、非线性fusion和prior calibration。版本看似“一次一变量”，
实际有时过窄到没有独立科学意义，有时又耦合多个必要修正。

快速提交和及时清理本身没有错；问题是版本号与formal仪式给局部修正制造了“已经推进一个科学阶段”的错觉。真正的最早失败
签名长期没有改变：task differences可以制造，own successful policy mapping与shared support不能同时成立。

### 4.7 资产复用做得比方法收敛更好

source policy、observer authority、expert bank、successful trajectories、phase analysis、stable prior和support banks大多被复用，
v22还正确复用了v6 direct compiler，没有重复长训。GPU任务通常在一到六分钟内完成并及时释放，v24也把full/prior audit合并成
一次panel execution。资源浪费的主因不是重复训练同一个大资产，而是反复增加schema、config、objective分支、evidence和解释
成本。active tree虽最终保持单路径，但Git认知负担与验证表面积仍快速增长。

### 4.8 没有证据支持“再训练久一点”

多个版本的梯度、loss、参数移动和task diversity都真实发生；v15/v16训练到38 updates，v1到190 updates，v21/v22 free Program
也到38 updates。失败不是普遍dead graph。相反，很多曲线在继续训练时只是更task-diverse、更远离shared，却没有更own。
因此v24不应靠续训、LR/rank/seed/fusion小扫挽救。

## 5. 哪些是真正保留下来的进展

1. Stage 0 v3建立了跨episode、跨task非退化的ordered event observer；Action Meta matched arm中性且已永久冻结。
2. validation8 local rank16 oracle `250/400`证明静态single-LoRA与source policy不是根本容量瓶颈。
3. multi-success完整trajectory/phase表示在held5得到`5/5`任务几何，证明policy内部存在可提取的task structure。
4. phase decoder与stable shared prior分别提供真实闭环增量，说明fixed functional coordinate和shared support都有价值。
5. v13证明baseline-relative barrier比无条件拉回shared更合理；v20--v22的互补锁把bounded image与Program inference分开裁决。
6. v22证明direct absolute amplitude打开task差异；v23/v24进一步排除了hard prior bypass、family-shared layer readout和simple
   static/process fusion作为充分解释。
7. exact action、owner-local response与paired simulator reward的梯度链均真实接通；失败在识别而非工程可微性。

所以“没有进展”应准确表述为：**没有Stage 1 Gate 2或最终闭环性能进展，但得到了一批可复用资产与负边界。** 这些负边界只在
下一步真正改变数据独立性、fixed compiler建立方式和closed-loop gate层级时有价值；若继续做LR-SSC局部版本，它们只会变成
更多历史负担。

## 6. 后续强制纠偏规则

### 6.1 版本与实现

- 只有checkpoint-incompatible、回答新科学问题的合同才获得新版本名；bug、梯度尺度、schedule、日志和underpowered节点使用
  同一版本revision，不计为新方法。
- 每个新方法先写一页falsification card：唯一主要变量、现有反证、必要输入、最早可观察结果、closed-loop裁决、失败后明确淘汰
  什么。没有这张卡不写新module/config。
- 一个formal节点最多改变一个主要因果变量。若两个修正逻辑上不可分，必须明确登记为joint hypothesis，不能事后分别归因。
- 同一失败签名连续两次不变时，停止该family；不得再由最近一个metric差异自动生成下一版。

### 6.2 证据门

- raw/effective LoRA cosine、retrieval、rank、norm只作collapse定位，不再作为成功策略的必要门。
- open-loop support仍可筛掉明显破坏shared的checkpoint，但任何新的support proxy在主导两轮决策前，必须用一次train-authorized
  held5 paired closed-loop校准其方向性。
- compiler oracle的首个方法节点直接报告source/shared/direct/generated closed-loop、retention、gain retention和breadth；不能
  再连续十余版只在geometry/support内部循环。
- profile只证明运行图、梯度、显存和吞吐；一个optimizer update不能产生科学负裁决。

### 6.3 数据与识别

- 不再用19个task mappings训练新的20M级target-local compiler后再询问泛化。先用经审计、排除validation/Test的non-held meta
  tasks建立更多独立policy mappings；同task更多episodes只增加state coverage，不计为mapping diversity。
- privileged policy teacher必须表示multiple-success等价类与uncertainty，compiler按multi-state policy function和closed-loop
  建立；不把某个task expert A/B、单一fingerprint或固定task-local Program当唯一真值。
- 在声称process understanding前，必须增加same-endpoint/different-procedure或真实中间约束的数据；现有LIBERO只足够裁决
  same-embodiment机制。

### 6.4 架构顺序

- 下一次Stage 1重建若发生，先建立**固定、layer/family-local、direct absolute、single-surface functional compiler**，并在
  authorized meta leave-task-out上通过closed-loop oracle；未通过就停止compiler路线，不训练`q_V`。
- compiler通过后才训练video-to-Program；full video必须相对language+scene/endpoints创造success，不能由结构性process开关自证。
- structured outer credit只能更新已经有过程识别证据的Program inference；不能同时发明video semantics、Program gauge和compiler。

### 6.5 工程与Git

- 一个科学hypothesis只保留一个canonical config、一个formal run和一个remote-safe verdict；重复用途只保留一个canonical
  artifact，临时profile完成后删除。
- 每个里程碑合并一次实现、一次结果，不再让几十个微提交替代方法复盘；仍保持main及时推送和task-owned worktree及时清理。
- 新source增长前先审计现有owner与复用面；Stage 1当前12k行新增代码不再继续线性扩张。

## 7. v24时点决定

1. v24关闭，不续训、不做小扫、不跑held5、不进入`q_V`。
2. 不创建或实现v25；LR-SSC配置标记为failed并封存，只由Git、formal artifacts和v24 evidence保留，不再是active
   executable hypothesis；精确复现使用训练时的detached `631aab7` authority。
3. EMBER-ECP总目标保持active，Stage 0 authority和全部昂贵资产继续保留。
4. 下一次实现前先形成一个新的单一科学合同。默认最有依据的方向不是LR-SSC修补，而是：在更大的audited meta-task policy
   mapping集合上建立fixed functional compiler，然后让一个直接的leave-task-out closed-loop oracle决定该compiler是否值得继续。
5. 若该oracle在合理的一次结构实现与一次修正后仍不能显著保留direct support，则把失败归到Program/compiler可识别性，停止
   当前zero-interaction compiler family，而不是再创造第N个head/fusion/selector版本。

本文不把上述默认方向升级为active design。只有完成falsification card、数据authority与一次closed-loop evidence计划后，
`progress.md`才能登记新的active Stage 1。

## 8. MDCO后验复盘

上述纠偏后来被MDCO完整执行，而不是停在计划层。90个task mappings、task-equal dense训练、fit90 structured simulator credit与
第一次不受geometry/support拦截的held5 strict paired250全部完成。训练图有效，但candidate只有`20/250`，低于source `21`和
shared `43`；direct-latest success/gain retention只有`10/108`和`3/96`，Goal/Long仍为0。完整门和near-pass均失败。

这使v1--v24复盘中的几个推断得到更严格区分：

1. 19个mapping确实是旧实验的欠识别风险，但不是当前失败的充分解释；扩到90后同类失败签名仍在。
2. structured simulator reward不是缺失：75/90 fit tasks有非零advantage并真实更新`q_pi + compiler`；问题是这种fit-local方向没有
   识别出source-unseen successful-policy等价类。
3. open-loop support不能作为可靠替代门；本轮full/prior与closed-loop符号都只对齐`2/5`，正式失去继续主导早期决策的资格。
4. 过去几十个版本的问题不只是“没有早点扩数据”，而是长期把可优化的局部代理、task差异和参数移动当成接近policy-effective
   mapping的证据。MDCO用一次闭环把这三者与真正迁移能力分开了。

跨版本同rows对照把“吸引子”进一步澄清：MDCO的20个成功有19个已经在source、shared或首版Stage 1任一checkpoint出现，
18个直接与shared重合；它保留的10个direct-latest成功又全部来自global task0。可是MDCO与首版1140的same-task LoRA
effective-update cosine均值只有`.09128`，own retrieval仅`2/5`。因此几十版反复出现的不是同一个LoRA参数局部最优，而是冻结
source在少量easy initial states上的outcome basin；不同compiler可以写出差异很大的参数，却没有让task-conditioned差异接管
闭环行为。后续若只证明参数更分散、norm更大或Program更可分，仍没有改变这个最早失败接口。

因此当前不再从最近一个metric差异自动生成下一版本。Stage 0与昂贵资产保留，当前Stage 1 compiler family停止；在形成新的、
真正改变可识别性假设且由早期闭环裁决的合同之前，仓库保持无active successor设计。MDCO事实证据见
`docs/evidence/ecp_20260823/stage1_mapping_diverse_compiler_oracle_tv540_gate.json`。

## 9. PECS后验复盘

MDCO之后没有继续修改Program-to-LoRA compiler，而是用PECS删除中间learned mapping：从成功expert在选定
teacher contexts上的精确policy effects出发，通过固定exact-VJP solver直接求一套rank16 LoRA。local-effect版把held5
strict250从MDCO的`20`提高到`58`，证明移除shared learned compiler后能够创造真实新能力，不能把过去所有失败都归因于
静态LoRA不可达。

随后唯一的增强把相同frames上的监督从单点velocity扩为PI0.5官方10步完整去噪action/flow trajectory。effect objective在
fit和held都稳定下降，LoRA与success rows也确实改变，但strict250只从`58`变为`59`；Goal/Long仍为0，direct-latest的
108个success只保留37个。这将断点从“decoder/solver没有学好”前移到**功能约束的state coverage**：16个teacher frames
附近的精确函数值，即使覆盖完整去噪时间，也没有定义policy在50个未见初态及其closed-loop visited/recovery states上的
完整行为。

PECS因此也解释了为什么“再加一个更精确的hidden/effect/action监督”已经不是有依据的默认下一步。当输入只有
action-hidden videos时，Writer需要从视频推断一个能支配整条状态占用的policy update；而过去几十版主要反复改进的是
task-level或稀疏context surrogate。这些目标可以更容易、更稳定地被优化，却不因此增加必要的状态约束。PECS的精确边界由
`1142e5b`与`docs/evidence/ecp_20260823/pecs_complete_trajectory_held5_gate_20260823.json`保留。

## 10. GOMQ峰值的迟到因果补审

跨版本复盘发现，历史上最高的shared Writer单checkpoint不是ECP/MDCO/PECS，而是GOMQ cycle 2的validation8
`151/400`。它当时只完成correct panel；same-task-other命令在CLI解析阶段失败后没有重试，wrong/shuffled/reversed也没有正式
运行。我随后在没有回答“151到底来自视频、时序还是shared/language support”之前，就把主线转向大规模新架构。这是本轮最关键的
证据顺序错误，所以本次只补齐同一冻结checkpoint的严格配对controls，不训练或修改模型。

| condition | successes/400 | breadth | correct-only / control-only | correct margin | exact p |
| --- | ---: | ---: | ---: | ---: | ---: |
| correct | 151 | 6 | — | — | — |
| same-task-other | 139 | 6 | 28 / 16 | 12 | `.09614` |
| cross-suite-wrong | 131 | 7 | 43 / 23 | 20 | `.01866` |
| shuffled frames | 127 | 6 | 40 / 16 | 24 | `.001842` |
| reversed frames | 115 | 6 | 50 / 14 | 36 | `7.07e-6` |
| frozen source identity proxy | 48 | 3 | 108 / 5 | 103 | `2.83e-26` |

五个video panels的episode key、env seed、policy seed root、policy-noise common prefix与teacher reference videos均零
mismatch。source identity也与correct在可比较的四项配对字段上零mismatch。正确视频相对wrong、shuffled与reversed都有显著、
广泛且方向一致的净优势，因此GOMQ不是纯language/no-video假象；它确实把视频内容和帧顺序写入了闭环有用的LoRA。尤其
shuffled与reversed是重新排列真实frames并完整forward，不能用cache复用或只改变标签解释。

但补审并没有把GOMQ升级为合格方法。same-task-other只保留`123/151=81.46%`的correct successes，低于90%鲁棒门；相邻
cycle为`151→135→131`，峰值不稳定；correct的Spatial task1与Goal task3仍为0，前三个task占`80.13%`成功；wrong仍有131、
shuffled仍有127，说明大量能力来自shared/language/static support，视频只是有用修正而非充分task program。source identity是
无adapter proxy，不是learned language-only Writer，故language净增量仍缺正式baseline。最后，GOMQ磁盘上序列化rank32，虽有
`A=[A0;A0]、B=[B0,deltaB]`使实数有效更新严格为`(B0+deltaB)A0`、rank不超过16，但BF16压为rank16并非bit-identical，尚未做
formal paired400。

因此准确裁决是：**GOMQ cycle 2成为当前最强的shared Writer视频因果锚点，但不是恢复训练的active design。** 后继不能只在
loss、hidden或LoRA几何上优于它，必须同时超过151的absolute、81.46%的same-video retention、151→135→131的稳定性、完整
controls和breadth分布。事实证据见
`docs/evidence/gomq_20260823/gomq_cycle2_causal_controls_strict400.json`与
`docs/evidence/gomq_20260823/gomq_cycle2_causal_adjudication.json`。

## 11. 为什么迭代几十个版本仍没有形成进展

这不是一个单点bug，也不能归结为“架构还不够复杂”。我在推进节奏、证据顺序与科学问题拆分上同时犯了几类错误：

1. **先换架构，后补最强结果的因果资格。** GOMQ 151本应立即触发same-task-other、wrong、shuffled、reversed与相邻稳定性；
   一次CLI失败后我没有把它作为未完成里程碑追到底，反而让未经审计的峰值退出主线。结果是后面几十个版本在没有明确强基线
   因果画像的情况下优化新接口。
2. **版本数量增长快于独立信息增长。** ECP v1--v24在不足22小时产生65个提交、约12,163行净新增和4,997次task visits，
   实际只覆盖约13个主要因果问题；窄尺度、schedule、head和fusion修正也获得新版本名。命名与代码变化制造了推进感，却没有
   等比例增加能改变路线的闭环证据。
3. **代理指标从诊断变成了决策目标。** v1 held5失败后，v2--v24连续23版没有新的held rows，own-direct cosine、retrieval、
   response与308-panel support主导了去留。MDCO最终用闭环证明这些proxy与held方向只对齐`2/5`，PECS又证明inner exact-effect
   大幅下降仍不能确定closed-loop policy。闭环门放得太晚，使大量优化都发生在尚未校准方向的代理空间。
4. **不断移动表示、Program与compiler坐标。** fixed compiler、direct absolute layer/target-local surface和single prior/full
   surface落实过晚；此前observer、Program、decoder与objective经常一起变化。即使每轮梯度有效，也难以知道是输入语义、输出
   坐标还是监督覆盖在失败，负结果便容易催生下一个head而非淘汰一个明确假设。
5. **把task数量不足与过程可识别性混在一起。** 扩到90 mappings是必要审计，但MDCO的`20/250`证明“更多现有任务”没有解决
   当前compiler。进一步核查又发现source71和target40 reward全部是最终状态合取，没有已证明的same-endpoint/
   different-required-procedure pair。它们能增加object/relation/task mappings，却不能让task reward迫使Writer读取中间过程；
   因而不能把更多LIBERO endpoint tasks自动等同于更多process supervision。
6. **低估了静态LoRA所需的状态占用覆盖。** PECS把learned decoder完全删除，用exact successful-expert action/flow function直接
   求LoRA，仍只有`58→59/250`且Goal/Long为0。问题不是局部target还不够精确，而是16个teacher contexts没有定义50个未见初态
   及其visited/recovery states上的完整policy。过去很多版本改善了task-level或稀疏context surrogate，却没有增加这一决定性约束。
7. **历史科学代码与最新运行基础没有正确分层。** 本轮补审从旧`8553b61`恢复GOMQ代码时，没有先叠加main已经存在的NFS-safe
   SQLite `DELETE` journal修复，导致reversed首轮在NFS WAL上出现重复claim。虽然原33/36 shards被精确复用、只补48 rows且
   最终配对有效，但这本可避免。精确恢复旧方法不应恢复已修复的基础设施缺陷。
8. **清理与文档虽频繁，却没有替代方法级停顿。** 我及时退休了大量active paths，但每次局部裁决后很快又建立successor；
   “只有一个canonical路径”仍可能是一条过快轮换的路径。真正缺失的是在强结果、负结果与数据合同之间做一次跨版本合并判断。

这些问题也给出更严格的保留项：task-local rank16 oracle `250/400`继续证明输出容量不是首因；GOMQ证明shared Writer可以从
action-hidden有序视频产生闭环有用且时序特异的更新；PECS证明固定功能求解比learned compiler更接近真实policy增量；MDCO与
PECS共同把剩余断点定位到**可识别的过程信息和跨状态policy约束**，而不是再加一个reader/head、延长训练或做小超参扫描。

## 12. 本轮停止决定

本轮不创建下一版本、不压缩GOMQ、不补language baseline、不重训任何Writer，也不把上述复盘立即翻译成新架构。当前需要保留的
只有三层事实：GOMQ是最强视频因果锚点但未过稳定/鲁棒/rank资格；ECP/MDCO的learned compiler没有从更多endpoint mappings中
识别held policy；PECS的exact local function也没有覆盖闭环状态分布。

任何后续设计在成为active前，必须先明确它新增的独立信息究竟来自哪里：能排除endpoint捷径的process-identifying task/data，
能够约束跨初始化occupancy的训练信号，或另一个同样明确且可早期闭环裁决的来源。若只改变Program shape、slot数量、decoder、
rank参数化、loss组合或训练时长，就没有跨过本次复盘定位的断点，不应获得新版本。

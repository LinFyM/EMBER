# 实际参数条件的经验编译：完整方法与下一批合同（2026-10-09）

## 1. 决定及其依据

本设计替代已完成的[首批经验编译](experience_conditioned_compiler_20261009.md)，是否active以progress顶部为准。
Owner已经授权在EMBER硬约束内自主改变方法、架构和GPU计算组织；本次不是等待逐项许可的候选清单。
唯一实现仍由`src/ember/experience_compiler`拥有，不增加并行Writer、旧版fallback或第二部署adapter。
主讨论负责科学取舍；实验session负责实现、消费者检查、实际profile、运行与Git集成，不把普通工程问题退回主讨论复验。

**本次要改变的联系是：从对隐状态Q的递推，改为学习对真实控制参数的条件修订。**
输入是合法教学、实际执行该实践的参数、真实经验；输出是下一套完整LoRA。
共享学习可以多次使用一个真实经验事件，不必在每个optimizer更新前重新采集整条实践及两条SDE信用rollout。
每个新教学condition依然只执行已学编译规则，不拟合采集到的动作轨迹，不运行任务内policy optimizer。

首批135/400低于同scene强MT153和T161；train末点initial/end/null均28/48，没有先在训练域形成净修订收益。
216条meta链中134条没有修订，82条发生修订，其中12条屏蔽经验；70条获得未遮蔽经验的修订分布于24个实际映射。
原success分支在U之前退出，成功经历不进入实际修订；这是原方法的既定规则，不是事后发现的工程bug。
原完整坐标e与共享decoder可以直接改变初始控制，identity初始化的D对Q的Jacobian则为零；
后期真实非零梯度没有转成有益修订。已修订条件的同query配对FM变化约万分之几，三个窗口没有持续改善趋势。
这些事实支持改变学习组织与参数接口，不能证明某一项是135分的唯一根因。

本次是一个有界的联合方法检验：固定强行为起点、实际参数条件、零更新保持的decoder、包含成功的经验事件和可复用FM学习。
不把最终分数差全部归因于某个单独模块，也不以改名清除首批及更早历史的负证据。
不选原图原样续训：其每次meta更新平均242.04秒、warm更新12.22秒（均为原world2），经验曝光仍稀疏且没有净修订趋势；
这些是完整更新计时，不能当作某一算子的精确耗时。

最近似历史约束保持：[首批完整分析](../analyses/experience_conditioned_compiler_20261009.md)中的旧T正确/另一正确/public为161/150/103，
说明既有FM学习已有有限视频增量，不能把它全称为language shortcut；旧self-read116没有真实环境后果，旧shared SDE父161→158没有经验条件编辑。
本次沿用有序教学/原生特征与真实FM，改变的是实际参数输入及经验事件学习，不能把“用了真实反馈”再次当成相对首批的新解释。
[先前固定MT候选](../analyses/independent_video_lora_compilation_20261009.md)尚未实施，不能算本次的正证据或已失败前身。
参数重构、functional decoder与已有全rank输出的历史又表明：可表示和功能可导均不等于能从视频学到可迁移控制。
旧Q在首步零导数也不能解释整个训练：128 warm早已越过初始化，后续通路确实接通；新零点设计是明确起点与信用的干预，不是永久断图修复。

## 2. 数据、资产和信息墙

沿用coverage_v1的显式24/8/8协议、既有语义排重和相同36个等权共享训练映射：
`0,1,2,4,5,7,12,13,14,15,17,19,20,21,22,25,28,29,32,34,35,36,37,38,42,43,51,55,56,62,64,73,95,96,97,101`。
support12仍对应LIBERO-90 local IDs `2,3,11,15,16,22,24,33,55,56,57,61`；不扩大allowlist或重划held。
复用原合同的source step1000、coverage MT step300、tokenizer、source normalization及38-target实际映射；
现存资产路径由原`experience_compiler.contract`及`configs/operator_read_write_v1/learning_spec.json`解析并写入新run contract，禁止复制大资产。
source基础参数和原生图文prefix冻结。每层rank=alpha=128、缩放1，最终实际算子仍为`W_src+B A`。

K=1；exact language加一条指定的action-hidden正确RGB教学，stride5加真实末帧。没有dynamic-K声明。
teacher动作、state/proprio、reward/terminal、task ID、文件名、物体位姿和hidden normalization不进入编译器。
自身实践采用RGB、8维proprio、实际7维动作、实际native hidden和reward/success/terminal/截断字段；
使用这些反馈依据Owner于10月8日对**自身交互来源**的开放，不把held离线配套标签混入。
本批不新增物体关系oracle、特权规划器或人工操作程序。
validation仅运行共享参数冻结的编译；不读取held离线动作、不产生共享梯度，不用最终成功回馈适应。Test封存。

## 3. 新condition的完整流水线

```text
指定教学RGB + exact language → 冻结原生Φ/H（缓存一次）
初始实际LoRA Λ0 = 强MT完整rank128
当前Λ真实实践 → 原始经验E（包含成功、失败和预算截断）
实际Λ编码 + E编码 → 改变教学空间读取 → 有序教学memory
共享参数编辑器 Gχ(V, language, Λ, E≤j) → 下一套完整Λ
完成本次编辑后，若刚才实践成功或1024环境steps预算耗尽则冻结；否则继续实践
最终唯一Λ固定，Writer退出，在未参与该condition适应的新初态执行
```

首批采用episode结束或预算截断作为实际编辑时点；这是本批计算安排，不是Owner规定的永久时点或固定轮数。
1024包括dummy settling和实际控制steps；settling-only尾段没有新decision时不强造编辑。
**成功经历先被编辑器消费，再决定结束。** 旧控制器的自身成功不认证编辑后控制器，也不认证新初态；这个风险由实际读出检验。
不设置目标J、最少轮数、轮数菜单或“成功后额外试几次”。实践state stream不含该condition最终读出state。
初始MT本身不由教学改变；第一次神经教学读取发生在第一次有真实decision经验的编辑。
教学native extraction、神经初读/重读、replay及实际动作成本分别计量，不把缓存算成没有观看。

### 3.1 实际特征和参数状态

教学沿用完整双相机Φ（每时点512×2048）和固定MT probe H（50×1024），probe使用seed1729、time=1及合法原生图文prefix，省略teacher state段。
空间读取仍为8 slots/帧、宽256，随后4层有序时序块；language、真实帧位置、camera/patch位置及首末标记保留。
经验沿用实际执行的time=1/.1 hidden、pre/post观测、proprio、最多5步动作及执行mask、反馈、episode/step身份；
先读每个decision事实，再用2层有序编码及16 slots聚合。失败decision不筛除。
只缓存冻结Φ/H和原始事实；不能跨共享参数更新缓存trainable slots或更新后的表示。

旧Q不作为持久条件状态。每次重新编码**实际**完整Λ：对target l、rank r，将A的真实行及B的真实列分别
用按实际矩阵宽度共享的线性映射压到128维，拼接成256维，加target/rank身份并归一化，得到q_in[l,r]。
同宽度因子复用投影；A/B角色独立。不引入逐task网络。因子以冻结MT每个实际矩阵的RMS（最小1e-6）为数值单位。
所有incoming因子和历史经验停止环境/历史行为梯度，参数编码器本身可训练。
这里依赖从同一MT出发的固定因子坐标，不声称对任意A/B等价旋转不变。

q_in和经验slots在空间压缩前改变教学query；所得教学memory与经验共同供2个轴向编辑块读取。
末端产生`δq[l,r] = sigmoid(g[l,r]) * P_out(h[l,r])`，gate bias −2，P_out权重和bias零初始化。
其余编码/注意力/投影正常fresh初始化。δq之后不加带非零bias的归一化或其它破坏零点的变换。

### 3.2 零更新保持、但具有功能导数的完整参数decoder

仍按实际A行/B列切64数值块，保留其真实target、A/B、rank和块位置。
对每个块的incoming值x，令s为上述固定矩阵RMS、e为64维可训练地址，使用：

```text
b = Wx (x / s) + We e + b0             # 512维，地址不存MT参数值
u = Wq δq[l,r]                         # 512维，同target/rank只算一次
δx = s Wo [SiLU(b + u) - SiLU(b)]       # 64维，无独立输出bias
x_out = x + δx
```

Wx/We/Wq/Wo普通非零初始化；同一块的两项使用同一参数和精度。初始δq=0使每个输出块严格保留incoming值。
同时`∂δx/∂δq = s Wo diag(SiLU'(b)) Wq`一般非零，真实FM可以从第一次更新教会零初始化的P_out；
初始decoder公共参数的两路导数抵消，不能单靠改地址/公共bias绕过零更新改变整套控制器。
之后上游学习通路打开；不声称所有模块在第一步均非零梯度，不添加第二个零初始化乘法瓶颈。

输出重新组装全部38个完整A/B，**没有**冻结MT rank128加另一个rank64、没有因子拼接、SVD截断、checkpoint融合或第二adapter。
这里使用“当前实际因子加编辑量”的参数化，允许最终完整A/B逐步离开MT；源policy中physical PEFT保持identity，不能再额外执行MT。
`(B+δB)(A+δA)`含交叉项，identity初始化不等于训练后不遗忘保证。
公共δq仍可能忽略教学或经历；零点性质只改善起点和信用结构，不证明有用视频知识已经被学习。

计算上，将原来每个64块重复的256维条件投影提到38×128个token上；块侧是64维实际值加64维地址。
这减少重复线性计算和地址参数量，但真实decoder速度仍须profile；不能从FLOPs推断整个系统提速。

## 4. 共享学习：真实事件复用，而非新condition轨迹微调

一个存储事件包含指定V/L、**实际行为参数Λ_in**、累计真实E、endpoint、行为版本、合法query数据来源及采集状态。
从这个事件训练`Λ_out=Gχ(V,L,Λ_in,E)`；χ是跨36任务共享的唯一可训练对象。
同一事件可在χ改变后重新使用，因为Λ_in和E是明确观察到的输入，不假装旧轨迹由当前χ生成，也不重新解释一个旧隐状态Q。
不对旧policy轨迹使用on-policy PG，不从held经验更新χ。此次监督的是一步有用编辑；没有声称对整条环境闭环精确反传。

每事件仍使用同task、不同于teacher episode的7条成功demonstration query，各4个时段取样，共28个。
真实完整native FM保持action offset1、50×32 latent、前7维动作消费、末动作padding和冻结source normalization。
同一事件的incoming/outgoing使用**同一**query、noise和flow time，原生prefix可以复用；损失为：

`L = mean(ell_FM(Λ_out)) + 0.2 mean(ReLU(ell_FM(Λ_out) - stopgrad(ell_FM(Λ_in))))`。

源模型及Λ_in不优化，梯度经实际Λ_out回到编辑块、参数编码、经验/教学读取和decoder全部共享模块。
第二项只是对本批query上变坏的额外功能压力，不是闭环保持定理；自身执行动作只作经验输入，不能改成正确动作拟合标签。
在小编辑近似下，FM差值的一阶项是`<∇Λ ell, δΛ>`，真实标签教共享编辑器从V/当前参数/E预测有利方向。
部署时没有这个标签梯度；能否从未见教学和经验推断合适方向，仍取决于实际跨条件学习和迁移，不能由链式求导宣称已成立。

新编译器和optimizer/scheduler fresh；AdamW lr3e-5、weight_decay0、global clip1，constant schedule，与首批保持相同量级。
本批完整学习只用上述FM，不加SDE PG、任务内轨迹optimizer、trust回滚或decoder蒸馏。
这是为检验经验到参数编辑的学习关系作出的本批取舍，不是撤销自身reward许可，也不把RL永久排除。
reward/success仍作为经验字段。固定1/8事件屏蔽E、按task轮转分配，以有限覆盖null输入；不宣称得到充分独训的无经验算法。

### 4.1 数据与更新的有限安排

1. **固定行为池0：** 每个非held任务4个指定teacher条件，共144。以固定MT采集至自身success或1024steps，
   保留每个有decision的episode/截断endpoint，包括成功终点。行为actor未使用教学，原件必须这样标明；不能虚报神经教学读取。
   Λ_in均引用同一MT资产，E是真实实践。训练共享G时才读取指定合法V，teacher与query跨episode。
2. **共享学习1–180：** 每update逻辑4事件，36任务等权轮转；先均匀选本task的condition，再均匀选该condition的endpoint，
   防止长失败链因为endpoint多而增加task/condition权重。180更新每task20次事件呈现，总720次、20160条FM query。
3. **一次真实分布刷新：** 冻结χ180，每task另2个condition，共72，使用本设计完整自适应算法采集。
   必须保存每个endpoint之前实际执行的完整Λ_in（或已验证相同的MT引用），以及对应真实E；不能只保存最终LoRA。
4. **共享学习181–360：** 每task在池0/刷新池之间各取1/2事件，其内部仍condition/endpoint均匀。
   总计1440次事件呈现、每task40次、40320条FM query；这些是训练呈现量，不冒称1440个独立任务或新采集条件。

完整采集条件216个，各condition独立状态/经历/参数；pool0和refresh的teacher选择在运行前按无放回task袋登记，尽量不重叠。
共享采集实践排除固定train面板最终state32–34，独立种子流预先登记；所有sampling/cursor由逻辑event ID决定，不能随GPU/队列完成顺序改变。
不按在线成功率偏置采样，不把旧首批缺少实际中间Λ的轨迹冒充本批参数条件数据。
已有固定行为经验确能合法复用是方法差异；不是反复看到同一轨迹便保证更多独立知识。
χ180之后的训练依赖刷新数据就绪；χ180的formal400与刷新采集相互独立，应及时并行，不等无关评测结束才训练下一段。

这项数据组织受[Watch, Try, Learn](https://arxiv.org/html/1906.03352)固定行为采集后复用成功/失败经验作共享监督的启发。
该工作输出在线上下文策略，任务和数据条件也不同；它不能证明EMBER的完整LoRA编译或本次参数编辑必然有效。

## 5. 完整读出、选择与可失败预测

正式节点预注册为χ180和χ360，各自完成correct strict single-checkpoint paired400。
复用原`video_schedule`全部50视频、固定state-video/RNG与MT/T真实scene合同；每checkpoint有400个不同task-video条件，
每个条件独立适应且最终state未参加该条件实践，不复用另一个condition经验。official preprocessing、10flow/5actions、horizon等保持。
MT153、T2340 161及上批135保留全部原始配对。选择只认完整400；不使用80-row screen、checkpoint union或每task挑点。
若批次完成，唯一待后续因果检验的checkpoint为360，180用于学习方向及相邻稳定性，不在两个分数间事后挑峰。
如果360比180退步，保留这个事实，不自动改选180；后续科学判断由主讨论承担。

两个节点沿用同一fixed train48条件，读取强MT/end/masked-E replay；强MT只需实际跑一次并复用于相同scene/RNG。
null从MT开始，按真实链的相同编辑次数和结束点递推G(V,Λ_null,empty)，不把真实Λ_end或中间真实Λ偷渡进null。
null不另采环境，明确保留真实链长度/停止点这一条件；它是有限上下文删减诊断，不是独立充分训练、同预算运行的算法对照。
完整神经replay与额外final48均计费。暂不自动追加所有held视频controls；选定完整方法后再依实际缺口登记，shuffle/reverse仍最后且不参与选点。

每节点报告per-task/suite、breadth、MT/T/相邻checkpoint retained/gained/lost/churn/Jaccard、实践stop/J分布及全部尾部成本。
重点判断：

- 从精确MT起点编辑后，train是否出现净控制收益，并是否优于同链null；若没有，不能把缺口只解释为held泛化。
- train改善但held不改善，降低跨任务迁移支持；考察真实映射多样性与表示局限，不能拿更多同task重复代替更多任务。
- held改善而null相同，只能支持完整参数生成/共享学习更有效，尚不能归因于真实经验或视频知识。
- 两个完整节点高于强MT且能力交换合理，才增强这条完整方法的支持；仍须后续公平视频因果证据，不能宣称EMBER问题解决。
- FM明显改善却上述闭环联系缺失，降低“成功dem query足以教会这种编辑”的支持，不用rank/LR/seed小扫或无期限续训保护它。

一次有限提前停止：若χ180完整400不超过135，且train end既不优于固定MT也不优于null，则在最近完整checkpoint边界
停止尚未完成的后段学习/采集，取消尚未启动的360正式条件；已并行完成或在执行的有效工作如实保留并计费。
正式队列先派完180待启动条件，再让尾部空闲资源接360；不为凑齐180退出而空占所有卡。
保留完整阴性并回报主讨论接续。135只是本批明确的严重退步/无训练收益合取停止线，不是通用接受门槛或Owner审批条件。
其余情况执行预注册360；不因内部梯度、loss或单task正例自动扩大预算。科学结论以完整证据为准。

## 6. GPU执行与实际预算

原首批额外profile已提供真实多condition同模型计算证据：B1/2/4/8含重置吞吐约8.21/11.24/14.25/15.78 control steps/s，
B8对B1为1.922倍，峰allocated9.22→10.42GiB。B8的reset/settle及env.step约占41.3%与24.8%；
这只是短程profile，不是完整新方法或全部任务的加速保证，B8也未被证明是吞吐最优。
各B公平复用同组persistent环境；不是原生产逐task重建环境的baseline。40-step短段的reset占比不直接外推到完整episode。
实际采集Phi/H的20-step实践短段B8/B1为1.666倍；B16单replan可运行（allocated11.90GiB），没有持续吞吐结论。
完整非held链的replay+VJP1.331秒、LoRA热搬运/pack占上述B8 wall不到0.2%；本次decoder改动服务学习接口，不宣称它解决主要执行瓶颈。
后续实现须把不同条件的实际LoRA、native prefix/flow与独立env slot一起批量化；不能仅把相同张量复制成batch。
继续复用cost-balanced dynamic queue、long-first和persistent workers；每slot保留独立语言、LoRA、完整scene、state、RNG和事实身份。
同task成组可减少环境切换，但不能静态绑GPU、改变task权重或等慢slot完成才放行所有其它condition。
在实际实现上测B4/8及有余量时B16、必要的CPU环境并行和reset重叠，以端到端吞吐选择；失败/无收益同样记录。
训练实测合适的1–4 ranks、FM/query microbatch、frame/experience/decoder chunk；保留有效batch4及所有逻辑权重。
单训练不跨节点；使用NCCL_P2P_DISABLE=1、GPU-local NUMA和deferred NCCL。每次launch现场检查双节点与现行6/8总卡上限。

冻结的教学/自身Φ只算一次并作有界缓存；同一pre/post图像使用观测ID引用，避免永久存两份相同Φ。
不在每个FM更新同步gzip整条历史链。行为采集原始RGB、实际H、动作、反馈及参数来源仍保存，可由异步CPU写入；
缓存丢失可从合法原始事实重建，但不能把重建后的其它policy hidden当成真实执行H。
训练query读取/预处理预取，shared prefix复用，GPU和CPU耗时分开计量。低UTL不直接等同CUDA核低效，不能为数字好看增加无益计算。

新root为`/data1/user/ymdai/ember_runs/parameter_conditioned_compiler_20261009`。
初估实现4–8小时、计算6–12小时；整批物理GPU区间总预算40GPUh、首次科学GPU开始后计算elapsed上界16小时，新增峰224GiB。
这是当前成本监控和执行计划，由主讨论依据真实成本、原因及预期判断价值自主规划与更新，无须Owner逐批批准；
调整投入计划不得静默改变科学样本、比较口径、checkpoint选择或停止规则，也不放宽实际quota、共享GPU准入和信息墙。
估计依据：上批完整20.518GPUh、正式单点约一半成本；本批有两次400但去掉每更新fresh实践/SDE rollout，且原生批量已有短程收益。
这些仍是预测；实现后profile必须用实测替换训练、采集、编译、两次400及失败余量的分项估计。
先查strg01的data1独立quota/实际个人用量和共享容量，包含原始经验、实际Λ快照、最多64GiB frozen-feature cache、checkpoint和临时文件。
实际参数相同可引用同一资产，不复制MT；不另建全1800视频大缓存，不在data0写新产物。
若存储不足，实验session已获授权依核实的生命周期与依赖自主清理冗余中间checkpoint/无恢复用途训练状态、失效缓存和临时产物，
保留必要科学原件、关键权重、有效恢复点及当前依赖，记录删除清单和实际释放量，不需要Owner另行批准。

环境上界分开登记：训练采集216×1024、两个train面板32×1024、两次formal800×1024，共1073152实践steps（包含settling）；
另有train final最多240行、formal final800行，按220/280/300/520及每次settling10计算，不计入适应预算但计入总账。
profile另限4096实际steps、2GPUh（计入总40GPUh），不读取Test。实际success早停、失败、重复state和所有读取均保留，不只报成功分母。
按固定suite组成，上述额外final最多353600环境steps（含settling），全批实践/final/profile合计上界1430848steps。
资源/profile若表明无法兑现预算，提交具体成本与可执行取舍给主讨论；普通实现故障自行修复，不能静默改科学合同或超额开跑。

## 7. 实施边界、交付与接续

科学合同变化涉及fresh架构/数据/学习目标，旧checkpoint不作resume。旧首批frozen代码/产物/Git保留，新schema明确区分。
在最新main建独占codex分支/worktree，检查现有owner和consumer后原位替换canonical路径；按code-architecture-gate控制新模块和尺寸。
应直接验证：MT identity与真实FM信用、参数输入/记录消费者、成功后编辑、不同条件batch的scene/RNG/LoRA身份和有效权重、完整checkpoint恢复。
只做与实际变化对应的检查，不重复上批已封口的无关工程验收、不用大规模tensor/hash扫描证明工作。
formal采集/训练/评测来自clean pushed detached frozen版本；现场物理拓扑与完整checkpoint/RNG/cursor/数据池版本记录。
相同科学语义的工程修复由实验session自主新冻结版本承接，保留原失败/有效产物/费用；科学变化或超预算才由主讨论裁决。
主讨论承担科学判断与投入计划调整，不能以等待Owner例行许可作为接续停点；回报后须核对完整原件与相关历史，
沿实际特征、算子、学习信号、参数及行为解释结果、比较竞争解释并推导有界改进，不能只交分数或追逐局部指标。

训练、采集、formal的ready依赖应事件驱动接续，整批只保留一个可靠完成/异常回报；不逐阶段Queue、自通知或发心跳。
有直接退出等待者时不另叠加自Queue。最终回报至主讨论`01a11b05-3460-7eb0-aca8-177d6d86ef48`，
通过既有codex-session-messaging的active Steer或idle/notLoaded resume/direct start，核实真实处理并保留原session设置。
交付完整原件、方法预测更新、负例与成本、Git clean pushed及tracked窗口移交。批次结束不代表EMBER完成，main须自主消费并继续作出取舍。

## 8. 本批执行封口注记（2026-10-09）

[完整科学消费](../analyses/parameter_conditioned_compiler_20261009.md)保存实际结果及适用范围。
180 strict paired400为127、train MT/end/null27/25/25，触发§5预注册科学停止；
刷新72和360共享训练/面板已经并行完成，360 formal只完成32个Long1条件20成功，其余368取消，没有selected checkpoint。
实际费用13.053185GPUh（原账本保守13.480595）、science elapsed4.863415h、峰6卡；当前新增ROOT275.60GiB超224登记51.60，历史峰未知。
strg01 data1独立quota仍有953.46GiB余量，但存储估计/监控不足保留并交main自主调整投入/保存计划，不产生Owner审批。
原科学比较、停止合同和所有有效/失败产物不据上述结果改写；本设计为已执行历史合同，新科学执行须由main另行登记。

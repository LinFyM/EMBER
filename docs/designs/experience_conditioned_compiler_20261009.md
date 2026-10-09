# 经验条件完整参数编译器：首批实施与学习（2026-10-09）

## 1. 当前决定、授权与证据位置

Owner要求现在开始、持续自主推进；观看/实践不固定任何轮数，当前采用唯一完整rank128输出，不采用冻结MT128＋残差64拼接。
本设计是唯一active方法，以progress登记为准。专家[完整建议](../review_materials/20261008_method_rethink/EXPERT_RESPONSE_FRESH.md)
与[交叉审阅](../review_materials/20261008_method_rethink/FRESH_RESPONSE_REVIEW.md)保留其原貌；下列具体修订优先。
主讨论负责科学分析、架构/学习取舍与下一批接续；实验session负责实现、实际消费者验证、资源/profile、运行、Git和整批交付。
阶段预算用于控制本批计算；主讨论可在Owner长期授权内依据新证据登记调整，不向Owner索取架构或下一实验的例行许可。
失败要更新假说并继续解决EMBER，不把一次批次结束变成整个项目停止。

**要解决的实际不足。** 历史T/C、自身hidden重读、准确关系和跨情境监督都没有取得稳定超过强MT的完整能力；
正确/另一正确视频及局部功能改善未转成广泛控制收益，强T在同任务新初态和已打开的困难访态也有明显失败。
这些结果不证明视频缺少知识，更不证明增加输入就会修复。此次改变的是共享学习事件：
让当前教学编译策略实际执行，用其真实后果改变下一次教学读取和完整参数生成，再在独立初态评价形成的控制器。
自身经历可能帮助区分“教学看到的动作过程”与“当前控制器实际能做到什么”，并把读出的教学内容与自身执行联系起来。
这是待检验的机制，不是已定位的唯一根因，也不宣称已经学会主动探索。

主要竞争解释是：(a)反馈条件重读能学到可迁移的修订；(b)共享先验变强而经历/中间教学内容仍被忽略；
(c)训练任务上记住修订关系但不能跨任务迁移；(d)功能监督/随机策略信用与正式ODE能力不一致。
首批先观察一个完整主模型怎样学习，避免在实现成本未实测时复制整条训练线；匹配独训首末帧参照仍是后续优先比较，
但不作为开工许可门槛。首批的null/阶段读数只能判断当前模型的有限输入作用，不能替代完整视频增量结论。

## 2. 数据、信息来源与固定资产

- 使用`configs/libero_24_8_8_coverage_v1/{protocol,manifest,evaluation}.json`的24/8/8协议及已审计support12。
  不使用support-diversity大allowlist，也不把source71自动当本批Writer训练范围。
- 共享训练global IDs为`0,1,2,4,5,7,12,13,14,15,17,19,20,21,22,25,28,29,32,34,35,36,37,38,42,43,51,55,56,62,64,73,95,96,97,101`，
  等权轮转。support对应LIBERO-90 local IDs `2,3,11,15,16,22,24,33,55,56,57,61`，provenance复用既有协议与语义排重。
- source复用`runs/outputs/pi05_source_aligned_seed7_1k_20260915/checkpoints/step_00001000`；
  tokenizer、source normalization及38-target映射沿既有canonical消费者。基础权重和原生图文prefix冻结。
- 强MT起点为既有coverage MT step300的`lora.safetensors`；按清理后的现存只读路径复用，禁止复制整套历史训练状态。
  其输出侧用途是初始化新的可训练完整参数解码器；其固定probe可供教学特征提取，均不构成最终的第二adapter。
- 新教学condition只输入exact language及指定action-hidden RGB视频。K=1，stride5及真实末帧；不声称已经验证dynamic-K。
  不读teacher配套动作/state/reward/terminal或held离线训练资料，不把task ID、文件名或初态编号送入Compiler。
- 自身真实实践的RGB、proprio、实际动作、后果及success/reward/terminal可以输入经验通路；模型推断与真实执行量分别标记。
  首批采用下节列出的实际可获取子集，不为凑齐所有合法状态构建额外特权模块。
- 新condition不准备动作训练集、不运行轨迹拟合或末端策略优化器。共享参数在编译期冻结，只通过已学前向规则形成和修订LoRA。
  非held共享训练可使用动作/FM及真实终局回报训练这条规则；validation/Test不会更新共享参数。Test本批不打开。

## 3. 完整计算链：直接输出全部rank128因子

令共享参数为χ，条件内参数状态为Q_j，全部自身经历为E_≤j。
初始`Q_0=Iχ(Readχ(language,V,empty))`；第j次修订是
`Q_j=Uχ(Q_{j-1}, Readχ(language,V,E_≤j,Q_{j-1}), Encodeχ(E_≤j))`。
每次均以`Λ_j=Dχ(Q_j)`直接解码全部38 targets的A/B，最终只保留一套Λ_J。
U共享权重，循环接口不预设j的目标值或上限；反复读取已有教学也不要求必须先取得一条新episode。
不设置固定J、轮数菜单或按J配额训练，正式编译也不规定两轮、三轮或其它轮数。J只作为实际发生次数的事后记录。
首批采用下述证据与动作预算驱动的停止规则；今后可自主修改这项具体方法，不把它升级为Owner硬约束。

### 3.1 教学与经历怎样影响参数状态

沿用专家可实施特征来源：每教学时点保留双相机原生prefix视觉embedding Φ（合计512×2048）和固定MT原生probe H（50×1024）。
probe使用既有noise seed1729、time=1、完整合法图文prefix、teacher state segment省略；不引入teacher state或动作。
原始Φ/H可以缓存；反馈改变空间query之后再做8-slot读取，不把旧压缩slots当全部教学事实。
slot读取宽256；4层时序处理保留真实帧位置、首末状态与有序变化；Q为38×128×256的target/rank状态，初始化器4个轴向块。
每层/每rank的Q携带实际参数位置身份，language与经历调制其读取；坐标身份不携带task身份。

实践由当前Λ执行真实ODE。每次replan记录实际RGB/proprio、真实执行的最多5个7维动作、下一观测及结果；
保存本次原生调用真实捕获的time=1和time=0.1 action hidden，不能用事后另造的probe冒充实际执行hidden。
事实通路直接编码这些量；教学—自身关系通路允许关注差异，但不能先按相似度筛掉不匹配的失败事实。
每个decision的事实/关系token经2层有序经验编码后读出16个slots，保留episode和时间身份；各轮累计经验参与重读。
U使用2个共享块，gated更新初始化温和（gate bias −2、末投影零初始化）；不以门控名称声称已经保持旧能力。
学习必须对经验编码、空间重读、I/U及完整解码器共同求导，不能只训练输出头或把中间Q截断为另一个课程。

### 3.2 MT只初始化可训练坐标，不作为固定部署加和项

每个A或B按实际矩阵轴切成64数值一块；共享解码器的一个输入为该target/rank的`LN(Q_l,r)`（256维），
另一输入是可训练坐标embedding `e_l,s,r,c`（128维，s区分A/B，c为64维块位置）。
直接输出为`d_l,s,r,c = W2 SiLU(W1 [LN(Q_l,r),e_l,s,r,c]+b1)+b2`，尺寸384→512→64。
按真实形状组装`A_l∈R^(128×d_in)`、`B_l∈R^(d_out×128)`；B的块轴须与实际PEFT消费者一致，不可用reshape暗换行列。
这是一个共享非线性坐标解码器，坐标在非线性前进入；没有固定低维线性输出basis，也不因此声称任意参数映射都能学会。

初始化时e前64维复制对应MT块，后64维以固定module seed随机初始化；前128个隐藏单元成对读取`+e[:64]`与`−e[:64]`，
输出取其差。恒等式`SiLU(x)−SiLU(−x)=x`使初始完整A/B等于MT；其它隐藏单元正常初始化，其输出列初始为零。
全部e、W1/W2/b及Q/读取模块均可训练，既无冻结MT skip，也无`Λ_MT+Λ_residual`出口、因子拼接或生成后截断SVD。
初始条件分支尚无功能是初始化事实；输出头开始学习后应获得完整真实功能梯度，不把第一步零读取梯度误判为永久断路。
不用MT初始化新Writer的optimizer、scheduler或旧Q；除明确的坐标数值外均fresh。

最终每层使用`W_src + B_l A_l`，rank=alpha=128、缩放1。physical PEFT保持identity，MT不得又在base里计算一次。
相比专家冻结MT＋rank64修订，这解除了“相对MT只能追加rank≤64算子”的限制，同时放弃了固定MT路径本身的不变性。
完整参数可变不保证控制保持；FM与keep只提供有限函数约束，最终由闭环得失裁决。
解码坐标可分块执行以控制activation；这是同一网络的物理分块，不得停止梯度、改变loss权重或削弱输出自由度。

## 4. 共享学习如何教会这条循环

### 4.1 一个完整学习事件

每个更新含4个condition，task等权轮转；每task teacher按无放回袋取样。
为每个condition取同task、不同teacher episode的7条query episodes，各按4个时间区间取1个query，共28个。
复用既有action offset1、50步末动作padding及前7维FM消费语义，不在本批悄悄更改padding mask或动作归一化。
先形成Λ0，再按下述开放循环采集当前策略实践并重读修订；最终Λend另在两个独立初态采集SDE query episodes。
这两个query回报只作合法共享训练的标签，不送回该condition的编译上下文；held编译没有这两个额外带训练信用的rollout。
每个condition内部实践、query与最终读出初态分开；跨condition即使初态编号相同也独立执行，不共享经历或参数。

每个condition登记最多1024次适应环境step（含settling），不登记实践或重读次数。当前可执行规则为：

1. 教学先编译当前Λ，在未参与最终读出的合法初态开始实践；以当前Λ执行，保留真实结果。
2. 若自身实践已完成任务，停止本condition适应并冻结当前Λ；这是合法自身success触发的结束，不是查看最终评测后选点。
3. 若未完成，经验进入U并重读教学，直接形成下一套完整Λ；动作预算仍有余量就继续实践，耗尽则交付当前Λ。
   预算只够部分episode时保留真实截断标识，仍使用已获得的事实修订；不伪造terminal或把预算耗尽当任务失败标签。
4. 每次重置消耗实际settling预算，选择确定性初态流中的下一个非final初态；不把初态池大小暗设成轮数上限。
   必要时可在适应池内复用初态，记录复用；final初态始终不参与该condition适应。

这里的停止证据不保证跨初态能力：一次自身成功可能来自容易初态；在新初态的query/FM/正式评测正是检验这一局限。
它是首个可执行的自适应结束策略，不是要求未来一旦成功必须停止，也不是学成了最优探索/停止控制器。
Owner进一步强调：达到有效调整时重读越少，学习效率越高。当前实际首次成功会结束适应，使已取得能力的条件不再强制重读；
但这个程序没有专门学过最优停止，成功也可能来自容易初态，须由新初态能力和完整成本验证。
记录初次读取/重读次数、全片等价教学读取量（实际读取帧数之和除以该video总帧数）、实践steps/重置、GPU时间及缓存复用。
读取固定特征也计教学重读，原生图像编码成本单列，不能因缓存了Φ就把后续读取记为零成本。
在相近能力与相同任务/视频/初态口径下优先采用更少读取、更少实践和更低计算的方案；同时保留失败条件的消耗和尾部成本，
不只统计成功样本，也不能以忽略教学或提前放弃得到的低次数冒充高效率。首批不凭任意惩罚系数牺牲闭环能力追求少读。

首批在新实践结果形成后修订，先用episode结束/预算截断作为实现切点；切点不构成轮数限制或禁止未来中途重读。
训练和正式编译使用同一规则，实际次数随自身结果、episode长短和剩余预算产生；不把训练覆盖事先限制到几个J。
54次共享meta更新提供216条完整条件链、每task6条；这是共享optimizer的首个记录节点，与每condition观看/实践轮数不同。
实际轮数、交互长度、success停止/预算停止比例和task覆盖全部报告；不能把FM queries或多次输出虚计为独立映射。

### 4.2 功能监督与终局信用

保留真实原生FM：`a_tau=(1−tau)a+tau ε`，`tau=0.001+0.999 u^(2/3)`，完整50×32 latent前向，前7维目标`ε−a`。
同一condition各Λj复用同一query/noise/time。J≥1时末阶段权重1/2，其余J个阶段各1/(2J)；J=0（warm或首次实践已成功）权重1，keep为空时取0。
keep为各j/query的`ReLU(ell(Λj)−stopgrad ell(Λj−1))`的平均，系数0.2；它不替代示范外恢复或闭环保持证据。

SDE核复用已核实的真实消费者：`m=z−h[(1+(1−tau)/2)v+z/2]`，方差`h*tau`；
loss对v的cotangent为`Adv*(1+(1−tau)/2)/tau*(z_next−m)`，完整50×32坐标参与score。
每episode最多reservoir16个replans、每replan抽2个真实transition；按`N/min(16,N)`及`10/2`补偿采样。
两个query对condition平均，再对4个condition平均；不得复用旧实现固定/16，物理GPU数也不能成为科学归一化。
value baseline只读当前query前可用context/init信息并detach，待当前actor信用计算完后再拟合本批return，避免自身标签泄漏进baseline。

一次meta更新中χ从实践采集到query、cotangent及replay保持同一版本；原始经历和实际捕获H作为停止梯度的环境证据。
仍对所有共享编译模块及所有阶段Λ共同回传；忽略实践分布对χ的导数是明确的semi-gradient取舍，不冒称无偏主动探索学习。
FM与终局信用联训：fresh AdamW，LR3e−5、weight decay0、clip1、constant scheduler；warm→meta保留同一optimizer连续状态。
β在首个FM和PG梯度均非零的合法训练batch按`min(1,0.25||g_FM||/||g_PG||)`一次定标并冻结；
若最初批PG为零则延后定标、如实登记，不能用虚构epsilon奖励制造梯度；若整批始终为零，则本批没有实质PG学习。
1/8条件按预登记采样屏蔽经验编码器输入，仍执行同一success/资源结束规则，用于让null读数有有限训练支持；实际task/长度曝光须记录。
null仍有真实实践和外部结束规则，不能称完全无反馈；它共享参数且样本少，也不能称独立充分训练的无交互方法。

## 5. 实施、首批学习与读出

本批是完整候选的首次真实学习，不以“机制smoke”冒充能力验证，也不把全部formal计算改名为诊断来隐去成本。
实验session从main新建独占codex分支/worktree，依据code-architecture-gate复用现有资产、native执行、功能信用和调度owner；
旧Writer只作历史读取消费者，不再新建平行fallback。保留一个canonical可继续学习/编译/评测的运行面，退役多余临时入口。

1. 实现完整循环、checkpoint/cursor及实际执行，做必要的信息墙/shape/finite、同版本信用、完整MT初始化行为和resume验证。
   profile使用合法non-held任务，最多12000个环境steps；比较真实microbatch/解码块/帧chunk吞吐与显存峰值，
   不按固定J造学习/评测分组，不为低位一致重复算，不做学习超参数扫描。
2. fresh单一完整视频主模型，128个Λ0 FM warm updates后直接进入共享meta学习，task/teacher流连续保存。
   不等待主讨论对工程阶段逐项验收，不因warm小面板未胜成熟MT而卡住。首批记录点为meta27/54，保留完整可恢复状态；
   它们不规定condition内要观看或实践多少次，也不关闭本批之后的自主接续。
3. 固定train读出选原36-task manifest顺序中每suite前2个target train tasks，共8个；每task固定2个teacher条件，
   每condition3个全新读出初态，共48行，面板在读分前保存。meta27读Λ0/Λend各48行，meta54读Λ0/Λend/null各48行。
   每condition按§4.1自行形成结束点，同一套最终LoRA覆盖3个新初态；实践次数不固定，两个节点共32条适应链，
   最多32768个适应steps；最终读出240 episodes。null以实际经验/结束点作屏蔽输入replay，保留它的有限干预解释。
4. meta54是首批唯一正式候选；K1 correct strict paired400逐row执行§4.1的自适应编译程序，再在独立final初态固定执行。
   400个合法task-video不重复，最多409600个适应steps；实际发生多少次实践/重读必须原样计量，不强行凑齐或截成固定轮数。
   每row的task-video组成condition，不按task共享经历/LoRA。新旧final state/video schedule及env/policy RNG沿canonical合同，
   实践state流另行冻结，与本condition全部final初态不重合；末点不根据最终结果重做适应或选择其它中间LoRA。
   候选是事前约定，不根据meta27小面板选点；工程无效修复后继续，科学低分不用于改面板或撤销负结果。

首批仅一条主训练线，无held梯度、Test或rank/LR/seed扫描。216条训练适应链最多221184个环境steps，
加432个SDE训练query episodes、train读出240 episodes、formal400 episodes和profile12000 steps。
formal最终执行另按suite horizon计132000个控制steps及4000 settling；其适应费用不能藏在最终400行外不报告。
整体以真实steps/GPUh计费，不再用固定每condition episode数推算或限制适应。
formal保留canonical final init k；实践从登记seed的完整50初态排列中排除k后循环取样，重复时如实记录，不因重复池终止。
train多final面板先冻结3个final IDs，再从其余初态流实践；训练SDE query初态事前独立划出，实践永不使用它们。
自身success可以结束适应；final结果不能改变任何适应选择。同一预算上限、结束规则与随机流用于匹配比较，但实际成本可以不同。

## 6. 成本、结果解释与持续接续

新路径速度尚未实测。实施与完整profile预计4–8小时；首批计算初估12–24小时、50–80 GPUh，最大新增峰值160GiB。
以上是待profile替换的计划量，不是已有测速。首批执行上界为80 GPUh、计算wall24小时及data1峰160GiB，环境费用按§5逐step计量；
实现时间单独记录。正式开跑前由执行方用实测吞吐、最长视频、checkpoint/缓存峰值核算，采用真实有收益的并行配置。
全部新增代码/缓存/输出在data1；先查strg01独立quota及现有用量，复用大资产，不另造0.6–0.8TiB缓存。
同时现场检查gpu01/gpu02，执行现行总6/8卡、单节点≤6及不干扰他人任务规则；单训练不跨节点拼rank。
物化/评测可与就绪工作合理并行；日志等待使用退出事件，不按固定间隔轮询共享缓存或给main发心跳。
预算明显不符时交main具体成本和可执行调整；main自主改本批投入或方法，不转交Owner例行审批，也不能只等待不处理。

原件保留真实teacher/state映射、完整checkpoint和optimizer/cursor/RNG、每轮原始经历、FM/PG实际尺度、raw evaluation rows、
per-task/suite、breadth、retained/gained/lost/churn及完成/异常/计费。当前只有一个正式节点，不能冒称相邻稳定性已经成立。
训练域Λend优于Λ0且held转移改善，增强“反馈教会修订”的支持；仅共用先验变强或null相同则支持替代解释；
训练有改善而held缺失主要收窄迁移假说；FM/SDE改善而正式ODE失效须保留目标错配解释。
首批54次共享meta更新每task仅6条链，即便阴性也不证明完整共享修订不可能；但“尚未充分”不能成为无证据续训的永久理由。
主讨论据实际学习速度、覆盖、控制得失及同等能力下的读取/交互效率决定增加学习、匹配端点参照、改变主要算子/监督或退休弱假说。
不恢复专家+4/48、180/400、80%等临时硬门槛；最终以完整能力、正确教学有益贡献及稳定性接近Owner目标。

整批完成或确有科学/资源边界需main裁决时，实验session仅回报一次到Owner最新确认的主讨论`01a11b05-3460-7eb0-aca8-177d6d86ef48`，
active默认Steer；idle/notLoaded通过既有app-server无设置覆盖地resume原session、直接turn/start并核实实际处理，首行包含agent来源。
Queue只用于明确要求延后或恢复原已接受消息；普通实现/冻结/profile修复自行闭环，不另开批准请求。
main消费原始科学证据后主动接续，批次完成不关闭EMBER。科研tracked/Git窗口按progress串行移交，禁止双方并发改同一树。

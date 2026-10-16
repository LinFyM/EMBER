# Source/rank8控制空间编译器：参考控制器与实施合同

2026-10-15；科学owner为主讨论01a13dd9-837d-7f33-b15a-4eb170bcf04b，唯一实验执行为
01a13dbf-dbc3-7621-82be-87be44a5bca6。Owner已授权自主落实、分析问题并接续，无例行许可停点。

## 1. 目标、当前窗口与历史边界

最终目标仍是合法教学/自身实践编译出唯一完整LoRA，在固定held协议明显超过强MT，同时说明教学作用和能力保持。
当前后继从冻结raw source起步，所有生成、实践和部署LoRA均为38-target/rank8、alpha=8、dropout=0；
不以MT初始化、合并或回退，不部署额外高rank分支。强MT153、G165、D175/168与所有负证据保持原身份。

完整方法的候选原理见[控制空间编译分析](../analyses/policy_space_meta_compiler_proposal_20261015.md)：
固定已验证的D(z)产生完整r8；冻结视频起点，以真实H条件的跨episode功能监督学习完整参数修订。
旧经验编译已有真实反馈学习，本次不是新的反馈架构原理；已知控制族有训练域先行证据，跨任务迁移和有用经历修订仍待实证。

**当前实施范围为§18：固定唯一ψ180/Reader900/linear64，完成validation8×50的真实编译与strict paired400。**
§2–17均保留完成/失败/恢复历史。§17在训练task/已见initial-H上89→93/144（R85/G8/L4）；
这是进行一次迁移检验的有限依据，不是稳定、held或视频/H因果资格。§16固定单位搜索与§14旧PG不恢复。
linear64标签侧484/576、旧Reader900/1800 held117/111与MT153/G165/D175/168均保持原身份。
新批0训练/新reference/重fit/尺度扫描/其它checkpoint/controls/Test；仅本节登记的validation任务内无梯度适应和最终400。

## 2. 已核资产与选择新参考教师的理由

2026-10-15只读审计没有找到可直接挂当前raw source的非零r8参考bank：现成全36 E是MT128合并底座上的r8残差；
旧rank16源模型不同。T2340有同source的r128参数函数，实际144条读出109成功，但每套仅一个init证据，
并非独立任务expert或已证明能压到r8。直接用它训练D会混淆教师容量与decoder学习失败。
因此本次先新训练同source、同rank参考控制器；旧资产只作历史参照，不投影、恢复或加载为本批教师。

唯一source：`runs/outputs/pi05_source_aligned_seed7_1k_20260915/checkpoints/step_00001000/policy/model.safetensors`；
固定normalization：`configs/pi05_source_corpus_v1/source_normalization.json`；
任务/评估：`configs/libero_24_8_8_coverage_v1/{protocol,evaluation}.json`；
38-target形状由`configs/pi05_lora_rank128_aligned.json`读取并通过现有derive接口只改rank/alpha为8。
tokenizer、LIBERO assets与50示范HDF5均原地复用。source只有冻结推理权重资格，不称可exact-resume的source训练状态。

首窗八任务固定为`[0,1,12,13,20,21,32,34]`，均属于train24，每suite两项，继承既有内部面板而非按新结果选任务。
完整将来池仍是train24＋已审support12的36任务：
`[0,1,2,4,5,7,12,13,14,15,17,19,20,21,22,25,28,29,32,34,35,36,37,38,42,43,51,55,56,62,64,73,95,96,97,101]`。
首窗不训练额外28任务；其接续现在由§8登记。support12的source已见身份不改变，不能把36全部称source未见。
审计原始report/assets/resource_estimate由执行在首个run的inputs中保存一次，当前临时来源为
`.codex/tmp/source_compiler_asset_audit_20261015/`；新增run contract另明确实际采用八任务，不把全目录资产当计算许可。

## 3. 参考LoRA训练

每task独立从canonical `identity_lora_state`开始：固定非零A、B=0，全部76个r8因子参与训练，source所有参数冻结。
不读旧teacher权重作初始化，不加MT proximal，不把r8残差放在隐藏MT底座上；每task只有一条fresh训练轨迹。
这是授权non-held训练监督侧的参考函数学习。新教学condition不收集动作后拟合最终LoRA。

数据为该task原始50条合法成功示范；使用真实RGB、8D proprio、exact L、7D动作及现有source normalization。
复用`FunctionalQueryDataset(action_chunk_size=50, action_start_offset=1)`和原生FM：独立Gaussian noise、
既有Beta时间采样，完整50步、真实7维的`mean_velocity_loss`。不以末端5步替代整个FM horizon。
logical batch固定28个query；episode与frame按登记seed7的permutation/cursor均匀轮转，保存实际采样行与noise/time root。
本task expert没有教学condition，因此可用全部50 episode；后续共享Writer的主项仍须排除当条teacher episode。
不把expert训练动作、state或路径输入后续teacher reader。

AdamW：lr=1e-4、betas=(0.9,0.95)、eps=1e-8、weight_decay=1e-4，global grad clip=1，constant LR；
全程同一optimizer，不作LR/rank/seed/scale菜单或重启挑峰。FP32因子/optimizer、正常BF16/TF32原生计算允许。
每次更新先按logical28完整平均再clip/step一次；物理microbatch与设备调度不能改变数据流/任务权重/有效batch。
首窗每task到960，预登记候选更新`[80,160,320,480,720,960]`；均保存完整因子、Adam、RNG与cursor。

本窗还明确授权同轨迹有界延伸：如果960时所选最好点后尚无两个已完成候选节点，继续该task原训练轨迹，
按`1200,1440,1680,1920`逐点读出，直到最好点后已有两个候选节点或到1920。其余task不为凑齐步数延伸。
本规则提供末端上升的接续与相邻稳定性证据，不证明数学上的全局峰值；1920是首窗结算边界，非充分训练证明。
到1920仍在上升/未有两点后验时，回报为“选点尚未收束”，主讨论消费后接续；不得提前宣布参考bank已准备好。
若早期整体已显示明确无效或出现真实边界，携具体闭环/学习证据报告main裁决，不为履行固定步数盲跑。

## 4. 充分选点与独立读出

每task固定selection init0..31，共32初态；source identity和每个预登记候选都跑同一面板。
不以80-row screen/训练loss挑正式Writer；这里是non-held教师选择，不赋予最终方法strict held400资格。
所有候选严格配对env/policy RNG、初态、official preprocessing/horizon与noise schedule，参数身份不进入噪声种子。
selection root固定202610151；init0..15复用旧正式MT128请求中的配对身份，可与其唯一67/128参照作限定比较；
其余16按同一规则事前物化。只读复用必须有实际request来源，不能用旧noV74替换或重跑选择MT高点。
旧MT身份若无法与新case完全对齐，明确只作背景数值；不改新case或新增MT执行来修饰比较。

候选按32初态真实成功数最大选择，平局取更早update；source identity为update0并纳入比较。
不按loss、因子误差、单task特例或report成绩改选择，不跨checkpoint融合。记录每task全部候选和相邻成功集合。
选点收束后冻结每task一个teacher，才读取独立report init32..49（18初态），与source identity严格配对，root202610152。
report不产生梯度、不回流该teacher的选点或训练延伸。没有收束的task先不打开report，真实报告未完成范围。
参数结构性仍为source时，预先共用同case source报告行，不重复执行制造“学习差异”；保留逻辑/物理计数。
不要求所有task过同一人为成功率阈值，不隐藏难例；无有效改善或成功稀疏的teacher不能仅因有文件而称强监督。

执行严格使用render256/model224、双相机180度rotate、8D state/7D action、完整10-flow、执行前5后replan、
dummy settling10、成功即停和220/280/300/520 horizon。evaluator为动态cost-balanced/long-first/persistent queue，
不能静态按task/GPU切评估；训练与已经就绪的评估按live资源并行，不让它们等待无关task完结。

独立report的selected teacher执行保存真实双相机uint8 RGB，原control时钟stride5并包含终帧，明确已采样索引，
后续不得再stride5造成stride25。保存实际producer完整LoRA身份、初态、camera/preprocess与成功/失败manifest。
完整保留失败的成本/记录；仅成功RGB可成为辅助正确教学，动作/state/reward等标签不得进入教学内容。
不以原始离线teacher视频和这个新expert作虚假的producer对应。当前只采集，不开始D/q训练。

## 5. 功能监督在完整方法中的位置

参考控制器被真正准备好后，标签侧E读取合法训练teacher的r8因子，产生z_ref；D(z_ref)也输出完整r8。
同一真实非held观测/proprio/noise下，reference与D输出分别跑完整10-flow，比较实际前5×7动作，
用固定source动作单位归一化后计算均方功能误差。reference为no_grad，梯度回E/D；这不是环境成功的梯度。
原始视频另通过跨episode FM训练video→z→D，保持原分布；真实reference成功视频才可携其producer的z_ref作为标签侧目标。
多个同task视频指向同一teacher并不证明视频增量；固定共同E的旧失败边界保持，后段不能给所有H机械重复同一个E标签。

整个系统仍输入exact L，但Compiler不直接接收task名称、L tokens或教学的语言融合hidden；
教学使用真实冻结图像塔patch。执行policy与监督query正常读exact L，自身实际hidden中的间接语言信息按Owner要求保留。
后续D/q/全路径RL的实际loss权重、完整数据流、节点与资源须在本design续节冻结后再运行；不是本窗的隐含后段。

## 6. 计算与存储边界

首窗最大训练更新8×1920=15360、logical FM query最多430080；实际只对触发延伸的task使用对应节点。
最多10个非零候选×8×32=2560 selection rows，source selection256，selected/source report各144：
最多3104个逻辑完整episode、1,024,320 control、31,040 dummy，总1,055,360环境步。
正常工程失败额外最多16次完整尝试，按每次520+10保守记账：总最多3120提交、1,032,640 control、31,200 dummy、
1,063,840步。科学失败不重试；有效行不重跑；未知执行保守占额，不能把一次失败扩成自动重复训练。
消费者须执行同一全局账本；profile仅使用上述有用训练/评估工作，若确需独立环境smoke须占16次尝试额度并保留记录。

canonical ROOT为`/data1/user/ymdai/ember_runs/source_policy_space_compiler_20261015/reference_seed7/`。
单套r8 FP32为643584标量/2.455MiB。峰值新增预估24GiB，包含全部候选完整状态、selected/raw RGB、原子保存及小缓存；
selected report若逐control存双RGB约17.5GiB，按stride5保存约3.6GiB；不将此预算继承给以后完整H/meta-RL。
source、数据、tokenizer、参考历史bank不复制；CPU缓存按全部worker合计有界，不预建全1800视频prefix池。
launch前在strg01核data1独立user quota/相关目录使用及共享容量，按实际实现schema更新峰值；不能切data0。

每次launch/resume同时live核gpu01/gpu02；按AGENTS合计6/8、单节点6、真实显存/进程与不挤掉他人任务准入。
无累计GPUh硬门槛，仍逐项记账。独立task训练不要求NCCL；如使用DDP，固定P2P禁用、NUMA/deferred NCCL合同。
首批实际训练段比较物理microbatch7/14/28（均同logical28）及有意义的共驻/排程；有显存余量须验证能改善吞吐的安排。
不只看GPU显存：同时记录CPU总PSS/线程/进程、环境数、samples/s与episodes/s。完整评估回执边界close空闲runner，
再需环境时按pending请求懒建/扩缩；仅release_renderers不算CPU释放。数值库import前控制子进程线程，
不从已初始化CUDA父进程随意fork，不丢RNG/episode/绝对时钟或以释放资源重跑有效行。

首次实施加首窗暂估3–6小时；有界延伸可能额外1–3小时，均为未profile估计，不是截止/强杀时刻。
执行用本批真实吞吐修正ETA，出现显著变化解释具体依据，不套用N16不同负载的吞吐。

## 7. 工程、验证、记录与回报

执行独占code/记录/Git窗口，在最新main的独立codex分支/worktree实现；main派发后只读，避免重叠写。
结构按code-architecture-gate守则：优先复用pi05_lora、DirectLoRAParameters、FM/query、native_actions、
checkpoint/RNG与动态queue；源模型/完整r8运行面只有一个canonical owner，不沿用硬加载MT的旧Runtime。
不为reference复制一套完整Writer框架；旧MT-prox/selector入口若被替代，按真实依赖退役或登记明确的暂留owner/移除触发。
架构大小/复用和生命周期由执行闭环，不成为main重复工程审查或Owner许可门槛。

针对实际消费者核实：source冻结且无MT载入；完整38r8身份/初始B0；真实FM更新和保存恢复继续同一逻辑流；
相同case候选配对、选点/独立report隔离；官方native10/first5；实际close后的环境进程与资源释放。
使用必要的小检查与本批真实工作，不加泛用测试框架、逐bit比较或新hash sidecars。
formal train/eval来自clean pushed detached worktree；故障修复经独占分支和针对性检查后新冻结，原失败/有效产物/费用保持。
checkpoint保存全部因子、Adam、step、scheduler状态（constant）、采样cursor、Python/NumPy/Torch/CUDA RNG、topology与schema。
按完整checkpoint边界承接恢复，迁移物理安排如实登记，不称bitwise exact。

整批交付包含逐task训练/候选/selected与是否收束、全部raw rows、per-suite、source/可匹配MT的保持得失、
相邻success-set overlap/churn、独立report、辅助视频真实producer清单、实际时长/GPUh/存储/CPU与吞吐、失败/未知及代码身份。
不宣称参考8任务证明36/held迁移；main亲自消费关键原件并作科学接续。
唯一owner直接等待整批退出并完成验收/Git后，一次回主讨论UUID；无心跳、普通ACK、阶段Queue或第二waiter。
真正边界或明确科学失效可提前携具体证据回main。到边界停止当前计算不等于项目等待Owner新许可。

## 8. 首窗科学消费后的参考库补齐窗口

### 8.1 决策依据与不变项

main已亲自核对首窗raw selection/report rows、原生FM/query/恢复实现与全部task轨迹。
7个已冻结task独立report从source26/126到103/126，保持22/得81/失4；这支持同source/r8参考函数可以学成，
因此继续准备完整已审计36任务池有实际依据。它不证明低维D、视频读出或held迁移已经成立。
task20/34的弱点、task21较高source能力和task13/34后期下降均保留；不依据report回调其教师。
首窗原件与结算保持在`analysis/engineering_completion_20261015/`，不覆盖原8任务结论、首次时钟和费用区间。

当前只增加以下两个互不依赖、可并行执行的工作：

- task0从末1920完整Adam/RNG/cursor续到2160；新增selection32，再按原规则冻结selected，之后才打开report18/source18。
  不从1680重启，不回放1920以前更新。1680已32/32且平局取早，2160读出用于补齐第二个后继节点，
  不是另搜更高分；本次不继续2400或改用末点。若真实恢复失败，保留事实并按工程边界处理。
- fresh训练其余28个已审计任务：`[2,4,5,7,14,15,17,19,22,25,28,29,35,36,37,38,42,43,51,55,56,62,64,73,95,96,97,101]`。
  每task仍同source identity、38r8、50合法示范、logical28/seed7/原Adam/FM/六个首窗节点；
  960后按原两后继规则只在需要时延伸1200/1440/1680/1920。没有重新训练首批已冻结7个task。

target16按原四suite horizon；support12复用已审`operator_seen_training36` authority、global ID=40+local ID、
horizon400及对应合法HDF5/50初态。不可把support当四suite任务做整数除法，也不扩到其它LIBERO-90任务。
selection仍init0..31/root202610151，冻结后report仍init32..49/root202610152，所有信息墙与真实RGB规则不变。
旧MT128只覆盖首八任务，其余28不补造MT配对、不新增MT评估；全36共同参照是同case raw source。
若某task在1920仍未收束，明确保持未准备好且不打开report，整批携原件回main自主接续。

### 8.2 原件、运行面和资源

继续使用同一canonical ROOT `reference_seed7`；按显式修订记录扩展计划/合同，完整保留首窗冻结输入、原队列与结算来源。
旧有效selection/report/CP只读复用，已冻结selected不能被新run的选择函数改写。
一个canonical reference运行面承接显式task/节点合同，按实际原消费者作最小修改；不复制第二套trainer/evaluator。
task0恢复保持原训练checkpoint语义；新调度代码身份与原训练来源分别登记，不称bitwise exact。
工程由执行在独占分支完成、通过实际消费者检查、集成/push后使用新的clean detached冻结版本。

新增最多54,000 optimizer更新/1,512,000 FM query，其中task0为240/6,720，其余28至多各1920。
新增正常逻辑episode上限10,932（28×388加task0的68），control3,926,000、dummy109,320、总4,035,320步。
原全局16次工程额外尝试尚未使用，本次继续共用这16次，不重置或另加16：
新增物理提交最多10,948、control3,934,320、dummy109,480、总4,043,800步。
连同首窗实际结算，ROOT累计上限为65,040更新/1,821,120 query、13,440提交、
4,522,123 control＋134,400 dummy＝4,656,523步。历史首次6 loader启动失败GPUh另保留，不能记0。
有效行不重跑，未知尝试占完整上界；科学失败不是重试理由，扩量不刷新工程失败额度。

新增峰值估计48GiB：新source/selected RGB按stride5加terminal的未压缩上界约27.93GiB，
加28任务完整候选训练状态、临时原子保存、小缓存及余量；已有ROOT约2.494GiB另计。
不预建全部video prefix池，不复制source/数据/旧CP。执行launch前用真实schema复核并live检查strg01 data1 quota、
相关目录和共享容量；不足先回报具体缺口，不能自行切data0。

依首窗实际14.58 GPUh/8task和2.08–2.13s/update，28task加task0预计约7–12小时，
取决于live可用6–8卡、任务horizon、成功早停及多少任务延伸；若大多到1920且失败长轨迹较多，可能到12–16小时。
这是含少量工程准备的wall估计，不是GPUh上限或强杀时刻；原09:05:48.038首窗派发保持，本次另记实际派发/启动时钟。
每次launch双节点live准入，合计6/8、单节点6；独立task可跨节点并行，不以旧world size人为锁死卡数。
复用实测microbatch与环境PSS/吞吐记录，变化时在本批有用工作中调整；CPU准入按节点合计而非把两主机内存混成一个池。
训练与就绪评估并行，真正idle环境完整close；不新增无判断价值profile或固定周期轮询。

### 8.3 交付与后继边界

输出完整36任务coverage/未收束列表、所有候选及相邻得失、冻结的唯一teacher、独立report与真实producer视频。
报告target24和source已见support12的分组结果、task/suite分布、source保持/新得/损失；不能只报池平均数。
弱teacher保留其资格限制，不根据report挑任务或重新选点；后续D训练不得把不同task数量当作同task视频数量。
本批不训练D/q/meta-RL，不读held/Test或sealed标签，不恢复旧N16/MT教师。
main在执行期间完成控制空间的具体学习与闭环判断合同；原件返回后自主分析接续，不等待Owner逐项许可。
执行接管canonical工程/记录/Git，main只读；只在整批退出验收/push或真实边界时回主讨论一次并归还写窗口。

## 9. 参考库末四项的有界收尾

### 9.1 原件消费与决策

2026-10-16主讨论亲自按global_task_id重算原始selection/report：32冻结独立238→496/576，
保持232/得264/失6；其中目标21项58→305/378，source已见support11项180→191/198。
五项选择source identity，另两项独立success-set与source相同；不把它们的高分记为新能力。
task4/29独立7/18与2/18、task34失3个source成功均保留，不按report回调这32个教师。
这批证据支持同source完整r8可以提供广泛参考控制，但没有证明D、视频编译或held泛化。

四项未收束各有具体继续依据，而非为了凑齐文件：5在1680达31、1920为27且只失4；
25近1200/1440/1680/1920为15/17/22/21，末相邻得5失6；38为10/9/13/11，末相邻得6失8，
能力仍弱且交换很大；43为29/30/30/31，最近两次净增分别0/1。
25与38不能被包装成平稳控制，38也不能仅因还有上升空间无限续训。
同一配方继续少量节点，能区分近期高点是否被后续超过，并满足原选点证据要求；不从best重启。

### 9.2 唯一新增计算与停止

只对`[5,25,38,43]`，从各自末1920完整Adam/RNG/cursor恢复，依次允许`2160,2400,2640,2880`。
每节点仍为原selection32，按成功数最大、平局取早；最好点之后已有两个完整候选节点即停止该task，
冻结原轨迹中的唯一最好点，才打开原init32..49/root202610152的selected/source各18及真实RGB。
不是每项必跑2880。5/25/38若2160不创新高，即在2160收束；43至少补两个节点，若创新高按同规则再判断。
2880仍缺两个后继时保持未收束、不打开report，携原件回主讨论；不自动3120、改LR/seed/rank或另起轨迹。
已经冻结的32项与所有旧有效行不重新训练/评估/选点。source/38r8/FM/logical28、原冻结归一化、
任务权重、候选噪声/初态及官方执行合同均保持；source-seen task43继续原support horizon400。

同一ROOT下登记第三个显式窗口，保留首窗与补齐窗的原contract/queue/analysis/owner/费用。
既有条件未触发的pending节点不能被误当作本次待执行；只新增四项合法接续，并复用已经完成的依赖。
训练来源0f8e0023、接续代码及实际恢复拓扑分别登记；新代码用clean pushed detached冻结，不热改旧树。
目前261个完整checkpoint是合法非held函数标签；较差点不是正确教学，独立report也不反哺参考选点。

### 9.3 有限资源、时钟与交付

本窗新增最多3,840更新/107,520 FM query、16个完整checkpoint。
正常最多656新episode：四项各最多4×32 selection及36 report；按horizon220/300/520/400，
control上界236,160、dummy6,560、总242,720步。
继续共用原尚未使用的全局16次工程额外尝试，不重置：新增物理最多672、control244,480、dummy6,720、总251,200。
据前两窗实际45,360更新/1,270,080 query/10,566提交/2,388,321步，新的ROOT累计上界为
49,200更新/1,377,600 query/11,238提交，2,527,141 control＋112,380 dummy＝2,639,521步。
这是已完成实耗加本窗上界，不把§8未用额度转成其它科学工作；全部旧失败、未知费用区间与原始时钟保持。

峰值新增预估6GiB，包含四项source/selected stride5真实RGB、16个完整CP、原子保存及小型记录余量；
原ROOT约9.309GiB另计。执行前live核strg01 data1独立quota/相关目录与共享容量。
双节点live准入及合计6/8、单节点6不变；四个独立trainer与已就绪动态eval可并行，
不能把四个训练task误当整批最多四卡。复用实测microbatch28和eval16，按节点CPU/PSS及真实显存安排，idle完整close。
依实测2.073s/update，最多训练约2.21 device-hours；eval工作随成功早停与Long失败变化。
含最小工程承接和验收预计1–2小时，设备受限或四项均跑满可约3小时；非GPUh硬限或强杀时刻。
本次派发/准备/启动单列，原首次2026-10-15T09:05:48.038Z及前两窗结束时刻不重置。

唯一执行者完成工程、真实退出、原始结果/恢复/费用/资源验收及Git后，一次回主讨论并归还写窗口；
不发阶段ACK/Queue、不设第二waiter。main并行落实控制空间完整学习合同，收到原件后自主接续。
本窗没有D/q/RL/held/Test计算；后段具体范围另登记，不能把参考训练完成当作方法成功。

## 10. 控制空间E/D首个有界学习窗口

### 10.1 范围、历史与实际要回答的问题

Owner持续自主推进授权下，主讨论依据32冻结参考的广泛控制能力登记本段。执行可以在§9实际运行期间实施CPU代码，
但E/D的科学训练须等待全部36项按§9冻结、独立report完成且bank_ready为真；最终selected及其余CP清单在训练前固定。
若§9仍有未收束项，本段实现可完成，不能偷偷缩为32-task训练或用暂优点冒充已冻结教师；整批携事实交主讨论接续。
这项前提确保已经承诺的参考准备完成，不要求每task达到任意统一成功阈值；弱4/29/38及五个source identity保留。

唯一问题是：共享的64维代码和完整r8解码器，能否学到并保留这些已存在的反馈控制函数。
不是视频→参数的成功证据，也不把36任务的若干参数轨迹当作更多独立task或已知好探索邻域。
最近似旧G的BA-only曾有held175/400，而同条件加入native的C为158，且两臂所读函数误差都恶化；
旧native训练是真实梯度，不能以本次“接通native”清零其负证据。原件与机制分析见
[旧同T科学裁决](/data1/user/ymdai/ember_runs/video_guided_proposal_writer_20261014/operator_native_joint_learning/analysis/main_scientific_readout/interpretation_20261014.md)。
本次改变为raw-source/r8同容量函数库、标签侧参数编码及同query双控制器；没有声称找到了旧失败唯一原因。
先用144更新和一次闭环判断学习是否发生，撤回初稿直接1152/双读出的投入安排，不把纸面步数当义务。

### 10.2 输入、模型与恒等锚

模型计算沿[具体学习说明§3–4](../analyses/source_control_space_learning_20261015.md#3-ed的实际算子)，以下字段必须落实：

- source/归一化/完整38-target rank8/alpha8/dropout0和全部36 allowlist原地复用，所有共享E/D参数fresh seed7。
  不加载MT、旧G、Reader值或高rank教师；标签侧只读最终冻结参考清单的完整因子。
- FactorLayout为10,064个64维块、643,584有效标量及512 padding；地址为target/rank/A-B侧/真实坐标，padding mask不参与统计。
  E为既有两层block→rank→target通信，width256/4 heads，38个normalized target作均值后线性映射到64维。
  D为六层同类通信，块输入为固定identity的零归一化坐标和mask/地址；64→256投影的单个memory token承载z。
  不把完整参考因子旁路送D，不用task ID、名字、分数或视频作为E/D输入。
- 精确形式为`E(r)=e(r)-e(0)`，`D(z)=Lambda_id+S*(h(z)-h(0))`；identity的A固定非零、B零，输出仍完整76因子。
  两个anchor分支都参与真实导数，不detach，不缓存跨optimizer更新的学习特征。D输出头零初始化，首输出为source。
  第一更新E或A侧尚无梯度可以来自此初始化，不要求每张量首步非零；后续真实信用缺失须区分工程与科学原因。
- S共76个因子tensor RMS标量：对每tensor真实元素的identity差求均方，按每task selected1/2与其它非零点合计1/2，
  再对36task等权平均、开方、floor1e-6，随后固定。它是non-held参数单位，不重算动作/state归一化或用held统计。
  最多277个非零CP来自36条轨迹；原32 selected及四项新selected冻结后不再因D成绩改变。

最终生成LoRA执行为`W_source+B(z)A(z)`；没有第二adapter。E只用于训练标签与本次函数保留读出，
后续视频编译不能检索held expert或读取其参数。D只有z输入，执行policy仍正常读取exact L与自身观测。

### 10.3 真实查询、学习目标与可复现流

每逻辑更新四个task、每task七个query，同时监督该task选定参考与另一个完整非零CP，两个点各权重1/2。
task按seed7的36-task permutation循环；144更新使每task恰好16次访问。其它CP排除selected，按确定性shuffle周期轮转，
各task权重不随CP数量变化。selected为source时仍占自己的1/2；其恒等零损失不转成其它task或点的额外权重。

查询来自该task已有50个合法原始episode的RGB、真实8D proprio及exact L；复用现有episode/frame均匀cursor，
每task第v次访问用它的第7v..7v+6个query。原processor将真实proprio按冻结source量化并写入原生prompt，
图像沿官方处理；不能以teacher缺失state构造fake输入。本段没有教学condition，query标签仅属non-held监督侧。
两个点共用相同x与同一个真实50×32 Gaussian noise，noise由阶段/seed7/task/visit/query身份确定，保存逻辑流而非worker顺序。

每点目标为no_grad的真实参考native10响应，学生为source+D(E(reference))的真实native10响应，取实际前5×7归一化动作：
`L=mean_task mean_query mean_two_points mean_5x7 (a_student-a_reference)^2`。
这是唯一loss，无BA、因子MSE、额外离线动作BC/FM项、对比margin、success梯度或MuJoCo求导；source/reference冻结，梯度回E/D。
full50×32内部flow与十步链不能缩成一步，实际生成因子在每次checkpoint重放中重新绑定，不能detach学生参数或使用旧MT Runtime。
38个LoRA target均在suffix；相同真实query的冻结prefix KV可复用，不跨不同RGB/L/state或改变prefix权重的模型误用缓存。

同query两个参考响应不同时，忽略z并输出同一c的最小配对MSE是`mean((a-b)^2)/4`。
从本次已计算的参考响应记录这个下界、实际MSE与selected/other分项即可，不新建探针池或额外forward。
差异为零的比值记未定义；低于此界只说明响应使用了代码差异，不证明语义、可探索性或闭环成功。
训练query来自离线成功示范观测分布，没有全失败分布覆盖；原report RGB不能臆造成完整proprio/hidden训练集。

### 10.4 有限更新、物理安排与恢复

唯一fresh轨迹为144个有效更新；36/72/108仅完整恢复点，144为唯一读出点，不用loss挑其中一点或融合。
AdamW lr1e-4、betas(.9,.95)、eps1e-8、weight_decay1e-4、global clip1、constant LR。
全逻辑四task×七query×两点均值后一次step；正常FP32参数/Adam和BF16/TF32原生计算，不要求逐bit一致。
本窗4,032个观测/noise query、8,064个reference/student配对，逻辑上各8,064个reference和student native10响应；
反向checkpoint重算、OOM/工程失败后的原query重算及恒等合法复用分别记录实际调用/成本，不增加监督query、不记成零费用。

按实际吞吐选择同节点1至6个训练rank，不能把四task当永久四卡限制，也不跨节点拼训练碎片。
任务/两参考/query可物理分片，先按全局逻辑平均累积再同步/clip；DDP平均与SUM不能重复除world size。
首批有用更新测native query batch与两参考共批：先尝试每task的7以及可容纳的2×7，再按实际余量测更大跨task批，
物理拆分不改变query/noise/point权重。只做有用更新，失败算成本；不用保守batch1、最低显存或GPU卡数代表高吞吐。
候选形状无收益或受live显存限制的依据落盘，不为填显存增加query。多卡NCCL/NUMA/deferred初始化沿AGENTS。
Checkpoint含E/D、全部Adam、scheduler、sampler/task与point/query cursor、各rank RNG、S/identity与bank版本、schema和topology。
同拓扑恢复真实承接；完整边界改变物理拓扑须保持逻辑流并记录RNG来源，不称bitwise exact。

### 10.5 唯一末点闭环与判读

144时对每task选定参考生成一次完整`D(E(Lambda_selected))`，36套固定LoRA，分别执行原init0..15/root202610151。
与原真实selected/source严格配对global task/state/env/native RNG及官方执行口径；原参照不重跑，raw rows必须可复核。
由恒等锚结构保证source输出的五项，可复用同case真实source行并明示80个逻辑row的来源，不能称学习。
其余每套生成LoRA真实部署；不按每task最终分选checkpoint或回退reference，不读未选点的report来优化E/D。

至多576个新完整episode；sum_horizon=6×(220+280+300+520)+12×400=12,720，
故control上界203,520、dummy5,760、总209,280步。本独立E/D窗口另登记最多16次工程额外尝试，
物理最多592、control211,840、dummy5,920、总217,760；这不刷新reference ROOT的原16次额度。
有效行不重跑，未知按完整上界计；不新采辅助教学RGB、H或参考动作轨迹。

交付每task/suite及target24/source-seen-support12的source/参考/重构成功集合、R/G/L/churn/重合、breadth，
特别区分参考新得能力被保存多少、source能力被丢多少、五项identity和弱task。训练功能误差只作定位，不替代闭环。
这是有参考参数作输入的训练域函数保留检查、且初态属于参考选点面板；没有视频因果、独立held或相邻学习稳定资格。

- 功能误差与控制均无实质学习：不直接开RL以环境量掩盖D失效；先根据真实接口/目标/优化证据改动具体假说。
- 功能误差改善但闭环明显损失：降低“这些离线查询足够保存函数”的支持，不能默认是工程bug或继续小扫。
- 部分广泛控制取得：结合保持、缺口、学习曲线和实测成本决定是否再给同轨迹有界曝光；16次task访问不宣称已充分拟合。
- 已大体保存强参考控制：尽快转入原始视频→代码→完整LoRA学习，不追加一串latent探针证明几何漂亮。

144结束携完整证据交主讨论；不自动288/576/1152、q/RL/held/Test。停止当前窗口不等于项目等待Owner新许可。

### 10.6 工程、资源及实际接续

新ROOT为`/data1/user/ymdai/ember_runs/source_policy_space_compiler_20261015/control_space_seed7`，
引用reference_seed7/source/数据而不复制。新增峰值预估12GiB，涵盖E/D与Adam/恢复点、36套LoRA、临时保存、日志与有界缓存；
reference ROOT独立计量。launch前核strg01 data1独立quota/目录实耗及共享容量；实际schema预测超此估计时说明具体项，不能转data0。
不预建全视频prefix池，不保存全native激活；同query KV优先进程内即时复用。

复用source-only Runtime、FactorLayout/FactorEncoder、native_actions、数据流、动态评估与checkpoint原语；
只新增E/D职责所需owner，不复制一整套Writer或恢复旧MT运行入口。代码增长/复用/生命周期由执行按architecture guard闭环。
正式train/eval来自clean pushed detached；CPU实际消费者与首批有效更新验证源冻结/anchor导数/完整76因子、同query双点、
真实native回传及恢复流，按实际语义检查，不新增通用测试框架或逐tensor/逐bit审计。

训练和已经就绪的参考/读出工作按live6/8总量、node6并行；各node分别管理CPU/PSS/线程和环境生命周期。
预计实施约1–2h、144更新约0.5–2h、末点576约0.3–1h，合计约2–5h，依据既有native工作及当前FM吞吐的粗估，
不能把2.073s的单FM更新直接当native10反向速度。首批有效更新实测后修正ETA，显著变化说明原因；无GPUh硬限。
§9时钟和费用不重置，§10接收/准备/启动/退出分别登记。若条件满足，参考结束后接续本段不等普通ACK。
唯一执行者串行拥有代码/记录/Git；保持一个实际退出等待者，无心跳、阶段Queue或self通知。
整批完成后验收参考收尾及E/D原件、完整状态/消费/费用/资源/Git，一次回主讨论；真实未收束/科学/资源边界可提前携证据回报。

## 11. 从144接续的直接算子学习窗口

### 11.1 实际不足、单项取舍与停止家族

36参考独立report为545/648对source248；但§10标签侧E/D144在原36×16面板仅216/576，对source218、参考502，
参考新增288例只保存10。main已直接重算576个raw source/reference/generated配对，并读取实际模型、学习消费者与日志。
31非identity的完整BA范数相对参考比率中位2.26254e-5，整体BA相对平方误差中位0.999999966；
原首plan动作MSE中，generated对source为6.98733e-6、对reference为0.044353，reference对source为0.044413。
四个原suite代表0/12/21/34的真实归一化输入RMS约0.936–1.550，E代码norm约0.0128–0.0273，
D头前条件差RMS约0.000371–0.000883，输出归一化变化RMS约3.38e-5–1.14e-4。
这是实际参数作用尚小的定位，不是dtype bug、完整忽略代码、唯一根因或低维不可学的证明。
完整论证及原件见[既有分析§7–9](../analyses/source_control_space_learning_20261015.md#7-144首窗的完整科学消费)。

本次只改变共享E/D的学习目标：停止继续加量当前native-only首段，用全部38层完整sBA误差直接约束参考算子。
E/D结构、64维、76单位S、source中心、两个live锚、完整bank、task/point权重及Adam超参均保持。
此目标绕过当前source附近原生十步流的动作Jacobian，但不能绕过E/D自己的弱条件通路；能否学到仍须实际验证。
旧BA充分优化有过真实控制正例，联合native曾退步；也存在BA变差但闭环上升及后继held不稳，不能称BA普遍更优。
本窗不是目标/曝光量的等量因果对照，不用它单独归因前窗失败。若仍无实质控制保存，消费后修正表示/写出假说，
不自动追加第二个同目标时间块、native权重混合或rank/LR/seed/scale小扫，也不把D交RL掩盖缺口。

### 11.2 完整算子目标与参数/标签边界

继承`control_space_seed7/checkpoints/update_00000144/state.pt`的完整E/D、Adam、constant scheduler、
rank RNG、实际mixed-dtype identity、S与bank；显式登记从native目标迁移至operator目标，不称相同训练目标的exact-resume。
除必要物理拓扑迁移外不重置optimizer或参数；schema及新phase cursor另登记，旧144及所有科学/工程原件保持。
标签仍是266个非零更新checkpoint及5个source选定端点，合计271个不同参数点；没有新增参考训练或中间点环境采集。

对于target l，令`U_l(Lambda)=s_l B_l A_l`，`s_l=alpha/rank=1`。每层固定正常数

    C_l = max(E_bank ||U_l(Lambda_ref)||_F^2, 1e-12)
    L_operator = mean_task mean_two_points mean_38
                 ||U_l(D(E(Lambda_ref))) - U_l(Lambda_ref)||_F^2 / C_l

E_bank沿原36 task等权、selected权重1/2、其余非零点共权1/2（其内均匀）；若无other则两半均指selected。
C_l只由固定训练bank一次计算，76个原S不改变；它是每target的统一函数单位，不是每条件重归一化、held统计或改动作normalization。
参考和C_l全部stopgrad。采用现有`operator_squared_error`的rank-sized Gram精确公式及FP32真实梯度，
不展开巨大完整权重矩阵、不把A/B坐标MSE当BA、不增加native/BC/FM/对比或latent margin损失。
每个逻辑更新仍4task×selected/other两点，8点等权；identity半权解析为零且不重分配。
task/point流从原逻辑cursor144、visit16继续，seed及shuffle domains保持；不按参考成功率、点数或新结果重加权。
原query cursor112/task作为parent来源保存但本段不消费RGB/proprio/action/noise query，不编造8064式功能响应计数。

训练只需E/D、完整因子和保存identity，不加载source大policy、MT、图像前端、Reader或环境；原生policy仅在最终读出加载。
E读取完整参考只在共享训练的标签侧；D条件只有64维代码，继续禁止把完整参考旁路送进D或复制为输出。
部署仍是一套完整source/r8 LoRA，无teacher回退。E/D全部原trainable参数参与；两anchor保留导数，source权重冻结。

### 11.3 渐进投入、完整池读出与唯一终点

本段最多新增10,800个逻辑更新（每task最多1,200次访问、86,400个参数点训练配对）；
同时给一次累计有效训练wall约15分钟的窗口，先到者在完整36-task周期（9更新）边界停止，最多跨该边界一次。
首个检查在累计训练wall达到5分钟后的首个完整周期；若更新硬上限先到，则直接作为末点，不另补5分钟。
计时包含有用forward/backward/sync/step及OOM重试，加载、完整池读回、保存另计真实wall和GPUh；恢复必须承接累计计时，不能重置窗口。
没有累计GPUh硬限；15分钟是这项具体学习假说的投入安排，不是所有后继架构的永久期限。

起点144、首5分钟节点及唯一终点，最多各一次读取完整固定271点的加权L_operator；源baseline按同C_l计算。
全池读回不用视频、native policy或环境，不增加训练标签。记录selected/other、task/suite与层误差，恒等零项明示；
不是用不同query首末loss拼收敛曲线，也不新增多种latent/插值/扰动探针。
若5分钟节点的全池误差较144起点下降不足5%，结束本次学习；这只停止继续投入，不宣布架构不可能。
否则继续到15分钟/10,800更新的唯一终点。不得回选中间最低loss或按task选点；若首节点与末点重合只读一次。
无论早停还是正常终点，有限且数值有效时均做下述一次完整控制读出，防止再以BA代理直接裁掉真实控制正例。
若出现非finite或真实工程失败，保存失败/有效更新，按现有工程与科学边界处理，不把坏数值LoRA送入环境。

每9更新是完整task周期；按现有完整checkpoint原语保留一个滚动恢复点及上一原子替换点，
首检查与最终点额外留完整状态。滚动保存建议每90更新或实际约1分钟的完整周期，具体纯恢复频率由执行按吞吐定；
不累计数百份相同用途中间点。完整保存E/D/Adam/scheduler/RNG/拓扑/S/C_l/bank/task-point cursor/累计计时/停止状态。
新phase/global update/原144来源分别记录，不覆盖旧四个checkpoint或将目标迁移称bitwise exact。

逻辑8点内按实际吞吐比较更大的有用point pack及同节点1–6 ranks，至多覆盖全部8点；
可复用同一步live anchors及进程内只读因子cache，不缓存随参数变化的E代码或D hidden。
首批有用更新完成比较并在完整边界选物理安排；不因过去native的4rank/14查询最快而沿用，也不加载大policy占显存。
有余量则比较可容纳的更大pack，记录真实point/s、有效update/s、峰值和未继续扩大理由；不得改变8点逻辑均值或新增科学样本。
各rank先按全局/8累积再SUM，或采用数学等价DDP平均，不能重复除world；优化与读取都复用现有模块，不新增平行控制编译器。

### 11.4 一次真实控制读出、资源与交付

唯一末点生成同36个selected的`D(E(reference))`，每task一套完整固定LoRA，原init0..15、noise root202610151及官方执行不变。
已有source/reference/144逐case只读配对复用；五source identity的80逻辑行仍明示复用。
其余31套真实执行496次新episode；常规control上界171,520、dummy4,960、合计176,480步。
本独立窗口另有16次工程额外尝试，最多512物理提交、control179,840、dummy5,120、合计184,960步。
它不刷新reference或旧144窗口额度；失败/未知保守计账，有效行不重跑，不新采RGB/完整H或参考动作轨迹。

报告所有task/suite、target/support/identity、source/reference/144/新末点success sets、R/G/L/churn/重合与breadth；
重点是参考原新增288例保住多少、source原218例损失多少，而非仅比失败144多几例。
已有同八task MT128/67只作原配对子集参照，不新增MT请求、拼成36任务对照或冒称held超MT。
本段仍是标签侧参考参数输入、训练域与参考选点初态上的控制保存检查；0视频、held/Test或相邻同目标学习稳定资格。
广泛保住参考能力才接原视频→代码的实际学习；BA低但闭环差则据完整层/任务分布修订保留假说，
两者仍弱则停止此E/D直接学习窗口的继续加量，不保护现有压缩结构或直接加RL。科学接续由main自主承担，无Owner例行许可停点。

新ROOT=`/data1/user/ymdai/ember_runs/source_policy_space_compiler_20261015/control_operator_from144_seed7`；
预计新增峰值12GiB含完整恢复/原子副本、36输出、指标与有界cache，原reference/144/source/数据只读引用；不复制大资产。
launch前查strg01独立data1 quota、实耗及共享容量，live双节点GPU/进程/显存、总6或8卡/node6，CPU按节点分别准入。
训练实际轻量需求不得套旧source-runtime的40GiB门槛；读出按真实峰值及环境PSS安排动态long-first/persistent workers。
首批实施与消费预计20–45分钟，学习5–15分钟加有界读回/保存，最终闭环约15–30分钟；整批约45–90分钟。
这是沿144实测8.51分钟含模型加载/16.17分钟readout与本次不加载source训练的预估，首有用工作后据实修正；
§9/10原时钟、失败和费用不重置，新接收/准备/启动/退出独立登记。
执行独占代码/记录/Git，修复及正式train/eval均新clean pushed detached，历史native运行面留在Git/frozen原件。
只保留一个真正退出等待者，整批验收/Git/释放后一次交main归还写窗口，无阶段Queue、心跳或第二waiter。

## 12. 固定64维线性参数基底与一次控制保留读出

### 12.1 取舍与完整作用链

§11的1170仅224/576，对source218、reference502；参考新增288只保13。全271池BA改善2.963931%，
31个实际输出的BA norm/reference中位.054568、方向cosine中位.062441。四代表的代码/头前条件变化已明显增大，
不能继续用144的微小条件差解释全部失败；完整论证及main原件见[控制空间分析§10–11](../analyses/source_control_space_learning_20261015.md#10-1170的实际变化不再是原来的微小代码但仍未学到参考方向)。
停止共同E/D的同条件加量，改为直接解出source为原点的线性64维因子子空间；不改source、输出r8、参考bank或单位S。
目标是看同一代码维数在去掉共同神经编解码之后能否保存广泛参考功能；这不是单层架构的等目标因果对照。

唯一固定bank与S从原`control_space_seed7/{bank,units}.json`引用；仍为36task、266非零checkpoint加5个identity，271不同点。
source中心取原144完整checkpoint所保存的FactorLayout.center，使用其实际有效坐标/FP32物化；
不重新生成一个数值不同的identity再要求逐元素一致。B=0使z=0为source函数；旧正常精度差异按AGENTS接受。
原38 targets/76 tensors、643,584个有效标量/512 padding、共同原rank顺序均保持，不做符号/置换/gauge对齐。
S继续是每tensor一个固定RMS，不改动作/proprio normalization，不引入held统计。

令`r_j=valid((pack(Lambda_j)-center)/S)`。权重严格沿原pool_points：task等权1/36、selected半权、
其它完整点均分另半权；无其它点时selected全权。五个source点为零向量但保留质量，不能重分配给非零任务。
构造`M[j,:]=sqrt(w_j)*r_j`，不减bank均值、不用成功率重加权，不只取selected确保其恰好落在子空间。
一次薄SVD或数学等价小Gram谱分解取得前64个右奇异向量U，按奇异值降序；仅有效坐标参与。
线性代数实现与CPU/GPU安排由执行按实际内存/吞吐选，允许小Gram更高精度以稳定分解；最终U/输出为FP32。
若实际数值秩低于64，保留64维接口并以零列填充未识别方向、登记有效秩；不增加随机方向或另扫容差/维数。
一次求解只产生一个basis，不以闭环、BA或谱解释率选择不同basis/截断值。保存实际采用的分解方法与数值秩定义。

编码/解码唯一为`z=U^T r`、`Lambda(z)=center+S*pad(Uz)`，其中padding始终零，使用现有FactorLayout.unpack。
完整38层的A/B同时由同一个64维z决定；实际层更新是`s B(z)A(z)`，不是把多个expert算子或最终video LoRA平均。
编码只在训练标签侧读取参考参数；解码接口只收连续z，运行时不收task ID、参考路径、点编号或完整参考旁路。
不得为selected加残差缓存、nearest-neighbor、精确端点插值或每task回退。U只在这一次合法共享拟合后冻结。
本段是新结构的fresh闭式拟合，无Adam/scheduler/optimizer updates；旧144/1170参数和optimizer不迁移。

### 12.2 拟合产物、直接检查与停止范围

仅完整拟合一次，不新增参考、查询、视频、实践或动作轨迹。优先在CPU内存用原只读因子，
需要GPU时先按live资源准入；不加载大source policy帮助分解，不把未使用GPU占住。
保存一个canonical basis checkpoint：U、saved center/S/layout、原bank及精确权重、code_dim64、有效秩/谱、
拟合代码/环境/方法、起止wall与费用、schema和完整状态；无optimizer时明确标注，不伪造训练step或exact-resume状态。
原子落盘和恢复复用现有机制；中断只从未完成阶段接续，已完成basis/有效物化/环境行不重做。

一次读取全271池并保存原单位下加权因子残差，以及沿§11相同固定C_l的完整BA误差；按task/suite/layer、selected/other报告。
同时记录所保留谱质量和实际有效维数，均为定位证据，不是控制资格或新选点依据。
不要求因子小误差、谱高解释率或BA低于人为门槛才允许闭环；finite且结构有效的唯一basis都进入§12.3。
仅非finite、错误shape/单位/信息墙或真实分解失败触发工程处理，不把科学投影损失当bug修补。
本批0神经E/D更新、0q/Reader/共享RL、0新native训练query或source/MT训练。
没有自动改32/128/256维、换单位、拼局部基底、补selected残差或第二拟合块。

### 12.3 唯一投影读出与结果分支

每task selected只通过上述E_linear/D_linear物化一次，36套完整固定LoRA；init0..15/root202610151/official协议不变。
只读复用原source、选定reference、native144及operator1170逐case作比较；旧schema由其明确元数据读回，
不加载旧神经E/D为active fallback。五source identity80行明确复用真实source，其余31套真实执行496新episode。
闭环阶段只载source和已物化的一套LoRA，basis/参考bank/任何编译辅助不在最终policy回路中运行。
新phase与basis checkpoint路径用真实名字登记，不能把closed-form fit虚构成1171或一次optimizer更新。

正常496提交的control上界171,520、dummy4,960、合计176,480；另有本独立窗口16次工程尝试，
最多512提交、control179,840、dummy5,120、总184,960步。旧窗口余额不转入、不重置旧成本；未知保守计额，有效行不重跑。
不新采RGB、H、参考动作轨迹或其它读出面板。弱参考、失败和source损失均保留。
交付五方逐task/suite/target-support-identity成功集合、R/G/L/churn/breadth和相邻1170重合；
重点是原参考288新增保留及source218损失，不能只超过弱1170即宣布成功。
同八task的旧MT67/128只作已有配对子集参照，不新增MT或扩成held/全36结论。

若广泛参考控制被保住，main应尽快登记并启动真实原始教学→z→固定D的学习，不另开latent几何探针链。
若内部重构好而控制仍差，更新因子近似与必要控制方向的判断；若内部和控制均差，降低固定全局64维子空间优先级。
两种阴性都不自动扫维数/单位、追加神经补偿或交RL掩盖缺口；main据原件改完整假说，不能停下来等待Owner逐项许可。
一次输入已知参考权重的训练域控制保留仍不证明视频获取、held迁移或跨checkpoint稳定；不拼成完整方法资格。

### 12.4 代码生命周期、资源与真实接续

复用唯一`ember.control_space`运行面、FactorLayout、只读References/pool权重、operator Gram、source Runtime、
dynamic long-first persistent queue、预算/退出原语。替换当前active神经E/D/Adam训练路径，旧实现由Git/frozen/原件保存；
不并存另一个PCA writer/package或按失败自动fallback。共享FactorEncoder若仍有真实其它消费者则保留，不为本次退役误删公共实现。
结构检查由执行随实际改动完成；main不重复工程测试。实际CPU消费者覆盖合法权重/center/shape、线性编码解码、
完整物化、原参照配对和新phase预算/恢复，检查真实机制，不新增镜像测试框架或大量逐tensor/bit校验。

新ROOT=`/data1/user/ymdai/ember_runs/source_policy_space_compiler_20261015/control_linear64_seed7`。
预计新增磁盘峰4GiB：约158MiB的FP32有效basis、完整原子副本、约89MiB的36输出和有界报告/临时产物；
原因子bank/source/数据只读引用，不复制。拟合矩阵约0.65GiB FP32，双精度或工作副本计入CPU内存，落盘临时也计磁盘。
开新root/正式fit前核strg01 data1独立quota、个人实耗/共享容量；CPU按实际node的MemAvailable/PSS准入。
每次GPU launch同时live查双节点、总6/8及node6；闭环按已有真实显存/CPU峰选择环境slots，进一步有用放大仍以吞吐实测为据。
不把仅CPU拟合套成多卡训练，不因闲显存扩科学面板；没有累计GPUh硬限。

预计实施/消费者15–35分钟，拟合/物化/读回约2–10分钟，正式496闭环约15–30分钟，整批约35–75分钟。
这是基于1170实际15.315分钟读出、一次小样本高维分解规模的估计，首次有效工作后按实测修正；显著超期说明具体原因。
本批接收/准备/fit/环境启动/退出分别记时，原reference首次09:05:48.038及此前所有科学/工程成本独立保留。
执行独占工程/科研记录/Git，集成push后从新的clean detached正式fit/eval；一个实际退出等待者，
整批验收/费用/资源释放/Git后一次交main归还写窗口，无心跳、阶段Queue、自通知或第二waiter。

## 13. 固定控制基底上的真实视频初始编译

### 13.1 本批解决的缺口与完整方法关系

§12同576行从source218/神经144的216/operator1170的224到linear484，reference502；参考新增288保252，
source保持212/218，reference仍失39。main复算496真实首plan：new-reference MSE为7.59479e-5，
new-source为.0454645，reference-source为.0444128；不是只改善内部因子误差。
因此现在不再研究参数压缩小探针，直接学习合法视频如何给出有实际控制作用的代码。
完整科学消费与最近似历史约束见[控制空间分析§12–13](../analyses/source_control_space_learning_20261015.md#12-线性64正例把缺口移到真实教学获取)。

固定唯一`control_linear64_seed7/basis/state.pt`中的U、source center、S/layout，不再次fit、不更新D、不加残差支路。
从fresh共享Reader学习`y_theta(V)`，`z=a*y_theta(V)`，`Lambda=D(z)`；最终仍source加一套完整38-target/r8。
本批实际condition没有practice，故H为空、0 interaction；不是整个候选改成永久无H或把历史H阴性当作可删除的理由。
后继经历编码、连续提案探索、全部提案/停止score及独立query回报须消费本批后登记，不能把当前FM梯度称为反馈学习。
原稿四分量分布在本次只实现确定性的初始中心；当前每task只有一个真实producer代码，不能声称已学会多种操作分布。
不先添加尚无消费者的mixture/停止/价值头；后继随机提案形式须按真实学习需要定案，不机械恢复四分量菜单。

### 13.2 教学特征、Reader与代码单位

K=1，原始教学使用该task合法50条视频，双相机真实有序RGB；沿RawTeacherVideoStore的stride5加末帧，rotate180一次、model224。
原始HDF5教学实际为128×128，已有producer执行RGB为256×256，均按既有resize_with_pad到224；不重渲染原示范或声称输入分辨率相同。
每帧通过冻结source视觉塔/投影得到双相机512个2048维patch，再只用这512个真实image tokens跑冻结Gemma完整图像prefix，
全有效双向mask、positions0..511、post-norm输出`[T,512,2048]`。不输入L/action/noise/proprio或伪造action query；
此读取复用旧pure-image-prefix的已实现数学，不称它自动包含操作语义。source所有参数冻结，Reader不读语言token或融合L的教学hidden。
执行policy和合法FM query继续正常读exact L；真实H中间接语言信息的允许边界保持。

Reader仅复用既有ImageReader的真实视觉/有序算子，全部参数seed7 fresh：2048→256投影，加512个空间位置embedding；
每帧8个learned queries、4-head attention读取全部patch。8个slots加原始frame_indices的既有256维sinusoidal positions，
按时间/slot展开，经4层pre-LN Transformer（width256、heads4、FF1024、dropout0）与LayerNorm。
一个learned global query（N(0,1/16²)初始化）以4-head attention读完整有序memory，query加attention响应后
LayerNorm、Linear256→64输出y；无第二因子decoder。
最后Linear权重/偏置均零初始化，初始z=0/source；其余按上述模块原初始化。首步Reader上游零梯度由零head导致，
后续实际非零head应把信用传回Reader；不能把接通当有效学习，也不能为了非零外观加随机LoRA扰动。
有序attention允许读取动作先后，不强迫使用运动而非静态task线索；不以注意力差异证明理解。

代码单位只用原271点同权重统计：`a_k=sqrt(sum_j w_j*z_jk^2)`，等于§12保存的前64奇异值。
不减代码均值、不使用held统计、成功重权或单位扫描；若数值实现需除法保护，固定`max(a_k,1e-6)`并记录实际是否触发。
全部64已识别方向沿原顺序。标签`y_ref=E_linear(Lambda_producer)/a`只在loss侧；teacher输入没有代码、task ID、路径或producer身份。
固定D的导数回Reader；D本身U/S/center无梯度，输出A/B都变化，原生`B(z)A(z)`交叉项保持。

### 13.3 数据、损失与有界共同学习

任务为原显式train24+审计support12，36等权；没有额外meta task或held offline梯度。
每logical update按seed7任务permutation连续取4个task，每9次完整覆盖36；每task一次访问包含两种独立教学：

1. 原始50示范按各task的seed7 permutation/cursor均匀轮转，每条完整视频生成一个代码/LoRA。
   取该task另外episode的28个合法RGB/proprio/action query，当前teacher episode严格排除；全50 query池可用。
   复用`FunctionalQueryDataset(chunk50, action_start_offset1)`、原固定normalization、原生`flow_sample`和
   `NativeFlowPrediction/mean_velocity_loss`，每个query一个真实noise/time样本，完整50×7 FM；不改成native10蒸馏或首5步训练。
2. 已有selected report的545条真实成功producer视频，按task内部独立seed7 permutation/cursor均匀轮转。
   同task池大小2..18不改变task权重；包括五identity任务的90条source成功RGB。失败103条原件和成本保留、不作为正确教学。
   producer只由原冻结selected决定，不按本次拟合表现再挑。以该producer完整LoRA投影代码作为辅助标签，原始视频不假称由它产生。
   report RGB已按原control时钟stride5加terminal采样，rotate180一次，不再stride5；保留其真实原frame_indices。

唯一loss为四task平均的`L_FM(original cross-episode) + 0.1*mean_64((y_theta(real_producer_RGB)-y_ref)^2)`。
FM经同一固定D和冻结原生policy反传到全部Reader/head；辅助项只经Reader/head，标签detach。
物理query分片/梯度缓存/重计算须保持这条实际链，不能detach生成LoRA后只更新代码辅助或重复计入一份task辅助。
不把FM系数1/辅助系数.1当成梯度贡献已经主导的证据，记录两项真实loss和必要的实际信用检查即可。
无BA辅助、同episode教师动作、pseudo-action、由视频恢复轨迹后微调、新交互动作BC或其它未登记目标。
原始视频和producer视频分别forward，不能平均两视频、特征或最终LoRA。没有dynamic-K主张。

每步共112 FM query/4原始视频/4辅助视频，按全部task/query完整平均后clip/step一次。
原query episode/frame采样可复用`writer.learning_data._episode_queries`的确定性cursor并明确teacher排除；
noise/time seed按task/visit/真实query行派生，物理microbatch/rank不能改变逻辑行、task权重或辅助重复计数。
AdamW lr1e-4、betas(.9,.95)、eps1e-8、weight_decay1e-4、global clip1、constant LR，正常BF16/TF32允许。
唯一fresh seed7轨迹最多1800 updates；每task200访问，即原50视频四次完整覆盖，共7200原始+7200辅助读取、201600 FM query。
900和1800为预登记闭环节点，900已有每task100访问/原始池两次覆盖。不是每个架构都必须履行的固定课程或充分收敛证明；
如出现明确广泛科学失效，可携已完成真实结果提前回main裁决，不以等终点保护弱假说，也不按训练loss单独选点。
没有自动1800后延伸、rank/LR/seed/权重扫描或重启挑峰。

保存完整Reader/Adam/constant scheduler/scaler、task/两类video/query cursor、各rank RNG、logical flow/topology/schema与固定basis引用。
至少保900/1800和最近完整rolling恢复点；常规每90步在完整10个task周期边界保存rolling，原子替换保前一有效点，
不累计全部90步状态。中断/物理迁移承接完整点，登记丢失未提交更新与实际重复成本，不称跨拓扑bitwise exact。
原144/1170/basis/source/reference正式资产原地保留，不加载其旧神经/Adam或MT为初始化。

### 13.4 两节点真实闭环与选择边界

900/1800各做两套已登记面板，所有输出仅从原始教学生成；一次生成后完整LoRA冻结，Reader/特征模型/D退出闭环。

- train36×init0..15，共576。teacher由`video_schedule.condition_demo_index(seed7, suite, local_task, state, correct,50)`
  固定选16条不同原始视频/task，跨checkpoint复用。policy/env root202610151与原source/reference/linear64逐case配对。
  这是训练域、参考选点初态及部分训练视频诊断，不能称无放回整轮50或held；新Reader在五identity任务也可能非零，全部576真实执行。
- validation8×50，共strict single-checkpoint400；固定task `[3,6,11,16,23,26,31,39]`。
  原`image_only_gemma_with_history/readout_panels/validation.json`的400项teacher/init/noise/registered-scene仅作既有配对来源，
  main核对全部与seed7官方schedule一致、每task恰好50条各一次；沿MT153原场景/噪声root7，跨新checkpoint保持。
  只读合法教学RGB，0 held离线action/state/terminal/reward标签、0适应与共享梯度；不用Test或最终评估反馈改参数。

official native10/执行5/双相机/8Dstate/7Daction/dummy10/原suite horizon保持，support12 horizon400。
只读比较匹配MT153、旧G165与D175/168的原件，明确不同方法/学习量；不新增MT或重选旧高点。
当前两qualification arms仅为900/1800 correct400；初始编译候选按correct400较高者、平局较早冻结，不按seen/loss/union挑选。
相邻两点必须同时报告per-task/suite、R/G/L/churn/breadth及success-set交并；某一点高而另一点或大片任务丢失，
只能记局部高点，不凭候选标签声称稳定超过MT。当前initial candidate不是最终完整带H方法的已选checkpoint。
同task-other/wrong/no-video因果controls在候选冻结、main消费后另登记；本批不自动开全矩阵，shuffled/reversed仍只在最终阶段。
原始视频效果强也不能由共同producer代码监督宣称动态内容必需；尽早完整循环后补适量公平证据，不能永久推迟。

两节点合计1952个新episode；正常control上界671040、dummy19520、总690560。
本独立批另16次完整工程尝试：最多1968提交、control679360、dummy19680、总699040步；不转入旧余额。
科学失败不重试、有效行不重跑，未知执行保守占额。0新参考训练/producer采集/实践H/RL/额外评测臂。
若seen无法取得实质控制，优先区分合法视频读出/功能学习缺口；若seen强而held弱，降低跨任务映射假说支持，
不以再给每task更多同producer视频代替独立task或自动追加RL。结果回main后继续解决完整目标，不等Owner逐步许可。

### 13.5 运行面、存储、吞吐与可靠交付

canonical ROOT=`/data1/user/ymdai/ember_runs/source_policy_space_compiler_20261015/video_initial_seed7`。
复用`ember.control_space`的冻结D、source-only Runtime、既有FM/query/视频schedule/动态队列/账本/close与checkpoint原语；
继承旧ImageReader/pure_features的必要视觉算子时归入唯一当前owner，不恢复整个旧ImageCompiler/VL/MT/H训练框架。
§12一次fit已完成，active入口不自动重fit；只读basis/历史消费者保留，obsolete执行入口依真实调用退役至Git/frozen。
具体模块边界、代码结构/实际消费者验证、独占分支与集成由实验侧闭环，main不重复工程测试。

原始1800视频按实际RawTeacherVideoStore的stride5加terminal共59114帧，BF16图像prefix约115.457GiB；
545 producer共16759帧约32.732GiB，总2345视频75873帧约148.189GiB。最大原始105帧、producer94帧；
此数已计非整stride的1456个末帧，不能沿用仅ceil(n/5)的旧粗估。不得裁长视频、平均帧或偷偷缩池以适应物理batch。
冻结特征可按真实source/相机/预处理/schema复用已有兼容cache，按路径引用不复制；旧额外L字段绝不作为本Reader输入。
新缓存峰值连同原子临时、完整CP、两节点1952套物化LoRA约4.8GiB、日志/临时held特征预留为192GiB。
held特征按shard有界处理/释放，临时落盘上限8GiB计入192，不预建整个held prefix副本；原大source/数据/参考RGB不复制。
建root/cache/训练前必须live核strg01 data1独立quota、相关个人用量、共享容量及实现峰值，不能仅凭上批约380GiB余量启动。
所有新增在data1；确有超额按实耗调整缓存生命周期或携具体缺口回main，不切data0或改科学池。

每次GPU launch live双节点、总6/8/node6/显存和process；CPU按实际node的总PSS、线程、MemAvailable和环境槽准入。
按首批真实有用工作比较frame chunk、FM microbatch7/14/28及有实际余量时的多task物理pack，保持logical112/辅助4。
有余量继续验证有意义放大，报告samples/s、videos/s、完整updates/s与显存/CPU峰及未继续扩大的实际原因；不只复述低显存。
训练可选1–6物理rank，同一node、P2P禁用/NUMA/deferred NCCL；query可分片但辅助任务权重只计算一次，不被rank数改变。
ready特征、训练、900/1800物化与评测按依赖和实测吞吐并行，不能等待无关任务，也不跨节点拼训练碎片或用dummy占卡。
eval继续dynamic cost-balanced/long-first persistent workers，空闲runner完整close；按GPU与node CPU实际成本调slots。

预计实施/消费者45–90分钟、特征20–60分钟、训练1–2小时、两点物化/评测40–70分钟，可重叠部分工作，整批约2.5–4.5小时。
依据原expert28-query约2.1s/update、linear496读出11.10分钟与新增视频规模；并非本新Reader实测，首有用工作后修正。
没有累计GPUh硬限；所有加载/缓存/训练/读出/失败/重算成本逐项计入，保留旧首次09:05:48.038及全部旧费用区间。
正式共享训练、特征物化与eval使用新clean pushed detached版本；一个真实owner持续等待整批退出，
整批验收/资源释放/Git后一次交main并归还窗口，0阶段Queue/心跳/self通知/第二waiter。真实边界可提前回报，不能失联空停。


## 14. 固定控制空间内的完整参数提案与真实反馈学习

### 14.1 科学取舍、唯一范围与初始化

主讨论已消费§13完整原件、实际算子、1952已存代码及两点任务分布；完整解释见
[控制空间分析§14–15](../analyses/source_control_space_learning_20261015.md#14-视频初始映射的实际进步与迁移分叉)。
训练域334→413、参考新增保116→193与held117→111分开；Object42→23和Long18→8等损失不能被Spatial37→64掩盖。
不再追加1800后的同条件FM/共同producer回归，也不以“少了H”重述全部旧失败。
本批检验实际教学/经历到完整参数提案的学习，独立初态回报是否能改善完整编译；不是单个梯度缺项的因果隔离。

仍为原36个non-held任务、raw source、唯一完整38-target/rank8。U/S/center/code_units全部固定，
只读`control_linear64_seed7/basis/state.pt`及`video_initial_seed7/checkpoints/update_00000900/state.pt`。
Reader900是依旧correct400规则选定的初始化，不因1800 seen更高改选；只继承Reader权重和原code units，
旧900/1800 Adam及所有产物保留，本批新目标/新增H结构使用fresh optimizer，不称exact-resume旧训练。
source、图像塔/Gemma、D与单位全无trainable参数，0新参考、0新basis fit、0MT基底或第二执行adapter。
本批不是Task内动作BC：自身实际动作仅解释H，任何condition均不先准备动作轨迹再拟合最终LoRA。

### 14.2 实际H读取、参数分布与信用入口

保留现有ImageReader的完整纯图像有序memory `M_V[T*8,256]`及初始均值`y0_theta(V)[64]`，
初始化为900；训练中其现有所有参数可更新。教学仍无L、教师state/action/结果/ID或文件名输入，
原始stride5加terminal、双相机与rotate一次的全部真实帧保持；现有冻结特征按路径复用。
执行policy继续读exact L，自身实际hidden保留由正常policy产生的间接语言信息，不另作去信息。

每次实际replan形成一个H事件，复用`History/EpisodeRunner/NativeVelocity`已实现的数据消费者：

- pre/post真实双相机RGB，经冻结source视觉塔/投影得到各512×2048的raw image patch；
  pre可复用同次native prefix的embed_image产物，post缺失时对真实RGB补视觉forward，不跑假suffix。
  这是冻结视觉投影，区别于教学的纯图像Gemma post-norm特征；不将两者宣称为相同语义空间。
- 同次真实10-flow第0/9次forward的action_out_proj前hidden，形状2×50×1024。
  后45个token是实际生成计算中的计划hidden，不伪称已执行动作；实际动作与mask另有明确字段。
- pre/post proprio 2×8、真正执行的5×7 actions与5D mask、自身reward/success/terminal/truncation/failure五事实、
  真实episode/step以及当次实际行为代码y。失败和未被最终选用的候选经历全部进入H，不挑成功轨迹。
  proprio/actions沿冻结source单位归一化，两个计数除以8192；不重算task统计，不引入task ID、文件名或正确操作程序。
  当前不新增接触/特权几何输入；原生RGB/proprio/动作/hidden/结果的真实经历保持，不称已使用所有可用物理信号。

新增H读取器seed7 fresh，直接归入当前`control_space`：独立image Linear2048→256和hidden Linear1024→256，
各自输入LayerNorm；512空间、pre/post2、flow_call2、action_position50 embedding。
行为y经Linear64→256作一个参数身份token；实际facts63经Linear63→256加入learned event query及真实事件ordinal位置。
该query以4-head attention读取该事件完整图像/hidden/身份tokens，形成一个256维事件，再以4-head attention读取M_V。
全部相关事件进入两层causal pre-LN Transformer（256、4 heads、FF1024、dropout0），保留真实顺序，
不按教学归一化时间硬对齐、不平均frames或把全部H压成一个DTW数。

decision query为learned256向量加Linear66→256：输入已完整实践的best_y64、best_success_fraction及remaining_steps/8192。
其4-head attention只读取本决策之前的H编码；加query残差、LayerNorm、Linear256→64得到delta_y。
该最后Linear权重/偏置零初始化，其余用既有模块默认初始化与seed7；因此初始提案均值为已选Reader900。
H尚为空时delta_y定义为0，不制造dummy observation。首次H body零梯度可能来自零head；后续是否实际获得信用须真实消费者检查。
不再增加旧FactorEncoder、语言分支、独立selector、混合分量、value网络或尚无消费者的停止head。

当前唯一随机提案为

    mu_theta(V,H) = y0_theta(V) + delta_y_theta(M_V,H,best_y,best_score,remaining),
    y_j = mu_theta(V,H_j) + 0.25 epsilon_j,   epsilon_j ~ N(0,I_64),
    z_j = code_units * y_j,                  Lambda_j = D(z_j).

sigma=.25是固定bank单位中的一次实现选择，不学习方差、不截断/拒绝正常有限样本、不扫描尺度。
真实sample/均值/单位、条件Gaussian联合log-density及RNG保存FP32；D输出原完整A/B，policy正常BF16/TF32执行。
对64坐标的log-density求和，不能改成每坐标平均；同样不能在训练时悄悄使用均值替代实际样本。
最终仅部署一套完整LoRA；共享q/D/视频/H/选择过程全部退出最终episode。

### 14.3 每个condition的真实编译规则

每个condition独立持有H、候选、RNG和**8192次环境steps，含全部settling/reset**。
初态和最终/query面板在开始前固定并分开。4个practice初态是经验比较面板，不是实践/重读轮数。

1. 先在该4初态实际执行source恒等LoRA；不借MT或旧成功标签替代这4次自身实践。H记录正常原生执行。
2. source四例全成功则按当前规则结束；否则，仅在余量足够一个完整面板`4*(horizon+10)`时，
   读取合法教学/全部已有H，实际抽一个y并完成同4初态/同policy-env RNG面板。
3. best按整个实践面板成功数取最高、平局严格取较早候选（source最早）。失败候选的全部事实仍可影响下次提案。
   某候选四例全成功，或余量不够下一完整面板时停止，冻结best的完整LoRA；不比较不完整面板成功率。
4. 每个y在其episode内固定，policy每5 actions正常重读自身观测/做10-flow。跨候选的env/policy随机数配对，
   proposal RNG另由seed7/task/condition/提案ordinal确定，与物理rank、调度、恢复无关。

这是本次固定的成功/资源结束规则，没有可学习停止随机变量，因此PG中不虚构停止log概率。
它不是Owner规定的永久轮数或“成功一次就够”；四初态选择仍可能过拟合，正由独立query和最终新初态判读。
允许source分支无需读取教学就结束，实际零读取与零非source提案如实报告，不能将这类成功归功于视频。
缓存源图像特征、M_V或行为期事件embedding只节省计算；按实际proposal/教学访问报告重读、frame-equivalent与原生编码成本，
不能把缓存重读算零。训练FM/score重放计算与部署编译读取分别计账。

所有正常有限科学失败保留、不重采。数值/环境工程失败按原实际/未知成本占额，已有效行不重跑；
恢复未完成condition须保留实际sample、已完成面板、完整H和行为模型身份，不能从相同seed偷偷重演已消耗实践。

### 14.4 有界non-held共享学习

完整候选计划144个condition=36task×4个不同原始教学、logical4task/更新、36次共享更新；
**当前只授权其第一完整周期：36个condition、每task一次、9次更新。** 后三周期和held须main消费首轮后登记。
每9更新等权覆盖36task；task顺序复用seed7的既有permutation流，不因物理rank/完成速度改权重。
每task第v个condition（完整计划v=0..3，当前仅v=0）的教学，固定为§13 seen面板init=v对应的原始teacher_demo。
这是明确有限面板，当前每task一条、拟议全窗四条；不声称本阶段覆盖全部50或增加独立task，现成原始特征只读复用。
其4个训练query初态为`[v,v+4,v+8,v+12]`，noise root202610151，正常原生执行；
practice固定从init16..49用SeedSequence([7,202610161,task,v])无放回取4，env/policy noise root202610161，同候选复用。
query四例永不进入本condition H/选择；最终R为这四例success平均，不用实践分数替代。
当前每task四个query case（拟议全窗16个）与旧source/reference同state/RNG可比较；只有anchor init=v与旧900同视频，
当前36条anchor现成配对单独报告，其余不冒称与900逐教学严格配对。首轮36链由不同训练时点产生，不合并成single-checkpoint资格。

每个logical batch先固定完整theta，完成4条全部编译与4×4独立query，然后只做一次共享更新。
所有样本的真实behavior logp/模型版本保存；不跨theta混用未修正的旧轨迹、不多轮PPO/replay、不用未来query结果重写先前H。
source转移、固定D/原生ODE和固定选择规则在给定实际y之后都不含theta，故整个编译路径的score为

    L_PG = -mean_i [(R_i - b_task(i)) * sum_j log q_theta(stopgrad(y_ij) | V_i,stopgrad(H_ij))].

每个实际提案都计入，包括未选中者；按4个condition平均，不除提案数、episode长短或物理world size。
全source早停没有参数样本，其PG为0而非伪造梯度。b仅作训练loss侧baseline，初始化为旧900该non-held task的16行成功率，
当前batch使用更新前b并detach；step完成后按`b <- .9*b+.1*R`更新。本字段不进入q、D、H或held部署，不是task-ID路由。
不从当前自身return拟合baseline后反过来给同一batch算advantage。

H事件编码只依赖该事件、当前合法完整V与既往行为y。因果history mask及每decision的prefix长度是完整信用合同：
可一次编码完整实际H并按决策mask读取，以复用计算，但任何较早logp都不能看后续事件/提案/query。
行为期与score重放使用同一theta、dropout0和真实y；只做正常数值容差的实际logp/梯度消费者核验，不追逐bitwise一致。
H原始环境证据stopgrad不等于丢掉分布信用；所有参数抽样score承担其改变后续H的效应。

同时，每task以该condition原始教学的H空均值产生LoRA，保留28条其它episode的完整50×7原生FM。
使用§13 FunctionalQueryDataset/action_offset1/source normalization/flow_sample/NativeFlowPrediction和teacher排除；
query cursor从Reader900登记的每task100访问/2800 query位置接续，当前一次/拟议全窗四次，teacher是本节明确面板，
noise/time按实际逻辑行派生。每步112 FM，当前总1008/拟议全窗4032；这不是自身动作BC。取消producer代码辅助，不强迫不同失败H都返回共同E。

联合loss为`L_FM + beta*L_PG`。在首个两项真实全logical梯度norm均非零的训练batch，
一次固定`beta=min(1, .25*||g_FM||/||g_PG||)`，之后不重标或扫权重。
零PG批只做FM并继续等待首真实PG；若FM为零而PG非零，固定beta=1并明示该分支；两者均零则无虚构更新。
整批无真实PG时明确记反馈共享学习未发生，不能凭FM涨分认证本方法。
全部梯度按逻辑4完整归一化、跨rank SUM后再global clip1/step；FM和PG必须真正更新均值/视频/H消费者。
新AdamW lr1e-4、betas(.9,.95)、eps1e-8、wd1e-4、constant scheduler，所有新H参数seed7 fresh。

CP命名使用独立meta_update，绝不伪称旧936或完整承接900 optimizer。
每个logical更新保存完整theta/Adam/scheduler/scaler/全部baseline/beta/任务与query cursor/rank RNG/topology/阶段与实际路径账本；
rolling原子替换保前一有效点，当前保完整9终点；拟议18/27/36没有实际执行或产物，不按loss/seen挑选。
9为当前唯一终点，完整36task原件一并交main；没有自动18/36/72更新、尺度/LR/seed扫描或新task扩池。
若已有明确工程边界或广泛科学失效，带具体原件提前交main，不以固定步数强行履行；main自主裁决，不等Owner新许可。

### 14.5 后继完整strict400的计划与当前解释边界

本节明确完整方法如何判断，**当前不执行held**，不以首轮小面板选模型或认定正式资格。
拟议共享meta36冻结后，使用§13同400个task/video/final-init条件；每task全部50条教学恰各一次、全程不换video。
每个condition重新实际编译，同8192步/同4初态面板/同结束规则，所有共享theta/D/source固定。
practice从本task另外49个init中，按SeedSequence([7,202610162,task,final_init])无放回取4；明确排除自己的final_init。
practice使用noise root202610161及普通official reset/settling；可复用受控scene资产的初始化机制，不能复用别的condition经历或标签。
不同condition即使共用某个init编号也独立实践，不共享H、已选代码或任务内参数状态。

编译冻结后仅用唯一best LoRA跑该condition原来的final row：原MT153 registered scene、env/policy root7、
10-flow/执行5/render256-model224/双相机rotate180/state8-action7/dummy10/原suite horizons全部保持。
400个final初态逐condition从未进入其适应；Writer/D/q/选择退出final，final反馈不回H、不产生共享或本地更新。
这是有真实交互适应的方法，逐condition报告实践/重置/读取/候选数量/成本，绝不记zero-interaction。
0 held离线action/state/reward/terminal标签，0Test，0held共享梯度。

后继若按该计划登记，唯一固定终点correct400不由训练域144链选模型；与旧900117/1800111、MT153、G165、D175/168逐case比较。
全部per-task/suite/breadth/R/G/L/churn/成功集合重合与source/初始视频分支占比保留，能力交换不能由总分掩盖。
旧900→新36属于方法变化，不能称同目标相邻稳定性；新方法相邻学习与视频/H因果作用仍待结果后按具体缺口登记。
当前不自动新增held、same-task-other/wrong/no-video/shuffled/reversed或CP0完整搜索对照；
因此即使提高，也仅能先归因整套已登记编译过程，不能唯一归因H学习、视频动态内容或优于同预算generic search。
若明显不足，结束本首次窗口的自动加量，结合真实proposal可用性/实践到query差异/任务分布作下一科学判断。
若有实质完整能力，尽早补有限公平视频证据与相邻稳定性，不把所有controls变成开工前提或永远推迟。

### 14.6 资源、生命周期与实际接续

新ROOT=`/data1/user/ymdai/ember_runs/source_policy_space_compiler_20261015/feedback_proposals_seed7`。
**当前首轮资源合同**：36条practice总上界294,912步；144独立query为50,880 control+1,440 dummy=52,320。
正常合计最多347,232环境steps，每condition8192独立封顶，不挪其它condition余额。
完整4-case panel且未全成功时最少消耗`horizon+43`步，开始下个panel预留`4*(horizon+10)`；
各220/280/300/520/400 horizon的面板数数学上界为28/22/21/11/15，不据此强制执行次数。
36task practice最多2,688 episode、query144，正常最多2,832提交；
本独立首轮另16次完整工程尝试，最多2,848提交、control上界352,672、dummy28,480、联合总355,712步。
这些独立上界不能同时相加替代联合总步数；所有有效行不重跑、未知保守占额、旧余额不转入。

拟议完整144链加held400的4,801,728正常步只是后继规模测算，**不计入当前授权额度**。
其旧rollout吞吐外推约44.3GPUh且尚未计H读取/score，不能未经首轮真实证据就承诺跑完。
旧参考首次09:05:48.038及所有旧费用/失败身份保留，首轮开始/退出时钟另记，不从main讨论结束重置。

复用唯一control_space/source Runtime、完整D、图像/查询/History/原生runner、动态queue/账本和checkpoint原语。
旧§13 active1800调度和双checkpoint入口按当前owner替换，旧初始化Reader加载和旧raw比较保留；
不恢复整套旧proposal_writer/MT/Meta或另建平行Writer。具体模块组织、结构gate、实际消费者与工程恢复由执行者闭环。
正式计算使用新clean pushed detached，旧冻结不热改；工程修复若不改变科学合同自行验证/推送/换冻结继续。
实际源码/数据/梯度/选择/信息墙变化则给出具体边界，不借“接口修复”自行改科学变量。

当前新增磁盘峰估48GiB；不是拟议整个144链/held的配额许可。
旧2345视频/148GiB级冻结特征、source、数据、参考/basis均只读引用，0大资产复制。
完整当前36条H（去除可再生dense image feature缓存）、所有实际样本代码/概率/行为身份、
actions/反馈/原始episode rows、4个独立query及最终LoRA保留；代码与固定D足以标明全部已试参数，不复制每次读H所带的全部LoRA。
main只读三个旧实际H归档测得gzip存储/raw-RGB+native-hidden约.57–.63；
当前practice最坏约33GiB原张量、压缩约19–21GiB仅作估计，须以首有用新H修正，不能称已验证整批峰值。
多worker dense H/prefix临时盘总上限16GiB计入48；CPU缓存按各node联合PSS/线程/可用RAM准入，不能按每GPU重复许诺同一内存。
若实际压缩/增长超出可用配额，完整边界报告并处理本批真实临时生命周期，不静默裁H/视频、丢标签或切data0。
建root/cache/训练前live核strg01 data1独立quota、相关个人实际用量、共享容量与实现峰值；旧约236GiB剩余不作当前准入证据。
后续144链/held如获登记，须重新核其新增峰值与完整H保留/临时生命周期，当前不提前删除旧缓存给未来任务腾空间。

每次launch live两节点，总6/8卡、单节点6、真实显存/process/CPU准入；不因原world或logical4把整个可并行批锁为4卡。
训练logical4含16个可并行practice/query episode，按实际有用吞吐选择1/2/4/6 worker和环境槽/事件chunk/视频chunk，
不为凑卡启动空模型；同theta的独立路径/episode、特征、score/FM按依赖可重叠，后续full400若登记再按其真实就绪吞吐安排，当前不启动。
多GPU共享梯度在同一节点、NCCL_P2P_DISABLE=1/NUMA/deferred NCCL；独立evaluator不跨节点拼DDP。
H物理event pack与causal-prefix复用可优化，但须保持完整事件/真实决策前缀/全部score与4task平均，不能变成未来H泄漏或重复辅助权重。
按首批真实有用工作测实际episodes/plans/s、完整meta update wall、最长H memory/PSS及更大batch的收益；
正常BF16/TF32差异接受，无hash/逐tensor一致性工程或为低位一致降低batch。

首轮暂估2–4小时：实现/实际消费者约1–2h，36条完整编译/9更新/144 query约1–2h。
依据上批实际readout4.6699GPUh/506,278步，当前正常347,232步按旧纯rollout折算约3.20GPUh；
新H读取/score、logical4并发、source早停比例尚未实测，不把该比例当本次性能保证。
先在首批实际有用工作修正ETA，完整36task一次交main，不能等owner催促或将有限首轮变成默认无限续训。
若实际准备或运行明显超期，明确来源；不按固定间隔反复读日志/cache，不因无GPUh硬限无限扩大。
一个owner持续等待整批退出，0阶段Queue/心跳/第二waiter；实施/运行/资源释放/Git完成后一次交主讨论并归还canonical窗口。
main结合真实proposal/独立query/信用/成本决定接续；当前没有自动后3周期、held400或controls。
这不是Owner许可门槛，main继续自主解决完整目标，不要求例行工程ACK。

## 15. 首轮已有控制器的同教学独立query比较

### 15.1 本批要改变的判断与冻结范围

§14首轮已结束，工程/Git写窗口交回主讨论；完整分析见[分析§16–17](../analyses/source_control_space_learning_20261015.md#16-首轮反馈的真实收益主要发生在哪里)。
79/144对source58不隔离新增反馈的贡献，旧900同视频36锚22→19尚未覆盖其余108行；
后97提案/101,102实践步只在7task增加10个实践成功。先判断这些后续工作是否形成独立初态收益，
不自动把拟议144链/36meta/strict400变为正在执行的任务。

本批0训练、0新proposal、0新practice/H、0新教学视频读取或Reader forward；仅评测已实际保存/抽到的完整控制器。
raw source、完整38-target/rank8、原basis/center/S/code_units和official执行合同保持；0MT底座、0第二adapter。
§14 meta9及完整Adam/RNG/stream不变，不选择或继续18/36。所有原训练路径、失败、CP和费用保留。

### 15.2 唯一144-case面板与五臂

36原non-held task，每task query初态[0,4,8,12]、env/policy noise root202610151，沿§14已登记scene与配对。
每task只使用该§14 v0教学（即§13 seen init0的teacher_demo），不以query结果换教学。

- **source/reference**：只读既有原selection对应初态的实际完整row，不新增运行。reference只是能力参照，不进入编译输入或选择。
- **Reader900**：只读`video_initial_seed7/generated/update_00000900/seen/task_XXXX/state_000.safetensors`，
  核其已登记teacher_demo与§14 v0相同；不重新生成/拟合。原同LoRA的init0行36个直接复用，
  新跑每task init4/8/12，共108行。其余旧seen行使用不同teacher，不能充当本臂。
- **first-best**：对每个完整§14路径，只在panel0（source）及panel1（若存在）之间，按原successes最大、平局取早选择；
  source早停没有panel1时保持source。只读已有实践，不增加或重演路径，query不参与这个决定。
  29个task与full best相同，复用该批116个query；task1/5/19选择source，复用canonical source12行；
  task4/12/14/43选择实际第一提案，与full不同，新增4×4=16个query。
  从path中真实首提案y和同一冻结D/units物化这4套完整LoRA；不用mu、重抽y、平均候选或选参考代码代替。
- **full**：§14现成144个query及最终唯一selected LoRA，只读复用，不重跑。

合计逻辑五臂各144行，新物理episode恰124。不同臂合法的实际LoRA和逻辑身份分别登记；
同task/state/scene/L、原native noise、10-flow/执行前5、render256/model224、双相机rotate一次、8D state/7D action、
dummy10、success即止与各suite horizon全部沿现有消费者。最终episode不看教学、H或做适应。
first-best源复用明确列出来源；同source在不同物理执行的正常数值/控制差异不得算作参数学习，
已知128行按上述固定口径为68成功，不将19个与full相同source条件另换旧行以选取较高结果。
不为消除低位差异固定batch1或重跑已有效行。

### 15.3 读出、停止与解释

唯一终点保存全部raw rows、144五方配对、task/suite/target24/support12/五reference identity分组、
R/G/L/churn/breadth/成功集合重合及reference新增保持。另单列7task/28case的first-best→full，
其中新16行与source复用12行分开；按source早停、读过教学仍选source、首次及后续非source最佳分别解释。
各臂实践/读取/候选成本采用既有真实原件：Reader900无实践；first-best为真实source面板加至多首提案面板；
full包含全部170,804实践步。反事实只截断真实路径前缀，不能把未实际运行的节省声称为本批已节省费用。
新124读出费用另计；旧§14 9.325162565GPUh/776提交/207,328步与更早全部费用不重复计入或抹去。

这是post-hoc non-held方法诊断，full/first-best源自不同训练theta，不是meta9 single-checkpoint资格。
Reader900来自已冻结旧correct400选择；新结果不得改变这个选择，不能回头修改本批的臂定义/参数/停止规则以挑选高点。
first-best对full仅检验后续实际探索与择优的边际控制作用，不能唯一归因H内容或训练本身。
0held/Test、0wrong/no-video/same-task-other/shuffled/reversed、0CP0同预算重跑或额外面板。

若full不高于Reader900或first-best，主讨论据原件重审当前探索/选择及稀疏信用组织，
不把“尚未充分训练”当原样扩量的默认理由；这不宣判全部反馈学习、rank8或64维不可行。
若取得实际跨任务增量，主讨论据保持/覆盖/成本决定有界续学和正式验证；有限正例亦不拼成完整方法成功。
124有效新行全部结束并分析后停止新计算，一次交主讨论自主裁决，无Owner例行确认门槛或自动后继。

### 15.4 资源与交付

新ROOT=`/data1/user/ymdai/ember_runs/source_policy_space_compiler_20261015/feedback_readout_seed7`。
108 Reader900新增最大38,160 control+1,080 dummy；16 first-best新增最大4,720 control+160 dummy；
正常合计124次/42,880 control+1,240 dummy=44,120环境步。
仅本批独立16次工程额外尝试，按最大520+10，合计上限140次/51,200 control+1,400 dummy=52,600步；
正常科学失败照常计入，不重试；未知保守占额，已有效行不重跑，不挪旧未用额度或自动充值。

新增磁盘峰2GiB：4套完整LoRA实测每套约2.47MiB，原108所用LoRA/全部source、assets、feature/basis均按路径只读；
余量包含代码冻结树、原始row/分析/日志和原子写临时量。本批不保存新H或视频，不复制大资产。
建root/物化/launch前仍须live查询strg01 data1独立quota、相关目录实际用量、共享容量及峰值，不能拿旧退出余量当准入。
每次launch检查gpu01/gpu02，合计6/8卡、单节点6，实际显存/PSS/线程与资源余量准入，不挤出foreign。
沿既有cost-balanced动态队列、long-first、persistent worker及完整runner.close，选择能实际提高吞吐的worker/physical batch；
复用既有16/8/4槽实测，在有明确余量和未被既有测量覆盖时按本124有用工作验证更大物理安排。
不为填显存扩大本面板或重复已有效row。

预计派发至退出45–90分钟，包含约25–60分钟工程接续/实际消费者与约10–25分钟读出/验收；
依据既有纯控制读出约0.997GPUh/99,045步，本正常44,120步约0.44GPUh的粗略执行外推，另计模型加载/调度，
不是速度保证。首有用工作修正，保留实际接收/准备/启动/退出和旧时钟。
沿当前唯一control_space运行面接入本次有限读出，复用reference/runtime/evaluator/账本/analysis；
不要复制整套Writer或为仅此一次读出建立通用新框架，旧冻结/科学结果不原地改。
正式来自新clean pushed detached；源码消费者检查、结构guard、运行恢复、Git集成/推送/开发树清理由现有执行者闭环。
一个整批owner及持续退出等待，不再给同一等待者Queue自通知；结束核原始结果、退出/资源/成本和Git后一次交main01a13dd9并归还写窗口。

### 15.5 attempt001渲染故障的有限恢复修订

main消费boundary_attempt001原件后明确授权：仅完成原124有效新query目标，已有80有效行只adopt，原12未启动首次执行；对boundary_attempt001.json逐名登记的32 reserved_unknown case各一次恢复。原科学五臂/参数/teacher/scene/RNG/选择/信息墙及2GiB峰不变，0新训练/实践/Reader forward/held/Test。

原32未知全部保留原status和14960control+320dummy=15280上界。event合计1955环境步只是lower bound，单一失败operation0不能代替整pack费用；不称已证实32失败、不记0、不用新恢复回填旧费用。新恢复attempt单独保存实际/未知成本；这32各限一次，不因通用额度再演，若同层失败再现携原件报告。

原16通用工程额度完整保留但不滚动重置；新累计172 submissions/66160control/1720dummy/67880total。算式为原124normal(42880+1240)+本次32旧未知(14960+320)+原16工程(8320+160)。除这32外其它额外工程尝试仍占原16；原valid不可重跑。原run_contract/manifest及attempt001身份保留，新增run_contract_recovery001显式指回原合同；真实reserve事务分别计算已用32恢复与原16普通额度，未用恢复额度不得借给其它重试。

已定位32路失败于MuJoCo MjrContext offscreen framebuffer创建0x8cdd，不是已证实CUDA OOM；24有效完成9.38794plan/s，保留完整环境/数值合同，本批active renderer最高24，不重跑32探针、不升级为永久上限。余44按实际就绪量减少worker；新clean pushed detached及live双节点/CPU/PSS/strg01准入后恢复，不热改e81fa5d4。

05:42:29.601265Z→05:46:33.197387Z的首owner故障、6job(5exit0/1exit1)全部费用0.238637107611GPUh保留；原派发05:27:06.038Z及参考09:05:48.038不重置。GPU/环境/owner已全部退出证据随boundary保留。沿原45–90min整批预计以实测修正，无GPUh硬限；一个owner持续退出等待，完整验收/Git后一次回main并归还窗口，无新增Queue/自通知/第二waiter。

## 16. 保留视频初始控制的任务内参数改进

### 16.1 决策依据、资产与唯一范围

§15已完成并交回canonical写窗口；本节是main消费全部原件后的新有限授权，不是恢复§14未实施的后三周期。
依据见[分析§18–19](../analyses/source_control_space_learning_20261015.md#18-同教学完整比较排除了什么又没有排除什么)：
Reader90089/144高于first73/full79，后续探索有净6但未补回初始能力；旧流程没有实践确定性Reader输出，
同链均值对经历变化小，初始均值又同时受共享更新与首次H修正影响。9次更新不等于充分学习，但不再默认按原组织加量。

当前仅原36个non-held任务、每task原v0一条教学、两个固定臂，各一条完整任务内编译和四个独立query，共72condition/288逻辑query。
共享Reader固定为按原strict400选择的900，完整38-target/rank8、raw source、原U/S/center/code_units全部冻结。
`video_initial_seed7/checkpoints/update_00000900/state.pt`、`control_linear64_seed7/basis/state.pt`及source只读，
不导入§14的meta9/Adam/H权重，不用旧full最佳样本或reference参数作为新候选，也不引入MT底座/回退/第二adapter。
0共享梯度/optimizer更新、0新expert或basis拟合、0teacher动作/state/结果、0held/Test/自动后继。
这是有交互的任务内参数学习；不得因共享模型冻结就记为zero-interaction，也不伪造新训练checkpoint。

### 16.2 两臂及实际参数更新

记原单位化代码为`u`，实际代码`z=code_units*u`，完整LoRA为原`D(z)`。

- **video**：初始`m`为Reader900对本task原v0合法教学的确定性输出。当前这36个精确视频输出已经存在，
  复用`video_initial_seed7/generated/update_00000900/seen/task_XXXX/state_000.{json,safetensors}`。
  JSON的`code`为z，按原900的正有限`code_units`恢复u；初始实际执行优先直接用该完整LoRA，避免重复生成已有产物。
  这是缓存既有真实视频编译，不是读取reference标签或task→expert字典；检查原checkpoint/basis/teacher_demo来源即可，0新hash。
  本批0新Reader forward/原生teacher编码；每condition有一条此前实际编译的合法教学，报告这项复用，不能把方法教学成本称为零。
  后续未缓存视频的正常方法仍须真实Reader forward，本节不提前执行。
- **no-video**：初始`m=0`、完整source identity。使用完全相同D、单位、原生执行policy/exact L及下述更新，
  不读取视频/code缓存或其它task参数。它有相同控制基底和正常语言策略，不是充分训练的language-only Compiler。

两臂都先完成初始m的整个实践面板，记录成功整数c∈{0,1,2,3,4}。随后每次：

    epsilon_j = Normal(0,I_64), fixed registered seed;
    u_j = m + 0.25 * epsilon_j;
    run all four practice cases with the single complete D(code_units*u_j);
    if candidate_successes > incumbent_successes: m = u_j;
    otherwise retain m, including every tie.

每次提案围绕当时实际保留的m；不能仍围绕最初u0、按某个旧task参考、或者平均若干成功样本。
原sigma=.25不变，不新增温度/方差学习、CMA协方差、学习率、局部梯度、尺度或seed扫描。
更新只读取本condition真实参数及整个四例成功面板；当前不使用learned H网络/价值头，也不宣称已理解H的视觉内容。
自身RGB/proprio/实际动作、native hidden和成功/失败事实仍按原History完整保存，不因为可能含语言信息而删减。
它们不是拟合动作轨迹的数据源；不由保存H暗加BC、离线动作query、FM、PG、producer回归或新共享训练。

每个candidate在整个episode固定，原生policy仍每执行前5 actions后重读自己的观测，正常做10-flow。
某完整面板四例全成功，或剩余预算不足下一完整面板的`4*(horizon+10)`时结束；最终冻结m的一套完整LoRA。
不固定修订/重读轮数，不平均LoRA，不选择未完整实践的样本；所有正常有限失败保留，不重抽直到成功。
经验面板分数不降只是本算法的经验性质，绝不作为独立query能力保持的保证。

### 16.3 配对、原件复用与信息墙

task集合和每task teacher_demo完全沿§14 v0。实践初态沿原
`SeedSequence([7,202610161,task,0])`从16..49无放回取4；env/policy noise root202610161。
query仍为[0,4,8,12]、root202610151，与§15五臂严格配对，永不进入H/选择/提案或停止。
两臂proposal ordinal j从0开始，复用原`SeedSequence([7,task,0,j,202610161])`导出的torch FP32 Normal64。
同task两臂有相同epsilon_j；臂、GPU、worker、就绪顺序和恢复不进入种子。停止不同可导致实际候选数不同，不能补跑一臂凑相同次数。

以下复用在结果前固定，不按新query结果选择来源：

1. no-video的初始source实践，复用§14 `conditions/task_XXXX_v0/path.json`的panel0及其全部144条原始有效rows/H。
   参数、四init、环境/policy RNG、官方执行合同逐项沿原登记；共52成功、40,377环境步（38,937 control+1,440 dummy）。
   这些步照常占各condition的8192额度。物理费用属于§14、不重计，也不能把复用释放出的预算挪给额外探索。
   原11个source四例全成功condition按同一停止规则直接结束，不重跑source获得另一个成绩。
2. video的初始确定性控制在这些practice初态没有已登记有效面板，须真实新增144 practice episode，正常失败也完整记录。
3. 最终m仍是video初始完整LoRA时，引用§15固定Reader900的四条query；no-video最终仍为0时，引用§15 canonical source四条query。
   这是预先固定的相同控制器复用。凡实际接受了新非零代码，须用其最终完整LoRA真实执行四条新query。
   不把相同source改换为较高的旧full行，不按单行结果混合来源，不复跑已有有效query。
4. 原§14/15 full、first-best、reference仅供完成后的报告。构建新适应输入只抽取允许的初始视频代码、source panel0、初态/RNG等字段；
   不把原path的query_rows/R、best_y或reference选择信息交给新参数更新。

每个arm/task有独立condition ID、m、成功面板、proposal游标、pending状态及8192账本；两臂不得共享新的H、成功代码或预算。
正常面板各有四条完整回执后才改变m。恢复保留实际样本/旧m/完整与未完成行、RNG游标及全部原始费用，有效行只adopt。
未知步骤保守占额；工程尝试仍占condition预算，不能用全局16额外额度覆盖每condition硬上限。
若无法在现有完整预算内恢复，带具体状态/原件交main，不静默裁面板、重置condition或把工程中断算科学失败。

render256/model224、双相机rotate一次、state8/action7、10-flow、执行5、dummy10、成功即止及suite horizon保持。
所有36为non-held，shared source已见support12身份与五reference identity分别报告；0validation/test离线标签或新任务程序。

### 16.4 完整读出与可失败判断

完成后形成video和no-video各144逻辑query，与固定Reader90089/source58/reference127/first73/full79共七臂读出；
复用与新增逐行标明。两新臂共享模型恒定，旧full仍是theta0…8历史算法，不能称同checkpoint H因果消融。
报告每task/suite、target24/support12/identity、breadth、R/G/L、churn、success-set重合及参考新增70的保留；
尤其保留Object12/14、Long及所有Reader能力损失，不只报实践最高分或成功condition子集。
按实际初始/最终代码、接受次数、候选ordinal、每次完整成功集合、stop reason分组；参数距离仅定位，不替代闭环。
原始经历保存但当前未被语义读取必须明示，不将搜索增益称作H网络学会纠错。

每臂分别报告逻辑实践成本、已复用旧成本、新物理steps/episode/GPUh、教学缓存来源/读取、query成本及工程失败/unknown。
相同8192上限不等于相同实际成本。source复用只节约重复计算，不提供额外适应信息或免费剩余额度。
本批一条教学/task是明确有限non-held面板，不声称50条视频无放回/held迁移或正式checkpoint资格。

若video完整query未超过其89起点，或改进只存在实践，当前固定单位局部搜索的自动加量/调参停止；
若和同预算上限no-video相近，不能宣称教学编译带来必要增量；若收益仅少数case/单task交换，保留不确定性，不自动扩400。
若多个target任务独立净增且合理保留已有能力，再尽快注册strict400，并按具体缺口补same-task/视频机制与稳定性。
无论哪个分支，main消费后自主接续，不等Owner例行许可；本节没有自动held400、第二种步数/尺度或其它新臂。

### 16.5 预算、执行效率与交付

ROOT=`/data1/user/ymdai/ember_runs/source_policy_space_compiler_20261015/incumbent_compilation_seed7`。
72condition各8192含settling/reset/失败未知，逻辑practice上界589,824；288query上界101,760 control+2,880 dummy=104,640。
正常逻辑总步上界694,464。沿§14完整面板计数，practice最多5,376 episode，query288，总5,664逻辑提交。
source初始144/40,377步是必须复用的已有效工作；本批正常新增物理提交至多5,520、总步654,087。
本独立批另16次完整工程尝试，不继承§14/15余量：**新增物理上限5,536提交、658,087 control、55,360 dummy、662,567联合总步**。
各分项独立上界不能相加替代联合上界。其它最终未改参数query复用会进一步减少真实新增，不释放为新科学矩阵。
没有有效行重跑、普通失败重抽或滚动充值；全局预算与每condition预算同时约束。

新增磁盘峰64GiB。原§14完整raw H压缩11.540GiB/170,804实践步，按本批逻辑最大实践粗估约39.85GiB；
留足原始H原子临时、完整path/row、少量最终LoRA、代码/冻结树及CPU物化余量，估计不是已测峰。
不复制source、模型、数据、原148GiB级特征或旧H；no-video初始H只读引用。当前无learned H消费者，不建旧16GiB dense H feature缓存。
所有新增在data1，建root/缓存/启动前live核strg01独立user quota、相关实耗和共享容量；执行者按首实际H增长核算峰值，
不足时报告具体缺口，不切data0、不删原始/不截视频。原费用区间与09:05参考首次时钟全部保留，新派发/准备/首启动另记。

唯一control_space运行面复用原D、native/History、原子账本、dynamic long-first persistent queue与完整close。
替换当前§15固定读出主动入口及§14共享H训练调度；旧原件/冻结/Git保留，只保留读取历史所需消费者，不恢复并行Writer。
任务内m更新与按arm隔离的condition预算归现有编译owner；代码组织/结构guard、真实消费者、工程恢复、Git与工作树生命周期由实验闭环。
源码必须新clean pushed detached；不原地改旧冻结。常规工程修复若不改科学计算/数据/选择/信息墙，执行者自行验证换冻结继续。

每次launch现场核gpu01/gpu02的process/显存及当前owner总量，合计6/8、单节点6、NUMA及node合计CPU/PSS准入保持；不排除仅高util设备，不挤掉foreign。
72条condition互相独立，就绪完整panel按成本动态拼有用工作，不沿用训练logical4造成16环境并发上限。
独立evaluator不使用NCCL。沿用本环境24槽成功/32槽EGL失败证据，本批不再探32；
在已有真实工作上比较16/24及实际有用worker数，H捕获/压缩与CPU限制单独测；显存余量须用于有收益的并行，不为填满占卡。
不为profile增加episode或额外parameter/sample；没有额外actor GPU常驻需求时，CPU物化与episode调度复用已有进程/显存，按实际吞吐选择。
idle/stop完整关闭环境子进程；最后核owner/GPU/env/worker退出及无残留，费用不能遗漏失败初始化或未回执区间。

含实施预计1.5–3小时：窄编译/预算/缓存复用实施约40–80分钟，实际双臂编译/读出约40–100分钟，部分就绪工作可重叠。
依据§14有H采集的真实170,804实践步与§15持久24槽约9.4–9.8plan/s；本批没有反复learned H/视频重算和共享训练，
但多condition并发、raw H压缩和实际早停比例未实测，不把纯query速率当整批保证。首有用工作修正ETA，显著超期明确解释；无GPUh硬限。
一个owner持续等待整批退出，0Queue自通知/心跳/第二waiter；完整原件、成本/资源/Git闭环后一次回main01a13dd9并归还canonical写窗口。
当前是科学合同登记，实际接手、正式启动和完成分别据实记录，不以发出消息替代已运行证据。

## 17. 冻结控制空间中的经历条件功能修订学习

### 17.1 科学判断、固定资产与唯一范围

§16实际完成并归还写窗口后，main核全路径、七臂配对、源码及最近似完整历史；
[分析§20–21](../analyses/source_control_space_learning_20261015.md#20-保住初始控制以后实际参数搜索仍没有教出总体改进)
给出原假设、竞争解释、实际负证据与本次改变。当前固定单位随机搜索不自动续跑；不是以更多候选或尺度菜单挽救86。
旧Experience已实际使用H条件FM/PG且held135，旧G真实多parent/H训练与完整165、native FM158等不利证据保持。
本次差别是固定已具有实际控制能力的linear64和Reader900初始映射，使密集功能误差直接教H到64维修订，
不是新提出反馈原理、宣称已解决held迁移，或把本来合法的路径score说成工程bug。

只使用既有显式36个non-held任务（train24＋原审计support12）；raw source、normalization、完整38-target/rank8、alpha8/dropout0不变。
`control_linear64_seed7/basis/state.pt`的U/S/center及`video_initial_seed7/checkpoints/update_00000900/state.pt`
中的Reader权重、code_units均冻结。视频图像塔/Gemma、source原生policy与D没有trainable参数。
0旧meta9/H权重或Adam、0新reference/basis、0MT初始化/底座/回退/第二adapter，0held/Test和未登记梯度。
fresh ψ seed7；唯一窗口**180次共享更新，终点180一次完整36condition/144 query**。90只作恢复点，无90读出/选点。
本次是共享离线经历条件学习后真实任务内编译，不称on-policy完整元RL；新教学condition不拟合任何动作轨迹。

### 17.2 教学、真实经历与修订算子

合法教学仍K=1、action-hidden、stride5加terminal、全部内部有序双相机RGB。沿§13.2的纯图像Gemma prefix与冻结Reader900，
得到完整`M_V[T*8,256]`及初始`u0[64]`；不输入直接L/task ID/路径/教师state/action/reward/outcome。
执行policy和训练functional query正常读exact L；真实执行hidden里的间接语言信息不删减。
原148GiB级纯图像特征只读复用；冻结Reader产生的完整M可以缓存，不平均帧、截短教学或缓存可训练H embedding跨更新。

ψ只包含§14.2已有数学形式的H读取/修订参数，从fresh初始化；实现可从冻结7a149c3e的必要算子归入唯一当前owner，
不恢复旧PG训练器或平行入口。每个真实replan事件输入为：

- pre/post双相机实际RGB经冻结视觉塔/投影的`[2,512,2048]`patch；post与下一pre相同时复用唯一实际观测。
  这是raw视觉投影，区别于教学的Gemma post-norm，不伪称相同语义空间或跑dummy suffix。
- 同次实际第0/9 flow的action_out_proj前hidden `[2,50,1024]`；后45计划token不是已执行动作。
- facts63：pre/post各8D状态、真正执行的5×7动作与5D mask、五个自身结果事实、episode/step两个计数。
  状态/动作沿固定source单位，计数除8192；五事实为自身reward/success/terminal/truncation/failure。
- 该事件实际行为代码u64；包含所有成功、失败及未最终选中的候选，不加入未来query或理想reference参数。

image与hidden分别LayerNorm后Linear2048/1024→256，加原512空间/pre-post2、flow2/action50位置embedding；
行为u经Linear64→256形成一个参数token。learned event query加Linear63(facts)与真实ordinal正弦位置，
以4-head attention读取该事件完整1125个image/hidden/参数token，再以4-head attention读取M_V。
两层causal pre-LN Transformer（width256、heads4、FF1024、dropout0）编码全部真实事件顺序，不按teacher时间硬对齐。
decision query为learned256＋Linear66(best_u64, best_success_fraction, remaining_steps/8192)，
以4-head attention读当前可见完整H，再做残差、LayerNorm、Linear256→64得到δψ。
最后Linear权重/偏置零初始化，其余沿既有模块默认初始化与seed7；H为空时δ定义为0，不造假事件。

唯一新候选为

    u_candidate = m + delta_psi(M_V, H, m, c/4, remaining/8192);
    z_candidate = code_units * u_candidate;
    Lambda_candidate = D(z_candidate).

这里m是当前已完整实践并保留的代码，**不是永远加在原y0上**。无Gaussian/sigma、局部Adam、PG、value/停止头或独立selector。
D输出原完整A/B，正常原生B*A*h与10-flow不变；每个候选在整个episode固定，最终部署一套完整LoRA。
代码/候选和关键状态以FP32保存，原生BF16/TF32/正常物理微小差异允许；不以norm、cosine或逐元素一致验收能力。

### 17.3 真实离线经历的采样与合法功能监督

H来源只读`incumbent_compilation_seed7/conditions/*/path.json`中**314个完整已发生面板**及它们的raw History：
video156/no-video158，1256条逻辑实践（其中144 source为§14旧有效引用）。只提取完整panel及其之前的经历，
由前缀自身严格成功数/平局取早规则重建m、c、已耗步数和remaining；不能读取整条path的最终best、query_rows/R、reference或未来面板。
行为代码必须来自当时实际执行的panel；no-video的H并不伪造为看过教学，新的教学与该旧H是明确同task重配的训练条件。
不同原始H/行为参数仍只构成36个独立task；两arm、更多前缀或更多视频不得计成更多meta-task mappings。

每logical update等权取4task，每task28个FM query，总112；每9更新覆盖36，180更新每task20访问，共720个经历条件/20160 FM query。
task顺序沿既有seed7 permutation流，从Reader900记录的logical origin900开始：`cycle,slot=divmod(900+new_update,9)`，
复用`permutation(0,0,cycle,36)`中的4个task；ψ更新编号从0开始，不称旧Reader第901步或exact-resume。
继承900中实际task visits/query cursor作为标签流起点，不能按猜测重建query cursor；不继承900 optimizer。

令某task新访问号k=0..19：偶数取video行为池、奇数取no-video行为池，每arm恰10次。
该arm所有完整panel ordinals（含initial及最后已观测完整panel）构成合法前缀集合，大小记N；
arm内访问号q=floor(k/2)，按`permutation(task,17+arm_index,floor(q/N),N)[q%N]`选择prefix，video index0/no-video1。
每个prefix的全部历史事件都可见，不只抽其最后成功episode；同task内前缀均匀轮转，不因某任务有更多候选而增加task权重。
20次访问不保证每个长池前缀都被见到，实际manifest保存所选前缀及覆盖。终止处完整H也可作为离线功能训练条件，
这不表示部署在4/4成功或预算耗尽后继续提出/实践候选；没有对未发生的环境结果造标签。

原始teacher50池按原seed7独立permutation均匀轮转：visit从900保存的100开始，
`demo=permutation(task,1,visit//50,50)[visit%50]`；每task本窗20条原始教学，不声称完整50覆盖。
它与同task真实旧H可以来自不同教学/参数，记录off-policy条件来源，不假称H由当前ψ或该新demo产生。
query沿`writer.learning_data._episode_queries`原stream，从900的实际cursor继续，domain3的50 episode顺序、count28、严格排除teacher_demo。
使用`FunctionalQueryDataset(chunk50,action_start_offset1)`、原source normalization、原生flow_sample真实noise/time及
`NativeFlowPrediction/mean_velocity_loss`；完整50×7 FM，不替换为首5步、native10蒸馏或假action query。
noise/time及真实query行按现有task/visit/cursor决定，物理rank/microbatch、H长度和缓存命中不得改逻辑样本或权重。

唯一loss为四task的`L_FM(D(code_units*(m+δψ(M,H,m,c,b))), independent_episode_queries)`均值。
标签只来自上述合法non-held离线动作query；自己的执行动作只作H输入，不作新动作BC，teacher及最终query标签不参与训练。
真实功能梯度经同一D/完整LoRA回到δ/decision/H/event读取；m、行为H原始记录和冻结M是常量，
不对产生旧H的环境分布求导，不用重放logp或宣称此梯度是完整随机实践目标的无偏估计。
无producer/code/E/BA辅助、无reward loss/PG/额外keep权重、无测试标签或新teacher监督。
第一步δ=0时只有head可得梯度是正常初始化；后续真实消费者须确认H获得信用，接通不替代闭环收益。

AdamW lr1e-4、betas(.9,.95)、eps1e-8、weight_decay1e-4、global clip1、constant LR；只更新fresh ψ。
这是沿既有配方的一条有界轨迹，不做LR/seed/rank/尺度小扫或多次重启挑峰。
完整checkpoint保存ψ、冻结视频/basis身份、optimizer/scheduler/scaler、H/teacher/task/query cursors、各rank RNG、topology/schema；
90/180在完整周期边界保存，原子rolling保最近及前一有效恢复点，不积累全部更新。
物理拓扑迁移只在完整点按既有合同承接逻辑流/权重/optimizer，记录新旧设备/RNG，不称bitwise exact；旧有效计算/费用不抹去。

### 17.4 终点完整编译、原件复用与停止

只有共享ψ180固定后，在原36个task、各原v0教学运行一个完整condition，不按训练loss/实践或144选择另一个checkpoint。
每condition独立8192环境步含settling/reset，practice还是§14/16从16..49抽出的四init、root202610161；
teacher还是Reader900/seen/state_000的原v0，query为[0,4,8,12]/root202610151。官方render256/model224/rotate一次/state8/action7、
10-flow/执行前5/dummy10及220/280/300/520、support400 horizons均不变。

初始m仍为Reader900已存v0代码，初始实际LoRA直接引用原`state_000.safetensors`；
必须复用§16video初始144条有效practice及完整H，共83成功、35207步（33767control＋1440dummy）。
原步数照常逐condition占8192；10个原4/4条件按同一规则止，不重复实践改变成绩或把复用省出的物理计算变成额外预算。
初始化只取该initial panel及允许的代码/教学/初态/RNG字段；旧§16后续搜索H只用于§17.3共享训练，
不得直接塞入新condition运行时H或继承其最佳参数、停止位置或query结果。

实际循环：若c=4，或剩余不足`4*(horizon+10)`完整面板，结束；否则由当前M/H/m/c/b生成确定性候选。
候选u64若**精确等于本condition某个已完整实践的代码**，保留原证据并结束，不重跑相同面板、不重复追加H；
这是有限输入下避免确定性重复的规则，不使用epsilon/norm/cosine阈值、参数hash或全LoRA逐tensor扫描。
其它有限候选先原子物化完整LoRA，再真实完成整个四例面板；严格成功数更高才保留m，平局取原m。
所有新失败/未接受候选的完整实际H加入本condition，下一次δ可以据之改变；没有预设轮数。
最后只部署已完整实践过的最佳一套LoRA，ψ/M/H/选择及所有适应辅助退出最终独立query，不在query中修订。

最终仍为初始m时，按预固定§15 Reader900原四query整组复用；任何实际保留的新代码均真实执行四条新query。
不按query择来源、改取旧较高行、以旧reference/§16最终参数替代生成或重跑有效case。
每condition/pending/code/已完成面板/实际H/步骤和工程尝试原子保存；恢复只adopt有效行，科学失败不重抽。
工程失败和unknown按实际/保守成本同时占condition和global额度；无法在现有预算内完成完整panel时携原件交main，
不缩面板、不重置预算、不把工程中断写成科学失败。

### 17.5 结果解释、交互上界与成本

本批新一臂144，加旧source58/ref127/Reader90089/first73/full79/video86/no-video59，八臂1152逻辑case作配对报告。
主比较是新完整方法相对固定Reader89，其次是旧§16video86；报告全部per-task/suite、target24/support12、identity五task、
breadth/RGL/churn/success-set/参考新增70保持，并分开实际改参数与保持初始代码条件。
列出每次候选、严格接受/平局/失败/停止、实践到query差别、首次有效修订、未测候选与不利case；不将未query候选说成存在oracle优势。
旧full是变化θ历史链，旧no-video59不是新ψ的匹配消融；本批无视频/H因果或held资格。
独立query初态不进入共享训练/H，但初始实践H属于本批共享训练池；这是训练task/已见初始经历条件上的完整流程诊断，
不称未见H泛化。后续新候选产生真实新H，其分布差异属于本次实际检验的一部分，不能由旧缓存代替。
分别报告缓存教学/读取帧当量、旧H复用/新H、训练20160query、训练与编译/最终读出真实成本；缓存不是零教学。

若只有FM下降或practice增高而独立query无整体收益，停止本组织自动续训/调参，不宣布所有反馈/64维/r8不可能；
若多target真实净增并合理保留，main尽快登记strict single-checkpoint400、相邻稳定性及适量匹配机制证据。
仅局部交换或一项surrogate变好不自动扩大；180不是充分训练保证，也没有自动360/其它checkpoint/controls/Test。
完整结果交main科学消费与自主接续，不增Owner例行许可或main工程ACK门槛。

新ROOT=`/data1/user/ymdai/ember_runs/source_policy_space_compiler_20261015/functional_feedback_seed7`。
36×8192实践逻辑上界294912；144query为50880control＋1440dummy＝52320，联合正常347232步，
沿原完整面板最小耗步/完整最坏面板准入计算，至多2688practice＋144query＝2832逻辑提交。
强制复用initial144/35207后，正常新增最多2688提交/312025步；旧query复用进一步减少新计算，不扩大矩阵。
本独立批另给16次工程尝试，不继承任何旧未用余额。新增硬上界为：

- **2704次提交**；
- **control318905**（294912−1440＋50880−33767＋16×520）；
- **dummy27040**；
- **联合total320505**（347232−35207＋16×530）。

分项是独立上界，不能相加当作联合额度；每condition8192同时有效，全局工程额不能覆盖它。
普通16不滚动充值、无有效行重算/正常失败重抽；训练不新增环境episode，0额外采集或profile科学行。
所有旧原件/成本/unknown区间与参考首次09:05:48.038时钟独立保留，本次真实派发/准备/首启动/退出另记。

### 17.6 缓存、吞吐、工程与交付

新增磁盘峰64GiB。旧§16新raw H20.603GiB/305006实践步按本批正常新增practice上界259705步粗估约17.54GiB，
另含工程失败/波动、完整raw H原子临时、最多16GiB冻结视觉H特征工作缓存、<1GiB冻结M缓存、恢复状态/代码/最终LoRA。
此为准入估计，不是已测峰；首真实归档核增长与剩余峰值，全部新增data1；建root/cache/train前现场核strg01 data1独立quota、
相关个人用量及共享容量，不能以df替代quota或不足时写data0。原source/data/model/basis/148GiB特征和旧H只读复用，不复制全池。

H视觉特征按实际需要逐archive顺序解压/编码，复用pre/post唯一观测；固定**全局16GiB**临时工作集及原子空间占用，
按当前有用batch/前缀需要缓存/释放仅本批可再生成特征，不全物化1256条H，不删raw/formal/他人文件。
完整H和视频不得截短；超出工作集用流式/重算承接真实消费者，不让容量设置改变训练条件。
ψ改变后事件/H embedding必须重新计算，不能把旧learned embedding当冻结输入。冻结M可按teacher复用。

唯一control_space承接现有D、Reader、native/FM、History、实际编译/账本/队列；替换§16随机候选主动入口，
将必要H读取和共享FM原语收进当前owner，不恢复旧PG框架、双路径Writer或复制通用训练器。
结构guard、实际消费者、源码/Git集成push、开发树生命周期与工程窄修由实验闭环；formal来自新clean pushed detached，不热改旧冻结。
普通BF16/TF32、physical microbatch/reduction微小差异不做逐bit追查；真实梯度消费者、信息墙、预算、OOM与resume正确性保持。

每次launch live核gpu01/gpu02/process/实际显存/owner总数，合计6或8、单node6及node合计CPU/PSS；不挤foreign。
训练在单node按吞吐选rank数，NCCL_P2P_DISABLE=1、NUMA/deferred NCCL；112有效query/4task和一次optimizer更新不变。
利用当前有用更新验证FM microbatch、H event chunk和流水安排；沿旧§14 chunk16..512/FM14/21等实测起点，
有显存余量须验证更大有用batch的真实吞吐，不为profile新增训练更新或重复已有效工作。
编译/读出用独立dynamic long-first persistent workers，无NCCL，不被训练logical4限制为16环境；
沿§16有用16端到端优于24及32 EGL失败证据，本批不重探32，按就绪工作/CPU/实际H成本安排worker。
不同完整condition、物化/读出可在有效总卡上限内并行；训练终点依赖不能伪装为提前完成的编译。

含实施预计2–4.5小时：必要H/FM组织与消费者约45–90分钟，180真实更新约45–120分钟，36完整编译/读出约20–50分钟，
准备/缓存部分可重叠；旧§14更新15–26秒仅作起点，新完整长H前缀更贵，不能把旧短H速度当保证。
以首有用更新/实际H长度/缓存miss和首编译成本修正ETA；显著超期明确原因，无GPUh硬限。
一个owner持续等待整批真实退出，0Queue自通知/心跳/第二waiter；完成原件/科学读出、全部费用、资源释放和Git后一次回main01a13dd9，
归还canonical工程/科研记录写窗口。实际边界可提前携证据报告，不因主讨论尚未ACK普通工程而空停。

### 17.7 具名缓存事故与一次有限恢复（main 2026-10-16裁决）

原两次owner/旧run contract/生产代码与全部成本保持。第二次跨node编译缓存key登记后丢失，原376行都是实际物理结果，
但tensor-only缓存命中身份无法认证；不称全部H读错、普通科学失败或unknown。实际两node CPU原方法夹具也复现丢key，
底层锁/rename可见性的唯一根因仍未确定。四train job仅gpu02，180结束11:14:27.930406Z早于首gpu01 worker11:14:49.969177Z，
故CP180保持唯一训练来源，0重训/补量/选90；若新增证据指向同node训练错key，须报告具体边界，不能强行认证。

科学保留集合事先由main按输入来源划定、不按成功分数追认：16个首提案的M/H全部hit=false且直接返回原始输入builder，
task `[1,2,5,7,12,17,19,20,25,28,32,34,38,43,56,73]` 的practice001共64行/17410步保留；
其中2/28/56合法首面板4/4后的12query同saved完整LoRA也保留。新增合法adopt共76行，18413control+760dummy=19173。
原initial即4/4的`[14,22,42,51,55,62,64,96,97,101]`继续保留；共有13完整condition
`[2,14,22,28,42,51,55,56,62,64,96,97,101]`不得重跑。144/35207初始practice/H及指定Reader900 query来源全部不变。

其余75proposal（71已实践、4未实践）及下游选择/停止不记标准§17；封存恰300物理行（284practice+16query），
81110control+3000dummy=84110，科学状态`uncertified-cache-provenance`。原实际参数/row/raw H/ledger/proposals/owner费用全部保留，
不改物理complete为零或unknown，不用其query选择恢复结果。精确76 case IDs/300行和逐condition费用以
`functional_feedback_seed7/analysis/main_contract/cache_incident_ruling_evidence.json`（原main小证据复制）为唯一具名清单。

仅恢复23task `[0,1,4,5,7,12,13,15,17,19,20,21,25,29,32,34,35,36,37,38,43,73,95]`，
其中13项保留认证首面板，其余10项回到原initial。按此前完整合法连续前缀重建m/c/H/remaining，用同CP180和正确输入
重新生成后缀，非采用未认证旧code重做环境。恢复后缀有独立`cache_recovery_001` leg/receipt/LoRA/H/queue身份；
task/state/env-policy噪声种子不随leg改变。旧有效76行、13完整condition、初始144均只adopt；4旧pending候选封存不采用。

原合法方法2704提交/control318905/dummy27040/total320505及原普通16额度保持；额外仅吸收本次具名事故300行84110已发生支出。
ROOT累计物理上界现为**3004提交/control400015/dummy30040/联合total404615**，分项独立不可相加；
扣除原376/99523control/3760dummy/103283total，未来至多2628/control300492/dummy26280/联合301332。
这不是未来滚动300额度或新16：普通16累计0/16，发现已耗则扣除。每condition合法轨迹仍8192（含initial/认证首面板/新实践/unknown），
物理实践另受8192+该condition具名旧uncertified practice steps限制；原事故practice79871、query4239分开。
原ledger不删行，以具名科学有效性表排除旧事故对合法轨迹的影响，全局物理支出始终全额计算。
同类身份问题再现须带原件报告，不再次清零后缀/充值。

旧run contract及§17.5保持；新增显式恢复authority/预算版本给实际consumer读取。新缓存payload必须绑定requested key/source/reader/观测身份，
窄工程修复与实际双node小CPU消费者通过后，新clean pushed detached直接恢复，无main工程ACK前置。
旧可再生特征在保留事故metadata及必要小payload证据后允许退役，不复制整16GiB；raw H/参数/CP/原件保持。
全局H16GiB/M<1GiB、新旧合计64GiB峰、现场data1 quota/shared/两node6或8/node6/CPU资源边界不变，0新梯度或其它科学面板。

旧13.930277132524383GPUh（训练12.17976899822553）、两次失败、08:30:38.195派发与旧09:05参考首时钟均保留。
本裁决后修复/有限恢复/完整读出估再45–90min，以实际首有用吞吐修正；超原4.5h时明确事故耗时，无GPUh硬限。
一个owner直接等待实际退出，完整36/144与八臂报告、事故分层成本/有效性、资源/Git后一次交main并归还写窗口，0自动后继。

## 18. 固定功能修订180的完整验证迁移

### 18.1 唯一范围与事前判断

main已消费§17合法原件、全部配对分布、实际特征/算子/信用及最近似历史；完整判断见
[分析§22–23](../analyses/source_control_space_learning_20261015.md#22-功能修订180有了小幅独立收益尚未证明经历纠错)。
训练task的Reader89→新93有4个target净增，R85/G8/L4，支持进行一次完整迁移检验，不认证H纠错或稳定能力。
旧随机搜索86、Experience135、首G/完整G165及后续native FM158等正反证据保持；不把本次称首次使用H功能监督。

唯一新增科学工作为**400个独立validation教学condition的完整实际编译及各自一个final query**。
冻结`functional_feedback_seed7/checkpoints/update_00000180/state.pt`的ψ，训练来源0b9b8080、完整180/720条件/20160 FM已结束；
Reader900、code_units、linear64 U/S/center、source、normalization、38-target/rank8/alpha8/dropout0全部沿§17冻结。
不复制/修改旧checkpoint、不使用90、不追加共享或局部optimizer/PG/FM/BA/E/producer训练、不再生成参考函数或重fit D。
只新增本节validation自身实践，0 held离线动作/state/reward/terminal标签、0 Test、0未登记controls或其它checkpoint读出。
旧source/Reader/reference资产仅按登记角色使用；不加载MT为起点/底座/回退，不部署第二adapter或expert。

这一批只读一个预固定完整方法，不以400中的部分结果选模型或改停止/预算。
主要判断相对Reader900117是否有实质跨task增益与保持，以及相对强MT153的完整差距；原>145等历史门槛不复活。
没有清楚迁移增益或广泛能力交换时，不自动续训同组织/调δ尺度/增加候选；main据实际缺口重新判断。
多task实质迁移且合理保持时，再按结果另登记相邻稳定性及匹配教学/H证据；单点超过Reader或MT不冒称最终稳定成功。
无自动第二400、90/360、same-task-other/wrong/no-video/H-null/shuffled/reversed或Test。

### 18.2 400个教学条件、初态与信息墙

唯一validation tasks为`[3,6,11,16,23,26,31,39]`，固定coverage-v1协议不改。
复用`video_initial_seed7/panels.json`的完整validation400：task、final_init、teacher_demo、video_ordinal、noise root7及registered_scene。
按`video_schedule.condition_demo_index(seed7, suite, local_task, final_init, correct,50)`核对映射；
每task50个final init0..49、50条合法教学各恰一次，与§13旧Reader及所有后续checkpoint完全一致，不按worker/GPU顺序生成。
调度task/path/video ordinal仅用于数据寻址/配对，不输入ψ/Reader作为task-ID route。

每condition实践集合沿§14.5已说明的规则明确启用：

    eligible = [i for i in range(50) if i != final_init]
    practice_states = default_rng(SeedSequence([7,202610162,task,final_init])).permutation(eligible)[:4]
    practice_noise_root = 202610161

保存实际四init顺序；同condition每个候选使用同四初态、相同env/policy/noise，不能按结果换面板。
自身final_init绝不进入它的适应。不同condition可能共用某个init编号，但各自独立实际执行，
不共享H、成功标签、incumbent、提案/停止状态或其它condition的已选LoRA；不是整个验证集从未见这些编号的主张。
每condition读取的是其同task固定一条有序教学，K=1/stride5加terminal；不择video、不融合多个LoRA。
exact L只按原合同进入执行policy，教学仍纯RGB/原frame_indices，不输入直接L、teacher配套标签/filename或人为操作程序。
真实H的间接语言信息、动作、state、接触/结果可用，不作去信息；它们不进入共享训练或其它condition。

初始控制直接引用`video_initial_seed7/generated/update_00000900/validation/task_XXXX/state_XXX.{json,safetensors}`。
metadata的code是z，以原正有限code_units恢复u0；完整实际LoRA原文件不重物化、不可改用1800或新的Reader输出。
旧900由§13原strict400选定，保留该历史，不伪称这次才在seen选定或没有用过validation作旧初始化选择。
只把白名单身份字段传给编译准备，不读取/传入旧final success、轨迹、动作或其它结果来决定实践。

400条held教学没有持久prefix缓存，不能误用原1800个non-held original entries。
需要ψ提案时，沿冻结5568f80c旧`features.load_video/extract`的真实RGB-only读取/完整纯图像Gemma数学产生M；
恢复必要算子至唯一当前owner，不恢复旧完整训练入口。完整M由相同Reader900生成，初始u0仍取上述已有产物。
只缓存冻结M（全局≤1GiB），不新建400-video高维prefix矩阵；完整prefix在内存消费后释放，未截帧/平均/降低分辨率。
初始已4/4而不调用ψ的条件无需重新读取M，但初始LoRA已有教学编译仍计入逻辑教学成本，不写成0教学。

### 18.3 同一实际编译规则与最终部署

每condition以初始Reader900完整LoRA新执行四例实践并保存实际完整H；本批没有可复用的合法初始H，
不能把旧final query（root7、无H）改称实践，或以别的condition相同init/不同教学的H替代。
initial面板及settling按实际步骤占本condition8192；每个完整候选面板仍含四例。
唯一提案保持§17实际数学：`u=m+δψ(M,H,m,c/4,remaining/8192)`，再由固定D产生同一完整38/r8 LoRA。
H事件形状、真实flow0/9 hidden、facts63、行为u、全部事件顺序、教学cross-attention/causal读取均不变。
无Gaussian、σ/scale菜单、局部LoRA更新、轨迹拟合、外部控制expert或新selector。

每次先检查c=4、剩余不足`4*(horizon+10)`或精确重复已完整实践的64维代码；分别按原规则结束。
其它有限候选原子保存真实LoRA后完成整个四例面板；严格成功数更高才接受，平局保留较早实际incumbent。
成功/失败/未选的完整H均可供后续读取，不能只保成功事件或用预算上界当固定提案轮数。
最终只能选本condition实际完整实践过的一套LoRA，ψ/Reader/M/H/搜索/选择退出final；final反馈绝不再回H或产生梯度。
pending真实候选及完整行在恢复时只adopt，不重新生成/抽样/按科学失败重试，不挑query来源。

final统一复用原registered-scene（demonstration_transfer_learning_20260927/scenes）、env/policy root7、
official render256/model224、双相机180度rotate一次、state8/action7、10 flow/执行5、dummy settling10、horizons220/280/300/520。
final初态、场景、实际噪声与旧Reader/MT严格配对；不把普通reset的近似场景当registered-scene。
最终u精确保持初始u且沿原初始实际LoRA文件时，在freeze之后引用原Reader900对应一条final row；
否则真实执行唯一新final query。两分支在事前固定，不根据旧/新query成绩、近似代码/norm或较高source行决定。
400是逻辑完整面板，新的物理query可因这种固定复用少于400，但不能删掉未改参数条件或重复有效行。

### 18.4 读出、成本和科学解释

完整400结束后报告固定方法对Reader900117、强MT153的每task/suite、breadth、R/G/L/churn、success-set交并/Jaccard。
只读复用旧Reader1800111、首G165、D72=175、D144=168的同400原件，形成七臂2800逻辑行，
不增加旧模型forward、重选旧高点或将它们称为相邻ψ点。来源沿§13 results.validation_references及两Reader raw_rows。
旧MT没有initial_proprio字段的限制保留，沿登记scene路径/bytes、task/state/env/policy和实际首noise核配对；
与Reader有proprio的配对照常核验，不新增hash侧文件或逐tensor扫描。

分别报告初始未动/实际改参数、初始实践0/非0、首/后续提案、接受次数/停止原因、实践→final的得失和任务覆盖。
逐condition列实际步数/重置/episode/候选/教学读取与重读/ψ调用/缓存命中，工程未知单列且收费。
最多一个final query/condition，未测中间候选不能推oracle；不同condition初始化或后段收益不能拼成另一完整方法分数。
这是有交互适应，不称zero-interaction。全局共享权重固定、held实践只留本condition状态，不视作共享held训练。
没有视频/H因果消融，故即使提高也先支持整套过程；不能从旧no-video59或注意力/代码几何认证动态操作理解。

### 18.5 新批独立环境上界

400×8192＝**3276800实践联合步骤**，逐condition8192含settling/实际工程未知，不能互借余额。
400 final最大132000 control＋4000 dummy＝136000；正常联合总上界**3412800**。
未全成功面板最少horizon+43、开始前须留4×(horizon+10)，各220/280/300/520的面板数数学上界28/22/21/11；
每suite100个condition，故practice最多32800 episode，加query400，正常最多33200提交。
这些仅为费用保护上界，不新增轮数菜单或强制跑满。

新独立工程余量**64次episode尝试**（相当于至多四个16槽pack的有限恢复余量），不继承§17或其它旧批未用余额、不滚动充值。
合计上界为**33264提交 / control3426080 / dummy332640 / 联合total3446720**。
control上界=3276800−16000最低initial dummy＋132000 query control＋64×520；联合总额另加64×530。
分项是独立保护，不得相加后替代联合上界；全局额也不覆盖每condition8192。
未知执行按登记完整上界占额，补跑成功不清旧未知费用；只恢复已存同候选/四初态/噪声，普通科学失败有效不重试。
无法在原condition与全局额内恢复完整面板、缓存身份不可信或模型科学计算改变时，保留有效原件和具体边界交main，
不能静默丢行、缩query/实践、扩大矩阵或再次清零后缀。纯工程窄修按现有权限新冻结后自主接续，无例行main工程ACK。

### 18.6 存储与原始经历生命周期

新唯一ROOT：`/data1/user/ymdai/ember_runs/source_policy_space_compiler_20261015/functional_feedback_validation180_seed7`。
新增峰值**96GiB**；旧模型、数据、source、teacher、CP180/Reader900、已生成初始400 LoRA、旧H及原件均只读复用、不复制。
最近quota快照余量约171GiB只是估算依据；建root/cache/launch前在strg01刷新data1独立quota、相关实际usage和shared容量。
不得以df代替quota或转入data0；现场不足时按实际缺口处理，不能先启动再解释空间。

本批原始H从创建时就分为以下明确生命周期，完整模型输入一律不变：

- **长期原始H证据**：事前固定每task的final-init0，共8个完整condition的全部实践H；不按结果挑选。
  工程失败/未知的必要原始残片与身份witness另保留，受64次有限工程余量和总峰约束。
- **在途完整H工作集**：其余条件的raw RGB、同次native hidden和其它实际输入必须完整保留到该condition冻结、
  final query完成/指定复用、逐事件事实与全部审计元数据成功原子落盘。只在无pending/worker消费、完整close后才可退役。
  最多32个在途condition；这是覆盖8GPU×16有效episode槽÷4的原始H存储工作集，不额外限制就绪GPU/最终query吞吐。
  动态long-first队列优先推进已准入condition并及时回收完成工作集，不能先物理采满400初始H再长期堆积。
- **全部400长期证据**：原始episode rows、每次真正执行动作/执行mask、pre/post state、reward/success/terminal/truncation/failure、
  事件顺序和pre/post观测引用、行为代码/真实完整候选LoRA、frame/step/seed、提案请求/receipt、接受/拒绝/停止和全部费用。
  删除raw payload前保留其schema/身份/shape/字节数和消费/退役回执；不能用summary覆盖原始row或省略失败。

仅对上述本批已声明临时文件实施逐condition退役，不删除旧formal结果/原始H、source数据、checkpoint或归属不明文件。
未保留raw的392个condition必须明确标记为只保事实/参数/读出，不再称可完整H replay或适应中exact-resume；
其最终LoRA仍可独立部署/评测，已完成condition不再重开。中断未完成condition的完整H保持，可合法接续。
审计8条、故障残片及全部实际闭环raw rows均长期保留；不能因归档规则丢掉旧事故/正常科学失败成本。

全局冻结H视觉缓存沿已修复SQLite/payload身份机制≤16GiB，冻结M≤1GiB；同目录ready/reserved/partial/控制文件都计入，
不得另开平行缓存绕界。原始H退役时同步退役只属于它的可再生缓存条目；其它仍有消费者的输入不得删除。
不在最后一次ψ读取之前裁H、压缩为少数token、降精度/分辨率或丢帧；不将预存learned H当作另一套近似算法。
按完整raw张量估算每condition不超过约1.1GiB，32在途＋8保留约44GiB；全部最多7800个新候选LoRA约19.3GiB，
另17GiB冻结缓存，余量覆盖故障残片、原子临时、元数据/代码。实际长H序列化/峰值须在有用工作核验，不能依赖gzip最好压缩率。
若超此估计，先暂停新的condition准入、保留已有有效进度并报告具体资源缺口，不截断科学输入或删除指定证据。

### 18.7 执行、吞吐与交付

复用唯一control_space的固定ψ/M/D、Compilation/practice/native Runtime/History/账本/动态队列和已修复缓存；
当前主动入口改为固定180的validation编译，不保留另一个平行Writer、旧训练恢复菜单或复制一套held实现。
当前36/四query/已有initial-H的硬编码须按本节显式400condition/八task/单final/新initial事实承接，
不能仅绕过count校验而让真实consumer仍取旧36 source/历史H或旧query。必要模块的结构/生命周期和实际消费者由执行者闭环。
保留训练producer0b9b8080与本次读取代码身份分离；新代码须独占开发树、实际消费者核验、集成push后用clean detached冻结。
不热改26a8fbb3或旧正式工作树；窄工程修复自主新冻结，不让无依赖就绪工作等待无关worker。

每launch/resume live同时核两node GPU/process/显存、quota/shared、CPU/PSS；合计8卡、空闲≤10时6卡，单node≤6，foreign不挤退。
按实测利用可用显存/有用worker/完整H chunk，不沿用单训练world size作全局吞吐上限；独立compiler/evaluator不启NCCL。
沿§17有用16槽和H后释放inactive CUDA cache的实际证据，长H阶段close无用环境，panel前给EGL留真实空间；
不重探已失败32槽或新增无判断价值的profile科学行。新的单final query可按实际就绪case打包，不能补假episode凑四例。
新冻结M的完整teacher读取可用既有16/32/64 frame chunk吞吐证据结合实际余量选择，不为低位一致固定batch1或逐tensor对比。

预计自实际派发起**6–12h**含必要实施，依据当前H/环境有效吞吐、400条件可能比seen更少早停和6–8张可用A40；
首批已授权有用condition修正吞吐/peak/总wall预测，不按其成绩选模型/改预算，也不另加profile矩阵。
GPUh准确记账但无硬限，设备不足不等待凑卡或擅自改科学规模；显著偏离预期应说明实际原因。
所有历史根的费用、事故区间和最初09:05:48.038参考时钟保持，本新批另记派发/准备/启动/首有用工作/退出。
单owner持续等待整批实际退出，0自Queue/心跳/第二waiter或固定间隔读共享日志；科学/资源实际边界另报。
完成后交400原始配对、全部条件/预算/归档原件、完整成本/吞吐/资源释放与Git，向main01a13dd9一次可靠回报并归还写窗口，
无自动后继。main消费实际结果后自主接续，不以Owner新许可或工程ACK造成停滞。

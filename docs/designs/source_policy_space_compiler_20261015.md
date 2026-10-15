# Source/rank8控制空间编译器：参考控制器与实施合同

2026-10-15；科学owner为主讨论01a13dd9-837d-7f33-b15a-4eb170bcf04b，唯一实验执行为
01a13dbf-dbc3-7621-82be-87be44a5bca6。Owner已授权自主落实、分析问题并接续，无例行许可停点。

## 1. 目标、当前窗口与历史边界

最终目标仍是合法教学/自身实践编译出唯一完整LoRA，在固定held协议明显超过强MT，同时说明教学作用和能力保持。
当前后继从冻结raw source起步，所有生成、实践和部署LoRA均为38-target/rank8、alpha=8、dropout=0；
不以MT初始化、合并或回退，不部署额外高rank分支。强MT153、G165、D175/168与所有负证据保持原身份。

完整方法的候选原理见[控制空间编译分析](../analyses/policy_space_meta_compiler_proposal_20261015.md)：
学习D(z)产生完整r8；视频及真实H给出连续参数提案；固定D时用独立query回报训练全部提案/停止的联合score。
旧经验编译已有真实反馈学习，本次不是新的反馈架构原理；source/r8、多任务控制空间、全路径score均仍待实证。

**当前执行§9：补齐窗已完成，32项冻结；只对5/25/38/43沿末1920完整状态作有界收尾。**
§2–8保留已完成窗口的合同；§9覆盖这四项的候选节点及本次预算。不是启动D、q或meta-RL，
更不是完整方法/held资格。参考库完成后仍须检验控制空间和视频编译；不因文件齐全而自动取得后段资格。
当前不恢复N16、旧Writer、MT教师、Phi、GT/sealed诊断、validation或Test。

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

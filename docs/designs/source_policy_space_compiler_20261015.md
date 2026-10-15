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

**当前执行§12：operator1170仍未保存参考控制，撤换共同E/D，一次拟合固定64维线性参数基底并做函数保留闭环。**
§2–11保留已完成窗口的合同；36参考均已冻结，两个E/D窗口都没有保存广泛参考能力。
§12是新的表示/拟合合同，不恢复native或operator优化器，也不自动扩大维数。没有q、视频Reader或meta-RL计算，
也不是完整方法/held资格；参考控制和已知函数重构均不能替代教学编译的实际检验。
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

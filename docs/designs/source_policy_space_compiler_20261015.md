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

**当前只执行参考控制器首窗：同source八个训练任务的fresh r8学习、充分闭环选点和独立读出。**
不是启动D、q或meta-RL，更不是完整方法/held资格。主讨论收到首窗结果后自主判断余下28任务与decoder学习；
若有实际学习缺口，先解释并解决，不因末端还上升而假称完成，也不机械扩到36或无限续训。
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

当前八任务固定为`[0,1,12,13,20,21,32,34]`，均属于train24，每suite两项，继承既有内部面板而非按新结果选任务。
完整将来池仍是train24＋已审support12的36任务：
`[0,1,2,4,5,7,12,13,14,15,17,19,20,21,22,25,28,29,32,34,35,36,37,38,42,43,51,55,56,62,64,73,95,96,97,101]`。
本窗不训练额外28任务；support12的source已见身份不改变，不能把36全部称source未见。
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

# Source时间对齐与v5.2复核（2026-09-15）

## 目标与授权

Owner评估了“同规格重训source→复现v5.2 A→测试B→按source影响决定后续”的顺序，随后明确要求
“你设置个合适的goal推进这件事吧”。本计划据此成为active design，授权实现、验证、正式训练／评测和有依据的后续推进。
先前C保持关闭，不恢复旧共现训练。实际状态只看progress，不把尚未运行的阶段写成结果。

2026-09-16 owner收敛本轮范围：先重新建立正确对齐的source、共享SFT和原始v5.2 A基线；
若三者与错位版本在对应节点没有实质差异，允许不重训B，直接分析已有B相对A的差距并停下来讨论。
只有基线变化使旧B不足以回答问题时才运行已登记的有限B窗口。无论是否补B，本轮结束于基线比较、
差距分析及与owner讨论，不自动复查历史失败方法、补2×2其它角、追加视频controls或继续寻找>145的方法。
上述最新范围覆盖下方旧的自动后续设想；>145及视频因果要求仍是项目正式方法资格，不是本轮无限续训的理由。

同日owner进一步要求：A900完整验证结果出来后停下来讨论。A900当前已登记的correct400／train96完成并审计后，
不启动A1200或新B；SFT按owner单独回复继续现有后台训练，到400步自动停止，不启动其后续评测或425／450续训。
Goal已由owner暂停。下文1200及SFT450均保留为原登记预算，不构成此暂停之后的自动执行授权。

随后owner单独授权新的A900视频特异性诊断：固定`macro_00000900`及已有correct140/400，不重新选点。
补same-task-other、cross-suite-wrong、no-video、shuffled、reversed各400；后两臂最后评测，真实RGB重排后
完整生成LoRA，exact target language不变。所有臂复用seed20260911的50初态／视频ordinal及env／policy RNG；
same-task-other逐行换视频且整轮无重复，wrong使用预定跨suite donor，no-video按现有零LoRA合同评测裸source。
全部训练／架构／checkpoint选择反馈关闭，Test封闭。A900仍未通过>145或稳定资格，此处仅是owner授权的
冻结模型原因诊断；完成逐task／suite、breadth、R/G/L、churn及任务bootstrap读出后报告并暂停。
原件集中在`runs/analysis/source_alignment_20260915/A/video_specificity_step900`，执行仍用冻结`575c189a`。

先消除已确认的观测—动作标签错位，再判断source与Writer各自的能力限制。长期资格仍为validation8
single-checkpoint strict correct严格>145/400、相邻稳定、低churn、四suite贡献、same-task换视频鲁棒性及冻结后的因果controls。
Test保持关闭，部署信息墙、train24／validation8／test8和source71审计排除均不改变。

## 已确认的时间关系

固定LIBERO生产代码先执行action再保存RGB／proprio，原states/actions保留同一原索引：
`states[i]=s_i, actions[i]=a_i, obs[i]=o_(i+1)`。本地train0/12/20/34及source allowlist首task的demo0前11行
关节观测均与下一行simulator state精确对应，间隔0.05秒；核验未使用held动作。
原source commit e2cc238和历史v5.2从同一frame_index取action chunk，现行Writer已采用offset1。
官方评测读取实时观测、执行预测chunk第0–4条后重规划，没有跳过第一条预测。

修正监督为`obs[i] -> actions[i+1:i+51]`，RGB和8维proprio保持同一时刻。
无未来动作的末观测不作执行query；不足50条沿原末动作padding，teacher完整视频仍保留真实末帧。
旧闭环分数仍是真实行为证据，不能由此认定错位解释全部历史负结果。

## S：同规格fresh source

原参照为`runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722`，新source保持：

- 同revision generic `lerobot/pi05_base`、审计后71个LIBERO-90 tasks、每task全部50个episodes，不引入target40 actions。
- 全参数SFT，包括vision／VLM／Action Expert；seed7、1000 updates、effective global batch256，共256000个queries。
- 原task／episode均衡采样、AdamW betas(.9,.95)、eps1e-8、weight_decay1e-10、global clip1。
- Linear warmup333后constant5e-5，BF16及原模块dtype、双相机、processor、chunk50、source normalization与tokenizer。
- EMA decay .999及完整恢复状态；预先固定raw step1000供后续使用，不在raw／EMA／中间点间选优。
- 只改未来动作标签与必要的有效frame支持；不声称与旧训练逐query完全相同。

原8×A100、物理batch32、约71GB峰值属于历史执行条件。当前以live A40资源和等效microbatch／梯度累积保持global256。
允许rank0 CPU保存不参与梯度的EMA，以及减少optimizer临时张量／共享DDP gradient bucket；这些须由真实更新profile验证。
不冻结参数、不压缩曝光、不降低训练dtype。旧120分钟属于旧机器窗口；本轮严格1000更新，墙钟由profile另行登记。
采样保留原8个logical task slots／每槽32queries，再按logical global query stream打包到物理rank／microbatch。
不能因累积次数增加而改成更多task、每task更少queries。原任务／episode／frame调度与历史实现已用71tasks、
1000 logical updates共256000 queries核对一致；offset1使可执行frame支持变化，仍不声称与旧标签逐query等同。
允许256不能整除world时按query数分配（如三rank为86／85／85），末microbatch取实际剩余queries，
每个mean loss乘`local_query_count * world_size / 256`后由DDP平均，保持每个query严格等权。
每个rank的每次microbatch均为真实非空query；world、GPU UUID及完整物理batch计划进入resume合同。
原生未参与action loss的language-output heads继续按原图处理；累积DDP使用unused-parameter发现，不人为增加额外forward／loss。
DDP保留已建立的gradient bucket views并原位清零；fresh／resume的首个真实microbatch先同步以建立views，
其余累积仅在末microbatch同步。首项已平均的梯度再次取平均不改global256权重，不增加训练样本或optimizer更新。

恢复一个source训练入口及cohesive训练owner，复用现存setup／contract／checkpoint／dataset，不复制source-SFT或Writer orchestrator。
新config单独登记，原sealed config及source保留。数据／模型复用固定revision、manifest、file sizes与provenance，
不重复整库hash扫描或复制资产。Profile验证实际全参数、下一步标签、finite更新、EMA和峰值，随后验证多卡global256及resume。
正式运行来自clean pushed detached树；中途恢复点只用于resume，允许新完整点发布后按keep_latest=1退休，raw1000完整保留。
物理拓扑、checkpoint interval与命令在profile后封存于唯一launch contract。

完成后固定raw1000做validation400及train96官方闭环，旧source47/400、17/96的同合同面板可明确复用。
不使用held动作诊断或梯度，不按验证分数延长source或挑选source checkpoint。

## 新source上的A与B

| 臂 | 教学输入 | 完整50-H读取 | 共有模型 |
| --- | --- | --- | --- |
| A结构复现 | 单agentview | fixed mean | v5.2 Core／Procedure／三Meta／完整38-target rank16 A/B |
| B | 同步agentview＋eye_in_hand | 现行learned H-read | 与A相同 |

Owner明确要求A结构参照，单视角／mean在本计划内获授权；不重新引入旧错位或同episode query。
两臂fresh，不直接给旧Writer换底座；保留raw frame indices RoPE、stride5＋末帧、原Gaussian probe和纯跨episode FM。
不增加RL、辅助loss、第二adapter或其它表示路径。Mean组保留相同零初始化q/b并固定为零，learned组学习它们。
共有参数保持相同构建顺序／RNG；保留B已有初始化流，不因诊断模式额外初始化或重设seed。

统一沿B：train24、teacher/query demo0–45严格跨episode、demo46–49固定train96；每更新4task、每task21queries，
baseline task permutation分组，不用C分组。共同事件、flow噪声、task权重、optimizer和全局LR时间轴一致。
每臂最多1200更新／4800条件／100800queries；每100保存，300／600／900／1200做correct400＋train96。
各节点复用B seed20260911的state–video映射、env/policy seed7及全50视频无放回合同。

先完成source、下节共享SFT及A，再决定是否需要新B；实现等独立工作可并行。旧B已有offset1，是source效应的主要下游参照。
旧A125／132只作历史定位；新A还修正了旧采样／标签合同，不能将其全部差额单归source。
旧A正式映射使用video seed7，本轮A／B沿用已登记的seed20260911；两者都在每task使用全部50条视频各一次，
但400组state–video配对仅8组相同。因此跨历史A的逐行R/G/L只在task／初态／env和policy RNG层面配对，
同时包含teacher分配变化；不得称作完整video-paired因果差额。本轮A的相邻节点及本轮A／B仍完全复用同一映射。
实测边界与旧A900→旧B900读数见study的`archival_A_B_comparison_boundary.json`。

## 共享SFT基线补齐（2026-09-16 owner追加）

历史109／107来自train24单套共享rank128 LoRA，而非source71全参数SFT或逐task experts。
固定原实际run `pi05_source_sft_rank128_mixed_dev_r4_b144_seed7_s2400_20260728`为训练规格参照：
24tasks、每task全部50episodes、每更新各24个query、global576、seed7、原episode／chunk均衡调度，
原AdamW／warmup100／cosine时间轴、相同38-target rank128参数化和冻结normalization。
原实际完成450更新／259200queries；400／425两个配对参考分别109／107，不把旧模板config的rank16／800步当实际运行。

新SFT从对齐source固定raw1000出发fresh初始化共享LoRA及optimizer／scheduler，正确使用offset1。
固定同样450更新，保存100／200／300／400／425／450，400／425／450均做完整validation400并比较相邻成功集合；
保留原声明2400训练时间轴的真实LR序列，不把调度压缩至450。物理rank／microbatch／累积由A40实测确定，
保持global576及24task等权；部署只有source加这一套共享LoRA，不读取teacher video。
旧run使用过在线validation action-loss监控，本次不复制该历史做法；validation/test actions不读取、不产生梯度，
不根据held action loss停训或选择。全部闭环只用现行固定官方400初态／RNG，Test保持关闭。

复用唯一`train_source_sft`及现有checkpoint／评测适配；修正过时的硬编码offset与训练调度，避免另建SFT runner。
先恢复科学合同、验证实际更新及恢复，再封存配置与formal launch contract。SFT的rank和动作曝光多于Writer，
它是普通共享监督的能力参照，不宣称同参数量或同query预算公平比较。

实测执行登记：两张A40、physical micro64，每rank按64/64/64/64/32累积五次，global576；
三次真实更新及step1→3完整恢复通过，原生混合BF16/F32 LoRA dtype与历史rank128一致。
后两步平均74.596秒、reserved峰值32.934GiB，450次纯更新估计9.324小时，另计加载、保存与三轮闭环。
正式仍从fresh identity开始；profile参数不复用，配置封存后从独立clean pushed detached运行树启动。
小型CPU／profile与完整恢复证据保存在`runs/analysis/source_alignment_20260915/SFT/`。

## 裁决与后续

裸source与Writer分别报告absolute、相对各自source的增量、per-task／suite、breadth、R/G/L、churn、相邻overlap及
配对task-bootstrap区间。Source同时影响原生观察与执行，裸source接近不能单独证明它对Writer影响小。
固定比较旧B四节点，重视900与1200的共同方向及任务分布；每400净增20（5pp）预先登记为有实际规模的影响参考，
必须结合不确定性、相邻一致性和覆盖，不作为新资格线。区间宽时保留“尚不确定”，不把不显著解释成等效。

- 先比较新旧source、SFT与A的对应节点、per-task／suite、覆盖及相邻保持；不把不显著写成等效。
- 三个基线若均无实质变化，按owner授权跳过新B，结合已有B及已核实的模型／监督／初始化差异分析。
  这是是否追加计算的决定，不证明错位完全无影响；宽区间和不同底座造成的归因限制保留。
- 若新基线存在重要变化，使旧B难以解释当前A，则完成既定B有限窗口后分析，不扩训练预算或做小扫。
- 分析区分获取不足、已有能力丢失、覆盖迁移和具体模块假设；只有两个角不能唯一分开双视角与learned H-read。
  无法唯一归因时列出仍存的竞争解释与最有区分力的后续，带证据停下来讨论，不自动实施。
- 历史失败路线、2×2补角、最终视频controls及Test不在本轮自动执行范围。科学non-pass不自动作为工程bug。

## 工程、存储与交付

主集成为main，Writer模式改动使用task-owned worktree；主agent拥有source及整合，formal jobs来自clean pushed detached树。
模型／data／config身份覆盖source引用、offset、camera与H-read；不同架构fresh，exact-resume锁world/topology。
运行差异以配置表达，活动树只有一个Writer实现；历史configs、source、checkpoints及正式证据保留。

首次strg01 data1 quota为908275108KiB used／1073741824KiB soft，约866.2／1024GiB；项目实测834GiB，共享约83TiB可用。
S+A+B追加峰值预算112GiB，包括旧＋正在写的source恢复点、EMA／optimizer、两Writer checkpoints／banks／rows和profiles。
本地已无generic权重，按原revision补回约13.5GiB并纳入上述预算；数据、tokenizer、旧source和环境复用。
下载期间允许旧source的同尺寸全参数容量smoke；它只定A40显存／吞吐，不作为fresh source或科研结果，
formal前仍须generic初始化的真实更新及resume核验。launch前按实时quota及实测checkpoint复核，后续旧架构另估预算。
当前状态进入既有ledger，完成证据进入research_history；验证后集成推送，清理已集成开发worktree及明确临时文件。

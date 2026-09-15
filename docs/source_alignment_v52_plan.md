# Source时间对齐与v5.2复核（2026-09-15）

## 目标与授权

Owner评估了“同规格重训source→复现v5.2 A→测试B→按source影响决定后续”的顺序，随后明确要求
“你设置个合适的goal推进这件事吧”。本计划据此成为active design，授权实现、验证、正式训练／评测和有依据的后续推进。
先前C保持关闭，不恢复旧共现训练。实际状态只看progress，不把尚未运行的阶段写成结果。

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
采样保留原8个logical task slots／每槽32queries，再按连续global query stream打包到物理rank／microbatch。
不能因累积次数增加而改成更多task、每task更少queries。原任务／episode／frame调度与历史实现已用71tasks、
1000 logical updates共256000 queries核对一致；offset1使可执行frame支持变化，仍不声称与旧标签逐query等同。

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

先完成source及A，再B；实现等独立工作可并行。旧B已有offset1，是source效应的主要下游参照。
旧A125／132只作历史定位；新A还修正了旧采样／标签合同，不能将其全部差额单归source。

## 裁决与后续

裸source与Writer分别报告absolute、相对各自source的增量、per-task／suite、breadth、R/G/L、churn、相邻overlap及
配对task-bootstrap区间。Source同时影响原生观察与执行，裸source接近不能单独证明它对Writer影响小。
固定比较旧B四节点，重视900与1200的共同方向及任务分布；每400净增20（5pp）预先登记为有实际规模的影响参考，
必须结合不确定性、相邻一致性和覆盖，不作为新资格线。区间宽时保留“尚不确定”，不把不显著解释成等效。

- 若有重复节点支持的下游改善，可有限复查具有局部正证据且依赖source表示／动作计算的历史架构。
  每项先说明新增证据、保持变量和停止条件，不一次恢复全部路线。
- 若未见重要source收益，或A/B差异仍是主要缺口，在正确source上继续研究v5.2获取、保持及B<A。
  A/B构成2×2两个角，必要时只补单视角＋learned、双视角＋mean，不同时改变其它变量。
- 明显负结果不靠无限续训或seed／rank／scale／LR小扫挽救；科学non-pass不自动作为工程bug。

先判断能力与稳定性，same-task-other按登记资格复核；选定single checkpoint冻结后才做wrong／no-video／shuffled／reversed。
若A/B均无资格，为回答本goal的视频优势问题，可固定各自1200终点做一次sealed post-hoc视频读出：
全程冻结、不选checkpoint、不将controls反哺训练、checkpoint选择或后继架构修正，Test仍关闭。
保留历史视频正证据，重新检验新模型，不把更强source带来的上涨当作视频特异性。

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

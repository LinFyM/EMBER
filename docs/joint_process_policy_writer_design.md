# Joint Process–Policy Complete LoRA Writer

## Authority and purpose

Owner于2026-09-06明确授权按讨论规划实际推进至最终科学目标。当前主案复用已有P/Q与完整原生证据读取，直接生成全部38-target
完整LoRA，无独立carrier。首选rank16；rank8须由实际成本及学习/行为证据支持。旧12+4、A2、P/Q有限训练结果和随机运行中断现场
由Git与formal artifacts保存，不是待恢复的执行清单。专家20260905原文作为证据与启发，不把其整套参数化当成已证明最优。

真正参照为train24 SFT历史109/400和旧Writer143/400。后者尚未证明相邻稳定与充分视频必要性；前者须补齐同口径比较后才能作
strict paired结论。当前source47/carrier72只作既有机制参考，不能将约80的Writer称为充分有效。

## Scientific change and evidence boundary

最近实现93540ff1有共同P/Q、完整50-horizon读取，但最终仍为carrier12+signed-native mobile4。其短四任务结果没有超过A2，
不支持已证明P/Q机制优势，也不否定所有完整生成方案。旧143整图包含任务grounding、语义/过程与全策略compiler及自由factor heads。
当前首先改变完整输出职责：移除独立carrier、raw-X/Y-only因子span和mobile-only静态零输出限制，用共享learned heads写全部A/B。
这是耦合的输出接口重构，不能将任何改善单独归因于rank、去carrier或head自由度。首轮不同时修改observer、监督、任务权重或扩大主干。

正证据仍保留：native response有可学习动态，G1存在局部native容量，旧Writer存在真实闭环能力。它们不等于共享生成已经解决。

## Data flow

```text
exact language + K internally ordered action-hidden correct videos (stride5)
 -> frozen source native capture: image/language prefix, full layer/probe/horizon responses
 -> independently per video:
      learned process states P[frame,M=8,d=128]
      joint policy states Q[target=38,rank=16,side=A/B,d=128]
      four repeated standard attention/MLP blocks:
        task-conditioned patch grounding, full-response read
        temporal P interaction; Q reads P and mixes across targets/ranks/sides
        Q feedback conditions the next evidence read
 -> permutation-invariant learned read over video policy states
 -> family-shared learned factor heads -> all complete A/B
 -> one complete LoRA installed on frozen source before rollout
```

Teacher frame time、relative action horizon、flow time、layer depth和probe保持独立。所有19 boundaries、18 residuals、50 horizons和
两个真实固定antithetic probes在task-conditioned learned read前保留，禁止mean/coarse等价无条件平滑。source响应不是正确动作真值。
每条视频内部保序，集合阶段才聚合learned states；不平均raw frames/features/LoRAs、不挑video、不部署第二adapter。
首个候选只声称K1；实现的集合接口不等于已证明dynamic K，后续使用K2/K4须训练真实覆盖。

## Module ownership and output

`policy_response_writer/process.py`复用93540ff1的完整证据tokenizer与JointProcessPolicyBlock；`composer.py`拥有共同P/Q与完整factor
heads；`model.py`拥有唯一Writer调用和完整LoRA映射。现有training/shared/materialization与dynamic evaluator复用，不复制第二套runner。
旧target-local A2、signed readout及专属task-query训练路径在canonical替换时退役；历史公共native工具若仍有别的真实调用者不连带删除。

语言用于任务约束与证据读取，生成内容必须吸收真实视频语义与过程。P/Q的公共target/rank身份不得成为task ID字典或部署专属参数。
完整LoRA不能强制继承mobile-only静态零约束；全视频与静态/语言的必要增量最终由冻结controls证明，不由代数零构造冒充。

输出A/B遵守真实target维度，family-shared heads批量处理同形targets。采用非零A、零B的标准函数identity起点，随后全部因子可训练；
source始终冻结。新增头与P/Q联合接受functional梯度；只复用语义/shape匹配的G2投影初始化，不声称继承旧143参数或完整行为。
不添加target cap、SVD、norm、gate等无新证据的输出数学链；保留shape/finite及实际LoRA安装检查。新增架构必须fresh optimizer/schema。

## Training and staged allocation

首轮监督仍为correct video -> one LoRA -> same-task cross-episode true action flow loss；Writer不读取query actions/state。
复用四任务及73-task allowlist、原两fit视频、fresh action query、现有冻结normalizer；旧carrier量仅可作为已冻结loss尺度/诊断元数据，
不能作为生成或部署参数。不得为删除名字而改变已匹配的loss reduction。

四任务只做新接口短学习/已有闭环对照，不重跑完整clone/shared、expert容量或SFT训练。主要数据仍是现有73-task组合（55meta+18target）；
留出的train侧任务继续用于迁移诊断，纳入全部train24时单独登记覆盖变化，validation/test及重复项始终排除梯度。

明确分别记录optimizer updates、per-task exposure、video/query覆盖，不能将固定64步当成充分训练/收敛。实际更新量和检查点在launch前
按profile与有信息量问题登记；若行为改善，训练到可判断相邻稳定，若明显停滞则区分输入/读出/共享/视频/任务迁移/occupancy解释。
不以低loss、梯度norm或几次success净差单独选模型；不靠无依据seed/LR/rank小扫挽救。

出现真实成功遗忘时才研究训练侧成功回放；离线学习与新视频成立、闭环因访问状态失配时才研究可信learner occupancy。
先对照guards/SEOD/GOMQ历史等价尝试；不把expert occupancy包装为从未做过的方向。teacher actions/reward只在授权non-held训练侧使用。
同拓扑fully-random fresh是值得研究的Final候选，放在有前景架构上；不自动续跑旧弱图的未完成random。

## Evaluation and decision

早期固定validation8每task10 states的80-row screen只用于分配训练/评测投入，不选最终checkpoint，不线性外推400分数。
使用correct及必要的same-task-other positive arms；negative视频controls不参与该阶段。两正确视频沿用事前无放回固定ordinal：
全局tasks1/3/11/13/23/26/31/32分别primary/other42/39、12/23、39/36、15/4、16/6、45/26、45/34、6/1。
使用同一state与env/policy RNG、官方预处理/horizon，不挑视频或initialization。

接近强基线/目标、覆盖较广且连续节点有可保留行为时，先做一次strict400核实规模；之后才展开相邻与另一正视频。
正式候选只从完整single-checkpoint400决定。相邻节点顺序在训练/eval启动前登记，不用screen挑出彼此不相邻的高分点充当稳定。
数值科学资格沿用：

- 相邻两个checkpoint × 两正确视频的四个400 panel均strict >145。
- 每panel至少6/8 tasks各>=5成功，四suite各>=5，Goal/Long各>=15。
- 每video相邻总分差<=20，churn<=40，retained/max>=.85，Jaccard>=.75。
- 每checkpoint跨视频总分差<=20，retained/max>=.85，Jaccard>=.75；另报逐task/suite R/G/L。
- 合格相邻pair内最大化两视频较低分，依次用两视频均分、较晚step破同分。禁止union、融合或选video。

正式宣称超过SFT前确认旧checkpoint与当前policy/normalization/evaluator兼容，优先复用权重做必要的一次同口径评测；不默认重训。
历史109和143不能未经配对直接作显著性结论。训练数据/rank差异透明报告。

选定冻结后补wrong/no-video/language/first-final，最后shuffled/reversed；只作最终视频/时序必要性裁决，不反馈训练、选择或架构。
方法冻结后按固定32 source /8 test fresh训练和最终评估，test结果不反向设计。

## Runtime and verification

复用真实batch capture、GPU tensor布局、checkpointed evidence读取、node-local共享cache、cost-balanced task dispatch、persistent
long-first evaluator。保持全部视频/horizon、正确梯度与task权重；不把整个证据展开成无必要的巨型全attention或逐target重复大forward。
最长真实视频分别测训练/LoRA generation吞吐与峰值，rank8只有在rank相关瓶颈确实重要时考虑。

每次launch live检查两节点、独立quota和共享容量，总量最多6GPU、单job单节点，NCCL/NUMA按repo合同。formal来自clean pushed detached
frozen tree，精确resume锁原world topology。只做与变化匹配的直接验证；文档/清理在GPU等待阶段完成，不阻塞科学结果。

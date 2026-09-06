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
这是耦合的输出接口重构，不能将任何改善单独归因于rank、去carrier或head自由度。首轮不同时修改observer、监督、任务权重或扩大主干。输出不再消费的逐target X/Y bank捕获与缓存移除，完整响应轴不变。

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
source始终冻结。新增头与P/Q联合接受functional梯度；只复用语义/shape匹配的G2投影、公共owner/family与首层读取attention初始化，不声称继承旧143参数或完整行为。
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

## Registered complete-output main scale-up (2026-09-06)

四任务complete输出32 fit/held为50/54、64为64/62（各150），超过旧有限输出P/Q的38/37、41/39与A2的43/41、44/45。
Goal32→64为21/22→31/27；训练侧学习和新视频有实质行为收益，但Spatial64低于A2，Long仅held视频4/50，当前仍非稳定最终方法。
训练后半段四任务loss均低于原P/Q，task6/79迁移诊断仍负。完整证据为`runs/analysis/pi05_ecp_prw_complete_shared4_20260906/`。

因此下一候选复用完全相同Writer，fresh component initialization/optimizer，在既有55meta+18target allowlist上联合训练128updates；
每步73任务各8fresh跨episode action-query、等权1/73及原冻结normalizer，K1、每任务两fit视频，合计9344task exposures/74752action rows。
AdamW、LR端点1e-4→1e-6、warmup5、clip1、weight decay与原query/video/RNG规则不变，decay horizon扩至128。
每任务8个query改用实测可行的microbatch8一次计算：最长87-frame profile峰值reserved34.67GiB，VJP约1.20秒（micro2约1.35秒）；
相同初始query/noise的loss差约4.05e-5，属于已接受BF16/batch低位差异，不改变task权重或数据。
这是训练规模扩大而非纯task-count消融；不把expanded task mean和更长learning schedule的效果拆作单一因果结论。
128是看到32→64仍有实质增益后的下一预登记预算，不是收敛声明；同拓扑fully-random Final候选继续保留。

Checkpoint固定32/64/96/128，相邻pair固定(32,64)/(64,96)/(96,128)。先完成每point primary正确视频的validation8 screen80，
只分配后续投入，不以screen选择最终checkpoint或线性外推400分数。若同一任务/套件集合有广泛、有保留的能力并接近历史SFT在相同
80-state前缀上的表现，先完成一组strict400检验规模；有希望再补预登记相邻和另一正视频。最终资格仍完全使用上面的strict400合同。
原有两个positive video ordinals及no-selection negative wall不变；没有新增held gradients或Test使用。

主配置为`configs/pi05_ecp_prw_complete_meta73_v1.json`，两positive物化配置为对应`...validation_correct_v1.json`与
`...validation_other_correct_v1.json`。已有87-frame最长视频位于原四任务且已完成真实gradient/profile；额外full73四卡两步profile已exit0，micro2每步34.27/32.73秒，146video缓存25.43GiB；micro8的单task最长样本验证也exit0。
按实时可用1–6卡启动，四卡训练预估60–75分钟，另计capture/初始化/Panel-B约10–15分钟；实际时间以formal记录为准。
预计新增磁盘峰值<36GiB，正式从clean pushed detached tree运行，exact resume锁实际world topology。


## Registered target18 objective counterfactual (2026-09-06)

Meta73的128updates与四组screen80全部完成，结果15/19/19/19，后三点18/19成功集中在Object1/Goal6；没有支持扩strict400的广泛能力。
训练侧6个meta诊断持续改善，而target72/93/94较弱，且原四任务同query loss高于短四任务。它们指向值得区分的任务目标混合取舍，
不是已确认的梯度冲突或数据比例根因。原mixed run保留，不续训；下一对照继续使用完全相同完整LoRA Writer。

唯一主要干预为移除55个meta任务的functional objectives。保留原18个target IDs、原两条fit视频、原Panel-A跨episode query规则、
K1/micro8、source/normalizer、component initialization、fresh optimizer、LR和128-step schedule。每步18targets各权重1/18，
共2304task executions/18432action rows，每target128exposures/1024queries；checkpoint及相邻pair仍32/64/96/128和(32,64)/(64,96)/(96,128)。
这同时改变target在总梯度中的贡献和共享梯度统计；不能单独归因为梯度抵消、梯度量级、数据多样性或表示容量。
不添加剩余6个train targets，避免把objective混合与task覆盖两项变化混在一起；没有新增任务、视频或held梯度。

诊断继续使用原13task及全部原video/Panel-B visits。原6个gradient meta和meta6在此run均零梯度，并明确是被历史读取过的诊断，
不称fresh task-held selector；target79仍零梯度。配置把7个meta登记为本run的held组，只改变梯度角色，不改变诊断样本。

解释边界：target训练与新视频功能/闭环恢复，才支持混合目标在当前图上造成了重要取舍；若target功能恢复而未见target仍弱，
则目标域学习与task transfer必须分开，不把它包装为完整修复。若target学习也未改善，不能靠继续扫描角色比例维持该解释。
历史§141 role-equal在旧factor-set-relative-gain图中失败，当前是完整输出重构后的边界反事实，不是未尝试的新思路。

仍先做相同四点primary validation screen80，只分配投入；有实际强趋势后才一次strict400，再补相邻/另一正视频。
最终数值资格、same-topology random候选、冻结后因果controls和32/8最终合同完全不变。此对照不是初始化或架构消融。

配置为`configs/pi05_ecp_prw_complete_target18_v1.json`及对应两validation正视频配置。复用刚完成meta73的同图、同18targets、
最长87-frame与micro8真实执行证据，CPU核对空meta组与全18target schedule；没有新张量轴、算子、最长输入或更高显存要求。
五卡预计train约12–22分钟，另计启动/capture及相同13task诊断约10分钟；实际以新run记录为准。新增峰值<16GiB，
正式从新clean pushed detached tree运行，exact resume锁实际launch topology。配置与元数据以外的src/scripts/tests保持不变。


## Matched terminal training-task behavior diagnostic (2026-09-06)

Target18完成后，五个gradient targets的fit与新视频功能收益在64/96/128均为正，128均高于meta73；meta7诊断则全部变负。
这确认内部功能学习的目标取舍，但尚未确认目标任务自身的行为是否恢复。为把监督任务学习与未见任务迁移分开，
在读取本轮validation screen结果前登记两条已完成run的固定terminal128闭环对照；不以validation或functional最优点挑选checkpoint。

复用原Spatial2、Goal20、Long38，global IDs[2,20,38]、authority[72,83,93]及原50个states；两run分别用原first-fit视频[3,5,2]
和原同task held视频[49,49,48]，共四套single-checkpoint strict150/600新rows。部署图、source、task/state/env及policy RNG配对，
唯一因果干预仍是meta73→target18的objective组成与梯度统计；world5/6低位差异按原数值政策接受。旧short4 m64只作不同任务数/预算的参照。

若两正视频下的训练任务行为恢复而validation仍窄，才把当前瓶颈进一步限定为task transfer/覆盖；若只有functional改善，
下一分析聚焦离线功能与真实行为的分离，不能立即断定occupancy。仍保留原§6/§8及SEOD/GOMQ等历史边界。
不从这些train-side panels选择最终checkpoint，不读取负视频或Test，不改变任何训练配置、权重或checkpoint；本诊断只生成完整LoRA和评测。
相同graph/inputs已真实验证，无新增profile；额外磁盘峰值<1GiB，沿用已核验quota预算，实际GPU按launch时两节点live证据安排。
配置为`pi05_ecp_prw_complete_{meta73,target18}_m128_train_{fit,held}_eval_v1.json`，复用原train-side task subset与canonical evaluator。


## Registered terminal training breadth completion (2026-09-06)

原三任务的严格行为与阶段诊断已完成，但Long38直接专家仅1–5/50，不能单独代表整套共享Writer。为区分当前目标学习是否广泛，
固定两run terminal128和原同task held视频，将剩余15个gradient target全部纳入每task原states0–9的screen，不按当前性能选择或剔除任务。
新增global[4,5,7,12,14,16,19,21,22,28,29,34,35,37,39]，两arms共300新rows；原global[2,20,38]复用已完成strict150各自前10state，
分别组成18tasks/180rows。原3和新增15必须同step、视频ordinal、source与task/state/env/policy RNG；逐task、suite、breadth和paired R/G/L。

原15 task的held demos由既有program-video split固定，无任何新视频选择；所有18 tasks都在两run授权gradient集合，且本轮不产生梯度。
这组training breadth仅用于区分已经学习的任务行为与未见任务迁移，既不选择最终checkpoint，也不改变原validation投入裁决。
保留直接rank16 task experts作不同训练预算的容量参照；不把其分数当Writer性能，不读取Test或negative controls。

配置为`pi05_ecp_prw_complete_training_breadth_subset_v1.json`及两`pi05_ecp_prw_complete_{meta73,target18}_m128_training_breadth_eval_v1.json`。
复用canonical materializer与dynamic evaluator、原完整图及两run；额外峰值<2GiB，formal artifact执行从clean pushed detached source，GPU按live余量。


## Registered matched single-task fitting diagnostic (2026-09-06)

完整训练breadth已封存：target18为55/180、Object5/40、Spatial15/40，同面板独立专家为111/180、29/40、31/40。目标任务本身仍有明显
学习缺口，不能仅以任务迁移或缺少6个训练任务解释。按专家原文§8的同图分别学/共同学原则，固定选择Spatial7/task75与Object2/task77，
其target18 held各2/10与1/10，而独立专家各10/10与8/10；选择是基于已读训练侧缺口，不称outcome-independent或held selector。
此前complete short4没有Object任务，已有正证据不能替代这两项的whole-Writer独立学习对照。

每task一个独立clone，仅作训练侧诊断：复用完全相同完整38-target rank16、P/Q、全部可训练模块与component初始化，不添加task query，
不冻结evidence。每个clone128updates、8fresh跨episode queries/update、原两fit视频（75为1/2；77为7/10），每视频64exposures；
与target18的per-task occurrence、query、policy RNG、normalizer及AdamW/LR schedule逐项匹配。唯一主要干预是全18task均值变为单task目标；
它改变共享梯度统计，不能独立区分容量、干扰与优化。Checkpoint仍32/64/96/128，但闭环固定terminal128，不从内部loss选点。
复用target79作为已读取的零梯度内部诊断；validation/test完全不使用。Panel-B仍无梯度；诊断范围只保留各clone与79以减少无关计算。

先对两个clone的原held视频48、原states0–9运行20新rows，与已完成target18各十行严格配对，只作定位和投入判断。若有明显独立学习
增量，可补相同task/video完整50states确认；若没有，则不能继续把全局共享冲突当作充分解释，也不能把128步等价于函数类收敛。
不部署clone集合，不将task ID或独立专家变成Writer输入；不读取negative/Test，不改最终strict400/同图random/因果controls合同。

配置为`pi05_ecp_prw_complete_single_task{75,77}_v1.json`、相应subset及held eval。复用已验证同图与原输入，单GPU分别训练两个clone、
总量2GPU可并发；不需要NCCL通信，每进程按实际GPU本地NUMA绑定。预计每clone启动/训练/诊断合计约6–10分钟，新增峰值合计<4GiB。
CPU核对真实config/panels/video split及128步采样后，从clean pushed detached提交运行。训练图/source/scripts不变；evaluator仅扩展显式
training_task_fitting_diagnostic登记，允许如实保留outcome-informed任务选择且禁止checkpoint selection，validation/test边界不变；有针对性回归验证。


## Registered shared18 learning trajectory on the same two tasks (2026-09-06)

前述两个whole-Writer clones已完成，固定terminal128 held-video screen合计14/20，对原shared18的3/20保留3、新增11、丢失0。
同图、同每task数据的单独学习成立，但clone可能学到近似固定LoRA，不证明跨task条件表示或视频必要性。下一未知是shared18一直没学起
还是曾经学会后丢失；两者需要不同训练干预，不能由terminal或loss判断。

不新增训练或改权重，只读取原target18 run的全部早期已存checkpoints32/64/96。每点对相同Spatial7/Object2、held demos48/48和原states0–9
做20-row screen，共60新rows，128复用已完成breadth中的20行；不选择最好checkpoint，不部署clone字典，不消费validation/Test或negative。
任务选择仍是显式outcome-informed training fitting diagnostic，fixed state/video选择不依赖新结果。逐task及合并报告每点success、相邻R/G/L、
churn与Jaccard，并和固定clones128比较。全部早期点均弱才进一步限定欠拟合；若存在可重复获得后丢失的行为，则保留具体集合证据后研究保持。
小面板只用于定位与投入；单点小差异不够命名根因，仍不允许依据它修改全局strict400选择合同。

配置为`pi05_ecp_prw_complete_shared_trajectory_{subset,held_eval}_v1.json`。源训练与Writer图不变，复用canonical materializer/evaluator；
额外峰值<1GiB，沿用本轮已核验4GiB资源预算。一次物化runtime生成三点banks，再用实际空闲GPU执行三个独立队列；总量<=6、每job单节点，
启动前两节点live准入，formal artifacts来自新clean pushed detached提交。无需新GPU profile或模型测试，验证两个真实config/subset loaders即可。


## Registered same-topology fully-random target18 candidate (2026-09-06)

同图单任务学习task75/77已达6/10、8/10，共享18对应完整32/64/96/128曲线仅2/3/4/3（各20）。这两个任务上没有强阶段后崩溃，
共同条件学习仍是主要缺口。当前完整图已有局部真实行为正证据，因此落实owner及专家已要求保留的same-topology fully-random端到端候选。
这检验部分G2组件继承对联合学习的影响，不预先宣称G2初始化是根因，也不恢复此前被中断的旧A2 random或更改主干结构。

唯一主要改变是跳过Writer的G2 component参数复制，使用现有`--initialization random`的标准随机初始化；全部4750208个可训练Writer参数
端到端学习，公开A模板/B零的functional identity起点保留。冻结source/observer和原完整38-target rank16图不变，没有独立carrier或task query。
原18targets、7meta+target79零梯度诊断、K1、两fit videos、Panel-A跨episode query、frozen normalizers、sampling seed、micro8、AdamW/LR及
128updates全部与component target18一致。每步18task各1/18，共2304task executions/18432queries；每task1024queries、每fit video64exposures。
这是初始化比较，不是seed/LR/rank扫描；实际query/video/RNG按原日志核对。128是匹配预算，不是充分收敛声明。

保存32/64/96/128，沿用相邻pairs与原两个validation正视频ordinals。先读取四个primary validation screen80，只分配后续投入；有广泛且
可保留的能力并接近同prefix SFT24时，再按原合同扩single-checkpoint strict400、相邻与same-task-other，不以screen选最终模型。
另外固定terminal128、同task75/77 held48/48和原十状态，做20-row training fitting诊断，与component完整曲线和已完成clones比较；它不参与
全局checkpoint选择。若随机初始化改善目标拟合及held能力，保留其实际适用范围；若同样弱，则初始化不能单独解释当前共同学习缺口。
全部negative与Test墙、唯一LoRA、最终稳定/breadth/视频因果及32+8合同不变。

配置为`pi05_ecp_prw_complete_target18_random_v1.json`、对应两validation正视频与training diagnostic eval配置。源码与算子/张量轴不变，
复用已验证最长87-frame/micro8和原target18的34.61GiB峰值证据；只核对实际config/schedule及启动后的finite/正确参数角色，不重复GPU profile。
按两节点live状态选择一个节点的1–6张可用/可共驻卡，锁实际world/topology，保持NCCL_P2P_DISABLE=1、GPU-local NUMA和deferred NCCL。
按原6卡实测预估训练约10–15分钟、加载/capture/Panel-B另约10分钟，另计按信息分配的评测；新增磁盘峰值<16GiB，launch前重新核验独立quota。

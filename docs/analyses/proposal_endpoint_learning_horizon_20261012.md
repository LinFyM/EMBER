# 同一端点图的有限学习时长检验：216 → 864

2026-10-12，main 科学决定。Owner 已授权硬边界内自主推进。原216有限批已经完整结束；
本文明确登记一个新批次，不把恢复资产当成旧批自动续跑授权。canonical/Git仍由实验独占。

## 已消费的事实与更新后的判断

main直接读了当前端点模型、Reader、训练与恢复消费者，完整结果报告、实际端点算术脚本/JSON，
以及两rank原始训练metrics；不是只据交付摘要作判断。没有新GPU、model forward或episode。
原件ROOT为 `/data1/user/ymdai/ember_runs/video_guided_proposal_writer_20261012/deterministic_endpoint_query16`。

同576条件 MT340/G216341/E455，G保留314、获得27、丢失26；train24净−2、support12净+3。
对E独有134成功条件只接住15，旧normalized为22；对双方成功321保住301，旧版294。
因此较旧版的变化主要是更接近MT保持，尚未形成所需控制增益；Spatial−8、Goal−4不可抹去。
task13的E就是MT却G由7到11，提醒我们：更接近某个参数标签与实际收益不是等价命题。

现有保存端点给出准确的误差分解。令 g_t/e_t 是同一固定S_G下的输出/目标，横线为36任务等权均值：

    mean ||g_t-e_t||² = ||g_bar-e_bar||² + mean ||(g_t-g_bar)-(e_t-e_bar)||²

MT基线1分为目标公共均值能量0.186116与任务变化0.813884。
G216的0.961878约分为公共均值误差0.149417与中心化误差0.812460；
总改善0.038122中约0.036699来自公共项，0.001424来自条件项。
G的公共输出占自身编辑能量97.677%，首5动作距MT .003809、距E .088033。
这降低了“当前有限学习已建立条件控制映射”的支持；不等于唯一证明视频无信息、编码器失效或参数回归普遍不可能。
没有部署任何参数均值，也没有根据这些内部指标挑模型。

但完整学习曲线还提供了此前终点摘要没有充分进入决策的证据：

| 完整36任务块结束更新 | endpoint MSE | E=MT九任务MSE | 非零E任务相对MT的MSE差 |
|---|---:|---:|---:|
| 9 | .999674 | .000166 | −.000490 |
| 54 | .995915 | .002118 | −.006152 |
| 108 | .986374 | .008160 | −.020888 |
| 162 | .972518 | .016165 | −.042031 |
| 198 | .966249 | .023556 | −.052853 |
| 207 | .964239 | .023742 | −.055595 |
| 216 | .962570 | .021115 | −.056946 |

数据来自ROOT/metrics/G_rank_{0,1}.jsonl，按9个update合并36条实际event等权计算。
216更新只有每task24次条件，原生meta/Reader/decoder均从fresh开始。
零末层第一步使上游梯度为零；同Gemma meta组的原始梯度均值从前几个块约1e−6，
到末三块约2.14e−4、4.30e−4、2.57e−3；action组也有实际非零信用。
梯度增长不是语义学习或有益方向证明，loss下降也主要可能继续由公共项解释。
它们只表明这次不再是前两版近零场的平曲线，也没有充分收敛的证据。

## 为什么先检验时长，暂不再改图

在当前生成器中，末层为 g_b = W_out LN(b) + bias，所有坐标块共享W_out。
MT parent的归一化输入p为零，块特征先由target/rank/side/block地址及Reader memory形成。
零初始化时，第一步只训练W_out/bias；Reader和meta的信用含W_out的转置，随后才打开。
若不同条件的块特征仍相近，末层梯度中共同标签与共同特征的外积会主导，
条件差异则依赖目标和特征的协方差。该分解解释一种可检验的早期共模学习机制，
不证明当前网络一定会在更多步数后克服它，也不把梯度大小当作条件协方差已学成。

另一方面，当前连最佳常量端点的参数误差0.813884都还未达到；未解释误差同时含
公共坐标模式拟合与条件映射两部分。此时立即同时换decoder或功能目标，会让有限学习量继续成为竞争解释。
实际曲线支持给同一计算图一次较长但有限的学习机会，不支持只凭末段下降无限续训。

这是对上一份端点决定§6“只学公共方向后默认改机制”的显式修正：216是早期行为节点，
不应被我未经学习曲线校准就当成充分学习后的架构裁决点。原216负结果仍完整保留，
不会改成正例、工程故障或声称本次一定能成功。学习时长是本新批唯一主要科学变量。

最近似旧rank16原始因子MSE+方向/尺度监督、地址修复后的48→75/400及公共方向收缩仍构成反证；
新鲜native条件、MT基底及更长训练都不能重置它。旧直接/功能写入也已有未保持的结果，
不能把“改成功能loss”作为无论何种结果都自动有效的出口。
本批终点后若仍公共拟合主导且控制无净增益，停止继续延长当前条件编码+全坐标端点回归组合，
不以换LR/rank/seed、再延长一点或新增归一化保护该假设。后继需重新论证映射/功能学习机制。

## 唯一新增执行合同

1. 从完整v7 G216检查点恢复到总update864，新增648更新、2592条件；
   每task新增72次、累计96次。保留原ψ、双meta、Reader、static、未训练θ、optimizer、
   sampler/cursor、rank RNG、version216及监督版本0；不是fresh重跑，也不丢弃前216更新。
2. 模型计算、零head既有状态、源策略、P0、S_G、共同E、36 allowlist、exact语言、
   action-hidden视频池/抽样、MT parent及其真实bootstrap H、loss、optimizer、clip、logical B4均不变。
   仍由当前唯一EndpointTrainer更新ψ，原生policy十步flow与RNG保持；不加loss/额外功能query。
3. **监督bank仍严格只有MT parent/H。** 不因旧ROOT里已存在G216学生文件就自动加入学生P/H，
   不读取G_student_bank，不把query结果写入H。必须在新root的实际refresh_bank消费者承接这一点。
   此约束用于隔离训练时长，不能宣称这批已学会任意parent反馈修订。
4. 只登记864为新科学读出点；可在432/648保存完整恢复CP，二者仅作恢复，不生成学生、
   不看闭环/视频controls、不挑checkpoint。864为事前指定出口，不与216按成绩择优交付。
5. 864后沿原36条teacher_demo/MT-H/P/实践初态/env及native policy RNG映射，
   每task一次确定G生成、一次真实实践，再将该固定LoRA做同query16，共新增612 episodes。
   与MT、G216、E形成四臂576严格配对；三套已有行兼容即引用，不改seed谋求复用。
   E只读解释，不入条件、候选、H、信用或梯度；9个E=MT保留同源引用。
6. 同14/16/16/4角色池、同72 teacher-selection重合行/504其余行分列，不选state或video。
   报告所有per-task/per-suite、train24/support12、breadth、R/G/L、churn、success overlap、
   E-only/MT-only/both-success/both-fail，与216的能力保持及得失。
   864的36套保存端点只做一次已有CPU误差分解；复用已有首5动作记录，不增model forward探针。
7. 仅此恢复学习+全36首学生读出，不自动Actor/warm/RL/held400/Teacher/controls/额外probe。
   到864与读出完成后结束新批，main消费结果自主接续，不另等Owner许可。

新ROOT建议：
`/data1/user/ymdai/ember_runs/video_guided_proposal_writer_20261012/deterministic_endpoint_learning864_query16`。
保留原216 run_contract/原件不变；新root登记parent CP、旧训练身份48fa1247、新读取/续训代码身份、
完整复用provenance和监督bank版本。复用canonical policy/data/E/SG/P0/H，不复制大模型/数据。
避免新root构造器把这次误写为fresh、把旧CP路径之外的模型状态当成恢复来源。

## 结果分支与停止线

- 若参数继续下降但中心化条件项仍几乎未改善、闭环仍近MT或更差：支持当前优化主要走公共方向；
  本次时长解释未救回完整假设，停止再延长此组合。即使loss仍有下降也不自动追加训练。
- 若条件映射确实学得更强而闭环不改善：参数目标/解码的功能传递不足更值得优先处理，
  不能用MSE或参数多样性宣布进步，也不能直接唯一归咎参数规范自由度。
- 若闭环有实际净增益且覆盖/保持合理：支持同一图有限学习量是此前重要限制；
  之后完成固定LoRA的held400、相邻稳定性和必要视频证据仍须登记，不能用train576代替它们。
- 若只在已见任务学到差异，不证明语言之外的视频/H价值或held迁移。
  216→864是真实同一训练链的变化，但视频/内部表示用途仍需有针对性的因果证据。

## 资源、时长与职责

基于前批两rank训练216用1.113h、读出.955h，新增648在两rank约3.34h，加同规模读出约.95h，
合计约4.3h；考虑2–4张实际有效A40、冷启动/准入/调度，暂报2.5–5.5h，工程接续另约20–40min。
这些是实测外推，不是已经测过864。若只能单卡，按实际预计延长并报告，不因GPUh无硬限就填卡。
常规项目物理6/8卡与单node6卡上限保持；两个节点现场准入、data1独立quota/共享容量仍由实验负责。
新root按16GiB增量预估，纳入恢复CP与新学生；没有大模型复制或新数据缓存矩阵。

物理world size不是科学变量或固定两卡限制。实验可在完整CP216边界选择实际有收益的拓扑，
正确承接optimizer/sampler/有效batch/任务权重与已有RNG，记录迁移及新rank RNG来源；
若迁移则不称bitwise exact，不绕过现有恢复校验。逻辑B4通常至多4个有用训练rank；
这不是整个训练/评测批次的新并发上限。相同吞吐布局已有最长视频profile可复用；
实测余量/设备改变时，只做必要的物理吞吐比较，profile更新不得污染CP216恢复状态。

工程实现、针对性实际消费者检查、Git集成push、clean detached、现场资源和运行恢复由实验闭环；
main不重复工程ACK、测试或GPU准入。代码若只改合同/续训/复用消费者，不顺手改模型。
将本完整决定和实际原件推理整合进现有analyses/唯一active design、findings/state/history，
把旧“216后默认改机制”修正明确写出，不回写旧formal结果、不改AGENTS动态年表。
只发一次实际启动身份/ETA（若有实质变动）及一条可存活的整批完成/真实异常回报，
不加阶段Queue、心跳、第二等待者或无人执行的计划状态。

## 工程承接

沿用 run.prepare / EndpointTrainer / shared_learning_job / stage_jobs 的唯一运行面，未新增模型或并行runner。
prepare --reuse-root 原216ROOT --resume 原CP216 --stop864登记继续学习身份；updates.G只有864，
G_checkpoint_updates为432/648/864。新root的G_bank_mode=MT_only在refresh_bank构建原MT/H后直接返回，
监督version非0拒绝，既不检查学生parent也不读取G_student_bank。
共享job首次明确引用外部CP216，恢复时优先新root内最高完整恢复点；训练端拒绝本批fresh起步，
核实际version/cursor并保存每rank恢复消费者收据。原模型、loss、标签、学习率与logical B4没有改变。
55项针对性检查通过（14.28秒），含已有学生文件无法污染MT-only、恢复点与资格点分离、v7继续身份。
实际硬件启动仍须clean pushed detached与现场准入；正式身份、资源和启动预测以progress及ROOT原收据为准。

实际启动：b1a21f70 clean pushed detached、原完整v7 CP216继续、同world2/gpu01:0/6/chunk16；
单一37-job有限依赖图。真实原件准入确认optimizer485/step216/RNG2/bank0及三套历史576兼容。
G退出后一次现场判定第三读出卡；所有job仍逐次live双节点GPU/独立quota准入。
预测新增G3.34h+读出.6–1.1h；原root/CP216与新冻结树保留，开发分支/worktree已安全退休。
准确命令、临时缓存路径、时间与回执存于新ROOT/coordinator/deterministic_G864_students_query16_01；
上述为实际启动时的记录。864现已完成，完整结果见[整批分析](proposal_endpoint_learning864_query16_20261012.md)：
参数MSE .961878→.843335，条件增益 .001424→.054377，闭环G216341→G864341/576，MT340/E455。
有限时长改变了参数条件学习，未救回完整控制假设；按原合同停止延长和自动后段。当前状态只看progress。

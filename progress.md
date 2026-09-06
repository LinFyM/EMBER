# EMBER progress

更新时间：2026-09-07 CST，新图真实Meta/VJP与最长K成本已验证，formal short4已完成48步，两组闭环比较进行中。

## 当前授权与方法状态

- Owner于9月7日明确授权：充分理解仓库后，连续自主开展实现、验证、训练、评测、诊断及证据支持的修正；
  进度说明不是审批节点。此前“未授权科研执行”“先报告再等待同意”等交接限制已被覆盖，跨上下文不得据此误停。
- 持续授权覆盖现有科学目标、信息墙、数据、资源与Git合同内的常规决策；不联系外部专家。
  只有改变科学目标/信息墙、未授权数据资源、不可自行裁决的重大投入分歧或越权破坏性操作才需owner决定。
- 已登记active design：[layered_relation_video_writer_design.md](docs/layered_relation_video_writer_design.md)，
  **唯一新图已实现并通过真实机制验证；第16步训练侧闭环未见广泛改善**。唯一主线为Writer与读取侧Meta fresh端到端联合训练、fresh optimizer/scheduler，
  source基础权重冻结；不实施G1--G3冻结课程，不额外建立阶段初始化候选。LoRA合法identity初始化保持。
- 当前候选：冻结图文prefix、单固定probe、Action Expert共享观察Meta、18层×完整50H；局部帧对独立50×50关系，
  两端分别softmax；关系MLP消费内容、rho和signed gap；同步邻居聚合、4个radius4 blocks、H-read；
  置换不变多视频compiler、坐标MLP生成唯一38-target完整rank16 LoRA。首版真实K1/2/4。
- 环境长期goal已创建：实现并以学习/闭环推进稳定视频条件迁移，满足validation8 strict paired single-checkpoint >145/400、
  相邻稳定/低churn/高breadth/四suite及GoalLong/同task视频鲁棒性；选点冻结后完成因果controls，方法冻结后32/8 fresh最终训练与Test。
  实现完成、训练结束、单点高分均不代表goal完成；未自设token预算、总工期或总尝试数。
- 接手Git实测：main干净且为交接基线9ea2034037e5c70b514198a70910aac5c2fb18f5，与本地origin/main一致。
  指定当前文档、相关9月5日完整专家原文、旧账本§1–3/9–20/164–165/172–181及相关分析原件已读；代码和canonical资产已核对。HANDOFF已消费并删除，长期内容留在正式文档。
- 未选出selected checkpoint，未达成最终科学目标。不自动恢复旧width256闭环或其它旧待办。
  当前formal short4按预登记16/48/96节点推进；历史GPU/quota快照不构成实时准入，后续launch及大增长前重新核验。

## 当前formal运行

- Clean pushed commit：8d934408；detached frozen workspace：.codex/worktrees/layered-frozen-8d934408。
- Run：runs/outputs/layered_relation_short4_joint_8d934408_gpu01p235_20260907；gpu01 physical2/3/5、world3，deferred NCCL与NUMA。
- tmux：gpu01 / ember-layered-short4-s16；第16步已自然完成，train_s16.exit=0、总339.39秒、64conditions/1024queries，完整checkpoint约171MiB。日志train_s16.log、退出码train_s16.exit；
  exact命令、环境、live资源及预算保存在runs/analysis/layered_relation_writer_20260907/launch_short4.json与launch_short4_s16.sh。
- 物化/评测入口已集成并推送d5b8119e；最新全CPU171 passed/18.50s。执行frozen worktree为.codex/worktrees/layered-eval-d5b8119e。
- 第16步correct/other各4套完整LoRA物化exit0，通过真实manifest/episode inspector；correct demos为7:48,12:49,20:48,35:46，
  other为7:49,12:47,20:47,35:47，均由seed20260907的固定held池无放回抽样，未使用结果选择。
- 第16步三组screen40已完成：source用p2、correct/other用p3/p5；每GPU3 persistent workers，各自cost-balanced队列。
  结果根为runs/analysis/layered_relation_writer_20260907/{short4_source_screen40,s16_correct_screen40,s16_same_task_other_screen40}。
  第16步闭环三组均exit0：source4/40、correct4/40、other6/40；correct/source RGL4/0/0，other/source4/2/0；breadth均2/4，Goal/Object0。exact命令/资源/退出码在同analysis root。
- 第16步只有256queries/task；按事前96步短学习合同继续到48（768queries/task），保持原run参数、optimizer/scheduler、sampler与world3；gpu01 tmux ember-layered-short4-s48已自然完成exit0，日志train_s48.log。
  该续段545.03秒，累计192conditions/3072queries；各task K1/2/4均16次。macro_00000048完整状态已保存；两组同一held视频正在物化，随后固定screen40。
  原生frame_chunk16的exploratory K4 joint16.88s（observer forward1.36/VJP3.13）及peak allocated34.65GiB，表明批量布局可加快约34.5%；
  该profile用step16权重，早期chunk4 profile用smoke权重，不能将loss差当batch收益或科学改进。下次fresh扩大训练采用已测布局。
- 第16步后登记冻结训练侧functional诊断：action42–45各8query/task，与训练queries及held videos互斥；无梯度、固定noise/time、16/48/96重复同面板，不能选点。
  第16步source减correct loss在四task为+8.90e-5/+2.86e-5/+5.11e-5/-1.87e-5；整体变化很小，不能当行为改善。
  原件functional_panel_registration.json、functional_s16.json位于同analysis root。
- 已合并task worktree及codex/layered-writer分支清理；两个仍供formal训练/评测使用的detached frozen worktrees保留。

## 当前实现与验证（2026-09-07）

- 授权记录已提交推送70b194ec；纯Writer实现4fa19c3a已集成main。训练/读取/数据于8d934408集成，物化评测于d5b8119e集成；各自formal运行使用对应clean pushed frozen authority。
- 唯一实现owner：writer/relation.py负责局部帧对和同步邻居更新；layered.py负责语言/H-read/集合compiler；coordinate.py负责完整坐标A/B；
  native.py负责原生Meta读取与R-leaf/observer VJP；learning_data.py负责固定split与跨episode采样；runtime.py负责加载和有界冻结prefix缓存；
  training.py及薄CLI负责全局任务权重、调度、checkpoint/resume；materialization/evaluation负责逐episode条件物化及已有队列接线。没有恢复旧Writer/fallback。
- 这是退役后从空缺重建部署图及必要训练面，训练加物化/评测源代码增长约2k行；各模块职责独立，主文件均小于400行，复用现有LoRA、functional、
  source、checkpoint与队列。现有checkpoint函数的局部增长仅添加sampler状态，trainer.run作为单一生命周期编排保留，避免机械拆分。
- 首版Writer14,112,544参数，读取Meta626,688参数；两者直接fresh联合训练，source0 trainable。配置入口configs/pi05_layered_writer_v1.json。
- 训练侧短面板global7/12/20/35覆盖Spatial/Object/Goal/Long；Long35历史专家40/50（完整原件已核）。
  新采样定义video0–15、action16–41、diagnostic action42–45、held video46–49，互斥；每task真实K1/2/4轮换、独立无放回抽K组，
  query按episode再frame分层抽样。4task×3visits真实数据读取已验证，源normalizer冻结，梯度normalizer明确为1（未继承旧carrier task reweight）。
- 新跑全CPU suite153 passed/20.06s；后续checkpoint/sampler与相关检查26 passed/14.85s。纯CPU通过不代表真实功能或行为通过。
- gpu01p3（GPU-e59426ed-ed41-cb75-2190-f50841cff288）实际两帧native smoke：[2,18,50,1024]、finite、requires_grad；真实最长train视频为
  global38/demo36，raw517、stride5含尾帧105。此smoke只验证读取接口，未给出Meta functional梯度结论。
- 完整GPU首轮在第一次功能更新前暴露BF16消息与FP32 scatter buffer dtype不匹配，已以消息dtype分配修复；重跑已通过真实功能VJP与最长K1/K4。
  临时记录：.codex/tmp/layered/joint_mechanism_retry.log、joint_mechanism.jsonl；任务包含identity启动后的真实Meta梯度、full/staged VJP限定比较、
  最长真实K1/K4（38的36/41/28/35；query另取0–15）及真实16条action queries。只作工程机制/profile，不能选择checkpoint或声称科学收益。
- 实测strg01：data1约465.4GiB、data0约54.8GiB，分别soft1T/hard约1.01T；du workspace约434GiB，data1其它约32GiB，data0约55GiB。
  共享空间data1约84TiB、data0约1.9TiB；初期新增预算<5GiB（完整模型/optimizer checkpoints和小证据），复用全部大资产。
  prefix只做每rank2GiB有界CPU缓存，临时R每step失效。launch前已同时核验两节点，未干扰其它用户作业。
- 真实functional检查：identity第0步Meta A/B均0，第1步B非零，第2步A/B均非零；source无梯度。
  full/staged loss同为0.1337032914；抽查Meta0q-B/rho0/decoderB的cosine为0.9999877/0.9999966/1.0，相对误差0.00502/0.00260/0。
  这是BF16链式语义验证，不是训练能力结论。采样只使用授权train任务与跨episode真实action queries。
- 最长profile：K1[105frames] prefix3.60s、joint8.27s、peak allocated34.23GiB；K4[105,102,102,97] prefix13.37s、joint25.78s、
  peak allocated34.75GiB/reserved36.10GiB。K4分项observer3.83、Writer1.80、policyVJP2.26、WriterVJP8.18、observerVJP9.70秒。
  完整原件保留runs/analysis/layered_relation_writer_20260907/mechanism/，明示exploratory，不选模型。
- 四任务formal短学习预登记96updates，每task1536真实queries、K1/2/4各32条件，global4task等权；16/48/96固定checkpoint。
  各节点global7/12/20/35×states0–9×correct/other两组K1 held视频，seed20260907，固定每task无放回分组；仅判断基础行为/新视频泛化，
  不能选择最终checkpoint。未见基础行为则诊断最早接口，不默认长训；后续完整train24的strict400/邻接口径在读分数前另登记。
- Subagent在隔离worktree完成纯图后，已交付物化与原有评测队列接线，验证后已集成；主agent负责GPU机制、训练及科学决策。

## 已封存的最近科学结果

- 旧complete P/Q short4 m64 fit/held为64/62（各150）；mixed meta73四点validation screen80为15/19/19/19。
- 相同18target对照四点screen为17/17/20/16；terminal128训练侧held视频breadth为55/180、13/18tasks，对meta73的42/180、9/18。
  Object仅5/40；Goal/Long改善仍不足以证明共享与迁移问题解决。
- 相同预算task75/77 whole-Writer clones为14/20，对shared18的3/20；shared相邻四点为2/3/4/3。
  说明共享训练存在代价，但不能单凭该差距确定容量、梯度冲突或优化根因。
- 同图fully-random target18四点screen为16/16/17/19；固定训练两task4/20，未消除共同学习缺口。
- width256原run已自然结束：128updates、15,660,800参数、train.exit=0；训练732.48秒，functional Panel-B498.24秒，总计1283.05秒。
  32/64/96/128 checkpoint均保留；暂停后未启动物化或闭环评测，**没有width256闭环分数**。

完整原件、样本/预算口径与边界在 [历史§6](docs/research_history.md#recent-learning)；跨轮解释在 [findings.md](findings.md)。
上述都是旧图结果，新候选不继承其性能结论。

## 前次仓库整理

- 新设计记录涵盖科学动机、完整数学与因果推导、张量shape、多视频、单probe选择、X/Y与坐标decoder、GPU staged VJP/cache与现有代码迁移。
- 重写concept、长期要求、分层history、findings、README与当前账本；原6份旧设计、11份专家原文和181节完整旧账本保留在Git
  `fcdb6e43706c5fcedf10eaa5d2d459602b263016`，历史§9可按问题定位原件。
- 退役旧P/Q、bank conditioning、Natural Program、joint primal、G1/G3与Stage 0专用训练面及专用配置/测试；
  `src/scripts/tests/configs`文件共476→123（src195→69、scripts42→8、tests42→24、configs197→22）。
- 整理通用functional重放、panel读取、task/K调度与成本分配；source/data/expert/evaluation/normalization/LoRA基础保留。
  `configs/pi05_writer_data_v1.json`集中可复用来源，历史角色不是新实验授权。新图仍无可运行训练入口。
- 本次由设计记录、代码整理、存储审计三个subagents并行完成各自范围；通用委派规则随后按owner纠正统一维护在用户级AGENTS，删除项目内重复要求。
- 删除222组可重建派生缓存，共82,122个payload文件、245,347,917,824 allocated bytes（228.50 GiB）；
  保留所有cache/entry JSON、生成配方、run contracts、metrics、raw rows、正常化参数和唯一checkpoints。
  缺上游Writer的两个小smoke缓存、两个各约44GiB正式run root及独特轨迹证据保留。
- strg01清理后quota复查：data1 488,035,348 KiB（465.43 GiB），soft1,073,741,824 KiB、hard1,084,227,584 KiB；
  清理前727,683,088 KiB。data0 57,471,972 KiB，使用独立quota。此为当时快照，下次大增长前须现场复查。
- 精确删除范围、保留例外、重建依赖、width256完成核验与工作树清理记录位于
  `runs/analysis/ember_handoff_cleanup_20260906/storage_cleanup.json`。

## 前次整理的验证与交付

- 在集成后的主工作区新跑 `PYTHONPATH=src .venv/bin/python -m pytest -q`：139 passed，20.75秒。
- 6个保留Python CLI的`--help`全部exit0；当前Python源码语法、22个JSON配置与两份shell脚本语法检查通过。
- 当前Markdown本地文件链接无缺失，17份退出活动树的设计/评审原件仍可由冻结Git读取；`git diff --check`通过。
- 10个已完成工作树已移除；两个写入subagent的交付范围与main集成内容一致，task分支已删除。临时启动/诊断副本与Python/pytest缓存清理完成。
- 代码整理与设计记录已合入main；本文及其余文档随交接提交推送。最终提交与remote一致性以Git实际状态为准。
- 验证覆盖当前保留的工程基础；新架构GPU forward、Meta梯度、profile和闭环尚未运行，不能据此宣称新方法有效。

## 本次设计修订与交接

- 完整设计更名为layered_relation_video_writer_design.md，保留一个canonical方法文档；重写过程推导、shape、GPU布局/成本与验证定义。
  原past-only文本在Git 12d9689c及此前提交中保留，历史§7记录修正原因与边界。
- 同步concept、长期要求、findings、README、task_plan、分层历史和HANDOFF；内容差与对应位置变化均有明确消费者，
  允许双向教学帧读取，禁止重新使用旧的未来帧不变性测试。
- 已审阅9份文档的相关差异，核对55个本地文件引用均存在，原初稿Git引用可恢复；旧单向定义只保留在历史或明确的退役说明中。
  本次git diff --check通过；只更新文档，未重跑前次139项工程测试，未进行GPU/新模型验证。
  main交付随本次文档提交推送，Git实际状态为准。

## 下一步

读取第48步correct/other固定screen40和无梯度functional面板，与source及第16步严格配对比较；再按短学习96步合同与实际趋势继续或定位最早接口。
通过基础训练行为后再登记完整train24与strict400；不能把几何或loss代替闭环。
按task_plan持续执行，不因例行检查、阶段汇报或一次实验结束停止。

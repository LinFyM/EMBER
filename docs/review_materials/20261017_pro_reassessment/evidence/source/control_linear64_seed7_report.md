> 既有报告摘存；来源：`runs/source_policy_space_compiler_20261015/control_linear64_seed7/analysis/engineering_completion.md`。历史执行状态不构成当前授权。下文未入Git的原目录链接改为文字引用；可读结果投影见[证据地图](../../EVIDENCE_MAP.md)。

# §12一次线性64维基底与完整控制保存交付

唯一owner1006050/supervisor1005668于2026-10-15T22:24:15.315407+00:00真实exit0，连续退出reader session48175/cell346已结束。派发2026-10-15T22:05:12.992Z、准备22:05:18.222Z、owner启动22:12:41.893413Z；CPU拟合开始2026-10-15T22:12:48.263222+00:00，首环境GPU job 2026-10-15T22:13:08.056767+00:00。owner wall11.557033min、派发至退出19.038723min；比35–75min初估及25–45min修正快：工程准备约7.5min、CPU拟合仅6.63秒、实际闭环11.10min，且实际control少于旧1170。不把这些差异称纯调度因果收益。

|36×16原训练域配对面板|source|reference|native144|operator1170|linear64|
|---|---:|---:|---:|---:|---:|
|全36/576|218|502|216|224|484|
|target24/384|48|315|45|50|300|
|support12/192|170|187|171|174|184|
|非identity31/496|138|422|136|144|404|

相对source R212/G272/L6，churn278；相对reference R463/G21/L39，churn60；相对native144 R211/G273/L5，churn278；相对operator1170 R217/G267/L7，churn274。参考原新增288例保252（87.5%，native144保10、operator1170保13），target238/270、support14/18；source218成功保212。新端点36/36task有成功，29任务净优于source、27净优于1170。五identity42/55/64/96/101共80行仍source复用，不能称新学习。

各suite source/ref/144/1170/new：Spatial14/93/16/16/86，Object2/84/2/2/80，Goal26/74/22/26/72，Long6/64/5/6/62（各96），support170/187/171/174/184（192）。参考仍有39个成功丢失，source六个损失为task0 init10、34 init4、19 init10、35 init15、51 init9、56 init9。弱例task38为5/16（ref6）、34为6（ref7）、20为7（ref9）、29为7（ref7）；不能抹去。全部五方success sets、10种配对R/G/L/churn/overlap、task/suite/target-support-identity及完整36任务表见`report.md`（原运行目录：`report.md`）、`results.json`（原运行目录：`results.json`）。

同原八task旧MT128严格配对，仅为已有子集参照：linear98/128对MT67，R55/G43/L12/churn55，0新MT权重/请求。不能称全36或held超过MT，亦不是视频条件EMBER结果。当前输入仍是标签侧合法参考参数，且读出使用原reference选点初态面板；独立reference report545/648对source248属于另一面板，不混成此处分母。

## 一次拟合、数学消费者与有效边界

同原271不同参数点（266非零CP+5source）、原76 S、原144实际FactorLayout.center；全部643584有效坐标参与，512padding不进入拟合。M_j=sqrt(w_j)*valid((pack(Lambda_j)-center)/S)，task等权1/36，selected半权/other均分另一半；五source质量5/72保留为零，不重分配，不减bank均值，不做gauge/rank重排、成功率重权或selected专属残差。

实际唯一FP64小Gram谱分解数值秩266、有效64；rank规则固定lambda>271*eps_FP64*lambda_max，阈值1.19202922045e-08，没有容差/维数搜索。前64右向量保存FP32；FP64右向量正交最大绝对误差1.82484236055e-14。谱质量0.988270296101。计算5.69202136秒、原子basis写0.931465168秒、fit wall6.633568432秒；CPU16线程，分解阶段峰RSS4.986316681GiB，PSS阶段样本0.703→4.048→4.227GiB/16→31线程。诊断另有工作副本，未连续采样整个CPU阶段；launch按含矩阵/FP64/cache/投影副本估计8GiB与实际node可用约470GiB准入，不能把分解峰当全CPU连续峰。CPU fit 0GPUh。

唯一完整271池读回1.683045417秒，未新增其他probe或候选：加权因子残差7549.039889 / source643583.159057 = .0117297038973，与1−谱质量相符。沿原固定38 C_l的BA误差.0129933280967，selected .0025831507141、other .0234035054794，source baseline .999999994218。完整task/suite/38层及因子残差汇总在results.json.fit、原271rows在diagnostics/pool.json。它们仅定位，没有用作闭环门槛或选点。

编码严格z=U^T r，解码center+S*pad(Uz)，同一64维代码同时决定完整38/r8 A/B；D只收连续z，无参考旁路、nearest-neighbor、task字典、端点残差或teacher回退。一次拟合得到固定U，36selected各投影一次，最终policy仅source+一套已物化LoRA，basis/参考bank和编码辅助不在执行回路运行。仅mmap原144 CP提取saved center，未实例化旧神经模型/Adam；0神经/optimizer/scheduler更新，未虚构1171或exact-resume。

保存一个canonical basis/state.pt（167467157B）及591853B轻量decomposition恢复阶段，完整U/center/S/layout/bank/精确权重/谱/版本/资源/时钟/schema；原子完成点不重拟合，有限未完成阶段才接续。原144/1170全部CP与其失败、费用、时钟原样保留。此训练域正例说明当前固定64表示可保住大量参考控制，不证明视频识别/新任务迁移、64是内在维数或旧神经失败的唯一原因；同时改变了表示/拟合目标，不是attention单因素对照。

## 实际评估、资源释放与费用

31套真实LoRA496新episode全部完成，另5 identity80行source原执行复用；control94085+dummy4960=99045步，0正式GPU/环境失败/未知/有效行重跑。独立16工程额外额度未用，512/179840control/5120dummy/184960total上限不变，旧窗口余额不转入。0新RGB/H/动作轨迹/参考训练/query/视频/held/Test；原真实成功producer RGB仍在reference原件。

6 persistent worker按long-first动态队列、node级PSS/RAM准入和idle完整close，实际保留环境槽最大48；16/8/4槽shard为1/29/1。中位端到端plans/s6.104/6.064/3.106，native8.337/7.576/4.672；是不同任务与成功后动态缩批，非配对microbatch profile，不能由此称全局最优。沿用已实际比较的16/8/4并以node合计内存约束环境总量，不为GPU余量增加科学面板。

读出wall11.098447208min；GPU峰allocated11212.206MiB、reserved24184MiB。single-worker PSS采样最高35.769GiB/18process/52thread，host MemAvailable最低274.377GiB；异步worker采样不是同刻node峰。7份完整runner.close回执0残留，记录的78个owner/worker/环境PID全部消失，退出现场两节点0本用户CUDA/0control_space module；foreign进程未动。resource_exit.json/gpu_exit.json为原件。

本ROOT六个正式GPU job均exit0，总0.996805969742GPUh，含加载/闭环/关闭全部实际成本，CPU fit另计wall且0GPU。旧reference区间、native144与operator1170全部成本保持；四关联ROOT累计[56.797533485946,56.905278990946]GPUh，其中原六loader未决区间仍在，不记0，N16等更早独立成本不混入。没有GPUh硬cap。

ROOT退出实占270524416B（后增少量验收文档，非连续峰），新增磁盘峰预算4GiB；basis约160MiB/36输出约89MiB，原大source/data/reference/旧模型只读复用，0复制/下载。storage-node launch used1748301904KiB、exit1748566664KiB/soft2147483648KiB，shared约80TiB，独立quota和shared均核过，所有新增data1。

## 工程身份与交接

formal fit/readout统一clean pushed detachedc99502cf0478dd047f7e845dc49166b48d25c520，冻结树control_linear64_frozen_20261016。唯一control_space内的fit.py替换training.py；神经E/D、Adam/滚动updateCP主动路径退役，历史只留Git/frozen/原件。FactorLayout/References/pool/Gram/Batch/原source Runtime/动态队列/账本/关闭原语复用；公共FactorEncoder因Generator/Actor等真实消费者保留。旧144/1170只按明确schema验证原行，不加载成fallback。净−195 source行、0新package，guard无hard；事务/五方分析/完整pool的16–20复杂度保留清楚职责。

实际CPU小矩阵Gram/SVD、低/零秩零填、保存center/单点完整编码物化、五方raw/schema/预算/幂等消费者，以及正式一次fit/全271读回/576逻辑控制全部验收。acceptance.json/resource_exit.json为最终证据。仅一个owner/实际退出reader，无Queue/心跳/第二waiter。完整Git/push/clean与task-owned树清理见Git_delivery.json；一次交main01a13dd9后归还canonical写窗口。后继真实视频→z学习由main消费后登记，本批不自动第二fit、扫维数/单位、拼神经补偿或开q/Reader/RL/held/Test。

# 确定完整端点/query16：更接近MT，教师控制优势仍未传递

全36 fresh G216及首学生有限批已实际退出0。固定新G为341/576，强MT340，旧normalized G339，E455；
新G相对MT保留314、获得27、丢失26，净增1，未形成明显超过强MT的控制信号。
参数端点MSE从MT基线1下降到0.961878，但约97.68%的输出编辑能量属于36个条件的共同均值，
条件差异只贡献0.001424的MSE增益。去掉独立xi输运和零head初始化改善了近MT保持，尚未解决条件到有益参数的映射。
这是train24+经审计non-held90 support12的首学生机制读出；没有Actor学习、held400、相邻稳定性或视频/H因果资格。

ROOT：`/data1/user/ymdai/ember_runs/video_guided_proposal_writer_20261012/deterministic_endpoint_query16`。
冻结训练/学生消费者：`48fa1247db1845831ec9bac5afd865ad86cbd4eb`，clean pushed detached；
唯一37-job协调器：`ROOT/coordinator/deterministic_G216_students_query16_01/exit.json`。
唯一执行合同是[原active设计§12](../designs/video_guided_proposal_writer_20261010.md#12-当前有界批确定mt中心完整端点与同面板query162026-10-12)，
完整事前推导和历史反证是[端点参数化决定](proposal_endpoint_parameterization_20261012.md)。
本报告消费已执行有限批，不登记新的训练或后继合同。

## 实际算子、学习与部署

固定P0为MT合并冻结基底上的38-target/rank8完整任务LoRA，A0非零/B0=0，S_G来自既有等权36-task共同E。
真实exact language、同task action-hidden有序教学、原生双meta及ContextReader、当前P和指定自身MT bootstrap H保留。
六层归一化通信decoder仅读[p=(P−P0)/S_G,valid]及实际context，零初始化末Linear预测e_hat；
生成完整P_new=P0+S_G*e_hat，直接masked MSE监督(E−P0)/S_G。
参数xi/time/x_t/16Euler已经移除，原生policy十步flow、环境及policy RNG保持；不在输入中混入教师标签或正确轨迹。
当前只有MT parent/H的初始化学习，不能把这批端点参数看作已学成反馈修订。

实际v7 CP216完整保存G/meta/Reader/static、Actor、optimizer、sampler和rank RNG；optimizer485个状态均step216，
world2/logical B4/cursor216/RNG2，full38/rank8/643584有效坐标，supervision_version0。
每rank216日志、共864训练事件，每task24；parent均MT，H路径与bootstrap complete.history一致，标签及video采样合法，分母4。
source全程无梯度、实际chunk16；零输出头首步双meta梯度为0符合定义，随后430/432记录两组梯度非零。
Gemma/action原始组梯度均值约2.289e−4/1.039e−7；这些范数不是语义利用或某模块唯一失效的证据。
首三完整36-condition loss块0.999674/0.998947/0.998321，末三0.966249/0.964239/0.962570，对同标签MT基线约1。
不直接把新endpoint训练loss与旧CFM loss比较；保存候选对共同E的参数距离可在同一S_G下比较。

216后每task沿旧首学生相同teacher_demo/path_ordinal/MT-H、实践初态和RNG，只有一次实际确定G_000生成及实践，
没有按成绩重采或选择video；全部36有效。首学生实际G实践成功21/36，之后参数固定，query不反哺H或梯度。
G query16为该固定G_000，不是未训练Actor按uniform final抽出的最终完整EMBER成绩。
保持practice14/query16/teacher report16/train report4四池互斥。MT/E/旧G原行和真实episode/参数/CP来源只读引用；
E和旧G不进入编译条件、候选、Actor信用或H，E=MT九task复用同一MT行，旧Writer仅保留冻结原消费者解释。

## 配对控制、保持、覆盖

| 配对（左→右） | 左成功 | 右成功 | 保留 | 获得 | 丢失 | 净变 | churn | success Jaccard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MT→新G | 340 | 341 | 314 | 27 | 26 | +1 | 53 | 0.855586 |
| MT→E | 340 | 455 | 321 | 134 | 19 | +115 | 153 | 0.677215 |
| MT→旧G | 340 | 339 | 304 | 35 | 36 | -1 | 71 | 0.810667 |
| 旧G→新G | 339 | 341 | 307 | 34 | 32 | +2 | 66 | 0.823056 |
| E→新G | 455 | 341 | 316 | 25 | 139 | -114 | 164 | 0.658333 |
| 旧G→E | 339 | 455 | 316 | 139 | 23 | +116 | 162 | 0.661088 |

新G较旧G净增2，MT保持增加10、churn下降18，但新G在E独有成功134行只成功15（旧G22）；
MT独有19行成功13（旧G10），双方成功321行成功301（旧G294），双方失败102行成功12（旧G13）。
因此观察到的变化更接近减少近MT的损失，而不是已传递E提供的操作能力。
任务净优/平/劣为11/16/9；有任何成功的任务MT33、新G35、旧G34、E36，新G在MT零成功task17/38各有1/16，task29仍0。
每condition固定选一个MT或G的经验oracle为354/576，四臂固定condition oracle464；仅描述候选上限，不允许逐state合并成功集或据此选模型。

| suite | 行数 | MT | 旧G | 新G | E | 新G相对MT保留/获得/丢失 | 净变 |
|---|---:|---:|---:|---:|---:|---|---:|
| libero_spatial | 96 | 53 | 47 | 45 | 79 | 42/3/11 | -8 |
| libero_object | 96 | 34 | 33 | 41 | 68 | 32/9/2 | +7 |
| libero_goal | 96 | 49 | 46 | 45 | 64 | 43/2/6 | -4 |
| libero_10 | 96 | 33 | 37 | 36 | 61 | 28/8/5 | +3 |
| libero_90 | 192 | 171 | 176 | 174 | 183 | 169/5/2 | +3 |

目标benchmark的train24小计MT169/384→新G167（−2），support12 MT171/192→174（+3）；不是广泛任务控制提升。
Object +7中task13贡献+4，而其E正是MT；不能把此局部收益归为强E特有能力传递。Spatial−8和Goal−4完整保留。

| query子集 | 行数 | MT | 旧G | 新G | E | 新G相对MT保留/获得/丢失 | 净变 |
|---|---:|---:|---:|---:|---:|---|---:|
| 曾teacher选点 | 72 | 43 | 50 | 46 | 55 | 42/4/1 | +3 |
| 其余query | 504 | 297 | 289 | 295 | 400 | 272/23/25 | -2 |

72行每task两个state曾参与E选点，仅作train机制读出；其余504独立于该次选点但仍为共享训练task，不称held或独立teacher泛化。
总净增1由选点重叠组+3和其余组−2构成，不改变事前query面板或选择资格。

## 保存端点与现有初始动作：机制认识更新

唯一保存端点CPU算术来自`ROOT/analysis/endpoint_arithmetic/consume_endpoints.py`及`statistics.json`，执行1.310s，未重复。
新G对自身E的归一化MSE0.961878，较MT1改善3.812%；G编辑RMS均值0.168181，编辑均方0.028385。
G到公共教师均值MSE0.150077，公共教师均值自身对各E为0.813884；没有部署或评估均值LoRA。
新G输出均值对各E为0.963301，相对各condition输出只差0.001424。
共同输出均值能量0.027726，占全输出编辑能量97.677%；条件变化能量0.000659，目标任务变化能量0.813884，比例约0.0810%。
G跨task cosine0.978810，centered target cosine均值0.046775；条件差异带来的MSE增益只占相对MT全部改善的约3.734%。
这些数字支持公共方向占主导，不能把小幅MSE下降视为已建立task/video到有效完整LoRA的条件映射。

task13真实E=MT，其目标为零，但G仍输出RMS0.172414、MSE0.029726；task17非零E的MSE从1.703362降至1.644773，
centered target cosine0.034377，query仍仅1/16对E12/16。另8个E=MT task也有约0.146–0.168 RMS编辑。
这与公共方向跨task传播相符，但不唯一证明context表示缺失或某个具体算子有bug。

现有query原件已保存首5原生归一化动作，无新增forward：新G与MT的RMSE0.003809，旧G与MT0.006278，E与MT0.088126，
新G与E0.088033；在相同初态最早执行接口，输出仍接近MT，尚未形成E幅度的动作改变。
这些初始动作量级与E-only传递弱相互支持；它们不能单独解释后续53行闭环得失，微小早期动作也可能使轨迹分岔。
实际增益、损失及teacher类别分布仍是判断依据，不因动作距离小直接宣称部署完全恒等。

事前干预同时取消参数噪声输运、采用零head和直接端点监督；巨大参数距离改善不能全归为学习，更不能只归因旧时间条件。
相对两轮CFM，当前真更新已改善端点误差并减少MT损失，原广播量级修复和零head上游梯度开启也保留为真实机制事实。
但E455与MT340之间的显著操作优势仍未传递；仅去掉随机副任务未足以解决本轮映射问题。
这加强了旧直接因子回归§24–29的公共方向风险在当前native/MT/归一化条件下仍重要的判断；不把不同版本当相邻checkpoint，
也不从这些局部阴性推断LoRA编译路线普遍不可能。
条件信息是否足够、decoder是否真正利用条件、参数目标几何与有限训练共同作用仍未唯一分解；216更新没有证明优化充分收敛，Video/H因果尚无证据。
本次按登记停止，没有用负结果触发工程修补、追加探针或无限续训。后继科学由主讨论结合完整历史决定。

## 原件、费用、恢复能力与交付

全臂原行`ROOT/analysis/first_student_query16_rows.jsonl`为2304逻辑行、576四臂配对条件；
保存摘要`ROOT/analysis/first_student_query16_summary.json`。
独立CPU原件消费为`ROOT/analysis/whole_completion_20261012/consume_originals.py`及`statistics.json`，最终anomalies=[]，
保存各job收据身份、实际CP/训练、native初态/RNG配对、引用来源、per-task/suite及账目，未覆盖原始formal输出。
同目录raw_rows_verification.json另外确认保存JSONL的2304唯一臂/task/state行与实际episode来源、成功及native RNG一致。
首个消费断言误把复用H的local文件名当作原件路径，已依据bootstrap complete.history实际消费者修正；
该CPU首尝试/原因在同目录attempt_001保留，训练/科学合同与产物没有故障或改动，参数算术没有重跑。
整个消费未新增model forward、GPU、episode、梯度、工程测试或科学矩阵。

实际新增36 G实践/8564环境步（成功21/36），加576 G query/135108步，
总612 episode/143672环境步；包含原件environment_steps口径的settling，不重计复用MT bootstrap。
MT/E/旧G合计1728逻辑引用，对应1584不同历史episode（9task E=MT造成144同源）；历史费用不再加到新ROOT。
新ROOT allocation GPUh=4.121798102：G2.226971441、学生1.830137654、profile0.064689007，零GPU失败尝试。
G wall1.113h、首学生读出0.955h、协调器整批2.078h，
UTC2026-10-11T21:34:18.759936→23:38:59.602226，在1.5–3.5h预测范围内；各job准入/独立strg01 quota原件保留，ROOT实占约1.9GiB。
allocation是分配device-hours，不把共驻与独立阶段费用误称独占物理时长。profile顶层计一次，不重复加入其attempt子回执。

完整v7 CP216/initial_models、全部36学生参数/真实H、原query、固定S_G/P0及source/MT/E资产引用保留，
正式48fa冻结树继续保留以正确读取；CP含完整optimizer/sampler/两rank RNG，可在同schema/已登记合同内恢复，不能自动授权续训。
本轮两个开发分支/worktree和未运行a791冻结候选已退休，清单`ROOT/coordinator/deterministic_G216_students_query16_01/development_tree_retirement.json`。
没有为清理删除当前方法关键权重、完整恢复点、旧formal小原件或不明所有权内容。
原唯一整批退出投递已processing_verified，消费与Git集成后只向主讨论01a12bb0发送一次完整回报；
交付/判重以该coordinator/result_main_delivery.json和completion_consumed.json为准，不添加心跳、Queue或重复等待者。

## 全36任务原件读出

每task同16-state条件；ID为已审计allowlist global task ID。旧G为normalized-v6历史方法，新G为当前v7确定端点。

| task | suite | MT | 旧G | 新G | E | 新G相对MT保留/获得/丢失 | 净变 |
|---|---|---:|---:|---:|---:|---|---:|
| 0 | libero_spatial | 14 | 14 | 13 | 15 | 12/1/2 | -1 |
| 1 | libero_spatial | 4 | 3 | 2 | 15 | 2/0/2 | -2 |
| 2 | libero_spatial | 14 | 13 | 13 | 16 | 13/0/1 | -1 |
| 4 | libero_spatial | 8 | 6 | 5 | 7 | 5/0/3 | -3 |
| 5 | libero_spatial | 6 | 7 | 5 | 14 | 5/0/1 | -1 |
| 7 | libero_spatial | 7 | 4 | 7 | 12 | 5/2/2 | +0 |
| 12 | libero_object | 7 | 7 | 7 | 11 | 7/0/0 | +0 |
| 13 | libero_object | 7 | 8 | 11 | 7 | 7/4/0 | +4 |
| 14 | libero_object | 11 | 10 | 11 | 16 | 10/1/1 | +0 |
| 15 | libero_object | 3 | 3 | 4 | 9 | 2/2/1 | +1 |
| 17 | libero_object | 0 | 0 | 1 | 12 | 0/1/0 | +1 |
| 19 | libero_object | 6 | 5 | 7 | 13 | 6/1/0 | +1 |
| 20 | libero_goal | 8 | 6 | 6 | 8 | 6/0/2 | -2 |
| 21 | libero_goal | 14 | 13 | 14 | 16 | 14/0/0 | +0 |
| 22 | libero_goal | 10 | 10 | 9 | 15 | 9/0/1 | -1 |
| 25 | libero_goal | 2 | 1 | 1 | 6 | 0/1/2 | -1 |
| 28 | libero_goal | 15 | 16 | 15 | 16 | 14/1/1 | +0 |
| 29 | libero_goal | 0 | 0 | 0 | 3 | 0/0/0 | +0 |
| 32 | libero_10 | 2 | 5 | 3 | 13 | 2/1/0 | +1 |
| 34 | libero_10 | 11 | 10 | 11 | 10 | 10/1/1 | +0 |
| 35 | libero_10 | 12 | 12 | 12 | 12 | 11/1/1 | +0 |
| 36 | libero_10 | 1 | 3 | 1 | 12 | 0/1/1 | +0 |
| 37 | libero_10 | 7 | 6 | 8 | 12 | 5/3/2 | +1 |
| 38 | libero_10 | 0 | 1 | 1 | 2 | 0/1/0 | +1 |
| 42 | libero_90 | 16 | 16 | 16 | 16 | 16/0/0 | +0 |
| 43 | libero_90 | 7 | 11 | 8 | 15 | 7/1/0 | +1 |
| 51 | libero_90 | 14 | 14 | 15 | 16 | 14/1/0 | +1 |
| 55 | libero_90 | 16 | 16 | 15 | 16 | 15/0/1 | -1 |
| 56 | libero_90 | 13 | 15 | 14 | 13 | 13/1/0 | +1 |
| 62 | libero_90 | 14 | 15 | 14 | 16 | 14/0/0 | +0 |
| 64 | libero_90 | 16 | 16 | 16 | 16 | 16/0/0 | +0 |
| 73 | libero_90 | 14 | 13 | 14 | 14 | 13/1/1 | +0 |
| 95 | libero_90 | 13 | 13 | 14 | 13 | 13/1/0 | +1 |
| 96 | libero_90 | 16 | 16 | 16 | 16 | 16/0/0 | +0 |
| 97 | libero_90 | 16 | 16 | 16 | 16 | 16/0/0 | +0 |
| 101 | libero_90 | 16 | 15 | 16 | 16 | 16/0/0 | +0 |

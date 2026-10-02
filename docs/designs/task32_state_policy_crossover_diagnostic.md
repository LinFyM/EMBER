# Task32抓取失败：自身状态与冻结LoRA的交叉续行诊断

2026-10-02。固定数据内的一次分析实验，不是新架构、训练、选点或正式性能评测。
唯一执行范围由progress登记；已撤回的prefix变化训练不恢复。主讨论先消费本批原件，再决定是否有依据改变方法。

## 1. 具体失败及尚缺的联系

最近条件A重表达诊断的Original，来自条件读写900，在train task32
`turn on the stove and put the moka pot on it`上，两teacher×四初态只有1/8成功。
八条都已开炉并靠近壶；只有teacher17、state2成功。其余七条失败不能隐去。
这不支持把该task笼统解释成“没读出第二阶段”，也没有证明抓取失败由某个视频特征缺失导致。
本批使用完整Original LoRA，不使用Reexpressed或已撤回批次的profile权重。

在同一个state2、相同scene/env seed和共同policy噪声流下，原件为：

| 已存轨迹 | 开炉步 | 开炉后首次EEF距壶体中心小于10cm | 后段首次夹爪闭合命令的零基action index | 壶首次抬高3cm | 完整结果 |
| --- | ---: | ---: | ---: | ---: | --- |
| teacher17 | 89 | 178 | 193 | 210 | 第277步成功 |
| teacher43 | 91 | 182 | 193 | 无 | 520步失败，最大抬高约0.472cm |

第180步两轨迹EEF位置相差1.321cm、朝向相差3.198度；第190步仍相差0.944cm、2.938度。
这削弱了“没有开始抓取”或“仅夹爪闭合时刻不同”的解释，却不能由距离直接判定接触/抓取几何是根因。
壶体中心不等于把手或接触点；成功例的最小中心距离也有7.773cm，不能用5cm阈值定义是否尝试抓取。

当前缺口是：两视频生成的不同控制函数在抓取阶段是否仍决定成败，还是此前动作造成的实际状态差异，
在这两套控制函数下已主导结果。先分清这层条件生成映射与自身执行的联系，才有依据定位更具体的读写或学习缺陷。
该案例是从已有train诊断中选出的可区分正负例，不估计总体成功率，不解释全部held性能差距。

## 2. 干预的实际量与推导

原模型对每个target生成一套因子

```
A_l(v) = A0_l + S_l(v),       B_l(v) = B0_l + M_l(v; A_l(v)),
LoRA_l(h; v) = scale_l B_l(v) A_l(v) h.
```

S/M由action-hidden RGB、exact language及合法公共native读取编译；此次直接复用已存的完整38处因子。
自身执行hidden h则依赖实际双相机图像、8维机器人state、语言、当前flow latent以及前层计算。
两套LoRA都会改变整条非线性网络和十步flow，不能用单层因子差当作动作差或成功原因。

令i表示重放哪条**模型自身**的既存物理动作前缀，s_i为重放至第180步、尚未执行action[180]时的完整环境状态，
包括真实物体、机器人和控制器历史。令j表示续行使用哪套完整冻结LoRA，q(s_i)为此时的合法policy输入：

```
F_ij = OfficialTenStepPolicy(q(s_i), W_j, common_noise_at_step180),
Y_ij = closed_loop_outcome(s_i, W_j, common_future_noise, original_horizon520).
```

W17和W43各自都是原Writer一次编译的合法完整LoRA；不拼接层、不平均、不重新观看视频或优化权重。
干预i改变到达的实际状态；固定i比较j，才检验同一自身状态上的条件控制函数。
这不是只改EEF位置的反事实，也不能把s效应归给单独RGB或8维state。

对首次续行产生的动作数组，以下是有限差的精确代数分解，无需假定state-tokenization可微：

```
D   = F_43,43 - F_17,17
D_W = ((F_17,43 - F_17,17) + (F_43,43 - F_43,17)) / 2
D_s = ((F_43,17 - F_17,17) + (F_43,43 - F_17,43)) / 2
D   = D_W + D_s
I   = F_43,43 - F_43,17 - F_17,43 + F_17,17
```

保存原50×7 normalized动作及实际physical前5，分别报告translation3、rotation3、gripper1的上述差异。
它们描述状态、LoRA及交互怎样改变实际动作；范数更大不等于更有效，必须结合四条闭环、抬壶及抓取行为解释。
不把这一恒等式本身当作任何架构应被训练的理由。

## 3. 唯一四行合同与配对有效性

固定一个task、一个初态、一个切换点，四行是(i,j)=(17,17)、(17,43)、(43,17)、(43,43)。
每行从原封存初始scene按原消费者恢复，重放对应continuous原件的physical action[0:180]，
使控制器与物理状态自然演化；随后从step180的新replan开始，用j完成剩余任务。
重放的不是teacher action，不新增教学数据或训练标签。初始settling/scene恢复语义沿用原合同，切换处不再次settle。
不能用只覆盖sim_state/body pose的通用初始scene恢复函数冒充任意中途控制器状态克隆。

第180步在两个原轨迹的接近阶段、首次闭合193之前，且是官方五步replan边界；事前固定，不扫描更多切换点。
续行采用原teacher43结果中完整104项policy_noise_seeds对应的共同时间表；原17公共前缀已经核对相同。
重放不消耗policy采样，也不能把step180的噪声重置为第0次。执行仍为官方双256→224、10 flow steps、前5 actions，
成功即止，总horizon仍为520，不为切换额外延长任务。

两条对角续行是必要参照，须检查原成功/失败对比能否再现；四行均保存完整双相机、state、normalized chunks、
实际动作、goal谓词和连续EEF/朝向/夹爪/物体轨迹。记录重放前缀与原件在anchor处的状态/谓词误差，
以及同一i的两行是否确实到达相同状态。只做配对合同所需检查，不做全树hash、逐tensor或低位数值追查。
若重放合同有已定位工程违约，按既有权限修复并保留失败及成本；若合法执行未复现原对角对比，
如实交付“本次无法识别预定对比”，不搜索seed、teacher、切换点或dtype追回成功。
首个有效对角消费者兼作实际工程验证并计入四行，不额外增设闭环smoke矩阵。

## 4. 预注册的结果解释与历史边界

只有对角对比和状态配对有效时，才按以下方向解释：

- 若两种前缀下均是W17成功、W43失败，说明此案例的抓取阶段控制函数具有可转移的补救/破坏作用；
  后续应定位视频条件写出的实际算子怎样改变自身动作，不能仍把失败仅归给先前路径或缺少阶段输入。
- 若两个LoRA都在s17成功、都在s43失败，则差别主要随到达状态保留；
  “此刻缺少正确阶段Value”不足以解释该案例，继续追加同类Value模块的优先级降低。
  这不是证明s43不可救或控制函数无关，也不自动推出新feedback loss或状态补全架构。
- 若只有(17,17)成功而交叉皆失败，或两个交叉皆成功等混合结果，说明存在状态与条件控制的交互；
  不强行选成单一根因。抬壶、放置和最终成功须分开，以免把后续失效当成抓取未修复。
- 若动作存在明显条件差、闭环却没有相应区分，保留这种反例；内部差异不能替代有效控制证据。

原四初态/两teacher的1/8背景保持。本批只有一个物理初态的干预案例，不bootstrap、不报paired400资格，
不声称补齐视频必要性、held泛化、超过MT或稳定性证据；也不由局部成功自动触发fresh训练。
最近条件A重表达在task32仍为1/8，说明其函数保留没有修复这一现象；本批不重做解析C。
历史T2340的共享/私有/teacher私有64步诊断已经有局部可学修正及不利个例，不能把“可能修复”当新发现。
旧state-Jacobian/反馈损失论证也已指出通用局部稳定性不教会视频目标；本批没有重启那条路线。
本批结果只改变上述缺口的定位及候选优先级；仍须继续建立到具体特征/算子或学习条件的证据，不能绕过Owner要求§3。

## 5. 原件、资源及交付

所有路径前缀为`/data1/user/ymdai/ember_runs/`：

- 原bank：`conditional_A_reexpression_diagnostic_20261002/Original/bank/task032_teacher17.safetensors`及`task032_teacher43.safetensors`。
- 原结果/噪声/compact：同root的`Original/evaluation/teacher0/results.json`与`teacher1/results.json`中task32/state2。
- 前缀17：`conditional_A_reexpression_diagnostic_20261002/Original/evaluation/teacher0/continuous_traces/libero_10_task_02_state_002_4bedb89b08684c219d10c2f001b982ff.npz`。
- 前缀43：`conditional_A_reexpression_diagnostic_20261002/Original/evaluation/teacher1/continuous_traces/libero_10_task_02_state_002_4c6fcf4921534331ac9450729a3a6897.npz`。
- 原初始scene：`operator_learning_limit_diagnosis_20260930/scenes/libero_10_task_02_state_002.npz`。
- 唯一新root：`task32_state_policy_crossover_20261002`；source/normalization从原sealed spec精确复用。

预计含工程45–75分钟；GPU计算预期0.05–0.3GPUh，硬限**1GPUh、4GiB新增峰值**。
依据上一批64闭环加有限功能读取共0.282GPUh；本批仅四行、前180步不运行policy，但仍计入加载、验证和失败成本。
显著超出wall-clock预期或到0.8GPUh仍未齐时交代具体剩余量，不自行追加算力预算或诊断范围。
执行者负责双节点现场准入、data1独立quota/实占/共享容量与峰值、吞吐配置、针对实现检查和clean pushed frozen读取。
按四行真实负载选择GPU及并行数，复用原官方消费者；不为占满卡增加工作，也不为微小数值一致强制低效batch1。
无Writer训练/梯度/optimizer、无native重新编译、无新teacher/任务、无Val/Test/controls读取、无RL或其它后继。

主讨论负责本合同与科学裁决；唯一实验session负责工程、运行、代码/Git、费用、退出和整批交付。
交付四行原件、首次动作2×2表及分解、逐行目标/抬壶时刻、配对有效性、原件来源、completion/readback和GPU释放。
科研记录与工程Git窗口串行；本批完成后只发一次整批回报，不另排阶段自通知，不自动恢复任何训练。

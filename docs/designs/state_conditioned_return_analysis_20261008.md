# 已存自身轨迹中的状态条件回报：有限预测与局部信用辨识

2026-10-08；依据 findings§395–396。经 progress 登记后为唯一 active 分析。
本批只使用已有训练侧原件，训练小型 CPU 预测器；不更新 Writer/source，不采环境、不运行 VLA/GPU。
目的是决定状态价值信用是否值得下一笔实际策略实验投入，不把 critic 名称、预测或余切能量当成控制改善。

## 1. 未解决问题与为何先做这项分析

当前 source 的四独立专家480从初态12/16、原T50接手9/16；救回T六条失败，也丢三条成功，task29接手0/4。
task12同一权重从初态搬对沙拉酱，接手T后却搬奶油奶酪；task29两起点都搬瓶但未完成OnRack。
这些事实加强“控制的状态适用范围仍不足”，但没有区分表示、示范覆盖和优化，也未建立可直接蒸馏的可靠教师。
不能继续假设教师存在，更不能把下一步简称为“改用RL”：旧Gaussian/RB及当前T的72步SDE回报学习均有实际阴性。

最近72步确为完整共享G的真实score学习：288组中140 mixed、22全失败、126全成功；
每组四独立初态、同一合法教学，优势 `A_i=R_i-mean(R_other3)` 被广播到该episode全部抽中replan/denoising时点。
没有V/Q、TD或GAE。父/末seen SDE均109/144，ODE109→102，official400为161→154/158；不能归咎于只剩采样转换问题。
task12/29/32/38的训练成功分别21/9/16/1（各32），task25为0；已有正回报并非普遍缺失。
8月semantic-progress曾97→104→102，9月25日真实Gaussian单步35→26，RB也未恢复原能力；这些正负边界继续约束判断。

现在尚未分清：旧组均值只描述同教学随机初态的平均难度，是否遗漏了对当前自身状态可预测的成功概率；
或者在已有交互量和特征上，新增状态预测只是在拟合episode身份、终止长度与噪声，不能产生可迁移的信用改进。
这不是已确认的EMBER独有根因。它与EMBER的关系是：同一教学写出的固定参数改变所访状态，
不同状态上的动作信用经完整LoRA、该condition的生成Jacobian回到共享G；广播回报可能给这条映射增加有限样本噪声。
须先检查这一候选是否有数据基础，再讨论它能否改变实际生成参数的更新与能力，不能靠通用RL公式直接派发正式学习。

## 2. 数学对象及本批能回答的范围

固定采集时的 `Theta=G_phi(V,L)`。对实际SDE转移，记 `S_tk=grad_phi log p(z_next|z_tk,o_t,Theta)`。
当b只依赖该转移前已知历史，且拟合b的数据早于被评分episode时，`E[b*S_tk]=0`，
所以 `E[(R-b)*S_tk]=E[R*S_tk]`。基线改变有限样本估计，不新增奖励，也不保证某条失败轨迹的哪次动作应受惩罚。
值预测 `E[R|state]` 也未必是参数梯度方差最优基线；后者还涉及score范数及跨时点、跨条件相关项。

本批不计算新的S_tk/VJP。使用原消费者已保存的局部velocity损失余切：

```
c_k = 1 + .5*(1-tau_k)
C_itk = (Q_i/M_i) * (10/2) * c_k/tau_k * (z_next-m_old)
q_itk(b) = (R_i-b_it) * C_itk
E_local(b) = sum ||q_itk(b)||^2
```

共同的1/16系数可约掉，必须在记录中说明；Q为原真实replan数，M=min(16,Q)，不重采reservoir。
这是**原抽样事件上、拼接局部velocity输出坐标的余切二阶能量**，不是完整Writer梯度方差、SNR或闭环收益。
后续的真实Jacobian可能放大、压小或改变方向，且本指标没有计入跨时点协方差；不能把能量下降写成机制修复。
Brier误差与E_local共同用于判断有无值得进一步检验的预测基础，均不能选Writer checkpoint。
如果原数据在某任务完全无成功，基线不能凭空提供朝成功的动作方向；不把task25/38的稀缺成功用平滑伪造为正标签。

## 3. 唯一数据、时间划分和信息墙

只读 `/data1/user/ymdai/ember_runs/denoising_return_writer_20261006/train/raw/` 的1152条原episode、
288个原四episode组、72个参数版本；scope严格为原36训练任务，逐一对应 `configs/operator_read_write_v1/seen_task_scope.json`。
复用同root的frozen采集/score消费者、row.json、continuous NPZ和score_records.pt；不读取held/Test或旧时序controls。
原reward就是官方episode终止成功0/1；不用人工阶段分、形状奖励、成功筛选、task重加权或补采失败。

固定三个前向时间窗，不能随机拆相邻frames，也不能让同episode或更晚episode进入该窗拟合：

| 拟合macros | 唯一后续读回macros | 每task拟合/读回episode |
| --- | --- | --- |
| 1–18 | 19–36 | 8 / 8 |
| 1–36 | 37–54 | 16 / 8 |
| 1–54 | 55–72 | 24 / 8 |

各窗预测器拟合后冻结；不看后续结果调参。后续窗使用更多过去数据模拟在线积累，这是预登记的三个拟合，
不是按得分挑点；864个后续episode都报告。参数在旧采集期间变化，故这是历史在线序列预测，不能称固定T能力估计。

每episode只消费原reservoir中保存的所有decision。状态取 `control_step=t` 的连续trace当前样本，
必须是执行本decision五动作之前；成功终止后的帧不当输入。原score record可供核t与tau、计算C，不把动作/未来噪声放进预测器。
每窗拟合权重为episode等权、其M个decision各1/M；36 task曝光本已相等，保持此定义。
除实际预测比较外，不额外拟合反向时间、打乱、更多模型或参数扫描。

## 4. 固定参照、特征和小型预测器

四种基线在完全相同的后续事件评分：

1. **原LOO**：每组其他三条真实R的平均，严格复算原使用的值；只作原信用参照，不送进其它预测器。
2. **过去task均值**：仅拟合窗的本task episode，Beta(1,1)平滑 `(successes+1)/(episodes+2)`；不按frame重复计数。
3. **Context**：one-hot训练task及 `t/H`；H为预先已知的官方suite horizon，绝不是该轨迹实际结束步数。
4. **State**：相同Context，再加下述当前/过去物理量；两学习器同算法和容量，不把task/剩时收益算成状态增量。

State额外输入固定为：当前EEF position3、quaternion4、双指qpos2；当前全部已捕获body position3；
各body相对当前EEF的三维差；这些position/qpos在 `max(0,t-5)` 到t的已发生差及past-available标志；
当前原生goal predicate向量及其有效位。四元数不做简单差分，不凭body原点构造未保存的物体旋转/接触/关节。
body按各task原registry的name排序、右补零并带有效mask；predicate按原BDDL顺序补零/mask。
最大宽度与task类别只从登记scope/原schema取，不读取后续R决定特征。共享模型利用task类别消歧各任务槽位，
这只是训练用privileged baseline，不是teacher-video输入、部署路由或新的任务表示能力证明。

禁止将teacher demo ID、init ID、文件名、replica、R、最终谓词、总steps/Q、未来body位置/动作、
动作生成的z_next/m、参数更新后的值或未来统计加入特征。Q/M只在规定的评分权重中使用。
本批没有RGB/native特征；成功只说明训练用特权状态预测有依据，阴性也不证明所有视觉critic不可学。

两学习器均为同一个固定HistGradientBoostingRegressor：squared_error、learning_rate=.05、max_iter=200、
max_leaf_nodes=15、max_depth=4、min_samples_leaf=20、l2_regularization=1、max_bins=128、
early_stopping=False、random_state=20261008；预测统一clip到[0,1]。不寻优，不用本批未来标签选容量。
可使用当前已有兼容sklearn；若缺失，仅在本批data1 root内安装固定版本的必要依赖并保存版本/命令，
不修改canonical环境、不复制大环境。若无法在预算内取得依赖，回报明确阻碍，不静默换算法。

## 5. 固定读回、反例与判断

保存每episode、decision、窗、四基线预测、原R、特征/来源索引、Q/M/所选tau和C的平方范数。
报告每窗及三个窗合并的episode等权Brier、原reservoir事件的E_local、State相对Context及原LOO的比值；
同时给36 task等权/逐task结果、suite、成功/失败分层以及 `t/H` 的[0,.25)、[.25,.5)、[.5,.75)、[.75,1)分层。
给固定10个概率区间的校准记录；未占用格标空。不能只交总体下降，保留state变差的任务、早期/晚期差异和单类任务。
特别列task12/25/29/32/38；这五项不是另开训练或择优候选。长episode的Q权重可能主导E_local，应并列按task归一化结果。
不将相邻frames当独立样本计算显著性；无需统计显著性包装，也没有独立训练重复的置信声明。

这批的投入参考为：State在至少两个后续窗同时降低Context的Brier，并使E_local至少降低20%，
且收益不是仅由最后四分之一时限或极少任务承担，才提高“进一步检验真实状态基线信用”的优先级。
这个20%是有限后续投入所需的实际余量，不是科学接受门槛，更不是可自动转入RL的资格。
若只有task/time优于LOO，不能归为状态反馈解释；若Brier好但E_local无改善，不能称对应评分信用噪声已减；
若State无稳定增益，关闭这份数据/特征/学习量即可提供有效基线的默认推断，不能扫模型来保护它。
即使两指标强阳性，完整G梯度、实际控制及held迁移仍未验证；main须综合原件决定下一项，不自动训练Writer或扩采样。

## 6. 资源、实现和交付

唯一root `/data1/user/ymdai/ember_runs/state_conditioned_return_analysis_20261008`。
预计20–45分钟，依据1152个现成小型capture的单次顺序读入及6个CPU树模型拟合；不是已测吞吐。
硬上限实际承接起90分钟、8 CPUh、新增data1峰1GiB、0 GPUh；不重复复制约8GiB原score captures。
先核data1独立quota/相关个人实占及shared容量，估计临时依赖/特征表/模型/原件索引峰值；不做GPU preflight。
可分离CPU数据读取与小型拟合，但避免多个进程反复扫共享raw；按实际内存设有界线程，不引入服务或持久训练进程。

实验session独占tracked/Git交付；先以run-scoped小脚本复用NPZ/torch CPU/现成分析工具，
不新增canonical trainer/CLI或修改历史冻结源。保存脚本、实际依赖、命令、边界核对、完整预测和资源计时，便于复核。
验证只针对真实消费者的数据对齐、窗口因果顺序、feature信息墙及计算；不建通用测试基础设施或完整VLA加载。
完成时回填canonical分析JSON、findings/progress/task_plan/history，push清理自有工作区，原episode/score完全保留。
只在整批完成或真实边界一次可靠回main并交回写窗口；无心跳/selfQueue/自动下一批。
main继续负责EMBER目标，不因本分析阳性或阴性而冒称最终方法成立或结束项目。

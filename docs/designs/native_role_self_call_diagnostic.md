# 冻结角色编译器在真实自身初态上的调用读回

2026-10-04，机制§134。原批已资源异常关闭；main又核出额外autocast与原闭环消费者不匹配。
本文件保留原科学合同，不再是active执行；后继仅按[同面板精度修复](native_role_self_call_precision_repair.md)，实际状态由progress记录。
新root：`/data1/user/ymdai/ember_runs/native_role_self_call_20261004/`。

## 1. 待辨问题与方法取舍

前批F/G/R已经关闭：train32为22/24/20，task16的23情境为3/0/2；R的教学读取有明确butter正例。
但G/R约.98的自身角色密度来自训练8task的B20，没有测过上述held失败的实际自身调用。
不能据此把剩余问题直接归到Value/动作，也不能继续假定只是教学选错对象。
本次区分初始情境下的三种事实：已见任务的读取获得是否延伸到真实初态、是否延伸到task16、
以及实际R输出方向与自身A h系数怎样共同产生原/换位差异。它不是新的角色loss、架构学习或资格评测。

若训练初态仍有强读取而held没有，降低“已获得可迁移角色接口、只差下游控制”的支持；
若held初态读取与R的实际正作用均有保持，而原完整操作仍大量失败，进一步降低单纯角色Q编译作为主修复的优先级。
若连训练初态都不保持，B20的成功范围应收紧到其查询分布，不能用它指定跨任务根因。
所有分支均只作这一次读回，不自动补晚期帧、层/质量/方向扫描、辅助项、续训或formal fresh。
首计划不是整条轨迹：初始角色密度、attention质量及局部R作用都不能证明后续角色一直正确或动作因果效益。

## 2. 唯一面板：原101个full的首次规划

只读`native_role_binding_compilation_20261004/evaluation/{parent,F,G,R}/results.json`中capture_level=full的全部行：

| 原臂 | train首次规划 | task16首次规划 | 合计 |
| --- | ---: | ---: | ---: |
| parent C900 | 8 | 0 | 8 |
| F64 | 8 | 23 | 31 |
| G64 | 8 | 23 | 31 |
| R64 | 8 | 23 | 31 |

共101个原条件，每个只读该轨迹`replan_steps[0]=0`的一次官方十步生成。没有新环境step或闭环行。
train固定task12/13/14/15/17/19/43/96、teacher0/init32；这是有限覆盖，不能冒称全部train32或B20同输入。
held继承原八格与原/交换16行合并后的23个case；teacher、physical_init、noise_init分别保持，不重新配对或选成功案例。
其中原/换位的8个init各对应原先固定teacher47/33/28/46/32/1/24/43；八格中的W40/W47仍完整保留。
101行只有25个不同自身物理初态：8个train和task16的17个布局；重复输入不算独立样本。

直接消费原PT的`observations[0]`（双图像tensor和已含自身state语义的language tokens/mask），
以及原`policy_noise_seeds[0]`。这些是实际preprocess之后的输入，不能再追加state文本、再次rotate/normalize/tokenize。
raw `states[0]`只用于来源与场景核对。用canonical CPU生成器和原50×32形状恢复同一Gaussian；
不要用physical_init替换已登记noise_init，不重置为另一device随机流。
保留原first action_chunks和新101份完整50×7输出，报告首5/全50数值差；只识别输入/消费者实质不一致，
不为正常BF16/batch/kernel差异追逐逐bit重放、强制batch1或重复forward。

全部38-target完整A/B直接来自前批sealed banks，R另只读同条件`*_binding.pt`中的18个R_l。
不加载Writer、不读native缓存、不重编译、不更新模型。bank继承metadata的旧400/scene字段已在原
`analysis/bank_execution_provenance.json`澄清；实际来源以原training_contract、bank条件和逐row scene_reference为准。
保留原训练/物化/消费commit与本次读取commit的区别，不重写旧原件。

## 3. 自身可见区域只作退出后的测量坐标

从25个已有post-dummy scene恢复CPU MuJoCo状态、model body pose与原counterfactual XY交换，0环境step。
task16交换严格复用旧object_position_transport及前批消费者的实际规则，不重新选择布局、重做settling或改其它body。
用实际visual geoms/遮挡得到两相机可见实体mask，并与上述保存RGB对应；不是从held教学state/action/pose取标签。
这是冻结checkpoint、无选择、预登记的一次held自身观测诊断；所有mask、实体名和物理state均不进policy输入，也不产生梯度。
图像映射沿真实256→model224、rotate和16×16 patch合同；以保存processed图像为准，避免双重翻转。
核对25个场景的mask叠图；不凭框/点位猜测不可见轮廓，无法对应的输入不得用近似标签替换。

候选实体沿前批role规则：正确被操作物体以及其它可见可搬物体，排除robot、fixtures与目标basket/tray。
43正确为butter_2并保留butter_1干扰，96为butter_1，16为butter_1。实际body/geom名逐task核定。
另保留目标承载容器的可见mask和attention质量作完整读取描述，不把容器添加到原rho的竞争集合。
无目标可见或少于两个候选时rho无效但保留该条件、动作与所有其它测量；不得通过筛可见情况改变101分母。

对实体o，令f_o[p]为patch内可见像素比例，q_o=f_o/sum_p f_o；q是原角色密度的面积归一测量权重。
在同次真实forward中保存全部10个tau、18层、8个head、执行前5槽的原角色scores和图像总attention质量。
rho仍按原`softmax_o(mean_(l,h,i) logsumexp_p(ell_p+log q_o[p]))`归约，不换为局部softmax的均值。
同时测实际全token softmax pi上的实体质量`mu_o=sum_p pi_p f_o[p]`和`mu_o/m_image`，
不把q加权平均attention叫作实体总质量，不用归一密度替代图像读取量。
所有mask在测量hook内只读；真实SDPA/Value输出、mask、GQA、prefix和flow保持原消费者，不注入oracle attention。

## 4. R的实际双线性调用，而非再测未训练的原生方向

本次R_l是前批64末点已学成并实际合入B的矩阵；A_l是同一R臂末点的最终A。
R的原canonical分支已经更新，不能把B-R叫作父C900，亦不运行去R策略。
对层l、head h、槽i、flow tau，在实际自身输入处记

```
a_l,i,tau(x) = A_l h_l,i,tau(x)
d_l,i,o(x) = RoPE_query,i^T [sum_p(q_target[p]-q_o[p]) Ktilde_l,p(x)] / sqrt(256)
w_l,h,i,o(x) = R_l,h^T d_l,i,o(x)
m_l,h,i,o,tau(x) = w_l,h,i,o(x)^T a_l,i,tau(x)
```

Ktilde是该自身prefix的真实RoPE后key，query旋转也用当前实际position_ids；没有teacher位置代替。
m是R对目标相对实体o的**区域平均logit差**的实数贡献；不是logsumexp密度差、目标attention质量或动作收益。
同forward另保留减掉直接R A h后的局部密度，沿原B20同h/own-key口径；不把它称为完整反事实policy。
保存R臂全部31条件的a、w和m（所有合法实体、tau/层/head/前5槽），以及原Q直接贡献汇总；
这些小数组足以独立重算本节，不保存或复制全部prefix/模型hidden大缓存。

对R的八个严格原/交换pair，主比较固定target butter对原主要错误角色orange，原/交换分别记o/s。
不选择最好层、head、flow或init；逐单元和固定均层/头/槽的每tau结果均保留。

```
m_s - m_o = (w_s-w_o)^T a_o
            + w_o^T(a_s-a_o)
            + (w_s-w_o)^T(a_s-a_o)
```

这区分当前实体key坐标变化、自己的调用变化及交互怎样构成同一固定R的局部有限差。
后两个a来自各自真实十步生成路径；交换双方w/a是退出后的代数记账，不是重新运行混合策略，不能由其符号归因闭环成败。
本读取检验已训练R的调用，区别于§127未训练teacher ROI key方向；原§127反例与§110的部分因果转移仍保留。

## 5. 汇总、停止与工程范围

报告全部101行、25场景覆盖及无效项，train8按task等权、held23按原面板和teacher/init/noise分列。
对相同原始输入的parent/F/G/R或F/G/R作配对；不把不同阶段的B20均值与此处初态均值作同分布因果差。
以原197行为、held错对象/抬高未完成/两条R交换成功和全部失败为背景，不重跑环境或把同一成功计入新成绩。
输出原始测量、归约脚本、实际输入/模型/来源manifest、completion、费用及必要图；不只交aggregate或最佳head。

这一次读回结束即关闭运行面。无追加帧、另一方向/ROI、训练、Writer/native、checkpoint选择、400、最终视频controls、Test或RL。
结果即使混合也不靠扩面板追确定性；main结合完整历史决定后继完整方法，不把此处局部阳性作为自动晋级。
复用canonical source、BatchedLoRAInference、官方噪声及已有被动hook所有权；临时模块在本批完成后退役，原件/Git/frozen保留。
发现明确工程违约可在本范围/预算内独占分支修复、针对实际消费者核验、push并new freeze；不得修改冻结树或借此改变科学测量。

预计含工程/CPU mask/读取/分析1–2小时；基于前批一次resident source的batch40十步读回和既存25场景，
纯GPU读取预计数分钟，硬限0.5完整GPUh（含加载/profile/失败/I-O/退出）与新增峰4GiB，保守估计1–2GiB。
实际实现完成后核算，不以低显存为目标；只用这些101合法输入分批提高吞吐，已消费输入不为profile再算。
launch前live检查双节点、总量/单节点上限和strg01 data1独立quota、个人用量、共享容量；所有新增写data1，旧大资产只读。
使用clean pushed detached frozen代码，单个退出事件等待，不定时轮询或向直接等待者发selfQueue。
实验session独占实施/集成/执行/退役，整批一次回报后交回canonical/Git；main接手前只读。

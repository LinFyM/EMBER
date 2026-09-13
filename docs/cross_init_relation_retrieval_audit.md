# 跨初态相对几何检索诊断（2026-09-13）

登记后只运行一次；当前状态由progress记录。本项是训练侧privileged数据诊断，不是action-hidden Writer或方法资格。

## 1. 判别问题与近邻区别

[可识别性分析§9](video_information_identifiability.md#9-跨初态操作关系新增监督必须区别于旧状态条件化与任务记忆)
确认了物体／部件状态可恢复，却不能由同task关系到LoRA拟合定位读取失败。这里先不训练任何映射：
从另一条成功演示取动作，检验以目标相关几何的**相对位置**匹配执行状态，是否比绝对位置匹配更可迁移。
两种检索读取完全相同的privileged内容、使用相同距离尺度与动作Value，唯一主变量是平移坐标表达。

这是对“跨初态相对几何能改进单演示局部动作传递”的有界检验。它不同于旧native激活span、可训练功能头、
同task oracle图拟合或RGB动作反演。没有学习的task→参数映射，因此不需要另训内部task留出Writer来排除该项记忆。
任务身份仍用于合法的同task演示配对及离线对象对应；这使它成为privileged参照，不能据此宣称合法RGB泛化。

正结果只支持该几何匹配原则，负结果停止直接采用本具体检索作为过程教师的依据。它不要求先证明完整Writer有效，
也不允许凭局部通过自动启动完整Writer：后续仍须说明合法RGB获取及唯一LoRA承接关系的实质机制与有界预测。

## 2. 固定输入与时间合同

- 使用固定train24全部任务；teacher参照来自action训练池的demo16–19，query来自原动作诊断池demo42–45。
  每条query episode分别与四条teacher配对，共24×4×4个episode对。每个condition始终仅有一条teacher。
- 每条episode只取`i=0,5,10,...`且`i+5<N`的位置。`obs[i]`与`states[i+1]`同为post-action时刻；
  Value与评分目标均为`actions[i+1:i+6]`的完整5×7控制chunk，不做末端padding或动作平均后复制。
- 从封存目标manifest确认train身份，使用现有HDF5、source normalization、官方BDDL及本地assets；
  不读取validation/test动作，不改24/8/8身份，不读teacher0–15或held teacher46–49的privileged字段。
- 真实state、BDDL对象身份与teacher actions只存在于本CPU参照。没有Writer、source模型forward、LoRA、梯度、
  rollout、reward、terminal、视频扰动或checkpoint选择；不作为新的deployment输入。

## 3. 几何、检索与参照

对象集合固定为该任务官方BDDL的`:obj_of_interest`，不得按动作／结果选运动最大的物体。
每个对象使用XML中该名称或该名称加下划线前缀的body/site；这包括抽屉region site、微波炉门部件等，
不将所有关系简化为物体根部。每个对象至少匹配一个点，名称与类型在同task跨episode保持一致，否则停止报告错误。
本集合不声称包含完整接触或动力学状态；关节部件的位姿仅提供运动学描述。

令e、Rₑ、g为obs中的末端位置、axis-angle转成的旋转矩阵、两指qpos；pₒₖ、Rₒₖ为恢复的body/site位置与旋转。
先在同对象的点内平均，再在对象间平均，记为`meanₒₖ`。两个检索的共有距离为：

```text
D_common = ||Rₑ−Rₑ′||²_F / 2
           + meanₒₖ(||Rₒₖ−Rₒₖ′||²_F / 2)
           + ||g−g′||² / (2 × 0.04²)
D_absolute = D_common + (||e−e′||² + meanₒₖ(||pₒₖ−pₒₖ′||²)) / 0.10²
D_relative = D_common + meanₒₖ(||(pₒₖ−e)−(pₒₖ′−e′)||²) / 0.10²
```

0.10m和0.04m是此次固定的距离尺度，不由评分拟合或扫描；旋转项采用弦距离，不存在Euler绕回比较。
这只消除平移原点相关性，不声称消除相机／具身／任意旋转差异。所有关系共同组成距离，未另加事件或阶段标签。
对每个query，各自取同一条teacher中距离最小的位置，完整返回其5×7动作；平局取最早位置。固定1-NN，不搜K。
第三参照`video_mean`为该teacher所有合法chunk逐位置均值，是不按当前状态检索的privileged动作参照；
它不是多视频平均，更不是LoRA平均。三臂所有动作统一使用冻结source action q01/q99变换后评分。

## 4. 统计、停止与解释

主要指标是完整5×7归一化动作MSE。另完整报告平移3维、旋转3维、夹爪1维，不由这些分项挑选新的主指标。
先在每个episode对内平均，再对四teacher与四query episode等权，最后对24task等权；suite与每条teacher结果全部报告。
固定seed20260916、20,000次task-cluster配对bootstrap，主要差额为`absolute−relative`与`video_mean−relative`。

两差额95%CI下界均严格>0，且relative对两参照在至少两个suite均有净改善，才支持此具体几何传递前提。
否则关闭本参照，不扫距离尺度、特征子集、邻居数、帧率、episode或任务；不能把关节／某一suite结果较好当作总体通过。
若relative优于absolute却不如video_mean，只有相对少损失，没有足够动作传递价值；若两种检索都优于mean而彼此相近，
保留状态匹配价值，但没有采用相对几何替换的依据。全部结果都只裁决本固定检索，不宣称唯一根因或全视频方法不可能。

即使通过，也没有证明这些关系可从合法RGB获得、时序必要性或唯一LoRA收益。尤其本参照读取实际teacher actions，
不把它与旧action-hidden source读出的MSE直接排行。固定模型最终视频controls及独立frame_set资格问题不受本诊断影响。

## 5. 运行与保留

只在CPU恢复存储状态，不推进仿真或渲染。一次性入口`scripts/audit_cross_init_relation_retrieval.py`在完成后退役，
由Git和冻结源码保留；来自clean pushed detached树，复用canonical环境与资产。
输出`runs/analysis/cross_init_relation_retrieval_20260913/`保留登记、精确命令、逐行真实／预测动作、索引、
全部task/suite/video指标、配对区间、完成与失败状态。没有新训练基础设施。

strg01现场data1 quota为991.9GiB/1TiB，共享空间83TiB；现有`.codex/tmp`为13GiB。
新增冻结源码与全部输出按512MiB峰值预算，预估峰值992.4GiB，复用所有大资产、不下载或复制模型。
这是只读数据分析，GPU不参与；不启动GPU预检或任何历史运行。

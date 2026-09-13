# 跨初态相对几何检索诊断（2026-09-13）

**已完成并关闭：相对几何替换前提未通过。** 原合同及评分前修正保留；本项是训练侧privileged数据诊断，
不是action-hidden Writer或方法资格。不追加检索变体或据本结果重开Writer。

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
- 每条episode只取`i=0,5,10,...`且`i+5<N`的位置。`obs[i]`对应`states[i+1]`的控制周期，
  几何恢复另按§6对齐末端传感器的子步缓存；
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

只在CPU恢复存储状态，不推进仿真或渲染。一次性入口`scripts/audit_cross_init_relation_retrieval.py`完成后退役，
由Git冻结9f90a14d保留；执行来自clean pushed detached树，复用canonical环境与资产。
输出`runs/analysis/cross_init_relation_retrieval_20260913/`保留登记、精确命令、逐行真实／预测动作、索引、
全部task/suite/video指标、配对区间、完成与失败状态。没有新训练基础设施。

已在存储authority核对data1独立user quota、相关目录用量和共享容量，现场数值保存在本地launch原件。
新增冻结源码与全部输出按512MiB峰值预算，验证在quota内；复用所有大资产、不下载或复制模型。
这是只读数据分析，GPU不参与；不启动GPU预检或任何历史运行。

## 6. 评分前的子步时刻修正

首次冻结54be9ca3运行在task0/demo16的坐标检查退出，尚未计算任何动作检索MSE。
`states[i+1]`直接forward与存储末端最大差.00061199859m，而旧同下标state最大差.01516064470m。
已安装robosuite的step在每个子步`sim.forward → sim.step → update_observables`，EEF传感器读取site_xpos缓存。
task0/demo16的17个登记位置中，从下一存储qpos按同一qvel回退一个模型子步再forward，最大位置差约1.09e−15m；
此模型子步.002s，外层控制周期.05s。这支持末次积分前运动学缓存的时刻对应，不是把未来动作标签改回同下标。

修正后的诊断从`states[i+1]`通过`mj_integratePos(..., -model.opt.timestep)`恢复缓存几何时刻再forward，
继续保留原obs末端／夹爪读数和`actions[i+1:i+6]`。每个episode仍核对恢复末端与存储末端在.0001m内，
记录实际子步与最大误差，不扫描阈值。此次修正不涉及RGB、source模型或生产Writer代码。
原失败registration／failure与日志保留；修正的完整输出放同一analysis根的`completed/`，不覆盖首轮原件。

## 7. 完整结果与裁决

9f90a14d冻结执行正常exit0，99.37秒；24task、192条episode的6,513个几何状态全部恢复，384个episode对、
13,064个唯一task/teacher/query/frame位置完整。所有episode的模型子步均为.002s，坐标对应检查通过；
全部真实／预测动作有限，按episode对、task等权从raw rows重新计算的MSE与summary一致。

| 动作MSE | 绝对几何 | 相对几何 | 单视频动作均值 |
| --- | ---: | ---: | ---: |
| 全24task | .12443553 | .12634790 | .25459689 |
| Spatial | .14956629 | .15157419 | .28452838 |
| Object | .10331134 | .10773631 | .25113665 |
| Goal | .13092024 | .13098083 | .24804866 |
| Long | .11394427 | .11510028 | .23467385 |

绝对−相对为−.00191237，task-cluster95%CI[−.00421347,+.00027549]跨零，仅8/24task为正；
四suite点差额均负，共同满足两参照改善的suite数为0。均值−相对为+.12824898，CI[+.11063319,+.14508720]，
24/24task为正；绝对几何也在24/24task优于均值。**两项要求未同时满足，按登记关闭本相对几何替换前提。**
没有证据宣称相对坐标普遍无效，也不能把其相对均值的优势写成相对绝对几何的优势。

按teacher16/17/18/19分别对query episode与task等权，相对−绝对MSE为+.00111197/+.00685869/+.00019884/−.00052002；
没有选择其中较好的teacher。两种状态匹配相对均值的收益在四条teacher上均保留。
预登记分项中，绝对／相对／均值的平移MSE为.08314998/.08391011/.21231973，
旋转为.09144200/.09160858/.08597376，夹爪为.34727279/.35787925/.88729774。
总体收益来自平移及夹爪，旋转并未胜过均值；这限制把参照称为完整动作教师，不能用总体MSE掩盖该分项。
分项不改变主裁决，不据它另扫旋转坐标或特征子集。

本结果支持“给定privileged任务对应、几何及真实单演示动作时，状态匹配能传递部分局部动作价值”；
它不支持“世界坐标混淆是当前Writer的主要缺口”，也未检验合法RGB获取、视频顺序或固定LoRA编译。
与旧source端点／局部头的输入和评分位置、horizon不同，不以近似MSE作跨实验排行。
后续不追加坐标／邻居／尺度变体，不默认将该oracle接为新Writer教师；新机制仍须说明缺失信息及行为传递的判别。

原件位于`runs/analysis/cross_init_relation_retrieval_20260913/`：顶层保留初始失败与launch，
`completed/`保存`registration.json`、`raw_rows.npz`、`geometry_schema.json`、`episode_scores.json`、
`summary.json`、`evidence_audit.json`、`run.log`和`completion.json`；总计约1.3MiB。没有新checkpoint或LoRA，
没有GPU、训练、环境rollout、Test或最终controls。一次性入口退役，整体goal未完成。

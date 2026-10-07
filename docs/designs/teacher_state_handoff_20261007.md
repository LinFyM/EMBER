# 真实教学动作控制器能否接住T已经访问的状态

2026-10-07。Owner持续自主授权；progress登记后为唯一active有限分析批次。
0训练、0新VLA/native/Writer forward，不恢复Gamma、G/F、P/Q或旧专家蒸馏。

## 1. 实际不足与本次方法判断

同一原GT绝对几何1-NN，真实教学动作从初态成功91/144、覆盖34/36 task，
而source预测值0、裸端点6、冻结Gamma校准值22。Gamma只保留真动作成功17/91，
Object/Long全零；其144条件误差改善98.3446%来自五步均值，剩余误差87.3064%也在均值项。
不能继续以“动作已读准、只差写入”为前提恢复校准配方，或由MSE选择一个新动作辅助项。

真实动作控制与T2340有具体互补：T109、NN91，R71、NN新增20、丢失38。
其中target24为T62/NN63、得19失18；support12为47/28、得1失20。
task25/29/38的T为0/2/0，NN为3/4/3。该事实不把NN认证为强完整教师，也不证明T所访状态可由NN纠正。

现有主FM用同task跨episode的成功示范查询；它已具有真实动作信用，但没有直接说明偏离后的控制。
后继若考虑训练期纠正监督，需要一个在学生实际访问状态上仍有用的教师，不能由教师从初态成功自动推定。
历史phase/state aggregation已经用decoder自身occupancy查询task expert的FM velocity，但只更新privileged-code decoder，
且没有检验expert在这些状态的实际恢复；旧SEOD/GOMQ主要用成功expert occupancy，P/I用预建几何成功轨迹。
它们的正反证据继续约束后继，不能把DAgger重新命名为尚未试过的方法。

本批只问：**原真实动作NN，在T执行50个真实动作后，到达的原训练状态上，是否仍具有有覆盖的继续控制能力？**
这决定是否值得研究该教师提供的条件反馈监督。即使成立，视频取得控制内容、单LoRA学习、稳定保持和held迁移仍未成立。
不是继续优化NN，不以接力取代一次合法LoRA部署，不从这一诊断直接启动学习。

## 2. 唯一干预与288个新episode

保持原36个训练任务（24 target＋12 support）、每task初态32–35、seed20260928的teacher映射。
用原T2340真正ODE读出的144条continuous `actions`，不是未执行的50步预测、teacher action或重新采样。
main已核144原件共28264条实际动作、2,184,564 bytes；condition/teacher/scene齐全，全部长度>50、无50步内成功。

```text
原封存scene + 原dummy/控制器初始化
  → 逐步执行同一T原件 actions[0:50]，不加载policy、不消耗flow噪声
  → after50、尚未执行action[50]时：
      T_replay：继续原件 actions[50:]，至原生成功或原件结束
      T50_demo_NN：live GT query → 原D_abs/最早1-NN → 原demo raw动作前5
                  每5步重查，至原生成功或原suite原总horizon
```

两臂各144，共288新行；不能只运行原T失败或NN成功子集。切换点固定50（十次五步执行），
使每行都在真实T访态上发生干预；不从结果搜索25/100/其它点。它只代表早期访态，不覆盖所有晚期失败状态。
切换处不reset、不settle、不重新初始化控制器，不把T前缀时间从horizon中扣除后再给一整轮预算。
不使用中途sim_state覆盖假装完整控制器状态克隆。

NN继续使用原144条teacher/4556合法位置、完整OOI body/site signature、原绝对距离各尺度及最早argmin。
Value只来自原`demo_actions50`及原valid length，实际执行前5，保持原offset1；无Gamma/source值、均值/平滑/scale/phase修补。
T重放动作直接使用原raw物理7维命令，不重新反归一化；若capture需要normalized值，只生成记录侧表示。
重放提案只将实际已有的最多5个动作标valid，未执行padding明确false，不冒称新的50步模型预测。
NN提案沿原50步真实Value/valid mask；两种proposal来源清楚区分。

## 3. 参照有效性与信息墙

T_replay是已存T轨迹的物理重放参照，不是一次新的T在线推理分数。
原109及全部成功集合必须与其重放结果一起报告；记录原件结束和实际原生终止步，不补动作、重复末动作或补造成功。
若合法重放不能复现预定T成败对比，保存完整结果与差异并报告本次对比的识别限制；
不通过筛行、延长、重抽seed、修改cut或dtype追回原分。已定位输入/资产/接口违约可在原预算内修复并保留失败。
原件耗尽却未达到原成功时明确记录replay-exhausted，不伪装成完整官方policy episode。

核对两臂原scene、teacher、语言、env seed、实际前缀raw命令和第50步状态/原生谓词；
保存相对原T continuous及两臂间的EEF、朝向、夹爪、body位置误差，不将低位差自动当作工程故障。
源点/前缀不匹配或改变实际解释的状态偏移须作为真实边界报告，不能只查文件路径便宣称达到T访态。
沿旧完整初始scene恢复与官方物理/相机、dummy10、执行5、成功即止、suite原horizon；support90沿其原合同。
保存原policy RNG索引作为配对provenance，但本批NN/replay均0flow、无policy噪声消费，不重置或伪称采样。

这两个臂都不是合法EMBER部署；NN读教学动作及自身GT，只在当前训练诊断内。
不加载旧expert bank、F、Writer或VLA，不重算native/cache，不读teacher HDF、held/Test，不产生梯度。
旧expert bank任务划分及source不兼容的事实保持；本批只复用当前已审计的36-task缓存。

## 4. 分析与可失败预测

主比较T50_demo_NN对同轮T_replay；同时完整保留原T109、NN从初态91、Gamma22/裸6及T/MT背景原行，不重跑这些模型。
报告逐task/suite、target/support、breadth、R/G/L、churn/Jaccard及全部成功集合。
按**原T**预登记35失败/109成功，以及T失败且原NN成功20、两者都失败15两层；完整面板不因分层而筛选。
区分接手后的真正新增、保持、丢失与可能的前缀终止；不以success-set union、事后router或逐例择优组成方法分数。

- 若能从T访态取得跨任务新增并保持较多原能力，支持该条件反馈教师在此访态域有用途；
  后续才考虑受控的纠正监督学习，仍须解释RGB视频到单LoRA怎样获取/保持它，不能据此宣称数据覆盖是唯一根因。
- 若从初态可行而T访态下大幅退化，降低该NN作为纠正教师的优先级；不调距离、阈值、phase、平滑或再换cut保护它。
- 若只有少数task改善或得失抵消，保留局部例但不认证强教师；不据可挑出的成功状态启动大训练。
- 若重放/配对不成立，只报告识别失败与具体原因；不将工程/参照问题写成科学阴性。

首步NN动作、选中teacher frame/距离和后续选择轨迹服务于实际行为解释，不将这些内部量当性能或唯一根因。
本批不能区分所有状态分布/动作误差/目标绑定机制；也不证明纠正可学、未见video泛化或改善相对强MT的完整方法。

每task/init32双臂full RGB，其余compact：72full/216compact，全部连续实际动作、提案、原生谓词/场景轨迹。
沿已有helper看12/25/29/32/38/73各init32双臂、每clip8个真实已存时刻/双camera，共192图，登记实际时刻与审看范围。
第50步原始capture/状态另外可追溯；不补未保存终态，不把物体中心运动/RGB外观写为contact、grasp或必要姿态真值。

## 5. 资产、预算、工程与交付

原T两份results（真实132＋12行）均在`/data1/user/ymdai/ember_runs/`下：
`denoising_return_writer_20261006/readouts/parent/seen/ODE/evaluation/results.json`，
`query_conditioned_transition_read_20261005/parent/evaluation/seen12/results.json`。
每行使用其continuous trace和原scene引用；不从trajectory中的未执行chunk拼物理前缀。
原NN：`privileged_action_memory_control_20261007/memory/demo_action_manifest.json`与所指geometry/action NPZ；
数学及controller源码复用该root `frozen_controller`（bfd76c99），必要读取沿已修正正式资产接口，旧frozen不改。
新唯一root：`/data1/user/ymdai/ember_runs/teacher_state_handoff_20261007`。

预计30–60分钟，硬限**实际承接起3wall小时、1完整GPU小时、16GiB新增峰**，含工程、失败、EGL、profile、分析、Git/冻结/临时。
依据最近288个CPU NN/双EGL各六worker约273/275秒、整批.169GPUh及8.2911GiB；本批无Gamma或VLA加载，但新重放接口有工程不确定性。
2wall小时或.75GPUh时若剩余难在硬限完成，只报一次真实边界，不减面板或扩预算。
先strg01独立data1 quota/个人用量/shared核峰值；每launch双节点live准入，沿项目8/空闲≤10时6及单节点6上限，EGL也计卡。
复用cost-balanced/long-first/persistent queue，两臂可独立并行。按实测CPU/渲染/显存余量选择worker数；
有明显余量须在既定有效行内检查更高并发吞吐，保留已完成行，不为profile重做有效科学case或增加矩阵，不加载无用policy占卡。

实验session独占代码/Git；复用原NN、initial-scene、capture、prefix/replay与评测器owner，新增最小有限重放/切换接口，不复制评测器。
针对真实消费者核raw命令、50步边界、原生终止、state配对、valid mask及恢复；独占分支集成/push后新clean detached冻结。
正常工程修复由执行者在原科学/资源范围闭环；改变状态/动作/面板、预算或原因不确定时回main裁决。
首合法episode兼工程验证，计入288；只等待整批退出，不轮询共享缓存/log。封口退役专用模块/flags/hooks，保留原件、Git、费用与失败。
完成或真实边界一次整批回main并交回canonical tracked/Git，无阶段Queue、自通知或自动训练/其它后继。

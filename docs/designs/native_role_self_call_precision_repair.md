# 自身初态读回的原消费者精度合同修复

状态：2026-10-04同101修复及main独立科学消费已结束，见机制§136/findings§315。
下文为封存执行合同；无第三次读取或由本修复自动产生的学习授权。旧资源违约与原件保留。

2026-10-04。只修复已定位的读取执行差异，科学面板与测量定义继承
[原101条件合同](native_role_self_call_diagnostic.md)，不新增架构、模型、训练、环境或诊断问题。

## 1. 已定位的差异与证据状态

原读取`b546c76f`在整个`policy.predict_action_chunk`外新增了
`torch.autocast('cuda', dtype=torch.bfloat16)`；旧闭环的
`binding_evaluation.evaluate → pi05_evaluation._plan_action_chunks → FrozenOperatorAdapter.predict_action_chunk`
没有该外层上下文。相同的`config.dtype=bfloat16`不意味着相同执行：安装版π0.5明确保留完整vision/projector
及action/time等FP32路径，整体autocast改变其中符合条件算子的运算精度。
`BatchedLoRAInference`又按物理adapter的destination dtype转换因子；不能由封存因子FP32推断全部38处原来都以FP32运行。
旧B20确有其自己的autocast，但在进入闭环评测前已退出；它与修复后的初态读取同时存在查询分布及精度上下文差别，
不能把两者数值差单独归因于任务、阶段或某一个架构接口。

已有首5 action7 RMS最大.243643不能概括为低位差异；也尚未证明其全部由autocast造成。
修复依据是已定位的消费者上下文不匹配，不是追求逐bit、固定batch1、关闭TF32或追到喜欢的结果。
原101数组仍是其实际autocast实现的完整测量，保留原件及来源；暂不以其单独裁决旧闭环策略的读取机制。

原批的两项I/O违约同时保留：data0默认assets缓存632个受影响文件约145.4MiB；
逐row view序列化重复批storage使观察新增约19.207GiB，超过4GiB硬限。
去重保留全部逻辑数组后的当前root加受影响data0约1.592GiB；原批`complete=false/closed=true`不改写为合规。
修复是main对已定位工程违约的显式处理，不撤销旧峰值、不重新解释为科学阴性。

## 2. 唯一修复与输入复用

恢复旧闭环实际生成上下文：原生配置的BF16/FP32混合精度、原TF32设置和inference/no-grad语义，
不在完整生成外额外启用autocast，不扩大FP32范围，也不修改模型权重、因子、SDPA、Value或flow。
实际消费入口记录外层CUDA autocast关闭；仅对已定位的原生精度路径作必要代码/运行检查，
不逐tensor扫描、重复模型forward、扫dtype/kernel/batch或用数值一致性阈值选数据。

只重读原`inputs.json`全部101条件一次；复用25个已核自身mask、原processed observations/seed、完整76因子、31份R。
不重做mask、不创建环境、不读Writer/native、不改变自身/teacher信息墙、实体集合或8对swap对应关系。
全部10tau/18层/8head/前5槽、原rho/实际mu、a/w/m/同h局部减R/八对分解和117配对按原合同保留。
这是同一面板的消费者修复；原101是上下文不匹配的读取，不能记作第二批独立科学样本或新闭环分数。
保留新输出与旧闭环原chunk、旧autocast读回的数值比较，报告所有不利行，不为追回旧动作再重复。
若恢复上下文后仍有较大差异，报告剩余边界；没有自动第三次读取或进一步精度修补。

## 3. 产物与存储修复

唯一修复输出目录：
`/data1/user/ymdai/ember_runs/native_role_self_call_20261004/attempts/consumer_precision_repair/`。
旧root的inputs、labels、banks引用、predictions、原报告/completion、失败和超限记录均只读；不覆盖旧数组。
修复目录独立保存预测、必要归约、来源/精度合同、费用/退出、completion及报告，明确指向原输入与旧闭环/旧读取。
不用复制已有trajectory、mask、bank、source或模型。main已有CPU分析属于旧autocast实现，不能转贴作修复结果。

保存单row时先使实际逻辑tensor拥有独立storage，避免序列化全batch backing storage。
在第一个正式row落盘处核逻辑大小与实际文件大小，并据剩余固定字段估算全量峰值；这是本次真实存储故障的针对性检查。
检查使用实际合法输出，不另做模型smoke或新输入。只记录必要大小，无新增hash/sidecar/全树完整性扫描。
发现同类膨胀立即停止写后续row并用现有内存/已存原数组修复，不依靠事后清理维持预算。
不再导入环境构造/默认assets下载路径；已有canonical资产只读，所有新cache/tmp/输出写data1，强制离线。
data0旧缓存原状不明，保持不动。

## 4. 费用、实现与停止

预计工程、同面板读取、归约、退役及交付30–60分钟；实际前批有效消费者93.23秒，其中101查询11.90秒。
预计新增约.03–.06完整GPUh，原已计.032267347；**整个读取及本次修复累计硬限仍为.5完整GPUh**。
不重开预算，把加载、失败、profile、写盘、退出全部加入原累计账本。
预计含保留旧root、受影响data0、新约.883GiB逻辑预测、工程/frozen及逐row临时文件的修复期间峰值2.8–3.4GiB。
**修复期间当前总占用硬限4GiB**，旧已发生19.207GiB峰值仍单列并保持原批资源不合规。
launch前必须以实时strg01 data1独立quota、实际相关用量、共享容量核算上述峰值，不能只用本估计。

GPU同时live核两节点、原有加新增卡数、8/6合计及单节点6限制；source一次resident，
用同101合法输入分批提高吞吐，不重算成功修复输入作profile、不增加工作填卡。
实验session独占工程/集成，clean pushed detached frozen运行，修复旧代码不能原地修改冻结树。
不得重新开放旧三臂/旧读取launcher；临时修复入口有独立生命周期，交付后退役并封闭。
除本已定位执行/序列化修复外，若改变科学口径、原因未定或预计越过上述剩余预算，具体回报main。

完成后保留全部结果及误差边界，不以分数判断是否“修好了”；本修复不选择新架构或checkpoint。
一次持续等待进程退出，整批一次回报并交回canonical/Git，不阶段selfQueue或定时轮询。
无自动晚期帧、另一ROI、层扫描、训练、400、controls、Test或RL。原批资源异常与修复科学可用性分别说明。

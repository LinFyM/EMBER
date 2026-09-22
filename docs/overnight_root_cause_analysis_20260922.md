# 2026-09-22 夜间：有界根因探索

## 目标与授权

Owner授权先分析与小规模验证，重点为新协议闭环绝对能力及正确视频的有益增量，时间可至9/23 08:00 CST。
首次探索预算约1–2小时。任何正式fresh训练或正式续训必须先停下，向Owner提交具体改法、试验结果、成本与方案，等待确认。
当前不启动Test、FT、RL，不改划分、不降低MT-BC300的155/400对照，不凭小面板声称正式改进；稳定性暂不作为主要优化目标。

统一study：`/data0/user/ymdai/ember_runs/overnight_root_cause_20260922`。复用已有Source、checkpoint和dataset，
预计新增峰值≤10GiB。两节点总使用≤6物理GPU；launch前实时检查gpu01/gpu02与strg01独立quota。
每项完整完成后才解释结果；原始不利结果保留。诊断脚本属于该study的独立artifact，不接入部署或canonical训练路径。

## 已登记的第一批冻结检查

### 1. 旧MT-BC早期节点：一次固定step200，补齐选点窗口证据

旧训练step200与新MT-BC300均为每task4,800查询；旧为24×24×200，新为36×16×300，任务池、优化步数和LR仍不同。
旧step200在旧训练原合同checkpoint列表中且完整载荷存在，选择理由在运行前锁定；不扫描其它旧早期节点，不改变旧425选点。
现有evaluator的旧validation authority要求全部8task，因此复用原旧协议完整400评测，完成后主要分析其中与新协议共同的
5task/250配对条件，同时如实报告8task/400。该旧validation8由当前train3与validation5组成，不含当前Test。

入口：现有`evaluate_pi05.py run`，原`pi05_source_aligned_evaluation.json`与`pi05_source_sft_aligned.json`；
Source aligned raw1000、MT-BC旧rank128 step200、原seed7、50states、官方flow10/replan5保持。来自clean pushed detached
runtime，先用gpu01两张合格卡、每卡3replicas，预计约30分钟。只冻结推理，不加载任何训练梯度。
判断：若旧200明显高于旧400/425/450，说明旧晚期评估遗漏较强早期节点；若没有，则该单节点不支持此解释，不能断言所有早期点都弱。
它不是纯数据划分消融，也不能把新MT-BC峰值155删除。

### 2. 因子变化是否落在实际无效方向

已有D1表显示C600→C1200的换P动作差额约缩小19倍，而A/B相对变化未同比缩小，故定位完整因子后的实际有效更新。
固定原D1八个共同Train tasks、两组teacher/query、32query与原随机流，独立读取O1200/C600/C1200。
复用原CC/CO/WW编译，增加仅替换WW的A或仅替换WW的B两个同模型冻结混接。每condition/module用薄矩阵Gram计算
`δ(BA)=B0δA+δB A0+δBδA`各项能量、内积与相消；同时测相同真实flow10动作变化。
只读每个模型自己的表示，不跨checkpoint交换hidden。记录每task/condition/层，比较原D1 CC/WW的量级，不要求逐bit一致。
判断：有效BA已缩小/相消与BA仍改变而执行动作不敏感分别定位不同接口；任何一种都不直接证明冻结A/B后会学得更好。
每模型单GPU预计数分钟，不做闭环、不训练、不读held动作。

### 3. Flow时间轴上的拟合与视频作用

同一D1八任务/32query、CC/WW和三模型，固定actions/noise，对`τ=1,.9,.7,.5,.3,.1`分别测velocity的
前5/全50/逐动作维误差及CC-WW差额；记录τ1一步动作误差、原真实flow10前5动作误差及其视频差额。
目的：区分视频作用在进入动作场前已减弱，或仍在端点存在却未传到实际多步生成；检验端点拟合与其余时间的变化是否分离。
此处辅助项只改变测量的时间点，所有模型冻结。曲线不能代替闭环、不能直接批准新loss或训练路线。
脚本复用`FlowSample`、`NativeFlowPrediction`、`compile_path_panel`和`probe_one`。

### 4. MT-BC有效秩的CPU几何检查

MT-BC为rank128，Writer为rank16且共享FactorHeads，不是任意MT-BC的严格表达超集。对M300各模块BA用薄QR/SVD计算
rank16/32等截断保留能量，先只判断该差异在当前权重上是否明显；不将能量保留率当成闭环能力，不新增rank扫参训练。
如确需冻结功能检查，应在看到该检查结果前登记唯一比较与训练侧面板；不得据低能量损失直接宣布瓶颈排除。

## 后继裁决

第一批结果合并后，只对得到具体支持且尚未被历史证据淘汰的一个机制登记短程配对验证。
须明确它与既有D3的J/Q/M及失败低LR续训有何不同，并保留原目标/数据/训练侧闭环的对照。
若没有得到支持的干预，不用新的泛化说法包装根因，也不自动投入正式训练。正式方案一律等待Owner确认。

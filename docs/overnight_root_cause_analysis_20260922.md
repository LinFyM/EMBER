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

CPU检查完成后登记唯一功能比较：M300原rank128对逐模块最优有效rank16截断，均保持原物理rank128形状/scale1，
后者用薄QR/SVD后的因子零填充；只改有效BA，不合并Source、不重新训练。采用原D1的32条Train query、八共同Train
tasks×state0/1各16闭环；使用同一冻结模型、官方环境/RNG和原生action生成函数。总64动作probe、32闭环，预计单卡15分钟。
保留两个完整臂后解释，不按部分成功选rank，不以该小面板分数代替M300正式155/400。

Flow时间检查补充实际十步路径观测：只在原生`predict_action_chunk`内部临时观察`denoise_step`的x/t/v，原样调用并恢复方法，
不增加forward；每模型32query×10步，逐步记录CC/WW差额。teacher-forced时间曲线与真实采样路径分开解释，避免把
Euler步长0.1带来的能量0.01倍误写成网络抑制。

## 第二批：额外七个查询的目标形式，54更新配对诊断

第一批未发现简单A/B相消或“仅τ1变好、其它flow时间变差”。辅助目标仍不是普通的多episode FM：
它只在τ1监督前5步；按坐标计，辅助样本相对主样本权重为10倍，前5步占名义监督质量32.5%，而普通50步FM为10%。
这只给出可检验的优化差异，尚未证明其有害。直接检验Owner提出、现有D3删除辅助并未检验的“保留额外查询、改为普通FM”。

- P：冻结C600父模型，不更新。
- J：C600完整Writer/Adam状态，原主21＋辅助7、辅助权重1/3、τ1/前5步。
- F：相同父模型、Adam、事件和权重；仅把辅助7的目标改为与主项相同的随机flow时间、完整50步FM。
  原辅助episode、frame、noise seed不变，仍跨episode；不是新采样28条独立episode。损失整体尺度维持原合同，
  不静默改成简单mean28（F相当于4/3倍的加权28查询均值）。这是时间分布与动作horizon的组合干预，不能单独归因其中一项。

J/F固定54更新，重放原events601–654、每更新4tasks、21+7查询、单次联合LoRA cotangent和Writer VJP、clip1、AdamW；
固定LR=.0002992056748283996，与既有D3相同，保持原optimizer状态，不声称同拓扑exact resume。该窗口长度为6个九步轮，
但首尾跨半轮，实际task曝光可能为5/6/7次；先从sampler登记实际计数，再核对两臂逐事件一致，不跳事件来人为凑平衡。
只保存终点诊断checkpoint，不评价中途节点，不选点，不追加步数或自动正式续训。

终点采用全部36个Train tasks，每task state0/1与teacher46/48固定配对，共72闭环/臂；目标24的48条与辅助12的24条分开报告。
另用独立query47/49在25%/75%位置，共144动作query/臂；noise seed为12345+task*100+pair*10+fraction_index。
测量query不产生梯度，更新仍只用demo0–45。所有臂仅correct教学视频，无新增wrong/shuffle/reverse，无Validation/Test。
部署仍只一套完整LoRA，source与normalization不变，官方真实flow10/replan5及环境合同不变。

预先将“值得提议正式验证的明显信号”定义为F相对J和P均净增至少8/72，且F相对J的目标24净增至少4/48，
至少两个目标suite净增为正；同时报告逐task/suite、共同成功/新增/丢失、任务簇区间，不以阈值代替不确定性。
若未满足，不据这个小面板宣称修复，也不据短程续训阴性断言fresh版本无效。满足仍只能支持该目标组合在训练侧有短程收益，
不能证明新任务泛化、正确视频的必要性、完整架构根因或正式155/400改善。

脚本为study/objective_pilot/objective_pilot.py，复用冻结runtime的SupervisedEngine与原rollout helper；无canonical训练源码改动。
预计三臂各一GPU，J/F训练加评测45–65分钟，P更早结束；每臂硬上限90分钟。结束后统一解释，正式训练仍须Owner确认。

## 比较口径补核：Spatial3错误视频donor的冻结2×2

核对旧新原始condition发现：旧O1500的Spatial3 wrong来自Object3/global13，新C600来自Object1/global11。
旧correct/wrong=35/0，新=41/45，但错误视频不是同一干预。共同五task中只有Object1和Goal3的wrong映射完整一致；
Goal6、Long1的donor也随协议改变。此问题不改已有各自面板事实，但限制跨历史因果解释。

因此在看新增结果前登记一次sealed post-hoc冻结审计，只补Spatial3的两个缺失格：O1500使用新donor11、C600使用旧donor13，
各50个固定原init states，逐行直接采用对方已封存wrong条件的demo/state映射，目标language仍为Spatial3；每臂全部50视频各一次。
原O1500/donor13=0与C600/donor11=45保持原件不重跑，原correct35/41只作各自参照。
仅RGB视频、冻结Writer/source与闭环；不读held action query、不产生梯度，不选checkpoint，不改P/J/F或后继架构/超参数。
用途限于裁决“旧新wrong对照改变是否足以改变原先视频证据的解释”，不能当作新的400资格成绩，也不触及Test或时序controls。
若同一模型两donor差额大，则跨旧新wrong差不能主要归成模型路径变化；若小，才弱化这个特定donor混杂。
固定两模型，不按第一臂结果取消第二臂；两格完整后统一读分，逐state保留env/policy RNG与compact trajectory。
脚本study/wrong_donor_audit/wrong_donor_audit.py，复用原rollout helper。预计2卡各10–15分钟，上限每臂30分钟；
与P/J/F同时总量≤5卡，launch前重新live检查。任何正式训练仍须Owner确认。

## 后继裁决

第一批结果合并后，只对得到具体支持且尚未被历史证据淘汰的一个机制登记短程配对验证。
须明确它与既有D3的J/Q/M及失败低LR续训有何不同，并保留原目标/数据/训练侧闭环的对照。
若没有得到支持的干预，不用新的泛化说法包装根因，也不自动投入正式训练。正式方案一律等待Owner确认。

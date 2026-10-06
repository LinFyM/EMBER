# 固定父B输出空间：已有有效修正的投影消费者

2026-10-06。依据为findings§251–252、§338–340、§360；是否执行只看progress。
这是一次无学习的有限train诊断，不是新Writer、Reader续训或正式checkpoint资格。

## 1. 科学问题与必要边界

上一轮L/R共同经过父完整 `B_T(V)=B0+M_T(V)`；每处新增作用均在其列空间内。
它们保留全部教学转移，却没有允许任意新的输出方向。原合同已承认此边界，当前不把它改写成执行bug。
在同一个T2340基础上，旧condition私有直接B学习D把有限闭环17/32变为23/32，有真实得失。
CPU投影发现，D相对T修正仅40.4252%的诱导BA能量在父B列空间内，Q/V分别36.9008%/81.6823%。
这与旧PZ的95.86%不同：PZ拟合八条件共同的自由δO，使δO Z逼近D修正；没有限制到各自B_T列空间。

本批只问：**把这些已经取得局部行为增益的修正限制到父B列空间，是否仍保留其作用？**
如果投影保留主要实际增益，降低输出范围解释，维持冻结内容读取族的低优先级；
如果损失D增益且旧PZ保留更多，说明上一轮L/R排除了一部分实际有用的既存修正，收紧其阴性的外推范围。
这两种结果均不能认证新的Reader、证明线性编译不可能，或说明跨任务共享生成已经可学。
投影后失效也不证明空间内不存在另一套有效参数；未重优化的投影可能破坏D的共同适应。
该空间是每处LoRA的局部输出空间，后续深层非线性仍会变换其作用；不能将其直接解释为最终7维动作或物理方向限制。
本批不追加空间内私有训练、更多投影/阈值/scale、层级删减、任务/初态搜索或新held测试。

## 2. 唯一输入与投影

旧root：`/data1/user/ymdai/ember_runs/operator_learning_limit_diagnosis_20260930`。
只读其parent/D八份完整bank、原scene、原B20固定查询与已有parent/S/P/D消费者结果；
旧PZ及其预测/逐行只读 `/data1/user/ymdai/ember_runs/operator_projected_repair_consumers_20260930`。
条件固定为task0:teacher40/11、task12:25/14、task20:38/42、task32:17/43。
父来源仍为T2340，见旧root的parent bank；A、source、normalization、processor、语言及原评测全部保持。
D是旧合法train更新得到的condition私有诊断权重，不是可部署字典、held teacher或新的性能参照上界。

每个condition、每个38-target独立定义：

```
delta = B_D - B_T
B_T = U diag(s) V^T
U_keep = U[:, s > 1e-6 * max(s)]
Pi = U_keep U_keep^T
A_PB = A_T
B_PB = B_T + Pi delta
```

唯一新臂名PB。CPU FP64 SVD/投影后落普通FP32 bank；不新增优化、native/teacher forward或重编译视频。
完整父B含B0，不能误投影到M、O、Z或A空间；不平均condition，不删除38处中的任何一处。
已有几何原件为 `docs/analyses/fixed_b_range_geometry_20261006.json`；固定1e−6，无消费者驱动的阈值修改。
实际父rank为Q/V/action_in各128、action_out为7；后者的D修正全部保留，不伪称32维满秩。
保存投影公式、来源、各condition/site保留与移除能量、完整bank及manifest；不覆盖旧资产。
检查shape/finite及实际消费者读取的因子，普通浮点差异接受；不添加hash或逐bit复核。

## 3. 实际动作与唯一32行闭环

复用旧§33/34的B20事件、query_offset=1、policy RNG及十步flow，PB生成全部50步归一化动作。
四task各20个独立于旧D新增64步更新的查询，每查询两teacher，共160个条件化读出；父T此前已见训练数据，不能称完全未见。
原始事件/噪声必须沿旧实际消费者的登记读取，不用重新抽样的同名B20替代；其它臂直接复用已有预测。
报告first5/full50/valid-future及motion6/gripper1风险，PB−D实际动作差、相对parent修正的cosine/相对L2。
同时保留PB相对parent/D/PZ的得失；不用动作接近度代替成功。

仅新增PB四task×两teacher×init0..3，共32行、16个不同物理初态。
scene、task/state/teacher、env/policy RNG与旧parent/D/PZ逐行配对，官方render256/model224、双相机rotate、8维state、
7维action、10flow、执行前5后replan、settling10、原suite horizon和成功即止均不改。
沿旧有限池两teacher面板登记，不称正式K1无放回、400资格或未见task迁移。
固定第一teacher/init0的四条full，其余compact；保存原有continuous trace/goal谓词，不新增姿态或stage成功门槛。
按task、teacher、state完整列出PB与parent/D/PZ/S的R/G/L、churn及成功集合，列清D的七新增与一丢失是否保留。
重点比较旧task32两个teacher的相反个例，但不据它们增加初态、画面或counterfactual消费者。
32条全部纳入，不因动作差较小、早期好坏或task0原已满分提前删行/选点；不重新评测旧臂。

## 4. 实施、资源、产物与停止

独立实验session `01a10a98-6d4b-7d61-b12c-da38a628cb45` 执行；main负责科学消费，双方tracked/Git写入串行。
唯一新root：`/data1/user/ymdai/ember_runs/fixed_b_output_range_20261006`。
旧PZ同规模消费者实耗0.301完整GPUh、留存峰约0.86GiB，作为本批预计依据。
预计含工程45–75分钟，硬限实际承接起2h wall、1.5完整GPUh、8完整CPUh、4GiB新增峰值，含加载/profile/失败/冻结树/临时输出。
预计明显超75分钟、GPU费用到1.1h仍未接近完成，或任何科学/配对/资源边界不满足时，一次回报具体剩余量；硬限不得静默突破。
创建root/冻结/输出前核strg01独立data1 quota、个人用量、共享容量及预计峰值；旧资产只读，新增全部data1。
每launch前live核两个节点、项目总占用及仓库8/6/单节点6上限；按实际吞吐选卡，可共驻，不抢占或dummy。
复用persistent dynamic evaluator；独立B20与闭环可并行。用固定授权事件验证较大物理batch，记录吞吐/显存及不再放大的实据；
只为测吞吐可重复已有固定模型计划，不增加新环境行、采样或科学臂，不把历史保守batch硬锁为上限。

工程沿canonical已有bank/FM读取/evaluator，新增仅限本批投影与注册所需的临时入口；
从clean pushed main隔离独占分支/worktree、必要实际消费者检查、集成push、clean detached frozen运行，不热改旧冻结树。
不恢复旧专用trainer/dispatcher，不复制长期实现；实际代码/读取来源与原训练来源分开保留。
工程接口修复在科学定义/配对/预算内自主完成，不由main重复工程验收；未知原因或科学语义变化回main裁决。
产物包括合同、投影bank/manifest、旧引用、160预测、32 raw rows、paired/aggregate、固定full图、失败/费用/退出和completion。
完成所有结果均退役本批专用入口/hooks，保留Git/frozen/原件；完整封口后只发一次整批消息并交回tracked/Git窗口。
直接等待进程退出，不定时心跳、自Queue或逐阶段回报。无自动Reader训练、新Writer、400、RL、Test或下一项分析。

# 教学中可见手轴在实际原生特征中的读出辨识

2026-10-06。唯一active分析合同；执行状态看progress，登记不表示已经运行。
这是冻结特征上的诊断学习，不训练Writer、policy或LoRA，不产生新的EMBER闭环成绩。

## 1. 具体问题、竞争解释与决策作用

findings§336–341表明：task39固定八条教学均可见提前转腕，而既有自身执行通常近竖直；
原训练task29已有大幅旋转及实际执行，task23成功反例又表明转腕不是全部失败的统一答案。
固定T内容的L/R学习未带来所需完整增量，不再扩大该读取族。新的分析问题仅为：
**现有教学原生读取，是否保留了RGB中可见的手轴方向及其逐帧变化，供后续Writer使用？**

竞争解释A：原生动作查询压缩后，这项具体可见运动量在最终H中不易跨任务读出；视觉前缀仍较易读出。
竞争解释B：H已经保留该信息，主要缺口在从它学习适用关系、写入和执行作用；增加视觉入口没有本项依据。
竞争解释C：两种浅读出均未学成或不能迁移，本诊断不足以定位信息丢失。
比较结果用于决定是否值得继续研究新的合法视频特征入口；不把可解码性当有益操作知识或完整修复。
axis只覆盖手的一个方向，不是完整SO(3)、抓持部位/接触真值或task39必要姿态。

与已撤回native_prefix_change_value不同，本项不创建dP、不替换prefix、不开展450 Writer学习/400评测；
旧合同是未完成科学检验，并非性能阴性。本项先检验由实际行为提出的一个具体特征见证。
旧Task-Grounded Visual-Value、VisibleObject、角色/crop及ProcessPullback负证据保持；即便K/V较好也不恢复这些方法。

## 2. 固定父模型、输入与两个实际字段

父为完整T2340：
`/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928/continuation2340/T/train/attempts/continuation/checkpoints/macro_00002340`。
source aligned1000、原normalization/tokenizer、父公共A0/B0、全部Writer参数冻结；不生成或评测新LoRA。
原state-free教学读取不变：exact language、真实双RGB/官方旋转及model224、stride5含末帧、probe1729的50×32、tau=1。
每帧只运行原有完整native一次，同时被动取得：

1. **H**：原`TargetWrite`实际使用的`hbar=H/sqrt(mean(H²)+1e-6)`，保留全部50×1024。
2. **KV**：最后原生block17的真实旋转后prefix K及原生V，只取前512个双图像tokens；
   每token拼接256维K与256维V成为512×512。保留其原生上下文化与位置编码，不声称是位置无关语义。

block17是最终H之前最近的原生视觉读取接口，不是层位搜索；不另取其它层或重跑不同probe/tau。
prefix经实际双向图文计算可依赖language，仍是合法条件；不把其语言上下文当额外监督。
复用`native._NativeFrameCall`与`NativeAttentionCapture`的实际observer语义，窄范围提取原字段；
不可直接调用会追加`response_difference`的prefix_change路径，也不为两个字段分别forward。
不保存38处X、Q、suffix K/V、大attention图或其它无用张量；内部原forward所需计算不删除。
K/V保存原生dtype；hbar按原Writer FP32计算/保存。训练头使用正常BF16/TF32/批量执行，不追低位一致。
这些字段的token数、宽度及输入投影参数量不同，比较的是两个实际接口的有限可读性，不是等参数容量定理。

## 3. 唯一数据与标签合同

仅原36个授权meta-train tasks，沿`configs/operator_read_write_v1/seen_task_scope.json`与原dataset
`data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a`，完整task/path映射可复用已审计training_tasks36.tsv。
全部仍在原每task demo0..49池；不扩任务、episode或source71。官方validation8/Test既不提特征也不读标签。

- 读出拟合32 tasks：原36排除[29,34,38,73]，各demo16、17，共64 clips。
- 唯一终点评估：原全部36 tasks各demo42、43，共72 clips；分32同task新video与四个读出未拟合task报告。
- 四task的选择来自已消费的行为对照：29大幅准备、38中等旋转、34同类杯竖直取物、73关门参照。
  它们只对诊断head留出；父T已在原36上学习，绝不称为父policy未见task泛化。
- 每clip所有stride5位置加真正末帧，标签对齐当前RGB的同一原始obs[t]，不套用action-query的offset1。
  CPU既有episode长度清点为拟合1,940帧、评估2,442帧，合计4,382帧；manifest以实际原源长度核对。

标签只在特征提取完成后的诊断head loss/评分使用：原HDF5 `obs/ee_ori`的axis-angle r_t，
`z_t=Exp([r_t]_x)e_z`，即世界坐标中的hand/site +Z单位向量；零角用连续Rodrigues极限。
依据前一审计hand/site轴一致，向下参考为(0,0,-1)，倾角`acos(-z_t[2])`。
禁止把ee_ori/state/action、task ID、时间位置、文件名、episode长、标签派生量送入head或native。
模型加载/样本索引可以使用task ID做分层与对齐，但不成为网络输入。
仅为被动分组可读取这些训练episode的actions[:,6]，登记首正命令前obs索引<c；c=0或不存在则该组为空。
这不是抓取/接触标签，不改变采样或loss。不得读取官方held教学的privileged信息。

## 4. 两个诊断头与固定学习

对每个frame分别读全token集合F，不先平均raw features/frames，不使用相邻帧或序号作为网络输入。
两臂同构：线性输入投影d→128并LayerNorm128；一个learned query经4-head attention读取全部tokens；
query加attention输出后LayerNorm，再接pre-LN的128→256→128 GELU残差FFN；线性输出3维并以eps1e-6归一化单位向量。
attention dropout0；无额外位置表/语言输入/任务embedding。真实字段原有位置和语言依赖保持。
输出weight为0、bias=(0,0,-1)；其余使用同seed7的标准模块初始化，共同形状下游参数用相同初值，
输入投影因d不同独立初始化并披露参数差。不得用一臂的训练权重初始化另一臂。

每臂恰好500 AdamW更新：lr1e-3、betas(.9,.95)、eps1e-8、weight_decay1e-4、clip1，无scheduler。
目标仅`mean(||zhat-z||²)`，32task等权，每更新每task16帧=逻辑batch512：
先在该task两个fit clips中均匀取clip，再在其stride5帧中均匀取frame；允许重用。
固定CPU sampler seed20261006，同一512-event列表供两个head；不按角度、阶段或结果重采样。
可拆物理microbatch但完整逻辑梯度/权重/更新不变；训练features全部detach，冻结父/source无梯度。
只有诊断head及其新optimizer训练；250保存恢复点，500为唯一评估点，不在中间看留出结果选点。
保存head/optimizer、sampler/RNG/cursor、物理拓扑/数值配置及父身份；不得将这称为Writer正式checkpoint。

## 5. 固定读回、反例与结果分支

只在500结束后评估一次全部fit帧与72评估clips，保存每frame z/zhat及来源，不需要新native。
指标按clip再task等权，完整逐task/clip表与不利例保留：

- 轴向角误差`acos(clip(zhat·z))`及倾角绝对误差，均值/中位数；向量MSE。
- 对原相邻stride5帧计算`(zhat_(t+1)-zhat_t)-(z_(t+1)-z_t)`的向量MSE；
  同时报逐clip真实/预测变化量和去均值后MSE，防止按task猜平均朝向被写成看懂动态。
- 全帧、倾角>30度帧、首正gripper命令前帧分组；低样本/空组原样报告，不删失败组。
- 无学习的竖直常量及仅fit数据的task/video等权全局平均单位轴基线；
  fit32新video另报本task fit平均轴基线，仅作评分参照。四留出task不能用其标签构造静态基线。

预先把5度绝对角误差差异及20%相对差异作为有意义量级参考，防止把小数改善包装成入口根因；不是显著性/控制门槛。
若H与KV在拟合、同task新视频、四task迁移和动态变化上接近，且明显超过常量，降低一般H入口丢失该信息的优先级。
若KV有上述量级且跨29/38等有效变化的优势、H head已充分降低fit误差而新视频仍差，
则提高实际前缀信息在当前H接口不易读出的支持度；必须同时报告34/73和全部同task新video的退步。
这仍受读出类、500更新、数据规模和父已见任务限制，不能证明H无信息，更不能宣称task39根因已确定。
若两头均弱、H拟合不成或优势只来自常量姿态，则保留优化/有限读出等解释，不追加层位、seed、宽度、采样或训练长度扫描。
无论哪种结果，整批停止交main；没有自动新Writer、姿态辅助、prefix改权重、dP、Reader、fresh/400、RL或Test。

## 6. 执行、预算与交付

唯一实验session01a10a98-6d4b-7d61-b12c-da38a628cb45接手工程/实际消费者检查/资源/Git；main只作科学消费。
唯一新增root：`/data1/user/ymdai/ember_runs/native_hand_axis_readout_20261006/`。
预计含工程1–2小时；硬3小时wall、2完整GPUh、8GiB新增峰，包含所有加载、profile、失败、缓存、冻结树与产物。
现有4382帧估计特征3–5.2GiB（按native K/V dtype），head/源码/临时余量<2GiB；先按实际dtype及首合法shard核算。
全部新产物在data1，开root/缓存前现场核strg01独立quota、相关个人实占及共享容量；复用canonical资产，不复制大模型。
每次launch现场核两节点和总8/空闲≤10时总6、单节点6规则，可共驻但不干扰他人，不附加无依据总2卡限制。
先以原定clips的同一批真实帧测frame_chunk16→32及可行更大值；head测物理batch实际吞吐/显存余量。
profiling至多每头两次可丢弃更新，之后恢复初始化/optimizer/RNG；重复提取测吞吐不新增科学样本，全部计费。
只按真实吞吐选择物理安排，能独立的提取分片可并行；不为填显存增加科学规模。
若预算预估无法完成，停止在可恢复边界报具体缺口，不能静默缩小数据/500更新或改变模型。

复用现有native owner和样本读取，不复制policy/trainer或增新canonical方法；必要的窄观察入口由执行者检查实际消费者。
clean pushed detached冻结代码执行；不原地改冻结树，工程修复不改科学合同并保留失败/费用。
整批退出后汇总manifest/输入标签来源、两字段与两head参数/更新、全部原预测及指标、head恢复状态、命令/费用/峰值/退出completion。
专用运行入口/hooks封口退役，保留Git/frozen/必要小原件与读出head；不删关键父模型或其它实验资产。
只为整批完成或真实科学/预算阻塞向main01a10a97-dedf-7042-a6a2-60214f0ef7b1回报一次并交回写窗口；
长任务直接等待退出，不阶段轮询、不同时再排自通知。main有持续自主推进授权，不把本批停止变成等待Owner逐项指路。

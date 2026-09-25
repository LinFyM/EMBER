# 任务分组lookahead是否改变有用学习信用：先检验计算前提

2026-09-25。是否active只看progress；机器合同`configs/metatask_lookahead_credit_v1/experiment_spec.json`。
主讨论01a0cd94-65da-7b22-8ca9-7ba35f454632；执行者为现有Sol 01a0cd90-ebb7-77a1-a20b-a858825d2f66。
本批是七组独立单步学习诊断，不是连续训练、候选模型选择或新部署评测。

## 1. 为什么先做这一小步

§143关系支持没有修复；§144将该端点差异主要定位到N之外的共享映射；§145又否定了
“新增76立即把Goal推向plate”及“保留77提供本窗口保护”的预定预测。不能接着给某个task贴固定有害标签。
共同更新怎样改变其它条件的计算作用仍未解释；单说参数共享、跨任务干扰或路径依赖不是根因答案。

待测机制是：普通FM训练只惩罚当前预测误差，没有显式训练“学一组条件后对另一组条件的影响”。
共享生成器可以通过改变条件Jacobian学习不同的迁移关系；训练期跨任务lookahead是干预这种关系的一种候选，
并不证明当前问题来自该缺项。这里先问它在真实EMBER图中是否形成可区分的学习作用，避免直接开长训练。

数学来源为[MLDG原论文](https://homepages.inf.ed.ac.uk/thospeda/papers/li2017metaDG.pdf)的训练内模拟跨域更新，
以及[Fish原论文](https://openreview.net/pdf?id=vDwBW49HmO)的梯度匹配与一阶近似思想。
EMBER不同task有不同控制目标，并非这些论文的同标签视觉域；外部结果不保证控制、视频增量或能力保持。
下面是本项目定义的对称预条件单步算子，不冒称逐项复现Fish或精确二阶MAML。

历史约束：§127删其它task梯度会丢掉有益迁移；§129只缩头未修复，private Writer也没有显著输给freeAB；
§130–131更大task batch的局部收益未变成fresh held收益。因此本批保留共享图与全部梯度，改变任务间学习信用的计算，
同时放入混合任务分组参照。旧FM、前5演示MSE、成功expert occupancy蒸馏均未稳定对应闭环；本批不能用这些代理宣布修复。
保留教学视频→原生读取→完整LoRA→冻结Source这一主线，部署无新模块、task ID、额外adapter或内循环。

## 2. 算子、假设与竞争解释

在同一父状态phi、m、v下，用四个合法任务的112个原查询定义原损失L0。
主21和辅7都是随机tau/fullH；每query系数均为1/84，因此L0=(4/3)*mean112。
将查询分为等量A/B，定义LA/LB为各自4/3倍均值；L0=(LA+LB)/2，g0=(gA+gB)/2。

- TASK分组：按原macro条件位置，A={0,2}、B={1,3}，每组2个完整task、56query。
- MIX分组：每个task的28query按`(query_ordinal + condition_position) % 2`分到两组，
  ordinal按主0..20再辅21..27；A取0、B取1。各组包含全部4task、各task14query，总56。
  这只改变分组，原teacher、actions/frames、tau/noise、loss总权重完全相同。

P为父Adam二阶矩的固定对角逆平方根，按父step1155/beta2偏差校正并使用原eps；source不在P中。
从父完整状态用原g0计算一次普通AdamW候选BASE，取排除decoupled decay的实际梯度位移d0。
每种分组定义

`alpha = ||d0||_2 / sqrt((||P gA||_2^2 + ||P gB||_2^2)/2)`，
`phi_A = phi - alpha P gA`，`phi_B = phi - alpha P gB`，
`g_look = [grad LB(phi_A) + grad LA(phi_B)]/2`。

alpha只用本case合法训练梯度与原Adam步幅确定，不用诊断loss、动作、奖励或held信息；没有LR/scale搜索。
它使两个虚拟步的均方参数长度与BASE的梯度位移相同，**不宣称实际功能步幅相同**，后者单独测量。
虚拟步不含momentum/decay，不更新optimizer/scheduler/RNG/cursor，P/alpha及内层方向均stop-gradient。
TASK和MIX各从同一父m/v及step1155应用一次原AdamW(g_look)，含原clip/decay，得到两个真实候选位移。
BASE/TASK/MIX都是独立的一步结果；下一case重新恢复父状态，不能串成七步。
零分母或零d0按spec记录为无可辨识虚拟作用，不补epsilon调参或擅自放大；保留相应BASE与零效应记录。

若梯度足够光滑、虚拟步在局部近似内，固定P/alpha有

`g_look = g0 - alpha/2 * (HB P gA + HA P gB) + O(alpha^2)`，
即局部近似`L0 - alpha/2 * gA^T P gB`的梯度。
它有机会改变不同task的更新兼容性；MIX也含一般曲率/重复样本效应，因此必须比较，不能只与BASE比loss。
有限步、BF16与实际Adam不保证此近似；记录真实虚拟前后损失和梯度，不用梯度cosine当根因。
本批不需要Hessian或二阶autograd，也不为符号形式加入昂贵高阶图。

H_transfer：真实跨task的有限更新确有互相伤害或收益不对称；TASK的修正与MIX有实质方向差，
在独立episode上比BASE/MIX保留更多有利函数变化，而非仅把所有更新缩小。
H_generic：TASK与MIX相近，变化主要是普通lookahead/曲率效应，不能归给跨task结构。
H_surrogate：FM迁移读数变好，真实10-flow前缀没有相应变化或反而恶化；此时FM兼容性不足以支持行为修复。
H_small：修正或实际函数变化太小/数值余项过大，当前单步未识别；不把阴性称为优化器或架构上限。

## 3. 固定父状态与七个case

父为C_S00@1155，路径、Source/normalization、数据协议、原seed和world2与§145相同；父训练7dc95edb只读。
使用原S00 `training_events.json` 的macro1156..1162七组，每组四个完整事件，恰好覆盖S00全部28task各一次。
每case均从父恢复，实际候选optimizer只做1155→1156一次；原macro只作事件来源，不伪造后续学习历史。
具体event index/teacher/query信息随机器合同登记。总784个不同原查询，各种梯度/虚拟读出复用它们。
事件由登记index显式构造，不在case2等用普通next_iteration冒充事件选择；原sampler的next macro仍是1156。
各case单列所用source_macro/occurrence/原query引用及一次候选更新，不修改或伪造父曝光记录。
所有新梯度限S00 fit28、episodes0..45；diagnostic-held8及官方Val/Test连query/teacher也不新增读取。
模型保持agentview教学、stride5、真实末帧、full H50、全Writer与三Meta活动、完整38-target rank16输出。

先完成case1作为正式工程pilot，计入七组。通过恢复/分组/虚拟还原/真实预测接口验收后完成余六组，
不根据pilot科学数值、某task难易或符号选择case。没有其它训练、工程rollout或环境初始化。

## 4. 独立读出与严格计数

每task固定teacher46、canonical diagnostic_batch(seed20260925,count16,teacher_demo46)，
query仅episodes47..49；冻结同一份query索引、RGB/8D执行state、future actions及随机tau/noise。
标签只用于合法训练任务的只读诊断，不进入Writer，不参与alpha、g_look或任何候选更新。
真实10-flow逐query的噪声seed使用现有task_logical_batch_policy_rng_seed，以seed20260925、task、visit0、
实际demo/frame的单元素列表生成，再以CPU Generator生成float32的50×32标准正态；与model/worker/batch无关。
这是专家query诊断的噪声身份，不伪称新的环境init state；FM仍使用canonical diagnostic_batch返回的原逻辑batch噪声。

每case八个参数状态：P、TASK_A、TASK_B、MIX_A、MIX_B、BASE、TASK、MIX。
每状态在本case四task各生成teacher46的完整LoRA，并对同16query算随机tau/fullH FM：
7×8×4×16=3584条无梯度FM读出，224套完整LoRA；同状态/task只生成一次并复用。
另为P/BASE/TASK/MIX在这16query的固定索引0..3做真实10-flow，
7×4×4×4=448次，使用同query的stateless初始噪声，输出full50×7 normalized/environment及前5真实OSC。
10-flow输入不能包含真实动作；动作标签只在输出之后计算误差。复用这112套LoRA，不另生成预测bank。

固定读出：
1. 分组身份/权重、原g0与A/B求和余项、两虚拟步/三实际位移的范数及方向、alpha/P/clip/Adam provenance。
2. 原训练组LA/LB在父及对应虚拟状态的实际变化；独立episode各task在全部八状态的FM，
   区分被用于虚拟梯度的task与另一组task。MIX两组都含全部task，不能冒称未见task迁移。
3. P→三实际候选的真实10-flow函数变化、前5×7误差；每query用
   `MSE_new-MSE_P = mean(deltaF^2) - 2*mean(deltaF*(expert-F_P))`
   分开方向与步幅，核对有限恒等式。fullH结果另列，不把normalized/OSC/实际EEF位移混称。
4. TASK−BASE、MIX−BASE、TASK−MIX的独立FM、真实前缀误差、位移能量与方向项；
   28task全表，按7个case成簇bootstrap20000/seed20260925，描述性95%区间，无checkpoint筛选。

这仍是专家观测上的代理诊断；单条expert动作不代表唯一正确动作，FM较小或前缀误差较小都不保证闭环。
不产生成功率/RGL，不把本批标签诊断叫新部署成绩，不声称视频必要性。

## 5. 裁决和后续边界

本批先回答：拟议学习算子是否在真实图上有效、是否具有真实task分组特异的计算作用、是否只改善原代理。
若TASK与MIX无可区分作用、只压小整体更新、虚拟步明显不在局部有效区或仅原训练query改善，
则没有据此投入长程meta-training的理由；报告未识别/否定的具体范围，不自动调alpha或更换父节点。
若独立FM变化与真实flow前缀变化相反，优先保留监督信用与执行函数不一致的解释，不能以梯度一致宣布成功。
即使计算前提积极，也只允许主讨论随后登记有限共享学习与配对闭环，检验保持、获取和有益视频；本批不自动启动它。
不以必须找到统一根因为前提，允许不同控制缺口有不同原因；这项小试不替代已完成§135–145的事实。

## 6. 实现、资源和交接

从最新main隔离开发，复用canonical Runtime/SupervisedEngine的FM/Writer VJP、checkpoint加载及真实10-flow路径。
允许受登记约束的per-query信用mask和诊断driver，不复制整个trainer或另写policy执行器。
分组loss保持每query1/42，补组梯度可由2g0−gA重用（须记录实际策略和正常数值余项）；
被mask的query仍保留原tau/noise/逻辑index。不能重新抽噪声或按缩小batch改变随机样本。
虚拟参数用保存的父张量还原，不能靠相减再相加累积舍入；不把虚拟步写入父optimizer或正式sampler。
适配后的视频激活不能跨虚拟参数状态缓存。world2全局求和、两rank相同参数状态和Source冻结须真实检查。
CPU验证loss权重、梯度mask、分组/补组及参数/Adam恢复；真实case1验证完整图、finite、合法输入与10-flow。
不为正常BF16低位差异扩dtype、固定batch1或做全tensor逐bit扫描。

所有新结果来自同一clean pushed detached提交。保存21个一步候选及28个虚拟状态的Writer权重/可重建delta，
标明weights-only诊断资产、不是完整续训checkpoint；父完整资产及source只读复用。
保留case输入/原event引用、实际命令/env/设备、各组信用与候选provenance、224bank、3584FM和448flow原件、分析及退出回执。
一个case完成就原子登记，恢复只补未完成case/读出，不重做已验收结果；不可还原的半case单列失败，不静默拼接。

study `/data0/user/ymdai/ember_runs/metatask_lookahead_credit_20260925`，data0新增≤6GiB、data1新代码≤768MiB。
预计正式计算约1–2 GPU-hours，硬上限4 GPU-hours；case1后以实测吞吐更新一次ETA/预算，工程开发时间另计。
world2在同一节点，正常优先2卡；预测可复用空闲卡但项目同时≤6物理卡，不跨节点拼训练。
新root前strg01独立quota和共享容量检查；每launch同时live核对两节点所有权/余量，不抢占别人任务。
正常进程一次持续等待退出，不轮询日志/cache。资源不能满足上限或核心科学/工程合同无法实现时保留原件并回报。
完成七组及全部读出后主动Queue主讨论，附completion/原件/缺项与实际资源，停止新增实验；不自动继续28步、闭环或新架构。

# 第二轮专家讨论：补充证据与仍未建立的事实

2026-10-07。回应[第一轮专家意见](EXPERT_RESPONSE.md)第十一节的资料缺口，供[深入推进提示词](FOLLOWUP_PROMPT.md)使用。
本次只导出已有非封存记录，没有新模型 forward、环境步、梯度、标签、选点或张量计算。
26份证据共12,053,698 bytes（约11.50 MiB），加[来源索引](followup_evidence/index.json)；原件的科学值、符号、行序及失败记录保留。
JSON压缩格式，服务器路径转为来源标识并移除worker/设备身份；三个既有PNG原样复制。
第一轮专家原文另行完整存档，不把其建议登记为已采纳的方法或active design。

## 1. 资料缺口的处理

| 第一轮指出的缺口 | 本次补充 | 仍然不能据此声称 |
| --- | --- | --- |
| 缺权重、中间张量与实际作用读回 | A64逐条件功能分解、修正消费者后的自身调用、全部操作单元分解、内容居中实验的逐项读回 | 没有上传完整权重或大张量；没有失败状态上完整teacher X/自身raw hidden的配对测量，也没有跨任务功能Jacobian |
| 部分关节、接触、连续行为未上传 | task23全部100条逐行统计、接触对、3条原有逐步时间线；task39全部100配对的200条回放统计与校验；post-open全部90条、首次动作差异、2张既有RGB sheet及映射 | 不是所有原始NPZ或每个物理积分步；接触采样只在控制步结束，不能代替全过程接触／抓持真值 |
| 可靠教师在多类接收状态的成功范围 | 既有NN接力与最新90条的完整正反结果已有入口；本次补90条详细记录 | 广泛有效的续行教师尚未建立；不能通过上传失败记录把它认证出来 |
| 同新增对应／续行数据的强MT及充分语言生成器 | 旧参照、已有对应数据的实际训练范围仍可查 | 这一完整公平比较尚未做过，不是一个漏传结果文件 |
| 稳定优势所需独立训练重复 | 现有节点及逐任务／逐初始化得失已在前包 | 相邻节点不是独立训练重复；本次没有新增随机性证据 |

现有资料足以继续提出和比较完整机制解释；不能把上述测量未知写成已经确认的原因。
若某项新材料会改变方法取舍，请专家指定最小对象：哪个checkpoint、layer/target、哪类teacher/query状态与配对条件，
需要观察什么量、区分哪两个解释，以及哪种结果会改变推荐。先分清本地已有但未上传、尚未计算、只有新实验才能建立三类。
不以泛泛索取全部权重或无限追加探针代替基于已有证据的独立判断。

## 2. 实际特征／算子作用

| 新增入口 | 覆盖及决策用途 | 必须继承的边界 |
| --- | --- | --- |
| [A64查询函数分解](followup_evidence/mechanism/a64_query_function_decomposition.json) | 8个条件；576细分、288任务汇总、72总体汇总。前5/full50、有效mask/unmasked、各动作通道和正负作用均保留。A的改变量共同项能量占14.14%；J更低离线误差未变成更强闭环。 | query变化同时含RGB、state、Gaussian与flow；不能称纯状态导数，能量也不是闭环收益归因。 |
| [修正消费者后的自身调用](followup_evidence/mechanism/role_self_call_corrected_consumer.json) | 101读回、25自身状态、16组、8换位配对；10flow的实体rho、实际attention mass、局部R作用及不利项。 | 采用`a019232a`消费者；旧整体autocast读回不作为证据。残留重放差、原闭环未重跑、局部减R不是完整removed-R策略等原边界都保留。 |
| [全部操作单元分解](followup_evidence/mechanism/role_swap_all_unit_effects.json) | 同8配对，每flow有18层×8头×5槽；包含key、自身调用及交互，避免只看有利均值。 | 对已有操作数的代数分解；单元不是独立样本，没有运行混合策略，不能把局部margin项当整条动作主因。 |
| [内容居中完整读回](followup_evidence/mechanism/role_centered_content.json) | 110行为行、62首次规划、640 B20记录、320配对差、25视频内容记录。成功22→21/55，R/G/L=16/5/6，原布局两臂均0/8。 | 任务12丢3、143/320查询恶化保留；前5误差改善93.78%来自gripper，内部拟合不等于控制修复。 |

这些文件可以约束“只差正确角色”“只差状态敏感性”“更低FM就能修复”的解释，但没有唯一识别失败根因。
已有保存的投影操作数（例如`a=A h`、`w=Rᵀd_key`）不能还原完整raw hidden；不能由本补包宣称已测teacher/query子空间或跨任务更新核。

## 3. task23：原命令回放的关节、接触与时间线

入口：[定义](followup_evidence/task23_drawer/definitions.json)、[逐行JSON](followup_evidence/task23_drawer/per_row.json)、
[逐行TSV](followup_evidence/task23_drawer/per_row.tsv)、[全量汇总](followup_evidence/task23_drawer/aggregate.json)、
[接触对](followup_evidence/task23_drawer/contact_pairs.json)、[序列摘要](followup_evidence/task23_drawer/sequence_summary.json)、
[既有关键时间线](followup_evidence/task23_drawer/key_timelines.json)、[有效性检查](followup_evidence/task23_drawer/verification.json)。

- 原T50/C50共100条，只有99条满足完整原过程解释准入。C900/init38为唯一未准入：原In在274步出现，回放273步出现，
  终局success一致仍不足以称逐步谓词匹配；保留其成功与全部记录，排除于原drawer过程归纳。
- 原T有9/50曾满足top Open，C有23/49；其中8/20条没有碗体原点至少1cm位移。不能把“开口出现更多”称为任务解决。
- 三条原有逐步时间线是T/init0、C/init0、C/init38，包含关节qpos/qvel、Open/In、EEF/gripper/碗位置与步末接触组。
  C38只作有边界的回放记录；不能用其新关节／接触过程解释原闭环唯一成功。
- 官方任务成功依原生In，描述中的Open是分析量；Open、接触、中心位移与抬升均不等价于抓稳、松手或完成。

## 4. task39：门碰撞干预的全量配对

入口：[读回与采样合同](followup_evidence/task39_door/readback.json)、[全量逐行JSON](followup_evidence/task39_door/per_row.json)、
[逐行TSV](followup_evidence/task39_door/per_row.tsv)、[原过程匹配检查](followup_evidence/task39_door/verification.json)、
[炉腔参照检查](followup_evidence/task39_door/original_cavity_reference_check.json)、[全部配对距离图](followup_evidence/task39_door/all_pair_distances.png)。

100条原命令分别在parent与door-noncolliding物理设置重放，共200条；parent对原轨迹／谓词匹配，第二臂改变碰撞而固定原命令。
两模型在第二臂均0/50 In；多数最近／最终炉腔距离变差，同时保留9条变近的反例及接触时序统计。
它只检验`x_next=F_collision(x,u_recorded)`，不是让策略在改动物理中重新观测并自适应的闭环试验，不能据此排除全部碰撞相关控制问题。
完整JSON补充接触对、区间与关节统计；PNG概览不代替逐行不利例。

## 5. 最新跨任务post-open接手

入口：[全部90条JSON](followup_evidence/post_open_transfer/per_row.json)、[TSV](followup_evidence/post_open_transfer/per_row.tsv)、
[全量矩阵](followup_evidence/post_open_transfer/summary.json)、[首次实际动作作用](followup_evidence/post_open_transfer/actual_first_plan_effects.json)。
九个原T实际post-open状态，自身M23及四个M42条件，各交叉L23/L42，十臂均0/9。
六个剩115–240步的状态也全部失败，三个剩≤40步的状态保留；不是90个独立状态，也不是新初态paired400。
这里是实际独立闭环续行，区别于上面的固定命令回放；原始task23环境、完整38-target单套LoRA及文本交叉的配对合同保持。

两张图沿用上一轮main已经查看并引用的样例，没有为本轮重新挑选有利样本：

- [自身M23+L42，init19图](followup_evidence/post_open_transfer/rgb/T2340_state019_M23_L42/contactsheet.png)及[来源映射](followup_evidence/post_open_transfer/rgb/T2340_state019_M23_L42/mapping.json)。
- [M42 demo09+L42，init8图](followup_evidence/post_open_transfer/rgb/T2340_state008_M42_demo09_L42/contactsheet.png)及[来源映射](followup_evidence/post_open_transfer/rgb/T2340_state008_M42_demo09_L42/mapping.json)。

各图8个实际replan、双相机，最后实际RGB在295步，数值终点在300步，没有terminal RGB。
这只上传原20个full记录中的两张sheet（原面板另有70条compact）；不宣称全部950个replan的图像已上传或审阅。
局部抬碗依然未形成In，自身M23配L42也有抬升，不能把全部局部变化归因于donor新知识。
这批降低“四个现成donor已能在这些接收状态完成控制”的支持，不证明接收状态不可救、所有primitive不可组合或更大Reader必需。

## 6. 第二轮判断的落点

补包不会把第一轮候选“自由完整A/B＋成功续行监督”变成已证方案：完整P/Q的共享失败、已有成功占据域、learner-state监督与P/I对应数据仍须共同解释。
专家应继续追问：这种改变为什么让合法视频学到强MT与语言条件生成器没有的未见任务适配，原训练为何不能学到或保留它，
以及该解释怎样同时约束实际行为、特征算子、学习条件和完整方法的选择。

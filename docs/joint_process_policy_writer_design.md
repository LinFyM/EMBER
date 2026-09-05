# EMBER：共享学习定位与 Joint Process–Policy Writer

状态：2026-09-05 23:24 CST owner完成对齐、要求建立goal并持续自主推进；本文件现为active design。当前先执行同图学习对照，
P/Q主干与非对称读出是依证据实施的候选，不是已验证结论。旧Unified v4及其正式结果保持sealed。

专家原文：`docs/expert_review_20260905_full_history_joint_process_policy_writer.md`。原文按owner提供的附件保存，不改写其观点；
它的建议、证据限制与下列实施裁决须分开解释。长期目标与边界以`docs/current_owner_requirements.md`及`AGENTS.md`为准。

## 1. 目标、授权与证据责任

最终目标仍是exact language + action-hidden ordered videos，一次Writer调用生成唯一完整38-target rank16 LoRA，冻结source
闭环执行；validation8 correct严格>145/400且通过相邻稳定、breadth/suite、same-task鲁棒性及最终视频因果资格。

已获准的接管范围：在现有科学、数据、存储、Git与GPU边界内自主做实验、深入分析、相关修复和有依据的模块重构。两个节点合计不超过6张
物理GPU；单job在一个节点。外部联系、改变目标/信息墙、新增数据或资源权限、以及证据无法裁决且显著改变投入方向的路线歧义，
仍须携具体证据与推荐选择回到owner。不得把普通负结果或例行实现变成新的审批节点；本次goal由owner明确请求建立。

性能低于或接近baseline只证明该条件下没有有效增量，不自动识别根因。对每个实质修改，先列出竞争解释、已有反证、最近历史等价
尝试及本次改变的主要变量。必要时多做分析实验；不能凭attention mass、rank、cosine、梯度非零或loss下降就宣布解决问题。
“根因”只用于有直接工程合同证据，或有可区分竞争解释且实际改变功能/行为的干预证据的适用范围内。

不以任意分析次数/版本数作上限，也不靠无新证据的seed/LR/width/scale小扫延长路线。局部机制与整体行为矛盾时保留该矛盾，
定位失败接口或重新审视解释，不用新术语覆盖负结果。每次新干预应能改变后续选择；不把所有局部机制完美通过设为进入闭环的前提。

## 2. 为什么先做同图学习对照

近期task-local控制冻结evidence、训练factor Writer并增加task query；shared训练whole Writer且无该query。v4每task只有约8–9次
exposure。旧正控不能单独证明部署图的单任务学习，更不能把shared non-pass直接归为任务泛化失败。

使用同一个现有shared runtime及部署图，比较单任务配置的独立clone与同一小集合的共享配置。独立clones只作诊断，绝不作为
部署dictionary、held adapter选择或checkpoint union。它们合计参数量较大，因此两组差异不能直接识别为梯度冲突或证明P/Q必需。

### 首轮实验合同

- 首轮锁定task1、72、83、93；task83是Goal20“打开中层抽屉”，用于补充程序类型且有已存在的29/50专家参照，
  不依据本轮结果选任务。精确语言、视频和首轮闭环名单见`configs/pi05_ecp_prw_samegraph_panel_v1.json`。
  首轮局部closed-loop为Spatial2、Goal20、Long38；meta task1先读functional。task93不能单独承担functional→behavior判断。
- K1；每task两条不同fit videos及至少一条same-task held video。视频与action query跨episode，Panel-B和held视频零梯度。
- 两组相同模型拓扑、component初始化、初始化seed、可训练模块、输出rank/cap、loss reduction、normalizer、每task视频/row序列。
  whole Writer训练、source/capture/carrier冻结、无task query、无Action Meta。未声称继承完整G2动态功能。
- shared每个逻辑update包含全部小集合任务，先各task action rows求均值再等权平均；clone每次只更新本任务。
  记录optimizer cadence、loss normalizer和实际task权重，不用“曝光相同”掩盖优化语义或总模型容量差异。
- 首轮在每task8、32、64 exposures读取曲线，每次8 action rows；64是预算节点，不是函数类不可学习的阈值。
  原始training loss与严格paired评估分开；训练日志的8-row generated减16-row carrier不作为收益证据。
- 记录fit、新action episode、新teacher video上的真实policy functional，以及同条件局部closed-loop；对照source/carrier及可用的
  直接可比局部参考。closed-loop任务、状态、video ordinal和RNG配对在launch前锁定，不挑video或事后挑task。
- 此阶段是train-side学习定位，不能以small panel选最终模型；task6/79若读取，只是已消费的诊断，不标为fresh selector。
- 不使用wrong/shuffle/reverse损失或架构诊断，不消费固定validation/test产生设计梯度。首轮只声称K1。

### 结果分支

| 结果 | 支持范围 | 下一项应回答什么 |
| --- | --- | --- |
| 无query clone也无有意义学习 | 当前输入/读出/优化/监督组合有缺口 | 先区分这些解释；有证据时matched检验非对称因子，不直接怪shared泛化 |
| clone明显能学、shared较弱 | 共同学习有困难 | 核对任务权重、实际更新与表达能力；P/Q是候选干预而非唯一解释 |
| fit成立、same-task新视频失败 | 条件过拟合或正确视频覆盖不足 | 核查采样与过程读取；不能以更多task数量代替 |
| 同任务迁移成立、task-held失败 | 共享规则迁移或任务/过程覆盖缺口 | 分析授权任务覆盖与表示迁移，再决定扩任务或换主干 |
| functional成立、真实闭环仍弱 | 离线目标/访问状态/执行影响可能不匹配 | 在实际行为证据支持时研究learner occupancy或成功行为保持 |
| 现有shared图产生可保留行为增量 | 训练剂量或旧正控错配的重要性上升 | 优先验证/扩大该能力；不为了实施P/Q而放弃它 |

继续训练须有有意义的功能或行为学习趋势及相称成本；绝对值长期接近baseline且成功集合不积累时，不能只靠内部下降续跑。
反过来，短预算没有通过也不自动否定整个函数类。必要的对照或额外分析依据竞争解释决定，不预设修正轮数。

### 首轮裁决与第二项：固定面板覆盖对照（2026-09-06）

首轮完整18条件表见`runs/analysis/pi05_ecp_prw_samegraph_local_b89ee997_20260906/comparison.md`。shared4 fit39→44、held40→41，carrier38；
发生实际学习，但增益与保留仍弱，Goal同视频到新视频的成功集合保留低，零梯度task6/79仍负。clones并非所有任务都强，因此不能把shared差异
直接归因梯度冲突或由此宣布P/Q必需。更广task覆盖、条件化表示/读出、有限action-query覆盖和离线目标到闭环的偏差仍是竞争解释。

审计首轮每task512次row使用只覆盖8个action episodes与115--126个不同episode/frame；每fit视频约60--64个不同rows，交集2--7。
固定16组policy RNG和visit循环还让两fit视频关联不同固定row子集。当前先用最直接的matched对照检验这一有限监督实例：

- 只训练一套shared4，task1/72/83/93、component-init、seed20260905、whole Writer、原64 updates/8 rows与32/64闭环保持不变。
- 每task occurrence从其原Panel-A全部16个授权action episodes中无放回抽8个episode，再各均匀抽一个真实frame；outcome-independent随机源
  只由固定seed、task与occurrence确定。原Panel-B、teacher videos与same-task held视频的episodes继续排除于梯度。
- video schedule与functional policy-noise seed逐occurrence沿用首轮，frozen functional normalizer复用原完整Panel-A定义，optimizer/lr/clip不变。
  因而本次主要变量是action observation/query覆盖及它与视频的固定循环关联；不同时增加task、更新数、K、视频数或改变模型/读出。
- checkpoints8/32/64照旧；原Panel-A继续提供episode池、normalizer与noise visit，Panel-B仍只读相同前8 visits作functional诊断。
  局部closed-loop仍为32/64、first-fit/held各strict150，
  复用首轮source/carrier配对基线。固定validation/test仍不消费，wrong/shuffle/reverse仍不执行。
- 本对照不选择最终checkpoint。若相邻、跨视频有可保留的实际行为增量，优先保留该能力并再审任务覆盖；若没有，不靠同设置延长训练，
  结合clone边界进一步检查当前native读出的可达性、监督到行为的差距或P/Q责任结构。负结果只排除本次小预算覆盖干预的收益。

## 3. 有条件的P/Q主干：模块与因果作用

```text
exact language + K internally ordered videos
  -> frozen per-frame source capture
       language/image prefix; H[probe,layer,horizon]; residual; velocity; raw X/Y
  -> each video independently:
       task-grounded process states P[frame,M,d]
       whole-policy states Q[target=38,rank=4,side=2,d]
       repeated learned block:
         P reads same-frame complete native responses
         P exchanges information along teacher time
         Q reads P; Q coordinates across targets/ranks/sides
         P reads current Q before its next native-response read
  -> permutation-invariant learned set read over videos
  -> frame-conditioned queries jointly select raw current-video X/Y candidates
  -> complete target factors, cap, one materialization
  -> carrier12 + mobile4 = one complete38-target rank16 LoRA
  -> frozen source closed-loop; no further Writer call
```

### 原生证据与P

Gemma图文prefix提供任务对象、关系与目标grounding；Action Expert的完整动作生成响应是视频过程理解的核心证据。
layer states保留`[T,2,19,50,1024]`，flow velocity保留原生padded动作轴；source响应不是teacher真实动作，也不是正确方向的真值。
teacher time、relative action horizon、flow time、layer depth与probe分开编码，不设`t+h`或把层深当任务阶段。

P每帧包含少量learned工作tokens。可先以`d=128,M=8`作为matched实现起点；M不是手工事件数。语言条件化patch读取先grounding，
随后在同帧完整layer/probe/horizon上attention。首次horizon压缩发生在task-conditioned learned read之后。
P沿视频真实时间交换信息；保留每帧状态，不先压成一个固定Program tuple、最后一帧或独立冻结的坐标瓶颈。

### Q与联合block

Q初始参数只携带公共target/rank/X-Y-side身份；没有task ID、每task专属参数或teacher动作输入。Q在38处参数写入职责之间
直接交换信息，并读取P；P下一层读取Q，继续从原始响应取证。所有learned模块联合接受同一个真实policy functional梯度。

最终因子选择仍与当前视频的具体frame/native candidates对应。禁止把一个全视频动态query广播后当作已经解决frame ownership。
native X/Y的owner保持明确，但过程理解不会从第一层就变成38个互不直接交互的factor问题。

使用标准attention/MLP同构block，不添加连续summary/solve/whitening/transport/anchor/gate链。P/Q负责不同工作，
不是强迫所有输入成为一个巨型token序列。较小的P/Q常驻，完整native evidence分块只读重放；不截断horizon或长视频换吞吐。

### 动态Value与跨视频

语言/静态context可以定位与条件化读取，但不能独立打开mobile residual。位置只作用于attention路由，不作为可伪造动态的Value。
静态重复输入不能仅凭位置/长度产生mobile；需要在具体block与readout实现上验证这一合同，不能由图示假定成立。

每条视频先独立保序计算P/Q，集合阶段对learned states置换不变读取，再对所有授权视频的真实native candidates联合生成一套LoRA。
不平均frames/raw features/最终LoRAs，不拼视频时间轴，不重复两条视频凑K4。后续声称dynamic K须真实覆盖相应cardinalities。

## 4. 输出参数化：与主干分开检验

首轮对照保持现有12+4与读出。专家指出当前两侧创新近零时`A(D)=O(D),B(D)=O(D)`，所以`BA=O(D²)`；
这是条件化局部推导，不是已证明v4训练停滞的根因。只有直接数据支持相关学习困难时，进行同预算、同主干的matched对照：

```text
A = A_context(current video bank) + A_dynamic
B = B_dynamic
static innovation = 0 -> B = 0 -> mobile BA = 0
```

A_context仍来自当前视频native X signed pooling，不是固定global A或任务字典。保持q native-head、action-in native-width grouping。
判断它是否有价值看真实functional、新视频和闭环，不看梯度变大。不能同时替换P/Q主干、因子参数化和训练目标后归因。

完整rank16是唯一部署输出；12+4是首轮隔离变量的选择。释放全部16个rank须有matched可达性/功能/行为证据，不能部署12+16。
训练raw factors与正式canonicalization的差异须按实际合同解释；不为正常BF16低位差异开启dtype/rank/seed挽救。

## 5. 训练、能力保持与数据识别

第一监督仍是正确视频生成LoRA后，同task独立episode的真实action flow loss。授权teacher actions给出训练方向，
source response给出表示证据和写入坐标。获得共享行为增量后，才按实际失败研究成功行为回放/保持或learner访问状态上的可信纠正。
这些属于历史相关方法的有条件重试；先核对SEOD/GOMQ/guard等价条件，不把expert occupancy重新包装为全新方法。

同时只在授权train侧核对过程/组合覆盖和功能歧义：同语言、执行观测和静态条件下，正确视频的过程信息是否改变需要的功能。
更多task、静态零mobile或动态Value约束都不自动证明视频必要性。元数据审计只是入口；不新增人工process数据、不用negative controls
指导架构，不从held outcome构造训练route。若现有数据无法识别最终主张，须如实向owner说明这一科学取舍。

component-init与fully-random是同拓扑的正式候选；初始小对照先固定component-init，避免同时改变主要变量。
不把部分投影继承宣称为完整继承G2能力，也不要求random图先通过G1--G3冻结课程。

## 6. 行为裁决与实施所有权

small train-side panel服务定位；有信息量节点及时回到预注册validation8 strict paired400。相邻点报告逐task/suite、breadth、
retained/gained/lost、churn count与rate、success-set overlap。历史不同teacher schedule/rank物化/评测panel只能按实际可比范围引用。
最终selected checkpoint冻结后才运行完整controls；固定test8保留到方法冻结，不靠union或融合达标。

正式低churn/稳定性数值标准在selector前依据评测合同登记，不能看到结果后定义；不机械恢复旧lost≤10或所有旧成功零损失硬门。

首项对照优先通过配置复用`scripts/train_ecp_policy_response_writer.py`和既有shared runtime，禁止复制第二套训练器。
若现有校验误把历史role batch固化为科学合同，做最窄的配置泛化并验证真实任务均值/采样序列，不改变实验权重。
只有首项证据支持P/Q时才替换learned主干；当前v4作为该比较的冻结Git参照，替换后从active树退役，不保留默认fallback。

formal train/eval来自clean pushed commit的detached frozen worktree。启动前同时检查两个节点与独立quota、选定设备/NUMA，
实测最长条件的执行吞吐与峰值，记录一个复用的launch合同。分析和文档不得阻塞已有有效科学条件的实验。

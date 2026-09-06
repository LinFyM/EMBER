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

### 覆盖对照裁决与下一项只读定位（2026-09-06）

四个完整strict150给出fit41→41、held39→45，carrier38；held相邻有7 gained/1 lost，但fit有4 gained/4 lost。
64时Goal跨视频只保留3/7，Long只有held出现3次，两个复用task-held功能点仍负。因此有限query覆盖解释只得到局部支持，
不能据此宣布已经稳定保留跨视频能力，也不能把混合结果解释为整个状态覆盖假说失败。

当前不增加task/训练剂量，先限定专家§4.2双侧动态零因子的解释范围：在现有三task、已注册first-fit/held正确视频、component初始化及
32/64 checkpoint下，只读记录真实context/innovation、signed两支均值与差、raw BA/cap与实际cap factor。使用未改变的完整真实视频，
不做动态缩放、wrong/shuffle/reverse/no-video，不读取actions/reward，不产生梯度。这个诊断不会选择checkpoint或以norm评价行为。
如果实际运行远离近零区域，就不把局部二阶推导当作当前non-pass的依据；如果两侧差确实接近零且相应功能影响受限，只支持开展
同主干/预算的非对称读出对照，仍须由新视频及真实闭环验证。任务覆盖、过程条件化与监督到行为偏差仍是竞争解释。

### 实际读出定位与matched非对称A对照（2026-09-06）

只读18个完整forward已完成：3 tasks × first-fit/held × 初始化/32/64。64-step held在38 targets上的常规中位数为：
Spatial A/B signed差相对各自两支均值RMS为`.0319/.0087`、raw BA/cap为`.0298`；Goal为`.0097/.0071`、`.00634`；
Long为`.0367/.0147`、`.0472`。各自只有7/4/8个target被cap，初始化更小。完整值见query-coverage analysis下的
`actual_readout_probe/{result.json,summary.json,summary.md}`。这支持双侧接近零的局部解释有实际适用区间，尚未识别闭环性能根因。

下一项只检验专家§5.6的A侧context职责，采用与fresh-query shared4匹配的单一共享训练：

- 相同task1/72/83/93、K1两fit视频、64 updates、8 action rows、原fresh Panel-A query/noise/video序列、normalizer与AdamW/clip。
- 相同38-target、whole Writer、component-init、4个target-local blocks、native grouping、full50与rank12+4/cap/一次物化。
- A侧head由`[C,D+,D-]`变为`[C+,C-,D+,D-]`，context的两支独立，当前视频raw X仍是唯一A Value；B保持原shared-base+dynamic offsets。
  这是对共同context与innovation的联合signed read，并非在pool后人为添加第二套factor或固定global-A。
- 保留旧A正支context和两个dynamic投影的初始数值；新负支context独立标准初始化，并隔离其RNG抽样，使所有共同参数和后续随机序列匹配。
  仅增加16,384参数（约0.55%），fresh optimizer/scheduler，不跨shape恢复旧checkpoint。
- 静态innovation为零时B仍严格零，因此完整mobile为零；允许A非零。CPU既有静态墙测试覆盖两条件，真实四task/long-video profile
  检查full50 functional VJP、所有学习模块和唯一rank16物化；不以synthetic static测试充当视频因果分数。
- checkpoints8/32/64与Panel-B保持原口径；32/64 × first-fit/held仍各strict150，复用已完成的fresh-query、carrier/source原paired rows。
  新analysis单独建根；不消费validation/test或negative controls，不以此train-side面板选择最终checkpoint。

若同预算改善绝对行为且保留跨视频/相邻成功，优先保留该读出再审共享规则与任务覆盖；若只有factor、梯度或functional变好而行为不提高，
则否定该干预在本条件下的行为价值，不能据此再扩大scale或cap。P/Q与occupancy仍按各自实际证据分支，不同时改变。

实现只在现有factor generator中以`input_context_branches=1/2`表达当前两臂，无新trainer/module或独立运行路径。主agent负责本次对照后
依据裁决保留唯一选定参数化；另一臂由frozen Git/config/results保存，不长期积累配置分支。

该阶段同时完成已授权的执行优化：static task adapter原未接入evaluator的batch接口，实际对各env逐条forward。补齐同task批处理后，
真实Goal/Spatial吞吐约提高2.9倍；仍沿用原逐env noise与全部50步推理，接受合同允许的正常BF16 batch差异。此执行变化与科学A侧干预
分别登记，所有分数注明actual evaluator commit；不以浮点低位一致为由保留低效逐env路径。进一步torch.compile不混入本轮首批评测。

批处理数值差异可能影响小幅行为增量的归因。因此，在非对称闭环结果收齐前补充同执行方式参照：复用fresh-query symmetric的64-step
first-fit/held两套已sealed LoRA banks，以`387f6d0b`的eager batch入口各重放strict150。状态、逐env RNG、视频与所有预处理仍匹配，
不重新训练或物化，不选择新视频。原serial结果保留作历史；A侧64-step干预的直接比较使用这两个batched参照，执行差异单独报告。
32/64非对称相邻稳定仍使用同一batch入口。若判断依赖32-step横向差异，再补对应参照；不把历史serial与batch之间的细小增量当成
纯A侧因果效应。补充参照仅属既有train-side panel，不消费validation/test或视频负controls。

### 非对称A完整裁决与正确视频覆盖对照（2026-09-06）

四个非对称strict150及两个同eager batch参照全部完成，900新行、10个历史参照的完整分析见
`runs/analysis/pi05_ecp_prw_samegraph_asymmetric_20260906/comparison.md`。64时对称fit/held为41/44，非对称为44/45；
非对称相对参照的R/G/L分别36/8/5与36/9/8，增益较小，held的1个净增伴随17 churn。非对称32→64的fit43→44、held41→45，
churn13/18；Goal fit只保留1/5，Long fit2→0。64跨视频R/G/L为39/6/5，Jaccard.78高于对称.70，但32跨视频只有.647。
因此没有形成稳定的整体能力，不能把实际近零诊断或Goal功能收益翻倍当作已识别根因。

下一对照以闭环略高、64跨视频重合较好的非对称A为起点；这是局部工作参数化的选择，不是已证明机制优势或最终checkpoint选择。
不因meta task1的功能指标转负而接受闭环较低方案，也不掩盖其负证据。主agent移除对称A的runtime选择分支；其checkpoint、配置与
行数据由原pushed Git和formal artifacts保存，新的canonical初始化严格沿用02a85314的已测A2初始化与后续RNG。

当前最直接尚未检验的变量是同task正确视频覆盖，而不是新任务数。原面板为task1/72/83/93分别提供7/12/13/12条与Panel-A/B
episode隔离的视频，当前仅两条进入训练。元数据已逐文件核验并保留为`available_correct_video_coverage.json`，但它不证明功能歧义。
新的单变量对照采用：

- 相同四tasks、whole Writer、非对称A/B、component-init/seed、rank12+4、完整50 horizon与全部读出/normalizer/loss/AdamW/clip。
- `fit_pool_max:2→4`，仍K1、64updates/8 action rows，每task每条fit视频恰16次；不增加更新数、动作行总数或改变task权重。
  task1为[5,6,16,20]，72为[3,8,14,25]，83为[5,7,13,15]，93为[2,3,5,8]；原held39/49/49/48不变。
  采样仍由task occurrence驱动，query pairs与policy noise不依赖视频池大小，逐occurrence与上一轮匹配。
- 额外两fit视频均来自原`program_video_demos`且与原Panel-A/B actions隔离；不制作数据、不读取新task梯度或改held身份。
  既有诊断tasks6/79的fit池也按同配置扩展，但仍零梯度、held视频不变，不把额外诊断fit均值与旧两视频均值混作直接比较。
- checkpoints8/32/64与Panel-B visits不变。functional同时报告四fit平均与原两fit平均；first-fit/held的32/64各strict150保持同一
  状态/video/RNG/387同等eager batch执行，以刚完成的非对称两视频四条件作直接参照。新分数仍不选最终模型。
- longest仍为task93 demo3的87个stride5帧；task1新增demo20为66帧。启动前复用已有真实profile并补当前4-video调度的真实检查，
  存储需覆盖更大的单份共享cache。训练和评测仍来自clean pushed detached authority，不延续旧optimizer。

若同预算增加正确视频覆盖能同时改善跨视频和相邻行为，保留该能力后再审任务/过程覆盖与真实K训练；若没有，就不继续增加同task
视频数或无新依据续训，而以已排查的读出、动作查询和视频覆盖边界重审过程表示及离线目标到行为的缺口。P/Q、occupancy和新增task
仍是有条件候选，不能仅凭本地非通过就宣布其中一个必然正确。

### 四视频裁决与P/Q matched主干比较（2026-09-06）

四视频32 fit/held=`41/38`、64=`41/39`，原两视频A2为43/41、44/45。64两→四的fit/held R/G/L为37/4/7和34/5/11；
四视频相邻fit35/6/6、held32/7/6，64跨视频33/6/8。Goal held只保留2/4，Long fit两点1/50不重合、held均0；没有稳定增量。
完整600新rows、原两fit与新增四fit的分开functional口径、所有配对和launch evidence保留在four-video analysis，历史§169。
本轮sealed，不增加同task视频数或直接延长它；后续比较回到闭环较高的原两fit数据条件。

当前证据没有证明shared必然弱于clone，也没有识别梯度冲突或occupancy根因。固定query覆盖、非对称A与视频覆盖均只给出有限或负的
行为结果；当前代码又明确把response按owner选层，并把全部learned blocks置于target循环内，因而没有共同过程状态或跨target反馈。
P/Q是对这个职责分配的有界方法比较，不是声称更多attention必然有益。最近历史LMMPC的全slot/axial交互不成功，旧Process–Composer
还有冻结坐标handoff问题；本次检验的是共同过程/整策略联合梯度、反馈后的原生证据重访和最后frame/native对应这一整体主干。
不能把更多参数或初始化差异单独归为反馈的因果优势。旧SEOD、F2 fixed-occupancy与Stage1 learner-panel事实已按历史索引复核，
不把expert occupancy重新命名为尚未尝试的突破。

本次实现与实验合同：

- 唯一主要变量为learned主干。直接参照02a85314训练、387f6d0b评测的A2两视频四结果；task1/72/83/93、原两fit/held视频、K1、
  64updates/8rows、fresh Panel-A query/noise、等task权重、normalizer、Panel-B、rank12+4、A2/B1 signed pooling、cap与loss不变。
  读出和训练目标不再附加新变换。component-init仍只继承可对应的G2 projection/attention/结构embedding；新结构fresh，不能exact-resume，
  新learned状态/层随机初始化造成的权重差异与参数量单列，不能声称所有异构参数逐元素匹配。
- `d=128, heads=4, blocks=4, M=8`作为本次固定实现，不做width/layer sweep。响应tokenizer一次投影完整19 boundary states、18个
  相邻layer residual、两个antithetic probe的noise/velocity，保留全部50 horizon和独立layer/channel身份；移除按38 owners重复投影及
  从第一层即owner-local的语义切分。even/odd两通道是双probe的可逆组合，不平均horizon或teacher frames。
- 每条视频独立保留P[frame,M,d]与Q[target,rank,side,d]。同一种重复block中，P先读上一层Q反馈，再用exact language条件化patch read、
  继而读取同frame完整response；三个来源各自softmax。P按work-token沿真实teacher time attention/MLP；Q读取全部P、在38target/rank/
  side间attention/MLP。下一block重新读取原始projected evidence，所有learned模块共同接受真实policy functional梯度。
- teacher-time位置只进入temporal attention的Q/K，Value不加时间位置；同一work-token在静态重复视频中的Value相同，因此不能只由位置
  伪造frame innovation。不同work-tokens的交互由共同Q读/反馈承担，不把时间轴与50-step action horizon混为一轴。
- 跨视频阶段以共同query对各视频learned Q做置换不变set attention，不输入video index、不拼接视频时间轴。随后共同Q按真实frame读取
  该视频P，一次生成全部target的frame queries；最后各target在当前frame的side-matched native X/Y上attention/MLP，保留已验证的
  bank-local非线性方向形成。learned frame factor states仍按每video拆成context/innovation，原A2/B1 signed head和全候选联合pooling
  只生成唯一完整LoRA。不得平均raw frames/features/最终LoRA。首轮只声称K1，K2/K4需以后真实训练和资源验证。
- GPU实现从开始即按frame批量处理prefix/response和共同P/Q；38个target只在最后native bank ownership/维度不同处保留分块循环。
  小P/Q常驻，完整native bank分块重放，使用need_weights=False的attention与既有checkpoint/VJP机制；不截断长视频或horizon换吞吐。
  必须真实profile最长87帧、报告Writer forward/backward、完整训练update时间与峰值，不能从token数估算宣布加速。
- 复用现有process/composer/model/training owner：process承接unpooled tokenizer与共同block，composer只留native bank/frame readout与
  orchestration，退役旧target-local block，不保留并行默认fallback。保留原pushed Git/checkpoints/raw rows作为精确参照。
- 正式仍保存8/32/64和完整resume状态；32/64 × first-fit/held各strict150，状态/RNG/video/官方执行匹配。新参数、时间与初始化复用范围
  单列。若只有内部loss/attention/梯度改善而闭环更弱，不能以机制解释选择它，应恢复较好的基线并重新判断剩余接口；不追加scale/cap/
  rank/seed小扫。若广泛且可保留行为改善，才扩task/真实K并比较fully-random，及时进入已补工程入口的validation8预注册资格。

### P/Q最终裁决与后续范围（2026-09-06）

本实例已完成并sealed：32fit/held38/37、64为41/39，四组均低于两fit A2的43/41、44/45。P/Q相邻held更稳定但低分，Goal下降、Long0；
fit Long成功相邻不重合，64跨视频Jaccard.667低于A2的.780。训练339.88秒相对781.62秒约2.30倍加速，不抵消行为损失。
实际训练query/noise配对也显示后半四task loss均高于A2；不能只归为新视频泛化，也未识别优化、容量或Q反馈的单独责任。完整证据见
research_history§170与`runs/analysis/pi05_ecp_prw_samegraph_joint_process_20260906/`。

按上述预注册分支恢复A2工作主干，不继续本实例、增rank/scale或做seed小扫。P/Q原代码/配置由93540ff1及formal artifacts保存，
active只保留A2；原attention block归入process、shape校验独立，计算与参数树/初始化保持。以下P/Q模块描述是该已检验候选的设计论证，
不是当前默认路径或继续它的授权。

首轮同图已证明有限shared可学习与净行为收益，但新task/跨视频稳定仍未成立。此前73-task v4只有约8次或更少per-task exposure，
不能充当充分meta-training的裁决；本轮四任务又没有Object梯度。后续优先重新审视独立meta映射及足量曝光，而不将局部稳定作为
永远停留四任务的工程门。已有55meta+18target的allowlist和原两fit元数据仍可用；正式任务/预算、profile、validation8视频与数值选择
合同必须在后继launch前登记。这是剩余条件的检验，不是已知数据修复；fully-random与真实K仍保留为正式候选责任。

### A2充分共享映射与validation8预注册（2026-09-06）

当前实验为`configs/pi05_ecp_prw_meta73_equal_exposure_v1.json`。沿用审计过的55 meta+18 target明确allowlist，
task6/79继续零梯度诊断；固定validation/test及其语义重复项不进入梯度。旧Native-Temporal、parallel-read、common-base的
73-task 100/200结果已失败，本实验不抹去它们，也不复活旧v4的50-step run。此前A2四task有有限净行为增量，但缺少Object
梯度且没有检验充分共享映射；新实验只检验当前图在这一剩余条件下能否迁移，不把数据不足认作根因。

每次optimizer update包含全部73 tasks，各权重严格1/73，64 updates使每task恰64 exposures、每条原fit视频32次。
与A2四task保持模型、component初始化、seed20260905、warmup5+effective59、AdamW/LR、8 cross-episode action rows、
fresh Panel-A sampling、normalizer和两fit K1不变；总4672 task executions。新增任务会改变global gradient mean及batch大小，
因此这是共享映射扩展干预，不能单独归因于数据数量或梯度冲突。保留32/48/64三个完整checkpoint；Panel-B只作定位，
覆盖meta1/2/8/9/32/52、target72/77/83/93/94和诊断6/79。只声称K1；fully-random同拓扑fresh联合训练仍为Final候选责任。

训练64步自然完成后，对三个checkpoint各运行两组correct K1 validation8 strict400，禁止先按functional筛选。
两组在每个task的50条成功demo中，使用`random.Random(20260906+104729*global_task_id).sample(range(50),2)`
无放回事前抽取；global IDs 1/3/11/13/23/26/31/32对应primary/other为42/39、12/23、39/36、15/4、16/6、45/26、45/34、6/1。
配置为`pi05_ecp_prw_meta73_validation_{correct,other_correct}_v1.json`。两组都是正式qualification arms；不是最后才打开的
negative controls。每组全部400个状态与env/policy RNG严格配对，official preprocessing、50 init states/task、成功即停止和
suite horizon不变，使用现有eager batch evaluator、long-first动态队列与persistent workers。
同一提交和400状态另测frozen source与stable carrier作可比参照，各一次；carrier无视频、零Writer调用，直接引用已冻结的唯一rank16。
这两组不参与checkpoint选择，也不从validation形成任何梯度。

事前数值资格定义如下，不在看到结果后放宽：

- 相邻pair只能是32→48或48→64；pair内两checkpoint的两正确视频panel均严格>145/400。
- 四个panel均至少6/8 tasks各有≥5成功，四suite各≥5，Goal与Long各≥15成功。
- 每一video panel内，相邻总分绝对差≤20，churn=(gained+lost)≤40/400，retained/max(前后成功数)≥.85、Jaccard≥.75。
- 每一checkpoint的跨视频总分差≤20，retained/max(两组成功数)≥.85、Jaccard≥.75；按suite/task另报全部R/G/L，不能由总分掩盖损伤。
- 存在合格pair才有候选：在合格pair所含checkpoint中最大化两视频较低分，依次以两视频均分、较晚step破同分。
  不使用checkpoint union、平均LoRA或挑视频。方法候选仍须完成fully-random同图比较，再冻结唯一selected checkpoint并补完整
  wrong/no-video/language/first-final/shuffled/reversed因果controls；负controls不反馈本轮选择或架构。

若未通过，按全部single-checkpoint闭环与相邻成功集合定位，保留实际正证据；不自动续训、扫LR/seed/rank、增同task视频或回到P/Q。
若接近门槛且趋势有信息，新的实验须提出能区分竞争解释的具体干预。任一结果均不消费test8。

资源：先用真实73-task完整group做两update profile，覆盖146条fit视频/4736 stride5帧和最长87帧；不缩rows或frames。
拟用gpu01空闲物理0/2/3/4/5/6，按GPU-local NUMA，deferred NCCL且NCCL_P2P_DISABLE=1；具体launch以live两节点快照为准。
单份node-local共享mmap预计约104GB，含Panel-B新增视频、checkpoint与评测的/data1新增峰值预算<150GiB；复用资产并自动移除临时cache。
实际profile外推总训练/评测时间后登记一次formal launch合同，来自clean pushed detached worktree，exact resume锁原topology。

### A2同拓扑完全随机初始化候选

owner要求Final保留整套Writer fully-random fresh联合训练；此前各同图、A2、四视频、P/Q及meta73运行均为component-init，
不能替代这项比较。meta73 component的32两视频80/78、48两视频81/81，相对carrier72只有有限增量，且不满足本轮相邻资格；
64两视频仍完整执行以保存后期趋势。下一候选只改变初始化，不把组件继承命名为已确认的根因，不增加新读出或解码器。

配置为`pi05_ecp_prw_meta73_random_init_v1.json`，使用现有`--initialization random`入口：整个2,970,368参数Writer从同seed
constructor状态出发，完全跳过G2的prefix/response projections、owner/family/layer/horizon embeddings及首个evidence attention复用。
source、frozen observer、normalization与rank12 carrier仍冻结；whole Writer直接联合训练，无专属task query或分段冻结课程。
完整部署图、两fit/K1、55meta+18target、全73等权group、每task64 exposures、8 cross-episode rows、Panel-A fresh query、
seed20260905、warmup5+effective59及全部optimizer设置与component相同。它比较的是同一有限预算下的初始化条件，不是seed小扫，
也不能由random的负结果宣称所有随机初始化训练预算均不可行。

32/48/64全部完整checkpoint及原13task Panel-B保留；不从loss选点。两validation配置为
`pi05_ecp_prw_meta73_random_validation_{correct,other_correct}_v1.json`，除了training config引用外与component完全相同：
固定同一8tasks、同一两视频、同一400 states/env/policy RNG、official preprocessing及一次唯一rank16物化。
三checkpoint × 两正确视频各strict400；复用本轮已完成的同执行source47/carrier72原始400 rows，无需再跑两个固定参照。
数值资格、相邻pair、跨视频稳定性及candidate排序完全沿用上一节；初始化比较只使用各自完整single checkpoints，不能融合两run。
当前component没有合格pair；random如通过仍须冻结唯一selected checkpoint并完成原定视频因果controls，Test8继续保留。
若random更弱，则保留较强工作参照及两者负证据，按竞争解释分析，不自动延长、扫seed/LR/rank或恢复P/Q。

本候选没有source算法或张量拓扑变化，复用4d57f816已通过的完整73-task、最长87帧、8rows真实profile与运行吞吐依据。
新增三配置及CLI初始化元数据做直接CPU解析、全字段差异和采样合同核对；新run启动后检查实际initialization记录reused为空。
预计训练及Panel-B约93分钟，后续六组strict400约60–70分钟；实际时间受可用设备和闭环早停影响。
GPU选择在当前评测结束后live核对两节点，单job单节点、总量≤6；fresh optimizer/scheduler、单份约98GiB临时cache，
新增峰值<150GiB，启动前重新核对/data1独立quota。精确启动合同与状态写入progress；配置、命令及formal均来自clean pushed frozen tree。

## 3. 已封存P/Q候选：模块与因果作用

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

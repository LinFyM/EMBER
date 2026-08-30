# EMBER task plan

更新时间：2026-08-30。

## 当前目标

owner已正式许可推进ECP Native-Factor Compiler。G1 task-local free-code capacity oracle、G2原动态Gate以及G3 P0/P1 current-bank
operator capacity均已通过；J2/J3在充分functional updates后non-pass。R1固定正交route把train/held-video从J2的`.1708/.1646`提高到
`.2678/.2798`，wrong-token margin`.2384`，证明清晰route被使用；但q/v/action-in仍约零，完整`.60/.50` Gate未过。checkpoint几何又证明
route在scorer内部没有塌缩，而task-local正控在开始functional优化前已由teacher初始化获得最终收益的约43%。R2证明set-valued critic
能恢复v/action-out几何，R3 owner×group decoder又把action-in/out提高到`.656/.669`，但q/v和真实functional仍未过门。R4用已验证
functional code初始化shared heads后，step110 train/held-video已达`.819/.839`，q/v/action-out及路由、跨video、稳定性均过门，仅
action-in为`.249<.30`。checkpoint/head/chart graft进一步证明，action-in失败来自minimum-norm初始化所依赖的feature chart在训练中
漂移：heads自身几乎未动，接回initial chart即可保留`.998` outer recovery。R5冻结初始化后的feature chart、只让233个native heads接受
同一真实functional loss后，step70/110均通过全部primary checks；step110 train/held为`.940/.963`，四family为`.816--.839`且相邻稳定，
所以moving-coordinate根因已被正式修复。R6把这套passed shared scorer接回真实Natural Program后，train/held却只有`.165/.143`，
Program-to-functional-code从fixed token的`.9985`降到约`.02`；同task跨video仍约`.9994`，证明失败是稳定但错误的内容坐标，而不是
video噪声或heads漂移。R7进一步冻结R5 heads、用task-level functional-code dense监督训练Natural Program和feature chart；内部方向
升至`.64--.74`，但step110 train/held仍为`-.133/-.130`，target role全部为负，证明Natural Program无法在冻结的任意R5 chart中取得
足够精确的功能坐标。当前下一资格只改变这一接口：同一validated code supervision下联合训练Program与完整primal scorer，让绝对
outer-update label锚定输出并消除冻结chart约束；先做真实fit/gradient smoke，越过R7上限后才formal。R1--R7都不是deployment
checkpoint；任何局部Gate、单次训练或内部指标都不代表整体项目goal完成。

R8 matched诊断已经排除fresh joint、raw process与完整raw Stage0字段遗漏：三者110步方向均停在约`.45`。R9从R5已通过的共享
functional chart初始化并联合获取Program/scorer坐标，内部四family升至约`.60--.75`且零梯度task2方向为正；但clean formal
step70/110的真实functional train/held/task-held分别只有`-.182/-.176/-.009`与`-.132/-.130/-.012`，五个target-role gradient
tasks全部为负，wrong-bank与interaction近零。R9因此正式non-pass，并把最早接口从随机chart可优化性进一步定位为
**outer-code近似不能替代policy utility**。当前R10只把R9 step110作为training-only稳定内容坐标初始化，冻结已有R4/R5证据表明会在
functional训练中漂移的feature chart，移除outer-code loss，只让Natural Program与native heads接受真实cross-episode panel-A flow。
R10 step110把train/held从R9负值提高到`.560/.544`，四family与Program因果margin通过，但task-held仅`.151`、target74为负，wrong-bank与
interaction近零，因此仍正式non-pass。预注册的R11 matched raw-Stage0 probe现已完成：step110 train/held/task-held为
`.292/.288/-.092`，相对R10 task-held下降`.244`；target gradient median仅`.110`，v-family仅`.102`，而q/action仍约
`.47--.55`。因此Program schema/压缩不是首因，也不能把局部v失败扩写成整个frozen Stage0无信息。R5/task-local成功primal的
cross-task wrong-bank upper-bound现已完成：错误bank保留`100.4%`中位收益，correct-minus-wrong中位`-.0038`，证明当前global
`C^+d` operator缺少bank交互可识别性。下一步先建立保持same-task capacity且产生correct-over-wrong必要增量的operator-level正控，
不再训练一个只改Program/scorer的新版本。R1--R11均不是deployment checkpoint，局部结果不代表整体goal完成。

operator正控现已闭合。R5成功primal的deployment-visible input projection p10在三条same-task与五个same-role wrong banks上AUC `1.0`、
逐task严格分离`10/10`；最终residual幅值gate失败，但固定内容阈值在full/half operator坐标间hard switch得到
correct/wrong/margin `.950915/.005173/.908899`与`10/10`强margin。soft query interpolation把correct降至`.238736`，故不是继续调
温度；R10原checkpoint的matched/mismatched support AUC又只有`.558160`、严格分离`0/12`，hard route令step110 train/held坍缩到
`-.482993/-.631937`。当前最早接口因此是Natural Program/scorer未学习bank compatibility，而不是operator端点缺容量。下一实现只在
R10稳定functional basin上加入共享、cross-video positive / same-role negative的projection calibration，并用近二值内容route选择
full/half；correct functional loss保留，其它authority全部冻结。它必须先同时恢复正确support、错误support margin和R10正确functional，
才有资格重跑完整12-task G3 Gate。

## R5--R10当前里程碑

- [x] 完成R3 clean formal与step70/110完整Gate，确认action-in/out恢复但q/v与train/held仍non-pass；
- [x] 用六task真实functional/critic gradient分解证明旧critic方向不适合继续加权，用utility-code gradient证明不能把成功code再作为
  一个永久joint loss；
- [x] 用同bank真实policy forward证明task-local成功方向改用冻结shared scale仍有`.9398`中位recovery，排除scale为首因；
- [x] 实现一次性functional-code minimum-norm shared-head初始化，保持真实X/Y、current-bank dual/exact signed replay、rank4、
  frozen scale与唯一rank16不变，并移除R4训练critic；
- [x] 25项定向CPU合同与gpu01真实六task forward/backward通过：step0 positive-code recovery中位约`.962`、Action Meta 0、teacher reads 0、
  全部有效gradient有限，峰值`21.882GB`；
- [x] 完成diff审查、clean main集成、push与detached formal launch contract；
- [x] fresh运行10 warmup+100 effective functional-only updates，在actual step70/110保存完整single checkpoints；
- [x] 用六个独立workers完成R4 step70/110 paired Gate；step110除action-in`.2493`外全部主检查通过，R4 formal non-pass；
- [x] 完成initial/step70/step110 head、chart和module graft，证明33个action-in heads在当前hidden可精确重拟合，失败来自整条
  feature chart的distributed coordinate drift；
- [x] 在唯一routing-control实现面加入`native_heads_only`参数所有权与R5 sealed config，不新增平行模型或fallback；
- [x] 完成13项定向CPU合同和单卡真实forward/backward/materialization profile：233 heads全部有gradient、chart冻结、Action Meta 0、
  teacher reads 0、唯一rank16及显存/吞吐成立；
- [x] 完成diff审查、clean main集成、push与detached formal launch contract；
- [x] fresh运行原10 warmup+100 effective、actual step70/110并执行同一paired Gate；R5相邻checkpoint保持强functional、四family、
  route与same-task稳定，step110完整Gate pass；
- [x] 依据R4/R5证据选定最小接回合同：R5 shared scorer初始化、Natural Program + native heads trainable、feature chart frozen、
  correct functional-only、fixed route不加载；
- [x] 完成R6定向合同与真实forward/backward/materialization smoke：Natural Program和全部native heads有gradient、feature chart冻结、
  fixed route/lookup/Action Meta为0、唯一rank16与吞吐成立；
- [x] 完成R6 diff审查、main集成、push与detached formal launch contract；
- [x] fresh运行R6原12-task 10 warmup+100 effective、actual step70/110并执行完整Gate；step110 train/held/task-held为
  `.165181/.143114/-.034333`，R6明确non-pass；
- [x] 用共同R5/R6 heads、三video minimum-norm fit和task2/74 held诊断证明fixed-token chart没有Natural Program内容几何，简单head refit
  只能插值fit views而不能泛化；
- [x] 接通R7 fit-only functional-code chart acquisition：R5 native heads冻结，只训练Natural Program+feature chart，使用task-level
  positive-control outer-update direction作为training-only label；Action Meta、policy functional action、task-local scale及held信息均不读；
- [x] 完成R7定向合同、旧配置兼容、真实单步gradient/ownership/吞吐检查；
- [x] 从clean pushed detached authority fresh运行R7 10 warmup+100 effective并评价step70/110原12-task完整functional Gate；
  step110 train/held为`-.133386/-.129792`，target gradient tasks全部为负，R7正式non-pass；
- [x] 用fresh scorer、raw process与完整raw Stage0 matched diagnostics排除遗漏Program字段；只以R5 passed shared chart初始化
  scorer后，同一joint outer-code训练把fit提高至约`.71`并在task-held取得`.640`中位方向；
- [x] 接通R9 retained config/schema与Gate，真实world6 step1逐值复现disposable结果，全部Program/scorer梯度、Action Meta 0、
  source/Stage0/scale冻结、无lookup和吞吐合同成立；
- [x] 从clean pushed detached R9 authority运行10 warmup+100 effective、actual step70/110并执行原12-task完整functional Gate；
  step110 train/held/task-held为`-.131825/-.129718/-.011724`，内部四family全部过门但真实primary明确non-pass；
- [x] 依据R9的role/task/causal分解把最早接口定位为code-to-utility：不是继续outer-code训练、raw Stage0、scale、bank或超参小扫；
- [x] 完成R10 R9-initialized functional refinement的retained实现、18项定向合同、真实world6 step1、clean formal step70/110与完整
  12-task Gate；step110 train/held为`.559896/.544189`、四family与Program margin通过，但task-held`.151475`、wrong-bank与interaction
  失败，R10正式non-pass；
- [x] 完成matched raw Stage0 sufficiency：相同12-task split、scorer容量、functional loss、预算和Gate，只交换部署可见上游表示；
  clean detached world6 step70/110与完整Gate均完成，step110 train/held/task-held为`.292321/.288053/-.092369`，相对R10
  task-held下降`.243844`，明确不支持Program schema/压缩首因；q/action仍可读而v仅`.101550`，不满足全Stage0停止条件；
- [x] 用R5/task-local成功primal执行只读cross-task wrong-bank functional upper-bound：正确/错误bank recovery中位
  `.930860/.945799`，correct-minus-wrong中位`-.003819`，正确bank仅`2/10`更好、`0/10`达到`.10`，错误bank`10/10`
  仍有正收益。该正控明确否定当前global-`C^+d` operator的bank交互可识别性，不能继续把wrong-bank失败归因shared
  Program/scorer。
- [x] 在不恢复退役candidate scorer或parameter teacher路线的前提下，先建立operator-level interaction正控：真实X/Y、signed
  pooling、rank4、唯一rank16与same-task跨video能力必须保留，同时correct bank相对same-role wrong bank产生必要功能增量；只有该
  正控通过，才恢复shared Program/scorer训练并重跑完整12-task Gate。half operator formal得到correct/wrong/margin中位
  `.725204/.188873/.541238`与correct-better `10/10`；bank interaction强过门但correct未达`.75`，因此总体严格non-pass。
- [x] 完成由两个已测端点夹定的唯一tempered bridge：`C_B^{-3/4}` dual + fit-only `C_B^{-1/4}`
  transport的correct/wrong/margin中位为`.925312/.885043/.054500`，correct-better `8/10`但margin达`.10`仅
  `2/10`，严格non-pass。该结果连同`.5/1.0`端点证明单一谱幂无法同时提供capacity与bank必要性，停止谱幂调参。
- [x] 完成full-inverse raw/normalized energy grid：R5成功primal的input projection p10在30个same-task video与50个same-role wrong
  banks上AUC `1.0`，且逐task/global均严格分离；普通raw dual energy方向相反，故只保留gauge-free projection作为compatibility authority。
- [x] 完成最终residual幅值gate、operator hard switch与固定soft query mixture三项因果分解：幅值gate margin`.031766`、soft mixture
  correct仅`.238736`均non-pass；hard switch correct/wrong/margin `.950915/.005173/.908899`且`10/10`通过，证明必须选择operator
  coordinate而非缩放或连续插值。
- [x] 将同一hard switch套到R10 step70/110完整12-task Gate并记录全部support：step110 matched/mismatched AUC`.558160`、严格
  分离`0/12`，train/held坍缩到`-.482993/-.631937`。因此固定门不是shared解，下一修正必须训练Program--bank compatibility。
- [x] 在唯一joint Program--primal执行面实现R10-initialized shared compatibility qualification：固定p10 support与hard full/half deployment
  route；同task fit videos交叉positive、同role cyclic wrong bank negative；正确functional teacher-force full endpoint。230项全量CPU合同和
  gpu01三卡真实六task forward/backward/materialization通过，Action Meta 0、teacher reads 0、唯一rank16。gradient-ratio证据把compatibility
  weight从会压倒functional约百倍的`1.0`固定为`.01`，复测总norm`.2451`且各关键gradient finite/nonzero。
- [ ] 从clean pushed detached authority运行R12 formal至step70；检查positive/negative support、route fraction、cross-role/task分解及
  correct functional训练曲线。只有机制资格成立才exact-resume到step110并运行两个single checkpoints的完整12-task paired G3 Gate；
  non-pass先定位support acquisition、held generalization或hard-route utility中最早失效者。

## 当前G1里程碑

- [x] 从`main@13ca366`建立隔离的`codex/ecp-native-factor-g1`实现面；
- [x] 接通真实38-target native X/Y hooks、四类output banks与跨chunk/video边界状态；
- [x] 接通task-local free-code signed pooling、rank4 outer products、small-core SVD和唯一rank12+4 rank16 materialization；
- [x] 接通四类G1 loss、optimizer、checkpoint、静态task-LoRA evaluator和four-arm Gate report；
- [x] 用最小真实CUDA forward/gradient/materialization smoke证明全部free variables有有限非零梯度、Action Meta实际未加载、
  source/Stage 0无trainable parameter且checkpoint为38-target/76-tensor rank16；
- [x] 完成首轮代码/测试/diff审查，集成到clean pushed`main@9a6f434`并从detached worktree执行formal；
- [x] 完成首轮5-task optimization、唯一rank16 strict250与four-arm Gate：`88/250`、breadth3/5、Goal/Long 0，结论non-pass；
- [x] 完成read-only output span与paired response projection，定位scalar q-output pooling的列空间上限，并以`109/250`、
  Goal/Long仍为0验证被排除方向具有闭环后果；
- [x] 实现真实q八头独立signed measure修正，并通过CPU合同及task93真实forward/gradient/materialization smoke；
- [x] 完成q-head修正diff审查、全量CPU回归、main集成与detached formal worktree；
- [x] fresh重跑q-head 5-task formal optimization、strict250与同一G1 Gate：`84/250`、breadth3/5、Goal/Long 0，仍non-pass；
- [x] 用真实bank稳定子空间投影和paired strict250把最早接口定位到free-logit可达优化：`94/250`、breadth5/5、Goal/Long非零，
  但retention`22/43`；
- [x] 实现reference-projected positive/negative simplex初始化与frozen native chunk cache，并通过task93真实一步gradient/materialization
  profile和140项CPU回归；
- [x] 保留optimizer前step0，完成latest-only五task formal与strict250：`100/250`、breadth4/5、Goal3/Long0、retention`22/43`，
  Gate non-pass；
- [x] 以paired fixed50证据定位set-valued选择接口，并实现每task在carrier/latest/independent/earliest中选最强verified member；
- [x] 通过141项CPU回归、task90 zero-residual initialization-only与task94 independent真实gradient/materialization smoke；
- [x] 完成set-valued clean formal与strict250：`111/250`、recovery`1.015`、retention`34/43`，但breadth4/5、Long0、仅3/5高于carrier；
- [x] 由task94 minimum direction cosine `0.978/0.883`定位FP32 inverse-scatter数值失真，并用真实FP64 smoke恢复到两侧
  `>=0.99999988`；
- [x] 完成FP64 signed-solve diff审查、全量回归、main集成与detached formal worktree；
- [x] fresh生成五task step0并完成single-checkpoint strict250：`116/250`、recovery`1.090`、retention`35/43`，但breadth4/5、
  Long0且仅3/5高于carrier，Gate仍non-pass；
- [x] 用paired response只把task94的action-in target恢复为known-success independent mobile，Long从`0/50`变为`1/50`；完整
  counterfactual为`118/250`、breadth5/5、4/5高于carrier、retention`35/43`，定位whole-vector action-in
  `span(column_space(W),bias)`上限具有独立闭环后果；该privileged替换不是G1 candidate；
- [x] 实现action-in按其native input width形成32个真实32D Y blocks的独立signed measures，并通过142项CPU合同检查；
- [x] 完成task94真实forward/gradient/materialization smoke：32个output blocks均stable rank32、两侧minimum cosine
  `>=0.99999988`、全部26,208,000个output logits有非零有限梯度、Action Meta 0、唯一rank16，峰值约29.77GB；
- [x] 完成全量diff审查、142项CPU回归、clean pushed `main@31f0053`集成与detached formal；
- [x] fresh生成五task step0并完成同一strict250 Gate：`114/250`、逐task`35/31/45/2/1`、relative recovery`1.060`、
  breadth5/5、Goal2/Long1、4/5 task高于carrier、retention`35/43`，G1正式通过；54/54 shards、250/250 rows、18/18 workers正常，
  Action Meta关闭且输出为single complete rank16。

## Phase R：全仓库理解与资产映射（已完成）

这是开始实现前的必需orientation，不是让owner重新解释项目。全程先只读，不启动GPU或修改科学状态。

1. 按`AGENTS.md`顺序完整阅读owner requirements、plan/findings/progress、concept、research history、专家原文和active design；
2. 用自己的话写清EMBER的输入、输出、zero-interaction目标、信息墙、成功标准与ECP Native-Factor因果链；
3. 对照历史分数解释哪些结论已成立、哪些接口失败、GOMQ/PECS/v24/人工process为何不属于当前主线；
4. 阅读并映射`configs/`、`src/ember/ecp/`、`source_sft/`、`expert_manifold/`、`writer/`、`reward/`、`pi05_eval/`、`scripts/`与
   `tests/`的职责、入口和当前保留/缺失能力；
5. 只读盘点ignored `data/`、`models/`、`runs/`、formal checkpoints/raw rows、task expert bank、Stage 0、carrier/mobile evidence、
   fold manifests与evaluator，不复制或删除资产；
6. 核对Git HEAD/origin、worktree/branch、当前无运行任务假设，以及GPU/storage约束的live检查方式；
7. 向owner给出一份简洁的“我对EMBER和当前仓库的理解”：目标、架构数据流、已有证据、可复用资产、实现缺口、后续阶段和G1
   首步。若发现文档、原文和代码冲突，先列出证据并修正文档，不凭摘要猜测。

Phase R已完成，`HANDOFF.md`已消费并删除；owner随后已明确许可进入G1。后续不重做orientation或退役路线考古，整体推进
不设阶段工期和修正次数上限，但每次scientific修正必须有机制证据。

## Phase G1：Native-factor capacity oracle

### 目的

不训练共享video-to-LoRA映射，只回答：自然视频在PI0.5各LoRA目标产生的native input/output vectors，能否通过小型task-local
selection形成强闭环mobile rank4 residual。

### 输入与冻结

- fold0 held5 natural teacher videos；
- frozen current Stage 0 v3；
- known successful task experts/mobile-rank4 projections；
- frozen carrier43、source与原始PI0.5；
- 不读取validation/test，不制作新数据。

### 实现工作

1. 为18层q/v与action-in/out增加真实linear input/output hooks，probe轴保留；
2. 输入候选使用`n_A=(video,frame,probe,horizon)`的真实`X`，输出候选使用额外带`type in {abs,adj,init,goal}`的`n_B`与真实
   `Y^type`；不得把`X`复制到无意义的type轴；
3. 构造absolute/adjacent/init/goal output banks，按frame chunk在线读取与累计；online softmax除running maximum、normalizer和weighted
   sum外，还须按video保持首帧、末帧及跨chunk previous activation，并与non-chunked reference数值等价；
4. 实现two-branch signed pooling、per-target scales、rank4 outer products与small-core SVD canonicalization；G1允许直接优化task-local
   selection logits/weights，不要求共享Program-query到candidate-key映射；
   首轮证据已证明对q的整条2048维value强制共享一个scalar measure会把所有输出限制在base-weight的1024维列空间；当前修正保持
   原candidate index和真实Y值不变，只按PI0.5原生八个query heads分别归一化signed measure并拼接，v/action-in/action-out不变；
5. 实现每task free-code optimizer，只优化4 rank queries、event weights、输入/输出pooling weights或logits和scales；`K>1`时固定
   `beta_k=1/K`并做video内assignment归一化，`K=1`为identity，不学习video reliability；
6. 明确走纯Native Stage 0 observer加载路径，在run contract与最小真实forward中核对实际module/trainable parameter，证明Action Meta
   未被旧loader装载；
7. 接通global-member effect、sensitivity-normalized update、independent functional与carrier-preservation loss；
8. 生成唯一rank12+rank4 complete adapter，接入现有strict evaluator。

先做最小真实forward/gradient smoke，再进行5-task optimization与strict250；不增加通用框架、checksum或与Gate无关的测试。
G1通过只证明native banks加signed pooling形式存在强rank4 residual，不证明deployment Writer或共享Program-to-attention映射成立。

### 通过门

同一strict250比较carrier、direct latest、known mobile projection和native-factor free-code，必须同时满足：

- relative oracle recovery至少0.70，按43/110参考约`>=90/250`；
- breadth 5/5，Goal与Long均非零，至少4/5 tasks高于carrier；
- carrier successes保留至少33/43；
- single rank16 adapter、strict pairing、无second adapter。

失败先做read-only span/response分析，再按最早失效机制修正hook、bank、pooling或优化并复评。不设修正次数上限；每次修正必须
有新证据和明确假设，不能变成slot/width/seed小扫。只有充分尝试后持续证明native basis不可达，才停止Native-Factor。

## G1通过后的固定序列

### G2 Natural Program

meta56+target-fit19，K均匀采样1/2/4；训练owner-specific language/scene、ordered events与K aggregation。meta-held15+target-held5检查
same-task separation、probe stability、event non-collapse、full相对endpoints的held loss增量，以及每video event alignment、variance、
uncertainty、`K=1` identity与video集合置换不变性。修正应限于证据定位到的native capture、event grounding或owner-specific
language/scene，不设次数上限。

当前实现节点：

- [x] 固定Program schema、owner-specific language/scene readers、两条antithetic probes、ordered event decoder和monotonic canonical alignment；
- [x] meta56/held15与target-fit19/held5角色、K=1/2/4 task-equal schedule、跨episode video/action监督和`beta_k=1/K`；
- [x] 95-task BDDL progress/rising、真实simulator contact及terminal contact mask的CPU label authority；
- [x] action/progress/predicate/contact/scene、same-task event、probe、speed/crop、contrast、occupancy/tau/uncertainty losses；
- [x] 每条video完全独立native encoding；真实K4检查把集合置换误差从`0.132`降到`2.38e-7`，K1保持exact identity；
- [x] 真实K4 forward/backward使84/84 trainable parameter tensors进入optimizer state，Action Meta module/parameter为0，峰值约18.85GB；
- [x] formal前复核已消除rank-local顺序导致的辅助loss不等权：每个task都计算一次speed/crop robustness，contrast对每task使用固定数量、
  两种fit role各半且与rank/world-size无关的language negatives；action与全部动态标签共享唯一action-episode query index；
- [x] clean pushed `main@141a110`的macro10 formal与held20 Gate完成；除full-vs-endpoints仅改善`0.0226%`外其它资格项全部通过，
  因而未进入G3；read-only消融把最早接口定位为training decoder的静态旁路，而非native动态捕获。
- [x] 移除`P_lang/P_scene`到`P_process` fusion及时序heads的直接加性旁路，保留独立scene head；clean pushed
  `main@30b98ef`的fresh macro10仍non-pass，full相对endpoints为`-0.0570%`，one-event `0.30`、probe margin `0.65`。
- [x] read-only temporal与event-grounding诊断定位到G2训练侵蚀已有Stage 0 v3结构：初始event/owner relative RMS
  `0.06069/0.36992`降为raw encoder `0.02601/0.22824`；这不是query-time weighting或静态旁路残留。
- [x] 冻结Stage 0 v3 observer，只训练新的Program readers/fusion/alignment/diagnostic heads；task92真实K4 smoke确认46/46个新增
  parameter tensors进入optimizer state、39个encoder tensors保持逐tensor不变，run contract确认native observer/source policy
  trainable均为0、observer处于eval且Action Meta为0。
- [x] 从clean pushed `main@db84a50` fresh训练并exact-resume到macro20；同一held20 Gate中same-task、K1/K4、active-event范围继续
  通过，但macro10/macro20的full相对endpoints分别仅`+0.0051%/-0.0207%`，macro20 probe margin为`0/40`，因此仍未进入G3。
- [x] macro20无梯度层级诊断确认frozen Stage 0 raw event/owner结构保持`0.06252/0.36771`，full/endpoints差异也真实存在；但共享
  `Linear(128,1)` owner score对固定38-owner轴严格置换不变，owner entropy为`0.99898`，action prediction temporal std仅
  `0.00173`而target为`0.32725`。继续训练没有修复最早readout接口。
- [x] 只把training-only temporal owner score替换为38个固定语义owner各自的shared-across-task linear query；queries从旧共享
  Linear完全相同的向量初始化以保持其余head的RNG序列，不修改Stage 0、
  Program schema、probe处理、scene head、数据、loss、seed/LR/slot/width或Gate。真实K4 profile确认owner-query gradient norm
  `0.01827`、一步后query rows已分化、46个Program tensors trainable、observer/source/Action Meta trainable均为0，peak约10.02GB。
- [x] 集成并推送owner-structured readout，从clean detached commit fresh训练并exact-resume到macro20；full相对endpoints为
  `+0.0158%/-0.0340%`、probe均为`0/40`，query分化从`1.58%`增至`2.94%`却未改变shared解，故该scalar selection不是充分修正。
- [x] 无梯度Stage0-transfer反事实确认raw process+既有action head把absolute action loss降至`0.20767`，但full增量仍仅`0.2467%`；
  最早接口不是简单旧head丢失，而是absolute MSE的trajectory-mean解未约束query-time residual。
- [x] 保留absolute action/progress并新增等权query-centered temporal residual MSE；真实K4 profile确认两个新loss有限、owner-query
  gradient非零、frozen observer 39 tensors不变且Action Meta为0。
- [x] temporal-residual objective由clean pushed `main@68f8705`完成并从fresh训练到macro10；held20的same-task、K1/K4、event范围
  继续通过，但full相对endpoints只改善`0.0381%`、probe margin为`0/40`，故仍未进入G3。
- [x] 冻结该轮Program做readout/label/optimization可证伪诊断：full-owner temporal readout相对endpoints可产生`15.17%`改善，证明
  动态信号可读；tied与independent query初始化曲线近乎相同，cross-episode监督可识别；而旧macro10实际只有10次Adam更新，
  frozen readout在10/60步几乎不动、200/500步才明显下降。最早接口因此是optimizer cadence，不是新的Program架构缺口。
- [x] 保持模型、数据、loss、K和Gate不变，实现每macro 10个role-balanced optimizer steps：常规2 target+2 meta，旋转尾部1+1；
  scheduler与resume按真实step计数。单卡与world4真实profile均完成finite forward/backward/materialization/checkpoint，world4实际聚合
  2+2任务、46/46参数进入Adam、Action Meta/source/observer trainable均为0。
- [x] 从clean detached `main@49e7769` fresh运行macro10/100 optimizer steps并复评同一held20 Gate：full相对endpoints改善
  `0.3080%`、probe `13/40`，其余资格项通过；相对旧10-update结果动态增量约`8.1x`且17/20 held task方向改善，但仍明确non-pass。
- [x] 冻结该checkpoint，用12个role/K平衡fit task做gradient diagnostic且held gradient为0：Program full/endpoints差异真实存在，
  temporal梯度没有被强方向性抵消，但在Program process/temporal decoder上分别比non-temporal小约`10x/21x`，prediction temporal std
  仍比target小约`93x/203x`。最早接口是近常数readout造成的temporal gradient starvation。
- [x] 按同一commit、world4 topology与run目录exact-resume到macro20/200 updates；held full增量升至`8.6878%`、probe升至
  `36/40`，fit prediction temporal std增长约`9x/30x`，验证readout学习时标；但所有K>1条件坍为one-event，Gate仍non-pass。
- [x] 用K分解与fit-only no-gradient alignment反事实定位根因：K1保留平均`6.42` events，K2/K4 local presence未坍缩但DP将约
  `6/8` path mass集中到单一canonical slot；boundary-only锚点恢复K>1为3 events且不损失视频增量。
- [x] 只把K>1 monotonic DP改为首尾canonical边界锚定、保留中间stay/skip与既有content/time score；真实K4 profile
  已完成4-video/102-frame forward、backward与optimizer step，active events为2、one-event为0，Action Meta module/parameter均为0，
  source policy与native observer均冻结；全量合同测试`155 passed`。
- [x] 从clean pushed `main@c1493a1`的detached frozen worktree fresh训练并exact-resume到macro20/200 updates；held20 Gate全部通过：
  full相对endpoints改善`22.2047%`，median active events为4、one-event为0，probe为`38/40`，same-task、K1 identity和K4 permutation
  均为1.0，tau violation仅`0.00357`。冻结`macro_00000020` Program进入G3。

G2只有一个canonical入口`scripts/train_ecp_natural_program.py`。模块ownership固定为：`natural_program.py`拥有部署Program schema与
Pass-A图，`natural_program_data.py`拥有fold/schedule及跨episode packing，`natural_program_labels.py`只拥有training-only派生标签，
`natural_program_objective.py`拥有机制loss，`natural_program_authority.py`拥有run provenance与信息墙inventory，
`natural_program_training.py`拥有macro/checkpoint编排，`natural_program_train_step.py`拥有唯一role-balanced optimizer update，
`natural_program_gate.py`拥有无梯度internal/official Gate；`behavior/codes.py`只拥有可由G2/G3共同复用的fit-only行为坐标与kernel authority，
`behavior/kernel.py`拥有固定Program feature、pairwise topology loss及evaluator-only readout，`behavior/gate.py`拥有rank16 basis及exact rank4
资格。Stage0 encoder、通用
checkpoint和既有video/action stores只复用，不复制。G3复用并冻结通过Gate的Program schema/model；G2 trainer、label sealer和Gate在
formal结论固化后仅作为可复现实证runner保留，不成为平行Writer或deployment fallback。

#### G2-B behavior-aligned sufficiency

- [x] 为meta56+target19 fit及meta15+target5 held各封存两组disjoint cross-episode flow-gradient panels；held任务没有optimizer梯度；
- [x] role-balanced rank16 behavior manifold在held20对独立panel-B/consensus达到`.7160/.8006`，而universal只有`.1908`，证明目标与
  rank16容量成立；rank32只小幅增加约`.0526`；
- [x] frozen `c1493a1/macro20` Program的full/process读出器在fit75达到约`.97`，held exact rank4却只有`.2695/.2470`，与
  language-only`.2687`同量级；最早失效接口确定为G2跨task behavior identifiability，而不是native bank、rank4、G3 dual/operator或训练不足；
- [x] 接通唯一G2路径上的training-only process decoder、role-balanced behavior loss、旧qualified model-only初始化与fresh optimizer；
  decoder不能读取task ID、`P_lang/P_scene`或deployment外信息，Program schema保持不变；
- [x] 接通official held20 exact-rank4 Gate：overall/consensus、四family、meta/target role、wrong Program、language增量、K1/K4与跨video一致性
  均有固定阈值；shuffled/reversed不使用；
- [x] 真实task74 K4 profile完成：behavior decoder与既有process fusion梯度`.6286/4.4444`，source/Stage0/Action Meta为0，单步
  `13.30s`；同一Gate对旧frozen Program正确non-pass而非自动放行；
- [x] clean detached `5cbe76e`的pointwise decoder formal完成macro10/20/40/60：behavior loss`1.2723 -> .7080`、旧动态增量
  `31.85% -> 39.40%`，但panel-B exact只有`.1837/.2622/.2938/.2828`，macro60 meta/target为`.3803/.1853`；Gate明确non-pass；
- [x] 冻结reader、Program geometry与fit-only kernel/linear反事实共同证明：reader能拟合fit code，但full Program fit topology仅
  `.1610 -> .1694`、official held约0，task-holdout读出仍约`.262`；最早接口是部署Program的跨task credit ownership，不是decoder
  容量、LR、训练长度、bank operator或rank；pointwise decoder/config已从active路径删除；
- [x] 固定role-stratified internal train60/held15，official held20不参与本轮训练/选模/修正；internal rank16 oracle对panel-B/
  consensus为`.6184/.7158`，四family为`.6556/.7373/.4550/.6676`，足以作为非平凡内部资格；
- [x] 接通decoder-free behavior kernel：固定block-equal完整Program feature、两组disjoint same-K views、role-equal panel-A+consensus
  factor-cosine topology loss、五卡5+5 role pairing、internal15 Gate及fit-only evaluator readout；没有新增deployment或training reader参数；
- [x] 真实三卡一步profile已证明两组video、global kernel loss、distributed gradient、现有language/scene/process路径、纯Native Stage0及
  Action Meta 0接通；峰值allocated约`9.98GB`、三卡均100% UTL、一步`19.73s`。该profile早于最终cosine-authority schema，只作执行证据，
  clean v3 authority仍须重跑；
- [x] 完成diff/全仓回归与clean pushed `main@c8fee96`集成；从detached authority重封v3 factor-cosine asset，确认fit60/
  internal-held15 tensors、official held20 tensor overlap 0和对角误差`1.19e-7`；
- [x] 从clean detached `c8fee96`完成三卡真实一步profile：两组video、global autograd kernel loss、Program全部梯度、
  source/Stage0冻结、Action Meta 0与显存/吞吐合同均通过；该profile只作执行证据，不代替Gate；
- [x] 从clean pushed `main@60fb18b`的detached worktree执行五卡v3 macro5 formal及旧动态+internal行为Gate；旧动态
  Gate保持通过，但train topology仅`.2315/.2358`、internal meta仅`.2152/.2332`、exact panel-B/consensus仅
  `.1207/.1253`，因此明确non-pass且official held20未读；
- [x] 用batch co-occurrence graph定位v3目标的最早失效：meta45的15个batches只覆盖126/990对且分成5个不连通
  components，local batch correlation升到`.70`不能确定全局Program几何；该证据不支持解冻Stage0或续训v3；
- [x] 接通v4 joint-role topology：在原5+5 batch内以`.5/.25/.25`等质量组合joint/meta/target kernels，使监督图成为
  唯一60-task component；不增加reader、参数、forward或deployment路径，且保持原task/video质量、Stage0冻结与official wall；
- [x] v4三卡真实一步profile完成：joint梯度确实进入Program，step `18.33s`、peak `9.98GB`、source/Stage0
  trainable 0、Action Meta 0；定向与全仓回归`19/19`与`204 passed`；
- [x] 完成v4 diff/architecture检查、`204 passed`、clean pushed `main@37885a6`集成与detached三卡profile；无hard
  violation，clean profile与dirty profile的loss/梯度/吞吐逐项一致；
- [x] 从clean pushed `main@4eb8b8c`的detached worktree完成五卡v4 macro5 formal及全部internal/旧动态Gate；旧动态
  Gate保持通过，但train topology仅`.2360/.2362`、internal meta仅`.2064/.2257`、exact panel-B/consensus仅
  `.1129/.1177`，因此明确non-pass且official held20未读；
- [x] 用固定checkpoint block geometry定位v4更早失效：full Program off-diagonal cosine均值/标准差约`.965/.020`，而teacher为
  `.145/.316`；逐batch centered+normalized loss允许batch-local affine gauge与near-collapse，连通credit graph不能约束绝对几何；
- [x] 接通唯一v5 global calibration：teacher固定lift为`(1+K)/2`，raw Program off-diagonal Gram按完整scope的固定teacher
  dispersion对齐；不增加参数、reader、task route或deployment路径，不改数据、task权重、动态Gate、Stage0冻结与official wall；
- [x] v5三卡真实一步profile完成：behavior/Program梯度`1.7323/2.7450`，step `18.35s`、peak `9.98GB`，source/Stage0
  trainable 0、Action Meta module/parameter 0；定向`19 passed`、全仓`204 passed`；
- [x] 从clean pushed `main@7f4df1b`的detached worktree完成五卡v5 macro5 formal及全部internal/旧动态Gate；旧动态增量
  `20.8602%`通过，但train topology仅`.2160/.2208`、internal meta`.2022/.2169`、exact panel-B/consensus
  `.1054/.1289`且wrong margin`-.0466`，明确non-pass；official held20未读。
- [x] 冻结v5 checkpoint拆解六个Program blocks：full/process跨task std由v4的`.020/.086`扩大到`.046/.220`，但其teacher
  consensus相关从`.150/.135`降到`.142/.131`；已定位为“产生区分但区分方向不对应policy behavior”，不是loss未接通。
- [x] 已向第四位专家提交v3--v5 formal、block geometry、训练轨迹及完整远程历史；1075行原文逐字保存为
  `docs/expert_review_20260829_joint_program_primal.md`。裁决取消独立Program behavior-Gram硬Gate，恢复推进的唯一机制是joint
  Program--primal functional credit；不续训v5，不用seed/LR/width/rank小扫，也不先解冻Stage0。

### J2 / G3 Joint Program--primal functional qualification（已完成，non-pass）

P0/P1已经把真实native banks、rank4、current-bank global dual与exact signed replay排除为当前首因。当前不再冻结Program后单训P2，
而只联合两个尚不可分的相邻接口：Natural Program与`ProgramNativePrimalScorer`。功能链必须实际生成唯一38-target rank16 LoRA并由
cross-episode teacher action/flow loss反传；source、Native Stage0、bank operator、carrier、scale和Action Meta保持冻结。

- [x] 固化第四次专家原文与active contract；v5仍作为protocol non-pass历史证据，但不再作为恢复G3前的硬Gate；
- [x] 审计并复用Natural Program frozen evidence、P2 compact condition cache、current-bank operator和
  `functional_lora_loss_gradient()`，建立唯一J2模块所有权，移除active optimizer对behavior-Gram的依赖；
- [x] 用相同functional objective先完成10个gradient tasks的task-local、两fit-video共享free-primal正控；第三video严格零梯度。
  held-video recovery median须`>=.80`，四family各`>=.70`且每task显著优于carrier，否则先修functional panel/scale/authority；
- [x] 完成12-task joint Gate：gradient meta`[1,8,9,32,52]`、gradient target`[72,73,75,93,94]`，true task-held meta`2`、
  target`74`。两条fit K1 views与panel A训练，第三video及disjoint panel B只读，selected八targets负责family报告但实际生成完整38-target LoRA；
- [x] 首轮以最多10步warmup后100个有效joint updates计数，在post-warmup 60/100保存相邻checkpoint；这消解专家原文“100 total”与
  “约100 post-warmup”两种口径，最多110个实际optimizer steps，不以warmup中的低LR步骤冒充充分优化；
- [x] Gate已形成明确non-pass：step70/110 train median`.1596/.1708`、held-video`.1487/.1646`；task-held平均`.0050/.0068`且task74
  为负；step110 q/v/action-in/action-out为`-.0010/.0014/.0090/-.0004`，full相对language/endpoints为`.0866/.0334`，wrong
  Program/bank margins`.0080/.0071`、interaction`.0024`。held/train、same-task、event/K1、信息墙和step110吞吐通过，但不改变primary
  non-pass。原Gate要求仍为train median`>=.60`、held-video`>=.50`、两个true task-held平均`>=.40`且各`>=.30`、held/train`>=.80`；q/v各
  `>=.35`、action-in/out各`>=.30`；full相对language/endpoints各`>=.10`；wrong Program与wrong bank margins各`>=.10`、
  interaction`>=.05`、same-task retention`>=.80`，checkpoint 60到100 task median回落不超过`.05`，event/K1/信息墙继续通过；
- [x] 速度资格：先缓存冻结language/Stage0 raw evidence、X/Y、covariance eigensystem和固定action batches，不缓存Program或LoRA；
  六卡global update目标`<=30s`、硬上限`45s`、每卡peak reserved`<35GiB`，完整评价墙钟不超过训练主体一半。world size按live吞吐弹性
  选择，不改变task权重或Gate；仅做最小真实forward/backward/materialization与定向合同检查，不跑冗余全仓测试。task1真实
  positive-control formal发现task93在physical microbatch4下peak reserved`37.07GiB`；同一logical16/bank/seed改为physical2后，
  task93一步loss相对差`.060%`、step`13.32s`、peak reserved`32.41GiB`，系统修正通过。六卡joint真实profile进一步为
  `11.73s/global step`、per-rank peak reserved`18.25--20.29GiB`，所有Program/primal梯度probe非零，速度Gate通过。

### J3 / G3 Counterfactual functional routing qualification（已完成，non-pass）

J2正控高而shared train低，已满足把问题定位到Program--primal函数类或functional credit的分支。零optimizer-step审计排除clip和断图：
十task Program/primal pairwise cosine约`.93--.95`，生成effective update median`.678`且action-in`.997`；相反成功free-primal input/output
code median仅`.203/.149`。同一panel的task functional-gradient pairwise cosine median`-.023`、`62.2%`为负，六task组gradient
cancellation ratio`.421--.536`。因此纯correct-pair loss允许公共残差捷径，不能再以延长J2或普通超参修改处理。

- [x] 在唯一`joint_program_primal`执行面加入配对counterfactual credit：每step保留两条correct fit views；仅新增一条交替的same-role
  cyclic wrong-Program或wrong-bank view，并用同task、同panel、同policy RNG的bounded margin surrogate。错误组合只在margin未满足时
  反传；部署forward、参数、Program schema、bank operator、rank12+4、Action Meta和信息墙全部不变；
- [x] 用zero-step/one-step真实检查证明correct/negative配对、梯度符号、task/role等权、wrong condition无teacher-factor读取、Action Meta 0、
  38-target唯一rank16及全部trainable/frozen ownership；profile目标仍为六卡`<=30s/update`、硬上限`45s`和`<35GiB/GPU`；
- [x] 从fresh Program initialization/scorer运行同一10 gradient+2 task-held、10 warmup+100 effective updates，并在60/100评价原J2 Gate。
  J3必须同时取得train/held-video数量级提升与wrong Program/bank因果margin；只让错误臂变坏、correct recovery仍低不通过；
- [x] J3 step70/110 Gate均non-pass。step110 train/held-video为`.148649/.147689`，q/v/action-in/action-out为
  `.000466/-.004513/.008217/-.001500`；wrong Program/bank与interaction为`.010192/.012540/.005426`。错误controls在多数task上改善，
  correct fit recovery却只有3/10高于J2，故训练主要学会破坏negative而不是选择正确task方向；
- [x] J3仍使train recovery低于`.40`且task routing未展开，停止继续叠加contrastive、normalization或optimizer技巧，按已满足的
  function-class停止证据重开Program-conditioned nonlinear capacity/representation接口；不得转去raw Stage0 probe，因为该probe只属于
  train和held-video已高而task-held低的分支。

### R1 / G3 training-only orthogonal routing-token boundary control（已完成，partial/non-pass）

该对照不是新deployment Writer，也不能通过G3。它只给10个gradient tasks各分配一个固定、均值零、unit-RMS、两两正交的128D
Sylvester-Hadamard token，并把同一token填入固定Program schema；authority ID只在训练期选择token，token不是参数，task/frame查表参数仍为0。
每task两fit videos共享同一token，held video零梯度；bank继续来自各自真实视频，`ProgramNativePrimalScorer`以外全部冻结。

- [x] 接通独立schema/checkpoint authority与同一train/eval脚本dispatch；保持唯一rank12+4 rank16、真实X/Y/current-bank dual、
  exact signed replay、panel A/B边界、Action Meta 0及shuffled/reversed 0；临时执行面owner为本R1诊断，Gate解释完成后、实现下一canonical
  deployment architecture前删除，仅由Git与formal artifacts保留；
- [x] 定向CPU合同与六卡真实一步通过：10 token Gram精确为`128I`且均值0；只有scorer的`7,512,196`参数trainable，所有关键scorer
  gradient probes有限非零，native teacher reads 0；global step`12.383s`，最大reserved`20.28GiB`，明显优于J3三forward；
- [x] 从clean pushed detached authority以10 warmup+100 effective、step70/110运行formal；同一两fit correct functional primary，不加入
  counterfactual loss。Gate要求train median`>=.60`、held-video`>=.50`、held/train`>=.80`、q/v各`>=.35`、action-in/out各`>=.30`、
  wrong-token margin`>=.10`、same-task retention`>=.80`及相邻稳定性；六卡评价只跑10 gradient tasks的correct、wrong-token和family必要臂；
- [x] step110 train/held-video为`.267809/.279828`、held/train`1.0449`、wrong-token margin`.238352`、same-task retention`.9910`；相对J2
  9/10 tasks改善，但q/v/action-in/action-out仅`.003698/.007820/.001111/.033335`，所以R1只证明route有实质增量，整体Gate non-pass；
- [x] checkpoint几何把该non-pass进一步定位：四family hidden cross-task cosine仅约`.18--.27`，q/v对正确task-local code检索8--9/10，
  因此清晰token没有在scorer内重新坍缩；但对强解的coupled primal alignment仍只有q`.0015`、v`.0051`、action-in`.0085`、
  action-out`.0675`。step70到110各family参数都有同量级更新，排除某个head没有梯度；固定hidden上的最优last-head least-squares虽可将
  input、v-output、action-out-output拟合到`1.0`，q/action-in output仍分别受当前group-shared 128D head限制在约`.658/.363`；
- [x] 复核task-local正控发现其code由fit-only teacher consensus初始化，optimizer step1已拥有最终functional benefit的中位约`.431`；它证明
  好方向可被functional loss精修，却没有证明从随机scorer能发现方向。R1因此不能简单解释成“完美route也无用”，而是同时暴露
  functional discovery credit不足和q/action-in grouped-output函数类上限。

### R2 / G3 training-only fixed-route set-valued critic control（已完成，non-pass）

R2保持R1固定token、现有`ProgramNativePrimalScorer`、两条真实fit banks、current-bank dual/exact replay、functional panel和唯一rank16
完全不变；每条fit view只额外读取已有fit-only consensus member set，以gauge-aware paired effective-update direction提供训练期稠密critic。
teacher/member/action都不进入deployment输入，held video、panel B、task2/74及validation/test仍零梯度。该对照仍不是deployment Writer。

- [x] 最小实现复用R1 trainer/evaluator，只在配置存在`privileged_critic`时累计set-valued paired-update loss并严格要求fit teacher实际被读；
  R1 functional-only路径继续要求native teacher reads 0。Action Meta、Program/source/Stage0/operator/scale/carrier仍全部冻结；
- [x] weight-1六卡真实profile为`15.094s/global step`、最大reserved`20.29GiB`，初始critic recovery约`-.00061`，联合gradient norm
  `.1201`；相同seed/group的R1 functional-only norm为`.02575`。因此formal固定critic weight`.2`，使两种credit在初始化处约等量；这是一次
  梯度量纲校准，不做weight/LR/seed sweep。即使weight1的速度也仅刚超过`15s`目标且远低于`25s`硬线，故R2没有吞吐阻塞；
- [x] clean detached `a4b91bb`完成fresh 110-step及step70/110 Gate。step110 train/held为`.205796/.193603`，低于R1
  `.267809/.279828`；q/v/action-in/action-out为`.220453/.407617/.166808/.663453`，说明critic真实恢复了两族并部分恢复另两族，
  但wrong-token margin降到`.090559`，整体Gate non-pass。critic recovery在最后20步平台于约`.322`，不续训或扫weight；
- [x] 固定R2 hidden对成功free-primal做解析last-head反事实：现有shared-group q/action-in output最优median仅`.691/.392`，改为
  owner×group独立head后四family median/min均为`1.0/1.0`，每组hidden rank为`40/40`。因此下一修正是decoder，不把R2解释成
  Program、bank或critic未生效。

### R3 / G3 training-only owner×group decoder control（已完成，non-pass）

- [x] 只把每个owner的共享output head替换为每个native output group独立的linear head；group仍是q固定8、action-in native-width 32，
  v/action-out为1。新增约426万参数，不含task/video/frame lookup，输入primal、Program、bank dual/replay、scale及唯一rank16均不变；
- [x] 保持R2 functional+fit-only set-valued critic、数据、预算、seed、Gate与Action Meta 0；后续配置移除人为35GiB门；
- [x] 用R2两checkpoint的纯timing evidence预注册R3 evaluator cost map，把六个长/四个短task按实测cost做LPT分配，预计最长worker由约
  `400s/checkpoint`降至约`303s`，不读取或利用科学结果；
- [x] 21项定向CPU合同及clean pushed三卡真实forward/gradient/materialization profile通过：global update `25.334s`，峰值reserved
  `21.815GB`，所有owner×group heads有finite nonzero gradient；Action Meta 0、source/Stage0/scale冻结、唯一rank16均由run contract实测；
- [x] 从只含本launch contract和world-flexible checkpoint reader的clean pushed detached descendant启动fresh formal与step70/110 Gate。
  四family及functional一起恢复才把critic+decoder接回Natural Program；若四family高而functional仍低，才判定set-valued teacher-to-utility
  不充分并重做credit；若解析容量存在但训练仍未取得，则定位优化/parameter ownership而非扫普通超参。

### R3 formal launch contract

- implementation authority为clean pushed `main@67a49f88a9825c7437ea2b985240561a3d4624a4`；formal只从加入本launch记录和
  world-flexible Gate checkpoint reader、不再改变训练图或科学配置的clean pushed detached descendant fresh启动，不resume R1/R2/J2/J3；
- 数据、seed、10 warmup+100 effective、step70/110、functional+weight`.2` fit-only critic、固定token、两fit views、panel A、
  real native X/Y/current-bank dual/exact replay、rank4、scale和唯一rank16完全锁定R3 config。panel B、same-task held、task2/74、
  validation/test零梯度；Action Meta实际不安装；
- formal world size从launch时同节点真正合适的1--6张A40选择，不等待凑6。六个role-balanced tasks及12个views的loss权重始终固定，
  gradients使用SUM all-reduce；world size只改变每rank串行task数和可接受的浮点reduction order。checkpoint完整保存实际world topology，
  exact resume锁该topology，Gate按manifest读取对应rank states；
- 2026-08-29 22:41--22:51 CST live状态：gpu01物理1/6空闲，0只有他人约`.55GB`间歇短作业且峰值余量充分，2--5持续高UTL；
  gpu02只有物理4低显存，其余为约17--33GB占用或高UTL，不跨节点拼卡。因此首选gpu01物理`0,1,6`、world3，若正式exec前状态
  改善则使用当时更多真正增吞吐的同节点卡；不得使用2--5干扰现有训练。三卡profile已实测该映射无OOM；同路径R2的task93/94最坏
  reserved约`32.95GB`，R3只新增约`16.2MiB`参数存储，46GB A40仍有合理余量，不设35GB人为门；
- `/data1` quota为`708448164/1073741824KiB`，约余`348.4GiB`；复用gpu01 `/dev/shm/ember_ecp_j2_pc_10task_c4704cb_gpu01_20260829`
  23GB cache且`/dev/shm`约余39GB，不重建Stage0/X/Y。formal根固定为
  `runs/outputs/pi05_ecp_routing_grouped_decoder_r3_s110_67a49f8_gpu01p016_r3_20260829/`，预计新增远低于1GB；
- training自然完成后，以最多六个独立single-GPU workers评价同一step70/110的两fit+held、wrong-token和八target family臂，使用R3 timing-only
  cost map。只按预注册functional/family/causal/stability Gate裁决；critic loss、internal cosine、checkpoint union或shuffled/reversed均不能
  选模或修架构。

后续canonical G3通过后，恢复完整40 fit/10 task-holdout、329 fit/40 held-video/82 task-held的shared functional qualification；primary改为
generated-LoRA functional recovery，而factor/update cosine只作诊断。若train与held-video强、true task-held弱，则只做matched raw frozen
Stage0 sufficiency probe：raw task-held比Program高`>=.15`且达到`.40`才把最早接口归为Program压缩；raw也低于`.25`才允许考虑窄解冻
Stage0 process/presence/uncertainty tail。不得直接解冻VLM/source/整个Stage0。

以下清单保留此前Frozen-Program G3的可复用实现与历史排除证据；其中P0/P1继续有效，旧P2 frozen-Program formal不再是当前下一步。

冻结Program。既有sketch/set-summary/query-conditioned路径均已按机制Gate淘汰；当前用共享full Program预测target-native primal，
由每条当前video的全局单位质量covariance确定性转换为dual query，再对真实X/Y exact signed pool。covariance/dual不进checkpoint，
禁止task/frame查表；首版跨video固定`beta_k=1/K`，有证据后才允许从uniform初始化有界correction。held5要求full
`>=60/250`、breadth`>=4/5`、retention`>=33/43`、Goal/Long至少一项非零、相对language和first+final各`+5`、same-task
retention`>=80%`。可以按mapping/compiler/critic证据修正，不设结构版本上限，但无机制差异的小变体不算推进。

- [x] 以通过G2 Gate的`c1493a1/macro_00000020`为唯一frozen Program authority，先复用G1真实native X/Y capture、四类output banks、
  action-in native blocks、small-core SVD和rank12+4 materialization；实现共享content-derived query-key signed attention、target scales与
  由uniform初始化的bounded K correction，不保留task/frame free-logit路径。
- [x] 完成上述共享Pass B、95-task/118-member与set-valued critic的CPU实现合同：input/output candidate索引分离、跨chunk视频边界、
  per-video单位measure、K=1 identity、K=4置换不变和无task/frame free logits均已有定向回归；真实GPU机制smoke仍属于下一项。
- [x] 完成最小真实forward/gradient/materialization、真实cross-episode action-flow、same-task consistency、长K4显存与信息墙检查；
  三步profile均exit0，Action Meta 0，source/Program trainable 0，唯一rank16被policy实际消费。
- [x] 接通冻结checkpoint的held5 `correct_full/first_final/same_task_other`静态rank16物化、fit75 learned language-only control和
  paired five-arm Gate report；language control只用frozen `P_lang`映射fit-task verified rank4 effect，不读取held video/action/reward。
- [x] 从clean pushed detached `5140362`完成macro5/95 updates formal、三条video bank、learned language control与五臂paired
  strict250 Gate：carrier/language/full/endpoints/same-task分别为`43/42/35/40/44`，full breadth`3/5`、carrier retention
  `28/43`、Goal/Long均0、相对language/endpoints为`-7/-5`；只有same-task retention `33/35=94.3%`及全部adapter authority检查通过，
  所以macro5明确non-pass。
- [x] 从同一clean detached `5140362`按原合同fresh完成macro10/190 updates及五臂strict250：carrier/language/full/endpoints/
  same-task=`43/42/38/39/40`，full breadth`3/5`、retention`32/43`、Goal/Long均0、相对language/endpoints为`-4/-1`，仅
  same-task retention `84.2%`通过；因此“只是macro5 warmup后更新不足”被证伪，没有续到macro20或做超参小扫。
- [x] 用fit-only、held-gradient 0的K1 native-span诊断定位最早接口：6 tasks/9 members的native update cosine median `0.7029`、
  named/global functional retention median `0.7855/0.7981`且action benefit `9/9`为正，说明rank4压缩与真实native bank有足够功能容量；
  v1的全局clip由scale path主导，shared query/key更新不足且没有直接native-feasible selection supervision。
- [x] 从clean pushed detached `93dffc7`在gpu02 p4/p5/p6封存formal40 schedule实际覆盖的50个fit tasks、451个K1
  task-video条件、68个covered-task verified members及662个teacher states；只保存pre-scale A/B directions、scales与provenance，
  root为828MiB，held/Action Meta/deployment reads均0，三个worker与aggregate均exit0。
- [x] 在保持frozen G2 Program、sampler/K/LR/rank/bounded beta不变的首版修正中：K1以detached set-valued responsibilities监督
  shared query-key产生的input/output subspace、paired update direction和独立small-core spectrum；K2/K4 teacher tensor reads为0；
  selection与scale/video各自clip且scale/video不反传shared context。真实三步K1/K2/K4 profile已验证teacher lookup、gradient wall、
  same-task、bounded beta、长K4与唯一rank16，峰值29.32GB，Action Meta/source/Program trainable均0。
- [x] 从clean pushed detached `2a7f760` fresh运行G3 v2到macro5/95 updates并完成同一五臂strict250：carrier/language/full/
  endpoints/same-task=`43/42/41/38/37`，full breadth`3/5`、carrier retention`33/43`、Goal/Long均0、相对language/endpoints
  `-1/+3`、same-task retention`73.2%`，故Gate明确non-pass；所有bank、单checkpoint、唯一rank16、配对及Action Meta 0检查通过，
  shuffled/reversed未使用。
- [x] 对同一真实fit K1 bank做固定条件、梯度分解和teacher-only反事实：macro5相对step0未改善最终paired update，spectrum反而更差；
  旧functional对selection的梯度范数约为teacher的`67x`，teacher spectrum与旧scale梯度cosine为`-0.9897`。因此最早失效接口是
  同一步内相互冲突的shared selection/scale credit，而非Program、native bank、rank、K聚合或Action Meta；不能沿v2混合loss直接续训。
- [x] 在同一真实target20 bank上完成free-query与解析dual对照：普通梯度500步仍仅`0.1624` update cosine，FP64 weighted
  inverse-covariance dual在冻结G2 measure下达到`0.9975`；缩到最大logit`0.1`后用现有antithetic softmax仍为`0.9975`。
  因此最早接口是高条件数dual的shared acquisition，不是banks、signed pooling或chunked实现。
- [x] 从clean pushed detached `e7d86b0`完成fit-only、按task leave-out的四family target-native dual-basis解析oracle：50 tasks、
  98个确定性K1 conditions的full-dual reference task-mean update cosine median/p10/min为
  `0.996949/0.995468/0.993884`；但最大128维LOTO basis只有`0.288444/0.249615`，`0/50` tasks达到`0.95`，q/v/action-out
  family median分别为`0.000490/-0.000586/0.146885`，故Gate明确non-pass。bank-conditioned functional least-squares虽证明旧欧氏
  投影不正确，128维仍不足，扩展raw/effect basis又需要约384--512维；因此淘汰compact fixed dual/effect basis，不扩到38 targets，
  不恢复fixed realizer。
- [x] 对owner-native direct-key候选完成same-task三video的raw-query、event-query、event-anchor、nonlinear content-key和直接score
  acquisition反事实。teacher update与G2 Program跨video稳定，但minimum-norm dual随bank covariance旋转；2000步已能明显拟合train
  score，held q/v仍弱或负相关，完整held update约为`-0.001/-0.003`，action-out也仅`0.114`。因此不把逐video analytic
  dual/score、raw native key或event query当作已确定的canonical G3修正，也不通过续训或width/LR/seed扫掩盖结构问题。
- [x] 全新专家已基于远程`main@ed2883b`及其完整可达历史复核结构分叉；1538行原文逐字保存为
  `docs/expert_review_20260826_bank_conditioned_native_factor.md`。裁决保留G1/G2、真实banks、exact signed pooling、rank4、small-core
  SVD、carrier12与唯一rank16；G3改为B0累计current-bank statistics/native anchors、regularized solve、B1重放exact pooling的
  bank-conditioned compiler，并在同一实现中保留一次`global_statistics_off`决定性消融。
- [x] owner明确Final保留整套Writer完全随机初始化、直接端到端fresh联合训练的正式可选项；G1--G3只作因果验证，不构成Final
  强制课程。该裁决覆盖专家“默认从通过组件初始化”的偏好，但不改写专家原文。
- [x] F0：clean pushed detached `19b5b3f`完成真实K1/K4 bank-conditioned forward/gradient/materialization。Program/source冻结、
  Action Meta 0、K4 teacher reads 0、76 tensors/38 targets及policy实际消费唯一rank16全部通过；同一cached native bank的chunk4与
  one-chunk最终更新cosine最低`0.99999976`、相对误差最高`0.00066443`，K4置换误差`1.43e-6`且权重严格均匀。raw rank-slot
  最大差`0.00311`只记录为small-core SVD gauge诊断，不再错误充当最终LoRA等价Gate。
- [x] F1：以50 tasks/98 conditions的既有authority完成operator capacity；free/analytic native anchors隔离shared mapping，要求q/v/
  action-in/out各family materialized与streaming replay median update cosine`>=0.995`、minimum`>=0.99`，chunk/full等价。若显式
  covariance实现不能恢复而materialized FP64 reference能恢复，才根据operator证据切换matrix-free block-CG/Lanczos。clean
  detached `435cb4a`的四family task-mean median为`0.999871/0.999824/0.999960/0.999884`，minimum为
  `0.999757/0.999544/0.999951/0.999743`，chunk/full row minimum`0.99999988`；Action Meta 0、held reads 0，Gate pass。
- [x] F2：从clean pushed detached `2199a76` fresh完成`global_statistics_off/C=I`到macro5/25 updates，并由六个独立worker完整评估
  50 K1-covered tasks/451 task-video（329 fit、40 held-video、82 task-holdout）。fit/held-video/task-holdout task recovery median为
  `.022243/.022858/.018919`；held-video action-in/action-out/q/v median为`.039958/.022185/.004722/.023158`，三个primary checks
  全部失败。Action Meta、held gradient及shuffled/reversed use均为0。由于fit本身已失败，结论是candidate-local first-moment
  compatibility没有形成，不继续F2或做LR/seed/width小扫；后续actual-operator证据指定v4后已删除off执行面，formal evidence保留。
- [ ] F3：开启bank-conditioned solve训练shared mapping；held-video recovery median`>=0.75`、p10`>=0.50`、train/held ratio
  （实际口径为held-video task median / fit task median）`>=0.8`且相邻checkpoint稳定。不得恢复逐video dual/score监督、task/video
  lookup、FactorHead或fixed realizer。clean detached `c1e26ce`已从fresh训练至macro5并按原world6 topology exact-resume至
  macro10；同一451-condition held median/p10由`.048433/.037740`升至`.089704/.072144`，held/fit由`1.019827`变为`.997650`，
  泛化和相邻task delta稳定但绝对Gate仍明确失败。macro10 held family median action-in/action-out/q/v为
  `.125947/.177230/.013288/.052761`。同checkpoint factor/gradient分解已定位到update-only双线性credit造成两侧subspace acquisition
  starvation：q/v的一侧span ceiling与key gradient显著弱于action families。首个单变量修正保留完整update选出的一个global member，
  对其input/output gauge-invariant subspace与paired update固定等权；不改变rank、width、group gain、LR、seed、data或Gate。六卡真实
  5-macro qualification三项loss均连续下降且Action Meta 0，但clean detached `84903aa` fresh macro5/macro10的451-condition
  held median/p10最终只有`.073029/.057174`，held/fit `.998320`；macro10四family仅
  `.098990/.146806/.008482/.040693`，所以equal-subspace credit作为充分修正已被formal证伪。跨family及两个独立fit condition的
  固定target gradient分解进一步显示近正交目标和最高约`20.5x` sensitivity imbalance；family/fixed-owner修正已由clean detached
  `c3fc8e3` formal到macro10，但fit/held/task-holdout仅`.074715/.074620/.081644`、held p10 `.058381`，仍明确non-pass。
  fixed-key/raw/FiLM tangent和direct-native F0后续复核又证明required directions处于极弱奇异尾，且direct-native solve代数上退化为
  raw-query transfer，因此没有启动无信息的`4117117` formal。same-task三video bank-global oracle显示共同feature code的transductive
  q/v/action-out约`.90--.93`，但两video minimum-norm inductive近零，最早接口为task-stable code识别。当前单一修正使用冻结G2
  exact-language-only `P_lang`生成same-task稳定anchor query，动态Program只控制event/frame measure，并对每video、每event candidate
  features做detached symmetric inverse-square-root；仍保留真实X/Y、native solve、两softmax之差与唯一rank16。clean detached
  `20acc33`的F0通过，fresh macro10完整451-condition fit/held/task-holdout提升到`.141080/.142120/.145828`，40/40 held tasks从
  macro5改善且held/fit为`1.00737`，证明稳定anchor修复了迁移但仍未过`.75/.50`。六task单任务20-step probes的另一fit/held均跟随
  train，但q仅`.0197--.0277`；task93 q的18个target在shared input/output query head上合成梯度只有norm和的`.272/.268`，而
  candidate trunk已有fixed-owner FiLM。当前单一修正因此给query trunks增加zero-init bounded fixed-owner input及fixed-owner/
  output-group FiLM；不重训G2，不改rank、bank、loss、data或Gate。实现通过184项CPU回归和architecture hard checks。
- [x] fixed-owner/group query FiLM formal F0：`7e232b0`首次运行在GPU计算前发现内部`_apply`覆盖PyTorch模块生命周期方法；
  `d64f7ad`以唯一rename修复并新增`.to(device)`回归。clean pushed detached `d64f7ad`随后通过真实K1/K4
  forward/gradient/chunk/materialization全部资格项，新input/output owner-query gradients为`.015828/.000958`，Action Meta 0，
  K4置换误差`1.91e-6`，chunk有效更新cosine最低`.99999826`且相对误差最高`.001863`，唯一38-target rank16被policy消费。
- [x] clean pushed detached `3e4e9a0`的fixed-owner/group query FiLM已从fresh运行macro5并按相同world6 topology exact-resume到
  macro10，451-condition fit/held/task-holdout为`.162011/.163128/.164562`、held p10 `.133783`；40/40 held tasks相对stable
  anchor改善，但增量主要来自action-in，`.75/.50` primary与相邻Gate仍non-pass，不续macro20。
- [x] 在保留F1 operator、G2 Program、rank、bank、data与Gate的前提下完成candidate-key/compatibility image解析容量裁决。浅层
  target0/1在`1e-6`谱尾可达约`.994/.997`，但layer9 target18/19仅`.5186/.5583`、layer17 target34/35仅
  `.6537/.6079`，相同direct-native reference仍约`.995--.997`；单target的多种pair-credit对照也只到约`.06--.21`。因此首因是
  线性dot-product score image对深层input失容，而不是再调loss、owner projection、LR或训练步数。
- [x] 用一个canonical family-shared additive joint compatibility替换“只有线性点积”的限制，同时保留点积残差及既有Program、
  candidate encoder、B0/solve/B1、rank、data、loss和Gate。先完成定向CPU合同、真实K1/K4 F0、显存/吞吐profile及Action Meta 0证明；
  通过后从clean pushed detached commit fresh运行相同451-condition F3，不把内部容量或loss下降冒充mapping Gate。首轮
  `a2a56a7` F0只在chunk有效更新上失败；固定bank因果对照排除单纯scale并证明antithetic signed rows加`.03`非零joint residual
  可恢复合同。clean pushed detached `e784eb9`完整F0的chunk cosine/relative error为`.9999965/.002641`，K4、全部joint梯度、
  Action Meta 0和唯一rank16均通过；world6真实一步为`89.83s`、单卡峰值约25.60GB，现从fresh进入相同F3。
- [x] clean pushed detached `55710bb`的joint compatibility已fresh完成macro5/macro10及完整451-condition F3；macro10
  fit/held/p10/task-holdout为`.126205/.128720/.103610/.129465`，Gate non-pass且低于`3e4e9a0`。四task path ablation证明
  dot-only更新几乎不变、joint-only recovery近零；wrong Program替换证明当前query近task-agnostic。task85跨三video final-factor
  对照又证明action-in可达约`.996`，但浅层q/v只有`.42/.49`、layer9 held约`.24/.22`、layer17 q held`.13`，raw-native
  projection也不充分。当前停止该checkpoint与macro20，不启动新formal；下一实现前先用fit-only最小正对照同时证明full Program
  task content、实质nonlinear interaction及condition-stable深层q/v factor acquisition，不能继续点积残差小修或超参扫。
- [x] 完成40-task fit-only稳定functional target审计：只用mapping-fit videos形成task/member rank4 consensus update，在预注册held
  video上的overall recovery median/p10/min为`.945820/.935622/.908717`；member set上界几乎相同，证明旧`.13`不是teacher
  inconsistency。固定target native input谱又显示family-shared top128对q/v仅约`.797`，target-owned top128在抽查深层targets为
  `.959--.976`，给出ownership修正的直接机制依据。
- [x] 用一个canonical replacement接通full G2 `rank_event` query、38-target native basis后接family trunk、无residual bypass的
  normalized bilinear compatibility，以及fit-video consensus paired-update-only mapping credit；原逐video input/output subspace只作
  诊断，mapping held video严格排除在consensus之外。真实K1/K4 smoke通过chunk/置换/唯一rank16/Action Meta 0及全部主路径梯度；
  world6真实一步profile覆盖6 tasks/12 K1 conditions，耗时`91.44s`、peak allocated约`25.65GB`，所有Q/K/owner/gain probes
  finite/nonzero。首次profile精确暴露旧P_lang-only稳定神经支路64个无梯度参数，已删除该冗余旁路并以确定性P_lang+owner/rank
  稳定视图修复，未放宽跨卡梯度完整性检查。
- [x] 上述唯一实现面已集成至clean pushed `main@3062de8`，并从detached frozen worktree fresh完成macro5/25 steps及完整
  451-condition F3。fit/held-video/p10/task-holdout task recovery为`.084298/.082754/.072027/.093856`、held/fit
  `.981684`；held q/v/action-in/action-out为`.020707/.065711/.084290/.171636`。primary Gate明确non-pass，不续macro10。
- [x] 完成non-pass后的结构根因审计。fit-only universal rank4在held-video/task-holdout上已达`.825054/.835443`，证明当前
  residual label含有很大的task-independent common correction；`carrier12 + universal4`重新压回rank12的update cosine为
  `.998741`，但把common项直接从task residual相减后，task85 q/v在真实bank中的解析input可达性只剩约`.828/.765`，所以不能把
  代数重心化冒充native-feasible新teacher。四task因果干预又显示wrong Program后的最终update仍为q/v/action-in/action-out
  `.973/.981/.992/.948`，而wrong bank已降为`.863/.834/.9999/.569`；Program路径虽连通，却没有成为task selection owner。
  task-local两fit-video正对照中，current keys对held q/v input subspace只到`.188/.177`，新target-native pointwise projection也仅
  `.171/.130`，相同原teacher的direct native reference约`.997`。fit-only backward约`99.88%`原始gradient energy落在candidate
  encoders/trunks。该证据链同时定位到错误carrier/residual分解与pointwise functional canonicalizer/Program acquisition，不是
  train/held泛化、operator、chunk、Action Meta、欠训或普通超参问题。
- [x] 在继续改结构前复核F1与canonical B1的数值口径，发现F1解析上限使用FP64，而runtime继承source初始化的TF32；既有F0只比较
  两条相同TF32路径，未能发现共同偏差。真实深层q、v与action-out的受监督native-anchor panel中，IEEE FP32相对FP64的最大
  update-cosine绝对误差仅`7.4e-5`，TF32误差median约`.52--.68`；held learned-anchor recovery分别由TF32
  `.256/.178/.262`恢复为IEEE `.705/.798/.673`。旧`3062de8` checkpoint改用IEEE/FP64只读重放仍约`.08165`，所以不得
  post-hoc冒充成功，必须从fresh检验正确梯度。
- [x] 将唯一canonical compiler的native dual score/reduction固定为IEEE FP32并保持该process setting穿过backward；不改Program、
  bank、query-key公式、loss、rank、data、optimizer或Gate。clean pushed `main@78b7e58`已通过`186`项CPU回归；4卡真实fresh一步
  profile覆盖固定6 tasks/12 K1 conditions，全部主路径gradient finite/nonzero、Action Meta 0，耗时`123.62s`、峰值约
  `25.65GB`。
- [x] clean pushed detached `78b7e58`的真实F0通过新增IEEE数值资格：TF32实际关闭，chunk4/one-chunk最低update cosine
  `.99999955`、最大相对误差`.000945`，K4置换误差`1.43e-6`，全部关键gradient非零，Action Meta 0，唯一38-target
  rank16被policy实际消费。
- [x] clean pushed detached `78b7e58`已从fresh完成IEEE F3 macro5与完整451-condition primary。fit/held/p10/task-holdout为
  `.086508/.083131/.072629/.096191`，held/fit `.960958`；q/v/action-in/action-out held median为
  `.021698/.065269/.085933/.173804`。除held/fit外Gate均失败，证明TF32是必要修正但不是`.08`量级shared mapping的主因。
- [x] 沿实际post-`Wk` bank把旧Euclidean query坐标与真实`C_r C_0^+ H` functional image逐层对照。深层q/v与两个action
  family的task-local functional-polar见证约为`.996/.999/1.000/.998`；跨rank共享polar使v/action-out降至`.915/.831`，raw
  non-whitened chart使q降至`.911`。因此最早接口是Program query在错误metric中被单位化，而不是bank/key/rank/G2/optimizer失容。
- [x] 对唯一v4 functional-polar实现完成真实K1分段profile和有边界的执行优化：`da3fd3e`复用单次frozen X/Y capture、按shape合批
  functional polar、以IEEE FP32累计/求解并用thin-QR small SVD；全仓`189 passed`。condition由`82.114s`降至`58.332s`，但25-step
  macro5在六卡上的理想训练下限仍约`49min`且未含451评测，故吞吐资格non-pass；未运行K4 F0、formal F3、训练或评测。
- [x] 全新专家已锁定远程`main@9b52e59`及其可达历史，完整审计G3 formal/diagnostic evidence与full-polar profile；1033行原文逐字
  保存为`docs/expert_review_20260828_g3_functional_sketch.md`。裁决full functional-polar只作fit-only teacher/reference，当前唯一
  deployment候选改为low-dimensional bank-adaptive sketch与轻量shared student；不再发射full v4 F0/F3。
- [x] S1：接通F1 condition authority、sealed fixed nested projection、current-bank native/key cross-image、`r_s={16,32,64}`共享前缀、
  `C_rQ` full-native free-query与exact signed replay。clean detached `27bde62`的task93/q20 formal早停反例中，同条件F1 positive为
  `.99556--.99791`，rank64仅`.15669--.15744`，chunk最低`.9999769`；因此含row minimum`.95`的容量Gate确定non-pass，不再运行其余
  96 conditions，也不把native-Q sketch训练成shared student。该早停不估计全panel分布。
- [x] S2：按专家的S1失败分支检验不经过`Q_g q_tilde`native瓶颈的pure low-dimensional set-summary student；task93/q20最小
  正控已足以触发合取Gate non-pass，故未扩大到原计划6 meta+6 target。
  原预注册面为固定6 meta+6 target，
  其中每role各1个true task-holdout；其余tasks两条fit video、一条zero-gradient video-holdout。scale、G2/source/carrier冻结；先在
  同一candidate-logit/exact-X/Y执行面跑task-local free-query正控，正控通过后才训练shared full-Program+bank-summary query。shared
  结果前用universal negative、free-query positive和`78b7e58`失败checkpoint一次校准并sealed absolute/causal Gate，禁止按结果移动。
  首先只跑task93/q20机制witness：同一task-local code共同拟合两条video，第三条严格zero-gradient；fit median`>=.90`、held及input/output
  pushforward各`>=.80`、held/fit`>=.8`才进入12-task正控。首版使用measure-normalized mean/variance DeepSets summary和共享bounded
  candidate scalar energy，冻结现有candidate encoder；失败只淘汰这一明确函数类。
  - [x] 接通共享frozen native-bank runtime、separable scalar-energy、exact signed pooling、固定final-step runner及三视频真实gradient smoke；
    31项定向合同通过，Action Meta和全部旧authority实际冻结。
  - [x] clean detached `4d84dee`完成首轮1000-step witness：fit/held仅`.328/.175`。nested free-logit oracle在global/eventwise均约
    `1.000`，而固定或fresh 128D score basis即使移除summary映射并加入强factor credit仍不超过约`.36`；首轮runtime进一步确认所谓
    existing candidate encoder实际是未加载checkpoint的fresh seeded projection，因此该结果只淘汰这一错误authority组合。
  - [x] 单一修正为显式加载并冻结`78b7e58/macro5` fit-trained candidate encoder/trunk/metadata/key projection；保持summary、score、loss、
    videos、1000 steps与Gate不变。clean detached `6b97100`的v2 fit/held仅`.349/.132`、held/fit`.377`，全部Gate失败；相对随机chart
    fit只增加约`.021`。因此不进入12-task/shared训练。
  - [x] 完成不保留的candidate-chart acquisition诊断：从`78b7e58`初始化并只解冻q20所选chart的`363,520`参数，与scorer/free code
    合计`2,648,100` trainable；正确的`rho * event-volume`测度下1000步fit/held仅`.30286/.03527`。因此不形成“解冻chart”formal
    修正，首版128D mean/variance separable scalar-energy函数类已淘汰；bound/branch/diagonal与低秩operator对照也均未恢复。
  - [x] 完成唯一一个机制不同的candidate score函数类：task-local rank/event query直接多步cross-attend当前candidate set，形成
    query-conditioned set state后再逐candidate产生bounded signed logits。它必须保持集合置换等变、正确event-volume mixing、真实X/Y
    exact pooling、跨三视频共享free code、video0零梯度及Action Meta 0；不得含task/video/frame lookup或full covariance/eig/SVD。
    strong credit与按专家最小目标修正的update-only credit分别只到fit/held约`.147/.097`与`.152/.092`，Gate明确失败；按实测秩指定的
    rank224/384 cross-image sketch也仍约`.159--.163`。因此S2 pure low-dimensional score函数类关闭，不进入12-task/shared训练，
    不扫token/width/depth/LR/bound版本链。
  - [x] S2b behavior-identifiability诊断完成：授权fit task93的cross-episode flow loss为`.09911`，真实q20 rank4 descent降至`.08802`、
    ascent升至`.11481`，rank4捕获dense gradient能量`.8058`。两fit及零梯度held video的optimistic signed update recovery为
    `.9104/.9044/.9024`；旧query-conditioned scorer训练后held仅`.0229`，bank-independent native dual held仅`.0745`。因此问题仍是
    bank-specific dual坐标，而非LoRA teacher独有；没有读取validation/test或使用shuffled/reversed。
  - [x] 单一机制反事实把同一behavior primal按每条bank全局covariance对偶化，并以同一全局measure replay：IEEE下fit为
    `.9112/.9043`、held为`.9005`，三条bank solve总计`.734s`；eventwise replay反而失真。该结果选择Program-primal→current-bank
    global dual接口，不恢复旧functional polar或继续scorer版本链；它仍只是one-task/one-target诊断，不冒充G3 Gate。
- [x] P0：clean pushed detached `e2f9d33`的真实38-target K1/K4 F0已通过；两次流式read、四类Y边界、chunk一致、全部primal/
  Program梯度、Action Meta 0、source/G2冻结、uniform K、唯一rank16 materialization与policy consumption均成立。固定candidate
  microblocks使chunk4/one-chunk raw/solve/conditioning error为0，minimum update cosine为`0.9999999999999998`。
- [x] P1：clean pushed detached `c9e8198`的六任务formal通过；fit/held median为`.971731/.954539`、held/fit`.982308`、
  held/optimistic`.992193`，q/v/action-in/action-out held family medians为`.939825/.941630/.995402/.945222`，minimum task held
  `.935001`。六task、四family、全部信息墙及fixed-step证据完整，scale与held保持零梯度；这只证明task-local primal current-bank
  dual/replay容量并解封P2，不等于shared mapping或闭环通过。
- [x] 旧P2执行面已接通但被joint functional裁决取代：共享full-Program-to-primal scorer、fit-only shared rank-scale template、run-local compact frozen-condition cache和完整
  451-condition evaluator已接通；真实同bank compact/chunk误差`2.384185791015625e-07`，cache-hit 38-target forward/backward
  `2.625s`，Action Meta/scale trainable均为0，全仓`198 passed`。clean detached world6 cold/hot step为`24.84/6.177s`，六卡均各处理
  一个task且12条cache build/hit、gradient和3+3 role合同完整；没有启动其frozen-Program formal。当前复用其cache/operator/scorer，
  由J2联合Program与primal并以generated-LoRA functional Gate取代旧factor-space primary；内部loss仍不能代替。
- [ ] 条件式做fit-only decomposition-feasibility oracle：仅当P2 shared mapping已显著取得selection而残余证据仍指向common
  correction/carrier时，只用授权fit tasks形成shared correction并候选重拟合carrier12；随后针对
  **新carrier**从完整expert update重新计算每task residual，再投影回每条真实native bank。必须同时证明carrier压缩/retention、
  四family native direct/free-code可达性、跨video consensus与唯一rank16；若不成立，不保留该carrier，不能直接复用代数差分factor。
- [ ] 条件式只有P2已取得task-specific selection而残余证据仍指向decomposition时，才重开carrier/common correction；不得绕回逐video
  dual、task/frame lookup、高维factor head，也不得用universal shortcut、内部loss或普通超参扫通过Gate。candidate basis、Program和
  scale必须保持明确parameter ownership；rank spectrum/scale只在selection取得后用隔离credit处理。
- [ ] F4：恢复全部75 fit tasks的scale/functional/flow/preservation职责；mapping loss保护selection，scale/video独立更新；teacher
  paired update不退化。只有mapping已学会而低置信随机residual仍破坏carrier时，才加入deployment-visible confidence退回机制。
- [ ] F5：按K1到K2再到K4恢复多视频职责，K2/K4 teacher reads保持0；验证K1 identity、集合置换不变、bounded beta和same-task
  mapping retention`>=80%`。
- [ ] F6：冻结单一checkpoint执行held5 carrier/language/full/first+final/same-task五臂strict250；沿用现有G3 Gate。formal runtime
  使用固定3+3 role-balanced全局task group并按实时1--6张有效GPU弹性分片；world size不得改变task权重、optimizer cadence或Gate
  口径，exact-resume锁定原topology。

### G4 Joint Writer

解冻全部Writer、冻结backbone/carrier/experts；默认直接联合优化已通过组件Gate的完整Writer，并优先保留支撑闭环因果问题的最小loss
集合。G3的native-feasible LoRA teacher是组件验证工具，不是G4/Final必须存在的数据依赖；正式训练不得预设有目标LoRA，可直接以
授权fit/meta teacher actions、functional与on-policy闭环信号监督完整Writer。只有真实机制证据表明需要时才使用有退出条件的functional
warmup；随后可在student visited states上用verified short
continuation筛选有效expert members。至少两个train24 folds要求recovery`>=0.40`、breadth5/5、Goal/Long均非零、carrier
retention`>=75%`、same-task retention`>=85%`和相邻checkpoint稳定。

### G5 Structured outer credit（条件阶段）

只有G4已证明full高于carrier/language/endpoints且breadth成立才进行。只更新event/Program/rank attention/scales；最终outer节点相对G4
净增至少10且breadth、Goal/Long、same-task retention不下降。无提升时按credit机制证据分析、修正或更换estimator，不设次数上限，
但不做无信息超参扫。

### Final fresh与正式评测

使用全部71 meta+train24，role仍50/50；Final保留两种fresh初始化候选：（a）从已验证Program/compiler初始化后完整Writer joint
training；（b）整套Writer完全随机初始化后从头直接端到端joint training，让梯度下降自行形成内部功能分化。两者均用fresh
optimizer/scheduler和同一信息墙、数据及closed-loop合同。G1--G3的冻结分段只作架构因果验证，不机械重演为最终训练流程；若联合
训练证据要求短暂分段预热，必须登记最早失效接口与退出条件。loss默认从最小充分集合开始，仅在closed-loop或明确机制证据支持时
增加；不预设Final有目标LoRA。只有base joint通过时才启用已验证的outer。
validation8只比较三个预注册相邻checkpoints；资格arm先跑correct/same-task/language/first+final。冻结selected后补完整controls，最后才跑
shuffled/reversed；方法完全冻结后只打开Test8一次。

`current_owner_requirements`中的32-task fresh refit与上述71 meta+train24 development recipe的精确顺序待Final前owner裁决；
该冲突不阻塞G1--G5，当前不为任一解释启动数据合并或训练。

## 路线边界

- 不恢复人工process dataset/controller acquisition；
- 不训练neural `q_pi`，不恢复fixed-effect/two-sided realizer；
- 不把GOMQ、PECS、v24或历史solver当ECP前置；
- 不在G1前训练fresh Stage 0、shared compiler、joint Writer或outer credit；
- 不用loss、geometry或checkpoint union替代single-checkpoint closed loop；
- shuffled/reversed只在最终selected checkpoint选定并冻结后测试时序特异性，不进入训练、loss、checkpoint选择、
  G1--G5 Gate或架构修正依据；
- Action Meta只在base Writer有闭环增量后做matched controls，只有明确净收益且无breadth/retention损害才启用；
- rank12 carrier + mobile rank4是G1的首版可证伪配置，不是永久硬约束；只有active design登记的rank-ceiling证据链成立才重开分配；
- 不人为限制各阶段时间、修正次数、结构版本或训练轮数；遇到scientific non-pass先按Gate定位接口，有新证据就修正，无新机制的
  slot/width/rank/seed版本链不算有效尝试；
- 优先复用、并行和提高吞吐，进展顺利时力争数天内完成整体架构实现并推进关键Gate。
- GPU显存不设`35GiB`或其它人为硬上限；以最长真实样本稳定不OOM、allocator波动及共驻进程仍有安全余量为边界，按实测吞吐和持续
  UTL选择microbatch、chunk与worker数。已完成实验中出现的`<35GiB`只记录当时profile合同，不约束后续运行。

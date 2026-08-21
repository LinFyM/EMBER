# Fixed Functional Adaptation Successor

状态：2026-08-21 active design；统一fingerprint上的flow/probe/exact Decoder与随后phase/learner-state Decoder均未通过held
support门，当前按专家挑战十二裁决稳定shared prior + task residual。本文只定义第二轮专家意见后继路线的当前可执行设计；旧LMMPC、V6、LPCP、GOMQ与
Expert-Manifold细节只从`docs/research_history.md`和对应sealed artifacts解释，不恢复为并行主线。

## 1. 核心假设

当前失败不是“视频完全无效”或“LoRA容量不足”，而是把两个本应分开的学习问题放进了同一个持续移动的坐标系：

1. 哪些完整LoRA在frozen source policy上实现了不同task skills；
2. exact language与action-hidden视频说明当前task应选择哪个skill。

后继路线先用train/meta privileged evidence学习并固定`code -> complete LoRA`，再训练`language/video -> code`。decoder
固定后，视频encoder无法再通过共同旋转latent与FactorHeads来降低loss；它必须预测一个已有policy-functional意义的code。

## 2. 完整部署流水线

```text
exact language
    -> frozen native text encoding
    -> learned language prior z_L

each action-hidden video
    -> frozen native image/language/Action carrier
    -> objects + initial/goal relations + ordered events/subgoals
    -> per-video posterior evidence

K independent video evidences
    -> permutation-invariant evidence aggregation
    -> delta(L, V_1:K)

z = z_L + delta
    -> frozen FunctionalAdapterDecoder
    -> one complete 38-target rank16 LoRA
    -> merge shared/base and residual contributions before rollout
    -> frozen source policy closed-loop rollout
```

Writer只在rollout前运行一次。部署不读取task ID、filename、teacher action/state/reward、expert bank或第二adapter。

## 3. 数据角色

`configs/libero90_nonheld_meta_v1/protocol.json`是后继meta-task authority：

- 71个已完成target40语义去重的LIBERO-90 tasks全部保留为source skill pool；
- 五个稳定task-level folds只按audited active task顺序分配，不读取loss、reward或policy outcome；
- 默认fold0作meta architecture validation，fold1--4训练；架构与干预冻结后轮换五fold复现；
- train/meta action、state、reward和task experts可以训练decoder、code inference、action alignment与outer objective；
- target validation8不产生梯度，并明确是长期复用的development panel；
- target Test8默认在方法冻结后才使用，sealed诊断不回流checkpoint选择。

source policy已在同一71-task clean corpus上建立primitive prior。这个重叠pool可作复用和定位面，但不能单独证明对新task的
adaptation meta-generalization。若后续coverage审计证明Long所需primitive缺失，才强化source并重建全部matched基线。
首个fold0 source formal为`646/750`，其中14/15 tasks达到80%--100%，因此该重叠本身也是需要控制的欠识别源：后续decoder
range不能以接近direct expert的aggregate或高absolute单独通过，必须在同一750行上分解source→direct与source→projected的
retained/gained/lost/churn及per-task增量。若uniform-step direct experts没有提供跨task的policy-effective净增量，则当前
source/meta重叠pool不能验证decoder泛化，触发role-disjoint meta-task构造，而不是把identity/source能力记为decoder成功。
当前flow/probe/exact Decoder分别得到`644/未评测/638`，shared-zero为`640`；task-conditioned输出始终未超过共享输出，
所以该触发条件已经成立。后继必须把source-skill、adaptation-meta和architecture-validation task identity分开，并优先复用
source从未训练过的target train24及现有多checkpoint experts；不会仅因数据角色变化重训已有71-task bank。

第一步先做不增加训练成本的最小role-disjoint诊断，而不是立即重训source：当前frozen source从未读取target train24，故固定
`ordinal mod 5`的19/5 fold把19个target train tasks作为adaptation fit、5个作为leave-task-out diagnosis。统一anchor只来自
19个fit tasks，PCA/whitening也只在这19个task拟合；code宽度受独立task数约束为16，held5只经过同一transform。这个小面板
仍不足以支持通用compiler claim，但能先隔离“source/meta identity重叠”这一混杂。只有它与validation8 ceiling共同给出正向
证据，才启动fresh source-skill/adaptation-meta/architecture-validation大规模分割与兼容expert重建。

## 4. Policy-functional task targets

### 4.1 Task experts

- 复用现有task-local rank16 LoRA训练、checkpoint和闭环评测基础设施；
- 为71个non-held meta tasks训练统一schedule的task-local experts，不按task挑checkpoint；
- 每个task至少保留一个全局统一step的successful adapter；若多个统一step都成功，可作为同一功能等价类样本；
- train24已有step250/500/1000/1500/2000的同轨迹checkpoints与严格50-state closed loop；首个set-valued诊断以固定
  `>=25/50`为成功集合，未过阈值task只保留其最高统一step作ceiling定位。fit-task集合可形成监督prototype，held-task outcomes
  只定义无梯度诊断集合，绝不参与decoder或checkpoint选择；这些checkpoint不是独立seed，不能单独证明广泛非唯一性；
- expert只服务train/meta decoder与诊断，不进入target held部署；
- validation8只允许训练一次彼此独立、step2000预注册的task-local诊断experts，不更新共享模型、不选checkpoint；Test不训练
  expert。两者都不得成为部署route，Test actions/reward默认保持sealed。

### 4.2 Functional probe panel

每个expert与generated adapter在完全相同的task-local observation/language、flow time与noise上比较：

- `flow response`：Action Expert输出的50-token、32维velocity response，不先对50 tokens求均值；
- `action response`：固定noise与denoising schedule得到的action chunk，作为无梯度功能验证；
- `policy Jacobian response`：只在小型定位panel比较flow response对noisy action的JVP，不作为每步重训练负担；
- `stage behavior`：train/meta closed-loop的阶段完成、最终success与failure stage。

decoder warm-start可使用gauge-invariant effective `BA`误差，但最终通过条件是policy response与closed-loop功能，不是raw
A/B、rank cosine或reconstruction loss。

### 4.3 Unified functional fingerprints

首版“joint learned train codebook + held从零自由优化code”已由macro10裁决淘汰：train codes平均norm`5.589`，held codes
仅`.505`，不是同一推断分布。canonical code改由统一policy queries直接定义：

1. 只从固定meta-train task顺序等距选择anchor tasks，不读取loss、reward或outcome；
2. 每个train/held expert都在完全相同的anchor observation/language/action-flow queries上计算`expert - frozen source`完整
   `50x32` response，并拼成task fingerprint；held task自身action/state不进入anchor panel；
3. PCA basis、mean与scale只用56个meta-train fingerprints拟合，取32维并whiten到train covariance identity；
4. 15个held fingerprints只经过同一固定transform，不重新拟合PCA、不优化held free code；
5. 这些codes固定后才训练`code -> complete LoRA` decoder。task-local panels继续用于decoder functional loss与closed-loop
   验证，但不再决定latent坐标。

这样latent gauge由统一policy function与train-only whitening一次性固定；held code与train code天然同坐标、同尺度，Writer
面对的leave-task-out目标不再被near-origin shared carrier替代。

首个formal已验证该坐标修复：32维解释train variance `.887516`，train/held coordinate std为`1.000/.7248`、平均norm为
`5.570/4.144`。但以task-local flow response直接训练Decoder后，held flow loss虽降至`.664218`，closed loop仅
`644/750`；其effective `BA`相对direct expert的relative-L2为`2.8576`、cosine`.0254`、norm ratio`2.7004`。因此
fingerprint继续作为唯一fixed code authority，但“flow-only Decoder objective”已经淘汰：它能在有限policy queries上拟合，
却可通过巨大off-manifold update实现，不能被低functional loss授权进入Writer。

## 5. Fixed FunctionalAdapterDecoder v1

### 5.1 Code与输出地址

- 每个task的functional code为一个`32`维向量；默认meta训练tasks为56，因此code维度低于task数并可估计协方差；
- code table是上述frozen fingerprint codes的只读索引，只在decoder学习阶段按task ownership取值，绝不进入部署；
- PCA/whitening只在meta-train拟合并固定；decoder训练不再移动code，language/video随后预测同一固定坐标；
- 38个LoRA target与16个public rank各自拥有learned output address embedding；这些embedding只负责把一个functional code
  写入完整PEFT拓扑，不把rank、layer或action-in/out宣称为预先已知的功能坐标；
- action-in与action-out拥有独立target address，不再伪装成首末Action Expert layer；
- decoder输出端不使用会无条件抹除task/cell magnitude的RMSNorm。

### 5.2 Decoder拓扑

对每个`target x rank`地址，将同一个task code与target/rank embeddings拼接，经共享trunk后送入按factor width分组的
row heads，生成A row或B column。八种native factor widths共享对应head，不为每个task建立独立参数。

```text
[z_task, target_address, rank_address]
    -> shared address-conditioned trunk
    -> width-specific A/B row head
    -> 76 tensors / 38 targets / rank16
```

输出以deterministic identity template为基点。当前shared-prior裁决不在A/B factor坐标直接做全rank相加，因为那会在
`BA`中产生无法解释的交叉项；而是把public rank16精确分成`shared rank12 + task residual rank4`。前12个rank只由
task-independent shared stage写入，后4个rank只由zero-code-centered task residual写入，因此每个target的有效更新严格为
`B_shared A_shared + B_task(z) A_task(z)`。两块在rollout前以rank维拼成同一套rank16 A/B tensors，evaluation cache和
rollout永远只看到一个complete LoRA，不部署第二adapter、不做checkpoint fusion或LoRA平均。shared-only使用同一rank16
state，其中residual块保持identity功能；task code为零时composite逐tensor严格退化为该shared-only state。

### 5.3 两阶段训练与冻结

1. 先按4.3从统一functional fingerprints冻结train/held codes；held role只transform，不产生梯度；
2. 只在meta-train tasks按task等权训练decoder。flow-only由`644/750`淘汰；固定8-probe因train full-BA relative-L2
   `1.1387`淘汰；exact低秩Gram虽把train/held BA cosine提高到`.5365/.3032`，closed loop仍只有`638/750`且低于
   shared-zero `640`。因此不再围绕单expert BA或有限flow panel更换objective；下一版先重建多成功adapter功能等价类和
   role-disjoint task surface；
3. decoder训练期间held codes与held panels都不产生梯度；训练结束后在held task-local panel和closed loop测试range；
4. range通过后永久冻结decoder，held fingerprint code只用于oracle诊断，不参与后续Writer训练；
5. semantic/video encoder必须从action-hidden输入预测同一code，完成真正leave-task-out闭环验证。

当前挑战十二的独立裁决复用已经收集的fit19 successful/learner panels与held5固定rows，不重建state bank：第一阶段固定
zero code、只训练rank0--11，得到一套fit19-only stable shared prior；第二阶段冻结该prior、只训练rank12--15的
task-code residual，并令`D(0)`精确为shared prior。两阶段均为6-rank、912 task visits，先评测shared-only matched
baseline，再在同一held5上评测earliest/latest composite。只有composite相对shared-only产生稳定task-conditioned增量且重新
通过direct support/breadth/member-stability联合门，才进入language/video Writer；失败只关闭这个12+4 exact-additive
参数化，不关闭shared prior、occupancy或train-task outer reward的一般方向。

fully fixed是主线。只有decoder在多个meta folds都出现系统性range欠拟合、且扩大code而非重训坐标不能解决时，才运行
明确two-timescale/EMA对照；不能因为video inference困难就重新共同移动decoder。

## 6. Language prior与video process posterior

### 6.1 Language prior

复用PI0.5原生text-only prefix forward，保留exact language和真实token语义，不构造zero-image或fake Action query。
trainable projection输出`z_L`。learned language-only必须独立生成有效complete LoRA并正式评测；no-video不再强制identity。

### 6.2 Per-video process evidence

每条video独立保留：

- task-relevant object/fixture tokens；
- initial relation与desired final relation；
- 视觉transition、contact/change events；
- ordered subgoal/event sequence与completion evidence；
- 完整50-token Action probe序列，按flow time、horizon phase与noise probe读取；
- 过程不确定性，用于K-video evidence weighting而非挑选最好video。

旧Core/Procedure职责、task-grounded patch读取、Core-addressed Reader和bounded refinement可复用；旧20x16x256 grid、
time-centered Value、Action-token mean、rank-index memory与moving FactorHeads不进入新接口。

### 6.3 K-video aggregation

K个video先各自完整forward并得到event/relation/code evidence，再在video集合轴置换不变聚合。禁止平均frames、raw
features、完整LoRA或checkpoint。若支持Dynamic-K，训练真实覆盖K1--K4并使用nested无放回配对。

### 6.4 Train/meta teacher-action alignment

在action可用的train/meta tasks加入：

- visual transition -> latent action/inverse dynamics；
- event phase与Action-token/horizon phase对应；
- 同task跨episode的procedure consistency。

这些只帮助学习表示；target validation/test video仍action-hidden，teacher action不成为input或部署route。

## 7. Process-identifying controls

当前已接入strict evaluator：correct、same-task-other、cross-suite-wrong、shuffle、keep-first、reverse、no-video、
first-only、final-only、first+final、endpoints-fixed-middle-shuffled、monotone-sparse、frame-count-matched
static-first-repeated与同episode同时间的eye-in-hand cross-view。所有条件选择、重复或重排真实RGB frames后重新完整
forward，并保留原始source-time位置语义；cross-view不读取action/state/reward。

HDF5审计确认只有agentview/eye-in-hand RGB，没有depth或segmentation；因此robot-mask/object-mask不能由现有输入诚实
产生，不能读取teacher state重渲染mask来绕过信息墙。后续仍需可信RGB-only motion表示、language paraphrase、
same-object-different-goal、stage success与same-endpoint-different-procedure数据；当前metadata不能证明same endpoint但
不同合法procedure，不能用随意剪帧冒充。

full video必须稳定优于learned language-only和first+final；否则只能声称使用appearance、goal、端点或粗motion，不能
声称学到Procedure。

## 8. Train/meta closed-loop outer credit

decoder与functional warm-start固定后：

1. train/meta simulator从correct action-hidden videos生成一次LoRA；
2. policy闭环rollout得到success、progress/stage与failure evidence；
3. 优化language/video code inference，不更新source policy和fixed decoder；
4. task等权聚合，和matched functional-only arm比较；
5. held validation仍zero-interaction，reward不产生梯度。

优先直接使用simulator success/progress；若训练方差要求critic，critic只在train/meta拟合且必须用held closed-loop快速裁决。
不再把raw parameter PCGrad、norm guard或functional loss当作最终外目标。

生成LoRA后的task-local RL是单独extension：先记录zero-interaction初始化，再报告达到成功所需episodes，不能混入核心分数。

## 9. Source coverage与ceiling

source coverage分reach/grasp/place/open/close/toggle/stack/multi-object sequencing/recovery报告。证据顺序为：

1. frozen source baseline；
2. non-held meta direct experts的uniform-step closed loop；
3. fixed-decoder oracle code；
4. predicted code；
5. 冻结后的sealed held flow/action/reward诊断。

owner已允许一次sealed validation8 task-local oracle。它只训练八套彼此独立、预注册step2000的诊断LoRA，不更新任何
共享Writer/decoder、不选checkpoint且不读取Test；其结果只能裁决frozen source + rank16的任务可实现性，不能成为部署
route或模型选择信号。若non-held多任务expert已经普遍失败，先修source；若expert成功而fixed decoder失败，修manifold；若decoder
oracle成功而predicted code失败，修task inference。
其中source已在default meta-validation fold得到`646/750`，所以第2--3步的通过对象是相对source新增并保留的功能，而不是
仅比较两个高aggregate。direct expert若不能证明有信息量的增量，第3步保持未识别，不得由source高分代替。

## 10. Formal selection

每个有信息量candidate必须在clean pushed commit的detached frozen worktree运行，并报告：

- single-checkpoint strict paired400；
- per-task、per-suite、Long、suite floor；
- breadth@1/@5/@10、top-task concentration；
- retained/gained/lost、churn、Jaccard和相邻checkpoint；
- same-task different-video success-row retention；
- source/language/video/L+V与完整process controls；
- default meta fold与冻结后的五fold复现。

约145是有价值而非自动pass；150+同样必须有breadth、stability与video causality。达到约145后立即补controls与相邻点，
不继续训练掩盖峰值。

## 11. 模块ownership与生命周期

- `ember.functional_adaptation`：新canonical successor owner，负责meta contract、process controls、fixed decoder、functional
  response、code inference与后续outer objective；
- `ember.expert_manifold`：保留task-local expert training/checkpoint/evaluation基础设施；`legacy_v6_*`与bank routing不被
  新模块import，最终只由Git/历史artifact保存；
- `ember.writer`：现有LMMPC是sealed baseline；通用video store、functional policy call、evaluation cache与rollout runtime
  可复用；
- `ember.pi05_eval`：继续拥有dynamic queue、paired rows与formal evaluator，新增successor adapter而不是复制evaluator。
- `ember.functional_adaptation.evaluation*`只实现successor特有的asset/episode合同与一次性LoRA生成；cache、worker handoff、
  rollout、aggregate和恢复仍由现有`ember.writer.evaluation_*`与`ember.pi05_eval`统一拥有。

过渡期允许sealed baseline与successor adapter并存以做matched比较，owner为本goal；移除触发点是successor通过fixed-decoder
leave-out与end-to-end smoke。届时训练入口和writer family registry只保留successor canonical，旧执行路径由Git与sealed
artifacts复现，不维持两个活动Writer。

### 11.1 结构门记录

- **新增边界的原因：** fixed decoder、task-level meta split与功能响应监督共同构成一个新的科研合同；把它继续塞进
  `writer`或`expert_manifold`会混淆部署Writer、privileged teacher与历史bank routing三种ownership，因此建立单一
  `ember.functional_adaptation` owner，而不复制policy/evaluator/data runtime；
- **已排除的近邻方案：** 不复活`legacy_v6_*`或task-ID route；不复制task expert trainer、PI0.5 evaluator或video store；
  不以raw A/B重建作为主目标；不让现有LMMPC decoder与新fixed decoder互相import；
- **过渡路径：** sealed LMMPC只用于matched历史比较，successor在独立入口训练。通过default-fold functional
  realizability和端到端smoke后，successor成为唯一活动训练/部署Writer；若未通过，则删除未证明的新入口并保留本轮
  诊断证据，不形成永久双轨；
- **复用责任：** task-local expert训练继续由`expert_manifold`提供公共训练核心，paired rollout继续由`pi05_eval`提供，
  `functional_adaptation`只提供新合同、decoder、probe、inference与objective。
- **训练入口生命周期：** train24机制profile已完成并冻结formal schedule；原parameter-only预热入口已删除，完整PI0.5
  flow入口已收敛为唯一`train_functional_adapter_decoder.py`。该入口同时拥有fixed decoder/codebook优化、task-equal cursor、
  optimizer/RNG exact resume、run contract与56/15 formal结果，不形成第二套Writer或部署路径。

## 12. 首轮实施范围与门

首轮只实现并裁决以下完整因果链，不同时堆outer RL或传感器变化：

1. 71-task meta contract、五fold与process controls；
2. compact-code fixed decoder及identity/single-LoRA合同；
3. expert-vs-generated flow response工具；
4. 先用现有train24 bank做机制/profile smoke；
5. 再训练non-held meta experts并做default-fold decoder range；
6. decoder range通过后才实现language prior/video posterior。

若train24 smoke连专家功能都无法重现，只修decoder/functional panel；若train24通过而71-task fold失败，定位task diversity和
source coverage；不能在fixed decoder未通过前用video encoder或outer RL掩盖问题。

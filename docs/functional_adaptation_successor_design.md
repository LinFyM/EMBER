# Fixed Functional Adaptation Successor

状态：2026-08-19 active design。本文只定义第二轮专家意见后继路线的当前可执行设计；旧LMMPC、V6、LPCP、GOMQ与
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

source policy已在同一71-task clean corpus上建立primitive prior。meta folds衡量的是从已有prior组合/选择task adaptation，
不是要求source从未见过所有primitive。若后续coverage审计证明Long所需primitive缺失，才强化source并重建全部matched基线。

## 4. Policy-functional task targets

### 4.1 Task experts

- 复用现有task-local rank16 LoRA训练、checkpoint和闭环评测基础设施；
- 为71个non-held meta tasks训练统一schedule的task-local experts，不按task挑checkpoint；
- 每个task至少保留一个全局统一step的successful adapter；若多个统一step都成功，可作为同一功能等价类样本；
- expert只服务train/meta decoder与诊断，不进入target held部署；
- target validation/test不训练task expert。其actions/reward只在授权sealed、冻结、无梯度诊断中读取。

### 4.2 Functional probe panel

每个expert与generated adapter在完全相同的task-local observation/language、flow time与noise上比较：

- `flow response`：Action Expert输出的50-token、32维velocity response，不先对50 tokens求均值；
- `action response`：固定noise与denoising schedule得到的action chunk，作为无梯度功能验证；
- `policy Jacobian response`：只在小型定位panel比较flow response对noisy action的JVP，不作为每步重训练负担；
- `stage behavior`：train/meta closed-loop的阶段完成、最终success与failure stage。

decoder warm-start可使用gauge-invariant effective `BA`误差，但最终通过条件是policy response与closed-loop功能，不是raw
A/B、rank cosine或reconstruction loss。

## 5. Fixed FunctionalAdapterDecoder v1

### 5.1 Code与输出地址

- 每个task的functional code为一个`32`维向量；默认meta训练tasks为56，因此code维度低于task数并可估计协方差；
- code table只在decoder学习阶段按task ownership索引，绝不进入部署；
- 对code做中心化、尺度约束与covariance-to-identity gauge loss；decoder冻结后，language/video预测同一固定坐标；
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

输出以deterministic identity template为基点；可学习shared base/bias与task residual，但forward立即相加成一套state，
evaluation cache和rollout永远只看到一个complete LoRA。共享base若未带来matched功能/稳定收益则保持identity。

### 5.3 两阶段训练与冻结

1. 在meta training folds联合学习task code table与decoder；每个macro按task等权；
2. effective-BA warm-start后尽快切换到expert-vs-generated full flow-response matching；
3. 固定decoder，在held-out meta fold只优化privileged oracle code，测试decoder range；
4. range通过后永久冻结decoder，held-out code只用于oracle诊断，不参与后续video inference训练；
5. semantic/video encoder必须从action-hidden输入预测code，完成真正leave-task-out闭环验证。

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
first-only、final-only、first+final、endpoints-fixed-middle-shuffled与monotone-sparse。所有条件重排/选择真实frames后重新
完整forward，并保留原始source-time位置语义。

后续在可信数据/segmentation可用时加入flow-only、static-only、robot-mask、object-mask、same-endpoint-different-procedure、
cross-view、language paraphrase、same-object-different-goal与stage success。当前metadata不能证明same endpoint但不同合法
procedure，因此该项明确是数据构造任务，不能用随意剪帧冒充。

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

target held不允许通过action/reward梯度训练oracle expert，因此“held ceiling”只能作为无梯度诊断边界，不能伪称精确
可达上限。若non-held多任务expert已经普遍失败，先修source；若expert成功而fixed decoder失败，修manifold；若decoder
oracle成功而predicted code失败，修task inference。

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

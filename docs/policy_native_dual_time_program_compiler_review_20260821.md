# Policy-Native Dual-Time Program Compiler

状态：2026-08-21 **已完成外部复核的历史送审稿，不是active implementation authority**。本文保留owner核心思考、最初
policy-native dual-time具体化方案及专家审查入口。专家随后纠正了hard video-time/action-horizon transport、deterministic
`P*`和target-family缺失等问题；owner最终采用的修正版为
`docs/event_conditioned_policy_compiler_design.md`中的EMBER-ECP。不得再据本文启动formal训练或held评测。

## 1. 本轮复核要回答的不是局部补丁

EMBER的输入输出目标不变：

```text
exact task language + K条same-task、action-hidden、内部有序的正确教学视频
    -> shared Writer在rollout前运行一次
    -> 一套完整38-target rank16 task-conditioned LoRA
    -> frozen source PI0.5
    -> 从未见初始化闭环完成同一task
```

这轮要重新判断的是：在PI0.5这个“冻结VLM理解当前静态语言/图像、Action Expert组织50步动作预测”的基底上，怎样让
教学视频的跨帧过程知识进入Action Expert自己的层次与时间结构，并从同一个结构直接编译为LoRA。我们不把问题重新定义为
`video -> pseudo action -> imitation`，也不建立部署时反复读取视频的第二策略。

本文给出一个具体候选，但希望专家可以推翻其任何方法细节，甚至指出核心假设不成立。复核重点应是EMBER的核心可行性、
PI0.5内部应该怎样承载视频过程知识、Program到静态LoRA怎样形成闭环功能，以及现有数据是否足以识别该问题，而不是为现有
代码寻找一个最小补丁或一条单变量下一步。

## 2. 核心判断

### 2.1 视频理解与LoRA生成应是一条policy-native计算链

PI0.5原生VLM擅长把语言和当前图像变成静态语境；Action Expert在每一层、50个horizon位置上把该语境变成有时间组织的
动作预测。若EMBER只在VLM帧特征外另加一个普通temporal encoder，再把视频压成全局latent后生成LoRA，模型很容易在少量
tasks上用对象、端点、绝对时间或task identity完成监督，而没有借用source policy已有的动态控制结构。

候选路线因此让每个视频时刻的真实视觉变化尽早进入Action Expert的`layer x horizon`结构；之后所有时序拼接、Program形成
和LoRA生成都保留policy layer地址。视频理解不是独立分支的产物，LoRA也不是在最后才读取一个与policy拓扑无关的摘要。

### 2.2 “读取Action Expert”不改变PI0.5原始输入合同

对每个教学帧，仍运行PI0.5本来就有意义的native graph：真实图像与exact language进入原生prefix；Action Expert接收原生
flow inference中的`x_t`和`t`，而不是teacher action、伪动作标签、空prefix或自造action token。部署视频始终action-hidden。
当前实现使用固定的有效noise probe与`t=1`；新方案首先把它提升为可审计的canonical native probe，再读取所有Action
Expert层的hidden lattice。少量配对noise/time或完整denoised response只用于校准“这个probe是否代表policy动态”，不再
成为一个任意扩张的语义维度。

train/meta teacher actions可以监督视觉transition与Action horizon/phase之间的对应关系，但不作为Writer输入，不作为
最终预测，也不构成`video -> action -> LoRA`中间任务。真正的部署Value仍只能来自视频帧变化。

### 2.3 一套静态LoRA可以表达多阶段行为，但必须在多阶段状态上训练

Writer仍只生成一套静态LoRA。它不需要按event生成多套adapter：冻结policy在不同当前观测与内部activation下本来就走不同
计算路径，同一LoRA可以对抓取前、搬运中、放置后产生不同影响。问题在于compiler训练必须同时覆盖successful expert状态、
decoded-policy learner状态和多个过程阶段，确保这套静态参数在完整occupancy上形成连贯闭环行为；不能只拟合某个成功轨迹
上的少量离线response。

## 3. 历史证据怎样约束新架构

| 历史证据 | 已证明的正机制 | 不能重复的失败 | 对本方案的约束 |
| --- | --- | --- | --- |
| Action-Forecast v4 | 视频顺序能进入LoRA/action | absolute-time与action-phase shortcut，shuffle最好 | Action结构只能作内容读取坐标，不能让时间戳本身成为Value |
| V5/V6，最高143 | task-grounded Core、真实visual transition、Procedure和native rank16 compiler可形成强support | Procedure会在fusion/compiler中衰减，后续checkpoint换手 | 语言作query，真实transition作Value；过程结构必须一直保留到target layer |
| LPCP 143 | 同一次native context forward里的18层、50-token Action probes是可用carrier | effective BA改动极小，仍在旧support邻域换手 | 继承native layer/horizon读取，不把carrier本身当方法完成 |
| Dynamic-K与K4 | 多视频可显著降低same-task nuisance | 也会更稳定地得到错误task mean | 在完整Program层聚合并保留不确定性，不平均frame/raw feature/LoRA |
| GOMQ 151 | learned memory query带来真实`+16`闭环增益 | 连续更新到135/131，独立memory贡献并不高coherence | memory可作实现，但固定坐标和support retention必须另解 |
| task-local experts与validation oracle | rank16 task LoRA具有高policy ceiling；validation8为250/400 | task expert对同task全部videos恒定，不含process可识别性 | expert只能提供privileged功能几何/校准，不能成为video字典或部署route |
| phase/multi-success表示 | 完整成功轨迹、多个成功checkpoint、显式phase后held5几何5/5 | fixed decoder只保留约26%--31% direct successes | phase表示有信息，但compiler必须在闭环occupancy与更丰富功能坐标上训练 |
| shared prior | held5 source21到shared43 | task residual把43降到37/33 | 可以有稳定底座，但必须用shared-only反事实证明video residual新增support |
| outer credit | reward图、resume和train advantage接通 | macro2 41到macro3 39，未越过shared support | 过程推断未成立前不直接扩大outer estimator |
| macro2 matched panel | 当前完整部署链可正式评测 | correct/language/video/first+final/same=`41/39/40/39/40`，Goal/Long全0 | 现有16维推断没有证明完整视频、过程或跨video鲁棒性 |

上述证据不支持“回到某个历史版本”，也不支持把全部失败归因于24 tasks、LoRA参数多、rank、memory数量或训练时长。新方案
继承policy-native carrier、真实transition、per-video保序、bounded communication、Dynamic-K数据合同、strict evaluator与
全部昂贵expert/source资产；丢弃的是全局低维瓶颈、LoRA-rank语义、无界后置重写、task-ID字典和moving latent gauge。

## 4. 当前代码与最新思想的精确差距

当前successor不是本文架构的小型实现：

1. `src/ember/writer/video_program.py`的joint forward只取得bridge最终返回的`suffix_hidden`，形状是
   `[frame, 50, width]`；没有保留每个Action Expert layer，因此没有`frame x layer x horizon` lattice。
2. `src/ember/functional_adaptation/inference.py`先用少量phase queries读取50 tokens，随后
   `phases.mean(dim=1)`形成frame feature；horizon结构没有继续进入video program或decoder。
3. 同文件的per-video路径把时间序列汇总为first、final、event mean和transition mean，再压成一个summary；跨K也主要对
   summary做集合均值/小残差，不存在video time与Action horizon的显式拼接。
4. 当前输出是一个16维code，再由旧fixed decoder写rank4 task residual。macro2面板已经证明这个接口没有稳定的full-video
   增量。`configs/pi05_train24_functional_process_outer_credit_v2.json`补充的action alignment与时序controls只会训练现有
   末端heads；即使formal通过，也不能证明完整`time x layer x horizon`方案。

因此建议暂不启动该config的formal warm-start。现有smoke、checkpoint、teacher-action store、controls和evaluator应复用为
新架构组件与反事实；若保留旧formal，只能把它登记成“当前16维压缩下的晚期process supervision诊断”，不能用来裁决本文
核心假设。

## 5. 候选完整流水线

```text
exact language L + K action-hidden ordered videos V_k
    |
    |-- frozen native PaliGemma/VLM
    |      -> task-grounded visual-language tokens S[k,t,p]
    |
    `-- native PI0.5 Action Expert probe at every frame
           -> full hidden lattice H[k,t,l,h]
              video time t x policy layer l x action horizon h

adjacent-frame task-relevant visual transition DeltaS[k,t,p]
    + H[k,t,l,h]
    -> early policy-native fusion C[k,t,l,h]
    -> banded/causal dual-time transport across t and h
    -> limited local communication across neighboring layers
    -> ordered per-video Program P_k[event, layer, channel]

{P_1 ... P_K}
    -> permutation-invariant consensus at complete-Program level
    -> P[event, layer, channel] + confidence/disagreement

P + target address
    -> frozen layer-local Program-to-LoRA compiler
    -> one complete 38-target rank16 LoRA
    -> frozen source PI0.5 closed-loop rollout
```

关键不是tensor名字，而是四个不可丢失的因果轴：真实视频时间、Action horizon、Action Expert layer、最终LoRA target。允许
在确定语义后压缩frame数和channel宽度，但不能在Program形成前把这四轴全部折叠进一个全局16维向量。

## 6. 各模块的具体化

### 6.1 Frozen semantic state `S[k,t,p]`

- exact language和每帧RGB通过冻结PI0.5 PaliGemma原生prefix；不训练Text/VL Meta-LoRA；
- language决定object/relation/goal的query，视频patch提供静态before/after内容；
- 可读取少量固定VLM层或最终task-grounded patch tokens，但不另建一个能独立完成task adaptation的通用video backbone；
- static semantic path可产生learned language-only baseline与共享prior，但task-conditioned video correction的动态Value不能
  只由单帧appearance、position embedding或端点产生。

### 6.2 Full Action lattice `H[k,t,l,h]`

- 对每帧运行同一native prefix与native flow probe，捕获所有Action Expert block输出，而不是只取最终suffix；
- `l`保留policy depth，`h`保留50个action-horizon positions；LoRA rank不被解释为event、phase或layer；
- 默认只用一个预注册canonical probe进入主forward，避免把多个noise seeds变成新的高维shortcut；
- paired multi-time/noise、denoised action和JVP只用于train/meta校准、稳定性检查或privileged Program target，不进入held输入；
- Action Meta-LoRA若保留，应先作为所有train/meta tasks共享的observer calibration学习，再冻结后训练Program/compiler，避免
  observer与输出坐标共同旋转。它是否必须存在仍是开放问题。

### 6.3 真实视觉transition而非hidden相减

相邻帧先通过task-conditioned token matching/cross-attention建立对象对应，产生
`DeltaS[k,t,p] = TransitionMatcher(S_t, S_{t+1}, L)`。它表达哪个task-relevant对象/关系发生了什么变化，而不是简单的
全局向量差或DCT频率。

随后让Action lattice主动读取该变化：

```text
C[k,t,l,h] = CrossAttention(
    query = H[k,t,l,h],
    key   = [S[k,t], S[k,t+1], language query],
    value = DeltaS[k,t]
)
```

如此，source policy的layer/horizon状态决定“这段视觉变化对控制意味着什么”，但真正的dynamic Value只能来自帧间变化。
静止重复帧应让dynamic channel退化为零或低置信度；语言和单帧仍可提供prior，却不能伪造过程posterior。

### 6.4 Dual-time transport

视频时间`t`表示示范中事件怎样实际推进；Action horizon`h`表示source policy在当前帧下组织未来动作的方式。两者不能等同，
也不能独立平均。候选使用每层局部、因果、带状的soft transport，把相邻视频时刻的Action lattice拼成连续事件：

- 默认支持停留、前进、不同执行速度和null/no-change，不强迫固定`Delta t -> Delta h`斜率；
- first implementation可用monotone/banded alignment，但不能把recovery、重复接触或短暂回退硬裁掉；
- train/meta teacher action只校准视觉transition与horizon/phase的对应，不是部署输入；
- 同层为主，只允许`l-1,l,l+1`或其它预注册窄带跨层通信，防止历史unbounded M2P再次覆盖已经形成的Program；
- transport输出必须接受correct/reversed/shuffled、同endpoint异middle与速度变化反事实，不能只对absolute frame index敏感。

### 6.5 Layer-preserving Program `P[event, layer, channel]`

每条video压缩为少量有序event slots，但保留layer轴。每个event至少区分：

- task-relevant before/after relation与视觉change；
- event发生/完成条件、顺序和持续置信度；
- source policy在各层/horizon对该change的动态解释；
- source与目标行为之间可能需要的layer-local correction evidence；
- null/no-change与跨video不一致性。

event slots可以由monotone soft segmentation、memory tokens或其它实现得到；memory不是硬要求。输出端不使用会抹掉task/
event magnitude的全局RMS normalization，也不把所有layer通信交给一个无界Transformer。`P`应是视频理解和参数生成共用的
唯一中间对象，不再建一个“视频语义encoder”和一个互不约束的“LoRA hypernetwork latent”。

### 6.6 Dynamic-K aggregation

每条video先独立得到完整`P_k[event,layer,channel]`，跨video再做置换不变的robust consensus：

- mean可作稳定底座，但必须同时保留bounded set residual、跨video variance/confidence和event缺失；
- 不平均frames、raw VLM/Action features或最终LoRA，不选best video；
- correct same-task videos应强化共同event/layer证据，视角、路径、速度与抓取角度差异应进入不确定性；
- 若方法声称K1--K4或更大K，训练必须真实覆盖每个cardinality，并用nested无放回视频集合评测。

### 6.7 Layer-local Program-to-LoRA compiler

对每个LoRA target `j`，先定义它在Action Expert topology中的native owner `N(j)`。target读取
`P[:, N(j)-1:N(j)+1, :]`与target type/address，再生成完整A/B；action-in、各层q/v和action-out保持自己的明确地址。主干和
heads可在targets间共享，但Value主要来自同层/邻层Program，不允许一个全局decoder任意重写所有层。

LoRA rank16只是每个target更新的数值容量，不预先承载16个events、lanes或policy layers。可以探索fixed functional bases，
但不应立即把输出锁到少量expert atoms：历史fixed decoder曾在内部loss改善时只保留约26%--31%的direct successes。首版
更倾向高维layer-local生成头，在多个成功adapters、多阶段expert/learner states和closed-loop gate上学到足够宽的功能面；
通过后冻结compiler，再训练视频到Program，阻断共同旋转gauge。

允许一个task-independent fixed intercept/shared prior，但必须在rollout前与video correction合并为同一套LoRA，并始终报告
zero/dynamic-Value-ablated Program的shared-only反事实。最终资格来自video path创造新的task-conditioned successes，而不是
继承shared support。

## 7. Program与compiler怎样获得监督

这里是专家最新思想尚未完全具体化、也是本方案风险最高的部分。

### 7.1 候选privileged Program target `P*`

我们的暂定方案不是把raw expert A/B或单一action chunk当标签，而是只在train/meta tasks上，从以下证据共同形成
`P*_task[event,layer,channel]`：

1. 同task多个真正successful task-local LoRA checkpoints/adapters，而不只是同一失败轨迹或一个参数点；
2. 每个member的完整successful occupancy与显式event/phase alignment；
3. 在同一真实状态、language、noise/time上，source与expert的全层Action hidden、denoised action和必要时局部JVP差；
4. decoded-policy learner occupancy上再次查询privileged expert，覆盖adapter偏离成功轨迹后的恢复方向；
5. 只在meta-train拟合的layer-local投影/whitening，held meta task只用同一固定变换；
6. 多个功能等价successful members的consensus与variance，而不是raw A/B均值。

这样`P*`描述“在每个过程事件和policy layer上，成功任务相对source需要什么功能改变”，而不是某一套LoRA gauge。视频Writer
学习从action-hidden过程证据预测同一结构，compiler学习把它变成一套静态LoRA。

但这存在潜在循环：`P*`由successful experts和policy response定义，可能仍把“结果策略怎样行动”强加成视频本身未能识别的
latent；如果其坐标只能通过task-local actions定义，video-to-Program可能仍欠识别。我们希望专家重点判断：该target是否在
理论上合理，是否应改成基于过程关系的离散/连续hybrid Program，或应完全取消显式`P*`、直接以layer-local closed-loop credit
训练compiler。

### 7.2 建议训练顺序

1. **Observer/transport calibration。** 在授权train/meta tasks上，用真实相邻帧、teacher actions、速度扰动和时序controls
   学习transition matching与`t x h`对应；source/VLM冻结，Action Meta-LoRA若用则本阶段后冻结。
2. **Privileged Program与compiler realizability。** 只用train/meta successful experts、successful/learner occupancy与`P*`
   训练layer-local compiler；在leave-task-out tasks直接安装oracle Program生成的一套LoRA做closed loop。它必须保留direct
   success/gain并增加source support，才能进入视频推断。
3. **固定compiler，训练video-to-Program。** exact language形成prior/query，full video形成dynamic posterior；teacher
   actions只提供阶段/transport辅助监督，主target仍是跨episode`P*`与policy-functional response。
4. **Inference gate。** full video在固定held meta tasks上稳定优于learned language-only、first+final、static、wrong、
   shuffled、reversed，并在same-task-other与Dynamic-K上保持success rows。
5. **Closed-loop outer credit。** 只有上一步成立，才在train/meta simulator用success/progress进一步更新video-to-Program；
   应在`event x layer`结构上分配credit，而不是再次用一个全局16维随机方向。compiler、source与held边界保持冻结。
6. **Formal qualification。** single-checkpoint paired400、相邻checkpoint、multi-fold、all suites、Long、breadth与完整视频因果
   面板共同裁决。

## 8. 数据角色与跨embodiment边界

现有资产适合先回答机制：target train24从未参与source训练，可做19/5 role-disjoint folds；已有47条successful trajectories、
validation8 local oracle、71-task expert bank、双视角RGB、完整evaluator与controls都可直接复用。它们不足以同时支持最终
meta-generalization和human/other-robot video claim：

- 当前71个non-held LIBERO-90 tasks全部参加过source训练，用它们训练Writer会混入task identity与source competence；
- target train24只有24个独立mapping，适合首轮机制与leave-task-out门，但可能不足以识别通用video-to-Program规律；
- LIBERO只有同embodiment robot RGB demonstrations，没有人类或其它robot运动分布。

建议先用现有source与train24资产做小而决定性的observer/compiler gate，不重跑昂贵source训练。只有出现真实full-video过程
增量后，再一次性建立fresh、source-skill与adaptation-meta task role-disjoint的训练合同并重训必要source，而不是先付出大规模
成本。这个顺序需要专家判断：如果source/meta分离是架构可识别性的前提而非最终验证要求，则应更早承担该成本。

跨embodiment长期目标不能靠当前LIBERO结果外推。XSkill/UniSkill一类跨embodiment skill encoder或大规模无标注视频预训练可
作为`DeltaS`的frozen actor-invariant visual Value/初始化；其skill policy不应成为EMBER部署的第二策略。还需真实的人类/异构
robot视频数据检查actor、视角与速度nuisance。专家需要判断该表示应从架构第一版就纳入，还是在同embodiment机制通过后作为
独立扩展。

## 9. 现有资产的复用与生命周期

- 复用：frozen source policy、38-target rank16 LoRA合同、task expert banks、successful trajectories、action/JVP/flow工具、
  privileged action store、video cache、Dynamic-K sampler、process controls、paired evaluator、persistent workers与formal
  evidence schema；
- 改造：`video_program`增加per-layer Action capture与task-grounded transition Value；`functional_adaptation.inference`由
  frame/phase mean + 16D code改成dual-time Program；compiler改成layer-local固定输出面；
- 封存：当前16维process warm-start与单方向outer estimator继续是历史反事实，不作为新架构fallback；
- 退役触发：新架构通过oracle compiler闭环门和端到端smoke后，仓库只保留一个canonical Writer运行面；历史实现由Git、
  frozen configs与formal artifacts复现。

## 10. 最小但决定性的证据门

这不是要求专家把意见压成“单变量下一步”，而是防止实现再次在内部指标里无限扩张。建议的最早证据链是：

1. **Native lattice gate**：相同task的不同successful videos在`event x layer`结构上比cross-task更一致；correct与真实时序controls
   的差异不能由frame index、端点或静态appearance解释。
2. **Compiler oracle gate**：video inference尚未加入时，leave-task-out oracle Program生成的一套LoRA必须显著复现direct
   expert的闭环success/gains；不以hidden、BA cosine或functional loss过门。
3. **Video necessity gate**：fixed compiler后，full video相对learned language、first+final与static有新success，relative
   correct arms不是只让negative变坏；same-task-other retention至少达到项目既定90%--95%参考区间。
4. **Method gate**：达到有意义absolute且在suite、Long、breadth、相邻checkpoint与多fold共同成立；约145可构成强结果，
   150+仍需视频因果资格。

若第1门失败，问题在Action lattice/transport或数据可识别性；第2门失败，问题在Program target/compiler；第3门失败，问题在
action-hidden video到Program推断；只有前三门成立后，outer reward的成败才主要归因于shared credit。

## 11. 希望专家重点判断的开放问题

1. **`h`的语义锚定。** native Action horizon hidden在不同noise/time与当前图像下到底能否作为稳定过程坐标？canonical
   probe + teacher-action/denoised/JVP校准是否足够，还是必须在完整denoising trajectory上定义lattice？
2. **Dual-time transport。** video time与Action horizon之间采用monotone/banded transport是否符合PI0.5机制，还是会错误
   排除recovery、循环或非单调任务？更合适的约束是什么？
3. **全层lattice的必要性。** 保留全部Action Expert层是否真正提供layer-to-LoRA对应，还是会把欠识别放大成更大的latent？
   应读取block output、residual delta、attention/MLP response，还是少量policy-functional summaries？
4. **early fusion的位置。** task-grounded视觉transition应在每层Action hidden内作为Value被读取，还是只需在少数层注入后沿
   Action Expert传播？怎样避免破坏source policy原生计算或建立第二video model？
5. **Action Meta-LoRA。** 它应作为共享observer校准后冻结、与compiler共同训练，还是完全删除？怎样判断它在读policy-native
   dynamic evidence而不是制造新的task identity通道？
6. **Program target `P*`。** 多successful experts + full-layer source/expert response + successful/learner occupancy构成的
   privileged target是否科学有效，还是循环、过度policy-specific或仍受expert gauge污染？更合理的Program监督是什么？
7. **compiler形式。** layer-local高维生成头、fixed functional bases、slow/EMA decoder或其它形式中，哪一种最可能在固定
   坐标下保留direct support？怎样吸取历史fixed decoder只保留约26%--31% successes的失败？
8. **静态LoRA与event程序。** 一套rollout前生成的静态LoRA是否足以承载多阶段Program？需要怎样的multi-state/phase训练才能
   确保event-specific知识通过当前观测激活，而不是需要runtime conditioning？
9. **语言与视频的因果分工。** 如何让language确实提供query/prior、video transition确实提供必要Value，同时不靠人为关闭
   language或给correct arm特殊identity开关来自证？
10. **Dynamic-K。** 完整Program的robust consensus应怎样处理不同速度、缺失event、视角和合法procedure差异，既不平均掉
    关键信号，也不选择best video？
11. **数据可识别性。** 现有train24 19/5机制面是否值得先做，还是必须先fresh重分source/meta tasks才有资格判断架构？
    LIBERO是否本身太容易被端点/对象捷径解释，无法识别真正process compiler？
12. **跨embodiment。** XSkill/UniSkill类actor-invariant过程表示应成为第一版核心Value，还是后续独立验证？若从第一版加入，
    怎样保持EMBER的贡献仍是`process -> frozen PI0.5 LoRA`而不是另一个video-to-action policy？
13. **现有process warm-start。** 是否值得把已接通的16维formal只做一次有边界诊断，还是它与新核心假设差距太大，应直接停止？
14. **总体判断。** 更根本地说，专家是否认同“视频理解与LoRA生成共享`event x layer` Program，Action Expert的完整内部时空
    结构是两者之间的桥”这一核心判断？如果不认同，最根本的错误是什么，替代路线应怎样重新定义？

## 12. 当前执行方的默认选择

若专家没有指出根本问题，我们倾向于：

- 先保留一个native canonical Action probe，捕获全层50-token lattice；其它response只作校准；
- 使用task-matched visual transition作唯一dynamic Value，在Action lattice内early fuse；
- 使用可停留/跳过的局部因果transport和邻层通信，不使用全局无界M2P；
- Program保留`event x layer`，不再经过16维全局瓶颈；
- 先构造privileged `P*`并证明layer-local compiler oracle闭环可实现，再冻结compiler学习video-to-Program；
- 现有train24 19/5与昂贵专家资产先做机制门，出现process净增量后再做fresh role-disjoint source/meta重训；
- 当前16维formal warm-start不启动，除非专家认为它能回答一个不会与新方案混淆的关键问题。

这些是可供审查的工程默认值，不是owner给专家设置的限制。

## 13. 远程复核入口

建议先读：

1. `docs/current_owner_requirements.md`：稳定目标与信息墙；
2. 本文：新候选架构、推导和问题；
3. `docs/research_history.md`：所有主要架构与正式实验因果链；
4. `findings.md`：跨实验持久结论；
5. `progress.md`与`task_plan.md`：当前暂停点、资产和执行边界；
6. `docs/functional_adaptation_successor_design.md`：当前已实现但未通过Inference Gate的前序路线；
7. `src/ember/writer/video_program.py`、`src/ember/functional_adaptation/inference.py`与
   `configs/pi05_train24_functional_process_outer_credit_v2.json`：当前实现和新方案的代码级差距。

相关外部方法只作思想参照：SHINE（backbone内layer-aware memory到LoRA）、Doc-to-LoRA（variable context到固定参数）、
XSkill与UniSkill（跨embodiment skill/process representation）。EMBER的核心贡献若成立，应是把action-hidden过程知识在
frozen PI0.5内部编译成一次性完整Action Expert LoRA，而不是复现这些工作的输入输出。

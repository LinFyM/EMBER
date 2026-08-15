# EMBER Current Owner Requirements

状态：2026-08-13 owner当前需求authority。本文不是handoff、实验设计或启动授权；它记录当前session已经
明确的目标、方法要求、设计偏好、开放变量和协作方式。后续工作不能再用旧handoff、历史design或某次实验的
“当前/下一步”覆盖本文。若owner之后明确改变要求，以最新明确表达为准，并同步更新本文。

## 1. Authority层级

1. owner最新明确表达；
2. 本文对当前owner要求的综合记录；
3. `AGENTS.md`中的安全、信息墙、Git、GPU、存储与正式评测合同；
4. `docs/active_session_handoff.md`、`docs/execution_brief.md`、`docs/research_history.md`及旧design只提供
   已完成实验、资产位置和负结果边界。

旧文档中的“active”“current”“next”“等待交接”“暂停讨论”等只描述当时状态，不是当前命令。当前session
已经持续工作，不处于重新接管或向另一个session交接的阶段。除非owner明确要求，不做机械handoff，也不把
重复阅读或交接过程当作科研推进。

## 2. 最终目标与现实启发

EMBER的现实启发是：人看一条正确教学视频后，并不会逐像素、逐关节复制示范者的轨迹；他会结合任务名称，
理解示范在完成什么、关键动作阶段是什么、先后因果关系是什么，然后在自己的身体、视角和初始状态下尝试。
初次尝试应已经明显优于完全不会；后续若允许与环境交互，还可以在这个起点上快速学好。

对应的核心目标是：

```text
exact task language + one or more action-hidden correct teaching videos
    -> one shared Writer
    -> one task-conditioned policy adaptation produced before rollout
    -> frozen pi0.5-LIBERO source policy
    -> closed-loop task completion from unseen initializations
```

当前主要研究目标是让初次生成的adaptation本身立即有效。后续在该adaptation上做task-local RL，是独立的后续
实验，不能拿来掩盖初始Writer性能不足。

性能仍应尽可能超过`150/400`并继续提高，但owner于2026-08-15补充了更重要的科学资格：若同一shared method的
连续相邻checkpoints都能稳定保持约`145+`、成功集合低换手，且same-task不同视频鲁棒、correct相对wrong/
shuffled/reversed/no-video有明显特异性，那么即使尚未到151也可视为很有价值的成立结果。反之，单点151若来自
波动winner、同任务换video即掉或没有视频因果性，也不算成功。历史v6-fast143仍需正面超过或在稳定性/因果性上
实质改进，不能靠checkpoint union、挑task checkpoint或平均模型绕过。

## 3. 真正的核心科学问题

以下问题是目标的组成部分，不是可有可无的诊断榜单：

1. **高层任务知识**：从视频提取跨初始化、视角、路径、速度、抓取角度和偶然扰动仍成立的对象、关系、目标
   状态、必要阶段和有向顺序，而不是复刻单条demo的低层轨迹。
2. **语言与视频缺一不可**：language说明“要完成什么”，video说明“正确完成方式是什么”。language可以作为
   query/context/address，但不能独立生成任务LoRA；video不能被降格为无关装饰。
3. **正确顺序具有结构作用**：correct应因展示有效的初态到目标态过程而有用；shuffled破坏阶段连续性，
   reversed破坏有向因果。不能只靠人为把negative LoRA推坏制造margin。
4. **policy-effective写出**：视频程序必须通过compiler/adaptation真实改变policy的有效方向；LoRA norm、rank、
   cosine、reconstruction或functional loss只能解释，不能替代closed-loop。
5. **共同积累而非能力换手**：多个tasks和多个videos的能力必须在同一个Writer checkpoint中共存，降低相邻
   checkpoint churn，而不是optimizer轮流把能力从一个task搬到另一个task。
6. **视频因果有效性**：correct最终必须实质优于wrong、shuffled、reversed和no-video；same-task其它视频应有
   良好鲁棒性。只证明hidden、LoRA或action随视频变化不够。

## 4. 输入合同与one-shot/few-shot立场

### 4.1 基本输入

- exact task language必须保留；
- 所有teacher videos必须action-hidden，不读teacher action、proprio、reward、terminal、task ID、filename、
  object pose或hidden normalization；
- 每条视频内部按真实顺序处理，frame stride当前保持5；
- Writer在rollout前运行一次，policy执行期间不反复观看teacher video。

### 4.2 视频数量

owner希望方法从架构上认真支持一条或多条视频，最好可接受动态数量，而不是为K1和K4维护两套模型。多条视频
的价值在于过滤demo-specific nuisance、提取同task demonstrations共有的高层程序。

若声称支持动态K，训练时就必须实际覆盖动态cardinality，不能只训练K1/K4再宣称任意K。K的采样需要兼顾：

- 各K得到足够且均衡的训练曝光；
- task仍等权；
- 各GPU按真实帧数和计算量平衡负载，而非纯随机造成长尾。

owner不要求为了论文形式强行做“同帧数、同FLOPs的K1对K4公平竞赛”。如果one-shot最好，就报告one-shot；
如果K4/K5/K8最好，就诚实报告few-shot；如果动态数量越多越好，可以把scaling behavior作为卖点。必要的对照是
确保结论真实，而不是让较强方案人为降配。

多视频方法必须：

- 每条video内部保序；
- video之间采用置换不变集合处理；
- 提取共同高层知识，同时保留各自的时序证据；
- 不简单平均frames、features或最终LoRAs；
- 不挑“最好视频”；
- 不把多个video分别生成的LoRA做平均或checkpoint融合。

可以研究“多视频共同表示指导单视频表示”的一致性训练；它必须是同一Writer中的表征学习，而不是引入额外
部署teacher、task ID或第二套LoRA。

## 5. 输出与adaptation形式

当前主方向仍是Writer一次生成一套完整task-conditioned LoRA，并加到冻结source policy上。LoRA之所以重要，
是它提供可保存、可部署、可在后续独立RL实验中继续优化的任务起点。

但必须区分：

- “生成policy-effective task adaptation”是核心要求；
- “一定是rank-16 LoRA”不是长期goal；
- memory数量、LoRA rank、factorization和decoder形式都是架构变量。

owner明确提出：完整LoRA即使是adapter仍然高维，当前Writer可能因此难以生成，能量集中和rank坐标相似也可能
与输出方式有关。下一版设计必须认真分析降低rank（例如rank-8）的方案，不能未经论证自动保留rank-16。
同时也不能因为rank更小或几何更漂亮就假定更好；历史rank14 uniform compression损伤support，只否定那种
具体compression/regeneration合同，不否定fresh rank-8 Writer。rank选择必须结合从零训练与closed-loop裁决。

输出架构应具有规模可扩展性：基础policy更大、LoRA targets或参数更多时，Writer可通过共享层对应解码、参数
复用或结构化生成扩展，而不是为每个target增加一个彼此独立的巨大wide head。

## 6. 昨晚Dynamic-K架构中memory token的准确含义

memory token是owner对昨晚Dynamic-K架构的重要设计要求/启发，但不是EMBER的最终goal。不能再偷换概念。当时
Dynamic-K Writer已经把这一要求落实为8个真实memory tokens：它们与每帧真实图像、exact language和50个固定
Action probes共同进入π0.5 joint backbone，并保留其经过18个Action Expert层后的逐层状态。这个具体实现正在
接受formal closed-loop检验；若失败，应定位最早失效接口，而不是把“使用memory token”本身写成不可修改目标。

### 6.1 什么才算memory token

真正的memory token应当：

- 与真实task language和真实video evidence处于有意义的backbone上下文中；
- 在backbone层内参与attention/状态更新，读取并整合输入信息；
- 与policy层、target group或LoRA生成位置有清楚、可解释的对应；
- 最后由共享、可扩展的mapper/readout生成相应LoRA参数。

以下不应冒充memory token：

- 视频全部处理完以后才产生的普通latent slots；
- 为了分块输出LoRA而建立的几千个parameter slots；
- 没有真实图像/语言上下文、只把memory塞进Action Expert的空输入；
- 任意固定数量的“phase tokens”，但没有解释它们与实际stride-5帧序列的关系。

当前`20x8x256`后处理状态只能叫policy-group/rank-aligned Program：20对应action-in、18个Action Expert层和
action-out，8对应LoRA rank coordinates。它们不是额外memory tokens；真正进入backbone的memory始终只有8个。

### 6.2 memory放在哪里

设计必须先尊重pi0.5的真实计算语义：Gemma/VLM负责视觉语言上下文，Action Expert根据有意义的视觉语言prefix、
state/noise/time等预测动作。若使用Action Expert内部memory，它必须和正常有效prefix一起运行；不能用blank/
zero image、无prefix、memory-only或凭空构造的action query来“为了使用Action Expert而使用Action Expert”。

当前实现选择让memory位于Action Expert suffix，但它不是无prefix空跑：同一次joint forward的prefix包含真实
image tokens与exact language。视频时间轴只在逐video causal encoder中交互，K轴只在set aggregator中交互，
policy-group/rank轴只在M2P中交互；不能因为policy有20个层级，就在每个阶段把所有维度混在一起attention。

memory token数量应少而有依据。此前随意提出约70个tokens没有依据；当前8个tokens与fresh rank-8 LoRA坐标
对齐，并已通过最长视频吞吐与显存profile。这个对应关系是当前可证伪假设，不是“rank永远必须等于token数”的
普遍定律。

### 6.3 与视频处理的关系

memory不能替代视频理解。设计仍需清楚区分：

1. 逐帧/逐视频如何用语言关注任务相关内容；
2. 如何保留单视频内部有向过程；
3. 多视频时如何提取共同程序；
4. memory如何把这些信息带入backbone层级计算；
5. 如何由memory生成对应LoRA。

不要求在时间、video、policy layer和LoRA parameter四个轴上到处做attention。应选择最少但足以表达必要关系的
交互，并用完整数据流解释。

## 7. 昨晚完整架构判断、成熟Hypernetwork参照与历史继承

昨晚讨论形成、随后已经实现并完成裁决的完整数据流是：

```text
exact language + K=1..4 same-task action-hidden ordered videos
    -> 每帧真实image/language/Action-probe joint backbone + 8 memory tokens
    -> 每video有向transition D、terminal goal residual G和Query-only semantic address
    -> causal temporal encoder（video内部保序）
    -> permutation-invariant set attention（videos之间提取共同程序）
    -> policy-group/rank M2P
    -> shared projector + shape-family readout
    -> one complete 38-target rank-8 task LoRA
```

当前架构同时吸收三类依据：

1. owner的现实启发与需求：语言说明任务，正确视频展示完成方式；模型提取跨初始化成立的高层程序，而不是复制
   teacher低层轨迹；同一Writer支持动态视频数量并一次生成task adaptation；
2. SHINE、Doc2LoRA等成熟Hypernetwork研究：少量memory进入原生backbone上下文，保留layer-aligned状态，再用
   共享、结构化mapper生成LoRA；这里学习其第一性原理与可扩展性，不照搬文本模型输入、token数或flat payload；
3. EMBER历史证据：保留v5/v6的Semantic Core与有向Procedure思想、K4的逐video保序/跨video集合边界，以及
   policy-effective完整LoRA；不继承已被否定的language-only bypass、简单平均、高rank/正交目标或独立wide heads。

引用成熟工作只影响架构设计，不引入额外target-task训练数据。当前Writer仍只使用封存train24视频与action
functional supervision，因此不会因为参考外部论文而改变与target-task LoRA基线的数据公平边界。

- 设计必须准确描述pi0.5中的Gemma/VLM和Action Expert，不能用未解释的“Q层”“QV action”等占位说法。
  38个LoRA targets的真实归属、shape和层对应必须从代码/合同给出。
- Action Meta-LoRA目前有保留价值。VLM Meta-LoRA是否需要保留是开放变量：如果VLM本身已是通用视觉语言
  backbone且部署不更新它，可以先不加；但应由兼容性与实验决定。
- owner没有要求完整保留v5.2/v6前端。应继承它们被证实有效的机制，而不是为了“继承历史”机械复制一个可能与
  memory架构不兼容的前端。
- 同样，不能因为owner提到SHINE/Doc2LoRA就一比一照搬。需要研究这些工作为什么能从原生backbone上下文中的
  memory生成LoRA、用了多少tokens、怎样做层对应和规模扩展，再选择适用于视频/policy的部分。
- 必须对比现有FactorHeads、wide head和SHINE式共享mapper的作用与瓶颈，不能默认历史实现已经最好，也不能
  因为新方法更“漂亮”就判定它有效。当前Direct-Family-B正是根据逐接口probe，只删除已定位造成common-direction
  增长的family hidden/GELU；它保留此前已认可的整个输入、memory、temporal、set与M2P链，而不是另起炉灶。

## 8. 训练要求与RL边界

### 8.1 监督训练

- video与action episode可以同task但跨episode错开；这是阻断低层逐帧复制、要求跨初始化泛化的关键训练方式；
- 同task恒定target也产生不可识别性，Writer可能只学task identity。因此不能只依赖“输入正确顺序”就声称理解
  视频；架构和controls必须让视频动态过程不可绕过；
- task-complete训练必须保持task等权并分析per-task gradient/成功集合/换手；
- 最终方法需要一套从零开始可复现的训练recipe。旧checkpoint可以用于机制诊断或开发，但不能成为论文方法
  只能工作的隐含前提；
- 不额外引入会破坏与target-task训练公平性的外部target数据。source policy已有预训练/源任务能力是所有方法
  共享基础，不等于给新Writer额外开数据口子。

### 8.2 Writer RL

如果监督functional objective继续与closed-loop错位，可以研究在AS cold start后用train24环境reward调整Writer。
这不是失败时随意换榜，而是针对credit alignment的候选训练阶段。它仍必须保持信息墙、task balance和single
checkpoint，并由strict closed-loop裁决。

### 8.3 生成LoRA后的task-local RL

“先看视频生成一个好LoRA，再在这个LoRA上与环境交互快速学好”是EMBER长期故事的重要后半段，但当前不是
`>150`初始性能实验的一部分。先证明Writer生成的LoRA在零交互时就是强起点；随后另做task-local RL sample
efficiency实验，不能把两阶段成绩混成一个初始Writer分数。

## 9. 历史实验应怎样被使用

过去的大量实验不是互不相关的版本库，也不是必须恢复的执行列表。每次新设计要明确：继承哪条有效机制、针对
哪个最早失效接口、哪个证据会快速否决。局部建议不能导致整套方案下一句话就180度翻转。

必须保留的连续认识包括：

- v5/v5.2证明语义与过程分离、视频特异性有价值，但正确时序可能在fusion/compiler后衰减；
- v6-fast `143`证明某套architecture x task-complete recipe能产生强absolute，但仍存在后期漂移；
- 更漂亮的内部时序、去DC、高rank、正交、更多atoms/lanes/experts并不自动改善closed-loop；
- SFB union远高于single checkpoint直接证明task能力换手；
- variance reduction、reconstruction和functional evidence都可能与closed-loop错位；
- K4说明多视频可改善部分稳定性，但旧实现不代表few-shot本身无效；
- experts证明task-local LoRA可policy-effective，但不能提供same-task视频差异、正确顺序或held support；
- rank14只否定具体uniform compression/regeneration，不否定所有低rank或fresh rank-8设计；
- 最新credit/guard路线把key、condition、Program-to-action链路做通后仍在closed-loop换手，说明不能只继续美化
  内部surrogate或叠加point guards。

旧版本失败只淘汰实际测试的组合。没有架构级证据，不得因一次负结果放弃其中未被否定的子机制；也不得恢复
完全相同的退役架构换名字重跑。

## 10. 实验裁决与效率要求

- 真实closed-loop absolute优先；LoRA健康度、视频特异性、时序margin、loss、rank、norm和cosine只作诊断；
- 正式选择只认single-checkpoint strict paired400；报告per-task/per-suite、breadth、retained/gained/lost、
  churn和相邻checkpoint能力集合；
- 及时评测，不让长时间surrogate训练替代真实性能；
- 一次尽量改变一个主要因果变量，不靠大量rank/scale/seed/dtype/temperature小扫救失败checkpoint；
- 如果新架构低于历史强方法，必须定位以前的优势在哪个接口丢失。

GPU/工程要求：

- GPU上限6张，不要求6张；有多少真正合适的同节点卡就用多少，不等待凑卡；
- 少量显存占用或低利用率进程不自动排除设备，只要峰值余量足够且不会明显干扰他人；
- 若空闲卡不足，owner已与`ycliu`沟通并授权在显存峰值余量充足时与其进程共驻；仍按实时util/显存选择且不
  pause、kill、reset或明显干扰，授权不自动扩展到其他用户；
- launch前同时live检查gpu01/gpu02，选择一个节点，不跨节点拼碎片；
- 训练按K、视频帧数和真实任务wall做负载均衡；evaluator用动态队列和persistent workers；
- 吞吐优先，接受正常BF16/TF32、batch和kernel低位差异；
- 不为防御性安心增加重复forward、batch1、无意义zero/no-video baseline、逐tensor扫描、hash、大量校验、
  dtype扩展或host/device小tensor往返；
- 必要的信息墙、shape、finite、pairing、OOM、asset、checkpoint和resume检查保持，但不搭建与科学结论无关的
  防御性体系。

## 11. 协作与表达要求

- owner主要使用语音输入。对明显同音词、术语识别和断句错误，要结合EMBER上下文主动纠正，不机械执行错词；
- owner提出想法是给研究判断提供启发，不代表要求盲从。必须独立判断；不能owner说一个局部问题，就把此前
  已认可的整套设计全部推翻；
- 修改方案时保留已认可部分，只针对真正被质疑的接口修改，并明确变化原因；
- 任何新架构先用通俗、完整、前后一致的数据流说明：输入是什么、每阶段处理什么、memory在哪里、怎样聚合
  videos、怎样生成LoRA、怎样训练、为什么正确顺序必要；
- 清楚区分最终goal、核心科学问题、当前方法合同、设计建议和实现细节；
- 不用大量半成品术语、临时缩写或未定义token数量让owner反向猜设计；
- 不播报“读到EOF”“正在交接”“正在检查第几份文档”等机械过程。只汇报会影响科研判断的证据、完成的实现、
  实验进展、结果和真实阻塞；
- 已授权自主持续推进；owner于2026-08-13最新要求暂时不使用subagents，后续实现、训练、评测和分析均由当前
  主任务完成，直到owner再次明确改变；
- 一个完整实验结束后，先完成全部逐task、因果和接口分析，再自主进入下一轮有因果依据的设计/实现/实验；
  不拿半分析结果打扰owner。只有owner当时明确要求暂停、出现真正权限阻塞或需要改变核心目标时才停下讨论。

## 12. 方法、原则和目标的边界

当前长期goal保持第2--3节，不把memory token、rank-8、K值、LoRA decoder或某个optimizer写成项目goal。
owner在2026-08-14进一步澄清：memory token是为“怎样让视频知识按policy层级进入Writer、怎样可扩展地生成
合理LoRA”提出的候选机制，不是必须保留的形式；如果证据表明沿V6更接近突破，可以继续V6。此前未经说明便从
Full-Factor切到V6/rank-16仍是错误的协作行为，但错误在于静默改变方法合同，不在于V6本身被禁止。后续每次
选择必须比较最早失效接口、已有absolute与新增假设，不能因owner一句局部意见机械地全盘切换。

Dynamic-K Direct-Family-B的K1/K4=`102/98`证明set能把same-task effective-BA相对方差约降`6.3x`，但没有修正
task mean；Visual-Value曲线`88/86/86/96`证明task-grounded视觉Value进入了LoRA，却未对齐held occupancy；
Full-Factor=`91`进一步暴露了更具体的最早断点：独立生成A/B后A norm增至`1.376x`、B缩至`.062x`，effective BA
仅`.245x`且与fixed-A近正交。逐样本最优rank-8仍保留约`.999999`强BA能量，所以该结果否定的是当前独立A/B
factor credit/gauge allocation，不是否定rank-8容量、memory位置、动态K、逐video有序编码或跨video集合原则。

当前证据优先选择V6 Actual-Delta Success-Support Projection：V6已经证明LoRA具备policy-effective几何，最新
reward一步又真实获得18条held success，但同时丢19条；因此当前最早接口是task汇合后对已有support的覆盖，而非
LoRA健康度。该轮只在同一raw on-policy reward AdamW candidate之后，用train24成功executed-prefix的一阶loss
约束最终actual Writer parameter delta。若这一精确support实验仍不能改善retention/absolute，则不再继续V6
constraint小修；下一架构候选是保留V6的absolute Core、有向Procedure与健康factor compiler，再把真实
layer-aligned memory作为视频到policy slot的接口，而不是原样恢复91分Full-Factor或把memory本身当答案。

该裁决现已完成：projection把6条raw violation降到0，strict仍为`138/400`且相对AS139 lost23/gained22，故
constraint方向终止。结合owner进一步澄清，active架构不先强加literal memory token，而先冻结AS139强路径，从
同一次真实image/language/50 Action-probe forward旁读18层native probe states，经shared rank-query与video内
causal delta形成zero-init、layer/rank-aligned Procedure-query conditioner。这样先检验分层读取本身，且不同时
改变rank16、factor heads或B20；只有native probes在carrier层缺少正确顺序/material差异时，下一轮才在相同下游
接口把carrier单独换成真实memory tokens。精确authority见
`docs/action_forecast_writer_v6_layerwise_probe_conditioned_procedure_design.md`。

该方案已按上述边界完成canonical实现、world6 fresh macro0->25和strict400。carrier/效率合同通过：没有第二次
backbone forward，每video一次shared causal controller，倒序使query-delta/Program material变化，常量视频近零；
K4 generation锁B32。真实性能为`143/400`、breadth7、per-task=`1/4/48/35/0/38/16/1`；相对AS139严格=
`120 retained / 23 gained / 19 lost`、churn42。它追平历史143但没有超过150，并触发`<144`与lost>10门，故
不resume50或补controls。

全400 effective-BA只相对AS139移动`.002653`，LoRA健康结构不变；first4 same-task correction coherence median
`.56804`，Goal3高coherence仍0，Long1小改写却净丢6。因此本轮不支持“native probe carrier失败，立刻换literal
memory”的分支；更早缺口是conditioned Procedure到冻结fusion/compiler的policy commitment，以及blind B20
functional credit对held occupancy的方向选择。memory token仍可成为以后可扩展LoRA生成的一部分，但若只替换已
通过的carrier并保留同一Query/credit，不是在检验当前证据指向的问题。

owner随后授权继续，并再次说明并非要求memory token必须成为下一架构：若沿V6能更直接找到突破口，可以继续；
memory的价值应由它是否解决真实LoRA生成接口来裁决。当前受控后继因此保留V6-LPCP完整部署图，利用AS139与
LPCP在严格同schedule下`120 both / 23 LPCP-only / 19 AS139-only`、union=`162`的事实，新增一次train24
paired causal success distillation：同初态对跑zero-query AS139 reference和当前LPCP candidate，只用两臂中唯一
成功的policy-generated trajectory校准最后65,536参数`query_delta`，不反向推坏失败轨迹、不选择checkpoint或
部署第二套LoRA。精确authority=
`docs/action_forecast_writer_v6_lpcp_paired_causal_success_distillation_design.md`。这仍是初始Writer的共享训练阶段，
不是生成LoRA后的task-local RL。

PCSD已完成并终局为`135/400`、breadth6；相对LPCP143为`121 retained / 14 gained / 22 lost`。它证明paired
success可产生连续LoRA/action credit，但FP64分析显示同task四个不同K4 conditions的更新pairwise cosine约
`-.00187`、均值只保留约四分之一能量。后继CV-CSD随后按约定把同一真实成功trajectory的exact functional credit
覆盖到四个互不重叠same-task correct K4 conditions，并只在共享query gradient处等权汇合；部署架构、rank16、
optimizer与rollout数量均不变。

CV-CSD full24机制与吞吐合同通过，但strict只有`134/400`、breadth7；相对LPCP143严格=
`122 retained / 12 gained / 21 lost`，四个suite全部下降。FP64中四correct K4 conditions的部署增量pairwise
cosine=`.000205`、mean/sample energy=`.250155`，相对PCSD也仍约`0/.25`。因此“正确cross-video objective
本身足以让现有query-only map形成一致policy-effective commitment”已被否定，按门不续cycle2或controls。
这不是放弃memory、few-shot或LoRA生成；它把memory的合理用途进一步收窄为**在真实图文context与实际policy
layer/rank/target topology之间建立可训练的commitment**，而不是替换已通过的视频carrier或增加静态token容量。
精确终局见`docs/action_forecast_writer_v6_lpcp_cross_video_causal_success_distillation_design.md`。

后续迭代遵循以下边界：

- 若结果失败，先按`input evidence -> per-video Program -> set -> M2P -> LoRA mapper -> effective BA -> action ->
  closed loop`定位最早失效接口；
- 已经被机制证据支持、且本轮没有被检验否定的上游不能因一个aggregate低分被整体推翻；
- owner的局部建议只修改对应局部，不能触发无证据的整套180度重写；
- 若证据最终否定当前memory位置、rank或LoRA形式，可以改方法，但必须保留“语言与正确视频共同提供高层任务
  知识、一次生成policy-effective adaptation、单checkpoint多任务共存”的核心问题；
- 后续task-local RL仍是初始Writer达成强zero-interaction起点之后的独立实验，不得提前混入当前分数。

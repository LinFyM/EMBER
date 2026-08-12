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

成功标准保持：同一shared method、同一single checkpoint的strict paired correct严格`>150/400`，并继续提高。
历史最好v6-fast `143/400`是必须正面超过的起点，不是可以靠checkpoint union、挑task checkpoint或平均模型
绕过的门。

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

## 6. 下一版架构中memory token的准确含义

memory token是owner对下一版架构的重要设计要求/启发，但不是EMBER的最终goal。不能再偷换概念。

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

此前草稿中的`20x256`后处理latent最多只能叫layer-aligned program slots，`5056`个位置只是LoRA参数分块；二者
都不是owner所说的SHINE式memory token。

### 6.2 memory放在哪里

设计必须先尊重pi0.5的真实计算语义：Gemma/VLM负责视觉语言上下文，Action Expert根据有意义的视觉语言prefix、
state/noise/time等预测动作。若使用Action Expert内部memory，它必须和正常有效prefix一起运行；不能用blank/
zero image、无prefix、memory-only或凭空构造的action query来“为了使用Action Expert而使用Action Expert”。

memory究竟进入VLM、Action Expert还是跨两者交互，需要由真实接口和科学作用决定。不能因为policy有20个层级，
就默认每个维度都做纵向、横向、视频维和时间维attention。每一种交互必须回答：它在哪个阶段发生、交换什么
信息、为何该阶段需要它。

memory token数量应少而有依据。此前随意提出约70个tokens没有依据；下一设计必须参照真实policy topology、
SHINE/Doc2LoRA等方法的机制和实际显存/吞吐，说明为什么需要这个数量。

### 6.3 与视频处理的关系

memory不能替代视频理解。设计仍需清楚区分：

1. 逐帧/逐视频如何用语言关注任务相关内容；
2. 如何保留单视频内部有向过程；
3. 多视频时如何提取共同程序；
4. memory如何把这些信息带入backbone层级计算；
5. 如何由memory生成对应LoRA。

不要求在时间、video、policy layer和LoRA parameter四个轴上到处做attention。应选择最少但足以表达必要关系的
交互，并用完整数据流解释。

## 7. policy结构、Meta-LoRA与旧架构继承

- 设计必须准确描述pi0.5中的Gemma/VLM和Action Expert，不能用未解释的“Q层”“QV action”等占位说法。
  38个LoRA targets的真实归属、shape和层对应必须从代码/合同给出。
- Action Meta-LoRA目前有保留价值。VLM Meta-LoRA是否需要保留是开放变量：如果VLM本身已是通用视觉语言
  backbone且部署不更新它，可以先不加；但应由兼容性与实验决定。
- owner没有要求完整保留v5.2/v6前端。应继承它们被证实有效的机制，而不是为了“继承历史”机械复制一个可能与
  memory架构不兼容的前端。
- 同样，不能因为owner提到SHINE/Doc2LoRA就一比一照搬。需要研究这些工作为什么能从原生backbone上下文中的
  memory生成LoRA、用了多少tokens、怎样做层对应和规模扩展，再选择适用于视频/policy的部分。
- 必须对比现有FactorHeads、wide head和SHINE式共享mapper的作用与瓶颈，不能默认历史实现已经最好，也不能
  因为新方法更“漂亮”就判定它有效。

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
- 已授权自主推进与使用subagents加速。主进程负责统一科研判断，不能把多个agent意见拼成摇摆方案；
- 一个完整实验结束后，先完成全部逐task、因果和接口分析，再自主进入下一轮有因果依据的设计/实现/实验；
  不拿半分析结果打扰owner。只有owner当时明确要求暂停、出现真正权限阻塞或需要改变核心目标时才停下讨论。

## 12. 当前纠偏与下一步边界

此前未提交的`action_forecast_writer_dynamic_k_program_rank_design.md`不是当前architecture authority。它把
post-encoder latent slots和LoRA parameter chunks误称为memory tokens，并在未充分回应owner要求时固定rank-16；
这些具体接口不能进入实现或实验。但该轮已经对齐的总体数据流没有被owner否定：dynamic-K action-hidden
videos逐条保序编码、提取跨video共同高层程序、与exact language共同条件化、一次生成一套完整task LoRA，
以及由shared layer/rank结构扩展Writer。后继只能修正被指出的memory位置、token含义、rank和decoder接口，
不能把整套已认可设计推翻后另起炉灶。

当前长期goal保持第2--3节，不把memory token、rank-8、K值或某个optimizer写成项目goal。下一版架构判断必须
同时使用本文要求与历史证据，先决定最早科学失效接口，再选择真正的memory机制、rank、video cardinality和
training objective。任何选择都要解释为什么能改善高层视频知识、正确顺序、policy-effective写出和multi-task
共同积累，而不是因为某个组件被提到就机械加入。

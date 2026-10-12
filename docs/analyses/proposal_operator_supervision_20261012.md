# 从因子拟合改为实际线性算子监督：G864后的有限接续

2026-10-12，main科学决定；Owner已授权在硬边界内自主推进。
旧§13 G864/query16整批已经完成。本决定登记新科学批，不自动延长旧因子MSE合同。
canonical代码/Git/运行仍由实验session独占；请将完整推导与合同并入现有analyses、唯一active design及状态记录。

## 1. 完整结果和这次新增的机制判断

原ROOT：`/data1/user/ymdai/ember_runs/video_guided_proposal_writer_20261012/deterministic_endpoint_learning864_query16`。
同576配对MT340/G216341/G864341/E455，G864相对MT R309/G32/L31，相对216 R313/G28/L28。
train24为169/167/168/272，support12为171/174/173/183。E-only134行只传递17，both-success321保持296。
保留task36的1→5和task29的0→1，也保留Spatial净−4等损失；无明显超MT控制，不是大范围遗忘。
学习时长使条件参数拟合增强，不能继续把864描述为几乎恒定输出。MSE .961878→.843335，
公共占比97.677%→69.057%，条件变化能量 .000659→.040073；这仍不证明L/V/H各自的作用。

main直接读完整报告、已保存算术脚本/JSON、实际model/native/runtime/path/learning，以及最近似历史。
为辨认参数改善怎样进入执行，main只对这36套已经保存的G/E/P0作了一次CPU A/B及BA分解：
`/data1/user/ymdai/ember_runs/video_guided_proposal_writer_20261012/deterministic_endpoint_learning864_query16/analysis/main_scientific_readout/endpoint864_factor_effects.py`及同名json。
实际算术2.232秒，无GPU、policy forward、episode、hybrid/mean/scaled adapter部署。
两份原件已归档到上述稳定目录，保留原内容；没有重复运行。

| 同一S_G下的量 | A坐标 | B坐标 |
|---|---:|---:|
| 坐标数 | 303360 | 340224 |
| MT对目标MSE | 1.000000 | 1.000000 |
| G864对目标MSE | .667975 | .999695 |
| G输出编辑均方 | .273228 | .001357 |
| G条件变化均方 | .084959 | .0000494 |
| 在27个非零E任务内部优于自身常量的条件MSE增益 | .058214 | .0000235 |

按原全部坐标权重，总参数改善 .156665中的 .156504来自A、仅 .000161来自B。
条件改善 .054377中的 .054339来自A、仅 .0000389来自B。不能再用总loss代表两个因子都学到了可执行编辑。
真实W更新为B@A（当前alpha/r=1）；原P0为A0非零/B0零。
跨36/38全部真实矩阵Frobenius能量聚合：G/E更新范数比 .021085、方向余弦 .020679、
目标相对平方误差 .999573。G权重更新几乎没有拟合E，与首5动作距MT .003722而E距MT .088126一致；
这里的一致是跨接口证据，不以矩阵范数替代hidden方向/后续动作/成功的因果分析。
所有这些统计只读已有权重，未选择新checkpoint或候选。

因此更准确的最早不足是：当前因子监督大部分学到了A的变化，而B及A/B共同的实际写入仍弱。
这不是工程bug证明，不是B梯度被硬阻断的证明，也没有唯一证明decoder容量、坐标规范或视频信息不足。
普通“再加时长”已按原合同结束；π选择能力也不能替代G产生有效候选。

## 2. 为什么选择监督BA，且不立刻叠加其它方案

当前逐因子平方误差可以在B基本为零时通过拟合A持续下降，但B=0时任意A都对应同一个MT执行。
这是实际目标与部署算子的可分离性；本次证据表明它确实影响了怎样解释学习进展。
改用部署所加的完整线性算子作为标签，使A和B必须形成配合，才能降低训练误差。

对target层l，记：

    W_G,l = s_l B_G,l A_G,l
    W_E,l = s_l B_E,l A_E,l
    D_l = (1/36) sum_t ||W_E,t,l||_F^2
    L_operator = (1/38) sum_l ||W_G,l-W_E,l||_F^2 / D_l

s_l为实际alpha/r，本批仍1。D_l一次由原36个合法E等权计算并冻结；每层包含所有矩阵元素。
这是层内按教师实际算子能量定单位、层间等权，不是逐样本归一化，也不按教师成功率或任务类型重权。
九个E=MT的标签是零矩阵，仍按完整36任务等权训练；D_l由全体任务决定，不除以单task零目标。
main现有CPU原件表明全部38个D_l正且有限，范围 .00229936至1.10991009；无需新floor或尺度扫描。
按这个事前固定口径，MT平均loss=1，G864当前loss=.999513；不与旧因子raw loss跨目标比较。

若B_G=0，此loss对任意A_G均不变；其实际导数为：

    dL/dB_G,l = 2 s_l (W_G,l-W_E,l) A_G,l^T / (38 D_l)
    dL/dA_G,l = 2 s_l B_G,l^T (W_G,l-W_E,l) / (38 D_l)

因此只有A改善不再构成loss进步；已非零的A0给B提供起始信用，B学起后A/B可共同改变。
同一实际W的等价A/B分解给相同标签，避免强求某套教师因子坐标本身；但本批未唯一诊断gauge是根因，
也不宣称这会让条件映射自然可学。若G等于E的全部38个实际更新，同一MT基底和官方原生policy会具有同一功能；
有限矩阵误差并不保证闭环相近，这仍须实际控制裁决。

梯度经实际A/B、原S_G输出变换、六层decoder、Reader和双meta回到ψ。没有额外学习率乘子、手工放大B、
冻结A、缩rank、换seed、辅助动作头、参数MSE/cosine/logscale混合项或新decoder。
为减少无意义大矩阵物化，可按现有原因子用r×r Gram恒等式精确计算Frobenius损失；
正常FP32计算loss/Gram，原模型正常BF16/数值政策不变。必须保持上述函数及其对完整A/B的梯度，
不能把一侧detach、分别回归矩阵摘要或靠clamp/替代损失改变实际目标。实现细节由实验选择最小正确消费者。

这是线性权重算子监督，不称为原生十步动作蒸馏或闭环RL。它不需要新增动作query、教师occupancy或环境数据，
保留已有合法共同E监督源。相比立即加入native FM/十步动作loss，本次直接检验已定位的A-only进步与实际BA缺口，
避免同时改变query分布、动作信用、计算路径和参数生成器。若无效，也不能自动把这些其它loss排成无限修补菜单。

## 3. 继承的历史及限制

已核对`3a6f801:docs/action_forecast_writer_video_expert_manifold_design.md`§24–29、findings§63，
`docs/analyses/flow_supervision_history_20260926.md`全文、functional_revision_learning_20261010全文，
architecture_rethink_discussion_20261010的§9.3–9.9相关完整论证。
旧raw因子MSE+cos/logscale由48→75仍公共收缩；旧版本的有效norm并不小但方向差，不能称所有历史都只是B未学。
SEOD/GOMQ真实十步端点蒸馏有局部正例和相邻退化，phase/learner-state函数蒸馏也未充分保留direct expert能力。
最近functional revision实际使用−P J^T q，其约1e−8写入和极弱Reader信用与本次直接参数decoder不同，
但不能因计算不同就忘记“功能目标/真实梯度不自动带来控制保持”的负证据。
本次理由是当前同一模型已有具体A/B分解和BA缺口，不是“换一种MSE还没试过”。

共同E仍允许语言或场景捷径；MT-only H仍未教会任意parent的反馈修订。矩阵目标仍高维、对所有输入通道等权，
没有按真实hidden占用加权；条件decoder可能仍学不到正确方向。匹配线性层不等同已经匹配完整非线性控制。
这些限制与旧负证据保留。完整EMBER目标、相邻稳定性、视频/经历因果和held迁移没有因本批而降低要求。

## 4. 唯一新批：同图共享状态承接，改变监督目标

建议新ROOT：`/data1/user/ymdai/ember_runs/video_guided_proposal_writer_20261012/operator_effect_learning1296_query16`。
从原完整v7 CP864承接ψ、双meta、Reader/static/decoder、未训练θ、AdamW状态、logical sampler及rank RNG。
新目标生效后新增432更新（48次/task），终点总cursor1296；不fresh重跑、不重置零head、不丢弃前864有效学习。
AdamW既有lr1e−4/betas(.9,.95)/eps1e−8/wd1e−4/clip1保持，动量完整承接；不进行优化器或参数小扫。
这是一项明确登记的目标变更承接，不称原训练合同exact-resume。参数图没有改变；新checkpoint objective/schema
须拒绝把旧因子MSE与新目标混为同一语义，原v7只允许这次显式受控导入。相同目标下的新恢复仍完整检查。

保持exact L、原action-hidden有序V/stride5、双native meta、真实非空MT-H及producer、当前P=MT、
P0/S_G、完整38/r8/643584参数、六层归一化通信、两层static、Actor/三head、原生policy10 flow与全部评测RNG。
唯一更新owner替换为operator目标；不保留旧factor-MSE活动fallback，不并列新Writer/runner，也不顺手修Actor。
数据仍train24+support12全36、每逻辑B4/task等权；仍严格MT_only/version0，不能读入已有学生parents或bank。
本批不使用新增离线动作/state标签，不读取held标签，E始终只在监督侧，不进入Reader、候选、H或奖励。
D_l仅标签尺度，随同新run contract保存38个数及E来源；不由新G/query结果重估。

可在总1080保存完整恢复点，禁止读出/选点；唯一科学出口1296。
实际训练记录operator loss、同批MT基线、A/B参数误差可作廉价诊断，但不能据曲线提前挑点或自动超过1296。
学习不增新的model forward探针；实际消费者的必要梯度/形状/finite及冻结检查由实验闭环完成。

## 5. 全36首学生与同576配对

终点1296后沿原G864那36个teacher_demo、MT-H/P、实践初态、scene/env/policy/native RNG，
每task一次确定生成和一次真实实践；每condition最终仍一套完整LoRA。
随后同query16新G576行，与MT/G864/E原各576组成四臂。三套对照实际兼容则直接引用，不改变seed求缓存。
E=MT九task同源引用保持，旧历史成本不重计；预计新增612 episodes，不增加Teacher/π/warm/RL/held400/controls。
同14/16/16/4角色池、72 teacher-selection重合行/504其余行分列；这仍是train机制面板，不是正式资格。

报告完整per-task/suite、train24/support12、成功覆盖、R/G/L、churn、重合以及E-only/MT-only/both分类。
从保存的36套端点作一次CPU A/B、BA（含本目标相同D_l）及原有公共/条件分解；
复用已有首次原生动作记录，不为首动作诊断添加forward、hybrid或scale sweep。
本次新批结束后全部结果一次交main；没有中途科学出口或后台自动后段。

## 6. 结果分支约束完整假设

- 若算子拟合/B有效写入仍弱且闭环不改善，降低当前Reader+分块完整因子decoder学习共同E的可行性，
  不再靠延长、B权重、rank/seed/LR、重启噪声或再加一层通信保护此组合。
- 若算子拟合明显改善而控制仍无净增，说明这些矩阵误差改善不足以保持所需实际功能；
  不能以BA norm/cos漂亮承认方法有效，也不能未经依据自动串接另一loss。
- 若同576产生真实有分布支持的净收益，才支持下一阶段完整反馈/选择学习与held资格判断；
  +1或孤立task高点不能代表明显超MT，所有保持代价和视频/H未知仍报告。
- 不论分支，到1296+36学生/query16结束；main消费后自主接续，不等Owner逐项许可。

## 7. 资源、时间与交付职责

原两rank新增648实际3.325h，按432外推G约2.22h，读出约.97h，总约3.2h。
新算子loss尚未实测，预计正式约2.5–4.5h（2–4张实际有效A40），工程/必要消费者验证约30–60min。
不得把预测当完成或把旧参数loss profile当新目标的完整消费者实测；只做最小必要可丢弃profile，
不污染导入CP864/Adam/RNG。读取/decoder没有变化，旧最长视频chunk16/32约20.5秒且后者无收益的证据可复用，
若新增目标或实际设备改变吞吐/峰值，再按实际余量验证物理配置，不扫科学超参。

新增root峰值预估16GiB；source/MT/E/dataset大资产原位只读复用、无复制。所有新输出/cache/TMPDIR在data1。
实验现场核data1独立quota和共享空间、双节点GPU及项目总卡数，保持常规合计6/8、node≤6和非干扰要求。
物理训练world按实际有效吞吐选择，不锁死2卡；完整边界迁移须承接逻辑B4/task/loss/optimizer/sampler/RNG并登记，
不称bitwise exact。没有新增GPUh硬门槛，也不以逻辑B4或world2作为整批常规并发上限。

实验自主完成最小工程、目标/梯度实际消费者检查、Git集成push、clean detached、profile/launch与运行恢复。
确认B=0时目标不因只改A下降、E自身目标为零及正常AB梯度语义即可；不要新建广泛测试框架或重复完整历史检查。
原864训练/原件不改；新计算身份、目标迁移、命令、费用和资源记录齐全。main不重复工程ACK/测试/GPU准入。
请同时整合main CPU新发现和本决定到已有analyses/design§14/findings/state/history；AGENTS不加动态年表。
一次真实启动身份/ETA（如有实质变动）及唯一整批完成/真实异常回报；不心跳、不子阶段Queue、无第二等待者。


## 8. G具体架构核查补充（main 2026-10-12，当前批合同不变）

2026-10-12，main科学分析。依据Owner最新要求“G的具体架构你也可以仔细研究下是否需要改进”。
本次直接核对当前实际model/native/runtime/learning/run源码、G864完整结果和最近似历史原件。
不改变已派发的BA监督432更新至1296有限批，不增开实验、checkpoint读出、GPU探针或重跑已有算术。
工程/Git窗口仍由实验session独占；本文应并入本轮现有科学分析与findings，不新建第二active design。

### 1. 当前判断

G不存在由源码定义的“B不能输出或没有梯度”限制。它确实输出全部38-target/rank8的643584个有效A/B坐标。
但当前归纳偏置是：把L/V/H读成256维内容，经每层38个target token读取，再靠层/rank/side/block地址和共享
非线性网络解码为完整因子。它没有显式利用真实A、B的执行角色来组织生成，条件信息进入细坐标的路径也较间接。
这些是可核对的结构事实及值得改进的风险，不是已经确诊的唯一失败原因。

G864参数MSE .843335、条件增益 .054377是真实学习，但约99.9%的总MSE改善来自A；B目标MSE .999695。
实际BA相对E误差 .999573、更新范数比 .021085，控制341/576对MT340/E455。现有信息足以否定“总参数loss
在下降所以G已经逐渐获得教师控制”，不足以把失败唯一归给Reader、decoder容量、梯度抵消或因子规范。

当前决策仍是同图BA监督的有界检验。若实际算子和闭环获得进展，不为架构美观重写G；若仍没有有效写入，
就降低并结束继续修补当前Reader＋分块广播decoder＋共同E回归组合的投入，不再靠延长、拆头、放大B或加层保护它。
该批未结束前不把结构假说变成同时改图的第二变量，也不需要Owner逐项许可才能在结果后自主接续。

### 2. 实际输入与压缩在哪里发生

`src/ember/proposal_writer/native.py`的MetaReader在冻结MT合并基底上，用真实双RGB、exact language、原生
非空prefix和固定独立Gaussian动作latent/time=1计算两条native流。两组可训练rank32 q/v meta只用于读取；
不是teacher动作标签、真实teacher proprio，也不保证native H已经代表正确操作程序。
每帧送出Z为512×2048、H为50×1024；语言返回冻结原始token embedding，图像/动作native hidden已受语言条件影响。

`ContextReader.teaching`把Z/H/language分别投到256维；8个learned frame queries读取该帧原生内容，
加真实frame index位置后，4层video transformer保留8×帧数个有序token。语言token再读这段视频。
这保留了时序输入，没有平均整段frames；但每帧数百native tokens先压为8槽，不能因输入保序就认定精细接触和运动已被保留。

自身经历按真实replan记录：前后双RGB、首末flow native hidden、实际5×7动作和执行mask、前后8维proprio、
reward/success/termination等实际反馈、episode/step及producer参数摘要。每次replan以facts构成的query读取
图像、hidden和参数memory，压成一个256维事件；事件对视频读取后再经4层history transformer。
最终memory包含视频token、history token和decision token。因此经历/反馈确实进入G，不是只写日志。
但事件压缩与注意力权重不证明已识别失败原因；当前也没有专门的object pose/contact标签输入。

这里有两处可学习的信息压缩。它们可能足够，也可能丢掉任务相关细节；本次没有对它们作因果干预，
不能从“8槽/1事件”字面判断太小、默认扩槽或重新引入更深native Reader。合法信息和当前训练的不足应分开解释。

### 3. 条件怎样到达完整A/B

`FactorLayout`按真实A的行、B的列及rank排列，每64个native坐标切一块，共10064块，含512个padding无效坐标。
每块有target/rank/side/block四类可学习地址，标签中的task ID没有作为地址输入。
入口是`[(P-P0)/S_G, valid] -> Linear(128,256) + addresses`。当前MT-only训练的P=P0，故数值入口为0；
绝对A0不作为输入送入decoder，仅在输出还原时与S_G*prediction相加。网络可经固定地址学会其作用，
但没有直接看到当前真实A0的数值。额外给A0也不会增加任务信息，不能把它本身称作新教学知识或必然修复。

每个BlockLayer的顺序是：

1. 每块pre-norm FFN，保留本地残差和地址；
2. 同target/rank的A、B所有块一起作加权池化，进入304个rank tokens；
3. 每target内8 rank自注意力，rank池化成38个target tokens，再作跨target自注意力；
4. **只有这38个target tokens直接cross-attend条件memory**；
5. 归一化的target消息进入rank，再把rank/target消息广播回同rank所有A/B块。

六层各有独立参数，最后对每块LayerNorm，共用一个256→64线性输出。
可将一层实际依赖简写为（省略attention的内部更新）：

    b_i^k = F_k(b_i^(k-1)) + U_k [ r_(target(i),rank(i))^k ; t_target(i)^k ]
    e_i   = W_out LN(b_i^6) + bias

同rank内A/B与各coordinate块收到相同的广播向量，但它们的本地地址、parent块和FFN状态不同，
后续FFN及LN使对条件的响应可以不同。因此“广播一样，所以所有B元素只能一样”是错误结论。
同样，256→64共享输出不构成每块64维输出的必然线性秩亏；不能把旧family head的特定几何限制直接套过来。

实际风险更具体：细坐标的不同内容选择，要由共同消息与本地地址之间的非线性交互间接学出；A/B角色先混池，
B不会先读取已经生成的完整A或A在真实hidden上的响应，再生成与之配合的写入。最后一次memory注入后也只有
LN/线性输出，没有再经过坐标本地FFN；前五轮仍有这种后续非线性，所以这不是整条路径被截断。
广播归一化已解决此前消息幅度冲淡本地坐标的问题；当前风险是信息组织和信用分配，不能把旧幅度故障改名重讲。

### 4. 为什么这些事实还不支持马上换架构

原factor loss在B未拟合非零目标时对B有非零监督导数，共享网络也可为不同条件输出不同B；不存在自动零梯度定理。
然而不同任务B方向对同一共享头的更新可能相抵，而容易学到的A公共/条件部分可以让总loss下降。
“可能相抵”是一般机制和竞争解释，本次未计算per-task梯度，不能写成当前实测根因。

BA监督已针对另一个确定缺口改变目标：B=0时，只改A不会使实际线性更新变好，也不会降低BA损失；
非零A给B的起始信用，随后A/B共同学习。它会改变通过同一网络回传的信用，却不会改变上面的内容读取路径，
也不保证内容可辨认、矩阵近似可学或闭环有效。由此可以获得比同时改loss与decoder更清晰的下一次证据。

还有一个独立的学习合同缺口：目前每task只有一段固定真实MT-H、一个共同E，P也是同一MT；视频虽变化，
真实尝试及失败类型没有在task内变化。因此模型可以用L或固定经历识别任务，不必学会“这次怎样失败，下一次应怎样改”。
同一任务的最终正确策略可以相同，所以共同E不是数学矛盾；但它没有单独保证反馈依赖的可泛化修订规律。
不应声称提高Reader容量、接入reward字段或完成这次BA批，就已完成EMBER的多轮教学指导学习。

### 5. 最近似历史实际约束了哪些选择

| 候选直觉 | 已核对原件 | 对当前判断的限制 |
|---|---|---|
| 将每层A/B换成更独立、更大的输出head | `3a6f801d:docs/action_forecast_writer_target_owned_factor_design.md`全文，尤其§8；真实76套head，99→76→86→68/400，独立方向确实形成 | 拆头不是未试过的新原理；不同旧前端/配方不唯一否定现代对照，但显著降低无新信用机制就增容的优先级 |
| 只把地址保留得更好或直接回归专家因子 | 原expert-manifold§24–29、findings§63：地址绑定由48到75仍公共收缩 | 当前已将坐标放入本地value/残差，不能把该修复当新构造；因子拟合有真实历史负证据 |
| 先生成A，再据完整A生成B，直接使用native坐标 | `conditional_read_write_architecture.md`§5–9、完整450/900结果137/140，对MT153；训练侧91→115/144 | 显式A→B计算和真实功能梯度曾提高训练获取，未形成held优势；不能凭数学配合关系保证迁移，也不能称全新 |
| 用动作含义更明确的中间量调制A/B | `control_calibrated_read_write_design.md`全文及完整结果122/400、77/144 | 动作局部获取增强仍可能不能改善完整控制；不默认再加动作辅助头 |

这些历史使用不同基底、rank、数据/监督与学习年龄，不能与当前576总分排成单变量名次；它们约束充分性主张，
不证明任意解码器、结构耦合或功能监督都不可能。当前在强MT上、真实H及rank8共同E的条件差别同样保留。

### 6. 结构改进的优先问题与本轮决定

如果本次BA目标仍学不出有用条件写入，后继应重新组织**参数坐标怎样从教学和经历选择内容**，
不优先增加同一广播链的深度/宽度或直接扩大末端head。一个具体、可区分但尚未采纳的结构是：
让含真实target/rank/side/block地址和当前参数块的query直接读取条件memory，再进入本地decoder，
而非先只由target读一次再给所有块广播。例如替换当前memory入口为

    b_i <- b_i + Attn(Q(LN(b_i)), K(memory), V(memory))

其区别是不同side/坐标可在softmax选择阶段就读取不同证据，而不仅在收到相同消息后学习不同响应；
rank/target通信仍可承担协作。这不是新增教学信息、分任务参数库或恢复旧独立大head，也不等价于保证B与A配合。
代价是更多query及其梯度；共享K/V、物理chunk等只涉及吞吐，不能先承诺零成本或用吞吐理由代替科学判断。
当前按源码尚不能知道原target广播究竟丢失了哪项有用证据，因此这只是有明确算子差异的候选，不是已确定的下一批。

显式将A的完整数值/响应供给B是另一种机制，已受旧CRW正反证据约束；本轮不把两者捆成一次“全面升级”。
若需要该改动，必须说明为何改变旧失败预测，实际特征从何而来，以及其梯度和部署作用；不能只换名称。
若BA拟合改善而实际动作/控制仍近MT，应首先解释改善落在哪些真实执行方向，再决定是否值得上述内容路由重构，
而不是把任何一种结果都解释成预定新架构的理由。

这次核查的产物是对现有G的机制判断和改进优先级，未声称新结构通过验证。已派发的同图BA有限批照原合同推进，
到唯一1296＋全36/576出口后，main根据参数算子、实际动作和闭环分布联合判断并自主接续。
同一目标仍是完整Writer明显超过强MT、说明教学/实践如何形成能力；内部BA或条件差异不会替代这一接受标准。

### 7. 核查范围

当前源码实际读取：`model.py`全图、`native.py`实际prefix/meta计算、`runtime.py`初始化/输入身份、
`learning.py`G更新及MT-only bank、`run.py`同initial教师起点；原件为当前G864报告及已完成A/B、BA CPU算术。
历史读取包括上述完整Target-Owned原设计/结果、CRW实际读写/梯度与完整结果、control-calibrated完整设计/结果。
没有新增训练、CPU模型前向、GPU占用、环境episode、held读取或对历史最终shuffle/reverse的重新设计选择。
源码核查与数学依赖支持上述结构事实；根因和新架构优越性仍未验证。本文不是工程bug报告或新的实现/启动许可边界。

## 9. 工程准入与真实消费者验证（2026-10-12）

唯一G更新owner已替换为完整BA目标；六层decoder/static/Actor和所有native输入/评测RNG未改。
新模型schema为v8，旧v7仅允许run contract指定完整CP864的显式目标导入；新同目标恢复保留所有共享参数、Adam、cursor与rank RNG。
D38直接复用main已有1368条teacher能量原件，未重跑旧864参数分解；范围.00229936198–1.10991009。
57项针对性检查通过，含Gram/稠密损失与完整AB梯度、B=0不奖励A-only变化、E自身和等价分解。

冻结计算commit 2a1fb81393b2e483031e41746dd6c5db23717560，clean pushed detached。
新ROOT/profile_new_objective/attempt_001真实L/V/MT-H的两rank验证退出0，485份Adam状态step864及各rank RNG实际恢复；
v8完整保存/恢复step865通过。5个可丢弃logicalB4/20条件，没有新增episode；源policy无梯度，原CP864不写。
profile wall136.697秒/.07594284GPUh，所有模型/optimizer更新discard，临时验证CP已在消费后清理。
最长92帧实测chunk16/32热更新20.9335/20.8611秒、峰值reserved26.3242/42.1875GiB。
32的单次0.35%改善在正常计时波动内，亦未推翻旧匹配profile的无收益证据；选择16，不再放大64。

本轮正式在2026-10-12 UTC05:16:28启动有限37-job协调器，G实际进程UTC05:16:36启动。
物理world2来自当前唯一具足够余量且空闲的同节点两卡，非科学上限；读出在G退出事件按live双节点重新选有效卡，并遵守常规6/8、node6。
独立data1 quota启动准入used1467814172KiB/soft2147483648KiB，16GiB新增峰值可容纳，大资产原位只读。
新loss真实profile外推中心3.23h（G约2.26h+两卡首学生约.97h），范围2.5–4.5h；完整私有命令/准入/账目留新ROOT。
只1296固定出口、1080恢复only、36实践+576 query16与一次CPU算术；不自动后段，整批唯一退出回报机制保持。

正式G启动后的两rank code/G_resume回执已确认485份Adam step864、cursor/version864及各rank RNG、MT-only bank0，fresh=false；不只依据profile或命令认定恢复。
开发分支/隔离树已在clean、集成push且无运行依赖后退休；正式冻结树、原CP864及全部证据保留。

## 10. Owner硬截止与物理读出修订

2026-10-12 Owner新增UTC+8 16:00前结束硬边界：目标15:50完成消费、15:55全部本批GPU退出，不能迁到gpu02延期。
原3.23h/2.5–4.5h预测保留历史。UTC05:27的真实update899、最近24步均18.403s/update外推G约15:30，
旧两卡读出3482.84s使完整链约16:27；只改ETA不能满足边界。原训练chunk32无稳定收益，且只在1080/1296保存完整恢复点，
没有任意信号保存/停在CP边界的消费者；不为拓扑迁移丢弃未保存更新，原G进程/图/Adam/采样/来源保持。

旧G864学生36个真实进程总wall6671.167s，其中compile1404.602、query3673.055，余1593.510包含加载/清理及原件I/O，
不能全归加载。即使理想去掉全部余项，两卡仍42.3min；因此改为一个模型/Runner常驻、独占领取原condition，
在现有多condition算子中合波原practice/query，按live GPU余量增加有效worker。高util不是共驻自动排除条件，
显存峰值、服务竞争和非干扰仍必须用真实准入/首原定组判断。六卡近旧速时读出约15min，保留15–25min安排预期与失败可能；
新端点可改变停止时长，共驻卡未必等速，不能承诺15:50完整结束。

统一refresh_group保留同真实MT-H、teacher_demo/P/RNG和一套完整LoRA；完成practice即时持久化，query逐行保存、引用MT/E去重。
模型/Actor/Writer/损失/学习合同没有改变，没有增加GPU探针或任何episode。57项针对性检查、真实SQLite独占领取及恢复消费者CPU验证通过；
暂态并发代码与后续冻结身份由实验整合，实际source/launch/physical/partial/billing原件留ROOT。

独立双节点tmux guard在UTC05:33初次armed，UTC05:53先armed V2再退休初版；绝对15:50停止新派发，15:54 TERM、
15:54:30 KILL、15:54:55核验，仅同UID/精确root+consumer argv/start_ticks及实际后代。V2读取execution_identities承接新source/owner。
CPU调度交接只SIGINT原coordinator PID、保留其G真实退出收据；新owner排他持原delivery.lock，原helper在打开socket前被非阻塞锁排除，
不是伪造已投递或新增等待者。只有新整批owner发一次真实整批退出；完成/部分读取、CP保全、原件/费用及缺项由实验消费后交main。
1080仅恢复、1296唯一读出，部分任务/行不得冒称完整576结论，截止或重启后不自动续跑。

## 11. 实际部分退出、配对消费与截止保护失败（2026-10-12）

本批没有完成原全36/576合同。G完整864→1296正常退出，36套真实端点和一次实践均保存；
读出只完成33条件/528行，其中31条件/496行在15:55前完成持久化，task5/7的32行整组在15:59:27才完成。
task32/34/35的48行缺失。唯一整批owner于15:59:53退出1；本批不恢复、不补行、不进入任何后段。
下文时间均为2026-10-12 UTC+8。原件ROOT仍为本文件§6/§9登记的operator_effect_learning1296_query16。

### 11.1 训练完成与控制读出的实际范围

训练来源2a1fb813，后续常驻读出来源c7e76b56，均保持原clean pushed detached身份；两者不混称同一消费者。
G真实job在13:16:36开始、15:29:43退出0，新增432更新/1728条件，每task48次、累计144次，
全程MT parent/原真实MT-H、bank0/logicalB4/chunk16、source无梯度。首/末完整36任务块BA loss .999555→.963048，
同批MT基线1；不能与原factor-MSE loss直接比较。末两rank实际meta梯度均为Gemma .0128606、action 1.96947e−6，
这是该监督回传的记录，不证明视频知识或执行因果。

完整v8 CP1080/1296均已保存。实际CP1296核得485份Adam状态step1296、两个rank RNG、
sampler/G_sampler/cursor/version1296、world2/logical4、supervision/bank0，以及明确的v7→v8目标迁移。
未保存G更新数为0；原CP864保持。1296仍唯一科学读出，1080仅恢复，完整恢复资产不授权截止后自动启动。
36个path的teacher_demo/MT-H/P/实践初态及RNG与原映射核对通过，均只有一次确定端点生成、一次真实实践；
query与兼容MT/G864/E原行逐行核对task/scene/state、环境和原生policy RNG及真实初始proprio。
所有新保存行只是train/non-held机制读出，没有held400或视频因果资格。

| 实际范围 | 条件/配对行 | MT | G864 | G1296 | E | MT→1296 R/G/L |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 15:55前完成 | 31/496 | 302 | 308 | 314 | 394 | 282/32/20 |
| 逾时保存task5/7 | 2/32 | 13 | 11 | 7 | 26 | 6/1/7 |
| 全部已保存，含逾时 | 33/528 | 315 | 319 | 321 | 420 | 288/33/27 |
| 缺失task32/34/35 | 3/48 | 只读参照存在 | 只读参照存在 | 缺失 | 只读参照存在 | 不补造 |

截止前MT→G net+12/churn52/Jaccard .844311，G864→G net+6（R281/G33/L27/churn60）。
全部已保存MT→G net+6/churn60/Jaccard .827586，G864→G net+2（R287/G34/L32/churn66）。
逾时两task对MT均net−3，不因不利而删除；保留原始32行并独立标记。
实际row.finished_at是Runner相对时钟，没有逐行UTC锚，故保守将两整组作为逾时原件；
不能从15:59持久化时刻断言其中每条episode的精确结束UTC。

全部保存的target train21/336为MT144/G864146/G147/E237；support12/192为171/173/174/183。
截止前target train19/304为131/135/140/211，support12不变；不将实际metadata中统一的train角色误当目标24任务齐全。
全部保存suite相对MT净变：Spatial−7/Object+6/Goal+1/Long三task+3/support+3；截止前Spatial−1，其余相同。
任务净优/平/劣分别为截止前12/13/6、全部保存12/13/8；task13真实+4和task29/36各+2保留，
同时保留task5/7各−3、task0−2及其它损失。33条件中E-only118行仅传递18（G86416），
both-success302行保持282（G864280），MT-only13保留6（G8649），both-fail95获15（G86414）。
teacher选点重合的实际66行MT42/G86441/G46/E53；其余462行为273/278/275/367。
截止前对应62行39/39/44/50和434行263/269/270/344；不以缺项面板写72/504或完整teacher泛化。

### 11.2 一次全36端点算术与机制含义

已有36个真实端点允许CPU算术，即使3个parent/query完成回执缺失，也不补造该回执。
一次消费外部wall8.542秒、算术6.406秒/GPUh0；原parent-required脚本保留，按真实path/attempt/权重读取，
现存33个parent交叉核对，A/B与BA公式未改。端点算术complete只指全36参数，不指整批或资格。
原件为ROOT/analysis/endpoint_arithmetic/{consume_endpoints.py,statistics.json,consumer_validation.json,launch.json,exit.json}。

完整BA层等权loss .961445（MT1，G864 .999513）；聚合norm ratio .092718/cos .099524/相对平方误差 .990141，
原864对应.021085/.020679/.999573。直接算子监督确实增大并改善实际BA写入，但仍远未重构E。
层等权loss与聚合能量相对误差的归一化不同，不把3.86%层等权改善与约.99%聚合改善混算。
原SG因子MSE为1.000878，A1.011741/B.991192；因子公共能量67.9293%，
相对自身常量均值的因子条件增益−.002835（A−.012717/B+.005977），27非零E内为−.008454。
这些是因子坐标诊断，不能直接断言BA空间没有条件信息或将退休factor目标的回退称为训练bug。
同一528行首5原生动作G1296/MT RMSE .007055、G864/MT .003741、G1296/G864 .006657；
G1296/E .091116、MT/E .090994，没有实际动作距离更接近E的信号，不能据参数写入增大解释主要闭环收益。

原假设“仅A因子改善掩盖实际算子近零”获得更直接的检验：完整A/B上的BA梯度能增加有效写入，
因此不是已证实的硬性B断路；实际算子拟合仍弱、已保存控制仅小幅净交换、E-only优势大部分未传递。
这削弱将目标更换本身视为充分修复的解释，不能把31条件小幅收益升级为全36优势，
也不能因工程缺项抹去已完成训练和不利个例。视频/H内容选择、细坐标条件映射、学习量和真实执行保持仍未唯一辨别。
本结果与§8列明的Target-Owned/CRW/动作调制负证据共同约束后继；不自动延长、拆头或放大B。
main消费实际部分证据后自主作出后继科学取舍，执行者不以缺48行自行重启原队列。

### 11.3 两项工程失败、硬截止违约及定向修复

gpu01:6 worker00在完成task32/34/35实践后，首次Long query初始化离屏环境失败：
mujoco.FatalError: Offscreen framebuffer is not complete, error0x8cdd。
原log为ROOT/jobs/student_resident_00_gpu01_6/attempt_001/process.log；这是EGL环境失败，不是G训练阴性或已确定CUDA OOM。
同为slots32的gpu01:0实际成功，现证据不足以断言所有32-slot配置不可用或证明唯一显存原因。
其3个SQL claim登记failed，没有G query行；整批owner排空其它在途worker后才退出1，没有截止后重试。

gpu02 guard V2在15:50对CPU coordinator做argv身份校验时抛出registered coordinator argv identity mismatch，
异常中断了之后15:54 TERM/15:54:30 KILL/15:54:55 verify时刻；dispatch_closed存在，后续清空回执缺失。
原日志未记录具体mismatch PID/argv，历史具体原因未知，不能写成已确诊zombie或PID复用。
实际consumer的时间检查仍阻止15:50以后新领取；task5/7在15:49:48领取、15:59:27才完成，gpu02:4在15:59:52退出。
这不是CPU回执晚写导致的假逾时：GPU consumer确实超过15:55，实验session的截止保护没有兑现Owner硬要求。
gpu01原guard最终15:54:55 cleared=true，所有gpu01 worker在15:51:53前已退出；当前双节点已核无ymdai GPU进程。
现场只读快照及可读清空结论留ROOT/coordinator/deadline_ops_20261012/post_exit_live_snapshot.json、post_exit_batch_clear.json。

原guard不修改，定向候选deadline_guard_next.py仅在ROOT ops目录保留：已停止/消失/僵尸身份跳过，
不匹配身份记录观察值且不发信号，CPU派发身份异常不取消后续GPU终止时刻。
CPU验证复现原异常，再注入同异常测试真实计时路径，owned临时CPU consumer被TERM、无关CPU进程保持运行、final cleared=true。
证据guard_failure_CPU_validation/{validate.py,validation.json,process.log}；0GPU/0forward/0episode。
候选没有重新armed或修改冻结训练，不能追认历史截止合规；未来复用仍须按实际本批身份和绝对时刻配置。

### 11.4 实际费用、原件和保全范围

7个真实GPU attempt：G4.437115、学生1.992868，formal合计6.429983GPUh；失败worker费用.073703已计入。
新目标准入profile .075943另计，总本root6.505926GPUh；不把历史864费用或兼容MT/E/G864行算作新增episode。
已完成36实践+528 query共564新episode，记录环境步下界124301=8595新实践+115706已保存query；
复用MT bootstrap8131步另列，path累计environment_steps不能全部作为新增费用。
失败EGL query可能有未完成初始化/执行，没有完整counter，实际失败query数量/步数未知，不写成0或伪精确全部episode账。
5个成功worker物理counter累计122941步，与另3个已保存失败组实践1360步合为上述下界。

实际读出启用6张distinct GPU，逐job准入核普通cap6、node最多4，未暂停或信号他人任务。
正常worker slots32/16/8的reserved峰约25.793/18.803/13.633GiB；gpu02:4结束只剩.030GiB且最晚退出，
共驻余量/服务竞争是执行限制，不能将单一world2或util高本身当并发禁令，也不从最终余量证明EGL唯一原因。
读出新增GPU窗口约30min，高于预估15–25min；guard缺陷使最后组未按15:55停止。原估计/启动/失败账均保留。

CPU原件消费命令/exit0/统计在ROOT/analysis/whole_completion_20261012，
consume_originals.py核job/SQL/实际CP/训练/36实践/配对/费用，complete_tables.py按注册target/support分列。
ROOT/analysis/first_student_query16_summary.json明确complete=false、missing/late范围，原始配对行在同目录rows.jsonl。
所有参数/信息墙/配对/完整CP消费检查通过，不表示整批成功；已知EGL、guard及逾时均显式保留。
原864及新1080/1296完整恢复状态、36权重/H、全部raw rows、失败记录及两个冻结工作树保留。
开发branch/worktree已在clean集成push后退休，不删除当前关键原件，不自动任何后段或截止后续跑。

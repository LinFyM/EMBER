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

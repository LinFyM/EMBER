# 生成场消息传播与query16：机制推导和有界检验依据

本文保存科学理由，执行 authority 仅为现有active design§11，不是第二份active合同。

2026-10-12。主讨论01a12bb0-5273-7462-b5e2-72a18aba7b8d作出科学决定；实验session负责实现、验证、Git、资源准入与执行。
这是一项有证据依据但尚未验证有效的修复。当前旧RL9/strict400继续原合同，不被本决定中途替换。

## 1. 目标和Owner最新优先级

第一目标是在已合并强MT的冻结基底上，使视频Writer生成的完整固定LoRA明显超过MT；暂不开展未merge MT的source方案。
优先级按当前影响、修正成本、预期收益排序，不限于已证实bug。4个query初态过粗这类直接影响训练信用的问题及时改；
耗时较长、尚未显示为主要瓶颈的教师全面完善继续后置。硬约束内主讨论自主裁决，无Owner逐项批准流程。

## 2. 原件与因果判断

只读诊断原件：`/data1/user/ymdai/ember_runs/video_guided_proposal_writer_20261012/normalized_flow_query16/analysis/previous_G216/native_diagnostic/`，
包含diagnose.py/statistics.json/analysis.md/launch.json/exit.json/accounting.json；均为previous_G216证据。
main已直接核对脚本、原统计及实际model.py/learning.py，未仅采信摘要。已完成G216原件CPU分析为
同一稳定previous_G216目录的`initial_generation_analysis.json`，推导工作稿`transport_reasoning.md`亦保留。

- 教师独立面板E453/576、MT350；首学生另一个面板G83/144、MT83，得9失9。两个面板不能直接比较传递率。
- 36首候选与E的归一化MSE均不优于MT；均值1.99996。完整CFM后段均值约2，符合零场基线。
- task13 E=MT的合法CFM精确标签为−xi；G216预测MSE约1.0004、零场约.99946，未学到此子问题。
  task17 E非零时预测MSE2.70245、零场2.70377，仅约.049%改善。两任务生成与原始xi余弦约.99947。
- 同group局部block差异RMS保持约1.177，六层block RMS由4.46升至4466；末端LayerNorm后差异仅.000264。
  各层实际将raw blocks汇入ranks、raw ranks汇入targets，再将raw targets与raw ranks广播回blocks。
  query/FF内的LayerNorm未约束这些循环消息的量级。真实memory RMS约1；条件和局部输入都可能被压弱。
- 合法t=0/.5噪声翻转时应有约2的速度变化，实测约.0024/.0018且正确方向分量近零；optimizer与代表参数确有216次更新。
  固定x只改t的比较也很弱，但它可能离开相应t的典型高维训练密度，不能单独作为时间错误的裁决。

设同group的block为b_j=c+ell_j，共享分量c很大，则LN(c+ell_j)的一阶局部差异由J_LN(c)ell_j给出；
其非退化方向的尺度随1/std(c)缩小。这里实际差异缩小比例约2.24e−4，与原件量级吻合。
标签包含每个坐标独立的−xi，靠同group共同输出不能预测它；在局部信息难以利用时，去掉无关输出可降低loss，
却让16步积分保留xi。以上加强具体传播机制解释，不证明它是唯一原因或修复必然提高闭环。

## 3. 唯一架构干预：生成场的跨层消息采用发送端归一化

只改变G.field的六个BlockLayer的跨层消息。Generator.static和Actor.static两层编码、ContextReader、meta、
三head选择器保持原算子。可在现有模块中按flow/static职责组织，共用已有参数结构；不得保留旧flow作为fallback运行面。
不同时增加AdaLN、额外时间网络、解析去噪、endpoint回归、输出直通头或新的loss。

对每层，记现有三组LayerNorm为N_b/N_r/N_t。具体算子：

1. b'=b+FF(N_b(b))；用z_b=N_b(b')计算原rank_query softmax权重，pool的**value也用z_b**；r'=r+pool(z_b)。
2. 原rank self-attention继续用N_r(r')作Q/K/V，得到r''；原target_query的权重与pool **value都用z_r=N_r(r'')**；t'=t+pool(z_r)。
3. 原target self-attention及真实memory cross-attention保持，分别在现有N_t归一化query上作用，得到t''。
4. 下行rank消息为r'''=r''+N_t(t'')[:,None]。
5. 下行block消息为Broadcast(concat(N_r(r''')[group],N_t(t'')[target]))；b_next=b'+该消息。
   原最后N_b之外的field.norm与64维读出、mask和16步Euler保持。

所有pool仍是原group内softmax加权平均，target/rank/address/原始坐标组织及643584个输出坐标不变。
每个跨层发送者在传出前归一化，因而在归一化仿射和线性权重有限时，单次消息不随累计接收状态的幅度线性增长；
切断原raw-state反复放大路径，同时保留b的局部残差。不是把完整block状态每层清零，也没有禁止幅度信息通过原始输入学习。
这改善可学习的条件：−xi所需局部坐标不再被同组巨量广播机械压低；E−P所需条件信号通过memory进入t/r/b，
真实CFM误差仍经同一读出、消息、Reader和双meta回传。learned仿射可能变化、归一化可能丢失部分幅度信息，
网络仍可能学公共均值或忽略video/H；这些是需由实际结果判断的剩余风险，不用一般梯度可达性宣称已解决。

初始化保持原seed及参数形状/顺序可比，所有共享G/meta/Actor与optimizer fresh；旧G216/RL参数不迁移到新算子。
新checkpoint/schema明确区分计算语义，不能把同shape当作exact resume。原E、S_G、初始task A0/B0、MT合并基底只读复用。

## 4. query16及显式数据角色

当前每task 50个登记初态，原teacher_report16及有限train_report4继续独立保留。
在新run root按既有seed/init_order，新的practice pool为原practice_states前14个；query pool为原query_states10个
再接原practice_states最后6个，共16个不同初态。即practice=init_order[:14]，
query=init_order[36:46]+init_order[14:16]+init_order[32:36]，teacher_report=init_order[16:32]，train_report=init_order[46:50]。
四个池覆盖50且互不相交，旧MT bootstrap为init_order[0]，仍可合法复用。
其中query的init_order[14:16]曾参与E选点；query独立于本condition的实践，不把它称为teacher选点独立验证。
原teacher独立report16仍保留原身份；本面板仅作合法train36的训练/机制读出，不构成泛化或formal资格。
此改变不减少当前每condition实际R5资源：first visit的6条实践所需初态仍来自原相同前缀；以实际路径核对，无新固定轮数限制。

每个实际已实践候选在相同16个state/env/policy RNG上各评一次；以实际16行均值给local/RL信用，query绝不进入H/G条件。
任何/4、四条循环、cache/complete判据或费用计数均须随实际cardinality改正，不能复制旧4行当16行；每候选跨prefix只query一次。
旧MT行只有实际参数/task/state/scene/RNG均兼容时可引用，记录reuse/new；共享权重或候选变化不复用旧G标签。
不按G成绩挑易初态，不重复state充数，不读取Val/Test离线标签或挪用teacher独立report来训练。
16降低评分粒度与抽样噪声，不保证细小差异显著，也不把重复使用同16个state称为新增独立面板。

本轮首学生读出另外对同一16-state/RNG执行固定E作为**只读控制参照**，E=MT的9task直接引用同一MT行。
E不加入Actor候选集、不把E score用于模型/教师选点、不把E轨迹或query反馈塞入H；这不是教师完善或重训。
此参照用来分辨教师实际优势、G传递损失和能力保持；统计E-only、MT-only、both-success、both-fail中G的表现。
不以这些分层结果反过来挑选本轮训练面板。某task的类别不存在时如实报0，不人为凑齐。

## 5. 预测、竞争解释与结论范围

本次检验在相同G216有效学习组织上只改变发送端算子，训练本身尚不消费query16，因而传播学习比较仍单一主要变量。
只在原task13/17真实context和首学生原件上重复同一简洁读出；完整执行范围、输出、数字及停止线唯一登记在active design§11。

读出同时回答：新场是否摆脱零场并生成相应任务编辑；在相同初态上是否传递E的实际净优势；保持/得失分布如何。
CFM下降但仍是无条件noise缩放或公共均值时，不能称教学映射已学会；近E但控制差时转向功能传递解释；
训练task有控制增益时，也还不能宣称held高于MT或video/H因果成立。若仍完全复现原压制/噪声保留，不自动延长/小扫保护假说。
已有完整循环结果若与本定位矛盾，保留并解释，不为证明修复必要而忽略旧方法正例。

## 6. 历史、资源与回报

历史expert-manifold已做raw factor MSE/cos/log-scale，macro50为48/75且公共方向收缩；直接native heads/AB residual也非新充分条件。
χ/φ已有完整参数、真实H和功能梯度，仍127/130/143。当前修复针对已实测的消息传播问题，未解决这些历史的共享映射/保持/迁移难题。
共同E允许忽略H/video、确定端点不保证有用随机探索，继续保留。teacher有真实损失，不能以拟合E承诺全task单调提高。

预计G约50–100分钟（旧4卡47.52分钟，具体拓扑/新算子实测修订）；首学生及扩大后的配对控制读出约1–3小时，
总计算暂按2–5小时、另约20–40分钟工程与准入安排。不是保证；只有2卡或现场负载明显增加时须按实测修订并通知main。
执行者按当前live双节点/总6或8张/单节点6、非干扰和实测吞吐决定物理安排，不能因旧world4而等待或额外收紧全项目并发。
新query最多36×3×16=1728逻辑行，加36次G实践；E=MT及兼容历史MT引用减少实际episode，按真实新增计费。
新root以复用为主，峰值按真实H/CP/日志估计并核data1独立quota；不据此创建新大缓存或复制资产。
原长批次及其唯一退出回报者保持。新有界任务只在整批完成/失败回报main一次，不按子阶段发心跳或叠加自Queue。
发生真实工程违约由实验自主修复；若需要改变本科学算子/标签/任务权重/角色/预算范围，报main裁决，非Owner许可门槛。


持久原件：本稿引用的旧G216诊断和CPU分析已保存于
`/data1/user/ymdai/ember_runs/video_guided_proposal_writer_20261012/normalized_flow_query16/analysis/previous_G216/`；
该目录只表示previous_G216证据，不是新方法结果。新批唯一执行合同见现有active design的§11。

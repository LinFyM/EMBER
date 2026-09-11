# 无变化参照的局部过程更新

状态：2026-09-11，完整冻结机制诊断已支持采用此候选进入真实profile；正式学习须使用下述合同、profile通过和clean pushed frozen runtime。当前执行状态以canonical `progress.md`为准。

## 已观察到的问题与未确认的因果关系

C200正确视频的独立FM为0.112128，source为0.152494；完整无序S为0.111931。重复首帧为0.112543、乱序为0.112538，正确视频有微小平均优势，但相对source的主要收益不要求正确过程。固定历史八task的语义S后，用静态视频替换P4，FM仅比正确过程高0.000044；完全替换错误语义则高0.000777。配对task bootstrap和全部逐task差额保存在`runs/analysis/video_mechanism_20260911/functional_comparison.json`，不能把小面板解读为全局唯一根因。

相同八task中，重复首帧的中心化P4 RMS为正常视频的0.692–0.935倍。当前GRU减初始hidden，不能扣除相同画面在时间输入、角色读取、非identity软对应和不同历史窗口下本来就会产生的响应。层层past汇聚可以传播这些边界差异。中心化后的时变Value因此不自动等于视觉运动。这是已验证的表示性质；闭环损害与修正收益仍待实验。

## 唯一主要改动

保留真实native R/Z、完整H、当前到过去四帧关系、完整H query、两端视觉读取、短GRU、四组past长程及前三次回写、C的语义先行读取、独立native D和普通正样本FM。只更换局部更新的参照：

\[
M_{tu}=f(U_t,U_u,Z_t,Z_u,\Delta_{tu},H,L_t),\quad
M^0_{tu}=f(U_t,U_t,Z_t,Z_t,\Delta_{tu},H,L_t).
\]

两条消息序列使用同一真实gap、角色、历史起点顺序、窗口长度、参数和初始hidden。各自运行原GRU，再用

\[
\widetilde U_t=U_t+W\,[\operatorname{GRU}(M_{tu})-\operatorname{GRU}(M^0_{tu})]
\]

进入原pointwise FFN。参照路径只重读当前帧已经生成的真实native activations，不额外调用backbone，不构造zero-image、fake action query或缺少原生prefix的forward。两条路径都有正常梯度；没有detach、错误视频标签、正确性检测器、额外惩罚或标量开关。

画面恒定时，实际与参照消息相同；相同时间输入和历史长度造成的GRU响应相消。pointwise FFN、H-read和只在Q/K使用时间的past attention保持恒定；四组归纳后P4仍恒定，原中心化过程Value为零。语义S保留，整个LoRA并不被强制归零。视频中的真实变化仍可使两条路径分开；错任务与乱序不会被预设为零或人为压低。

这也改变了信用来源：对恒定原生输入，局部差额对任意共享参数都为零，因此共享参数不能仅靠相同画面与窗口长度生成的变化获得过程分支收益。对于真实视频，梯度仍同时经过实际与参照两条链；这不是冻结上游或给错误条件加负梯度。

此性质不保证过程语义正确、优化会使用过程、correct会超过wrong或绝对性能提高。特别是C的直接语义路径仍可承担公共收益；若新学习仍忽略有效过程，必须返回信用和消费定位，不能宣称静态不变性解决了全部问题。

## 实现所有权与验证范围

`LocalRelationBlock`继续拥有消息与邻居汇聚，新增计算使用同一模块两次，GRU把实际/参照作为额外batch轴执行。没有新runner、模块或参数，没有当前与旧更新的运行开关；历史由冻结7fedbe85保留。`local_relation_update=paired_nochange_reference_v1`区分科学计算合同，旧checkpoint不得被最新运行面误解为此方法。架构语义改变后必须fresh，不把C200权重迁入新训练。

候选在独立`codex/video-change-reference`准备，由主agent单独拥有；若闭环证据支持进入学习，完成正式配置和真实GPU验证后集成main，冻结新runtime；若不采用，保留Git论证并删除已保存且干净的候选worktree。没有恢复95-task或其暂停worktree。

已通过26项现有/相关CPU检查，包括完整H依赖、两端视觉、过去单向、集合顺序、梯度重放，以及1/3/8帧的恒定输入无中心化过程Value和真实变化的native-input梯度。尚未检验实际BF16/native视频数值、最长视频显存、吞吐或新学习。

若进入学习，保持C原来的2条件/task、每条件32queries、每更新4task及固定采样池、初始化seed和optimizer；不得同时改覆盖、学习率、D或增加辅助目标。完整观察范围与相邻闭环节点须在launch前登记，不能只看100/200两个早期节点就宣布收敛。本次机制goal要求correct可信稳定地高于source及错误视频；项目最终绝对性能资格与当前goal分开。


## 冻结行为依据与有限学习合同

完整原件与解释见[机制复核§10](video_mechanism_reassessment.md)。C100/200正确32→42、静态29→49、S30→51，各96；C200两错46、乱序45、source15。固定八task的语义S后正确/静态过程同为15/32。正确过程没有随能力获取出现净收益，局部参照修正具有可检验的具体对象；这不证明C收敛，也不保证新候选成功。

唯一正式配置为`configs/pi05_video_change_reference.json`。梯度仍为train24四suite各一task/update，每task两条独立K1视频条件，各32个跨episode动作queries；每更新256queries/8套独立训练条件。视频0–15、训练动作16–41，动作诊断42–45、行为视频46–49；stride5，native probe1729/flow time1，single agentview，source/normalization冻结，所有Writer和Meta从合法fresh共同训练。seed7、Adam3e−5、warmup8后constant不变；普通正样本FM，无错误视频训练、RL、aux或任务覆盖改变。

预注册四个学习节点100/200/300/400，对应25600/51200/76800/102400queries。训练器沿现有两节点segment合同：fresh100/200后完整exact-resume300/400；不从profile或C checkpoint初始化。默认完成四个节点以观察后续获取，不把两点早期波动当收敛。只有finite/OOM/合同破坏等工程失败才先暂停诊断；资源变化等待合法同topology或单独登记迁移，不静默重启。

各节点做validation8 correct strict paired400，single checkpoint，不用80row筛选、union或融合；同现有C/source固定tasks、states0–49、seed20260907、per-init ordinal视频。100/200/300/400亦做train24固定states32–35、teacher46的correct96；200/400补同task另一视频47、同suite错、跨suite错、真实乱序、首帧静态各96，复用本轮donor与呈现合同，无放回选同task教学输入，不挑视频。此开发诊断用于检验机制是否自然学成，不按最大差额选checkpoint。独立动作诊断0/200/400仅train24×128queries，不碰held actions。

主要判据为新candidate相邻节点的correct绝对成功率、相对source/两错/乱序的配对差额、same-task鲁棒性、逐task/suite与breadth、R/G/L/churn/J及paired task-bootstrap不确定性；和旧C的100/200比较只在相同节点与配对输入下进行。correct须实质提高，错误条件下降本身不能满足goal。若到400仍无净收益，则返回原因分析；不自动续到无限步、不扫seed/LR/rank。若存在可信获取或已出现自然优势但相邻不足，依据完整证据另登记延伸验证，不把本预算结束写成收敛。

项目后续最终方法仍以>145/400、相邻稳定、四suite与Goal/Long贡献及same-task-other qualification裁决，>145不作为本次机制goal完成门槛；之后冻结选择，再用未用于开发的独立输入执行最终视频controls。Test封存，本次训练任务诊断不能替代最终资格。

部署保持一套完整38-target LoRA与一次Writer调用。现有unified/frame-set模块暂只保留为本次机制对照的测试/初始化结构，旧R/C/S配置没有新架构身份，须使用冻结7fedbe85；不把它们登记成新训练候选。主agent负责在此局部修正裁决后清理不再需要的对照路径，当前不同时改变D随机初始化流或已暂停95-task worktree。

输出根预定`runs/outputs/video_change_reference_seed7_20260911/`；分析根`runs/analysis/video_change_reference_20260911/`。四个完整checkpoint约18GiB，LoRA banks/rows及临时profile峰值预算40GiB；共享source/data/env不复制。正式launch按真实吞吐选择单节点至多六张有用卡，固定NUMA/NCCL_P2P_DISABLE及world topology；命令、UUID、quota与frozen commit只在launch record记录一次。


## Launch前实际验证

26项局部数学检查与完整相关接口合计128项通过，包含旧架构身份拒绝、配对物化和完整恢复合同；其中一个测试只因隔离worktree未链接canonical data失败，补symlink后通过。没有为测试复制数据。

真实2步四卡/六卡profile均完成，source trainable与梯度均0，第二步梯度到达D、semantic、process、native输入和Meta。最长真实457原帧/93个stride5帧完成FM VJP及Meta重放、sampler未变。六卡step1/2为14.99/14.21秒，对比四卡26.05/25.02秒；采用六卡混合micro8/6，实际8LoRAs/256queries每更新，第二步约0.563 LoRA/s、18.02 queries/s。四卡峰值39.50GiB，六卡空卡≤38.16GiB、共驻micro6≤33.24GiB；选择按实际速度与余量。

实际BF16静态中心化P4约.00055，正确约.06071，静态为正确约0.9%；保留正常低精度数值，不把非零低位残差作为失败。仅证明计算属性和真实图可训练，不能说明闭环已经有效。原件`runs/analysis/video_change_reference_20260911/profile_summary.json`，所有一次性profile checkpoint已删除，不作为formal初始化。

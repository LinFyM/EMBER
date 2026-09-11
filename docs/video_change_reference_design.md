# 无变化参照的局部过程更新：候选准备

状态：2026-09-11，依据已完成冻结动作诊断与静态表示检查准备；尚未登记为active训练设计。C100/C200与路径替换的闭环尚未完整，当前不启动新训练。当前执行授权仍以canonical `progress.md`为准。

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

此性质不保证过程语义正确、优化会使用过程、correct会超过wrong或绝对性能提高。特别是C的直接语义路径仍可承担公共收益；若新学习仍忽略有效过程，必须返回信用和消费定位，不能宣称静态不变性解决了全部问题。

## 实现所有权与验证范围

`LocalRelationBlock`继续拥有消息与邻居汇聚，新增计算使用同一模块两次，GRU把实际/参照作为额外batch轴执行。没有新runner、模块或参数，没有当前与旧更新的运行开关；历史由冻结7fedbe85保留。`local_relation_update=paired_nochange_reference_v1`区分科学计算合同，旧checkpoint不得被最新运行面误解为此方法。架构语义改变后必须fresh，不把C200权重迁入新训练。

候选在独立`codex/video-change-reference`准备，由主agent单独拥有；若闭环证据支持进入学习，完成正式配置和真实GPU验证后集成main，冻结新runtime；若不采用，保留Git论证并删除已保存且干净的候选worktree。没有恢复95-task或其暂停worktree。

已通过26项现有/相关CPU检查，包括完整H依赖、两端视觉、过去单向、集合顺序、梯度重放，以及1/3/8帧的恒定输入无中心化过程Value和真实变化的native-input梯度。尚未检验实际BF16/native视频数值、最长视频显存、吞吐或新学习。

若进入学习，保持C原来的2条件/task、每条件32queries、每更新4task及固定采样池、初始化seed和optimizer；不得同时改覆盖、学习率、D或增加辅助目标。完整观察范围与相邻闭环节点须在launch前登记，不能只看100/200两个早期节点就宣布收敛。最终仍要求correct高于source及错误视频，并满足EMBER的绝对能力与稳定性要求。

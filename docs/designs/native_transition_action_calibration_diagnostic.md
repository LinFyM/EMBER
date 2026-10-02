# 原生动作先验的可见转移校准：有界学习辨识

2026-10-03。主讨论依据机制§104登记本项；Owner的自主推进和固定现有数据要求持续有效。
这是一次非部署学习辨识，不是新Writer或动作后验已有效的声明。实际接手、运行与结果只看progress。

## 1. 要改变的方法判断

问题是：在已有合法原生动作估计之上，真实到达画面是否能提供可跨task获取的已发生控制信息？
若不能，本轮不投资“先解释可见转移的控制内容，再编译”的具体native校准路线。
若能，才获得值得研究其LoRA消费方式的内容；本项不能证明跨初态控制、完整任务知识或超过MT。
不把中间通过变成后继训练的自动许可，也不把失败扩大为所有视频反演不可能。

最近似完整历史必须共同保留：

- v4已经将native十步动作forecast/Revision送入LoRA，但没有实际前后转移的动作校准。
- LocalActionGrounded已经用真实四帧反演期间15动作；旧fresh独立FM头的生成未胜task均值，其输出也未进入LoRA。
- LocalField已让真实动作产生的高维cotangent监督同一实际写入场；不能再把“直接监督写入”称作新原理。
- ProcessPullback已把自由7维q经native导数传到38层；其固定出口/free-q及共享出口负例约束任何后继编译。
- §103的当前执行hidden辅助信用未改善学生，且没有得到强Reader；本项不恢复该Reader、扩头或蒸馏。

本项改变的学习条件是：固定真实source的动作估计作为共同起点，只学习其对实际已发生5步动作的校准；
两臂有相同可训练函数，只改变第二份memory是否来自真实到达画面。没有LoRA/Writer学习或环境读取。
它检验的是这条具体获取假设，不宣称已识别当前EMBER闭环失败的唯一根因。

## 2. 固定数据与输入墙

复用当前coverage协议：`configs/libero_24_8_8_coverage_v1/protocol.json`及对应evaluation/dataset owner。
target train24固定为
`[0,1,2,4,5,7,12,13,14,15,17,19,20,21,22,25,28,29,32,34,35,36,37,38]`。

- 本次读出fit20：从上述列表排除`[0,12,20,32]`，仅使用每task既有demo16–19，共80视频。
- 本次读出内部留task四项：`[0,12,20,32]`，不产生gamma梯度。它们仍是项目train任务，不是Validation/Test。
- 统一无梯度读出：train24各demo42–45，共96视频；分别报告fit20的新episode及内部留task4。
- 总共176条现有视频，无新任务/示范/来源，不读取support71动作，不产生环境轨迹。
- Source采用`configs/operator_read_write_v1/conditional_read_write_fresh_spec.json`登记的aligned Source1000、
  tokenizer和冻结source normalization。不得换成旧source、T/public LoRA、MT、task expert或pi05_libero。
  source71训练对目标40的既有排除合同保持；不以本次内部留task冒称所有预训练来源均未知。

输入只有真实双RGB、exact language和固定public probe。teacher prompt省略State段；不补零state。
normalization只用于训练输出标签和输出评分，不能成为条件输入。task/demo/frame只用于合法采样和记录。
不读取teacher pose、reward、terminal，不把动作或由真动作加噪的latent送进source或读出头。

每条完整视频按stride5并保留真实末帧读取native，保存实际frame indices。
本诊断的监督单位只取相邻采样帧差恰为5且标签完整的区间`obs[p]→obs[p+5]`；
目标严格为`actions[p+1:p+6]`，使用当前post-action/offset1合同。
不足5的真实末间隔仍保留其native/索引，但没有本项5步监督；报告覆盖，不padding成5或改stride。
176条的全部合法5步区间均进入相应采样池/最终读回，不按动作、阶段、成败或误差筛选。

## 3. 相同原生先验、相同读出，只有到达memory不同

冻结裸Source1000在每个真实帧、双相机prefix、固定`probe_seed=1729`的`50×32`噪声、tau=1下读取：

`H_p ∈ R^(50×1024)`；`mu_p = (epsilon_fixed − velocity_source[p])[:5,:7]`。

H是原生action_out之前的真实hidden。完整50位置保留到下述学习attention，不先平均horizon或复制末层作多层特征。
mu是固定native点估计，不宣称它已经等于真实条件均值或完整十步生成。Source全部参数冻结，native一次缓存，两臂共用。
不得根据后继分数换flow时刻、probe、native层、source版本或先做十步扫读。

两臂从完全相同fresh gamma起步，均输出

`a_hat[p] = mu_p + r_gamma(H_p, H_context)[:5,:7]`。

- **F：当前画面的学习校准。** `H_context=H_p`。
- **P：已观察转移的学习校准。** `H_context=H_(p+5)`。

F只是复用已读的当前H，不生成假图像或做fake native forward。P读取的到达帧来自真实有序视频；
如果把区间表示记在到达时刻，这是已经观察到的转移，不需要看到到达时刻以后的画面。
本项不做shuffled/reversed，也不把F称为完整Writer的语言/静态因果control。

gamma的唯一固定实例：token内parameterless RMS（eps=1e-6），宽256、4个64维attention heads、dropout0。
对完整50个departure tokens形成`Q=Wq RMS(H_p)`；定义共享参数的学习读出

`C(Q,H)=Wo MultiHeadAttention(Q, Wk RMS(H), Wv RMS(H))`。

两次调用同一个C：`c0=C(Q,H_p)`、`c1=C(Q,H_context)`；每个horizon位置形成

`r_gamma = W2 GELU(W1 concat(Q,c0,c1)+b1)+b2`。

W1为768→1024，W2为1024→7；W2/b2零初始化，其余常规fresh seed7。
完整50位置参加实际Q/K/V读出；只对有真实监督的前5输出计loss，其他45输出不称为反演标签或用于部署。
使用标准线性/attention实现，不新增独立视频编码器、source复制、私有task参数、task embedding或多层扫描。

F里c1=c0。P可令W1的c1块为0、c0块为F中两块之和，精确表示任意相同参数类的F函数；
因此没有为了显示到达信息而削弱F的函数类。这个包含不保证相同优化预算会得到更好P，正是本次学习比较的未知。
两臂在零输出初始化均等于同一mu；不要求合法首步所有上游参数有非零梯度。

## 4. 有界学习和完整读回

只有gamma学习，Source及H/mu缓存始终冻结。每臂固定500更新、fresh AdamW：
lr=3e-4，betas=(.9,.95)，eps=1e-8，weight_decay=1e-4，clip=1；前10更新线性warm-up，随后保持该LR。
这是一组预先固定的实现选择，不声称最优；不扫LR、seed、步数、宽度、层位或辅助系数。

每更新同时包括全部fit20任务，各8个区间，共160；各task权重1/20。
task内先均匀取demo16–19，再在该episode合法5步区间均匀取样；两臂逐行复用事件流，seed20261003。
loss为source-normalized动作的前5×7 MSE，再query/task等权；不是带真动作latent的FM。
每臂80,000区间曝光，只是重复现有80视频，不称80,000独立示范。

0/250/500保存可恢复状态，250只供恢复，唯一结果点500。保存gamma、optimizer/scheduler/scaler（实际使用者）、
采样cursor/RNG、实际拓扑与schema；两臂恢复不能错开共同事件流。没有旧Reader/Writer/checkpoint初始化。

最终一次性对全部80 fit视频及96 diagnostic视频的完整合法区间保存三份实际预测：mu、F500、P500，以及标签/索引/mask。
逐视频先对区间/5位置等权，再episode等权、task等权；训练和读出权重同义。
报告all7、motion6、translation3、rotation3、gripper风险、逐task差额、全部不利视频/区间及gripper符号准确率。
按fit20新episode和内部留task4分别汇总；不得把同视频重叠区间当作独立统计样本。
task-cluster区间只作描述（20,000次、seed20261003）；四task区间尤其不能当作确定的广泛迁移结论。

主要可失败预测：内部留task4的P等task all7风险同时低于F和mu，至少3/4任务相对F改善；
完整查看效应大小、运动/夹爪和fit20结果，微小净正不自动支持下一笔完整方法投入。
如果只在fit20有效、内部留task失败，或者没有超越学习F/原生先验，降低并停止本固定native校准构造的投入。
不以另一局部指标、个别任务或换头/多训来挽救；若通过也没有自动LoRA、蒸馏、正式fresh或环境评测。
本项不会给合法LoRA学生记分，不读取Validation/Test，不做checkpoint选择或与MT闭环分数直接比较。

## 5. 资源、工程与停止

预计工程45–90分钟，native物化/学习/CPU读回合计25–50分钟，总70–140分钟；
依据已有frame-chunk64约19–20frame/s与小型缓存读出，不把这些估计当作本新consumer的实测吞吐。
硬限3完整GPUh、data1新增峰8GiB（包含工程/冻结树、H缓存、完整恢复点和临时文件）；失败/加载/profile均计入。
唯一新root：`/data1/user/ymdai/ember_runs/native_transition_action_calibration_20261003`。
开始前先从现有长度metadata核算精确frame/缓存峰，再在strg01核data1独立quota及共享容量；不复制source/dataset。

每次launch双节点live准入，按AGENTS当前总卡数及单节点上限；不因本诊断小就额外规定更低整批卡数。
native加载一次，按合法已有帧试64/128或受真实峰限制的最大frame chunk；缓存完成后释放无用source实例。
小读出优先打包两个独立gamma和同一批H，或按实际吞吐并行；每臂最多两次丢弃profile更新后恢复fresh初值/RNG。
验证最大合法物理batch（每臂160，不为填显存增加逻辑样本）；如同时打包两臂可实际比较，以吞吐选择。
不为占用更多卡复制数据或增加任务；GPU剩余显存的未放大依据、完整阶段吞吐和峰值须实报。

实验session独占工程及canonical tracked/Git窗口，在隔离codex worktree实施并自行完成比例适当验证、集成和推送；
native/学习/读回均使用clean pushed detached代码。复用真实native、source/data/tokenizer owner，不接入第二canonical Writer。
针对性验证覆盖数据/offset/信息墙、同初始化/同事件、source冻结、F/P函数包含、零输出初值和实际全区间消费者；
不新增防御性hash、逐tensor低位一致或与实现同义的大测试框架。
遇到已定位且不改变本科学合同的工程问题可在原预算修复、冻结新来源后继续；不能借修复改变特征/标签/目标/分组。

整批正常执行只等一次退出事件，不固定轮询日志。若执行session已直接等待子进程，不向自身Queue阶段通知。
全部完成后统一核退出/原件/费用，退役本诊断专用训练入口/模型/测试及临时接入，由Git与frozen树保留可复现来源。
向主讨论只回报一条整批结果（含全部正反/缺项、代码身份、成本、释放及canonical窗口），随后停止新增计算。
主讨论负责消费原件、修订机制和下一科学取舍；整体EMBER目标没有因本项结束而完成。

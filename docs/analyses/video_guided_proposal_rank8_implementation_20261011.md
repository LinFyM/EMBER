# rank8多起点Writer实施与消费者记录

正式科学依据是main fd5ae913的active design全文；旧八rank128教师ROOT只读，既有费用18.640412103864882GPUh作为新ROOT的carry-in。
实验01a12384-17a9-7382-aeec-cee89479428f已接回tracked/Git/docs/实现与运行窗口，main只读科学分析。本记录不把工程机制写成能力。

唯一`ember.proposal_writer`运行面承接38-target/rank8残差。`PolicyContexts`复用一份source policy，执行时绑定38个冻结MT合并target；
离开执行即恢复原source，教学meta在原source上读取，恢复专家在显式source_expert上下文使用原128。教师functional_call显式绑定合并target与完整任务A/B；
原生10步checkpoint每次重算重入对应权重上下文。meta的144个读取A/B由独立ParameterList持有，不进入任务导出。
ROOT/base/merged_targets.safetensors引用profile_01中只由冻结source+MT计算的81.125MiB缓存；没有装入profile学习值或复制第二份大source。

Λ0使用既有canonical identity helper的seed7、真实非零Kaiming A0/B0。643584有效坐标、10064块、304个rank组及512padding；
G/π的静态parent以Λ0居中。教师prox独立读取actual-A0/MT dense S_prox，使用实际D；S_G必须在完整16事件/selection bank后按task/event/q_T一次形成。
CFM从首个更新四事件均匀，保存240/480；旧320与学生刷新消费者已删除。fresh模型、完整teacher/ψ/θ optimizer/RNG/sampler/坐标/schema由新v2消费者保存。

seed160之后独立采H160，seed480之后独立采H480；mid/late从各自parent fresh优化，return保留H160 producer而parent回MT。
return引用mid已现场保存的同一原始H，不复制或重采成功轨迹。seed160先退出可立刻使H160及后继就绪，seed原optimizer续480；
所有16×160/480及各parent都保留selection4/audit16。q_T不读audit；全bank无合法标签或合法节点均无parent audit净增时停止G/π并交主讨论。

固定全部ψ后才local128/RL16；三head完整condition SUM、固定θ一batch一actor更新、独立baseline、forced score0及raw→fresh θ编码沿既有消费者。
RL教学按预注册循环交替；两48行及π-RL16两个指定完整U/other/wrong由既有队列消费，local128不追加U诊断。
队列支持同节点多worker的G/π job，逐卡live准入、总预算预留、实际卡数乘wall计费；单worker与多worker保持逻辑batch4/任务等权，未跨节点拼训练。

真实profile_01(gpu01:1)与profile_02(gpu02:7，高util共驻)全部exit0，总计0.083197897GPUh，包含两次micro112 OOM的测量开销。
MT128与缓存合并基底+rank8初值在相同实际own RGB/proprio/noise的原生前5动作RMSE为0.001242885；正常BF16误差不作bitwise承诺。
两组meta经rank8 CFM收到真实梯度、frozen source无grad；rank8非零A/B0副本的B梯度非零。teacher完整optimizer恢复实际续到第2步，ψ/meta完整optimizer恢复实际加载378个状态。
这些均为丢弃学习值的工程验证，尚无formal rank8教师、G/meta/π能力结论。

84帧、完整CFM读取chunk8/16为18.733/17.319秒，reserved17.957/25.623GiB，按余量用16；低余量可用8而保持梯度语义。
112-query micro28/56在快卡为9.962/9.822秒，在共驻卡为17.177/16.838秒；reserved约17/24.5GiB，micro112均OOM。
有27000MiB余量的教师用56，较小余量用28；function4比1/2快，快/共驻卡分别1.751/2.255秒。完整keep4+rec4的外侧预留约14/22.5秒而非demo秒数。
真实10ODE纯计算batch4/8/16/32为3.967/4.522/4.904/5.110观测每秒，reserved最高16.742GiB。
候选实际响应支持16观测，采用physical16；audit/report优先32 slots，低余量16。环境吞吐留待本来就规定的正式读出原件，未把重复观测profile冒称独立场景或实际环境吞吐。

结构在现有13个source owner内替换，删除旧refresh与旧重复profile流程，不新建rank8 runner/fallback或测试栈。
基线及当前architecture guard无hard violation；主要复杂度来自原配对分析、实际状态机和完整恢复/队列，保持各自原owner。
11项针对性工程检查通过；formal身份、精确命令、live资源、恢复和实际结果随后写入ROOT/progress及整批报告。

正式seed空H事件直接在CPU物化；只保存实际Λ0、空记录与无功能标签，不加载policy或占GPU。教师依赖图共48个有计算作用的job。
新增/替换750行、退役551行，净增长199行（13个既有source owner、没有新active模块），最大owner仍低于400行。
Batch只负责live准入、预算/退出和依赖调度；quota解析独立于设备准入，完整多卡在途估计乘实际物理卡数。
保留复杂度的函数分别拥有配对、教师优化、条件状态机与完整分阶段consumer，避免把当前职责切成重复runner。

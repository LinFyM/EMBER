# Task16公共控制参照：同一900参数点的四个固定执行情境

2026-10-04。接续机制§129的角色调用问题；只登记冻结辨识，不训练或选择模型。
实际授权与执行状态见[progress](../../progress.md)，推导见[机制§130](../analyses/feature_to_operator_mechanism_20260926.md#130-区分条件适配破坏已有角色控制与尚未补上共同控制缺口2026-10-04)。

## 1. 需要区分的实际解释

原task16八格中，W47四格搬orange；W40两格搬orange、一格搬ketchup、仅原scene46/noise46搬butter并入篮。
七失败均未获取正确目标。首规划已存在条件与自身情境的交互，却没有得到可跨这些情境使用的正确角色规则。
现有结果没有读过**同一个C900**的公共A0/B0在这四个情境中的行为。其它T/Context的公共分数不能代替它，
训练侧A28公共风险也不能回答当前held案例的对象控制。

令W_beta为C900公共有效权重，各适配层为W_source+B0 A0，其它层沿source；Delta(V)为视频引起的同形有效权重差。
当前八格只给出F(s,W_beta+Delta47,xi)与F(s,W_beta+Delta40,xi)。二者之差不确定它们相对F(s,W_beta,xi)的方向。
对各层，Delta(V)=B0 S+M A0+M S；去掉它必须同时恢复A0和B0，不能只去M或把S留在执行器中。
真实F包含所有层、十步flow及后续闭环，不能用固定hidden上的矩阵作用代替。

要区分的是：适配是否破坏了当前公共参数已经具有的正确角色控制，或公共参数也未取得该控制、视频只作了不足的修正。
公共A0/B0由完整目标共同学习，不是独立训练的MT、无视频最优解或知识的唯一分区。四行也不能唯一定位某个神经模块。
本比较服务于是否有依据把后继主修复定位为条件作用的破坏；不把证明video优于删减输入设为研究目标或晋级门槛。

## 2. 唯一新计算及只读比较

新增恰好四个full闭环：scene i∈{0,46} × 原noise stream k∈{0,46}，都只部署完整38-target的C900 A0/B0。
固定task16 exact language、官方butter入篮目标及原scene；无teacher输入或LoRA重新编译。
与已完成八格的每个同(i,k)下W47/W40逐一比较；原八行只读复用，不重跑，不增加环境smoke、其它场景或噪声。
四份初次50×7归一化动作及physical前5保存，与原八格作两套完整条件−公共的配对差，全部通道与原数组保留。
不新增内部activation、ROI、Jacobian、逐层/因子混合或参数插值。

每行保存原full双RGB、所有action chunks/实际actions、自身state及T+1全部7对象/EEF/quat/gripper/官方goal。
报告正确对象获取、错误对象运动及最终目标，保留移动但未成功、第三对象、扰动、反向效果。
物体中心、3cm或距离不是抓持/接触证明，不从早期动作差直接归因完整成功。

若公共在原失败情境正确获取butter、完整条件却转向其它对象，支持这些条件作用造成角色控制损害；
后继优先解释教学如何改变了一个已有正确的自身响应，不能先把它归为公共表征缺少能力。
若公共同样未获取butter，则不支持以“恢复这个现成公共控制”作为主要修复依据；
唯一完整成功仍是条件作用的有限正例，继续解释如何形成新的可迁移角色响应。
混合结果按四情境保留，不设置多数门槛，不因beta失败证明视频正确、因beta成功就选择public部署或缩小残差。
这些分支都不自动授权公共FM、正则、学习、完整400或其它controls。

## 3. 资产及消费者合同

公共参数唯一来源：
`/data1/user/ymdai/ember_runs/conditional_read_write_continuation900_20261002/conditional_read_write/train/attempts/continuation/checkpoints/macro_00000900/ecp.safetensors`。
main已读header：545个keys中76个`common.values.*`，实际names按`DirectLoRAParameters`的sorted完整LoRA名字映射。
复用`operator_writer.public_beta.public_state/factor_map`的实际映射/shape/finite消费者，不调用其中固定旧T1800的materialize/inspect合同。
原C900 bank的`shared.safetensors`只有38个A0，是provenance而不是本批完整公共LoRA；不得用它补零B或误当完整beta。
只导出一份约41.2MB完整公共因子，保留来源与映射，不复制799MB Writer、optimizer、source或整个bank。

其余资产与原[八格合同§3](task16_condition_context_crossover.md#3-精确资产与执行合同)完全相同：
source aligned1000（训练b8ea00e9）、C900训练/物化85919994、450父a0e0248d；新读取代码身份另列。
原八格有效consumer4d3762a3、root `task16_condition_context_crossover_20261004`、所有失败与原件只读保留。
scene为原`demonstration_transfer_learning_20260927/scenes/libero_object_task_06_state_000.npz`与`046.npz`。
完整post-dummy sim/controller/RGB恢复、256→224、双相机180度、自身state8/action7、10flow/前5/replan、horizon280/成功停保持。
noise使用原`policy_noise_seed(7,'libero_object',6,k,replan_index)`及CPU Generator/50×32，完整56次时钟按原函数；不得另抽seed。
记录真实physical_init_state_id与noise_stream_init_state_id；本批condition为public_beta900、teacher为空，不伪造teacher或formal K1映射。
旧八格metadata中的teacher只是旧条件来源，本批0teacher文件读取、0Writer/native编译、0梯度/held特权标签/Test/RL。

复用canonical scene/episode/noise/BatchedLoRAInference及完整因子安装。必要的专用scoped入口只接受这四行，退出还原；
不能放松formal evaluator守卫或冒充完整K1 paired400。检查实际beta映射/完整因子、scene和噪声承接即可，不逐bit追查正常差异。
已定位工程违约在独占分支修复、新push/freeze后继续同范围，原失败及费用保留；不因科学阴性追加case/重试。

## 4. 资源、执行和停止

新root `/data1/user/ymdai/ember_runs/task16_public_role_reference_20261004/`。
预计含工程30–60分钟、计算约.03–.1完整GPUh；硬限**.5完整GPUh、3GiB新增峰值**。
依据原八格有效127.97秒/.03555GPUh及约.66GiB full捕获，本批四行加41.2MB beta、工程/frozen/失败预计峰1.5GiB。
launch前按AGENTS现场核双节点/GPU总量8或6上限、data1独立quota/相关实际用量/共享空间；所有新增data1，旧大资产只读。
四合法case按真实吞吐同resident打包、source一次加载；无额外样本/profile填显存，不继承旧world为并发上限。

既有实验session接手后独占canonical tracked/Git，main只读；隔离工程分支、实际消费者核验、push与新clean detached运行。
正常运行由唯一退出等待者等整批结束，不轮询日志或给直接等待者自Queue。
四行完成后统一读取12行比较及全部正反、退役专用入口/hooks、封闭新执行并集成push；保留frozen/Git/原件/费用/退出。
只发一次有来源的完整回报并交回canonical窗口；无自动训练、公共目标、缩放、因子拆分、更多controls或新模型。

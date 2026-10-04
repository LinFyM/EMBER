# Task16角色调用：既存LoRA、初始场景与原噪声流的固定交叉

2026-10-04。固定数据内的冻结控制诊断；不训练、重新编译或选择模型。
机制依据见[主分析§128](../analyses/feature_to_operator_mechanism_20260926.md#128-回到实际条件控制器用原成功与错对象案例拆开参数场景和采样的作用2026-10-04)。
实际授权与状态只看[progress](../../progress.md)。上一项24-prefix已经结束，不追加其特征阶段、方向或ROI。

## 1. 同一未解决问题与具体缺口

当前条件读写900在task16“pick up the butter and place it in the basket”只成功1/50；48/50几乎不动butter，
全50条中39条将orange juice抬高超过3cm。§122的同LoRA对象换位又呈区域、错误实体及第三对象的混合响应。
§125没有取得精确角色坐标与额外控制，§127也没有为直接复制原生方向提供跨位置前提。
这些证据不能替代对**实际生成参数是否造成角色控制差异**的干预。

现有正式原件中，同任务有以下明确对比，main已读完整continuous和两份原始scene双RGB：

| 原init / 原teacher | 实际对象运动 | 官方结果 |
| --- | --- | --- |
| 0 / 47 | butter全程静止；orange最高抬23.848cm，154步首次超过3cm | 280步失败 |
| 46 / 40 | orange全程静止；butter最高抬38.831cm，154步首次超过3cm | 269步butter入篮成功 |

两场景butter初始中心差约.543cm、EEF差1.786cm，全部实体/机器人/相机与控制器仍按原scene恢复，
不是只改变这两个位置。原成功只有compact过程记录，不能把初始RGB说成成功过程视频。
两个原案例同时改变了教学生成的LoRA、物理初态与policy noise stream；已有对角记录不能区分三者。
成功案例选自已知原件，失败固定为既有主案例init0；这是解释性案例选择，不估计总体成功率或挑部署视频。

## 2. 唯一八行与真实干预

全部来自同一个C900 Writer、同一冻结source，保持exact language与官方task16物理目标。

```
scene i ∈ {0,46}
complete LoRA j ∈ {teacher47,teacher40}
noise stream k ∈ {original init0 stream, original init46 stream}
Y[i,j,k] = official closed loop from scene_i under W_j and stream_k
```

完整2×2×2共8行，每行只运行一次，四份不同scene/noise输入各用两套完整LoRA。
原对角为Y[0,47,0]与Y[46,40,46]；同(i,k)换j才是固定执行情境下的条件参数效应，
同(j,k)换i才是固定LoRA和噪声流的scene效应，同(i,j)换k只检验这两条既有噪声流的影响。
保留所有交互，不用主效应平均掩盖相反个例；两条噪声流不足以估计分布、鲁棒性或泛化成功率。

每套W均为原Writer一次产生的38-target完整`A0+S,B0+M`，不拼接层、平均因子、插值或重编译。
最终完整局部权重差为

`W40-W47 = B0(S40-S47) + (M40-M47)A0 + M40 S40-M47 S47`。

这表明干预同时保留视频产生的读取与写出及其交互，不能把结果先归给A、B或一个范数。
自身h依赖真实图像/state、flow latent、tau和前层LoRA；初次50×7动作由原十步flow整体产生，
闭环再由自己的观测更新。实际条件差必须由完整消费者测量，不能用同一固定h上的矩阵差冒充动作或成功效应。

记录每个(i,k)的完整初次action chunk、前5 normalized与physical动作的W40−W47差，按translation3/rotation3/gripper1分列。
同时保留同(j,k)的scene差、同(i,j)的noise差及全部原数组；量级仅描述函数变化，不称行为贡献比例。
不增加内部activation、Jacobian、ROI、探针或新的query forward。

## 3. 精确资产与执行合同

根前缀`/data1/user/ymdai/ember_runs/`，以下原件全部只读复用：

- C900原结果：`conditional_read_write_continuation900_20261002/conditional_read_write/evaluation/900/correct400/results.json`。
- 原bank manifest：同root `conditional_read_write/banks/900/manifest.json`。
- 两完整LoRA：同bank目录`task_16_demos_47.safetensors`与`task_16_demos_40.safetensors`。
  `condition_factors=complete_A0_plus_S_B0_plus_M`；shared A0文件只作provenance，不能重新叠加一次。
- 两scene位于`demonstration_transfer_learning_20260927/scenes/`，文件名分别为
  `libero_object_task_06_state_000.npz`与`libero_object_task_06_state_046.npz`。
  实际路径以原results各自scene_reference为准；复用完整body pose、post-dummy sim、controller与双RGB恢复。
- source aligned1000、normalization/tokenizer、原spec和suite资产由上述manifest/run contract解析，不复制大资产。
  source训练b8ea00e9，C900训练及物化85919994（450父a0e0248d），新评测代码身份单列。

noise stream按现有`policy_noise_seed(7,'libero_object',6,k,replan_index)`及canonical CPU Generator/50×32采样实现。
两个k固定为0/46。main已核它与原记录56/54项的已有前缀相符，完整56项保存在准备材料；init46早停后的最后两次replan仍由同一函数确定，
不重复最后噪声、不重新抽一个seed。flow十步内部语义、每5步replan与各case噪声时钟不改变。
必须显式区分physical_init_state_id、teacher/condition及noise_stream_init_state_id，不能通过伪造scene ID或视频mapping隐藏交叉。
正式50条无放回映射原件不改；本诊断两condition各复用4次、两scene各复用4次，明确不是正式K1 paired400。

render256/model224、双相机180度rotate、state8/action7、10 flow、执行前5、dummy settling10、成功即停、horizon280均保持。
始终从各自原post-settling scene进入第0次规划；不重放teacher action、不改对象位置/环境目标、重复settling或延长时限。
每行保存full双RGB、所有action chunks/实际actions、state、goal/continuous与原对象registry；不恢复held teacher动作、pose或标签。
没有Writer/native重新读取、训练/optimizer、held梯度、Test、RL、no-video/wrong/shuffle/reverse或其它策略臂。

两原对角必须报告是否重现原正确/错误对象及成功对比；普通数值差异不逐bit追查。
若合法消费者未复现，保留真实结果及差异，限定原事前对比不可识别；不换seed、teacher、dtype、batch或追加重试追回原成功。
若已定位为scene/adapter/noise等工程合同违约，可在同科学范围内修复、重新push/freeze，保留全部失败与费用；
原因不明、需改变科学计算或超预算时才报告具体边界。

## 4. 结果怎样改变同一问题的判断

- 若W40在固定scene/noise下能重复把W47的orange行为改为butter获取乃至入篮，
  则同一已学生成器确有可搬到这些失败情境的条件控制。优先解释两条正确教学为何编出不同角色作用；
  不能继续将这些案例全归因于source根本不能看见/控制butter，也不据此部署“最好视频”或held字典。
- 若成败/对象主要随scene或noise变化，两套LoRA在每个固定情境表现接近，
  则原唯一成功不能作为一套可稳定转移的“好条件LoRA”证据。降低仅重排条件参数、直接搬用成功LoRA或任务LoRA相加的依据；
  继续解释条件作用与自身反馈的结合，不把后继改成通用VLA降噪、seed扫描或单独解决MT失败。
- 若角色获取改善而放置失败、只在某一scene/noise有效，或出现第三对象/反向损害，保留逐行交互和效应大小；
  不将内部动作差、移动或抬高替代官方成功，也不强行归为单一模块。

这不是给任何新架构签发资格。§98的task32中途crossover曾显示成败主要随到达状态；
本次是task16从初始时刻的错误角色与唯一正确角色，另把原不同噪声流分开，不能将旧结论直接移植。
§122混合换位、§125坐标阴性、§127原生方向边界，以及T161/MT153完整参照仍保持。
八行结束即停，不追加scene、video、noise、时间点、层/因子拆分或训练/400；所有分支由main消费后另行裁决。

## 5. 资源、所有权与交付

新root：`/data1/user/ymdai/ember_runs/task16_condition_context_crossover_20261004/`。
预计含工程45–90分钟；计算预期.05–.3完整GPUh，硬限**1GPUh、4GiB新增峰值**。
依据此前48行对象换位含加载/捕获约.260GPUh；本批仅8行，但full RGB增加I/O，失败、加载和退出全部计费。
8份full RGB按每行旧88.4MB约.66GiB，另计continuous、输出、工程/frozen与失败，预计峰2GiB；现场核实际预算后launch。
所有新增data1；复用两41.2MB完整LoRA/source/dataset，不另复制400-bank或checkpoint。

既有实验session接手后独占canonical tracked/Git，在隔离分支实施最小专用入口，复用official scene/episode、
policy noise、BatchedLoRAInference及persistent cost-balanced消费者。若需scoped override，仅本固定panel生效并写真实交叉证据，
不能削弱formal evaluator配对/guard或复制另一套policy/evaluator。main接手前只登记与CPU原件读取。
新consumer检查真正用到的完整38因子、scene/state/noise和metadata；首个合法对角算入8行，不另加环境smoke矩阵。
从clean pushed detached冻结树运行，源冻结树不改；完成后退役临时入口/hooks，保留Git/frozen/科学与失败原件。

launch前live核gpu01/gpu02、本用户总占用、data1独立quota/相关用量/共享空间，遵守8/6卡与单节点6上限。
按8行真实吞吐使用有益设备及batch，不为填卡新增case或重复profile；不继承上一训练world size为并发上限。
正常运行只等完整退出事件，不读日志心跳、反复poll或让进程向直接等待者自Queue。

main准备材料在canonical `.codex/tmp/task16_condition_crossover/`；准入root后整体移入`analysis/preparation/`留一份。
CPU准备首次误将MuJoCo body名用作已存logical body_names，已按原件修正后保存；0模型/环境，不算科学失败。
交付完整8行/两原参照/初次动作固定差/全部对象运动及目标、source/code/命令、配对有效性、completion和完整成本/退出证据。
整批Git集成push后只发一次有来源回报、交回canonical窗口并停止；main负责科学分析与下一取舍。

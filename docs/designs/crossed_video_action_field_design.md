# 同状态交叉视频动作函数诊断

> 2026-09-24登记并完成关闭；15900原始预测已独立复核，裁决见findings§138。无训练或新部署成绩。

Study：`crossed_video_action_field_20260924`；spec：`configs/crossed_video_action_field_v1/experiment_spec.json`。
研究根：`/data0/user/ymdai/ember_runs/crossed_video_action_field_20260924`，由Sol核对quota后创建。

## 1. 本批要区分什么

findings§137已确认：Goal21固定C接手时，前25步只替换B的x/y使成功27→40，目标持续位移37→49、干扰物位移16→1；
只替换z确实抬高末端，却使成功27→19。Object14没有普遍同向收益，C的正确对象移动之后也有失败。
这是前段命令的局部因果作用，未证明视频内容如何造成该作用，亦未单独定位某个Writer模块。

下一步沿实际动作函数上溯：保持同一C@420模型，在相同query/噪声下交叉所有50条同task正确视频生成的现有LoRA。
将“具体正确视频带来的变化”与“正确视频池共有的C/B差异”分开，再用已封存wrong条件观察同一C模型的条件作用。
不再增添动作通道置换，不以单个更好视频选模型，不把language B当expert或理想动作标签。

H_shared：在这些早期query上，C相对B的水平动作差异主要是各正确视频共有的分量；更换具体正确视频不能移除这项均值差异。
H_condition：同状态下视频条件变化对水平动作有较大贡献，或显著改变其方向；需继续检查表示/编译中的视频作用。
H_interaction：视频的影响依赖当前执行状态，不能用一个视频特定的全局动作偏移概括。
三者可并存。小幅动作变化也可能跨过接触边界，因此功能分量大小不能直接等同闭环因果贡献大小。

## 2. 固定数据、模型与查询

- tasks为global14/21（Object suite/local4、Goal/local1），init states全部0..49。仍为已暴露的Train诊断任务。
- 固定Source1000、B_language@630、C_video_fm@420；原四臂commit43d801b1的模型、normalization、tokenizer不变。
- C正确条件使用原`materialization/C_video_fm_420_held/manifest.json`，每task完整50视频条件，不新生成LoRA。
  在每个query上使用该task全部50套adapter；这是明确登记的有限条件池重复读取，不是新的paired400部署评测。
- B和C_wrong参考使用原`B_language_630_held`及`C_video_fm_420_wrong`的每state固定条件映射。
  wrong仅是已封存条件的冻结诊断；不新增wrong配对、视频条件或shuffled/reversed，不反向选择原checkpoint。
- query来自原四臂C_correct自然rollout的固定前段：每task/state在settling10后，重放原已保存环境动作，
  只读控制步`0,10,20`的机器人观测。到20停止，不执行本批预测动作、不采集成功率。
  实际环境初态、seed和前20步来自原件；优先复用已经存在的真实full observations，否则通过canonical replay采集同状态图像。
  禁止把compact记录当成含RGB，也不能用教师episode的图像冒充执行query。
- 共`2×50×3=300`个query，每query50个C正确条件加B/Source/C_wrong各1，共15900次实际10-flow预测。
  同task两个不同state是不同query；同一个query下50视频严格共享观测、语言和噪声。
- global replan indices为`0,2,4`，policy_noise_seed仍按原root7、suite/task/state/index产生。
  官方render256/model224、双相机旋转、8维自身state、完整50-horizon/32维内部action和10 flow保持。
  原已执行前段是策略输出，不是新增expert标签；不读取任何新的teacher action/state/reward或官方Validation/Test。

## 3. 数学读出与解释范围

记q=(初态、控制步、固定实际观测、固定噪声)，v为同task的50条正确视频之一。
`A(q,v)`是C真实10-flow输出经官方反归一化后，前5动作的x/y控制命令；主读数进一步使用原OSC的clip/scale，单位为位置增量命令。
这不是实际EEF位移。完整50×7输出保留；原始环境输入及其它通道分组作次级读出，不能混合不同物理单位计算一个总范数。
令`b(q)`为同query的B输出、`m(q)=mean_v A(q,v)`，则有限池上有精确分解：

`T = mean_(q,v)||A(q,v)-b(q)||² = M + V`，
`M = mean_q||m(q)-b(q)||²`，`V = mean_(q,v)||A(q,v)-m(q)||²`。

各task和控制步分开计算，先对5×2坐标等权，再对50states/50videos等权；不能把一个q下50条件当独立初始化。
报告M、V、M−V、V/T（T=0则记undefined）、有符号x/y均值及逐state分布，不只报告单个比例。
这个代数分解用于区分函数差异来源，不把M命名为“语言分量”、V命名为“有益视频分量”，也不把B距离称动作误差。

每个固定task/time完整50×50矩阵还作两因素分解：
`A(q,v)=mu+r_q+c_v+i_qv`，行列效应分别去总体均值，交互双中心化。
报告video主效应c与query×video交互i的平方量；它们把V分开，但不能直接识别目标语义、位置复制或意图理解。
同task正确视频低变异可以是合理不变性；仍可强烈依赖这些视频共有的内容。因此低V不证明忽略视频。

附加同query读数：固定原C diagonal、原scheduled other对应的正确池项、固定C_wrong、B与Source的输出。
报告wrong变化相对B−m方向的投影（分母为0则undefined）以及真实分量；正投影只表示方向接近B，不是动作正确或视频有益。
从既有/重放的真实sim几何读取EEF→目标body中心方向，报告前5 x/y命令的方向投影；body中心不是抓取点，不构造新oracle标签。
这里只使用“未见该query前的完整教学内容”；没有数据依赖的teacher挑选、条件平均部署、adapter平均或闭环选优。

300个query实际属于100个固定初态。按task内50 states联合重采样20000次、seed20260924，三个time与所有条件同步重采样。
视频池视为固定完整池，不以50视频另乘独立N；区间只对所测固定task/模型/条件池描述，不外推training seed或全40tasks。
按预先定义分量检验强版本的H_shared/H_condition；分量相近或相位不一致时保留混合/未识别，不能设事后阈值宣布根因。

## 4. 实现与验收

Sol从最新main隔离开发，复用canonical policy loader、bank安装、preprocess、stateless noise及prefix replay。
一次性编排可留研究根launch目录；需要的保留源码仅做已有owner的有界扩展，不另造Writer或evaluator运行面。
旧`run_action_probe.py`只可参考实际10-flow推理调用，不能重跑其main或FunctionalQueryDataset，因它会读取已封存的expert actions。
所有15900预测来自同一个clean pushed detached实现提交；旧三个study/冻结树/原件只读。

1. 启动前锁定300-query manifest与每task50正确adapter清单、固定B/wrong映射；核对物化bank源模型和输入身份。
2. 所有source/adapter均eval、requires_grad=False且inference_mode；Source参考确保没有前一个B/C adapter残留。
   不训练Writer、不新物化、不改rank/scale/dtype/flow。使用已验证批处理提高吞吐，不为逐bit一致固定batch1。
3. 对重放query复用原8维state容差position/gripper1e−4、axis-angle1e−3；保存真实双相机观测、state、sim/controller/对象几何及原件路径。
   同q不同v必须实际消费同一输入与噪声；允许复用经现有接口确认不受adapter影响的frozen prefix，禁止缓存会随adapter改变的激活。
4. 先固定Goal21/state0/t0，50正确条件+B/Source/wrong共53次作pilot，计入15900。
   验证动作shape/finite、完整LoRA切换及Source清除、噪声语义、输入一致和正常批处理。
   C diagonal相对原保存预测、B在初态相对原行的差异单列；接受正常推理数值差异，仅调查有实际语义/数值故障证据的差异。
5. 保存每次full50×7环境动作、必要normalized输出、前5控制命令、全部条件/噪声/adapter/query元数据，保留诊断完整矩阵。
   查询图像一次保存引用，不为50视频重复落盘。保留source和B参照，不按任何数值删除query或视频。
6. 验收300唯一query、15000正确预测、900参照、pilot计入、所有worker退出、信息墙及完整矩阵；
   独立由原输出复算T=M+V及两因素分解，恒等式只验证分析实现，不作为科学通过标准。

本批没有完整闭环新证据，不报告修复通过，也不把模型间差值唯一归罪于参数化、视频编码或训练目标。
后继须结合本批函数定位、已验证的动作干预和历史证据，再决定上游模块/学习机制的针对性干预。

## 5. 资源、停止与交接

研究根新data0峰值≤4GiB（共享模型/banks只读、300query图像与矩阵输出），data1≤1GiB；建根前查strg01独立quota和共享容量。
最多4物理GPU，包含replay rendering与policy inference；每launch核对两节点live所有权及项目全局/单节点上限。
无DDP、无dummy占卡，不等待凑卡；profile按真实10-flow queries/s和显存确定batch。
正常长任务一次持续等退出/完成事件，不轮询cache/checkpoint或定期读日志。

身份、信息墙、重放、输入/噪声配对、adapter清除、shape/finite或资源合同失败时停止受影响执行并回报。
完整15900预测及登记分析完成后，主动Queue主讨论`01a0cd94-65da-7b22-8ca9-7ba35f454632`，给出原件和完成信号并停止新增实验。
不自动加训练、闭环、任务、视频条件、教师标签或参数扫描。具体执行者仍为Sol `01a0cd90-ebb7-77a1-a20b-a858825d2f66`。

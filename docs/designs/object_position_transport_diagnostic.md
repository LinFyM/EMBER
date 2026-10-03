# 自身对象位置交换诊断

2026-10-04，机制§121。辨识C900在task16的错误搬运更接近固定场景区域，还是对错误实体的可迁移跟踪。
这是冻结策略的有限输入干预，不是训练、数据扩张、官方评测、checkpoint选择或新Writer；实际状态只看progress。

## 1. 固定模型、任务及来源

只用global16 / libero_object:6，exact language为`pick up the butter and place it in the basket`。
原init0..7的teacher依次为47、33、28、46、32、1、24、43；两种布局及C/T共用原映射。
MT不接收teacher；保留同一场景与policy RNG字段用于配对，不称它使用了视频。

三个模型保持全部原值：

- C900：`/data1/user/ymdai/ember_runs/conditional_read_write_continuation900_20261002/conditional_read_write/`，
  原`banks/900/manifest.json`、`evaluation/900/correct400/run_contract.json`；学习身份85919994。
- T2340：`/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928/continuation2340/T/`，
  原`banks/2340/manifest.json`、`evaluation/2340/correct400/run_contract.json`；原合同记学习身份e2afbfd7。
- 强MT300：`/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928/stage1/MT/evaluation/correct400/run_contract.json`，
  其实际共享权重`/data0/user/ymdai/ember_runs/coverage_retraining_20260920/training/mtbc/checkpoints/step_00000300/lora.safetensors`只读复用。

原Source1000、normalization、tokenizer及assets沿以上实际合同，只读复用，不复制模型，不新增Writer/native forward或LoRA银行。
C为完整条件A/B，T为原shared与M的合法组合，MT为原共享LoRA；不能把三种bank格式混同或平均因素。
新运行身份为实验session集成后的clean pushed detached commit，和既存学习身份分列。

## 2. 唯一物理干预

原scene来自`/data1/user/ymdai/ember_runs/demonstration_transfer_learning_20260927/scenes/`，task16的八份post-dummy快照。
每条先按canonical scene消费者完整恢复；在首个policy query之前，只有交换臂执行一次以下变换：

1. 按实际对象/关节registry定位`butter_1`和`orange_juice_1`的free-joint根姿态，不按假定数组序号写入。
2. 交换两物体的世界水平x/y坐标。各自z、quaternion、速度、身份、资产、碰撞几何及其它对象/robot/controller状态保持。
3. 正常sim.forward并刷新真实双相机观测；不推进环境，不额外settling，不修改goal或自己的state8。
4. 保存交换前/后物理状态及实际body坐标，核交换语义、初始自身state与其余场景保持；同一交换初态供三个模型配对。

若实际层级坐标需要换算，以world x/y交换为合同而非简单猜qpos下标；若会造成初始明显穿透/悬空或无法维持上述条件，
报告具体几何/接口边界，不自行移动其它物体、加settling、改z、排除case或替换布局。只修复已定位且不改科学变换的工程问题。
原布局臂不作位置变换，也不额外推进。原模型body/scene文件绝不覆盖；新快照只进入本次root。

教学视频、exact language、生成LoRA、policy/env/root seed7与原绝对replan噪声索引在两臂配对。
rollout开始后不再交换/干预状态，机器人按原完整策略和真实物理闭环执行。
render256/model224、双相机180度rotate、state8/action7、10 flow、前5 actions后replan、原dummy10、horizon280与成功即停均保持。
交换发生在原dummy10后的快照上；不能再做10步导致额外状态差异。

## 3. 固定48行及被动读取

3个模型 × 2个布局 × 原init0..7，共48条新环境行；不按旧成功与否删case，也不增加其它task、teacher、init或位置幅度。
原布局同批运行，是为了同一消费者的行为及新增被动谓词配对；原历史24行结果并列保留，正常数值差异不触发逐bit追查或重复择优。
本面板只有8个原始物理条件，不能把48当独立task样本，也不称strict400、泛化分数或正式qualification。
本批新轨迹仅用于诊断，不得并入后续训练或补充数据集。

每个模型×布局的init0保存full双RGB/state8（共6 full），其余42 compact；全48保留完整50×7生成chunk、实际执行actions、
T+1 continuous的所有现存对象/EEF/gripper及官方goal轨迹。末段只记真实执行动作，不把未执行chunk当物理数据。
仅额外被动记录butter和orange各自对原basket contain_region的原生In谓词及首次/末态；不能用orange的谓词触发成功终止。
该标签不进入policy或Writer，没有任何梯度。源mask/asset信息不新增部署输入。

读回逐行报告原/换位成功、结束步、两对象位移/高度随时间及最大值、首次/末态In、EEF轨迹与全对象反例。
沿已有1cm位移/3cm抬高描述便于对照，并保留连续量；不将body中心/高度阈值称为抓持、接触或语义理解。
以真实6 full双RGB解释对应的个例；compact没有图像时不补造。不能只汇报换位后成功变多。
原布局对历史及各布局模型间的R/G/L/churn完整列出；这是单任务选择性面板，不作总体性能区间或科学排名。

## 4. 事前预测与停止线

机制§121推导的两种主预测：原orange位置在换位后放着butter；跟固定区域会转向该butter，跟orange身份则应转向原butter位置的orange。
结合两物体实际运动、EEF接近及RGB判断，不能只由末态成功或单个attention数值强行归类。
不同实例可能混合、转选第三个物体、未接触或失败；全部保留，允许结果不足以区分两解释。
T/MT作为已知强策略参照，用于判断观察是否共有；它们失败不降低对EMBER问题的关注，也不成为停止分析的理由。

- 持续走向旧区域并移动butter：支持该案例的位置依赖，降低纯错误身份跟踪的解释；不证明某层或教师表示是根因。
- 跨位置继续跟随orange：支持错误实体跟踪，降低纯区域习惯解释；后继应解释教学到角色选择的联系，不能再用缺少状态敏感性绕开它。
- 混合、整体受损或不操作两对象：保留未区分；不把换布局的分布变化称工程故障，也不扩大面板寻找清晰结果。

所有分支到本批48行及一次完整读回为止，不自动追加对象/方向/位移/层/强度扫描、训练、grounding辅助、视频controls、Test或RL。
本项没有改进方法，不选C/T/MT或宣称根因已找到；下一科学判断由main结合实际原件与完整历史形成。

## 5. 执行、预算与交付

唯一新增root为`/data1/user/ymdai/ember_runs/object_position_transport_20261004/`，所有新增文件/工程冻结树在data1；旧data0权重只读。
预计含工程2–3小时，硬限2完整GPUh、12GiB新增峰值；预计输出和工程合计4–6GiB，不复制任何大型公共资产。
依据是旧48行图像选择诊断约.1855GPUh和400行约1.25GPUh；本批不增加donor forward，但仍把加载/验证/渲染/I/O/失败/退出全部计入。
若到1.5GPUh或显著超期后预计不能在硬限内完成，报告剩余工作与原因；不能通过删行、少计成本或扩大预算继续。

launch前核两节点live GPU、当前/启动后物理卡数、适用8/6合计及单节点6上限，核data1独立user quota/相关用量/峰值与shared容量。
按实际吞吐选择并行与真实batch，复用cost-balanced dynamic queue/persistent workers及source/prefix缓存，不因三模型就锁死3卡，也不dummy占卡。
只有需要校验同一已登记初态的物理变换/捕获接口时才作最小消费者检查，不额外扩模型推理或环境面板；成本并入本批。
接口核验聚焦free-joint/world XY、初态配对、真实刷新、teacher/LoRA路由和仅butter终止，无hash/全树审计或通用测试框架。

既有实验session独占canonical tracked/Git窗口，从最新main隔离实现，集成push后使用clean detached frozen消费者。
沿canonical场景恢复与评测owner作必要接入，不新增常驻evaluator/Writer；运行结束退役本项临时入口/干预hooks，Git/frozen/完整原件保留。
正常运行等待实际子进程退出，不周期查日志或发送自通知；整批完成/异常只回报main一次，并标明来源。
完成物理与科学读回、48原行/6 full/42 compact/比较、completion/完整费用/退出与释放证据、Git推送后停止并交回窗口。
main不重复执行者工程审查或测试，负责原件消费、竞争解释和后继；无自动新批。

## 6. 实际执行交付（2026-10-04）

固定48行/6 full/42 compact与旧24同行已完整读回；C旧/本批原/换位0/0/2，T1/1/2，MT2/1/1。
全部混合/负例、初态XY语义与官方butter终止检查保留；不存在一致分组或部署性能主张，科学解释由main直接消费。
实际48读取clean pushed detached dac0ad57fb2e10b86ecfc9224f6bb2af208554c1；物理/首次前加载失败8a2c8496、模型来源保持。
完整GPUh.259836593305/2、新root+工程阶段观察高水1.297657GiB/准入保守6/硬12（非连续精确峰）；全部进程exit、GPU释放。
原件在唯一root的completion/readback及analysis/report/rows/per_case/RGB/行为/成本/资源、evaluation全48捕获与launch。
专用入口/模块/五hooks完成退役，canonical拒绝该归档合同；Git和两frozen保存重现原件。整批Git/一次Queue交付后交回窗口并停止，无自动后继。

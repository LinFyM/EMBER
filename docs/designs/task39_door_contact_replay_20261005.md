# task39原动作重放：炉门碰撞的有界因果辨识

2026-10-05。这是冻结已有策略输出后的物理诊断，不是新Writer、训练或EMBER评测成绩。
是否实际执行只看progress。依据findings§330–332；主讨论形成科学合同，实验session负责工程与运行闭环。

## 1. 具体问题与决策价值

T2340/C900各50条task39都抬起正确白黄杯，却没有一次In或Close。
main根据正式XML直接附于固定炉体的heating_region、每条自身scene的初始body姿态和原In谓词，
在0环境/模型计算下重建炉腔目标box。全部100条杯中心始终位于前边界外；
到目标box最近距离中位数T为11.875cm、C为7.806cm；最近处同时在左边界外为41/50、29/50。
这不是由body原点替代炉腔的描述。两个有full RGB的init0在门/左前侧互动，不能据此给其余98条标注接触。

竞争解释：

- 原动作序列有把杯向内输送的作用，但实际炉门接触阻断了运动，策略没有产生有效的绕行/纠偏。
- 即使解除炉门的物理阻挡，原动作序列仍不能把杯送进目标region；门接触不是这项失败的充分解释。

本批只改变炉门碰撞这一物理因素，保持已保存命令，辨别上述有限问题。
它不读取模型内部意图，不能证明语义已懂/未懂、source与Writer谁是根因，亦不区分其它障碍、抓持或控制误差。
正结果提高研究接触条件与路径反馈的优先级；明确阴性降低“炉门挡住了原有正确搬运”这一解释的优先级。
不从任一结果自动恢复Reader、阶段辅助、位置oracle或新训练；仍须回到完整LoRA机制判断。

## 2. 固定面板、输入与两臂

读取以下原results中global39/libero_10 local9、init0..49全部100行，保留原checkpoint身份、teacher、noise及原件路径：

- `/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928/continuation2340/T/evaluation/2340/correct400/`
- `/data1/user/ymdai/ember_runs/conditional_read_write_continuation900_20261002/conditional_read_write/evaluation/900/correct400/`

复用每行continuous中实际执行的520×7动作和scene_reference中**自身**完整model body pose、sim/controller初态。
不载入策略权重或Writer，不编译LoRA，不重新采样动作；不读取held teacher actions/state/pose/reward、Test或时序controls。
只读基于这些原件的main_cavity_readback.json作为路径/结果参照，不把临时文件当唯一来源。

每行两个CPU物理消费者，合计固定200条重放：

1. `recorded_parent`：原场景、原炉门碰撞、原OSC/频率/物理参数，逐个执行已保存动作。
2. `door_noncolliding`：从同一初态重新开始，只将microwave门的可动child body及其后代的碰撞geom关闭contact；
   炉体、炉腔壁/底、桌面、robot、杯及其它物体碰撞保持。门的几何、质量/惯量、关节、初始角度与阻尼保持。

取消contact须作用于原collision geoms的contype/conaffinity或等价精确实现；记录实体/geom映射和修改前后字段。
不能移动、移除、隐藏或固定门，不能同时取消炉体/robot碰撞，不能改动作、控制器、关节限位、gravity或horizon。
原资产与原件只读，修改只属于新的运行实例。无渲染消费者；不能因helper默认加载相机而使用GPU。
仍保留原520动作；本诊断不因In出现提前结束，不延长horizon，也不把无法由无接触门实现的Close当主判据。

## 3. 重放可信度与被动记录

先用固定T/init0与C/init0的parent验证可重放性及真实CPU吞吐；这些行属于200，不重复计科学样本。
起点body/EEF/gripper与原scene对应；全过程对原continuous给出body/EEF轨迹偏差，保留真实最大值及谓词一致性。
不用bitwise要求。主要解释要求每行mug与EEF全过程最大位置差均不超过1cm，且In/Close原false序列保持。
这条1cm是相对本批约1.5–26cm区域距离的解释容限，不代表所有接触相位精确复现。
若起始两行不满足，先定位明确的恢复/控制器消费者差异；允许在同合同/预算内修复。
若原件缺少必要状态、原因不确定或只能靠改物理/命令达标，回报具体缺口，不扩大容限或伪造历史状态。
后续所有行报告同一偏差口径；未达解释容限的行完整保留并标记，不能选择性删掉或当成有效配对。
若连续出现系统性重放失败，停止受影响计算，按真实边界回报，不能靠增加重复次数选匹配的一次。

每条保存T+1自身body/EEF/gripper、门关节角、原生In/Close、实际命令，及每个控制步末的接触摘要：
白黄杯—门、robot—门、杯—robot、杯—炉体、杯—桌面；记录真实geom pairs、contact距离及可直接取得的力。
没有取得力就明确缺项，不以接触存在冒充抓持或阻挡强度。控制步末接触未覆盖所有积分substep，必须标注。
不得改变integrator/control频率补采样；不保存无界全contact大数组或额外图像。
可在两臂中以相同被动机制记录首次目标接近、接触与区域进入的时序，不让记录影响物理计算。

## 4. 预定读数与解释边界

主读数：两臂各模型50行的In-ever及新增/丢失；mug到真实native In box的最小/最终距离，首次In步数、
最近点分轴缺口、门接触起止/样本数。全部逐行公开，并列原模型轨迹、正常重放和无门碰撞重放。
保留近/远和晚取杯反例，不只展示被碰撞解除改善的个例。

数学对象是`x_(t+1)=F_c(x_t,u_t_recorded)`，c只有门碰撞两种设置。
在反事实状态分叉后，仍使用原轨迹上得到的命令；这是固定命令的物理因果作用，**不是原policy的反事实闭环能力**。
因此即使In增加，也不能说合法策略已能应对新观测，或它已学会正确目标区域；穿过原门几何只是一项分析机会。
若没有In增加且距离/输送也无实质改善，只降低此物理阻挡解释，不据此确诊视频语义或任意神经模块。
若效果只在一部分行成立，报告范围与代价，不强行形成单一根因。无额外强度、物体、障碍或checkpoint扫描。

## 5. 资源、交付与停止

唯一root：`/data1/user/ymdai/ember_runs/task39_door_contact_replay_20261005/`。
预期工程与运行合计60–120分钟；依据仅有100×520步原命令、两臂CPU模拟，无模型加载/推理。
前两行实测后更新完整ETA；这是估计，不能把未profile的CPU物理吞吐当已测数值。
硬限0 GPU/CUDA、16 CPU-hours、2.5小时wall-clock、新增峰1GiB（含源码冻结树、失败与临时产物）。
可使用最多8个CPU worker，按实际吞吐和内存选并行，不固定串行或占用GPU等待。
建根前核data1独立quota、个人实占和共享容量；所有cache/temp也在data1，复用原assets/scene/动作。
若无GPU的物理路径不可用、预计无法在预算内完成或需要改变科学语义，回报main具体边界；不能自动借GPU或扩量。

实验session独占工程分支与main集成窗口，复用既有场景/控制/被动采集owner，不新建长期第二evaluator。
运行代码来自clean pushed detached；原冻结树不改。工程窄修自行闭环，保留来源与失败成本；main不重复代码测试/工程验收。
交付原始连续记录、逐行配对/偏差/接触/region读数、简短科学报告、来源、完整命令、退出与成本；不复制策略或大型资产。
整批完成或真实科学/预算边界只通知main一次；无中途常规通知/heartbeat/自Queue。完成后停止新环境计算，
退役本批专用运行面，保留Git/frozen/原件，集成push并交回写窗口。没有自动后继模型、学习、闭环或物理干预。

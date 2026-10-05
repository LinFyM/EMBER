# task23已有执行的抽屉状态重放

2026-10-06。依据findings§341、349–350。只补齐已有执行中缺失的物理状态，不是新策略评测、
因果干预、训练或新Writer；实际承接与运行状态只看progress。

## 1. 问题与会改变的判断

task23为`open the top drawer and put the bowl inside`。原T2340/C900各50条分别0/1成功，
碗全程位置完全不变49/38条；现有continuous只有柜体根body，没有三层抽屉关节。
main已实际查看init0/teacher30及init38/teacher16两条完整合法教学的stride5双RGB：都展示先操作抽屉、再取碗。
T/C init0各60次已存双RGB显示长时间在柜体附近；C后段有明显抽屉运动，不能只凭图像确定哪层及开度。
因此不能把“碗没动”直接归为已开柜后的阶段切换失败，也不能把所有行归为没拉开。

本项区分：正确上层尚未获得足够开度、实际操作了其它层、上层已拉开但未取得碗，以及混合/仍不明确。
若主要停在上层操作前，后继应先解释该具体接触控制为何未由完整教学形成，不能优先改为阶段切换目标；
若已有较早上层打开而长期不取碗，则加强研究已发生状态变化如何调用后续作用的依据。
这只是定位实际最早缺口，不据物理记录唯一定位native、地址、Value、loss或架构；没有自动后继方案。

## 2. 唯一面板与信息范围

固定global23 / libero_goal local3，读取以下原results中init0..49全部100行：

- T2340：`/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928/continuation2340/T/evaluation/2340/correct400/`。
- C900：`/data1/user/ymdai/ember_runs/conditional_read_write_continuation900_20261002/conditional_read_write/evaluation/900/correct400/`。

只消费每行原continuous已执行actions、原scene_reference的自身完整sim/controller/model body pose，
以及其原结果/teacher/seed来源记录；不加载policy/Writer/LoRA，不重新预测动作。
不读取任何held教学actions/state/pose/reward/terminal、Test或其它视频；main已有RGB只是科学背景。
所有原始模型、场景、资产和执行原件只读。task39旧重放的场景恢复和CPU物理消费者可复用，不能恢复其旧批次。

每行只有一次**原物理、原命令重放**，不是两臂：完整恢复原post-dummy状态后，逐步执行原动作至其实际记录长度。
不再settling、不改goal、物体、关节初值、碰撞、控制器、物理频率、动作或horizon，不因新被动Open记录提前终止。
公式为`s[t+1]=F(s[t],u_recorded[t])`；这项没有改变F或u，新增的是对同一过程的状态读取。
不初始化相机/render，不使用GPU/CUDA；不得为补图像另起消费者或生成新policy rollout。

## 3. 重放有效性与实际记录

固定T/init0和C/init0先校验原消费者；两行计入100，不为验证另增科学行。
沿已验证场景恢复/OSC接口，报告原与重放的EEF及全部对象body轨迹最大位置差、夹爪差和完整原生In序列。
解释容限为EEF与碗全过程最大位置差各≤1cm，原生In完整序列相同；其余body/夹爪差不隐藏。
这是相对厘米尺度动作/物体差别的解释容限，不代表精确接触重现；不用逐bit或新增hash验证。
若首两行不满足，允许定位并修复确定的恢复/接口错误；原因不明、需改变物理或扩大容限时回报main。
后续100行均报告同一口径；失败行不删、不重抽、不重复择优。系统性失败时停受影响计算并说明实际缺口。

每条保存原长度T+1的：

- 原body/EEF/gripper、实际执行动作及原生`In(bowl,top_region)`。
- wooden_cabinet三层真实关节名称、qpos/qvel及各自原生Open；按asset/site registry解析，不能猜数组下标。
- 控制步末robot与柜体固定部分、top/middle/bottom可动body及其后代的contact摘要，保留实际geom配对/距离；
  能直接取得contact力则保留，不能取得则明确缺项。只采控制步末，不改integrator或补积分substep扫描。

保留真实asset/body/joint/geom映射，避免把柜体根body不动误作抽屉未动。
当前WoodenCabinet原生Open为qpos小于−.14，原范围[-.16,-.14]；执行者核实际消费者，并同时保留连续开度。
**官方task23目标仅为In，未要求Open谓词。** 因而Open为描述量，不是成功必要条件或新增gate；
部分开度也可能足以放碗，C900/init38的原成功必须保留，不能因阈值把它重新标成失败。
接触存在/力大小亦不直接称抓持、阻挡或策略意图。

## 4. 固定读回

逐行列出来源、原/重放成功和长度、配对偏差、三层初/末/最小qpos、Open首次/末态/总时长，
碗首次可测移动及3cm抬高、最大位移/高度、首次In、EEF实际轨迹和接触时间范围。
沿既有1cm位移/3cm抬高口径描述，并保留连续量；不得把阈值当动作阶段真值。
按模型汇总全部50行，交叉列出上层开度/其它层操作/碗运动，不用单个init代表全体。
对T/C init0与C900/init38逐时解释；不能用compact缺失的RGB声称看过画面，亦不补造教师动作标签。

全部正反例和重放不符行保留；本项不是400成绩、checkpoint选择、模型资格或改进方案。
任一结果都到本批100条原命令及一次完整读回为止，无自动初态开柜、碰撞反事实、语言替换、阶段/姿态辅助、
新Reader/Writer学习、重训、数据扩充、RL或Test。后继由main结合这些事实与已有功能证据重新判断。

## 5. 资源、执行和交付

唯一root：`/data1/user/ymdai/ember_runs/task23_drawer_state_replay_20261006/`。
预计含工程45–90分钟，依据既有task39 CPU重放接口与本次最多100×300控制步；首两行实测后核剩余ETA。
硬限从实际承接起2小时wall-clock、16完整CPU-hours、0 GPU/CUDA、新增峰1GiB；失败/加载/冻结树/临时均计入。
最多8个CPU worker，按实际吞吐、可用CPU与内存调度；不靠常驻轮询或保留空进程占资源。
建根前核strg01 data1独立user quota、个人实占及共享容量，复用大资产，不在data0新增文件。

实验session接管canonical tracked/Git独占窗口，从最新main隔离实现；工程、实际消费者检查、明确接口修复、
集成push及clean detached frozen运行自行闭环。科学范围、原件有效性或预算需改变才回主讨论裁决。
源代码冻结后不原地改；必要修复用新的clean pushed版本并保留旧失败/有效原件与完整计费。
交付原始连续记录、逐行/aggregate/接触映射、简短报告、source/scene/命令/退出/费用及completion，
完成后退役仅本批专用入口/hooks，保留Git/frozen/原件，推送并交回窗口。
正常运行等待退出事件，不重复轮询日志；整批完成或实际边界时只向主讨论回报一次，无阶段自通知或自动下一批。

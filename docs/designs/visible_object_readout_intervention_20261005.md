# 可见物体读取重分配：冻结策略的因果辨识

2026-10-05，主讨论登记。当前接手和运行状态以progress为准。
这是带自身场景特权分割的分析控制器，不是合法部署Writer、训练方法或正式成绩。

## 1. 要回答的问题及最近似证据

task16的C900八格有六条搬orange_juice、一条搬ketchup、一条butter入篮；T2340也有原布局成功、换位后搬错物体的案例。
已有G在交换布局的八条中butter角色rho均占优却全部失败；rho是跨层/head/槽的相对密度，并不规定实际有多少视觉Value进入控制。
旧self_image_attention_transfer在三个训练任务间移植另一个策略的图像分布，确有有限获取转移，但供体本身没有提供正确对象读取保证。
角色/crop学习改变了多个参数和读取量，没有在冻结策略上直接设置下面这个变量。因此不把旧阴性抹去，也不把本项说成首次研究attention。

本项只问：**在原完整LoRA与冻结prefix下，把分给非目标可搬物体的实际attention质量转给可见butter，能否修复具体的目标获取和入篮？**
竞争解释是现有视觉Value与后续控制不能有效利用这次重分配，或原来的物体读取分配并非这些失败的主要可修复环节。
正确路由、错误orange_juice路由和原策略共用同一机制与固定面板，避免只看一个受强外部提示的新控制器。
正结果只能证明此实际接口具有有限修复作用；尚不解释合法视频如何学会产生它，不自动取得完整重训资格。

36-task支持审计已经排除“训练没有角色变化”与“butter从未为正”的简单解释。
主FM有五组完整BDDL布局相同、目标不同的任务；旧八任务角色辅助没有覆盖任何一组内的两个目标。
原seen144中C450/C900在95/96/97各为4/4，butter→tray已会；C900六个basket训练任务合计20/24。
这不证明真实RGB/state匹配，也不把未见组合直接命名为根因；本项不因此扩充数据或重跑角色辅助。

## 2. 唯一实际干预

在当前自身环境、真实RGB对应时刻，以可见visual geom的分割得到每个实体在两相机512个image patches上的覆盖比例f_o(p)。
它是patch内可见像素比例，非归一后的对象概率；0≤f_o≤1，各可搬实体覆盖之和≤1。
沿已验证的180度rotate、256→224、16×16 patch映射和body/geom registry；robot、basket及其它fixture/container承载物不属于被转移的实体集合。
食品包装/瓶等被操控实体属于集合。使用真实可见部分，遮挡后不得补不可见轮廓。
既有q标签已做对象内归一，不能直接当f或恢复不存在的面积。复用既有分割/registry机制，读取当前sim的分割，不恢复或读取held teacher state。
分割仅在自身每次replan读取当前画面时生成并供该次十步flow共用；无额外env.step、settling或状态改变。

记当前层/head/action-slot的原完整attention概率为a，I为实际有效双相机image token集合。
指定接收实体r，D为除r外的可搬实体，定义：

```
d(p) = sum_(o in D) f_o(p)
t_r(p) = f_r(p) / sum_(u in I) f_r(u)
m = sum_(p in I) a(p) d(p)
a_new(p) = a(p) (1-d(p)) + m t_r(p)   for p in I
a_new(j) = a(j)                       for j outside I
```

若r当前完全不可见，原样使用a并记录此条件；不添加可见性阈值、虚构mask或选择相机。
该式保持概率和、image总质量及本层当前输入下的非image概率；真实输出变化为
`delta_y = sum_(p in I) (a_new(p)-a(p)) V_image(p)`。
它改变真正进入o_proj/残差/后续MLP的Value混合；后层hidden和后续flow/action由完整原策略继续计算。
物体共享patch、上下文已混入prefix Value，故这不是精确的对象特征分离或“完美语义绑定”，不得如此命名。
也不保证动作方向、抓持或完整成功。必须用实际行为判读。

全部18个Action Expert层、8个heads、50个action slots、10个flow时刻及所有replan应用同一规则。
prefix、LoRA、action_in/out、自己的RGB/state、正常精度与noise均不修改；没有第二策略、donor前向或Writer重读。
不选层/head/时段，不加强度系数、不扫温度或覆盖阈值。复用旧真实attention干预owner，不复制policy/evaluator。

三臂固定为parent、route_butter、route_orange_juice。后两臂分别以butter_1及orange_juice_1为r。
只有butter官方In决定成功；orange In被动记录，错误路由不会改变任务目标或终止条件。
分割与实体映射只存在于此分析消费者，不能进入deployment Writer、训练输入或后续参数选择。

## 3. 固定12个case，每臂12条，共36条新闭环

1. C900：复用task16_condition_context_crossover_20261004的完整八格。
   scene0/46 × teacher47/40 × noise stream0/46；均原布局。复用原完整bank、scene和绝对replan noise函数。
2. T2340：复用object_position_transport_20261004中init2和init4，各原布局/XY交换布局，共四格。
   teacher分别28/32，noise沿各原init；复用已经验证的两对象XY变换及原场景，不重新选择位置。

全部global16/libero_object:6，exact language保持；不读Test、held actions或teacher state/pose/reward。
不新编译LoRA、不更新任何参数。T强参照可避免把C特有现象当EMBER共同问题，但此小面板不估计整体性能。
同批parent重放作为干预参照；旧原件并列保留，正常数值分叉不追逐重现、不择seed或补跑高值。
官方render256/model224、双相机rotate、state8/action7、10flow/执行前5、dummy10后原scene恢复、horizon280及成功即停保持。

36条均保存full双RGB、全部50×7计划、实际physical actions、T+1 body/EEF/quat/gripper及目标谓词。
保存紧凑的逐replan/flow/layer物体attention质量、转移m、接收物可见性及实际delta_y量；不存巨大全head/token矩阵。
首个query的分割叠图和实际image token映射用于消费者核验，不另加模型/环境面板。
物体位移、抬高与中心距离按既有原件描述，不能替代接触、抓持或官方In。

## 4. 事前判读及停止

- 若butter路由在多个原失败格改变错误获取并完成正确入篮，错误orange路由呈对应的错误实体作用，且保留T原正例，
  支持该实际读取分配是一个可修复的因果环节。仍须另行说明合法条件如何学会它及全任务能力，不能把oracle成绩当EMBER提升。
- 若仅attention质量、delta_y或butter运动改变而完整入篮不改善，只承认部分获取作用；不据此自动添加同类loss/模块。
- 若两种路由都主要破坏原正例，或干预因不可见/原物体质量很低而作用微弱，明确其识别限制；不据此宣判全部视觉绑定无用。
- 若真实Value混合被改变仍未修复行为，降低这一具体重分配作为主修复的依据。保留后续控制与相互作用未知，停止本批。

逐case报告parent/正确/错误路由的取得对象、完整过程、成功集合R/G/L/churn；不以总数抵消相反效果。
结果不会自动启动训练、扩大面板、换干预强度、加prefix targets或再做一轮更强oracle。
本项不是对video必要性、训练收敛或全部task16失败的完整裁决，主讨论须结合旧正负证据再独立取舍。

## 5. 执行、预算及交接

唯一root：/data1/user/ymdai/ember_runs/visible_object_readout_intervention_20261005/。
预计含工程与读取2–4小时；硬限2完整GPUh、8GiB新增峰，包括核验、失败、模型加载和所有36条。
依据旧48条位置交换约.26GPUh/64分钟及旧attention移植已存在消费者；本批不增加donor前向，但新增在线分割。
若工程超过4小时、资源预测越界或现有接口不能忠实实现，报告具体剩余与原因，不静默换成rho/坐标读出或缩面板。
执行者核data1独立quota、相关实占/共享空间与新增峰；每launch现场同时核gpu01/02并遵循现行总卡数、吞吐和NUMA规则。
复用source/原bank/scene，cost-balanced persistent workers；按真实显存和吞吐安排，不为本小批占满卡数，不增加GPU特例。

推送本合同后实验session独占tracked工程/Git，main不并发写科研文件。
在独占分支实现、按实际source/attention/分割消费者作必要核验，push后使用clean detached冻结运行，不改旧frozen或旧原件。
小型核验确认概率非负/归一、非图像保留、当前RGB与分割一致、mask无目标时identity，以及hook移除后原消费者恢复；不做逐bit审计。
已定位接口错误在本科学/资源范围内由实验session闭环修复并保留失败与成本；改变科学语义或超预算才回报边界。
完整原件、completion、资源释放及代码集成后退役专用入口/hooks，保存Git/frozen和科学记录；不留平行可部署策略。
用一次持续等待进程退出；只在整批完成/异常给main一条带来源回报，不分阶段Queue或反复轮询。

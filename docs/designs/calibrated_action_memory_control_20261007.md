# 冻结视频动作校准的输出，是否已经可用于控制

2026-10-07。Owner持续自主授权；由progress登记后为唯一active分析批次。
这是原36个训练任务上的冻结功能诊断，0训练更新，不恢复ControlCalibrated、G/F或关系编译的学习。

## 1. 具体问题与最近历史

上一批同一特权几何1-NN下，真实教学动作91/144、同帧source十步预测0/144；
真实动作完成task25/29/32/38各3/4、4/4、3/4、3/4，而T为0/4、2/4、1/4、0/4。
同时demo对T总数−18，target净+1是得19失18，support净−19；不能把这项有覆盖正例直接当强完整教师。
它建立了在给定对应下实际动作Value的价值，没有证明合法视频读取已经取得同样可用的内容。

最近完整ControlCalibrated恰有可复用的已学函数Gamma：用action-hidden教学转移预测5×7动作，
真实q同时接受动作辅助与最终LoRA FM，并直接调制A/B写入Value；不是独立头与真实消费者脱节。
其八teacher q误差.139893→.083044，98.977%改善来自五步均值；完整450为122/400、seen77/144，
弱于同龄C450137/91。该配方已关闭，不能由新NN阳性自动恢复它、换loss或增加训练。

这里问一个尚未由MSE或原整体成绩回答的问题：**原冻结Gamma产出的动作，给定与上一批相同的几何调用，
是否已经比它自身的裸端点mu0形成更好的闭环控制？**
若仍不能控制，不能说该方法已获得可用动作、只差写入；若可以，则保留一项有实际控制后果的合法视频Value获取正例，
但原LoRA利用、自身对应及完整held迁移仍未建立，不能归因唯一Compiler故障或自动进入新Writer训练。

LocalActionGrounded局部动作学习、NativeCorrection/LocalField直接监督实际参数/场、旧联合Reader及双域actual-A坐标
的完整正负证据继续有效（findings§173、§286、§304、§328、§377）；不把它们统称没接动作信用或没试自身状态。
本批只是读取同一个已学Gamma在另一明确特权消费者中的行为，不创造新校准函数、控制器变体或更强训练教师。

## 2. 匹配两臂与唯一干预

固定原ControlCalibrated450 checkpoint的全部八个`gamma.*`张量，保持原`ControlCalibration.controls`计算：

```text
H0_i, mu0_i：原裸Source1000、RGB/exact L、无State、probe1729/tau1的已有缓存
mu0_i = (epsilon1729 − velocity0_i)[:5,:7]
delta_i = Gamma450(H0_i,H0_(i+5))[:5,:7]
bare_endpoint 的归一化Value_i = mu0_i
calibrated_value 的归一化Value_i = mu0_i + delta_i
raw_Value_i = 原冻结source action反归一化(Value_i)

每次自身replan：同一live GT query → 上批原D_abs/1-NN → 一个raw_Value_i → 执行5步
```

Gamma参数不更新，裸特征不重算。两臂具有相同缓存/teacher/key/归一化/初始scene和调用规则，
唯一干预是加入或删除该冻结Gamma残差。自身后续轨迹及索引分叉是动作干预的结果。
Gamma同时包含已学参数、departure与arrival context；两臂差额**不单独识别arrival或时序的必要性**，
不把更好控制直接称为已证实的动态理解。本批不加shuffle/reverse、新辅助项或动作拟合。

上批source_action有真实teacher8state和完整10步ODE，本批mu0无State且是tau1端点；
故上批0/144不能代替本批bare_endpoint，必须实际完成匹配裸端点臂。
同样，q的35维在旧Writer内调制Value，本批直接作动作；效果不能倒推旧q已被有效使用或未使用。

## 3. 唯一输入、checkpoint与已有缓存

原checkpoint：`/data1/user/ymdai/ember_runs/control_calibrated_read_write_20261003/`
`control_calibrated_read_write/train/attempts/fresh/checkpoints/macro_00000450/ecp.safetensors`。
只加载其八个Gamma张量到原数学模块，不加载整个Writer、公共beta、38头、optimizer、source VLA或task experts。
保留训练来源2c630fb3及新reader来源；不选其它checkpoint或由本批分数改选点。
当前canonical `src/ember/operator_writer/control_calibration.py`为同一数学定义；使用原FP32参数与正常BF16 autocast，
RMS eps1e-6、四头attention及完整50 native位置都保留。不得只读前5 hidden而改变attention。

裸特征只读原root `bare_native/taskXXX_demoYY.pt`，仅H0、mu0、frame_indices、schema和来源字段。
main已核原144条件共4556个合法departure都有i和i+5，144文件合计988,168,960 bytes；全部八个Gamma张量仍在。
这是metadata/帧覆盖核对，未运行Gamma或新source。原cache不复制，不从旧`control_labels`读值，不扩缓存至其它teacher。
若实际source identity/schema/必需帧不符，按具体边界回报；不能另跑source、换frame、补标签或重新训练来补齐。

几何、条件与数据墙直接引用上批root `/data1/user/ymdai/ember_runs/privileged_action_memory_control_20261007/`
的`memory/registration.json`及已有geometry NPZ：原36 train24+support12、每task原init32…35、seed20260928、
同一144条teacher，合法i=0,5,...且i+5<N，4556个key。OOI全部body/site、对象等权、D_abs、0.10m/0.04m、
最早argmin、live缓存EEF/gripper与sim p/R均原样；不恢复几何标签、改点集或加phase/阈值/平滑/残差/相对坐标/K。

Gamma条件输入只有原视频/exact language派生的H0/mu0；真实action/state/OOI不得进入Gamma。
own GT与teacher对象对应仍在NN调用侧，因此两臂完整控制器**仍是特权诊断，不能称合法EMBER**。
这些teacher均属于Gamma原训练36×50池，阳性也不是新video或新task泛化证明；自身初态与teacher不同的边界保持。
没有held/Test、梯度、LoRA生成、teacher挑选、第二adapter或部署优化。

新cache保存两臂实际归一化/raw `N×5×7`及原frame/source/Gamma身份，不改变控制chunk。
若通用capture必须50提案，仅把第5个预测动作复制到未执行尾部，valid mask严格只有前5为真；
不得把这些padding称为完整50步预测。实际执行始终只读5个真实预测值，动作单位/clip/夹爪约定沿canonical消费者。

## 4. 固定闭环与完整判断

bare_endpoint与calibrated_value各原seen144，共**288条新episode**；0训练、0新source/native forward。
固定原scene/env RNG/teacher ordinal与未消费policy RNG元数据；自身NN无模型/采样/flow，不能写成10-step policy。
沿原dummy10、success即停、各suite既定horizon（含原support90合同）、前5动作replan、双RGB采集和persistent动态队列。
source与Writer在这些rollout中均不加载，Gamma也仅物化时运行一次，闭环直接读取固定Value。

主比较calibrated_value对bare_endpoint，完整报告36任务、各suite、target24/support12、breadth、R/G/L/churn/J。
同时对上批demo91成功的91行与其53失败行作预登记分层描述，仍运行全部144、不筛条件；
该分层不是新的选择标准或因果中介结论，目的是说明已知可用动作带来的控制有多少被学得Value保留。
上批demo91/source0与旧完整ControlCalibrated77、T109、MT93、G84、F66只引用原行作有边界的背景，不重跑。
MT原行应使用`operator_seen_task_diagnosis_20260929/attempts/scene_canonical144/MT/evaluation/correct144/results.json`，
上一背景记录漏中间目录的字符串由main记录更正，原分数/有效行未变。

用上批已保留的demo动作cache仅在分析侧报告两新Value的同memory5×7归一化MSE/平移/旋转/夹爪，
不得把真实值接入Gamma输入、做回归校正或据误差选择case/参数。内部改善必须与完整控制分开判断。
保留全部实际动作/提案与valid mask、检索索引/距离、原生谓词、scene/teacher/RNG和Value来源。
每臂每task init32 full，其余compact，共72full/216compact；固定tasks12/25/29/32/38/73的init32两臂，
12clip各8个已拍时刻双RGB（192图），复用原捕获/可视化方式，不补终态render、不从图片或中心位移发明contact/grasp。

- 若Gamma使多个task获得真实控制、并覆盖上一批真实动作可完成的相当部分条件，则加强“此已学视频读出具有可调用控制内容”的支持。
  仍须交代与demo的损失/增益、原target缺口和support退化；不凭一个成功或胜裸端点就称强教师。
- 若误差下降而闭环仍弱，降低旧Gamma获取足够控制精度的判断，不再沿它加loss、续训、调幅或NN机制变体。
- 若两臂相近或裸端点更好，说明本接口没有显示已学残差的有益贡献；保留全正反例及直接动作消费不同于旧调制的范围。

任何分支均不自动授权新F/Reader、Writer fresh、LoRA接口修改、more-seed/400/controls/Test或新的局部探针链。
main收到后结合完整历史与本原件承担方法取舍；最终目标仍是合法视频生成的一套LoRA稳定大幅超过强MT。

## 5. 资源与工程生命周期

唯一root `/data1/user/ymdai/ember_runs/calibrated_action_memory_control_20261007`，新增只在data1。
预计45–90分钟；硬实际承接起3wall-hours、1完整GPU-hour、16GiB新增峰，包含工程/失败/加载/profile/Γ/EGL/冻结/临时/分析。
依据上批288NN读出合计约487秒/一张物理卡逐臂计时、7.980GiB全部新增，以及本批可复用144裸cache且只运行小Gamma；
是估计，不保证新工程时间。2wallh或.75GPUh时若剩余难在硬限完成，一次报具体边界，不减面板或暗增预算。

实验session接手后独占tracked/Git，先strg01独立data1 quota、个人用量/shared及逐GPU启动双节点live准入；
沿总8/空闲≤10时6、单节点6，真实EGL占用也计费。不为小Gamma加载整source或dummy占卡，已就绪的两臂读出可独立并行。
GPU Gamma物化保持原正常BF16、先按现有余量选择可装下的较大transition batch；最多两次首既定transition块吞吐profile，
保留第一次有效值、全计成本。不能因内存低就维持过小batch，亦不为满显存扩大科学case或新增大模型。
小Gamma物化后立即释放，NN actor仍CPU；首合法episode检查计入288，无额外smoke环境。

独占codex worktree实现、真实消费者检查、main集成/push、新clean detached冻结；旧frozen/cache/raw不热改。
复用原Gamma数学函数及上批冻结几何/NN/队列/capture，只加本批typed Value与必要薄注册，不复制第二评测器或大trainer。
接口检查聚焦Gamma参数/完整50hidden输入、i/i+5、反归一化一次、first5与mask、信息墙、相同几何/scene及恢复，
不加hash/全树扫描/逐tensor低位一致检查。两个新臂不能误装上批source10-step缓存或把mu0当已有0分结果。
专用缓存/入口/临时注册由本批持有，整批完成退役；保留Git/frozen/新Value/raw/失败/费用与原旧资产。

原范围工程故障自主隔离修复、新push/freeze并复用有效行，不等main重复工程验收；科学输入、算子、面板或预算变化回main裁决。
只在整批完成或真实边界一次可靠回报main并交回canonical tracked/Git，不逐阶段Queue、心跳或自通知。

# 冻结动作读出与内部适配的因果分解

> 2026-09-24登记；仅冻结计算干预，不训练、不选择部署方法。执行状态与实际接收方见progress。

Study：`readout_realization_causality_20260924`。
规格：`configs/readout_realization_causality_v1/experiment_spec.json`。
拟建根：`/data0/user/ymdai/ember_runs/readout_realization_causality_20260924`，Sol核对quota后创建。

## 1. 为什么检验这个接口

findings§135–138区分了三件事：Goal21存在早期水平命令的局部因果作用；C的正确视频池内函数变化远小于C/B共有差异；
Source原有Goal40/50能力在C中变为25/50，而Object从Source0/50变为C46/50。不能把统一减弱适配当修复。
低同task视频方差可以是合理不变性，wrong仍改变函数；尚未证明视频被忽略或Writer不能生成有效参数。

下一项检验一个有精确计算边界的竞争解释：共同动作变化主要由最终动作读出直接产生，还是由进入读出前的隐藏表示改变产生，
或者两者的配合同时负责能力获得与损失。这里不重复Writer Core/Procedure donor交换，也不扫18个层或LoRA尺度。
最后一层是唯一从1024维Action Expert表示到32维flow velocity的线性接口；其余37个LoRA一起作为上游上下文。
这是参数作用位置的因果定位；即使定位成功，也仍须追查学习为何形成这种参数作用。

历史限制：旧模型的Core/Procedure功能正例、当前C600关闭Procedure仍有缺口、末P边界修正的匹配阴性、
私有Writer不输自由A/B、任务梯度隔离未恢复能力均保留（findings§119/127–129）。它们不支持直接删P、扩head、
默认梯度冲突或先训练公共底座。当前CPU读出算子检查中，C action_out增量在source读出行空间中的真实7维Frobenius能量
仅约2.27%；这不是激活加权的功能分量，也不支持把增量概括成简单全局动作倍率。不得按矩阵norm直接裁决。

## 2. 四格计算与可反驳预测

固定C_video_fm@420原correct bank；R表示除`model.action_out_proj`外全部37个LoRA，O表示这一最终读出LoRA。

| 条件 | R | O | 含义 |
| --- | --- | --- | --- |
| 00 | off | off | Source1000，所有LoRA增量为零 |
| 01 | off | on | 仅保留C原读出增量 |
| 10 | on | off | 保留C内部适配，读出恢复source |
| 11 | on | on | 完整原C |

off仅将对应LoRA B置零，A、其余组和所有source权重不变；alpha/rank不改。每个条件仍是完整38-target状态，
不能把B_language的模块移入C、混合两个训练模型的隐藏坐标或平均视频LoRA。00可复用canonical无adapter Source执行。
11复用原C bank；01/10由原bank确定性派生并登记逐条件provenance，不运行Writer或读取新teacher标签。

以下预测分别判断，允许混合和未识别：

- H_O：读出在完整C背景下直接贡献Goal的不利动作变化。移除O的实际动作变化应朝既有B−C方向投影为正，
  且Goal闭环`S10−S11>0`；只满足函数方向不算行为机制通过。Object的`S10−S00`检验内部适配能否独立保留新能力。
  要进一步声称无损修复，还必须检验Object的`S10−S11`，不能把有部分新能力说成完整保持。
- H_R：不利作用主要在内部适配；Goal的10仍低于00，而01更接近00的行为。结果须联系实际函数和配对成功集合，
  不能仅用某个模块参数变化量支持此解释。
- H_coupled：单独保留任一组不能重现Object的获得，或Goal的损害只在联合时出现；交互不能忽略。
  这将反对把某一组普遍关闭作为修复。无显著区间不等于两组等价或没有作用。

对同query q、相同初始噪声，令`Fij(q)`为真实10-step flow输出，`hij,k`为第k步进入读出的隐藏表示。
`vij,k = W0 hij,k + b0 + j ΔWo hij,k`，`dt=−0.1`，`Fij=epsilon+sum_k dt*vij,k`。
因而在规范化动作空间上，

`F11−F10 = sum_k dt*ΔWo h11,k + sum_k dt*W0(h11,k−h10,k)`。

第一项是沿完整C真实计算轨迹的直接读出作用；第二项包含改变读出后经后续flow状态反馈造成的隐藏表示变化。
它们相加是这次冻结干预的总作用，不能把只算第一项当成完整反事实。source bias在差中抵消。
01−00作相同分解；四格函数交互为`F11−F10−F01+F00`。它不等于两个组训练梯度的交互。
FP32正常舍入残差单列；不提高dtype或重复模型forward追逐逐bit一致。

同query的B−C不是前批各自自然轨迹上保存命令之差。若只有闭环收益而未兑现函数方向预测，
登记冻结读出干预的经验性收益，不宣称已经解释前批B前段救援或全部共有函数差异。

## 3. 固定查询与预测（1200次）

直接复用crossed_video_action_field study的300个真实query NPZ和query_index：任务14/21、states0..49、控制步0/10/20，
均来自原C自然rollout；不重新重放采图，不读取expert或held action标签。每query仅用原state对应的correct teacher，
四格共享相同query/语言/噪声；与上一批全交叉50视频不同，本批不再扩大视频矩阵。

每query四次真实10-flow，共1200次；pilot为Goal21/state0/step0四次，计入总数。原Source/B/C预测只作封存参照，
四格主比较均来自本次同一运行合同。00/11与上一批Source/C diagonal的差异单列，正常数值差异不要求消失。

复用canonical预处理、loader、BatchedLoRAInference、stateless RNG；root7、replan0/2/4、full H50、内部32D/实际7D。
保存每query每格full50×7 normalized/environment、first5 OSC和十步的实际readout base/total velocity（full50×32）。
不必存1024维完整hidden。观察hook不得修改forward输出，必须分清physical PEFT identity、per-sample delta及hook顺序；
记录的base应为同一次、同一实际hidden上的W0h+b0，total为该步实际消费的输出。
不得用重新执行一次source网络的hidden替代当前hidden。00/10的读出增量应为零；01/11的实际增量应与原action_out一致。

机械验收：每条件恰好10个真实readout事件、规范化输出满足Euler重建与上述总作用分解，max abs residual≤1e−4。
这只核验观察/分析实现，不是科学通过线。若失败先检查hook位置、维度、是否record了真实velocity和flow调用次数，
不以精度小扫隐藏缺项。pilot观察启用/关闭的模型输出在正常推理容差内一致；额外观察验收调用单列，不冒充正式1200。

主要函数读出使用前5×xy的实际OSC clipped/scaled命令：四格配对差、有符号分量、变化平方量、交互和
`10−11`朝上一批同query `B−11`方向的投影（分母0则undefined）。B是参照，不是expert；朝B变化对Object不自动有益。
直接/反馈分解首先在规范化空间报告，xy可按冻结反归一化仿射尺度转换为未限幅环境命令；
不能把拆分项分别clip后声称仍为精确加法分解，不能把不同物理单位合成一个控制范数。

## 4. 闭环（400条）与固定分析

机械pilot通过后，无论函数结果倾向哪项假设，都完成四格×两任务×50初态=400条配对闭环。
重新从原初态执行完整策略；不是前段命令交换。四格一直保持各自冻结LoRA，不切换策略、不执行离线分解项。
固定原correct state–teacher映射，每task四格各用50正确teacher一次；00不消费teacher，仍保留配对键。
所有官方预处理、dummy10、10flow、前5replan、RNG7、成功即止及Object280/Goal300 horizon保持。
这只是冻结诊断，不是新部署方法成绩、strict paired400资格评测或正式Validation/Test。

闭环pilot固定两任务state0×四格=8条，计入400；states0/25共16条双相机full，其余compact。
保留动作、state、stage predicates、对象/EEF/夹爪连续轨迹、已有控制步接触读数和全部worker回执；不新增人工成功判据。
00/11相对原四臂的成功集合变化单列，主比较限本次四格内部；不能把历史源模型成绩直接代入新的配对差。

每task报告四格成功数、Source(00)相对R/G/L、breadth/任务边界，及六个预定差：10−11、01−00、10−00、11−01、11−00、10−01；
另报告成功率交互`11−10−01+00`。每个差保存逐state配对和R/G/L，不能只报告最好的组合。
task内50states联合bootstrap20000/seed20260924；函数三个time及闭环四格使用同一组state draws。
固定两任务/一个训练seed的描述性95%区间，不作全40任务或跨seed推断，也不按这些区间选超参。
几何/接触/持续位移只辅助解释结果，不自动称抓取或语义理解。

## 5. 实现、资源与停止

Sol从最新main隔离开发；复用canonical预测/evaluator、bank provenance与capture，必要时提取已有预测helper供两个真实调用方共享。
不复制一个完整预测器、不修改四个旧formal树或原件、不另造Writer运行面。一次性编排留study launch；结构变更按code-architecture-gate。
正式1200预测与400闭环来自同一个clean pushed detached实现提交；全eval/inference_mode、source及adapter均无梯度。

新data0峰值≤6GiB，data1≤1GiB，复用source、query与原banks；00/11不复制原资产，01/10预计约1GiB派生banks。
建根前查strg01对应独立quota、实际用量和共享容量；最多4物理GPU，每launch双节点live准入及项目全局/单节点限制不变。
不为凑卡等待、不dummy占卡；普通长任务持续等退出/完成事件，不固定轮询日志/cache。

身份/信息墙、确定性参数组遮罩、source clearing、query/RNG配对、10-flow计数、观察分解、shape/finite或资源合同失败时
停止受影响执行并回报。仅bad科学读数不构成工程失败；不增层扫描、尺度扫描、训练、教师条件或评测任务。
完成后由Sol `01a0cd90-ebb7-77a1-a20b-a858825d2f66`主动Queue主讨论`01a0cd94-65da-7b22-8ca9-7ba35f454632`，
提供原始矩阵、四格全部行/对比、分解与数值余项、case索引/worker退出/缺项，然后停止新增实验。主讨论核对和裁决。

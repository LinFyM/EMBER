# 固定真实incoming的一步经验与控制诊断（2026-10-10交付）

## 1. 完整结果与它改变的判断

按[唯一合同](../designs/parameter_edit_credit_diagnostic_20261009.md)完成8原condition的E0、三臂72闭环、
576次新增完整十步和672次单time FM；没有pilot、重跑挑分、新学习、适应、held或Test。
真实最后incoming I、原end E+、相同I/V/L下empty-E输出E0各 **13/24**，breadth均4/4。
相同总数包含实质能力交换：I→E+为R11/G2/L2；E0→E+为R8/G5/L5。
本轮没有取得总体净改善，也没有把经验判为统一无用。E帮助开抽屉及一条Long2保持，
同时丢失Spatial、Object、另一条Long2及Goal中的成功。Spatial43在最后编辑前已经全面失败；
Spatial29和Long2/44则确实发生最后一步损失，且专家full50 FM更好。

因此降低三项充分性判断：只停止成功后的编辑、统一删除经验、继续原样FM/递推曝光，都不能解释并解决当前所有不足。
加强“专家query功能信用与实际参数修订效用仍有缺口”的解释，但没有识别唯一编码层或唯一loss根因，
也没有证明某个保持项、RL或架构替换可修复。main须消费正反原件后另作完整后继决定；本诊断不自动开启学习。
原完整χ360 **130/400 vs MT153**、Long1晚期改善及旧T的视频正证据保持原样。
这里是已见结果定向选择的训练面板，不能作held总体估计、strict400资格、独立确认或视频必要性结论。

## 2. 资产、真实参数身份和信息来源

新ROOT：`/data1/user/ymdai/ember_runs/parameter_edit_credit_diagnostic_20261009`。
父ROOT：`/data1/user/ymdai/ember_runs/parameter_conditioned_compiler_20261009`，全程只读。
χ固定原`training/checkpoints/macro_00000360`。`run_contract.json.conditions`逐项记录exact language、
最后event、完整incoming/E+/E0路径、原experience、state32/33/34及RNG；没有使用旧null替代E0。

| 原condition | 原J / 结束 | 本次I | 原最后episode | I成功states | E+成功states | E0成功states |
|---|---|---|---:|---|---|---|
| Spatial0 / teacher43 | 5 / 预算 | incoming_004 | 4 | ∅ | ∅ | ∅ |
| Spatial0 / teacher29 | 1 / 成功 | 强MT | 0 | 32,33,34 | 32,34 | 32,33,34 |
| Object3 / teacher16 | 1 / 成功 | 强MT | 0 | 32,34 | 32,34 | 32,33,34 |
| Object3 / teacher4 | 1 / 成功 | 强MT | 0 | 32,34 | 32,34 | 32,33,34 |
| Goal0 / teacher15 | 2 / 成功 | incoming_001 | 1 | 32 | 32,34 | ∅ |
| Goal0 / teacher6 | 2 / 成功 | incoming_001 | 1 | 32 | 32,33 | 34 |
| Long2 / teacher16 | 1 / 成功 | 强MT | 0 | 32,34 | 32,34 | 32 |
| Long2 / teacher44 | 2 / 预算 | incoming_001 | 1 | 32,34 | 34 | 32,34 |

E+是原实际end权重，不重新计算后择优；只新物化8套E0。E0使用相同实际I而不是自身递推的null，
仍继承I包含的历史和原编辑时点。empty E返回学习过的16个empty slots，仍经过教学读取和参数编辑，**E0不等于不编辑I**。
source/MT、normalization、tokenizer、38-target/rank128及原生prefix冻结；每臂只执行一套完整LoRA，缩放1。
teacher只读双RGB和exact language；原自身动作/反馈用于冻结诊断及E+原历史，授权train的非teacher动作只供FM读出。
被动环境对象/predicate仅用于解释行为，不进入Compiler；无optimizer或梯度、无动作轨迹拟合。

三臂复用父`evaluation/train360/evaluation_contract.json.environment_contract`的完整scene；
actual consumer确认合同完整相等。官方render256/model224、双相机rotate180、state8/action7、10-flow、前5后replan、
dummy10、220/280/300/520 horizon、成功即停和原env-policy RNG保持。没有用普通reset替代scene恢复。
72份新行均保留；实际双RGB/proprio、noise、50×7归一化输出/真实命令、执行mask、终止以及T+1被动记录完整。
`analysis/final_facts.json`核验全部72条的decision/执行数/被动时钟及672 FM的非teacher来源。

## 3. 成功集合、覆盖及与历史重算的差别

| 配对 | reference→candidate | R/G/L | churn | Jaccard |
|---|---:|---|---|---:|
| I→E+ | 13→13 | 11/2/2 | 4/24 =16.67% | 11/15 =0.7333 |
| E0→E+ | 13→13 | 8/5/5 | 10/24 =41.67% | 8/18 =0.4444 |
| I→E0 | 13→13 | 10/3/3 | 6/24 =25% | 10/16 =0.6250 |

本面板每suite仅一个task，per-suite与per-task相同，不把4/4覆盖外推为8-task breadth。

| task / suite | I / E+ / E0（各6） | I→E+ R/G/L | churn / Jaccard | E0→E+ R/G/L | churn / Jaccard |
|---|---|---|---|---|---|
| 0 / Spatial | 3 / 2 / 3 | 2/0/1 | 1/6 / .6667 | 2/0/1 | 1/6 / .6667 |
| 13 / Object | 4 / 4 / 6 | 4/0/0 | 0/6 / 1 | 4/0/2 | 2/6 / .6667 |
| 20 / Goal | 2 / 4 / 1 | 2/2/0 | 2/6 / .5 | 0/4/1 | 5/6 / 0 |
| 32 / Long | 4 / 3 / 3 | 3/0/1 | 1/6 / .75 | 2/1/1 | 2/6 / .5 |

| condition | I→E+ R/G/L；churn/3；Jaccard | E0→E+ R/G/L；churn/3；Jaccard |
|---|---|---|
| Spatial43 | 0/0/0；0/3；空集未定义 | 0/0/0；0/3；空集未定义 |
| Spatial29 | 2/0/1；1/3；.6667 | 2/0/1；1/3；.6667 |
| Object16 | 2/0/0；0/3；1 | 2/0/1；1/3；.6667 |
| Object4 | 2/0/0；0/3；1 | 2/0/1；1/3；.6667 |
| Goal15 | 1/1/0；1/3；.5 | 0/2/0；2/3；0 |
| Goal6 | 1/1/0；1/3；.5 | 0/2/1；3/3；0 |
| Long16 | 2/0/0；0/3；1 | 1/1/0；1/3；.5 |
| Long44 | 1/0/1；1/3；.5 | 1/0/1；1/3；.5 |

E相对E0新增：Goal15 states32/34、Goal6 states32/33、Long16 state34。
E相对E0丢失：Spatial29 state33、Object16/4各state33、Goal6 state34、Long44 state32。
I→E+的两增是Goal15 state34和Goal6 state33；两损是Spatial29 state33和Long44 state32。

事后分组仅描述原事件，不能解释J的因果效应或新增固定轮数：

| 原组 | conditions / rows | I→E+ / R,G,L | E0→E+ / R,G,L |
|---|---|---|---|
| J1 | 4 / 12 | 9→8 / 8,0,1 | 10→8 / 7,1,3 |
| J≥2 | 4 / 12 | 4→5 / 3,2,1 | 3→5 / 1,4,2 |
| 原实践成功 | 6 / 18 | 11→12 / 10,2,1 | 11→12 / 7,5,4 |
| 原预算停止 | 2 / 6 | 2→1 / 1,0,1 | 2→1 / 1,0,1 |

原panel的MT/end/null总数是13/11/12，本次I/E+/E0是13/13/13。
新E+相对原end存在6条成功翻转：Object16/4的state33从成功变失败；Goal15的states32/34和Goal6的states32/33从失败变成功。
原end与新E+数差+2，原件没有替换、反复复现或择分。I多轮不等于MT，E0也不等于旧null。
计算批量、设备和数值顺序允许正常差异，但这6条翻转意味着不能把本次小面板当作稳定重复资格，
更不能用新13改写旧11或完整130；本轮配对结论只由本次完整72行计算。

## 4. 实际行为：最后一步、既有失败和E的正负作用

全8teacher/最后实践的双相机board、24个三臂board和72个MP4索引在
`analysis/behavior/index.json`及`index.md`。短片使用真实replan前帧（至多每5control一步）及末帧、4fps，
不是每个仿真子步的视频；I/E+ board使用共同6点，E0独立6点，同索引不保证同操作阶段。
以下step均为settling后的实际control步；物体名字以被动body registry为准，而非仅凭图像颜色猜测。

**Spatial43：损失在最后编辑前已经存在。** I/E+/E0在三初态都失败，目标碗的位置、最大位移和高度在保存精度上均未改变。
最后实践board同样显示手离开目标区，而teacher完成拾取与放置。最后一次编辑未恢复控制，也没有新增二值损失。
原历史MT在这3行全成功，故原MT→end损失不能全部归因于最后一步；本轮没有定位先前哪次编辑首先失效，
也没有把预算失败自身动作当正确标签。这一condition的专家FM还进一步变差，支持降低原样继续编辑的预期。

**Spatial29/state33：最后一步确实损失，E0恢复成功。** I在132步首次满足碗on plate，E0在77步满足，E+至220步仍不满足。
三者都移动目标碗；最大抬升分别约0.110/0.078/0.077m（I/E+/E0），不符合“完全没找对目标”的统一解释。
E+末碗中心约(0.077,0.228,0.911)，盘中心(0.077,0.194,0.903)，图中碗在盘边；E0末碗(0.077,0.222,0.917)，
盘(0.075,0.194,0.904)，predicate为真。记录支持放置关系/末端几何与时序存在差别，
没有接触力、物体旋转或grasp约束原件，不能指定唯一释放/接触原因。
这是原实践成功后编辑且full50及front5专家FM均更优、实际新初态成功却丢失的反例。

**Object3：同为失败可以是不同操作。** 两teacher的E+都保留I的32/34，却均丢失E0新增的state33。
teacher4/state33的I明显抬起非目标绿色瓶；registry确认是`salad_dressing_1`，最大位移0.292m、抬升0.181m，
目标bbq_sauce仅抬升0.0066m。E+转向目标bbq_sauce，目标最大抬升0.125m，但最终落回桌面，未进basket；
E0把目标抬升0.213m并转移到basket，116步predicate成立。
teacher16/state33的E+则已把bbq移到basket附近，最大位移0.421m、抬升0.204m，同时basket移动0.039m，最终contain仍假；
E0在116步成功。因而E+并非所有失败都只停在目标识别，目标接近、提起、转移和放置须逐段解释；
二值失败相等不能把I/E+的功能变化写成零。

**Goal0：E有实际收益，也有反例。** 两teacher的E+共4/6，对I2/6、E01/6。
teacher15/state32的I/E+在112/118步开drawer，E0不打开；state34仅E+在148步开drawer，I/E0失败。
teacher6/state33仅E+在268步完成，state32同样恢复了E0缺失的成功。
但teacher6/state34反向：E0在108步打开，I/E+都失败，board显示I/E+长时间在drawer前偏低区而未完成拉开。
这是两个真实成功实践的non-MT incoming，不是从MT的单步变动；4增/1损不支持把当前E入口统一关掉，
也不支持声明它在开抽屉类任务上单调有效。被动trace保存open predicate，未保存drawer内部joint位移，
不能用cabinet主body静止误判drawer没动。

**Long2：E既保持成功，也破坏第二阶段。** teacher16/state34，I/E+完成开炉与pot on stove，
E0只开炉、未完成pot转移；E+的两个predicate首次成立90/248步，对I122/233。
teacher44/state32，I/E+/E0开炉均完成，首次99/103/103步；故本条损失不能解释为主要没有开炉。
I/E0随后在254/268步满足pot on stove，E+至520步未完成。pot最大抬升I约0.146m、E+0.025m、E00.097m；
E+还使非目标frypan移动0.187m并抬升0.174m。行为定位到开炉之后的目标获取/抬升/转移，
未确认唯一grasp、接触或几何原因。state33各臂本就失败，多条编辑后的轨迹显著移动frypan而不转移pot，
这是已有失败没有被修复的反证；state34各臂成功，但所需control步仍不同。

## 5. 同观测十步函数、实践采样和数值边界

24对I/E+共同前缀各6点，共144点；在真实相同noise上增加E+@I、E0@I、I@E+，432次十步。
原最后实践episode各6点，共48点，对I/E+/E0各一次，144次十步。旧实践只存前5及mask，
没有伪造保存过原full50；新三方完整预测单列。以下是前5命令前六轴经controller输入clamp[-1,1]后的平均MSE，
并非末端物理位移、成功保持保证或参数范数。每通道、真实已执行mask和未执行尾部均在原件中。

| condition | 新初态E+@I−I@I | 新初态E+@I−E0@I | 最后实践E+−I | 最后实践E+−E0 |
|---|---:|---:|---:|---:|
| Spatial43 | 1.668e-3 | 1.376e-6 | 2.439e-3 | 3.810e-6 |
| Spatial29 | 1.009e-3 | 1.136e-6 | 1.100e-3 | 1.328e-5 |
| Object16 | 4.876e-4 | 6.441e-7 | 4.642e-4 | 8.118e-7 |
| Object4 | 5.290e-4 | 7.647e-7 | 1.831e-4 | 5.395e-7 |
| Goal15 | 2.823e-4 | 9.449e-7 | 3.972e-4 | 1.538e-6 |
| Goal6 | 2.156e-4 | 9.842e-7 | 1.534e-4 | 5.666e-7 |
| Long16 | 1.430e-3 | 4.749e-7 | 9.581e-5 | 6.498e-7 |
| Long44 | 1.491e-4 | 4.492e-7 | 1.211e-3 | 4.069e-7 |

安装的实际消费者是Panda OSC_POSE：前6输入先clamp[-1,1]，translation/rotation再分别映射到±0.05/±0.5；
PandaGripper按第7维的sign每步更新关节开合目标。两个相反方向的gripper qpos分别报告，不能相加充当夹爪宽度。
在144个新初态同观测采样点的各前5提议中，E+相对E0没有夹爪sign翻转；E+相对I有7个，集中在Spatial43/Long16/Long44。
最后实践48点，E+相对E0仍无sign翻转，E+相对I有2个，均来自预算失败的Long44。
这里统计反事实提议，包含末chunk尚未执行的命令；实际行为只能用相应执行mask和T+1状态判断。
未采样时点、不同已到状态下的夹爪时序或接触仍可能不同，不能据零sign计数排除其作用。

E路径在这些共同观测处的数值改动多数比完整编辑小得多，但E0→E+仍交换10条成功。
因此不能由“小差值”断言经验没有进入控制，也不能由存在差值声称它理解了纠偏。
实际早期变化经下一次观测、replan和接触继续改变状态；例如Object4/state33到275步的状态函数项MSE约0.221，
而该点参数项约5.64e-5。它说明已到状态差不能与参数差混为一项，**不是失败原因的百分比归因**。

同观测分解严格复用原实际I@I/E+@E+，另三方重新前向。
`F_E+(o_E+,xi)-F_I(o_I,xi) = [F_E+(o_I,xi)-F_I(o_I,xi)] + [F_E+(o_E+,xi)-F_E+(o_I,xi)]`。
除浮点加减外是代数恒等式，不是因果分摊；第二项也含保存预测与不同物理batch重算的普通数值差。
原实践I重算相对保存已执行命令的MSE均值约5.86e-6–1.57e-5（未clamp、7维），是实际数值参照，
不是bitwise identity闸门，与上表6轴clipped口径不同，不能直接相减成“校正效果”。
E+为历史编译、E0为本次编译，正常设备/chunk计算差异同样保留；未增加逐tensor一致扫描或重算择分。
小E效应不能宣称精确到排除了数值敏感性；成功翻转的重复稳定性未由本小面板建立。
最后实践六点也不覆盖全部成功支持，局部动作近似保持不能证明未见初态泛化；本轮未在原成功scene重新执行E+。

## 6. 专家FM与真实功能的具体关系

每condition7个非teacher成功episode×4等分区间，共28同noise/time query；三臂使用实际训练offset1、50×32 latent、
前7维与包含padding的full50 loss。`predictions.pt.fm`保留demo/frame、noise/time、target、padding和全部逐query/channel差。

| condition | I full50 | E+ full50 | E0 full50 | E+优于I的query /28 | E+相对I闭环 |
|---|---:|---:|---:|---:|---|
| Spatial43 | .128323 | .135148 | .134865 | 5 | 0→0 |
| Spatial29 | .158984 | .157287 | .157333 | 17 | 3→2 |
| Object16 | .099789 | .098990 | .099003 | 14 | 2→2 |
| Object4 | .102488 | .101270 | .101238 | 18 | 2→2 |
| Goal15 | .092755 | .092740 | .092768 | 14 | 1→2 |
| Goal6 | .096918 | .096623 | .096600 | 13 | 1→2 |
| Long16 | .107410 | .106750 | .106771 | 12 | 2→2 |
| Long44 | .116885 | .116350 | .116334 | 15 | 2→1 |

8条件等query均值I/E+/E0为 **.112944/.113145/.113114**；7/8条件均值改善，
但Spatial43恶化较大，使总full50稍差。224对query中108改善、116变差，不能把7/8说成广泛逐query改进。
front5均值为.130676/.129993/.130037，改善同样不等于闭环净收益。
Spatial29 full50约改善1.07%，front5差−.001256，却丢state33；Long44 full50约改善0.46%，front5差+.000245，也丢state32。
Long16 full50更好而front5更差，仍保留I的两成功。因此“专家FM更优、自己的控制更差”在两个具体条件中确实出现，
但不能推广到全部8条件，也不能把改成front5监督当充分修复。
Goal15的E+比E0 full50微优、front5反而略差，却多2成功；Goal6的E+两项略差于E0，却多1成功并交换全部成功位置。
E+−E0平均FM差多在1e-5到1e-4，没有形成统一的函数质量排序。

## 7. 从实际不足连续推导特征—算子—信用—行为

**出发点是整体能力净损，而不是某个内部loss难看。** 完整130对153的损失已发生在单次成功、多次成功与预算三组。
本次固定真实I后，Spatial43证明一部分损失早已存在；Spatial29/Long44证明最后一步也会损失；Goal15/6证明最后一步可以取得控制收益。
因此后继必须解释“哪种编辑在何种实际能力下有用”，不能从全局阴性直接关掉E，或从一个成功后的损失直接采用成功门控。

**现有输入怎样到达参数。** `ParameterEncoder`读取实际38套A行/B列，以固定MT RMS归一化，
按role/width投影为128+128维，加target/rank地址后形成38×128×256的q；它提供当前因子的条件，不等于能力语义已被学会。
`ExperienceEncoder`用自身pre/post双相机Phi、当前实际actor两flow时点H，加上proprio16、前5动作35、mask5、feedback4共60个数值特征，
编码真实episode/step顺序，经两temporal blocks汇成16 facts。原E/H停止梯度，学习过的投影、选择和汇聚属于共享χ。
`TeachingReader`让8个空间query+exact language先读取facts及当前q，再对teacher Phi/H做每帧压缩，
保留实际stride5索引/边界，经4个temporal blocks读取有序视频。facts又直接进入两个Axial editor。
所以删E同时改变压缩前的教学query和编辑memory，不能由E+−E0把原因单独定位为经验编码、视频注意力或最后decoder。

**参数编辑如何作用于执行。** editor产生`δq=sigmoid(g(q))*update_out(q)`。
每个incoming A/B的64维块以实际值/RMS和地址形成512维b，再由
`block_out = block_in + RMS * Wo[SiLU(b+Wq δq)-SiLU(b)]`生成完整因子。
δq=0时identity是初始化性质；训练后的δq没有闭环单调改善或成功保持约束。
完整单LoRA把原生target算子改为`W+BA`，随后每次真实十步Euler对50×32噪声积分，
前7维反归一化、前5进入controller，再由新观测重规划。参数差必须沿这条函数和状态反馈链才改变成功。
本轮已测共同观测处的完整十步与真实送环境动作，并用body/predicate定位阶段；没有用范数代替这一链。

**现有训练究竟教了什么。** 当前每事件的28跨episode专家query及`.2 relu(l_out-stopgrad(l_in))`只给专家FM信用。
每query在归一化系数之外有
`∇χ L = [1+.2*1(l_out>l_in)] (∂Λ_out/∂χ)^T ∇Λ_out l_FM`。
它沿完整decoder/编辑/经验与教学读取反传，但没有独立的incoming函数保持方向，也没有本批自身闭环return的信用。
反馈作为条件输入，不等于成功自动成为“保持当前动作/跨初态改进”的训练目标；共享FM合法学习也不等于新condition轨迹拟合。
这一区分是由实际`credit.py`和本轮两个FM/闭环反例共同支持，不是从常见术语推断。

**为什么不能跳到一个充分修复。** full50单time专家误差约束的是非teacher专家状态上的速度场；
部署需要自身已到状态、完整十步、真正执行前5、contact及后续纠偏。Spatial29在front5 FM也更好仍丢失，
故只把loss改到前5无法解释全部缺口；Goal E改善控制而FM排序微弱/反向，又说明统一压小参数/函数差可能压掉有用修改。
已有ADSP成功FM一阶22约束139→138、SKNC严格保持15完整LoRA 134→137仍丢13、
CV-CSD/DJNFR成功信用143→134/136降低了“保护有限成功支持自然迁移”的预期，不能因本轮换成真实incoming就清零。

目前支持度：经验能通过现有读取/编辑改变功能及成功集合的**有限证据增强**，净正向能力与稳定保持仍不成立；
共同教学/参数编辑漂移及早期累计失败仍有证据；专家FM信用不足以保证实际控制的解释增强。
“E统一有害”“只因成功后编辑”“只因未见non-MT incoming”“只因full50未加重前5”作为完整解释均受反例限制。
不同condition的target获取、放置几何与有限成功支持也可能是竞争原因；本轮没有唯一接触/编码根因、E信息的语义正确性，
也没有video必要性或后继学习可行性的直接检验。旧T161/other150、Long1晚期27的正证据与phase/SEOD/shared-SDE等阴性保留。
main下一取舍须从这些具体得失推导能够学习有用修改且解释保持/迁移的合同，不自动追加RL、保持loss、固定轮数或新panel。

## 8. 实际投入、端到端吞吐与退出

费用账本`costs.jsonl`的18个分配/child时间窗全部闭合，重叠同卡按并集合计：
**0.283373 GPUh**，首次科学开始至最后计算退出 **17.09分钟**，最多4张物理卡；
closedloop attempt wall5.78分钟，其中包含E0冷加载/读取与worker启动；两个attempt均exit0/complete。
elapsed包含读出实现集成的中间时间；不是纯GPU busy time。无新增失败、partial、重试、pilot或额外环境condition。

72 final共17,580真实环境steps，包含720 settling、3,391 replans；每臂11失败，共33失败，
失败耗11,110steps（63.20%），其中各失败最后50 control步合计1,650已包含，不另加费用。
I/E+/E0 steps分别5,810/5,992/5,778；没有新实践/重置适应或重读修订。
只为8套E0执行8次完整神经教学读取，共275实际frames，native encoder合计约17.52秒；
单materialize worker含冷加载104.12秒，allocated/reserved峰14.61/21.29GiB。
图像报告另只读8条teacher RGB，不能混成额外适应或漏记读取用途。
父8条件已有8初读+7重读=15全片、3,618实践steps/15重置、失败2,668/尾部598；仅引用其历史成本，未重复计入新GPUh。

现场两节点准入，gpu01:1/6与gpu02:0/2共驻可用A40；max UTL100允许已有高利用率进程，
保留真实显存余量、6/8总卡及单节点限制，未操作他人进程。闭环实际B1–16跨不同LoRA/persistent slots，
physical_batch_histogram和每workercomponent都保存，未dummy填卡。

| worker | 真实steps | 含加载wall秒 | steps/s | allocated / reserved GiB |
|---|---:|---:|---:|---|
| gpu01:1 | 3,225 | 140.90 | 22.89 | 11.85 / 29.29 |
| gpu01:6 | 2,965 | 126.00 | 23.53 | 11.46 / 23.98 |
| gpu02:0 | 5,889 | 174.34 | 33.78 | 11.85 / 27.12 |
| gpu02:2 | 5,501 | 169.00 | 32.55 | 11.85 / 28.35 |

环境独立slot的operation秒互相重叠，不能除worker wall作为GPU算子占比。
实测prefix/flow/wait约为15/76/26、13/64/25、22/108/18、21/103/19秒（按表顺序），
source加载、IO、组batch及CPU调度剩余不假称已细分。gpu01分到短task且受共驻/启动影响，不能从本表指定唯一瓶颈。

在同一注册离线矩阵内主动放大物理batch，四worker各两不同condition、144次完整十步：

| worker / microbatch | 含加载wall秒 | native144前向秒 | allocated / reserved GiB |
|---|---:|---:|---|
| gpu01:1 / 28 | 38.26 | 17.45 | 13.49 / 16.33 |
| gpu01:6 / 56 | 37.30 | 13.91 | 17.13 / 21.83 |
| gpu02:0 / 72 | 30.51 | 13.49 | 19.48 / 25.50 |
| gpu02:2 / 72 | 30.67 | 13.35 | 19.48 / 25.50 |

每condition的72新增十步分别3/2/1个实际batch，全部shape/finite/实际消费者完成。
条件、节点、共驻不同，不能称同条件因果speedup或持续最优72；每condition合法查询只72，
FM同条件固定28，未为继续填显存制造额外query。读出source冷加载、RGB/query IO、预测保存等也计入wall/GPUh。

启动前strg01 data1独立quota约1198.85/2048GiB、shared余量约80TiB，24GiB计划准入；
实际末次ROOT allocated约2.24GiB（含两个约0.28GiB冻结树），全程峰未连续测量，不能把末次du称历史峰。
仅建每host128MiB小feature cache，复用source/MT/原经验，没有新64GiB全片cache或全层hidden。
合并、完成且无恢复用途的本批工作树退役明细/实际释放及最终存储值在`analysis/cleanup.json`，科学原件/8E0/父checkpoint保留。
`analysis/resource_exit.json`现场确认两节点本批worker/ROOT进程均退出，并无ymdai compute GPU进程残留；未发送信号或修改设备。

## 9. 可复核原件、代码与交回

- 8原identity/scene/范围：`run_contract.json`；资源准入：`admission/*`及实际batch receipts。
- 72新行/全成功集合：`evaluation/results.json`；raw/noise/actions：`evaluation/traces/*`；被动T+1：`evaluation/capture/*`。
- 8×完整十步/FM及索引：`readouts/*/predictions.pt`、`result.json`和`readouts/results.json`。
- 图/短片：`analysis/behavior/index.json`、`index.md`；只读消费：`scientific_analysis.json`及`analysis/final_facts.json`。
- 实际退出：`batch_attempts/closedloop_001`与`readout_001`；全部成本：`costs.jsonl`；最终科学完成：`scientific_completion.json`。
- 代码：E0/72闭环来自clean pushed detached **c381f178**；离线读出来自 **6204e027**；E+的历史编译身份保留在原record。

复用单canonical Runner/native flow/BatchedLoRAInference/persistent environment，不另建Writer、Policy、仿真或大缓存运行面。
窄诊断owner分别为`edit_diagnostic.py`（300行，面板/queue/E0/完成）和`edit_readouts.py`（338行，固定自身/FM读出）；
针对现有执行、配对、scene/capture和实际FM消费者的37项检查通过，日志`analysis/integrated_consumer_final.log`。
结构检查保留已有Runner/IPC复杂度事实，不为一次诊断拆出第二框架；新增测试合入既有execution owner，未扩大42-peer目录。
最终科研记录按正常Git集成推送；计算冻结树和task-owned合并树按生命周期退役，代码可从已推送commit恢复，
退役后的历史路径不继续冒称直接可运行。没有删除父原件、optimizer或关键权重。
本批没有自Queue、逐阶段通知或第二退出等待者；实际计算退出由持续等待直接消费。
最终一次报告使用当前main UUID01a11b05-3460-7eb0-aca8-177d6d86ef48，active Steer或idle/notLoaded无覆盖resume-start，
接受与真实处理分别记录在`main_delivery_receipt.json`。可靠交回后main继续科研取舍；本批complete不关闭EMBER。

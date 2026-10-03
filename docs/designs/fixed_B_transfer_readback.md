# 既存共享B修正的跨任务完整读回

2026-10-03，机制§118。唯一研究问题：固定900表示/A、四task八teacher普通FM学得的共享B修正，
是否能为未参与该学习的任务产生真实闭环收益。一次冻结读取，不是新训练、辅助项、课程或最终selected声明。
本合同登记后才产生新增读取授权；§116固定学习本身已经结束，不能从其旧预算或清单自行续训。

## 1. 已有依据、竞争解释及固定模型

§117父/A/J分别13/24/22，共32条件、16物理初态。A对父得14失3，其中task32取得7个原失败的放置。
J的额外位移信用没有净收益，已关闭；当前选择由A真实控制正例及其尚未验证的跨任务作用共同支持。
旧PZ转移失败、条件读写seen强增而held弱增均保留，不能假定正例能迁移，也不再用更多局部拟合代替完整读取。

只消费以下完整既存权重，不选择其它update、teacher、参数缩放或输出融合：

`/data1/user/ymdai/ember_runs/joint_action_effect_credit_20261003/A/checkpoint64/writer.safetensors`

使用其checkpoint manifest和原A学习合同作为来源，保持全部参数值及`conditional_read_write`计算。
只更新过38处五组B参数的事实来自b0df34ee真实学习；读取时整个Writer/source均冻结。
原8条件的训练bank只是证据，不是validation bank，也不能按task/相似视频插值或建立字典。
为每条合法validation视频以完整模型重新一次性编译唯一38-target LoRA，不复制源模型，不构造新数据或优化器。

来源必须准确分列：

| 资产/阶段 | 实际身份 |
|---|---|
| Source1000 | b8ea00e9，沿原source/normalization/tokenizer/assets路径只读复用 |
| 条件读写450父训练 | a0e0248d |
| 900续训及原correct400读取 | 85919994；900 launch/code_identity.json及实际run_contract.json为准 |
| 原32行诊断Original bank读取 | 923ff89b；其training_git亦为85919994 |
| A64共享B学习及原功能读取 | b0df34ee1bc3fd09ff48bc603f08a272dd8ae359 |
| A64原32行环境读取 | a270fb01 |
| 本次物化/环境读取 | 实施者新clean pushed detached commit，运行前登记 |

§116及已封存identities曾把900训练/物化简称a0e0248d；它是450父身份。权重路径没有改变，
旧原件保留，不覆盖、不因此重跑A训练或原功能面板；本次必须附这条来源纠正。

## 2. 唯一400行及信息墙

严格复用父900下列实际结果及run contract中的任务、scene与全部state–video映射：

`/data1/user/ymdai/ember_runs/conditional_read_write_continuation900_20261002/conditional_read_write/evaluation/900/correct400/`

global任务为3、6、11、16、23、26、31、39，各init0..49；每task50个正确teacher各一次，
整个400为400个不同task–video条件。禁止改成八条teacher复用的screen，禁止按模型表现改映射。
调度继续使用原`video_schedule.py`合同；物化/worker/shard顺序不改变condition或RNG。
完整Writer输入只有exact language及原action-hidden双RGB、stride5，无teacher action/state/pose/reward/task-ID route。
本次不读held动作计算FM、不生成标签、不产生任何梯度；不扩训练任务、episode、数据源或输入字段。

官方render256/model224、双相机180度rotate、state8/action7、10 flow、执行前5、settling10、
各suite horizon及成功即停均保持。scene复用父900的原scene合同，policy/env/root seed7与原固定噪声时钟配对。
每task/init0 full双RGB，共8 full；其余392 compact。全400保存完整action chunks/真实physical actions、
goals/continuous及既有对象registry，不新增接触标签、轨迹挑选或额外model forward。

canonical cost-balanced long-first dynamic queue与persistent workers继续使用；允许独立物化和已就绪的评测在资源内并行。
Compiler对同一视频只需一次完整编译；normal数值差异接受，不追逐低位/逐tensor一致。
不长期保存400份全部native激活；streaming读取并只保留本批正式bank及必要来源，避免数十GiB不必要缓存。

## 3. 比较、统计及会改变什么判断

唯一新增结果为A64 correct400。匹配比较仅读以下旧原件，不重评父/T/MT：

- 父900 140：上节目录results.json。
- 成熟T2340 161：`/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928/continuation2340/T/evaluation/2340/correct400/results.json`。
- 强MT300 153：`/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928/stage1/MT/evaluation/correct400/results.json`。

检查既有显式任务/scene、preprocessing和env/policy RNG来源；T还须对应相同teacher映射，MT为同物理case的共享策略参照。
若旧参照在任何必要配对字段有真实差异，如实分开matched与背景，不静默重评或更换强参照。
报告total、per-task、per-suite、breadth、R/G/L、churn、Jaccard与全部成败行。
沿既有whole-task配对不确定性口径即可，明确仅8个task簇、没有训练seed不确定性；不以阈值掩盖实际效应与能力交换。

结合所有原有轨迹/goal/continuous消费新得与丢失：尤其父16的对象误选、31的双目标、23/39原弱能力及其它task保持，
但不按这些现象另加环境case、渲染、接触probe或held动作分析。主讨论负责科学解释，不要求执行者定位唯一根因。

- 若有跨task完整收益且与强参照相比有实际进步，支持共享B功能修正具有一定可复用性；仍没有证明视频动态必要性或稳定方法。
- 若仅比140小增、仍低于强T/MT或主要是大规模交换，保留局部转移，不视为达到Owner的性能目标。
- 若未保留训练侧获取、无益或恶化，降低将本次有限B修正作为可复用主方案的支持；不靠继续训练/挑视频/扫描挽救。

这些分支都不触发自动新训练、J400、other/wrong/no-video/shuffle/reverse、相邻checkpoint、官方Test或RL。
只有一个固定读取点，不能声称相邻稳定、正式selected或已确认视频因果机制。完整400后停止，由main结合原件继续判断。

## 4. 成本、实施和结束

唯一新增root：`/data1/user/ymdai/ember_runs/fixed_B_transfer_readback_20261003/`。
预计含工程/读取/交付45–90分钟；硬限**3完整GPUh、24GiB新增峰值**，预计18–20GiB。
原900同架构400-bank .299424 GPUh、correct400 1.025261 GPUh为吞吐依据；400×39.293MiB完整bank约15.35GiB，
另计8 full/392 compact、连续轨迹、临时写入、工程/冻结代码及分析，不复制已有A checkpoint、optimizer或Source。
到2.5GPUh若剩余预计超3，则报告剩余量/原因；不得删case、少计加载/I/O/失败、扩大预算或擅用screen代替。

正式物化/环境执行前live检查两节点GPU、data1独立quota/个人用量/shared空间，记录当前与启动后物理卡数。
按实际可用吞吐选择资源，遵守仓库8/6合计、单节点6上限；不把上批world1或两臂并行安排变成本批卡数上限。
沿有效frame chunk/batching和persistent worker配置，不以最低显存为目标、不等待凑卡或dummy占用。

既有实验session独占canonical tracked/Git窗口完成实现、针对性真实消费者检查、集成push与clean detached冻结。
复用现有完整Compiler、bank、官方evaluator与场景合同；只作必要的归档A64读取接入，不恢复已退役学习或建立第二常驻运行面。
必要检查针对完整Writer加载/38-LoRA、原资产/视频映射/场景/RNG、0梯度和400完整产物，不加hash/全树/逐tensor审计。
工程原因若只影响接入/调度，按仓库规则修复并保留原件及费用；若需改计算、数据、选择或预算，报告具体边界由main判断。

正常运行持续等退出事件，一次整批完成或异常回报；不周期日志轮询、不对主讨论自Queue或发多段完成心跳。
保留run contract、bank manifest、raw rows、aggregate、配对/科学读回、来源纠正、completion及完整账本。
交付前退役本项临时入口/hooks，保留Git/frozen/正式和失败原件；完成Git集成push后交回窗口，无自动后继。

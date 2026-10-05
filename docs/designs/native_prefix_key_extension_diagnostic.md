# 前缀key条件编译：已撤回的56-target候选，未实施

**状态：2026-10-05已撤回，未实施、未派发、无新增计算；不是active design，不再等待38→56确认。**

Owner明确前缀尽可能不允许改动，因为会改变图像理解。主讨论据此撤回本候选，后继保持原生图文prefix冻结。
原推导只证明局部直接读出约束，未证明有用语义丢失或改前缀的必要性；该不足与Owner方法边界分别记录于机制§139.5。
下文仅保留原候选的历史计算与拟范围，不构成执行授权；拟预算从未使用，原38-target合同保持。

## 1. 要改变的判断

raw_key/centered_rgb原布局均0/8，居中没有取得正确获取；换位读取和完整完成分离。
考虑执行侧冻结prefix-key读出是否限制了教学角色对应与完整控制的共同学习。
竞争解释包括未学成可迁移语义、后续反馈相容性、其它输入路径及有限学习；投影核的存在不证明实际信息丢失。
只做一次共同学习的完整功能干预，不补分类probe/局部对齐阶段，不称formal fresh或总体性能比较。

## 2. 唯一改动与所需范围确认

候选一套LoRA含原38目标及下列18目标，rank128/scale1，共112个A/B张量：

`model.paligemma_with_expert.paligemma.model.language_model.layers.{0..17}.self_attn.k_proj`

每处shape为A_K[128,2048]、B_K[256,128]。source基础参数始终冻结；没有第二expert、运行时Writer或task-local优化。
标准k_proj作用于所有prefix token，包含图像、语言及自身state；不得为取得图像专属解释暗加token mask或特殊部署路径。
旧38合同/正式银行读取器继续保留；本次56只能由独立诊断合同明确接受，不放松原guard让任意目标混入正式消费者。

对照复用`/data1/user/ymdai/ember_runs/native_role_centered_content_20261004`的raw_key组：
学习在`raw_key/checkpoint64`，银行/环境/B20在该root的banks/evaluation/predictions及其实际manifest，不重跑旧组。
两者同父C900、固定locator、原38头/P/D初始化、事件/RNG/损失/64终点；不继承raw_key64控制权重再额外训练64。
新分支初始化应在独立RNG上下文中进行，避免改变原38/P/D初值或query/noise流。

## 3. 内容与编译

复用原25视频、758 origin、旧c/d/X/H、冻结R64 locator Pq/Pk；无新task、episode、数据来源或held教学特权。
按原raw教师完整像素及语言，一次只读真实双相机prefix，在18个k_proj输入捕获实际2048维U。
沿旧alpha逐origin、50槽归约`e=normalize_2(sum(alpha*rms0(U)))`；eps固定1e-6，零保持。
保留原origin/alpha/camera配对，输入未crop；不保存全patch×18层大缓存，不运行无消费者的action suffix。
归约后的e用FP32保存，理论有效载荷约5.205GiB；缓存身份应准确列出实际新producer及旧locator/c/d来源。
teacher不读state，query正常使用自身state；接口同维不代表两端特征分布已对齐，不增加teacher伪state弥补。

新增每层D_K[128,1024]、C_K[256,128]。D_K为seed7下普通非零线性初始化，C_K为零；无bias。
`r=D_K rms0(d[1:])`，`A_K=mean_(origin,j) r e^T`，`B_K=C_K`。
原38生成仍为原raw_key的C900条件头加原角色R；公共/source/interpreter/locator均冻结。
18个C_K与D_K共2,949,120参数，与原38十类头及原fresh P/D共同学习。
一次条件只编译一套56因子；每次环境replan照常处理自身观测，但不重新读取teacher或重新编译。

## 4. 固定学习

严格复用旧64训练事件、八task[12,13,14,15,17,19,43,96]、每task teacher0/1和query demo2..29。
14,336主FM查询及同epsilon/tau1角色查询，原无效角色零项及均值分母不变；只在train标签上求梯度。
完整FM+0.1原LQ、原AdamW设定，无额外scheduler或新辅助项。只取64终点，不按中间/held结果选择checkpoint。
新增prefix适配改变自身前缀：训练不得在其外包no_grad而断开C_K/D_K功能梯度，不复用旧condition或旧参数版prefix cache。
官方原生混合精度、TF32及官方10flow消费者保持；不增加完整生成外autocast或追逐bitwise一致。

在本稿拟固定的真实训练事件内至多两次可丢弃profile，比较实际吞吐/峰值后恢复全部模型、optimizer、sampler及RNG。
验证初始新增K确为identity、首步C_K及后续D_K余切实际到达，并核原38共同学习和同版56因子消费。
不能仅凭梯度非零声明机制通过。checkpoint保存全部训练参数、optimizer、RNG/cursor/topology/schema与新旧身份。

## 5. 固定读回与行为裁决

25 condition各编译一次，共25套完整56-target银行；不另挑teacher、混权重或添加第二adapter。
复用旧raw_key的55环境case（32 train、23唯一held，31 full/24 compact）及320 B20，scene/noise/teacher和长度口径保持。
原布局8、交换8、八格与其重合行的去重规则不变；全部正确目标、错误实体、43 In/Close、完整成功与失败均保留。
强T/MT/父C900的已有同场景结果只读保留，缺少的强参照case不自动补跑。

31 full在真实首次规划同一次forward被动读取10tau×18层×8头×前5槽的角色密度、实体/图像实际质量。
必须使用**新56策略实际产生的prefix K与Q**，不能沿用旧source key数组计算假分数。
复用原25自身mask，无新标签环境或held teacher状态/动作。B20原标签只有q，实体总质量不可得的边界保持。
保存320完整50×7动作/目标/valid mask；按task/teacher/query等权报告前5/全50及translation/rotation/gripper，保留全部不利项。
新K影响prefix自注意力及语言/state，读回不能把完整效应唯一归因于图像K或原投影信息丢失。

最有意义的正例是原布局出现多个正确获取并完整完成、交换布局控制同时改善，且没有大范围训练能力丢失。
角色rho、训练拟合、gripper误差改善或单个孤立成功均不足以称修复；所有R/G/L、churn、breadth和成功交集必须报告。
即使阳性，也只是该完整构造的有限证据，无formal/400资格或必要视频增量结论；负例降低prefix-key主修复假设的优先级。
固定范围后停止，不自动接V/MLP/层/rank/seed/LR/loss扫描、续训、formal fresh、400、视频controls、Test或RL。

## 6. 拟资源与交付边界

预计含工程4–6小时；基于旧两臂学习及物化/评测等共1.446GPUh、prefix-only758个origin约70秒，本候选仅一臂，
但新增前缀反传的实际成本未知，不能照搬旧吞吐。拟硬8完整GPUh、新增峰24GiB；>6小时或费用达6GPUh且预计不能在硬限内完成时回报具体缺口。
固定25套FP32银行约1.453GiB，e约5.205GiB；合计恢复checkpoint/原行/B20/失败/frozen/原子暂存，拟预留16–20GiB而非只计模型。
新增全部data1；建root/cache之前核strg01 data1独立quota、个人实占、shared及预计峰，每次launch现场核双节点GPU和总卡上限。
当前尚未获目标范围确认，不创建拟root、不安排GPU，也不将本预算记成已使用或已经获准执行。

若确认后的拟root为`/data1/user/ymdai/ember_runs/native_prefix_key_extension_20261005`，执行前仍核是否已有同名资产，不能覆盖。
代码由实验session独占隔离实现、真实消费者检查、集成push，再从clean pushed detached执行；main不重复工程审查。
所有旧来源/失败/完整费用保持。专用实现和56诊断入口在固定批次结束后退役，旧38正式合同保持可读。
完成全部原件、正反读回、退出/释放、退役/Git后一次整批交回main；无阶段自通知或自动后继。

## 7. 最终状态

候选已撤回；没有实现、profile、GPU、训练、环境或新的模型结果。
不再请求或等待38→56许可，不由本稿恢复执行，也不通过其它模块名改变部署前缀。

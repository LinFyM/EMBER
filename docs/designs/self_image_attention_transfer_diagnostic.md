# 自身图像读取分布交叉诊断

本合同登记于2026-10-03，机制依据见[主分析§109](../analyses/feature_to_operator_mechanism_20260926.md#109-把自身图像读取与后续控制作一次可失败的分离2026-10-03)。
这是两个已冻结策略之间的有限因果干预，不是新Writer、部署候选或正式checkpoint选择；状态只看[progress](../../progress.md)。

## 1. 问题、依据与竞争解释

校准读写N450的teacher动作拟合改善，但correct400=122、seen144=77，低于同龄C450的137/91。
已读配对RGB：train task13的C搬bbq sauce入篮，N转向salad dressing；task4则N从已开抽屉取碗放盘而C未完成。
support task56两者都移动前碗，但N没有完成旧C做到的叠放。三类实际行为均保留，不能把它们统称为感知失败。
§108的匹配指令反转另说明更准局部动作不自动成为自身控制；它没有证明这三个任务的图像读取就是根因。

本项区分：两完整策略之间可迁移的部分控制差异，是否由**自身Action Expert如何在当前可见图像tokens之间选择**承载。
竞争解释包括后续Value/MLP/执行控制的差别、读取与其它计算的强交互，以及图像读取总量而非内部选择的差别。
不从attention可视化或错物体本身认定根因；需要实际有限干预及闭环行为。

## 2. 唯一干预及精确计算

C指原conditional_read_write450，N指control_calibrated_read_write450。二者使用同一冻结source，38-target均仅在Action Expert及动作投影。
在相同当前RGB、exact language、自身state、同一x_tau与tau下，它们的冻结图文prefix及其K/V相同。
这个事实来自实际target拓扑；执行前须核实际消费者的合法图像位置、mask与prefix复用，不能依据硬编码长度猜第三相机是否存在。

对某层、head、action slot，记recipient R的完整attention概率为p_R，I为两个实际可见相机的图像token集合：

`m_R = sum_{i in I} p_R[i]`，`pi_R[i] = p_R[i]/m_R`。

donor D在同一真实输入上运行其**原完整、未干预的**suffix，提供`pi_D=softmax(logits_D[I])`。
只替换recipient的图像选择分布：

`p_R<-D[i] = m_R pi_D[i] (i in I)`，`p_R<-D[j] = p_R[j] (j not in I)`。

因此在该层的当前输入处，精确attention输出增量为

`delta y = m_R sum_{i in I} (pi_D[i] - pi_R[i]) V_image[i]`。

recipient自己的图像总质量、非图像attention、suffix K/V、残差门、o_proj、MLP及action_in/out权重均保留。
原生RoPE、实际mask、KV重复和正常BF16/FP32 softmax语义保留；donor条件分布用稳定log-softmax/softmax计算，不因m_D很小改阈值或丢head。
后层hidden/suffix K/V会因前层干预而变化，m_R亦在当前混合前向中重算；不能声称整个轨迹或所有原recipient激活保持。
在全部18个Action Expert层、全部head/50 action slots、每次官方10步flow上应用同一个规则，不扫层、head、时间、强度或相机。

每个flow步必须先以**recipient当时同一x_tau**计算donor原前向，再计算recipient混合前向；
不得让donor独立积分自己的动作轨迹，不移植donor最终动作/velocity、suffix hidden或图像读取总质量。
donor前向没有额外噪声/随机采样，不与上一flow步或另一环境状态错配。每个replan只建立一次当前合法prefix/cache，两次suffix共用。
可在同一个source实例上切换两份既存LoRA，按实际接口实现；不复制基础模型，不改canonical环境、flow或归一化。

四臂为C原策略、N原策略、C<-N图像选择、N<-C图像选择。原策略也在本批相同实际环境/runtime重放，
避免把旧运行差异当成干预收益；原历史行仍保存。混合两臂是诊断控制器，明确违反论文的单策略部署形式，不能作为EMBER成绩。

## 3. 固定面板及原件

只使用已完成seen144中的global tasks[4,13,56]和init states[32,33,34,35]，共12个原条件×四臂=48条环境行。
这是针对已观察得失的选择性分析面板，不是随机总体或strict400资格；不添加其它task、init、video或重试择优。

| global task / 实际suite与index | init32/33/34/35固定teacher | C历史成功 | N历史成功 | 事前用途 |
|---|---|---:|---:|---|
| 4 / libero_spatial:4 | 28 / 8 / 0 / 30 | 0/4 | 4/4 | N取得碗的正例与反向移植 |
| 13 / libero_object:3 | 15 / 8 / 31 / 30 | 3/4 | 0/4 | 错对象/目标未取得的损失 |
| 56 / libero_90:16 | 4 / 47 / 30 / 24 | 4/4 | 1/4 | 已移动目标仍未叠放的边界 |

严格复用原scene_canonical144场景、seed20260928 teacher映射、env/policy RNG与官方动作协议；
task56使用原seen定义的400 horizon，不误套目标40的suite列表。其它分别220/280；成功立即终止。
normalization仍只用原source；没有teacher action/state输入、官方Validation/Test标签或梯度，也没有新增训练样本。

C bank：`/data1/user/ymdai/ember_runs/conditional_read_write_fresh_20261001/conditional_read_write_seen/banks/450/manifest.json`。
N bank：`/data1/user/ymdai/ember_runs/control_calibrated_read_write_20261003/control_calibrated_read_write_seen/banks/450/manifest.json`。
两者已有全部固定teacher最终LoRA，复用原bank loader及公开/条件组合口径，不新跑Writer或重物化。
原results在各对应`evaluation/correct144/results.json`；source沿用aligned1000的既存权重、tokenizer、assets和scene来源。

每task×四臂的init32保存full双RGB，共12 full；其余36 compact。全部48行保存实际physical actions、goal trace、continuous EEF/物体/夹爪。
末段只将成功停止前实际执行的动作计为physical，不把未执行的规划前缀补入轨迹。
针对介入是否生效，保留每层/flow时刻的recipient/donor图像质量、条件分布差与**实际加权Value增量**的紧凑统计。
可保存full行的少量固定时刻、平均后的空间图供视觉检查，但不保存巨大逐token全矩阵，也不把平均图当作head/角色绑定证明。
不新增几何标签恢复、动作标签、探针训练、图像分割、额外反事实环境或行为阈值扫描。

## 4. 事前解释与停止线

- task13若N<-C恢复目标bbq的取得/完成，且C<-N出现相应选择损失，支持这部分差异可经自身图像选择传递；
  task4的反向预测是C<-N取得更多正确碗操作、N<-C损失原有获取。两方向及不利行均报告，不只交最好混合臂。
- task56检验相同接口是否也影响放置关系；不预设它应当跟随donor，更不把两个阶段直接命名为两个head或层组。
- 只有单方向或少量变化时，解释为有限贡献/交互；若差异主要保留在recipient，降低“直接约束自身图像选择可作为当前主修复”的优先级。
- 无论哪种结果，均不能证明q是唯一根因、attention就是物体理解、donor含正确因果知识，或合法一次LoRA已经获得新性能。
  原图像总质量保留，所以本项阴性不否定图像读取总量或整个Q路径的作用。
- 原策略重放的正常数值差异如实报告；若成功分叉，干预以同批原策略为配对基准，不追逐逐bit复现或由旧分数宣布工程故障。
  实际scene/RNG/输入错配则按工程合同修复后从新的clean pushed frozen版本继续，失败成本与原件保留。
- 结束后主讨论根据全部行为与特征效应决定是否有依据研究自身对象关系的学习约束；不自动追加grounding loss、蒸馏、formal fresh、层位/质量/强度扫描。

这项只回答下一方法取舍的一处证据缺口。即使局部读图作用能够转移，也须另行说明合法教学如何产生该作用、共享学习是否保留它，以及相对强MT的完整能力；不能拼接局部正例。

## 5. 执行、预算与交付

唯一实验session负责工程、实际消费者验证、隔离分支、Git集成及运行；main登记后交出canonical tracked/Git窗口。
预计含工程3–5小时；硬限3完整GPUh、新增峰8GiB，所有新增位于`/data1/user/ymdai/ember_runs/self_image_attention_transfer_20261003/`及data1工程树。
估算参照旧144行两卡每卡3 persistent replicas约683秒；本批48行中24行增加一次suffix读取，另外计加载、hook核验、失败与I/O。
吞吐/显存变化由真实消费者的小型核验决定，不以这个估算保证时长或为了填卡新增样本。
launch前按AGENTS同时live核两节点、适用总卡数与data1独立quota/相关用量/峰值及共享容量；本批没有新增资源例外。
使用既有cost-balanced队列、long-first和persistent workers；source保持一次resident、复用prefix，不训练、不启NCCL。

仅做必要的接口检查：图像mask/token范围；同一x_tau/tau/prefix传递；四臂路由；self-donor在一个真实query上的正常数值一致；
真实计算的概率总质量和非图像分量保留；hook退出后移除。不为逐bit一致退回慢kernel，不新建通用测试框架。
实现/运行必须来自clean pushed detached冻结树，不原地修改旧源码/产物；语义未变的明确接口修复由实验session闭环，不新增主讨论工程验收停点。
超预算、改变科学干预或原因不明时报告具体边界，不自动缩面板、加臂、改变模型或科学目标。

交付registration、代码/精确命令与来源、48原行/逐task和R/G/L/churn、全部captures、紧凑作用统计、completion及完整GPU账本/设备释放证据。
专用入口/hooks在交付时从活动树退役，保留Git/frozen原件及分析；不保留第二套Writer/evaluator或可部署混合路径。
整批Git推送、一次有来源标识的完成/异常回报后停止本项，交回canonical窗口；没有自Queue陪跑、周期状态回报或自动后继。

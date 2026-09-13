# 语义状态路径模型的冻结行为回放

2026-09-14。固定128次回放及全部32组人工观察已完成关闭，结果见§5；§1–4保留原登记。
不恢复已关闭的Writer训练，不选checkpoint。

## 1. 要区分的解释

[完整50／100比较](semantic_path_writer_design.md#8-完整50100结果与关闭裁决2026-09-14)中，
训练动作FM继续改善，但两臂validation均回退。同为cream cheese，basket任务source为5/50，
有序50／100为35／27；bowl任务source为41/50，有序为36／13。
这些分数不能区分对象／目的地判断错误、抓取失败、运输失效和最终放置失效。
两个任务还包含场景等差别，不能把跨任务差额解释成目的地的单变量因果效应。

本诊断定位正常correct条件下最早可观察的行为分歧，为下一项机制判断提供证据。
它不直接识别网络中的唯一失败模块，也不把动作FM、参数漂移或视觉猜测当作根因。

## 2. 固定条件与复用

- 模型固定为本轮ordered／frame_set各step50和step100，四者全部覆盖，不挑较好节点。
- 每模型完整validation8，各取排序等距初态`0,12,25,37`，四面板各32条，共128次。
  选择不读取单条成功结果；不按成功／失败、增益／丢失条件筛例或追加初态。
- 每条复用原400面板的同task、同state、同teacher ordinal、同完整LoRA与环境／policy RNG。
  原400次是性能证据，回放分数不替代它，也不用于模型选择或再次训练。
- 复用clean pushed frozen代码`5116deb0efb3c2d77dbb6228772e859176496fdc`及原source、
  normalization、tokenizer、官方环境与执行参数。没有新Writer调用、LoRA合并、部署优化或教学视频controls。
- 不读Test，不产生held梯度。只保存冻结policy正常执行时已有的图像、含量化state的tokenized prompt、动作chunk及BDDL目标谓词变化。
  这些是执行侧诊断信息，不成为Writer部署输入或训练数据。

研究根为`runs/analysis/semantic_path_writer_20260914/behavior_replay/`。
四个`registration_<arm>_<step>.json`使用既有
`ember_pi05_frozen_replay_registration_v1`，role=`validation`，上述四个init，
`training_gradient_use/checkpoint_selection_use/test_use/outcome_dependent_selection`均为false。
`reference_output`逐一指向原研究根`training/<arm>/evaluation/validation_correct_step<step>`；
adapter仍指向原`training/<arm>/materialized/validation_correct_step<step>/manifest.json`。
使用既有`scripts/evaluate_pi05.py run --role validation --mode screen --state-count 4
--init-state-ids 0,12,25,37 --frozen-replay-registration ...`，由合同验证原完整bank及配对范围。
不放宽完整task集合或原policy身份检查。

## 3. 观察与裁决

全部32组task/state均逐模型登记：原成功与重放成功、输入／RNG配对、谓词变化、首次可见行为分歧，
以及是否可观察到接近正确对象、夹持与抬起、运输、释放／放置。
对象身份、正确目的地或夹持状态无法从画面确定时明确标为不确定；仅凭接近画面不能断言接触物理。
保存固定均匀时点的双相机图像索引及必要的局部连续片段，不只展示成功或最显著失败。
若回放结果有正常数值／调度差异，保留原结果和差异，不反复重放到符合预期。

- 若回退主要是明确可见的错误对象／目标选择，条件绑定成为有依据的竞争解释；
  仍需与历史语义／角色模型比较，不能直接启动新解析模块。
- 若对象与目标均正确而主要在抓取／运输／放置阶段失效，则撤去“只需修正语义目的地”的直接依据，
  转而讨论条件控制学习与保持；不得把阶段现象等同某个参数模块的因果定位。
- 若不同任务／初态呈现异质失败或图像不能区分，则停止单一接口故事，明确本回放的识别边界。

回放完成即关闭此固定诊断，不扩展状态、checkpoint、模型或controls。无论哪个分支都先更新整体解释，
不会自动触发另一完整Writer、超参数扫描或旧假设重启。

## 4. 执行与资源

复用现有冻结checkout、环境及所有大资产，只新增正常replan轨迹与小型图像索引。
四模型×四init×全部suite horizon上界为8,448个replans；保存的是resize前双256×256 RGB FP32，约12.375GiB，
含动作／state／token、索引和必要片段，新增峰值预算16GiB，全部置于data0。
登记前strg01 data0个人使用81,496,036KiB，独立soft quota为1,073,741,824KiB；
研究根10,582,248KiB，共享可用约1.6TiB，预算充足。data1不新增大输出。

每次launch前检查两节点GPU与本任务占用，使用有余量的设备和已有实测worker设置，
总占卡遵守现行六卡边界；dynamic queue、long-first、persistent workers保持。
代码commit、登记commit、实际命令／设备、quota与completion记录于研究根launch contract。
本项复用已验证evaluator，不新增实现或测试框架。

首批ordered50／frame_set50已通过原面板合同准备，分别在gpu01／gpu02的4、5、6运行，每卡两个persistent workers。
轨迹图像分辨率经processor实现核对为256，16GiB总新增预算不变；模型内部仍采用官方224预处理。

## 5. 完整结果与关闭

四面板各32次、128条原始轨迹与目标谓词、32张四模型双相机拼图齐全；全32组已人工检查。
24个worker全部exit0，累计launcher812.30秒，轨迹11.74GiB，两节点已无本研究在途进程。
原400对应子集→回放成功：O50为7→5、F50为5→9、O100为3→5、F100为1→1。
120/128条结果相同；8条差异完整保留，模型／LoRA／teacher／task／state及RNG公共前缀配对通过。
本次图像只解释本次回放，原400成绩和已关闭Writer裁决不变。

奶油奶酪Goal6的100节点六条失败均可见目标留在桌面、手已转向碗附近；另一个50节点失败已将目标带到碗上方，
但最终谓词始终未成立。Object1有抓取／运输迟滞，也有有序100比50更快成功的反例。
这组没有提供明确错误目的地证据，但不能外推为全局语义正确：其它任务存在错实例碗、错瓶、错包装盒和平底锅，
Goal3停留抽屉开合，Long还出现正确子目标完成后丢失或未与另一子目标同时成立。

稀疏画面有识别边界：Long1/state0/F50的cream cheese目标谓词223步成立、361步失去；
Long2/state12/F50开炉70步成立、120步失去。若只看九个抽样时点，就可能误判为从未完成。
完整谓词已与人工观察分开登记；不把接近、遮挡或末帧重复当接触／释放证据。

按§3第三分支关闭：**失败异质，未唯一定位模块或单一操作阶段。** 停止以“只改目的地解析”、
“全部是抓取失败”或“统一由occupancy漂移造成”为当前充分修复理由。
本次没有证明新的正向方法，也不自动触发保持蒸馏、局部动作头、状态扩池或下一完整Writer。
不扩状态、checkpoint、controls或新的GPU回放；整体goal仍未完成。

全量原件、逐组图像入口与两份人工观察：
[behavior_replay/READOUT.md](../runs/analysis/semantic_path_writer_20260914/behavior_replay/READOUT.md)。
`replay_readout.json`保存配对与谓词，`decision.json`为关闭裁决，`launch_contract.json`保存执行和退出。

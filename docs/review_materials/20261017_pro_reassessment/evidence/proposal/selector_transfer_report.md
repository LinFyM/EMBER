# 固定G38349的selector72整批验收

ROOT: `runs/video_guided_proposal_writer_20261013/selector_transfer72_fixedG38349`。原owner06 exit0；A180+B18和唯一correct validation400完整。验收仅CPU消费既有原件，0新forward/rollout/Test。

唯一B18结果163/MT153，R123/G40/L30/net+10/churn70/Jaccard0.6373，覆盖6/8。相对旧完整165，R139/G24/L26/net−2/churn50。没有明显广泛且稳定的总体改善，完整目标尚未达成。

| task | MT | 新流程 | retained | gained | lost | net |
|---|---:|---:|---:|---:|---:|---:|
| 3 | 41 | 44 | 39 | 5 | 2 | 3 |
| 6 | 7 | 9 | 4 | 5 | 3 | 2 |
| 11 | 36 | 39 | 31 | 8 | 5 | 3 |
| 16 | 10 | 17 | 4 | 13 | 6 | 7 |
| 23 | 0 | 0 | 0 | 0 | 0 | 0 |
| 26 | 35 | 37 | 31 | 6 | 4 | 2 |
| 31 | 24 | 17 | 14 | 3 | 10 | -7 |
| 39 | 0 | 0 | 0 | 0 | 0 | 0 |

Spatial48→53、Object46→56、Goal35→37、Long24→17；task16净+7，task31净−7，task23/39仍0。MT最终回退仅4/400，396部署生成的完整LoRA；最终G000/001/002/003/004分别69/71/88/82/86。395条件外部五轮停止，5实际STOP；轮数0/3/4/5为1/3/1/395。实践1988次，明显多于旧流程1465次，尚未变成更高完整闭环表现。

固定的是G参数，parent、真实H、后续G输出与最终LoRA均可改变；收益/交换先归完整流程。没有新的held候选counterfactual query，不能区分候选本身与最终选择的全部贡献。本结果不认证视频/H必要性、相邻稳定或Test。

完整状态：local180与RL18 CPU实际载入，全G/θ/baseline、各Adam、samplers、四rank/source RNG、物理迁移/前端身份齐全。G cursor与Adam步38349/bank9冻结。A180为720条件，每task10次，每B4 seen2+novel2；B18为72条件/每task一次，各路径均actor_next_update=u−1、五层conditional_continue、query16、.001代价及当前实际query/STOP/行为/final信用，未实践G无query/credit，source未更新。没有重训pair/A/B已保存部分，B1–17仅恢复，出口唯一B18。

B阶段信用流仅作训练读数：seen36的MT均值.590278，末prefix expected query均值.759936；source见/G未训36为.734375→.749809。不同θ采集核不能拼为single-checkpoint资格，后者不是36独立新操作。原首G72整行分析不重复。

400固定task/state/video映射与旧400一致，每task50不同教学；原场景/env及完整policy-noise共同前缀配对通过，held最终state未用于该H实践。396新G行有真实initial_proprio/5x7 finite动作，4复用MT行和旧MT400缺initial_proprio，保留该限制，不能宣称全部400有观测级双边比对。官方单冻结LoRA与Writer退出/0held梯度消费者沿冻结原核。

新增有效保存交互：pair1260episode/283579步（36MT+72G实践+1152query，seen1152只读）；B360实践+3392query=3752episode/816654步，MTquery1152只读复用；held1988实践+396新最终=2384episode/682258步。合计7396episode/1782491环境步，含正常dummy。400旧MT-H1/114908步与4旧MT最终引用为历史依赖，未当新episode收费。失败部分未落盘的实际reset/settling/步骤额外未知，原GPU尝试、日志和费用全部保留，不写0或把保存总量称精确physical starts。

费用34.272177007GPUh，18份GPU收据含三失败2.242225011GPUh；CPU硬headroom拒绝、旧归档session投递失败、主动owner交接均作为原失败保留。首实际GPU owner05:09:44Z→最终14:42:31Z为9.546h，不重置08:24恢复时钟；超过原整批5–8h预估，保留重复renderer恢复/调度空档、较长B及formal实际共享资源耗时。GPUh无硬上限，无新训练/矩阵/自动后继授权。

完整CPU核验/逐批信用/费用/原行指针：`acceptance.json`；formal原件：`validation/RL_00000018/correct/results.json`；checkpoint：`checkpoints/RL/update_00000018`。原运行clean pushed detached bf2c88b0，训练与评测身份分别沿收据保留。§12独立两臂的原末点读出继续其唯一owner/整批唤醒，不读取其中途分数或增加等待者。

Git交付：main `09902fb1a9f7b82f5efe0d61920d156a02e260a7`，已push origin/main且clean；根三状态及research_history已串行更新，运行来源仍bf2c88b0。实际A/B峰值reserved13.340/18.711GiB；formal实际worker201=16slot，其余五worker=8slot，原件`physical_runtime.json`，未以低显存或world4重新收紧整批上限。

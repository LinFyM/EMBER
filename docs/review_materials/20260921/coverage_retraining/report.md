# EMBER 覆盖重训：正式证据报告（编制中）

> 状态：Writer 训练与 correct/other Validation 已完成；MT-BC 训练和 Writer 视频因果对照仍在后台运行。本稿只记载已经完成且通过正式 exit、400 行及 worker 检查的结果。Test、FT、RL 和外部比较尚未执行，不从本稿推断其结果。

## 研究问题与协议边界

本轮检验：在冻结 Source-71、同一 source normalization 和原单教学相机 K=1 Writer 架构下，改用经过语义覆盖审计的任务划分及 12 个辅助任务 fresh 训练一次 Writer 和一次 MT-BC，是否获得足够的零交互任务能力和视频特异性。Source 不重训，旧 Writer/MT-BC 权重也不续训。

新协议把目标40任务固定为每个 suite 的 6 train、2 validation、2 test，合计24/8/8；训练还包括 12 个审计过的 LIBERO-90 辅助任务，因此两种新模型各训练36任务。明确任务 ID 和源文件见 [protocol.json](../../../../configs/libero_24_8_8_coverage_v1/protocol.json) 与 [coverage.md](../../../../configs/libero_24_8_8_coverage_v1/coverage.md)。Writer 梯度使用每任务 demo0..45，MT-BC 使用0..49，这项示范数量差异在比较中保留。Long9 的 cup→microwave、Goal6 的 cream cheese On bowl、Long8 的 stove turn-on 来源及 Spatial9 的 cabinet-top 方向仍是覆盖审计边界；不把原语支持写成完整 held task 监督。

旧冻结 Test 的 Source/MT-BC/EMBER 为 **78/74/82／400**，EMBER−MT-BC 仅 **+8／400**，未达到至少 **+40／400** 的门槛；这是正式负结果，见[旧报告](../../20260920/test_capacity/report.md)。旧 Test 已经使用并参与后续讨论。新划分受到该结果与语义覆盖审计影响，所以本报告不会声称整个项目历史上从未查看 Test，也不会把两种划分的分数合并或当成同一 held 样本的独立重复。

## 完整 Validation 与选点

Source 在新 Validation8 为 **51/400**。Writer 每200次更新完成一个 correct400 节点，九个完整节点如下。预注册的持续下降规则在1800节点触发；唯一 correct 最高点为1000节点 **117/400**，没有 other 破同分需求，已冻结该 checkpoint。所有数字均为 single-checkpoint、8任务×50初态的完整结果。

| Writer 更新 | 200 | 400 | 600 | 800 | **1000（选中）** | 1200 | 1400 | 1600 | 1800（停止） |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| correct 成功／400 | 110 | 92 | 115 | 87 | **117** | 80 | 97 | 111 | 92 |

相对于新 Validation 的 Source，选中 Writer 为 **117 对 51／400**，逐行配对保留/获得/丢失 **36/81/15**，成功任务覆盖均为 **4/8**。提升集中于 Object1（1→43）与 Long1（5→27），Spatial3 为0→8；Goal6 为42→39、Spatial6 为3→0，Long9、Goal3、Object6 仍为0。该总分优势不代表八个任务都改进。8任务簇、各50配对初态的20,000次 bootstrap 给出 +16.5 个百分点点估计与 **[−1.0，+39.0]** 百分位区间；仅一个训练 seed，区间不能证明跨训练 seed 稳定性。

选中 Writer 同任务换视频 other400 已完成 **119/400**，与 correct117配对保留/获得/丢失 **100/19/17**。两臂各任务50条视频无放回，逐行视频也不同；other−correct 点估计 **+0.5** 个百分点，任务簇 bootstrap 95%百分位区间 **[−2.0，+3.25]**。此结果支持同任务视频条件更换后总体能力未出现严重下降；wrong/shuffle/reverse 仍需确定视频内容与顺序是否有必要增量。

MT-BC 的完整曲线和选点、选中 Writer 的 wrong/shuffle/reverse、各自正式材料路径、完整失败案例将在后台作业结束后补入。

## 成本与限制

Writer 的1800次更新循环实测合计 **17,684.20 秒（约4.91小时）**，平均 **9.825 秒/更新**，四卡最大记录 reserved **22.803 GiB/卡**；累计201,600查询，其中151,200条跨 episode 主查询、50,400条同视频教学查询。该时长不含模型物化、Validation、启动和阶段切换。选中 Writer correct400 的纯评测墙钟为873.25秒，other400为871.91秒，均使用4卡×3 worker。物化的准确墙钟尚无正式计时，不从文件时间戳推造数值。

本轮 Writer 按完整 correct Validation 选择，即便1000后的节点出现反弹也未改早停规则。其高分集中于少数任务，视频因果对照和 MT-BC 仍是资格判断的重要部分。新 Test 严格等待模型与视频资格冻结后才运行；若 EMBER−MT-BC 少于40/400，即停止 Test 视频 controls 与所有后继 FT/RL/外部比较。

## 原始材料（待总表完成后封装）

当前正式原件保存在 `/data0/user/ymdai/ember_runs/coverage_retraining_20260920`：`writer_selection.json`、`method_freeze.json`、`writer_validation_history.json`、`evaluation/{source_validation,writer_00001000,writer_00001000_other}`、`analysis/writer_validation_nodes.csv`、`analysis/source_to_selected_writer_1000_paired.json`、`analysis/writer_1000_correct_vs_other_validation.json`、相关逐任务CSV、bootstrap JSON、绘图源码与SVG/PNG。最终提交时把报告所用小型图表及CSV一并封装，并列出全部正式 checkpoint、exit 与 manifest 的溯源路径。

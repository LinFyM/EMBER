# 可直接发送给专家的讨论材料

我们已完成你最后修订的“经最终LoRA的同视频教学”候选，以及之后追加的后续趋势、匹配视频controls和双相机。请依据以下完整事实给出研究判断与一个最有价值的下一步，不只复述分数，也不要默认所有组件都必须保留或整套重做。

目标仍是exact language＋action-hidden视频在rollout前一次生成完整38-target LoRA，从未见初始化立即闭环。我们接受约130–140的能力水平，但希望视频相对语言／静态prior有更清楚的必要增量，并保留已有能力。本轮尚无learned language-only/static-prior参照，不能把source50当成该参照。

实际图保留A的Core、Core条件化P、中心化/AdaLN、八组共享完整A/B输出；P有两个重复块读取完整H50、真实相邻E和时间关系。三组读取Meta与整个Writer fresh联合训练，source冻结。每次四task等权；每条件21个跨episode主FM＋7个同视频tau=1五步教学query，两项均值按1＋1/3相加。两项经同一套最终LoRA更新全部Writer/三Meta，没有P-only额外梯度或独立教学头。

匹配消融保持图、21＋7数量、loss、噪声、优化与预算，只把辅助query改为同task另一episode。原窗口均1500，后来各用完整optimizer/scheduler/sampler/rank RNG/world2状态延长到2100，尾LR固定2.959936e-5。双相机是额外fresh1500同视频臂，只添加同步eye-in-hand教学RGB；其余科学配方和全部6000事件相同。

全部预登记节点如下；validation各400，train各96：

| 更新 | 同视频 /400 | 消融 /400 | 双相机 /400 | 同视频 train /96 | 消融 train /96 | 双相机 train /96 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 900 | 149 | 125 | 117 | 50 | 59 | 52 |
| 1200 | 174 | 136 | 108 | 67 | 60 | 64 |
| 1500 | 165 | 147 | 108 | 65 | 62 | 67 |
| 1800 | 160 | 136 | 未运行 | 64 | 65 | 未运行 |
| 2100 | 158 | 159 | 未运行 | 67 | 67 | 未运行 |

主组other900/1200/1500为140/159/165，2100为156；消融other1500为155、2100为161；双相机fixed1500 other为105。主组原相邻qualification选定1500后冻结，续训和双相机没有另选checkpoint。主组1200的174不是被挑出的selected结果。

匹配消融→同视频correct差值CI依次为[0,12.5]、[2,21]、[0.25,9]、[0.75,12.5]、[-5.75,4.5]pp。主组1500→2100correct的R/G/L=132/26/33、CI[-5.75,3]pp；other165→156的CI[-5,-0.25]pp。消融correct147→136→159，末点相对1500的CI[-3.25,10.25]pp。双方held train-action FM持续微降；同视频原窗口优势未持续到末点，不证明消融优胜或等价。

1500固定视频检查：

| 固定1500（各400） | correct | 同task换视频 | 错task视频 | 打乱 | 逆序 | 共同零LoRA/source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 同视频 | 165 | 165 | 124 | 113 | 113 | 50 |
| 匹配消融 | 147 | 155 | 122 | 104 | 86 | 50 |

两组shuffle/reverse组内差值区间均支持敏感性，但针对教学增强的匹配差值之差如下：

| 参照 | control | 同视频 gap | 消融 gap | 差值之差 | 95% CI (pp) |
| --- | --- | ---: | ---: | ---: | --- |
| correct | wrong | 41 | 25 | +16 | [-4.50, +12.00] |
| correct | shuffled | 52 | 43 | +9 | [-5.50, +9.00] |
| correct | reversed | 52 | 61 | -9 | [-11.25, +3.50] |
| 同task换视频 | wrong | 41 | 33 | +8 | [-7.50, +9.75] |
| 同task换视频 | shuffled | 52 | 51 | +1 | [-6.50, +6.25] |
| 同task换视频 | reversed | 52 | 69 | -17 | [-12.75, +1.75] |

六个DID区间都跨零。主组task3 correct/other/wrong/shuffle/reverse=35/32/0/6/0，Long则23/21/33/23/17；效应并不均匀。task23（开抽屉放碗）和32（开灶放壶）主组直到2100仍没有成功，task1到2100才有一次；没有可靠任务能力时不作强顺序理解主张。

双相机读出：

当前同视频教学配方加入同步手眼RGB没有带来验证能力收益。双相机900/1200/1500 correct为117/108/108，对单相机149/174/165；fixed1500 other为105，对单相机165。三个correct差值的任务簇95%CI都低于零，末点为−14.25pp、CI[-27,-4]pp，other为−15pp、CI[-27.75,-2.75]pp。本轮不能把双相机补充视为有效改进；该结论限于当前配方、预算与一个训练seed，不表示所有双相机方法无效。

训练任务的双相机成绩为52→64→67/96，单相机为50→67→65；固定train-action FM从0.104100628降到0.098245136，末点与单相机0.098259576接近。因此不能用训练任务提升或FM下降替代未见任务闭环，也尚未唯一定位表示、泛化或优化方面的原因。实际事件、查询噪声、输入相机、source冻结、完整checkpoint与四组梯度核对均通过。

1500单→双correct保留93、获得15、丢失72；S/O/G/L由35/59/48/23变为10/43/36/19。双相机correct breadth为6/8，仅因task1出现一次成功；other breadth为7/8，但task1与23也各只有一次，task32仍为零。覆盖计数增加不能掩盖既有任务能力下降。

双相机自身1200→1500虽同为108，仍保留78、获得30、丢失30；correct→other108→105保留76、获得29、丢失32。总分接近不等于成功集合稳定。没有双相机wrong/shuffle/reverse检查，不能继承单相机1500的顺序敏感性结论，也不据此自动追加训练或相机扫描。

| 比较 | 成功数 | 净差 | 95% CI (pp) | 保留/获得/丢失 | churn | Jaccard |
| --- | --- | ---: | --- | --- | --- | ---: |
| 900：单→双相机 | 149→117 | -32 | [-14.50, -1.50] | 91/26/58 | 84 | 0.520 |
| 1200：单→双相机 | 174→108 | -66 | [-31.50, -4.25] | 91/17/83 | 100 | 0.476 |
| 1500：单→双相机 | 165→108 | -57 | [-27.00, -4.00] | 93/15/72 | 87 | 0.517 |
| fixed1500 other：单→双相机 | 165→105 | -60 | [-27.75, -2.75] | 88/17/77 | 94 | 0.484 |

历史A900/1200/1500为140/135/112；与本候选有读取、辅助监督、LR尾段多处差别，不能据此识别单模块收益。旧frame-set900/1200为142/118，对A140/135；A900 train诊断关P58→34、换其它Core仍成功56/96，只证明已有P通路有用，不证明Core可删、晚入口瓶颈或新方法已成功。历史短预算／局部诊断边界继续有效。

数据口径：固定24 train/8 validation/8 test；本轮无Test或held梯度。validation逐task50初态＋全部50视频各一次，配对相同state/video/env-policy RNG；controls始终保留目标exact language，仅换或重排真实RGB后完整重编码。CI为八task簇bootstrap20,000次、seed20260915，单训练seed、未多重校正。40个面板/12,048条新闭环及全部实际事件、source冻结、checkpoint恢复和worker完成已核对。

请重点回答：

1. 哪些关于同视频教学、反复完整H/相邻视觉读取的原假设得到支持，哪些应降低置信度？请区分匹配消融能识别与不能识别的因素。
2. 原窗口优势、2100趋近、FM下降和成功集合交换，允许怎样解释学习与保持？请不要把未识别的内部原因直接当成已证明根因。
3. 两组均顺序敏感但DID没有明确增益，该如何修改“视频中的操作证据必须通过LoRA被保存”的论断和下一步验证标准？
4. 双相机结果最小支持什么，是否值得固定输入选择？不能把相机增量唯一归因于遮挡、接触或过程理解。
5. 若只允许下一次改变一个主要科学变量，你建议什么？请给完整数据／信用路径、与已失败近似路线的实质差异、正反结果的具体预测、必要配对参照与有界停止条件；不要默认rank/LR/seed扫描，也不要把不同checkpoint或历史路线优点拼成一个结果。

请综合已完成历史，不因这轮局部结果禁止整体重构或强制只改P；也不要把尚未测过的设计当作已失败。shuffled/reversed仅作冻结点最终判读，不作为新loss、选点或架构修正依据。建议供Owner讨论，本材料不构成新实验启动授权。

附件入口：[总报告](https://github.com/LinFyM/EMBER/blob/main/docs/review_materials/20260919/final_report.md)、[方法／输入示例](https://github.com/LinFyM/EMBER/blob/main/docs/review_materials/20260919/discussion_context.md)、[续训](https://github.com/LinFyM/EMBER/blob/main/docs/review_materials/20260919/continuation_report.md)、[匹配controls](https://github.com/LinFyM/EMBER/blob/main/docs/review_materials/20260919/ablation_controls_report.md)、[双相机](https://github.com/LinFyM/EMBER/blob/main/docs/review_materials/20260919/dual_camera_report.md)、[单相机逐task表](https://github.com/LinFyM/EMBER/blob/main/docs/review_materials/20260919/discussion_tables.md)、[全体逐task指标](https://github.com/LinFyM/EMBER/blob/main/docs/review_materials/20260919/per_task_metrics.csv)、[全体逐suite指标](https://github.com/LinFyM/EMBER/blob/main/docs/review_materials/20260919/per_suite_metrics.csv)、[全体原始精简行](https://github.com/LinFyM/EMBER/blob/main/docs/review_materials/20260919/all_completed_rollouts.csv)。

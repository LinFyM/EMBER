# 旧专家包之后的事实、比较边界与原件

本页补齐9月11日旧咨询包之后的记录。数值来自已存在的结果，本次只导出、复算和整理；不报告新的训练或rollout。
原科学decision字段描述当时判断，不能覆盖当前Owner的“先恢复特异性、暂不强制绝对性能”阶段目标。

## 1. 面板口径先分开

| 名称 | 实际样本 | 用途与边界 |
| --- | --- | --- |
| validation400 | 固定validation8，每task states0–49；同task同臂50条teacher各一次，跨checkpoint固定映射 | 未见task闭环；不同阶段的checkpoint各自保留，不取union |
| 轮换视频train96 | train24，每task states32–35分别使用held teacher46–49各一次 | 已训练task的新视频/初态；不能当未见task迁移 |
| 固定视频机制96 | train24，states32–35全部用teacher46；other用47，wrong使用登记donor | 冻结开发输入干预；单task四初态复用视频，不冒充整轮50视频无放回 |
| 路径替换32 | 原已登记八task×四初态，固定语义S或过程P4进行替换 | 区分作用路径的有限干预；混合表示未经共同训练，存在分布变化 |
| 独立动作诊断 | train24合法留出动作；相同queries、flow time/noise下比较各视频条件 | 无梯度的动作函数证据，不等于闭环或held-task资格 |

因此旧C200的轮换视频49/96与固定视频42/96都成立；S对应52/51，off400对应56/53。
不要从数字差异推断评测不一致，也不要在不同视频日程间计算所谓严格配对差额。

## 2. R/C/S与监督续训

R/C/S都使用train24、普通正样本FM、agentview、Action Meta、完整H和独立native D。每update四suite各一task，
每task两个独立K1条件各32queries，条件权重1/8、task权重1/4，总256queries。R保持off图，C加入语义条件化过程消费，
S使用无序完整帧集合。三者fresh，不把两条视频拼成K2或平均LoRA。

| 配方 | validation100/200/300/400（每点400） | 轮换视频train96 | 实际观察范围 |
| --- | --- | --- | --- |
| R | 37 / 83 / 63 / 85 | 200=46，400=56 | 400更新、102400queries、3200条件调用、383种task-video |
| C | 43 / 50 / 未执行 / 未执行 | 200=49 | 200更新、51200queries；未证明收敛 |
| S | 59 / 45 / 未执行 / 未执行 | 200=52 | 200更新、51200queries；不是单帧static |
| 旧off7参照 | 200=108，400=126 | 200=40，400=56 | 每task原一条K1×64queries；参照在旧包 |

R400与旧off400动作query预算相同，但视频条件分配不同。更多条件没有带来终点迁移收益；该结果不检验更多独立meta-task，
也不证明普通FM或整个有序过程函数类不可行。

旧off400完整保留学习状态继续至500/600，validation为126→73→54，轮换视频train96却56→64。
400→600保留43/新增11/丢失83，churn94；已有能力与新获取的相互作用仍需解释，不能只由loss或总分命名根因。
该续训保留exploratory lineage，没有重标为formal fresh。

原件：[首轮汇总](analysis/consumption/first_round_summary.json)、[R续训](analysis/consumption/r_continuation_summary.json)、
[off续训](analysis/consumption/off_continuation_summary.json)、[旧off曝光](analysis/consumption/old_off_exposure_reference.json)、
[query对齐](analysis/consumption/old_off_query_alignment.json)。逐条rows见[学习面板](panels/learning/)，原配置、学习曲线与曝光见[model_records](model_records/)。

## 3. C完整冻结输入诊断

| 固定视频机制，每臂96 | C100 | C200 | 旧off400参照 |
| --- | ---: | ---: | ---: |
| correct | 32 | 42 | 53 |
| same-task other | 31 | 48 | 60 |
| same-suite wrong | 27 | 46 | 59 |
| cross-suite wrong | 33 | 46 | 59 |
| shuffled | 29 | 45 | 55 |
| static first | 29 | 49 | 58 |
| 匹配无序S | 30 | 51 | 不适用 |
| source | 15 | 15 | 15 |

C200比source多27，task-bootstrap差额区间为正；比static少7，区间[-.125,-.031]。
C100→200正确增加10，static增加20、S增加21。取得训练task能力并未同时形成正确过程净收益；两个节点不证明C收敛。
wrong保留目标语言，shuffle先重排真实RGB再完整重算；呈现索引单调，原索引只保存为provenance，模型不能借原索引还原顺序。

登记与方法：[registration](analysis/mechanism/registration.json)、[输入变换脚本](methods/mechanism_panel.py.txt)、
[统计方法](methods/mechanism_analysis.py.txt)。[全部配对比较](analysis/mechanism/behavior_summary.json)与
[C200逐条rows](panels/mechanism/c200/)可直接复核；旧off完整九臂复用[旧包](../20260911/panels/diagnostics/best_model_video_controls/step400/)。

## 4. 表示与功能路径：已观察到什么

同一八task面板中，重复首帧仍产生正常视频.692–.935倍的中心化P4，说明时间/窗口/递推也能制造时变表示。
这不是“69%–94%的行为来自假动态”的因果归因。表示变化不等于视频真实运动，更不等于运动被策略有益使用。

固定正确语义S后，correct/替换wrong过程/替换static过程/zero过程/替换错误语义的闭环为15/18/15/14/21，各32。
过程static替换保留12/新增3/丢失3；有限面板及混合表示的分布差异限制外推，不能把S、D或P4命名为唯一根因。

| C200固定动作面板 | mean FM |
| --- | ---: |
| correct | .112128 |
| same-task other | .112159 |
| same-suite wrong | .112430 |
| cross-suite wrong | .113185 |
| shuffled | .112538 |
| static first | .112543 |
| S200 | .111931 |
| source | .152494 |

correct对source的误差下降发生在24/24task；相对wrong/shuffled/static的差额远小于这项共同收益。
correct对shuffle有微小平均优势，不能说完全没有顺序信息，也不能据此宣称已经恢复有益过程。
固定S时换static过程仅增加FM .000044；监督函数中的微小差异不能直接替代行为。

原件：[表示trace与路径manifest](analysis/paths/manifest.json)、[路径行为](analysis/paths/behavior_summary.json)、
[路径rows](panels/paths/c200/)、[路径方法](methods/path_intervention.py.txt)、
[主动作表](analysis/mechanism/functional_summary.json)、[主动作配对](analysis/mechanism/functional_comparison.json)、
[路径动作表](analysis/paths/functional_summary.json)。主面板12288与路径2048次预测合计14336，无模型更新。

## 5. 无变化参照：局部数学修正与真实结果分开

局部更新把真实历史GRU响应减去“过去端使用当前帧相同内容”的GRU响应，保持实际gap、角色、窗口长度、顺序与参数相同。
不新增backbone、不输入错误标签、不惩罚错误条件，不更改C其它主要结构或普通FM。恒定输入的中心化过程应归零，语义S仍保留。
真实native/BF16 profile上静态过程约为真实视频的0.9%，source无梯度，真实变化有梯度；这只验证一个表示性质和工程机制。

fresh100/200训练完成，累计51200queries；独立留出FM由.151451降至.109323，24/24task改善。

| 固定训练机制条件 | 100步 /96 | 200步 /96 |
| --- | ---: | ---: |
| correct | 36 | 45 |
| same-task other | 未登记 | 42 |
| same-suite wrong | 未登记 | 41 |
| cross-suite wrong | 未登记 | 37 |
| shuffled | 未登记 | 41 |
| static first | 未登记 | 40 |

200的correct对same-suite wrong/cross-suite wrong/shuffled/static差额分别+4/+8/+4/+5；task-bootstrap成功率差95%区间为
[-.0521,.1354]/[-.0417,.2083]/[-.0521,.1354]/[-.0625,.1563]。全部包含零，correct相对旧C仅+3，不能称机制修复。
100→200保留22/新增23/丢失14，churn37/J=.37288，总分上升不等于成功集合稳定。

200 validation=34/400，source47、旧C20050；Spatial/Object/Goal/Long为4/24/5/1，breadth5。
对source保留6/新增28/丢失41；task26从source41降2，task11从source5升24，存在任务间得失。
绝对结果继续公开，但不把它改成当前阶段的预设硬门槛。

**100 validation未完成，300/400未执行。** 暂停来自Owner指令，不是已证明收敛；新包没有为它们补分数。

原件：[完整模型合同](model_records/nochange/run_contract.json)、[profile](analysis/nochange/profile_summary.json)、
[动作诊断](analysis/nochange/action_diagnostic.json)、[100摘要](analysis/nochange/step100_summary.json)、
[200配对与区间](analysis/nochange/step200_summary.json)、[暂停记录](analysis/nochange/owner_pause.json)、
[完整学习与行为rows](panels/learning/nochange/)、[训练机制rows](panels/mechanism/nochange200/)。
数学合同见[无变化参照设计](../../video_change_reference_design.md)，真实统计方法见[脚本快照](methods/nochange_analysis.py.txt)。

## 6. 尚未建立的结论

这些记录没有识别一个覆盖全部失败的唯一根因，没有证明纯FM不可能，也没有证明某个辅助目标、RL、D共享或更大task覆盖必然解决问题。
它们更没有证明应该恢复v5.2。下一步应综合这些事实提出完整理论及设计，由其可区分的预测决定需要哪些验证。
新包7216条outcome中，4480为R/C/S及两项续训、1472为C新诊断、1072为无变化参照已完成行为，另192为复用参照；
该计数表示本次发布覆盖，不是本次重新执行量。

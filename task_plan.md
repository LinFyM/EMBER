# EMBER task plan

## 已完成：仓库整理与远程专家咨询材料（2026-09-11）

Owner要求整理仓库、推送远程，并准备说明现状、分析与看法的prompt，咨询专家如何改进现有方法和继续推进。专家只能看到远程仓库。本次不启动训练/评测、不正式采纳新架构、不向专家或其它人代发消息。

交付内容已完成，远程版本以本文件所属提交为准：

- 更新README，明确main、126候选及历史D绑定实验的区别。
- 将本文件与progress收敛为当前状态；逐次执行历史由research_history、findings、原件和已推送Git保存。
- 在`docs/review_materials/20260911/`提供最新学习与机制证据、可复算逐条结果、原配置/commit、行为图、证据等级和专家咨询prompt；保留旧9月7/8日材料各自的历史用途。
- 清理明确可再生缓存和已确认重复临时文件。保留唯一checkpoint、formal evidence、数据、smoke/profile证据及所有权不清的工作。
- 核验导出保真、汇总、引用及任务范围diff，提交并推送main。

下一步由Owner转交咨询prompt并讨论专家反馈，不自动恢复科研执行。

## 科学状态与执行边界

现有canonical设计仍为`docs/horizon_relation_video_writer_design.md`，没有采纳新的active design。当前最强Horizon候选为Compiler额外语言query关闭、独立D、单视角的init7 macro400，validation126/400；另一初始化119/400。既有训练与冻结诊断已结束。最新九臂补测的完整/另一正确/同suite错/跨suite错/乱序/倒序/首/中/末静态为53/60/59/59/55/52/58/53/53（各96），见findings§49。

当前持续停在正式方法修改/采纳和下一轮训练前，等待Owner复核。GPU诊断授权仅覆盖已登记且已完成的实验，不从过去的自主执行文字恢复任务；禁止再用完整400步重训或拆段绕过此前分析预算。专家建议属于待讨论意见，不是执行授权。Test保持封存。

## 已登记但未执行的后续事项

- Compiler额外query关闭、保留独立D候选的后续学习与保持；两个初始化200→400仍增长，尚无平台证据。
- 真正训练同步双视角，再与单视角同口径比较；输入smoke不算双视角效果。
- 若正式续训，每100步保存完整checkpoint并做strict paired400；具体区间须在获准推进后登记。
- Compiler关闭×D绑定的交互仍未识别，不将all背景绑定负结果外推全部组合。
- learned language-only/static prior参照及完整视频的真实条件增量需要正式设计；冻结替换不能冒充学习后的baseline。
- 功能保持约束、prior加视频残差等仅为未验证设想，须由专家独立审视证据和历史近等价失败，尚无采纳决定。

上述事项不是本轮咨询问题的全部答案；需要专家具体说明哪些修正有证据支持、针对哪个最早失效接口、为何区别于历史失败、怎样用最少有信息量的实验裁决。

## 历史入口

- 持久结论与原件索引：`findings.md`、`docs/research_history.md`。
- 本轮既有诊断注册：`docs/horizon_causal_learning_plan_20260909.md`，文内旧授权/待办只代表当时。
- 整理前完整计划：[043b58ca的task_plan](https://github.com/LinFyM/EMBER/blob/043b58ca3f1f3e7ed699b876f4964be96654c7e5/task_plan.md)。保留历史，不在当前文件继续堆叠失效执行段落。

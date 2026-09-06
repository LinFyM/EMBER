# EMBER task plan

更新时间：2026-09-06 16:06 CST。

## 当前目标与启动授权

owner最新明确同意实际开始，要求重新建立goal，终点为最终目标完成；权限内的问题由agent自主解决并连续推进。
新的unbudgeted goal已建立。本授权取代今天上午的讨论暂停；不恢复此前弱A2的待执行清单。
最终为validation8 strict paired single-checkpoint correct >145/400，加相邻稳定、跨视频、breadth/四suite/Goal/Long与冻结后视频因果资格；
方法冻结后按32/8合同fresh最终训练和test。train24 SFT历史109与旧Writer143是实质参照。

## Active design与当前阶段

- Active design：`docs/joint_process_policy_writer_design.md`，当前为完整LoRA共同过程—策略生成。
- 复用93540ff1的P/Q及完整native-response读取；由learned heads联合生成38-target全部A/B，首选rank16，无独立carrier。
- 初期保持现有监督、数据、权重与主要P/Q配置；输出重构的耦合变化不能伪称某个单组件的因果优势。
- 先实现并做最小真实forward/gradient/materialization/吞吐验证；现有四任务作短学习对照，73-task承接有行为反馈的主训练。
- 小面板判断投入，最终选择只用预登记strict400；接近强基线/目标后先一次完整评测，再展开相邻和同任务另一视频。
- 原A2 component已sealed，random只完成至48且此前被中断，未完成/未恢复；已有checkpoint保留，不自动续跑或执行原六套400。

## 推进顺序

1. [x] 重建goal并同步最新授权、能力参照和完整输出方向。
2. [x] 在独立实现worktree复用P/Q，替换唯一输出接口、同步训练与物化合同并退役旧默认路径。
3. [x] 最小真实图与最长样本验证、四任务短学习及四组配对闭环完成，64为64/62，明显超过旧P/Q与A2。
4. [x] 同完整图73任务128updates及四点Panel-B已完成，完整采样审计通过；meta与target学习差异仍需闭环裁决。
5. [x] 四组完整screen80为15/19/19/19，后半能力窄且停滞；本mixed实例不扩strict400，完整证据保留。
6. [ ] 同图同18target的128步训练与四点新视频功能诊断已完成，目标学习恢复；四套primary bank生成中，随后完成screen80闭环判断。
7. [ ] 完成强候选的同口径SFT比较、相邻与跨视频strict400及同图fresh候选裁决；冻结selected后补完整视频因果controls。
8. [ ] 方法冻结后完成规定32/8 fresh训练与最终test，交付完整科学证据。

## 判断原则与历史入口

- 一次non-pass只淘汰实际检验的组合；先对照专家原文和已定位接口，不能用任意seed/LR/rank小扫替代分析。
- closed-loop能力决定方法和投入；内部功能loss只负责区分训练学习、同task新视频与未见task迁移。
- 明确坏结果不无限续训；有希望的变化继续相邻验证，不把固定更新预算称为充分训练或收敛。
- 同图component-init与fully-random fresh保留为Final候选；negative controls只在selected checkpoint冻结后使用。
- 训练等待期间优先准备下一科学节点，只在没有相关实质工作时整理可中断的workspace任务。
- 当前执行证据见`progress.md`，跨轮结论见`findings.md`，旧路线和专家/实验索引见`docs/research_history.md`；
  旧已完成清单及过时待办由Git保存，不在当前计划中重新激活。

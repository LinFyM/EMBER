# EMBER task plan

## 当前目标与授权（Owner 2026-09-17，整套实验执行goal）

Owner已明确授权：先整理整个仓库及data1可删除资产，再实现已选定的新架构并优化GPU吞吐，
完成与匹配v5.2的训练/评测；结果不佳时，仅在有具体证据支持的改进空间时修正并重新训练，
若充分证据显示没有可行改进则停止，最后详细汇报整套实验。本授权替代上一轮“只分析、不实现/实验”的限制。
这是一项持续到实际实验与交付完成的goal，不以建立计划、启动训练或单个分数作为完成。

## 计划与完成标准

1. **进行中：仓库与空间整理。** 审计tracked源码、tests、scripts、configs、docs以及ignored运行资产；
   在strg01核实data0/data1独立quota与实际用量。按keep/consolidate/delete/review分类，核对调用与运行生命周期后删除。
   保留canonical数据/source、唯一checkpoint、原始正式证据及专家论证；压缩过时状态叙述，历史通过Git/research_history追溯。
2. **待完成：统一架构实现与效率。** 按[active design](docs/v52_evidence_based_writer_design.md)整体替换canonical Writer：
   j9/j18各两层同构Z/H联合块、一次双写回、两层连续参数decoder和共享完整A/B heads；三Meta fresh联合纯FM。
   复用采样、真实FM/VJP重放、checkpoint及evaluator。通过有效梯度、原生接口、最长视频、数值稳定和完整恢复检查，
   用实际LoRA/s、queries/s及峰值选择physical batch/并行方式，不删视频、相机或H位置换速度。
3. **待完成：首轮训练与匹配比较。** 冻结可审阅的训练/预算/节点/停止合同；从clean pushed detached trees运行。
   新架构与fresh v5.2匹配raw1000 source、双RGB、fullH实际learned read、train24、采样事件、FM和优化时钟。
   正式节点使用single-checkpoint correct400和train96，比较per-task/suite、breadth、相邻retained/gained/lost与churn。
   性能仍以>145及相邻资格为长期目标；不以内部指标代替真实能力，视频相对语言/静态增量另以有效参照验证。
4. **待证据裁决：有依据的修正。** 区分工程错误、获取不足、迁移回落和合理科学non-pass。
   只有具体失败接口、可检验改进假设及预先登记的新比较支持时，才实施集中修正并fresh重训；
   不做无依据的rank/scale/seed/LR/dtype小扫，不用无限续训挽救明确坏结果，也不把一次局部失败扩大为全路线推翻。
5. **待完成：全套裁决与交付。** 完成必要的正确视频稳定性、same-task-other、learned language/static参照及冻结后的最终controls；
   汇总能力、视频作用、保持、成本、负结果与未解决问题，核对资产保留、单一运行面和Git交付，向owner详细报告后结束goal。

## 研究与执行边界

- Active design为[统一Writer设计](docs/v52_evidence_based_writer_design.md)；其结构已选定，运行参数只在真实profile后封存。
- 固定source71/train24/validation8/test8、信息墙、normalization与official评测不变。无held梯度、无RL、Test仍关闭。
- 初始建议窗口为2400更新（9600条件/201600queries），600起每300更新正式主面板，每100保存完整恢复状态；
  具体成本与运行资源在launch前记录，不把建议预算冒充已分配资源或已启动运行。
- 首轮整体方法比较不自动识别中层写回的独立贡献；若要提出该因果主张，需同一统一尾端且无j9读写的fresh比较。
- 最终wrong/no-video/shuffled/reversed不进入训练、选点或架构返工。改进依据来自允许的主面板、训练证据及真实工程合同。
- 单节点最多6张真正提高吞吐的A40；每次launch前双节点live检查，尊重其它任务，正式exact-resume锁拓扑。
- 大资产复用canonical根。新大输出优先使用已有data0运行父目录，仍须其独立quota/峰值核价；data1清理不意味着可忽略配额。
- 不恢复旧A3000评测、旧C或其它无关实验。旧A/SFT为历史参照，新v5.2对照fresh并匹配当前合同。

## 已完成工作与历史入口

- 2026-09-17证据审计保留46组证据及12组bank中间路线；[审计原件](docs/v52_evidence_audit_20260917.md)。
- 首版中层桥设计保存在Git `426dc5be`；第二轮统一设计及数学/接口复核在`176759a7`，跨轮结论见findings§115–116。
- 旧A已完成3000训练，最新完整闭环为2700；SFT400/425/450已完成。其完整历史、原件和原暂停状态见
  [research_history](docs/research_history.md)、findings§107–114及`runs/analysis/source_alignment_20260915/`。
  旧任务的未完成清单已由本次明确goal替代，保留事实，不保留并行执行路线。

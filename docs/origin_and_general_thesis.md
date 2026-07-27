# Origin and General Thesis

状态：动机与思想来源说明。若与当前实验合同冲突，以根目录 `AGENTS.md` 和
`docs/execution_brief.md` 为准；当前Writer设计与实时执行分别见
`docs/action_forecast_writer_v5_design.md`和`docs/active_session_handoff.md`。

EMBER 的原始动机是把任务描述或示范“编译”为一个可直接使用、又能继续适应的策略状态。owner 最初强调的是：

- language/action-hidden video 提供任务规格；
- Writer 输出 task-specific adapter/LoRA；
- 输出应有 zero-interaction utility；
- source action supervision 和 reward 可训练 Writer；
- 后续目标任务 reward 可继续优化同一 task-local state；
- held 时共享模块冻结。

后来出现的 canonical bank、shared update subspace、soft geometry 和 residual escape 是 assistant/expert 添加的增强假设，不是 owner 原始核心，现已明确移出当前项目。

## 当前具身实例

当前以过滤 exact semantic/composition overlap 后的 LIBERO-90 作为 source-base corpus，
以 LIBERO-Spatial/Object/Goal/Long 的固定 24/8/8、final 32/8 作为目标 benchmark：

- source base 使用过滤后 LIBERO-90 成功 robot actions，随后冻结并由所有方法共享；
- AS-Writer 在目标 source tasks 上从 language + 恰好一条 action-hidden video 学习；
- RL-Writer 与完整AS best分离：从新架构fresh初态做短、task-balanced AS
  cold start，直到24个development-train tasks逐task至少一次official
  random-reset success，再永久关闭action入口并只用source reward学习；
- held zero-interaction Writer 只看 language/video，test-task RL 可再使用目标 reward；
- 最后的 direct target-action reference 在 8 个 test tasks 上联合训练一套 shared LoRA，
  只作 privileged oracle。

## 更一般的研究命题

如果成立，EMBER 支持一个较窄但有意义的结论：任务规格可以被摊销地映射到局部参数初始化，并且这种初始化比统一起点更适合当前任务和后续标准适应。

它不自动推出：

- 任意 embodiment 迁移；
- human-to-robot domain transfer；
- 真机安全；
- universal optimizer；
- shared low-dimensional update geometry。

这些必须由独立项目和证据支持。

# Origin and General Thesis

EMBER 的原始动机是把任务描述或示范“编译”为一个可直接使用、又能继续适应的策略状态。owner 最初强调的是：

- language/action-hidden video 提供任务规格；
- Writer 输出 task-specific adapter/LoRA；
- 输出应有 zero-interaction utility；
- source action supervision 和 reward 可训练 Writer；
- 后续目标任务 reward 可继续优化同一 task-local state；
- held 时共享模块冻结。

后来出现的 canonical bank、shared update subspace、soft geometry 和 residual escape 是 assistant/expert 添加的增强假设，不是 owner 原始核心，现已明确移出当前项目。

## 当前具身实例

在 LIBERO-90 中：

- source tasks 给出成功 robot teacher episodes；
- Writer 可以从这些 episodes 的 action-hidden 视频和 language 学习；
- validation/test 只给 Writer language/video；
- task-local RL 可以使用目标 reward，但不能把目标 action 泄露给 Writer；
- direct target LoRA 用 action，只作 oracle。

## 更一般的研究命题

如果成立，EMBER 支持一个较窄但有意义的结论：任务规格可以被摊销地映射到局部参数初始化，并且这种初始化比统一起点更适合当前任务和后续标准适应。

它不自动推出：

- 任意 embodiment 迁移；
- human-to-robot domain transfer；
- 真机安全；
- universal optimizer；
- shared low-dimensional update geometry。

这些必须由独立项目和证据支持。

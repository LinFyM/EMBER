# Prior MemLLM Lessons

状态：历史方法论来源，不是活动架构或执行authority。当前合同见`AGENTS.md`、
`docs/action_forecast_writer_v5_design.md`和`docs/active_session_handoff.md`。

EMBER 是完全独立仓库，不依赖、不复制 MemLLM Wiki/QA 代码。可复用的只有研究经验：

- 先证明生成状态具有即时功能价值，再讨论后续优化。
- 区分“共享生成器学到了规律”与“单任务直接拟合有能力”。
- 参数 factor 有 gauge/non-identifiability，raw factor MSE 不适合作主目标。
- functional/behavioral loss 比参数距离更接近真实 claim。
- 强 baseline 必须匹配信息、容量、数据和交互预算。
- held surface 必须在 shared-frozen 条件下隔离。
- 失败先区分数据、表示、acquisition、optimization 和 implementation。
- 只保留一条 canonical runner；实验日志和恢复设施服务科学，而不是替代科学。

明确不可复用：

- Wiki/QA 数据管线；
- MemLLM 模型代码；
- 因旧项目方便而形成的接口；
- 未经当前 VLA/LIBERO 证据支持的 bank/shared structure 假设。

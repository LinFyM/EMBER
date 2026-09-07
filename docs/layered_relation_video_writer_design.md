# 旧分层局部关系 Writer 设计入口（已退役）

本路径对应2026-09-07已实现的18层×50H分层图及其旧训练配置；该run永久止于384，不能继续旧672步schedule。
这里保留轻量入口供旧源码/config与历史引用定位，不再保留第二份活动方法正文。

- 当前唯一设计：[过去定向完整 Horizon Writer](horizon_relation_video_writer_design.md)。
- 旧设计完整原文：[e868de525fda0a20c597dee2c7bffe5717f5e2fd中的本文件](https://github.com/LinFyM/EMBER/blob/e868de525fda0a20c597dee2c7bffe5717f5e2fd/docs/layered_relation_video_writer_design.md)。
- 本地恢复读取：`git show e868de525fda0a20c597dee2c7bffe5717f5e2fd:docs/layered_relation_video_writer_design.md`。
- 旧训练与评测证据：[research_history](research_history.md) §17–19；当前授权与接续状态：[progress](../progress.md)。

现有代码仍包含旧图，下一session负责在其可复用基础上替换为新canonical实现；这不构成恢复旧实验的授权。

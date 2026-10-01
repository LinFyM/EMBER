# Configuration authorities

当前Writer使用`operator_read_write_v1/conditional_read_write_fresh_spec.json`，官方与已见任务capture分别为
`conditional_read_write_official_capture.json`、`conditional_read_write_seen_capture.json`。
对应计算、学习、选择与预算见[active design](../docs/designs/conditional_read_write_architecture.md#13-首批实施与完整学习检验2026-10-01授权)，
是否运行及最新状态只看[progress](../progress.md)。

数据authority由`libero_24_8_8_coverage_v1/`、`libero90_nonheld_meta_v1/`、`pi05_source_corpus_v1/`及
`pi05_target_data_v1/`拥有；不同历史protocol保持独立，不能覆盖原始划分或按结果换task。
Source、MT-BC、LoRA和官方评测配置由各自实际CLI及bank reader消费。

同目录中的self_read/context/joint/continuation旧spec仍被`operator_writer/specification.py`递归使用，
且封存bank/source/capture记录依赖其科学身份；保留配置不表示保留旧训练运行面。
`conditional_compilation_diagnostics_v1/`、`relational_support_causality_v1/`、语言内容与首帧等配置
服务封存bank和metadata校验。其余已关闭诊断的`experiment_spec.json`是必要的预注册范围与provenance，
实际结果沿[research_history](../docs/research_history.md)及其formal原件读取。

封存配置中的旧脚本、路径、hash和“下一步”按所登记commit及当时时点解释；不为文档导航改动重新绑定历史模型身份。
重放必须同时检查冻结代码和checkpoint/payload退休记录；配置本身不能证明权重或optimizer仍在。

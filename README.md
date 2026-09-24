# EMBER

EMBER研究把exact task language与action-hidden教学视频，在rollout前一次编译为冻结π0.5 source的一套完整
task-conditioned LoRA，使机器人从未见初始化闭环完成任务。正确视频中的操作内容应贡献真实执行价值。

**五批诊断已完成并独立复核；最新读出干预未支持冻结最后一层修复，下一步分析这些适配作用的学习来源。** 尚无经过验证的统一根因或修复。
实际状态及证据位置见[progress](progress.md)，科学解释和下一步计划见[task_plan](task_plan.md)。仓库收尾已完成，无需重做。

## 阅读顺序与唯一职责

| 入口 | 职责 |
| --- | --- |
| [Owner要求](docs/current_owner_requirements.md) | 理论下界、研究目标、解释深度、方法与协作边界 |
| [当前进度](progress.md)／[当前计划](task_plan.md) | 当前授权、运行快照与科学计划；历史许可不在此持续生效 |
| [Concept](docs/concept.md) | 完整信息流、模块因果职责、数学解释框架与待验证假设 |
| [AGENTS](AGENTS.md) | 科学、数据、评测、资源、工程与Git合同 |
| [Findings](findings.md) | 编号的跨轮发现；结论保留适用范围，最新四臂与冻结诊断结果见§135–139 |
| [研究历史](docs/research_history.md) | 按时点追溯设计、专家讨论、原始证据和复现commit |

新任务先读Owner要求和当前状态，按问题沿历史索引追溯；不把所有旧设计的“下一步”合并成待办。
完整相关历史需要综合，但不要求每次重读全部文件。理论下界不等于优化保证，也不指定公共底座课程。

## 目录职责

| 目录 | 内容与生命周期 |
| --- | --- |
| `src/ember/` | 唯一维护中的Writer、Source/MT-BC、数据及评测实现 |
| `scripts/` | 薄CLI、环境构建、数据封存和结果比较入口；已结束的专用诊断脚本由Git保存 |
| `tests/` | 对当前实现及稳定科学/恢复/配对合同的CPU检查 |
| `configs/` | 显式数据协议、方法配置和审计；不同协议分别保留，不能覆盖旧结果 |
| `docs/designs/` | 有独立科学合同价值的设计/计划，文件头说明当前/历史状态；是否active只看progress |
| `docs/analyses/` | 有独立论证价值的机制分析和审计；不是运行授权 |
| `docs/review_materials/` | 按日期/研究组织的小型专家材料、原始行、图表与证据包；各README解释当时范围 |
| `evidence/` | Git跟踪的资产manifest与迁移provenance，不存模型权重 |
| `data/`、`models/`、`runs/`、`.venv/` | ignored本地资产与环境，远程仓库不包含这些大文件 |

文档职责分开：稳定规则不记录动态分数，进度不复制整段历史，历史设计不伪装成当前方法。
封存材料中的旧源码路径按其记录的Git commit解释；已退役入口不在main维持兼容副本。

## 当前代码所有权

| 职责 | `src/ember/`中的owner |
| --- | --- |
| 原生图文／完整H读取、三组Meta | `writer/video_program.py`、`writer/meta_lora.py` |
| Core、Procedure与条件化参数slots | `writer/temporal.py`、`writer/procedure.py` |
| 唯一38-target完整A/B生成 | `writer/model.py`、`pi05_lora.py` |
| 主/辅助功能监督、采样、训练与恢复 | `writer/{supervised,function_credit,learning_data,training,continuation}.py`、`ecp/checkpoint.py` |
| Writer运行时与物化 | `writer/runtime.py`、`writer/materialization.py`、`writer/evaluation.py` |
| Source与共享LoRA监督 | `pi05_source_training.py`、`source_sft/` |
| 配对闭环、队列、协议与结果 | `pi05_eval/`、`pi05_eval_queue.py`、`pi05_eval_contract.py`、`pi05_eval_results.py` |

Writer入口为`scripts/train_writer.py`、`scripts/materialize_writer.py`、`scripts/evaluate_pi05.py`。
Source/MT-BC入口为`scripts/train_source_base.py`和`scripts/train_source_sft.py`；复用同一评测合同。
已结束的stability/output-space/causal诊断与low-LR专用执行面已退役，原实现可从`7b18030c`及各run记录的commit恢复。
常规完整恢复和已登记的1500→2100 continuation保留；旧low-LR phase配置不再接受，避免静默改变学习率。

## 数据与证据入口

当前coverage协议在[configs/libero_24_8_8_coverage_v1](configs/libero_24_8_8_coverage_v1/coverage.md)，
旧`libero_24_8_8_v1`仅按封存合同解释；source71审计位于`configs/pi05_source_corpus_v1/`。
新的fit28/diagnostic-held8诊断合同在[接续设计](docs/designs/conditional_compilation_diagnostics_design.md)，官方24/8/8未改。

历史重点可从[46组证据审计](docs/analyses/v52_evidence_audit_20260917.md)、
[旧教学候选总报告](docs/review_materials/20260919/final_report.md)、
[旧划分Test](docs/review_materials/20260920/test_capacity/report.md)、
[因果诊断](docs/review_materials/20260922/writer_causal_diagnostics/README.md)和
[夜间结果包](docs/review_materials/20260923/overnight_results/README.md)进入。
这些材料服务不同历史问题，不合并成一条未经匹配的性能曲线。

数据集、实际使用的Source、当前比较/诊断依赖和正式原始证据保留。历史checkpoint按用途择点保存关键权重，
已结束路线的续训状态及非关键中间权重不永久保留；本次裁剪范围和回收量见[progress](progress.md)。
各checkpoint中的`checkpoint_retirement.json`记录当前可用性，原manifest记录的是历史完整状态。
`weights_only`存档不能精确续训，部分历史CLI仍要求完整trainer，重放前须显式适配权重加载；`metadata_only`不再含模型权重。
已退役物化载荷的manifest、条件映射和复现commit保留，重建前同时检查`payload_retirement.json`及其上游checkpoint状态。

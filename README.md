# EMBER

EMBER研究从exact task language与action-hidden教学视频，在rollout前一次生成冻结π0.5 source的一套完整task-conditioned LoRA，
让机器人从未见初始化闭环执行。语言说明目标，正确视频中的操作内容与顺序应贡献真实执行价值。

当前goal、授权与运行状态以[进度](progress.md)为准。[覆盖重训合同](docs/coverage_retraining_design.md)登记新的24/8/8＋12辅助任务协议及两方法独立选点；
后续FT/RL/外部比较沿用[论文实验合同](docs/paper_experiments_design.md)，须先通过性能与视频对照。
历史证据包括[教学候选总报告](docs/review_materials/20260919/final_report.md)和[旧划分Test负结果](docs/review_materials/20260920/test_capacity/report.md)，
旧实验分数及冻结checkpoint不改写为新协议结果。

## 阅读入口

| 文档 | 职责 |
| --- | --- |
| [Owner要求](docs/current_owner_requirements.md) | 稳定目标、研究原则与最新裁决 |
| [科学动机](docs/concept.md) | 完整方法链条、因果职责与待检验假设 |
| [覆盖重训合同](docs/coverage_retraining_design.md)／[语义覆盖审计](configs/libero_24_8_8_coverage_v1/coverage.md) | 新任务协议、采样、早停及冻结后的裁决 |
| [封存教学候选](docs/video_teaching_writer_design.md)／[专家最终修订](docs/review_materials/20260919/expert_proposal.md) | 部署图、联合损失、有界训练评测与交付合同 |
| [教学候选总报告](docs/review_materials/20260919/final_report.md)／[专家提示词](docs/review_materials/20260919/expert_discussion_prompt.md) | 完整结果、比较边界、原件入口与讨论问题 |
| [完整历史证据审计](docs/v52_evidence_audit_20260917.md) | 46组实验及bank/chart补表的机制、预算、正负证据与比较边界 |
| [当前计划](task_plan.md)／[当前进度](progress.md) | 当前goal、授权、实施证据与下一阶段 |
| [AGENTS](AGENTS.md) | 科学、数据、评测、资源与Git合同 |
| [Findings](findings.md)／[研究历史](docs/research_history.md) | 跨轮结论，以及封存设计、专家评审与formal原件索引 |

讨论时先读Owner要求和证据审计，再看[findings](findings.md)§117–124：统一Writer终局、A900机制诊断、Core/Procedure交叉、匹配frame-set、同视频教学完整窗口与双相机对照。
A900机制与Core/Procedure交叉两次冻结诊断只支持其固定模型、train24有限面板上的结论，没有证明下一架构必须保留Core/P、只能改Procedure或不能整体重构。
旧实验与咨询均从研究历史按问题追溯；历史中的“当前／下一步”不构成执行授权。

## 代码所有权与运行入口

| 代码职责 | `src/ember/`下的owner |
| --- | --- |
| 原生图文／完整H读取与三组Meta | `writer/video_program.py`、`writer/meta_lora.py` |
| 语义Core、重复过程读取与条件化参数slots | `writer/temporal.py`、`writer/procedure.py` |
| 唯一38-target完整A/B生成 | `writer/model.py`、`pi05_lora.py` |
| 主FM与同视频教学、采样与完整checkpoint | `writer/supervised.py`、`writer/function_credit.py`、`writer/training.py`、`writer/learning_data.py`、`writer/continuation.py`、`ecp/checkpoint.py` |
| 运行时、物化与strict闭环评测 | `writer/runtime.py`、`writer/materialization.py`、`writer/evaluation.py`、`pi05_eval/` |

Canonical入口为`scripts/train_writer.py`、`scripts/materialize_writer.py`和`scripts/evaluate_pi05.py`，
教学候选使用独立schema与fresh初始化；同架构的后续窗口继承其完整训练状态。
旧checkpoint须从记录的Git commit重建冻结runtime；已结束实验不长期保留worktree。
本轮之外的旧实验不由保留入口自动恢复。

## 数据与资产

新覆盖split与两方法配置在`configs/libero_24_8_8_coverage_v1/`；旧`configs/libero_24_8_8_v1/`只按封存合同解释。
source71审计在`configs/pi05_source_corpus_v1/`；变更split须Owner明确授权，不能按结果自行改ID。
`data/`、`models/`、`runs/`、`.venv/`为ignored本地资产；复用canonical根，不复制大资产。
源码退役不删除数据集、源模型、唯一checkpoint或formal证据；远程读取者不应假定能访问这些本地资产。
验证应按实际改动选择已有检查，具体运行及通过范围记录在progress。

# EMBER

EMBER研究从exact task language与action-hidden教学视频，在rollout前一次生成冻结π0.5 source的一套完整task-conditioned LoRA，
让机器人从未见初始化闭环执行。语言说明目标，正确视频中的操作内容与顺序应贡献真实执行价值。

**A的learned frame-set匹配诊断已完成，训练停在1200。**
集合参照900/1200为142/118，A为140/135：能达到约140，但未证明相邻能力保持同样强。
[完整结果与分析](docs/review_materials/20260918/frameset_report.md)包含四节点、逐任务配对与success-set保持；当前无运行中的实验或active design。

[封存匹配设计](docs/learned_frameset_reference_design.md)保留A的Core、Procedure、AdaLN、三Meta与共享完整A/B头，
仅移除给定视频帧序处理并fresh训练；这不是未来架构选择。长期要求仍是保留已有能力、增强可信的视频特异性与自然可扩展结构。
已结束统一Writer由Git和[封存设计](docs/v52_evidence_based_writer_design.md)保存。状态见[progress](progress.md)。

## 阅读入口

| 文档 | 职责 |
| --- | --- |
| [Owner要求](docs/current_owner_requirements.md) | 稳定目标、研究原则与最新裁决 |
| [科学动机](docs/concept.md) | 完整方法链条、因果职责与待检验假设 |
| [当前匹配诊断](docs/learned_frameset_reference_design.md)／[专家原文](docs/review_materials/20260918/expert_review.md) | 唯一干预、固定节点、比较与交付合同 |
| [完整历史证据审计](docs/v52_evidence_audit_20260917.md) | 46组实验及bank/chart补表的机制、预算、正负证据与比较边界 |
| [封存统一设计](docs/v52_evidence_based_writer_design.md) | 已完成实例的接口、训练和证据合同 |
| [当前计划](task_plan.md)／[当前进度](progress.md) | 当前goal、授权、实施证据与下一阶段 |
| [AGENTS](AGENTS.md) | 科学、数据、评测、资源与Git合同 |
| [Findings](findings.md)／[研究历史](docs/research_history.md) | 跨轮结论，以及封存设计、专家评审与formal原件索引 |

讨论时先读Owner要求和证据审计，再看[findings](findings.md)§117–120：统一Writer终局、A900机制诊断、Core/Procedure交叉及匹配frame-set训练。
A900机制与Core/Procedure交叉两次冻结诊断只支持其固定模型、train24有限面板上的结论，没有证明下一架构必须保留Core/P、只能改Procedure或不能整体重构。
旧实验与咨询均从研究历史按问题追溯；历史中的“当前／下一步”不构成执行授权。

## 代码所有权与运行入口

| 代码职责 | `src/ember/`下的owner |
| --- | --- |
| 原生图文／完整H读取与三组Meta | `writer/video_program.py`、`writer/meta_lora.py` |
| 语义Core、帧集合Procedure与条件化参数slots | `writer/temporal.py` |
| 唯一38-target完整A/B生成 | `writer/model.py`、`pi05_lora.py` |
| 纯FM学习、采样与完整checkpoint | `writer/supervised.py`、`writer/function_credit.py`、`writer/training.py`、`writer/learning_data.py`、`ecp/checkpoint.py` |
| 运行时、物化与strict闭环评测 | `writer/runtime.py`、`writer/materialization.py`、`writer/evaluation.py`、`pi05_eval/` |

Canonical入口为`scripts/train_writer.py`、`scripts/materialize_writer.py`和`scripts/evaluate_pi05.py`，
frame-set参照使用独立schema与fresh初始化；A的历史checkpoint使用其冻结runtime，不装入参照继续训练。
本轮之外的旧实验不由保留入口自动恢复。

## 数据与资产

固定split在`configs/libero_24_8_8_v1/`，source71审计在`configs/pi05_source_corpus_v1/`。
`data/`、`models/`、`runs/`、`.venv/`为ignored本地资产；复用canonical根，不复制大资产。
源码退役不删除数据集、源模型、唯一checkpoint或formal证据；远程读取者不应假定能访问这些本地资产。
验证应按实际改动选择已有检查，具体运行及通过范围记录在progress。

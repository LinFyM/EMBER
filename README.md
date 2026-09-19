# EMBER

EMBER研究从exact task language与action-hidden教学视频，在rollout前一次生成冻结π0.5 source的一套完整task-conditioned LoRA，
让机器人从未见初始化闭环执行。语言说明目标，正确视频中的操作内容与顺序应贡献真实执行价值。

**同视频教学候选、匹配消融、两组续训和固定视频 controls已完成，双相机对照正在运行。** 主实验单相机 agentview；重复完整 H／相邻视觉读取与有序 Procedure，
同视频五步教学和跨 episode 主 FM 联合更新整个 Writer 与三组 Meta。两臂2100 correct为158/159，原1500的主组增量未持续到末点；[续训完整结果](docs/review_materials/20260919/continuation_report.md)。
按Owner凌晨授权新增的双相机fresh1500已通过真实四卡profile并启动；
[匹配视频controls](docs/review_materials/20260919/ablation_controls_report.md)显示尚未证明教学项增强了顺序特异性。
原窗口与追加合同见[设计](docs/video_teaching_writer_design.md)，实际状态见[progress](progress.md)。

上轮 A 的 learned frame-set 匹配诊断已经完成：900/1200 为142/118，A为140/135，未证明相邻等强；
[完整报告](docs/review_materials/20260918/frameset_report.md)与[封存合同](docs/learned_frameset_reference_design.md)保留原比较边界。
旧实验只作为历史依据，不自动恢复执行。

## 阅读入口

| 文档 | 职责 |
| --- | --- |
| [Owner要求](docs/current_owner_requirements.md) | 稳定目标、研究原则与最新裁决 |
| [科学动机](docs/concept.md) | 完整方法链条、因果职责与待检验假设 |
| [当前教学候选](docs/video_teaching_writer_design.md)／[专家最终修订](docs/review_materials/20260919/expert_proposal.md) | 部署图、联合损失、有界训练评测与交付合同 |
| [完整历史证据审计](docs/v52_evidence_audit_20260917.md) | 46组实验及bank/chart补表的机制、预算、正负证据与比较边界 |
| [封存统一设计](docs/v52_evidence_based_writer_design.md) | 已完成实例的接口、训练和证据合同 |
| [当前计划](task_plan.md)／[当前进度](progress.md) | 当前goal、授权、实施证据与下一阶段 |
| [AGENTS](AGENTS.md) | 科学、数据、评测、资源与Git合同 |
| [Findings](findings.md)／[研究历史](docs/research_history.md) | 跨轮结论，以及封存设计、专家评审与formal原件索引 |

讨论时先读Owner要求和证据审计，再看[findings](findings.md)§117–121：统一Writer终局、A900机制诊断、Core/Procedure交叉、匹配frame-set和同视频教学完整窗口。
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
旧A和frame-set的checkpoint使用其冻结runtime，不装入新模型继续训练。
本轮之外的旧实验不由保留入口自动恢复。

## 数据与资产

固定split在`configs/libero_24_8_8_v1/`，source71审计在`configs/pi05_source_corpus_v1/`。
`data/`、`models/`、`runs/`、`.venv/`为ignored本地资产；复用canonical根，不复制大资产。
源码退役不删除数据集、源模型、唯一checkpoint或formal证据；远程读取者不应假定能访问这些本地资产。
验证应按实际改动选择已有检查，具体运行及通过范围记录在progress。

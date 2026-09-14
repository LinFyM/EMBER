# EMBER

EMBER研究从exact task language与action-hidden教学视频，在rollout前一次生成冻结π0.5 source的一套完整task-conditioned LoRA，
让机器人从未见初始化闭环执行。语言说明目标，正确视频中的操作内容与顺序应贡献真实执行价值。

当前方法为[Process Pullback Writer](docs/process_pullback_writer_design.md)：同步双路RGB形成语义状态与变化驱动的过程表示，
共享网络输出动作作用码，经裸冻结source的固定导数和PCA rank16投影编译为唯一38-target LoRA。
Writer及读取侧两组Meta以纯跨episode FM从头共同学习。当前优先检验能力、视频特异性与保持，暂不强制长期性能线。

## 阅读入口

| 文档 | 职责 |
| --- | --- |
| [Owner要求](docs/current_owner_requirements.md) | 稳定目标、研究原则与最新裁决 |
| [科学动机](docs/concept.md) | 完整方法链条、因果职责与待检验假设 |
| [Active design](docs/process_pullback_writer_design.md) | 本方法的接口、训练和证据合同 |
| [当前计划](task_plan.md)／[当前进度](progress.md) | 当前goal、授权、实施证据与下一阶段 |
| [AGENTS](AGENTS.md) | 科学、数据、评测、资源与Git合同 |
| [Findings](findings.md)／[研究历史](docs/research_history.md) | 跨轮结论，以及封存设计、专家评审与formal原件索引 |

旧实验与咨询均从研究历史按问题追溯；历史中的“当前／下一步”不构成执行授权。

## 当前代码与运行入口

| 代码职责 | `src/ember/`下的owner |
| --- | --- |
| 原生图文／完整H读取与观察Meta | `writer/native.py`、`writer/meta_lora.py`、`ecp/policy_effects.py` |
| 语义状态、过程读取与动作作用码 | `writer/video.py`、`writer/attention.py` |
| 固定source导数、PCA投影与完整LoRA编译 | `writer/correction.py`、`writer/factor.py`、`pi05_lora.py` |
| 纯FM学习、采样与完整checkpoint | `writer/supervised.py`、`writer/function_credit.py`、`writer/training.py`、`writer/learning_data.py`、`ecp/checkpoint.py` |
| 运行时、物化与strict闭环评测 | `writer/runtime.py`、`writer/materialization.py`、`writer/evaluation.py`、`pi05_eval/` |

Canonical入口为`scripts/train_writer.py`、`scripts/materialize_writer.py`和`scripts/evaluate_pi05.py`，
训练配置为`configs/pi05_process_pullback_writer.json`。正式学习窗口在真实profile和新出口功能前提之后登记；
旧checkpoint使用其原冻结runtime，不装入新架构。是否已有运行或结果只看progress。

## 数据与资产

固定split在`configs/libero_24_8_8_v1/`，source71审计在`configs/pi05_source_corpus_v1/`。
`data/`、`models/`、`runs/`、`.venv/`为ignored本地资产；复用canonical根，不复制大资产。
源码退役不删除数据集、源模型、唯一checkpoint或formal证据；远程读取者不应假定能访问这些本地资产。
验证应按实际改动选择已有检查，具体运行及通过范围记录在progress。

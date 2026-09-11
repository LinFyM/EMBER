# EMBER

EMBER研究从exact task language与action-hidden教学视频，在rollout前一次生成冻结π0.5 source的一套完整task-conditioned LoRA，
让机器人从未见初始化闭环执行。语言说明目标，正确视频中的操作内容与顺序应贡献真实执行价值。

**当前阶段先解决有益的视频特异性，暂不强制绝对性能。** Owner要求综合全部历史证据提炼理论，重新推导从输入到训练的完整方案，
并请一位没有旧对话上下文的外部专家独立分析。科研训练与评测继续暂停，本次只整理材料与咨询。

不直接回到v5.2，也不预设以它为底座改进。v5.2的性能与稳定性不达标，但其普通监督视频依赖是关键正证据；
后继更高单项成绩和其它局部正证据同样保留。项目长期strict400>145及稳定/广度/视频因果资格仍保留，不能覆盖本阶段优先级。

## 全新专家先读

1. **[本次唯一咨询入口](docs/review_materials/video_specificity_20260911/README.md)**：阶段目标、阅读路线、证据范围及模型身份。
2. [综合历史机制地图](docs/review_materials/video_specificity_20260911/EVIDENCE_MAP.md)：正负证据、混杂、反例和原件路径；不是预定根因。
3. [可复制咨询prompt](docs/review_materials/video_specificity_20260911/EXPERT_PROMPT.md)。
4. [当前状态](progress.md)、[计划](task_plan.md)、[长期要求](docs/current_owner_requirements.md)、[科学合同](AGENTS.md)。

新包补齐R/C/S、off/R续训、C冻结诊断与无变化参照首段。早期强模型、GOMQ/PQ等原件复用
[9月7日包](docs/review_materials/20260907/README.md)，Horizon all/off及真实更新诊断复用
[9月11日旧包](docs/review_materials/20260911/README.md)。历史专家意见按需追溯，不能代替原始事实或最新Owner要求。

## 当前保留实现与实验状态

源码保留无变化参照候选：真实图文prefix → Action Expert与观察Meta完整50-H响应 → 四组过去局部对应、完整H-query、
两端视觉读取、配对无变化参照GRU、单向长程与逐H回写 → 语义条件化过程编译 → native D → 唯一38-target/76-tensor A/B。
这是待解释的已实现模型，没有因本次整理被选为后续方案。

候选fresh100/200已训练，200 validation34/400、固定训练面板correct45/96，各开发视频差额区间含零；
100 validation未完成，300/400未启动。详细口径见[progress](progress.md)和[findings](findings.md)§55。

| 代码职责 | src/ember下的owner |
| --- | --- |
| 原生图文证据与观察Meta | `writer/native.py`、`writer/meta_lora.py`、`ecp/policy_effects.py` |
| 完整H过程与编译 | `writer/relation.py`、`writer/horizon.py`、`writer/semantic.py`、`writer/attention.py` |
| 完整LoRA输出 | `writer/native_factor.py`、`pi05_lora.py` |
| 监督学习与采样 | `writer/supervised.py`、`writer/functional.py`、`writer/training.py`、`writer/learning_data.py` |
| 物化、闭环与恢复 | `writer/runtime.py`、`writer/materialization.py`、`writer/evaluation.py`、`pi05_eval/`、`ecp/checkpoint.py` |

入口为`scripts/train_horizon_writer.py`、`scripts/materialize_horizon_writer.py`、`scripts/evaluate_pi05.py`。
训练默认配置`configs/pi05_video_change_reference.json`；它不是新的launch授权。旧checkpoint须使用原冻结runtime。

## 科学与资产入口

[concept](docs/concept.md)解释科学动机；[局部参照设计](docs/video_change_reference_design.md)与
[完整Horizon设计](docs/horizon_relation_video_writer_design.md)保存已实现方法；[findings](findings.md)保存跨轮结论，
[research_history](docs/research_history.md)索引全部分层历史、冻结设计和专家修订。历史“当前/下一步”均按当时时点解释。

`data/`、`models/`、`runs/`、`.venv/`为ignored本地资产；远程专家只使用已提交副本与其索引，不假定能访问本地路径。
固定split在`configs/libero_24_8_8_v1/`，source71审计在`configs/pi05_source_corpus_v1/`。
共享源码测试入口为`PYTHONPATH=src .venv/bin/python -m pytest -q`；材料整理只运行相应证据复算、链接与diff检查，不启动ML实验。

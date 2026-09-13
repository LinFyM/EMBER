# EMBER

EMBER研究从exact task language与action-hidden教学视频，在rollout前一次生成冻结π0.5 source的一套完整task-conditioned LoRA，
让机器人从未见初始化闭环执行。语言说明目标，正确视频中的操作内容与顺序应贡献真实执行价值。

**当前阶段先解决有益的视频特异性，暂不强制绝对性能。** Owner已授权依据综合证据修正理论、实现与实验。
当前canonical实现为[语义状态路径Writer](docs/semantic_path_writer_design.md)：完整原生图文／Action响应解释视频状态，
状态路径提供有向变化关系，经条件调制生成自由完整A/B，由跨episode主FM联合学习。
当前授权、唯一active design及实际资源／实验状态只看[progress](progress.md)，计划见[task_plan](task_plan.md)；旧设计不自行恢复。

## 历史专家评审索引

1. **[9月11日咨询入口](docs/review_materials/video_specificity_20260911/README.md)**：当时的目标、阅读路线、证据范围及模型身份。
2. [综合历史机制地图](docs/review_materials/video_specificity_20260911/EVIDENCE_MAP.md)：正负证据、混杂、反例和原件路径；不是预定根因。
3. [可复制咨询prompt](docs/review_materials/video_specificity_20260911/EXPERT_PROMPT.md)。
4. [当前状态](progress.md)、[计划](task_plan.md)、[长期要求](docs/current_owner_requirements.md)、[科学合同](AGENTS.md)。

新包补齐R/C/S、off/R续训、C冻结诊断与无变化参照首段。早期强模型、GOMQ/PQ等原件复用
[9月7日包](docs/review_materials/20260907/README.md)，Horizon all/off及真实更新诊断复用
[9月11日旧包](docs/review_materials/20260911/README.md)。历史专家意见按需追溯，不能代替原始事实或最新Owner要求。

## 当前保留实现与实验状态

当前数据流：双相机teacher RGB与exact language形成同版本原生contextual Z／完整50-H响应；
任务语义查询完整T×50动作响应，双向解释每帧状态。共享语义坐标的二阶有向路径与语义内容一起生成唯一38-target/76-tensor A/B。
独立训练的frame_set参照保留全部画面与相同学习模块，只把路径变换替换为匹配维度的全帧语义二阶矩。
仅跨episode主FM联合更新Writer及两组teacher Meta；query obs[i]的动作标签从i+1开始，最后无未来标签的观测不参与监督。
source执行权重冻结；额外视频encoder、裸X输入bank、空间／纠正参数辅助路径已退出活动实现。

| 代码职责 | src/ember下的owner |
| --- | --- |
| 原生图文证据与观察Meta | `writer/native.py`、`writer/meta_lora.py`、`ecp/policy_effects.py` |
| 完整H读取、语义状态路径与编译 | `writer/video.py`、`writer/attention.py` |
| 完整LoRA输出 | `writer/factor.py`、`pi05_lora.py` |
| 监督学习与采样 | `writer/supervised.py`、`writer/function_credit.py`、`writer/training.py`、`writer/learning_data.py` |
| 物化、闭环与恢复 | `writer/runtime.py`、`writer/materialization.py`、`writer/evaluation.py`、`pi05_eval/`、`ecp/checkpoint.py` |

入口为`scripts/train_horizon_writer.py`、`scripts/materialize_horizon_writer.py`、`scripts/evaluate_pi05.py`。
训练默认配置`configs/pi05_semantic_path_writer.json`与`_frame_set.json`是匹配的有序／全帧集合比较，使用同一实现。
节点与formal准入在最长真实profile后登记；旧checkpoint须使用原冻结runtime，不能装入新架构。
旧配置与实现通过Git、正式run contract和research_history保留，原始模型／数据／label／checkpoint不随活动代码退役删除。


## 科学与资产入口

[concept](docs/concept.md)解释科学动机；[局部参照设计](docs/video_change_reference_design.md)与
[完整Horizon设计](docs/horizon_relation_video_writer_design.md)保存历史方法；[findings](findings.md)保存跨轮结论，
[research_history](docs/research_history.md)索引全部分层历史、冻结设计和专家修订。历史“当前/下一步”均按当时时点解释。

`data/`、`models/`、`runs/`、`.venv/`为ignored本地资产；远程专家只使用已提交副本与其索引，不假定能访问本地路径。
固定split在`configs/libero_24_8_8_v1/`，source71审计在`configs/pi05_source_corpus_v1/`。
共享源码测试入口为`PYTHONPATH=src .venv/bin/python -m pytest -q`。

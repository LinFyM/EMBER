# EMBER

EMBER研究从exact task language与action-hidden教学视频，在rollout前一次生成冻结π0.5 source的一套完整task-conditioned LoRA，
让机器人从未见初始化闭环执行。语言说明目标，正确视频中的操作内容与顺序应贡献真实执行价值。

**当前阶段先解决有益的视频特异性，暂不强制绝对性能。** Owner已接受综合历史与专家修正后的候选，授权实现与实验。
唯一active design是[Video Functional Writer](docs/video_functional_writer_design.md)：任务token视频表示、训练期执行query辅助功能信用、一次编译完整LoRA。
实际实现、资源与实验进度见[progress](progress.md)，当前计划见[task_plan](task_plan.md)。旧C/无变化参照和95-task不恢复。

## 全新专家先读

1. **[本次唯一咨询入口](docs/review_materials/video_specificity_20260911/README.md)**：阶段目标、阅读路线、证据范围及模型身份。
2. [综合历史机制地图](docs/review_materials/video_specificity_20260911/EVIDENCE_MAP.md)：正负证据、混杂、反例和原件路径；不是预定根因。
3. [可复制咨询prompt](docs/review_materials/video_specificity_20260911/EXPERT_PROMPT.md)。
4. [当前状态](progress.md)、[计划](task_plan.md)、[长期要求](docs/current_owner_requirements.md)、[科学合同](AGENTS.md)。

新包补齐R/C/S、off/R续训、C冻结诊断与无变化参照首段。早期强模型、GOMQ/PQ等原件复用
[9月7日包](docs/review_materials/20260907/README.md)，Horizon all/off及真实更新诊断复用
[9月11日旧包](docs/review_materials/20260911/README.md)。历史专家意见按需追溯，不能代替原始事实或最新Owner要求。

## 当前保留实现与实验状态

当前数据流：冻结图像/词嵌入 → teacher侧Gemma VL Meta形成同次Z/KV → Action Expert与观察Meta完整50-H响应 → 任务token视觉grounding、相邻完整H读取与语言/时间轴交互 → Compiler → native D → 唯一38-target/76-tensor A/B。
训练期辅助读取器以冻结source真实FM query查询共享视频表示；当前mu1/rho0，联合反传直接Z及R经KV两条路径。source基础权重与执行prefix始终冻结，部署不保留读取Meta或辅助控制器。

| 代码职责 | src/ember下的owner |
| --- | --- |
| 原生图文证据与观察Meta | `writer/native.py`、`writer/meta_lora.py`、`ecp/policy_effects.py` |
| 完整H过程与编译 | `writer/video.py`、`writer/attention.py` |
| 完整LoRA输出 | `writer/native_factor.py`、`pi05_lora.py` |
| 监督学习与采样 | `writer/supervised.py`、`writer/function_credit.py`、`writer/function_reader.py`、`writer/training.py`、`writer/learning_data.py` |
| 物化、闭环与恢复 | `writer/runtime.py`、`writer/materialization.py`、`writer/evaluation.py`、`pi05_eval/`、`ecp/checkpoint.py` |

入口为`scripts/train_horizon_writer.py`、`scripts/materialize_horizon_writer.py`、`scripts/evaluate_pi05.py`。
训练默认配置`configs/pi05_video_functional.json`；同表示纯FM与训练无序参照采用同一实现的显式配置。旧checkpoint须使用原冻结runtime。

## 科学与资产入口

[concept](docs/concept.md)解释科学动机；[局部参照设计](docs/video_change_reference_design.md)与
[完整Horizon设计](docs/horizon_relation_video_writer_design.md)保存历史方法；[findings](findings.md)保存跨轮结论，
[research_history](docs/research_history.md)索引全部分层历史、冻结设计和专家修订。历史“当前/下一步”均按当时时点解释。

`data/`、`models/`、`runs/`、`.venv/`为ignored本地资产；远程专家只使用已提交副本与其索引，不假定能访问本地路径。
固定split在`configs/libero_24_8_8_v1/`，source71审计在`configs/pi05_source_corpus_v1/`。
共享源码测试入口为`PYTHONPATH=src .venv/bin/python -m pytest -q`。

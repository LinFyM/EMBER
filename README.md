# EMBER

EMBER研究从exact task language和action-hidden正确教学视频，在rollout前一次生成冻结π0.5 source的一套完整task-conditioned LoRA，使机器人从未见初始化闭环执行。目标是validation8 single-checkpoint strict paired correct **>145/400**，并同时满足相邻稳定、低churn、任务广度、四suite/Goal/Long、换视频鲁棒性和最终视频因果要求。

**当前处于专家复核阶段，没有运行中的训练或评测。** Horizon这轮最好的验证候选为126/400，另一个初始化119/400；均未达到科学目标。126候选的最新视频对照也未建立正确动态教学的稳定整体优势。当前工作是审视证据并咨询如何改进现有方法，不能从历史计划自行恢复训练。

## 先读哪里

1. **[最新专家材料与证据索引](docs/review_materials/20260911/README.md)**，以及[可直接使用的咨询prompt](docs/review_materials/20260911/EXPERT_PROMPT.md)。最新学习对照、视频控制、有限更新、机制数据与行为图片均有远程可读副本；不要求专家访问本地runs。
2. [当前状态](progress.md)、[当前计划与暂停边界](task_plan.md)、[长期要求](docs/current_owner_requirements.md)、[项目合同](AGENTS.md)。
3. [科学动机与数据流](docs/concept.md)、[当前canonical设计](docs/horizon_relation_video_writer_design.md)。
4. [持久发现](findings.md)、[分层研究历史](docs/research_history.md)，再按需读取[9月7日历史证据包](docs/review_materials/20260907/README.md)和[9月8日专家讨论及Owner裁决](docs/review_materials/20260908/README.md)。

旧实验中的“当前”“下一步”只描述当时状态。此前专家建议的双向长程，已被Owner选定的过去单向长程覆盖。

## main与最强候选的区别

| 运行面 | Compiler额外语言query | D | teacher视角 | 状态 |
| --- | --- | --- | --- | --- |
| main canonical | 启用 | 每target/rank/side独立，跨task共享 | 默认agentview；已支持dual输入 | 双视角仅通过输入/梯度smoke，无双视角训练分数 |
| 最强候选[45e16633](https://github.com/LinFyM/EMBER/tree/45e16633394e1baa9af8cefba698661b41aab539)的local_h_read macro400 | 关闭该额外route；其余exact language/local/H-read保留 | 与main相同独立D | 实际训练为agentview | init7验证126/400；init11同修正119/400，未正式采纳 |
| 历史同target跨rank D绑定候选 | 启用 | 同target跨rank绑定 | agentview | 验证115→82，未采纳；不能当作126候选 |

因此，直接运行main默认配置不等于复现126模型。各实验的精确配置、commit、checkpoint节点和结果见[最新模型索引](docs/review_materials/20260911/index.json)。保留原独立分支作为冻结实验身份，不合入未获采纳的科学行为。

## 当前代码归属

数据流：真实图文prefix → Action Expert与观察侧Meta的完整50-horizon响应 → 四组过去局部对应、H-query、两端视觉核实、有序GRU及单向长程 → 集合Compiler → 唯一38-target/76-tensor A/B。Writer当前阶段端到端纯FM，同task跨episode监督，source始终冻结。

| 责任 | owner（相对src/ember） |
| --- | --- |
| 原生证据与Meta | `writer/native.py`、`writer/meta_lora.py`、`ecp/policy_effects.py` |
| 完整过程图与Compiler | `writer/relation.py`、`writer/horizon.py`、`writer/attention.py` |
| 完整策略输出 | `writer/native_factor.py`、`pi05_lora.py` |
| 监督更新与采样 | `writer/supervised.py`、`writer/functional.py`、`writer/training.py`、`writer/learning_data.py` |
| 物化与闭环 | `writer/runtime.py`、`writer/materialization.py`、`writer/evaluation.py`、`pi05_eval/` |
| 完整checkpoint | `ecp/checkpoint.py` |

入口为`scripts/train_horizon_writer.py`、`scripts/materialize_horizon_writer.py`、`scripts/evaluate_pi05.py`。`configs/pi05_horizon_writer_v1.json`保留canonical首段配置快照，**不是当前授权的下一次launch计划**。未来执行须先登记方法、数据/曝光、节点和资源合同。

## 资产与验证

`data/`、`models/`、`runs/`、`.venv/`是ignored本地资产；复用canonical来源，不复制大模型、数据或环境。固定split在`configs/libero_24_8_8_v1/`，source71-task审计在`configs/pi05_source_corpus_v1/`，执行/LoRA合同在`configs/pi05_target_evaluation_v1.json`、`configs/pi05_lora_v1.json`。

共享源码修改按影响范围运行已有测试；标准CPU测试入口为`PYTHONPATH=src .venv/bin/python -m pytest -q`。文档维护检查对应diff与新材料引用。formal train/eval来自clean pushed detached worktree，launch前核对两GPU节点与独立quota；旧资源快照和历史待办不构成当前授权。

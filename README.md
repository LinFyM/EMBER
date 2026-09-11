# EMBER

EMBER研究从exact task language和action-hidden正确教学视频，在rollout前一次生成冻结π0.5 source的一套完整task-conditioned LoRA，使机器人从未见初始化闭环执行。目标是validation8 single-checkpoint strict paired correct **>145/400**，并同时满足相邻稳定、低churn、任务广度、四suite/Goal/Long、换视频鲁棒性和最终视频因果要求。

**Owner已授权休息期间由Codex持续自主修正、实验与分析，active goal已建立。** 当前登记[扩大非held独立meta-task映射](docs/nonheld_meta_writer_design.md)，在较强off结构上进行95-task有限对照；实现与formal进度见progress。既有Horizon最好候选为126/400、另一初始化119/400；均未达到科学目标，视频动态增量也未建立。

## 先读哪里

1. **[最新专家材料与证据索引](docs/review_materials/20260911/README.md)**，以及[可直接使用的咨询prompt](docs/review_materials/20260911/EXPERT_PROMPT.md)。最新学习对照、视频控制、有限更新、机制数据与行为图片均有远程可读副本；不要求专家访问本地runs。
2. [当前状态](progress.md)、[当前计划与授权](task_plan.md)、[长期要求](docs/current_owner_requirements.md)、[项目合同](AGENTS.md)。
3. [科学动机与数据流](docs/concept.md)、[当前active设计](docs/nonheld_meta_writer_design.md)、[保留的前代完整H关系图细节](docs/horizon_relation_video_writer_design.md)。
4. [持久发现](findings.md)、[分层研究历史](docs/research_history.md)，再按需读取[9月7日历史证据包](docs/review_materials/20260907/README.md)和[9月8日专家讨论及Owner裁决](docs/review_materials/20260908/README.md)。

旧实验中的“当前”“下一步”只描述当时状态。此前专家建议的双向长程，已被Owner选定的过去单向长程覆盖。

## 当前方法与历史参照

当前选择保留off结构：完整H、过去定向关系、local_h_read Compiler（关闭额外语言query）与独立D，agentview、K1。C/S专属消费路径已判定不追加，正在按新设计退役；双视角仍只有输入/梯度smoke，没有训练分数。

历史off init7 macro400为126/400、init11为119，均未达目标；init7追加500/600降至73/54。双条件R终点400为85/400、train56/96，相同target动作曝光的旧off为126/56。当前95-task候选将fresh训练，不继承上述Writer checkpoint；相同结构或初始化语义不等于复现旧模型分数。

精确历史配置、commit、checkpoint和行为证据见[研究历史](docs/research_history.md)、[本轮发现](findings.md)与[专家包模型索引](docs/review_materials/20260911/index.json)。实际权限和后续只看当前plan/progress。

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

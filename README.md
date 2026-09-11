# EMBER

EMBER研究从exact task language和action-hidden正确教学视频，在rollout前一次生成冻结π0.5 source的一套完整task-conditioned LoRA，使机器人从未见初始化闭环执行。目标是validation8 single-checkpoint strict paired correct **>145/400**，并同时满足相邻稳定、低churn、任务广度、四suite/Goal/Long、换视频鲁棒性和最终视频因果要求。

研究按“还原科学与专家形成链 → 定位正确视频失效 → 依据原因修正并验证”持续推进。正确视频须自然优于错任务和乱序视频，并高于冻结source；当前授权、运行状态和未完成事项只看[progress](progress.md)与[task_plan](task_plan.md)。

## 先读哪里

1. **[最新专家材料与证据索引](docs/review_materials/20260911/README.md)**，以及[可直接使用的咨询prompt](docs/review_materials/20260911/EXPERT_PROMPT.md)。最新学习对照、视频控制、有限更新、机制数据与行为图片均有远程可读副本；不要求专家访问本地runs。
2. [当前状态](progress.md)、[当前计划与授权](task_plan.md)、[长期要求](docs/current_owner_requirements.md)、[项目合同](AGENTS.md)。
3. [科学动机与数据流](docs/concept.md)、[无变化参照局部过程设计](docs/video_change_reference_design.md)、[保留的前代完整H关系图细节](docs/horizon_relation_video_writer_design.md)。
4. [持久发现](findings.md)、[分层研究历史](docs/research_history.md)，再按需读取[9月7日历史证据包](docs/review_materials/20260907/README.md)和[9月8日专家讨论及Owner裁决](docs/review_materials/20260908/README.md)。

旧实验中的“当前”“下一步”只描述当时状态。此前专家建议的双向长程，已被Owner选定的过去单向长程覆盖。

## 当前方法与历史参照

当前受控候选保留C的完整H、过去定向关系、语义条件化过程读取与独立D，agentview、K1、普通正样本FM；只将局部更新改为真实历史响应减同gap/窗口下的无变化参照响应。静态不变性只是表示性质，是否学出正确视频净收益由闭环判断。

原off/R/C/S与95-task提案的适用范围、负结果及暂停状态保存在[形成链与机制复核](docs/video_mechanism_reassessment.md)和研究历史；历史“下一步”不构成执行授权。

精确历史配置、commit、checkpoint和行为证据见[研究历史](docs/research_history.md)、[本轮发现](findings.md)与[专家包模型索引](docs/review_materials/20260911/index.json)。实际权限和后续只看当前plan/progress。

## 当前代码归属

数据流：真实图文prefix → Action Expert与观察侧Meta的完整50-horizon响应 → 四组过去局部对应、H-query、两端视觉核实、带无变化参照的有序GRU及单向长程 → 语义条件化过程Compiler → 唯一38-target/76-tensor A/B。Writer当前阶段端到端纯FM，同task跨episode监督，source始终冻结。

| 责任 | owner（相对src/ember） |
| --- | --- |
| 原生证据与Meta | `writer/native.py`、`writer/meta_lora.py`、`ecp/policy_effects.py` |
| 完整过程图与Compiler | `writer/relation.py`、`writer/horizon.py`、`writer/semantic.py`、`writer/attention.py` |
| 完整策略输出 | `writer/native_factor.py`、`pi05_lora.py` |
| 监督更新与采样 | `writer/supervised.py`、`writer/functional.py`、`writer/training.py`、`writer/learning_data.py` |
| 物化与闭环 | `writer/runtime.py`、`writer/materialization.py`、`writer/evaluation.py`、`pi05_eval/` |
| 完整checkpoint | `ecp/checkpoint.py` |

入口为`scripts/train_horizon_writer.py`、`scripts/materialize_horizon_writer.py`、`scripts/evaluate_pi05.py`。训练入口默认配置为`configs/pi05_video_change_reference.json`；旧配置/检查点需使用对应冻结runtime。实际节点、resume和资源以已登记design与launch contract为准。

## 资产与验证

`data/`、`models/`、`runs/`、`.venv/`是ignored本地资产；复用canonical来源，不复制大模型、数据或环境。固定split在`configs/libero_24_8_8_v1/`，source71-task审计在`configs/pi05_source_corpus_v1/`，执行/LoRA合同在`configs/pi05_target_evaluation_v1.json`、`configs/pi05_lora_v1.json`。

共享源码修改按影响范围运行已有测试；标准CPU测试入口为`PYTHONPATH=src .venv/bin/python -m pytest -q`。文档维护检查对应diff与新材料引用。formal train/eval来自clean pushed detached worktree，launch前核对两GPU节点与独立quota；旧资源快照和历史待办不构成当前授权。

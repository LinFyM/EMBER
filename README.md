# EMBER

EMBER研究能否将exact task language和一条或多条action-hidden正确教学视频，在rollout前一次性编译为冻结π0.5 source的
一套完整task-conditioned LoRA，使机器人从自己的未见初始化闭环执行。正式目标是validation8 single-checkpoint strict paired
correct **>145/400**，并满足相邻/跨视频稳定、breadth、四suite、Goal/Long及最终视频因果要求。

当前方法已完成专家讨论和Owner裁决：**末层完整H → 过去四帧对应 → 完整H-query → 两端视觉核实 → 历史u有序GRU
→ 四组过去单向长程交替/前三组逐H回写 → 集合compiler → 完整native A/B**。首版采用FM辅助共享Writer RL。
**方法已定、尚未实现或取得新分数。**本次交接准备不启动科研作业；后续由Owner指定的新session创建全程goal，全面阅读仓库、完整实现并开展正式实验，依据结果持续迭代推进到达标。

现存正式参照为source47/400、train24 rank128 SFT相邻109/107；它们是历史结果，不是新模型重跑。
有信息量学习后仍不及或仅略超这些参照应当认真定位实质能力缺口，不能靠小调参和内部指标解释。
Owner授权在核心思想与硬合同内依据证据修改具体方法，必要时重构；当前设计是完整起点，不是永久冻结的模块清单。
具体口径与权限见[正式设计§1.1](docs/horizon_relation_video_writer_design.md#11-基线压力历史教训与后续自主权)。

## 接手阅读

1. [AGENTS.md](AGENTS.md)、[长期要求](docs/current_owner_requirements.md)、[当前状态](progress.md)与[计划](task_plan.md)。
2. [科学动机](docs/concept.md)、[唯一正式设计](docs/horizon_relation_video_writer_design.md)。
3. [完整专家回复与Owner裁决](docs/review_materials/20260908/README.md)。重点：**专家原文的双向长程已被Owner的过去单向选择覆盖**。
4. [持久发现](findings.md)、[分层研究历史](docs/research_history.md)及所需原件，保留强结果与负结果各自的适用范围。
5. [临时接续入口](HANDOFF.md)。长期内容已在上述正式文件，消费后删除HANDOFF即可，不丢失设计或授权。

## 当前代码与迁移入口

main现有源码仍是2026-09-07的18层layered Writer，不是上述新图。其run永久止于384：correct69→67、other72→64，
熟悉/held训练视频21/18（各120），未通过科学资格。旧设计全文保存在Git，见[旧设计历史入口](docs/layered_relation_video_writer_design.md)。
**不要直接用旧 `configs/pi05_layered_writer_v1.json` 启动新run或续训旧672步schedule。**

| 责任 | 现有owner | 新实现边界 |
| --- | --- | --- |
| 原生证据与Meta | `writer/native.py`、`ecp/policy_effects.py`、`ecp/observer.py`、`writer/meta_lora.py` | 增加最终Z；直接读取action_out_proj实际输入。旧末项是final norm之前，不能冒充新R |
| 局部、H与长程过程 | `writer/relation.py`、`writer/layered.py` | 过去对应、H-query、两端Z、GRU、四组单向long和回写均需落实 |
| 集合与输出 | `writer/layered.py`、`writer/coordinate.py`、`pi05_lora.py`、`batched_lora.py` | 保留整策略queries与完整contract；native D取代旧坐标decoder |
| FM与反传 | `writer/functional.py`、`replay.py`、`training.py` | 已有真实FM和R-leaf/observer VJP；新增同版本action-Gaussian RL、10步flow重放及trust更新 |
| 数据和调度 | `writer/learning_data.py`、`data.py`、`task_schedule.py`、`task_execution.py` | 复用互斥角色；新每suite随机1task、64FM+4RL，权重独立于GPU/K/长度 |
| 物化与缓存 | `writer/runtime.py`、`materialization.py`、`evaluation.py` | 更新模型/schema；冻结Z/KV与可学习R/过程缓存的生命周期不同 |
| checkpoint/分布式 | `ecp/checkpoint.py`、`writer/topology.py` | fresh新schema、attempt/accepted/RNG/版本；exact-resume锁topology |
| 环境与评测 | `reward/rollout.py`、`pi05_eval/`、`pi05_evaluation.py`、`pi05_eval_contract.py` | 复用canonical执行和动态队列；旧antithetic credit不是新RL |
| source/专家基础 | `source_sft/`、`expert_manifold/` | 只保留合法来源和容量证据，不恢复held字典或旧支线 |

表内代码路径相对 `src/ember/`。完整差异与验证在正式设计§9。
旧P/Q等实现已退役，由 `fcdb6e43706c5fcedf10eaa5d2d459602b263016` 与research_history索引恢复；不为本次实现重新铺回活动树。
未合并的native-factor-readout工作树仍有dirty草稿，只更换旧上游的末端，不能整支直接集成为新方案。

现有CLI为 `scripts/train_layered_writer.py`、`scripts/materialize_layered_writer.py`、`scripts/evaluate_pi05.py`，
以及source/data/expert准备入口。前两者需在新实现中同步迁移；不额外保留平行v2/fallback运行面。
历史物化的`--requests-json`可复用resident source，但新模型签名/checkpoint协议必须先完成，不能把旧bank当作新图输出。

## Canonical资产与证据

工作目录的 `data/`、`models/`、`runs/`、`.venv/`为现有ignored资产；复用它们，不重复安装环境或复制模型/数据。

- `configs/pi05_writer_data_v1.json`：统一资产provenance；历史73/18角色不等于新run授权，首版使用固定train24。
- `configs/libero_24_8_8_v1/protocol.json`：固定development split；旧feasibility段只是历史，不覆盖现行normalization/eval合同。
- `configs/pi05_source_corpus_v1/`：71-task source、去重审计和冻结normalization。
- Source：`runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000/policy/model.safetensors`。
- Tokenizer：`models/tokenizers/openpi/paligemma_tokenizer.model`；目标/source HDF5按manifest复用`data/datasets/`。
- `configs/pi05_lora_v1.json`、`configs/pi05_target_evaluation_v1.json`：完整LoRA与canonical执行合同；旧eval配置中的卡数默认不能覆盖当前≤6卡合同。
- LIBERO BDDL/init和`data/simulation/ember_assets/`已存在，`pi05_assets.prepare_libero_config`负责run-local配置；使用现有`.venv/bin/python`。
- `runs/analysis/layered_relation_writer_20260907/train24_shared/`：旧384裁决与熟悉/held诊断；旧训练run及checkpoint保留，不恢复执行。
- [20260907补证据包](docs/review_materials/20260907/README.md)：现存科学记录副本；[20260908讨论原文](docs/review_materials/20260908/README.md)：后续推导与最终裁决。
- `runs/analysis/ember_handoff_cleanup_20260906/storage_cleanup.json`：历史已删除派生缓存的范围；旧manifest不保证tensor仍在，唯一checkpoint与原始证据保留。

2026-09-08交接准备只核对关键资产存在性，未重新读取数据内容、运行GPU检查或给出当前配额承诺。

## 验证和正式执行

共享源码修改后的CPU测试入口：

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```

仅文档修改检查相应diff。新模型的真实机制/profile与科学学习分开记录，工程测试不等于科学通过。
formal train/eval来自clean pushed commit的detached frozen worktree；launch前检查两GPU节点和strg01对应独立quota，
按真实吞吐使用同节点1–6张合适GPU，不继承旧GPU空闲、拓扑、32GiB预算或旧run的待执行清单。

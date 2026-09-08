# EMBER

EMBER研究能否将exact task language和一条或多条action-hidden正确教学视频，在rollout前一次性编译为冻结π0.5 source的
一套完整task-conditioned LoRA，使机器人从自己的未见初始化闭环执行。正式目标是validation8 single-checkpoint strict paired
correct **>145/400**，并满足相邻/跨视频稳定、breadth、四suite、Goal/Long及最终视频因果要求。

当前方法已完成专家讨论和Owner裁决：**末层完整H → 过去四帧对应 → 完整H-query → 两端视觉核实 → 历史u有序GRU
→ 四组过去单向长程交替/前三组逐H回写 → 集合compiler → 完整native A/B**。当前先纯监督FM达到有证据平台，再接独立共享Writer RL。
**方法已定，新实现尚无分数。**Owner已启动持续科研执行；全程goal已创建，全面阅读与审计已完成，当前落实纯监督正式训练与证据驱动迭代。

现存正式参照为source47/400、train24 rank128 SFT相邻109/107；它们是历史结果，不是新模型重跑。
有信息量学习后仍不及或仅略超这些参照应当认真定位实质能力缺口，不能靠小调参和内部指标解释。
Owner授权在核心思想与硬合同内依据证据修改具体方法，必要时重构；当前设计是完整起点，不是永久冻结的模块清单。
具体口径与权限见[正式设计§1.1](docs/horizon_relation_video_writer_design.md#11-基线压力历史教训与后续自主权)。

## 接手阅读

1. [AGENTS.md](AGENTS.md)、[长期要求](docs/current_owner_requirements.md)、[当前状态](progress.md)与[计划](task_plan.md)。
2. [科学动机](docs/concept.md)、[唯一正式设计](docs/horizon_relation_video_writer_design.md)。
3. [完整专家回复与Owner裁决](docs/review_materials/20260908/README.md)。重点：**专家原文的双向长程已被Owner的过去单向选择覆盖**。
4. [持久发现](findings.md)、[分层研究历史](docs/research_history.md)及所需原件，保留强结果与负结果各自的适用范围。

## 当前代码与运行入口

main采用完整Horizon图及纯FM端到端监督；新方法仍无正式闭环分数。旧18层layered run永久止于384：correct69→67、other72→64，
熟悉/held训练视频21/18（各120），未通过科学资格。旧图、坐标decoder与旧训练入口已由新路径替换，历史由Git与
[研究历史](docs/research_history.md)保留；旧checkpoint拒绝作为新图resume。

| 责任 | 唯一owner | 合同 |
| --- | --- | --- |
| 原生证据与Meta | `writer/native.py`、`ecp/policy_effects.py`、`writer/meta_lora.py` | 同forward最终Z/KV，实际action_out_proj输入完整H；仅冻结prefix可跨参数版本缓存 |
| 完整过程图 | `writer/relation.py`、`writer/horizon.py`、`writer/attention.py` | 过去4帧、H双向query、两端Z、顺序GRU、四组past+self和前三逐H回写 |
| 完整策略输出 | `writer/native_factor.py`、`pi05_lora.py` | 集合compiler一次生成38-target/76-tensor native A/B |
| 监督更新 | `writer/supervised.py`、`writer/functional.py`、`writer/training.py` | 同task跨episode FM，一次完整Writer反传与直接AdamW更新，无RL/trust |
| 采样和诊断 | `writer/learning_data.py` | 每suite随机1task、每task64FM；独立随机流，固定held动作验证无梯度 |
| 物化与执行 | `writer/runtime.py`、`writer/materialization.py`、`writer/evaluation.py`、`pi05_eval/` | supervised horizon schema、J0 strict paired动态队列、单adapter |
| checkpoint | `ecp/checkpoint.py` | 完整optimizer-update边界、sampler/RNG/阶段/版本；exact-resume锁topology |

表内路径相对 `src/ember/`。模块按过程图、监督反传、无梯度诊断及checkpoint生命周期分工，复用现有FM、
checkpoint与evaluator；未保留旧图或第二套训练fallback。原native-factor-readout dirty草稿不属于活动实现，保留其用户工作。

CLI：`scripts/train_horizon_writer.py`、`scripts/materialize_horizon_writer.py`、`scripts/evaluate_pi05.py`。
训练配置为 `configs/pi05_horizon_writer_v1.json`；监督checkpoint24/64/128/192，strict400在64/128/192，平台与继续规则见design §8.2。
物化`--requests-json`复用resident source；当前监督闭环均用J0，训练侧states32–36与validation/test分开。

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

当前核验、现场资源和真实机制结果见progress；历史资源数字不构成新launch承诺。

## 验证和正式执行

共享源码修改后的CPU测试入口：

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```

仅文档修改检查相应diff。新模型的真实机制/profile与科学学习分开记录，工程测试不等于科学通过。
formal train/eval来自clean pushed commit的detached frozen worktree；launch前检查两GPU节点和strg01对应独立quota，
按真实吞吐使用同节点1–6张合适GPU，不继承旧GPU空闲、拓扑、32GiB预算或旧run的待执行清单。

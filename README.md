# EMBER

EMBER把exact task language与action-hidden教学视频，通过最终评估前的观看与自身实践编译为冻结π0.5 source的一套完整
38-target task-conditioned LoRA，研究它能否从未见初始化闭环完成任务，并取得有益视频增量与能力保持。

当前工作是[双meta读取、MT合并基底与多起点rank8 Writer](docs/designs/video_guided_proposal_writer_20261010.md)：
Owner已授权自主实施；不用T，fresh Gemma/action-expert读取meta与G共同学习，固定生成核后训练π完整决策。
原八条rank128教师已取得有限train净增；后继检验MT基底上的小rank残差，从首次G更新混合真实不同parent/H，
完成四个non-held任务的教师—G—实践决策—新初态闭环，再据证据扩大36任务与正式400。
[审阅入口与专家原文](docs/review_materials/20261010_writer_design_review/README.md)保留审阅依据和独立判断。
实际实现/运行状态看[progress](progress.md)，active设计和首批预算不等于已经取得控制传递或视频收益。

当前唯一运行面为`python -m ember.proposal_writer.run`，拥有双meta读取、完整因子CFM、三head决策及独立训练教师。
官方固定参数episode、真实十步执行与现场恢复快照由`ember.writer.practice`承接。
上一轮功能修订算法及专用测试已退役，源码可从Git c76615fc及原formal冻结版本复核。
科学合同见[功能修订编译器](docs/designs/functional_revision_compiler_20261010.md)；首批fresh FM360、72刷新与固定400已完成，
[完整消费](docs/analyses/functional_revision_learning_20261010.md)为143/400，对强MT153尚无净增。
当前授权、写入分工及后继只看[progress](progress.md)，导数接通不等于方法有效。

## 阅读与目录

| 入口 | 职责 |
| --- | --- |
| [Owner要求](docs/current_owner_requirements.md)／[AGENTS](AGENTS.md) | 稳定目标、科学与执行边界、协作职责 |
| [progress](progress.md)／[task_plan](task_plan.md) | 当前状态与计划；不从历史许可恢复执行 |
| [concept](docs/concept.md) | 科学对象、信息流与证据标准 |
| [findings](findings.md)／[research_history](docs/research_history.md) | 跨轮发现、正负历史、专家意见和原件索引 |
| `src/ember/proposal_writer/` | 当前完整参数G、独立三head π、双meta读取、训练教师与唯一CLI |
| `src/ember/writer/practice/` | 通用固定episode、十步执行、真实H与现场恢复快照 |
| `src/ember/operator_writer/` | 复用native/data与封存operator读取；旧训练CLI已退役 |
| `src/ember/writer/` | 共享视频、FM、重放、拓扑与物化调度；旧视频Writer仅保留bank/配置读取 |
| `src/ember/source_sft/`、`pi05_source_*`、`expert_manifold/` | Source、共享MT-BC与授权task expert组件 |
| `src/ember/pi05_eval/`、`pi05_eval_*.py` | 官方配对闭环、scene、persistent动态队列与原行校验 |
| `scripts/` | 环境构建、数据封存、Source/MT/expert CLI、官方评测、比较与固定A28读出 |
| [configs](configs/README.md) | 数据authority、当前spec及必要封存配置；不是实验许可 |
| `tests/` | 当前实际入口与稳定科学、数据、配对及恢复合同的CPU检查 |
| `docs/designs/`、`docs/analyses/`、`docs/review_materials/` | 原设计、机制论证、专家材料和小型原始证据；历史文字按时点解释 |
| `data/`、`models/`、`runs/`、`.venv/` | ignored本地资产与环境；大资产不入Git |

## 环境与命令

Python 3.12及依赖由[pyproject.toml](pyproject.toml)和`uv.lock`固定，首次构建使用
[scripts/bootstrap_env.sh](scripts/bootstrap_env.sh)，其中`scripts/zig-cxx`服务既有本地编译。
既有环境与Source/data/tokenizer直接复用。以下help可核对接口，运行仍需满足当前合同：

```bash
export PYTHONPATH="$PWD/src"
python -m ember.proposal_writer.run --help
python scripts/operator_joint_readouts.py --help
python scripts/evaluate_pi05.py --help
python scripts/compare_pi05_results.py --help
python scripts/train_source_base.py --help
python scripts/train_source_sft.py --help
python scripts/train_task_experts.py --help
python -m pytest -q
```

正式train/eval使用clean pushed commit的detached frozen worktree；并发开发隔离，集成回main串行协调。
所有新增EMBER资产和产物放data1，存储与GPU准入按AGENTS；历史data0资产只读，不随整理迁移。

## 历史读取与资产可用性

`operator_writer/bank.py`、`writer/evaluation.py`、`writer/conditional_velocity_bank.py`及
`demonstration_learning/bank.py`保留实际封存bank消费者。旧Writer训练器、已关闭的专用诊断脚本与干预执行入口
由Git及各run登记的冻结commit保存；读取历史原行不恢复旧训练或干预。
旧spec因当前消费者的metadata继承和封存provenance而保留，不能仅凭目录年龄删除。

旧Writer清理前实现可从`797ae01f`读取；条件A重表达诊断生成入口已退役，原八条件诊断采用其实际冻结`923ff89b`。
当前清理记录见[progress](progress.md)与[存储清理摘要](docs/analyses/workspace_cleanup_20261008.json)。已退休的checkout可从登记Git commit重建，
当前bank/spec实际消费者依赖的冻结路径保留。
代码存在不保证资产可重放：原manifest记录历史完整状态，现状还要看资产旁的
`checkpoint_retirement.json`与`payload_retirement.json`。`weights_only`不能exact resume，
`metadata_only`没有模型权重；冻结版本中的旧CLI可能要求完整trainer，不能把剩余权重称为完整恢复资产。
source/dataset、关键权重、formal原始rows/metrics及当前依赖保留；科学负结果不会被整理抹去。

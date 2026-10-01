# EMBER

EMBER把exact task language与action-hidden教学视频，在rollout前一次编译为冻结π0.5 source的一套完整
38-target task-conditioned LoRA，研究它能否从未见初始化闭环完成任务，并取得有益视频增量与能力保持。

当前唯一Writer运行入口是`python -m ember.operator_writer.run`，计算由
[conditional_read_write.py](src/ember/operator_writer/conditional_read_write.py)和
[model.py](src/ember/operator_writer/model.py)拥有：因果c/d解释、A₀感知S、最终A再编译M、完整FM共同学习。
科学合同见[条件读写设计§13](docs/designs/conditional_read_write_architecture.md#13-首批实施与完整学习检验2026-10-01授权)；
当前授权、运行、写入分工和结果只看[progress](progress.md)，架构接通不等于方法有效。

## 阅读与目录

| 入口 | 职责 |
| --- | --- |
| [Owner要求](docs/current_owner_requirements.md)／[AGENTS](AGENTS.md) | 稳定目标、科学与执行边界、协作职责 |
| [progress](progress.md)／[task_plan](task_plan.md) | 当前状态与计划；不从历史许可恢复执行 |
| [concept](docs/concept.md) | 科学对象、信息流与证据标准 |
| [findings](findings.md)／[research_history](docs/research_history.md) | 跨轮发现、正负历史、专家意见和原件索引 |
| `src/ember/operator_writer/` | 唯一活跃Writer；共享native/data/credit/runtime及封存operator读取 |
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
python -m ember.operator_writer.run --help
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

本次退役的旧实现可从清理前快照`797ae01f`读取，更早运行须采用其实际训练/读取commit。
代码存在不保证资产可重放：原manifest记录历史完整状态，现状还要看资产旁的
`checkpoint_retirement.json`与`payload_retirement.json`。`weights_only`不能exact resume，
`metadata_only`没有模型权重；冻结版本中的旧CLI可能要求完整trainer，不能把剩余权重称为完整恢复资产。
source/dataset、关键权重、formal原始rows/metrics及当前依赖保留；科学负结果不会被整理抹去。

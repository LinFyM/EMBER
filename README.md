# EMBER

EMBER把exact task language与action-hidden教学视频，在rollout前一次编译为冻结π0.5 source的一套完整
38-target task-conditioned LoRA，研究它能否从未见初始化闭环完成任务，并取得有益视频增量与能力保持。

当前唯一Compiler运行入口是`python -m ember.experience_compiler.run`，由
[model.py](src/ember/experience_compiler/model.py)、[decoder.py](src/ember/experience_compiler/decoder.py)
与[learning.py](src/ember/experience_compiler/learning.py)拥有：教学直接编译完整rank128 LoRA，
当前LoRA真实实践，累计后果在空间压缩前改变教学重读，共享U修订Q与全部38-target参数。
观看/实践次数由实际success或环境step预算产生；condition内不运行轨迹拟合optimizer。
科学合同见[经验条件完整参数编译器](docs/designs/experience_conditioned_compiler_20261009.md)；
当前授权、运行、写入分工和结果只看[progress](progress.md)，架构接通不等于方法有效。

## 阅读与目录

| 入口 | 职责 |
| --- | --- |
| [Owner要求](docs/current_owner_requirements.md)／[AGENTS](AGENTS.md) | 稳定目标、科学与执行边界、协作职责 |
| [progress](progress.md)／[task_plan](task_plan.md) | 当前状态与计划；不从历史许可恢复执行 |
| [concept](docs/concept.md) | 科学对象、信息流与证据标准 |
| [findings](findings.md)／[research_history](docs/research_history.md) | 跨轮发现、正负历史、专家意见和原件索引 |
| `src/ember/experience_compiler/` | 唯一活跃Compiler；直接解码、实际实践、共享信用/恢复与condition动态评测 |
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
python -m ember.experience_compiler.run --help
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

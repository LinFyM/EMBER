# EMBER

EMBER研究如何把任务语言和action-hidden正确教学视频，一次性编译为冻结机器人policy的一套task-conditioned LoRA，
使policy能从未见初始化闭环完成任务。

```text
exact task language + ordered teaching video set
    -> shared Writer
    -> one complete task LoRA
    -> frozen π0.5-LIBERO source policy
    -> closed-loop task execution
```

项目强调从视频提取跨初始化成立的高层程序，而不是逐帧复制teacher轨迹。语言提供任务query与目标，视频必须提供
正确过程的动态Value和有向顺序证据。

## Scientific contract

- teacher videos可为一条或多条，但全部action-hidden；
- 每条video内部保序，多video集合置换不变；
- Writer在rollout前只运行一次，并生成一套完整38-target LoRA；
- source policy在Writer训练和部署时冻结；
- 不读teacher action/proprio/reward/task ID，不挑video，不平均LoRA，不融合checkpoint；
- 正式选择只认single-checkpoint strict paired400 closed-loop结果；
- 高分还必须具备相邻checkpoint稳定性、same-task不同视频鲁棒性和correct相对wrong/shuffled/reversed/no-video的
  因果优势。

完整问题定义见[concept](docs/concept.md)，owner稳定要求见[current owner requirements](docs/current_owner_requirements.md)。

## Repository state

动态状态不写入README或AGENTS。请按以下三个仓库级文件查看：

- [task_plan.md](task_plan.md)：当前goal、完成标准和工作计划；
- [findings.md](findings.md)：跨实验仍成立的持久结论；
- [progress.md](progress.md)：当前进度、运行状态和即时边界。

完整架构与实验账本见[research history](docs/research_history.md)。

## Repository map

```text
EMBER/
├── AGENTS.md                     # 稳定项目总览、科研与工程原则
├── task_plan.md                  # 当前计划
├── findings.md                   # 持久发现
├── progress.md                   # 当前进度
├── docs/
│   ├── concept.md                # 第一性原理问题定义
│   ├── current_owner_requirements.md
│   ├── research_history.md       # 唯一实验/架构ledger
│   ├── architecture_reasoning.md # 从证据到未来设计约束的逐步推理
│   ├── benchmark_validity_report.md
│   └── novelty_and_landscape.md
├── configs/                      # 固定数据、source、LoRA与保留运行authority
├── scripts/                      # canonical launch/evaluation/analysis entrypoints
├── src/ember/
│   ├── source_sft/               # source adaptation baselines
│   ├── expert_manifold/          # task-expert训练/评测；名称因artifact兼容保留
│   ├── writer/                   # canonical LMMPC Writer训练与部署
│   ├── reward/                   # train24 privileged reward utilities
│   └── pi05_eval/                # paired evaluator、queue与历史分析
└── tests/
```

`data/`、`models/`、`runs/`和`evidence/`是本地大资产/正式证据根，均由`.gitignore`排除，不应复制或提交。

## Canonical entrypoints

- `scripts/train_source_base.py`：过滤LIBERO-90 source policy训练；
- `scripts/train_source_sft.py`：shared Source-SFT参考；
- `scripts/train_task_experts.py`：train24 task-local expert参考；
- `scripts/train_as_writer.py`：LMMPC Dynamic-K Writer训练；
- `scripts/evaluate_pi05.py`：source/expert/Writer strict paired评测；
- `scripts/analyze_task_expert_bank.py`：task-expert诊断。

已退役架构没有平行可执行入口；精确旧代码和设计通过Git快照检索。

## Environment and tests

环境要求Python 3.12、CUDA 12.8栈与仓库锁定的`uv.lock`。首次安装需把cache放在storage-backed目录：

```bash
export EMBER_CACHE_ROOT=/data1/user/ymdai/.cache/ember
scripts/bootstrap_env.sh
```

本地host路径和私有环境变量放在未跟踪的`.env.local`。CPU验证：

```bash
set -a
source .env.local
set +a
PYTHONPATH=src .venv/bin/pytest -q
```

GPU launch前必须按`AGENTS.md`同时检查gpu01/gpu02、对应quota、clean frozen commit和完整run contract。

## Historical retrieval

active tree只保留少数结构锚点和统一ledger。整理前的完整设计语料可从commit
`8553b613de7791df50e0f3ef85678fcaca1cac0c`读取：

```bash
git show 8553b61:docs/<historical-design>.md
git show 8553b61:findings.md
git show 8553b61:task_plan.md
```

formal run的精确command、manifest、raw rows和分析应从对应`runs/outputs/` artifact读取，不能由旧文档中的动态
“下一步”恢复执行。

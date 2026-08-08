# EMBER

EMBER研究能否把没有目标机器人action标注的教学视频，一次性编译成能让
frozen VLA完成对应任务的完整task-specific LoRA。当前唯一活动方法是one-shot
Video-Conditioned Expert-Manifold Topological Writer：

```text
exact task language + exactly one action-hidden teaching video
                    -> shared Writer
                    -> sealed rank-16 task LoRA
                    -> frozen π0.5-LIBERO source policy
```

Writer只在rollout前运行一次；policy随后依据实时observation/state闭环执行。
Writer不读取teacher action、proprio、reward、terminal、task ID、filename或隐藏
normalization。

## 当前状态

EMBER已经迁回BCI。多卡训练、exact resume、动态队列评测、NUMA与deferred-NCCL入口均已完成
迁移验收；当前A40边界是每次live核对`gpu01/gpu02`、只用空闲卡、合计最多6张并显式设置
`NCCL_P2P_DISABLE=1`。正式研究必须从本项目目录、clean pushed commit和fresh identity启动，
并遵守`AGENTS.md`和当前owner授权。

- `main`保留迁移封存历史；当前BCI写分支为`codex/bci-continuation`。
- 旧Target-Bound、Semantic Factor-Basis及K4路线均已封存并负裁决；不得从历史“下一步”恢复。
- 24套train-task rank-16 task experts已在clean`81101fe`完成统一step1000。正式
  development-train闭环step250/500/1000=`432/557/624` of 1200；step1000在四suite均不低于
  step500且24/24 tasks非零，因此下一阶段是沿原root统一exact-resume到2000并评1500/2000。
- full24 expert几何与phase16×3072 action-hidden feature profile均已完成；feature formal cache已
  seal。当前还没有Expert-Manifold Writer checkpoint或新held strict成绩。
- 当前科研结论、下一实验边界看
  [`docs/active_session_handoff.md`](docs/active_session_handoff.md)。
- A100清理、Git/SSH/重下载分流、BCI路径映射和新Codex接手步骤看
  [`docs/a100_to_bci_migration_handoff.md`](docs/a100_to_bci_migration_handoff.md)。

## 研究基线与最新结论

- 共同起点是generic `lerobot/pi05_base`，不是读过目标LIBERO-40 actions的
  `pi05_libero`。LIBERO-90 specification-only audit排除了19个与目标40 exact
  semantic/composition重合的source tasks；71 tasks×50成功episodes用于共享
  source-base action-SFT。
- 目标benchmark为LIBERO-Spatial/Object/Goal/Long共40 tasks。固定development
  split是24 train / 8 validation / 8 test；不得按结果改变task IDs。
- frozen π0.5-LIBERO source base约`48/400`；corrected mixed-task rank-128
  Source-SFT observed-best为`109/400`。
- v5.2旧recipe的single winner五臂为`132/138/74/82/83`，是当前最好的视频
  特异性形态；v6 fast task-complete winner为`143/135/125/128/129`，absolute最高
  但视频margin弱。
- recipe互换后两者并不呈同向变化：v5.2 task-complete为
  `120/109/107/111/124`，v6 old recipe为`121/122/111/84/47`。task-complete在两
  架构上都压弱动态写出，但correct absolute分别下降和上升，故架构与训练方式必须
  联合分析，不能把post-v5设计一棒子判死，也不能简单退回旧recipe。
- 最新CV-ADR RAW完整correct400曲线为
  `76/111/99/117/77/69/80/82`，normalized GROUP4为`82/77/73/110`。两者均未解决
  task漂移；matched梯度分析显示video主效应约`.1%`，query/flow噪声主导，functional
  surrogate继续改善时closed-loop会退化。
- 当前Expert-Manifold先用真实task-local SFT LoRA定义policy-effective参数流形，再让视频条件的
  168-chunk rank-axial decoder生成完整38-target LoRA。task experts只提供训练target，不是部署输入，
  也不能自行证明视频时序因果性；最终仍由correct/same/wrong/shuffled/reversed严格配对裁决。

## 不变合同

- one teacher video生成一套完整rank-16 LoRA；不平均多video、多LoRA或checkpoint。
- frame stride固定为5；正式LIBERO preprocessing、horizon、dummy settling、成功
  终止和frozen normalization不变。
- validation/test actions不产生Writer梯度；test数据边界由`AGENTS.md`管理。
- 不使用teacher action/state/reward作为Writer输入，不增加额外shared adapter、
  bank、checkpoint fusion或静态旁路。
- GPU设备范围必须以owner迁移后重新给出的BCI authority为准；A100时期的GPU4–7
  约束不能自动复制到另一台机器。
- sealed历史config和artifact contract中的旧绝对路径是provenance，不应原位改写；
  新运行通过CLI显式传入source/checkpoint/tokenizer/data/output路径。

## BCI目录、环境与路径

BCI上的canonical入口是`/data1/user/ymdai/projects/EMBER`。代码和项目资产按项目
归并，不再从个人目录顶层按资源类型拆分：

```text
EMBER/
├── data/       # datasets与LIBERO simulation assets
├── models/     # tokenizer及独立模型资产
├── runs/       # 训练、评测、checkpoint、日志与运行验收
├── evidence/   # 迁移清单与验收证据
├── .venv/      # 本项目Python环境
└── .cache/     # 本项目可重建缓存
```

进入仓库后执行`source .venv/bin/activate`即可自动加载`.env.local`中的BCI本地路径，
不需要逐项手工设置。环境仍由`pyproject.toml`和`uv.lock`约束，必要时可运行
`scripts/bootstrap_env.sh`原位校验或修复。

评测preflight的项目容量根也可显式覆盖：

```bash
export EMBER_STORAGE_ROOT=/path/to/bci/EMBER
export EMBER_STORAGE_CAP_BYTES=REPLACE_WITH_OWNER_CAP
export EMBER_LIBERO_ASSETS_ROOT=/path/to/libero-assets/revision
```

训练与评测入口继续要求显式资产路径：

```text
scripts/train_as_writer.py
scripts/evaluate_pi05.py
```

完整BCI映射和恢复校验在迁移handoff中。

## 阅读顺序

在只读了解或迁移时先读：

1. `AGENTS.md`
2. `docs/a100_to_bci_migration_handoff.md`
3. `docs/active_session_handoff.md`
4. `docs/execution_brief.md`
5. `docs/action_forecast_writer_contextual_value_dual_read_design.md`
6. 远端Target-Bound分支的
   `docs/action_forecast_writer_target_bound_role_program_design.md`
7. `findings.md`与`progress.md`

改变实验状态前仍必须完整阅读`AGENTS.md`列出的全部authority文件；这里的短顺序
不能替代该要求。

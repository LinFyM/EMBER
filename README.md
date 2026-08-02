# EMBER

EMBER研究能否把一条没有目标机器人action标注的教学视频，一次性编译成能让
frozen VLA完成对应任务的完整task-specific LoRA：

```text
task language + exactly one action-hidden teaching video
                    -> shared Writer
                    -> sealed rank-16 task LoRA
                    -> frozen π0.5-LIBERO source policy
```

Writer只在rollout前运行一次；policy随后依据实时observation/state闭环执行。
Writer不读取teacher action、proprio、reward、terminal、task ID、filename或隐藏
normalization。

## 当前状态

2026-08-02因A100主机即将到期，新的训练、评测和架构设计已暂停。当前没有活动
EMBER/MemLLM训练、评测或tmux。项目正在迁回BGR；迁移操作由后续智能体执行，本
checkout只负责清理和形成可验证交接。

- `main`保留已封存的CV-ADR canonical实现和证据。
- Target-Bound Role-Preserving Program已在远端分支
  `origin/codex/target-bound-role-program`实现到CPU vertical path，commit
  `b260a57a94dc21bd3446b212bfa42f71b037ce13`；尚未做任何GPU profile、训练或rollout。
- 当前科研结论、下一实验边界看
  [`docs/active_session_handoff.md`](docs/active_session_handoff.md)。
- A100清理、Git/SSH/重下载分流、BGR路径映射和新Codex接手步骤看
  [`docs/a100_to_bgr_migration_handoff.md`](docs/a100_to_bgr_migration_handoff.md)。

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
- Target-Bound的下一目标是在真实38-target拓扑中保留Core semantic carrier和
  private Action/Effect/Change causal channels，同时不破坏coherent高增益LoRA几何。
  它仍是假设，不是已验证结果。

## 不变合同

- one teacher video生成一套完整rank-16 LoRA；不平均多video、多LoRA或checkpoint。
- frame stride固定为5；正式LIBERO preprocessing、horizon、dummy settling、成功
  终止和frozen normalization不变。
- validation/test actions不产生Writer梯度；test数据边界由`AGENTS.md`管理。
- 不使用teacher action/state/reward作为Writer输入，不增加额外shared adapter、
  bank、checkpoint fusion或静态旁路。
- GPU设备范围必须以owner迁移后重新给出的BGR authority为准；A100时期的GPU4–7
  约束不能自动复制到另一台机器。
- sealed历史config和artifact contract中的旧绝对路径是provenance，不应原位改写；
  新运行通过CLI显式传入source/checkpoint/tokenizer/data/output路径。

## 环境与路径

本仓库使用`pyproject.toml`和`uv.lock`重建环境，不迁移`.venv`。评测preflight的
个人容量根可通过环境变量设置：

```bash
export EMBER_STORAGE_ROOT=/path/to/bgr/user/root
export EMBER_STORAGE_CAP_BYTES=REPLACE_WITH_OWNER_CAP
export EMBER_LIBERO_ASSETS_ROOT=/path/to/libero-assets/revision
```

训练与评测入口继续要求显式资产路径：

```text
scripts/train_as_writer.py
scripts/evaluate_pi05.py
```

完整BGR映射和恢复校验在迁移handoff中。

## 阅读顺序

在只读了解或迁移时先读：

1. `AGENTS.md`
2. `docs/a100_to_bgr_migration_handoff.md`
3. `docs/active_session_handoff.md`
4. `docs/execution_brief.md`
5. `docs/action_forecast_writer_contextual_value_dual_read_design.md`
6. 远端Target-Bound分支的
   `docs/action_forecast_writer_target_bound_role_program_design.md`
7. `findings.md`与`progress.md`

改变实验状态前仍必须完整阅读`AGENTS.md`列出的全部authority文件；这里的短顺序
不能替代该要求。

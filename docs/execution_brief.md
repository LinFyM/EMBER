# EMBER Current Execution Brief

更新时间：2026-08-02。本文是操作层authority；科研结果取
`docs/active_session_handoff.md`，迁移取`docs/a100_to_bgr_migration_handoff.md`，
长期边界取`AGENTS.md`。

## 1. 当前操作状态

- A100即将到期，owner已暂停新训练、评测、GPU profile和下一架构设计。
- 当前无EMBER/MemLLM训练、评测、torchrun或tmux需要继承。
- 当前允许：清理、迁移准备、只读hash/manifest检查、路径可移植性、CPU回归、文档、
  Git commit/push。
- 当前不允许：Target-Bound B20 profile、exact resume、formal训练、rollout、重跑旧
  smoke，或利用迁移窗口开始新的架构设计。
- 迁移完成后必须由owner重新授权；本文件不自动授予BGR GPU使用权。

## 2. Canonical Git state

迁移准备前EMBER：

```text
main/origin-main = f0b123f20f531baf4bfc5c6f75eb96af27f33ac1
current canonical implementation = CV-ADR
Target-Bound remote branch commit = b260a57a94dc21bd3446b212bfa42f71b037ce13
```

本次迁移准备提交会成为新的`origin/main`。Target-Bound只在远端feature branch，
没有合并到main；它完成CPU vertical path但没有GPU实验。A100本地只保留main一个
worktree/branch，历史refs由完整bundle保存。

迁移后默认动作：clone GitHub main、fetch Target-Bound远端分支、核验commit；不要
复制`.git`或恢复所有Codex refs。路径可移植性提交应先rebase/merge到Target-Bound，
再运行其CPU回归。

## 3. Canonical assets

### Frozen source policy

```text
source run:
/data/ymdai/outputs/ember/pi05_source_base_v1_seed7_1k_e2cc238_20260722

source checkpoint:
/data/ymdai/outputs/ember/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000

selected raw policy SHA256:
60ea7ee898629321cf34522e5f0e45f4f1c2659c5f5dbc7b02ed9eb46a8cdf36
```

rejected EMA与optimizer/DDP/scheduler训练态已经删除；raw policy、trainer state和
原manifest保留，并通过formal source inspector。它仍能作为所有Writer/SFT/eval的
frozen source asset，但不能exact-resume source training。

### Data and tokenizer

```text
tokenizer SHA256:
8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6

LIBERO dataset revision:
f13aa24a3da8c43c7225569f28c562979fa0e35a

canonical feature cache:
pi05_writer_feature_cache_v2_development32_raw_e4c19f9_b32_20260722

LIBERO simulation assets:
lerobot/libero-assets@0b3ea86be5fe169d0fd036ae63d1070ec09e90f6
586 files; file-list SHA256 721aa2484de396be5267e936f115ddd5f03ffd12e0849cc1cd05bb17454996b9
```

BGR路径和容量上限不得从A100字符串猜测；先设置`EMBER_STORAGE_ROOT`与owner给出的
`EMBER_STORAGE_CAP_BYTES`，再由CLI显式传入source/checkpoint/tokenizer/data/output，
并用`EMBER_LIBERO_ASSETS_ROOT`指向精确simulation snapshot。历史sealed artifacts中的
旧绝对路径不改写。

## 4. Latest scientific decision

CV-ADR RAW在macro50/100/150/200/250/300/350/400的paired correct400为：

```text
76 / 111 / 99 / 117 / 77 / 69 / 80 / 82
```

single winner为macro200；第二小时出现明显能力崩落，未做五臂。normalized GROUP4
在cycle50/100/150/200为`82/77/73/110`，四点均值和winner均低于RAW，且内部A/E/D
职责进一步变弱，不作为默认operator。

matched诊断把late gradient方差定位为video主效应约`.1%`、query/flow主导；
functional loss继续改善而closed-loop`117→82`。因此后续必须联合处理semantic
carrier、causal write value、single-video低SNR和functional/closed-loop错位。

v5.2/v6的old/task-complete四格又证明recipe作用依赖架构。不得把post-v5低分架构
整体判死、不得简单退回old six-update，也不得恢复CP-24、gate、scale、rank loss、
multi-video或checkpoint fusion作为默认解。

## 5. Next experiment after migration and explicit authorization

候选：Target-Bound Role-Preserving Program。完整design只在远端分支：

```text
origin/codex/target-bound-role-program
docs/action_forecast_writer_target_bound_role_program_design.md
```

执行顺序固定：

1. 核验BGR Git、data、source、tokenizer、output root和新GPU authority；
2. 合入main上的`EMBER_STORAGE_ROOT`路径修复并跑CPU regression；
3. 从clean frozen commit建立独立formal worktree；
4. 只在owner授权设备上做最长105-frame、4-rank、B20真实profile；
5. fresh identity 0→1，再exact-resume1→3；
6. 通过shape、identity、freeze、causality、gradient、OOM和resume后，fresh训练
   cycle0→200、every25 checkpoint；
7. paired correct400评测50/100/150/200；
8. 根据absolute、breadth、右端趋势、task churn和A/E/D→effective BA→action传递
   决定是否第二小时；不机械按150停止。

profile/smoke root必须全新，不得使用A100上的smoke权重或正式output路径。B20只有
真实OOM或连续非有限才降，不扫描B17–B19/B21。

## 6. Formal training contract

除非Target-Bound design在结果前更新authority，否则：

- one teacher video → one complete rank-16 LoRA；
- frame stride=5；
- 24 train tasks每cycle完整覆盖，每task一条video；
- B20同task跨episode独立action queries，task内mean后task等权；
- task-query RAW full24，一次clip、一次AdamW、一次scheduler/cycle；
- 每25 cycle checkpoint；fresh identity；
- validation/test actions不读、不产生梯度；test video不进入development训练；
- frozen source policy trainable parameters=0；
- cost-balanced long-first，真实最长105-frame验证；
- exact resume不得改写既有checkpoint payload，task/video/query/RNG/cursor连续。

任何operator改变必须fresh incompatible schema并在看到outcome前seal。pairwise负
gradient cosine本身不授权projection；无真实candidate conflict时投影必须退化为raw
mean。

## 7. Paired evaluation contract

- 8 validation tasks×50 states，共400 rollouts；
- 每task 50 teacher videos无放回，每条一次；
- candidate arms严格配对state、video ordinal、env/policy/noise schedule；
- dynamic cost-balanced queue，long shards优先，persistent model/env；
- 只选择single checkpoint；不融合或挑video；
- candidate报告aggregate、per-task、per-suite、gained/lost、breadth、top-task集中度、
  success-set Jaccard、train/held loss、LoRA norm和数据曝光量；
- 只有strong single winner才做correct/same/wrong/shuffled/reversed五臂；
- shuffled/reversed必须真实改变输入帧顺序后完整forward。

内部分析优先effective BA、Gram、singular spectrum、norm和fixed-query policy action，
不用raw A/B gauge符号做跨模型结论。每task报告Core、Program、compiler、factor、
effective BA与action，定位最早失效接口。

## 8. Launch preflight on BGR

每个expensive formal run只做一次live preflight并记录：

1. canonical repo、branch/commit、clean status、origin同步；
2. frozen worktree commit与sealed config/command；
3. owner新授权的GPU IDs、进程、显存、温度和CUDA runtime；
4. 不查询或触碰未授权GPU；
5. BGR个人storage root、容量上限和预计峰值；
6. source、checkpoint、tokenizer、data root存在且identity正确；
7. output root此前不存在；tmux/log名称无冲突；
8. DDP ranks与`CUDA_VISIBLE_DEVICES`只包含获批设备；
9. 不干扰他人进程。

A100上的NUMA node1和物理GPU4–7只是历史合同，不得复制到BGR。

## 9. Evidence and retention

- formal run保留config、command、commit、parameter count、macros/videos/queries、
  wall/GPU-hours、checkpoint curve、per-task、五臂（若过门）、internal analysis、
  root cause和retain/reject。
- `/data/ymdai/outputs/ember`当前保留60个checkpoint roots和406个complete eval roots；
  它们用于训练漂移与架构×recipe审计，迁移时不能只留winner。
- writer/eval LoRA caches、profile/resume、reseal和退役SmolVLA已按manifest清理；历史
  文档引用已删工程root不是重新运行指令。
- 不提交checkpoint、dataset、cache或大binary。formal artifacts经SSH迁移，源码和
  文档经GitHub迁移。

## 10. Git and handoff

meaningful状态更新：

- `AGENTS.md`
- `docs/a100_to_bgr_migration_handoff.md`
- `docs/active_session_handoff.md`
- 本文件
- 对应architecture design
- `task_plan.md`
- `findings.md`
- `progress.md`

commit只含任务相关改动并push。A100 Codex不迁移；新Codex从Git文档、formal
artifacts和migration manifests接手。迁移完成前与owner恢复授权前，停止在无GPU
作业状态。

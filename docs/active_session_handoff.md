# EMBER Active Session Handoff

更新时间：2026-08-02。本文只记录迁回 BGR 前的当前真相。历史执行流水仍在
`progress.md`，证据与解释仍在`findings.md`及各架构设计文档；不要用其中旧的
“当前”“下一步”覆盖本文。

## 1. 当前边界

- owner 因 A100 主机即将到期，已暂停新的训练、评测、GPU profile和新架构设计；
  当前任务只做清理与迁移准备。
- 当前没有需要继承的 EMBER/MemLLM 训练、评测、torchrun 或 tmux 进程。
- EMBER `main` 的科研 authority 是 CV-ADR 最终封存，迁移准备前
  `HEAD=origin/main=f0b123f20f531baf4bfc5c6f75eb96af27f33ac1`。
- Target-Bound Role-Preserving Program 已在远端分支
  `origin/codex/target-bound-role-program`实现，commit
  `b260a57a94dc21bd3446b212bfa42f71b037ce13`。它只完成 CPU shape、identity、
  causality、gradient、checkpoint 等结构验证；没有做 B20 profile、resume、训练或
  rollout。不得把它写成实验结果。
- 所有辅助 worktree、本地实验 branch 和旧 stash 已在完整 Git bundle 验证后删除；
  `main`是唯一在 A100 上保留的本地 checkout。Target-Bound仍由远端分支保存。
- 迁移步骤、路径映射、资产分流和新 Codex 接手顺序统一看
  [`a100_to_bgr_migration_handoff.md`](a100_to_bgr_migration_handoff.md)。

## 2. 最新 closed-loop 结论

### 2.1 CV-ADR RAW

canonical root：

```text
/data/ymdai/outputs/ember/pi05_as_writer_cvadr_rawfull24_taskcomplete_decay400_formal_dev_r4_b20_seed7_254ade4_20260802_retry1
```

- fresh identity 完成 macro0→400，192,000 action queries、9,600 one-video
  conditions，all finite、0 clip，validation/test action reads为0。
- paired correct400 在 macro50/100/150/200/250/300/350/400 为：

```text
76 / 111 / 99 / 117 / 77 / 69 / 80 / 82
```

- single winner 是 macro200=`117/400`。第二小时不是成熟化：200→250为
  16 gained / 56 lost，后段 LoRA norm没有坍缩而行为持续退化，因此未做五臂。
- macro200与400的matched梯度方差分解显示，video主效应仅约
  `.1211%/.1060%`且0/24 tasks主导；query约`48.59%/49.53%`，flow及
  query×flow约`48.78%/48.50%`。24/24 matched train functional loss继续下降，
  correct400却`117→82`。
- 晚期factor block约占task-gradient energy的`94%`；参数段方向在低LR仍不稳定，
  held functional loss横盘。最可信根因是视频条件梯度低SNR、query/flow噪声、
  shared compiler写出与closed-loop有效流形错位共同作用，不是单纯LR、rank或norm。

内部根：

```text
/data/ymdai/outputs/ember/pi05_as_writer_cvadr_rawfull24_macro0200_internal_exact50_seed7_ff988dc_20260802
```

exact50确认Core与Program两路都必要，但Action/order仍弱：remove-A只在1/8 tasks达
预注册门，remove-D为5/8；same-task effective-BA centered variance/sample energy
约`.10494%`，fixed-action中位变化约`.00856%`。LoRA norm`64.24`、stable rank
`1.0072`，所以不是Target-Spectral式增益或coherence坍缩。

### 2.2 CV-ADR normalized GROUP4

canonical root：

```text
/data/ymdai/outputs/ember/pi05_as_writer_cvadr_group4_taskcomplete_decay400_formal_dev_r4_b20_seed7_51c0ba5_20260802
```

- 完成1200 physical updates=200 cycles、96,000 queries、4,800 videos，all finite、
  1 clip；cycle50/100/150/200 paired correct400为：

```text
82 / 77 / 73 / 110
```

- single winner cycle200/step1200=`110`，低于RAW winner`117`，四点均值
  `85.5<100.75`；breadth6、top2占`71.82%`，未解决能力轮换，不做五臂。
- GROUP4比RAW保留更多source successes（42/48 vs 34/48），但没有共同获得更多新
  能力。effective norm反而`64.24→72.06`，held loss略低而closed-loop更差。
- exact50中A+D、remove-A、remove-D职责门由RAW的`8/1/5 of 8`降为`0/0/0`；
  Effect-only到full BA的relative L2由`.06744`降为`.01882`，contextual-memory
  reversal由`.00607`降为`.00311`。它学到更大、更coherent、却更static和
  off-manifold的写入。

内部根：

```text
/data/ymdai/outputs/ember/pi05_as_writer_cvadr_group4_cycle0200_step1200_internal_exact50_seed7_51c0ba5_20260802
```

结论：normalized GROUP4、CP式负冲突解释和“减少optimizer gain即可稳定”均没有
获得支持。full24 raw mean很少直接对task candidate为负，pairwise negative cosine
不能自动解释漂移。

## 3. 架构×训练方法的关键反事实

以下四个single winner均从正式400-row paired artifacts逐行重验：

| 架构×recipe | correct | same | wrong | shuffled | reversed |
| --- | ---: | ---: | ---: | ---: | ---: |
| v5.2 old recipe, step900 | 132 | 138 | 74 | 82 | 83 |
| v5.2 task-complete, macro400 | 120 | 109 | 107 | 111 | 124 |
| v6 old recipe, step500 | 121 | 122 | 111 | 84 | 47 |
| v6 fast task-complete, macro400 | 143 | 135 | 125 | 128 | 129 |

必须继承的解释：

1. task-complete在v5.2和v6上都压弱Procedure→effective BA/action与顺序margin，
   但correct absolute分别`-12/+22`；架构和recipe不能独立判死。
2. matched 150-video visits时，v5.2 task-complete相对old为`-81`，v6为`+16`；
   v6的Visual Transition/Core-conditioned transition是正证据，但其selected
   `+22`又几乎由一个Object task的`+24`贡献，不能说漂移已解。
3. old recipe每task-cycle六次Adam会恢复更强slots/AdaLN/动态写出，也会产生低
   breadth和近正交参数轨迹；退回old recipe不是解法。
4. post-v5的v7、v8、v10、Loom、Recenter、Core-Program、Prior、Target-Spectral、
   SPG、UCP、AP和CV负结果都与训练operator混杂。它们各自的局部失败接口有正式
   证据，但不能据此宣布其全部思想在任意训练方式下无效。
5. functional loss下降不等于closed-loop改善；强行高rank/正交、全局scale、gate、
   B-only residual、多video/LoRA平均或checkpoint融合都没有当前依据。

四格正式联合审计root：

```text
/data/ymdai/outputs/ember/pi05_as_writer_v52_v6_recipe_video_causality_audit_seed7_20260802
```

analysis SHA256：
`98371337e2cf1f7cec09d04e81445b419fc21c654fe173cb081a4b5e63092efa`。

## 4. 当前代码与下一实验边界

`main`保持CV-ADR canonical path，便于解释所有已封存artifact。Target-Bound分支从
`51c0ba5`独立实现，核心职责为：

- 38个真实policy targets先读Core；
- target-bound地读取Action、Effect与Change；
- A/E/D使用private causal temporal channels和private rank reads；
- 16 rank coordinates最后展开；
- identities只进入Q/K，raw evidence进入V；
- conventional factor heads保持coherent near-rank1高增益，不加谱/正交约束。

迁移并由owner重新授权实验后，下一步不是重新设计或重跑旧profile，而是：

1. 从GitHub恢复`main`和`origin/codex/target-bound-role-program`；
2. 将本次main上的路径可移植性提交rebase/merge到Target-Bound分支；
3. 现场确认BGR GPU和数据路径，仅在获批设备上做最长105-frame B20 profile；
4. 验证fresh0→1→exact-resume1→3；
5. 若vertical path健康，从fresh identity训练首小时cycle0→200，评测
   50/100/150/200 paired correct400；
6. 只按absolute、breadth、趋势、漂移和内部A/E/D→BA→action传递决定第二小时。

不得从A100上的smoke/profile权重warm-start；不得自动启动。Target-Bound的完整设计
在远端分支文件：
`docs/action_forecast_writer_target_bound_role_program_design.md`。

## 5. 迁移时必须保留的EMBER科学资产

- frozen source raw policy：
  `/data/ymdai/outputs/ember/pi05_source_base_v1_seed7_1k_e2cc238_20260722`；
  policy SHA256
  `60ea7ee898629321cf34522e5f0e45f4f1c2659c5f5dbc7b02ed9eb46a8cdf36`。
  rejected EMA和训练resume状态已清理；它现在是inference/source asset，不是完整
  source-SFT resume包。
- canonical feature cache v2：
  `/data/ymdai/outputs/ember/pi05_writer_feature_cache_v2_development32_raw_e4c19f9_b32_20260722`。
- `/data/ymdai/outputs/ember`中保留的60个正式/历史训练checkpoint roots、406个完成
  evaluation roots及内部analysis。它们是训练漂移与架构×recipe复核的唯一证据，
  不能只迁winner。
- `/data/ymdai/logs/ember`、tokenizer、精确revision的426.57MB LIBERO simulation
  assets和`/data/ymdai/migration_manifests`。

cleanup已删除的profile/resume/reseal/cache路径若仍出现在历史文档中，表示工程
provenance，不表示artifact损坏，也不授权重跑。精确删除清单和SHA都在：

```text
/data/ymdai/migration_manifests/a100_cleanup_20260802
```

## 6. 新Codex接手顺序

本机Codex sessions、archive、auth、cache和worktree不迁移；它们不是authority。
新Codex在BGR上应先：

1. 核验Git HEAD、origin、工作区和迁移资产hash；
2. 完整阅读`AGENTS.md`要求的authority文件；
3. 优先读本文件、迁移handoff、`docs/execution_brief.md`、CV与Target-Bound设计；
4. 检查BGR实际路径并设置`EMBER_STORAGE_ROOT`、owner cap及
   `EMBER_LIBERO_ASSETS_ROOT`，所有source/checkpoint/tokenizer/data/output路径继续
   通过CLI显式传入；
5. 在owner恢复实验授权前保持无GPU作业状态。

旧Codex对话不能代替上述Git文档。任何与本交接冲突的历史“live”段落均视为过期。

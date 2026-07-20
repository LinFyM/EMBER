# Historical Gate -1 and Gate 0 Evidence

状态：历史证据摘要；不定义下一轮 split 或 runner。

## Gate -1

官方 SmolVLA/LIBERO checkpoint、BDDL、init-state、camera、controller、normalization 和视频生成路径已机械复现。action-hidden-video probe 的最终 bounded recovery：

| 指标 | 结果 |
| --- | ---: |
| ordered balanced accuracy | 19/24 = 0.7917 |
| same-scene wrong-video specificity | 19/24 = 0.7917 |
| paired both-correct | 15/24 |
| first-frame | 0.5208 |
| static median | 0.5417 |
| last-frame | 0.4792 |
| reversed | 0.4375 |
| shuffled | 0.5625 |
| drop-last-20% | 0.6458 |

ordered 对 static/reversed/shuffled 有明显优势，证明动作隐藏视频含任务相关时序信号。原 0.80 阈值、paired 和 drop-last 残差保留。owner 最终将 Gate -1 记为“passed with residuals”，不再为跨过 0.80 重跑。

## 旧 split 的作用和退役

旧 60/15/15 曾通过 specification-only role parser 修复一次 primitive coverage 缺陷；全过程未读取 held actions/reward/policy results。它证明：

- 90 条 instruction 可做 role-aware deterministic parsing；
- split 需要显式平衡 task-relevant roles，而不是靠 scene distractor；
- specification-only seal 可以防止 outcome-driven split。

owner 现改用同分布 70/10/10，因此旧 IDs、hash、normalization 和 manifest 全部退役。新 split 必须重新生成，不能把旧 seal 当 authority。

## Gate 0

历史 task 3/4 action-supervised LoRA closed-loop packet在 h16、每臂 n=32：

- task 3 base 22/32，LoRA 28/32；
- task 4 base 16/32，LoRA 20/32。

它只说明 task-local LoRA 能形成有用更新，证据覆盖有限。旧 Gate 的严格 CI 下界规则和“LoRA 必须进一步超过 source-trained base”解释已 superseded。

## Provenance

- 完整旧报告、configs、source 和 tests：Git commit `999df28`。
- 外部 raw evidence：本机 `EMBER_OUTPUT_ROOT` 下对应 Gate -1/Gate 0 checksummed directories。
- 当前活动合同：`docs/execution_brief.md`。

不要从本报告启动旧实验。

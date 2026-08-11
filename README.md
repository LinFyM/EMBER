# EMBER

EMBER研究能否把机器人教学视频中的任务知识一次性编译成策略参数：

```text
exact task language + one action-hidden teaching video
                    -> shared Writer
                    -> one complete rank-16 task LoRA
                    -> frozen π0.5-LIBERO source policy
                    -> closed-loop execution on unseen initializations
```

Writer只在rollout前运行一次。它不读取teacher action、proprio、reward、terminal、task ID、filename、
object pose或hidden normalization；video是唯一dynamic value，不允许language-only LoRA bypass。

## Current status

- workspace：`/data1/user/ymdai/projects/EMBER`
- branch：`codex/bci-continuation`
- 长期目标：同一shared method、同一single checkpoint的strict paired correct严格`>150/400`，并继续提高
  absolute、breadth、稳定积累和视频时序因果性。
- 当前目标未完成；历史最好single checkpoint仍是v6-fast macro400：
  `correct/same/wrong/shuffled/reversed=143/135/125/128/129`。
- PICK的raw-frame门通过，但world6 discarded full48 profile只因condition=`483.61515>200` non-pass；其余
  14项机制、动作与吞吐门通过，未获formal训练资格。
- PICK-GC已完成formal fresh`0→10`与strict paired400：`138/400`、breadth6，相对immutable macro0
  retained/gained/lost=`118/20/16`。它未过`correct>=144`与`lost<=8`门，resume、controls和参数sweep均关闭。
- OSG-PC的唯一world6 discarded profile在full48前因rank-local长尾触发600s NCCL watchdog；从run-contract发布
  到timeout至少`969.9709s`，相对matched baseline至少`1.912x>1.25x`。无mechanism report/checkpoint，formal、
  deployment与评测均关闭；当前full-replay per-success VJP执行图已退役，但不能据此否定所有success constraint。
- 当前唯一active successor design是SKNC：同一K4 panel只用`4/4` binary success认证condition key，并把完整
  shared Program update限制在这些keys的nullspace；每train task只持久化第一条success key。它不保存/回放
  trajectory action，也不求reward VJP。canonical实现、fresh schema与完整CPU回归`334 passed`已完成；尚未
  profile、训练或评测，下一步是clean pushed head的唯一discarded live profile。

最新uniform pivot-rank14路线已终局否决：

| arm | correct | breadth | 相对old134 retained/gained/lost |
| --- | ---: | ---: | ---: |
| old full-rank cache | 134 | 6 | reference |
| compiler-only rank14 | 138 | 7 | `119/19/15` |
| online-regenerated rank14 | 128 | 7 | `113/15/21` |

compiler-only虽然净增4，但预注册要求lost`<=10`，实际lost15；且总增益由Long1净`+11`掩盖Spatial/Object
净`-3/-4`。compiler→online又发生`23 lost/13 gained`。因此uniform compression和online regeneration
都是独立换手源；Gate C、cycle1、controls和训练均未授权。该结论只淘汰当前uniform rank14合同，不淘汰
视频、Reward、continuous tangent或所有parameter-manifold思路。

PICK证明frozen-policy innovation能稳定读取视频与顺序并传到Program、LoRA和action；它最早在full48 key
conditioning失效。PICK-GC把condition修到`152.61`、让FP32 Program连续积累并传出非零effective-BA tangent，
但strict仍只有138且lost16。故最早失败接口已经推进到blind train24 source-action cotangent不能覆盖held
on-policy有用support与共同积累；不应再用扩大LoRA norm/rank或恢复训练补救。
详见[`docs/action_forecast_writer_policy_innovation_goal_causal_key_design.md`](docs/action_forecast_writer_policy_innovation_goal_causal_key_design.md)。

## What matters scientifically

EMBER不是“生成一个看起来像SFT的LoRA”就算成功。最终方法必须同时具备：

- 高strict closed-loop absolute；
- 多task breadth与低checkpoint能力轮换；
- correct明显优于wrong、shuffled、reversed和no-video；
- same-task不同teacher video鲁棒；
- 视频语义/时序真实传到effective LoRA和policy action；
- one-shot生成后对不同初始化有效，而非复制示范的低层轨迹。

LoRA norm、stable rank、能量分布、factor cosine、reconstruction/functional loss与hidden差异只是诊断指标。
历史已经多次出现“几何更健康、loss更低、内部margin更漂亮，但closed-loop更差”。正式选择只认同一
single checkpoint的paired400及视频controls。

## Core evidence in one view

- frozen source base：`48/400`；privileged mixed-task Source-SFT：`109/400`。
- v5.2 old具有强视频特异性`132/138/74/82/83`；v6-fast task-complete具有最高absolute
  `143/135/125/128/129`。二者交叉结果证明架构与recipe强耦合。
- SFB checkpoint union达到193但single best只有127，直接暴露能力换手。
- variance reduction让functional evidence更漂亮却没有改善closed-loop，证明surrogate不能选方法。
- K4/few-shot可减少单video偶然性并改善内部same/LOO稳定性，但没有解决full24 shared credit或正确时序。
- 24个task experts统一step2000的train direct-expert=`658/1200`、23/24 tasks非零；它们提供policy-effective
  task-level target，却不提供held泛化、same-task video specificity或时间顺序。
- Expert-Manifold address binding、barycentric和soft/hard bank依次证明：地址辨识、视频路由、expert重建或
  近精确expert复现都不是held performance的充分条件。
- Balanced residual、RLS、Reward-Credit把失效链收窄到跨video correction近正交、offline row保留不等于
  on-policy support、以及q/v continuous tangent低于native BF16 factor ULP。
- 最终rank14反事实又证明：tiny reconstruction error不保证closed-loop support；压缩与regeneration都能
  引发target-heterogeneous rotation。

完整不重复清单和每条路线结果见[`docs/research_history.md`](docs/research_history.md)。

## Data and evaluation

- 起点是generic`lerobot/pi05_base`，不是读过目标LIBERO-40 actions的`pi05_libero`。
- source corpus为与目标40 specification-only去重后的71个LIBERO-90 tasks，每task 50条成功episodes。
- 目标40固定split是24 train / 8 validation / 8 test，封存在`configs/libero_24_8_8_v1/`。
- normalization只来自source corpus并冻结；validation/test actions不产生梯度。
- canonical deployment目前保持exact language + exactly one action-hidden video；未来few-shot必须另立matched
  authority，不能悄悄替换one-shot基线。
- official evaluator严格配对correct/same/wrong/shuffled/reversed/no-video的state、env/policy RNG和video。
- checkpoint只由真实paired closed-loop选择；80-row screen、loss、几何或checkpoint union不能代表真实水平。

## Runtime layout

```text
EMBER/
├── configs/    # frozen split/source/evaluation and retained manifests
├── data/       # datasets and LIBERO assets
├── models/     # tokenizer/model assets
├── runs/       # formal training/evaluation/checkpoints/results
├── evidence/   # retained migration/manifests
├── src/ember/  # reusable source policy, Writer core and evaluator
├── scripts/    # canonical entrypoints
└── tests/
```

进入仓库后使用`.venv`；`.env.local`只提供BCI本地默认路径，关键资产仍由CLI显式传入。OSG-PC已封存
non-pass；SKNC canonical实现与CPU门已完成，只授权从clean pushed head运行预注册discarded live profile，
不过门不训练，不能从旧文档恢复GPU命令。

GPU工作每次同时live检查`gpu01/gpu02`，选一个节点并使用至多6张健康、低利用率、显存余量足够且能提高
吞吐的A40；非零显存或低利用率进程不自动排除，但不得抢占或明显干扰他人。不等待凑满6卡、不dummy占位、
不跨节点拼碎片。
多卡训练遵守`NCCL_P2P_DISABLE=1`、NUMA physical/local rank映射和deferred-NCCL。

吞吐优先：接受正常BF16/TF32/batch低位差异，不为微小roundoff固定batch1、重复forward、扩dtype或逐tensor
校验；不新增SHA-256/MD5或大量内容hash。正式launch仍必须保证信息墙、pairing、finite、OOM、asset、
checkpoint与resume合同正确。

## Documentation

修改或运行前完整阅读[`AGENTS.md`](AGENTS.md)以及其中的最小清单。主要入口：

- [`docs/active_session_handoff.md`](docs/active_session_handoff.md)：唯一当前状态、资产与开放问题；
- [`docs/execution_brief.md`](docs/execution_brief.md)：通用实验、评测、GPU与吞吐合同；
- [`docs/action_forecast_writer_success_key_nullspace_consolidation_design.md`](docs/action_forecast_writer_success_key_nullspace_consolidation_design.md)：
  当前唯一SKNC单变量设计、nullspace公式与证伪门；
- [`docs/action_forecast_writer_on_policy_success_guarded_program_credit_design.md`](docs/action_forecast_writer_on_policy_success_guarded_program_credit_design.md)：
  retired OSG-PC设计、工程non-pass与负结果边界；
- [`docs/action_forecast_writer_policy_innovation_goal_causal_key_design.md`](docs/action_forecast_writer_policy_innovation_goal_causal_key_design.md)：
  PICK-GC设计、formal结果与退役边界；
- [`docs/research_history.md`](docs/research_history.md)：完整精炼实验谱系与禁止重复项；
- [`task_plan.md`](task_plan.md)：当前整理/交接状态；
- [`findings.md`](findings.md)：最重要的第一性原理结论；
- [`docs/concept.md`](docs/concept.md)：长期科学定义。

大量旧design和逐日ledger已从active tree移除；它们仍可由Git commit`3a6f801`精确读取。正式artifacts和
关键结果没有因文档/代码退役而删除。

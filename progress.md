# EMBER Progress

更新时间：2026-08-19。本文只记录当前可执行状态；稳定目标见`docs/current_owner_requirements.md`，耐久结论见
`findings.md`，完整历史见`docs/research_history.md`。

## Current authority and executable state

- 外部专家A--G/F0--F5逐项复核goal已完成；113个编号claim均已实施、反驳或以有证据的
  `not-applicable` / `underdetermined-after-audit`收口，没有queued项。
- 第二轮专家意见后继goal已经由owner正式启动。active design为
  `docs/functional_adaptation_successor_design.md`；持久计划见`task_plan.md`，逐项覆盖见
  `docs/expert_second_round_implementation_ledger_20260819.md`；Phase 0的数据、架构与评测合同已冻结，当前进入
  Phase 1/2的functional realizability基线，尚未启动formal GPU训练。
- canonical workspace与集成目标为`/data1/user/ymdai/projects/EMBER`的`main`；当前结构性开发位于从最新`main`创建的
  `/data1/user/ymdai/projects/EMBER-nonheld-meta-experts`、分支`codex/nonheld-meta-experts`。
  验证后的独立里程碑及时合并`main`并推送，不长期积压巨型分支。
- 来自clean pushed `7b6d768`的train24 fold0 fixed functional decoder formal评测已完成；修正投影wiring后的F4
  free-Program reference仍在gpu02两卡运行，完成前不解释partial rows。
- `main`上的已封存Writer仍是Core-Addressed Reader主架构：Dynamic-K、rank16、38 targets、Action Meta-LoRA、
  layer/rank memory、Reader、K-set、bounded M2P和FactorHeads；原生language保留，Text/VL Meta-LoRA已从
  canonical config/code contract移除。该实现只作为sealed baseline和可复用组件来源，不再作为后继增量路线。
- 不直接返回V6/LPCP/GOMQ，也不恢复旧Expert-Manifold为held dictionary；历史实现只提供paired反事实、functional
  probe、checkpoint/evaluation等可审计复用候选。

## Latest owner decisions for successor planning

- 允许train24 privileged experts训练共享functional decoder；也允许使用LIBERO-90中经审计、排除固定validation/test
  tasks及其重复项的non-held任务，必须保留显式allowlist与provenance；
- 允许learned language-only adapter作为baseline，用于裁决video条件增量；
- 允许在授权train/meta tasks上用simulator reward训练共享Writer/functional code inference。该outer RL仍以held
  zero-interaction LoRA为部署对象，不等于生成LoRA后的task-local RL；
- 允许冻结模型、无梯度、无checkpoint选择的sealed held action/reward诊断；Test默认留到最终方法冻结后；
- 合理的新架构均可考虑，包括rollout前合并为唯一完整LoRA的shared prior/base adapter + video-conditioned residual；
  不允许部署第二adapter、expert route、task-ID字典或checkpoint融合；
- 主写与集成目标改为`main`；需要隔离时从最新`main`创建`codex/<topic>`分支/worktree，验证后及时合并并推送。

## Active successor phase

当前核心顺序为：

1. 审计现有expert manifold、Writer、reward/evaluation与LIBERO数据owner，建立non-held meta allowlist、task-level folds、
   process controls和source/task-expert ceiling协议；
2. 用policy-functional response而非raw A/B几何学习compact code与固定complete-LoRA decoder，并以leave-task-out
   closed loop作为进入门；
3. 固定decoder后学习language prior + action-hidden video process posterior，保留完整Action probe与有向阶段结构；
4. functional warm-start后在train/meta simulator接入closed-loop outer credit；
5. 用strict paired400、相邻checkpoint、same-task不同视频、Long、breadth和多split复现选择或停止方法。

专家方向A--N和五个替代研究问题都已进入ledger。runtime video policy、task-local RL、richer sensing以及
video-to-reward/skill/plan不是被丢弃，而是在核心single-LoRA路线触发预注册stop gate后按证据启动；train/meta action
alignment、mergeable base+residual与sealed diagnostics已经获准进入对应phase。

已完成的Phase 0实现：

- `configs/libero90_nonheld_meta_v1/protocol.json`显式保留71个去重non-held tasks、排除19个target-overlap tasks，并建立
  5个不读取结果的task-level folds；默认56 meta-train / 15 meta-validation，冻结后轮换复现；
- `ember.functional_adaptation.contract`加载allowlist/folds并验证source manifest与语义overlap audit一致；
- strict video conditions已增加first-only、final-only、first+final、endpoints-fixed-middle-shuffled与monotone-sparse，
  真实frames经选择/重排后重新完整forward；
- 新模块owner与旧`expert_manifold`/Writer/evaluator的复用、退役边界已写入active design；旧bank route不恢复。
- `FunctionalCodebook`与`FunctionalAdapterDecoder`已经建立32维whitened task code到全部38-target/76-tensor LoRA的
  单一生成面；decoder以functional identity初始化，Action in/out保持独立，不import旧V6 bank route；
- policy-functional probe会捕获完整`[batch, 50, 32]` Action Expert flow response，并以expert相对identity的响应能量
  归一化监督，避免source policy的大幅公共响应淹没task adapter信号；首轮相关20项CPU测试通过。
- non-held meta expert合同已固定71 tasks中的56 meta-train / 15 meta-validation-oracle，并复用唯一task-expert训练owner；
  fixed decoder也已能从该bank按角色拟合/冻结和导出32维code，不建立task-ID deployment route。
- 后继Writer运行面已实现为`language prior z_L + ordered-video posterior delta(L,V) -> frozen decoder -> one complete LoRA`：
  每条视频独立保序编码initial/goal/event/transition，跨K只聚合完整video program；保留50个Action probe并加入仅训练期
  meta-action phase alignment，同时提供真正不读language/action probe的video-only baseline。模块按decoder、inference、
  schedule/step/checkpoint和privileged-action owner拆分；旧LMMPC继续只作为sealed历史基线，不形成并行active fallback。

当前train24非正式机制profile（不是模型选择或closed-loop证据）：

- 结果无关fold0以19 tasks拟合decoder、5 tasks冻结decoder后只拟合新code；五折将轮换，19/5不是永久丢弃任务；
- gauge-invariant `BA·probe`预热在380/250步把fit mean从`1.000`降到`0.447`、held code mean降到`0.805`，但其
  PI0.5完整flow初始loss仍约`0.999/1.008`，证明effective-update几何不能替代policy-functional目标；
- 完整50-token flow短profile仅给每个fit task 2步、held task 5步，独立demo40--49评测从`0.999→0.833`和
  `1.008→0.933`；18/19 fit与4/5 held优于identity，仍各有一个退化task，因此只支持“链路有可学习信号”，尚不通过
  fixed-decoder realizability gate；
- A40单卡峰值18.81 GB，38+25个实际更新约22秒，主要固定成本是policy加载与成对probe缓存。下一节点应扩大独立panel和
  task-equal更新次数，而不是扫rank、scale、seed或dtype。

当前train24 fold0 formal closed-loop结果：

- fixed decoder单checkpoint为`388/1200`，direct task experts为`658/1200`；严格配对是332 retained、56 gained、
  326 lost，Jaccard `.46499`；
- 19个decoder-fit tasks为`326/950`，对应direct `550/950`；5个decoder-held tasks为`62/250`，对应direct
  `108/250`。fit与held都只保留约六成expert aggregate，不是只在held code拟合处失效；
- 因此train24版明确不通过functional realizability gate，内部flow loss下降不能替代该结论。下一步不是扫小超参，
  而是按已冻结合同训练56/15 non-held meta expert family，再重新拟合和裁决固定decoder。

## Final external-review result

| arm | macro25 | macro50 | 25→50 retained/gained/lost | breadth@1 |
| --- | ---: | ---: | ---: | ---: |
| A Text+detach | 123 | 84 | 71 / 13 / 52 | 8→5 |
| B noText+detach | 104 | — | — | 6 |
| C noText+credit | 110 | 101 | 77 / 24 / 33 | 6→4 |
| F5 C+PCGrad | 107 | 96 | 82 / 14 / 25 | 6→4 |
| F3 A+frozen heads | 123 | 117 | 90 / 27 / 33 | 8→6 |

完整macro25视频面板（correct / same / wrong / shuffle / keep-first / reverse / no-video）：

- A：`123 / 125 / 81 / 122 / 131 / 90 / 48`；
- B：`104 / 101 / 65 / 83 / 90 / 96 / 47`；
- C：`110 / 111 / 54 / 91 / 93 / 69 / 47`；
- F5：`107 / 111 / 51 / 92 / 105 / 53 / 47`。

三个no-Text arm均显著优于no-video和wrong，说明Writer确实使用视频，不是language-only。C是唯一在aggregate上
同时显著优于wrong/shuffle/keep-first/reverse/no-video的arm，但收益高度集中Object、Long reverse反向，且
same-task correct-success retention只有87.27%。因此视频因果资格得到部分改善，方法未达absolute、
稳定、same-video robustness和跨suite高层Program的联合目标。

## Root-cause adjudication

1. **Fresh front-end detach是真实工程缺陷。** A/B在macro1/25的`patch_grounding`/
   `interaction_projection`均无gradient；C修macro1首次有credit。修复将correct-reverse margin从8提到41，
   但correct只104→110且继续漂移，所以它是视频方向资格的一个前端因素，不是absolute/stability首因。
2. **Text Meta-LoRA提供真实但混合的support。** 移除它使correct掉19，同时shuffle/keep-first各掉39/41、
   reverse反而升6；这不是纯language shortcut，也不是科学上干净的正机制。owner的no-Text边界继续有效。
3. **简单self-occupancy divergence未获支持。** lost rows没有出现预期的macro50-self-occupancy disagreement增大；
   validation expert不存在且held teacher action受信息墙禁止，动作正确性只能记为审计后不可判。
4. **FactorHead co-drift是放大器，不是reachability瓶颈。** freeze使84升到117但仍丢33；固定head的free-Program oracle
   为659/1200，direct experts为658/1200，故不扩大head/rank/decoder。
5. **Cross-task conflict会改变换手，standard PCGrad不是解法。** 它将lost 33→25、churn 57→39，但gained
   24→14且有显著抑制，score更低、breadth仍收缩，并把keep-first margin压到2。Adam moment独立效应仍不可由本arm裁决。

当前最早未解接口被收窄为：四条信息流能否生成跨suite、跨初始化的policy-effective learned Program，以及
shared objective/更新能否在同一checkpoint保留这些方向。本轮没有性能pass，也没有登记下一套架构。

## Remote-visible review map

- 原专家报告：`docs/external_review_20260818.md`；
- 113项claim ledger：`docs/external_review_claim_ledger_20260818.md`；
- 本轮面向专家的结果报告：`docs/external_review_followup_20260819.md`；
- 给新session复制的独立复核prompt：`docs/external_review_followup_prompt_20260819.md`；
- 证据索引与全部remote-safe JSON：`docs/evidence/external_review_20260818/README.md`；
- 持久结论与历史：`findings.md`、`docs/research_history.md`。

## Verification and cleanup

- 完整CPU测试：`293 passed`；
- B/C/F5各7个视频面板均为400 rows，pairing mismatch全为0；全部tracked/forced evidence JSON可解析；
- 本轮临时partial JSON、worktrees与试验分支已清理；formal runs、checkpoint、raw rows和唯一evidence均保留；
- 封存状态无EMBER GPU进程；`codex/bci-continuation`仅保留为已集成历史分支，不再作为主写分支。

# EMBER Task Plan

更新时间：2026-08-11。本文只记录当前收尾与下一 session 的决策流程；已完成实验、精确数值和禁止重复项见
`docs/research_history.md`，实时状态见`docs/active_session_handoff.md`。

## Long-term goal

- [ ] 同一 shared method、同一 single checkpoint 的 strict paired correct 严格`>150/400`，随后继续提高。
- [ ] 高分必须同时具有较高 task breadth、低 checkpoint 换手、same-task-other video 鲁棒性，以及 correct
  对 wrong/shuffled/reversed/no-video 的实质优势。
- [x] canonical one-shot 信息墙固定为 exact task language + exactly one action-hidden teacher video；video 是
  唯一 dynamic value；一次生成完整 38-target rank-16 LoRA。
- [x] frozen source policy、source-only normalization、24/8/8 split 和 official paired evaluator 已封存。

## Current decision

- [x] 历史最好 single checkpoint 仍是 v6-fast macro400：`143/400`。
- [x] uniform pivot-rank14 online Gate B=`128/400`，相对 old134 lost21，未过门。
- [x] old-cache compiler-only 反事实=`138/400`，相对 old134 retained/gained/lost=`119/19/15`；虽然净增4，
  但预注册 lost 上限为10，仍未过门。
- [x] compression 与 online regeneration 被分离为两个独立换手源；uniform rank14、Gate C、cycle1、controls
  和训练全部退役。
- [x] 从最新证据选择唯一 successor：PICK只替换Balanced-v2 condition evidence，保留frozen-v6原生full-rank
  base、Program memory、blind full48 update和one-shot部署。
- [x] 新design authority已写入`docs/action_forecast_writer_policy_innovation_consensus_key_design.md`；当前尚无
  EMBER GPU进程，尚未实现、profile、训练或rollout。

## Active PICK phase

- [ ] 按canonical owner原位替换旧Balanced-v2 key，迁移通用policy-innovation encoder而不保留import shim。
- [ ] 完成sealed train24×50 cache统计、zero/order/identity/freeze/resume和fresh-incompatible CPU门。
- [ ] clean commit/push后重新做双节点GPU、进程、quota与峰值preflight；通过raw-frame和full48 mechanism门。
- [ ] profile B8/16/32并封存deployment vertical；不得用低吞吐数值补丁过门。
- [ ] formal fresh0→10后立即strict paired400，按`144/breadth6/lost8`门裁决；首次过144补完整视频controls。

## Repository closeout

- [x] 将项目目标、信息墙、吞吐原则和最新裁决统一到`AGENTS.md`与`README.md`。
- [x] 将当前状态、执行合同和历史实验分别集中到`docs/active_session_handoff.md`、
  `docs/execution_brief.md`和`docs/research_history.md`。
- [x] 把`findings.md`压缩为第一性原理结论，把`docs/concept.md`与
  `docs/novelty_and_landscape.md`改为稳定定义。
- [x] 保留 v4 root-cause、v6、Expert-Manifold history、最终 rank14 decision 和 benchmark validity 五类深证据；
  其余追加式历史设计由 Git commit`3a6f801`保存，不再留在 active tree。
- [x] 删除已退役 rank/compiler 可执行路径和孤立 artifact helper；完整CPU回归`361 passed`。
- [x] 移除已合并clean worktree/branch、明确临时文件和38组非formal profile/smoke checkpoint payload；formal
  evidence与可复用cache保留。
- [x] 核对Markdown links、compileall、CLI、`git diff --check`、工作树与主/远端branch结构；收尾后输出
  new-session prompt。

## Next-session decision procedure

新 session 不应从旧文档中的某条“下一步”直接启动。顺序固定为：

1. 完整阅读`AGENTS.md`规定的最小 authority，确认 workspace、branch、process、storage 与 GPU 都是实时状态。
2. 以 v6-fast 143、latest old134/compiler138/online128 的逐 task 成功集合为共同参照，先选择一个最早失效接口。
3. 写一份单变量、可证伪的 design authority，明确它保留哪些历史有效子机制、改变什么、为什么能同时改善
   absolute 与 retention；不得按 validation task 得失反向挑 topology。
4. 先做最小 CPU/机制闭合；若要 GPU，按 live 空卡和吞吐 profile 选择 batch/world size，不为低位数值一致性
   降低效率。
5. 训练到足以判断趋势后尽快做 strict paired400；报告 aggregate、per-task、breadth、retained/gained/lost、
   churn 和必要的视频 controls。
6. 结果未达标时定位最早失效接口，只做有因果指向的调整；没有架构级证据不得180度转向。

## Open scientific choices, not active plans

- 如何让不同 video 条件产生的有用更新在一个 checkpoint 内共同积累，而非近正交换手。
- 如何在不依赖 held expert dictionary 的情况下，把 policy-effective task target 与视频时序证据绑定。
- one-shot 下如何使正确顺序成为必要信息；few-shot 是否能提取跨 demo 不变量，以及怎样避免它退化为平均或
  增加部署计算的无效补丁。
- 是否存在由 train24 policy geometry 预先推出的 heterogeneous target/rank topology；uniform rank14 已被否决，
  但这不授权 pivot15+1、mixed rank 或其它未经推导的变体。
- 怎样构造比历史 source-action functional surrogate 更贴近 held on-policy support 的 credit，同时保持吞吐。

## Non-negotiable efficiency boundary

- 不为普通 BF16/TF32、batch shape、kernel 或 reduction order 的低位差异固定 batch1、重复 forward、扩 dtype、
  加逐 tensor 内容扫描或 SHA-256/MD5。
- GPU launch 时同时 live 检查`gpu01/gpu02`，选择一个节点，使用该节点所有真正空闲、健康且能提高吞吐的
  A40；没有固定6卡上限，不等待凑卡、不跨节点拼碎片、不干扰他人。
- independent rollout 使用动态队列；多卡训练遵守`NCCL_P2P_DISABLE=1`、NUMA physical/local rank 映射和
  deferred-NCCL。exact-resume 仍锁原 topology。

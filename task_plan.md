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
- [x] PICK canonical implementation、`345 passed` CPU门和raw-frame probe通过；world6 discarded full48只因
  condition=`483.61515>200` non-pass，其余14项机制、动作与吞吐门通过，未获formal训练资格。
- [x] 从该最早失效接口选择唯一successor PICK-GC：只把static block换为goal residual并保留causal prefix；
  新authority是`docs/action_forecast_writer_policy_innovation_goal_causal_key_design.md`。
- [x] PICK-GC formal macro10 strict=`138/400`、breadth6，相对macro0 retained/gained/lost=`118/20/16`；未过
  `correct>=144`与`lost<=8`门，PICK-GC+blind offline credit退役。
- [x] effective-BA不是identity或能量塌缩：norm中位比`1.000016`、相对L2`.002397`；最早失败接口推进到
  blind offline cotangent→held on-policy useful support/coexistence。

## Completed PICK-GC phase

- [x] train24×50 cache门：same mean/median`.90260/.91604`、cross`.13455/.11375`、reverse
  `-.80305/-.80877`、shuffle约0、correct24 condition max`21.62`；GE漏中间shuffle、SDE same`.86177`，均拒绝。
- [x] 原位替换PICK descriptor/config/tests，不保留并行strategy；完整CPU回归`345 passed`。
- [x] implementation commit`717b561`已push并建立clean detached frozen worktree。
- [x] exact raw full48 rank48/condition`152.45803`与discarded world6 mechanism condition`152.61008`均过门；
  retained/null=`24/24`、Program/LoRA/action/throughput闭合。
- [x] longest-video B8/16/32均stable，按`.47119/.47244/.47299` LoRAs/s选B32；zero-memory四suite
  LoRA/action bit-exact，canonical cache与8/8 rollout通过。
- [x] world6/deployment formal-ready evidence由`5200bee`封存并push，完整回归`346 passed`；对应clean detached
  frozen worktree已建立。
- [x] live无单节点world6后只改执行拓扑为world4/local6；profile-only authority由本commit封存，push后从新
  frozen worktree重过discarded mechanism与归一吞吐门，失败不训练。
- [x] `09bbed3` world4 profile相对world6机制payload逐字段exact，14 checks全true；step`34.94275s`、归一
  ratio`.89871<1.25`、exit0、无checkpoint，formal-ready reseal由本commit封存。
- [x] world4 formal fresh0→10完整结束并保存macro10 checkpoint/cursor/RNG/metrics；每步rank48且Program memory
  非零积累。
- [x] strict paired400完成48/48 shards、400/400 rows、12/12 workers；结果138/breadth6/lost16，正式关闭
  resume10→25、six-arm controls与参数sweep。
- [x] formal result、paired transition与effective-BA诊断写入config、handoff、design和retained decision evidence。

## Active successor synthesis

- [x] 从第一性原理选择一个只改变credit/occupancy接口的successor；不得恢复Reward/RLS/rank14 executable，
  也不得并行实现多个候选。
- [x] OSG-PC明确保留PICK-GC ordered goal-causal key、condition-local FP32 Program与native full-rank16 compiler，
  以每条成功train24 rollout的executed-prefix half-space解析保护已有support；authority见
  `docs/action_forecast_writer_on_policy_success_guarded_program_credit_design.md`；并解释train24 on-policy
  reward/occupancy如何形成连续cotangent、保护task-complete已有support。
- [x] 预注册CPU投影、discarded full24 K4+B20、formal macro5与strict paired400门；实现前不launch GPU。
- [ ] 原位替换唯一canonical runtime/config/tests，不保留PICK-GC或旧Reward可执行分支；先过CPU和architecture gate。

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

## Retained PICK live launch contract and result: 2026-08-11

- frozen workspace：`/data1/user/ymdai/worktrees/EMBER-pick-f4a61a8-20260811`，detached自包含本段的clean
  pushed authority；`PYTHONPATH`显式指向该worktree的`src`，只把ignored `runs`链接到canonical artifact root。
- assets：source run `runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722`及其step1000 checkpoint；tokenizer
  `models/tokenizers/openpi/paligemma_tokenizer.model`；target data
  `data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a`；simulation path由canonical `.env.local`提供。
- live allocation：`gpu02` physical GPU`0,1,2,3,4,5`，world6、每rank4 tasks、每rank2 loader workers；GPU6–7
  属于他人。launch前再次要求0–5 memory/utilization/process/ECC空闲健康。
- raw-key gate：先在`gpu02:0`运行retained probe
  `runs/outputs/pi05_pick_policy_innovation_raw_key_probe_f4a61a8_20260811/raw_key_probe.py`；必须重现zero、same/cross、
  reverse/shuffle、wrong-target-language，以及hidden重排与raw-frame重跑等价，失败即不启动full48。
- discarded full48 profile exact command由
  `runs/logs/pi05_pick_policy_innovation_full48_profile_macro0_r6_b20_f4a61a8_20260811.launch.sh`执行：

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=6 scripts/train_v6_prior_writer.py \
  --config configs/pi05_v6_policy_innovation_consensus_key_v1.json \
  --mode mechanism-profile \
  --source-run /data1/user/ymdai/projects/EMBER/runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722 \
  --checkpoint /data1/user/ymdai/projects/EMBER/runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000 \
  --tokenizer-path /data1/user/ymdai/projects/EMBER/models/tokenizers/openpi/paligemma_tokenizer.model \
  --data-root /data1/user/ymdai/projects/EMBER/data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a \
  --output-dir /data1/user/ymdai/projects/EMBER/runs/outputs/pi05_pick_policy_innovation_full48_profile_macro0_r6_b20_f4a61a8_20260811 \
  --stop-after-macro 1 --num-workers 2
```

  environment固定`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5`、`NCCL_P2P_DISABLE=1`、`NCCL_ALGO=Ring`、
  `NCCL_PROTO=Simple`、native BF16/TF32和internal GPU-local NUMA binding/deferred NCCL。
- scale/gates：24 train tasks×one action-hidden video×B20 queries，full48 rank必须48、condition≤200、retained≥21、
  negative null≥18且每类≥6、aggregate leakage≤.15、Program/BA/action闭合、0 OOM/nonfinite/negative action forward，
  production wall≤sealed world6 baseline`21.0951s`的`1.75x`。profile不保留memory checkpoint。
- storage/resume：`strg01 /data1`当前`508230484/1073741824 KiB`，raw/profile/log/temp峰值新增估计<2GiB；两个
  output roots必须fresh，均不resume/overwrite。失败root作为non-pass evidence保留；只有raw、full48与后续
  B8/16/32 deployment vertical全部通过才允许创建formal fresh0→10 root。
- result：authority-matched raw probe通过；full48 profile rank48、correct retained24/24、negative null24/24、
  leakage`.03815`、Program/LoRA/action/throughput全部闭合，但condition=`483.61515`，故`passed=false`且没有
  deployment profile、formal训练或rollout。

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

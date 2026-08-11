# EMBER Task Plan

更新时间：2026-08-12。本文只记录当前收尾与下一 session 的决策流程；已完成实验、精确数值和禁止重复项见
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
- [x] SKNC formal macro5 strict=`137/400`、breadth7，相对old134 retained/gained/lost=`121/16/13`、churn29；
  Program/constraint/energy门通过但`correct>=140`、`lost<=8`和单task增益集中门失败，SKNC+blind B20退役。
- [x] SKNC终局后先关闭active successor；随后只从train24 success-key neighborhood不能代表held video/
  initialization occupancy，以及blind B20与真实reward不对齐的最早接口选择SRTP。

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

## Completed SKNC successor

- [x] 从第一性原理选择一个只改变credit/occupancy接口的successor；不得恢复Reward/RLS/rank14 executable，
  也不得并行实现多个候选。
- [x] OSG-PC明确保留PICK-GC ordered goal-causal key、condition-local FP32 Program与native full-rank16 compiler，
  以每条成功train24 rollout的executed-prefix half-space解析保护已有support；authority见
  `docs/action_forecast_writer_on_policy_success_guarded_program_credit_design.md`；并解释train24 on-policy
  reward/occupancy如何形成连续cotangent、保护task-complete已有support。
- [x] 预注册CPU投影、discarded full24 K4+B20、formal macro5与strict paired400门；实现前不launch GPU。
- [x] 原位替换唯一canonical runtime/config/tests，不保留PICK-GC或旧Reward可执行分支；success-only replay、
  per-success retention VJP、解析KKT投影、full48实际guard诊断和fresh schema均闭合。
- [x] 加载`.env.local`后的fresh完整CPU回归`340 passed`，compileall与diff-check通过；architecture gate确认只有
  一个trainer/Writer/full48 solver，没有parallel family或compatibility fallback。
- [x] 22:22--22:23+08:00双节点、进程、深度GPU健康、quota与fresh-root preflight完成；选择`gpu02:0--5`
  单节点world6/local4，`:6/:7`属于他人不触碰，GPU1历史已纠正ECC/remap须发射前后监测；profile峰值`<2GiB`。
- [x] `9263851` world6 authority从clean detached worktree完成唯一discarded profile；rank-local长尾使sequence12
  all-reduce等待600s后watchdog，wall lower bound=`1.912x>1.25x`，exit1且无report/checkpoint，GPU已健康释放。
- [x] OSG-PC按hard gate封存为`profile_result_sealed_nonpass`；同配置重跑、formal、deployment、评测和参数sweep
  全部关闭。负结果只淘汰current full-replay per-success VJP执行图。
- [x] 从第一性原理选择下一单变量successor SKNC：保留PICK-GC视频key、FP32 Program、B20 objective与native
  compiler；K4只保留binary outcome，4/4 current/persisted success keys直接约束最终shared memory write的
  nullspace，不再随成功轨迹长度/episode数做replay VJP。
- [x] 完成`docs/action_forecast_writer_success_key_nullspace_consolidation_design.md`：明确first-only per-task
  anchor bank、解析投影公式、非scalar-gate边界、CPU/live/formal门与macro5立即strict paired400。
- [x] 原位替换OSG-PC executable path并泛化唯一full48 solver；建立fresh-incompatible config/checkpoint，完成
  synthetic/bank/resume/outcome-only/Program→LoRA→action CPU门；加载`.env.local`的完整回归`334 passed`，
  compileall与diff-check通过，无active replay/VJP或并行solver残留。
- [x] 从clean pushed implementation seal完成双节点/quota/health/NUMA preflight，按显存峰值选择
  `gpu02:3,4,5` world3；首个full24 K4+B20 profile通过15/16 checks，step=`487.002s`、scaled ratio=
  `.47999`，唯一non-pass是TF32 equality diagnostic的`1.1228e-4>1e-5`。
- [x] 同真实success-key basis的CPU/GPU FP32 ratio约`7.2e-8/7.10e-8`，TF32为`8.44e-5`，且live
  protected LoRA/BA/action exact zero；定位为measurement contract violation，不追认首root pass。
- [x] 只把既有hard-equality diagnostics切到FP32 GEMM，production TF32/BF16、method/gate/forward数不变；
  clean `f4fdac7` fresh reprofile的16项checks全部通过：Program ratio=`8.95e-8`、step=`478.627s`、scaled
  wall ratio=`.47173`，未保留checkpoint。
- [x] 完成实际evaluation adapter的B8/16/32 Writer deployment profile；全部stable、无OOM/nonfinite或hidden
  teacher read，选择B32=`.47166 LoRA/s`；该formal-ready阶段随后由正式结果终结。
- [x] 从clean pushed `e3863cb` detached worktree在`gpu02:3,4,5` world3完成formal fresh`0→5`；5个macro
  functional loss无趋势，bank增长到15 tasks，macro5 rank36、projected energy`.592`、Program closure通过。
- [x] macro5 strict paired400以9个persistent workers完成400/400 rows、exit0；结果137/breadth7，old134→SKNC
  `121 retained/16 gained/13 lost`，Long净`+7`但Object/Spatial净`-5/-1`，按预注册门停止。
- [x] exact-resume`5→10`、six-arm controls、threshold/rank/scale/dtype/seed sweep均关闭；formal checkpoint、raw
  rows、paired transition和decision evidence保留。

## Next successor authority

- [x] 从第一性原理选择一个主要变量，解释怎样用视频中的有向高层过程形成跨video/初始化可共享的policy credit，
  同时阻断language/static bypass并让多task能力在同一checkpoint共存。
- [x] 选择SRTP：保留SKNC `D0`/anchors/PICK key/native compiler；mixed K4每episode最多4个constant-memory
  occupancy landmarks产生LOO reward tangent，所有half-spaces直接投影task汇合后的最终shared Program update。
- [x] 完成`docs/action_forecast_writer_shared_reward_tangent_projection_design.md`，明确direct Reward sub-ULP、OSG
  full-prefix长尾与task-local guard失真边界，以及解析small-dual QP、CPU/live/formal门。
- [x] 原位实现fresh-incompatible SRTP：每episode first/last+reservoir2、mixed-only Nmc4 B16 Program tangent、
  final shared NNLS projection与fresh checkpoint/config schema均进入唯一canonical path；初始完整CPU回归
  `358 passed`，graph-lifetime修复后为`359 passed`。
- [x] `d172add` clean commit/push、frozen worktree和双节点/quota/NUMA preflight完成；按实时余量选择gpu02物理
  3/4/5做首个world3 macro，但三rank在mixed reward CFM处因decoder graph跨K4保留同时OOM，未写mechanism
  report/checkpoint，退出后设备正常释放。
- [x] 原位改成blind VJP立即释放graph、mixed Nmc4后compiler-only重解一次，scientific合同不变且完整CPU
  `359 passed`；`e31e2fd` clean pushed后的唯一同合同reprofile仍三rank在mixed CFM处OOM，未写mechanism
  report/checkpoint。SRTP按门终局退役，不降batch/dtype、不开allocator、不做第三次修补。
- [x] SRTP deployment、formal fresh`0→5`与strict paired400因profile non-pass全部关闭；config封存为
  `profile_result_sealed_nonpass`，两个failure roots/logs保留。
- [x] 从完整logical landmark policy-gradient本体显存与历史credit错位出发选择PCUG：严格配对actual base/
  candidate closed-loop outcomes，只对当前harmful/stable tasks做final closest zero-motion projection；每task总
  rollout仍4条且无policy backward。design authority已写。
- [x] 原位替换SRTP runtime为fresh-incompatible PCUG两阶段macro，完成exact candidate Program、paired K2×2、
  ephemeral harm guard、persistent stable bank、closest projection、checkpoint隔离与完整CPU/architecture gate；
  SRTP active path已删除，完整CPU回归`344 passed`。
- [x] clean commit/push并建立detached frozen worktree；按live双节点状态使用4张合适A40运行唯一discarded
  fresh macro0 profile。live hard gate未过，故B8/16/32 deployment smoke和formal fresh`0→5`均关闭。
- [x] PCUG以clean pushed`238cab4`、gpu02物理3/4/5/6 world4启动；Phase A在full24 gather前达到
  `809.72185s / 2.25568x>1.5x`wall下界。按不可逆hard gate停止自己的torchrun；无OOM、paired report或checkpoint，
  GPU正常释放，failure artifact与config non-pass状态已封存。
- [x] 从Phase-A rank-local tail封存Work-Queue PCUG单变量authority：保留未被检验的actual candidate pairing，
  只把static rank ownership改为task-keyed runtime claim queue；每rank最多8个retained graphs，full24仍按
  task ordinal等权汇聚。CPU/live/formal快速否决门已预注册。
- [x] 原位实现task-addressable exact B20、24-job atomic queue、variable-count padded gather和24条task timing；
  fresh schema/config/eval family已替换旧PCUG，完整CPU`345 passed`。没有worker pool、preload、hash或fallback。
- [x] clean pushed`7c86bf8` detached worktree在gpu02物理3/4/5 world3运行首次discarded macro；Phase-A本体
  `72.9700s / 0.15246x SKNC`、24 tasks完整且每rank8，但host-local cursor误落共享`/data1`，两次`flock`
  各等待约`30.4s`，累计claim=`60.8736s>1s`，因此paired前工程non-pass且无checkpoint。
- [ ] 只修cursor到节点本地`/tmp`，clean push后按同world3、同架构/objective/gates做唯一reprofile；若仍不过门则
  按实际最早接口裁决，不改queue order、cap、world size、threshold或科学合同。

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
- GPU launch 时同时 live 检查`gpu01/gpu02`，选择一个节点，使用至多6张健康、低利用率、显存余量足够且能
  提高吞吐的A40；非零显存或低利用率进程不自动排除。不等待凑满6卡、不跨节点拼碎片，不抢占或明显干扰他人。
- independent rollout 使用动态队列；多卡训练遵守`NCCL_P2P_DISABLE=1`、NUMA physical/local rank 映射和
  deferred-NCCL。exact-resume 仍锁原 topology。

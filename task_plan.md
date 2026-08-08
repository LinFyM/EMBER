# EMBER Task Plan

更新时间：2026-08-09。本文只保留当前可执行计划；旧架构的命令、分支和逐次流水由Git、
`progress.md`及formal artifacts保存，不在活动计划中保留可误执行副本。

## Goal与不可变边界

- [ ] 同一shared method/single asset bundle在strict paired correct上严格超过`150/400`，并继续提高
  absolute、8-task breadth、稳定性和可重复性。
- [ ] correct必须在严格配对下实质优于wrong、shuffled、reversed与no-video；same-task-other保持鲁棒。
- [x] 保持one-shot：部署输入只有exact task language和恰好一条action-hidden teacher video；视频是
  唯一dynamic value，不增加language-only LoRA bypass、多video/LoRA平均或checkpoint融合。
- [x] validation/test actions不进入任何训练或选择梯度；task experts只使用train24 actions。
- [x] GPU工作前实时比较`gpu01/gpu02`，只使用空闲A40、合计最多6张，不干扰他人；多卡保持
  `NCCL_P2P_DISABLE=1`、NUMA、physical/local rank和deferred-NCCL合同。

## 已封存的关键证据

- [x] 统一step2000 task-expert bank完成：
  `runs/outputs/pi05_task_expert_bank_formal_step1000_r6_81101fe_20260807`。24/24 tasks、
  step250/500/1000/1500/2000共120个checkpoints；development-train direct-expert闭环=
  `432/557/624/638/658` of 1200，正式统一选择step2000，不按task混点。
- [x] train24×50 action-hidden phase16×3072 feature cache完成：
  `runs/outputs/pi05_expert_manifold_feature_cache_train24x50_r6_222d3ac_20260808`。
- [x] learned address-binding Writer macro50 strict correct=`75/400`、breadth=`4/8`；完整400-LoRA
  cross-task/task-mean cosine=`.94197/.94270`、nearest expert=`.12734`。正式拒绝resume100和五臂。
- [x] Causal Barycentric CPU leave-one-task-out门完成：24 folds、每折23 experts、7,200套完整LoRA，
  ridge `.3`的topological correct/reversed/phase-shuffled effective-target cosine=
  `.38302/.09900/.18539`；correct norm/stable-rank/top-energy=`3.84385/1.15056/.89540`。
  artifact：
  `runs/outputs/pi05_expert_manifold_causal_barycentric_loo_step2000_cpu_20260809/analysis.json`。
- [x] Causal Barycentric strict correct400完成并负裁决：`63/400`、breadth=`5/8`，72/72 jobs、
  400 unique rows/LoRAs、18 workers attempt1/exit0、0 retry/error/OOM/nonfinite/forbidden reads。
  strict same-video source/addressless对照gained/lost=`46/31`、exact `p=.1100`；未达到可信absolute门，
  不做其余五臂。root=
  `runs/outputs/pi05_expert_manifold_causal_barycentric_correct400_noreplacement_seed7_0397be6_20260809`。
- [x] full400几何排除低能量/低秩：norm/stable/top=`3.958/1.155/.894`、16 coordinates active；
  同时定位raw factor compiler风险：coefficients abs support中位`13.75`，same/cross/task-mean cosine=
  `.988/.685/.697`。分别混合A/B会引入`k!=j` cross-expert effective-update项。
- [x] Policy-Effective strict correct80完成并负裁决：`15/80`、breadth=`5/8`，36/36 jobs、80 unique
  LoRAs、9 workers attempt1/exit0、0 retry/error/OOM/nonfinite/forbidden reads。相对same-video raw
  gained/lost=`6/3`，未过`22`分消歧门；禁止扩跑160/400和五臂。root同本文件下方launch合同。
- [x] exact effective`BA`分析覆盖3,160 generated pairs、1,920 generated-expert pairs和80 matched raw
  pairs：norm/stable/top=`4.148/1.234/.847`，current/raw cosine=`.958`、norm ratio=`1.055`；
  same/cross/task-mean=`.989/.703/.712`、nearest expert=`.641`。compiler修复真实但不是主导瓶颈。

## 当前唯一canonical实现

- [x] 唯一config现为
  `configs/pi05_video_expert_manifold_hard_routed_policy_effective_v2.json`；formal状态现为
  `blocked_until_live_a40_online_smoke`。旧soft config已删除且只由Git/artifact保留；实现提交=`1619631`
  已push，尚无本hard-route专属GPU smoke或closed-loop成绩。
- [x] 一条视频先形成`mean_phase(phase_centered_causal_memory(video_innovation))`；train24×50
  centroids单位化后，用ridge `.3` centered-kernel affine solve产生24个scores；部署固定取signed argmax
  one-hot，support1，soft scores只作审计。
- [x] 每个38 policy targets分别在effective`BA`空间混合24个expert的unit-Frobenius direction与
  envelope内log norm；shared rank96 left/right basis内做best-rank16 SVD。template-A Procrustes gauge
  只固定因子表示，不改变policy update。
- [x] zero/phase-constant表示逐tensor精确返回template-A/zero-B identity；Writer无Parameter，language
  没有独立value通路。真实24-expert one-hot effective cosine中位/最小=`.99838/.99665`。
- [x] evaluator只接受新config + 统一expert bank + feature cache + video data，不再接受learned Writer
  checkpoint。adapter/episode schema已升为hard-routed v4，旧trainer/checkpoint/model/compiler
  executable均不保留。
- [x] hard-route全仓`182/182` CPU tests、compile及真实fixed-asset门通过。24/24 centroids与1,200/1,200
  videos self-route；ordered/reversed与fixed-shuffle选择改变=`1200/1200`、`699/1200`；24 experts全覆盖。
  zero exact，912个one-hot target effective cosine中位/最小=`.998982/.961962`。

## 下一证据门

1. [x] 当前实现已clean commit/push为`1d9d030`；后续工作从该提交或其纯authority后继提交执行。
2. [x] 按live GPU与quota preflight选择空闲`gpu02:0`完成validation8×1-state online smoke：
   feature→coefficients→full LoRA cache→release Writer/encoder→复用同一source policy rollout。
   实得8 rows/8 generated/8 cache entries、3 workers exit0、0 retry/failure/OOM/nonfinite/forbidden
   reads；`1/8` success只作execution smoke。root=
   `runs/outputs/pi05_expert_manifold_causal_barycentric_online_smoke_gpu02_3c8ce25_20260809`。
3. [x] 精确smoke evidence已写回config并seal；真实formal inspector与全仓180 tests通过。authority
   commit/push完成后只从其clean frozen worktree继续。
4. [x] strict paired correct400与400-LoRA审计完成；absolute失败，已停止五臂并定位到raw-factor
   reconstruction的policy-effective语义不守恒。
5. [x] CPU-only effective-BA门通过：pure affine norm ratio仅`.527`而拒绝；per-target effective
   direction+log-norm为`.986`。shared rank96 + public rank16对400 queries的cosine中位/最小=
   `.99682/.99532`，24 experts captured-energy中位/最小=`.99677/.99331`；8个full-span样本的
   captured-energy中位`.99523`。artifact=`policy_effective_compiler_feasibility_full400_rank128_v2.json`。
6. [x] 已在唯一runtime原位替换compiler；保持video reader、coefficients、expert/cache、38 targets、
   rank16、one-shot与zero identity不变，CPU合同与真实资产检查全部通过。
7. [x] 专属单卡A40 online generation/cache/release/rollout smoke已通过并seal；下一步闭环固定为
   validation8×前10 states=`80`条correct screen。历史相同前缀为source/addressless/address-binding/
   raw-barycentric/v6-fast=`9/10/13/12/28`，breadth=`2/3/2/3/5`。强通过门为score`>=28/80`、
   breadth`>=5`且相对raw-barycentric paired `gained-lost>=10`；随后才做formal correct400。score
   `22--27`且breadth`>=4`只扩到预注册160-row消歧；更低或breadth`<=3`则拒绝当前compiler并回到
   最早失效接口。实际=`15/80`、breadth5、raw paired净`+3`，低于ambiguity门；本candidate停止。
8. [ ] 若one-shot的same-task视频方差成为经证据定位的最早限制，再评估固定K的few-shot set/sequence
   aggregation；不能把few-shot当作掩盖task identity或视频时序失败的捷径。
9. [x] video-routed hard-one-hot expert已在唯一runtime实现并通过真实资产CPU门；artifact=
   `runs/outputs/pi05_expert_manifold_hard_routed_cpu_real_assets_20260809/analysis.json`。不把train self-route
   或旧correct80 implied routes当成validation闭环结果。
10. [ ] 从当前clean pushed authority做live GPU/quota preflight，只选一张空闲A40，完成validation8×1
    state correct/without-replacement online generation→cache→release→source-policy reuse smoke；成功后
    才写回evidence并seal。
11. [ ] seal后从新frozen worktree只跑与旧candidate完全相同的validation8×states0--9 correct80 panel。
    若未实质超过`15/80`并保持/提高breadth，停止expert-mixture内调参，转向v6先验的可迁移Writer。
   若不能实质超过15并提高breadth，则24-expert support不足，停止继续修soft compiler；若显著提高，
   才研究可微稀疏routing和order-negative credit。任何GPU前重新live preflight并另行封存launch合同。

## Policy-Effective online smoke launch合同（2026-08-09）

- scientific code seal=`fb5b367`；有效retry只允许来自包含本段launch record、clean pushed、以
  `origin/codex/bci-continuation`为upstream的冻结分支`codex/policy-effective-smoke-20260809`，worktree=
  `/data1/user/ymdai/worktrees/EMBER-policy-effective-smoke-20260809`。fresh retry1 root/log/tmux固定为
  `runs/outputs/pi05_expert_manifold_policy_effective_online_smoke_retry1_gpu02_fb5b367_20260809`、
  `runs/logs/pi05_expert_manifold_policy_effective_online_smoke_retry1_gpu02_fb5b367_20260809.log`和
  `ember_policy_effective_smoke_r1_fb5b367`；登记时三者均不存在。
- 06:36 CST attempt0在GPU加载前被`git_state()`拒绝：detached worktree没有`@{upstream}`。物理0始终
  0 MiB、0%，未创建run contract/cache/worker或scientific row；只留下含`libero_config`的35-byte
  partial root和2,006-byte错误日志，二者保留为工程失败证据，不复用、不计重试或性能结果。
- 06:33 CST live比较：只选`gpu02`物理0（UUID `GPU-2f8ac922-...`）这一张0 MiB、0% A40，固定
  physical0→local0、NUMA0；`gpu02:6/7`及`gpu01:3`的他人进程明确不触碰。gpu02 available host memory=
  `480GiB`。启动前必须再检查物理0仍为空闲，否则不启动。
- `/data1`个人quota现场blocks=`563,383,492/1,073,741,824 KiB`；8套FP32 rank16 LoRA约41.3MB，连同
  queue/results/log保守峰值低于1GiB，远低于剩余预算。只跑validation8 tasks×1 state的correct/
  without-replacement工程smoke；成功或失败都不是性能证据，不扩展到训练或长期run。
- exact command（在上述frozen worktree、`gpu02`执行）：

```bash
env PYTHONPATH=$PWD/src CUDA_DEVICE_ORDER=PCI_BUS_ID NCCL_P2P_DISABLE=1 TOKENIZERS_PARALLELISM=false EMBER_STORAGE_ROOT=/data1/user/ymdai EMBER_STORAGE_CAP_BYTES=1099511627776 EMBER_LIBERO_ASSETS_ROOT=/data1/user/ymdai/projects/EMBER/data/simulation/ember_assets/datasets/libero-assets/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6 numactl --cpunodebind=0 --membind=0 /data1/user/ymdai/projects/EMBER/.venv/bin/python scripts/evaluate_pi05.py run --config configs/pi05_target_evaluation_v1.json --source-run /data1/user/ymdai/projects/EMBER/runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722 --checkpoint /data1/user/ymdai/projects/EMBER/runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000 --tokenizer-path /data1/user/ymdai/projects/EMBER/models/tokenizers/openpi/paligemma_tokenizer.model --output-dir /data1/user/ymdai/projects/EMBER/runs/outputs/pi05_expert_manifold_policy_effective_online_smoke_retry1_gpu02_fb5b367_20260809 --role validation --mode smoke --state-count 1 --replicas-per-gpu 3 --writer-generators-per-gpu 1 --writer-generation-batch-size 4 --gpu-indices 0 --expert-manifold-config configs/pi05_video_expert_manifold_policy_effective_barycentric_v1.json --expert-manifold-expert-bank-root /data1/user/ymdai/projects/EMBER/runs/outputs/pi05_task_expert_bank_formal_step1000_r6_81101fe_20260807 --expert-manifold-feature-cache-root /data1/user/ymdai/projects/EMBER/runs/outputs/pi05_expert_manifold_feature_cache_train24x50_r6_222d3ac_20260808 --expert-manifold-video-data-root /data1/user/ymdai/projects/EMBER/data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a --expert-manifold-video-condition correct --expert-manifold-video-sampling without_replacement
```

- 验收门：8 unique rows、8 unique generated/cache entries、3 workers自然exit0、0 retry/failure/OOM/
  nonfinite；teacher action/state/reward/terminal reads全0；Writer modules在cache后释放且source policy原进程
  复用于rollout。**已全部通过**：8/8 rows/LoRAs/cache、3/3 exit0、0异常/forbidden reads，generation=
  `11.070s`、peak allocated=`10,576,896,000` bytes；GPU已释放，专属evidence已写回并seal。

## Policy-Effective strict correct80 screen launch合同（2026-08-09）

- scientific code seal=`ffed252`。实际run只允许来自包含本段launch record、clean pushed、以
  `origin/codex/bci-continuation`为upstream的冻结分支`codex/policy-effective-screen80-20260809`，worktree=
  `/data1/user/ymdai/worktrees/EMBER-policy-effective-screen80-20260809`。fresh root/log/tmux固定为
  `runs/outputs/pi05_expert_manifold_policy_effective_correct80_screen_noreplacement_seed7_ffed252_20260809`、
  `runs/logs/pi05_expert_manifold_policy_effective_correct80_screen_noreplacement_seed7_ffed252_20260809.log`和
  `ember_policy_effective_screen80_ffed252`；登记时branch/worktree/root/log/tmux均不存在。
- 06:51 CST live比较双节点后只选`gpu02:0,1,2`三张0 MiB、0% A40，严格physical0/1/2→local0/1/2、
  同属NUMA0；`gpu02:6/7`及`gpu01:3`他人进程明确不触碰。gpu02 available host memory=`480GiB`，
  三卡×3 replicas的历史峰值低于120GiB host RAM。启动前必须再次live复核三卡，任一卡变忙即不启动。
- `/data1`个人quota blocks=`563,425,168/1,073,741,824 KiB`。80套FP32 LoRA约`411,893,760` bytes，
  连同results/queue/log保守新增低于1GiB。panel固定8 validation tasks各state0--9、correct one-shot、seed7、
  每task 10条teacher video无放回；与addressless/address-binding/raw-barycentric逐row严格同state/video/
  env/policy RNG配对，source只同state/env/policy RNG，v6-fast因历史video schedule不同只作same-state参考。
- exact command（从上述frozen worktree在`gpu02`执行）：

```bash
env PYTHONPATH=$PWD/src CUDA_DEVICE_ORDER=PCI_BUS_ID NCCL_P2P_DISABLE=1 TOKENIZERS_PARALLELISM=false EMBER_STORAGE_ROOT=/data1/user/ymdai EMBER_STORAGE_CAP_BYTES=1099511627776 EMBER_LIBERO_ASSETS_ROOT=/data1/user/ymdai/projects/EMBER/data/simulation/ember_assets/datasets/libero-assets/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6 numactl --cpunodebind=0 --membind=0 /data1/user/ymdai/projects/EMBER/.venv/bin/python scripts/evaluate_pi05.py run --config configs/pi05_target_evaluation_v1.json --source-run /data1/user/ymdai/projects/EMBER/runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722 --checkpoint /data1/user/ymdai/projects/EMBER/runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000 --tokenizer-path /data1/user/ymdai/projects/EMBER/models/tokenizers/openpi/paligemma_tokenizer.model --output-dir /data1/user/ymdai/projects/EMBER/runs/outputs/pi05_expert_manifold_policy_effective_correct80_screen_noreplacement_seed7_ffed252_20260809 --role validation --mode screen --state-count 10 --replicas-per-gpu 3 --writer-generators-per-gpu 1 --writer-generation-batch-size 4 --gpu-indices 0,1,2 --expert-manifold-config configs/pi05_video_expert_manifold_policy_effective_barycentric_v1.json --expert-manifold-expert-bank-root /data1/user/ymdai/projects/EMBER/runs/outputs/pi05_task_expert_bank_formal_step1000_r6_81101fe_20260807 --expert-manifold-feature-cache-root /data1/user/ymdai/projects/EMBER/runs/outputs/pi05_expert_manifold_feature_cache_train24x50_r6_222d3ac_20260808 --expert-manifold-video-data-root /data1/user/ymdai/projects/EMBER/data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a --expert-manifold-video-condition correct --expert-manifold-video-sampling without_replacement
```

- 工程验收为80 unique rows/LoRAs/cache entries、所有builder生成的jobs与9 workers自然exit0、0 retry/
  failure/OOM/nonfinite、四类forbidden reads全0、每行恰好一条correct frames-used video、Writer release和
  source-policy reuse闭合。科研门严格使用本节预注册的score/breadth/paired gained-lost，不因结果改阈值。

## 退役边界

`scripts/train_expert_manifold_writer.py`、learned Writer training/checkpoint模块及旧
`--expert-manifold-checkpoint`参数均已退役。当前动态Writer只有Policy-Effective Barycentric路径；任何历史
learned-Writer命令都只作provenance，不得从本文或旧记录复制恢复为并行实现。

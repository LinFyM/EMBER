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

## 当前唯一canonical实现

- [x] 唯一config现为
  `configs/pi05_video_expert_manifold_policy_effective_barycentric_v1.json`；formal状态故意保持
  `blocked_until_live_a40_online_smoke`，旧raw-factor config已删除且只由Git/artifact保留。
- [x] 一条视频先形成`mean_phase(phase_centered_causal_memory(video_innovation))`；train24×50
  centroids单位化后，用ridge `.3` centered-kernel affine solve产生24个coefficients。
- [x] 每个38 policy targets分别在effective`BA`空间混合24个expert的unit-Frobenius direction与
  envelope内log norm；shared rank96 left/right basis内做best-rank16 SVD。template-A Procrustes gauge
  只固定因子表示，不改变policy update。
- [x] zero/phase-constant表示逐tensor精确返回template-A/zero-B identity；Writer无Parameter，language
  没有独立value通路。真实24-expert one-hot effective cosine中位/最小=`.99838/.99665`。
- [x] evaluator只接受新config + 统一expert bank + feature cache + video data，不再接受learned Writer
  checkpoint。adapter/episode schema已升为policy-effective v3，旧trainer/checkpoint/model/compiler
  executable均不保留。
- [x] 全仓`182/182` CPU tests、compile、real fixed-asset inspector通过。真实资产runtime为0 parameters、
  68,863,192 persistent buffer bytes、build=`2.33s`、CPU batch24 compile=`.85s`；demo0 target cosine
  `.99836`，norm/stable/top=`4.179/1.125/.910`，A/B RMS=`.01891/.00846`，16 coordinates active。

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
7. [ ] 新compiler闭环先做预注册小panel筛选；只有absolute/breadth明确支持才做formal correct400，
   过门后再做same/wrong/shuffled/reversed/no-video严格配对。此前先从clean pushed commit做单卡A40
   online generation/cache/release/rollout smoke并重新seal formal。
8. [ ] 若one-shot的same-task视频方差成为经证据定位的最早限制，再评估固定K的few-shot set/sequence
   aggregation；不能把few-shot当作掩盖task identity或视频时序失败的捷径。

## 退役边界

`scripts/train_expert_manifold_writer.py`、learned Writer training/checkpoint模块及旧
`--expert-manifold-checkpoint`参数均已退役。当前动态Writer只有Policy-Effective Barycentric路径；任何历史
learned-Writer命令都只作provenance，不得从本文或旧记录复制恢复为并行实现。

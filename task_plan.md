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

## 当前唯一canonical实现

- [x] `configs/pi05_video_expert_manifold_causal_barycentric_v1.json`封存方法合同；formal状态在live
  A40 smoke前为`blocked_until_live_a40_online_smoke`。
- [x] 一条视频先形成`mean_phase(phase_centered_causal_memory(video_innovation))`；train24×50
  centroids单位化后，用ridge `.3` centered-kernel affine solve产生24个coefficients。
- [x] 168 chunks分别混合24个expert的unit-RMS direction和log-RMS，scale限制在该chunk的train24
  expert envelope内，再重构全部38 targets的rank-16 LoRA。
- [x] zero/phase-constant表示逐tensor精确返回template-A/zero-B identity；one-hot coefficient精确重建
  对应完整expert；Writer无Parameter，language没有独立value通路。
- [x] evaluator只接受新config + 统一expert bank + feature cache + video data，不再接受learned Writer
  checkpoint。旧trainer/checkpoint/model可执行路径与专属测试已删除，历史由Git和artifacts保留。
- [x] CPU回归覆盖合同漂移、信息墙、identity、one-hot重建、affine sum、ordered/reversed差异、
  scale envelope、完整shape与finite。真实24-basis只读检查：1,287,168 valid values、0 learned
  parameters、one-hot最大误差`2.235e-8`、zero identity exact、24/24 ordered/reversed coefficients不同。

## 下一证据门

1. [x] 当前实现已clean commit/push为`1d9d030`；后续工作从该提交或其纯authority后继提交执行。
2. [ ] 按live GPU与quota preflight选择一张空闲A40，做validation8×1-state online smoke：
   feature→coefficients→full LoRA cache→release Writer/encoder→复用同一source policy rollout。
   必须8 rows/8 generated/8 cache entries、0 retry/failure/OOM/nonfinite/forbidden reads；success只作
   execution smoke，不作性能证据。
3. [ ] smoke通过后把精确evidence写回barycentric config并seal，CPU回归、clean commit/push。
4. [ ] 从clean pushed frozen worktree做全新strict paired correct400；不复用learned Writer root或
   checkpoint。同步审计400 unique rows/LoRAs、完整task/suite breadth和generated-LoRA task separation。
5. [ ] correct400若同时改善absolute、breadth与task separation，再做same/wrong/shuffled/reversed和
   no-video严格配对；否则先定位video representation、coefficient solve、expert support或LoRA
   reconstruction中最早失效接口，再做单变量修订。
6. [ ] 若one-shot的same-task视频方差成为经证据定位的最早限制，再评估固定K的few-shot set/sequence
   aggregation；不能把few-shot当作掩盖task identity或视频时序失败的捷径。

## 退役边界

`scripts/train_expert_manifold_writer.py`、learned Writer training/checkpoint模块及旧
`--expert-manifold-checkpoint`参数均已退役。当前动态Writer只有Causal Barycentric路径；任何历史
learned-Writer命令都只作provenance，不得从本文或旧记录复制恢复为并行实现。

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

- [x] `configs/pi05_video_expert_manifold_causal_barycentric_v1.json`封存方法合同；live A40 online
  smoke证据已写回，formal状态现为`sealed`。
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
6. [ ] 按已通过CPU门在唯一runtime原位替换compiler；保持video reader、
   coefficients、expert/cache、38 targets、rank16、one-shot与zero identity不变，完成CPU合同后再申请
   A40 online smoke证据。
7. [ ] 新compiler闭环先做预注册小panel筛选；只有absolute/breadth明确支持才做formal correct400，
   过门后再做same/wrong/shuffled/reversed/no-video严格配对。
8. [ ] 若one-shot的same-task视频方差成为经证据定位的最早限制，再评估固定K的few-shot set/sequence
   aggregation；不能把few-shot当作掩盖task identity或视频时序失败的捷径。

## Causal Barycentric strict correct400 launch合同（2026-08-09）

- scientific seal=`0397be6`；实际评测必须来自包含本段launch record、clean pushed且与upstream一致的
  frozen branch`codex/causal-barycentric-correct400-20260809`，worktree固定为
  `/data1/user/ymdai/worktrees/EMBER-causal-barycentric-correct400-20260809`。评测没有Writer checkpoint，
  只接受sealed config、统一step2000 expert bank、train24×50 cache和每row一条在线action-hidden video。
- 05:10 CST live比较两节点：选择`gpu01:0,1,2|4,5,7`六张14MiB、0%空闲A40，严格保持NUMA0三张+
  NUMA1三张；物理3上41.6GiB他人VLLM、物理6及`gpu02:6/7`他人进程均不触碰。gpu01 host available
  memory=`479GiB`。启动前必须再次live复核目标六卡，任一卡变忙即不启动或重新登记空闲组合。
- `/data1`个人quota现场为`561,350,572/1,073,741,824 KiB`。400套FP32完整LoRA tensor预算=
  `2,064,364,800` bytes，加results/log/queue保守新增低于3GiB，远低于剩余额度。fresh root/log/tmux固定为
  `runs/outputs/pi05_expert_manifold_causal_barycentric_correct400_noreplacement_seed7_0397be6_20260809`、
  `runs/logs/pi05_expert_manifold_causal_barycentric_correct400_noreplacement_seed7_0397be6_20260809.log`和
  `ember_causal_barycentric_correct400_0397be6`；登记时root/log/worktree/branch/tmux均不存在。
- exact command（从上述frozen worktree在`gpu01`执行）：

```bash
env PYTHONPATH=$PWD/src CUDA_DEVICE_ORDER=PCI_BUS_ID NCCL_P2P_DISABLE=1 TOKENIZERS_PARALLELISM=false EMBER_STORAGE_ROOT=/data1/user/ymdai EMBER_STORAGE_CAP_BYTES=1099511627776 EMBER_LIBERO_ASSETS_ROOT=/data1/user/ymdai/projects/EMBER/data/simulation/ember_assets/datasets/libero-assets/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6 /data1/user/ymdai/projects/EMBER/.venv/bin/python scripts/evaluate_pi05.py run --config configs/pi05_target_evaluation_v1.json --source-run /data1/user/ymdai/projects/EMBER/runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722 --checkpoint /data1/user/ymdai/projects/EMBER/runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000 --tokenizer-path /data1/user/ymdai/projects/EMBER/models/tokenizers/openpi/paligemma_tokenizer.model --output-dir /data1/user/ymdai/projects/EMBER/runs/outputs/pi05_expert_manifold_causal_barycentric_correct400_noreplacement_seed7_0397be6_20260809 --role validation --mode formal --state-count 50 --replicas-per-gpu 3 --writer-generators-per-gpu 3 --writer-generation-batch-size 4 --gpu-indices 0,1,2,4,5,7 --expert-manifold-config configs/pi05_video_expert_manifold_causal_barycentric_v1.json --expert-manifold-expert-bank-root /data1/user/ymdai/projects/EMBER/runs/outputs/pi05_task_expert_bank_formal_step1000_r6_81101fe_20260807 --expert-manifold-feature-cache-root /data1/user/ymdai/projects/EMBER/runs/outputs/pi05_expert_manifold_feature_cache_train24x50_r6_222d3ac_20260808 --expert-manifold-video-data-root /data1/user/ymdai/projects/EMBER/data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a --expert-manifold-video-condition correct --expert-manifold-video-sampling without_replacement
```

- 科学panel固定8 validation tasks×50 states，correct one-shot、每task50 teacher videos无放回；task/state/
  video ordinal、env/policy RNG与旧source/address-binding panel严格配对。动态long-first queue，6卡×3
  persistent replicas、每卡3 generators、batch4；不挑video、state或outcome。
- 验收门为400 unique rows、400 unique LoRA references、72/72 jobs、18 workers attempt1/exit0、0
  retry/failure/OOM/nonfinite；每row teacher frames used且action/state/reward/terminal reads全0。同步报告
  aggregate、per-task/suite、breadth、相对source/address-binding/v6-fast gained/lost及400-LoRA geometry。
  correct的absolute、breadth或task separation不过可信门时，不启动其余五臂，先定位最早失效接口。

## 退役边界

`scripts/train_expert_manifold_writer.py`、learned Writer training/checkpoint模块及旧
`--expert-manifold-checkpoint`参数均已退役。当前动态Writer只有Causal Barycentric路径；任何历史
learned-Writer命令都只作provenance，不得从本文或旧记录复制恢复为并行实现。

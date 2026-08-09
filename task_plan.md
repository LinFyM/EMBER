# EMBER Task Plan

更新时间：2026-08-09。本文只保留当前可执行计划；旧架构的命令、分支和逐次流水由Git、
`progress.md`及formal artifacts保存，不在活动计划中保留可误执行副本。

## Goal与不可变边界

- [ ] 同一shared method/single checkpoint的strict paired correct严格超过`150/400`，并继续提高
  absolute、8-task breadth、稳定性与可重复性。
- [ ] correct在严格配对下实质优于wrong、shuffled、reversed与no-video；same-task-other保持鲁棒。
- [x] 保持one-shot：部署输入只有exact task language和恰好一条action-hidden teacher video；视频是
  唯一dynamic value，不增加language-only LoRA bypass、多video/LoRA平均或checkpoint融合。
- [x] validation/test actions不进入训练或选择梯度；task experts只使用train24 actions。
- [x] GPU工作前实时比较`gpu01/gpu02`，只使用空闲A40、合计最多6张，不干扰他人；多卡保持
  `NCCL_P2P_DISABLE=1`、NUMA physical/local rank映射和deferred-NCCL合同。

## 当前科学证据

- [x] 历史single-checkpoint最好仍是v6-fast macro400：strict correct=`143/400`、breadth=`6/8`，
  五臂correct/same/wrong/shuffled/reversed=`143/135/125/128/129`；目标仍未达到。
- [x] 统一step2000 task-expert bank完成：
  `runs/outputs/pi05_task_expert_bank_formal_step1000_r6_81101fe_20260807`。24/24 tasks、五个统一
  checkpoints；development-train direct-expert closed loop=`432/557/624/638/658` of 1200，故选择
  step2000且不按task混点。
- [x] learned address-binding、Causal Barycentric、Policy-Effective soft与hard部署路线分别在strict
  closed loop得到`75/400`、`63/400`、`15/80`和`3/80`，均已负裁决。hard/soft/sparse expert
  deployment dictionary关闭，不再调top-k、temperature、scale、confidence、rank或few-shot平均。
- [x] task experts能定义train-task policy-effective parameter targets，但不能保证held-task support、
  same-task video specificity或视频时间顺序因果性；这些仍必须由Writer和严格五臂闭环裁决。

## 当前唯一canonical方法与实现

- [x] 唯一方法是design第33节的v6-Prior Policy-Effective Temporal-Ranking Writer。部署恢复历史v6
  one-shot完整动态生成器：language+一条raw action-hidden video经Semantic Core、Causal Procedure、
  compiler和factor heads直接生成38-target rank16 LoRA；expert bank不进入部署。
- [x] 历史v6 macro400只作load-only初始化，不冒充exact resume。冻结encoder/Core/transition/Procedure
  共483 tensors、`7,060,992` parameters；只训练compiler+factor heads共41 tensors、`3,714,304`
  parameters，新建optimizer/scheduler/RNG。
- [x] 训练目标为correct positive functional loss + gauge-invariant effective`BA` expert direction/norm +
  bounded correct-over-reversed/shuffled/wrong ranking。same-task不同视频仍是共同positive分布。
- [x] 训练runtime由`dd57edc`及后续合同提交封存：6 ranks×4 tasks、train24等权、每task B20、一次flat
  all-reduce、50-video无放回cycle、完整six-rank RNG与exact-resume checkpoint。
- [x] canonical evaluator/runtime由clean pushed`bca3f6d`完成原位替换。它只接受v6-prior config、一个
  historical或本方法Writer checkpoint、raw video root和video condition；旧expert-bank/feature-cache
  deployment参数fail closed，hard-route config/model被删除。
- [x] evaluator的no-video臂不读取frames且精确返回source identity；correct/same/wrong/shuffled/reversed
  每episode恰好一条raw video。乱序与倒序只重排真实frame content并保留新的展示位置，随后做完整
  v6 forward；Writer生成FP32 LoRA cache后释放，原source policy原位复用。
- [x] CPU门：全仓`210 passed`；真实validation8资产inspect与CLI prepare通过。历史macro400 state=
  600 tensors、12,064,064 values；8 tasks映射到8个one-shot cache requests，部署expert-bank reads=`0`。

## 下一证据门

### 1. 单卡historical warm-start reproduction smoke

- [ ] 从包含`bca3f6d`及当前authority文档的clean pushed frozen worktree执行；不从活动checkout运行。
- [ ] 启动前重新检查两节点GPU ownership/telemetry/process与`/data1`个人quota，只选一张完全空闲A40。
- [ ] 固定validation8×state0、correct、seed7、without-replacement；历史macro400为method macro0。
- [ ] 生成8套完整LoRA并完成cache→Writer release→同一source policy rollout，要求8 rows/entries、
  0 retry/failure/OOM/nonfinite/forbidden reads，结束后GPU自然释放。
- [ ] 对同一批输入比较batched staged evaluator与逐episode direct v6 forward；全部76 tensors逐值比较，
  max abs difference必须`<=1e-5`。canonical smoke runtime会在写cache前自动执行并记录该比较；不用
  SHA/MD5，也不能事后凭观察补写通过。
- [ ] 通过后把精确device/root/commit/counts/release/reuse/direct-match evidence写回config并clean push；
  同一证据同时把gradient-profile从blocked改为ready。任一项失败先修工程合同，不启动六卡训练。

### 2. 六卡gradient与exact-resume profile

- [ ] 重新live比较两节点，最多选择6张空闲A40；按所选节点的NUMA建立physical/local rank映射，设置
  `NCCL_P2P_DISABLE=1`、Ring/Simple和deferred process group。
- [ ] gradient profile只运行预注册macro49，覆盖train24×B20=480 unique queries及最长105 sampled-frame
  video。分别测positive/expert/ranking在compiler与factor heads的未加权gradient norm。
- [ ] 一次性封存`lambda_expert/lambda_rank`，使每个auxiliary在两个trainable blocks中都不超过positive
  gradient的`.25`；不按validation outcome sweep或在线自适应。
- [ ] 用丢弃型profile权重完成fresh0→1、exact-resume1→3及独立contiguous0→3；科学metrics一致，Writer/
  RNG exact，optimizer/scheduler/cursor语义一致。profile权重不得warm-start formal。

### 3. Formal 0→50与严格选择

- [ ] 另建clean pushed frozen worktree、fresh root和formal run contract，从historical macro400 load-only
  初始化全新optimizer，训练50 macros并保存10/25/50。
- [ ] method macro0/10/25/50先跑相同validation8×states0--9 strict correct80；不得因中间task结果改变
  panel、video schedule、loss或checkpoint。
- [ ] 三个训练checkpoint全部跑paired correct400；method macro0也在同一当前schedule评测，历史143只作
  different-schedule reference。
- [ ] 只有single checkpoint correct严格`>150/400`，或至少不低于同schedule macro0且breadth不降、
  多task净增并有可信上升趋势，才运行correct/same/wrong/shuffled/reversed/no-video完整配对裁决。
- [ ] 若三点均低于macro0或只发生单task换手，停止该训练干预；不得用更长训练、loss sweep、scale/gate、
  checkpoint融合或立即解冻上游救点。

## 后续单变量顺序

- 若absolute提高而顺序/错误视频margin仍弱，下一变量才调整counterfactual credit；不得同时改encoder。
- 若margin提高但absolute下降，判该目标伤害policy performance，不把它解释成训练不足。
- 只有one-shot同task不同video方差被严格证据定位为最早限制，才设计固定K few-shot聚合；动态shot数后置。
- AS continuation仍不足时，再在同一输入墙和single-Writer图上设计短、task-balanced RL阶段；不能恢复
  flat Writer-RL、task-local RL或使用validation/test actions。

## 退役边界

旧hard-route config、online expert-bank/feature-cache部署、HardRouted Writer类及其CLI均已从canonical
runtime删除；历史只由Git和formal artifacts保存。task-expert trainer/evaluator、expert geometry和旧
feature artifacts仍可作为训练监督与历史分析工具，但不得重新成为Writer部署输入。当前没有运行中的
v6-prior GPU任务或长期实验。

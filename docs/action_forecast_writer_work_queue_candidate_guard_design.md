# Work-Queue Paired Candidate Guard

状态：2026-08-12 terminal mechanism non-pass。host-local cursor修复后的clean world3 reprofile完整执行96条
paired rollouts，19项checks中只有`negative_null`失败；deployment/formal关闭，不重跑或扫参数。本设计只淘汰
WQ-PCUG的correct-only final projection，不否定work queue、actual candidate pairing或negative-preserving
correction。唯一active successor见`docs/action_forecast_writer_negative_preserving_candidate_guard_design.md`。
PCUG的actual candidate paired-guard科学假设没有被检验；本设计只替换其最早失效的Phase-A任务所有权接口，
不得被写成新的Writer科学架构或PCUG同配置重跑。

## 1. Latest evidence and failure boundary

PCUG在clean world4 discarded macro中尚未完成full24 gather，wall下界已达`809.72185s`，是scaled matched
SKNC的`2.25568x>1.5x`。物理GPU6对应rank先等待，物理3--5仍忙；无OOM、nonfinite、paired outcome、
mechanism report或checkpoint。因此当前证据只支持：

1. actual candidate、K2x2 paired rollout、harmful classification与final closest guard均未被执行；
2. full24 global candidate形成前存在不可接受的rank-local tail；
3. 旧`cost_balanced_long_first_dynamic_uneven`不是运行时dynamic queue，只是按correct-video sampled-frame
   cost预先分配固定rank ownership；
4. world4四rank的静态video costs为`207/216/206/204`，但完成时间明显不按该cost排序，故这个proxy不足以
   控制真实Phase-A critical path；
5. 现有B20样本、teacher-video ordinal、counterfactual与policy RNG都是`(task_id, task_visit)`纯函数，
   task换rank不要求改变任何科学样本。

没有per-task live stage rows，不能把旧tail进一步武断归因到某个simulator、DataLoader、VJP或collective。
下一设计必须先得到task-level完成证据并消除固定ownership；不能通过重跑world size、放宽wall或猜一个静态
cost表来解释旧结果。

## 2. Single causal variable

Work-Queue PCUG保留PCUG全部科学合同：

- historical v6-fast frozen Writer、PICK-GC ordered condition与full48 negative zero-RHS；
- exact language + exactly one action-hidden correct video、stride5、Q/M/G、Core/Program；
- 每task一条video、B20跨episodeaction queries、full24等权blind `D0`；
- exact candidate `base_slots + cast(residual + motion)`与native 38-target rank16 compiler；
- 每task两个strictly paired initializations、base K2、candidate K2；
- current harmful/stable keys对`D0`的final closest equality projection；
- harmful guard ephemeral、first-stable bank persistent、FP32 Program、checkpoint与部署图。

唯一主要变化是：**Phase A的24个确定性task jobs不再预先属于某个rank；空闲rank完成一个job后从同节点共享
队列领取下一个。** task ordinal、visit、B20 rows、video、negative、policy RNG、objective、global sort与最终
update均不变。调度只改变physical executor，不改变数学样本或每task权重。

这不是通过更多GPU掩盖低效。world size仍按launch前live合适卡数决定，至多6、不等待凑满；队列要在同一
world size和同一24-task工作量下直接缩短critical path。

## 3. Task-addressable exact B20

`MixedTaskBatchSampler`新增一个公开的task-addressable入口：

```text
batch_indices_for_task_visit(step, task_id, task_visit) -> exactly 20 dataset rows
```

它必须复用现有episode order、teacher exclusion、Latin phase strata和jitter函数。对旧静态iterator中的每个
task，同一入口返回逐项相同的20个row indices；不得重新采样、按rank加seed或改变same-episode exclusion。

领取job后，rank直接对这20行做一次`default_collate`并交给现有processor。没有大规模preload、数据副本、
worker farm、逐tensor hash或额外forward。现有persistent DataLoader只服务固定rank batch stream，不能表达
完成驱动的跨rank领取，因此由这个窄入口原位替换；profile直接裁决同步HDF5读取是否足够快。

## 4. Host-local work queue

每macro由rank0在output root初始化一个只含原子cursor的小队列，随后所有rank进入同一个claim循环：

```text
while local_count < retained_task_cap:
    index = atomic_claim()
    if index >= 24: break
    task_id = deterministic_long_first_queue[index]
    load exact B20(task_id, task_visit)
    build one unchanged TaskObjective
```

队列用同节点共享文件的短`flock`保护单个整数；24次claim之外没有轮询、后台daemon、网络service或GPU同步。
rank0初始化与macro结束各一次distributed barrier。每个job记录`task_ordinal/rank/start/end`四个小标量，目的仅是
判断tail发生在哪个task与queue是否真正工作，不扫描tensor、不进入训练决策。

队列顺序沿用sealed sampled-correct-video long-first order；本实验不同时更换cost model。因为领取发生在每个
job完成后，错误cost只影响先后顺序，不再永久锁定rank ownership。

### 4.1 Memory bound

TaskObjective graph仍保留到full24 blind solve及paired probe，因此一个特别快的rank不能无限领取。live A40
合同在world size>=3时固定`retained_task_cap=8`：这是matched SKNC world3已经实际承载的每rank graph数，
不新增offload、recompute或compression。world4--6总capacity大于24，可发生真实work stealing；world3退化为
每rank8 tasks但保持相同科学结果。world1--2代码可用于CPU/synthetic，GPU不在未profile显存下授权。

若24 tasks尚未被领取而全部rank均达到cap，或任一rank超过8，implementation/profile直接失败；不自动增cap、
offload graph或减少B20。部分占显存但低利用率的GPU仍可用，但launch preflight必须为该rank最坏8-task graph
留出真实headroom。

### 4.2 Variable-count gather

full48 feature/cotangent、paired outcomes与task records继续使用一次per-phase collective，但padded local width固定
为8而非静态`ceil(24/world)`。present rows按task ordinal排序并要求恰好覆盖0--23一次。rank ownership和claim
order不得进入ridge solve、guard classification、bank slot或checkpoint。

exact resume仍只发生在macro boundary。completed macro已把Program、bank、sampler cursor和RNG写入checkpoint；
下个macro重新初始化空队列。调度的低位CUDA差异按项目正常BF16/TF32边界接受，不为逐元素一致固定rank。

## 5. Why the retained science still targets task knowledge

Work queue本身不宣称增加视频理解；它只让尚未被执行的PCUG科学检验变得可运行。高层任务知识路径完整继承：

- ordered video通过PICK condition与Program决定完整LoRA，不监督teacher低层动作；
- B20 action episodes与video错开，迫使同一video-conditioned LoRA服务不同source states；
- correct/negative full48约束保留对象、目标状态和阶段顺序的有向证据；
- actual candidate paired outcomes在相同未见初始化上直接辨识shared write是否损害闭环任务完成；
- 多macro更换video与state，只有在不同condition addresses上不造成可见伤害的shared Program能力才可保留。

language仍只作video query/context，没有LoRA value bypass；reward、task ordinal和queue ownership不进入Writer或
deployment。correct若不能严格优于wrong/shuffled/reversed/no-video，即使absolute过门仍是科学non-pass。

## 6. Coexistence and policy-effective write

共同积累机制也不变：B20 full24 blind solve提供acquisition，persisted first-stable keys保护长期support，当前
paired harmful/stable keys在最终shared write上阻断实测损伤，closest projection只做必要最小旋转。Program直接
写入已有FP32 memory并使用v6-fast native rank16 compiler；不引入第二LoRA、expert route、rank reservation、
global scalar reject或checkpoint融合。

队列的必要性在于：只要一个固定rank长尾阻塞`D0`，上述机制就根本没有机会被检验。work stealing使每张合适
GPU持续消费尚未完成的task，而不是让先完成rank在collective中空等。

## 7. Fast falsification

### 7.1 CPU and synthetic gates

1. task-addressable indices对world3/4/5旧静态schedule逐task逐row完全相同；
2. queue在不同claim interleaving下每macro恰好覆盖24 tasks一次，task visit与global sorted order不变；
3. rank task counts可变、每rank不超过8，padded gather仍严格对齐feature/cotangent/outcome rows；
4. task-keyed policy/video/counterfactual seeds不含rank或claim ordinal；
5. world3 cap8退化为完整8/8/8覆盖；world4--6允许快rank领取额外job；
6. queue cursor、timing rows和owner mapping不进入Writer、solver、checkpoint或deployment；
7. no-guard/blind candidate、exact candidate cast、paired classification和closest projection继续通过全部PCUG测试；
8. 一个canonical trainer与deployment family，旧static PCUG config不可resume。

### 7.2 One discarded live macro

从historical v6-fast、zero Program、empty bank运行train24 macro0。launch前同时live检查gpu01/gpu02，选一个节点，
按显存headroom、利用率、健康和实际吞吐使用3--6张合适A40；低利用率且只占少量显存的卡可共享，不等待6张，
不跨节点、不抢占。多卡保持`NCCL_P2P_DISABLE=1`、NUMA physical/local rank映射和deferred NCCL。

Phase A首先必须满足：

- 24 jobs恰好完成一次，B20 identities与sealed schedule一致，0 forbidden read/OOM/nonfinite；
- claim timestamps证明tasks在前一job完成后增量领取而非macro前预分配；实际task count允许恰好相等；
- max tasks/rank<=8，queue claim总CPU wall<1s，task timing rows完整；
- Phase-A full24 gather完成时间不超过matched SKNC world-size线性缩放的完整step wall；world4当前界为
  `358.970267s`。超过即停止，不进入96-rollout probe。

Phase A过门后，PCUG原live gates全部保持：48/48 exact pairs、base/candidate各48 rollouts，四suite
Program/LoRA/action response非零，discordant states>=4、harmful tasks>=2且跨至少2 suites、gains>=1，final
guard closure为零并实际改变`D0`，final/D0 energy>=.25且alignment为正，projected rank>=24，negative/correct
motion<=.15，production total wall<=matched SKNC`1.5x`。

任一hard gate失败即淘汰当前Work-Queue PCUG，不扫queue order、cap、worker数、world size、pair数、threshold、
scale、seed、dtype或projection tolerance。task timing只能定位失败，不得按task历史时间制作held-aware路由。

### 7.3 Sealed live result

`d799758` world3 reprofile的Phase A=`44.74125s`、claim=`.00558s`、total=`558.05862s / 1.16596x`，全部
吞吐、显存与coverage门通过。48 exact pairs产生7个discordance、3 gains/4 losses、3 harmful tasks跨2 suites；
15-row correct guard保留`.76492`能量、rank33且closure为零。blind solve的negative ratio原为`.03991`，但
correct-only final projection后升至`.50179`，wrong/shuffled/reversed均`0/8`达门。paired前接口、reward evidence、
projection可行性和action传递都已接通；最早失败明确是final guard composition丢失negative-video抑制。

## 8. Formal decision

live机制与B8/16/32 deployment profile全过后，fresh训练`0->5`并立即跑与old134相同的strict paired
correct400。只有同时满足以下条件才允许`5->10`：

- macro5 correct>=142、breadth>=6；
- old134->macro5 lost<=8且gained>lost；
- 至少3个suites不降，最大单task净增不超过全部正净增的`.5`；
- 每macro有非零discordant/harmful evidence，guard closure、rank、energy和queue throughput不坍缩。

首次correct>=144才补same/wrong/shuffled/reversed/no-video。最终成功仍要求同一single checkpoint strict
correct>150、correct严格优于全部negative controls、same-task-other至少保留correct的`.9`。macro5未过门则
不resume、不补controls、不做调度或科学参数sweep。

## 9. Rejected alternatives

- 不重跑static PCUG换world size：它不能解释或修复错误ownership proxy；
- 不用一次profile拟合task-specific静态latency表：容易把瞬时系统条件写成科学route，且仍会被单task尾拖住；
- 不只把base rollout挪到gather前：所有rank都会增加工作，不能减少最慢Phase-A critical path；
- 不拆B20、降低batch、offload/recompute graph或扩大dtype：这些改变数值/显存合同且不是当前最早证据；
- 不增加stage journal、tensor hash或防御性watchdog体系：24条task timing与既有profile门已足够裁决；
- 不转few-shot、expert manifold、rank/topology或新video encoder：actual candidate pairing尚未被检验，当前没有
  架构级证据支持这种转向。

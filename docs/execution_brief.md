# EMBER Current Execution Brief

## 0. Outcome and current operation

目标是把同一shared method、同一single checkpoint的strict paired correct从历史最好`143/400`推进到
严格`>150/400`并继续提高，同时保留真实视频时序因果、same-task鲁棒、breadth和稳定积累。当前没有
运行中的EMBER GPU任务，也没有v6-prior新性能结果。

当前操作顺序：

1. 封存吞吐纠偏代码和统一authority；
2. 真实资产CPU prepare/inspect；
3. clean pushed frozen worktree；
4. live检查`gpu01/gpu02`后，一张空闲A40做Writer batch/VRAM/端到端smoke；
5. 最多六张空闲A40做gradient weight、fresh/resume/contiguous和训练吞吐profile；
6. formal continuation和关键checkpoint strict rollout；
7. 将结果与完整历史谱系作逐task/机制对比，只改最早失效接口，循环到达标。

不得从下文自行跳到later stage，也不得从历史文档恢复已退役命令。

## 1. Fixed scientific contract

- 方法：one-shot v6-Prior Policy-Effective Temporal-Ranking Writer。
- 输入：exact task language + exactly one action-hidden raw teacher video。
- 视频是唯一dynamic value；无language-only LoRA bypass、expert-bank部署、multi-video/LoRA/checkpoint
  平均或融合。
- 输出：一套完整38-target public rank-16 LoRA；Writer在rollout前运行一次后释放。
- source policy、normalization、split、frame stride5、LIBERO preprocessing与paired evaluator固定。
- historical v6-fast macro400只作load-only Writer初始化；冻结encoder/Core/transition/Procedure，只训练
  compiler+factor heads；全新optimizer/scheduler/sampler/RNG。
- train24 task-complete、每task B20跨episodequeries、每visit一条correct video、24-task等权、一次flat
  all-reduce。
- objective固定为positive functional + effective-BA expert direction/norm + bounded temporal/wrong
  ranking；auxiliary weight只由预注册train24 gradient profile选择一次。
- step2000 task experts仅作train supervision，不进入deployment或held选择。

## 2. Throughput-first runtime contract

“精度”指科学合同和真实closed-loop，不指底层逐元素复现。执行中遵守：

- 接受正常BF16 kernel、batch shape和reduction order的低位差异；不因`.001953125`级LoRA roundoff固定
  batch1、重复single forward或逐tensor扫描。
- 只门禁shape、finite、信息墙、cache完整性、明显跨样本污染、OOM、错误asset和resume语义。
- Writer生成从batch8起profile，逐步增加到16/32/更高，直到吞吐进入平台、allocator明显抖动或接近
  OOM；选择samples/s最高且能稳定完成真实最长video batch的点。目标是让A40 reserved memory保持高
  利用而非留出大块空闲，不用dummy tensor填充。
- LoRA cache保持template原生72 BF16 + 4 F32 tensors，单entry tensor data=`2,641,920` bytes；不强制
  扩宽FP32。每batch集中nonblocking D2H，只同步一次。
- action query DataLoader默认2 spawn workers、persistent workers、prefetch2；profile若显示GPU仍等待
  data，再实测4 workers/prefetch而不是猜测。
- functional B20能单物理batch时走直接gradient path；只有显存要求真实microbatch时才使用FP32 gradient
  accumulation。microbatch、activation checkpoint和worker数按samples/s、step wall和peak memory联合
  选择，不以数值逐位相同为门。
- 不做SHA/MD5，不重复全仓hash或历史artifact扫描。CPU全仓回归只在代码合同变化后运行一次。

保留FP32 RMSNorm/softmax/ROPE/image normalization和policy-effective reduction，除非profile证明它们是
显著瓶颈且降低精度不伤真实闭环；吞吐优先不是盲目改变模型数学。

## 3. CPU seal before GPU

GPU前一次性要求：

- `git diff --check`和聚焦/全仓CPU tests通过；
- historical 600-tensor Writer strict-load，frozen/trainable ownership为
  `7,060,992/3,714,304` parameters；
- validation8 real-asset inspect和CLI prepare产生正确one-shot requests，部署expert-bank reads=`0`；
- no-video不读frames并返回template-A/zero-B identity；wrong/shuffle/reverse只改变允许的视频输入；
- native LoRA storage descriptor从checkpoint metadata贯穿run contract与cache write/load；resident policy的
  destination dtype由已验证的同一template决定，正常路径不发生额外转换，不在每次replan加dtype扫描；
- 2-worker prefetched sampler与serial、prefix+resume逐row一致；
- config状态仍blocked，直到单卡vertical smoke真实通过。

CPU门不要求batched Writer与single Writer逐元素相同，也不解释性能。

## 4. Single-A40 throughput and vertical smoke

### 4.1 Live preflight

同时查询`gpu01/gpu02`，记录每张卡index/UUID、memory、utilization、process和owner；只选一张完全空闲
A40。检查`/data1`个人quota、当前项目用量和本次cache/log峰值。不得查询后长时间等待再沿用旧快照；
真正launch前再做一次短确认。

### 4.2 Writer batch sweep

使用historical macro400、真实validation视频和包含最长/异长样本的固定请求集。canonical入口是现有
`scripts/evaluate_pi05.py profile-writer-generation`子命令，不另建平行evaluator。batch候选至少覆盖
`8,16,32`，若32仍有明显显存和吞吐空间再在新profile加入更高点。所有候选必须处理同一个由最大候选
确定的longest-first request panel、相同entry IDs和相同总sampled frames；较小候选把该panel切成多个
forward，不能各自只测不同长度的prefix。每点最低必须记录：

- 一次warmup、至少两次真实完整video→Writer→LoRA→native D2H的repeat wall与LoRAs/s；
- actual forward batch分组、完整固定entry IDs、sampled-frame counts/total和panel最长视频覆盖；
- peak allocated/reserved、device total、显存余量与稳定性；
- 76 tensor shape/dtype/finite、0 forbidden reads和模型释放证据。

同一模型一次load后完成候选点，profile输出保留为evaluation-seal evidence root，任何OOM候选不重复盲试。选择吞吐最高且
最长真实batch连续运行稳定、保留必要headroom的batch；不是选择数值最接近single的batch。如果候选吞吐
接近、重复波动大或GPU未吃满而原因不清，再补encoder/compiler/D2H分段计时、allocator retry、data wait和
连续GPU utilization trace；第一轮不为尚未出现的瓶颈增加同步和instrumentation。

### 4.3 End-to-end smoke

在选定batch上从fresh root运行validation8×state0、correct、seed7、without-replacement：8个唯一video
→8套完整LoRA→cache→释放Writer/encoder/tokenizer→原位复用同一source policy→8个rollouts。要求：

- 8 requests/generated/cache/rows，0 retry/failure/OOM/nonfinite；
- teacher action/proprio/reward/task-ID等forbidden reads均0；
- Writer modules释放，source policy不reload；
- cache native dtype/bytes正确；进程退出后GPU自然释放。

smoke的success count只作执行信息，不是性能证据。通过后把device、commit、root、batch、wall、peak、
release/reuse和错误计数写回config，状态改为gradient-profile ready。不得人工手填evidence：必须用
`assemble_v6_prior_evaluation_smoke_evidence(profile_root=..., vertical_root=...)`从两个retained roots重建
并复验seal。

## 5. Six-A40 gradient, resume, and throughput profile

重新live比较两节点，最多选择6张空闲A40；按实际设备建立3+3或合同允许的NUMA physical/local rank
拓扑，显式`NCCL_P2P_DISABLE=1`并使用repository deferred-NCCL launcher。

profile固定train24 macro49，覆盖24×B20=480 unique跨episodequeries并包含最长105 sampled-frame video：

1. 分别测positive、expert、ranking在compiler和factor heads的未加权gradient norm；
2. 一次性选择`lambda_expert/lambda_rank`，使每个auxiliary在两个trainable blocks都不超过positive的
   `.25`；若初始化已满足而梯度近零，不人为放大；
3. 同时记录每rank video encode、policy functional、auxiliary、backward、flat all-reduce、optimizer、
   input wait、GPU utilization和peak memory；
4. 在不改变B20/full24 objective的前提下实测physical microbatch、query loader workers/prefetch和必要的
   activation checkpoint；取吞吐最高配置；
5. 用丢弃型profile权重完成fresh0→1、same-root exact-resume1→3、独立contiguous0→3。要求scientific
   metrics、cursor、Writer/RNG和optimizer/scheduler语义一致；允许正常并行低位roundoff，不要求不同
   reduction schedule逐bit相同；
6. profile checkpoint永久禁止warm-start formal。

gradient结果封存前需由结构化verifier从`gradient_profile.json`、completion和run contract重算推荐权重并
核对24 tasks、macro49和六卡拓扑；fresh/resume/contiguous完成后也需由artifact verifier核对同一config
bytes/commit/topology、completion、cursor/RNG和optimizer/scheduler语义。二者都是低成本CPU封存，不做
hash，也不能被人工status替代。

工程故障按rank/device/process-group/CUDA/I/O/NUMA层定位。不得用加timeout、关watchdog、减少B20或盲重试
掩盖问题。

## 6. Formal training and truthful evaluation

### 6.1 Baseline and training

另建clean pushed frozen worktree、fresh output root和formal launch contract。method macro0是同一当前
video schedule下的historical macro400 load-only状态；不能仅引用旧143替代当前paired baseline。

第一段训练0→50 macros，保存10/25/50。训练期间持续记录：

- positive/expert/ranking loss和unweighted component gradient；
- per-task loss、effective cosine/norm、correct-negative margins；
- compiler/factor task-gradient cosine、full24 retention、clip/nonfinite和step wall；
- video schedule、counterfactual counts、sampler cursor和每rank RNG。

loss下降不能延迟或替代rollout。若出现非finite、明显全task退化或合同破坏，立即停；普通loss波动不作
科学淘汰。

### 6.2 Checkpoint cadence

- macro0/10/25/50先在固定validation8×states0--9跑strict paired correct80作灾难/趋势screen；panel、
  state、video和RNG不因结果改变。
- 训练checkpoint 10/25/50和同schedule macro0都跑correct400。小screen不能选winner或代表真实水平。
- 每点与macro0、历史143、v5.2-old和v6交叉recipe逐task比较：per-suite、breadth、gained/lost、
  union/intersection、capability churn和视频ordinal依赖。
- 若多个task共同上升且50仍有可信趋势，按同合同exact-resume到100（必要时200），每25或50保存并及时
  correct400；不因单点低分180度转向，也不因loss好看无限续训。
- 任何single checkpoint correct严格`>150/400`，或成为有可信共同提升的当前winner时，运行完整paired
  correct/same/wrong/shuffled/reversed/no-video。达标后仍继续验证更高性能、breadth和稳定性。

### 6.3 What counts as real improvement

方法改善至少需要absolute和任务分布证据，而不是一个suite换手。完整报告：

- aggregate/per-suite/per-task successes；breadth；
- 相对macro0和历史best的retained/gained/lost/both-fail；
- 相邻checkpoint成功集合Jaccard、union/intersection和single-envelope gap；
- paired correct-vs-same/wrong/shuffled/reversed/no-video；
- same-task不同teacher video方差；
- Core→Procedure→effective BA→fixed-action的条件传递；
- LoRA norm、stable rank、top energy、B-column cosine和跨target energy作为机制参照。

LoRA健康但分数低仍是失败；LoRA近rank1但分数提高不能仅因“不像SFT”淘汰。

## 7. Post-result diagnosis and next single variable

按以下顺序定位最早接口：

1. **evidence extraction**：correct/shuffle/reverse/wrong在Core前是否含任务和时序信息；
2. **Core/Procedure**：信息是否形成正确阶段/顺序，而非仅对低层异常敏感；
3. **compiler**：Procedure差异是否保留到policy-effective BA，是否被task-common方向压缩；
4. **factor heads/topology**：能量是否进入正确targets/rank coordinates而非同向高能量；
5. **functional/expert/ranking credit**：梯度是否对task可累积、是否由auxiliary绑到train expert流形；
6. **optimizer/task aggregation**：full24更新是否抵消并导致checkpoint轮换；
7. **closed-loop mismatch**：内部改善是否落到真实action和成功集合。

相应决策：

- absolute升、视频margin弱：下一变量只改counterfactual credit或Procedure temporal objective；
- margin升、absolute降：ranking伤害policy，不解释成“还没训够”；
- expert cosine升、held迁移降：减弱/重构train-expert流形约束，不转回online expert bank；
- compiler梯度健康但BA传递弱：修改compiler ownership/topology，不换video encoder；
- BA与action传递健康但任务继续换手：处理task aggregation/credit coexistence，不靠增rank/scale；
- same-task跨video方差成为最早限制：才设计固定K few-shot invariant aggregation；
- 所有下游目标与传递成立而上游无正确顺序语义：证据才允许解冻/重构Procedure或更早表示。

每个新设计必须明确继承旧方法已证明有效的组件、只改变哪个假设、预期影响哪些内部和closed-loop指标、
何种结果淘汰该假设。禁止东一榔头西一棒。

## 8. Evidence, Git, and retention

- canonical branch：`codex/bci-continuation`；正式root绑定包含该run contract的clean pushed commit。
- current config：`configs/pi05_v6_prior_policy_effective_writer_v1.json`。
- current design：`docs/action_forecast_writer_video_expert_manifold_design.md`第33节。
- current training/eval entry：`scripts/train_v6_prior_writer.py`、`scripts/evaluate_pi05.py`。
- formal保留config、command/env、GPU topology、checkpoint schema、raw rows、aggregate、completion和必要
  mechanism analysis；不提交checkpoints/cache/data/binaries。
- failed `30b2ccf` smoke及其`batch_equivalence.json`保留为BF16诊断；不resume、不作为性能证据。
- 历史design/findings/progress/Git/formal artifacts保存所有已验证经验；当前authority不重复堆叠旧
  “下一步”。清理只移除已确认obsolete的活动path/worktree/temporary output。

## 9. Stop or ask conditions

只有以下情形暂停并请求owner：设备边界/ownership不明确、需要干扰他人进程、quota无法容纳、必须改变
split/信息墙/test边界、需要删除含唯一证据或不明ownership的资产、或同一实质阻塞在安全替代方案后仍
无法推进。普通科学负结果、一次OOM、工程bug或候选不达标不构成沟通门，应按上述证据链自主继续。

# EMBER-ECP architecture contract

状态：等待owner已发送的最新专家复核。本文记录已经对齐的架构骨架、当前最具体的数据流和必须由回复裁决的开放点；它不是
对尚未确认部分的执行授权。人工process数据路线已经关闭，后续只使用现成授权LIBERO tasks。

## 1. 部署图

```text
exact language L + K ordered action-hidden videos V
  -> frozen PI0.5-native frame observer
  -> learned ordered event binding, independently per video
  -> permutation-invariant cross-video posterior q_V(P | L,V)
  -> one shared Program-to-LoRA realizer
  -> one complete 38-target rank16 LoRA
  -> frozen source PI0.5, zero-interaction closed loop
```

Writer只在rollout前运行一次。部署图没有teacher action、state、reward、task ID、checkpoint选择、第二adapter或环境交互。

固定符号：

- `K`：视频数；训练若声称Dynamic-K，必须真实覆盖多个cardinality；
- `T_k`：第`k`条视频stride-5后的帧数；
- `D=128`：当前PI0.5 Action Expert hidden宽度；
- `H=50`：action horizon token数；
- `O=38`：完整LoRA target owner数；
- `E=8`：当前Program的最大event slot容量。

## 2. Frame observer

每个视频帧都使用原始PI0.5应有的输入：exact language、真实双相机图像prefix和Action Expert action suffix。教学视频没有
action，因此suffix不是teacher action，而是一组固定、可复现的Gaussian noise probes：

```text
language tokens       X_lang[k,t,l]       -> [L_k, D]
visual patch states   X_patch[k,t,p]      -> [256, D]
noise action tokens   epsilon[k,t,h]      -> [50, 32]
Action Expert states  Z_ae[k,t,layer,h]   -> [N_layer, 50, D]
owner-aligned states  Z_owner[k,t,o,h]    -> [38, 50, D]
```

flow时刻`t_flow=1`是噪声端点。50个输入token按未来动作horizon排列，Action Expert在层内与token间处理它们，所以hidden同时
带有当前帧条件、层语义和时间位置；但它们不是“预测好的50步动作”。固定probe只提供一个统一坐标，用于比较不同帧和任务下
source policy的内部响应。

第一版冻结PI0.5全部参数，只训练post-capture projection。Action Meta-LoRA不作为首版依赖；随后必须做一次matched attempt，
若无负面且有净收益则启用并永久冻结。

## 3. 帧到有序event

每条视频独立处理，不先平均frames。observer从`Z_owner`、视觉变化和language query形成每帧transition representation：

```text
R[k,t,o] = FrameProject(Z_owner[k,t,o,:,:], X_patch[k,t,:,:], X_lang[k,t,:,:])
           -> [38, D]
```

一个单调有序的soft segmenter输出frame-to-slot assignment与slot presence：

```text
A[k,e,t]   -> [E, T_k],  sum_e A[k,e,t] = 1
rho_k[e]   -> [E]
V[k,e,o,d] = ordered_pool_t(A[k,e,t], R[k,t,o,d])
             -> [E, 38, D]
```

`E=8`是固定容量，不是固定任务复杂度。简单任务可以只激活少量slot，复杂任务可以激活更多；具体视频段落落入哪个slot、边界
在哪里都由训练学习。约束只规定slot有序，防止event身份在时间上任意交换。

## 4. Dynamic-K Program posterior

每条视频先得到自己的`V[k]`和`rho_k`。跨视频模块以language/owner/event query读取这些已保序的event表示，再做集合级
置换不变聚合；它可以学习可靠性权重和不确定性，但不得平均最终LoRA或先把原始frames压成一个均值。

当前候选Program schema为：

```text
P_lang    [38, 128]
P_scene   [38, 128]
P_process [8, 38, 128]
rho       [8]
sigma     [8, 38, 128]
```

部署网络`q_V`输出这个结构的posterior，而不只输出一个deterministic tensor。posterior可具体实现为particles/mixture或
mean+structured scale；最终形式尚待专家裁决，但必须保留event、owner/layer和跨视频不确定性。

## 5. Privileged posterior `q_pi`

`q_pi`是训练期共享网络，不是手工Program、task-local free code或外部专家。它只能在train24及审计后的non-held LIBERO
meta tasks上读取多个独立successful policies、跨状态policy responses、actions/occupancies/reward与uncertainty，输出与`q_V`
完全同构的Program posterior：

```text
q_pi(P | successful policy evidence) == same schema as q_V(P | language, videos)
```

多个policy member必须来自独立优化lineages或预登记fixed checkpoints；不能按held结果挑member，也不能把raw LoRA factors、
filename或task ID直接编码成Program。rollout-only recovery信息只能帮助共享prior、uncertainty或realizer训练，不能要求`q_V`从
视频预测不可观察变量。

`q_pi`是否合理不由“它是teacher”这一名称保证，而由以下证据保证：

- task-disjoint held inference，没有held free code或optimizer；
- 冻结Program schema与realizer后仍能生成唯一LoRA；
- held closed-loop显著高于source/shared carrier并覆盖Goal/Long；
- posterior在不同successful lineages间一致，同时保留必要不确定性；
- `q_V`能从language+video逼近其可观察部分并取得视频因果增量。

## 6. Program-to-LoRA realizer

realizer的权重跨所有训练任务共享；“共享机制”指同一组网络参数把不同任务的Program映射成LoRA，而不是给每个task独立训练
一个decoder或查表。输出必须一次性覆盖38 targets的rank16 `A/B` factors，并在materialize后成为唯一adapter。

已经排除的实现组合包括：任意低维code到raw factors的无约束hyperdecoder、固定`A`只解`Delta B`、历史rank12 carrier+
rank4 residual惯性方案、直接在held effect bank上做task-local solver，以及Program与decoder共同旋转的deterministic latent。

仍需专家明确的核心选择是：

1. realizer直接生成`A/B`，还是先预测owner/event-conditioned policy-effect distribution再用固定可微operator实现；
2. 如何固定LoRA坐标，避免`A/B`因子等价性让Program监督漂移；
3. posterior marginalization在生成一套LoRA之前发生在哪一层；
4. realizer应先于`q_pi`单独校准，还是与`q_pi`分阶段联合但周期性冻结。

任何选择都必须先在train24 leave-out和多个non-held mappings上做冻结共享映射的closed-loop Gate，不能用reconstruction loss
替代。

## 7. 训练流程（待专家确认的最小完整版本）

### Stage A：资产与自然任务映射

复用现有source policy、train24/non-held task experts、successful trajectories、LIBERO videos和固定split。审计独立task mappings
数量、language/video重复项与可用成功lineages；不生成新任务或人工数据。

输出：明确的train/meta/held映射表和可复用资产清单。通过条件是没有validation/test梯度、重复任务泄漏或task-ID route。

### Stage B：Program与privileged shared realization

固定Program schema。用授权meta tasks训练`q_pi`与shared realizer，先做模块冻结阶段以阻止坐标共同旋转，再用held5和多fold
检查：privileged evidence是否能通过同一realizer直接产生一套有效LoRA。

通过条件：显著超过source/shared基线、接近独立successful member breadth、5/5 held task有贡献、Goal/Long非零、相邻checkpoint
稳定，并恢复有信息量比例的oracle gap。失败先定位Program、posterior还是realizer，不继续训练`q_V`。

### Stage C：Video posterior

冻结通过的Program坐标和realizer，只训练frame observer、event binding、Dynamic-K aggregator和`q_V`，用`q_pi` posterior的可观察
部分、event alignment和policy functional loss监督。训练video与action query跨episode。

通过条件：在未用于梯度的held/validation开发任务上，full video生成的LoRA显著优于language/no-video、scene/first+final和wrong
video；same-task其它视频高retention，Goal/Long有增量。

### Stage D：全Writer联合训练

当Stage C建立正向闭环信号后，冻结PI0.5 backbone与固定坐标，解冻所有Writer模块做低学习率联合训练：frame projections、event
segmenter、Dynamic-K posterior、Program compiler和realizer中被专家允许联合的部分。必须保留阶段checkpoint，防止联合训练破坏
已通过接口。

若Action Meta-LoRA matched attempt通过，也在本阶段之前校准并冻结。最终Writer checkpoint必须单独可加载与复现。

### Stage E：outer credit与正式评测

只有full-video闭环增量已成立，才可用train/meta simulator reward做共享structured outer objective；deployment仍zero-interaction。
随后用全部授权开发训练数据从fresh训练最终方法，做validation8 strict paired400和相邻checkpoint稳定性。方法冻结后才测试
shuffled/reversed；Test8只在最终选择完成后使用。

## 8. 决策与停止条件

- 单次主要改变一个因果接口，但不把架构研究降格为无穷小超参扫。
- smoke只证明图接通；任何“方法有效”结论必须来自single-checkpoint closed loop。
- 若privileged shared realization在合理Program/coordinate/realizer替代下仍无法跨task产生Goal/Long增量，暂停并讨论ECP核心可识别性。
- 若privileged Gate通过而`q_V`失败，问题定位为video-to-Program，不推翻realizer。
- 若full video不优于language/static controls，不进入outer credit或最终测试。
- 不再恢复人工process路线、GOMQ基线工程、PECS solver或v24小变体作为ECP主线。

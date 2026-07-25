# Action-Forecast Writer execution handoff

状态：2026-07-25 owner 最终对齐，供新的独立 session 直接接手。

本文件是当前 Writer 子任务的活动设计与执行 authority。它覆盖此前
Action-Memory / temporal-RoPE Writer 的活动实现口径，但不改写那些实验的
历史结果。若旧配置、旧测试、`task_plan.md` 的已完成历史条目或早期设计文档
与本文件冲突，以本文件和 owner 最新指令为准。

### 2026-07-25 Belief-v3 owner override

本节覆盖下文仍将Plan/Revision写成两个token、使用adjacent revision、
additive/type routing、普通K/V共享归一化、可调stride或约30分钟训练segment的
活动描述。那些内容只保留为v1/v2 provenance。

- 唯一活动配置升级为
  `configs/pi05_as_writer_action_forecast_v3.json`；新架构必须fresh训练，不得
  resume任何v1/v2 Writer checkpoint。
- `frame_stride`固定为`5`，不再profile stride10。只profile每rank action-query
  batch和`frame_microbatch_size`等纯效率参数。
- state前端保持28个content-only virtual-state tokens；VL Meta-LoRA rank4、
  Action Meta-LoRA rank8、完整10-step flow、两层Temporal、320个queries和
  public rank16 LoRA schema保持不变。
- 每个绝对控制时刻只产生一个256维
  `Belief_u=[Plan_u(128)|Revision_u(128)]`。`Plan_u`只编码覆盖`u`的最新
  forecast 7维action；lead只能进入Q/K routing。
- `Revision_u`不再比较adjacent forecasts，也不包含old/new绝对action。所有
  更早且覆盖`u`的forecast都相对最新Plan构造
  `r_i,u=Plan_u-a_i,u`；value只含signed residual、absolute residual和
  disagreement magnitude statistics，lead/age/count只作routing。
- raw source-normalized residual RMS记为`m_u`，方向content记为
  `D_u=RMSNorm(z_u)`；最终
  `Revision_u=stopgrad(m_u)*D_u`。无较早forecast或所有forecast完全一致时
  Revision严格为零。`m_u`直接使用frozen source normalization下的无量纲
  7维action residual RMS，不设置`tau`、训练集分位数或其他人工尺度超参数；
  routing中的strength同样detach。AS loss仍可训练Revision方向，但不能通过
  主动放大或缩小forecast来操纵显式分歧强度。
- concat后不做独立half normalization或任意post-concat projection。
  Temporal只可在Q/K或FFN pre-norm中对整个Belief使用共同归一化；Q/K还可读取
  lead/count/strength routing和absolute-time RoPE，V只读取raw Belief。
  token type、routing、position和bias都不得进入content residual。
- LoRA query decoder保持routing/content分离：静态module/layer/rank identity
  只进入Q/K，`Z_0=0`，self-attention V只读Z，cross-attention V只读raw
  temporal memory，factor heads只读Z。人工将Belief content置零时，无论长度、
  routing或position如何，Temporal、query content和dynamic LoRA都必须为零。
- AS仍只有normal positive functional action loss，不增加contrast loss。
- 新架构从随机初态一次连续训练`0→600` optimizer steps；可每75 steps密集保存
  exact-resume checkpoint，但不中途主动停训或切换评测。600完成后统一评测
  多个validation checkpoints。step600的顺序特异性先做低成本内部数值诊断；
  只有内部差异明确且跨多个tasks/videos稳定，才运行实际shuffled/reversed
  paired validation arms。
- 若顺序特异性明确，再检查绝对性能并以更大step跨度继续寻找AS validation
  observed-best和显著、稳健、多task共同贡献的峰后下降；双门通过后才推进独立
  cold-start RL。任何科学gate经第一性原理分析仍无法修复时，停下向owner汇报；
  不用contrast loss挽救。
- GPU只使用0、1、2、3；即使这些卡已有进程也按owner授权共享，4–7绝不触碰。

### 2026-07-25 v2 historical architecture override（已被Belief-v3覆盖）

owner 已明确拒绝用 contrast loss 人为制造顺序差异，并要求从第一性原理修正
信息流。以下口径覆盖本文件中仍描述旧 8-scalar imagined-state、additive
revision stability、static-query residual decoder 或 order-contrast 的段落：

- state 前端直接从完整 SigLIP/projector image tokens 生成 28 个
  PaliGemma-width virtual-state tokens；保留的一个真实 whitespace token使
  `State:` 到 `Action:` 区域共29个位置，对齐真实train state文本的
  mean `28.8898`、median `29`。不再生成8个scalar，也不再经过
  `tanh`/Fourier scalar bottleneck。
- 28个state slot的learned identity只作attention routing。content从零开始；
  self-attention使用`Q/K=Norm(Z)+R, V=Norm(Z)`，cross-attention使用
  `Q=Norm(Z)+R, K/V=image memory`，residual和最终2048-d projection只读取
  `Z`。因此静态slot不能绕过视频内容直接生成state tokens。
- `Revision_u`仍读取有向
  `[old_7,new_7,new-old,lead_old,lead_new,delta_t]` event，但learned query只
  定位event，不进入输出residual。directed content独立RMSNorm；
  count/delta-norm mean/std/max先归一化，只形成
  `1+0.25*tanh(MLP(stats))`的有限乘法gate，不再作为additive embedding覆盖
  有向内容。`Plan_u`也独立RMSNorm后再进入temporal Transformer。
- LoRA decoder同样将routing identity与content拆开：`Z_0=0`，
  self-attention为`Q/K=Norm(Z)+R,V=Norm(Z)`，cross-attention为
  `Q=Norm(Z)+R,K/V=temporal memory`，factor heads只读取`Norm(Z)`。
  factor heads全部bias-free，静态query没有直达public LoRA的旁路。
- AS只保留normal positive functional action loss。旧order-contrast配置和
  训练分支已退役，不得恢复。
- 唯一活动配置是
  `configs/pi05_as_writer_action_forecast_v2.json`。新架构从fresh identity
  训练，GPU0–3、stride5、frame-microbatch32、每rank 16 action queries；
  每75 steps保存checkpoint，先训练到600并完整评测300/600。若尚未出现经
  多task和独立复测确认的明显峰后下降，则以600-step大段继续到1200及之后。

在旧step1200 checkpoint上做的无训练counterfactual机制诊断显示：新Revision
合成的time-centered normal→reversed/shuffled相对L2中位数为
`0.3554/0.2418`，旧Revision token只有`0.0281/0.0316`；raw directed events
为`0.2233/0.2296`。证据位于
`/data/ymdai/outputs/ember/pi05_action_forecast_step1200_revision_v2_counterfactual_val8x2_20260725/summary.json`。
它只证明新合成没有在进入temporal前抹掉上游顺序信号，不替代新架构的正式
训练、closed-loop performance或最终specificity评测。

## 0. Active Goal原文

新session若`get_goal`没有返回active Goal，使用以下原文创建且不设置
`token_budget`：

> 在 EMBER 中完整实现并核查 owner 已认可的单-token Belief_u Action-Forecast Writer 架构；固定 frame stride=5，只优化训练/评测的 batch 与 frame-microbatch 等效率参数；从新架构随机初态连续训练到 600 optimizer steps（期间可密集保存 checkpoint，但不按约半小时 segment 主动停训或切换评测），随后评测多个 validation checkpoint并对 step-600 做视频顺序特异性诊断。若特异性明确，则继续按较大步长充分探索 AS Writer 的 validation observed-best 与显著峰后下降，并在绝对性能良好后推进独立 cold-start RL Writer；若任一科学 gate 经分析和第一性原理架构修正仍无法通过，则停下向 owner 汇报。全程不引入对比损失，不恢复旧 Action-Memory/平行 runner，仅使用 GPU 0、1、2、3且不触碰4–7。

## 1. 当前目标和停止条件

当前先完成一个闭合的 Action-Forecast Writer 子任务：

1. 用唯一 canonical 新架构原位替换旧 Action-Memory Writer；旧活动代码、
   schema、配置和只服务旧架构的测试退役，历史由 Git 和已有结果保存。
2. `frame_stride=5`固定；在 GPU 0–3 上只实测并封存训练的每 rank
   action-query batch、`frame_microbatch_size`，以及评测的最佳并发参数。
3. Belief-v3从fresh identity一次连续训练`0→600`，每75 steps保存
   checkpoint但不中途主动评测；600完成后统一评测多个validation checkpoints，
   并先对step600做低成本内部顺序特异性诊断。
4. 找到 AS-Writer 的 validation 最佳 checkpoint，并在其后观察到幅度非常
   明显、复测后仍成立的validation下降趋势。多个后续checkpoint仅略低于best
   绝对不算饱和。主要门槛是不要明显落后于四卡rank-128 Source-SFT
   observed-best `108/400`；超过旧八卡全局incumbent `122/400`是stretch目标。
5. step600先做shuffled/reversed内部数值诊断，逐层检查forecast residual、
   Revision、Temporal memory、query content和effective LoRA；只有最终输出
   差异明确、跨多个tasks/videos稳定，才做昂贵的closed-loop paired arms。
   通过后再按绝对性能和后续validation曲线选择observed-best，并对最终best补齐
   correct-video、cross-suite wrong-video及顺序机制证据。
6. 若 AS 同时通过性能与视频/顺序特异性门槛，才推进 cold-start RL-Writer；
   若经过最小、证据驱动的架构/训练修正仍过不了，保留完整证据并停止汇报。
7. RL-Writer 先做一个独立、短 AS cold start，直到 24 个 source/train tasks
   每个在 official random-reset reward rollout 中至少成功一次，再切换为纯
   reward 更新。它不是从完整 AS-Writer best checkpoint 继续训练。
8. RL 阶段先在 train tasks 上训练到曲线平台，再持续评测合适validation
   checkpoints，直到定位observed-best并看到幅度非常明显且复测稳健的峰后
   validation下降；train平台本身不构成停止条件。

本子任务完成前不要自动继续 final-32、test task-local RL、joint target-action
oracle 或 ViVLA。长期 EMBER Goal 仍存在，但本轮交接的边界是上述 AS/RL
Writer 子任务。

## 2. 不可改变的科学合同

- 输入始终是正确 task language 加恰好一条 action-hidden teacher video。
- Writer 输出 sealed π0.5 LoRA 空间中的完整 rank-16 task-specific LoRA。
- frozen source base、24 development-train tasks、独立 video/action episode、
  source normalization 和 functional policy contract不变。
- Writer 不得接收 teacher action、proprio、reward、terminal、task ID、
  filename 或隐藏 normalization。AS action 只进入 frozen policy 的 functional
  behavior loss，不进入 Writer。
- policy 在 AS 训练、correct/wrong/shuffled 评测的所有主分支中都接收正确的
  evaluation/action-query language；替换的只有 Writer 视频。
- 初始 AS 只用 normal positive functional action loss，不先加 contrast loss。
- 不使用 `pi05_libero`、MemLLM、bank、geometry、shared update subspace、
  residual escape、额外 shared trainable adapter 或未 merge 的 source LoRA。
- RL rollout 与 checkpoint selection 只用 LIBERO official BDDL/random reset；
  fixed `.pruned_init` 只用于与 RL 数据隔离的 fresh evaluation。
- 四卡就是四个 native DDP ranks；不使用 gradient accumulation 模拟八卡。
- 只有一个实际 `action_query_batch_size_per_rank`。不得再暴露一个与它不同的
  `functional_policy_microbatch_size`。`frame_microbatch_size` 是同一视频内
  T 帧穿过图文/action-forecast 路径时的内存切片，不改变 optimizer batch。

## 3. 唯一 canonical Action-Forecast Writer

### 3.1 变长视频与图文理解

- 以固定物理/控制时间间隔采样视频，而不是把所有视频均匀压成固定帧数。
- 视频长度 `T` 保持可变；batch 内只为张量化做局部 padding，并始终携带
  padding mask 和真实原始 frame index。
- 首轮必须比较 stride 5 与 stride 10。LIBERO 约 20 Hz 时分别对应约
  0.25 s 与 0.5 s。stride 5 的 24-task 数据统计为：平均 35.6 帧、
  中位 30、p95 70、最大 105。
- 每个采样帧和完整 task language 进入 π0.5 的视觉语言 backbone。语言必须
  使用 frozen PaliGemma 经过上下文处理后的表示，不能只取 embedding table。
- 当前 sealed teacher input 是`RawTeacherVideoStore`读取的单条
  `obs/agentview_rgb`视频；不要静默加入wrist image、第二条视频或真实robot
  state。架构允许未来换成人类/第三视角视频，正因如此才需要imagined-state
  与VL Meta-LoRA，而不是假定teacher view等同于policy执行视角。
- frozen SigLIP/full projected image tokens 可以缓存；task token IDs、固定语言
  embedding、固定 flow noise、对齐索引也可缓存。因为 VL Meta-LoRA 和
  imagined-state path 会更新，contextual PaliGemma prefix 不能跨 optimizer
  step 缓存。

一个视频条件的端到端张量合同是：

```text
frames [T,3,H,W] + frame_indices [T] + task tokens [L]
  -> frozen SigLIP/projector full image tokens [T,N_img,2048]
  -> content-only state token decoder [T,28,2048]
  -> PaliGemma + VL Meta-LoRA contextual prefix/KV
  -> Action Expert + Action Meta-LoRA + 10 flow iterations
  -> final normalized plans [T,50,7]
  -> absolute-time Plan/Revision tokens [U,2,256]
  -> variable-time temporal memory [2U,256]
  -> 320 one-way LoRA queries [320,256]
  -> sealed complete rank-16 LoRA state
```

旧的16个Action-Memory tokens和18层memory-state tensor不进入新路径。新路径
中的learned tokens只有明确职责的visual state routing slots、no-revision/
type tokens和LoRA output queries。

### 3.2 Visual State Head 与 VL Meta-LoRA

- projected image tokens先从2048无bias投影到state width 128；28个learned
  slot identities经过两个content-only self/cross-attention + expansion-4
  FFN blocks，4 heads，直接输出28个2048-d virtual-state tokens。
- 每层维护routing `R`和content `Z`两个职责。`R`只进入attention Q/K，
  `Z_0=0`且只有image memory能首次写入content；self/cross attention的V、
  residual、FFN和最终projection均不读取`R`。零image memory必须产生严格零
  state-token content，这个结构不允许静态slot形成视频无关捷径。
- 这些virtual tokens直接插入原生state-prefix位置，全部梯度只来自最终AS
  functional loss。这里没有state label、scalar监督、离散化、`tanh`或Fourier
  features。
- π0.5并不存在独立的原生`state_proj`。正常LIBERO processor把8维state量化
  后写成文本`Task: ..., State: <8 integers>;\nAction:`。Writer必须保留同一
  语义布局：正常token embeddings负责BOS/task/`State:`/`Action:`分隔文本，
  用28个连续virtual tokens替换离散整数片段。一个保留的真实whitespace token
  加28个virtual slots使state区域长度为29；不要把virtual state放进Action
  suffix。
- PaliGemma 18 层 q/k/v/o 使用 Writer 内部 identity-init VL Meta-LoRA，
  首选 rank 4。作用是把 teacher-view 图像、正确语言和 virtual state 融合成
  source policy Action Expert 能理解的 agent-centric prefix KV。
- VL Meta-LoRA 只存在于 Writer 的教师视频理解路径；生成的公开 task LoRA
  执行时不携带它。
- rank 4/8并非要求两条Meta-LoRA表达力相同：PaliGemma width 2048使rank4已约
  0.922M参数，且其预训练图文语义应尽量少改；Action Expert承担teacher-view
  到robot action plan的主要适配，rank8约1.253M，因此给更高rank。只有真实
  profile或训练证据才能改变这两个rank，不能因“看起来不对称”自行拉齐。
- 两类Meta-LoRA都采用functional identity initialization：A用Kaiming初始化、
  B为零。28个state routing slots使用固定seed的`Normal(0,0.02)`初值。
  旧Action-Memory曾采用
  action-in manifold orthogonal initialization，但那些16个tokens已经退役，
  不得把该初始化连同旧tokens恢复到新Action Expert suffix。
- frozen SigLIP/projector可在`no_grad`下运行或读取cache；从virtual state、
  VL Meta-LoRA开始的PaliGemma contextual forward不能包在`torch.no_grad`
  中。base weights保持`requires_grad=False`，但梯度必须穿过冻结PaliGemma
  回到state head/VL Meta-LoRA，prefix KV也不能detach。

### 3.3 Action Expert forecast

- Action Expert 18 层 q/k/v/o 使用 Writer 内部 identity-init Action
  Meta-LoRA，首选 rank 8。它将上述 prefix 转换为 agent-centric action plan。
- 对每个采样帧执行完整 π0.5 flow inference：固定 `num_flow_steps=10`，
  action horizon 50。十次 flow 是同一个 18-layer Action Expert 的十次
  迭代，不是十层，也不是只预测十个动作。
- 同一 teacher video 的所有帧从完全相同的固定 `[50,32]` flow noise 开始，
  避免随机噪声被误当成 frame-to-frame revision。
- “固定”指一个video condition内部共用同一Gaussian noise，并由保存的
  Writer-flow RNG/visit schedule可恢复；不同训练visit可以获得新的可复现noise。
  correct/wrong/shuffled/reversed paired诊断必须显式复用同一noise。
- 每个 frame 的 PaliGemma prefix 只算一次；prefix KV 在十次 flow 迭代中
  原地复用。不要每个 flow step 重算视觉语言 prefix。
- 只保留最终 normalized `[T,50,7]` action plans。不要保存 10×18 层 hidden
  states，也不要把 imagined state 另行喂给 temporal encoder。
- 十次flow和frozen Action Expert base同样不能整体`no_grad`：base参数冻结，
  但梯度要穿过十次迭代回到Action Meta-LoRA、VL Meta-LoRA和state head。
  `output_hidden_states=False`，中间flow state只为ODE更新临时存在并通过
  checkpoint/rematerialization控制显存。
- pinned LeRobot的`PI05Pytorch.sample_actions`带有`@torch.no_grad()`，训练
  Writer时不能直接调用它。应在仓库内实现一个可微forecast wrapper：复用
  `embed_suffix`、mask/position构造、prefix-cache格式和原样10次
  `denoise_step` Euler更新，但用自定义含virtual-state/VL-Meta-LoRA的prefix
  构造；不得修改`.venv` site-packages，也不得另写近似flow公式。

### 3.4 同绝对时刻的 Plan / Revision tokens

令采样帧原始控制步为 `t_i`，该帧预测为 `P_i[k]`，`k=0..49`。一个预测对应
的绝对动作时刻是：

```text
u = t_i + k
```

不能只比较一对相邻 chunk，也不能把同一未来时刻的四十几个预测全部平均掉。

- `Plan_u` 使用在时刻 `u` 前最近的一帧产生的 receding-horizon 决策：
  `i*(u)=max{i | t_i <= u}`，`Plan_u=P_i*(u-t_i*)`。这等价于把每次最新计划
  的可执行前缀拼起来，表示“模型在这个时刻最终决定做什么”。
- 对所有覆盖同一 `u` 的连续 forecasts，构造有序 revision：
  `Delta_i(u)=P_{i+1}[u-t_{i+1}]-P_i[u-t_i]`。
- 一个共享 Revision encoder 读取 old action、new action、delta、两个 lead
  time和真实 `Delta t`形成有向content；revision count和稳定性统计只生成
  有限乘法gate，汇总成一个
  `Revision_u`。它表示随着新 teacher frame 到来，同一绝对未来动作被怎样
  修正。没有 revision 的边界使用 learned `no_revision` token 加 count/mask，
  不能用全零冒充稳定。
- 每个绝对时间点输出 `[Plan_u, Revision_u]` 两个 width-256 tokens，得到
  变长序列 `[U,2,256]`。两类 token 使用相同 absolute-time RoPE，再加
  token-type embedding。
- 具体编码保持简单且可批处理：
  - Plan encoder读取`[latest_action(7), normalized_lead_time(1)]`，用共享
    `8 -> 256 -> 256` MLP产生`Plan_u`；
  - 每条revision event读取
    `[old_action(7), new_action(7), delta(7), old_lead, new_lead, Delta_t]`
    共24维，用共享`24 -> 256 -> 256` MLP；
  - 同一`u`下数量可变的revision events由一个routing-only learned revision
    query通过一层pre-norm cross-attention + expansion-4 FFN单向聚合；query
    本身不进入输出residual；
  - directed content先独立RMSNorm。count和delta-norm mean/std/max均被压到
    bounded range，只产生`1+0.25*tanh(MLP(stats))`的逐通道乘法gate；
    gate范围严格为`[0.75,1.25]`，不得恢复additive stability支路；
  - 所有`(batch,u)`组合padding成一个并行batch处理，不用Python逐时刻循环；
    没有event时直接使用learned `no_revision`表示并保留count=0 mask。
- 对未来人类视频，核心仍是按物理时间对齐：视频帧间隔对应机器人控制周期
  中能执行多少 actions，而不是把 “5” 写死。当前 LIBERO 可以严格用原始
  frame index 对齐。

### 3.5 Temporal encoder 与单向 LoRA query decoder

- Temporal encoder：width 256、8 heads、2 blocks、真实 absolute-time RoPE；
  接受可变 `U` 和 padding mask，不固定视频长度。
- 使用 320 个 learned LoRA queries：
  - `18 layers × 16 rank slots = 288` 个 expert layer/rank queries；
  - 16 个 `action_in_proj` rank queries；
  - 16 个 `action_out_proj` rank queries。
- 一个 expert layer/rank query 同时服务 q-A/q-B/v-A/v-B factor heads；不要
  为四个 factor 各复制一套 query。
- query decoder 用两个 block，包含 query self-attention 和显式
  cross-attention `Q(query) -> K/V(procedural memory)`。这是单向读取：
  temporal/video memory 不反向读取 320 个 output queries。不要用无 mask 的
  拼接 self-attention 制造不必要的双向计算。
- 每个block显式拆分routing `R`与content `Z`。320个query都带明确的
  module/layer/rank routing identity：expert queries区分18个layer和16个rank
  slots；action-in/out queries区分module type和rank。`Z_0=0`；
  self-attention使用`Q/K=Norm(Z)+R,V=Norm(Z)`，query-to-memory
  cross-attention使用`Q=Norm(Z)+R,K/V=memory`，expansion-4 FFN和residual
  只更新`Z`。factor heads只读取最终`Norm(Z)`，不能读取raw routing identity。
- coordinate/query/type/no-revision embeddings都使用config seed下的确定性
  initialization并进入checkpoint；LoRA query table与identity embeddings默认
  `Normal(0,0.02)`。这些初始化不是额外shared adapter，且不得依赖task ID。
- factor heads 生成当前 sealed rank-16 PEFT tensors：18 层 q/v，加
  `action_in_proj`、`action_out_proj` 的 A/B factors。factor heads使用
  `bias=False`，final projection weight从零初始化；它们输出的delta加到真实
  identity template（A为既有Kaiming、B为零）上，使fresh public task LoRA
  严格functional identity且首步仍能通过B factor获得梯度。
- 每个factor head固定为`RMSNorm(256) -> Linear(256,256) -> GELU ->
  final projection`。expert layer/rank query同时送到：
  `q_A:1024`、`q_B:2048`、`v_A:1024`、`v_B:256`；action-in queries送到
  `A:32/B:1024`，action-out queries送到`A:1024/B:32`。B factors按真实PEFT
  tensor方向转置，最终state的name/shape必须逐项等于从真实identity template
  生成的`LoraTensorSpec`，不能手写猜名字。
- 不增加独立公共 LoRA 支路。这里不是全局 `bias=False`；conditional modules
  的 bias 可以学习，但所有输出必须由当前 language/video procedural memory
  经 query decoder 产生。

### 3.6 参数预算

目标是让 Writer trainable parameter count 与 rank-128 Source-SFT 的
`10,297,344` 接近，不能再用 10× 参数量解释优势。Belief-v3真实参数量：

| 模块 | 目标参数量 |
|---|---:|
| content-only 28-slot state token decoder | 1,053,440 |
| VL Meta-LoRA，PaliGemma q/k/v/o，rank 4 | 921,600 |
| Action Meta-LoRA，Action Expert q/k/v/o，rank 8 | 1,253,376 |
| Plan-relative Belief encoder | 1,007,040 |
| zero-preserving Temporal encoder | 1,640,192 |
| 2-block content-only LoRA query decoder | 2,191,104 |
| Factor heads | 2,181,120 |
| 合计 | `10,247,872` |

正式实现必须从真实 model/config 计算参数量；允许在不改变上述信息流的前提下
微调 hidden widths，使总量接近 `10.297M`，并记录每个模块的真实 count。
Writer 生成的 public rank-16 task LoRA 本身仍为 `1,287,168` scalars。

## 4. 代码所有权与退役边界

保留一条 runner，不创建 `v4`/`new`/`experimental` 平行执行路径：

- `scripts/train_as_writer.py` 仍是唯一 AS 入口。
- `scripts/evaluate_pi05.py` 仍是唯一 π0.5 rollout 入口。
- `src/ember/writer/model.py` 仍拥有完整 LoRA schema/decoder。
- 将 `src/ember/writer/action_memory.py` 退役，由一个职责清楚的
  `action_forecast.py`（或同等单一 owner）替换。
- `src/ember/writer/temporal.py` 原位改为 Plan/Revision variable-time owner。
- `as_contract.py`、checkpoint schema、training/inference/evaluator 调用点和
  targeted tests 同步更新为一个 `action_forecast` schema。
- 旧 `configs/pi05_as_writer_action_memory_v1.json` 由新的 canonical
  `configs/pi05_as_writer_action_forecast_v3.json` 替换；旧活动配置和只验证
  Action-Memory internals 的测试删除。历史结果由 Git、`findings.md` 和
  `progress.md` 保存，不创建 in-tree archive。
- 先用 `rg` 建立 callers/import/checkpoint ownership map。完成后要求活动
  source/config/test 中不再有 Action-Memory schema/import；provenance 文档
  可以保留历史名称。

结构变化受 `code-architecture-gate` 约束：优先 replacement/deletion，入口
保持薄，新增文件按职责拆分，避免让现有 700–800 行 legacy 文件继续膨胀。

接手时先完整阅读这些实际 owner，而不是从历史 runner 猜接口：

- `src/ember/writer/action_memory.py`：待退役的frame-prefix/Meta-LoRA owner；
- `src/ember/writer/model.py`：sealed LoRA tensor specs、factor ownership和
  `CompleteLoRAWriter`；
- `src/ember/writer/temporal.py`：待替换的旧variable-length temporal owner；
- `src/ember/writer/as_contract.py`、`as_step.py`、`training.py`、
  `checkpoint.py`、`inference.py`：config、functional gradient、DDP、
  exact-resume与评测装载合同；
- `tests/test_writer_model.py`、`test_writer_training.py`、
  `test_writer_checkpoint.py`、`test_writer_functional.py`：需要保留的机械不变量
  与需要随旧架构退役的内部断言；
- `.venv/lib/python3.12/site-packages/lerobot/policies/pi05/modeling_pi05.py`：
  当前 pinned LeRobot π0.5真实实现。重点核对`PI05Pytorch.embed_prefix`、
  `sample_actions`、`denoise_step`、`paligemma_with_expert`和
  `gemma_expert`；该文件只读，所有适配留在EMBER owner中。

## 5. 训练实现与显存/吞吐优化

- action-query policy batch 与 frame microbatch 是两个正交维度：
  `action_query_batch_size_per_rank` 是一个 task/video adapter 下的独立
  functional action queries；`frame_microbatch_size` 只是同一视频内部逐帧
  forecast 的切片。二者名字要直接表达这个区别。
- 不保留第二个 policy microbatch 参数；若显存只容纳每 rank 8 queries，
  logical/physical batch 都是 8。
- 先使用 BF16、fused SDPA/FlashAttention、静态安装的 Meta-LoRA、prefix KV
  reuse、固定十步 flow loop、`output_hidden_states=False`。
- 对 frame forecast 大块使用 activation checkpoint/rematerialization；只保存
  最终 action plans 和构建 Plan/Revision 所需的最小张量。
- 当前 functional LoRA leaf-gradient 机制可保留。若 Writer graph 与 frozen
  policy functional forward 同时驻留仍 OOM，可使用 exact two-pass replay：
  第一遍生成 detached adapter leaf 并求 policy 对 adapter 的梯度，释放 policy
  graph；第二遍重放可微 Writer，用 VJP 回传。它增加计算但不能改变 loss、
  sample/RNG 或 optimizer step 定义。
- 固定 checkpoint 的 evaluation 应按 checkpoint/task/language/video hash
  缓存生成的 public LoRA；复用已实现的 `per_sample_lora_batched_replan`，
  不退回逐 rollout materialize + sequential replan。
- 旧真实profile是一个Action-Expert pass、16 memory positions、batch16/rank时
  steady约`1.95–2.34s/step`且allocated约76GB。新架构十次50-position flow
  明显更重；交接前的工程估算是典型优化后约`8–18s/step`，若需要two-pass
  Writer replay约`12–25s/step`，p95长视频可能`16–30s/step`。这些只用于安排
  首轮smoke，不能写成结果，必须由真实median/p95 profile替换。

## 6. Profile 与封存顺序

训练profile已完成并封存，不应在正式0→600前重复：

1. shape、padding、绝对时间对齐、no-revision boundary、identity-init、
   frozen base、Meta-LoRA gradients和checkpoint exact-resume均已通过。
2. `frame_stride=5`固定；最终选择`frame_microbatch_size=32`和每rank
   action-query batch20，不做gradient accumulation。
3. 同一Belief-v3 tensor拓扑的12-step profile稳态中位为`6.49s/step`、
   `12.32 global queries/s`；frame-microbatch40更慢，48在首步前达到
   `81,153/81,920 MiB`且无法稳定前进。
4. 最终无`tau`的raw-RMS Revision实现另行通过fresh step1和step1→2
   exact-resume；resumed step为`6.918s`、`11.563 global queries/s`，
   峰值allocated/reserved为`77,090,931,200/83,730,890,752` bytes。
5. 不再profile stride10、frame/action batch或未充分训练的specificity；
   下一步直接fresh连续训练0→600。
6. 评测基于当前 4 replicas/GPU、8 envs/replica 的稳定点，实测邻近组合和
   adapter 预生成/cache。旧 6 replicas/GPU 在 Writer 视频编码阶段 OOM；
   只有新路径通过真实 profile 才能采用，不能凭空宣称更快。
7. 以真实 rollouts/s 和完整 400-rollout wall 选评测配置；显存利用率不是
   主要指标。
8. long-horizon拖尾必须按实际可用GPU数处理，而不是固定切八份：先把Long
   tasks依据`states × horizon`切成至少覆盖每个device的cost-balanced shards，
   让各device尽早承担一份Long工作；完成后所有workers从同一动态队列继续接
   普通task shards并work-steal。原始rows仍按task/init恢复聚合。

唯一训练选择已经写入canonical config和docs；评测并发仍按真实rollouts/s
另行选择。不得重新引入profile-only训练开关。

## 7. AS 连续训练与 checkpoint 选择

- 只用 GPU 0,1,2,3，一卡一个同角色 DDP rank。GPU0 不承担额外 model
  server/controller。
- 第一趟从fresh identity连续运行step0→600，每75 steps保存完整checkpoint；
  checkpoint保存不是暂停训练或插入评测的理由。
- step600完成后统一选择多个checkpoint做完整validation，并直接对step600做
  视频顺序特异性诊断。若特异性明确但性能曲线尚未充分探索，再按600-step或
  证据支持的更大跨度exact-resume继续。
- val functional loss只作很弱的辅助线索。最终 best 由相同 paired
  `8 tasks × 50 fixed states` closed-loop success决定；报告 per-task counts、
  paired flips 和重复评测的不确定性，不能把一个 noisy 400-rollout点写成
  精确上限。
- 若相邻候选在同一400-rollout panel上接近、paired flips不能支持方向，不能
  凭aggregate差几次就宣称峰值；优先补中间checkpoint。仍无法区分时，给两个
  候选增加一个预先封存的独立evaluation seed/panel复测，再做选择，并把重复
  测量波动本身报告出来。
- AS只有在validation observed-best之后出现幅度非常明显的性能下降，并且在
  复测后仍然成立，才算把最佳点和饱和区间找全。下降必须明显超过400-rollout
  正常波动，aggregate上肉眼清楚，并由多个tasks共同贡献，不能只是一个task
  掉点。多个后续checkpoint仅略低、paired统计刚好可区分、loss平台、一个较差
  checkpoint或success持平都绝对不够；没有明显下降就继续增加大步长训练。
- 不因为 GPU 数变化机械缩放 step；steps、global queries、task/video
  conditions、wall/GPU-hours都同时记录。
- 每次先启动可运行训练/评测，再在不修改其 import/config/output contract 的
  前提下并行更新文档、parser、下一阶段代码。
- step600先做shuffled/reversed内部数值特异性；只有effective LoRA等最终输出
  已呈现明确且跨多个tasks/videos稳定的差异，才做对应paired rollout。
  最终observed-best再补齐correct/wrong/shuffled/reversed。correct/wrong rollout保持
  task/init/policy/video seeds配对；shuffled/reversed保留完全相同帧集合，
  只变顺序。

当前旧架构参考不是新模型的初始化：

- Action-Memory temporal-RoPE best：step 400，correct `108/400`。
- 同 checkpoint 的 cross-suite wrong video 会明显改变 adapter，但倒序/乱序
  的 effective-LoRA relative L2 中位数仅 `0.00937/0.00699`，说明近似
  bag-of-states。
- 四卡rank-128 SFT的完整step100–1100曲线为
  `81/95/68/78/94/99/108/97/95/104/94`，observed-best是step700的
  `108/400`。step800/900/1000/1100虽非严格单调，但全部低于108，已经用多个
  峰后点括住该四卡best。因此AS的必须比较口径是“不明显落后于108/400”。
- 上述SFT历史判断不定义Writer停止标准。owner明确要求Writer必须出现幅度
  非常明显的validation下降；不能把“多个点都稍低于best”的SFT口径套给AS或RL。
- `122/400`不是四卡成绩，而是旧八卡rank-128 SFT step400的全局incumbent；
  新Writer超过122是stretch目标，不能在文档或论文中误写成四卡baseline。
- `122/400`的已封存结果root为
  `/data/ymdai/outputs/ember/pi05_source_sft_rank128_val8x50_step0400_77ec0ae_g67_r5_20260723`
  （逐task`15/2/4/29/42/30/0/0`，合计122）。当前focused task不重新训练
  Source-SFT；除非artifact校验失败或owner另行要求，不要把GPU时间转回SFT。

## 8. RL-Writer 接续合同

只有 AS 性能和视频/顺序特异性都通过后执行：

1. 新建独立 RL-Writer run，从新架构的规定 identity initialization开始。
2. 做短、task-balanced AS cold start；不是加载完整 AS best。持续做官方
   random-reset reward screen，直到 24 个 train tasks 每个至少有一次真实
   success。记录每 task first-success step、teacher action queries 和 wall。
3. 达到全 task coverage 的 checkpoint 后，冻结 cold-start action数据入口，
   转为纯 official env reward 更新 Writer。
4. RL rollout 保存 env/policy/worker RNG、seed schedule、interaction cursor、
   per-task reward rows、optimizer/scheduler和exact-resume状态。
5. 在 train tasks 上按多个任务覆盖和aggregate曲线判断平台，避免只由一两个
   易任务支撑。train平台后对少量相邻checkpoint运行完整validation，但必须
   继续到validation best之后出现幅度非常明显、多个tasks共同贡献且复测稳健
   的下降趋势。
6. 评测 correct-video，并在 selected best 上做一次 cross-suite wrong-video；
   不读取 validation actions，不用 fixed states训练或选 RL checkpoint。

RL和AS采用同一严格停止语义：必须报告validation observed-best及其两侧证据；
“train reward平台”“validation loss平台”“一个后续checkpoint变差”或“多个
后续checkpoint略低”都不能触发停止。下降不必逐点单调，但幅度必须非常明显，
明显超过rollout噪声并由多个tasks贡献，且用独立evaluation seed/panel复测后
仍成立。

现有 `train_rl_writer.py` 和 `src/ember/pi05_rl_writer*` 只能复用通用
RNG/checkpoint/env-pool机制；任何绑定旧 Action-Memory schema、旧冷启动含义
或旧数据墙的部分必须适配/替换，不能原样恢复。

### 8.1 明确留给实测、不是交接遗漏的变量

以下内容仍需按本文件的证据规则实测，而不是猜：

- 是否值得构建full-token cache及其精度；
- evaluation replicas/env batch的最终组合；
- normal AS最优checkpoint位于哪个step、非常明显的峰后下降从何时开始，以及
  是否需要比第二/第四点更密评测；
- 若AS不过关，下一次最小修正的具体内容；
- RL reward estimator/optimizer的最终高效配置；其validation停止门槛不是待定
  项，仍必须找到best并观察幅度非常明显、复测稳健的下降趋势。

Source-SFT不在这份待定列表中：其参考结果已经封存，本focused task不重训它。
训练stride/批量也不在待定列表中：固定为stride5、frame-microbatch32、
batch20/rank；正式训练是连续0→600，不使用30分钟segment。
任何未固定变量都应先用最小真实profile/paired evidence决定，不能变成等待owner
确认普通实现细节的门槛。

## 9. 当前仓库与实物状态快照

以下是交接时只读快照；新 session 启动后必须重新核验：

- canonical checkout：`/data/ymdai/projects/EMBER`
- 未改实现的源码基线为`b78584ab05e7f639cf1c022fdf457b3a971d64e6`；
  首次handoff文档commit为`9beb0de4499e7b464f3107a6ab8a434dd52e9b81`。
  新session始终使用最新`origin/main`，不要把这两个provenance hash当成必须
  checkout的旧目标。
- source base run：
  `/data/ymdai/outputs/ember/pi05_source_base_v1_seed7_1k_e2cc238_20260722`
- source checkpoint：
  `/data/ymdai/outputs/ember/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000`
- tokenizer：
  `/data/ymdai/ember_data/openpi/paligemma_tokenizer.model`
- target data root：
  `/data/ymdai/ember_data/LIBERO-datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a`
- 现有17GB cache
  `/data/ymdai/outputs/ember/pi05_writer_feature_cache_v2_development32_raw_e4c19f9_b32_20260722`
  只存每帧`16×2048`粗空间pool，不是完整SigLIP/projected image tokens，不能
  冒充新架构所需full-token cache。stride5的24-task full-token BF16 cache约
  42GiB（int8/FP8约21GiB）；先实测online路径和量化误差，再决定是否构建，
  并纳入500GB peak预算。
- 最终交接审计时 `/data/ymdai` 占用 `278,464,562,859` bytes；`/data` 可用
  `3,083,979,264,000` bytes。个人硬上限仍为 500GB。
- 最终交接审计时 GPU 0–3 各 `0 MiB`，GPU 4–7 被另一用户进程使用约
  37GB/card。
  这些都是瞬时状态，launch前必须重查。
- 最终审计确认tmux `ember_as_bias_r4_s3200`只有空bash且无训练/eval子进程，
  随后已删除该空session；当前没有EMBER训练/eval进程。
- main之外的15个历史worktree均clean且无活跃写进程。不要因“看起来旧”删除；
  main仍干净且无并发写时直接使用main。只有实际重叠写、branch切换或活跃进程
  import同一checkout时才新建隔离worktree。
- 代码仍完整停在旧Action-Memory owner：存在`action_memory.py`及其旧config/
  tests，不存在半成品`action_forecast.py`或新active config。新session应按
  第4节原位完整替换，而不是尝试续写不明半成品。
- 交接runtime为Python `3.12.3`、PyTorch `2.11.0+cu128`、CUDA runtime
  `12.8`、driver `570.158.01`；新session在真实launch前仍须重查。

## 10. 启动顺序和证据要求

新 session 应：

1. 先 `get_goal`；若无 active Goal，使用本文件第0节原文创建一个
   无 token budget 的完整 Goal。
2. `git pull --ff-only origin main`，检查 status/HEAD/worktrees/活动进程。
3. 完整阅读根 `AGENTS.md` 规定的 authority 文档，再完整阅读本文件。
4. 读取 `code-architecture-gate`；建立 imports/callers/schema/checkpoint map。
5. 先实现最短可运行垂直切片并做 shape/gradient smoke；不要先花数小时只写
   文档或大规模cache。
6. GPU launch 前使用 live GPU/storage preflight；不得 kill/reset/干扰他人。
   owner指定使用0–3不等于授权终止别人进程。
7. expensive canonical training 前使用 formal launch contract，保存 exact
   command、commit、config/hash、topology、预计storage peak和output ownership。
8. meaningful milestone 更新 `task_plan.md`、`findings.md`、`progress.md`；
   不提交 dataset、weights、cache、checkpoint或凭据。
9. 每个可复现里程碑做 task-scoped diff/tests，commit并push main。
10. 只有本文件第1节全部完成才把该子任务 Goal 标记 complete；代码完成、
    smoke、loss下降或一个 validation点都不够。

所有校验都以推进效率为准：只保留会直接防止无效科学结果、信息墙泄漏、OOM、
错误冻结/LoRA schema或不可恢复checkpoint的最小检查。最短垂直路径通过
shape、gradient、identity/freeze和一次resume smoke后立即开始真实profile或
训练；不运行与当前改动无关的广泛全仓测试，不叠加重复launch ceremony，也不
为了整理文档延迟GPU工作。必要config/hash/rows/resume证据在meaningful
milestone随手封存，而不是变成启动前的额外门槛。

除新增权限、不可恢复数据、500GB cap、必须干扰他人进程或实质改变科学问题
外，不要为普通实现细节逐项停下来询问。一次smoke、训练segment或评测失败后
先定位工程/科学层次、做最小修正并继续；能启动GPU工作时先启动，再推进互不
污染的次要文档和后续代码。

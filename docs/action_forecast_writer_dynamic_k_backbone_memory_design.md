# Dynamic-K Backbone-Memory Rank-8 Writer

状态：2026-08-13 active architecture authority。本文只授权这一条canonical successor；历史v6、K4、
condition-residual和guard实现仍由Git与formal artifacts保存，不得并行恢复为另一条可训练Writer。

## 1. 决策

下一版从零训练一个shared Writer：输入exact task language和`K=1..4`条同task、action-hidden正确教学视频，
一次生成一套完整38-target rank-8 LoRA。K1与K>1使用同一网络和同一参数；policy rollout前只运行一次Writer，
rollout期间不再看视频。

它改变的主要科学接口不是给旧v6再加一个后处理层，而是把视频程序与LoRA写出的连接点前移到PI0.5自身：

```text
每个真实frame + exact language + 原生50个固定Action probe tokens
                         + 8个Writer memory tokens
→ 同一次PaliGemma/Action-Expert联合18层forward
→ 每层8个contextual memory states
→ 每条video独立的有向程序
→ 同task K条video的置换不变共识
→ layer × rank双轴通信
→ shared shape-family mapper
→ 一套完整38-target fresh rank-8 LoRA
```

`8`是LoRA rank坐标数，不是视频阶段数，也不是输出参数分块数。8个token只追加一次并贯穿18层；不是每层再
追加8个。layer/rank program slots和LoRA parameter rows都不称为memory token。

## 2. 针对的最早失效接口

历史已经分别做到视频可读、顺序可扰动、LoRA有能量、effective BA和fixed-action可变，但没有做到同一checkpoint
稳定共存：

- v6-fast给出最高`143/400`，证明task-grounded semantic + causal Procedure + task-complete recipe是强起点；
  其后`131/130/132/126`又证明能力在换手。
- 三条K4路线best都不超过108。它们并非没读视频；最早共同失败是full24 shared credit retention约
  `.043--.046`，接近`1/24`。
- Phase-Aligned K4先压到16个phase再同phase算术平均；Trace K4把极低能DCT高频逐项归一放大；Invariant K4
  使用固定四基、随机128投影和逐token unit norm。三者都不能代表few-shot本身。
- 同task不同video correction近正交；在单视频点上继续增加guard、condition或functional闭合没有改善held
  closed-loop。

因此本设计直接让同task多个demo在Writer表示学习阶段比较并形成共同程序，同时把layer/rank对应放进真实
policy上下文，减少`video representation → generic 320-slot compiler → LoRA`的压缩与错位。它不声称仅靠
memory或rank8必然解决task drift；paired400仍是裁决。

## 3. 输入、信息墙与dynamic K

输入只有：

- exact task language；
- 每个condition的`K∈{1,2,3,4}`条同task RGB teacher videos；
- 每条video先按stride5建立有序候选并保留末帧及真实sampled-frame ordinal；进入昂贵joint backbone前，
  每个condition固定总预算64帧，按`floor(64/K)`给每条video相同cap，并以包含首尾的确定性均匀索引取帧。
  这保留真实ordinal、视频内方向与跨video置换等变性，同时使K1--K4计算有明确上界。

不得读取teacher action、proprio/state、reward、terminal、task ID、filename、object pose、hidden
normalization或rollout outcome。AS训练中的actions只进入frozen source-policy functional loss；video与B20
action episodes先按task采样再取严格episode补集。

统一ragged接口：

```text
frames                    [N,3,H,W]
frame_indices             [N]
video_offsets             [V+1]       # frames → video
condition_video_offsets   [B+1]       # videos → condition
language_tokens/masks     [B,T]
```

不为`Kmax`或longest video实体padding frames，不设K专属branch、shot ordinal或best-video selector。

训练从step0真实混合K1--K4。对24个train tasks使用sealed task permutation`p(t)`，macro `m`取
`K(t,m)=1+((p(t)+m) mod 4)`：每macro各6个K1/K2/K3/K4，每task每4宏覆盖全部K，K不与task/suite绑定。
正式结果分别标注K1 one-shot和K>1 few-shot，不把few-shot分数冒充one-shot。

## 4. 真实backbone memory

### 4.1 正常有效上下文

每个frame继续使用真实224图像的256个image tokens、exact language tokens和现有固定Gaussian
`[50,32]` Action probe at `t=1`。Action Expert suffix在`action_in_proj`后为`[F,50,1024]`。

追加一组learned `M∈R^(8×1024)`到该suffix尾部：

```text
prefix: [real image tokens, exact language tokens]
suffix: [50 native Action probe tokens, 8 memory tokens]
```

attention block markers为：

```text
prefix: [0, ..., 0]
action: [1, 0, ..., 0]
memory: [1, 0, ..., 0]
```

由PI0.5 cumulative block mask得到：prefix只看prefix；action看prefix与完整action block、不看后置memory；
memory看prefix、action和整组memory。这样memory不会改变原50个probe的因果可见集合，同时能读取真实图文和
Action Expert交互。没有blank/zero image、没有memory-only forward、没有第二次baseline forward。

memory使用与`t=1` probe相同AdaRMS condition，因为它们处于同一个Action Expert suffix；它们不是虚构的
action query，也不进入action prediction输出。

### 4.2 逐层状态

保留Action Meta-LoRA rank4，使memory能学习怎样读取联合上下文；首版不训练Text/VL Meta-LoRA，避免在通用
VLM侧增加无证据的自由度。source policy的所有真实参数仍冻结。

Writer-owned joint bridge在每个Action Expert layer完成attention residual、post-attention norm、MLP residual后，
保留该层8个memory hidden；最终得到：

```text
Z_f ∈ R^(18 × 8 × 1024)
```

这要求把pinned LeRobot的18层联合loop封装为一个cohesive Writer module，公式逐项等价；不修改site-package，
不依赖脆弱hook，也不重复forward。最终原生50个probe hidden仍可作为v6兼容的interaction诊断，但LoRA动态
value不从其绝对均值旁路写出。

## 5. 单视频有向程序

language说明要关注什么，但不能单独提供LoRA value。对一条视频的逐帧grid `Z_0...Z_(F-1)`，先构造：

```text
D_0 = 0
D_f = Z_f - Z_(f-1), f>0
G_f = Z_f - Z_0
```

然后共享投影`1024→256`，对每个`layer × rank-token` cell沿真实frame ordinal运行2层causal temporal block。
Q/K可RMSNorm并使用真实ordinal RoPE；V与FFN residual保留物理幅度，不做逐频/逐cell unit normalization。
输出使用末帧causal state与终态goal residual的zero-preserving融合，形成：

```text
U_k ∈ R^(18 × 8 × 256)
```

重要性质：

- 所有dynamic values来自`D/G`，若video所有帧相同则精确为零；静态language或learned memory本身不能写LoRA；
- language和层/rank identity可以进入Q/K address，不能作为V/content旁路；
- shuffled/reversed必须先重排真实frames，再重算`D/G`和causal program；
- 时间token仍是真实stride5 frame states而非学习出的“阶段token”；A40实测完整K-set可超过16分钟/宏并触发
  NCCL heartbeat，因此joint backbone只处理上述固定预算内的真实帧，所有下游D/G仍使用其真实ordinal。

这继承v6中task-grounded address、adjacent visual transition、真实frame ordinal和causal Procedure；不机械复制
其separate text-only/VL前端，因为真实memory已经在联合backbone中进行task-conditioned视觉读取。

## 6. 跨video共同程序

每条video必须先独立完成第5节，绝不跨video计算transition。对同一condition的`U_1...U_K`，在每个固定
`layer × rank-token` cell上做两层无shot-position的set self-attention：K维输入先彼此比较，Q/K使用内容，V保持
各视频dynamic program。输出再做等权集合归约：

```text
U_shared[l,r] = (1/K) Σ_k SetBlock(U_1...U_K)[k,l,r]
```

这不是raw frame/feature平均：每条video先独立保序形成程序，set block先显式建模一致与分歧，最后的均值只是
把equivariant集合变成严格permutation-invariant的一个共享状态。不得content-pool到单条“最好video”，不得
平均分别生成的LoRA。

训练时对`K>=2`复用同一次已经计算的video programs，取一个sealed singleton subset走同一个set block，并加：

```text
L_consistency = SmoothL1(P_singleton, stopgrad(P_full_set))
```

它让一条video学会逼近多demo共同程序，不增加frame/backbone forward、不引入额外数据或第二套LoRA。
首版固定低权重`0.05`；主functional loss防止共同collapse。K1时该项为零。

## 7. layer/rank通信与LoRA写出

### 7.1 M2P grid

跨video后得到`[18,8,256]`。首尾各建立一组endpoint parameter-query rows：action-in从早层grid读取，
action-out从末层grid读取；它们是readout queries，不是backbone memory。于是得到20个policy groups：

```text
action-in, 18 Action Expert layers, action-out
× 8 rank coordinates
× 256
```

使用两个alternating blocks：

1. 固定rank，在20个policy groups之间做layer-axis attention；
2. 固定group，在8个rank coordinates之间做rank-axis attention。

layer/group/rank identities只进入Q/K；dynamic video program是V。没有frame、video、layer、parameter四个维度
到处交替attention：时间轴只在单视频encoder，video轴只在set aggregator，layer/rank轴只在M2P。

### 7.2 为什么是rank8和8个memory tokens

fresh rank8完整payload为643,584 scalars：

| target family | count | each A/B shape | total scalars |
| --- | ---: | --- | ---: |
| q_proj | 18 | `[8,1024]`, `[2048,8]` | 442,368 |
| v_proj | 18 | `[8,1024]`, `[256,8]` | 184,320 |
| action_in | 1 | `[8,32]`, `[1024,8]` | 8,448 |
| action_out | 1 | `[8,1024]`, `[32,8]` | 8,448 |

rank8把输出维数减半，并令每个memory token可解释为一个rank coordinate。它必须fresh训练，不能从rank16/14
压缩或resume。LoRA config采用`rank=8, alpha=8`，保持`alpha/r=1`。

SHINE式“payload容量完全等于memory容量”在本policy每层q+v需要34个1024-d tokens；完整flat payload需要629
tokens，端点还异构。34-token direct reshape是可信后备反事实，但首版不采用：它会让每个frame的正常50-token
suffix变84，真实context计算是8-token方案的显著倍数，而且34容易再次退化为parameter chunks。只有当第9节
证明有用差异在shared mapper后衰减，才允许单变量换到34-token direct。

### 7.3 shared shape-family mapper

每个`[group,rank,256]`状态先用shared bias-free `256→1024` projector，再由四个跨layer/rank共享的
shape-family mapper一次输出拼接的A row与B column。每个mapper使用一个共享hidden layer再分A/B输出支路；
A支路保留为后续可证伪接口但首版关闭，B支路严格zero-init：

```text
q:          1024 → 3072 = 1024(A) + 2048(B)
v:          1024 → 1280 = 1024(A) + 256(B)
action_in:  1024 → 1056 = 32(A)   + 1024(B)
action_out: 1024 → 1056 = 1024(A) + 32(B)
```

q/v mapper跨18层共享，所有mapper跨8 rank coordinates共享；不是38个独立wide heads。首版A精确保持确定性
随机template，只动态生成B；这不是历史已否定的“B-only residual”，因为这里是fresh rank8完整adapter的正常
LoRA gauge，而非给已有rank16 LoRA叠加小B correction。B readout严格zero-init，保证fresh step0/no-video
functional identity。初始第一步的**functional梯度**只有B支路获得；B非零后，functional梯度再传回shared
hidden、memory/temporal/set/M2P路径，和历史template-A/zero-B机制一致。K>1时低权重representation
consistency从第一步起可以训练singleton/set上游表示，但不能绕过zero-B直接改变policy；这是第6节一致性目标的
预期作用，不应误写为functional路径已经打开。

这里保留历史FactorHeads已证明的shared family ownership，但移除`320 generic slots→8个256-hidden two-layer heads`
的额外压缩。最早风险是同一shared family basis仍把不同task/layer方向压成共同更新；必须直接测mapper前后传递。

## 8. 训练与多task共存

首版只用正常顺序correct videos做主AS functional supervision，不加入shuffled/reversed margin或人工破坏negative
LoRA。correct顺序的作用来自有符号`D/G`与causal processing；controls用于验证其有用性。

每个macro覆盖24 train tasks：

```text
每task一组K1--K4 videos → 一套LoRA → B20跨episode action queries
task_loss = mean(B20 functional losses) + 0.05 * representation consistency
global_loss = equal mean over 24 tasks
→ one AdamW update
```

首版沿用v6-fast的AdamW/cosine、peak LR、B20和full24 raw equal mean，不同时加入新的gradient surgery/guard/RL，
避免把新representation的效果与credit optimizer混在一起。不同video共同程序和singleton-to-set consistency是
本轮减少same-task correction正交性的主要机制；task-complete等权与shared mapper负责让不同task在同一参数中
学习。若表示一致而仍换手，才把最早接口推进到optimizer/functional credit。

任意fresh world size 1--6：K和实际videos选定后，以`B20固定成本 + budget后总帧数`估task cost，在当次实际
world上LPT分配，允许world5为`5/5/5/5/4`，rank内long-first。每task backward系数为`world_size/24`，DDP均值
后严格得到24-task equal mean；rank内只在最后task同步。不padding/dummy task、不等待6卡。

checkpoint保存Writer、optimizer、scheduler、macro/K/action cursors、每rank/worker RNG和world topology；fresh
可选1--6卡，exact resume锁原world/topology。

## 9. 快速否决与真实裁决

CPU/单GPU机制门只回答实现是否值得训练：

1. memory与真实image/language/native probe处于同一次forward；原50 action probes不看后置memory；
2. step0与no-video完整LoRA functional identity；source policy零trainable params；
3. K1走同一ragged graph；K2--K4 video permutation不变；每video内部shuffle/reverse产生非零program差异；
4. constant-frame video的dynamic program为零；language-only不能写LoRA；
5. gradients到达Action Meta-LoRA、temporal、set、M2P和B mapper；
6. longest-video profile选择真实samples/s平台，不能靠重复forward、dtype扩展或防御性校验换数值一致。

formal训练每25 macro保存，最迟50/100/150/200及时做strict paired correct400；只有absolute、breadth、趋势和
传递共同支持才续。首个必须报告的最早接口链：

```text
per-layer memory grid
→ directed per-video program
→ cross-video shared program
→ M2P grid
→ family-head LoRA/effective BA
→ fixed-action response
→ strict closed-loop
```

快速否决：

- constant/static或language-only仍生成非identity LoRA：信息墙失败；
- correct/wrong/order在memory/program有差异，但M2P后消失：layer/rank compiler失败；
- M2P差异健康但family head后的effective BA/action显著衰减：8-token shared mapper失败，下一单变量才是34-token
  layer-direct readout；
- functional loss下降而paired400 absolute/breadth/churn不改善：shared AS credit仍错位，不能靠rank/scale小扫；
- K>1提高内部一致性但K1/K>1 closed-loop均低于v6：高absolute子机制在memory frontend丢失，优先检查
  task-grounded visual address与Procedure幅度，而非继续堆set层。

正式选择只认同一个single checkpoint。必须与v6-fast143、old134/compiler138/online128以及最接近K4逐task
比较；报告K、aggregate、per-task/per-suite、breadth、retained/gained/lost、checkpoint churn、same-task-other、
wrong/shuffled/reversed/no-video。目标保持严格`>150/400`，达到后继续提高。

部署吞吐使用同构budget64 profile checkpoint做单A40真实K1 generation profile。validation 8 tasks各4个
state形成固定32-request longest-first panel，B8/B16/B32各warmup1、measured2；三者全部stable且无OOM，
LoRA/s分别为`.97433/.96463/.96598`，峰值reserved约`13.38/13.38/13.40GB`，因此按预注册最高实测吞吐
选择B8。正式评测必须精确使用B8，不能把minimum batch当作任意更大batch；profile root为
`runs/outputs/pi05_dynamic_k_writer_generation_profile_val8x4_correct_gpu02p0_6288fbb_20260813`。该证据只选择
部署吞吐，不构成closed-loop或科学性能证据。

## 10. 工程所有权与参考

- `CompleteLoRAWriter`保留外部ragged batch→76 LoRA tensors API、tensor specs、identity template和安装逻辑；
  不新增V7/legacy并行Writer。
- 新cohesive owner为`writer/backbone_memory.py`与`writer/memory_program.py`；替换旧
  `video_program→Core/Procedure→320-slot compiler`职责，而不是继续增长已有大文件。
- 恢复精简的canonical end-to-end `train_as_writer.py`；不在已退役`v6_prior_training.py`加successor mode。
- 新路径机制验证后移除旧可变更的condition-residual训练入口；历史结果只由Git/artifacts/raw rows保存。
- 评测保留当前dynamic queue、persistent workers和batched cache，不恢复旧静态GPU分配。

外部机制参考只支持设计原则，不替代EMBER实验：

- SHINE：真实context中append memory，同一批tokens逐层更新，layer/token双轴M2P并结构化映射参数；
  https://arxiv.org/abs/2602.06358 ，https://github.com/MuLabPKU/SHINE
- Doc-to-LoRA：每rank latent加layer/module-aware shared heads，支持本设计的rank-token/shape-family mapper；
  https://github.com/SakanaAI/Doc-to-LoRA

## 11. Formal macro50 terminal verdict

clean`5319022` world6 fresh`0→50`已完整结束，训练root为
`runs/outputs/pi05_dynamic_k_backbone_memory_rank8_budget64_formal_fresh0to50_r6_b20_micro8_5319022_20260813`。
functional loss首/末5宏`.15307→.12095`，23/24 train tasks为负斜率，说明优化执行图稳定。

macro50 single-checkpoint strict paired correct400 root为
`runs/outputs/pi05_dynamic_k_backbone_memory_rank8_correct400_noreplacement_seed7_macro0050_b541785_gpu02_20260813`：
`100/400`、breadth4、per-task=`0/0/42/18/0/36/4/0`、per-suite=`0/60/36/4`。相对old134严格配对为
`82 retained/18 gained/52 lost`，相对compiler138为`81/19/57`，相对online128为`80/20/48`；相对
v6-fast143低43且breadth少2。该结果否决当前完整架构与recipe的absolute价值，禁止exact-resume`50→100`。

effective-BA诊断显示candidate总norm均值`135.64`，因此不是近identity；但action-target norm均值仅`.513`，
stable rank约`1.00`，相对old134 effective cosine仅约`.01--.02`，八个task mean BA方向的off-diagonal cosine
均值`.702`。同task视频方差明显高于old134，说明Writer确实读到video，却把差异编译到错误且跨task高度共同的
policy geometry。

最早代码级失配出现在mapper之前：backbone输出的absolute `task_hidden`与`probe_hidden`没有任何消费者，
`layer_memory`又在`_encode_video`入口被严格转换为相邻差分与终点差分。因而架构保留“怎么变化”的Procedure，
却在有向编码前删除了“哪些对象、关系与目标”的Semantic Core。下一authority只恢复absolute memory作为temporal
Query address；D/G继续作为唯一Key/Value content。本设计由Git和上述formal artifacts封存，不再是active实现。

# V6 Layerwise Action-Probe Conditioned Procedure Reader

状态：2026-08-14 design authority；canonical CPU实现与机制测试完成，尚未启动GPU。简称只用于文件/配置
identity：`V6-LPCP`。

## 1. Decision

下一轮不原样恢复已经只有`91--102/400`的Dynamic-K memory Writer，也不把memory token写成必须满足的形式。
保留当前最强且已经证明policy-effective的V6 Shared-Core Ordered-Procedure K4 AS checkpoint（`139/400`）及其
完整rank-16 factor compiler，只增加一个**零初始化、逐Action-Expert层与LoRA rank对齐的Procedure query
conditioner**。

它读取同一次V6真实frame forward中已有的50个native Action probes在18个Action Expert层后的状态；不追加
memory token、不重复backbone forward、不改LoRA rank、不改factor heads、不改B20 objective。若该最小分层读取
不能改善closed-loop，则先定位native probe evidence、temporal conditioner或Procedure readout中最早失效处；只有
证据显示native probes本身缺少可写的分层载体时，才允许在完全相同的下游接口把它替换为真实backbone memory
tokens。

本轮检验的单一主要假设是：

> V6把每帧Action Expert证据只在最终层对50个probe取mean，再用generic Core/Procedure路由生成全部policy
> slots，丢失了“哪一层、哪一个rank应读取视频过程中的哪部分”这一条件；从同一次真实context forward保留
> layer/rank-aligned probe states，并只用它们调制原生有向Procedure reader的Query，能在不破坏V6健康LoRA
> geometry的前提下形成更task-specific、可共同积累的更新。

## 2. Why this interface follows the evidence

- v6-fast达到`143/400`，当前shared-Core K4底座达到`139/400`，证明V6 Core、directed Procedure、native
  compiler、完整动态A/B与rank-16 factor heads共同处于policy-effective区域；下一轮不应替换这些强路径。
- Dynamic-K memory系列证明真实图文prefix下的8个memory tokens、dynamic K和layer/rank M2P能机械工作，但
  `100/101/102/98/96/91`说明整套替换V6前端与factor generation会丢失absolute能力。
- ADSP把22条train24 success-support约束从6个violation投影到0，仍只有`138/400`并产生45条paired churn；
  因此不再继续optimizer constraint小修。局部support、LoRA距离和held closed-loop已经明确错位。
- 当前V6虽有320个policy/rank routing slots，但每帧Action probe evidence只来自最后一层50-token mean；
  layer identity是learned routing identity，不是从对应Action Expert层读到的contextual evidence。这是仍未被当前
  强V6路线检验的结构接口。
- SHINE表明可从真实context中的全层hidden states进行layer-aligned parameter generation；Doc-to-LoRA进一步
  表明不一定需要显式memory tokens，也可以直接读取冻结target backbone的逐层context activations，再用共享
  rank queries生成LoRA。这里只继承分层读取与共享映射原则，不引入其额外语料或训练阶段。

Primary references：

- SHINE: https://arxiv.org/abs/2602.06358 ，https://github.com/MuLabPKU/SHINE
- Doc-to-LoRA: https://arxiv.org/abs/2602.15902 ，https://github.com/SakanaAI/doc-to-lora

## 3. Complete deployment data flow

```text
exact language + K=1..4 same-task action-hidden ordered videos (stride 5)
  -> existing V6 per-frame real image + exact language + 50 fixed Action probes
  -> same single PI0.5 joint backbone forward
       ├─ existing final evidence -> Semantic Core + visual transition + causal Procedure
       └─ new tap: 18 layers x 50 native probe hidden states
  -> shared rank-query reader -> per-frame 18 x 16 layer/rank probe states
  -> adjacent frame deltas + shared causal temporal encoder, independently per video
  -> per-video 320 policy-slot query conditioners
  -> zero-init projection added only to native Procedure-reader Query
  -> existing per-video ordered Procedure readout
  -> frozen trained K-set common-Value aggregation
  -> frozen native AdaLN/post-fusion/factor heads
  -> one complete 38-target rank-16 task-conditioned LoRA
```

Writer仍只在rollout前运行一次；policy rollout期间不再观看视频。teacher actions、proprio、reward、terminal、
task ID、filename、pose或hidden normalization均不进入Writer。

## 4. Layerwise native Action-probe readout

### 4.1 Carrier

现有V6每帧已经构造真实image/language prefix和50个固定noise/time Action probes，并在同一次18层joint forward
中运行。新路径只旁读每个Action Expert block完成后的50个suffix states：

```text
H[f, l] in R^(50 x 1024), l=0..17
```

不运行zero image、blank prefix、memory-only Action Expert，也不增加第二次forward。V6原final hidden、Core和
Procedure数值路径不被新readout替换。

warm-start机制实验中V6底座完全冻结，`H`对底座detach；trainable readout在detach之后运行。这与Doc-to-LoRA
读取冻结target backbone activations的边界一致，也使本轮只检验新增接口，不让Meta-LoRA重训掩盖结论。

### 4.2 Rank-query reader

对每一层，16个learned rank queries共享同一组Q/K/V/O参数，并加入layer identity与rank identity。它们对该层
50个native probe states做cross-attention，得到：

```text
R[f, l, r] in R^256, r=0..15
```

16是当前保留的public LoRA rank坐标，不是memory-token数量，也不是视频阶段数。50轴是PI0.5原生action horizon，
video时间轴仍由stride-5真实frame序列单独处理。reader参数跨18层和16个rank共享，避免38套独立wide heads。

本轮不增加SHINE式额外layer/token M2P：V6的320-slot routing和全slot post-fusion block已经负责policy-layer/rank
通信，旧memory路线另加M2P并未带来absolute收益。新增reader只负责把真实分层evidence对齐到已有slot topology。

## 5. Directed per-video conditioner

对每条video内部按真实frame ordinal构造：

```text
D[0,l,r] = 0
D[t,l,r] = R[t,l,r] - R[t-1,l,r]
```

先在`(l,r)`轴求mean，沿video时间轴只运行一次共享、带真实sampled-frame ordinal RoPE的causal temporal block；
最后一个valid状态作为该video的有向过程context，再按每个`(l,r)` delta与该context的匹配度沿frame轴加权汇聚，
形成该video的expert-layer conditioners。这样layer/rank Value保持独立，但不会为288个slots重复运行重型时序网络。
action-in的16个slots读取layer0结果，action-out的16个slots
读取layer17结果；已有native routing继续区分expert、action-in与action-out ownership，不给新增动态Value混入
静态endpoint identity，最终得到：

```text
C_video in R^(320 x 256)
```

所有Value来自frame-to-frame change；constant-frame输入使新增conditioner严格为零。语言与静态外观可以影响
rank-query的解释方式，但不能单独给新增路径提供非零动态Value。causal block在video内部运行，绝不把不同video
拼成一条虚假物理序列。

## 6. Injection point and exact identity

对每条video，原生V6在shared Core条件下形成`P_video`。新增bias-free projection `W_delta`严格zero-init：

```text
Q_delta = W_delta(C_video)
ProcedureQuery = native_routing + normalized_shared_core + Q_delta
```

Procedure的K来自原生ordered Procedure memory，V仍是原生centered directed Procedure Value。新路径只改变每个
policy layer/rank从同一视频过程读取什么，不直接生成A/B、不绕过Procedure Value，也不建立第二套LoRA。

之后沿用AS139 checkpoint中已经训练好的、冻结的`PolicyProcedureCommonValueFusion`聚合K条per-video
Procedure readouts，再运行冻结native fusion和factor heads一次。

`W_delta=0`时，K1--K4的Program及全部76个LoRA tensors必须逐tensor等于AS139 deployment graph；不仅step0，
任何时候显式zero-conditioner反事实都能恢复同一底座。训练不能更新或偷偷复制底座参数。

## 7. Dynamic K and video causality

- 每macro保持K1/K2/K3/K4各6个train24 tasks，四个macro内每task轮到全部K；不能只训练K1/K4却宣称动态K。
- 每条video独立通过完整有序路径；K轴仍只由冻结的permutation-invariant V6 set聚合，不平均frames、final
  features或LoRAs，不挑最好video。
- shuffled/reversed必须在真实输入frame层重排后完整forward。它们同时改变layerwise probe deltas和原生causal
  Procedure；set无法恢复被破坏的单video过程。
- exact language仍是Core与probe context，视频变化是新增conditioner与Procedure Value；二者缺一不可。

## 8. Training contract

首轮是warm-start架构机制实验，不冒充最终fresh论文recipe：

- 从sealed AS139 Writer weights开始，完整V6、trained K-set、compiler和factor heads全部冻结；
- 新建rank-query reader、layer/rank/endpoint identities、causal conditioner与zero-init query projection；
- optimizer/scheduler fresh，不载入AS139 optimizer；
- 训练仍用train24 task-complete、每task一套LoRA、B20同task跨episodeaction queries、task内mean后24-task等权；
- K1--K4按上节均衡，video与action query继续跨episode；
- 不加reward、negative、expert target、consistency、reconstruction、norm/rank、orthogonality、scale gate或额外数据；
- public LoRA保持rank16。fresh rank8是否值得做是后续独立变量，不能与本轮layerwise interface混在一起。

若该接口通过strict门，才写独立authority把它整合进完整Writer并从random Writer initialization fresh训练，证明
结果不长期依赖AS139 checkpoint。warm-start失败则不靠“最终应fresh”替它辩护。

## 9. Mechanism and efficiency gates

实现后只做能裁决真实合同的检查：

1. 同一次joint forward确实得到18层native probe states；forward count与V6完全相同；
2. zero-conditioner下K1--K4 Program/76 tensors等于AS139；base与K-set参数0 gradient；
3. rank reader、causal block和`W_delta`在首两步内获得finite nonzero gradient；
4. constant-frame新增conditioner严格零，video内reverse/shuffle产生非零`C_video`与Procedure readout变化；
5. K2--K4输入video置换只允许正常BF16 reduction低位差异；
6. full24最长video、K1--K4均衡、B20真实profile无OOM/nonfinite，按最高实测samples/s选择batch/world size；
7. 不为逐元素低位一致做batch1、重复forward、扩dtype、逐tensor扫描或内容hash。

内部还报告full24 task-mean conditioner/gradient off-diagonal cosine、`Q_delta -> Procedure slots -> effective BA ->
fixed action`传递，用于定位collapse；它们不能替代closed-loop选择。

## 10. Formal and falsification gates

从clean pushed commit和detached frozen worktree训练。先到macro25并立即做single-checkpoint K4 strict paired400；
只有趋势与机制共同合理才续到macro50，不做LR/rank/width/token/seed sweep。

- `<144`、breadth`<7`、相对AS139 gained不超过lost，或lost`>10`：本轮终局non-pass；
- `144..150`且breadth至少7、lost不超过10、gained>lost、至少三suite不降：允许一次exact 25->50；
- `>150`：先封存single checkpoint，再补K1--K4 dose与correct/same/wrong/shuffled/reversed/no-video严格配对；
- correct不实质优于controls时，即使absolute过150也不能写成有效教学视频学习。

快速否决与下一步边界：

- native layer probes在correct/reverse/static之间已无material差异：证据carrier最早失败；此时才有理由在**相同
  rank reader、temporal、query injection、rank16 compiler和training**下把carrier单独换成真实memory tokens；
- probes与conditioner有差异，但`Q_delta`后衰减：readout/injection失败，不加memory；
- Procedure/effective BA/action均material变化而strict仍换手或下降：functional credit/held occupancy仍失败，
  literal memory不会自动解决，转向训练credit或shared coexistence的架构级设计；
- strict改善但视频controls失败：说明新增路径仍在利用task/static shortcut，不能靠negative margin修饰结果。

本设计不否定memory token。它把memory的价值变成一个有明确触发条件的后继反事实，而不是在V6、rank、decoder、
front-end同时变化时无法解释的架构标签。

## 11. Canonical ownership

- `writer/model.py`唯一拥有AS139 base、layerwise conditioner接线、shared-Core/per-video Procedure与LoRA输出；
- `writer/video_program.py`拥有同一次joint-forward的layer probe tap/readout，不复制或重复policy forward；
- `writer/temporal.py`拥有共享causal conditioner与conditioned Procedure query接口；
- `writer/training.py`与现有AS task-complete runtime继续拥有K-balanced full24训练；
- config/checkpoint/evaluator schema fresh-incompatible，退役ADSP/reward deployment只由Git与formal artifacts保存；
- 不保留literal-memory executable fallback。若后续证据触发memory实验，再原位替换carrier并写新authority。

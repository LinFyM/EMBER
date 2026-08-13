# Dynamic-K Task-Grounded Visual-Value Writer

状态：2026-08-13 active successor design authority。它从fresh训练，不加载或resume任何Direct-Family-B Writer
checkpoint。长期目标、信息墙、GPU与评测边界仍由`AGENTS.md`和`docs/current_owner_requirements.md`定义。

## 1. 决策

下一轮只改变**set之前的per-video evidence Value**：把同一次π0.5 joint backbone前向中已经得到的、由exact
language定位的真实视觉内容，变成task-grounded terminal goal与有向visual transition，并直接融合进现有
layer/rank memory Procedure。

以下保持不变：

- dynamic K1--K4，训练每macro各6 tasks；
- 每帧真实image + exact language + 50个固定Action probes + 8个memory tokens的joint backbone；
- 每video独立保序、video之间置换不变set；
- `20 policy groups × 8 rank coordinates` M2P；
- fixed template A、direct family-B readout、一套完整38-target rank-8 LoRA；
- train24 full-task equal mean、跨episode B20 functional AS、optimizer与frame budget64；
- rollout前Writer只运行一次，policy闭环时不再看视频。

不加入future-prediction auxiliary head、contrast/order margin、shuffled/reversed negative supervision、expert target、
reward、额外text-only/no-video forward、VL Meta-LoRA、第二套LoRA或新mapper。

## 2. 为什么现在改这个接口

Direct-Family-B的同一macro50 checkpoint得到K1/K4=`102/98`。严格nested K1→K4为
`80 retained/18 gained/22 lost`，没有解锁新task。更关键的机制证据是：K4把same-task effective-BA
centered variance/sample从`.021674`降到`.003438`，约`6.3x`，但task-mean K1→K4 cosine仍`.99604`。

因此：

1. 动态K、per-video编码和set不是空图；它们确实过滤demo-specific nuisance；
2. 被set稳定的per-video task mean本身不够高层或不够policy-effective；
3. 继续增加K、改set、改mapper或美化LoRA几何不会提供缺失的任务过程知识。

当前backbone实际已经保留每帧final task-token hidden，却没有把task-grounded visual content作为LoRA Value：
absolute Action memory只作temporal Query，Value只有Action-memory相邻差分和端点差分。历史v5.2/v6则证明
task-token/patch grounding和visual transition是强机制；v6-fast最终达到历史最好143。本轮把这项未被否定的
机制与昨晚已经对齐的memory-token、动态K和rank-8结构结合，而不恢复整套旧v6。

## 3. 完整数据流

```text
exact language + K=1..4 same-task action-hidden ordered videos
  -> 每帧一次真实π0.5 joint forward
       prefix: 256 image tokens + exact language
       suffix: 50 fixed Action probes + 8 learned memory tokens
  -> 两组同源证据
       M[f,l,r] : 18层 × 8 memory states，policy-native carrier
       E[f,t]   : task-grounded visual evidence，t为有效task token
  -> 每video按实际输入顺序重算
       Dm[f] = M[f] - M[f-1], Dm[0]=0
       Gm     = M[last] - M[first]
       Dv[f] = E[f] - E[f-1], Dv[0]=0
       Gv     = E[last] - E[first]
  -> 每个layer/rank memory cell以absolute memory address作Query
       raw Dv/Gv只作Value，被读成cell-aligned视觉过程修正
  -> D = project(Dm) + read(Dv)
     G = project(Gm) + read(Gv)
  -> causal temporal encoder -> one Program per video
  -> permutation-invariant cross-video set attention/reduction
  -> 20×8 M2P -> shared projector -> direct family-B readouts
  -> one complete 38-target rank-8 LoRA
```

这里仍只有8个真正进入backbone的memory tokens。`20×8`是LoRA层级/秩对齐Program，不冒充memory tokens。

## 4. Task-grounded视觉证据

同一个joint forward完成18层后，取final prefix hidden：

- `H_task[f,t]`：有效task-language positions，已经在真实image context中更新；
- `H_patch[f,p]`：256个真实image positions，也已经被exact language上下文化。

冻结PaliGemma本体不更新，不增加VL Meta-LoRA。共享bias-free投影`W_e: 2048→256`：

```text
T[f,t] = W_e(H_task[f,t])
P[f,p] = W_e(H_patch[f,p])
R[f,t] = W_o Attn(
             W_q RMSNorm(T[f,t]),
             W_k RMSNorm(P[f,p]),
             value=P[f,p])
E[f,t] = T[f,t] + R[f,t]
```

`R`复用v5.2/v6已经验证过的task-queried patch grounding原则：task positions决定关注哪些patch，Value只能来自
真实image positions；无learned V projection、无bias。与旧v6不同，本轮不额外运行text-only encoder，也不训练
VL Meta-LoRA，因此没有第二次backbone forward和额外前端。

`T`含有language context，但它不直接进入LoRA Value。下游只读取`Dv`与`Gv`；相同图像重复时language/static
分量相减为零。这样既利用语言定位对象和关系，又没有`language -> LoRA`静态旁路。

## 5. 视觉变化怎样进入memory grid

对每条video，现有absolute memory semantic address为：

```text
A[l,r] = W_a RMSNorm(mean_f M[f,l,r])
```

它和layer/rank route只进入Query。共享visual reader对每个cell执行：

```text
Q[l,r]   = Wq(RMSNorm(A[l,r]) + route[l,r])
K[f,t]   = Wk(RMSNorm(Dv[f,t]))
V[f,t]   = Dv[f,t]
Rv[f,l,r]= Wo Attention(Q[l,r], K[f,t], V[f,t])
```

端点`Gv[t]`通过完全相同、共享参数的reader得到`Rg[l,r]`。reader是bias-free且没有learned V projection；若
visual transition/goal为零，输出严格为零。最终：

```text
D[f,l,r] = W_m(Dm[f,l,r]) + Rv[f,l,r]
G[l,r]   = W_m(Gm[l,r])   + Rg[l,r]
```

后续causal temporal、goal fusion、set、M2P和LoRA mapper完全沿用当前canonical路径。

## 6. 为什么正确顺序是结构必需的

visual transition和terminal goal都在**实际输入顺序变换之后重新计算**：

- correct展示由初态到目标态的真实有向变化；
- reversed将`Gv`方向翻转，并把阶段因果倒置；
- shuffled改变每个相邻差分，破坏连续阶段；
- 原始frame ordinal只能提供位置，不能恢复被打乱的视觉演化。

模型不会通过训练negative来人为推坏错误视频；训练只输入positive correct videos。顺序作用来自Value本身的有向
定义和causal encoder。最终是否“正确方向有用”仍必须由strict closed-loop五臂证明。

## 7. 为什么目标是高层知识而不是轨迹复制

- `E`由task tokens查询patch content，优先保留语言相关对象、关系和状态，而不是所有像素变化；
- `Gv`只表示初态到终态的语义变化，删除速度和绝对时长；
- `Dv`提供必要阶段的方向，causal encoder整合整条视频，不输出逐帧动作；
- video与B20 action queries同task但跨episode，LoRA必须跨初始化解释独立policy states；
- K>1时每条video先独立产生Program，set只保留跨demo共同结构，已由K4方差证据证明其nuisance reduction有效；
- 输出仍是一套完整task LoRA，rollout期间没有teacher frame或动作轨迹可供复制。

## 8. Policy-effective写出与多task共存

视觉Value不是训练专用旁支：它在causal temporal之前直接加到18层×8 memory cells，随后经过同一个set、M2P和
direct family-B生成所有38个LoRA targets。B20 functional gradient因此必须通过真实LoRA和冻结policy action
才能训练这条路径。

每个task token、memory layer和rank cell保留不同坐标，避免先把所有tasks压成单个pooled task vector；full24
仍按task等权更新。K4已经证明set会压低same-task nuisance；新变量检验的是，加入task-grounded visual goal/
transition后，被稳定的共同方向能否同时支持更多tasks。

这不能保证消除所有gradient conflict；若闭环仍换手，下一断点才进入训练credit或Writer reward，而不是把本轮
内部差异写成成功。

## 9. 历史继承与没有选择的路线

保留：

- v5/v5.2的language-grounded visual semantics与raw patch Value原则；
- v6的Action/policy carrier读取task-grounded adjacent visual transition；
- 当前Dynamic-K的真实joint-backbone memory、每video保序/跨video集合边界；
- SHINE/Doc2LoRA类少量memory、layer-aligned状态和共享结构化LoRA生成原则；
- Direct-Family-B已验证的低开销rank-8完整LoRA写出。

不恢复：

- v6的text-only额外forward、Text/VL Meta-LoRA、旧Semantic Core compiler、rank16 factor heads；
- 任何旧checkpoint或optimizer state；
- future-prediction auxiliary objective。它是后续可证伪候选，但当前增加训练专用head/loss会再次面临
  “预测指标改善却未进入LoRA”的历史错位，不能与视觉Value主变量混在一轮；
- expert reconstruction或dictionary；同task恒定expert target仍不能识别video过程；
- Writer RL；先测试结构性video Value能否恢复强AS cold start。

## 10. 实现所有权和结构边界

- `src/ember/writer/backbone_memory.py`继续唯一拥有joint forward，并在该次前向内压缩task/patch hidden；
- 复用`TaskQueriedPatchGrounding`，不复制第二个patch reader；
- `src/ember/writer/memory_program.py`唯一拥有visual D/G到layer/rank cell的reader；
- `model.py`只传递新增的compact evidence，不新增parallel Writer class或mode flag；
- 原位切换canonical config/checkpoint/evaluation schema；旧实现由Git、旧config和formal artifacts保存；
- 不新增runner、dataset、cache或第二套evaluator。

## 11. 机制与吞吐否决门

CPU/单卡机制必须证明：

1. step0全部38-target LoRA仍是exact functional identity；
2. constant identical frames时`Dv=Gv=0`，visual reader和最终Program动态贡献严格为零；
3. natural、shuffled、reversed在实际输入变换后重算D/G，Program不同；
4. video轴置换不改变full-set Program；K1仍走同一set图；
5. language/static route不能在zero visual value下写出；
6. functional gradient能从LoRA B到visual reader、patch grounding、projection、memory/Action Meta-LoRA；
7. source policy全部冻结；没有额外backbone forward；
8. full24一macro保持K1--K4各6、B20、finite和一次全梯度reduction。

随后用真实最长视频做full24 B20 profile。matched Direct-Family-B world5为`39.4234s/macro`；新分支没有额外
backbone forward，预注册吞吐上限为matched `1.15x`且不得OOM。超过后只优化张量布局/重复计算，不通过删除科学
Value、固定batch1或扩dtype规避。

## 12. Fresh训练与closed-loop裁决

从clean pushed commit的detached worktree fresh训练，不加载Direct-Family-B state。初段固定macro0→200，
checkpoint every25；按K/真实frame cost平衡GPU负载。历史v6-fast在macro50也只有106、到macro200才133，故不再
用`macro50<120`单点提前杀死新架构。

strict paired correct400评测macro50/100/150/200，报告：

- aggregate、8个task、4 suites、breadth、top3 concentration；
- 与最近Dynamic-K K1 102、v6-fast143、old134/compiler138/online128逐task比较；
- 相邻checkpoint retained/gained/lost/churn和是否共同积累；
- K1为主部署读数；若训练曲线支持，再以同一checkpoint补nested K4判断新视觉Value是否让nuisance reduction
  转为真实few-shot增益。

只有初段best至少`125/400`、breadth至少6，并且macro200没有相对best发生大于15的能力崩落，才exact-resume
200→400；否则该结构终局non-pass。达到`>150`后立即补correct/same/wrong/shuffled/reversed/no-video严格
controls。任何loss、transition RMS、cosine或内部order difference都不能替代这套裁决。

## 13. 可证伪解释边界

若成功，只支持：在现有Dynamic-K memory Hypernetwork中，task-grounded raw visual goal/transition Value能让
set聚合到更policy-effective、可共同积累的任务程序。

若失败，只淘汰本轮具体组合：无VL Meta、同forward multimodal task query、raw D/G reader、当前B20 functional
recipe。它不否定dynamic K、few-shot、memory tokens、所有视觉预测目标、Writer reward或LoRA输出整体。

# V6-LPCP Cross-Video Causal Success Distillation

状态：2026-08-15 active design authority。本文在PCSD终局`135/400`及其FP64跨video分析之后，只改变
success credit覆盖的same-task correct video conditions；V6-LPCP部署图、K4输入、65,536参数query commitment、
AS139/rank16 tail、reference/candidate rollout语义与optimizer均保持不变。

## 1. Decision

PCSD证明了两件同时成立的事实：

1. 同初态/同policy RNG的AS139-reference与LPCP-candidate会产生真实discordant success，唯一成功轨迹可形成
   非零continuous LoRA cotangent、Writer gradient、effective BA与action response；
2. anchor-only credit更新后held strict从LPCP143降到135；同task四个不同K4 correct video sets的PCSD增量在
   FP64 effective-BA空间pairwise cosine跨task平均`-.00187`，mean/sample energy ratio=`.24860`。

第二项几乎就是四个互相正交修正平均后的`1/4`。因此当前最早失效接口不是视频carrier、reward存在性、LoRA
rank/能量或Program→action传递，而是：**一条成功轨迹只在产生它的anchor video set上反传，shared query map对
其它same-task videos作出互相近正交的局部外推。**

本设计简称 **CV-CSD**。每个active train task仍只用一个anchor K4条件运行AS139/LPCP paired arms并选出唯一
成功trajectory；随后把同一成功trajectory作为跨初始化成立的task行为，分别在四个互不重叠、原序、action-hidden、
same-task K4 correct conditions下计算完整policy CFM loss，四个独立Writer→LoRA→policy graphs只在共享
`query_delta.weight`梯度处等权汇合。

## 2. Goal, non-goals, and success definition

目标是让一次真实成功credit在不同正确教学视频下形成共同的policy-effective方向，从而同时提高：

- single-checkpoint correct absolute与task breadth；
- 相邻checkpoint support retention；
- same-task不同teacher video输入的鲁棒性；
- correct相对wrong/shuffled/reversed/no-video的闭环价值。

本轮不：

- 增加literal memory token、改Action-probe carrier或新增第二套backbone forward；
- 把rank16降为rank8、改FactorHeads/compiler、scale、dtype或LoRA topology；
- 平均frames、features、Programs或生成后的LoRAs；
- 加negative margin、failed anti-imitation、support projection、expert bank或checkpoint union；
- 做生成LoRA后的task-local RL；这里仍只训练初始Writer。

`>150/400`仍是更高性能目标。owner接受约145成为有效结果，但必须是稳定方法：相邻single checkpoints低churn，
same-task-other基本保持，correct显著优于wrong/shuffled/reversed/no-video。单点145或151都不算成功。

## 3. Input and deployment contract

部署完全不变：

```text
exact task language + one K1--K4 action-hidden correct-video set
  -> one V6-LPCP Writer forward before rollout
  -> one complete 38-target rank16 LoRA
  -> frozen source policy closed loop; no repeated video viewing
```

canonical validation仍使用K4。Writer继续支持训练时已经见过的K1--K4动态数量；CV-CSD只在reward calibration
阶段使用四个独立K4 conditions，不把16条视频同时作为一次部署输入。每个condition内部含4条有序视频并经原有
置换不变set fusion；四个conditions之间只共享训练目标和Writer参数，不做输入或输出平均。

信息墙保持：teacher action/state/proprio/reward/terminal、task ID route、filename、pose与hidden normalization均不可
读；validation/test action/reward reads为0。成功trajectory来自train24 policy自己的on-policy rollout executed
prefix，不是teacher动作，也不与任何teacher video逐帧对齐。

## 4. Why cross-video success is the right invariant

一个成功rollout的低层动作只对它的环境初态有效，但其“这个LoRA在该初态完成了任务”的policy credit与同task所有
正确教学视频相容。让同一executed trajectory在相同初态queries下监督多个video-conditioned LoRAs，正面实现：

- video和action episode跨开，阻断逐帧轨迹复制；
- demo-specific速度、路径、视角或抓取扰动在四个conditions间不共同，梯度会相消；
- 对象、目标关系和有向阶段是四条正确教学集合的共同内容，能在shared query map中反复强化；
- trajectory来自真实成功arm，方向比blind source B20更接近on-policy occupancy。

语言仍是必要query/grounding，但不能单独写LoRA：query commitment只读取同一次真实joint context中的Action-probe
video conditioners，constant-frame路径在LPCP中近零。四个conditions都保留正确原序；只有正确有向过程收到positive
success credit。最终仍必须用wrong/shuffled/reversed/no-video closed-loop controls排除task-language/static shortcut。

## 5. Exact training semantics

对train task `i`、cycle `c`：

1. anchor set `S_i^0`保持PCSD schedule：`visit=c-1`的4条correct demos；
2. support sets `S_i^1,S_i^2,S_i^3`依次使用后续visit的确定性permutation，并排除本cycle已用demos；四个K4 sets
   共16条demo互不重叠；
3. 只在`S_i^0`下缓存一次conditioning，生成exact-zero-query AS139 reference与current LPCP candidate；
4. 两个arms在2个相同reset/state与相同policy RNG prefix下闭环；ties为zero，只保留唯一成功arm trajectory；
5. 若task没有discordant pair，本task仍为zero credit，不为凑满task读取额外videos；
6. 若active，完成同一selected trajectory batch一次，并对四个conditions分别生成完整candidate LoRA与CFM loss。

记Writer参数为`w`，同一selected-success replay为`tau_i`，四个独立LoRAs为
`Lora(w,T_i,S_i^v)`：

```text
J_i^v(w) = CFM(pi[Lora(w,T_i,S_i^v)], tau_i; shared flow time/noise panel)
J_i(w)   = (1/4) * Sum_v J_i^v(w)
J(w)     = equal mean over active tasks of J_i(w)
```

四个view复用完全相同的trajectory chunks、executed-prefix mask、Beta times与Gaussian noises，使view差异只来自
teacher condition。每条selected trajectory在view内仍等权；四个views总task权重为1，duplicate四次同一view时
梯度必须严格退化为PCSD anchor-only梯度。view顺序任意置换不得改变task gradient。

每个view独立执行：

```text
ordered videos + exact language
  -> cached LPCP conditioning state
  -> current query delta
  -> frozen AS139 set/fusion/compiler
  -> complete rank16 LoRA
  -> selected-success functional CFM cotangent
  -> backprop through that same view to shared query_delta.weight
```

不是把anchor LoRA cotangent机械复制给其它views，也不是平均四个LoRAs；每个view在自己的当前LoRA处计算精确
functional gradient。只训练`query_delta.weight`，AdamW recipe、active-task等权与一次full24 gradient reduction保持
PCSD不变。

## 6. Why this is not an old experiment repeated

- **PVJFC**：旧one-shot Program-memory路线，用两条video的offline B20 cotangents、correct/negative rows与正规方程；
  因condition>200未获formal。CV-CSD使用当前LPCP143 K4图、真实selected-success trajectory、四个disjoint views、
  Adam query map，无negative solve。
- **PICK/PICK-GC**：替换condition key并继续blind B20；CV-CSD不改condition carrier或key。
- **K4 Slot/Procedure Set**：证明set aggregation可降低same-task nuisance，但没有一个跨view真实成功target；CV-CSD
  保留其强图，只改变credit coverage。
- **PCSD**：只在产生trajectory的anchor set反传；CV-CSD唯一新增的是同一成功行为在其它correct sets上的完整
  functional supervision。

历史“same-task corrections可能有各自有用方向”不等于应保留当前近随机正交外推。CV-CSD不强制四个LoRAs或
Programs相等，只要求它们各自在相同成功behavior上下降，因此仍允许不同初始video条件保留有用差异。

## 7. Canonical implementation boundary

1. 原位替换PCSD active reward recipe；PCSD代码历史由Git、design和formal artifacts保存，不保留runtime mode开关；
2. reward cycle保留anchor paired rollout owner，只在active task的credit阶段编码三个额外disjoint K4 conditions；
3. selected trajectory batch只collate一次；四view policy functional graphs串行执行并立即释放，不同时保留四套
   policy graph；
4. view-specific Writer conditioning可逐view编码，先profile后才决定安全microbatch；不为逐元素一致固定batch1；
5. gradient在task内除4，再沿现有one-vector all-reduce做active-task等权；不增加逐tensor扫描或hash；
6. config/run/checkpoint/eval schema fresh-incompatible，模型topology虽相同也不得exact-resume PCSD optimizer；
7. deployment checkpoint仍只含一个Writer；evaluator不读train-only support views或reference arm；
8. source policy和全部frozen Writer参数0 trainable，完整checkpoint/resume与动态work queue语义保持。

## 8. Fast falsification and profile

### 8.1 CPU/mechanism contracts

- anchor schedule与PCSD cycle1一致，四个K4 sets各自unique且16 demos互不重叠；
- support views与selected trajectory来自同task但无teacher-action pairing；
- 四view复用同一replay rows、flow times/noises，task总权重恰为1；
- view permutation invariance、duplicate-view degeneration到PCSD、inactive task zero extra video reads；
- 四个view都产生finite/nonzero LoRA cotangent与query gradient；只`query_delta.weight`可训练；
- step0 candidate仍逐tensor等于LPCP macro25，reference仍是exact AS139；
- 0 teacher/target/validation/test forbidden reads。

### 8.2 Live profile

先跑一个已知active task真实smoke，再跑一次full24 cycle profile/fresh formal。必须确认：

- 24 tasks、48 paired states、96 rollouts语义不变；support views不增加rollouts；
- active task各有4个独立condition gradients，ties仍zero；
- qdelta、effective BA与fixed-action response非零；0 OOM/nonfinite/watchdog；
- cost-balanced动态队列没有因active view数使rank长尾失衡；
- full24 wall不超过PCSD `837.694s`的`1.75x`；超限先消除重复collate、graph驻留和host sync，不缩科学合同。

post-cycle报告train-active四view的FP64 effective-BA correction coherence与mean/sample energy，但这只验证改变确实
作用于目标接口，不以漂亮几何替代strict400。

## 9. Closed-loop and stability gates

机制与吞吐通过后，从同一LPCP macro25 Writer weights和fresh optimizer做cycle1，立即运行与LPCP/PCSD相同K4
correct strict paired400，并报告AS139、LPCP143、PCSD135、v6-fast143、old134/compiler138/online128的逐task、
per-suite、breadth、retained/gained/lost/churn。

- cycle1 correct`<144`、breadth`<7`、相对LPCP lost`>10`、gained不大于lost或少于3 suites不降：终局；
- cycle1过门才exact-resume cycle2并再次跑同schedule strict correct400；
- 稳定资格：两点都`>=144`、mean`>=145`、breadth都`>=7`、相邻churn`<=20`、Jaccard`>=.85`，没有suite或
  持续task坍塌；
- 稳定过门后在最终single checkpoint运行same-task-other、wrong、shuffled、reversed、no-video；same/correct
  至少`.9`，correct对每个negative/no-video margin至少8且paired correct-wins多于control-wins；
- 即使某点`>150`，也不能跳过相邻checkpoint和视频资格；checkpoint union/挑点不计。

若cycle1 absolute提高但support仍高churn，只能说明跨video target改善了总量，没有解决稳定共同积累；不得用
cycle2以外的LR/seed/scale扫救。若absolute与stability通过而controls失败，则说明cross-view训练收缩成task/static
shortcut，方法仍non-pass。

## 10. Expected evidence and falsifiers

本假设快速被以下证据否决：

1. 四view exact loss无法在现有query map上同时形成非零下降，或duplicate/swap合同不成立；
2. post-cycle same-task跨video correction仍保持PCSD的近零coherence，说明query commitment容量/位置而非credit覆盖
   是更早限制；这会触发下一轮考虑layer-aligned memory commitment，而不是继续加view数；
3. strict仍低于144、breadth下降或相对LPCP继续lost>gained；
4. 相邻checkpoint仍大换手，说明cross-video共识没有解决跨task共存；
5. correct与same稳但wrong/shuffle/reverse/no-video同步提高，说明language/static bypass；
6. 训练只能靠显著牺牲吞吐、batch1、重复backbone forward或dtype扩展运行。

若失败，只淘汰“LPCP query-only map + four-disjoint-K4 exact selected-success functional mean”这一组合，不否定
few-shot、memory token、其它policy-aligned credit或生成LoRA本身。memory token的下一触发证据将是：正确的跨video
reward objective已经给出，但当前query commitment仍无法把它表达成一致且policy-effective的layer-aligned写出。


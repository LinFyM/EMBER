# V6 Ordered-Procedure Actual-Delta Success-Support Projection

状态：2026-08-14 sealed active design authority；简称ADSP。实现、完整CPU门和真实GPU smoke均已通过；下一步
从同一个AS macro25 checkpoint fresh启动formal，不resume已经终局的raw reward cycle1。

## 1. Decision

V6 Shared-Core Ordered-Procedure Common-Value的K4 AS checkpoint为`139/400`。一次on-policy LOO reward
cycle得到`138/400`，相对AS严格配对为：

```text
120 retained / 18 gained / 19 lost / 243 both-fail
```

reward不是零信号：24 tasks、96 rollouts中有64 successes、14 mixed tasks；q/k/output、effective BA和fixed
action response全部非零。AS到reward的effective-BA relative-L2均值只有`.003323`、cosine
`.99999505`，却翻转37条闭环结果。最早失效接口因此是**24-task reward credit汇合成一个shared AdamW
parameter update后没有保住已有support**，不是LoRA能量、rank、Procedure、compiler或reward没有穿过Writer。

本轮只检验：

> 在完全相同的raw on-policy preference proposal之后，把最终actual AdamW parameter delta投影到所有
> train24成功executed-prefix loss的一阶非增半空间，能否保留raw proposal的18项acquisition并显著减少19项
> support loss。

若该投影真实active但strict仍低于门，本轮终局；不继续调margin、LR、scale、rank、MC、constraint阈值或cycle。

## 2. Why V6 now, and where memory tokens stand

memory token是重要的候选架构机制，不是项目goal或强制形式。它可能提供真实backbone context中的逐层证据，
帮助视频知识按policy topology进入Writer；但它不会自动解决A/B gauge或multi-task credit。现有纯memory路线的
K1/K4最好为`102/98`，Full-Factor为`91`；V6则已经证明完整LoRA具备`143`级policy-effective几何。

因此当前不把“如何生成合理LoRA”和“如何保住shared reward support”混成一轮。ADSP保留V6的健康LoRA生成器，
只处理最新结果直接暴露的support接口。若ADSP失败，下一架构候选不是原样恢复91分Full-Factor，而是保留V6的
absolute Core、有向Procedure与factor compiler，再把layer-aligned memory作为视频到policy slots的接口。

## 3. Unchanged deployment and reward graph

```text
exact language + K4 same-task ordered action-hidden videos
  -> frozen v6 image/language/Action-probe evidence
  -> shared Semantic Core + per-video causal Procedure
  -> trainable Procedure Common-Value set
  -> native compiler + factor heads
  -> one complete 38-target rank-16 LoRA
  -> four train24 random-reset rollouts
  -> binary outcomes + current-policy executed prefixes
```

部署仍在rollout前运行Writer一次；validation rollout不更新任何参数。teacher action、target dataset action、
validation/test action或reward均不产生gradient。reward阶段只使用当前policy自己执行的train24 prefixes。

raw proposal与已完成reward cycle完全相同：

- K4 LOO binary advantage；
- replan chunks先episode内等权，再四episode等权；
- exact-Beta time与task-keyed independent Gaussian noise，Nmc4；
- 24 tasks等权形成一个shared preference gradient；
- 相同AdamW、gradient clip、BF16/TF32与19.7万trainable Procedure-set参数。

## 4. Success-support tangent

对task `i`的四条rollout，令成功集合为`S_i`。若`S_i`非空，定义一个task-equal support loss：

```text
L_keep_i =
  mean over successful episodes
    mean over that episode's executed replan chunks
      mean over Nmc4 CFM loss(current executed prefix)

g_keep_i = grad_theta L_keep_i
```

每个task只形成一条support row，避免长horizon或更多success获得更大权重。mixed task在同一次per-chunk policy
forward中同时得到preference与support cotangent；all-success task只计算support；all-failure task两者都为零。
不重复policy forward来获得同一个per-chunk loss，不保存reward replay进checkpoint。

support loss不是teacher trajectory imitation。它只要求一次参数更新不要在当前成功occupancy附近提高模型对
自己已执行动作prefix的局部flow loss；binary success负责认证哪些prefix属于已有support。

## 5. Projection of the actual shared AdamW delta

先完全按raw reward合同得到24-task preference mean、clip后的gradient和AdamW candidate。记更新前参数为
`theta_0`，AdamW候选为`theta_raw`：

```text
d_raw = theta_raw - theta_0
```

把每个非零`g_keep_i`按正尺度单位化为`n_i`，求唯一Euclidean projection：

```text
d_star = argmin_d 0.5 * ||d - d_raw||^2
         subject to n_i dot d <= 0 for every task with success
```

约束数最多24。small dual只在CPU FP64上对不超过24x24 Gram求NNLS，大参数delta保持GPU FP32；不物化
P-by-P矩阵。最终参数写为`theta_0 + d_star`。Adam moments仍由完全相同的raw preference gradient更新；这使
cycle2若被授权时仍有明确exact-resume语义，同时本轮实际部署参数严格等于projected candidate。

无support或raw candidate已经可行时必须逐元素退化为raw AdamW结果。投影必须满足：

- 每条support row的最终directional derivative非正；
- `d_star`不由task顺序、constraint正尺度、重复或线性相关rows改变；
- 若可行集合只保留近零方向，机制门直接non-pass，不靠软化约束强推；
- projected preference directional derivative仍须为负，且保留足够raw descent。

## 6. What is and is not changed

| Interface | Raw reward | ADSP |
| --- | --- | --- |
| exact language / K4 videos | unchanged | unchanged |
| Core / ordered Procedure / set | unchanged | unchanged |
| public LoRA | 38 targets, rank16 | unchanged |
| on-policy rollouts | train24 x 4 | unchanged |
| LOO preference gradient | full24 equal mean | unchanged |
| AdamW proposal | one shared candidate | unchanged |
| success prefixes | only enter mixed preference | also form one keep row per successful task |
| final parameter update | raw AdamW delta | nearest support-feasible actual delta |

不加入memory token、rank8、B20、negative loss、task expert、task ID route、checkpoint/model averaging、第二套LoRA
或生成LoRA后的task-local RL。

## 7. Difference from historical guards

- OSG-PC把B20 task-local Program proposal先按每条success轨迹投影，再经过full48 shared solve；它因full-prefix
  VJP长尾未到formal。ADSP的proposal本身就是已跑通的on-policy reward，约束发生在task汇合和AdamW之后。
- SKNC保护condition-key的零阶Program equality，formal为137且held lost13；ADSP不保存key/bank，直接约束当前
  成功occupancy loss的一阶方向。
- SRTP在PICK/SKNC Program memory上使用固定landmarks，两个profile在mixed CFM OOM；ADSP复用已经以B8、
  Nmc4、graph-release跑完full24的当前reward执行图，并在同一次forward中提取两种cotangent。
- RLS保护offline rows，不代表held occupancy；ADSP使用当前policy真实成功rollout，但仍诚实承认train24
  occupancy未必外推validation，最终只认strict400。

## 8. Canonical implementation ownership

- `writer/reward_preference.py`唯一拥有preference/support LoRA cotangent与small-dual actual-delta projection；
- `writer/reward_cycle.py`唯一拥有K4 rollout、dynamic task queue、full24 reduction和projected AdamW step；
- `writer/reward_training.py`只拥有runtime/asset准备、launch contract、checkpoint循环与CLI；
- preference flat gradient仍只做一次P-vector all-reduce；support使用一个`[24,P]` FP32 task-row tensor做一次
  all-reduce，不按task发送Python大对象；
- checkpoint、config、evaluation schema整体fresh-incompatible；raw reward config保持results-only sealed；
- 不新增第二trainer、第二Writer、strategy flag或fallback。历史runtime由Git和formal artifacts保存。

## 9. CPU and synthetic gates

1. K4 LOO advantage与raw preference microbatch invariance保持；
2. success weights使每个success episode等权、每task support row总权重为1；
3. mixed task一次per-chunk forward得到两个finite nonzero cotangent；all-success只产生support；all-failure零forward；
4. no-constraint、raw-feasible、单冲突、多冲突、duplicate/permuted/rank-deficient cases满足解析/KKT；
5. actual AdamW delta包含moment、epsilon和weight decay；最终参数逐元素等于`theta_0+d_star`；
6. source policy、frozen v6、teacher/validation/test forbidden reads保持0；
7. fresh checkpoint不能误载raw reward cycle，evaluation只接受新deployment kind；
8. 一个canonical reward path，完整CPU回归、compileall和diff check通过。

完整CPU回归为`400 passed`，compileall与diff check通过；架构门禁无hard violation。以上只证明方法被真实实现，
不预测closed-loop。

## 10. Live smoke and formal gate

先在live合适A40上运行历史同seed为`1/4`的train task4单task smoke。它必须同时得到preference与support
cotangent、raw constraint violation或明确raw-feasible fallback、finite projected delta和非零BA/action response，
且不超过A40显存。all-success/all-failure语义由CPU直接覆盖，不为形式再跑重复GPU forward。

task4真实smoke在gpu02物理1完整exit0：同旧raw smoke均为`1/4` success、157 replay chunks、80次policy forward，
preference梯度与effective-BA/action response逐位一致；新增support LoRA gradient RMS=`5.00225e-5`、16次support
backward。该task的raw proposal已可行，故1条constraint严格identity fallback，final violation=0、descent/energy
ratio均为1。cycle=`157.599s`，相对raw `146.383s`为`1.077x`；peak reserved=`36.774GB`，0 OOM/nonfinite。

smoke证据已seal进config。formal从同一个AS139 macro25 Writer weights和fresh optimizer启动一个full24 cycle，
并必须满足：
formal必须满足：

- 24 tasks、96 rollouts、四suite完整，preference/support task rows与outcomes一致；
- projection实际active且final support violations为0；
- projected preference directional derivative为负，保留raw preference descent至少25%；
- projected/raw delta energy ratio至少0.10，避免约束把更新机械归零；
- q/k/output、effective BA与fixed action response非零；
- 0 OOM/nonfinite/watchdog/forbidden read，cycle wall不超过raw reward matched wall的2.0倍。

未过机制门不做strict评测。通过后立即运行同schedule K4 strict paired correct400。

## 11. Closed-loop adjudication

只认single checkpoint strict paired结果，并相对AS139与raw reward138逐episode报告retained/gained/lost：

- `<144`、breadth`<7`、相对AS139 lost`>10`或gained不超过lost：终局non-pass；
- `144..150`且breadth至少7、lost不超过10、gained>lost、至少三suite不降：才允许讨论一次exact cycle2；
- `>150`：先封存checkpoint，再补correct/same/wrong/shuffled/reversed/no-video严格配对controls；
- correct没有真实优于controls时，即使absolute过门也不是最终有效方法。

若ADSP projection active、内部约束全过而closed-loop仍失败，说明train24成功prefix的一阶support不能代表held
support，或当前19.7万Procedure-set参数没有共同改善空间。届时停止V6 constraint小修，下一轮转向V6强
Core/Procedure/factor compiler与layer-aligned memory结合的架构级接口。

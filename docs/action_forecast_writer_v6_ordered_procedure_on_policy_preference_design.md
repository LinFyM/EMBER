# V6 Ordered-Procedure On-Policy Preference Writer

状态：2026-08-14 active implemented authority；CPU与单task真实GPU smoke已通过，等待formal cycle1与strict400。
该设计只改变Shared-Core Ordered-Procedure Common-Value Writer的训练credit；输入、表示、compiler、LoRA
topology与部署图全部保持不变。

## 1. Decision

V6 Shared-Core Ordered-Procedure Common-Value macro25已经给出完整终局证据：

- K4 strict paired correct=`139/400`、breadth6，恰好等于matched parameter-free Shared-Core=`139`；
- strict paired为`120 retained / 19 gained / 19 lost`，Long1净`+4`完全由Spatial/Object各净`-2`支付；
- raw ordered Procedure correction relative-L2=`.09601`，current→output-zero effective-BA=`.01397`，说明路径
  已经打开且能改变policy；
- train-seen 8 tasks×10 states的同一output-zero反事实为trained/zero=`64/64`，严格`4 gained / 4 lost`；
- held和train-seen都只有能力换手，没有任何on-policy净收益。

所以最早断点不再是Value相消、Procedure顺序、写出幅度、rank、compiler或held-only泛化。当前full24 B20
functional objective本身没有把已打开的Procedure residual对到真实闭环有益方向。下一轮不得继续放大Value、
增加capacity、降低rank、调attention或续训macro25。

本设计保留macro25作为一次短、task-balanced AS cold start，随后关闭target action入口，只用train24真实
on-policy binary reward优化同一个shared Writer。部署仍是zero-interaction：语言与K条action-hidden视频在
rollout前一次生成一套LoRA，执行时不再访问视频、reward或环境历史。这不是生成LoRA后的task-local RL。

## 2. The one causal variable

冻结不变：

- exact language + `K=1..4` same-task ordered action-hidden videos；首个reward实验固定真实最强候选`K=4`；
- frozen historical v6-fast evidence、shared-Core union、per-video ordered Procedure reader；
- `PolicyProcedureCommonValueFusion`的q/k/output结构与`197120`个FP32 trainable parameters；
- native AdaLN、post-fusion、38-target rank16 factor heads和完整frozen source policy；
- frame stride5、video schedule、official LIBERO preprocessing和train24/validation8 split；
- deployment只运行一次Writer并输出一套LoRA。

唯一变化：

```text
old:  cross-episode source-action B20 functional loss
new:  task-relative on-policy success/failure preference over executed prefixes
```

reward阶段不读取target dataset action、teacher action/proprio/state/reward/terminal、validation/test action或
reward。唯一动作是当前policy在train24闭环中自己执行过的action prefix。

## 3. Exact training graph

初始化只加载AS macro25 Writer权重；不加载AS optimizer、scheduler或sampler state，新建reward-stage optimizer。
一次reward cycle对24个train tasks各做一次：

```text
exact language + four ordered action-hidden videos
  -> frozen v6 evidence
  -> detached shared Core + four detached ordered Procedure slot tensors
  -> current trainable Procedure Common-Value
  -> frozen native compiler
  -> one complete rank16 LoRA
  -> four official random-reset on-policy rollouts
  -> success/failure and executed observation/action prefixes
  -> recompute only small Procedure fusion + frozen compiler, never video backbone
  -> frozen-policy executed-prefix flow objective
  -> gradient to shared Procedure q/k/output only

24 task-equal gradients
  -> one shared all-reduced AdamW update
  -> one Writer checkpoint
```

rollout前的LoRA generation在inference mode完成，长simulator执行期间不保留autograd graph。每task只保留
detached shared-Core/per-video Procedure slots与CPU executed-prefix replay；reward backward时重解小型fusion和
compiler一次，不重读视频、不重复backbone forward。一个task完成后立即释放replay和policy graph。

task ownership使用同节点完成驱动队列。video、environment、policy noise和flow-credit seeds只由task/cycle/lane
决定，不含rank或claim顺序；因此world size 1--6只改变executor，不改变科学样本。每rank累积小型Writer gradient，
最后一次all-reduce并严格除以24，保持full24等权。

## 4. Reward preference objective

每task固定四个rollouts，binary success为`R_e in {0,1}`。沿用已经验证有内容的leave-one-out advantage：

```text
A_e = (4 R_e - sum_j R_j) / 3
```

每个replan只监督真实执行的前1--5步，不把未执行action chunk尾部当成轨迹。对task `i`：

```text
J_i = (1/4) sum_e A_e
      mean_{executed replans in e}
      mean_{m=1..4} L_CFM(action_prefix | observation, LoRA_i)

J = (1/24) sum_i J_i
```

梯度下降会降低successful trajectory的flow loss、提高failed trajectory的flow loss；同task四条episode等权，
不同长度不能靠更多replans获得更大权重。homogeneous 4/4或0/4 tasks的advantage严格为零且不做reward CFM
forward，不能伪造没有比较证据的方向。

flow time/noise沿用exact Beta(1.5,1)与task-keyed Gaussian、Nmc4、完整logical panel先生成再按physical B8切片；
正常BF16/TF32和batch低位差异被接受，不固定batch1、不重复forward、不扩dtype。

首版optimizer继承AS的AdamW语义与当前约`3e-4`学习率、betas、eps、weight decay、clip1，但使用fresh moments。
不扫learning rate、advantage scale、Nmc、K、rollout数或temperature。

## 5. Why this is not historical Reward-Credit or task-local RL

历史Reward-Credit把每taskreward cotangent先压成一个`[320,256]`Program方向，再以一次manual full48 reconciliation
写入约2100万维residual memory。cycle1虽形成nonzero Program，但q/v native factor运动约`1e-8 RMS`，低于非零
BF16 factor约`1e-4`局部ULP，strict仍`134`且`14/14`换手。

本设计不直接部署一次微小cotangent：reward loss经完整native compiler反传到19.7万FP32 Writer参数，由Adam
累积后重新生成LoRA。机制门要求effective BA和fixed-action response实际改变，因而不能用sub-ULP内部gradient
冒充写出。它也不恢复Program bank、RLS、candidate guard、success key、rank reservation或第二套LoRA。

生成LoRA后的task-local RL仍是未来独立实验。本轮训练结束后，validation rollout没有任何在线参数更新；被评估的
仍是一个shared Writer checkpoint生成的初始LoRA。

## 6. High-level video knowledge and order

reward不提供task ID route或reward-only LoRA。产生梯度的LoRA仍由exact language查询四条action-hidden视频：

- shared Core只提取跨video对象、关系和目标语义；
- 每条video内部的causal Procedure encoder、rotary ordered reader和centered Procedure Value保留有向阶段；
- K轴只做置换不变的common-Value aggregation，不拼接不同demo为虚假物理序列；
- executed trajectories来自不同random resets，无法逐帧复刻任一teacher video；
- 同一Writer参数必须让24种video-conditioned LoRA都改善，reward、task ordinal和queue ownership不进入部署。

这仍不自动证明正确顺序有用。只有checkpoint absolute先过门后，correct严格优于wrong/shuffled/reversed/no-video、
same-task-other接近correct，才能支持视频因果claim。

## 7. Coexistence and policy-effective write

本轮不再用offline gradient conflict surrogate推断共存。每cycle只有一次full24 task-equal shared update；不存在
per-task optimizer、checkpoint选择、expert route、LoRA平均或task-specific保留。reward直接来自同一LoRA在多个
初始化的完整任务结果，因而比B20 flow loss更接近最终occupancy，但train24 reward仍可能不泛化到validation8；
strict paired400必须裁决。

AS macro25保持强139底座与已打开的q/k/output。reward阶段只允许这些small shared parameters变化，frozen v6和
source policy逐参数不更新。每cycle报告q/k/output delta、Procedure correction、effective BA、fixed-action response、
train outcomes和24-task gradient cosine/冲突；这些只解释closed-loop，不选择checkpoint。

## 8. Implementation and checkpoint boundary

- 新增一个reward-stage trainer/entrypoint；不把simulator/replay逻辑塞进AS B20 owner；
- `writer/model.py`只增加“从detached frozen readouts重解同一Program”的窄接口，不增加第二Writer；
- `reward/rollout.py`保持唯一official random-reset rollout owner，统一K4 replay而非复制environment代码；
- reward checkpoint保存Writer、fresh optimizer、cycle cursor、每rank RNG与完整schema；初始化AS checkpoint只作
  provenance，不能把reward run误称AS exact-resume；
- evaluator显式识别reward training contract，但继续构造同一个canonical Writer与同一个K4/B8 deployment graph；
- 历史Reward/PCUG/SRTP公式只选择性读取，不能整条恢复退役Program-memory trainer。

## 9. Fast falsification and formal gate

### 9.1 CPU and one-task GPU smoke

1. cold-start Writer逐tensor等于AS macro25；v6 base与source policy全部frozen；
2. detached-readout recompile与正常forward的76 LoRA tensors在正常batch低位范围内等价；
3. mixed K4产生finite/nonzero q/k/output gradient，homogeneous task为zero objective/zero reward forward；
4. executed-prefix mask、episode/task等权、LOO符号和Nmc physical-batch invariance正确；
5. source/teacher/validation/test action与reward forbidden reads为零；
6. reward optimizer step后LoRA、effective BA和fixed-action response均发生非零可部署变化；
7. queue interleaving不改变task/video/env/policy/flow identities；0 OOM/nonfinite。

实现commit=`e06a14b3f593536a7c5889bb4ce776876f43c76f`，正确LIBERO assets下full CPU=`395 passed`。gpu02物理1
的task4真实smoke为`1/4` success、157 replay chunks、80次B8×Nmc4 functional forwards；reward LoRA gradient
RMS=`1.3138e-5`、Writer grad norm=`8.0119e-4`，q/k/output均产生约`1e-4`量级FP32更新。更新后的LoRA factor、
effective BA与同query/noise fixed action response RMS分别非零，其中BA=`1.8146e-4`、action=`5.5719e-3`；peak
reserved=`40.775GB`、wall=`146.295s`、exit0。此前task0四条rollout同质并严格走zero-CFM/zero-gradient硬停，证明
homogeneous skip有效；task4是预先由历史同seed reward面板判定的mixed smoke，不是参数或结果sweep。

### 9.2 First complete cycle

small smoke通过后，从clean pushed commit做一次formal full24 reward cycle，不先重复一遍discarded full24 profile。
同一run同时封存真实wall、outcome、gradient、checkpoint和mechanism report。必须满足：

- 24 tasks、96 rollouts、四suite完整，mixed tasks至少6且覆盖至少3 suites；
- mixed task reward gradient finite/nonzero，homogeneous forward exact zero；
- full24 shared gradient/update非零，q/k/output与effective BA没有sub-ULP闭合；
- 一个task不保留第二套LoRA或trajectory graph，peak/headroom和dynamic queue无长尾失控；
- 0 forbidden read、OOM、nonfinite、watchdog，生产wall不超过历史Reward-Credit matched topology的`1.25x`。

任一机制hard gate失败就终止，不扫B、Nmc、rollout数、LR、K、dtype或allocator。

### 9.3 Real-performance decision

cycle1完整后立即运行与当前K4 139相同的single-checkpoint strict paired correct400：

- `correct < 144`、breadth`<7`、相对139 lost`>10`或gained不超过lost：终局non-pass；
- `144..150`且breadth/retention/三suite趋势过门：只允许同合同cycle2，再次strict400；
- 首次`>=144`才补correct/same/wrong/shuffled/reversed/no-video；
- 成功仍要求同一checkpoint strict correct严格`>150`，correct优于全部negative controls，same/correct`>=.9`。

不使用train80、reward objective、union、LoRA norm或内部gradient挑checkpoint；train-seen panel只可解释结果。

## 10. Interpretation boundaries

- mixed gradient存在但BA不变：仍是compiler/native precision接口，当前direct-parameter假设失败；
- train reward改善、held不增：train24 on-policy preference未形成跨task可组合映射；
- held gained/lost仍相等：reward surrogate或shared optimizer仍只造成能力换手；
- retention改善但absolute不升：reward学会保护，却没有all-failure acquisition；
- absolute升而negative同步升：language/static shortcut，科学non-pass；
- strict超过150且controls健康：才说明K4有向视频Program经reward-trained shared Writer形成有效初始LoRA。

负结果只淘汰“当前K4 ordered-Procedure bridge + LOO executed-prefix preference + small shared Adam update”，不否定
所有Writer-RL、few-shot、memory-token Hypernetwork或未来生成LoRA后的task-local RL。

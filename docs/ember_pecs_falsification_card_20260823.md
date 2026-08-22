# EMBER-PECS：Policy-Effect Constrained Solver falsification card

状态：2026-08-23 **local-effect held5为58/250并未过Gate 2；只授权一次预登记的完整去噪trajectory target复验**。

本文是MDCO失败和Stage 1全链复盘后的单一科学合同，不是`v25`。它保留EMBER-ECP已经通过的Stage 0
event/Action observer与最终single-LoRA部署目标，但停止当前learned `q_pi + Program-to-LoRA compiler` family。
只有本文第一道privileged oracle通过后，才实现video effect predictor；未通过时不得用新的head、fusion、rank、LR、seed或
更多同类mapping继续修补旧compiler。

## 1. 决策与唯一主要变量

当前最早失败接口不是LoRA容量、输出幅度、task数量、gradient、layer address或simulator credit。MDCO已经在90个task-equal
mappings、108次dense updates和一次fit90 structured credit后，于source-unseen held5只得到`20/250`，低于source`21`和
stable shared`43`；不同Stage 1版本虽然LoRA方向差异很大，却反复停留在几乎同一批easy/shared outcome rows。

因此本合同唯一主要变量是：

> **删除需要跨任务学习的Program-to-LoRA参数映射；让视频/privileged teacher只定义policy-native effect constraints，再由一个
> task-agnostic、固定算法的proximal inner solver在冻结PI0.5上求出唯一一套完整rank16 LoRA。**

简称 **EMBER-PECS（Policy-Effect Constrained Solver）**。

这不是把compiler换成另一组更大的heads。PECS没有可按task记忆mapping的decoder参数；同一个固定求解算法对每个task运行，
task信息只能来自exact language、action-hidden视频及由它们预测的policy-effect target。

当前不选择的可信替代是从Stage 0直接训练一个end-to-end video-to-LoRA Writer并用simulator reward共同更新。GOMQ的
`151→135→131`与MDCO的fit-local credit都说明这种joint shared update会重新落入support换手/easy-state basin，而且再次把
video semantics、policy坐标和LoRA实现混为一个优化问题。它不作为下一实验。

## 2. 保留与停止的边界

保留：

- frozen source PI0.5、PaliGemma、native Action Expert和Stage 0 v3 + Action Meta v3 authority；
- exact language、每视频独立保序、ordered events、38 owner、50 joint horizon positions和Dynamic-K接口；
- stable shared prior作为求解起点与no-process反事实，最终仍merge/物化为一套complete rank16 LoRA；
- 95-task evidence/support assets、successful members、held5 direct experts和strict paired evaluator；
- train/meta privileged action、expert和reward只作训练teacher，validation/Test信息墙不变。

停止：

- learned target/rank query到A/B heads的当前compiler family；
- task-local free Program、task-ID codebook、raw A/B重建、LoRA cosine/retrieval成功门；
- 通过增加layer heads、width、fusion、selector、rank或mapping visits继续挽救MDCO；
- 在privileged PECS oracle通过前训练`q_V`或接新的outer credit。

## 3. 完整矩阵流水线

### 3.1 Deployment-visible evidence

输入仍只有：

```text
exact language L + K条same-task action-hidden ordered videos V_1...V_K
```

冻结Stage 0 authority逐帧运行native prefix和`u=1` fixed Gaussian suffix，得到：

```text
H_src[k,s,j,h,:] : [K,T_k,38,50,128]
P_process         : [8,38,128]
rho               : [8]
sigma_event       : [8,38,128]
P_lang/P_scene    : [38,128] each
alpha             : per-video ordered frame-to-event posterior
```

`j`仍精确对应18层q、18层v、action-in和action-out；`h`仍是joint future coordinate，不被解释为第二条因果时间轴。
每条视频先独立编码，跨视频只在event层对齐和聚合。

Action Meta v3只安装在这条已经冻结的Stage 0 observer路径中。下面的expert-effect capture、inner solver和最终rollout policy均不
安装Action Meta；candidate生成的complete rank16 LoRA仍是部署policy上的唯一adapter。

### 3.2 Policy-effect target，而不是LoRA target

训练期在同一language、同一action-hidden视频帧和同一fixed probes上分别安装source/shared与一个或多个successful expert members。
teacher action不进入这些forward。对每个member收集：

```text
owner response effect : [frame,38,4,128]
flow velocity effect  : [frame,2,50,32]
```

`4`是50 horizon positions的固定DCT basis；`2`是canonical/antithetic `u=1` noise probes。经Stage 0的有序`alpha`聚合后，
privileged teacher distribution为：

```text
mu_owner       : [8,38,4,128]
var_owner      : [8,38,4,128]
mu_flow        : [8,2,50,32]
var_flow       : [8,2,50,32]
effect_presence: [8]
```

均值/方差在policy function中由多个successful members形成；不平均raw A/B，不指定唯一expert参数，不把同一优化轨迹的多个
checkpoint虚报为独立task mappings。只有一个member时保留其effect，并使用跨state/probe的不确定性而不是伪造member variance。

第一道oracle直接使用上述exact privileged effects，完全绕过video predictor。oracle通过后才训练一个deployment effect
predictor：

```text
R_V(P_lang,P_scene,P_process,H_src,rho,sigma_event)
    -> predicted mu/var_owner, mu/var_flow, effect_presence
```

这个网络预测的是同一视频中可识别的policy response constraints，不直接输出A/B，也不能只凭language或owner address写出
process effect。language+scene arm将process置零，作为learned prior baseline。

### 3.3 固定proximal inner solver

令`Theta_0`为一套完整rank16 stable shared LoRA。对输入视频的event-support frames，候选`Theta_n`经过冻结PI0.5产生
`C_owner(Theta_n)`与`C_flow(Theta_n)`。固定目标函数为：

```text
L_effect = uncertainty-weighted owner-effect error
         + uncertainty-weighted canonical flow-velocity error
         + baseline-relative shared support barrier
         + bounded effective-update trust region
```

其中support barrier只惩罚candidate相对同帧own target比shared更差的部分；candidate优于shared时不把它拉回。
trust region只作用于gauge-invariant effective updates `BA`，不比较raw A/B。

所有task使用同一组固定step、同一global owner scaling和同一停止规则。每一步只通过LoRA leaves求冻结policy的exact VJP；随后对
每个target用thin-QR + `16x16` core SVD重新规整为rank16，保持一套76 tensors，不累加第二adapter。solver不接收task ID、
expert identity、filename或任何task-local parameter。

solver的step count、global step scale和owner normalization只允许在一个fit-task运行profile中按finite、单调effect下降与资源上限
冻结，不能读取held closed-loop结果。held5上不做per-task early stop、不保存可继续优化的task-local optimizer state、不挑step；
给定effect tensors、shared起点和固定solver contract，输出LoRA是确定性的compiler inference，而不是held code fitting。

输出：

```text
Theta_N = 76 tensors / 38 targets / rank16 / 1,287,168 values
```

Writer可以在rollout前内部执行有限次固定迭代；它仍只运行一次并只输出一套静态LoRA，闭环中不再观看视频。

## 4. 第一道实验：privileged effect realizability oracle

### 4.1 它回答什么

在没有video effect predictor、没有shared decoder训练、没有task mapping泛化问题时，exact successful-policy effects能否通过固定
PECS solver在held5形成policy-effective LoRA？

如果答案是否定的，则从视频学习这些effects没有意义；如果答案肯定，才把问题干净地后移到action-hidden video能否预测它们。

### 4.2 数据与信息边界

- 使用train24固定fold0 held5；它们不更新任何shared参数；
- 每task从既有successful members与action-hidden videos无放回取K=2，视频和最终rollout episode分离；
- exact effect只在输入视频帧、canonical/antithetic `u=1` probes上由冻结expert生成，不读取视频action；
- validation8与Test8读取均为0；
- 复用现有stable shared、direct-earliest/latest、paired250 seeds和raw rows，不重训expert、source、observer或shared prior。

### 4.3 最小执行节奏

1. CPU/单卡只读asset audit确认5 tasks都有视频、successful member、shared/direct state和既有paired rows；
2. 一个fit task真实profile只证明candidate response可微、solver确实下降effect objective、regauge保持rank16以及显存/吞吐可接受；
3. 不用geometry/open-loop proxy选节点；固定solver步数后直接为held5各生成一套LoRA并运行一次strict paired250；
4. 报告source/shared/direct/generated、per-task、breadth、retained/gained/lost、Goal/Long及paired mismatch。

不建立中间版本，不为每个solver step跑rollout，不用union挑step。

### 4.4 通过门

沿用专家Gate 2的功能口径：

- 保留direct-latest successes至少`81/108`；
- 保留direct-latest在source-failure rows上的gains至少`58/96`；
- 保留stable-shared successes至少`33/43`；
- 5/5 breadth，且Goal与Long均非零；
- 相对source和shared都产生跨task净增量；
- pairing mismatch为0。

raw/effective LoRA cosine、retrieval、norm与inner loss只解释失败，不是通过条件。

### 4.5 失败分叉与唯一合理修正

- 若exact effect loss本身不能明显下降：只允许修正solver数值坐标/预条件，使相同目标可达；这属于同一PECS revision，不创建
  新方法。若一次修正后仍不可达，PECS实现失败。
- 若effect loss明显下降但closed-loop仍低于shared或direct retention门：说明输入视频帧上的canonical/antithetic
  policy effects不足以定义跨初始化策略。只允许一次已登记的更强functional target：加入expert在同一视频帧上从fixed noise
  完整去噪得到的action/flow trajectory，再复跑同一oracle。它把“需要latent action信息”作为明确的新信息变量，而不是暗中
  修改head/loss。
- 若更强target仍失败：停止当前zero-interaction full-LoRA compiler/solver family，不训练`R_V`；下一研究转向专家列出的
  video-to-progress/reward或video-initialized task-local RL，而不是再造compiler。

## 5. Oracle通过后的训练路径

只有第4节通过后才执行：

1. **Effect prediction**：在71 non-held + train24 fit tasks上，以correct videos训练`R_V`预测privileged effect distribution；
   multiple members、cross-episode videos和Dynamic-K提供mean/variance/presence，不训练wrong/shuffled/reversed。
2. **Held video gate**：冻结`R_V`与solver，held5只用language+action-hidden videos生成LoRA并直接strict paired250；同时报告
   learned language+scene、full和same-task-other。exact privileged effects只作sealed诊断，不选checkpoint。
3. **Joint Writer**：前两门成立后，解冻普通video/effect predictor与Stage 0 post-capture projections，通过固定solver的
   unrolled/implicit gradient接train/meta cross-episode action与task-equal simulator reward；source、backbones、Action Meta
   和solver algorithm冻结。
4. **Final development**：方法冻结后用全部授权train数据fresh训练，validation8只读deployment输入做single-checkpoint
   paired400；最终候选冻结后才运行wrong/shuffled/reversed/no-video/static/endpoints controls。

full video必须相对learned language+scene/endpoints创造新的success；否则PECS即使absolute较高，也不能证明视频过程价值。

## 6. 现有资产与预计新增面

直接复用：

- Stage 0 v3与Action Meta v3 checkpoints及event assignments；
- `95-task / 118-member` MDCO evidence authority；
- 95-task owner-resolved、target-local activation support bank；
- held5 direct experts、successful trajectories、video store、stable shared LoRA和strict250 rows；
- existing functional LoRA leaf-gradient、policy response capture、thin-QR/core-SVD、persistent evaluator与paired analysis。

第一道oracle只需要新增一个窄owner：视频帧effect capture + fixed solver。不得修改或恢复现有`compiler.py`作为PECS实现，
也不得同时搭建`R_V`。只有oracle通过后，effect predictor才成为第二个实现里程碑。

## 7. 预期裁决价值

PECS把旧Stage 1混在一起的三个问题拆开：

```text
exact successful-policy function
    -> fixed optimization能否形成LoRA？       （第一oracle）
action-hidden video
    -> 能否预测所需policy effects？           （第二gate）
predicted effects
    -> 能否经joint/outer训练稳定泛化？         （后续gate）
```

第一oracle通过并不证明EMBER成功；它只证明learned Program-to-LoRA mapping不是必要条件。第一oracle失败则是比MDCO更直接的
根本证据：即使给定exact privileged policy effects，输入视频frame support上的固定功能约束仍不能确定可闭环的完整LoRA。

## 8. 当前实现锁定

当前canonical实现把本卡的首个oracle收敛为一个窄运行面：

- 每条K2视频先由冻结Stage 0 v3 + Action Meta v3形成ordered event posterior；每个event从每条视频选一个最大posterior frame；
- effect capture与solver forward不安装Action Meta，也不读取action、state或proprio；canonical/antithetic probe均固定在`u=1`；
- owner effect保留38 owners与DCT4 horizon basis，flow effect保留两个probe的完整`50x32`；多个successful members及两条视频的
  同event frames共同形成mean/variance；
- solver从stable shared开始，固定12步、base step RMS `.0002`、inverse-sqrt decay power `.5`，无per-task early stop或持久
  optimizer；每步只更新一套LoRA leaves，随后逐target thin-QR/core-SVD regauge回rank16；
- 首个profile固定为fit ordinal71，held ordinals90--94不能用于step/scale选择；旧learned compiler运行面已从active tree退休。

实现owner为`src/ember/ecp/effect_solver.py`与`src/ember/ecp/effect_oracle.py`。clean pushed `b7c87e7`在fit ordinal71的正式
profile把effect从`3.660019`单调降至`.774046`，final/initial为`.211487`、0次回升，峰值`18,721,906,176` bytes、耗时
`153.67s`。此前constant-step候选虽降至`.748501`，但有2次回升；合同允许的唯一一次数值修正只加入所有task共享的
inverse-sqrt decay，没有修改effect target、架构、数据、步数或held信息。数值/资源合同由此冻结；held5裁决见下一节。

## 9. Local-effect oracle裁决与唯一后续

clean pushed detached `c400feb`为held ordinals90--94各生成一套完整LoRA。五项effect final/initial分别为
`.3170/.2040/.1892/.2905/.1793`且全程严格单调。strict paired250为candidate/source/stable-shared/direct-earliest/
direct-latest=`58/21/43/74/108`，candidate per global0/9/18/25/36=`31/11/16/0/0`。

candidate相对source净`+37`、相对shared净`+15`且exact McNemar `p=.02753`，因此local owner与单点`u=1` flow effect确实能形成
新的闭环能力；但direct-latest success/gain retention只有`34/108、25/96`，shared retention`30/43`，breadth`3/5`，Goal/Long
均为0。全部配对字段零mismatch。故第4.4节Gate 2失败，inner effect下降不能升级为successful-policy realizability。

该结果精确触发第4.5节唯一已登记的增强：保持support frames、K2、solver、rank16、12步、shared起点及信息墙不变，新增successful
expert从同一fixed noise按official 10-step integration产生的完整action/flow denoising trajectory作为functional target。只允许
一次资源profile与同一held5 strict250复验；不得调solver、换seed/rank、恢复learned compiler或提前训练`R_V`。若该复验仍未过
原Gate 2，停止当前zero-interaction PECS compiler/solver family。

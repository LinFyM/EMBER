# V6-LPCP CFMG Unit-Secant Finite Commitment

状态：2026-08-17 terminal scientific non-pass。简称 **USFC**。本轮从sealed LPCP fresh开始，不resume USEP；模型、
输入、LoRA、loss、optimizer和训练数据逐项保持USEP不变，唯一改变科研裁决中raw infinitesimal gradient门与
actual finite deployed commitment门的主从关系。

## 1. Decision

USEP clean `6033330` fixed world3已经解决它实际针对的endpoint尺度问题：task38相对次大task的gradient norm从
CFMG的`58.73x`降到`6.1538x`，三task mean gradients两两全正，raw shared coverage=`3/3`。同一个未经调参的
actual Adam `j0`使task4/34/38全部12个normalized deployed endpoint margins严格下降，并产生非零q/v/action
effective BA与fixed-action response。

USEP仍依其预注册门终局，因为task34四个raw view gradients只`2/4`沿本task算术均值下降。本轮不改写这个历史，
但新增stage localization证明该raw门不是当前有限部署更新的必要条件：task34四组K4从endpoint、temporal、
K-set、layer/token M2P到pre-gate content grid的pairwise cosine约`.984--.986`、energy约`.987--.988`；carrier
A/B/BA和由共同content形成的`delta-B A0`也都约`.99`一致。raw冲突发生在这些共同BA经过四个condition的局部
policy/action Jacobian回传时，而不是视频Program、memory、K聚合或LoRA Value已经分裂。

EMBER最终选择的是finite LoRA在closed-loop中的真实效果，不是raw parameter-gradient cosine。既然同一个原生
有限更新已经越过12/12 deployed margins，再为修复raw局部指标重做视频前端或LoRA架构会针对错误接口。因此
USFC只检验：unit-secant校准后的同一个actual finite commitment能否在full24形成nonzero shared checkpoint，
并在strict400带来真实收益。

## 2. Unchanged model and objective

```text
exact language + K=4 ordered action-hidden same-task videos
 -> one carrier-exact native PI0.5 context forward
 -> 37 layer-matched one-way memory tokens
 -> per-video signed temporal program + permutation-invariant K-set
 -> layer/token M2P content grid
 -> zero payload gate -> native rank16 B residual
 -> concatenate frozen LPCP rank16 carrier
 -> one complete rank32 LoRA -> frozen policy rollout
```

每个selected matched state仍使用同一个pair-local unit-secant objective：

```text
D+ = mean_valid (a_hat - a+)^2
D- = mean_valid (a_hat - a-)^2
s  = sqrt(mean_valid (a+ - a-)^2)
J  = softplus((D+ - D-) / s)
```

`s` detached且逐state计算。四个disjoint correct K4 views等权、active tasks等权；AdamW、LR、betas、clip、
`j=0..10` backtracking、rank32、frame stride5、两paired states、八occupancy strata、RNG、dtype、batch与信息墙
全部不变。没有task ID、gradient norm reweight、PCGrad、MSE分母、epsilon/temperature、额外forward、第二套LoRA、
expert route或生成后RL。

## 3. The single changed falsifier

USEP把以下两项都设为硬门：

1. raw equal-view task gradient必须对本task四views达到`4/4`局部下降；
2. 一个actual native Adam/backtracking candidate必须让所有task-view deployed margins严格下降。

world3证明第1项失败而第2项通过。USFC把第1项降为stage diagnostic，并只保留第2项为训练更新的硬合同。这不是
忽略same-task鲁棒性：每个candidate仍必须在每个view上重新完整compile LoRA并以同一matched action panel验证
真实finite margin，任一view不下降就拒绝整个global update。它也不是post-hoc选scale：搜索序列、首个接受规则和
全部数值均与USEP冻结一致。

## 4. Why no new video or LoRA architecture yet

- task34 action-hidden画面没有支持“不同合法子任务顺序相消”的初步猜测，不能据此搭partial-order新前端；
- stage localization显示视频Program和direct effective BA已经跨K4条件高度一致，改temporal/K-set/M2P不针对最早
  分裂接口；
- duplicated carrier `A0`、carrier BA与`delta-B A0`本身也高度一致，当前没有证据授权rank8、full A/B或另一套
  factor transport作为本轮变量；
- actual `j0`已经12/12下降，说明当前native finite route具有共同可达步。只有full24或strict失败后，才能据其
  最早接口决定是否需要condition-specific functional transport，而不能用raw cosine预设答案。

memory token与rank8仍是开放的长期机制，不因本轮保持现图而被否定。

## 5. Fresh identity and implementation gate

USFC使用fresh config/checkpoint/launch/completion/evaluation family，USEP checkpoint不得resume。canonical代码只需
切换identity和authority；`unit_secant_endpoint_preference`、Writer model、forward与参数枚举不得变化。CPU必须
证明：

1. config记录raw four-view coverage只作diagnostic，global finite all-view descent仍为硬门；
2. formula、mask、detached denominator、step0、information wall、one-forward和checkpoint fresh拒绝合同继续通过；
3. model state/parameter count/forward owner与USEP相同，无并行runtime或兼容fallback；
4. 正常BF16/TF32、batch/kernel低位差异不构成逐元素复现门。

## 6. Formal cycle1 decision

实现门通过后，不重复必然相同的fixed world3或CFMG held representation screen；只允许一次从sealed LPCP fresh开始
的full24 cycle1。此前world3、CFMG held8和本轮stage localization只授权formal，不能选择checkpoint。

cycle1必须报告24 tasks/48 paired states/96 rollouts、active tasks/suites、每task四view、unit-secant action RMS、
task gradient norm/cosine、actual candidate search、parameter delta、q/v/action/fixed-action response、禁读与wall。
唯一可接受更新仍是让所有active task的全部四个deployed margins严格下降的同一个candidate；若11个candidate
均失败、final delta为0、任一rank不一致或native response为0，本轮终局，不补raw-gradient solver、PCGrad、
task weights、epsilon、temperature、rank、scale、seed或其它小扫。

若cycle1产生nonzero checkpoint，立即做single-checkpoint K4 strict paired400。只有同时满足correct至少142、
breadth至少7、相对LPCP143 lost不超过15且gained不少于lost，才允许exact cycle2。cycle2必须从同一formal
checkpoint exact-resume并完成相邻checkpoint稳定性裁决；两cycle约145或更高、churn<=20、Jaccard>=.85且最终
lost<=10才具稳定资格。首次达到约145且retention过门立即补same-task-other、wrong、shuffled、reversed和
no-video；correct必须沿有用policy direction明显更好。

## 7. Negative boundary

若full24不能接受finite global update，淘汰“CFMG memory/content grid + unit-secant endpoint credit + 当前actual
Adam finite commitment”作为一次shared full24更新，不等于raw gradient solver值得恢复。若接受但strict明显下降，
说明finite train endpoint preference仍没有选择held closed-loop有用方向；下一轮必须根据per-task gained/lost和
BA/action最早分裂接口设计condition-specific functional transport，而不是继续美化内部geometry。

任何失败都不否定memory token、dynamic K/few-shot、rank8、完整A/B LoRA生成、正确视频因果目标或未来独立的
task-local RL。

## 8. Terminal result

clean `db7ab24`在gpu02 world6完成full24 cycle1：24 tasks、48 paired states、96 rollouts，candidate/reference
success=`33/32`，gains=`3/2`，5个active tasks覆盖四suite，cycle=`480.284s`，0禁读/OOM/nonfinite。unit-secant
把task38相对次大task的梯度支配保持在`7.6807x`；active tasks pairwise gradient cosine mean=`-.00834`。

exact Adam `j0`的delta L2=`.242816`，让20个task×view margins中的17个下降。task4/25/34/38均为4/4；只有task19
为1/4，mean/max harm=`2.249e-5/9.637e-5`。其余10个缩放候选没有改善为20/20，最终按本设计原始硬门恢复exact
LPCP，saved delta=0，故strict400没有运行。精确artifact为run root下`usfc_full24_terminal_adjudication.json`。

该结果终局否定本设计的“all-view monotone是保存更新的绝对前置条件”，但不否定unit-secant、CFMG memory、
direct Adam候选或其潜在closed-loop效果。owner随后明确：20/20应保留为理想诊断，不能让未验证过的endpoint
surrogate永久阻止真实闭环裁决。USFC不得resume；fresh successor USDC只改变commitment acceptance。

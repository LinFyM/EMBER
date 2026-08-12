# Negative-Preserving Candidate Guard

状态：2026-08-12 matched world3 reprofile的20项机制门全部通过；随后同commit部署B8/16/32均稳定并按真实
LoRA/s选择B32。canonical config已封存profile/deployment evidence并打开formal fresh`0→5`。本设计仍是唯一
active successor，没有保留WQ-PCUG并行可执行版本。

formal fresh`0→5`已由clean pushed `f8491e9`、gpu02物理3/4/5 world3完成并exit0；首次空启动仅因launcher
沿用旧SKNC的`num_workers=2`而在模型加载前被NPCG sealed `num_workers=0`拒绝，retry没有改变科学合同。
五个macro均无OOM/nonfinite，step seconds=`551.61/560.17/565.03/530.64/555.81`；first-stable bank从
`12→13→15→15→16`增长，macro5 guard rows/rank=`29/29`、feature rank=`34`、energy=`.38951`、cosine=
`.60652`、blind negative ratio=`.03234`且closure约`3e-13`。macro5 checkpoint已完整写出；下一步是该同一
checkpoint的strict paired correct400，尚无absolute closed-loop结论。

首个NPCG profile来自clean pushed `ef0008d`、gpu02物理3/4/5 world3：Phase A=`44.67883s`、total=
`554.57395s / 1.15868x SKNC`，paired outcomes与WQ-PCUG完全一致。final negative ratio=`.03524`，三类各
`8/8`达门，negative correction violation=`0`；rank=`33`、energy=`.35360`、alignment=`.52158`，LoRA、
effective BA与fixed-action closure均通过。唯一失败的protected Program ratio=`1.5831e-4>1e-5`，绝对残差
`3.43e-10`。代码检查定位到right-hand side与correction的FP32 matmul允许TF32，而真实constraint read显式关闭
TF32；这是同一公式的数值实现不一致，不是放宽门或科学sweep，首个root保留且无checkpoint。

full-FP32窄修后的clean pushed `4156012` matched reprofile保持paired outcomes完全不变，20/20 checks全过：
Phase A=`44.34676s`、total=`554.99255s / 1.15955x SKNC`；protected Program ratio降到`5.7508e-8`，
negative ratio=`.03524`且wrong/shuffled/reversed各`8/8`，rank33、energy`.35360`及LoRA/effective BA/action
closure全部健康。B8/16/32部署profile均包含67-frame longest video、0 OOM/nonfinite，LoRA/s分别为
`.46856/.47052/.47144`，选择B32；这只证明机制与部署吞吐，尚不证明closed-loop提升。

## 1. Latest evidence and earliest failure

Work-Queue PCUG在clean pushed `d799758`、gpu02物理3/4/5 world3完成了此前从未执行过的完整macro0：

- Phase A=`44.74125s`，是matched SKNC的`.09348x`；24 tasks完整、每rank 8个，queue claim总计`.00558s`；
- total step=`558.05862s / 1.16596x SKNC<1.5x`，无OOM、nonfinite或checkpoint；
- base/candidate各48 rollouts，base/candidate successes=`34/33`，7个discordant states、3 gains、4 losses；
- harmful tasks=`3`，跨Object与Goal两个suites；stable-success tasks=`12`，paired reward evidence有内容；
- final correct guard有15 rows/rank15，投影后feature rank=`33>=24`、保留能量`.76492`、guard violation=`0`；
- Program、effective BA与fixed-action protected closure均为零，四suite unprotected action response非零；
- 19项机制checks中只有`negative_null=false`。

失败发生在blind solve之后而不是视频读取、functional cotangent、candidate rollout或correct guard：

- full48 blind `D0`对negative的预测motion ratio=`.03991<.15`；
- 旧final closest projection只把persisted/stable/harmful **correct** rows作为零约束；
- final update的negative ratio因此升到`.50179`，wrong/shuffled/reversed各`0/8` tasks达门；
- paired probe之前的全部数值和outcomes有效，不能把这个结果归因于queue、wall、显存、reward稀疏或无效candidate。

所以最早失效接口是：`correct paired guard correction`破坏了`full48 blind solve`已经得到的negative-video抑制。

## 2. Single causal variable

保持blind `D0`、paired outcomes和guard membership不变，只改变final correction的可行域。

记：

- `D0 in R^(d x p)`：full48 blind shared Program update，`d=256`；
- `N in R^(24 x d)`：本macro的wrong/shuffled/reversed condition rows；
- `G in R^(m x d)`：persisted first-stable rows与当前stable/harmful correct rows；
- `C in R^(d x p)`：对`D0`的final correction。

旧WQ-PCUG使用`D_old=P_null(G) D0`，其correct guard闭合，但没有限制投影对`N`的影响。

新方法解唯一的最小修正：

```text
C* = argmin_C ||C||_F^2
     subject to N C = 0
                G (D0 + C) = 0
D1 = D0 + C*
```

因此`N D1 = N D0`：paired correct guard不能重新打开blind solve已压低的negative response。同时`G D1=0`，
已观察到的stable success和harmful task仍被保护。这个变量改变的是correction subspace，不改变reward定义、
negative target、update scale、condition map、compiler、rank或训练样本。

实现上令`Q_N`为`Null(N)`的正交基，`H=G Q_N`。在restricted guard rows可行时：

```text
Z* = -(G Q_N)^+ G D0
C* = Q_N Z*
```

只对小feature matrices做FP64 SVD/rank solve；large `D0`、motion与correction保持FP32；约束乘法关闭TF32并做一次
固定residual refinement，使solver和实际Program full-FP32 constraint read具有同一数值语义，不扩policy/video dtype，
不增加forward、host scan、hash或per-task route。若restricted guard不满秩或任一closure失败，profile直接non-pass。

## 3. Why this is not union-nullspace compression

不使用`D1=P_null([G;N])D0`。把15个correct guards和24个negative rows都作为final homogeneous rows，可能把
full48 condition evidence的可用rank从48压到约9，并无必要地删除`D0`在negative row span内但不改变negative
response的成分。历史uniform rank14已证明support compression本身会损伤旧能力。

本设计让**修正量**位于`Null(N)`，而不是让完整update位于`Null(N)`。它只撤销实测有害/稳定correct rows上的
motion，保留blind update其余结构和原negative response，是更小且直接针对证据的改变。

## 4. One-shot knowledge and temporal necessity

Writer部署合同仍是exact task language + exactly one action-hidden teacher video，rollout前一次生成完整38-target
rank16 LoRA。训练中correct video与B20 action episodes跨episode错开，不能逐帧复制动作。

full48条件把每个correct process与一个wrong/shuffled/reversed counterfactual共同放进blind solve：

- wrong改变对象/关系或目标过程；
- shuffled保持frames但破坏阶段连续性；
- reversed保持frames但反转有向因果；
- static language或frame-set bypass会同时作用到这些negative rows，因而被zero-response条件压制。

WQ-PCUG证明blind solve原本已把negative ratio压到`.03991`，但final correct-only guard丢失了该性质。本设计不是
人为破坏negative LoRA，而是保证reward-derived correct protection不覆盖已经学到的视频时序辨识。最终有效性仍
必须由single-checkpoint closed-loop absolute和correct相对wrong/shuffled/reversed/no-video的真实成功率证明。

## 5. Multi-task coexistence and policy-effective write

- `D0`继续来自train24等权B20 functional cotangent，保留v6-fast/PICK-GC的condition、Core/Program和native
  rank16 compiler；
- actual base/candidate paired rollouts仍决定哪些correct rows是harmful或stable，不用surrogate reward magnitude；
- `G D1=0`保护当前及历史first-stable correct support；
- `N D1=N D0`保护视频因果抑制；
- 最小`C`让未受约束tasks保留尽可能多的blind useful direction；
- 不做global reject、task route、expert bank、checkpoint融合、LoRA平均、rank reservation或regeneration。

negative rows只参与训练时shared correction，不进入checkpoint bank，也不在部署时要求第二条视频。部署仍只生成
一套correct-video LoRA。

## 6. Canonical implementation boundary

1. 原位替换`v6_candidate_guard.py`中的correct-only projection，不新增parallel trainer/module；
2. candidate生成、K2x2 rollout、classification和success-key bank在final correction之前完全不变；
3. final solver显式接收当前24 negative features，并报告restricted guard rank、negative-correction closure、
   final negative preservation、energy/alignment与analysis feature rank；
4. profile继续用真实final `D1`做Program、LoRA、effective BA和fixed-action验证；
5. schema/config/checkpoint/eval family fresh-incompatible替换WQ-PCUG；历史由Git和frozen artifact保存；
6. formal仍使用相同single Program memory与one complete LoRA，不保存task-local guards或第二套adapter。

## 7. Fast falsification

### 7.1 CPU and synthetic gates

- 当`G D0=0`时solver严格返回`D1=D0`；
- 一般情形满足`N(D1-D0)`数值闭合、`G D1`数值闭合且是restricted minimum-norm correction；
- rank-deficient但一致rows按数值rank处理；不可行rows fail closed，不降tolerance；
- paired outcomes、task/video/action rows、RNG与WQ-PCUG完全不受final solver变化影响；
- no extra policy/video forward，negative policy forwards仍为0；
- checkpoint只保存shared Program与first-stable correct bank，fresh/resume schema不误载WQ-PCUG。

### 7.2 One discarded macro0

从historical v6-fast、zero Program、empty bank运行与sealed WQ-PCUG完全matched的world3--6 live profile。GPU按
live gpu01/gpu02状态用一个节点至多6张真正合适A40，不等待6张；低利用率小占用卡有足够峰值余量即可共享。

必须同时满足WQ-PCUG全部原hard gates，尤其：

- Phase A、exact 48 pairs、discordance/harmful suites/gains与candidate四suite response继续通过；
- paired outcome summary应与sealed WQ macro0一致，否则说明final-only改动泄漏到candidate probe；
- `G D1` closure、`N(D1-D0)` relative motion都`<=1e-5`；
- final negative/unprotected ratio`<=.15`、至少18/24 tasks且wrong/shuffled/reversed各至少6/8达门；
- projected analysis rank`>=24`、energy/D0`>=.25`、alignment为正；
- protected Program/LoRA/fixed-action closure、四suite unprotected response与predicted-observed closure通过；
- total wall相对matched SKNC`<=1.5x`，无OOM/nonfinite。

任何一项失败即淘汰NPCG，不扫rank、scale、threshold、SVD tolerance、negative weight、seed、dtype、world size或
pair count，不通过降低negative门或projected-rank门救结果。

## 8. Deployment and formal decision

profile与B8/16/32 deployment已全过并选择B32；下一步fresh `0->5`并立即strict paired correct400。继续`5->10`
仍要求macro5 correct>=142、breadth>=6、相对immutable macro0 lost<=8且gained>lost、至少3 suites不降、最大
单task净增不超过全部正净增`.5`，并且每macro两类closure、rank、energy和paired evidence不坍缩。

首次correct>=144才补same/wrong/shuffled/reversed/no-video。最终成功仍是同一single checkpoint correct>150，
correct严格优于全部negative controls，same-task-other至少保留correct的`.9`，且不依赖checkpoint/task轮换。

## 9. Rejected alternatives

- 不放宽`.15` negative门：失败正是视频因果性丢失；
- 不把`[G;N]`直接做完整update homogeneous nullspace：会无必要压缩support；
- 不增加negative闭环rollouts：当前最早失败已由action-free negative condition rows精确定位，新增72+ rollouts
  改变cost与科学变量；
- 不减少stable/harmful guards：这会牺牲已经验证的correct support保护；
- 不改D0 solver、step scale、rank16 compiler、B20、K2、queue或success classification；
- 不转few-shot、expert routing、language gate或新encoder：当前证据支持先修final constraint composition接口。

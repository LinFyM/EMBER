# V6-LPCP Direct-Factor Matched-Batch Stratified Occupancy Preference

状态：2026-08-16 terminal mechanism non-pass；不得full24、strict、cycle2或小扫。简称`MB-SOP`。本轮从sealed LPCP
fresh启动，完整保留LPCP/DJNFR已经通过的video-language carrier、K-set、八direct native-factor heads与rank16
LoRA部署图，只替换successful-occupancy credit panel的构造。

## 1. 决策与最早失败接口

DF-SOCP证明完整winner occupancy能让task9/15/18都形成强train/held跨video共同方向，但其positive来自rollout时
动态B2/B1 winner action，negative来自事后B8 loser query。task9/15的正常BF16 batch-shape差异分别是名义策略
contrast的`1.086x/1.693x`。同时穷举26/65/44个replans使wall达到DF-PCSP的
`3.083x/5.335x/3.887x`，成本主要来自四view×Nmc4 functional CFM，不是action query。

最早失败接口因此是：

```text
successful policy identity + successful occupancy
 -> B2/B1 stored winner actions versus B8 loser actions
 -> batch-shape-contaminated exhaustive preference panel
 -> expensive and scientifically ambiguous Writer cotangent
```

本轮唯一主要因果变量是：**把完整成功occupancy变成matched-batch、时间分层、策略分歧优先的稀疏credit panel。**
不同时改carrier、memory token、rank、LoRA mapper、optimizer、Nmc、四view、rollout数或source policy。

## 2. 已比较但不采用的方案

1. 只把winner也按B8重查、仍穷举全部replans：修复可比性但保留实测`3--5x`瓶颈，不满足效率优先；
2. 每条轨迹均匀取固定8个索引：便宜且覆盖时间，但可能大量选择两臂动作几乎相同、不能解释策略差异的states；
3. 全轨迹只取top-8 action differences：信息量高，但可能全部挤在抓取或收尾的单一阶段，丢失有向程序覆盖；
4. 每个state真实branch rollout到终点：因果性最强，但会把96 rollouts膨胀为数千次，当前成本不可接受。

最终选择“8个等进度strata，每stratum取matched两臂action RMS最大的一项”。它同时保留全程覆盖和策略可辨识性，
且不依赖任意动作阈值或结果后超参选择。

## 3. 完整保留项

- exact language + dynamic `K=1..4` action-hidden ordered videos；formal固定K4；
- stride5、每video内部保序、videos间置换不变K-set aggregation；
- 同一次真实图像+语言+50 Action probes forward、18层LPCP carrier；
- `X=M*RMSNorm(L)/sqrt(256)`与八个DJNFR direct factor-shape heads；
- 完整38-target rank16 public LoRA，step0/no-video/constant exact LPCP；
- 两个paired states、AS139 reference与LPCP+direct-head candidate、四个互斥correct K4 views；
- exact model/state restore、每lane一次hard reset+settling、每臂deterministic soft reset；
- train24 task等权、AdamW、Nmc4、BF16/TF32与single-checkpoint部署；
- Writer rollout前运行一次，部署时没有reference、reward、counterfactual policy或第二套LoRA。

## 4. Matched-batch双臂action panel

每个discordant pair仍只保留winner真实访问的全部observations、policy-noise seeds与executed-prefix lengths。rollout
结束后，把这些records按固定trajectory/replan顺序组成同一panel；先装reference LoRA、再装candidate LoRA，分别用
完全相同的physical B8 chunks运行：

```text
a_ref[j]  = reference_policy(o_winner[j], xi[j])
a_cand[j] = candidate_policy(o_winner[j], xi[j])
```

两次query的observation顺序、noise、num_steps、BF16 autocast、batch sizes与最后一个partial batch逐项相同。winner
label只决定哪一个是positive；不能复用stored winner action进入loss。stored actions仅用于确认这条occupancy确实由
成功臂访问，以及保留原executed-prefix length。这样不追求batch1逐元素复现，而是比较相同高效执行合同下的两套
policy。

## 5. 时间分层的informative occupancy选择

对每条discordant winner trajectory的`T`个replans设`S=min(8,T)`。第`s`个stratum是半开区间：

```text
[ floor(s*T/S), floor((s+1)*T/S) )
```

在每个stratum内，只按winner实际执行的`h_j`步计算：

```text
d_j = RMS(a_winner[j, :h_j] - a_loser[j, :h_j])
```

选择`d_j`最大的replan；严格tie取较早索引。所有strata等权，不按`d_j`加权；一个task有两条discordant成功轨迹时
两轨迹仍各占二分之一。由固定已知轨迹长度，task9/15/18的functional credit pairs从`26/65/44`变为
`8/16/8`。全部26/65/44 states仍只做便宜的双臂action inference，用于选择和审计，但不进入CFM backward。

等进度strata不是声称任务恰有8个语义阶段；它只是一个预声明的覆盖约束，避免top-k全挤在局部。被选state必须同时
具有策略分歧，因而比盲目均匀抽样更接近“哪部分policy behavior可能解释两臂结果不同”。final binary success仍不
证明每个局部action的严格因果贡献，这是本轮证据边界。

## 6. Objective与多video共同知识

对每个selected state，winner/loser requery actions共享同一CFM time与Gaussian noise：

```text
J_i(v) = mean_strata mean_m softplus(
    CFM(a_winner | o, lambda_v, t, eps)
  - CFM(a_loser  | o, lambda_v, t, eps))
```

trajectory、四个disjoint correct K4 views与active tasks逐级等权。同一matched selected panel跨四views复用；四组
不同teacher videos必须各自生成支持同一时间分布成功行为的LoRA，不能分别挑state、生成negative或平均LoRA。

teacher video与reward trajectory跨episode、跨初始化，且teacher actions从未读取。language定义对象/关系/目标，
动态有序video是新增LoRA的唯一Value；正确顺序仍经LPCP temporal Program进入direct heads，constant保持identity。

## 7. 预期吞吐

B8每次处理4个winner/loser CFM pairs。四views、Nmc4下，task9/15/18 functional forward/backward理论上从
`112/272/176`降为`32/64/32`，约`3.5x/4.25x/5.5x`减少。双臂action inference会把forward从
`4/9/6`变为`8/18/12`，但DF-SOCP实测该部分仅`2.8--6.7s`，不是主成本。仍锁B8，不靠batch1、扩dtype、重复
same-arm forward或逐tensor扫描换数值一致。

## 8. Canonical实现边界

- 原位替换DF-SOCP action-panel owner、config/checkpoint/completion/evaluator schemas，不保留runtime strategy switch；
- 新接口一次返回每条trajectory的selected indices、matched winner actions、matched loser actions与审计metrics；
- selection在CPU detached actions上完成，不创建额外Writer或policy backward graph；
- selected batch继续按完整winner/loser pairs microbatch，不能拆散pair；
- 仍复用同一四view Writer gradient owner与trajectory/view/task权重；
- fresh incompatible，不能加载DF-SOCP smoke factors或任何旧optimizer state；
- 旧DF-SOCP由Git、terminal design与三个run roots保存，不保留可执行平行config或registry family。

canonical实现已原位完成：旧DF-SOCP executable config、checkpoint/completion/evaluator family与runtime函数均被
fresh MB-SOP schema替换；双臂按完全相同batch-size序列query，stored rollout action不再进入functional batch；
selection owner独立完成等进度分区、区内max-disagreement、matched batch组装与审计metrics。定向CPU=`48 passed`、
全量CPU=`402 passed`，compileall通过，architecture guard无hard violation；`reward_cycle.py`净缩减且新panel owner
复杂度低于硬门。这些只证明运行图与合同，不提供真实GPU机制或closed-loop结论。

## 9. Formal前固定机制门

固定task9/15/18，三项必须全部满足：

1. outcomes复现`2/1`、`2/0`、`1/2`，原完整occupancy counts仍为`26/65/44`；
2. reference与candidate query batch-size序列逐项相同，0 stored action进入functional preference；
3. selected pairs严格为`8/16/8`，每trajectory每stratum恰一项且索引递增、覆盖首尾各八分之一区域；
4. matched winner/loser selected action RMS finite/nonzero，八个strata不因threshold被丢弃；
5. preference objective、LoRA cotangent与八head gradient finite/nonzero，真实Adam step后margin下降；
6. q/v/action native BA与fixed-action response非零；
7. 每anchor train four-view BA cosine/energy至少`.40/.55`；
8. validation8 aggregate至少`.30/.48`、至少6/8 tasks过`.15/.40`，raw factor cosine至少`.30`、action cosine
   至少`.15`、held/train BA L2至少`.30x`；
9. natural/reversed BA relative-L2至少`.50`，constant/natural不超过`.005`；
10. 0 forbidden read/OOM/nonfinite；每anchor cycle wall不超过matched DF-PCSP的`2.5x`。

任一失败即终局，不换anchor，不调strata数、selection规则、Nmc、LR、rank、scale或batch。机制门只能决定是否值得
full24，不能选择最终方法。

## 10. Full24与真实性能裁决

三anchor全过后，从sealed LPCP fresh做一次full24 cycle1并立即K4 strict paired400。cycle1须同时满足correct至少
`142/400`、breadth至少7、相对LPCP143 lost不超过15、gained不少于lost、无suite清空且post-train三anchor/held门
不坍塌，才允许exact-resume cycle2。稳定资格仍是相邻两个single checkpoints均至少142、均值至少145、churn不超过
20、Jaccard至少`.85`、final lost不超过10且gained不少于lost。首次约145且retention过门立即补same-task-other、
wrong、shuffled、reversed与no-video。

若机制全过而strict仍低或换手，最早接口后移到binary outcome对局部action的不可识别性或shared multi-task update，
下一步应测试真正branch value/其它reward，而不是继续改strata或生成拓扑。

## 11. 负结果边界

本轮只检验“LPCP/DJNFR direct LoRA + exact final-success pair + matched-batch dual-arm actions + eight-stratum
max-disagreement occupancy panel + four-view one-cycle update”。负结果不否定memory token、rank8、few-shot、生成LoRA、
完整matched occupancy、真正state branching或未来生成LoRA后的task-local RL。

## 12. 固定三anchor终局结果

clean commit=`ad65347ddcdcc23b745548ad043d3d14c42582d2`在gpu01物理`0/1/2`并行完成task9/15/18。
三任务outcomes、完整occupancy与selected pairs均精确复现为`2/1,26,8`、`2/0,65,16`、`1/2,44,8`；两臂
query batch序列逐项相同，0 stored rollout action进入functional preference，八head及q/v/action native写出全部非零，
0 forbidden read/OOM/nonfinite。

| task | wall / DF-PCSP | selected / complete action RMS mean | margin before -> after | train BA cos/energy | held8 BA cos/energy | held/train L2 |
|---|---:|---:|---:|---:|---:|---:|
| 9 | `1.6555x` | `.001635/.001309` | `-.0001479 -> -.0001779` | `.8520/.8540` | `.7818/.7808` | `.1096x` |
| 15 | `2.1190x` | `.001599/.001236` | `-.0003242 -> +.0000303` | `.6937/.6943` | `.7704/.7810` | `.3525x` |
| 18 | `1.5423x` | `.005170/.001880` | `-.0022745 -> -.0018652` | `.6988/.7227` | `.7747/.7826` | `.8381x` |

因此matched panel与时间分层同时解决了DF-SOCP的批形混杂和`3--5.3x`成本：三个wall都低于`2.5x`，selected
states比完整occupancy均值更可辨，且跨video/native写出健康。但task15/18的同一retained panel、同一flow noise、
同一view0在真实AdamW step后margin反而增加；task9虽下降，held/train又只有`.1096x`。按预注册门终局，不启动
full24或strict。

为区分视频梯度冲突与更新器失真，只增加一次无额外forward的四view flat-gradient几何诊断。task9/15/18的
pairwise cosine mean/min分别为`.423/.291`、`.339/.169`、`.343/-.173`，但每个view与等权均值的cosine minimum仍为
`.695/.629/.601`，三任务共同下降覆盖率均为`4/4`。所以原始等权均值本身已是所有正确视频条件的一阶下降方向；
最早缺口不是缺少共同梯度，而是：

```text
four correct-video functional gradients
 -> equal raw mean is common descent for 4/4 views
 -> coordinate-preconditioned finite AdamW parameter step
 -> task15/task18 retained functional margin ascends
```

下一变量应只检验descent-preserving parameter commitment，在保持Adam候选步全局半径的同时保留raw shared gradient
方向；不能借此回改carrier、K-set、memory、rank、FactorHeads或matched credit panel。精确终局artifact为task9 run root下
`mbsop_terminal_adjudication.json`。

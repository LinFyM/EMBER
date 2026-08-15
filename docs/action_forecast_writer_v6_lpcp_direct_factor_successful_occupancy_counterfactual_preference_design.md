# V6-LPCP Direct-Factor Successful-Occupancy Counterfactual Preference

状态：2026-08-16 active design authority，implementation pending。简称`DF-SOCP`。本轮从sealed LPCP fresh启动，
保留LPCP/DJNFR已经通过的视频carrier、K-set与direct native-factor LoRA生成图，只改变long-horizon reward怎样在
exact shared observations上形成正负动作credit。

## 1. 决策与最早失败接口

DJNFR证明有序video-language Value能跨video稳定写成native q/v/action LoRA，但selected-success-only full-trajectory
credit在strict136上改坏persistent failures。DF-PCSP随后修复了一个此前未发现的工程混杂：相同seed hard reset并不
产生相同图像；正确合同必须在同一LIBERO model上deterministic soft reset controller/observables并恢复同一动态
state。该修正后，task9/15/18均产生真实discordant success、preference descent与非零LoRA/action响应。

但DF-PCSP把数百步后的final success全部归因给两臂分叉前第一段约`6.6e-4--7.4e-4` RMS的动作差异。三个有效train
anchors只有task15全门通过：task9 held/train BA只有`.105x`，task18同task四个video views只有`.290/.428`。因此
最早失败接口是：

```text
exact paired final success
 -> first shared action prefix receives all long-horizon credit
 -> task-dependent direct-head update
 -> inconsistent same-task-video or held commitment
```

本轮唯一主要因果变量是：**不再只比较第一prefix；沿成功臂真实访问的全部replan observations，逐状态查询失败臂
在完全相同observation与policy-noise下会做什么，再以成功动作相对这个counterfactual失败动作构造全轨迹paired
preference。**

## 2. 完整保留项与非目标

- exact task language + dynamic `K=1..4` action-hidden ordered videos；formal固定K4；
- stride5、每video内部保序、videos间置换不变K-set aggregation；
- AS139/LPCP底座、同一次真实图像+语言+50 Action probes forward、18层LPCP carrier；
- `X=M*RMSNorm(L)/sqrt(256)`与八个DJNFR direct factor-shape heads；
- 完整38-target rank16 public LoRA、step0/no-video/constant exact LPCP；
- 两个paired states、AS139 reference/LPCP+direct-head candidate两臂、四个互斥correct K4 credit views；
- train24 task等权、AdamW、Nmc4、BF16/TF32、source policy、split、信息墙与single-checkpoint部署；
- 每lane一次hard reset+10步settling；每臂deterministic soft reset后恢复同一captured qpos/qvel；
- Writer rollout前运行一次，部署时无reward、reference、counterfactual policy、expert或第二套LoRA。

本轮不加memory token，不改rank、scale、carrier、K、optimizer、env rollout数或source-action loss；不resume DF-PCSP
task checkpoints。memory/rank8仍是开放方法，但当前最早失败在credit assignment，不应同时改生成拓扑。

## 3. Successful-occupancy counterfactual replay

对每个task的两个exact paired states分别运行reference和candidate。tie严格为0。若只有一臂最终成功，记成功臂为
winner、失败臂为loser。winner trajectory已经保留每个replan的：

```text
o_ij             # winner实际访问、已进入policy的observation
a_ij+            # winner在该observation生成的normalized action chunk
h_ij             # 该chunk实际执行的1--5步
xi_ij            # 当时的policy flow-noise seed
```

rollout结束后，把loser LoRA装入同一frozen source policy；对所有`o_ij`按B8批量、使用同一`xi_ij`与10 flow steps，
一次性生成：

```text
a_ij-cf = loser_policy(o_ij, xi_ij)
```

这些queries不step environment、不读取新reward，也不重新看teacher video。winner和counterfactual loser动作共享同一
observation；失败轨迹后来访问的不同occupancy不进入objective。第一replan只是其中一项，不获得特殊权重。

## 4. Objective与权重

对任一correct K4 view生成的candidate LoRA `lambda_v`，winner/counterfactual loser在每个成功occupancy state上
共享同一CFM time与Gaussian noise：

```text
ell_ij+(v,m)  = CFM(a_ij+[:h_ij]    | o_ij, lambda_v, t_ijm, epsilon_ijm)
ell_ij-(v,m)  = CFM(a_ij-cf[:h_ij]  | o_ij, lambda_v, t_ijm, epsilon_ijm)

J_i(v) = mean_j mean_m softplus(ell_ij+(v,m) - ell_ij-(v,m))
```

一个task内1或2个discordant pairs等权；每pair内所有winner replans等权，不能让长trajectory按chunk数量支配；四个
disjoint K4 views分别求完整Writer gradient后等权；最后active tasks等权。counterfactual actions全部detach，只有
`lambda_v -> policy CFM -> LoRA cotangent -> shared direct heads`接收梯度。

跨video views复用同一winner/counterfactual action panel。这样四个不同correct teacher sets必须把各自有序视频证据
编译成支持同一成功on-policy行为的LoRA；不为每个view生成不同negative，也不平均LoRA。

## 5. 为什么该变量针对DF-PCSP而非重复历史方法

- 相对DJNFR selected-success：保留其完整成功occupancy与trajectory等权，但每个状态都有同状态loser negative；
- 相对DF-PCSP：保留exact common-state preference，但从第一prefix扩展到整个成功执行过程；
- 相对OPPP138：不比较来自不同rollouts的observations，不用LOO advantage，也不把失败轨迹后来状态当负例；
- 相对true branch-and-rollout：不从每个replan重新跑到horizon，因此不把96个rollouts膨胀成数千个；
- 相对固定8/16阶段抽样：不引入任意阶段数量或漏掉关键动作；直接复用已经存在的完整成功replay。

final success仍不是逐动作因果证明，但它现在监督成功策略在整条成功occupancy上的一致行为，而不是把所有结果压到
第一个微小动作差。若本轮仍跨task/video不稳定，就说明仅靠两臂最终binary outcome不足以提供shared Writer所需的
long-horizon credit，下一步应进入真正branch value或其它reward formulation，而不是继续局部改prefix。

## 6. 视频、语言、高层知识与LoRA

部署数据流不变：

```text
exact language + ordered action-hidden video set
 -> LPCP layerwise Action-probe temporal Programs
 -> permutation-invariant K-set Value M
 -> language-conditioned joint payload X
 -> eight direct native factor heads
 -> one complete rank16 task LoRA
```

language定义对象、关系与目标；`M=0 -> X=0`，language不能独立写新增LoRA。正确顺序通过layerwise temporal deltas与
Program进入M；reverse/shuffle必须重新完整forward并改变LoRA。reward actions来自与teacher跨episode、跨初始化的
frozen policy rollout，不是teacher actions，因此Writer不能逐帧复制教学视频。四个互斥K4 views支持同一个成功
occupancy panel，要求提取跨demo共同任务程序而非单条demo的速度、路径或抓取角度。

direct factor heads已在DJNFR证明能保留跨video共同方向并写到policy-effective q/v/action；本轮没有证据重开W2、
memory容量或rank问题。

## 7. 多task共存假设

DF-PCSP的单prefix gradient由一个远期binary outcome决定，因偶然微小首动作而对task9过度局部、对task18跨video
不一致。DF-SOCP在winner trajectory的多个阶段反复比较同一对策略，假设真正属于成功程序的方向会跨replans累积，
偶然局部动作会在trajectory mean中减弱。pair、task与view等权继续阻止horizon、success count或GPU ownership改变
task权重。

该假设不靠loss选择：preformal必须在固定task9/15/18 exact-discordant panel上同时成立，full24后仍只认single
checkpoint strict closed-loop、retained/gained/lost和task churn。

## 8. Canonical实现边界

- 原位替换DF-PCSP active reward objective，不保留runtime strategy switch或第二trainer；
- paired env rollout、exact state restore、LPCP/DJNFR Writer与八head owner保持；
- 新增的唯一计算是：把winner replay observations按loser arm分组，装载对应LoRA后B8生成counterfactual chunks；
- counterfactual query必须复用stored policy-noise seed、winner observation和winner executed-prefix length；
- flow preference physical batch按完整pairs切分，winner/loser不能跨microbatch；
- 每pair所有replans先等权，再pair、view、task等权；
- fresh-incompatible config/checkpoint/evaluator schema；旧DF-PCSP由Git与artifacts保存；
- 不新增逐tensor扫描、hash、重复video backbone、额外env step或防御性fallback。

## 9. Formal前机制门

固定使用已由exact pairing确认discordant的train task9/15/18，不再按结果挑anchor。三项都必须满足：

1. candidate/reference outcomes复现`2/1`、`2/0`、`1/2`；task4/task7 tie不产生gradient；
2. counterfactual replay分别覆盖winner active trajectories的全部26、65、44个replan chunks，0额外env step；
3. 每个replan observation的winner/loser两行逐tensor相同，counterfactual action RMS finite/nonzero；
4. preference objective、LoRA cotangent与Writer gradient finite/nonzero，真实Adam step后同panel margin下降；
5. 八个direct heads全部更新，q/v/action native BA与fixed-action response非零；
6. 每个anchor的train four-view BA cosine/energy至少`.40/.55`；
7. 每个anchor的validation8 aggregate至少`.30/.48`、至少6/8 tasks过`.15/.40`，raw factor cosine至少`.30`、
   action cosine至少`.15`、held/train BA L2至少`.30x`；
8. natural/reversed BA relative-L2至少`.50`，constant/natural不超过`.005`；
9. 0 forbidden read/OOM/nonfinite；counterfactual action generation使用B8且每winner chunk只查询一次；
10. 每个anchor cycle wall不超过对应DF-PCSP exact-prefix smoke的`2.5x`；若超过，只允许数学等价batch/profile优化。

三项任一失败即终局，不用另一个task替代，不调trajectory采样数、margin、temperature、LR、Nmc、rank或scale。

## 10. Full24与closed-loop裁决

机制三项全过后，才从sealed LPCP fresh做一次full24 cycle1并立即K4 strict paired400。cycle1只有同时满足：

- correct至少`142/400`、breadth至少7；
- 相对LPCP143 lost不超过15、gained不少于lost、无suite清空；
- post-train task9/15/18及held跨video门不坍塌；
- common-state full-occupancy margin与q/v/action response保持非零；

才允许exact-resume cycle2。稳定资格仍要求相邻两个single checkpoints均至少142、均值至少145、churn不超过20、
Jaccard至少`.85`、final相对LPCP lost不超过10且gained不少于lost。首次约145且retention过门立即补
same-task-other/wrong/shuffled/reversed/no-video。

## 11. 快速否决与负结果边界

- counterfactual first query不能复现loser行为到正常batch数值范围：query panel实现错误；
- margin不下降：paired LoRA cotangent接口失败；
- task9仍只在train放大或task18仍跨video分裂：full-occupancy evidence没有解决task-dependent credit；
- 三anchor机制稳定而full24 strict仍换手：shared multi-task aggregation/optimizer是下一接口；
- absolute过门但correct不优于视频controls：视频教学claim仍失败。

负结果只淘汰“LPCP/DJNFR生成图 + exact paired final success + successful-occupancy same-state loser counterfactual +
four-view one-cycle shared update”组合；不否定memory token、rank8、few-shot、生成LoRA、真正state branching或未来
生成LoRA后的task-local RL。

# V6-LPCP CFMG Unit-Secant Direct Commitment

状态：2026-08-17 active formal authority。简称 **USDC**。本轮从sealed LPCP fresh开始，不resume USFC；它完整保留
CFMG/USEP/USFC的输入、memory、rank32 LoRA、unit-secant reward、四视频梯度、active-task等权聚合和AdamW，唯一
改变是：提交未经缩放的一次原生Adam候选，把task×view 20/20 endpoint下降从保存硬门降为完整诊断。

## 1. Why this is the next falsifier

USFC full24的exact `j0`已经让`17/20` margins下降，task4/25/34/38全部4/4；唯一例外task19平均harm只有
`2.249e-5`。但原设计要求20/20，因而保存了exact LPCP并根本没有测closed-loop。endpoint margin是由有限train24
states构造的局部surrogate，不是EMBER最终目标；没有证据证明20/20既可达，也没有证据证明19/20或17/20与strict
成功存在阈值关系。

本轮不是把门从20/20随意改成17/20，也不从11个scale中挑最好者。它删除scale search，只提交优化器按冻结recipe
自然产生的唯一`j0`，然后立即用strict paired400裁决。这样既没有post-hoc选择，也不会因内部理想指标不可达而
永远回避真实任务。

## 2. Unchanged graph and information wall

```text
exact language + K=4 ordered action-hidden same-task videos
 -> one carrier-exact native PI0.5 context forward
 -> 37 layer-matched one-way memory tokens
 -> per-video temporal program + permutation-invariant K-set
 -> layer/token M2P content grid -> native rank16 B residual
 -> concatenate frozen LPCP rank16 carrier
 -> one complete rank32 38-target LoRA -> frozen policy rollout
```

frame stride5、两paired states、每成功轨迹8个等进度max-disagreement states、四个互斥correct K4 views、active
tasks等权、BF16/TF32、batch8、LR/betas/clip/RNG逐项不变。teacher actions/state/reward/task ID/filename仍不可读；
validation/test不产生梯度；部署只有一套完整LoRA。没有task-gradient normalization、PCGrad、task tangent balance、
rank/scale/seed sweep、expert route、第二adapter或生成后RL。

## 3. Direct commitment

每个active task仍先把四个view gradients等权平均，再在active tasks间等权平均。执行一次正常AdamW step，得到
精确candidate delta `j0`。只要gradient与delta有限且非零，distributed all-reduce下各rank共享同一个更新，就保存
`j0`。不做power-of-two backtracking，不恢复step0，不计算候选集合。

保存后只用原梯度路径已经记录的pre-update margins作为reference，对`j0`完整重算一次每个active task的四个
correct-video endpoint margins，记录下降数、逐task均值/最大harm和q/v/action response。20/20仍是理想诊断，
但不参与保存与checkpoint选择。为避免追逐batch/kernel低位差异，不再额外重复一次step0 inference baseline；
也不增加参数扫描、hash或逐元素跨rank校验。

## 4. Formal and training-volume contract

USFC full24已经证明当前图、信息墙、unit-secant、active-task credit和exact `j0`均finite/nonzero，因此不再重复
fixed-task smoke或held surrogate screen。canonical CPU和architecture gate通过后，从sealed LPCP做一次fresh full24
cycle1，保存唯一`j0` checkpoint并立即做K4 strict paired400。

canonical实现现已原位完成：active source相对USFC净减88行，没有新module、parallel runtime或compatibility path；
完整CPU=`413 passed`，compileall与diff check通过，architecture guard 0 hard violation。以上只关闭工程门，不是
GPU或closed-loop证据。

一次cycle训练量很小：24 tasks只有48 paired states/96 rollouts，且只有产生单臂成功分歧的tasks给梯度。因此
cycle1 strict只是方向筛查，不是稳定结论：

- 若correct、breadth或retention明显恶化，本轮终局，不用盲目延长训练救方向；
- 若cycle1 correct至少142、breadth至少7、相对LPCP lost不超过15且gained不少于lost，则按原world topology
  exact-resume cycle2并再次strict400；
- 若cycle1/2保持正信号且success-set重合良好，可继续cycle3；用相邻checkpoint曲线判断共同积累，不挑最高点；
- 两个相邻checkpoint约145或更高、churn<=20、Jaccard>=.85且最终lost<=10，才具稳定候选资格；首次达到该资格
  立即补same-task-other、wrong、shuffled、reversed和no-video。

## 5. Fast falsification and negative boundary

cycle1必须报告per-task/per-suite、breadth、相对LPCP/V6-fast的retained/gained/lost/churn、active credit tasks、
逐view margin诊断、effective BA和fixed-action response。正式选择只认同一single checkpoint的strict400。

若`j0` strict明显下降，淘汰“当前CFMG unit-secant等task梯度的一次直接Adam commitment”作为有用更新；下一接口
才是跨task functional tangent coexistence，例如预先定义的task-tangent balance，而不是回头改已通过的视频
carrier或随意扫scale。若cycle1好而cycle2换手严重，说明问题是持续训练中的shared credit coexistence；若连续
checkpoint稳定，再测试视频因果性。

任何负结果都不否定memory token、dynamic K/few-shot、rank8、生成完整LoRA或未来独立的task-local RL。

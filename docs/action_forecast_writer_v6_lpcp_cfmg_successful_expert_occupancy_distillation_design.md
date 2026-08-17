# V6-LPCP CFMG Successful-Expert Occupancy Distillation

状态：2026-08-17 canonical实现、CPU合同与真实GPU机制门均通过；fresh formal已完成cycle1--3及各自strict400，
现按owner最新训练量澄清exact-resume一个有界cycle4稳定性节点。简称 **SEOD**。本轮从sealed V6-LPCP143 fresh开始，
不resume MCTC checkpoint；保留已经验证的视频carrier、literal memory、K4集合、rank32 native LoRA写出、
four-view共享与median-capped natural Adam，只替换训练credit的来源。

## 1. 要解决的最早失效接口

MCTC连续三个cycles及三个strict400为`142/142/136`，相邻churn=`36/34`。cycle3训练内four-view gradient
cosine/energy=`.949481/.911623`，cycle2到cycle3的held first4 LoRA更新cosine/energy=`.988724/.990306`，
all400 effective-BA relative-L2=`.001199`。因此视频memory、跨视频共同Program和native public LoRA写出都已
工作；失败发生在**binary candidate/reference reward差所给的shared direction不覆盖held on-policy有用方向，
也不能连续保留多task support**。

该credit每cycle需96个rollouts，却只有`5/12/10`个active tasks；tie完全无梯度。继续增加训练轮数、调cap、LR、
rank或scale不会增加方向的因果内容，MCTC已终局。

## 2. 为什么不用expert权重重建

task experts是train24 privileged teachers，不是deployment dictionary。对8个train-seen tasks、每task四个真实K4
conditions的exact current LPCP rank16 carrier-A做只读FP64投影，匹配step2000 expert effective-BA可达能量仅
`.381712`，全局cosine=`.617828`；每task约`.307--.481`且几乎不随video set变化。直接把expert factor或BA重建
loss接到当前B-only residual会把约62%的不可达target当成训练误差，并重复历史expert reconstruction/manifold的
失败接口。

SEOD不要求Writer复刻expert LoRA。它只问：在expert已经真实闭环成功的状态分布上，Writer生成的policy能否逼近
expert的动作选择。这样训练目标位于policy behavior空间，当前factor topology只需找到其可实现的有用近似。

## 3. 输入、Writer与部署输出

部署图完全不变：

```text
exact task language + four same-task action-hidden ordered teacher videos
        -> carrier-exact V6-LPCP context forward
        -> 37 one-way memory tokens updated through 18 Action-Expert layers
        -> ordered per-video causal reduction
        -> permutation-invariant K4 set aggregation
        -> layer/token topology mixing
        -> one complete 38-target rank32 LoRA
        -> frozen source policy rollout
```

rank32由完整rank16 LPCP carrier与一个rank16 B-only residual bank组成；不压缩carrier、不增加第二adapter、不平均
video features或生成后的LoRAs。CFMG temporal delta和end-minus-start使静态/无动态画面不能独立产生residual；
language负责task address，但不能绕过视频动态Value。当前实验只声称K4，不声称未训练过的dynamic-K能力。

## 4. Privileged successful-expert occupancy

只在development train24训练中使用统一step2000 task expert bank：

1. 每task、每cycle按sealed随机reset schedule运行两条expert rollouts；不预挑成功状态，不按结果更换seed；
2. 只保留expert自身真实闭环成功的trajectory，失败trajectory不给正target；
3. 对每条成功trajectory的所有replan observations，用同一observation、noise、BF16/TF32、B8 batch顺序分别
   重新查询expert与当前anchor-video Writer policy；不使用stored rollout action作functional target；
4. 把轨迹按progress分成8个等区间，每区选expert/student action RMS差最大的一个state；
5. expert action成为该state的成功行为target，trajectory之间等权，task内一条或两条成功trajectory总权重相同；
6. 同一selected panel在四个互不重叠的correct K4 conditions下分别回传Writer gradient，再先按view等权平均。

expert rank16 adapter只在训练查询时零填充为rank32，保持scale1；它不进入Writer输入、checkpoint或部署。task39的
step2000 expert direct评测为0/50；若本轮随机rollout也无成功，它保持sealed LPCP baseline并获得零expert credit，
首轮不混入B20、binary preference或其它fallback objective。

## 5. Objective与shared commitment

对selected state `s`、matched noise `z`、expert target `a_e`和Writer condition `c_v`，令
`a_w = pi(s,z; LoRA(c_v))`。每state使用executed prefix上的unit-residual distillation：

```text
d2 = mean_valid((a_w - stopgrad(a_e))^2)
L_state = d2 / stopgrad(max(sqrt(d2), eps))
```

它与MCTC unit-secant一样把当前action discrepancy的尺度从gradient幅度中剥离，但没有人工loser或负方向。
每trajectory内8个states等权、成功trajectories等权、四个video views等权。task gradients进入shared mean前仍只对
高于active-task norm中位数者做upper cap；小task不放大、方向不旋转，然后提交一次原生AdamW step。

## 6. 与历史路线的区别

- 不是Expert-Manifold：不重建、route、mix或部署expert weights；
- 不是旧Expert-Flow Teacher Audit：旧audit在离线B20 target-action queries上比较expert与baseline，expert仅
  `2/24` tasks、`0/4` suites更低flow loss；SEOD只信任expert自身已经闭环成功的on-policy occupancy；
- 不是PCSD/MCTC：不要求LPCP与candidate在两个随机states上出现稀疏binary discordance；expert成功即提供稠密的
  8-strata behavior target；
- 不是生成LoRA后的task-local RL：当前仍训练一个shared Writer，评测时生成LoRA后零interaction、零更新；
- 不打开dynamic A或改rank。若SEOD仍失败，才可把“行为credit好坏”和“factor可达性”分开后讨论full-factor输出。

## 7. 信息墙与公平性

teacher videos始终action-hidden，且与expert rollout states跨episode。expert action/reward只属于train24 privileged
training signal，不能成为video内容、validation/test target或deployment asset。validation/test不训练expert、
不读取action/reward。每个condition仍只生成一套LoRA，rollout前运行Writer一次。

高分仍必须补correct/same-task-other/wrong/shuffled/reversed/no-video严格配对controls。SEOD只在correct videos上
训练并不自动证明有向时序；若language/static bypass或wrong/shuffled/reversed不劣于correct，即使absolute高也
不能成立。

## 8. 训练量与裁决

fresh formal原预注册最多三个full24 cycles，每cycle为24 tasks、每task两个expert states，最多48而不是96个rollouts。
cycle1主要打开zero payload gate，cycle2才让temporal/set/M2P content modules得到梯度，因此：

- cycle1完成机制分析和strict400；只有严重退化或机制失败才提前终局；
- compatible exact-resume cycle2并再次strict400；
- 若前两轮不是明确坏方向，完成cycle3和第三个strict400，以相邻success-set而非单点分数判断稳定性；
- 首次约145且retention合理时立即补六臂controls，并用下一checkpoint检验稳定性，不等到150。

门槛是科学裁决依据，不是为了过门补丁。目标仍优先追求`>150/400`；约145只有在breadth、低churn/high Jaccard、
same-task-video鲁棒和视频因果性同时成立时才有资格。不得cycle后扫seed/LR/rank/scale或按task选择checkpoint。

### 8.1 Owner澄清后的有界cycle4扩展

cycle1/2/3 strict实际为`129/135/143`，而非提前回落；cycle2到cycle3严格为`120 retained / 23 gained /
15 lost`、net`+8`。因此owner关于“有好结果时训练量可能不足，应多训练后再判断”的最新澄清使原三轮上限不再能
机械充当终局。与此同时cycle3 breadth从6降到5、相对LPCP143仍为`121/22/22`，所以143也不能被宣称为稳定好结果。

在不改变代码、数据、optimizer、LR、rank、scale、seed、world6 topology或任何科学变量的前提下，只增加一个
exact-resume cycle4，并对它完成同一K4 strict400。cycle4是预先记录的相邻稳定性检验，不是按结果挑峰：

- 若cycle4低于143、继续丢失breadth，或相对cycle3 gained<lost，则SEOD终局，不cycle5；
- 若cycle4保持至少143、breadth恢复到至少6且gained>=lost，则可再做最后一个cycle5确认平台；
- 任一checkpoint首次达到约145且retention合理，立即补六臂因果controls，同时仍需后续相邻checkpoint证明稳定；
- 全部checkpoint与逐行transition都必须报告，不能只选择cycle3或cycle4。

## 9. Pre-formal falsification

固定四suite train anchors=`2/12/21/35`，只用于证明图接通，不用于选择formal state。必须确认：

- expert rank16到rank32 zero-padding与effective BA/action等价；
- 至少三suite出现真实expert success，successful-only selection与8-strata panel正确；
- expert/student matched query具有完全相同batch sizes、observations、noise和执行prefix；
- 四个correct K4 views均有非零LoRA/Writer gradient，shared task gradient、native q/v/action BA和fixed action响应非零；
- step0仍等于LPCP，constant dynamic residual近零，reverse对memory/LoRA有material影响；
- wall、显存与rollout吞吐不比MCTC明显恶化。

若expert成功occupancy仍给出近零/互相冲突的four-view gradient，或真实Adam不能降低大多数train panel的expert
distance，SEOD在full24前终局。若机制通过但strict仍低、lost多或连续换手，则否定的是当前successful-expert
occupancy + CFMG B-only commitment组合，不否定memory token、rank8、few-shot、task expert作为其它privileged
teacher或生成LoRA本身。

## 10. Canonical implementation state

旧binary candidate/reference arm credit已原位退休；canonical runtime现在每task只运行两条expert trajectories，并
复用matched B8 query、8-strata、four-view Writer cotangent、median cap与exact Adam。新增的唯一source owner是
`reward/expert_teacher.py`，只负责sealed train24 bank inspection和rank16→rank32 zero padding；没有新entrypoint、
平行runtime或deployment fallback。active source相对authority commit净减少221行，architecture guard无hard
violation；定向/完整CPU=`56/416 passed`，compileall和diff check通过。24个step2000 adapters已在CPU实读并全部
成功pad为rank32。

clean pushed `08c5edc`在gpu02物理`1/2/3/4`完成四suite world4真实smoke：8/8 expert trajectories成功，
64个selected states、16个互斥K4 credit conditions和64条unique videos均按合同产生；四task的four-view gradient
pairwise cosine mean分别为`.987182/.775646/.953631/.971174`，shared与actual Adam descent覆盖均为4/4，
post-update 16/16 task-view expert distances下降。q/v/action effective-BA RMS与fixed-action response在四task均
非零，final delta L2=`.236847`；cycle=`87.094s`、peak reserved=`19,369,295,872` bytes，0 OOM/nonfinite/
forbidden read，exit0。constant/reverse与step0 identity由未改动的sealed carrier/memory路径继承；本轮新credit没有
改变该前向图。该smoke只证明SEOD可进入formal，不提供absolute、retention或视频因果性能结论。

首次frozen-worktree world6 formal在rollout前终止：expert bank相对路径被错误解析到detached worktree而非canonical
artifact root，因而报告0而非实际6个workers；没有run contract、rollout、gradient或checkpoint，不能作为科学
结果。根因已由主工作树/冻结worktree路径反事实验证。canonical loader现以source-run所在project artifact root解析
retained bank；同一frozen config直接复现为6/6 workers，定向/完整CPU仍为`56/416 passed`。该修正只改变训练资产
定位，不改变SEOD数据、目标、参数、随机性或topology。

clean frozen `d2f765c`随后在gpu01物理`0/1/2/4/5/6` world6完成cycle1--3。三轮训练active tasks=
`19/17/16`，strict score/breadth=`129/6 -> 135/6 -> 143/5`。cycle3相对cycle2为`120/23/15`、churn38、
Jaccard`.759494`；相对LPCP143为`121/22/22`。cycle2到cycle3 all400 effective-BA relative-L2 mean/median=
`.002354/.002020`，first4同task更新cosine/energy=`.993193/.992576`；gained/lost幅度=`.002439/.002350`
仍不可分。训练内部跨video共同写出随训练保持健康，但跨task pairwise gradient cosine从cycle1的`.1380`降到
cycle3的`.0554`。这些证据只授权上述cycle4稳定性扩展；尚未授权六臂或性能资格声明。

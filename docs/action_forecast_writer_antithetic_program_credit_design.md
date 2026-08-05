# Antithetic Program-Credit Writer 设计

状态：**2026-08-05 pre-implementation authority；任何GPU训练须先完成canonical原位实现、
聚焦合同与独立A40 profile。**

## 1. 这次只解决哪个接口

长期目标仍是同一single checkpoint的strict correct严格`>150/400`，并继续提高absolute、
breadth、视频因果性与跨checkpoint能力保留。

Policy-Lane已经把此前最可疑的LoRA外观问题真正修复：16个lane广泛参与，effective stable
rank达到`1.34--1.54`，q/v跨层能量专门化也达到direct SFT量级；但四点correct只有
`70/63/37/61`，same-task video在hidden/BA的能量仍只有约`.05%/.02%`。因此本轮不再
增加store、lane、head、rank或强制几何，而只检验：

> 若把Writer在decoder之前产生的完整policy program本身视为一个episode-level高层动作，
> 用成对closed-loop return直接估计该program的方向导数，再反传到条件生成网络，能否绕过
> functional action surrogate的错位，让task language + one hidden-action video获得真正的
> policy credit并在full24共同累积。

这不是把RL当成LoRA质量问题的替代品。LoRA decoder先由同一fresh AS轨迹学出；RL阶段只
改变“条件应落到decoder输入空间的哪里”，且正式裁决仍直接检查生成LoRA、视频差异和
closed-loop性能。

## 2. 为什么不用刚完成的Policy-Lane或历史v6 best

- Policy-Lane的fresh winner只有`70`。在它上面做reward训练会把弱AS起点、全新decoder和
  credit transport三个变量混在一起，不能解释结果。
- 历史v6-fast macro400=`143`适合做消融，但把它当主方法起点会再次变成“拿observed-best
  checkpoint再做校准”，不能回答从generic source开始的一套训练方法是否成立。
- 本方法的唯一cold start是现有fresh v6 AS root的macro125：

```text
runs/outputs/pi05_as_writer_v6_relative_flow_coldstart_formal_r6_b20_seed7_b75cb19_20260804/checkpoints/step_00000125
```

它从functional identity训练了125个full24 macros、60,000个action queries和3,000个
one-video conditions，不是历史best、reward checkpoint或profile权重。AS125 strict
correct=`97`，train K4有`19/24` coverage；弱起点和all-failure tasks正好使直接credit
假设可被证伪。若本方法成立，未来正式复现从generic source依次执行同一AS125阶段和本
reward阶段；复用已封存的AS125 artifact只是跳过完全相同的重复计算。

## 3. 唯一活动架构：v6 policy program + 冻结LoRA decoder

恢复历史已验证的v6时序路径作为canonical Writer，不保留Policy-Lane并行实现。其确定性
前向为：

```text
task language + one action-hidden video
  -> frozen PI05 semantic encoder
  -> Semantic Core + Visual Transition + Causal Procedure
  -> SlotNormalizedCoreProcedureCompiler
  -> H in R^(320 x 256)
  -> 8 FactorHeads
  -> one complete public rank-16 LoRA over 38 targets
```

`H`的320个slot恰好是18 expert layers×16 public ranks，加action-in/action-out各16个；
同一个expert slot共同进入q/v的A/B heads，因此它已经是policy-coordinated program，而
不是某一个raw LoRA tensor。

canonical Writer新增的只是显式接口，不改变确定性函数：

- `encode_program(...) -> H`；
- `decode_program(H) -> complete LoRA state`；
- 普通`forward`严格等价于两者顺序组合。

AS125载入后永久冻结source policy、normalization、semantic encoder和全部8个FactorHeads。
reward只训练Semantic Core、Visual Transition、Causal Procedure和compiler。FactorHeads的
输入、非线性和output matrices共同组成AS学得的固定LoRA decoder；它们不再随不同task的
reward互相旋转。与旧Tangent-Basis消融不同，本方法不仅冻结完整decoder，而且完全删除
executed-action CFM ratio，credit直接作用在decoder输入`H`。

## 4. K4 antithetic高层探索

每个task/cycle仍读取恰好一条same-task action-hidden teacher video，并生成确定性program
`H`。K4不再是同一LoRA的四条独立trajectory，而是两组成对program扰动。对pair
`j in {0,1}`，由`(seed_root, cycle, global_task_id, pair)`唯一生成
`Delta_j in {-1,+1}^(320 x 256)`：

```text
H_j+ = H + sigma * Delta_j
H_j- = H - sigma * Delta_j
```

首个唯一`σ=0.05`，因为v6 compiler输出逐slot经过RMS normalization，因而它表示每个
program coordinate的5%扰动，不依赖task或LoRA norm。不得按reward扫描sigma；正式前只
允许在固定train panel确认finite、LoRA非identity且没有数值爆炸。若真实profile出现OOM/
nonfinite或四个扰动LoRA逐元素相同，属于机制失败，须在formal前重新封存authority，不能
边看结果边调。

同一pair的`+/-`必须共享environment reset seed和完整policy-noise seed stream，只有program
符号不同；两个pair使用不同随机性。ledger cursor仍各自唯一，因此runtime显式分离artifact
cursor与randomness cursor，不能靠重用row ID伪造配对。方向seed不含rank、worker、queue
顺序或outcome，resume可精确再生；ledger只保存seed、符号和方向摘要，不保存大tensor。

## 5. Binary-first pair credit

每条trajectory仍使用official random-reset binary success。冻结的task-grounded progress
observer保持上一方法已经通过的action-free合同：它只读task language、teacher首尾RGB和
rollout自身首尾RGB，不读teacher/rollout action、proprio、pose、reward、terminal、task ID、
filename或video time。

对一个`(+,-)`pair，定义有界有符号差：

```text
if success+ != success-:
    dU = success+ - success-                 # exactly +1 or -1
elif success+ and success-:
    dU = 0                                   # preserve paired successes
else:
    dU = (Phi+ - Phi-) / 2                   # in [-1, 1]
```

因此official success永远优先；semantic progress只在pair双失败时打破平局，双成功不因
dense proxy被移动。没有critic、reward-to-go、episode length、LIBERO object state或
task-specific shaping。

## 6. 直接program梯度

两个pair给出SPSA/antithetic方向估计。标准中心差分与下面使用的梯度只差全局正常数；为使
每task贡献不随`D=320*256`机械增长，canonical program cotangent写为：

```text
G_H = sum_j dU_j * Delta_j / (2 * sqrt(D))
```

`sigma`、pair count和`D`均为全局常数，其余正比例由optimizer learning rate吸收，不改变
方向。每个task先形成自己的`G_H`，再除以24做full24 equal-task mean。runtime重新计算带
梯度的确定性`H`并调用：

```text
torch.autograd.backward(H, G_H)
```

然后只对四个可训练block做一次AdamW update。每个cycle恰好一次update，不对同一批return
做第二个off-policy epoch；不计算LoRA/action functional loss，不保存executed-prefix replay，
也不通过source policy反向。

这是smoothed closed-loop return对Writer参数的链式梯度：环境和source policy可以完全
black-box，而Writer的condition-to-program路径仍可微。它与ES/parameter-space policy
search的antithetic estimator一致，但探索的是一个有语义、由AS学得decoder解释的policy
program，而不是全部10M Writer参数。相关可行性边界见
[Evolution Strategies](https://arxiv.org/abs/1703.03864)、
[dimensionality-reduced policy search](https://proceedings.mlr.press/v151/memmel22a.html)和
[latent-space diffusion policy RL](https://proceedings.mlr.press/v305/wagenmaker25a.html)；
这些工作只支持黑盒/latent credit的合理性，不预先证明EMBER会成功。

## 7. optimizer与训练剂量

- AdamW，`lr=1e-5`、betas=`.9/.95`、eps=`1e-8`、weight decay0、global grad clip1；
- 24 train tasks/cycle、K4=two antithetic pairs、one teacher video/task；
- one full24 equal-task update/cycle；
- teacher video按既有50条no-replacement schedule，pair directions按cycle独立；
- formal先封存最多8 cycles，checkpoint=`1/2/4/8`，但每个点是否继续由预注册held门决定；
- source policy、normalization、progress observer、FactorHeads永久冻结；teacher action reads
  after AS125=`0`，validation/test action/reward reads=`0`。

旧RL的Nmc4、PPO/SPO、flow replay、two learning epochs和ratio clip全部退役；它们不是本方法
的“安全fallback”。K4与full24保留的是交互预算和task公平性，不是假装科学batch未改变。

## 8. formal前的最短证据链

### 8.1 CPU/单卡聚焦合同

只验证会改变结论的接口：

1. `forward == decode_program(encode_program)`逐tensor等价；
2. `+/-`方向相反、seed与rank无关、resume再生一致；
3. pair共享env/policy randomness但artifact cursor唯一；
4. binary-first pair credit全部分支与边界；
5. cotangent只能到四个上游block，semantic encoder/FactorHeads/source policy grad为0；
6. checkpoint包含optimizer、cycle、各rank RNG、direction/rollout/credit prefix与实际world
   size，fresh/resume family严格区分。

不为退役CFM ratio保留兼容分支，不新增大而泛的测试harness。

### 8.2 六卡one-cycle profile

从AS125权重复制到全新profile root，真实24×K4、两组antithetic、one update：

- 96 rollouts、48严格paired comparisons、全部24 tasks覆盖；
- 至少一个binary-discordant pair或双失败semantic-nonzero pair；若全48均零credit则机制失败；
- 四个上游block中由结构可达的block均有finite gradient，冻结对象0 grad/0 optimizer owner；
- perturbation LoRA finite且`+/-`摘要不同；program cotangent和Writer grad finite；
- A40峰值、NCCL ready、world-size ownership与完整cycle checkpoint健康；
- 用独立fresh0→1再resume1→2证明exact-resume，profile权重永久弃用。

BCI launcher继续显式`NCCL_P2P_DISABLE=1`，六rank在各自CUDA完成后通过唯一session原子marker
再进入对称gradient sum。不得用timeout、降K、减少task或少卡绕过故障。

## 9. formal与held续训门

formal必须从AS125重新进入全新root，不加载profile、旧RL、v6-fast400或Policy-Lane权重。
先只跑cycle0→1，然后用与AS125严格配对的correct400裁决：

- 若cycle1严格`>150`，继续cycle2/4/8并做winner五臂与内部分析；
- 若未过150，但相对AS125至少净增10、breadth不降且至少两个suite净增，允许续cycle2；
- 若aggregate下降、breadth下降、只有单task换手，或gained不大于lost，立即停止该轴；
- cycle2后只有相对当前single winner继续净增、breadth不降或已过150，才允许cycle4；
- 不按train success、progress、program gradient、LoRA norm或functional loss续训。

每个held点报告correct、breadth、per-task/suite、gained/lost、union/intersection和paired身份。
winner才补same/wrong/shuffled/reversed；若correct改善但三种反事实同步改善，不能声称视频
credit成立。

## 10. 必做内部分析与可证伪结论

至少比较AS125、cycle1和single winner：

- deterministic program、effective BA与fixed-action的更新幅度/方向；
- same-task五video、wrong/shuffled/reversed在program→BA→action的传递；
- train24 task-pair program gradient cosine、负pair比例、full24 energy retention；
- binary-discordant与semantic-tie-break pair分别贡献多少cotangent；
- gained、lost、retained state是否在program/BA/action上可区分；
- FactorHeads、semantic encoder和source policy逐tensor不变。

若program方向随cycle可复现、视频传递增强且correct共同上升，支持“直接LoRA-program credit”
假设；若train return上升但held换手不减，失败在跨task共享condition map；若program变化可控
但BA/action无响应，失败在冻结decoder覆盖；若随机program扰动几乎不能改变trajectory，
失败在exploration interface或sigma，而不是再回到LoRA rank外观。

## 11. 通用性与退役边界

该方法只要求：一个条件生成器、一个可微的低层policy-program接口、一个可由环境评价的
episode return。AS可由监督、离线RL或其它policy-aware预训练替换；binary和当前progress
observer也可由其它任务的真实return/value替换。部署仍是one-shot确定性mean program和
exactly one LoRA，没有交互、随机探索、critic或额外视频。

唯一实现owner继续是`src/ember/writer/`与`src/ember/rl_writer/`，launcher仍为
`scripts/train_rl_writer.py`。恢复v6时删除Policy-Lane executable module/config family；
旧Policy-Lane、PWAD、Tangent-Basis和CFM-ratio实现只由Git、design与frozen artifacts保存。
不得保留第二套active model、trainer、checkpoint loader或evaluation adapter。

## 12. 实现与CPU合同状态（2026-08-05）

- canonical v6已原位恢复，`forward`严格组合显式program encode/decode；Policy-Lane、旧
  Flow-Credit和progress diagnostic executable families已删除。
- direct program-credit runtime、pair ledger、actual-world-size checkpoint与evaluation
  authority已经接通；semantic encoder、完整FactorHeads与source policy由optimizer owner
  fail-closed冻结。
- 69项聚焦合同与项目正式activation下全仓220项全部通过，另有py_compile和diff check；
  AS125 manifest定向确认来自fresh identity、125 full24 macros、60,000 queries和3,000
  one-video conditions。
- 以上只seal A40 profile，不构成GPU机制或closed-loop性能证据。formal继续blocked，必须
  先完成独立cycle0→1及exact-resume1→2 profile并按第8.2节裁决。

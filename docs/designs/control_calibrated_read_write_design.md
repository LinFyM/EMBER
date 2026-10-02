# 让可见控制内容参与同一LoRA读写：完整学习检验

2026-10-03。依据机制§104–105及Owner持续自主推进授权登记。
本项是新的、可失败的完整方法干预，不是由动作获取通过自动晋级，也不声明已找到主要根因。
当前状态只看progress；本合同只授权下述fresh450及固定读回，后继相邻学习另行裁决。

2026-10-03完整450及科学消费已完成，裁决见机制§107/findings§286：correct400=122、seen144=77，
动作获取改善未成为完整控制收益，本配方关闭；不续900、不扫辅助/调制参数，不据此恢复独立Reader或Pullback。
以下保留原预注册合同；原件及完整450恢复状态保留，不构成继续运行授权。

## 1. 为什么投入这一项

最近原生校准P在内部四task对F均改善，平均风险下降19.857%，对裸source下降13.728%；
主要是五步区间平均指令，task32运动反例及P−mu簇区间跨0保持。它提供了可学习控制内容的实际依据。
但q是视频的派生表示，不增加原视频信息，也不能独立决定新初始化的反馈规律。

当前要检验的解释：已有动态特征虽受完整FM监督，却可能没有形成跨task一致的控制坐标；
把经真实动作约束的、由可见转移推断的控制内容送进实际A/B写入，可让这些写入更容易复用。
这里“更容易”是有限学习假说，不由可表示性、非零梯度或q拟合推出。原有FM已经有自身状态信用，
§103不支持再把新增query辅助边当成修复；对象选择和跨状态调用也可能完全不因本项改善。

最近似历史约束：v4的原生forecast已进入LoRA，但未以到达画面校准实际转移；LocalActionGrounded的反演
输出未进入实际LoRA；LocalField已监督同一写入场，故不能把“监督写入”本身称新机制；Pullback的固定及共享
native传递出口均有负例。本项使用有物理动作标签含义的q调制自由学习Value，不恢复固定Jacobian/PCA出口。
标签学习与其实际消费共同构成一个新增机制；本轮不能分别归因其中每个组件，不能称为纯粹增加一条输入的单因素证明。

选择已有条件c/d、S/M计算作为受控载体，保持完整状态/变化内容和两侧自由读写；不是维护其140/400结果。
绝对能力始终对强MT153和成熟T2340的161等完整参照判断，不能只胜旧CRW450的137便宣称方法有价值。

## 2. 两种原生读取与输入墙

source、38-target/rank128/alpha128、tokenizer、normalization、双RGB/stride5/保留真实末帧沿
`configs/operator_read_write_v1/conditional_read_write_fresh_spec.json`；source基础权重始终冻结。

1. **裸source读取。** 在没有任何公共/条件LoRA的Source1000上，以真实RGB/language、无State、probe1729/tau1
   得到`H0_t[50,1024]`和同次`mu0_t=(epsilon−velocity0_t)[:5,:7]`。其数值不随Writer更新，可只读缓存。
2. **公共读取。** 沿当前唯一canonical native，在正在共同学习的β=(A0,B0)下得到完整`Hβ_t`和38处真实`Xβ_l,t`；
   这条读取、解释器及其原native梯度保持，不用裸H0替换，不切断β的真实FM信用。

两路是同一冻结基础policy的两种rollout前特征计算；部署最终仍只有一套(A0+S,B0+M)，没有第二执行adapter或Reader。
Writer只接收真实RGB、exact language及由它们计算的原生特征；task/demo/path/episode length只由数据调度和记录使用。
teacher state、pose、reward、terminal及真动作latent均不得进入任一读取、Gamma或Compiler。

裸缓存必须是RGB/language派生的纯特征记录：H0、mu0、真实frame indices及必要来源，不含动作/状态/奖励标签。
可在授权训练侧从上一诊断的已有H0/mu提取复用，但不得让部署读取包含target的旧native PT；不修改或删除旧证据。
新标签只由训练/分析侧独立读取。validation物化不打开动作dataset，不因缓存方便越过信息墙。
预计完整1800条训练teacher的裸特征新增约12–18GiB，启动前以实际metadata核算，必要缓存只保留一份供本研究复用。

## 3. 同一控制估计直接参与A/B的Value

Gamma完全复用最近P的数学函数：parameterless RMS eps1e−6；1024→256的Q/K/V、4×64头、共享C读departure/context两次；
concat(Q,c0,c1)经768→1024 GELU→7，完整50个native位置参与学习attention；最后层/bias零初始化。
所有Gamma参数fresh seed7，不加载诊断P/F权重；不使用F作为部署候选、不选source/层/probe或flow时刻。

对真实转移p→p+5：

`q_t = mu0_(t−1) + Gamma(H0_(t−1), H0_t)[:5,:7]`，`u_t=vec(q_t) ∈ R^35`。

q是实际预测，训练时也不以真动作替换。保留有序5×7，不将其平均成一条命令，也不把它对应为50个native槽的真动作。
每个target分别新增两个无bias线性映射`Ua_l,Ub_l:35→256`，全部零初始化；在原有256维Value隐层作如下调制：

```
ga_t = 1 + Ua_l u_t                       # 沿50个真实native位置广播
gb_t = 1 + Ub_l u_t
Va_t = Oa_l [ GELU(Wx xi_t + Wz A0_l xi_t + Wc [c_t,Hβ_t])
              ⊙ Da_l d_t ⊙ ga_t ]
S_t  = S_(t−1) + (Va_t − S_(t−1) Xi_t) Xi_t^T / 50
A_l  = A0_l + S_T
K_t  = normalize(A_l Xβ_l,t−1),           DeltaZ_t = S_T Xβ_l,t−1
Vb_t = Ob_l [ GELU(Wk K_t + Ws DeltaZ_t + Wc' [c_t,Hβ_t])
              ⊙ Db_l d_t ⊙ gb_t ]
M_t  = M_(t−1) + (Vb_t − M_(t−1) K_t) K_t^T / 50
B_l  = B0_l + M_T
```

xi_t是原normalized departure X，解释器c/d、位置、因果mask、完整50槽及原两次有序写入均保持。
U初始化为0时该调制为identity；原有参数初始化顺序保持，新模块用独立seed域追加，不加载旧Writer。
不足5的真实末间隔保留原完整Hβ/X及c/d写入，只令该间隔ga=gb=1，不构造五步标签或把短间隔伪装为gap5。
本项不新增动态K；K=1沿旧正式事件，每condition只输出一套完整38-target LoRA。

该形式保留q=0时的原动态Value，避免“source在teacher路径预测正确，视频就完全没有内容”的残差反例。
q包含完整控制而非q−mu；状态/context/真实d均保留，q不单独承担对象与目标语义。
整段d=0仍使S=M=0，语言或动作先验不能越过真实动态Value独立写出条件LoRA。
Value的几何提升O/W/D/U和最终A均学习，输出不被固定J_teacher^T或一个共同7维列空间限制。
最终自身作用仍是`B0 S h_self + sum_t Vb_t omega_t(h_self)`；omega含教学地址、后续覆盖及自身hidden，
不是按teacher时钟播放q。Writer在rollout前退出，原生policy使用自身RGB/state与10步flow生成动作。

## 4. 学习、标签与真实梯度

完整跨episode FM保持系数1。新增辅助项也是系数1，但只监督本condition所有完整gap5区间的q：

`L_aux = mean_interval,horizon,channel ||q_t − normalize_source(actions[p+1:p+6])||²`。

每condition先对完整区间等权，再按原四task各1/4；长视频不获得更大的task权重。
零/短尾不padding为监督。报告实际额外标签区间数/动作位置数，这是新增监督使用，不冒称新独立task或新示范。
Gamma使用同一实际q接受辅助信用和完整LoRA FM信用；辅助项不回传裸H0/mu，因而不直接更新公共β、c/d或U。
主FM更新全部Gamma、U、原解释器、写入头和公共A0/B0；所有source基础参数冻结。
辅助只在每个condition计一次，不在38个target或replay/shard重复加权；实现保存精确消费者与权重。

局部偏导明确为`dL_main/du = sum_(l,h) U_l^T diag(z_l,h ⊙ D_l d_t,h) O_l^T gV_l,h`，
另加同一q的辅助误差信用。main学习怎样把控制坐标用于实际隐层作用，辅助学习如何从合法观察获取这些坐标。
Ua/Ub及O的合法零初始化会使部分首步信用为0；后续实际路径打开是工程条件，不是有效性证据。
多目标可能冲突、Gamma可能被忽略、teacher控制可能不能迁移；不以此预设成功，也不事后调λ挽救。

只用既有36task、50条/任务及原显式allowlist/provenance，不扩充source任务或示范：
`[0,1,2,4,5,7,12,13,14,15,17,19,20,21,22,25,28,29,32,34,35,36,37,38,42,43,51,55,56,62,64,73,95,96,97,101]`。
前24为target train，其余为原CRW450已审support12；固定coverage protocol/manifest及完整语义排除沿原合同复用。
本项四个诊断内部task不再作为留出：它们本来就是train24，完整方法按原36task学习；不得继续冒称其为Gamma held证据。
validation/test及重复项不产生任何梯度，source71的其余任务不加入本项。

fresh450、每步4task×28跨episode query、共50,400完整50×7 FM query；每task首轮50teacher各一次。
严格复用CRW450的task/teacher/query/flow事件定义seed20260928与query offset1；teacher自身动作只用于上述Gamma辅助。
所有模块、optimizer/scheduler全部fresh；AdamW lr3e−4、betas(.9,.95)、eps1e−8、wd1e−4、clip1、warmup150、decay1200、floor1e−5，
沿旧CRW合同，不另调Gamma LR或loss权重。完整ECP保留0/90/180/270/360/450；中间点只恢复，唯一科学终点450。
保存全部共享参数/optimizer/scheduler/scaler/sampler/cursor/rank RNG/拓扑/schema；允许完整边界按AGENTS迁移物理卡数，不称bitwise exact。
本项没有RL、public FM、teacher forcing、private fitting、蒸馏、冻结课程或task-local优化。

## 5. 唯一450读出与方法裁决

- **Validation correct400。** 固定tasks[3,6,11,16,23,26,31,39]、init0..49，复用原scenes、video_schedule seed7、
  state/video/env/policy RNG。每task50teacher各一次；8个init0 full/392 compact及全部continuous/actions/goal/raw rows。
  配对报告CRW450137、CRW900140、T450122/T900148/T2340161、Context900151、强MT300153；保留不同学习年龄和监督条件的差别。
  per-task/suite、breadth、R/G/L、churn/Jaccard与whole-task不确定性齐备；不能以checkpoint union或旧坏点作接受标准。
- **Seen144。** 同36task×init32..35、原scope seed20260928、0–49池的固定四teacher映射及scene_canonical144/场景/RNG，36个init32 full/108 compact。
  对旧CRW45091、CRW900115、MT93配对，单列train24/support12；只称训练有限池诊断，不称held资格。
- **原train-only A28。** 原task0:40/11、12:25/14、20:38/42、32:17/43及固定query/noise/tau，
  保存8 full/4 public FM预测，口径与旧CRW读回相同，不增加query面板。
  在这8次实际编译中被动保存H0/mu0、实际q、真实索引，以及原Q8/V8/action_out的Xβ、A0/S/B0/M、c/d。
  训练侧另外保存合法5步标签/mask供q评分，标签文件不进入部署特征缓存；不新增forward、层扫描、反向探针或官方held动作分析。

这些形成一个完整方法在固定学习年龄的判定。主项是合法完整LoRA的绝对闭环能力和任务分布；
Gamma误差下降、相对137的提升或seen拟合都不能独自支持续训，更不能拼成超过强MT的资格。
若只在上述内部量改善、完整控制无实质广泛进步，降低并收束“动作校准调制该读写即可修复”的假说，
不追加λ/seed/LR/rank/宽度/层位小扫，不恢复同机制的独立Reader或固定Pullback。
若绝对能力与分布支持，main再据全部正反判断是否值得相邻学习；450不称稳定selected，也不自动续至900。
本批没有other/wrong/shuffled/reversed、Test或RL；视频因果controls只在后继正式selected冻结后登记。

## 6. 资源、工程与交付

唯一root：`/data1/user/ymdai/ember_runs/control_calibrated_read_write_20261003`。
预计含工程6–10小时，依据原CRW450整批19.273GPUh与约12–18GiB额外裸特征、小Gamma/调制开销；
旧费用含当时hold等成本，不恢复hold，也不将估计当新吞吐实测。硬限32完整GPUh、data1新增峰80GiB。
先metadata核帧/标签/缓存/ECP/物化峰，再strg01核data1独立quota、个人用量及共享容量；不复制source/dataset。
失败、加载、cache、profile、训练、物化/评测、I/O占用和退出全部计费；预计明显超期或超预算须报告具体原因与边界。

每次launch双节点live准入，遵循现行总卡数/单节点上限；训练、裸缓存、物化及评测按实际吞吐及时并行。
不把4个逻辑task或一次profile的world size变成整批并发上限；也不为占更多卡增加计算或dummy占卡。
NCCL_P2P_DISABLE=1、GPU-local NUMA/deferred NCCL按现行实现；独立评测无NCCL。
复用旧最长train38/demo36的105帧/28query做最多3次丢弃真实更新，覆盖Gamma辅助与完整same-version replay，随后恢复fresh所有状态。
profile实际最大有益frame/query batch和拓扑；接受正常BF16/TF32/高效kernel，不为低位一致降吞吐。

执行者独占canonical tracked/Git与隔离codex工程worktree，完整实现/验证/集成push，从clean pushed detached启动；
复用canonical operator/data/native/训练/checkpoint/物化/evaluator，只有一个新active模式，不建立第二长期trainer或Reader。
针对性检查只覆盖真实输入墙、gap5/offset、Gamma与U初始化、同一q消费、零动态保持、辅助只计一次、完整梯度/replay与实际消费者。
明显不改变科学合同的工程故障由执行者在原预算自主修复、push/freeze新代码后继续；涉及特征/标签/目标/评测/预算改变再报main。
检查通过直接运行，不等待main重复工程验收。正式任务一次持续等待退出，不固定轮询日志，不向已有持续等待者发送自Queue。

450全部交付后统一核实退出、全部原件、成本和设备释放，记录实际来源与正反/缺项，集成push后只回报一次整批结果并交回窗口。
保留唯一新canonical实现及完整450恢复能力等待main科学裁决，不自动延长或恢复旧模式；若main淘汰本构造，再按明确生命周期退役其专用路径。

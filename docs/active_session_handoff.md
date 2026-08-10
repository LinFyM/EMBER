# EMBER Active Session Handoff

## 1. Current truth

这是当前跨session科研入口，覆盖所有历史文档里的旧“当前”“下一步”和阶段性暂停。

- Goal仍未完成：同一shared method、同一single checkpoint的strict paired correct必须严格超过
  `150/400`并尽可能继续提高，同时要求真实视频时序因果性、same-task鲁棒、breadth、低checkpoint
  漂移和可重复累积。
- Owner已授权持续自主推进，只有实质阻塞才沟通，并已允许subagent承担独立、只读或隔离写入的
  加速工作；主进程仍负责统一科研判断和最终验证。
- canonical仓库是`/data1/user/ymdai/projects/EMBER`，主写分支是`codex/bci-continuation`。
  retained formal GPU工作必须来自clean pushed commit的frozen worktree。
- 当前没有EMBER GPU进程。任何新GPU工作前必须实时比较`gpu01/gpu02`，只用空闲A40、合计最多6张，
  不干扰他人；多卡固定`NCCL_P2P_DISABLE=1`、NUMA physical/local rank映射和deferred-NCCL。
- **2026-08-10最新裁决**：第35节Tangent Tube已从clean frozen`b308941`完成formal fresh0→10和
  同一one-shot correct400。训练10 macros总step wall=`207.444s`、input wait=`.265s`、peak
  allocated/reserved=`43,316,440,064/47,112,519,680` bytes，0 OOM/nonfinite。macro10两臂
  relative-anchor tube中位=`.01390/.01408`，但directional ratio中位=`108.93/126.88`、两臂
  `0/24` tasks过`≤1`，completion error task median=`.25229`且`0/24` tasks过`.05`。
- strict correct400=`131`、correct80=`27`、breadth5、per-task=`0/3/46/31/0/40/11/0`、per-suite=
  `3/77/40/11`。相对同schedule macro0=`134`严格paired gained/lost=`16/19`、churn35、net`-3`、
  `p=.735879`；所以不续25、不补六臂、不扫weight/LR/WD。当前config/runtime已封为formal non-pass并
  fail-closed。该证据淘汰当时的tangent recipe/window，但completion从未成立，不能宣称expert-component
  假设已被干净证伪。
- **2026-08-10最新audit裁决**：第36节matched Expert-Flow Teacher Viability Audit已从clean frozen
  `e8e4728`自然exit0。formal root=`runs/outputs/pi05_v6_expert_flow_teacher_audit_r6_lb20_mb10_e8e4728_20260810`；
  480/480 queries、24 tasks、suite 6×4、reversed/shuffled/wrong 8/8/8、144 policy forwards、0 update/
  rollout/OOM/nonfinite。wall/input wait=`39.698/.684s`，peak allocated/reserved=
  `43,418,974,720/47,133,491,200` bytes，所选六张A40结束后自然释放。
- expert/macro0/tangent10 matched真实7维flow loss=`.098631330/.091801740/.091843160`。expert只在
  `2/24` tasks且`0/4` suite means同时优于两baseline，远未达到预注册`18/24+3/4`；四suite相对macro0
  都差约`4.89%--11.90%`，剔除最差task后仍差约`6.07%`。teacher-quality是方向性non-pass，不是边缘
  fail或单一outlier。
- CEFD gradient在compiler/factor相对existing span的residual=`.6864/.8387`，finite且非冗余；但
  “不同方向”不能把整体更差teacher变成有用teacher，且distillation loss最大的位置反而偏向direct expert
  较弱tasks。因此`authorize_cefd=false`：不做CEFD weight profile、训练或事后扫描其它expert steps。
  一次性audit config已formal non-pass并fail-closed，runtime按第36节触发退役。
- **2026-08-10第37节v1 profile正式non-pass**：clean frozen`6903ee6`在`gpu02:0--5`自然完成root=
  `runs/outputs/pi05_v6_condition_residual_mechanism_profile_macro49_r6_lb20_mb10_6903ee6_20260810`；没有
  checkpoint/retained weight，0 OOM/nonfinite/negative policy forward，六卡退出后均回到0MiB。
  13项机制门中10项通过：feature rank48、correct motion/cotangent=`.807966`且`24/24` task retention、
  application closure relative RMS=`0`、A/B response均非零、四suite fixed-action=`4/4`。因此显式kernel、
  frozen-v6 decoder和Program→完整LoRA→action传递成立，未发现gather/order/sign/solver工程bug。
- non-pass是同一个key-geometry根因：regularized Gram condition=`1315.33`，aggregate negative/correct=
  `.264351>.25`，task-local null=`15/24<18/24`。shuffled/reversed/wrong的paired feature cosine mean=
  `.98552/.95645/.90627`，null过门=`2/8,6/8,7/8`；全部9个失败row cosine均`>=.97099`。单位pair ridge
  leakage解析式对实测相关`.99021`，证明v1未平衡DC主导了顺序key。v1不训练、不扫lambda/seed/P/阈值。
- v1 production wall=`23.530704s`，相对sealed baseline ratio=`1.115458>1.10`，按原门保留non-pass；只
  超上限`.326083s`，而跨host input-wait差`.633711s`，所以不扩大解释为稳定计算退化，也不为它单独重跑。
- **当前唯一active实现是第38节v2 Balanced DC--Causal Condition Key**：historical v6的600 tensors和
  `[256,320,256]` memory/full48/`.01` damping/step1/B20/B10+10全部不变，只把fixed key改为video-DC
  static与phase-centered sqrt-causal-prefix dynamic两个独立JL128 blocks，各自zero-L2后拼成P256。
  no-video仍精确zero；same-frame-set reverse/shuffle共享static但RHS为`g/0`，所以static不能单独拟合。
- **第38节v2 mechanism profile已正式13/13通过**：clean frozen`5d93434`在与sealed baseline相同的
  `gpu01:0,1,2|4,5,7`完成root=
  `runs/outputs/pi05_v6_balanced_causal_condition_residual_mechanism_profile_macro49_r6_lb20_mb10_5d93434_20260810`。
  rank48、condition=`106.114`、correct/cotangent=`.968254`、negative/correct=`.0218514`；24/24 correct
  retention且最小`.942261`，24/24 null且最大leakage`.048462`，A/B、4/4 fixed-action和closure全通过。
- shuffled/reversed/wrong cosine mean=`.479565/.013732/.507178`，各臂最大`.851083/.023307/.762135`；
  leakage mean=`.024184/.018664/.025999`且三类均8/8过门。production=`20.021842s`、对sealed baseline
  ratio=`.949122`，input wait=`.069295s`与baseline`.076318s`匹配；0 checkpoint/OOM/nonfinite/negative
  forward，selected GPUs结束后回到14MiB。
- 一次性teacher-audit/effective-objective/flow-teacher owners及tests已删除；checkpoint只拥有单个Program
  memory、cursor和六rank RNG，base600和fixed projection不被保存或覆盖。v8 residual deployment adapter、
  strict paired evaluator和analysis family已经联锁，错误Writer family不能借用本候选profile seal。
- 最后合同复核已封住artifact自报通过、formal状态空壳和任意checkpoint lineage三条缺口：profile从raw
  macro重算13项门并匹配完整scientific run；formal result必须绑定completion、50-row metrics和10/25/50
  manifests；deployment training commit必须属于active remote authority lineage。clean detached frozen
  authority ancestor现可直接用于v8 evaluator，不需制造第二主分支。
- v2聚焦`52 passed`、带LIBERO assets最终全仓`281 passed in 21.34s`；compileall、26份JSON、diff-check与
  architecture guard通过。profile artifact已从raw macro/run/completion重算并写入config；mechanism状态
  sealed。**当前仍没有v2训练或formal strict成绩**；唯一v2 rollout只是下述8-row deployment execution smoke。
- 当前deployment verifier已恢复并收敛为一个双root owner：必须同时重读同commit的32-request batch8/16/32
  profile、validation8×state0 correct results和native LoRA cache manifest，核对单卡A40、selected batch、
  8 rows/entries、单次launcher、0 retry/runtime failure/forbidden reads及Writer release/source reuse。旧的profile-only
  evidence不能seal，formal runtime也同时要求该evaluation artifact，不再靠文档顺序防止误启动。
- 该GPU前修复的最终CPU门为全仓`283 passed in 26.10s`、compileall、Black、26份JSON、真实config load与
  diff-check；architecture guard相对`5d93434`为`+968/-318`且0 hard violation，原contract缩到1101行，
  没有parallel family或训练/推理热路径变化。
- clean frozen`2af82aa`已在实时空闲`gpu02:0`完成deployment双root。固定32-request/1093-frame panel的
  batch8/16/32吞吐=`.911238/.901898/.906482 LoRA/s`，repeat分别约`34.97/35.27`、`35.63/35.33`、
  `35.30/35.30s`；三者稳定且reserved约12.9GB（约12.0GiB），按最高实测吞吐选择batch8。
- validation8×state0 correct vertical root真实执行8 videos→8完整LoRAs→native cache→释放Writer→复用
  source policy→8条LIBERO闭环；`4/8` success、总wall=`336.056s`、rollout window=`199.799s`，8/8 rows、
  单次launcher、0 retry/runtime failure/forbidden reads。双root assembler已通过且GPU释放；`4/8`不是正式性能分数。
- config现为`active_deployment_sealed_formal_ready`。下一GPU动作从新clean pushed/frozen seal先评测
  zero-memory macro0 strict correct400；只有该真实基线封存后才fresh0→10并立即strict correct400。
- deployment写回由clean pushed`d228d0d`封存。其frozen worktree的首次CPU-only formal prepare在0 CUDA
  worker/0 scientific row时暴露一个工程合同错误：`runs`软链接经`.resolve()`落到canonical仓库后被旧
  evaluation verifier误判为越出worktree。`af7b101`只修artifact路径owner，允许词法
  `runs/outputs/...`且resolved target仍在canonical outputs root；nested symlink和manifest越界继续拒绝。
  全仓`285 passed in 21.38s`，clean frozen`af7b101`的同一formal prepare已exit0，精确登记8 tasks×50
  states、correct/without-replacement、method macro0 historical-v6 load-only + fresh elementwise-zero residual、
  18 rollout workers + 18 Writer generators和batch8；临时prepare root已清理。该prepare没有启动GPU或
  形成性能证据。
- **zero-memory macro0 strict400已正式封存**：clean frozen`6b5f7a6`在实时空闲`gpu02:0--5`自然exit0，
  root=`runs/outputs/pi05_v6_balanced_causal_condition_residual_correct400_noreplacement_seed7_method_macro0000_6b5f7a6_20260810`。
  strict correct=`134/400`、correct80=`26/80`、breadth6；per-task按Spatial/Object/Goal/Long=
  `0/5/48/34/0/35/11/1`，per-suite=`5/82/35/12`。wall/rollout window=`867.152/616.138s`，72/72 shards、
  400 rows、18 workers均attempt1/exit0。
- 400套native LoRA由18 generators以54 batches全部fresh生成，configured/max batch均8，0 reuse/redundant
  forward；Writer全释放、source policy全原进程复用且未reload。max per-generator allocated/reserved=
  `11,745,421,312/12,895,387,648B`，0 retry/OOM/nonfinite/forbidden reads；六卡结束后0MiB/P8。
- 与历史native v6 macro0 root的400-row严格identity检查中，state/language/env seed/policy-noise、teacher demo/
  order/selection seed和video mapping均0差异；success也逐行完全相同，gained/lost=`0/0`、共同成功/失败=
  `134/266`。新旧400 cache entry IDs相同，逐tensor CPU直比30,400 tensors、514,867,200 values全部
  bit-exact。唯一微差为一条共同成功episode终止step `106→107`，其余399 rows steps相同；不改变formal
  结论，也不为它降低吞吐。每task demos0--49各一次。由此v2 zero-memory部署图基线成立，不把`134`写成改进；当前下一
  科学动作是formal fresh0→10与即时strict correct400，仍无非零Program memory成绩。
- 历史最好single checkpoint仍是v6-fast macro400的`143/400`。v6-prior已完成formal 0→50；同一
  schedule macro0/10/25/50 strict correct400=`134/127/105/123`，correct80=`26/26/24/27`。小panel在
  macro50看似上升而full400仍下降，进一步证明不能用screen替代正式裁决。
- 四点严格逐row配对分析：0→10 gained/lost=`19/26`；0→25=`19/48`、McNemar
  `p=.000522`；0→50=`20/31`。四点success union=`172`、intersection=`77`、逐task envelope=`147`，
  但union/envelope均不能作checkpoint融合。macro0仍是当前schedule单点winner。
- 内部根因不是“expert loss没动”：generated norm约`140.97→107.00`，cosine仅
  `.02194→.02630`，expert loss下降的约`94.2%`来自log-norm径向项；绝对expert投影系数
  `a=<G,E>/||E||²`反而`.736→.662`且23/24 tasks下降。held state0 macro50相对macro0的norm ratio/
  cosine/radial coefficient/orthogonal residual/base/delta/base均值=`.7180/.9755/.7007/.1551/.3373`。
  因而当前训练主要缩小已有v6 LoRA，而不是补足有用expert方向。
- 当前代码已完成对`0894856`后batch1/逐元素复现策略的撤回。A40已证明batch8只产生普通BF16
  batch-shape roundoff（max`.001953125`、mean约`4.70e-5`，direct-repeat为零）；owner明确要求不为这种
  微差牺牲吞吐。新路径在同一个固定request/总帧panel上从稳定且有显存余量的候选中选择实测LoRAs/s
  最高的batch，使用原生BF16/F32
  LoRA cache、零重复Writer forward、更少host sync和2-worker action prefetch。clean pushed
  `ded0c80`的A40 fixed-panel profile选择batch8；随后8-task纵向smoke完整通过并由retained artifacts
  组装evaluation seal。gradient/resume结构化artifact verifier与只读checkpoint comparator也已完成
  CPU验证。clean frozen`a17805c`随后两次启动六卡macro49 gradient profile：默认allocator和唯一一次
  `expandable_segments:True`重试都在第一个PI05 policy functional B20的Gemma MLP前向发生容量OOM；后者把
  reserved-unallocated从约`1.29GiB`降到约`157MiB`仍无法分配`606MiB`，因此碎片不是主因。两个root均只有
  run contract/invocation、没有gradient/completion，不能seal或resume，也没有产生方法性能结论。
- 当前修复保持logical B20、每task mean、train24×20=`480/480` unique queries和objective分布不变；
  完整有序logical-B20 panel keyed的physical slicing使用FP32 leaf-gradient加权累积，seed为轻量固定64-bit
  整数mix，不调用SHA/MD5。clean frozen `eddba96`的B16+4在六rank第一条functional attention一致OOM：
  allocated=`42.49GiB`、reserved-unallocated=`1.25GiB`、free=`235.31MiB`，尚需`254MiB`。所以当前只把
  physical microbatch改成balanced B10+10；这仍是A40容量实现变量，不是减小scientific batch。
  policy activation checkpointing目前不启用，因为
  OOM在frozen PI05 policy而现有checkpoint flag只覆盖Writer，启用policy重算会是更侵入且可能更慢的变量。
- clean frozen `9c814ff`的balanced B10+10已完整通过macro49：wall=`21.095s`、input wait=`.076s`
  （`.36%`）、peak allocated/reserved=`40.332/43.859GiB`、0 OOM/nonfinite；完整assembler通过。唯一权重为
  expert=`.008355172068998324`、ranking=`.28570466890490887`，两者在compiler各等于positive梯度的`.25`，
  在factor仅`.05254/.03993`。config已原样写入gradient evidence。
- strict后继`5fbcb27`已在`gpu02:0--5`完成fresh0→1+same-root exact-resume1→3和独立contiguous0→3；
  两root合同相同、各3 metrics、macro1/3 checkpoints和completion，0 OOM/nonfinite/clip。contiguous/
  resumed总step wall=`61.368/64.450s`，峰值allocated/reserved=`43.266/47.119GB`，steady-state input wait
  约`.0006s`。所有cursor/RNG/scheduler/AMP/frozen tensors精确相等，trainable Writer与Adam逐tensor科学
  门通过。原比较器误把近零Adam moments套入Writer aggregate relative门，已只修离线state-specific
  tolerance而未改训练或重跑GPU；retained artifacts重新assemble通过，当时的v2 profile/formal已正式sealed。
  该历史状态不解锁当前第38节config。

第33节whole-LoRA direction/norm与第34节Expert-Component Projection（ECP）均已由formal
closed-loop证据退役。ECP保留原v6架构和上游冻结边界，只把auxiliary换为
`a=<G,E>/(||E||²+epsilon)→1`与bounded negative ranking；因此这是对expert component假设的
干净单变量检验，不是新video encoder或LoRA topology实验。

clean pushed/frozen`450e688`的formal root=
`runs/outputs/pi05_v6_ecp_formal_r6_lb20_mb10_450e688_20260809`，fresh0→10后又按预注册门
exact-resume10→25；25 metrics、macro10/25 checkpoints、optimizer/scheduler/sampler/六rank RNG与
completion完整，0 OOM/nonfinite/clip。macro1→10的`a_correct=.736184→.828442`、expert
component=`3.06189→3.44394`、generated norm=`140.973→151.343`，24/24 tasks的`a`向1移动且
component上升。macro10 strict correct=`133/400`、breadth6，对同schedule macro0=`134`的精确
gained/lost=`22/23`、net=`-1`。这证明ECP修复了部分旧objective的径向伤害，但没有建立
held共同改善。

10→25的内部机制继续按目标运作：`a_correct=.828442→.884127`、component=
`3.44394→3.67225`、generated norm=`151.343→159.817`；23/24 tasks的`a`向1移动，24/24
component和norm上升。但expert-orthogonal norm约`151.303→159.774`，增量`8.471`远大于
component增量`.228`。macro25 strict correct反而降到`120/400`、breadth6、per-task=
`0/1/43/27/0/33/15/1`；相对macro0的严格配对gained/lost=`13/27`、net=`-14`、
McNemar `p=.038477`，suite net=`-4/-12/-2/+4`。macro10→25也是`18/31`、net=`-13`，
四个suite全部净下降。

裁决：ECP不续50/100、不扫权重、不为loser补六臂。直接增大expert component权重已被
证据禁止。第35节已完成与历史anchor/tangent/distillation去重，并在同一canonical vertical path
原位实现Condition-Local Dynamic Expert Tangent Tube：historical v6对correct和当前negative的同一
language/video/order输出分别作局部baseline，只惩罚student增量的expert-orthogonal分量。新v3 config、
training-only decoder anchor、trainable-only resume/deployment load、双臂metrics及独立评测family已通过
exact-D/gauge/gradient oracle、seal后formal-lineage guard和全仓`277 passed`、compileall与diff-check。
clean pushed/frozen
`2616773`随后在live空闲`gpu01:0,1,2|4,5,7`完成唯一六卡gradient/whole-macro profile：24 tasks、
480/480 unique queries、8/8/8 negatives、最长105帧，wall/input wait=`21.53076/.60603s`，peak
allocated/reserved=`43,353,948,672/47,112,519,680` bytes，0 OOM/nonfinite，六卡自然回到14MiB。
correct/negative的student与same-input anchor在24/24 tasks上完全一致，全部tube/delta指标exact zero；
projection/ranking唯一权重=`.00686480847114155/.010514453175708578`，assembler完整通过并写回config，
只解锁严格后继的resume profile。

strict后继clean pushed/frozen`c1bdcae`随后在live空闲`gpu01:0,1,2|4,5,7`完成resumed root的
fresh0→1与same-root exact-resume1→3，以及独立contiguous0→3。首段后的inter-phase selected-GPU
preflight发现设备不再满足expected-idle合同并安全停止；重新live检查六卡满足合同后分别启动剩余两段，
3个scientific invocations均exit0，没有重跑fresh或混用root。两轨各3 metrics、macro1/3 checkpoints和completion；
step wall=
`62.34061/61.95860s`、input wait=`.09366/.13220s`、macros/s=`.048123/.048419`，peak
allocated/reserved=`43,316,387,840/47,137,685,504` bytes，0 OOM/nonfinite，结束后六卡自然释放。

retained roots为：

```text
runs/outputs/pi05_v6_tangent_tube_profile_resume_r6_lb20_mb10_c1bdcae_20260809
runs/outputs/pi05_v6_tangent_tube_profile_contiguous_r6_lb20_mb10_c1bdcae_20260809
```

artifact assembler证明两份run contract完全相等，scientific metrics最大tolerance ratio=`.67790`；
macro1/3的cursor、checkpoint contract、6-rank RNG、scheduler/AMP语义相等，559个frozen tensors exact，
41个trainable Writer tensors的macro3 maxabs/relative-L2=`8.5067e-6/1.14428e-6`。82个Adam moments的
最低cosine与symmetric norm ratio均远高于`.999/.99`门。evidence已原样写入v3 config，profile和formal
同时置为`sealed_from_live_a40_resume_profile_evidence`，`runtime_for_mode(..., formal)`返回
`(50,(10,25,50))`；profile checkpoint永久不得进入formal。

以下是formal前profile阶段的历史判断，不再覆盖上面的最新裁决。三步pre-update轨迹中，macro1→2有21/24 tasks把`a_correct`推向1，
但macro1→3为0/24；macro3 correct/negative的orthogonal-relative-anchor task median约
`.03158/.03173`，仅`10/24`和`6/24`低于`.03`，orthogonal-to-direction中位约`60.98/61.2`。
这符合“quadratic tube在anchor处一阶梯度为0、首步可能先发生正交漂移”的结构风险，也说明不能把
resume seal写成mechanism pass。随后formal0→10与paired correct400已经按该门完成，结果如本节顶部；
当前recipe已停止，不能从这段历史表述恢复macro25。

## 2. EMBER problem and information wall

EMBER不是video imitation replay。它要求：

```text
exact task language + one action-hidden teacher video
                    │
                    ▼
               shared Writer
                    │  one pre-rollout forward
                    ▼
       one complete 38-target rank-16 LoRA
                    │
                    ▼
       frozen π0.5 source policy + live observation/state
                    │
                    ▼
             closed-loop task execution
```

Writer只能从语言与视频抽取任务的高层语义、对象关系、阶段和动作顺序；teacher action、proprio、reward、
terminal、task ID、filename、object pose和hidden normalization均在墙外。视频是唯一dynamic value；语言
可以定位“要解决什么”，不能单独形成LoRA旁路。video和functional action query必须错开episode，避免把
教学轨迹的低层运动与监督动作机械对齐。最终要求同一生成LoRA泛化到该任务不同初始化，而不是复现视频
轨迹。

one-shot是当前目标合同。few-shot的合理作用是从多个同任务视频中提取共同程序并消除单视频偶然细节，
但历史K4已证明“看多个视频”本身不解决condition-to-policy credit、共享梯度抵消或正确时序辨识。
因此只有当当前one-shot的同任务跨video方差被closed-loop证据定位为最早瓶颈后，才恢复固定K聚合；不能
通过平均多个无效LoRA或expert route伪造提升。

## 3. Current deployment architecture and active diagnostic

部署图恢复历史v6-fast完整video-to-LoRA生成器：

1. frozen π0.5 multimodal hidden对exact language和raw video形成task-grounded per-frame evidence；
2. Semantic Core聚合跨帧稳定语义；visual transition显式计算相邻变化；Causal Procedure保留动作阶段
   和顺序；
3. 320-slot compiler将Core/Procedure写入public LoRA topology；
4. 8个factor heads直接生成38个policy targets的完整rank-16 A/B；
5. LoRA只生成一次，Writer释放，原source policy原位加载该LoRA做闭环rollout。

唯一部署图仍由历史v6-fast macro400 checkpoint初始化：

`runs/outputs/pi05_as_writer_v6_decay400_taskcomplete_dev_r4_b20_seed7_s2400_4efa737_20260729/checkpoints/step_00000400`

其600个Writer tensors只作load-only初始化。encoder/Core/transition/Procedure共483 tensors、
`7,060,992` parameters冻结；compiler+factor heads的41 tensors、`3,714,304` parameters在audit中仅用于
求梯度，不创建optimizer/scheduler、不更新，也没有active训练方法。

第36节diagnostic在不改变部署图的前提下增加三条严格匹配的比较臂：

- macro0 Writer读取exact language和一条correct action-hidden video，生成完整LoRA；统一step2000 task
  expert是task-local privileged teacher，sealed tangent macro10只加载41个compiler/factor tensors作为
  历史比较decoder，二者都`no_grad`；
- train24每task仍读取20条同task跨episode action queries。logical B20只因A40容量切为B10+10；每个slice
  依次执行expert、tangent10和唯一一次可微macro0 policy forward，三臂复用同一flow time/noise及offset，
  因而正式runtime固定每task 6次PI05 policy forward；
- action projection实际width必须为7；三臂velocity裁到真实7维后仅在小tensor上转FP32计算expert、macro0、
  tangent10 target loss和CEFD distillation loss，不改变BF16主干吞吐；
- positive、completion、ranking和候选distillation四类gradient先rank内4-task等权mean，再一次stacked
  all-reduce/world6形成full24 mean；CEFD向量只与现有三向量的归一化Gram span比较，不做parameter update；
- reversed、shuffled、wrong保持8/8/8 schedule；audit不读held action、不rollout、不创建checkpoint或长期
  cache。部署仍是单一600-tensor Writer，不读expert bank、tangent decoder或任何诊断artifact。

## 4. What task experts solve and do not solve

正式expert root：

`runs/outputs/pi05_task_expert_bank_formal_step1000_r6_81101fe_20260807`

24/24 train tasks均统一续到step2000，共120 checkpoints。统一step250/500/1000/1500/2000的
development-train direct-expert closed-loop为`432/557/624/638/658` of 1200；step2000有23/24 tasks非零、
task9仍为0，因此统一选择aggregate最强的step2000但不把它写成完美oracle，也不按task混点。

Experts解决：

- 在同一frozen source policy和public rank-16 topology上，给出“这个train task的什么参数更新确实能
  闭环工作”的policy-effective target；
- 提供正常task-local SFT LoRA的能量、rank坐标、跨target分配和有效方向参考；
- 避免meta-Writer只被高方差functional query推动、完全不知道有效参数流形在哪里。

Experts不解决：

- held task如何从video生成可迁移LoRA；
- 同一task的多个video之间应关注哪些共同程序；同task expert target对所有video恒定；
- correct、shuffled和reversed为什么不同，或时间顺序是否被真正理解；
- train24 expert是否可直接充当validation expert字典。soft/hard部署的`15/80`和`3/80`已明确否定这一点。

因此expert reconstruction loss和健康几何只能作训练辅助；真正裁决仍是paired closed-loop五臂/六臂。

## 5. Cumulative experiment lineage

下表是设计决策的连续证据链，不是候选排行榜。精确roots和完整分析仍保留在对应design、`findings.md`、
`progress.md`、Git和formal artifacts中。

| 方法/干预 | 最强strict证据 | 实际证明 | 失败或未证明 | 当前保留结论 |
| --- | ---: | --- | --- | --- |
| frozen source base | `48/400` | generic-source policy无目标适配也有非零能力 | 不读video，未检验video adaptation | 所有Writer共享的frozen起点 |
| mixed-task Source-SFT rank128 | `109/400` | direct target action可形成共享适配 | privileged且仍低于目标 | 参数预算/闭环参照，不是同信息墙baseline |
| v5.2 old recipe | `132/138/74/82/83` | 当前最强correct-vs-negative视频特异性 | absolute未过150 | 动态写出与顺序margin可实现，不能遗忘 |
| v5.2 task-complete | `120/109/107/111/124` | recipe会改变video传递 | absolute和margin均退化 | task-complete并非普遍改进 |
| v6 old recipe | `121/122/111/84/47` | 强时序差异可传到闭环 | absolute低、task旋转 | old recipe增强动态也增强不稳定 |
| v6-fast task-complete | `143/135/125/128/129` | 历史最佳eligible raw single-checkpoint absolute；Procedure差异可传到LoRA/action | 原objective续到450/500/550/600降为`131/130/132/126`；冻结上游迁移性仍是假设 | 当前representation prior与macro0基线，不续旧objective |
| CV-ADR RAW / GROUP4 | best`117` / `110` | 更大coherent更新不等于更好闭环 | 曲线漂移；video梯度主效应约`.1%` | query/flow variation主导；仅flow MC noise可约且不是主因，credit仍错位 |
| Target-Bound | best`120` | remove-A/D和memory reversal 8/8达门，动态路径工作 | correct漂移、共享factor共存失败 | 不再把首因归为video完全未使用 |
| Semantic Factor-Basis | best`127`，union`193` | common accumulation一度改善 | envelope gap66、严重换手 | shared credit仍未稳定积累 |
| variance-reduced estimator | best`126` | 精确Beta/antithetic MC只小幅改善gradient consistency | held loss改善但closed-loop退化 | flow Monte Carlo方差不是主因 |
| Semantic Direction Store | best`129` | 独立store改善早期acquisition | 同分checkpoint breadth不同；Program→factor压缩 | parameter coexistence是局部因素，不是根因 |
| Policy-Target-Owned Factor | best`99` | 解除38-target共享显著改善跨层异质性 | action效果和性能仍差 | 健康跨target几何不是充分条件 |
| Policy-Lane | best`70` | 形成约10条有效output lanes和SFT量级专门化 | video BA能量约`.02%` | 容量/形态健康不能替代动态credit |
| Policy-Wide Atom Dictionary | best`80` | 64 atoms广泛使用 | mixing/effective LoRA近rank1 | 不用增atom/rank/正交loss救活 |
| Factorized Condition-Kernel | best`49` | kernel full-rank、stable且跨video差异大 | LoRA约比direct SFT小200×，identity-like | 低增益decoder曾是明确瓶颈，但非唯一根因 |
| Few-Shot Invariant-Program K4 | best`108` | K4置换、same/LOO/wrong/order路径都工作 | full24 gradient retention约`.043` | few-shot可去偶然性，不能自动解决共享credit |
| K4 Policy-Layer Trace | best`99` | all-layer trace带来correct>wrong | reversal仍高、逐频单位化把低能量DCT高频放大约`140×` | 被放大的高频不能替代有物理意义的时序程序 |
| Energy-Preserving Trace | best`85` | 修复原始频率能量比例 | correct/wrong从`99/57`收缩到`85/80`；effective groups`13.97→10.63` | 能量保真本身不等于语义保真 |
| Evidence-Factorized Trace | best`84` | correct>wrong且trace→BA→action闭合 | shared Reader retention约`.05` | 参数隔离值得检验，但不是直接答案 |
| Sparse Semantic-Expert | best`78` | expert-local retention提高 | language route固定owners，wrong/order更成功 | language-only ownership不够；video须参与credit |
| Grounded-Video Expert | best`88` | video route、Reader、BA、action和rank均material | correct无margin、task轮换 | video sensitivity与parameter isolation仍不充分 |
| K4 Phase-Aligned v6 | best`108`; reversed`121` | video未被忽略 | 近rank1、高能量、程序retention约`.04` | phase alignment不能独立解决语义/credit |
| AS125 + semantic progress RL | `97→104→102` | failure轨迹可获得非零semantic credit | breadth下降、继续训练换手 | reward信号存在，但共享更新不稳定 |
| Program-Credit RL | `106` | lockstep CRN和program gradient可达 | task cotangent几乎正交却被压成common update | shared condition map会吞掉task-specific credit |
| SFT-Anchored Tangent-Basis | `143→142` | 在强warm-start上小幅reward更新可运行 | gained/lost`20/21`，无净提升 | 不能把warm-start保持分数冒充生成器改进 |
| task experts step2000 | train`658/1200` | aggregate最高且`23/24` tasks非零的privileged train target | task9仍为0、存在state turnover；不证明video或held泛化 | 保留为监督流形，不作部署字典 |
| addressless Expert-Manifold | `48/400` | raw-expert reconstruction能训练出norm约`4.55` | 与source exact同分、paired`5/5`；topology identity在decoder后坍缩，nearest expert cosine约`.008` | 无显式topology address的decoder已证伪 |
| topology-address binding | `75/400` | 静态chunk/rank坐标可乘性调制video dynamic value并进入闭环 | 输出仍高度task-common，held绝对性能低 | 地址辨识修复有效；不能单独调address解决迁移 |
| Causal Barycentric | `63/400` | temporal coefficients和raw-factor组合可运行 | `k≠j` cross terms使raw A/B组合不保持effective update；未单独隔离held support | policy-effective compiler必须先于组合几何 |
| policy-effective soft / hard bank | `15/80` / `3/80` | hard compiler近精确复现所选expert | 当前causal reader + 24个step2000 experts的soft/hard held support均失败 | 关闭当前24-expert online部署字典，不外推所有未来流形方法 |
| v6-prior whole-LoRA objective | `134→127→105→123` | 冻结上游、只训写出端可高吞吐稳定运行；晚段可部分回升/breadth7 | 整体方向+norm吸引主要径向收缩，macro0仍最佳，绝对expert投影下降 | 退役该objective，不外推v6表示无效 |
| v6 Expert-Component Projection | `134→133→120` | `a_correct`与component按构造上升，修复旧径向收缩 | 正交漂移继续增大，macro25 paired net`-14`、`p=.038477` | 退役；不续、不扫权重 |
| Condition-Local Tangent Tube | `134→131` | relative-anchor tube中位`.01390/.01408`，证明局部半径约束工作且吞吐可接受 | direction ratio=`108.93/126.88`、completion`0/24`、breadth`6→5`；只压小更新而未旋进expert方向 | 已退役；不续25、不扫权重、不补六臂 |
| Expert-Flow Teacher Audit | 无rollout | gradient residual`.6864/.8387`，teacher方向非冗余 | expert flow loss只在`2/24` tasks、`0/4` suites优于两baseline | CEFD否决；一次性runtime已删除 |
| Frozen-v6 Counterfactual-Null Program Residual v1 | 无rollout | correct retention `.807966`、A/B/action/closure成立 | DC key导致condition`1315.33`、null仅15/24，吞吐门亦non-pass | v1退役；不训练、不调lambda/seed/P |
| Balanced DC--Causal Program Residual v2 | zero-memory macro0=`134`、breadth6、历史native逐行0/0 | 13/13机制门、24/24 null、A/B/action/吞吐、真实部署链路与exact baseline identity全通过 | 非零memory的absolute、same-task噪声、多步累积仍未知 | 当前唯一active implementation；formal fresh0→10后立即strict裁决 |

任何需要精确数字的决策必须回到对应design/artifact，而不是从本表反推未列指标。

## 6. Stable cross-experiment cognition

1. **视频被使用不等于视频被正确使用。** 多条路线都证明wrong/shuffle/reverse能显著改变hidden、BA和
   action，但correct仍可能更差。下一分析必须问改变是否沿policy-effective方向，而非只看差异大小。
2. **LoRA健康度是约束，不是目标。** 低能量、过度rank1和高列相似度曾解释部分失败；但形成SFT量级
   能量、多个lanes、跨target异质或正确expert cosine也未自动提高closed loop。不得单指标优化。
3. **functional surrogate长期错位。** held functional loss下降、gradient更稳定或full-rank kernel均曾与
   closed-loop退化共存；关键checkpoint必须及时rollout，不能用loss挑点。
4. **task drift不是一种表象对应一个单因。** query/flow variation（其中只有flow MC noise可约）、full24正交抵消、shared parameter
   coexistence、Program→factor压缩和condition-map common update都被逐步检验；其中每个只解释局部。
5. **正常时序必须有因果意义。** shuffled/reversed真正破坏frame展示顺序；模型不能依靠原时间戳恢复。
   correct必须同时接近有效policy update并超过negative，不能仅把negative推向坏LoRA。
6. **架构与recipe耦合。** v5.2/v6交叉结果证明不能按某一architecture aggregate整体判死，也不能直接
   恢复old recipe；需要对比最早传递接口和任务换手。
7. **当前假设是局部且可证伪的。** v6 checkpoint已达143且上游仍保留material Procedure response；
   冻结部分是否足以迁移仍是假设。目前只把shared parameter/Adam输运替换成显式condition kernel，不改
   video encoder、Procedure、FactorHeads、functional target或scientific batch。若Program/action传递成立而
   closed-loop仍无共同提高，才重新打开reward credit或更早表示，不因一次低分立即换整条路线。

## 7. Current engineering state

当前第38节v2实现状态如下：

- 唯一CLI仍是`scripts/train_v6_prior_writer.py`，活动mode仅`mechanism-profile/formal`；一次性
  `teacher-audit`、flow teacher和旧effective objective owner/tests已经删除，没有第二runner或部署路径；
- `writer/condition_update.py`拥有balanced static/causal fixed feature、single FP32 Program memory和full48 solve/apply；
  `v6_prior_step.py`拥有correct graph/counterfactual feature，training只做local objectives、两次all-gather和
  identical manual write；`v6_prior_profile.py`只拥有一次性verification和gates；
- 600-tensor historical v6 strict-load后全部冻结，fixed projection nonpersistent，Program memory是唯一
  checkpoint/deployment mutable state；fresh incompatible checkpoint只保存memory/cursor/六rank RNG，
  不存在optimizer/scheduler/scaler或memory all-reduce；
- config/runtime/contract/checkpoint/adapter/evaluator/analysis状态机已经联锁；fresh必须等于当前remote
  authority，exact-resume保持原frozen commit且要求它仍为authority ancestor；错误family和stale artifact
  fail closed；
- CPU seal为聚焦`52 passed`、带LIBERO assets全仓`281 passed in 21.34s`、compileall、26份JSON和
  diff-check通过，architecture guard无hard violation。v2 mechanism与deployment artifacts均已seal，config为
  `active_deployment_sealed_formal_ready`；只有validation8×state0 execution smoke，没有v2 formal strict结果。

以下是仍被当前候选继承或用作比较的historical throughput/runtime provenance，不是当前seal：

- evaluator取消historical smoke中的8次冗余direct Writer forward和`1e-5`逐tensor门；batch默认8并要求
  profile至少实测`8/16/32`。三者处理同一32-request longest-first panel和同一总帧数，只改变实际
  forward分批，最终从稳定且有显存余量的候选中取LoRAs/s最高值；
- 76-tensor LoRA保持template原生dtype：72个BF16、4个F32，单entry tensor bytes从强制FP32的
  `5,148,672`降到`2,641,920`；batch GPU→CPU staging只同步一次；
- functional objective仍是logical B20；live A40已依次证明single physical B20和B16+4不能容纳，当前使用
  balanced physical B10+10，通过完整有序logical panel identity重放同一20个独立Beta/Gaussian draws，
  并用76个FP32 leaf-gradient buffers按`10/20`和`10/20`加权累积；
- PI05 formal functional路径不再调用只供日志使用的`.cpu().numpy().tolist()`/`.item()`；loss-only实现与
  原forward的loss及LoRA leaf gradients由固定noise/time测试验证一致，通用details接口保持不变；
- correct effective alignment只计算一次，task metrics和gradient norms合并成少量host transfer；
- action DataLoader默认2 workers、spawn、persistent workers和prefetch2；确定性sampler的serial、
  prefetched和prefix+resume rows已精确一致；
- Writer offsets、frame ordinal/order、language span/condition ownership的重复CUDA→host门已合并或
  hoist到CPU，vectorized token packing不再逐row产生动态CUDA selection；必要的D2H handoff和宏步wall
  synchronization保留；
- 新`profile-writer-generation`在同一loaded source policy/Writer上做真实video→LoRA→native D2H sweep，
  记录fixed-panel actual forward batches、repeat wall、longest video、peak allocated/reserved和headroom；
  launcher及独立单卡worker均在模型load前live检查空闲NVIDIA A40，worker还核对clean pushed checkout；
  普通evaluator在spawn前拒绝忙卡、非A40或合计超过6张卡，历史8卡root也不能绕过该启动门；
- evaluation seal只能由profile root与vertical smoke root的retained artifacts组装，校验三候选完整request/
  sampled-frame panel严格相同、warmup/repeats、最长视频、selected throughput、native cache、release/reuse
  和单次成功launcher；
- config状态图已修为`blocked → evaluation sealed/gradient ready → gradient+aux sealed/profile ready →
  profile sealed/formal ready`，不再自我依赖；task-expert authority使用窄loader，不会因已退役topological
  Writer/meta字段变化阻断当前训练监督；
- clean frozen `ded0c80`在live空闲`gpu02:0`完成32-request、1093 sampled-frame fixed panel。
  batch8/16/32吞吐分别为`.911427/.905107/.906432 LoRA/s`，三者均稳定且峰值reserved约
  `12.82--12.85GB`；batch8按封存规则实测最快。大batch没有吞吐收益，剩余显存本身不是选慢配置的理由；
- 同提交fresh vertical root完成8 videos→8 LoRAs/cache→8 rollouts，单次attempt、`0` retry/failure/
  OOM/nonfinite/forbidden reads。Writer生成`10.597s`，peak allocated/reserved=
  `11,651,564,544/12,811,501,568` bytes；release后source policy原位复用且未reload。总wall=
  `325.540s`、rollout window=`196.816s`，进程结束后GPU回到0MiB；
- artifact assembler已从两个单卡retained roots重建evaluation evidence；gradient assembler又从macro49
  retained root重建权重和完整provenance；
  gradient assembler现会精确重建macro49的24-task teacher-demo/counterfactual schedule、480 unique
  queries、canonical config、clean pushed Git、frozen target manifest/HDF5 frame metadata与六卡拓扑；
  resume assembler会比较fresh/resume/
  contiguous的contract、cursor、6-rank RNG、600 Writer tensors、41 trainable tensors、Adam moments、
  scheduler/AMP和scientific tolerance，并要求gradient→profile的strict Git ancestry；
- frozen`a17805c`在当时live空闲`gpu01:0,1,2,4,5,7`的3+3 NUMA拓扑完成了两次有效工程诊断。默认allocator
  OOM时PyTorch allocated=`42.29GiB`、reserved-unallocated=`1.29GiB`、free=`395.31MiB`；唯一allocator
  retry为allocated=`43.43GiB`、reserved-unallocated约`157MiB`、free=`389.31MiB`，仍请求`606MiB`失败。
  这关闭“只调allocator即可保留physical B20”，但不关闭logical B20、当前Writer或任何科研假设。两次
  launcher退出后所选六卡均释放，未触碰当时由他人占用的GPU3/6。
- clean frozen`eddba96`在新一次live preflight后复用当时仍为空闲的同一3+3拓扑。首个非持久SSH后台
  launcher只写contract/invocation便exit0，没有start/gradient/completion，作为无效进程托管证据保留；
  改用tmux的fresh retry完整进入start，六rank均在第一条functional eager-attention申请`254MiB`时OOM，
  allocated=`42.49GiB`、reserved-unallocated=`1.25GiB`、free=`235.31MiB`。因此B16不存在可比较的吞吐点，
  当前canonical config已转为B10+10；上述root都不能seal、resume或选择auxiliary weight。
- clean frozen`9c814ff`随后用同一拓扑完成B10+10。24-task/480-query/105-frame panel、Git/config/
  HDF5、NUMA/NCCL、default allocator和single invocation全部经assembler闭合；input wait仅`.36%`，
  不再测试workers4。macro0 generated/expert effective norm mean=`140.52/4.182`、cosine=`.02196`，而
  reversed/shuffled margin仅`.000832/.000634`；这是当前要由受控expert/ranking更新纠正并由closed-loop
  证伪的核心机制矛盾，不是新增性能成绩。
- clean frozen`5fbcb27`的正式retry1比较root位于
  `runs/outputs/pi05_v6_prior_profile_resume_r6_lb20_mb10_5fbcb27_retry1_20260809`和
  `runs/outputs/pi05_v6_prior_profile_contiguous_r6_lb20_mb10_5fbcb27_retry1_20260809`。macro3 Writer
  maxabs/relative-L2=`4.6033e-5/1.06393e-5`，只占两步update L2的`1.023%`；Adam maxabs=`2.6865e-6`，
  `.007719` relative值来自近零moment分母。离线v2门改为Writer global relative L2`≤.002`，Adam每个
  moment的symmetric norm ratio`≥.99`且cosine`≥.999`；raw maxabs/relative-L2只诊断，并保留逐tensor
  `2e-4/2e-3`、全部语义exact和frozen exact门。config现为profile/formal
  `sealed_from_live_a40_resume_profile_evidence`；该seal只证明工程连续性，不是性能成绩。
- v2 retained assembler、config load和状态机均重新通过；聚焦checkpoint/contract为`11 passed`，加载
  `.env.local`后的全仓CPU回归为`247 passed`。未为本次seal启动任何额外GPU工作。

被撤回的失败root仍保留科学诊断：

`runs/outputs/pi05_v6_prior_warmstart_reproduction_smoke_validation8_correct_gpu02g0_30b2ccf_20260809`

其中`diagnostics/batch_equivalence.json`只证明BF16 batch-shape差异，不是performance evidence。

## 8. Key uncertainties and structural risks

- **frozen prior风险**：v6-fast提供143高增益起点，也可能把task-complete造成的弱时序margin一起冻结。当前
  residual只能在fused Program后修正，若真实order feature健康而正确时序知识根本没进入Program，显式kernel
  也无法创造上游语义。
- **feature sufficiency风险**：固定256维JL key只保留四个时序矩和frame-evidence/text-query innovation。
  它原理上区分content/order且无language-only value，但可能对细粒度阶段、接触事件或same-task视频变化
  不充分。首先看真实full48 rank、task-local retained/null及五臂，不能按漂亮Gram宣告成功。
- **functional-to-closed-loop错位**：Program cotangent是真实source-action pointwise functional descent，
  但历史已反复证明functional loss可与closed-loop错位。即使application、LoRA和fixed-action传递都通过，
  strict correct仍可能不升；这时应把证据指向credit objective，而非继续调rank/能量或延长训练。
- **累积与漂移风险**：manual memory没有Adam、momentum、global scale或cap，避免shared optimizer旋转，
  也意味着不同macro的kernel writes可能在未见condition上叠加、抵消或放大。只在10/25/50边界评测完整
  paired correct400，并以per-task gained/lost、breadth和相邻checkpoint churn判断共同积累。
- **counterfactual覆盖风险**：每个task/visit只取wrong/shuffled/reversed之一，full48只把当前24个negative
  motion压近零；它可能学习局部异常特征而非可迁移程序。必须由same-task-other、cross-suite wrong、
  shuffled、reversed和no-video同checkpoint六臂裁决。
- **验证指标边界**：4-task fixed-action probe只证明Program motion穿过frozen decoder和policy，不是性能、
  breadth或视频因果结果；8次verification forward不计入production throughput。最终目标仍只有strict
  closed-loop及其真实配对控制。
- 吞吐优化允许BF16低位、batch shape和parallel reduction普通差异，但不能改变logical B20、信息墙、完整
  LoRA topology或paired evaluation；不得为逐元素复现牺牲batch/显存利用。

## 9. Evidence order and next action

证据关系固定为：

1. full-bank geometry、expert closed-loop和统一step2000续训已经完成；它们只证明train expert target的
   policy effectiveness，不进入当前cotangent/deployment，也不能证明held video causality。
2. whole-LoRA/ECP/Tangent/audit连续证据把首个失效接口定位到shared condition update transport，并否决
   expert-flow teacher；这是选择显式condition kernel的依据，不是新方法成绩。
3. v1 macro49已经证明full48 algebra、zero identity、freeze、LoRA/action传递，但因DC主导key导致null失败；
   v2只修key且CPU seal不能跳过新的live A40机制门。
4. v2六卡macro49 profile先验证真实feature rank、至少18/24 task-local correct retention和negative null、
   4/4 fixed-action breadth、application closure、0 negative forward/OOM/nonfinite及production ratio`≤1.10`。
   profile memory永久弃用，不进入formal。
5. mechanism seal后单卡实测新residual deployment graph的batch8/16/32和correct smoke；旧v6吞吐seal只作
   参照。两类profile都不证明方法有效。
6. 两类seal齐全后，先得到同schedule zero-memory macro0，再formal0→10并立即跑paired correct400；与
   macro0=`134`、历史143、ECP10=`133`、Tangent10=`131`逐task/per-suite/breadth/gained-lost比较。
7. macro10`≤129`且广泛净损失即退役；`130--134`仅在breadth/churn/mechanics同时改善时续25；`≥135`
   续25；首次`≥144`补完整六臂。single winner严格`>150`后必须以同checkpoint六臂确认真实视频/时序因果，
   再继续提高；未过门按最早失败接口做单变量修正，不扫P/lambda/eta或180度转向。

当前具体顺序：先把上述macro0 formal artifact、严格0/0 paired identity和当前authority clean commit/push；
从新严格后继建立fresh formal detached worktree/root。实时重查`gpu01/gpu02`与`/data1` quota，只在六张
空闲健康A40上运行Program memory fresh0→10；保持train24×logical B20、physical B10+10、六rank×4 tasks、
full48 exact write、0 negative policy forward和既有NCCL/NUMA映射。macro10完成后立即跑同一correct400，
按第38.3门与native macro0=`134`逐task/breadth/gained-lost裁决，不先看80-row screen。设备不空闲、拓扑
不符或storage不足都fail close，不触碰他人进程。

## 10. Canonical assets

- source policy：由当前config/CLI显式传入的frozen generic-source step1000 asset；不是
  `pi05_libero`，也不支持source-SFT exact resume。
- task experts：上述formal root的统一step2000 checkpoints。
- historical Writer prior：上述v6-fast macro400 checkpoint。
- retired config：`configs/pi05_v6_condition_local_tangent_tube_writer_v3.json`；Tangent和teacher audit均
  formal non-pass/fail-closed。
- audit/tangent comparison assets只作retained provenance，不进入第37节runtime。
- active config：`configs/pi05_v6_counterfactual_null_condition_kernel_program_residual_v2.json`；mechanism与
  v8 deployment双root均已seal，formal runtime ready；zero-memory macro0 strict400已封存并exact identity，
  当前执行顺序进入fresh0→10。
- training/evaluation entries：`scripts/train_v6_prior_writer.py`与`scripts/evaluate_pi05.py`。
- target split：`configs/libero_24_8_8_v1/`。
- current source policy、tokenizer、data、video和simulation asset的BCI绝对路径均由CLI或`.env.local`
  提供；历史A100路径只作provenance，不原位改写。

旧formal outputs、diagnostics和设计文档保留。旧活动worktree、临时cache或重复本地branch只有在确认无进程、
无未合并唯一改动且远端/Git/artifact已保存证据后才清理。

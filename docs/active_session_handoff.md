# EMBER Active Session Handoff

更新时间：2026-08-09 UTC。本文只记录迁回 BCI 前后的当前真相。历史执行流水仍在
`progress.md`，证据与解释仍在`findings.md`及各架构设计文档；不要用其中旧的
“当前”“下一步”覆盖本文。

## 0. 当前交接边界：Video-Conditioned Expert-Manifold Topological Writer

**最新覆盖（2026-08-09，优先于本节全部后续历史条目）**：Causal Barycentric正式strict
correct400已经自然完成并负裁决。唯一root为
`runs/outputs/pi05_expert_manifold_causal_barycentric_correct400_noreplacement_seed7_0397be6_20260809`；
结果=`63/400`、breadth=`5/8`，逐task按Spatial/Object/Goal/Long为`[0,6]/[38,0]/[0,17]/[1,1]`。
72/72 jobs、400 unique rows、400 unique action-hidden-video LoRAs和18 workers均attempt1/exit0，
0 retry/error/OOM/nonfinite，四类forbidden reads为0，GPU已释放。相对strict same-video source/
addressless panel均为gained/lost=`46/31`、exact `p=.1100`；相对address-binding `75`为`27/39`、
`p=.1753`。它没有可靠提高aggregate，也把Goal-6能力从`47`降到`17`，故不做其余五臂。

full400 coefficient inversion与gauge-invariant几何位于同root的
`generated_lora_geometry_and_coefficients_full400_v1.json`。LoRA norm/stable-rank/top-energy中位=
`3.958/1.155/.894`、16/16 rank coordinates active、top4 energy=`.271`，q/v/action B-column cosine=
`.712/.744/.351`；这些已接近step2000 expert形态，明确排除“能量不足、近rank1或rank-coordinate
塌缩”为当前首因。真正异常是每条视频的24维坐标effective abs support中位`13.75`、negative mass
`.162`，same-task不同video/cross-task/task-mean cosine=`.988/.685/.697`，而真实experts跨task中位约
`.100`。语义近邻有时有效（Object-1=`38/50`），同样合理的近邻也可完全失败（Object-3=`0/50`）；
task-local expert质量并不保证跨物体、跨场景或组合迁移。

最早结构断点现定位为**raw A/B factor barycentric compiler**。对coefficients `c`分别混合factor会
形成`B(c)A(c)=sum_{k,j} c_k c_j B_k A_j`，引入大量没有任何task expert监督的cross-expert项；
想表达的policy update却是`sum_k c_k B_k A_k`。因此“系数语义合理 + raw LoRA谱健康”仍可产生错误的
策略更新。下一唯一候选为**Policy-Effective Barycentric Topological Writer**：先保持完全相同的
one-shot causal representation与coefficient rule，只在有效`BA`空间组合，再用共享joint rank-16
subspace或exact mixture compression编译回同一public LoRA。先做CPU投影能量/cosine与mixture fidelity
门；未通过不实现，当前没有GPU工作。系数reader的CPU反事实已表明时序识别与有效更新幅度是两个接口：
contrastive reader可使correct/reversed/shuffled方向约`.394/-.392/-.008`，但held norm ratio仅`.106`；
rectified prototype可令reversed近identity并保留correct cosine约`.381`，但仍不能解决跨task expert
transfer。故本轮不同时更换reader，不用时序margin掩盖compiler因果。

**最新覆盖（2026-08-09，优先于本节后续历史条目）**：address-binding learned Writer的
macro50 strict correct400已完整结束并正式负裁决。唯一有效root为
`runs/outputs/pi05_expert_manifold_writer_addressbind_correct400_noreplacement_seed7_macro0050_925e7b1_20260809`，
结果=`75/400`、nonzero breadth=`4/8`，逐task按Long/Goal/Object/Spatial为
`[2,0]/[1,47]/[25,0]/[0,0]`。72/72 jobs、400 unique rows/LoRA、18 workers均首次完成，
0 retry/error/OOM/nonfinite；teacher frames和四类forbidden-read证据闭合。相对旧addressless
macro50，它在exact task/state/video/env/policy-noise公共前缀panel上gained/lost=`31/4`、exact
`p=3.47e-6`；相对source base也是`31/4`。该结构修复有真实净增，但远低于v6-fast `143`和长期
`>150`门，且能力仍集中于Goal-6与Object-1，所以原root不得resume到100，也不做该checkpoint五臂。

完整400-LoRA精确effective-BA审计位于同root的
`generated_lora_geometry_full400_v1.json`。norm/stable-rank/top-energy中位=
`3.201/1.318/.778`，16/16 coordinates active，说明失败不是能量不足或近rank1；但same-task不同
video cosine中位=`.99791`、cross-task=`.94197`、八个task均值之间=`.94270`，nearest train-expert
cosine仅`.12734`。对比真实step2000 experts跨task中位约`.100`，最早剩余断点是learned decoder
把不同task压成公共LoRA方向。macro3八taskcosine曾为`.54184`而macro50升至约`.94`，训练正在向
共同均值收缩，不支持“再训一段自然分离”。严格panel与配对审计为同root
`strict_panel_and_paired_audit_v1.json`。

下一唯一canonical候选已由CPU leave-one-task-out证据选定为
**Causal Barycentric Topological Writer**，仍属于同一Video-Conditioned Expert-Manifold方法，
保持one-shot、exact language+一条action-hidden video和video-only dynamic value。它不再训练
68M decoder直接回归1.29M坐标，而用phase-centered causal video representation在24个train-video
centroid上求ridge=`.3`的affine barycentric coefficients，再分别混合168个expert chunk的方向与
log-scale并重构完整rank-16 LoRA；zero/phase-constant representation令全部coefficients为0并精确
返回source identity，language没有独立value路径。LOO artifact为
`runs/outputs/pi05_expert_manifold_causal_barycentric_loo_step2000_cpu_20260809/analysis.json`：每折只用
其余23 experts，topological correct/reversed/phase-shuffled effective target cosine中位=
`.38302/.09900/.18539`，correct margins=`.28403/.19763`；correct LoRA
norm/stable-rank/top-energy=`3.84385/1.15056/.89540`、16 coordinates active，已同时进入expert方向与
能量形态。它只是train-task LOO机制证据，phase shuffle也是16-slot proxy，不是validation closed-loop
成绩；Goal/Long若干task margin仍弱。

canonical实现已由clean pushed`1d9d030`完成。新config为
`configs/pi05_video_expert_manifold_causal_barycentric_v1.json`；runtime不再读取learned Writer
checkpoint，而是显式检查统一step2000 expert bank、train24×50 feature cache和在线一条teacher video。
learned Writer trainer/checkpoint/model可执行路径及旧CLI参数已原位删除，Git与formal artifacts保留历史。
全仓`180/180` CPU测试、`py_compile`、diff check通过；architecture guard无hard violation、无parallel
family，active diff净删941行。真实24-basis只读检查得到0 learned parameters、1,287,168 valid values、
24个one-hot expert最大重建误差`2.235e-8`、zero identity逐tensor exact、coefficient sum误差
`1.192e-7`、24/24 demo0 ordered/reversed coefficients不同，四类forbidden reads均为0。

online链路已在implementation commit`3c8ce25`上用live空闲`gpu02:0`完成。唯一smoke root为
`runs/outputs/pi05_expert_manifold_causal_barycentric_online_smoke_gpu02_3c8ce25_20260809`：validation
8 tasks×1 state、correct/without-replacement、8套唯一FP32 LoRA、2个batch4、3 workers全部exit0，
0 retry/failure/OOM/nonfinite，四类forbidden reads均为0；generation wall=`9.895s`，peak allocated/
reserved=`10,645,668,864/11,305,746,432` bytes，Writer/encoder释放后source policy原位复用且未reload。
GPU随后回到0MiB/0%且无进程。`1/8` success只作execution smoke，不作性能结论。

8套生成LoRA的norm/stable-rank/top-energy中位=`3.9802/1.1555/.89243`，16/16 coordinates active，
top4 coordinate energy=`.27103`；cross-task effective cosine中位=`.69277`，nearest step2000 expert
cosine中位=`.65624`。这相对learned address-binding Writer的`.94197/.12734`同时改善task separation与
expert-manifold落点，但样本仅8，不能替代closed-loop。精确evidence已写回新config并将formal状态设为
`sealed`；真实24-basis、validation8 panel的`require_formal=True` inspector也已通过。下一唯一科研
裁决是从clean pushed frozen worktree启动全新strict correct400并同步审计400套LoRA。05:10 CST live
preflight已预选`gpu01:0,1,2|4,5,7`六张空闲A40并确认3+3 NUMA、479GiB host available memory及
`561,350,572/1,073,741,824 KiB`个人quota；精确branch/worktree/root/命令和验收门取`task_plan.md`顶部
launch合同。启动前仍须再live复核，当前没有GPU进程。

- **最新覆盖（2026-08-09）**：macro50地址塌缩的单变量根修已由clean pushed`cd95281`实现并完成
  两道新图专属A40工程门。六卡fresh0→1/resume1→3与独立contiguous0→3科学指标、Writer/RNG及
  optimizer/scheduler语义一致；单卡online smoke又完成8/8唯一task/state rows、8套唯一FP32 LoRA、
  cache/release/source-policy复用与3-worker rollout，0 retry/failure/OOM/nonfinite，信息墙四类读取均为0。
  两组evidence均显式绑定
  `normalized_dynamic_times_normalized_chunk_plus_rank_address`，meta formal config现已重新seal。
- smoke root为
  `runs/outputs/pi05_expert_manifold_writer_addressbind_macro0003_online_smoke_gpu02_cd95281_20260809`：
  generation为8 entries/2个batch4、`9.731s`，peak allocated/reserved=
  `10,576,056,320/11,182,014,464` bytes；Writer/encoder释放后source policy原位复用且没有reload。
  8行中`1/8` success只作执行证据，不作性能结论。8套macro3 LoRA全finite，初步几何为norm中位
  `.7007`、stable rank中位`1.983`、top singular energy中位`.512`、16/16 coordinates active；它只说明
  已消除旧decoder的结构性单lane塌缩，尚不能证明expert proximity或closed-loop性能。
- profile与smoke设备均已自然释放，profile权重永久弃用。下一正式动作必须从含新seal与launch record的
  clean pushed frozen worktree做identity-fresh 0→50；不加载旧macro50或任何profile checkpoint。启动前
  仍实时比较`gpu01/gpu02`、最多使用6张空闲A40，并重新核验quota、3+3 NUMA与输出root不存在。
- 03:19 CST最新live比较选择`gpu01:0,1,2|4,5,7`六张空闲A40的3+3 NUMA；物理3他人VLLM、物理6及
  `gpu02:6/7`他人进程均不触碰。个人quota为533.2GiB/1TiB，fresh formal root/log/worktree/branch/tmux
  全部不存在，预计新增低于300MiB。scientific seal=`448f760`，唯一exact command与验收门已登记在
  `task_plan.md`的“Address-binding identity-fresh formal0→50”段；真正启动前仍须再live复核。
- fresh formal已由clean pushed launch-record`925e7b1`自然完成0→50：50/50 finite、1,200 one-shot
  conditions、完整Writer/trainer/六rank RNG checkpoint、0 OOM/nonfinite；body=`10.204s`，peak
  reserved=`836,763,648` bytes，六卡已释放。profile/旧macro50权重均未加载。
- train24 demo0内部artifact显示最早接口已按预期改变：axial后chunk/rank centered energy中位仍只有
  `5.64e-6/6.14e-6`，乘性address后恢复到`.493/.477`，最终output为`.467/.616`；own-expert
  effective cosine中位由旧图约`.0108`升到`.1342`，LoRA norm/stable-rank/top-energy=
  `3.360/1.349/.757`且16 coordinates active。风险也很清楚：24个generated task LoRA两两effective
  cosine中位仍`.8686`、仅8/24最近expert是本task，因此不能用内部改善宣告成功或直接resume。
- 下一唯一科研裁决是该macro50的strict correct400。evaluation frozen branch/worktree、fresh root、
  6卡×3 replicas/3 generators、batch4 exact command与72-job/400-row信息墙门已登记在`task_plan.md`；
  启动前重新live看卡。只有closed-loop absolute/breadth共同支持才进入resume和五臂。
- 当前唯一方法authority仍是
  `docs/action_forecast_writer_video_expert_manifold_design.md`。它保持one-shot，视频是唯一dynamic
  value：frozen π0.5逐帧提取2048维joint multimodal hidden与1024维Action-Expert suffix hidden，
  均减matched no-image baseline并保留phase16；language只能参与query/context，不能单独生成LoRA。
  第一次meta profile前已进一步封住静态与unordered-set捷径：full projected innovation只生成phase
  keys，attention value固定为phase-centered dynamics的sqrt-normalized causal-prefix integral；
  zero/phase-constant输入精确identity，即使learned phase key被忽略也没有原始frame-set value路径。
- 24套train-task rank-16 policy experts已从同一source/identity完成统一step2000。唯一root仍为
  `runs/outputs/pi05_task_expert_bank_formal_step1000_r6_81101fe_20260807`：6个独立workers各4 tasks，
  24/24 completion、step250/500/1000/1500/2000共120个checkpoints、约938MiB。1000→2000严格从
  clean`81101fe` frozen contract沿原root exact-resume，未使用当前Writer代码伪装原训练合同。
- 五个统一checkpoint最后50步的24-task等权mean action loss依次为
  step250/500/1000/1500/2000=`.115355/.107207/.105372/.103881/.103526`。晚期loss仍小幅下降，
  但它只说明task-local AS拟合，不能替代development-train closed-loop统一选点。
- Expert-Manifold完整实现现已并入`codex/bci-continuation`：train24 bank evaluator、全bank LoRA
  geometry、action-hidden phase16×3072 feature cache、168个`[16,512]`chunk/rank axial decoder、
  六rank task-complete exact-resume meta trainer，以及one-shot五臂严格配对evaluator均已存在。
  feature cache profile与formal已完成；六卡meta A40 exact-resume core profile和单卡macro3 online-
  generation/cached-rollout smoke曾对旧decoder通过，但已按顶部覆盖失效，当前meta formal为blocked。
  旧K4/AS/RL executable、入口、
  配置和专属测试已按design第12节原位退役，当前dynamic Writer只有Expert-Manifold；identity-fresh
  formal0→50与macro50 strict correct400均已完成；该checkpoint正式负裁决，不得resume100。
- full24×50 cache的CPU审计表明phase-DC能量中位`.98057`、temporal residual中位`.01943`，但
  ordered/reversed/phase-shuffled temporal-template cosine中位=`.88284/-.32402/-.02194`；时序
  task geometry与expert B target geometry Spearman=`.45087`。固定causal-prefix uniform-pool的
  template correct/reversed/shuffled=`.96263/-.94287/-.04463`，linear B proxy=
  `.38820/.06042/.19110`，3/5-shot correct只到`.39379/.39558`。这支持先强制顺序绑定并保持one-shot。
- phase-centered causal-prefix canonical实现与第六个`no_video`反事实已在隔离分支闭合：no-video保留paired
  task/state/RNG/demo ordinal与exact language但不读frames，以zero innovation完整运行Writer并必须
  生成identity。meta六卡入口也已补齐GPU-local NUMA fail-fast及逐rank physical/local/NUMA/affinity
  run-contract记录；profile的step wall与peak allocated/reserved显存均跨全部rank取`MAX`；
  evaluator已修正为smoke只接受声明的profile macro、formal只接受声明且sealed的formal macro，
  因而可在正式训练前实测online generation拓扑。architecture gate无hard violation；随后累计到
  聚焦49/49与全仓223/223 CPU测试通过。六卡core profile已有GPU工程结论，但尚无meta训练或性能结果。
- profile前的cached-rollout纵向审计发现统一adapter wrapper新增Expert-Manifold dispatch时漏传了既有
  `evidence_schema`参数：LoRA cache可正常生成，但释放Writer后进入scale-out episode evidence构造会
  `TypeError`，所以任何由此产生的profile都不能成立。当时的scoped修复让Expert-Manifold schema
  显式fail-close；旧Writer兼容分派现已随canonical退役删除。该修复不改变模型、输入、LoRA、训练或
  rollout数值。
- full24统一五点正式geometry已完成，artifact为
  `runs/outputs/pi05_task_expert_bank_geometry_full24_steps0250_0500_1000_1500_2000_1362d15_20260808/analysis.json`。
  effective-LoRA norm中位=`2.792/3.652/4.170/4.212/4.212`，stable rank中位均约`1.129`，
  top singular energy均约`.909`，跨task effective cosine中位=`.108/.095/.100/.100/.100`。
  16个rank coordinates全部active、top4 coordinate energy约`.262/.260/.258`，但q/v B-column
  cosine仍高且随训练不降。1000→1500 effective update energy已很小，1500→2000几乎收敛；geometry
  说明bank稳定但仍不能替代closed-loop checkpoint选择。
- development-train direct-expert三点正式闭环已在clean`1362d15`完成。有效roots统一使用
  `...step0250/0500/1000_formal_r3_1362d15_20260808`，每点1200 rows、108/108 shards、6 workers
  exit0、0 retry/failure且task/state/RNG严格配对；结果=`432/557/624`，四suite依次为
  Spatial=`123/147/170`、Object=`125/191/208`、Goal=`142/163/164`、Long=`42/56/82`。
  500→1000为`143/76` gains/losses、18/4/2 tasks升/降/平、非零breadth=`23→24`；该分布式
  改善正式触发全部24 experts统一exact-resume1000→2000，并在1500/2000再次闭环选择。
- 1500/2000两点现已从同一clean evaluator自然完成，分别=`638/658`。本轮每点因3卡×3 replicas
  生成126个唯一queue jobs与9 workers，均126/126、attempt1、exit0、1200 unique rows、0
  retry/failure；跨五点task/state/env/policy-noise公共前缀pairing mismatch=0。1500→2000四suite
  为Spatial=`178→181`、Object=`216→228`、Goal=`164→166`、Long=`80→83`，paired
  gained/lost=`77/57`、tasks升/降/平=`17/5/2`。统一target已选择step2000；不按task混点。
- 首次每点12 workers（总36）在0 scientific rows前耗尽gpu02安全主机内存；第二次每点8 workers
  （每卡4 replicas）在首个inference因每卡约37.7GB静态占用而A40 OOM。两批roots均有
  `ABORTED.md`且不得resume；有效r3使用每卡3 replicas、总18 workers，约30.3GB/卡。
- feature cache最小A40 profile也已在clean`1362d15`完成：root=
  `runs/outputs/pi05_expert_manifold_feature_profile_task00_1362d15_20260808`，4条视频的task wall=
  `4.372s`，peak allocated/reserved=`10,468,548,096/19,232,980,992` bytes，输出
  `[4,16,3072]` BF16且action/state/reward/terminal reads全0；formal cache config已seal。
- train24×50正式feature cache已在clean pushed`222d3ac`用6个独立workers完成并seal：
  `runs/outputs/pi05_expert_manifold_feature_cache_train24x50_r6_222d3ac_20260808`。24/24 task
  records与6/6 summaries齐全，每task feature=`[50,16,3072]` BF16，task ordinal恰好覆盖
  `0--23`，50 demo ordinals恰好覆盖`0--49`，cache约113MiB。peak allocated/reserved=
  `10,504,039,936/19,232,980,992` bytes，teacher action/state/reward/terminal reads合计0；
  canonical `cache_manifest.json`已由仓库seal入口生成，无worker error。
- 当前唯一formal macro50 Expert-Manifold Writer的strict correct=`48/400`，与严格配对source base
  同分且gained/lost=`5/5`；per-task为Spatial=`0/0`、Object=`4/0`、Goal=`0/42`、Long=`2/0`。
  72/72 jobs、400 unique rows、18 workers exit0且forbidden reads全0，因此这是有效科研non-pass。
  已验证single-checkpoint最好仍是v6-fast`143/400`，长期严格门`>150/400`尚未完成。
- 全400生成LoRA norm中位`4.549`并不低，但stable rank=`1.0000014`、top energy=`.9999986`、
  nearest train-expert cosine中位仅`.00797`。train24 demo0 effective target cosine也只有`.01081`；
  rank/chunk地址在cross-attention后centered energy约`1e-6`，axial输出进一步到`1e-8/1e-10`，
  对比expert target约`.936/.994`。最早失效是video dynamics没有保留topological address，而非
  validation泛化或LoRA能量；原轨迹不续训。zero-preserving video×address乘性绑定现已按顶部覆盖实现，
  但尚无新profile、训练或strict成绩。
- owner已在本session完成讨论后明确恢复持续自主执行，并设定长期Goal：同一single checkpoint的
  strict paired correct必须严格超过`150/400`，同时保持真实视频时序因果性、same-task鲁棒性、
  task breadth和低checkpoint漂移；只有实质性阻塞才回报owner。当前证据顺序固定为：先做full24
  expert geometry、development-train closed-loop统一评250/500/1000、train24×50 feature cache和
  全24 experts统一resume2000、1500/2000闭环和唯一step2000 target选择、meta-Writer六卡core
  profile、online-generation/cached-rollout smoke、formal0→50和macro50 correct400均已完成；当前按
  已定位的topology-address接口做单一结构修复，再fresh profile/train/strict评测。
  不得按单task挑不同expert step。
- 统一expert continuation已于2026-08-08 22:39 CST自然完成：24/24 completion、6/6 summaries、
  24个1500和24个2000 checkpoints、0 error/OOM/nonfinite；GPU3他人进程和GPU6全程未触碰。
  causal B-transfer proxy在1000/1500/2000为`.38820/.38685/.38678`，reversed=
  `.06042/.06399/.06425`、phase-shuffled=`.19110/.19195/.19199`，晚期target可预测性没有改善。
- 2026-08-08 22:41 CST live比较两节点后，正式1500/2000闭环固定在host memory更空闲的
  `gpu02:0,1,2`与`3,4,5`并发，每卡3 replicas；GPU6的`yfwang`进程和空闲GPU7均不使用，gpu01:3
  的`nlge`进程也不触碰。两个run仍由clean pushed`1362d15` evaluator产生，各1200严格paired rows。
- 2026-08-09 00:01 CST新的profile preflight选择gpu01物理`0,1,2|4,5,7`六张空闲A40，满足3+3
  NUMA；物理3的`nlge` VLLM不触碰。gpu02物理6/7有他人进程，空闲0--5只能形成4+2 NUMA，故不用于
  本轮DDP。gpu01 available host memory约516.5GB，`/data1` quota=
  `552,249,764/1,073,741,824 KiB`。formal仍blocked；先做fresh/resume/contiguous profile及macro3
  online generation→cached rollout smoke，profile权重不进入正式训练。
- 统一target与config已由clean pushed`d96f0fb`封存；六卡profile的两个固定roots、三条exact command
  与fresh/resume/contiguous验收门已写入`task_plan.md`顶部。实际运行必须来自含launch record的
  clean/pushed checkout，formal状态仍未seal。
- clean`ac56ab8`首轮六卡profile在finite/NUMA/显存上通过，但macro3 exact byte parity失败；macro1
  Writer、optimizer/scheduler和六rank RNG跨roots完全一致。deterministic/cuBLAS/math-SDPA probes仍
  复现系统性双轨；当前working root cause是meta DDP遗漏既有`static_graph=True`、导致重启后首次
  reducer生命周期不同。canonical已按source-base/Source-SFT模式修复并写入run contract，但仍须新
  profile逐字节确认；旧profile/probe权重弃用，
  formal继续blocked，必须从新clean/pushed commit和新roots重做。
- 候选修复已由clean pushed`12727b8`封存，聚焦CPU合同46/46；新static-graph roots、三条exact
  commands及byte-parity验收门已登记在`task_plan.md`顶部。它尚未通过GPU reprofile，不能写成根因
  已确认或formal已seal。
- static-graph真实reprofile在macro1、0 optimizer step触发PyTorch 2.11 reducer内部断言；只关闭
  buffer broadcast的dynamic probe仍复现A/B分叉，因此`12727b8` root不得resume，两个候选均否决。
  current canonical已删除DDP hidden reducer，改为每rank local 4-task mean后单一flat ordered
  Ring/Simple NCCL all-reduce mean；数学上仍是train24等权梯度。它尚须新commit/new roots byte parity，
  formal继续blocked。
- stateless flat-reduction实现/config已由clean pushed`c33a16b`封存，聚焦49/49、全仓223/223且
  architecture无hard/parallel family。新roots与exact commands在`task_plan.md`顶部。
- clean pushed launch-record`b00024b`的flat-reduction六卡profile已通过预注册core门：resume路径与
  独立contiguous路径三步科学metrics逐值一致，macro1/macro3 Writer及六份macro3 RNG逐字节一致；
  macro3 `trainer.pt`原始序列化字节不同，但反序列化后的optimizer/scheduler逐项0差异，未把它误写成
  byte-exact。两root峰值allocated/reserved分别约`.736/.877GB`与`.736/.816GB`，0 OOM/nonfinite，
  逐rank physical/local/NUMA、P2P disable、Ring/Simple及无DDP wrapper合同均正确。profile权重永久弃用。
- 2026-08-09 00:48 CST live比较后，online smoke选择`gpu02:0`单张空闲A40；`gpu02:6/7`与
  `gpu01:3`他人进程均不触碰。smoke固定validation 8 tasks×1 state、1 generator、batch4、3 rollout
  replicas、correct/without-replacement；只验在线视频编码、完整LoRA cache、释放Writer/encoder后复用
  source policy及episode evidence，不把8-row success当科研成绩。exact command和验收门取`task_plan.md`。
- 首次online smoke在CPU-only prepare、任何CUDA worker和scientific row之前fail-close：profile training
  source按formal检查含`source_run_summary`，而同一final source checkpoint的smoke inspector将该模式相关
  字段表示为`null`，其余source字段逐项相同。失败root已写`ABORTED.md`且不得resume。canonical只允许
  smoke缺省这一项，同时重新验证training contract记录的summary path/bytes/schema；任何其他source差异
  仍拒绝。真实profile smoke authority已通过，聚焦58/58、正式assets环境全仓224/224；replacement
  fresh root与命令取`task_plan.md`。
- replacement root由clean pushed`31d41d8`自然完成：8/8 unique rows和8个唯一LoRA references，
  一个generator生成8 entries/2个batch4、wall=`12.634s`，3 workers均attempt1/exit0、0
  retry/failure/OOM/nonfinite。peak allocated/reserved=`10,576,054,272/11,182,014,464` bytes；
  Writer/encoder释放后source policy原位复用且没有reload，forbidden reads全0。`1/8` success只作
  execution smoke，不作性能证据。profile与online evidence当时封入旧decoder config并seal；当前状态
  已由顶部address-binding覆盖为blocked。
- 旧K4/AS/RL executable已完成原位退役，通用data/topology/functional/evaluation owner保留并收敛到
  Expert-Manifold唯一canonical Writer。CPU-only全仓`186/186`、compileall与diff check通过；architecture
  guard无hard violation或parallel family。下一步从clean pushed identity fresh启动分段formal。
- identity-fresh formal首段现已预注册为0→50：科学/退役seal=`fcaf733`，唯一fresh root为
  `runs/outputs/pi05_expert_manifold_writer_formal_fresh0_800_r6_step2000_fcaf733_20260809`，scheduler仍按
  800宏步、warmup25运行且绝不加载profile checkpoint。01:47 CST live比较后预选
  `gpu01:0,1,2|4,5,7`六张空闲A40，保持3+3 NUMA并避开物理3他人VLLM；精确frozen worktree、命令、
  quota与macro50验收门取`task_plan.md`顶部。实际启动前仍须再次live复核。
- clean pushed launch-record`446cd42`已在该root自然完成0→50：50/50 finite、完整macro50
  checkpoint、0 OOM/nonfinite，训练body=`10.239s`，peak allocated/reserved=
  `737,273,344/815,792,128` bytes；3+3 NUMA、physical/local rank、deferred NCCL、P2P-disable和
  Ring/Simple single-flat mean全部通过，GPU已释放。macro50 correct400的唯一root、frozen worktree、
  18-worker r3/batch4 exact command与门已写入`task_plan.md`顶部。
- 首次macro50 correct400在CPU-only prepare、0 CUDA worker/0 row时被旧inspector拒绝：training与
  evaluation来自两个合法frozen worktree，但config绝对前缀不同。失败root已ABORTED且不得resume。
  scoped修复改为同一仓库相对路径+bytes/schema，并继续逐项比较完整method/meta/source/checkpoint；
  错路径/bytes回归仍拒绝。聚焦36/36、真实macro50 inspector与全仓189/189通过，等待clean push后
  用replacement fresh root重新live启动。
- 根修已clean push为`d59841e`；replacement frozen branch/worktree、r2 root/log/tmux和完整exact command
  已登记在`task_plan.md`顶部。它只替换失败root和evaluator commit，correct/without-replacement、
  macro50 checkpoint、400 states、6卡×r3、每卡3 generators、batch4与paired合同全部不变。

## 0.0 已完成并负裁决：K4 Phase-Aligned Language-Axial Semantic-Procedure Writer

- Grounded-Video Expert正式四点已完成并负裁决：correct=`76/88/77/82`、breadth>=5=
  `3/4/3/3`、union/intersection=`125/40`；winner macro100五臂=
  `88/87/82/86/86`。全部pairing mismatch为0，correct对任一视频control都没有material margin，
  checkpoint换手仍明显。
- refs1内部证据证明视频并未被旁路：wrong到Reader/BA/action的relative-L2中位约
  `.293/.433/.099`，shuffled/reversed到BA约`.426/.435`；correct LoRA stable-rank/top-energy=
  `1.463/.773`。完整expert隔离也把局部gradient retention保持在约`.45--.60`。因此正式拒绝
  hard video-route/parameter isolation作为漂移根修；失败是共享高层video program与可迁移policy
  write没有被同时保留。
- 当时唯一authority切换为
  `docs/action_forecast_writer_k4_phase_aligned_v6_design.md`：exact language+K4四条action-hidden
  videos逐帧走历史v6 trainable PI05 high-level encoder；每条视频独立可微重采样到phase16，Semantic
  Core读取4×16无序联合证据，causal Procedure逐video计算后按phase等权组合，再由历史v6 exact
  compiler只生成一套LoRA。AS与未来RL共用同一部署图。
- canonical源已原位切换，Grounded-Video `fewshot_m2p.py`活动实现已退休；fresh config为
  `configs/pi05_as_writer_k4_phase_aligned_v6_bci_v1.json`，fresh checkpoint family为
  `k4_phase_aligned_language_axial_rawfull24_v1`。全仓CPU回归`190 passed`，参数量保持历史v6的
  `10,775,296`，step0 identity、K4 set permutation、phase/procedure因果和五个梯度owner合同通过。
- clean`e1d0b62`已在`gpu01:0,1,2|4,5,7`完成fresh0→1与same-root exact-resume1→3：三步
  `86.20/87.52/87.47s`，0 clip/OOM/nonfinite，peak allocated/reserved=
  `34,968,286,720/47,016,050,688` bytes；step3五个owner全可达，累计1,440 queries/288 videos，
  source trainable=0且held reads=0。K4/B20/B2/full24/phase16未降低，profile权重弃用。config现已seal；
  config与profile seal已由clean pushed`ac812a5`封存，profile权重不得进入formal。
- provenance guard的upstream判定已由`2356d33`根修；随后同一clean/pushed commit从
  identity自然完成formal0→200。唯一root为
  `runs/outputs/pi05_as_writer_k4_phase_aligned_v6_formal_fresh0_200_r6_2356d33_20260807`；
  200 finite macros、96,000 queries、19,200 K4 videos、8 checkpoints，wall=`16,228.904s`，
  peak reserved=`39,187,382,272` bytes，source trainable=0且held action reads=0。
- 四点correct=`88/108/80/99`，winner五臂=`108/115/94/101/121`。八task内部分析已完成，
  证明视频传递material但LoRA几乎单方向、task credit仍抵消；本方法负裁决。精确
  数值与root取对应design第9--11节。

## 0.1 已完成并负裁决：Grounded-Video Semantic-Expert Route

- 当前唯一活动authority为
  `docs/action_forecast_writer_grounded_video_expert_route_design.md`。继续使用exact task language+
  K4 action-hidden videos，且让视频同时拥有high-level parameter route与20-group trace dynamic
  value；不是忽略视频或退回language-only LoRA。
- Sparse Semantic-Expert routefix formal已完成200/200 macros：96,000 queries、19,200 K4 videos、
  0 clip/OOM/nonfinite，peak reserved`42,857,398,272` bytes。四点correct=
  `74/74/78/75`、breadth=`6/5/5/5`；single winner macro150=78，不续400。
- winner五臂=`78/85/90/83/92`，correct最低；wrong/reversed相对correct gained/lost=
  `20/8`与`26/12`。五臂union/intersection=`123/55`。parameter isolation虽把expert-local
  Reader/axis最后窗retention提升到约`.205/.196`，却没有形成absolute或video semantic margin。
- 内部production-batch root为
  `runs/outputs/pi05_as_writer_k4_sparse_semantic_expert_trace_m2p_routefix_internal_macro0150_507ae6e_20260807`。
  wrong的`physical/Reader/program/BA/action` relative-L2中位=
  `.309/.194/.209/.279/.050`，reversed=`.335/.197/.205/.278/.044`；correct LoRA
  norm/stable-rank/top-energy=`44.79/1.412/.791`。视频没有被忽略且LoRA有真实杠杆，失败是
  language route让五臂固定同一parameter map，高层视频语义没有参与owner寻址。
- clean`563089a`已在live空闲`gpu01:0,1,2|4,5,7`完成train24×50 grounded address提取；root=
  `runs/outputs/pi05_grounded_video_expert_route_train24x50_563089a_20260807`，teacher action/state/
  reward/terminal与validation/test video reads均为0。初始top2随机K4 primary/exact/overlap=
  `1.0/.984833/.992417`，但task35 secondary导致batch4/singleton只有`23/24` exact，不能作为占
  `.5`完整参数的稳定owner。
- 最终authority基于同一input-only evidence收敛为top1 one-hot：随机K4 route稳定率`1.0`、
  batch4/singleton=`24/24` exact、8-expert usage=`2/6/7/3/1/1/2/2`。artifact为
  `configs/pi05_grounded_video_expert_route_v1.json`；fresh config/schema/checkpoint、唯一canonical
  runtime与聚焦`30 passed`已闭合。
- clean`0be3627`在`gpu01:0,1,2|4,5,7`完成fresh0→1与same-root exact-resume1→3 profile：
  loss=`.150377/.152826/.148513`、step time=`42.63/41.72/41.20s`，0 clip/OOM/nonfinite，peak
  allocated/reserved=`36,709,136,896/45,237,665,792` bytes；step1八Reader可达，step2起16 blocks
  全可达，train24真实route逐expert与authority一致。累计1,440 queries/288 videos、source
  trainable=0，六卡已释放；profile权重弃用。
- clean`a758bba`随后从functional identity自然完成formal0→200；唯一正式root为
  `runs/outputs/pi05_as_writer_k4_grounded_video_expert_trace_m2p_formal_fresh0_200_r6_a758bba_20260807`。
  200 finite macros、96,000 queries、19,200 K4 videos、8 checkpoints、0 clip/OOM/nonfinite，
  wall=`8828.911s`，peak allocated/reserved=`36,708,964,864/42,727,374,848` bytes，source
  trainable=0且validation/test action/video reads=0。六卡已自然释放；旧sparse checkpoint、profile
  权重与language route不得resume或warm-start。
- 当前唯一下一步是评正式root的macro50/100/150/200 strict paired correct400。canonical evaluator
  已切到hashless launch v2：artifact/authority不计算SHA-256或MD5，只以path/schema/size、真实加载、
  explicit run UUID与direct paired-control字段封存；policy-noise RNG保持既有科学配对算法。聚焦
  `55 passed`与validation Writer prepare vertical path通过，尚未看到任何四点rollout结果。

## 0.1 已完成并负裁决：Sparse Semantic-Expert route

- 该阶段authority为
  `docs/action_forecast_writer_sparse_semantic_expert_trace_design.md`。它保留exact task language、
  K4 action-hidden videos、20-group DCT16 direction/physical/evidence trace和single-LoRA部署，
  只重构冻结descriptor之后的trainable parameter ownership。
- Evidence-Factorized fresh0→200、四点、五臂与全部内部分析已经结束并负裁决：correct曲线
  `74/59/65/84`、breadth=`6/6/5/5`，macro200五臂=`84/85/66/83/78`。correct相对wrong的
  gained/lost=`36/18,p=.01983`证明video task identity真实进入closed loop；same/order无margin，
  不能写成视频被忽略或顺序问题已解。
- internal root为
  `runs/outputs/pi05_as_writer_k4_evidence_factorized_layer_trace_m2p_macro0200_internal_refs1_r6_8c8b502_20260807`。
  wrong的`physical/direction/attention/Reader/BA/action`差异中位为
  `.310/1.319/.647/.432/.620/.155`；direction/physical两支均工作，LoRA norm/stable-rank/
  top-energy=`60.31/1.291/.847`，identity action effect`.373`。最早故障不再是视频表示、
  Reader、增益或完全rank collapse。
- 最后50步shared Reader/axis的full24 retention只有`.05527/.04650`、pair cosine=
  `.00954/.00266`、负pair=`.47464/.48007`，满足预注册的sparse-expert开启门。新设计用只由
  train24 language生成并冻结的semantic top2 route，选择两个完整独立Reader+axis M2P experts；
  route只寻址，video trace仍是唯一动态value，zero-video仍严格identity。
- canonical实现现已完成：冻结PI05 task anchor、fixed top2 router、八套完整独立Reader+axis
  experts、memory级等权组合和single decode均在唯一runtime中；fresh config/schema/checkpoint
  family与expert-local gradient ownership已接通。真实trainable=`487,415,808`。
- 首次formal在macro28主动停止：expert-local Gram证明task9实际route为`2/1`，旧artifact却声明
  `2/7`。根因是旧route以24-language BF16 batch生成anchor，而训练逐task forward；这是fixed
  semantic owner的工程合同失败，不是科学负结果。中断root
  `runs/outputs/pi05_as_writer_k4_sparse_semantic_expert_trace_m2p_formal_fresh0_200_r6_ec2375e_20260807`
  与旧profile均不得resume、评测或作为formal证据。
- 根修后task anchor逐exact language独立forward，route generator以singleton anchors拟合并
  复核co-batch路径；最大anchor差`1.49e-8`且top2完全一致。新primary/top2 usage=
  `5/7/6/1/1/2/1/1`与`7/11/6/5/4/4/3/8`，无塌缩。video trace、top2 memory和single-LoRA
  scientific graph未变。
- clean`bf1aae6`上的六卡profile已完成fresh0→1与same-root exact-resume1→3：三步
  `48.051/48.732/48.535s`，0 clip/OOM/nonfinite，peak reserved`45,589,987,328` bytes；step2
  起全部8 experts×Reader/axis 16 blocks可达，累计1,440 queries/288 videos，source trainable=0。
  该profile绑定旧route buffer，现已作废。修复后的clean`bbe5cf2`已用新root
  `runs/outputs/pi05_as_writer_k4_sparse_semantic_expert_trace_m2p_profile_routefix_r6_b20_bbe5cf2_20260807`
  重做fresh0→1与same-root exact-resume1→3：三步`42.299/43.074/42.275s`，0 clip/OOM/nonfinite，
  peak reserved`45,592,084,480` bytes；step1真实route与authority逐expert完全一致，step2起16
  blocks全可达。随后routefix formal、四点、五臂与内部分析已按上文完成并负裁决。所有旧profile/
  中断formal/历史Writer权重禁止加载。

## 0.0 已完成并负裁决：Evidence-Factorized Policy-Layer Trace M2P

- 正式训练root为
  `runs/outputs/pi05_as_writer_k4_evidence_factorized_layer_trace_m2p_formal_fresh0_200_r6_692ab5e_20260806`；
  200 finite macros、96,000 queries、19,200 K4 videos、8 checkpoints、0 clip，wall
  `7272.774s`、peak reserved20.30GB，source trainable=0且held action reads=0。
- 四点与五臂均按strict paired panel完成；Evidence-Factorized不得续训、warm-start或恢复为
  active parallel path。精确branch、attention、LoRA和gradient数值只取对应design与internal root。

## 0.1 已完成并负裁决：Energy-Preserving Policy-Layer Trace M2P

- 当前唯一活动authority为
  `docs/action_forecast_writer_energy_preserving_layer_trace_design.md`。它保留exact task language与
  K=4条action-hidden same-task videos共同生成一套LoRA，保留冻结PI05的20组all-layer
  innovations、DCT16、Reader/M2P、rank16与full24 B20，不忽略视频。
- 上一版macro100五臂已完成：`correct/same/wrong/shuffled/reversed=99/92/57/94/105`。
  correct相对wrong的paired gained/lost=`61/19,p=2.73e-6`，证明video task identity真实传到
  LoRA与closed loop；但order controls不降，顺序语义仍未对齐任务程序。
- 8-task内部probe确认trace→reader→axis→BA→fixed-action链路非零，LoRA norm中位
  `48.28`、stable rank`1.34`、top singular energy`.836`，reader/axis覆盖约12--14个policy groups；
  因此不是视频被旁路、LoRA无杆杆或layer/group坍缩。
- 已定位更早的表示故障：原始DCT中DC energy占比中位`.95664`，高频8项仅
  `.003592`，而旧`temporal_trace_tokens`对每个`group × frequency`独立L2单位化，
  将低能量高频相对放大约140倍。reversal因奇频符号翻转产生强BA/action操纵，却没有
  形成更好closed-loop任务顺序，这比先增加experts更早。
- 新方法只用每视频一个全局scalar将raw pooled trace总能量匹配旧输入总scale，同时保留
  group/frequency间原始相对能量、零项和符号。它对functional AS与未来reward credit使用
  同一接口，不是监督学习trick。
- 新方法必须fresh architecture/config/checkpoint family，不加载macro100或任何历史Writer。
  实现和CPU合同通过后，才live选择`gpu01/gpu02`最多6张空闲A40做fresh0→1、
  exact-resume1→3 profile，再从identity formal0→200与严格四点评测。
- clean`22234c4`已原位实现：新唯一config为
  `configs/pi05_as_writer_k4_energy_preserving_layer_trace_m2p_bci_v1.json`，旧config/family已退休且
  不得resume。聚焦合同、全仓`191 passed`、compileall、real config load和diff check通过；
  formal原先被config显式blocked，只能先完成live profile。
- live`gpu01:0,1,2|4,5,7`六卡profile已严格fresh0→1、exact-resume1→3完成；三步
  loss=`.150377/.152822/.148504`、grad norm=`.000589/.000636/.000639`，0 clip/OOM/
  nonfinite，step2起reader/axis均finite可达。步时约36.7--37.0s，peak allocated/reserved=
  `18,113,258,496/20,375,928,832` bytes，累计1,440 queries/288 videos，source trainable=0，
  六rank/3+3 NUMA/exact-resume闭合。profile权重弃用，config seal=`3b7eb4a`已push。
- launch commit`d833961`的identity-fresh formal0→200已自然完成：200 finite macros、
  96,000 queries、19,200 K4 action-hidden video conditions、8个checkpoints、0 clip，
  source trainable=0且validation/test action reads=0；wall=`7373.955s`、peak reserved=
  `20,478,689,280` bytes。六张A40已自然释放。
- 前三个50步窗口的full24 gradient retention为`.12497/.08564/.08050`，明显高于上一版；
  最后窗口回落到`.05079`且pair cosine`.00555`，说明频谱修复改善早中期共存但尚不能写成
  解决漂移。当前只执行`task_plan.md`顶部macro50/100/150/200 strict correct400合同，
  不用functional loss或gradient挑点。
- 四点strict correct400已自然完成：`67/83/74/85`、breadth=`5/6/7/7`，相邻
  gained/lost=`28/12,18/27,28/17`，union/intersection=`122/40`。macro200以最高correct/
  并列最高breadth成为single winner，但远低于上一版99与v6-fast143，频谱修复目前是行为负结果。
- 当前只按`task_plan.md`顶部合同完成macro200其余四臂与内部分析；不续200、不按loss挑点。
- macro200五臂与8-task内部分析现已全部完成：`85/85/80/74/87`。correct相对wrong
  gained/lost=`25/30,p=.590`，视频task identity的行为margin已经消失；reverse仍高于correct。
- raw amplitude把same/wrong/shuffle/reverse的trace差异中位压到`.135/.310/.251/.335`，
  Reader/BA对应只剩`.030/.297/.060/.079`与`.049/.478/.092/.117`；上一版分别远高。
  Reader effective groups也从约13.97降到10.63。
- LoRA norm/stable-rank/top-energy中位`58.71/1.410/.793`，identity action effect`.581`，
  排除low-gain/rank bottleneck。Energy-Preserving正式负裁决，不续训；下一活动设计必须将
  normalized direction、raw physical support与K4 consistency显式分解后联合读取，暂不打开
  更晚的sparse experts。

## 0.1 已完成并负裁决：K4 Policy-Layer Trace M2P

- 当前唯一活动authority为
  `docs/action_forecast_writer_k4_layer_trace_m2p_design.md`。它保留exact task language与K4
  action-hidden same-task videos联合生成一套LoRA，不忽略视频；但用冻结PI05 action expert的
  action-in、18层pre-q/v normalized hidden和action-out共20组video innovation取代旧final
  hidden固定随机128维压缩。
- 每条视频在每个policy group形成16个DCT-II temporal tokens，K4为每组64 tokens；20组
  layer-matched reader输出`20×68×1024`memory，再用两次column/row交替M2P直接reshape 38个
  public targets的完整rank16 A/B。Q/K可用group/slot/temporal address，V只来自减去no-image
  baseline的视频trace；zero video必须严格回到identity。
- 选择该设计而不是立即复制8个完整experts：旧K4先把3072维最终层信号随机压到128维，再让
  仅在24 tasks上fresh训练的256维decoder猜PI05层级拓扑。直接专家化会分桶同一未对齐表示并
  扩大到约2.5亿fresh参数。只有layer-aligned版本仍实证group-wise credit抵消，才打开稀疏
  共享/experts。
- 新方法从functional identity fresh训练，保持24×B20 full24 equal、信息墙、source freeze、
  K4部署接口和未来RL兼容；不加SFT reconstruction/rank/contrastive auxiliary loss，不加载
  历史Writer。clean`a2c6d94`已完成原位实现：唯一活动config为
  `configs/pi05_as_writer_k4_layer_trace_m2p_bci_v1.json`，旧config/family已退休；全仓BCI
  assets下`190 passed`，compileall与config/schema real load闭合。后续A40 profile、fresh
  formal与预注册四点裁决均已按下文完成。
- 首个`89f5384`三步diagnostic未按fresh0→1再resume1→3分段，本就不是可封存profile；它暴露
  axis FFN pre-LayerNorm把step1极小memory放大到O(1)，loss`.1504→58.93→96.82`、三步clip。
  peak reserved约20.48GB且无OOM，故不是A40容量问题。该root禁止resume。clean`ed4f46e`已
  根修value-path幅度连续性并通过全仓`191 passed`；下一launch必须从新root严格分段profile。
- clean`44e248b`已在live空闲`gpu01:0,1,2|4,5,7`完成fresh0→1与exact-resume1→3：三步
  loss=`.150377/.152815/.148507`、grad norm约`.001`，0 clip/OOM/nonfinite；step2起reader/
  axis M2P都finite可达。步时约34.6--34.7秒，peak allocated/reserved=
  `18,112,406,528/20,375,928,832` bytes，累计1,440 queries/288 videos、source trainable=0，
  六rank/3+3 NUMA与完整resume闭合。profile权重永久弃用；formal现已可从identity新root启动。
- implementation/config/profile seal已更新到clean/pushed`d3f568d`；正式fresh0→200的唯一
  output、scale、设备边界、storage预算、exact command和后续四点裁决取`task_plan.md`顶部
  launch合同。正式run不得从任一profile或历史Writer resume/warm-start。
- launch commit`1b868ed`的独立fresh formal已自然完成0→200：200 metrics、96,000 action
  queries、19,200 K4 videos、8个every25 checkpoints、wall=`7350.114s`、peak allocated/
  reserved=`18,096,154,112/20,478,689,280` bytes，0 clip/OOM/nonfinite，source trainable=0，
  validation/test action reads均为0。formal root为
  `runs/outputs/pi05_as_writer_k4_layer_trace_m2p_formal_fresh0_200_r6_d3f568d_20260806`；GPU已自然释放。
- 训练期task-gradient coexistence相对旧K4只形成早期改善，最后50步full24 retention/
  median cosine/negative-pair中位=`.04573/.00400/.47464`，仅略高于旧K4
  `.04326/.00038/.49275`，不能写成task drift已解。四点完成前没有按step200 functional
  loss`.10194`或该梯度证据挑点。
- 四点strict correct400已全部自然完成，每点400 rows、42 shards、9 workers exit0；
  correct=`69/99/88/94`、breadth=`5/6/6/6`，逐task为
  `[4,0,0,33,29,2,0,1]`、`[5,3,0,34,41,12,0,4]`、
  `[12,1,0,34,26,13,0,2]`、`[15,1,0,34,27,11,0,6]`。相邻gained/lost=
  `42/12,28/39,28/22`，union/intersection=`145/37`，K4 set/state/env/policy RNG严格配对。
- single winner固定macro100=`99`，仍低于旧K4 winner108、v6-fast143和严格门`>150`；
  layer alignment只改善旧K4同点94，未形成持续共同累积。macro100其余四个
  video-control arms与预注册内部分析均已完成；本方法不续训、不warm-start、不按loss另挑点。

## 0.2 已完成：K4 Invariant-Program M2P四点与内部裁决

- owner已解除讨论暂停、恢复自主持续推进，并明确EMBER不能忽略视频；允许few-shot。
  当前唯一活动authority为
  `docs/action_forecast_writer_fewshot_invariant_m2p_design.md`。每个task condition联合读取
  四条action-hidden same-task videos，生成一个video-owned invariant program，再由
  38 targets×16 ranks的608-token policy-wide M2P生成一套完整rank16 LoRA。
- task language只ground视频并作为attention address；首次program read和M2P read均没有
  routing-query residual，video values全零时动态LoRA严格回到functional identity。禁止
  language-only value bypass、逐视频LoRA平均/挑选、generic-language contrast、SFT重建、
  static store、scale/rank trick和历史Writer warm-start。
- canonical model/AS/checkpoint/K4 schedule/live+cached evaluation已原位替换；已退役的
  Condition-Kernel、online functional validation和method-specific analysis runtime删除。
  B20、full24等权、source freeze、rank16/38 targets、信息墙与single-checkpoint裁决不变。
- CPU全仓`189 passed`，compileall和diff check通过。clean`8807ae0`在
  `gpu01:0,1,2|4,5,7`六张空闲A40、3+3 NUMA、显式`NCCL_P2P_DISABLE=1`完成fresh0→1和
  同root exact-resume1→3；三步`34.055/33.955/33.831s`，peak allocated/reserved=
  `17,142,612,480/19,690,160,128` bytes，0 OOM/clip/nonfinite，step2起四个声明block全可达。
- profile总scheduler轴固定为正式200步，仅在step3 early-stop；对应LR=
  `1.154e-5/2.308e-5/3.462e-5`。更早三步压缩scheduler的诊断root因首步误用peak LR而不封存、
  不进入正式证据。sealed root累计1,440 queries、288 videos，source trainable=0，六rank
  exact-resume完整，GPU已自然释放；任何profile权重不得进入正式训练。
- 独立fresh formal已从clean`500294c`自然完成0→200：200 finite macros、96,000 action
  queries、19,200 action-hidden teacher videos、wall=`6879.816s`、peak reserved=
  `19,690,160,128` bytes，0 OOM/clip/nonfinite，source trainable=0，validation/test action
  reads均为0；8个every25 checkpoints完整，GPU自然释放。
- macro50/100/150/200 strict correct400已全部完成，correct=`70/94/99/108`、breadth=
  `6/6/6/7`，逐task为`[1,1,0,32,32,1,0,3]`、`[9,1,0,38,34,6,0,6]`、
  `[3,1,0,35,45,13,0,2]`、`[5,2,0,36,42,17,2,4]`。相邻gained/lost=
  `42/18,30/25,25/16`，union/intersection=`150/42`。每点400 rows/42 shards/9 workers，
  每task 50个K4 sets覆盖50条unique videos且checkpoint间K4 set/state/RNG严格配对。
  GPU rollout完成后聚合器暴露旧K1字段遗漏；唯一`pi05_eval_results` owner已兼容K4
  `teacher_demo_indices`并从既有sealed shards完成CPU聚合，未重跑或改变任何rollout。
- macro200内部root为
  `runs/outputs/pi05_as_writer_k4_invariant_m2p_macro0200_internal_refs1_r6_4951d4e_20260806`：
  K4置换与zero-video identity严格成立；same-task另一set与leave-one-out的Program→BA→action
  变化有界且显著小于wrong，shuffled/reversed又强穿透到action。LoRA norm中位27.59、
  identity→correct action差异中位.2543，证明当前方法没有忽略视频且不缺policy leverage。
- 决定性剩余失败是共享参数中的condition credit cancellation：最后50步full24 gradient
  retention中位`.04326`、pair cosine`.000376`、negative pair`.49275`，Program、M2P、A/B
  heads各层都接近1/24正交极限。K4曲线单调但macro200仍仅108、Goal-3为0、envelope gap42，
  因此不续同一schedule、不warm-start、不用loss挑点；下一架构必须保留K4 video-owned
  program并从condition-specific parameter coexistence重构。长期single-checkpoint correct
  严格`>150`不变。
- 当前Writer-specific新artifact只用路径/schema/size/shape/real-load证据，不生成或复核
  SHA-256/MD5等文件内容hash。GPU仍只可live选择`gpu01/gpu02`最多6张空闲卡，显式
  `NCCL_P2P_DISABLE=1`并保持formal resume的3+3 NUMA topology；当前仍不使用subagent。

## 0.0 历史状态：Condition-Kernel完整负裁决（讨论暂停已解除）

- Factorized Condition-Kernel Program Memory已完成fresh AS0→200、50/100/150/200四点strict
  correct400和全部预注册内部分析；当前没有EMBER GPU进程，不实现/启动reward或下一方法。
  formal root为
  `runs/outputs/pi05_as_writer_condition_kernel_memory_formal_fresh0_200_r6_4038960_20260805`：
  200 finite macros、96,000 queries、4,800 videos、wall=`3951.928s`、peak reserved=
  `19,344,130,048` bytes，0 clip/OOM和0 validation/test action reads。
- 四点correct/breadth=`46/3,46/3,45/3,49/3`；相邻gained/lost=`5/5,4/5,6/2`，四点
  success union/intersection=`55/40`。macro200的49 successes中Goal-6占42、Object-1占5、
  Long-1占2，其余5 tasks全0。AS200未过预注册`correct≥120 && breadth≥6`，所以direct
  reward阶段按authority禁止；不得挑历史点、延长bootstrap或用RL补救。
- 四点state、video ordinal、env/policy seed和实际执行长度的policy-noise common prefix均
  0 mismatch。表面换手少不是task drift得到解决：40个四点共同success中Goal-6占37、
  Object-1占3，Writer只是停在接近source identity的低增益平台，没有形成多task共同累积。
- 显式kernel机制本身通过：200步Gram均rank24、condition number=`5.139--7.750`、cap scale
  始终1；macro50/100/150/200 predicted/observed Program update relative RMS=
  `.002184/.001731/.001718/.001304`，macro51--200 FactorHeads freeze violation为0。raw
  cotangent与observed task delta的cosine/negative/retention保持对应，旧共享condition-map
  把credit重新压成公共方向的问题在该接口确实被消除。
- 六卡内部分析完成96/96 rows与6/6 payload，wall=`273.968s`、peak reserved=
  `19,277,021,184` bytes，0 target-action/validation/test reads；root为
  `runs/outputs/pi05_as_writer_condition_kernel_memory_internal_all4_r6_2972f8f_20260806`。
  same-task demo1 fixed-feature/Program/BA relative-L2中位约`.786/.784/.775→.767`，
  reversed/shuffled BA约`1.39/1.36`，说明视频与顺序差异真实穿过完整LoRA写出。
- 决定性失败是绝对policy leverage：LoRA norm中位仅`.1761→.1779`，比corrected direct
  SFT的`35.7362`小约200倍；虽然stable rank=`3.794→3.724`、top singular energy约`.28`、
  q/v B-column cosine约`.19/.205`，fixed action的same/reversed/shuffled与identity效应都只有
  约`.19--.24%`。显式kernel修复了存储和credit混合，却被macro50冻结的fresh、low-gain
  Program→LoRA decoder锁在无闭环效用的tangent中。
- same-task checkpoint update的task-mean energy fraction在Program为
  `.730/.718/.672`、BA为`.784/.781/.727`，比旧Program-Credit的`.830/.916`改善但仍偏
  task-common。最早失效接口正式定位为zero-B fresh decoder在固定50步内没有bootstrap出
  足够增益、policy-effective的写出基底，而不是address、kernel solve、rank或多卡工程。
  精确汇总为internal root的`experiment_analysis.json`；design第11节给出完整裁决。

- Antithetic Program-Credit formal cycle1与strict correct400已完成：96 rollouts、48个
  valid CRN pairs、54 successes、6个binary-discordant pairs和一次finite full24 update；
  cycle1=`106/400`、breadth5，相对AS125=`97/5`的gained/lost=`18/9`。它是真实`+9`，但未过
  预注册净`+10`门，故cycle2/4/8永久禁止；不能因“只差一条”修改门。
- 六卡只读内部分析已经完成24 tasks×AS125/cycle1共48 rows、6/6 ownership，wall=
  `272.876s`、peak reserved=`19,304,284,160` bytes，target-action与validation/test reads均0；
  root为
  `runs/outputs/pi05_antithetic_program_credit_internal_as125_cycle1_r6_129cab6_20260805`，
  GPU已自然释放。
- 532个冻结tensors逐元素不变；四个上游block relative-L2为
  `.000231/.000151/.000245/.000204`。AS125→cycle1的program/BA/fixed-action变化中位=
  `.006782/.004713/.002279`，说明decoder传递非零，但same-task video centered/sample energy
  在program仅`.002153→.002149`、BA仅`.001154→.001178`，视频因果性没有改善。
- exact train24 program cotangent本来近正交：pair cosine mean/median=`.000107/0`、负pair
  `.2464`、full24 energy retention=`.041874`；经过共享Writer参数更新后，24-task program
  delta却变成pair cosine mean/median=`.5801/.6128`、负pair0、retention=`.55537`。同task
  五video更新的task-mean energy fraction在program/BA为`.82990/.91623`。最早失效接口由此
  定位为不同closed-loop credit经过共享condition-map Jacobian后被压成task-common、
  video-insensitive更新。
- binary/semantic cotangent energy=`.00261635/.00003600`，binary约`72.7×`主导；因此裁决
  不是functional surrogate或LIBERO semantic tie-break的产物。400个held LoRA的BA变化
  中位`.005519`，gained/lost=`.004726/.004742`，stable rank/top1/B-column几乎不变；禁止
  回到rank、scale、head/store扩容或续旧RL。
- 当前唯一活动authority切换为
  `docs/action_forecast_writer_factorized_condition_kernel_memory_design.md`。新方法从generic
  source和functional identity全新训练：冻结foundation task/video descriptor形成固定
  task×video RFF feature，线性读取完整`1024×320×256`Program Value Memory；full24在
  24×24 condition Gram上做显式regularized kernel correction，使program更新可预测而不再
  穿过可漂移condition encoder。fresh FactorHeads只bootstrap到固定macro50，随后永久冻结；
  AS与direct reward都只向同一memory value提供program cotangent。
- 新方法不加载AS125、v6-fast、Policy-Lane或任何历史Writer权重，不恢复Direction Store或
  learned router。canonical AS路径已经原位替换：旧Program-Credit一次性analysis runtime和
  v6 trainable condition path已删除；当前Writer只有冻结descriptor/address、83,886,080参数
  Program Value Memory和2,179,072参数fresh FactorHeads，完整参数86,065,152。M不进入Adam，
  每个full24宏步只接受显式kernel-corrected Program cotangent；FactorHeads在macro50后由
  scheduler/checkpoint/resume合同永久冻结。
- action-hidden地址审计已在train24×50与validation8×50 apply-only完成：50个no-replacement
  schedule的Gram全部rank24，最坏regularized condition=`7.5471`、最大off-diagonal=`.4270`；
  same-task video/cross-task demo0 feature距离中位=`.8718/1.4058`，reversed最小/中位=
  `1.1567/1.4064`。train/validation/test action与reward reads全0，authority SHA为
  `7a49226e...0f86`，root=`runs/outputs/condition_kernel_address_audit_r6_seed2026080501_20260805`。
- `gpu01:0,1,2|4,5,6`六卡fresh0→1再exact-resume1→3 profile已完成并释放GPU。三步wall=
  `20.713/19.842/19.448s`，峰值allocated/reserved=`16,556,672,000/19,344,130,048` bytes，
  longest105、logical B20/B2、0 OOM/clip。step1因zero final layer使Program cotangent严格0；
  step2/3 cotangent RMS=`1.9946e-7/3.5717e-7`，predicted update RMS=
  `1.9684e-7/3.5240e-7`且全局cap均未触发，Gram rank24、condition=`6.632/6.023`。累计
  1,440 queries/72 videos，scheduler/sampler/RNG/六rank checkpoint连续，validation/test
  action reads=0；profile root=`runs/outputs/pi05_as_writer_condition_kernel_memory_profile_r6_b20_seed7_20260805`。
- profile权重永久弃用；独立fresh identity AS0→200、四点strict rollout和内部分析已经按
  上述结果结束。当前只做结果讨论，不续reward、不启动下一架构。
- implementation/config seal=`4038960`已push branch/main，全仓`198 passed`。正式launch
  现场选择全空闲`gpu01:0,1,2|4,5,6`，保持3+3 NUMA；`gpu02:5/6`有他人进程不使用。
  `/data1` quota=`310,538,532/1,073,741,824 KiB`，预计新增小于2GiB。fresh formal root、
  tmux与exact command取`task_plan.md`顶部launch合同。
- Program-Credit只读analysis owner的retirement trigger现已满足；新design落地时删除其
  method-specific runtime/authority，纯gauge-invariant metrics只有出现当前第二用途才迁入
  既有analysis owner。历史可复现性由Git与上述artifact保留。

## 0.1 Policy-Lane正式负裁决

- Policy-Lane正式fresh0→200已在同一未恢复进程完成：200 finite macros、96,000 logical
  queries、4,800 one-video conditions、8个完整checkpoint，wall=`6651.965s`，峰值
  allocated/reserved=`36,174,262,272/42,150,658,048` bytes，0 OOM/clip/nonfinite/
  collective stall，validation/test action reads=0。contract=`a8ce75f2...00f6`，root为
  `runs/outputs/pi05_as_writer_policy_lane_hyperdecoder_formal_fresh0_200_r6_fbc320a_20260805`。
- 固定50/100/150/200 strict correct400已完成：correct=`70/63/37/61`、breadth=
  `6/4/6/6`，相邻gained/lost=`17/24,14/40,40/16`，四点union/intersection=
  `117/14`、single envelope gap=`47`。四root均400 rows、42 shards、一次启动、全部
  worker exit0、每task 50 unique无放回视频且共同noise prefix严格配对。
- macro50 single winner=`70`，低于PWAD80、v6-fast143与严格门151；macro150崩到37后
  macro200又回61，说明能力仍大幅轮换。禁止resume400、warm-start或按functional loss
  选择点位。
- clean`3869d20`的四checkpoint内部分析已完成96/96 cells、6/6 payload，wall=
  `318.446s`、peak reserved=`19,295,895,552` bytes，target-action/validation/test reads=0；
  root为`runs/outputs/pi05_as_writer_policy_lane_hyperdecoder_internal_all4_r6_20260805`。
- lane storage effective count约`15.96--15.97`，demo0 hidden/output effective lanes约
  `11.64--12.50/9.57--10.85`；LoRA norm=`10.77→22.21`、stable rank=
  `1.336→1.542`、top singular energy=`.809→.707`，没有PWAD的伪rank/单lane塌缩。
  400个held LoRA/cache的精确gauge-invariant复核进一步给出q/v跨layer signed cosine约0、
  energy CV约`.75--.83/1.03--1.15`、top4约`47--52%/58--61%`，与direct SFT层专门化
  量级一致。因此漂亮的rank/层组织不是充分条件，也不是当前性能上限的最早根因。
- cross-task demo0 hidden centered/sample energy从`.503→.660`、pair cosine从`.488→.313`，
  而same-task video hidden/BA centered energy始终仅约`.046--.059%/.017--.023%`；macro50
  demo1/reversed/shuffled到BA为`.0176/.0281/.0133`，到fixed action仅
  `.00577/.00977/.00597`。模型越来越区分train task，却几乎不从单条视频获得可累积
  policy credit。
- 下一方法边界不是再改LoRA外观或存储容量，而是让Writer/LoRA生成本身直接获得闭环相对
  credit，避免继续依赖与行为错位的functional surrogate。新design authority封存前不
  实现或launch；Policy-Lane、PWAD和历史store/head扩容均保持负裁决。

- clean pushed`2aeb22a`的Policy-Lane canonical实现已在`gpu01`完成六卡longest105、
  logical-B20/full24三步profile：step max wall=`33.457/31.024/31.007s`，峰值
  allocated/reserved=`36,168,858,624/47,053,799,424` bytes，0 OOM/clip/nonfinite，累计
  1,440 queries/72 one-video conditions。step1只有Policy-Lane梯度符合zero-B阶段；step2
  起Semantic Frontend、Core、Program、Composer、Policy-Lane五个主块全部可达。
- 独立fresh0→1→exact-resume1→3也已通过，optimizer/scheduler/sampler/RNG/task-cycle与
  六rank state连续，合同SHA=`f0f3ec32...55261`。fresh段结束后物理GPU0被他人占用，恢复段
  自主切到`gpu01:1,2,3,4,5,7`，仍严格保持sealed `3+3 NUMA`；未共享或干扰他人进程。
  profile/smoke权重禁止进入正式轨迹。
- config已seal并完成16-frame encoder chunk、logical B20、policy microbatch2、six ranks、
  fresh0→200/every25。训练与rollout精确launch contract均写入`task_plan.md`；长期
  single-checkpoint严格`>150/400`不变。

- Policy-Wide Atom Dictionary已完成clean`69563a0` fresh0→200：200 macros、96,000
  logical queries、4,800 one-video conditions、8个checkpoint，0 OOM/clip/nonfinite且
  validation/test action reads=0。50/100/150/200 strict correct400=`77/71/80/80`，
  breadth=`5/6/5/5`；相邻gained/lost=`19/25,21/12,16/16`，四点union/intersection=
  `115/44`、single envelope gap=`35`。远低于v6-fast143与严格门，禁止resume到400。
- clean`941c5e3`的24 tasks×4 checkpoints内部GPU分析完成96/96 cells和6份rank payload。
  首次summary只因non-action rows没有可选`reversed_0`字段失败；clean`c08d985`在CPU从既有
  payload重建results/completion，`rank_payloads_reused=true`，没有GPU重跑或科学数据改变。
  root为
  `runs/outputs/pi05_as_writer_policy_wide_atom_dictionary_internal_all4_r6_941c5e3_20260805`。
- 64/64 atoms全部active，storage effective count=`63.62→63.93`，condition combined
  effective participation=`50.50→54.19`；字典容量不是最早故障。A/B mixing mean stable
  row rank却始终约`1.000002`，effective LoRA约`1.0000002`、q/v B-column cosine约
  `.999998`。same-task video mixing/BA centered energy只有约`.022--.054%/.022--.047%`，
  action energy share又从`.1197%`降至`.0136%`：模型形成强task/common q-dominant写入，
  没有形成视频条件policy program。
- PWAD独立`M_A/M_B`使真实effective update包含全部`B_j A_k`交叉项，所以所谓完整atom
  并不是实际存储单位；public lane差异完全依赖两张已经塌缩的shared mixing matrix。
  该结果禁止增加K、调scale或加rank/正交loss。
- 当时authority切换为
  `docs/action_forecast_writer_policy_lane_hyperdecoder_design.md`。新架构让16个public lanes
  各自以同一个32维condition hidden共同生成全部38 targets的A/B向量；lane内保持
  policy-wide协调，lane间拥有独立输出参数，取消PWAD的A/B独立mixing和atom cross-product。
  它仍从functional identity fresh训练，Writer输入、信息墙、rank16 public LoRA与AS/RL
  兼容性不变。
- canonical实现已原位替换：删除`policy_dictionary.py`并新增`policy_lane.py`；旧PWAD
  config/checkpoint family不再被活动loader接受。完整Writer参数`49,041,664`，其中
  hyperdecoder=`41,320,448`、composer=`660,224`；38-target每lane A/B输出宽度=
  `37,920/42,528`。聚焦Writer合同`84 passed`，py_compile/diff check通过，architecture
  guard无hard与parallel family；formal config已由上述live profile与resume证据seal。
- 当前profile与resume进程均已自然结束并释放GPU；任何新launch仍须live比较双节点，
  只用最多6张实时空闲卡并显式`NCCL_P2P_DISABLE=1`。

## 0.2 Tangent-Basis正式消融负裁决

- clean`059d40f`的fresh formal root为
  `runs/outputs/pi05_sft_anchored_tangent_basis_writer_formal_macro400_r6_k4_nmc4_e2_c8_059d40f_20260805`。
  它从历史v6-fast macro400 Writer warm-start完成cycle0→1：96 rollout、61 successes、
  11 mixed、5 all-failure、two finite updates，五个系数侧block全可达，8个factor-output
  basis与semantic encoder冻结，0 action-wall read/OOM/watchdog。wall`2046.03s`、peak
  reserved`19,478,347,776` bytes，cycle1 checkpoint与6 rank状态完整。
- cycle1 strict correct400 root为
  `runs/outputs/pi05_sft_anchored_tangent_basis_cycle001_bci_correct400_noreplacement_seed7_059d40f_20260805`：
  400 rows、0 error/retry、correct=`142`、breadth=`7`；v6-fast macro400 baseline为
  `143`、breadth`6`。严格paired gained/lost/retained/both-fail=`20/21/122/237`，
  exact two-sided`p=1.0`，union/intersection=`163/122`。Spatial由`3→6`并打开
  Spatial-1，但Long/Goal/Object分别净`-1/-1/-2`；aggregate稳定仍由能力换手组成。
- 完整配对裁决位于该eval root的`paired_to_macro400_analysis.json`。静态row、task、
  state、video、env/policy seed与共同policy-noise prefix全部严格一致；完整noise list仅
  298/400相同是因episode长度/replan次数不同，不是配对破坏。
- 预注册续训门未过，禁止resume cycle2。`142`是继承`143`的v6 Writer经一次保守RL后
  的结果，不是fresh新架构从identity训练到142；本实验只否定“factor-output basis旋转
  是task drift主要根因”，不得写成LoRA生成质量、视频特异性或新主路线已成立。
- 该负裁决已经导向上节fresh Policy-Wide Atom Dictionary；RL不能继续在v6 warm-start上
  承担修复LoRA generator的职责。

### Tangent-Basis profile与diagnostic前置证据

- clean`2f934bd`六卡profile root为
  `runs/outputs/pi05_sft_anchored_tangent_basis_profile_macro400_r6_2f934bd_20260805`：
  96 rollout、61 successes、11 mixed、5 all-failure，two finite epochs，两轮五个
  trainable block全可达，observer grad0，0 action-wall read/OOM/watchdog。wall
  `2033.38s`，peak reserved`19,478,347,776` bytes；两轮均在6/6 CUDA-ready后再进
  NCCL，cycle1完整checkpoint已原子封存，GPU全部释放。逐张量比较证明8 basis +
  440 semantic-encoder tensors完全不变，恰好76个系数侧tensors改变。

- clean`303e714`六卡只读diagnostic root为
  `runs/outputs/pi05_sft_anchored_tangent_basis_diagnostic_macro400_r6_303e714_20260805`：
  96/96、61 successes、11 mixed、8 all-success、5 all-failure；mixed agreement=`1.0`、
  AUC=`.91429`，all-failure range `4/5`过`.05`且中位`.27273`。correct胜wrong/shuffled/
  reversed=`1.0/.90164/1.0`，margin中位`.55919/.37889/1.53747`，pixel Spearman `.48421`；
  六个联合门全过。wall`388.80s`、peak reserved`19,289,604,096` bytes，0 optimizer/
  backward/checkpoint/action-wall reads，GPU已释放。
- clean`67b245a`的AS125→cycle2参数hybrid正式root为
  `runs/outputs/pi05_progress_credit_parameter_hybrid_as125_cycle2_r6_67b245a_20260805`：
  24 tasks×7 conditions×4 arms、8-task fixed action全部完成，wall`333.52s`、peak
  reserved`19,365,101,568` bytes、0 target-action/validation/test reads。effective BA
  中upstream residual中位`.611`优于factor-output`.727`，fixed action却由
  factor-output`.489`优于upstream`.668`；policy leverage随suite反转，证实共享
  decoder重定向而非单纯参数位移或LoRA rank问题。
- 当前authority为
  `docs/action_forecast_writer_sft_anchored_tangent_basis_design.md`。canonical RL config
  已原位升级：cold start改为历史v6-fast macro400（strict143），冻结semantic encoder与
  8个`factor_heads.*.network.2.weight` policy basis，只训练上游和factor-input
  coefficients；reward/K4/Nmc4/LR/two epochs不变。参数hybrid一次性入口已按retirement
  trigger删除。
- macro400保留旧A100绝对路径；warm-start使用既有`source_reference_matches`按已验证
  source identity跨host重绑定，不改历史run contract。首次diagnostic在GPU分配后、模型
  构造前被退役旧v6 schema拒绝，0 rollout/checkpoint且GPU已释放；根修为继续引用当前受支持
  v6 config，并由RL config显式把non-parameter encoder chunk恢复为macro400原始32。解析后
  authorities与Writer字段逐项匹配旧run contract。第二次到达checkpoint后由旧manifest
  schema fail-fast；0 rollout/checkpoint且GPU释放。根修只在`initialize_writer_phase`接受
  该manifest/launch schema作load-only warm start，逐文件/contract验证不变，exact-resume
  与AS evaluator仍拒绝。真实macro400单进程检查全部通过，聚焦回归22 passed；
  后续diagnostic与profile均已按上述结果完成。
- 当前GPU已全部释放，无EMBER进程。最近live检查时`gpu01`八卡空闲、`gpu02:6`属于其他
  用户；任何新launch仍须重新比较两节点且最多6张。

### 已封存的AS125 cycle2负裁决

- formal root
  `runs/outputs/pi05_task_grounded_progress_credit_writer_formal_as125_r6_k4_nmc4_e2_c8_retry1_30977b5_20260805`
  已在原sealed`gpu01:1,2,3|4,5,7` 3+3 topology exact-resume到cycle2。第二cycle为
  49/96 successes、16 mixed、5 all-failure semantic、3 all-success与21 active-credit
  tasks；两epoch finite、observer grad0、完整checkpoint/双ledger、0 watchdog/OOM，
  peak reserved`19,457,376,256` bytes。formal summary保持`next_cycle=2`，但科学裁决
  已禁止继续4/8。
- cycle2 strict correct400 root为
  `runs/outputs/pi05_task_grounded_progress_credit_cycle002_bci_correct400_noreplacement_seed7_56a167d_20260805`：
  400 rows、success102、breadth4、逐task`11/0/0/43/26/22/0/0`。相对cycle1=104
  gained/lost/retained/both-fail=`15/17/87/281`，全部state/video/env seed与共同noise
  prefix严格配对。Object-1`31→26`、Object-3`19→22`，其余净0；没有共同积累或新task
  coverage。
- AS125/cycle1/cycle2 success=`97/104/102`，union/intersection=`128/79`、single
  envelope gap24。cycle1→2 effective BA relative-L2/cosine/norm ratio中位=
  `.01493/.999894/1.00214`；gained/lost变化幅度`.014725/.014724`与norm增长几乎相同，
  stable rank仍约`1.000016`。同recipe续训正式负裁决，不得resume cycle4/8。
- Writer权重分块显示raw factor gradient大不等于Adam后数百倍位移：cycle1→2
  delta-L2/sqrtN为semantic`1.13e-5`、visual`6.29e-6`、procedure`9.33e-6`、compiler
  `1.13e-5`、factor input/output`1.25e-5/1.24e-5`。所以当前下一步不是直接冻结factor，
  而是在固定train-task/video/action panel上把AS125→cycle2分解成factor-output basis与
  upstream composition两套hybrid，先测它们到effective BA/action的实际贡献。之后只选择
  basis freeze、全task policy-distance anchor或显式basis/coefficients重构之一。
- 完整cycle2裁决在
  `runs/outputs/pi05_task_grounded_progress_credit_cycle002_bci_correct400_noreplacement_seed7_56a167d_20260805/paired_to_cycle1_and_as125_analysis.json`。
  当前`gpu01`评测卡已释放且无EMBER进程；新GPU动作仍须live比较两节点、最多6张空闲卡。

### 已封存的AS125→cycle1证据链

- 同一fresh v6 AS root已从step100 exact-resume到125：累计60,000 logical queries、
  3,000 one-video conditions和125个finite full24 macros；本段wall`806.928s`，0
  OOM/clip、0 validation/test action reads，完整checkpoint为
  `checkpoints/step_00000125`。第一次resume命令漏传sealed`--num-workers 0`，在step101
  前被contract拒绝且未写metrics/checkpoint；补齐完整CLI后原root正常完成。
- step125 reward profile root为
  `runs/outputs/pi05_rl_writer_relative_flow_profile_from_v6_macro125_r6_bci_6fe4e52_20260805`：
  96条K4 ledger、24,600 actions、50 successes、19/24 coverage、14 mixed、5
  all-success、5 all-failure；suite success spatial/object/goal/libero10=`11/20/12/7`，
  coverage=`5/6/5/3`。全失败task=`4/20/36/38/39`。相对step100严格配对静态身份和共同
  policy-noise prefix全部一致，gained/lost/retained/both-fail=`10/12/40/34`；task5/29
  新获coverage且没有旧coverage完全掉出，但24-task门仍未过。
- 五点K4 success=`25/38/47/52/50`、coverage=`12/14/18/17/19`。task36/38/39在全部
  五点均0/4；task4在25/50有成功后连续归零，task20在50/75有成功后连续归零。因此
  继续同一AS轴没有足够依据，binary-only正式RL不启动，所有profile cycle1权重弃用。
- step125两epoch ratio范围=`[.98710,1.01237]`与`[.76458,1.10147]`，mean近1、clip均0、
  grad norm=`.03615/.05310`，peak reserved=`40,338,718,720` bytes；两轮FileStore
  all-rank-ready、完整cycle1 checkpoint和0 watchdog/OOM/nonfinite。实现健康不改变
  科学coverage负裁决。
- clean`6fe4e52`上的step100/125两点内部审计root为
  `runs/outputs/pi05_as_writer_v6_relative_flow_coldstart_internal_audit_step100_125_r6_6fe4e52_20260805`：
  48/48 rows、6 rank payloads、wall`194.743s`、peak reserved`19,306,381,312` bytes，
  0 target-action/validation/test reads。norm中位`99.18→109.11`，但same-task video
  energy`.1300%→.1154%`、demo1 BA差异`.0475→.0448`、fixed-action demo差异
  `.0101→.0086`均未增强；BA/action churn仍为`.536/.138`。
- 全24任务中success变化与video-energy变化Spearman=`-.521,p=.0090`，与BA churn
  `=.416,p=.0430`。新增coverage task5/29的step125 video-energy中位`.1154%`，持续
  全失败组反而`.2101%`；后者demo1 BA差异和churn也更大。现有Writer不是不会生成差异，
  而是binary reward无法为失败轨迹判别这些差异是否朝teacher所示目标状态推进。
- 下一authority已封存为
  `docs/action_forecast_writer_task_grounded_progress_credit_design.md`。它冻结AS125
  `semantic_encoder`，以task-token grounded patch与固定Action-Expert interaction组成
  teacher首尾内容方向，再用rollout自身首尾变化计算bounded start-relative potential。
  mixed task仍只用binary LOO、all-success零梯度，仅all-failure允许semantic LOO。
- clean`c483497`六卡只读root为
  `runs/outputs/pi05_task_grounded_progress_credit_diagnostic_as125_r6_c483497_20260805`：
  96 rollouts/24,600 actions、50 successes、14 mixed、5 all-success、5 all-failure，
  wall`401.874s`、peak reserved`19,289,604,096` bytes。96/96旧step125 profile身份、
  policy-noise与outcome逐项一致；0 optimizer update、0 Writer backward、0 checkpoint。
- 全部预注册门通过：mixed agreement=`13/14`、同task pair AUC=`.8913`；task4/20/36/
  38/39 utility range=`.1228/.5712/.3338/.2554/.2371`；successful rollout的correct对
  wrong/shuffled/reversed胜率=`1/.88/1`、margin中位=`.4889/.3557/1.6208`；all-failure
  utility与pixel-change Spearman=`.5564`。这只证明observer可作credit，不证明LoRA性能
  已提高。
- clean`84d856c`的fresh profile root为
  `runs/outputs/pi05_task_grounded_progress_credit_writer_profile_as125_r6_84d856c_20260805`：
  96 rollouts/24,593 actions、50 successes、14 mixed、5 all-success、5 all-failure；
  two-epoch wall`2129.187s`、peak reserved`19,455,279,104` bytes，完整cycle1 checkpoint、
  0 OOM/clip/watchdog。5/5 all-failure task都有nonzero generated-LoRA gradient，五个
  downstream blocks两epoch均可达，observer grad=0；ratio=`[.99077,1.02504]`与
  `[.74545,1.09294]`，grad=`.03715/.05521`。
- 相对只读诊断95/96 rollout完整一致；唯一task28/cursor1在相同初态/seed/LoRA/outcome
  下提前一个chunk成功，少7 actions。五个all-failure task的20条utility排序不变，最大/
  平均绝对差`.01622/.00318`，不改变credit owner。该差异封存为success termination
  边界微扰，不以重复profile追求字节一致。
- formal首次从AS125 fresh启动的root为
  `runs/outputs/pi05_task_grounded_progress_credit_writer_formal_as125_r6_k4_nmc4_e2_c8_bc4ff60_20260805`。
  96 rollout与24 task progress-credit均完整且outcome分组仍为14 mixed、5 all-success、
  5 all-failure，但第一轮gradient sum发生collective序列分裂：rank0/1/2/5进入seq18，
  rank3/4停在seq17，600秒watchdog终止。该root没有optimizer update、metrics或checkpoint，
  禁止resume/评测；GPU已自然释放。
- 旧`FileStore` ready只证明Python走到barrier，不能证明rank-local CUDA backlog结束，且
  临时store生命周期在高度错峰时不能可靠提供一次性all-rank barrier。canonical修复现为
  每rank先CUDA synchronize，再以torchrun唯一session/cycle/epoch写原子rank marker，
  6/6 marker可见后才进入NCCL；marker不在run内删除。相同root连续两个新session的真实
  六卡探针均得到6/6 marker和all-reduce sum21，旧session未污染重启。
- clean/pushed`30977b5`的全新retry1正式root为
  `runs/outputs/pi05_task_grounded_progress_credit_writer_formal_as125_r6_k4_nmc4_e2_c8_retry1_30977b5_20260805`，
  contract=`129c49d9...26c8`。两轮分别形成6/6 CUDA-complete marker后才进入NCCL，完成
  2次finite update、完整cycle1 checkpoint与0 watchdog/OOM；wall`2125.726s`、peak
  reserved`19,455,279,104` bytes。50/96 successes、14 mixed、5 all-success、5
  all-failure与机制诊断一致。
- 两epoch ratio=`[.99077,1.02504]`与`[.77339,1.09274]`、grad norm=
  `.03635/.05018`、clip0；5/5 all-failure task有nonzero generated-LoRA gradient，五个
  downstream block均可达，observer grad=0。checkpoint validator确认next_cycle1、6个
  rank各16 rollout/4 progress-credit双ledger及96条全覆盖。
- 相对失败root，95/96 rollout字节一致；唯一task28/cursor1保持同初态/seed/LoRA/成功
  outcome，但成功终止从76步变83步，多7 actions。24个progress-credit文件完全一致，
  这是已封存的成功终止边界微扰，不改变credit或formal裁决。
- strict correct400已用`gpu01:1,2,3`评AS125、`gpu01:4,5,7`评cycle1并行完成；两个root
  分别为`runs/outputs/pi05_as_writer_v6_coldstart_as125_bci_correct400_noreplacement_seed7_df413de_20260805`
  与`runs/outputs/pi05_task_grounded_progress_credit_cycle001_bci_correct400_noreplacement_seed7_df413de_20260805`。
  两边均为400 unique rows、同state/video/env seed与共同policy-noise prefix，correct=
  `97/104`，gained/lost/retained/both-fail=`22/15/82/281`，breadth=`5/4`。
- 逐task（Long-1/2、Goal-3/6、Object-1/3、Spatial-1/3）为AS125
  `10/0/0/43/24/19/1/0`、cycle1`11/0/0/43/31/19/0/0`。净增7集中于Object-1；
  Spatial-1丢失唯一成功，discordant exact two-sided `p=.3240`，不能写成task drift已解。
- 400对LoRA内部分析显示effective BA relative-L2中位`.01677`、方向余弦`.999860`、
  norm ratio`.99965`。1,520个target谱样本中AS/cycle1 top-1 energy中位均`.999983`、
  stable rank均约`1.000017`，B-column cosine`.998846/.998840`。更新有闭环作用但没有
  改变near-rank1结构，主要是条件相关的小幅方向/幅度微调。
- 当前裁决：cycle1仅有2次full24 optimizer update，held aggregate净升且19/24 train
  tasks实际有credit，故只exact-resume同一formal root `1→2`，科学合同、两epoch、任务/
  视频schedule与3+3拓扑不变；cycle2再跑同一correct400。若改善仍集中单task或breadth
  不恢复，则不续4/8，回到condition-to-policy结构。当前GPU已释放且无EMBER进程。

### 已封存的step0--100证据链

- owner已恢复持续推进并要求科学/工程问题自行深入分析。当时唯一活动方法为
  `docs/action_forecast_writer_relative_flow_credit_design.md`：恢复v6条件生成路径做
  fresh独立AS cold start，随后关闭teacher action入口，以full24 official random-reset
  reward、同task K4 leave-one-out advantage和per-CFM-sample PPO/SPO ratio训练Writer。
  one-shot、信息墙、single checkpoint、不使用subagent和最多6张live空闲A40不变。
- canonical源码已原位完成替换：旧success-only self-imitation、flat task-local RL、
  Target-Owned与Direction Store活动实现均退役；success/failure executed prefixes、
  deterministic Nmc4 flow credit、实际world-size full24 assignment、deferred NCCL、完整
  cycle checkpoint/resume和raw-video evaluator接线已落在唯一Writer/RL路径。聚焦
  RL/reward/eval 43项通过；全仓按内存边界拆为135+75项，合计210项通过。
- v6 AS profile root为
  `runs/outputs/pi05_as_writer_v6_relative_flow_coldstart_profile_r6_bci_20260804`：6 ranks×4
  tasks、logical B20、policy microbatch2、16-frame chunk，三步wall
  `33.464/30.886/30.977s`，峰值allocated/reserved
  `34,948,858,880/44,816,138,240` bytes，最长105帧、0 OOM/clip、0 validation/test
  action reads。step1按zero-init只有factor梯度，step3五个声明主block均finite/nonzero。
- 独立resume root为
  `runs/outputs/pi05_as_writer_v6_relative_flow_coldstart_resume_smoke_r6_bci_20260804`：fresh
  0→1再exact-resume1→3，合同`1d2290ea...d457a87`不变，metrics严格1/2/3、累计
  1,440 queries与72 one-shot videos，source policy trainable=0。profile结束后
  `gpu02:1,2,3,4,5,7`已自然回到10--11MiB；0和6始终属于其他用户且未触碰。
- 正式AS root
  `runs/outputs/pi05_as_writer_v6_relative_flow_coldstart_formal_r6_b20_seed7_b75cb19_20260804`
  已完成fresh0→25并三次exact-resume到100：累计48,000 queries、2,400 videos、100个
  finite macros、0 OOM/clip，四个segment wall=
  `810.991/816.191/805.356/805.085s`，完整checkpoint=`checkpoints/step_00000100`。
  这是独立v6 cold start，不含profile或历史权重；24 tasks各2,000 queries、100次video
  visits并覆盖全部50条video。
- 有效reward profile root为
  `runs/outputs/pi05_rl_writer_relative_flow_profile_from_v6_macro025_r6_bci_retry2_6ff7599_20260804`：
  96条K4 ledgers、28,085 actions、25 successes、12/24 task success coverage、9 mixed、
  3 all-success、12 all-failure；coverage失败。两epoch ratio/clip/grad健康，峰值reserved
  `45,183,139,840` bytes，说明机制与A40负载通过，但其cycle1 checkpoint禁止作为正式
  reward cold start继续。
- step50有效reward profile为
  `runs/outputs/pi05_rl_writer_relative_flow_profile_from_v6_macro050_r6_bci_retry1_e5bca71_20260804`：
  96条K4 ledgers、25,878 actions、38 successes、14/24 task coverage、10 mixed、4
  all-success、10 all-failure；suite success spatial/object/goal/libero10=`9/12/11/6`，
  coverage=`4/4/4/2`，仍未过门。相对step25的96个task/cursor，env seed、初态hash、
  policy seed、teacher demo和共同noise prefix全部严格一致；gained/lost/retained=
  `19/6/19`，coverage`12→14`，说明总体积累同时仍有task换手。
- step50首次profile的96条pre-update ledger与上述有效root逐文件字节级一致，但其
  outcome-skewed local credit耗时使0-mixed rank提前进入NCCL all-reduce，480秒后被
  watchdog终止，没有metrics/checkpoint。`e5bca71`加入每epoch本地反向后的独立
  FileStore all-rank-ready；在原六卡、96 rollout、两epoch规模重放后完成finite更新和
  cycle1 checkpoint，0 watchdog/traceback。两epoch ratio范围=`[.9905,1.0094]`和
  `[.8555,1.0559]`，clip=`0/0`，grad norm=`.02872/.02697`，峰值reserved
  `40,342,913,024` bytes。
- step75 reward profile root为
  `runs/outputs/pi05_rl_writer_relative_flow_profile_from_v6_macro075_r6_bci_04dbc4d_20260804`：
  96条K4 ledgers、25,223 actions、47 successes、18/24 task coverage、13 mixed、5
  all-success、6 all-failure；suite success spatial/object/goal/libero10=`12/17/12/6`，
  coverage=`4/6/5/3`。相对step50严格配对gained/lost/retained/both-fail=
  `21/12/26/37`，新获得coverage为task`9/16/19/25/37`、失去task4。success
  `38→47`与coverage`14→18`证明净积累，但换手仍在，不能宣称漂移已解决。
- step75两epoch ratio范围=`[.97777,1.02659]`与`[.91171,1.08119]`，positive clip=
  `0/.000247`，grad norm=`.03184/.02709`，峰值reserved=`40,340,815,872` bytes；13个
  mixed tasks按rank为`3/1/4/1/3/1`，FileStore barrier在第二个真实不均衡分布上完成
  两轮finite update、完整cycle1 checkpoint和0 watchdog。
- AS50→75首次resume选择的物理卡形成`4+2` NUMA rank分布，与root封存的`3+3`不符，
  因而在训练前被resume contract正确fail-close，没有metrics/checkpoint。改用同节点
  `1,2,3,4,5,7`保持原`3+3` topology后完成；这不是科学负结果或代码故障。
- step100 reward profile root为
  `runs/outputs/pi05_rl_writer_relative_flow_profile_from_v6_macro100_r6_bci_4fff21c_20260805`：
  96条K4 ledgers、24,275 actions、52 successes、17/24 task coverage、11 mixed、6
  all-success、7 all-failure；suite success spatial/object/goal/libero10=`14/19/13/6`，
  coverage=`4/6/4/3`。相对step75的task/cursor、env seed、初态、policy seed、teacher demo
  和共同noise prefix均严格一致，gained/lost/retained/both-fail=`14/9/38/35`；task20
  失去coverage，没有新task进入，故success`47→52`不能解释为breadth单调积累。
- step100两epoch ratio范围=`[.98452,1.00771]`与`[.88801,1.06045]`，positive clip均0，
  grad norm=`.02535/.02563`，峰值reserved=`45,183,139,840` bytes；两轮finite update、
  完整cycle1 checkpoint、0 watchdog/OOM/nonfinite。该checkpoint仍只作profile，正式RL
  不启动。
- owner已强调先从LoRA生成质量与模型内部条件传递理解问题，且RL不是默认答案。下一步
  对同一AS step25/50/75/100做24-train-task真`BA`谱/能量/跨video方向审计，并在按split
  结构固定的8-task面板上测固定观测、固定policy noise的action传递；不读target action，
  不用functional loss选点。审计后才裁决续AS、正式Relative-Flow RL、v5.2同credit
  对照或条件生成重构；不得借历史best或reward checkpoint warm-start。
- 该审计已由clean/pushed`2b775f0`在`gpu01:0--5`完成，root为
  `runs/outputs/pi05_as_writer_v6_relative_flow_coldstart_internal_audit_step025_100_r6_2b775f0_20260805`：
  4 checkpoints×24 tasks=96 rows，wall`291.333s`，peak reserved
  `19,308,478,464` bytes，0 target-action/validation/test reads。step25/50/75/100的
  norm中位=`53.40/80.37/93.17/99.18`，stable rank中位=
  `1.000028/1.000055/1.000153/1.000176`，top-singular share=
  `.999972/.999945/.999848/.999825`。这是对历史v6 near-rank1的当前复核，不是重新把
  rank宣布为根因；direct SFT约1.52与Target-Spectral correct34仍禁止强制谱/正交。
- 五条same-task video centered/sample energy中位=`.0813%/.1309%/.1333%/.1300%`，
  demo0→demo1 BA relative-L2中位=`.0380/.0506/.0499/.0475`，从step50起没有继续增强；
  fixed-action same-video仅`.0081/.0071/.0094/.0101`。相反reversed/shuffled fixed-action
  从`.0299/.0216`升到step100`.0498/.0474`，说明v6时序链有效但video-instance方向弱。
- step100有success与全失败tasks的norm中位=`99.13/102.37`、video variance=
  `.1290%/.1311%`、demo1 BA差异=`.0405/.0634`；失败组并不缺LoRA幅度或视频变化。
  相邻checkpoint BA/action churn中位从step50`1.116/.187`降到step100`.608/.142`，仍
  足以造成能力换手。最早可行动接口仍是reward对condition-to-policy方向的credit，
  不是rank、scale或更多store。
- 因正式RL的24-task coverage门尚未过，下一步按sealed cold-start合同只从AS step100
  exact-resume到125并重做K4；该段只为获得全task reward support，不代表AS surrogate
  已解决方向质量。不得继承任何profile cycle1 update。
- 本轮根修RL环境池未绑定sealed asset cache，以及非连续选卡时把local rank误当物理EGL
  card的问题；有效run contract已记录physical GPU=`1,2,3,4,5,7`。相关长期规则已写入
  `AGENTS.md`，诊断root不进入科研结论。

## 0.0a 历史裁决：Policy-Target-Owned Factor已负裁决

- owner授权下的本轮架构、profile、fresh正式训练、四点rollout和全部预注册内部分析
  已完成；按owner此前要求，现在暂停，不启动下一架构、训练或评测。长期
  single-checkpoint `correct>150/400`目标未完成；strict one-shot、不使用subagent、
  效率优先和每次live选择`gpu01/gpu02`最多6张空闲卡的边界继续有效。
- clean pushed`34be4a0`在frozen worktree从fresh identity完成macro0→200：200次
  full24 update、96,000 logical queries、4,800 one-video conditions、8 checkpoints，
  wall`6678.957s`；0 clip/OOM、峰值allocated/reserved`33.696/38.729GiB`、0
  validation/test action reads。正式root为
  `runs/outputs/pi05_as_writer_target_owned_factor_bci_rawfull24_decay400_formal_r6_b20_micro2_seed7_formalvideo20260722_34be4a0_20260804T051244Z`，runtime contract
  `6af3b4fe...904b`；profile或历史Writer权重均未进入。
- 50/100/150/200 strict paired correct400=`99/76/86/68`，breadth=`6/6/7/5`；逐task
  为`9/0/1/44/38/6/1/0`、`5/0/4/33/28/2/0/4`、
  `7/0/1/26/39/10/1/2`、`7/0/0/31/27/2/1/0`。相邻gained/lost=
  `15/38,35/25,18/36`，union/intersection=`136/37`、envelope gap37。winner
  macro50=99，低于Direction Store129和v6-fast143；Long-2四点全0，故不续400。
  四个sealed roots为
  `runs/outputs/pi05_as_writer_target_owned_factor_bci_correct400_noreplacement_seed7_macro{050,100,150,200}_34be4a0_20260804`；每个root均有400 unique rows、42 shards、
  9 workers exit0、50个teacher demos/task且无retry/adoption。
- macro50 refs1内部root为
  `runs/outputs/pi05_as_writer_target_owned_factor_bci_macro050_internal_refs1_seed7_34be4a0_20260804`：
  六rank、8 tasks、correct/same/wrong/shuffled/reversed完整，wall`100.864s`，0 rollout、
  0信息墙违规、strict replay/rank gauge/checkpoint unchanged全通过。分析完成后六张GPU
  自然释放。
- 76 heads确实解除旧policy-target硬共享：q/v cross-layer effective-BA cosine从
  Direction Store`.9319/.9666`降到`-.00011/-.00030`。但correct LoRA norm均值仅
  `19.0257`，layer-energy CV=`1.9607`，q/v top-4占`.7329/.8529`，比直接SFT的
  `.464--.469/.544--.589`更过度集中；action heads能量占比仅`.000085`。
- same-task Program/factor/BA/action relative-L2为
  `.90933/.05842/.09119/.03161`：独立heads把BA差异放大，却没有写入等比例的
  policy-action有效方向。A/E、Core mean、Core-only、Program-only和memory reversal
  都能到BA/action，动态路径未断。高分Goal-6/Object-1对视频很不敏感，最敏感的
  Object-3只有6/50，也说明condition dependence没有与competence绑定。
- factor承担单task梯度能量中位数`69.25%`，24-task median cosine`.0040`、负pair
  `.4457`、full24能量保留`.0484`。CountSketch里task identity只解释factor方向方差
  `.0168`（随机基线约`.0048`），同task+demo隔50 macros的重现余弦仅`.0046`。
  正式拒绝policy-target sharing作为主要task-drift根因；最早剩余接口更新为
  condition-to-policy credit缺少稳定、闭环有效、跨随机query可累积的task/video方向。
  下一轮不得继续加heads、layer gate/scale、强制SFT profile或监督专用trick。

## 0. BCI运行交接（优先于下文旧A100操作描述）

- EMBER已迁至`/data1/user/ymdai/projects/EMBER`并使用项目`.venv`。source、data、
  tokenizer、checkpoint和output继续由CLI显式传入；每次进程还需显式设置
  `EMBER_STORAGE_ROOT=/data1/user/ymdai`、owner容量上限和项目内
  `EMBER_LIBERO_ASSETS_ROOT`，不能假定`.env.local`自动提供这些值。不要再把
  `/data/ymdai`绝对路径写入新命令或新artifact。
- 当前VR A40配置是
  `configs/pi05_as_writer_semantic_factor_basis_variance_reduced_long105_profile_v1.json`；
  它使用6 ranks×4 tasks、16-frame encoder microbatch、逻辑B20和policy microbatch2，
  不固定物理GPU编号。一个LoRA仍读取完整B20随机样本，只把frozen-policy forward
  切成10个B2并按样本数加权；full24 raw mean、一次AdamW与scheduler合同不变。
- BCI四卡迁移验收已完成：NCCL/BF16 collective通过，真实Writer fresh 0→1通过，
  exact resume 1→2通过，最长真实视频105帧，峰值CUDA reserved
  `44,853,886,976` bytes；随后8/8 validation smoke rollouts完成并聚合。
- 当前torch/NCCL在gpu02直接P2P传输会挂死；EMBER环境已自动设置
  `NCCL_P2P_DISABLE=1`使用稳定的共享内存传输。无需在每条命令里重复设置。
- 评测preflight已移除对整个个人目录的递归`du`和个人容量硬门，只保留快速文件系统
  余量及所选GPU现场检查。不要恢复全目录扫描或A100的固定GPU4--7约束。
- 验收root为
  `/data1/user/ymdai/projects/EMBER/runs/acceptance/ember_bci_gpu_acceptance_20260803T1232`；
  迁移证据在
  `/data1/user/ymdai/projects/EMBER/evidence/migration/20260803/gpu-acceptance/`。
  这些profile/smoke checkpoint只证明运行链路，后续VR正式实验仍须fresh identity，
  不得从验收权重warm-start。
- 验收结束后无EMBER训练、评测worker或tmux进程，四张验收GPU均已释放。
- owner现授权每次实时比较`gpu01`与`gpu02`，只用空闲卡且总数最多6张。2026-08-03
  07:00 UTC附近快照中`gpu01`八卡均忙，`gpu02`的0/1/2/3/4/7空闲，因此工程profile
  只使用这六卡；5/6有他人任务且从未触碰。该分配是易变快照，每次launch必须重查。
- 六卡NCCL/BF16 smoke通过；未冻结工程profile在
  `runs/acceptance/ember_bci_vr_effective_b20_micro2_r6_profile_20260803T1600/train`。
  fresh0→1再exact-resume1→3完成；每步24 tasks、480 logical queries、240 physical
  forwards，三步wall为`33.973/31.686/31.240s`，loss为
  `.157415/.152420/.148585`。峰值allocated/reserved为
  `34,970,270,208/47,108,325,376` bytes，五个主block从macro2起finite/nonzero，
  validation/test action reads为0，step1/2/3 checkpoint齐全。由于运行时源码未提交，
  这里只算工程证据；提交后必须fresh重放0→1与exact-resume1→3再seal。
- 实现经23项focused、226项全仓CPU回归、compileall和architecture guard无hard
  violation后提交/push为`391f183`。同一logical-B20配置随后从clean pushed commit在
  `runs/acceptance/ember_bci_vr_effective_b20_micro2_r6_profile_391f183_20260803T0735Z/train`
  完成fresh0→1与exact-resume1→3：三步`33.514/32.050/31.326s`，loss
  `.157415/.152418/.148564`，峰值allocated/reserved
  `34,970,270,720/47,108,325,376` bytes；最长105帧、1440 queries、72 videos、
  五主block从macro2起finite/nonzero、validation/test action reads为0。contract为
  `31ea4bc9...55de0`，step3 payload为`2b50bafd...618f7`，profile已seal。
- 第一次frozen resume尝试在第二条invocation前出现一次15分钟setup collective卡死；
  只终止本方进程。随后相同六卡`all_gather_object`/`broadcast_object_list`最小探针
  通过，同一原命令重试也完整通过，因此目前只能标记为未复现的一次性runtime观察，
  不能伪称软件根因。formal fresh0→200不读取profile权重，launch保留live timeout与
  进程/GPU监控。
- commit`6f18499`的首次BCI formal在macro10前发现配置仍使用longest105 profile专用
  `teacher_video_seed=172`，而同一配置的sealed字段及ordinary SFB正式基线都要求
  formal seed`20260722`。本方在任何checkpoint前主动终止，六张卡完整释放；partial
  root和log只作aborted合同审计，禁止resume、评测或性能引用。修复把实际seed切回
  `20260722`，并在config loader增加sealed formal seed不一致即fail-close的回归门。
  root内`aborted_contract_incident.json`记录10 rows/0 checkpoint及四份证据hash，文件
  SHA256=`9d5d03b8...cf9907`。

### 0.1 BCI VR fresh 0→200 formal retry1 launch contract

- canonical workspace为`/data1/user/ymdai/projects/EMBER`；launch必须使用包含本段
  记录的clean commit，且现场核验`HEAD == origin/main`。分支名不改变run identity，
  精确branch/commit由自动`run_contract.json`记录。
- sealed config为
  `configs/pi05_as_writer_semantic_factor_basis_variance_reduced_long105_profile_v1.json`，
  当前SHA256=`333e4d6a...044492`，实际`teacher_video_seed=20260722`并由loader与
  `formal_teacher_video_seed_after_profile_seal`强制一致。source step1000 manifest SHA256=
  `c236cb2d...cd6bf`，tokenizer SHA256=`8986bb4f...8fc6`；source selected raw policy
  identity仍取sealed manifest的`60ea7ee8...df36`。data root为项目内迁移核验后的
  filtered LIBERO数据；launch执行sealed size/schema检查并按合同跳过重复全量SHA。
- output root固定为
  `/data1/user/ymdai/projects/EMBER/runs/outputs/pi05_as_writer_semfactor_vr_bci_rawfull24_decay400_formal_r6_b20_micro2_seed7_formalvideo20260722_retry1_20260803`；
  启动前必须不存在。log固定为
  `/data1/user/ymdai/projects/EMBER/runs/logs/ember_vr_bci_rawfull24_r6_b20_micro2_seed7_formalvideo20260722_retry1_20260803.log`，
  tmux固定为`ember_vr_bci_r6_b20_seed7_retry1_20260803`。错误seed的旧root/log原位
  保留并明确禁止作为retry输入。
- 规模为fresh macro0→200：96,000 logical action queries、4,800 one-video conditions、
  48,000 physical B2 policy forwards、8个every25 checkpoints。6-rank DDP每rank 4 tasks，
  logical B20、full24 raw mean、一次clip/AdamW/scheduler不变；profile checkpoint绝不
  warm-start。estimated peak新增容量按1.5GiB计；2026-08-03 08:03 UTC `/data1`
  personal quota为`256,638,532/1,073,741,824 KiB`，共享余量86TiB。
- exact inner command固定为：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,7 torchrun --standalone --nproc-per-node=6 \
  scripts/train_as_writer.py \
  --config configs/pi05_as_writer_semantic_factor_basis_variance_reduced_long105_profile_v1.json \
  --mode formal \
  --source-run runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722 \
  --checkpoint runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000 \
  --tokenizer-path models/tokenizers/openpi/paligemma_tokenizer.model \
  --data-root data/datasets/f13aa24a3da8c43c7225569f28c562979fa0e35a \
  --output-dir runs/outputs/pi05_as_writer_semfactor_vr_bci_rawfull24_decay400_formal_r6_b20_micro2_seed7_formalvideo20260722_retry1_20260803 \
  --skip-data-sha
```

- GPU assignment不是永久保留：launch前重新比较`gpu01`/`gpu02`，只在六张目标卡仍
  candidate时使用上式；若设备集合变化则先更新实际`CUDA_VISIBLE_DEVICES`与现场记录。
  `.venv` activation提供`NCCL_P2P_DISABLE=1`。启动后监控commit/root/device、invocation、
  metrics推进、finite/clip/OOM、quota和他人进程；15分钟内无invocation/start则只停止
  本方进程组并保留证据。
- formal自然完成后只评测single checkpoints 50/100/150/200的paired correct400；不
  根据loss挑点、不融合checkpoint。只有absolute/breadth/trend/internal path过门才
  exact-resume同一root到400；任何合同改变都fresh。profile和失败启动root不得作为
  resume来源。

### 0.2 BCI VR 0→200正式结果与阶段暂停

- 有效训练从clean pushed`d9130c9`和fresh identity自然完成到预注册stage stop200。
  canonical root仍为0.1节所列retry1 root；contract SHA256=
  `0f9ed99d...13599a`，config SHA256=`333e4d6a...044492`。200个macro严格对应
  optimizer step1--200，累计96,000 logical action queries、4,800 one-video
  conditions、48,000 physical B2 forwards，wall=`6619.670s`；all finite、0 clip、
  source trainable=0，validation/test action reads=0。8个every25 checkpoints的
  64/64 payload size与SHA均复验通过，8个512-row held functional panels完整。
- 在每次live GPU/quota preflight后，只使用`gpu02`物理0/1/2/3/4/7；5/6上他人进程
  从未触碰。A40 evaluator采用3 GPUs/panel、3 persistent replicas/GPU、3 Writer
  generators/GPU、generation batch4；正式root为：

```text
runs/outputs/pi05_as_writer_semfactor_vr_bci_correct400_noreplacement_seed7_macro0050_d9130c9_20260803
runs/outputs/pi05_as_writer_semfactor_vr_bci_correct400_noreplacement_seed7_macro0100_d9130c9_20260803
runs/outputs/pi05_as_writer_semfactor_vr_bci_correct400_noreplacement_seed7_macro0150_d9130c9_20260803
runs/outputs/pi05_as_writer_semfactor_vr_bci_correct400_noreplacement_seed7_macro0200_d9130c9_20260803
```

  四个root均为400 unique rows、42/42 complete shards、9/9 workers return0；每task
  50 states和teacher demos 0--49各一次，results/launcher-completion hash均通过。
  checkpoint之间以及各自ordinary SFB comparator的state、teacher video、env/policy
  RNG均400/400配对；成功提前终止只缩短noise数组，公共前缀逐项相同。
- paired correct400曲线与逐task结果如下；task列顺序为Long-1/2、Goal-3/6、
  Object-1/3、Spatial-1/3：

| macro | correct | breadth | per-task successes |
| ---: | ---: | ---: | --- |
| 50 | 76 | 7 | `3/1/0/37/29/4/1/1` |
| 100 | 88 | 4 | `4/0/0/37/25/22/0/0` |
| 150 | 126 | 7 | `4/2/1/41/42/34/0/2` |
| 200 | 107 | 5 | `8/0/0/39/33/24/0/3` |

  single winner为macro150=`126`，比严格最低通过分151少25、比v6-fast winner143少17，
  也未超过ordinary SFB winner127。相邻gained/lost为50→100=`30/18`、
  100→150=`49/11`、150→200=`21/40`；四点success union/intersection=`158/49`，
  single envelope gap=`32`。breadth在`7/4/7/5`之间切换，VR没有解决task漂移。
- 对同点ordinary SFB的paired delta为`+7/-3/+8/-20`；对应gained/lost为
  `23/16`、`18/21`、`33/25`、`21/41`。macro200下降覆盖7/8 tasks，只有
  Object-1和Spatial-3略升。VR winner150相对source base gained/lost=`83/5`，说明
  Writer产生真实新能力；相对v6-fast macro400为`27/44`，没有形成新上限。
- 与ordinary SFB完全matched的前200步诊断中，VR仅把same-task successive all-block
  CountSketch cosine平均提高`.002634`、factor提高`.005104`，raw mean/sample energy
  retention提高`.001914`、factor提高`.001121`；51--100段方向cosine反而更差，
  151--200段energy retention也没有改善。这个效应小、分阶段反复，不构成material
  gradient stabilization。
- held functional loss在macro200达到VR四点最好`.129146`，且优于同点SFB
  `.131776`，但closed-loop同时从VR macro150的126跌到107、并比SFB macro200少20。
  逐task loss与绝对成功的相关性主要反映任务难度；24个相邻task变化的Spearman仅
  `.263`。因此正式拒绝“可约flow Monte Carlo方差是主要漂移根因”，把最早剩余接口
  升级为functional action surrogate与source-policy closed-loop有效流形错位。
- owner要求在四点rollout与全部分析完成后先暂停。当前无EMBER tmux、训练/eval
  worker或本方GPU占用；不续到400、不做五臂、不启动下一架构/训练目标，等待owner
  看完本次状态后继续指示。长期`>150` Goal仍未完成。

### 0.3 owner恢复推进与Semantic Direction Store

- owner随后已解除上述阶段暂停：继续严格one-shot，取消Writer参数量软上限，优先
  重构条件生成方向的存储/组合，并允许配套修改训练方式；仍不使用subagent。推进以
  效率优先，只做shape、信息墙、identity、freeze、gradient、OOM、resume和正式结果
  所需的聚焦检查，不重复全量hash或无关历史扫描。
- 当前不再把“held functional loss不能预测rollout”本身称为task漂移根因。更直接的
  内部边界是SFB route已task-conditioned，而shared factor仍占约97% gradient energy、
  task mean只保留约4.2%条件能量且一阶方向持续轮换。
- 新设计authority为
  `docs/action_forecast_writer_semantic_direction_store_design.md`：frozen text-only
  language anchor只用24 train languages建立8个固定semantic centers；每task稳定等权
  top2，每个store拥有完整独立1024→256→factor-width参数。完整Core/A/E/D仍是唯一
  factor value，Writer实际参数37,355,776。在进入formal前，canonical实现、
  24-train-language center authority、61项focused CPU合同与clean六卡profile已完成；
  当时尚无效果结论。
- 当时封存的下一执行顺序是从sealed formal seed`20260722`和clean origin-main
  fresh0→200，再做
  50/100/150/200四点paired correct400；不复用profile/VR checkpoint或
  Latin/antithetic estimator。
- clean `7b13b6c`首次六卡profile在训练循环前复现NCCL 480秒heartbeat失败；六rank日志
  均明确`only active collectives: 0`，且当时只有rank-local source CUDA构造在运行，
  不是Direction Store collective、OOM或科学non-pass。该root已停止且禁止resume。
  根因修复为延后NCCL生命周期：rank先完成local policy/Writer/optimizer CUDA构造，
  经独立FileStore all-rank-ready rendezvous后才允许任何rank建process group；不得让
  快rank提前创建NCCL，也不得用放宽heartbeat或timeout封口。
- `78d8b4f`重放确认生命周期修复后，六rank统一进入`SeqNum=1/ALLREDUCE/Numel=1`，
  随后暴露BCI迁移期已裁决的第二层transport合同：显式launch漏传
  `NCCL_P2P_DISABLE=1`，direct P2P/CUMEM在600秒超时。相同`gpu02:0--5`六卡加该变量
  后，scalar sum=`21`、BF16 matmul finite及第二次all-reduce在10.5秒内全部通过。
  因此BCI A40 launcher与代码现同时显式/fail-fast要求SHM transport；第二root同样
  aborted且禁止resume。
- clean `eaa8bce`随后在精确空闲拓扑`gpu02:1,2,3,4,5,7`完成两次collective sum21、
  all-rank CUDA-ready、NCCL与run-contract发布，证明两层多卡根因均已越过；进入step0
  后由`as_step.py`一份退役的重复method白名单拒绝新Direction Store method。canonical
  `as_config.py`此前已完整验证该conditioning合同，因此修复是删除第二份字符串白名单，
  让step owner只执行已验证合同；该root无metric/checkpoint，不跨commit resume。
- clean pushed`1d0507e`最终在`gpu02:0--5`完成fresh0→1和exact-resume1→3，contract
  `749773d8...8fd6`。三步`33.451/31.823/31.025s`，loss
  `.150377/.152492/.142434`，最长105帧，峰值allocated/reserved
  `35,827,363,840/47,129,296,896` bytes；1,440 queries、72 one-video conditions，
  validation/test action reads=0且无clip/OOM。step2起五个主块全部finite/nonzero；
  配置现切回formal seed`20260722`并seal，正式run必须fresh identity。

### 0.4 Semantic Direction Store正式结果、内部裁决与当前暂停

- clean pushed`91feeef`从fresh identity在`gpu02:0--5`完成macro0→200。canonical
  root为
  `/data1/user/ymdai/projects/EMBER/runs/outputs/pi05_as_writer_direction_store_bci_rawfull24_decay400_formal_r6_b20_micro2_seed7_formalvideo20260722_91feeef_20260803`。
  200个macro累计96,000 logical action queries、4,800 one-video conditions、8个
  every25 checkpoints，wall=`6619.255s`；all finite、0 clip/OOM、0 validation/test
  action reads，峰值CUDA reserved=`39,806,042,112` bytes。profile/VR/SFB权重均未
  warm-start。
- 只用live preflight后`gpu02`六张空闲卡完成macro50/100/150/200的strict paired
  correct400；gpu01持续有他人任务，gpu02:6有他人进程且从未触碰。四个root依次为：

```text
runs/outputs/pi05_as_writer_direction_store_bci_correct400_noreplacement_seed7_macro0050_91feeef_20260803
runs/outputs/pi05_as_writer_direction_store_bci_correct400_noreplacement_seed7_macro0100_91feeef_20260803
runs/outputs/pi05_as_writer_direction_store_bci_correct400_noreplacement_seed7_macro0150_91feeef_20260803
runs/outputs/pi05_as_writer_direction_store_bci_correct400_noreplacement_seed7_macro0200_91feeef_20260803
```

  state、teacher demo和policy RNG公共前缀全部严格配对，0 retry/failure。task顺序为
  Long-1/2、Goal-3/6、Object-1/3、Spatial-1/3：

| macro | correct | breadth | per-task successes |
| ---: | ---: | ---: | --- |
| 50 | 129 | 7 | `7/2/0/42/45/31/1/1` |
| 100 | 107 | 7 | `5/1/1/37/37/22/0/4` |
| 150 | 120 | 7 | `9/2/0/40/40/26/2/1` |
| 200 | 129 | 5 | `10/0/0/38/41/36/0/4` |

  macro50与200同分，按更高breadth和更早成本选macro50为唯一winner。相邻gained/lost=
  `17/39,43/30,27/18`，四点union/intersection=`174/65`、single envelope gap45。
  相比SFB macro50提高60，但未超过v6-fast143或严格门151，且后续仍明显换手，因此不
  续到400、不做五臂。
- step133的task-pair梯度分层显示shared0/1/2 stores的factor cosine均值为
  `-.00043/.00664/.02249`：fixed semantic stores局部化了干扰，但store内部仍近正交。
- winner macro50的完整refs1五条件内部分析成功root为
  `runs/outputs/pi05_as_writer_direction_store_bci_macro0050_internal_refs1_seed7_retry2_a115b06_20260803`。
  8 tasks的ordered top2数组均不同（其中`1,5`与`5,1`是同一无序组合），且route跨
  video固定；same-task-other的Program/factor/
  effective-BA relative-L2为`.93377/.01935/.03242`，shuffled为
  `.81049/.04731/.07193`，reversed为`.93086/.09808/.15963`。A/E与Core mean
  carrier均传到BA/action，动态路径没有断路，但其差异在compiler后被强压缩。
- 全部16个rank坐标active，effective LoRA norm均值`43.86494`，但rank90/rank99均为
  1、stable rank=`1.000043`、entropy rank=`1.000371`、top singular energy=
  `.999957`、B-column cosine=`.999971`。Direction Store改善了早期acquisition和参数
  ownership，却仍把public rank16写成几乎同一B方向；正式拒绝
  “factor parameter coexistence是主要完整根因”。
- 内部分析首次重放暴露assignment隐藏4-rank默认，第二次暴露final seal固定4 payload/
  每rank2 tasks。`f82c7cd`与`a115b06`分别把LPT ownership和Cartesian sealing绑定
  实际`world_size`；8项定向测试及clean六卡真实规模均通过。该根修与BCI transport/
  process-group生命周期规则均写入`AGENTS.md`并push到branch/main。
- owner要求rollout和全部分析后暂停了解现状。当前正式训练、四点rollout与winner内部
  分析均结束；没有EMBER训练/eval/analysis进程或本方GPU占用。不得启动下一架构、
  training target或GPU工作，等待owner明确继续指示；长期`>150` Goal仍未完成。

## 1. 当前边界

- owner此前授权在当前BCI上继续环境适配、架构/训练设计、profile、正式训练、严格配对
  评测和内部分析；目标是缓解task漂移，并使同一single checkpoint的correct aggregate
  严格超过`150/400`后继续提高。Direction Store rollout与全部内部分析完成后，owner
  最新边界是先暂停了解现状；当前不得自动启动下一实验。推进期间仍不使用subagent。
- 当前写分支为`codex/bci-continuation`，BCI新增输出只写项目`runs/`，证据写
  `evidence/`。下列A100窗口、旧分支和`/data/ymdai`只保留历史provenance。
- owner在迁移由另一session启动后重新开放约十小时A100 post-seal研究窗口，允许在
  原信息墙/split/安全合同和物理GPU4--7边界内继续架构、训练、评测与分析。窗口以
  `2026-08-02 19:18 UTC`起算，约`2026-08-03 05:18 UTC`硬停；操作上最迟`03:45 UTC`
  冻结新实验，为二次迁移留出时间。
- 已迁移封存基线为`f9a144c`；本轮所有Git与artifact都是post-seal delta，外部登记根
  为`/data/ymdai/migration_manifests/ember_postseal_20260802/`。迁移仍由另一session
  执行，本session不修改其现有副本，只提供增量清单。
- 本A100研究窗口的训练、评测、内部分析和GPU profile均已结束；当前没有需要继承的
  tmux、torchrun、评测worker或GPU实验。MemLLM同样没有活动实验。
- EMBER迁移封存基线为`f9a144c94e71bb44373d7247ed0fded2ed835305`；Semantic
  Factor-Basis仍是canonical Writer；A100最后push的VR实现commit为`50662a8`。
- Target-Bound Role-Preserving Program 已在远端分支
  `origin/codex/target-bound-role-program`实现，commit
  `b260a57a94dc21bd3446b212bfa42f71b037ce13`。它只完成 CPU shape、identity、
  causality、gradient、checkpoint 等结构验证；没有做 B20 profile、resume、训练或
  rollout。不得把它写成实验结果。
- Target-Bound已完成fresh0→200；macro50/100/150/200 paired correct400为
  `75/120/90/110`，winner macro100仍明显轮换，因此不续训、不做行为五臂。winner
  refs1证明remove-A、remove-D、causal-memory reversal均8/8过门，Core-only与
  Program-only都不能复现full BA；视频主路径真实到达BA/action，最早剩余失败接口是
  shared factor conditional coexistence。
- Semantic Factor-Basis只替换这一接口：Core以Q/K软选择四个unit-mean factor value
  bases，完整Core/A/E/D仍作为value；不加task ID、gate、scale或额外loss。精确参数
  11,159,296。`e87363f`的longest105 B20三macro及formal-seed fresh0→1/
  exact-resume1→3均通过，五个主block从macro2起finite/nonzero；seal/push commit为
  `f5ddfe3`。
- clean frozen`f5ddfe3`从fresh identity完成0→400、every25；不从profile/smoke
  warm-start。完整paired correct400为`69/91/118/127/117/81/126/120`，single
  winner仍是macro200。八点success union/intersection=`193/39`、single envelope
  gap=`66`；250→300 lost52、300→350 gained60，第二小时明确证明能力轮换而非成熟化。
  formal root：
  `/data/ymdai/outputs/ember/pi05_as_writer_semfactor_postseal_rawfull24_decay400_formal_r4_b20_seed7_f5ddfe3_20260802`；log：
  `/data/ymdai/logs/ember/pi05_as_writer_semfactor_postseal_resume200to400_r4_b20_seed7_f5ddfe3_20260803.log`。
- variance-reduced estimator保持SFB拓扑、objective期望、B20/full24/optimizer不变，
  只对flow time做exact-Beta Latin分层并对Gaussian noise做随机antithetic pairing。
  BCI正式0→200与四点correct400=`76/88/126/107`均已完成；机制改善小且非持续，
  held functional loss与closed-loop在macro200明确错位，方法已负裁决，不续到400。
- 迁移步骤、路径映射、资产分流和新 Codex 接手顺序统一看
  [`a100_to_bci_migration_handoff.md`](a100_to_bci_migration_handoff.md)。

## 2. 最新 closed-loop 结论

### 2.1 CV-ADR RAW

canonical root：

```text
/data/ymdai/outputs/ember/pi05_as_writer_cvadr_rawfull24_taskcomplete_decay400_formal_dev_r4_b20_seed7_254ade4_20260802_retry1
```

- fresh identity 完成 macro0→400，192,000 action queries、9,600 one-video
  conditions，all finite、0 clip，validation/test action reads为0。
- paired correct400 在 macro50/100/150/200/250/300/350/400 为：

```text
76 / 111 / 99 / 117 / 77 / 69 / 80 / 82
```

- single winner 是 macro200=`117/400`。第二小时不是成熟化：200→250为
  16 gained / 56 lost，后段 LoRA norm没有坍缩而行为持续退化，因此未做五臂。
- macro200与400的matched梯度方差分解显示，video主效应仅约
  `.1211%/.1060%`且0/24 tasks主导；query约`48.59%/49.53%`，flow及
  query×flow约`48.78%/48.50%`。24/24 matched train functional loss继续下降，
  correct400却`117→82`。
- 晚期factor block约占task-gradient energy的`94%`；参数段方向在低LR仍不稳定，
  held functional loss横盘。最可信根因是视频条件梯度低SNR、query/flow噪声、
  shared compiler写出与closed-loop有效流形错位共同作用，不是单纯LR、rank或norm。

内部根：

```text
/data/ymdai/outputs/ember/pi05_as_writer_cvadr_rawfull24_macro0200_internal_exact50_seed7_ff988dc_20260802
```

exact50确认Core与Program两路都必要，但Action/order仍弱：remove-A只在1/8 tasks达
预注册门，remove-D为5/8；same-task effective-BA centered variance/sample energy
约`.10494%`，fixed-action中位变化约`.00856%`。LoRA norm`64.24`、stable rank
`1.0072`，所以不是Target-Spectral式增益或coherence坍缩。

### 2.2 CV-ADR normalized GROUP4

canonical root：

```text
/data/ymdai/outputs/ember/pi05_as_writer_cvadr_group4_taskcomplete_decay400_formal_dev_r4_b20_seed7_51c0ba5_20260802
```

- 完成1200 physical updates=200 cycles、96,000 queries、4,800 videos，all finite、
  1 clip；cycle50/100/150/200 paired correct400为：

```text
82 / 77 / 73 / 110
```

- single winner cycle200/step1200=`110`，低于RAW winner`117`，四点均值
  `85.5<100.75`；breadth6、top2占`71.82%`，未解决能力轮换，不做五臂。
- GROUP4比RAW保留更多source successes（42/48 vs 34/48），但没有共同获得更多新
  能力。effective norm反而`64.24→72.06`，held loss略低而closed-loop更差。
- exact50中A+D、remove-A、remove-D职责门由RAW的`8/1/5 of 8`降为`0/0/0`；
  Effect-only到full BA的relative L2由`.06744`降为`.01882`，contextual-memory
  reversal由`.00607`降为`.00311`。它学到更大、更coherent、却更static和
  off-manifold的写入。

内部根：

```text
/data/ymdai/outputs/ember/pi05_as_writer_cvadr_group4_cycle0200_step1200_internal_exact50_seed7_51c0ba5_20260802
```

结论：normalized GROUP4、CP式负冲突解释和“减少optimizer gain即可稳定”均没有
获得支持。full24 raw mean很少直接对task candidate为负，pairwise negative cosine
不能自动解释漂移。

## 3. 架构×训练方法的关键反事实

以下四个single winner均从正式400-row paired artifacts逐行重验：

| 架构×recipe | correct | same | wrong | shuffled | reversed |
| --- | ---: | ---: | ---: | ---: | ---: |
| v5.2 old recipe, step900 | 132 | 138 | 74 | 82 | 83 |
| v5.2 task-complete, macro400 | 120 | 109 | 107 | 111 | 124 |
| v6 old recipe, step500 | 121 | 122 | 111 | 84 | 47 |
| v6 fast task-complete, macro400 | 143 | 135 | 125 | 128 | 129 |

必须继承的解释：

1. task-complete在v5.2和v6上都压弱Procedure→effective BA/action与顺序margin，
   但correct absolute分别`-12/+22`；架构和recipe不能独立判死。
2. matched 150-video visits时，v5.2 task-complete相对old为`-81`，v6为`+16`；
   v6的Visual Transition/Core-conditioned transition是正证据，但其selected
   `+22`又几乎由一个Object task的`+24`贡献，不能说漂移已解。
3. old recipe每task-cycle六次Adam会恢复更强slots/AdaLN/动态写出，也会产生低
   breadth和近正交参数轨迹；退回old recipe不是解法。
4. post-v5的v7、v8、v10、Loom、Recenter、Core-Program、Prior、Target-Spectral、
   SPG、UCP、AP和CV负结果都与训练operator混杂。它们各自的局部失败接口有正式
   证据，但不能据此宣布其全部思想在任意训练方式下无效。
5. functional loss下降不等于closed-loop改善；强行高rank/正交、全局scale、gate、
   B-only residual、多video/LoRA平均或checkpoint融合都没有当前依据。

四格正式联合审计root：

```text
/data/ymdai/outputs/ember/pi05_as_writer_v52_v6_recipe_video_causality_audit_seed7_20260802
```

analysis SHA256：
`98371337e2cf1f7cec09d04e81445b419fc21c654fe173cb081a4b5e63092efa`。

### 3.1 Semantic Factor-Basis最终裁决

完整曲线与逐task（Long-1/2、Goal-3/6、Object-1/3、Spatial-1/3）为：

| macro | correct | per-task |
| ---: | ---: | --- |
| 50 | 69 | `3/1/0/39/17/7/1/1` |
| 100 | 91 | `7/0/1/38/26/15/2/2` |
| 150 | 118 | `14/1/0/40/32/28/3/0` |
| 200 | 127 | `13/2/1/44/31/32/3/1` |
| 250 | 117 | `14/1/0/42/30/27/1/2` |
| 300 | 81 | `12/0/1/42/17/9/0/0` |
| 350 | 126 | `22/0/1/40/33/29/0/1` |
| 400 | 120 | `20/0/1/43/23/32/1/0` |

macro200相对source base paired gained/lost=`84/5`，证明Writer提供真实新能力；相对
v5.2-old为`49/54`、v6-fast macro200为`33/39`，没有提高现有上限。后半段
raw-full24 candidate-negative tasks始终为0，但gradient energy retention从
201--250的`.04443`降到351--400的`.04203`，factor share从`.9586`升到`.9691`，
same-task successive cosine从`.0676`降到`-.0099`。相邻checkpoint Adam一阶moment
近正交而二阶moment高度稳定，说明主要现象是条件方向/functional sample持续轮换，
不是全局mean即时伤害某些task，也不是scale统计失控。

macro200既有内部root仍是本版winner机制authority：

```text
/data/ymdai/outputs/ember/pi05_as_writer_semfactor_postseal_macro0200_internal_refs1_seed7_18d3e89_20260802
```

它证明route与A/E/D→BA→action工作，但task routing只部分解决shared-factor共存。由于
absolute未达到strong门，第二小时不新增same/wrong/shuffled/reversed 1600个rollout。

## 4. 当前代码与下一实验边界

Semantic Direction Store已原位替换为canonical Writer path；历史SFB、Target-Bound、
CV-ADR与VR由Git、frozen config和artifacts保存，不保留并行活动模型。核心职责为：

- 38个真实policy targets先读Core；
- target-bound地读取Action、Effect与Change；
- A/E/D使用private causal temporal channels和private rank reads；
- 16 rank coordinates最后展开；
- identities只进入Q/K，raw evidence进入V；
- frozen language anchor减去train24均值后固定等权选择top2/8 stores；
- 每个store独立拥有八个完整factor input/output heads，所有value仍来自完整Core/A/E/D；
- factor heads保持coherent near-rank1高增益，不加谱/正交/entropy约束。

当前A100临时授权窗口、BCI VR 0→200和四点正式评测均已完成。owner已恢复推进并
取消Writer参数量上限；紧邻候选不是继续给SFB加窄basis，而是用固定language语义地址
组合完整独立factor direction stores。设计见
`docs/action_forecast_writer_semantic_direction_store_design.md`；实现必须fresh schema，
当前fresh schema、37,355,776参数、center authority和focused CPU合同已完成，不复用
VR checkpoint或estimator。

不得从smoke/profile权重warm-start。当前完整设计为
`docs/action_forecast_writer_semantic_factor_basis_design.md`；VR设计及其正式负结果在
`docs/action_forecast_writer_variance_reduced_functional_estimator_design.md`。
Target-Bound设计与正式负结果保留在Git、该文档和post-seal artifacts中。

## 5. 迁移时必须保留的EMBER科学资产

- frozen source raw policy：
  `/data/ymdai/outputs/ember/pi05_source_base_v1_seed7_1k_e2cc238_20260722`；
  policy SHA256
  `60ea7ee898629321cf34522e5f0e45f4f1c2659c5f5dbc7b02ed9eb46a8cdf36`。
  rejected EMA和训练resume状态已清理；它现在是inference/source asset，不是完整
  source-SFT resume包。
- canonical feature cache v2：
  `/data/ymdai/outputs/ember/pi05_writer_feature_cache_v2_development32_raw_e4c19f9_b32_20260722`。
- 原迁移封存的60个正式/历史训练checkpoint roots、406个完成evaluation roots，
  加上post-seal的2个正式训练root、12个formal correct400 roots及内部analysis。
  它们是训练漂移与架构×recipe复核的唯一证据，不能只迁winner；精确增量只取
  `/data/ymdai/migration_manifests/ember_postseal_20260802/assets.tsv`。
- `/data/ymdai/logs/ember`、tokenizer、精确revision的426.57MB LIBERO simulation
  assets和`/data/ymdai/migration_manifests`。

cleanup已删除的profile/resume/reseal/cache路径若仍出现在历史文档中，表示工程
provenance，不表示artifact损坏，也不授权重跑。精确删除清单和SHA都在：

```text
/data/ymdai/migration_manifests/a100_cleanup_20260802
```

## 6. 新Codex接手顺序

本机Codex sessions、archive、auth、cache和worktree不迁移；它们不是authority。
新Codex在BCI上应先：

1. 核验Git HEAD、origin、工作区和迁移资产hash；
2. 完整阅读`AGENTS.md`要求的authority文件；
3. 优先读本文件、迁移handoff、`docs/execution_brief.md`、CV与Target-Bound设计；
4. 检查BCI实际路径并设置`EMBER_STORAGE_ROOT`、owner cap及
   `EMBER_LIBERO_ASSETS_ROOT`，所有source/checkpoint/tokenizer/data/output路径继续
   通过CLI显式传入；
5. 在owner恢复实验授权前保持无GPU作业状态。

旧Codex对话不能代替上述Git文档。任何与本交接冲突的历史“live”段落均视为过期。

# EMBER Execution Brief

## 0. Current operation

PICK只因full48 condition=`483.61515>200`退役；PICK-GC随后把condition降到`152.61`并通过
Program→LoRA→action、world4吞吐和zero-memory部署门，但formal fresh`0→10`后的single-checkpoint strict
paired correct只有`138/400`、breadth6。相对immutable macro0 retained/gained/lost=`118/20/16`、churn36，
未过`correct>=144`与`lost<=8`门，因此PICK-GC+blind offline source-action credit也已退役，不resume、不补
controls、不sweep。OSG-PC随后在唯一world6 profile中因rank-local长尾触发600s NCCL watchdog，wall lower bound
至少是matched baseline的`1.912x>1.25x`，未到full48 report。SKNC用K4 4/4 binary success keys约束最终shared
Program write，clean world3机制与B32部署门均通过；formal fresh`0→5`后strict paired400=`137/400`、breadth7，
相对old134 retained/gained/lost=`121/16/13`、churn29。macro5 bank15、rank36、projected energy`.592`与Program
closure健康，但Long净`+7`掩盖Object/Spatial净`-5/-1`，未过`correct>=140`、`lost<=8`和单task集中门。
SKNC已退役，不resume、不补controls、不sweep。SRTP随后让fixed-landmark mixed reward tangent约束最终shared
Program update，但两个clean world3 attempts都在mixed reward CFM处三rank OOM。`d172add`首轮暴露decoder graph
跨K4保留；`e31e2fd`释放graph并只在Nmc4后重解compiler仍申请`254/484/484 MiB`失败，最差free仅16.31MiB。
两次均无mechanism report/checkpoint，deployment/formal未授权；当前SRTP执行合同终局退役，不降batch、改dtype、
开allocator或做第三次修补。PCUG随后完成canonical实现与`344 passed`，但唯一clean world4 discarded macro在
Phase A full24 gather前就越过wall hard gate：`809.72185s / 2.25568x>1.5x`下界，物理3--5持续100%、物理6
先等待；无OOM/nonfinite、paired probe、mechanism report或checkpoint。PCUG当前execution contract终局退役，
deployment/formal与同配置重跑关闭。Work-Queue successor随后完整执行actual pairing：world3 Phase A=
`44.74125s`、total=`558.05862s / 1.16596x SKNC`，48 exact pairs有7 discordance、3 gains/4 losses、3 harmful
tasks跨2 suites；correct guard rank33、energy`.76492`和全部closure健康。唯一失败是final negative-null：blind
ratio`.03991`经correct-only projection变成`.50179`，wrong/shuffled/reversed各`0/8`达门。因此WQ-PCUG退役，
deployment/formal、重跑和sweep关闭。

Negative-Preserving Candidate Guard随后修复了WQ的final negative-null：matched reprofile 20/20全过，final
negative ratio`.03524`、三类各`8/8`，部署选择B32；formal五宏的correct/negative closure、rank和energy也持续
健康。但macro5 strict paired400只有`135/400`、breadth5、per-task=`0/2/46/36/0/37/14/0`，相对old134为
retained/gained/lost=`117/18/17`、churn35，未过correct、breadth与lost门。train24×50 action-hidden audit显示
first-stable point对held same-task videos仍有`.40954`平均正交残差；故最早失败推进到point address无法定义
跨视频/held occupancy support neighborhood。NPCG已退役，不resume、不补controls、不小扫。

CVEG的机制与one-shot B32部署门已通过，但formal把最早失败推进到离散candidate credit反馈。canonical
`ad7d9bd` fresh run先因NCCL把超过480秒的rank-local rollout误判为watchdog hang而零宏退出；只关闭heartbeat
monitor后的retry完成macro1/2，却在macro3 sealed feasible-set gate主动拒绝，无checkpoint。为取得拒绝值加入的
异常可观测性不改变成功路径；同world3诊断复现的macro1逐值一致，但macro2 task23在不同物理rank上的一条paired
rollout从candidate gain翻成candidate loss，令current harmful guard不同，Program轨迹随即分叉并完成5宏。
诊断五宏bank=`10/14/16/16/16`、energy=`.354/.375/.287/.274/.498`，K2 candidate净值依次
`-3/-1/-3/-3/0`，全部E/negative/guard closure仍通过。这否决当前`binary K2 -> current hard equality guard`
作为稳定共同积累机制。诊断macro5随后完成唯一directional strict400：`131/400`、breadth6、per-task=
`1/2/47/30/0/35/16/0`、per-suite=`3/77/35/16`。相对old134严格配对为`113 retained/18 gained/21 lost`，
相对NPCG135为`114/17/21`；Object3相对NPCG净`-6`，Long1虽净`+2`却有`10 gained/8 lost`。因此当前hard E
组合也没有absolute或retention价值，不能原样继承；CVEG不resume、不补controls。

长期成功条件是同一single checkpoint strict paired correct严格`>150/400`，并具备breadth、低换手、
same-task鲁棒和correct优于wrong/shuffled/reversed/no-video。历史最好仍为v6-fast`143/400`。

## 1. Successor design gate

任何新successor在实现前必须回答：

1. 它改变的唯一主要变量是什么？
2. 它针对`docs/research_history.md`中的哪个最早失效接口？
3. 与最接近历史架构相比，保留了哪些已验证优势？
4. 什么内部证据能快速判定机制是否接通？
5. 何时做真实paired400，什么结果立即停止？
6. 如何避免按held task outcome反向选择target/rank/route？
7. 预计GPU、wall、显存、存储峰值和可恢复状态是什么？

最新接口裁决是：PICK-GC key、condition-local FP32 Program和native rank16 compiler已接通；最早科学失败仍在
blind train24 offline cotangent→held on-policy useful support/coexistence。OSG-PC试图用成功train24 executed-prefix
half-space保护support，但current full-replay per-success VJP graph先在吞吐/长尾接口失败，尚未产生shared guard
transfer证据。后继必须保留信息墙与已通过接口，同时正面限制on-policy credit的cost/occupancy length。
SKNC把Program到action的protected motion压到零，但strict仍lost13并发生suite换手，说明single-video key不是
当前shared candidate是否有害的充分证据。SRTP又证明完整reward policy-gradient执行图不适合A40合同。PCUG因此
先产生真正准备写入的blind `D0`，再在两个相同initializations上配对base/candidate闭环结果；harmful与当前
stable-success keys只在最终write前对`D0`做closest equality projection，harm不持久化。它测量实际update的因果
损失而非surrogate gradient，不让task-local guard被后续full24 solve改写。完整公式与hard gates见
`docs/action_forecast_writer_paired_candidate_update_guard_design.md`。
Work-Queue已证明actual pairing有内容且吞吐可接受，也定位correct-only projection会破坏blind negative suppression。
Negative-Preserving successor不把完整update硬压进`Null([G;N])`，而只令最小guard correction `C`满足`NC=0`，
从而保留`ND1=ND0`；它已证明该constraint composition可行，但135/400结果否定point guards足以保护held
policy-effective support。完整retired authority见
`docs/action_forecast_writer_negative_preserving_candidate_guard_design.md`；WQ sealed边界见
`docs/action_forecast_writer_work_queue_candidate_guard_design.md`。
退役CVEG的完整公式、采样合同、稳定性与directional strict裁决见
`docs/action_forecast_writer_cross_video_equivariant_candidate_guard_design.md`。

active successor是PVJFC，只替换shared credit acquisition：每task的primary/companion都构造完整one-shot
Program与cotangent，两view共享同一个B20及policy RNG，以固定半权进入96-row continuous solve；各自negative为
zero RHS。它不平均video/feature/LoRA/cotangent，不读outcome，不保存success bank，也不强制两view motion相同。
swap invariance和duplicate-view退化为single-view是hard contract；完整design与快速否决门见
`docs/action_forecast_writer_paired_video_joint_functional_credit_design.md`。canonical实现已原位替换CVEG：两个view
串行释放policy graph，full24 dynamic queue后一次gather 48 correct、48 zero-RHS negative和两套Program cotangent，
只保留一份FP32 Program memory；旧outcome/bank/hard-E executable与测试已删除，完整CPU`329 passed`。唯一live
macro0 profile为`11/12` checks通过但总门non-pass：rank`48/96`、24/24双view descent、negative`.07266`、全链路
与`51.543s` wall通过，regularized condition=`597.861>200`失败。没有checkpoint，formal关闭且不做damping/
threshold等小扫。active successor是CGIK-JC：只把condition组合改为
`[causal, goal*causal]`，保留PVJFC continuous paired credit与full96合同。cache预验selected correct48
condition=`163.88`且rank48；完整公式、边界和一次性profile门见
`docs/action_forecast_writer_causal_goal_interaction_key_design.md`。condition过门仍不能冒充closed-loop结论。

## 2. Fixed information contract

- 输入：exact task language + exactly one action-hidden teacher video。
- video是唯一dynamic value；language不能形成LoRA bypass。
- 禁止teacher action/proprio/reward/terminal、task ID、filename、object pose和hidden normalization。
- 输出：一套完整38-target public rank-16 LoRA；Writer在rollout前运行一次后释放。
- source policy、normalization、split、frame stride5、LIBERO preprocessing和policy interface固定。
- 每rollout随机无放回取正确task的一条teacher video；不挑最好video。
- 不做video/LoRA/checkpoint平均、融合或第二套LoRA。

few-shot若被新设计采用，必须显式改变本节合同：固定`k`或定义动态集合语义，仍action-hidden，不挑video，
同时保留matched one-shot arm和计算成本报告。

## 3. Data and assets

- split：`configs/libero_24_8_8_v1/`，24 train / 8 validation / 8 test。
- source corpus：`configs/pi05_source_corpus_v1/`，过滤后71 tasks。
- target manifest：`configs/pi05_target_data_v1/`。
- source policy：
  `runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722/checkpoints/step_00001000`。
- task experts：
  `runs/outputs/pi05_task_expert_bank_formal_step1000_r6_81101fe_20260807`中的统一step2000。
- tokenizer、model、dataset与simulation asset由CLI/`.env.local`指向canonical BCI roots。

不复制大型资产；优先复用canonical path/symlink/manifest。历史A100绝对路径只作provenance。

## 4. GPU selection and throughput

每次launch都重新：

1. 同时检查`gpu01`、`gpu02`的GPU index/UUID、memory、utilization、health和compute process；
2. 选择一个节点；
3. 使用该节点至多6张健康、低利用率、显存余量足够且能提高吞吐的A40；非零显存或低利用率进程不自动排除；
4. 在进程spawn前再次核对；
5. 结束后确认本次进程退出并释放设备。

上限6卡，但不要求6卡。不等待凑满、不dummy occupancy、不跨节点拼碎片，不抢占或明显干扰他人进程。单卡
mechanism/profile按其科学目的使用单卡；strict evaluator按selected node至多6张有益卡动态扩展，不使用NCCL。
多卡训练必须：

- `NCCL_P2P_DISABLE=1`
- physical GPU到local rank显式映射
- 每rank绑定GPU-local NUMA CPU/memory
- 在大资产加载/CPU准备完成后deferred NCCL init
- exact-resume保持原world size/topology

吞吐优化规则：

- 从能充分利用显存的batch开始profile，再向上扩到吞吐平台、allocator抖动或OOM边界；
- 选择真实longest-video panel上samples/s最高的stable点，不按空闲显存或理论batch选择；
- 原生BF16/TF32、SDPA、batched env、persistent workers、prefetch和集中D2H默认开启；
- 合并重复host sync、per-row forward、重复token/video处理和小tensor传输；
- 不为底层微差固定batch1、重复single forward、扩dtype或逐tensor扫描。

## 5. Storage preflight

expensive profile/training/evaluation/cache前必须：

- 在`strg01`查询目标filesystem的`ymdai`独立quota；
- 测量canonical root当前使用；
- 估计peak新增，包括checkpoints、cache、shards、logs、temporary和partial resume；
- 确认峰值不超过quota；`df -h`不能替代user quota。

`/data0`与`/data1`预算分开。正式outputs默认放`/data1/user/ymdai/projects/EMBER/runs/outputs`并通过CLI显式
指定。profile/smoke checkpoint在完成裁决且无consumer后不长期保留；formal checkpoints和paired raw rows保留。

## 6. Formal launch contract

expensive retained run前登记一个简洁合同：

- clean pushed source commit与detached frozen worktree；
- exact config/command/env；
- source model/tokenizer/data/split/normalization/video assets；
- output root必须fresh或由exact-resume schema唯一允许；
- node、physical GPU、world size、NUMA/rank topology；
- batch、precision、optimizer/stateful estimator、sampler和RNG；
- peak storage/wall估计；
- 科学问题、hard gate和下一评测点。

同一run未改变合同的resume复用原记录；command、scale、inputs、devices、overwrite、cost或scientific contract
变化时更新。不得在dirty checkout启动formal，不得让两个writer写同一root。

smoke只证明load、shape、freeze、gradient、OOM、resume和env；profile只选吞吐；mechanism只定位接口。三者都
不能解释closed-loop或冒充formal。

## 7. Training semantics

任何future AS-like训练默认：

- 24 train tasks task-complete等权；
- 每task一条video生成一套LoRA；
- B20 logical same-task cross-episode action queries，先task内mean再task等权；
- video/action episode独立，不制造逐帧低层对应；
- frozen source policy没有trainable参数；
- validation/test actions或reward不产生梯度；
- checkpoint含完整model/stateful update、cursor/sampler、RNG和topology。

新objective若改变task aggregation、video数量、query语义、policy interaction或optimizer，必须在design中明确，
不能冒充旧合同exact resume。

训练时：

- 及时记录per-task loss/gradient、norm/clip、update方向、task coexistence和wall；
- 不因弱指标改善自动延长；
- 到预注册checkpoint尽快跑paired400；
- absolute提高但lost/churn恶化时按能力换手处理，不写成稳定进步；
- 若趋势和内部路径不足，尽早停止，保留正式non-pass。

## 8. Official closed-loop evaluation

preprocessing固定：render256/model224、两相机180° rotate、state/action 7D、10 flow steps、执行前5 actions后
replan、dummy settling10、成功即终止、horizon 220/280/300/520。

strict controls：

- correct
- same-task-other teacher
- cross-suite-wrong
- shuffled
- reversed
- no-video

每个arm严格配对task/state、env seed、policy RNG、video ordinal和初始化。shuffle/reverse必须对真实输入frames
重排后完整forward。evaluator使用cost-balanced dynamic queue、long-first和persistent model/env；卡数只影响
吞吐，不改变request batch membership或科学输入。

至少报告：

- aggregate、per-suite、per-task和breadth；
- 与closest baseline的retained/gained/lost、net、churn、Jaccard和McNemar；
- gained/lost在哪些tasks/suites集中；
- correct与五个controls的同checkpoint差异；
- representation→compiler→effective BA→fixed-action传递。

80-row只作工程screen。正式checkpoint选择必须使用400 rows；不使用checkpoint union、平均、融合或挑task
checkpoint。严格`>150`后仍须补完整controls才能支持视频因果claim。

## 9. Numerical and verification policy

必须验证：

- shape、dtype contract与finite；
- no forbidden read；
- source freeze、no-video/step0 identity；
- request/video/task/state/RNG pairing；
- cache/manifest/completion与resume cursor；
- OOM、nonfinite、stale asset、cross-sample contamination；
- retained code的import/targeted tests和CLI parse。

不验证或不热路径门禁：

- 普通BF16/TF32最后几位；
- batch1相对batchN逐tensorbitwise；
- 大量SHA-256/MD5/content hash；
- 为漂亮数字重复forward或降吞吐；
- 与scientific decision无关的广泛防御性test harness。

## 10. Git, docs and lifecycle

- 一个canonical active implementation；新路线替换旧路线时删除旧executable/runtime/config/tests，历史由Git、
  `docs/research_history.md`和formal artifacts保存。
- 主分支保持clean；并发写用独立worktree，集成后删除临时worktree/branch。
- meaningful状态只更新`AGENTS.md`、handoff、execution brief、research history、task plan和findings，不再向
  数十个历史design和逐日ledger重复追加。
- 删除临时、profile checkpoint或worktree前核验进程、dirty状态、unique commits和consumer。
- 不提交datasets、checkpoints、cache、大binary、credentials或private host信息。

## 11. Current stop boundary

截至本brief，以下全部禁止启动：

- rank14 Gate C/cycle1/controls/fresh training；
- Reward-Credit cycle2或超参扫描；
- RLS、ECP、Tangent、Expert-Flow、old v*/K4/Expert-Manifold的resume；
- pivot15+1、mixed topology、few-shot或其它新候选的profile/training/rollout。

它们只能在新的单变量design authority明确授权后重新获得执行资格。任何旧artifact中的
`formal_ready`只描述历史时点，不构成当前授权。

PICK-GC的formal macro10与strict400已经完成并封存为138/breadth6/lost16 non-pass；其resume、controls、额外
checkpoint与参数sweep全部关闭。OSG-PC profile exit1且无checkpoint/mechanism report；同配置重跑、formal、
deployment和评测全部关闭。SKNC canonical实现、clean world3 `16/16` profile、B32 deployment smoke、formal
fresh`0→5`与strict400均已sealed；结果137/breadth7/lost13，config为`formal_result_sealed`，resume、controls和
sweep关闭。SRTP的`d172add`与`e31e2fd` world3 attempts均exit1，后者证明释放decoder graph仍不能让完整logical
B<=16 reward CFM gradient进入A40；config已封存为`profile_result_sealed_nonpass`，同配置重跑、deployment与formal
关闭。PCUG唯一world4 run在paired probe前因wall下界`2.25568x`封存non-pass，config为
`profile_result_sealed_nonpass`，同配置重跑、deployment和formal关闭。CVEG当前训练合同也已由K2 hard-guard
路径分叉判为稳定性non-pass；仅开放diagnostic macro5的一次directional strict400。任何后继GPU动作前必须有
新的单变量design authority、clean commit/push和detached frozen worktree，再复核gpu01/gpu02、quota、fresh
root、物理/NUMA topology与现有进程。有多少合适的同节点卡就用多少、最多6张，不等待凑卡。

NPCG formal macro5与strict400也已sealed：135/breadth5/lost17，config为`formal_result_sealed`；resume、controls、
point-count/threshold/seed sweep全部关闭。下一GPU动作必须来自新的单变量design authority，且直接针对
same-task action-hidden video neighborhood而不是恢复更多point guards。

2026-08-11 22:23+08:00资源快照选择`gpu02:0--5`的单节点world6/local4；`:6/:7`属于他人。GPU1只有历史已
纠正ECC/remap且当前无pending/failure。OSG-PC attempt结束后物理0--5均0MiB且volatile/uncorrectable ECC全0；
该空闲快照不是未来设备预约。

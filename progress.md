# EMBER Progress

更新时间：2026-08-21。本文只记录当前可执行状态；稳定目标见`docs/current_owner_requirements.md`，耐久结论见
`findings.md`，完整历史见`docs/research_history.md`。

## Current authority and executable state

- 外部专家A--G/F0--F5逐项复核goal已完成；113个编号claim均已实施、反驳或以有证据的
  `not-applicable` / `underdetermined-after-audit`收口，没有queued项。
- 第二轮专家意见后继goal已经由owner正式启动。active design为
  `docs/functional_adaptation_successor_design.md`；持久计划见`task_plan.md`，逐项覆盖见
  `docs/expert_second_round_implementation_ledger_20260819.md`；Phase 0的数据、架构与评测合同已冻结，default fold0的
  non-held fixed-decoder曾取得closed-loop `qualified_pass_to_writer_inference`；Writer macro10 formal、generation profile与
  七臂matched screen已完成。统一functional fingerprint已修复train/held坐标，但flow-only、fixed-probe与exact-BA
  Decoder均未过Gate 2；sealed validation8 rank16 oracle现已以`250/400`广泛通过，但8-row successful/on-policy
  action/JVP直接task-vector只过`2/4`与`1/4`。当前按专家原始建议转入多成功adapter、显式phase-aligned的role-disjoint
  manifold重构，未进入新Writer或outer RL。
- canonical workspace与集成目标为`/data1/user/ymdai/projects/EMBER`的`main`；开发只在从最新`main`创建的短期
  `codex/<topic>` worktree隔离，验证后的独立里程碑立即合并、推送并清理，不长期积压巨型分支。
- 来自clean pushed `7b6d768`的train24 fold0 fixed functional decoder formal评测已完成。修正投影wiring后的F4
  free-Program 1200行也已完成：projected=`307/1200`、direct=`658/1200`，253 retained、54 gained、405 lost、
  Jaccard `.35534`，只保留direct的`46.66%`，明确未过90%门。旧`659/1200`来自没有实际安装投影LoRA的错误评测，
  已撤销并重导出remote-safe证据。
- non-held meta expert bank已经按固定uniform-step合同完成71/71 tasks、每task 1000 steps；canonical输出仍为
  `runs/outputs/pi05_nonheld_meta_expert_bank_step1000_r6_650d922_gpu01p012457_20260819/`，后续direct、decoder与projection均
  复用这套昂贵产物，没有重训。相同固定rows上的direct experts为meta-train `2519/2800`、meta-validation `684/750`；
  相对source分别净增`+247`与`+38`，证明当前pool存在可裁决的policy-effective增量，而不是由高source identity自动过门。
- 同一15-task meta-validation面板的frozen source为`646/750`、direct为`684/750`、fixed-decoder projected为
  `659/750`。source→projected为612 retained、47 gained、34 lost、净`+13`、churn81，exact McNemar `p=.18208`；
  direct→projected保留621/684=`90.79%`的direct successes。15/15 tasks均有breadth@10，task73从source `4/50`
  提升到projected `15/50`，但增量不显著、6 tasks正/6负，且只复现direct gain rows的54.67%，不能宣称decoder方法通过。
- 其余56个meta-train tasks的source/direct/projected分别为`2272/2800`、`2519/2800`、`2451/2800`。
  source→projected为2157 retained、294 gained、115 lost、净`+179`、churn409，projected保留direct successes的
  92.62%；34 tasks正、11负、11持平。Study净`+99`、pick-place净`+102`，说明改善不是只靠高base aggregate；结合完整
  71-task source `2918/3550`，当前不触发无差别source重训。source覆盖摘要见
  `docs/evidence/functional_adaptation_20260819/nonheld_meta_source_coverage_71.json`。
- default fold0 functional decoder formal从fit loss `1.040443`降至`.481957`、held loss从`1.035567`降至`.830093`，
  14/15 held tasks改善；冻结后导出71套完整LoRA并完成上述严格paired closed loop。当时Gate 2据此只作
  `qualified_pass_to_writer_inference`并启动macro10；后续坐标审计已撤销其Writer泛化authority，因为train/held codes
  分布不一致。该结果只继续支持decoder range值得重验，不能升级为fixed-coordinate科学结论。完整原裁决见
  `docs/evidence/functional_adaptation_20260819/nonheld_meta_fixed_decoder_fold0_20260820.json`。
- successor Writer formal已经从clean pushed detached `main@a2999d5`完成56-task、6-GPU、macro1--10，耗时
  780.43秒。total objective由`2.082406→1.853265`，combined/language/video分别由
  `.995820/.994485/.984880→.838643/.861964/.974670`，10行metrics与gradient均finite；峰值32.60 GB。
  Text/VL Meta-LoRA均为0，fixed decoder与VLM保持冻结，唯一macro10 checkpoint含Writer、trainer与world-size6状态。
- 同一macro10 checkpoint已在一张空闲A40完成真实video→完整LoRA的B8/B16/B32 profile；三者均稳定、零OOM，吞吐分别
  `.898928/.897234/.896828 LoRA/s`，B8按预注册规则胜出，最长48 sampled frames，最小显存余量35.50 GB。active config
  已封存该证据并只允许profile支持的batch；该昂贵checkpoint与cache继续保留作反事实，不重复训练。
- macro10的15-task×10-state matched screen为correct/language-only/video-only/first+final/reversed/shuffled/static分别
  `131/130/134/130/134/133/132`。correct相对source为123 retained、8 gained、8 lost、净0；相对video-only与reversed均
  只有1 gained、4 lost。所有arms breadth@1均15，但correct没有full-video、order或dynamic Value优势，Gate 3明确失败。
- 最早失效接口不是简单“macro10还没训够”：56个训练code的coordinate std约1、平均task norm`5.589`，15个held codes
  从零自由拟合后std仅`.089`、平均norm`.505`，最近train code平均距离`4.523`。Writer训练与leave-out评测不在同一个
  code分布。effective-BA诊断进一步显示video-only的task-mean只有1/15最近邻到正确projected task，却距15-task共享均值
  仅`.000998` relative error；其134分主要是shared carrier/base competence，不是video-conditioned task inference。
- 统一fingerprint formal已复用现有71-task experts完成：feature width `25600`，train-only 32维PCA解释方差`.887516`；
  train/held coordinate std为`1.0000/.7248`、平均norm为`5.5701/4.1436`。held只做同一固定变换且不产生梯度，旧版
  `.505` near-zero held code问题已解决。这套昂贵fingerprint artifact继续复用，不重复收集。
- 固定上述codes的flow-only Decoder将fit/held flow loss由`1.040443→.445721`和`1.035567→.664218`，但同一750 rows
  closed loop仅`644/750`，低于source `646`、direct `684`和旧projected `659`。相对source净`-2`，相对direct净`-40`
  且`p=3.67e-5`；task73仍为`4/50`。因此没有启动56-task复评或Writer。
- flow-only生成effective `BA`相对direct的relative-L2 `2.8576`、cosine `.0254`、norm ratio `2.7004`，说明有限flow
  queries允许巨大近正交off-manifold解；该objective已关闭。
- shared-zero carrier formal已完成`640/750`，相对source净`-6`；shared-zero→task fingerprint只有30 gained、26 lost、
  净`+4`、`p=.68888`。上一轮`644`几乎由共享输出解释，task-specific code无可靠增量；该carrier只作诊断，不作为fallback。
- 固定8-probe effective Decoder从clean pushed `c3e5bc1`完成1120 steps，7分46秒、峰值18.92 GB；完整effective `BA`
  的train/held relative-L2为`1.1387/1.1292`、cosine仅`.0642/.0449`，按固定方向过拟合关闭。
- exact低秩Gram Decoder从clean pushed `423a9b2`完成1120 steps，324秒、峰值18.92 GB；train/held BA geometry改善到
  relative-L2 `.8423/.9591`、cosine`.5365/.3032`，但同一held750 rows只有`638`，低于source646、direct684、旧
  projected659、flow-only644与shared-zero640。相对source净`-8`、相对direct净`-46`且`p=1.96e-5`；held loss在
  step280--1120约`.926→.921`平台化，不续训。
- 无训练仿射full-BA诊断的train/held relative-L2为`.5244/.9797`、cosine`.8439/.3648`，设计condition number`1.009`；
  排除“只需更小线性decoder或canonical factor”的窄解释。当前最早失效接口上移到单expert功能标签与source/meta任务角色。
- validation8 sealed task-local oracle合同已完成：八套独立rank16 LoRA统一训练到预注册step2000，只在step1000
  exact-resume，不更新共享Writer/decoder、不选checkpoint、不读取Test。clean detached `5fd224a`上的strict400为
  `250/400`，既有frozen source为`48/400`；严格配对43 retained、207 gained、5 lost、145 retained failures，净`+202`、
  churn212，McNemar exact `p≈1.06e-54`。八项全为正净增量，breadth@1/@5/@10均8，四suite分别为Spatial `73`、
  Object `78`、Goal `58`、Long `41`，预注册强门明确返回`advance_to_successful_on_policy_manifold_panel`。remote-safe
  400-row与stage证据见`docs/evidence/functional_adaptation_20260819/validation8_task_local_oracle_step2000_20260821.json`。
- oracle stage trace显示Long的剩余问题主要是多阶段完成而非完全无primitive：Long task1中cream-cheese ever/final为
  `31/27`、butter为`13/13`、full peak与最终成功均`12/50`，四行在完成第一对象后又丢失；Long task2的stove-on为
  `50/50`、moka-on-stove与最终成功均`29/50`。Goal task3的BDDL只有最终谓词，不能观测开drawer中间阶段。该trace是
  无序final-goal代理，不是完整有序procedure标签。
- source/meta角色分离的19/5 formal诊断已经完成。相对旧重叠任务面的无训练仿射held cosine `.3648`，role-disjoint
  flow/action单标签分别为`.4310/.4341`，说明改变数据角色后几何有方向性改善；但任务集合也改变，不能把差值当成纯因果
  归因。两种16维train-only-whitened code的held std均约`.78`，不再出现near-zero held坐标。
- 同一轨迹的成功阈值多checkpoint等价集只把flow/action aggregate held cosine提高到`.4355/.4394`，证明“多checkpoint”
  本身没有解决标签欠识别，也不能冒充独立成功策略分布。关键分层是：有checkpoint达到`25/50`的held tasks 0/9/18上，
  action-response达到cosine `.5942`、relative-L2 `.8394`；没有成功checkpoint的tasks 25/36只有`.2071/1.0766`。
  这把下一裁决集中到task-local ceiling与成功occupancy，而不是继续换Decoder objective。
- denoised action-response已按共享显式noise、official 10-step integration与完整`50×7`action chunk完成formal收集；相同
  anchors的flow为`50×32`。action在有成功expert的held子集优于flow的`.5753/.8712`，但全5-task aggregate仍未过
  预注册几何screen；两者都只是无闭环、未物化adapter的定位证据。remote-safe摘要见
  `docs/evidence/functional_adaptation_20260819/role_disjoint_manifold_20260821.json`。
- 精确policy-JVP机制smoke已在一项成功train expert上接通：完整`50×32`action-sequence JVP finite，峰值`17.44 GiB`，
  expert-source JVP差RMS `.028757`、cosine `.999663`。它只证明专家建议的Jacobian标签可实际计算；在validation8
  ceiling裁决前不扩为全量fingerprint或Decoder objective。证据见
  `docs/evidence/functional_adaptation_20260819/policy_jvp_feasibility_20260821.json`。
- formal-validation-only BDDL stage capture已接通：同一rollout内记录goal predicates的初值与change points、ever/final及
  peak count；真实LIBERO wrapper与两谓词任务解析通过，聚焦测试3项通过。它使用privileged simulator state但不读
  teacher action/reward、不产生梯度或改变success，且只作为无序最终合取的阶段代理。step2000 strict400将同时收集，
  不额外重跑rollout。证据见
  `docs/evidence/functional_adaptation_20260819/stage_predicate_capture_smoke_20260821.json`。
- successful/on-policy panel已从clean detached `febdff0`完成：四个non-held tasks的8/8预注册direct rows全部成功，
  没有替换；task23两条为208/233步、task26为178/105、task80为107/120、task86为135/139。八条trajectory sidecars共
  约372 MiB，动作、RNG和BDDL stage同轮保存，未读取held或重训expert。
- clean detached `1e45c66`在四张A40上对每轨迹8个progress strata重新配对source/expert：denoised action只有task23/86
  通过full+early same-task mutual-cosine-nearest门，即`2/4`；task26的gained trajectory最近是task86，task80的retained
  trajectory最近也为task86。exact JVP只有task80通过，即`1/4`，按预注册规则不能覆盖action失败。全部early states和
  事实上全部64个selected states都尚未完成BDDL goal conjunction，因此不是final-predicate捷径造成的假失败。
  该结果淘汰直接concatenation，不否定phase-aligned action family。证据见
  `docs/evidence/functional_adaptation_20260819/successful_onpolicy_response_panel_20260821.json`。
- 回查专家原始意见后，当前最早接口进一步具体化为`successful adapter equivalence + occupancy/phase alignment`：下一面
  复用target train24现有step250--2000 checkpoints与formal rows，每task取最早/最晚成功checkpoint各自一条最短成功轨迹；
  23 tasks可形成K2，唯一只有step1000一次成功的task形成K1。fit19学习单调phase alignment与固定坐标，held5只做固定
  变换；JVP不再是primary label，aligned representation过门前不重建decoder。
- 该面已在clean detached `545b43c/7258487`完成：四个checkpoint capture共47/47条预注册成功轨迹，无替换、无expert
  重训；完整每-replan `50x7` action delta经fit19-only、task/member/state等权PCA/whitening后，32维解释方差`.923430`。
  held5的等时间与功能弧长表示均为`5/5`同task mutual-nearest，功能弧长在`4/5`任务提高same-task cosine，按预注册门
  `advance_to_phase_aligned_fixed_decoder`。fit19的mutual-nearest从等时间`15/18`变为弧长`14/18`，因此当前结论是组合
  标签已具备leave-task-out可识别性，不是弧长在所有面板单调占优。证据见
  `docs/evidence/functional_adaptation_20260819/train24_successful_equivalence_phase_20260821.json`。
- fresh Decoder合同已在任何优化前写入`configs/pi05_train24_phase_aligned_decoder_v1.json`：task consensus PCA16只拟合
  fit19，多个成功成员各自的8个真实phase states轮换提供完整flow监督；held5 earliest/latest member code均零步优化、分别
  物化完整LoRA。训练固定5-rank、950 task visits/190 optimizer updates；最终functional门只作安全诊断，是否进入新Writer
  由预注册held5两套strict250闭环联合门决定。新入口`train_phase_aligned_functional_decoder.py`是本轮唯一active Decoder
  candidate；旧trainer只维持sealed历史复现，若本轮闭环通过则退役旧入口，若失败则删除新candidate并保留失败证据。
  实现所有权按单一流水线拆分：`phase_code_building`只构造冻结坐标，`phase_decoder_codes`只校验code authority，
  `phase_decoder_panels`只绑定成功轨迹监督，`phase_decoder_training`只负责分布式优化/精确恢复，
  `phase_decoder_projection`只物化两套评测bank；两个`scripts/`文件均为薄入口，不构成平行算法实现。
- `main`上的已封存Writer仍是Core-Addressed Reader主架构：Dynamic-K、rank16、38 targets、Action Meta-LoRA、
  layer/rank memory、Reader、K-set、bounded M2P和FactorHeads；原生language保留，Text/VL Meta-LoRA已从
  canonical config/code contract移除。该实现只作为sealed baseline和可复用组件来源，不再作为后继增量路线。
- 不直接返回V6/LPCP/GOMQ，也不恢复旧Expert-Manifold为held dictionary；历史实现只提供paired反事实、functional
  probe、checkpoint/evaluation等可审计复用候选。

## Latest owner decisions for successor planning

- 允许train24 privileged experts训练共享functional decoder；也允许使用LIBERO-90中经审计、排除固定validation/test
  tasks及其重复项的non-held任务，必须保留显式allowlist与provenance；
- 允许learned language-only adapter作为baseline，用于裁决video条件增量；
- 允许在授权train/meta tasks上用simulator reward训练共享Writer/functional code inference。该outer RL仍以held
  zero-interaction LoRA为部署对象，不等于生成LoRA后的task-local RL；
- 允许冻结模型、无梯度、无checkpoint选择的sealed held action/reward诊断；Test默认留到最终方法冻结后；
- 合理的新架构均可考虑，包括rollout前合并为唯一完整LoRA的shared prior/base adapter + video-conditioned residual；
  不允许部署第二adapter、expert route、task-ID字典或checkpoint融合；
- 主写与集成目标改为`main`；需要隔离时从最新`main`创建`codex/<topic>`分支/worktree，验证后及时合并并推送。

## Active successor phase

当前核心顺序为：

1. 审计现有expert manifold、Writer、reward/evaluation与LIBERO数据owner，建立non-held meta allowlist、task-level folds、
   process controls和source/task-expert ceiling协议；
2. 用policy-functional response而非raw A/B几何学习compact code与固定complete-LoRA decoder，并以leave-task-out
   closed loop作为进入门；
3. 固定decoder后学习language prior + action-hidden video process posterior，保留完整Action probe与有向阶段结构；
4. functional warm-start后在train/meta simulator接入closed-loop outer credit；
5. 用strict paired400、相邻checkpoint、same-task不同视频、Long、breadth和多split复现选择或停止方法。

专家方向A--N和五个替代研究问题都已进入ledger。runtime video policy、task-local RL、richer sensing以及
video-to-reward/skill/plan不是被丢弃，而是在核心single-LoRA路线触发预注册stop gate后按证据启动；train/meta action
alignment、mergeable base+residual与sealed diagnostics已经获准进入对应phase。

已完成的Phase 0实现：

- `configs/libero90_nonheld_meta_v1/protocol.json`显式保留71个去重non-held tasks、排除19个target-overlap tasks，并建立
  5个不读取结果的task-level folds；默认56 meta-train / 15 meta-validation，冻结后轮换复现；
- `ember.functional_adaptation.contract`加载allowlist/folds并验证source manifest与语义overlap audit一致；
- strict video conditions已增加first-only、final-only、first+final、endpoints-fixed-middle-shuffled与monotone-sparse，
  真实frames经选择/重排后重新完整forward；
- 新模块owner与旧`expert_manifold`/Writer/evaluator的复用、退役边界已写入active design；旧bank route不恢复。
- `FunctionalCodebook`与`FunctionalAdapterDecoder`已经建立32维whitened task code到全部38-target/76-tensor LoRA的
  单一生成面；decoder以functional identity初始化，Action in/out保持独立，不import旧V6 bank route；
- policy-functional probe会捕获完整`[batch, 50, 32]` Action Expert flow response，并以expert相对identity的响应能量
  归一化监督，避免source policy的大幅公共响应淹没task adapter信号；首轮相关20项CPU测试通过。
- non-held meta expert合同已固定71 tasks中的56 meta-train / 15 meta-validation-oracle，并复用唯一task-expert训练owner；
  fixed decoder也已能从该bank按角色拟合/冻结和导出32维code，不建立task-ID deployment route。
- 后继Writer运行面已实现为`language prior z_L + ordered-video posterior delta(L,V) -> frozen decoder -> one complete LoRA`：
  每条视频独立保序编码initial/goal/event/transition，跨K只聚合完整video program；保留50个Action probe并加入仅训练期
  meta-action phase alignment，同时提供真正不读language/action probe的video-only baseline。模块按decoder、inference、
  schedule/step/checkpoint和privileged-action owner拆分；旧LMMPC继续只作为sealed历史基线，不形成并行active fallback。
- successor已经接入现有唯一PI0.5 evaluator、episode LoRA cache、persistent rollout worker和online generation profiler；支持
  fixed 56-task meta-train / 15-task architecture-validation角色及correct、same-task-other、wrong、language-only、video-only与
  真实帧时序controls。learned language-only部署分支不打开视频，video-only分支不读取language或Action probes；训练期
  teacher-action alignment改为同task但确定性不同episode，并按归一化过程相位配对，避免逐帧动作复制。
- process panel新增两个诚实条件：把真实首帧重复到原长度并保留source-time positions的`static_first_repeated`，以及读取
  同一episode同一时刻`eye_in_hand_rgb`的`eye_in_hand_view`。HDF5只含双路RGB且没有depth/segmentation；robot/object mask
  不会通过teacher state重渲染伪造，可信RGB-only flow仍登记为未解决数据/表示缺口。
- 当前代码里程碑的67项定向evaluator/cache/runtime测试、honest baseline分支smoke和统一cache dispatch smoke均通过；结构门
  无hard violation。该证据只说明运行面接通，不是Writer性能或fixed-decoder gate通过。
- 原两个decoder profile入口已收敛为唯一`train_functional_adapter_decoder.py`：直接优化完整PI0.5 flow response，保存
  task-equal phase cursor、system/held codes、optimizer与Python/NumPy/Torch RNG，可从阶段checkpoint精确续训；56/15 formal
  schedule及下游formal-authority门已经冻结。它只补齐正式训练责任，不声称non-held decoder结果已经通过。

当前train24非正式机制profile（不是模型选择或closed-loop证据）：

- 结果无关fold0以19 tasks拟合decoder、5 tasks冻结decoder后只拟合新code；五折将轮换，19/5不是永久丢弃任务；
- gauge-invariant `BA·probe`预热在380/250步把fit mean从`1.000`降到`0.447`、held code mean降到`0.805`，但其
  PI0.5完整flow初始loss仍约`0.999/1.008`，证明effective-update几何不能替代policy-functional目标；
- 完整50-token flow短profile仅给每个fit task 2步、held task 5步，独立demo40--49评测从`0.999→0.833`和
  `1.008→0.933`；18/19 fit与4/5 held优于identity，仍各有一个退化task，因此只支持“链路有可学习信号”，尚不通过
  fixed-decoder realizability gate；
- A40单卡峰值18.81 GB，38+25个实际更新约22秒，主要固定成本是policy加载与成对probe缓存。下一节点应扩大独立panel和
  task-equal更新次数，而不是扫rank、scale、seed或dtype。

当前train24 fold0 formal closed-loop结果：

- fixed decoder单checkpoint为`388/1200`，direct task experts为`658/1200`；严格配对是332 retained、56 gained、
  326 lost，Jaccard `.46499`；
- 19个decoder-fit tasks为`326/950`，对应direct `550/950`；5个decoder-held tasks为`62/250`，对应direct
  `108/250`。fit与held都只保留约六成expert aggregate，不是只在held code拟合处失效；
- 因此train24版明确不通过functional realizability gate，内部flow loss下降不能替代该结论。下一步不是扫小超参，
  而是按已冻结合同训练56/15 non-held meta expert family，再重新拟合和裁决固定decoder。

## Final external-review result

| arm | macro25 | macro50 | 25→50 retained/gained/lost | breadth@1 |
| --- | ---: | ---: | ---: | ---: |
| A Text+detach | 123 | 84 | 71 / 13 / 52 | 8→5 |
| B noText+detach | 104 | — | — | 6 |
| C noText+credit | 110 | 101 | 77 / 24 / 33 | 6→4 |
| F5 C+PCGrad | 107 | 96 | 82 / 14 / 25 | 6→4 |
| F3 A+frozen heads | 123 | 117 | 90 / 27 / 33 | 8→6 |

完整macro25视频面板（correct / same / wrong / shuffle / keep-first / reverse / no-video）：

- A：`123 / 125 / 81 / 122 / 131 / 90 / 48`；
- B：`104 / 101 / 65 / 83 / 90 / 96 / 47`；
- C：`110 / 111 / 54 / 91 / 93 / 69 / 47`；
- F5：`107 / 111 / 51 / 92 / 105 / 53 / 47`。

三个no-Text arm均显著优于no-video和wrong，说明Writer确实使用视频，不是language-only。C是唯一在aggregate上
同时显著优于wrong/shuffle/keep-first/reverse/no-video的arm，但收益高度集中Object、Long reverse反向，且
same-task correct-success retention只有87.27%。因此视频因果资格得到部分改善，方法未达absolute、
稳定、same-video robustness和跨suite高层Program的联合目标。

## Root-cause adjudication

1. **Fresh front-end detach是真实工程缺陷。** A/B在macro1/25的`patch_grounding`/
   `interaction_projection`均无gradient；C修macro1首次有credit。修复将correct-reverse margin从8提到41，
   但correct只104→110且继续漂移，所以它是视频方向资格的一个前端因素，不是absolute/stability首因。
2. **Text Meta-LoRA提供真实但混合的support。** 移除它使correct掉19，同时shuffle/keep-first各掉39/41、
   reverse反而升6；这不是纯language shortcut，也不是科学上干净的正机制。owner的no-Text边界继续有效。
3. **简单self-occupancy divergence未获支持。** lost rows没有出现预期的macro50-self-occupancy disagreement增大；
   validation expert不存在且held teacher action受信息墙禁止，动作正确性只能记为审计后不可判。
4. **FactorHead co-drift和reachability都是实证问题。** freeze使84升到117但仍丢33；修正wiring后的fixed-head
   free-Program仅307/1200，对照direct experts 658/1200，253 retained / 54 gained / 405 lost，未过90%门。旧659
   是未安装投影LoRA的无效结果。后继fixed functional decoder正面检验稳定功能坐标；若nonheld held-task gate仍失败，
   必须考虑架构性扩大或重参数化，不能以小扫掩盖。
5. **Cross-task conflict会改变换手，standard PCGrad不是解法。** 它将lost 33→25、churn 57→39，但gained
   24→14且有显著抑制，score更低、breadth仍收缩，并把keep-first margin压到2。Adam moment独立效应仍不可由本arm裁决。

当前最早未解接口被收窄为：固定输出坐标能否覆盖policy-effective directions、四条信息流能否为未见task预测这些
directions，以及shared objective/更新能否在同一checkpoint保留它们。本轮没有性能pass；当前登记的后继架构用
functional fingerprints + fixed decoder把前两项拆成独立gate。

## Remote-visible review map

- 原专家报告：`docs/external_review_20260818.md`；
- 113项claim ledger：`docs/external_review_claim_ledger_20260818.md`；
- 本轮面向专家的结果报告：`docs/external_review_followup_20260819.md`；
- 给新session复制的独立复核prompt：`docs/external_review_followup_prompt_20260819.md`；
- 证据索引与全部remote-safe JSON：`docs/evidence/external_review_20260818/README.md`；
- 持久结论与历史：`findings.md`、`docs/research_history.md`。

## Verification and cleanup

- 最近完整回归仍为`293 passed`；fixed-decoder正式训练入口另有8项聚焦测试通过；
- B/C/F5各7个视频面板均为400 rows，pairing mismatch全为0；全部tracked/forced evidence JSON可解析；
- 本轮只运行必要的聚焦回归：autocast-safe confidence objective、successor authority配置与detached frozen authority
  各1项通过；Writer profile成功后
  未重复大规模训练。已完成的direct/projected evaluator均退出且无遗留`ymdai` GPU进程；正式证据、唯一expert bank、
  decoder/projection与成功profile保留，失败profile临时目录已删除。
- projected formal实际使用commit `247e6a8`的SQLite `DELETE` journal；继承run contract中的旧`WAL`描述只是标签滞后，
  不改变rows、pairing或adapter。active evaluation config已更正为`sqlite_delete_full_sync_atomic_claim`，证据中显式记录
  该provenance。
- validation8 strict400、8-row occupancy capture及四task action/JVP分析的worker均已exit0，GPU显存已释放；对应三个
  detached formal worktree均在证据落盘后删除，当前`git worktree list`只保留canonical `main`。372 MiB成功trajectory是
  唯一phase follow-up输入而保留，不重复rollout；临时selection与旧冻结worktree未残留。

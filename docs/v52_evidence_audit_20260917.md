# v5.2 至当前 A 的证据审计

2026-09-17。对应[架构推导](v52_evidence_based_writer_design.md)。本次只读源码、冻结合同、结果、历史评审和 train24 specification/RGB，
没有新模型 forward、训练、rollout、held-action 诊断或 Test 使用。

## 1. 审计口径

“充分／不足”须带具体命题。一个完整400行结果足以说明该 checkpoint 的能力，不等于该架构已训练收敛；
一个 privileged 正控可以证明局部存在可用更新，不等于合法 RGB 已能生成它。本文区分：

- **行为事实**：实际闭环及逐行配对，注明400/250/150/96/80等面板，不跨面板混排。
- **局部干预**：相同起点／支持集下改变一个明确接口，仍记录同时改变的因素与求解预算。
- **功能／表示证据**：FM、recovery、动作预测、动态读出；不代替行为。
- **结构性质**：由实际代码、mask、线性代数得到的可表达性／不可能路径，不冒充学习结果。
- **尚未执行**：只有设计、profile、训练而没有闭环等，明确不能归入科学失败。

条件 C 指一次 task-video 条件调用；queries Q 指监督调用，不保证都是独立样本。
K>1另计 video appearances。step、C、Q、每task曝光、有效上游更新次数和优化时钟不能互相替代。
不同 source、动作时间对齐、teacher/query池、K、帧预算、task权重和初始化都影响可比性。

当前 A 的按节点视频诊断由 owner 明确授权用于问题分析。旧 selected／sealed 的视频 controls 和 Test 只保留其冻结方法的事实，
不据其分数选择本次结构、loss、层位置或 checkpoint。旧正例本身不能被忽略，也不能反向优化到这些面板。

### 1.1 两个需要纠正的历史口径

1. Layered192/384 的文件名虽称 strict400，correct 实际仅255个不同 task-video 条件、每task28–36条视频，other259；
   其69→67／72→64是旧固定分布内事实，不满足当前每task50视频各一次。早期混合K64 Horizon的99/95也已撤销该资格。
2. **不能由日期推断全部早期结果重复视频。** 旧 v5.2 原始900合同已明确 seed7、每task50条无放回，合计400条件；
   其132和9月15日同映射重放125均与新 A seed20260911只有8/400相同state-video组合。
   因此旧新单点差额不是同一paired实验。参考
   [比较边界](../runs/analysis/source_alignment_20260915/archival_A_B_comparison_boundary.json)与
   [原始／重放分别比较](../runs/analysis/source_alignment_20260915/A/node900_reference_comparisons.json)。

### 1.2 原件入口

当前[research history](research_history.md#9-按问题恢复原件不重读所有历史)索引了完整旧账本
`fcdb6e43706c5fcedf10eaa5d2d459602b263016:docs/research_history.md`（181节），更早基线见
`ac233fa0:docs/research_history.md`。删除旧活动代码／设计不是抹除证据，精确版本按 Git 对象读取。
三组只读审计分别覆盖早期 v5.2/后继、native/bank/PQ/纠正场、Horizon/视频功能路线；主审统一任务、数学、比较语义和设计。
下面按科学命题聚合小修订；不把每个只调整阈值、求解器或监控的分支写成一个独立方法。

## 2. v5.2 及早期后继

| ID／方法 | 真实机制与训练 | 事实、充分程度和限制 |
| --- | --- | --- |
| E01 旧 v5.2 | task语言查询真实patch＋native task-span组成Core；H50均值→两层P；centered-P/AdaLN；最终slot norm；8共享完整A/B heads。三Meta fresh joint，纯FM | 900＝3600C/75600Q，132/400；完整登记100/500/700/900/1200/1400/1600/1800＝72/79/120/132/92/106/114/102。**实际续到1800**；不能称900早停或已稳定通过 |
| E02 v5.2 task-complete | 主图和参数量不变，每次聚合全部24task，改变Adam次数和LR阶段 | 150＝3600C/72000Q，51；200/350/400＝91/106/120。400＝9600C/192000Q，尾段仍升；不是确定平台 |
| E03 v6 old／TC | mean-backed Core加中心化选择残差；P前加入相邻grounded visual transition；head hidden216→256，10.775M参数 | old100/500/700/900＝98/121/76/95，900为3600C/72000Q。TC50至600每50一步＝106/64/111/133/132/117/138/143/131/130/132/126，400为9600C/192000Q。既不能以100步64早判，也不能只记400步143 |
| E04 Target-Owned | target独立读Core和A/E/D interval，拼为1024维；76个独立head，取消原centered-P AdaLN和最终slot norm；47.858M，其中head40.518M | 50/100/150/200＝99/76/86/68；终点4800C/96000Q。同曝光v6-TC200为133，但前端、head共享、归一化和容量同时变，不能孤立归因 |
| E05 初版Dynamic-K/BackboneMemory | 每帧8个one-way native memory贯穿18层；仅delta/end−start值→外部时序/集合/mixer→fixed-A dynamic-B；仅Action Meta；rank8、总64帧 | 50＝1200C/24000Q/3000 video appearances，100/400。FM仍降、LR尚高，短预算；损失为FM＋.05 consistency，**不是纯FM** |
| E06 Dynamic-K变体 | SemanticAddress增加语义Q，DirectFamilyB简化解码，VisualValue仍主要变化量，FullFactor开放A/B；仍未恢复v5.2全部内容/训练条件 | SemanticAddress101；DirectB K1/K4 102/98；VisualValue50/100/150/200 88/86/86/96；FullFactor50为91，对matched fixed-A88的R/G/L70/21/18。仅24000Q不能否定fullAB长期能力 |
| E07 Multi-K SlotSet/SharedCore | 冻结v6-fast400，仅新增197120参数；前者末端集合融合，后者多视频Core union后读P | 各25＝600C/12000Q，130/139。K1恒等不给新增模块梯度，有效新增模块曝光最多450C/9000Q。不是fresh全模型比较 |
| E08 LPCP | 从另一个OrderedProcedure CommonValue AS139起步；新增1.518M逐层50probe→rank-read→delta/controller，只改P查询；旧10.972M冻结 | 25＝600C/12000Q，143，对139 R/G/L120/23/19。BA cosine约.999995仍42次交换。不要把AS139与E07 SharedCore139当同一checkpoint |
| E09 GOMQ | 冻结LPCP，新增2.829M learned one-way queries→native-B residual；专家occupancy的executed-prefix residual distillation | 只有4次Adam，memory仅3次有效更新；280 K4条件、1120 appearances、3456 query chunks。cycle2/3/4 151/135/131，cycle2比matched fixed-query135多16。局部正效应成立，充分训练／稳定资格不成立 |

E01实际参数：Writer10,237,704；8组 heads1,838,592。每个family head为256→216→native width，
320个slot涵盖18层×16及in/out各16；每层q/v共用slot但有不同head。初始化A模板非零、B为零，head末层及调制为零。

**配方交互是强证据。** 同每task150次访问，v5.2 old900/TC150为132/51，v6为95/111；
4→24 task/update、900→150次Adam、LR阶段与flow RNG时钟共同变化，说明配方能显著改变结果，不能从中选出唯一原因。
Adam作用于聚合梯度的非线性矩统计，等条件数不等于等更新过程。

**GOMQ的rank纠正。** 名义rank32为A=[A0;A0]、B=[B0,ΔB]，实为BA=(B0+ΔB)A0，rank≤16；
151→136的rank16重物化差额不能解释为丢失了32→16的有效容量。

主要原件：

- [v5.2/v6匹配曝光](../runs/outputs/pi05_as_writer_v52_v6_recipe_matched_exposure_seed7_20260801/analysis.json)。
- [Target-Owned合同](../runs/outputs/pi05_as_writer_target_owned_factor_bci_rawfull24_decay400_formal_r6_b20_micro2_seed7_formalvideo20260722_34be4a0_20260804T051244Z/run_contract.json)。
- [Dynamic-K合同](../runs/outputs/pi05_dynamic_k_backbone_memory_rank8_budget64_formal_fresh0to50_r6_b20_micro8_5319022_20260813/run_contract.json)。
- [LPCP合同](../runs/outputs/pi05_v6_layerwise_probe_conditioned_procedure_formal_fresh0to25_r6_b20_515f91e_gpu01_20260814/run_contract.json)。
- [GOMQ合同](../runs/outputs/pi05_v6_lpcp_cfmg_gomq_formal_fresh_cycle0to2_r6_k4_views4_b8_8553b61_gpu02p123467_20260817/run_contract.json)。
- 代码：`529da6b`的`writer/{model,temporal,video_program}.py`；`5319022`的`backbone_memory.py:339–362,449–451`；
  `aecbce5`的`backbone_memory.py:430–482,528–564`及`model.py:464–529`；`8553b61:model.py:382–441`。

## 3. native空间、表示到编译、P/Q

本节许多早期闭环是五task/250行，不是validation8/400。privileged teacher只证明训练侧存在性。

| ID／方法 | 真实机制与预算 | 事实、充分程度和限制 |
| --- | --- | --- |
| E10 固定A/privileged realization | 已知成功task-local更新投影；raw-factor短solver；centered two-sided坐标 | 三个成功成员投影后49/41/35 /250，Goal/Long归零；two-sided80仍失去两类。具体坐标损伤行为成立；12-step自由solver49不是自由A/B上界 |
| E11 G1 Native-Factor | native X/Y signed pooling，carrier12/mobile4；task-local free-logits500；解析初始化、q-head与action-in分组；observer/source冻结 | scalar Y-span将120→109 /250，Goal11→0、Long8→0；随机500为88，q-head500为84；最终解析114、breadth5、Goal2/Long1。最后正例依赖privileged成员与求解，不是合法共享生成 |
| E12 G2 Natural Program | action/progress辅助；修正静态旁路、observer、时间残差、cadence、多视频alignment；fresh100后resume200，约760 task-presentations、K1/2/4 | 200 full/endpoints .28167/.36207，改善22.2047%，probe38/40、active events4。动态信息可读；没有生成LoRA或闭环证据。旧macro10仅10次Adam |
| E13 G3/F1–F3 | 冻结G2/source/Stage0，仅训compiler/scorer；v1 95/190、v2 95、F2 25、F3 25→50 | v1 full35→38 /250，v2 41、carrier43；F1解析cos≈.9998只证容量；F3 held recovery .048→.090仍增。不能称50步已排除共享学习 |
| E14 Primal/J2/routing | P1自由primal；J2复制G2部件、fresh优化器，共同训Program/scorer110更新，约660任务曝光/1320视频LoRA | P1跨视频回放.972/.954不是行为；J2 train/held .171/.165、task-held .123/−.109；固定正交task-token/chart R5 .940/.963，真实Program R6 .165/.143，功能信用R10 .560/.544。坐标可解不等于真实视频可获取 |
| E15 PNBTT | bank learned keys、Cholesky白化、真实native signed replay；**只跑E1 task-local free query＋shared keys，E2 Natural Program未跑** | gate-aligned70→110：task1 recovery .542→.562、task93 .677→.686，仍在学。旧margin .10相对裁决.50提前停梯度，已fresh修正。不能写成真实Program共享链已被否定 |
| E16 Process/Composer/Axial/Unified | 多为carrier12＋native signed residual4；全H、frame-local、native-temporal与统一block逐步修订；observer冻结、组件初始化 | Process约35–44 /250；Axial73的50仅300 task-exposures；NativeTemporal100→200 42→35；Unified parallel40→38、common-base35→31、source-separated25→50 45→40。相邻弱行为成立，不是所有统一块的终局否定 |
| E17 P/Q restricted→complete | 四task、两fit视频、64更新、每task/update8queries；四层P/Q、family共享全AB，去carrier/native-span/static-zero；4.750M全部可训，observer/source冻结 | full32→64 fit/held 50/54→64/62 /150；restricted38/37→41/39，A2 43/41→44/45。出口组合正效应，256C/2048Q；同时解除多个限制，未测validation400 |
| E18 P/Q meta73/target18 | 各128更新，每次全任务；meta73 9344C/74752Q，target18 2304C/18432Q；每task128次、每视频64次 | screen80 meta73 15/19/19/19，target18 17/17/20/16；终点三训练task fit/held32/36 vs53/53 /150，18task held180为42/55。任务数与task权重、更新组成同时变 |
| E19 P/Q clone/random/width | 两弱task clone128、各1024Q；random取消G2组件复制；width256四层扩大至15.66M | shared3/20、clone14/20；shared早期2/3/4/3，未出现强后遗忘。random screen16/16/17/19。**width256训练与功能面板完成，但未物化／闭环，不能记为失败** |

由真实线性层 y=Wx+b 的 scalar signed pooling 得到的 Y 值处于span(W,b)；q无bias时原生输入1024、输出2048，
因此该限制不是“更native”这个词能消除的。按head分组可以扩大为不同子空间的直和；E11行为投影是具体限制有害的证据。
同时 E17/E18 已有真实视频、full H、四层共享主干、完整A/B和功能训练，足以反驳“这些一般属性就是v5.2优势原因”。

原件：旧账本§7、10–29、43–63、91–103、109–181；
`2050de9e:src/ember/ecp/key_value_replay.py`的实际白化为Cholesky L^{-T}q并含score-RMS；
`b2bb03ce:src/ember/ecp/policy_response_writer/{model,process,composer}.py`；
[P/Q四任务合同](../runs/outputs/pi05_ecp_prw_complete_shared4_s64_b2bb03ce_gpu02p235_20260906/run_contract.json)；
[width256完成范围](../runs/outputs/pi05_ecp_prw_complete_target18_width256_s128_14bc7605_gpu02p012356_20260906/result.json)。

### 3.1 bank／chart／interaction 中间路线的完整补表

以下补齐旧账本§27–51、64–90。主要是功能、几何或局部资格，没有新增环境成功率；
其中wrong-bank是当时训练侧内部干预，不是selected validation最终视频controls。

| 范围／路线 | 实际正负证据 | 预算与推断边界 |
| --- | --- | --- |
| §27–32 current-bank/equal-subspace/family-owner | F1四family解析回放约.9998；共享F3 held .0897，equal-subspace .0730，family-owner .0746 | F2 25；F3 25→50，50＝300 task-exposures/600 K1条件；后半仍上升。operator容量与共享获取有差距，未证收敛 |
| §33–38 stable-anchor/owner-query/additive | same-task transductive code约.91，inductive差；stable-anchor held .1421、owner-query .1631、additive .1287；关joint几乎不变 | 多数50更新；FiLM/free-query probe仅20，另task85 current-key200深层仍弱。某个joint模块未取得职责，不等于所有交互无效 |
| §39–42 consensus/functional-anchor/polar | fit-video更新consensus到held .9458；full-Program anchor .0828、IEEE修正后.0831；公共rank4也可约.83，暴露旧Gate可被公共更新绕过；polar解析witness .996–1.000 | anchor/IEEE各25更新；**polar只有58.33秒/condition profile，无formal训练、K4或科学Gate** |
| §43–46 functional-sketch/set-summary/primal | native-Q rank64在四个预注册witness .1567–.1574，bank参考约.996；set-summary两轮fit/held .328/.175、.349/.132，free-logits约1；P1六task .9717/.9545 | sketch无训练局部门non-pass；summary确实各1000步，但仅task93/q20、冻结随机或旧25-step chart；P1只证局部跨video容量 |
| §47–48 behavior sufficiency/G2-B | 四个1000-step reader fit可.97–.98，held full .2695、language-only .2687；G2-B动态增量39.4%，新task行为读出约.28–.30 | G2-B macro60实际**420更新/2280 task条件**，从G2 m20初始化、fresh optimizer；是有限读出反证，不是所有原生信息利用的反证 |
| §49–51 role-local/joint/global behavior kernel | v3监督图meta45分成5个不连通分量；连通和全局calibration修正后panel-B .1207/.1129/.1054，task spread扩大但行为几何未形成 | 三轮各15更新/150 presentations，G2组件初始化；图不连通是目标缺陷，不能由末两macro近稳断言长期收敛 |
| §64–65 functional refinement/raw Stage0 | R10从R9继续、固定chart后held-video .5442、task-held .1515；R11换raw Stage0为.2881/−.0924 | 各110新增更新，10warmup+100effective，非fresh；功能信用有局部正效应，单删Program压缩没有修复 |
| §66–70 full/half/tempered operator | R5正确/错误bank .9309/.9458，full inverse可把bank变成坐标；half正确/错.7252/.1889；.75幂.9253/.8850；已知primal hard switch .9509/.0052、soft正确仅.2387 | 10task、两fit-bank、100步局部优化及无梯度干预；是当前operator的capacity/specificity折衷，不是共享判别器已学成 |
| §71–73 hard-route/R12/R13 | 局部阈值用于R10后held−.6319；R13 held-video/task-held .0487/.0330；support阈值差约.00006可让fit−.208→.964，held仍失败 | R12/R13各110新增，分别从R10/R12起；R13仅训4.854M probe。离散route对小兼容误差敏感是具体事实 |
| §74–80 candidate scalar/base-score/vector | 初版wrong梯度被归一化放大15.7–359.6倍；修正后correct/held约.67/.66但wrong仍高；正锚/base-score约.93/.95，wrong约.93；局部vector可分离却使correct到.711/.570 | scalar各110×6＝660 presentations，每task两correct/一wrong；vector两task各80步。当前chart/目标不能同时保容量与选择性，不是所有candidate函数上界 |
| §81–86 EBSRI summary/shared/polish | S0/S1两task通过，S1 held .962/.881、wrong负；fresh direct S2 held正确.949/.899、wrong.931/.900；polish训wrong下降但held正确.498/.614，task93 margin .030 | S0/S1各task110；S2各110共享，direct为660分支条件；polish非fresh。仍主要fixed routing/R5 chart，不能当Natural Program joint成败 |
| §87–90 quotient/owner/relation | 去absolute后task93 correct-held .725，补owner .662，保留target-centered关系 .724；未恢复full-z容量 | 各两task fresh110，只到S0，无S1/S2；同时删B0/B1并有free-token/真实summary拓扑不匹配，不能唯一归责关系表示或认定absolute code必需 |

补表直接入口：[stable-anchor合同](../runs/outputs/pi05_ecp_shared_compiler_g3_f3_stable_anchor_fold0_m5_20acc33_gpu01p012346_r6_20260827/run_contract.json)、
[P1报告](../runs/analysis/pi05_ecp_primal_capacity_p1_v1_c9e8198_gpu01p012345_20260829/report.json)、
[G2-B420步合同](../runs/outputs/pi05_ecp_natural_program_g2_behavior_fold0_m10_5cbe76e_gpu01p012345_r6_20260829/run_contract.json)、
[candidate interaction](../runs/outputs/pi05_ecp_program_bank_candidate_interaction_v3_anchor_s110_fd20251_gpu01p012_r3_20260831/run_contract.json)、
[EBSRI direct-S2](../runs/outputs/pi05_ecp_event_bank_set_s2_direct_functional_s110_25477c9_gpu01p013456_r6_20260901/run_contract.json)、
[functional polish](../runs/outputs/pi05_ecp_event_bank_set_s2_functional_polish_gate_s70s110_bb98b81_gpu01p0256_w4_20260901/aggregate.json)、
[relational quotient](../runs/outputs/pi05_ecp_event_bank_set_relational_quotient_s0_gate_s110_ad64757_gpu01p34_20260901/aggregate.json)。

## 4. Layered、Horizon与视频功能路线

除注明外，本组四suite各一task/update、每task64queries，即4C/256Q；多数是fresh Writer/reading Meta，source冻结。
teacher0–15/action16–41较常见；Semantic-Path改为同池16–41且排除同episode。
`2ecf1770`起主监督offset1，但source仍旧；不能与后来source也对齐的新A作单变量比较。

| ID／方法 | 真实机制与预算 | 闭环及推断边界 |
| --- | --- | --- |
| E20 Layered e4ca5998 | 完成18层AE后，外部处理[T,18,50]；四轮H对应／邻帧更新，p64 coordinate A/B decoder | short4 fresh96＝384C/6144Q；train24 192/384＝768/1536C、12288/24576Q。旧400行correct69→67、other72→64；train120 22→seen21/held18，Long0。视频重复边界见§1.1，每task终点仅1024Q |
| E21 Horizon K1 b6d70d98 | 最终H50→外部U256；四轮过去4帧H对应、两端真实Z、GRU、H-read、P/U回写；P4→Compiler→329.5M独立D | fresh100–600 correct55/110/86/87/70/82，600＝2400C/153600Q；train200/400/600 52/59/67 /96，FM仍改善。获取存在，泛化弱；600的81/82成功集中于source支持的三个task |
| E22 Horizon语言与D干预 | first-query-only、contextual、local/H-read/Compiler语言删除、target内rank共享D，均fresh | first-query-only100–400 75/110/106/103；contextual52/103/79/90；matched200→400 all103→90、none108→92、local-only114→110、Compiler-off108→126、H-read-off98→90。共享D115→82、train32→60；每400＝1600C/102400Q |
| E23 Compiler-off复核／续训 | E22局部删除在其它初始化／数据关联复核，再原配方延长 | init11 all100→108/off104→119；新关联all99→82/off101→126；off500/600却73/54，train仍升。局部因果正效应与后来退化都须保留，不能写成删语言即可解决 |
| E24 Consumption R/C/S | R沿用off；C保留语义S、S条件读中心化P4；S读全部真实帧/fullH但无跨帧有序处理；8C×32Q/update | 各200＝1600C/51200Q；100→200 R37→83、C43→50、S59→45，train46/49/52。R300/400 63/85、train400为56，与旧off126/train56有缺口；更多条件调用并未自动修复 |
| E25 No-change reference | 真实历史GRU减同gap静态自配对GRU，两项都反传；无额外AE/loss | fresh200＝1600C/51200Q；重复静态P降至真实约.9%，结构性质成立；train36→45，validation34。计划400未执行，不能称已验证平台 |
| E26 Video Functional a81a38ed | task-token读patch、两端100个H→E[T,L,256]；两轮组织；辅助头读同一E预测另一episode功能；KD只训Compiler/D | main/frame-set各300＝1200C/76800Q；50/100/200/300 correct69/72/74/32 vs66/73/69/27；other71/71/74/32 vs67/74/61/31。无可信有序增量，两臂后退 |
| E27 Functional后续 | 去KD保留辅助FM，再加VL Meta，再关辅助FM保留主FM | 各200＝800C/51200Q；去KD100/200 correct/other57/61→72/67，train42/38→55/55；加VL61/61→62/64；pureFM+VL60/55→64/54。去KD有获取正效应，未形成足够迁移；单加VL或纯FM也未修复 |
| E28 Local Action | 同encoder四帧片段预测15步动作；主LoRA仍跨episode，另有局部时间Key | 每臂200：800主C/51200主Q＋800片段/6400 noise-time draws；100 correct/other45/48 vs58/53；200 30/27 vs30/26，Long0。局部FM小幅有序优势不能证明E已经足够；无同图无辅助第三臂 |
| E29 V-JEPA2.1 | 冻结EMA过去四帧、576 dense tokens经task读入外部grounded表示，再读fullH；无native中层写回 | 各200＝800C/51200Q；train ordered41→52/frame-set41→40，validation53→33/48→48。24task×4video×8state固定复核359/768 vs361/768，三个CI跨0；保留旧局部正例，但不称稳定有序优势 |
| E30 Execution-aligned | E29主动作改post-action offset1，旧source未重训 | 各200＝800C/51200Q；train44→51 vs36→53；validation61→72 vs65→81。正确对齐应保留，有序优势仍未成立；采样支持也变，非纯offset历史差额 |
| E31 Native双相机 | 真实两相机进native prefix，V-JEPA仍仅agentview | 各200＝800C/51200Q；train36→54 vs32→50；validation64→53 vs56→39，ordered200 Spatial/Goal0、breadth6→3。相对差额更大不等于更强绝对能力 |
| E32 空间信用 | OOI/motion标签只监督native/prior attention Q/K，不成为部署输入 | 各200＝800C/51200Q；train32→50 vs33→50；validation20→71 vs24→70；ordered相邻R/G/L19/52/1。对无空间监督+18/+31均有正CI；一般行为获取真实，有序−无序−4→+1无资格，Goal仍低于source |
| E33 Semantic-Path | 保留L；每语义位置读全部T×50H；双向交互→32维状态→二阶log-signature528；语义调制＋约2.188M共享family fullAB heads | 各100＝400C/25600Q；50/100 train16→29 vs22→26，validation77→48 vs75→44；ordered R/G/L30/18/47，J=.31579。全部task独立FM仍改善，100不是充分平台；紧凑共享输出与纯FM本身不够 |

E21的原生路径先完成各帧18层；四轮 P→U→下一轮双端 Z 取证是真实交互，不能被简写成一次静态读。
但 U 是外部256维，既未写回真实中层 prefix，也未让剩余原生层续算。
E20同样只是捕获全部层之后外部计算；VL Meta 可训不等于当前视频的跨帧结果进入 native prefix。

本组主要原件：

- [Layered384裁决](../runs/analysis/layered_relation_writer_20260907/train24_shared/decision_after384.json)。
- [Horizon K1合同](../runs/outputs/horizon_k1_supervised_v1_seed7_20260908/run_contract.json)、
  [共享D矩阵](../runs/analysis/horizon_relation_writer_20260908/causal_learning_20260909/sharing/completed_matrix.json)。
- [Consumption](../runs/analysis/video_consumption_20260911/first_round_summary.json)、
  [No-change](../runs/analysis/video_change_reference_20260911/step200/paired_summary.json)。
- [Functional完整比较](../runs/analysis/video_functional_20260911/paired_summary.json)、
  [pure-FM+VL结果目录](../runs/analysis/video_functional_20260911/direct_fm_vl/)。
- [Local Action](../runs/analysis/local_action_grounded_20260912/bounded_200_decision.json)、
  [V-JEPA固定交叉复核](../runs/analysis/frozen_positive_replication_20260913/paired_replication_summary.json)。
- [offset1](../runs/analysis/execution_aligned_video_20260912/paired_summary.json)、
  [双相机](../runs/analysis/native_dual_video_20260913/camera_comparison.json)、
  [空间监督](../runs/analysis/visible_object_grounding_20260913/supervision_comparison.json)、
  [Semantic-Path](../runs/analysis/semantic_path_writer_20260914/paired_readout.json)。

### 4.1 本组局部诊断不能升级成什么

- Horizon first-query-only 的74496次冻结预测中 own-task FM .111621→.106557，400时24/24task自己的条件优于其它task平均，
  两video／两query半份都成立。它反对“完全没有学到任务映射”，但语言和视频一起变，不能单独证明视频因果。
  [functional assignment](../runs/analysis/horizon_relation_writer_20260908/k1_first_query_only/functional_assignment/summary.json)。
- P4/C/A-B局部oracle用了很小的固定支持集重复拟合；A/B仅6.8% fit收益迁到新noise，独立episode八task都变差；
  闭环A/B15/32、normal18/32、free-C20/32。task7的normal/P4各1/4而C/A-B各4/4是局部正例，不能外推为整个decoder充足。
  [完整诊断及预算](horizon_k1_causal_diagnostics_20260909.md)。
- `b1d3fa25:native_reader_diagnostic.py:55–73`确实在**执行侧**AE第10层 input_layernorm 后注入fixed-E读头并原生续算；
  source/Meta/E/Compiler/D都冻结，只训新reader，没有写VL prefix。1536支持queries×32epochs＝49152复用、192updates；
  held FM .154897→.139792，旧末端reader .140281，LoRA学生 .110005，24/24仍逊学生，16→32仍在改善，无rollout。
  这是该固定表示/读头/预算的诊断，不能否定fresh联合teacher Z/H桥。
  [合同](../runs/analysis/video_functional_20260911/native_reader_diagnostic/run_contract.json)、
  [配对分析](../runs/analysis/video_functional_20260911/native_reader_analysis.json)。
- Semantic-Path固定回放128次有120次复现结果，失败同时包括选错物体、抽屉子阶段、搬运/放置困难、子目标后丢失。
  不存在从这些片段唯一推出“没有顺序理解”或“只差Compiler”的依据；原400分数没有被回放替换。
  [行为回放](semantic_path_behavior_replay.md)。

## 5. 真纠正、Local Field、Pullback

| ID／方法 | 真实机制与预算 | 事实、充分程度和限制 |
| --- | --- | --- |
| E34 真纠正G | train动作误差产生裸source38-target rank16更新；无共享学习 | 独立full10 MSE .16493→.15575；480episode中source17/96，四teacher21/25/20/24，净+5.73pp、CI[1.04,12.5]。存在性成立，但Object无收益，task35占一半净增 |
| E35 Native Correction | native X生成A、自由B预测纠正；Writer＋VL/Action Meta fresh joint；FM+.1空间+.1完整更新监督 | 每臂200＝800C/51200Q；100/200 ordered validation50→50、frame-set51→49，train15→19/15→20。合法RGB共享获取未成立，不否定E34存在性 |
| E36 Local Correction Field | 预测同位置Ĉ=RUᵀ，再与裸source X收缩；fresh joint、FM+.1cotangent场loss | 每臂100＝400C/25600Q，单task实际8–25C；50/100 ordered train18→17、validation48→47，frame-set16→17/47→50。FM末段仍改善；privileged rank16(C)函数效果接近G但无该oracle闭环 |
| E37 Process Pullback | 全视频→每帧50×7 q→固定裸source Jacobian→native X PCA16；fresh joint、纯FM | 900＝3600C/230400Q；train300/600/900 24/22/26，validation64/72/64。每task126–169C、623/624不同条件；比100步场实验充分，但行为获取仍弱 |
| E38 learned outlet | E37再加共享identity起点左右低秩变换，仍通过同q通道 | 900同C/Q；train21/20/18，validation50/75/79，79中77来自两奶酪任务。FM持续改善，不能称完全没学；新增行为狭窄，训练行为下降 |
| E39 free-q/free-A/B诊断 | 共同Writer900起点、24task/teacher16、64个跨episode支持query×两套FM随机性；LBFGS最多32iter/soft48eval | 原输出/free-q/freeAB closed-loop20/18/46 /96，freeAB breadth19、四suite非零；MSE .171922/.150169/.140512。q实际4–32iter、AB均32，**未认证收敛**；支持固定compiler限制了此预算可取得行为，不能唯一定位PCA |

E35固定A的误差可正交拆分为 ||G(I−P_A)||²＋||BA−GP_A||²；后项占66.9%–70.6%，24/24task均更大。
最优全局非负倍率只消除3.87%–6.32%误差，因此“只扩A”或“仅幅度错”不足；但这个分解允许任意B，不证明有限共享B头足够。

E36的作用为 Δy(o)=T^{-1}Σ_i ĉ_i〈x_i,x(o)〉，是native相似度核上的纠正，不是teacher动作回放。
参数误差界 ||Ĝ−G||≤||Ĉ−C||·||X||/T 不能直接推出闭环成功界。

E37/E38限制为 vecΔW=M_V q，q∈R^{T×50×7}。它不是“只有7个数”，但所有层被固定Jacobian像、PCA右空间和共同q耦合；
共享左右变换没有变成任意task-conditioned完整A/B。

原件：

- [真G功能](../runs/analysis/native_corrective_transfer_20260913/paired_summary.json)、
  [真G行为](../runs/analysis/native_correction_writer_20260913/oracle_rollout/decision.json)、
  [A/B误差分解](../runs/analysis/native_correction_writer_20260913/acquisition_audit/summary.json)。
- [Local Field](../runs/analysis/local_correction_field_writer_20260914/paired_readout.json)、
  [Pullback训练审计](../runs/analysis/process_pullback_writer_20260914/training_audit.json)、
  [free-q/AB预注册](../runs/analysis/process_pullback_writer_20260914/causal_diagnostics/registration.json)、
  [learned outlet](../runs/analysis/process_pullback_learned_outlet_20260915/paired_readout.json)。

## 6. 原生读出、source对齐、当前A与SFT

| ID | 事实与范围 | 判断 |
| --- | --- | --- |
| E40 冻结source端点读出 | train24共384位置；task动作均值MSE .250597；agentview public t1 .347319；dual public t1 .131768，22/24task优于均值；dual full10 .136727 | 双视角下原生动作输出有用，增加denoise深度没有支持；不证明视频动态、中层H9、Meta后表示或LoRA闭环收益 |
| E41 source对齐 | 过滤71task source重训1000、256000Q；raw1000 source validation旧47→新50/400，train17→13/96；source更新末次保存失败经原状态恢复，未重做权重更新 | 修正后裸分接近，不说明native hidden／Writer梯度几何接近；source本身未建立稳健净改善 |
| E42 当前A主曲线 | agentview＋fixed-mean恢复v5.2近等价主图；三Meta fresh纯FM。300/600/900/1200/1500/1800/2100/2400/2700 correct99/88/140/135/112/122/122/106/108，train36/47/54/62/56/63/60/58/64 | 900有真实获取，未过>145；1200→2700 validation135→108而train62→64，较长泛化回落；不能只看900或只看最后两点 |
| E43 当前A曝光与保持 | 每更新4C/84Q；2700为10800C/226800Q，每task450C；2400→2700 validation R/G/L81/27/25，churn52，J=.609；train48/16/10 | 106→108净增2仍有大量换手；2700 breadth5，S/O/G/L6/50/37/15。3000训练已完12000C/252000Q，但**3000闭环未运行** |
| E44 共享SFT | 对齐source，rank128，global576，450更新/259200Q；400/425/450 validation85/89/86 | 两个相邻R/G/L66/23/19、64/22/25，churn42/47，两个task始终0；未过资格。旧109/107不是当前backend/source的已验证成绩 |
| E45 9月15日旧source B | v5.2同Core/P/AdaLN/8heads；改dual＋learned-H50、offset1、跨episode排除与46/4分池；fresh纯FM、原12000步LR时钟 | 300/600/900/1200 correct97/105/77/85，train35/49/50/56；终点4800C/100800Q、每task200C、全部1104条件各4–5次。不能以source单步预测正例或读取更完整宣称这一路已更好 |
| E46 旧source C任务共现 | 与E45相同模型、同事件多重集合与曝光，只重排4task/update的预登记语义共现；改变轮内Adam历史 | 300/600 correct91/65、train28/43；600比B105少40，CI[−18.5,−2.75]pp。两臂在300新获的58个验证成功，600时B留42、C留28；611更新后owner停止，完整点止于600，非完成1200 |

E40原件见[source读出审计](source_endpoint_readout_audit.md)。E41–44参见
[A训练合同](../runs/analysis/source_alignment_20260915/A/training/run_contract.json)、
[A累计读出](../runs/analysis/source_alignment_20260915/A/continuation_readout.json)、
[2700主面板](../runs/analysis/source_alignment_20260915/A/node2700_primary_readout.json)、
[SFT配对读出](../runs/analysis/source_alignment_20260915/SFT/paired_readout.json)以及findings§107–114。

E45/E46原件见[旧窗口合同](v52_return_plan.md)、
[B完整配对](../runs/analysis/v52_return_20260915/baseline/paired_readout.json)、
[C已完成配对](../runs/analysis/v52_return_20260915/cooccurrence/paired_readout.json)。
旧B四点的H-read q范数为.11328/.16078/.21298/.26040；由RMS归一化logit可导出每个位置最大权重
至多2.533%/2.797%/3.106%/3.417%。它排除强权重集中，不是实际分布测量，也不能证明H-read变化无影响或把差距全归双相机。
[精确参数上界](../runs/analysis/source_alignment_20260915/archival_B_horizon_weight_bound.json)。
E46则表明按语义分组不自动改善保持；优化顺序和Adam历史仍是干预组成，不能仅称梯度冲突导致差额。

E42与旧v5.2的`temporal.py`主计算近等价；`529da6b→575c189a`只有注释/格式变化。
fixed-mean的零query/bias与旧H均值数学等价，新增1074个冻结参数不消耗随机初始化。
未对齐的是source、动作／采样合同、跨episode排除、46/4池及state-video映射；不能杜撰Compiler变化解释差异。

### 6.1 视频证据原样保存及用途边界

下表除A外是历史各自冻结方法的结果，不作为本次新架构选择指标。顺序为correct/other/wrong/shuffled/reversed，旧比较没有在此添加不存在的no-video臂。

| 冻结历史点 | 五臂 |
| --- | --- |
| v5.2 old900 | 132/138/74/82/83 |
| v5.2 TC400 | 120/109/107/111/124 |
| v6 old500 | 121/122/111/84/47 |
| v6 TC400 | 143/135/125/128/129 |

[历史视频审计](../runs/outputs/pi05_as_writer_v52_v6_recipe_video_causality_audit_seed7_20260802/analysis.json)。
它们是不同曝光下的winner snapshots，不是上文同曝光配方估计。
正确输入相对干预有真实依赖证据，仍不单独等于胜过充分训练的language-only／全帧无序参照。

当前A由owner授权的问题分析使用六臂correct/other/wrong/no-video/shuffled/reversed：

| A节点 | 六臂 |
| --- | --- |
| 900 | 140/136/116/48/128/139 |
| 1200 | 135/128/106/48/107/121 |
| 1500 | 112/120/104/48/108/89 |
| 1800 | 122/127/98/48/112/113 |
| 2100 | 122/121/110/48/106/111 |
| 2400 | 106/103/91/48/107/95 |
| 2700 | 108/111/93/48/106/115 |

旧强特异性**没有在新A900复现出来**，不能淡化为措辞上的“尚未稳定建立”；后续也没有形成可靠一致的正确时序优势。
这些干预不证明模型完全不看视频或顺序。no-video=零LoRA只说明相对该source参照的适应效果，不是learned language-only消融。
2700 correct减wrong/shuffled/reversed的区间均跨0；不能只挑其中一个早期较大差额作最终方法性质。

## 7. 专家论证、修正与证据缺口

相关完整评审按旧账本索引阅读，包括native-factor、bank-conditioned、functional-sketch、joint-primal、
Program-bank interaction、EBSRI、Program-through-bank、全局路线、Policy-native Meta-Writer、P/Q，以及9月的process/visual-loop和joint functional评审。
它们提供了可检验假设和接口分析，不能自动升级为实验证实：

- “native坐标普遍优于自由A/B”“不能从第一天生成A/B”是历史建议；没有匹配总体因果证据，后续完整P/Q局部正例限制了其适用范围。
- 9/2评审把PNBTT E1说成Natural Program→bank共享失败，范围越界：E1是task-local free query，E2未跑。
- GOMQ rank32→16导致失分的旧解释被有效rank≤16的代数纠正。
- G3190步、PNBTT110步、GOMQ3次有效memory更新、Semantic-Path100步、LocalField8–25C/task不能一概写成“所有方法都已充分训练”。
- 动态可解码不推出可编译；oracle函数可用不推出RGB可预测；非零信用不推出训练已给出足够行为；小norm/cosine漂移不推出闭环保持。
- 低梯度retention不是梯度冲突的充分证据：等范数正交梯度的retention自然是1/n；跨step比较同时改变参数、query和noise更不能唯一归因。

**尚缺且没有被本文伪造的消融：** v5.2同前端/训练下删除Procedure或AdaLN、P两层改summary、仅更换共享head、
逐一冻结三个Meta、以及匹配新source/dual/H50的中层跨帧写回比较。
历史Horizon语言删除、frozen置零、顺序干预和Semantic-Path整体替换不能替代这些实验。
保留v5.2尾端的依据是完整正例与功能解释，不是已证明所有部件必需或绝对最简。

本次审到的Dynamic-K/LMMPC/GOMQ native memory均逐帧计算，one-way mask不让prefix／Action读memory；
跨帧阶段在native之后。Horizon有强外部反复取证，native-reader有执行侧中层注入，但都不同于拟议的
**teacher读取侧跨帧联合Z/H写回真实中间层并原生续算**。
“在所审集合未实测”只建立区别，不建立更易学习、更强迁移或必然成功。

## 8. 从证据到设计的裁决边界

可以认真保留的正证据是：普通FM的完整v5.2组合能学到行为；某些完整出口能恢复受限出口丢失的行为；
native双视角有功能先验；空间监督、Compiler-language删除、GOMQ等有各自局部正效应。
它们不构成一个同时拥有所有优点的虚构模型。

足够明确的负证据是：现有方法没有达到稳定最终资格；多条路线熟悉task/video也弱；
一些短窗口内部性质修复没有转成行为；长一些的A/Pullback/Horizon窗口暴露了获取、泛化和保持的不同缺口。
它们排除的是实际检验的组合与预算，不能统一证明“视频没用”“只差Compiler”“只需更多参数”或“所有方法还没训够”。

据此，新设计保留绝对内容和可学习完整输出，对一个具体结构限制提出新假设：旧Core对顺序不变、显式时序只经晚期P调制；
同时写回原生Z/H可使过程进入主内容生成路径，并复用原生后半段条件函数。
这个结构推论可由代码和链式法则检查；它能否改善视频教学、迁移及合理条件下的保持，仍须未来合法matched训练和闭环验证。

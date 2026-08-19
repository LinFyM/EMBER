# EMBER External Review Follow-up

状态：2026-08-19已完成并封存。本文面向原外部复核专家，逐项汇报
`docs/external_review_20260818.md`建议的执行结果。完整逐条状态以
`docs/external_review_claim_ledger_20260818.md`为索引，本文不给出下一套架构方案，以免影响后续独立判断。

## 1. 执行边界

- 方法始终基于当前Core-Addressed Reader主架构，没有恢复V6/LPCP/GOMQ；历史方法只作paired对照。
- 生成LoRA保持38 targets、rank16、一次Writer forward、Action Meta-LoRA、Dynamic-K、memory Reader、K-set、
  bounded M2P和FactorHeads。
- owner明确要求后续canonical Writer不再使用Text或VL Meta-LoRA；冻结原生language输入与Action Meta-LoRA保留。
- validation/test teacher action、reward、state、task ID和deployment expert bank始终不进入Writer或梯度。
- 所有方法判断仍以single-checkpoint strict paired400为准；loss、gradient、projection error和oracle仅作根因证据。
- 专家给出的`+10/+15`、`90%`等门槛逐项并列报告，但不追溯改写EMBER原有科学合同。

## 2. 本轮实际改变了什么

本轮没有设计新架构。唯一canonical代码修复是：保留frozen native hidden的detach，只移除Writer-local
`frame_evidence`、`grounded_evidence`和`interactions`返回处的第二次detach，使fresh
`patch_grounding`与`interaction_projection`能从functional objective获得credit。为了把owner的no-Text要求与
该修复分开，采用三点链：

| arm | Text/VL Meta-LoRA | projected output detach | 用途 |
| --- | --- | --- | --- |
| A | Text rank4 / VL 0 | 保留 | sealed历史基线 |
| B | 均移除 | 保留 | 只归因Text移除 |
| C | 均移除 | 移除 | 只归因front-end credit恢复 |

B与C均从fresh训练；B/C间除三项返回tensor的autograd连接外，rank、拓扑、data、B20、K schedule、optimizer、LR、
seed和world6均一致。

## 3. A--E意见的当前裁决

### 3.1 我们同意并由代码或artifact确认的部分

1. language、video、Action representation、layer-addressed memory和一次性完整LoRA的职责划分符合EMBER目标；但
   “符合设计意图”不等于已证明高层过程理解。
2. source freeze、one-way memory mask和information wall成立；没有teacher action、reward、outcome、task-ID route或
   deployment expert bank泄漏。
3. 原实现确有第二次detach；它切断grounding/interaction自身credit，但没有让整个video路径失活。
4. Reader是时间中心化，不是真正保留首帧reference；它去掉静态Value，也丢失绝对/终点Value。
5. memory/Reader/Core/M2P多套address只证明index preservation，不证明policy-functional correspondence。
6. Action 50-token mean、M2P RMSNorm、endpoint派生和FactorHead 256维输出子空间均为真实结构事实；其性能后果不能
   仅由结构推断。
7. `task drift`只是结果描述；Procedure趋同有LPCP143反例；functional mismatch必须拆成state cotangent、shared
   update、generated occupancy与retention；不能未经matched干预归罪mean或AdamW。
8. 123的absolute缺口与25→50崩落可能有不同首因，不能被一个故事吞并。

### 3.2 原复核中仍属推断、经本轮被限制或修正的部分

- “Program主要学task identity/carrier”仍未被证明。A的correct显著优于wrong/reverse/no-video，说明视频内容与粗方向
  有用；但correct不优于shuffle，仍未证明正确中间顺序是必要evidence。
- “generated occupancy shift导致52 lost”的简单版本未获支持；详见F2。
- “FactorHeads不可达”由F4最终闭环oracle裁决；仅凭256维子空间不能下结论。
- “head co-drift是漂移根因”只得到部分支持：冻结heads大幅优于正常续训，但仍丢失大量旧support。

## 4. F.0：零训练成本证据

### 4.1 全模块functional gradient audit

三次clean、真实source-policy functional backward给出：

- A/B在heads已打开的macro25仍对`patch_grounding.query/key/output/norm`和
  `interaction_projection`完全无gradient；不是小梯度，而是autograd图未连接。
- C恢复三项连接后，在macro1上述所有组首次观测到nonzero finite gradient；其它Action Meta、Core、Procedure、
  memory、Reader、K-set、M2P与八个head family也保持有credit。
- fresh identity点仍遵循B-family先开、A-family和upstream后开的正常一阶顺序。
- 三个arm的source policy nonzero gradient tensor均为0；所有trainable参数均被分组，无unclassified参数。
- 稳定regression覆盖projected outputs局部credit、frozen native replay不回传source，以及intended-path分组完整性。

远程证据：`docs/evidence/external_review_20260818/gradient_credit_evidence.json`。

### 4.2 A macro25完整视频因果面板

| condition | strict400 | breadth@1 | breadth@5 | breadth@10 |
| --- | ---: | ---: | ---: | ---: |
| correct | 123 | 8 | 3 | 3 |
| same-task-other | 125 | 7 | 4 | 3 |
| cross-suite-wrong | 81 | 6 | 4 | 3 |
| shuffled | 122 | 4 | 4 | 4 |
| shuffled-keep-first | 131 | 7 | 5 | 3 |
| reversed | 90 | 6 | 4 | 3 |
| no-video | 48 | 3 | 2 | 1 |

correct相对wrong/reversed/no-video分别净`+42/+33/+75`且显著；相对shuffled仅`+1`，相对
shuffled-keep-first为`-8`。same-task-other总分满足±10，但只保留correct成功行的85.37%，低于专家90%建议门。
因此A不是language-only，也不是任意nonzero carrier即可；但正确中间顺序没有通过因果门。

## 5. F.1：Text移除与front-end credit的matched结果

### 5.1 A→B：只移除Text Meta-LoRA

B macro25 correct为`104/400`，相对A的123为`81 retained / 23 gained / 42 lost`，净`-19`，exact McNemar
`p=.024812`，success-set Jaccard`.55479`。suite净变化为Spatial`-2`、Object`-12`、Goal`-4`、Long`-1`；
breadth@1由8降到6，breadth@5/@10仍为3/3，top-3 share由`.91057`升到`.93269`。最大单task净损失是Object task3
的`-13`。

这证明Text Meta-LoRA在A中提供了真实闭环support，而不是可无损清理的冗余；它是否以科学上理想的方式提供support，
仍需结合B/C视频controls判断。owner边界仍要求未来canonical不使用它，因此B是必要归因基线，不是最终候选。

### 5.2 B→C：只恢复projected front-end credit

C macro25 correct为`110/400`。相对B为`85 retained / 25 gained / 19 lost`，净`+6`，exact McNemar
`p=.45138`，Jaccard`.65891`；breadth@1/@5/@10仍为`6/3/3`，top-3 share反而由`.93269`升到`.95455`。
suite变化为Spatial`0`、Object`+7`、Goal`+1`、Long`-2`。相对A为`90 retained / 20 gained / 33 lost`，净`-13`，
`p=.09837`。相对LPCP143为20 gained / 53 lost，净`-33`，`p=.000142`；相对GOMQ151为18/59，净`-41`，
`p=3.06e-6`。

所以F1的mechanism事实成立，但absolute根因支持门失败：B→C没有达到净`+10`或显著性，breadth没有增加，C仍显著低于
143/151。C继续到macro50为`101/400`；相对C macro25是`77 retained / 24 gained / 33 lost`，churn57，
Jaccard`.57463`，breadth@1从6降到4。净分只降9掩盖了33行旧support丢失，因此credit修复也没有解决稳定积累。
controls显示它确实改善了视频内容/方向资格，但仍不授权围绕detach继续调参；两类结论必须拆开。

### 5.3 B/C视频因果与same-task稳定性

以下均为同一8×50 rows、state/env/policy RNG、K4 teacher ordinal严格配对；括号为`correct-control`净分：

| arm | correct | same-task-other | wrong | shuffled | keep-first | reversed | no-video |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A Text+detach | 123 | 125 (-2) | 81 (+42) | 122 (+1) | 131 (-8) | 90 (+33) | 48 (+75) |
| B noText+detach | 104 | 101 (+3) | 65 (+39) | 83 (+21) | 90 (+14) | 96 (+8) | 47 (+57) |
| C noText+credit | 110 | 111 (-1) | 54 (+56) | 91 (+19) | 93 (+17) | 69 (+41) | 47 (+63) |
| F5 C+PCGrad | 107 | 111 (-4) | 51 (+56) | 92 (+15) | 105 (+2) | 53 (+54) | 47 (+60) |

paired细节给出更清楚的分解：

- A→B移除Text后，correct丢19，但shuffled/keep-first分别丢39/41，correct-shuffle margin从1增到21；reversed却
  从90升到96，correct-reverse margin从33降到8。Text路径有真实support，也混合了有用方向与order-corrupted旁路，
  不能写成纯language-only shortcut或纯正机制。
- B→C只恢复front-end credit，correct仅+6，但wrong从65降到54、reversed从96降到69；correct-reverse从不显著
  `+8 (p=.38173)`变为显著`+41 (p=9.81e-6)`，correct-wrong从+39增到+56。shuffle margin`21→19`近似不变，
  keep-first为`14→17`。B的correct-no-video为显著`+57 (p=1.26e-12)`，C为`+63`，说明修复前后都不是
  language-only。因此修复主要改善视频内容和全局正向过程利用，不是absolute或中间顺序margin的主因。
- C是四个arm中唯一同时显著优于wrong、shuffle、keep-first、reverse和no-video的arm；但no-video→correct的净收益
  几乎全在Object（+59），correct-reverse在Long反而`-12`，keep-first优势也主要来自Object，不能据aggregate宣称
  跨suite高层过程理解。
- F5相对C把macro25 breadth@5/suite minimum从`3/1`变为`4/5`，wrong/reverse/no-video区别更广；但
  correct-keep-first只有+2且不显著，对照C的+17。它更偏向全局箭头/端点证据，而没有保留中间阶段顺序。
- same-task-other总分四个arm都在correct±10，但correct-success row retention仅
  A/B/C/F5=`85.37/83.65/87.27/85.05%`，均低于专家建议90%；总分稳定不等于同初始化稳定。

完整per-task、per-suite、breadth、top-3、400-row outcomes和provenance见
`b_video_causality_evidence.json`、`c_video_causality_evidence.json`与`f5_video_causality_evidence.json`。

## 6. F.2：fixed-occupancy反事实

对A macro25→50的52 lost、13 gained、71 retained共136行，分别采集两checkpoint真实rollout occupancy，并在
`S25 union S50`上重查询两checkpoint动作：

- offline B20 functional loss由`.113644`降到`.105405`；
- 136/136在初始状态第一次replan就已出现行为分歧，初态action RMS范围`.04284–.48279`；
- lost rows的`macro50 occupancy - macro25 occupancy` checkpoint disagreement均值为`-.006548`，Wilcoxon
  `p=.05466`；gained为`+.011293`，`p=.00610`；retained为`+.002409`，`p=.1993`；
- replay-consistent子集保持相同方向。

这反驳“macro50只在lost rows自己的occupancy上与macro25分歧更大”这一简单解释，方向对lost/gained反而相反。
但validation task expert不存在，读取held teacher action又违反information wall，因此本轮不能合法判断哪个checkpoint
动作更正确；fixed-union expert error明确标记为`underdetermined-after-audit`，不以checkpoint disagreement冒充。
由于支持前提未成立，F2.05 occupancy-matched B20替换为`not-applicable`，没有启动训练。

远程证据：`docs/evidence/external_review_20260818/occupancy_evidence.json`。

## 7. F.3：FactorHead co-drift

从A macro25冻结八个FactorHeads，唯一继续训练upstream到macro50：

- 正常macro50为84；frozen-head macro50为117，相对正常为49 gained / 16 lost，`p=5.08e-5`；
- 相对起点123为90 retained / 27 gained / 33 lost，success retention 73.17%；
- 16个head tensor逐元素不变，481个upstream tensor中404个变化；
- 满足专家score≥110门，但不满足lost≤20、retention≥90%和breadth不降。

结论限定为：head co-drift是25→50崩落的重要放大器，不是充分根因；固定heads仍会因upstream/objective变化丢33行。

远程证据：`docs/evidence/external_review_20260818/f3_headfreeze_evidence.json`。

## 8. F.4：fixed-head reachability oracle

在A macro25的八个FactorHeads完全固定时，为24个train tasks分别自由优化`20×16×256` Program 3000步，再把投影
LoRA与step2000 direct task experts在同一24×50 closed-loop rows上严格配对：

- direct experts为`658/1200`，投影LoRA为`659/1200`，达到direct的`100.15%`，远高于预注册90%门；
- 逐行只有`7 gained / 6 lost`，652 retained，success-set Jaccard`.98045`，McNemar `p=1.0`；
- Object/Goal/Long三套300 rows逐行完全相同，全部13-row churn集中在Spatial；
- effective-BA relative L2均值仍高达`.93571`，family为q`.95212`、v`.84250`、action-in`.59157`、
  action-out`.46702`；raw-factor relative L2均值`.41116`。

因此“FactorHeads必须精确重建expert tensor才policy-effective”被反驳，raw/effective L2也不能替代闭环oracle。当前
固定head manifold在train24上足以到达与experts功能等价的LoRA；没有依据继续扩大head、rank或decoder capacity。
这不证明视频到Program映射已解决，也不把privileged per-task Program变成deployment route。

远程证据：`docs/evidence/external_review_20260818/reachability_evidence.json`。

## 9. F.5：shared-gradient conflict条件分支

最后一个matched干预只把24-task arithmetic mean替换为固定顺序、无扫参的standard sequential PCGrad；per-task
gradients、AdamW及其moments、LR、tasks、B20、Dynamic-K、rank16、Writer和source均不变。50个macro中每轮实际
发生121--263次projection，PCGrad相对mean方向cosine最低`.94038`、均值`.98652`，因此不是等价空操作。

| arm | macro25 | macro50 | retained | gained | lost | churn | Jaccard | breadth@1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| C arithmetic mean | 110 | 101 | 77 | 24 | 33 | 57 | .57463 | 6→4 |
| F5 PCGrad | 107 | 96 | 82 | 14 | 25 | 39 | .67769 | 6→4 |

PCGrad减少8个lost与18个churn，却也少10个gained，两个checkpoint分别低3/5分且breadth@1同样下降。它在macro25
的breadth@5与suite minimum为`4/5`，优于C的`3/1`，但到macro50两者suite minimum都为0、top-3都约98%，早期分布
改善没有被保留。逐行比较loss
indicator时，22行仅C lost、14行仅PCGrad lost，exact paired `p=.24298`；gain indicator为15行仅C gain、5行仅
PCGrad gain，`p=.04139`。因此它收窄了task exchange，但没有通过“显著减少lost并稳定breadth/score”的根因门；
结合视频controls，它还强化wrong/reverse区别却弱化keep-first中间顺序。arithmetic mean下的gradient conflict是换手和
能力分布的影响因素，不能解释当前absolute上限或作为主要根因。由于AdamW在两臂
保持不变，本实验不能独立裁决Adam moment污染，必须保留该不确定性边界。

远程证据：`docs/evidence/external_review_20260818/f5_pcgrad_training_evidence.json`和
`f5_pcgrad_paired_evidence.json`。

## 10. G：远程证据缺口的逐项收口

| ID | 原要求 | 最终状态 |
| --- | --- | --- |
| G1 | 六checkpoint逐行paired400、video/RNG reference | completed；`paired_evidence.json` |
| G2 | A macro25全部视频controls与不同K4 sets | completed；`video_causality_evidence.json`，并额外封存B/C/F5三个完整面板 |
| G3 | intended modules实际gradient与首次credit | completed；A/B追加macro1真实backward，非detach upstream在第一次更新后首次非零，旧detach组在macro1/25均无gradient，C修复组macro1首次非零 |
| G4 | formal provenance与hash | owner-adjusted；clean commit、dirty paths、schema、manifest、bytes和contract公开；按owner效率原则不新增SHA/MD5逐文件扫描 |
| G5 | per-module delta与cross-decode | completed；`writer_drift_evidence.json` |
| G6 | occupancy、首次分歧、fixed union error | completed with boundary；轨迹与disagreement公开，held expert error不可合法获得 |
| G7 | head-manifold projection与closed loop | completed；`reachability_evidence.json`公开24-task投影误差、family拆分、1200-row paired closed loop和provenance；投影后659/1200、direct 658/1200 |
| G8 | objective/mean/Adam/head/Program matched区分 | completed with boundary；F2/F3/F4/F5分别裁决occupancy disagreement、heads、Program-to-head reachability和aggregation；AdamW保持不变，moment独立效应明确不可由本轮判定 |
| G9 | breadth@1/@5/@10、histogram、suite minimum、top3 | completed并用于所有新panel |

## 11. 最终综合裁决

1. **专家指出的detach是实锤工程缺陷，修复也有真实科学作用。** A/B无credit、C macro1首次有credit均由真实
   functional backward确认；B→C显著增强correct相对wrong/reverse的方向资格。因此不能说修复无用。
2. **它不是123低上限或checkpoint漂移的主要单因。** correct只104→110、breadth不增，C 110→101仍发生33 lost；
   当前no-Text canonical没有达到145、143或151附近。
3. **Text Meta-LoRA的作用是混合的。** 它提供19分correct support，并帮助correct相对reverse；同时更强支撑
   shuffled/keep-first。owner的no-Text边界合理地消除了该不干净路径，但留下必须由其它机制补回的absolute缺口。
4. **当前Writer确实使用视频，但只部分实现高层过程。** C对五种negative都显著，反驳language-only或任意carrier
   充分；然而收益高度task/suite集中、same-row retention<90%，F5又把全局方向和中间顺序分离，不能宣称已得到统一
   跨初始化Program。
5. **简单occupancy divergence未获支持。** fixed-union disagreement方向不符合预言；缺合法validation action
   reference使正确性保持不可判，故occupancy-matched训练分支不适用，而不是被偷偷跳过。
6. **FactorHead co-drift是放大器，decoder reachability不是当前首因。** freeze从84提高到117但仍丢33；fixed-head
   free-Program达到659/1200、与direct expert 658/1200近等价，所以不扩大rank/head/decoder。
7. **cross-task conflict存在但standard PCGrad不是解法。** 它改善早期suite floor、减少churn/lost并强化部分视频
   controls，却显著抑制gained、最终更低且丢失中间顺序优势；Adam moment独立效应仍不可由该matched arm裁决。

因此本轮没有性能pass，也没有预选下一架构。absolute问题最早落在四流到learned Program能否产生跨suite、跨初始化的
policy-effective breadth；稳定性问题落在shared objective/有限长更新能否保留这些方向。两者相互作用，但现有证据已
排除把前端credit、LoRA写出、rank16、FactorHead容量、简单self-occupancy或arithmetic mean任何一个单独当作总根因。
本报告给出的是根因空间的实质收缩，而不是用更好的内部指标包装失败分数。

## 12. 请专家重点复核的问题

1. 你是否同意把原“front-end credit首因”更新为：它是视频内容/箭头方向资格的首因之一，但不是absolute或retention
   首因？B/C完整controls是否支持这一拆分，还是存在更合适解释？
2. C对全部negative显著却高度集中Object、Long reverse反向；F5更均衡却丢keep-first顺序。应如何区分高层过程、
   端点/箭头线索和suite-specific object affordance？仓库措辞是否仍过强？
3. validation expert不存在且held teacher action受信息墙禁止。对F2 action correctness，你是否认可
   `underdetermined-after-audit`，或有不越过信息墙的固定状态reference？
4. F3/F4是否足以撤销“当前FactorHead manifold不可达”的优先假设？free Program oracle还遗漏了哪些会改变该结论的
   deployment约束？
5. PCGrad的早期breadth/causality改善、后期gain抑制和keep-first退化，是否改变你对shared-gradient conflict与Adam
   moment优先级的排序？本轮对Adam moment明确未独立裁决。
6. 请逐项指出ledger中任何遗漏、因果偷换、证据不足或不可复现处；完成复核后，再提出你认为最小、最可证伪的下一项
   证据，而不是直接给出一套大架构。

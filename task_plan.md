# EMBER task plan

更新时间：2026-09-04。

## 当前goal

从最新clean pushed main实现、训练并科学裁决可扩展的Policy-Response Event-to-Factor Writer：以冻结PI0.5的
layer x horizon x probe policy-response和当前正确视频产生的真实native X/Y为视频价值路径，用可重复的Frame、Event与
Factor-Composer blocks一次生成唯一38-target rank16 LoRA；仅依靠正确视频的cross-episode functional与positive process监督，
持续推进component-init、fully-random Final joint及规模化训练，直到validation8 strict paired correct稳定严格大于145/400并满足
breadth、四suite非零、Goal/Long、same-task鲁棒性及冻结后视频因果controls，或者在完成必要正控、matched对照和有信息量的规模化
实验后，获得足以停止当前函数类或EMBER总体路线的可复核证据。

系统goal已于2026-09-02正式建立并保持active，不设置token或阶段工期预算。

## 当前active design

唯一active design为：

- docs/policy_response_event_to_factor_writer_design.md

专家原始依据为：

- docs/expert_review_20260902_full_history_policy_native_meta_writer.md
- docs/expert_review_20260902_policy_response_event_to_factor_writer_clarification.md

PNBTT及此前Program--bank实现均已裁决，不是active fallback。

## 已确认的科学基线

- validation8 source为48/400，task-local rank16 oracle为250/400，输出合同本身有容量。
- held5 stable carrier为43/250。
- G1真实native X/Y、signed pooling与rank4 free-code正式通过：114/250、breadth5/5、Goal2、Long1。
- G2 boundary-anchored ordered Natural Program正式通过：full相对endpoints改善22.2047%，probe38/40、median active
  events 4、same-task/K1/K4通过。
- G3及后续多类Program--bank mapping长期没有学得稳定shared功能映射。
- Program-through-bank、bank-conditioned primal与PNBTT共同证明：旧summary/gate/anchor或whitening/transport函数类不能稳定同时
  保持correct capacity与bank specificity。
- full-rank16 PNBTT在task1与task93呈稳定相反行为，没有证明rank4是当前主要瓶颈。
- 以上负证据不淘汰Stage0原生观测、ordered events、真实native X/Y、signed pooling、rank4或整个ECP。

## 最后设计审查修正

以下四项已并入active design，必须在首个真实图中成立：

1. causal process prediction使用严格prefix-only auxiliary view，不能读取future frames、current-to-final relation或future event
   assignment；
2. process target来自冻结raw evidence或固定teacher projection，可训练compact projection不能同时充当可漂移target；
3. task1/task93的task-local Composer正控与最小smoke一起完成，用于确认去掉PNBTT solve后仍保留G1 bank容量；
4. owner于2026-09-03明确否决coarse/final-layer horizon mean作为active路线；50-step horizon必须完整保留到
   task/relation-conditioned learned read。旧full/coarse对照只保留为历史诊断，不能授权继续coarse。

## 当前执行顺序

1. [x] 逐字归档并核验完整历史专家意见与补充澄清；
2. [x] owner确认Policy-Response Event-to-Factor Writer主案；
3. [x] 建立持续推进goal；
4. [x] 登记active design并校正owner requirements、concept、findings、task plan与progress；
5. [x] 从最新clean pushed main建立唯一codex/policy-response-writer分支和worktree；
6. [x] 盘点并复用observer、events、native X/Y、chunked replay、materializer、J2 data/functional runtime与evaluator，确定唯一
   runtime owner及旧路径退休触发；
7. [x] 实现Frozen Capture接口、Frame Policy-Response Blocks、Ordered Event Blocks、Current-Video Native Factor Composer与
   positive-only objectives；
8. [x] 完成最小真实forward/gradient/materialization smoke及task1/task93两步task-local工程profile；
9. [x] 从clean pushed detached authority正式运行task1/task93 task-local Composer正控并评估step70/110；两task、两checkpoint的
   fit与held-video均自发优于carrier，确认Composer保留可泛化current-bank功能容量；
10. [x] live检查gpu01/gpu02、storage quota与真实吞吐，立即运行component-init 12-task K1 full/coarse matched实验；
11. [x] 已对12-task full/coarse的step70/110全部运行held5 correct-only strict250；coarse为`43/41`，full为`33/31`，
    四个checkpoint均未超过carrier，且Goal/Long全部为0；
12. [ ] correct-only选定并冻结checkpoint后一次性运行same-task-other、wrong、no-video、language-only、first+final、shuffled与
    reversed controls；controls不回流训练；
13. [x] 首轮定位到10-task shared映射的task-disjoint方向与组合覆盖不足；当时曾以coarse、K1扩大到55 meta + 18 target，
    但后续证据表明该参数化仍不能恢复方向或闭环性能，且owner现已明确禁止以horizon mean规避full问题；
14. [x] 原coarse shared信号后的扩展节点未执行，并已由owner的full-only边界及下面的新full执行节点取代；
15. [x] 完成55 meta + 18 target的旧scale component-init训练。macro610 held5 correct250为`26/250`、breadth`2/5`，相对carrier
    retained/gained/lost=`22/4/21`，为显著non-pass；只读policy-effect对照又显示Writer仅`1/5` task优于carrier、四个非零G1
    task的功能方向cosine中位仅`.14753`。完整`s_ref`事后限幅为`33/250`、breadth`1/5`，只局部恢复Spatial0，确认缩放不能补出
    缺失方向。macro1210为`30/250`、breadth`3/5`、Goal/Long仍为0；相对macro610高churn净增4但相对carrier仍净丢13，完成对其
    实际未限幅/global-clip/static-slot parameterization的相邻裁决；
16. [x] 完成per-target `B@A` RMS cap、独立gradient clipping与dynamic-value合同实现和真实smoke；原corrected coarse shared/task-local
    在owner更新方法边界后分别于step121/47/29主动中止，不形成checkpoint裁决，也不再续跑；
17. [x] 只用full裁决corrected shared。task1/task93 task-local已完成且保持跨视频容量；首个shared在审计发现
    Composer辅助bank context仍提前平均horizon后于macro前主动停止。现已改为对完整frame/probe/horizon/bank-type
    tokens做exact chunked learned attention并通过最长视频真实profile。owner于2026-09-03要求架构未证明前不付出10小时扩展成本，
    因此73-task x 1210-step fresh运行在optimizer25/effective15主动停止，不形成科学裁决。先在同一clean pushed commit上
    运行12-task x 110-step四卡资格实验并完成macro70/110 held5 correct-only strict250；两点均为`35/250`，breadth分别`3/5`与
    `2/5`，Goal/Long均为0，稳定低于carrier `43/250`，所以不恢复旧参数化73-task长跑、不运行其负controls；
18. [x] 修正Process到Composer最早失效接口：首个relation-summed scoring已完成12-task相邻裁决，held5 macro70/110为
    `42/34`、breadth均`3/5`且Goal/Long为0，两个true-task-held继续为负，因此该“先跨event求和、后非线性score”的具体接口
    non-pass。当前只把soft `alpha(e,t,m)`改为event x relation候选的base measure，以未求和`D(e,j)`产生动态logit；full
    50-horizon、真实X/Y、positive-only loss、rank12+4和唯一rank16不变。显式枚举等价测试与最长task93 formal-rows16真实profile
    已通过，单步`8.93/8.20s`且峰值reserved `46.43GB`；clean pushed detached短资格的110步及两点Panel-B已完成。
    gradient task的m110 fit/held recovery为`.1365/.1319`，但true-task-held仍`0/2`且为负。held5 macro70/110为
    `40/42`，breadth均`3/5`、Goal/Long均为0，m110相对carrier43为`7 gained/8 lost`。该matched接口稳定non-pass；
    70/110是10步warmup后effective 60/100的历史可比节点，不是不可调整的理论步数；
19. [ ] 吞吐线继续并行推进：第一阶段已完成选择性CPU evidence cache复制、outcome-independent动态task放置、
    dense/streaming exact bank attention、整视频signed pooling与output-group归约，四卡10-step由`34.39s/step`降至`4.05s/step`
    （`8.48x`）；relation scorer等价收缩又使task93快约`36%`。microbatch4、CPU activation offload和gradient packing经实测
    收益不足或明显变慢而淘汰；Evaluator当前保留实测更快的`3 replicas x 8 envs`。选择性8GiB cache复制已把当前event-measure
    完整110步均值为`10.06s`且四卡稳定段平均SM约`88--90%`；`4 * functional_rows + sampled_frames`新cost模型在旧110步
    真实回放预计再省`2.30%`，已通过22项测试、集成并推送，且不改变当前frozen run的逐步分配。
    73-task profile又确认两步均为每卡3 tasks；Panel-B后续已从training-cache ownership解耦，按完整视频与functional工作量把
    本轮预期`2/4/5/1`任务布局改为`3/3/3/3`。microbatch2/8实测只差约1%，保留2。node-local单份safetensors mmap又使
    每个local rank无需物理复制即可读取全部105.02GB frozen evidence；同commit、同两卡7-step schedule相对0GiB/8GiB
    private cache由`21.37/18.54s`降到`17.81s/step`，相对当前8GiB方案平均快`4.05%`、最坏step快`24.4%`，rank实际工作
    gap由`3.12s`降到`.34s`。当前四卡rows16的126步真实task timing反事实估计约可再省`21--23%`。继续以真实phase timing、
    峰值显存、rank idle tail与Evaluator LoRA/s定位剩余瓶颈，彻底优化可复用执行层，但不得等待性能工程完美才获取阶段科学结果；
    task batch与meta/target比例始终由实验配置决定；
20. [x] 已按专家的task-disjoint失败映射完成train24 + 审计non-held meta tasks的factorial coverage审计。73个gradient tasks中，
    7组同语言跨场景组合有5组包含至少两个gradient tasks、4组形成gradient-to-held桥；三类人工protocol contrast分别有
    `5/9/5`组train pair与`3/7/3`组held桥。task2、task74及held Spatial/Object/Long均有可见component重组依据；held Goal的
    `push` procedure没有任何Writer-gradient peer，是明确覆盖缺口。故数据并非完全欠识别，允许最小full扩展映射，但审计本身
    不证明video-dependent最优adapter已经可学；
21. [x] 已用当前单节点全部4张安全空闲卡完成73-task、每update 12 tasks的完整event-measure两步真实profile，并从clean pushed
    detached authority完成full K1 component-init扩展资格。55 meta + 18 target近似task等权，每步显式`9 meta + 3 target`
    只属于该配置；optimizer step200/400各task约`32--34/65--67`次暴露。m200/m400 Panel-B的gradient fit/held benefit虽由
    `.000740/.000316`升到`.001023/.000547`，两个true-task-held均值仍为负；held5 strict250仅`30/32`，breadth`2/5`与
    `3/5`，Goal/Long均为0。m200到m400为`20 retained/12 gained/10 lost`，相对carrier43的m400仍为
    `27/5/16`，故数据扩大和训练加倍没有解决shared闭环；
22. [ ] 已把最早失效接口定位为Composer query seed的数值rank坍缩：raw相加时约`67`范数的Process common压过约`1`范数的
    rank token，m200/m400的实际mobile update均接近rank1。只对rank context与shared task context分别做无参数LayerNorm后相加，
    冻结m200反事实恢复rank query与部分有效factor谱；不新增loss、正交约束、solve或网络分支。修正已通过定向测试、真实
    forward/gradient/materialization smoke并合并推送。当前从clean detached `3e589695`运行73-task fresh m200/m400 shared资格；
    task1/task93的两次启动在任何optimizer step前依次暴露缺少正控声明及旧单panel合同假设，均不构成科学结果。`89ca865d`已让
    task-local v2合同直接封存resolved task panel并通过27项Writer测试和配置全字段预检。两条正控已从clean detached
    `ef00f446`完成；macro70/110的task1 fit/held为`.2224/.1153`、`.3283/.2282`，task93为`.3047/.3115`、
    `.3570/.3223`，四点全部视频优于carrier并满足信息墙，确认修正后的task-local容量。全线继续使用full 50-horizon、
    positive-only、唯一rank16及node-local单份mmap，当前等待shared m200/m400；
23. [ ] full shared信号成立后进入mixed-K、fully-random fresh Final joint和validation8相邻single-checkpoint strict paired400；
24. [ ] selected checkpoint冻结后补齐最终因果controls；只有base Writer稳定且剩余错误集中在action detail时才评估Action Meta；
25. [ ] 达到最终合同，或在完整信息量证据下形成当前函数类乃至EMBER总体停止裁决。

## 推进与决策原则

- 效率优先：实现接通后立即获取真实GPU与closed-loop证据，文档、通用重构和冗余测试不能延迟科学结果。
- 缜密修正：每次架构改变必须对应已定位的最早接口，不以随手改结构、LR/seed/width/rank/scale小扫替代分析。
- 不作弊：只用正确视频训练；validation/test无梯度；负controls只在checkpoint冻结后评测。
- 一次non-pass只淘汰实际检验的组合，不因局部失败推翻全部正证据。
- 明确坏结果不无限续训；有新的机制证据时也不受人为版本次数限制。
- closed-loop absolute性能优先，内部loss、factor cosine、attention和representation指标只负责定位。
- Final必须包含component-init与fully-random同拓扑fresh候选。
- GPU launch前同时live检查gpu01/gpu02；有1至6张有效A40就使用，不等待凑卡，可安全共驻但不干扰他人。
- 长训练期间优先完成cache、分析、评测准备和下一科学节点；只有没有推进相关的实质性工作时才做可中断的增量workspace清理，
  训练或评测结果一到立即停止清理。

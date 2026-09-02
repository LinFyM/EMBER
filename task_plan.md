# EMBER task plan

更新时间：2026-09-03。

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
4. full-response与coarse-response两臂都读取完整dynamic X/Y bank，因此对照只裁决前端event representation的增量。

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
13. [x] 首轮最早接口定位为10-task shared映射的task-disjoint方向与组合覆盖不足；full前端没有提供增量且更破坏carrier，故首个
    有依据修正固定为coarse、K1并把自然梯度映射扩大到55 meta + 18 target，其余模型与loss不变；
14. [ ] shared信号成立后混合K属于1、2、4，完成component-init与fully-random fresh matched Final joint；
15. [ ] 完成55 meta + 18 target的scale component-init训练。macro610 held5 correct250已完成为`26/250`、breadth`2/5`，相对carrier
    retained/gained/lost=`22/4/21`，为显著non-pass；只读policy-effect对照又显示Writer仅`1/5` task优于carrier、四个非零G1
    task的功能方向cosine中位仅`.14753`，排除“只需缩小scale”。训练继续到预注册macro1210作相邻裁决。只有shared信号成立后才进入
    mixed-K、fully-random及validation8相邻single-checkpoint strict paired400；
16. [ ] selected checkpoint冻结后补齐最终因果controls；只有base Writer稳定且剩余错误集中在action detail时才评估Action Meta；
17. [ ] 达到最终合同，或在完整信息量证据下形成当前函数类乃至EMBER总体停止裁决。

## 推进与决策原则

- 效率优先：实现接通后立即获取真实GPU与closed-loop证据，文档、通用重构和冗余测试不能延迟科学结果。
- 缜密修正：每次架构改变必须对应已定位的最早接口，不以随手改结构、LR/seed/width/rank/scale小扫替代分析。
- 不作弊：只用正确视频训练；validation/test无梯度；负controls只在checkpoint冻结后评测。
- 一次non-pass只淘汰实际检验的组合，不因局部失败推翻全部正证据。
- 明确坏结果不无限续训；有新的机制证据时也不受人为版本次数限制。
- closed-loop absolute性能优先，内部loss、factor cosine、attention和representation指标只负责定位。
- Final必须包含component-init与fully-random同拓扑fresh候选。
- GPU launch前同时live检查gpu01/gpu02；有1至6张有效A40就使用，不等待凑卡，可安全共驻但不干扰他人。
- 长训练期间并行完成cache、评测准备和已确认workspace清理。

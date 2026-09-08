# EMBER task plan

## 当前目标与协作安排

2026-09-08 Owner已发出接管启动指令：全面理解仓库，完整实现当前设计，开展正式实验，依据真实结果持续迭代直到性能和科学要求达标。
全程goal已创建且active，无token预算、总工期或总尝试次数。前次只做交接准备的session限制已结束。
全面阅读与证据审计已完成；当前阶段为接受停滞的数值接口定位；八档回溯仍1/4接受，停止同样续试，尚未进入formal；既定边界内无需再次请求实施、训练或方法修订批准。

唯一 active design：[过去定向完整 Horizon Writer](docs/horizon_relation_video_writer_design.md)。
末层完整H → 过去4帧对应 → 完整H-query → 两端Z核实 → 历史u有序GRU → 四组单向长程交替/前三回写
→ 集合compiler → 完整native A/B；fresh Writer/Meta、FM辅助真实Writer RL、同版本单次联合更新。
**长程四组全部past+self，H-query沿H双向；专家原文中的双向长程已被Owner覆盖。**

完整Horizon过程图、native D、物化/评测已集成；同版本FM/RL训练接口与runner已集成、真实采集及完整checkpoint已验证；首profile因数值batch自比较超阈值而0/2接受，最小修复已由真实重放验证，新方案尚无正式checkpoint或qualification行为证据。旧train24永久止于384，旧P/Q width256不补评，
未合并native-factor-readout只作待审视草稿，不整支集成或恢复旧对照。详情见[progress](progress.md)。

## 接续执行计划

1. **准备与方法登记（前次完成）。** 对齐Owner要求、concept、完整设计、专家原文和裁决、历史索引与临时HANDOFF；
   检查已有代码和关键资产，提交推送main。只做文档对应的验证，不执行新训练或用旧GPU快照占位。
2. **创建goal、全面理解并实现唯一canonical路径（已完成首版）。** 按Owner要求建立覆盖正式实验、迭代和性能达标的goal，不填token预算。
   系统阅读整个仓库的源码/测试/脚本/配置/文档及历史和原始证据，完整阅读最终架构，不能只看摘要就动手。
   从最新main开始；按并发/结构风险需要隔离。
   在既有owner内落实post-norm PreActionOut与最终Z、过去对应/完整H-query/GRU、四组因果long和回写、native heads，
   同步训练/物化/配置/checkpoint schema。复用真实FM/VJP/evaluator，新增明确的Gaussian动作探索与10步可微重放。
   首个正式实现不允许静默省略、缩水或偷换关键模块/训练合同；逐步替换旧可执行路径和对应测试，完成后及时集成push。
   不先起平行fallback或仅合并旧native-head草稿。后续凭真实结果改进，不能用表面补丁掩盖未解决的问题。
   若充分已有证据或实际问题要求修订，先说明理由并更新正式合同再完整实现；不为套用初稿强行运行已知错误配置。
3. **真实机制与成本。** 按design §8.1验证必要依赖与信息墙、权重、采样版本和梯度；用最长真实K1/K4、FM与rollout/replay测吞吐和峰值。
   功能检查与学习节点不是冻结课程。所有GPU/大输出操作前执行相应live两节点与独立quota检查，旧资源预算不继承。
4. **结果前冻结首个run合同并开始有效学习。** 使用design §7的数据、采样、优化与RL默认；profile后确定实际batch/chunk/topology、
   命令/output root/新schema、首段checkpoint与strict400节点、继续/止损规则。首段以24个接受更新及训练侧Sigma/0配对诊断为起点；
   不默认长跑，不把旧672 schedule或旧48/96节点原样搬来。具体资源与成本依赖现场，接班者自主落实，不形成新的人工审批门。
5. **共享学习、迁移与稳定性。** 分开判断训练task、同task新视频、未见task及噪声去除后的行为。
   有信息量且出现广泛能力时及时strict400；每个关键点报告task/suite/breadth/RGL/churn/相邻及跨视频重合与真实曝光。
   好趋势继续到足以判定相邻稳定；明确non-pass及时定位最早接口，禁止无依据rank/seed/LR/scale/dtype小扫或无限续训。
   若有信息量学习后仍只略超source或长期在SFT附近，按design §1.1视为严重能力缺口；查历史等价尝试，允许有证据的实质机制调整，
   不把这种落差当成只差一点小调参，不用基线rank不同或内部指标改善作为免责。
6. **正式选择与最终交付。** 只按预登记single-checkpoint qualification与相邻稳定选点；selected冻结后做必要视频controls，
   shuffled/reversed最后测试且不反哺设计；方法冻结后按32/8合同fresh训练与Test，交付可复核代码和formal evidence。

## Done when

唯一正式性能线是validation8 strict paired correct>145/400，同时达到design §8.3与AGENTS的相邻稳定、低churn、
高breadth、四suite非零、Goal/Long贡献、same-task视频鲁棒性及最终因果要求。实现完成、loss下降、训练结束或单次高分都不算完成。
新session按Owner要求设置并持续推进goal，不自行设置token预算、总工期或总尝试数；只有真实权限/资源/信息边界才回到Owner。

## 需要刷新、首版选择与自主修订

- **需要现场刷新：** Git/concurrent work、两节点GPU/process、strg01各filesystem quota、真实个人用量、实际吞吐与新run预算。
- **实施时自主确定：** cohesive源码owner、统一新schema/入口配置、profile后的执行batch、首次结果前的科学节点与续训规则。
- **首版已定：** 末层而非18层、多K独立保序、过去4帧、完整H-query、短GRU、四组过去单向long及逐H回写、完整native输出；不以无依据疑问重开讨论。
- **后续可自主修订：** 在总体思想和硬合同内，根据充分证据改变具体模块、监督/RL或优化机制，必要时重构，记录假设/证据/合同与验证。
- **不能按历史自动恢复：** 384后的旧schedule、旧P/Q width256补评、旧分层native-head-only对照或专家原文双向long；不能突破teacher信息墙。

正式train/eval须来自clean pushed commit的detached frozen worktree；canonical代码集成main并push。
只保留一套当前实现，历史通过Git、sealed artifacts和research_history追溯；不覆盖dirty草稿或删除唯一资产。

# EMBER task plan

更新时间：2026-09-02。

## 当前目标与状态

EMBER最终目标仍是：从exact language与一条或多条same-task、action-hidden、ordered teacher videos，在rollout前一次性生成唯一一套
38-target rank16 LoRA，并在固定validation8取得稳定、具备breadth、Goal/Long贡献、same-task鲁棒性和视频因果性的strict paired
correct `>145/400`。

owner已正式采纳2026-09-02全局专家复核的主选A，并建立active goal：以PNBTT替换已停止的Program--bank接口，依次完成
E0--E4与matched whole-Writer joint adjudication，效率优先，不增加专家未要求的流程性Gate，持续推进到满足最终合同或出现
经过充分尝试仍无法继续的真实阻塞。

当前active design为`docs/program_conditioned_native_bank_tangent_transport_design.md`。实现必须从最新clean pushed `main`建立唯一
`codex/pnbtt`分支/worktree；最小真实forward/gradient/materialization smoke后立即进入E1/E2，不用全量测试、通用重构或文档整理
阻塞科学结果。GPU只在实际launch前同时live检查gpu01/gpu02。

## 当前科学停止点

- G1 native-factor free-code已正式通过：held5 strict250为`114/250`，breadth5/5、Goal2、Long1；真实native X/Y、signed pooling、
  rank4 residual与唯一rank12+4 rank16存在强task-local闭环容量。
- G2 boundary-anchored Natural Program已正式通过：held full相对endpoints改善`22.2047%`，probe`38/40`，median active events`4`，
  same-task、K1 identity与K4 permutation invariance均通过。
- G3长期未通过共享Program--bank映射。完整逐架构证据见`docs/research_history.md`第21--96节。
- 第七次专家的Program-through-bank链已完整执行：scope-matched free-summary S0通过；real Program-through-bank S1在task1/93的
  correct/held上正式non-pass，因此按预注册条件未启动shared S2。
- 其失败分支bank-conditioned primal恢复了correct容量，但原query、calibrated Q_free、base-LR A_free和充分校准A_free均不能同时
  保持correct并压低wrong。最终task93 correct fit0/fit1/held为`.853296/.858892/.818467`，wrong为`.611592/.668511`，
  wrong与margin正式non-pass。
- 当前只停止`summary -> family-scalar gate -> shared event-additive anchor`这一具体参数化；不外推为Program schema、Stage0、
  native X/Y、signed pooling、rank4或整个ECP失败。

## 当前执行顺序

1. [x] 逐字归档第八次专家原文并核对其引用的代码、配置、提交和formal artifacts；
2. [x] owner确认主选A/PNBTT并建立持续推进goal；
3. [x] 登记active authority，建立唯一`codex/pnbtt`实现worktree；
4. [x] 实现PNBTT唯一运行面：Program query、real-bank key、differentiable key whitening、joint-K signed real-value transport与rank4
   materialization；退役旧base/gate/anchor在deployment主路径中的所有权；
5. [x] E0最小真实smoke：专家十项hard tests、chunked replay、finite forward/gradient、38-target rank4与唯一rank16真实policy load；
6. [ ] E1 task1/93 topology-matched free-query transport capacity；macro70 Panel-B因correct/held不足为`non_pass`，wrong、all-pairs、
   near-bound与信息墙大部分成立；同run已原拓扑resume到step110，再由两个相邻checkpoint联合裁决；
7. [ ] E2真实frozen G2 Natural Program到bank transport，含K1/K2/K4、same-task与full-vs-language/endpoints；
8. [ ] E2通过后立即把E3作为whole-Writer run的早期shared资格；若E1通过而E2系统性失败，严格触发专家B路线；
9. [ ] E4 matched component-init与fully-random whole-Writer joint，最终由validation8 strict paired400合同裁决；
10. [ ] 只有matched两臂都无法形成稳定闭环增量时，才进入ECP/zero-interaction根本停止讨论。

实现参数`m/lambda/epsilon/theta`、projection分解、solver、cache shard、microbatch和GPU数量由吞吐与train-side机制证据选择，不升级为
额外科学Gate。明确失败不得靠seed/LR/width小扫或无限续训挽救；有新机制证据时也不受人为版本数或修正次数限制。

## 继续推进时仍有效的边界

- validation/test不产生梯度；shuffled/reversed只在最终selected checkpoint冻结后测试；
- source、信息墙、唯一rank16与Action Meta默认关闭的合同保持；
- G1--G3只是机制验证，不是Final强制课程；Final必须保留component-init与fully-random Writer两个matched fresh候选；
- non-pass先定位最早接口，不用内部loss、cosine、reconstruction或无意义超参小扫代替closed-loop；
- GPU、吞吐、并行、Git、formal worktree与storage要求以`docs/current_owner_requirements.md`和`AGENTS.md`为准。

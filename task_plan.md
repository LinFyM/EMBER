# EMBER task plan

更新时间：2026-09-02。

## 当前目标与状态

EMBER最终目标仍是：从exact language与一条或多条same-task、action-hidden、ordered teacher videos，在rollout前一次性生成唯一一套
38-target rank16 LoRA，并在固定validation8取得稳定、具备breadth、Goal/Long贡献、same-task鲁棒性和视频因果性的strict paired
correct `>145/400`。

owner已正式采纳2026-09-02全局专家复核的主选A，并建立active goal：以PNBTT替换已停止的Program--bank接口，效率优先地
推进E0--E4与matched whole-Writer joint adjudication，直到满足最终合同或出现经充分尝试仍无法继续的真实阻塞。

PNBTT E0和专家允许的E1容量修订序列现已全部完成；最后一次同构task-local full-rank16 oracle仍在macro70/110相邻一致
`non_pass`，且没有相对rank4呈现专家要求的“明显更优”。因此`docs/program_conditioned_native_bank_tangent_transport_design.md`
作为刚结束的科学authority保留，但当前没有获准的新active design或implementation route。E2、次选B和whole-Writer D的
专家前置触发条件均未满足；下一动作是将这一formal证据返回专家，或由owner明确扩展方法authority，不自行发明新路线。

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
- PNBTT family-key v2也已在macro70/110相邻一致`non_pass`：它显著压低wrong，但correct/held容量仍不足。该结果只停止当前
  family-shared nonlinear key chart + target-specific rank16 key residual + rank4 transport，不裁决Natural Program或整个PNBTT。
- 唯一full-rank16 oracle也在macro70/110稳定`non_pass`。macro110上task1 correct fit0/fit1/held为
  `.960297/.941644/.948351`，但wrong为`.634156/.711548`；task93 wrong为`-.006466/-.021862`，但correct仅
  `.586174/.595686/.449605`。这一任务依赖的容量--特异性反转在70到110保持，没有呈现一致广泛的明显改善，
  所以按专家§5.10停止rank扩展，不进入E2。
- 这是当前路线/authority阻塞，不是整个ECP的根本停止证据：E1冻结Natural Program，因而未形成“PNBTT容量已通过但真实
  Program系统失败”的B路线触发条件；matched whole-Writer两臂也尚未被允许运行。

## 当前执行顺序

1. [x] 逐字归档第八次专家原文并核对其引用的代码、配置、提交和formal artifacts；
2. [x] owner确认主选A/PNBTT并建立持续推进goal；
3. [x] 登记active authority，建立唯一`codex/pnbtt`实现worktree；
4. [x] 实现PNBTT唯一运行面：Program query、real-bank key、differentiable key whitening、joint-K signed real-value transport与rank4
   materialization；退役旧base/gate/anchor在deployment主路径中的所有权；
5. [x] E0最小真实smoke：专家十项hard tests、chunked replay、finite forward/gradient、38-target rank4与唯一rank16真实policy load；
6. [x] 首个single-key-chart E1 task1/93已在macro70/110得到相邻一致`non_pass`：all-pairs、near-bound、信息墙及大部分wrong
   约束成立，但correct/held与`.50` margin稳定不足；
7. [x] 按专家§5.10在task1/93、Panel-A三条gradient arms和step110上完成`T=Cov(v,k)`功能梯度投影谱：`m=128`
   没有截断有效谱，但q/v各side的功能梯度保留率与correct/wrong operator几何显示单一线性key坐标不足；
8. [x] 保持`m=128`、residual rank4及全部E1数据/loss/Gate不变，完成family-shared nonlinear trunk + target-specific rank16
   low-rank key projection的fresh E1；macro70/110均为`non_pass`，wrong明显改善但correct/held稳定不足；
9. [x] 在v2 macro110上复跑同一train-only tangent spectrum：`m=128`仍未截断有效谱，family chart只在部分output side降低
   correct--wrong operator重合，没有恢复q/v correct功能梯度可达性；停止继续修改key chart或增加`m`；
10. [x] 完成专家允许的唯一同构PNBTT task-local full-rank16 oracle；macro70/110相邻一致`non_pass`，稳定呈现task1高correct/
    高wrong与task93低wrong/低correct的反转，未相对rank4明显更优，因此不重开carrier/task rank分配；
11. [ ] E1通过后才能进入E2真实frozen G2 Natural Program到bank transport；当前E1未通过，本步未触发；
12. [ ] E2通过后才能进入E3；只有E1容量通过而E2真实Program系统失败才触发次选B，当前未触发；
13. [ ] E4 matched component-init与fully-random whole-Writer joint只在上游Gate满足后运行，当前未获准；
14. [ ] 只有matched两臂都无法形成稳定闭环增量时，才进入ECP/zero-interaction根本停止讨论。

实现参数`m/lambda/epsilon/theta`、projection分解、solver、cache shard、microbatch和GPU数量由吞吐与train-side机制证据选择，不升级为
额外科学Gate。明确失败不得靠seed/LR/width小扫或无限续训挽救；有新机制证据时也不受人为版本数或修正次数限制。

## 继续推进时仍有效的边界

- validation/test不产生梯度；shuffled/reversed只在最终selected checkpoint冻结后测试；
- source、信息墙、唯一rank16与Action Meta默认关闭的合同保持；
- G1--G3只是机制验证，不是Final强制课程；Final必须保留component-init与fully-random Writer两个matched fresh候选；
- non-pass先定位最早接口，不用内部loss、cosine、reconstruction或无意义超参小扫代替closed-loop；
- GPU、吞吐、并行、Git、formal worktree与storage要求以`docs/current_owner_requirements.md`和`AGENTS.md`为准。

# EMBER task plan

更新时间：2026-09-01。

## 当前目标与状态

EMBER最终目标仍是：从exact language与一条或多条same-task、action-hidden、ordered teacher videos，在rollout前一次性生成唯一一套
38-target rank16 LoRA，并在固定validation8取得稳定、具备breadth、Goal/Long贡献、same-task鲁棒性和视频因果性的strict paired
correct `>145/400`。

当前没有active实现、训练、评测或GPU任务，也没有active goal。owner已把远程
`main@a185fe223d1ef77635d83696c3e164a48520edbf`交给一位此前未参与项目的全新专家做全历史与路线复核。专家正在判断：

1. 继续ECP Native-Factor并替换已定位的Program--bank联合方向接口；
2. 保留部分Native-Factor证据但另开新的Writer架构分支；
3. 直接进行更整体的端到端Writer裁决；
4. 现有证据是否已接近ECP或zero-interaction合同的根本停止条件。

专家回复返回并经owner确认前，不创建新架构版本、不启动formal、不修改实验配置、不占用GPU，也不自行恢复旧路线。

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

## 等待专家期间

- [x] 把最新实现、结果、七次专家原文、96节研究历史和停止边界推送远程；
- [x] 向全新专家提交锁定`a185fe2`的全历史复核prompt；
- [x] 收敛当前计划/进度文档，记录owner长期效率、GPU和协作要求；
- [x] 清理已完成worktree、local task branch与`.codex/tmp`，保留formal evidence、checkpoint、dataset、cache authority与Git历史；
- [x] 建立一次性`HANDOFF.md`供新session消费；
- [ ] 等待owner转交专家原始回复。

## 专家回复后的唯一顺序

1. 将专家原文完整、逐字保存到新的`docs/expert_review_*.md`，不改写历史原文；
2. 对照专家引用的文件、提交、代码和结果复核其事实与归因，区分专家明确规定、owner裁决和实现建议；
3. 先向owner给出通俗核心结论、分歧和推荐路线，由owner确认方向；
4. 只有owner授权后，才更新active design与本计划，从最新clean pushed `main`建立唯一`codex/*`实现面；
5. 新路线仍必须从最早可证伪接口开始，先做最小真实smoke，再迅速启动有信息量的实验；不得恢复旧Writer、GOMQ、PECS、v24、
   quotient、surrogate/polish或同类scalar-gate小修，除非新证据明确要求。

## 继续推进时仍有效的边界

- validation/test不产生梯度；shuffled/reversed只在最终selected checkpoint冻结后测试；
- source、信息墙、唯一rank16与Action Meta默认关闭的合同保持；
- G1--G3只是机制验证，不是Final强制课程；Final必须保留component-init与fully-random Writer两个matched fresh候选；
- non-pass先定位最早接口，不用内部loss、cosine、reconstruction或无意义超参小扫代替closed-loop；
- GPU、吞吐、并行、Git、formal worktree与storage要求以`docs/current_owner_requirements.md`和`AGENTS.md`为准。

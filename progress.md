# EMBER progress

更新时间：2026-08-24。分支：`main`。本轮科学起点：`7ab5a04`；仓库瘦身基础：`6fdaeb8`；最终交接提交见当前Git HEAD。

## 当前状态

专家回复已经收到、完整阅读并固化。active design现为`docs/event_conditioned_policy_compiler_design.md`中的
**ECP Native-Factor Compiler**，核心架构、数据角色、阶段Gate、最终controls和停止条件均已明确。

本轮没有启动GPU训练、没有实现新Writer、没有向专家发送任何消息。当前有意停在跨session交接点；下一session消费
`HANDOFF.md`后才开始G1 capacity oracle。

专家复核锁定的是远程`main@7ab5a04`。其后`6fdaeb8`只删除退役代码/人工资产并整合文档，没有新增实验结果；专家指出的当前
Stage 0实现缺口已在瘦身后的代码中复核：q/v owner仍来自layer input/residual，尚无真实38-target input/output hooks。因此该科学
裁决可直接应用于当前活动树。

## 专家裁决已固化

- ECP继续推进，名称细化为ECP Native-Factor Compiler；
- 取消neural `q_pi -> fixed effect-code realizer -> LoRA`前置链；
- privileged experts/effects只作nonparametric set-valued training critic；
- Video Program固定为owner-specific language/scene/ordered events及`rho/tau/sigma`；
- 第二pass读取38个target的真实native inputs/outputs与动态differences；
- Program通过signed pooling产生mobile rank4，与frozen rank12 carrier拼成唯一rank16；
- 当前唯一下一步是fold0 held5 task-local free-code strict250，不先训练fresh Program/compiler；
- 通过后依次进行Natural Program、frozen-Program shared compiler、joint Writer、conditional outer credit和final fresh；
- validation8与完整video controls的资格门、Test8 sealed规则及ECP根本失败条件均已固定。

唯一保留的后期政策差异：专家建议Action Meta只有明确净收益才启用；owner此前要求必须matched尝试且无负面即可启用。它不阻塞
当前G1--G3，执行到该门时按owner最新指示。

## 本轮仓库整理结果

- 退役Writer、functional decoder、ECP v1--v24后继、MDCO/PECS、fixed/two-sided realizers与人工process模块已删除；
- evaluator保留source/task-expert adapter、dynamic queue、occupancy diagnostics和strict aggregation；
- canonical基础模块为source/corpus/SFT、LoRA、task experts、Stage 0、policy effects、functional loss、reward/occupancy与evaluation；
- 旧41份Markdown、87份分散证据JSON、退役配置/测试及约11.6GB人工datasets/runs已清除；
- 瘦身提交`6fdaeb8`的126项活动CPU测试、compile、脚本入口与引用审计均通过；
- 当前只有`main`一个worktree，无task-owned branch或GPU job。

## 当前可复用资产

- 固定24/8/8 split、71-task source corpus与五fold meta/target manifests；
- frozen source PI0.5 authority、rank16 LoRA topology/materialization；
- task-expert bank、independent successful members、mobile-rank4解析容量与effect calibration；
- Stage 0 v3 full-layer/horizon observer、transition matcher、event binding/segmenter；
- cross-episode video/action schedule、functional flow loss与detached LoRA gradient bridge；
- natural reward rollout、occupancy capture、BDDL progress与cost-balanced strict evaluator；
- ignored `runs/`中的唯一formal checkpoints、raw rows和aggregate。

## 下一实现缺口

活动树尚无：

1. 38-target native linear input/output hooks；
2. absolute/adjacent/init/goal banks的chunked online accumulator；
3. Program-conditioned signed rank4 compiler与scale/SVD canonicalization；
4. task-local free-code optimizer；
5. G1 four-arm strict250 wiring与Gate report。

这些组成下一session的唯一实现面。先做最小真实smoke，再完成G1；不得恢复旧realizer、建立平行版本或跳到fresh Stage 0/joint。

## 交接动作

1. 当前session完成文档一致性与Git审查，提交并推送；
2. owner开启新session；
3. 新session完整读取mandatory docs与`HANDOFF.md`，核对当前HEAD；
4. 删除已消费的`HANDOFF.md`并提交；
5. 按`task_plan.md`从G1开始自主推进，只在关键Gate、显著跃升或真实权限/路线歧义时暂停。

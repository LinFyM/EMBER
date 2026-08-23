# ECP Mobile-Rank4 Residual Closed-Loop Capacity Diagnostic

状态：2026-08-23 **在formal adapter materialization与GPU rollout前预注册；现已完成并裁决为mixed。**

## Scientific question

fixed-A已经被闭环判为capacity-binding。当前唯一问题不是再改effect objective或直接写一个free-A/B solver，而是：

> 已验证的rank12 stable carrier加一个行、列空间均可移动的rank4 effective correction，是否足以保留三个已知成功
> task-local policies的闭环行为？

这直接检验专家建议的“strong carrier + small effective-update correction”能否成为下一realization参数化。它不训练
Program、compiler、Writer或solver，也不检验action-hidden video inference。

## Why this question comes next

stable carrier的38个targets都具有同一精确结构：`B[:, 0:12]`有效、`B[:, 12:16]`为零，`A`的16行均为有限且
full-row-rank。历史`shared12 + task residual4`的learned phase decoder只得到`37/33`，低于carrier的`43`；该结果关闭了
当时的code/objective/decoder组合，但没有回答任意mobile rank4 residual的容量。

在本卡冻结前完成的零训练几何审计显示，对latest、independent、earliest三个members和held5共15个matched adapters，
每target最佳rank4 `expert-carrier` correction的聚合energy coverage为`99.49%--99.69%`，最终effective update相对expert的
energy coverage为`95.34%--98.90%`。这些数值只说明值得做闭环容量门，不能选择member、task或checkpoint，也不能替代
closed loop。

## Exact parameterization

每个target把stable carrier的前12 ranks保持不变：

```text
W_carrier = B_c12 A_c12
C = W_expert - W_carrier
C_4 = argmin_{rank(X) <= 4} ||C - X||_F = U_4 S_4 V_4^T
B_r = U_4 sqrt(S_4)
A_r = sqrt(S_4) V_4^T

B_final = [B_c12 | B_r]
A_final = [A_c12 ; A_r]
W_final = W_carrier + C_4
```

因此carrier与correction在effective-update坐标中严格相加，不存在carrier/residual raw-factor cross term；最终仍是一套38-target
rank16 LoRA，不部署第二adapter、rank32 union或LoRA平均。未来zero-process route可令`B_r=0`并直接返回原carrier state，
精确恢复carrier behavior。本诊断只物化非零的解析最优capacity adapters。

实现使用已有thin-QR/core-SVD低秩规整，不物化完整dense weight matrices，不使用optimizer、reward、action、video、
interpolation或checkpoint/member选择。

## Frozen arms and rows

只运行以下三个预先固定的matched arms，每arm使用其原strict250 rows：

1. `latest -> carrier + best-rank4(expert-carrier)`；
2. `independent -> carrier + best-rank4(expert-carrier)`；
3. `earliest -> carrier + best-rank4(expert-carrier)`。

matched direct、carrier和刚完成的fixed-A projection均只作既有paired references。总计750个新rollout rows；三个arms并行完成，
不先跑一个arm后决定其余arms，不挑member，不评估validation8或Test。

## Required reports

每arm及pooled必须报告：

- absolute score和逐global `0/9/18/25/36`；
- 相对matched direct的retained/gained/lost、success retention与Jaccard；
- 相对carrier43的retained/gained/lost；
- Goal与Long逐member retention；
- correction/expert energy coverage与closed-loop retention的相关性只作定位；
- episode key、environment seed、policy seed root、language与policy-noise common-prefix pairing；
- single-rank16、finite tensors、worker return codes和physical GPU allocation。

## Pre-registered adjudication

- **capacity-supported**：pooled matched-direct success retention至少`70%`；Goal与Long各自pooled retention至少`50%`；且
  Goal、Long各自至少两个members非零。只有该结果才授权另写一张卡，用同一48-state三particle objective求mobile rank4
  residual；它不直接授权Stage 1C或Writer训练。
- **capacity-binding**：pooled retention不高于`50%`；或者Goal/Long任一pooled retention不高于`25%`且至少两个members为0。
  这会停止rank12+mobile-rank4主线，下一问题才是full-rank16 effective-additive retraction。
- **mixed**：介于两者之间。暂停分析最早失效的target/suite；不自动实现residual solver，也不通过rank、插值或member选择救结果。

不论结果如何，不能把本诊断的privileged expert adapters作为deployment route，也不能用它选择video Writer checkpoint。

## Execution boundary

formal materialization与rollout必须来自clean pushed commit的detached frozen worktree。launch前live检查gpu01与gpu02；gpu01
physical0为owner明确Prohibited，绝不使用。只在一个节点使用最多六张真正增加吞吐的GPU，复用现有policy、carrier、experts、
normalization、fixed rows与evaluator，不重训任何checkpoint、不复制大资产。

## Completed adjudication

三个解析arms的strict250结果为：

- latest：`110`，逐global `0/9/18/25/36 = 28/31/40/4/7`；matched direct为`108`，
  retained/gained/lost=`90/20/18`；
- independent：`120`，逐task `27/36/38/11/8`；matched direct为`113`，
  retained/gained/lost=`98/22/15`；
- earliest：`76`，逐task `18/18/33/2/5`；matched direct为`74`，
  retained/gained/lost=`57/19/17`。

pooled projected为`306`，高于matched direct合计`295`；逐member matched retention合计
`245/295=83.05%`。Goal保留`15/24=62.5%`且三个members均非零，均通过capacity-supported条款。Long绝对成功从direct
合计11增加到20，三个members同样全部非零，但逐member matched-row只保留`4/11=36.36%`，低于预注册`50%`。因此
capacity-supported为false、capacity-binding也为false，正式裁决必须保持**mixed**；不能因absolute更高而事后改门或自动启动
residual solver。

按mixed条款完成的无新rollout定位显示：三个direct Long policies的success union为11 rows，三个projected policies的union为
16，重合6，equivalence-class union retention为`6/11=54.55%`。这说明唯一未过条款集中于“同一个member保留同一Long row”，
而不是rank4残差无法产生Long能力；但该union是post-hoc定位，不能重写本卡。episode key、env seed、policy seed root、language
和policy-noise common prefix全部零mismatch；18个workers均返回0。正式证据：
`docs/evidence/ecp_20260823/ecp_mobile_rank4_residual_capacity_gate_20260823.json`。

当前不授权solver、rank sweep或Stage 1C。下一步必须先明确：是否按专家“multiple successful policy equivalence、exact-row只作
辅助”的原则，把本轮视为足以继续检验同一rank4 residual的objective-identification；还是要求一个更严格的capacity operator。
这个决策不得通过复跑当前750 rows或挑member解决。

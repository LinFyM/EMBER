# ECP Phase 2B fallback：centered two-sided functional coordinate

状态：2026-08-24 **在retained implementation与任何新rollout前预注册。只先执行fit-only coordinate与held latest
transform-only oracle；未授权新realizer训练。**

## 为什么重新打开专家保留的第二种坐标

首个balanced-SVD factor realizer的fold0 strict250为step800/1000=`33/37`，低于carrier `43`，breadth均为`2/5`且
Goal/Long为0。这个non-pass保持不变。随后在两个固定checkpoint都结束后进行的无梯度、无新rollout定位发现：

1. fit residual的task-equal expected effective-update energy为`94.1161`，fit-task shared mean energy为`89.1989`；
   即`94.775%`是共享均值，真正task-specific centered variance只有`5.225%`；
2. 当前step1000完整update相对held target仍有`.468--.799` cosine，但减去同一fit mean后，prediction与target innovation
   cosine只有`.0012--.0573`；absolute factor/effective loss主要奖励了共享部分；
3. held latest effect-code RMS中四项达到或超过fit分布上沿，Goal为`2.572`而fit maximum为`1.728`；当前learned heads需要
   在绝对A/B坐标上同时做OOD幅度外推；
4. scalar reliability被直接加进owner state。把同一checkpoint的reliability固定为1会把Long residual energy从target的
   `.229`变为`.940`，说明它已成为非线性坐标门，而不是只作member validity weighting；未来`q_V`也没有teacher success-rate
   这个deployment输入。

这些证据不证明“改一个reliability数值”能解决问题，也不授权inference常数选择或超参扫描。它们说明首版输出坐标和损失没有
识别低能量但行为关键的task innovation。专家原始方案明确保留
`fixed two-sided effective-update sketches + deterministic rank4 reconstruction`作为第二种principled coordinate；现在有
新的、直接针对失败接口的依据检查它，而不是恢复旧solver或任意新版本。

## 唯一科学问题

在完全不学习held mapping的前提下，一个只由fit90固定的、去shared mean并task-variance aware的线性functional coordinate，
能否表达fold0五个known-success mobile-rank4 corrections并保留其真实closed-loop support？

本轮是coordinate expressivity oracle，不是deployment Writer。held latest residual只在fit coordinate冻结后做一次transform；
它不能成为未来realizer输入、free code、checkpoint选择或task route。

## 固定坐标

对每个LoRA target `j`固定两组与task无关的宽度8正交probe：

```text
Omega_j[in,8], Psi_j[out,8]
Y_j = DeltaW_j Omega_j       [out,8]
Z_j = Psi_j^T DeltaW_j       [8,in]
```

probe只由全局seed `20260824 + 97*j`生成Gaussian列后做reduced QR；不按结果、task或member改变。与balanced-SVD factors不同，
`Y/Z`对effective update线性且无A/B gauge。

只用fold0的90 fit tasks/108 members，按task等权、task内member等权，计算每target的task-equal sketch mean。对centered
`concat(Y,Z)`在sample space拟合固定basis：保留relative eigenvalue大于`1e-7`的全部方向，最多128维；不足128时zero-pad并保存
active mask。held不更新mean、basis、scale或probe。

因此未来canonical owner code固定为：

```text
U_effect[target=38, width=128]
```

训练时只对白化后的centered innovation评分；shared mean不再淹没task variance。当前oracle从held exact sketch投影到fit span，
只回答这套固定坐标的表达能力。

## 固定rank4 reconstruction

decode后的`Y_hat/Z_hat`形成一致化core：

```text
C = 0.5 * (Psi^T Y_hat + Z_hat Omega)
```

只保留`C`的前4个singular directions构造rank4 pseudoinverse，再以
`Y_hat C_rank4^+ Z_hat`得到rank4 effective update；最终用small-core balanced SVD和deterministic sign gauge生成A/B。不得用
full pseudoinverse放大fit-span外噪声，不做per-task damping、alpha、trust或rank选择。

CPU只读预审已经验证：exact held sketches经该rank4 reconstruction的aggregate relative error约`1e-12`；fit-span投影后的held
aggregate effective cosine逐global `0/9/18/25/36 = .957/.950/.960/.877/.953`，relative error为
`.084/.098/.079/.231/.092`。这些数值只授权一次闭环oracle，不用于改变probe、width或Gate。

## Formal oracle与Gate

1. 从clean pushed commit在CPU构建唯一fold0 coordinate authority；保存fit provenance、probe、mean、basis、scale、active rank及
   118个transform codes；
2. coordinate冻结后只读取五个held latest residual，物化五套single
   `stable carrier rank12 + reconstructed residual rank4`完整rank16 LoRA；held optimizer steps为0；
3. 不做screen，直接运行同一held5 × 50 strict paired250；与carrier和既有exact mobile-rank4 latest `110/250`逐row配对。

通过必须同时满足：

- total `>=83/250`，即至少保留known-latest 110的75%；
- breadth `5/5`，Goal与Long均非零；
- 保留carrier至少`33/43` successes；
- 保留known-latest至少`83/110` successes；
- 250行episode key、environment seed、policy seed root、language与policy-noise common prefix零mismatch。

## 裁决边界

- 通过：冻结two-sided coordinate，fresh训练一个只预测centered whitened innovation code的小型realizer；reliability只允许在
  多member集合聚合时作权重，单member时必须消去，不能进入owner semantic state；不再生成absolute A/B factors；
- 失败：停止当前128维fit-span two-sided coordinate，不扫probe seed/width/rank/PCA阈值，不训练新realizer；回到专家讨论
  shared realizer是否应放弃或需要新的source-unseen mappings；
- 无论结果如何，本轮不裁决video、Program、`q_pi/q_V`或完整ECP，也不读取validation/Test action/reward。

配置：`configs/pi05_ecp_centered_two_sided_coordinate.json`。

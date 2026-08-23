# ECP Effective-Update Reachability Oracle

状态：2026-08-23 **在retained implementation、fit profile与held GPU launch前预注册；随后只按本卡实现，科学合同未改。**

## Scientific question

mobile-rank4解析投影已经证明`carrier12 + residual4`能保留direct级闭环；raw-factor solver却在五项都只走到
`.0009--.0012` trust、约known-success correction 1%的effective norm，并得到`49/250`。本卡只回答：

> 保持同一48-state three-particle bank、effect objective、stable carrier、rank4 residual、trust上限、12次VJP预算与原closed-loop
> Gate不变，若从raw A/B RMS更新改为gauge-invariant effective-update方向和确定性trust-region回溯，是否能到达已知存在的
> successful-policy effect basin并产生广泛闭环能力？

不改数据、member、response字段、loss权重、temperature、Program、Writer或held rows；不把LR、steps、初始化、seed或rank扫
包装成新架构。

## Why this is a distinct authorized question

当前non-pass不是capacity failure：三个known-success mobile-rank4 arms为`110/120/76`且全部5/5非零。也不是objective没有
successful endpoint：同一bank上的best member objective逐task只有`.084/.120/.163/.060/.078`，而raw solver final仍为
`1.915/2.401/2.019/1.987/3.262`。

当前operator从`B_r=0`开始。此时`A_r`梯度必为0，首个方向只能位于任意inactive carrier-A rows规定的row space；随后raw
factor RMS、factor gauge与每步SVD retraction共同决定有效更新尺度。12步后correction与known-success corrections的cosine仅
`.041--.077`、norm ratio约`.009--.0105`。因此在否定effect-distribution target前，必须先检验一个直接定义在
`Delta W=B_r A_r`上的固定求解几何。

## Frozen scientific contract

- effect banks、state IDs、category/stage weights、three members及reliability完全不变；
- owner/flow/action、carrier barrier、preservation、confidence、temperature与trust objective完全不变；
- stable carrier前12 ranks冻结，输出仍为一套38-target complete rank16 LoRA；
- residual rank固定4，effective-update严格为`W_carrier + Delta W`，不部署第二adapter；
- trust定义与上限继续为mean per-target
  `||Delta W_t||_F^2 / ||W_carrier,t||_F^2 <= 1.5`；
- 最多12次full-objective VJP；不增加训练steps，不学习shared parameters；
- fold0 held5不选择数值，不读取validation/test action或reward。

## Fixed matrix-free solver

### 1. Zero-residual gradient sketch

在exact carrier处不物化dense gradient。每target用固定seed、与task内容无关的8列orthonormal Gaussian input probe，分成两个
rank4 chunks：

```text
Y_j = G V_j                         # grad_B with A=V_j^T, B=0
U   = qr([Y_1,Y_2])
Z_j = U_j^T G                       # grad_A with B=U_j, A=0
Z   = [Z_1;Z_2]
Delta W_0 = - rank4_svd(U Z)
```

四次VJP得到一个不依赖raw factor gauge的rank4 steepest-descent sketch。每个target只按其carrier effective energy归一化，使
`Delta W_0`的per-target relative trust为1；不按held task、member或结果选择probe seed。

### 2. Monotone trust-region acceptance

首步和后续步都从共同`alpha=1`开始，仅做固定的`1, 1/2, 1/4, 1/8, 1/16`回溯。候选必须同时：

- mean trust不超过1.5；
- full frozen objective严格低于当前点；
- 全部factor与response finite。

接受第一个满足条件的alpha；五个alpha都失败则认为固定solver收敛并停止。该回溯是solver内部objective-only acceptance，不读取
closed-loop结果、不形成checkpoint sweep；final始终是最后一个单调接受点。

### 3. Gauge-invariant tangent updates

首步后residual已非零并每target balanced。剩余最多8次VJP使用当前raw gradients，但先在effective-update metric中预条件：

```text
dB = -grad_B (A A^T + eps I)^-1
dA = -(B^T B + eps I)^-1 grad_A
T  = dB A + B dA
```

`eps`固定为每个4x4 Gram mean diagonal的`1e-6`倍并只作数值正则。每target把`T`归一到unit relative trust；候选
`rank4_retract(Delta W + alpha T)`通过上述同一回溯接受。retraction只对最多rank12的低秩core做thin-QR/SVD，不物化dense
`W`，不产生carrier/raw-factor交叉项。

总VJP预算固定为4次initial sketch + 最多8次preconditioned tangent update = 12。不得改变sketch width、backtrack grid、
damping、VJP预算或trust上限。

## Canonical implementation lifecycle

- `src/ember/ecp/stage1_realization.py`成为唯一solver owner；秩预留加法与解析capacity projection集中到
  `src/ember/ecp/stage1_parameterization.py`，避免退役solver与解析诊断继续堆叠；
- active runtime只保留一个solver入口，不保留fixed-A或raw-factor fallback；历史由Git、cards和formal artifacts恢复；
- 配置改为一个effective-update oracle config，CLI默认只指向它；不新增平行版本目录或第二entrypoint；
- focused tests只验证matrix-free sketch、gauge invariance、trust/backtracking、rank4 retraction、single-LoRA与既有投影helper。

## Non-held profile gate

先只在既有ordinal71/global2 profile bank运行一次。它的carrier objective为`2.214329`，best successful-member objective为
`.127698`，known-success rank4 projection trust为`1.198--1.512`。profile必须同时满足：

- initial state逐元素等价于carrier；
- sketch directional derivative为负，至少接受一个trust-region step；
- final objective gap recovery
  `(carrier-final)/(carrier-best_member) >= .50`；
- final trust位于`[.10,1.50]`，12次以内VJP、全部finite；
- 峰值显存适合单张A40，且没有改变任何冻结数值。

profile只决定机制是否成立。若未过，停止本solver，不在ordinal71扫seed、trust、backtrack或damping；held5不启动。

## Held5 and direct closed-loop gate

profile通过后，held ordinals`90--94`各自从exact carrier独立求一套final LoRA；task间不共享factor、gradient、line-search或选择。
五套final立即进入原strict paired250，不用inner objective、geometry或member cosine预筛。

完整Pass Gate保持不变：

- candidate至少`74/250`，相对carrier43净增至少20；
- 5/5 task非零，至少4/5严格高于carrier；
- Goal与Long各自非零；
- carrier success retention至少`33/43`；
- multiple-member union normalized recovery至少`.35`且至少4/5 tasks为正；
- pairing、single-LoRA与information wall有效。

exact-row、trust、objective gap recovery与correction cosine只作定位。只有final接近完整门才允许相邻稳定性；不挑line-search
中间点，不融合LoRA或checkpoint。

## Allowed outcomes

- **Pass**：effective-update realization通过，才授权进入shared `Program -> effect distribution` Stage 1C；
- **Profile non-pass**：matrix-free/preconditioned solver自身不可达，停止而不消耗held rollout；
- **Closed-loop non-pass with strong effect-gap recovery**：若五项都明显进入member-effect低值区但仍闭环失败，effect objective
  sufficiency成为最早主要断点，暂停并重新处理policy-effect target或process-identifying data；
- **Engineering invalidation**：只修复可复现的实现、OOM、asset、pairing或runtime错误，重跑同一卡。

Action Meta-LoRA本卡保持关闭，以免同时改变observer坐标。owner要求的后续matched Action Meta尝试没有取消：只有本realization
坐标成立后才单独做control；若无负面则启用并永久冻结。

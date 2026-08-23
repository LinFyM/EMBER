# ECP Rank-Reserved Mobile-Rank4 Realization Oracle

状态：2026-08-23 **在retained implementation、fit profile与held GPU launch前预注册。**

## Scientific question

保持已经完成的48-state、three-particle occupancy bank、policy-effect objective、stable carrier和12-step更新合同不变，只把
已判为capacity-binding的fixed-A参数面替换为`frozen carrier rank12 + jointly mobile residual rank4`，能否把同一privileged
effect distribution实现为覆盖5/5 tasks、Goal/Long非零并恢复multiple-successful-policy union的一套完整rank16 LoRA？

本卡只回答`effect objective -> mobile residual -> closed-loop policy`。它不训练video predictor、Program、Writer或shared
compiler，也不改变Stage 0、数据、effect target、loss权重、member reliability或held rows。

## Why this question is authorized

前一张解析容量卡的正式结论保持`mixed`，不事后改写：pooled matched-row retention为`83.05%`，Goal为`62.5%`，但Long
同member同row retention为`36.36%`，低于当时预注册的50%。然而专家明确要求新的Gate同时使用candidate absolute、breadth、
Goal/Long、shared retention与多个successful members的success union，并把某一个direct policy的exact-row retention作为辅助。

同一批无训练解析LoRA已经得到`110/120/76`，三臂均5/5非零；Long absolute由11增至20，三个members均非零，Long
successful-policy union保留`6/11=54.55%`。因此本轮不宣称旧卡`capacity-supported`，只作一个新的、显式的科学决策：没有证据
表明rank12+mobile-rank4拓扑存在明显successful-policy-equivalence capacity binding，值得用唯一一次固定solver实验检验
objective identification。

## Fixed realization operator

每个target只保留carrier的前12个有效ranks，并使用后4个rank slots承载一个可移动低秩修正：

```text
W = B_c12 A_c12 + B_r4 A_r4
  = W_carrier + W_residual
```

- `A_c12/B_c12`全程冻结；
- `A_r4`初始化为carrier未激活的rows 12:16，`B_r4`初始化为0，所以step0逐元素等价于carrier；
- 每步先在当前candidate上detach three-particle responsibilities与barrier mask，再对完整48 states做exact microbatch VJP；
- `A_r4/B_r4`按target联合gradient RMS归一化，固定使用既有`0.0002 / sqrt(step+1)`，不加第二学习率；
- 每步后只对`B_r4 A_r4`做thin-QR/core-SVD balanced regauge，保持同一个effective correction与rank4，不触碰carrier；
- trust距离定义为每target `||B_r4 A_r4||_F^2 / ||B_c12 A_c12||_F^2`后平均；
- final通过rank拼接序列化为唯一一套38-target rank16 LoRA。carrier与residual在effective-update层严格相加，不存在raw
  factor交叉项，也不部署第二adapter。

solver其余合同冻结在`configs/pi05_ecp_stage1b_mobile_rank4_oracle_v1.json`：12 steps、inverse-sqrt decay、temperature、
owner/flow/action、carrier barrier、preservation、confidence、trust与microbatch均不变。不得扫rank、step、LR、初始化、插值、
objective weight或member。

## Profile and formal panel

1. 非held ordinal71只作一次数值/显存profile：必须step0精确为carrier、12步finite、mobile A/B均获得非零gradient、final
   objective低于initial。只允许修复可复现实现错误、OOM或batching；不能凭profile或held loss改变上述算法与数值。
2. profile有效后，fold0 held ordinals `90--94`各独立从同一carrier求一套final12 LoRA；held tasks之间不共享梯度、optimizer、
   factors或选择。
3. 五套final12直接进入既有strict paired250 rows，与source、carrier43、direct earliest/latest/independent及fixed-A78作既有
   reference；不挑row、不挑member、不复跑当前mobile projection 750 rows。

validation/test actions与reward读取仍为0；本卡是train-fold privileged oracle，不是deployment Writer。

## Pre-registered closed-loop gate

沿用被本实验单变量替换的occupancy-complete oracle Gate，不因解析投影结果降低标准。Pass必须同时满足：

- candidate至少`74/250`，相对carrier43净增至少20；
- 5/5 task非零，至少4/5严格高于carrier；
- Goal与Long各自非零；
- carrier success retention至少`33/43`；
- multiple-member success union之外的carrier既有support与union内recoverable support按原定义统计，overall
  oracle-normalized recovery至少`.35`且至少4/5 tasks为正；
- strict pairing、single-LoRA与information wall全部有效。

exact-row、factor geometry与inner objective只作定位；不能覆盖上述closed-loop gate。若final12达到或紧邻完整门，才允许补
step10/11相邻稳定性，不改变final选择。

## Allowed outcomes

- **Pass**：mobile-rank4 realization通过，才授权为同一effect-distribution interface设计shared video-conditioned predictor；
- **Objective/solver non-pass**：若inner objective有效下降但闭环未过，停止本operator，不扫solver；结合解析capacity正证据把
  最早缺口定位在effect objective identification或固定优化动力学，并在建立下一科学卡前暂停复盘；
- **Engineering invalidation**：只修复可复现的实现、asset、OOM、pairing或runtime问题，重跑同一卡，不增加科学版本。

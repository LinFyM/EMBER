# ECP Phase 2A：known-success effect-path calibration

状态：2026-08-24 **在retained实现与formal GPU capture前预注册。**

## 科学问题

本卡只回答：现有PI0.5 owner/flow/action policy-effect坐标，沿15条已经由strict closed-loop证明有效的
`carrier rank12 + mobile residual rank4`真实修正路径，是否给出稳定、可用于固定realizer接口的方向信号。

它不是新solver，不训练`q_pi`、`q_V`、Program、Writer或realizer，不读取validation/test action/reward，也不产生新LoRA供
closed-loop选择。15个endpoint已在既有三个strict250 arms中分别得到`110/120/76`且均5/5 task非零；本轮只重新查询它们在
同一48-state effect panels上的response。

## 固定输入

- held-fold train tasks固定为ordinal `90..94`、global `0/9/18/25/36`；
- 每task固定latest、independent、earliest三个known-success mobile-rank4 projections；
- stable carrier、38-target rank16合同、五个既有48-state effect banks、native observer与source PI0.5全部冻结；
- 路径系数固定为`alpha = 0, 1/8, 1/4, 1/2, 3/4, 1`；
- 对每target保持carrier前12 ranks不变，只把canonical balanced-SVD residual4的A/B同时乘`sqrt(alpha)`，因此
  `Delta W(alpha)=alpha*Delta W(endpoint)`；
- `alpha=0`直接复用bank中已保存的carrier response，其余75个path points各完整forward一次，不做梯度、checkpoint选择或
  路径点closed-loop。

## member-state validity与global identity

本轮不再允许旧objective按category/stage分别选择不同member。一个candidate在整条trajectory上只对应一个global member
particle；multi-member只在整条trajectory loss形成之后做一次posterior mixture。

已有资产能严格证明的validity固定为：

1. 8个initial states：某member在其matched fixed50结果的同一init-state row成功时有效；
2. 24个successful states：只对生成该成功trajectory的member有效；
3. cross-member successful states、candidate和recovery states：当前没有continuation success/Q/progress证据，不能伪装成
   equivalence target；只单独报告相对carrier的response drift。

这是一张保守下界mask，不声称已经完成未来`q_pi`所需的全部continuation/recovery validity。后继effect-bank必须保留probe轴并
补齐更丰富validity；本卡只校准现有antithetic-averaged response coordinate，不能证明probe distribution已经合格。

## 固定指标

对每个path point同时报告：

- legacy stage-wise objective，仅用于说明旧soft-min是否产生误导，不作为后继authority；
- matching-member verified loss：owner/flow/action按旧bank scales归一化，在该member有效的initial与on-policy successful两类内
  分别平均，再对存在的类别等权；
- global-particle objective：先为每个member形成上述整trajectory scalar，再按固定member reliability与temperature只做一次
  soft-min；
- initial、successful、candidate、recovery四类相对carrier response drift；
- endpoint trust、每段变化、endpoint是否为该路径最小值；全部结果按task、member和family报告。

## 预注册Gate

Phase 2A通过必须同时满足：

1. 15/15 matching endpoints的verified loss严格低于各自carrier；
2. 至少12/15 paths在`alpha=1/8`已低于carrier，且5/5 tasks各至少一条满足；
3. 至少12/15 paths的最低verified loss出现在`alpha=3/4`或`1`，避免只有极小局部下降后反向恶化；
4. 5/5 tasks的最佳known endpoint均使global-particle objective低于carrier，Goal与Long不得例外；
5. 所有responses、loss与trust finite，15个endpoint identity与既有projection authority完全对应。

若1--3通过而4失败，裁决为`single-member supported / mixture non-pass`：balanced-SVD coordinate仍可继续校准，但必须先修正
multi-member posterior，不训练realizer。若single-member门失败，现有owner/flow/action response coordinate不足以监督realizer，
先重建effect representation，不做solver或alpha sweep。全部通过才进入Phase 2B，并默认选择已经确定sign gauge的
balanced-SVD rank4 coordinate；fixed two-sided sketches只保留为一次有原则的fallback。

## 执行边界

formal capture来自clean pushed detached authority；launch前重新检查gpu01与gpu02及`/data1`quota。单节点最多五张GPU，一张
task一张卡；gpu01 physical0禁止使用。结果只保留每point scalar与必要metadata，不保存重复response tensor或adapter副本。

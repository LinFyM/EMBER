# V6 Dynamic Slot-Set Bridge

状态：2026-08-14 active controlled bridge authority，canonical实现、真实GPU机制门和full24吞吐profile已通过，
formal合同已seal并进入fresh macro0→25。
本文授权以历史v6-fast macro400为冻结性能底座，只新增一个K=1严格恒等、K>1置换不变的跨视频Program-slot
集合层。它是机制开发实验；若成功，仍需在同一train24信息墙内完成从零训练，warm start不能成为最终论文方法
的隐含前提。

## 1. 为什么现在回到v6边界

最新Full-Factor rank8在macro50只有`91/400`。它与fixed-A取得近似offline functional loss，却让B norm降到
`.0622x`、effective BA降到`.2448x`并近乎正交；继续改A/B mapper没有科学依据。另一方面，v6-fast的
`143/400`仍是唯一证明过强absolute的Writer路径，其language-conditioned Semantic Core、有向Procedure、
320个policy/rank aligned slots和完整rank16 factor heads尚未被后续低分架构否定。

本实验不是把昨晚的讨论作废。保留的原则是：语言与视频共同输入；每条视频内部保序；多视频间只做集合处理；
一次生成一套完整LoRA；不平均最终LoRA。暂不继续绑定8 memory tokens和rank8，是因为它们刚在同一完整链上接受
了`88/86/86/96`与`91`的closed-loop检验。以v6为baseline后，本轮唯一主要变量才真正是“动态K集合桥接”。

## 2. 完整数据流

```text
exact task language + K=1..4 same-task action-hidden ordered videos
    -> 每条video独立运行原生v6 joint image/language/Action-probe encoder
    -> 每条video独立形成language-axis Semantic Core
       和带真实帧顺序的visual-transition Causal Procedure
    -> 原生v6 compiler为每条video产生320 x 256 policy-aligned slots
    -> 对每一个对应slot，仅沿K视频轴做置换不变Slot-Set聚合
    -> 原生v6 factor heads只运行一次
    -> one complete 38-target rank-16 task-conditioned LoRA
```

每条视频保留全部stride-5采样帧及final frame，不为了K1/K4形式公平把总帧数硬裁到64；K增加就提供更多真实教学
证据。训练调度按实际各视频帧数做long-first负载均衡。Writer仍在rollout前运行一次，执行期间不再看视频。

## 3. 唯一新增的Slot-Set算子

对一个condition、一个原生v6 policy slot，K条视频给出`x_k in R^256`：

```text
mu       = mean_k(x_k)
center_k = x_k - mu
q        = Wq(RMS(mu))
k_k      = Wk(RMS(x_k))
a_k      = softmax_k(q dot k_k / sqrt(d))
residual = Wo(sum_k a_k * center_k)
x_set    = mu + residual
```

`Wq/Wk/Wo`沿全部320个slots共享，`Wo` zero-init、全部linear无bias。该算子不是简单feature平均：均值只提供稳定
共同骨架，task/video内容选择的中心化残差决定保留或抑制哪条demo偏差；也没有分别生成K套LoRA再平均。

它有两个结构保证：

- video permutation不改变`mu`、attention加权和或输出；
- K=1时`center_1=0`，所以无论训练到何时`x_set=x_1`严格成立，原生v6 K1 graph和LoRA不会被集合层改坏。

视频顺序的重要性发生在集合层之前：每条video的Procedure使用真实ordinal causal attention和相邻视觉变化。
reversed/shuffled必须先重排真实输入帧并重新运行整条per-video v6路径；集合层不能从无序帧恢复正确Procedure。

## 4. 参数、初始化与训练

首轮controlled bridge加载历史v6-fast macro400的600个Writer tensors并全部冻结；仅训练一个共享Slot-Set层，
约`197k`参数。冻结v6 evidence/compiler在`no_grad`下运行，只有Slot-Set和穿过冻结factor heads的输入梯度保留，
避免为机制开发支付完整v6反向成本。

- train24、每macro full24等权、B20跨episode source-action queries不变；
- 每macro K1/K2/K3/K4各6 tasks，各task每四个macro覆盖全部K；
- K条video同task、互不重复且与B20 action episodes错开；
- 只用positive functional loss，不加singleton imitation、negative、expert、reward、norm/rank或LoRA重建loss；
- K1提供严格保留检查但对Slot-Set梯度按构造为零，K2--K4学习跨demo共同程序；
- source policy和完整v6底座均冻结，部署仍只有一个checkpoint和一套LoRA。

warm start只回答“few-shot集合层能否在已知强Writer上增加而不破坏K1”。若它通过，下一独立authority才会训练
同一架构的fresh recipe；若它失败，只淘汰这个slot边界与该集合算子，不否定memory tokens、few-shot或所有v6
前端移植。

## 5. 快速否决门与closed-loop裁决

实现前后的最小机制门：

1. 同一K1输入的新Writer slots和76个LoRA tensors与历史v6逐元素相等，训练一次后仍相等；
2. K2--K4 video permutation输出一致，改变某条video内部顺序会改变slots/LoRA；
3. K>1的centered residual、Slot-Set参数梯度和Program→LoRA→functional链有限非零；
4. v6与source policy均无trainable parameter，只有Slot-Set进入optimizer；
5. full24一macro保持K1--K4各6、B20、真实帧成本均衡且无OOM/nonfinite。

profile后只做短段训练并尽快评K4 strict paired correct400；K1由严格tensor等价和既有old134 paired panel作为
保留基线。首个裁决点macro25，必要时只续到50：

- K4若没有明确超过old134或breadth低于7，不续训、不扫K、LR、温度或seed；
- K4若超过`150/400`，立即封存single-checkpoint结果，再补K1/K2/K3 scaling和correct/same/wrong/shuffled/
  reversed/no-video因果controls；
- 任何aggregate都报告per-task/per-suite、retained/gained/lost和top-task concentration，不能以K1/K4 union冒充
  同一condition的能力。

## 6. Ownership与生命周期

- `src/ember/writer/model.py`是唯一active Writer入口，复用`legacy_v6_model.py`中checkpoint-compatible v6 owner，
  不复制第二套Core/Procedure/compiler/factor实现；
- 新Slot-Set放在一个小而独立的owner中；旧Dynamic-K memory/rank8 active实现由Git和formal artifacts保存，不保留
  runtime flag或并行canonical config；
- trainer、task/video schedule、functional B20、checkpoint与dynamic evaluator只做该新合同所需的原位切换；
- 正式run仍需clean pushed/frozen commit、live双节点GPU检查、quota与profile seal。

## 7. 实现与机制门证据

canonical实现只新增`src/ember/writer/slot_set.py`，复用checkpoint-compatible native v6 owner；同时删除退役的
rank8 backbone-memory、memory-program和LoRA-mapper三个active modules及其专属tests，净减少约2100行active
source，不保留runtime双路径。全量CPU=`370 passed`。

gpu01物理GPU4上的真实source-policy smoke给出：

- K1 native-v6与bridge的76个LoRA tensors逐元素完全相等；
- trainable=`197120`，frozen v6/source均无梯度，Slot-Set output grad norm=`1.92e-5`；
- K2交换video后Program/LoRA max abs=`.02656/.001953`，K4 Program max abs=`.01373`；这是同一BF16
  batched backbone在样本换位时的正常kernel/reduction低位差异，不为逐元素一致拆成K次forward；
- 同一K1真实视频倒序使Program mean abs变化=`.21703`，显著大于换序低位差异，内部时序路径保持有效；
- peak allocated/reserved=`18.75/19.27GB`，未发生OOM或nonfinite。

这些结果只通过机制门，不是closed-loop成绩。

full24 B20 profile使用clean detached `07e9477`、gpu01物理`0,1,4,5,6` world5：K1--K4各6，24/24 tasks
finite，所有视频完整保留，最长condition=`323`帧；`30.7422s/macro`，global functional=`.101173`，gradient
norm=`1.7725e-6`，peak allocated/reserved=`36.48/40.75GB`，0 OOM/nonfinite。第一份`8278f74`尝试在任何
optimizer step前因frozen worktree错误解析相对v6资产路径fail-closed；`07e9477`把同一checkpoint从runtime显式
canonical asset root解析，未改科学图。profile checkpoint不进入formal训练；下一步fresh macro0→25。

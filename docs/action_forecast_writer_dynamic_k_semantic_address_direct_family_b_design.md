# Dynamic-K Semantic-Address Direct-Family-B Rank-8 Writer

日期：2026-08-13。状态：formal macro50与K1 strict paired correct400已完成，终局non-pass；不resume到100，
不再作为fresh训练active successor。本文保留实际受检验的单变量、预注册门与最终裁决。

## 1. 这轮只解决哪个最早失效接口

上一版Dynamic-K semantic-address Writer已经完成正式裁决。clean `9e70b81`的fresh `0→50`稳定结束，macro50
strict paired correct400为`101/400`、breadth6，per-task按Spatial1/3、Object1/3、Goal3/6、Long1/2为
`2/1/40/18/0/37/3/0`，per-suite=`3/58/37/3`。相对Dynamic-K v1的`100`为
`84 retained/17 gained/16 lost`；相对old134为`82/19/52`，相对compiler138为`80/21/58`，相对online128为
`83/18/45`。它低于v6-fast143共42分，按预注册门终局non-pass，不resume到100。

该结果否定了“只给temporal Query恢复absolute Semantic Core，就能修正Dynamic-K的policy方向”这一具体假设；
它不否定真实backbone memory、dynamic K、rank8、semantic query address或有向Procedure。generated BA总norm均值
`152.34`，并非identity；但八个held task mean BA的off-diagonal cosine为`.77568`，比Dynamic-K v1的`.70203`
更同向，action-target norm仅`.5074`，任务能力仍集中在三类通用pick/place。

为了不再凭最终BA猜compiler，本轮在同一个macro50 checkpoint上完成validation8 × 4个无放回video/state ordinals ×
五臂真实raw-frame forward的逐接口probe。关键correct task geometry为：

| stage | task-mean offdiag cosine | Vtask | Vvideo | task/video SNR | 与final Program的task-pair Spearman |
| --- | ---: | ---: | ---: | ---: | ---: |
| M2P input | `.49234` | `.46568` | `.04020` | `11.58` | `.9551` |
| final Program | `.52901` | `.44376` | `.02772` | `16.01` | `1.0000` |
| shared project `256→1024` | `.52986` | `.44681` | `.02557` | `17.48` | `.9787` |
| family hidden + GELU | `.63406` | `.37798` | `.01900` | `19.90` | `.9595` |
| dynamic B | `.77900` | `.26612` | `.01159` | `22.96` | `.7937` |
| effective BA | `.77922` | `.26598` | `.01158` | `22.96` | `.7926` |

final Program到shared projector几乎不损失task几何；第一个明显common-direction增长发生在四个family hidden/GELU，
最终B又继续把task difference压缩。五臂relative-L2也给出同一定位：

| stage | same | wrong | shuffled | reversed |
| --- | ---: | ---: | ---: | ---: |
| final Program | `.3011` | `.9837` | `1.3546` | `2.1617` |
| shared project | `.2911` | `.9889` | `1.3384` | `2.1637` |
| family hidden | `.2536` | `.9133` | `1.0872` | `1.4844` |
| dynamic B | `.1899` | `.7534` | `1.0403` | `1.7813` |

8/8 tasks在所有阶段仍满足same<wrong，说明mapper同时保留了有益same-video invariance；不能把所有压缩都叫坏事。
但shared projector明确健康，而后续非线性family bottleneck将task差异与三类order contrast一起系统压小。因此本轮只
改变这个最早新增失真接口，不再改视频前端、Semantic Core、Procedure、set或M2P。

完整probe保存在：
`runs/outputs/pi05_dynamic_k_semantic_address_rank8_correct400_noreplacement_seed7_macro0050_3f630da_gpu01_retry1_20260813/semantic_mapper_stage_localization.json`。

## 2. 唯一主要变量

保留final Program `P ∈ R[B,20,8,256]`与shared bias-free projector：

```text
H[b,g,r] = W_project P[b,g,r]                 # 256 → 1024
```

旧mapper对每个shape family再做：

```text
U_family = GELU(W_hidden_family H)             # 1024 → 1024
B_family = W_B_family U_family                 # 1024 → output width
```

新mapper删除四个`W_hidden_family`、GELU以及从未启用的dynamic-A heads，只保留四个bias-free、zero-init、跨layer与
rank共享的direct family-B readouts：

```text
B_q          = W_B_q(H[:, 1:19])               # 1024 → 2048
B_v          = W_B_v(H[:, 1:19])               # 1024 → 256
B_action_in  = W_B_in(H[:, 0])                  # 1024 → 1024
B_action_out = W_B_out(H[:, 19])                # 1024 → 32
```

同一个`W_B_q/W_B_v`服务18个Action Expert层与8个rank coordinates；不是18/38/76个target-owned heads，不按task、
layer、video或held outcome路由。A始终直接取固定template buffer，不生成、不训练。每个B结果仍按真实38-target owner
与PEFT shape转置回完整rank8 LoRA。

当前mapper含shared project、四个hidden、四个冻结A heads与四个B heads，共`11,075,584`参数，其中
`7,897,088` trainable。新mapper为shared project加四个B readouts，共`3,702,784`参数，全部trainable；删除
`7,372,800`总参数和`4,194,304` trainable参数。输出的public LoRA仍是完整`643,584` scalars，没有裁剪target、
平均LoRA或降低policy覆盖面。

## 3. 为什么这是policy-topology对应而不是高维parameter slots

8个真实memory tokens仍在每帧的正常图像、exact language和50个native Action probes上下文中，逐Action Expert层
形成`[18,8,1024]`memory states。视频内有向encoder、跨video set与20×8 M2P继续负责时间、video、policy-group和
rank轴的必要交互。direct readout只在最后把已经对齐的`[group,rank]`cell映射到其真实shape family：

- 增加policy层数时复用同一个q/v family matrix，不线性增加独立head参数；
- 增加同shape targets时仍复用family matrix；
- 只有出现新的输入/输出shape family时才需要一个新readout；
- rank坐标由8个memory tokens和M2P rank轴承载，不把几千个LoRA chunks冒充memory tokens。

这继承SHINE的有效原则——真实context中的少量memory按layer/token拓扑处理后结构化生成LoRA——但不一比一复制
SHINE的文本模型、4-token/rank4配置或flat payload。SHINE官方实现把逐层memory states经layer/token M2P后展平为
LoRA payload；EMBER因q/v/action四种异构shape保留一个共享projector和四个family linear readout。参考：
https://arxiv.org/abs/2602.06358 ，https://github.com/MuLabPKU/SHINE 。

## 4. 输入、高层视频知识与正确顺序为何保持不变

本轮不重新设计输入处理：

```text
exact language + K=1..4 same-task action-hidden ordered videos
→ 每帧真实joint image/language/Action-probe backbone + 8 memory tokens
→ 每video signed adjacent transition D 与 terminal goal residual G
→ absolute mean memory只作temporal Query semantic address
→ causal temporal encoder（视频内保序）
→ permutation-invariant set aggregator（视频间无序）
→ layer/rank M2P
→ shared project
→ direct family-B readout
→ one complete task-conditioned rank8 LoRA
```

语言说明要关注的对象、关系和目标；video的D/G提供唯一dynamic value，展示达到目标的有向过程。shuffled/reversed
在真实frames上重排后会改变D/G、RoPE顺序与causal composition；absolute Semantic Core不能自己写LoRA。若所有
frames相同，则D=G=0，Program、B、BA与action delta仍必须为零。dynamic K训练仍每macro精确K1--K4各6 tasks；
每video内部独立保序，跨video只在set阶段聚合，不平均frames、features或生成后的LoRA。

因此本轮既不声称mapper负责视频理解，也不牺牲此前已经接通的视频顺序路径；它只检验健康的task/order Program
能否少经过一层无依据的共享非线性压缩而写到policy。

## 5. 与历史实验的准确边界

- v6的shared two-layer factor heads曾达到143，说明shared coherent readout可以有效；它不证明额外hidden对当前
  Dynamic-K Program仍必要。
- Target-Owned用76个独立two-layer heads，best99；它否定“解除所有target sharing就能解决漂移”。本轮方向相反：
  保持四个family跨layer/rank共享，只删除共同非线性瓶颈。
- Target-Spectral、Policy-Lane、Atom分别否定强制谱/正交、更多lanes和更多atoms，不涉及本轮direct B变量。
- K4 Policy-Layer Trace曾用大payload直接reshape A+B，best99且reversed105>correct99；它否定“SHINE式direct
  payload自动有效”，但同时改变memory payload数、M2P、dynamic A/B和K4前端，不是当前fixed-A rank8窄对照。
- SFB/Direction Store继续使用two-layer factor heads并增加basis/store routing；它们没有测试本轮线性readout。

仓库与全refs审计未发现“保留当前Dynamic-K上游与shared projector，只删除family hidden/GELU并direct写B”的
fresh实验。本轮仍不能预先宣称解决task drift：如果full24 functional credit本身不对应held on-policy useful
direction，减少mapper压缩也会失败。

## 6. 保留不变的科学合同

- source policy、normalization、24/8/8 split、official LIBERO preprocessing和38-target topology不变；
- fresh public rank8、alpha8、fixed template A、dynamic B不变；不压缩/迁移旧LoRA；
- exact language、dynamic K1--K4、frame stride5、每condition 64真实frame预算不变；
- 8个真实memory tokens、Action Meta-LoRA、temporal/query address、set、M2P全部不变；
- train24 task-complete、B20 cross-episode functional objective、singleton consistency、AdamW/scheduler、BF16/TF32
  与full24等权raw mean不变；
- validation/test action/reward不产生梯度；不加外部target数据、task experts、pretraining或Writer RL；
- 不加language/static bypass、negative margin、confidence、scalar gate、专家route、第二套LoRA或checkpoint融合。

旧semantic macro50的任何Writer、projector、optimizer、RNG或sampler state都不得迁移。新schema必须fresh macro0；
新方法内部exact-resume仍锁world size/topology。

## 7. 快速机制与工程否决

实现后必须同时满足：

1. canonical mapper只有一个shared `256→1024` projector和四个bias-free direct B linears；无hidden/GELU、dynamic-A
   head、target-owned并行path或compatibility fallback；
2. 四个B weights精确zero-init，step0全部76 tensors与template一致；A永远是固定template且无trainable A参数；
3. synthetic direct公式逐family、layer、rank、batch与实际输出一致；完整38-target shape/rank正确；
4. 第一步functional gradient只打开B，projector与上游functional gradient为零；B非零后gradient能到projector、
   M2P、set、temporal、semantic address与Action Meta-LoRA；
5. constant-video identity、K-set permutation、video order sensitivity、K1同图与fresh/exact-resume tests继续通过；
6. 旧semantic config/checkpoint/adapter schema被新active runtime fail-closed；不得部分加载相同shape的projector；
7. live full24 B20一宏finite、K1--K4各6、checkpoint完整；同world matched时wall不劣于semantic-address `39.2367s`
   的`1.10x`，不同world只使用预注册的task-normalized对照，不靠降batch或改dtype过门；
8. 新部署图在真实longest-video validation panel profile B8/B16/B32，选择stable LoRA/s最高者；旧B8 profile不能继承。

任一失败先判断是明确工程违约还是方法合同本体；修复若要改变scientific variable，则本设计退役，不做小补丁sweep。

## 8. Formal训练与真实性能裁决

机制与吞吐门通过后，从fresh随机初始化训练`0→50`，每25 macro保存。macro50立即做同一个single checkpoint、
correct-video、without-replacement seed7 strict paired400，并逐episode与semantic101、Dynamic-K v1 100、old134、
compiler138和online128严格比较；与v6-fast143按aggregate/per-task比较。

macro50预注册裁决沿用上一轮，避免看结果改门：

- `<120/400`或breadth`<6`：direct-family-B终局non-pass，不resume；
- `120..133`：仍低于old134，除非发现预注册前无法解释的工程合同违约，否则不resume；
- `134..143`：只有相对old134 gained>lost、至少3 suites不下降且task mean BA共线显著弱于`.779`，才exact-resume到100；
- `>=144`：exact-resume到100；目标仍是同一single checkpoint strict `>150/400`；
- 达到`>150`后补same-task-other、wrong、shuffled、reversed、no-video严格配对controls。correct必须沿有用policy
  direction获益，不能只把negative LoRA破坏。

所有区间报告per-task/per-suite、breadth、retained/gained/lost、churn、top3 share、task-mean BA、action-target
energy，以及相邻checkpoint能力轮换。loss、Program geometry、norm、rank和cosine只解释结果，不选择checkpoint。

## 9. 实现与GPU边界

- 原位替换`src/ember/writer/lora_mapper.py`的唯一canonical owner；删除旧ShapeFamilyMapper，不保留strategy flag；
- 新建fresh config/schema/arm identity，旧semantic正式结果由Git与artifact保存，current runtime不得加载；
- 不新增runner、evaluator、cache实现或大诊断框架；通用Dynamic-K执行路径继续复用；
- source diff完成targeted与完整CPU tests、clean commit/push后建立detached frozen worktree；
- GPU launch前同时live检查gpu01/gpu02；单节点使用至多6张真正有益且不明显干扰他人的A40，有几张用几张；
- 训练固定`NCCL_P2P_DISABLE=1`、GPU-local NUMA映射与deferred NCCL；评测继续dynamic queue与persistent workers；
- 不为低位一致性固定batch1、重复forward、扩dtype、逐tensor扫描或新增内容hash。

## 10. 解释边界

成功只支持：“当前Dynamic-K semantic Program中的task/order结构会被额外family hidden/GELU压成common direction，
跨layer/rank共享的direct family-B readout能保留更多policy-effective差异。”失败只淘汰这一具体direct readout；
不否定dynamic K、memory token、rank8、所有shared mapper、few-shot或未来reward credit。无论内部几何是否改善，
最终方法仍只由single-checkpoint strict paired closed-loop裁决。

## 11. Formal terminal verdict

fresh world5训练完整到macro50，正式K1 correct root为：

`runs/outputs/pi05_dynamic_k_semantic_address_direct_family_b_rank8_correct400_noreplacement_seed7_macro0050_trainr5_evalr6_c5353f3_gpu01_retry1_20260813`

72/72 shards complete、18/18 workers exit0、400 rows无缺失或重复。single-checkpoint strict结果为
`102/400`、breadth5；per-task按Spatial1/3、Object1/3、Goal3/6、Long1/2为
`0/1/40/11/0/43/7/0`，per-suite=`1/51/43/7`。top3 tasks贡献`94/102=92.16%`，能力仍集中在少数
pick/place任务。

严格逐episode比较为：

| reference | retained | gained | lost | churn | net | McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| semantic-address 101 | 82 | 20 | 19 | 39 | +1 | 1.000000 |
| Dynamic-K 100 | 79 | 23 | 21 | 44 | +2 | .880396 |
| old 134 | 80 | 22 | 54 | 76 | -32 | .000313 |
| compiler 138 | 79 | 23 | 59 | 82 | -36 | .000087 |
| online 128 | 79 | 23 | 49 | 72 | -26 | .002943 |

相对v6-fast143的per-task差为`0/-2/-6/-26/0/+7/-13/-1`，仅Goal6增加，Object3与Long1明显丢失。
Direct-B相对semantic101也不是共同积累：Object3 `18→11`，Goal6 `37→43`，Long1 `3→7`，aggregate只`+1`却
有39个episode换手。

同一correct K1前4个video/state ordinal的exact implicit effective-BA诊断复现semantic基线task-mean offdiag
cosine `.77947`，Direct-B降至`.74895`；`Vtask .26607→.27793`，但`Vvideo .01151→.01701`，task/video SNR
`23.11→16.34`。effective norm均值`152.43→136.64`，stable rank均值仍约`1.0033→1.0046`。所以删除
hidden/GELU确实轻微减弱common direction，却没有带来closed-loop增益或多task共存；mapper局部压缩不是主因。
完整artifact为同一root下的`benchmark_comparison.json`和`effective_ba_task_geometry_comparison.json`。

按`<120`或breadth`<6`预注册门，本方法终局non-pass，不resume、不扫超参、不补K1五臂controls。该负结果只
否定“删除family hidden/GELU即可把现有Program变成policy-effective LoRA”；不否定Dynamic-K、memory token、
rank8或few-shot。K2--K4尚未被正式evaluator测试，后续若评测只是独立判定同一checkpoint的真实few-shot部署
能力，不得反向改写本K1裁决。

## 12. Same-checkpoint Dynamic-K deployment audit

K1结果没有裁决训练时真实覆盖的K2--K4 set path。后续评测因此只扩展同一个canonical evaluator，不增加新Writer、
不平均features或生成后的LoRA，也不改变macro50 checkpoint。CLI显式选择`evaluation_k=1..4`；每个condition的K条
action-hidden视频在一次ragged Writer forward中共同生成一套完整rank-8 LoRA，总frame budget仍为64。

without-replacement seed7 schedule采用同一task/state permutation的nested prefix：K1保持历史选中的第一条，K4再取
后续三条；每个50-state block内无重复集合成员，且每条teacher video在K4 panel中恰出现4次。K1原adapter、pairing、
episode与cache-key合同逐字段保留；K>1使用独立video-set pairing/evidence identity，避免把K1 cache误作few-shot。
K2--K4在完成单A40 B8/B16/B32真实generation profile前只能做smoke，formal fail closed。第一项正式实验固定K4
strict correct400；它只回答当前cross-video set是否产生真实few-shot增益，不救援或改写K1 non-pass。

clean detached `9c5cec2`在gpu01物理2完成K4 validation 8 tasks×4 states、固定32-request longest-first profile。
B8/B16/B32吞吐分别为`.51618/.51298/.51056 LoRA/s`，两次repeat均稳定、无OOM/nonfinite；peak reserved约
`13.40/13.43/14.59GB`，因此按最高实测吞吐锁B8。formal K4只接受该batch。profile root为
`runs/outputs/pi05_dynamic_k_semantic_address_direct_family_b_rank8_k4_writer_generation_profile_val8x4_correct_gpu01p2_9c5cec2_retry2_20260813`。
gpu02物理0的首次尝试被其它用户中途占满而OOM；物理1 retry1又被外部进程临时共享导致B8/B16计时污染，二者均不
作为authority，不能据其异常数值选择batch。

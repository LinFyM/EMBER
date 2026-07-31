# EMBER Coherent-Procedure Writer：证据收敛与下一设计

最后更新：2026-07-31 UTC。

本文是当前 focused AS-Writer 的架构与实验交接 authority。它不宣称新模型已经
通过；它把现有证据收敛为一个最小、可证伪的下一步。当前 session 已完成内部
分析，但没有启动正式训练或 rollout。

## 1. 我们真正需要的函数

Writer 的输入是任务语言和恰好一条 action-hidden teacher video，输出是一套可在
同任务任意初态上帮助 frozen source policy 的完整 rank-16 LoRA。训练 action
queries 与 teacher video 同 task、跨 episode 独立，目的是阻止 Writer 复述某条
轨迹的低层动作，迫使它抽取可跨初态迁移的任务方法；这不降低“必须使用视频”的
要求。

模型需要两类信息：

- Semantic Core：任务、语义角色、目标关系和跨帧不变量；
- Causal Procedure：teacher video 展示的高层动作过程及其时序。

Core 可以小幅帮助 source policy 理解任务，但不能绕过 Procedure 单独产生完整
教学收益。无可靠教学信息时，生成的 LoRA 应尽量接近 base，而不是破坏 base。

## 2. 已被实验支持的最小主干

当前 evidence-backed canonical model 是原版 v5.2，而不是从失败版本继续修补：

```text
task language + one teacher video
        │
        ├─ stable text queries Q_text
        ├─ multimodal task-token evidence M_f
        └─ task-query → real patch values G_f
                         │
                   X_f = M_f + G_f
                    ┌────┴────┐
                    │         │
        permutation-invariant  native 50-suffix mean Action
        mean-anchored Core               │
                                         │ true frame positions
                                  2-layer causal Procedure
                    │                    │
                    └─ Core-primary / Procedure-AdaLN compiler
                                 │
                         320 routing slots
                                 │
                  conventional coherent factor heads
                                 │
                    complete rank-16 public LoRA
```

保留项及理由：

- `Q_text`、`M_f+G_f`提供稳定任务语义轴和真实patch evidence；
- frame-set Core 保持顺序不变，承担任务不变量；
- frozen Action Expert 的原生50-token suffix mean是在teacher帧上的高层动作假设；
- 两层causal Procedure保留真实时序；
- v5.2的Core-primary、Procedure-AdaLN compiler同时取得了absolute与强五臂
  视频因果证据；
- conventional factor heads保留高增益、q-dominant、跨层协调的有效写入流形。

## 3. v5.2 step900 LoRA几何补充实验

在exact historical commit `529da6b`上重新生成正式validation的8 tasks×50
correct-video LoRA，只做LoRA生成和数值分析，`rollout_shards_executed=0`。
分析文件为：

```text
/data/ymdai/outputs/ember/
pi05_as_writer_v5_2_step900_lora_geometry_529da6b_20260731/analysis.json
```

SHA256：
`9d816baadace851153415a06334efad6f9927bf334f014d5e8ae760be357e1af`。
生成cache manifest payload SHA256为
`5c37574d4d67a10ea7819042942bc7113e769c89e93854a3fafb020b7d1508f5`；
临时generation/analysis driver SHA256分别为
`2e4c5f9ffcd087f13aed963bb8fdb3707023ac5981b6fc294364650a41e0c270`与
`c5253eb7096de71d58fe2f3d058d6742f19383439fb7986fd0ebf3a4679493f4`。
临时cache与drivers在结果封存后删除，只保留`analysis.json`。

主要gauge-invariant结果：

```text
effective LoRA norm mean                         140.441
per-module stable rank mean                       1.01256
entropy effective rank mean                       1.04467
top singular energy mean                         99.0244%
q / v energy                                     73.4476% / 26.5502%
q / v layer-energy CV                             .1609 / .1161
q / v cross-layer effective BA cosine             .9621 / .9817
same-task centered variance / sample energy       1.6655%
same-task pair cosine                              .98438
centered variation: orthogonal / scale-like       89.35% / 10.65%
```

fixed-gauge辅助结果显示，q/v约使用`15.83/15.92 of 16`个能量坐标，最大单坐标
仅占`7.09%/6.84%`；q/v负component pair均为`0%`，B列abs cosine为
`.9828/.9893`。因此v5.2并不存在“16个坐标能量极不均或彼此负向相消”。
effective update接近rank1，是16条建设性同向分量协同形成的结果。

与之对照，Target-Spectral把stable rank强制提高到`3.3245`，correct400却降到
`34`，并摧毁q-dominant、跨层coherent高增益方向。结论是：低effective rank
本身不是病，不能以正交化、均匀奇异值或强制使用16个rank为优化目标。

v5.2仍比v6保留更多视频创新：历史matched panel的same-video BA absolute
delta为`20.188 vs 7.496`，fixed-query action RMS为`.04842 vs .00646`；本次
v5.2 exact50 task-centered方差为`1.6655%`，而v6五视频估计约`.30–.44%`。
估计器不同，不能把比例直接当严格倍数，但方向一致。

## 4. 仍未识别的唯一主要模型变量

v6相对v5.2增加Visual Transition：

```text
D_f = G_f - G_(f-1)
A_f queries D_f
Z_f = A_f + R_f
```

v6在task-complete recipe下达到`143`，v5.2只在旧recipe下达到`132`；v6在旧
recipe下仅`121`。因此现有数据不能判断`143`来自transition还是训练recipe。
下一实验必须先补齐`v5.2 topology × task-complete recipe`，而不是继续发明结构。

判定规则：

- v5.2 task-complete达到或超过v6并保持更强视频创新：选择更简单的v5.2；
- v5.2显著低于v6且差异跨多个tasks：Visual Transition才获得保留资格；
- absolute接近时，优先single-checkpoint特异性更强且拓扑更简单者。

## 5. 下一session的第一实验

当前main已把exact v5.2 topology接到成熟task-complete recipe：

```text
config: configs/pi05_as_writer_language_axial_v5_2_taskcomplete_decay400_v1.json
Writer params: 10,237,704
24 tasks/macro
每task 1 video → 1 LoRA
task内 B20 independent action queries 求均值
24 tasks等权，一次clip/AdamW/scheduler update
4 DDP ranks，真实长度cost balance，rank内long-first
fast decay400，每25 macro checkpoint
```

longest-video B20三macro与fresh0→1→exact-resume1→3已通过。下一session先现场核验
Git、存储和GPU4–7，再从identity fresh运行macro0→200，并默认exact-resume到
400；比较macro150/200/350/400的paired correct400。不得融合checkpoint。

这项正式训练尚未启动，当前没有run root、tmux或训练进程。GPU0–3不得查询或
使用；GPU4–7可按owner授权与他人共卡，但不得干扰他人进程。

## 6. 只有证据触发时才采用的后续改动

若v5.2 task-complete的absolute仍有竞争力，但视频中心化Procedure差异到B写出
端明显被压弱，可测试一个可精确退回v5.2的B-only video innovation residual：

```text
A = A_v5.2
B = B_v5.2 + ΔB_video(procedure_slots)
```

`ΔB_video`应bias-free、final-zero-init，只读现有centered Procedure slots；不做
固定scale、正交、rank分区、diversity loss或第二套adapter。四个B residual heads
按hidden216约增加0.95M参数，总量约11.18M。没有上述压缩证据时不实现它。

任务能力漂移也不能先验归咎于full24 averaging：旧recipe v6、task-complete v6
和Source-SFT都发生过轮换。若漂移继续，先在固定checkpoint按Meta/Core、
Procedure、compiler、factor heads测24-task gradient Gram/cancellation。只有真实
广泛负冲突成立，才考虑一次optimizer update内的single-stage common-descent/
projected full24方向；不使用两阶段训练、task-local optimizer、多checkpoint或
多video平均。若梯度冲突弱而functional loss与closed-loop success错位，则转向
独立short-AS cold-start后的pure-reward RL Writer，而不是继续改AS decoder。

## 7. 明确退役的方向

不得恢复v4静态/absolute-time旁路，v7 joint `8×L` softmax，v8 strict
Action–Effect binding/EventRead，v10高增益Procedure gate，Loom latent gap/
confidence，Recenter DC删除，Core-Program strict bilinear，Prior replacement，
Target-Spectral强制正交，以及checkpoint、video或LoRA平均。它们已有充分负证据。

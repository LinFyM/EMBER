# Dynamic-K Task-Grounded Full-Factor Rank-8 Writer

状态：2026-08-14 terminal non-pass historical authority。本文曾只授权在Dynamic-K Task-Grounded
Visual-Value Writer上，把fixed template-A改为同一task/video Program条件化的dynamic A residual；该单变量
实验已经完成并由本文第10节封存，不再续训或做mapper小修。

## 1. Predecessor adjudication

Task-Grounded Visual-Value的完整K1 strict correct400曲线为`88/86/86/96`，macro200 breadth=`6`，逐task按
Spatial1/3、Object1/3、Goal3/6、Long1/2为`1/0/37/2/0/42/13/1`。top3贡献`92/96`。
macro150到200为`71 retained / 25 gained / 15 lost`、churn40；相对old134为`68/28/66`，相对compiler138为
`73/23/65`。因此晚期净增10仍是集中在少数task的能力换手，不是稳定共同积累。该arm终局non-pass，不resume、
不补五臂controls。

前一arm已经排除以下首因：

- Writer没有读到image/language/video order；
- K-axis set本身不稳定；
- LoRA接近identity或整体能量太小；
- rank-8天然不足以表达历史强effective BA。

CPU诊断给出更早且可隔离的mapper约束：old134逐样本最优rank-8可保留`.99999946` effective-BA能量，当前
fixed random A行空间仅保留`.0195042`；train24 experts对应`.998094/.184501`。同task三视频拟合A、第四视频
leave-one-out仍保留`.9997255`，说明需要的是same-task跨video稳定、但随task变化的A行空间，不是全局静态A、
expert bank或更高rank。

## 2. Single falsifiable hypothesis

当前所有condition只能通过dynamic B在同一个fixed-A右子空间内写policy update。即使Program包含task/video
差异，它也不能为不同task选择不同input-direction row space。若让同一个layer/rank-aligned Program同时生成
`A`和`B`，Writer将能为不同task写出不同的policy-effective右子空间，并有机会恢复v6完整factor生成的绝对性能，
而不放弃昨晚对齐的dynamic-K、memory-token和rank-8结构。

本轮只检验该假设。若失败，不能再用A/B mapper小修挽救当前前端；下一主线应以v6-fast为性能骨架，把本路线已
验证的dynamic-K/memory/set机制作为受控移植变量。

## 3. Unchanged dataflow

```text
exact task language + K=1..4 ordered action-hidden same-task videos
-> 每帧一次真实π0.5 joint image/language/50 Action-probe context
   + 8 Writer memory tokens
-> task-grounded raw visual Value
-> per-video ordered transition D + terminal goal residual G
-> causal temporal encoder
-> permutation-invariant cross-video set aggregation
-> 20 policy groups x 8 rank coordinates M2P
-> shared bias-free 256->1024 projector
-> direct family A/B readouts
-> one complete 38-target rank-8 task-conditioned LoRA
```

每条视频内部保序；K条视频先独立形成Program，再沿video轴做置换不变聚合。frame stride仍为5，总frame budget仍
为64。正确、shuffled和reversed输入必须经过真实帧重排后完整重算joint backbone、D/G、temporal和LoRA。

## 4. Full-factor mapper

对condition `c`、policy group `g`、rank coordinate `r`，现有M2P输出为
`P[c,g,r] in R^256`，共享投影后为`H[c,g,r] in R^1024`。每个shape family `f`只新增一个bias-free A readout：

```text
delta_A[c,g,r,f] = W_A[f](H[c,g,r])
B[c,g,r,f]       = W_B[f](H[c,g,r])
A[c,g,r,f]       = A_template[g,r,f] + delta_A[c,g,r,f]
```

`q/v`使用Action Expert groups 1--18；`action_in`使用group0；`action_out`使用group19。A与B都由同一个
condition Program产生，不新增route、gate、expert、basis、第二套LoRA或额外forward。

四个A readout和四个B readout全部zero-init。于是step0严格为`A=A_template, B=0`，全部38 targets的
`BA=0`，source policy functional identity不变。第一步真实functional gradient先打开B；B非零后，同一个policy
loss自然给A、shared projector和前端提供梯度。这里不人为给A加reconstruction、norm、rank或expert loss。

## 5. Exactly one changed causal variable

| Interface | predecessor | successor |
| --- | --- | --- |
| input/video schedule | dynamic K1--K4 | unchanged |
| joint backbone memory | 8 tokens, one forward | unchanged |
| visual D/G, temporal, set, M2P | task-grounded | unchanged |
| public LoRA rank/targets | rank8, 38 targets | unchanged |
| A ownership | fixed random template | template + condition dynamic residual |
| B ownership | condition dynamic | unchanged |
| AS objective | full24 B20 functional + K consistency | unchanged |
| optimizer/precision | AdamW, BF16/TF32 | unchanged |

不同时恢复v6 hidden/GELU factor heads、rank16、VL Meta-LoRA、text-only forward、expert targets、negative loss、
reward、new set、new objective或scale sweep。

## 6. Why this remains a video-learning architecture

- language只参与joint task query和semantic address，不能直接作为LoRA Value bypass；
- dynamic A和B都只读取经过真实video Value、ordered temporal与set处理后的Program；
- K4 consistency同时约束A/B的共同Program，不平均生成后的LoRA；
- same-task不同video应在滤除nuisance后形成相近task row space，而不同task可形成不同row space；
- 后续只有absolute超过门，才做correct/same/wrong/shuffled/reversed/no-video五臂，验证正确视频沿有用policy方向
  提升，而非仅改变内部表示。

## 7. Mechanism and efficiency gates

最小CPU/单卡验证必须证明：

1. step0 76 tensors逐元素等于template，effective BA严格为零；
2. A/B共八个readout均bias-free、zero-init，source policy保持全冻结；
3. 初始functional BA gradient只打开B，不通过非物理的A直接loss提前打开A；
4. B打开后，A、shared projector、Program与task-grounded visual path均获得finite非零梯度；
5. 两个不同Program产生不同A row spaces和不同effective BA；
6. natural/reversed与video-axis permutation的既有机制合同继续通过；
7. full24一macro仍为K1/K2/K3/K4各6、B20、一次flat gradient reduction、无OOM/nonfinite。

新增A heads约`3.18M`参数，不增加backbone forward或LoRA payload。用live合适A40做真实full24 B20 profile；以
samples/s、最长视频稳定性和峰值显存裁决，不增加逐tensor扫描、hash、重复forward、batch1或dtype扩展。

## 8. Formal training and closed-loop gates

profile通过后fresh训练，不加载96分predecessor。checkpoint every25。K1 strict paired correct400读取
macro50/100；只有出现相对fixed-A的实质改善或可信上升趋势才继续150/200。最晚macro200裁决：

- best至少`125/400`且breadth至少6；
- 相邻checkpoint不能只靠少数task换手维持aggregate；
- 相对fixed-A96、Direct-B102、old134、compiler138、online128和v6-fast143报告逐task与retained/gained/lost；
- 若best低于125，full-factor当前组合终局non-pass，不做rank/scale/LR/seed小扫。

达到`>150/400`后立即补correct/same/wrong/shuffled/reversed/no-video严格配对controls；达到高absolute但视频因果
不成立时仍不能成为最终方法。

## 9. Ownership and lifecycle

- `src/ember/writer/lora_mapper.py`继续唯一拥有Program到完整LoRA的映射；原位替换，不新增parallel mapper；
- trainer、dataset、joint backbone、temporal/set/M2P、evaluator全部复用现有owner；
- canonical config与checkpoint/eval schema切换到Full-Factor；旧fixed-A实现由Git、frozen worktree与formal artifacts
  保存，不保留可执行策略flag；
- registry中旧family只作为历史raw-row分析合同，不是可启动实现。

## 10. Terminal result

formal fresh macro0→50完整结束，K1 strict paired correct400为`91/400`、breadth5，逐task按
Spatial1/3、Object1/3、Goal3/6、Long1/2为`4/1/38/0/0/37/11/0`。相对matched fixed-A macro50的
`88`仅净增3（`70 retained / 21 gained / 18 lost`），低于fixed-A历史最好`96`、Direct-B`102`、old`134`、
compiler`138`和v6-fast`143`，因此未达到本文预注册的125续训门。

机制分析进一步说明它不是简单“还没训够”：Full-Factor与fixed-A的raw A overall cosine=`.73515`、norm ratio=
`1.37621`，但B cosine仅`.24855`、norm ratio仅`.06223`；最终effective BA cosine=`.05853`、norm ratio=
`.24479`。也就是说，dynamic A让同一个offline B20 functional objective找到了一套更大的A和极小的B，得到与
fixed-A近似的训练loss，却把policy update旋到近乎正交且更弱的区域。最早失败接口是functional credit下的
factorization/gauge allocation，而不是rank8理论容量、视频没有被读取或吞吐不够。

正式裁决：不resume到100，不扫rank/scale/LR/seed，不再修当前mapper。下一受控实验以v6-fast原生性能路径为
baseline，只增加动态K的跨视频Program集合桥接；memory token和rank8仍是已认真检验过、可在强底座稳定后逐项
重引入的方法变量，不是被该负结果永久否定的原则。

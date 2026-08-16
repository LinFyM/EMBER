# V6-LPCP Native-Zero Residual Bank Commitment

状态：2026-08-16 active single-variable design authority，简称`NZRB-C`。本轮从sealed LPCP fresh启动，完整保留
LPCP视频carrier、MB-SOP matched successful-occupancy credit、四个disjoint correct K4 views、PAV native acceptance
与ALB-NV的fixed-A/B-only reward参数化。唯一变量是：不再把`delta-B`加到非零`B0`上，而是在同一套public LoRA中
追加一个从native zero开始的rank16 residual bank。

## 1. ALB-NV留下的最早断点

ALB-NV删除A residual与双边cross term后并非全面失败：它把PAV的task15/18 exact no-op分别变成j5与j0，task18
得到held8 BA `.774/.785`、held/train `1.030x`并全门通过。这证明：

```text
joint video-language Value -> fixed-A delta-B A0
```

可以形成跨视频、跨condition的policy-facing方向。剩余失败更窄：

- task9的four-view continuous gradients cosine/energy=`.415/.559`且共同下降覆盖4/4，但同一Adam ray的11个
  native points没有一个四路同时下降；
- task15只有j5可接受，continuous direct rows仍`.948/.938`，写入held native B后raw coherence降到
  `.101/.323`且只有5/8 held tasks过门；
- ALB三anchor合计wall仅为PAV的`.752x`，所以不是吞吐或训练图问题。

ALB的public B实际是：

```text
B_native = cast_native(B0 + delta-B)
```

q/v的`B0`为非零BF16。小`delta-B`必须先跨过每个已有元素的局部ULP；不同video/task跨过的元素集合可不同。
因此连续共同方向可能在native state中变成空集或video-local稀疏变化。本轮只检验这一解释。

## 2. 唯一输出变量

LPCP仍生成原rank16 `{A0(c), B0(c)}`，ALB同一四个heads仍生成`delta-B(c)`。public LoRA改为一个完整rank32
state：

```text
A_public(c) = concat_rows(A0(c), A0(c))
B_public(c) = concat_columns(B0(c), cast_native(delta-B(c)))
```

PEFT contract保持`alpha=rank=32`，所以scale仍精确为1。effective adaptation为：

```text
B_public A_public = B0 A0 + delta-B A0
```

这与ALB的连续新增BA完全相同，但`delta-B`不再与非零`B0`做逐元素相加。第二bank的B从精确zero开始，BF16在
zero附近没有`B0`局部ULP门槛。第一bank逐元素保持LPCP，第二bank不压缩、不重分解、不替换任何baseline support。

这不是两套LoRA：每个condition只输出一套38-target rank32 public state，policy只挂载一个adapter，两个rank banks
在同一次LoRA matmul中共同形成一个BA。不能分别部署、选择、平均或关闭其中一个bank。rank32是本轮为保持完整
rank16 carrier并对ALB residual做exact native-zero反事实所需的最小无压缩表示；本轮不声称rank越大越好。

## 3. Writer输入、Program与trainable保持不变

输入仍为exact task language与四条同task action-hidden correct videos，stride5，每条video内部有序，K轴置换不变。
完整路径保持：

```text
language + ordered videos
 -> one joint context forward
 -> 18-layer Action probes
 -> causal per-video Procedure
 -> K-set aggregation
 -> video-required joint Value X
 -> four shared zero-init B heads
 -> native-zero residual rank bank
 -> one complete task LoRA
```

`X=M_video*RMSNorm(L_language)/sqrt(256)`不变。no-video或constant dynamics使`M_video`近零，不能由language单独写
residual。trainable仍只有q/v/action-in/action-out四个B heads，共860,160参数；新增A bank只是逐condition复制A0，
没有新parameter、route、expert、memory、loss或第二次backbone forward。

## 4. 为什么这不是旧rank reservation

历史uniform rank14与rank14+reserved方案先把已有rank16 support压缩到更小carrier，再用空lane；compiler-only已证明
compression本身会丢old support。NZRB-C不压缩、不SVD、不重新生成LPCP：原16 lanes逐元素原样保留，另加完整16
lanes只承载与ALB数学相同的residual。因而本轮只改变residual的native origin。

它也不同于早期fresh rank8 Direct-Family-B：后者固定随机A并从弱Writer训练；本轮第二bank的A逐condition复制强
LPCP A0，第一bank完整保留LPCP143。

## 5. Step0、reference与数值合同

第二bank的B在step0、no-video与commitment rejection时精确为zero，因此：

```text
candidate step0 BA = LPCP BA
reference BA = AS139 BA
```

以上以FP64 effective BA逐target验证；不要求rank16与rank32 GEMM逐元素bitwise action一致，也不为此关闭高效kernel、
固定batch1或扩dtype。rank32 contract的`alpha/rank=1`必须与rank16一致。q/v仍BF16，action factors仍原生FP32。

public state从1,287,168增至2,574,336个factor参数，但Writer新增学习输出仍只有原`delta-B`，没有生成第二份A/B。
正式profile以真实cycle wall、峰值显存和LoRA/action吞吐裁决；不以最低显存为目标。

## 6. 完整保留的训练合同

- sealed LPCP cold start、AS139 reference、K4、stride5、同task跨episode错开；
- 两个paired states、四rollouts、8-strata matched successful occupancy、Nmc4；
- 四个互斥correct K4 credit views、view/task等权、同一AdamW状态；
- actual Adam candidate的`j=0..10` first-all-view native acceptance，失败exact no-op；
- train24-only reward、source policy frozen、信息墙、one-shot Writer execution与single checkpoint；
- 不改video carrier、memory、query、gradient ray、LR、scale、seed、dtype或matched panel。

## 7. 固定三anchor快速否决

仍只运行task9/15/18，三项必须全部满足：

1. outcomes=`2/1,2/0,1/2`、complete=`26/65/44`、selected=`8/16/8`、0禁读；
2. rank32 first bank逐tensor等于LPCP rank16，second-B step0精确zero，FP64 base effective BA相对LPCP误差
   不超过`1e-12`；所有public A与first-bank B在训练前后不变；
3. 只有四个B heads trainable且全部finite/nonzero；三任务都在`j<=10`找到first all-view native candidate；
4. accepted residual的FP64新增BA与同一`delta-B A0`反事实relative-L2不超过`1e-6`，q/v/action与fixed-action
   response均非零；
5. train four-view BA cosine/energy至少`.40/.55`；
6. validation8 aggregate至少`.30/.48`、至少6/8过`.15/.40`、raw residual-B至少`.30`、action至少`.15`；
7. held/train effective-BA L2至少`.30x`；task9不得no-op，task15不得再只有5/8/raw-B失败；
8. natural/reversed relative-L2至少`.50`、constant/natural不超过`.005`，0 OOM/nonfinite；
9. 三anchor合计cycle wall不超过ALB的`1.15x`；记录rank32 policy峰值显存和fixed-action throughput。

任一失败即终局，不减少/增加reserved lanes、不改rank32 scale、不补A side、不混family side、不扫optimizer/ray。
三项全过后才实现distributed/evaluation正式接口并启动full24。

## 8. Full24与closed-loop裁决

若三anchor全过，先完成rank32 formal injection、cache与evaluator合同，再fresh full24 cycle1，随后立即K4 strict
paired400。continuation门与ALB保持：correct至少142、breadth至少7、相对LPCP lost不超过15且gained不少于lost。
稳定资格仍需相邻两个single checkpoints均至少142、均值至少145、churn不超过20、Jaccard至少`.85`、final lost
不超过10。首次约145且retention过门立即补same-task-other、wrong、shuffled、reversed、no-video。

rank32若closed-loop更强，后续是否压缩为rank8/16是独立效率实验，不能在本轮提前牺牲真实性能或混入主变量。

## 9. 快速否决解释

- task9仍无native candidate：非零B0局部ULP不是主要断点，下一步应转向reward functional metric/Value方向，不再
  改factor native origin；
- task9接受但task15 raw/held仍失败：zero origin解决visibility但未解决跨condition Value幅度或task覆盖；
- 三anchor过而full24换手：native写出成立，最早缺口后移到shared multi-task coexistence；
- correct高而视频controls失败：rank bank只放大了shortcut，不构成有效视频教学。

负结果只淘汰`LPCP rank16 carrier + duplicated-A native-zero rank16 residual bank + ALB four-head Value + one reward
cycle`。不否定memory token、rank8、few-shot、生成LoRA或未来task-local RL。

## 10. Canonical实现状态

canonical active config为
`configs/pi05_writer_v6_lpcp_native_zero_residual_bank_commitment_v1.json`，旧ALB executable config已从active tree
移除。policy由sealed PI05 contract只派生rank32/alpha32，foundation、38 targets、identity seed与其余authority不变；
rank32 identity的前16 lanes逐元素等于rank16 identity。Native V6 Writer仍只解码原320 slots/rank16 carrier，public
assembly在同一次decode中复制A0并把四head B rows放入第二bank，不增加encoder/compiler forward。

fresh reward config/checkpoint/completion/evaluator identity已切换到NZRB-C；historical rank16 base Writer仍通过显式
deployment rank使用同一canonical builder。定向CPU=`50 passed`，完整CPU在`.env.local` LIBERO assets合同下=
`405 passed`，compileall与diff check通过；architecture guard无hard violation，active source diff净增长165行，
无新module/entrypoint。以上只证明rank32 factor、fresh边界与CPU机制正确，尚无GPU anchor结果。

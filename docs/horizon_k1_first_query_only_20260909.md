# 首层语言内容对照：首段真实行为证据

2026-09-09。注册见[active design §8.2.5](horizon_relation_video_writer_design.md#825-单变量fresh对照保留首次语言检索移除直接语言内容残差)。这是fresh学习后的对照，区别于此前A/A2/A3/B冻结诊断。

## 结论与边界

新100/200 correct为75/110，原同节点55/110。100步有20个净成功增益，但200步总分与breadth均未提高；新训练任务held-video为46/96，低于原52/96。当前不能称整体修复，也不能把首层裸language内容路径认定为唯一泛化根因。

新100→200仍有35个净成功增长，四suite均有净增；但breadth一直6/8，global1、23保持0，相邻churn75/400、J=.4231，远未通过>145及稳定性资格。训练任务虽弱于原200，仍从source15/96增加到46/96，表明存在真实学习获取，尚非无能力或已充分饱和的证明。

这两个节点还没有覆盖原模型110→86/87的300/400回落区间。因此继续**同一改动、同一学习状态**至固定300/400，专门检验后期能力保持，而不是为了内部loss好看无界续训。后续结果可以否定本次内容移除假设；不混入rank-sharing、24-task组织、LR/rank/seed扫描或RL。

## 对照与信息墙

唯一模型变化：第一compiler块由`F(e+l+Cross(LN(e+l),P))`改为`F(e+Cross(LN(e+l),P))`；首cross的language query保留，第二块沿实际新内容运行。所有其它视频前端、完整50H、Meta、38-target rank16独立native D、4task×64queries及pure FM保持。P4仍能包含语言引导的静态语义；本实验不证明视频动态必要性。

整个Writer/Meta/AdamW/scheduler/sampler/RNG fresh seed7，合法LoRA identity，source可训练参数0。100/200完整checkpoint独立保留。实际前800个训练条件的task、video、action episode/frame、query/policy seed和权重与原基线逐条一致，共51200queries；每个新validation bank都是400个新LoRA条件，50个视频每task各一次。所有评测完整完成，task/state/language/env与policy-noise prefix、normalization及source合同配对通过。

计算采用正常BF16/TF32和不同物理分块/worker topology；正常数值分叉保留，不重跑挑分，也不将每一条状态换手唯一归因模型改动。未做same-task-other或最终causal controls，未使用validation/test梯度。

## Validation完整结果

每格分母50，本表仅列首段匹配节点。原300/400的86/87另作为下一段固定参照。

| 任务 | 原100 | 新100 | 原200 | 新200 |
|---|---:|---:|---:|---:|
| 1 ramekin旁黑碗→盘 | 0 | 0 | 0 | 0 |
| 3 cookie box上黑碗→盘 | 0 | 1 | 1 | 3 |
| 11 cream cheese→篮 | 21 | 33 | 37 | 31 |
| 13 BBQ sauce→篮 | 1 | 3 | 24 | 28 |
| 23 开抽屉并放碗 | 2 | 0 | 0 | 0 |
| 26 cream cheese→碗 | 25 | 31 | 41 | 38 |
| 31 cream cheese和butter→篮 | 4 | 3 | 4 | 9 |
| 32 开炉并放moka pot | 2 | 4 | 3 | 1 |
| 合计 /400 | 55 | 75 | 110 | 110 |
| breadth /8 | 6 | 6 | 6 | 6 |

新100→200的S/O/G/L为1/36/31/7→3/59/38/10。原200为1/61/41/7；新200的Long和Spatial小幅增加被Object、Goal损失抵消。

| 完整配对比较 | 保留 | 新增 | 丢失 | churn | 成功集Jaccard |
|---|---:|---:|---:|---:|---:|
| 原100→新100 | 44 | 31 | 11 | 42/400 | .5116 |
| 原200→新200 | 86 | 24 | 24 | 48/400 | .6418 |
| 新100→新200 | 55 | 55 | 20 | 75/400 | .4231 |
| source47→新100 | 33 | 42 | 14 | 56/400 | .3708 |
| source47→新200 | 39 | 71 | 8 | 79/400 | .3305 |

前两行是跨模型比较，不把其churn误作新模型相邻资格；第三行才是本轮相邻证据。

## 固定held-video训练任务诊断

全train24，每task states32–35与teacher46–49各一次，4条/任务，共96。不能将不同难度的train与validation成功率直接相减命名为因果泛化损失。

| Suite | 原200 | 新200 |
|---|---:|---:|
| Spatial | 12/24 | 11/24 |
| Object | 15/24 | 12/24 |
| Goal | 16/24 | 14/24 |
| Long | 9/24 | 9/24 |
| 合计 | 52/96 | 46/96 |
| breadth | 20/24 | 18/24 |

原200→新200保留39、新增7、丢失13，churn20/96、J=.6610。训练侧Object减少3、Goal减少2、Spatial减少1，Long持平。相对固定source15/96，新200保留12、新增34、丢失3；该source子集来自原完整120行中的预登记32–35状态，无新source评测或结果筛选。

同一3072个held-action queries与真实teacher/query/noise配对：新200 FM .111183766，原200 .111352643，只有10/24任务新loss更低。均值接近不代表逐任务或行为等价，更不能替代上述46/96对52/96。此前A1的16query/task冻结面板口径不同，其.115832不能混用为这里的原200参照。

## 执行与可复核原件

科学实现与训练/物化为`fea45593`，profile仅验证运行且不继承；正式200更新墙钟2921.40秒、完整训练3290.54秒。100bank400条完整1353.86秒（包含另一节点冷加载）；200的独立400+96 banks共1004.68秒。100/200评测分别使用2/4个worker，完整wrapper墙钟保存在原件中，不将不同资源拓扑的墙钟差归因模型速度。

200及train96评测使用`f478076f`：只去掉旧“已用显存≤8GiB”准入条件，保留free≥32GiB/util≤10、GPU/process记录和全部推理合同；31项launcher/queue定向检查通过，实际双worker运行仍有约13GiB余量。100评测继续其原冻结代码。此操作不改变模型、数据、LoRA或RNG。

- 原始新run：`runs/outputs/horizon_k1_first_query_only_v1_seed7_20260909/`。
- 聚合与全部比较索引：`runs/analysis/horizon_relation_writer_20260908/k1_first_query_only/first_segment_evidence.json`。
- 同目录`step100/`、`step200/`、`train96_step200/`保留完整bank检查、launcher/return codes、实际rows比较和wall time。
- `paired_held_fm_comparison.json`保留全24task逐任务3072query结果；`launch_contract.json`和`evaluation_admission_correction.json`保留精确执行与资源依据。

下一段依[active design §8.2.6](horizon_relation_video_writer_design.md#826-首段后固定300400保持检验)执行，资格门槛和最终视频controls保持。

已复核原200→300/400的逐任务保持参照：BBQ sauce为24→1/3，cream cheese→篮为37→36/37、→碗为41→45/42；breadth6→5/4。原200→300保留71/新增15/丢39，→400保留67/新增20/丢43。因此后续除总分外须保留逐任务配对，区分旧能力保持、弱任务扩展与其它任务补偿；不由单个任务替代既定整体资格。原始比较在A/`segment200_400/baseline_retention_reference.json`及两份完整R/G/L JSON。
